# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# IBGDA ordering litmus driver. One process per node (srun -N2 -n2), GPU 0 on each node:
# rank 0 = sender, rank 1 = receiver. Channel setup copies the mscclpp IBGDA benchmark
# comms/mscclpp/bench/bench_ibgda.py: build_p2p_channels(..., ibgda=True), one IBGDA QP per
# PortChannel. The kernel (ibgda_litmus.cu) is compiled here with nvcc against the same include
# dirs KernelBuilder(..., ibgda=True) uses, and the cubin is kept for SASS inspection.
#
#   python driver.py --plans p0,p1,... --iters 100000 --out results.jsonl [--work DIR]
import argparse
import ctypes  # noqa: F401
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

M_P1, M_P2, M_P3, M_P4, M_P5, M_P6A, M_P6B = 1, 2, 3, 4, 5, 6, 7
V_PF_NONE, V_BATCH, V_FLAG4 = 0x10, 0x20, 0x40
BUF_BYTES = 16 << 20
DST_OFF, CTRL_OFF = 4 << 20, 8 << 20
KiB, MiB = 1024, 1 << 20
RES_N = 64
R = dict(
    status=0,
    iters=1,
    bad_iters=2,
    stale=3,
    future=4,
    new=5,
    other=6,
    first_it=7,
    first_w=8,
    first_v=9,
    cq_status=10,
    ns=11,
    zero=12,
    p7_word=13,
    p7_gpu=14,
    p7_nic=15,
    synd=16,
)
STATUS = {
    0: "ok",
    1: "token/flag timeout",
    2: "ack timeout",
    3: "error CQE",
    4: "CQ poll timeout",
    5: "unsupported nic_handler",
}
RECV = {
    0: "acq.sys token, relaxed.sys data",
    1: "relaxed.sys token, no fence, weak data",
    2: "acq.sys token, weak data",
    3: "relaxed.sys token + fence.acquire.sys, relaxed.sys data",
}
DBF = {
    0: "none",
    1: "fence.release.cta",
    2: "fence.release.gpu",
    3: "fence.release.sys",
    4: "fence.sc.sys",
}


def V_RECV(r):
    return r << 8


def cfgs_for(plan, iters):
    out = []

    def add(probe, mode, variant, size, n=iters, desc=""):
        out.append(
            dict(probe=probe, mode=mode, variant=variant, size=size, iters=n, desc=desc)
        )

    if plan == "p0":  # endianness of NIC fetch-add, few iterations of P1 8 B
        add("P0", M_P1, 0, 8, 8, "raw token / sink bytes after k fetch-adds")
    elif plan == "p1":
        for size in (8, 4 * KiB, MiB):
            for rv in (0, 3, 2, 1):
                add(
                    "P1",
                    M_P1,
                    V_RECV(rv),
                    size,
                    desc=f"DOCA put+signal same QP; recv: {RECV[rv]}",
                )
            add(
                "P1",
                M_P1,
                V_BATCH,
                size,
                desc="DOCA put(skip doorbell)+signal same QP, one doorbell; recv: "
                + RECV[0],
            )
    elif plan == "p3":
        # control first (DOCA path), then custom posts from strongest to weakest fence
        for size in (8, MiB):
            add(
                "P3",
                M_P1,
                V_PF_NONE,
                size,
                desc="DOCA put+signal, payload: bar.sync only (DOCA's internal fence.release.gpu)",
            )
            for pf in (0, V_PF_NONE):
                for k in (4, 3, 2, 1, 0):
                    pfd = "bar.sync only" if pf else "bar.sync+fence.sc.sys"
                    add(
                        "P3",
                        M_P3,
                        pf | k,
                        size,
                        desc=f"custom write+FA, 1 doorbell; WQE->DB fence: {DBF[k]}; payload: {pfd}",
                    )
    elif plan.startswith("p3:"):  # single P3 config: p3:<pf 0|1>:<k>:<size>
        _, pf, k, size = plan.split(":")
        pf, k, size = int(pf), int(k), int(size)
        pfd = "bar.sync only" if pf else "bar.sync+fence.sc.sys"
        add(
            "P3",
            M_P3,
            (V_PF_NONE if pf else 0) | k,
            size,
            desc=f"custom write+FA, 1 doorbell; WQE->DB fence: {DBF[k]}; payload: {pfd}",
        )
    elif plan == "p5":
        for size in (8, 4 * KiB, MiB):
            add(
                "P5",
                M_P5,
                0,
                size,
                desc="put QP A, poll A's CQE, signal QP B; recv: " + RECV[0],
            )
    elif plan == "p6":
        for size in (4 * KiB, MiB):
            add(
                "P6a",
                M_P6A,
                0,
                size,
                desc="put, overwrite source (no CQ poll), signal same QP",
            )
            add(
                "P6b",
                M_P6B,
                0,
                size,
                desc="put, poll CQE, overwrite source, signal same QP",
            )
    elif plan == "p2":
        for size in (8, 4 * KiB, MiB):
            add(
                "P2",
                M_P2,
                0,
                size,
                desc="put QP A, signal QP B, no flush; recv: " + RECV[0],
            )
    elif plan == "p4":
        for size in (4 * KiB, MiB):
            add(
                "P4",
                M_P4,
                0,
                size,
                desc="put large, put 8 B flag, same QP, 2 doorbells; recv polls flag (acq.sys)",
            )
            add(
                "P4",
                M_P4,
                V_BATCH,
                size,
                desc="put large (skip doorbell), put 8 B flag, 1 doorbell",
            )
    elif plan == "p4b":
        for size in (4 * KiB, MiB):
            add(
                "P4",
                M_P4,
                V_FLAG4,
                size,
                desc="put large, put 4 B flag, same QP, 2 doorbells; recv polls flag (acq.sys)",
            )
            add(
                "P4",
                M_P4,
                V_FLAG4 | V_BATCH,
                size,
                desc="put large (skip doorbell), put 4 B flag, 1 doorbell",
            )
    elif plan == "p7":
        out.append(
            dict(
                probe="P7",
                mode=-7,
                nfa=iters,
                gpu_on=1,
                desc="NIC FA + GPU red.add.sys same word",
            )
        )
        out.append(
            dict(probe="P7", mode=-7, nfa=iters, gpu_on=0, desc="NIC FA only (control)")
        )
    else:
        raise SystemExit(f"unknown plan {plan}")
    return out


def compile_cubin(work):
    from comms.mscclpp.utils import doca_include_dir, kernel_include_dir

    nvcc = os.environ.get("NVCC", "nvcc")
    cubin = os.path.join(work, "ibgda_litmus.cubin")
    inc = kernel_include_dir()
    doca = doca_include_dir()
    cmd = [
        nvcc,
        "-std=c++17",
        "-cubin",
        "--expt-relaxed-constexpr",
        "-gencode",
        "arch=compute_103a,code=sm_103a",
        f"-I{inc}",
        f"-I{doca}",
        os.path.join(HERE, "ibgda_litmus.cu"),
        "-o",
        cubin,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"nvcc failed: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return cubin, dict(nvcc_cmd=" ".join(cmd), mscclpp_include=inc, doca_include=doca)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default="p0,p1")
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--timeout_s", type=float, default=3.0)
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--work", default=os.environ.get("NIC_WORK", "/tmp/ptxv-nic"))
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.environ["MSL"], "comms", "mscclpp", "bench"))
    import cuda_env  # noqa: F401  (fix CUDA_PATH before cupy)
    import cupy as cp
    import numpy as np
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    dist.init_process_group(backend="gloo", rank=rank, world_size=2)
    cp.cuda.Device(0).use()
    from comms.mscclpp import CommGroup, EndpointConfig, GpuBuffer, IbgdaService
    from comms.mscclpp._mscclpp import CppTransportFlags

    work = os.path.join(args.work, f"r{rank}")
    os.makedirs(work, exist_ok=True)
    cubin, build = compile_cubin(work)

    group = CommGroup(torch_group=dist.group.WORLD)
    peer = 1 - rank
    tran = group.my_ib_device(0)
    buf = GpuBuffer(BUF_BYTES, dtype=cp.uint8)
    buf[:] = 0
    cp.cuda.runtime.deviceSynchronize()

    service = IbgdaService()
    comm = group.communicator
    ep = EndpointConfig(tran)
    ep.ibgda = True
    nqp = 2
    conns = [comm.connect(ep, peer, c).get() for c in range(nqp)]
    tf = CppTransportFlags()
    tf |= tran
    reg_local = comm.register_memory(buf.data.ptr, buf.nbytes, tf)
    comm.send_memory(reg_local, peer, 1 << 20)
    reg_remote = comm.recv_memory(peer, 1 << 20).get()
    dst_id = service.add_memory(reg_remote)
    src_id = service.add_memory(reg_local)
    chans = []
    for c in range(nqp):
        sem = comm.build_semaphore(conns[c], peer, c).get()
        chans.append(service.port_channel(service.add_semaphore(sem), dst_id, src_id))
    raw = b"".join(ch.device_handle().raw for ch in chans)
    d_handles = cp.asarray(memoryview(raw), dtype=cp.uint8)
    group.barrier()

    mod = cp.RawModule(path=cubin)
    k_send = mod.get_function("probe_sender")
    k_recv = mod.get_function("probe_receiver")
    k_p7s = mod.get_function("p7_sender")
    k_p7r = mod.get_function("p7_receiver")
    k_info = mod.get_function("qp_info")

    info = cp.zeros(16 * nqp, dtype=cp.uint64)
    k_info((1,), (32,), (d_handles, np.int32(nqp), info))
    cp.cuda.runtime.deviceSynchronize()
    qinfo = []
    for c in range(nqp):
        o = [int(x) for x in info[16 * c : 16 * c + 16].get()]
        d = dict(
            nic_handler=o[0],
            qp_mem_type=o[1],
            sq_wqe_num=o[2],
            sq_wqe_daddr=hex(o[3]),
            sq_dbrec=hex(o[4]),
            sq_db=hex(o[5]),
            cqe_daddr=hex(o[6]),
            cqe_num=o[7],
            cq_mem_type=o[8],
            cq_dbrec=hex(o[9]),
            sq_num=o[10],
            sq_rsvd_index=o[11],
            remote_base=hex(o[12]),
            local_base=hex(o[13]),
            sink=hex(o[14]),
        )
        for key in (
            "sq_wqe_daddr",
            "sq_dbrec",
            "sq_db",
            "cqe_daddr",
            "cq_dbrec",
            "sink",
        ):
            try:
                a = cp.cuda.runtime.pointerGetAttributes(int(d[key], 16))
                d[key + "_memtype"] = {
                    0: "unregistered",
                    1: "host",
                    2: "device",
                    3: "managed",
                }.get(a.type, a.type)
            except Exception as e:  # noqa: BLE001
                d[key + "_memtype"] = f"err {e}"
        qinfo.append(d)
    hdr = dict(
        rank=rank,
        host=socket.gethostname(),
        tran=str(tran),
        build=build,
        handle_bytes=len(raw) // nqp,
        qp_info=qinfo,
        env={
            k: os.environ.get(k)
            for k in (
                "MSCCLPP_IB_OOO_DP",
                "MSCCLPP_HCA_DEVICES",
                "MSCCLPP_IB_GID_INDEX",
            )
        },
    )
    hdrs = [None, None]
    dist.all_gather_object(hdrs, hdr)
    fout = open(args.out, "a") if rank == 0 else None
    if rank == 0:
        fout.write(
            json.dumps(
                dict(
                    kind="header",
                    date=time.strftime("%Y-%m-%d %H:%M:%S"),
                    job=os.environ.get("SLURM_JOB_ID"),
                    ranks=hdrs,
                )
            )
            + "\n"
        )
        fout.flush()
        print(json.dumps(hdrs, indent=1), flush=True)

    timeout = np.uint64(int(args.timeout_s * 1e9))
    be_nic = None  # measured by p0
    cfg_id = 0
    for plan in args.plans.split(","):
        for cfg in cfgs_for(plan, args.iters):
            cfg_id += 1
            buf[DST_OFF:BUF_BYTES] = 0
            res = cp.zeros(RES_N, dtype=cp.uint64)
            cp.cuda.runtime.deviceSynchronize()
            dist.barrier()
            t0 = time.time()
            if cfg["mode"] == -7:
                if rank == 0:
                    k_p7s(
                        (1,),
                        (32,),
                        (
                            d_handles,
                            buf,
                            res,
                            np.int32(cfg["nfa"]),
                            np.int32(32),
                            timeout,
                        ),
                    )
                else:
                    addend = 1 if (be_nic is None or be_nic) else (1 << 32)
                    cfg["gpu_addend"] = addend
                    k_p7r(
                        (1,),
                        (1024,),
                        (
                            buf,
                            res,
                            np.int32(cfg["gpu_on"]),
                            np.uint64(1 << 14),
                            np.uint64(addend),
                            timeout,
                        ),
                    )
            else:
                kern = k_send if rank == 0 else k_recv
                kern(
                    (1,),
                    (1024,),
                    (
                        d_handles,
                        buf,
                        res,
                        np.int32(cfg["mode"]),
                        np.int32(cfg["variant"]),
                        np.int32(cfg_id % 200 + 1),
                        np.int32(cfg["iters"]),
                        np.uint64(cfg["size"]),
                        timeout,
                    ),
                )
            cp.cuda.runtime.deviceSynchronize()
            wall = time.time() - t0
            r = [int(x) for x in res.get()]
            allr = [None, None]
            dist.all_gather_object(allr, dict(res=r, cfg=cfg, wall=wall))
            if rank == 0:
                s, rv = allr[0]["res"], allr[1]["res"]
                cfg = allr[1]["cfg"]
                rec = dict(
                    kind="result",
                    cfg_id=cfg_id,
                    **cfg,
                    wall_s=round(wall, 3),
                    sender_status=STATUS.get(s[0], s[0]),
                    receiver_status=STATUS.get(rv[0], rv[0]),
                    sender_iters=s[1],
                    receiver_iters=rv[1],
                    sender_cq_status=s[R["cq_status"]] - (1 << 64)
                    if s[R["cq_status"]] >> 63
                    else s[R["cq_status"]],
                    sender_synd=hex(s[R["synd"]]),
                    receiver_synd=hex(rv[R["synd"]]),
                )
                if cfg["mode"] == -7:
                    w = rv[R["p7_word"]]
                    rec.update(
                        word=hex(w),
                        nic_fa=s[R["p7_nic"]],
                        gpu_adds=rv[R["p7_gpu"]],
                        word_le=w,
                        word_be=int.from_bytes(w.to_bytes(8, "little"), "big"),
                    )
                else:
                    rec.update(
                        {
                            k: rv[R[k]]
                            for k in (
                                "bad_iters",
                                "stale",
                                "future",
                                "new",
                                "other",
                                "zero",
                                "first_it",
                                "first_w",
                            )
                        },
                        first_v=hex(rv[R["first_v"]]),
                        iters_with_new=rv[32],
                        ns_per_iter=round(rv[R["ns"]] / max(1, rv[1]), 1),
                        token_raw=[hex(x) for x in rv[33:37]],
                        sink_raw=[hex(x) for x in s[33:37]],
                        ack_raw=[hex(x) for x in s[41:45]],
                    )
                    if cfg["probe"] == "P0":
                        tok = rv[33:36]
                        le = tok == [1, 2, 3]
                        be = tok == [1 << 56, 2 << 56, 3 << 56]
                        rec["nic_fa_byte_order"] = (
                            "little" if le else ("big" if be else "unknown")
                        )
                print(json.dumps(rec), flush=True)
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                if cfg["probe"] == "P0":
                    be_nic = rec["nic_fa_byte_order"] == "big"
            be_box = [be_nic]
            dist.broadcast_object_list(be_box, src=0)
            be_nic = be_box[0]
            st = [allr[0]["res"][0], allr[1]["res"][0]]
            if any(st):
                if rank == 0:
                    print(
                        f"[driver] stopping after cfg {cfg_id}: statuses {st}",
                        flush=True,
                    )
                group_ok = False
                break
        else:
            continue
        break
    # Keep the QPs alive until both sides are done.
    dist.barrier()
    if fout:
        fout.close()
    os._exit(0)


if __name__ == "__main__":
    main()

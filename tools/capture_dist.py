# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-process capture: record a `torchrun --nproc-per-node N` run (one process per GPU) and
# merge every rank's kernels into ONE multi-GPU ptxv case: per-device buffers and launches,
# pointers into peers' memory resolved to the peer's buffer, `replicas` for the outputs that
# must agree across ranks, and every rank's GPU output bytes.
#
#   CUDA_VISIBLE_DEVICES=0,1,3 $PTXV_PYTHON tools/capture_dist.py --nproc 3 \
#       --file tools/capture_examples/symm_oneshot.py --out fixtures/r4a_oneshot
#
# The file defines `setup(rank, world, group_name) -> {name: value}` (tensors, symmetric-memory
# handles, anything) and `run(**inputs)` returning the output tensor(s) (a tensor, tuple or
# dict). Tensors run() changes in place are outputs too.
#
# How (the multi-process version of tools/capture.py, docs/rung4.md):
# * The script runs torchrun with itself as the worker, under probes/launchcap (launches with
#   stream, device and host time; memcpy/memset; with LAUNCHCAP_VMM=1 every cuMemCreate /
#   export / import / reserve / map / multicast / IPC call and every event record / wait) and
#   probes/launchcap/cubincap. "%r" in their output paths is the rank: one directory per rank.
# * Each rank runs setup(), warms up, synchronizes with the others (barrier), snapshots its
#   memory (caching-allocator blocks, its own VMM allocations, the named tensors' bytes), runs
#   run() once more between two barriers, and records its launches and outputs.
# * Peer memory. torch's symmetric memory allocates each rank's buffer with cuMemCreate,
#   exports it as a FABRIC handle (64 bytes, identical in exporter and importer), and every
#   peer imports and maps it at its own virtual address. So a physical allocation is
#   identified by its fabric handle (or (rank, handle) when never exported), its owner is the
#   rank that created it, and a pointer v in rank r resolves to (owner, owner's address) through
#   r's mappings. POSIX-fd exports are matched by the fd's (st_dev, st_ino), legacy IPC by the
#   64-byte cudaIpcMemHandle.
# * Multicast (NVLS): cuMulticastCreate / AddDevice / BindMem / BindAddr and cuMemMap of the
#   multicast handle are recorded. A pointer into a multicast mapping (a kernel parameter or a
#   table word) becomes a reference to a case `[[multicast]]` object, whose `binds` say which
#   buffer of which GPU backs which range of it (the bound allocation cut at the case's
#   buffers; S-MULTIMEM).
# * Pointer tables. torch passes peer pointers through device arrays (`T** buffer_ptrs_dev`,
#   `uint32_t** signal_pads_dev`), so every buffer a parameter references is scanned for 8-byte
#   words that resolve; those become `ptrs` of the buffer (crates/case `Buffer::ptrs`), patched
#   with the right buffer's address in the simulator and on hardware.
# * Buffers. A rank's caching-allocator blocks become buffers on that device (`_r<rank>`
#   suffix). A symmetric allocation is split at the layout its handle reports (data buffer,
#   signal pad) into `<name>_r<owner>` and `sig_r<owner>`; other VMM allocations are kept whole.
# * Launch order: each rank's launches as `[[launches]]` with `device = rank` (only
#   same-device launches are ordered, E-PDL), interleaved by host submission time (kept in
#   capture.json). Within a rank only the order its streams and stream order calls (event
#   records / waits, graph launches) enforce is kept (`after`, `waits`, `pdl`,
#   launch_order = "recorded"; python/ptxv/_order.py, docs/capture.md "Launch order").
# * Replicas: the outputs the module lists in REPLICAS (an all-reduce result) form a replica
#   group per name across the ranks; their bytes must be identical (N4, `divergent`).
# * Several streams per rank (DeepEP's communication stream): a stream switch that an event
#   orders becomes an edge; one that nothing orders leaves the two launches concurrent (noted).
# * cudaMalloc buffers, legacy IPC (cudaIpcGetMemHandle / OpenMemHandle) and mapped host
#   memory are tracked like VMM allocations; host-to-device uploads between kernels (a
#   host-computed routing table) are folded into the destination's initial bytes (noted).
# * GPU-initiated RDMA (IBGDA, docs/nic.md): when the file defines `nic(inputs)` returning
#   [{"peer": rank, "addr": device address of an mscclpp PortChannelIbgdaDeviceHandle,
#   "size": its byte size}], each handle's DOCA QP (send ring, CQ, doorbell record, read from
#   the doca_gpu_dev_verbs_qp struct the handle points to) becomes a `[[nic_qps]]` entry, the
#   doorbell register a synthetic 8-byte buffer, the handle's remote addresses pointers into
#   the peer's buffers, and its keys `[[nic_mrs]]` covering the buffer each address falls in.
#   `DOMAINS` (one NVLink domain per rank) becomes `domains`.
# * Several hosts (--nnodes N --node-rank i --master-addr --master-port, the script run once
#   per host with one shared --work): each host's torchrun records its local ranks, node 0
#   waits for every rank.json and merges. rank.json carries the host name, the GPU's NVLink
#   fabric state (nvidia-smi) and the process's RC QPs (`rdma res show qp`: local / remote
#   QPN). POSIX-fd and legacy-IPC keys are per host; launch times are taken relative to a
#   common barrier per host. Each IBGDA handle's peer is the rank whose QP is its QP's remote
#   QPN, else the rank whose handle has its rkey as lkey at its remote base, else the nic()
#   label (capture.json "nic_pairs"). `domains`: one NVLink domain per host, or per fabric
#   clique when a kernel pointer of one rank resolves into memory owned on another host.
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python")
)
from ptxv._cubin import decode_tmap  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LC = os.path.join(ROOT, "probes/launchcap")
CUDA = os.environ.get("PTXV_CUDA_HOME", "/usr/local/cuda-13.0")
ESIZE = {
    "bf16": 2,
    "f16": 2,
    "u16": 2,
    "i16": 2,
    "f32": 4,
    "u32": 4,
    "i32": 4,
    "f64": 8,
    "u64": 8,
    "i64": 8,
    "u8": 1,
    "i8": 1,
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--file", required=True, help="Python file defining setup() and run()"
    )
    ap.add_argument("--out", required=True, help="case directory to write")
    ap.add_argument(
        "--nproc", type=int, default=3, help="ranks (torchrun --nproc-per-node)"
    )
    ap.add_argument("--name", help="case name (default: the directory name)")
    ap.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="hosts: run this script once per host (--node-rank), same --work on a shared filesystem",
    )
    ap.add_argument("--node-rank", type=int, default=0)
    ap.add_argument("--master-addr", help="node 0's address (--nnodes > 1)")
    ap.add_argument("--master-port", type=int)
    ap.add_argument(
        "--wait",
        type=int,
        default=600,
        help="node 0: seconds to wait for the other hosts' rank.json after its torchrun exits",
    )
    ap.add_argument(
        "--kernels", help="regex: keep only launches whose kernel name matches"
    )
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument(
        "--exit-fence",
        choices=["sys", "gpu"],
        help="write exit_fence (E-EXIT-FENCE: kernel completion acquires) into the case",
    )
    ap.add_argument(
        "--allow-fd-guess",
        action="store_true",
        help="S-VA-ALIAS: accept POSIX-fd imports matched to an export by time order among several candidates (default: a capture error, the physical identity is unknown)",
    )
    ap.add_argument("--work", default=None, help="scratch directory, kept")
    ap.add_argument(
        "--merge-only", action="store_true", help="re-merge the recordings in --work"
    )
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args()


A = parse_args()
WORK = os.path.abspath(A.work or f"/tmp/ptxv_capture_dist_{os.getpid()}")


def u64(raw, off):
    return struct.unpack_from("<Q", raw, off)[0]


# ================================================================ parent: run torchrun


def build_probes():
    lib = os.path.join(sys.prefix, "lib")
    inc = glob.glob(f"{CUDA}/targets/*/include")[0]
    # Per host: the hosts share WORK and may differ in toolchain.
    d = f"{WORK}/node{A.node_rank}" if A.nnodes > 1 else WORK
    os.makedirs(d, exist_ok=True)
    so_lc, so_cc = f"{d}/launchcap.so", f"{d}/cubincap.so"
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-O2", "-o", so_lc, f"{LC}/launchcap.c", "-ldl"]
        + [f"-I{CUDA}/include"],
        check=True,
    )
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-O2", "-o", so_cc, f"{LC}/cubincap.c", f"-I{inc}"]
        + [f"-L{lib}", "-l:libcupti.so.13", f"-Wl,-rpath,{lib}"],
        check=True,
    )
    return so_lc, so_cc


def parent():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is None:
        sys.exit(
            "capture_dist: set CUDA_VISIBLE_DEVICES (e.g. 0,1,3) to the GPUs to use"
        )
    if len(vis.split(",")) < A.nproc:
        sys.exit(f"capture_dist: {A.nproc} ranks but CUDA_VISIBLE_DEVICES={vis}")
    if A.nnodes == 1:
        shutil.rmtree(WORK, ignore_errors=True)
    else:
        # WORK is shared by the hosts and cleaned by the caller; each host makes its ranks'.
        if not (A.master_addr and A.master_port and A.work):
            sys.exit(
                "capture_dist: --nnodes > 1 needs --master-addr, --master-port, --work"
            )
        if glob.glob(f"{WORK}/r*/rank.json"):
            sys.exit(f"capture_dist: {WORK} holds an earlier recording; clean it first")
    for r in range(A.node_rank * A.nproc, (A.node_rank + 1) * A.nproc):
        for d in ("cupti", "images"):
            os.makedirs(f"{WORK}/r{r}/{d}", exist_ok=A.nnodes > 1)
    so_lc, so_cc = build_probes()
    env = dict(
        os.environ,
        LD_PRELOAD=f"{so_lc}:{so_cc}",
        LAUNCHCAP_OUT=f"{WORK}/r%r/launches.jsonl",
        LAUNCHCAP_IMAGES=f"{WORK}/r%r/images",
        LAUNCHCAP_MEMOPS="1",
        LAUNCHCAP_VMM="1",
        CUBINCAP_DIR=f"{WORK}/r%r/cupti",
        PTXV_CAPTURE_DIST_WORK=WORK,
    )
    port = A.master_port or 29500 + os.getpid() % 2000
    hosts = []
    if A.nnodes > 1:
        hosts = ["--nnodes", str(A.nnodes), "--node-rank", str(A.node_rank)]
        hosts += ["--master-addr", A.master_addr]
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(A.nproc),
        *hosts,
        "--master-port",
        str(port),
        os.path.abspath(__file__),
        "--worker",
    ] + [a for a in sys.argv[1:] if a != "--worker"]
    if "--work" not in sys.argv:
        cmd += ["--work", WORK]
    r = subprocess.run(cmd, env=env)
    if r.returncode:
        sys.exit(
            f"capture_dist: torchrun failed ({r.returncode}); recordings in {WORK}"
        )
    if A.node_rank:
        return  # node 0 merges
    want = [f"{WORK}/r{r}/rank.json" for r in range(A.nproc * A.nnodes)]
    for _ in range(A.wait * 2):
        if all(os.path.exists(w) for w in want):
            break
        time.sleep(0.5)
    miss = [w for w in want if not os.path.exists(w)]
    if miss:
        sys.exit(
            f"capture_dist: {len(miss)} rank(s) never wrote rank.json, e.g. {miss[0]}"
        )
    merge()
    if not A.work:
        shutil.rmtree(WORK, ignore_errors=True)


# ================================================================ worker: one rank


# doca_gpu_dev_verbs_qp (DOCA GPUNetIO common/doca_gpunetio_verbs_dev.h, packed): byte offsets.
QP_STRUCT = 296
QP_F = {
    "sq_rsvd_index": (0, 8),
    "sq_wqe_pi": (16, 8),
    "sq_num": (24, 4),
    "sq_wqe_num": (60, 2),
    "sq_wqe_daddr": (64, 8),
    "sq_dbrec": (72, 8),
    "sq_db": (80, 8),
    "cqe_daddr": (160, 8),
    "cqe_num": (172, 4),
    "cq_dbrec": (176, 8),
    "cqe_ci": (184, 8),
    "cqe_rsvd": (200, 8),
    "nic_handler": (288, 4),
}
# The tail of mscclpp's PortChannelIbgdaDeviceHandle (port_channel_ibgda_device.hpp), after
# the semaphore handle: qp_, rkeyBe_, lkeyBe_, remoteTokenRkeyBe_, sinkLkeyBe_, remoteBase_,
# localBase_, remoteInboundToken_, sinkAddr_ (56 bytes).
H_TAIL = 56


def nic_describe(mod, inputs, dev_read):
    """Read each IBGDA handle the target's nic() names and the DOCA QP it points to."""
    out = []
    for d in mod.nic(inputs):
        if "region" in d:
            # A buffer the handles name that the allocation records miss (mscclpp's window).
            a, z = d["region"]
            out.append({"regions": [(a, z)], "extra": True})
            continue
        raw = dev_read(d["addr"], d["size"])
        t = d["size"] - H_TAIL
        q = {
            "peer": int(d["peer"]),
            "addr": d["addr"],
            "size": d["size"],
            "tail": t,
            "qp": u64(raw, t),
            "rkey": int.from_bytes(raw[t + 8 : t + 12], "big"),
            "lkey": int.from_bytes(raw[t + 12 : t + 16], "big"),
            "token_rkey": int.from_bytes(raw[t + 16 : t + 20], "big"),
            "sink_lkey": int.from_bytes(raw[t + 20 : t + 24], "big"),
            "remote_base": u64(raw, t + 24),
            "local_base": u64(raw, t + 32),
            "remote_token": u64(raw, t + 40),
            "sink": u64(raw, t + 48),
        }
        qs = dev_read(q["qp"], QP_STRUCT)
        q["qp_fields"] = {
            k: int.from_bytes(qs[o : o + n], "little") for k, (o, n) in QP_F.items()
        }
        f = q["qp_fields"]
        # The semaphore handle at the front: inboundToken*, expectedInboundToken*,
        # numInboundTokens (semaphore_device.hpp), this rank's token words.
        ntok = int.from_bytes(raw[16:20], "little") if t >= 20 else 1
        # The handle's semaphore: inboundToken (the per-QP words the peer's NIC adds to).
        ntok = int.from_bytes(raw[16:20], "little") if t >= 20 else 1
        q["regions"] = [
            (u64(raw, 0), 8 * max(1, ntok)),
            (q["qp"], QP_STRUCT),
            (f["sq_wqe_daddr"], 64 * f["sq_wqe_num"]),
            (f["cqe_daddr"], 64 * f["cqe_num"]),
            (f["sq_dbrec"], 4),
            (u64(raw, 0), 8 * max(1, ntok)),
            (u64(raw, 8), 8),
        ] + [tuple(x) for x in d.get("regions", [])]
        out.append(q)
    return out


def rdma_qps_of(pid):
    """This process's RC QPs as the kernel lists them (`rdma res show qp`): local and remote QPN
    per HCA. The remote QPN is what pairs a GPU's QP with its peer's across hosts."""
    try:
        out = subprocess.run(
            ["rdma", "-j", "res", "show", "qp"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        qs = json.loads(out or "[]")
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    return [
        {k: q.get(k) for k in ("ifname", "port", "lqpn", "rqpn", "type", "state")}
        for q in qs
        if q.get("pid") == pid and q.get("type") == "RC"
    ]


def gpu_fabric(uuid):
    """NVLink fabric membership of this rank's GPU (nvidia-smi -q: cluster UUID, clique id)."""
    idx = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
    try:
        out = subprocess.run(
            ["nvidia-smi", "-q", "-i", idx], capture_output=True, text=True, timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    f, sec = {}, False
    for ln in out.splitlines():
        if ln.strip() == "Fabric":
            sec = True
        elif sec and ":" in ln:
            k, v = (x.strip() for x in ln.split(":", 1))
            if k in ("State", "Status", "CliqueId", "ClusterUUID"):
                f[k] = v
        elif sec and ln.strip() and not ln.startswith(" " * 8):
            sec = False
    return f


def worker():  # noqa: C901 (per-rank capture worker: one linear pass over shared capture state)
    import importlib.util

    import torch
    import torch.distributed as dist

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    group = dist.group.WORLD.group_name
    W = f"{WORK}/r{rank}"
    CAP = f"{W}/launches.jsonl"
    spec = importlib.util.spec_from_file_location("capture_target", A.file)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(A.file)))
    spec.loader.exec_module(mod)
    inputs = mod.setup(rank, world, group)

    import ctypes

    from torch.distributed._symmetric_memory import _SymmetricMemory

    rt = ctypes.CDLL("libcudart.so.13")
    rt.cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]

    def _dev_read(addr, n):
        buf = ctypes.create_string_buffer(n)
        e = rt.cudaMemcpy(buf, ctypes.c_void_p(addr), n, 2)
        if e:
            raise RuntimeError(
                f"rank {rank}: cudaMemcpy from {addr:#x} ({n} bytes): {e}"
            )
        return buf.raw

    def storage_range(t):
        st = t.untyped_storage()
        return st.data_ptr(), st.nbytes()

    def active_blocks():
        out = []
        for seg in torch.cuda.memory_snapshot():
            addr = seg["address"]
            for blk in seg["blocks"]:
                if blk["state"] == "active_allocated":
                    out.append((blk.get("address", addr), blk["size"]))
                addr += blk["size"]
        return out

    def flat_tensors(x, prefix="out"):
        if isinstance(x, torch.Tensor):
            return [(prefix, x)]
        if isinstance(x, dict):
            return [t for k, v in x.items() for t in flat_tensors(v, k)]
        if isinstance(x, (tuple, list)):
            return [t for i, v in enumerate(x) for t in flat_tensors(v, f"{prefix}{i}")]
        return []

    DT = {
        torch.bfloat16: "bf16",
        torch.float16: "f16",
        torch.float32: "f32",
        torch.float64: "f64",
        torch.int32: "i32",
        torch.int64: "i64",
        torch.int16: "i16",
        torch.int8: "i8",
        torch.uint8: "u8",
        torch.bool: "u8",
    }

    def tinfo(t):
        p, z = storage_range(t)
        return {
            "ptr": p,
            "size": z,
            "data_ptr": t.data_ptr(),
            "nbytes": t.numel() * t.element_size(),
            "dtype": DT.get(t.dtype, "u8"),
            "shape": list(t.shape),
            "contiguous": t.is_contiguous(),
        }

    # Symmetric-memory handles: their layout (data buffer, signal pad) for every rank.
    symm = []
    for n, v in inputs.items():
        if isinstance(v, _SymmetricMemory):
            symm.append(
                {
                    "name": n,
                    "rank": v.rank,
                    "world": v.world_size,
                    "buffer_ptrs": list(v.buffer_ptrs),
                    "signal_pad_ptrs": list(v.signal_pad_ptrs),
                    "buffer_ptrs_dev": v.buffer_ptrs_dev,
                    "signal_pad_ptrs_dev": v.signal_pad_ptrs_dev,
                    "buffer_size": v.buffer_size,
                    "signal_pad_size": v.signal_pad_size,
                    "multicast_ptr": v.multicast_ptr,
                    "offset": v.offset,
                }
            )
    tensors = {
        n: t for n, t in inputs.items() if isinstance(t, torch.Tensor) and t.is_cuda
    }

    def run():
        return mod.run(**inputs)

    if not A.no_warmup:
        run()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    first = sum(1 for _ in open(CAP)) if os.path.exists(CAP) else 0
    # Host clock at a point every rank passed together (after the barrier): launch times of
    # ranks on different hosts are compared relative to it.
    t_ref = time.monotonic_ns()
    nic = nic_describe(mod, inputs, _dev_read) if hasattr(mod, "nic") else []
    rdma_qps = rdma_qps_of(os.getpid()) if nic else []
    vmm = [json.loads(ln) for ln in open(CAP)][:first]
    vmm = [r for r in vmm if "vmm" in r]
    pre_blocks = active_blocks()
    pre = {}  # address -> bytes, every region whose initial contents a case may need
    for _n, t in tensors.items():
        p, z = storage_range(t)
        if z:
            pre[p] = _dev_read(p, z)
    for a, z in pre_blocks:
        if a not in pre and z <= 4 << 20:
            pre[a] = _dev_read(a, z)
    # This rank's own VMM allocations (mapped, created here): the symmetric buffers.
    created = {r["handle"]: r for r in vmm if r["vmm"] == "create" and not r["r"]}
    released = set()
    live_maps = []
    for r in vmm:
        if r["vmm"] == "map" and not r["r"]:
            live_maps.append(r)
        elif r["vmm"] == "unmap":
            live_maps = [m for m in live_maps if m["ptr"] != r["ptr"]]
    for m in live_maps:
        if m["handle"] in created:
            n = m["size"]
            # Only as far as a symmetric handle's layout reaches (the 2 MB granule is mostly
            # unused); whole otherwise.
            for s in symm:
                b, sp = s["buffer_ptrs"][s["rank"]], s["signal_pad_ptrs"][s["rank"]]
                if m["ptr"] <= b < m["ptr"] + m["size"]:
                    n = max(b + s["buffer_size"], sp + s["signal_pad_size"]) - m["ptr"]
            pre[m["ptr"]] = _dev_read(m["ptr"], min(n, m["size"]))
    del released
    # Plain cudaMalloc allocations (not the caching allocator's own segments) and mapped host
    # memory: DeepEP's NVLink buffer, its device pointer tables, its host-visible counters.
    segments = [seg["address"] for seg in torch.cuda.memory_snapshot()]
    plain = {}
    for r in vmm:
        if r["vmm"] in ("alloc", "host_alloc") and not r["r"]:
            plain[r["ptr"]] = (r["size"], r["vmm"])
        elif r["vmm"] == "free":
            plain.pop(r["ptr"], None)
    plain = {a: v for a, v in plain.items() if a not in segments}
    for a, (z, _) in plain.items():
        if z <= 16 << 20 and a not in pre:
            pre[a] = _dev_read(a, z)
    # Module data globals (`__device__` variables, S-GLOBAL-INIT) the launched kernels'
    # constant banks name: their device contents now, through the module / library handle
    # launchcap recorded for the warm-up launch (cuModuleGetGlobal / cuLibraryGetGlobal).
    # String literals ($str) are left to their ELF bytes.
    globals_pre = []
    cu = ctypes.CDLL("libcuda.so.1")
    gfun = {"mod": cu.cuModuleGetGlobal_v2, "lib": cu.cuLibraryGetGlobal}
    for f in gfun.values():
        f.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
    cupti = sorted(glob.glob(f"{W}/cupti/*.cubin"), key=os.path.getsize)
    seen_k = set()
    for x in [json.loads(ln) for ln in open(CAP)][:first]:
        if "name" not in x or x["name"] in seen_k:
            continue
        if A.kernels and not re.search(A.kernels, x["name"]):
            continue
        seen_k.add(x["name"])
        cub = next((c for c in cupti if elf_defines(c, x["name"])), None)
        if not cub:
            continue
        for sym, size, _init in data_globals(open(cub, "rb").read(), x["name"]):
            if sym.startswith("$"):
                continue
            ptr, n = ctypes.c_uint64(0), ctypes.c_size_t(0)
            kind = "lib" if x.get("lib") else "mod"
            h = x.get(kind) or 0
            err = (
                gfun[kind](ctypes.byref(ptr), ctypes.byref(n), h, sym.encode())
                if h
                else -1
            )
            if err:
                globals_pre.append(
                    {"kernel": x["name"], "symbol": sym, "size": size, "error": err}
                )
                continue
            if ptr.value not in pre:
                pre[ptr.value] = _dev_read(ptr.value, n.value)
            globals_pre.append(
                {"kernel": x["name"], "symbol": sym, "ptr": ptr.value, "size": n.value}
            )
    # IBGDA queue memory (DOCA allocations the caching allocator does not see).
    for q in nic:
        for a, z in q["regions"]:
            if not any(b <= a and a + z <= b + len(pre[b]) for b in pre):
                pre[a] = _dev_read(a, z)
    named_pre = {n: tinfo(t) for n, t in tensors.items()}
    pre_bytes_of = {
        n: _dev_read(t.data_ptr(), t.numel() * t.element_size())
        for n, t in tensors.items()
    }
    dist.barrier()  # before recording: the barrier's own allocations are not the op's
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    t0 = sum(1 for _ in open(CAP))
    result = run()
    torch.cuda.synchronize()
    history = [e for tr in torch.cuda.memory._snapshot()["device_traces"] for e in tr]
    torch.cuda.memory._record_memory_history(enabled=None)
    recs = [json.loads(ln) for ln in open(CAP)][t0:]
    dist.barrier()
    during = [(e["addr"], e["size"]) for e in history if e["action"] == "alloc"]
    outs = flat_tensors(result)
    for n, t in tensors.items():
        if _dev_read(t.data_ptr(), t.numel() * t.element_size()) != pre_bytes_of[
            n
        ] and not any(storage_range(o)[0] == storage_range(t)[0] for _, o in outs):
            outs.append((n, t))
    seen, uniq = set(), []
    for n, t in outs:
        if (t.data_ptr(), n) not in seen and not any(
            t.data_ptr() == o.data_ptr() for _, o in uniq
        ):
            seen.add((t.data_ptr(), n))
            uniq.append((n, t))
    outs = uniq
    post = {}
    for n, t in outs:
        post[n] = _dev_read(t.data_ptr(), t.numel() * t.element_size())
    os.makedirs(f"{W}/pre", exist_ok=True)
    os.makedirs(f"{W}/post", exist_ok=True)
    for a, b in pre.items():
        open(f"{W}/pre/{a:x}.bin", "wb").write(b)
    for n, b in post.items():
        open(f"{W}/post/{n}.bin", "wb").write(b)
    meta = {
        "rank": rank,
        "world": world,
        "device": torch.cuda.current_device(),
        "uuid": str(getattr(torch.cuda.get_device_properties(local), "uuid", "")),
        "arch": "sm_%d%da" % torch.cuda.get_device_capability(local),
        "host": socket.gethostname(),
        "node": int(os.environ.get("GROUP_RANK", A.node_rank)),
        "local_rank": local,
        "t_ref": t_ref,
        "fabric": gpu_fabric(
            str(getattr(torch.cuda.get_device_properties(local), "uuid", ""))
        ),
        "rdma_qps": rdma_qps,
        "torch": torch.__version__,
        "pre_blocks": pre_blocks,
        "plain": [(a, z, k) for a, (z, k) in plain.items()],
        "during": during,
        "named": named_pre,
        "outputs": [(n, tinfo(t)) for n, t in outs],
        "symm": symm,
        "globals": globals_pre,
        "replicas": list(getattr(mod, "REPLICAS", [])),
        "nic": nic,
        "domains": list(getattr(mod, "DOMAINS", [])),
        "pre": sorted(pre),
        "records": recs,
        "setup_records": [json.loads(ln) for ln in open(CAP)][:first],
    }
    json.dump(meta, open(f"{W}/rank.json", "w"))
    dist.barrier()
    dist.destroy_process_group()


# ================================================================ merge (parent)


def _load_cubin_helpers():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ptxv_cubin", os.path.join(ROOT, "python/ptxv/_cubin.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_cubin = _load_cubin_helpers()
data_globals = _cubin.data_globals
decode_tmap = _cubin.decode_tmap


def elf_defines(path, name):
    with open(path, "rb") as f:
        d = f.read()
    return d[:4] == b"\x7fELF" and (b".text." + name.encode() + b"\x00") in d


def prune(data, keep):
    """The module cubin with every entry kernel not in `keep` overwritten by NOPs (as
    tools/capture.py: same layout, symbols, relocations; compresses to a fraction)."""
    d = bytearray(data)
    shoff = u64(d, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x3A)
    secs = [
        struct.unpack_from("<IIQQQQIIQQ", d, shoff + i * shentsize)
        for i in range(shnum)
    ]

    def cstr(off):
        return bytes(d[off : d.index(b"\0", off)]).decode()

    names = [cstr(secs[shstrndx][4] + sec[0]) for sec in secs]
    entries = set()
    for st in (sec for sec in secs if sec[1] == 2):
        strtab = secs[st[6]][4]
        for o in range(st[4], st[4] + st[5], 24):
            n, info, other = struct.unpack_from("<IBB", d, o)
            if info & 15 == 2 and other & 0x10:
                entries.add(cstr(strtab + n))
    text = {n[6:]: sec for n, sec in zip(names, secs) if n.startswith(".text.")}
    k = text[next(iter(keep))]
    nop = bytes(d[k[4] + k[5] - 16 : k[4] + k[5]])
    for n, sec in text.items():
        if n in entries and n not in keep and sec[1] != 8:
            d[sec[4] : sec[4] + sec[5]] = nop * (sec[5] // 16)
    return bytes(d)


def kparam_layout(cubin, kernel):
    """[(offset, size)] of `kernel`'s parameters, from the KPARAM_INFO records of its .nv.info."""
    out = subprocess.run(
        [f"{CUDA}/bin/cuobjdump", "-elf", cubin], capture_output=True, text=True
    ).stdout
    lines = out.splitlines()
    try:
        i = next(k for k, ln in enumerate(lines) if ln.strip() == f".nv.info.{kernel}")
    except StopIteration:
        return None
    ps = []
    for ln in lines[i + 1 :]:
        if ln.startswith("."):
            break
        v = ln.strip()
        if v.startswith("Value:") and "Ordinal" in v and "Offset" in v:
            f = dict(re.findall(r"(Ordinal|Offset|Size)\s*:\s*(0x[0-9a-fA-F]+)", v))
            ps.append((int(f["Ordinal"], 16), int(f["Offset"], 16), int(f["Size"], 16)))
    return [(o, z) for _, o, z in sorted(ps)] or None


_LAYOUTS = {}


def split_packed(r, x):
    """A launch through CU_LAUNCH_PARAM_BUFFER_POINTER (mscclpp / cupy `extra` launches) is
    recorded as one packed argument block; split it into the kernel's parameters."""
    ps = x["params"]
    if len(ps) != 1 or not ps[0].get("packed"):
        return
    key = (r, x["name"])
    if key not in _LAYOUTS:
        cands = glob.glob(f"{WORK}/r{r}/cupti/*.cubin") + glob.glob(
            f"{WORK}/r{r}/images/*.bin"
        )
        lay = None
        for c in sorted(cands, key=os.path.getsize):
            if elf_defines(c, x["name"]):
                lay = kparam_layout(c, x["name"])
                if lay:
                    break
        _LAYOUTS[key] = lay
    lay = _LAYOUTS[key]
    raw = bytes.fromhex(ps[0]["hex"])
    if not lay:
        return
    end = max(o + z for o, z in lay)
    raw = raw + bytes(max(0, end - len(raw)))
    x["params"] = [{"offset": o, "hex": raw[o : o + z].hex()} for o, z in lay]


def nic_pairing(ranks, warnings, notes):
    """Peer of each IBGDA handle, (rank, handle index) -> (peer rank, how), from what the
    hosts recorded rather than the target's nic() label:
    * "qpn": the kernel's view of the rank's process (`rdma res show qp`) gives the handle's
      QP (sq_num) a remote QPN; the peer is the one rank owning a handle with that QPN whose
      own remote QPN is this QP. QPNs are per HCA, so this holds across hosts.
    * "rkey": else, the one other rank with a handle whose lkey is this handle's rkey and whose
      local base is this handle's remote base (one registered buffer per direction pair).
    Disagreement with nic() is a warning; the recorded pairing wins."""
    qs = {}  # (rank, handle index) -> handle
    for r, m in enumerate(ranks):
        for qi, q in enumerate(m.get("nic", [])):
            if not q.get("extra"):
                qs[(r, qi)] = q
    rq = {}  # (rank, lqpn) -> {rqpn}
    for r, m in enumerate(ranks):
        for e in m.get("rdma_qps", []):
            if e.get("rqpn") is not None:
                rq.setdefault((r, e["lqpn"]), set()).add(e["rqpn"])
    out, counts, bad = {}, {"qpn": 0, "rkey": 0, "nic()": 0}, 0
    for (r, qi), q in qs.items():
        lq = q["qp_fields"]["sq_num"]
        how, peer = "nic()", q["peer"]
        rem = rq.get((r, lq), set())
        if len(rem) == 1:
            (n,) = rem
            c = {
                p
                for (p, pj), x in qs.items()
                if p != r and x["qp_fields"]["sq_num"] == n and lq in rq.get((p, n), ())
            }
            if len(c) == 1:
                how, peer = "qpn", c.pop()
        if how == "nic()":
            c = {
                p
                for (p, pj), x in qs.items()
                if p != r
                and x["lkey"] == q["rkey"]
                and x["local_base"] == q["remote_base"]
            }
            if len(c) == 1:
                how, peer = "rkey", c.pop()
        if peer != q["peer"]:
            bad += 1
            warnings.append(
                f"rank {r}: IBGDA handle {qi} (QP {lq:#x}): nic() names peer {q['peer']}, "
                f"the recorded pairing ({how}) gives {peer}; {peer} used"
            )
        counts[how] += 1
        out[(r, qi)] = (peer, how)
    if qs:
        notes.append(
            f"IBGDA pairing of {len(qs)} QP(s): {counts['qpn']} by QPN (rdma res show qp), "
            f"{counts['rkey']} by rkey + remote base, {counts['nic()']} by nic() alone; "
            f"{bad} disagree with nic()"
        )
    return out


def merge():  # noqa: C901 (cross-rank merge: one linear pass over shared capture state)
    R = A.nproc * A.nnodes
    ranks = [json.load(open(f"{WORK}/r{r}/rank.json")) for r in range(R)]
    warnings, unresolved, notes = [], [], []
    # S-VA-ALIAS: mappings whose physical identity the recording does not determine. Two
    # mappings of one allocation taken as distinct memory (or two allocations as one) would
    # hide aliasing and data flow, so each is a capture error rather than a guess.
    identity_errors = []
    # Hosts: POSIX fds and legacy IPC handles name allocations only within one host. Host
    # clocks (launch "t", CLOCK_MONOTONIC) are made comparable across hosts by subtracting
    # each host's earliest t_ref (taken by every rank right after a common barrier).
    host = [m.get("host", "") for m in ranks]
    hosts = list(dict.fromkeys(host))
    if len(hosts) > 1:
        ref = {h: min(m["t_ref"] for m in ranks if m["host"] == h) for h in hosts}
        for m in ranks:
            for x in m["setup_records"] + m["records"]:
                if "t" in x:
                    x["t"] -= ref[m["host"]]

    def fdk(r, x):
        return f"fd:{host[r]}:{x['fd_dev']}:{x['fd_ino']}"

    def ipck(r, x):
        return f"ipc:{host[r]}:{x['ipc']}" if len(hosts) > 1 else f"ipc:{x['ipc']}"

    kfilter = re.compile(A.kernels) if A.kernels else None

    # ------------------------------------------------ physical allocations across ranks
    # phys id -> dict(owner, kind); per rank: handle -> phys id; VA maps.
    phys = {}
    handle_phys = [{} for _ in range(R)]
    maps = [[] for _ in range(R)]  # (start, end, phys, offset in phys)
    mc_bind = {}  # multicast phys -> [(rank, mem phys, mem offset, mc offset, size)]
    for r, m in enumerate(ranks):
        recs = m["setup_records"] + m["records"]
        vm = [x for x in recs if "vmm" in x]
        for x in vm:
            if x["vmm"] == "create" and not x["r"]:
                handle_phys[r][x["handle"]] = pid = f"local:{r}:{x['handle']}"
                phys[pid] = {"owner": r, "kind": "mem", "size": x["size"]}
            elif x["vmm"] == "mc_create" and not x["r"]:
                handle_phys[r][x["handle"]] = pid = f"local:{r}:{x['handle']}"
                phys[pid] = {
                    "owner": r,
                    "kind": "multicast",
                    "size": x["size"],
                    "ndev": x["ndev"],
                }
    # A POSIX-fd export is named by the fd's (device, inode). cuMemExportToShareableHandle
    # returns anonymous-inode fds that can share one inode; such keys are ambiguous and are
    # matched export -> import below instead.
    fd_count = {}
    for r, m in enumerate(ranks):
        for x in m["setup_records"] + m["records"]:
            if (
                x.get("vmm") == "export"
                and not x["r"]
                and not x["fabric"]
                and x["fd_ino"]
            ):
                k = fdk(r, x)
                fd_count[k] = fd_count.get(k, 0) + 1
    fd_exports = []  # ambiguous fd exports: [t, rank, phys id, matched]
    for r, m in enumerate(ranks):
        vm = [x for x in m["setup_records"] + m["records"] if "vmm" in x]
        for x in vm:  # exports name a physical allocation globally
            if x["vmm"] == "export" and not x["r"]:
                key = x["fabric"] or (x["fd_ino"] and fdk(r, x))
                if key and not x["fabric"] and fd_count.get(key, 0) > 1:
                    continue
                old = handle_phys[r].get(x["handle"])
                if old and key:
                    phys[f"g:{key}"] = phys[old]
                    handle_phys[r][x["handle"]] = f"g:{key}"
        for x in vm:
            if (
                x["vmm"] == "export"
                and not x["r"]
                and not x["fabric"]
                and fd_count.get(fdk(r, x), 0) > 1
                and handle_phys[r].get(x["handle"])
            ):
                fd_exports.append(
                    [x.get("t", 0), r, handle_phys[r][x["handle"]], False]
                )
    # Ambiguous fd imports: each export is sent to one importer. Imports in time order take the
    # earliest unmatched earlier export of another rank whose kind (memory / multicast) and size
    # fit the importer's later use of the handle.
    fd_import = {}  # id(import record) -> phys id
    fd_guess = 0
    imps = []
    for r, m in enumerate(ranks):
        recs = [x for x in m["setup_records"] + m["records"] if "vmm" in x]
        for i, x in enumerate(recs):
            if not (
                x["vmm"] == "import"
                and not x["r"]
                and not x["fabric"]
                and fd_count.get(fdk(r, x), 0) > 1
            ):
                continue
            h, kind, need = x["handle"], "mem", 0
            for y in recs[i + 1 :]:
                if y["vmm"] == "release" and y.get("handle") == h:
                    break
                if y["vmm"] in ("mc_add_device",) and y.get("handle") == h:
                    kind = "multicast"
                if y["vmm"] in ("mc_bind_mem", "mc_bind_addr") and y.get("mc") == h:
                    kind = "multicast"
                if y["vmm"] == "map" and y.get("handle") == h:
                    need = max(need, y["offset"] + y["size"])
            imps.append((x.get("t", 0), r, id(x), kind, need))
    for t, r, xid, kind, need in sorted(imps):
        cand = [
            e
            for e in fd_exports
            if not e[3]
            and e[1] != r
            and host[e[1]] == host[r]
            and e[0] < t
            and phys.get(e[2], {}).get("kind") == kind
            and (phys[e[2]].get("size") or 0) >= need
        ]
        if not cand:
            continue
        e = min(cand, key=lambda e: e[0])
        e[3] = True
        fd_import[xid] = e[2]
        fd_guess += len({c[2] for c in cand}) > 1
    if imps:
        warnings.append(
            f"{len(imps)} POSIX-fd import(s) with a shared fd inode: {len(fd_import)} matched "
            f"to an export by time order, {fd_guess} of them with several candidates"
        )
    if fd_guess and not A.allow_fd_guess:
        identity_errors.append(
            f"{fd_guess} POSIX-fd import(s) matched by time order among several exports of the same kind and size (--allow-fd-guess accepts the guess)"
        )
    elif fd_guess:
        notes.append(
            f"S-VA-ALIAS: {fd_guess} POSIX-fd import(s) matched to an export by time order among several candidates (--allow-fd-guess)"
        )
    # Legacy IPC (cudaIpcGetMemHandle on a cudaMalloc allocation; DeepEP's NVLink buffer):
    # the 64-byte handle names the exporter's allocation, sized from its cuMemAlloc.
    for r, m in enumerate(ranks):
        recs = m["setup_records"] + m["records"]
        allocs = {
            x["ptr"]: x["size"] for x in recs if x.get("vmm") == "alloc" and not x["r"]
        }
        for x in recs:
            if x.get("vmm") == "ipc_get" and not x["r"]:
                base = max(
                    (a for a in allocs if a <= x["ptr"] < a + allocs[a]), default=None
                )
                # The handle names the whole allocation; an importer's pointer is taken as the
                # exporter's pointer, which holds only when the exporter passed the base.
                if base is None or base != x["ptr"]:
                    identity_errors.append(
                        f"rank {r}: cudaIpcGetMemHandle of {x['ptr']:#x}, "
                        + (
                            "not in a recorded allocation"
                            if base is None
                            else f"inside the allocation at {base:#x}"
                        )
                        + ": the importer's offset is unknown"
                    )
                size = allocs[base] - (x["ptr"] - base) if base is not None else 1
                phys[ipck(r, x)] = {"owner": r, "kind": "mem", "size": size}
                maps[r].append((x["ptr"], x["ptr"] + size, ipck(r, x), 0))
    for r, m in enumerate(ranks):
        recs = m["setup_records"] + m["records"]
        window = {id(x) for x in m["records"]}
        live = list(maps[r])
        for x in (x for x in recs if "vmm" in x):
            if (
                id(x) in window
                and not x.get("r")
                and x["vmm"]
                in ("map", "unmap", "mc_bind_mem", "mc_bind_addr", "mc_unbind")
            ):
                identity_errors.append(
                    f"rank {r}: {x['vmm']} during the captured window (the mappings the launches see are not one set)"
                )
            if x["vmm"] == "import" and not x["r"] and id(x) in fd_import:
                handle_phys[r][x["handle"]] = fd_import[id(x)]
            elif x["vmm"] == "import" and not x["r"]:
                key = x["fabric"] or fdk(r, x)
                if f"g:{key}" not in phys:
                    warnings.append(
                        f"rank {r}: imported allocation {key[:24]}... has no exporter"
                    )
                    phys[f"g:{key}"] = {"owner": None, "kind": "mem", "size": 0}
                handle_phys[r][x["handle"]] = f"g:{key}"
            elif x["vmm"] == "map" and not x["r"]:
                pid = handle_phys[r].get(x["handle"])
                live.append((x["ptr"], x["ptr"] + x["size"], pid, x["offset"]))
            elif x["vmm"] == "unmap":
                live = [ln for ln in live if ln[0] != x["ptr"]]
            elif x["vmm"] == "mc_bind_mem" and not x["r"]:
                mc_bind.setdefault(handle_phys[r].get(x["mc"]), []).append(
                    (
                        r,
                        handle_phys[r].get(x["handle"]),
                        x.get("offset", 0),
                        x.get("mc_offset", 0),
                        x["size"],
                    )
                )
            elif x["vmm"] == "mc_bind_addr" and not x["r"]:
                hit = [ln for ln in live if ln[0] <= x["ptr"] < ln[1]]
                if hit:
                    s_, _e, pid_, off_ = hit[-1]
                    mc_bind.setdefault(handle_phys[r].get(x["mc"]), []).append(
                        (
                            r,
                            pid_,
                            off_ + x["ptr"] - s_,
                            x.get("mc_offset", 0),
                            x["size"],
                        )
                    )
                else:
                    warnings.append(
                        f"rank {r}: cuMulticastBindAddr of {x['ptr']:#x}: no mapping"
                    )
            elif x["vmm"] == "mc_unbind" and not x["r"]:
                # Before the window (in it: an identity error above): the device's binds of
                # that multicast object over the unbound range end.
                mcp = handle_phys[r].get(x["mc"])
                lo = x.get("mc_offset", 0)
                hi = lo + x.get("size", 1 << 62)
                before = mc_bind.get(mcp, [])
                mc_bind[mcp] = [
                    b
                    for b in before
                    if not (b[0] == r and b[3] < hi and lo < b[3] + b[4])
                ]
                if len(mc_bind[mcp]) == len(before):
                    warnings.append(f"rank {r}: cuMulticastUnbind of no recorded bind")
            elif x["vmm"] == "ipc_open" and not x["r"]:
                pid = ipck(r, x)
                size = phys.get(pid, {}).get("size", 1)
                live.append((x["ptr"], x["ptr"] + size, pid, 0))
        maps[r] = live
    # Owner-side address of each physical allocation (its creator's own mappings). An
    # allocation may be mapped in pieces or more than once by its owner: a physical byte
    # resolves through the owner mapping that covers it (`owner_addr`), the first one when
    # several do; none covering it leaves the pointer unresolved.
    owner_va = {}
    owner_maps = {}  # phys id -> [(start, end, offset in phys)] in mapping order
    for r in range(R):
        for s, e, pid, off in maps[r]:
            if (
                pid
                and phys.get(pid, {}).get("owner") == r
                and phys[pid]["kind"] == "mem"
            ):
                owner_va.setdefault(pid, s - off)
                owner_maps.setdefault(pid, []).append((s, e, off))

    def owner_addr(pid, p, n=1):
        """(owner's address of physical byte p of `pid`, that mapping's address of byte 0) for
        an owner mapping covering [p, p + n), or None (S-VA-ALIAS: never guessed)."""
        for s, e, off in owner_maps.get(pid, []):
            if off <= p and p + n <= off + (e - s):
                return s + (p - off), s - off
        return None

    mc_report = []
    for mc, binds in mc_bind.items():
        mc_report.append(
            {
                "multicast": mc[:40],
                "size": phys.get(mc, {}).get("size"),
                "members": [
                    (r, phys.get(p, {}).get("owner"), moff, mcoff, size)
                    for r, p, moff, mcoff, size in binds
                ],
                "mapped_in": [r for r in range(R) if any(m[2] == mc for m in maps[r])],
            }
        )

    # ------------------------------------------------ regions (future case buffers)
    # Each rank's caching-allocator components plus the owner's view of its VMM allocations,
    # split at the symmetric-memory layout. Region: dict(dev, start, end, name, dtype, shape).
    regions = []

    def add_region(dev, start, end, name, dtype="u8", shape=None, kind="block"):
        regions.append(
            {
                "dev": dev,
                "start": start,
                "end": end,
                "name": name,
                "dtype": dtype,
                "shape": shape or [end - start],
                "kind": kind,
            }
        )

    for r, m in enumerate(ranks):
        named = m["named"]
        outs = dict(m["outputs"])
        ivs = []
        for n, t in list(named.items()) + list(outs.items()):
            if t["size"]:
                ivs.append((t["ptr"], t["ptr"] + t["size"], n))
        for a, z in m["pre_blocks"]:
            ivs.append((a, a + z, None))
        for a, z in m["during"]:
            ivs.append((a, a + z, None))
        # VMM allocations are not caching-allocator blocks: drop named tensors inside them.
        vmm_ranges = [
            (s, e)
            for s, e, pid, _ in maps[r]
            if pid and phys.get(pid, {}).get("owner") is not None
        ]
        ivs = [iv for iv in ivs if not any(s <= iv[0] < e for s, e in vmm_ranges)]
        ivs.sort(key=lambda iv: (iv[0], iv[1], iv[2] or ""))
        comps = []
        for iv in ivs:
            if comps and iv[0] < comps[-1][1]:
                comps[-1][1] = max(comps[-1][1], iv[1])
                comps[-1][2].append(iv)
            else:
                comps.append([iv[0], iv[1], [iv]])
        nws = 0
        for s, e, members in comps:
            nm = [x for x in members if x[2]]
            solo = [x for x in nm if x[0] == s and x[1] == e]
            if solo:
                n = solo[0][2]
                t = named.get(n) or outs.get(n)
                if (
                    t["contiguous"]
                    and t["nbytes"] == t["size"]
                    and t["data_ptr"] == t["ptr"]
                ):
                    add_region(r, s, e, f"{n}_r{r}", t["dtype"], t["shape"] or [1])
                    continue
            if len(nm) == 1 and nm[0][0] == s:
                # One tensor at the start of its (rounded-up) block: its element type.
                n = nm[0][2]
                t = named.get(n) or outs.get(n)
                es = ESIZE.get(t["dtype"], 1)
                if (e - s) % es == 0:
                    add_region(r, s, e, f"{n}_r{r}", t["dtype"], [(e - s) // es])
                    continue
            if nm:
                name = f"{nm[0][2]}_r{r}"
            else:
                name = f"W{nws}_r{r}"
                nws += 1
            add_region(r, s, e, name)
        # Plain cudaMalloc / mapped-host allocations (not the caching allocator's): one buffer
        # each (an exported IPC buffer is the owner's view of its peers' mappings).
        for k, (a, z, kind) in enumerate(m.get("plain", [])):
            pre = [p for p in m["pre"] if p == a]
            if any(g["dev"] == r and g["start"] <= a < g["end"] for g in regions):
                continue
            add_region(
                r,
                a,
                a + z,
                f"{'host' if kind == 'host_alloc' else 'dev'}{k}_r{r}",
                kind="plain",
            )
        # Own VMM allocations: symmetric layout if a handle describes it.
        for s, e, pid, _off in maps[r]:
            if (
                not pid
                or phys.get(pid, {}).get("owner") != r
                or phys[pid]["kind"] != "mem"
            ):
                continue
            if pid.startswith("ipc:"):
                continue  # a cudaMalloc buffer: added above
            lay = [h for h in m["symm"] if s <= h["buffer_ptrs"][r] < e]
            if lay:
                h = lay[0]
                b, sp = h["buffer_ptrs"][r], h["signal_pad_ptrs"][r]
                # The tensor living in the data buffer names it (with its dtype).
                tn = [
                    (n, t)
                    for n, t in named.items()
                    if t["data_ptr"] == b + h["offset"] and t["contiguous"]
                ]
                if tn and h["offset"] == 0 and tn[0][1]["nbytes"] == h["buffer_size"]:
                    n, t = tn[0]
                    add_region(
                        r,
                        b,
                        b + h["buffer_size"],
                        f"{n}_r{r}",
                        t["dtype"],
                        t["shape"],
                        "symm",
                    )
                else:
                    add_region(r, b, b + h["buffer_size"], f"symm_r{r}", kind="symm")
                add_region(
                    r,
                    sp,
                    sp + h["signal_pad_size"],
                    f"sig_r{r}",
                    "u32",
                    [h["signal_pad_size"] // 4],
                    "signal",
                )
            else:
                pre = [a for a in m["pre"] if s <= a < e]
                add_region(
                    r,
                    s,
                    s
                    + (
                        len(open(f"{WORK}/r{r}/pre/{pre[0]:x}.bin", "rb").read())
                        if pre
                        else e - s
                    ),
                    f"vmm_r{r}_{len(regions)}",
                    kind="vmm",
                )

    # Module data globals the launched kernels' banks name, captured by the worker: one region
    # each (written with `symbol`), scanned for pointers like a parameter's target.
    for r, m in enumerate(ranks):
        # Globals of setup launches (outside the captured window) are not the case's.
        launched = {x["name"] for x in m["records"] if "name" in x}
        for x in m.get("globals", []):
            if x["kernel"] not in launched or (
                kfilter and not kfilter.search(x["kernel"])
            ):
                continue
            if "ptr" not in x:
                warnings.append(
                    f"rank {r}: global {x['symbol']} of {x['kernel'][:50]}: not captured (error {x.get('error')}); starts from its ELF value"
                )
                continue
            if any(g["dev"] == r and g["start"] == x["ptr"] for g in regions):
                continue
            nm = re.sub(r"[^A-Za-z0-9_]", "_", x["symbol"])
            add_region(r, x["ptr"], x["ptr"] + x["size"], f"g_{nm}_r{r}", kind="global")
            regions[-1]["symbol"] = x["symbol"]
            regions[-1]["kernel"] = x["kernel"]
    # IBGDA (docs/nic.md): DOCA queue memory not in any region yet, and one synthetic 8-byte
    # buffer per doorbell register (the UAR page is MMIO; its address only names it).
    for r, m in enumerate(ranks):
        for qi, q in enumerate(m.get("nic", [])):
            for a, z in q["regions"]:
                if not any(
                    g["dev"] == r and g["start"] <= a < g["end"] for g in regions
                ):
                    add_region(r, a, a + z, f"nicmem{len(regions)}_r{r}", kind="nic")
            if q.get("extra"):
                continue
            db = q["qp_fields"]["sq_db"]
            if not any(g["dev"] == r and g["start"] <= db < g["end"] for g in regions):
                add_region(r, db, db + 8, f"uar_r{r}_{qi}", "u64", [1], "uar")

    # Bounding boxes per device (rebuilt when regions are added): most words a pointer-table
    # scan looks at are data far outside every region.
    bbox = {}

    # Bounding boxes per device (rebuilt when regions are added): most words a pointer-table
    # scan looks at are data far outside every region.
    bbox = {}

    def region_at(dev, a):
        if bbox.get("n") != len(regions):
            bbox.clear()
            bbox["n"] = len(regions)
            for g in regions:
                lo, hi = bbox.get(g["dev"], (g["start"], g["end"]))
                bbox[g["dev"]] = (min(lo, g["start"]), max(hi, g["end"]))
        lo, hi = bbox.get(dev, (1, 0))
        if not lo <= a <= hi:
            return None
        for i, g in enumerate(regions):
            if g["dev"] == dev and g["start"] <= a < g["end"]:
                return i, a - g["start"]
        for i, g in enumerate(regions):
            if g["dev"] == dev and a == g["end"]:
                return i, a - g["start"]
        return None

    maps_box = {}

    xhost_refs = (
        set()
    )  # (rank, owner): a pointer of rank r into memory owned on another host

    class VRef(int):
        """A region index reached through a further mapping (S-VA-ALIAS): `view` is (rank,
        physical id, the mapping's address of the allocation's byte 0). Behaves as the
        index everywhere; `bname` names the mapping."""

        def __new__(cls, i, view):
            o = int.__new__(cls, i)
            o.view = view
            return o

    def resolve(r, v):
        """(region index, offset) | ("multicast", ...) | None, for address v seen by rank r.
        Through a mapping other than the owner's own one (a peer's import, a second cuMemMap)
        the index is a `VRef` naming that mapping."""
        if not v:
            return None
        if r not in maps_box:
            maps_box[r] = (
                min((m[0] for m in maps[r]), default=1),
                max((m[1] for m in maps[r]), default=0),
            )
        if not maps_box[r][0] <= v < maps_box[r][1]:
            return region_at(r, v)
        for s, e, pid, off in maps[r]:
            if s <= v < e:
                p = phys.get(pid)
                if p is None or p["owner"] is None:
                    return None
                if p["kind"] == "multicast":
                    return ("multicast", pid, v - s + off)
                oa = owner_addr(pid, off + (v - s))
                if oa is None:
                    return None
                if host[p["owner"]] != host[r]:
                    xhost_refs.add((r, p["owner"]))
                loc = region_at(p["owner"], oa[0])
                if loc and (r != p["owner"] or s - off != oa[1]):
                    return (VRef(loc[0], (r, pid, s - off)), loc[1])
                return loc
        return region_at(r, v)

    def near(v):
        return any(abs(v - g["start"]) < (1 << 30) for g in regions) or any(
            s - (1 << 30) <= v < e + (1 << 30) for mm in maps for s, e, _, _ in mm
        )

    refs = {}
    mc_used = set()  # multicast phys ids kernels reference
    tmap_dt = {}  # region index -> element types of the tensor maps over it

    launch_regions = {}  # id(kept launch record) -> region indices its parameters point into
    launch_mc = {}  # id(kept launch record) -> multicast phys ids its parameters point into
    cur_launch = [None]

    def note(ri, off):
        refs.setdefault(int(ri), set()).add(off)
        if cur_launch[0] is not None:
            launch_regions.setdefault(cur_launch[0], set()).add(int(ri))

    if identity_errors:
        sys.exit(
            "capture_dist: physical identity not determined (S-VA-ALIAS):\n  "
            + "\n  ".join(sorted(set(identity_errors)))
        )

    # ------------------------------------------------ launches and parameters
    launches = []  # (rank, rec, params)
    streams = {}
    rank_order = {}  # rank -> its launches' order (_order.LaunchOrder)
    import importlib.util as _ilu  # (merge() imports importlib itself further down)

    _spec = _ilu.spec_from_file_location("ptxv_order", os.path.join(ROOT, "python/ptxv/_order.py"))
    _order = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_order)
    for r, m in enumerate(ranks):
        recs = [x for x in m["records"] if "name" in x]
        if kfilter:
            recs = [x for x in recs if kfilter.search(x["name"])]
        streams[r] = sorted({x["stream"] for x in recs})
        # Launch order (E-LAUNCH-ORDER, python/ptxv/_order.py): the order the rank's streams and
        # stream order calls (event records / waits, graph launches) enforce between its kept
        # launches; launches on streams no event orders (DeepEP's communication stream) stay
        # concurrent. Mapped into the case's interleaving below. The case puts a rank's launches
        # on the rank's GPU: kept launches that name several devices give an unknown order.
        rpos = {id(x): i for i, x in enumerate(m["records"])}
        rank_order[r] = _order.launch_order(m["records"], [rpos[id(x)] for x in recs])
        for x in recs:
            split_packed(r, x)
            cur_launch[0] = id(x)
            params = []
            for pi, p in enumerate(x["params"]):
                raw = bytes.fromhex(p["hex"])
                if len(raw) == 8:
                    v = u64(raw, 0)
                    loc = resolve(r, v)
                    if loc and loc[0] == "multicast":
                        mc_used.add(loc[1])
                        launch_mc.setdefault(id(x), set()).add(loc[1])
                        params.append(("mcptr", loc[1], loc[2]))
                    elif loc:
                        note(*loc)
                        params.append(("ptr",) + loc)
                    else:
                        if v and near(v):
                            unresolved.append(
                                {
                                    "rank": r,
                                    "launch": x["name"][:60],
                                    "param": pi,
                                    "value": hex(v),
                                }
                            )
                        params.append(("u64", v))
                elif len(raw) == 4:
                    params.append(("u32", struct.unpack("<I", raw)[0]))
                else:
                    fields = []
                    off = 0
                    while off + 8 <= len(raw):
                        v = u64(raw, off)
                        loc = resolve(r, v)
                        # A 128-byte CUtensorMap (64-byte aligned) over a known region
                        # (CuTeDSL's grid-constant maps), as tools/capture.py.
                        g_ = (
                            decode_tmap(raw[off : off + 128])
                            if loc
                            and loc[0] != "multicast"
                            and off % 64 == 0
                            and off + 128 <= len(raw)
                            and raw[off + 8 : off + 16] != bytes(8)
                            else None
                        )
                        if g_:
                            note(*loc)
                            tmap_dt.setdefault(loc[0], set()).add(g_["dtype"])
                            n_ = 64 if raw[off + 8] & 2 else 128
                            fields.append(
                                (off, loc[0], loc[1], (g_, raw[off : off + n_]))
                            )
                            off += n_
                            continue
                        if loc and loc[0] == "multicast":
                            mc_used.add(loc[1])
                            launch_mc.setdefault(id(x), set()).add(loc[1])
                            fields.append((off, ("mc", loc[1]), loc[2]))
                        elif loc:
                            note(*loc)
                            fields.append((off,) + loc)
                        elif v and near(v):
                            unresolved.append(
                                {
                                    "rank": r,
                                    "launch": x["name"][:60],
                                    "param": pi,
                                    "offset": off,
                                    "value": hex(v),
                                }
                            )
                        off += 8
                    if (
                        len(raw) == 128
                        and len(fields) == 1
                        and fields[0][0] == 0
                        and len(fields[0]) == 4
                    ):
                        params.append(("tmap",) + fields[0][1:])
                    else:
                        params.append(("bytes", raw, fields))
            launches.append((r, x, params))
            cur_launch[0] = None
    # Vacuous-input guard (as python/ptxv/_casewriter.py): a launch dropped by --kernels that
    # points into a buffer the case's launches use, before the rank's last kept launch, may
    # write bytes that the case's initial contents (the snapshot before run()) miss.
    if kfilter:
        used = set(refs)
        for r, m in enumerate(ranks):
            allrec = [x for x in m["records"] if "name" in x]
            kept_pos = [i for i, x in enumerate(allrec) if kfilter.search(x["name"])]
            if not kept_pos:
                continue
            for x in allrec[: kept_pos[-1]]:
                if kfilter.search(x["name"]):
                    continue
                hit = set()
                for p in x.get("params", []):
                    raw = bytes.fromhex(p["hex"])
                    for o in range(0, len(raw) - 7, 8):
                        loc = resolve(r, u64(raw, o))
                        if loc and isinstance(loc[0], int) and loc[0] in used:
                            g = regions[loc[0]]
                            hit.add(f"GPU {g['dev']} {g['start']:#x}")
                if hit:
                    warnings.append(
                        f"rank {r}: launch {x['name'][:60]}, dropped by --kernels, points into "
                        f"buffer(s) {sorted(hit)} of the case after the snapshot and before a kept "
                        "launch: bytes it writes are missing from the initial contents"
                    )
    if not launches:
        sys.exit("capture_dist: no kernel launches recorded")
    for (
        i,
        dts,
    ) in tmap_dt.items():  # a byte region under a tensor map takes its element type
        g, td = regions[i], sorted(dts)[0]
        if len(dts) > 1:
            warnings.append(
                f"{g['name']}: tensor maps of several element types {sorted(dts)}"
            )
        size = g["end"] - g["start"]
        if ESIZE.get(g["dtype"], 1) != ESIZE[td]:
            if g["dtype"] == "u8" and size % ESIZE[td] == 0:
                g["dtype"], g["shape"] = td, [size // ESIZE[td]]
            else:
                warnings.append(
                    f"{g['name']}: dtype {g['dtype']} but a tensor map reads {td}"
                )
    for i, g in enumerate(regions):
        if g["kind"] == "global":
            note(i, 0)

    # ------------------------------------------------ initial and final bytes
    # Snapshot extents from the file sizes: only overlapping snapshots are read (a rank can
    # have over a thousand of them).
    pre_len = [
        {a: os.path.getsize(f"{WORK}/r{r}/pre/{a:x}.bin") for a in m["pre"]}
        for r, m in enumerate(ranks)
    ]
    for _i, g in enumerate(regions):
        r = g["dev"]
        size = g["end"] - g["start"]
        data, found = bytearray(size), False
        for a in ranks[r]["pre"]:
            if a >= g["end"] or a + pre_len[r][a] <= g["start"]:
                continue
            b = open(f"{WORK}/r{r}/pre/{a:x}.bin", "rb").read()
            lo, hi = max(a, g["start"]), min(a + len(b), g["end"])
            if lo < hi:
                data[lo - g["start"] : hi - g["start"]] = b[lo - a : hi - a]
                found = True
        g["data"] = bytes(data) if found and any(data) else None

    # Device copies / fills a rank made before its first kernel (an op's clone(), a copy_
    # into the symmetric buffer) fold into the initial bytes. A case cannot express a copy or
    # fill between kernels (MISS 173): one into a region that a later kept launch of the rank
    # reads refuses the case, as does a host upload whose bytes were not recorded into a region
    # a kept launch reads, or one between kernels into a region an earlier kept launch used
    # (folding it would give that launch bytes it did not see).
    refuse = []
    ptr_targets = {}  # region index -> regions its 8-byte words point into (pointer tables)

    def mc_regions(pid):
        """The regions bound to multicast object pid (its members' owner-side bytes)."""
        out = set()
        for _br, mpid, moff, _mcoff, size in mc_bind.get(pid, []):
            p = phys.get(mpid)
            oa = owner_addr(mpid, moff, size) if p and p["owner"] is not None else None
            if oa is None:
                continue
            lo, hi = oa[0], oa[0] + size
            for i, g in enumerate(regions):
                if g["dev"] == p["owner"] and max(lo, g["start"]) < min(hi, g["end"]):
                    out.add(i)
        return out

    def targets(i):
        if i not in ptr_targets:
            g, out = regions[i], set()
            if g["data"] and g["kind"] not in ("symm", "signal"):
                for off in range(0, len(g["data"]) - 7, 8):
                    loc = resolve(g["dev"], u64(g["data"], off))
                    if loc and loc[0] == "multicast":
                        out |= mc_regions(loc[1])
                    elif loc:
                        out.add(int(loc[0]))
            ptr_targets[i] = out
        return ptr_targets[i]

    def reaches(y, ri):
        """Whether kept launch y points into region ri: through a parameter, a multicast
        address bound to it, or pointer words in a region it reaches (the pointer tables
        below)."""
        seen, todo = set(), list(launch_regions.get(id(y), ()))
        for pid in launch_mc.get(id(y), ()):
            todo.extend(mc_regions(pid))
        while todo:
            i = todo.pop()
            if i == ri:
                return True
            if i not in seen:
                seen.add(i)
                todo.extend(targets(i))
        return False

    for r, m in enumerate(ranks):
        seen_kernel = False
        kept_seq = [
            x for x in m["records"] if "name" in x and (not kfilter or kfilter.search(x["name"]))
        ]
        done = []  # kept launches of the rank before the record at hand
        for x in m["records"]:
            if "name" in x and (not kfilter or kfilter.search(x["name"])):
                seen_kernel = True
                done.append(x)
                continue
            if "op" not in x or x.get("kind", 0) not in (0, 1, 3, 4):
                continue  # device -> host copies touch no case buffer
            loc = resolve(r, x["dst"])
            ri = int(loc[0]) if loc and loc[0] != "multicast" else None
            later = [
                y["name"][:60]
                for y in kept_seq[len(done) :]
                if ri is not None and reaches(y, ri)
            ]
            earlier = [
                y["name"][:60] for y in done if ri is not None and reaches(y, ri)
            ]
            if x.get("kind") == 1 and "data" not in x:
                if later:
                    refuse.append(
                        f"rank {r}: host upload of {x['bytes']} bytes to {x['dst']:#x} (bytes not "
                        f"recorded) into a region kept launch {later[0]} reads"
                    )
                continue
            if x.get("kind") == 1 and seen_kernel:
                if earlier:
                    refuse.append(
                        f"rank {r}: host upload of {x['bytes']} bytes to {x['dst']:#x} between "
                        f"kernels into a region that kept launch {earlier[-1]} used before it"
                    )
                    continue
                # A host-computed table uploaded between kernels (MoE routing order) into memory
                # no earlier kept launch of the rank used: its bytes are the destination's initial
                # contents.
                notes.append(
                    f"rank {r}: host upload of {x['bytes']} bytes to {x['dst']:#x} between kernels folded into initial contents"
                )
            elif seen_kernel:
                if later:
                    refuse.append(
                        f"rank {r}: {x['op']} of {x['bytes']} bytes to {x['dst']:#x} between "
                        f"kernels into a region that kept launch {later[0]} reads after it"
                    )
                else:
                    warnings.append(
                        f"rank {r}: {x['op']} of {x['bytes']} bytes to {x['dst']:#x} between "
                        "kernels: not in the case (no later kept launch reads it)"
                    )
                continue
            if not loc or loc[0] == "multicast":
                warnings.append(
                    f"rank {r}: {x['op']} to {x['dst']:#x} (no buffer) ignored"
                )
                continue
            i, off = loc
            g = regions[i]
            n = min(x["bytes"], g["end"] - g["start"] - off)
            d = bytearray(g["data"] or bytes(g["end"] - g["start"]))
            if x["op"] == "memset":
                d[off : off + n] = bytes([x["value"] & 255]) * n
            elif x["op"] == "memset32":
                d[off : off + n] = struct.pack("<I", x["value"] & 0xFFFFFFFF) * (n // 4)
            elif x.get("kind") == 1:
                d[off : off + n] = bytes.fromhex(x["data"])[:n]
            else:
                sl = resolve(r, x["src"])
                src = None
                if sl and sl[0] != "multicast":
                    sg = regions[sl[0]]
                    if sg["data"] is not None or True:
                        sd = sg["data"] or bytes(sg["end"] - sg["start"])
                        src = sd[sl[1] : sl[1] + n]
                if src is None or len(src) != n:
                    if later:
                        refuse.append(
                            f"rank {r}: memcpy into {g['name']} from {x['src']:#x} (unknown "
                            f"contents), which kept launch {later[0]} reads"
                        )
                    else:
                        warnings.append(
                            f"rank {r}: memcpy into {g['name']} from {x['src']:#x} (unknown contents) ignored"
                        )
                    continue
                d[off : off + n] = src
            g["data"] = bytes(d)

    if refuse:
        for why in refuse:
            print("capture_dist: cannot write the case:", why, file=sys.stderr)
        sys.exit(
            "capture_dist: refused (MISS 173): a copy, fill or upload inside the capture writes "
            "memory a kept launch reads, and a case cannot express it"
        )

    # IBGDA handles, QP structs and queues are scanned for pointers like any table.
    for r, m in enumerate(ranks):
        for q in m.get("nic", []):
            if q.get("extra"):
                continue
            for a in (q["addr"], q["qp"]):
                loc = region_at(r, a)
                if loc:
                    note(*loc)

    # Pointer tables: words inside referenced regions that resolve (peer pointer arrays).
    todo = sorted(refs)
    done = set()
    while todo:
        i = todo.pop()
        if i in done:
            continue
        done.add(i)
        g = regions[i]
        if not g["data"] or g["kind"] in ("symm", "signal"):
            continue
        g["ptrs"] = []
        for off in range(0, len(g["data"]) - 7, 8):
            v = u64(g["data"], off)
            loc = resolve(g["dev"], v)
            if loc and loc[0] == "multicast":
                mc_used.add(loc[1])
                g["ptrs"].append((off, ("mc", loc[1]), loc[2]))
            elif loc:
                g["ptrs"].append((off,) + loc)
                note(*loc)
                todo.append(loc[0])
        # A pointer table is not data: the simulator fills the words in.
        if g["ptrs"] and len(g["data"]) % 8 == 0:
            g["dtype"], g["shape"] = "u64", [len(g["data"]) // 8]

    # IBGDA handles: their remote addresses are VAs of the peer rank (they may resolve to an
    # unrelated buffer of this rank by accident): point them at the peer's buffer.
    nic_qps, nic_mrs = [], []
    nic_peer = nic_pairing(ranks, warnings, notes)
    nic_pairs = []
    for r, m in enumerate(ranks):
        for qi, q in enumerate(m.get("nic", [])):
            if q.get("extra"):
                continue
            f = q["qp_fields"]
            q["peer"], how = nic_peer.get((r, qi), (q["peer"], "nic()"))
            nic_pairs.append(
                {
                    "rank": r,
                    "host": host[r],
                    "qpn": f["sq_num"],
                    "peer": q["peer"],
                    "peer_host": host[q["peer"]],
                    "by": how,
                }
            )
            hl = region_at(r, q["addr"])
            loc = {
                k: region_at(r, f[k]) for k in ("sq_wqe_daddr", "cqe_daddr", "sq_db")
            }
            if not hl or not all(loc.values()):
                warnings.append(
                    f"rank {r}: IBGDA handle {qi} or its queues are in no buffer"
                )
                continue
            if f["sq_rsvd_index"] != f["sq_wqe_pi"]:
                warnings.append(
                    f"rank {r}: IBGDA QP {f['sq_num']:#x} not quiescent at the snapshot (reserved {f['sq_rsvd_index']}, posted {f['sq_wqe_pi']}): the case starts the NIC at the posted index"
                )
            elif f["cqe_ci"] != f["sq_wqe_pi"]:
                notes.append(
                    f"rank {r}: IBGDA QP {f['sq_num']:#x}: {f['sq_wqe_pi'] - f['cqe_ci']} posted WQE(s) not polled before the snapshot, taken as completed (their CQEs are in the snapshot)"
                )
            hi, hoff = hl
            g = regions[hi]
            g.setdefault("ptrs", [])
            peer = q["peer"]
            for fo, key in ((24, "remote_base"), (40, "remote_token")):
                off = hoff + q["tail"] + fo
                tgt = region_at(peer, q[key])
                g["ptrs"] = [p for p in g["ptrs"] if p[0] != off]
                if tgt:
                    g["ptrs"].append((off,) + tgt)
                    note(*tgt)
                else:
                    unresolved.append(
                        {
                            "rank": r,
                            "buffer": g["name"],
                            "offset": off,
                            "value": hex(q[key]),
                            "why": f"IBGDA {key} of peer {peer} in no buffer",
                        }
                    )
            g["ptrs"].sort()
            for k, dev, a in (
                ("rkey", peer, q["remote_base"]),
                ("token_rkey", peer, q["remote_token"]),
                ("lkey", r, q["local_base"]),
                ("sink_lkey", r, q["sink"]),
            ):
                tgt = region_at(dev, a)
                if tgt:
                    note(*tgt)
                    nic_mrs.append((q[k], tgt[0]))
            for k in loc.values():
                note(*k)
            nic_qps.append(
                dict(
                    name=f"qp_r{r}_{qi}",
                    device=r,
                    peer=peer,
                    qpn=f["sq_num"],
                    sq=loc["sq_wqe_daddr"],
                    sq_wqes=f["sq_wqe_num"],
                    cq=loc["cqe_daddr"],
                    cq_cqes=f["cqe_num"],
                    db=loc["sq_db"],
                    sq_ci=f["sq_wqe_pi"],
                    cq_pi=f["sq_wqe_pi"] + f["cqe_rsvd"],
                    handler=f["nic_handler"],
                )
            )
    domains = ranks[0].get("domains", [])
    # Outputs: regions holding an output tensor of their rank.
    out_regions = {}  # output name -> {rank: region index}
    for r, m in enumerate(ranks):
        for n, t in m["outputs"]:
            loc = region_at(r, t["data_ptr"])
            if not loc:
                warnings.append(f"rank {r}: output {n} is in no region")
                continue
            i, off = loc
            g = regions[i]
            g["expect"] = open(f"{WORK}/r{r}/post/{n}.bin", "rb").read()
            g["expect_offset"] = off
            g["out_name"] = n
            out_regions.setdefault(n, {})[r] = i

    # Multicast objects the kernels reference: each bind cut at the regions of its owner.
    mc_objs = []  # (pid, name, size, [(region index, region offset, mc offset, size)])
    for k, pid in enumerate(sorted(mc_used)):
        cuts = []
        for br, mpid, moff, mcoff, size in mc_bind.get(pid, []):
            p = phys.get(mpid)
            oa = owner_addr(mpid, moff, size) if p and p["owner"] is not None else None
            if oa is None:
                # The bound bytes reach no buffer: the model would drop that rank's member of
                # every multimem store and reduction.
                identity_errors.append(
                    f"multicast mc{k}: rank {br}'s bound range [{moff:#x}, +{size:#x}) has no owner mapping covering it"
                )
                continue
            lo_va = oa[0]
            for i, g in enumerate(regions):
                if g["dev"] != p["owner"]:
                    continue
                lo, hi = max(lo_va, g["start"]), min(lo_va + size, g["end"])
                if lo < hi:
                    cuts.append((i, lo - g["start"], mcoff + (lo - lo_va), hi - lo))
        if not cuts:
            warnings.append(f"multicast mc{k}: no bound bytes in any buffer")
        mc_objs.append((pid, f"mc{k}", phys.get(pid, {}).get("size") or 0, cuts))
    mc_name = {pid: n for pid, n, _, _ in mc_objs}
    if identity_errors:
        sys.exit(
            "capture_dist: physical identity not determined (S-VA-ALIAS):\n  "
            + "\n  ".join(sorted(set(identity_errors)))
        )

    # Keep only regions something references (plus outputs and multicast backing).
    keep = set(refs) | {i for d in out_regions.values() for i in d.values()}
    keep |= {c[0] for _, _, _, cuts in mc_objs for c in cuts}
    for i in list(keep):
        for p in regions[i].get("ptrs", []):
            if not isinstance(p[1], tuple):
                keep.add(int(p[1]))
    kept = sorted(keep)
    names_taken = set()
    for i in kept:
        g = regions[i]
        base, k = g["name"], 1
        while g["name"] in names_taken:
            g["name"] = f"{base}_{k}"
            k += 1
        names_taken.add(g["name"])

    va_views = {}  # (region index, mapping) -> view name (S-VA-ALIAS)

    def bname(i):
        if isinstance(i, tuple):  # ("mc", multicast phys id)
            return mc_name[i[1]]
        view = getattr(i, "view", None)
        if view is not None:
            key = (int(i), view)
            if key not in va_views:
                n = f"{regions[i]['name']}_m{view[0]}"
                k = 1
                while n in names_taken or n in va_views.values():
                    n = f"{regions[i]['name']}_m{view[0]}_{k}"
                    k += 1
                va_views[key] = n
            return va_views[key]
        return regions[i]["name"]

    # ------------------------------------------------ cubins
    images = []
    for r in range(R):
        images += glob.glob(f"{WORK}/r{r}/cupti/*.cubin") + glob.glob(
            f"{WORK}/r{r}/images/*.bin"
        )
    # Per (rank, kernel): JIT frameworks compile per-rank constants into same-named kernels
    # (CuTeDSL's rank_id in dist_gemm), so each rank's launch takes its own rank's image.
    # Keyed by (rank, kernel, parameter count): a JIT framework can also emit same-named
    # specializations with different signatures (quack GemmDefaultSm100 with 24 and 26
    # parameters in one run); each launch takes an image whose KPARAM_INFO matches its count.
    knames = []
    for r, x, _ in launches:
        key = (r, x["name"], len(x["params"]))
        if key not in knames:
            knames.append(key)
    cubin_of = {}
    for r, n, np_ in knames:
        mine = [p for p in images if f"/r{r}/" in p]
        c = (
            [p for p in mine if "/cupti/" in p and elf_defines(p, n)]
            or [p for p in mine if elf_defines(p, n)]
            or [p for p in images if "/cupti/" in p and elf_defines(p, n)]
            or [p for p in images if elf_defines(p, n)]
        )
        if not c:
            sys.exit(f"capture_dist: no captured cubin defines {n}")
        fit = [p for p in c if len(kparam_layout(p, n) or []) == np_]
        if not fit and len({len(kparam_layout(p, n) or []) for p in c}) > 1:
            warnings.append(f"rank {r}: no image of {n[:60]} has {np_} parameters")
        cubin_of[(r, n, np_)] = min(fit or c, key=os.path.getsize)
    out = os.path.abspath(A.out)
    os.makedirs(out, exist_ok=True)
    for sub in ("data", "expected"):
        shutil.rmtree(f"{out}/{sub}", ignore_errors=True)
        os.makedirs(f"{out}/{sub}")
    for f in glob.glob(f"{out}/k*.cubin"):
        os.remove(f)
    by_hash, cubin_file = {}, {}
    for n in knames:
        data = open(cubin_of[n], "rb").read()
        h = hashlib.sha1(data).hexdigest()
        if h not in by_hash:
            by_hash[h] = f"k{len(by_hash)}.cubin"
            keepk = {
                k[1]
                for k in knames
                if hashlib.sha1(open(cubin_of[k], "rb").read()).hexdigest() == h
            }
            dst = f"{out}/{by_hash[h]}"
            open(dst, "wb").write(data if A.no_prune else prune(data, keepk))
            if not A.no_prune:
                # Shrink to the launched kernels (and what they call): library modules are
                # MBs, and DeepEP's carries NVSHMEM relocations nvdisasm cannot decode.
                r = subprocess.run(
                    [
                        sys.executable,
                        os.path.join(ROOT, "tools/cubin_slim.py"),
                        dst,
                        dst,
                    ]
                    + sorted(keepk),
                    capture_output=True,
                    text=True,
                )
                if r.returncode:
                    warnings.append(
                        f"{by_hash[h]}: cubin_slim failed ({r.stderr.strip()[-200:]}); NOP-pruned instead"
                    )
                    open(dst, "wb").write(prune(data, keepk))
        cubin_file[n] = by_hash[h]

    # ------------------------------------------------ case.toml
    q = json.dumps

    def ck(r, x):
        return cubin_file[(r, x["name"], len(x["params"]))]

    def toml_tmap(prefix, ri, ro, gr, extra=""):
        g_, raw_ = gr
        return (
            f'{prefix}{extra}kind = "tensormap"\nbuffer = {q(bname(ri))}\nbyte_offset = {ro}\n'
            f"box = {g_['box']}\ndims = {g_['dims']}\nstrides = {g_['strides']}\n"
            f'swizzle = "{g_["swizzle"]}"\nraw = "{raw_.hex()}"\n'
            + ("elem4 = true\n" if g_.get("elem4") else "")
        )

    def toml_params(params, key):
        s = []
        for p in params:
            if p[0] == "ptr":
                s.append(
                    f'[[{key}]]\nkind = "ptr"\nbuffer = {q(bname(p[1]))}\nbyte_offset = {p[2]}\n'
                )
            elif p[0] == "mcptr":
                s.append(
                    f'[[{key}]]\nkind = "ptr"\nbuffer = {q(mc_name[p[1]])}\nbyte_offset = {p[2]}\n'
                )
            elif p[0] == "tmap":
                s.append(toml_tmap(f"[[{key}]]\n", p[1], p[2], p[3]))
            elif p[0] == "u32":
                s.append(f'[[{key}]]\nkind = "u32"\nvalue = {p[1]}\n')
            elif p[0] == "u64":
                s.append(f'[[{key}]]\nkind = "u64"\nvalue = {p[1]}\n')
            else:
                _, raw, fields = p
                s.append(f'[[{key}]]\nkind = "bytes"\nhex = "{raw.hex()}"\n')
                for off, bi, bo, *tm in fields:
                    if tm:
                        s.append(
                            toml_tmap(
                                f"[[{key}.fields]]\n",
                                bi,
                                bo,
                                tm[0],
                                f"offset = {off}\n",
                            )
                        )
                        continue
                    s.append(
                        f'[[{key}.fields]]\noffset = {off}\nkind = "ptr"\nbuffer = {q(bname(bi))}\nbyte_offset = {bo}\n'
                    )
        return s

    # Order: every rank's launches in its stream order, interleaved by host submission time
    # (only same-device order is meaningful to the model).
    order = sorted(range(len(launches)), key=lambda j: launches[j][1].get("t", 0))
    # The first launch must be on device 0 (the manifest's top level).
    first = next(j for j in order if launches[j][0] == 0)
    order.remove(first)
    order = [first] + order
    L = [launches[j] for j in order]
    # Each rank's launch order, placed into this interleaving (a rank is one GPU of the case,
    # and its launches keep their own order in it).
    gpos = {j: g for g, j in enumerate(order)}
    parts = []
    for r in sorted(rank_order):
        idx = [gpos[j] for j in range(len(launches)) if launches[j][0] == r]
        parts.append((rank_order[r], idx))
        notes += [f"rank {r}: {n}" for n in rank_order[r].notes]
    lorder = _order.place(parts, len(L))
    if len(L) > 1 and not lorder.recorded:
        warnings.append(
            f"launch order not recorded ({lorder.why}): launch-order exclusions do not apply "
            "to this case (E-LAUNCH-ORDER)"
        )
    # Replicas: the outputs the target module names in REPLICAS (an all-reduce result), when
    # every rank has them with the same size. A MoE layer's outputs are per-rank tokens: none.
    rep_names = set(ranks[0].get("replicas", []))
    replicas = [
        [bname(d[r]) for r in sorted(d)]
        for n, d in out_regions.items()
        if n in rep_names
        and len(d) == R
        and len({regions[i]["end"] - regions[i]["start"] for i in d.values()}) == 1
        and len({len(regions[i]["expect"]) for i in d.values()}) == 1
    ]
    name = A.name or os.path.basename(out.rstrip("/"))
    r0 = L[0][1]
    kern_lines = "\n".join(
        f"  rank {r}: {x['name'][:100]} grid {x['grid']} block {x['block']}"
        for r, x, _ in L
    )
    cmdline = " ".join(
        a
        for a in sys.argv[1:]
        if a != "--worker" and not a.startswith("--work") and a != WORK
    )
    desc = f"""
Captured by tools/capture_dist.py (torch {ranks[0]["torch"]}, torchrun --nproc-per-node {A.nproc}{f" on {len(hosts)} hosts" if len(hosts) > 1 else ""}):
run() of {os.path.basename(A.file)}. {len(L)} launch(es) on {R} GPUs, binary only:
{kern_lines}
Regenerate: CUDA_VISIBLE_DEVICES=0,1,3 $PTXV_PYTHON tools/capture_dist.py {cmdline}"""
    desc = desc.replace("\\", "\\\\").replace('"', '\\"')
    outputs = [bname(i) for i in kept if "expect" in regions[i]]
    # The architecture the ranks ran on (the case default is sm_103a); PTXV_ARCH overrides.
    arch = os.environ.get("PTXV_ARCH") or ranks[0].get("arch")
    toml = [
        f'name = {q(name)}\ndescription = """{desc}"""\ncubin = "{ck(L[0][0], r0)}"\n'
        + (f'arch = "{arch}"\n' if arch else "")
        + f"entry = {q(r0['name'])}\ngrid = {r0['grid']}\nblock = {r0['block']}\n"
        f"dynamic_smem = {r0['smem']}\noutputs = {q(outputs)}\nreplicas = {q(replicas)}\n"
        # S-CGA-SHAPE: a preferred cluster needs the explicit regular one, even 1 x 1 x 1.
        + (
            f"cluster = {r0['cluster']}\n"
            if r0["cluster"] != [1, 1, 1] or r0.get("preferred_cluster")
            else ""
        )
        + (
            f"preferred_cluster = {r0['preferred_cluster']}\n"
            if r0.get("preferred_cluster")
            else ""
        )
        + (f'exit_fence = "{A.exit_fence}"\n' if A.exit_fence else "")
        + (_order.toml_top(lorder) if len(L) > 1 else "")
    ]
    kept_phys = {
        pid
        for pid, va in owner_va.items()
        for i in kept
        if regions[i]["dev"] == phys[pid]["owner"]
        and va <= regions[i]["start"] < va + phys[pid]["size"]
    }
    if len(hosts) > 1:
        # Ranks on different hosts: one NVLink domain per host, unless a kernel parameter or
        # pointer-table word of one rank points through a mapping into memory owned on another
        # host (a fabric handle: the hosts share an NVLink fabric, e.g. symmetric memory across
        # GB300 trays); then one domain per fabric clique (nvidia-smi). A mapping no pointer
        # uses (NCCL's MNNVL buffers) does not count.
        def fab(r):
            f = ranks[r].get("fabric") or {}
            return (
                (f.get("ClusterUUID"), f.get("CliqueId"))
                if f.get("ClusterUUID")
                else host[r]
            )

        cross = sorted(xhost_refs)
        if cross:
            cl = list(dict.fromkeys(fab(r) for r in range(R)))
            phys_dom = [cl.index(fab(r)) for r in range(R)]
            notes.append(
                f"{len(cross)} (rank, owner) pair(s) point into memory across hosts, e.g. {cross[0]}: "
                f"domains follow the NVLink fabric cliques {phys_dom}"
            )
        else:
            phys_dom = [hosts.index(h) for h in host]
        if domains and domains != phys_dom:
            warnings.append(
                f"the target's DOMAINS {domains} differ from the hosts' {phys_dom}; hosts' used"
            )
        domains = phys_dom

    if domains:
        toml.append(f"domains = {q(domains)}\n")
    # S-VA-ALIAS: every rank is its own process (torchrun), so its own virtual address space.
    if R > 1:
        toml.append(f"spaces = {list(range(R))}\n")
    for x in nic_qps:
        toml.append(
            f"[[nic_qps]]\nname = {q(x['name'])}\ndevice = {x['device']}\npeer = {x['peer']}\n"
            f"qpn = {x['qpn']}\nsq = {q(bname(x['sq'][0]))}\nsq_offset = {x['sq'][1]}\n"
            f"sq_wqes = {x['sq_wqes']}\ncq = {q(bname(x['cq'][0]))}\ncq_offset = {x['cq'][1]}\n"
            f"cq_cqes = {x['cq_cqes']}\ndb = {q(bname(x['db'][0]))}\ndb_offset = {x['db'][1]}\n"
            f"sq_ci = {x['sq_ci']}\ncq_pi = {x['cq_pi']}\n"
        )
        if x["handler"] not in (2, 10):  # GPU_SM_DB, GPU_SM_NO_DBR
            warnings.append(
                f"{x['name']}: DOCA nic_handler {x['handler']} (CPU proxy / BlueFlame: not modeled)"
            )
    for key, bi in sorted(set(nic_mrs)):
        toml.append(f"[[nic_mrs]]\nkey = {key}\nbuffer = {q(bname(bi))}\n")
    for i in kept:
        g = regions[i]
        s = f'[[buffers]]\nname = {q(g["name"])}\ndtype = "{g["dtype"]}"\nshape = {g["shape"]}\ndevice = {g["dev"]}\n'
        if g["data"] and not g.get("ptrs"):
            open(f"{out}/data/{g['name']}.bin", "wb").write(g["data"])
            s += f'init = "file"\nfile = "data/{g["name"]}.bin"\n'
        elif g.get("ptrs"):
            # Only the pointers matter; other words are whatever was there (kept as file).
            d = bytearray(g["data"])
            for off, _, _ in g["ptrs"]:
                d[off : off + 8] = bytes(8)
            if any(d):
                open(f"{out}/data/{g['name']}.bin", "wb").write(bytes(d))
                s += f'init = "file"\nfile = "data/{g["name"]}.bin"\n'
            else:
                s += 'init = "zeros"\n'
        else:
            s += 'init = "zeros"\n'
        if g["kind"] == "global":
            s += f"symbol = {q(g['symbol'])}\n"
            gk = next(
                (v for k, v in cubin_file.items() if k[:2] == (g["dev"], g["kernel"])),
                None,
            ) or next((v for k, v in cubin_file.items() if k[1] == g["kernel"]), None)
            if gk is None:
                # Found through a kernel launched only before the recorded run (setup): the
                # launched kernel of the same module on this GPU.
                gk = next(
                    (
                        v
                        for k, v in cubin_file.items()
                        if k[0] == g["dev"] and elf_defines(cubin_of[k], g["kernel"])
                    ),
                    ck(L[0][0], r0),
                )
            if gk != ck(L[0][0], r0):
                s += f'cubin = "{gk}"\n'
        if "expect" in g:
            open(f"{out}/expected/{g['name']}.bin", "wb").write(g["expect"])
            s += f'expect = "expected/{g["name"]}.bin"\n'
            if g["expect_offset"]:
                s += f"expect_offset = {g['expect_offset']}\n"
        for off, bi, bo in g.get("ptrs", []):
            s += f"[[buffers.ptrs]]\noffset = {off}\nbuffer = {q(bname(bi))}\nbyte_offset = {bo}\n"
        toml.append(s)
    for _pid, mname, size, cuts in mc_objs:
        s = f"[[multicast]]\nname = {q(mname)}\nsize = {size}\n"
        for i, roff, mcoff, n in cuts:
            s += (
                f"[[multicast.binds]]\ndevice = {regions[i]['dev']}\noffset = {mcoff}\n"
                f"buffer = {q(regions[i]['name'])}\nbyte_offset = {roff}\nsize = {n}\n"
            )
        toml.append(s)
    toml += toml_params(L[0][2], "params")
    for gi, (r, x, params) in enumerate(L[1:], 1):
        s = f"[[launches]]\nentry = {q(x['name'])}\ngrid = {x['grid']}\nblock = {x['block']}\n"
        if x["cluster"] != [1, 1, 1] or x.get("preferred_cluster"):
            s += f"cluster = {x['cluster']}\n"
        if x.get("preferred_cluster"):
            s += f"preferred_cluster = {x['preferred_cluster']}\n"
        s += f"dynamic_smem = {x['smem']}\ndevice = {r}\n"
        s += _order.toml_launch(lorder, gi, bool(x.get("pdl")))
        if ck(r, x) != ck(L[0][0], r0):
            s += f'cubin = "{ck(r, x)}"\n'
        toml.append(s)
        toml += toml_params(params, "launches.params")
    toml.append(
        '[[variants]]\nid = "good"\nsummary = "as captured"\nexpect = ["clean"]\n'
    )
    # Further mappings the parameters and pointer words went through (S-VA-ALIAS).
    for (i, view), n in va_views.items():
        toml.append(
            f"[[views]]\nname = {q(n)}\nbuffer = {q(regions[i]['name'])}\n"
            f"# rank {view[0]}'s mapping\n"
        )
    if va_views:
        notes.append(
            f"{len(va_views)} further mapping(s) of case buffers (peer imports, second maps): [[views]]"
        )
    open(f"{out}/case.toml", "w").write("\n".join(toml))

    meta = {
        "argv": [a for a in sys.argv[1:] if a != "--worker"],
        "torch": ranks[0]["torch"],
        "ranks": [
            {
                "rank": m["rank"],
                "device": m["device"],
                "uuid": m.get("uuid"),
                "streams": [hex(s) for s in streams[m["rank"]]],
            }
            for m in ranks
        ],
        "launches": [
            {
                "rank": r,
                "name": x["name"],
                "grid": x["grid"],
                "block": x["block"],
                "stream": hex(x["stream"]),
                "t_ns": x.get("t", 0) - min(y[1].get("t", 0) for y in L),
                "pdl": x.get("pdl", 0),
                "cubin": ck(r, x),
            }
            for r, x, _ in L
        ],
        # The VMM mappings of the case's buffers (owner's and every peer's view) and of
        # multicast objects; NCCL's own cuMem buffers are left out.
        "memory_map": [
            {
                "rank": r,
                "va": hex(s),
                "size": e - s,
                "phys": (pid or "?")[:48],
                "kind": phys.get(pid, {}).get("kind"),
                "owner": phys.get(pid, {}).get("owner"),
            }
            for r in range(R)
            for s, e, pid, _ in maps[r]
            if phys.get(pid, {}).get("kind") == "multicast" or pid in kept_phys
        ],
        "multicast": mc_report,
        "hosts": [
            {
                "rank": m["rank"],
                "host": m.get("host"),
                "node": m.get("node"),
                "fabric": m.get("fabric"),
            }
            for m in ranks
        ],
        "nic_pairs": nic_pairs,
        "symmetric": [
            dict(
                **{
                    k: (
                        hex(v)
                        if isinstance(v, int) and k.endswith(("ptr", "dev"))
                        else v
                    )
                    for k, v in h.items()
                    if k != "name"
                }
            )
            for m in ranks
            for h in m["symm"]
        ],
        "buffers": [
            {
                "name": regions[i]["name"],
                "device": regions[i]["dev"],
                "kind": regions[i]["kind"],
                "size": regions[i]["end"] - regions[i]["start"],
                "ptrs": len(regions[i].get("ptrs", [])),
            }
            for i in kept
        ],
        "replicas": replicas,
        "unresolved": unresolved,
        "warnings": warnings,
        "notes": notes,
        **({"launch_order": _order.summary(lorder)} if len(L) > 1 else {}),
        "events": [
            dict(
                rank=m["rank"],
                **{
                    k: (hex(v) if k in ("event", "stream") else v)
                    for k, v in x.items()
                    if k not in ("t", "t0", "t1")
                },
            )
            for m in ranks
            for x in m["records"]
            if "ev" in x
        ],
    }
    json.dump(meta, open(f"{out}/capture.json", "w"), indent=1)
    # Uniform inputs (every byte one value: a check on them may be vacuous), as
    # fb/kernel_db/scan_inputs.py reports them.
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "scan_inputs", os.path.join(ROOT, "fb/kernel_db/scan_inputs.py")
        )
        si = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(si)
        for n, w in si.uniform_inputs(Path(out)):
            warnings.append(f"uniform input {n}: {w} (the check may be vacuous)")
    except Exception as e:  # noqa: BLE001 (diagnostic only)
        warnings.append(f"uniform-input scan failed: {e}")
    meta["warnings"] = sorted(set(warnings))
    json.dump(meta, open(f"{out}/capture.json", "w"), indent=1)
    print(
        f"capture_dist: wrote {out}: {len(L)} launches on {R} GPUs, {len(kept)} buffers, "
        f"outputs {outputs}, replicas {replicas}"
    )
    for w in sorted(set(warnings)):
        print("capture_dist: warning:", w)
    for u in unresolved:
        print("capture_dist: unresolved pointer:", u)
    if mc_report:
        print(
            f"capture_dist: {len(mc_report)} multicast object(s) mapped, {len(mc_objs)} referenced by the kernels: "
            + ", ".join(f"{n} ({len(c)} binds)" for _, n, _, c in mc_objs)
        )


if A.worker:
    worker()
elif A.merge_only:
    merge()
else:
    parent()

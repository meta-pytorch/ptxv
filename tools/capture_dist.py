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
#   64-byte cudaIpcMemHandle. Multicast objects (NVLS: cuMulticastCreate + BindMem) are
#   recorded and reported; a pointer into one is a `multicast` reference (not modeled).
# * Pointer tables. torch passes peer pointers through device arrays (`T** buffer_ptrs_dev`,
#   `uint32_t** signal_pads_dev`), so every buffer a parameter references is scanned for 8-byte
#   words that resolve; those become `ptrs` of the buffer (crates/case `Buffer::ptrs`), patched
#   with the right buffer's address in the simulator and on hardware.
# * Buffers. A rank's caching-allocator blocks become buffers on that device (`_r<rank>`
#   suffix). A symmetric allocation is split at the layout its handle reports (data buffer,
#   signal pad) into `<name>_r<owner>` and `sig_r<owner>`; other VMM allocations are kept whole.
# * Launch order: each rank's launches in stream order, as `[[launches]]` with `device = rank`
#   (only same-device launches are ordered, E-PDL). Cross-rank host submission times are kept
#   in capture.json. A rank using several streams, or events, is warned about (not modeled).
# * Replicas: the outputs the module lists in REPLICAS (an all-reduce result) form a replica
#   group per name across the ranks; their bytes must be identical (N4, `divergent`).
# * Several streams per rank (DeepEP's communication stream) are serialized in submission
#   order; each stream switch must be ordered by an event record/wait (checked, else warned).
# * cudaMalloc buffers, legacy IPC (cudaIpcGetMemHandle / OpenMemHandle) and mapped host
#   memory are tracked like VMM allocations; host-to-device uploads between kernels (a
#   host-computed routing table) are folded into the destination's initial bytes (noted).
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LC = os.path.join(ROOT, "probes/launchcap")
CUDA = "/usr/local/cuda-13.0"
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
        "--kernels", help="regex: keep only launches whose kernel name matches"
    )
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument(
        "--exit-fence",
        choices=["sys", "gpu"],
        help="write exit_fence (E-EXIT-FENCE: kernel completion acquires) into the case",
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
    so_lc, so_cc = f"{WORK}/launchcap.so", f"{WORK}/cubincap.so"
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
    shutil.rmtree(WORK, ignore_errors=True)
    for r in range(A.nproc):
        for d in ("cupti", "images"):
            os.makedirs(f"{WORK}/r{r}/{d}")
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
    port = 29500 + os.getpid() % 2000
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(A.nproc),
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
    merge()
    if not A.work:
        shutil.rmtree(WORK, ignore_errors=True)


# ================================================================ worker: one rank


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
        "torch": torch.__version__,
        "pre_blocks": pre_blocks,
        "plain": [(a, z, k) for a, (z, k) in plain.items()],
        "during": during,
        "named": named_pre,
        "outputs": [(n, tinfo(t)) for n, t in outs],
        "symm": symm,
        "replicas": list(getattr(mod, "REPLICAS", [])),
        "pre": sorted(pre),
        "records": recs,
        "setup_records": [json.loads(ln) for ln in open(CAP)][:first],
    }
    json.dump(meta, open(f"{W}/rank.json", "w"))
    dist.barrier()
    dist.destroy_process_group()


# ================================================================ merge (parent)


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


def merge():  # noqa: C901 (cross-rank merge: one linear pass over shared capture state)
    R = A.nproc
    ranks = [json.load(open(f"{WORK}/r{r}/rank.json")) for r in range(R)]
    warnings, unresolved, notes = [], [], []
    kfilter = re.compile(A.kernels) if A.kernels else None

    # ------------------------------------------------ physical allocations across ranks
    # phys id -> dict(owner, kind); per rank: handle -> phys id; VA maps.
    phys = {}
    handle_phys = [{} for _ in range(R)]
    maps = [[] for _ in range(R)]  # (start, end, phys, offset in phys)
    mc_bind = {}  # multicast phys -> [(owner phys, size)]
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
        for x in vm:  # exports name a physical allocation globally
            if x["vmm"] == "export" and not x["r"]:
                key = x["fabric"] or (x["fd_ino"] and f"fd:{x['fd_dev']}:{x['fd_ino']}")
                old = handle_phys[r].get(x["handle"])
                if old and key:
                    phys[f"g:{key}"] = phys[old]
                    handle_phys[r][x["handle"]] = f"g:{key}"
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
                size = allocs[base] - (x["ptr"] - base) if base is not None else 1
                phys[f"ipc:{x['ipc']}"] = {"owner": r, "kind": "mem", "size": size}
                maps[r].append((x["ptr"], x["ptr"] + size, f"ipc:{x['ipc']}", 0))
    for r, m in enumerate(ranks):
        recs = m["setup_records"] + m["records"]
        live = list(maps[r])
        for x in (x for x in recs if "vmm" in x):
            if x["vmm"] == "import" and not x["r"]:
                key = x["fabric"] or f"fd:{x['fd_dev']}:{x['fd_ino']}"
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
                    (r, handle_phys[r].get(x["handle"]), x["size"])
                )
            elif x["vmm"] == "ipc_open" and not x["r"]:
                pid = f"ipc:{x['ipc']}"
                size = phys.get(pid, {}).get("size", 1)
                live.append((x["ptr"], x["ptr"] + size, pid, 0))
        maps[r] = live
    # Owner-side address of each physical allocation (its creator's own mapping).
    owner_va = {}
    for r in range(R):
        for s, _e, pid, off in maps[r]:
            if (
                pid
                and phys.get(pid, {}).get("owner") == r
                and phys[pid]["kind"] == "mem"
            ):
                owner_va[pid] = s - off
    mc_report = []
    for mc, binds in mc_bind.items():
        mc_report.append(
            {
                "multicast": mc[:40],
                "size": phys.get(mc, {}).get("size"),
                "members": [(r, phys.get(p, {}).get("owner")) for r, p, _ in binds],
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

    def region_at(dev, a):
        for i, g in enumerate(regions):
            if g["dev"] == dev and g["start"] <= a < g["end"]:
                return i, a - g["start"]
        for i, g in enumerate(regions):
            if g["dev"] == dev and a == g["end"]:
                return i, a - g["start"]
        return None

    def resolve(r, v):
        """(region index, offset) | ("multicast", ...) | None, for address v seen by rank r."""
        if not v:
            return None
        for s, e, pid, off in maps[r]:
            if s <= v < e:
                p = phys.get(pid)
                if p is None or p["owner"] is None:
                    return None
                if p["kind"] == "multicast":
                    return ("multicast", pid, v - s + off)
                if pid not in owner_va:
                    return None
                return region_at(p["owner"], owner_va[pid] + off + (v - s))
        return region_at(r, v)

    def near(v):
        return any(abs(v - g["start"]) < (1 << 30) for g in regions) or any(
            s - (1 << 30) <= v < e + (1 << 30) for mm in maps for s, e, _, _ in mm
        )

    refs = {}

    def note(ri, off):
        refs.setdefault(ri, set()).add(off)

    # ------------------------------------------------ launches and parameters
    launches = []  # (rank, rec, params)
    streams = {}
    for r, m in enumerate(ranks):
        recs = [x for x in m["records"] if "name" in x]
        if kfilter:
            recs = [x for x in recs if kfilter.search(x["name"])]
        streams[r] = sorted({x["stream"] for x in recs})
        # Several streams (DeepEP's communication stream): the case orders a GPU's grids as
        # one stream, in submission order. That is exact when every switch of stream between
        # consecutive launches goes through an event: recorded on the first one's stream
        # after it, waited on by the second one's stream before it. Checked here.
        if len(streams[r]) > 1:
            seq = [x for x in m["records"] if ("name" in x and x in recs) or "ev" in x]
            last = None  # (index in seq, stream) of the previous launch
            for i, x in enumerate(seq):
                if "name" not in x:
                    continue
                if last and last[1] != x["stream"]:
                    between = seq[last[0] + 1 : i]
                    evs = {
                        e["event"]
                        for e in between
                        if e["ev"] == "record" and e["stream"] == last[1]
                    }
                    ok = any(
                        e["ev"] == "wait"
                        and e["stream"] == x["stream"]
                        and e["event"] in evs
                        for e in between
                    )
                    if not ok:
                        warnings.append(
                            f"rank {r}: {x['name'][:50]} on stream {x['stream']:#x} may overlap the previous launch on {last[1]:#x} (no event between them): the case serializes them"
                        )
                    else:
                        notes.append(
                            f"rank {r}: stream {last[1]:#x} -> {x['stream']:#x} ordered by an event before {x['name'][:50]}"
                        )
                last = (i, x["stream"])
        for x in recs:
            params = []
            for pi, p in enumerate(x["params"]):
                raw = bytes.fromhex(p["hex"])
                if len(raw) == 8:
                    v = u64(raw, 0)
                    loc = resolve(r, v)
                    if loc and loc[0] == "multicast":
                        unresolved.append(
                            {
                                "rank": r,
                                "launch": x["name"][:60],
                                "param": pi,
                                "value": hex(v),
                                "why": "multicast (NVLS) address",
                            }
                        )
                        params.append(("u64", v))
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
                    for off in range(0, len(raw) - 7, 8):
                        v = u64(raw, off)
                        loc = resolve(r, v)
                        if loc and loc[0] != "multicast":
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
                    params.append(("bytes", raw, fields))
            launches.append((r, x, params))
    if not launches:
        sys.exit("capture_dist: no kernel launches recorded")

    # ------------------------------------------------ initial and final bytes
    for _i, g in enumerate(regions):
        r = g["dev"]
        size = g["end"] - g["start"]
        data, found = bytearray(size), False
        for a in ranks[r]["pre"]:
            b = open(f"{WORK}/r{r}/pre/{a:x}.bin", "rb").read()
            lo, hi = max(a, g["start"]), min(a + len(b), g["end"])
            if lo < hi:
                data[lo - g["start"] : hi - g["start"]] = b[lo - a : hi - a]
                found = True
        g["data"] = bytes(data) if found and any(data) else None

    # Device copies / fills a rank made before its first kernel (an op's clone(), a copy_
    # into the symmetric buffer) fold into the initial bytes; later ones are not modeled.
    for r, m in enumerate(ranks):
        seen_kernel = False
        for x in m["records"]:
            if "name" in x and (not kfilter or kfilter.search(x["name"])):
                seen_kernel = True
                continue
            if "op" not in x or x.get("kind", 0) not in (0, 1, 3, 4):
                continue  # device -> host copies touch no case buffer
            if x.get("kind") == 1 and "data" not in x:
                continue
            if x.get("kind") == 1 and seen_kernel:
                # A host-computed table uploaded between kernels (MoE routing order): its bytes
                # become the destination's initial contents. Exact when no earlier kernel
                # touches that memory (a fresh allocation), which the warning asks to confirm.
                notes.append(
                    f"rank {r}: host upload of {x['bytes']} bytes to {x['dst']:#x} between kernels folded into initial contents"
                )
            elif seen_kernel:
                warnings.append(
                    f"rank {r}: {x['op']} of {x['bytes']} bytes to {x['dst']:#x} between kernels: not in the case"
                )
                continue
            loc = resolve(r, x["dst"])
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
                    warnings.append(
                        f"rank {r}: memcpy into {g['name']} from {x['src']:#x} (unknown contents) ignored"
                    )
                    continue
                d[off : off + n] = src
            g["data"] = bytes(d)

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
                unresolved.append(
                    {
                        "rank": g["dev"],
                        "buffer": g["name"],
                        "offset": off,
                        "value": hex(v),
                        "why": "multicast (NVLS) address in a table",
                    }
                )
            elif loc:
                g["ptrs"].append((off,) + loc)
                note(*loc)
                todo.append(loc[0])
        # A pointer table is not data: the simulator fills the words in.
        if g["ptrs"] and len(g["data"]) % 8 == 0:
            g["dtype"], g["shape"] = "u64", [len(g["data"]) // 8]

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

    # Keep only regions something references (plus outputs).
    keep = set(refs) | {i for d in out_regions.values() for i in d.values()}
    for i in list(keep):
        for p in regions[i].get("ptrs", []):
            keep.add(p[1])
    kept = sorted(keep)
    names_taken = set()
    for i in kept:
        g = regions[i]
        base, k = g["name"], 1
        while g["name"] in names_taken:
            g["name"] = f"{base}_{k}"
            k += 1
        names_taken.add(g["name"])

    def bname(i):
        return regions[i]["name"]

    # ------------------------------------------------ cubins
    images = []
    for r in range(R):
        images += glob.glob(f"{WORK}/r{r}/cupti/*.cubin") + glob.glob(
            f"{WORK}/r{r}/images/*.bin"
        )
    knames = []
    for _, x, _ in launches:
        if x["name"] not in knames:
            knames.append(x["name"])
    cubin_of = {}
    for n in knames:
        c = [p for p in images if "/cupti/" in p and elf_defines(p, n)] or [
            p for p in images if elf_defines(p, n)
        ]
        if not c:
            sys.exit(f"capture_dist: no captured cubin defines {n}")
        cubin_of[n] = min(c, key=os.path.getsize)
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
                k
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

    def toml_params(params, key):
        s = []
        for p in params:
            if p[0] == "ptr":
                s.append(
                    f'[[{key}]]\nkind = "ptr"\nbuffer = {q(bname(p[1]))}\nbyte_offset = {p[2]}\n'
                )
            elif p[0] == "u32":
                s.append(f'[[{key}]]\nkind = "u32"\nvalue = {p[1]}\n')
            elif p[0] == "u64":
                s.append(f'[[{key}]]\nkind = "u64"\nvalue = {p[1]}\n')
            else:
                _, raw, fields = p
                s.append(f'[[{key}]]\nkind = "bytes"\nhex = "{raw.hex()}"\n')
                for off, bi, bo in fields:
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
Captured by tools/capture_dist.py (torch {ranks[0]["torch"]}, torchrun --nproc-per-node {R}):
run() of {os.path.basename(A.file)}. {len(L)} launch(es) on {R} GPUs, binary only:
{kern_lines}
Regenerate: CUDA_VISIBLE_DEVICES=0,1,3 $PTXV_PYTHON tools/capture_dist.py {cmdline}"""
    desc = desc.replace("\\", "\\\\").replace('"', '\\"')
    outputs = [bname(i) for i in kept if "expect" in regions[i]]
    toml = [
        f'name = {q(name)}\ndescription = """{desc}"""\ncubin = "{cubin_file[r0["name"]]}"\n'
        f"entry = {q(r0['name'])}\ngrid = {r0['grid']}\nblock = {r0['block']}\n"
        f"dynamic_smem = {r0['smem']}\noutputs = {q(outputs)}\nreplicas = {q(replicas)}\n"
        + (f'exit_fence = "{A.exit_fence}"\n' if A.exit_fence else "")
    ]
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
        if "expect" in g:
            open(f"{out}/expected/{g['name']}.bin", "wb").write(g["expect"])
            s += f'expect = "expected/{g["name"]}.bin"\n'
            if g["expect_offset"]:
                s += f"expect_offset = {g['expect_offset']}\n"
        for off, bi, bo in g.get("ptrs", []):
            s += f"[[buffers.ptrs]]\noffset = {off}\nbuffer = {q(bname(bi))}\nbyte_offset = {bo}\n"
        toml.append(s)
    toml += toml_params(L[0][2], "params")
    for r, x, params in L[1:]:
        s = f"[[launches]]\nentry = {q(x['name'])}\ngrid = {x['grid']}\nblock = {x['block']}\n"
        if x["cluster"] != [1, 1, 1]:
            s += f"cluster = {x['cluster']}\n"
        s += f"dynamic_smem = {x['smem']}\ndevice = {r}\n"
        if x.get("pdl"):
            s += "pdl = true\n"
        if cubin_file[x["name"]] != cubin_file[r0["name"]]:
            s += f'cubin = "{cubin_file[x["name"]]}"\n'
        toml.append(s)
        toml += toml_params(params, "launches.params")
    toml.append(
        '[[variants]]\nid = "good"\nsummary = "as captured"\nexpect = ["clean"]\n'
    )
    open(f"{out}/case.toml", "w").write("\n".join(toml))

    kept_phys = {
        pid
        for pid, va in owner_va.items()
        for i in kept
        if regions[i]["dev"] == phys[pid]["owner"]
        and va <= regions[i]["start"] < va + phys[pid]["size"]
    }
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
                "cubin": cubin_file[x["name"]],
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
        "events": [
            dict(
                rank=m["rank"],
                **{
                    k: (hex(v) if k in ("event", "stream") else v)
                    for k, v in x.items()
                    if k != "t"
                },
            )
            for m in ranks
            for x in m["records"]
            if "ev" in x
        ],
    }
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
            f"capture_dist: {len(mc_report)} multicast object(s) mapped (not referenced by kernels unless listed as unresolved)"
        )


if A.worker:
    worker()
elif A.merge_only:
    merge()
else:
    parent()

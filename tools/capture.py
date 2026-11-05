# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Turn any torch op, or a Python function that launches kernels (Triton, CuTeDSL, torch), into
# a ptxv case: every kernel launched, in order, with its cubin(s), its parameters mapped to
# named buffers, exact input bytes and the GPU's output bytes.
#
#   CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/capture.py \
#       --op 'torch.softmax(x, -1)' --inputs 'x=randn(4,128,bf16)' --out fixtures/c_softmax
#   ... tools/capture.py --file run.py --out DIR            # captures everything run() launches
#   ... tools/capture.py --triton k.py:softmax_kernel --out DIR   # only that kernel, PTX kept
#
# --inputs takes NAME=EXPR (repeatable, or comma-separated at the top level). EXPR is Python
# over helpers that build seeded CPU tensors and move them to the GPU: randn / rand / zeros /
# ones / arange / randint(lo, hi, *shape) / full(*shape, value) / eye(n), with an optional
# trailing dtype (bf16 f16 f32 f64 i32 i64 i16 i8 u8 bool), plus `torch`, `F`.
# --op is a Python expression over the inputs (and torch, F); its tensors are the outputs.
#
# A --file / --triton module defines `run(**inputs)`, returning the output tensor(s), and
# optionally `inputs()` returning {name: tensor}; without it, tensors run() creates count as
# workspace. Tensors that run() changes in place are outputs too.
#
# How (generalizing fixtures/r3_grouped_mm/capture.py and r2_cublas_gemm/capture.py):
# * The script re-executes itself under probes/launchcap (every launch: grid, block, cluster,
#   PDL, smem, parameter bytes) and probes/launchcap/cubincap (CUPTI: every module the driver
#   loads, including libtorch_cuda's lazily loaded ones) plus launchcap's image dump (cuBLAS /
#   cuDNN library images). Triton and CuTeDSL dump their PTX (TRITON_KERNEL_DUMP,
#   CUTE_DSL_KEEP).
# * The op runs once to warm up (lazy loading, autotuning, library workspaces) and is captured
#   on its second call, between two snapshots of the caching allocator: the blocks live before
#   the call and the allocations made during it. Every 8-byte word of every parameter that
#   points into one of them becomes a `ptr` (or `fields` entry of a `bytes` parameter), and a
#   128-byte CUtensorMap (top level, or 64-byte aligned inside a packed argument block) whose
#   address does becomes a `tensormap` with its geometry decoded (docs/tensormap.md) and the
#   raw bytes kept for hardware runs.
# * Overlapping blocks (an allocation reused during the call) merge into one buffer. A buffer
#   that only a single named tensor's bytes are referenced in is that tensor (its dtype and
#   shape); otherwise it is the whole block range as u8, the tensor at an offset.
# * Buffers live before the call get their exact bytes (`init = "file"`, data/NAME.bin);
#   allocations made during it start as zeros (the simulator's; the GPU's are whatever the
#   allocator hands out, which a correct kernel does not read). Outputs carry the GPU's bytes
#   (`expect`, expected/NAME.bin), which `ptxv check` compares with the simulator.
# * Each kernel's cubin is the smallest captured ELF image defining it; kernels from other
#   modules than the first launch's name their own (`[[launches]] cubin`). When every launch is
#   one Triton / CuTeDSL kernel with dumped PTX, the case is a PTX case (both front-ends) and
#   the JIT's cubin is kept as jit.cubin for reference.
# Not captured: cudaMemcpy / cudaMemset inside the op (not kernel launches), work on other
# streams, pointers to memory outside the caching allocator (listed under `unresolved` in
# capture.json).
import argparse
import ctypes
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


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--op", help="Python expression over the inputs")
    g.add_argument("--file", help="Python file defining run(**inputs) [and inputs()]")
    g.add_argument(
        "--triton", help="FILE.py:KERNEL: like --file, keeping only KERNEL's launches"
    )
    ap.add_argument(
        "--inputs",
        action="append",
        default=[],
        help="NAME=EXPR (--op; with --file / --triton: instead of the module's inputs())",
    )
    ap.add_argument("--out", required=True, help="case directory to write")
    ap.add_argument("--name", help="case name (default: the directory name)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--kernels", help="regex: keep only launches whose kernel name matches"
    )
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument(
        "--binary", action="store_true", help="binary-only even when PTX is available"
    )
    ap.add_argument(
        "--no-prune",
        action="store_true",
        help="keep other kernels' code in module cubins",
    )
    ap.add_argument(
        "--work",
        default=None,
        help="scratch directory, kept (default: a temporary one)",
    )
    return ap.parse_args()


A = parse_args()
WORK = os.path.abspath(A.work or f"/tmp/ptxv_capture_{os.getpid()}")
CAP = f"{WORK}/launches.jsonl"
CUB = f"{WORK}/cupti"
IMG = f"{WORK}/images"
DUMP = f"{WORK}/jit"


def build_probes():
    import sysconfig  # noqa: F401  (the conda env's lib holds libcupti)

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


if os.environ.get("PTXV_CAPTURE_CHILD") != "1":  # re-exec under the recorders
    shutil.rmtree(WORK, ignore_errors=True)
    for d in (CUB, IMG, DUMP + "/triton", DUMP + "/cute"):
        os.makedirs(d)
    so_lc, so_cc = build_probes()
    env = dict(
        os.environ,
        PTXV_CAPTURE_CHILD="1",
        LD_PRELOAD=f"{so_lc}:{so_cc}",
        LAUNCHCAP_OUT=CAP,
        LAUNCHCAP_IMAGES=IMG,
        LAUNCHCAP_MEMOPS="1",
        CUBINCAP_DIR=CUB,
        TRITON_KERNEL_DUMP="1",
        TRITON_DUMP_DIR=DUMP + "/triton",
        TRITON_ALWAYS_COMPILE="1",
        CUTE_DSL_DUMP_DIR=DUMP + "/cute",
        CUTE_DSL_KEEP="ptx,cubin",
        CUTE_DSL_NO_CACHE="1",
        CUTE_DSL_DISABLE_FILE_CACHING="1",
    )
    if A.kernels:
        # Snapshots around the kept launches (probes/launchcap): their inputs may come from
        # launches filtered out.
        os.makedirs(f"{WORK}/snap")
        env.update(LAUNCHCAP_SNAP=A.kernels, LAUNCHCAP_SNAP_DIR=f"{WORK}/snap")
    # Triton's bundled tools do not know sm_103a (the kernel dump disassembles the cubin).
    for tool in ("ptxas", "cuobjdump", "nvdisasm"):
        env.setdefault(f"TRITON_{tool.upper()}_PATH", f"{CUDA}/bin/{tool}")
    r = subprocess.run([sys.executable] + sys.argv + ["--work", WORK], env=env)
    if not A.work:  # the recorded images can be large (every module the process loaded)
        shutil.rmtree(WORK, ignore_errors=True)
    sys.exit(r.returncode)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

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

# ---------------------------------------------------------------- inputs


def make_input_ns(seed):
    gen = torch.Generator().manual_seed(seed)
    dts = {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "f32": torch.float32,
        "f64": torch.float64,
        "i32": torch.int32,
        "i64": torch.int64,
        "i16": torch.int16,
        "i8": torch.int8,
        "u8": torch.uint8,
    }
    dts["bool"] = torch.bool

    def split(args, default):
        args = list(args)
        dt = args.pop() if args and isinstance(args[-1], torch.dtype) else default
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            args = list(args[0])
        return [int(a) for a in args], dt

    def mk(f, default=torch.float32):
        def g(*a):
            shape, dt = split(a, default)
            return f(shape, dt).cuda()

        return g

    ns = dict(dts, torch=torch, F=F)
    ns["randn"] = mk(lambda s, d: torch.randn(s, generator=gen).to(d))
    ns["rand"] = mk(lambda s, d: torch.rand(s, generator=gen).to(d))
    ns["zeros"] = mk(lambda s, d: torch.zeros(s, dtype=d))
    ns["ones"] = mk(lambda s, d: torch.ones(s, dtype=d))

    def arange(n, dt=torch.int64):
        return torch.arange(n, dtype=dt).cuda()

    def randint(lo, hi, *a):
        shape, dt = split(a, torch.int64)
        return torch.randint(lo, hi, shape, generator=gen, dtype=dt).cuda()

    def full(*a):
        a = list(a)
        dt = a.pop() if isinstance(a[-1], torch.dtype) else torch.float32
        v = a.pop()
        return torch.full([int(x) for x in a], v, dtype=dt).cuda()

    def eye(n, dt=torch.float32):
        return torch.eye(n, dtype=dt).cuda()

    ns.update(arange=arange, randint=randint, full=full, eye=eye)
    return ns


def split_top(s):
    """Split 'a=f(1,2),b=g(3)' at top-level commas."""
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += ch in "([{"
        depth -= ch in ")]}"
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def flat_tensors(x, prefix="out"):
    """Named output tensors of a result (tensor, tuple / list / namedtuple, dict)."""
    if isinstance(x, torch.Tensor):
        return [(prefix, x)]
    if isinstance(x, dict):
        return [t for k, v in x.items() for t in flat_tensors(v, f"{prefix}_{k}")]
    if isinstance(x, (tuple, list)):
        return [t for i, v in enumerate(x) for t in flat_tensors(v, f"{prefix}{i}")]
    return []


def load_module(path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("capture_target", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    return mod


torch.manual_seed(A.seed)
kfilter = re.compile(A.kernels) if A.kernels else None
if A.op:
    ns = make_input_ns(A.seed)
    inputs = {}
    for spec in A.inputs:
        for item in split_top(spec):
            n, e = item.split("=", 1)
            inputs[n.strip()] = eval(e, ns)
    env = dict(torch=torch, F=F, math=__import__("math"), **inputs)

    def run():
        return eval(A.op, env)

    what = f"`{A.op}` with " + ", ".join(
        f"{n} = {e}"
        for s in A.inputs
        for n, e in (x.split("=", 1) for x in split_top(s))
    )
else:
    path = A.file or A.triton.rsplit(":", 1)[0]
    if A.triton and ":" in A.triton:
        kname = A.triton.rsplit(":", 1)[1]
        kfilter = re.compile(rf"^{re.escape(kname)}$")
    mod = load_module(path)
    if A.inputs:
        # --inputs overrides the module's inputs() (tools/compare.py: both sides of a
        # comparison get the same seeded tensors).
        ns = make_input_ns(A.seed)
        inputs = {}
        for spec in A.inputs:
            for item in split_top(spec):
                n, e = item.split("=", 1)
                inputs[n.strip()] = eval(e, ns)
    else:
        inputs = dict(mod.inputs()) if hasattr(mod, "inputs") else {}

    def run():
        return mod.run(**inputs)

    what = f"run() of {os.path.basename(path)}" + (
        f" (kernel {kfilter.pattern})" if A.triton else ""
    )

# ---------------------------------------------------------------- raw device memory

_rt = None


def dev_read(addr, n):
    """Bytes of device memory (cudaMemcpy through the runtime torch loaded)."""
    global _rt
    if _rt is None:
        _rt = ctypes.CDLL("libcudart.so.13")
        _rt.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
    buf = ctypes.create_string_buffer(n)
    r = _rt.cudaMemcpy(buf, ctypes.c_void_p(addr), n, 2)  # cudaMemcpyDeviceToHost
    if r != 0:
        raise RuntimeError(f"cudaMemcpy from {addr:#x} ({n} bytes) failed: {r}")
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


# ---------------------------------------------------------------- run

if not A.no_warmup:
    # An op that changes an input in place (index_add_, an Inductor in_out_ptr kernel) would
    # otherwise be captured on the warm-up's result: restore the inputs before the capture.
    saved = {
        n: t.clone()
        for n, t in inputs.items()
        if isinstance(t, torch.Tensor) and t.is_cuda
    }
    run()
    torch.cuda.synchronize()
    for n, t in saved.items():
        inputs[n].copy_(t)
    del saved
torch.cuda.synchronize()
first = sum(1 for _ in open(CAP)) if os.path.exists(CAP) else 0
pre_blocks = active_blocks()
pre_named = {
    n: storage_range(t) for n, t in inputs.items() if isinstance(t, torch.Tensor)
}
pre_bytes = {n: dev_read(p, z) for n, (p, z) in pre_named.items() if z}
pre_block_bytes = {}
for a, z in pre_blocks:
    if not any(a <= p < a + z for p, _ in pre_named.values()) and z <= 4 << 20:
        pre_block_bytes[a] = dev_read(a, z)
torch.cuda.memory._record_memory_history(max_entries=1000000)
result = run()
torch.cuda.synchronize()
history = [e for tr in torch.cuda.memory._snapshot()["device_traces"] for e in tr]
torch.cuda.memory._record_memory_history(enabled=None)
during = [(e["addr"], e["size"]) for e in history if e["action"] == "alloc"]

outs = flat_tensors(result)
# Inputs changed in place are outputs too (index_add_, out= arguments).
for n, (p, z) in pre_named.items():
    if (
        z
        and dev_read(p, z) != pre_bytes[n]
        and not any(storage_range(t)[0] == p for _, t in outs)
    ):
        outs.append((n, inputs[n]))
seen, uniq = set(), []
for n, t in outs:
    if storage_range(t) not in seen:
        seen.add(storage_range(t))
        uniq.append((n, t))
outs = uniq
named = dict(inputs)
for n, t in outs:
    named.setdefault(n, t)
named = {n: t for n, t in named.items() if isinstance(t, torch.Tensor) and t.is_cuda}

recs = [json.loads(ln) for ln in open(CAP)][first:]
# cudaLaunchKernelEx without a cluster attribute reports cluster dims 0: no cluster.
for r in recs:
    if r.get("cluster") == [0, 0, 0]:
        r["cluster"] = [1, 1, 1]
# Memory operations (cudaMemcpy / cudaMemset) are not kernels. Device-side ones before the
# first launch fold into the buffers' initial contents (an op's clone() or zero fill);
# later ones cannot be expressed in a case and are reported.
early_memops = []
for r in recs:
    if "op" in r:
        early_memops.append(r)
    elif not kfilter or kfilter.search(r["name"]):
        break
late_memops = [r for r in recs if "op" in r][len(early_memops) :]
# Host <-> device copies (an .item(), a pinned staging buffer) touch no case buffer.
late_memops = [r for r in late_memops if r["op"] == "memset" or r["kind"] in (3, 4)]
recs = [r for r in recs if "name" in r]
dropped = []
if kfilter:
    dropped = [r["name"] for r in recs if not kfilter.search(r["name"])]
    recs = [r for r in recs if kfilter.search(r["name"])]
if not recs:
    sys.exit("capture: no kernel launches recorded")
print(
    f"capture: {len(recs)} launches"
    + (f" ({len(dropped)} filtered out)" if dropped else "")
)


def load_snap(n, tag):
    """{base address: bytes} of allocation snapshot n (`pre` / `post`), probes/launchcap."""
    got = {}
    for f in glob.glob(f"{WORK}/snap/s{n}_{tag}_*.bin"):
        got[int(f.rsplit("_", 1)[1][:-4], 16)] = open(f, "rb").read()
    return got


# --kernels drops launches: with snapshots, the buffers start as they were just before the
# first kept launch and the outputs are what the kept launches changed, as of the last one.
snap_pre, snap_post = {}, {}
if dropped and all("snap" in r for r in recs):
    snap_pre, snap_post = (
        load_snap(recs[0]["snap"], "pre"),
        load_snap(recs[-1]["snap"], "post"),
    )
    if [r["snap"] for r in recs] != list(range(recs[0]["snap"], recs[-1]["snap"] + 1)):
        print(
            "capture: warning: kept launches are not contiguous; snapshots span the gap"
        )
elif dropped:
    print(
        "capture: warning: launches filtered out and no snapshots: buffers they wrote start as zeros"
    )

# ---------------------------------------------------------------- memory regions

# Intervals: named tensors' storages, blocks live before the call, allocations during it.
ivs = []  # (start, end, kind, name)
for n, t in named.items():
    p, z = storage_range(t)
    if z:
        ivs.append((p, p + z, "named", n))
for a, z in pre_blocks:
    ivs.append((a, a + z, "pre", None))
for a, z in during:
    ivs.append((a, a + z, "during", None))
# Allocations outside the caching allocator (a library's cudaMalloc) that a kept launch points
# into: known from the snapshots.
for b, d in snap_pre.items():
    if not any(s <= b < e for s, e, _, _ in ivs) and not any(
        b <= s < b + len(d) for s, _, _, _ in ivs
    ):
        ivs.append((b, b + len(d), "snap", None))
ivs.sort()
comps = []  # merged: [start, end, [intervals]]
for iv in ivs:
    if comps and iv[0] < comps[-1][1]:
        comps[-1][1] = max(comps[-1][1], iv[1])
        comps[-1][2].append(iv)
    else:
        comps.append([iv[0], iv[1], [iv]])


def comp_of(v):
    """(component index, offset) holding device address v; one past the end also counts."""
    for i, (s, e, _) in enumerate(comps):
        if s <= v < e:
            return i, v - s
    for i, (s, e, _) in enumerate(comps):
        if v == e:
            return i, v - s
    return None


# CUtensorMap element type codes (docs/tensormap.md) -> (dtype, bytes).
TMAP_DT = {
    0: "u8",
    11: "u8",  # 16U4_ALIGN8B, decoded in bytes
    1: "u16",
    2: "u32",
    3: "i32",
    4: "u64",
    5: "i64",
    6: "f16",
    7: "f32",
    8: "f32",
    9: "f64",
    10: "bf16",
}


def decode_tmap(raw):  # noqa: C901 (flat field-by-field decoder of the CUtensorMap layout)
    """Geometry of a 128-byte tiled CUtensorMap, or None if it does not look like one."""
    if len(raw) < 128:
        return None
    w2 = struct.unpack_from("<I", raw, 8)[0]
    if (
        w2 & 0xD or (w2 >> 11) & 3 or w2 >> 22
    ):  # im2col, unused bits 2-3, interleave, 22-31
        return None
    rank = ((w2 >> 4) & 7) + 1
    code = (w2 >> 7) & 15
    if rank > 5 or code not in TMAP_DT:
        return None
    dt = TMAP_DT[code]
    es = ESIZE[dt]
    dims = [struct.unpack_from("<I", raw, 0x20 + 4 * i)[0] + 1 for i in range(rank)]
    box = [raw[0x37 + i] + 1 for i in range(rank)]
    elem4 = code == 11  # 16U4_ALIGN8B: the case gives it in bytes (`elem4`, S-TMA)
    if elem4:
        if dims[0] % 2 or box[0] % 2:
            return None
        dims[0] //= 2
        box[0] //= 2
    hi = struct.unpack_from("<I", raw, 0x1C)[0]
    strides = []
    for i in range(1, rank):
        lo = struct.unpack_from("<I", raw, 8 + 4 * i)[0]
        s = (lo << 4) | (((hi >> (4 * (i - 1))) & 15) << 36)
        # A dimension of extent 1 is never stepped: keep its stride consistent.
        strides.append(
            s if dims[i] > 1 or s else dims[i - 1] * (strides[-1] if strides else es)
        )
    if any(b > 256 for b in box) or (strides and strides[0] < es):
        return None
    # Slots past the rank are 0 in every descriptor seen (docs/tensormap.md: the dims and
    # boxes; strides may hold junk), and a row fits its stride.
    if any(struct.unpack_from("<I", raw, 0x20 + 4 * i)[0] for i in range(rank, 5)):
        return None
    if any(raw[0x37 + i] for i in range(rank, 5)) or any(raw[0x3C:0x40]):
        return None
    if strides and dims[0] * es > strides[0]:
        return None
    if not w2 & 2:
        # Driver-built: the derived words (box bytes, swizzle repeat) are consistent.
        span = [0, 32, 64, 128][(w2 >> 13) & 3]
        bb = es
        for b in box:
            bb *= b
        if u64(raw, 0x40) != bb or u64(raw, 0x48) != (8 * span if span else 16):
            return None
    estr = struct.unpack_from("<I", raw, 0x34)[0]
    notes = []
    if estr & 0x7FFF:
        notes.append("element strides != 1 (not modeled)")
    if (w2 >> 19) & 3:
        notes.append("swizzle atom != 16B (not modeled)")
    swz = ["none", "32B", "64B", "128B"][(w2 >> 13) & 3]
    return {
        "dtype": dt,
        "dims": dims,
        "box": box,
        "strides": strides,
        "swizzle": swz,
        "notes": notes,
        "elem4": elem4,
    }


def u64(raw, off):
    return struct.unpack_from("<Q", raw, off)[0]


# Per parameter: ("ptr", comp, off) | ("tmap", comp, off, geometry, raw) | ("bytes", raw, fields)
# | ("u32", value).
refs = {}  # comp -> list of (offset, length) referenced (length 1 for plain pointers)
tmap_refs = {}  # comp -> dtypes used by tensor maps over it
unresolved = []


def near_alloc(v):
    """Within 1 GB of an allocation: probably a pointer this capture does not know."""
    return any(s - (1 << 30) <= v < e + (1 << 30) for s, e, _ in comps)


if os.environ.get("PTXV_CAPTURE_DEBUG"):
    print("comps", [(hex(s), e - s, [m[2:] for m in ms]) for s, e, ms in comps])


def note_ref(ci, off, n=1):
    refs.setdefault(ci, []).append((off, n))


def scan_blob(raw, li, pi):
    fields = []
    off = 0
    while off + 8 <= len(raw):
        v = u64(raw, off)
        loc = comp_of(v) if v else None
        if loc and off % 64 == 0 and off + 128 <= len(raw):
            g = decode_tmap(raw[off : off + 128])
            if g and raw[off + 8 : off + 16] != bytes(8):
                ci, o = loc
                tbytes = max(
                    g["dims"][-1]
                    * (g["strides"][-1] if g["strides"] else ESIZE[g["dtype"]]),
                    1,
                )
                note_ref(ci, o, tbytes)
                tmap_refs.setdefault(ci, set()).add(g["dtype"])
                n = 64 if raw[off + 8] & 2 else 128  # a bit-1 map is its first 64 bytes
                fields.append((off, ("tmap", ci, o, g, raw[off : off + n])))
                # cuBLAS packs its (bit-1) maps 64 bytes apart: bytes 64..127 of one are the
                # next map (docs/tensormap.md).
                off += n
                continue
        if loc:
            note_ref(*loc)
            fields.append((off, ("ptr",) + loc))
        elif v and near_alloc(v):
            unresolved.append(
                {"launch": li, "param": pi, "offset": off, "value": hex(v)}
            )
        off += 8
    return fields


launches = []
for li, r in enumerate(recs):
    params = []
    for pi, p in enumerate(r["params"]):
        raw = bytes.fromhex(p["hex"])
        if len(raw) == 8 and u64(raw, 0) and comp_of(u64(raw, 0)):
            loc = comp_of(u64(raw, 0))
            note_ref(*loc)
            params.append(("ptr",) + loc)
            continue
        if len(raw) == 128 and not p.get("packed"):
            f = scan_blob(raw, li, pi)
            if len(f) == 1 and f[0][0] == 0 and f[0][1][0] == "tmap":
                params.append(f[0][1])
                continue
        if len(raw) == 4:
            params.append(("u32", struct.unpack("<I", raw)[0]))
            continue
        params.append(("bytes", raw, scan_blob(raw, li, pi)))
    launches.append((r, params))

# ---------------------------------------------------------------- buffers

buffers = []  # dicts: name, dtype, shape, start (device address), size, init, data, expect
comp_buf = {}  # comp -> (buffer index, address of the buffer's byte 0 minus the comp start)
out_names = {n for n, _ in outs}
nws = 0


def tensor_dtype_shape(t, nbytes):
    dt = DT.get(t.dtype)
    if dt is None:
        return "u8", [nbytes]
    n = nbytes // ESIZE[dt]
    if t.is_contiguous() and t.storage_offset() == 0 and t.numel() == n and t.dim() > 0:
        return dt, list(t.shape)
    return dt, [n]


needed = set(refs)
if snap_pre:  # the op's outputs come from launches filtered out
    outs = []
for _n, t in outs:  # outputs are buffers even if no parameter points at them (warned)
    p, _ = storage_range(t)
    loc = comp_of(p)
    if loc:
        needed.add(loc[0])
warnings = []
for mo in late_memops:
    warnings.append(
        f"{mo['op']} of {mo['bytes']} bytes to {mo['dst']:#x} between kernels: not in the case"
    )
for ci in sorted(needed):  # noqa: C901 (script body: one linear pass over shared capture state)
    s, e, members = comps[ci]
    nm = [m for m in members if m[2] == "named"]
    rs = refs.get(ci, [])
    # A single named tensor whose bytes hold every reference: the buffer is that tensor.
    solo = None
    for m in nm:
        if (
            all(m[0] <= s + o and s + o + max(k, 1) - 1 <= m[1] for o, k in rs)
            or not rs
        ):
            solo = m
            break
    if solo:
        t = named[solo[3]]
        start, size = solo[0], solo[1] - solo[0]
        dt, shape = tensor_dtype_shape(t, size)
        name = solo[3]
    else:
        start, size = s, e - s
        if nm:
            name = nm[0][3]
        else:
            name = f"W{nws}"
            nws += 1
        dt, shape = "u8", [size]
    tds = tmap_refs.get(ci, set())
    if tds:
        td = sorted(tds)[0]
        if len(tds) > 1:
            warnings.append(
                f"{name}: tensor maps of several element types {sorted(tds)}"
            )
        if ESIZE[dt] != ESIZE[td]:
            if dt == "u8" and size % ESIZE[td] == 0:
                dt, shape = td, [size // ESIZE[td]]
            else:
                warnings.append(f"{name}: dtype {dt} but a tensor map reads {td}")
    b = {"name": name, "dtype": dt, "shape": shape, "start": start, "size": size}
    # Initial contents: exact bytes of whatever was live before the call.
    live = [
        n
        for n in pre_named
        if pre_named[n][0] >= start and pre_named[n][0] < start + size
    ]
    if solo and solo[3] in pre_bytes:
        b["data"] = pre_bytes[solo[3]]
    else:
        data = bytearray(size)
        anylive = False
        for n in live:
            p, z = pre_named[n]
            o = p - start
            data[o : o + min(z, size - o)] = pre_bytes[n][: size - o]
            anylive = True
        for a, z in pre_blocks:
            if a in pre_block_bytes and start <= a < start + size:
                o = a - start
                data[o : o + min(z, size - o)] = pre_block_bytes[a][: size - o]
                anylive = True
            elif (
                a not in pre_block_bytes
                and start <= a < start + size
                and not any(
                    pre_named[n][0] >= a and pre_named[n][0] < a + z for n in pre_named
                )
            ):
                warnings.append(
                    f"{name}: live block of {z} bytes before the call left as zeros"
                )
        if anylive and any(data):
            b["data"] = bytes(data)
    # Memory operations before the first kernel, in order.
    for mo in early_memops:
        d0, n = mo["dst"], mo["bytes"]
        lo, hi = max(d0, start), min(d0 + n, start + size)
        if lo >= hi:
            continue
        data = bytearray(b.get("data", bytes(size)))
        if mo["op"] == "memset":
            data[lo - start : hi - start] = bytes([mo["value"] & 255]) * (hi - lo)
        else:
            src = mo["src"] + (lo - d0)
            got = None
            for nn, (p, z) in pre_named.items():
                if p <= src and src + (hi - lo) <= p + z:
                    got = pre_bytes[nn][src - p : src - p + hi - lo]
            for a, z in pre_blocks:
                if (
                    got is None
                    and a in pre_block_bytes
                    and a <= src
                    and src + hi - lo <= a + z
                ):
                    got = pre_block_bytes[a][src - a : src - a + hi - lo]
            if got is None:
                warnings.append(
                    f"{name}: memcpy into it from {src:#x} (unknown contents) ignored"
                )
                continue
            data[lo - start : hi - start] = got
        b["data"] = bytes(data)
    if snap_pre:
        pre = bytearray(b.get("data", bytes(size)))
        post = bytearray(pre)
        for snap, img in ((snap_pre, pre), (snap_post, post)):
            for a0, d in snap.items():
                lo, hi = max(a0, start), min(a0 + len(d), start + size)
                if lo < hi:
                    img[lo - start : hi - start] = d[lo - a0 : hi - a0]
        if any(pre):
            b["data"] = bytes(pre)
        if post != pre:
            b["expect"], b["expect_offset"], b["is_out"] = bytes(post), 0, True
    # Outputs: the GPU's bytes after the call.
    for n, t in outs:
        p, z = storage_range(t)
        if start <= p < start + size:
            if "expect" in b:
                warnings.append(f"{name}: also holds output {n} (not compared)")
                continue
            b["expect"] = dev_read(p, z)
            b["expect_offset"] = p - start
            b["is_out"] = True
    comp_buf[ci] = (len(buffers), start - s)
    buffers.append(b)
for n, t in outs:
    p, _ = storage_range(t)
    loc = comp_of(p)
    if not loc or loc[0] not in refs:
        warnings.append(f"output {n} is not referenced by any parameter")

# Unique names.
taken = set()
for b in buffers:
    base, k = b["name"], 1
    while b["name"] in taken:
        b["name"] = f"{base}_{k}"
        k += 1
    taken.add(b["name"])


def buf_ref(ci, off):
    bi, delta = comp_buf[ci]
    return buffers[bi]["name"], off - delta


# ---------------------------------------------------------------- cubins and PTX


def elf_defines(path, name):
    with open(path, "rb") as f:
        d = f.read()
    return d[:4] == b"\x7fELF" and (b".text." + name.encode() + b"\x00") in d


def cubin_cluster(d, name):
    """The kernel's compiled-in cluster shape: EIATTR_CTA_PER_CLUSTER (0x3d) of .nv.info.<name>,
    or None."""
    shoff = u64(d, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x3A)
    secs = [
        struct.unpack_from("<IIQQQQIIQQ", d, shoff + i * shentsize)
        for i in range(shnum)
    ]
    want = (".nv.info." + name).encode()
    for sec in secs:
        o = secs[shstrndx][4] + sec[0]
        if d[o : d.index(b"\0", o)] != want:
            continue
        b, i = d[sec[4] : sec[4] + sec[5]], 0
        while i + 4 <= len(b):
            fmt, attr = b[i], b[i + 1]
            z = struct.unpack_from("<H", b, i + 2)[0] if fmt == 4 else 0  # EIFMT_SVAL
            if attr == 0x3D and z == 12:
                return list(struct.unpack_from("<III", b, i + 4))
            i += 4 + z
    return None


def prune(data, keep):
    """A module cubin (libtorch_cuda's hold hundreds of kernels) with the code of every entry
    kernel not in `keep` overwritten by NOPs: same layout, symbols and relocations, so it
    loads and disassembles as before, but compresses to a fraction (fixtures stay small).
    Device functions are kept (a kept kernel may call them)."""
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
    for st in (sec for sec in secs if sec[1] == 2):  # SHT_SYMTAB
        strtab = secs[st[6]][4]
        for o in range(st[4], st[4] + st[5], 24):
            n, info, other = struct.unpack_from("<IBB", d, o)
            if info & 15 == 2 and other & 0x10:  # STT_FUNC, CUDA entry
                entries.add(cstr(strtab + n))
    text = {n[6:]: sec for n, sec in zip(names, secs) if n.startswith(".text.")}
    k = text[next(iter(keep))]
    nop = bytes(d[k[4] + k[5] - 16 : k[4] + k[5]])  # ptxas pads every kernel with NOPs
    for n, sec in text.items():
        if n in entries and n not in keep and sec[1] != 8:  # not SHT_NOBITS
            d[sec[4] : sec[4] + sec[5]] = nop * (sec[5] // 16)
    return bytes(d)


images = glob.glob(f"{CUB}/*.cubin") + glob.glob(f"{IMG}/*.bin")
cubin_of = {}
for r, _ in launches:
    n = r["name"]
    if n in cubin_of:
        continue
    # CUPTI's module images are exact; launchcap's image dump (library images CUPTI may not
    # see) sizes an ELF from its headers.
    c = [p for p in images if p.startswith(CUB) and elf_defines(p, n)]
    c = c or [p for p in images if elf_defines(p, n)]
    if not c:
        sys.exit(f"capture: no captured cubin defines {n}")
    cubin_of[n] = min(c, key=os.path.getsize)

ptx_of = {}
for p in glob.glob(f"{DUMP}/**/*.ptx", recursive=True):
    src = open(p).read()
    for m in re.finditer(r"\.entry\s+([\w$]+)\s*\(", src):
        ptx_of.setdefault(m.group(1), (p, src))
names = {r["name"] for r, _ in launches}
use_ptx = (
    not A.binary
    and all(n in ptx_of for n in names)
    and len({ptx_of[n][0] for n in names}) == 1
)

out = os.path.abspath(A.out)
os.makedirs(out, exist_ok=True)
for sub in ("data", "expected"):
    shutil.rmtree(f"{out}/{sub}", ignore_errors=True)
for f in glob.glob(f"{out}/*.cubin") + glob.glob(f"{out}/*.ptx"):
    os.remove(f)
cubin_file = {}  # source path -> case-relative file
by_hash = {}
for n in [r["name"] for r, _ in launches]:
    src = cubin_of[n]
    h = hashlib.sha1(open(src, "rb").read()).hexdigest()
    if h not in by_hash:
        by_hash[h] = "jit.cubin" if use_ptx else f"k{len(by_hash)}.cubin"
        keep = {m for m in names if cubin_of[m] == src}
        data = open(src, "rb").read()
        open(f"{out}/{by_hash[h]}", "wb").write(
            data if A.no_prune else prune(data, keep)
        )
    cubin_file[n] = by_hash[h]
    # A compiled-in cluster shape (`__cluster_dims__`, routing's cluster kernel) applies when
    # the launch names none.
    cd = cubin_cluster(open(src, "rb").read(), n)
    for r, _ in launches:
        if r["name"] == n and r["cluster"] == [1, 1, 1] and cd and cd != [1, 1, 1]:
            r["cluster"] = cd
if not use_ptx and len(names) == 1 and next(iter(names)) in ptx_of:
    open(f"{out}/jit.ptx", "w").write(
        ptx_of[next(iter(names))][1]
    )  # for reference (--binary)
if use_ptx:
    src = ptx_of[next(iter(names))][1]
    # Newer JIT compilers stamp a PTX ISA version CUDA 13.0's ptxas (ptxv's) does not take;
    # the kernels use nothing past 9.0 (fixtures/r2_cute_gemm).
    m = re.search(r"^\.version (\d+)\.(\d+)", src, re.M)
    if m and (int(m.group(1)), int(m.group(2))) > (9, 0):
        src = src.replace(m.group(0), ".version 9.0", 1)
        warnings.append(f"PTX .version {m.group(1)}.{m.group(2)} rewritten to 9.0")
    open(f"{out}/kernel.ptx", "w").write(src)

# ---------------------------------------------------------------- case.toml


def q(s):
    return json.dumps(s)


def toml_tmap(prefix, ci, o, g, raw, extra=""):
    b, bo = buf_ref(ci, o)
    return (
        f'{prefix}{extra}kind = "tensormap"\nbuffer = {q(b)}\nbyte_offset = {bo}\n'
        f"box = {g['box']}\ndims = {g['dims']}\nstrides = {g['strides']}\n"
        f'swizzle = "{g["swizzle"]}"\nraw = "{raw.hex()}"\n'
        + ("elem4 = true\n" if g.get("elem4") else "")
    )


def toml_params(params, key):
    s = []
    for p in params:
        if p[0] == "ptr":
            b, bo = buf_ref(p[1], p[2])
            s.append(f'[[{key}]]\nkind = "ptr"\nbuffer = {q(b)}\nbyte_offset = {bo}\n')
        elif p[0] == "tmap":
            s.append(toml_tmap(f"[[{key}]]\n", *p[1:]))
        elif p[0] == "u32":
            s.append(f'[[{key}]]\nkind = "u32"\nvalue = {p[1]}\n')
        else:
            _, raw, fields = p
            s.append(f'[[{key}]]\nkind = "bytes"\nhex = "{raw.hex()}"\n')
            for off, f in fields:
                if f[0] == "tmap":
                    s.append(
                        toml_tmap(
                            f"[[{key}.fields]]\n", *f[1:], extra=f"offset = {off}\n"
                        )
                    )
                else:
                    b, bo = buf_ref(f[1], f[2])
                    s.append(
                        f'[[{key}.fields]]\noffset = {off}\nkind = "ptr"\nbuffer = {q(b)}\n'
                        f"byte_offset = {bo}\n"
                    )
    return s


name = A.name or os.path.basename(out.rstrip("/"))
kern_lines = "\n".join(
    f"  {i}: {r['name'][:110]} grid {r['grid']} block {r['block']}"
    + (f" cluster {r['cluster']}" if r["cluster"] != [1, 1, 1] else "")
    + (" pdl" if r.get("pdl") else "")
    for i, (r, _) in enumerate(launches)
)
r0 = launches[0][0]
cmdline = " ".join(
    a if re.fullmatch(r"[\w./:=,-]+", a) else "'" + a.replace("'", "'\\''") + "'"
    for a in sys.argv[1:]
    if not a.startswith("--work") and a != WORK
)
src_line = (
    "PTX from the JIT (kernel.ptx; its own cubin kept as jit.cubin)"
    if use_ptx
    else "binary only (cubins captured from the loaded modules)"
)
desc = f"""
Captured by tools/capture.py (torch {torch.__version__}): {what}. {len(launches)} launch(es),
{src_line}:
{kern_lines}
Regenerate: CUDA_VISIBLE_DEVICES=<gpu> $PTXV_PYTHON tools/capture.py {cmdline}"""
# A TOML basic string: backslashes (the shell quoting of --op) and quotes escaped.
desc = desc.replace("\\", "\\\\").replace('"', '\\"')
toml = [
    f'''name = {q(name)}
description = """{desc}"""
'''
    + ('ptx = "kernel.ptx"\n' if use_ptx else f'cubin = "{cubin_file[r0["name"]]}"\n')
    + f"""entry = {q(r0["name"])}
grid = {r0["grid"]}
block = {r0["block"]}
"""
    + (f"cluster = {r0['cluster']}\n" if r0["cluster"] != [1, 1, 1] else "")
    + f"""dynamic_smem = {r0["smem"]}
outputs = {json.dumps([b["name"] for b in buffers if b.get("is_out")])}
"""
]
os.makedirs(f"{out}/data", exist_ok=True)
os.makedirs(f"{out}/expected", exist_ok=True)
for b in buffers:
    s = f'[[buffers]]\nname = {q(b["name"])}\ndtype = "{b["dtype"]}"\nshape = {b["shape"]}\n'
    if "data" in b:
        open(f"{out}/data/{b['name']}.bin", "wb").write(b["data"])
        s += f'init = "file"\nfile = "data/{b["name"]}.bin"\n'
    else:
        s += 'init = "zeros"\n'
    if "expect" in b:
        open(f"{out}/expected/{b['name']}.bin", "wb").write(b["expect"])
        s += f'expect = "expected/{b["name"]}.bin"\n'
        if b["expect_offset"]:
            s += f"expect_offset = {b['expect_offset']}\n"
    toml.append(s)
toml += toml_params(launches[0][1], "params")
for r, params in launches[1:]:
    s = f"[[launches]]\nentry = {q(r['name'])}\ngrid = {r['grid']}\nblock = {r['block']}\n"
    if r["cluster"] != [1, 1, 1]:
        s += f"cluster = {r['cluster']}\n"
    s += f"dynamic_smem = {r['smem']}\n"
    if r.get("pdl"):
        s += "pdl = true\n"
    if not use_ptx and cubin_file[r["name"]] != cubin_file[r0["name"]]:
        s += f'cubin = "{cubin_file[r["name"]]}"\n'
    toml.append(s)
    toml += toml_params(params, "launches.params")
toml.append('[[variants]]\nid = "good"\nsummary = "as captured"\nexpect = ["clean"]\n')
if not any(b.get("is_out") for b in buffers):
    sys.exit(
        "capture: no output buffer (the op returned no CUDA tensor and changed none)"
    )
open(f"{out}/case.toml", "w").write("\n".join(toml))
meta = {
    "what": what,
    "argv": sys.argv[1:],
    "torch": torch.__version__,
    "launches": [
        {
            "name": r["name"],
            "grid": r["grid"],
            "block": r["block"],
            "cluster": r["cluster"],
            "smem": r["smem"],
            "pdl": r.get("pdl", 0),
            "cubin": cubin_file[r["name"]],
        }
        for r, _ in launches
    ],
    "filtered_out": dropped,
    "ptx": use_ptx,
    "buffers": [
        {"name": b["name"], "dtype": b["dtype"], "shape": b["shape"], "size": b["size"]}
        for b in buffers
    ],
    "unresolved": unresolved,
    "warnings": warnings,
}
json.dump(meta, open(f"{out}/capture.json", "w"), indent=1)
print(
    f"capture: wrote {out} ({len(buffers)} buffers, outputs {[b['name'] for b in buffers if b.get('is_out')]})"
)
for w in warnings:
    print("capture: warning:", w)
if unresolved:
    print(
        f"capture: {len(unresolved)} pointer-like parameter words not in any allocation"
    )

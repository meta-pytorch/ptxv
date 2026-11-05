# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Capture torch._grouped_mm (bf16, CUTLASS sm100 grouped GEMM inside libtorch_cuda) as a
# binary-only two-kernel case.
#   CUDA_VISIBLE_DEVICES=0 $PTXV_PYTHON capture.py [G M N K]
# Runs out[g] = A[offs[g-1]:offs[g]] @ B[g] under probes/launchcap (launch records) and
# probes/launchcap/cubincap (CUPTI: the modules the driver loads; GroupMM.cu's cubin is not
# among the ELFs cuobjdump lists). Two launches: prepare_grouped_gemm_data, which writes the
# per-group problem shapes, pointers and strides into a workspace, then the persistent
# GemmUniversal kernel. Every pointer found in the parameters is mapped to the allocation
# holding it (the caching allocator's snapshot): the inputs and output by name, the rest as
# workspace buffers W0, W1, ... sized from the allocator's history (the allocations made
# during the call); pointers into the middle of one get an offset.
import glob
import json
import os
import shutil
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LC = os.path.join(HERE, "../../probes/launchcap")
CAP = "/tmp/ptxv_launchcap_gmm.jsonl"
CUB = "/tmp/ptxv_cubincap_gmm"
if "LD_PRELOAD" not in os.environ:  # re-exec under the recorders
    lib = os.path.join(sys.prefix, "lib")
    cuda = "/usr/local/cuda-13.0"
    inc = glob.glob(f"{cuda}/targets/*/include")[0]
    subprocess.run(
        [
            "gcc",
            "-shared",
            "-fPIC",
            "-O2",
            "-o",
            "/tmp/ptxv_launchcap.so",
            f"{LC}/launchcap.c",
            "-ldl",
            f"-I{cuda}/include",
        ],
        check=True,
    )
    subprocess.run(
        [
            "gcc",
            "-shared",
            "-fPIC",
            "-O2",
            "-o",
            "/tmp/ptxv_cubincap.so",
            f"{LC}/cubincap.c",
            f"-I{inc}",
            f"-L{lib}",
            "-l:libcupti.so.13",
            f"-Wl,-rpath,{lib}",
        ],
        check=True,
    )
    if os.path.exists(CAP):
        os.remove(CAP)
    shutil.rmtree(CUB, ignore_errors=True)
    os.makedirs(CUB)
    env = dict(
        os.environ,
        LD_PRELOAD="/tmp/ptxv_launchcap.so:/tmp/ptxv_cubincap.so",
        LAUNCHCAP_OUT=CAP,
        CUBINCAP_DIR=CUB,
    )
    sys.exit(subprocess.run([sys.executable] + sys.argv, env=env).returncode)
os.chdir(HERE)

import torch

G, M, N, K = (
    int(x) for x in (sys.argv[1:5] if len(sys.argv) > 4 else (4, 128, 128, 128))
)

a = torch.randn(G * M, K, dtype=torch.bfloat16, device="cuda")
b = torch.randn(G, N, K, dtype=torch.bfloat16, device="cuda").transpose(-2, -1)
offs = torch.arange(M, G * M + 1, M, dtype=torch.int32, device="cuda")
torch.cuda.synchronize()
first = sum(1 for _ in open(CAP)) if os.path.exists(CAP) else 0
# The allocator's history gives each allocation's exact extent; its post-call snapshot does
# not (freed workspaces merge into one inactive block).
torch.cuda.memory._record_memory_history(max_entries=100000)
c = torch._grouped_mm(a, b, offs=offs)
torch.cuda.synchronize()
history = [e for tr in torch.cuda.memory._snapshot()["device_traces"] for e in tr]
torch.cuda.memory._record_memory_history(enabled=None)
ref = torch.cat([a[g * M : (g + 1) * M].float() @ b[g].float() for g in range(G)])
print("max err", (c.float() - ref).abs().max().item(), c.dtype, tuple(c.shape))

recs = [json.loads(ln) for ln in open(CAP)][first:]
recs = [r for r in recs if "grouped_gemm" in r["name"] or "GemmUniversal" in r["name"]]
assert len(recs) == 2, [r["name"][:80] for r in recs]
prep, gemm = recs
assert "prepare_grouped_gemm_data" in prep["name"], prep["name"]

# The module defining both kernels.
mods = [
    p
    for p in glob.glob(f"{CUB}/*.cubin")
    if prep["name"].encode() in open(p, "rb").read()
    and gemm["name"].encode() in open(p, "rb").read()
]
assert len(mods) == 1, mods
shutil.copy(mods[0], "kernel.cubin")

# Allocations: the named tensors, and every other block of the caching allocator.
named = {"A": a, "B": b, "offs": offs, "out": c}
blocks = [(e["addr"], e["size"]) for e in history if e["action"] == "alloc"]
for seg in torch.cuda.memory_snapshot():
    addr = seg["address"]
    for blk in seg["blocks"]:
        if blk["state"] == "active_allocated":
            blocks.append((addr, blk["size"]))
        addr += blk["size"]
names = {}  # block start -> (buffer name, dtype, shape, base pointer)
for n, t in named.items():
    p = t.data_ptr()
    blk = next(bl for bl in blocks if bl[0] <= p < bl[0] + bl[1])
    names[blk[0]] = (n, t, p)
workspaces = []


def locate(v):
    """(buffer, offset) for a device address inside a known allocation, else None."""
    for start, size in blocks:
        if start <= v < start + size:
            if start in names:
                n, t, p = names[start]
                if p <= v < p + t.untyped_storage().nbytes():
                    return n, v - p
                return None
            for i, (s, _z) in enumerate(workspaces):
                if s == start:
                    return f"W{i}", v - start
            workspaces.append((start, size))
            return f"W{len(workspaces) - 1}", v - start
    return None


def fields_of(raw):
    out = []
    for off in range(0, len(raw) - 7, 8):
        v = struct.unpack_from("<Q", raw, off)[0]
        loc = locate(v) if v else None
        if loc:
            out.append((off, loc))
    return out


launches = []
for r in (prep, gemm):
    params = []
    for p in r["params"]:
        raw = bytes.fromhex(p["hex"])
        params.append((raw, fields_of(raw)))
    launches.append((r, params))
for r, params in launches:
    print(
        r["name"][:70],
        r["grid"],
        r["block"],
        r["cluster"],
        r["smem"],
        [(len(raw), f) for raw, f in params if f],
    )

# Workspace contents are written by the first kernel; dump them anyway, to see what the
# second kernel reads that the first does not write (none expected).
dt = {torch.bfloat16: "bf16", torch.float32: "f32", torch.int32: "i32"}


def toml_params(params, key):
    s = []
    for raw, fs in params:
        s.append(f'[[{key}]]\nkind = "bytes"\nhex = "{raw.hex()}"\n')
        for off, (buf, o) in fs:
            s.append(
                f'[[{key}.fields]]\noffset = {off}\nkind = "ptr"\nbuffer = "{buf}"\nbyte_offset = {o}\n'
            )
    return s


out = [
    f'''name = "r3_grouped_mm"
description = """
Rung 3a. torch._grouped_mm (torch {torch.__version__}): out[g] = A[offs[g-1]:offs[g]] @ B[g], G={G},
M={M} rows per group, N={N}, K={K}, bf16 in and out. Binary only: the CUTLASS sm100 grouped GEMM
in libtorch_cuda (GroupMM.cu cubin via probes/launchcap/cubincap), launched after
prepare_grouped_gemm_data, which fills the workspace W* with per-group shapes, pointers and
strides. Captured by capture.py."""
cubin = "kernel.cubin"
entry = "{prep["name"]}"
grid = {prep["grid"]}
block = {prep["block"]}
dynamic_smem = {prep["smem"]}
outputs = ["out"]
'''
]
offs_vals = ", ".join(f"[{i}, {v}]" for i, v in enumerate(offs.tolist()))
for n, t in named.items():
    init, seed, extra = ("normal", {"A": 1, "B": 2}.get(n, 0), "")
    if n == "offs":
        init, extra = "zeros", f"values = [{offs_vals}]\n"
    if n == "out":
        init = "zeros"
    shape = list(t.shape) if n != "B" else [G, N, K]
    out.append(
        f'[[buffers]]\nname = "{n}"\ndtype = "{dt[t.dtype]}"\nshape = {shape}\n'
        f'init = "{init}"\nseed = {seed}\n{extra}'
    )
for i, (_s, z) in enumerate(workspaces):
    out.append(
        f'[[buffers]]\nname = "W{i}"\ndtype = "u8"\nshape = [{z}]\ninit = "zeros"\n'
    )
out += toml_params(launches[0][1], "params")
r = gemm
out.append(
    f'[[launches]]\nentry = "{r["name"]}"\ngrid = {r["grid"]}\nblock = {r["block"]}\n'
    f"cluster = {r['cluster']}\ndynamic_smem = {r['smem']}\n"
)
out += toml_params(launches[1][1], "launches.params")
out.append(
    '[[variants]]\nid = "good"\nsummary = "as shipped in PyTorch"\nexpect = ["clean"]\n'
)
open("case.toml", "w").write("\n".join(out))
print("workspaces", [(hex(s), z) for s, z in workspaces])

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Capture for rung 2 (CuTeDSL half): JIT-compiles the CUTLASS 4.6.1 CuTeDSL Blackwell
# persistent dense GEMM example (flashinfer's bundled copy) with a fixed number of persistent
# CTAs, records the real launch (probes/launchcap: grid, block, smem and every parameter's
# bytes), and writes kernel.ptx + case.toml. The three CUtensorMap parameters are matched
# against cuTensorMapEncodeTiled candidates to recover box and swizzle. Run with the conda
# python on a free GPU:
#   CUDA_VISIBLE_DEVICES=0 $PTXV_PYTHON capture.py [M N K NUM_CTAS]
import glob
import importlib.util
import itertools
import json
import os
import shutil
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LC = os.path.join(HERE, "../../probes/launchcap")
CAP = "/tmp/ptxv_launchcap.jsonl"
if "LD_PRELOAD" not in os.environ:  # re-exec under the launch recorder
    so = "/tmp/ptxv_launchcap.so"
    subprocess.run(
        [
            "gcc",
            "-shared",
            "-fPIC",
            "-O2",
            "-o",
            so,
            f"{LC}/launchcap.c",
            "-ldl",
            "-I/usr/local/cuda-13.0/include",
        ],
        check=True,
    )
    if os.path.exists(CAP):
        os.remove(CAP)
    env = dict(os.environ, LD_PRELOAD=so, LAUNCHCAP_OUT=CAP)
    sys.exit(subprocess.run([sys.executable] + sys.argv, env=env).returncode)
os.chdir(HERE)

dump = os.path.abspath("dump")  # the DSL reads these at import
shutil.rmtree(dump, ignore_errors=True)
os.environ["CUTE_DSL_DUMP_DIR"] = dump
os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
os.environ["CUTE_DSL_NO_CACHE"] = "1"
os.environ["CUTE_DSL_DISABLE_FILE_CACHING"] = "1"
os.makedirs(dump)
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch  # noqa: F401  (kept from the upstream example: loads CuTe's torch interop before the kernel is built)
import torch
from cutlass.cute.runtime import from_dlpack

# The bundled example predates 4.6.1 renames.
if not hasattr(cute, "make_fragment"):
    cute.make_fragment = cute.make_rmem_tensor
if not hasattr(cute.arch, "ProxyKind"):
    from types import SimpleNamespace as _NS

    cute.arch.ProxyKind = _NS(async_shared="async.shared", async_="async")
    cute.arch.SharedSpace = _NS(shared_cta="cta", shared_cluster="cluster")

EX = os.path.join(
    os.path.dirname(__import__("flashinfer").__file__),
    "data/cutlass/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py",
)
spec = importlib.util.spec_from_file_location("dgp", EX)
dgp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dgp)

# Optional pipeline depths (CUTE_STAGES="ab,acc,c"): a shallow ring wraps within a small problem,
# so parity / phase reuse is exercised at a grid ptxv can explore.
if os.environ.get("CUTE_STAGES"):
    _st = tuple(int(x) for x in os.environ["CUTE_STAGES"].split(","))
    _orig = dgp.PersistentDenseGemmKernel._compute_stages

    def _stages(*a, **k):
        acc, ab, c = _orig(*a, **k)
        print(
            f"stages: computed acc={acc} ab={ab} c={c}; using ab={_st[0]} acc={_st[1]} c={_st[2]}"
        )
        return _st[1], _st[0], _st[2]

    dgp.PersistentDenseGemmKernel._compute_stages = staticmethod(_stages)

M, N, K, NCTA = (
    int(x) for x in (sys.argv[1:5] if len(sys.argv) > 4 else (512, 128, 128, 2))
)

torch.manual_seed(1)
a = (
    torch.randn(1, M, K, dtype=torch.bfloat16).permute(1, 2, 0).cuda()
)  # (M, K, 1), K-major
b = torch.randn(1, N, K, dtype=torch.bfloat16).permute(1, 2, 0).cuda()  # (N, K, 1)
c = torch.zeros(1, M, N, dtype=torch.float32).permute(1, 2, 0).cuda()  # (M, N, 1)


def ct(t):
    x = from_dlpack(t, assumed_align=16)
    return x.mark_layout_dynamic(leading_dim=1)


gemm = dgp.PersistentDenseGemmKernel(cutlass.Float32, False, (128, 128), (1, 1), True)
stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
f = cute.compile(gemm, ct(a), ct(b), ct(c), NCTA, stream)
f(ct(a), ct(b), ct(c), stream)
torch.cuda.synchronize()
ref = torch.einsum("mkl,nkl->mnl", a.float(), b.float())
print("max err", (c - ref).abs().max().item())
assert (c - ref).abs().max().item() < 1e-3
# CuTeDSL's bundled compiler emits PTX 9.3; CUDA 13.0's ptxas (which ptxv uses) takes 9.0,
# and the kernel uses nothing newer.
src = (
    open(glob.glob(dump + "/*.ptx")[0])
    .read()
    .replace(".version 9.3", ".version 9.0", 1)
)
open("kernel.ptx", "w").write(src)
entry = (
    [ln for ln in open("kernel.ptx") if ln.startswith(".visible .entry")][0]
    .split()[2]
    .rstrip("(")
)
rec = [r for r in map(json.loads, open(CAP)) if r["name"] == entry]
assert len(rec) == 1, f"{len(rec)} launches of {entry}"
rec = rec[0]

# Recover each tensor map's box / swizzle by re-encoding candidates over the same tensor.
D = cuda
SWZ = {
    "none": D.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    "32B": D.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
    "64B": D.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
    "128B": D.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
}
L2 = [
    getattr(D.CUtensorMapL2promotion, f"CU_TENSOR_MAP_L2_PROMOTION_{x}")
    for x in ("NONE", "L2_64B", "L2_128B", "L2_256B")
]
OOB = [
    D.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    D.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA,
]
DT = {
    torch.bfloat16: D.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    torch.float32: D.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
}


def match_tmap(hexs, t, name):
    got = bytes.fromhex(hexs)[:64]  # the encoder leaves bytes 64..127 unspecified
    # t is (rows, inner, 1) after permute: dims innermost first, strides of dims 1.. in bytes
    rows, inner = t.shape[0], t.shape[1]
    es = t.element_size()
    dims = [inner, rows, 1]
    strides = [inner * es, rows * inner * es]
    sizes = [8, 16, 32, 64, 128, 256]
    # The DSL builds descriptors itself and sets two bits of word 2 (0x40002) that
    # cuTensorMapEncodeTiled never sets; compare without them, and keep the raw bytes for the
    # hardware run (case `raw`).
    mask = (
        lambda b: b[:8]
        + struct.pack("<I", struct.unpack("<I", b[8:12])[0] & ~0x40002)
        + b[12:]
    )
    got = mask(got)
    for b0, b1, (sn, sw), l2, oob in itertools.product(
        sizes, sizes, SWZ.items(), L2, OOB
    ):
        u64, u32 = D.cuuint64_t, D.cuuint32_t
        err, tm = D.cuTensorMapEncodeTiled(
            DT[t.dtype],
            3,
            t.data_ptr(),
            [u64(x) for x in dims],
            [u64(x) for x in strides],
            [u32(b0), u32(b1), u32(1)],
            [u32(1)] * 3,
            D.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            sw,
            l2,
            oob,
        )
        if err != D.CUresult.CUDA_SUCCESS:
            continue
        mine = b"".join(int(x).to_bytes(8, "little") for x in tm.opaque)[:64]
        if mask(mine) == got:
            return {
                "buffer": name,
                "box": [b0, b1, 1],
                "dims": dims,
                "strides": strides,
                "swizzle": sn,
                "raw": hexs[:128],
            }
    raise SystemExit(f"no tensor map candidate matches {name}")


ps = rec["params"]
tmaps = {
    1: match_tmap(ps[1]["hex"], a, "A"),
    3: match_tmap(ps[3]["hex"], b, "B"),
    5: match_tmap(ps[5]["hex"], c, "C"),
}
print("tensor maps:", tmaps)

out = [
    f'''name = "r2_cute_gemm"
description = """
Rung 2. CUTLASS 4.6.1 CuTeDSL Blackwell persistent dense GEMM (examples/python/CuTeDSL/
blackwell/dense_gemm_persistent.py, TMA load + tcgen05.mma + TMEM epilogue + TMA store),
C[{M}x{N}] f32 = A[{M}x{K}] bf16 @ B[{N}x{K}]^T, {NCTA} persistent CTAs, 6 warps
(TMA, MMA, 4 epilogue). Captured by capture.py: the launch and parameter bytes are recorded."""
ptx = "kernel.ptx"
entry = "{entry}"
grid = {rec["grid"]}
block = {rec["block"]}
dynamic_smem = {rec["smem"]}
outputs = ["C"]
spec = "gemm_tn"
'''
]
for n, dt, shape, init, seed in (
    ("A", "bf16", [M, K], "normal", 1),
    ("B", "bf16", [N, K], "normal", 2),
    ("C", "f32", [M, N], "zeros", 0),
):
    out.append(
        f'[[buffers]]\nname = "{n}"\ndtype = "{dt}"\nshape = {shape}\ninit = "{init}"\nseed = {seed}\n'
    )
for i, p in enumerate(ps):
    if i in tmaps:
        t = tmaps[i]
        out.append(
            f'[[params]]\nkind = "tensormap"\nbuffer = "{t["buffer"]}"\nbox = {t["box"]}\n'
            f'dims = {t["dims"]}\nstrides = {t["strides"]}\nswizzle = "{t["swizzle"]}"\n'
            f'raw = "{t["raw"]}"\n'
        )
    else:
        out.append(f'[[params]]\nkind = "bytes"\nhex = "{p["hex"]}"\n')
out.append(
    '[[variants]]\nid = "good"\nsummary = "as compiled by CuTeDSL"\nexpect = ["clean"]\n'
)
open("case.toml", "w").write("\n".join(out))
print("captured", entry[:60], rec["grid"], rec["block"], rec["smem"])

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Capture of the CUTLASS 4.6.1 CuTeDSL Blackwell persistent block-scaled dense GEMM example
# (dense_blockscaled_gemm_persistent.py, flashinfer's bundled copy) for NVFP4 (A, B e2m1, two
# per byte, low nibble first; ue4m3 scales per 16 K elements; C f32), as r3_cute_mxf4's,
# records the real launch (probes/launchcap) and writes kernel.ptx, the input files and
# case.toml. Tensor maps are recovered by re-encoding candidates (as in r2_cute_small).
#   CUDA_VISIBLE_DEVICES=<free gpu> $PTXV_PYTHON capture.py [M N K [2cta OUTDIR]]
# 2cta: mma tiler (256, 128), cluster (2, 1) (tcgen05 cta_group::2), written to OUTDIR. Not a
# fixture: its PTX converts shared addresses through the generic window (cvta.shared, SASS
# SR_SWINHI), which ptxv does not model.
import glob
import importlib.util
import itertools
import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LC = os.path.join(HERE, "../../probes/launchcap")
CAP = "/tmp/ptxv_launchcap_r3nv4.jsonl"
if "LD_PRELOAD" not in os.environ:  # re-exec under the launch recorder
    so = "/tmp/ptxv_launchcap_r3nv4.so"
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
M, N, K = (int(x) for x in (sys.argv[1:4] if len(sys.argv) > 3 else (128, 128, 256)))
PAIR = len(sys.argv) > 4 and sys.argv[4] == "2cta"
OUTDIR = os.path.abspath(sys.argv[5]) if PAIR else HERE
NAME = os.path.basename(OUTDIR)
os.makedirs(OUTDIR, exist_ok=True)
os.chdir(OUTDIR)

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
    "data/cutlass/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py",
)
spec = importlib.util.spec_from_file_location("dbgp", EX)
dbgp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dbgp)

assert M % (256 if PAIR else 128) == 0 and N % 128 == 0 and K % 128 == 0
VEC = 16
SK = K // VEC

# Inputs: uniform e2m1 codes, ue4m3 scales 2^-3 .. 1.875 x 2^3 (codes 0x20..0x57, none 0).
rng = np.random.default_rng(32)
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
ac = rng.integers(0, 16, (M, K)).astype(np.uint8)
bc = rng.integers(0, 16, (N, K)).astype(np.uint8)


def pack(c):
    return c[:, 0::2] | c[:, 1::2] << 4


sfa_mk = rng.integers(0x20, 0x58, (M, SK)).astype(np.uint8)
sfb_nk = rng.integers(0x20, 0x58, (N, SK)).astype(np.uint8)


def sf_atom(mk):
    # (mn, sk) -> the M32x4xrm_K4xrk_L byte layout the kernel reads:
    # offset = ((mn/128 * rk + sk/4) * 32 + mn%32) * 16 + (mn/32)%4 * 4 + sk%4
    mn, sk = mk.shape
    rk = sk // 4
    out = np.zeros(mn * sk, np.uint8)
    for r in range(mn):
        for k in range(sk):
            off = (
                (((r // 128) * rk + k // 4) * 32 + r % 32) * 16
                + (r // 32) % 4 * 4
                + k % 4
            )
            out[off] = mk[r, k]
    return out


def ue4m3(c):
    e, m = (c >> 3).astype(np.float32), (c & 7).astype(np.float32)
    v = np.where(e == 0, m * 2.0**-9, (1 + m / 8) * 2.0 ** (e - 7))
    return v.astype(np.float32)


sfa_b, sfb_b = sf_atom(sfa_mk), sf_atom(sfb_nk)
pack(ac).tofile("A.bin")
pack(bc).tofile("B.bin")
sfa_b.tofile("SFA.bin")
sfb_b.tofile("SFB.bin")

# Device tensors in the example's layouts, as uint8 storage viewed with the cute element type
# (cutlass.torch.cute_tensor_like): A (M, K, 1) K-major, B (N, K, 1), C (M, N, 1), SF
# (L, rm, rk, 32, 4, 4) contiguous permuted to (32, 4, rm, 4, rk, L).
from cutlass.torch import get_leading_dim


# FP4 storage as cutlass.torch.cute_tensor_like makes it: a u8 (M, K, 1) tensor whose first
# M K / 2 bytes hold the packed codes (the cute tensor's strides count 4-bit elements).
def fp4_dev(c):
    t = torch.zeros(c.shape[0], c.shape[1], 1, dtype=torch.uint8)
    t.view(-1)[: c.size // 2] = torch.from_numpy(pack(c).reshape(-1))
    return t.cuda()


a = fp4_dev(ac)
b = fp4_dev(bc)
c = torch.zeros(1, M, N, dtype=torch.float32).permute(1, 2, 0).cuda()
rk = SK // 4
sfa = (
    torch.from_numpy(sfa_b)
    .cuda()
    .view(1, M // 128, rk, 32, 4, 4)
    .permute(3, 4, 1, 5, 2, 0)
)
sfb = (
    torch.from_numpy(sfb_b)
    .cuda()
    .view(1, N // 128, rk, 32, 4, 4)
    .permute(3, 4, 1, 5, 2, 0)
)


def ct(t, ty, div=None):
    x = from_dlpack(t, assumed_align=16)
    x.element_type = ty
    if div is None:
        return x.mark_layout_dynamic(leading_dim=get_leading_dim(t))
    return x.mark_compact_shape_dynamic(
        mode=1, stride_order=(2, 0, 1), divisibility=div
    )


args = (
    ct(a, cutlass.Float4E2M1FN, 2),
    ct(b, cutlass.Float4E2M1FN, 2),
    ct(sfa, cutlass.Float8E4M3FN),
    ct(sfb, cutlass.Float8E4M3FN),
    ct(c, cutlass.Float32, 1),
)
gemm = dbgp.Sm100BlockScaledPersistentDenseGemmKernel(
    VEC, (256, 128) if PAIR else (128, 128), (2, 1) if PAIR else (1, 1)
)
stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
f = cute.compile(gemm, *args, 1, stream)
f(*args, stream)
torch.cuda.synchronize()
af = torch.from_numpy(E2M1[ac]).float() * torch.from_numpy(
    ue4m3(sfa_mk)
).repeat_interleave(VEC, 1)
bf = torch.from_numpy(E2M1[bc]).float() * torch.from_numpy(
    ue4m3(sfb_nk)
).repeat_interleave(VEC, 1)
ref = (af.double() @ bf.double().T).float()
got = c[:, :, 0].cpu()
print("max rel err", ((got - ref).abs().max() / ref.abs().max()).item())
assert ((got - ref).abs().max() / ref.abs().max()).item() < 1e-5
got.numpy().tofile("gb300_C.bin")
# CuTeDSL's bundled compiler emits PTX 9.3; CUDA 13.0's ptxas (which ptxv uses) takes 9.0.
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
json.dump(rec, open(os.path.join(dump, "launch.json"), "w"), indent=1)
print(
    "captured",
    entry[:60],
    rec["grid"],
    rec["block"],
    rec["smem"],
    [len(p["hex"]) // 2 for p in rec["params"]],
)

# Recover each tensor map's box / swizzle by re-encoding candidates over the same geometry.
# CuTe drops the L = 1 mode (rank 2) and loads the scale factors as u16 rows of one
# 512-byte atom (4 K-blocks x 128 rows).
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
    1: D.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    2: D.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
    4: D.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    "u4": D.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
}


def match_tmap(hexs, ptr, es, dims, strides, name):
    # The DSL sets two bits of word 2 (0x40002) that cuTensorMapEncodeTiled never sets;
    # compare without them, and keep the raw bytes for the hardware run (case `raw`).
    mask = (
        lambda b: b[:8]
        + struct.pack("<I", struct.unpack("<I", b[8:12])[0] & ~0x40002)
        + b[12:64]
    )
    got = mask(bytes.fromhex(hexs))
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for bx, (sn, sw), l2, oob in itertools.product(
        itertools.product(sizes, repeat=len(dims)), SWZ.items(), L2, OOB
    ):
        u64, u32 = D.cuuint64_t, D.cuuint32_t
        err, tm = D.cuTensorMapEncodeTiled(
            DT[es],
            len(dims),
            ptr,
            [u64(x) for x in dims],
            [u64(x) for x in strides],
            [u32(x) for x in bx],
            [u32(1)] * len(dims),
            D.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            sw,
            l2,
            oob,
        )
        if err != D.CUresult.CUDA_SUCCESS:
            continue
        mine = b"".join(int(x).to_bytes(8, "little") for x in tm.opaque)[:64]
        if mask(mine) == got:
            if es == "u4":  # the case gives a 4-bit map in bytes (elem4)
                return {
                    "buffer": name,
                    "box": [bx[0] // 2] + list(bx[1:]),
                    "dims": [dims[0] // 2] + dims[1:],
                    "strides": strides,
                    "swizzle": sn,
                    "raw": hexs[:128],
                    "elem4": True,
                }
            return {
                "buffer": name,
                "box": list(bx),
                "dims": dims,
                "strides": strides,
                "swizzle": sn,
                "raw": hexs[:128],
            }
    raise SystemExit(f"no tensor map candidate matches {name}: {hexs[:128]}")


ps = rec["params"]


def sf_geom(hexs, mn):
    # Rank 2 (256, mn / 128 * rk), or rank 3 (256, rk, mn / 128) (the 2-CTA kernel's).
    if (struct.unpack("<I", bytes.fromhex(hexs)[8:12])[0] >> 4) & 7 == 2:
        return [256, rk, mn // 128], [512, 512 * rk]
    return [256, mn // 128 * rk], [512]


tmaps = {
    2: match_tmap(ps[2]["hex"], a.data_ptr(), "u4", [K, M], [K // 2], "A"),
    4: match_tmap(ps[4]["hex"], b.data_ptr(), "u4", [K, N], [K // 2], "B"),
    6: match_tmap(ps[6]["hex"], sfa.data_ptr(), 2, *sf_geom(ps[6]["hex"], M), "SFA"),
    8: match_tmap(ps[8]["hex"], sfb.data_ptr(), 2, *sf_geom(ps[8]["hex"], N), "SFB"),
    10: match_tmap(ps[10]["hex"], c.data_ptr(), 4, [N, M], [N * 4], "C"),
}
print("tensor maps:", {k: (v["box"], v["swizzle"]) for k, v in tmaps.items()})

out = [
    f'''name = "{NAME}"
description = """
CUTLASS 4.6.1 CuTeDSL Blackwell persistent block-scaled dense GEMM
(examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py: TMA loads of A, B
(4-bit tensor maps) and the scale factors, tcgen05.cp of the scale factors into TMEM,
block-scaled FP4 tcgen05.mma, TMEM epilogue, TMA store), NVFP4{" (2-CTA pairs)" if PAIR else ""}:
C[{M}x{N}] f32 = (A * SFA) @ (B * SFB)^T, A[{M}x{K}] / B[{N}x{K}] e2m1 (two per byte, low
nibble first), ue4m3 scales per 16 K elements (none 0), {"a 2-CTA cluster" if PAIR else "one persistent CTA"}, 6 warps (TMA, MMA, 4
epilogue). Captured by capture.py; the inputs are fixed files and gb300_C.bin is the GB300
output."""
ptx = "kernel.ptx"
entry = "{entry}"
grid = {rec["grid"]}
{"cluster = " + str(rec["cluster"]) + chr(10) if PAIR else ""}block = {rec["block"]}
dynamic_smem = {rec["smem"]}
outputs = ["C"]
'''
]
for n, dt, shape, init in (
    ("A", "u8", [M, K // 2], "A.bin"),
    ("B", "u8", [N, K // 2], "B.bin"),
    ("SFA", "u16", [M // 128 * rk, 256], "SFA.bin"),
    ("SFB", "u16", [N // 128 * rk, 256], "SFB.bin"),
    ("C", "f32", [M, N], None),
):
    init = f'init = "file"\nfile = "{init}"' if init else 'init = "zeros"'
    out.append(f'[[buffers]]\nname = "{n}"\ndtype = "{dt}"\nshape = {shape}\n{init}\n')
for i, p in enumerate(ps):
    if i in tmaps:
        t = tmaps[i]
        out.append(
            f'[[params]]\nkind = "tensormap"\nbuffer = "{t["buffer"]}"\nbox = {t["box"]}\n'
            f'dims = {t["dims"]}\nstrides = {t["strides"]}\nswizzle = "{t["swizzle"]}"\n'
            f'raw = "{t["raw"]}"\n' + ("elem4 = true\n" if t.get("elem4") else "")
        )
    else:
        out.append(f'[[params]]\nkind = "bytes"\nhex = "{p["hex"]}"\n')
out.append(
    '[[variants]]\nid = "good"\nsummary = "as compiled by CuTeDSL"\nexpect = ["race", "violation"]\nexpect_sass = ["clean"]\n'
)
open("case.toml", "w").write("\n".join(out))

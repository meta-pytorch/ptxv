# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# GB300 trials of the MXF8 GEMM pairs (fixtures/c_equiv_mxf8_cublas, c_equiv_mxf8_triton;
# docs/equiv-mma.md §5): the CuTeDSL block-scaled GEMM (fixtures/r3_cute_mxf8) against
# torch._scaled_mm (cuBLASLt, fixtures/c_smm_mxf8) and the Triton tl.dot_scaled GEMM in K order
# and K-reversed (fixtures/c_triton_mxf8, c_triton_mxf8_rev), on the same random inputs (as `ptxv equiv` draws them: e4m3 codes but
# NaN, subnormals and -0 included; ue8m0 scales 2^-8 .. 2^7). Writes, per trial, the inputs
# (one f32 per byte code) and both outputs for PTXV_EQUIV_HW_TRIALS, and prints the bitwise
# comparison. --fp4: the MXF4 pair instead (fixtures/c_equiv_mxf4_triton: r3_cute_mxf4 against
# the Triton e2m1 kernels; random e2m1 bytes, the same scales). --nvf4: the NVFP4 pairs
# (fixtures/c_equiv_nvf4_cublas, c_equiv_nvf4_dq: r3_cute_nvf4 against torch._scaled_mm,
# fixtures/c_smm_nvf4, and the dequantize-then-bf16 Triton GEMM, fixtures/c_triton_nvf4_dq;
# random e2m1 bytes, ue4m3 scales per 16: any finite nonzero code, subnormals included).
#   CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/mx_gpu_check.py --trials 2000 --dump /tmp/e-mx
#   CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/mx_gpu_check.py --fp4 --trials 1000 --dump /tmp/e-mx
#   CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/mx_gpu_check.py --nvf4 --trials 1000 --dump /tmp/e-mx
#   PTXV_EQUIV_HW_TRIALS=/tmp/e-mx ptxv equiv fixtures/c_equiv_mxf8_cublas/equiv.toml --no-hw
#   PTXV_EQUIV_HW_TRIALS=/tmp/e-mx ptxv equiv fixtures/c_equiv_mxf8_triton/equiv.toml --no-hw
import argparse
import importlib.util
import os

import numpy as np
import torch

os.environ.setdefault("CUTE_DSL_NO_CACHE", "1")
# Triton's bundled ptxas does not know sm_103a (as in tools/capture.py).
for _t in ("ptxas", "cuobjdump", "nvdisasm"):
    os.environ.setdefault(f"TRITON_{_t.upper()}_PATH", f"/usr/local/cuda-13.0/bin/{_t}")
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.torch import get_leading_dim

# The bundled example predates 4.6.1 renames (as in fixtures/r3_cute_mxf8/capture.py).
if not hasattr(cute, "make_fragment"):
    cute.make_fragment = cute.make_rmem_tensor
if not hasattr(cute.arch, "ProxyKind"):
    from types import SimpleNamespace as _NS

    cute.arch.ProxyKind = _NS(async_shared="async.shared", async_="async")
    cute.arch.SharedSpace = _NS(shared_cta="cta", shared_cluster="cluster")

M = N = 128
K = 256


def cute_gemm(fp4=False, nv=False):
    vec = 16 if nv else 32
    SK = K // vec
    RK = SK // 4
    ex = os.path.join(
        os.path.dirname(__import__("flashinfer").__file__),
        "data/cutlass/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py",
    )
    spec = importlib.util.spec_from_file_location("dbgp", ex)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    # FP4: the packed codes in the first M K / 2 bytes of a u8 (M, K, 1) tensor (as
    # fixtures/r3_cute_mxf4/capture.py).
    a = torch.zeros(M, K, 1, dtype=torch.uint8, device="cuda")
    b = torch.zeros(N, K, 1, dtype=torch.uint8, device="cuda")
    ab = (cutlass.Float4E2M1FN, 2) if fp4 else (cutlass.Float8E4M3FN, 1)
    c = torch.zeros(1, M, N, dtype=torch.float32).permute(1, 2, 0).cuda()
    sfa = torch.zeros(M * SK, dtype=torch.uint8, device="cuda")
    sfb = torch.zeros(N * SK, dtype=torch.uint8, device="cuda")

    def perm(s, mn):
        return s.view(1, mn // 128, RK, 32, 4, 4).permute(3, 4, 1, 5, 2, 0)

    sft = cutlass.Float8E4M3FN if nv else cutlass.Float8E8M0FNU

    def ct(t, ty, div=None):
        x = from_dlpack(t, assumed_align=16)
        x.element_type = ty
        if div is None:
            return x.mark_layout_dynamic(leading_dim=get_leading_dim(t))
        return x.mark_compact_shape_dynamic(
            mode=1, stride_order=(2, 0, 1), divisibility=div
        )

    args = (
        ct(a, *ab),
        ct(b, *ab),
        ct(perm(sfa, M), sft),
        ct(perm(sfb, N), sft),
        ct(c, cutlass.Float32, 1),
    )
    gemm = m.Sm100BlockScaledPersistentDenseGemmKernel(vec, (128, 128), (1, 1))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    f = cute.compile(gemm, *args, 1, stream)

    def run(A, B, SFA, SFB):
        if fp4:
            a.view(-1)[: A.numel()].copy_(A.view(-1))
            b.view(-1)[: B.numel()].copy_(B.view(-1))
        else:
            a[:, :, 0].copy_(A)
            b[:, :, 0].copy_(B)
        sfa.copy_(SFA)
        sfb.copy_(SFB)
        c.zero_()
        f(*args, stream)
        return c[:, :, 0].clone()

    return run


def cublas(A, B, SFA, SFB):
    e4, e8 = torch.float8_e4m3fn, torch.float8_e8m0fnu
    return torch._scaled_mm(
        A.view(e4), B.view(e4).t(), SFA.view(e8), SFB.view(e8), out_dtype=torch.float32
    )


def cublas_nvf4(A, B, SFA, SFB):
    f4, e4 = torch.float4_e2m1fn_x2, torch.float8_e4m3fn
    return torch._scaled_mm(
        A.view(f4), B.view(f4).t(), SFA.view(e4), SFB.view(e4), out_dtype=torch.float32
    )


def triton_nvf4_dq():
    tm = importlib.util.spec_from_file_location(
        "tdq",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "capture_examples",
            "triton_nvf4_dequant.py",
        ),
    )
    m = importlib.util.module_from_spec(tm)
    tm.loader.exec_module(m)
    return m.run


def triton(rev, bk, fp4=False):
    src = "triton_mxf4.py" if fp4 else "triton_mxf8.py"
    tm = importlib.util.spec_from_file_location(
        "tmx",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "capture_examples", src
        ),
    )
    m = importlib.util.module_from_spec(tm)
    tm.loader.exec_module(m)

    def run(A, B, SFA, SFB):
        C = torch.empty(M, N, dtype=torch.float32, device=A.device)
        if fp4:
            m.mxf4_kernel[(1,)](
                A,
                B,
                SFA,
                SFB,
                C,
                M=M,
                N=N,
                K=K,
                BK=bk,
                REV=rev,
                num_warps=4,
                num_stages=1,
            )
        else:
            e4 = torch.float8_e4m3fn
            m.mxf8_kernel[(1,)](
                A.view(e4),
                B.view(e4),
                SFA,
                SFB,
                C,
                M=M,
                N=N,
                K=K,
                BK=bk,
                REV=rev,
                num_warps=4,
                num_stages=1,
            )
        return C

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--dump")
    ap.add_argument("--fp4", action="store_true")
    ap.add_argument("--nvf4", action="store_true")
    a = ap.parse_args()
    a.fp4 |= a.nvf4
    ref = cute_gemm(a.fp4, a.nvf4)
    SK = K // (16 if a.nvf4 else 32)
    if a.nvf4:
        cands = {"c_smm_nvf4": cublas_nvf4, "c_triton_nvf4_dq": triton_nvf4_dq()}
    elif a.fp4:
        cands = {
            "c_triton_mxf4": triton(0, 256, True),
            "c_triton_mxf4_rev": triton(1, 64, True),
        }
    else:
        cands = {
            "c_smm_mxf8": cublas,
            "c_triton_mxf8": triton(0, 128),
            "c_triton_mxf8_rev": triton(1, 32),
        }
    if a.dump:
        for n in cands:
            os.makedirs(os.path.join(a.dump, n), exist_ok=True)
    stats = {n: [0, 0, 0.0] for n in cands}
    rng = np.random.default_rng(7)
    codes = (
        np.arange(256, dtype=np.uint8)
        if a.fp4
        else np.array([c for c in range(256) if c & 0x7F != 0x7F], np.uint8)
    )
    kb = K // 2 if a.fp4 else K
    sfr = (1, 127) if a.nvf4 else (119, 135)
    for t in range(a.trials):
        ins = {
            "A": codes[rng.integers(0, len(codes), (M, kb))],
            "B": codes[rng.integers(0, len(codes), (N, kb))],
            "SFA": rng.integers(*sfr, M * SK).astype(np.uint8),
            "SFB": rng.integers(*sfr, N * SK).astype(np.uint8),
        }
        g = {k: torch.from_numpy(v).cuda() for k, v in ins.items()}
        r = ref(**g).cpu().numpy()
        for name, cand in cands.items():
            c = cand(**g).cpu().numpy()
            ne = r.view(np.uint32) != c.view(np.uint32)
            z = ne & (r == 0) & (c == 0)
            st = stats[name]
            st[0] += int((ne & ~z).sum())
            st[1] += int(z.sum())
            st[2] = max(st[2], float(np.abs(r.astype(np.float64) - c).max()))
            if a.dump:
                d = os.path.join(a.dump, name)
                for k, v in ins.items():
                    v.astype(np.float32).tofile(os.path.join(d, f"t{t}.{k}.bin"))
                r.tofile(os.path.join(d, f"t{t}.ref.bin"))
                c.tofile(os.path.join(d, f"t{t}.cand.bin"))
    n = a.trials * M * N
    for name, (diff, zsign, mx) in stats.items():
        print(
            f"{name}: {a.trials} trials, {n} outputs: {diff} differ in value (max |C - R| {mx:g}), "
            f"{zsign} only in the sign of zero"
        )


if __name__ == "__main__":
    main()

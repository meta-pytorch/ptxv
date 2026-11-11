# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NVFP4 GEMM by dequantization (docs/equiv-mma.md §5): each e2m1 code and its ue4m3 scale
# are converted to f16 by cvt (`cvt.rn.f16x2.e2m1x2`; `cvt.rn.f16x2.e4m3x2` of code & 0x7f;
# both exact), widened to f32, multiplied (f32), rounded to bf16, and a bf16 tl.dot accumulates in
# f32: C[128, 128] f32 = bf16(A * SFA) @ bf16(B * SFB)^T. Same bytes and names as
# fixtures/r3_cute_nvf4 (A / B [128, 256] e2m1 K-major, two codes per byte, low nibble first;
# ue4m3 scales per 16 K in CUTLASS's 128x4 scale atom).
#   tools/capture.py --file tools/capture_examples/triton_nvf4_dequant.py --binary --out DIR
import os

import numpy as np
import torch
import triton
import triton.language as tl

FIX = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../fixtures/r3_cute_nvf4"
)
M = N = 128
K = 256


@triton.jit
def _f16x2_e2m1(x):
    # Two e2m1 codes (one byte) -> f16x2, the low nibble in the low half.
    return tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.u8.u32 t, $1; cvt.rn.f16x2.e2m1x2 $0, t; }",
        "=r,r",
        [x.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f16x2_e4m3(x):
    # A ue4m3 code in the low byte -> f16 in the low half: e4m3 of (code & 0x7f) is the
    # code's ue4m3 value (ue4m3 has no sign bit; bit 7 is ignored).
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.u16.u32 t, $1; cvt.rn.f16x2.e4m3x2 $0, t; }",
        "=r,r",
        [x.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _lo(u):
    return (u & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _hi(u):
    return (u >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _dequant(
    c_ptr, s_ptr, rows, k0, K: tl.constexpr, R: tl.constexpr, BK: tl.constexpr
):
    RK: tl.constexpr = K // 64
    ob = k0 // 2 + tl.arange(0, BK // 2)
    u = _f16x2_e2m1(tl.load(c_ptr + rows[:, None] * (K // 2) + ob[None, :]))
    # Scale (r, s), s = 16-element block: byte ((r/128 * RK + s/4) * 32 + r%32) * 16
    # + (r/32)%4 * 4 + s%4; one per code pair (block of the pair's first element).
    s = (k0 + 2 * tl.arange(0, BK // 2)) // 16
    sc = tl.load(
        s_ptr
        + (((rows[:, None] // 128) * RK + s[None, :] // 4) * 32 + rows[:, None] % 32)
        * 16
        + (rows[:, None] // 32) % 4 * 4
        + s[None, :] % 4
    )
    f = _lo(_f16x2_e4m3(sc & 0x7F))
    lo = (_lo(u) * f).to(tl.bfloat16)
    hi = (_hi(u) * f).to(tl.bfloat16)
    return tl.join(lo, hi).reshape(R, BK)


@triton.jit
def nvf4_dq_kernel(
    a_ptr,
    b_ptr,
    sfa_ptr,
    sfb_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    om = tl.arange(0, M)
    on = tl.arange(0, N)
    acc = tl.zeros([M, N], dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = _dequant(a_ptr, sfa_ptr, om, k0, K, M, BK)
        b = _dequant(b_ptr, sfb_ptr, on, k0, K, N, BK)
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc)


def inputs():
    def rd(f):
        return torch.from_numpy(np.fromfile(os.path.join(FIX, f), np.uint8)).cuda()

    return {
        "A": rd("A.bin").view(M, K // 2),
        "B": rd("B.bin").view(N, K // 2),
        "SFA": rd("SFA.bin"),
        "SFB": rd("SFB.bin"),
    }


def run(A, B, SFA, SFB):
    C = torch.empty(M, N, dtype=torch.float32, device=A.device)
    nvf4_dq_kernel[(1,)](
        A, B, SFA, SFB, C, M=M, N=N, K=K, BK=64, num_warps=4, num_stages=1
    )
    return C

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MXF4 GEMM in Triton (tl.dot_scaled, "e2m1"): C[128, 128] f32 = (A * SFA) @ (B * SFB)^T, A / B
# e2m1 [128, 256] K-major, two codes per byte (low nibble first), ue8m0 scales per 32 K read
# from CUTLASS's 128x4 scale atom (the bytes and names of fixtures/r3_cute_mxf4). Triton 3.5
# lowers it to tcgen05.mma.kind::mxf4.block_scale.scale_vec::2X. One program, K blocks of BK;
# PTXV_MX_REV=1 walks them backwards. Candidates of fixtures/c_equiv_mxf4_triton.
#   tools/capture.py --file tools/capture_examples/triton_mxf4.py --binary --out DIR
import os

import numpy as np
import torch
import triton
import triton.language as tl

FIX = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../fixtures/r3_cute_mxf4"
)
M = N = 128
K = 256


@triton.jit
def mxf4_kernel(
    a_ptr,
    b_ptr,
    sfa_ptr,
    sfb_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    REV: tl.constexpr,
):
    RK: tl.constexpr = K // 128
    om = tl.arange(0, M)
    on = tl.arange(0, N)
    acc = tl.zeros([M, N], dtype=tl.float32)
    for kb in range(0, K, BK):
        k0 = K - BK - kb if REV else kb
        ok = k0 // 2 + tl.arange(0, BK // 2)
        a = tl.load(a_ptr + om[:, None] * (K // 2) + ok[None, :])
        b = tl.load(b_ptr + on[None, :] * (K // 2) + ok[:, None])
        # Scale (r, s), s = K block: byte ((r/128 * RK + s/4) * 32 + r%32) * 16 + (r/32)%4 * 4 + s%4.
        s = k0 // 32 + tl.arange(0, BK // 32)
        sa = tl.load(
            sfa_ptr
            + (((om[:, None] // 128) * RK + s[None, :] // 4) * 32 + om[:, None] % 32)
            * 16
            + (om[:, None] // 32) % 4 * 4
            + s[None, :] % 4
        )
        sb = tl.load(
            sfb_ptr
            + (((on[:, None] // 128) * RK + s[None, :] // 4) * 32 + on[:, None] % 32)
            * 16
            + (on[:, None] // 32) % 4 * 4
            + s[None, :] % 4
        )
        acc = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", acc)
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
    mxf4_kernel[(1,)](
        A,
        B,
        SFA,
        SFB,
        C,
        M=M,
        N=N,
        K=K,
        BK=int(os.environ.get("PTXV_MX_BK", "256")),
        REV=int(os.environ.get("PTXV_MX_REV", "0")),
        num_warps=4,
        num_stages=1,
    )
    return C

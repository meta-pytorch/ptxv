# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MXF8 GEMM in Triton (tl.dot_scaled): C[128, 128] f32 = (A * SFA) @ (B * SFB)^T, A / B e4m3
# [128, 256] K-major, ue8m0 scales per 32 K read from CUTLASS's 128x4 scale atom (the bytes and
# names of fixtures/r3_cute_mxf8). One program, K blocks of BK; PTXV_MX_REV=1 walks them
# backwards (another accumulation order). Candidate of fixtures/c_equiv_mxf8_triton.
#   tools/capture.py --file tools/capture_examples/triton_mxf8.py --out DIR
import os

import numpy as np
import torch
import triton
import triton.language as tl

FIX = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../fixtures/r3_cute_mxf8"
)
M = N = 128
K = 256


@triton.jit
def mxf8_kernel(
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
        ok = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + om[:, None] * K + ok[None, :])
        b = tl.load(b_ptr + on[None, :] * K + ok[:, None])
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
        acc = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc)


def inputs():
    def rd(f):
        return torch.from_numpy(np.fromfile(os.path.join(FIX, f), np.uint8)).cuda()

    return {
        "A": rd("A.bin").view(M, K),
        "B": rd("B.bin").view(N, K),
        "SFA": rd("SFA.bin"),
        "SFB": rd("SFB.bin"),
    }


def run(A, B, SFA, SFB):
    C = torch.empty(M, N, dtype=torch.float32, device=A.device)
    e4 = torch.float8_e4m3fn
    mxf8_kernel[(1,)](
        A.view(e4),
        B.view(e4),
        SFA,
        SFB,
        C,
        M=M,
        N=N,
        K=K,
        BK=int(os.environ.get("PTXV_MX_BK", "128")),
        REV=int(os.environ.get("PTXV_MX_REV", "0")),
        num_warps=4,
        num_stages=1,
    )
    return C

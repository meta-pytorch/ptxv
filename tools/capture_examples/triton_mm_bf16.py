# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# bf16 matmul, f32 accumulation, bf16 output: c = a @ b for a[64, 64], b[64, 64] (row-major),
# one program, two K blocks of 32 (tcgen05), num_stages=1 (with the default pipelining Triton
# 3.5 omits the proxy fence between its cp.async writes and the MMA: a race). The candidates of
# fixtures/c_equiv_mm_bf16 against torch.mm (cuBLAS nvjet, fixtures/c_mm_bf16_nnt;
# docs/equiv-mma.md). REV=1 walks the K blocks backwards (another accumulation order).
#   tools/capture.py --triton tools/capture_examples/triton_mm_bf16.py:mm_bf16_kernel
#   PTXV_MM_REV=1 tools/capture.py --triton ...   (fixtures/c_triton_mm_bf16_rev)
import os

import torch
import triton
import triton.language as tl


@triton.jit
def mm_bf16_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    REV: tl.constexpr,
):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    acc = tl.zeros([M, N], dtype=tl.float32)
    for kb in range(0, K, BK):
        k0 = K - BK - kb if REV else kb
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16))


def inputs():
    g = torch.Generator().manual_seed(0)
    return {
        "a": torch.randn(64, 64, generator=g).bfloat16().cuda(),
        "b": torch.randn(64, 64, generator=g).bfloat16().cuda(),
    }


def run(a, b):
    c = torch.empty(64, 64, dtype=torch.bfloat16, device=a.device)
    mm_bf16_kernel[(1,)](
        a,
        b,
        c,
        M=64,
        N=64,
        K=64,
        BK=32,
        REV=int(os.environ.get("PTXV_MM_REV", "0")),
        num_warps=4,
        num_stages=1,
    )
    return c

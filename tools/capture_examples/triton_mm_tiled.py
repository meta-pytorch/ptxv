# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Tiled bf16 matmul, f32 accumulation, bf16 output: c = a @ b, a[M, K], b[K, N] row-major.
# Adapted from ptxv tools/capture_examples/triton_mm_bf16.py (one program, whole tile) to a
# grid of BM x BN tiles so it scales to realistic sizes. num_stages=1 as in the original
# (Triton 3.5 pipelining omits a proxy fence: a race). REV=1 walks the K blocks backwards.
# Env: PTXV_MM_M, PTXV_MM_N, PTXV_MM_K, PTXV_MM_REV, PTXV_MM_BM/BN/BK.
import os

import torch
import triton
import triton.language as tl

E = os.environ
M = int(E.get("PTXV_MM_M", "1024"))
N = int(E.get("PTXV_MM_N", "1024"))
K = int(E.get("PTXV_MM_K", "4096"))
BM = int(E.get("PTXV_MM_BM", "128"))
BN = int(E.get("PTXV_MM_BN", "128"))
BK = int(E.get("PTXV_MM_BK", "64"))
REV = int(E.get("PTXV_MM_REV", "1"))


@triton.jit
def mm_tiled_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    REV: tl.constexpr,
):
    pid = tl.program_id(0)
    nn = N // BN
    pm = pid // nn
    pn = pid % nn
    offs_m = pm * BM + tl.arange(0, BM)
    offs_n = pn * BN + tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
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
        "a": torch.randn(M, K, generator=g).bfloat16().cuda(),
        "b": torch.randn(K, N, generator=g).bfloat16().cuda(),
    }


def run(a, b):
    c = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    mm_tiled_kernel[((M // BM) * (N // BN),)](
        a, b, c, M=M, N=N, K=K, BM=BM, BN=BN, BK=BK, REV=REV, num_warps=4, num_stages=1
    )
    return c

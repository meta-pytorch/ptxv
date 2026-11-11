# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Attention, f32, scores computed: o = softmax(q k^T) @ v (scale 1) in the flash form: per query
# row, blocks of 16 keys, their scores q . k_j, a running max and sum, the accumulator rescaled
# by exp(m_old - m_new). The candidate of fixtures/c_equiv_attn against
# triton_attn_f32_ref.py (same inputs).
#   tools/capture.py --triton tools/capture_examples/triton_attn_f32_flash.py:attn_f32_flash_kernel
import torch
import triton
import triton.language as tl


@triton.jit
def attn_f32_flash_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, N: tl.constexpr, D: tl.constexpr, BN: tl.constexpr
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + row * D + offs_d)
    m = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for start in range(0, N, BN):
        offs_n = start + tl.arange(0, BN)
        k = tl.load(k_ptr + offs_n[:, None] * D + offs_d[None, :])
        s = tl.sum(q[None, :] * k, axis=1)
        m_new = tl.maximum(m, tl.max(s, axis=0))
        alpha = tl.exp(m - m_new)
        e = tl.exp(s - m_new)
        l_i = l_i * alpha + tl.sum(e, axis=0)
        v = tl.load(v_ptr + offs_n[:, None] * D + offs_d[None, :])
        acc = acc * alpha + tl.sum(e[:, None] * v, axis=0)
        m = m_new
    tl.store(o_ptr + row * D + offs_d, acc / l_i)


def inputs():
    g = torch.Generator().manual_seed(0)
    return {
        "q": torch.randn(16, 16, generator=g).cuda(),
        "k": torch.randn(64, 16, generator=g).cuda(),
        "v": torch.randn(64, 16, generator=g).cuda(),
    }


def run(q, k, v):
    o = torch.empty_like(q)
    attn_f32_flash_kernel[(q.shape[0],)](q, k, v, o, N=64, D=16, BN=16, num_warps=1)
    return o

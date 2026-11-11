# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Attention core, f32, scores given: o = softmax(s) @ v in the flash-attention form, one
# program per query row: blocks of 16 keys, a running max m and sum l, and the accumulator
# rescaled by exp(m_old - m_new) whenever the max grows; o = acc / l at the end. The
# candidate of fixtures/c_equiv_attn (same inputs as triton_attn_core_ref.py).
#   tools/capture.py --triton tools/capture_examples/triton_attn_core_flash.py:attn_flash_kernel
import torch
import triton
import triton.language as tl


@triton.jit
def attn_flash_kernel(
    s_ptr, v_ptr, o_ptr, N: tl.constexpr, D: tl.constexpr, BN: tl.constexpr
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, D)
    m = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for start in range(0, N, BN):
        offs_n = start + tl.arange(0, BN)
        s = tl.load(s_ptr + row * N + offs_n)
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
        "s": torch.randn(16, 64, generator=g).cuda(),
        "v": torch.randn(64, 16, generator=g).cuda(),
    }


def run(s, v):
    o = torch.empty(s.shape[0], v.shape[1], device=s.device)
    attn_flash_kernel[(s.shape[0],)](s, v, o, N=64, D=16, BN=16, num_warps=1)
    return o

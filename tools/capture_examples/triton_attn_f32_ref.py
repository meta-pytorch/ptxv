# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Attention, f32, scores computed: o = softmax(q k^T) @ v (scale 1), the reference form: per
# query row the 64 scores q . k_j (FFMA sums), softmax normalized first, then the P.V sums.
# q[16, 16], k[64, 16], v[64, 16]. The reference of fixtures/c_equiv_attn (second candidate
# pair) against triton_attn_f32_flash.py (docs/equiv-reassoc.md, "Attention").
#   tools/capture.py --triton tools/capture_examples/triton_attn_f32_ref.py:attn_f32_ref_kernel
import torch
import triton
import triton.language as tl


@triton.jit
def attn_f32_ref_kernel(q_ptr, k_ptr, v_ptr, o_ptr, N: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    offs_n = tl.arange(0, N)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + row * D + offs_d)
    k = tl.load(k_ptr + offs_n[:, None] * D + offs_d[None, :])
    s = tl.sum(q[None, :] * k, axis=1)
    e = tl.exp(s - tl.max(s, axis=0))
    p = e / tl.sum(e, axis=0)
    v = tl.load(v_ptr + offs_n[:, None] * D + offs_d[None, :])
    tl.store(o_ptr + row * D + offs_d, tl.sum(p[:, None] * v, axis=0))


def inputs():
    g = torch.Generator().manual_seed(0)
    return {
        "q": torch.randn(16, 16, generator=g).cuda(),
        "k": torch.randn(64, 16, generator=g).cuda(),
        "v": torch.randn(64, 16, generator=g).cuda(),
    }


def run(q, k, v):
    o = torch.empty_like(q)
    attn_f32_ref_kernel[(q.shape[0],)](q, k, v, o, N=64, D=16, num_warps=1)
    return o

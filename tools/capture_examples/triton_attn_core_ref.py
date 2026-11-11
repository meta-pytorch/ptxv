# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Attention core, f32, scores given: o = softmax(s) @ v, the reference form (normalize P first,
# then the P.V sums), one program per query row: s[16, 64] (scores), v[64, 16]. The reference
# of fixtures/c_equiv_attn against triton_attn_core_flash.py (docs/equiv-reassoc.md,
# "Attention").
#   tools/capture.py --triton tools/capture_examples/triton_attn_core_ref.py:attn_ref_kernel
import torch
import triton
import triton.language as tl


@triton.jit
def attn_ref_kernel(s_ptr, v_ptr, o_ptr, N: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    offs_n = tl.arange(0, N)
    offs_d = tl.arange(0, D)
    s = tl.load(s_ptr + row * N + offs_n)
    e = tl.exp(s - tl.max(s, axis=0))
    p = e / tl.sum(e, axis=0)
    v = tl.load(v_ptr + offs_n[:, None] * D + offs_d[None, :])
    o = tl.sum(p[:, None] * v, axis=0)
    tl.store(o_ptr + row * D + offs_d, o)


def inputs():
    g = torch.Generator().manual_seed(0)
    return {
        "s": torch.randn(16, 64, generator=g).cuda(),
        "v": torch.randn(64, 16, generator=g).cuda(),
    }


def run(s, v):
    o = torch.empty(s.shape[0], v.shape[1], device=s.device)
    attn_ref_kernel[(s.shape[0],)](s, v, o, N=64, D=16, num_warps=1)
    return o

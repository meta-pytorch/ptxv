# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Row softmax with a boolean mask (masked elements -inf). A fully masked row has max -inf, so
# x - max = -inf - -inf = NaN: NaN outputs from finite inputs (ptxv check --nonfinite).
import torch
import triton
import triton.language as tl


@triton.jit
def nf_masked_softmax_kernel(x_ptr, m_ptr, y_ptr, N: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, N)
    x = tl.load(x_ptr + row * N + i)
    m = tl.load(m_ptr + row * N + i)
    x = tl.where(m != 0, float("-inf"), x)
    z = x - tl.max(x, 0)
    e = tl.exp(z)
    tl.store(y_ptr + row * N + i, e / tl.sum(e, 0))


def inputs():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 128, generator=g)
    m = torch.zeros(4, 128, dtype=torch.uint8)
    m[1] = 1
    m[2, :64] = 1
    return {"x": x.cuda(), "m": m.cuda()}


def run(x, m):
    y = torch.empty_like(x)
    nf_masked_softmax_kernel[(x.shape[0],)](x, m, y, N=x.shape[1])
    return y

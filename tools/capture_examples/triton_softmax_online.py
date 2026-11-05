# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Triton online softmax (one program per row): a first loop over BLOCK-wide chunks keeps the
# running max m and the running sum l, rescaled by exp(m_old - m_new) whenever the max grows
# (the online-normalizer form used by flash attention); a second loop writes exp(x - m) / l.
# The online candidate for an equivalence check against torch.softmax(x, -1)
# (tools/capture.py --triton triton_softmax_online.py:softmax_online_kernel). Same inputs as
# `--op 'torch.softmax(x, -1)' --inputs 'x=randn(4,128,f32)'` (seed 0).
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_online_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    m = -float("inf")
    l_i = 0.0
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(
            x_ptr + row * n_cols + cols, mask=cols < n_cols, other=-float("inf")
        )
        m_new = tl.maximum(m, tl.max(x, axis=0))
        l_i = l_i * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(
            x_ptr + row * n_cols + cols, mask=cols < n_cols, other=-float("inf")
        )
        tl.store(y_ptr + row * n_cols + cols, tl.exp(x - m) / l_i, mask=cols < n_cols)


def inputs():
    g = torch.Generator().manual_seed(0)
    return {"x": torch.randn(4, 128, generator=g).cuda()}


def run(x):
    y = torch.empty_like(x)
    softmax_online_kernel[(x.shape[0],)](x, y, x.shape[1], BLOCK=32, num_warps=1)
    return y

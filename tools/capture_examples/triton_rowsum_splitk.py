# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Triton row sum as split-K: one launch writes SPLIT partial sums per row into a workspace,
# a second launch combines them (tools/capture.py --file triton_rowsum_splitk.py). The
# split-K candidate for an equivalence check against torch.sum(x, -1); same inputs as
# `--op 'torch.sum(x, -1)' --inputs 'x=randn(4,128,f32)'` (seed 0).
import torch
import triton
import triton.language as tl


@triton.jit
def partial_kernel(x_ptr, ws_ptr, n_cols, SPLIT: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.program_id(1)
    cols = k * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(x_ptr + row * n_cols + cols, mask=cols < n_cols, other=0.0)
    tl.store(ws_ptr + row * SPLIT + k, tl.sum(x, axis=0))


@triton.jit
def combine_kernel(ws_ptr, out_ptr, SPLIT: tl.constexpr):
    row = tl.program_id(0)
    p = tl.load(ws_ptr + row * SPLIT + tl.arange(0, SPLIT))
    tl.store(out_ptr + row, tl.sum(p, axis=0))


def inputs():
    g = torch.Generator().manual_seed(0)
    return {"x": torch.randn(4, 128, generator=g).cuda()}


def run(x):
    split = 4
    ws = torch.empty(x.shape[0], split, device=x.device)
    out = torch.empty(x.shape[0], device=x.device)
    partial_kernel[(x.shape[0], split)](
        x, ws, x.shape[1], SPLIT=split, CHUNK=32, num_warps=1
    )
    combine_kernel[(x.shape[0],)](ws, out, SPLIT=split, num_warps=1)
    return out

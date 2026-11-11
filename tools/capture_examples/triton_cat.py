# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Triton concatenation along dim 0 (one program per output row), the candidate for an
# equivalence check against torch.cat (tools/capture.py --triton triton_cat.py:cat_kernel).
# Same inputs as `--op 'torch.cat([x, y], 0)' --inputs 'x=randn(8,32,f32),y=randn(8,32,f32)'`.
import torch
import triton
import triton.language as tl


@triton.jit
def cat_kernel(x_ptr, y_ptr, out_ptr, rows_x, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < n_cols
    if row < rows_x:
        v = tl.load(x_ptr + row * n_cols + cols, mask=m)
    else:
        v = tl.load(y_ptr + (row - rows_x) * n_cols + cols, mask=m)
    tl.store(out_ptr + row * n_cols + cols, v, mask=m)


def inputs():
    g = torch.Generator().manual_seed(0)
    return {
        "x": torch.randn(8, 32, generator=g).cuda(),
        "y": torch.randn(8, 32, generator=g).cuda(),
    }


def run(x, y):
    out = torch.empty(x.shape[0] + y.shape[0], x.shape[1], device=x.device)
    cat_kernel[(out.shape[0],)](x, y, out, x.shape[0], x.shape[1], BLOCK=32)
    return out

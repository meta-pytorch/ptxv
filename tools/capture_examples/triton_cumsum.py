# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Triton row cumsum (tl.cumsum, one program per row), the candidate for an equivalence check
# against torch.cumsum (tools/capture.py --triton triton_cumsum.py:cumsum_kernel). Same inputs
# as `--op 'torch.cumsum(x, -1)' --inputs 'x=randn(4,128,f32)'` (seed 0).
import torch
import triton
import triton.language as tl


@triton.jit
def cumsum_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=m, other=0.0)
    tl.store(y_ptr + row * n_cols + cols, tl.cumsum(x, axis=0), mask=m)


def inputs():
    g = torch.Generator().manual_seed(0)
    return {"x": torch.randn(4, 128, generator=g).cuda()}


def run(x):
    y = torch.empty_like(x)
    cumsum_kernel[(x.shape[0],)](
        x, y, x.shape[1], BLOCK=triton.next_power_of_2(x.shape[1])
    )
    return y

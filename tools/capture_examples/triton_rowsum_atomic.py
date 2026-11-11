# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Triton row sum split along the row into SPLIT programs that combine with fp32 atomics
# (tl.atomic_add: REDG.E.ADD.F32.FTZ.RN), the atomic candidate for an equivalence check against
# torch.sum(x, -1) (tools/capture.py --triton triton_rowsum_atomic.py:rowsum_atomic_kernel).
# Same inputs as `--op 'torch.sum(x, -1)' --inputs 'x=randn(4,128,f32)'` (seed 0).
import torch
import triton
import triton.language as tl


@triton.jit
def rowsum_atomic_kernel(x_ptr, out_ptr, n_cols, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.program_id(1)
    cols = k * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(x_ptr + row * n_cols + cols, mask=cols < n_cols, other=0.0)
    tl.atomic_add(out_ptr + row, tl.sum(x, axis=0), sem="relaxed")


def inputs():
    g = torch.Generator().manual_seed(0)
    return {"x": torch.randn(4, 128, generator=g).cuda()}


def run(x):
    out = torch.zeros(x.shape[0], device=x.device)
    rowsum_atomic_kernel[(x.shape[0], 4)](x, out, x.shape[1], CHUNK=32, num_warps=1)
    return out

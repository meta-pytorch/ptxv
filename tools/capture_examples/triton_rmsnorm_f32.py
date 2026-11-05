# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Triton RMS norm, f32 in and out (rung 4's rmsnorm_kernel: mean of squares, tl.rsqrt), one
# program per 128-wide row, num_warps=2: the candidate of fixtures/c_equiv_rmsnorm against
# F.rms_norm(x, (128,), w, 1e-6) on the same f32 inputs (names x, w, seed 0).
#   tools/capture.py --triton tools/capture_examples/triton_rmsnorm_f32.py:rmsnorm_kernel
import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0)
    ms = tl.sum(x * x, axis=0) / n_cols
    r = tl.rsqrt(ms + eps)
    w = tl.load(w_ptr + offs, mask=mask)
    tl.store(y_ptr + row * stride + offs, x * r * w, mask=mask)


def inputs():
    g = torch.Generator().manual_seed(0)
    return {
        "x": torch.randn(4, 128, generator=g).cuda(),
        "w": torch.randn(128, generator=g).cuda(),
    }


def run(x, w):
    y = torch.empty_like(x)
    rmsnorm_kernel[(x.shape[0],)](x, w, y, 128, 128, 1e-6, BLOCK=128, num_warps=2)
    return y

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Triton layer norm, f32 in and out (rung 4's layernorm_kernel, two-pass mean / variance,
# 1 / sqrt), one program per 128-wide row, num_warps=2: the candidate of fixtures/c_equiv_layernorm
# against F.layer_norm(x, (128,), w, b, 1e-5) on the same f32 inputs (names x, w, b, seed 0).
#   tools/capture.py --triton tools/capture_examples/triton_layernorm_f32.py:layernorm_kernel
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, stride, n_cols, eps, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    y = d * rstd * w + b
    tl.store(y_ptr + row * stride + offs, y, mask=mask)


def inputs():
    g = torch.Generator().manual_seed(0)
    return {
        "x": torch.randn(4, 128, generator=g).cuda(),
        "w": torch.randn(128, generator=g).cuda(),
        "b": torch.randn(128, generator=g).cuda(),
    }


def run(x, w, b):
    y = torch.empty_like(x)
    layernorm_kernel[(x.shape[0],)](x, w, b, y, 128, 128, 1e-5, BLOCK=128, num_warps=2)
    return y

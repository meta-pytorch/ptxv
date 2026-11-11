# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Triton layer norm forward (the 05-layer-norm tutorial kernel, one program per row), for
# tools/capture.py --triton triton_layernorm.py:layernorm_kernel
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr, stride, n_cols, eps, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_cols
    rstd = 1 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask).to(tl.float32)
    y = xc * rstd * w + b
    tl.store(y_ptr + row * stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def inputs():
    g = torch.Generator().manual_seed(0)

    def r(*s):
        return torch.randn(*s, generator=g).to(torch.bfloat16).cuda()

    return {"x": r(4, 128), "w": r(128), "b": r(128)}


def run(x, w, b):
    y = torch.empty_like(x)
    layernorm_kernel[(x.shape[0],)](
        x,
        y,
        w,
        b,
        x.stride(0),
        x.shape[1],
        1e-5,
        BLOCK=triton.next_power_of_2(x.shape[1]),
    )
    return y

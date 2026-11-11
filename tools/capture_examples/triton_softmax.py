# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Triton row softmax (the 02-fused-softmax tutorial kernel, one program per row), for
# tools/capture.py --triton triton_softmax.py:softmax_kernel
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=-float("inf")).to(
        tl.float32
    )
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(
        out_ptr + row * out_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask
    )


def inputs():
    g = torch.Generator().manual_seed(0)
    return {"x": torch.randn(4, 128, generator=g).to(torch.bfloat16).cuda()}


def run(x):
    y = torch.empty_like(x)
    softmax_kernel[(x.shape[0],)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        x.shape[1],
        BLOCK=triton.next_power_of_2(x.shape[1]),
    )
    return y

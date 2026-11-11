# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Triton tanh-GELU on the hardware tanh (`tanh.approx.f32` -> MUFU.TANH, as fast-math kernels
# use it): 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). The same inputs as fixtures/c_gelu_tanh
# (torch's GELU, whose tanhf is an EX2/RCP sequence).
#   tools/capture.py --triton tools/capture_examples/triton_gelu_tanh.py:gelu_kernel --out fixtures/c_triton_gelu_tanh
import torch
import triton
import triton.language as tl


@triton.jit
def tanh_approx(x):
    return tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(x_ptr + i, mask=m)
    t = tanh_approx(0.7978845608028654 * (x + 0.044715 * x * x * x))
    tl.store(out_ptr + i, 0.5 * x * (1.0 + t), mask=m)


def inputs():
    g = torch.Generator().manual_seed(0)
    return {"x": (torch.randn(4, 256, generator=g) * 4).cuda()}


def run(x):
    out = torch.empty_like(x)
    gelu_kernel[(triton.cdiv(x.numel(), 256),)](x, out, x.numel(), BLOCK=256)
    return out

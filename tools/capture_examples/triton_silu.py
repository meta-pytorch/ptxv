# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Triton SiLU: x * sigmoid(x) = x / (1 + exp(-x)), computed in f32 and stored in the input's
# dtype. Elementwise-equivalence demo against torch's F.silu (docs/equiv-elementwise.md).
#   tools/compare.py --ref 'F.silu(x)' --cand tools/capture_examples/triton_silu.py:silu_kernel \
#       --inputs 'x=randn(4,256,f32)*4'
import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(x_ptr + i, mask=m).to(tl.float32)
    y = x / (1.0 + tl.exp(-x))
    tl.store(out_ptr + i, y.to(out_ptr.dtype.element_ty), mask=m)


def inputs():
    g = torch.Generator().manual_seed(0)
    return {"x": (torch.randn(4, 256, generator=g) * 4).cuda()}


def run(x):
    out = torch.empty_like(x)
    silu_kernel[(triton.cdiv(x.numel(), 256),)](x, out, x.numel(), BLOCK=256)
    return out

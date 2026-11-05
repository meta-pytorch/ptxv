# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Triton erf-GELU (libdevice erf): 0.5 x (1 + erf(x / sqrt 2)), computed in f32 and stored in
# the input's dtype (f32 or bf16). Elementwise-equivalence demo against torch's F.gelu
# (docs/equiv-elementwise.md).
#   tools/compare.py --ref 'F.gelu(x)' --cand tools/capture_examples/triton_gelu_erf.py:gelu_kernel \
#       --inputs 'x=randn(4,256,f32)*4'
import torch
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(x_ptr + i, mask=m).to(tl.float32)
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(out_ptr + i, y.to(out_ptr.dtype.element_ty), mask=m)


def inputs():
    g = torch.Generator().manual_seed(0)
    return {"x": (torch.randn(4, 256, generator=g) * 4).cuda()}


def run(x):
    out = torch.empty_like(x)
    gelu_kernel[(triton.cdiv(x.numel(), 256),)](x, out, x.numel(), BLOCK=256)
    return out

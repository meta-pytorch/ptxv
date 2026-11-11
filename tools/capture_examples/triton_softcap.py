# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Triton logit softcap 30 tanh(x / 30) (as x * (1/30)) on the hardware tanh (`tanh.approx.f32` -> MUFU.TANH, as
# attention kernels with softcapping use it). The same inputs as fixtures/c_softcap (torch).
#   tools/capture.py --triton tools/capture_examples/triton_softcap.py:softcap_kernel --out fixtures/c_triton_softcap
import torch
import triton
import triton.language as tl


@triton.jit
def tanh_approx(x):
    return tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def softcap_kernel(x_ptr, out_ptr, n, CAP: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(x_ptr + i, mask=m)
    # x * (1 / CAP), as attention kernels write it (`x / CAP` would be PTX div.full.f32).
    tl.store(out_ptr + i, CAP * tanh_approx(x * (1.0 / CAP)), mask=m)


def inputs():
    g = torch.Generator().manual_seed(0)
    return {"x": (torch.randn(4, 256, generator=g) * 50).cuda()}


def run(x):
    out = torch.empty_like(x)
    softcap_kernel[(triton.cdiv(x.numel(), 256),)](
        x, out, x.numel(), CAP=30.0, BLOCK=256
    )
    return out

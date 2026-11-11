# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# out = f32(bf16(x + y)) + z, elementwise in f32: the bf16 rounding of the sum is the dominant
# error. `ptxv num` must either track the bf16 round trip or make no claim (N-NUM-TRACK); a
# claim that drops it is caught against the fp64 value of x + y + z.
import torch
import triton
import triton.language as tl


@triton.jit
def num_bf16rt_kernel(x_ptr, y_ptr, z_ptr, o_ptr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.load(x_ptr + i) + tl.load(y_ptr + i)
    w = s.to(tl.bfloat16).to(tl.float32)
    tl.store(o_ptr + i, w + tl.load(z_ptr + i))


def inputs():
    g = torch.Generator().manual_seed(0)
    return {
        "x": torch.randn(1024, generator=g).cuda(),
        "y": torch.randn(1024, generator=g).cuda(),
        "z": torch.randn(1024, generator=g).cuda(),
    }


def run(x, y, z):
    o = torch.empty_like(x)
    num_bf16rt_kernel[(x.numel() // 256,)](x, y, z, o, BLOCK=256)
    return o

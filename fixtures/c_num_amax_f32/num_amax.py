# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Row maxima of |x + y| (an amax, as quantization scales use), one thread per row: FMNMX of an
# |x| operand has no node in `ptxv num`'s DAG, so num must not claim an error here
# (N-NUM-TRACK); the winning sum's rounding error reaches the output.
import torch
import triton
import triton.language as tl


@triton.jit
def num_amax_kernel(x_ptr, y_ptr, o_ptr, K: tl.constexpr, R: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    acc = tl.zeros([R], dtype=tl.float32)
    for k in range(K):
        v = tl.load(x_ptr + rows * K + k) + tl.load(y_ptr + rows * K + k)
        acc = tl.maximum(acc, tl.abs(v))
    tl.store(o_ptr + rows, acc)


def inputs():
    g = torch.Generator().manual_seed(0)
    return {
        "x": torch.randn(256, 64, generator=g).cuda(),
        "y": (torch.randn(256, 64, generator=g) * 1e-3).cuda(),
    }


def run(x, y):
    o = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    num_amax_kernel[(x.shape[0] // 64,)](x, y, o, K=x.shape[1], R=64)
    return o

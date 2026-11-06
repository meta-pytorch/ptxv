# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Log-sum-exp per row, naive and max-shifted (agent AN's errdist demo, fixtures/c_lse_naive,
# fixtures/c_lse_stable). Triton's log is a software polynomial with an integer exponent
# extraction (I2FP.F32.S32), not MUFU.LG2; errdist identifies it as a log slice.
#   tools/capture.py --triton tools/capture_examples/triton_lse.py:lse_naive
import torch
import triton
import triton.language as tl

N = 64
ROWS = 4


@triton.jit
def lse_naive(x_ptr, y_ptr, N: tl.constexpr):
    r = tl.program_id(0)
    o = tl.arange(0, N)
    x = tl.load(x_ptr + r * N + o)
    tl.store(y_ptr + r, tl.log(tl.sum(tl.exp(x), 0)))


@triton.jit
def lse_stable(x_ptr, y_ptr, N: tl.constexpr):
    r = tl.program_id(0)
    o = tl.arange(0, N)
    x = tl.load(x_ptr + r * N + o)
    m = tl.max(x, 0)
    tl.store(y_ptr + r, m + tl.log(tl.sum(tl.exp(x - m), 0)))


def inputs():
    g = torch.Generator().manual_seed(0)
    return {"x": torch.randn(ROWS, N, generator=g).cuda()}


def run(x):
    y = torch.empty(ROWS, device=x.device, dtype=torch.float32)
    lse_naive[(ROWS,)](x, y, N=N)
    return y

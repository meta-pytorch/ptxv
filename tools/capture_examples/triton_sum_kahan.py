# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Kahan-compensated row sum, one program per row (agent AN's errdist demo,
# fixtures/c_sum_kahan): the compensation c = (t - s) - v is exact algebra, so a numerics
# DAG that folds it loses the dependence on the inputs; errdist's probe catches that and
# re-extracts in segment mode (docs/errdist.md §1).
#   tools/capture.py --triton tools/capture_examples/triton_sum_kahan.py:sum_kahan
import torch
import triton
import triton.language as tl

N = 512
ROWS = 4


@triton.jit
def sum_kahan(x_ptr, y_ptr, N: tl.constexpr):
    r = tl.program_id(0)
    s = 0.0
    c = 0.0
    for i in range(N):
        v = tl.load(x_ptr + r * N + i) - c
        t = s + v
        c = (t - s) - v
        s = t
    tl.store(y_ptr + r, s)


def inputs():
    g = torch.Generator().manual_seed(0)
    return {"x": torch.randn(ROWS, N, generator=g).cuda()}


def run(x):
    y = torch.empty(ROWS, device=x.device, dtype=torch.float32)
    sum_kahan[(ROWS,)](x, y, N=N)
    return y

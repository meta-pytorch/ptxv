# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal example: two torch kernels computing the same row sums in different orders.

    python python/examples/rowsum.py

Captures `x.sum(-1)` (a reduction kernel) and `x.cumsum(-1)[:, -1]` (a scan kernel plus a
copy), proves they compute the same function within a bound, evaluates the distribution of
that bound under several input distributions, and checks fresh GPU results against it.
"""

import ptxv  # import before the first CUDA work, so ptxv sees the kernels' modules load
import torch
from ptxv import dist as D

torch.manual_seed(0)
x = torch.randn(8, 512, device="cuda")

with ptxv.capture() as ref:
    x.sum(-1)
with ptxv.capture() as cand:
    x.cumsum(-1)[:, -1].contiguous()

# The candidate is two launches (scan, then the strided copy): compare the whole region.
eq = ptxv.compare(ref.kernels[0], cand.all())
print(eq.headline)

# Distribution of the proven worst-case bound tau(x) for x drawn from each distribution.
for name, d in [
    ("normal", D.Normal(0, 1)),
    ("student-t(3)", D.StudentT(3)),
    ("bf16-quantized normal", D.Normal(0, 1).quantize("bf16")),
]:
    print(f"{name:>22}: {eq.bound.distribution({'x': d}, n=200)}")

# Fresh inputs on the GPU: the actual difference never exceeds tau(x).
xs = torch.randn(8, 512, device="cuda")
print("check:", eq.bound.check(xs.sum(-1), xs.cumsum(-1)[:, -1], {"x": xs}))

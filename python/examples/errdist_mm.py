# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal errdist example: how far a bf16 cuBLAS matmul is from the correctly rounded result.

    python python/examples/errdist_mm.py

Captures `a @ b` and asks `ptxv.errdist` for the distribution of (kernel output - exact product
rounded to bf16) under a few input distributions. Each result says how often the output bits
differ from the correctly rounded value and by how many ulps; `gpu_trials` re-runs the kernel
on fresh inputs on the GPU and checks the prediction against what it observes.
"""

import ptxv  # import before the first CUDA work, so ptxv sees the kernels' modules load
import torch
from ptxv import dist as D

a = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
b = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)

with ptxv.capture() as cap:
    a @ b

for name, d in [
    ("normal", D.Normal(0, 1)),
    ("uniform[-1,1]", D.Uniform(-1, 1)),
    ("small integers", D.Integers(-4, 5)),
]:
    e = ptxv.errdist(cap.kernels[0], d, gpu_trials=50, metric="ulp")
    print(f"--- {name}\n{e}")
    print(
        "    P(output bits differ from correctly rounded): [p, lo, hi] =",
        e.meta["p_differ"],
    )
    for v in e.meta["validation"] or []:
        print(
            f"    {v['source']} vs {v['against']}: predicted {v['p_differ_pred']:.3g}, "
            f"observed {v['p_differ_obs']}, agree={v['agree']}"
        )

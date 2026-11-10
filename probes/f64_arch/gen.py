# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs and exact expected output of probes/f64_arch: 1024 threads x (a1, a2, a3, a0).
expected/out.bin is IEEE round-to-nearest of the kernel's PTX, computed exactly with rationals."""

import os
from fractions import Fraction as F

import numpy as np

rng = np.random.default_rng(11)
n = 1024
a1 = rng.standard_normal(n)
a2 = rng.uniform(0.5, 1.5, n)
a3 = rng.uniform(-1, 1, n)
a0 = rng.standard_normal(n)
np.stack([a1, a2, a3, a0], 1).astype(np.float64).tofile("in.bin")
r = lambda q: float(q)  # correctly rounded
out = np.zeros((n, 4))
for i in range(n):
    A1, A2, A3, A0 = map(F, (a1[i], a2[i], a3[i], a0[i]))
    x = F(r(A1 * A2 + A3))
    for _ in range(15):
        x = F(r(x * A2 + A3))
    x20 = F(r(A1 + A3))
    x21 = F(r(x20 * A2))
    x22 = F(r(x21 + A1))
    x23 = F(r(x22 * x22))
    x24 = F(r(x23 + x20))
    x25 = F(r(x24 * A3))
    x30 = F(r(x25 + A0))
    x31 = x if x30 < A1 else x25
    out[i] = [float(x), float(x25), float(x30), float(x31)]
os.makedirs("expected", exist_ok=True)
out.tofile("expected/out.bin")

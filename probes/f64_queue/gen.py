# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/f64_queue: 1024 (a, b) f64 pairs (in.bin), a in [-4, 4] away from -1, b in
[-8, 8]: the division a-1 / a+1 stays on ptxas's fast path."""

import numpy as np

rng = np.random.default_rng(11)
a = rng.uniform(-4, 4, 1024)
a[np.abs(a + 1) < 0.25] += 1.0
b = rng.uniform(-8, 8, 1024)
np.stack([a, b], 1).astype(np.float64).tofile("in.bin")

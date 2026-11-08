# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/xu_war: 64 x 16 f32 values (in.bin): normal, with subnormal, infinite and
NaN lanes. expected/out.bin is GB300's output (`ptxv hw-worker probes/xu_war xuwar.cubin --runs 1
--dump`)."""

import numpy as np

rng = np.random.default_rng(7)
x = rng.standard_normal(64 * 16).astype(np.float32)
x[::97] = np.float32(1e-40)
x[5::131] = np.inf
x[11::307] = np.nan
x.tofile("in.bin")

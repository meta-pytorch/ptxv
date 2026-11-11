# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/ldst256: 1024 rows of 16 random 32-bit words (in.bin)."""

import numpy as np

np.random.default_rng(7).integers(0, 2**32, size=(1024, 16), dtype=np.uint64).astype(
    np.uint32
).tofile("in.bin")

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for the MATCH.ANY probes: 1024 rows of (u32 key, pad, u64 key) (in.bin). Keys repeat
within a warp in groups of varied sizes (few distinct values per warp), 64-bit keys differ
only in the high word for some lanes."""

import numpy as np

rng = np.random.default_rng(37)
rows = []
for i in range(1024):
    w = i // 32
    k = int(rng.integers(0, 2 + w % 5))
    k64 = (int(rng.integers(0, 3)) << 32) | k
    rows.append((k * 0x9E3779B1 & 0xFFFFFFFF, 0, k64 & 0xFFFFFFFF, k64 >> 32))
np.array(rows, dtype=np.uint64).astype(np.uint32).tofile("in.bin")

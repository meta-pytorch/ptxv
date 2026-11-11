# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/hsetp2: 1024 (a, b) word pairs (in.bin). The first 576 rows pair 24 half
patterns that are special as f16 or as bf16 (signed zeros, subnormals, +-1, +-Inf, NaNs of
both formats, 8 and 16 as f16) in both halves of a and b; the rest are random."""

import numpy as np

H = [
    0x0000,
    0x8000,
    0x0001,
    0x8001,
    0x3C00,
    0xBC00,
    0x3F80,
    0xBF80,
    0x7C00,
    0xFC00,
    0x7F80,
    0xFF80,
    0x7E00,
    0xFE00,
    0x7FC0,
    0xFFC0,
    0x4800,
    0x4C00,
    0x4801,
    0x4BFF,
    0x7C01,
    0x7F81,
    0x03FF,
    0x007F,
]
rng = np.random.default_rng(29)
rows = []
for i in range(1024):
    if i < 576:
        x, y = H[i % 24], H[i // 24]
        rows.append((x | (y << 16), y | (x << 16)))
    else:
        rows.append(tuple(rng.integers(0, 2**32, 2, dtype=np.uint64)))
np.array(rows, dtype=np.uint64).astype(np.uint32).tofile("in.bin")

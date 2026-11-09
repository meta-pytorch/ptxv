# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/int_misc: 1024 (a, b, c_lo, c_hi) words (in.bin): 32-bit extremes and
random words; c a 64-bit value (0, 1, 2^63, all ones, random)."""

import random

import numpy as np

random.seed(3)
X = [0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFFE, 0x55555555]
C = [0, 1, 1 << 63, (1 << 64) - 1, (1 << 63) - 1, 1 << 32, 0xFFFFFFFF]
rows = []
for i in range(1024):
    a = X[i % 8] if i < 512 else random.getrandbits(32)
    b = X[(i // 8) % 8] if i < 512 else random.getrandbits(32)
    c = C[(i // 64) % 7] if i < 512 else random.getrandbits(64)
    rows.append((a, b, c & 0xFFFFFFFF, c >> 32))
np.array(rows, np.uint32).tofile("in.bin")

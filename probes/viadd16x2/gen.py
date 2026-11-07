# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/viadd16x2: 1024 (a, b) words (in.bin). The first 512 rows pair 16-bit lane
extremes (0, 1, 0x7fff, 0x8000, 0xffff, 0xfffe, 0x5555, 0xaaaa) in both lanes, so a carry or
a signed overflow out of the low lane is visible; the rest are random."""

import random

import numpy as np

random.seed(5)
H = [0, 1, 0x7FFF, 0x8000, 0xFFFF, 0xFFFE, 0x5555, 0xAAAA]
rows = []
for i in range(1024):
    if i < 512:
        a = H[i % 8] | (H[(i // 8) % 8] << 16)
        b = H[(i // 64) % 8] | (H[(i * 3) % 8] << 16)
    else:
        a, b = random.getrandbits(32), random.getrandbits(32)
    rows.append((a, b))
np.array(rows, np.uint32).tofile("in.bin")

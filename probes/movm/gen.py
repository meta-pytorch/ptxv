# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# in.bin: 256 words: lane-coded halves (row, column, warp) for the first warp, random bits
# for the rest, seed 16.
import random
import struct

random.seed(16)
w = []
for i in range(256):
    l = i % 32
    g, t = l // 4, l % 4
    if i < 32:
        w.append((g << 4 | 2 * t) | (g << 4 | 2 * t + 1) << 16)
    else:
        w.append(random.getrandbits(32))
open("in.bin", "wb").write(struct.pack("<256I", *w))

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# in.bin: 1024 u64: edge values (0, +-1, 2^24+1, 2^53+1, int64 / uint64 limits, f32 ties) then
# random 64-bit words with a random number of leading zeros / ones, seed 10.
import random
import struct

random.seed(10)
M = (1 << 64) - 1
E = [
    0,
    1,
    M,
    (1 << 24) + 1,
    (1 << 53) + 1,
    (1 << 63) - 1,
    1 << 63,
    M - 1,
    (1 << 25) + 3,
    (1 << 24) + 2,
    0xFFFF,
    0x8000,
    0x7FFF,
    0xFFFFFFFF,
    0x80000000,
    0x7FFFFFFF,
    ((1 << 24) + 1) << 30,
    (M - (1 << 40)) & M,
]
w = list(E)
while len(w) < 1024:
    k = random.randint(0, 63)
    x = random.getrandbits(64) >> k
    w.append((M ^ x) if random.random() < 0.3 else x)
open("in.bin", "wb").write(struct.pack("<1024Q", *w))

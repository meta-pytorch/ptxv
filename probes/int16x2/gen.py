# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/int16x2: 1024 triples (a, b, c) of u32 words (in.bin) and 1024 u64 (y,
in64.bin). Words are built from 16-bit halves of an edge set (0, 1, 0x7fff, 0x8000, 0x8001,
0xffff, values around the 0x4312 immediate) mixed with random halves, so each lane sees equal,
signed-only-different and unsigned-only-different pairs; y covers 0, 1, borrows across the
32-bit boundary, the extremes and random values."""

import random

import numpy as np

random.seed(5)
N = 1024
H = [
    0x0000,
    0x0001,
    0x7FFF,
    0x8000,
    0x8001,
    0xFFFF,
    0x4311,
    0x4312,
    0x4313,
    0xC312,
    0x00FF,
    0xFF00,
]


def half():
    return random.choice(H) if random.random() < 0.6 else random.getrandbits(16)


def word():
    return half() << 16 | half()


rows = []
for i in range(N):
    a, b, c = word(), word(), word()
    if i % 7 == 0:
        b = a
    if i % 11 == 0:
        c = a ^ 0x80008000
    rows.append((a, b, c))
np.array(rows, np.uint32).tofile("in.bin")
Y = [
    0,
    1,
    0xFFFFFFFF,
    0x100000000,
    0x7FFFFFFFFFFFFFFF,
    0x8000000000000000,
    0xFFFFFFFFFFFFFFFF,
    0x00000000FFFFFFFC,
    0xFFFFFFFF00000000,
]
y = [
    random.choice(Y) if random.random() < 0.3 else random.getrandbits(64)
    for _ in range(N)
]
np.array(y, np.uint64).tofile("in64.bin")

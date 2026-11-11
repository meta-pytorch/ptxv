# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/lea_hi_not: ar.bin (2048 x {a, c, r, 0} u32) and x.bin (2048 u64).
Rows 0..503: every triple of 14 edge values of a (c = another of them), 6 of x's low word and
6 of its high word. Rows 504..1271: borrow boundaries of the first shape: for a random a and
shift s, x's low word within 1 of (a << s) mod 2^32 and its high word within 1 of
a >> (32 - s) (the low word's carry flips and the high word passes through 0). Rows
1272..1527: the same boundaries for the second shape (x << s within 1 << s of (c << s) | r).
The rest: random words. Where not stated, a, c and r are random, r with 0 and all ones mixed
in."""

import random

import numpy as np

random.seed(77)
M = 0xFFFFFFFF
A = [
    0,
    1,
    2,
    0x7FFFFFFF,
    0x80000000,
    0x80000001,
    0xFFFFFFFF,
    0xFFFFFFFE,
    0x55555555,
    0xAAAAAAAA,
    0x00010000,
    0x0000FFFF,
    0x12345678,
    0xDEADBEEF,
]
W = [0, 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFE]


def rnd_r():
    return random.choice([0, M, random.getrandbits(32), random.getrandbits(32)])


rows = [
    (a, A[(i + 5) % 14], W[(i + j) % 6], hi << 32 | lo)
    for i, a in enumerate(A)
    for j, lo in enumerate(W)
    for hi in W
]
while len(rows) < 1272:
    a, s = random.getrandbits(32), random.randint(1, 30)
    lo = ((a << s) + random.choice([-1, 0, 1])) & M
    hi = ((a >> (32 - s)) + random.choice([-1, 0, 1])) & M
    rows.append((a, random.getrandbits(32), rnd_r(), hi << 32 | lo))
while len(rows) < 1528:
    c, s, r = random.getrandbits(32), random.randint(1, 30), random.getrandbits(32)
    t = ((c << s) | r) + (random.choice([-1, 0, 1]) << s)
    x = (t >> s) & (1 << 64 - s) - 1 | random.getrandbits(s) << 64 - s
    rows.append((random.getrandbits(32), c, r, x))
while len(rows) < 2048:
    rows.append((random.getrandbits(32), random.getrandbits(32), rnd_r(), random.getrandbits(64)))
ar = np.zeros((2048, 4), np.uint32)
x = np.zeros(2048, np.uint64)
for g, (a, c, r, v) in enumerate(rows):
    ar[g] = (a, c, r, 0)
    x[g] = v
ar.tofile("ar.bin")
x.tofile("x.bin")

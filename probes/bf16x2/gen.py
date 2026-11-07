# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/bf16x2: 4096 (a, b, c) triples of packed bf16x2 words (in.bin) and the
initial atomic words (m.bin: a, a, a, a per thread).

Lane pairs (a, b): every pair of the edge set E (signed zeros, subnormals, normals at the range
ends, values one ulp apart, infinities, NaNs with several payloads), then pairs whose exact sum
or product is a rounding tie (or one bit off one), then random bf16 patterns. c: edge values,
ties for fma (a*b + c exactly halfway), random."""

import random
from fractions import Fraction

import numpy as np

random.seed(11)
N = 4096

E = [
    0x0000,
    0x8000,
    0x0001,
    0x8001,
    0x0002,
    0x007F,
    0x807F,
    0x0080,
    0x8080,
    0x00FF,
    0x0100,
    0x3F80,
    0xBF80,
    0x3F81,
    0xBF81,
    0x3FFF,
    0x4000,
    0xC000,
    0x3B80,
    0xBB80,
    0x3C00,
    0x3400,
    0x7F7F,
    0xFF7F,
    0x7F00,
    0x7F7E,
    0x7F80,
    0xFF80,
    0x7FC0,
    0xFFC0,
    0x7F81,
    0xFF81,
    0x7FFF,
    0x7FA5,
    0x0040,
    0x1F80,
    0x5F80,
    0x4B00,
    0x3FC0,
    0x4040,
    0xC0A0,
    0x3EAB,
]


def val(h):
    """Exact value of a finite bf16 pattern."""
    e = (h >> 7) & 0xFF
    m = h & 0x7F
    s = -1 if h & 0x8000 else 1
    if e == 0:
        return s * Fraction(m, 1 << 133)
    return s * Fraction(128 + m, 128) * Fraction(2) ** (e - 127)


def finite(h):
    return (h >> 7) & 0xFF != 0xFF


def is_tie(x):
    """x lies exactly halfway between two adjacent bf16 values (normal range)."""
    if x == 0:
        return False
    a = abs(x)
    e = 0
    while a >= 2:
        a /= 2
        e += 1
    while a < 1:
        a *= 2
        e -= 1
    if e < -126 or e > 127:
        return False
    t = a * 256  # 1 xxxxxxx . g
    return t.denominator == 1 and t.numerator % 2 == 1


def rnd16():
    while True:
        h = random.getrandbits(16)
        if finite(h) and 0x30 <= (h >> 7) & 0xFF <= 0xD0:
            return h


pairs = [(x, y) for x in E for y in E]
ties_mul, ties_add = [], []
while len(ties_mul) < 800 or len(ties_add) < 800:
    x, y = rnd16(), rnd16()
    if len(ties_mul) < 800 and is_tie(val(x) * val(y)):
        ties_mul.append((x, y))
    y2 = (y & 0x807F) | ((((x >> 7) & 0xFF) - random.randint(1, 9)) << 7)
    if len(ties_add) < 800 and is_tie(val(x) + val(y2)):
        ties_add.append((x, y2))
pairs += ties_mul + ties_add
while len(pairs) < 2 * N:
    pairs.append((random.getrandbits(16), random.getrandbits(16)))
pairs = pairs[: 2 * N]

cs = []
for i, (x, y) in enumerate(pairs):
    k = i % 4
    if k == 0:
        cs.append(E[(i // 4) % len(E)])
    elif k == 1 and finite(x) and finite(y):
        # c such that a*b + c is a tie, if one is close by.
        p = val(x) * val(y)
        best = None
        for _ in range(40):
            c = rnd16()
            if is_tie(p + val(c)):
                best = c
                break
        cs.append(best if best is not None else rnd16())
    else:
        cs.append(random.getrandbits(16))

a = np.zeros(N, np.uint32)
b = np.zeros(N, np.uint32)
c = np.zeros(N, np.uint32)
for t in range(N):
    (a0, b0), (a1, b1) = pairs[2 * t], pairs[2 * t + 1]
    a[t] = a1 << 16 | a0
    b[t] = b1 << 16 | b0
    c[t] = cs[2 * t + 1] << 16 | cs[2 * t]
np.stack([a, b, c], 1).tofile("in.bin")
np.stack([a, a, a, a], 0).tofile("m.bin")

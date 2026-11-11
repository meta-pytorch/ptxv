# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs of probes/tma_red_order: per element (g, a0, a1) bf16 with a0-first, a1-first and the
single rounding of g + a0 + a1 three different bf16 words (random search, seed 7).
Writes g.bin (4096 bf16) and a.bin (2 x 4096 bf16). usage: gen.py OUTDIR"""

import math
import random
import struct
import sys


def f(h):
    return struct.unpack("<f", struct.pack("<I", h << 16))[0]


def rn(x):
    """bf16 round to nearest even of a float (exact sums of bf16 values in range)."""
    if x == 0:
        return 0x8000 if math.copysign(1, x) < 0 else 0
    m, e = math.frexp(abs(x))  # abs(x) = m * 2^e, 0.5 <= m < 1
    q = m * 256
    r = math.floor(q)
    d = q - r
    if d > 0.5 or d == 0.5 and r % 2 == 1:
        r += 1
    v = math.ldexp(r, e - 8)
    w = struct.unpack("<I", struct.pack("<f", v))[0] >> 16
    return w | (0x8000 if x < 0 else 0)


def rand():
    e = random.randint(120, 134)
    return random.getrandbits(1) << 15 | e << 7 | random.getrandbits(7)


if __name__ == "__main__":
    random.seed(7)
    g, a0, a1 = [], [], []
    while len(g) < 4096:
        x, y, z = rand(), rand(), rand()
        p = rn(f(rn(f(x) + f(y))) + f(z))
        q = rn(f(rn(f(x) + f(z))) + f(y))
        s = rn(f(x) + f(y) + f(z))
        if len({p, q, s}) == 3:
            g.append(x), a0.append(y), a1.append(z)
    out = sys.argv[1]
    open(f"{out}/g.bin", "wb").write(struct.pack("<4096H", *g))
    open(f"{out}/a.bin", "wb").write(struct.pack("<8192H", *(a0 + a1)))
    print(len(g))

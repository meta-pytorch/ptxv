# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/half_add: 4096 (a, b) f16x2 word pairs (inh.bin) and 4096 (c, e) bf16x2
word pairs (inb.bin), and the initial atomic words (m.bin: a, a, a, a, c, c, c, c per thread).

Per format, 8192 lane pairs (x, y) (lane 0 of thread t holds pair 2t, lane 1 pair 2t + 1):
every pair of an edge set E (signed zeros, subnormals, the normal range ends, infinities, NaNs
with several payloads and both signs, quiet and signalling); exact-sum ties of both rounding
directions (the even neighbour below and above in magnitude, both signs); sums at and around
the overflow threshold; subnormal results (cancellation and subnormal sums); every NaN code
with 1.0, NaN codes with 1.0 first, and NaN + NaN; signed-zero sums, inf - inf and NaN with inf
or -0, 16 times each; then random patterns."""

import random
from fractions import Fraction

import numpy as np

from half import BF16, F16, add, exact_sum, is_nan, is_tie, round_val, val

random.seed(76)
N = 4096

E16 = [
    0x0000, 0x8000, 0x0001, 0x8001, 0x0002, 0x03FF, 0x83FF, 0x0400, 0x8400, 0x0401, 0x0200,
    0x3C00, 0xBC00, 0x3C01, 0xBC01, 0x3BFF, 0x4000, 0xC000, 0x1400, 0x9400, 0x1000, 0x7800,
    0x7BFF, 0xFBFF, 0x7BFE, 0x7C00, 0xFC00, 0x7C01, 0xFC01, 0x7DFF, 0x7E00, 0xFE00, 0x7E55,
    0xFC03, 0x7FFF, 0xFFFF, 0x5000, 0x4C00, 0x3555, 0x0155, 0x8155, 0x6400, 0xE400, 0x0800,
]
EB = [
    0x0000, 0x8000, 0x0001, 0x8001, 0x0002, 0x007F, 0x807F, 0x0080, 0x8080, 0x0081, 0x0040,
    0x3F80, 0xBF80, 0x3F81, 0xBF81, 0x3F7F, 0x4000, 0xC000, 0x3B80, 0xBB80, 0x3B00, 0x7F00,
    0x7F7F, 0xFF7F, 0x7F7E, 0x7F80, 0xFF80, 0x7F81, 0xFF81, 0x7FBF, 0x7FC0, 0xFFC0, 0x7FA5,
    0xFF83, 0x7FFF, 0xFFFF, 0x7B80, 0x7B00, 0x3EAB, 0x0055, 0x8055, 0x5F80, 0xDF80, 0x0100,
]


def finite(fmt, h):
    return not is_nan(fmt, h) and (h & 0x7FFF) != (((1 << fmt[0]) - 1) << fmt[1])


def nans(fmt):
    eb, fb = fmt
    top = ((1 << eb) - 1) << fb
    return [s | top | f for s in (0, 0x8000) for f in range(1, 1 << fb)]


def rnd_normal(fmt, lo, hi):
    eb, fb = fmt
    while True:
        h = random.getrandbits(16)
        e = h >> fb & ((1 << eb) - 1)
        if lo <= e <= hi:
            return h


def tie_dir(fmt, x, y):
    """+1 if the RN sum of the tie x + y is the neighbour above in magnitude, -1 below."""
    s = exact_sum(fmt, x, y)
    r = add(fmt, x, y)
    return 1 if abs(val(fmt, r)) > abs(s) else -1


def pairs_for(fmt, E):
    eb, fb = fmt
    emax = (1 << eb) - 2
    one = 0x3C00 if fmt == F16 else 0x3F80
    out = [(x, y) for x in E for y in E]
    # Ties, both directions.
    ups, downs = [], []
    while len(ups) < 600 or len(downs) < 600:
        x = rnd_normal(fmt, 3, emax - 3)
        ex = x >> fb & ((1 << eb) - 1)
        d = random.randint(1, fb + 1)
        y = (random.getrandbits(1) << 15) | (max(ex - d, 1) << fb) | random.getrandbits(fb)
        if not is_tie(fmt, exact_sum(fmt, x, y)):
            continue
        (ups if tie_dir(fmt, x, y) > 0 else downs).append((x, y))
    out += ups[:600] + downs[:600]
    # The overflow threshold: max + y for y around half an ulp of max (y = half an ulp times
    # 1 + k / 16, rounded into the format), both signs; and max - 1 ulp + y.
    mx = max(h for h in range(0x7FFF) if finite(fmt, h))
    half_ulp = val(fmt, mx) - val(fmt, mx - 1)
    half_ulp /= 2
    for k in range(-8, 9):
        y, _ = round_val(fmt, half_ulp * (1 + Fraction(k, 16)), "rn")
        for s in (0, 0x8000):
            out += [(s | mx, s | y), (s | y, s | mx), (s | (mx - 1), s | y)]
    # Subnormal results: x + (-x + small), and subnormal + subnormal.
    for _ in range(300):
        x = rnd_normal(fmt, 1, 3)
        k = random.randint(1, (1 << fb) - 1)
        y = (x ^ 0x8000) - k if (x & 0x7FFF) - k > 0 else x ^ 0x8000
        out.append((x, y))
    for _ in range(200):
        x = random.getrandbits(fb) | (random.getrandbits(1) << 15)
        y = random.getrandbits(fb) | (random.getrandbits(1) << 15)
        out.append((x, y))
    # NaNs: every code with 1.0, a sample with 1.0 first, NaN + NaN.
    ns = nans(fmt)
    out += [(n, one) for n in ns]
    out += [(one, n) for n in random.sample(ns, min(400, len(ns)))]
    out += [(random.choice(ns), random.choice(ns)) for _ in range(200)]
    # Signed zeros, inf - inf and NaN with inf / zero, repeated so each lands in many lane
    # and thread positions.
    inf = ((1 << eb) - 1) << fb
    for _ in range(16):
        out += [(x, y) for x in (0, 0x8000) for y in (0, 0x8000)]
        out += [(inf, inf | 0x8000), (inf | 0x8000, inf), (inf, inf), (inf | 0x8000, inf | 0x8000)]
        out += [(random.choice(ns), inf), (0x8000, random.choice(ns))]
    while len(out) < 2 * N:
        out.append((random.getrandbits(16), random.getrandbits(16)))
    if len(out) > 2 * N:
        # Keep every structured pair; drop the excess from the NaN-with-1.0 sweep's tail only
        # when it would not fit (it does not happen with the sets above for bf16).
        raise SystemExit(f"{len(out)} pairs > {2 * N}")
    random.shuffle(out)
    return out


def words(ps):
    a = np.zeros(N, np.uint32)
    b = np.zeros(N, np.uint32)
    for t in range(N):
        (a0, b0), (a1, b1) = ps[2 * t], ps[2 * t + 1]
        a[t] = a1 << 16 | a0
        b[t] = b1 << 16 | b0
    return a, b


if __name__ == "__main__":
    ah, bh = words(pairs_for(F16, E16))
    ab, bb = words(pairs_for(BF16, EB))
    np.stack([ah, bh], 1).tofile("inh.bin")
    np.stack([ab, bb], 1).tofile("inb.bin")
    np.stack([ah, ah, ah, ah, ab, ab, ab, ab], 0).tofile("m.bin")

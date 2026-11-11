# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""probes/f2i_neg: check.py [OUT.bin] compares a GB300 output (default expected/out.bin) with
exact arithmetic: out[j][g] = the conversion of form j applied to in[g] with its sign bit
flipped. Also counts the words on which three other readings of `-Ra` would differ: the
modifier ignored, the integer result negated instead of the source, and |Ra|."""

import math
import sys
from fractions import Fraction

import numpy as np

N = 2048
x = np.fromfile("in.bin", np.uint32)
out = np.fromfile(sys.argv[1] if len(sys.argv) > 1 else "expected/out.bin", np.uint32)
assert x.size == N and out.size == 16 * N


def f2i(u, rnd, signed, ftz):
    """F2I of the f32 with bits u: round to an integer, saturate, NaN 0."""
    e, m = u >> 23 & 0xFF, u & 0x7FFFFF
    if e == 0xFF and m:
        return 0
    lo, hi = (-(2**31), 2**31 - 1) if signed else (0, 2**32 - 1)
    if e == 0xFF:
        return (lo if u >> 31 else hi) & 0xFFFFFFFF
    if e == 0 and ftz:
        m = 0
    v = Fraction(m | (1 << 23 if e else 0)) * Fraction(2) ** (max(e, 1) - 150)
    v = -v if u >> 31 else v
    if rnd == 0:  # nearest, ties to even
        r = math.floor(v)
        d = v - r
        r += 1 if d > Fraction(1, 2) or d == Fraction(1, 2) and r % 2 else 0
    elif rnd == 1:  # toward zero
        r = math.trunc(v)
    elif rnd == 2:
        r = math.floor(v)
    else:
        r = math.ceil(v)
    return min(max(r, lo), hi) & 0xFFFFFFFF


bad = 0
alt = {"modifier ignored": 0, "result negated": 0, "|Ra|": 0}
for j in range(16):
    rnd, signed, ftz = j % 4, j // 4 % 2 == 0, j >= 8
    for g in range(N):
        u = int(x[g])
        got = int(out[j * N + g])
        if got != f2i(u ^ 0x80000000, rnd, signed, ftz):
            bad += 1
            if bad <= 8:
                print(f"form {j} input {u:#010x}: GB300 {got:#010x}")
        alt["modifier ignored"] += got != f2i(u, rnd, signed, ftz)
        alt["result negated"] += got != -f2i(u, rnd, signed, ftz) & 0xFFFFFFFF
        alt["|Ra|"] += got != f2i(u & 0x7FFFFFFF, rnd, signed, ftz)
print(f"{bad} of {16 * N} words differ from F2I(-x)")
for k, n in alt.items():
    print(f"  reading '{k}': {n} words differ")
sys.exit(1 if bad else 0)

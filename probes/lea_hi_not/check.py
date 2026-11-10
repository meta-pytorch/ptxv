# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""probes/lea_hi_not: check.py [OUT.bin] compares a GB300 output (default expected/out.bin)
with 64-bit arithmetic, and each high word with the instruction's formula
  d = ((a >> (32 - s)) + ~b + P) mod 2^32,
b the subtrahend's high word and P the carry out of the low word. Also counts the high words
on which three other readings would differ: the carry-in ignored, `-b` for `~b`, and the
complement ignored."""

import sys

import numpy as np

N = 2048
M = 0xFFFFFFFF
ar = np.fromfile("ar.bin", np.uint32).reshape(N, 4)
x = np.fromfile("x.bin", np.uint64)
out = np.fromfile(sys.argv[1] if len(sys.argv) > 1 else "expected/out.bin", np.uint64)
assert out.size == 60 * N
bad = badhi = borrows = wraps = 0
alt = {"carry-in ignored": 0, "-b for ~b": 0, "complement ignored": 0}
for j in range(60):
    s = j % 30 + 1
    for g in range(N):
        a, c, r, v = int(ar[g][0]), int(ar[g][1]), int(ar[g][2]), int(x[g])
        if j < 30:
            minuend, sub = a << s, v
        else:
            a = c
            minuend, sub = (a << s) | r, (v << s) & (1 << 64) - 1
        want = (minuend - sub) & (1 << 64) - 1
        got = int(out[j * N + g])
        bad += got != want
        if got != want and bad <= 8:
            print(f"row {j} thread {g}: GB300 {got:#018x}, 64-bit arithmetic {want:#018x}")
        # The low word's carry out: (minuend_lo + ~sub_lo + 1) >> 32.
        p = ((minuend & M) + (~sub & M) + 1) >> 32
        b = sub >> 32
        hi = a >> (32 - s)
        full = hi + (~b & M) + p
        badhi += got >> 32 != full & M
        borrows += p == 0
        wraps += full >> 32
        alt["carry-in ignored"] += got >> 32 != (hi + (~b & M)) & M
        alt["-b for ~b"] += got >> 32 != (hi - b + p) & M
        alt["complement ignored"] += got >> 32 != (hi + b + p) & M
print(f"{bad} of {60 * N} words differ from 64-bit arithmetic; {badhi} high words differ from")
print(f"  (a >> (32 - s)) + ~b + P; {borrows} rows with a low-word borrow (P = 0), {wraps}")
print("  whose high-word sum carries out of 32 bits")
for k, n in alt.items():
    print(f"  reading '{k}': {n} high words differ")
sys.exit(1 if bad or badhi else 0)

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/int_sat input: 4096 x (a, b) 32-bit words (deterministic). 1024 pairs: every pair of 32
# edge words (0, +-1, +-2, INT_MIN / INT_MAX and neighbours, 2^30 edges, 16-bit lane edges, and
# 647 / 648 / 649, the overflow edge of the immediate 2147483000); 1024 at the saturation
# thresholds (a + b and a - b exactly INT_MAX / INT_MIN and one past); 1024 random; 1024 large
# same-sign and opposite-sign operands.
import random
import struct
import sys

M = 0xFFFFFFFF
IMIN, IMAX = -(1 << 31), (1 << 31) - 1
r = random.Random(23)
E = [
    0,
    1,
    2,
    M,
    M - 1,
    0x7FFFFFFF,
    0x7FFFFFFE,
    0x7FFFFFFD,
    0x80000000,
    0x80000001,
    0x80000002,
    0x40000000,
    0x3FFFFFFF,
    0xC0000000,
    0xBFFFFFFF,
    0x00008000,
    0x00007FFF,
    0x0000FFFF,
    0x00010000,
    0x80008000,
    0x7FFF7FFF,
    0xFFFF0001,
    0x7FFFFD78,
    0x80000288,
    0x00000286,
    0x00000287,
    0x00000288,
    0x00000289,
    0xFFFFFD78,
    0xFFFFFD77,
    0x7FFFFFFF ^ 0x10,
    0x80000010,
]
assert len(E) == 32
out = []
for a in E:
    for b in E:
        out += [a, b]


def s32(x):
    return x - (1 << 32) if x >> 31 else x


for k in range(1024):
    kind = k % 8
    if kind < 4:  # add thresholds
        if kind < 2:
            a = r.randint(1, IMAX)
            b = IMAX - a + (kind & 1)  # exact max, then one past
        else:
            a = r.randint(IMIN, -2)
            b = IMIN - a - (kind & 1)  # exact min, then one past
    else:  # sub thresholds
        if kind < 6:
            a = r.randint(0, IMAX)
            b = a - IMAX - (kind & 1)
        else:
            a = r.randint(IMIN, -2)
            b = a - IMIN + (kind & 1)
    assert IMIN <= b <= IMAX, (kind, a, b)
    out += [a & M, b & M]
for _ in range(1024):
    out += [r.getrandbits(32), r.getrandbits(32)]
for _ in range(1024):
    a = r.getrandbits(30) | 0x40000000
    b = r.getrandbits(30) | 0x40000000
    sa, sb = r.getrandbits(1), r.getrandbits(1)
    out += [(a ^ (M if sa else 0)) & M, (b ^ (M if sb else 0)) & M]
assert len(out) == 8192
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<8192I", *out)
)

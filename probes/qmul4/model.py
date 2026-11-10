# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""usage: model.py in.bin expected/out.bin. Model of QMUL4.E4M3x4.E2M1x4.E4M3x4 d, a.N3210 / .N7654, b.B0 and comparison with GB300."""

import struct
import sys
from fractions import Fraction as F


def e2m1(c):
    s = -1 if c & 8 else 1
    e, m = (c >> 1) & 3, c & 1
    v = F(m, 2) if e == 0 else (1 + F(m, 2)) * 2 ** (e - 1)
    return s, v


def e4m3(c):
    s = -1 if c & 0x80 else 1
    e, m = (c >> 3) & 15, c & 7
    if e == 15 and m == 7:
        return s, None  # NaN
    v = F(m, 8) * F(1, 64) if e == 0 else (1 + F(m, 8)) * F(2) ** (e - 7)
    return s, v


E4M3_VALS = sorted({e4m3(c)[1] for c in range(0x7F)})  # non-negative finite values


def to_e4m3(sign, v, sat):
    """Round |v| to nearest even e4m3; sat: clamp to 448, else NaN on overflow."""
    best = None
    # exact rounding via candidate neighbours
    lo = max([x for x in E4M3_VALS if x <= v], default=F(0))
    hi = min([x for x in E4M3_VALS if x >= v], default=None)
    if hi is None:
        if not sat and v >= 464:  # halfway between 448 and next (480): RN overflow
            return 0x7F
        r = F(448)
    elif lo == hi:
        r = lo
    else:
        if v - lo < hi - v:
            r = lo
        elif hi - v < v - lo:
            r = hi
        else:
            r = lo if (code(lo) & 1) == 0 else hi
    c = code(r)
    return c | (0x80 if sign < 0 else 0)


def code(v):
    for c in range(0x7F):
        if e4m3(c)[1] == v:
            return c
    raise ValueError(v)


def qmul(nib, sc, sat):
    sa, va = e2m1(nib)
    sb, vb = e4m3(sc)
    if vb is None:
        return 0x7F
    return to_e4m3(sa * sb, va * vb, sat)


inp = struct.unpack("<8192I", open(sys.argv[1], "rb").read())
out = struct.unpack("<8192I", open(sys.argv[2], "rb").read())
for sat in (True, False):
    bad = 0
    ex = []
    for t in range(4096):
        a, b = inp[2 * t], inp[2 * t + 1]
        for half in (0, 1):
            w = out[2 * t + half]
            for k in range(4):
                nib = (a >> (4 * (4 * half + k))) & 15
                want = qmul(nib, b & 255, sat)
                got = (w >> (8 * k)) & 255
                if got != want:
                    bad += 1
                    if len(ex) < 8:
                        ex.append((hex(nib), hex(b & 255), hex(got), hex(want)))
    print("sat" if sat else "nan-on-overflow", "mismatches", bad, ex)

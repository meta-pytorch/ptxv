#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""MISS 208 second sweep (after gen1.py): tells the result flush (a sum whose leading bit is
below 16 floor(eB / 16) - 23 gives +0) from a member drop (members below it dropped), and probes
the f32 add. 4 members, exact results only. Writes OUT/in4/<test>.in<i>.bin and OUT/lanes.json.
Kinds:
- pairw (bf16x8, bf16x8a, f16x8, f16x8a): (B, -B, R1, R2), R1 = +-2^k at depth lim - 6 + j
  (j = 0..5), R2 = R1 2^-6 (depth lim + j, below the floor for j >= 1): the sum R1 (1 + 2^-6)
  is above the floor (flush rule: kept), R2 is below it for j >= 1 (member drop: R1).
- cancelf (f32x4, f32): (B, -B, R, 0), R at depth d = 16..72 below B (f32 eB 40..230): R if
  kept (the f32 model keeps R within its 63-bit window).
usage: gen2.py OUT"""
import json
import os
import random
import struct
import sys

OUT = sys.argv[1]
rng = random.Random(2082)
NV = 1 << 16
FMT = {"bf16": (8, 7), "f16": (5, 10), "f32": (8, 23)}


def bits(fmt, s, e, m):
    eb, mb = FMT[fmt]
    return s << (eb + mb) | e << mb | m


def pow2(fmt, s, e):
    """+-2^(e - bias) (e: biased-equivalent exponent, below 1 for subnormals), or None."""
    eb, mb = FMT[fmt]
    if e >= 1:
        return bits(fmt, s, e, 0)
    if e - 1 + mb < 0:
        return None
    return bits(fmt, s, 0, 1 << (e - 1 + mb))


def enc(fmt, v):
    """The bits of v if exactly representable (normal or subnormal)."""
    eb, mb = FMT[fmt]
    bias = (1 << (eb - 1)) - 1
    s = int(v < 0)
    a = abs(v)
    import math
    e = math.floor(math.log2(a)) + bias
    if e >= 1:
        m = a / 2.0 ** (e - bias) - 1
        mm = round(m * (1 << mb))
        assert mm / (1 << mb) == m, v
        return bits(fmt, s, e, mm)
    mm = a / 2.0 ** (1 - bias - mb)
    assert mm == int(mm), v
    return bits(fmt, s, 0, int(mm))


def val(fmt, x):
    eb, mb = FMT[fmt]
    bias = (1 << (eb - 1)) - 1
    s = -1 if x >> (eb + mb) & 1 else 1
    e, m = (x >> mb) & ((1 << eb) - 1), x & ((1 << mb) - 1)
    return s * (m * 2.0 ** (1 - bias - mb) if e == 0 else (1 + m / 2 ** mb) * 2.0 ** (e - bias))


def rot(ms):
    r = rng.randrange(4)
    return ms[r:] + ms[:r]


man = {}
os.makedirs(f"{OUT}/in4", exist_ok=True)
for test, fmt in (("bf16x8", "bf16"), ("bf16x8a", "bf16"), ("f16x8", "f16"), ("f16x8a", "f16")):
    eb_, mb = FMT[fmt]
    sign = 1 << (eb_ + mb)
    lanes = []
    for eB in range(1, (1 << eb_) - 1):
        lim = 23 + eB % 16
        for j in range(6):
            eR1 = eB - (lim - 6 + j)
            s = rng.randrange(2)
            R1, R2 = pow2(fmt, s, eR1), pow2(fmt, s, eR1 - 6)
            if R1 is None or R2 is None:
                continue
            B = bits(fmt, rng.randrange(2), eB, rng.randrange(1 << mb))
            lanes.append((rot([B, B ^ sign, R1, R2]), "pairw", {"eB": eB, "j": j, "R1": R1, "R2": R2,
                                                                 "sum": enc(fmt, val(fmt, R1) + val(fmt, R2))}))
    while len(lanes) % 8:
        lanes.append(([0, 0, 0, 0], "pad", {}))
    for i in range(4):
        b = struct.pack("<%dH" % len(lanes), *[l[0][i] for l in lanes])
        open(f"{OUT}/in4/{test}.in{i}.bin", "wb").write(b.ljust(16 * NV, b"\0"))
    man[f"4.{test}"] = [[l[0], l[1], l[2]] for l in lanes]
    print(test, len(lanes), "lanes")
lanes = []
for eB in range(40, 231, 3):
    for d in range(16, 73):
        eR = eB - d
        if eR < 1:
            continue
        B = bits("f32", rng.randrange(2), eB, rng.randrange(1 << 23))
        R = bits("f32", rng.randrange(2), eR, rng.randrange(1 << 23))
        lanes.append((rot([B, B ^ (1 << 31), R, 0]), "cancelf", {"eB": eB, "d": d, "R": R}))
while len(lanes) % 4:
    lanes.append(([0, 0, 0, 0], "pad", {}))
for test in ("f32x4", "f32"):
    for i in range(4):
        b = struct.pack("<%dI" % len(lanes), *[l[0][i] for l in lanes])
        open(f"{OUT}/in4/{test}.in{i}.bin", "wb").write(b.ljust(16 * NV, b"\0"))
    man[f"4.{test}"] = [[l[0], l[1], l[2]] for l in lanes]
    print(test, len(lanes), "lanes")
json.dump(man, open(f"{OUT}/lanes.json", "w"))

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""MISS 208 discriminating sweep for multimem.ld_reduce f16 / bf16 add (mm ldr with MM_IN).
Writes OUT/in4/<test>.in<i>.bin (4 members), OUT/in2/<test>.in<i>.bin (2 members) and
OUT/lanes.json: per (members, test) the lanes, each with its members' bits, a kind and the value
each candidate rule predicts. Every lane's result is exact (no rounding) except the 2-member
probes, whose other three lanes of the 4-lane group are 1.0 (zero mantissa: the calibrated
rounding's dither is 0, so an inexact sum rounds toward zero).
Kinds (B: the largest member, biased exponent eB; R: a member d binades below it; lim = 23 + eB
mod 16, the deepest R the alignment rule keeps for bf16):
- cancel4: (B, -B, R, 0) rotated over the members, d in lim-2 .. lim+2 for every bf16 eB
  1..254 (R normal), or d 1 .. eB+9 for every f16 eB (R normal or subnormal): R if kept, +0 if
  dropped.
- pair4: (B, -B, R1, R2), R1 = +-2^k at depth lim, R2 = R1 / 2 (depth lim + 1) or 2 R1
  (depth lim - 1): which of the two small members is kept.
- probe2: (B, -R) with R at depth d (2 members): B's previous encoding if R is kept, B if
  dropped; control lanes at d = 9..12.
- zero: every member -0 / mixed zero signs / x + -x with -0 members (both member counts).
f32 / f32x4: zero lanes only (does the f32 add give +0 for all -0 members too?).
usage: gen1.py OUT"""
import json
import os
import random
import struct
import sys

OUT = sys.argv[1]
rng = random.Random(208)
NV = 1 << 16  # vectors of 16 bytes per member file (mm ldr's buffer)
FMT = {"bf16": (8, 7), "f16": (5, 10), "f32": (8, 23)}


def bits(fmt, s, e, m):
    eb, mb = FMT[fmt]
    return s << (eb + mb) | e << mb | m


def lim_of(eb):
    return 23 + eb % 16


def lanes_half(fmt, n):
    """Lane tuples (members' bits, kind, info) for a 16-bit format and n members."""
    eb_, mb = FMT[fmt]
    emax = (1 << eb_) - 2
    out = []
    if n == 4:
        for eB in range(1, emax + 1):
            ds = range(lim_of(eB) - 2, lim_of(eB) + 3) if fmt == "bf16" else range(1, eB + mb)
            for d in ds:
                eR = eB - d
                for rep in range(2 if fmt == "bf16" else 1):
                    sB, sR = rng.randrange(2), rng.randrange(2)
                    B = bits(fmt, sB, eB, rng.randrange(1 << mb))
                    if eR >= 1:
                        R = bits(fmt, sR, eR, rng.randrange(1 << mb))
                    elif eR > -mb:
                        # subnormal: top bit d binades below B's
                        R = bits(fmt, sR, 0, (1 << (mb - 1 + eR)) | rng.randrange(1 << (mb - 1 + eR)))
                    else:
                        continue
                    ms = [B, B ^ (1 << (eb_ + mb)), R, 0]
                    rot = rng.randrange(4)
                    ms = ms[rot:] + ms[:rot]
                    out.append((ms, "cancel4", {"eB": eB, "d": d, "R": R}))
        if fmt == "bf16":
            for eB in range(1, emax + 1):
                lim = lim_of(eB)
                eR1 = eB - lim
                if eR1 < 3:
                    continue
                for other in (1, -1):
                    s = rng.randrange(2)
                    B = bits(fmt, rng.randrange(2), eB, rng.randrange(1 << mb))
                    R1 = bits(fmt, s, eR1, 0)
                    R2 = bits(fmt, s, eR1 - other, 0)  # R1 / 2 (depth lim + 1) or 2 R1 (lim - 1)
                    ms = [B, B ^ (1 << (eb_ + mb)), R1, R2]
                    rot = rng.randrange(4)
                    ms = ms[rot:] + ms[:rot]
                    out.append((ms, "pair4", {"eB": eB, "R1": R1, "R2": R2, "R2_depth": lim + other}))
    else:
        for eB in range(1, emax + 1):
            ds = list(range(9, 13)) + (list(range(lim_of(eB) - 2, lim_of(eB) + 3)) if fmt == "bf16" else list(range(13, eB + mb)))
            for d in ds:
                eR = eB - d
                if eR < 1:
                    continue
                B = bits(fmt, 0, eB, 1 + rng.randrange((1 << mb) - 1))
                R = bits(fmt, 1, eR, rng.randrange(1 << mb))
                ms = [B, R] if rng.randrange(2) else [R, B]
                out.append((ms, "probe2", {"eB": eB, "d": d, "B": B}))
    nz = 1 << (eb_ + mb)
    zs = [[nz] * n, [0] + [nz] * (n - 1), [nz] * (n - 1) + [0]]
    one = bits(fmt, 0, (1 << (eb_ - 1)) - 1, 0)
    if n == 4:
        zs += [[one, one ^ nz, nz, nz], [one ^ nz, one, nz, nz]]
    else:
        zs += [[one, one ^ nz], [one ^ nz, one]]
    for z in zs:
        for _ in range(4):
            out.append((list(z), "zero", {}))
    return out


def pack_half(fmt, lanes, n):
    """Vectors of 8 lanes. probe2 lanes sit in lane 0 of a 4-lane group whose lanes 1..3 are
    (1.0, +0): exact 1.0 results with zero mantissas."""
    eb_, mb = FMT[fmt]
    one = bits(fmt, 0, (1 << (eb_ - 1)) - 1, 0)
    filler = [one] + [0] * (n - 1)
    slots = []  # 4-lane groups
    plain = [l for l in lanes if l[1] != "probe2"]
    for l in lanes:
        if l[1] == "probe2":
            slots.append([l, (filler, "filler", {}), (filler, "filler", {}), (filler, "filler", {})])
    for i in range(0, len(plain), 4):
        g = plain[i : i + 4]
        while len(g) < 4:
            g.append(([0] * n, "pad", {}))
        slots.append(g)
    if len(slots) % 2:
        slots.append([([0] * n, "pad", {})] * 4)
    return [slots[i] + slots[i + 1] for i in range(0, len(slots), 2)]


def write(sub, test, vecs, n, w):
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    assert len(vecs) <= NV
    for i in range(n):
        b = b"".join(struct.pack("<%d%s" % (len(v), "H" if w == 2 else "I"), *[l[0][i] for l in v]) for v in vecs)
        open(f"{OUT}/{sub}/{test}.in{i}.bin", "wb").write(b.ljust(16 * NV, b"\0"))


manifest = {}
for n, sub in ((4, "in4"), (2, "in2")):
    for test, fmt in (("bf16x8", "bf16"), ("bf16x8a", "bf16"), ("f16x8", "f16"), ("f16x8a", "f16")):
        vecs = pack_half(fmt, lanes_half(fmt, n), n)
        write(sub, test, vecs, n, 2)
        manifest[f"{n}.{test}"] = [[list(l[0]), l[1], l[2]] for v in vecs for l in v]
        print(f"m{n} {test}: {len(vecs)} vectors, {sum(1 for v in vecs for l in v if l[1] not in ('pad', 'filler'))} lanes")
    # f32: zero lanes, 4-byte lanes (f32x4: 4 per vector; f32: one per 4 bytes)
    nz = 1 << 31
    one = 0x3F800000
    zl = [[nz] * n, [0] + [nz] * (n - 1), [nz] * (n - 1) + [0]] + ([[one, one ^ nz, nz, nz]] if n == 4 else [[one, one ^ nz]])
    lanes = [(z, "zero", {}) for z in zl for _ in range(4)]
    vecs = [lanes[i : i + 4] for i in range(0, len(lanes), 4)]
    for test in ("f32x4", "f32"):
        write(sub, test, vecs, n, 4)
        manifest[f"{n}.{test}"] = [[list(l[0]), l[1], l[2]] for v in vecs for l in v]
        print(f"m{n} {test}: {len(vecs)} vectors, {len(lanes)} lanes")
json.dump(manifest, open(f"{OUT}/lanes.json", "w"))

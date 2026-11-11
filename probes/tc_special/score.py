# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Independent scorer for probes/tc_special (N-MMA-SPECIAL): 64 candidate models of
tcgen05.mma.kind::f16 on special values, in exact rational arithmetic, against the GB300 outputs
(<set>/expected/D.bin).

    python3 score.py probes/tc_special f16_sub_align bf16_res_sub ...   (slow: minutes per set)
"""

import collections
import itertools
import math
import struct
import sys
from fractions import Fraction

D = sys.argv[1]
SETS = sys.argv[2:]
M, N, K = 128, 64, 64


def dec(fmt, c):
    if fmt == "f16":
        return struct.unpack("<e", struct.pack("<H", c))[0]
    return struct.unpack("<f", struct.pack("<I", c << 16))[0]


def expo(x):  # floor(log2 |x|), x finite nonzero
    m, e = math.frexp(abs(x))
    return e - 1


def is_sub(fmt, c):
    eb, mb = (5, 10) if fmt == "f16" else (8, 7)
    return (c >> mb) & ((1 << eb) - 1) == 0 and c & ((1 << mb) - 1) != 0


def f32_rz(fr):
    """Fraction -> f32 bits, round toward zero, subnormals kept; None on >= 2^128."""
    if fr == 0:
        return 0
    s = 0x80000000 if fr < 0 else 0
    a = abs(fr)
    e = math.floor(math.log2(a.numerator) - math.log2(a.denominator))
    # fix e exactly
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    if e >= 128:
        return None
    if e < -126:
        q = math.floor(a / Fraction(2) ** -149)
        return s | q
    mant = math.floor(a / Fraction(2) ** (e - 23))  # 24 bits
    return s | ((e + 127) << 23) | (mant & 0x7FFFFF)


def bits_f(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def step(acc_bits, prods, cfg, fmt):
    """prods: list of (a_code, b_code). Returns f32 bits."""
    emin_in = -14 if fmt == "f16" else -126
    nanv = cfg["nan"]
    vals = []
    pinf = ninf = False
    for ca, cb in prods:
        a, b = dec(fmt, ca), dec(fmt, cb)
        if cfg["ftz_in"]:
            if is_sub(fmt, ca):
                a = math.copysign(0.0, a)
            if is_sub(fmt, cb):
                b = math.copysign(0.0, b)
        if (
            math.isnan(a)
            or math.isnan(b)
            or (math.isinf(a) and b == 0)
            or (math.isinf(b) and a == 0)
        ):
            return nanv
        if math.isinf(a) or math.isinf(b):
            if (a > 0) == (b > 0):
                pinf = True
            else:
                ninf = True
            continue
        vals.append((a, b))
    acc = bits_f(acc_bits)
    if math.isnan(acc):
        return nanv
    if math.isinf(acc):
        if acc > 0:
            pinf = True
        else:
            ninf = True
    if pinf and ninf:
        return nanv
    if pinf:
        return 0x7F800000
    if ninf:
        return 0xFF800000
    if cfg["ftz_acc"] and acc != 0 and abs(acc) < 2.0**-126:
        acc = 0.0
    ex_in = (lambda x: max(expo(x), emin_in)) if cfg["emin"] == "field" else expo
    ex_acc = (lambda x: max(expo(x), -126)) if cfg["emin"] == "field" else expo
    terms = [
        (Fraction(a) * Fraction(b), ex_in(a) + ex_in(b))
        for a, b in vals
        if a != 0 and b != 0
    ]
    if acc != 0:
        terms.append((Fraction(acc), ex_acc(acc)))
    if not terms:
        return 0
    q = max(e for _, e in terms) - 25
    s = 0
    for v, _ in terms:
        x = v / Fraction(2) ** q
        s += math.floor(x) if x >= 0 else -math.floor(-x)
    if s == 0:
        return 0
    r = f32_rz(Fraction(s) * Fraction(2) ** q)
    if r is None:
        if cfg["ovf"] == "inf":
            return 0x7F800000 if s > 0 else 0xFF800000
        return 0x7F7FFFFF if s > 0 else 0xFF7FFFFF
    if cfg["ftz_res"] and (r & 0x7F800000) == 0:
        r &= 0x80000000
    return r


def run(fmt, A, B, cfg):
    out = []
    for m in range(M):
        for n in range(N):
            acc = 0
            for j in range(4):
                prods = [(A[m][k], B[n][k]) for k in range(16 * j, 16 * j + 16)]
                acc = step(acc, prods, cfg, fmt)
            out.append(acc)
    return out


CFGS = []
for emin, fi, fa, fr, nan, ovf in itertools.product(
    ["field", "true"],
    [0, 1],
    [0, 1],
    [0, 1],
    [0x7FFFFFFF, 0x7FC00000],
    ["inf", "fltmax"],
):
    CFGS.append(dict(emin=emin, ftz_in=fi, ftz_acc=fa, ftz_res=fr, nan=nan, ovf=ovf))

score = collections.Counter()
tot = 0
for s in SETS:
    fmt = s.split("_")[0]
    ra = open(f"{D}/{s}/A.bin", "rb").read()
    rb = open(f"{D}/{s}/B.bin", "rb").read()
    a = struct.unpack("<%dH" % (M * K), ra)
    b = struct.unpack("<%dH" % (N * K), rb)
    A = [a[m * K : (m + 1) * K] for m in range(M)]
    B = [b[n * K : (n + 1) * K] for n in range(N)]
    hw = struct.unpack("<%dI" % (M * N), open(f"{D}/{s}/expected/D.bin", "rb").read())
    nans = collections.Counter(
        x for x in hw if (x & 0x7F800000) == 0x7F800000 and x & 0x7FFFFF
    )
    subs = sum(1 for x in hw if (x & 0x7F800000) == 0 and x & 0x7FFFFF)
    infs = sum(1 for x in hw if (x & 0x7FFFFFFF) == 0x7F800000)
    line = []
    best = None
    for i, c in enumerate(CFGS):
        o = run(fmt, A, B, c)
        ok = sum(1 for x, y in zip(o, hw) if x == y)
        score[i] += ok
        line.append(ok)
    tot += M * N
    full = [i for i, v in enumerate(line) if v == M * N]
    print(
        f"{s}: hw NaN patterns {dict((hex(k), v) for k, v in nans.items())} subnormal {subs} inf {infs}; "
        f"best {max(line)}/{M * N}; full-match cfgs {len(full)}"
    )
print("total outputs", tot)
for i, v in score.most_common(8):
    print(v, CFGS[i])
print("cfgs matching all:", [CFGS[i] for i, v in score.items() if v == tot])

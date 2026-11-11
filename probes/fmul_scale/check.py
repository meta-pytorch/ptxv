import math
import struct
import sys
from fractions import Fraction as F

d = sys.argv[1]
o = sys.argv[2]
A = struct.unpack("<1024I", open(d + "/a.bin", "rb").read())
B = struct.unpack("<1024I", open(d + "/b.bin", "rb").read())
out = struct.unpack("<10240I", open(o, "rb").read())


def val(u):
    s = -1 if u >> 31 else 1
    e = (u >> 23) & 255
    m = u & 0x7FFFFF
    if e == 255:
        return "nan" if m else ("inf", s)
    if e == 0:
        return F(s * m, 2**149)
    return F(s * (m | 1 << 23)) * F(2) ** (e - 150)


def rne(x, neg_zero=False, ftz=False):
    # x: Fraction (exact) -> f32 bits, RNE, subnormals unless ftz
    if x == 0:
        return 0x80000000 if neg_zero else 0
    s = 0x80000000 if x < 0 else 0
    ax = abs(x)
    e = math.floor(math.log2(ax.numerator) - math.log2(ax.denominator))
    while F(2) ** e > ax:
        e -= 1
    while F(2) ** (e + 1) <= ax:
        e += 1
    if e > 127:
        return s | 0x7F800000
    q = max(e, -126) - 23
    m = ax / F(2) ** q
    fl = m.numerator // m.denominator
    r = m - fl
    if r > F(1, 2) or (r == F(1, 2) and fl & 1):
        fl += 1
    if fl >= 1 << 24:
        fl >>= 1
        q += 1
    if q + 23 > 127 and fl >= 1 << 23:
        return s | 0x7F800000
    if fl < 1 << 23:  # subnormal
        if ftz:
            return s
        return s | fl
    E = q + 23 + 127
    if E >= 255:
        return s | 0x7F800000
    return s | (E << 23) | (fl & 0x7FFFFF)


def mul(ua, ub, scale=F(1), ftz_in=False):
    a, b = val(ua), val(ub)
    sa = -1 if ua >> 31 else 1
    sb = -1 if ub >> 31 else 1
    if ftz_in:
        if isinstance(a, F) and (ua & 0x7F800000) == 0:
            a = F(0)
        if isinstance(b, F) and (ub & 0x7F800000) == 0:
            b = F(0)
    if a == "nan" or b == "nan":
        return "nan"
    ia = isinstance(a, tuple)
    ib = isinstance(b, tuple)
    if ia or ib:
        other = b if ia else a
        if not isinstance(other, tuple) and other == 0:
            return "nan"
        return ("inf", sa * sb)
    return (a * b * scale, sa * sb < 0)


def enc(r, ftz=False, sat=False):
    if r == "nan":
        return 0 if sat else 0x7FFFFFFF
    if isinstance(r[0], str):
        if sat:
            return 0x3F800000 if r[1] > 0 else 0
        return 0x7F800000 | (0x80000000 if r[1] < 0 else 0)
    x, neg = r
    if sat:
        if x <= 0:
            return 0
        if x >= 1:
            return 0x3F800000
    return rne(x, neg, ftz)


def double(ua, ub, scale, ftz=False, sat=False):
    r = mul(ua, ub, F(1), ftz)
    if r == "nan":
        return enc(r, sat=sat)
    if isinstance(r[0], str):
        return enc(r, sat=sat)
    t = rne(r[0], r[1], ftz)
    r2 = mul(t, struct.unpack("<I", struct.pack("<f", float(scale)))[0], F(1), ftz)
    return enc(r2, ftz, sat)


ops = [
    ("D2", F(1, 2), 0, 0),
    ("D4", F(1, 4), 0, 0),
    ("D8", F(1, 8), 0, 0),
    ("M2", F(2), 0, 0),
    ("M4", F(4), 0, 0),
    ("FTZ.D2", F(1, 2), 1, 0),
    ("D2.SAT", F(1, 2), 0, 1),
    ("REF2", F(1, 2), 0, 0),
    ("M8", F(8), 0, 0),
    ("D2cf", F(1, 2), 0, 0),
]


def same(x, y):
    nx = (x & 0x7F800000) == 0x7F800000 and x & 0x7FFFFF
    ny = (y & 0x7F800000) == 0x7F800000 and y & 0x7FFFFF
    return x == y


for k, (n, sc, ftz, sat) in enumerate(ops):
    h1 = h2 = 0
    ex1 = ex2 = None
    for i in range(1024):
        g = out[k * 1024 + i]
        v1 = enc(mul(A[i], B[i], sc, bool(ftz)), bool(ftz), bool(sat))
        v2 = double(A[i], B[i], sc, bool(ftz), bool(sat))
        if g != v1:
            h1 += 1
            ex1 = ex1 or (i, hex(A[i]), hex(B[i]), hex(g), hex(v1))
        if g != v2:
            h2 += 1
            ex2 = ex2 or (i, hex(A[i]), hex(B[i]), hex(g), hex(v2))
    print(
        f"{n:7s} single-rounding mismatches {h1:4d} {ex1}\n        double-rounding mismatches {h2:4d} {ex2}"
    )

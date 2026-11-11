"""Model of FHADD / FHFMA without .BF16 (f16 half widened exactly, f32 add / fma, one rounding in
the instruction's mode, gradual underflow, canonical NaN 0x7fffffff, .SAT clamps to [0, 1] with
NaN -> 0), compared with GB300. usage: model.py IN OUT"""

import sys

import numpy as np

np.seterr(all="ignore")
inp = np.fromfile(sys.argv[1], np.uint32).reshape(-1, 2)
out = np.fromfile(sys.argv[2], np.uint32).reshape(-1, 8)
lo = (inp[:, 0] & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float64)
hi = (inp[:, 0] >> 16).astype(np.uint16).view(np.float16).astype(np.float64)
c = inp[:, 1].view(np.float32).astype(np.float64)


def rnd(x, y, mode, sat=False):
    """f32 rounding of x + y (x, y exact f64; x + y computed with TwoSum)."""
    s = x + y
    bb = s - x
    e = (x - (s - bb)) + (y - bb)
    e = np.where(np.isfinite(s), e, 0.0)
    f = s.astype(np.float32)
    f64 = f.astype(np.float64)
    # bracket the true value t = s + e between f32 neighbours lo32 <= t <= hi32
    up = np.nextafter(f, np.float32(np.inf))
    dn = np.nextafter(f, np.float32(-np.inf))
    t_gt = (s > f64) | ((s == f64) & (e > 0))
    t_lt = (s < f64) | ((s == f64) & (e < 0))
    exact = (s == f64) & (e == 0)
    lo32 = np.where(t_gt, f, np.where(t_lt, dn, f))
    hi32 = np.where(t_gt, up, np.where(t_lt, f, f))
    if mode == "rz":
        r = np.where(s > 0, lo32, hi32)
        r = np.where((s == 0), f, r)
    elif mode == "rm":
        r = lo32
    elif mode == "rp":
        r = hi32
    else:
        r = f  # cast is RN of s; fix double rounding at exact midpoints of s with e != 0
        mid = (lo32.astype(np.float64) + hi32.astype(np.float64)) / 2
        at_mid = (s == mid) & (e != 0) & ~exact
        r = np.where(at_mid, np.where(e > 0, hi32, lo32), r)
    r = np.where(exact, f, r)
    # signed zeros: an exact zero sum is +0 except in rm (and -0 + -0 = -0)
    zero = (s == 0) & (e == 0)
    both_neg = np.signbit(x) & np.signbit(y)
    zs = np.where(
        both_neg | (mode == "rm") & ~(np.signbit(x) == np.signbit(y)) & False, -0.0, 0.0
    )
    if mode == "rm":
        zs = np.where(both_neg | (np.signbit(x) != np.signbit(y)), -0.0, 0.0)
    else:
        zs = np.where(both_neg, -0.0, 0.0)
    r = np.where(zero, zs.astype(np.float32), r)
    # overflow by mode
    big = np.float32(3.4028234663852886e38)
    if mode == "rz":
        r = np.where(
            np.isinf(r) & np.isfinite(s), np.copysign(big, s).astype(np.float32), r
        )
    if mode == "rm":
        r = np.where(np.isinf(r) & np.isfinite(s) & (s > 0), big, r)
    if mode == "rp":
        r = np.where(np.isinf(r) & np.isfinite(s) & (s < 0), -big, r)
    bits = r.astype(np.float32).view(np.uint32)
    nan = np.isnan(s)
    bits = np.where(nan, np.uint32(0x7FFFFFFF), bits)
    if sat:
        v = bits.view(np.float32)
        v = np.where(np.isnan(v), 0, np.clip(v, 0, 1)).astype(np.float32)
        v = np.where(v == 0, np.float32(0), v)
        bits = v.view(np.uint32)
    return bits


ref = [
    rnd(lo, c, "rn"),
    rnd(hi, -c, "rn"),
    rnd(lo, c, "rz"),
    rnd(hi, c, "rm"),
    rnd(lo, c, "rp"),
    rnd(hi, c, "rn", True),
    rnd(lo * hi, c, "rn"),
    rnd(lo, -c, "rz", True),
]
names = [
    "add.rn lo",
    "sub.rn hi",
    "add.rz lo",
    "add.rm hi",
    "add.rp lo",
    "add.rn.sat hi",
    "fma.rn lo*hi",
    "sub.rz.sat lo",
]
tot = 0
for k in range(8):
    m = out[:, k] != ref[k]
    tot += m.sum()
    ex = np.nonzero(m)[0][:3]
    print(
        f"{names[k]:14} {m.sum():8d}",
        [
            f"a={inp[i, 0]:08x} c={inp[i, 1]:08x} hw={out[i, k]:08x} model={ref[k][i]:08x}"
            for i in ex
        ],
    )
print("total", tot, "of", out.size)

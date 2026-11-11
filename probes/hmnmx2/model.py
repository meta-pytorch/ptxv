# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Independent model of HMNMX2 (f16x2), compared with GB300 output (the exhaustive run).
usage: model.py IN OUT  (u32 [n,2] inputs, u32 [n,16] outputs of kernel.ptx)"""

import sys

import numpy as np


def isnan(h):
    return ((h & 0x7C00) == 0x7C00) & ((h & 0x3FF) != 0)


def key(
    h,
):  # sign-magnitude order on f16 bits (-0 < +0); same order as the f32 widening
    h = h.astype(np.int64)
    return np.where(h >> 15 == 1, -(h & 0x7FFF) - 1, h)


def mnmx(x, y, mn, nan):
    an, bn = isnan(x), isnan(y)
    lt = key(x) < key(y)
    r = np.where(lt == mn, x, y)
    if nan:
        r = np.where(an | bn, 0x7FFF, r)
    else:
        r = np.where(an & ~bn, y, r)
        r = np.where(bn & ~an, x, r)
        r = np.where(an & bn, 0x7FFF, r)
    return r.astype(np.uint32)


def xs(x, y, mn, nan):
    r = mnmx(x & 0x7FFF, y & 0x7FFF, mn, nan)
    return np.where(isnan(r), r, r | ((x ^ y) & 0x8000)).astype(np.uint32)


def lanes(w):
    return [w & 0xFFFF, w >> 16]


def pair(f, a, b):
    (a0, a1), (b0, b1) = lanes(a), lanes(b)
    return f(a0, b0) | f(a1, b1) << 16


inp = np.fromfile(sys.argv[1], np.uint32).reshape(-1, 2)
out = np.fromfile(sys.argv[2], np.uint32).reshape(-1, 16)
a, b = inp[:, 0], inp[:, 1]
absb = b & 0x7FFF7FFF
c = lambda v: np.full_like(a, v)
ref = [
    pair(lambda x, y: mnmx(x, y, True, False), a, b),
    pair(lambda x, y: mnmx(x, y, False, False), a, b),
    pair(lambda x, y: mnmx(x, y, True, True), a, b),
    pair(lambda x, y: mnmx(x, y, False, True), a, b),
    pair(lambda x, y: xs(x, y, True, False), a, b),
    pair(lambda x, y: xs(x, y, False, False), a, b),
    pair(lambda x, y: xs(x, y, True, True), a, b),
    pair(lambda x, y: xs(x, y, False, True), a, b),
    pair(lambda x, y: mnmx(x, y, True, False), a, absb),
    pair(lambda x, y: mnmx(x, y, False, True), a, absb),
    pair(lambda x, y: xs(x, y, True, False), a, c(0x7BFF7BFF)),
    pair(lambda x, y: xs(x, y, False, False), a, c(0x3C003C00)),
    pair(lambda x, y: mnmx(x, y, True, False), a, c(0x7BFF7BFF)),
    pair(lambda x, y: mnmx(x, y, False, True), a, c(0xFC00FC00)),
    mnmx(a >> 16, b & 0xFFFF, True, False)
    | xs(a & 0xFFFF, b >> 16, False, False) << 16,
    a,
]
names = [
    "min",
    "max",
    "min.NaN",
    "max.NaN",
    "min.xs",
    "max.xs",
    "min.NaN.xs",
    "max.NaN.xs",
    "min |b|",
    "max.NaN |b|",
    "min.xs 65504",
    "max.xs 1",
    "min 65504",
    "max.NaN -inf",
    "scalar min/max.xs",
    "a",
]
bad = 0
for k, r in enumerate(ref):
    m = out[:, k] != r
    bad += m.sum()
    ex = np.nonzero(m)[0][:4]
    print(
        f"{names[k]:18} mismatches {m.sum():8d}",
        [f"a={a[i]:08x} b={b[i]:08x} hw={out[i, k]:08x} ref={r[i]:08x}" for i in ex],
    )
print("total mismatches", bad, "of", out.size)

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""check.py [OUT.bin]: per shape, how many of GB300's 1024 results differ from the IEEE
round-to-nearest quotient composed with the shape's other operations (float64 then f32,
subnormals kept): div.approx.ftz is not the IEEE division, so this counts the inputs on which
the measured MUFU.RCP + FMUL.FTZ semantics (the model) are exercised. NaNs compare equal."""

import math
import struct
import sys

N = 1024
f = lambda name: struct.unpack(f"<{N}f", open(name, "rb").read())
a, b, c = f("a.bin"), f("b.bin"), f("c.bin")
out = struct.unpack(
    f"<{16 * N}I",
    open(sys.argv[1] if len(sys.argv) > 1 else "expected/out.bin", "rb").read(),
)


def f32(x):
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def q(x, y):
    if y == 0:
        return (
            math.nan
            if x == 0 or math.isnan(x)
            else math.copysign(math.inf, x) * math.copysign(1, y)
        )
    if math.isinf(y):
        return (
            math.nan
            if math.isinf(x) or math.isnan(x)
            else 0.0 * math.copysign(1, x) * math.copysign(1, y)
        )
    return x / y


ref = [
    lambda i: q(a[i], b[i]),
    lambda i: q(1.0, b[i]),
    lambda i: q(-1.0, b[i]),
    lambda i: q(3.0, b[i]),
    lambda i: f32(q(a[i], b[i])) + c[i],
    lambda i: f32(q(a[i], b[i])) + c[i],
    lambda i: f32(q(a[i], b[i])) * 0.5,
    lambda i: q(-a[i], b[i]),
    lambda i: q(a[i], abs(b[i])),
    lambda i: q(a[i], 2.0**127),
    lambda i: c[i] - f32(q(a[i], b[i])),
    lambda i: f32(q(a[i], b[i])) * c[i] + a[i],
    lambda i: f32(q(a[i], b[i])) + f32(q(c[i], b[i])),
    lambda i: f32(q(a[i], b[i])) * 2.0,
    lambda i: q(b[i], b[i]),
    lambda i: q(a[i], b[i]) if c[i] > 0 else c[i],
]
for s in range(16):
    diff = 0
    for i in range(N):
        try:
            r = f32(ref[s](i))
        except (OverflowError, ValueError, ZeroDivisionError):
            r = math.nan
        g = struct.unpack("<f", struct.pack("<I", out[s * N + i]))[0]
        if not (math.isnan(r) and math.isnan(g)) and struct.pack(
            "<f", r
        ) != struct.pack("<f", g):
            diff += 1
    print(f"shape {s:2d}: {diff:4d} of {N} differ from the IEEE-quotient composition")

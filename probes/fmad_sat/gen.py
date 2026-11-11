# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/fmad_sat input: 8192 x (a, b, c) f32 words (deterministic). 4096: every triple of 16
# special values (+-0, the smallest and largest subnormals, the smallest normal, +-1 and its
# neighbours, 0.5, -2, the largest finite, +-Inf, NaNs with payloads and sign); 1024 with
# a * b straddling 1.0 (the clamp edge) and c of either sign; 1024 with a * b and a * c tiny or
# zero of either sign, c / b = +-0 (the sign of a clamped zero); 1024 with a * c close to -b (the
# FFMA.SAT sum cancels to either side of 0) and subnormal products; 1024 in [-4, 4].
import random
import struct
import sys

r = random.Random(41)


def f(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def fl(u):
    return struct.unpack("<f", struct.pack("<I", u))[0]


V = [
    0x00000000,
    0x80000000,
    0x00000001,
    0x807FFFFF,
    0x00800000,
    0x3F800000,
    0xBF800000,
    0x3F800001,
    0x3F7FFFFF,
    0x3F000000,
    0xC0000000,
    0x7F7FFFFF,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC00001,
]
out = []
for a in V:
    for b in V:
        for c in V:
            out += [a, b, c]
for _ in range(1024):  # a * b near 1
    a = r.uniform(0.5, 2.0)
    u = f(1.0 / a) + r.randint(-3, 3)
    out += [f(a), u, f(r.choice([-1, 1]) * r.uniform(0, 2))]
for _ in range(1024):  # tiny / zero products, signed zero addends
    a = r.choice([1, -1]) * r.uniform(1e-30, 1e-20)
    b = r.choice([1, -1]) * r.uniform(1e-30, 1e-18)
    out += [f(a), r.choice([f(b), 0x80000000, 0]), r.choice([0x80000000, 0, f(b)])]
for _ in range(1024):  # cancellation in the FFMA.SAT sum; subnormal products
    a = r.uniform(-2, 2)
    c = r.uniform(-2, 2)
    p = fl(f(a)) * fl(f(c))
    b = (
        f(-p) + r.randint(-2, 2)
        if r.random() < 0.7
        else r.choice([1, 2, 0x807FFFF0, 0x00400000])
    )
    out += [f(a), b & 0xFFFFFFFF, f(c) if r.random() < 0.7 else 0x00000100]
for _ in range(1024):
    out += [f(r.uniform(-4, 4)) for _ in range(3)]
assert len(out) == 3 * 8192
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<24576I", *out)
)

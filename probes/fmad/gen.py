# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/fmad input: 256 x (a, b, c) f32 words (deterministic). Mixed: random words over all
# exponents, products near -c (cancellation exposes the extra rounding), subnormals, signed
# zeros, Inf / NaN, values near 1 for the .sat forms.
import random
import struct
import sys

r = random.Random(7)
f = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]
out = []
for t in range(256):
    k = t % 8
    if k == 0:
        a, b, c = (r.getrandbits(32) for _ in range(3))
    elif k in (1, 2, 3):
        a = f(1 + r.random())
        b = f(1 + r.random())
        p = (
            struct.unpack("<f", struct.pack("<I", a))[0]
            * struct.unpack("<f", struct.pack("<I", b))[0]
        )
        c = f(-p) ^ r.choice([0, 1, 2])
    elif k == 4:
        a = r.getrandbits(23) | (r.getrandbits(1) << 31)
        b = f(r.choice([0.5, 1.5, 3.0]))
        c = r.getrandbits(23)
    elif k == 5:
        a, b, c = r.choice(
            [
                [0x80000000, 0x3F800000, 0],
                [0x7F800000, 0, 0x3F800000],
                [0x7FC00001, 0x3F800000, 0],
                [0x7F800000, 0x3F800000, 0xFF800000],
            ]
        )
    elif k == 6:
        a = f(r.uniform(-1, 2))
        b = f(r.uniform(-1, 2))
        c = f(r.uniform(-0.5, 0.5))
    else:
        a = f(r.uniform(-1e3, 1e3))
        b = f(r.uniform(-1e3, 1e3))
        c = f(r.uniform(-1e6, 1e6))
    out += [a, b, c]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<768I", *out)
)

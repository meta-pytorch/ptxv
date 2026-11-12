# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# fixtures/c_num_gather_f32 input (deterministic): x in [1, 2), y = f32(1/x); in even lanes the
# exact x*y lies in (1 - 2^-25, 1) and rounds to 1. Odd lanes: any pair whose product is not
# exactly 1. z uniform in [-1, 1).
import random
import struct
import sys
from fractions import Fraction as F

r = random.Random(36)


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


xy = []
while len(xy) < 32:
    a = f32(1 + r.random())
    b = f32(1 / a)
    e = F(a) * F(b)
    if len(xy) % 2 == 0:
        if F(1) - F(1, 2**25) < e < 1 and f32(a * b) == 1.0:
            xy.append((a, b))
    elif e != 1:
        xy.append((a, b))
z = [f32(r.uniform(-1, 1)) for _ in range(32)]
d = sys.argv[1] if len(sys.argv) > 1 else "data"
for name, vals in (("x", [p[0] for p in xy]), ("y", [p[1] for p in xy]), ("z", z)):
    open(f"{d}/{name}.bin", "wb").write(struct.pack("<32f", *vals))

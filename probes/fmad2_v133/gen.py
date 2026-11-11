# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/fmad2_v133 input: 128 threads x 32 words: five triples x, y, z of f32 pairs
# (x, y at word 4 k for 16-byte alignment, z at word 20 + 2 k), z = -(x y) rounded, +- 1 ulp,
# so a fused x y + z keeps the low bits a separate rounding loses.
import random
import struct
import sys

r = random.Random(17)
f = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]
g = lambda w: struct.unpack("<f", struct.pack("<I", w))[0]
out = []
for t in range(128):
    xy, zs = [], []
    for k in range(5):
        x = [f(1 + r.random()), f(-(1 + r.random()))]
        y = [f(1 + r.random()), f(1 + r.random())]
        z = [f(-g(x[i]) * g(y[i])) ^ r.choice([0, 1]) for i in range(2)]
        xy += x + y
        zs += z
    w = xy + zs + [0, 0]
    out += w
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<4096I", *out)
)

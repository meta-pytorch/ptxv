# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/sqrt_rn input: 4096 f32 bit patterns (deterministic): +-0, subnormals, normals at
# every exponent, +-inf, NaNs (quiet / signalling, both signs), negatives, random words.
import random
import struct
import sys

r = random.Random(41)
xs = [
    0,
    0x80000000,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC00000,
    0x7F800001,
    0x7FBFFFFF,
    0x00000001,
    0x80000001,
    0x007FFFFF,
    0x00400000,
    0x00800000,
    0x3F800000,
    0xBF800000,
    0x7F7FFFFF,
    0x40000000,
    0x40800000,
]
for e in range(256):
    for _ in range(6):
        xs.append((e << 23) | r.getrandbits(23))
for _ in range(64):
    xs.append(r.getrandbits(23))
while len(xs) < 4096:
    xs.append(r.getrandbits(32))
xs = xs[:4096]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<4096I", *xs)
)

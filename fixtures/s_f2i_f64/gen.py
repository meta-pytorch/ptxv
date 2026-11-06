# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# data/in.bin for fixtures/s_f2i_f64: 256 f64 operands. Special values (signed zeros,
# subnormals, infinities, NaNs with payloads and sign, halves and ties, the 32 / 64-bit integer
# range boundaries and their neighbours), then values of every magnitude class from a fixed
# seed. python3 gen.py > data/in.bin
import math
import random
import struct
import sys


def f(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def up(x):
    return f(x) + 1 if x >= 0 else f(x) - 1  # next away from zero (bit pattern)


bits = [
    0,
    1 << 63,
    1,
    (1 << 63) | 1,
    0x000FFFFFFFFFFFFF,
    0x0010000000000000,
    0x7FF0000000000000,
    0xFFF0000000000000,
    0x7FF8000000000000,
    0xFFF8000000000000,
    0x7FF0000000000001,
    0x7FF4000000000000,
    0xFFFFFFFFFFFFFFFF,
    0x7FFFFFFFFFFFFFFF,
    0x7FEFFFFFFFFFFFFF,
    0xFFEFFFFFFFFFFFFF,
]
vals = [
    0.5,
    -0.5,
    1.5,
    -1.5,
    2.5,
    -2.5,
    0.49999999999999994,
    -0.49999999999999994,
    1.0,
    -1.0,
    0.75,
    -0.75,
    1e-300,
    -1e-300,
    3.5,
    -3.5,
    1023.5,
    -1024.5,
    4503599627370495.5,
    -4503599627370495.5,
    4503599627370496.0,
    9007199254740993.0,
]
for b in (31, 32, 63, 64):
    for s in (1, -1):
        x = s * 2.0**b
        vals += [x]
        bits += [up(x), f(x) - 1 if x > 0 else f(x) + 1]
vals += [
    2147483647.0,
    2147483647.5,
    2147483646.5,
    -2147483648.5,
    -2147483649.0,
    4294967295.0,
    4294967295.5,
    4294967296.0,
    -0.9999999999999999,
    0.9999999999999999,
    9223372036854774784.0,
    -9223372036854774784.0,
    18446744073709549568.0,
]
bits += [f(v) for v in vals]
rng = random.Random(23)
while len(bits) < 192:
    e = rng.randrange(-20, 70)
    m = rng.random() * 2.0**e
    bits.append(f(m if rng.random() < 0.5 else -m))
while len(bits) < 256:
    bits.append(rng.getrandbits(64))
assert len(bits) == 256
sys.stdout.buffer.write(struct.pack("<256Q", *bits))

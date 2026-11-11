# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# in.bin: 1024 threads x 8 u32 (a pair, b pair, c pair, 2 unused) for probes/f32x2_ftz: products
# and sums landing on both sides of the f32 normal range's lower edge, subnormal and edge
# operands, and ordinary random values. Seed 11.
import random
import struct

random.seed(11)
EDGE = [
    0x00000000,
    0x80000000,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0x00800000,
    0x80800000,
    0x007FFFFF,
    0x807FFFFF,
    0x00000001,
    0x80000001,
    0x3F800000,
    0xBF800000,
    0x7F7FFFFF,
    0x00400000,
    0x00FFFFFF,
]


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def rnd_exp(lo, hi):
    m = random.uniform(1.0, 2.0) * random.choice([1.0, -1.0])
    return f2u(m * 2.0 ** random.randint(lo, hi))


def sub():
    return random.randint(1, 0x7FFFFF) | random.choice([0, 0x80000000])


words = []
for _i in range(1024):
    k = random.random()
    lanes = []
    for _ in range(2):
        if k < 0.25:  # product near 2^-126
            a, b = rnd_exp(-70, -56), rnd_exp(-70, -56)
            c = random.choice([0, 0x80000000, rnd_exp(-140, -118), sub()])
        elif k < 0.45:  # subnormal operands
            a = sub() if random.random() < 0.6 else rnd_exp(-10, 10)
            b = sub() if random.random() < 0.4 else rnd_exp(-10, 30)
            c = sub() if random.random() < 0.5 else rnd_exp(-130, -110)
        elif k < 0.65:  # a + c / a - c cancelling into the subnormal range
            x = random.uniform(1.0, 2.0) * 2.0 ** random.randint(-125, -110)
            a = f2u(x)
            c = f2u(-x * (1 + random.choice([1, -1]) * 2.0 ** -random.randint(1, 23)))
            if random.random() < 0.5:
                c ^= 0x80000000
            b = random.choice([0x3F800000, rnd_exp(-2, 2)])
        elif k < 0.80:
            a, b, c = random.choice(EDGE), random.choice(EDGE), random.choice(EDGE)
        else:
            a, b, c = rnd_exp(-20, 20), rnd_exp(-20, 20), rnd_exp(-20, 20)
        lanes.append((a, b, c))
    (a0, b0, c0), (a1, b1, c1) = lanes
    words += [a0, a1, b0, b1, c0, c1, 0, 0]
open("in.bin", "wb").write(struct.pack("<%dI" % len(words), *words))

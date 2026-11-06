# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# data/in.bin for fixtures/s_ptx_packed: 256 threads x 8 words (a0, a1, b0, b1, c0, c1, n, d).
# Lane 0 crosses 32 special f32 values (signed zeros, subnormals, infinities, NaNs with payloads,
# extremes, bf16 / f16 rounding ties and overflow boundaries); lane 1 holds products that
# underflow to subnormals or zero, overflow, cancel exactly against c (a fused vs an unfused
# result), and random bit patterns. (n, d) are div.s32 operands: INT_MIN / -1, signs, random
# (no zero divisor: its result is unspecified, a fault on the PTX front-end).
# python3 gen.py > data/in.bin
import random
import struct
import sys


def f(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def fl(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


S = [
    0x00000000,
    0x80000000,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC00001,
    0x7F800001,
    0x00000001,
    0x80000001,
    0x007FFFFF,
    0x00800000,
    0x80800000,
    0x3F800000,
    0xBF800000,
    0x3FC00000,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x3F000000,
    0x40400000,
    0x3F800001,
    0x3F808000,
    0x3F818000,
    0x3F80C000,
    0x7F7F8000,
    0x477FF000,
    0x477FE000,
    0x33800000,
    0x33000000,
    0x33000001,
    0x1F800000,
    0x5F800000,
    0x00400000,
]
assert len(S) == 32
rng = random.Random(7)
words = []
for t in range(256):
    a0, b0, c0 = S[t % 32], S[(t // 8) % 32], S[(t * 7 + t // 32) % 32]
    k = t % 8
    if k == 0:  # product underflows to a subnormal (or zero), c = 0
        a1 = f(rng.uniform(1, 2) * 2.0 ** -rng.randrange(60, 80))
        b1 = f(rng.uniform(1, 2) * 2.0 ** -rng.randrange(60, 80))
        c1 = rng.choice([0, 0x80000000, 1, 0x80000001])
    elif k == 1:  # overflow
        a1 = f(rng.uniform(1, 2) * 2.0 ** rng.randrange(60, 70))
        b1 = f(-rng.uniform(1, 2) * 2.0 ** rng.randrange(60, 70))
        c1 = f(rng.uniform(-1, 1))
    elif k == 2:  # c = -round(a * b): the fused result is the exact residual
        a1 = f(rng.uniform(-4, 4))
        b1 = f(rng.uniform(-4, 4))
        c1 = f(-struct.unpack("<f", struct.pack("<f", fl(a1) * fl(b1)))[0])
    elif k == 3:  # subnormal operands and sums
        a1, b1, c1 = (
            rng.getrandbits(23),
            f(rng.uniform(-2, 2)),
            rng.getrandbits(23) | 0x80000000,
        )
    else:
        a1, b1, c1 = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)
    if t < 8:
        n, d = [
            (0x80000000, 0xFFFFFFFF),
            (5, 3),
            (0xFFFFFFFB, 3),
            (0, 0xFFFFFFFF),
            (0x80000000, 1),
            (0x7FFFFFFF, 0xFFFFFFFF),
            (0xFFFFFFF9, 2),
            (7, 0xFFFFFFFE),
        ][t]
    else:
        n, d = rng.getrandbits(32), (rng.getrandbits(32) >> rng.randrange(0, 32)) or 1
    words += [a0, a1, b0, b1, c0, c1, n, d]
sys.stdout.buffer.write(struct.pack("<2048I", *words))

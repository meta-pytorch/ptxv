# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# in.bin: 512 words of 16-bit halves: random bits (1/4), specials (bf16 and f16 zeros, infs,
# NaNs, 1.6875), copies of the other operand's half (1/4, so eq / ne hold), small integers
# as bf16 / f16 (rest), seed 15.
import random
import struct

random.seed(15)
S = [
    0x0000,
    0x8000,
    0x7F80,
    0xFF80,
    0x7FC0,
    0x7C00,
    0xFC00,
    0x7E00,
    0x3FD8,
    0x3C00,
    0x3F80,
]


def h():
    r = random.random()
    if r < 0.25:
        return random.getrandbits(16)
    if r < 0.5:
        return random.choice(S)
    return random.choice(
        [0x3F80, 0x4000, 0x4040, 0xBF80, 0x3C00, 0x4000, 0x4200, 0xBC00]
    )


w = []
for _ in range(256):
    a = h() | h() << 16
    b = h() | h() << 16
    if random.random() < 0.25:
        b = (b & 0xFFFF0000) | (a & 0xFFFF)
    w += [a, b]
open("in.bin", "wb").write(struct.pack("<512I", *w))

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/bf16_ext: 4096 (a, b, c, e) words (in.bin). a, b: packed bf16x2, every
pair of the bf16 edge set E in their lanes, then random patterns; c, e: f32 words from an edge
set (signed zeros, subnormals, range ends, infinities, NaN payloads) and random patterns, some
chosen so a bf16 product plus c lands near an f32 rounding boundary."""

import random

import numpy as np

random.seed(23)
N = 4096
E = [
    0x0000,
    0x8000,
    0x0001,
    0x8001,
    0x007F,
    0x0080,
    0x8080,
    0x3F80,
    0xBF80,
    0x3F81,
    0xBF81,
    0x4000,
    0xC000,
    0x3B80,
    0x7F7F,
    0xFF7F,
    0x7F80,
    0xFF80,
    0x7FC0,
    0xFFC0,
    0x7F81,
    0xFF81,
    0x7FFF,
    0x1F80,
    0x5F80,
    0x3FC0,
    0xC0A0,
    0x3EAB,
    0x0040,
    0x4B00,
    0xBEAB,
    0x0100,
]
F = [
    0x00000000,
    0x80000000,
    0x00000001,
    0x80000001,
    0x007FFFFF,
    0x00800000,
    0x80800000,
    0x3F800000,
    0xBF800000,
    0x3F800001,
    0x33800000,
    0xB3800000,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC00000,
    0x7F800001,
    0xFF812345,
    0x3F000000,
    0x4B000000,
    0xCB000001,
    0x3EAAAAAB,
]
pairs = [(x, y) for x in E for y in E]
while len(pairs) < 2 * N:
    pairs.append((random.getrandbits(16), random.getrandbits(16)))
a = np.zeros(N, np.uint32)
b = np.zeros(N, np.uint32)
c = np.zeros(N, np.uint32)
e = np.zeros(N, np.uint32)
for t in range(N):
    (a0, b0), (a1, b1) = pairs[2 * t], pairs[(2 * t + 1 + 517) % (2 * N)]
    a[t] = a1 << 16 | a0
    b[t] = b1 << 16 | b0
    k = t % 3
    c[t] = F[t % len(F)] if k == 0 else random.getrandbits(32)
    if k == 2:
        # c near -(a.lo * b.lo) or with the product's exponent: cancellation and ties
        p = np.float32(np.uint32(a0 << 16).view(np.float32)) * np.float32(
            np.uint32(b0 << 16).view(np.float32)
        )
        if np.isfinite(p) and p != 0:
            c[t] = (np.float32(-p).view(np.uint32) ^ random.getrandbits(3)) & 0xFFFFFFFF
    e[t] = F[(t // 3) % len(F)] if t % 5 == 0 else random.getrandbits(32)
np.stack([a, b, c, e], 1).tofile("in.bin")

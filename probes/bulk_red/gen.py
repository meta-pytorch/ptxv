# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/bulk_red:
- a.bin (4096 f32, the shared-memory addends of CTA 0) and out.bin's first 4096 words (b, the
  global words they are added to): every pair of 32 f32 edge patterns (signed zeros,
  subnormals, normals at the range ends, one-ulp neighbours, infinities, NaN payloads), then
  rounding ties of the sum and random pairs.
- c.bin (4 x 1024 small integers as f32) and out.bin's words 4096..5120 (d, integers): the
  four concurrent bulk adds of CTAs 1..4 (exact, order-independent sums)."""

import random
import struct

import numpy as np

random.seed(9)
E = [
    0x00000000,
    0x80000000,
    0x00000001,
    0x80000001,
    0x007FFFFF,
    0x807FFFFF,
    0x00800000,
    0x80800000,
    0x3F800000,
    0xBF800000,
    0x33800000,
    0xB3800000,
    0x3F800001,
    0xBF7FFFFF,
    0x4B000000,
    0xCB000001,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC00000,
    0x7F800001,
    0xFFA5A5A5,
    0x7FFFFFFF,
    0x00400000,
    0x80400000,
    0x73800000,
    0x7F000000,
    0x34000000,
    0x00000003,
    0x80000002,
]
a = [x for x in E for _ in E]
b = [y for _ in E for y in E]


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


while len(a) < 3072:  # ties: b = 1.xxx * 2^e, a = half an ulp of b (+- a little)
    e = random.randrange(-100, 100)
    m = 1 + random.getrandbits(23) / 2**23
    bb = f2u(m * 2.0**e)
    s = random.choice([1, -1])
    aa = f2u(s * 2.0 ** (e - 24) * random.choice([1, 1, 3, 1.5, 0.75]))
    a.append(aa)
    b.append(bb)
while len(a) < 4096:
    a.append(random.getrandbits(32))
    b.append(random.getrandbits(32))
np.array(a, np.uint32).tofile("a.bin")
c = np.array([random.randrange(-1000, 1000) for _ in range(4096)], np.float32)
c.tofile("c.bin")
d = np.array([random.randrange(-100000, 100000) for _ in range(1024)], np.float32)
np.concatenate([np.array(b, np.uint32), d.view(np.uint32)]).tofile("out.bin")

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/f2i_neg (in.bin, 2048 f32 patterns): signed zeros, subnormals, halves and
ties (+-0.5 .. +-2.5), the values around the s32 / u32 range ends in both signs (2^31 and 2^32
with their neighbours), the f32 range ends, infinities, quiet and signalling NaNs of both signs
with payloads, then random values in (-2^33, 2^33), random values in (-4, 4), random subnormals
and random patterns."""

import random
import struct

import numpy as np

random.seed(41)


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


E = [
    0.0,
    0.5,
    1.5,
    2.5,
    3.5,
    0.49999997,
    0.50000006,
    0.75,
    1.0,
    1.0000001,
    0.99999994,
    1.0e-30,
    100.5,
    101.5,
    8388607.5,
    8388608.0,
    16777216.0,
    2147483520.0,
    2147483648.0,
    2147483904.0,
    4294967040.0,
    4294967296.0,
    4294967808.0,
    1e10,
    1e20,
    3.4028235e38,
]
v = []
for x in E:
    v += [f2u(x), f2u(x) ^ 0x80000000]
v += [
    0x00000001,
    0x80000001,
    0x007FFFFF,
    0x807FFFFF,
    0x00800000,
    0x80800000,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC00000,
    0x7F800001,
    0xFF800001,
    0x7FFFFFFF,
    0xFFFFFFFF,
    0x7FA5A5A5,
    0xFFA5A5A5,
]
while len(v) < 768:
    v.append(f2u(random.uniform(-(2.0**33), 2.0**33)))
while len(v) < 1280:
    v.append(f2u(random.uniform(-4, 4)))
while len(v) < 1408:
    v.append(random.getrandbits(23) | random.getrandbits(1) << 31)
while len(v) < 2048:
    v.append(random.getrandbits(32))
np.array(v, np.uint32).tofile("in.bin")

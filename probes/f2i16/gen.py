# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/f2i16 (in.bin, 2048 f32 patterns): signed zeros, subnormals, halves
(+-0.5 .. +-2.5), values around the s16 / u16 ends (+-32767.5, 32768, -32768.5, -32769,
65535.5, 65536), the f32 range ends, infinities, NaNs, then random values in (-70000, 70000),
random values in (-4, 4) and random patterns."""

import random
import struct

import numpy as np

random.seed(13)


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


E = [
    0.0,
    -0.0,
    0.5,
    -0.5,
    1.5,
    -1.5,
    2.5,
    -2.5,
    0.49999997,
    -0.49999997,
    1.0,
    -1.0,
    32767.0,
    32767.5,
    32767.75,
    32768.0,
    32768.5,
    -32768.0,
    -32768.5,
    -32768.75,
    -32769.0,
    65535.0,
    65535.5,
    65535.75,
    65536.0,
    65536.5,
    -0.75,
    -1.0e-30,
    1.0e-30,
    1e10,
    -1e10,
    3.4028235e38,
    -3.4028235e38,
    100.5,
    -100.5,
    101.5,
]
v = [f2u(x) for x in E] + [
    0x00000001,
    0x80000001,
    0x007FFFFF,
    0x807FFFFF,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC00000,
    0x7F800001,
    0xFFA5A5A5,
]
while len(v) < 1024:
    v.append(f2u(random.uniform(-70000, 70000)))
while len(v) < 1536:
    v.append(f2u(random.uniform(-4, 4)))
while len(v) < 2048:
    v.append(random.getrandbits(32))
np.array(v, np.uint32).tofile("in.bin")

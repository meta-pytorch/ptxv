# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inputs for probes/fcvt (8192 of each):
- in16.bin (u16): 64 edge patterns, then every 8th 16-bit pattern with a rotating low-bit
  offset (every exponent and sign of f16 and bf16, both NaN kinds, subnormals).
- in32.bin (u32): f32 edge values (signed zeros, subnormals, 0.5 / 1.5 / 2.5 ties, values
  around 2^23 and 2^24, the range ends, infinities, NaN payloads), then random values with
  exponents in [-3, 26) (where rounding to an integer matters), then random patterns. The high
  half doubles as an f16 for the `.H1` forms.
- in64.bin (u64): f64 edges as above (2^52 / 2^53), values exactly halfway between adjacent
  bf16 / f16 values and one f64 ulp off them, the f16 / bf16 overflow and subnormal
  thresholds, NaN payloads, random values with exponents in the f16 and bf16 ranges, random
  patterns."""

import random
import struct

import numpy as np

random.seed(7)
N = 8192


def d2u(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


E16 = [
    0x0000,
    0x8000,
    0x0001,
    0x8001,
    0x03FF,
    0x0400,
    0x3C00,
    0xBC00,
    0x3800,
    0x3E00,
    0x4100,
    0x7BFF,
    0xFBFF,
    0x7C00,
    0xFC00,
    0x7E00,
    0xFE00,
    0x7C01,
    0x7D55,
    0x7FFF,
    0x0080,
    0x8080,
    0x007F,
    0x3F80,
    0xBF80,
    0x3F00,
    0x3FC0,
    0x4020,
    0x7F7F,
    0xFF7F,
    0x7F80,
    0xFF80,
    0x7FC0,
    0xFFC0,
    0x7F81,
    0x7FA5,
    0x4B00,
    0x4AFF,
    0x6400,
    0x63FF,
    0x5800,
    0x57FF,
    0x6BFF,
    0x0200,
    0x0201,
    0x0003,
    0x3BFF,
    0x3C01,
    0x4500,
    0x4580,
    0xC500,
    0xC580,
    0x3A00,
    0xBA00,
    0x3F40,
    0xBF40,
    0x47FF,
    0x4800,
    0x4C00,
    0x4BFF,
    0x7C7F,
    0x0100,
    0x8100,
    0x3B80,
]
assert len(E16) == 64
in16 = list(E16) + [(8 * k + k % 8) & 0xFFFF for k in range(N - 64)]

E32 = [
    0.0,
    -0.0,
    0.5,
    -0.5,
    1.5,
    -1.5,
    2.5,
    -2.5,
    0.49999997,
    0.50000006,
    1.0,
    -1.0,
    2.0**23,
    2.0**23 + 1,
    2.0**23 - 0.5,
    -(2.0**23 - 0.5),
    2.0**22 + 0.5,
    2.0**24,
    1e30,
    -1e30,
    3.4028235e38,
    -3.4028235e38,
    0.7,
    -0.7,
    1.2e-38,
    1e-45,
    -1e-45,
    3.5,
    -3.5,
    100.25,
    -100.75,
    8388607.5,
]
in32 = [f2u(x) for x in E32] + [
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC00000,
    0x7F800001,
    0x7FA5A5A5,
    0x00000001,
    0x807FFFFF,
    0x00400000,
    0x80400000,
]
while len(in32) < N // 2:
    e = random.randrange(-3, 26)
    m = random.getrandbits(23)
    s = random.getrandbits(1)
    in32.append(s << 31 | (e + 127) << 23 | m)
while len(in32) < N:
    in32.append(random.getrandbits(32))


def ties(p, emin, emax):
    """Halfway points between adjacent values of a format with p significand bits."""
    out = []
    for _ in range(300):
        e = random.randrange(emin - p, emax + 1)
        m = random.getrandbits(p - 1) | 1 << (p - 1)
        if e < emin:  # subnormal grid
            q = emin - (p - 1)
            m = random.getrandbits(p - 1)
        else:
            q = e - (p - 1)
        x = (2 * m + 1) * 2.0 ** (q - 1)
        s = -1 if random.getrandbits(1) else 1
        out += [s * x, np.nextafter(s * x, np.inf), np.nextafter(s * x, -np.inf)]
    return out


E64 = [
    0.0,
    -0.0,
    0.5,
    -0.5,
    1.5,
    -1.5,
    2.5,
    -2.5,
    2.0**52,
    2.0**52 + 1,
    2.0**52 - 0.5,
    2.0**53,
    -(2.0**52 - 0.5),
    4503599627370495.5,
    1e300,
    -1e300,
    0.7,
    -0.7,
    5e-324,
    -5e-324,
    2.2250738585072014e-308,
    65504.0,
    65519.99,
    65520.0,
    -65520.0,
    65536.0,
    2.0**-24,
    2.0**-25,
    2.0**-25 + 2.0**-60,
    2.0**-26,
    2.0**-14,
    3.3895313892515355e38,
    3.3961775292304e38,
    3.39e38,
    2.0**128,
    2.0**-133,
    2.0**-134,
    2.0**-135,
    2.0**-149,
    1.0e-40,
    1e-50,
    3.4e38,
]
in64 = [d2u(x) for x in E64] + [
    0x7FF0000000000000,
    0xFFF0000000000000,
    0x7FF8000000000000,
    0xFFF8000000000000,
    0x7FF0000000000001,
    0x7FF4000000000000,
    0x7FF0000020000000,
    0x7FFFFFFFFFFFFFFF,
    0x7FF0001000000000,
    0x0000000000000001,
    0x000FFFFFFFFFFFFF,
]
in64 += [d2u(float(x)) for x in ties(11, -14, 15) + ties(8, -126, 127)]
while len(in64) < 3 * N // 4:
    e = random.choice(
        [
            random.randrange(-30, 18),
            random.randrange(-140, 130),
            random.randrange(-3, 55),
        ]
    )
    s = random.getrandbits(1)
    in64.append(s << 63 | (e + 1023) << 52 | random.getrandbits(52))
while len(in64) < N:
    in64.append(random.getrandbits(64))
assert len(in16) == len(in32) == len(in64) == N
np.array(in16, np.uint16).tofile("in16.bin")
np.array(in32, np.uint32).tofile("in32.bin")
np.array(in64, np.uint64).tofile("in64.bin")

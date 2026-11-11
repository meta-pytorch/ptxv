# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/div_const input: 256 f32 words (a = in[t], b = in[128 + t]): specials (signed zeros,
# infinities, NaNs, subnormals, FLT_MAX) and random values over a wide exponent range.
import random
import struct
import sys

r = random.Random(41)
SPECIAL = [
    0x00000000,
    0x80000000,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0x7FA00001,
    0x00000001,
    0x807FFFFF,
    0x00800000,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x3F800000,
    0x3EAAAAAB,
    0x40400000,
]
xs = SPECIAL + [
    (r.getrandbits(1) << 31) | (r.randint(1, 254) << 23) | r.getrandbits(23)
    for _ in range(256 - len(SPECIAL))
]
r.shuffle(xs)
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<256I", *xs)
)

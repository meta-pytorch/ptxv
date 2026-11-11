# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/ftz_div_sqrt input: 256 (a, b) f32 pairs (deterministic): specials (signed zeros,
# infinities, NaNs, subnormals, smallest normals), quotients landing just above / below
# 2^-126 (flushed or not), and random words.
import random
import struct
import sys

r = random.Random(19)


def bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


sp = [
    0,
    0x80000000,
    1,
    0x80000001,
    0x007FFFFF,
    0x807FFFFF,
    0x00800000,
    0x80800000,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0x7F800001,
    bits(1.0),
    bits(-1.0),
    bits(3.0),
    0x7F7FFFFF,
    bits(2.0**-127),
    bits(1e-38),
]
ws = []
for a in sp:
    for b in sp[:7]:
        ws += [a, b]
while len(ws) < 384:
    m = bits(1.0 + r.random())
    e = r.choice([-1, 0, 1])
    ws += [bits(2.0**-126 * (1 + r.random())), bits((1 + r.random()) * 2.0**e)]
while len(ws) < 512:
    ws += [r.getrandbits(32), r.getrandbits(32)]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<512I", *ws[:512])
)

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/copysign input: 256 x (a, b) f32 words (deterministic): random words and specials.
import random
import struct
import sys

r = random.Random(13)
sp = [
    0,
    0x80000000,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC00001,
    0x7F800001,
    0x00000001,
    0x807FFFFF,
    0x3F800000,
]
out = []
for t in range(256):
    if t % 2:
        out += [r.choice(sp), r.choice(sp)]
    else:
        out += [r.getrandbits(32), r.getrandbits(32)]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<512I", *out)
)

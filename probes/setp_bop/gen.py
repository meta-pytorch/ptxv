# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/setp_bop input: 256 x (a, b) f32 words (deterministic): equal pairs, NaNs, signed
# zeros, subnormals and random words.
import random
import struct
import sys

r = random.Random(3)
sp = [
    0x7FC00000,
    0xFFC00001,
    0x00000000,
    0x80000000,
    0x00000001,
    0x80000001,
    0x3F800000,
    0xBF800000,
    0x7F800000,
    0xFF800000,
]
out = []
for t in range(256):
    k = (t >> 1) % 4
    if k == 0:
        a = r.choice(sp)
        b = r.choice(sp)
    elif k == 1:
        a = r.getrandbits(32)
        b = a
    elif k == 2:
        a = r.choice(sp)
        b = r.getrandbits(32)
    else:
        a = r.getrandbits(32)
        b = r.getrandbits(32)
    out += [a, b]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<512I", *out)
)

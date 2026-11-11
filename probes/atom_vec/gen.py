# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/atom_vec inputs: in.bin = 384 f32 addends, acc.bin = 384 f32 starting values (64 threads
# x (4 + 2)); specials (signed zeros, infinities, NaNs with payloads, subnormals, FLT_MAX) and
# random values over a wide exponent range.
import random
import struct
import sys

r = random.Random(23)
SPECIAL = [
    0x00000000,
    0x80000000,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0x7FA12345,
    0xFFC00001,
    0x00000001,
    0x807FFFFF,
    0x00400000,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x3F800000,
    0x00800000,
    0x80800000,
    0x33800000,
]


def word():
    k = r.random()
    if k < 0.3:
        return r.choice(SPECIAL)
    e = r.randint(1, 254) if k < 0.6 else r.randint(110, 145)
    return (r.getrandbits(1) << 31) | (e << 23) | r.getrandbits(23)


d = sys.argv[1] if len(sys.argv) > 1 else "."
for name in ("in.bin", "acc.bin"):
    open(f"{d}/{name}", "wb").write(struct.pack("<384I", *[word() for _ in range(384)]))

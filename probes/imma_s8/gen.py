# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/imma_s8 input: 32 threads x 32 words (deterministic). C: random s32, a quarter of
# the lanes within 2^19 of INT32_MAX / INT32_MIN; step 0 random bytes, step 1 A all -128
# and B mostly 127, step 2 all -128 x -128 on A and B (+524288 per output), step 3 bytes
# from {-128, 127} (mixed signs after saturation).
import random
import struct
import sys

r = random.Random(41)


def w(bs):
    return sum((b & 0xFF) << (8 * i) for i, b in enumerate(bs))


ws = []
for t in range(32):
    c = []
    for j in range(4):
        k = r.randrange(4)
        if k == 0:
            v = 0x7FFFFFFF - r.randrange(1 << 19)
        elif k == 1:
            v = -0x80000000 + r.randrange(1 << 19)
        else:
            v = r.randrange(-(1 << 31), 1 << 31)
        c.append(v & 0xFFFFFFFF)
    row = c
    for s in range(4):
        if s == 0:
            pick = lambda: r.randrange(-128, 128)
        elif s == 1:
            pick = lambda: r.choice([-128, 127, -127, 0])
        elif s == 2:
            pick = lambda: -128
        else:
            pick = lambda: r.choice([-128, 127])
        a = [w([pick() for _ in range(4)]) for _ in range(4)]
        b = [w([pick() for _ in range(4)]) for _ in range(2)]
        if s == 1:
            # A all -128, B mostly 127: about -500000 per output (INT32_MIN side)
            a = [w([-128] * 4)] * 4
            b = [w([r.choice([127] * 7 + [-128]) for _ in range(4)]) for _ in range(2)]
        row += a + b
    row += [0] * (32 - len(row))
    ws += row
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<1024I", *ws)
)

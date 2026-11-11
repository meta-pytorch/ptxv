# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/cvt_tf32_rn input: 65,536 f32 words, every sign and exponent (NaN / Inf / subnormal /
# zero included) with 128 mantissas each: the low 13 bits at the rounding edges (0, 1, 0x0fff,
# 0x1000, 0x1001, 0x1fff, random) under the extreme and random upper 10 bits.
import random
import struct
import sys

r = random.Random(31)
LOW = [0x0000, 0x0001, 0x0FFF, 0x1000, 0x1001, 0x1FFF]
HIGH = [0x000, 0x3FF, 0x001, 0x3FE]
xs = []
for s in (0, 1):
    for e in range(256):
        ms = [(h << 13) | l for h in HIGH for l in LOW]
        while len(ms) < 128:
            h = r.getrandbits(10)
            ms.append(
                (h << 13) | (r.choice(LOW) if r.random() < 0.5 else r.getrandbits(13))
            )
        xs += [(s << 31) | (e << 23) | m for m in ms]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<65536I", *xs)
)

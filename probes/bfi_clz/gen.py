# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/bfi_clz input: 64 x (a, b, pos, len) words (deterministic). pos / len from edge
# values (0, 1, 31, 32, 33, 255, 256, 257, 0x100001f) or random; a includes 0, 1, 0x80000000,
# all ones and random words with varying leading zeros.
import random
import struct
import sys

r = random.Random(13)
edge = [0, 1, 4, 16, 31, 32, 33, 255, 256, 257, 0x100001F, 0xFFFFFFFF]
ws = []
for t in range(64):
    a = (
        [0, 1, 0x80000000, 0xFFFFFFFF][t]
        if t < 4
        else r.getrandbits(32) >> r.randrange(32)
    )
    b = r.getrandbits(32)
    pos = r.choice(edge) if t % 2 else r.randrange(40)
    ln = r.choice(edge) if t % 3 else r.randrange(40)
    ws += [a, b, pos, ln]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<256I", *ws)
)

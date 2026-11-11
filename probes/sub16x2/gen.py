# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/sub16x2 input: 6144 x (a, b) 32-bit words (deterministic). 4096 pairs: every combination
# of 8 edge values (0, 1, 2, 0x7ffe, 0x7fff, 0x8000, 0x8001, 0xffff) in the four 16-bit lanes of
# a and b (each lane's borrow and signed / unsigned wrap, and every low-lane borrow under every
# high-lane pair); 1024 random words; 512 pairs whose low lanes borrow (252 of them under equal
# high lanes, where a 32-bit sub turns the high lane into 0xffff); 512 whose low lanes do not.
import random
import struct
import sys

r = random.Random(29)
L = [0x0000, 0x0001, 0x0002, 0x7FFE, 0x7FFF, 0x8000, 0x8001, 0xFFFF]
out = []
for ah in L:
    for al in L:
        for bh in L:
            for bl in L:
                out += [ah << 16 | al, bh << 16 | bl]
for _ in range(1024):
    out += [r.getrandbits(32), r.getrandbits(32)]
for k in range(1024):
    h = r.getrandbits(16)
    x, y = r.getrandbits(16), r.getrandbits(16)
    lo_a, lo_b = (
        (min(x, y), max(x, y) + (x == y)) if k < 512 else (max(x, y), min(x, y))
    )
    out += [
        h << 16 | (lo_a & 0xFFFF),
        r.choice([h, r.getrandbits(16)]) << 16 | (lo_b & 0xFFFF),
    ]
assert len(out) == 12288
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<12288I", *out)
)

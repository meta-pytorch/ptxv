# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""probes/cvt_f16 input: 32,768 f32 words (deterministic) at the f16 and bf16 rounding edges.
Every sign and f32 exponent with 64 mantissas each: the bits below the f16 rounding position
(13 for f16 normals, more for f16 subnormals) and below the bf16 position (16) at their edges
(0, 1, half - 1, half, half + 1, all ones), the kept bits' lowest bit both ways (ties to even),
under extreme and random upper bits; NaN payloads and the overflow thresholds (65504 / 65520,
bf16 0x7f7f / 0x7f7f8000) come with the exponents 142 and 254 / 255."""

import random
import struct
import sys

r = random.Random(37)


def edges(drop):
    """Low-bit patterns around the rounding position `drop` (bits dropped)."""
    if drop >= 24:
        return [0, 1, 0x7FFFFF, 0x400000, 0x3FFFFF, 0x400001]
    half = 1 << (drop - 1)
    low = [0, 1, half - 1, half, half + 1, 2 * half - 1]
    out = []
    for kept_lsb in (0, 1):
        out += [(kept_lsb << drop) | x for x in low]
    return [x & 0x7FFFFF for x in out]


xs = []
for s in (0, 1):
    for e in range(256):
        # f16 rounding position: 13 bits for f16 normals (f32 exponent >= 113), more below.
        d16 = 13 if e >= 113 else 13 + (113 - e)
        ms = []
        for drop in (d16, 16):
            for low in edges(drop):
                for hi in (0, 0x7FFFFF):
                    keep_mask = ~((1 << min(drop + 1, 23)) - 1) & 0x7FFFFF
                    ms.append((hi & keep_mask) | low)
        ms = list(dict.fromkeys(ms))
        while len(ms) < 64:
            drop = r.choice((d16, 16))
            hi = r.getrandbits(23) & ~((1 << min(drop + 1, 23)) - 1)
            low = r.choice(edges(drop)) if r.random() < 0.6 else r.getrandbits(23)
            ms.append((hi | low) & 0x7FFFFF)
        xs += [(s << 31) | (e << 23) | m for m in ms[:64]]
assert len(xs) == 32768
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<32768I", *xs)
)

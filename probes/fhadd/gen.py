"""Inputs of probes/fhadd: (a, c) per thread. Full: every f16 pattern in a.lo (a.hi a bijection
of it) against each f32 edge value of C. small: every pair of 32 f16 edges x the 24 C values,
then random words, 2048 threads. usage: gen.py OUT [small]"""

import random
import struct
import sys

H = [
    0x0000,
    0x8000,
    0x0001,
    0x8001,
    0x03FF,
    0x83FF,
    0x0400,
    0x8400,
    0x3C00,
    0xBC00,
    0x3C01,
    0x3BFF,
    0x7BFF,
    0xFBFF,
    0x7C00,
    0xFC00,
    0x7E00,
    0xFE00,
    0x7C01,
    0x7FFF,
    0x5640,
    0xD640,
    0x0200,
    0x1234,
    0x9234,
    0x6543,
    0xE543,
    0x3555,
    0x2E66,
    0xAE66,
    0x7A00,
    0x4000,
]
C = [
    0x00000000,
    0x80000000,
    0x3F800000,
    0xBF800000,
    0x3F800001,
    0x33800000,
    0xB3800000,
    0x337FFFFF,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC00001,
    0x7F800001,
    0x00000001,
    0x80000001,
    0x007FFFFF,
    0x00800000,
    0x477FE000,
    0xC77FE000,
    0x38800000,
    0x3EAAAAAB,
    0x4B800001,
]
out = sys.argv[1]
rows = []
if len(sys.argv) > 2:
    pairs = [(h0, c) for h0 in H for c in C][:768]
    for i, (h0, c) in enumerate(pairs):
        rows.append((h0 | H[(i * 7) % len(H)] << 16, c))
    random.seed(5)
    while len(rows) < 2048:
        rows.append((random.getrandbits(32), random.getrandbits(32)))
else:
    for c in C:
        for p in range(1 << 16):
            rows.append((p | ((p * 40503) & 0xFFFF) << 16, c))
open(out, "wb").write(b"".join(struct.pack("<II", a, c) for a, c in rows))
print(out, len(rows))

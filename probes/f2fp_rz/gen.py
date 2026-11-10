"""Inputs of probes/f2fp_rz: (x, y) f32 pairs, every pair of the edge patterns E, then random
words; 8192 threads. usage: gen.py OUT"""

import random
import struct
import sys

E = [
    0x00000000,
    0x80000000,
    0x00000001,
    0x80000001,
    0x007FFFFF,
    0x00800000,
    0x33800000,
    0x337FFFFF,
    0x33000000,
    0x33C00000,
    0x387FC000,
    0x387FE000,
    0x38800000,
    0x387FFFFF,
    0x3F800000,
    0xBF800000,
    0x3F801FFF,
    0x3F802000,
    0x477FE000,
    0x477FEFFF,
    0x477FF000,
    0x477FFFFF,
    0x47800000,
    0xC7800000,
    0x4F000000,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC00000,
    0x7F800001,
    0xFFBFFFFF,
    0x3EAAAAAB,
    0xB5000001,
    0x38000000,
]
rows = [(x, y) for x in E for y in E]
random.seed(11)
while len(rows) < 8192:
    rows.append((random.getrandbits(32), random.getrandbits(32)))
open(sys.argv[1], "wb").write(b"".join(struct.pack("<II", *r) for r in rows))
print(len(rows))

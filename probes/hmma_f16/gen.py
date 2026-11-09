# probes/hmma_f16 inputs: A, B [128 x 64] f16 from N(0, 1) (deterministic), magnitudes below
# 2^-14 (f16 subnormals, refused by the model) redrawn.
import os
import random
import struct
import sys

r = random.Random(29)


def f16(v):
    return struct.unpack("<H", struct.pack("<e", v))[0]


def draw():
    while True:
        h = f16(r.gauss(0, 1))
        if h & 0x7C00:
            return h


d = sys.argv[1] if len(sys.argv) > 1 else "data"
for name in ("A", "B"):
    with open(os.path.join(d, name + ".bin"), "wb") as f:
        f.write(struct.pack("<8192H", *[draw() for _ in range(8192)]))

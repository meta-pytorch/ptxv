# probes/cvt_tf32 input: 1024 f32 words (deterministic): random, ties, near overflow, NaN,
# Inf, subnormal.
import random
import struct
import sys

r = random.Random(19)
xs = []
for i in range(1024):
    k = i % 8
    if k == 0:
        xs.append(r.getrandbits(32))
    elif k == 1:
        xs.append((r.getrandbits(32) & ~0x1FFF) | 0x1000)
    elif k == 2:
        xs.append(0x7F7FF000 | r.getrandbits(12) | (r.getrandbits(1) << 31))
    elif k == 3:
        xs.append(
            0x7F800000
            | r.choice([1, 0x1000, 0xFFF, 0x2000, 0x400000])
            | (r.getrandbits(1) << 31)
        )
    elif k == 4:
        xs.append(r.choice([0x7F800000, 0xFF800000, 0, 0x80000000]))
    elif k == 5:
        xs.append(r.getrandbits(23) | (r.getrandbits(1) << 31))
    else:
        xs.append(r.getrandbits(32))
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<1024I", *xs)
)

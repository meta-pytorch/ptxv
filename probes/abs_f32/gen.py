# in.bin: 1024 words: special f32 / s32 patterns, then random words. Seed 3.
import random
import struct

random.seed(3)
sp = [
    0,
    0x80000000,
    1,
    0x80000001,
    0x007FFFFF,
    0x807FFFFF,
    0x00800000,
    0x3F800000,
    0xBF800000,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC00000,
    0x7F800001,
    0xFF800001,
    0x7FFFFFFF,
    0xFFFFFFFF,
    0x7FA12345,
    0xFFA12345,
    0x80000000,
    0x7FFFFFFF,
]
w = sp + [random.getrandbits(32) for _ in range(1024 - len(sp))]
open("in.bin", "wb").write(struct.pack("<1024I", *w))

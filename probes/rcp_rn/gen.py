# in.bin: 4096 f32 words: specials (+-0, +-inf, NaNs, smallest / largest subnormal and
# normal, 2^126, 2^127, max finite) first, then random bits (1/2), random subnormals (1/8),
# magnitudes near 2^+-126..127 (1/8), random normals (rest), seed 7.
import random
import struct

random.seed(7)
SPEC = [
    0x00000000,
    0x80000000,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC00001,
    0x7F800001,
    0x00000001,
    0x80000001,
    0x007FFFFF,
    0x807FFFFF,
    0x00800000,
    0x80800000,
    0x7E800000,
    0x7F000000,
    0xFF000000,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x3F800000,
    0xBF800000,
    0x00400000,
    0x00200000,
    0x7E7FFFFF,
    0x00C00000,
]
w = list(SPEC)
while len(w) < 4096:
    r = random.random()
    s = random.getrandbits(1) << 31
    if r < 0.5:
        w.append(random.getrandbits(32))
    elif r < 0.625:
        w.append(s | random.randint(1, 0x7FFFFF))
    elif r < 0.75:
        e = random.choice([0x01, 0x02, 0xFC, 0xFD, 0xFE])
        w.append(s | e << 23 | random.getrandbits(23))
    else:
        w.append(s | random.randint(0x01, 0xFE) << 23 | random.getrandbits(23))
open("in.bin", "wb").write(struct.pack("<4096I", *w))

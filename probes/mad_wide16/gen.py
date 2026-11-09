# in.bin: 512 words of random bits with 16-bit edge values in the even words, seed 14.
import random
import struct

random.seed(14)
w = [random.getrandbits(32) for _ in range(512)]
for i, e in enumerate([0, 1, 0x7FFF, 0x8000, 0xFFFF, 0x8001, 0xFFFE]):
    w[2 * i] = (w[2 * i] & 0xFFFF0000) | e
open("in.bin", "wb").write(struct.pack("<512I", *w))

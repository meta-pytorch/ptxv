# in.bin: 1024 x 2 u32 of random.getrandbits(32) (seed 11) with every 8th word 0 (bfind -> -1).
import random
import struct

random.seed(11)
w = [
    0 if i % 8 == 0 else random.getrandbits(32) >> random.randrange(32)
    for i in range(2048)
]
open("in.bin", "wb").write(struct.pack("<2048I", *w))

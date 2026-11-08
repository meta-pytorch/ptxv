# in.bin: 256 words; bits 0..2 random with P(1) = 1/2, 1/64 and 63/64 (so and / or / popc see
# both outcomes), seed 8.
import random
import struct

random.seed(8)
w = [
    (random.random() < 0.5)
    | (random.random() < 1 / 64) << 1
    | (random.random() < 63 / 64) << 2
    for _ in range(256)
]
open("in.bin", "wb").write(struct.pack("<256I", *w))

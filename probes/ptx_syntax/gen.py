# in.bin: 256 words of random.getrandbits(32), seed 5.
import random
import struct

random.seed(5)
open("in.bin", "wb").write(
    struct.pack("<256I", *[random.getrandbits(32) for _ in range(256)])
)

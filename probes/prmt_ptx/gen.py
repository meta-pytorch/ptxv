# in.bin: 768 words of random.getrandbits(32), seed 4.
import random
import struct

random.seed(4)
open("in.bin", "wb").write(
    struct.pack("<768I", *[random.getrandbits(32) for _ in range(768)])
)

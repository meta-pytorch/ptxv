# probes/mad_lo16 input: 256 x 2 random u32 words (deterministic).
import random
import struct
import sys

r = random.Random(7)
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<512I", *[r.getrandbits(32) for _ in range(512)])
)

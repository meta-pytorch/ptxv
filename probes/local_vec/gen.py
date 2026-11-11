# probes/local_vec input: 128 random words (deterministic).
import random
import struct
import sys

r = random.Random(47)
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<128I", *[r.getrandbits(32) for _ in range(128)])
)

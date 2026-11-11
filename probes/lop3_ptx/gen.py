# probes/lop3_ptx input: 256 x (a, b, c) random u32 words (deterministic).
import random
import struct
import sys

r = random.Random(5)
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<768I", *[r.getrandbits(32) for _ in range(768)])
)

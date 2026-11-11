# probes/mov_pack input: 32 x (x, y) random words with distinct bytes (deterministic).
import random
import struct
import sys

r = random.Random(17)
ws = [r.getrandbits(32) for _ in range(64)]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<64I", *ws)
)

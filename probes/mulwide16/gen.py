# probes/mulwide16 input: 256 x (a, b) 32-bit words (deterministic): random words and 16-bit
# edges (0, +-1, 0x7fff, 0x8000, 0xffff) in either half.
import random
import struct
import sys

r = random.Random(31)
h = [0, 1, 0xFFFF, 0x7FFF, 0x8000, 0x8001, 0xFFFE, 0x00FF, 0xFF00]
out = []
for t in range(256):
    if t % 2:
        out += [r.choice(h) | r.choice(h) << 16, r.choice(h) | r.choice(h) << 16]
    else:
        out += [r.getrandbits(32), r.getrandbits(32)]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<512I", *out)
)

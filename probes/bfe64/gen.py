# probes/bfe64 input: 256 x (a: u64, pos: u32, len: u32) (deterministic): random a (half of
# them with the top bit set), edge (pos, len) pairs around 32 and 64 and above 255 (taken
# mod 256), and random pairs.
import random
import struct
import sys

r = random.Random(37)
edges = [
    (0, 0),
    (0, 1),
    (0, 32),
    (0, 63),
    (0, 64),
    (0, 65),
    (0, 255),
    (1, 63),
    (1, 64),
    (31, 2),
    (31, 33),
    (32, 1),
    (32, 31),
    (32, 32),
    (32, 33),
    (33, 31),
    (60, 8),
    (63, 1),
    (63, 2),
    (64, 1),
    (64, 0),
    (100, 5),
    (255, 1),
    (256, 8),
    (259, 5),
    (3, 260),
    (7, 0),
    (16, 48),
    (48, 16),
    (40, 30),
    (0x12345603, 0xABCDEF09),
    (5, 70),
]
out = []
for t in range(256):
    a = r.getrandbits(64)
    if t % 2:
        a |= 1 << 63
    if t < len(edges) * 2:
        p, n = edges[t % len(edges)]
    elif t % 3 == 0:
        p, n = r.getrandbits(32), r.getrandbits(32)
    else:
        p, n = r.randrange(70), r.randrange(70)
    out.append(struct.pack("<QII", a, p & 0xFFFFFFFF, n & 0xFFFFFFFF))
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(b"".join(out))

# probes/rsqrt_f64 input: 4096 f64 words (deterministic): specials, subnormals, boundary
# neighbours of 2^-1022 and 2^1023, random positives over every exponent, random words.
import random
import struct
import sys

r = random.Random(43)


def bits(v):
    return struct.unpack("<Q", struct.pack("<d", v))[0]


sp = [
    0,
    1 << 63,
    0x7FF0000000000000,
    0xFFF0000000000000,
    0x7FF8000000000000,
    0x7FF0000000000001,
    0xFFF8000000001234,
    0x7FF4000000000000,
    1,
    0x000FFFFFFFFFFFFF,
    0x0008000000000000,
    0x8000000000000001,
    bits(-1.0),
    bits(-1e-300),
    bits(1.0),
    bits(4.0),
    bits(0.25),
    0x7FEFFFFFFFFFFFFF,
]
for b in (0x0010000000000000, 0x7FE0000000000000, 0x3FF0000000000000):
    sp += [b - 2, b - 1, b, b + 1, b + 2]
ws = sp[:]
while len(ws) < 2048:
    e = r.randrange(0, 2047)
    ws.append(e << 52 | r.getrandbits(52))
while len(ws) < 3072:
    ws.append(r.getrandbits(52))  # subnormals
while len(ws) < 4096:
    ws.append(r.getrandbits(64))
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<4096Q", *ws)
)

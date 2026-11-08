# probes/f32x2_sinks input: 256 x 8 f32 words (deterministic): x0 x1 y0 y1 c0 c1 s0 s1; c cancels
# x y to +- 1 ulp in 3 of 4 threads, random words in the 4th.
import random
import struct

r = random.Random(31)


def f(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def g(w):
    return struct.unpack("<f", struct.pack("<I", w))[0]


out = []
for t in range(256):
    if t % 4 == 3:
        out += [r.getrandbits(32) for _ in range(8)]
        continue
    x = [f(r.choice([1, -1]) * (1 + r.random())) for _ in range(2)]
    y = [f(r.choice([1, -1]) * (1 + r.random())) for _ in range(2)]
    c = [f(-(g(x[i]) * g(y[i]))) ^ r.choice([0, 1]) for i in range(2)]
    s = [f(r.choice([1, -1]) * (1 + r.random())) for _ in range(2)]
    out += x + y + c + s
open("in.bin", "wb").write(struct.pack("<2048I", *out))

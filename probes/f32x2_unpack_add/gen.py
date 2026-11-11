# probes/f32x2_unpack_add input (same as probes/f32x2_unpack): 256 x (x.lo, x.hi, y.lo, y.hi) f32 words (deterministic):
# products near -y (cancellation exposes a contraction) and random words.
import random
import struct
import sys

r = random.Random(11)
f = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]
g = lambda w: struct.unpack("<f", struct.pack("<I", w))[0]
out = []
for t in range(256):
    if t % 4 == 3:
        out += [r.getrandbits(32) for _ in range(4)]
        continue
    x = [f(1 + r.random()), f(1 + r.random())]
    y = [f(-(1 + r.random())), f(-(1 + r.random()))]
    out += [x[0], x[1], y[0], y[1]]
    # t % 4 == 1: y replaced by -(x * y) rounded, +- 1 ulp (cancellation in the adds)
    y2 = [
        f(-g(x[0]) * g(y[0])) ^ r.choice([0, 1]),
        f(-g(x[1]) * g(y[1])) ^ r.choice([0, 1]),
    ]
    if t % 4 == 1:
        out[-2:] = y2
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<1024I", *out)
)

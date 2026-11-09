# probes/f32x2_repack input: 256 x 16 f32 words (deterministic): x, y, u, w, c, e, k, pad.
# c, e cancel the product lanes they are added to (+- 1 ulp) in 3 of 4 threads; the 4th is
# random words.
import random
import struct
import sys

r = random.Random(23)
f = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]
g = lambda w: struct.unpack("<f", struct.pack("<I", w))[0]
s = lambda: r.choice([1, -1])
out = []
for t in range(256):
    if t % 4 == 3:
        out += [r.getrandbits(32) for _ in range(16)]
        continue
    x, y, u, w = ([f(s() * (1 + r.random())) for _ in range(2)] for _ in range(4))
    p = [g(x[i]) * g(y[i]) for i in range(2)]
    q = [g(u[i]) * g(w[i]) for i in range(2)]
    neg = lambda v: f(-v) ^ r.choice([0, 1])
    c = [neg(q[0]), neg(p[0])]
    e = [neg(p[0]), f(r.random())]
    k = f(1 + r.random())
    out += x + y + u + w + c + e + [k, 0, 0, 0]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<4096I", *out)
)

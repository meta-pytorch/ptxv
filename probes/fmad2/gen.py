# probes/fmad2 input: 32 threads x 4 f32 pairs x, y, z, w (deterministic): x, y random in
# [1, 2) with random signs, z = -(x y rounded to f32) nudged by -1..1 ulp, so a fused
# x y + z (the rounding error of the product) differs from the unfused sum; w = 1 + k ulp,
# k in 1..8 (z w near -(x y) too).
import random
import struct
import sys

r = random.Random(7)


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


ws = []
for t in range(32):
    xs, ys, zs, vs = [], [], [], []
    for lane in range(2):
        x = f32(r.choice([-1, 1]) * (1 + r.random()))
        y = f32(r.choice([-1, 1]) * (1 + r.random()))
        z = bits(-f32(x * y)) + r.choice([-1, 0, 1])
        xs.append(bits(x))
        ys.append(bits(y))
        zs.append(z)
        vs.append(bits(1.0) + r.randrange(1, 9))
    ws += xs + ys + zs + vs
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<256I", *ws)
)

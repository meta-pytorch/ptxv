# a.bin / b.bin: 1024 f32 pairs. Specials (NaN, inf, 0 signs, inf * 0), products that
# overflow before the scale and not after (D2 / D4 / D8), products below 2^-126 (subnormal
# results: the scale's rounding, ties at the last subnormal bit, FTZ), products near 1 for
# .SAT, then random normals of nearby and distant magnitudes (seed 7).
import random
import struct

random.seed(7)
inf = float("inf")
nan = float("nan")


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(x):
    return struct.unpack("<f", struct.pack("<I", x))[0]


P = [
    (nan, 1.0),
    (1.0, nan),
    (inf, 0.0),
    (-inf, 2.0),
    (inf, -inf),
    (0.0, -0.0),
    (-0.0, -0.0),
    (0.0, 5.0),
    (1.0, 1.0),
    (0.5, 2.0),
    (2.0**127, 2.0),
    (2.0**127, 4.0),
    (2.0**127, 8.0),
    (3.0 * 2.0**126, 3.0),
    (bits(0x7F7FFFFF), bits(0x7F7FFFFF)),
    (bits(0x7F7FFFFF), 1.5),
    (2.0**-126, 1.0),
    (2.0**-126, 0.5),
    (2.0**-126, 0.75),
    (2.0**-149, 1.0),
    (2.0**-149, 0.5),
    (2.0**-149, 1.5),
    (3 * 2.0**-149, 0.5),
    (bits(0x00000001), bits(0x3F800000)),
    (bits(0x00800001), 1.0),
    (bits(0x007FFFFF), 1.0),
    (bits(0x007FFFFF), 2.0),
    (bits(0x007FFFFF), 4.0),
    (0.999999, 1.0),
    (1.0000001, 1.0),
    (-1.0, 0.5),
    (2.0, 1.0),
    (1.9999999, 1.0),
]
a = [f32(p[0]) for p in P]
b = [f32(p[1]) for p in P]
while len(a) < 1024:
    k = random.random()
    if k < 0.3:  # subnormal products (ties included: small odd significands)
        e = random.randint(-152, -124)
        x = random.choice(
            [random.randint(1, 63), random.randint(1 << 22, (1 << 24) - 1)]
        )
        y = random.choice([1, 3, 5, 7, random.randint(1, 1 << 23)])
        ea = random.randint(-80, -40)
        av = x * 2.0 ** (ea - 23)
        bv = y * 2.0 ** (e - ea)
    elif k < 0.45:  # overflow before the scale
        av = random.uniform(1, 2) * 2.0 ** random.randint(60, 127)
        bv = random.uniform(1, 2) * 2.0 ** (
            random.randint(125, 130) - round(__import__("math").log2(av))
        )
    elif k < 0.55:  # near 1 (.SAT)
        av = random.uniform(0.5, 4)
        bv = random.uniform(0.25, 1)
    else:
        e = random.randint(-60, 60)
        av = random.uniform(1, 2) * 2.0**e
        bv = random.uniform(1, 2) * 2.0 ** random.randint(-60, 60)
    a.append(f32(random.choice([-1, 1]) * av))
    b.append(f32(random.choice([-1, 1]) * bv))
open("a.bin", "wb").write(struct.pack("<1024f", *a))
open("b.bin", "wb").write(struct.pack("<1024f", *b))

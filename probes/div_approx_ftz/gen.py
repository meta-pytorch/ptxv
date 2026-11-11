"""gen.py: a.bin / b.bin / c.bin, 1024 f32 triples for kernel.ptx. Specials (NaN, +-inf,
signed zeros), subnormal numerators and divisors (flushed), divisors near 2^126 / 2^127 /
FLT_MAX (reciprocals below 2^-126), quotients that overflow or fall below 2^-126, ties for
the add / sub (c = -a / b and nearby), then random normals of near and far magnitudes
(seed 11)."""

import random
import struct

random.seed(11)
inf, nan = float("inf"), float("nan")


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(u):
    return struct.unpack("<f", struct.pack("<I", u))[0]


SP = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    inf,
    -inf,
    nan,
    2.0**-126,
    -(2.0**-126),
    bits(0x00000001),
    bits(0x807FFFFF),
    bits(0x00400000),
    2.0**126,
    2.0**127,
    -(2.0**127),
    bits(0x7F7FFFFF),
    bits(0x7EFFFFFF),
    3.0,
    0.5,
    1.5 * 2.0**126,
    2.0**-125,
    1.0000001,
    0.99999994,
]
T = []
for a in SP:
    for b in SP:
        T.append((a, b, random.choice(SP)))
random.shuffle(T)
T = T[:400]
while len(T) < 1024:
    k = random.random()
    ea = random.randint(-140, 127)
    eb = random.randint(-130, 127)
    if k < 0.3:  # quotients near the bottom of the normal range
        eb = ea + random.randint(120, 140)
        eb = min(eb, 127)
    elif k < 0.5:  # quotients near overflow
        eb = ea - random.randint(120, 130)
        eb = max(eb, -126)
    elif k < 0.7:
        eb = ea + random.randint(-3, 3)
    a = random.choice([-1, 1]) * random.uniform(1, 2) * 2.0**ea
    b = random.choice([-1, 1]) * random.uniform(1, 2) * 2.0 ** max(min(eb, 127), -149)
    a, b = f32(a), f32(b)
    q = a / b if b != 0 else inf
    r = random.random()
    if r < 0.3 and abs(q) < 3e38:
        c = f32(-q)  # cancellation in the add
    elif r < 0.5 and abs(q) < 3e38:
        c = f32(-q * (1 + random.choice([-1, 1]) * 2.0 ** -random.randint(18, 24)))
    else:
        c = f32(
            random.choice([-1, 1])
            * random.uniform(1, 2)
            * 2.0 ** random.randint(-130, 127)
        )
    T.append((a, b, c))
for name, k in (("a", 0), ("b", 1), ("c", 2)):
    with open(f"{name}.bin", "wb") as f:
        f.write(struct.pack("<1024f", *[t[k] for t in T]))

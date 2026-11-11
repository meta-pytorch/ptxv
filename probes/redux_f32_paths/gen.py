# probes/redux_f32_paths input: 16 warps x 32 lanes of f32 words (deterministic), one mix per
# warp: random words, one NaN among finite values, all NaN, zeros of both signs, +-inf with
# finite values, subnormals, payload NaNs of both signs, equal magnitudes of both signs, ...
import random
import struct
import sys

r = random.Random(13)
NAN = [0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFBFFFFF, 0x7FFFFFFF]


def warp(k):
    if k == 0:
        return [r.getrandbits(32) for _ in range(32)]
    if k == 1:
        v = [
            struct.unpack("<I", struct.pack("<f", r.uniform(-9, 9)))[0]
            for _ in range(32)
        ]
        v[r.randrange(32)] = r.choice(NAN)
        return v
    if k == 2:
        return [r.choice(NAN) for _ in range(32)]
    if k == 3:
        return [r.choice([0, 0x80000000]) for _ in range(32)]
    if k == 4:
        v = [
            struct.unpack("<I", struct.pack("<f", r.uniform(-9, 9)))[0]
            for _ in range(32)
        ]
        v[3] = 0x7F800000
        v[17] = 0xFF800000
        return v
    if k == 5:
        return [r.getrandbits(23) | (r.getrandbits(1) << 31) for _ in range(32)]
    if k == 6:
        return [r.choice(NAN + [0x3F800000, 0xBF800000]) for _ in range(32)]
    if k == 7:
        m = r.getrandbits(31)
        return [m | (r.getrandbits(1) << 31) for _ in range(32)]
    if k == 8:
        return [0x80000000] * 31 + [0]
    if k == 9:
        return [0] * 31 + [0x80000000]
    if k == 10:
        v = [0xFF800000] * 32
        v[5] = 0x7FC00000
        return v
    if k == 11:
        return [0x00000001 | (r.getrandbits(1) << 31) for _ in range(32)]
    return [r.getrandbits(32) if r.random() < 0.8 else r.choice(NAN) for _ in range(32)]


out = []
for w in range(16):
    out += warp(w)
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<512I", *out)
)

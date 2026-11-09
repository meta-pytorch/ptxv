# probes/mma_scale_d inputs (data/): A (128 x 16 bf16), B (256 x 16 bf16), C (128 lanes x 256
# columns f32, TMEM's initial contents) and C_special (hardware-only exploration).
# A rows 0..63 are zero (D = C * 2^-s exactly when the scaled value is a normal f32); rows
# 64..127 random. C: normal f32 values that stay normal after any scale 2^-15 (exponent >= -110),
# rows 0..63 over the whole exponent range, rows 64..127 within 2^-24..2^24 of the products.
import os
import random
import struct
import sys

r = random.Random(103)


def bf16(e_lo, e_hi):
    s = r.getrandbits(1)
    e = r.randint(e_lo, e_hi) + 127
    return s << 15 | e << 7 | r.getrandbits(7)


A = [0] * (64 * 16) + [bf16(-6, 6) for _ in range(64 * 16)]
B = [bf16(-6, 6) for _ in range(256 * 16)]


def f32(e_lo, e_hi):
    s = r.getrandbits(1)
    e = r.randint(e_lo, e_hi) + 127
    return s << 31 | e << 23 | r.getrandbits(23)


C = []
for lane in range(128):
    for col in range(256):
        if lane < 64:
            C.append(f32(-110, 127) if r.random() < 0.7 else f32(-110, -90))
        else:
            C.append(f32(-24, 24))
# Exploration only: zeros, subnormals, values that become subnormal, Inf, NaN, max finite.
SP = [
    0x00000000,
    0x80000000,
    0x00000001,
    0x807FFFFF,
    0x00400000,
    0x00800000,
    0x80800000,
    0x01000000,
    0x03800000,
    0x03FFFFFF,
    0x04000001,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0x7F800001,
    0xFFC12345,
    0x7F7FFFFF,
    0xFF7FFFFF,
]
CS = []
for lane in range(128):
    for col in range(256):
        CS.append(
            SP[(lane * 256 + col) % len(SP)]
            if r.random() < 0.6
            else f32(-140 + 13, -100)
        )
d = sys.argv[1] if len(sys.argv) > 1 else "data"
os.makedirs(d, exist_ok=True)
open(f"{d}/A.bin", "wb").write(struct.pack(f"<{len(A)}H", *A))
open(f"{d}/B.bin", "wb").write(struct.pack(f"<{len(B)}H", *B))
open(f"{d}/C.bin", "wb").write(struct.pack(f"<{len(C)}I", *C))
open(f"{d}/C_special.bin", "wb").write(struct.pack(f"<{len(CS)}I", *CS))

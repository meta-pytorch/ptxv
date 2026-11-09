# probes/tc_f16 inputs: A[128][64], B[64][64] f16, normal(0, 1) values rounded to f16, with
# magnitudes below 2^-14 (f16 subnormals) moved to +-2^-14 (deterministic).
import random
import struct
import sys

r = random.Random(11)


def vals(n):
    out = []
    for _ in range(n):
        x = r.gauss(0.0, 1.0)
        if abs(x) < 2.0**-14:
            x = 2.0**-14 if x >= 0 else -(2.0**-14)
        out.append(x)
    return out


d = sys.argv[1] if len(sys.argv) > 1 else "."
open(f"{d}/A.bin", "wb").write(struct.pack("<8192e", *vals(128 * 64)))
open(f"{d}/B.bin", "wb").write(struct.pack("<4096e", *vals(64 * 64)))

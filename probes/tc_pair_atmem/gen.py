# probes/tc_pair_atmem pattern inputs (bf16, K = 16, per CTA rank c): A[c][i] = (i + 128 c,
# 1, 1, 0...) for i < 128, B[c][j] = (1, 512 (j + 1), 2^20 c, 0...) for j < 256.
import os
import struct
import sys


def bf(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0] >> 16


A, B = [], []
for c in range(2):
    for i in range(128):
        A += [bf(i + 128 * c), bf(1), bf(1)] + [0] * 13
    for j in range(256):
        B += [bf(1), bf(512 * (j + 1)), bf(2**20 * c)] + [0] * 13
d = sys.argv[1] if len(sys.argv) > 1 else "data"
open(os.path.join(d, "a.bin"), "wb").write(struct.pack("<%dH" % len(A), *A))
open(os.path.join(d, "b.bin"), "wb").write(struct.pack("<%dH" % len(B), *B))

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/tc_tf32_mn shared-memory image (18,432 bytes, deterministic): A[128][8], A2[128][8],
# B[16][8], B2[16][8] random tf32-exact f32 values (low 13 bits zero). MN-major
# SWIZZLE_128B_BASE32B: element (r, k) at swz(start + (r % 32) 4 + (k % 4) 128 + (r / 32) LBO
# + (k / 4) SBO), swz(a) = a ^ (((a >> 7) & 3) << 5). A at 0 (LBO 1024, SBO 4096); A2
# K-major no swizzle at 8192 (SBO 128, LBO 2048); B2 MN-major at 12288 (LBO 1024, SBO 1024);
# B K-major no swizzle at 16384 (SBO 128, LBO 256).
import random
import struct
import sys

r = random.Random(37)


def tf(v):
    return struct.unpack("<I", struct.pack("<f", v))[0] & 0xFFFFE000


A = [[tf(r.uniform(-2, 2)) for _ in range(8)] for _ in range(128)]
A2 = [[tf(r.uniform(-2, 2)) for _ in range(8)] for _ in range(128)]
B = [[tf(r.uniform(-2, 2)) for _ in range(8)] for _ in range(16)]
B2 = [[tf(r.uniform(-2, 2)) for _ in range(8)] for _ in range(16)]
img = bytearray(18432)


def put(a, w):
    img[a : a + 4] = struct.pack("<I", w)


def mn(start, lbo, sbo, r, k):
    a = start + (r % 32) * 4 + (k % 4) * 128 + (r // 32) * lbo + (k // 4) * sbo
    return a ^ (((a >> 7) & 3) << 5)


def kmaj(start, lbo, sbo, r, k):
    return start + (r // 8) * sbo + (r % 8) * 16 + (k // 4) * lbo + (k % 4) * 4


for m in range(128):
    for k in range(8):
        put(mn(0, 1024, 4096, m, k), A[m][k])
        put(kmaj(8192, 2048, 128, m, k), A2[m][k])
for n in range(16):
    for k in range(8):
        put(mn(12288, 1024, 1024, n, k), B2[n][k])
        put(kmaj(16384, 256, 128, n, k), B[n][k])
open(sys.argv[1] if len(sys.argv) > 1 else "img.bin", "wb").write(bytes(img))

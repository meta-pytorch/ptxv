# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# probes/hmma_special input: 160 tests (deterministic), each a logical A[16][16] bf16,
# B[16][8] bf16, C[16][8] f32 -> per lane 12 words: a0..a3, b0, b1, c0..c3, pad, pad.
# Writes in.bin (and tests.json with the logical matrices when given a second argument).
import json
import random
import struct
import sys

r = random.Random(31)


def bf(v):  # f32 value -> bf16 bits (RNE), v given as float
    x = struct.unpack("<I", struct.pack("<f", v))[0]
    return ((x + 0x7FFF + ((x >> 16) & 1)) >> 16) & 0xFFFF


def f32b(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


SUB = lambda: r.choice([1, 2, 0x7F, 0x40, 0x13, 0x55]) | (
    r.choice([0, 0x8000])
)  # bf16 subnormal


def NRM(lo=-2, hi=2):
    return bf(r.choice([-1, 1]) * (1 + r.random()) * 2.0 ** r.randint(lo, hi))


INF, NINF, QNAN = 0x7F80, 0xFF80, 0x7FC0
tests = []


def T(kind, A, B, C):
    tests.append((kind, A, B, C))


Z16 = lambda: [[0] * 16 for _ in range(16)]
ZB = lambda: [[0] * 8 for _ in range(16)]
ZC = lambda: [[0] * 8 for _ in range(16)]
for _ in range(16):  # 1 subnormal product alone (A sub, B normal)
    A, B, C = Z16(), [[NRM(-3, 3) for _ in range(8)] for _ in range(16)], ZC()
    for m in range(16):
        A[m][r.randrange(16)] = SUB()
    T("sub_alone", A, B, C)
for _ in range(16):  # subnormal A next to tiny normal products (2^-120 .. 2^-110)
    A = [
        [(SUB() if r.random() < 0.5 else NRM(-62, -55)) for _ in range(16)]
        for _ in range(16)
    ]
    B = [
        [NRM(-60, -55) if r.random() < 0.5 else NRM(-2, 2) for _ in range(8)]
        for _ in range(16)
    ]
    T("sub_mixed", A, B, ZC())
for _ in range(16):  # normal inputs, results in the f32 subnormal range
    A = [[NRM(-72, -66) for _ in range(16)] for _ in range(16)]
    B = [[NRM(-72, -66) for _ in range(8)] for _ in range(16)]
    T("res_sub", A, B, ZC())
for _ in range(16):  # subnormal accumulators, zero or tiny products
    A = [
        [(NRM(-75, -68) if r.random() < 0.3 else 0) for _ in range(16)]
        for _ in range(16)
    ]
    B = [[NRM(-75, -68) for _ in range(8)] for _ in range(16)]
    C = [
        [
            r.choice([1, 0x7FFFFF, 0x400000, 0x80000001, 0x80123456, 0x2A])
            for _ in range(8)
        ]
        for _ in range(16)
    ]
    T("acc_sub", A, B, C)
for _ in range(16):  # Inf / NaN inputs
    A = [[NRM() for _ in range(16)] for _ in range(16)]
    B = [[NRM() for _ in range(8)] for _ in range(16)]
    for m in range(16):
        k = r.randrange(16)
        A[m][k] = r.choice([INF, NINF, QNAN, 0x7F81, 0xFFC5, 0x7FA0])
        if r.random() < 0.3:
            B[k][r.randrange(8)] = r.choice([0, 0x8000, INF, NINF, SUB()])
    T("infnan_in", A, B, ZC())
for _ in range(16):  # Inf / NaN accumulators
    A = [[NRM() for _ in range(16)] for _ in range(16)]
    B = [[NRM() for _ in range(8)] for _ in range(16)]
    if r.random() < 0.5:
        for m in range(16):
            A[m][r.randrange(16)] = r.choice([INF, NINF])
    C = [
        [
            r.choice(
                [0x7F800000, 0xFF800000, 0x7FC00000, 0x7FC01234, 0xFF800001, f32b(1.0)]
            )
            for _ in range(8)
        ]
        for _ in range(16)
    ]
    T("infnan_acc", A, B, C)
for _ in range(16):  # overflow from finite inputs
    A = [[NRM(60, 64) for _ in range(16)] for _ in range(16)]
    B = [
        [NRM(60, 64) if r.random() < 0.7 else NRM(-2, 2) for _ in range(8)]
        for _ in range(16)
    ]
    C = [[f32b(r.choice([0.0, 3.0e38, -3.0e38])) for _ in range(8)] for _ in range(16)]
    T("overflow", A, B, C)
for _ in range(16):  # subnormal x Inf (FTZ of inputs: NaN; exact: Inf)
    A, B = Z16(), ZB()
    for m in range(16):
        k = r.randrange(16)
        A[m][k] = SUB()
        for n in range(8):
            B[k][n] = r.choice([INF, NINF])
    T("sub_inf", A, B, ZC())
for _ in range(
    32
):  # subnormal input x huge B next to products near 2^-6: alignment exponent
    A, B, C = Z16(), ZB(), ZC()
    for m in range(16):
        for k in range(16):
            A[m][k] = SUB() if k == m % 16 else NRM(-16, -12)
    for k in range(16):
        for n in range(8):
            B[k][n] = (
                bf(r.choice([1, -1]) * (1 + r.random()) * 2.0 ** r.randint(124, 127))
                if k == n
                else NRM(-16, -12)
            )
    T("sub_align", A, B, ZC())
words = []
for kind, A, B, C in tests:
    for l in range(32):
        g, q = l // 4, l % 4
        a = []
        for j in range(4):
            row, col = g + 8 * (j & 1), 2 * q + 8 * (j >> 1)
            a.append(A[row][col] | A[row][col + 1] << 16)
        b = []
        for j in range(2):
            k = 2 * q + 8 * j
            b.append(B[k][g] | B[k + 1][g] << 16)
        c = [C[g + 8 * (j >> 1)][2 * q + (j & 1)] for j in range(4)]
        words += a + b + c + [0, 0]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<%dI" % len(words), *words)
)
if len(sys.argv) > 2:
    json.dump([[k, A, B, C] for k, A, B, C in tests], open(sys.argv[2], "w"))

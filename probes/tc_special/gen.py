# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""probes/tc_special: tcgen05.mma.kind::f16 (f32 D) on special values, f16 and bf16 operands.

Writes kernel.ptx (entries tcs_f16: A / B format 0, idesc 0x08100010; tcs_bf16: format 1,
idesc 0x08100490; both probes/tc_f16's kernel: D[128x64] = A[128x64] * B[64x64]^T in 4 MMAs of
K = 16, the first overwriting D) and one case directory per input set (A.bin, B.bin as 16-bit
codes, case.toml). Deterministic. Sets, per format:
  sub_alone   one subnormal A element per row, the rest 0; B normal
  sub_mixed   subnormal and small normal A elements, small and unit B elements
  sub_align   a subnormal A element times a large B element next to products just below it:
              the alignment exponent of a subnormal operand (field minimum or true exponent)
  sub_rand    random finite codes, half of them subnormal
  infnan_in   one Inf / NaN A element per row (several payloads), some zero / Inf / subnormal B
  sub_inf     subnormal x Inf
  acc_special K chunk 0 gives Inf / NaN / large results, chunks 1-3 normal products: special
              accumulators of MMAs 1-3
bf16 also:
  res_sub     normal inputs, results in the f32 subnormal range (accumulated as subnormals)
  acc_sub     chunk 0 gives f32-subnormal results, chunks 1-3 tiny or zero products
  overflow    finite products near 2^128
"""

import os
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M, N, K = 128, 64, 64

FMT = {
    "f16": dict(idesc=0x08100010, code=0, mbits=10, ebits=5, inf=0x7C00, qnan=0x7E00),
    "bf16": dict(idesc=0x08100490, code=1, mbits=7, ebits=8, inf=0x7F80, qnan=0x7FC0),
}


def enc(fmt, v):
    """Finite value -> 16-bit code, round to nearest even (no overflow handling: v in range)."""
    if fmt == "f16":
        return struct.unpack("<H", struct.pack("<e", v))[0]
    x = struct.unpack("<I", struct.pack("<f", v))[0]
    return ((x + 0x7FFF + ((x >> 16) & 1)) >> 16) & 0xFFFF


def sets(fmt, r):
    f = FMT[fmt]
    mb, eb = f["mbits"], f["ebits"]
    bias = (1 << (eb - 1)) - 1
    INF, NINF, QNAN = f["inf"], f["inf"] | 0x8000, f["qnan"]
    NANS = [QNAN, QNAN | 0x8000, INF | 1, INF | 0x8000 | 0x15, QNAN | 0x3]

    def sub():
        return r.randrange(1, 1 << mb) | r.choice([0, 0x8000])

    def nrm(lo, hi):
        return enc(fmt, r.choice([-1, 1]) * (1 + r.random()) * 2.0 ** r.randint(lo, hi))

    emin = 1 - bias  # smallest normal exponent
    zA = lambda: [[0] * K for _ in range(M)]
    zB = lambda: [[0] * K for _ in range(N)]
    out = {}

    A, B = zA(), [[nrm(-3, 3) for _ in range(K)] for _ in range(N)]
    for m in range(M):
        A[m][r.randrange(K)] = sub()
    out["sub_alone"] = (A, B)

    A = [
        [sub() if r.random() < 0.5 else nrm(emin, emin + 6) for _ in range(K)]
        for _ in range(M)
    ]
    B = [
        [nrm(-6, -1) if r.random() < 0.5 else nrm(-1, 1) for _ in range(K)]
        for _ in range(N)
    ]
    out["sub_mixed"] = (A, B)

    # Row m: a subnormal at k = 16 j + (m % 16) in every chunk j; B[n][k] large when
    # k % 16 == n % 16, else products of normals placed 20 .. 30 binades below the
    # subnormal product's field-exponent alignment.
    hi = min(bias, 15)  # B exponent of the large element
    A, B = zA(), zB()
    for m in range(M):
        for k in range(K):
            A[m][k] = sub() if k % 16 == m % 16 else nrm(emin, emin + 2)
    for n in range(N):
        for k in range(K):
            B[n][k] = (
                nrm(hi - 1, hi) if k % 16 == n % 16 else nrm(hi - mb - 22, hi - mb - 14)
            )
    out["sub_align"] = (A, B)

    def rnd_fin():
        if r.random() < 0.5:
            return sub()
        e = r.randrange(1, (1 << eb) - 1)
        if fmt == "bf16":
            e = r.randrange(1, 40) if r.random() < 0.5 else r.randrange(100, 150)
        return (e << mb) | r.randrange(1 << mb) | r.choice([0, 0x8000])

    out["sub_rand"] = (
        [[rnd_fin() for _ in range(K)] for _ in range(M)],
        [[rnd_fin() for _ in range(K)] for _ in range(N)],
    )

    A = [[nrm(-2, 2) for _ in range(K)] for _ in range(M)]
    B = [[nrm(-2, 2) for _ in range(K)] for _ in range(N)]
    for m in range(M):
        k = r.randrange(K)
        A[m][k] = r.choice([INF, NINF] + NANS)
        if r.random() < 0.4:
            B[r.randrange(N)][k] = r.choice([0, 0x8000, INF, NINF, sub()])
    out["infnan_in"] = (A, B)

    A, B = zA(), zB()
    for m in range(M):
        k = r.randrange(K)
        A[m][k] = sub()
        for n in range(N):
            if r.random() < 0.7:
                B[n][k] = r.choice([INF, NINF])
    out["sub_inf"] = (A, B)

    # Chunk 0 (k < 16): per row Inf, -Inf, NaN or a large finite product; chunks 1-3 normal.
    A = [[nrm(-2, 2) for _ in range(K)] for _ in range(M)]
    B = [[nrm(-2, 2) for _ in range(K)] for _ in range(N)]
    for m in range(M):
        kind = m % 4
        k = r.randrange(16)
        A[m][k] = [INF, NINF, QNAN, nrm(emin + 2 * bias - 4, emin + 2 * bias - 2)][kind]
    for n in range(N):
        if n % 8 == 0:
            B[n][r.randrange(16)] = INF
    out["acc_special"] = (A, B)

    if fmt == "bf16":
        out["res_sub"] = (
            [[nrm(-72, -66) for _ in range(K)] for _ in range(M)],
            [[nrm(-72, -66) for _ in range(K)] for _ in range(N)],
        )
        A = [
            [
                nrm(-72, -66) if k < 16 else (nrm(-80, -70) if r.random() < 0.3 else 0)
                for k in range(K)
            ]
            for _ in range(M)
        ]
        B = [[nrm(-72, -66) for _ in range(K)] for _ in range(N)]
        out["acc_sub"] = (A, B)
        A = [[nrm(60, 64) for _ in range(K)] for _ in range(M)]
        B = [
            [nrm(60, 64) if r.random() < 0.7 else nrm(-2, 2) for _ in range(K)]
            for _ in range(N)
        ]
        out["overflow"] = (A, B)
    return out


CASE = """name = "probe_tc_special_{name}"
description = \"\"\"
probes/tc_special set {name} ({fmt}): tcgen05.mma.kind::f16 on special values (gen.py). expected/
is GB300's output.\"\"\"
ptx = "../kernel.ptx"
entry = "tcs_{fmt}"
grid = [1, 1, 1]
block = [128, 1, 1]
outputs = ["D"]

[[buffers]]
name = "A"
dtype = "u16"
shape = [128, 64]
init = "file"
file = "A.bin"

[[buffers]]
name = "B"
dtype = "u16"
shape = [64, 64]
init = "file"
file = "B.bin"

[[buffers]]
name = "D"
dtype = "f32"
shape = [128, 64]
init = "nan"
{expect}
# A[m][k] row-major viewed as 3-D (k%8, m, k/8): box (8, 128, 8) lands in shared memory as
# 8 K-chunks of 128 rows x 16 B (LBO = 2048 B, SBO = 128 B).
[[params]]
kind = "tensormap"
buffer = "A"
box = [8, 128, 8]
dims = [8, 128, 8]
strides = [128, 16]

[[params]]
kind = "tensormap"
buffer = "B"
box = [8, 64, 8]
dims = [8, 64, 8]
strides = [128, 16]

[[params]]
kind = "ptr"
buffer = "D"

[[variants]]
id = "good"
expect = ["clean"]
"""


def main():
    hdr = open(os.path.join(HERE, "header.tmpl")).read()
    body = open(os.path.join(HERE, "kernel.tmpl")).read()
    lines = body.split("\n")
    directives, rest = "\n".join(lines[:3]), "\n".join(lines[3:])
    ptx = hdr.replace("{FMT}", "0 (f16, tcs_f16) / 1 (bf16, tcs_bf16)").replace(
        "{IDESC}", "0x08100010 / 0x08100490"
    )
    ptx += directives + "\n"
    for fmt in ("f16", "bf16"):
        f = FMT[fmt]
        ptx += (
            rest.replace("{ENTRY}", f"tcs_{fmt}")
            .replace("{IDESC}", f"{f['idesc']:#010x}")
            .replace("{FMT}", f"{f['code']} ({fmt})")
        )
    open(os.path.join(HERE, "kernel.ptx"), "w").write(ptx)
    for fmt in ("f16", "bf16"):
        r = random.Random(1000 + FMT[fmt]["code"])
        for name, (A, B) in sets(fmt, r).items():
            d = os.path.join(HERE, f"{fmt}_{name}")
            os.makedirs(d, exist_ok=True)
            flat = lambda X: struct.pack(
                "<%dH" % (len(X) * len(X[0])), *[c for row in X for c in row]
            )
            open(os.path.join(d, "A.bin"), "wb").write(flat(A))
            open(os.path.join(d, "B.bin"), "wb").write(flat(B))
            exp = os.path.exists(os.path.join(d, "expected", "D.bin"))
            open(os.path.join(d, "case.toml"), "w").write(
                CASE.format(
                    name=f"{fmt}_{name}",
                    fmt=fmt,
                    expect='expect = "expected/D.bin"\n' if exp else "",
                )
            )


if __name__ == "__main__":
    main()

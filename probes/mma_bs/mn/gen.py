# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Writes the MN-major probes of the block-scaled tcgen05.mma (N-MMA-BS): one case directory per
operand layout, kind::mxf8f6f4 scale_vec::1X, e4m3 x e4m3, ue8m0 scales, K = 128 as four K = 32
MMAs.

The host side (this script) writes each operand's shared-memory image for its layout, and the
eight smem descriptors (A's four K slices, then B's) with the start field relative to the image.
The kernels copy the images into shared memory, add the image base to each descriptor's start
and issue the four MMAs, so one kernel serves every layout. The logical operands are the same in
every case of a kernel, so every case's D must equal the K-major case's: a layout the hardware
reads differently from the formula here gives a different D.

- kernel.ptx (cta_group::1, M = 128, N = 64): probes/mma_bs's mma_mxf8 operands and scale
  factors; D must equal probes/mma_bs's expected mma_mxf8 D (GB300), which GB200 also returns.
- pair.ptx (cta_group::2, M = 256, N = 128, each CTA 128 A rows and 64 B rows): random operands
  (data/pair_*.bin); mma_mn_pair_ts has A in TMEM (the NVFP4x2-KV attention's P.V shape),
  mma_mn_pair_ss A in shared memory. D must equal expected/pair_D.bin (GB200, K-major case).

Layouts (8-bit elements, CUTLASS's canonical smem layouts):
- K-major, no swizzle: 8-row x 16-byte core matrices, rows 16 B apart, 8-row groups SBO apart,
  16-byte K chunks LBO apart.
- MN-major, swizzled with span s (32, 64, 128 B): rows of s bytes along M/N (s elements), 8 K-rows
  per atom, M/N atoms LBO apart, K-groups of 8 SBO apart, then the address swizzle
  (Swizzle<log2(s/16), 4, 3> on the absolute address).
- MN-major, no swizzle: 16 M/N elements per 16-byte row, 8 K-rows 16 B apart per core matrix,
  core matrices SBO apart along M/N, K-groups of 8 LBO apart.

Usage: python3 gen.py (from this directory; numpy for the pair operands, written once).
"""

import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "expected", "mma_mxf8e4m3xe4m3f32ue8m0.codes")
K = 128
A_BYTES, B_BYTES = (
    17408,
    8192,
)  # smA, smB in the kernels: 16 KiB (+1 KiB for an offset start)
LAYOUT = {0: 0, 128: 2, 64: 4, 32: 6}  # smem descriptor bits 61-63 by swizzle span


def swz(a, span):
    mask = {128: 7, 64: 3, 32: 1, 0: 0}[span]
    return a ^ (((a >> 7) & mask) << 4)


def addr(kind, r, k, span, lbo, sbo):
    if kind == "k":  # K-major, no swizzle
        return (r // 8) * sbo + (r % 8) * 16 + (k // 16) * lbo + k % 16
    if span == 0:  # MN-major, no swizzle
        return (r % 16) + (r // 16) * sbo + (k % 8) * 16 + (k // 8) * lbo
    return (r % span) + (r // span) * lbo + (k % 8) * span + (k // 8) * sbo


def image(codes, rows, kind, span, lbo, sbo, off, size):
    img = bytearray(size)
    seen = set()
    for r in range(rows):
        for k in range(K):
            a = off + addr(kind, r, k, span, lbo, sbo)
            p = swz(a, span) if kind == "mn" else a
            assert p < size and p not in seen, (kind, span, r, k, p)
            seen.add(p)
            img[p] = codes[r * K + k]
    return bytes(img)


def descs(kind, span, lbo, sbo, off):
    out = []
    for s in range(4):  # K slice s: elements 32 s .. 32 s + 31
        if kind == "k":
            start = off + s * 2 * lbo
        elif span == 0:
            start = off + s * 4 * lbo
        else:
            start = off + s * 4 * sbo
        assert start % 16 == 0
        out.append(
            (start >> 4)
            | ((lbo >> 4) << 16)
            | ((sbo >> 4) << 32)
            | (1 << 46)
            | (LAYOUT[span if kind == "mn" else 0] << 61)
        )
    return out


# Layout = (kind, span, LBO, SBO, start offset).
KA = ("k", 0, 2048, 128, 0)  # 128 rows
KB = ("k", 0, 1024, 128, 0)  # 64 rows
CASES = [  # kernel.ptx: (name, A layout, B layout, what)
    ("kmajor", KA, KB, "K-major A and B (probes/mma_bs's layout)"),
    (
        "a_sw128",
        ("mn", 128, 0, 1024, 0),
        KB,
        "MN-major A, SWIZZLE_128B (one atom along M)",
    ),
    (
        "a_sw128_off512",
        ("mn", 128, 0, 1024, 512),
        KB,
        "MN-major A, SWIZZLE_128B, start 512 B into a 1024-byte swizzle atom (an mxfp8 decode's)",
    ),
    (
        "a_sw64",
        ("mn", 64, 512, 1024, 0),
        KB,
        "MN-major A, SWIZZLE_64B (two atoms along M, LBO 512)",
    ),
    (
        "a_sw32",
        ("mn", 32, 256, 1024, 0),
        KB,
        "MN-major A, SWIZZLE_32B (four atoms along M, LBO 256)",
    ),
    (
        "a_none",
        ("mn", 0, 1024, 128, 0),
        KB,
        "MN-major A without swizzle (SBO 128 along M, LBO 1024 along K)",
    ),
    ("b_sw64", KA, ("mn", 64, 0, 512, 0), "MN-major B, SWIZZLE_64B (one atom along N)"),
    (
        "b_sw32",
        KA,
        ("mn", 32, 256, 512, 0),
        "MN-major B, SWIZZLE_32B (two atoms along N, LBO 256)",
    ),
    (
        "ab_sw128_sw64",
        ("mn", 128, 0, 1024, 0),
        ("mn", 64, 0, 512, 0),
        "MN-major A (SWIZZLE_128B) and B (SWIZZLE_64B)",
    ),
]
PAIR = [  # pair.ptx: (name, A layout or None = A in TMEM, B layout per CTA, what)
    ("pair_ts_kmajor", None, KB, "A in TMEM, K-major B"),
    (
        "pair_ts_b_sw64",
        None,
        ("mn", 64, 0, 512, 0),
        "A in TMEM, MN-major B, SWIZZLE_64B (the NVFP4x2-KV attention's P.V)",
    ),
    (
        "pair_ts_b_sw32",
        None,
        ("mn", 32, 256, 512, 0),
        "A in TMEM, MN-major B, SWIZZLE_32B (LBO 256)",
    ),
    (
        "pair_ts_b_none",
        None,
        ("mn", 0, 512, 128, 0),
        "A in TMEM, MN-major B without swizzle (SBO 128, LBO 512)",
    ),
    ("pair_ss_kmajor", KA, KB, "K-major A and B in shared memory"),
    (
        "pair_ss_a_sw128",
        ("mn", 128, 0, 1024, 0),
        KB,
        "MN-major A, SWIZZLE_128B, K-major B",
    ),
    (
        "pair_ss_ab",
        ("mn", 128, 0, 1024, 0),
        ("mn", 64, 0, 512, 0),
        "MN-major A (SWIZZLE_128B) and B (SWIZZLE_64B)",
    ),
]

HEAD = '''name = "mma_bs_mn_{name}"
description = """
N-MMA-BS probe (probes/mma_bs/mn, gen.py): block-scaled kind::mxf8f6f4 scale_vec::1X, {shape},
with {what}. The kernel copies the operand images (data/) into shared memory and issues four
K = 32 MMAs with data/desc.bin's descriptors. {ref}"""
arch = "sm_100a"
ptx = "../{ptx}"
entry = "{entry}"
grid = [{grid}, 1, 1]
{cluster}block = [128, 1, 1]
outputs = ["D"]
'''
BUF = """
[[buffers]]
name = "{name}"
dtype = "{dtype}"
shape = {shape}
init = "file"
file = "{file}"
"""
TAIL = """
[[buffers]]
name = "D"
dtype = "f32"
shape = {dshape}
init = "nan"
expect = "{expect}"

[[params]]
kind = "ptr"
buffer = "{a}"

[[params]]
kind = "ptr"
buffer = "Bimg"

[[params]]
kind = "ptr"
buffer = "desc"

[[params]]
kind = "ptr"
buffer = "D"

# Instruction descriptor: {idesc_note}
[[params]]
kind = "u32"
value = {idesc:#010x}

[[params]]
kind = "ptr"
buffer = "SFA"

[[params]]
kind = "ptr"
buffer = "SFB"

[[variants]]
id = "good"
expect = ["clean"]
{extra}"""
# Model-only refusal variants (kernel.ptx's @A0 / @IDESC lines) on the a_sw128 case; the
# violation is the unwritten accumulator read back (S-INIT).
REFUSALS = """
[[variants]]
id = "mid_row"
summary = "A's start 16 B into a SWIZZLE_128B row (MN-major starts measured on row boundaries only): refused"
marker = "@A0"
find = "%rd10, %rd4;"
replace = "%rd10, %rd4;\\n    add.u64         %rd10, %rd10, 1;"
expect = ["unmodeled", "violation"]

[[variants]]
id = "fp4_mn"
summary = "A / B formats e2m1 with MN-major A (FP4 MN-major is not measured): refused"
marker = "@IDESC"
find = "[idesc];"
replace = "[idesc];\\n    or.b32          %r11, %r11, 0x1680;"
expect = ["unmodeled", "violation"]
"""


def note(m, n, la, lb):
    s = f"e4m3 x e4m3, N = {n}, ue8m0, M = {m}"
    if la is not None and la[0] == "mn":
        s += ", A MN-major (bit 15)"
    if lb[0] == "mn":
        s += ", B MN-major (bit 16)"
    return s


def write_case(name, files, toml):
    d = os.path.join(HERE, name)
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    for fn, data in files:
        open(os.path.join(d, "data", fn), "wb").write(data)
    open(os.path.join(d, "case.toml"), "w").write(toml)


def pair_operands():
    """data/pair_*.bin: random e4m3 codes (no NaN: 0x7f / 0xff) and ue8m0 scale words (every
    byte 0x70..0x8f: 2^-15 .. 2^16), SFA / SFB per (CTA, lane, column)."""
    paths = [
        os.path.join(HERE, "data", f"pair_{n}.bin") for n in ("A", "B", "SFA", "SFB")
    ]
    if not all(os.path.exists(p) for p in paths):
        import numpy as np

        rng = np.random.default_rng(100)
        os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
        for p, shape in zip(paths[:2], [(256, K), (128, K)]):
            c = rng.integers(0, 256, shape, dtype=np.uint8)
            c[(c & 0x7F) == 0x7F] ^= 1
            c.tofile(p)
        for p in paths[2:]:
            rng.integers(0x70, 0x90, (2, 128, 16, 4), dtype=np.uint8).tofile(p)
    return [open(p, "rb").read() for p in paths[:2]]


def main():
    a = open(SRC + ".A.bin", "rb").read()
    b = open(SRC + ".B.bin", "rb").read()
    assert len(a) == 128 * K and len(b) == 64 * K
    sf = "../../expected/mma_mxf8e4m3xe4m3f32ue8m0.codes"
    for name, la, lb, what in CASES:
        idesc = 0x08900000 | ((la[0] == "mn") << 15) | ((lb[0] == "mn") << 16)
        toml = HEAD.format(
            name=name,
            shape="M = 128, N = 64, probes/mma_bs's mma_mxf8 operands",
            what=what,
            ref="GB200's D equals probes/mma_bs's mma_mxf8 D (GB300) for every case.",
            ptx="kernel.ptx",
            entry="mma_mn",
            grid=1,
            cluster="",
        )
        toml += BUF.format(
            name="Aimg", dtype="u8", shape=f"[{A_BYTES}]", file="data/Aimg.bin"
        )
        toml += BUF.format(
            name="Bimg", dtype="u8", shape=f"[{B_BYTES}]", file="data/Bimg.bin"
        )
        toml += BUF.format(name="desc", dtype="u64", shape="[8]", file="data/desc.bin")
        toml += BUF.format(
            name="SFA", dtype="u32", shape="[128, 16]", file=sf + ".SFA.bin"
        )
        toml += BUF.format(
            name="SFB", dtype="u32", shape="[128, 16]", file=sf + ".SFB.bin"
        )
        toml += TAIL.format(
            dshape="[128, 64]",
            expect=sf + ".D.bin",
            a="Aimg",
            idesc_note=note(128, 64, la, lb),
            idesc=idesc,
            extra=REFUSALS if name == "a_sw128" else "",
        )
        files = [
            ("Aimg.bin", image(a, 128, *la, size=A_BYTES)),
            ("Bimg.bin", image(b, 64, *lb, size=B_BYTES)),
            ("desc.bin", struct.pack("<8Q", *(descs(*la) + descs(*lb)))),
        ]
        write_case(name, files, toml)
    pa, pb = pair_operands()
    for name, la, lb, what in PAIR:
        idesc = (
            0x10A00000
            | ((la is not None and la[0] == "mn") << 15)
            | ((lb[0] == "mn") << 16)
        )
        toml = HEAD.format(
            name=name,
            shape="cta_group::2, M = 256, N = 128 (data/pair_*.bin operands)",
            what=what,
            ref="GB200's D equals the K-major cases' (expected/pair_D.bin) for every case.",
            ptx="pair.ptx",
            entry="mma_mn_pair_ts" if la is None else "mma_mn_pair_ss",
            grid=2,
            cluster="cluster = [2, 1, 1]\n",
        )
        if la is None:
            toml += BUF.format(
                name="A", dtype="u8", shape=f"[256, {K}]", file="../data/pair_A.bin"
            )
        else:
            toml += BUF.format(
                name="Aimg", dtype="u8", shape=f"[2, {A_BYTES}]", file="data/Aimg.bin"
            )
        toml += BUF.format(
            name="Bimg", dtype="u8", shape=f"[2, {B_BYTES}]", file="data/Bimg.bin"
        )
        toml += BUF.format(name="desc", dtype="u64", shape="[8]", file="data/desc.bin")
        toml += BUF.format(
            name="SFA", dtype="u32", shape="[2, 128, 16]", file="../data/pair_SFA.bin"
        )
        toml += BUF.format(
            name="SFB", dtype="u32", shape="[2, 128, 16]", file="../data/pair_SFB.bin"
        )
        toml += TAIL.format(
            dshape="[256, 128]",
            expect="../expected/pair_D.bin",
            a="A" if la is None else "Aimg",
            idesc_note=note(256, 128, la, lb),
            idesc=idesc,
            extra="",
        )
        # CTA r holds A rows 128r.. and B rows 64r.., each in its own image at the same address.
        files = [
            (
                "Bimg.bin",
                b"".join(
                    image(pb[64 * K * r :], 64, *lb, size=B_BYTES) for r in (0, 1)
                ),
            ),
            ("desc.bin", struct.pack("<8Q", *(descs(*(la or KA)) + descs(*lb)))),
        ]
        if la is not None:
            files.append(
                (
                    "Aimg.bin",
                    b"".join(
                        image(pa[128 * K * r :], 128, *la, size=A_BYTES) for r in (0, 1)
                    ),
                )
            )
        write_case(name, files, toml)


if __name__ == "__main__":
    main()

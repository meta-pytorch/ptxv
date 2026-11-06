#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TMEM layout and operand split of tcgen05.mma.kind::f16 at M = 128 (probes/tc_m128, README.md).

  python3 run.py gen              # write expected/<cfg>.{A,B}.bin and expected/configs.json
  python3 run.py run [cfg...]     # run every config on GB300 (`ptxv probe`), write <cfg>.D.bin
  python3 run.py report [cfg...]  # decode the pattern configs: which (A row, B row, CTA) lands where

Inputs (bf16, K = 16, per CTA rank c: A[c] 128 rows, B[c] 256 rows):
  pattern: A[c][i] = (i + 128 c, 1, 1, 0...), B[c][j] = (1, 512 (j + 1), 2^20 c, 0...), so the
           exact f32 result a.b = (i + 128 c_A) + 512 (j + 1) + 2^20 c_B names the A row, the
           B row and the CTA each came from;
  random:  every element normal(0, 1) rounded to bf16 (numerics through the layout);
  e4m3:    kind::f8f6f4 entries, random normal e4m3 codes (32 bytes per row: K = 32).
D is the dump of all 128 lanes x 256 columns of each CTA's TMEM after the MMA; unwritten cells
hold the sentinel 0xff800000 | lane << 8 | column.
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "expected")
PTXV = os.environ.get("PTXV", "ptxv")

# (name, entry, M, N, data); mma_pair is cta_group::2 (cluster of 2), mma_one cta_group::1.
CONFIGS = (
    [(f"pair_m128_n{n}", "mma_pair", 128, n, "pattern") for n in (16, 32, 48, 64, 128, 256)]
    + [(f"pair_m128_n{n}_rand", "mma_pair", 128, n, "random") for n in (32, 64, 256)]
    + [(f"pair_m256_n{n}", "mma_pair", 256, n, "pattern") for n in (32, 64)]
    + [(f"one_m128_n{n}", "mma_one", 128, n, "pattern") for n in (8, 16, 24, 40, 136, 256)]
    + [(f"one_m128_n{n}_rand", "mma_one", 128, n, "random") for n in (8, 24)]
    + [("one_m64_n8", "mma_one", 64, 8, "pattern")]
    + [(f"pair_f8_m128_n{n}", "mma_pair_f8", 128, n, "e4m3") for n in (16, 32, 64)]
    + [(f"one_f8_m128_n{n}", "mma_one_f8", 128, n, "e4m3") for n in (8, 24)]
)

# mma_img configs: (name, M, N, data, A layout, B layout); a layout is (MN-major?, LBO, SBO),
# no swizzle. K-major: element (r, k) at (r/8) SBO + (r%8) 16 + (k/8) LBO + (k%8) 2;
# MN-major (CUTLASS INTERLEAVE: ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))): at (r/8) SBO + (r%8) 2 +
# (k/8) LBO + (k%8) 16.
IMG = [
    ("img_k_m128_n8", 128, 8, "pattern", (0, 2048, 128), (0, 4096, 128)),
    ("img_mna_m128_n8", 128, 8, "pattern", (1, 2048, 128), (0, 4096, 128)),
    ("img_mna_m128_n8_x", 128, 8, "pattern", (1, 128, 256), (0, 4096, 128)),
    ("img_mna_m128_n8_rand", 128, 8, "random", (1, 2048, 128), (0, 4096, 128)),
    ("img_mna_m128_n16_rand", 128, 16, "random", (1, 128, 256), (0, 4096, 128)),
    ("img_mna_m64_n8_rand", 64, 8, "random", (1, 1024, 128), (0, 4096, 128)),
    ("img_mnb_m128_n16", 128, 16, "pattern", (0, 2048, 128), (1, 256, 128)),
    ("img_mnb_m128_n32_rand", 128, 32, "random", (0, 2048, 128), (1, 128, 512)),
    ("img_mnab_m128_n8_rand", 128, 8, "random", (1, 128, 256), (1, 128, 256)),
]


def image(mat, lay, nbytes):
    """The shared-memory image of a rows x 16 bf16 matrix (u16 codes) in layout `lay`."""
    mn, lbo, sbo = lay
    img = np.zeros(nbytes // 2, np.uint16)
    seen = set()
    for r in range(mat.shape[0]):
        for k in range(16):
            if mn:
                off = (r // 8) * sbo + (r % 8) * 2 + (k // 8) * lbo + (k % 8) * 16
            else:
                off = (r // 8) * sbo + (r % 8) * 16 + (k // 8) * lbo + (k % 8) * 2
            assert off + 2 <= nbytes and off not in seen
            seen.add(off)
            img[off // 2] = mat[r, k]
    return img


def flags(lay):
    _, lbo, sbo = lay
    return (lbo >> 4) << 16 | (sbo >> 4) << 32 | 1 << 46


def idesc(m, n, f8=False):
    # D f32 (bits 4-5 = 1), A / B bf16 (bits 7-9, 10-12 = 1; e4m3 = 0 for kind::f8f6f4),
    # N >> 3 at bit 17, M >> 4 at bit 24.
    ab = 0 if f8 else (1 << 7) | (1 << 10)
    return (1 << 4) | ab | ((n >> 3) << 17) | ((m >> 4) << 24)


def bf16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    # round to nearest even
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16)


def inputs(data, ctas, seed):
    if data == "pattern":
        a = np.zeros((ctas, 128, 16), np.float32)
        b = np.zeros((ctas, 256, 16), np.float32)
        for c in range(ctas):
            a[c, :, 0] = np.arange(128) + 128 * c
            a[c, :, 1] = 1
            a[c, :, 2] = 1
            b[c, :, 0] = 1
            b[c, :, 1] = 512 * (np.arange(256) + 1)
            b[c, :, 2] = 2.0**20 * c
    elif data == "e4m3":
        # random normal e4m3 codes (exponent field 1..15, not the NaN code 0x7f / 0xff)
        rng = np.random.default_rng(seed)
        def codes(shape):
            c = rng.integers(0, 256, shape, dtype=np.uint16).astype(np.uint8)
            e = (c >> 3) & 15
            c = np.where(e == 0, c | 0x08, c)
            c = np.where((c & 0x7F) == 0x7F, c ^ 1, c)
            return c.astype(np.uint8).view(np.uint16).reshape(shape[:-1] + (16,))
        return codes((ctas, 128, 32)), codes((ctas, 256, 32))
    else:
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((ctas, 128, 16)).astype(np.float32)
        b = rng.standard_normal((ctas, 256, 16)).astype(np.float32)
    return bf16(a), bf16(b)


def gen():
    os.makedirs(EXP, exist_ok=True)
    meta = []
    for i, (name, entry, m, n, data) in enumerate(CONFIGS):
        ctas = 2 if entry.startswith("mma_pair") else 1
        a, b = inputs(data, ctas, 1000 + i)
        a.tofile(os.path.join(EXP, f"{name}.A.bin"))
        b.tofile(os.path.join(EXP, f"{name}.B.bin"))
        meta.append(dict(name=name, entry=entry, m=m, n=n, data=data, ctas=ctas,
                         idesc=idesc(m, n, data == "e4m3")))
    for i, (name, m, n, data, la, lb) in enumerate(IMG):
        a, b = inputs(data, 1, 2000 + i)
        image(a[0, :m], la, 4096).tofile(os.path.join(EXP, f"{name}.A.bin"))
        image(b[0, :n], lb, 8192).tofile(os.path.join(EXP, f"{name}.B.bin"))
        meta.append(dict(name=name, entry="mma_img", m=m, n=n, data=data, ctas=1,
                         idesc=idesc(m, n) | la[0] << 15 | lb[0] << 16,
                         aflags=flags(la), bflags=flags(lb)))
    with open(os.path.join(EXP, "configs.json"), "w") as f:
        json.dump(meta, f, indent=0)


CASE_TOML = """name = "tc_m128_{name}"
description = "probes/tc_m128 config {name} (run.py)"
ptx = "kernel.ptx"
entry = "{entry}"
grid = [{ctas}, 1, 1]
block = [128, 1, 1]
{cluster}outputs = ["D"]

[[buffers]]
name = "A"
dtype = "u16"
shape = [{ctas}, 128, 16]
init = "file"
file = "{exp}/{name}.A.bin"

[[buffers]]
name = "B"
dtype = "u16"
shape = [{ctas}, 256, 16]
init = "file"
file = "{exp}/{name}.B.bin"

[[buffers]]
name = "D"
dtype = "u32"
shape = [{ctas}, 128, 256]
init = "zeros"

[[params]]
kind = "ptr"
buffer = "A"

[[params]]
kind = "ptr"
buffer = "B"

[[params]]
kind = "ptr"
buffer = "D"

[[params]]
kind = "u32"
value = {idesc}

[[variants]]
id = "good"
expect = ["clean"]
"""


def case_toml(cfg, exp=EXP):
    cluster = "cluster = [2, 1, 1]\n" if cfg["ctas"] == 2 else ""
    t = CASE_TOML.format(cluster=cluster, exp=exp, **cfg)
    if cfg["entry"] == "mma_img":
        # raw images (A 4096 bytes, B 8192 bytes) and the two descriptor-flag parameters
        t = t.replace("shape = [1, 128, 16]", "shape = [2048]").replace(
            "shape = [1, 256, 16]", "shape = [4096]")
        extra = "".join(f'\n[[params]]\nkind = "u64"\nvalue = {cfg[k]}\n'
                        for k in ("aflags", "bflags"))
        t = t.replace("\n[[variants]]", extra + "\n[[variants]]")
    return t


def run(names):
    meta = json.load(open(os.path.join(EXP, "configs.json")))
    for cfg in meta:
        if names and cfg["name"] not in names:
            continue
        d = tempfile.mkdtemp(prefix="tc_m128_")
        shutil.copy(os.path.join(HERE, "kernel.ptx"), d)
        with open(os.path.join(d, "case.toml"), "w") as f:
            f.write(case_toml(cfg))
        out = os.path.join(d, "out")
        r = subprocess.run([PTXV, "probe", d, "--out", out], capture_output=True, text=True)
        dump = os.path.join(out, "good.probe.bin")
        if r.returncode or not os.path.exists(dump):
            print(cfg["name"], "FAILED", r.stdout[-2000:], r.stderr[-2000:])
            continue
        shutil.copy(dump, os.path.join(EXP, f"{cfg['name']}.D.bin"))
        print(cfg["name"], "ok")
        shutil.rmtree(d)


def report(names):
    meta = json.load(open(os.path.join(EXP, "configs.json")))
    for cfg in meta:
        if cfg["data"] != "pattern" or (names and cfg["name"] not in names):
            continue
        p = os.path.join(EXP, f"{cfg['name']}.D.bin")
        if not os.path.exists(p):
            continue
        d = np.fromfile(p, np.uint32).reshape(cfg["ctas"], 128, 256)
        print(f"== {cfg['name']} (M {cfg['m']}, N {cfg['n']})")
        for c in range(cfg["ctas"]):
            # Per lane: the written columns and the (A row, B row, B CTA) of the first / last.
            runs = []
            for lane in range(128):
                cols = [col for col in range(256)
                        if d[c, lane, col] != (0xFF800000 | lane << 8 | col)]
                if not cols:
                    continue
                dec = []
                for col in (cols[0], cols[-1]):
                    v = int(struct.unpack("<f", struct.pack("<I", int(d[c, lane, col])))[0])
                    dec.append((v % 512, (v >> 9) % 2048 - 1, v >> 20))
                runs.append((lane, cols[0], cols[-1], len(cols), dec))
            # Print lane groups with the same column run.
            i = 0
            while i < len(runs):
                j = i
                while j + 1 < len(runs) and runs[j + 1][1:4] == runs[i][1:4] \
                        and runs[j + 1][0] == runs[j][0] + 1:
                    j += 1
                l0, c0, c1, nc, d0 = runs[i]
                print(f"  cta {c} lanes {l0}..{runs[j][0]} cols {c0}..{c1} ({nc}):"
                      f" first lane (arow, brow, bcta) {d0[0]}..{d0[1]};"
                      f" last lane {runs[j][4][0]}..{runs[j][4][1]}")
                i = j + 1


if __name__ == "__main__":
    {"gen": lambda a: gen(), "run": run, "report": report}[sys.argv[1]](sys.argv[2:])

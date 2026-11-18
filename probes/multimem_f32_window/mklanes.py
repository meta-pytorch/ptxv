#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 225 repro inputs (run.sh, check.py): picks f32 lanes out of GB300 multimem.ld_reduce
sweeps (probes/multimem/mm.cu `ldr`; result dirs as `ldr` writes them: <test>.in<i>.bin,
<test>.out<dev>.<rep>.bin) and writes the probe's data/ and expected/. Tests: f32
(add.f32, relaxed.sys), f32w (add.f32, weak), f32x4 (add.v4.f32: 16-byte groups kept whole).

Lanes picked, per test:
- m4 (4 members):
  - CANCEL (probes/multimem_add16_rules/sweep/gen2.py, kind cancelf: B, -B, R, 0 with R 16 to 72
    binades below B, eB 40 to 230): every lane where the model before MISS 225 (each member
    truncated 63 bits below the largest member's last mantissa bit, summed exactly, rounded to
    nearest even) differs from GB300, and every 16th lane as a control;
  - ZERO (gen1.py, kind zero): every lane (signed zeros; x, -x, -0, -0);
  - RAND (mm ldr's own fill, MM_SEED 1, 2, 3): every lane where that model differs from GB300.
- m2 (2 members): ZERO's 2-member lanes.
Each test's section is padded to whole 16-byte groups with lanes of +0 members (kind pad, not
measured; check.py skips them). Writes data/m<N>.in<d>.bin, expected/m<N>.out.bin,
expected/lanes.csv and m<N>/case.toml.
usage: mklanes.py CANCEL ZERO RAND1 RAND2 RAND3 OUTDIR
  CANCEL: dir with in4/ (inputs), gb4/ (GB300 outputs) and lanes.json (gen2.py);
  ZERO: dir with in4/, gb4/, in2/, gb2/ and lanes.json (gen1.py); RAND<s>: `mm ldr` output dirs."""
import json
import os
import struct
import sys

CANCEL, ZERO, R1, R2, R3, OUT = sys.argv[1:7]
TESTS = ["f32", "f32w", "f32x4"]


def words(path):
    b = open(path, "rb").read()
    return list(struct.unpack("<%dI" % (len(b) // 4), b))


def old_model(xs):
    """The f32 lane of the model before MISS 225 (finite members only; None otherwise)."""
    ds = []
    for x in xs:
        s, e, m = x >> 31, (x >> 23) & 0xFF, x & 0x7FFFFF
        if e == 255:
            return None
        ds.append((s, m | (0x800000 if e else 0), (e or 1) - 150))
    if all(m == 0 for _, m, _ in ds):
        return 0x80000000 if all(s for s, _, _ in ds) else 0
    lsb = max(x for _, m, x in ds if m) - 63
    tot = 0
    for s, m, x in ds:
        if m:
            q = m << (x - lsb) if x >= lsb else m >> (lsb - x)
            tot += -q if s else q
    if tot == 0:
        return 0
    neg, mag = tot < 0, abs(tot)
    top = mag.bit_length() - 1
    lo = max(top + lsb - 23, -149)
    sh = lo - lsb
    if sh <= 0:
        q = mag << -sh
    else:
        q, rem, half = mag >> sh, mag & ((1 << sh) - 1), 1 << (sh - 1)
        q += rem > half or (rem == half and q & 1)
    if q >= 1 << 24:
        q >>= 1
        lo += 1
    e, m = (0, q) if lo == -149 and q < 1 << 23 else (lo + 150, q - (1 << 23))
    return neg << 31 | (0x7F800000 if e >= 255 else e << 23 | m)


def out_of(d, t):
    for p in (f"{d}/{t}.out0.0.bin", f"{d}/{t}.out.bin"):
        if os.path.exists(p):
            return words(p)
    return None


rows = {4: {t: [] for t in TESTS}, 2: {t: [] for t in TESTS}}
index = ["members,test,lane,source,source lane,kind"]


def take(nm, t, src, ins, gb, lanes, kind):
    """Lanes (a set of word indices) of one source, whole 16-byte groups for f32x4."""
    if t == "f32x4":
        lanes = sorted({4 * (k // 4) + j for k in lanes for j in range(4)})
    for k in sorted(lanes):
        rows[nm][t].append(([ins[i][k] for i in range(nm)], gb[k]))
        index.append(f"{nm},{t},{len(rows[nm][t]) - 1},{src},{k},{kind}")


man = json.load(open(f"{CANCEL}/lanes.json"))
for t in TESTS:
    if f"4.{t}" not in man:
        continue
    ins = [words(f"{CANCEL}/in4/{t}.in{i}.bin") for i in range(4)]
    gb = out_of(f"{CANCEL}/gb4", t)
    kinds = [l[1] for l in man[f"4.{t}"]]
    cancel = [k for k in range(len(kinds)) if kinds[k] == "cancelf"]
    diff = {k for k in cancel if old_model([ins[i][k] for i in range(4)]) != gb[k]}
    take(4, t, "cancel", ins, gb, diff | set(cancel[::16]), "cancelf")
man = json.load(open(f"{ZERO}/lanes.json"))
for nm in (4, 2):
    for t in TESTS:
        if f"{nm}.{t}" not in man:
            continue
        ins = [words(f"{ZERO}/in{nm}/{t}.in{i}.bin") for i in range(nm)]
        gb = out_of(f"{ZERO}/gb{nm}", t)
        zero = {k for k, l in enumerate(man[f"{nm}.{t}"]) if l[1] == "zero"}
        take(nm, t, "zero", ins, gb, zero, "zero")
for s, d in (("rand1", R1), ("rand2", R2), ("rand3", R3)):
    for t in TESTS:
        ins = [words(f"{d}/{t}.in{i}.bin") for i in range(4)]
        gb = out_of(d, t)
        diff = set()
        for k in range(len(gb)):
            m = old_model([ins[i][k] for i in range(4)])
            if m is not None and m != gb[k]:
                diff.add(k)
        take(4, t, s, ins, gb, diff, "random")

# Each test's section is padded to whole 16-byte groups with lanes of +0 members (kind pad), so
# the f32x4 section starts 16-byte aligned.
for nm in (4, 2):
    for t in TESTS:
        while len(rows[nm][t]) % 4:
            rows[nm][t].append(([0] * nm, 0))
            index.append(f"{nm},{t},{len(rows[nm][t]) - 1},,,pad")
for nm in (4, 2):
    for i in range(nm):
        with open(f"{OUT}/data/m{nm}.in{i}.bin", "wb") as f:
            for t in TESTS:
                f.write(struct.pack("<%dI" % len(rows[nm][t]), *[r[0][i] for r in rows[nm][t]]))
    with open(f"{OUT}/expected/m{nm}.out.bin", "wb") as f:
        for t in TESTS:
            f.write(struct.pack("<%dI" % len(rows[nm][t]), *[r[1] for r in rows[nm][t]]))
    print(f"m{nm}: " + ", ".join(f"{t} {len(rows[nm][t])} lanes" for t in TESTS))
open(f"{OUT}/expected/lanes.csv", "w").write("\n".join(index) + "\n")

# The cases: m<N>/case.toml (m2/kernel.ptx is a copy of m4's).
for nm in (4, 2):
    n0, n1, n4 = (len(rows[nm][t]) for t in TESTS)
    words_ = n0 + n1 + n4
    threads = max(n0, n1, n4 // 4, 1)
    toml = [f'name = "multimem_f32_window_m{nm}"', 'description = """',
            f"MISS 225 repro: multimem.ld_reduce f32 adds of {nm} members (one per GPU) on lanes picked from",
            "GB300 sweeps (mklanes.py, expected/lanes.csv): out = the reduced lanes of the tests f32, f32w",
            f"and f32x4 ({n0}, {n1}, {n4} lanes). GB300's output for the same lanes: expected/m{nm}.out.bin",
            '(check.py compares lane by lane)."""', 'ptx = "kernel.ptx"', 'arch = "sm_103a"', 'entry = "mm32"',
            f"grid = [{(threads + 255) // 256}, 1, 1]", "block = [256, 1, 1]", 'outputs = ["out"]', ""]
    for d in range(nm):
        toml += ["[[buffers]]", f'name = "mem{d}"', 'dtype = "u32"', f"shape = [{words_}]", 'init = "file"',
                 f'file = "../data/m{nm}.in{d}.bin"', f"device = {d}", ""]
    toml += ["[[buffers]]", 'name = "out"', 'dtype = "u32"', f"shape = [{words_}]", 'init = "zeros"', "device = 0", "",
             "[[multicast]]", 'name = "mc"', f"size = {4 * words_}", "binds = ["]
    toml += [f'    {{ device = {d}, offset = 0, buffer = "mem{d}", byte_offset = 0, size = {4 * words_} }},' for d in range(nm)]
    toml += ["]", "", "[[params]]", 'kind = "ptr"', 'buffer = "mc"', "", "[[params]]", 'kind = "ptr"', 'buffer = "out"', ""]
    for v in (n0, n1, n4 // 4, n0, n0 + n1):
        toml += ["[[params]]", 'kind = "u32"', f"value = {v}", ""]
    toml += ["[[variants]]", 'id = "good"', 'summary = "as written"', 'expect = ["clean"]']
    os.makedirs(f"{OUT}/m{nm}", exist_ok=True)
    open(f"{OUT}/m{nm}/case.toml", "w").write("\n".join(toml) + "\n")
open(f"{OUT}/m2/kernel.ptx", "w").write(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m4", "kernel.ptx")).read())

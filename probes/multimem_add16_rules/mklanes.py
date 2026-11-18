#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 208 repro inputs (README.md): picks whole 16-byte vectors, as measured, out of GB300
multimem.ld_reduce sweeps (probes/multimem/mm.cu `ldr` with MM_IN; result dirs as `ldr` writes
them: <test>.in<i>.bin, <test>.out<dev>.<rep>.bin) and writes the probe's data/ and expected/.
A vector keeps its 8 lanes together: the calibrated rounding of a lane depends on the other
lanes of its 4-lane group, so GB300's recorded output holds for the vector only.

Vectors picked, per test (bf16x8, bf16x8a = .acc::f32, f16x8, f16x8a):
- m4 (4 members): bf16, from the depth sweep (lanes B, -B, R, 0, R's top bit d binades below
  B's): vectors until every (biased exponent eB of B, side) is covered, side = R at the largest
  depth GB300 keeps (d = 23 + eB mod 16) and at the next (dropped); f16: 2 vectors per depth
  20..29 (kept); every test: 2 vectors whose lanes are all -0; every vector of the random and
  targeted runs where GB300 differs from the model other than as MISS 162's other rounding
  (listed in MISSES below).
- m2 (2 members): every test: 2 all -0 vectors of the depth sweep, 4 vectors of lanes (B, -B),
  and every random-run vector with a lane whose members are all -0 (GB300 +0) and no infinite or
  NaN member.
usage: mklanes.py DEP4 DEP4IN RAND4 TGT4 DEP2 DEP2IN RAND2 OUTDIR"""
import os
import struct
import sys

DEP4, DEP4IN, RAND4, TGT4, DEP2, DEP2IN, RAND2, OUT = sys.argv[1:9]
TESTS = ["bf16x8", "bf16x8a", "f16x8", "f16x8a"]
# (run, test, vector index): the vectors of the random / targeted misses.
MISSES = {
    "rand4": {"bf16x8": [6084 // 8, 12066 // 8, 29044 // 8, 33588 // 8, 50295 // 8, 50402 // 8, 56589 // 8],
              "bf16x8a": [5911 // 8, 20225 // 8, 36016 // 8, 44022 // 8, 49003 // 8]},
    "tgt4": {"bf16x8a": [27629 // 8]},
}


def rd(path):
    return open(path, "rb").read()


def vec(buf, v):
    return buf[16 * v : 16 * v + 16]


def lanes(buf, v):
    return struct.unpack("<8H", vec(buf, v))


def pick_m4(test):
    ins = [rd(f"{DEP4}/{test}.in{i}.bin") for i in range(4)]
    out = rd(f"{DEP4}/{test}.out0.0.bin")
    dep = rd(f"{DEP4IN}/{test}.depth.bin")
    picked = []
    if test.startswith("bf16"):
        need = set()
        for v in range(len(dep)):
            if dep[v] == 255:
                continue
            for l in range(8):
                hs = [lanes(b, v)[l] for b in ins]
                big = max(h & 0x7FFF for h in hs)
                eb = big >> 7
                lim = 23 + eb % 16
                need |= {(eb, "kept"), (eb, "dropped")} if dep[v] in (lim, lim + 1) else set()
        covered = set()
        for v in range(len(dep)):
            if dep[v] == 255:
                continue
            new = set()
            for l in range(8):
                hs = [lanes(b, v)[l] for b in ins]
                eb = max(h & 0x7FFF for h in hs) >> 7
                lim = 23 + eb % 16
                side = {lim: "kept", lim + 1: "dropped"}.get(dep[v])
                if side and (eb, side) not in covered:
                    new.add((eb, side))
            if new:
                covered |= new
                picked.append(("dep4", v))
            if covered == need:
                break
    else:
        for d in range(20, 30):
            picked += [("dep4", v) for v in [v for v in range(len(dep)) if dep[v] == d][:2]]
    picked += [("dep4", v) for v in [v for v in range(len(dep)) if dep[v] == 255][:2]]
    for run, d in (("rand4", RAND4), ("tgt4", TGT4)):
        picked += [(run, v) for v in MISSES.get(run, {}).get(test, [])]
    src = {"dep4": DEP4, "rand4": RAND4, "tgt4": TGT4}
    return [(r, v, [vec(rd(f"{src[r]}/{test}.in{i}.bin"), v) for i in range(4)],
             vec(rd(f"{src[r]}/{test}.out0.0.bin"), v)) for r, v in picked], out


def pick_m2(test):
    dep = rd(f"{DEP2IN}/{test}.depth.bin")
    picked = [("dep2", v) for v in [v for v in range(len(dep)) if dep[v] == 255][:2]]
    picked += [("dep2", v) for v in [v for v in range(len(dep)) if dep[v] != 255][:4]]
    # rand2: every vector holding a lane whose members are all -0 and GB300 gives +0, without
    # an infinite or NaN member (which NaN GB300 returns is not modeled: the check refuses).
    ins = [rd(f"{RAND2}/{test}.in{i}.bin") for i in range(2)]
    out = rd(f"{RAND2}/{test}.out0.0.bin")
    top = 0x7F80 if test.startswith("bf16") else 0x7C00
    for v in range(len(out) // 16):
        special = any(h & top == top for b in ins for h in lanes(b, v))
        if not special and any(all(lanes(b, v)[l] == 0x8000 for b in ins) and lanes(out, v)[l] == 0 for l in range(8)):
            picked.append(("rand2", v))
    src = {"dep2": DEP2, "rand2": RAND2}
    return [(r, v, [vec(rd(f"{src[r]}/{test}.in{i}.bin"), v) for i in range(2)],
             vec(rd(f"{src[r]}/{test}.out0.0.bin"), v)) for r, v in picked]


os.makedirs(f"{OUT}/data", exist_ok=True)
os.makedirs(f"{OUT}/expected", exist_ok=True)
index = ["members,test,vector,source run,source vector"]
for nm, pick in ((4, lambda t: pick_m4(t)[0]), (2, pick_m2)):
    for test in TESTS:
        rows = pick(test)
        for i in range(nm):
            with open(f"{OUT}/data/m{nm}.{test}.in{i}.bin", "wb") as f:
                f.write(b"".join(r[2][i] for r in rows))
        with open(f"{OUT}/expected/m{nm}.{test}.out.bin", "wb") as f:
            f.write(b"".join(r[3] for r in rows))
        index += [f"{nm},{test},{k},{r[0]},{r[1]}" for k, r in enumerate(rows)]
        print(f"m{nm} {test}: {len(rows)} vectors")
open(f"{OUT}/expected/vectors.csv", "w").write("\n".join(index) + "\n")

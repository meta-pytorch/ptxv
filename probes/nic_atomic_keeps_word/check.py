#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 257 repro (run.sh): NIC atomics and the numerics shadow.

cas_fail/, fa0/, fa1/ (one GPU, a loopback IBGDA queue pair, thread 0): dst[i] = rn(A_i + B_i)
for (A, B) = (0.1, 0.2), (1, 1e-8), (3, 0.1), (1e8, 1); slot 0 RDMA-writes into dst2, slot 1 is
the NIC atomic on dst's first 8-byte word (a compare-and-swap that fails, a fetch-add of 0, a
fetch-add of 1), the fetched value to sink; after the completion out = dst, dst2, sink, src.
Expected at out[0], out[1] (the two words of the atomic's target), per front: a claim equal to
the word's exact error |rn(A + B) - (A + B)|, or no claim; for fa1's out[0] (the low word the
add changed, which held a rounding) no claim. Until MISS 257 the NIC atomic dropped both words'
nodes without taint, so `ptxv num` took the rounded bits as exact: abs_err 0 and bound 0.
arith/: the payload alone, measured on GB300 (expected/arith.gb300.txt); with PTXV_GPUS set it
is replayed (`ptxv hw --runs 20 --scrub`). Exit 1 while a claim is refuted, 2 on a tool failure,
3 when GB300 no longer matches the simulator. usage: check.py PTXV"""
import os
import re
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
PTXV = sys.argv[1]
f32 = lambda v: struct.unpack("<f", struct.pack("<f", v))[0]
AB = [(0.1, 0.2), (1.0, 1e-8)]
ERR = []
for a, b in AB:
    ex = Fraction(f32(a)) + Fraction(f32(b))
    ERR.append(float(abs(Fraction(f32(float(ex))) - ex)))
# expected per case: out[0], out[1] -> exact error, or None (no claim expected)
EXPECT = {"cas_fail": [ERR[0], ERR[1]], "fa0": [ERR[0], ERR[1]], "fa1": [None, ERR[1]]}
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PTXV_THREADS=os.environ.get("PTXV_THREADS", "4"))
tmp = tempfile.mkdtemp(prefix="nic_atomic_keeps_word.")
rc = 0
print(f"{'case':9} {'front':5} {'i':2} {'abs_err':>12} {'bound':>12} {'exact error':>12} verdict")
for case, exp in EXPECT.items():
    out = os.path.join(tmp, case)
    p = subprocess.run([PTXV, "num", os.path.join(HERE, case), "--out", out], env=env, capture_output=True, text=True)
    for front in ("ptx", "sass"):
        csv = os.path.join(out, f"good.{front}.num.csv")
        if not os.path.exists(csv):
            print(f"{case}: no {front} CSV (rc {p.returncode})\n{(p.stdout + p.stderr)[-1500:]}")
            sys.exit(2)
        rows = [l.split(",") for l in open(csv).read().splitlines()]
        hdr, rows = rows[0], {int(r[0]): dict(zip(rows[0], r)) for r in rows[1:]}
        for i, e in enumerate(exp):
            r = rows[i]
            claimed = r["untracked"] == "false" and r["nonfinite"] == "false"
            ae = float(r["abs_err"])
            b = r["bound_cor1a"]
            bv = float("nan") if b == "unknown" else float(b)
            if e is None:
                bad = claimed
                why = "claimed (an integer-modified rounded word: no claim expected)" if bad else "ok (no claim)"
            elif not claimed:
                bad, why = False, "ok (no claim)"
            else:
                bad = abs(ae - e) > 1e-6 * e or (bv == bv and bv < e)
                why = "WRONG (claim below the exact error)" if bad else "ok"
            rc |= 1 if bad else 0
            print(f"{case:9} {front:5} {i:2} {ae:12.4e} {b:>12} {('-' if e is None else format(e, '.4e')):>12} {why}")
if os.environ.get("PTXV_GPUS"):
    hw_env = dict(os.environ)
    hw_env.pop("CUDA_VISIBLE_DEVICES", None)
    p = subprocess.run([PTXV, "hw", os.path.join(HERE, "arith"), "--runs", "20", "--scrub", "--out", os.path.join(tmp, "hw")],
                       env=hw_env, capture_output=True, text=True)
    line = next((l for l in p.stdout.splitlines() if l.startswith("hw ")), (p.stdout + p.stderr).strip()[-300:])
    want = open(os.path.join(HERE, "expected", "arith.gb300.txt")).read()
    same = ("matches simulator bitwise" in line) == ("matches simulator bitwise" in want)
    print(f"\nGB300 arith/ now: {line}\nrecorded: {want.strip()}")
    if not same:
        rc = 3
print("\nexit", rc)
sys.exit(rc)

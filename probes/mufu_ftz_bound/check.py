#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MUFU bound repro (README.md, run.sh): `ptxv num` of r5_mufu next to GB300's recorded output
(expected/r5_mufu.out.bin, `ptxv hw --runs 20 --scrub`, 20 of 20 runs bitwise equal).

out[32 g + t] = f(x[t]): g = 0 rcp.approx.ftz.f32, 1 rcp.approx.f32, 2 ex2.approx.ftz.f32,
3 ex2.approx.f32. The true error of GB300's value is |GB300 - f(x)|: 1/x exactly (fractions),
2^x = exp(x ln 2) in 100-digit decimal arithmetic (relative slack 10^-90). An element whose
claimed bound (bound_cor1a or bound_running; a bound printed as unknown is no claim) lies below
its true error is under-claimed; an element whose two bounds are both unknown is counted apart.
GROUPS selects the instructions this probe is about (argument 2). With PTXV_GPUS set the kernel
is also replayed on GB300 (`ptxv hw --runs 5 --scrub`) and compared with expected/. Exit 1 while
an element of GROUPS is under-claimed, 2 on a tool failure, 3 when GB300 no longer gives the
recorded output. usage: check.py PTXV GROUPS (e.g. 1, or 0,2)"""
import csv
import decimal
import glob
import math
import os
import struct
import subprocess
import sys
import tempfile
from decimal import Decimal
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
PTXV = sys.argv[1]
GROUPS = [int(g) for g in sys.argv[2].split(",")]
OPS = ["rcp.approx.ftz.f32", "rcp.approx.f32", "ex2.approx.ftz.f32", "ex2.approx.f32"]
CASE = os.path.join(HERE, "r5_mufu")
xb = struct.unpack("<32I", open(os.path.join(CASE, "data", "x.bin"), "rb").read()[:128])
gb = struct.unpack("<128I", open(os.path.join(HERE, "expected", "r5_mufu.out.bin"), "rb").read()[:512])
f32 = lambda b: struct.unpack("<f", struct.pack("<I", b))[0]


def true_err(i):
    g, t = divmod(i, 32)
    x, hw = f32(xb[t]), f32(gb[i])
    if not math.isfinite(hw) or (g < 2 and x == 0):
        return None
    if g < 2:
        return abs(Fraction(hw) - 1 / Fraction(x)), Fraction(0)
    with decimal.localcontext() as c:
        c.prec = 100
        ex = (Decimal(x) * Decimal(2).ln()).exp()
        return abs(Fraction(hw) - Fraction(ex)), Fraction(ex) / 10**90


def num_float(s):
    try:
        return float(s)
    except ValueError:
        return None  # "unknown" or empty: no claim


tmp = tempfile.mkdtemp(prefix="mufu_bound.")
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PTXV_THREADS=os.environ.get("PTXV_THREADS", "4"))
p = subprocess.run([PTXV, "num", CASE, "--out", tmp], env=env, capture_output=True, text=True)
rc = 0
print(f"ptxv num r5_mufu against GB300 (instructions: {', '.join(OPS[g] for g in GROUPS)})")
print(f"{'front':5} {'out':>4} {'instruction':19} {'x':>10} {'GB300':>10} {'|GB300 - f(x)|':>14} "
      f"{'bound_cor1a':>12} {'bound_running':>13} verdict")
for front in ("ptx", "sass"):
    paths = glob.glob(os.path.join(tmp, "**", f"good.{front}.num.csv"), recursive=True)
    if not paths:
        print(f"{front}: no num csv (rc {p.returncode})\n{(p.stdout + p.stderr)[-2000:]}")
        sys.exit(2)
    rows = {int(r["index"]): r for r in csv.DictReader(open(paths[0]))}
    claimed = under = unknown = 0
    listed = []
    for i in range(128):
        g, t = divmod(i, 32)
        r, te = rows.get(i), true_err(i)
        if g not in GROUPS or r is None or te is None:
            continue
        if r["untracked"] != "false" or r["nonfinite"] != "false":
            continue
        te, slack = te
        b1, b2 = num_float(r["bound_cor1a"]), num_float(r["bound_running"])
        if b1 is None and b2 is None:
            unknown += 1
            continue
        claimed += 1
        bad = [b for b in (b1, b2) if b is not None and Fraction(b) < te - slack]
        if bad:
            under += 1
            listed.append((te, i, g, t, b1, b2))
    fmt = lambda b: "unknown" if b is None else f"{b:.4e}"
    for te, i, g, t, b1, b2 in sorted(listed, reverse=True)[:12]:
        print(f"{front:5} {i:>4} {OPS[g]:19} {xb[t]:#010x} {gb[i]:#010x} {float(te):>14.4e} "
              f"{fmt(b1):>12} {fmt(b2):>13} UNDER-CLAIMED")
    rc |= 1 if under else 0
    more = " (the 12 largest errors listed)" if under > 12 else ""
    apart = f"; {unknown} with both bounds unknown (no claim)" if unknown else ""
    print(f"{front:5} {claimed} claimed elements, {under} under-claimed{more}{apart}")

if os.environ.get("PTXV_GPUS"):
    hw_env = dict(os.environ)
    hw_env.pop("CUDA_VISIBLE_DEVICES", None)
    dump = os.path.join(tmp, "hwdump")
    q = subprocess.run([PTXV, "hw", CASE, "--runs", "5", "--scrub", "--out", os.path.join(tmp, "hw")],
                       env=dict(hw_env, PTXV_HW_DUMP=dump), capture_output=True, text=True)
    got = sorted(glob.glob(os.path.join(dump, "hw_*.bin")))
    exp = open(os.path.join(HERE, "expected", "r5_mufu.out.bin"), "rb").read()
    same = len(got) == 1 and open(got[0], "rb").read() == exp
    print(f"GB300 now (PTXV_GPUS={os.environ['PTXV_GPUS']}): "
          + ("equal to expected/" if same else f"DIFFERS from expected/ ({len(got)} distinct outputs, rc {q.returncode})"))
    if not same:
        rc = 3
print("exit", rc)
sys.exit(rc)

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 258 repro (run.sh): MUFU results whose true value is not real.

sqrt_negsub/: out = sqrt.approx.ftz.f32(x); lg2_ex2_negsub/: out = ex2.approx.ftz(lg2.approx.ftz(x)).
x[i % 8 < 3] are negative subnormals (0x80000003, 0x807fffff, 0x80000100), which the .ftz forms
flush to -0 (GB300: sqrt -0, ex2(lg2) 0); sqrt and lg2 of a negative number have no real value,
so N-NUM-NONFINITE leaves those elements unanalyzed. Counts, per case and front, the
negative-subnormal lanes `ptxv num` analyzes (untracked=false and nonfinite=false). With
PTXV_GPUS set both cases are replayed on GB300 (`ptxv hw --runs 20 --scrub`). Exit 1 while such
a lane is analyzed, 2 on a tool failure, 3 when GB300 no longer matches. usage: check.py PTXV"""
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PTXV = sys.argv[1]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PTXV_THREADS=os.environ.get("PTXV_THREADS", "4"))
tmp = tempfile.mkdtemp(prefix="mufu_nan_exact.")
rc = 0
for case in ("sqrt_negsub", "lg2_ex2_negsub"):
    xs = struct.unpack("<32I", open(os.path.join(HERE, case, "x.bin"), "rb").read())
    out = os.path.join(tmp, case)
    p = subprocess.run([PTXV, "num", os.path.join(HERE, case), "--out", out], env=env, capture_output=True, text=True)
    for front in ("ptx", "sass"):
        csv = os.path.join(out, f"good.{front}.num.csv")
        if not os.path.exists(csv):
            print(f"{case}: no {front} CSV (rc {p.returncode})\n{(p.stdout + p.stderr)[-1500:]}")
            sys.exit(2)
        lines = open(csv).read().splitlines()
        hdr = lines[0].split(",")
        n_bad = 0
        for ln in lines[1:]:
            r = dict(zip(hdr, ln.split(",")))
            i = int(r["index"])
            if xs[i] >> 31 and (xs[i] & 0x7FFFFFFF) < 0x00800000:
                analyzed = r["untracked"] == "false" and r["nonfinite"] == "false"
                n_bad += analyzed
                if analyzed and i < 8:
                    print(f"{case} {front} x=0x{xs[i]:08x}: hw {r['hw']} exact {r['exact']} abs_err {r['abs_err']} bounds {r['bound_cor1a']} / {r['bound_running']} untracked={r['untracked']} nonfinite={r['nonfinite']}  ANALYZED")
        print(f"{case} {front}: {n_bad} negative-subnormal lanes analyzed")
        rc |= 1 if n_bad else 0
if os.environ.get("PTXV_GPUS"):
    hw_env = dict(os.environ)
    hw_env.pop("CUDA_VISIBLE_DEVICES", None)
    for case in ("sqrt_negsub", "lg2_ex2_negsub"):
        p = subprocess.run([PTXV, "hw", os.path.join(HERE, case), "--runs", "20", "--scrub", "--out", os.path.join(tmp, "hw." + case)],
                           env=hw_env, capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("hw ")), (p.stdout + p.stderr).strip()[-300:])
        print(f"GB300 {case}: {line}")
        if "matches simulator bitwise" not in line:
            rc = 3
print("exit", rc)
sys.exit(rc)

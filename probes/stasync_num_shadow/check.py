#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 205 repro: st.async writes and the numerics shadow (README.md, run.sh).

ptxv's claims next to GB300's recorded outputs (expected/<case>.out.bin, `ptxv hw --runs 20
--scrub`, 20 of 20 runs bitwise equal):
- `ptxv equiv` of each reference against its candidate at the captured input (w = x), both
  fronts. A level other than "none" says both compute the same function; GB300 refutes it where
  the two outputs differ at either input (w = x or w = x + 1).
- `ptxv num` of r5_stasync_co_c1: out[t] = add.rn(t, 0.1f) stored by st.async over the input
  x[t] (the same bits). A claimed abs_err (or bound) below |GB300 - (t + 0.1f)| under-claims.
With PTXV_GPUS set, every case is also replayed on GB300 (`ptxv hw --runs 5 --scrub`) and its
output compared with expected/.
Exit 1 while a claim is refuted, 2 on a tool failure, 3 when GB300 no longer gives the recorded
output. usage: check.py PTXV"""
import csv
import glob
import os
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
PTXV = sys.argv[1]
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", PTXV_THREADS=os.environ.get("PTXV_THREADS", "4"))
PAIRS = [
    ("c1_ref_stasyncw", "cand_copy", "local st.async of w over S[t] = x; out = S[t]"),
    ("c2_ref_rsaw", "c2_cand_copy", "remote st.async of w over the peer's S[t] = x"),
    ("c2_ref_rsaz", "c2_cand_zero", "remote st.async of rn(w - x) into the peer's S[t]"),
    ("c1_ref_stasync0", "cand_copy", "local st.async of 0 over S[t] = x (MISS 177 node check)"),
]


def outs(case):
    b = open(os.path.join(HERE, "expected", case + ".out.bin"), "rb").read()
    return struct.unpack("<%dI" % (len(b) // 4), b)


def ndiff(a, b):
    return sum(x != y for x, y in zip(outs(a), outs(b))), len(outs(a))


def run(args, out):
    p = subprocess.run([PTXV] + args + ["--out", out], cwd=HERE, env=ENV, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


rc = 0
tmp = tempfile.mkdtemp(prefix="stasync_num_shadow.")
print("equiv at the captured input (w = x); GB300 outputs of reference vs candidate at w = x and w = x + 1")
print(f"{'reference / candidate':32} {'front':5} {'ptxv equiv':20} {'GB300 w = x':16} {'GB300 w = x + 1':16} verdict")
for ref, cand, what in PAIRS:
    d0, n = ndiff(ref, cand)
    d1, _ = ndiff(ref + "_plus1", cand + "_plus1")
    for front in ("ptx", "sass"):
        code, log = run(["equiv", f"eq_{ref}_{front}.toml", "--no-hw"], os.path.join(tmp, f"{ref}.{front}"))
        lv = [ln.split("strongest level:", 1)[1].strip() for ln in log.splitlines() if "strongest level:" in ln]
        if not lv:
            print(f"{ref} {front}: no level (rc {code})\n{log[-2000:]}")
            sys.exit(2)
        level = lv[-1]
        refuted = not level.startswith("none") and (d0 or d1)
        rc |= 1 if refuted else 0
        verdict = "REFUTED (claims the same function)" if refuted else "ok"
        print(f"{ref + ' / ' + cand:32} {front:5} {level:20} {f'{d0} of {n} differ':16} {f'{d1} of {n} differ':16} {verdict}")
print("  (" + "; ".join(f"{r}: {w}" for r, _, w in PAIRS) + ")")

# num: out[t] = add.rn(t, 0.1f), exact t + 0.1f.
gpu = outs("r5_stasync_co_c1")
tenth = Fraction(struct.unpack("<f", struct.pack("<I", 0x3DCCCCCD))[0])
code, log = run(["num", "r5_stasync_co_c1"], os.path.join(tmp, "num"))
print("\nnum r5_stasync_co_c1: out[t] = add.rn(t, 0.1f) by st.async over x[t] (same bits); exact t + 0.1f")
for front in ("ptx", "sass"):
    paths = glob.glob(os.path.join(tmp, "num", "**", f"good.{front}.num.csv"), recursive=True)
    if not paths:
        print(f"{front}: no num csv (rc {code})\n{log[-2000:]}")
        sys.exit(2)
    under, worst, claimed = 0, (Fraction(0), None, None), 0
    for row in csv.DictReader(open(paths[0])):
        i = int(row["index"])
        if row["untracked"] != "false" or row["nonfinite"] != "false":
            continue
        claimed += 1
        hw = Fraction(struct.unpack("<f", struct.pack("<I", gpu[i]))[0])
        te = abs(hw - (i + tenth))
        ae, b1 = Fraction(float(row["abs_err"])), Fraction(float(row["bound_cor1a"]))
        slack = Fraction(1, 10**9)
        if ae * (1 + slack) < te or b1 * (1 + slack) < te:
            under += 1
            if te > worst[0]:
                worst = (te, i, (float(ae), float(b1)))
    rc |= 1 if under else 0
    if under:
        te, i, (ae, b1) = worst
        print(f"  {front}: {claimed} elements claimed; GB300 error above the claim at {under}: worst out[{i}] "
              f"|GB300 - exact| {float(te):.4e} vs claimed abs_err {ae:.4e}, bound_cor1a {b1:.4e}: UNDER-CLAIMED")
    else:
        print(f"  {front}: {claimed} elements claimed; every claim covers GB300's error: ok")

if os.environ.get("PTXV_GPUS"):
    print("\nGB300 now (ptxv hw --runs 5 --scrub, PTXV_GPUS=%s):" % os.environ["PTXV_GPUS"])
    hw_env = dict(os.environ)
    hw_env.pop("CUDA_VISIBLE_DEVICES", None)
    for exp in sorted(glob.glob(os.path.join(HERE, "expected", "*.out.bin"))):
        case = os.path.basename(exp)[: -len(".out.bin")]
        dump = os.path.join(tmp, "hw", case)
        p = subprocess.run([PTXV, "hw", case, "--runs", "5", "--scrub", "--out", os.path.join(tmp, "hwout", case)],
                           cwd=HERE, env=dict(hw_env, PTXV_HW_DUMP=dump), capture_output=True, text=True)
        got = sorted(glob.glob(os.path.join(dump, "hw_*.bin")))
        same = len(got) == 1 and open(got[0], "rb").read() == open(exp, "rb").read()
        print(f"  {case}: {'equal to expected/' if same else 'DIFFERS from expected/ (%d distinct outputs, rc %d)' % (len(got), p.returncode)}")
        if not same:
            rc = 3
print("\nexit", rc, "(1: a claim GB300 refutes; 0: none)" if rc in (0, 1) else "")
sys.exit(rc)

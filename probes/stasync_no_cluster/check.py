#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 209 repro (README.md, run.sh): st.async in a launch without cluster dimensions.

ref_stasync0 / ref_stasyncw: an st.async into the CTA's own shared memory (mbarrier complete_tx),
launched without cluster dimensions and without .reqnctapercluster. GB300 (`ptxv hw --runs 20
--scrub`): "an illegal instruction was encountered", no run completes (expected/*.gb300.txt).
c1_ref_stasync0 / c1_ref_stasyncw: the same kernels with .reqnctapercluster 1, 1, 1 and
cluster = [1, 1, 1]: GB300 runs them and matches the simulator bitwise (20 of 20 runs).
Prints the verdict of `ptxv check` and `ptxv check --certify` on both fronts next to GB300's
result, and exits 1 while a case GB300 traps on gets no fault (2 on a tool failure). With
PTXV_GPUS set every case is also run on GB300 (`ptxv hw --runs 3`); exit 3 when GB300's result
changed. usage: check.py PTXV"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PTXV = sys.argv[1]
CASES = ["ref_stasync0", "ref_stasyncw", "c1_ref_stasync0", "c1_ref_stasyncw"]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PTXV_THREADS=os.environ.get("PTXV_THREADS", "4"))
tmp = tempfile.mkdtemp(prefix="stasync_no_cluster.")


def gb300(case):
    line = open(os.path.join(HERE, "expected", case + ".gb300.txt")).read()
    return "illegal instruction" if "illegal instruction" in line else ("runs, matches the simulator" if "matches simulator" in line else line.strip())


def verdicts(case, mode):
    args = [PTXV, "check", case, "--variant", "good", "--no-input-coverage", "--out", os.path.join(tmp, case + mode)]
    if mode:
        args.append(mode)
    p = subprocess.run(args, cwd=HERE, env=env, capture_output=True, text=True)
    got = {m.group(2): m.group(3) for m in re.finditer(r"^(PASS|FAIL) +good +(ptx|sass) +\[([^]]*)\]", p.stdout, re.M)}
    if not got:
        print(f"{case} {mode}: no verdict (rc {p.returncode})\n{(p.stdout + p.stderr)[-2000:]}")
        sys.exit(2)
    return got


rc = 0
print(f"{'case':16} {'GB300':28} {'front':5} {'ptxv check':22} {'ptxv check --certify':22} verdict")
for case in CASES:
    g = gb300(case)
    exh, cert = verdicts(case, ""), verdicts(case, "--certify")
    for front in ("ptx", "sass"):
        e, c = exh.get(front, "?"), cert.get(front, "?")
        traps = g == "illegal instruction"
        bad = traps and not ("fault" in e.split(",") and "fault" in c.split(","))
        bad |= not traps and (e != "clean" or c != "clean")
        rc |= 1 if bad else 0
        mark = ("MISSED (no fault)" if traps else "differs") if bad else "ok"
        print(f"{case:16} {g:28} {front:5} {e:22} {c:22} {mark}")

if os.environ.get("PTXV_GPUS"):
    print("\nGB300 now (ptxv hw --runs 3):")
    hw_env = dict(os.environ)
    hw_env.pop("CUDA_VISIBLE_DEVICES", None)
    for case in CASES:
        p = subprocess.run([PTXV, "hw", case, "--runs", "3", "--out", os.path.join(tmp, "hw." + case)],
                           cwd=HERE, env=hw_env, capture_output=True, text=True)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("hw ")), (p.stdout + p.stderr).strip()[-300:])
        now = "illegal instruction" if "illegal instruction" in line else ("runs, matches the simulator" if "matches simulator" in line else line)
        same = now == gb300(case)
        print(f"  {case}: {now}{'' if same else '  (recorded: ' + gb300(case) + ')'}")
        if not same:
            rc = 3
print("\nexit", rc)
sys.exit(rc)

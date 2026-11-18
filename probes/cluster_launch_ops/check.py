#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MISS 209 calibration (run.sh): cluster-scope operations with and without cluster dimensions.

Each case (gen.py) is one cluster-scope operation; variant cluster1 declares .reqnctapercluster
1, 1, 1, variant none no cluster dimensions; stasync_attr1 (launch attribute 1 x 1 x 1) and
stasync_explicit (.explicitcluster) have one variant, good. GB300 (`ptxv probe`,
expected/<case>.gb300.txt) traps with an illegal instruction where the variant has no
expected/<case>.<variant>.out.bin, and runs otherwise. Runs `ptxv check` on every case the
simulator lifts (sregs: the SASS front only, the special registers it reads are SASS's) and
compares its verdict with GB300's: a trap must be a fault, a run must be clean. Exit 1 while one
differs, 2 on a tool failure. usage: check.py PTXV"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PTXV = sys.argv[1]
OPS = ["stasync", "stasync_v2", "bulk_s2s", "st_cluster", "ld_cluster", "atom_cluster", "mapa_cta",
       "barrier_cluster", "fence_cluster", "wait_cluster", "mbar_remote_arrive",
       "mbar_remote_expect_tx", "bulk_g2s", "clc_cancel", "clc_cancel_mc"]
CASES = [(op, v, ("ptx", "sass")) for op in OPS for v in ("cluster1", "none")]
CASES += [("sregs", v, ("sass",)) for v in ("cluster1", "none")]
CASES += [("stasync_attr1", "good", ("ptx", "sass")), ("stasync_explicit", "good", ("ptx", "sass"))]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PTXV_THREADS=os.environ.get("PTXV_THREADS", "4"))
tmp = tempfile.mkdtemp(prefix="cluster_launch_ops.")
rc = 0
print(f"{'case':22} {'variant':9} {'front':5} {'GB300':6} {'ptxv check':14} verdict")
for op, v, fronts in CASES:
    front_arg = ["--front", fronts[0]] if len(fronts) == 1 else []
    p = subprocess.run([PTXV, "check", op, "--variant", v, *front_arg, "--no-input-coverage", "--out",
                        os.path.join(tmp, f"{op}.{v}")], cwd=HERE, env=env, capture_output=True, text=True)
    got = {m.group(1): m.group(2) for m in re.finditer(rf"^(?:PASS|FAIL) +{re.escape(v)} +(ptx|sass) +\[([^]]*)\]", p.stdout, re.M)}
    traps = not os.path.exists(os.path.join(HERE, "expected", f"{op}.{v}.out.bin"))
    for front in fronts:
        if front not in got:
            print(f"{op} {v} {front}: no verdict (rc {p.returncode})\n{(p.stdout + p.stderr)[-2000:]}")
            sys.exit(2)
        classes = got[front].split(",")
        bad = ("fault" not in classes) if traps else (classes != ["clean"])
        rc |= 1 if bad else 0
        print(f"{op:22} {v:9} {front:5} {'trap' if traps else 'runs':6} {got[front]:14} {'DIFFERS' if bad else 'ok'}")
print("\nexit", rc)
sys.exit(rc)

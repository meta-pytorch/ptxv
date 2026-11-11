#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run every instantiated litmus config under several stress settings on GPUs 0,1,3 ($GPUS;
never a GPU in $PTXV_EXCLUDE_GPUS).
Usage: sweep.py <outfile> [iters_scale] [config-regex, matched against "TEST mem group wf rf"]"""

import concurrent.futures as cf
import itertools
import os
import re
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
TN = [
    "MP",
    "MPS",
    "SB",
    "LB",
    "CORR",
    "COWR",
    "IRIW",
    "TEAR32",
    "TEAR64",
    "TEAR128",
    "MPV",
    "MPG",
    "MPL1",
    "MPGN",
]
TM = {"T_" + n: n for n in TN}
MM = {
    "M_SHV": "shv",
    "M_SHW": "shw",
    "M_GLW": "glw",
    "M_GLS": "gls",
    "M_GLV": "glv",
    "M_DSM": "dsm",
    "M_DSR": "dsr",
}
GM = {"G_CTA": "cta", "G_CLU": "clu"}
src = open(os.path.join(HERE, "litmus.cu")).read()
cfgs = [
    (TM[t], MM[m], GM[g], wf, rf)
    for t, m, g, wf, rf in re.findall(
        r"K\((T_\w+), (M_\w+), (G_\w+), (\d), (\d)\)", src
    )
    if t in TM
]
scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
if len(sys.argv) > 3:
    cfgs = [c for c in cfgs if re.search(sys.argv[3], " ".join(c))]
SETTINGS = [  # name, args
    ("rand", "nw=16 dmask=63 smask=63 iters=1000000"),
    ("tight", "nw=16 dmask=0 smask=0 iters=500000"),
    ("wide", "nw=4 dmask=511 smask=0 iters=300000"),
    ("fixed4", "nw=4 fixed=1 dmask=31 smask=0 iters=500000"),
    ("bank", "nw=32 dmask=63 smask=255 is=32 lsep=1024 iters=300000"),
    ("samewarp", "nw=16 same=1 dmask=31 smask=63 iters=300000"),
]
_excluded = set(filter(None, os.environ.get("PTXV_EXCLUDE_GPUS", "").split(",")))
gpus = [
    int(g) for g in os.environ.get("GPUS", "0,1,3").split(",") if g not in _excluded
]
lock = threading.Lock()
free = list(gpus)
cv = threading.Condition(lock)


def run(job):
    (t, m, g, wf, rf), (sn, args) = job
    if sn == "samewarp" and g == "clu":
        return None
    a = args.split()
    a = [
        re.sub(
            r"iters=(\d+)",
            lambda x: f"iters={max(1000, int(int(x.group(1)) * scale * (0.5 if g == 'clu' else 1)))}",
            s,
        )
        for s in a
    ]
    if t.startswith("TEAR"):
        a = [s for s in a if not s.startswith("is=")] + ["is=4"]
    if t == "MPV":
        a = [s for s in a if not s.startswith("is=")] + [
            "is=32" if sn == "bank" else "is=2"
        ]
    with cv:
        while not free:
            cv.wait()
        gpu = free.pop()
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        p = subprocess.run(
            [
                "timeout",
                "600",
                os.path.join(HERE, os.environ.get("BIN", "litmus")),
                t,
                m,
                g,
                wf,
                rf,
                f"seed={hash(sn) & 0xFFFF}",
            ]
            + a,
            env=env,
            capture_output=True,
            text=True,
        )
        line = p.stdout.strip() or ("ERROR rc=%d %s" % (p.returncode, p.stderr.strip()))
    finally:
        with cv:
            free.append(gpu)
            cv.notify()
    return f"[{sn}] {line}"


jobs = list(itertools.product(cfgs, SETTINGS))
with open(sys.argv[1], "w") as f, cf.ThreadPoolExecutor(len(gpus)) as ex:
    for r in ex.map(run, jobs):
        if r:
            print(r, flush=True)
            f.write(r + "\n")
            f.flush()

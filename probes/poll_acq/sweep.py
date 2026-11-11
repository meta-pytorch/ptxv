#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run the poll_acq litmus matrix on GB300: cross-GPU runs on GPUs 0+1, intra-GPU runs on GPU 3
(never a GPU in $PTXV_EXCLUDE_GPUS). Each config is calibrated with a short run, then run for about --secs seconds.
A run waits while a process this script did not start is on one of its GPUs.

Usage: sweep.py <plan> <outfile> [--secs S] [--only REGEX]
  plan: smoke | refute | deepep | cumul | controls | all
Output lines: "[tag] RESULT ...", tag = plan/config name/gpus."""

import argparse
import concurrent.futures as cf
import itertools
import os
import random
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("BIN", os.path.join(HERE, "pollacq"))
CROSS = [
    ("0,1", "cdev=0 pdev=1"),
    ("0,1", "cdev=1 pdev=0"),
]  # (CUDA_VISIBLE_DEVICES, args)
INTRA = [("3", "cdev=0 pdev=0")]
GPU_BUS = {}  # physical index -> bus id


def bus_ids():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout
    for line in out.strip().splitlines():
        i, b = [x.strip() for x in line.split(",")]
        GPU_BUS[int(i)] = b.lower()


mine = set()
mine_lock = threading.Lock()


def foreign_on(gpus):
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout
    busy = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        b, pid = [x.strip() for x in line.split(",")]
        for g in gpus:
            if GPU_BUS[g].endswith(b.lower()[-12:]) and int(pid) not in mine:
                busy.append((g, pid))
    return busy


def run1(vis, args, timeout):
    gpus = [int(g) for g in vis.split(",")]
    excluded = set(filter(None, os.environ.get("PTXV_EXCLUDE_GPUS", "").split(",")))
    assert not {str(g) for g in gpus} & excluded, "GPU in PTXV_EXCLUDE_GPUS"
    while True:
        busy = foreign_on(gpus)
        if not busy:
            break
        print(f"# waiting: foreign processes {busy}", file=sys.stderr, flush=True)
        time.sleep(10)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=vis)
    p = subprocess.Popen(
        ["timeout", str(timeout), BIN] + args.split(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    with mine_lock:
        mine.add(p.pid)
    out, err = p.communicate()
    with mine_lock:
        mine.discard(p.pid)
    line = next((ln for ln in out.splitlines() if ln.startswith("RESULT")), None)
    return line or f"ERROR rc={p.returncode} {err.strip()[:300]}"


def secs_of(line):
    m = re.search(r"secs=([\d.]+)", line)
    return float(m.group(1)) if m else None


def run_cfg(job, target):
    tag, vis, args = job
    calib = run1(vis, args + " iters=4000", 300)
    s = secs_of(calib)
    if s is None:
        return f"[{tag}/gpu{vis}] {calib}"
    iters = int(min(2_000_000_000, max(4000, 4000 * target / max(s, 0.01))))
    line = run1(vis, args + f" iters={iters}", int(target * 4 + 120))
    return f"[{tag}/gpu{vis}] {line}"


# ------------------------------------------------------------------ plans
STRESS = {
    "none": "stress=0",
    "l2": "stress=4 smode=1",
    "atom": "stress=4 smode=2",
    "nvl": "stress=4 smode=3",
    "l1": "stress=4 smode=4",
    "payld": "stress=4 smode=5",
}


def placements(cross_only=False):
    out = [
        ("x01", *CROSS[0], ""),
        ("x01-loc1", *CROSS[0], "loc=1"),
        ("x10", *CROSS[1], ""),
    ]
    if not cross_only:
        out += [
            ("i-far", *INTRA[0], "pp=0"),
            ("i-near", *INTRA[0], "pp=1"),
            ("i-same", *INTRA[0], "pp=2"),
        ]
    return out


def pairs_for(pl):
    return {"pp=0": 76, "pp=1": 76, "pp=2": 148}.get(pl.split()[0] if pl else "", 148)


def varied(rng):
    """Per-config random choices of the dimensions not swept explicitly."""
    return " ".join(
        [
            f"dmask={rng.choice([0, 63, 255, 4095])}",
            f"nf={rng.choice([1, 2, 2, 8, 32])}",
            f"ps={rng.choice([4, 32, 32])}",
            f"W={rng.choice([2, 4, 4])}",
            f"seed={rng.randrange(1 << 16)}",
        ]
    )


def plan(name):  # noqa: C901 (flat table of probe plans, one branch per plan name)
    rng = random.Random(hash(name) & 0xFFFF if name != "all" else 1)
    jobs = []

    def add(tag, test, form, rd, pl, extra, opts):
        plname, vis, devargs, plargs = pl
        intra = "pp=" in plargs
        if extra.startswith("stress") and "smode=3" in extra and intra:
            extra = extra  # NVLink stress in an intra run: peer scratch = local scratch (L2 traffic)
        p = pairs_for(plargs) if intra else 148
        args = f"{test} {form} {rd} {devargs} {plargs} pairs={p} {opts} {extra} {varied(rng)}"
        jobs.append((f"{name}/{tag}/{plname}", vis, re.sub(r"\s+", " ", args).strip()))

    if name == "smoke":
        for pl in placements():
            add("relay-deepep", "relay", 0, 2, pl, "", "wf=1 pr=3")
    if name in ("refute", "all"):
        # E-POLL-ACQ as stated: weak reads after leaving the poll loop, payload line L1-valid
        for pl in placements():
            for form, rd, pr in itertools.product([0, 1, 2], [0, 1, 4], [1, 2, 3]):
                if rd == 4 and form != 0:
                    continue
                for st in ["none", "l1"]:
                    add(
                        f"poll-f{form}-rd{rd}-pr{pr}-{st}",
                        "poll",
                        form,
                        rd,
                        pl,
                        STRESS[st],
                        f"wf={rng.choice([0, 1])} pr={pr}",
                    )
            # DeepEP's entry fence, but other warps of the SM keep loading the payload lines
            for form in [0, 2]:
                add(
                    f"poll-f{form}-rd0-pr0-payld",
                    "poll",
                    form,
                    0,
                    pl,
                    STRESS["payld"],
                    "wf=0 pr=0",
                )
    if name in ("deepep", "all"):
        # barrier_block as DeepEP compiles it: entry fence.acq_rel.sys, ATOM.E.ADD.STRONG.SYS signal,
        # LDG.E.STRONG.SYS + VOTE.ALL poll, (bar.sync), weak LD.E reads
        for pl in placements():
            for st in ["none", "l2", "atom", "nvl", "l1"]:
                for form in [0, 2]:
                    add(
                        f"bb-f{form}-{st}", "poll", form, 0, pl, STRESS[st], "wf=0 pr=0"
                    )
                # volatile (STRONG.SYS) reads after a relaxed warp poll, payload L1-valid
                add(
                    f"poll-f0-rd2-pr3-{st}",
                    "poll",
                    0,
                    2,
                    pl,
                    STRESS[st],
                    f"wf={rng.choice([0, 1])} pr=3",
                )
    if name in ("cumul", "all"):
        # E-SMEM-CUMUL: coordinator ld.acquire.sys + st.volatile.shared relay, receiver LDS poll
        for pl in placements():
            for st in STRESS:
                for rd, pr in [(1, 3), (1, 1), (2, 3), (4, 3), (0, 2)]:
                    add(
                        f"relay-f0-rd{rd}-pr{pr}-{st}",
                        "relay",
                        0,
                        rd,
                        pl,
                        STRESS[st],
                        f"wf={rng.choice([0, 1])} pr={pr}",
                    )
    if name in ("controls", "all"):
        for pl in placements():
            for st in STRESS:
                add(
                    f"ctl-pos-relay-f1-{st}", "relay", 1, 1, pl, STRESS[st], "wf=1 pr=1"
                )
                add(
                    f"ctl-pos-relay-f1-pr2-{st}",
                    "relay",
                    1,
                    1,
                    pl,
                    STRESS[st],
                    "wf=1 pr=2",
                )
                add(f"ctl-neg-poll-rd3-{st}", "poll", 0, 3, pl, STRESS[st], "wf=0 pr=3")
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("out")
    ap.add_argument("--secs", type=float, default=30)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    bus_ids()
    jobs = plan(a.plan)
    if a.only:
        jobs = [j for j in jobs if re.search(a.only, j[0])]
    # one cross run (GPUs 0+1) and one intra run (GPU 3) at a time
    queues = {
        "0,1": [j for j in jobs if j[1] == "0,1"],
        "3": [j for j in jobs if j[1] == "3"],
    }
    lock = threading.Lock()
    with open(a.out, "a") as f:

        def worker(q):
            for j in q:
                r = run_cfg(j, a.secs)
                with lock:
                    print(r, flush=True)
                    f.write(r + "\n")
                    f.flush()

        with cf.ThreadPoolExecutor(2) as ex:
            list(ex.map(worker, queues.values()))


if __name__ == "__main__":
    main()

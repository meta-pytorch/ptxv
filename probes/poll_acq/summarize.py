#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Aggregate results/*.txt (sweep.py output) into per-family totals.

Usage: summarize.py [results/*.txt]     (default: every file in results/)
Family = the config tag without placement and stress, so each row sums over placements
(cross-GPU 0->1, 1->0, payload in producer memory, intra-GPU far / near / same SM) and stress modes
unless --by is given (--by placement | stress | both)."""

import glob
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["reads", "ok", "stale", "other", "timeout", "waits", "waited"]


def parse(paths):
    for p in paths:
        for line in open(p):
            m = re.match(r"\[([^\]]+)\] (RESULT .*|ERROR.*)", line.strip())
            if not m:
                continue
            tag, rest = m.groups()
            if rest.startswith("ERROR"):
                yield tag, None
                continue
            kv = dict(re.findall(r"(\w+)=([\w.,]+)", rest))
            yield tag, kv


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    by = sys.argv[sys.argv.index("--by") + 1] if "--by" in sys.argv else ""
    if "--by" in sys.argv:
        args = [a for a in args if a != by]
    paths = args or sorted(glob.glob(os.path.join(HERE, "results", "*.txt")))
    tot = defaultdict(Counter)
    runs = Counter()
    errors = []
    for tag, kv in parse(paths):
        plan, cfg, place, gpu = tag.split("/")
        cfgparts = cfg.split("-")
        stress = (
            cfgparts[-1]
            if cfgparts[-1] in ("none", "l2", "atom", "nvl", "l1", "payld")
            else ""
        )
        fam = "-".join(cfgparts[:-1]) if stress else cfg
        key = f"{plan}/{fam}"
        if by in ("placement", "both"):
            key += f" @{place}"
        if by in ("stress", "both"):
            key += f" [{stress}]"
        if kv is None:
            errors.append(tag)
            continue
        runs[key] += 1
        for k in KEYS:
            tot[key][k] += int(kv[k])
    print(
        f"{'family':48s} {'runs':>4s} {'reads':>10s} {'stale':>10s} {'other':>6s} {'tmo':>4s} {'stale%':>8s} {'waited%':>8s}"
    )
    for key in sorted(tot):
        t = tot[key]
        sp = 100.0 * t["stale"] / t["reads"] if t["reads"] else 0
        wp = 100.0 * t["waited"] / t["waits"] if t["waits"] else 0
        print(
            f"{key:48s} {runs[key]:4d} {t['reads']:10.3g} {t['stale']:10d} {t['other']:6d} {t['timeout']:4d} {sp:8.4f} {wp:8.2f}"
        )
    g = Counter()
    for t in tot.values():
        for k in KEYS:
            g[k] += t[k]
    print(
        f"{'TOTAL':48s} {sum(runs.values()):4d} {g['reads']:10.3g} {g['stale']:10d} {g['other']:6d} {g['timeout']:4d}"
    )
    if errors:
        print(f"{len(errors)} errors: {errors[:10]}")


if __name__ == "__main__":
    main()

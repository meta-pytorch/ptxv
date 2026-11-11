#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Aggregate litmus result lines (from sweep.py / asym.sh) per config: total instances,
forbidden count, settings run, and summed outcome histogram. Prints a markdown table."""

import collections
import re
import sys

agg = collections.OrderedDict()
for fn in sys.argv[1:]:
    for line in open(fn):
        m = re.match(
            r"(?:\[(\S+)\] )?(\S+ \S+ \S+ wf=\d rf=\d) .*\| n=(\d+) FORBIDDEN=(\d+) \| (.*) \| [\d.]+s",
            line.strip(),
        )
        if not m:
            continue
        sn, cfg, n, bad, hist = m.groups()
        a = agg.setdefault(
            cfg, {"n": 0, "bad": 0, "settings": set(), "hist": collections.Counter()}
        )
        a["n"] += int(n)
        a["bad"] += int(bad)
        a["settings"].add(sn or "asym")
        for kv in hist.split():
            k, v = kv.rsplit(":", 1)
            a["hist"][k] += int(v)
print("| config | settings | instances | forbidden | outcome histogram (summed) |")
print("|---|---|---:|---:|---|")
for cfg, a in agg.items():
    h = " ".join(
        f"{k}:{v:.2e}" if v >= 1e5 else f"{k}:{v}" for k, v in a["hist"].items()
    )
    print(f"| {cfg} | {len(a['settings'])} | {a['n']:.3e} | {a['bad']} | {h} |")

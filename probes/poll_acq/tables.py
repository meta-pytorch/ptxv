#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Print the markdown tables of docs/poll-acq-litmus.md from results/*.txt.
Cross-SM = every placement except i-same (producer and consumer on one SM)."""

import collections
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
STRESS = ("none", "l2", "atom", "nvl", "l1", "payld")
rows = []


def field(body, k):
    return int(re.search(r" %s=(\d+)" % k, body).group(1))


for f in sorted(glob.glob(os.path.join(HERE, "results", "*.txt"))):
    for line in open(f):
        m = re.match(r"\[([^/]+)/([^/]+)/([^/]+)/[^\]]+\] RESULT(.*)", line.strip())
        plan, cfg, pl, body = m.groups()
        base, st = cfg.rsplit("-", 1)
        fam = base if st in STRESS else cfg
        if plan == "refute" and st == "payld":
            fam = cfg
        rows.append(
            {
                "plan": plan,
                "cfg": cfg,
                "fam": fam,
                "pl": pl,
                "st": st,
                "reads": field(body, "reads"),
                "stale": field(body, "stale"),
                "waits": field(body, "waits"),
                "waited": field(body, "waited"),
            }
        )


def agg(rs):
    return (
        len(rs),
        sum(r["reads"] for r in rs),
        sum(r["stale"] for r in rs),
        sum(r["waited"] for r in rs),
        sum(r["waits"] for r in rs),
    )


def grp(n):
    if n in ("ctl-pos", "ctl-neg"):
        return [r for r in rows if r["cfg"].startswith(n)]
    return [r for r in rows if r["plan"] == n]


NAMES = {
    "deepep": "DeepEP shapes",
    "cumul": "E-SMEM-CUMUL relay",
    "refute": "E-POLL-ACQ refutation",
    "ctl-pos": "positive controls",
    "ctl-neg": "negative controls",
}
print("Exact totals (all placements, and cross-SM = without `i-same`):\n")
print(
    "| group | runs | reads | stale | cross-SM runs | cross-SM reads | cross-SM stale |"
)
print("|---|---:|---:|---:|---:|---:|---:|")
for n, name in NAMES.items():
    a, b = agg(grp(n)), agg([r for r in grp(n) if r["pl"] != "i-same"])
    print(f"| {name} | {a[0]} | {a[1]} | {a[2]} | {b[0]} | {b[1]} | {b[2]} |")
a = agg(rows)
print(f"| all | {a[0]} | {a[1]} | {a[2]} | | | |")

RD = {
    "0": "`LD.E`",
    "1": "`LDG.E`",
    "2": "`LDG.E.128.STRONG.SYS`",
    "3": "`fence.acq_rel.sys`; `LDG.E`",
    "4": "`LDG.E.128.CONSTANT`",
}
PR = {
    "0": "entry fence (pr0)",
    "1": "line from previous read (pr1)",
    "2": "reloaded before the poll (pr2)",
    "3": "reloaded in every poll iteration (pr3)",
}
FORM = {
    "0": "warp poll",
    "1": "lane spin",
    "2": "warp poll + `bar.sync`, second warp reads",
}


def desc(k):
    if k == "bb-f0":
        return "barrier_block (wf0), consumer warp reads, " + RD["0"] + ", " + PR["0"]
    if k == "bb-f2":
        return (
            "barrier_block (wf0) + `bar.sync`, second warp reads, "
            + RD["0"]
            + ", "
            + PR["0"]
        )
    m = re.match(r"(poll|relay|ctl-pos-relay|ctl-neg-poll)-f?(\d)?-?rd(\d)-pr(\d)", k)
    if k.startswith("ctl-pos-relay-f1"):
        return (
            "relay, coordinator relaxed poll without `CCTL.IVALL`, "
            + RD["1"]
            + ", "
            + (PR["2"] if "pr2" in k else PR["1"])
        )
    if k.startswith("ctl-neg"):
        return "warp poll, " + RD["3"] + ", " + PR["3"]
    t, f, rd, pr = m.groups()
    s = (
        (
            FORM[f]
            if t == "poll"
            else "relay: coordinator `LDG.E.STRONG.SYS; CCTL.IVALL; STS`"
        )
        + ", "
        + RD[rd]
        + ", "
        + PR[pr]
    )
    if k.endswith("payld"):
        s += ", stress warps load the payload lines"
    return s


def table(title, rs):
    print(f"\n### {title}\n")
    print(
        "| family | shape | runs | reads (cross-SM) | stale (cross-SM) | stale % | reads (i-same) | stale (i-same) | waited % |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    fams = collections.OrderedDict()
    for r in rs:
        fams.setdefault(r["fam"], []).append(r)
    for k, v in fams.items():
        x, y, al = (
            agg([r for r in v if r["pl"] != "i-same"]),
            agg([r for r in v if r["pl"] == "i-same"]),
            agg(v),
        )
        print(
            f"| `{k}` | {desc(k)} | {al[0]} | {x[1]:.3g} | {x[2]} | {100 * x[2] / max(x[1], 1):.3f} | "
            f"{y[1]:.3g} | {y[2]} | {100 * al[3] / al[4]:.1f} |"
        )


table(
    "DeepEP shapes (plan `deepep`, ~30 s per run, stress none / l2 / atom / nvl / l1)",
    grp("deepep"),
)
table(
    "E-SMEM-CUMUL (plan `cumul`, ~30 s per run, no stress and five stress modes)",
    grp("cumul"),
)
table(
    "E-POLL-ACQ refutation (plan `refute`, ~3 s per run, stress none / l1)",
    grp("refute"),
)
table(
    "Controls (plan `controls`, ~3 s per run, every placement x {none + five stress modes})",
    grp("ctl-pos") + grp("ctl-neg"),
)

print("\n### E-POLL-ACQ refutation by placement\n")
print(
    "| placement | rows | runs | reads | stale | stale % |\n|---|---|---:|---:|---:|---:|"
)
t = collections.OrderedDict()
for r in grp("refute"):
    kind = (
        "pr0 + payld"
        if r["st"] == "payld"
        else ("pr1-3, l1 thrash" if r["st"] == "l1" else "pr1-3, no stress")
    )
    t.setdefault((r["pl"], kind), []).append(r)
for (pl, kind), v in sorted(t.items()):
    a = agg(v)
    print(f"| {pl} | {kind} | {a[0]} | {a[1]:.3g} | {a[2]} | {100 * a[2] / a[1]:.3f} |")

print(
    "\n### Positive control (relay without `CCTL.IVALL`), stale % by placement and stress\n"
)
print("| placement | " + " | ".join(STRESS) + " |\n|---|" + "---:|" * len(STRESS))
for pl in ["x01", "x10", "x01-loc1", "i-far", "i-near", "i-same"]:
    cells = []
    for s in STRESS:
        a = agg([r for r in grp("ctl-pos") if r["pl"] == pl and r["st"] == s])
        cells.append(f"{100 * a[2] / a[1]:.1f}")
    print(f"| {pl} | " + " | ".join(cells) + " |")

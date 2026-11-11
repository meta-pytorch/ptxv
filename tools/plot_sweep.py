# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scoreboard figure: what the control bits say, and the mutation sweep against GB300.

    $PTXV_PYTHON tools/plot_sweep.py out -o docs/img/sass_ctrl.png

(a) outcome of every one-bit scoreboard mutant, per kernel.
(b) the r1a epilogue as the hardware sees it: per scoreboard, where it is set by a write
    (filled dot) or read (ring), how long it is outstanding (bar), and where it is waited on
    (triangle). Right: outcome of removing each bit of that instruction.
"""

import argparse
import json
import os
import re
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e5e5e0"
SURF = "#fcfcfb"
SB_C = "#2a78d6"
CLS = [  # status palette; every use also carries a text label
    ("caught", "#0ca30c", "caught: flagged, GPU wrong"),
    ("hazard (hw won)", "#fab219", "flagged, GPU won the race"),
    ("benign", "#c3c2b7", "not flagged, GPU correct"),
    ("missed", "#d03b3b", "MISSED: not flagged, GPU wrong"),
]
COL = {k: c for k, c, _ in CLS}


def ctrl_lines(cubin, kernel):
    nv = os.environ.get("NVDISASM", "/usr/local/cuda-13.0/bin/nvdisasm")
    t = subprocess.run(
        [nv, "-c", "-hex", cubin], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    out, i, inside = [], 0, False
    while i < len(t):
        s = t[i].strip()
        if s.startswith(".text."):
            inside = s == f".text.{kernel}:"
        m = re.match(r"/\*([0-9a-f]{4})\*/\s+(.*?)\s*;?\s*/\* 0x[0-9a-f]{16} \*/", s)
        if inside and m:
            hi = int(re.search(r"0x([0-9a-f]{16})", t[i + 1]).group(1), 16)
            c = hi >> 41

            def sb(x):
                return None if x == 7 else x

            out.append(
                {
                    "addr": int(m.group(1), 16),
                    "text": m.group(2).strip().rstrip(";").strip(),
                    "wbar": sb((c >> 5) & 7),
                    "rbar": sb((c >> 8) & 7),
                    "wait": (c >> 11) & 0x3F,
                }
            )
            i += 2
        else:
            i += 1
    return out


def main():  # noqa: C901 (plotting script: one sequential pass per figure)
    ap = argparse.ArgumentParser()
    ap.add_argument("outroot")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    cases = [
        ("r0_tma_sum", "rung 0: TMA + mbarrier"),
        ("r1a_tcgen05_tile", "rung 1a: tcgen05 tile"),
    ]
    sweeps = {
        c: json.load(open(os.path.join(a.outroot, c, "sb_sweep", "sweep.json")))
        for c, _ in cases
    }

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
        }
    )
    fig = plt.figure(figsize=(16, 8.6), facecolor=SURF)
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.25, 1], height_ratios=[1, 2.2], hspace=0.35, wspace=0.08
    )

    # (a) outcomes per kernel
    ax = fig.add_subplot(gs[0, 0], facecolor=SURF)
    for yi, (c, _label) in enumerate(cases):
        rows = sweeps[c]
        left = 0
        for k, col, _ in CLS:
            n = sum(r["class"] == k for r in rows)
            if n:
                ax.barh(
                    yi,
                    n,
                    left=left,
                    color=col,
                    height=0.55,
                    edgecolor=SURF,
                    linewidth=2,
                )
                ax.text(
                    left + n / 2,
                    yi,
                    str(n),
                    ha="center",
                    va="center",
                    color=INK,
                    fontsize=9,
                )
            left += n
        miss = sum(r["class"] == "missed" for r in rows)
        ax.text(
            left + 2,
            yi,
            f"{len(rows)} mutants, {miss} missed",
            va="center",
            color=INK,
            fontsize=9,
        )
    ax.set_yticks(range(len(cases)), [ln for _, ln in cases])
    ax.set_xlim(0, max(len(r) for r in sweeps.values()) * 1.38)
    ax.invert_yaxis()
    ax.set_xlabel(
        "one-bit scoreboard mutants of the good binary (200 GPU runs each, smem scrubbed)"
    )
    ax.set_title("(a) static scoreboard check vs. GB300", loc="left", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(
        handles=[Patch(color=c, label=ln) for _, c, ln in CLS],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )

    # (b) r1a epilogue scoreboard timeline
    c = "r1a_tcgen05_tile"
    lines = [
        ln
        for ln in ctrl_lines(os.path.join(a.outroot, c, "good.cubin"), c)
        if 0x09C0 <= ln["addr"] <= 0x0BC0
    ]
    by_addr = {}
    for r in sweeps[c]:
        by_addr.setdefault(r["addr"], []).append(r)
    ax = fig.add_subplot(gs[:, 1], facecolor=SURF)
    n = len(lines)
    used = sorted(
        {k for ln in lines for k in (ln["wbar"], ln["rbar"]) if k is not None}
        | {k for ln in lines for k in range(6) if ln["wait"] >> k & 1}
    )
    NS = len(used)
    X = {k: i for i, k in enumerate(used)}
    open_at = {}  # sb -> (row, filled)
    for i, ln in enumerate(lines):
        for k in used:
            if ln["wait"] >> k & 1:
                ax.plot(X[k], i, marker="v", color=INK, ms=8, zorder=4)
                if k in open_at:
                    r0 = open_at.pop(k)
                    ax.plot(
                        [X[k], X[k]],
                        [r0, i],
                        color=SB_C,
                        lw=2.5,
                        alpha=0.55,
                        solid_capstyle="round",
                        zorder=2,
                    )
        for k, filled in ((ln["wbar"], True), (ln["rbar"], False)):
            if k is not None:
                ax.plot(
                    X[k],
                    i,
                    marker="o",
                    ms=8,
                    color=SB_C,
                    markerfacecolor=SB_C if filled else SURF,
                    markeredgewidth=2,
                    zorder=5,
                )
                open_at.setdefault(k, i)
    for k, r0 in open_at.items():
        ax.plot([X[k], X[k]], [r0, n - 0.5], color=SB_C, lw=2.5, alpha=0.55, zorder=2)
    for i, ln in enumerate(lines):
        for j, r in enumerate(by_addr.get(ln["addr"], [])):
            ax.scatter(
                NS + 0.2 + 0.4 * j,
                i,
                marker="s",
                s=60,
                color=COL[r["class"]],
                edgecolor=SURF,
                linewidth=1.5,
                zorder=5,
            )
    ax.set_yticks(
        range(n),
        [f"{ln['addr']:04x}  {ln['text'][:46]}" for ln in lines],
        fontsize=7.6,
        family="monospace",
    )
    ax.set_xticks(list(range(NS)) + [NS + 0.4], [f"SB{k}" for k in used] + ["mutants"])
    ax.set_xlim(-0.6, NS + 1.2)
    ax.set_ylim(n - 0.5, -0.5)
    ax.yaxis.tick_right()
    ax.grid(axis="x", color=GRID, lw=0.8)
    for s in ("top", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(
        "(b) rung 1a epilogue: scoreboards set, outstanding, waited on",
        loc="left",
        color=INK,
    )
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                color=SB_C,
                ls="",
                ms=8,
                label="write scoreboard set",
            ),
            Line2D(
                [],
                [],
                marker="o",
                color=SB_C,
                markerfacecolor=SURF,
                markeredgewidth=2,
                ls="",
                ms=8,
                label="read scoreboard set",
            ),
            Line2D(
                [], [], color=SB_C, lw=2.5, alpha=0.55, label="outstanding until a wait"
            ),
            Line2D(
                [], [], marker="v", color=INK, ls="", ms=8, label="wait (before issue)"
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(-0.02, -0.04),
        ncol=4,
        fontsize=8,
        frameon=False,
    )

    # note for (a)
    ax = fig.add_subplot(gs[1, 0], facecolor=SURF)
    ax.axis("off")
    ax.text(
        0,
        0.98,
        "How to read (b)\n\n"
        "LDTM R36 at 0x0a30 sets SB1. The wait on SB1 sits on the next LDTM\n"
        "(0x0a70), two instructions before the first STG of R36: the binary's\n"
        "only equivalent of tcgen05.wait::ld. Removing that one bit (B44)\n"
        "is flagged; GB300 still computes correctly, because the STG's read\n"
        "of R36 queues behind the LDTM in the memory pipe.\n\n"
        "Removing the wait at 0x09d0 (the branch that consumes TRYWAIT's P0)\n"
        "lets the epilogue run before the MMA finished: 200/200 wrong.",
        va="top",
        ha="left",
        color=INK2,
        fontsize=9,
        family="sans-serif",
    )
    fig.savefig(a.out, dpi=150, bbox_inches="tight", facecolor=SURF)


if __name__ == "__main__":
    main()

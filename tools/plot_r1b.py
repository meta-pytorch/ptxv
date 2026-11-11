# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rung 1b figure.

    $PTXV_PYTHON tools/plot_r1b.py -o docs/img/r1b.png

(a) states explored for the good kernel (PTX) by each explorer version (log scale).
(b) verdict per variant: PTX model, SASS model, GB300 (1000 runs, smem scrubbed).
Numbers are from docs/rung1b.md (measured 2026-10-30).
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e5e5e0"
SURF = "#fcfcfb"
BAR = "#2a78d6"
BAR_OPEN = "#c3c2b7"  # (unused: every run completed)

# (label, states, seconds, complete)
STEPS = [
    ("stateful explorer", 1_001_621, 3203, True),
    ("DPOR: warps", 218_345, 583, True),
    ("DPOR: splits", 3_263, 10.6, True),
    ("DPOR: + merge-sensitive pcs", 481, 1.5, True),
]

# status palette; every cell also carries its text
ST = {
    "ok": ("#0ca30c", "as expected"),
    "bug": ("#d03b3b", "GPU fails (predicted)"),
    "quiet": ("#fab219", "GPU won the race (latent)"),
}
ROWS = [
    (
        "good",
        [
            ("clean\nexhaustive", "ok"),
            ("clean\nexhaustive", "ok"),
            ("bitwise\n1000/1000", "ok"),
        ],
    ),
    (
        "B04 no proxy fence",
        [("race", "ok"), ("race", "ok"), ("wrong\n1000/1000", "bug")],
    ),
    ("B09 no tmem_empty wait", [("race", "ok"), ("race", "ok"), ("correct", "quiet")]),
    (
        "B10 wrong parity after wrap",
        [("race+fault", "ok"), ("race+fault", "ok"), ("launch\nfailure", "bug")],
    ),
    ("B11 skipped commit", [("deadlock", "ok"), ("deadlock", "ok"), ("hang", "bug")]),
    ("B50 no wait_group.read", [("race", "ok"), ("race", "ok"), ("correct", "quiet")]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="docs/img/r1b.png")
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2})
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.25]}, facecolor=SURF
    )

    # (a) state counts
    y = list(range(len(STEPS)))[::-1]
    for yi, (_lbl, n, secs, done) in zip(y, STEPS):
        a.barh(yi, n, height=0.5, color=BAR if done else BAR_OPEN, zorder=2)
        txt = f"{n:,} states, {secs:g} s" if done else f">{n:,}: incomplete"
        a.text(n * 1.25, yi, txt, va="center", color=INK, fontsize=8.5)
    a.set_yticks(y, [s[0] for s in STEPS])
    a.set_xscale("log")
    a.set_xlim(100, 1e9)
    a.set_xlabel("states explored (log scale)")
    a.set_title("(a) exhaustive search of the good kernel (PTX)", loc="left", color=INK)
    a.grid(axis="x", color=GRID, zorder=0)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.spines["bottom"].set_color(GRID)
    a.tick_params(length=0, colors=INK2)
    a.set_facecolor(SURF)

    # (b) verdict matrix
    cols = ["PTX model", "SASS model", "GB300"]
    for r, (_name, cells) in enumerate(ROWS):
        for c, (txt, st) in enumerate(cells):
            b.add_patch(
                plt.Rectangle(
                    (c + 0.03, -r - 0.47), 0.94, 0.94, color=ST[st][0], alpha=0.22, lw=0
                )
            )
            b.text(c + 0.5, -r, txt, ha="center", va="center", fontsize=8, color=INK)
    b.set_xlim(0, 3)
    b.set_ylim(-len(ROWS) + 0.5, 0.9)
    b.set_xticks([0.5, 1.5, 2.5], cols)
    b.xaxis.tick_top()
    b.set_yticks([-r for r in range(len(ROWS))], [r[0] for r in ROWS])
    b.tick_params(length=0, colors=INK2)
    for s in b.spines.values():
        s.set_visible(False)
    b.set_title(
        "(b) verdicts: every GPU failure was predicted", loc="left", color=INK, pad=22
    )
    b.legend(
        handles=[Patch(color=c, alpha=0.4, label=l) for c, l in ST.values()],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    b.set_facecolor(SURF)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, facecolor=SURF, bbox_inches="tight", pad_inches=0.15)


if __name__ == "__main__":
    main()

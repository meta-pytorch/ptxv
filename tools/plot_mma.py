# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rung 1a figure: tensor-core calibration and what the calibrated arithmetic means.

$PTXV_PYTHON tools/plot_mma.py out/r1a_tcgen05_tile -o docs/img/r1a_mma.png
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

INK = "#1f2933"
C_HW = "#2563eb"
C_ID = "#9ca3af"
C_EPS = "#dc2626"
GRID = "#e5e7eb"
U = 2.0**-24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    cal = json.load(open(os.path.join(a.outdir, "calib", "calib.json")))
    hw = json.load(open(os.path.join(a.outdir, "good.sass.num.json")))
    ideal = json.load(
        open(
            os.path.join(a.outdir, "good.sass.num.g16-exact-rn-accin-tzero-eprod.json")
        )
    )

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    fig = plt.figure(figsize=(17, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1])

    # (a) calibration: mismatches of the best candidates on each input set
    ax = fig.add_subplot(gs[0])
    sets = [s[0] for s in cal["sets"]]
    top = cal["scores"][:14]
    mism = np.array([[s["total"] - m for m in s["matches"]] for s in top], dtype=float)
    im = ax.imshow(
        np.maximum(mism, 0.5),
        cmap="Blues",
        norm=LogNorm(vmin=0.5, vmax=8192),
        aspect="auto",
    )
    for i in range(mism.shape[0]):
        for j in range(mism.shape[1]):
            v = int(mism[i, j])
            ax.text(
                j,
                i,
                "0" if v == 0 else str(v),
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if v > 600 else INK,
            )
    ax.set_xticks(range(len(sets)), sets, rotation=20)
    ax.set_yticks(range(len(top)), [s["model"] for s in top], fontsize=7.5)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_title(
        f"(a) Calibration: wrong outputs (of 8192) per input set\n{len(cal['scores'])} candidate models; exactly one matches everything",
        loc="left",
        color=INK,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("mismatching outputs (log)")

    # (b) direction of the error: truncation + RZ pulls results toward zero
    ax = fig.add_subplot(gs[1])

    def toward_zero(recs):
        r = np.array(
            [
                x["signed_err"] * np.sign(x["exact"]) / (U * x["abs_sum"])
                for x in recs
                if x["abs_sum"] > 0
            ]
        )
        return r

    th, ti = toward_zero(hw), toward_zero(ideal)
    lim = max(np.percentile(np.abs(th), 99.5), 1)
    bins = np.linspace(-lim, lim, 61)
    ax.hist(
        ti,
        bins=bins,
        color=C_ID,
        alpha=0.7,
        label="ideal tensor core (exact dot, round-to-nearest)",
    )
    ax.hist(
        th,
        bins=bins,
        color=C_HW,
        alpha=0.75,
        label="GB300 tensor core (calibrated model)",
    )
    ax.axvline(0, color=INK, lw=0.8)
    fz = np.mean(th < 0) * 100
    fzi = np.mean(ti < 0) * 100
    ax.set_xlabel(
        "signed error × sign(exact) / (u · Σ|a_k b_k|)      (negative = toward zero)"
    )
    ax.set_ylabel("outputs")
    ax.set_title(
        f"(b) GB300 errors point toward zero in {fz:.0f}% of outputs\n(ideal RN: {fzi:.0f}%); mean {np.mean(th):.2f} vs {np.mean(ti):.2f}",
        loc="left",
        color=INK,
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, color=GRID, lw=0.6)

    # (c) error relative to the natural scale vs the model's error contract
    ax = fig.add_subplot(gs[2])
    eh = np.array([x["abs_err"] / (U * x["abs_sum"]) for x in hw if x["abs_sum"] > 0])
    ei = np.array(
        [x["abs_err"] / (U * x["abs_sum"]) for x in ideal if x["abs_sum"] > 0]
    )
    for e, c, lab in [(ei, C_ID, "ideal RN"), (eh, C_HW, "GB300")]:
        s = np.sort(e)
        ax.plot(s, 1 - np.arange(len(s)) / len(s), color=c, lw=2, label=lab)
    eps_steps = 4  # four chained MMA instructions per output (K = 64)
    eps = (
        17 * 2 * 2.0**-24 + 2.0**-23
    )  # calibrated model contract per step (see fp.rs eps())
    ax.axvline(
        eps_steps * eps / U,
        color=C_EPS,
        ls="--",
        lw=1.2,
        label="first-order rigorous bound\n(4 steps × model eps)",
    )
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yscale("log")
    ax.set_xlabel("|error| / (u · Σ|a_k b_k|)")
    ax.set_ylabel("fraction of outputs with larger error")
    ax.set_title(
        "(c) Error tail: GB300 ~2x the ideal,\n~100x inside the rigorous bound",
        loc="left",
        color=INK,
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(True, color=GRID, lw=0.6)

    fig.suptitle(
        "Rung 1a: tcgen05.mma (bf16 x bf16 -> fp32, K=64 as 4 x K16) on GB300 / sm_103a",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(a.out)


if __name__ == "__main__":
    main()

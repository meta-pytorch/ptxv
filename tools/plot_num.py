# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plot ptxv numerics for one case: per-output error vs bounds, ulp histogram, and the
exact error identity (fp_noise Thm 1) for the worst output.

    $PTXV_PYTHON tools/plot_num.py out/r0_tma_sum/good.sass.num.json -o docs/img/r0_numerics.png
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#1f2933"
C_ERR = "#2563eb"
C_RUN = "#d97706"
C_COR = "#9333ea"
C_GRID = "#e5e7eb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    recs = json.load(open(a.json))
    idx = np.array([r["index"] for r in recs])
    err = np.array([r["abs_err"] for r in recs])
    cor = np.array([r["bound_cor1a"] for r in recs])
    run = np.array([r["bound_running"] for r in recs])
    ulps = np.array([r["err_ulps"] for r in recs])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    fig, ax = plt.subplots(
        1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [2.2, 1, 1.6]}
    )

    # (a) per-output error and bounds
    a0 = ax[0]
    floor = max(err[err > 0].min() / 4, 1e-12) if (err > 0).any() else 1e-12
    a0.semilogy(idx, cor, color=C_COR, lw=1.6, label="a-priori bound (Cor 1a)")
    a0.semilogy(idx, run, color=C_RUN, lw=1.6, label="running bound (a-posteriori)")
    a0.scatter(
        idx,
        np.maximum(err, floor),
        s=14,
        color=C_ERR,
        zorder=3,
        label="actual |error| (exact)",
    )
    zero = err == 0
    if zero.any():
        a0.scatter(
            idx[zero],
            np.full(zero.sum(), floor),
            s=14,
            marker="v",
            color="#9ca3af",
            zorder=3,
            label="exact result (error = 0)",
        )
    a0.set_xlabel("output index t  (out[t] = fp32 sum of 32 bf16)")
    a0.set_ylabel("absolute error")
    a0.set_title("Per output: actual error vs. the two bounds", loc="left", color=INK)
    a0.grid(True, color=C_GRID, lw=0.6)
    a0.set_ylim(floor / 3, max(cor.max(), run.max()) * 30)
    a0.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)

    # (b) error in ulps
    a1 = ax[1]
    bins = np.linspace(0, max(0.55, ulps.max() * 1.1), 23)
    a1.hist(ulps, bins=bins, color=C_ERR, alpha=0.85)
    a1.axvline(0.5, color=INK, ls="--", lw=1)
    a1.text(
        0.505,
        a1.get_ylim()[1] * 0.92,
        "0.5 ulp\n(correct rounding)",
        fontsize=8,
        color=INK,
    )
    a1.set_xlabel("|error| in ulps of the fp32 result")
    a1.set_ylabel("outputs")
    a1.set_title(
        f"{int(zero.sum())}/{len(err)} outputs are exactly correct",
        loc="left",
        color=INK,
    )
    a1.grid(True, axis="y", color=C_GRID, lw=0.6)

    # (c) Thm 1 waterfall for the worst output
    a2 = ax[2]
    w = int(np.argmax(err))
    le = np.array(recs[w]["local_errs"])
    cum = np.cumsum(le)
    k = np.arange(1, len(le) + 1)
    a2.bar(
        k,
        le,
        color=np.where(le >= 0, C_ERR, "#dc2626"),
        alpha=0.75,
        label="local rounding error e_v of add #k",
    )
    a2.plot(k, cum, color=INK, lw=1.8, marker="o", ms=3, label="running sum of e_v")
    a2.axhline(
        recs[w]["signed_err"],
        color=C_RUN,
        ls="--",
        lw=1.2,
        label="final error  hw − exact",
    )
    a2.set_xlabel("add number k along the sequential sum")
    a2.set_ylabel("error")
    a2.set_title(
        f"Thm 1, worst output t={recs[w]['index']}:\nsum of local errors = hw − exact (checked exactly)",
        loc="left",
        color=INK,
        fontsize=10,
    )
    a2.grid(True, color=C_GRID, lw=0.6)
    a2.legend(frameon=False, fontsize=8, loc="best")

    if a.title:
        fig.suptitle(a.title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(a.out)


if __name__ == "__main__":
    main()

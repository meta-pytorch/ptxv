#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validation matrix for `ptxv errdist` (docs/errdist.md section 7).

Runs precision x input family x mode rows through the CLI (Tier 3 Monte Carlo as ground
truth, with the analytic tiers alongside) and writes a TSV and a markdown table.

    tools/errdist_matrix.py --bin target/release/ptxv --out runs/matrix [--rows REGEX] [-j 4]
    tools/errdist_matrix.py --bin target/release/ptxv --out runs/fast --fast   # ~8 min, exit 1 on a failure

`--fast` runs FAST_ROWS (one row per tier domain) and fails when the `best` analytic tier
(1 or 2, when it reports itself applicable) misses Tier 3 by more than the tolerance.

A row passes when |P_tier - P_3| <= max(Tier 3 99.9% CI half-width, 5% of P_3) and, for
single kernels with an f32 output, the tier's ulp quantiles q0.01 / q0.99 are within one
bin (1/16 ulp) or 10% of Tier 3's.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "fixtures"
SP = FX / "errdist"

# (label, precision, family, mode, target, candidate, spec)
ROWS = []


def row(prec, fam, mode, target, spec, cand=None, label=None):
    lab = label or f"{prec}/{mode}/{fam}"
    ROWS.append((lab, prec, fam, mode, target, cand, spec))


for fam in [
    "normal",
    "uniform12",
    "student_t",
    "laplace",
    "lognormal",
    "outliers",
    "integers",
    "ints_scaled",
]:
    row("bf16", fam, "single", "c_mm_bf16_nnt", f"{fam}.json")
for fam in ["normal", "student_t", "uniform12", "ints_scaled"]:
    row("bf16", fam, "pair-rev", "c_equiv_mm_bf16/equiv.toml", f"{fam}.json", cand=1)
row("bf16", "normal", "pair-fwd", "c_equiv_mm_bf16/equiv.toml", "normal.json", cand=0)
for fam in ["normal", "student_t", "uniform12"]:
    row("f16", fam, "single", "r2_cute_gemm", f"{fam}.json")
for fam, spec in [
    ("normal-mx", "mxf8_gauss"),
    ("student_t-mx", "mxf8_student_t"),
    ("laplace-mx", "mxf8_laplace"),
    ("lognormal-mx", "mxf8_lognormal"),
    ("outliers-mx", "mxf8_outliers"),
    ("codes", "mxf8_codes"),
    ("ints", "mxf8_ints"),
    ("ints_scaled", "mxf8_ints_scaled"),
]:
    row("mxf8", fam, "single", "r3_cute_mxf8", f"{spec}.json")
for fam, spec in [
    ("normal-mx", "mxf8_gauss"),
    ("codes", "mxf8_codes"),
    ("ints_scaled", "mxf8_ints_scaled"),
    ("student_t-mx", "mxf8_student_t"),
]:
    row(
        "mxf8",
        fam,
        "pair-rev",
        "c_equiv_mxf8_triton/equiv.toml",
        f"{spec}.json",
        cand=1,
    )
row(
    "mxf8",
    "normal-mx",
    "pair-cublas",
    "c_equiv_mxf8_cublas/equiv.toml",
    "mxf8_gauss.json",
    cand=0,
)
for fam, spec in [
    ("normal-mx", "mxf4_gauss"),
    ("codes", "mxf4_codes"),
    ("student_t-mx", "mxf4_student_t"),
    ("laplace-mx", "mxf4_laplace"),
]:
    row("mxf4", fam, "single", "r3_cute_mxf4", f"{spec}.json")
for fam, spec in [("codes", "mxf4_codes"), ("student_t-mx", "mxf4_student_t")]:
    row(
        "mxf4",
        fam,
        "pair-rev",
        "c_equiv_mxf4_triton/equiv.toml",
        f"{spec}.json",
        cand=1,
    )
for fam, spec in [
    ("normal-nv", "nvf4_gauss"),
    ("student_t-nv", "nvf4_student_t"),
    ("codes", "nvf4_codes"),
]:
    row("nvf4", fam, "single", "r3_cute_nvf4", f"{spec}.json")
for fam, spec in [("normal-nv", "nvf4_gauss"), ("codes", "nvf4_codes")]:
    row("nvf4", fam, "pair-dq", "c_equiv_nvf4_dq/equiv.toml", f"{spec}.json", cand=0)
row(
    "nvf4",
    "normal-nv",
    "pair-cublas",
    "c_equiv_nvf4_cublas/equiv.toml",
    "nvf4_gauss.json",
    cand=0,
)
for fam in ["normal", "student_t", "uniform12"]:
    row("tf32", fam, "single", "c_tf32_mm_1x", f"{fam}.json")
    row("3xtf32", fam, "single", "c_tf32_mm_3x", f"{fam}.json")
row("3xtf32", "normal", "pair-rev", "c_equiv_tf32/equiv.toml", "normal.json", cand=1)
row(
    "3xtf32",
    "normal",
    "pair-3x-vs-f32",
    "c_equiv_tf32/equiv.toml",
    "normal.json",
    cand=0,
)
for fam in ["normal", "uniform12", "student_t"]:
    row("f32-sum", fam, "single", "r4_sum_torch", f"{fam}.json")
    row("f32-fma", fam, "single", "c_mm_f32_gemmsn", f"{fam}.json")
for c, m in [(0, "pair-rowreduce"), (1, "pair-splitk"), (2, "pair-atomic")]:
    row("f32-sum", "normal", m, "c_equiv_sum/equiv.toml", "normal.json", cand=c)
row("f32-scan", "normal", "single", "c_cumsum", "normal.json")


# One row per analytic domain: Tier 1 (bf16 chain, MX scaled integers), Tier 2 (reduction,
# split-K pair, 3xTF32).
FAST_ROWS = [
    "bf16/single/normal",
    "mxf8/single/ints_scaled",
    "f32-sum/single/normal",
    "f32-sum/pair-splitk/normal",
    "3xtf32/single/normal",
]


def tier(doc, t):
    for e in doc.get("estimates", []):
        if e.get("tier") == t:
            return e
    return None


def run_row(r, a):
    lab, prec, fam, mode, target, cand, spec = r
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", lab)
    out = Path(a.out)
    js = out / f"{safe}.json"
    log = out / f"{safe}.log"
    cmd = [
        "systemd-run",
        "--user",
        "--scope",
        "-q",
        "-p",
        f"MemoryMax={a.mem}",
        "-p",
        "MemorySwapMax=0",
        a.bin,
        "errdist",
        str(FX / target),
        "--inputs",
        str(SP / spec),
        "--mc",
        str(a.mc),
        "--threads",
        str(a.threads),
        "--json",
        str(js),
        "--out",
        str(out / "scratch" / safe),
    ]
    if cand is not None:
        cmd += ["--candidate", str(cand)]
    cmd += a.extra
    env = dict(os.environ)
    env["PTXV_ERRDIST_THREADS"] = str(a.threads)
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    t0 = time.time()
    if not (a.reuse and js.exists()):
        if js.exists():
            js.unlink()
        try:
            with open(log, "w") as f:
                subprocess.run(
                    cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=a.timeout
                )
        except subprocess.TimeoutExpired:
            pass
    wall = time.time() - t0
    res = {"label": lab, "prec": prec, "family": fam, "mode": mode, "wall": wall}
    if not js.exists():
        tail = log.read_text().strip().splitlines()[-1:] if log.exists() else ["?"]
        res["error"] = tail[0][:200] if tail else "?"
        return res
    doc = json.loads(js.read_text())
    res["best"] = (
        doc["estimates"][doc.get("best", 0)]["tier"] if doc.get("estimates") else None
    )
    res["applicable"] = {
        e["tier"]: e.get("applicable") for e in doc.get("estimates", [])
    }
    for t in (0, 1, 2, 3):
        e = tier(doc, t)
        if e is None or e.get("p_differ") is None:
            continue
        res[f"p{t}"] = e["p_differ"][0]
        res[f"s{t}"] = e.get("seconds")
        if t == 3:
            res["lo3"], res["hi3"] = e["p_differ"][1], e["p_differ"][2]
        u = e.get("ulp") or {}
        q = {round(x[0], 4): x[1] for x in u.get("quantiles", [])}
        res[f"q{t}"] = (q.get(0.01), q.get(0.99))
    return res


def verdict(res, t):
    if f"p{t}" not in res or "p3" not in res:
        return ""
    p, p3 = res[f"p{t}"], res["p3"]
    half = max(res["p3"] - res["lo3"], res["hi3"] - res["p3"])
    tol = max(half, 0.05 * p3)
    ok = abs(p - p3) <= tol or (p3 == 0 and p < 1e-6)
    return "pass" if ok else "FAIL"


def fmt(x, n=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{n}g}" if x != 0 else "0"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=".")
    ap.add_argument("--mc", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("-j", type=int, default=3)
    ap.add_argument("--mem", default="60G")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--reuse", action="store_true", help="reuse existing JSON reports")
    ap.add_argument("--list", action="store_true")
    ap.add_argument(
        "--fast",
        action="store_true",
        help="the fast subset; exit 1 on a failing best tier",
    )
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()
    rows = [
        r for r in ROWS if re.search(a.rows, r[0]) and (not a.fast or r[0] in FAST_ROWS)
    ]
    if a.list:
        for r in rows:
            print(r[0])
        return
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "scratch").mkdir(exist_ok=True)
    results = []
    with cf.ThreadPoolExecutor(a.j) as ex:
        futs = {ex.submit(run_row, r, a): r for r in rows}
        for f in cf.as_completed(futs):
            res = f.result()
            results.append(res)
            print(
                f"{res['label']:40s} "
                + " ".join(
                    f"p{t}={fmt(res.get(f'p{t}'))}"
                    for t in (3, 1, 2, 0)
                    if f"p{t}" in res
                )
                + (f" ERROR {res['error']}" if "error" in res else ""),
                flush=True,
            )
    order = {r[0]: i for i, r in enumerate(ROWS)}
    results.sort(key=lambda r: order[r["label"]])
    (Path(a.out) / "matrix.json").write_text(json.dumps(results, indent=1))
    lines = [
        "| row | Tier 3 P(differ) [99.9% CI] | Tier 1 | T1/T3 | Tier 2 | T2/T3 | Tier 0 | best | T1 s | T2 s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['label']} | error: {r['error']} | | | | | | | | |")
            continue
        p3 = r.get("p3")
        ci = (
            f"{fmt(p3)} [{fmt(r.get('lo3'))}, {fmt(r.get('hi3'))}]"
            if p3 is not None
            else "-"
        )

        def ratio(t, r=r, p3=p3):
            p = r.get(f"p{t}")
            if p is None or not p3:
                return "-"
            return f"{p / p3:.3f} {verdict(r, t)}"

        lines.append(
            f"| {r['label']} | {ci} | {fmt(r.get('p1'))} | {ratio(1)} | {fmt(r.get('p2'))} | {ratio(2)} | "
            f"{fmt(r.get('p0'))} | {fmt(r.get('best'))} | {fmt(r.get('s1'), 3)} | {fmt(r.get('s2'), 3)} |"
        )
    (Path(a.out) / "matrix.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if a.fast:
        bad = [
            r["label"]
            for r in results
            if "error" in r
            or (r.get("best") in (1, 2) and verdict(r, r["best"]) == "FAIL")
        ]
        if bad:
            print("FAIL:", ", ".join(bad))
            sys.exit(1)
        print("fast subset: every best analytic tier within tolerance")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""errdist Tier 2 scorecard against stored GB300 measurements.

The rows are unstable / stable Triton pairs (softmax, variance, dot bf16 / f32
accumulator, sum sequential / Kahan, log-sum-exp) under the refresh's input specs, plus the
input-parameter sweeps (mean x std grids, dot scale x distribution). The GB300 histograms were
measured once (errdist --gpu-trials on the same cubins) and are stored in the data directory,
so a scorecard reruns only the ptxv side (Tier 3 and the analytic tiers, --gpu-trials 0).

  errdist_scorecard.py gpu-ref GPU_RUNS_DIR      # once: GB300 reference from errdist JSON runs
  errdist_scorecard.py run --name NAME [--mode real|float|both] [--sweeps] [--baseline NAME]
  errdist_scorecard.py diff NAME BASELINE
  errdist_scorecard.py selftest

Data directory (--data, or $PTXV_SCORECARD_DATA; required): cases/<case>/ (captured kernels), specs/,
specs/sweep/, gpu_ref.json, and scorecards/<NAME>.{json,md} written by `run`. Where the shared
data directory lives: fb/errdist_scorecard.md. New rows' GB300 references:
`gpu-ref --merge RUNS_DIR` (RUNS_DIR/runs/<row key>.json, errdist JSON with --gpu-trials).

Alignment rule (thresholds from errdist's own checks): a row's Tier 2 is
aligned with GB300 when |P_T2 - P_GPU| <= max(5% of P_GPU, half the GB300 95% interval) and
the total variation distance between the Tier 2 PMF and the GB300 histogram, binned into signed
log2 ulp buckets (0, +-1, +-2-3, +-4-7, ...), is at most 0.05. KS is on the integer CDF.
The primary Tier 2 estimate of a row is the one errdist's policy would pick: the first
applicable tier-2 estimate, else the first tier-2 estimate.
"""

import argparse
import concurrent.futures as cf
import glob
import json
import os
import subprocess
import sys
import time

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.environ.get("PTXV_SCORECARD_DATA")

# (pair, case, spec, label, kernel, mc): the refresh's base rows.
ROWS = [
    ("softmax", "softmax_naive", "normal", "N(0, 1)", "naive", 20000),
    ("softmax", "softmax_stable", "normal", "N(0, 1)", "stable", 20000),
    ("softmax", "softmax_naive", "shift50", "N(50, 1)", "naive", 20000),
    ("softmax", "softmax_stable", "shift50", "N(50, 1)", "stable", 20000),
    ("softmax", "softmax_naive", "shift88", "N(88, 1)", "naive", 20000),
    ("softmax", "softmax_stable", "shift88", "N(88, 1)", "stable", 20000),
    ("softmax", "softmax_naive", "shift100", "N(100, 1)", "naive", 20000),
    ("softmax", "softmax_stable", "shift100", "N(100, 1)", "stable", 20000),
    ("variance", "var_naive", "normal", "N(0, 1)", "one pass", 20000),
    ("variance", "var_twopass", "normal", "N(0, 1)", "two pass", 20000),
    ("variance", "var_naive", "shift10", "N(10, 1)", "one pass", 20000),
    ("variance", "var_twopass", "shift10", "N(10, 1)", "two pass", 20000),
    ("variance", "var_naive", "shift100", "N(100, 1)", "one pass", 20000),
    ("variance", "var_twopass", "shift100", "N(100, 1)", "two pass", 20000),
    ("dot", "dot_bf16acc", "normal", "N(0, 1)", "bf16", 20000),
    ("dot", "dot_f32acc", "normal", "N(0, 1)", "f32", 400000),
    ("dot", "dot_bf16acc", "unif01", "U(0, 1)", "bf16", 20000),
    ("dot", "dot_f32acc", "unif01", "U(0, 1)", "f32", 400000),
    ("sum", "sum_naive", "unif01", "U(0, 1)", "sequential", 20000),
    ("sum", "sum_kahan", "unif01", "U(0, 1)", "Kahan", 100000),
    ("sum", "sum_naive", "normal", "N(0, 1)", "sequential", 20000),
    ("sum", "sum_kahan", "normal", "N(0, 1)", "Kahan", 20000),
    ("lse", "lse_naive", "normal", "N(0, 1)", "naive", 20000),
    ("lse", "lse_stable", "normal", "N(0, 1)", "stable", 20000),
    # Tracked linearized-tier gaps (refused by the Tier 3 cross-check, not misses): repo fixtures
    # under fixtures/errdist/lin_gap, specs beside them.
    (
        "lin_gap",
        "lin_gap/sel",
        "lin_gap/normal",
        "x, w, v ~ N(0, 1)",
        "1a: sel(RN(RN(x w) + v), fma)",
        20000,
    ),
    (
        "lin_gap",
        "lin_gap/sel",
        "lin_gap/v16",
        "v ~ N(0, 16^2)",
        "1b: sel(RN(RN(x w) + v), fma)",
        20000,
    ),
    (
        "lin_gap",
        "lin_gap/plain",
        "lin_gap/normal",
        "x, w, v ~ N(0, 1)",
        "1c: RN(RN(x w) + v)",
        20000,
    ),
    (
        "lin_gap",
        "lin_gap/plain",
        "lin_gap/v16",
        "v ~ N(0, 16^2)",
        "1d: RN(RN(x w) + v)",
        20000,
    ),
    (
        "lin_gap",
        "lin_gap/mix",
        "lin_gap/vmix",
        "v = 1 (95%) else N(0, 1)",
        "2a: sel(RN(RN(x w) v), RN(x RN(w v)))",
        20000,
    ),
    (
        "lin_gap",
        "lin_gap/mix",
        "lin_gap/normal",
        "x, w, v ~ N(0, 1)",
        "2b control",
        20000,
    ),
    (
        "lin_gap",
        "lin_gap/exact",
        "lin_gap/exact95",
        "v = 1 (95%) else 2",
        "2c: RN(x w) v, exact second op",
        20000,
    ),
]
# Rows whose case and spec live in the repo (fixtures/errdist/...), not in the data directory.
REPO = os.path.join(R, "fixtures", "errdist")


def row_key(case, spec):
    return f"{case.replace('/', '_')}.{os.path.basename(spec)}"


def case_path(a, case):
    return (
        os.path.join(REPO, case)
        if case.startswith("lin_gap/")
        else os.path.join(a.data, "cases", case)
    )


def spec_path(a, spec):
    return (
        os.path.join(REPO, spec[len("specs/") :])
        if spec.startswith("specs/lin_gap/")
        else os.path.join(a.data, spec)
    )


SWEEP_MC = 5000
# A stored GB300 PMF larger than this keeps only its log2 buckets (KS not available).
MAX_PMF = 200000


# ------------------------------------------------------------------ metrics
def pmf_dict(p):
    d = {}
    for k, q in p or []:
        d[int(k)] = d.get(int(k), 0.0) + q
    t = sum(d.values())
    return {k: v / t for k, v in d.items()} if t > 0 else {}


def bucket(d):
    """Signed log2 buckets: 0, +-1, +-[2,3], +-[4,7], ..."""
    o = {}
    for k, v in d.items():
        b = 0 if k == 0 else (1 if k > 0 else -1) * abs(k).bit_length()
        o[b] = o.get(b, 0.0) + v
    return o


def tv(a, b):
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def ks(a, b):
    ca = cb = m = 0.0
    for k in sorted(set(a) | set(b)):
        ca += a.get(k, 0.0)
        cb += b.get(k, 0.0)
        m = max(m, abs(ca - cb))
    return m


def abs_quantile(d, q):
    a = {}
    for k, v in d.items():
        a[abs(k)] = a.get(abs(k), 0.0) + v
    c = 0.0
    for m in sorted(a):
        c += a[m]
        if c >= q - 1e-12:
            return m
    return max(a) if a else None


def aligned(p2, gpu_p, tvb):
    """The refresh's rule. gpu_p = [p, lo, hi]."""
    if p2 is None or gpu_p is None or tvb is None:
        return None
    pg, lo, hi = gpu_p
    return abs(p2 - pg) <= max(0.05 * pg, (hi - lo) / 2) and tvb <= 0.05


# ------------------------------------------------------------------ GB300 reference
def gpu_entry(d):
    gpu = next(
        (
            v
            for v in d.get("validation", [])
            if v["source"] == "gpu" and v["against"] == "tier 3"
        ),
        None,
    )
    if gpu is None:
        return None
    e = dict(
        inputs=gpu["inputs"],
        outputs=gpu["outputs"],
        p=gpu["p_differ_obs"] if gpu["outputs"] else None,
        mismatch=gpu.get("dag_mismatch"),
        p_nonfinite=d.get("p_nonfinite"),
    )
    pmf = gpu.get("pmf_ulp") or []
    if len(pmf) > MAX_PMF:
        e["buckets"] = sorted(bucket(pmf_dict(pmf)).items())
    else:
        e["pmf"] = pmf
    return e


def cmd_gpu_ref(a):
    out = os.path.join(a.data, "gpu_ref.json")
    ref = {"rows": {}, "sweep": {}, "source": os.path.abspath(a.runs)}
    if a.merge and os.path.exists(out):
        ref = json.load(open(out))
        ref["source"] = ref.get("source", "") + " + " + os.path.abspath(a.runs)
    for r in ROWS:
        k = row_key(r[1], r[2])
        f = os.path.join(a.runs, "runs", f"{k}.json")
        if os.path.exists(f):
            e = gpu_entry(json.load(open(f)))
            if e:
                ref["rows"][k] = e
    for f in sorted(glob.glob(os.path.join(a.runs, "sweep", "*.json"))):
        e = gpu_entry(json.load(open(f)))
        if e:
            ref["sweep"][os.path.basename(f)[:-5]] = e
    json.dump(ref, open(out, "w"))
    print(f"{out}: {len(ref['rows'])} rows, {len(ref['sweep'])} sweep points")


# ------------------------------------------------------------------ runs
def run_one(job, a, outdir):
    key, case, spec, mc = job
    js = os.path.join(outdir, "json", key + ".json")
    if os.path.exists(js) and not a.force:
        return key, 0, 0.0
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PTXV_THREADS=str(a.threads))
    if a.mode:
        env["PTXV_ERRDIST_LIN_TRAJ"] = a.mode
    cmd = [
        a.bin,
        "errdist",
        case_path(a, case),
        "--inputs",
        spec_path(a, spec),
        "--mc",
        str(mc),
        "--gpu-trials",
        "0",
        "--threads",
        str(a.threads),
        "--json",
        js,
    ]
    if a.heavy:
        cmd = [a.heavy, "-m", "32G"] + cmd
    t = time.time()
    with open(os.path.join(outdir, "txt", key + ".txt"), "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    return key, rc, time.time() - t


def score(d, g):
    """One row / point: GB300 reference g against the run's JSON d."""
    est = d["estimates"] if d else []
    t3 = next((e for e in est if e["tier"] == 3), None)
    t2s = [e for e in est if e["tier"] == 2 and e.get("p_differ")]
    t2 = next((e for e in t2s if e.get("applicable")), t2s[0] if t2s else None)
    G = pmf_dict(g.get("pmf")) if g and g.get("pmf") else None
    GB = (
        dict((int(k), v) for k, v in g["buckets"])
        if g and g.get("buckets")
        else (bucket(G) if G else None)
    )
    out = dict(
        gpu_p=g.get("p") if g else None,
        gpu_inputs=g.get("inputs") if g else None,
        gpu_q999=abs_quantile(G, 0.999) if G else None,
        t3_p=t3["p_differ"] if t3 else None,
        best_tier=est[d["best"]]["tier"]
        if d and est and d.get("best") is not None
        else None,
        t2=[],
    )
    if t3 and GB:
        T3 = pmf_dict(t3.get("pmf_ulp"))
        if T3:
            out["t3_tvb"] = tv(bucket(T3), GB)
            out["t3_ks"] = ks(T3, G) if G else None
            out["t3_q999"] = abs_quantile(T3, 0.999)
            out["t3_aligned"] = aligned(t3["p_differ"][0], out["gpu_p"], out["t3_tvb"])
    for e in t2s:
        T2 = pmf_dict(e.get("pmf_ulp"))
        x = dict(
            method=e["method"],
            p=e["p_differ"],
            applicable=e.get("applicable"),
            primary=e is t2,
            q999=abs_quantile(T2, 0.999) if T2 else None,
        )
        if T2 and GB:
            x["tvb"] = tv(bucket(T2), GB)
            x["ks"] = ks(T2, G) if G else None
            x["aligned"] = aligned(e["p_differ"][0], out["gpu_p"], x["tvb"])
        out["t2"].append(x)
    return out


def cmd_run(a):
    ref = json.load(open(os.path.join(a.data, "gpu_ref.json")))
    outdir = os.path.join(a.data, "scorecards", a.name)
    os.makedirs(os.path.join(outdir, "json"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "txt"), exist_ok=True)
    jobs = [(row_key(r[1], r[2]), r[1], f"specs/{r[2]}.json", r[5]) for r in ROWS]
    if a.sweeps:
        for k in sorted(ref["sweep"]):
            case, name = k.split(".", 1)
            jobs.append((k, case, f"specs/sweep/{name}.json", SWEEP_MC))
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for key, rc, dt in ex.map(lambda j: run_one(j, a, outdir), jobs):
            if rc:
                print(f"  {key}: exit {rc} ({dt:.0f}s)", file=sys.stderr)
    sc = {
        "name": a.name,
        "mode": a.mode or "real",
        "bin": a.bin,
        "seconds": round(time.time() - t0),
        "rows": {},
        "sweep": {},
    }
    for key, case, spec, mc in jobs:
        js = os.path.join(outdir, "json", key + ".json")
        d = json.load(open(js)) if os.path.exists(js) else None
        sect = "sweep" if key in ref["sweep"] and key not in ref["rows"] else "rows"
        g = ref[sect].get(key)
        s = (
            score(d, g)
            if d
            else {
                "error": open(os.path.join(outdir, "txt", key + ".txt"))
                .read()
                .strip()
                .splitlines()[-1:]
            }
        )
        sc[sect][key] = s
    sc["summary"] = summarize(sc)
    json.dump(
        sc, open(os.path.join(a.data, "scorecards", a.name + ".json"), "w"), indent=1
    )
    md = render(sc, load(a.data, a.baseline) if a.baseline else None)
    open(os.path.join(a.data, "scorecards", a.name + ".md"), "w").write(md)
    print(md)


def primary(s):
    return next((x for x in s.get("t2", []) if x.get("primary")), None)


def summarize(sc):
    def count(sect):
        # Tracked gap rows are listed apart and left out of the headline counts.
        rows = [
            s
            for k, s in sc[sect].items()
            if "error" not in s and not k.startswith("lin_gap_")
        ]
        cmp_ = [s for s in rows if primary(s) and primary(s).get("aligned") is not None]
        return dict(
            n=len(cmp_),
            aligned=sum(1 for s in cmp_ if primary(s)["aligned"]),
            accepted=sum(1 for s in cmp_ if primary(s)["applicable"]),
            accepted_not_aligned=sum(
                1
                for s in cmp_
                if primary(s)["applicable"] and not primary(s)["aligned"]
            ),
            t3_aligned=sum(1 for s in rows if s.get("t3_aligned")),
            t3_n=sum(1 for s in rows if s.get("t3_aligned") is not None),
        )

    return {"rows": count("rows"), "sweep": count("sweep")}


def load(data, name):
    return json.load(open(os.path.join(data, "scorecards", name + ".json")))


def f(v, d=4):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if v != 0 and abs(v) < 1e-3:
        return f"{v:.2e}"
    return f"{v:.{d}f}"


def render(sc, base=None):
    L = [
        f"# errdist scorecard `{sc['name']}` (linearization trajectory: {sc['mode']})",
        "",
    ]
    for sect in ("rows", "sweep"):
        s = sc["summary"][sect]
        if not s["n"] and sect == "sweep":
            continue
        line = (
            f"- {sect}: Tier 2 aligned **{s['aligned']} / {s['n']}**; own check accepts {s['accepted']}, "
            f"of which {s['accepted_not_aligned']} not aligned; Tier 3 aligned {s['t3_aligned']} / {s['t3_n']}"
        )
        if base and base["summary"].get(sect, {}).get("n"):
            b = base["summary"][sect]
            line += f" (baseline `{base['name']}`: {b['aligned']} / {b['n']}, accepted-not-aligned {b['accepted_not_aligned']})"
        L.append(line)
    L += [
        "",
        "| row | GB300 P [95%] | Tier 3 | Tier 2 (check) | 99.9% GPU / T2 | TVb | KS | aligned |"
        + (" ΔTVb | ΔKS | Δ|relP| | was |" if base else ""),
        "|---|---|---|---|---|---|---|---|" + ("---|---|---|---|" if base else ""),
    ]
    for key, s in sc["rows"].items():
        if "error" in s:
            L.append(f"| {key} | refused: {' '.join(s['error'])[:120]} |")
            continue
        t2 = primary(s) or {}
        g = s.get("gpu_p")
        gp = f"{f(g[0])} [{f(g[1])}, {f(g[2])}]" if g else "no finite outputs"
        rel = abs(t2["p"][0] - g[0]) / g[0] if t2.get("p") and g and g[0] else None
        row = (
            f"| {key} | {gp} | {f(s['t3_p'][0]) if s.get('t3_p') else '–'} | "
            f"{f(t2['p'][0]) if t2.get('p') else '–'} ({'acc' if t2.get('applicable') else ('rej' if t2 else '–')}) | "
            f"{f(s.get('gpu_q999'))} / {f(t2.get('q999'))} | {f(t2.get('tvb'))} | {f(t2.get('ks'))} | {f(t2.get('aligned'))} |"
        )
        if base:
            bs = base["rows"].get(key) or {}
            bt = primary(bs) or {}
            bg = bs.get("gpu_p")
            brel = (
                abs(bt["p"][0] - bg[0]) / bg[0]
                if bt.get("p") and bg and bg[0]
                else None
            )
            dd = (
                lambda x, y: f"{x - y:+.4f}" if x is not None and y is not None else "–"
            )
            row += f" {dd(t2.get('tvb'), bt.get('tvb'))} | {dd(t2.get('ks'), bt.get('ks'))} | {dd(rel, brel)} | {f(bt.get('aligned'))} |"
        L.append(row)
    tracked = [
        (k, s)
        for k, s in sc["rows"].items()
        if k.startswith("lin_gap_") and "error" not in s
    ]
    if tracked:
        L += [
            "",
            "Tracked linearized-tier gaps (expected to be refused by the cross-check):",
            "",
            "| row | Tier 3 | linearized Tier 2 (check) | T2 / T3 | DAG CLT |",
            "|---|---|---|---|---|",
        ]
        for key, s in tracked:
            t3 = (s.get("t3_p") or [None])[0]
            lin = next(
                (x for x in s["t2"] if x["method"].startswith("linearized")), None
            )
            clt = next((x for x in s["t2"] if x["method"].startswith("CLT")), None)
            r = lin["p"][0] / t3 if lin and t3 else None
            L.append(
                f"| {key} | {f(t3)} | {f(lin['p'][0]) if lin else '–'} ({'acc' if lin and lin.get('applicable') else 'rej'}) | "
                f"{f(r, 3)} | {f(clt['p'][0]) if clt else '–'} |"
            )
    if sc["sweep"]:
        L += [
            "",
            "Sweep points whose alignment changed against the baseline:"
            if base
            else "Sweep points not aligned:",
            "",
        ]
        for key, s in sc["sweep"].items():
            t2 = primary(s) or {}
            if base:
                bt = primary(base["sweep"].get(key) or {}) or {}
                if t2.get("aligned") == bt.get("aligned"):
                    continue
                L.append(
                    f"- {key}: {f(bt.get('aligned'))} -> {f(t2.get('aligned'))} (T2 {f((t2.get('p') or [None])[0])}, GPU {f((s.get('gpu_p') or [None])[0])}, TVb {f(t2.get('tvb'))})"
                )
            elif t2.get("aligned") is False:
                L.append(
                    f"- {key}: T2 {f(t2['p'][0])} ({'acc' if t2.get('applicable') else 'rej'}), GPU {f(s['gpu_p'][0])}, TVb {f(t2.get('tvb'))}"
                )
    return "\n".join(L) + "\n"


def cmd_diff(a):
    print(render(load(a.data, a.name), load(a.data, a.baseline)))


# ------------------------------------------------------------------ self test
def cmd_selftest(_a=None):
    a = pmf_dict([[0, 0.5], [1, 0.25], [-1, 0.25]])
    b = pmf_dict([[0, 0.5], [1, 0.5]])
    assert abs(tv(a, b) - 0.25) < 1e-12
    assert abs(ks(a, b) - 0.25) < 1e-12
    assert bucket({5: 1.0, -3: 1.0, 0: 1.0, 1: 1.0}) == {
        3: 1.0,
        -2: 1.0,
        0: 1.0,
        1: 1.0,
    }
    assert abs_quantile(a, 0.5) == 0 and abs_quantile(a, 0.75) == 1
    # the rule: 5% relative, or inside the GB300 interval, and TVb <= 0.05
    assert aligned(0.96, [1.0, 0.99, 1.0], 0.01) is True
    assert aligned(0.94, [1.0, 0.99, 1.0], 0.01) is False
    assert aligned(0.5, [1.0, 0.0, 1.0], 0.01) is True  # wide interval
    assert aligned(1.0, [1.0, 0.99, 1.0], 0.06) is False
    # score(): primary is the first applicable tier-2 estimate
    d = {
        "best": 0,
        "estimates": [
            {"tier": 3, "p_differ": [0.5, 0.49, 0.51], "pmf_ulp": [[0, 0.5], [1, 0.5]]},
            {
                "tier": 2,
                "method": "a",
                "p_differ": [0.3, 0.3, 0.3],
                "applicable": False,
                "pmf_ulp": [[0, 0.7], [1, 0.3]],
            },
            {
                "tier": 2,
                "method": "b",
                "p_differ": [0.5, 0.5, 0.5],
                "applicable": True,
                "pmf_ulp": [[0, 0.5], [1, 0.5]],
            },
        ],
    }
    g = {"p": [0.5, 0.49, 0.51], "pmf": [[0, 0.5], [1, 0.5]], "inputs": 10}
    s = score(d, g)
    assert (
        primary(s)["method"] == "b"
        and primary(s)["aligned"] is True
        and s["t3_aligned"] is True
    )
    assert s["t2"][0]["aligned"] is False and abs(s["t2"][0]["tvb"] - 0.2) < 1e-12
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="data directory (default: $PTXV_SCORECARD_DATA)",
    )
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gpu-ref")
    g.add_argument(
        "runs",
        help="directory with runs/ and sweep/ errdist JSON files measured with --gpu-trials",
    )
    g.add_argument(
        "--merge", action="store_true", help="keep the entries already in gpu_ref.json"
    )
    r = sp.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument(
        "--mode",
        choices=["real", "float", "both"],
        default=None,
        help="PTXV_ERRDIST_LIN_TRAJ",
    )
    r.add_argument("--sweeps", action="store_true")
    r.add_argument("--baseline")
    r.add_argument(
        "--bin",
        default=os.environ.get("PTXV_BIN", os.path.join(R, "target/release/ptxv")),
    )
    r.add_argument("--jobs", type=int, default=4)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument(
        "--heavy",
        default=os.environ.get("PTXV_HEAVY"),
        help="wrapper for each run (e.g. a slot limiter)",
    )
    r.add_argument("--force", action="store_true", help="rerun rows whose JSON exists")
    d = sp.add_parser("diff")
    d.add_argument("name")
    d.add_argument("baseline")
    sp.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd != "selftest" and not a.data:
        ap.error(
            "no data directory: pass --data or set PTXV_SCORECARD_DATA (shared location: fb/errdist_scorecard.md)"
        )
    {
        "gpu-ref": cmd_gpu_ref,
        "run": cmd_run,
        "diff": cmd_diff,
        "selftest": cmd_selftest,
    }[a.cmd](a)


if __name__ == "__main__":
    main()

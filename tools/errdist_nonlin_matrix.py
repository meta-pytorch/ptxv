#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""errdist non-chain validation matrix: analytic tiers vs tier 3 (agent AG)."""

import json
import os
import subprocess
import sys
import time

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.environ.get("PTXV_BIN", os.path.join(R, "target/release/ptxv"))
S = os.path.join(R, "fixtures/errdist")
SPECS = {
    "normal": f"{R}/fixtures/errdist/normal.json",
    "uniform": f"{S}/uniform.json",
    "laplace": f"{S}/laplace.json",
    "student3": f"{S}/student3.json",
    "shift100": f"{S}/shifted.json",
    "normal8": f"{S}/big.json",
    "int8": f"{R}/fixtures/errdist/integers.json",
}
SINGLE = [
    "r4_softmax_torch",
    "r4_softmax_triton",
    "c_triton_softmax_online",
    "r4_rmsnorm_torch",
    "r4_rmsnorm_triton",
    "r4_layernorm_torch",
    "r4_layernorm_triton",
    "c_attn_core_ref",
    "c_silu",
    "c_gelu_tanh",
    "c_gelu_erf",
    "c_gelu_bf16",
    "c_triton_gelu_bf16",
    "c_softcap",
    "c_cve_silumul_bf16_inductor",
    "c_cve_addnorm_inductor",
]
PAIRS = [
    ("c_equiv_softmax/equiv.toml", 0),
    ("c_equiv_softmax/equiv.toml", 1),
    ("c_equiv_rmsnorm/equiv.toml", 0),
    ("c_equiv_layernorm/equiv.toml", 0),
    ("c_equiv_attn/equiv.toml", 0),
    ("c_equiv_elementwise/gelu_tanh.toml", 0),
    ("c_equiv_elementwise/silu.toml", 0),
    ("c_equiv_elementwise/gelu_bf16.toml", 0),
    ("c_equiv_elementwise/gelu_erf.toml", 0),
    ("errdist/cve/addnorm.toml", 0),
    ("errdist/cve/silumul.toml", 0),
    ("errdist/cve/add_bf16.toml", 0),
    ("errdist/cve/silumul_bf16.toml", 0),
]
# FAST=1: one single kernel and one pair per family (softmax, norm, elementwise) on the
# normal spec at 4000 inputs; exits 1 when a headline analytic estimate misses Tier 3.
FAST = os.environ.get("FAST") == "1"
FAST_SINGLE = ["r4_softmax_torch", "r4_rmsnorm_torch", "c_gelu_tanh"]
FAST_PAIRS = [("c_equiv_softmax/equiv.toml", 0), ("c_equiv_elementwise/silu.toml", 0)]
out = sys.argv[1]
specs = (
    sys.argv[2].split(",")
    if len(sys.argv) > 2
    else (["normal"] if FAST else list(SPECS))
)
mc = int(os.environ.get("MC", "4000" if FAST else "20000"))
jobs = (
    [(k, None) for k in FAST_SINGLE] + FAST_PAIRS
    if FAST
    else [(k, None) for k in SINGLE] + list(PAIRS)
)
bad = []
for sp in specs:
    for name, cand in jobs:
        js = f"/tmp/agAG_matrix_{os.getpid()}.json"
        cmd = [
            P,
            "errdist",
            f"{R}/fixtures/{name}",
            "--inputs",
            SPECS[sp],
            "--mc",
            str(mc),
            "--sim-trials",
            "0",
            "--json",
            js,
        ]
        if cand is not None:
            cmd += ["--candidate", str(cand)]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        row = {
            "case": name + ("" if cand is None else f":{cand}"),
            "spec": sp,
            "wall": round(time.time() - t0, 1),
        }
        try:
            rep = json.load(open(js))
            os.unlink(js)
            row["best"] = rep["best"]
            row["est"] = [
                {
                    "tier": e["tier"],
                    "kind": e["method"].split(":")[0].split(" (")[0][:30],
                    "p": e.get("p_differ"),
                    "s": round(e.get("seconds", 0), 2),
                }
                for e in rep["estimates"]
            ]
            row["notes"] = [
                n for n in rep.get("notes", []) if "not run" in n or "does not" in n
            ][:4]
        except Exception:
            row["error"] = (r.stderr or r.stdout)[-300:]
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row)[:300], flush=True)
        # The headline must be Tier 0 / 3, or an analytic estimate within tolerance of Tier 3.
        if "error" in row:
            bad.append(row["case"])
        elif FAST:
            est = row["est"]
            b = est[row["best"]]
            t3 = next((e for e in est if e["tier"] == 3), None)
            if b["tier"] in (1, 2) and t3 and t3["p"] and b["p"]:
                p3, lo, hi = t3["p"]
                if (
                    abs(b["p"][0] - p3) > max(0.05 * p3, 0.5 * (hi - lo))
                    and not lo <= b["p"][0] <= hi
                ):
                    bad.append(row["case"])
if FAST:
    print(
        "FAIL: " + ", ".join(bad)
        if bad
        else "fast subset: every headline within tolerance"
    )
    sys.exit(1 if bad else 0)

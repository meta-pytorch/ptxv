# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The only module that knows `ptxv errdist`'s formats (agent T, docs/errdist.md).

- `t_spec`: `ptxv.dist` distributions (by input buffer) -> the input spec, version 1 (§2).
- `argv`: the command line (`ptxv errdist <case | equiv.toml> --inputs spec.json --json out`).
- `to_errdist`: the report, version 1 (§3) -> `ErrDist`.

If the CLI or its formats change, only this file changes.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

SPEC_VERSION = 1
REPORT_VERSION = 1


class AdapterError(ValueError):
    """A distribution ptxv errdist's spec cannot express."""


def _dist(s: dict, work: Path, name: str) -> dict:
    """One element distribution (our spec dict -> T's `dist`)."""
    k = s["kind"]
    if k == "normal":
        return {"kind": "normal", "mean": s["mu"], "std": s["sigma"]}
    if k == "uniform":
        return {"kind": "uniform", "lo": s["lo"], "hi": s["hi"]}
    if k == "laplace":
        return {"kind": "laplace", "mean": s["mu"], "scale": s["b"]}
    if k == "student_t":
        return {
            "kind": "student_t",
            "df": s["df"],
            "mean": s["mu"],
            "scale": s["scale"],
        }
    if k == "lognormal":
        return {"kind": "log_normal", "mu": s["mu"], "sigma": s["sigma"]}
    if k == "const":
        return {"kind": "const", "value": s["value"]}
    if k == "discrete":
        return {"kind": "discrete", "values": s["values"], "probs": s["probs"]}
    if k == "integers":  # ours: [lo, hi); T's: lo..=hi
        return {"kind": "integers", "lo": s["lo"], "hi": s["hi"] - 1}
    if k == "mixture":
        return {
            "kind": "mixture",
            "components": [
                {"weight": w, "dist": _dist(c, work, name)}
                for w, c in zip(s["weights"], s["components"])
            ],
        }
    if k == "affine":
        return {
            "kind": "affine",
            "a": s["scale"],
            "b": s["shift"],
            "dist": _dist(s["base"], work, name),
        }
    if k == "empirical":
        raw = base64.b64decode(s["f32_base64"])
        f = work / f"empirical_{name}_{hashlib.sha1(raw).hexdigest()[:12]}.f32"
        f.write_bytes(raw)
        return {"kind": "empirical", "file": f.name, "dtype": "f32"}
    raise AdapterError(
        f"{name}: {k} is not expressible in ptxv errdist's element distributions"
    )


def _quant(s: dict, name: str) -> dict:
    r = s["recipe"]
    if r in ("bf16", "f16"):
        return {"kind": "round"}
    if r in ("e4m3", "e5m2", "e2m1"):
        return {"kind": "cast", "format": r}
    if r in ("mxf8", "mxf8_e5m2", "mxfp4"):
        elem = {"mxf8": "e4m3", "mxf8_e5m2": "e5m2", "mxfp4": "e2m1"}[r]
        q = {
            "kind": "mx",
            "elem": elem,
            "block": s["block"],
            "scale_rule": s.get("scale_rule", "floor"),
        }
        if s.get("scales"):
            q["scales"] = s["scales"]
        return q
    if r == "nvfp4":
        q = {
            "kind": "nvfp4",
            "block": s["block"],
            "global_scale": bool(s.get("global_scale", True)),
        }
        if s.get("scales"):
            q["scales"] = s["scales"]
        return q
    raise AdapterError(
        f"{name}: recipe {r!r} has no ptxv errdist equivalent (per-tensor fp8 scales, mxf6)"
    )


def tensor(s: dict, work: Path, name: str) -> dict:
    """Our spec of one input -> T's tensor entry ({dist, scales, outliers, quant}).

    Wrappers apply outermost first: quantize(outliers(per_channel(base))) and any order of
    per_channel / outliers below quantize."""
    out: dict[str, Any] = {}
    if s["kind"] == "quantize":
        out["quant"] = _quant(s, name)
        s = s["base"]
    scales = []
    while s["kind"] in ("per_channel", "outliers"):
        if s["kind"] == "per_channel":
            scales.append({"axis": s["axis"], "dist": _dist(s["scale"], work, name)})
        else:
            if "outliers" in out:
                raise AdapterError(f"{name}: one outliers wrapper per tensor")
            out["outliers"] = {
                "axis": s["axis"],
                "prob": s["prob"],
                "factor": s["factor"],
            }
        s = s["base"]
    if s["kind"] in ("quantize", "clip"):
        raise AdapterError(
            f"{name}: {s['kind']} must be the outermost wrapper (clip: unsupported)"
        )
    if scales:
        out["scales"] = scales[::-1]
    out["dist"] = _dist(s, work, name)
    return out


def t_spec(args: dict, work: Path) -> dict:
    """Job arguments -> input spec version 1."""
    spec: dict[str, Any] = {
        "version": SPEC_VERSION,
        "seed": int(args.get("seed", 1)),
        "tensors": {
            n: tensor(d, work, n) for n, d in (args.get("inputs") or {}).items()
        },
    }
    if args.get("name"):
        spec["name"] = args["name"]
    if args.get("default"):
        spec["default"] = tensor(args["default"], work, "default")
    eng = {
        k: args[k]
        for k in ("mc_inputs", "sim_trials", "gpu_trials", "per_output")
        if args.get(k) is not None
    }
    if eng:
        spec["engine"] = eng
    return spec


def argv(job: Any, work: Path) -> list[str]:
    """The ptxv errdist command for a job (cases: [candidate] or [reference, candidate])."""
    work.mkdir(parents=True, exist_ok=True)
    spec = t_spec(job.args, work)
    sp = work / "spec.json"
    sp.write_text(json.dumps(spec, indent=1))
    if len(job.cases) == 1:
        target = job.cases[0]
    else:
        ref, cand = job.cases
        t = work / "equiv.toml"
        t.write_text(
            f'reference = {{ case = "{ref}", variant = "good", front = "sass", certify = true }}\n\n'
            f'[[candidates]]\ncase = "{cand}"\nvariant = "good"\nfront = "sass"\ncertify = true\n'
        )
        target = str(t)
    cmd = ["errdist", target, "--inputs", str(sp), "--json", str(work / "errdist.json")]
    if job.args.get("per_output"):
        cmd.append("--per-output")
    # Tier 1 can take minutes on wide scale ranges: opt in with tier1=True (or use
    # ptxv.errdist_model, which compiles once and queries in seconds).
    if not job.args.get("tier1"):
        cmd.append("--no-tier1")
    return cmd


def equiv_toml(ref: str, cand: str, work: Path) -> Path:
    """A one-candidate equiv.toml for a pair."""
    work.mkdir(parents=True, exist_ok=True)
    t = work / "equiv.toml"
    t.write_text(
        f'reference = {{ case = "{ref}", variant = "good", front = "sass", certify = true }}\n\n'
        f'[[candidates]]\ncase = "{cand}"\nvariant = "good"\nfront = "sass"\ncertify = true\n'
    )
    return t


def compile_argv(target: str, out: Path) -> list[str]:
    """Compile stage: simulate once, write the model (docs/errdist.md §1.1)."""
    return ["errdist", target, "--compile", str(out)]


def query_argv(model: Path, args: dict, work: Path) -> list[str]:
    """Query stage: a compiled model under an input distribution (job-style args)."""
    work.mkdir(parents=True, exist_ok=True)
    sp = work / "spec.json"
    sp.write_text(json.dumps(t_spec(args, work), indent=1))
    cmd = [
        "errdist",
        "--model",
        str(model),
        "--inputs",
        str(sp),
        "--json",
        str(work / DOC_FILE),
    ]
    if args.get("mc_inputs"):
        cmd += ["--mc", str(int(args["mc_inputs"]))]
    if args.get("per_output"):
        cmd.append("--per-output")
    if not args.get("tier1", True):
        cmd.append("--no-tier1")
    return cmd


DOC_FILE = "errdist.json"


def to_errdist(doc: dict, metric: str = "abs"):
    """Report version 1 -> ErrDist (the `best` estimate's moments and quantiles)."""
    from .results import ErrDist

    if doc.get("kind") != "ptxv-errdist" or doc.get("version") != REPORT_VERSION:
        raise ValueError(f"not a ptxv-errdist v{REPORT_VERSION} report")
    est = doc["estimates"][doc.get("best", 0)]
    key = {"abs": "abs", "u": "u_scale", "ulp": "ulp"}[metric]
    m = est.get(key)
    if m is None:
        raise ValueError(f"the report's best estimate has no {key!r} moments")
    hist = est.get("hist_u") if metric == "u" else None
    return ErrDist.from_summary(
        quantiles=[tuple(q) for q in m.get("quantiles", [])],
        moments=m,
        metric=metric,
        source=f"ptxv errdist tier {est['tier']} ({est['method']})",
        n=est.get("n", 0),
        hist=hist,
        meta={
            "mode": doc.get("mode"),
            "quantity": doc.get("quantity"),
            "out_of_domain": doc.get("out_of_domain"),
            "p_differ": est.get("p_differ"),
            "error_bar": est.get("error_bar"),
            "validation": doc.get("validation"),
        },
        report=doc,
    )

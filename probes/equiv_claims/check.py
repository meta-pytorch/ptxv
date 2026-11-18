# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GB300's recorded outputs against `ptxv equiv`'s claims (README.md).

    python3 -I check.py <probe dir> [--ptxv PTXV] [--hw RUNS]

<probe dir>/pairs.toml lists the pairs. Each names its equiv.toml files (one per front; ptxv's
claim is taken from a copy of the probe with every captured output removed, so the claim is the
model's proof for every input) and its input sets: per set, the reference and candidate case
whose expected/out.bin GB300 recorded. The kernels are elementwise (out[i] from the inputs'
element i), so an output is admissible under N-FINITE when its input elements are finite.

Per output and claimed level:
  L0             the two GB300 words are equal;
  L0~0           equal, or +0 against -0;
  L1 / L2 / L3   `value = "equal"`: equal as values (+0 = -0; two NaNs agree), for pairs whose
                 kernels compute the same exact selection or rounding of their inputs;
                 `value = "bound"`: |C - R| <= k u S + A for the printed per-output bound
                 (k = bound_u, A = abs_extra, S = the `terms` expression), each kernel within
                 its own bound of the exact value (`exact`), and L3's tolerance tau u S + A.
A claim that relies on N-FINITE covers the admissible outputs only; any other claim covers every
output. A claim GB300 contradicts is a DEFECT: the exit status is 1 if any pair has one, 2 if a
run fails, else 0. With --hw RUNS (a GPU visible), every case is first rerun RUNS times through
`ptxv hw --no-num` against its expected/out.bin.
"""

import argparse
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import tomllib
from fractions import Fraction
from pathlib import Path

U = Fraction(1, 2**24)


def words(path):
    b = Path(path).read_bytes()
    return list(struct.unpack(f"<{len(b) // 4}I", b))


def f32(w):
    return struct.unpack("<f", struct.pack("<I", w))[0]


def inputs(case_dir):
    """Name -> words of the case's file-initialized f32 buffers."""
    c = tomllib.loads((case_dir / "case.toml").read_text())
    return {
        b["name"]: words(case_dir / b["file"])
        for b in c["buffers"]
        if b.get("init") == "file" and b.get("dtype") == "f32"
    }


def gb300(case_dir):
    return words(case_dir / "expected" / "out.bin")


def kind(r, c):
    """How two output words differ: None (equal), 'zero sign', 'NaN' or 'value'."""
    if r == c:
        return None
    if {r, c} == {0, 0x80000000}:
        return "zero sign"
    if math.isnan(f32(r)) or math.isnan(f32(c)):
        return "NaN"
    return "value"


def strip_captures(src, dst):
    """A copy of the probe without captured outputs: equiv then proves for every input."""
    shutil.copytree(src, dst)
    for t in dst.rglob("case.toml"):
        s = t.read_text()
        t.write_text(re.sub(r'(?m)^expect = "expected/[^"]*"\n', "", s))


def claim(ptxv, tree, toml, out):
    """ptxv equiv's claim for one equiv.toml: (equiv.json entry, log) or (None, log)."""
    out.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(
        [ptxv, "equiv", str(tree / toml), "--no-hw", "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=tree,
    )
    log = p.stdout + p.stderr
    (out / "equiv.log").write_text(log)
    j = out / "equiv.json"
    if not j.exists():
        return None, log
    return json.loads(j.read_text())[0], log


def relies_on_finite(c):
    notes = list(c.get("relies_on", [])) + list(c["report"].get("relies", []))
    return any("N-FINITE" in n for n in notes)


def describe(c):
    if c is None:
        return "ptxv equiv failed"
    if c["strongest"].startswith(("none", "contradicted")):
        return c["strongest"]
    s = f"{c['strongest']} {'PROVEN' if c['proven'] else '(not proven)'}"
    if c["proven"] and relies_on_finite(c):
        s += ", relies on N-FINITE"
    return s


def check_pair(probe, pair, cfg, c, sets):
    """DEFECT reasons for one claim against every input set."""
    if c is None or not c["proven"] or c["strongest"].startswith(("none", "contradicted")):
        return []
    rep = c["report"]
    per = {o["index"]: o for o in rep["per_output"]}
    l3 = next((lv for lv in rep["levels"] if lv["level"].startswith("L3")), None)
    tau = None
    if l3 is not None and l3["holds"]:
        tau = Fraction(float(re.search(r"\(([0-9.]+)u\)", l3["level"]).group(1)))
    finite_only = relies_on_finite(c)
    mode = cfg.get("value", "equal")
    bad = {}
    worst = {}

    def note(what, ratio=None):
        bad[what] = bad.get(what, 0) + 1
        if ratio is not None:
            worst[what] = max(worst.get(what, 0), ratio)

    for ref, cand in sets:
        rw, cw = gb300(probe / ref), gb300(probe / cand)
        ins = inputs(probe / ref)
        for i, (r, w) in enumerate(zip(rw, cw)):
            o = per.get(i)
            if o is None:
                continue
            xs = {k: f32(v[i]) for k, v in ins.items()}
            if finite_only and not all(math.isfinite(x) for x in xs.values()):
                continue
            lvl, d = o["level"], kind(r, w)
            if lvl == "L0":
                if d is not None:
                    note("L0 claimed: words differ")
                continue
            if lvl == "L0~0":
                if d not in (None, "zero sign"):
                    note("L0~0 claimed: words differ beyond the sign of zero")
                continue
            fr, fc = f32(r), f32(w)
            if math.isnan(fr) and math.isnan(fc):
                continue
            if mode == "equal":
                if fr != fc:
                    note("value-level claim: values differ")
                continue
            # mode == "bound"
            if math.isnan(fr) or math.isnan(fc) or math.isinf(fr) or math.isinf(fc):
                if fr != fc:
                    note("value-level claim: a non-finite output differs")
                continue
            env = {k: Fraction(x) for k, x in xs.items()}
            s = eval(cfg["terms"], {"abs": abs}, env)
            y = eval(cfg["exact"], {"abs": abs}, env)
            a = Fraction(o.get("abs_extra") or 0.0)
            diff = abs(Fraction(fc) - Fraction(fr))
            for what, k, dev in (
                ("bound", o.get("bound_u"), diff),
                ("reference's own bound", o.get("ref_u"), abs(Fraction(fr) - y)),
                ("candidate's own bound", o.get("cand_u"), abs(Fraction(fc) - y)),
            ):
                if k is None:
                    continue
                lim = Fraction(k) * U * s + a
                if dev > lim:
                    r = float(dev / lim) if lim else math.inf
                    note(f"{what} k u sum|terms| + A exceeded", r)
            if tau is not None:
                lim = tau * U * s + a
                if diff > lim:
                    note(f"L3 ({float(tau):g}u) exceeded", float(diff / lim) if lim else math.inf)
    out = []
    for what, n in bad.items():
        s = f"{what} at {n} outputs"
        if what in worst:
            s += f" (up to {worst[what]:.4g}x)"
        out.append(s)
    return out


def gb300_summary(probe, sets):
    parts = []
    for ref, cand in sets:
        rw, cw = gb300(probe / ref), gb300(probe / cand)
        kinds = [kind(r, c) for r, c in zip(rw, cw)]
        n = sum(k is not None for k in kinds)
        tag = ref.rsplit("_", 1)[-1]
        detail = ", ".join(
            f"{sum(k == x for k in kinds)} {x}"
            for x in ("zero sign", "NaN", "value")
            if any(k == x for k in kinds)
        )
        parts.append(f"{tag}: {n} of {len(rw)} words differ" + (f" ({detail})" if n else ""))
    return "; ".join(parts)


def rerun(ptxv, probe, runs):
    """Every case's GB300 rerun against its expected/out.bin: (cases reproduced, cases)."""
    cases = sorted(d for d in probe.iterdir() if (d / "expected" / "out.bin").exists())
    ok = 0
    for d in cases:
        p = subprocess.run(
            [ptxv, "hw", str(d), "--variant", "good", "--runs", str(runs), "--no-num"],
            capture_output=True,
            text=True,
        )
        m = re.search(r"distinct outputs \(count, words differing\): (\[.*\])", p.stdout)
        if m and re.findall(r"(\d+), (\d+)", m.group(1)) == [(str(runs), "0")]:
            ok += 1
        else:
            print(f"  GB300 rerun of {d.name}: {m.group(1) if m else (p.stdout + p.stderr)[-300:]}")
    return ok, len(cases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe")
    ap.add_argument("--ptxv", default="ptxv")
    ap.add_argument("--hw", type=int, default=0)
    args = ap.parse_args()
    probe = Path(args.probe).resolve()
    cfg = tomllib.loads((probe / "pairs.toml").read_text())
    print(f"{probe.parent.name}/{probe.name}: {cfg['title']}")
    if args.hw:
        ok, n = rerun(args.ptxv, probe, args.hw)
        print(f"GB300 rerun: {ok} of {n} cases reproduce expected/out.bin ({args.hw} runs each)")
        if ok != n:
            return 2
    else:
        print("GB300: the recorded expected/out.bin (no rerun)")
    defects = errors = 0
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "nocap"
        strip_captures(probe, tree)
        for pair in cfg["pairs"]:
            print(f"\n{pair['name']}")
            print(f"  GB300: {gb300_summary(probe, pair['sets'])}")
            for toml in pair["equiv"]:
                front = "sass" if "sass" in toml else "ptx"
                c, log = claim(args.ptxv, tree, toml, Path(tmp) / "out" / toml)
                if c is None:
                    errors += 1
                    print(f"  ptxv equiv ({front}): failed\n{log[-800:]}")
                    continue
                why = check_pair(probe, pair, cfg, c, pair["sets"])
                verdict = "DEFECT: " + "; ".join(why) if why else "consistent"
                defects += bool(why)
                print(f"  ptxv equiv ({front}): {describe(c)} -> {verdict}")
    print(f"\n{defects} claim(s) GB300 contradicts, {errors} failed run(s)")
    return 2 if errors else (1 if defects else 0)


if __name__ == "__main__":
    sys.exit(main())

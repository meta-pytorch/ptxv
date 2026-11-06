#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ptxv soundness harness: mutants -> ptxv check -> GB300 triage (docs/soundness-harness.md).

Subcommands (all state lives under --runs DIR):

  baseline   check every case's base variant (--certify); records which are clean
  gen        write the mutants of every eligible case (tools/mutate/mutate.py)
  check      ptxv check every mutant: per front-end, with and without --certify
  hw         GB300 triage of the candidate misses (unsteered + random --delay runs)
  stress     GB300 random-delay runs of every clean base kernel
  report     results table (markdown) from the JSONL files

Cases come from --cases FILE (one per line: `path [base-variant] [arch]`) or default to every
fixtures/*/case.toml whose `good` exists.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LOCK = threading.Lock()

CLEANISH = {"clean", "contract"}


# --------------------------------------------------------------------------------------------
# running ptxv


def capped(
    cmd: list[str], mem: str, timeout: float, env: dict | None = None
) -> tuple[int, str, str, float]:
    """Run `cmd` in its own memory-capped scope with a wall-clock limit."""
    full = [
        "systemd-run",
        "--user",
        "--scope",
        "-q",
        "-p",
        f"MemoryMax={mem}",
        "-p",
        "MemorySwapMax=0",
        "timeout",
        "-k",
        "10",
        str(int(timeout)),
    ] + cmd
    for attempt in range(10):
        t0 = time.time()
        p = subprocess.run(
            full, capture_output=True, text=True, env={**os.environ, **(env or {})}
        )
        # systemd's user bus is sometimes briefly unreachable on a loaded box: not the case's fault.
        if not any(
            x in p.stderr
            for x in (
                "Failed to connect to user scope bus",
                "Failed to request invocation ID",
            )
        ):
            break
        time.sleep(5 + 5 * attempt)
    return p.returncode, p.stdout, p.stderr, time.time() - t0


_CERT_RACE = re.compile(r"^certificate run race \((.*?)\): (.+?):(?: |$)")


def cert_run_race(err: str) -> str | None:
    """The certificate whose own run found a race that stands (`ptxv check --certify` prints
    `certificate run race (WHAT): SIG: ...` before exploring: a race in a feasible execution,
    in the LDS/STS-strong model), as `WHAT: SIG`, or None."""
    for line in err.splitlines():
        m = _CERT_RACE.match(line)
        if m:
            return f"{m.group(1)}: {m.group(2)}"
    return None


def classify_check(rc: int, out: str, err: str, timeout_hit: bool) -> dict:
    """One `ptxv check --json` run -> {status, got, ...}.

    status: caught (a finding that is not clean / contract) | contract (only contract
    deviations) | clean | budget (incomplete without a finding, timeout, OOM) | rejected (the
    case does not load / lift / compile).
    """
    if timeout_hit or rc in (124, 137) or rc < 0:
        # The certificate's report comes before the exploration that ran out of time.
        if w := cert_run_race(err):
            return {
                "status": "caught",
                "why": f"certificate run reported a race ({w}); exploration timed out rc={rc}",
                "cert_run": True,
            }
        return {"status": "budget", "why": f"timeout/killed rc={rc}"}
    try:
        doc = json.loads(out.strip().splitlines()[-1]) if out.strip() else None
    except (json.JSONDecodeError, IndexError):
        doc = None
    if doc is None:
        tail = err.strip().splitlines()[-3:]
        return {"status": "rejected", "why": " | ".join(tail)[:400] or f"rc={rc}"}
    if "error" in doc:
        return {"status": "rejected", "why": doc["error"][:400]}
    rows = doc.get("rows", [])
    if not rows:
        return {"status": "rejected", "why": "no rows"}
    r = rows[0]
    v = r["verdict"]
    got = r["got"]
    bad = [g for g in got if g not in CLEANISH]
    res = {
        "got": got,
        "complete": v.get("complete"),
        "states": v.get("states"),
        "certified": "E-WELLSYNC" in v.get("assumptions", []),
        "sigs": sorted({f.get("sig", "") for f in v.get("findings", [])})[:8],
        "seconds": v.get("seconds"),
        "assumes": v.get("assumptions", []),
    }
    if bad:
        res["status"] = "caught"
    elif not v.get("complete") and (w := cert_run_race(err)):
        # The certificate's run is one feasible execution: a race it reported (with LDS/STS
        # strong, as ptxv's own strong re-check) is a finding even when the exploration that
        # replaces the certificate runs out of budget before reaching it (r3_grouped_mm's
        # epilogue mutants: 152 CTAs).
        res["status"] = "caught"
        res["why"] = f"certificate run reported a race ({w})"
        res["cert_run"] = True
    elif not v.get("complete"):
        res["status"] = "budget"
        res["why"] = "state budget"
    elif "contract" in got:
        res["status"] = "contract"
    else:
        res["status"] = "clean"
    return res


def run_check(
    case: Path, variant: str, front: str, certify: bool, a, outdir: Path
) -> dict:
    # One output directory per front-end and mode: concurrent runs of one case otherwise
    # rewrite each other's mut.ptx / mut.cubin.
    outdir = outdir / f"{front}-{'cert' if certify else 'explore'}"
    cmd = [
        a.ptxv,
        "check",
        str(case),
        "--variant",
        variant,
        "--front",
        front,
        "--json",
        "--max-states",
        str(a.max_states),
        "--out",
        str(outdir),
    ]
    if certify:
        cmd.append("--certify")
    if getattr(a, "slices", False):
        cmd.append("--slices")
    rc, out, err, secs = capped(cmd, a.mem, a.timeout, {"PTXV_THREADS": str(a.threads)})
    r = classify_check(rc, out, err, secs >= a.timeout)
    r.update(
        {
            "front": front,
            "mode": "cert" if certify else "explore",
            "wall": round(secs, 2),
        }
    )
    if getattr(a, "slices", False):
        r["slices"] = True
    return r


def fronts_of(case: Path) -> list[str]:
    m = tomllib.loads((case / "case.toml").read_text())
    return ["sass"] if m.get("cubin") else ["ptx", "sass"]


def append_jsonl(path: Path, rec: dict):
    with LOCK:
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------------------------
# cases


def read_cases(a) -> list[dict]:
    out = []
    if a.cases:
        for line in Path(a.cases).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            out.append(
                {
                    "path": str(Path(parts[0]).resolve()),
                    "base": parts[1] if len(parts) > 1 else "good",
                    "arch": parts[2] if len(parts) > 2 else None,
                }
            )
    else:
        for p in sorted((ROOT / "fixtures").glob("*/case.toml")):
            m = tomllib.loads(p.read_text())
            if any(v["id"] == "good" for v in m.get("variants", [])):
                out.append({"path": str(p.parent), "base": "good", "arch": None})
    if a.only:
        rx = re.compile(a.only)
        out = [c for c in out if rx.search(case_key(c))]
    return out


def case_key(c: dict) -> str:
    p = Path(c["path"])
    name = p.name
    if re.match(
        r"D\d+$", p.parent.name
    ):  # bugs/<diff>/<case> (an outside study's layout)
        name = f"{p.parent.name[:4]}_{name}"
    return name if c["base"] == "good" else f"{name}@{c['base']}"


def materialized_base(c: dict, runs: Path) -> Path:
    """A case dir whose `good` is the case's base variant, with the arch override applied
    (identity when base = good and no arch)."""
    if c["base"] == "good" and not c["arch"]:
        return Path(c["path"])
    d = runs / "bases" / case_key(c)
    m = mutate.Mutant(id="base", op="none", summary="base kernel")
    case_dir = Path(c["path"])
    man = tomllib.loads((case_dir / "case.toml").read_text())
    bv = next((v for v in man.get("variants", []) if v["id"] == c["base"]), None)
    src = None
    if not man.get("cubin"):
        src = (case_dir / man["ptx"]).read_text()
        if bv and (bv.get("marker") or bv.get("edits")):
            src = mutate.apply_variant_ptx(src, bv)
        if c["arch"]:
            src = re.sub(r"(?m)^(\s*\.target\s+)sm_\w+", r"\g<1>" + c["arch"], src)
    mutate.write_mutant(case_dir, d, m, src, bv, c["arch"])
    return d


# --------------------------------------------------------------------------------------------
# subcommands


def cmd_baseline(a):
    runs = Path(a.runs)
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / "baseline.jsonl"
    done = {(r["case"], r["front"], r["mode"]) for r in load_jsonl(path)}
    jobs = []
    for c in read_cases(a):
        base = materialized_base(c, runs)
        for f in fronts_of(base):
            for mode in a.modes:
                if (case_key(c), f, mode) not in done:
                    jobs.append((c, base, f, mode))

    def job(j):
        c, base, f, mode = j
        r = run_check(
            base, "good", f, mode == "cert", a, runs / "out" / "baseline" / case_key(c)
        )
        r.update(
            {
                "case": case_key(c),
                "path": str(base),
                "src": c["path"],
                "base": c["base"],
            }
        )
        append_jsonl(path, r)
        print(
            f"baseline {case_key(c):32} {f:4} {mode:7} {r['status']:9} {r.get('got')} "
            f"{r.get('why', '')[:100]}",
            flush=True,
        )

    with cf.ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(job, jobs))


def eligible(runs: Path) -> dict[str, dict]:
    """case -> {path, fronts: [front...]} for cases whose base is clean (or contract-only) and
    complete in cert mode on that front."""
    out: dict[str, dict] = {}
    for r in load_jsonl(runs / "baseline.jsonl"):
        if r["mode"] != "cert":
            continue
        if r["status"] in ("clean", "contract"):
            e = out.setdefault(
                r["case"],
                {
                    "path": r["path"],
                    "src": r["src"],
                    "base": r["base"],
                    "fronts": [],
                    "baseline": {},
                },
            )
            e["fronts"].append(r["front"])
            e["baseline"][r["front"]] = r["got"]
    return out


def cmd_gen(a):
    runs = Path(a.runs)
    el = eligible(runs)
    cases = {case_key(c): c for c in read_cases(a)}
    n = 0
    for key, e in sorted(el.items()):
        if key not in cases:
            continue
        c = cases[key]
        base = Path(e["path"])
        try:
            ds = mutate.generate(
                base,
                runs / "mutants" / key,
                "good",
                None,
                set(a.ops.split(",")) if a.ops else None,
            )
        except Exception as ex:  # noqa: BLE001
            print(f"gen {key}: {ex}")
            continue
        n += len(ds)
        print(f"gen {key:32} {len(ds):4} mutants")
    print(f"{n} mutants")


def mutant_dirs(runs: Path, only: str | None):
    rx = re.compile(only) if only else None
    for d in sorted((runs / "mutants").glob("*/*/mutant.json")):
        key = d.parent.parent.name
        if rx and not rx.search(key):
            continue
        yield key, d.parent


def cmd_check(a):
    runs = Path(a.runs)
    el = eligible(runs)
    path = runs / "check.jsonl"
    prev = load_jsonl(path)
    if (
        a.retry_rejected
    ):  # drop rejected runs (e.g. transient tool failures) and run them again
        prev = [r for r in prev if r["status"] != "rejected"]
        path.write_text("".join(json.dumps(r) + "\n" for r in prev))
    mrx = re.compile(a.mutants) if getattr(a, "mutants", None) else None
    crx = re.compile(a.only) if a.only else None
    if getattr(a, "redo", False):
        # Rerun the selected mutants: their old records go (the modes being run only).
        sel = lambda r: (
            (not crx or crx.search(r["case"]))
            and (not mrx or mrx.search(r["mutant"]))
            and r["mode"] in a.modes
        )
        prev = [r for r in prev if not sel(r)]
        path.write_text("".join(json.dumps(r) + "\n" for r in prev))
    done = {(r["case"], r["mutant"], r["front"], r["mode"]) for r in prev}
    # With --certify, a failed certificate falls back to exploration, so plain exploration
    # adds information only where --certify said clean: --explore-all to run it everywhere.
    reach = reach_map(runs)  # dead sites (`probe`) need no exploration
    cert_clean = {
        (r["case"], r["mutant"], r["front"])
        for r in prev
        if r["mode"] == "cert" and r["status"] in ("clean", "contract")
    }
    jobs = []
    for key, d in mutant_dirs(runs, a.only):
        if key not in el or (mrx and not mrx.search(d.name)):
            continue
        for f in el[key]["fronts"]:
            for mode in a.modes:
                if (key, d.name, f, mode) in done:
                    continue
                if (
                    mode == "explore"
                    and not a.explore_all
                    and (
                        (key, d.name, f) not in cert_clean
                        or reach.get((key, d.name)) == "unreached"
                    )
                ):
                    continue
                jobs.append((key, d, f, mode))
    print(f"{len(jobs)} check runs", flush=True)

    def job(j):
        key, d, f, mode = j
        r = run_check(
            d, "mut", f, mode == "cert", a, runs / "out" / "check" / key / d.name
        )
        mj = json.loads((d / "mutant.json").read_text())
        r.update(
            {"case": key, "mutant": d.name, "op": mj["op"], "summary": mj["summary"]}
        )
        append_jsonl(path, r)
        print(
            f"check {key:28} {d.name:28} {f:4} {mode:7} {r['status']:9} {r.get('got')} "
            f"{r.get('why', '')[:80]}",
            flush=True,
        )

    with cf.ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(job, jobs))


def summarize(runs: Path) -> dict[tuple[str, str], dict]:
    """(case, mutant) -> per-front status and the mutant's label.

    Per front: caught if any mode caught; else clean / contract if a mode completed clean;
    else budget / rejected. `cert_disagrees`: --certify clean but plain exploration caught it.
    Label: caught (every runnable front caught) | candidate (some front clean or contract-only)
    | budget | rejected.
    """
    out: dict[tuple[str, str], dict] = {}
    for r in load_jsonl(runs / "check.jsonl"):
        e = out.setdefault(
            (r["case"], r["mutant"]),
            {"op": r["op"], "summary": r["summary"], "runs": {}},
        )
        e["runs"][(r["front"], r["mode"])] = r
    for (k, m), e in out.items():
        mj = runs / "mutants" / k / m / "mutant.json"
        if (
            mj.exists()
        ):  # the record's copy can be stale if the mutants were regenerated
            j = json.loads(mj.read_text())
            e["op"], e["summary"] = j["op"], j["summary"]
        fr = {}
        for (f, mode), r in e["runs"].items():
            fr.setdefault(f, []).append(r)
        e["front"] = {}
        e["cert_disagrees"] = []
        for f, rs in fr.items():
            st = {r["mode"]: r["status"] for r in rs}
            if "caught" in st.values():
                s = "caught"
            elif "clean" in st.values():
                s = "clean"
            elif "contract" in st.values():
                s = "contract"
            elif "budget" in st.values():
                s = "budget"
            else:
                s = "rejected"
            e["front"][f] = s
            if (
                st.get("cert") in ("clean", "contract")
                and st.get("explore") == "caught"
            ):
                e["cert_disagrees"].append(f)
        ss = set(e["front"].values())
        if ss <= {"caught"}:
            e["label"] = "caught"
        elif ss & {"clean", "contract"}:
            e["label"] = "candidate"
        elif "caught" in ss:
            e["label"] = "caught"
        elif "budget" in ss:
            e["label"] = "budget"
        else:
            e["label"] = "rejected"
    return out


# --------------------------------------------------------------------------------------------
# hardware


_HW_LINE = re.compile(r"^hw (\S+)\s+(.*?) \((\d+) runs, ([\d.]+)s\)")


def run_hw(
    case: Path,
    variant: str,
    delays: list[str],
    a,
    outdir: Path,
    runs: int | None = None,
    expect: str | None = None,
) -> dict:
    """One `ptxv hw` invocation. Waits for the GPU when another job holds it; falls back to
    PTXV_HW_EXPECT=run when the good kernel's numerics run fails (e.g. a PTX-only spurious
    finding)."""
    cmd = [
        a.ptxv,
        "hw",
        str(case),
        "--variant",
        variant,
        "--runs",
        str(runs or a.hw_runs),
        "--timeout",
        str(a.hw_timeout),
        "--out",
        str(outdir),
    ]
    for d in delays:
        cmd += ["--delay", d]
    env = {"PTXV_GPUS": a.gpus, "PTXV_THREADS": "2"}
    expect = expect or a.hw_expect
    if expect:
        env["PTXV_HW_EXPECT"] = expect
    for attempt in range(60):
        rc, out, err, secs = capped(cmd, a.mem, a.hw_timeout * 3 + 300, env)
        if "no free GPU" in err:
            time.sleep(30)
            continue
        break
    res = {"rc": rc, "wall": round(secs, 1), "delays": delays, "variant": variant}
    if expect:
        res["expect"] = expect
    for line in out.splitlines():
        m = _HW_LINE.match(line)
        if m and m.group(1) == variant:
            o = m.group(2)
            res["outcome"] = o
            if o.startswith("matches"):
                res["status"] = "ok"
            elif o.startswith("hang"):
                res["status"] = "hang"
            elif o.startswith("error"):
                res["status"] = "error"
            elif "differ" in o:
                res["status"] = "wrong"
            else:
                res["status"] = "?"
            res["runs"] = int(m.group(3))
            mm = re.match(r"(\d+)/(\d+) runs differ", o)
            if mm:
                res["mismatches"] = int(mm.group(1))
    if "status" not in res:
        why = err.strip().splitlines() or [f"rc={rc}"]
        if (
            expect is None and "no free GPU" not in err
        ):  # e.g. the numerics run of good failed
            return run_hw(case, variant, delays, a, outdir, runs, expect="run")
        res["status"] = "harness-error"
        res["why"] = why[0][:300] + (" | " + why[-1][:200] if len(why) > 1 else "")
    return res


def sync_lines(ptx: str) -> list[int]:
    """1-based lines next to synchronization instructions (candidate delay sites)."""
    lines = ptx.split("\n")
    sites = set()
    keys = (
        "bar.",
        "barrier.",
        "mbarrier.",
        "fence",
        "membar",
        "tcgen05.",
        "cp.async",
        "ld.acquire",
        "st.release",
        "atom",
        "red.",
        "griddepcontrol",
        "st.async",
        "ld.global",
        "st.global",
        "ld.shared",
        "st.shared",
        "ld.volatile",
        "ret",
        "exit",
    )
    for ins in mutate.parse_insts(lines):
        if any(ins.opcode.startswith(k) for k in keys):
            for d in (0, 1):
                l = ins.idx + 1 + d
                if 1 <= l <= len(lines):
                    sites.add(l)
    # A delay goes before a line; not before declarations / braces / directives.
    ok = []
    for l in sorted(sites):
        t = mutate.strip_comment(lines[l - 1]).strip()
        if not t or t.startswith((".", "{", "}")) or t.endswith("{"):
            continue
        ok.append(l)
    return ok


def delay_plans(ptx: str, focus: list[int], n: int, seed: int) -> list[list[str]]:
    """n random delay sets: 1-3 sites (near the mutated lines half the time), 0.5-100 us."""
    rng = random.Random(seed)
    sites = sync_lines(ptx)
    if not sites:
        return []
    near = [s for s in sites if any(abs(s - f) <= 8 for f in focus)] or sites
    plans = []
    for i in range(n):
        k = rng.choice([1, 1, 2, 3])
        pool = near if i % 2 == 0 else sites
        chosen = sorted(set(rng.choice(pool) for _ in range(k)))
        plans.append(
            [f"{l}:{rng.choice([500, 2000, 10000, 50000, 100000])}" for l in chosen]
        )
    return plans


def variant_ptx_text(case: Path, variant: str) -> str | None:
    m = tomllib.loads((case / "case.toml").read_text())
    if m.get("cubin"):
        return None
    src = (case / m["ptx"]).read_text()
    v = next(v for v in m["variants"] if v["id"] == variant)
    if v.get("marker") or v.get("edits"):
        src = mutate.apply_variant_ptx(src, v)
    return src


def triage_one(
    case: Path, key: str, name: str, focus: list[int], a, runs: Path
) -> dict:
    """Unsteered runs, then random delay plans. A failing plan is re-run on `good` (control)."""
    outd = runs / "out" / "hw" / key / name
    rec = {"case": key, "mutant": name, "trials": []}
    base = run_hw(case, "mut", [], a, outd)
    rec["trials"].append(base)
    if base["status"] == "harness-error":
        rec["label"] = "hw-harness-error"
        rec["why"] = base.get("why")
        return rec
    if base["status"] in ("wrong", "hang", "error"):
        base["control"] = run_hw(case, "good", [], a, outd)
    ptx = variant_ptx_text(case, "mut")
    # Each `ptxv hw` recomputes the expected output (the good kernel's numerics run): for slow
    # cases (> 8 s a trial) a third of the plans.
    n = a.plans if base["wall"] < 8 else max(4, a.plans // 3)
    plans = (
        delay_plans(ptx, focus, n, zlib.crc32(f"{key}/{name}".encode())) if ptx else []
    )
    for p in plans:
        r = run_hw(case, "mut", p, a, outd)
        rec["trials"].append(r)
        if r["status"] in ("wrong", "hang", "error"):
            ctl = run_hw(case, "good", p, a, outd)
            r["control"] = ctl
            if a.stop_on_fail and ctl["status"] == "ok":
                break
    fails = [t for t in rec["trials"] if t["status"] in ("wrong", "hang", "error")]
    confirmed = [t for t in fails if t.get("control", {}).get("status") == "ok"]
    if not fails:
        rec["label"] = "unobserved"
    elif confirmed:
        rec["label"] = "confirmed"
        rec["witness"] = confirmed[0]
    else:
        rec["label"] = "control-fails"
    return rec


def hw_batch(runs: Path, key: str, ms: list[str], a) -> list[dict]:
    """GB300 runs of several binary-only mutants of one case in one `ptxv hw` invocation."""
    d0 = runs / "mutants" / key / ms[0]
    bd = runs / "batch" / key
    if bd.exists():
        shutil.rmtree(bd)
    bd.mkdir(parents=True)
    for f in d0.iterdir():
        if f.name not in ("case.toml", "mutant.json"):
            os.symlink(f.resolve(), bd / f.name)
    text = (d0 / "case.toml").read_text()
    body = mutate.strip_variants(text)
    man = tomllib.loads(text)
    good = next(v for v in man["variants"] if v["id"] == "good")
    vs = [
        '[[variants]]\nid = "good"\nexpect = ["clean"]'
        + (
            f"\npatch = {mutate._toml_patch(good['patch'])}"
            if good.get("patch")
            else ""
        )
    ]
    for m in ms:
        mv = next(
            v
            for v in tomllib.loads(
                (runs / "mutants" / key / m / "case.toml").read_text()
            )["variants"]
            if v["id"] == "mut"
        )
        vs.append(
            f'[[variants]]\nid = "{m}"\nexpect = ["clean"]\npatch = {mutate._toml_patch(mv["patch"])}'
        )
    (bd / "case.toml").write_text(body + "\n" + "\n\n".join(vs) + "\n")
    cmd = [
        a.ptxv,
        "hw",
        str(bd),
        "--variant",
        ",".join(["good"] + ms),
        "--runs",
        str(a.hw_runs),
        "--timeout",
        str(a.hw_timeout),
        "--out",
        str(runs / "out" / "hw-batch" / key),
    ]
    env = {"PTXV_GPUS": a.gpus, "PTXV_THREADS": "2"}
    if a.hw_expect:
        env["PTXV_HW_EXPECT"] = a.hw_expect
    for attempt in range(60):
        rc, out, err, secs = capped(
            cmd, a.mem, (a.hw_timeout + 30) * (len(ms) + 1) + 600, env
        )
        if "no free GPU" in err:
            time.sleep(30)
            continue
        if "Error" in err and "numerics run" in err and "PTXV_HW_EXPECT" not in env:
            env["PTXV_HW_EXPECT"] = "run"
            continue
        break
    rows = {}
    for line in out.splitlines():
        mm = _HW_LINE.match(line)
        if mm:
            o = mm.group(2)
            st = (
                "ok"
                if o.startswith("matches")
                else "hang"
                if o.startswith("hang")
                else "error"
                if o.startswith("error")
                else "wrong"
                if "differ" in o
                else "?"
            )
            rows[mm.group(1)] = {
                "status": st,
                "outcome": o,
                "runs": int(mm.group(3)),
                "delays": [],
                "variant": mm.group(1),
                "wall": float(mm.group(4)),
            }
    ctl = rows.get("good")
    recs = []
    for m in ms:
        rec = {"case": key, "mutant": m, "trials": [], "batch": True}
        r = rows.get(m)
        if r is None or ctl is None:
            rec["label"] = "hw-harness-error"
            rec["why"] = (err.strip().splitlines() or [f"rc={rc}"])[-1][:300]
        else:
            r["control"] = ctl
            rec["trials"].append(r)
            if r["status"] == "ok":
                rec["label"] = "unobserved"
            elif ctl["status"] == "ok":
                rec["label"] = "confirmed"
                rec["witness"] = r
            else:
                rec["label"] = "control-fails"
        recs.append(rec)
    return recs


def cmd_hw(a):
    runs = Path(a.runs)
    path = runs / "hw.jsonl"
    done = {
        (r["case"], r["mutant"])
        for r in load_jsonl(path)
        if r["label"] != "hw-harness-error"
    }
    s = summarize(runs)
    reach = reach_map(runs)
    same = {
        (r["case"], r["mutant"])
        for r in load_jsonl(runs / "sasseq.jsonl")
        if r["same_sass"]
    }
    todo = [
        (k, m)
        for (k, m), e in sorted(s.items())
        if e["label"] == "candidate"
        and (k, m) not in done
        and reach.get((k, m)) != "unreached"
        and (k, m) not in same
        and (not a.only or re.search(a.only, k))
    ]
    print(f"{len(todo)} candidate misses to triage on GPU(s) {a.gpus}", flush=True)
    # Binary-only mutants take no delays: one `ptxv hw` per case runs all of them (and the
    # good kernel as the control), so the expected output is computed once.
    batch: dict[str, list[str]] = {}
    for k, m in todo:
        mj = json.loads((runs / "mutants" / k / m / "mutant.json").read_text())
        if mj.get("patch") and not mj.get("edits"):
            batch.setdefault(k, []).append(m)
    chunks = [
        (k, ms[i : i + 20])
        for k, ms in sorted(batch.items())
        for i in range(0, len(ms), 20)
    ]
    for k, ms in chunks:  # 20 a run: each variant's lifted kernel stays in memory
        for rec in hw_batch(runs, k, ms, a):
            rec["front"] = s[(k, rec["mutant"])]["front"]
            append_jsonl(path, rec)
            w = rec.get("witness", {})
            print(
                f"hw {k:28} {rec['mutant']:28} {rec['label']:16} {w.get('outcome', '')}",
                flush=True,
            )
    todo = [(k, m) for k, m in todo if m not in batch.get(k, [])]
    for k, m in todo:
        d = runs / "mutants" / k / m
        mj = json.loads((d / "mutant.json").read_text())
        rec = triage_one(d, k, m, mj.get("lines", []), a, runs)
        rec["front"] = s[(k, m)]["front"]
        append_jsonl(path, rec)
        w = rec.get("witness", {})
        print(
            f"hw {k:28} {m:28} {rec['label']:16} {w.get('delays', '')} {w.get('outcome', '')}",
            flush=True,
        )


def cmd_probe(a):
    """Reach probes of the candidate misses: the mutated site replaced by a trap, checked on
    the front-ends where the mutant was clean. `unreached` (clean): no schedule executes the
    site at this launch, so the mutant is equivalent to the base kernel. With --gpus, the trap
    probe of every unreached site also runs on GB300 (a trap there would make the model's
    reachability wrong)."""
    runs = Path(a.runs)
    path = runs / "probe.jsonl"
    prev = load_jsonl(path)
    if a.retry_rejected:  # probes that ended `unknown` run again
        prev = [r for r in prev if r["reach"] != "unknown"]
        path.write_text("".join(json.dumps(r) + "\n" for r in prev))
    done = {(r["case"], r["mutant"]) for r in prev}
    s = summarize(runs)
    todo = [
        (k, m)
        for (k, m), e in sorted(s.items())
        if e["label"] == "candidate"
        and (k, m) not in done
        and (not a.only or re.search(a.only, k))
    ]
    print(f"{len(todo)} reach probes", flush=True)

    def job(km):
        k, m = km
        d = runs / "mutants" / k / m
        pd = mutate.write_probe(d, runs / "probes" / k / m)
        fronts = [f for f, st in s[km]["front"].items() if st in ("clean", "contract")]
        # One deterministic run first (PTXV_RUN_ONCE): a trap there settles it cheaply; only
        # a site that run does not reach needs the exhaustive check.
        once = []
        for f in fronts:
            cmd = [
                a.ptxv,
                "check",
                str(pd),
                "--variant",
                "mut",
                "--front",
                f,
                "--no-gb300",
                "--out",
                str(runs / "out" / "probe" / k / m / f"{f}-once"),
            ]
            rc, out, err, secs = capped(
                cmd,
                a.mem,
                a.timeout,
                {"PTXV_RUN_ONCE": "1", "PTXV_THREADS": str(a.threads)},
            )
            once.append("    TRAP:" in out)
        if any(once):
            rs = []
        else:
            rs = [
                run_check(pd, "mut", f, True, a, runs / "out" / "probe" / k / m)
                for f in fronts
            ]
        got = [g for r in rs for g in (r.get("got") or [])]
        if any(once) or "trap" in got:
            reach = "reached"
        elif rs and all(r["status"] in ("clean", "contract") for r in rs):
            reach = "unreached"
        else:
            reach = "unknown"
        rec = {"case": k, "mutant": m, "reach": reach, "runs": rs}
        append_jsonl(path, rec)
        print(f"probe {k:28} {m:28} {reach}", flush=True)

    with cf.ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(job, todo))
    if a.gpus and not a.no_hw_probe:
        hpath = runs / "probe_hw.jsonl"
        hdone = {(r["case"], r["mutant"]) for r in load_jsonl(hpath)}
        for r in load_jsonl(path):
            km = (r["case"], r["mutant"])
            if r["reach"] != "unreached" or km in hdone:
                continue
            pd = runs / "probes" / km[0] / km[1]
            h = run_hw(
                pd, "mut", [], a, runs / "out" / "probe_hw" / km[0] / km[1], runs=20
            )
            rec = {
                "case": km[0],
                "mutant": km[1],
                "hw": h,
                "hw_reached": h["status"] in ("error", "hang", "wrong"),
            }
            append_jsonl(hpath, rec)
            print(
                f"probe-hw {km[0]:28} {km[1]:28} {h['status']} {h.get('outcome', h.get('why', ''))[:80]}",
                flush=True,
            )


def _sass_text(ptx: str, arch: str, tmp: Path) -> str:
    """ptxas + nvdisasm of a PTX text, addresses and comments stripped."""
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "k.ptx").write_text(ptx)
    subprocess.run(
        ["ptxas", f"-arch={arch}", str(tmp / "k.ptx"), "-o", str(tmp / "k.cubin")],
        check=True,
        capture_output=True,
    )
    dis = mutate.nvdisasm(tmp / "k.cubin")
    out = []
    for line in dis.splitlines():
        line = re.sub(r"/\*[0-9a-f]+\*/", "", line).split("//")[0].strip()
        if line:
            out.append(line)
    return "\n".join(out)


def cmd_sasseq(a):
    """For every PTX mutant that is not caught: does ptxas emit the same SASS as for the base
    kernel? Identical SASS means the mutant is the base kernel on GB300 (redundant there)."""
    runs = Path(a.runs)
    path = runs / "sasseq.jsonl"
    done = {(r["case"], r["mutant"]) for r in load_jsonl(path)}
    s = summarize(runs)
    base_sass: dict[str, str] = {}
    for (k, m), e in sorted(s.items()):
        if e["label"] == "caught" or (k, m) in done or m.startswith("sass-"):
            continue
        if a.only and not re.search(a.only, k):
            continue
        d = runs / "mutants" / k / m
        man = tomllib.loads((d / "case.toml").read_text())
        arch = man.get("arch", "sm_103a")
        tmp = runs / "out" / "sasseq" / k / m
        try:
            if k not in base_sass:
                base_sass[k] = _sass_text(
                    variant_ptx_text(d, "good"), arch, tmp / "good"
                )
            same = (
                _sass_text(variant_ptx_text(d, "mut"), arch, tmp / "mut")
                == base_sass[k]
            )
        except subprocess.CalledProcessError as ex:
            same = None
            print(f"sasseq {k} {m}: {ex.stderr[:200] if ex.stderr else ex}")
        append_jsonl(path, {"case": k, "mutant": m, "same_sass": same})
        print(f"sasseq {k:28} {m:28} {same}", flush=True)


def reach_map(runs: Path) -> dict[tuple[str, str], str]:
    return {
        (r["case"], r["mutant"]): r["reach"] for r in load_jsonl(runs / "probe.jsonl")
    }


def cmd_stress(a):
    runs = Path(a.runs)
    path = runs / "stress.jsonl"
    done = {r["case"] for r in load_jsonl(path)}
    el = eligible(runs)
    for key, e in sorted(el.items()):
        if key in done or (a.only and not re.search(a.only, key)):
            continue
        case = Path(e["path"])
        outd = runs / "out" / "stress" / key
        rec = {"case": key, "trials": []}
        base = run_hw(case, "good", [], a, outd)
        rec["trials"].append(base)
        if base["status"] == "harness-error":
            rec["label"] = "hw-harness-error"
            rec["why"] = base.get("why")
        else:
            ptx = variant_ptx_text(case, "good")
            for p in (
                delay_plans(ptx, [], a.plans, zlib.crc32(key.encode())) if ptx else []
            ):
                rec["trials"].append(run_hw(case, "good", p, a, outd))
            fails = [
                t for t in rec["trials"] if t["status"] in ("wrong", "hang", "error")
            ]
            rec["label"] = "fails" if fails else "ok"
            if fails:
                rec["witness"] = fails[0]
        append_jsonl(path, rec)
        w = rec.get("witness", {})
        print(
            f"stress {key:32} {rec['label']:16} {len(rec['trials'])} trials "
            f"{w.get('delays', '')} {w.get('outcome', '')} {rec.get('why', '')}",
            flush=True,
        )


# --------------------------------------------------------------------------------------------
# report


def cmd_report(a):
    runs = Path(a.runs)
    s = summarize(runs)
    hw = {(r["case"], r["mutant"]): r for r in load_jsonl(runs / "hw.jsonl")}
    manual = {}
    mp = runs / "manual.jsonl"
    for r in load_jsonl(mp):  # hand triage: {case, mutant, label, note}
        manual[(r["case"], r["mutant"])] = r
    reach = reach_map(runs)
    sasseq = {
        (r["case"], r["mutant"]): r["same_sass"]
        for r in load_jsonl(runs / "sasseq.jsonl")
    }
    probe_hw = {
        (r["case"], r["mutant"]): r for r in load_jsonl(runs / "probe_hw.jsonl")
    }
    final = {}
    for km, e in s.items():
        lab = e["label"]
        if e["cert_disagrees"]:
            lab = "cert-miss"  # --certify clean, exploration caught it: a certificate miss
        elif lab == "candidate" and reach.get(km) == "unreached":
            lab = "dead"
            if probe_hw.get(km, {}).get("hw_reached"):
                lab = "dead-but-hw-reached"
        elif lab == "candidate":
            h = hw.get(km)
            if h is None:
                lab = "candidate"
            elif h["label"] == "confirmed":
                lab = "confirmed"
            elif h["label"] == "unobserved":
                lab = "unobserved"
            else:
                lab = "candidate:" + h["label"]
        if lab in ("unobserved", "candidate") and sasseq.get(km):
            lab = "redundant"  # ptxas emits the base kernel's SASS: GB300 runs the same code
        if km in manual:
            lab = manual[km]["label"]
        final[km] = lab
    cases = sorted({k for k, _ in s})
    cols = [
        "caught",
        "cert-miss",
        "dead",
        "candidate",
        "confirmed",
        "unobserved",
        "redundant",
        "budget",
        "rejected",
    ]
    lines = [
        "| case | mutants | " + " | ".join(cols) + " | other |",
        "|---|---:|" + "---:|" * (len(cols) + 1),
    ]
    tot = {c: 0 for c in cols + ["other", "n"]}
    for c in cases:
        labs = [final[km] for km in final if km[0] == c]
        cnt = {k: sum(1 for l in labs if l == k) for k in cols}
        other = len(labs) - sum(cnt.values())
        for k in cols:
            tot[k] += cnt[k]
        tot["other"] += other
        tot["n"] += len(labs)
        lines.append(
            f"| {c} | {len(labs)} | "
            + " | ".join(str(cnt[k] or "") for k in cols)
            + f" | {other or ''} |"
        )
    lines.append(
        f"| **total** | {tot['n']} | "
        + " | ".join(str(tot[k]) for k in cols)
        + f" | {tot['other']} |"
    )
    by_op: dict[str, dict[str, int]] = {}
    for km, lab in final.items():
        by_op.setdefault(s[km]["op"], {}).setdefault(lab.split(":")[0], 0)
        by_op[s[km]["op"]][lab.split(":")[0]] += 1
    lines += [
        "",
        "| operator | " + " | ".join(cols) + " |",
        "|---|" + "---:|" * len(cols),
    ]
    for op, d in sorted(by_op.items()):
        lines.append(
            f"| {op} | " + " | ".join(str(d.get(k, "") or "") for k in cols) + " |"
        )
    det = [
        "",
        "| case | mutant | ptx | sass | final | note |",
        "|---|---|---|---|---|---|",
    ]
    for km in sorted(final):
        if final[km] in ("caught", "dead", "rejected", "budget"):
            continue
        e = s[km]
        note = manual.get(km, {}).get("note") or e["summary"]
        h = hw.get(km)
        if h and h.get("witness"):
            w = h["witness"]
            note += f"; GB300: {w.get('outcome')} with delays {w.get('delays')}"
        if e["cert_disagrees"]:
            note += f"; --certify clean but exploration caught it ({','.join(e['cert_disagrees'])})"
        det.append(
            f"| {km[0]} | {km[1]} | {e['front'].get('ptx', '-')} | {e['front'].get('sass', '-')} "
            f"| {final[km]} | {note} |"
        )
    dis = [
        f"| {k} | {m} | {','.join(e['cert_disagrees'])} | {e['summary']} |"
        for (k, m), e in sorted(s.items())
        if e["cert_disagrees"]
    ]
    if dis:
        det += [
            "",
            "Certificate disagreements (cert clean, exploration caught):",
            "",
            "| case | mutant | front | mutant |",
            "|---|---|---|---|",
        ] + dis
    st = load_jsonl(runs / "stress.jsonl")
    if st:
        det += [
            "",
            "Unmutated kernels under random delays on GB300:",
            "",
            "| case | label | trials | witness |",
            "|---|---|---:|---|",
        ]
        for r in st:
            w = r.get("witness", {})
            det.append(
                f"| {r['case']} | {r['label']} | {len(r['trials'])} | "
                f"{w.get('delays', '')} {w.get('outcome', '') or r.get('why', '')} |"
            )
    text = "\n".join(lines + det) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    print(text)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "cmd",
        choices=[
            "baseline",
            "gen",
            "check",
            "probe",
            "sasseq",
            "hw",
            "stress",
            "report",
        ],
    )
    ap.add_argument("--runs", required=True, help="state directory")
    ap.add_argument(
        "--ptxv",
        default=os.environ.get("PTXV_BIN", "ptxv"),
        help="ptxv binary (copy it to a private path first)",
    )
    ap.add_argument(
        "--cases", help="case list file (path [base-variant] [arch] per line)"
    )
    ap.add_argument("--only", help="regex on the case key")
    ap.add_argument("--mutants", help="check: regex on the mutant name")
    ap.add_argument("--ops", help="comma list of mutation operators (gen)")
    ap.add_argument(
        "--modes",
        default="cert,explore",
        type=lambda s: s.split(","),
        help="check modes: cert (--certify), explore (default: both)",
    )
    ap.add_argument(
        "--explore-all",
        action="store_true",
        help="check: plain exploration of every mutant, not only where --certify was clean",
    )
    ap.add_argument(
        "--retry-rejected", action="store_true", help="check: rerun rejected runs"
    )
    ap.add_argument(
        "--slices",
        action="store_true",
        help="check: pass --slices to ptxv check (independent clusters one at a time, "
        "E-CTA-INDEP); with --only for the cases whose whole launch is too large",
    )
    ap.add_argument(
        "--redo",
        action="store_true",
        help="check: rerun the selected (--only) mutants even if they have a record; the "
        "new record replaces the old one",
    )
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2, help="PTXV_THREADS per check")
    ap.add_argument("--max-states", type=int, default=300000)
    ap.add_argument("--timeout", type=float, default=180, help="seconds per check run")
    ap.add_argument("--mem", default="16G", help="MemoryMax per run")
    ap.add_argument(
        "--gpus",
        default=None,
        help="PTXV_GPUS for hw / stress / probe (idle GPUs only; never one in use by others)",
    )
    ap.add_argument(
        "--no-hw-probe", action="store_true", help="probe: skip the GB300 trap runs"
    )
    ap.add_argument("--hw-runs", type=int, default=200)
    ap.add_argument("--hw-timeout", type=float, default=60)
    ap.add_argument(
        "--hw-expect",
        default=None,
        help="PTXV_HW_EXPECT (run: expected from one sim run)",
    )
    ap.add_argument(
        "--plans", type=int, default=12, help="random delay plans per kernel"
    )
    ap.add_argument("--stop-on-fail", action="store_true", default=True)
    ap.add_argument("--out", help="report: write markdown here")
    a = ap.parse_args(argv)
    {
        "baseline": cmd_baseline,
        "gen": cmd_gen,
        "check": cmd_check,
        "probe": cmd_probe,
        "sasseq": cmd_sasseq,
        "hw": cmd_hw,
        "stress": cmd_stress,
        "report": cmd_report,
    }[a.cmd](a)


if __name__ == "__main__":
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Certificate disagreement sweep (docs/soundness-harness.md, "adversarial tests").

Every fixture variant, on each front-end it has, is checked twice: with `--certify` and by plain
exploration. A certified verdict ("well-synchronized") whose classes differ from a complete
exploration's is a certificate miss and is printed as `MISS`.

    certsweep.py --ptxv PTXV --out DIR [--jobs N] [--only REGEX]

Results: DIR/results.jsonl (one record per fixture / variant / front). Run it inside a
memory-capped job runner; each check gets 120 s and 300k states.
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def classes(out: str):
    for ln in out.splitlines():
        if ln.startswith(("PASS ", "FAIL ")):
            return sorted(
                x for x in ln.split("[", 1)[1].split("]", 1)[0].split(",") if x
            )
    return None


def jobs(only):
    for d in sorted((ROOT / "fixtures").iterdir()):
        if not (d / "case.toml").exists() or (only and not re.search(only, d.name)):
            continue
        try:
            c = tomllib.loads((d / "case.toml").read_text())
        except Exception:
            continue
        for v in c.get("variants", []):
            for f in ["sass"] if "cubin" in c else ["ptx", "sass"]:
                yield d, v["id"], f


def run(job, ptxv, out):
    d, v, f = job
    key = f"{d.name}@{v}@{f}"
    res = {"key": key}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PTXV_THREADS="2")
    for mode, extra in (("cert", ["--certify"]), ("explore", [])):
        cmd = [
            "timeout",
            "120",
            ptxv,
            "check",
            str(d),
            "--variant",
            v,
            "--front",
            f,
            "--no-input-coverage",
            "--max-states",
            "300000",
            "--out",
            str(out / "out" / key / mode),
            *extra,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        res[mode] = classes(r.stdout)
        if mode == "cert":
            res["certified"] = "well-synchronized" in r.stdout
        else:
            res["incomplete"] = "incomplete" in r.stdout or r.returncode == 124
    res["miss"] = (
        res["certified"] and res["cert"] != res["explore"] and not res["incomplete"]
    )
    return res


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ptxv", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    js = list(jobs(a.only))
    with cf.ThreadPoolExecutor(a.jobs) as ex, open(a.out / "results.jsonl", "w") as fo:
        for r in ex.map(lambda j: run(j, a.ptxv, a.out), js):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            if r["miss"]:
                print("MISS", r["key"], r["cert"], r["explore"], flush=True)
    print("DONE", len(js), flush=True)


if __name__ == "__main__":
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regenerate the third-party kernel binaries of fixtures listed in fixtures/VENDOR_FIXTURES.toml.

    python3 tools/regen_fixture.py r3b_trtllm_bmm c_smm_mxf8     # or --all, or --missing
    python3 tools/regen_fixture.py --list

Needs a Blackwell GPU and the entry's library (see `library`). Set CUDA_VISIBLE_DEVICES to
idle GPUs; $PTXV_PYTHON (default: python3) runs the capture. Each entry's command runs in a
scratch directory; only the entry's `files` are copied into fixtures/<name>/ (case.toml,
inputs and expected outputs are kept). `--keep` leaves the scratch directory for inspection.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures"


def manifest() -> list[dict]:
    with open(FIX / "VENDOR_FIXTURES.toml", "rb") as f:
        return tomllib.load(f)["fixture"]


def entries_by_cubin(case_toml: Path) -> dict[str, set[str]]:
    """cubin file -> the kernels case.toml launches from it (top level and [[launches]])."""
    with open(case_toml, "rb") as f:
        m = tomllib.load(f)
    out: dict[str, set[str]] = {}
    for t in [m] + list(m.get("launches", [])):
        if t.get("cubin") and t.get("entry"):
            out.setdefault(t["cubin"], set()).add(t["entry"])
    return out


def regen(e: dict, keep: bool) -> None:
    name = e["name"]
    dst = FIX / name
    work = Path(tempfile.mkdtemp(prefix=f"ptxv-regen-{name}-"))
    env = dict(os.environ, OUT=str(work), PY=os.environ.get("PTXV_PYTHON", "python3"))
    if e["kind"] == "dir":
        (work / "fixtures").mkdir()
        for d in ("probes", "tools", "python"):
            (work / d).symlink_to(ROOT / d)
        run_dir = work / "fixtures" / name
        shutil.copytree(dst, run_dir)
        out_dir = run_dir
    else:
        run_dir, out_dir = ROOT, work
    print(f"[{name}] {e['cmd']}  (cwd {run_dir}; needs {e['library']})", flush=True)
    subprocess.run(["bash", "-c", e["cmd"]], cwd=run_dir, env=env, check=True)
    if e.get("slim"):
        ents = entries_by_cubin(dst / "case.toml")
        for f in e["files"]:
            if f.endswith(".cubin") and ents.get(f):
                p = str(out_dir / f)
                subprocess.run(
                    [sys.executable, str(ROOT / "tools/cubin_slim.py"), p, p, *sorted(ents[f])],
                    check=True,
                )
    for f in e["files"]:
        src = out_dir / f
        if not src.is_file():
            sys.exit(f"[{name}] the capture did not produce {f} (in {out_dir})")
        shutil.copyfile(src, dst / f)
        print(f"[{name}] wrote {dst / f} ({src.stat().st_size} bytes)")
    if keep:
        print(f"[{name}] scratch kept: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="*", help="fixture names (regex allowed)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--missing", action="store_true", help="every entry with a missing file")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    es = manifest()
    if a.list:
        for e in es:
            miss = [f for f in e["files"] if not (FIX / e["name"] / f).is_file()]
            state = f"missing {', '.join(miss)}" if miss else "present"
            print(f"{e['name']:28} {e['vendor']:58} {state}")
        return
    if a.all:
        sel = es
    elif a.missing:
        sel = [e for e in es if any(not (FIX / e["name"] / f).is_file() for f in e["files"])]
    else:
        sel = [e for e in es if any(re.fullmatch(n, e["name"]) for n in a.names)]
        if not sel:
            sys.exit("no matching fixture; see --list")
    for e in sel:
        regen(e, a.keep)


if __name__ == "__main__":
    main()

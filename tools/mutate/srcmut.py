# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Source mutants of ptxv's own rule tables (docs/soundness-harness.md, "source mutants").

Each mutant removes ONE entry of a table a soundness argument depends on, builds ptxv from a
detached copy of the tree, and runs the fixture variants that entry exists for. A mutant is
caught when some run's classes differ from the variant's expectation (the fixture notices the
missing entry); it survives otherwise, which means the entry has no fixture.

    srcmut.py --out DIR [--only NAME] [--jobs N]

Entries that no fixture can reach carry a documented exclusion (`EXCLUDED`) instead of a
silent survival.
"""

import argparse
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SrcMutant:
    name: str
    file: str
    find: str  # exact text, must occur once
    replace: str
    # (fixture, variant, front, extra check flags)
    runs: list
    # (cargo package, test binary, test name): the mutant is caught when the test fails
    # (claims checked against recorded GPU outputs, e.g. num's error bounds).
    tests: list = field(default_factory=list)


# num.rs `value_fault_operands`: each entry removed, under the CTA reductions.
REP = ["--certify", "--rep"]
MUTANTS = [
    # wellsync.rs `post_inst`: one token for every member again (merged splits count as
    # one convergence event; MISS 30).
    SrcMutant(
        "tok-merge",
        "crates/sim/src/wellsync.rs",
        "            let nt = match remap.iter().find(|x| x.0 == old) {",
        "            let nt = match remap.first() {",
        [("e_lockstep_nobsync", "NOBSYNC", "sass", ["--certify"])],
    ),
    SrcMutant(
        "vfo-div",
        "crates/sim/src/num.rs",
        "IntOp::DivU | IntOp::DivS | IntOp::RemU | IntOp::RemS => regs(b),",
        "IntOp::DivU | IntOp::DivS | IntOp::RemU | IntOp::RemS => vec![],",
        [("e_rep_divzero", v, "ptx", REP) for v in ("DATA", "POS")],
    ),
    SrcMutant(
        "vfo-cvta",
        "crates/sim/src/num.rs",
        "            to_generic: false,\n            ..\n        } => regs(a),",
        "            to_generic: false,\n            ..\n        } => vec![],",
        [("e_rep_divzero", "CVTA", "ptx", REP)],
    ),
    SrcMutant(
        "vfo-ldconst",
        "crates/sim/src/num.rs",
        "Op::LdConst { base, .. } => regs(base),",
        "Op::LdConst { base, .. } => vec![],",
        [("e_rep_divzero", "LDC", "sass", REP)],
    ),
]

# Entries no fixture reaches, with the reason.
EXCLUDED = {
    "vfo-iadd3": (
        "IADD3 with one carry-out: ptxas emits a single carry-out only where the operands bound "
        "the carry to 1 (a zero or carry-in third operand), so no PTX reaches a carry of 2, and "
        "the SASS front has no assembler for a hand-built cubin (variants can only NOP "
        "instructions of a compiled one)"
    ),
}


def expect_for(case: Path, variant: str, front: str) -> list[str]:
    c = tomllib.loads((case / "case.toml").read_text())
    for v in c.get("variants", []):
        if v["id"] == variant:
            if front == "sass" and "expect_sass" in v:
                return sorted(v["expect_sass"])
            return sorted(v["expect"])
    raise KeyError(variant)


def classes_of(out: str) -> list[str] | None:
    for line in out.splitlines():
        if line.startswith(("PASS ", "FAIL ")):
            got = line.split("[", 1)[1].split("]", 1)[0]
            return sorted(x for x in got.split(",") if x)
    return None


def build(tree: Path, target: Path) -> bool:
    env = dict(os.environ, CARGO_TARGET_DIR=str(target))
    # PTXV_JOB_WRAPPER: an optional command prefix for heavy jobs (a memory-capped slot runner).
    wrap = os.environ.get("PTXV_JOB_WRAPPER", "").split()
    cmd = [*wrap, "cargo", "build", "--release", "-p", "ptxv"]
    r = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True)
    return r.returncode == 0


def run_mutant(m: SrcMutant, out: Path) -> dict:
    tree = out / "tree" / m.name
    target = out / "target"
    if tree.exists():
        subprocess.run(
            ["git", "-C", str(ROOT), "worktree", "remove", "--force", str(tree)],
            check=False,
        )
    subprocess.run(
        ["git", "-C", str(ROOT), "worktree", "add", "--detach", str(tree), "HEAD"],
        check=True,
        capture_output=True,
    )
    # The working tree's uncommitted state too (the mutant is of what is being checked).
    diff = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "HEAD"], capture_output=True, text=True
    ).stdout
    if diff:
        subprocess.run(
            ["git", "-C", str(tree), "apply"], input=diff, text=True, check=True
        )
    f = tree / m.file
    src = f.read_text()
    if src.count(m.find) != 1:
        return {"name": m.name, "status": "error", "why": "find text not unique"}
    f.write_text(src.replace(m.find, m.replace))
    if not build(tree, target):
        return {"name": m.name, "status": "error", "why": "build failed"}
    runs = []
    caught = False
    for case, variant, front, flags in m.runs:
        cdir = ROOT / "fixtures" / case
        cmd = [
            str(target / "release/ptxv"),
            "check",
            str(cdir),
            "--variant",
            variant,
            "--front",
            front,
            "--no-input-coverage",
            "--out",
            str(out / "check" / m.name),
            *flags,
        ]
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
            timeout=900,
        )
        got = classes_of(r.stdout)
        want = expect_for(cdir, variant, front)
        hit = got != want
        caught |= hit
        runs.append(
            {
                "case": case,
                "variant": variant,
                "front": front,
                "got": got,
                "want": want,
                "caught": hit,
            }
        )
    wrap = os.environ.get("PTXV_JOB_WRAPPER", "").split()
    for pkg, binary, name in m.tests:
        cmd = [
            *wrap,
            "cargo",
            "test",
            "--release",
            "-p",
            pkg,
            "--test",
            binary,
            "--",
            "--exact",
            name,
        ]
        r = subprocess.run(
            cmd,
            cwd=tree,
            capture_output=True,
            text=True,
            timeout=3600,
            env=dict(os.environ, CARGO_TARGET_DIR=str(target), CUDA_VISIBLE_DEVICES=""),
        )
        ran = f"test {name} ..." in r.stdout
        hit = ran and r.returncode != 0
        caught |= hit
        runs.append(
            {
                "test": f"{pkg}/{binary}/{name}",
                "ran": ran,
                "rc": r.returncode,
                "caught": hit,
            }
        )
    subprocess.run(
        ["git", "-C", str(ROOT), "worktree", "remove", "--force", str(tree)],
        check=False,
    )
    return {"name": m.name, "status": "caught" if caught else "survived", "runs": runs}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only")
    a = ap.parse_args(argv)
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    res = []
    for m in MUTANTS:
        if a.only and m.name != a.only:
            continue
        r = run_mutant(m, a.out)
        print(f"{r['name']:14} {r['status']}", flush=True)
        res.append(r)
    for name, why in EXCLUDED.items():
        print(f"{name:14} excluded: {why}")
        res.append({"name": name, "status": "excluded", "why": why})
    (a.out / "srcmut.json").write_text(json.dumps(res, indent=1))
    return 0 if all(r["status"] in ("caught", "excluded") for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())

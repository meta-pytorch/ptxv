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
import shutil
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
    # (package, test binary, name filter): `cargo test --release -p PKG --test BIN FILTER` in
    # the mutant's tree; the mutant is caught when the test fails (adversarial tests that build
    # their machines by hand, crates/sim/tests/adversarial.rs; claims checked against recorded
    # GPU outputs, e.g. num's error bounds).
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
    # exec.rs `split_enabled`: a SASS SHFL waits for the rest of the warp again (MISS 39;
    # GB300 runs it with the arriving split only).
    SrcMutant(
        "shfl-wait",
        "crates/sim/src/exec.rs",
        "        if self.k.front == Front::Ptx\n            && (inst.op.is_warp_aligned()",
        "        if (self.k.front == Front::Ptx || matches!(inst.op, Op::Shfl { .. }))\n            && (inst.op.is_warp_aligned()",
        [("e_warpcoll_split", "NOBSYNC", "sass", [])],
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

# Adversarial tests (crates/sim/tests/adversarial.rs): each mutant weakens one proof rule.
ADV = "adversarial"
MUTANTS += [
    # E-CONST-BRA: the execution check of folded branches is what the rule rests on.
    SrcMutant(
        "cbra-nocheck",
        "crates/sim/src/exec.rs",
        "if members.iter().any(|t| active.contains(t) != v) {",
        "if false && members.iter().any(|t| active.contains(t) != v) {",
        [],
        [("ptxv-sim", ADV, "e_const_bra")],
    ),
    # E-RET: the execution check of return targets.
    SrcMutant(
        "ret-nocheck",
        "crates/sim/src/exec.rs",
        '                if !site {\n                    let msg = format!(\n                        "RET at',
        '                if false && !site {\n                    let msg = format!(\n                        "RET at',
        [],
        [("ptxv-sim", ADV, "e_ret")],
    ),
    # E-ELECT (PTX): only the lowest leader explored.
    SrcMutant(
        "elect-lowonly",
        "crates/sim/src/explore.rs",
        "    let mut hi = m0.clone();\n    hi.elect_high = true;",
        "    let mut hi = m0.clone();\n    hi.elect_high = false;",
        [("e_elect", "B46", "ptx", [])],
    ),
    # E-ELECT (SASS): the highest active lane instead of GB300's lowest.
    SrcMutant(
        "elect-high-sass",
        "crates/sim/src/exec.rs",
        "let lowest = active.iter().copied().min_by_key(|&t| self.threads[t].lane);",
        "let lowest = active.iter().copied().max_by_key(|&t| self.threads[t].lane);",
        [],
        [("ptxv-sim", ADV, "e_elect_lane")],
    ),
    # S-BRA-COND: `@P0 BRA P1` / `@P0 EXIT P1` taken where the guard holds, ignoring P1.
    SrcMutant(
        "bracond-ignore",
        "crates/sim/src/exec.rs",
        "                            && cond.is_none_or(|c| self.prd(t, c)));",
        "                            && true);",
        [],
        [
            ("ptxv-sim", "fixtures", "probe_div_ftz_matches_hardware"),
            ("ptxv-sim", "fixtures", "r4_kernel_classes"),
        ],
    ),
    # E-BSYNC: a BSYNC waits only for the lanes that reached it (not for its BSSY's mask).
    SrcMutant(
        "bsync-own",
        "crates/sim/src/exec.rs",
        "w.bmask[*b as usize] | own | parked",
        "own | parked",
        [("e_lockstep_nobsync", "good", "sass", [])],
        [("ptxv-sim", ADV, "e_bsync")],
    ),
    # E-BAR-EXIT: a counted barrier credits one warp's arrivals that never come (as if a
    # wholly exited warp arrived); probes/bar_exit `bar.sync 0, 64` with a warp exited and
    # probes/bar_exit_warp `bar.sync 1, 96` with the third warp exited hang GB300.
    SrcMutant(
        "barexit-phantom",
        "crates/sim/src/exec.rs",
        "                32usize.saturating_sub(lanes.len()) + gone",
        "                32usize.saturating_sub(lanes.len()) + gone + 16",
        [],
        [("ptxv-sim", "fixtures", "probe_bar_exit")],
    ),
    # E-ALIGNED: no violation for an aligned barrier executed by part of the warp.
    SrcMutant(
        "aligned-noviol",
        "crates/sim/src/exec.rs",
        "            if active.len() < live.len() {",
        "            if false && active.len() < live.len() {",
        [("e_aligned_bar", "AB3", "ptx", []), ("e_aligned_bar", "AB1", "ptx", [])],
    ),
    # E-DRF (TMEM): a TMEM read or write after an unordered TMEM write is no race.
    SrcMutant(
        "tmem-norace",
        "crates/sim/src/exec.rs",
        "            let sh = tm.sh.get(i);\n            if let Some(w) = sh.w {\n                if w.actor != actor && !w.hb(vc) {",
        "            let sh = tm.sh.get(i);\n            if let Some(w) = sh.w {\n                if false && w.actor != actor && !w.hb(vc) {",
        [("e_drf_tmem", "NOBAR", "ptx", []), ("e_drf_tmem", "NOBAR", "sass", [])],
    ),
]

# Entries no fixture reaches, with the reason.
EXCLUDED = {
    "cbra-agree": (
        "E-CONST-BRA's fold across grids (a pc reached by several grids is folded only where they "
        "agree): grids share code only within one launch (a preferred cluster shape), with equal "
        "parameters; ptxas reads the shape-dependent table through a register offset "
        "(c[0x0][x - 0xa0 * SR_CgaSize], S-CGA-SHAPE), which is never a folded constant. A "
        "multi-kernel launch concatenates one code copy per launch (fixtures/e_const_bra2)"
    ),
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
    # Untracked files too (new fixtures and tests are not in `git diff`).
    new = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    for rel in filter(None, new):
        dst = tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    f = tree / m.file
    src = f.read_text()
    if src.count(m.find) != 1:
        return {"name": m.name, "status": "error", "why": "find text not unique"}
    f.write_text(src.replace(m.find, m.replace))
    if not build(tree, target):
        return {"name": m.name, "status": "error", "why": "build failed"}
    runs = []
    caught = False
    for pkg, binary, filt in m.tests:
        env = dict(os.environ, CARGO_TARGET_DIR=str(target), CUDA_VISIBLE_DEVICES="")
        wrap = os.environ.get("PTXV_JOB_WRAPPER", "").split()
        cmd = [*wrap, "cargo", "test", "--release", "-p", pkg, "--test", binary, filt]
        r = subprocess.run(
            cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=3600
        )
        # Caught only by a test that ran and failed: a build error, a missing test target or a
        # filter matching nothing is an error of the harness.
        ran = any(
            ln.startswith("running ") and not ln.startswith("running 0 ")
            for ln in r.stdout.splitlines()
        )
        if "could not compile" in r.stderr or not ran:
            return {
                "name": m.name,
                "status": "error",
                "why": f"test did not run: {binary} {filt}",
            }
        hit = r.returncode != 0
        caught |= hit
        runs.append(
            {"test": f"{pkg}/{binary}/{filt}", "rc": r.returncode, "caught": hit}
        )
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
    ap.add_argument("--only", help="comma-separated mutant names")
    a = ap.parse_args(argv)
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    res = []
    for m in MUTANTS:
        if a.only and m.name not in a.only.split(","):
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

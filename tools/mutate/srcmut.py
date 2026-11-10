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
    # S-ALIGN: only 4-byte alignment checked (8- and 16-byte accesses at +4 / +8 pass).
    SrcMutant(
        "align-4",
        "crates/sim/src/exec.rs",
        "if !a.is_multiple_of(n as u64) {",
        "if !a.is_multiple_of((n as u64).min(4)) {",
        [
            ("e_misaligned", v, f, [])
            for v in ("MA1", "MA2", "MA3")
            for f in ("ptx", "sass")
        ],
    ),
    # S-DSMEM: a shared::cta address naming another CTA's rank is taken as the CTA's own.
    SrcMutant(
        "dsmem-rank",
        "crates/sim/src/exec.rs",
        "if self.launch.g(cta as u32).explicit_cluster && a >> 24 != own {",
        "if false && self.launch.g(cta as u32).explicit_cluster && a >> 24 != own {",
        [("e_dsmem_rank", "PEER", f, []) for f in ("ptx", "sass")],
        [("ptxv-sim", ADV, "e_dsmem")],
    ),
    # S-SMEM-ADDR32: shared address parts add in 64 bits.
    SrcMutant(
        "addr32-nowrap",
        "crates/sim/src/exec.rs",
        "Space::Shared | Space::SharedCluster => (space, a & 0xffff_ffff),",
        "Space::Shared | Space::SharedCluster => (space, a),",
        [],
        [("ptxv-sim", "fixtures", "probe_smem_addr_wrap_matches_hardware")],
    ),
    # S-GLOBAL-INIT: module globals start zero instead of their ELF initial bytes.
    SrcMutant(
        "ginit-zero",
        "crates/sim/src/lib.rs",
        "init: g.init.clone().unwrap_or_else(|| vec![0; g.size as usize]),",
        "init: vec![0; g.size as usize],",
        [],
        [
            ("ptxv-sim", "fixtures", "s_globals_variants"),
            ("ptxv-sim", "fixtures", "probe_ptx_globals_matches_hardware"),
        ],
    ),
    # E-STSM-DUP: lanes of one stmatrix writing one word with different values: no violation.
    SrcMutant(
        "stsm-dup-nocheck",
        "crates/sim/src/exec.rs",
        "                                    if w0 != w {",
        "                                    if false && w0 != w {",
        [("e_stsm_warp", "dupdiff", f, []) for f in ("ptx", "sass")],
    ),
    # E-CLUSTER-BAR: a relaxed PTX cluster arrive releases every earlier mbarrier init, with no
    # fence.mbarrier_init.
    SrcMutant(
        "clbar-relaxed-inits",
        "crates/sim/src/exec.rs",
        "                    let mut v = th.minit_vc.clone();",
        "                    let mut v = th.vc.clone();",
        [("r1c_cluster", "B14", "ptx", [])],
    ),
    # E-BAR-RED: bar.red.and gives "any" instead of "all".
    SrcMutant(
        "barred-and-or",
        "crates/sim/src/exec.rs",
        "                        B::And => all,",
        "                        B::And => n > 0,",
        [],
        [("ptxv-sim", "fixtures", "probe_bar_red_matches_hardware")],
    ),
    # E-BAR-ARRIVE: a bar.arrive also releases the arriving thread's later accesses.
    SrcMutant(
        "bararv-notick",
        "crates/sim/src/exec.rs",
        "                    // Later accesses of the arriving thread are not released.\n                    self.tick(t);",
        "                    // Later accesses of the arriving thread are not released.",
        [("e_bararv", "AB1", f, []) for f in ("ptx", "sass")],
    ),
    # E-BAR-ARRIVE / E-BAR-OVER reduction: every bar.arrive taken to land in the current
    # generation (e_bararv AB3 loses its deadlock).
    SrcMutant(
        "bararv-gen",
        "crates/sim/src/exec.rs",
        "if arrive && !(self.fp.is_some() && self.arv_current(t, cta, id, count)) {",
        "if arrive && !self.fp.is_some() {",
        [("e_bararv", v, f, []) for v in ("AB3", "AB4") for f in ("ptx", "sass")],
    ),
    # S-MBAR: a phase completes when its arrivals are in, whatever its pending transaction
    # bytes (r0_tma_sum B01: expect_tx larger than the TMA, a hang on GB300).
    SrcMutant(
        "mbar-tx-ignore",
        "crates/sim/src/exec.rs",
        "        if mb.pending == 0 && mb.tx == 0 {\n            let p = (mb.phase % 2) as usize;",
        "        if mb.pending == 0 {\n            let p = (mb.phase % 2) as usize;",
        [("r0_tma_sum", "B01", f, []) for f in ("ptx", "sass")],
    ),
    # S-MBAR-MEM: the expected count decoded from an STS.64-initialized mbarrier one short.
    SrcMutant(
        "mbarmem-count",
        "crates/sim/src/exec.rs",
        "        let n = 0x10_0000 - x;",
        "        let n = 0x10_0000 - x - 1;",
        [("e_mbar_sts64", "COUNT3", "sass", [])],
        [("ptxv-sim", ADV, "e_mbar_sts64")],
    ),
    # E-BAR-OVER: a counted barrier taken as over-subscribed only past twice its count (the
    # explorer then treats arrivals as commuting where surplus ones start a new generation).
    SrcMutant(
        "barover-2x",
        "crates/sim/src/explore.rs",
        "                if n > c as usize {\n                    return true;\n                }\n            }\n        }\n        // ASSUME(E-BAR-EXIT): a warp that arrives counts the warp size.\n        per_warp.values().map(|&k| k.max(32)).sum::<usize>() > c as usize",
        "                if n > 2 * c as usize {\n                    return true;\n                }\n            }\n        }\n        // ASSUME(E-BAR-EXIT): a warp that arrives counts the warp size.\n        per_warp.values().map(|&k| k.max(32)).sum::<usize>() > 2 * c as usize",
        [
            ("e_barover", v, f, [])
            for v in ("BO1", "BO2", "BO3")
            for f in ("ptx", "sass")
        ],
    ),
    # E-POLL-ANY: a strong load whose step leaves different live registers per value commutes
    # with its location's writes (liveness check skipped).
    SrcMutant(
        "vip-nolive",
        "crates/sim/src/exec.rs",
        "        if dead.is_empty() {\n            return true;\n        }\n        o0.iter().all(",
        "        if true {\n            return true;\n        }\n        o0.iter().all(",
        [("e_poll_any_live", "good", f, []) for f in ("ptx", "sass")],
    ),
    # E-ASYNC-STEP: a SASS bulk wait on one scoreboard drains every group (soundness-harness
    # miss 3, fixtures/e_gmm_epilogue_reuse).
    SrcMutant(
        "bulk-anysb",
        "crates/sim/src/exec.rs",
        "            .filter(|&g| {\n                sbs.is_none_or(|m| {",
        "            .filter(|&g| {\n                true || sbs.is_none_or(|m| {",
        [],
        [("ptxv-sim", "wellsync", "tma_store_wait_counts_its_own_scoreboard")],
    ),
    # S-SCOPE: .cta scope taken to include every thread.
    SrcMutant(
        "scope-cta-all",
        "crates/sim/src/exec.rs",
        "            Scope::Cta => cx == cy,",
        "            Scope::Cta => true,",
        [
            ("e_scope_cta", v, f, [])
            for v in ("CTAREL", "CTAACQ")
            for f in ("ptx", "sass")
        ],
        [("ptxv-sim", ADV, "e_scope")],
    ),
    # E-ALLOC: a TMEM alloc is enabled even when the CTA has no free columns.
    SrcMutant(
        "alloc-noblock",
        "crates/sim/src/exec.rs",
        "                        .tmem_find(self.threads[t].cta, (n / 32) as u32)\n                        .is_some(),",
        "                        .tmem_find(self.threads[t].cta, (n / 32) as u32)\n                        .is_some() || true,",
        [("e_tmem_alloc_block", "HOLD", f, []) for f in ("ptx", "sass")],
    ),
    # L-OPCLASS: MUFU dropped from the variable-latency set (its results taken as interlocked).
    SrcMutant(
        "opclass-nomufu",
        "crates/sass/src/scoreboard.rs",
        '    "UTCATOMSWS",\n    "MUFU",\n',
        '    "UTCATOMSWS",\n',
        [("e_sb_mufu", v, "sass", []) for v in ("NOWAIT", "NOSB")],
    ),
]

# Numerics and lowering rules: each mutant changes one hardware-measured rule; the replay test
# of its GB300 probe must fail.
MUTANTS += [
    # N-FADD / N-FALU: a NaN operand of FADD / FFMA passes its payload through instead of the
    # canonical NaN (FADD is f32_fma(x, 1, y); `f32_add_rn` is not on its path).
    SrcMutant(
        "fma-nan-payload",
        "crates/sim/src/fp.rs",
        "    if f32_is_nan(a) || f32_is_nan(b) || c.is_some_and(f32_is_nan) {\n        return F32_CANON_NAN;",
        "    if f32_is_nan(a) || f32_is_nan(b) || c.is_some_and(f32_is_nan) {\n        return if f32_is_nan(a) { a } else { b | 0x40_0000 };",
        [],
        [("ptxv-sim", ADV, "e_fadd_nan"), ("ptxv-sim", "fixtures", "s_alu")],
    ),
    # N-FCHK: dividend and divisor swapped in the exponent-range test.
    SrcMutant(
        "fchk-swap",
        "crates/sim/src/exec.rs",
        "                self.wp(t, *d, fp::fchk(x, y));",
        "                self.wp(t, *d, fp::fchk(y, x));",
        [],
        [("ptxv-sim", "fixtures", "probe_fchk")],
    ),
    # N-XORSIGN: min and max exchanged under .XORSIGN.
    SrcMutant(
        "xorsign-minmax",
        "crates/sim/src/exec.rs",
        "let r = fp::f32_minmax_xorsign(x, y, lo, *nan, *ftz);",
        "let r = fp::f32_minmax_xorsign(x, y, !lo, *nan, *ftz);",
        [],
        [("ptxv-sim", "fixtures", "probe_fmnmx_xorsign")],
    ),
    # N-F2I16: F2I.S16 zero-extended instead of sign-extended in the register.
    SrcMutant(
        "f2i16-zext",
        "crates/sim/src/exec.rs",
        "                    (16, true) => x as i16 as i32 as u32 as u64,",
        "                    (16, true) => x as i16 as u16 as u64,",
        [],
        [("ptxv-sim", "fixtures", "probe_f2i16")],
    ),
    # N-F2FP-RZ: overflow to infinity instead of the largest finite f16.
    SrcMutant(
        "f2fprz-inf",
        "crates/sim/src/fp.rs",
        "        143.. => s | 0x7bff,",
        "        143.. => s | 0x7c00,",
        [],
        [("ptxv-sim", "fixtures", "probe_f2fp_rz")],
    ),
    # N-QMUL4: a NaN product gives 0x7e instead of 0x7f.
    SrcMutant(
        "qmul4-nan",
        "crates/sim/src/lowp.rs",
        "        let c = if p.is_nan() {\n            0x7f\n",
        "        let c = if p.is_nan() {\n            0x7e\n",
        [],
        [("ptxv-sim", "fixtures", "probe_qmul4")],
    ),
    # N-IMMA: the s32 result wraps instead of saturating.
    SrcMutant(
        "imma-wrap",
        "crates/sim/src/exec.rs",
        "let r = s.clamp(i32::MIN as i64, i32::MAX as i64) as i32 as u32;",
        "let r = s as i32 as u32;",
        [],
        [("ptxv-sim", "fixtures", "probe_imma_s8")],
    ),
    # L-SHF: shift amounts mod the width instead of clamped.
    SrcMutant(
        "shf-wrap",
        "crates/sim/src/exec.rs",
        "        n.min(if wide { 64 } else { 32 })",
        "        n & if wide { 63 } else { 31 }",
        [],
        [("ptxv-sim", "fixtures", "s_shf")],
    ),
    # L-LOP3-P: the default LOP3 predicate mode ANDs Pq instead of ORing it.
    SrcMutant(
        "lop3p-and",
        "crates/sim/src/exec.rs",
        "                            Some((q, false)) => nz || q,",
        "                            Some((q, false)) => nz && q,",
        [],
        [("ptxv-sim", "fixtures", "probe_lop3_pmode")],
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
    try:
        return run_mutant_in_tree(m, out)
    finally:
        tree = out / "tree" / m.name
        if tree.exists() and not os.environ.get("SRCMUT_KEEP"):
            subprocess.run(
                ["git", "-C", str(ROOT), "worktree", "remove", "--force", str(tree)],
                check=False,
            )


def run_mutant_in_tree(m: SrcMutant, out: Path) -> dict:
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
            tail = r.stderr.strip().splitlines()[-3:]
            return {
                "name": m.name,
                "status": "error",
                "why": f"test did not run: {binary} {filt}: {tail}",
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

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
    srcmut.py --check-anchors [--only NAME] [--root DIR]

Entries that no fixture can reach carry a documented exclusion (`EXCLUDED`) instead of a
silent survival.

--check-anchors builds and runs nothing: it prints every mutant whose find text does not occur
exactly once in its file in this checkout, or in the tree --root names (the anchors test plants
a broken anchor in a scratch tree), and exits 1 if there is one (a run reports such a mutant as
an error without building it). Either mode exits 2, before a run creates --out, when --only
is given but names nothing, when it names a name that is not a mutant that runs, or when no
mutant is chosen.
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
    # GPU outputs, e.g. num's error bounds). BIN "lib" runs the package's unit tests, "bin:NAME"
    # binary NAME's (FILTER their full path, `module::tests::name`).
    tests: list = field(default_factory=list)
    # Python (directory, file pattern, -k selector): `$PTXV_PYTHON -m unittest discover -s DIR
    # -p PAT -k SEL` in the mutant's tree (tooling outside the binary, e.g. fb/kernel_db's
    # status rules, the python package's readers; PTXV_PYTHON has the package's dependencies,
    # default this interpreter); caught when a test that ran fails.
    py: list = field(default_factory=list)


def test_target(pkg: str, binary: str) -> list[str]:
    """The cargo test arguments selecting `binary` of package `pkg` (`SrcMutant.tests`)."""
    if binary == "lib":
        return ["-p", pkg, "--lib"]
    if binary.startswith("bin:"):
        return ["-p", pkg, "--bin", binary[4:]]
    return ["-p", pkg, "--test", binary]


# num.rs `value_fault_operands`: each entry removed, under the CTA reductions.
REP = ["--certify", "--rep"]
# E-BAR-OVER walk cache unit tests (crates/sim/src/explore.rs `walk_cache_tests`).
LIB = ("ptxv-sim", "lib", "explore::walk_cache_tests::walk_cache_keys_every_input_of_a_walk")
CUT = ("ptxv-sim", "lib", "explore::walk_cache_tests::walk_cache_keys_the_cut")
WCT = "explore::walk_cache_tests::"
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
    # exec.rs `warpsync_release`: the GB300 profile's BSYNC waits for lanes parked at a
    # WARPSYNC.COLLECTIVE again (E-HW-BSYNC-COLLECTIVE): the profile verdict deadlocks.
    SrcMutant(
        "bsync-coll",
        "crates/sim/src/exec.rs",
        "            if here.is_empty() || !lanes.iter().all(|&i| parked_here(i) || coll(i)) {",
        "            if true || here.is_empty() || !lanes.iter().all(|&i| parked_here(i) || coll(i)) {",
        [],
        tests=[("ptxv-sim", "fixtures", "bsync_releases_beside_a_parked_collective")],
    ),
    # exec.rs: an EXIT no longer lets in a WARPSYNC.COLLECTIVE group whose live lanes are all
    # parked (E-COLLECTIVE): the profile verdict of e_bsync_parked_collective deadlocks.
    SrcMutant(
        "exit-coll",
        "crates/sim/src/exec.rs",
        "                self.exit_collective_release(warp)?;",
        "",
        [],
        tests=[("ptxv-sim", "fixtures", "bsync_releases_beside_a_parked_collective")],
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
    # exec.rs `split_enabled`: griddepcontrol.wait does not block (E-PDL).
    SrcMutant(
        "pdl-noblock",
        "crates/sim/src/exec.rs",
        "                self.launch\n                    .prev_grid(gi)\n                    .is_none_or(|p| self.grid_complete(p))",
        "                let _ = gi;\n                true",
        [("e_pdl_wait", "AFTER", f, []) for f in ("ptx", "sass")],
    ),
    # spin.rs `blocked_poll_read`: a blocked poll of never-written memory is not reported
    # again (MISS 40).
    SrcMutant(
        "spin-undef-block",
        "crates/sim/src/spin.rs",
        "    if !undef {\n        return;\n    }\n    let msg",
        "    if true {\n        return;\n    }\n    let msg",
        [("s_smem_acquire", "NOINIT", f, []) for f in ("ptx", "sass")],
    ),
    # sass `smem_acquires`: no LDS is read as ld.acquire.cta (S-SMEM-STRONG).
    SrcMutant(
        "smem-noacq",
        "crates/sass/src/lib.rs",
        "        if acq {\n            if let Op::Ld { sync, .. } = &mut code[i].op {",
        "        if acq && false {\n            if let Op::Ld { sync, .. } = &mut code[i].op {",
        [("s_smem_acquire", v, "sass", []) for v in ("good", "RELAX")],
    ),
    # sim `f32_mul_scaled`: the product rounded, then scaled (two roundings; MISS 55).
    SrcMutant(
        "fmul-double-round",
        "crates/sim/src/fp.rs",
        "    round_f32_ftz(ps, pm, ea + eb + k, rnd, ftz)\n}",
        "    let p = round_f32_ftz(ps, pm, ea + eb, rnd, ftz);\n    f32_fma(p, ((127 + k) as u32) << 23, None, rnd, ftz)\n}",
        [],
        tests=[("ptxv-sim", "fixtures", "probe_fmul_scale_matches_hardware")],
    ),
    # ptx `fmul_scale_pairs`: the fold applied after a .rn product (PTX-FMUL-SCALE).
    SrcMutant(
        "fmul-fold-rn",
        "crates/ptx/src/lib.rs",
        '        if rnd1 == Some("rn") || sat1 {\n            continue;\n        }',
        '        let rnd1 = if rnd1 == Some("rn") { None } else { rnd1 };\n        if sat1 {\n            continue;\n        }',
        [],
        tests=[("ptxv-sim", "fixtures", "probe_fmul_scale_matches_hardware")],
    ),
    # ptx `div_approx_ftz`: the product without .ftz (no FFMA.FTZ contraction, subnormal
    # products kept).
    SrcMutant(
        "div-approx-noftz",
        "crates/ptx/src/lib.rs",
        '            text: format!("{guard}mul.ftz.f32 {d}, {a}, {r}"),',
        '            text: format!("{guard}mul.f32 {d}, {a}, {r}"),',
        [],
        tests=[("ptxv-sim", "fixtures", "probe_div_approx_ftz_matches_hardware")],
    ),
    # sass `atomg_op`: .BF16x4 read as one BF16x2 atomic (the high half dropped; N-BF16X2).
    SrcMutant(
        "bf16x4-half",
        "crates/sass/src/lib.rs",
        '["ADD", "BF16x4", "RN"] => Some((AtomOp::AddBf16x2, 8)),',
        '["ADD", "BF16x4", "RN"] => Some((AtomOp::AddBf16x2, 4)),',
        [],
        tests=[("ptxv-sim", "fixtures", "probe_atom_bf16x4_matches_hardware")],
    ),
    # sim `build_launch`: a scratch workspace that is also an output counted dead (S-INIT-DEAD).
    SrcMutant(
        "dead-output",
        "crates/sim/src/lib.rs",
        'dead: b.init == "undef" && b.scratch && !m.outputs.contains(&b.name),',
        'dead: b.init == "undef" && b.scratch,',
        [("s_init_dead_out", "good", f, []) for f in ("ptx", "sass")],
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
    # E-ELECT taint (explore.rs `elect_sensitivity_kernel`): elect's lane number
    # leader-dependent only again (MISS 49).
    SrcMutant(
        "elect-lane-ls",
        "crates/sim/src/explore.rs",
        "                        ls.insert(*r);\n                        lv.insert(*r);",
        "                        ls.insert(*r);",
        [("e_elect_lane_number", "good", "ptx", [])],
    ),
    # E-ELECT taint: any register once assigned %tid.x is %tid.x at every use (MISS 50).
    SrcMutant(
        "elect-tidx-flow",
        "crates/sim/src/explore.rs",
        "if i.guard.reg.is_none() && defs.get(d) == Some(&1) => Some(*d),",
        "if i.guard.reg.is_none() => Some(*d),",
        [("e_elect_tid_redef", "good", "ptx", [])],
    ),
    # E-ELECT taint memory: tcgen05.cp reads no shared memory (MISS 51).
    SrcMutant(
        "elect-tccp-noread",
        "crates/sim/src/explore.rs",
        "        Op::TcCp { .. } => (SMEM, &[0]),",
        "        Op::TcCp { .. } => (NONE, &[0]),",
        [("e_elect_tccp", "good", "ptx", [])],
    ),
    # E-ELECT taint memory: tcgen05.mma reads no shared operands (MISS 51).
    SrcMutant(
        "elect-mma-noread",
        "crates/sim/src/explore.rs",
        "        Op::Mma { .. } => (&[0, 2], &[0]),",
        "        Op::Mma { .. } => (&[0], &[0]),",
        [],
        [("ptxv-sim", "fixtures", "e_elect_taint_mutants")],
    ),
    # E-ELECT taint memory: a load reached by a leader-dependent write is not leader-dependent.
    SrcMutant(
        "elect-mem-ls",
        "crates/sim/src/explore.rs",
        "                let m_ls = rd.iter().any(|&sp| mls[sp]);",
        "                let m_ls = false;",
        [("e_elect_tccp", "good", "ptx", [])],
        [("ptxv-sim", "fixtures", "e_elect_taint_mutants")],
    ),
    # E-ELECT taint memory: per-lane storage (local, tensor memory rows) read as one location.
    SrcMutant(
        "elect-mem-perlane",
        "crates/sim/src/explore.rs",
        "                lv_in |= rd.iter().any(|&sp| sp <= 1 && mlv[sp] && !tile);",
        "                lv_in |= rd.iter().any(|&sp| sp > 3 && mlv[sp] && !tile);",
        [],
        [("ptxv-sim", "fixtures", "e_elect_taint_mutants")],
    ),
    # E-ELECT taint memory: tcgen05.mma reads tensor memory per lane (a precision entry: the
    # deciding gate must clear the leader's MMA over rows each lane stored).
    SrcMutant(
        "elect-mma-tile",
        "crates/sim/src/explore.rs",
        "                let tile = matches!(inst.op, Op::Mma { .. });",
        "                let tile = false;",
        [],
        [("ptxv-sim", "fixtures", "e_elect_taint_mutants")],
    ),
    # E-ELECT lane classes (elect_lanes.rs `lane_classes`): elect's lane number read.
    SrcMutant(
        "elect-cls-lanereg",
        "crates/sim/src/elect_lanes.rs",
        "            if srcs.iter().any(|r| lane_regs.contains(r)) {",
        "            if false && srcs.iter().any(|r| lane_regs.contains(r)) {",
        [],
        [("ptxv-sim", "fixtures", "e_elect_lane_classes")],
    ),
    # E-ELECT lane classes: a collective under or of elect's predicate.
    SrcMutant(
        "elect-cls-coll",
        "crates/sim/src/elect_lanes.rs",
        "            if tainted && collective(&i.op) {",
        "            if false && tainted && collective(&i.op) {",
        [],
        [("ptxv-sim", "fixtures", "e_elect_lane_classes")],
    ),
    # E-ELECT lane classes: a write of elect's predicate by lanes it does not select.
    SrcMutant(
        "elect-cls-uncontrolled",
        "crates/sim/src/elect_lanes.rs",
        "            if !controlled\n",
        "            if false && !controlled\n",
        [],
        [("ptxv-sim", "fixtures", "e_elect_lane_classes")],
    ),
    # E-ELECT lane classes: the operands of effects no longer tell lanes apart.
    SrcMutant(
        "elect-cls-effects",
        "crates/sim/src/elect_lanes.rs",
        "                all = meet(&all, &restrict(&p, e.known));",
        "",
        [],
        [("ptxv-sim", "fixtures", "e_elect_lane_classes")],
    ),
    # E-HW-SPLIT-UR (urwin.rs, GB300 profile): no uniform-register windows, the splits switch anywhere
    # again (MISS 52: a false deadlock on probes/warpsync_pairs under the cross-check).
    SrcMutant(
        "split-ur-off",
        "crates/sim/src/urwin.rs",
        "    if any { inside } else { Vec::new() }",
        "    let _ = (any, inside);\n    Vec::new()",
        [],
        [
            (
                "ptxv-sim",
                "wellsync",
                "cert_crosscheck_warpsync_pairs_certifies_under_gb300",
            )
        ],
    ),
    # S-PROXY-WAR (exec.rs `proxy_wcheck`): the strict level stops checking a generic read before
    # an async-proxy write (the exemption before the placement change): WAR is clean, not contract.
    SrcMutant(
        "proxy-war-off",
        "crates/sim/src/exec.rs",
        "        self.proxy_war_check(cta, off, n, vc, pc, what);\n",
        "",
        [("e_mbar_lww", "WAR", "ptx", []), ("e_mbar_lww", "WAR", "sass", [])],
    ),
    # E-HW-PROXY-WAR (hwprofile.rs): the GB300 profile no longer discharges proxy-war, so the
    # level-2 verdict of WAR stays a contract finding.
    SrcMutant(
        "proxy-war-nodischarge",
        "crates/sim/src/hwprofile.rs",
        '        discharges: &["proxy-war:"],',
        "        discharges: &[],",
        [],
        [("ptxv-sim", "fixtures", "e_mbar_lww_proxy_war_contract")],
    ),
    # S-MBAR-SCOPE (exec.rs `tmem_block`): MMA / tcgen05.cp TMEM accesses ignore their issuer's
    # SM-local happens-before again: the SASS leader's MMA after a PHASECHK with no CCTL.IVALL
    # races with the peer's handed-back tcgen05.ld (e_tc_handback ACTA: clean on SASS).
    SrcMutant(
        "tmem-block-nolocal",
        "crates/sim/src/exec.rs",
        "        // (an MMA or tcgen05.cp: its issuer's at issue, `AsyncOp::sh`), as in `tmem_access`.\n        let aug;\n        let vc = match self.local_vc(actor, vc, None) {",
        "        // (an MMA or tcgen05.cp: its issuer's at issue, `AsyncOp::sh`), as in `tmem_access`.\n        let aug;\n        let vc = match None::<VC> {",
        [("e_tc_handback", "ACTA", "sass", [])],
    ),
    # MISS 111: the dealloc takes the thread's whole tcgen05 view again, un-waited tcgen05.ld /
    # st included (e_dealloc_unwaited LDNW on PTX: no race between lane 0's own load and its
    # dealloc; the other lanes' still race, MISS 105).
    SrcMutant(
        "dealloc-unwaited",
        "crates/sim/src/exec.rs",
        "                let mut vc = self.threads[t].vc.clone();\n                let tc = self.tc_released(t);",
        "                let mut vc = self.threads[t].vc.clone();\n                let tc = self.threads[t].tc_vc.clone();",
        [],
        [("ptxv-sim", "fixtures", "e_dealloc_unwaited_races")],
    ),
    # MISS 116: the PTX front orders a thread's tcgen05.st after its own un-waited tcgen05.ld /
    # tcgen05.st by their shared actor again (e_tmem_queue Q, QQ on PTX: clean).
    SrcMutant(
        "tmem-queue-ptx-off",
        "crates/sim/src/exec.rs",
        "            Some(_) if self.k.front == Front::Ptx && (n..2 * n).contains(&(actor as usize)) => {",
        "            Some(_) if false && (n..2 * n).contains(&(actor as usize)) => {",
        [("e_tmem_queue", "Q", "ptx", []), ("e_tmem_queue", "QQ", "ptx", [])],
    ),
    # MISS 121: the PTX front orders a thread's tcgen05.ld after its own un-waited tcgen05.st
    # by their shared actor again (e_tmem_queue SL on PTX: contract only, no race).
    SrcMutant(
        "tmem-st-ld-ptx-off",
        "crates/sim/src/exec.rs",
        "            None if self.k.front == Front::Ptx && (n..2 * n).contains(&(actor as usize)) => {",
        "            None if false && (n..2 * n).contains(&(actor as usize)) => {",
        [("e_tmem_queue", "SL", "ptx", [])],
    ),
    # E-TMEM-QUEUE: a write examines only its first unordered access again; the store's own
    # pending LDTM, sorted first, makes the finding a tmem-queue contract and hides the other
    # warp's racing LDTM (e_queue_mask S0 on SASS: contract, no race).
    SrcMutant(
        "tmem-queue-first-only",
        "crates/sim/src/exec.rs",
        "        for &(o, kind) in &offenders {",
        "        for &(o, kind) in offenders.iter().take(1) {",
        [("e_queue_mask", "S0", "sass", [])],
    ),
    # MISS 122: tmem_access reports only its first racer again; tc_fence_contract turns that
    # one (ordered once the fences are implied) into a contract and the second racer is lost
    # (e_rerun_drop_ptx XLOW on PTX: contract only).
    SrcMutant(
        "tmem-first-racer-only",
        "crates/sim/src/exec.rs",
        "            if !queued(&o) && !racers.iter().any(|x| x.0.pc == o.pc && x.1 == kind) {",
        "            if !queued(&o) && racers.is_empty() {",
        [("e_rerun_drop_ptx", "XLOW", "ptx", [])],
    ),
    # MISS 124: proxy_check_dir reports only its first uncovered generic writer again;
    # mbar_async_contract reclassifies that one and the second racing writer is lost
    # (e_rerun_drop_proxy_mb XLOW on PTX: contract only).
    SrcMutant(
        "proxy-first-writer-only",
        "crates/sim/src/exec.rs",
        "                if gen_seen.contains(&(w.actor, w.clock)) || gen_reported.contains(&w.pc) {",
        "                if gen_seen.contains(&(w.actor, w.clock)) || !gen_reported.is_empty() {",
        [("e_rerun_drop_proxy_mb", "XLOW", "ptx", [])],
    ),
    # MISS 136: DPOR no longer backtracks a process whose async op the taken transition
    # consumes (a drain inside a thread step); e_dpor_drain on SASS: DPOR [race] against the
    # stateful [nondeterministic, race, violation] (the test runs both explorers).
    SrcMutant(
        "dpor-consumed-nobacktrack",
        "crates/sim/src/dpor.rs",
        "        frames[n].backtrack.extend(disables);",
        "        drop(disables);",
        [],
        [("ptxv-sim", "fixtures", "dpor_backtracks_a_consumed_async_op")],
    ),
    # MISS 146: a trap that exits every thread no longer backtracks the splits it kills;
    # e_dpor_trap: DPOR [trap] against the stateful [fault, trap] (the test runs both).
    SrcMutant(
        "dpor-killed-nobacktrack",
        "crates/sim/src/dpor.rs",
        "                            .all(|&t| a.succ.threads[t as usize].st == St::Exited)",
        "                            .all(|_| false)",
        [],
        [("ptxv-sim", "fixtures", "dpor_backtracks_a_disabled_process")],
    ),
    # MISS 146 / MISS 136 follow-up: only consumed ops that are an alternative's head disable
    # their process again; e_dpor_nonhead NH consumes B behind a pending head, and the test
    # requires that backtrack (dpor::NONHEAD_BACKTRACKS).
    SrcMutant(
        "dpor-nonhead-nobacktrack",
        "crates/sim/src/dpor.rs",
        "            if Some(id) != a.op && !live.contains(&id) {",
        "            if Some(id) != a.op && !live.contains(&id) && alts.iter().any(|b| b.op == Some(id)) {",
        [],
        [("ptxv-sim", "fixtures", "dpor_backtracks_a_disabled_process")],
    ),
    # MISS 149: the SASS lockstep join no longer follows FENCE.VIEW.ASYNC.T; the softmax warps'
    # STTM writes (complete at the thread's own fence since MISS 107) do not reach the elected
    # lane's arrive and race with the PV UTCHMMA: c_fa4_fwd's certificate is refused.
    SrcMutant(
        "lockstep-nowaitst",
        "crates/sim/src/exec.rs",
        "                        | Op::TmemSt { .. }\n                        | Op::TmemWaitSt\n",
        "                        | Op::TmemSt { .. }\n",
        [],
        [("ptxv-sim", "fixtures", "c_fa4_fwd_certifies")],
    ),
    # E-HW-SPLIT-UR at every level: the strict level drops the switches too (MISS 52's
    # refutation at the strict level must stand: a measured null result is a bound).
    SrcMutant(
        "split-ur-ungated",
        "crates/sim/src/exec.rs",
        "        if self.launch.hw_split_ur\n"
        "            && !self.ur_window.is_empty()\n"
        "            && self.ur_window_blocks(members, pc)",
        "        if !self.ur_window.is_empty() && self.ur_window_blocks(members, pc)",
        [],
        [("ptxv-sim", "wellsync", "cert_crosscheck_deadlocking_schedule_refutes")],
    ),
    # E-HW-SPLIT-UR: a stalling instruction (STS) ends a window, so the window after
    # `VOTEU.ANY UR5` misses the LOP3 that reads it.
    SrcMutant(
        "split-ur-sts",
        "crates/sim/src/urwin.rs",
        '    t.starts_with("YIELD")',
        '    t.starts_with("STS")\n        || t.starts_with("YIELD")',
        [],
        [
            (
                "ptxv-sim",
                "wellsync",
                "cert_crosscheck_warpsync_pairs_certifies_under_gb300",
            )
        ],
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
        "            if let Some(w) = sh.w {\n                if w.actor != actor && !w.hb(vc) {\n                    offenders.push((",
        "            if let Some(w) = sh.w {\n                if false && w.actor != actor && !w.hb(vc) {\n                    offenders.push((",
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
    # E-BAR-OVER walk cache (explore.rs `thread_may_arrive_v`, `WalkTrie::find`, `walk_same`,
    # `Machine::walk_id`): one key component or check dropped at a time. A cached walk then
    # answers for a thread, machine or query it was not made for.
    SrcMutant(
        "walkkey-pc",
        "crates/sim/src/explore.rs",
        "        let key = (self.walk_id(), self.threads[t].pc, bit, cut, values);",
        "        let key = (self.walk_id(), 0, bit, cut, values);",
        [],
        tests=[LIB],
    ),
    SrcMutant(
        "walkkey-bit",
        "crates/sim/src/explore.rs",
        "        let key = (self.walk_id(), self.threads[t].pc, bit, cut, values);",
        "        let key = (self.walk_id(), self.threads[t].pc, 0, cut, values);",
        [],
        tests=[LIB],
    ),
    SrcMutant(
        "walkkey-cut",
        "crates/sim/src/explore.rs",
        "        let key = (self.walk_id(), self.threads[t].pc, bit, cut, values);",
        "        let key = (self.walk_id(), self.threads[t].pc, bit, 0, values);",
        [],
        tests=[CUT, ("ptxv-sim", "lib", "explore::bar_cut_tests::bar_cut_applies_only_when_its_premises_hold")],
    ),
    SrcMutant(
        "walkkey-values",
        "crates/sim/src/explore.rs",
        "        let key = (self.walk_id(), self.threads[t].pc, bit, cut, values);",
        "        let key = (self.walk_id(), self.threads[t].pc, bit, cut, true);",
        [],
        tests=[LIB],
    ),
    SrcMutant(
        "walk-noreread",
        "crates/sim/src/explore.rs",
        "                    let v = m.bar_read_c(t, k, wc)?;\n                    let i = kids.binary_search_by_key(&v, |e| e.0).ok()?;",
        "                    let (v, i) = (kids[0].0, 0usize);",
        [("e_barwalk", v, f, []) for v in ("good", "BW1") for f in ("ptx", "sass")],
        tests=[LIB, ("ptxv-sim", "lib", "explore::walk_cache_tests::walk_cache_keys_register_barrier_ids")],
    ),
    SrcMutant(
        "walkid-ops",
        "crates/sim/src/explore.rs",
        "            .all(|(x, y)| x.guard == y.guard && x.op == y.op)",
        "            .all(|(x, y)| x.guard == y.guard)",
        [],
        tests=[LIB],
    ),
    SrcMutant(
        "walkid-guards",
        "crates/sim/src/explore.rs",
        "            .all(|(x, y)| x.guard == y.guard && x.op == y.op)",
        "            .all(|(x, y)| x.op == y.op)",
        [],
        tests=[LIB],
    ),
    SrcMutant(
        "walkid-barcfg",
        "crates/sim/src/explore.rs",
        "    a.0 == b.0\n        && a.1.front == b.1.front",
        "    true\n        && a.1.front == b.1.front",
        [],
        tests=[LIB],
    ),
    SrcMutant(
        "walkid-memo",
        "crates/sim/src/explore.rs",
        "                .find(|e| std::ptr::eq(e.0.as_ptr(), c) && std::ptr::eq(e.1.as_ptr(), k))",
        "                .find(|e| std::ptr::eq(e.0.as_ptr(), c))",
        [],
        tests=[LIB],
    ),
    # E-BAR-OVER walks (explore.rs `thread_may_arrive_walk*`): a value the walk reads left out
    # of its reads (a cached walk then answers for a thread that reads another value there),
    # or the register-id walk's state set kept as 4-bit hashes (a new state taken as seen).
    SrcMutant(
        "walk-norenv",
        "crates/sim/src/explore.rs",
        (
            "                                if let Some((_, src)) = renv.iter().find(|e| e.0 == *r) {\n"
            "                                    let k = BarKey::Val(src.clone());\n"
            "                                    let x = self.bar_read(t, &k);\n"
            "                                    reads.push((k, x.unwrap_or(0)));\n"
            "                                    return x;\n"
            "                                }"
        ),
        (
            "                                if let Some((_, src)) = renv.iter().find(|e| e.0 == *r) {\n"
            "                                    let k = BarKey::Val(src.clone());\n"
            "                                    let x = self.bar_read(t, &k);\n"
            "                                    return x;\n"
            "                                }"
        ),
        [],
        tests=[("ptxv-sim", "lib", WCT + "walk_cache_keys_a_tracked_registers_source")],
    ),
    SrcMutant(
        "walk-nosetpread",
        "crates/sim/src/explore.rs",
        (
            "                        let k = BarKey::Val(v.clone());\n"
            "                        let x = self.bar_read(t, &k);\n"
            "                        // (A failed read is recorded as 0: a thread reading 0 there reuses a\n"
            "                        // walk that knew less, which is conservative.)\n"
            "                        reads.push((k, x.unwrap_or(0)));\n"
            "                        x"
        ),
        (
            "                        let k = BarKey::Val(v.clone());\n"
            "                        let x = self.bar_read(t, &k);\n"
            "                        let _ = (&k, &reads);\n"
            "                        x"
        ),
        [],
        tests=[("ptxv-sim", "lib", WCT + "walk_cache_keys_a_setp_operand")],
    ),
    SrcMutant(
        "walk-noconv",
        "crates/sim/src/explore.rs",
        "            let c = self.bar_read(t, &BarKey::Conv).unwrap_or(0);\n            reads.push((BarKey::Conv, c));",
        "            let c = self.bar_read(t, &BarKey::Conv).unwrap_or(0);\n            let _ = c;",
        [],
        tests=[("ptxv-sim", "lib", WCT + "walk_cache_keys_the_warps_convergence")],
    ),
    SrcMutant(
        "walkseen-hash4",
        "crates/sim/src/explore.rs",
        "            let mut seen: HashSet<(usize, Env, REnv), BuildHasherDefault<WalkHasher>> =\n                HashSet::default();",
        (
            "            #[derive(Default)]\n"
            "            struct HashOnly(HashSet<u64>);\n"
            "            impl HashOnly {\n"
            "                fn h<T: std::hash::Hash>(x: &T) -> u64 {\n"
            "                    let mut s = WalkHasher::default();\n"
            "                    x.hash(&mut s);\n"
            "                    std::hash::Hasher::finish(&s) >> 60\n"
            "                }\n"
            "                fn contains<T: std::hash::Hash>(&self, x: &T) -> bool {\n"
            "                    self.0.contains(&Self::h(x))\n"
            "                }\n"
            "                fn insert<T: std::hash::Hash>(&mut self, x: T) -> bool {\n"
            "                    self.0.insert(Self::h(&x))\n"
            "                }\n"
            "            }\n"
            "            let mut seen = HashOnly::default();"
        ),
        [],
        tests=[("ptxv-sim", "lib", WCT + "register_id_walk_visits_every_state")],
    ),
    # E-BAR-OVER walks: a predicate read of the register-id walk, or R2UR's warp-uniformity
    # read of the plain walk, left out of the walk's reads; a walk's state set keyed by the pc
    # alone (a second environment at a seen pc taken as seen).
    SrcMutant(
        "walk-regs-nopredread",
        "crates/sim/src/explore.rs",
        (
            "                        if unknown.contains(&r) || !ucur(&r) {\n"
            "                            return None;\n"
            "                        }\n"
            "                        used.push(r);\n"
            "                        let k = BarKey::Pred(r);\n"
            "                        let v = self.bar_read(t, &k);\n"
            "                        reads.push((k, v.unwrap_or(0)));"
        ),
        (
            "                        if unknown.contains(&r) || !ucur(&r) {\n"
            "                            return None;\n"
            "                        }\n"
            "                        used.push(r);\n"
            "                        let k = BarKey::Pred(r);\n"
            "                        let v = self.bar_read(t, &k);\n"
            "                        let _ = (&k, &reads);"
        ),
        [],
        tests=[("ptxv-sim", "lib", WCT + "walk_cache_keys_a_register_id_walks_predicate")],
    ),
    SrcMutant(
        "walk-nouniread",
        "crates/sim/src/explore.rs",
        (
            "                                    let k = BarKey::Uni(a.clone());\n"
            "                                    let u = self.bar_read(t, &k).unwrap_or(0);\n"
            "                                    reads.push((k, u));"
        ),
        (
            "                                    let k = BarKey::Uni(a.clone());\n"
            "                                    let u = self.bar_read(t, &k).unwrap_or(0);\n"
            "                                    let _ = &k;"
        ),
        [],
        tests=[("ptxv-sim", "lib", WCT + "walk_cache_keys_an_r2urs_uniformity")],
    ),
    SrcMutant(
        "walkseen-pconly",
        "crates/sim/src/explore.rs",
        (
            "            let mut seen: HashSet<(usize, Env, REnv), BuildHasherDefault<WalkHasher>> =\n"
            "                HashSet::default();"
        ),
        (
            "            #[derive(Default)]\n"
            "            struct PcOnly(HashSet<usize>);\n"
            "            #[allow(dead_code)]\n"
            "            impl PcOnly {\n"
            "                fn contains(&self, x: &(usize, Env, REnv)) -> bool {\n"
            "                    self.0.contains(&x.0)\n"
            "                }\n"
            "                fn insert(&mut self, x: (usize, Env, REnv)) -> bool {\n"
            "                    self.0.insert(x.0)\n"
            "                }\n"
            "                fn len(&self) -> usize {\n"
            "                    self.0.len()\n"
            "                }\n"
            "            }\n"
            "            let mut seen = PcOnly::default();"
        ),
        [],
        tests=[("ptxv-sim", "lib", WCT + "register_id_walk_visits_every_state")],
    ),
    SrcMutant(
        "plainseen-pconly",
        "crates/sim/src/explore.rs",
        (
            "            let mut seen: HashSet<(usize, Env, VEnv), BuildHasherDefault<WalkHasher>> =\n"
            "                HashSet::default();"
        ),
        (
            "            #[derive(Default)]\n"
            "            struct PcOnly(HashSet<usize>);\n"
            "            #[allow(dead_code)]\n"
            "            impl PcOnly {\n"
            "                fn contains(&self, x: &(usize, Env, VEnv)) -> bool {\n"
            "                    self.0.contains(&x.0)\n"
            "                }\n"
            "                fn insert(&mut self, x: (usize, Env, VEnv)) -> bool {\n"
            "                    self.0.insert(x.0)\n"
            "                }\n"
            "                fn len(&self) -> usize {\n"
            "                    self.0.len()\n"
            "                }\n"
            "            }\n"
            "            let mut seen = PcOnly::default();"
        ),
        [],
        tests=[("ptxv-sim", "lib", WCT + "plain_walk_visits_every_state")],
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
    # E-ALIGNED (SASS): guarded-off lanes at the warp's BAR are again a divergent-barrier
    # violation and do not count as arrived.
    SrcMutant(
        "aligned-sass-guarded",
        "crates/sim/src/exec.rs",
        "let sass_guarded = k.front == Front::Sass && elsewhere.is_empty();",
        "let sass_guarded = false;",
        [("e_aligned_exit", "good", "sass", [])],
        [("ptxv-sim", ADV, "e_aligned_exit")],
    ),
    # E-ALIGNED (PTX contract): an aligned barrier after part of the warp exited is no
    # violation (MISS 37).
    SrcMutant(
        "aligned-ptx-exit",
        "crates/sim/src/exec.rs",
        "            if k.front == Front::Ptx {\n                let gone: Vec<usize> = self",
        "            if false {\n                let gone: Vec<usize> = self",
        [("e_aligned_exit", "good", "ptx", []), ("e_aligned_exit2", "good", "ptx", [])],
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

# S-NC (MISS 21): a non-coherent load (ld.global.nc, LDG .CONSTANT) of bytes written while its
# kernel runs is a violation. Each mutant weakens one part; the e_ld_nc* fixtures catch it.
NC_TEST = ("ptxv-sim", "fixtures", "e_ld_nc_written_during_kernel")
NC_FORMS = ("ptxv-sim", "fixtures", "e_ld_nc_every_form")
NC_PDL = ("ptxv-sim", "fixtures", "e_ld_nc_pdl_overlapping_launches")
NC_XGPU = ("ptxv-sim", "fixtures", "e_ld_nc_xgpu_write_before_start")
NC_PROFILE = ("ptxv-sim", "fixtures", "e_ld_nc_xgpu_no_exemption_at_sass_gb300")
MUTANTS += [
    # .nc reads are coherent weak loads again (the miss: exec.rs never marks them).
    SrcMutant(
        "nc-coherent",
        "crates/sim/src/exec.rs",
        "        let nc_read = write.is_none()\n            && strong == 0",
        "        let nc_read = false\n            && strong == 0",
        [("e_ld_nc", v, f, []) for v in ("written", "inplace") for f in ("ptx", "sass")],
        [NC_TEST],
    ),
    # A write after the .nc load of its bytes is not checked (e_ld_nc inplace: the loading
    # thread overwrites the word; e_ld_nc_forms RED: an atomic after the barrier).
    SrcMutant(
        "nc-after-skip",
        "crates/sim/src/exec.rs",
        "                            if let Some((g, gated)) = nc_hit_mask(mask) {",
        "                            if let Some((g, gated)) = nc_hit_mask(mask).filter(|_| false) {",
        [("e_ld_nc", "inplace", f, []) for f in ("ptx", "sass")]
        + [("e_ld_nc_forms", "RED", f, []) for f in ("ptx", "sass")],
        [NC_TEST],
    ),
    # An .nc load of bytes written before it (ordered by bar.sync) is not checked.
    SrcMutant(
        "nc-before-skip",
        "crates/sim/src/exec.rs",
        "                            !w.sv && nc_conflict(my_grid, grid_of(w.actor)) && !before_start(w)",
        "                            false && !w.sv && nc_conflict(my_grid, grid_of(w.actor)) && !before_start(w)",
        [("e_ld_nc", "written", f, []) for f in ("ptx", "sass")],
        [NC_TEST],
    ),
    # SASS: LDG .CONSTANT lifted as a coherent load.
    SrcMutant(
        "nc-sass-ldg",
        "crates/sass/src/lib.rs",
        '                // `.CONSTANT`: ld.global.nc (S-NC).\n                nc: has("CONSTANT"),',
        '                // `.CONSTANT`: ld.global.nc (S-NC).\n                nc: false,',
        [("e_ld_nc", "written", "sass", []), ("e_ld_nc_forms", "BYTE", "sass", []),
         ("e_ld_nc_forms", "V4", "sass", [])],
        [NC_TEST, NC_FORMS],
    ),
    # SASS: the 32-byte LDG.E.ENL2.256.CONSTANT lifted as a coherent load.
    SrcMutant(
        "nc-sass-256",
        "crates/sass/src/lib.rs",
        '                mask,\n                nc: has("CONSTANT"),',
        '                mask,\n                nc: false,',
        [("e_ld_nc_forms", "V8", "sass", [])],
        [NC_FORMS],
    ),
    # PTX: `.nc` dropped on a scalar ld (the original MISS 21 state of the PTX front).
    SrcMutant(
        "nc-ptx-drop",
        "crates/ptx/src/lib.rs",
        "bits < 32,\n                    nc: has(\"nc\"),",
        "bits < 32,\n                    nc: false,",
        [("e_ld_nc", v, "ptx", []) for v in ("written", "inplace")],
        [NC_TEST],
    ),
    # PTX: `.nc` dropped on ld.global.nc.v2 / .v4 / .v8.
    SrcMutant(
        "nc-ptx-vec-drop",
        "crates/ptx/src/lib.rs",
        "bits < 32,\n                            nc: has(\"nc\"),",
        "bits < 32,\n                            nc: false,",
        [("e_ld_nc_forms", v, "ptx", []) for v in ("V4", "V8")],
        [NC_FORMS],
    ),
    # A PDL dependent counts as stream-ordered after its primary again: its .nc read of the
    # primary's output after griddepcontrol.wait is clean (probes/nc_pdl: GB300 old word 200/200).
    # (machine.rs `Launch::completes_before`, which S-NC's `before` calls since MISS 113.)
    SrcMutant(
        "nc-pdl-ordered",
        "crates/sim/src/machine.rs",
        "            if self.full_preds(x).any(|p| p == j) {",
        "            if self.prev_grid(x) == Some(j) || self.full_preds(x).any(|p| p == j) {",
        [("e_ld_nc_pdl", v, f, []) for v in ("PDLNC", "EARLY") for f in ("ptx", "sass")],
        [NC_PDL],
    ),
    # Precision: a write after the completion of the grid that read its bytes with .nc is
    # reported again (e_ld_nc_pdl good: a PDL dependent overwrites its primary's .nc input after
    # griddepcontrol.wait).
    SrcMutant(
        "nc-done-off",
        "crates/sim/src/exec.rs",
        "                mask >> g & 1 == 1 && nc_conflict(g, my_grid) && !(g < 63 && done_before(g))",
        "                mask >> g & 1 == 1 && nc_conflict(g, my_grid)",
        [("e_ld_nc_pdl", "good", f, []) for f in ("ptx", "sass")],
        [NC_PDL],
    ),
    # Precision: a write that happens before the reading grid started (its launch clock, the
    # done clocks of its full predecessors) is reported again (e_ld_nc_xgpu good: a peer GPU's
    # writes, then a release / acquire hand-off to the reader's stream predecessor; PTX front,
    # the SASS front's strict level does not apply the exemption).
    SrcMutant(
        "nc-start-off",
        "crates/sim/src/exec.rs",
        "        let before_start = |w: &Epoch| launch.order_recorded && started_after(w);",
        "        let before_start = |w: &Epoch| false && launch.order_recorded && started_after(w);",
        [("e_ld_nc_xgpu", v, "ptx", []) for v in ("good", "SIGFENCE", "ACQFENCE", "ACQONLYSYS")]
        + [("e_ld_nc_xgpu_frel", "SYSF", "ptx", [])],
        [NC_XGPU],
    ),
    # The launch-clock exemption at the SASS front's strict level too: its CCTL.IVALL acquire has
    # no scope (fence.acquire.gpu and fence.acquire.sys compile to the same bare CCTL.IVALL), so a
    # .gpu acquire fence of a peer GPU's release exempts the peer's writes (e_ld_nc_xgpu
    # ACQONLYGPU and ACQFENCEGPU on SASS: clean, the violation lost).
    SrcMutant(
        "nc-start-sass",
        "crates/sim/src/exec.rs",
        "        let exempt = hw_order || kern.front == Front::Ptx;",
        "        let exempt = hw_order || true;",
        [("e_ld_nc_xgpu", v, "sass", []) for v in ("good", "ACQFENCEGPU", "ACQONLYGPU")],
        [NC_XGPU],
    ),
    # The launch-clock exemption in the SASS front's GB300 profile re-run too (its E-HW-L1
    # switch on, no hardware ordering): e_ld_nc_xgpu good, SIGFENCE, ACQFENCE and ACQONLYSYS
    # lose their violation at sass_gb300 only, which the strict-level checks do not see.
    SrcMutant(
        "nc-start-profile",
        "crates/sim/src/exec.rs",
        "        let exempt = hw_order || kern.front == Front::Ptx;",
        "        let exempt = hw_order || launch.hw_l1 || kern.front == Front::Ptx;",
        [],
        [NC_PROFILE],
    ),
    # The read's own clock in place of its grid's launch clock: a write ordered before the read
    # by a bar.sync, an acquire or a griddepcontrol.wait is exempt, which none of them is (the
    # load moves above them: probes/nc_hoist, nc_pdl hoist).
    SrcMutant(
        "nc-start-readclock",
        "crates/sim/src/exec.rs",
        "        let before_start = |w: &Epoch| launch.order_recorded && started_after(w);",
        "        let _ = &started_after; let before_start = |w: &Epoch| w.clock <= vc_get(vc, w.actor);",
        [("e_ld_nc", "written", f, []) for f in ("ptx", "sass")]
        + [("e_ld_nc_pdl", "PDLNC", f, []) for f in ("ptx", "sass")],
        [NC_TEST, NC_PDL],
    ),
    # E-LAUNCH-ORDER: the launch-clock exemption for a list whose launch order is unknown (an
    # older capture's list may serialize grids that ran at the same time).
    SrcMutant(
        "nc-start-gate",
        "crates/sim/src/exec.rs",
        "        let before_start = |w: &Epoch| launch.order_recorded && started_after(w);",
        "        let before_start = |w: &Epoch| started_after(w);",
        [("e_ld_nc_xgpu_unord", "good", "ptx", [])],
        [("ptxv-sim", "fixtures", "e_ld_nc_xgpu_unord_launch_order")],
    ),
    # The launch clock from the grid before it only, without the grids it `waits` for (a join).
    SrcMutant(
        "nc-start-waits",
        "crates/sim/src/exec.rs",
        "                && launch.full_preds(my_grid).any(|p| {",
        "                && launch.prev_grid(my_grid).filter(|_| !launch.grids[my_grid].pdl).into_iter().any(|p| {",
        [("e_ld_nc_xgpu", "WAITS", "ptx", [])],
        [NC_XGPU],
    ),
]

# E-LAUNCH-ORDER (MISS 113): launch-order exclusions apply only to a recorded or declared launch
# order (case `launch_order`); an older capture's list may serialize grids of different streams
# that ran at the same time. `waits` (event waits, graph joins) order like a stream predecessor;
# `mem_sync_domain` narrows .gpu scope to one domain.
LO_UNKNOWN = ("ptxv-sim", "fixtures", "e_ld_nc_pdl_unknown_order")
LO_STREAMS = ("ptxv-sim", "fixtures", "e_launch_order_two_streams")
LO_WAITS = ("ptxv-sim", "fixtures", "e_launch_waits_join")
LO_L1 = ("ptxv-sim", "fixtures", "e_launch_order_l1_boundary")
MUTANTS += [
    # S-NC's stream-order exclusion applies to an unknown-order list again.
    SrcMutant(
        "lo-nc-gate",
        "crates/sim/src/machine.rs",
        "        self.order_recorded && self.completes_before(j, i)",
        "        self.completes_before(j, i)",
        [("e_launch_order", "UNKNOWN", f, []) for f in ("ptx", "sass")]
        + [("e_launch_waits", "UNKNOWN", f, []) for f in ("ptx", "sass")],
        [LO_STREAMS, LO_WAITS],
    ),
    # S-NC's done-clock exemption applies to an unknown-order list again.
    SrcMutant(
        "lo-done-gate",
        "crates/sim/src/exec.rs",
        "        let done_before = |g: usize| launch.order_recorded && done_by(g);",
        "        let done_before = |g: usize| done_by(g);",
        [("e_ld_nc_pdl", "UNKNOWN_DONE", f, []) for f in ("ptx", "sass")],
        [LO_UNKNOWN],
    ),
    # E-L1-CORES / E-L1-HAVOC: the kernel boundary of an unknown-order list counts again
    # (`grids_ordered`, then the done-clock check).
    SrcMutant(
        "lo-l1-gate",
        "crates/sim/src/l1_cores.rs",
        "    l.order_recorded && l.hb_before(a, b)",
        "    l.hb_before(a, b)",
        [("e_launch_order_l1", "UNKNOWN", f, []) for f in ("ptx", "sass")],
        [LO_L1],
    ),
    SrcMutant(
        "lo-l1-done-gate",
        "crates/sim/src/l1_cores.rs",
        "            if self.launch.order_recorded\n                && self.launch.full_preds(rg).any(|p| {",
        "            if self.launch.full_preds(rg).any(|p| {",
        [("e_launch_order_l1", "UNKNOWN", f, []) for f in ("ptx", "sass")],
        [LO_L1],
    ),
    # A wait (`waits`) no longer joins the waited grid's done clocks at launch: kc's read of
    # kb's output races.
    SrcMutant(
        "lo-waits-hb",
        "crates/sim/src/exec.rs",
        "            match &mut done {\n                Some(v) => vc_join(v, &d),\n                None => done = Some(d),\n            }",
        "            let _ = d;",
        [("e_launch_waits", "good", f, []) for f in ("ptx", "sass")],
        [LO_WAITS],
    ),
    # A wait no longer counts as completing before the grid starts (S-NC).
    SrcMutant(
        "lo-waits-nc",
        "crates/sim/src/machine.rs",
        "        p.into_iter().chain(self.grids[gi].waits.iter().copied())",
        "        p.into_iter().chain(self.grids[gi].waits.iter().copied().filter(|_| false))",
        [("e_launch_waits", "good", f, []) for f in ("ptx", "sass")],
        [LO_WAITS],
    ),
    # A wait no longer holds the grid's launch back (only the join of done clocks stays): kc
    # launches before kb completes, sets the flag kb spins on, and the deadlock is gone.
    SrcMutant(
        "lo-waits-gate",
        "crates/sim/src/exec.rs",
        "            .waits\n            .iter()\n            .all(|&w| self.grid_complete(w))",
        "            .waits\n            .iter()\n            .all(|_| true)",
        [("e_launch_waits_gate", "good", f, []) for f in ("ptx", "sass")],
        [("ptxv-sim", "fixtures", "e_launch_waits_gate")],
    ),
    # A recorded launch order accepts a stream switch without `after` again.
    SrcMutant(
        "lo-stream-check",
        "crates/sim/src/lib.rs",
        "                if a != b {\n                    bail!(\n                        \"launch {gi} (stream {b})",
        "                if false && a != b {\n                    bail!(\n                        \"launch {gi} (stream {b})",
        [],
        [("ptxv-sim", "fixtures", "e_launch_order_stream_switch_needs_after")],
    ),
    # .gpu scope includes the threads of another memory synchronization domain again.
    SrcMutant(
        "lo-mem-domain",
        "crates/sim/src/exec.rs",
        "                gx.device == gy.device && gx.mem_domain == gy.mem_domain",
        "                gx.device == gy.device",
        [("e_mem_sync_domain", "good", f, []) for f in ("ptx", "sass")],
        [("ptxv-sim", "fixtures", "e_mem_sync_domain_scope")],
    ),
]

# S-SAME-VALUE per byte across a uniform shadow page (MISS 69): exec.rs `access_x` splits a
# uniform page before `for_run` when a write keeps the value of only some of its bytes. Without
# the split the page's first byte stands for the page again: s_sv_page PAGE and CONC (a reload
# that keeps only the first byte, then reads of the tile) are clean.
SV_PAGE = ("ptxv-sim", "fixtures", "s_sv_page_split_by_byte")
MUTANTS += [
    SrcMutant(
        "sv-page-onebyte",
        "crates/sim/src/exec.rs",
        "                    if wundef != 0 || (kept != 0 && kept != per) {",
        "                    if false && (wundef != 0 || (kept != 0 && kept != per)) {",
        [("s_sv_page", v, f, []) for v in ("PAGE", "CONC") for f in ("ptx", "sass")],
        [SV_PAGE],
    ),
]

# S-SCOPE, complete overlap (MISS 80; PTX ISA 9.5 section 8.7): exec.rs `same_location` decides
# whether two strong thread accesses of the PTX front are one location (scalar element sizes,
# `elem_bytes`). scope-overlap: every pair is one location again (scopes only): a u8 / u64
# acquire of a u32 release flag is clean (e_scope_mixed MIXED, WIDE) and the warp's u8 load of
# a u32 store nondeterministic (e_warp_order_mixed MIXED). scope-overlap-race: the race
# filter's moral strength ignores overlap (`morally_strong`), the acquire still does not
# synchronize. scope-overlap-sync: a partially overlapping acquire synchronizes again
# (`sync_observe_cell`): the flag race remains, the data race goes (test only).
# scope-elem-vector: a vector load is one location of its whole width (a precision mutant:
# VEC races).
SCOPE_OVERLAP = ("ptxv-sim", "fixtures", "scope_complete_overlap")
MUTANTS += [
    SrcMutant(
        "scope-overlap",
        "crates/sim/src/exec.rs",
        "    x == 0 || y == 0 || x == y\n",
        "    x == 0 || y == 0 || x == y || true\n",
        [("e_scope_mixed", v, "ptx", []) for v in ("MIXED", "WIDE")]
        + [("e_warp_order_mixed", "MIXED", "ptx", [])],
        [SCOPE_OVERLAP],
    ),
    SrcMutant(
        "scope-overlap-race",
        "crates/sim/src/exec.rs",
        "        self.overlap_complete(a, pa, b, pb) && self.scopes_mutual(a, sa, b, sb)",
        "        self.scopes_mutual(a, sa, b, sb)",
        [("e_warp_order_mixed", "MIXED", "ptx", [])],
        [SCOPE_OVERLAP],
    ),
    SrcMutant(
        "scope-overlap-sync",
        "crates/sim/src/exec.rs",
        "        if !self.overlap_complete(c.writer, c.pc, me, pc) {",
        "        if false && !self.overlap_complete(c.writer, c.pc, me, pc) {",
        [],
        [SCOPE_OVERLAP],
    ),
    SrcMutant(
        "scope-elem-vector",
        "crates/sim/src/exec.rs",
        "        Some(Op::Ld { bytes, n, .. }) => bytes / (*n).max(1),",
        "        Some(Op::Ld { bytes, n, .. }) => bytes / (*n).clamp(1, 1),",
        [("e_scope_mixed", "VEC", "ptx", [])],
        [SCOPE_OVERLAP],
    ),
    # Precision: a packed multimem.st is one location of its register width again, not one per
    # lane (e_scope_mm_lane good: a mixed-size race against an f16x2 multimem.ld_reduce).
    SrcMutant(
        "scope-elem-mmst",
        "crates/sim/src/exec.rs",
        "        Some(Op::St { bytes, v, .. }) => multimem_st_lane(&k.code[pc as usize].text)\n",
        "        Some(Op::St { bytes, v, .. }) => multimem_st_lane(&k.code[pc as usize].text)\n            .filter(|_| false)\n",
        [("e_scope_mm_lane", "good", "ptx", [])],
        [("ptxv-sim", "fixtures", "scope_multimem_lane")],
    ),
]

# S-SCOPE release sequences (MISS 100): exec.rs `sync_publish` lets a read-modify-write continue
# the release sequence of the write it reads only when the two are morally strong. Without the
# scope check a .cta release continues through an atomic in another CTA again: e_rseq_scope
# CTAREL is clean.
RSEQ = ("ptxv-sim", "fixtures", "rseq_rmw_morally_strong")
MUTANTS += [
    SrcMutant(
        "rseq-scope",
        "crates/sim/src/exec.rs",
        "                    && (old.writer == t as u32\n                        || self.launch.hw_order\n",
        "                    && (true\n                        || self.launch.hw_order\n",
        [("e_rseq_scope", "CTAREL", f, []) for f in ("ptx", "sass")],
        [RSEQ],
    ),
]

# S-SCOPE release sequences (MISS 112): the release patterns a read-modify-write continues stay
# apart (`SyncCell::cont`), each checked against the reader: its release scope includes the
# reader (rseq-head-rel), the acquiring read's scope includes its thread (rseq-head-acq), and a
# relaxed read leaves it to the acquire fence under its own thread (rseq-head-fence);
# rseq-head-join joins them into the read-modify-write's release again (the fix undone). Their
# join (`SyncCell::cont_join`) stands for them only when every one passes: rseq-head-cache drops
# that condition for another thread's read, rseq-head-own for the atomic's own thread.
RSEQ_HEAD = ("ptxv-sim", "fixtures", "rseq_head_morally_strong")
MUTANTS += [
    SrcMutant(
        "rseq-head-rel",
        "crates/sim/src/exec.rs",
        "        let cont_in = |x: &Self, e: &SyncRel| hw || x.scope_includes(e.rel_scope, e.writer, me);",
        "        let cont_in = |x: &Self, e: &SyncRel| hw || true || x.scope_includes(e.rel_scope, e.writer, me);",
        [("e_rseq_head", "FREL", f, []) for f in ("ptx", "sass")],
        [RSEQ_HEAD],
    ),
    SrcMutant(
        "rseq-head-acq",
        "crates/sim/src/exec.rs",
        "                    if self.scope_includes(m.scope, me, e.writer) {",
        "                    if true {",
        [],
        [RSEQ_HEAD],
    ),
    SrcMutant(
        "rseq-head-fence",
        "crates/sim/src/exec.rs",
        "            for e in cont {\n                rels.push((e.writer, e.rel.clone(), e.rel_scope));",
        "            for e in cont {\n                rels.push((c.writer, e.rel.clone(), e.rel_scope));",
        [("e_rseq_facq", "CTAFENCE", "ptx", [])],
        [RSEQ_HEAD],
    ),
    SrcMutant(
        "rseq-head-join",
        "crates/sim/src/exec.rs",
        "                        cont = old.cont.clone();\n"
        "                        cont_join = old.cont_join.clone();\n"
        "                        SyncCell::add_cont(\n"
        "                            &mut cont,\n"
        "                            &mut cont_join,\n"
        "                            old.writer,\n"
        "                            &old.rel,\n"
        "                            old.rel_scope,\n"
        "                        );",
        "                        vc_join(&mut rel, &old.rel);",
        [("e_rseq_head", "FREL", f, []) for f in ("ptx", "sass")]
        + [("e_rseq_facq", "CTAFENCE", "ptx", [])],
        [RSEQ_HEAD],
    ),
    SrcMutant(
        "rseq-head-cache",
        "crates/sim/src/exec.rs",
        "        let all = !cont.is_empty() && cont.len() == c.cont.len();",
        "        let all = !cont.is_empty();",
        [("e_rseq_head", "TWOHEAD", f, []) for f in ("ptx", "sass")],
        [RSEQ_HEAD],
    ),
    SrcMutant(
        "rseq-head-own",
        "crates/sim/src/exec.rs",
        "                if !c.cont.is_empty() && c.cont.iter().all(pass) {",
        "                if !c.cont.is_empty() {",
        [("e_rseq_head", "OWN", "ptx", [])],
        [RSEQ_HEAD],
    ),
    # The SASS front's shortcut for a relaxed read (one pending entry, keyed by the first
    # pattern's writer, for all continued patterns; CCTL.IVALL takes every entry) on the PTX
    # front too: a .cta acquire fence then takes a pattern whose thread is in another CTA
    # (e_rseq_twopat CTAFENCE: clean).
    SrcMutant(
        "rseq-head-shortcut",
        "crates/sim/src/exec.rs",
        "        } else if all && self.k.front == Front::Sass && self.launch.exit_fence.is_none() {",
        "        } else if all && self.launch.exit_fence.is_none() {",
        [("e_rseq_twopat", "CTAFENCE", "ptx", [])],
        [RSEQ_HEAD],
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


# MISS 53: errdist refuses a run with data-dependent preconditions (errdist.rs
# `refuse_data_dependent`, fixtures/errdist/miss53).
MUTANTS += [
    # The refusal never fires (the blind spot again).
    SrcMutant(
        "dd-refuse-off",
        "crates/cli/src/errdist.rs",
        "        .filter(|p| p.starts_with(super::equiv::DATA_DEPENDENT));",
        "        .filter(|_| false);",
        [],
        [
            ("ptxv", "errdist_guard", "data_dependent_exit_is_refused"),
            ("ptxv", "errdist_guard", "data_dependent_address_is_refused"),
        ],
    ),
    # The reference alone is checked: a pair's candidate passes.
    SrcMutant(
        "dd-reference-only",
        "crates/cli/src/errdist.rs",
        "        for k in &ks {\n            refuse_data_dependent(k)?;",
        "        for k in ks.iter().take(1) {\n            refuse_data_dependent(k)?;",
        [],
        [("ptxv", "errdist_guard", "data_dependent_candidate_is_refused")],
    ),
    # Selections count as data-dependent too: a guarded store that segment mode models
    # (N-ERRDIST-SEL) is refused (a false refusal).
    SrcMutant(
        "dd-ignores-selection",
        "crates/cli/src/equiv.rs",
        "            sel_excused += 1;\n            continue;",
        "            sel_excused += 1;",
        [],
        [("ptxv", "errdist_guard", "guarded_store_is_a_selection_not_a_refusal")],
    ),
    # fb/kernel_db/run.py `two_level_gap`: a partial verdict counts as complete again (an
    # incomplete check whose GB300-profile level certifies becomes contract_only).
    SrcMutant(
        "kdb-tolerated-complete",
        "fb/kernel_db/run.py",
        '    if v.get("complete") is not True:\n        return f"incomplete verdict',
        '    if False:\n        return f"incomplete verdict',
        [],
        py=[("fb/kernel_db", "test_run_collect.py", "ContractTolerated")],
    ),
    # fb/kernel_db/run.py `two_level_gap`: any two-level verdict clears the findings again (a
    # GB300-profile level that still reports a race makes the record contract_only).
    SrcMutant(
        "kdb-tolerated-sass",
        "fb/kernel_db/run.py",
        '    if tl.get("sass_gb300") not in ("certified", "clean"):',
        "    if False:",
        [],
        py=[("fb/kernel_db", "test_run_collect.py", "ContractTolerated")],
    ),
]


# MISS 22 (PTX modifier audit, N-CVT-F16): cvt between f32 and f16 / bf16 matches exact
# modifier lists (probes/cvt_f16 = GB300; the refusal test rebuilds each unmodeled form).
CVT_REFUSED = [("ptxv-sim", "fixtures", "cvt_f16_unmodeled_forms_refused")]
MUTANTS += [
    # cvt.f32.f16 matched on its types again: cvt.sat.f32.f16 loses its clamp (GB300 differs
    # from the unclamped value on 50,175 of the 65,536 codes).
    SrcMutant(
        "cvt-f32f16-types",
        "crates/ptx/src/lib.rs",
        '            } else if mods == ["f32", "f16"] {',
        '            } else if to == "f32" && from == "f16" {',
        [],
        CVT_REFUSED,
    ),
    # cvt.f32.bf16 likewise: cvt.ftz.f32.bf16 loses its flush of the 254 subnormal codes.
    SrcMutant(
        "cvt-f32bf16-types",
        "crates/ptx/src/lib.rs",
        '            } else if mods == ["f32", "bf16"] {',
        '            } else if to == "f32" && from == "bf16" {',
        [],
        CVT_REFUSED,
    ),
    # cvt.rn.f16.f32 drops .relu / .satfinite / .ftz again.
    SrcMutant(
        "cvt-rn-f16-mods",
        "crates/ptx/src/lib.rs",
        '                if mods != ["rn", "f16", "f32"] && mods != ["rn", "sat", "f16", "f32"] {',
        "                if false {",
        [],
        CVT_REFUSED,
    ),
    # cvt.rn.bf16.f32 likewise.
    SrcMutant(
        "cvt-rn-bf16-mods",
        "crates/ptx/src/lib.rs",
        '                if mods != ["rn", "bf16", "f32"] {',
        "                if false {",
        [],
        CVT_REFUSED,
    ),
    # cvt.rn.sat.f16.f32 without its clamp to [0, 1] (the FADD.SAT before the F2FP).
    SrcMutant(
        "cvt-f16-sat-clamp",
        "crates/ptx/src/lib.rs",
        '                if has("sat") {\n                    let tmp = lw.scratch_v()?;',
        "                if false {\n                    let tmp = lw.scratch_v()?;",
        [],
        [("ptxv-sim", "fixtures", "probe_cvt_f16_matches_hardware")],
    ),
    # cvt_lp ignores modifiers it does not know again (.ftz, a second rounding, .scaled::*).
    SrcMutant(
        "cvt-lp-mods",
        "crates/ptx/src/lib.rs",
        "        return Ok(None);\n    }\n    let has = |m: &str| mods.contains(&m);\n    let lp = |t: &str|",
        "    }\n    let has = |m: &str| mods.contains(&m);\n    let lp = |t: &str|",
        [],
        CVT_REFUSED,
    ),
]


# MISS 23 (PTX modifier audit, L-ROSETTA): integer add / sub / logic / shift take one type, or
# add / sub.sat.s32 and the packed 16x2 forms (probes/int_sat, probes/sub16x2 = GB300).
INT_REFUSED = [("ptxv-sim", "fixtures", "int_arith_unmodeled_forms_refused")]
MUTANTS += [
    # The modifier gate off: .cc, packed byte forms and .sat on other types lowered as the
    # plain op (and add.sat.u32 as the signed clamp).
    SrcMutant(
        "int-mods-gate",
        "crates/ptx/src/lib.rs",
        '            if !(matches!(base, "add" | "sub") && mods == ["sat", "s32"]) {\n                mods_only(lw, base, mods, types)?;',
        "            if false {\n                mods_only(lw, base, mods, types)?;",
        [],
        INT_REFUSED,
    ),
    # sub.{u,s}16x2 lowered on the PTX front again (not in the PTX ISA; the integrator's
    # placement keeps measured-only semantics off the contract front).
    SrcMutant(
        "sub16x2-ptx-lowered",
        "crates/ptx/src/lib.rs",
        '            bail!(lw.err("sub.u16x2 / sub.s16x2 (not in the PTX ISA) is not modeled"))',
        "            Op::Int {\n                op: IntOp::Sub16x2,\n                bits: 32,\n"
        "                d: lw.dst(op_n(0)?)?,\n                a: lw.val(op_n(1)?)?,\n"
        "                b: lw.val(op_n(2)?)?,\n            }",
        [],
        [
            ("ptxv-sim", "fixtures", "int_arith_unmodeled_forms_refused"),
            ("ptxv-sim", "fixtures", "probe_sub16x2_matches_hardware"),
        ],
    ),
    # A .SIGN register taken as a predicate destination again.
    SrcMutant(
        "sign-pdst",
        "crates/sass/src/lib.rs",
        '                    _ => bail!(self.err(format!("bad predicate destination {s}"))),',
        "                    r => Ok(r.unwrap()),",
        [],
        [("ptxv-sim", "fixtures", "sass_sign_predicate_not_a_destination")],
    ),
    # IntOp::Sub16x2 borrows from the high lane (both fronts; SASS VIADD.16x2 d, a, -b).
    SrcMutant(
        "sub16x2-borrow",
        "crates/sim/src/exec.rs",
        "                        (x.wrapping_sub(y) & 0xffff)\n                            | ((x >> 16).wrapping_sub(y >> 16) & 0xffff) << 16",
        "                        x.wrapping_sub(y) & 0xffff_ffff",
        [],
        [("ptxv-sim", "fixtures", "probe_sub16x2_matches_hardware")],
    ),
    # PLOP3 R.SIGN / UR.SIGN reads bit 0 instead of the sign bit (add / sub.sat.s32, SASS).
    SrcMutant(
        "plop3-sign-bit",
        "crates/sim/src/exec.rs",
        "File::R | File::UR => self.rd(t, &Val::Reg(r, 1)).is_ok_and(|v| v >> 31 & 1 == 1),",
        "File::R | File::UR => self.rd(t, &Val::Reg(r, 1)).is_ok_and(|v| v & 1 == 1),",
        [],
        [("ptxv-sim", "fixtures", "probe_int_sat_matches_hardware")],
    ),
    # add.sat.s32 wraps (PTX front).
    SrcMutant(
        "addsat-wrap",
        "crates/sim/src/exec.rs",
        "                    IntOp::AddSatS => (x as i32).saturating_add(y as i32) as u32 as u64,",
        "                    IntOp::AddSatS => (x as i32).wrapping_add(y as i32) as u32 as u64,",
        [],
        [("ptxv-sim", "fixtures", "probe_int_sat_matches_hardware")],
    ),
    # sub.sat.s32 wraps (PTX front).
    SrcMutant(
        "subsat-wrap",
        "crates/sim/src/exec.rs",
        "                    IntOp::SubSatS => (x as i32).saturating_sub(y as i32) as u32 as u64,",
        "                    IntOp::SubSatS => (x as i32).wrapping_sub(y as i32) as u32 as u64,",
        [],
        [("ptxv-sim", "fixtures", "probe_int_sat_matches_hardware")],
    ),
]


# MISS 24 (PTX-FMAD): a mul.sat is never contracted into an add (ptxas 13.0 / 13.3 keep
# FMUL.SAT + FADD; probes/fmad_sat = GB300).
MUTANTS += [
    SrcMutant(
        "fmad-mul-sat",
        "crates/ptx/src/lib.rs",
        '        let ("mul", ty, ftz, false, false) = fop(op)? else {',
        '        let ("mul", ty, ftz, _, false) = fop(op)? else {',
        [],
        [("ptxv-sim", "fixtures", "probe_fmad_sat_matches_hardware")],
    ),
]


# MISS 25 (N-MULTIMEM): multimem .acc::f16, and .acc::f32 outside a 16-bit float ld_reduce, are
# refused.
MM_REFUSED = [("ptxv-sim", "fixtures", "multimem_acc_unmodeled_forms_refused")]
MUTANTS += [
    # .acc::f16 accepted again.
    SrcMutant(
        "mm-acc-f16",
        "crates/ptx/src/lib.rs",
        '            if rest.contains(&"acc::f16")\n                || rest.contains(&"acc::f32")',
        '            if false\n                || rest.contains(&"acc::f32")',
        [],
        MM_REFUSED,
    ),
    # .acc::f32 accepted on every multimem op and type.
    SrcMutant(
        "mm-acc-placement",
        "crates/ptx/src/lib.rs",
        '                    && !(kind == "ld_reduce" && matches!(ety, McTy::F16 | McTy::Bf16))',
        "                    && false",
        [],
        MM_REFUSED,
    ),
]


# MISS 59 (N-CVT-F16): cvt's integer path takes the eight integer types only
# (probes/cvt_bf16_same = GB300 on the SASS front; the PTX front refuses cvt.bf16.bf16).
MUTANTS += [
    # Any type starting with u / s / b counts again: cvt.bf16.bf16 is a 16-bit copy (253 of
    # 65,536 GB300 words differ) and bf16x2 / narrow-format pairs panic.
    SrcMutant(
        "cvt-int-prefix",
        "crates/ptx/src/lib.rs",
        '            let int = |t: &str| {\n                matches!(\n                    t,\n                    "u8" | "u16" | "u32" | "u64" | "s8" | "s16" | "s32" | "s64"\n                )\n            };',
        "            let int = |t: &str| t.starts_with('u') || t.starts_with('s') || t.starts_with('b');",
        [],
        [
            ("ptxv-sim", "fixtures", "probe_cvt_bf16_same_verdicts"),
            ("ptxv-sim", "fixtures", "cvt_non_integer_types_refused"),
        ],
    ),
]


# MISS 60 (N-CVT-LP): .satfinite on a bf16x2 decode is refused (GB300 maps an e5m2 +-Inf to
# +-0x7f7f, which the decode op does not model; probes/cvt_lp_satfinite).
MUTANTS += [
    # .satfinite dropped on the decode again.
    SrcMutant(
        "cvt-lp-decode-sat",
        "crates/ptx/src/lib.rs",
        "                    && rnd == LpRnd::Rn\n                    && !sat\n",
        "                    && rnd == LpRnd::Rn\n",
        [],
        [("ptxv-sim", "fixtures", "probe_cvt_lp_satfinite_verdicts")],
    ),
]


# MISS 61 (N-CVT-LP): SASS F2FP.BF16.<fmt>.UNPACK_B lifts only the identity scale 0x7f7f.H0 of
# the unscaled decodes (probes/cvt_lp_scaled: ptxas 13.3's scaled decodes = GB300).
MUTANTS += [
    # The scale operand ignored again.
    SrcMutant(
        "unpack-b-scale",
        "crates/sass/src/lib.rs",
        "                    if ops.len() > 3\n                        || ops.get(2).is_some_and(|s| {",
        "                    if false\n                        && ops.get(2).is_some_and(|s| {",
        [],
        [("ptxv-sim", "fixtures", "probe_cvt_lp_scaled_verdicts")],
    ),
]


# MISS 62 (PTX-FMAD): a neg.f32 of a mul.ftz product read by add.ftz / sub.ftz counts as the
# readers it feeds (ptxas 13.0 / 13.3 fuse it; probes/fmad_ftz_neg, probes/fmad_multi = GB300).
MUTANTS += [
    # The neg not followed for a mul.ftz again (the mul kept apart).
    SrcMutant(
        "fmad-ftz-neg",
        "crates/ptx/src/lib.rs",
        '                        && (*jop == format!("neg.{ty}")\n                            || matches!(jbase, "mov")',
        '                        && (*jop == format!("neg.{ty}") && !ftz\n                            || matches!(jbase, "mov")',
        [],
        [
            ("ptxv-sim", "fixtures", "probe_fmad_ftz_neg_matches_hardware"),
            ("ptxv-sim", "fixtures", "probe_fmad_multi_matches_hardware"),
        ],
    ),
]


# N-NUM-TRACK: `ptxv num` taints rounding results without a node whatever their operands
# (num.rs `rounding_op`, `NumState::rounding_taint`) and makes no claim for a run whose control
# flow or effects depend on input data (`Taint::control_pcs`); fixtures c_num_param_f32 and
# c_num_branch_f32 (GB300 outputs), tests/num_untracked.rs.
NUM_PARAM = (
    "ptxv",
    "num_untracked",
    "parameter_and_tid_roundings_are_not_claimed_exact",
)
NUM_BRANCH = (
    "ptxv",
    "num_untracked",
    "control_flow_on_a_rounded_value_is_not_analyzed",
)
NUM_GATHER = (
    "ptxv",
    "num_untracked",
    "addresses_from_a_rounded_value_are_not_analyzed",
)
NUM_IDIOMS = ("ptxv", "num_untracked", "index_arithmetic_idioms_keep_their_claims")
NUM_ATOMS = (
    "ptxv",
    "num_untracked",
    "roundings_through_immediates_bit_operations_and_atomics_are_not_exact",
)
NUM_CTL = ("ptxv", "num_untracked", "control_from_rounded_values_is_refused")
NUM_UDIV = ("ptxv", "num_untracked", "integer_division_idiom_keeps_its_claims")
NUM_VOTE = ("ptxv", "num_untracked", "control_from_a_ballot_of_other_lanes_is_refused")
ERRDIST_VOTE = ("ptxv", "errdist_guard", "ballot_decided_control_is_refused")
EQUIV_VOTE = ("ptxv", "equiv", "ballot_decided_branch_blocks_the_proof")
NUM_MATCH = (
    "ptxv",
    "num_untracked",
    "control_from_a_match_of_another_lanes_rounded_value_is_refused",
)
NUM_NONFINITE = ("ptxv", "num_untracked", "nonfinite_errors_are_never_skipped")
NUM_ATOM_RMW = ("ptxv", "num_untracked", "atomics_other_than_f32_adds_keep_roundings_out_of_claims")
NUM_PARTIAL = ("ptxv", "num_untracked", "partial_and_rearranging_accesses_of_rounded_words_are_not_exact")
NUM_FOLD = ("ptxv", "num_untracked", "sass_claims_need_a_ptx_claim_with_the_same_exact_value")
# MISS 110 (N-MULTIMEM-ROUND16 scope): fixtures/r1h_nvls_bf16 and the review probes on both
# fronts; the case-level tolerance rules.
ROUND16_LANES = ("ptxv", "gpu_compare", "only_calibrated_bf16_lanes_are_within_one_ulp")
ROUND16_PROBES = ("ptxv", "gpu_compare", "review_probes_are_mismatches")
ROUND16_CASE = ("ptxv-case", "lib", "tests::expect_report_tol_never_crosses_an_infinity")
MUTANTS += [
    # Rounding results of parameters, %tid and constants are exact constants again.
    SrcMutant(
        "num-round-taint",
        "crates/cli/src/main.rs",
        "        ns.rounding_taint = true;",
        "        ns.rounding_taint = false;",
        [],
        [NUM_PARAM],
    ),
    # One `rounding_op` entry dropped: MUFU (ex2.approx of a parameter counts as exact).
    SrcMutant(
        "num-round-mufu",
        "crates/sim/src/num.rs",
        "        if ns.rounding_taint && rounding_op(&inst.op) && !const_mat {",
        "        if ns.rounding_taint && rounding_op(&inst.op) && !const_mat && !matches!(inst.op, Op::Mufu { .. }) {",
        [],
        [NUM_PARAM],
    ),
    # The refusal never fires: num follows this run's path again.
    SrcMutant(
        "num-control-off",
        "crates/cli/src/main.rs",
        "            .or(control);",
        "            .or(control.filter(|_| false));",
        [],
        [NUM_BRANCH],
    ),
    # A guarded branch, exit or store is not recorded (only a data-dependent operand is).
    SrcMutant(
        "num-control-guard",
        "crates/sim/src/num.rs",
        "                    if effect_op(&inst.op) && input_guard {",
        "                    if false && effect_op(&inst.op) && input_guard {",
        [],
        [NUM_BRANCH],
    ),
    # Only a control transfer's target counts again: a load, store or shuffle at an address
    # or lane computed from input data keeps its claims.
    SrcMutant(
        "num-control-addr",
        "crates/sim/src/num.rs",
        "            // or access other elements.\n            if active",
        "            // or access other elements.\n"
        "            if matches!(inst.op, Op::Bra { .. } | Op::Brx { .. } | Op::Ret { .. }) && active",
        [],
        [NUM_GATHER],
    ),
    # Precision: a SASS constant materialization (HFMA2 R, -RZ, RZ, hi, lo) is a rounding again
    # (c_num_index_f32's loop counter starts at one and is summed as f32(i): 128 untracked).
    SrcMutant(
        "num-const-mat-off",
        "crates/sim/src/num.rs",
        "        let const_mat = self.k.front == Front::Sass",
        "        let const_mat = false && self.k.front == Front::Sass",
        [],
        [NUM_IDIOMS],
    ),
    # Precision: an exact integer to f32 conversion (|result| < 2^24) is tainted too, and the
    # sums x + f32(%tid) lose their claims (a false refusal).
    SrcMutant(
        "num-i2f-exact",
        "crates/sim/src/num.rs",
        "                if self.num.as_ref().is_some_and(|n| n.rounding_taint) =>",
        "                if false && self.num.as_ref().is_some_and(|n| n.rounding_taint) =>",
        [],
        [NUM_PARAM],
    ),
    # The all-immediate exemption spares every SASS instruction whose operands are immediates
    # again, not only the HFMA2 constant materialization: MUFU.TANH of an immediate counts as an
    # exact constant (c_num_mufu_imm_f32: tanh(0.1f) claimed exact, 44 ulp off on GB300).
    SrcMutant(
        "num-const-mat-any",
        "crates/sim/src/num.rs",
        "            && matches!(&self.k.code[pc as usize].op, Op::Hfma2 { a, b, .. }\n"
        "                if matches!(a, Val::Imm(0)) || matches!(b, Val::Imm(0)))\n",
        "            && true\n",
        [],
        [NUM_ATOMS],
    ),
    # The value a float atomic add returns is not tainted again (c_num_atom_old_f32: the rounded
    # partial sums taken as exact constants).
    SrcMutant(
        "num-atom-old",
        "crates/sim/src/num.rs",
        "                    op: AtomOp::FAddFtz | AtomOp::FAdd64 | AtomOp::AddBf16x2 | AtomOp::AddBf16,",
        "                    op: AtomOp::Exch,",
        [],
        [NUM_ATOMS],
    ),
    # N-NUM-TRACK: a value computed outside the DAG from a node holding a rounding is not tainted
    # again: control decided by an FADD of parameters, a float loop counter or an atomic sum is
    # followed (c_num_ctl_branch_f32, c_num_ctl_loop_f32, c_num_ctl_atom_f32 claim).
    SrcMutant(
        "num-rounded-off",
        "crates/sim/src/num.rs",
        "            let cur = std::mem::take(&mut tn.cur_rounded);",
        "            let cur: Vec<usize> = std::mem::take(&mut tn.cur_rounded).into_iter().take(0).collect();",
        [],
        [NUM_CTL],
    ),
    # N-NUM-UDIV: a division's quotient is exact even when its operands are input data: the
    # load address of c_num_udiv_data_f32 (idx[t] / 7) counts as input-independent.
    SrcMutant(
        "num-udiv-taint",
        "crates/sim/src/num.rs",
        "                            (Some(&(d, false)), Some(&(n, false))) if d != 0 => {",
        "                            (Some(&(d, _)), Some(&(n, _))) if d != 0 => {",
        [],
        [NUM_UDIV],
    ),
    # Precision: no division idiom is matched; the u32 / s32 quotients deciding addresses and
    # guards refuse c_num_index_f32 and c_num_udiv_f32 (SASS front).
    SrcMutant(
        "num-udiv-off",
        "crates/sim/src/num.rs",
        "    for (i0, ins0) in code.iter().enumerate() {",
        "    for (i0, ins0) in code.iter().enumerate().take(0) {",
        [],
        [NUM_UDIV, NUM_IDIOMS],
    ),
    # MISS 65 (N-EQUIV): a ballot, vote or match result carries only its own lane's operand
    # taint again: a branch on a ballot of other lanes' input data looks input-independent
    # (c_num_vote_f32 claimed; errdist and equiv report from the captured path).
    SrcMutant(
        "num-vote-lanes",
        "crates/sim/src/num.rs",
        "                Op::Redux { .. } | Op::VoteAny { .. } | Op::VoteAll { .. } | Op::MatchAny { .. }\n"
        "            ) {",
        "                Op::Redux { .. }\n            ) {",
        [],
        [NUM_VOTE, ERRDIST_VOTE, EQUIV_VOTE],
    ),
    # MISS 63 (N-NUM-NONFINITE): a non-finite element's error is the f64 expansion difference
    # again (NaN for an overflow or a NaN output, skipped by max_err).
    SrcMutant(
        "num-nonfinite-off",
        "crates/sim/src/num.rs",
        "        let finite = hw64.is_finite() && exact_v.is_finite();",
        "        let finite = true;",
        [],
        [NUM_NONFINITE],
    ),
    # An identical non-finite value counts as error 0 even where a value in the tree overflowed
    # (the overflowed x + y plus -inf: hw NaN, expansion NaN, exact value -inf).
    SrcMutant(
        "num-nonfinite-trust-off",
        "crates/sim/src/num.rs",
        "            let trusted = !rnodes.iter().any(|&n| self.nonfinite_inside(n));",
        "            let trusted = true;",
        [],
        [NUM_NONFINITE],
    ),
    # Precision: copies of inf / NaN are not analyzed.
    SrcMutant(
        "num-nonfinite-ident-off",
        "crates/sim/src/num.rs",
        "            if trusted && identical {",
        "            if false && trusted && identical {",
        [],
        [NUM_NONFINITE],
    ),
    # MISS 67 (N-MULTIMEM-ROUND16): the one-ulp tolerance is the plain bit-difference test
    # again: an infinity against the largest finite value, or a NaN against an infinity, is
    # "within one ulp" (fixtures/r1h_nvls_inf, expected/ constructed).
    SrcMutant(
        "ulp-tol-nonfinite",
        "crates/case/src/lib.rs",
        "    if nonfinite(a) || nonfinite(b) {\n",
        "    if false && (nonfinite(a) || nonfinite(b)) {\n",
        [],
        [
            ("ptxv", "nonfinite", "one_ulp_tolerance_never_crosses_an_infinity"),
            ROUND16_CASE,
        ],
    ),
    # MISS 95 (N-NUM-TRACK): a match, vote or reduction gives each lane only its own operand's
    # rounding again: a branch on a match of another lane's rounded value looks exact
    # (c_num_match_f32 claimed on the SASS front).
    SrcMutant(
        "num-xlane-rounded",
        "crates/sim/src/num.rs",
        "        if xlane {\n            cur = active.to_vec();\n",
        "        if false && xlane {\n            cur = active.to_vec();\n",
        [],
        [NUM_MATCH],
    ),
    # MISS 114 (N-NUM-TRACK, atomics other than f32 adds), each part off: the old value read from
    # a word holding a rounding is an exact constant (c_num_atom_read_f32 claimed) ...
    SrcMutant(
        "num-rmw-old",
        "crates/sim/src/num.rs",
        "            if taint {\n                for i in 0..w {\n",
        "            if false && taint {\n                for i in 0..w {\n",
        [],
        [NUM_ATOM_RMW],
    ),
    # ... the word an integer read-modify-write computes from one (the red.or 0 words of
    # c_num_atom_store_f32 claimed) ...
    SrcMutant(
        "num-rmw-word",
        "crates/sim/src/num.rs",
        "            _ => a.old_rounded || x_inexact,\n",
        "            _ => false,\n",
        [],
        [NUM_ATOM_RMW],
    ),
    # ... the word an f64 atomic add leaves (c_num_atom_f64_f32's branch not refused) ...
    SrcMutant(
        "num-rmw-f64",
        "crates/sim/src/num.rs",
        "            AtomOp::FAdd64 => true,\n",
        "            AtomOp::FAdd64 => false,\n",
        [],
        [NUM_ATOM_RMW],
    ),
    # ... a compare-and-swap decided by a rounded word keeps the word (c_num_atom_cas_f32: the
    # run's sum claimed where the exact computation swapped) ...
    SrcMutant(
        "num-rmw-cas",
        "crates/sim/src/num.rs",
        "        let decided = cas && (a.old_rounded || a.old_input_derived || x_inexact);\n",
        "        let decided = false && cas && (a.old_rounded || a.old_input_derived || x_inexact);\n",
        [],
        [NUM_ATOM_RMW],
    ),
    # ... an exchange drops the node it writes (c_num_atom_store_f32: s an exact constant) ...
    SrcMutant(
        "num-rmw-exch",
        "crates/sim/src/num.rs",
        "            AtomOp::Exch => Some(a.x.clone()),\n",
        "            AtomOp::Exch => None,\n",
        [],
        [NUM_ATOM_RMW],
    ),
    # ... a bf16 atomic lane without a node is an exact constant (c_num_atom_cas_f32's bf16x2
    # sum claimed).
    SrcMutant(
        "num-rmw-bf16",
        "crates/sim/src/num.rs",
        "                if s.is_none() {\n                    ns.taint.mem_set((k.0, k.1, k.2 + i as u64), true);\n",
        "                if false && s.is_none() {\n                    ns.taint.mem_set((k.0, k.1, k.2 + i as u64), true);\n",
        [],
        [NUM_ATOM_RMW],
    ),
    # MISS 115 (N-NUM-TRACK, partial-width and rearranging accesses), each part off: a 16-bit
    # load of a word holding a rounding is an exact constant (c_num_ld16_f32 claimed) ...
    SrcMutant(
        "num-ld16",
        "crates/sim/src/num.rs",
        "                if ns.rounding_taint && w.is_none() && !rt {\n",
        "                if false && ns.rounding_taint && w.is_none() && !rt {\n",
        [],
        [NUM_PARTIAL],
    ),
    # ... so is a word ldmatrix .trans rearranges (c_num_ldsm_trans_f32's rebuilt words) ...
    SrcMutant(
        "num-ldsm",
        "crates/sim/src/num.rs",
        "            if ns.rounding_taint && locs.iter().any(|&l| ns.words_rounded(mem_key(l), 1)) {\n",
        "            if false && locs.iter().any(|&l| ns.words_rounded(mem_key(l), 1)) {\n",
        [],
        [NUM_PARTIAL],
    ),
    # ... a word a partial store mixes (c_num_st16_f32) ...
    SrcMutant(
        "num-st16",
        "crates/sim/src/num.rs",
        "                if ns.rounding_taint && (src_rounded || old.is_some_and(|n| ns.rounded(n))) {\n",
        "                if false && (src_rounded || old.is_some_and(|n| ns.rounded(n))) {\n",
        [],
        [NUM_PARTIAL],
    ),
    # ... a word a copy of 2-byte elements assembles (c_num_tma16_f32) ...
    SrcMutant(
        "num-copy16",
        "crates/sim/src/num.rs",
        "                    && (copied_rounded(lo, hi) || old.is_some_and(|n| ns.rounded(n)))\n",
        "                    && false\n                    && (copied_rounded(lo, hi) || old.is_some_and(|n| ns.rounded(n)))\n",
        [],
        [NUM_PARTIAL],
    ),
    # ... and stmatrix .b8 leaves the words' old nodes and takes no taint (c_num_stsm8_f32).
    SrcMutant(
        "num-stsm8",
        "crates/sim/src/exec.rs",
        "                                self.num_st(src, vi, loc, &[byte as u8], &origins, pc);\n",
        "                                let _ = &origins;\n",
        [],
        [NUM_PARTIAL],
    ),
    # MISS 108 (N-NUM-TRACK, constants ptxas folds): the SASS front claims elements whose exact
    # value differs from the PTX front's (c_num_fold_add_f32 claimed on SASS) ...
    SrcMutant(
        "num-fold-exact",
        "crates/cli/src/main.rs",
        "                    if !r.untracked && !same {\n",
        "                    if false && !r.untracked && !same {\n",
        [],
        [NUM_FOLD],
    ),
    # ... and outputs the PTX front refuses (c_num_fold_ctl_f32 claimed on SASS).
    SrcMutant(
        "num-fold-refused",
        "crates/cli/src/main.rs",
        "                matches!(ptx_out, Some(None)).then(|| {\n",
        "                (false && matches!(ptx_out, Some(None))).then(|| {\n",
        [],
        [NUM_FOLD],
    ),
]


# Time and memory budgets (membudget.rs): what a deadline or the memory budget must stop.
MUTANTS += [
    # explore.rs `seed_walk`: the seed walks ignore the budgets again (they ran to their end
    # past a deadline; no note, the verdict marked incomplete only by the search after them).
    SrcMutant(
        "seed-deadline",
        "crates/sim/src/explore.rs",
        "        if budget && over_budget() {",
        "        if false {",
        [],
        [("ptxv-sim", "seed_deadline", "seed_walks_stop_at_the_deadline")],
    ),
    # main.rs `cmd_check`: --no-gb300 / PTXV_GB300=0 no longer stop the GB300-profile re-run
    # (two_level is reported again, beside two_level_skipped).
    SrcMutant(
        "nogb300-rerun",
        "crates/cli/src/main.rs",
        "            if let Some(p) = profile.filter(|_| wanted2 && gb300 && !spent && !no_sass) {",
        "            if let Some(p) = profile.filter(|_| wanted2 && !spent && !no_sass) {",
        [],
        [("ptxv", "two_level_gate", "no_gb300_skips_the_profile_rerun")],
    ),
    # fb/kernel_db/run.py `parse_check`: a check run without the two-level verdict
    # (--no-gb300 / PTXV_GB300=0) is read as if the verdict were missing for other reasons
    # (a two-level verdict from a run with the GB300 checks off makes contract findings
    # contract_only again).
    SrcMutant(
        "kdb-nogb300",
        "fb/kernel_db/run.py",
        "    if gb300_off or skipped:",
        "    if False:",
        [],
        py=[
            ("fb/kernel_db", "test_run_collect.py", "NoGb300"),
            ("fb/kernel_db", "test_run_collect.py", "test_no_gb300_task_needs_triage"),
        ],
    ),
    # membudget.rs `time_budget_spent`: the whole check's time budget (--time-budget) never
    # counts as spent (certificates hold past it, the GB300-profile re-run starts after it).
    SrcMutant(
        "budget-spent",
        "crates/sim/src/membudget.rs",
        "pub fn time_budget_spent() -> bool {\n    let t = TOTAL_MS.load(Ordering::Relaxed);",
        "pub fn time_budget_spent() -> bool {\n    let t = 0u64;",
        [],
        [
            ("ptxv-sim", "seed_deadline", "certificates_stop_at_the_time_budget"),
            ("ptxv", "time_budget", "spent_budget_cuts_every_phase"),
        ],
    ),
    # wellsync.rs `run_logged`: the certificate's run no longer stops at the time budget (it
    # runs to its end; `certify_inner` still refuses the certificate).
    SrcMutant(
        "budget-cert-run",
        "crates/sim/src/wellsync.rs",
        "    while let Some(a) = sched.pick(&m) {\n        if crate::membudget::time_budget_spent() {",
        "    while let Some(a) = sched.pick(&m) {\n        if false {",
        [],
        [("ptxv-sim", "seed_deadline", "certificates_stop_at_the_time_budget")],
    ),
    # main.rs `cmd_check`: the GB300-profile re-run starts after the time budget is spent
    # (no `two_level_skipped`).
    SrcMutant(
        "budget-rerun",
        "crates/cli/src/main.rs",
        "            let spent = ptxv_sim::membudget::time_budget_spent() && level2.is_none();",
        "            let spent = false && level2.is_none();",
        [],
        [("ptxv", "time_budget", "spent_budget_cuts_every_phase")],
    ),
    # case/src/lib.rs `expect_compare_tol`: an output within one ulp of the captured GPU output
    # is reported as a bitwise match in `gpu_compare` (the report line is unchanged).
    SrcMutant(
        "gpu-within-ulp",
        "crates/case/src/lib.rs",
        "                res.push((o.clone(), GpuMatch::WithinUlp, true));",
        "                res.push((o.clone(), GpuMatch::Bitwise, true));",
        [],
        # Since MISS 110 an f32 output one ulp off is a mismatch: the within-one-ulp results
        # are those of calibrated bf16 lanes.
        [("ptxv", "gpu_compare", "gpu_compare_states_each_comparison"), ROUND16_LANES],
    ),
    # fb/kernel_db/run.py `gpu_match`: within one ulp counts as matching the GPU output again
    # (a clean verdict on it becomes a clean record).
    SrcMutant(
        "kdb-within-ulp",
        "fb/kernel_db/run.py",
        '    if all(r == "bitwise" for r in res):',
        '    if all(r in ("bitwise", "within_ulp") for r in res):',
        [],
        py=[("fb/kernel_db", "test_run_collect.py", "GpuCompare")],
    ),
    # MISS 110 (N-MULTIMEM-ROUND16 scope): the one-ulp tolerance ignores the provenance marks
    # again: any element one encoding off is within one ulp in a multicast case (an f32 sum, a
    # doubled lane, an overwritten lane, an exact lane).
    SrcMutant(
        "round16-unmarked",
        "crates/case/src/lib.rs",
        "                            .all(|k| r16[start + off + e + k].is_some_and(|f| f.dtype() == b.dtype))",
        "                            .all(|k| r16[start + off + e + k].is_some() || true)",
        [],
        [ROUND16_LANES, ROUND16_PROBES, ROUND16_CASE],
    ),
    # MISS 110: a byte marked with the other 16-bit format (or an f32 element of marked bytes)
    # takes the tolerance.
    SrcMutant(
        "round16-format",
        "crates/case/src/lib.rs",
        "                            .all(|k| r16[start + off + e + k].is_some_and(|f| f.dtype() == b.dtype))",
        "                            .all(|k| r16[start + off + e + k].is_some())",
        [],
        [ROUND16_CASE],
    ),
    # MISS 110: NaN against a NaN with another payload is one ulp apart again (`ulp_class`).
    SrcMutant(
        "round16-nan",
        "crates/case/src/lib.rs",
        "    if nan(a) || nan(b) {\n",
        "    if nan(a) && nan(b) {\n        return true;\n    }\n    if nan(a) || nan(b) {\n",
        [],
        [
            ROUND16_CASE,
            ("ptxv-case", "lib", "tests::one_ulp_tolerance_needs_finite_values"),
            ("ptxv-case", "gpu_compare", "nan_payloads_are_a_mismatch"),
        ],
    ),
    # MISS 110: every lane of an f16 / bf16 multimem.ld_reduce with a calibrated lane is marked,
    # exact lanes included.
    SrcMutant(
        "round16-every-lane",
        "crates/sim/src/multimem.rs",
        "                if c {\n                    m[j * w..(j + 1) * w].fill(Some(f));",
        "                if true || c {\n                    m[j * w..(j + 1) * w].fill(Some(f));",
        [],
        [ROUND16_LANES],
    ),
    # MISS 110: a register write keeps the mark of the lane the register held (the pair doubled
    # in its register passes as the lane).
    SrcMutant(
        "round16-reg-keep",
        "crates/sim/src/exec.rs",
        "        if self.round16.is_some() {\n            self.round16_forget_reg(t, r, slots);",
        "        if false && self.round16.is_some() {\n            self.round16_forget_reg(t, r, slots);",
        [],
        [ROUND16_LANES],
    ),
    # MISS 110: a memory write keeps the mark of the lane the bytes held (the pair an exchange
    # wrote over the stored lane passes as the lane).
    SrcMutant(
        "round16-mem-keep",
        "crates/sim/src/exec.rs",
        "        if write.is_some() && self.round16.is_some() {\n            self.round16_forget_mem(loc, n);",
        "        if false && write.is_some() && self.round16.is_some() {\n            self.round16_forget_mem(loc, n);",
        [],
        [ROUND16_LANES],
    ),
    # main.rs `out_dir`: without --out, every run of a case materializes its variants in the
    # case's shared default directory again (no per-invocation `run_dir`).
    SrcMutant(
        "outdir-shared",
        "crates/cli/src/main.rs",
        '            run_dir(&default_out(&root.join("out"), case)?)?',
        '            default_out(&root.join("out"), case)?',
        [],
        [
            (
                "ptxv",
                "outdir_race",
                "another_runs_copy_in_the_shared_directory_is_not_read",
            )
        ],
    ),
    # main.rs `time_budget_of`: a time budget in multi-node mode (PTXV_DIST_DIR, more than one
    # rank) is accepted again (every rank cuts its own phases; rank 0 waits for the others).
    SrcMutant(
        "budget-multinode",
        "crates/cli/src/main.rs",
        "        anyhow::ensure!(\n            !multi_node(),",
        "        anyhow::ensure!(\n            true,",
        [],
        [("ptxv", "time_budget", "multi_node_refuses_the_time_budget")],
    ),
    # explore.rs `explore_one_models`: a plain-model rerun cut short replaces what the search
    # with the model reductions found again (`keep_found` not applied).
    SrcMutant(
        "rerun-cut-drops",
        "crates/sim/src/explore.rs",
        "        if !w.complete {\n            keep_found(&mut w,",
        "        if false {\n            keep_found(&mut w,",
        [],
        [("ptxv-sim", "lib", "a_cut_rerun_keeps_what_the_first_search_found")],
    ),
    # python/ptxv/results.py `Verdict.gpu_state`: an output within one ulp of the captured GPU
    # output counts as bitwise equal again (`sim_equals_gpu` True on it).
    SrcMutant(
        "py-within-ulp",
        "python/ptxv/results.py",
        '        if all(r == "bitwise" for r in res):\n            return "bitwise"',
        '        if all(r in ("bitwise", "within_ulp") for r in res):\n            return "bitwise"',
        [],
        py=[("python/tests", "test_results.py", "GpuCompare")],
    ),
    # python/ptxv/results.py `Verdict.gpu_state`, outputs without `gpu_compare`: any line naming
    # "bitwise equal" matches again (the within-one-ulp line ends "the rest bitwise equal").
    SrcMutant(
        "py-bitwise-substring",
        "python/ptxv/results.py",
        "                if _BITWISE_LINE.match(s)\n",
        '                if "bitwise equal" in s\n',
        [],
        py=[("python/tests", "test_results.py", "GpuCompare")],
    ),
    # python/ptxv/results.py `Verdict.from_json`: an incomplete verdict listing E-WELLSYNC is
    # certified again.
    SrcMutant(
        "py-certified-partial",
        "python/ptxv/results.py",
        'certified="E-WELLSYNC" in v.get("assumptions", []) and v.get("complete") is True,',
        'certified="E-WELLSYNC" in v.get("assumptions", []),',
        [],
        py=[("python/tests", "test_results.py", "Certified")],
    ),
    # fb/kernel_db/run.py `parse_check`: an incomplete verdict listing E-WELLSYNC is stored as
    # certified again.
    SrcMutant(
        "kdb-certified-parse",
        "fb/kernel_db/run.py",
        '        "E-WELLSYNC" in v.get("assumptions", []) and v.get("complete") is True\n',
        '        "E-WELLSYNC" in v.get("assumptions", [])\n',
        [],
        py=[("fb/kernel_db", "test_run_collect.py", "CertifiedComplete")],
    ),
    # fb/kernel_db/run.py `certified`: a stored check's flag counts without a complete verdict
    # again (errdist eligibility of records stored before `parse_check` required it).
    SrcMutant(
        "kdb-certified-stored",
        "fb/kernel_db/run.py",
        '    return bool(chk.get("certified")) and chk.get("complete") is True',
        '    return bool(chk.get("certified"))',
        [],
        py=[("fb/kernel_db", "test_run_collect.py", "CertifiedComplete")],
    ),
    # fb/kernel_db/ptx_front.py `verdict`: a stored check's flag makes the record "certified"
    # without a complete verdict again.
    SrcMutant(
        "kdb-certified-ptx",
        "fb/kernel_db/ptx_front.py",
        '        return "certified" if run_certified(ch) else "clean"',
        '        return "certified" if ch.get("certified") else "clean"',
        [],
        py=[("fb/kernel_db", "test_run_collect.py", "CertifiedComplete")],
    ),
    # fb/kernel_db/run.py `two_level_gap`: the GB300-profile level's certified / clean counts
    # without a complete verdict again.
    SrcMutant(
        "kdb-certified-gb300",
        "fb/kernel_db/run.py",
        '    if tl.get("sass_complete") is not True:',
        "    if False:",
        [],
        py=[("fb/kernel_db", "test_run_collect.py", "CertifiedComplete")],
    ),
    # main.rs `certified`: E-WELLSYNC among the assumptions certifies an incomplete verdict again
    # (`two_level.sass_certified`, `mutated_inputs[].certified`, the report's "incomplete").
    SrcMutant(
        "certified-partial",
        "crates/cli/src/main.rs",
        '    v.complete && v.assumptions.iter().any(|a| a == "E-WELLSYNC")',
        '    v.assumptions.iter().any(|a| a == "E-WELLSYNC")',
        [],
        [
            (
                "ptxv",
                "certified_complete",
                "an_incomplete_verdict_listing_e_wellsync_is_not_certified",
            )
        ],
    ),
    # main.rs `checked_hw_profile` (check, suite): an unknown hardware profile name is accepted
    # again (the GB300-profile re-run silently off).
    SrcMutant(
        "hwprofile-unknown",
        "crates/cli/src/main.rs",
        '        profile.is_some() || name == "none",',
        "        true,",
        [],
        [("ptxv", "two_level_gate", "unknown_hw_profiles_are_refused")],
    ),
    # main.rs `cmd_check`: --hw-profile none turns the re-run off without `two_level_skipped`
    # again.
    SrcMutant(
        "hwprofile-none-silent",
        "crates/cli/src/main.rs",
        "            let wanted2 = ptxv_sim::hwprofile::wanted(&verdict);",
        "            let wanted2 = profile.is_some() && ptxv_sim::hwprofile::wanted(&verdict);",
        [],
        [("ptxv", "two_level_gate", "hw_profile_none_says_the_rerun_was_skipped")],
    ),
    # fb/kernel_db/run.py `gb300_disabled`: a hardware profile other than gb300 no longer
    # counts as turning the re-run off.
    SrcMutant(
        "kdb-hwprofile",
        "fb/kernel_db/run.py",
        '    if profile in (None, "--hw-profile gb300", "PTXV_HW_PROFILE=gb300"):',
        "    if True:",
        [],
        py=[("fb/kernel_db", "test_run_collect.py", "test_hw_profile")],
    ),
    # fb/kernel_db/soundness.py `certified`: a check without `complete` counts as certified
    # again (run.py reads it as not certified).
    SrcMutant(
        "kdb-snd-certified-missing",
        "fb/kernel_db/soundness.py",
        '        return bool(chk and chk.get("certified")) and chk.get("complete") is True',
        '        return bool(chk and chk.get("certified")) and chk.get("complete") is not False',
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_certified_incomplete_is_not_certified")],
    ),
    # soundness.py `record_tags`: the certify tag follows the strict `certified` again, so a
    # check flagged certified without `complete` (status clean in run.py) escapes certificate
    # misses.
    SrcMutant(
        "kdb-snd-certify-tag-strict",
        "fb/kernel_db/soundness.py",
        "        if cls.on_certificate(chk):\n",
        "        if cls.certified(chk):\n",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_certified_incomplete_is_not_certified")],
    ),
    # soundness.py `on_certificate`: E-WELLSYNC under `assumptions` (some stored checks) is not
    # read, only under `assumes`.
    SrcMutant(
        "kdb-snd-assumptions-key",
        "fb/kernel_db/soundness.py",
        '            "E-WELLSYNC" in (chk.get(k) or []) for k in ("assumes", "assumptions")',
        '            "E-WELLSYNC" in (chk.get(k) or []) for k in ("assumes",)',
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_certified_incomplete_is_not_certified")],
    ),
    # soundness.py `on_certificate`: a certified two-level SASS level is not read.
    SrcMutant(
        "kdb-snd-two-level-ignored",
        "fb/kernel_db/soundness.py",
        '            or (isinstance(tl, dict) and bool(tl.get("sass_certified")))\n',
        "            or False\n",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_certificate_signals_beside_the_check_flag")],
    ),
    # soundness.py `record_tags`: the entries under checks_extra are not read.
    SrcMutant(
        "kdb-snd-extra-ignored",
        "fb/kernel_db/soundness.py",
        '        for chk in [r.get("check"), *(extra if isinstance(extra, list) else [])]:\n',
        '        for chk in [r.get("check")]:\n',
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_certificate_signals_beside_the_check_flag")],
    ),
    # soundness.py `record_tags`: check_ptx counts only beside another check again.
    SrcMutant(
        "kdb-snd-check-ptx-needs-check",
        "fb/kernel_db/soundness.py",
        "        if isinstance(cp, dict):\n",
        "        if isinstance(cp, dict) and t:\n",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_certificate_signals_beside_the_check_flag")],
    ),
    # soundness.py `record_check`: check_ptx is not a record's check without `check` (a
    # check_ptx-only timeout grades unknown again, not unconfirmed).
    SrcMutant(
        "kdb-snd-record-check-ptx-ignored",
        "fb/kernel_db/soundness.py",
        '        for k in ("check", "check_ptx"):\n',
        '        for k in ("check",):\n',
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_check_ptx_is_the_check_without_a_check")],
    ),
    # soundness.py `of_record`: its rev and no-verdict reason read `check` only again.
    SrcMutant(
        "kdb-snd-of-record-check-only",
        "fb/kernel_db/soundness.py",
        "        chk = self.record_check(r)\n",
        '        chk = r.get("check") if isinstance(r.get("check"), dict) else {}\n',
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_check_ptx_is_the_check_without_a_check")],
    ),
    # soundness.py `of_record`: last_verdict reads `check` only again.
    SrcMutant(
        "kdb-snd-last-verdict-check-only",
        "fb/kernel_db/soundness.py",
        "                lc = self.record_check(lv)\n",
        '                lc = lv.get("check") or {}\n',
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_check_ptx_is_the_check_without_a_check")],
    ),
    # export_records.py `collect`: the last verdicts read `check` only again, so a check_ptx-only
    # record has no last_verdict.
    SrcMutant(
        "kdb-export-verdicts-check-only",
        "fb/kernel_db/export_records.py",
        "        chk = soundness.Soundness.record_check(r)\n        if verdicts is not None and chk:\n",
        '        chk = r.get("check") if isinstance(r.get("check"), dict) else {}\n'
        "        if verdicts is not None and chk:\n",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_export_last_verdict")],
    ),
    # export_records.py `base`: the export status and ptxv_rev read `check` only again, so a
    # check_ptx-only record is not_checked ("no check result") beside its grade.
    SrcMutant(
        "kdb-export-base-check-only",
        "fb/kernel_db/export_records.py",
        '        chk = soundness.Soundness.record_check(r)\n        rev = str(chk.get("ptxv_rev") or "")\n',
        '        chk = r.get("check")\n        rev = str((chk or {}).get("ptxv_rev") or "")\n',
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_annotation")],
    ),
    # soundness.py `record_tags`: check_ptx gets no coverage tag again, so misses of input
    # coverage leave its open list.
    SrcMutant(
        "kdb-snd-check-ptx-no-coverage",
        "fb/kernel_db/soundness.py",
        '            t |= cls.check_tags({**cp, "front_end": "ptx"})\n',
        '            t |= {"kdb-tooling", "race-verdict:ptx"} | ({"certify"} if cls.on_certificate(cp) else set())\n',
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_check_ptx_tags_are_a_ptx_checks")],
    ),
    # soundness.py `record_tags`: check_ptx's tags follow its front_end, so one without it gets
    # the SASS levels' tags instead of race-verdict:ptx.
    SrcMutant(
        "kdb-snd-check-ptx-front",
        "fb/kernel_db/soundness.py",
        '            t |= cls.check_tags({**cp, "front_end": "ptx"})\n',
        "            t |= cls.check_tags(cp)\n",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_check_ptx_tags_are_a_ptx_checks")],
    ),
    # soundness.py `of_record`: check_ptx beside a check is judged at the check's rev again,
    # whatever rev of its own it stores (an older build's open misses go).
    SrcMutant(
        "kdb-snd-check-ptx-at-check-rev",
        "fb/kernel_db/soundness.py",
        '            and cp["ptxv_rev"] != rev\n',
        "            and False\n",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_check_ptx_judged_at_its_own_rev")],
    ),
    # soundness.py `of_record`: with a rev of its own, check_ptx's tags are not judged at all.
    SrcMutant(
        "kdb-snd-check-ptx-own-rev-dropped",
        "fb/kernel_db/soundness.py",
        '                    (cp["ptxv_rev"], self.record_tags({"check_ptx": cp})["race"]),\n',
        "",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_check_ptx_judged_at_its_own_rev")],
    ),
    # export_records.py `apply_rev_fix`: a stored check_ptx rev is not audited.
    SrcMutant(
        "kdb-export-rev-fix-check-only",
        "fb/kernel_db/export_records.py",
        '    for k in ("check", "check_ptx"):\n',
        '    for k in ("check",):\n',
        [],
        py=[("fb/kernel_db", "test_run_collect.py", "test_rev_fix")],
    ),
    # audit_ptxv_rev.py `resolve`: check_ptx's rev is taken from the runner's logs of the check
    # task (a run of the check is no evidence of check_ptx's binary).
    SrcMutant(
        "kdb-audit-check-ptx-run-logs",
        "fb/kernel_db/audit_ptxv_rev.py",
        "    if field != \"check\":\n        # The runner's logs below are of the record's check task, not of this check.\n",
        "    if False:\n        # The runner's logs below are of the record's check task, not of this check.\n",
        [],
        py=[("fb/kernel_db", "test_run_collect.py", "test_check_ptx_rev_on_names_only")],
    ),
    # audit_ptxv_rev.py `checks_to_audit`: a stored check_ptx rev is not audited.
    SrcMutant(
        "kdb-audit-check-ptx-ignored",
        "fb/kernel_db/audit_ptxv_rev.py",
        '    if isinstance(rec.get("check_ptx"), dict) and "ptxv_rev" in rec["check_ptx"]:\n',
        "    if False:\n",
        [],
        py=[("fb/kernel_db", "test_run_collect.py", "test_check_ptx_rev_on_names_only")],
    ),
    # `cmd_check` (check and suite): the profile name is read without the check again; only
    # `check`'s early refusal is left, so `suite` runs with an unknown name.
    SrcMutant(
        "suite-hwprofile-unchecked",
        "crates/cli/src/main.rs",
        "    let (profile_name, profile) = checked_hw_profile()?;",
        "    let (profile_name, profile) = hw_profile();",
        [],
        [("ptxv", "two_level_gate", "suite_refuses_unknown_hw_profiles")],
    ),
    # The skip reason names --no-gb300 for PTXV_GB300=0 too (the shared wording again).
    SrcMutant(
        "gb300-off-cause",
        "crates/cli/src/main.rs",
        '        Some("PTXV_GB300=0")',
        '        Some("--no-gb300")',
        [],
        [("ptxv", "two_level_gate", "no_gb300_skips_the_profile_rerun")],
    ),
    # `run_dir`: no reaping; a killed run's directory stays for good.
    SrcMutant(
        "reap-off",
        "crates/cli/src/main.rs",
        "    reap_run_dirs(base);\n",
        "",
        [],
        [("ptxv", "outdir_race", "killed_runs_directories_are_removed_by_the_next_run")],
    ),
    # `reap_run_dirs` judges other hosts' (and boots') directories by this host's locks.
    SrcMutant(
        "reap-other-hosts",
        "crates/cli/src/main.rs",
        '        if h == host && mine && !d.join(".kept").exists() && run_dir_dead(&d) {',
        '        if mine && !d.join(".kept").exists() && run_dir_dead(&d) {',
        [],
        [("ptxv", "outdir_race", "killed_runs_directories_are_removed_by_the_next_run")],
    ),
    # Kept run directories (calib, probe, sb-sweep results) are reaped too.
    SrcMutant(
        "reap-kept",
        "crates/cli/src/main.rs",
        '        if h == host && mine && !d.join(".kept").exists() && run_dir_dead(&d) {',
        "        if h == host && mine && run_dir_dead(&d) {",
        [],
        [("ptxv", "outdir_race", "killed_runs_directories_are_removed_by_the_next_run")],
    ),
    # `run_dir_dead` does not ask the lock: a live run's directory (its lock held, in this or
    # another time namespace) is removed.
    SrcMutant(
        "reap-lock-ignored",
        "crates/cli/src/main.rs",
        "        .is_ok_and(|f| f.try_lock().is_ok())\n",
        "        .is_ok_and(|_| true)\n",
        [],
        [
            ("ptxv", "outdir_race", "killed_runs_directories_are_removed_by_the_next_run"),
            ("ptxv", "outdir_race", "a_live_run_in_another_time_namespace_keeps_its_directory"),
        ],
    ),
    # A directory without its lock file (its run is making it) counts as dead.
    SrcMutant(
        "reap-lock-missing",
        "crates/cli/src/main.rs",
        "        .is_ok_and(|f| f.try_lock().is_ok())\n",
        "        .map_or(true, |f| f.try_lock().is_ok())\n",
        [],
        [("ptxv", "outdir_race", "killed_runs_directories_are_removed_by_the_next_run")],
    ),
    # `lock_run_dir` creates `.lock` itself and locks it after: in between it reads as a dead
    # run's lock, and another run removes the live run's new directory.
    SrcMutant(
        "reap-lock-window",
        "crates/cli/src/main.rs",
        '    let new = d.join(".lock.new");\n',
        '    let new = d.join(".lock");\n',
        [],
        [("ptxv", "outdir_race", "a_run_taking_its_lock_keeps_its_directory")],
    ),
    # `lock_run_dir` lets the lock go when the rename reports a failure: a rename that happened
    # all the same left `.lock` there unlocked, and another run removes the live run's directory.
    SrcMutant(
        "reap-lock-rename-lost",
        "crates/cli/src/main.rs",
        '    if std::fs::rename(&new, d.join(".lock")).is_err() {\n'
        "        let _ = std::fs::remove_file(&new);\n    }\n",
        '    if std::fs::rename(&new, d.join(".lock")).is_err() {\n'
        "        drop(f);\n        let _ = std::fs::remove_file(&new);\n        return;\n    }\n",
        [],
        [("ptxv", "outdir_race", "a_run_whose_lock_rename_is_reported_failed_keeps_its_directory")],
    ),
    # `reap_run_dirs` removes a dead directory with `remove_dir_all` again: it stops at the first
    # entry it cannot remove, after `.lock` where that comes first, and no run reaps the rest.
    SrcMutant(
        "reap-remove-dir-all",
        "crates/cli/src/main.rs",
        "            remove_run_dir(&d);\n",
        "            let _ = std::fs::remove_dir_all(&d);\n",
        [],
        [("ptxv", "outdir_race", "what_a_reaper_cannot_remove_keeps_the_lock_file")],
    ),
    # `remove_run_dir` removes `.lock` with the other entries, not last.
    SrcMutant(
        "reap-lock-not-last",
        "crates/cli/src/main.rs",
        '    if !empty_dir(d, Some(".lock"), &mut false) {\n',
        "    if !empty_dir(d, None, &mut false) {\n",
        [],
        [("ptxv", "outdir_race", "what_a_reaper_cannot_remove_keeps_the_lock_file")],
    ),
    # `empty_dir` retries its rounds at every level again after a round that removed nothing:
    # what a run cannot remove k levels down costs 8^k listings.
    SrcMutant(
        "reap-rounds-nested",
        "crates/cli/src/main.rs",
        "        if !round {\n            return false;\n        }\n",
        "",
        [],
        [("ptxv", "outdir_race", "a_reaper_stops_at_what_it_cannot_remove_deep_down")],
    ),
    # `empty_dir`: a subdirectory removed by another process before the run reads it (read_dir:
    # NotFound) is not gone, so the round that met it makes no progress and the run leaves the
    # dead directory with `.lock`.
    SrcMutant(
        "reap-vanished-not-gone",
        "crates/cli/src/main.rs",
        "            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return true,\n",
        "            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return false,\n",
        [],
        [("ptxv", "outdir_race", "a_subdirectory_removed_under_a_reaper_counts_as_gone")],
    ),
    # The anchors check of --quick (crates/cli/tests/srcmut_anchors.rs): a line that a mutant
    # anchors on (reap-lock-window's) changes and the mutant is not re-pointed.
    SrcMutant(
        "srcmut-anchor-moved",
        "crates/cli/src/main.rs",
        '    let new = d.join(".lock.new");\n',
        '    let new = d.join(".lock.new"); // moved\n',
        [],
        [("ptxv", "srcmut_anchors", "every_srcmut_mutant_finds_its_text_once")],
    ),
    # srcmut.py `main`: no mutant chosen is no error again, so --check-anchors with an --only
    # name that is not a mutant says "0 of 0 anchors broken" and exits 0 (the anchors test's
    # negative control).
    SrcMutant(
        "srcmut-none-chosen-ok",
        "tools/mutate/srcmut.py",
        "    if bad or not chosen:\n",
        "    if False:\n",
        [],
        [("ptxv", "srcmut_anchors", "every_srcmut_mutant_finds_its_text_once")],
    ),
    # srcmut.py `broken_anchors` reports nothing, so --check-anchors passes with an anchor gone
    # (the anchors test's positive control: a broken anchor planted in a scratch tree).
    SrcMutant(
        "srcmut-anchors-none",
        "tools/mutate/srcmut.py",
        "        if n != 1:\n",
        "        if n < 0:\n",
        [],
        [("ptxv", "srcmut_anchors", "every_srcmut_mutant_finds_its_text_once")],
    ),
    # --check-anchors ignores --root and checks this checkout (the positive control's planted
    # anchor is never read).
    SrcMutant(
        "srcmut-root-ignored",
        "tools/mutate/srcmut.py",
        "        broken = broken_anchors(ROOT if a.root is None else a.root, chosen)\n",
        "        broken = broken_anchors(ROOT, chosen)\n",
        [],
        [("ptxv", "srcmut_anchors", "every_srcmut_mutant_finds_its_text_once")],
    ),
    # srcmut.py `main`: an --only that is given but names nothing ("", ",", " ") chooses every
    # mutant again (the anchors test's negative control with an empty --only).
    SrcMutant(
        "srcmut-only-nothing-all",
        "tools/mutate/srcmut.py",
        "    if a.only is not None and not only:\n",
        "    if False:\n",
        [],
        [("ptxv", "srcmut_anchors", "every_srcmut_mutant_finds_its_text_once")],
    ),
    # srcmut.py --check-anchors on the path the anchors check takes (no --root, every mutant),
    # where the positive control does not go (one mutant, --root): test_mutate's
    # test_srcmut_check_anchors, in --quick through the_mutation_tools_tests_pass. Only the first
    # mutant chosen is checked; mutants of .rs files are skipped; nothing is checked without
    # --root.
    SrcMutant(
        "srcmut-anchors-first-only",
        "tools/mutate/srcmut.py",
        "    for m in mutants:\n",
        "    for m in mutants[:1]:\n",
        [],
        [("ptxv", "srcmut_anchors", "the_mutation_tools_tests_pass")],
    ),
    SrcMutant(
        "srcmut-anchors-skip-rs",
        "tools/mutate/srcmut.py",
        "    for m in mutants:\n",
        '    for m in mutants:\n        if m.file.endswith(".rs"):\n            continue\n',
        [],
        [("ptxv", "srcmut_anchors", "the_mutation_tools_tests_pass")],
    ),
    SrcMutant(
        "srcmut-anchors-default-none",
        "tools/mutate/srcmut.py",
        "        broken = broken_anchors(ROOT if a.root is None else a.root, chosen)\n",
        "        broken = broken_anchors(a.root, chosen) if a.root is not None else []\n",
        [],
        [("ptxv", "srcmut_anchors", "the_mutation_tools_tests_pass")],
    ),
    # The host token without the boot: another machine of the same name, or this one before a
    # reboot, is this host.
    SrcMutant(
        "reap-boot-dropped",
        "crates/cli/src/main.rs",
        '        let host = format!("{name}.boot{boot}");\n',
        "        let host = name.clone();\n",
        [],
        [("ptxv", "outdir_race", "killed_runs_directories_are_removed_by_the_next_run")],
    ),
    # vcache.rs `eat_dir`: ptxv's output directories in the case count as case input again.
    SrcMutant(
        "vcache-output-dirs",
        "crates/cli/src/vcache.rs",
        "            if !is_output_dir(&p) {",
        "            if true {",
        [],
        [("ptxv", "outdir_race", "runs_inside_the_case_share_the_verdict_cache")],
    ),
    # compare without --out: the pair's directory, shared by concurrent runs, again.
    SrcMutant(
        "compare-shared-dir",
        "crates/cli/src/main.rs",
        "                    let d = result_dir(&pair)?;",
        "                    let d = pair.clone();",
        [],
        [("ptxv", "outdir_race", "concurrent_compares_without_out_keep_their_own_reports")],
    ),
    # equiv without --out: the file's directory, shared by concurrent runs, again.
    SrcMutant(
        "equiv-shared-dir",
        "crates/cli/src/equiv.rs",
        "        None => crate::result_dir(&std::env::temp_dir().join(format!(",
        "        None => std::path::absolute(&std::env::temp_dir().join(format!(",
        [],
        [("ptxv", "outdir_race", "concurrent_equivs_without_out_keep_their_own_summaries")],
    ),
    # main: a rank that stops with an error no longer tells the others (dist::refuse).
    SrcMutant(
        "dist-refuse-off",
        "crates/cli/src/main.rs",
        '        ptxv_sim::dist::refuse(&format!("{e:#}"));',
        "        let _ = e;",
        [],
        [
            ("ptxv", "multinode_launch", "a_rank_that_refuses_stops_the_others"),
            ("ptxv", "multinode_launch", "a_rank_that_fails_after_connecting_stops_the_others"),
        ],
    ),
    # dist.rs `connect`: waiting for higher ranks ignores refusals (it blocked in accept).
    SrcMutant(
        "dist-accept-unwatched",
        "crates/sim/src/dist.rs",
        "                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {\n                    watch();\n",
        "                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {\n",
        [],
        [("ptxv", "multinode_launch", "a_rank_that_refuses_stops_the_others")],
    ),
    # dist.rs reader: a peer's closed connection with its refusal is ignored again.
    SrcMutant(
        "dist-eof-ignored",
        "crates/sim/src/dist.rs",
        "                        stop_for(r, why.trim());",
        "                        let _ = why;",
        [],
        [("ptxv", "multinode_launch", "a_rank_that_fails_after_connecting_stops_the_others")],
    ),
    # check: input mutation and the concolic modes run in multi-node mode again.
    SrcMutant(
        "multinode-mutation-allowed",
        "crates/cli/src/main.rs",
        "        on.is_empty() || !multi_node(),",
        "        true,",
        [],
        [("ptxv", "multinode_launch", "multi_node_refuses_clock_dependent_mutation")],
    ),
]


# MISS 56: the address / control check covers every member that executes an instruction on
# some input (num.rs `may_execute`), so a guarded store whose address depends on input data
# is not a selection (N-ERRDIST-SEL) at the captured address (fixtures/errdist/miss56).
MUTANTS += [
    # The members whose guard holds on the captured run only (the miss).
    SrcMutant(
        "sel-addr-active",
        "crates/sim/src/num.rs",
        "            && may_exec\n                .iter()\n                .any(|&t| ctl.iter().any(|&r| tn.reg_t(t, r)))",
        "            && active\n                .iter()\n                .any(|&t| ctl.iter().any(|&r| tn.reg_t(t, r)))",
        [],
        [("ptxv", "errdist_guard", "selection_at_an_input_address_is_refused")],
    ),
    # Every member, also those whose guard is false and input-independent (they execute it
    # on no input): a false refusal.
    SrcMutant(
        "sel-addr-all",
        "crates/sim/src/num.rs",
        "            .filter(|t| active.contains(t) || g.is_some_and(|g| ns.taint.reg_t(*t, g)))",
        "            .filter(|t| active.contains(t) || g.is_some_and(|g| ns.taint.reg_t(*t, g)) || true)",
        [],
        [("ptxv", "errdist_guard", "input_independent_guard_keeps_its_addresses")],
    ),
]


# MISS 58: the GPU validation refuses a DAG that GB300 contradicts, except at outputs fed by
# fp32 atomic adds (errdist/validate.rs `gpu_dag_refusal`, errdist/chain.rs `atomic_fed`;
# crates/sim/tests/errdist_gpu_gate.rs). The CLI's call (errdist.rs `refuse_gpu_mismatch`)
# needs a GPU; its recorded runs are in the fix's report.
MUTANTS += [
    # A GPU mismatch never refuses (the miss).
    SrcMutant(
        "gpu-refuse-off",
        "crates/sim/src/errdist/validate.rs",
        "    (exact > 0).then(|| {",
        "    (exact > 0 && false).then(|| {",
        [],
        [("ptxv-sim", "errdist_gpu_gate", "gpu_mismatch_refuses")],
    ),
    # Atomic-fed outputs refuse too: GB300's order of the adds becomes a false refusal.
    SrcMutant(
        "gpu-refuse-atomic",
        "crates/sim/src/errdist/validate.rs",
        "    let exact = mismatch.saturating_sub(atomic);\n    (exact > 0).then(|| {",
        "    let exact = mismatch;\n    (exact > 0).then(|| {",
        [],
        [("ptxv-sim", "errdist_gpu_gate", "atomic_order_mismatch_is_not_a_refusal")],
    ),
    # Only the atomic add itself is atomic-fed, not the ops above it.
    SrcMutant(
        "gpu-fed-off",
        "crates/sim/src/errdist/chain.rs",
        "                fed[i] = fed[i] || op_args(op).iter().any(|&k| fed[k as usize]);",
        "                fed[i] = fed[i] || op_args(op).is_empty();",
        [],
        [("ptxv-sim", "errdist_gpu_gate", "atomic_fed_follows_the_dag")],
    ),
]


# MISS 58 follow-up (MISS 42's rule for Tier 3): a simulator or GPU validation refuses Tier 3 on
# the analytic tiers' criteria, a bitwise mismatch outside atomic-fed outputs or a contradicted
# P(differ) (validate.rs `tier3_refusal`), and the headline never takes a refused estimate in
# any tier (errdist.rs `headline`); with nothing standing the run refuses
# (fixtures/errdist/miss58/rare).
MUTANTS += [
    # The headline's Tier 3 arm takes Tier 3 refused or not (the regression).
    SrcMutant(
        "t3-headline-any",
        "crates/cli/src/errdist.rs",
        "        .or_else(|| es.iter().position(|e| e.tier == 3 && stands(e)))",
        "        .or_else(|| es.iter().position(|e| e.tier == 3))",
        [],
        [
            (
                "ptxv",
                "errdist_guard",
                "tier3_contradicted_by_a_validation_is_no_headline",
            )
        ],
    ),
    # No validation refuses Tier 3 on its P(differ).
    SrcMutant(
        "t3-refusal-off",
        "crates/sim/src/errdist/validate.rs",
        "    if !p_differ_agrees(pred, obs) {\n        let (lo, hi)",
        "    if false && !p_differ_agrees(pred, obs) {\n        let (lo, hi)",
        [],
        [
            (
                "ptxv-sim",
                "errdist_gpu_gate",
                "validation_refuses_tier3_on_p_differ_or_bits",
            ),
            (
                "ptxv",
                "errdist_guard",
                "tier3_contradicted_by_a_validation_is_no_headline",
            ),
        ],
    ),
    # Atomic-fed mismatches refuse Tier 3: GB300's order of the adds becomes a false refusal.
    SrcMutant(
        "t3-refusal-atomic",
        "crates/sim/src/errdist/validate.rs",
        "    let exact = mismatch.saturating_sub(atomic);\n    if exact > 0 {",
        "    let exact = mismatch;\n    if exact > 0 {",
        [],
        [
            (
                "ptxv-sim",
                "errdist_gpu_gate",
                "validation_refuses_tier3_on_p_differ_or_bits",
            )
        ],
    ),
]


# MISS 57: an identified slice function is the exact reference only on the outcome patterns of
# the slice's data-dependent decisions that its probes took (ident.rs `Outcomes`, seg.rs
# `SProg::decisions`, real.rs `eval_table`); outputs elsewhere are unknown: Tier 3 leaves them
# out and refuses past 10% of P(differ) (errdist.rs `unknown_exact_gate`), every estimate
# widens by the share's bound (fixtures/errdist/miss57).
MUTANTS += [
    # No outcome patterns recorded: the fit stands for every outcome (the miss).
    SrcMutant(
        "ident-outcomes-off",
        "crates/sim/src/errdist/nonlin/ident.rs",
        "            f.outcomes = Outcomes::of(prog.decision_count(&f.steps), &taken);",
        "            f.outcomes = Outcomes::of(0, &taken);",
        [],
        [("ptxv", "errdist_guard", "slice_off_its_probed_guard_outcome_is_refused")],
    ),
    # Guards only: a select's (or carry's) predicate is no decision.
    SrcMutant(
        "ident-select-blind",
        "crates/sim/src/seg.rs",
        "                        if matches!(r.file, File::P | File::UP) && !matches!(v, SVal::Const(_)) {",
        "                        if false && matches!(r.file, File::P | File::UP) && !matches!(v, SVal::Const(_)) {",
        [],
        [
            (
                "ptxv",
                "errdist_guard",
                "select_or_maximum_off_its_probed_outcome_is_refused",
            )
        ],
    ),
    # Tier 3 never refuses: the unknown share only widens the interval.
    SrcMutant(
        "ident-gate-off",
        "crates/cli/src/errdist.rs",
        "    if k > 0 && (obs.compared == 0 || u > EST_MAX_REL * p) {",
        "    if false && k > 0 && (obs.compared == 0 || u > EST_MAX_REL * p) {",
        [],
        [("ptxv", "errdist_guard", "slice_off_its_probed_guard_outcome_is_refused")],
    ),
    # A maximum of an unknown and a known value passes on the known one.
    SrcMutant(
        "ident-max-blind",
        "crates/sim/src/errdist/nonlin/real.rs",
        "            _ => super::super::chain::op_args(op)",
        "            Op::Nl(NlOp::Max { .. }) => false,\n            _ => super::super::chain::op_args(op)",
        [],
        [
            (
                "ptxv",
                "errdist_guard",
                "select_or_maximum_off_its_probed_outcome_is_refused",
            )
        ],
    ),
    # The elementwise law takes the fit on every element value.
    SrcMutant(
        "ident-elem-off",
        "crates/sim/src/errdist/nonlin/enumerate.rs",
        "        if real.outside[root] {",
        "        if false && real.outside[root] {",
        [],
        [("ptxv", "errdist_guard", "elementwise_law_leaves_out_the_unknown_mass")],
    ),
    # The analytic tiers take the bound only when some Tier 3 input fell outside.
    SrcMutant(
        "ident-analytic-k0",
        "crates/cli/src/errdist.rs",
        "    if let Some(u) = unknown {\n        for e in rep",
        "    if let (Some(u), true) = (unknown, obs.ident_off_inputs > 0) {\n        for e in rep",
        [],
        [
            (
                "ptxv",
                "errdist_guard",
                "analytic_tiers_take_the_unknown_bound_at_zero_inputs_off",
            )
        ],
    ),
]


# Review misses of the MISS 56-58 fixes (2026-11-11), each a check of its own.
# The store word of a selection (MISS 56 class): a thread that did not execute a store under
# a data-dependent guard taints its words (num.rs `num_gst_post`; fixtures/errdist/storeword).
MUTANTS += [
    # The words keep the taint of the captured run (the miss).
    SrcMutant(
        "sel-word-taint-off",
        "crates/sim/src/num.rs",
        "        if !executed {\n            let ns = self.num.as_mut().unwrap();\n            for &(sp, al, off) in &g.keys {",
        "        if false && !executed {\n            let ns = self.num.as_mut().unwrap();\n            for &(sp, al, off) in &g.keys {",
        [],
        [
            (
                "ptxv",
                "errdist_guard",
                "selection_store_word_read_by_another_thread_is_refused",
            )
        ],
    ),
]

# fp32 atomic adds (MISS 58 class): the DAG flushes as red/atom.add.f32 does (chain.rs
# `NodeKind::AtomAdd` lowering; fixtures/errdist/atomftz), and a GB300 mismatch is exempt only
# where another order of an atomic location's chain explains it (chain.rs `AtomChain`).
MUTANTS += [
    # The atomic add without its flush (the miss).
    SrcMutant(
        "atom-ftz-off",
        "crates/sim/src/errdist/chain.rs",
        "                    rz: false,\n                    ftz: *ftz,\n                },",
        "                    rz: false,\n                    ftz: false && *ftz,\n                },",
        [],
        [("ptxv", "errdist_guard", "fp32_atomic_adds_flush_subnormal_operands")],
    ),
    # Two operands from zero count as a free order.
    SrcMutant(
        "atom-operands-off",
        "crates/sim/src/errdist/chain.rs",
        "        if k + usize::from(init != 0.0) < 3 {",
        "        if false && k + usize::from(init != 0.0) < 3 {",
        [],
        [
            (
                "ptxv-sim",
                "errdist_gpu_gate",
                "atomic_order_explains_only_a_free_order_within_its_bound",
            )
        ],
    ),
    # Any value is some order's.
    SrcMutant(
        "atom-bound-off",
        "crates/sim/src/errdist/chain.rs",
        "        (gpu as f64 - dag as f64).abs() <= bound",
        "        bound >= 0.0",
        [],
        [
            (
                "ptxv-sim",
                "errdist_gpu_gate",
                "atomic_order_explains_only_a_free_order_within_its_bound",
            )
        ],
    ),
    # With explained atomic mismatches the moments still raise no refusal.
    SrcMutant(
        "t3-moments-off",
        "crates/sim/src/errdist/validate.rs",
        "    if atomic > 0 && !moments_agree(pred, obs) {",
        "    if false && atomic > 0 && !moments_agree(pred, obs) {",
        [],
        [("ptxv-sim", "errdist_gpu_gate", "moments_count_where_atomic_order_differs")],
    ),
    # The captured-output check skipped for any kernel with atomic adds again.
    SrcMutant(
        "atom-capture-skip",
        "crates/cli/src/errdist.rs",
        "    let skip_all = fed.iter().any(|&f| f) && run.captured_out.is_some();",
        "    let skip_all = fed.iter().any(|&f| f);",
        [],
        [
            (
                "ptxv",
                "errdist_guard",
                "captured_atomic_outputs_outside_any_order_are_refused",
            )
        ],
    ),
]
# One thread's adds counted as a free order (`AtomChain::order_explains` without its
# `self.sources < 2`): equivalent since the MISS 84 follow-up, which added `free`.
EXCLUDED["atom-sources-off"] = (
    "AtomChain::order_explains without `self.sources < 2`: since the MISS 84 follow-up a chain "
    "is free only where two adds adjacent in the run's order come from different actors and "
    "happens-before leaves them unordered (num.rs `note_atom_order`: `pa != actor`); a source "
    "is one actor, so a chain of one source is never free and `!self.free` refuses it first. "
    "atom-free-off and atom-hb-off mutate that check"
)

# Value pieces inside a slice (MISS 57 class): a clamp, a minimum / maximum, a sign are
# decisions (seg.rs `inst_pieces`; fixtures/errdist/clamp).
MUTANTS += [
    # No value piece is a decision (the miss).
    SrcMutant(
        "pieces-off",
        "crates/sim/src/seg.rs",
        "                    inst_pieces(op, *dst, &rv, &rp, &data, f);",
        "                    let _ = (op, dst, &rv, &rp, &data);",
        [],
        [("ptxv", "errdist_guard", "clamps_inside_a_slice_are_decisions")],
    ),
    # `.SAT` alone is no decision.
    SrcMutant(
        "pieces-sat-off",
        "crates/sim/src/seg.rs",
        "        f(low);\n        f(!low && r > 0x3f80_0000);",
        "        f(false && low);\n        f(false);",
        [],
        [("ptxv", "errdist_guard", "clamps_inside_a_slice_are_decisions")],
    ),
    # A minimum / maximum's pick alone is no decision.
    SrcMutant(
        "pieces-minmax-off",
        "crates/sim/src/seg.rs",
        "        r == if ftz { fp::f32_ftz(x) } else { x } && !fp::f32_is_nan(r)",
        "        false && r == if ftz { fp::f32_ftz(x) } else { x } && !fp::f32_is_nan(r)",
        [],
        [("ptxv", "errdist_guard", "clamps_inside_a_slice_are_decisions")],
    ),
]

# MISS 57 at 0 Tier 3 inputs outside: Tier 3 widens by the share's bound as well (errdist.rs;
# fixtures/errdist/k0).
MUTANTS += [
    SrcMutant(
        "t3-widen-k0-off",
        "crates/cli/src/errdist.rs",
        "    if let Some(u) = unknown {\n        widen_unknown(&mut est, u);\n    }",
        "    if let (Some(u), true) = (unknown, obs.ident_off_inputs > 0) {\n        widen_unknown(&mut est, u);\n    }",
        [],
        [("ptxv", "errdist_guard", "tier3_takes_the_unknown_bound_at_zero_inputs_off")],
    ),
]

# --pre-round keeps Tier 3's validation refusal (errdist.rs `t3_claim`; fixtures/errdist/miss58/rare):
# the mutant skips it whenever --pre-round is given, as before MISS 87. Since the MISS 87
# follow-up the refusal holds whatever --pre-round takes (pre-mark-all, pre-wc-off: per output).
MUTANTS += [
    SrcMutant(
        "preround-bypass",
        "crates/cli/src/errdist.rs",
        "    if let Some(claim) = t3_claim {",
        "    if let (false, Some(claim)) = (a.pre_round, t3_claim) {",
        [],
        [
            (
                "ptxv",
                "errdist_guard",
                "pre_round_keeps_the_refusals_where_it_changes_nothing",
            )
        ],
    ),
]

# A run's inputs bound P(differ) exactly against a claimed interval (validate.rs
# `p_contradicts`): a discrepancy confined to 10 or fewer inputs passes the batch-means test.
MUTANTS += [
    SrcMutant(
        "p-exact-off",
        "crates/sim/src/errdist/validate.rs",
        "    (claim.1 < lo || claim.0 > hi).then_some((lo, hi))",
        "    (false && (claim.1 < lo || claim.0 > hi)).then_some((lo, hi))",
        [],
        [("ptxv-sim", "errdist_gpu_gate", "few_inputs_contradict_a_claimed_interval")],
    ),
]

# A pair's elementwise law counts P(differ) where the exact value is unknown (enumerate.rs;
# MISS 85 follow-up): the unknown mass left out again refuses the gelu pair's exact Tier 0.
MUTANTS += [
    SrcMutant(
        "pair-elem-unknown-off",
        "crates/sim/src/errdist/nonlin/enumerate.rs",
        "                        pair_out += p;\n                        pair_p2 += p * p;",
        "                        outside += p;\n                        pair_p2 += 0.0 * p;",
        [],
        [
            (
                "ptxv",
                "errdist_guard",
                "pair_elementwise_law_counts_codes_with_an_unknown_exact_value",
            )
        ],
    ),
]


# MISS 84 follow-up: another order of an atomic chain's adds excuses a GB300 value only where
# the bound stands (k u < 1/2) and the run's happens-before leaves two adds unordered
# (num.rs `note_atom_order`, chain.rs `AtomChain::free`; fixtures/errdist/atomord).
MUTANTS += [
    # A vacuous bound excuses any value again.
    SrcMutant(
        "atom-vacuous-excuses",
        "crates/sim/src/errdist/chain.rs",
        "            // (until the MISS 84 follow-up, any value was).\n            return false;",
        "            // (until the MISS 84 follow-up, any value was).\n            return true;",
        [],
        [
            (
                "ptxv-sim",
                "errdist_gpu_gate",
                "atomic_order_explains_nothing_past_a_vacuous_bound",
            )
        ],
    ),
    # The chain's freedom ignored: barrier-ordered adds are excused again.
    SrcMutant(
        "atom-free-off",
        "crates/sim/src/errdist/chain.rs",
        "        if self.sources < 2 || !self.free || !dag.is_finite() || !gpu.is_finite() {",
        "        if self.sources < 2 || !dag.is_finite() || !gpu.is_finite() {",
        [],
        [
            (
                "ptxv-sim",
                "errdist_gpu_gate",
                "atomic_order_explains_only_a_free_order_within_its_bound",
            ),
            ("ptxv", "errdist_guard", "barrier_ordered_atomic_adds_have_one_order"),
        ],
    ),
    # The happens-before test dropped: adds of two actors count as unordered whatever orders them.
    SrcMutant(
        "atom-hb-off",
        "crates/sim/src/num.rs",
        "            .is_some_and(|&(pk, pa, pc)| pk == k && pa != actor && pc > vc_get(vc, pa));",
        "            .is_some_and(|&(pk, pa, pc)| pk == k && pa != actor && pc >= 0 * vc_get(vc, pa));",
        [],
        [("ptxv", "errdist_guard", "barrier_ordered_atomic_adds_have_one_order")],
    ),
]

# MISS 87 follow-up: --pre-round marks an output only where its value changes (chain.rs
# `Prog::pre_round`), and the validations observe the DAG's value there (errdist.rs
# `pre_take`) instead of skipping the worst-case check and Tier 3's refusal run-wide.
MUTANTS += [
    # Every chain output marked again, an f32 D included.
    SrcMutant(
        "pre-mark-all",
        "crates/sim/src/errdist/chain.rs",
        "                p.pre_rounded[o] = last != self.outputs[o];",
        "                p.pre_rounded[o] = true;",
        [],
        [
            (
                "ptxv-sim",
                "lib",
                "errdist::chain::tests::pre_round_marks_only_outputs_it_changes",
            ),
            (
                "ptxv",
                "errdist_guard",
                "pre_round_keeps_the_checks_of_an_output_it_leaves_alone",
            ),
        ],
    ),
    # The worst-case check skipped at every output, as --pre-round used to skip it run-wide.
    SrcMutant(
        "pre-wc-off",
        "crates/sim/src/errdist/mc.rs",
        "        if let (Some((rel, k, ab, one)), false) = (st.wc, pre) {",
        "        if let (Some((rel, k, ab, one)), false) = (st.wc, pre || st.r.pre_rounded.is_empty()) {",
        [],
        [
            (
                "ptxv",
                "errdist_guard",
                "pre_round_keeps_the_checks_of_an_output_it_leaves_alone",
            )
        ],
    ),
]

# MISS 99: integer steps inside a slice are decisions (seg.rs `inst_pieces`: an F2I's
# integer; `bit_pieces` / `cut_pieces`: fields of a float cut by LOP3 / SHF / PRMT;
# fixtures/errdist/trunc).
MUTANTS += [
    # The F2I's integer no decision again (NaN and saturation only).
    SrcMutant(
        "f2i-steps-off",
        "crates/sim/src/seg.rs",
        "            for i in 0..32 {\n                f(r >> i & 1 == 1);\n            }\n        }",
        "            for i in 0..0 {\n                f(r >> i & 1 == 1);\n            }\n        }",
        [],
        [("ptxv", "errdist_guard", "integer_steps_inside_a_slice_are_decisions")],
    ),
    # No bit field decides (only the operands' signs, as before MISS 99).
    SrcMutant(
        "bit-pieces-off",
        "crates/sim/src/seg.rs",
        "fn bit_pieces(cl: &[Bit; 32], ops: &[(u32, bool)], r: u32, f: &mut dyn FnMut(bool)) {\n",
        "fn bit_pieces(cl: &[Bit; 32], ops: &[(u32, bool)], r: u32, f: &mut dyn FnMut(bool)) {\n    if r != r.wrapping_add(1) {\n        return;\n    }\n",
        [],
        [("ptxv", "errdist_guard", "integer_steps_inside_a_slice_are_decisions")],
    ),
    # A wide field (a mantissa given a fixed exponent) decides nothing.
    SrcMutant(
        "cut-pieces-off",
        "crates/sim/src/seg.rs",
        "    cut_pieces(ops, &low_cut, r, f);\n}",
        "    let _ = (&low_cut, r, &f);\n}",
        [],
        [("ptxv", "errdist_guard", "integer_steps_inside_a_slice_are_decisions")],
    ),
]

# MISS 99 follow-up: a rounding whose result lane took at most two values on the
# identification's probes while a data operand took more is a decision by that lane's bits
# (seg.rs `SProg::rounding_steps`; fixtures/errdist/trunc/fpstep.ptx).
MUTANTS += [
    # No rounding acts as a step: each departs from the fit by its error again (the miss).
    SrcMutant(
        "round-steps-off",
        "crates/sim/src/errdist/nonlin/ident.rs",
        "            f.steps = prog.rounding_steps(fin);",
        "            f.steps = vec![];",
        [],
        [("ptxv", "errdist_guard", "fp_rounding_steps_inside_a_slice_are_decisions")],
    ),
]

# MISS 120: Tier 3's P(differ) interval counts inputs where few differ (mc.rs `p_interval` /
# `input_bound`), and Tier 3 grows at P 0 (errdist.rs, the growth loop; fixtures/errdist/clus).
# An observed run that refutes a claim keeps the narrower interval (`p_interval_obs`).
MUTANTS += [
    # The outputs' interval alone at 0 differing inputs again (the miss).
    SrcMutant(
        "t3-input-bound-off",
        "crates/sim/src/errdist/mc.rs",
        "    if !f.is_empty() && f.iter().filter(|&&x| x > 0.0).count() < SMALL_K {",
        "    if false && !f.is_empty() && f.iter().filter(|&&x| x > 0.0).count() < SMALL_K {",
        [],
        [
            ("ptxv-sim", "lib", "errdist::mc::tests::p_interval_counts_inputs_where_few_differ"),
            ("ptxv", "errdist_adv", "tier3_interval_counts_inputs_where_outputs_differ_together"),
        ],
    ),
    # No growth at P 0.
    SrcMutant(
        "t3-grow-p0-off",
        "crates/cli/src/errdist.rs",
        "            let (mut want, mut bud) = if p == 0.0 && obs.compared > 0 {\n                (f64::INFINITY, budget)",
        "            let (mut want, mut bud) = if p == 0.0 && obs.compared > 0 {\n                (0.0, 0.0)",
        [],
        [("ptxv", "errdist_adv", "tier3_interval_counts_inputs_where_outputs_differ_together")],
    ),
    # Growth at P 0 where Tier 3 compared nothing (every output out of domain): up to 206 s
    # per kernel_db run for an interval that stays [0, 1].
    SrcMutant(
        "t3-grow-ood",
        "crates/cli/src/errdist.rs",
        "            let (mut want, mut bud) = if p == 0.0 && obs.compared > 0 {",
        "            let (mut want, mut bud) = if p == 0.0 {",
        [],
        [("ptxv", "errdist_guard", "nonfinite_outputs_are_out_of_domain")],
    ),
    # A validation's observed side takes the input-level bound too (an analytic estimate's
    # check, validate.rs `against_estimate`).
    SrcMutant(
        "obs-input-bound",
        "crates/sim/src/errdist/validate.rs",
        "    let ci = super::mc::p_interval_obs(obs, 0.05);\n    let oci = super::mc::p_interval_obs(obs, 0.001);",
        "    let ci = super::mc::p_interval(obs, 0.05);\n    let oci = super::mc::p_interval(obs, 0.001);",
        [],
        [("ptxv-sim", "errdist_gpu_gate", "an_observed_run_refutes_with_the_outputs_and_inputs_intervals")],
    ),
]

# MISS 64: a Tier 3 that compared no output is refused and cross-checks nothing (errdist.rs
# `refuse_no_compared`).
MUTANTS += [
    # Kept: the headline P(differ) 0 over [0, 1] again, and it confirms analytic estimates.
    SrcMutant(
        "ed-no-cmp-kept",
        "crates/cli/src/errdist.rs",
        "    if obs.compared > 0 {\n        return;\n    }",
        "    if true {\n        return;\n    }",
        [],
        [
            ("ptxv", "nonfinite", "errdist_every_input_out_of_domain_is_refused"),
            ("ptxv", "errdist_guard", "nonfinite_outputs_are_out_of_domain"),
        ],
    ),
]

# Matched roundings of a pair are paired in op order (nonlin/agg.rs `pair_matches`).
MUTANTS += [
    # The hash map's iteration order picks the pairs again.
    SrcMutant(
        "pair-hash-order",
        "crates/sim/src/errdist/nonlin/agg.rs",
        "    order.sort_by_key(|(kr, kc)| (at(op_r, kr), at(op_c, kc), *kr, *kc));",
        "    let _ = &at;",
        [],
        [("ptxv-sim", "lib", "errdist::nonlin::agg::tests::matched_roundings_pair_in_op_order")],
    ),
]

# MISS 91: a linearized estimate needs its Tier 3 cross-check resolved to 5% of p3,
# and its interval joins Tier 3's (cli/src/errdist.rs, the cross-check).
MUTANTS += [
    # The tolerance (Tier 3's half-width, up to 20% of p3) and a Tier 3 of 0 confirm Tier 2
    # again.
    SrcMutant(
        "xcheck-t2-res-off",
        "crates/cli/src/errdist.rs",
        "                if ok && e.tier == 2 && !(p3 > 0.0 && hw <= XCHECK_T2_HW * p3) {",
        "                if ok && e.tier == 2 && !(p3 >= 0.0) {",
        [],
        [
            ("ptxv", "errdist_adv", "crosscheck_needs_a_tier3_difference_for_tier2"),
            ("ptxv", "errdist_adv", "tier2_needs_resolution_and_joins_tier3s_interval"),
        ],
    ),
    # A confirmed Tier 2 keeps its own interval (the spread over its nominal inputs) again.
    SrcMutant(
        "xcheck-join-off",
        "crates/cli/src/errdist.rs",
        "                    if jlo < plo || jhi > phi {",
        "                    if jlo < plo && jhi > phi && jlo < 0.0 {",
        [],
        [("ptxv", "errdist_adv", "tier2_needs_resolution_and_joins_tier3s_interval")],
    ),
]


# MISS 32 / MISS 34 and the review after them: when an asynchronous copy's writes happen
# relative to its waits (E-ASYNC-STEP, E-CPASYNC, E-CPASYNC-MBAR).
MUTANTS += [
    # E-ASYNC-STEP: `cp.async.bulk.wait_group.read` acquires the bulk store's global writes
    # again, as the full wait (MISS 32: the PTX front called READ clean).
    SrcMutant(
        "bulk-read-is-full-wait",
        "crates/sim/src/exec.rs",
        "*read);",
        "false);",
        [("e_tma_store_readback", "READ", "ptx", [])],
        [("ptxv-sim", "fixtures", "tma_store_readback_after_wait_read_races")],
    ),
    # E-ASYNC-STEP: a bulk store's global writes at the epoch of its shared-memory reads, so
    # `.read` orders them too.
    SrcMutant(
        "bulk-write-read-epoch",
        "crates/sim/src/exec.rs",
        "        vc_tick(&mut wop.vc, op.actor);\n",
        "",
        [("e_tma_store_readback", "READ", "ptx", [])],
        [("ptxv-sim", "fixtures", "tma_store_readback_after_wait_read_races")],
    ),
    # E-ASYNC-STEP, SASS: a bulk `DEPBAR` acquires the store's global writes at once again,
    # without the `CCTL.IVALL` (MISS 102: READ4 clean and certified on the SASS front, so its
    # GB300 level, which reports the stale L1 read, never ran).
    SrcMutant(
        "bulk-depbar-acquires-writes",
        "crates/sim/src/exec.rs",
        "let defer = !read && self.k.front == Front::Sass && self.l1.is_none();",
        "let defer = false;",
        [
            ("e_tma_read_l1", "READ4", "sass", []),
            ("e_tma_read_l1", "READ4", "sass", ["--certify"]),
            ("e_bulk_read_l1", "RREAD", "sass", []),
            ("e_bulk_read_l1_other", "READ", "sass", []),
        ],
        [
            ("ptxv-sim", "fixtures", "tma_store_read_l1_needs_cctl"),
            ("ptxv-sim", "fixtures", "bulk_store_and_reduction_after_wait_read_race"),
            ("ptxv-sim", "fixtures", "bulk_store_read_by_another_thread_after_wait_read_races"),
        ],
    ),
    # E-CPASYNC: a copy's write runs as its issuing thread at its issue clock again (MISS 34:
    # a barrier before the wait ordered it; LATEWAIT clean and certified).
    SrcMutant(
        "cpasync-write-issue-clock",
        "crates/sim/src/exec.rs",
        "self.access(actor, &vc, c.pc",
        "self.access(t as u32, &c.vc, c.pc",
        [("e_cpasync_early", "LATEWAIT", "sass", ["--certify"])],
        [("ptxv-sim", "fixtures", "cpasync_write_is_not_ordered_by_a_barrier_before_the_wait")],
    ),
    # E-CPASYNC-MBAR: copies tracked by cp.async.mbarrier.arrive land at the arrive's clock
    # again (EARLY and OWN clean and certified; GB300 194/200 and 200/200 wrong).
    SrcMutant(
        "cparrive-arrive-clock",
        "crates/sim/src/exec.rs",
        "        self.cp_land(op.issuer as usize, op.actor, 1, writes);\n",
        "        for c in writes {\n"
        "            let loc = Loc::Shared { cta, off: c.off };\n"
        "            self.access(op.actor, &op.vc, c.pc, loc, c.data.len(), Some(&c.data));\n"
        "            self.sync_clear(loc);\n"
        "        }\n",
        [("e_cpasync_mbar_early", "EARLY", "ptx", []), ("e_cpasync_mbar_early", "OWN", "sass", ["--certify"])],
        [("ptxv-sim", "fixtures", "cpasync_mbar_copies_land_at_their_issue_clocks")],
    ),
    # E-CPASYNC: copies retired together are not checked against each other (SAME, GROUPS,
    # MBAR clean; GB300 keeps the first copy's bytes in 1000/1000 runs of SAME).
    SrcMutant(
        "cp-overlap-off",
        "crates/sim/src/exec.rs",
        "        self.cp_overlap(t, &done);\n",
        "",
        [("e_cpasync_overlap", "SAME", "ptx", []), ("e_cpasync_overlap", "MBAR", "sass", [])],
        [("ptxv-sim", "fixtures", "cpasync_copies_in_flight_together_race")],
    ),
    # MISS 68: no fail-closed check past a thread's async actor ids (with the ids cut to 4096,
    # e_async_actor_cap MANY's 4097th id is the next thread's, past the only thread).
    SrcMutant(
        "async-ids-check-off",
        "crates/sim/src/machine.rs",
        "        if serial + 1 >= ids {\n            let msg = format!(\n                \"{} starts its asynchronous",
        "        if false {\n            let msg = format!(\n                \"{} starts its asynchronous",
        [],
        [("ptxv-sim", "fixtures", "async_actor_ids_fail_closed")],
    ),
    # MISS 68: the old wrap back (report unmodeled, then an id within the thread's range, which
    # aliases one of its own earlier actors; the run goes on).
    SrcMutant(
        "async-ids-wrap",
        "crates/sim/src/machine.rs",
        "            return Err(msg);\n        }\n        self.threads[t].make_mut().tma_count += 1;\n        Ok(self.tma_actor_id(t, serial))\n",
        "            self.report(\"unmodeled\", \"async-actor-ids\".into(), msg);\n        }\n        self.threads[t].make_mut().tma_count += 1;\n        Ok(self.tma_actor_id(t, serial % (ids - 1)))\n",
        [],
        [("ptxv-sim", "fixtures", "async_actor_ids_fail_closed")],
    ),
    # MISS 68: the fixed stride of 4096 ids per thread back (with the check: the repros' 4097th
    # async operation fails closed instead of taking the next thread's id).
    SrcMutant(
        "async-ids-stride-4096",
        "crates/sim/src/machine.rs",
        "    room.max(1).ilog2()\n",
        "    let _ = room;\n    12\n",
        [("e_async_ids_nc", "good", "ptx", []), ("e_async_ids_stream", "good", "sass", []),
         ("e_async_ids_last", "good", "ptx", [])],
        [("ptxv-sim", "fixtures", "async_ids_past_4096_do_not_alias")],
    ),
    # E-CPASYNC: copies landed by two waits of one thread (one wait actor, two epochs) are not
    # checked against each other (LATER clean).
    SrcMutant(
        "cp-relanded-off",
        "crates/sim/src/exec.rs",
        "            self.cp_relanded(cta, actor, epoch, &c);\n",
        "",
        [("e_cpasync_overlap", "LATER", "ptx", []), ("e_cpasync_overlap", "LATER", "sass", [])],
        [("ptxv-sim", "fixtures", "cpasync_copies_in_flight_together_race")],
    ),
    # E-CPASYNC: every wait lands at the same epoch of the wait actor, so a clock that knows
    # one landing knows them all (LATER clean).
    SrcMutant(
        "cp-wait-epoch-fixed",
        "crates/sim/src/exec.rs",
        "        th.cp_epoch += 1;\n",
        "        th.cp_epoch = 1;\n",
        [("e_cpasync_overlap", "LATER", "ptx", [])],
        [("ptxv-sim", "fixtures", "cpasync_copies_in_flight_together_race")],
    ),
    # E-CPASYNC (SASS): lanes not at the DEPBAR that retired their copies are not ordered after
    # them at their own later DEPBAR (e_cpasync_div_wait good: a race on the SASS front).
    SrcMutant(
        "cp-warp-owed-off",
        "crates/sim/src/exec.rs",
        "                if *g < upto && *rest & act != 0 {\n",
        "                if false {\n",
        [("e_cpasync_div_wait", "good", "sass", [])],
        [("ptxv-sim", "fixtures", "cpasync_divergent_waits_order_their_lanes")],
    ),
    # E-CPASYNC-PROXY: a fence that happens-before the async-proxy read covers a cp.async
    # write whether or not it follows the copy's completion (e_cpasync_proxy_early EARLY clean).
    SrcMutant(
        "cpasync-proxy-fence-any-epoch",
        "crates/sim/src/exec.rs",
        "                            w.clock <= vc_get(fv, w.actor) && vc_get(fv, f as u32) <= seen\n",
        "                            vc_get(fv, f as u32) <= seen\n",
        [("e_cpasync_proxy_early", "EARLY", "ptx", []), ("e_cpasync_proxy_early", "EARLY", "sass", [])],
        [("ptxv-sim", "fixtures", "cpasync_proxy_fence_must_follow_the_copy_s_completion")],
    ),
    # E-CPASYNC (SASS): owed lanes join the copies another path's DEPBAR retired at any later
    # DEPBAR, covering them or not (e_cpasync_div_cover NEIGH: contract, the race missed).
    SrcMutant(
        "cp-landed-uncovered",
        "crates/sim/src/exec.rs",
        "                if *g < upto && *rest & act != 0 {\n",
        "                if *rest & act != 0 {\n",
        [("e_cpasync_div_cover", "NEIGH", "sass", [])],
        [("ptxv-sim", "fixtures", "cpasync_divergent_waits_join_only_the_copies_they_cover")],
    ),
    # E-ASYNC-STEP: a bulk reduction's global writes at the epoch of its shared-memory reads,
    # so `.read` orders them (e_bulk_read_l1 RREAD clean on the PTX front).
    SrcMutant(
        "bulk-red-read-epoch",
        "crates/sim/src/exec.rs",
        "                if let Err(e) = self.bulk_red(\n                    &wop,\n",
        "                if let Err(e) = self.bulk_red(\n                    &op,\n",
        [("e_bulk_read_l1", "RREAD", "ptx", [])],
        [("ptxv-sim", "fixtures", "bulk_store_and_reduction_after_wait_read_race")],
    ),
    # E-ASYNC-STEP (precision): `.read` drops the group, so a later full wait no longer orders
    # the store's writes (e_tma_read_then_full good: a race).
    SrcMutant(
        "bulk-read-drops-group",
        "crates/sim/src/exec.rs",
        "                    vc_join(if defer { &mut w } else { &mut j }, wvc);\n                }\n                read\n",
        "                    vc_join(if defer { &mut w } else { &mut j }, wvc);\n                }\n                false\n",
        [("e_tma_read_then_full", "good", "ptx", [])],
        [("ptxv-sim", "fixtures", "tma_store_full_wait_after_wait_read_orders_the_writes")],
    ),
    # E-ASYNC-STEP (precision): a launch-group boundary folds a completed bulk store at its
    # reads' epoch only, so a later group's load of the destination races with the writes.
    SrcMutant(
        "fold-read-clock",
        "crates/sim/src/groups.rs",
        "            vc_join(&mut th.vc, &vc_single(actor, clock + 1));\n",
        "            vc_join(&mut th.vc, &vc_single(actor, clock));\n",
        [],
        [("ptxv-sim", "fixtures", "launch_group_boundary_orders_a_bulk_store_s_writes")],
    ),
]


# The ported over-arrival, SASS BRA.U and s2f6 rules.
MUTANTS += [
    # Over-arrivals: an update that lands after a completion its actor does not know of no
    # longer writes the over-arrival key, so the reductions keep commuting it
    # (fixtures/mbar_over_arrival clean on both explorers).
    SrcMutant(
        "over-arrival-late-off",
        "crates/sim/src/exec.rs",
        "let late = k > 0 && !vc_covers(vc, &mb.done_land[((k - 1) % 2) as usize]);",
        "let late = false;",
        [("mbar_over_arrival", "good", "sass", []), ("mbar_over_arrival", "good", "ptx", [])],
        [("ptxv-sim", "fixtures", "mbar_over_arrival_reaches_late_order")],
    ),
    # SASS `@P BRA.U` as a per-lane branch again (probes/bra_u: outputs differ from GB300's).
    SrcMutant(
        "bra-u-per-lane",
        "crates/sass/src/lib.rs",
        'all: has("U") && !has("ANY") && !has("NOT_TID0") && !guard.is_always(),',
        "all: false,",
        [],
        [("ptxv-sim", "fixtures", "probe_bra_u_matches_hardware")],
    ),
    # s2f6 `.relu`: negative lanes no longer map to +0 (probes/s2f6_cvt/forms differs from GB300).
    SrcMutant(
        "s2f6-relu-dropped",
        "crates/sim/src/lowp.rs",
        "    if relu && i < 0 {\n        return 0;\n    }\n",
        "",
        [],
        [
            ("ptxv-sim", "fixtures", "probe_s2f6_cvt_forms_match_hardware"),
            ("ptxv-sim", "lowp", "s2f6_model_replays_gpu_sweeps"),
        ],
    ),
]


# MISS 97: a `.SIGN` register (a PLOP3 input, ptxas's add.sat.s32 / sub.sat.s32) read as its
# bit 31 by every model reader of predicate operands (`Reg::pred_bit`), and accepted only as a
# PLOP3 input. Each mutant restores one site's old reading (the register as nonzero).
MUTANTS += [
    # Slices (N-SEG): a predicate argument of a slice read as nonzero again.
    SrcMutant(
        "sign-seg",
        "crates/sim/src/seg.rs",
        "                                .is_none_or(|(_, s)| r.pred_bit(get(*s, vals))),",
        "                                .is_none_or(|(_, s)| get(*s, vals) != 0),",
        [],
        [
            ("ptxv-sim", "lib", "seg::tests::sign_predicate_reads_bit_31"),
            ("ptxv", "equiv", "sign_predicate_slices_are_swept_apart"),
        ],
    ),
    # N-FINITE's slice values: a constant predicate argument read as nonzero again.
    SrcMutant(
        "sign-num",
        "crates/sim/src/num.rs",
        "                            Some(K(c)) => Some(r.pred_bit(c) != pr.neg),",
        "                            Some(K(c)) => Some((c != 0) != pr.neg),",
        [],
        [("ptxv-sim", "lib", "num::fin_tests::constant_sign_predicate_reads_bit_31")],
    ),
    # E-CONST-BRA's constant guards.
    SrcMutant(
        "sign-constguard",
        "crates/sim/src/explore.rs",
        "            env.get(&r).map(|&v| r.pred_bit(v) ^ p.neg)",
        "            env.get(&r).map(|&v| (v != 0) ^ p.neg)",
        [],
        [("ptxv-sim", "lib", "explore::sign_pred_tests::const_guard_reads_sign_bit")],
    ),
    # The look-ahead walk's predicates (constant and per lane).
    SrcMutant(
        "sign-lacfg",
        "crates/sim/src/explore.rs",
        "            Av::K(v) => Av::K((r.pred_bit(v) ^ p.neg) as u64),\n"
        "            Av::L(j) => {\n"
        "                let v = self.lv[j as usize]\n"
        "                    .iter()\n"
        "                    .map(|&x| (r.pred_bit(x) ^ p.neg) as u64)",
        "            Av::K(v) => Av::K(((v != 0) ^ p.neg) as u64),\n"
        "            Av::L(j) => {\n"
        "                let v = self.lv[j as usize]\n"
        "                    .iter()\n"
        "                    .map(|&x| ((x != 0) ^ p.neg) as u64)",
        [],
        [("ptxv-sim", "lib", "explore::sign_pred_tests::lookahead_pred_reads_sign_bit")],
    ),
    # elect_lanes' per-lane predicates.
    SrcMutant(
        "sign-elect",
        "crates/sim/src/elect_lanes.rs",
        "            Av::Lanes(v) => Av::Lanes(v.map(|x| (r.pred_bit(x) ^ p.neg) as u64)),",
        "            Av::Lanes(v) => Av::Lanes(v.map(|x| ((x != 0) ^ p.neg) as u64)),",
        [],
        [("ptxv-sim", "lib", "elect_lanes::sign_pred_tests::sign_register_reads_bit_31")],
    ),
    # S-INIT: a `.SIGN` input's undefined bit 31 not reported again.
    SrcMutant(
        "sign-predundef",
        "crates/sim/src/exec.rs",
        "            matches!(r.file, File::P | File::UP | File::R | File::UR)\n"
        "                && r.pred_bit(self.reg_umask(t, *r) as u64)",
        "            matches!(r.file, File::P | File::UP) && self.reg_umask(t, *r) != 0",
        [],
        [("ptxv-sim", "register_definedness", "plop3_sign_input_with_undefined_bit_31_is_undefined")],
    ),
    # S-INIT: a `.SIGN` input undefined when any bit of its mask is (not bit 31 alone).
    SrcMutant(
        "sign-predundef-mask",
        "crates/sim/src/exec.rs",
        "                && r.pred_bit(self.reg_umask(t, *r) as u64)",
        "                && self.reg_umask(t, *r) != 0",
        [],
        [("ptxv-sim", "register_definedness", "plop3_sign_input_definedness_is_bit_31_of_its_mask")],
    ),
    # The SASS parser: `.SIGN` accepted as any predicate operand again.
    SrcMutant(
        "sign-parser",
        "crates/sass/src/lib.rs",
        "        if p.reg.is_some_and(|r| matches!(r.file, File::R | File::UR)) {",
        "        if false {",
        [],
        [("ptxv-sim", "fixtures", "sass_sign_predicate_plop3_inputs_only")],
    ),
]


# MISS 98: `ptxv equiv` refuses a proven level the GPU outputs contradict
# (fixtures/c_equiv_captured).
MUTANTS += [
    # The refusal removed (the level kept, as before).
    SrcMutant(
        "captured-refusal",
        "crates/cli/src/equiv.rs",
        "                (CONTRADICTED.to_string(), false)",
        "                (got, proven)",
        [],
        [("ptxv", "equiv", "captured_gpu_mismatch_refuses_the_level")],
    ),
    # A kernel's simulated output no longer held to its capture.
    SrcMutant(
        "captured-simgpu",
        "crates/cli/src/equiv.rs",
        "    (differs.is_some() || exempt.0 > 0).then_some(SimGpu { differs, exempt })",
        "    (exempt.0 > 0).then_some(SimGpu {\n        differs: None,\n        exempt,\n    })",
        [],
        [("ptxv", "equiv", "captured_gpu_mismatch_refuses_the_level")],
    ),
    # Elementwise mode: the element functions no longer held to the captured outputs.
    SrcMutant(
        "captured-ew",
        "crates/cli/src/equiv.rs",
        "                contradicted |= bad > 0;\n                cap.push(",
        "                cap.push(",
        [],
        [("ptxv", "equiv", "captured_gpu_mismatch_refuses_the_level")],
    ),
    # L0 against the pair of captures.
    SrcMutant(
        "captured-l0",
        "crates/cli/src/equiv.rs",
        "        return (eq < n).then(|| {",
        "        return (eq < eq).then(|| {",
        [],
        [("ptxv", "bin:ptxv", "equiv::captured_tests::captured_outputs_against_the_level")],
    ),
    # L1 / L2 / L1~ bounds against the pair of captures.
    SrcMutant(
        "captured-bound",
        "crates/cli/src/equiv.rs",
        "        if k > 0 {\n            why.push(",
        "        if k > k {\n            why.push(",
        [],
        [("ptxv", "bin:ptxv", "equiv::captured_tests::captured_outputs_against_the_level")],
    ),
    # The float-atomic exemption removed: the atomic row sum's simulated output (one order of
    # the adds) held to its capture (the GPU's order).
    SrcMutant(
        "captured-atomic-exempt",
        "crates/cli/src/equiv.rs",
        "    let Some(ns) = m.num.as_ref().filter(|n| n.atomic_adds > 0) else {",
        "    let Some(ns) = m.num.as_ref().filter(|_| false) else {",
        [],
        [("ptxv", "equiv", "row_sums_are_bounded")],
    ),
    # The exemption for every f32 output of a kernel with a float atomic add (the batch-3 rule):
    # a scratch-buffer atomic hides a changed capture at any output again.
    SrcMutant(
        "captured-atomic-whole",
        "crates/cli/src/equiv.rs",
        "            if node.is_some_and(|x| ns.atomic_order_dependent(x, &mut memo)) {\n"
        "                dep[off..off + 4].fill(true);",
        "            if true {\n                dep[off..off + 4].fill(true);",
        [],
        [("ptxv", "equiv", "atomic_adds_exempt_only_their_outputs")],
    ),
    # Any atomic add exempts its output, not only an order-dependent chain: two adds into +0
    # (one order's bits) hide a changed capture.
    SrcMutant(
        "captured-atomic-chain",
        "crates/sim/src/num.rs",
        "            s >= 3 || (s == 2 && (!base_zero || mixed))",
        "            s >= 1",
        [],
        [("ptxv", "equiv", "atomic_adds_exempt_only_their_outputs")],
    ),
    # A non-finite captured output against a finite one passes L1 again.
    SrcMutant(
        "captured-nonfinite",
        "crates/cli/src/equiv.rs",
        "        !finite && r.0 != c.0 && !(a.is_nan() && b.is_nan()),",
        "        false,",
        [],
        [
            ("ptxv", "equiv", "atomic_adds_exempt_only_their_outputs"),
            ("ptxv", "bin:ptxv", "equiv::captured_tests::captured_pairs_against_the_level"),
        ],
    ),
    # L0 takes two NaNs as equal again (another payload passes).
    SrcMutant(
        "captured-l0-nan",
        "crates/cli/src/equiv.rs",
        "        r.0 == c.0,\n        finite && bound",
        "        r.0 == c.0 || a.is_nan() && b.is_nan(),\n        finite && bound",
        [],
        [("ptxv", "bin:ptxv", "equiv::captured_tests::captured_pairs_against_the_level")],
    ),
    # An unproven level not held to the captures again.
    SrcMutant(
        "captured-unproven",
        "crates/cli/src/equiv.rs",
        "        let claims = !level.starts_with(\"none\");",
        "        let claims = got == level && !level.starts_with(\"none\");",
        [],
        [
            ("ptxv", "equiv", "unproven_levels_are_held_to_the_captures"),
            ("ptxv", "compare", "compare_unproven_level_contradicted_by_the_captures"),
        ],
    ),
    # compare: a proven segment-mode rerun replaces a contradicted plain run again.
    SrcMutant(
        "compare-rerun-contradicted",
        "crates/cli/src/compare.rs",
        "    !contradicted(plain) && (contradicted(seg) || rank(seg) > rank(plain))",
        "    rank(seg) > rank(plain)",
        [],
        [("ptxv", "bin:ptxv", "compare::tests::a_contradicted_run_is_kept")],
    ),
    # The captures read from byte 0 again (`expect_offset` aside).
    SrcMutant(
        "captured-offset",
        "crates/cli/src/equiv.rs",
        "    let off = b.expect_offset as usize;\n    let h = ",
        "    let off = 0usize;\n    let h = ",
        [],
        [("ptxv", "equiv", "partial_captures_are_compared_where_captured")],
    ),
    # `expect_ignore` aside again.
    SrcMutant(
        "captured-ignore",
        "crates/cli/src/equiv.rs",
        "                if s < off || t > off + bytes.len() || ignored {",
        "                if s < off || t > off + bytes.len() {",
        [],
        [("ptxv", "equiv", "partial_captures_are_compared_where_captured")],
    ),
]

# MISS 109: L0 up to the sign of zero (an output that is the other's plus +0) is its own level,
# L0~0, never L0. Each mutant names it L0 again somewhere, or loosens / tightens the capture
# check of the relaxed level (fixtures/c_equiv_zero_sign).
ZERO_SIGN_TESTS = [
    ("ptxv", "equiv", "zero_sign_is_its_own_level"),
    ("ptxv", "compare", "compare_zero_sign_is_not_bitwise"),
]
ZERO_SIGN_UNIT = [("ptxv", "bin:ptxv", "equiv::captured_tests::captured_zero_signs_against_the_level")]
MUTANTS += [
    # The peel relabeled as L0: L0 holds at an output equal only up to the sign of zero.
    SrcMutant(
        "l0z-relabel",
        "crates/sim/src/equiv.rs",
        "            l0_zero_sign += 1;\n            note!(0, ZERO_SIGN.into());",
        "            l0_zero_sign += 1;",
        [],
        ZERO_SIGN_TESTS,
    ),
    # `strongest` names the relaxed level L0.
    SrcMutant(
        "l0z-strongest",
        "crates/cli/src/equiv.rs",
        "    } else if holds(\"L0~0\") {\n        \"L0~0\".into()",
        "    } else if holds(\"L0~0\") {\n        \"L0\".into()",
        [],
        ZERO_SIGN_TESTS,
    ),
    # `strongest` matches a level by prefix again: "L0~0 up to the sign of zero" holds "L0".
    SrcMutant(
        "l0z-prefix",
        "crates/cli/src/equiv.rs",
        "            .any(|l| l.level.split(' ').next() == Some(p) && l.holds)",
        "            .any(|l| l.level.starts_with(p) && l.holds)",
        [],
        ZERO_SIGN_TESTS,
    ),
    # compare's headline calls L0~0 "bitwise identical (L0)" again.
    SrcMutant(
        "l0z-headline",
        "crates/cli/src/compare.rs",
        "    } else if level == \"L0\" && proven {",
        "    } else if level.starts_with(\"L0\") && proven {",
        [],
        [
            ("ptxv", "compare", "compare_zero_sign_is_not_bitwise"),
            ("ptxv", "bin:ptxv", "compare::tests::the_zero_sign_level_is_not_bitwise"),
        ],
    ),
    # The capture check holds L0~0 to bitwise equality (GB300's +0 against -0 refuses it).
    SrcMutant(
        "l0z-captured-strict",
        "crates/cli/src/equiv.rs",
        "    r.0 != c.0 && !(level == \"L0~0\" && opposite_zeros(r, c))",
        "    r.0 != c.0",
        [],
        ZERO_SIGN_TESTS[:1] + ZERO_SIGN_UNIT,
    ),
    # The capture check allows any difference at an L0~0 output.
    SrcMutant(
        "l0z-captured-any",
        "crates/cli/src/equiv.rs",
        "    r.0 != c.0 && !(level == \"L0~0\" && opposite_zeros(r, c))",
        "    r.0 != c.0 && level != \"L0~0\"",
        [],
        ZERO_SIGN_UNIT,
    ),
    # The capture check allows +0 against -0 at an L0 output too.
    SrcMutant(
        "l0z-captured-l0",
        "crates/cli/src/equiv.rs",
        "    r.0 != c.0 && !(level == \"L0~0\" && opposite_zeros(r, c))",
        "    r.0 != c.0 && !opposite_zeros(r, c)",
        [],
        ZERO_SIGN_UNIT,
    ),
]

# T2: narrow-format cvt forms ptxas 13.0 and 13.3 both reject are refused by the PTX front
# (cvt_lp; the cvt enumeration). Each mutant lowers one rule's forms again.
CVT_LP_T2 = [("ptxv-ptx", "lib", "tests::cvt_lp_forms_ptxas_rejects_refused")]
# ... and their operands' register types (the review's 8,718 lines with other types).
CVT_LP_TYPES = [("ptxv-ptx", "lib", "tests::cvt_lp_operand_types_ptxas_rejects_refused")]
MUTANTS += [
    # f8 / f6 / f4 x2 encodes with any rounding, without `.satfinite`.
    SrcMutant(
        "lp-encode-rn-sat",
        "crates/ptx/src/lib.rs",
        "        _ => rnd == LpRnd::Rn && sat,",
        "        _ => true,",
        [],
        CVT_LP_T2,
    ),
    # ue8m0x2 encodes with `.rn` / `.rs` or `.relu`.
    SrcMutant(
        "lp-ue8m0-encode",
        "crates/ptx/src/lib.rs",
        "        Lp::Ue8m0 => matches!(rnd, LpRnd::Rz | LpRnd::Rp) && !relu,",
        "        Lp::Ue8m0 => true,",
        [],
        CVT_LP_T2,
    ),
    # ue8m0x2 encodes from f16x2.
    SrcMutant(
        "lp-ue8m0-f16x2",
        "crates/ptx/src/lib.rs",
        '                && (t != Lp::Ue8m0 || from == "bf16x2")\n',
        "                && true\n",
        [],
        CVT_LP_T2,
    ),
    # e2m1x2 operands in 16-bit registers.
    SrcMutant(
        "lp-e2m1-byte",
        "crates/ptx/src/lib.rs",
        '{ if t == Lp::E2m1 { &["b8"] } else { &["b16"] } }',
        '{ if t == Lp::E2m1 { &["b8", "b16"] } else { &["b16"] } }',
        [],
        CVT_LP_T2,
    ),
    # `.rs` x4 encodes without `.satfinite`.
    SrcMutant(
        "lp-x4-sat",
        "crates/ptx/src/lib.rs",
        '            if rnd == LpRnd::Rs\n                && sat\n',
        '            if rnd == LpRnd::Rs\n',
        [],
        CVT_LP_T2,
    ),
    # ue8m0x2 decodes to f16x2 or with `.relu`.
    SrcMutant(
        "lp-ue8m0-decode",
        "crates/ptx/src/lib.rs",
        "                    && (f != Lp::Ue8m0 || t == Lp::Bf16 && !relu)\n",
        "                    && true\n",
        [],
        CVT_LP_T2,
    ),
    # Any declared register type accepted (an x2 code pair in a .b32, f32 values in .u32, ...).
    SrcMutant(
        "lp-reg-types",
        "crates/ptx/src/lib.rs",
        "            Some(t) => ok.contains(&t.as_str()),",
        "            Some(_) => true,",
        [],
        CVT_LP_TYPES,
    ),
    # x4 random bits of any 32-bit or 16-bit type, or an immediate.
    SrcMutant(
        "lp-x4-rbits",
        "crates/ptx/src/lib.rs",
        '                && class(2, &["b32"], false)',
        '                && class(2, &["b32", "u32", "b16"], true)',
        [],
        CVT_LP_TYPES,
    ),
    # A half-pair encode's source of any 32-bit type.
    SrcMutant(
        "lp-half-src",
        "crates/ptx/src/lib.rs",
        "                    if from == \"bf16x2\" && matches!(t, Lp::E2m1 | Lp::Ue8m0) {",
        "                    if true {",
        [],
        CVT_LP_TYPES,
    ),
    # A decode's source of any type.
    SrcMutant(
        "lp-decode-src",
        "crates/ptx/src/lib.rs",
        "                    && class(1, code2(f), false) =>",
        "                    && true =>",
        [],
        CVT_LP_TYPES,
    ),
]


# MISS 134: an fp32 atomic add's flush flag is part of equiv's canonical form, and a chain of
# flushing and non-flushing adds is unordered (fixtures/c_equiv_atomic_ftz). Each mutant proves
# L0 between such kernels again, or holds such a chain's outputs to one order's capture.
ATOM_FTZ_TESTS = [("ptxv", "equiv", "atomic_flush_modes_are_not_bitwise")]
MUTANTS += [
    # The flush flag dropped from the hash: red.add.f32 against a bulk add of the same value
    # into +0 is L0 again.
    SrcMutant(
        "atom-l0-noflag",
        "crates/sim/src/equiv.rs",
        "                    hs.push((ftz, hx));",
        "                    hs.push((false && ftz, hx));",
        [],
        [("ptxv-sim", "lib", "equiv::soundness_tests::atomic_flush_modes_are_part_of_the_canonical_form")]
        + ATOM_FTZ_TESTS,
    ),
    # A chain of both flush modes sorted as one of a single mode: two adds into +0 in opposite
    # orders are L0 again.
    SrcMutant(
        "atom-l0-mixed-sorted",
        "crates/sim/src/equiv.rs",
        "                if mixed {\n                    cx.unordered.get_or_insert_with(|| {",
        "                if false && mixed {\n                    cx.unordered.get_or_insert_with(|| {",
        [],
        [("ptxv-sim", "lib", "equiv::soundness_tests::atomic_flush_modes_are_part_of_the_canonical_form")]
        + ATOM_FTZ_TESTS,
    ),
    # The capture exemption without the rule: such a chain's outputs held to the simulated
    # order's bits.
    SrcMutant(
        "atom-order-mixed-off",
        "crates/sim/src/num.rs",
        "                mixed |= f != ftz;",
        "                mixed |= false && f != ftz;",
        [],
        [("ptxv-sim", "lib", "num::untracked_producer_tests::mixed_flush_atomic_adds_are_order_dependent")]
        + ATOM_FTZ_TESTS,
    ),
]


# MISS 135: L0~0 only where the peeled x + (+0) keeps x's NaNs (x holds no NaN but the canonical
# one; fixtures/c_equiv_zero_sign_nan). Each mutant peels an operand that can hold a NaN payload.
ZERO_NAN_UNIT = [("ptxv-sim", "lib", "equiv::soundness_tests::zero_peel_needs_canonical_nans")]
MUTANTS += [
    # Any operand peeled again: a copy against the copy + 0 is L0~0.
    SrcMutant(
        "l0z-nan-any",
        "crates/sim/src/equiv.rs",
        "                    if let Some(x) = x.filter(|&x| self.nan_canonical(x)) {",
        "                    if let Some(x) = x.filter(|&x| self.nan_canonical(x) || true) {",
        [],
        ZERO_NAN_UNIT + [("ptxv", "equiv", "zero_sign_needs_canonical_nans")],
    ),
    # A min / max or a guarded store's choice accepted without its operands.
    SrcMutant(
        "l0z-nan-choice",
        "crates/sim/src/equiv.rs",
        "                NodeKind::Max { a, b, .. } | NodeKind::Sel { a, b, .. } => stack.extend([*a, *b]),",
        "                NodeKind::Max { .. } | NodeKind::Sel { .. } => {}",
        [],
        ZERO_NAN_UNIT,
    ),
    # A constant NaN with a payload accepted.
    SrcMutant(
        "l0z-nan-leaf",
        "crates/sim/src/equiv.rs",
        "                    if f32::from_bits(nd.hw).is_nan() && nd.hw != crate::fp::F32_CANON_NAN {",
        "                    if false {",
        [],
        ZERO_NAN_UNIT,
    ),
]

# MISS 114 follow-up (the independent review's surviving mutant): a compare-and-swap decided by
# its old value alone, input data (fixtures/c_num_cas_after_or_f32).
MUTANTS += [
    # MISS 114: a compare-and-swap decided by its old value alone (input data) is not decided:
    # c_num_cas_after_or_f32 claims out[0] exact (true error 2.2 absolute).
    SrcMutant(
        "num-cas-old-input",
        "crates/sim/src/num.rs",
        "        let decided = cas && (a.old_rounded || a.old_input_derived || x_inexact);\n",
        "        let decided = cas && (a.old_rounded || x_inexact);\n",
        [],
        [NUM_ATOM_RMW],
    ),
]

# MISS 110 follow-up (the independent review's surviving mutant): equiv's comparison with the
# capture takes only calibrated lanes as within one ulp (fixtures/r1h_nvls_bf16 copied, an exact
# lane one encoding off).
ROUND16_EXACT = ("ptxv", "equiv", "exact_multimem_lane_one_ulp_off_differs_from_its_capture")
MUTANTS += [
    # MISS 110: every byte of a multimem output taken as a calibrated bf16 lane in equiv's
    # comparison with the capture: an exact lane one encoding off passes as within one ulp.
    SrcMutant(
        "round16-equiv-fab",
        "crates/cli/src/equiv.rs",
        "    let marks = run.m.round16_marks();\n",
        "    let marks = run.m.round16_marks().map(|v| vec![Some(ptxv_case::Round16::Bf16); v.len()]);\n",
        [],
        [ROUND16_EXACT],
    ),
]

# MISS 158: local memory keeps taint and nodes (fixtures/c_num_local_f32, c_num_local_mix_f32,
# c_num_localctl_rnd_f32, c_equiv_local).
NUM_LOCAL = ("ptxv", "num_untracked", "local_memory_keeps_taint_and_nodes")
EQUIV_LOCAL = ("ptxv", "equiv", "local_memory_keeps_input_taint")
MUTANTS += [
    # MISS 158: an LDL.U16 (ld.local.b16) of tainted bytes is not input data (ref_local16
    # L0 PROVEN) ...
    SrcMutant(
        "local-sub-ld-taint",
        "crates/sim/src/num.rs",
        "        if (0..n as u64).any(|b| ns.taint.mem_has((k.0, k.1, k.2 + b))) {\n",
        "        if false && (0..n as u64).any(|b| ns.taint.mem_has((k.0, k.1, k.2 + b))) {\n",
        [],
        [EQUIV_LOCAL],
    ),
    # ... nor are bytes of a word holding a rounding, without its node (c_num_local_f32:
    # out[0..32] exact constants) ...
    SrcMutant(
        "local-sub-ld-rounded",
        "crates/sim/src/num.rs",
        "        } else if ns.rounding_taint && ns.words_rounded(k, n as u64) {\n",
        "        } else if false && ns.rounding_taint && ns.words_rounded(k, n as u64) {\n",
        [],
        [NUM_LOCAL],
    ),
    # ... an STL.U16 (st.local.u16) drops its register's taint (out[32..64] claimed on the PTX
    # front; ref_local16) ...
    SrcMutant(
        "local-sub-st-taint",
        "crates/sim/src/num.rs",
        "        let tainted = matches!(v, Val::Reg(r, _) if ns.taint.reg_t(t, *r));\n"
        "        for b in 0..n as u64 {\n",
        "        let tainted = false && matches!(v, Val::Reg(r, _) if ns.taint.reg_t(t, *r));\n"
        "        for b in 0..n as u64 {\n",
        [],
        [NUM_LOCAL, EQUIV_LOCAL],
    ),
    # ... or the mix of a rounding with other bytes (c_num_local_mix_f32: a constant half or byte
    # stored over a word holding s is an exact constant) ...
    SrcMutant(
        "local-sub-st-rounded",
        "crates/sim/src/num.rs",
        "        if ns.rounding_taint && (src_rounded || old.is_some_and(|m| ns.rounded(m))) {\n",
        "        if false && (src_rounded || old.is_some_and(|m| ns.rounded(m))) {\n",
        [],
        [NUM_LOCAL],
    ),
    # ... an ld.local.b64 drops its bytes' taint (ref_local64 L0 PROVEN) ...
    SrcMutant(
        "local-ld64-taint",
        "crates/sim/src/num.rs",
        "        if (a..a + 8).any(|b| ns.taint.mem_has((4, t64, b))) {\n",
        "        if false && (a..a + 8).any(|b| ns.taint.mem_has((4, t64, b))) {\n",
        [],
        [EQUIV_LOCAL],
    ),
    # ... or its words' nodes (out[64..128] exact constants) ...
    SrcMutant(
        "local-ld64-nodes",
        "crates/sim/src/num.rs",
        "        ns.set(t, d, lo);\n        ns.set(t, lane_reg(d, 1), hi);\n",
        "        ns.set(t, d, lo.filter(|_| false));\n        ns.set(t, lane_reg(d, 1), hi.filter(|_| false));\n",
        [],
        [NUM_LOCAL],
    ),
    # ... an st.local.b64 drops its register's taint (ref_local64) ...
    SrcMutant(
        "local-st64-taint",
        "crates/sim/src/num.rs",
        "        let tainted = ns.taint.reg_t(t, r);\n        for b in a..a + 8 {\n",
        "        let tainted = false && ns.taint.reg_t(t, r);\n        for b in a..a + 8 {\n",
        [],
        [EQUIV_LOCAL],
    ),
    # ... or its lanes' nodes (out[64..128] exact constants).
    SrcMutant(
        "local-st64-nodes",
        "crates/sim/src/num.rs",
        "            match ns.node_of_reg(t, lane) {\n",
        "            match ns.node_of_reg(t, lane).filter(|_| false) {\n",
        [],
        [NUM_LOCAL],
    ),
]

# MISS 159: a compare-and-swap whose compare is input data has an input-dependent outcome
# (fixtures/c_equiv_cas).
EQUIV_CAS = ("ptxv", "equiv", "failed_cas_with_input_compare_taints_its_word")
MUTANTS += [
    # MISS 159: a failed compare-and-swap whose compare is input data leaves its word's taint
    # (ref_cas L0 PROVEN).
    SrcMutant(
        "equiv-cas-compare",
        "crates/sim/src/num.rs",
        "                    if cmp || old_input_derived {\n",
        "                    if false && cmp || old_input_derived {\n",
        [],
        [EQUIV_CAS],
    ),
]


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
    if (m.runs or m.tests) and not build(tree, target):
        return {"name": m.name, "status": "error", "why": "build failed"}
    runs = []
    caught = False
    for d, pat, sel in m.py:
        cmd = [
            os.environ.get("PTXV_PYTHON", sys.executable),
            "-m",
            "unittest",
            "discover",
            "-s",
            d,
            "-p",
            pat,
            "-k",
            sel,
        ]
        r = subprocess.run(cmd, cwd=tree, capture_output=True, text=True, timeout=1800)
        ran = next(
            (
                int(ln.split()[1])
                for ln in r.stderr.splitlines()
                if ln.startswith("Ran ")
            ),
            0,
        )
        if ran == 0:
            tail = r.stderr.strip().splitlines()[-3:]
            return {
                "name": m.name,
                "status": "error",
                "why": f"python tests did not run: {d}/{pat} -k {sel}: {tail}",
            }
        hit = r.returncode != 0
        caught |= hit
        runs.append(
            {"py": f"{d}/{pat} -k {sel}", "ran": ran, "rc": r.returncode, "caught": hit}
        )
    for pkg, binary, filt in m.tests:
        env = dict(os.environ, CARGO_TARGET_DIR=str(target), CUDA_VISIBLE_DEVICES="")
        wrap = os.environ.get("PTXV_JOB_WRAPPER", "").split()
        cmd = [*wrap, "cargo", "test", "--release", *test_target(pkg, binary), filt]
        r = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=3600)
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
        # Heavy (#[ignore]d) tests run too: a mutant's test is named on purpose.
        cmd = [*wrap, "cargo", "test", "--release", *test_target(pkg, binary), "--", "--exact", name, "--include-ignored"]
        r = subprocess.run(cmd, cwd=tree, capture_output=True, text=True, timeout=3600,
                           env=dict(os.environ, CARGO_TARGET_DIR=str(target), CUDA_VISIBLE_DEVICES=""))
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


def broken_anchors(tree: Path, mutants: list) -> list:
    """(mutant, why) for every mutant whose find text does not occur exactly once in its file
    under `tree`: the rule run_mutant_in_tree applies before it builds."""
    bad = []
    for m in mutants:
        f = tree / m.file
        if not f.is_file():
            bad.append((m, "no such file"))
            continue
        n = f.read_text().count(m.find)
        if n != 1:
            bad.append((m, f"find text occurs {n} times"))
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, help="output directory (required unless --check-anchors)")
    ap.add_argument("--only", help="comma-separated mutant names")
    ap.add_argument("--check-anchors", action="store_true",
                    help="print the mutants whose find text does not occur exactly once in its file, "
                    "exit 1 if there is one; builds and runs nothing")
    ap.add_argument("--root", type=Path,
                    help="with --check-anchors: the tree whose files are checked (default this checkout)")
    a = ap.parse_args(argv)
    if a.root is not None and not a.check_anchors:
        ap.error("--root needs --check-anchors")
    # Every name --only gives must be a mutant that runs: an unknown or excluded name ran
    # nothing and still exited 0, which hid mutants registered after the main guard. An --only
    # that is given but names nothing ("", ",", " ") chose every mutant, so a run ran them all.
    # No mutant chosen at all fails the same way, in either mode: --check-anchors said "0 of 0
    # anchors broken" and exited 0.
    only = [x for x in (a.only or "").split(",") if x.strip()]
    if a.only is not None and not only:
        print("srcmut: --only names nothing", file=sys.stderr)
        return 2
    names = {m.name for m in MUTANTS}
    bad = [x if x not in EXCLUDED else f"{x} (excluded)" for x in only if x not in names]
    chosen = [m for m in MUTANTS if not only or m.name in only]
    if bad or not chosen:
        why = f"--only names no mutant that runs: {', '.join(bad)}" if bad else "no mutant is chosen"
        print(f"srcmut: {why}", file=sys.stderr)
        return 2
    if a.check_anchors:
        broken = broken_anchors(ROOT if a.root is None else a.root, chosen)
        for m, why in broken:
            print(f"{m.name:14} {m.file}: {why}", flush=True)
        print(f"{len(broken)} of {len(chosen)} anchors broken", file=sys.stderr)
        return 1 if broken else 0
    if a.out is None:
        ap.error("--out is required (except with --check-anchors)")
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    res = []
    for m in chosen:
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

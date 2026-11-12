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
    # the mutant's tree (BIN "lib": the crate's unit tests, `--lib`); the mutant is caught when
    # the test fails (adversarial tests that build their machines by hand,
    # crates/sim/tests/adversarial.rs; claims checked against recorded GPU outputs, e.g. num's
    # error bounds).
    tests: list = field(default_factory=list)
    # Python (directory, file pattern, -k selector): `$PTXV_PYTHON -m unittest discover -s DIR
    # -p PAT -k SEL` in the mutant's tree (tooling outside the binary, e.g. fb/kernel_db's
    # status rules, the python package's readers; PTXV_PYTHON has the package's dependencies,
    # default this interpreter); caught when a test that ran fails.
    py: list = field(default_factory=list)


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
        "        let racer = offenders.iter().find(|(o, _)| !queued(o)).copied();",
        "        let racer = offenders.first().filter(|(o, _)| !queued(o)).copied();",
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
    # E-HW-SPLIT-UR at every level: the strict level drops the switches too (MISS 52's
    # refutation at the strict level must stand: a measured null result is a bound).
    SrcMutant(
        "split-ur-ungated",
        "crates/sim/src/exec.rs",
        "        if self.launch.hw_split_ur && !self.ur_window.is_empty() && self.ur_window_blocks(members, pc)",
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

# S-NC: a non-coherent load (ld.global.nc, LDG .CONSTANT) of bytes written while its kernel
# runs is a violation, also when a PDL dependent reads its primary's output.
NC_PDL = ("ptxv-sim", "fixtures", "e_ld_nc_pdl_overlapping_launches")
MUTANTS += [
    # A PDL dependent counts as stream-ordered after its primary again: its .nc read of the
    # primary's output after griddepcontrol.wait is clean (probes/nc_pdl: GB300 old word 200/200).
    SrcMutant(
        "nc-pdl-ordered",
        "crates/sim/src/exec.rs",
        "                    return !launch.grids[x].pdl;",
        "                    return true;",
        [
            ("e_ld_nc_pdl", v, f, [])
            for v in ("PDLNC", "EARLY")
            for f in ("ptx", "sass")
        ],
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
        '            if mods == ["f32", "bf16"] {',
        '            if to == "f32" && from == "bf16" {',
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
        "            Some((f, 2)) if !matches!(f, Lp::F16 | Lp::Bf16) && rnd == LpRnd::Rn && !sat => {",
        "            Some((f, 2)) if !matches!(f, Lp::F16 | Lp::Bf16) && rnd == LpRnd::Rn => {",
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
        [("ptxv", "nonfinite", "one_ulp_tolerance_never_crosses_an_infinity")],
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
        "            if let Some(p) = profile.filter(|_| wanted2 && gb300 && !spent) {",
        "            if let Some(p) = profile.filter(|_| wanted2 && !spent) {",
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
        [("ptxv", "gpu_compare", "gpu_compare_states_each_comparison")],
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
    # main.rs `run` (check): an unknown hardware profile name is accepted again (the
    # GB300-profile re-run silently off).
    SrcMutant(
        "hwprofile-unknown",
        "crates/cli/src/main.rs",
        '                profile.is_some() || name == "none",',
        "                true,",
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
        "                prog.decision_count(),\n                taken.get(k)",
        "                0,\n                taken.get(k)",
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
    # One thread's adds count as a free order.
    SrcMutant(
        "atom-sources-off",
        "crates/sim/src/errdist/chain.rs",
        "        if self.sources < 2 || !dag.is_finite() || !gpu.is_finite() {",
        "        if !dag.is_finite() || !gpu.is_finite() {",
        [],
        [
            (
                "ptxv-sim",
                "errdist_gpu_gate",
                "atomic_order_explains_only_a_free_order_within_its_bound",
            )
        ],
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

# --pre-round skips only what it changes (errdist.rs `pre_changed`; fixtures/errdist/miss58/rare).
MUTANTS += [
    SrcMutant(
        "preround-bypass",
        "crates/cli/src/errdist.rs",
        "    if let (false, Some(claim)) = (pre_changed, t3_claim) {",
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
        "                vc_join(&mut j, if read { rvc } else { wvc });\n                read\n",
        "                vc_join(&mut j, if read { rvc } else { wvc });\n                false\n",
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
        # "lib": the crate's unit tests (a full path as the name).
        sel = ["--lib"] if binary == "lib" else ["--test", binary]
        cmd = [*wrap, "cargo", "test", "--release", "-p", pkg, *sel, filt]
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
        # Heavy (#[ignore]d) tests run too: a mutant's test is named on purpose.
        sel = ["--lib"] if binary == "lib" else ["--test", binary]
        cmd = [
            *wrap,
            "cargo",
            "test",
            "--release",
            "-p",
            pkg,
            *sel,
            "--",
            "--exact",
            name,
            "--include-ignored",
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

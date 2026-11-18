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
a broken anchor in a scratch tree), or occurs once but only inside a comment (a mutant there
would edit the comment), and exits 1 if there is one (a run reports such a mutant as an error
without building it). Either mode exits 2, before a run creates --out, when --only
is given but names nothing, when it names a name that is not a mutant that runs, or when no
mutant is chosen.

A mutant of a file under fb/ (`SrcMutant.fb_only`: internal tooling) is skipped, in a run and in
--check-anchors, in a tree that has neither the file nor the export manifest
fb/OSS_EXCLUDE.txt: the open-source export, which drops fb/. The count of skipped mutants goes
to stderr; the others are checked or run as usual. In the internal tree a missing fb/ file is
a broken anchor like any other.
"""

import argparse
import json
import os
import re
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

    @property
    def fb_only(self) -> bool:
        """The mutated file is under fb/: internal tooling that the open-source export drops."""
        return self.file.startswith("fb/")


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
        [
            ("e_warpcoll_split", "NOBSYNC", "sass", []),
            # Forum post 74: ptxas's warp-aggregated atomic (one lane at the SHFL, the rest at
            # the BSYNC) deadlocks.
            ("e_shfl_warpagg", "good", "sass", []),
        ],
    ),
    # exec.rs `split_enabled`: the 2026-11-07 rule, a SASS SHFL waiting for every live lane
    # not parked at a BSYNC, another SHFL or a WARPSYNC: the warp-aggregated atomic with the
    # other lanes at a barrier instead of the BSYNC deadlocks (forum post 74).
    SrcMutant(
        "shfl-wait-bsync",
        "crates/sim/src/exec.rs",
        "        if self.k.front == Front::Ptx\n            && (inst.op.is_warp_aligned()",
        "        if self.k.front == Front::Sass && matches!(inst.op, Op::Shfl { .. }) {\n"
        "            let w = self.threads[members[0]].warp;\n"
        "            let at = |u: usize| self.k.code.get(self.threads[u].pc as usize).map(|i| &i.op);\n"
        "            let live = self\n"
        "                .warp_threads(w)\n"
        "                .filter(|&u| {\n"
        "                    let th = &self.threads[u];\n"
        "                    th.st != St::Exited\n"
        "                        && !(!members.contains(&u)\n"
        "                            && (matches!(at(u), Some(Op::Bsync { .. }))\n"
        "                                || th.pc != pc && matches!(at(u), Some(Op::Shfl { .. }))\n"
        "                                || th.st == St::WarpSync))\n"
        "                })\n"
        "                .count();\n"
        "            if members.len() < live {\n"
        "                return false;\n"
        "            }\n"
        "        }\n"
        "        if self.k.front == Front::Ptx\n            && (inst.op.is_warp_aligned()",
        [("e_shfl_warpagg", "NOBSYNC", "sass", [])],
    ),
    # exec.rs `split_enabled`: forum post 74's proposed rule (a local patch): a SASS SHFL
    # waits for the warp's other live lanes unless every one of them is parked at a BSYNC /
    # WARPSYNC or at an aligned BAR at another pc. An early split still waits for lanes on
    # their way and merges with them: MISS 39's wrong-clean (e_warpcoll_split NOBSYNC clean).
    SrcMutant(
        "shfl-wait-parked",
        "crates/sim/src/exec.rs",
        "        if self.k.front == Front::Ptx\n            && (inst.op.is_warp_aligned()",
        "        if self.k.front == Front::Sass && matches!(inst.op, Op::Shfl { .. }) {\n"
        "            let w = self.threads[members[0]].warp;\n"
        "            let live = self\n"
        "                .warp_threads(w)\n"
        "                .filter(|&u| self.threads[u].st != St::Exited)\n"
        "                .count();\n"
        "            let parked = self.warp_threads(w).all(|u| {\n"
        "                let th = &self.threads[u];\n"
        "                members.contains(&u)\n"
        "                    || th.st == St::Exited\n"
        "                    || th.pc != pc\n"
        "                        && self.k.code.get(th.pc as usize).is_some_and(|i| {\n"
        "                            (th.st == St::WarpSync\n"
        "                                && matches!(i.op, Op::Bsync { .. } | Op::WarpSync { .. }))\n"
        "                                || (th.st == St::Run && i.op.is_aligned_bar())\n"
        "                        })\n"
        "            });\n"
        "            if members.len() < live && !parked {\n"
        "                return false;\n"
        "            }\n"
        "        }\n"
        "        if self.k.front == Front::Ptx\n            && (inst.op.is_warp_aligned()",
        [("e_warpcoll_split", "NOBSYNC", "sass", []), ("e_shfl_warpagg", "good", "sass", [])],
    ),
    # exec.rs `exec` (SASS SHFL arm): a source lane that is not executing the SHFL reads the
    # executing lane's own register instead of faulting (a defined value for an undefined read).
    SrcMutant(
        "shfl-undef-src",
        "crates/sim/src/exec.rs",
        "                    let Some(st) = by_lane[src] else {",
        "                    let Some(st) = by_lane[src].or(Some(t)) else {",
        [("e_shfl_warpagg", "XLANE", "sass", [])],
    ),
    # exec.rs `exec_combined` (PTX shfl.sync with a membermask): the same for the PTX front.
    SrcMutant(
        "shfl-undef-src-ptx",
        "crates/sim/src/exec.rs",
        "                    let Some((st, sq)) = by_lane[src] else {",
        "                    let Some((st, sq)) = by_lane[src].or(Some((t, q))) else {",
        [("e_shfl_warpagg", "XLANE", "ptx", []), ("e_warpcoll_mask", "XSRC", "ptx", [])],
    ),
    # exec.rs `sc_note_write` (S-FENCE-SC): writes count whatever their moral strength:
    # CTA-scope atomics of different CTAs (no coherence order between them) force an order.
    SrcMutant(
        "sc-any-write",
        "crates/sim/src/exec.rs",
        "                && self.morally_strong(w.actor, w.strong, w.pc, me, strong, pc)\n            {\n                vc_join(&mut pre, &w.vc);",
        "                && true\n            {\n                vc_join(&mut pre, &w.vc);",
        [],
        # The class stays `race` (the CTA-scope atomics race with each other); the test checks
        # that the partials still race.
        tests=[("ptxv-sim", "fixtures", "fence_sc_publication")],
    ),
    # exec.rs `sc_fence` (S-FENCE-SC): SC fences ordered as executed, without the forcing pair
    # of atomics. Two lanes' fences then order a plain store and load that race in every PTX
    # execution (fixtures/e_fence_sc_unforced: the fence step runs in lane order only).
    SrcMutant(
        "sc-drop-forcing",
        "crates/sim/src/exec.rs",
        "                && vc_get(&seen, f.actor) >= f.own\n",
        "                && (true || vc_get(&seen, f.actor) >= f.own)\n",
        [],
        tests=[("ptxv-sim", "fixtures", "fence_sc_unforced_orders")],
    ),
    # exec.rs `sc_note_write`: the coherence order of two lanes of one converged PTX atom step
    # (lane order only, MISS 140) forces a Fence-SC edge again (fixtures/e_fence_sc_lanes good
    # clean on PTX).
    SrcMutant(
        "sc-lane-step",
        "crates/sim/src/exec.rs",
        "            let lane_step = (w.actor as usize) < self.threads.len()\n",
        "            let lane_step = false && (w.actor as usize) < self.threads.len()\n",
        [],
        tests=[("ptxv-sim", "fixtures", "fence_sc_unforced_orders")],
    ),
    # exec.rs `sc_note_write`: the lane-step exclusion keyed on the instruction again (round 1):
    # a lane-step write still orders a later write of another lane at another pc, and lanes of
    # a converged warp at consecutive instructions run in step order only
    # (fixtures/e_fence_sc_lanes INDIRECT / PCS clean on PTX, PCS certified).
    SrcMutant(
        "sc-lane-pc",
        "crates/sim/src/exec.rs",
        "            let lane_step = (w.actor as usize) < self.threads.len()\n",
        "            let lane_step = w.pc == pc && (w.actor as usize) < self.threads.len()\n",
        [],
        tests=[("ptxv-sim", "fixtures", "fence_sc_unforced_orders")],
    ),
    # exec.rs `sc_note_write`: claims on a declared C-TICKET-ADDR counter are logged in the
    # certificate's runs, so the run's claim order creates a Fence-SC edge
    # (fixtures/e_ticket_addr_fence SC certified on PTX).
    SrcMutant(
        "sc-ticket-addr",
        "crates/sim/src/exec.rs",
        "        if (self.wlog.is_on() || self.perm.is_some())\n            && self.launch.perm.iter().any(|p| p.0 as u32 == key.1)\n",
        "        if false\n            && self.launch.perm.iter().any(|p| p.0 as u32 == key.1)\n",
        [],
        tests=[("ptxv-sim", "fixtures", "e_ticket_addr_fence_acquired_claims_refused")],
    ),
    # exec.rs `sc_fence_scope`: a SASS MEMBAR.SC counts as an SC fence without the CCTL.IVALL
    # that is its acquire at the strict SASS level (fixtures/e_fence_sc_publish NOCCTL clean).
    SrcMutant(
        "sc-membar-acq",
        "crates/sim/src/exec.rs",
        "        ) if i.text.contains(\"MEMBAR.SC.\") && sass_sc_acquires(k, pc, i.guard) => Some(*s),",
        "        ) if i.text.contains(\"MEMBAR.SC.\") => Some(*s),",
        [],
        tests=[("ptxv-sim", "fixtures", "fence_sc_publication")],
    ),
    # exec.rs `sc_note_write`: atomics of different widths force a Fence-SC order (on SASS
    # `same_location` holds for any two accesses at one address: fixtures/e_fence_sc_mixed
    # good clean on SASS).
    SrcMutant(
        "sc-width",
        "crates/sim/src/exec.rs",
        "                && atom_width(&self.k, w.pc) == width\n",
        "",
        [],
        tests=[("ptxv-sim", "fixtures", "fence_sc_unforced_orders")],
    ),
    # exec.rs `sync_observe_cell` (MISS 140): a lane synchronizes with what a lane of its warp
    # wrote at the same PTX instruction in lane order (fixtures/e_atom_lane_order good clean).
    SrcMutant(
        "obs-lane-step",
        "crates/sim/src/exec.rs",
        "        let lane = c.pc == pc\n",
        "        let lane = false && c.pc == pc\n",
        [],
        tests=[("ptxv-sim", "fixtures", "atom_lane_order_edges")],
    ),
    # exec.rs `sync_observe_cell` (MISS 140 precision): the lane-step rule drops release
    # sequences continued from writers outside the warp too (fixtures/e_atom_lane_cont good: a
    # false race).
    SrcMutant(
        "obs-lane-cont",
        "crates/sim/src/exec.rs",
        "            .filter(|e| cont_in(self, e) && !(lane && e.writer != me && same_warp(self, e.writer)))\n",
        "            .filter(|e| cont_in(self, e) && !lane)\n",
        [],
        tests=[("ptxv-sim", "fixtures", "atom_lane_order_edges")],
    ),
    # exec.rs `sync_observe_cell` (MISS 140): the lane-step rule keeps release sequences
    # continued from other lanes of the warp (fixtures/e_atom_lane_order THREE: lane 2 reads
    # lane 1's cell, which continued lane 0's sequence; clean on PTX).
    SrcMutant(
        "obs-lane-samewarp",
        "crates/sim/src/exec.rs",
        "            .filter(|e| cont_in(self, e) && !(lane && e.writer != me && same_warp(self, e.writer)))\n",
        "            .filter(|e| cont_in(self, e))\n",
        [],
        tests=[("ptxv-sim", "fixtures", "atom_lane_order_edges")],
    ),
    # wellsync/ticket.rs (MISS 140): C-TICKET accepts claims by lanes of one warp that
    # happen-before does not order and drops the other order as infeasible
    # (fixtures/e_atom_lane_order NOREAD certified).
    SrcMutant(
        "ticket-warp",
        "crates/sim/src/wellsync/ticket.rs",
        "                    if p.ev.actor != t && !p.ev.hb(t, a.ev.seq, &a.vc) {\n",
        "                    if false && p.ev.actor != t && !p.ev.hb(t, a.ev.seq, &a.vc) {\n",
        [],
        tests=[
            ("ptxv-sim", "fixtures", "atom_lane_order_edges"),
            ("ptxv-sim", "fixtures", "ticket_warp_check_order_created_edges"),
        ],
    ),
    # wellsync/ticket.rs (MISS 140, MISS 187): the warp check only for claims at one pc (the
    # round-2 one-step scope). Lanes of a converged warp at different instructions run in step
    # order only, so the other order is dropped as infeasible (fixtures/e_fence_sc_lanes
    # PCS_READER0 certified on PTX; e_ticket_warp_chain / e_ticket_warp_sc_chain refused without
    # the warp check).
    SrcMutant(
        "ticket-samepc",
        "crates/sim/src/wellsync/ticket.rs",
        "                    if p.ev.actor != t && !p.ev.hb(t, a.ev.seq, &a.vc) {\n",
        "                    if p.ev.actor != t && p.ev.pc == a.ev.pc && !p.ev.hb(t, a.ev.seq, &a.vc) {\n",
        [],
        tests=[
            ("ptxv-sim", "fixtures", "fence_sc_unforced_orders"),
            ("ptxv-sim", "fixtures", "ticket_warp_check_order_created_edges"),
        ],
    ),
    # exec.rs `warpsync_release`: the GB300 profile's BSYNC waits for lanes parked at a
    # WARPSYNC.COLLECTIVE again (E-HW-BSYNC-COLLECTIVE). Survives since MISS 81 and gap G1: no
    # BSYNC that still holds has such a lane in its own region, and a lane blocked at a
    # collective opens any other BSYNC (E-HW-BSYNC-YIELD); ASSUMPTIONS.md E-HW-BSYNC-COLLECTIVE.
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
    # spin `blocked_wpoll_read`: a blocked warp-level poll's reads without the mapping their
    # dry iteration resolved (S-VA-ALIAS; every such read was `unmodeled`).
    SrcMutant(
        "wpoll-view",
        "crates/sim/src/spin.rs",
        "        if let (Some(v), crate::exec::Loc::Global { alloc, off }) = (view, loc) {\n"
        "            crate::exec::va_note(t, alloc, off, bytes as u64, v);\n        }\n",
        "        let _ = view;\n",
        [],
        tests=[("ptxv-sim", "fixtures", "r1h_va_wpoll")],
    ),
    # sass/elf.rs `bank`: R_CUDA type 35 (a function descriptor) refused again.
    SrcMutant(
        "desc-reloc",
        "crates/sass/src/elf.rs",
        "if typ != 2 && typ != R_CUDA_DESCRIPTOR {",
        "if typ != 2 {",
        [],
        tests=[("ptxv-sim", "fixtures", "s_func_desc")],
    ),
    # ptx `parse_kernel_for`: an entry prototype taken as the kernel again.
    SrcMutant(
        "entry-proto",
        "crates/ptx/src/lib.rs",
        "            if src[rp..].find('{').is_none_or(|b| semi < b) {",
        "            if false && src[rp..].find('{').is_none_or(|b| semi < b) {",
        [],
        tests=[("ptxv-ptx", "lib", "tests::entry_prototype_before_definition")],
    ),
    # exec `exit_checks`: a remote delivery drained by its issuer's later release loses the
    # target's wait as cover again (S-CLUSTER-LIFE false violation).
    SrcMutant(
        "drain-wait",
        "crates/sim/src/exec.rs",
        "            if !waited && !self.observed_by(cta, issuer, clock) {",
        "            if !self.observed_by(cta, issuer, clock) {",
        [],
        tests=[("ptxv-sim", "fixtures", "e_stasync_drain")],
    ),
    # exec `mbar_reset_seen`: the waits recorded on an mbarrier survive its inval / init again
    # (MISS 203: a wait on the old object covers a delivery into the new one; --certify clean).
    SrcMutant(
        "mbar-reinit-seen",
        "crates/sim/src/exec.rs",
        "        if let Some(s) = self.mbar_seen.remove(&(cta, off)) {\n",
        "        if let Some(s) = self.mbar_seen.get(&(cta, off)).copied().filter(|_| false) {\n",
        [
            ("e_mbar_reinit_drain", "good", "ptx", ["--certify"]),
            ("e_mbar_reinit_drain", "good", "sass", ["--certify"]),
            ("e_mbar_reinit_drain", "NOFENCE", "ptx", ["--certify"]),
            ("e_mbar_reinit_drain", "NOFENCE", "sass", ["--certify"]),
        ],
    ),
    # The reset at mbarrier.init only (NOINV: init over the live object, no inval).
    SrcMutant(
        "mbar-reinit-seen-init",
        "crates/sim/src/exec.rs",
        "        self.mbar_reset_seen(cta, off);\n        self.mbars.insert(\n",
        "        self.mbars.insert(\n",
        [
            ("e_mbar_reinit_drain", "NOINV", "ptx", ["--certify"]),
            ("e_mbar_reinit_drain", "NOINV", "sass", ["--certify"]),
        ],
    ),
    # exec `store_over_mbar`: an STS.64 state word over a live mbarrier keeps the old object
    # again (MISS 218: SASS misses the race / deadlock the PTX front reports).
    SrcMutant(
        "mbar-sts-live-reinit",
        "crates/sim/src/exec.rs",
        "            if self.k.front == Front::Sass && o == off && n == 8 {\n",
        "            if false && self.k.front == Front::Sass && o == off && n == 8 {\n",
        [
            ("e_mbar_sts_live", "good", "sass", []),
            ("e_mbar_sts_live", "CNT2", "sass", ["--certify"]),
        ],
    ),
    # ir `Op::is_visible`: mbarrier.inval runs eagerly inside a step again (MISS 242: an
    # arrival unordered with the inval is explored in one order only).
    SrcMutant(
        "mbar-inval-visible",
        "crates/ir/src/lib.rs",
        "                | Op::MbarInval { .. }\n                | Op::MbarArrive { .. }\n",
        "                | Op::MbarArrive { .. }\n",
        [
            ("e_mbar_store_order", "ARRVS", "sass", []),
        ],
    ),
    # exec `store_over_mbar`: a store not ordered after the inval of the object is not checked
    # (MISS 241: --certify runs the inval first and calls SWAP clean).
    SrcMutant(
        "mbar-dead-store",
        "crates/sim/src/exec.rs",
        "                if d.hb(&svc) || self.mbars.contains_key(&(cta, o)) {\n",
        "                if true || d.hb(&svc) || self.mbars.contains_key(&(cta, o)) {\n",
        [
            ("e_mbar_store_order", "SWAP", "ptx", ["--certify"]),
            ("e_mbar_store_order", "SWAP", "sass", ["--certify"]),
        ],
    ),
    # exec `mbar_init`: an init with no live object does not check the bytes' last plain write
    # (MISS 241: the store run before the first init, PREU on SASS, is clean).
    SrcMutant(
        "mbar-init-store-unordered",
        "crates/sim/src/exec.rs",
        "                    .find(|e| e.actor != t as u32 && !e.hb(&svc))\n",
        "                    .find(|e| false && e.actor != t as u32 && !e.hb(&svc))\n",
        [
            ("e_mbar_store_order", "PREU", "sass", []),
            ("e_mbar_store_order", "PREU", "sass", ["--certify"]),
        ],
    ),
    # exec `mbar_init`: an init not ordered after the inval of the previous object is not
    # checked against it (MISS 241 / 242: the certificate's run takes REINITUS's inval first
    # and reports nothing).
    SrcMutant(
        "mbar-dead-init",
        "crates/sim/src/exec.rs",
        "                if d.actor != t as u32 && !d.hb(&svc) {\n",
        "                if false && d.actor != t as u32 && !d.hb(&svc) {\n",
        [
            ("e_mbar_store_order", "REINITUS", "ptx", ["--certify"]),
            ("e_mbar_store_order", "REINITUS", "sass", ["--certify"]),
        ],
    ),
    # exec `mbar_init`: the last write of each byte only, not the earlier ones the shadow keeps
    # in `hw` (RMK4 F1: a plain store then a strong one of the same thread, unordered with the
    # init, was clean).
    SrcMutant(
        "mbar-init-hidden-write",
        "crates/sim/src/exec.rs",
        "                    .chain(sh.hw.iter().flat_map(|h| h.iter().copied()))\n",
        "",
        [
            ("e_mbar_hidden_write", "SWAP", "sass", []),
            ("e_mbar_hidden_write", "STRONGS", "sass", []),
            ("e_mbar_hidden_write", "STRONGS", "sass", ["--certify"]),
        ],
    ),
    # exec `mbar_init`: the full clock instead of the shared-memory-only one (RMK4 F7: an inval
    # ordered before the init only through shared memory was a race).
    SrcMutant(
        "mbar-init-restrict",
        "crates/sim/src/exec.rs",
        "            let svc = self.sh_vc(t as u32, &vc, cta).unwrap_or_else(|| vc.clone());\n            if let Some(d)",
        "            let svc = vc.clone();\n            if let Some(d)",
        [
            ("e_mbar_init_restrict", "good", "ptx", []),
            ("e_mbar_init_restrict", "INITP", "ptx", []),
            ("e_mbar_init_restrict", "INITP", "ptx", ["--certify"]),
        ],
    ),
    # exec `mbar_atomic`: an mbarrier update does not read the object's (re)initialization key
    # (RMK4 F10: the synchronization reduction ran expect_tx / the SASS A0TR arrive before the
    # inval only, EXPTX / EXPTXS clean).
    SrcMutant(
        "mbar-inval-key-atomic",
        "crates/sim/src/exec.rs",
        '        // the update first only (MISS 241 follow-up, fixtures/e_mbar_inval_vis EXPTX).\n        self.rec_sync(\n            Resource {\n                space: ResourceSpace::MbarInitialization,\n                owner: cta as u32,\n                start: off as u64,\n                end: off as u64 + 1,\n            },\n            false,\n        );\n        self.mbar_materialize(cta, off);\n',
        '        // the update first only (MISS 241 follow-up, fixtures/e_mbar_inval_vis EXPTX).\n        self.mbar_materialize(cta, off);\n',
        [
            ("e_mbar_inval_vis", "EXPTX", "ptx", []),
            ("e_mbar_inval_vis", "EXPTX", "sass", []),
            ("e_mbar_inval_vis", "EXPTXS", "sass", []),
            ("e_mbar_inval_vis", "ARRTXS", "sass", []),
        ],
    ),
    # exec E-LOCKSTEP: no lane join after an inval / init (RMK4 F8: a lane's store after its
    # converged warp's SYNCS.CCTL.IV was "not ordered" with it).
    SrcMutant(
        "mbar-lockstep-inval",
        "crates/sim/src/exec.rs",
        "                            | Op::MbarInval { .. }\n                            | Op::MbarInit { .. }\n                    ))\n",
        "                    ))\n",
        [
            ("e_mbar_lane_inval", "LANESYNC", "sass", []),
            ("e_mbar_lane_inval", "LANESYNC", "sass", ["--certify"]),
        ],
    ),
    # explore.rs `needs_plain_rerun`: races only (RMK4 F4: a reuse ordered by st.release /
    # ld.acquire after the inval was a violation without the synchronization reduction, which
    # st.shared::cluster turns off).
    SrcMutant(
        "mbar-store-plain-rerun",
        "crates/sim/src/explore.rs",
        '    f.kind == "race" || f.sig.starts_with("mbar-store:")\n',
        '    f.kind == "race"\n',
        [
            ("e_mbar_reuse_ordered", "CL", "ptx", []),
            ("e_mbar_reuse_ordered", "CL", "sass", []),
        ],
    ),
    # exec `store_over_mbar`: a SASS state word over an invalidated object, not ordered after the
    # inval, is a store over a live mbarrier (violation) instead of an init racing with the inval
    # (RMK4). Reached only where the store cannot run before the inval: the default explorer
    # runs a plain store that can before it (E-DRF), and the init in that order races with the
    # inval (REINITS); FLAG's store waits for a relaxed flag set after the inval.
    SrcMutant(
        "rmk4-dead-sass-word",
        "crates/sim/src/exec.rs",
        "                    && n == 8\n                    && self.mbar_decode_mem(cta, o).is_some()\n",
        "                    && n == 8 && false\n                    && self.mbar_decode_mem(cta, o).is_some()\n",
        [
            ("e_mbar_reinits", "FLAG", "sass", []),
            ("e_mbar_reinits", "FLAG", "sass", ["--certify"]),
        ],
    ),
    # exec `mbar_init`: the low 4 bytes' writes only (RMK4).
    SrcMutant(
        "rmk4-init-bytes",
        "crates/sim/src/exec.rs",
        "        let w = (0..8u32).find_map(|i| {\n",
        "        let w = (0..4u32).find_map(|i| {\n",
        [
            ("e_mbar_partial_store", "PREU_W4", "sass", []),
            ("e_mbar_partial_store", "PREU_W4", "sass", ["--certify"]),
            ("e_mbar_partial_store", "PREU_W1", "sass", []),
        ],
    ),
    # exec `store_over_mbar`: the dead-object check only for a store at the object's offset (RMK4).
    SrcMutant(
        "rmk4-dead-range",
        "crates/sim/src/exec.rs",
        "            .filter(|&(o, d)| o.saturating_add(8) > off && d.actor != t as u32)\n",
        "            .filter(|&(o, d)| o == off && d.actor != t as u32)\n",
        [
            ("e_mbar_partial_store", "DEADS_W4", "sass", []),
            ("e_mbar_partial_store", "DEADS_W4", "sass", ["--certify"]),
            ("e_mbar_partial_store", "DEADS_W1", "sass", []),
        ],
    ),
    # explore.rs `mbar_init_pcs`: MbarInit pcs only again (RMK3's Q3: forget_dead_sync drops
    # the arrival before the SASS STS.64 re-init).
    SrcMutant(
        "mbar-sts-init-pcs",
        "crates/sim/src/explore.rs",
        "            } => k.front == ptxv_ir::Front::Sass,\n",
        "            } => false && k.front == ptxv_ir::Front::Sass,\n",
        [
            ("e_mbar_forget_initpcs", "good", "sass", []),
            ("e_mbar_forget_initpcs", "good", "sass", ["--certify"]),
        ],
    ),
    # exec `store_over_mbar`: every lane of one STS.64 re-initializes the object again (each
    # lane's init then races with the previous lane's: WARP on SASS reports a race).
    SrcMutant(
        "mbar-sts-live-same",
        "crates/sim/src/exec.rs",
        "                    let same = old.atomics.is_empty()\n",
        "                    let same = false && old.atomics.is_empty()\n",
        [
            ("e_mbar_sts_live", "WARP", "sass", []),
        ],
    ),
    # exec `St`: no check of a plain store against the live mbarriers (MISS 218: a plain store
    # over a live mbarrier is clean on both fronts).
    SrcMutant(
        "mbar-sts-live-hook",
        "crates/sim/src/exec.rs",
        "                            self.store_over_mbar(t, pc, cta, off, *bytes as u32);\n",
        "",
        [
            ("e_mbar_sts_live", "PLAIN", "ptx", []),
            ("e_mbar_sts_live", "ZERO", "sass", ["--certify"]),
            ("e_mbar_sts_live", "CNT2PLAIN", "sass", []),
        ],
    ),
    # The reset at mbarrier.inval only (STS on SASS: SYNCS.CCTL.IV, then an STS.64 init word,
    # which materializes the new object without an MbarInit).
    SrcMutant(
        "mbar-reinit-seen-inval",
        "crates/sim/src/exec.rs",
        "                self.mbar_reset_seen(cta, off);\n                self.mbars.remove(&(cta, off));\n",
        "                self.mbars.remove(&(cta, off));\n",
        [
            ("e_mbar_reinit_drain", "STS", "sass", ["--certify"]),
        ],
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
    # E-ELECT taint: a collective of elect's predicate (a ballot, shfl.idx from a fixed lane)
    # is warp-uniform again (MISS 71, 79: it names the leader's lane).
    SrcMutant(
        "elect-coll-pattern",
        "crates/sim/src/explore.rs",
        "if collective && (pat_ls || ctl_ls) && !matches!(inst.op, Op::VoteAll { .. }) {",
        "if false {",
        [
            ("e_elect_ballot", "good", "ptx", []),
            ("e_elect_ballot", "POPC", "ptx", []),
            ("e_elect_shfl", "good", "ptx", []),
        ],
    ),
    # E-ELECT taint: a ballot under lane-varying control is warp-uniform again (MISS 72's
    # family: the lanes that skip it keep their old values).
    SrcMutant(
        "elect-coll-ctl",
        "crates/sim/src/explore.rs",
        "Op::VoteAny { .. } | Op::VoteAll { .. } | Op::Redux { .. } => out_lv = ctl_lv,",
        "Op::VoteAny { .. } | Op::VoteAll { .. } | Op::Redux { .. } => out_lv = false,",
        [("e_elect_tid_ctl", "VOTE", "ptx", [])],
    ),
    # E-ELECT taint: `%tid.x >> 5` is warp-uniform under lane-varying control again (MISS 72).
    SrcMutant(
        "elect-tid-ctl",
        "crates/sim/src/explore.rs",
        "} => rows && *sh >= 5 && tidx.contains(r) && inst.guard.reg.is_none() && !rlv[pc],",
        "} => rows && *sh >= 5 && tidx.contains(r) && inst.guard.reg.is_none(),",
        [("e_elect_tid_ctl", "good", "ptx", [])],
    ),
    # E-CLOCK taint (clock.rs `succs`): a RET has no successor again, so a clock value returned
    # by a CALL.REL callee is clock-free in the caller (MISS 70).
    SrcMutant(
        "clock-ret-sites",
        "crates/sim/src/clock.rs",
        "            let mut v = rets.to_vec();",
        "            let mut v = Vec::new();",
        [("e_clock_callee", "good", "sass", [])],
    ),
    # E-CLOCK: a clock choice of several lanes on a lane-varying predicate is not reported
    # (MISS 76: one side for the whole split).
    SrcMutant(
        "clock-lanes-off",
        "crates/sim/src/exec.rs",
        "            if members.len() > 1 && self.clock_lane_pcs.contains(&pc) {",
        "            if false {",
        [("e_clock_lanes", "good", "ptx", []), ("e_clock_lanes", "good", "sass", [])],
    ),
    # E-CLOCK: an indexed branch's clock-derived index is ignored again (MISS 77).
    SrcMutant(
        "clock-brx-index",
        "crates/sim/src/clock.rs",
        "                let idx = matches!(i.op, Op::Brx { .. })",
        "                let idx = false && matches!(i.op, Op::Brx { .. })",
        [("e_clock_brx", "good", "ptx", []), ("e_clock_brx", "good", "sass", [])],
    ),
    # E-CLOCK: every hang guard is exempt again, whatever its bound (MISS 78).
    SrcMutant(
        "clock-hang-any",
        "crates/sim/src/machine.rs",
        "                    if m.prd(t, g) == sl.trap_when {",
        "                    if false {",
        [("e_clock_hang", "good", "ptx", []), ("e_clock_hang", "good", "sass", [])],
    ),
    # E-CLOCK: a hang-guard bound of 1024 clock units is enough (the 2000-cycle guard exempt).
    # E-CLOCK: a hang guard's AND / OR combine peeled on both sides, also where the trap side
    # does not force the operand (OR true, AND false: the short compare alone traps).
    SrcMutant(
        "clock-hang-peel",
        "crates/sim/src/clock.rs",
        "        if *or == v {",
        "        if false {",
        [
            ("e_clock_hang_or", "good", "ptx", []),
            ("e_clock_hang_or", "good", "sass", []),
            ("e_clock_hang_or", "ANDF", "ptx", []),
            ("e_clock_hang_or", "ANDF", "sass", []),
        ],
    ),
    # E-CLOCK: a register with no definition on some path from the entry has its one
    # definition as the only reaching one (DL: the deadline computed on one path only; in
    # good the clock reads' dominance rule refuses the skipped start read as well).
    SrcMutant(
        "clock-hang-undef",
        "crates/sim/src/clock.rs",
        """        if q == entry {
            return Vec::new();
        }
        work.extend(preds[q].iter().copied());""",
        """        work.extend(preds[q].iter().copied());""",
        [
            ("e_clock_hang_undef", "good", "ptx", []),
            ("e_clock_hang_undef", "DL", "ptx", []),
            ("e_clock_hang_undef", "DL", "sass", []),
        ],
    ),
    SrcMutant(
        "clock-hang-threshold",
        "crates/sim/src/clock.rs",
        "pub const HANG_BOUND: u64 = 1 << 31;",
        "pub const HANG_BOUND: u64 = 1 << 10;",
        [("e_clock_hang", "good", "ptx", []), ("e_clock_hang", "good", "sass", [])],
    ),
    # Op::dst_regs: the SASS 64-bit clock read writes its low register only again (the taint
    # misses the high half).
    SrcMutant(
        "clock-pair-hi",
        "crates/ir/src/lib.rs",
        "                Val::SReg(SReg::Clock) if d.file == File::R => span(*d, 2),",
        "                Val::SReg(SReg::Clock) if d.file == File::R => span(*d, 1),",
        [("e_clock_hi", "good", "sass", [])],
    ),
    # E-HW-SB-QUEUE: the GB300 queue rules apply at the strict level too (no `sb-hwq:`
    # contract findings: REDUX in order on a measured null).
    SrcMutant(
        "sb-hwq-strict",
        "crates/sass/src/scoreboard.rs",
        "    let strict = check_rules(lines, code, sm, false, false)?;",
        "    let strict = check_rules(lines, code, sm, true, false)?;",
        [("../probes/redux_queue", "COVER", "sass", [])],
    ),
    # L-OPCLASS: FRND back in the conversion queue (refuted on GB300 by probes/frnd_queue).
    SrcMutant(
        "frnd-xu",
        "crates/sass/src/scoreboard.rs",
        '        "POPC" | "FLO" if hw => "bit",',
        '        "MUFU" | "FRND" => "xu",\n        "POPC" | "FLO" if hw => "bit",',
        [("../probes/frnd_queue", "COVER", "sass", [])],
    ),
    # E-CLOCK hang guards: starts with low 32 bits 0 only (a wrapping deadline is "proven").
    SrcMutant(
        "clock-hang-starts",
        "crates/sim/src/machine.rs",
        "                starts.push(hi | lo);",
        "                starts.push(hi);",
        [("e_clock_hang_wrap", "good", "ptx", []), ("e_clock_hang_wrap", "good", "sass", [])],
    ),
    # elect_lanes::Cfg: an indexed branch falls through again (no lane-varying region).
    SrcMutant(
        "cfg-brx",
        "crates/sim/src/elect_lanes.rs",
        "                    Op::Brx { targets, .. } => targets.clone(),",
        "                    Op::Brx { .. } => vec![pc + 1],",
        [("e_elect_tid_brx", "good", "ptx", []), ("e_clock_lanes_brx", "good", "ptx", [])],
    ),
    # E-ELECT taint: an indexed branch's index starts no region.
    SrcMutant(
        "elect-brx-idx",
        "crates/sim/src/explore.rs",
        "                    Op::Brx { .. } => inst.op.src_regs(),",
        "                    Op::Brx { .. } => Vec::new(),",
        [("e_elect_tid_brx", "good", "ptx", [])],
    ),
    # E-CLOCK: an indexed branch on a loaded clock value is no effect.
    SrcMutant(
        "clock-brx-mem",
        "crates/sim/src/clock.rs",
        "    let mt = if clock_stores(k).is_empty() {",
        "    let mt = if true {",
        [("e_clock_brx_mem", "good", "ptx", []), ("e_clock_brx_mem", "good", "sass", [])],
    ),
    # L-OPREAD (MISS 101): a TMA op's bracketed base register is its only read again (since MISS
    # 181 the whole uniform group is the lift's: `def_use_lifted` drops the lift's URs). Also
    # MISS 181 H1: UR pairs one register again.
    SrcMutant(
        "tma-ur-group",
        "crates/sass/src/scoreboard.rs",
        "        File::UR if r.idx < crate::URZ => Some((b'U', r.idx)),\n",
        "        File::UR if false => Some((b'U', r.idx)),\n",
        [("../probes/utmastg_queue", "COVER", "sass", [])],
        tests=[("ptxv-sass", "lib", "lift_widths_are_tracked"), ("ptxv-sim", "fixtures", "miss181_lift_width_repros")],
    ),
    # E-HW-SB-QUEUE (review item 6): GB300-measured queue rules at the strict level again.
    SrcMutant(
        "f64-strict",
        "crates/sass/src/scoreboard.rs",
        '        "DADD" | "DMUL" | "DFMA" | "DSETP" if hw => "f64",',
        '        "DADD" | "DMUL" | "DFMA" | "DSETP" => "f64",',
        # A DADD with no write scoreboard covered by a later f64 op's (probes/f64_queue's
        # strict contract is an operand-read one: opread-strict).
        [("../probes/f64_dsetp", "good", "sass", [])],
    ),
    SrcMutant(
        "popc-strict",
        "crates/sass/src/scoreboard.rs",
        '        "POPC" | "FLO" if hw => "bit",',
        '        "POPC" | "FLO" => "bit",',
        [("../probes/flo_popc", "good", "sass", [])],
    ),
    SrcMutant(
        "frnd-strict",
        "crates/sass/src/scoreboard.rs",
        '            ) || hw && matches!(base, "REDUX" | "FRND")',
        '            ) || base == "FRND" || hw && base == "REDUX"',
        [("../probes/frnd_queue", "good", "sass", [])],
    ),
    SrcMutant(
        "opread-strict",
        "crates/sass/src/scoreboard.rs",
        "    if !hw {\n        return None;\n    }\n    let q = inorder_queue(opcode, true, broad);",
        "    let q = inorder_queue(opcode, true, broad);",
        [("../probes/xu_war", "rd_sb", "sass", [])],
    ),
    # L-OPCLASS (MISS 96): MUFU / F2I / I2F / F2F one queue again (refuted on GB300 by
    # probes/xu_queue COVER).
    SrcMutant(
        "xu-mufu",
        "crates/sass/src/scoreboard.rs",
        '        "POPC" | "FLO" if hw => "bit",',
        '        "MUFU" | "F2I" | "I2F" | "F2F" => "xu",\n        "POPC" | "FLO" if hw => "bit",',
        [
            ("../probes/xu_queue", "COVER", "sass", []),
            ("../probes/xu_queue_rev", "COVER", "sass", []),
        ],
    ),
    # L-OPCLASS (MISS 106): an earlier SHFL covered by a later LDS's write scoreboard again
    # (refuted on GB300 by probes/shared_queue REV_COVER).
    SrcMutant(
        "shfl-shared",
        "crates/sass/src/scoreboard.rs",
        "                            && (!shfl(&lines[k.2].opcode) || shfl(base))\n",
        "",
        [("../probes/shared_queue", "REV_COVER", "sass", [])],
    ),
    # E-HW-SB-QUEUE: the shared-memory queue across opcodes at the strict level again.
    SrcMutant(
        "shared-strict",
        "crates/sass/src/scoreboard.rs",
        '        "LDS" | "SHFL" if hw => "shared",',
        '        "LDS" | "SHFL" => "shared",',
        [("../probes/shared_queue", "COVER", "sass", [])],
        # COVER's strict verdict also carries the STS write cover's contract (gb300 only):
        # the test asserts the LDS contract itself.
        tests=[("ptxv-sim", "fixtures", "probe_shared_queue")],
    ),
    # E-HW-SB-QUEUE (review item 4): untracked reads attach to another opcode's read scoreboard
    # at the strict level again (sb_mio_order rd_on_ldg is a strict contract).
    SrcMutant(
        "att-cross-strict",
        "crates/sass/src/scoreboard.rs",
        '                            || !matches!(ub, "UTMASTG" | "utc") && kb == base\n',
        '                            || !matches!(ub, "UTMASTG" | "utc")\n',
        [("../probes/sb_mio_order/fixture", "rd_on_ldg", "sass", [])],
    ),
    # The same-opcode exception: same-opcode read attachment refused at the strict level too
    # (sb_mio_order inorder, GB300 0 / 1000 wrong, is strict clean).
    SrcMutant(
        "att-same-gated",
        "crates/sass/src/scoreboard.rs",
        '                            || !matches!(ub, "UTMASTG" | "utc") && kb == base\n',
        '                            || hw && !matches!(ub, "UTMASTG" | "utc") && kb == base\n',
        [("../probes/sb_mio_order/fixture", "inorder", "sass", [])],
    ),
    # E-HW-SB-QUEUE: a memory write's completion covers another opcode's earlier reads at the
    # strict level again (sb_mio_order wcov is a strict contract).
    SrcMutant(
        "wcov-strict",
        "crates/sass/src/scoreboard.rs",
        "                    if !k.1 && MIO_INORDER.contains(&kb) && (kb == base || broad || hw && measured())\n",
        "                    if !k.1 && MIO_INORDER.contains(&kb)\n",
        [("../probes/sb_mio_order/fixture", "wcov", "sass", [])],
    ),
    # E-HW-SB-QUEUE: a load writes after another opcode's earlier reads at the strict level
    # again (sb_mio_order qwar_lds is a strict contract).
    SrcMutant(
        "qwar-strict",
        "crates/sass/src/scoreboard.rs",
        "                    && (base_of(pc) == base_of(ppc)\n",
        "                    && (true\n",
        [("../probes/sb_mio_order/fixture", "qwar_lds", "sass", [])],
    ),
    # MISS 128: a SHFL clears earlier untracked memory-queue reads again (refuted on GB300 by
    # probes/sb_mio_order rvs_shfl_retain).
    SrcMutant(
        "shfl-retain",
        "crates/sass/src/scoreboard.rs",
        "        let cguard = l.guard.as_deref();\n",
        "        if l.opcode.split('.').next() == Some(\"SHFL\") {\n"
        "            st.retain(|k, _| k.0 != UNTRACKED || !MIO_INORDER.contains(&lines[k.2].opcode.split('.').next().unwrap_or(\"\")));\n"
        "        }\n"
        "        let cguard = l.guard.as_deref();\n",
        [("../probes/sb_mio_order/fixture", "rvs_shfl_retain", "sass", [])],
    ),
    # MISS 129: a SHFL reads an earlier unscoreboarded LDS result in the queue again (refuted by
    # probes/shared_queue RVS_QREAD_LDS_S4).
    SrcMutant(
        "shfl-queued-read",
        "crates/sass/src/scoreboard.rs",
        "                let kind = if write && uses.contains(&r) {",
        "                let kind = if write && uses.contains(&r) && !(sb == NO_SB && l.opcode.split('.').next() == Some(\"SHFL\") && lines[ppc].opcode.split('.').next() == Some(\"LDS\")) {",
        [("../probes/shared_queue", "RVS_QREAD_LDS_S4", "sass", [])],
    ),
    # MISS 130: an earlier 64-bit-integer conversion covered by a later 32-bit one again.
    SrcMutant(
        "conv-width",
        "crates/sass/src/scoreboard.rs",
        "    if w64(earlier) && !w64(later) {\n        return false;\n    }\n",
        "",
        [("../probes/cvt64_queue", "F2I64_COVER_S4", "sass", [])],
    ),
    # MISS 131: UBLKCP's operand reads untracked again.
    SrcMutant(
        "ublkcp-async",
        "crates/sass/src/scoreboard.rs",
        '    "UBLKCP",\n',
        "",
        [("../probes/bulk_s2s", "NORB", "sass", [])],
    ),
    # Narrowing: unmeasured conversion pairs (ptxas's I2F.U32 covered by a later I2F.S64) as
    # measured again: probes/cvt64_queue good is an sb-unmeasured: contract.
    SrcMutant(
        "conv-unmeasured",
        "crates/sass/src/scoreboard.rs",
        "    earlier == later\n        || broad\n",
        "    earlier == later\n        || true\n",
        [("../probes/cvt64_queue", "good", "sass", [])],
        # good's strict classes are a contract either way (sb-hwq: or sb-unmeasured:): the test
        # asserts the sb-unmeasured: one.
        tests=[("ptxv-sim", "fixtures", "probe_cvt64_queue")],
    ),
    # E-HW-SB-QUEUE (check levels): a hazard every level finds is labelled an sb-unmeasured:
    # contract again, so a read no order covers is no violation (probes/i2f8_queue NOWAIT).
    # (Retargeted at the per-hazard level merge, RVS2 P1: its old anchor no longer existed.)
    SrcMutant(
        "broad-kind",
        "crates/sass/src/scoreboard.rs",
        "        } else if !broad.contains_key(&k) {\n",
        "        } else if true {\n",
        [("../probes/i2f8_queue", "NOWAIT_S4", "sass", [])],
        tests=[("ptxv-sim", "fixtures", "probe_i2f8_queue")],
    ),
    # E-HW-SB-QUEUE: the broad rules cover an I2F.S8 by a later I2F.S16 again
    # (probes/i2f8_queue COVER: GB300 wrong).
    SrcMutant(
        "i2f-narrow-cover",
        "crates/sass/src/scoreboard.rs",
        "    if narrow_i2f(earlier) {\n        return false;",
        "    if narrow_i2f(earlier) && !broad {\n        return false;",
        [("../probes/i2f8_queue", "COVER_S4", "sass", [])],
        tests=[("ptxv-sim", "fixtures", "probe_i2f8_queue")],
    ),
    # MISS 155: a narrow I2F covered by a later I2F of its own full opcode again (RVS2 W1).
    SrcMutant(
        "i2f-same-cover",
        "crates/sass/src/scoreboard.rs",
        "    if narrow_i2f(earlier) {\n        return false;",
        "    if narrow_i2f(earlier) && earlier != later {\n        return false;",
        [("../probes/i2f8_same", "SAME_COVER_S4", "sass", [])],
        tests=[("ptxv-sim", "fixtures", "probe_i2f8_same")],
    ),
    # MISS 156: FRND keyed by base opcode at the gb300 level again (RVS2 W2).
    SrcMutant(
        "frnd-width",
        "crates/sass/src/scoreboard.rs",
        "        \"FRND\" if hw && f64_op => \"FRND.F64\",\n",
        "",
        [("../probes/frnd64", "COVER_S2", "sass", [])],
        tests=[("ptxv-sim", "fixtures", "probe_frnd_width_queues")],
    ),
    # Item 1 (SND3): a late WAW on a narrow I2F's register a violation again (not `sb:lat:`).
    SrcMutant(
        "waw-lat",
        "crates/sass/src/scoreboard.rs",
        '(kind == "nosb" || kind == "waw" && sb == NO_SB)',
        'kind == "nosb"',
        [],
        tests=[("ptxv-sim", "fixtures", "probe_i2f8_lat_classes")],
    ),
    # The f64 narrow I2F in the narrow (uncovered) class again.
    SrcMutant(
        "lat-f64",
        "crates/sass/src/scoreboard.rs",
        '        && !opcode.split(\'.\').any(|x| x == "F64")\n',
        '',
        [],
        tests=[("ptxv-sass", "lib", "hw_measured_windows")],
    ),
    # MISS 170 R4: FLO by a later POPC at the gb300 level again.
    SrcMutant(
        "r4-flo-popc",
        "crates/sass/src/scoreboard.rs",
        '        ("LDS", "SHFL") => true,\n',
        '        ("LDS", "SHFL") | ("FLO", "POPC") => true,\n',
        [],
        tests=[("ptxv-sim", "fixtures", "miss170_unmeasured_gb300_rules"),
("ptxv-sass", "lib", "hw_measured_windows")],
    ),
    # MISS 170 R5: the f64 pipe across DADD / DMUL / DFMA / DSETP at the gb300 level again.
    SrcMutant(
        "r5-f64-pipe",
        "crates/sass/src/scoreboard.rs",
        '        ("LDS", "SHFL") => true,\n',
        '        ("LDS", "SHFL") => true,\n        ("DADD" | "DMUL" | "DFMA" | "DSETP", "DADD" | "DMUL" | "DFMA" | "DSETP") => true,\n',
        [],
        tests=[("ptxv-sim", "fixtures", "miss170_unmeasured_gb300_rules"),
("ptxv-sass", "lib", "hw_measured_windows")],
    ),
    # MISS 170 R2: every f64 conversion pair of one base opcode at the gb300 level again.
    SrcMutant(
        "r2-f64-conv",
        "crates/sass/src/scoreboard.rs",
        '        _ => matches!(\n            (earlier, later),',
        '        _ if f64_op(earlier) => true,\n        _ => matches!(\n            (earlier, later),',
        [],
        tests=[("ptxv-sim", "fixtures", "miss170_unmeasured_gb300_rules"),
("ptxv-sass", "lib", "hw_measured_windows")],
    ),
    # MISS 170 R8: the conversion / f64 units' queued WAR at the gb300 level again.
    SrcMutant(
        "r8-qwar",
        "crates/sass/src/scoreboard.rs",
        'fn hw_queued_war_measured(_reader: &str, _writer: &str) -> bool {\n    false',
        'fn hw_queued_war_measured(_reader: &str, _writer: &str) -> bool {\n    true',
        [],
        tests=[("ptxv-sim", "fixtures", "probe_xu_war_matches_hardware"),
("ptxv-sass", "lib", "hw_measured_windows")],
    ),
    # MISS 170 R10 / R11: every memory-queue cover across opcodes at the gb300 level again.
    SrcMutant(
        "r10-r11-cover",
        "crates/sass/src/scoreboard.rs",
        '    pair && gap.0 >= 4 && gap.1 <= HW_QUEUE_GAP_MAX && !later_guarded\n',
        '    let _ = (pair, gap, later_guarded);\n    true\n',
        [],
        tests=[("ptxv-sim", "fixtures", "probe_sb_mio_order_matches_hardware"),
("ptxv-sim", "fixtures", "probe_sb_mio_gap_2048"),
("ptxv-sass", "lib", "hw_measured_windows")],
    ),
    # MISS 170 (probes/sb_mio_gap): the R10 to R12 windows' upper end at 2048, where no control
    # is wrong (the windows at 2048 become clean).
    SrcMutant(
        "queue-gap-2048",
        "crates/sass/src/scoreboard.rs",
        "const HW_QUEUE_GAP_MAX: u16 = 1024;\n",
        "const HW_QUEUE_GAP_MAX: u16 = 2048;\n",
        [],
        tests=[("ptxv-sim", "fixtures", "probe_sb_mio_gap_2048"),
("ptxv-sass", "lib", "hw_measured_windows")],
    ),
    # ... back at 22 (the windows measured at 64 and 1024 become contracts).
    SrcMutant(
        "queue-gap-22",
        "crates/sass/src/scoreboard.rs",
        "const HW_QUEUE_GAP_MAX: u16 = 1024;\n",
        "const HW_QUEUE_GAP_MAX: u16 = 22;\n",
        [],
        tests=[("ptxv-sim", "fixtures", "probe_sb_mio_gap_64_address")],
    ),
    # ... the gap's maximum over the paths capped at 64 again: a longer (or a loop's) gap reads
    # as 64, inside the windows (the windows at 2048 become clean).
    SrcMutant(
        "gap-cap-64",
        "crates/sass/src/scoreboard.rs",
        "const CMAX_CAP: u16 = HW_QUEUE_GAP_MAX + 1;\n",
        "const CMAX_CAP: u16 = 64;\n",
        [],
        tests=[("ptxv-sim", "fixtures", "probe_sb_mio_gap_2048"),
("ptxv-sass", "lib", "hw_measured_windows")],
    ),
    # ... the new pairs (other widths, LDG / SHFL covers, STG readers) unmeasured again.
    SrcMutant(
        "cover-wide-off",
        "crates/sass/src/scoreboard.rs",
        "    let pair = both || write && cover == Cover::Write;\n",
        "    let pair = matches!((earlier, later), (\"STS\", \"LDS\")) && (both || write);\n",
        [],
        tests=[("ptxv-sim", "fixtures", "probe_sb_mio_gap_64_covers")],
    ),
    # ... the write-scoreboard-only pairs trusted through the read scoreboard too.
    SrcMutant(
        "cover-write-only-read",
        "crates/sass/src/scoreboard.rs",
        "    let pair = both || write && cover == Cover::Write;\n",
        "    let pair = both || write;\n    let _ = cover;\n",
        [],
        tests=[("ptxv-sass", "lib", "hw_measured_windows")],
    ),
    SrcMutant(
        "war-wide-off",
        "crates/sass/src/scoreboard.rs",
        '''        | ("STSM.16.M88.4", "LDS.128" | "LDSM.16.M88.4" | "LDS") => {
            data && within(HW_QUEUE_GAP_MAX)
        }
''',
        '''        | ("STSM.16.M88.4", "LDS.128" | "LDSM.16.M88.4" | "LDS") => {
            data && within(HW_QUEUE_GAP_MAX) && reader == "STS"
        }
''',
        [],
        tests=[("ptxv-sim", "fixtures", "probe_sb_mio_gap_64_data")],
    ),
    # ... the data-only pairs trusted on address registers too.
    SrcMutant(
        "war-data-only-addr",
        "crates/sass/src/scoreboard.rs",
        '''        | ("STSM.16.M88.4", "LDS.128" | "LDSM.16.M88.4" | "LDS") => {
            data && within(HW_QUEUE_GAP_MAX)
        }
''',
        '''        | ("STSM.16.M88.4", "LDS.128" | "LDSM.16.M88.4" | "LDS") => {
            (data || true) && within(HW_QUEUE_GAP_MAX)
        }
''',
        [],
        tests=[("ptxv-sass", "lib", "hw_measured_windows")],
    ),
    # MISS 250: an attached entry's gap counters restart at the attach (the windows then count
    # from the later instruction: an R12 WAR 2009 cycles after an untracked STS reads as 1005).
    SrcMutant(
        "attach-gap-reset",
        "crates/sass/src/scoreboard.rs",
        "        (e.cyc, e.cmax, e.cov) = (p.cyc, p.cmax, p.cov);\n",
        "        (e.cyc, e.cmax, e.cov) = (0, 0, p.cov);\n",
        [],
        tests=[("ptxv-sim", "fixtures", "probe_sb_mio_gap_attach")],
    ),
    # MISS 264 (a): a fresh attached entry one DEPBAR count too high (the attach before the
    # attaching op's issue): DEPBAR.LE SBk, 1 drops it with only that op outstanding.
    SrcMutant(
        "attach-newer-fresh-one",
        "crates/sass/src/scoreboard.rs",
        "    e.newer = 0;\n    if fresh {\n",
        "    e.newer = u8::from(fresh);\n    if fresh {\n",
        [],
        tests=[("ptxv-sass", "lib", "attach")],
    ),
    # MISS 264 (b): attached onto an entry already there, its larger count kept.
    SrcMutant(
        "attach-newer-merge-keeps",
        "crates/sass/src/scoreboard.rs",
        "    e.newer = 0;\n    if fresh {\n",
        "    if fresh {\n",
        [],
        tests=[("ptxv-sass", "lib", "attach")],
    ),
    # attach onto an entry already there: the incoming covers replace the existing ones'.
    SrcMutant(
        "attach-cov-overwrite",
        "crates/sass/src/scoreboard.rs",
        "        e.cov = e.cov.intersection(&p.cov).copied().collect();\n",
        "        e.cov = p.cov;\n",
        [],
        tests=[("ptxv-sass", "lib", "attach")],
    ),
    # ... the guard holds if either held.
    SrcMutant(
        "attach-gok-either",
        "crates/sass/src/scoreboard.rs",
        "    e.gok &= p.gok;\n    e.regs.extend(p.regs);\n",
        "    e.gok |= p.gok;\n    e.regs.extend(p.regs);\n",
        [],
        tests=[("ptxv-sass", "lib", "attach")],
    ),
    # ... the existing entry's gaps kept.
    SrcMutant(
        "attach-existing-keeps-own-gap",
        "crates/sass/src/scoreboard.rs",
        "        e.cyc = e.cyc.min(p.cyc);\n        e.cmax = e.cmax.max(p.cmax);\n        e.cov = e.cov",
        "        e.cov = e.cov",
        [],
        tests=[("ptxv-sass", "lib", "attach")],
    ),
    # MISS 170 R12: every queued load WAR across opcodes at the gb300 level again.
    SrcMutant(
        "r12-load-war",
        "crates/sass/src/scoreboard.rs",
        '    let within = |hi: u16| gap.0 >= 4 && gap.1 <= hi;\n',
        '    let within = |_hi: u16| true;\n    let data = data || true;\n',
        [],
        tests=[("ptxv-sim", "fixtures", "probe_sb_mio_order_dq_windows"),
("ptxv-sass", "lib", "hw_measured_windows")],
    ),
    # MISS 170 R13: TMA stores' and tcgen05 ops' reads attached to a later read scoreboard of
    # their unit at the gb300 level again.
    SrcMutant(
        "r13-tma-attach",
        "crates/sass/src/scoreboard.rs",
        '                            || !matches!(ub, "UTMASTG" | "utc") && kb == base\n',
        '                            || !matches!(ub, "UTMASTG" | "utc") && kb == base\n'
        '                            || hw && matches!(ub, "UTMASTG" | "utc")\n',
        [],
        tests=[("ptxv-sim", "fixtures", "tma_store_reads_its_coordinate_registers")],
    ),
    # MISS 170 R9: UGETNEXTWORKID's reads at issue at the gb300 level again.
    SrcMutant(
        "r9-ugnwi",
        "crates/sass/src/scoreboard.rs",
        'fn hw_ugetnextworkid_measured() -> bool {\n    false',
        'fn hw_ugetnextworkid_measured() -> bool {\n    true',
        [],
        tests=[("ptxv-sim", "fixtures", "clc_war_operands_read_at_issue"),
("ptxv-sass", "lib", "hw_measured_windows")],
    ),
    # Item 4: the LDSM / LDS covers past the measured gaps (2 to 8 stall cycles).
    SrcMutant(
        "smem-gap-hi",
        "crates/sass/src/scoreboard.rs",
        '                && gap.1 <= 8\n',
        '                && gap.1 <= 64\n',
        [],
        tests=[("ptxv-sass", "lib", "hw_measured_windows")],
    ),
    # Item 4: the LDSM / LDS covers below the measured gaps.
    SrcMutant(
        "smem-gap-lo",
        "crates/sass/src/scoreboard.rs",
        '            ) && gap.0 >= 2\n',
        '            ) && gap.0 >= 1\n',
        [],
        tests=[("ptxv-sass", "lib", "hw_measured_windows")],
    ),
    # Item 4: the LDSM / LDS covers by a guarded later op (only unguarded measured).
    SrcMutant(
        "smem-guard",
        "crates/sass/src/scoreboard.rs",
        '                && !later_guarded\n',
        '\n',
        [],
        tests=[("ptxv-sass", "lib", "hw_measured_windows")],
    ),
    # Item 4: the LDSM covers by any later shared op width / SHFL mode.
    SrcMutant(
        "smem-width",
        "crates/sass/src/scoreboard.rs",
        '                ("LDSM.16.M88.4", "LDS" | "SHFL.BFLY")\n',
        '                ("LDSM.16.M88.4", _)\n',
        [],
        tests=[("ptxv-sass", "lib", "hw_measured_windows")],
    ),
    # Item 4: the measured LDSM / LDS covers dropped (precision: the probes' COVER must be sb-hwq:).
    SrcMutant(
        "smem-pairs-off",
        "crates/sass/src/scoreboard.rs",
        '        "LDSM" if hw => "shared",\n',
        '',
        [],
        tests=[("ptxv-sim", "fixtures", "probe_smem_queue_classes")],
    ),
    # Item 4: the gap's upper end joined as a minimum (a longer path's gap unseen).
    SrcMutant(
        "cmax-join",
        "crates/sass/src/scoreboard.rs",
        '                if f.cmax > p.cmax {',
        '                if f.cmax < p.cmax {',
        [],
        tests=[("ptxv-sass", "lib", "join_gap_bounds")],
    ),
    # MISS 181: an f64 operand one register again (a read of the high word unchecked).
    SrcMutant(
        "f64-width",
        "crates/sass/src/scoreboard.rs",
        '        "DADD" | "DMUL" | "DFMA" | "DMNMX" => all(true),\n',
        '        "DADD" | "DMUL" | "DFMA" | "DMNMX" => none(false),\n',
        [],
        tests=[("ptxv-sim", "fixtures", "f64_pairs_are_tracked"), ("ptxv-sass", "lib", "def_use_forms")],
    ),
    # MISS 181 (RVS3 H1..H8): the lift's register groups untracked again (the text's widths only:
    # `LDCU.64 UR6` defines UR6, STG.E.ENL2.256 reads 2 of its 8 data registers).
    SrcMutant(
        "lift-widths-off",
        "crates/sass/src/scoreboard.rs",
        "        for (regs, out) in [(op.dst_regs(), &mut defs), (op.src_regs(), &mut uses)] {\n",
        "        for (regs, out) in [(Vec::new(), &mut defs), (Vec::<ptxv_ir::Reg>::new(), &mut uses)] {\n",
        [],
        tests=[("ptxv-sass", "lib", "lift_widths_are_tracked"), ("ptxv-sim", "fixtures", "miss181_lift_width_repros")],
    ),
    # MISS 181: only the lift's destinations (sources untracked again: STG.256 data, MATCH.U64).
    SrcMutant(
        "lift-widths-uses",
        "crates/sass/src/scoreboard.rs",
        "        for (regs, out) in [(op.dst_regs(), &mut defs), (op.src_regs(), &mut uses)] {\n",
        "        for (regs, out) in [(op.dst_regs(), &mut defs), (Vec::new(), &mut uses)] {\n",
        [],
        tests=[("ptxv-sim", "fixtures", "miss181_lift_width_repros")],
    ),
    # MISS 181 (RVS3 R1): a REDG / ATOMG .ADD.BF16x4's data one register again (R5 unchecked).
    SrcMutant(
        "atom-bf16x4-width",
        "crates/ir/src/lib.rs",
        "                        if matches!(op, AtomOp::FAddFtz | AtomOp::AddBf16x2) && *bytes > 4 =>\n",
        "                        if matches!(op, AtomOp::FAddFtz) && *bytes > 4 =>\n",
        [],
        tests=[("ptxv-sass", "lib", "lift_widths_are_tracked")],
    ),
    # MISS 151: ptxas's 128 stall cycles trusted again (f64 reads measured wrong at 256 become
    # contracts).
    SrcMutant(
        "f64-lat-128",
        "crates/sass/src/scoreboard.rs",
        "                        || f64 && pend.cyc >= F64_LAT);",
        "                        || f64 && pend.cyc >= 128);",
        [],
        tests=[("ptxv-sim", "fixtures", "f64_lat_classes")],
    ),
    # MISS 151: no f64 latency contract at all (a late read a violation at every distance).
    SrcMutant(
        "f64-lat-off",
        "crates/sass/src/scoreboard.rs",
        "                        || f64 && pend.cyc >= F64_LAT);",
        "                        || f64 && pend.cyc >= u16::MAX);",
        [],
        tests=[("ptxv-sim", "fixtures", "f64_lat_classes")],
    ),
    # MISS 181: the 64-bit conversion operands one register again.
    SrcMutant(
        "cvt64-width",
        "crates/sass/src/scoreboard.rs",
        '        "I2F" => (has("F64"), Box::new(move |_| int64)),\n',
        '        "I2F" => (false, Box::new(move |_| false)),\n',
        [],
        tests=[("ptxv-sass", "lib", "def_use_forms")],
    ),
    # MISS 157: STAS's operand reads at issue again (RVS2 W3).
    SrcMutant(
        "stas-async",
        "crates/sass/src/scoreboard.rs",
        "    \"UBLKCP\",\n    \"STAS\",\n",
        "    \"UBLKCP\",\n",
        [],
        tests=[("ptxv-sim", "fixtures", "async_read_ops_read_late")],
    ),
    # RVS2 P1: what only a broader level finds is reported again (the strict rules prove
    # probes/sq_guard MERGE_FALSEV safe).
    SrcMutant(
        "merge-strict-only",
        "crates/sass/src/scoreboard.rs",
        "    for (k, h) in strict {\n",
        "    let mut strict = strict;\n    for (k, h) in hw.iter().chain(broad.iter()) {\n        strict.entry(*k).or_insert_with(|| h.clone());\n    }\n    for (k, h) in strict {\n",
        [("../probes/sq_guard", "MERGE_FALSEV", "sass", [])],
        tests=[("ptxv-sim", "fixtures", "level_merge_per_hazard")],
    ),
    # L-OPCLASS: an I2F.S8 read past CONV_LAT_CYC is a violation again (no latency contract).
    SrcMutant(
        "i2f-lat-class",
        "crates/sass/src/scoreboard.rs",
        "                    && (lat_measured(&lines[ppc].opcode) && pend.cyc >= CONV_LAT_CYC\n",
        "                    && (lat_measured(&lines[ppc].opcode) && pend.cyc >= u16::MAX\n",
        [],
        tests=[("ptxv-sim", "fixtures", "probe_i2f8_lat_classes")],
    ),
    # MISS 186: the latency contract from 17 stall cycles again (reads measured wrong at 33
    # under saturation become contracts).
    SrcMutant(
        "i2f-lat-16",
        "crates/sass/src/scoreboard.rs",
        "                    && (lat_measured(&lines[ppc].opcode) && pend.cyc >= CONV_LAT_CYC\n",
        "                    && (lat_measured(&lines[ppc].opcode) && pend.cyc > 16\n",
        [],
        tests=[("ptxv-sim", "fixtures", "probe_i2f8_lat_classes")],
    ),
    # MISS 186: every narrow I2F in the latency class again (forms no probe measured).
    SrcMutant(
        "i2f-lat-forms",
        "crates/sass/src/scoreboard.rs",
        '    opcode == "I2F.S8"\n',
        "    narrow_i2f(opcode)\n",
        [],
        tests=[("ptxv-sass", "lib", "hw_measured_windows")],
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
    # MISS 232: a remote op's landing writes its pipe key in DPOR footprints only again, so the
    # synchronization reduction expands the landing alone ahead of another lane's drain:
    # probes/sync_pipe_issue_order SASS [clean] in the default explorer, and
    # probes/sync_pipe_landing_count SASS [clean] where the last-arrival keys (MISS 231
    # follow-up) order the first probe's landing without the pipe key.
    # Coverage: since MISS 275 this mutant survives. Both probes observe the landing through a
    # test_wait, which now writes the count keys of the phases it reads; the landing reads them,
    # so the reduction orders the landing against the test without the pipe key. Other
    # observers of a landing read or write space 16 too (arrivals, tx changes); a blocking wait
    # is enabled only after the completion. What the key alone still separates is the landing's
    # clock (inside a drain the op joins the drainer's clock): an st.async variant observed by a
    # read after the drainer's release gave [race] with and without the key (the landing alone
    # has fewer happens-before edges, so every race of the drained order). No kernel found that
    # depends on the key; not shown equivalent either. Both probes pin the default explorer,
    # nosyncpor, certify, DPOR and the unreduced search (run.sh). The two keys overlap: with
    # both the landing key and the peek's count keys left out, sync_pipe_landing_count SASS is
    # [clean] again (RDP2), so the landing key stays (a coverage gap, MISS 232 residual; not
    # equivalent).
    SrcMutant(
        "sync-pipe-landing-nokey",
        "crates/sim/src/exec.rs",
        '        // the landing alone; probes/sync_pipe_issue_order SASS was clean).\n        if self.fp.is_some() {',
        '        // the landing alone; probes/sync_pipe_issue_order SASS was clean).\n        if self.fp.as_ref().is_some_and(|f| !f.sync_only) {',
        [],
        [("ptxv-sim", "fixtures", "sync_reduction_orders_a_landing_against_a_drain")],
    ),
    # MISS 230: the race check joins the clock of every update a blocking wait acquires again,
    # also for a parity poll's later phase (the releaser's whole history counts as ordered
    # before the wait): probes/dpor_aba_wait good DPOR [clean] against the stateful [race].
    SrcMutant(
        "dpor-wait-joins-history",
        "crates/sim/src/dpor.rs",
        "                            if fp.waits_on_once(&e.fp) {\n",
        "                            if fp.waits_on(&e.fp) {\n",
        [],
        [("ptxv-sim", "fixtures", "dpor_explores_a_blocking_wait_before_earlier_phases")],
    ),
    # MISS 230 follow-up: no join for waits whose generation cannot repeat (skip only the
    # releasers everywhere): e_exitbound XB2 DPOR keeps one outcome, e_sym good PTX 713 states.
    SrcMutant(
        "dpor-no-once-join",
        "crates/sim/src/dpor.rs",
        "                            if fp.waits_on_once(&e.fp) {\n",
        "                            if fp.waits_on_once(&e.fp) && false {\n",
        [],
        [
            (
                "ptxv-sim",
                "fixtures",
                "dpor_orders_releasers_history_before_a_wait_that_cannot_repeat",
            )
        ],
    ),
    # MISS 231: a pure tx increment made with no arrival pending writes no count key (it
    # commutes with decrements in the footprints): probes/dpor_tx_commute DPOR [clean].
    SrcMutant(
        "tx-increment-nokey",
        "crates/sim/src/exec.rs",
        "        let write = (delta > 0 && !arrival) || (under && pending == 0);\n",
        "        let write = under && pending == 0;\n",
        [],
        [("ptxv-sim", "fixtures", "dpor_orders_tx_increments_against_decrements")],
    ),
    # MISS 231: the phase's last arrival does not write its last-arrival key, so DPOR never
    # runs it before a pure expect_tx made with an arrival pending (probes/dpor_tx_commute).
    SrcMutant(
        "tx-last-arrival-nokey",
        "crates/sim/src/exec.rs",
        "        if last {\n            // The phase's last arrival writes its own tx increment's level",
        "        if false && last {\n            // The phase's last arrival writes its own tx increment's level",
        [],
        [("ptxv-sim", "fixtures", "dpor_orders_tx_increments_against_decrements")],
    ),
    # MISS 231: a pure expect_tx made with an arrival pending reads no last-arrival key.
    SrcMutant(
        "tx-increment-pending-nokey",
        "crates/sim/src/exec.rs",
        '            let (lo, hi) = mbar_last_keys(key, 0, u64::MAX);\n            self.rec_mbar_count(cta, lo, hi, false);\n            return;\n',
        '            return;\n',
        [],
        [("ptxv-sim", "fixtures", "dpor_orders_tx_increments_against_decrements")],
    ),
    # MISS 231: a decrement that takes tx below 0 reads no last-arrival key, so DPOR never runs
    # the last arrival before it (with tx 0 that arrival completes the phase first).
    SrcMutant(
        "tx-underflow-nolast",
        "crates/sim/src/exec.rs",
        "        if under && pending > 0 {\n",
        "        if false {\n",
        [],
        [("ptxv-sim", "fixtures", "dpor_orders_tx_increments_against_decrements")],
    ),
    # MISS 273: a decrement that takes tx below 0 with an arrival pending writes no under key,
    # so DPOR runs one order of the decrements, whose deficits against the last arrival differ
    # by order: probes/dpor_tx_level DPOR [deadlock], probes/tx_level_po stateful [deadlock].
    SrcMutant(
        "tx-under-nowrite",
        "crates/sim/src/exec.rs",
        '            self.rec_mbar_count(cta, u, u + 1, under);',
        '            self.rec_mbar_count(cta, u, u + 1, false);',
        [],
        [("ptxv-sim", "fixtures", "dpor_orders_tx_increments_against_decrements")],
    ),
    # MISS 231: a decrement that takes tx below 0 with no arrival pending writes no count key,
    # so DPOR and the synchronization reduction take it as independent of another decrement
    # that would complete the phase first: probes/dpor_tx_overdec DPOR [deadlock],
    # probes/sync_tx_overdec_tma stateful [deadlock].
    SrcMutant(
        "tx-overdec-nowrite",
        "crates/sim/src/exec.rs",
        "        let write = (delta > 0 && !arrival) || (under && pending == 0);\n",
        "        let write = delta > 0 && !arrival;\n",
        [],
        [("ptxv-sim", "fixtures", "dpor_orders_tx_increments_against_decrements")],
    ),
    # MISS 231: a TMA landing's complete_tx records no count key: probes/dpor_tx_commute_async
    # TMA DPOR [clean].
    SrcMutant(
        "tx-tma-landing-nokey",
        "crates/sim/src/exec.rs",
        "                self.rec_tx(cta, mbar, -bytes, false);\n",
        "",
        [],
        [("ptxv-sim", "fixtures", "dpor_orders_tx_increments_against_decrements")],
    ),
    # MISS 231: a CLC response's complete_tx records no count key: probes/dpor_tx_commute_async
    # CLC DPOR [clean].
    SrcMutant(
        "tx-clc-nokey",
        "crates/sim/src/exec.rs",
        "                self.rec_tx(c, mbar, -16, false);\n",
        "",
        [],
        [("ptxv-sim", "fixtures", "dpor_orders_tx_increments_against_decrements")],
    ),
    # MISS 231: a landing remote expect_tx records no count key: probes/dpor_tx_commute_remote
    # SASS DPOR [clean].
    SrcMutant(
        "tx-remote-landing-nokey",
        "crates/sim/src/exec.rs",
        "            self.rec_tx(cta, mbar, tx, n > 0);\n",
        "",
        [],
        [("ptxv-sim", "fixtures", "dpor_orders_tx_increments_against_decrements")],
    ),
    # MISS 233: tcgen05.alloc / dealloc record only the allocator key again:
    # probes/dpor_tmem_dealloc NOFREEWAIT SASS DPOR [race] against [race, violation].
    SrcMutant(
        "tmem-mark-nowords",
        "crates/sim/src/exec.rs",
        '                space: ResourceSpace::Tmem,\n                owner: cta as u32,\n                start: lo as u64,\n                end: hi as u64,\n',
        '                space: ResourceSpace::Tmem,\n                owner: cta as u32,\n                start: lo as u64,\n                end: lo as u64,\n',
        [],
        [("ptxv-sim", "fixtures", "dpor_orders_tmem_dealloc_against_tmem_access")],
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
    # MISS 221: a peek's success acquires as a blocking wait again, so DPOR's race check skips
    # the completion it observed and never runs the test before it: probes/dpor_testwait good
    # DPOR [clean] against the stateful [nondeterministic] (the test runs both explorers).
    SrcMutant(
        "dpor-peek-acquires",
        "crates/sim/src/exec.rs",
        "                            self.rec_obs(sync_key(2, cta as u64, off as u64, k));",
        "                            self.rec_acq(sync_key(2, cta as u64, off as u64, k));",
        [],
        [("ptxv-sim", "fixtures", "dpor_explores_both_outcomes_of_a_test_wait")],
    ),
    # MISS 274: DPOR enters a successor equal to a state on its stack again, and a thread that
    # repeats one state (probes/mbar_zero upto0) gives one path without end: the test bounds
    # DPOR to 10000 states and fails on the incomplete search.
    SrcMutant(
        "dpor-no-cycle-cut",
        "crates/sim/src/dpor.rs",
        "        if let Some(&j) = on_stack.get(&h) {",
        "        if let Some(&j) = on_stack.get(&h).filter(|_| false) {",
        [],
        [("ptxv-sim", "fixtures", "dpor_stops_at_a_cycle")],
    ),
    # MISS 279: a poll of an uninitialized mbarrier records no footprint again, so the
    # reductions run it before the init only: probes/uninit_poll_publish good SASS default and
    # DPOR [violation] against [nondeterministic, race, violation].
    SrcMutant(
        "uninit-poll-nokey",
        "crates/sim/src/exec.rs",
        '                        {\n                            self.rec_mbar_query(cta, off, None);\n                        }\n                        _ => self.rec_fault(),',
        '                        {\n                            let _ = (cta, off);\n                        }\n                        _ => self.rec_fault(),',
        [],
        [("ptxv-sim", "fixtures", "reductions_order_a_poll_before_init_against_the_init")],
    ),
    # MISS 279 (RDP2 finding J): the landings of a bulk copy, a tcgen05.commit, a
    # cp.async.mbarrier.arrive and a CLC response read no (re)initialization key before the
    # init again: probes/uninit_mbar_landing TMA / COMMIT default [deadlock, violation].
    SrcMutant(
        "landing-no-init-key",
        "crates/sim/src/exec.rs",
        '    fn rec_mbar_init_key(&mut self, cta: u16, off: u32) {\n        self.rec_sync(\n            Resource {\n                space: ResourceSpace::MbarInitialization,\n                owner: cta as u32,\n                start: off as u64,\n                end: off as u64 + 1,\n            },\n            false,\n        );\n    }\n',
        '    fn rec_mbar_init_key(&mut self, cta: u16, off: u32) {\n        let _ = (cta, off);\n    }\n',
        [],
        [("ptxv-sim", "fixtures", "reductions_order_a_poll_before_init_against_the_init")],
    ),
    # MISS 282: E-POLL-ANY takes a load of a currently undefined byte for a poll again, and its
    # concrete candidates drop the undefined bits: probes/vip_undef_load LDW/LDY/LDG default
    # [clean] against [violation].
    SrcMutant(
        "vip-undef-now",
        "crates/sim/src/exec.rs",
        "        if (0..*bytes as usize).any(undef) {\n            return false;\n        }\n",
        "        let _ = undef;\n",
        [],
        [("ptxv-sim", "fixtures", "e_poll_any_keeps_undefined_bytes")],
    ),
    # MISS 282: a write that leaves a watched location undefined is not learned, so a load
    # that sees defined bytes passes for a poll before that write runs: probes/vip_undef_load
    # UST default [undef-insensitive] against [violation].
    SrcMutant(
        "vip-undef-unlearned",
        "crates/sim/src/exec.rs",
        "                                if und && vl.undef.lock().unwrap().insert((a, o, nb)) {",
        "                                if und && false {",
        [],
        [("ptxv-sim", "fixtures", "e_poll_any_keeps_undefined_bytes")],
    ),
    # MISS 295: the synchronization reduction learns a value-insensitive poll's step without
    # its read again: probes/sync_vip_learned ACQ SASS default [clean] against
    # [nondeterministic, violation] (four workers).
    SrcMutant(
        "sync-vip-unlearned",
        "crates/sim/src/explore.rs",
        "    for r in std::mem::take(&mut m.vip_learn) {\n        fp.r.push(r);\n    }\n",
        "    m.vip_learn.clear();\n",
        [],
        [("ptxv-sim", "fixtures", "sync_reduction_learns_a_value_insensitive_polls_read")],
    ),
    # MISS 274: DPOR cuts a cycle without adding the processes enabled on it (the stack
    # proviso): probes/dpor_cycle_proviso PTX DPOR [violation] against [race, violation] (the
    # repeating thread is picked first and the racing warps never run).
    SrcMutant(
        "dpor-no-proviso",
        "crates/sim/src/dpor.rs",
        "                f.backtrack.extend(en);",
        "                let _ = en;",
        [],
        [("ptxv-sim", "fixtures", "dpor_stops_at_a_cycle")],
    ),
    # MISS 275: a test of an mbarrier phase writes no count key again, so an arrival that does not
    # complete the phase in DPOR's first trace never conflicts with the test, and DPOR never
    # runs the test before it: probes/dpor_peek_arrival DPOR [clean] against the stateful
    # [nondeterministic] (the test runs both explorers).
    SrcMutant(
        "dpor-peek-nocounts",
        "crates/sim/src/exec.rs",
        "        for p in ph.saturating_sub(2)..=ph {",
        "        for p in ph..ph {",
        [],
        [("ptxv-sim", "fixtures", "dpor_explores_both_outcomes_of_a_test_wait")],
    ),
    # MISS 222: issuing into a remote pipe records no footprint again, so DPOR never reverses
    # two lanes' issues into one pipe: probes/dpor_pipe_issue_order SWAP PTX DPOR [clean]
    # against the stateful [nondeterministic] (the test runs both explorers).
    SrcMutant(
        "dpor-pipe-issue-nokey",
        "crates/sim/src/exec.rs",
        '                space: ResourceSpace::RemotePipe,\n                owner: 2,\n                start: k,\n                end: k.saturating_add(1),\n',
        '                space: ResourceSpace::RemotePipe,\n                owner: 2,\n                start: k,\n                end: k,\n',
        [],
        [("ptxv-sim", "fixtures", "dpor_reverses_issues_into_one_pipe")],
    ),
    # S-REMOTE-INFLIGHT: an op enters its in-order pipe at position 0 (every op of a pipe can
    # land first; MISS 154's issue-order positions dropped).
    SrcMutant(
        "pipe-seq-zero",
        "crates/sim/src/exec.rs",
        "op.seq = self.pending.iter().filter(|o| o.fifo == Some(f)).count() as u32;",
        "op.seq = self.pending.iter().filter(|o| o.fifo == Some(f)).count() as u32 * 0;",
        [("e_remote_pipe_order", "good", f, []) for f in ("ptx", "sass")],
    ),
    # E-LOCKSTEP: no join after an instruction whose scoreboard wait completed an LDTM (MISS
    # 163).
    SrcMutant(
        "lockstep-noldtmwait",
        "crates/sim/src/exec.rs",
        ".any(|(&t, &n)| self.threads[t].ldtm_pend.len() < n);",
        ".any(|(&t, &n)| self.threads[t].ldtm_pend.len() < n && false);",
        [("e_ldtm_join", "good", "sass", [])],
    ),
    # exec.rs `dead_target`: the issue-time liveness check of a remote access records no read
    # of the target's exit key again, so DPOR takes the issue and the target's exit as
    # independent (MISS 152; e_stasync SA2's DPOR cross-check then misses the race).
    SrcMutant(
        "dead-target-nokey",
        "crates/sim/src/exec.rs",
        '                owner: target as u32,\n                start: 1,\n                end: 2,\n',
        '                owner: target as u32,\n                start: 1,\n                end: 1,\n',
        [],
        [("ptxv-sim", "fixtures", "e_stasync_complete_tx")],
    ),
    # S-REMOTE-INFLIGHT: an in-flight st.async writes under its issuer's clock again, so the
    # issuer's later releases cover the write before it lands (MISS 184; the certificate's one
    # schedule certifies e_dpor_nonhead good).
    SrcMutant(
        "stasync-issuer-actor",
        "crates/sim/src/exec.rs",
        "                    let actor = self.async_actor(t)?;\n"
        "                    let mut vc = vc;\n"
        "                    vc_tick(&mut vc, actor);\n"
        "                    let r = self.mbar_async_rel(t as u32, cta, vc.clone());",
        "                    let actor = t as u32;\n"
        "                    let r = self.mbar_async_rel(t as u32, cta, vc.clone());",
        [],
        [("ptxv-sim", "fixtures", "stasync_inflight_is_not_covered_by_issuer_releases")],
    ),
    # S-REMOTE-INFLIGHT: a cluster release drains its thread's remote ops in pending order,
    # ahead of another lane's earlier op in the same pipe (MISS 184; e_remote_pipe_order DRN
    # becomes a false race).
    SrcMutant(
        "drain-skips-pipe-head",
        "crates/sim/src/exec.rs",
        "            } else {\n                self.take_pending(k)\n            };",
        "            } else {\n                self.take_pending(i + 0 * k)\n            };",
        [("e_remote_pipe_order", v, f, []) for v in ("good", "DRN") for f in ("ptx", "sass")],
    ),
    # S-REMOTE-INFLIGHT: a remote store's words are not recorded for their lane's next drain
    # (`Thread::remote_landed`), so no drain orders them, whether they land in it or before
    # (MISS 184 follow-up; e_stasync_early_land FEN / FENLD and e_stasync_coalesce C1 race).
    SrcMutant(
        "remote-landed-nojoin",
        "crates/sim/src/exec.rs",
        "            if vc_get(&th.remote_landed, l.actor) < c {",
        "            if false && vc_get(&th.remote_landed, l.actor) < c {",
        [("e_stasync_early_land", "FENLD", "ptx", [])],
        [("ptxv-sim", "fixtures", "stasync_landed_before_its_drain_is_ordered_by_it")],
    ),
    # S-REMOTE-INFLIGHT: a lane's drain lands the whole coalesced operation that carries its
    # part again, the other lanes' words and transaction bytes with it, before its release
    # (the review's wrong-cleans: e_stasync_coalesce C0 loses the unwritten-y outcome,
    # probes/stasync_coalesce_lanes L2_D0_TW / L2_D1_TW turn clean).
    SrcMutant(
        "drain-lands-whole-coalesced",
        "crates/sim/src/exec.rs",
        "            let op = if mine(&self.pending[k]) && several {",
        "            let op = if mine(&self.pending[k]) && several && false {",
        [("e_stasync_coalesce", "C0", f, []) for f in ("ptx", "sass")],
        [("ptxv-sim", "fixtures", "stasync_landed_before_its_drain_is_ordered_by_it")],
    ),
    # S-REMOTE-INFLIGHT: a lane's drain takes only operations whose first lane it is, skipping
    # its part in an operation another lane of its instruction issued, which then stays
    # unordered (e_stasync_coalesce C1 races again, the old false race).
    SrcMutant(
        "drain-skips-lane-words",
        "crates/sim/src/exec.rs",
        "                if lanes.iter().any(|l| l.thread == tt))",
        "                if lanes.iter().take(1).any(|l| l.thread == tt))",
        [("e_stasync_coalesce", "C1", f, []) for f in ("ptx", "sass")],
        [("ptxv-sim", "fixtures", "stasync_landed_before_its_drain_is_ordered_by_it")],
    ),
    # E-REMOTE-FIFO / S-REMOTE-INFLIGHT: the delivery before a landing takes the ordered op
    # behind another lane's pending head of its pipe again (MISS 154's order broken; DPOR's
    # non-head backtrack runs on e_remote_fifo_nonhead SASS, dpor::NONHEAD_BACKTRACKS).
    SrcMutant(
        "fifo-skips-pipe-head",
        "crates/sim/src/exec.rs",
        "            let a = self.take_pending(k);",
        "            let a = self.take_pending(i + 0 * k);",
        [],
        [("ptxv-sim", "fixtures", "deliver_fifo_before_lands_the_pipe_head_first")],
    ),
    # MISS 149: the SASS lockstep join no longer follows FENCE.VIEW.ASYNC.T; the softmax warps'
    # STTM writes (complete at the thread's own fence since MISS 107) do not reach the elected
    # lane's arrive and race with the PV UTCHMMA: c_fa4_fwd's certificate is refused.
    SrcMutant(
        "lockstep-nowaitst",
        "crates/sim/src/exec.rs",
        "                            | Op::TmemSt { .. }\n                            | Op::TmemWaitSt\n",
        "                            | Op::TmemSt { .. }\n",
        [],
        [("ptxv-sim", "fixtures", "c_fa4_fwd_certifies")],
    ),
    # E-HW-SPLIT-UR at every level: the strict level drops the switches too (a measured null
    # result is a bound): e_split_ur's strict verdict, nondeterministic through the other
    # split's VOTEU inside a window, turns clean.
    SrcMutant(
        "split-ur-ungated",
        "crates/sim/src/exec.rs",
        "        if self.launch.hw_split_ur\n"
        "            && !self.ur_window.is_empty()\n"
        "            && self.ur_window_blocks(members, pc)",
        "        if !self.ur_window.is_empty() && self.ur_window_blocks(members, pc)",
        [],
        [("ptxv-sim", "wellsync", "cert_crosscheck_deadlocking_schedule_refutes"),
         ("ptxv-sim", "fixtures", "e_split_ur_strict_level_switches_inside_a_window")],
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
    # E-BSYNC (MISS 81): every BSYNC form is a reconvergence point again: ptxas's plain
    # BSSY B0 / BSYNC B0 around the elected lane's nanosleep spin holds the waiting lanes, and
    # e_bsync_plain good is SASS clean (GB300: 10,000 of 10,000 runs wrong).
    SrcMutant(
        "bsync-plain-holds",
        "crates/sim/src/exec.rs",
        "    form == BarForm::Reconvergent\n}",
        "    let _ = form;\n    true\n}",
        [("e_bsync_plain", "good", "sass", [])],
        [("ptxv-sim", "fixtures", "e_bsync_plain_is_no_reconvergence_point")],
    ),
    # E-BSYNC (MISS 81): the lifter drops .RECONVERGENT: BSYNC.RECONVERGENT holds no lane, and
    # e_bsync_plain LATE_RECONV (ptxas reconverges the clock spin's region) races on SASS.
    SrcMutant(
        "bsync-form-plain",
        "crates/sass/src/lib.rs",
        "                let form = if has(\"RECONVERGENT\") {",
        "                let form = if has(\"RECONVERGENT_\") {",
        [("e_bsync_plain", "LATE_RECONV", "sass", [])],
        [("ptxv-sim", "fixtures", "e_bsync_plain_is_no_reconvergence_point")],
    ),
    # E-BSYNC gap G1: the lifter keeps every .RECONVERGENT BSYNC a reconvergence point, also
    # around a region that can yield: e_bsync_plain SLEEP_RECONV (the elected lane's nanosleep
    # spin joined by BSYNC.RECONVERGENT) is SASS clean (GB300: 10,000 of 10,000 runs wrong).
    SrcMutant(
        "bsync-g1-off",
        "crates/sass/src/lib.rs",
        "    reconvergent_yields(&lines, &mut code);\n",
        "",
        [("e_bsync_plain", "SLEEP_RECONV", "sass", [])],
        [("ptxv-sim", "fixtures", "e_bsync_plain_is_no_reconvergence_point")],
    ),
    # G1's region walk: a called function's yield is outside the region.
    SrcMutant(
        "bsync-g1-nocall",
        "crates/sass/src/lib.rs",
        "                    work.push((*target, true));\n",
        "",
        [],
        [("ptxv-sass", "lib", "tests::reconvergent_bsync_region_walk")],
    ),
    # G1's region walk: a WARPSYNC.COLLECTIVE (BRA.DIV's slow path) does not count.
    SrcMutant(
        "bsync-g1-nocoll",
        "crates/sass/src/lib.rs",
        "                || o.starts_with(\"NANOSLEEP\")\n                || matches!(",
        "                || o.starts_with(\"NANOSLEEP\")\n                || false && matches!(",
        [],
        [("ptxv-sass", "lib", "tests::reconvergent_bsync_region_walk")],
    ),
    # G1's region walk: any guarded BREAK of the barrier ends the path (the lanes that do not
    # break stay in the region).
    SrcMutant(
        "bsync-g1-break",
        "crates/sass/src/lib.rs",
        "                Op::Break { b: c, cond: None } if *c == b && always => {}",
        "                Op::Break { b: c, cond: None } if *c == b => {}",
        [],
        [("ptxv-sass", "lib", "tests::reconvergent_bsync_region_walk")],
    ),
    # E-HW-BSYNC-YIELD (GB300 profile): no yield or block opens a barrier, so a BSYNC without
    # .RECONVERGENT holds as .RECONVERGENT does: e_bsync_plain good is clean under the profile
    # (GB300: wrong in every run).
    SrcMutant(
        "bsync-yield-off",
        "crates/sim/src/exec.rs",
        "            if self.launch.hw_bsync_yield\n                && k.front == Front::Sass\n                && self.bsync_yield_check(warp, pc, &members)",
        "            if false\n                && k.front == Front::Sass\n                && self.bsync_yield_check(warp, pc, &members)",
        [],
        [("ptxv-sim", "fixtures", "e_bsync_plain_is_no_reconvergence_point")],
    ),
    # E-HW-BSYNC-YIELD at the strict level (a measured null as a strict rule): the plain BSYNC
    # holds probes/syncwarp_fast's passing lanes until the elected lane yields, and the strict
    # level's uniform-register clobber (a fault at the bulk copy) disappears.
    SrcMutant(
        "bsync-yield-strict",
        "crates/sim/src/lib.rs",
        "        hw_bsync_yield: false,",
        "        hw_bsync_yield: true,",
        [],
        [("ptxv-sim", "fixtures", "probe_syncwarp_fast_matches_hardware")],
    ),
    # E-HW-BSYNC-YIELD: only a YIELD / NANOSLEEP opens a barrier, a blocked lane does not:
    # e_bsync_block good and WARPSYNC (lane 1 blocked at another barrier's BSYNC / at a
    # WARPSYNC while lanes 2-31 wait at a plain BSYNC) are clean under the profile.
    SrcMutant(
        "bsync-yield-noblock",
        "crates/sim/src/exec.rs",
        "            if rec & !parked[b] & (yl | blocked | elsewhere) == 0 {",
        "            if rec & !parked[b] & yl == 0 {",
        [],
        [("ptxv-sim", "fixtures", "s_warpcoll_matches_hardware"),
         ("ptxv-sim", "fixtures", "probe_syncwarp_fast_matches_hardware"),
         ("ptxv-sim", "fixtures", "bsync_releases"),
         ("ptxv-sim", "fixtures", "e_bsync_block_opens_a_plain_bsync_under_the_gb300_profile")],
    ),
    # C-WARP under the GB300 profile: the certificate follows its run through a BSYNC without
    # .RECONVERGENT, whose holding depends on when lanes yield or block.
    SrcMutant(
        "bsync-cert-yield",
        "crates/sim/src/wellsync.rs",
        "        Op::Bsync { form, .. } if m.launch.hw_bsync_yield && !crate::exec::bsync_holds(*form) => {",
        "        Op::Bsync { form, .. } if false && !crate::exec::bsync_holds(*form) => {",
        [],
        [("ptxv-sim", "fixtures", "e_bsync_plain_is_no_reconvergence_point")],
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
    # explore.rs `exit_ends_reachable_smem`: the exit of a CTA's last running threads taken
    # as commuting with a peer CTA's access to its shared memory again (MISS 153).
    SrcMutant(
        "exit-ample-smem",
        "crates/sim/src/explore.rs",
        "let Some(&t0) = split.first() else {",
        "let Some(&t0) = split.first().filter(|_| false) else {",
        [("e_dsmem_rank", "CLUSTER", "ptx", [])]
        + [("e_dsmem_exit", "NOBAR", f, []) for f in ("ptx", "sass")],
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
        "init: std::sync::Arc::new(g.init.clone().unwrap_or_else(|| vec![0; g.size as usize])),",
        "init: std::sync::Arc::new(vec![0; g.size as usize]),",
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
        "crates/memory-write/src/barrier.rs",
        "    let completed = phase_completes(state.pending, state.transactions == 0);",
        "    let completed = phase_completes(state.pending, true);",
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
        (
            "                    let v = m.bar_read_c(t, k, wc);\n"
            "                    let i = kids.binary_search_by_key(&v, |e| e.0).ok()?;"
        ),
        "                    let (v, i) = (kids[0].0, 0usize);",
        [("e_barwalk", v, f, []) for v in ("good", "BW1") for f in ("ptx", "sass")],
        tests=[LIB, ("ptxv-sim", "lib", "explore::walk_cache_tests::walk_cache_keys_register_barrier_ids")],
    ),
    SrcMutant(
        "walkid-ops",
        "crates/sim/src/explore.rs",
        (
            "fn walk_same(a: (&BarCfg, &ptxv_ir::Kernel), b: (&BarCfg, &ptxv_ir::Kernel)) -> bool {\n"
            "    a.0 == b.0\n"
            "        && match (a.0.program.kernel_option(), b.0.program.kernel_option()) {\n"
            "            (None, None) => true,\n"
            "            (Some(left), Some(right)) => {\n"
            "                left.front == right.front\n"
            "                    && left.code.len() == right.code.len()\n"
            "                    && left\n"
            "                        .code\n"
            "                        .iter()\n"
            "                        .zip(&right.code)\n"
            "                        .all(|(x, y)| x.guard == y.guard && x.op == y.op)\n"
            "            }\n"
            "            _ => false,\n"
            "        }\n"
            "        && a.1.front == b.1.front\n"
            "        && a.1.code.len() == b.1.code.len()\n"
            "        && a.1\n"
            "            .code\n"
            "            .iter()\n"
            "            .zip(&b.1.code)\n"
            "            .all(|(x, y)| x.guard == y.guard && x.op == y.op)\n"
            "}"
        ),
        (
            "fn walk_same(a: (&BarCfg, &ptxv_ir::Kernel), b: (&BarCfg, &ptxv_ir::Kernel)) -> bool {\n"
            "    a.0 == b.0\n"
            "        && match (a.0.program.kernel_option(), b.0.program.kernel_option()) {\n"
            "            (None, None) => true,\n"
            "            (Some(left), Some(right)) => {\n"
            "                left.front == right.front\n"
            "                    && left.code.len() == right.code.len()\n"
            "                    && left\n"
            "                        .code\n"
            "                        .iter()\n"
            "                        .zip(&right.code)\n"
            "                        .all(|(x, y)| x.guard == y.guard)\n"
            "            }\n"
            "            _ => false,\n"
            "        }\n"
            "        && a.1.front == b.1.front\n"
            "        && a.1.code.len() == b.1.code.len()\n"
            "        && a.1\n"
            "            .code\n"
            "            .iter()\n"
            "            .zip(&b.1.code)\n"
            "            .all(|(x, y)| x.guard == y.guard)\n"
            "}"
        ),
        [],
        tests=[LIB],
    ),
    SrcMutant(
        "walkid-guards",
        "crates/sim/src/explore.rs",
        (
            "fn walk_same(a: (&BarCfg, &ptxv_ir::Kernel), b: (&BarCfg, &ptxv_ir::Kernel)) -> bool {\n"
            "    a.0 == b.0\n"
            "        && match (a.0.program.kernel_option(), b.0.program.kernel_option()) {\n"
            "            (None, None) => true,\n"
            "            (Some(left), Some(right)) => {\n"
            "                left.front == right.front\n"
            "                    && left.code.len() == right.code.len()\n"
            "                    && left\n"
            "                        .code\n"
            "                        .iter()\n"
            "                        .zip(&right.code)\n"
            "                        .all(|(x, y)| x.guard == y.guard && x.op == y.op)\n"
            "            }\n"
            "            _ => false,\n"
            "        }\n"
            "        && a.1.front == b.1.front\n"
            "        && a.1.code.len() == b.1.code.len()\n"
            "        && a.1\n"
            "            .code\n"
            "            .iter()\n"
            "            .zip(&b.1.code)\n"
            "            .all(|(x, y)| x.guard == y.guard && x.op == y.op)\n"
            "}"
        ),
        (
            "fn walk_same(a: (&BarCfg, &ptxv_ir::Kernel), b: (&BarCfg, &ptxv_ir::Kernel)) -> bool {\n"
            "    a.0 == b.0\n"
            "        && match (a.0.program.kernel_option(), b.0.program.kernel_option()) {\n"
            "            (None, None) => true,\n"
            "            (Some(left), Some(right)) => {\n"
            "                left.front == right.front\n"
            "                    && left.code.len() == right.code.len()\n"
            "                    && left\n"
            "                        .code\n"
            "                        .iter()\n"
            "                        .zip(&right.code)\n"
            "                        .all(|(x, y)| x.op == y.op)\n"
            "            }\n"
            "            _ => false,\n"
            "        }\n"
            "        && a.1.front == b.1.front\n"
            "        && a.1.code.len() == b.1.code.len()\n"
            "        && a.1\n"
            "            .code\n"
            "            .iter()\n"
            "            .zip(&b.1.code)\n"
            "            .all(|(x, y)| x.op == y.op)\n"
            "}"
        ),
        [],
        tests=[LIB],
    ),
    SrcMutant(
        "walkid-barcfg",
        "crates/sim/src/explore.rs",
        (
            "fn walk_same(a: (&BarCfg, &ptxv_ir::Kernel), b: (&BarCfg, &ptxv_ir::Kernel)) -> bool {\n"
            "    a.0 == b.0"
        ),
        (
            "fn walk_same(a: (&BarCfg, &ptxv_ir::Kernel), b: (&BarCfg, &ptxv_ir::Kernel)) -> bool {\n"
            "    true"
        ),
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
            "                                    if !ptxv_ir::analysis_knowledge::value_is_thread_private(src) {\n"
            "                                        return None;\n"
            "                                    }\n"
            "                                    let k = BarKey::Val(src.clone());\n"
            "                                    let x = machine.bar_read(t, &k);\n"
            "                                    reads.push((k, x));\n"
            "                                    return x;\n"
            "                                }"
        ),
        (
            "                                if let Some((_, src)) = renv.iter().find(|e| e.0 == *r) {\n"
            "                                    if !ptxv_ir::analysis_knowledge::value_is_thread_private(src) {\n"
            "                                        return None;\n"
            "                                    }\n"
            "                                    let k = BarKey::Val(src.clone());\n"
            "                                    let x = machine.bar_read(t, &k);\n"
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
            "                                let k = BarKey::Val(v.clone());\n"
            "                                let x = machine.bar_read(t, &k);\n"
            "                                reads.push((k, x));\n"
            "                                x.map(|x| (x, 0))"
        ),
        (
            "                                let k = BarKey::Val(v.clone());\n"
            "                                let x = machine.bar_read(t, &k);\n"
            "                                x.map(|x| (x, 0))"
        ),
        [],
        tests=[("ptxv-sim", "lib", WCT + "walk_cache_keys_a_setp_operand")],
    ),
    SrcMutant(
        "walk-noconv",
        "crates/sim/src/explore.rs",
        (
            "            let c = self.bar_read(t, &BarKey::Conv);\n"
            "            reads.push((BarKey::Conv, c));"
        ),
        (
            "            let c = self.bar_read(t, &BarKey::Conv);\n"
            "            let _ = c;"
        ),
        [],
        tests=[("ptxv-sim", "lib", WCT + "walk_cache_keys_the_warps_convergence")],
    ),
    SrcMutant(
        "walkseen-hash4",
        "crates/sim/src/explore.rs",
        (
            "            fn fresh(&mut self, state: &(usize, Env, REnv)) -> bool {\n"
            "                self.seen.insert(state.clone())\n"
            "            }"
        ),
        (
            "            fn fresh(&mut self, state: &(usize, Env, REnv)) -> bool {\n"
            "                let mut hash = WalkHasher::default();\n"
            "                std::hash::Hash::hash(state, &mut hash);\n"
            "                let bucket = (std::hash::Hasher::finish(&hash) >> 60) as usize;\n"
            "                self.seen.insert((bucket, vec![], vec![]))\n"
            "            }"
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
            "        let key = BarKey::Pred(register);\n"
            "        let value = self.machine.bar_read(self.thread, &key);\n"
            "        self.reads.push((key, value));"
        ),
        (
            "        let key = BarKey::Pred(register);\n"
            "        let value = self.machine.bar_read(self.thread, &key);\n"
            "        let _ = &key;"
        ),
        [],
        tests=[("ptxv-sim", "lib", WCT + "walk_cache_keys_a_register_id_walks_predicate")],
    ),
    SrcMutant(
        "walk-nouniread",
        "crates/sim/src/explore.rs",
        (
            "                                    let k = BarKey::Uni(a.clone());\n"
            "                                    let u = machine.bar_read(t, &k);\n"
            "                                    reads.push((k, u));"
        ),
        (
            "                                    let k = BarKey::Uni(a.clone());\n"
            "                                    let u = machine.bar_read(t, &k);\n"
            "                                    let _ = &k;"
        ),
        [],
        tests=[("ptxv-sim", "lib", WCT + "walk_cache_keys_an_r2urs_uniformity")],
    ),
    SrcMutant(
        "walkseen-pconly",
        "crates/sim/src/explore.rs",
        (
            "            fn fresh(&mut self, state: &(usize, Env, REnv)) -> bool {\n"
            "                self.seen.insert(state.clone())\n"
            "            }"
        ),
        (
            "            fn fresh(&mut self, state: &(usize, Env, REnv)) -> bool {\n"
            "                self.seen.insert((state.0, vec![], vec![]))\n"
            "            }"
        ),
        [],
        tests=[("ptxv-sim", "lib", WCT + "register_id_walk_visits_every_state")],
    ),
    SrcMutant(
        "plainseen-pconly",
        "crates/sim/src/explore.rs",
        (
            "            fn fresh(&mut self, state: &(usize, Env, VEnv)) -> bool {\n"
            "                self.seen.insert(state.clone())\n"
            "            }"
        ),
        (
            "            fn fresh(&mut self, state: &(usize, Env, VEnv)) -> bool {\n"
            "                self.seen.insert((state.0, vec![], vec![]))\n"
            "            }"
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
        "            Scope::Cta => cx == cy && !self.nic_apart(x, y),",
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
        (
            '                            available: self\n                                .tmem_find(self.threads[t].cta, (n / 32) as u32)\n                                .is_some(),'
        ),
        (
            '                            available: self\n                                .tmem_find(self.threads[t].cta, (n / 32) as u32)\n                                .is_some() || true,'
        ),
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
    # older capture's list may serialize grids that ran at the same time). Under the strict
    # default good's reader has no predecessor whose clock could cover the write, so the
    # ordered-reading test (PTXV_UNRECORDED_ORDER=ordered) is the one that tells.
    SrcMutant(
        "nc-start-gate",
        "crates/sim/src/exec.rs",
        "        let before_start = |w: &Epoch| launch.order_recorded && started_after(w);",
        "        let before_start = |w: &Epoch| started_after(w);",
        [("e_ld_nc_xgpu_unord", "good", "ptx", [])],
        [
            ("ptxv-sim", "fixtures", "e_ld_nc_xgpu_unord_launch_order"),
            ("ptxv", "launch_order", "ordered_reading_keeps_the_launch_order_gates"),
        ],
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
# The gates below matter for an unknown order only under PTXV_UNRECORDED_ORDER=ordered (the
# strict default leaves an unknown-order grid no predecessor to exclude by): checks with it.
LO_ORDERED = ("ptxv", "launch_order", "ordered_reading_keeps_the_launch_order_gates")
MUTANTS += [
    # S-NC's stream-order exclusion applies to an unknown-order list again.
    SrcMutant(
        "lo-nc-gate",
        "crates/sim/src/machine.rs",
        "        self.order_recorded && self.completes_before(j, i)",
        "        self.completes_before(j, i)",
        [("e_launch_order", "UNKNOWN", f, []) for f in ("ptx", "sass")]
        + [("e_launch_waits", "UNKNOWN", f, []) for f in ("ptx", "sass")],
        [LO_STREAMS, LO_WAITS, LO_ORDERED],
    ),
    # S-NC's done-clock exemption applies to an unknown-order list again.
    SrcMutant(
        "lo-done-gate",
        "crates/sim/src/exec.rs",
        "        let done_before = |g: usize| launch.order_recorded && done_by(g);",
        "        let done_before = |g: usize| done_by(g);",
        [("e_ld_nc_pdl", "UNKNOWN_DONE", f, []) for f in ("ptx", "sass")],
        [LO_UNKNOWN, LO_ORDERED],
    ),
    # E-L1-CORES / E-L1-HAVOC: the kernel boundary of an unknown-order list counts again
    # (`grids_ordered`, then the done-clock check).
    SrcMutant(
        "lo-l1-gate",
        "crates/sim/src/l1_cores.rs",
        "    l.order_recorded && l.hb_before(a, b)",
        "    l.hb_before(a, b)",
        [("e_launch_order_l1", "UNKNOWN", f, []) for f in ("ptx", "sass")],
        [LO_L1, LO_ORDERED],
    ),
    SrcMutant(
        "lo-l1-done-gate",
        "crates/sim/src/l1_cores.rs",
        "            if self.launch.order_recorded\n                && self.launch.full_preds(rg).any(|p| {",
        "            if self.launch.full_preds(rg).any(|p| {",
        [("e_launch_order_l1", "UNKNOWN", f, []) for f in ("ptx", "sass")],
        [LO_L1, LO_ORDERED],
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

# E-LAUNCH-ORDER in equiv, errdist and compare: their checks are of the variant's launch order
# (`Case::for_variant`, as `ptxv check`), and the certificate check's verdict cache serves only
# variants without launch-order edits (`Variant::edits_order`).
LO_CLI = "launch_order"
MUTANTS += [
    # equiv / errdist check the case as loaded again (fixtures/e_equiv_order UNK, UNORD clean).
    SrcMutant(
        "lo-equiv-variant",
        "crates/cli/src/equiv.rs",
        "    let case = Case::load(base.join(&k.case))?.for_variant(&k.variant)?;",
        "    let case = Case::load(base.join(&k.case))?;",
        [],
        [
            ("ptxv", LO_CLI, "equiv_precondition_is_the_variants_launch_order"),
            (
                "ptxv",
                "bin:ptxv",
                "equiv::launch_order_tests::precondition_is_the_variants_launch_order",
            ),
        ],
    ),
    # compare checks the case as loaded again.
    SrcMutant(
        "lo-compare-variant",
        "crates/cli/src/compare.rs",
        "        case.for_variant(&variant)?\n",
        "        case.clone()\n",
        [],
        [("ptxv", "bin:ptxv", "compare::tests::a_side_is_checked_in_its_variants_launch_order")],
    ),
] + [
    # One kind of launch-order edit no longer keeps a variant out of the verdict cache.
    SrcMutant(
        f"lo-cache-{name}",
        "crates/case/src/lib.rs",
        "        !self.after.is_empty() || !self.waits.is_empty() || self.launch_order.is_some()",
        repl,
        [],
        [("ptxv", LO_CLI, "launch_order_edits_keep_out_of_the_verdict_cache")],
    )
    for name, repl in [
        ("after", "        !self.waits.is_empty() || self.launch_order.is_some()"),
        ("waits", "        !self.after.is_empty() || self.launch_order.is_some()"),
        ("order", "        !self.after.is_empty() || !self.waits.is_empty()"),
    ]
]

# E-LAUNCH-ORDER, the strict reading of an unknown order as the default (MISS 127).
MUTANTS += [
    # An unknown order keeps E-PDL's order by default again (the grids of an older capture's
    # list stay ordered; the strict reading only with PTXV_UNRECORDED_ORDER=concurrent).
    SrcMutant(
        "lo-strict-default",
        "crates/sim/src/lib.rs",
        '        Ok("ordered") => Ok(true),\n        Err(_) | Ok("concurrent") => Ok(false),',
        '        Err(_) | Ok("ordered") => Ok(true),\n        Ok("concurrent") => Ok(false),',
        [("e_launch_order", "PLAIN_UNKNOWN", f, []) for f in ("ptx", "sass")],
        [
            ("ptxv", LO_CLI, "unrecorded_order_concurrent_drops_unknown_predecessors"),
            ("ptxv-sim", "fixtures", "e_launch_order_two_streams"),
        ],
    ),
]

# E-LAUNCH-ORDER, the GB300 runs of equiv and errdist (MISS 165): their workers run the
# variant's launch order (`--variant`; main.rs `worker_case`).
MUTANTS += [
    # equiv's witness replay and elementwise sweep run the case's own order again.
    SrcMutant(
        "hw-replay-variant",
        "crates/cli/src/equiv.rs",
        '        dump.to_str().unwrap().into(),\n        "--variant".into(),\n        variant.into(),\n    ]',
        "        dump.to_str().unwrap().into(),\n    ]",
        [],
        [("ptxv", "bin:ptxv", "equiv::launch_order_tests::gb300_replays_run_the_variant")],
    ),
    # errdist's GPU validation runs the case's own order again.
    SrcMutant(
        "errdist-gpu-variant",
        "crates/cli/src/errdist.rs",
        '        tag.into(),\n        "--variant".into(),\n        variant.into(),\n',
        "        tag.into(),\n",
        [],
        [("ptxv", "bin:ptxv", "errdist::tests::gpu_worker_runs_the_variant")],
    ),
    # A worker loads the case as on disk, whatever variant it is given.
    SrcMutant(
        "worker-case-variant",
        "crates/cli/src/main.rs",
        "    match variant {\n        Some(v) => c.for_variant(v),\n        None => Ok(c),\n    }",
        "    let _ = variant;\n    Ok(c)",
        [],
        [("ptxv", "bin:ptxv", "worker_tests::worker_case_applies_the_variants_launch_order")],
    ),
]

# errdist's compiled models (MISS 165): a model compiled before the variant's launch
# order and the strict reading applied is refused, and the key covers the reading.
MUTANTS += [
    SrcMutant(
        "model-version-11",
        "crates/cli/src/errdist.rs",
        "pub(crate) const MODEL_VERSION: u32 = 26;",
        "pub(crate) const MODEL_VERSION: u32 = 11;",
        [],
        [("ptxv", "bin:ptxv", "errdist::tests::models_before_the_variant_order_are_refused")],
    ),
    SrcMutant(
        "model-key-reading",
        "crates/cli/src/errdist.rs",
        "    eat(reading.as_bytes());\n",
        "    let _ = reading;\n",
        [],
        [("ptxv", "bin:ptxv", "errdist::tests::model_key_covers_the_unknown_order_reading")],
    ),
]

# `ptxv num`, `ptxv hw`'s simulator runs and `ptxv first-access` (MISS 165): the variant's
# launch order.
MUTANTS += [
    SrcMutant(
        "num-variant-order",
        "crates/cli/src/main.rs",
        "    let mut case = case.for_variant(variant)?;",
        "    let mut case = case.clone();",
        [],
        [("ptxv", LO_CLI, "num_and_hw_expect_run_the_variants_launch_order")],
    ),
    SrcMutant(
        "hw-expect-variant-order",
        "crates/cli/src/main.rs",
        "ptxv_sim::explore::run_once(machine(&c.for_variant(&v)?, k)?);",
        "ptxv_sim::explore::run_once(machine(&c, k)?);",
        [],
        [("ptxv", LO_CLI, "num_and_hw_expect_run_the_variants_launch_order")],
    ),
    SrcMutant(
        "hw-expect-run-order",
        "crates/cli/src/main.rs",
        'ptxv_sim::explore::run_once(machine(&c.for_variant("good")?, k)?);',
        "ptxv_sim::explore::run_once(machine(&c, k)?);",
        [],
        [("ptxv", LO_CLI, "num_and_hw_expect_run_the_variants_launch_order")],
    ),
    # `ptxv first-access` runs the case's own launch order again.
    SrcMutant(
        "first-access-variant",
        "crates/cli/src/main.rs",
        "            let c = c.for_variant(&variant)?;\n            let launch = ptxv_sim::build_launch(&c, &k)?;",
        "            let launch = ptxv_sim::build_launch(&c, &k)?;",
        [],
        [("ptxv", LO_CLI, "first_access_runs_the_variants_launch_order")],
    ),
]

# The verdict cache (MISS 167): only a check of the case as its directory holds it is stored.
MUTANTS += [
    SrcMutant(
        "vcache-in-memory",
        "crates/cli/src/main.rs",
        '                && std::env::var("PTXV_EXPLORER").as_deref() != Ok("dpor")\n'
        "                && vcache::as_on_disk(case);",
        '                && std::env::var("PTXV_EXPLORER").as_deref() != Ok("dpor");',
        [],
        [
            ("ptxv", "vcache_key", "a_residency_override_is_not_stored"),
            ("ptxv", "bin:ptxv", "vcache::tests::only_the_case_on_disk_is_stored"),
        ],
    ),
]

# E-LAUNCH-ORDER, the hardware runner (crates/hw `stream_plan`): a launch takes its case
# stream's hardware stream only when the stream's latest launch is its predecessor or completes
# before it starts in the case (transitively; not through a programmatic edge), and waits for
# every other predecessor through an event.
HW_PLAN = "stream_plan_tests::"
MUTANTS += [
    # A launch always takes its case stream's hardware stream again (behind the latest launch).
    SrcMutant(
        "hw-plan-latest",
        "crates/hw/src/lib.rs",
        "                if l.after != Some(-1)\n                    && last_on[s]\n"
        "                        .is_none_or(|q| Some(q) == pred || d[q / 64] >> (q % 64) & 1 == 1) =>",
        "                if l.after != Some(-1) =>",
        [],
        [
            ("ptxv-hw", "lib", HW_PLAN + "after_an_earlier_launch_of_the_stream_gets_a_stream_of_its_own"),
            ("ptxv-hw", "lib", HW_PLAN + "fixture_variants_wait_only_for_their_predecessors"),
        ],
    ),
    # Only the predecessor itself keeps the stream (no launch that completes before).
    SrcMutant(
        "hw-plan-done",
        "crates/hw/src/lib.rs",
        "                        .is_none_or(|q| Some(q) == pred || d[q / 64] >> (q % 64) & 1 == 1) =>",
        "                        .is_none_or(|q| Some(q) == pred) =>",
        [],
        [("ptxv-hw", "lib", HW_PLAN + "a_latest_launch_done_before_keeps_the_stream")],
    ),
    # What completes before a predecessor no longer completes before the launch.
    SrcMutant(
        "hw-plan-transitive",
        "crates/hw/src/lib.rs",
        "            for (x, y) in d.iter_mut().zip(&done[p]) {\n                *x |= y;\n            }\n",
        "",
        [],
        [("ptxv-hw", "lib", HW_PLAN + "a_latest_launch_done_before_keeps_the_stream")],
    ),
    # A programmatic predecessor counts as completed before the launch.
    SrcMutant(
        "hw-plan-pdl-done",
        "crates/hw/src/lib.rs",
        "            if !programmatic(p) {\n                d[p / 64] |= 1 << (p % 64);",
        "            if true {\n                d[p / 64] |= 1 << (p % 64);",
        [],
        [("ptxv-hw", "lib", HW_PLAN + "a_programmatic_edge_does_not_complete_its_predecessor")],
    ),
    # An earlier predecessor on the launch's own stream is left to the stream order again.
    SrcMutant(
        "hw-plan-same-stream",
        "crates/hw/src/lib.rs",
        "            if plan.stream[p] == s && Some(p) == last_on[s] {",
        "            if plan.stream[p] == s {",
        [],
        [("ptxv-hw", "lib", HW_PLAN + "an_earlier_predecessor_of_the_stream_is_an_event")],
    ),
    # A programmatic predecessor that is also in `waits` keeps programmatic serialization.
    SrcMutant(
        "hw-plan-waited-pdl",
        "crates/hw/src/lib.rs",
        "l.pdl && Some(p) == pred && !l.waits.contains(&(p as i64));",
        "l.pdl && Some(p) == pred;",
        [],
        [("ptxv-hw", "lib", HW_PLAN + "a_waited_predecessor_is_not_programmatic")],
    ),
]

# The runner's done sets past one word of launches, and its GPU path (crates/hw `run_with`).
MUTANTS += [
    # The done bit of launch p lands in word 0 whatever p is: plans of more than 64 launches.
    SrcMutant(
        "hw-plan-word-index",
        "crates/hw/src/lib.rs",
        "                d[p / 64] |= 1 << (p % 64);",
        "                d[0] |= 1 << (p % 64);",
        [],
        [("ptxv-hw", "lib", HW_PLAN + "done_sets_past_64_launches")],
    ),
    # Each launch's event is recorded on the NULL stream instead of the launch's hardware stream
    # (non-blocking streams: a wait for it no longer waits for the launch). Only a GPU run shows
    # it: caught with PTXV_TEST_GPU=<an idle GPU>; without, the test is skipped.
    SrcMutant(
        "hw-event-null-stream",
        "crates/hw/src/lib.rs",
        '                    ck(cuEventRecord(events[i], hw_stream[i]), "cuEventRecord")?;',
        '                    ck(cuEventRecord(events[i], std::ptr::null_mut()), "cuEventRecord")?;',
        [],
        [("ptxv", "hw_runner", "gpu_runs_keep_the_case_order")],
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
# (`sync_observe_cell`): the flag race remains, the data race goes (test only; with the NIC's
# fetch-add, e_nic_cta PART_SRC, MISS 194).
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
        "            Overlap::Partial => (false, false),\n",
        "            Overlap::Partial => (self.scopes_mutual(a, sa, b, sb), false),\n",
        [("e_warp_order_mixed", "MIXED", "ptx", [])],
        [SCOPE_OVERLAP],
    ),
    SrcMutant(
        "scope-overlap-sync",
        "crates/sim/src/exec.rs",
        "        if ov == Overlap::Partial {\n            return;\n        }\n",
        "        if false && ov == Overlap::Partial {\n            return;\n        }\n",
        [],
        [SCOPE_OVERLAP, ("ptxv-sim", "fixtures", "e_nic_cta_overlap")],
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

# The NIC sibling of MISS 100 / MISS 112 (`nic_publish`): a NIC atomic continues the release
# sequence of the write it reads only when the two are morally strong (nic-rseq-moral: always),
# and keeps the write's own release pattern apart with its thread and scope (nic-rseq-join: the
# pattern joined into the NIC atomic's system-scope release again, the fix undone; FREL, a .cta
# fence heading the pattern, and NICHEAD, a NIC atomic heading it, MISS 166).
NIC_RSEQ = ("ptxv-sim", "fixtures", "e_nic_rseq_morally_strong")
MUTANTS += [
    SrcMutant(
        "nic-rseq-moral",
        "crates/sim/src/nic.rs",
        "                    || self.scopes_mutual(old.writer, strong_code(old.scope), actor, STRONG_SYS)\n",
        "                    || true\n",
        [("e_nic_rseq", "good", f, []) for f in ("ptx", "sass")],
        [NIC_RSEQ],
    ),
    SrcMutant(
        "nic-rseq-join",
        "crates/sim/src/nic.rs",
        "                        cont = old.cont.clone();\n"
        "                        cont_join = old.cont_join.clone();\n"
        "                        SyncCell::add_cont(\n"
        "                            &mut cont,\n"
        "                            &mut cont_join,\n"
        "                            old.writer,\n"
        "                            &old.rel,\n"
        "                            old.rel_scope,\n"
        "                        );",
        "                        vc_join(&mut rel, &old.rel);\n"
        "                        cont = old.cont.clone();\n"
        "                        cont_join = old.cont_join.clone();",
        [("e_nic_rseq", "FREL", f, []) for f in ("ptx", "sass")]
        + [("e_nic_rseq", "NICHEAD", "ptx", [])],
        [NIC_RSEQ],
    ),
]

# MISS 193: the NIC is in no CTA or cluster (`nic_apart`); nic-scope-cta and nic-scope-cluster
# put it back in its doorbell thread's CTA or cluster.
NIC_CTA = ("ptxv-sim", "fixtures", "e_nic_cta_scope")
MUTANTS += [
    SrcMutant(
        "nic-scope-cta",
        "crates/sim/src/exec.rs",
        "            Scope::Cta => cx == cy && !self.nic_apart(x, y),\n",
        "            Scope::Cta => cx == cy,\n",
        [("e_nic_cta", "CTA", "ptx", [])],
        [NIC_CTA],
    ),
    SrcMutant(
        "nic-scope-cluster",
        "crates/sim/src/exec.rs",
        "                self.launch.cluster_of(cx).0 == self.launch.cluster_of(cy).0\n"
        "                    && !self.nic_apart(x, y)\n",
        "                self.launch.cluster_of(cx).0 == self.launch.cluster_of(cy).0\n",
        [("e_nic_cta", "CLUSTER", "ptx", [])],
        [NIC_CTA],
    ),
]

# MISS 194: complete overlap with the NIC's strong accesses (`overlap_with_nic`,
# `NicState::elem`). nic-overlap: a pair with the NIC is one location again (PART, WORDPART,
# PART_SRC clean, CQEPART clean). nic-overlap-word4: an RDMA write's 4-byte edge chunk is not
# one (WORD4 races; the unit test compares `NicState::elem` with `complete_nic`'s chunks).
# nic-overlap-cqe: a CQE field is an exact location, so a read spanning fields is a mixed-size
# race, not unmodeled (CQEPART: its masked value makes that race dead, clean).
# nic-overlap-scope: a pair the scopes already decide is reported unmodeled too (CQECTA).
# nic-overlap-rseq: a NIC atomic continues a partial write's release sequence (e_nic_rseq
# SYSREL_PART: the d2 race goes, the mixed-size race stays; test only). acc-unhashed: the state
# hash leaves out NicState::acc, so states that differ only in the element a NIC access covers
# merge (e_nic_conv2 CONV2_XY: race -> clean). cqe-rule-off: two CQE writes of one CQE are not one
# location (e_nic_sharedcq good: clean -> unmodeled).
NIC_OVERLAP = ("ptxv-sim", "fixtures", "e_nic_cta_overlap")
NIC_ACC_CQE = ("ptxv-sim", "fixtures", "e_nic_acc_hash_and_shared_cqe")
MUTANTS += [
    SrcMutant(
        "nic-overlap",
        "crates/sim/src/exec.rs",
        "    let (Some((alloc, off)), true) = (at, na.is_some() || nb.is_some()) else {\n",
        "    let (Some((alloc, off)), true) = (at, false && (na.is_some() || nb.is_some())) else {\n",
        [("e_nic_cta", v, "ptx", []) for v in ("PART", "WORDPART", "PART_SRC", "CQEPART")],
        [NIC_OVERLAP],
    ),
    SrcMutant(
        "nic-overlap-word4",
        "crates/sim/src/nic.rs",
        "                    } else if w4 >= x.addr && w4 + 4 <= end {\n",
        "                    } else if false && w4 >= x.addr && w4 + 4 <= end {\n",
        [("e_nic_cta", "WORD4", "ptx", [])],
        [NIC_OVERLAP, ("ptxv-sim", "lib", "nic::elem_tests::words_elem_matches_the_chunks")],
    ),
    SrcMutant(
        "nic-overlap-cqe",
        "crates/sim/src/exec.rs",
        "            NicElem::Field(_, w) => Some(Some((w, true))),\n",
        "            NicElem::Field(_, w) => Some(Some((w, false))),\n",
        [("e_nic_cta", "CQEPART", "ptx", [])],
        [NIC_OVERLAP],
    ),
    SrcMutant(
        "nic-overlap-scope",
        "crates/sim/src/exec.rs",
        "                (ms, ms && o == Overlap::Unmodeled)\n",
        "                (ms, o == Overlap::Unmodeled)\n",
        [("e_nic_cta", "CQECTA", "ptx", [])],
        [NIC_OVERLAP],
    ),
    SrcMutant(
        "nic-overlap-rseq",
        "crates/sim/src/nic.rs",
        "                        && self.overlap_at(old.writer, old.pc, actor, u32::MAX, Some((alloc, off)))\n"
        "                            != crate::exec::Overlap::Partial\n",
        "",
        [],
        [NIC_OVERLAP],
    ),
    SrcMutant(
        "acc-unhashed",
        "crates/sim/src/nic.rs",
        "        self.acc.hash(h);\n",
        "",
        [("e_nic_conv2", "CONV2_XY", "ptx", [])],
        [NIC_ACC_CQE],
    ),
    SrcMutant(
        "cqe-rule-off",
        "crates/sim/src/exec.rs",
        "        if p.is_some() && p == q {\n",
        "        if false && p.is_some() && p == q {\n",
        [("e_nic_sharedcq", "good", "ptx", [])],
        [NIC_ACC_CQE],
    ),
]

# Entries no fixture reaches, with the reason.
EXCLUDED = {
    "clock-hang-callee": (
        "E-CLOCK hang guards, no slice instruction in a callee: subsumed by the path rule. A "
        "callee called from two sites is in a cycle through its return edges, and the uses of "
        "its one clock read after each call are users not on every path from the read to the "
        "guard, so the path rule refuses fixtures/e_clock_hang_callee's slice as well (the "
        "mutant leaves its verdict unchanged); a callee called once is one execution"
    ),
    "cfg-call": (
        "elect_lanes::Cfg call edges (a callee under lane-varying control in the region): "
        "ptxas copies a callee's result into its caller-side register right after the CALL, "
        "inside the region, so the value is lane-varying through that copy with or without "
        "the edge (fixtures/e_clock_lanes_call: `IMAD.MOV.U32 R2, RZ, RZ, R4` after "
        "CALL.REL); a callee's other results reach the caller through memory, lane-varying "
        "as loads. No compiled kernel tells the edge apart; it is kept for completeness"
    ),
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
        "        let broken = match (broken, control) {",
        "        let broken = match (broken, control.filter(|_| false)) {",
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
        "                    op: AtomOp::FAddFtz\n                        | AtomOp::FAdd64\n                        | AtomOp::AddBf16x2\n                        | AtomOp::AddBf16\n                        | AtomOp::AddF16x2\n                        | AtomOp::AddF16,",
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
        "                            mark(e).is_some_and(|m| w == m.other(g)) && ulp_class(&b.dtype, g, w)",
        "                            ulp_class(&b.dtype, g, w)",
        [],
        [ROUND16_LANES, ROUND16_PROBES, ROUND16_CASE],
    ),
    # MISS 110: a byte marked with the other 16-bit format (or an f32 element of marked bytes)
    # takes the tolerance.
    SrcMutant(
        "round16-format",
        "crates/case/src/lib.rs",
        "                        let m = r16[start + off + e].filter(|f| f.dtype() == b.dtype)?;",
        "                        let m = r16[start + off + e]?;",
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
        "                if let Some(next) = c {\n",
        "                if let Some(next) = c.or(Some(true)) {\n",
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
    # soundness.py `race_parts`: the entries under checks_extra are not read.
    SrcMutant(
        "kdb-snd-extra-ignored",
        "fb/kernel_db/soundness.py",
        '            *(("checks_extra", x) for x in (extra if isinstance(extra, list) else [])),\n',
        "",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_certificate_signals_beside_the_check_flag")],
    ),
    # soundness.py `race_parts`: check_ptx counts only beside a check again.
    SrcMutant(
        "kdb-snd-check-ptx-needs-check",
        "fb/kernel_db/soundness.py",
        '            ("check_ptx", cp),\n',
        '            ("check_ptx", cp if isinstance(r.get("check"), dict) else None),\n',
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
    # soundness.py `race_parts`: check_ptx gets no coverage tag again, so misses of input
    # coverage leave its open list.
    SrcMutant(
        "kdb-snd-check-ptx-no-coverage",
        "fb/kernel_db/soundness.py",
        '            t = cls.check_tags({**c, "front_end": "ptx"} if c is cp else c)\n',
        '            t = cls.check_tags(c) if c is not cp else {"kdb-tooling", "race-verdict:ptx"} | '
        '({"certify"} if cls.on_certificate(c) else set())\n',
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_check_ptx_tags_are_a_ptx_checks")],
    ),
    # soundness.py `race_parts`: check_ptx's tags follow its front_end, so one without it gets
    # the SASS levels' tags instead of race-verdict:ptx.
    SrcMutant(
        "kdb-snd-check-ptx-front",
        "fb/kernel_db/soundness.py",
        '            t = cls.check_tags({**c, "front_end": "ptx"} if c is cp else c)\n',
        "            t = cls.check_tags(c)\n",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_check_ptx_tags_are_a_ptx_checks")],
    ),
    # soundness.py `race_parts`: check_ptx beside a check is judged at the check's rev again,
    # whatever rev of its own it stores (an older build's open misses go).
    SrcMutant(
        "kdb-snd-check-ptx-at-check-rev",
        "fb/kernel_db/soundness.py",
        '            if c is not chk and "ptxv_rev" in c and c["ptxv_rev"] != rev:\n',
        '            if c is not cp and c is not chk and "ptxv_rev" in c and c["ptxv_rev"] != rev:\n',
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_check_ptx_judged_at_its_own_rev")],
    ),
    # soundness.py `race_parts`: with a rev of its own, check_ptx's tags are not judged at all.
    SrcMutant(
        "kdb-snd-check-ptx-own-rev-dropped",
        "fb/kernel_db/soundness.py",
        '                own.append((c["ptxv_rev"], t, which))\n',
        '                own.append((c["ptxv_rev"], t, which)) if c is not cp else None\n',
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_check_ptx_judged_at_its_own_rev")],
    ),
    # soundness.py `race_parts`: the check's tags are judged at check_ptx's own rev, when it
    # stores one, instead of the check's (the check's open misses go when check_ptx's build is
    # newer).
    SrcMutant(
        "kdb-snd-check-at-check-ptx-rev",
        "fb/kernel_db/soundness.py",
        '        return ([(rev, at_check, "check")] if at_check else []) + own\n',
        '        return ([(own[-1][0] if own else rev, at_check, "check")] if at_check else []) + own\n',
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_check_judged_at_its_own_rev_beside_a_newer_check_ptx")],
    ),
    # soundness.py `race_parts`: entries under checks_extra are judged at the check's rev again,
    # whatever rev of their own they store.
    SrcMutant(
        "kdb-snd-extras-at-check-rev",
        "fb/kernel_db/soundness.py",
        '            if c is not chk and "ptxv_rev" in c and c["ptxv_rev"] != rev:\n',
        '            if c is cp and c is not chk and "ptxv_rev" in c and c["ptxv_rev"] != rev:\n',
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_checks_extra_judged_at_their_own_revs")],
    ),
    # soundness.py `race_parts`: a record with checks_extra but no check or check_ptx is judged
    # at its extras' own builds again (extras that only timed out grade there).
    SrcMutant(
        "kdb-snd-extras-only-own-builds",
        "fb/kernel_db/soundness.py",
        "        if not chk:\n            # checks_extra only:",
        "        if False:\n            # checks_extra only:",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_checks_extra_only_is_unknown")],
    ),
    # soundness.py `of_record`: a part at an unknown rev drops its tags' entries from the open
    # list again, those open at int7's tip included.
    SrcMutant(
        "kdb-snd-unknown-part-dropped",
        "fb/kernel_db/soundness.py",
        "                    krev = tip\n",
        "                    continue\n",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_unknown_build_lists_the_entries_open_at_the_tip")],
    ),
    # export_records.py `base`: the export status and missing_fixes ignore the builds of
    # check_ptx and checks_extra again ("current" beside a stale grade).
    SrcMutant(
        "kdb-export-base-own-builds-ignored",
        "fb/kernel_db/export_records.py",
        '        own = [p[0] for p in soundness.Soundness.race_parts(r) if p[2] != "check"]\n',
        "        own = []\n",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_base_counts_builds_of_their_own")],
    ),
    # ptx_front.py `summarize_run`: check_ptx stores no build rev again (judged at the check's).
    SrcMutant(
        "kdb-ptx-front-rev-dropped",
        "fb/kernel_db/ptx_front.py",
        '        "ptxv_rev": bin_rev(rj.get("binary")),\n',
        '        "ptxv_rev": None,\n',
        [],
        py=[("fb/kernel_db", "test_run_collect.py", "test_ptx_front_build_rev")],
    ),
    # MISS 192, soundness.py `num_parts`: an errdist result's null stamp (an unstamped errdist
    # binary) is judged at the check's rev again, not as an unknown build.
    SrcMutant(
        "kdb-snd-num-null-at-check-rev",
        "fb/kernel_db/soundness.py",
        '            if isinstance(n, dict) and "ptxv_rev" in n and n["ptxv_rev"] != crev:\n',
        '            if isinstance(n, dict) and n.get("ptxv_rev") and n["ptxv_rev"] != crev:\n',
        [],
        py=[
            ("fb/kernel_db", "test_soundness.py", "test_errdist_results_judged_at_their_own_builds"),
            ("fb/kernel_db", "test_soundness.py", "test_base_counts_errdist_builds"),
        ],
    ),
    # MISS 192, soundness.py `num_parts`: entries under numerics_extra are judged at the check's
    # rev again, whatever errdist build they store.
    SrcMutant(
        "kdb-snd-num-extra-at-check-rev",
        "fb/kernel_db/soundness.py",
        '        results += [("numerics_extra", x) for x in extra]\n',
        '        results += [("numerics_extra", {}) for x in extra]\n',
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_errdist_results_judged_at_their_own_builds")],
    ),
    # MISS 192, soundness.py `of_record`: the errdist part is judged at the check's rev only, not
    # at the builds num_parts names.
    SrcMutant(
        "kdb-snd-num-parts-unused",
        "fb/kernel_db/soundness.py",
        "                judged = [(p_rev, p_tags) for p_rev, p_tags, _ in nparts]\n",
        "                judged = [(rev, tags)]\n",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_errdist_results_judged_at_their_own_builds")],
    ),
    # MISS 192, export_records.py `base`: the errdist builds are left out of the export status
    # and missing_fixes again.
    SrcMutant(
        "kdb-export-base-num-builds-ignored",
        "fb/kernel_db/export_records.py",
        '        own += [p[0] for p in soundness.Soundness.num_parts(r) if p[2] != "check"]\n',
        "        own += []\n",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_base_counts_errdist_builds")],
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
    # export_records.py `apply_stale`: the stale list is not consulted, so a record whose case is
    # on it grades from its check alone (clean on a case whose stored bytes are no capture).
    SrcMutant(
        "kdb-export-stale-case-ignored",
        "fb/kernel_db/export_records.py",
        "    row = stale_case(r, stale)\n",
        "    row = None\n",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_stale_case_cannot_grade_clean")],
    ),
    # kdb_summary.py `load_trees` (--glob): the records it grades skip the stale list.
    SrcMutant(
        "kdb-summary-stale-ignored-glob",
        "fb/kernel_db/kdb_summary.py",
        '        r["soundness"] = snd.of(r, lv)\n        export_records.apply_stale(r, stale)\n',
        '        r["soundness"] = snd.of(r, lv)\n',
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_summary_reads_the_stale_list")],
    ),
    # kdb_summary.py `main` (--recompute, or an exported record without a grade): the records it
    # grades skip the stale list.
    SrcMutant(
        "kdb-summary-stale-ignored-recompute",
        "fb/kernel_db/kdb_summary.py",
        "            export_records.apply_stale(r, stale)\n",
        "            pass\n",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_summary_reads_the_stale_list")],
    ),
    # export_records.py `stale_case`: any hash matches, so a case captured again since it was
    # listed stays unconfirmed.
    SrcMutant(
        "kdb-export-stale-case-any-hash",
        "fb/kernel_db/export_records.py",
        '            if row["case_hash"] == h:\n',
        "            if rows:\n",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_stale_case_cannot_grade_clean")],
    ),
    # export_records.py `stale_case`: a listed case this host cannot read is passed over, so its
    # record grades from its check alone (fails open where the store is absent).
    SrcMutant(
        "kdb-export-stale-unreadable-open",
        "fb/kernel_db/export_records.py",
        "            return dict(rows[0], unreadable=True)\n",
        "            continue\n",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_stale_list_fails_closed")],
    ),
    # export_records.py `load_stale`: a missing list reads as an empty one, so the export grades
    # every record from its check alone without saying so.
    SrcMutant(
        "kdb-export-stale-missing-list-open",
        "fb/kernel_db/export_records.py",
        '    with open(path, newline="") as fh:\n        for row in csv.DictReader(fh):\n',
        '    if not os.path.isfile(path):\n        return stale\n'
        '    with open(path, newline="") as fh:\n        for row in csv.DictReader(fh):\n',
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_stale_list_fails_closed")],
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
    # srcmut.py `anchor_problem` takes any count, so --check-anchors passes with an anchor gone
    # (the anchors test's positive control: a broken anchor planted in a scratch tree).
    SrcMutant(
        "srcmut-anchors-none",
        "tools/mutate/srcmut.py",
        "    if n != 1:\n",
        "    if n < 0:\n",
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
    # srcmut.py `anchor_problem`: a find text kept only inside a comment counts as an anchor
    # again (test_srcmut_check_anchors' comment cases).
    SrcMutant(
        "srcmut-anchors-comment-ignored",
        "tools/mutate/srcmut.py",
        "    if inside and all(any(a <= k < b for a, b in inside) for k in code):\n",
        "    if False:\n",
        [],
        [("ptxv", "srcmut_anchors", "the_mutation_tools_tests_pass")],
    ),
    # srcmut.py `broken_anchors` over the whole list (test_srcmut_anchors_whole_list): only the
    # first mutant of each file is checked; mutants are checked in batches of 64 and a last
    # partial batch is dropped (the loop line kept).
    SrcMutant(
        "srcmut-anchors-per-file-once",
        "tools/mutate/srcmut.py",
        "        if m.file not in files:\n            f = tree / m.file\n",
        "        if m.file in files:\n            continue\n        if m.file not in files:\n            f = tree / m.file\n",
        [],
        [("ptxv", "srcmut_anchors", "the_mutation_tools_tests_pass")],
    ),
    SrcMutant(
        "srcmut-anchors-tail-dropped",
        "tools/mutate/srcmut.py",
        "    for m in mutants:\n",
        "    mutants = mutants[: (len(mutants) - len(mutants) % 64) or len(mutants)]\n    for m in mutants:\n",
        [],
        [("ptxv", "srcmut_anchors", "the_mutation_tools_tests_pass")],
    ),
    # srcmut.py `comment_spans`: doc comments (`///`, `//!`) are read as code, so an anchor kept
    # only in one passes (test_srcmut_check_anchors' doc-comment cases).
    SrcMutant(
        "srcmut-anchors-doc-comment-code",
        "tools/mutate/srcmut.py",
        '        if tok in ("//", "#"):\n',
        '        if tok in ("//", "#") and not text.startswith(("///", "//!"), s):\n',
        [],
        [("ptxv", "srcmut_anchors", "the_mutation_tools_tests_pass")],
    ),
    # srcmut.py `broken_anchors`: a find text over several lines is not checked
    # (test_srcmut_check_anchors' multi-line cases).
    SrcMutant(
        "srcmut-anchors-multiline-skipped",
        "tools/mutate/srcmut.py",
        "        why = anchor_problem(text, m.file, m.find, spans)\n",
        '        why = None if "\\n" in m.find.rstrip("\\n") else anchor_problem(text, m.file, m.find, spans)\n',
        [],
        [("ptxv", "srcmut_anchors", "the_mutation_tools_tests_pass")],
    ),
    # srcmut.py `fb_only_skipped`: the export's skip of the mutants of fb/ files applies in the
    # internal tree too (its manifest is not read), so an fb/ file gone from there is skipped
    # instead of a broken anchor (test_srcmut_skips_fb_only_mutants_in_an_export).
    SrcMutant(
        "srcmut-fb-skip-internal",
        "tools/mutate/srcmut.py",
        '    if (tree / "fb" / "OSS_EXCLUDE.txt").is_file():\n        return []\n',
        "",
        [],
        [("ptxv", "srcmut_anchors", "the_mutation_tools_tests_pass")],
    ),
    # srcmut.py `fb_only_skipped`: the mutants of fb/ files are skipped whether the tree holds
    # the file or not, so the whole-list test's scratch tree, which holds them all, loses them
    # (test_srcmut_anchors_whole_list, test_srcmut_skips_fb_only_mutants_in_an_export).
    SrcMutant(
        "srcmut-fb-skip-present",
        "tools/mutate/srcmut.py",
        "    return [m for m in mutants if m.fb_only and not (tree / m.file).is_file()]\n",
        "    return [m for m in mutants if m.fb_only]\n",
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
# MISS 146 / MISS 136 follow-up (dpor.rs `successors`): only consumed ops that are an
# alternative's head disable their process again ("if Some(id) != a.op &&
# !live.contains(&id) {" gains "&& alts.iter().any(|b| b.op == Some(id))").
# MISS 278: local_ample's plain-arrival rule ignoring peeks. The rule itself is gone since
# the merge of upstream #82 (explore.rs local_ample: "Even plain arrivals can cross a
# generation before this action runs"), so nothing is left to mutate; the regression test
# local_reduction_keeps_a_peek_after_its_own_arrival still runs.
EXCLUDED["local-arrival-ignores-peeks"] = (
    "local_ample's plain-arrival rule (MISS 278 narrowed it for peeks) was removed by upstream "
    "#82: every visible mbarrier arrival is a scheduling point of local_ample, so the peek "
    "condition has no code left; probes/local_peek_arrival stays pinned by "
    "local_reduction_keeps_a_peek_after_its_own_arrival"
)
EXCLUDED["dpor-nonhead-nobacktrack"] = (
    "DPOR's backtrack of a FIFO whose op behind the head was consumed while the head stayed "
    "pending: dead. Every op leaves a pipe as its head. Async steps take heads only; "
    "drain_remote lands a remote pipe's head before the drained op behind it (MISS 184; "
    "fixtures/e_dpor_nonhead NH reached the case until then); deliver_fifo_before (the opt-in "
    "E-REMOTE-FIFO) does the same since the MISS 184 follow-up (fixtures/e_remote_fifo_nonhead "
    "SASS reached the case until then). The remaining removals (complete_async, complete_clc) "
    "take an action's own op. fixtures.rs deliver_fifo_before_lands_the_pipe_head_first "
    "asserts dpor::NONHEAD_BACKTRACKS stays 0 (srcmut fifo-skips-pipe-head and "
    "drain-skips-pipe-head break the head-first landing). The rule stays as a conservative "
    "backtrack"
)

# S-REMOTE-INFLIGHT / S-CLUSTER-LIFE (exec.rs `complete_remote_op`): a coalesced remote
# operation that lands outside a drain records the first lane's cluster-lifetime obligation
# only ("for l in lanes {" in the remote_obs block becomes "for l in lanes.iter().take(1) {").
EXCLUDED["remote-obs-first-lane"] = (
    "a non-first lane's remote_obs entry matters only when the first lane's is discharged and "
    "its own is not. The target's wait on the phase discharges every lane alike (they share it), "
    "so the first lane must be discharged by its own drain happening before the target's exit; "
    "in another schedule that drain splits the first lane out first, and the undrained lanes' "
    "remainder then lands as its own operation whose first lane is undrained: it records that "
    "lane's obligation or lands after the exit, a violation either way. So the per-lane entries "
    "add only obligations another schedule already reports (complete exploration assumed); the "
    "mutant gave identical verdicts on 46 generated and 24 hand-written runs and on "
    "fixtures/e_stasync_lanes, both fronts. The per-lane entries stay (conservative)"
)

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
# fixtures/errdist/k0). Redundant for Tier 3 since MISS 139 (below).
EXCLUDED["t3-widen-k0-off"] = (
    "Tier 3 without MISS 57's widening at 0 inputs outside (`widen_unknown` only where some "
    "input took a pattern outside): since MISS 139 Tier 3's P(differ) interval takes the "
    "input-level bound at every count (mc.rs `p_interval`, `input_bound`), which holds however "
    "the outputs of an input differ together, an unseen share of inputs outside included: an "
    "input of that share is a value g in [0, 1] like any other, and at 0 seen the bound's upper "
    "end is Clopper-Pearson over inputs, which covers a share up to U at the same level. So the "
    "widening cannot move Tier 3's claim out of a sound one; it is kept (conservative), and the "
    "analytic tiers, which use the identified functions on every input, keep it as their only "
    "cover (fifth review, item 5: the argument holds once the bound is taken at every k, "
    "left_share included)"
)

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
        "                    pair_mass += p;\n                    pair_p2 += p * p;\n                    pair_pd += p * d;\n                    pair_out += p;",
        "                    outside += p;\n                    pair_p2 += 0.0 * p;\n                    pair_pd += 0.0 * d;",
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

# MISS 138: an identified form reads every argument the slice depends on, has exact constants,
# and no form with fitted constants fits EXACT_VETO times better (ident.rs `arg_dependence`,
# `admit`, `choose`; fixtures/errdist/ident).
IDENT_FS3 = ("ptxv", "errdist_guard", "slice_form_that_leaves_out_an_argument_is_refused")
MUTANTS += [
    # Forms that leave out an argument the slice depends on are kept (the miss: y = u for
    # RM(floor(x / 1024) + u); fs3c's step of one ulp is within the veto's margin).
    SrcMutant(
        "ident-dep-off",
        "crates/sim/src/errdist/nonlin/ident.rs",
        "    if (0..dep.len()).all(|k| !dep[k] || reads_arg(&r, k)) {",
        "    if true || (0..dep.len()).all(|k| !dep[k] || reads_arg(&r, k)) {",
        [],
        [IDENT_FS3],
    ),
    # A form with fitted constants stands where none with exact constants fits.
    SrcMutant(
        "ident-fitted-ok",
        "crates/sim/src/errdist/nonlin/ident.rs",
        "    match (best, fitted) {\n        (Some(b), Some(f)) if EXACT_VETO",
        "    if let (None, Some(f)) = (best, fitted) {\n        return Ok(f.clone());\n    }\n    match (best, fitted) {\n        (Some(b), Some(f)) if EXACT_VETO",
        [],
        [
            IDENT_FS3,
            ("ptxv", "errdist_guard", "slice_fit_with_fitted_constants_is_refused"),
            (
                "ptxv-sim",
                "lib",
                "errdist::nonlin::ident::tests::a_fitted_form_alone_is_no_exact_reference",
            ),
        ],
    ),
    # A form with exact constants stands however much better a fitted one fits.
    SrcMutant(
        "ident-veto-off",
        "crates/sim/src/errdist/nonlin/ident.rs",
        "        (Some(b), Some(f)) if EXACT_VETO * f.fit_err + 2f64.powi(-23) < b.fit_err => Err(format!(",
        "        (Some(b), Some(f)) if false && EXACT_VETO * f.fit_err + 2f64.powi(-23) < b.fit_err => Err(format!(",
        [],
        [
            (
                "ptxv-sim",
                "lib",
                "errdist::nonlin::ident::tests::a_much_better_fitted_form_vetoes_an_exact_one",
            ),
        ],
    ),
    # No products of two slice constants among the snap candidates: the affine control
    # (rE_ex2ovf, slope 2^100 C) has fitted constants and is refused (precision only).
    SrcMutant(
        "ident-snap-products-off",
        "crates/sim/src/errdist/nonlin/ident.rs",
        "            if v > 1e-30 && v < 1e30 {\n                out.push((v, 2));",
        "            if false {\n                out.push((v, 2));",
        [],
        [("ptxv", "errdist_guard", "slice_slope_from_two_constants_is_exact")],
    ),
]

# MISS 147: forms of another shape that fit the probes as well are the identified form's
# alternatives; where they disagree the exact value is not established and the outputs above
# are left out (ident.rs `alternatives`, `RealFn::agrees`; real.rs `Table::ambig`;
# fixtures/errdist/ident/ex2ovf.ptx).
IDENT_ALTS = ("ptxv", "errdist_guard", "forms_the_probes_cannot_tell_apart_leave_the_exact_value_open")
MUTANTS += [
    # No alternatives: the first of equally good forms stands alone (the miss).
    SrcMutant(
        "ident-alts-off",
        "crates/sim/src/errdist/nonlin/ident.rs",
        "            b.alts = alternatives(&found, &b, n);",
        "            b.alts = Vec::new();",
        [],
        [
            IDENT_ALTS,
            (
                "ptxv-sim",
                "lib",
                "errdist::nonlin::ident::tests::forms_the_probes_cannot_tell_apart_are_alternatives",
            ),
        ],
    ),
    # The exact table ignores the alternatives' disagreement.
    SrcMutant(
        "real-alts-ignored",
        "crates/sim/src/errdist/nonlin/real.rs",
        "                            am |= !f.agrees(&a, v);",
        "                            am |= false && !f.agrees(&a, v);",
        [],
        [IDENT_ALTS],
    ),
    # A slice of FP adds, multiplies and FMAs takes the curved forms too: the exact scaling's
    # tanh match and ex2ovf2's exp2 match are alternatives again (refused where the affine form
    # is exact).
    SrcMutant(
        "ident-poly-off",
        "crates/sim/src/errdist/nonlin/ident.rs",
        "    let poly = polynomial(p);",
        "    let poly = false && polynomial(p);",
        [],
        [IDENT_ALTS],
    ),
    # Their outputs labelled as above an unidentified slice.
    SrcMutant(
        "ambig-mislabelled",
        "crates/sim/src/errdist/mc.rs",
        "            } else if ambig {\n                obs.ambiguous += 1;",
        "            } else if ambig {\n                obs.unidentified += 1;",
        [],
        [IDENT_ALTS],
    ),
    # A form with fitted constants counts as an alternative (a rounding-size coefficient on an
    # argument the output does not move with leaves out every output where it is large).
    SrcMutant(
        "ident-alts-fitted",
        "crates/sim/src/errdist/nonlin/ident.rs",
        "    for r in found.iter().filter(|r| r.snapped && r.fit_err <= tol) {",
        "    for r in found.iter().filter(|r| r.fit_err <= tol) {",
        [],
        [
            (
                "ptxv-sim",
                "lib",
                "errdist::nonlin::ident::tests::forms_the_probes_cannot_tell_apart_are_alternatives",
            ),
        ],
    ),
    # NaN from both forms disagrees: an output whose exact value is not determined is labelled
    # as having alternatives that disagree.
    SrcMutant(
        "ident-agrees-nan",
        "crates/sim/src/errdist/nonlin/ident.rs",
        "            if w == v || (w.is_nan() && v.is_nan()) {",
        "            if w == v {",
        [],
        [
            (
                "ptxv-sim",
                "lib",
                "errdist::nonlin::ident::tests::forms_the_probes_cannot_tell_apart_are_alternatives",
            ),
        ],
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
        "    if !obs.input_differ.is_empty() {",
        "    if false && !obs.input_differ.is_empty() {",
        [],
        [
            ("ptxv-sim", "lib", "errdist::mc::tests::p_interval_counts_inputs_where_few_differ"),
            ("ptxv", "errdist_adv", "tier3_interval_counts_inputs_where_outputs_differ_together"),
        ],
    ),
    # MISS 139: the input-level bound only below 20 inputs with a differing output again (the
    # third review's rE_mix, fixtures/errdist/mix: single outputs differ on most inputs, a rare
    # cluster of 4032 outputs on 2e-3 of them).
    SrcMutant(
        "t3-input-bound-small-k",
        "crates/sim/src/errdist/mc.rs",
        "    if !obs.input_differ.is_empty() {",
        "    if !obs.input_differ.is_empty() && obs.input_differ.iter().filter(|&&x| x > 0.0).count() < 20 {",
        [],
        [
            ("ptxv-sim", "lib", "errdist::mc::tests::p_interval_counts_inputs_at_any_count"),
            ("ptxv", "errdist_adv", "tier3_interval_counts_inputs_at_any_count"),
        ],
    ),
    # No growth at P 0.
    SrcMutant(
        "t3-grow-p0-off",
        "crates/cli/src/errdist.rs",
        "    let (mut want, mut bud) = if p == 0.0 && compared {\n        (f64::INFINITY, budget)",
        "    let (mut want, mut bud) = if p == 0.0 && compared {\n        (0.0, 0.0)",
        [],
        [("ptxv", "errdist_adv", "tier3_interval_counts_inputs_where_outputs_differ_together")],
    ),
    # Growth at P 0 where Tier 3 compared nothing (every output out of domain): up to 206 s
    # per kernel_db run for an interval that stays [0, 1].
    SrcMutant(
        "t3-grow-ood",
        "crates/cli/src/errdist.rs",
        "    let (mut want, mut bud) = if p == 0.0 && compared {",
        "    let (mut want, mut bud) = if p == 0.0 {",
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
        "    if obs.compared > obs.nonfinite_cmp || obs.nonfinite_differ > 0 {\n        return;\n    }",
        "    if true {\n        return;\n    }",
        [],
        [
            ("ptxv", "nonfinite", "errdist_every_input_out_of_domain_is_refused"),
            ("ptxv", "errdist_guard", "nonfinite_outputs_are_out_of_domain"),
        ],
    ),
]

# MISS 66 (N-ERRDIST-NONFINITE): outputs with a NaN / Inf value are compared against their
# reference (mc.rs `nonfinite_outcome`, `count_nonfinite`; the elementwise law; errdist.rs
# `refuse_analytic_nonfinite`, `flag_nonfinite`, `refuse_no_compared`).
MUTANTS += [
    # A single kernel's NaN / Inf output left out of the domain again (the miss).
    SrcMutant(
        "nf-skip-restored",
        "crates/sim/src/errdist/mc.rs",
        "        if !rv.is_finite() {\n            count_nonfinite(\n                obs,\n                st,\n                o,\n                ev.r_out[k],\n                NonfiniteRef::Exact(round_out(ex.y, ex.y_lo, f)),\n            );\n            continue;\n        }",
        "        if !rv.is_finite() {\n            obs.nonfinite += 1;\n            continue;\n        }",
        [],
        [("ptxv", "nonfinite", "errdist_nonfinite_outputs_that_differ_are_counted")],
    ),
    # An output equal to RN of its exact value as the same infinity counted as differing.
    SrcMutant(
        "nf-inf-differs",
        "crates/sim/src/errdist/mc.rs",
        "        NonfiniteRef::Exact(rn) => Some(out as f64 != rn),",
        "        NonfiniteRef::Exact(_) => Some(true),",
        [],
        [("ptxv", "nonfinite", "errdist_nonfinite_outputs_equal_to_rn_exact_match")],
    ),
    # Linearized tiers kept where Tier 3 has NaN / Inf outputs that differ.
    SrcMutant(
        "nf-analytic-kept",
        "crates/cli/src/errdist.rs",
        "    if obs.nonfinite_differ == 0 {\n        return;\n    }",
        "    if true {\n        return;\n    }",
        [],
        [("ptxv", "nonfinite", "errdist_nonfinite_outputs_that_differ_are_counted")],
    ),
    # A validation of a linearized tier agrees although its observed run has such outputs.
    SrcMutant(
        "nf-validation-kept",
        "crates/cli/src/errdist.rs",
        "        if o.nonfinite_differ > 0 && (!models || t3.nonfinite_differ == 0) {",
        "        if false {",
        [],
        [("ptxv", "nonfinite", "errdist_nonfinite_outputs_that_differ_are_counted")],
    ),
    # The elementwise law skips NaN / Inf outputs again (a P(differ) conditional on finite ones).
    SrcMutant(
        "nf-elem-skip",
        "crates/sim/src/errdist/nonlin/enumerate.rs",
        "            Some(Some(d)) => {\n                tot += p;\n                sum_p2 += p * p;\n                pd += p * d as u8 as f64;\n                nf_mass += p;\n                nf_pd += p * d as u8 as f64;\n                if d {\n                    off_lattice += p;\n                } else {\n                    *pmf.entry(0).or_default() += p;\n                }\n                continue;\n            }",
        "            Some(Some(_)) => {\n                skipped += p;\n                continue;\n            }",
        [],
        [("ptxv", "nonfinite", "errdist_nonfinite_outputs_that_differ_are_counted")],
    ),
    # A Tier 3 that compared only NaN / Inf outputs kept (every output +Inf: P(differ) 0).
    SrcMutant(
        "nf-no-finite-kept",
        "crates/cli/src/errdist.rs",
        "    if obs.compared > obs.nonfinite_cmp || obs.nonfinite_differ > 0 {\n        return;\n    }",
        "    if obs.compared > 0 {\n        return;\n    }",
        [],
        [("ptxv", "errdist_guard", "nonfinite_outputs_are_out_of_domain")],
    ),
]

# MISS 66 / 88: outputs left out of P(differ) widen every estimate's interval by their share's
# bound and gate it (mc.rs `left_share`, errdist.rs `widen_left`, the elementwise law's own
# share, importance sampling); an unidentified slice's outputs have no exact value, by their
# own cause, above a maximum too (real.rs `Table::unident`).
MUTANTS += [
    # No widening: the interval covers the compared outputs only (the miss).
    SrcMutant(
        "ed-left-unwidened",
        "crates/cli/src/errdist.rs",
        "    e.p_differ = Some((p, (lo - q).max(0.0), (hi + q).min(1.0)));\n    let ok = q <= EST_MAX_REL * p;",
        "    let ok = q <= EST_MAX_REL * p;",
        [],
        [("ptxv", "nonfinite", "errdist_outputs_left_out_widen_the_interval")],
    ),
    # No gate: an estimate stands with a share left out more than 10% of its P.
    SrcMutant(
        "ed-left-gate-off",
        "crates/cli/src/errdist.rs",
        "    let ok = q <= EST_MAX_REL * p;",
        "    let ok = true;",
        [],
        [("ptxv", "nonfinite", "errdist_a_large_share_left_out_is_refused")],
    ),
    # The share's interval from the outputs alone (outputs of an input left out together).
    SrcMutant(
        "ed-left-share-outputs",
        "crates/sim/src/errdist/mc.rs",
        "    let (lo, hi) = p_interval(&view, alpha);",
        "    let (lo, hi) = super::special::clopper_pearson(view.differ, view.compared, alpha);",
        [],
        [("ptxv-sim", "lib", "errdist::mc::tests::left_share_counts_inputs")],
    ),
    # The elementwise law skips its mass without a determined exact value again.
    SrcMutant(
        "ed-elem-undet-skip",
        "crates/sim/src/errdist/nonlin/enumerate.rs",
        "                None => undet += p,",
        "                None => skipped += p,",
        [],
        [("ptxv", "nonfinite", "errdist_a_large_share_left_out_is_refused")],
    ),
    # An unidentified slice's outputs counted as a non-finite exact value again.
    SrcMutant(
        "ed-unidentified-mislabelled",
        "crates/sim/src/errdist/mc.rs",
        "            } else if unident {\n                obs.unidentified += 1;",
        "            } else if unident {\n                obs.nonfinite += 1;",
        [],
        [
            ("ptxv", "nonfinite", "errdist_unidentified_slice_is_not_nonfinite"),
            ("ptxv", "nonfinite", "errdist_outputs_left_out_widen_the_interval"),
        ],
    ),
    # The unknown exact value not propagated: a maximum passes on its other child.
    SrcMutant(
        "ed-unident-max-masked",
        "crates/sim/src/errdist/nonlin/real.rs",
        "        let mut un = reads(&unident) || matches!(op, Op::Nl(NlOp::Seg { real: None, .. }));",
        "        let mut un = matches!(op, Op::Nl(NlOp::Seg { real: None, .. }));",
        [],
        [("ptxv", "nonfinite", "errdist_unidentified_slice_is_not_passed_over_by_a_maximum")],
    ),
    # Importance sampling over inputs with outputs left out kept.
    SrcMutant(
        "is-left-kept",
        "crates/cli/src/errdist.rs",
        "    if obs.left() > 0 {",
        "    if false {",
        [],
        [("ptxv", "errdist_is", "is_refuses_outputs_left_out")],
    ),
]

# MISS 66 follow-ups (the fifth review): round_out at the output format's overflow tie (D1,
# mc.rs `round_out`; fixtures/errdist/tie), correct overflows outside the worst-case bound (D5,
# `count_nonfinite`).
TIE = ("ptxv", "errdist_guard", "overflow_tie_rounds_by_the_low_part")
TIE_UNIT = (
    "ptxv-sim",
    "lib",
    "errdist::mc::tests::round_out_breaks_the_overflow_tie_by_the_low_part",
)
MUTANTS += [
    # The infinity at the tie whatever the low part (the miss).
    SrcMutant(
        "round-out-tie-off",
        "crates/sim/src/errdist/mc.rs",
        "        return if y.abs() == max + ulp_out(max, f) / 2.0 && y.signum() != y_lo.signum() {",
        "        return if false && y.abs() == max + ulp_out(max, f) / 2.0 {",
        [],
        [TIE, TIE_UNIT],
    ),
    # The fix as first proposed (a negative low part gives max): wrong at the negative tie.
    SrcMutant(
        "round-out-tie-neg-lo",
        "crates/sim/src/errdist/mc.rs",
        "        return if y.abs() == max + ulp_out(max, f) / 2.0 && y.signum() != y_lo.signum() {",
        "        return if y.abs() == max + ulp_out(max, f) / 2.0 && y_lo < 0.0 {",
        [],
        [TIE, TIE_UNIT],
    ),
    # Every single kernel's NaN / Inf output over the worst-case bound again.
    SrcMutant(
        "wc-equal-inf-over",
        "crates/sim/src/errdist/mc.rs",
        "    if st.wc.is_some() && !pre && d {",
        "    if st.wc.is_some() && !pre && (d || matches!(reference, NonfiniteRef::Exact(_))) {",
        [],
        [TIE],
    ),
]

# MISS 66 follow-ups (the fifth review): the widened interval's level (D7, mc.rs
# `level_share`; fixtures/errdist/lmix), the validation's left-out share against Tier 3's bound
# (D6, errdist.rs `flag_nonfinite`).
MUTANTS += [
    # Tier 3's own interval at 95% under the bounds that widen it again (about 92.5% together).
    SrcMutant(
        "t3-level-unsplit",
        "crates/cli/src/errdist.rs",
        "    if let (Some((p, _, _)), true) = (est.p_differ, lv < 0.05) {",
        "    if let (Some((p, _, _)), true) = (est.p_differ, false) {",
        [],
        [("ptxv", "errdist_guard", "left_out_share_holds_at_any_count")],
    ),
    # The validation prints the observed left-out share without testing it against q.
    SrcMutant(
        "val-left-share-untested",
        "crates/cli/src/errdist.rs",
        "        if slo > q {",
        "        if false && slo > q {",
        [],
        [
            (
                "ptxv",
                "bin:ptxv",
                "errdist::tests::validation_tests_the_left_out_share_against_tier3s_bound",
            ),
        ],
    ),
]

# MISS 64 follow-up (the fifth review): a Tier 3 whose compared outputs are all NaN / Inf
# stands where some differ from their reference (errdist.rs `refuse_no_compared`), and so does
# a validation's agreement on them (`flag_nonfinite`).
MUTANTS += [
    SrcMutant(
        "nf-differ-refused",
        "crates/cli/src/errdist.rs",
        "    if obs.compared > obs.nonfinite_cmp || obs.nonfinite_differ > 0 {\n        return;\n    }",
        "    if obs.compared > obs.nonfinite_cmp {\n        return;\n    }",
        [],
        [
            ("ptxv", "errdist_guard", "all_nonfinite_tier3_with_differing_outputs_stands"),
            (
                "ptxv",
                "bin:ptxv",
                "errdist::tests::all_nonfinite_tier3_stands_only_with_differing_outputs",
            ),
        ],
    ),
    # A validation whose NaN / Inf outputs differ disagrees again for want of a finite output.
    SrcMutant(
        "nf-val-differ-disagrees",
        "crates/cli/src/errdist.rs",
        "    let vacuous = o.compared == o.nonfinite_cmp && o.nonfinite_differ == 0;",
        "    let vacuous = o.compared == o.nonfinite_cmp;",
        [],
        [
            (
                "ptxv",
                "bin:ptxv",
                "errdist::tests::all_nonfinite_tier3_stands_only_with_differing_outputs",
            ),
        ],
    ),
    # A validation of equal infinities alone agrees (the rule MISS 64's follow-up keeps).
    SrcMutant(
        "nf-val-vacuous-agrees",
        "crates/cli/src/errdist.rs",
        "    let vacuous = o.compared == o.nonfinite_cmp && o.nonfinite_differ == 0;",
        "    let vacuous = false;",
        [],
        [
            ("ptxv", "errdist_guard", "nonfinite_outputs_are_out_of_domain"),
            (
                "ptxv",
                "bin:ptxv",
                "errdist::tests::all_nonfinite_tier3_stands_only_with_differing_outputs",
            ),
        ],
    ),
    # A compiled model of another version loads (errdist.rs `Model::load`): its slices'
    # identified functions are those of another ptxv (MISS 138, MISS 147 bumped the version).
    SrcMutant(
        "model-version-unchecked",
        "crates/cli/src/errdist.rs",
        "        if version != Some(MODEL_VERSION as u64) {",
        "        if version.is_none() {",
        [],
        [
            ("ptxv", "errdist_nonlin", "a_model_of_another_version_is_refused"),
            ("ptxv", "bin:ptxv", "errdist::tests::models_before_the_variant_order_are_refused"),
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
    # The resolution a linearized headline needs, Tier 3's half-width at most 2.5% of p3, at 5%
    # (the review of the heavy-tier fixes: every test passed with it).
    SrcMutant(
        "xcheck-t2-hw-5",
        "crates/cli/src/errdist.rs",
        "const XCHECK_T2_HW: f64 = 0.025;",
        "const XCHECK_T2_HW: f64 = 0.05;",
        [],
        [("ptxv", "errdist_inerr", "errstate_feeds_the_next_kernel")],
    ),
]

# Tier 3's growth (cli/src/errdist.rs `grow_step`): the flip bound drives it where Tier 3's own
# target asks for fewer inputs (N-ERRDIST-SEL; the review of the heavy-tier fixes: since MISS
# 139 no test took the override).
MUTANTS += [
    # The flip bound never asks for inputs.
    SrcMutant(
        "flip-override-off",
        "crates/cli/src/errdist.rs",
        "                (want, bud, for_flips) = (w, budget, true);",
        "                (want, bud, for_flips) = (want, bud, false);",
        [],
        [("ptxv", "bin:ptxv", "errdist::tests::the_flip_bound_drives_growth_where_tier3_is_fine")],
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
    # groups.rs `explore_meet`: `explore` inside a second smem::two_pass again, so an explored
    # SASS group whose weak verdict goes to the strong pass runs the weak pass twice (the
    # strong pass's `explore` runs it again) and its verdict carries the S-SMEM-STRONG note
    # twice.
    SrcMutant(
        "groups-meet-two-pass-twice",
        "crates/sim/src/groups.rs",
        "    let v = crate::explore::explore(m, max_states);\n",
        "    let v = crate::smem::two_pass(m, |m| crate::explore::explore(m, max_states));\n",
        [],
        [("ptxv-sim", "fixtures", "explored_launch_group_runs_each_smem_pass_once")],
    ),
    # The first-race option of group checks (groups::Opts::first_race, opt-in): not passed to
    # the explored group's machine, so the search runs on past the race and is rerun.
    SrcMutant(
        "groups-first-race-dropped",
        "crates/sim/src/groups.rs",
        "                start.race_stop = opts.first_race;\n",
        "                start.race_stop = false;\n",
        [],
        [("ptxv-sim", "fixtures", "group_first_race_option_stops_at_the_race")],
    ),
    # explore.rs: a search stopped at its first race (Machine::race_stop) is rerun in the
    # plain model after the race.
    SrcMutant(
        "first-race-rerun",
        "crates/sim/src/explore.rs",
        "        && !m0.race_stop\n        && std::env::var(\"PTXV_NO_RACE_RERUN\").is_err()\n",
        "        && std::env::var(\"PTXV_NO_RACE_RERUN\").is_err()\n",
        [],
        [("ptxv-sim", "fixtures", "group_first_race_option_stops_at_the_race")],
    ),
    # explore.rs: a search stopped at its first race before any terminal state gets the
    # deterministic depth run.
    SrcMutant(
        "first-race-depth-run",
        "crates/sim/src/explore.rs",
        "        && v.terminal_states == 0\n        && !m0.race_stop\n",
        "        && v.terminal_states == 0\n",
        [],
        [("ptxv-sim", "fixtures", "group_first_race_option_stops_at_the_race")],
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
        "        let late = k > 0 && {\n            let prev = &mb.done_land[((k - 1) % 2) as usize];\n",
        "        let late = false && k > 0 && {\n            let prev = &mb.done_land[((k - 1) % 2) as usize];\n",
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
        "crates/sim/src/seg/binding.rs",
        "            register.pred_bit(read_sval(source, inputs.args, inputs.prior))",
        "            read_sval(source, inputs.args, inputs.prior) != 0",
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
        "crates/ir/src/guard_certificate.rs",
        "        if !matches!(register.file, File::P | File::UP) {",
        "        if !matches!(register.file, File::P | File::UP | File::R | File::UR) {",
        [],
        [("ptxv-sim", "lib", "explore::sign_pred_tests::const_guard_refuses_sign_bit")],
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


# MISS 134 follow-up: the report names an order-dependent chain as such (one thread's mixed adds
# are in program order, not schedule-dependent), and a 16-bit bulk-reduction chain that changes
# format is unordered rather than cut short silently.
MUTANTS += [
    # The header back to "schedule-dependent order".
    SrcMutant(
        "unordered-header",
        "crates/cli/src/equiv.rs",
        "            println!(\"    order-dependent atomic adds (not L0): {u}\");",
        "            println!(\"    schedule-dependent order: {u}\");",
        [],
        ATOM_FTZ_TESTS,
    ),
    # A format change inside a 16-bit chain read as one format's add.
    SrcMutant(
        "atom16-mix-silent",
        "crates/sim/src/equiv.rs",
        "                    if g != f16 {\n                        // Not reached:",
        "                    if g != f16 && false {\n                        // Not reached:",
        [],
        [("ptxv-sim", "lib", "equiv::soundness_tests::atomic16_format_change_is_unordered")],
    ),
]


# MISS 135: L0~0 only where the peeled x + (+0) keeps x's NaNs (x holds no NaN but the canonical
# one; fixtures/c_equiv_zero_sign_nan). Each mutant peels an operand that can hold a NaN payload.
# l0z-nan-leaf changes no level end to end: a NaN constant's non-finite coefficient refuses every
# level first. l0z-nan-choice is caught end to end on the PTX front: ptxas folds max.f32(a, a)
# into a copy, which keeps a's NaN payloads, and the mutant's L0~0 for it plus +0 is false on
# GB300 (fixtures/c_equiv_zero_sign_fmnmx, fmaxaa). In segment mode, where selects and guarded
# stores can pass a payload, x + (+0) is part of a slice and is not peeled
# (fixtures/c_equiv_zero_sign_choice). The unit test catches both mutants.
ZERO_NAN_UNIT = [("ptxv-sim", "lib", "equiv::soundness_tests::zero_peel_needs_canonical_nans")]
ZERO_HW_CLI = [("ptxv", "equiv", "zero_peel_takes_hardware_maxima")]
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
        ZERO_NAN_UNIT + ZERO_HW_CLI,
    ),
    # A hardware FMNMX's max peeled only where its operands are again (the old rule).
    SrcMutant(
        "l0z-fmnmx-hw-off",
        "crates/sim/src/equiv.rs",
        "                NodeKind::Max { .. } if self.ns.fmnmx_hw.contains(&k) => {}",
        "                NodeKind::Max { .. } if false && self.ns.fmnmx_hw.contains(&k) => {}",
        [],
        ZERO_NAN_UNIT + ZERO_HW_CLI,
    ),
    # A PTX max.f32's node taken for a hardware FMNMX.
    SrcMutant(
        "l0z-fmnmx-hw-any-front",
        "crates/sim/src/num.rs",
        "        let sass = self.k.front == ptxv_ir::Front::Sass;",
        "        let sass = true;",
        [],
        ZERO_HW_CLI,
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

# Forum post 76 (MISS 174 / 175): every condition of the CAS retry-loop recognizer
# (`ptxv_ir::casloop::cas_retry_loop`, E-SPIN) and of the seed-load rule (`seed_load`,
# E-CAS-SEED). Each mutant drops one condition; `spin::cas_tests` holds a SASS loop that needs it.
CAS_POLL = [("ptxv-sim", "lib", "spin::cas_tests::cas_loops_that_poll_stay_polls")]
CAS_SEED = [("ptxv-sim", "lib", "spin::cas_tests::seed_loads_used_elsewhere_stay_plain")]
CASLOOP = "crates/ir/src/casloop.rs"
MUTANTS += [
    SrcMutant(
        "casretry-refresh-any",
        CASLOOP,
        "    if c <= l || copy_src(&code[c].op) != Some((cmp, old)) || code[c].op.dst_regs() != [cmp] {",
        "    if c <= l || code[c].op.dst_regs() != [cmp] {",
        [],
        CAS_POLL,
    ),
    SrcMutant(
        "casretry-cmp-once",
        CASLOOP,
        "    let [c] = cmp_writes[..] else {",
        "    let Some(&c) = cmp_writes.last() else {",
        [],
        CAS_POLL,
    ),
    SrcMutant(
        "casretry-old-kept",
        CASLOOP,
        "    if (l + 1..=b).any(|q| code[q].op.dst_regs().contains(&old)) {",
        "    if false {",
        [],
        CAS_POLL,
    ),
    SrcMutant(
        "casretry-p-once",
        CASLOOP,
        "    let [s] = p_writes[..] else {",
        "    let Some(&s) = p_writes.first() else {",
        [],
        CAS_POLL,
    ),
    SrcMutant(
        "casretry-exit-operands",
        CASLOOP,
        "    if *d != p || s <= l || s >= c || !and.is_always() || !operands || !repeats_on_failure {",
        "    if *d != p || s <= l || s >= c || !and.is_always() || !repeats_on_failure {",
        [],
        CAS_POLL,
    ),
    SrcMutant(
        "casretry-exit-direction",
        CASLOOP,
        "    if *d != p || s <= l || s >= c || !and.is_always() || !operands || !repeats_on_failure {",
        "    if *d != p || s <= l || s >= c || !and.is_always() || !operands {",
        [],
        CAS_POLL,
    ),
    SrcMutant(
        "casretry-exit-and",
        CASLOOP,
        "    if *d != p || s <= l || s >= c || !and.is_always() || !operands || !repeats_on_failure {",
        "    if *d != p || s <= l || s >= c || !operands || !repeats_on_failure {",
        [],
        CAS_POLL,
    ),
    SrcMutant(
        "casretry-exit-before-refresh",
        CASLOOP,
        "    if *d != p || s <= l || s >= c || !and.is_always() || !operands || !repeats_on_failure {",
        "    if *d != p || s <= l || !and.is_always() || !operands || !repeats_on_failure {",
        [],
        CAS_POLL,
    ),
    SrcMutant(
        "casretry-addr-invariant",
        CASLOOP,
        "    if (h..=b).any(|q| q != l && code[q].op.dst_regs().iter().any(|r| aregs.contains(r))) {",
        "    if false {",
        [],
        CAS_POLL,
    ),
    SrcMutant(
        "casretry-carried",
        CASLOOP,
        "            .any(|r| *r != cmp && written.contains(r) && !done.contains(r))",
        "            .any(|r| false && *r != cmp && written.contains(r) && !done.contains(r))",
        [],
        CAS_POLL,
    ),
    SrcMutant(
        "casretry-pure",
        CASLOOP,
        "    if h > l || !(h..b).all(|q| q == l || (code[q].guard.is_always() && pure(&code[q].op))) {",
        "    if h > l || !(h..b).all(|q| q == l || code[q].guard.is_always()) {",
        [],
        CAS_POLL,
    ),
    SrcMutant(
        "casretry-unguarded",
        CASLOOP,
        "    if h > l || !(h..b).all(|q| q == l || (code[q].guard.is_always() && pure(&code[q].op))) {",
        "    if h > l || !(h..b).all(|q| q == l || pure(&code[q].op)) {",
        [],
        CAS_POLL,
    ),
    SrcMutant(
        "seed-offset",
        CASLOOP,
        "        || addr.off != caddr.off",
        "",
        [("e_cas_retry", "seedaddr", "sass", [])],
        CAS_SEED,
    ),
    SrcMutant(
        "seed-base",
        CASLOOP,
        "        || addr.base != caddr.base\n",
        "\n",
        [],
        CAS_SEED,
    ),
    SrcMutant(
        "seed-space",
        CASLOOP,
        "        || space != cspace",
        "",
        [],
        CAS_SEED,
    ),
    SrcMutant(
        "seed-guard",
        CASLOOP,
        "        || !code[s].guard.is_always()",
        "",
        [],
        CAS_SEED,
    ),
    SrcMutant(
        "seed-width",
        CASLOOP,
        "        bytes: 4,\n        addr,\n        sync,",
        "        addr,\n        sync,",
        [],
        CAS_SEED,
    ),
    SrcMutant(
        "seed-gap-reads",
        CASLOOP,
        "            && !reads(&code[q]).contains(&cr.cmp)",
        "",
        [("e_cas_retry", "seedused", "sass", [])],
        CAS_SEED,
    ),
    SrcMutant(
        "seed-gap-pure",
        CASLOOP,
        "            && (pure(&code[q].op) || matches!(code[q].op, Op::Bssy { .. }))",
        "",
        [],
        CAS_SEED,
    ),
    SrcMutant(
        "seed-gap-addr",
        CASLOOP,
        "            && !code[q].op.dst_regs().iter().any(|r| aregs.contains(r))",
        "",
        [],
        CAS_SEED,
    ),
    SrcMutant(
        "seed-enter",
        CASLOOP,
        "            Op::Bra { target, .. } => inside(*target) && !(q == cr.branch && *target == cr.head),",
        "            Op::Bra { .. } => false,",
        [],
        CAS_SEED,
    ),
    # The lifts and their semantics (probes/half_add = GB300 on both fronts).
    SrcMutant(
        "half-atom-generic-type",
        "crates/sass/src/lib.rs",
        "                op: if has(\"F16x2\") {\n                    AtomOp::AddF16x2",
        "                op: if !has(\"F16x2\") {\n                    AtomOp::AddF16x2",
        [],
        [("ptxv-sim", "fixtures", "probe_half_add_matches_hardware")],
    ),
    SrcMutant(
        "half-atomg-f16x2",
        "crates/sass/src/lib.rs",
        "        [\"ADD\", \"F16x2\", \"RN\"] => Some((AtomOp::AddF16x2, 4)),",
        "        [\"ADD\", \"F16x2\", \"RN\"] => Some((AtomOp::AddBf16x2, 4)),",
        [],
        [("ptxv-sim", "fixtures", "probe_half_add_matches_hardware")],
    ),
    SrcMutant(
        "half-atom-f16-semantics",
        "crates/sim/src/atomic_exec.rs",
        '        fp::f16_fma_rn(old, 0x3c00, value)\n',
        '        fp::bf16_fma(old, 0x3c00, value)\n',
        [],
        [("ptxv-sim", "fixtures", "probe_half_add_matches_hardware")],
    ),
    SrcMutant(
        "half-ptx-f16x2",
        "crates/ptx/src/lib.rs",
        "                    (true, true) => AtomOp::AddF16x2,",
        "                    (true, true) => AtomOp::AddBf16x2,",
        [],
        [("ptxv-sim", "fixtures", "probe_half_add_matches_hardware")],
    ),
    # A faulting thread left parked-on: the derived deadlock of e_atom_half exact_size returns.
    SrcMutant(
        "fault-release",
        "crates/sim/src/exec.rs",
        "                if let Err(e) = self.release_after_stop(&members) {",
        "                if let Err(e) = Ok::<(), String>(()) {",
        [("e_atom_half", "exact_size", "sass", [])],
        [("ptxv-sim", "fixtures", "half_atomics_emulated_and_packed")],
    ),
    # MISS 171: two NaNs with different payloads counted bitwise equal again.
    SrcMutant(
        "compare-nan-bitwise",
        "crates/cli/src/compare.rs",
        "        if ra[i] == rb[i] {",
        "        if ra[i] == rb[i] || (x.is_nan() && y.is_nan()) {",
        [],
        [("ptxv", "bin:ptxv", "compare::tests::nan_payloads_are_not_bitwise_equal")],
    ),
    # MISS 174 / 175 (RHA hole A): the exit branch's `.U`, second condition and `.DIV` forms
    # accepted as a CAS retry loop's plain `@P BRA` again.
    SrcMutant(
        "casretry-bra-plain",
        CASLOOP,
        "        all: false,\n        cond: None,\n        div: None,\n    } = &code[b].op",
        "        ..\n    } = &code[b].op",
        [],
        CAS_POLL,
    ),
    # MISS 171 (RHA): raw codes of two output dtypes compared and counted bitwise equal again.
    SrcMutant(
        "compare-dtype-mixed",
        "crates/cli/src/compare.rs",
        "    let mixed = dt[0] != dt[1];",
        "    let mixed = false;",
        [],
        [("ptxv", "bin:ptxv", "compare::tests::mixed_dtype_codes_have_no_bitwise_count")],
    ),
    # A generic access to the (unmapped) local window reported as a stray global address again.
    SrcMutant(
        "fault-local-window",
        "crates/sim/src/exec.rs",
        "                            } else if space == Space::Generic\n                                && a >> 32 == LOCAL_GENERIC_BASE >> 32\n",
        "                            } else if false\n",
        [],
        [("ptxv-sim", "fixtures", "generic_half_atomics_on_local_memory_fault")],
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
        "    let marks = run.m.round16_marks().map(|v| vec![Some(ptxv_case::Round16::Bf16Next); v.len()]);\n",
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
        "                    if cmp || a.old_input_derived {\n",
        "                    if false && cmp || a.old_input_derived {\n",
        [],
        [EQUIV_CAS],
    ),
]

# MISS 160: SASS claims of a kernel whose PTX the PTX front-end refuses, and the PTX forms it now
# takes (fixtures/c_num_fold_ptxref_f32, c_atom_dec_cas16).
NUM_PTXREF = ("ptxv", "num_untracked", "sass_claims_need_a_ptx_front_that_runs")
MUTANTS += [
    # The SASS front claims alone again where the PTX front-end refuses the PTX (variant
    # refused: 32 claims at 0.50 ulp, 19 under-claimed).
    SrcMutant(
        "num-ptx-refused",
        "crates/cli/src/main.rs",
        "                ptx_refused.as_ref().filter(|_| f == \"sass\").map(|why| {\n",
        "                ptx_refused.as_ref().filter(|_| false).map(|why| {\n",
        [],
        [NUM_PTXREF],
    ),
    # atom.dec refused by the PTX front-end again (variant good: no PTX run).
    SrcMutant(
        "ptx-atom-dec",
        "crates/ptx/src/lib.rs",
        "                        | \"dec\"\n",
        "",
        [],
        [NUM_PTXREF, ("ptxv", "gpu_compare", "ptx_atom_dec_and_cas16_match_gb300")],
    ),
    # atom.cas.b16 refused again.
    SrcMutant(
        "ptx-cas16",
        "crates/ptx/src/lib.rs",
        "            let cas16 = bits == 16 && op == AtomOp::Cas;\n",
        "            let cas16 = false;\n",
        [],
        [("ptxv", "gpu_compare", "ptx_atom_dec_and_cas16_match_gb300")],
    ),
]

# MISS 161: the ulp figure at +-FLT_MAX (fixtures/c_num_fltmax_f32).
MUTANTS += [
    # FLT_MAX's ulp is its distance to +inf again: the even lanes' figure is 0.
    SrcMutant(
        "ulp-top-binade",
        "crates/sim/src/num.rs",
        "    if a == f32::MAX {\n        return (a - f32::from_bits(a.to_bits() - 1)) as f64;\n    }\n",
        "",
        [],
        [("ptxv", "num_untracked", "ulp_figure_at_flt_max_uses_the_top_binade")],
    ),
]

# MISS 162: a calibrated lane is within one ulp only of the other rounding of its sum
# (fixtures/r1h_nvls_bf16 copied with constructed captures).
MUTANTS += [
    # Either neighbour of the simulated value passes again.
    SrcMutant(
        "round16-direction",
        "crates/case/src/lib.rs",
        "                            mark(e).is_some_and(|m| w == m.other(g)) && ulp_class(&b.dtype, g, w)",
        "                            mark(e).is_some() && ulp_class(&b.dtype, g, w)",
        [],
        [
            ("ptxv", "gpu_compare", "only_the_other_rounding_of_a_calibrated_lane_is_within_one_ulp"),
            ("ptxv", "equiv", "multimem_lane_past_the_simulated_value_differs_from_its_capture"),
            ("ptxv-case", "gpu_compare", "only_the_other_rounding_of_the_sum_is_within_one_ulp"),
            ROUND16_CASE,
        ],
    ),
    # The other rounding taken on the wrong side: a lane rounded toward zero is marked as if
    # rounded away.
    SrcMutant(
        "round16-side",
        "crates/sim/src/multimem.rs",
        "                calibrated[i] = Some(!up);\n",
        "                calibrated[i] = Some(up);\n",
        [],
        [
            ("ptxv", "gpu_compare", "only_the_other_rounding_of_a_calibrated_lane_is_within_one_ulp"),
            ("ptxv", "equiv", "multimem_lane_past_the_simulated_value_differs_from_its_capture"),
            ROUND16_LANES,
        ],
    ),
]

# MISS 177 (defense in depth): an output whose node does not hold its simulated bits is untracked
# in equiv and errdist (fixtures/c_equiv_atom64, the unit test).
EQUIV_STALE = ("ptxv", "equiv", "stale_output_nodes_count_as_untracked")
MUTANTS += [
    # Stale nodes kept: an output's node is used whatever value it holds.
    SrcMutant(
        "equiv-node-bits",
        "crates/cli/src/equiv.rs",
        "            let stale = node.is_some_and(|n| nodes[n as usize].hw != *bits);\n",
        "            let stale = false;\n",
        [],
        [("ptxv", "bin:ptxv", "equiv::captured_tests::stale_output_nodes_are_dropped"), EQUIV_STALE],
    ),
]

# MISS 177: every word an atomic covers loses its node in equiv and errdist
# (fixtures/c_equiv_atom_words; the probes whose stale node equals the output at the captured inputs
# are past the node check, so only the fix catches them).
EQUIV_WORDS = ("ptxv", "equiv", "every_word_an_atomic_covers_loses_its_node")
MUTANTS += [
    # Only the word at the atomic's address loses its node again: ref_exch64same and
    # ref_cas64maybe are L0 PROVEN against cand_copy.
    SrcMutant(
        "equiv-atom-words",
        "crates/sim/src/num.rs",
        "                let words = ((k.2 & !3)..k.2 + a.bytes as u64).step_by(4);\n",
        "                let words = (k.2..k.2 + 1).step_by(4);\n",
        [],
        [EQUIV_WORDS],
    ),
    # A 16-bit atomic on a word's upper half leaves the word's node again (num_atom_half):
    # ref_bf16hisame is L0 PROVEN.
    SrcMutant(
        "equiv-atom-bf16-word",
        "crates/sim/src/num.rs",
        "        if !ns.rounding_taint {\n            let w = (k.0, k.1, k.2 & !3);\n",
        "        if false {\n            let w = (k.0, k.1, k.2 & !3);\n",
        [],
        [EQUIV_WORDS],
    ),
]

# MISS 205: an st.async word takes the stored value's numerics node and taint (probes/stasync_num_shadow,
# fixtures/c_num_stasync_f32).
STASYNC_EQ = ("ptxv", "equiv", "stasync_words_take_the_stored_value")
STASYNC_NUM = ("ptxv", "num_untracked", "stasync_words_take_the_stored_value_node")
MUTANTS += [
    # The local st.async (one step) writes no node again: c1_ref_stasyncw is L0 PROVEN against
    # cand_copy, and num claims abs_err 0 at c_num_stasync_f32's rounded elements.
    SrcMutant(
        "stasync-num-local",
        "crates/sim/src/exec.rs",
        "                self.num_st(t, v, loc, &data, &origins, pc);\n                if self.round16.is_some() {\n                    self.round16_st(t, v, n, &[loc]);\n",
        "                if self.round16.is_some() {\n                    self.round16_st(t, v, n, &[loc]);\n",
        [],
        [STASYNC_EQ, STASYNC_NUM],
    ),
    # A remote st.async lands without its value's node and taint again: c2_ref_rsaw and
    # c2_ref_rsaz are L0 PROVEN.
    SrcMutant(
        "stasync-num-remote",
        "crates/sim/src/exec.rs",
        "                if let Some(snap) = l.snaps.get(k) {\n",
        "                if let Some(snap) = l.snaps.get(k).filter(|_| false) {\n",
        [],
        [STASYNC_EQ],
    ),
]


# S-INIT: the success predicate of a compare-and-swap loop.
MUTANTS += [
    # `ATOMS.CAST.SPIN P` over undefined old bytes writes P as defined again
    # (fixtures/s_init_cas_spin: the loop's back branch no longer reads an undefined P).
    SrcMutant(
        "cas-spin-pred-defined",
        "crates/sim/src/exec.rs",
        '                    if old_undef && *op == AtomOp::CasSpin {\n                        self.stale = Some(None);\n                    }\n                    self.wp(t, *p, outcome.predicate);',
        '                    self.wp(t, *p, outcome.predicate);',
        [],
        [("ptxv-sim", "fixtures", "cas_spin_success_predicate_of_undefined_bytes_is_undefined")],
    ),
]


# MISS 204 (interim): an output match on a case whose output check is vacuous (no compared byte of
# any output differs from the buffer's initial bytes) is no evidence: fb/kernel_db records it as
# null.
VACUOUS_PY = ("fb/kernel_db", "test_run_collect.py", "VacuousOutputCheck")
MUTANTS += [
    # run.py `apply_output_check`: the match stays true again.
    SrcMutant(
        "kdb-vacuous-match-kept",
        "fb/kernel_db/run.py",
        '            chk["output_matches_gpu"] = None\n',
        "            pass\n",
        [],
        py=[VACUOUS_PY],
    ),
    # run.py `output_unwritten`: expect_offset ignored (a stored prefix compared from byte 0).
    SrcMutant(
        "kdb-vacuous-offset",
        "fb/kernel_db/run.py",
        '            off = int(b.get("expect_offset", 0) or 0)\n            if _differs(',
        "            off = 0\n            if _differs(",
        [],
        py=[VACUOUS_PY],
    ),
    # run.py `_differs`: expect_ignore ranges compared like any other byte.
    SrcMutant(
        "kdb-vacuous-ignore",
        "fb/kernel_db/run.py",
        "                            if not any(s <= at < e for s, e in ranges):\n"
        "                                return True\n"
        "                done += len(a)\n"
        "                step = min(step * 4, 1 << 24)\n",
        "                            return True\n"
        "                done += len(a)\n"
        "                step = min(step * 4, 1 << 24)\n",
        [],
        py=[VACUOUS_PY],
    ),
    # run.py `output_unwritten`: the [[devices]] entries' outputs are not read.
    SrcMutant(
        "kdb-vacuous-devices",
        "fb/kernel_db/run.py",
        '    for scope in [m] + list(m.get("devices") or []):\n',
        "    for scope in [m]:\n",
        [],
        py=[VACUOUS_PY],
    ),
    # run.py `output_unwritten`: an undef output counts as unwritten.
    SrcMutant(
        "kdb-vacuous-undef",
        "fb/kernel_db/run.py",
        "            else:\n                return n, o\n            off = int(",
        "            else:\n                f = None\n            off = int(",
        [],
        py=[VACUOUS_PY],
    ),
    # run.py `store_local`: a cluster path is not re-rooted under this machine's store.
    SrcMutant(
        "kdb-vacuous-store-local",
        "fb/kernel_db/run.py",
        '    if d and not os.path.isdir(d) and "/ptxv-kdb/" in str(d):\n',
        "    if False:\n",
        [],
        py=[VACUOUS_PY],
    ),
    # export_records.py `apply_output_checks`: check_ptx is exported as stored again.
    SrcMutant(
        "kdb-vacuous-export-ptx",
        "fb/kernel_db/export_records.py",
        "    if isinstance(cp, dict):\n        targets.append(",
        "    if False:\n        targets.append(",
        [],
        py=[VACUOUS_PY],
    ),
]

# MISS 190: the verdict cache key covers the CUDA tools ptxv runs (CUDA_HOME's ptxas, nvdisasm,
# cuobjdump), so a verdict computed with one toolkit does not serve another (probes/
# miss190_vcache_cuda_home).
MUTANTS += [
    # vcache.rs `key_with`: the toolkit is out of the key again.
    SrcMutant(
        "vcache-toolkit",
        "crates/cli/src/vcache.rs",
        "    eat_toolkit(&mut h, cuda);\n",
        "",
        [],
        [("ptxv", "bin:ptxv", "vcache::tests::key_follows_the_toolkit")],
    ),
]

# MISS 191: a check row names the launch order its verdict was computed under, with the reading
# of an unknown order (probes/miss191_order_reading).
MUTANTS += [
    # main.rs `order_row`: an unknown order reads as concurrent in the row, whatever the setting.
    SrcMutant(
        "order-row-reading",
        "crates/cli/src/main.rs",
        '            Ok("ordered") => "ordered",\n',
        '            Ok("ordered") => "concurrent",\n',
        [],
        [("ptxv", LO_CLI, "check_rows_record_the_reading_of_an_unknown_order")],
    ),
    # fb/kernel_db/run.py `parse_check`: the row's launch order is not stored with the check.
    SrcMutant(
        "kdb-order-reading",
        "fb/kernel_db/run.py",
        '        chk["launch_order"] = row["launch_order"]\n',
        "        pass\n",
        [],
        py=[("fb/kernel_db", "test_run_collect.py", "LaunchOrderReading")],
    ),
]

# MISS 189: an unknown order of several launches under the strict reading is also checked in the
# list's order, for its deadlocks (probes/miss189_order_hang: good is deadlock, clean before).
L189 = [
    ("../probes/miss189_order_hang", "good", "ptx", []),
    ("../probes/miss189_order_hang", "good", "sass", []),
]
L189_SIM = ("ptxv-sim", "fixtures", "miss189_order_hang_list_order_deadlock")
L189_PASS = ("ptxv-sim", "fixtures", "miss189_list_order_pass_outcomes")
L189_DUP = ("ptxv-sim", "fixtures", "miss189_list_order_deadlock_the_strict_reading_has_is_not_added")
MUTANTS += [
    # sim lib.rs `serialized_order_deadlocks`: no deadlock of the list's order joins the verdict.
    SrcMutant(
        "list-order-deadlocks",
        "crates/sim/src/lib.rs",
        '    for f in vo.findings.iter().filter(|f| f.kind == "deadlock") {\n',
        '    for f in vo.findings.iter().filter(|f| f.kind == "deadlock" && false) {\n',
        L189,
        [L189_SIM, L189_PASS],
    ),
    # the list's order built under the strict reading again: the pass sees the overlap.
    SrcMutant(
        "list-order-reading",
        "crates/sim/src/lib.rs",
        "    let launch = build_launch_reading(case, k, true)?;\n",
        "    let launch = build_launch_reading(case, k, false)?;\n",
        L189,
        [L189_SIM],
    ),
    # a deadlock under the list's order keeps E-WELLSYNC: the verdict stays "certified".
    SrcMutant(
        "list-order-wellsync",
        "crates/sim/src/lib.rs",
        '        v.assumptions.retain(|a| a != "E-WELLSYNC");\n',
        "        v.assumptions.retain(|_| true);\n",
        [],
        [L189_PASS, ("ptxv", "compare", "compare_list_order_deadlock_blocks")],
    ),
    # a pass that does not finish leaves the verdict complete.
    SrcMutant(
        "list-order-incomplete",
        "crates/sim/src/lib.rs",
        "    } else if !vo.complete {\n        v.complete = false;\n",
        "    } else if !vo.complete {\n",
        [],
        [L189_PASS],
    ),
    # a deadlock the strict reading reports too joins the verdict again, as `list-order:`
    # (probes/miss189_dupsig).
    SrcMutant(
        "list-order-strict-dup",
        "crates/sim/src/lib.rs",
        "        if v.findings.iter().any(|x| x.sig == f.sig) {\n"
        "            strict += 1;\n"
        "            continue;\n"
        "        }\n",
        "",
        [],
        [L189_DUP],
    ),
    # main.rs `cmd_check`: the pass is not run.
    SrcMutant(
        "check-list-order",
        "crates/cli/src/main.rs",
        "            ptxv_sim::serialized_order_deadlocks(&mut verdict, case, &k, &mut |m| {\n"
        '                check(&|| m.clone(), "list order: ")\n'
        "            })?;\n",
        "",
        L189,
    ),
    # equiv.rs `prepare_with` (equiv's and errdist's precondition): the pass is not run.
    SrcMutant(
        "equiv-list-order",
        "crates/cli/src/equiv.rs",
        "        ptxv_sim::serialized_order_deadlocks(v, &case, &kern, &mut |m| run(m))\n",
        "        let _ = (v, run);\n        Ok(())\n",
        [],
        [("ptxv", LO_CLI, "equiv_precondition_checks_the_list_order_for_deadlocks")],
    ),
    # compare.rs `side`: the pass is not run.
    SrcMutant(
        "compare-list-order",
        "crates/cli/src/compare.rs",
        "            ptxv_sim::serialized_order_deadlocks(&mut v, &case, &k, &mut |m| {\n"
        "                verdict_of(&|| m.clone()).0\n"
        "            })?;\n",
        "            let _ = &mut v;\n",
        [],
        [("ptxv", "compare", "compare_list_order_deadlock_blocks")],
    ),
]


# MISS 142: an f32 atomic add flushes subnormals in global memory only; the SASS generic f32 add
# is not performed at a shared-window address, and QSPC answers for the shared window
# (probes/atom_f32_space, probes/qspc_window, fixtures/c_equiv_atomic_space). Each mutant
# flushes a shared-memory add, performs the generic add there, or misanswers a QSPC.
ATOM_SPACE_PROBES = [("ptxv-sim", "fixtures", "probe_atom_f32_space_matches_hardware")]
ATOM_SPACE_EQUIV = [("ptxv", "equiv", "shared_atomic_adds_do_not_flush")]
QSPC_PROBE = [("ptxv-sim", "fixtures", "probe_qspc_window_matches_hardware")]
MUTANTS += [
    # Every f32 atomic add flushing again (the executor's value).
    SrcMutant(
        "atom-shared-flush",
        "crates/sim/src/exec.rs",
        '                        == ptxv_memory_write::atomic_exec::Operation::AddF32Ftz\n                        && matches!(loc, Loc::Shared { .. })\n',
        '                        == ptxv_memory_write::atomic_exec::Operation::AddF32Ftz\n                        && false\n',
        [],
        ATOM_SPACE_PROBES
        + [("ptxv-sim", "wellsync", "fadd_orders_in_shared_memory_keep_subnormals")]
        + ATOM_SPACE_EQUIV,
    ),
    # The numerics node of a shared-memory add marked flushing: the PTX pairs are L0 again.
    SrcMutant(
        "atom-shared-numftz",
        "crates/sim/src/exec.rs",
        "                        n as u32,\n                        !matches!(loc, Loc::Shared { .. }),\n                    ));",
        "                        n as u32,\n                        true,\n                    ));",
        [],
        ATOM_SPACE_EQUIV,
    ),
    # The SASS generic ATOM.E.ADD.F32.FTZ performed at a shared-window address (P true, the
    # fallback skipped).
    SrcMutant(
        "atom-generic-shm-performed",
        "crates/sim/src/exec.rs",
        "                if self.k.front != Front::Ptx\n                    && *space == Space::Generic\n                    && *op != AtomOp::CasSpin",
        "                if false\n                    && *space == Space::Generic\n                    && *op != AtomOp::CasSpin",
        [],
        ATOM_SPACE_EQUIV,
    ),
    # QSPC.E.D only for the CTA's own rank.
    SrcMutant(
        "qspc-cluster-own",
        "crates/sim/src/exec.rs",
        "                    *cluster || target == me",
        "                    target == me",
        [],
        QSPC_PROBE,
    ),
    # QSPC.E.S for any rank of the cluster.
    SrcMutant(
        "qspc-cta-any",
        "crates/sim/src/exec.rs",
        "                    *cluster || target == me",
        "                    true",
        [],
        QSPC_PROBE + ATOM_SPACE_PROBES,
    ),
    # C-FADD's fold over the orders flushing in shared memory too.
    SrcMutant(
        "cfadd-shared-flush",
        "crates/sim/src/wellsync.rs",
        "                    key.0 != 0,\n                );",
        "                    true,\n                );",
        [],
        [("ptxv-sim", "wellsync", "fadd_orders_in_shared_memory_keep_subnormals")],
    ),
]


# MISS 143: only flushing fp32 atomic adds carry the absolute flush term, and L2 compares the
# absolute terms (fixtures/c_equiv_atomic_ftz reversed pairs). Each mutant reports a flushing
# candidate L2 against a non-flushing reference again.
L2_ABS_TESTS = [
    ("ptxv-sim", "lib", "equiv::soundness_tests::l2_compares_the_flush_term"),
    ("ptxv", "equiv", "l2_needs_the_flush_term"),
]
MUTANTS += [
    # The flush term charged to every add of the chain, flushing or not.
    SrcMutant(
        "abs-ftz-all-adds",
        "crates/sim/src/equiv.rs",
        "                let nf = vals.iter().filter(|v| v.1).count();",
        "                let nf = vals.len();",
        [],
        L2_ABS_TESTS,
    ),
    # L2 without the absolute terms.
    SrcMutant(
        "l2-abs-off",
        "crates/sim/src/equiv.rs",
        "        if cc.abs_extra > cr.abs_extra {",
        "        if false && cc.abs_extra > cr.abs_extra {",
        [],
        L2_ABS_TESTS,
    ),
]


# MISS 144: a narrow-format tensor-core step whose f32 result leaves the range is reported as
# not modeled (GB300 gives +-Inf; probes/mma_bs/ovf). Each mutant lets such a run stay clean or
# reports it only through the next step's accumulator.
MMA_OVF_TESTS = [("ptxv-sim", "fixtures", "probe_mma_bs_overflow_is_unmodeled")]
MUTANTS += [
    # Round toward zero saturating to +-FLT_MAX again.
    SrcMutant(
        "mma-ovf-saturate",
        "crates/sim/src/mma_lp.rs",
        "    let val = if qe + tgt.mbits > tgt.emax {\n        f64::INFINITY\n    } else {",
        "    let val = if qe + tgt.mbits > tgt.emax {\n        if rz {\n            (2.0 - 2f64.powi(-tgt.mbits)) * 2f64.powi(tgt.emax)\n        } else {\n            f64::INFINITY\n        }\n    } else {",
        [],
        MMA_OVF_TESTS,
    ),
    # The overflowing step itself not reported.
    SrcMutant(
        "mma-ovf-unreported",
        "crates/sim/src/exec.rs",
        "        if outside.is_none() && !f16_d && results.iter().any(|&w| f32::from_bits(w).is_infinite()) {",
        "        if false && results.iter().any(|&w| f32::from_bits(w).is_infinite()) {",
        [],
        MMA_OVF_TESTS,
    ),
]


# MISS 145: compare's per-output stats carry the largest absolute term and every headline form
# prints it (the unit test and fixtures/c_equiv_atomic_ftz through `ptxv compare`). Each mutant
# drops it from the stats or from the text again.
HEADLINE_ABS_TESTS = [
    ("ptxv", "bin:ptxv", "compare::tests::the_headline_carries_the_absolute_term"),
    ("ptxv", "compare", "compare_headline_carries_the_absolute_term"),
]
MUTANTS += [
    # The stats without the absolute term (report.json and every headline).
    SrcMutant(
        "stats-abs-off",
        "crates/cli/src/compare.rs",
        "        .filter(|o| o[\"bound_u\"].is_f64())\n        .filter_map(|o| o[\"abs_extra\"].as_f64())",
        "        .filter(|o| o[\"bound_u\"].is_f64() && false)\n        .filter_map(|o| o[\"abs_extra\"].as_f64())",
        [],
        HEADLINE_ABS_TESTS,
    ),
    # The headline forms without " + A".
    SrcMutant(
        "headline-abs-off",
        "crates/cli/src/compare.rs",
        "        Some(a) if a > 0.0 => format!(\" + {}\", up_e(a, 1)),",
        "        Some(a) if false && a > 0.0 => format!(\" + {}\", up_e(a, 1)),",
        [],
        HEADLINE_ABS_TESTS,
    ),
]

MUTANTS += [
    # explore.rs `keep_found`: a cut rerun that found every earlier finding itself says again
    # "0 finding(s) ... are kept, unconfirmed by the plain model".
    SrcMutant(
        "keep-found-zero-kept-note",
        "crates/sim/src/explore.rs",
        "    w.notes.push(if kept == 0 {\n",
        "    w.notes.push(if kept == usize::MAX {\n",
        [],
        [("ptxv-sim", "lib", "explore::bar_cut_tests::keep_found_says_whether_the_rerun_found_everything")],
    ),
]

MUTANTS += [
    # run.py `keep_errdist_revs`: a recheck leaves errdist results stored without a ptxv_rev
    # unstamped again, so the new check's rev judges them.
    SrcMutant(
        "kdb-run-recheck-errdist-unstamped",
        "fb/kernel_db/run.py",
        "            n[\"ptxv_rev\"] = rev\n",
        "            pass\n",
        [],
        py=[("fb/kernel_db", "test_run_collect.py", "test_recheck_keeps_the_errdist_builds")],
    ),
    # run.py `audited_check_rev`: the stamp is the replaced check's recorded rev again, not the
    # rev ptxv_rev_audit.json puts in its place, which the export judged the results at.
    SrcMutant(
        "kdb-run-recheck-errdist-unaudited",
        "fb/kernel_db/run.py",
        "    export_records.apply_rev_fix(view, _rev_fixes(path, version))\n",
        "    pass\n",
        [],
        py=[("fb/kernel_db", "test_run_collect.py", "test_recheck_stamps_the_audited_rev")],
    ),
    # run.py `audited_check_rev`: the audit rewrites the record's own check_ptx (kept beside the
    # new check) instead of a copy.
    SrcMutant(
        "kdb-run-audit-rewrites-the-record",
        "fb/kernel_db/run.py",
        "            view[k] = dict(rec[k])  # apply_rev_fix rewrites the check it is given\n",
        "            view[k] = rec[k]\n",
        [],
        py=[("fb/kernel_db", "test_run_collect.py", "test_recheck_stamps_the_audited_rev")],
    ),
    # run.py `run_check`: `run.py check` replaces the check without stamping them.
    SrcMutant(
        "kdb-run-check-errdist-unstamped",
        "fb/kernel_db/run.py",
        "    keep_errdist_revs(rec)\n    rec[\"check\"] = apply_output_check(chk, case)\n",
        "    rec[\"check\"] = apply_output_check(chk, case)\n",
        [],
        py=[("fb/kernel_db", "test_run_collect.py", "test_recheck_keeps_the_errdist_builds")],
    ),
]

MUTANTS += [
    # soundness.py `race_parts`: a record with checks_extra but no check or check_ptx lists only
    # the entries open at int7's tip again, not those open at its extras' own builds.
    SrcMutant(
        "kdb-snd-extras-only-tip-entries",
        "fb/kernel_db/soundness.py",
        "            at_check = at_check.union(*(t for _, t, _ in own))\n",
        "            at_check, own = at_check.union(*(t for _, t, _ in own)), []\n",
        [],
        py=[("fb/kernel_db", "test_soundness.py", "test_checks_extra_only_is_unknown")],
    ),
]

MUTANTS += [
    # wellsync.rs `check_certified_inner`: the certificate fallback explores inside a second
    # smem::two_pass again (the weak pass twice, the S-SMEM-STRONG note twice).
    SrcMutant(
        "certify-fallback-two-pass-twice",
        "crates/sim/src/wellsync.rs",
        "        let mut v = crate::explore::explore(m0, max_states);\n        v.notes\n",
        "        let mut v = crate::smem::two_pass(m0, |m| crate::explore::explore(m, max_states));\n"
        "        v.notes\n",
        [],
        [("ptxv-sim", "fixtures", "certify_fallback_runs_each_smem_pass_once")],
    ),
]

MUTANTS += [
    # explore.rs `explore_one_models_with`: with the first-race option, the verdict no longer
    # says that the race found with the model reductions is unconfirmed by the plain model.
    SrcMutant(
        "first-race-unconfirmed-note-dropped",
        "crates/sim/src/explore.rs",
        "    if m0.race_stop\n        && (peek_model || vip_model && vip_can_poll(&m0))\n",
        "    if !m0.race_stop\n        && (peek_model || vip_model && vip_can_poll(&m0))\n",
        [],
        [("ptxv-sim", "fixtures", "group_first_race_option_says_its_race_is_unconfirmed")],
    ),
    # explore.rs `explore_one_models_with`: the note again whenever value-insensitive polls are
    # on, also for a kernel with no load they can act on (its first search is the plain model's).
    SrcMutant(
        "first-race-note-without-reductions",
        "crates/sim/src/explore.rs",
        "        && (peek_model || vip_model && vip_can_poll(&m0))\n",
        "        && (peek_model || vip_model)\n",
        [],
        [("ptxv-sim", "fixtures", "group_first_race_option_plain_model_race_is_not_unconfirmed")],
    ),
    # explore.rs `vip_can_poll`: every load counts as one value-insensitive polls can act on.
    SrcMutant(
        "vip-can-poll-any-load",
        "crates/sim/src/explore.rs",
        "    m.k.code.iter().any(|i| crate::spin::strong_poll(&i.op))\n",
        "    m.k.code.iter().any(|i| matches!(i.op, ptxv_ir::Op::Ld { .. }))\n",
        [],
        [("ptxv-sim", "fixtures", "group_first_race_option_plain_model_race_is_not_unconfirmed")],
    ),
]


# E-BAR-OVER's walks and SASS `@P BRA.U` (MISS 141).
BRAU_WALK = ("ptxv-sim", "fixtures", "e_barover_walk_follows_both_edges_of_bra_u")
MUTANTS += [
    # The plain walk decides `@P BRA.U` by the thread's own guard again (fixtures/
    # e_barover_brau BRAU clean).
    SrcMutant(
        "barwalk-plain-brau-own-guard",
        "crates/sim/src/explore.rs",
        "                // split, so the thread's own guard does not decide it: both edges (MISS 141).\n"
        "                let decided = match &i.op {\n"
        "                    Op::Bra {\n"
        "                        cond: None,\n"
        "                        any: false,\n"
        "                        tid0_falls: false,\n"
        "                        all: false,\n",
        "                // split, so the thread's own guard does not decide it: both edges (MISS 141).\n"
        "                let decided = match &i.op {\n"
        "                    Op::Bra {\n"
        "                        cond: None,\n"
        "                        any: false,\n"
        "                        tid0_falls: false,\n",
        [("e_barover_brau", "BRAU", "sass", [])],
        [BRAU_WALK],
    ),
    # The register-id walk does (fixtures/e_barover_brau_regid BRAU clean).
    SrcMutant(
        "barwalk-regs-brau-own-guard",
        "crates/sim/src/explore.rs",
        "                // (`@P BRA.U`: both edges, as in the plain walk.)\n"
        "                let decided = match &i.op {\n"
        "                    Op::Bra {\n"
        "                        cond: None,\n"
        "                        any: false,\n"
        "                        tid0_falls: false,\n"
        "                        all: false,\n",
        "                // (`@P BRA.U`: both edges, as in the plain walk.)\n"
        "                let decided = match &i.op {\n"
        "                    Op::Bra {\n"
        "                        cond: None,\n"
        "                        any: false,\n"
        "                        tid0_falls: false,\n",
        [("e_barover_brau_regid", "BRAU", "sass", [])],
        [BRAU_WALK],
    ),
]


# SASS `BRA.U P[, UPT]` (a thread predicate as the operand; MISS 133).
MUTANTS += [
    # Per lane again (probes/bra_u operand forms: pp_u / upt_u differ from GB300 where P is
    # mixed).
    SrcMutant(
        "bra-u-operand-per-lane",
        "crates/sass/src/lib.rs",
        'all: has("U") && !has("ANY") && c.reg.is_some_and(|r| r.file == File::P),',
        "all: false,",
        [],
        [
            ("ptxv-sim", "fixtures", "probe_bra_u_operand_forms_match_hardware"),
            ("ptxv-sass", "lib", "tests::bra_u_operand_all_form_only_for_thread_predicates"),
        ],
    ),
    # A uniform predicate operand (`BRA.U [!]UPn`, `[!]UPT`) all-form too, as before the
    # restriction to thread predicates.
    SrcMutant(
        "bra-u-operand-uniform-all",
        "crates/sass/src/lib.rs",
        'all: has("U") && !has("ANY") && c.reg.is_some_and(|r| r.file == File::P),',
        'all: has("U") && !has("ANY"),',
        [],
        [("ptxv-sass", "lib", "tests::bra_u_operand_all_form_only_for_thread_predicates")],
    ),
]


# E-CPASYNC: a cp.async's global load (MISS 117).
MUTANTS += [
    # Checked at issue only, as the issuer (fixtures/e_cpasync_src_war BAR / OWN clean).
    SrcMutant(
        "cpasync-src-issue-only",
        "crates/sim/src/exec.rs",
        "self.recheck_read(actor, &vc, c.pc, src, c.data.len());",
        "let _ = src;",
        [
            ("e_cpasync_src_war", "BAR", "ptx", []),
            ("e_cpasync_src_war", "OWN", "sass", []),
        ],
        [("ptxv-sim", "fixtures", "cpasync_source_load_may_happen_until_the_wait")],
    ),
]


# E-CPASYNC-MBAR: what the arrive-on of cp.async.mbarrier.arrive releases (MISS 118).
MUTANTS += [
    # The issuer's whole clock again (fixtures/e_cpasync_arrive_release WHOLE / GLOBAL clean).
    SrcMutant(
        "cparrive-whole-clock",
        "crates/sim/src/exec.rs",
        '                self.mbar_note_async_own_with(cta, mbar, op.issuer, op.actor, &op.vc, prior);\n',
        '                let _ = prior;\n                self.mbar_note_async(cta, mbar, op.issuer, op.actor, &op.vc);\n',
        [
            ("e_cpasync_arrive_release", "WHOLE", "ptx", []),
            ("e_cpasync_arrive_release", "GLOBAL", "sass", []),
        ],
        [("ptxv-sim", "fixtures", "cpasync_arrive_on_releases_only_the_copies")],
    ),
    # The arrive-on releases its own copies only again, not the thread's prior copies that had
    # landed at its issue (fixtures/e_cpasync_arrive_prior good races on the PTX front).
    SrcMutant(
        "cparrive-no-prior",
        "crates/sim/src/exec.rs",
        "        vc_join(&mut own, also);\n",
        "        let _ = also;\n",
        [("e_cpasync_arrive_prior", "good", "ptx", [])],
        [("ptxv-sim", "fixtures", "cpasync_arrive_on_releases_prior_copies")],
    ),
    # The arrive-on at its copies' epoch again: a later arrive-on's release of those copies
    # covers this arrive-on too (fixtures/e_cpasync_arrive_order NOWAIT certifies clean).
    SrcMutant(
        "cparrive-prior-covers-arrive",
        "crates/sim/src/exec.rs",
        "                vc_raise(&mut vc, actor, 2);\n",
        "                vc_raise(&mut vc, actor, 1);\n",
        [("e_cpasync_arrive_order", "NOWAIT", "ptx", ["--certify"])],
        [("ptxv-sim", "fixtures", "cpasync_arrive_on_is_not_ordered_after_an_earlier_arrive_on")],
    ),
    # An arrive takes only the copies landed at its issue again, not those of its thread's
    # earlier arrive operations still in flight (fixtures/e_cpasync_arrive_inflight good races).
    SrcMutant(
        "cparrive-prior-landed-only",
        "crates/sim/src/exec.rs",
        "                    if o.issuer == t as u32 && matches!(o.kind, AsyncKind::CpArrive { .. }) {\n"
        "                        vc_join(&mut prior, &vc_single(o.actor, 1));\n",
        "                    if o.issuer == t as u32 && matches!(o.kind, AsyncKind::CpArrive { .. }) && false {\n"
        "                        vc_join(&mut prior, &vc_single(o.actor, 1));\n",
        [("e_cpasync_arrive_inflight", "good", "ptx", [])],
        [("ptxv-sim", "fixtures", "cpasync_arrive_on_releases_prior_copies_in_flight")],
    ),
    # The earlier arrive operations' copies still pending do not land before the later
    # arrive-on: it releases their epoch before their writes exist (good: the reader of slot A
    # can run before the copy lands).
    SrcMutant(
        "cparrive-no-early-landing",
        "crates/sim/src/exec.rs",
        "            let w = std::mem::take(w);\n"
        "            if !w.is_empty() {\n"
        "                self.cp_land(issuer as usize, actor, 1, w);\n"
        "            }\n",
        "            let _ = (w, actor);\n",
        [("e_cpasync_arrive_inflight", "good", "ptx", [])],
        [("ptxv-sim", "fixtures", "cpasync_arrive_on_releases_prior_copies_in_flight")],
    ),
    # The prior copies read at the completion instead of the issue: copies retired after the
    # arrive (by wait_group) are released too (fixtures/e_cpasync_arrive_inflight LATER clean).
    SrcMutant(
        "cparrive-prior-at-completion",
        "crates/sim/src/exec.rs",
        '                self.mbar_note_async_own_with(cta, mbar, op.issuer, op.actor, &op.vc, prior);\n',
        '                let _ = prior;\n                let prior = self.threads[op.issuer as usize].cp_done.clone();\n                self.mbar_note_async_own_with(cta, mbar, op.issuer, op.actor, &op.vc, &prior);\n',
        [
            ("e_cpasync_arrive_inflight", "LATER", "ptx", []),
            ("e_cpasync_arrive_inflight", "LATER", "sass", []),
        ],
        [("ptxv-sim", "fixtures", "cpasync_arrive_on_releases_prior_copies_in_flight")],
    ),
    # The over-arrival test on the memory clock again, not the order of events: DPOR hands
    # fixtures/e_cpasync_arrive_order good (SASS) to the stateful explorer.
    SrcMutant(
        "mbar-late-memory-clock",
        "crates/sim/src/exec.rs",
        "            !vc_covers(vc, prev) && !vc_covers(&self.order_vc_of(actor, vc), prev)\n",
        "            let _ = actor;\n            !vc_covers(vc, prev)\n",
        [],
        [("ptxv-sim", "fixtures", "mbar_late_updates_judged_by_the_order_of_events")],
    ),
]


# S-MBAR-SCOPE: what a bulk copy's complete-tx releases (MISS 119).
MUTANTS += [
    # The issuer's whole clock again (fixtures/e_bulk_ctx_release RELAXED / GLOBAL clean).
    SrcMutant(
        "bulk-ctx-whole-clock",
        "crates/sim/src/exec.rs",
        '                        self.mbar_note_async_own(cta, mbar, op.issuer, op.actor, &op.vc);\n',
        '                        self.mbar_note_async(cta, mbar, op.issuer, op.actor, &op.vc);\n',
        [
            ("e_bulk_ctx_release", "RELAXED", "ptx", []),
            ("e_bulk_ctx_release", "GLOBAL", "sass", []),
        ],
        [("ptxv-sim", "fixtures", "bulk_complete_tx_releases_only_the_copy")],
    ),
]


# E-PAIR: tcgen05.alloc.cta_group::2 as a rendezvous of a warp of each CTA (MISS 179).
MUTANTS += [
    # Each CTA allocates on its own again: the peer's alloc races with the leader's paired MMA
    # (fixtures/r1c_2sm good on the PTX front), and ONESIDED runs to the end.
    SrcMutant(
        "pair-alloc-per-cta",
        "crates/ptx/src/lib.rs",
        "                        ncols: lw.val(op_n(1)?)?,\n                        pair,\n",
        "                        ncols: lw.val(op_n(1)?)?,\n                        pair: pair && false,\n",
        [("e_tmem_pair_alloc", "ONESIDED", "ptx", [])],
        [
            ("ptxv-sim", "fixtures", "tmem_pair_alloc_is_a_rendezvous"),
            ("ptxv-sim", "fixtures", "r1c_2sm_good"),
        ],
    ),
    # The first warp runs on without waiting for its peer (fixtures/e_tmem_pair_alloc ONESIDED
    # does not deadlock; good reads an address not written yet).
    SrcMutant(
        "pair-alloc-first-runs-on",
        "crates/sim/src/exec.rs",
        "            return Ok(Flow::Stay);\n        };\n        if m != n {",
        "            return Ok(Flow::Next);\n        };\n        if m != n {",
        [("e_tmem_pair_alloc", "ONESIDED", "ptx", [])],
        [("ptxv-sim", "fixtures", "tmem_pair_alloc_is_a_rendezvous")],
    ),
    # The marks as the second warp's own write: the first warp, which takes only the marks'
    # epoch, is not ordered after them (fixtures/r1c_2sm good races when the peer comes second).
    SrcMutant(
        "pair-alloc-marks-own-thread",
        "crates/sim/src/exec.rs",
        "            self.tmem_reset(c, col, n, actor, &vc, pc);\n        }\n        self.tmem[peer as usize].pair",
        "            self.tmem_reset(c, col, n, t as u32, &vc, pc);\n        }\n        self.tmem[peer as usize].pair",
        [],
        [("ptxv-sim", "fixtures", "r1c_2sm_good")],
    ),
]

# wellsync.rs `run_logged` (MISS 235): the certificate's run stops at a step that stops threads
# at an error, and the certificate does not hold. `nostop` takes the stop out (the run goes on
# past the error: r4c_nowrap B36 SASS never ends, e_atom_half exact_size certifies its fault);
# `dropcontinue` also drops the error's finding. The guards run the certificate on a thread of
# their own and fail after 120 s.
MISS235 = [
    ("ptxv-sim", "wellsync", "cert_run_stops_at_a_thread_stopping_error"),
    ("ptxv-sim", "wellsync", "cert_refuses_a_faulting_run_and_keeps_the_fault"),
]
MUTANTS += [
    SrcMutant(
        "cert-error-stop-nostop",
        "crates/sim/src/wellsync.rs",
        '        if let Some(f) = delivery\n            .fresh(&m)\n            .iter()\n            .find(|f| f.sig.starts_with("fault:"))\n        {\n            stopped_at_error = true;\n',
        '        if let Some(f) = None::<&Finding> {\n            stopped_at_error = true;\n',
        [],
        MISS235,
    ),
    SrcMutant(
        "cert-error-stop-dropcontinue",
        "crates/sim/src/wellsync.rs",
        '        if let Some(f) = delivery\n            .fresh(&m)\n            .iter()\n            .find(|f| f.sig.starts_with("fault:"))\n        {\n            stopped_at_error = true;\n',
        '        m.findings.retain(|f| !f.sig.starts_with("fault:"));\n        if let Some(f) = None::<&Finding> {\n            stopped_at_error = true;\n',
        [],
        MISS235,
    ),
]


# MISS 197, 202: a max keeps its kind (FMNMX, FMNMX.NAN, a compare and select's mask) in its L0
# form, with whether FMNMX without .NAN dropped an infinity; maxima share a value form only where
# both operands are non-NaN for every admissible input, relying on N-FINITE
# (probes/equiv_claims: select_max, max_identity, fmnmx_nan). Each mutant restores one site of
# the old FMNMX-only form or its unconditional sharing.
MAX_KIND_CLI = [("ptxv", "equiv", "max_kinds_are_not_l0")]
MAX_KIND_UNIT = [("ptxv-sim", "lib", "equiv::soundness_tests::max_kinds_are_part_of_the_l0_form")]
MAX_IDENT_UNIT = [("ptxv-sim", "lib", "equiv::soundness_tests::a_dropped_identity_is_part_of_the_l0_form")]
MAX_NAN_CLI = [("ptxv", "equiv", "maxima_of_maybe_nan_operands_share_no_value_form")]
MUTANTS += [
    # A compare and select taken as FMNMX again.
    SrcMutant(
        "max-select-as-fmnmx",
        "crates/sim/src/num.rs",
        "                        let n = ns.max_node(x, y, gt_y, MaxKind::Select(mask), pc);",
        "                        let n = ns.max_node(x, y, gt_y, MaxKind::FMNMX, pc);",
        [],
        MAX_KIND_CLI,
    ),
    # max.NaN.f32 taken as max.f32 again.
    SrcMutant(
        "max-nan-flag-dropped",
        "crates/sim/src/num.rs",
        "            acc = ns.max_node(acc, n, min, MaxKind::Fmnmx { nan }, pc);\n            if sass {",
        "            acc = ns.max_node(acc, n, min, MaxKind::Fmnmx { nan: false }, pc);\n            if sass {",
        [],
        MAX_KIND_CLI + MAX_NAN_CLI,
    ),
    # Every select of the same operands one L0 form, whatever it returns at ties and NaNs.
    SrcMutant(
        "max-select-mask-off",
        "crates/sim/src/equiv.rs",
        '                        h.put(("select-max", k, ha, hb));',
        '                        h.put(("select-max", MaxKind::FMNMX, ha, hb));',
        [],
        MAX_KIND_UNIT + MAX_KIND_CLI,
    ),
    # A dropped -inf / +inf left out of FMNMX's L0 form again.
    SrcMutant(
        "max-identity-flag-off",
        "crates/sim/src/equiv.rs",
        '                        h.put(("max", *min, *nan, dropped && !*nan, hs));',
        '                        h.put(("max", *min, *nan, false, hs));',
        [],
        MAX_IDENT_UNIT + MAX_KIND_CLI,
    ),
    # One value form for every kind, whether or not an operand may be NaN.
    SrcMutant(
        "max-value-share-any",
        "crates/sim/src/equiv.rs",
        "                if *fin {\n                    let mut relies = false;",
        "                if true {\n                    let mut relies = false;",
        [],
        MAX_KIND_UNIT + MAX_NAN_CLI,
    ),
    # The shared value form without the N-FINITE reliance.
    SrcMutant(
        "max-value-reliance-off",
        "crates/sim/src/equiv.rs",
        "                    if relies {\n                        cx.finite_max = true;",
        "                    if relies {\n                        cx.finite_max = false;",
        [],
        MAX_KIND_UNIT + MAX_KIND_CLI,
    ),
    # A dropped identity in the shared value form without the reliance.
    SrcMutant(
        "max-identity-reliance-off",
        "crates/sim/src/equiv.rs",
        "                            NodeKind::Leaf if ix(x).hw == ident => relies = true,",
        "                            NodeKind::Leaf if ix(x).hw == ident => {}",
        [],
        MAX_IDENT_UNIT + MAX_KIND_CLI,
    ),
    # The errdist model version left at a number below 26 (17: a retired number).
    SrcMutant(
        "errdist-model-version-17",
        "crates/cli/src/errdist.rs",
        "pub(crate) const MODEL_VERSION: u32 = 26;",
        "pub(crate) const MODEL_VERSION: u32 = 17;",
        [],
        [("ptxv", "bin:ptxv", "errdist::tests::models_of_earlier_versions_are_refused")],
    ),
    # errdist's maximum evaluated as FMNMX whatever its kind.
    SrcMutant(
        "errdist-max-eval-fmnmx",
        "crates/sim/src/errdist/nonlin/ops.rs",
        "                f32::from_bits(kind.eval(*min, val(*a).to_bits(), val(*b).to_bits()))",
        "                f32::from_bits(MaxKind::FMNMX.eval(*min, val(*a).to_bits(), val(*b).to_bits()))",
        [],
        [("ptxv-sim", "lib", "errdist::nonlin::ops::tests::maxima_evaluate_as_their_kind")],
    ),
    # The softmax form's row maximum reads a select of maybe-NaN operands.
    SrcMutant(
        "softmax-max-any-kind",
        "crates/sim/src/softmax.rs",
        '            } if *kind == MaxKind::FMNMX || *fin => Some((*a, *b)),',
        '            } if (*kind == MaxKind::FMNMX || *fin) || true => Some((*a, *b)),',
        [],
        [("ptxv-sim", "lib", "softmax::max_kind_tests::row_maxima_read_other_kinds_only_where_finite")],
    ),
    # The softmax form's row maximum reads a comparison-selection segment of maybe-NaN operands.
    SrcMutant(
        "softmax-seg-max-any",
        "crates/sim/src/softmax.rs",
        "                ns.non_nan_with(&mut self.fin_memo, a) && ns.non_nan_with(&mut self.fin_memo, b)\n",
        "                (ns.non_nan_with(&mut self.fin_memo, a) && ns.non_nan_with(&mut self.fin_memo, b)) || true\n",
        [],
        [("ptxv-sim", "lib", "softmax::comparison_maximum_projection_tests::real_projection_rejects_polynomial_operands_with_non_finite_constants")],
    ),
]

# MISS 201: an f32 product or fma rounded to a subnormal errs by up to 2^-150 absolute; a later
# product scales that term by its other factor, so there is no polynomial bound and neither L2
# nor L3 (probes/equiv_claims/product_underflow). Each mutant restores one site of the old
# relative-only model, or drops the grid test that keeps exact subnormal results term-free.
PROD_SUB_UNIT = [("ptxv-sim", "lib", "equiv::soundness_tests::a_subnormal_product_rounding_carries_an_absolute_term")]
PROD_SUB_CLI = [("ptxv", "equiv", "products_rounded_to_subnormals_have_no_bound")]
PROD_MMA_UNIT = [("ptxv-sim", "lib", "equiv::tf32_tests::an_absolute_term_in_an_mma_operand_has_no_bound")]
MUTANTS += [
    # No term for a product rounded to a subnormal (the grids never sum below -1000).
    SrcMutant(
        "product-sub-term-off",
        "crates/sim/src/equiv.rs",
        "                let sub = if eps != 0.0 && self.grid(*a) + self.grid(*b) < -149 {",
        "                let sub = if eps != 0.0 && self.grid(*a) + self.grid(*b) < -1000 {",
        [],
        PROD_SUB_UNIT + PROD_SUB_CLI,
    ),
    # A term for every inexact product, also where a subnormal result is exact (x * 1 - y).
    SrcMutant(
        "product-sub-grid-off",
        "crates/sim/src/equiv.rs",
        "                let sub = if eps != 0.0 && self.grid(*a) + self.grid(*b) < -149 {",
        "                let sub = if eps != 0.0 && self.grid(*a) + self.grid(*b) < 1 << 30 {",
        [],
        PROD_SUB_UNIT,
    ),
    # L2 and L3 held where an absolute term enters a product.
    SrcMutant(
        "abs-in-product-levels-off",
        "crates/sim/src/equiv.rs",
        "        if !lv[4] && (cr.round_in_product || cc.round_in_product) {\n            note!(2, ABS_IN_PRODUCT.into());",
        "        if false {\n            note!(2, ABS_IN_PRODUCT.into());",
        [],
        PROD_SUB_UNIT + PROD_SUB_CLI,
    ),
    # ... and failed also where the two outputs are equal (L0, L0~0).
    SrcMutant(
        "abs-in-product-at-l0",
        "crates/sim/src/equiv.rs",
        "        if !lv[4] && (cr.round_in_product || cc.round_in_product) {",
        "        if true && (cr.round_in_product || cc.round_in_product) {",
        [],
        PROD_SUB_UNIT,
    ),
    # An MMA operand's absolute term counted only for 16-bit roundings.
    SrcMutant(
        "mma-operand-abs-round-only",
        "crates/sim/src/equiv.rs",
        "                                .any(|k| memo.get(k).is_some_and(|e| e.2.abs_extra > 0.0))",
        "                                .any(|k| memo.get(k).is_some_and(|e| e.2.round_abs))",
        [],
        PROD_MMA_UNIT,
    ),
    # The same for a TF32 MMA chain's operands.
    SrcMutant(
        "tf32-operand-abs-round-only",
        "crates/sim/src/equiv.rs",
        "                    |k: u32, memo: &Memo| memo.get(&k).is_some_and(|e| e.2.abs_extra > 0.0);",
        "                    |k: u32, memo: &Memo| memo.get(&k).is_some_and(|e| e.2.round_abs);",
        [],
        PROD_MMA_UNIT,
    ),
]

# MISS 143, 145 residuals: the L3 line says "+ A" where L3 holds and the bound has an absolute
# term; printed bounds round up and printed condition limits down (ptxv equiv, ptxv compare).
# Each mutant restores the old label or the old rounding to nearest at one site.
PRINT_UNIT = [("ptxv", "bin:ptxv", "compare::tests::printed_bounds_round_outward")]
L3_CLI = [("ptxv", "equiv", "l3_line_carries_the_absolute_term")]
MUTANTS += [
    # The L3 line without "+ A".
    SrcMutant(
        "l3-line-abs-off",
        "crates/cli/src/equiv.rs",
        '            let l3a = l.level.starts_with("L3") && l.holds && abs_max > 0.0;',
        '            let l3a = l.level.starts_with("L3") && l.holds && abs_max < 0.0;',
        [],
        L3_CLI,
    ),
    # Bounds printed rounded to nearest.
    SrcMutant(
        "printed-bound-nearest",
        "crates/cli/src/compare.rs",
        "    fixed(x, d, true)\n}",
        '    format!("{x:.d$}")\n}',
        [],
        PRINT_UNIT + [("ptxv", "equiv", "tf32_gemms_are_bounded")],
    ),
    # Absolute terms (scientific form) printed rounded to nearest.
    SrcMutant(
        "printed-abs-nearest",
        "crates/cli/src/compare.rs",
        '    let all = format!("{i}{f}");\n',
        '    let all = format!("{i}{f}");\n    if true {\n        return format!("{x:.d$e}");\n    }\n',
        [],
        PRINT_UNIT + L3_CLI + [("ptxv", "compare", "compare_headline_carries_the_absolute_term")],
    ),
    # Condition limits printed rounded to nearest.
    SrcMutant(
        "printed-limit-nearest",
        "crates/cli/src/compare.rs",
        "    fixed(x, d, false)\n}",
        '    format!("{x:.d$}")\n}',
        [],
        PRINT_UNIT + [("ptxv", "equiv", "f16_outputs_are_bounded")],
    ),
]

# MISS 226: E-RACE-DEAD leaves out mixed-size data races (exec.rs `race_log`, PTX ISA 9.5
# section 8.7.2), decided where S-SCOPE decides overlap (the strict PTX level, the NIC's accesses
# included). race-dead-mixed undoes the fix; race-dead-uniform-off keeps every pair out
# (uniform-size races with a dead value become races again); race-dead-mixed-hw decides overlap
# under `hw_order` too (the level-2 hardware model, which takes every pair as one location);
# race-dead-mixed-nic leaves the NIC's accesses undecided (no byte); race-dead-unmodeled takes a
# CQE overlap the model does not decide as one location.
RACE_DEAD_MIXED = ("ptxv-sim", "reductions", "mixed_size_races_are_not_race_dead")
RACE_DEAD_FIND = (
    "                let ok = ok.filter(|_| {\n"
    "                    self.overlap_at(other.actor, other.pc, actor, pc, galloc.map(|al| (al, a)))\n"
    "                        == Overlap::Complete\n"
    "                });\n"
)
MUTANTS += [
    SrcMutant(
        "race-dead-mixed",
        "crates/sim/src/exec.rs",
        RACE_DEAD_FIND,
        "                let ok = ok.filter(|_| true);\n",
        [("e_race_dead", "RDMIX", "ptx", []), ("e_nic_cta", "WORDDEAD", "ptx", [])],
        [RACE_DEAD_MIXED],
    ),
    SrcMutant(
        "race-dead-uniform-off",
        "crates/sim/src/exec.rs",
        RACE_DEAD_FIND,
        "                let ok = ok.filter(|_| false);\n",
        [("e_race_dead", "good", f, []) for f in ("ptx", "sass")],
        [RACE_DEAD_MIXED],
    ),
    SrcMutant(
        "race-dead-mixed-hw",
        "crates/sim/src/exec.rs",
        RACE_DEAD_FIND,
        "                let ok = ok.filter(|_| {\n"
        "                    overlap_with_nic(\n"
        "                        &self.k,\n"
        "                        self.threads.len() as u32,\n"
        "                        true,\n"
        "                        self.nic.as_deref(),\n"
        "                        self.async_base(),\n"
        "                        (other.actor, other.pc),\n"
        "                        (actor, pc),\n"
        "                        galloc.map(|al| (al, a)),\n"
        "                    ) == Overlap::Complete\n"
        "                });\n",
        [],
        [RACE_DEAD_MIXED],
    ),
    SrcMutant(
        "race-dead-mixed-nic",
        "crates/sim/src/exec.rs",
        "                    self.overlap_at(other.actor, other.pc, actor, pc, galloc.map(|al| (al, a)))\n"
        "                        == Overlap::Complete\n",
        "                    self.overlap_at(other.actor, other.pc, actor, pc, None)\n"
        "                        == Overlap::Complete\n",
        [("e_nic_cta", "WORDDEAD", "ptx", [])],
        [RACE_DEAD_MIXED],
    ),
    SrcMutant(
        "race-dead-unmodeled",
        "crates/sim/src/exec.rs",
        RACE_DEAD_FIND,
        "                let ok = ok.filter(|_| {\n"
        "                    self.overlap_at(other.actor, other.pc, actor, pc, galloc.map(|al| (al, a)))\n"
        "                        != Overlap::Partial\n"
        "                });\n",
        [("e_nic_cta", "CQEDEAD", "ptx", [])],
        [RACE_DEAD_MIXED],
    ),
]

# MISS 28 (E-CLOCK, clock.rs): clock-dependent control is explored both ways or reported. Each
# mutant drops one part of the analysis; the fixture or test that part exists for must notice.
MUTANTS += [
    # No clock choices: a timeout branch is decided by the clock read as 0 (the miss itself).
    SrcMutant(
        "clock-choice-off",
        "crates/sim/src/clock.rs",
        "    choices(k, delay, &clock_taint(k))\n}",
        "    let _ = (k, delay);\n    HashSet::new()\n}",
        [("e_clock_timeout", "good", f, []) for f in ("ptx", "sass")],
    ),
    # 32-bit clock / timer reads are the constant 0 again, not clock registers (PTX, SASS).
    SrcMutant(
        "clock32-ptx",
        "crates/ptx/src/lib.rs",
        '"%clock" | "%globaltimer_lo" => return Ok(Val::SReg(SReg::ClockLo32)),',
        '"%clock" | "%globaltimer_lo" => return Ok(Val::Imm(0)),',
        [("e_clock_timeout32", "good", "ptx", [])],
    ),
    SrcMutant(
        "clock32-sass",
        "crates/sass/src/lib.rs",
        '"CLOCKLO" | "GLOBALTIMERLO" => return Ok(Val::SReg(SReg::ClockLo32)),',
        '"CLOCKLO" | "GLOBALTIMERLO" => return Ok(Val::Imm(0)),',
        [("e_clock_timeout32", "good", "sass", [])],
    ),
    # A loop with a store in its body counts as a pure delay loop (its deadline passes at once).
    SrcMutant(
        "clock-delay-impure",
        "crates/sim/src/clock.rs",
        "        if !body.iter().all(|i| crate::num::pure_op(&i.op)) {",
        "        if false && !body.iter().all(|i| crate::num::pure_op(&i.op)) {",
        [("clock_pace_store", "good", "ptx", [])],
        [("ptxv-sim", "fixtures", "clock_paced_loops")],
    ),
    # A loop whose registers are live after it counts as a pure delay loop.
    SrcMutant(
        "clock-delay-live",
        "crates/sim/src/clock.rs",
        "        if crate::spin::live_after(&k.code, b + 1, &written).is_some() {",
        "        if false && crate::spin::live_after(&k.code, b + 1, &written).is_some() {",
        [("clock_pace_live", "good", "ptx", [])],
        [("ptxv-sim", "fixtures", "clock_paced_loops")],
    ),
    # Clock-derived bytes in memory are not marked: a branch on a loaded timestamp is decided
    # by the clock read as 0.
    SrcMutant(
        "clock-mem-off",
        "crates/sim/src/exec.rs",
        "        if !self.clock_store_pcs.is_empty() {",
        "        if false && !self.clock_store_pcs.is_empty() {",
        [("e_clock_mem", "good", f, []) for f in ("ptx", "sass")],
    ),
    # A clock-dependent output (a stored select on the clock) is claimed complete.
    SrcMutant(
        "clock-out-complete",
        "crates/sim/src/clock.rs",
        "    v.complete = false;\n",
        "",
        [("e_clock_data", "good", "ptx", [])],
        [("ptxv-sim", "fixtures", "e_clock_data_output_incomplete")],
    ),
    # The certificate takes one side of a clock choice.
    SrcMutant(
        "clock-cert-choice",
        "crates/sim/src/wellsync.rs",
        "    if !m0.clock_choice_pcs.is_empty() || !m0.mem_branch_pcs.is_empty() {",
        "    if false && (!m0.clock_choice_pcs.is_empty() || !m0.mem_branch_pcs.is_empty()) {",
        [("e_clock_timeout", "good", "ptx", ["--certify"])],
        [("ptxv-sim", "fixtures", "e_clock_timeout_branch_both_ways")],
    ),
    # The certificate claims a kernel that stores clock-derived values.
    SrcMutant(
        "clock-cert-store",
        "crates/sim/src/wellsync.rs",
        "    if !m0.clock_store_pcs.is_empty() {",
        "    if false && !m0.clock_store_pcs.is_empty() {",
        [],
        [("ptxv-sim", "fixtures", "probe_clock32_matches_hardware")],
    ),
    # A clock-derived address (a control operand) is not reported (GB300 stores either word of
    # fixtures/e_clock_addr: 116 / 200 runs out[1], 84 / 200 out[0]).
    SrcMutant(
        "clock-effect-off",
        "crates/sim/src/clock.rs",
        "pub fn clock_effects(k: &Kernel) -> HashSet<u32> {\n",
        "pub fn clock_effects(k: &Kernel) -> HashSet<u32> {\n    if !k.code.is_empty() {\n        return HashSet::new();\n    }\n",
        [("e_clock_addr", v, f, []) for v in ("good", "SELP") for f in ("ptx", "sass")],
    ),
    # A clock value spilled to a stack slot is not followed to its reload.
    SrcMutant(
        "clock-local-slot",
        "crates/sim/src/clock.rs",
        "                            local.insert((k.entry_of(pc), lo, hi));",
        "                            let _ = (lo, hi);",
        [],
        [("ptxv-sim", "fixtures", "clock_taint_no_third_category")],
    ),
    # A clock-free write under a guard clears the register for every read, not only for reads
    # under the same guard.
    SrcMutant(
        "clock-guard-kill",
        "crates/sim/src/clock.rs",
        "        self.t.contains(r) || self.k.get(r).is_some_and(|h| g.reg.is_none() || *h != g)",
        "        let _ = g;\n        self.t.contains(r)",
        [],
        [("ptxv-sim", "fixtures", "clock_taint_guarded_kill")],
    ),
    # A clock guard does not make what its instruction writes clock-dependent data.
    SrcMutant(
        "clock-data-guard",
        "crates/sim/src/clock.rs",
        "                || data\n",
        "                || false && data\n",
        [],
        [("ptxv-sim", "fixtures", "clock_taint_no_third_category")],
    ),
]


# MISS 206 (defense in depth): a bound one of its output's analyzed elements exceeds is reported as
# unknown.
MUTANTS += [
    # The bounds print whatever they are again: c_num_nonfinite_f32's CSV carries numbers.
    SrcMutant(
        "num-bound-unknown",
        "crates/cli/src/main.rs",
        "                recs.iter_mut().for_each(|r| r.bound_cor1a = f64::NAN);\n",
        "",
        [],
        [("ptxv", "num_untracked", "bounds_an_output_violates_are_unknown")],
    ),
    # MISS 206: the MUFU bound takes ulp_f32 of the overflowed |hw| (1 + 2^-20) again (2^-149):
    # c_num_mufu_top_f32's two top-magnitude elements exceed it, and the output's bounds are unknown.
    SrcMutant(
        "mufu-top-ulp",
        "crates/sim/src/num.rs",
        "    } else {\n        2f64.powi(105)\n    };\n",
        "    } else {\n        ulp_f32(above)\n    };\n",
        [],
        [("ptxv", "num_untracked", "mufu_bound_at_the_top_magnitudes_covers_gb300")],
    ),
]


# MISS 207: PTX .ftz approx results (slices) carry the MUFU bound (fixtures/c_num_mufu_ftz_f32).
FTZ_BOUND = ("ptxv", "num_untracked", "ptx_ftz_approx_results_carry_the_sweep_bound")
MUTANTS += [
    # The slice is a plain term again (no own bound): the PTX front's bounds are 0, below GB300's
    # errors, and every output's bounds are unknown.
    SrcMutant(
        "mufu-ftz-bound",
        "crates/sim/src/num.rs",
        "                NodeKind::Seg(_) if self.ftz_mufu.contains_key(&n) => {\n                    rnodes.push(n);\n",
        "                NodeKind::Seg(_) if false && self.ftz_mufu.contains_key(&n) => {\n                    rnodes.push(n);\n",
        [],
        [FTZ_BOUND],
    ),
    # No flush term: sqrt.approx.ftz of a subnormal is bounded by 0.944 2^-149 against sqrt(x).
    SrcMutant(
        "mufu-ftz-flush",
        "crates/sim/src/num.rs",
        "            let own = mufu_local_bound(func, ptx, x as f32, f32::from_bits(nd.hw)) + flush_in;\n",
        "            let own = mufu_local_bound(func, ptx, x as f32, f32::from_bits(nd.hw)) + 0.0 * flush_in;\n",
        [],
        [FTZ_BOUND],
    ),
]


# MISS 208: an f16 / bf16 multimem.ld_reduce sum below GB300's 16-aligned bound is +0, and a sum of
# zeros is +0 (probes/multimem_add16_rules/sweep, GB300).
HALF_FLUSH = ("ptxv-sim", "multimem", "half_sums_follow_the_flush_sweeps")
# MISS 225: the f32 sum's 16-aligned window and flush (probes/multimem_f32_window, the sweeps).
F32_WINDOW_TESTS = [("ptxv-sim", "multimem", "f32_sums_follow_the_window_sweeps")]
MUTANTS += [
    # -0 when every member is -0 again.
    SrcMutant(
        "mm-zero-sign",
        "crates/sim/src/multimem.rs",
        "        return Ok(Lane::Done(0));\n    }\n    // 2^a16",
        "        return Ok(Lane::Done((fin.iter().all(|(d, _)| d.neg) as u64) << (eb + mb)));\n    }\n    // 2^a16",
        [],
        [HALF_FLUSH] + F32_WINDOW_TESTS,
    ),
    # No flush of f16 / bf16 sums: B - B + R keeps R below the bound.
    SrcMutant(
        "mm-half-flush",
        "crates/sim/src/multimem.rs",
        "            (emax - HALF_WINDOW, a16 - 23)\n",
        "            (emax - HALF_WINDOW, i32::MIN)\n",
        [],
        [HALF_FLUSH],
    ),
    # The bound and window at eB itself, not aligned to 16 binades.
    SrcMutant(
        "mm-align16",
        "crates/sim/src/multimem.rs",
        "    let a16 = 16 * (e_b / 16) - ((1i32 << (eb - 1)) - 1);\n",
        "    let a16 = e_b - ((1i32 << (eb - 1)) - 1);\n",
        [],
        [HALF_FLUSH] + F32_WINDOW_TESTS,
    ),
    # The former f32 window: 63 bits below the largest member's last mantissa bit, no flush.
    SrcMutant(
        "mm-f32-window",
        "crates/sim/src/multimem.rs",
        "        McTy::F32 => (a16 - 80, a16 - 71),\n",
        "        McTy::F32 => (\n            fin.iter().filter(|(d, _)| d.man != 0).map(|(d, _)| d.exp).max().unwrap() - 63,\n            i32::MIN,\n        ),\n",
        [],
        F32_WINDOW_TESTS,
    ),
    # The f32 window without its flush: B - B + R keeps the truncated R.
    SrcMutant(
        "mm-f32-flush",
        "crates/sim/src/multimem.rs",
        "        McTy::F32 => (a16 - 80, a16 - 71),\n",
        "        McTy::F32 => (a16 - 80, i32::MIN),\n",
        [],
        F32_WINDOW_TESTS,
    ),
]


# MISS 209: st.async and UBLKCP.S.S need a cluster launch, and a kernel's compiled-in cluster
# makes one (probes/cluster_launch_ops, GB300).
CLUSTER_OPS = ("ptxv-sim", "fixtures", "probe_cluster_launch_ops_match_hardware")
MUTANTS += [
    # No fault without cluster dimensions: st.async and UBLKCP.S.S run, as before.
    SrcMutant(
        "cluster-launch-fault",
        "crates/sim/src/exec.rs",
        "        if self.launch.g(cta as u32).explicit_cluster {\n            return Ok(());\n",
        "        if true || self.launch.g(cta as u32).explicit_cluster {\n            return Ok(());\n",
        [],
        [CLUSTER_OPS],
    ),
    # A compiled-in cluster shape ignored (only the case's `cluster` counts): the cluster1
    # variants fault at st.async and read %is_explicit_cluster 0.
    SrcMutant(
        "cluster-decl-ignored",
        "crates/ir/src/lib.rs",
        "            (Some(ClusterDecl::Dims(d)), None) => Ok(Some(d)),\n",
        "            (Some(ClusterDecl::Dims(_)), None) => Ok(None),\n",
        [],
        [CLUSTER_OPS, ("ptxv-ir", "lib", "tests::cluster_decl_resolve")],
    ),
    # The SASS front reads no EIATTR_CTA_PER_CLUSTER: the SASS cluster1 variants fault.
    SrcMutant(
        "cluster-eiattr-ignored",
        "crates/sass/src/lib.rs",
        "                EIATTR_CTA_PER_CLUSTER if v.len() >= 12 => {\n",
        "                EIATTR_CTA_PER_CLUSTER if false => {\n",
        [],
        [CLUSTER_OPS],
    ),
]


# MISS 224: an RDMA write moves the source words' numerics nodes (fixtures/c_num_nic_write_f32,
# probes/nic_num_shadow).
MUTANTS += [
    # The destination keeps the nodes of what it held: the copied input's exact value.
    SrcMutant(
        "nic-num-write",
        "crates/sim/src/nic.rs",
        "                    self.num_copy_snapshot(loc, &data, snapshot.as_ref(), pc);\n",
        "",
        [],
        [("ptxv", "num_untracked", "nic_writes_move_numerics_nodes")],
    ),
]

# MISS 257: a NIC atomic keeps the nodes of the words it leaves unchanged; words whose outcome may
# differ in the exact computation are input data (fixtures/c_num_nic_atomic_f32,
# probes/nic_atomic_keeps_word).
MUTANTS += [
    # Unchanged words lose their nodes, as at MISS 224's fix: the rounded values they still hold
    # become exact constants (W1, W3's low word claimed exact).
    SrcMutant(
        "nic-atomic-keep",
        "crates/sim/src/num.rs",
        "            if input.is_some_and(|i| !i[k]) && word(a.new, k) == word(a.old, k) {\n",
        "            if false && input.is_some_and(|i| !i[k]) && word(a.new, k) == word(a.old, k) {\n",
        [],
        [("ptxv", "num_untracked", "nic_atomics_keep_unchanged_words_and_taint_decided_ones")],
    ),
    # Words the atomic computes from a rounding lose their nodes without taint: exact constants
    # (W2, W3's high word, and W0 claimed).
    SrcMutant(
        "nic-atomic-untracked",
        "crates/sim/src/num.rs",
        "            if input.is_some_and(|i| i[k]) {\n                for b in key.2..key.2 + 4 {\n",
        "            if false {\n                for b in key.2..key.2 + 4 {\n",
        [],
        [("ptxv", "num_untracked", "nic_atomics_keep_unchanged_words_and_taint_decided_ones")],
    ),
    # A compare-and-swap on rounded words decides nothing: W0's unchanged words keep their nodes.
    SrcMutant(
        "nic-atomic-decided",
        "crates/sim/src/num.rs",
        "                let decided = a.compare_inexact || a.old_inexact[0] || a.old_inexact[1];\n",
        "                let decided = a.compare_inexact;\n",
        [],
        [("ptxv", "num_untracked", "nic_atomics_keep_unchanged_words_and_taint_decided_ones")],
    ),
    # An unchanged word does not get its 16-bit provenance back: W5's bf16 elements lose their
    # nodes (outh untracked).
    SrcMutant(
        "nic-atomic-half",
        "crates/sim/src/num.rs",
        "        if let Some(s) = snapshot.filter(|s| s.half.len() == 8) {\n",
        "        if let Some(s) = snapshot.filter(|_| false) {\n",
        [],
        [("ptxv", "num_untracked", "nic_atomics_keep_unchanged_words_and_taint_decided_ones")],
    ),
    # A fetch-add's carry into the high word ignores the low word: W2's high word keeps its node.
    SrcMutant(
        "nic-atomic-carry",
        "crates/sim/src/num.rs",
        "                    a.operand_inexact || carry || high_changed && a.old_inexact[1],\n",
        "                    a.operand_inexact || high_changed && a.old_inexact[1],\n",
        [],
        [("ptxv", "num_untracked", "nic_atomics_keep_unchanged_words_and_taint_decided_ones")],
    ),
]


# MISS 29 (E-ELECT, PTX): every leader function is explored where the deciding gate fires
# (explore.rs `per_leader_policy`, the leader search in `actions` and `elect_lanes::candidates`,
# wellsync.rs `certify_runs`). Each mutant falls back to end-lane leaders where the gate fired.
MUTANTS += [
    # The explorer ignores the gate: the two end-lane policies for every kernel (the miss).
    SrcMutant(
        "elect-fn-off",
        "crates/sim/src/explore.rs",
        "    if let Some(at) = elect_sensitivity(&m0, false)\n        .map(|x| x.0)\n        .or(forced.then_some(u32::MAX))",
        "    if let Some(at) = forced.then_some(u32::MAX)",
        [("e_elect_mid", "good", "ptx", [])],
    ),
    # The certificate ignores the gate: one run per end-lane policy.
    SrcMutant(
        "elect-fn-cert-off",
        "crates/sim/src/wellsync.rs",
        "            || crate::explore::elect_sensitivity(&m0, false).is_some());",
        "            || false);",
        [("e_elect_mid", "good", "ptx", ["--certify"])],
    ),
    # The explorer's leader search, entered, offers the lowest active lane only.
    SrcMutant(
        "elect-fn-lowest",
        "crates/sim/src/explore.rs",
        "                        } else if self.elect_any {",
        "                        } else if self.elect_any && false {",
        [("e_elect_mid", "good", "ptx", [])],
    ),
    # The candidates of an active set are its two end lanes (both the explorer and the
    # certificate draw them from `candidates`).
    SrcMutant(
        "elect-cand-ends",
        "crates/sim/src/elect_lanes.rs",
        "    let lanes = (0..32).filter(|&l| mask >> l & 1 == 1);",
        "    let lanes = (0..32).filter(|&l| {\n        mask >> l & 1 == 1 && (l == mask.trailing_zeros() || l == 31 - mask.leading_zeros())\n    });",
        [("e_elect_mid", "good", "ptx", []), ("e_elect_mid", "good", "ptx", ["--certify"])],
    ),
]

# MISS 31 (E-WARP-ORDER, PTX): unordered strong pairs of a converged warp on different
# instructions are nondeterministic (exec.rs `access_x`, `warp_nd`).
MUTANTS += [
    # No report: the converged warp's program-order outcome alone, clean (the miss).
    SrcMutant(
        "warp-order-off",
        "crates/sim/src/exec.rs",
        "let warp_nd: Vec<(u32, u32, &'static str, u64, bool)> = if self.k.front == Front::Ptx {",
        "let warp_nd: Vec<(u32, u32, &'static str, u64, bool)> = if false && self.k.front == Front::Ptx {",
        [("e_warp_order_div", "CONV", "ptx", []), ("s_atom_ext", "good", "ptx", [])],
    ),
    # Any two result-less reductions count as commuting (red.min with red.max of one word).
    SrcMutant(
        "warp-order-commute",
        "crates/sim/src/exec.rs",
        "            red(x).is_some() && red(x) == red(y)",
        "            red(x).is_some() && red(y).is_some()",
        [("s_atom_ext", "good", "ptx", [])],
    ),
    # Shared-memory pairs only (soundness review): a global-memory pair goes unreported.
    SrcMutant(
        "warp-order-shared-only",
        "crates/sim/src/exec.rs",
        "                    other.pc != pc\n                        && !commuting(other.pc, pc)",
        "                    other.pc != pc\n                        && matches!(loc, Loc::Shared { .. })\n                        && !commuting(other.pc, pc)",
        [("e_warp_order_kinds", "good", "ptx", [])],
    ),
    # No write-write pairs (soundness review).
    SrcMutant(
        "warp-order-no-ww",
        "crates/sim/src/exec.rs",
        "                .map(|(other, kind, a)| {\n                    let hw = ",
        "                .filter(|(_, kind, _)| *kind != \"write-write\")\n                .map(|(other, kind, a)| {\n                    let hw = ",
        [("e_warp_order_kinds", "WW", "ptx", [])],
    ),
]

# MISS 73 (E-WARP-ORDER, PTX): the strong-pair report needs the two lanes in one lockstep group
# (`Machine::mgrp`): a group survives a branch other lanes take, and a barrier release, a
# bar.warp.sync release or a warp collective gathers its lanes into one (exec.rs `regroup`,
# `note_gathered`).
MUTANTS += [
    # A group dissolves into single lanes whenever the warp moves: only gathered lanes pair.
    SrcMutant(
        "warp-order-group-split",
        "crates/sim/src/exec.rs",
        "                    self.mgrp.get(u).copied().unwrap_or(u32::MAX),",
        "                    u as u32,",
        [("e_warp_order_partial", "NOHALF", "ptx", []), ("e_warp_order_div", "CONV", "ptx", [])],
    ),
    # Barrier releases gather nothing: the strong pair after a divergence and a bar.sync is
    # PTX clean again (the miss).
    SrcMutant(
        "warp-order-gather-bar",
        "crates/sim/src/exec.rs",
        "        if self.k.front == Front::Ptx {\n            self.gathered_by_release(arrived.iter()",
        "        if false && self.k.front == Front::Ptx {\n            self.gathered_by_release(arrived.iter()",
        [("e_warp_order_reconv", "good", "ptx", []), ("e_warp_order_reconv", "BARV", "ptx", [])],
    ),
    # A bar.warp.sync release gathers nothing (it ended a whole-warp divergence before MISS 73's
    # fix too).
    SrcMutant(
        "warp-order-gather-warpsync",
        "crates/sim/src/exec.rs",
        "    fn note_convergence(&mut self, warp: u32, lanes: &[usize]) {\n        self.note_gathered(warp, lanes, false);",
        "    fn note_convergence(&mut self, warp: u32, lanes: &[usize]) {\n        self.note_gathered(warp, &lanes[..0], false);",
        [("e_warp_order_reconv", "WARPV", "ptx", [])],
    ),
    # Warp collectives gather nothing (shfl.sync, elect.sync).
    SrcMutant(
        "warp-order-gather-coll",
        "crates/sim/src/exec.rs",
        "            for lanes in &gather {\n                self.note_gathered(warp, lanes, false);",
        "            for lanes in gather.iter().filter(|_| false) {\n                self.note_gathered(warp, lanes, false);",
        [
            ("e_warp_order_reconv", "SHFLV", "ptx", []),
            ("e_warp_order_reconv", "ELECTV", "ptx", []),
            ("e_warp_order_reconv", "SPAN", "ptx", []),
        ],
    ),
    # The groups are not hashed: two PTX states that differ in them merge, and the path
    # explored second loses its reports.
    SrcMutant(
        "warp-order-hash",
        "crates/sim/src/machine.rs",
        "        if self.k.front == Front::Ptx {\n            self.mgrp.hash(h);",
        "        if false && self.k.front == Front::Ptx {\n            self.mgrp.hash(h);",
        [],
        [("ptxv-sim", "fixtures", "warp_order_model_divergence_is_hashed")],
    ),
]


# MISS 74 (PTX), MISS 75, 168 (SASS stores): the lanes of one store / exchange / atomic on one word
# take effect in any order (exec.rs `lane_orders`, explore.rs `actions_with` and
# `mark_unexplored`, wellsync.rs `run_logged`).
MUTANTS += [
    # No lane orders: lane order only (the miss).
    SrcMutant(
        "lane-order-off",
        "crates/sim/src/exec.rs",
        "        if members.len() < 2 {\n            return None;\n        }\n        let inst = &self.k.code[pc as usize];\n        if inst.uniform {",
        "        if true || members.len() < 2 {\n            return None;\n        }\n        let inst = &self.k.code[pc as usize];\n        if inst.uniform {",
        [
            ("e_lane_order", "good", "ptx", []),
            ("e_lane_order_ticket", "THREE", "ptx", []),
            ("e_sts_keep", "good", "sass", []),
        ],
    ),
    # Stores: lane order only (the last lane's value remains).
    SrcMutant(
        "lane-order-last",
        "crates/sim/src/exec.rs",
        "                (*space, *bytes, addr, Kind::Last)\n",
        "                (*space, *bytes, addr, Kind::Commute)\n",
        [
            ("e_lane_order", "good", "ptx", []),
            ("e_lane_order", "PAIR", "ptx", []),
            ("e_sts_keep", "OUT", "sass", []),
        ],
    ),
    # Every atomic's result counts as dead: tickets commute.
    SrcMutant(
        "lane-order-live",
        "crates/sim/src/exec.rs",
        "                let dead =\n                    !publishes && p.is_none() && d.is_none_or(|_| self.atom_dead.contains(&pc));",
        "                let dead = true || !publishes;",
        [("e_lane_order_ticket", "THREE", "ptx", []), ("e_lane_order_ticket", "good", "ptx", [])],
    ),
    # Every atomic group takes lane order and its reverse only.
    SrcMutant(
        "lane-order-perm",
        "crates/sim/src/exec.rs",
        "                _ if lanes.len() <= LANE_PERM_ALL => permutations(&lanes),",
        "                _ if false => permutations(&lanes),",
        [],
        [("ptxv-sim", "fixtures", "e_lane_order_tickets_take_any_order")],
    ),
    # Truncated lane orders leave the launch check complete.
    SrcMutant(
        "lane-order-mark",
        "crates/sim/src/explore.rs",
        "        v.complete = false;\n        if !v.notes.contains(&note) {",
        "        if !v.notes.contains(&note) {",
        [],
        [("ptxv-sim", "fixtures", "e_lane_order_tickets_take_any_order")],
    ),
    # The certificate takes lane order (C-TICKET certified THREE clean).
    SrcMutant(
        "lane-order-cert",
        "crates/sim/src/wellsync.rs",
        "            if m.lane_orders(&members, pc).is_some() {",
        "            if false && m.lane_orders(&members, pc).is_some() {",
        [],
        [("ptxv-sim", "fixtures", "e_lane_order_tickets_take_any_order")],
    ),
    # A vector float add's operand is its first element only (VEC2: lane order only).
    SrcMutant(
        "lane-order-vec-first",
        "crates/sim/src/exec.rs",
        "                } if matches!(bytes, 8 | 16) => {\n                    let mut x = Vec::with_capacity(bytes as usize);",
        "                } if false => {\n                    let mut x = Vec::with_capacity(bytes as usize);",
        [("e_lane_order_ticket", "VEC2", "ptx", [])],
    ),
    # Float adds exact in every order still take every order (VECX: lane order and its
    # reverse with the note; probes/red_vec PTX: no longer completes in 2M states).
    SrcMutant(
        "lane-order-exact-off",
        "crates/sim/src/exec.rs",
        "                    adds_exact(*op, &init, &xs)\n",
        "                    adds_exact(*op, &init, &xs) && false\n",
        [],
        [("ptxv-sim", "fixtures", "e_lane_order_tickets_take_any_order")],
    ),
    # Order-free whenever the values are finite (no exactness bound).
    SrcMutant(
        "lane-order-exact-loose",
        "crates/sim/src/exec.rs",
        "            if sum >= 1u128 << p {\n                return false;",
        "            if sum >= 1u128 << 120 {\n                return false;",
        [("e_lane_order_ticket", "FADD3", "ptx", [])],
    ),
    # Gap A: an address with several locations (multicast) keeps lane order, silently.
    SrcMutant(
        "lane-order-multi-off",
        "crates/sim/src/exec.rs",
        "            let &loc = locs.first()?;\n            if locs.len() > 1 || multi {",
        "            let [loc] = locs[..] else {\n                return None;\n            };\n            if multi {",
        [("e_mm_lane", "good", "ptx", []), ("e_mm_lane", "TRAP0", "ptx", [])],
        [("ptxv-sim", "fixtures", "e_mm_lane_multicast_lanes_take_any_order")],
    ),
    # Gap B: dead-result integer atomics commute even when their lanes publish a clock.
    SrcMutant(
        "lane-order-sync-commute",
        "crates/sim/src/exec.rs",
        "                    !publishes && p.is_none() && d.is_none_or(|_| self.atom_dead.contains(&pc));",
        "                    !(publishes && false) && p.is_none() && d.is_none_or(|_| self.atom_dead.contains(&pc));",
        [("e_lo_sync", "good", "ptx", []), ("e_lo_sync", "FENCED", "ptx", [])],
        [("ptxv-sim", "fixtures", "e_lo_sync_synchronizing_lanes_take_any_order")],
    ),
    # Gap B: a release fence before a relaxed atomic does not count (only release atomics).
    SrcMutant(
        "lane-order-fence",
        "crates/sim/src/exec.rs",
        "                            && (!vc_entries(&self.threads[t].membar_vc).is_empty()",
        "                            && (false && !vc_entries(&self.threads[t].membar_vc).is_empty()",
        [("e_lo_sync", "FENCED", "ptx", [])],
        [("ptxv-sim", "fixtures", "e_lo_sync_synchronizing_lanes_take_any_order")],
    ),
]

# The E-WARP-ORDER completeness defect (soundness audit): a reported strong pair's other order
# is not explored, so the launch check's verdict is incomplete (explore.rs `mark_unexplored`).
MUTANTS += [
    # The verdict claims complete again.
    SrcMutant(
        "warp-order-complete",
        "crates/sim/src/explore.rs",
        "        .filter(|f| f.sig.starts_with(\"warp-order:\"))",
        "        .filter(|f| false && f.sig.starts_with(\"warp-order:\"))",
        [],
        [("ptxv-sim", "fixtures", "e_warp_order_other_order_incomplete")],
    ),
]

# MISS 75: strict SASS explores the lanes of one shared store on one word; under the GB300
# profile (E-HW-STS-LOWEST) the lowest active lane of the store's last phase with an active lane
# leaves its bytes (hwprofile.rs, exec.rs `hw_store_phase`).
MUTANTS += [
    # The strict SASS level keeps lane order for one store's lanes (lane 31's value: the miss
    # at the strict level, good clean).
    SrcMutant(
        "sts-lane-order",
        "crates/sim/src/exec.rs",
        "                    Front::Sass => self.hw_store_phase(inst).is_none(),\n",
        "                    Front::Sass => *space != Space::Shared && self.hw_store_phase(inst).is_none(),\n",
        [("e_sts_keep", "good", "sass", []), ("e_sts_keep", "OUT", "sass", [])],
        [("ptxv-sim", "fixtures", "e_sts_keep_strict_sass_takes_every_lane")],
    ),
    # The profile explores every store's lanes too (its measured order unused).
    SrcMutant(
        "sts-lowest-explored",
        "crates/sim/src/exec.rs",
        "                    Front::Sass => self.hw_store_phase(inst).is_none(),\n",
        "                    Front::Sass => true,\n",
        [],
        [("ptxv-sim", "fixtures", "gb300_profile_keeps_the_lowest_lane_of_one_store")],
    ),
    # The width rule off: every width one phase (the lowest active lane remains, as measured
    # for 4 bytes and less only; GB300 keeps lane 16 of one STS.64 by every lane, 24 of STS.128).
    SrcMutant(
        "sts-width-off",
        "crates/sim/src/exec.rs",
        "            } if self.launch.hw_sts_lowest => Some(32 / (*bytes as usize / 4).max(1)),",
        "            } if self.launch.hw_sts_lowest => Some(32 + 0 * *bytes as usize),",
        [],
        [("ptxv-sim", "fixtures", "gb300_profile_keeps_the_last_phase_of_one_store")],
    ),
    # The profile keeps lane order (lane 31's value).
    SrcMutant(
        "sts-lowest-off",
        "crates/sim/src/hwprofile.rs",
        "            \"E-HW-STS-LOWEST\" => h.hw_sts_lowest = true,",
        "            \"E-HW-STS-LOWEST\" => {}",
        [],
        [("ptxv-sim", "fixtures", "gb300_profile_keeps_the_lowest_lane_of_one_store")],
    ),
]

# MISS 168: strict SASS explores the lanes of one global store on one word; under the GB300
# profile (E-HW-STG-LOWEST) one 32-bit STG other than STG.E.MMIO.SYS leaves the lowest active
# lane's bytes (exec.rs `hw_store_phase`, hwprofile.rs).
MUTANTS += [
    # The strict SASS level keeps lane order for global stores (lane 31's value: good clean).
    SrcMutant(
        "stg-lane-order",
        "crates/sim/src/exec.rs",
        "                    Front::Sass => self.hw_store_phase(inst).is_none(),\n",
        "                    Front::Sass => *space != Space::Global && self.hw_store_phase(inst).is_none(),\n",
        [("e_stg_keep", "good", "sass", []), ("e_stg_keep", "OUT", "sass", [])],
        [("ptxv-sim", "fixtures", "e_stg_keep_strict_sass_takes_every_lane")],
    ),
    # The profile explores one 32-bit STG's lanes (no measured order).
    SrcMutant(
        "stg-lowest-off",
        "crates/sim/src/hwprofile.rs",
        "            \"E-HW-STG-LOWEST\" => h.hw_stg_lowest = true,",
        "            \"E-HW-STG-LOWEST\" => {}",
        [],
        [("ptxv-sim", "fixtures", "e_stg_keep_strict_sass_takes_every_lane")],
    ),
    # The rule for a multicast STG too (one multicast shape measured: not encoded).
    SrcMutant(
        "stg-lowest-mcast",
        "crates/sim/src/exec.rs",
        "            } if self.launch.hw_stg_lowest\n                && self.launch.mcast.is_empty()\n",
        "            } if self.launch.hw_stg_lowest\n                && true\n",
        [],
        [("ptxv-sim", "fixtures", "e_mm_lane_multicast_lanes_take_any_order")],
    ),
    # The rule for STG.E.MMIO.SYS too (GB300 keeps the highest active lane's bytes there:
    # e_stg_keep MMIO_TRAP31 clean under the profile while GB300 traps in every launch;
    # probes/stg_mmio_lane).
    SrcMutant(
        "stg-lowest-mmio",
        "crates/sim/src/exec.rs",
        "                && !inst.text.contains(\".MMIO\") =>\n",
        "                && true =>\n",
        [],
        [("ptxv-sim", "fixtures", "e_stg_keep_strict_sass_takes_every_lane")],
    ),
    # The rule at every STG width (STG.64 by every lane: lane 0, GB300 keeps lane 16).
    SrcMutant(
        "stg-lowest-wide",
        "crates/sim/src/exec.rs",
        "                space: Space::Global,\n                bytes: 4,\n                ..\n",
        "                space: Space::Global,\n                ..\n",
        [],
        [("ptxv-sim", "fixtures", "e_stg_keep_strict_sass_takes_every_lane")],
    ),
]

# MISS 258: a MUFU node whose true function has no finite value at its operand's exact value (sqrt,
# rsqrt, lg2 of a negative number; lg2, rsqrt, rcp of 0) leaves the element's exact value
# undetermined: not analyzed, NaN error fields (fixtures/c_num_mufu_nan_f32).
MUFU_NAN = ("ptxv", "num_untracked", "mufu_results_without_a_real_value_are_not_analyzed")
# The same for a true value that is infinite or a pole (lg2, rsqrt, rcp of an exact 0;
# fixtures/c_num_mufu_inf_f32).
MUFU_INF = ("ptxv", "num_untracked", "mufu_results_without_a_finite_value_are_not_analyzed")
MUTANTS += [
    # The made-up exact value 0 is taken again: sqrt.approx.ftz of a negative subnormal is claimed
    # exact, ex2(lg2(.)) of it has exact 1, and -2's NaN results carry infinite errors.
    SrcMutant(
        "mufu-nan-exact",
        "crates/sim/src/num.rs",
        "            if self.mufu_exact_undetermined(arg, (func, ptx), (neg, abs)) {\n",
        "            if self.mufu_exact_undetermined(arg, (func, ptx), (neg, abs)) && false {\n",
        [],
        [MUFU_NAN, MUFU_INF],
    ),
    # Only a NaN true value counts again: lg2, rsqrt and rcp of an exact 0 get the exact value 0
    # (max(lg2.approx.ftz(x), -1000) claimed exact where the error is 1000).
    SrcMutant(
        "mufu-inf-exact",
        "crates/sim/src/num.rs",
        "        xe.is_finite() && !mufu_true(func, ptx, xe).is_finite()\n",
        "        xe.is_finite() && mufu_true(func, ptx, xe).is_nan()\n",
        [],
        [MUFU_INF],
    ),
    # out_rec stops walking MMA factors: a MUFU without a real value under one is claimed again
    # (fixtures/c_num_mufu_nan_mma: abs_err 0 against the exact value made up for it).
    SrcMutant(
        "mma-factor-mufu",
        "crates/sim/src/num.rs",
        "                    if !factor_undetermined\n",
        "                    if false\n",
        [],
        [("ptxv", "num_untracked", "mufu_under_an_mma_factor_without_a_real_value_is_not_analyzed")],
    ),
    # A non-finite element keeps the error and the exact value made up for it.
    SrcMutant(
        "nonfinite-nan-fields",
        "crates/sim/src/num.rs",
        "        let (exact_v, signed_err, abs_err, rel_err, err_ulps, thm1) = if nonfinite {\n",
        "        let (exact_v, signed_err, abs_err, rel_err, err_ulps, thm1) = if false {\n",
        [],
        [MUFU_NAN],
    ),
]

# MISS 259: a launch with the cluster dimension attribute is a cluster launch even at 1 x 1 x 1
# (the capture libraries record it as `cluster_attr`, and a kernel's required cluster shape too,
# MISS 281; python/ptxv/_casewriter.py `_launch_cluster`).
MUTANTS += [
    # The attribute is ignored again: a 1 x 1 x 1 launch with it gets no `cluster` line.
    SrcMutant(
        "casewriter-cluster-attr",
        "python/ptxv/_casewriter.py",
        '    return la.cluster != (1, 1, 1) or bool(la.record.get("cluster_attr"))\n',
        '    return la.cluster != (1, 1, 1)\n',
        [],
        py=[("python/tests", "test_casewriter_cluster.py", "attribute")],
    ),
    # tools/capture.py ignores the attribute again: a 1 x 1 x 1 launchcap record with it gets no
    # `cluster` line at the first launch or a [[launches]] entry, nor a cluster in the
    # description.
    SrcMutant(
        "capture-cluster-attr",
        "tools/capture.py",
        '    return r["cluster"] != [1, 1, 1] or bool(r.get("cluster_attr"))\n',
        '    return r["cluster"] != [1, 1, 1]\n',
        [],
        py=[("python/tests", "test_capture_cluster.py", "attribute")],
    ),
    # MISS 281: a kernel's required cluster shape other than the launch attribute's is no longer
    # named in the case description.
    SrcMutant(
        "casewriter-required-cluster-note",
        "python/ptxv/_casewriter.py",
        '    if rec.get("required_cluster"):\n',
        "    if False:\n",
        [],
        py=[("python/tests", "test_casewriter_cluster.py", "required_cluster")],
    ),
    SrcMutant(
        "capture-required-cluster-note",
        "tools/capture.py",
        '    if r.get("required_cluster"):\n',
        "    if False:\n",
        [],
        py=[("python/tests", "test_capture_cluster.py", "required_cluster")],
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
    why = anchor_problem(src, m.file, m.find)
    if why:
        return {"name": m.name, "status": "error", "why": why}
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


# Comment spans (start, end) of Rust and Python sources, for the anchor rule: a token scan in
# which string and character literals are code, so a comment marker inside one starts nothing.
_RS_TOKEN = re.compile(
    r"""//|/\*|b?r(#*)"|b?"|"""  # comments, raw strings, strings
    r"""b?'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]{1,6}\}|.)|[^\\'\n])'"""  # characters
)
_RS_STRING_END = re.compile(r'(?:\\.|[^"\\])*"', re.S)
_RS_BLOCK = re.compile(r"/\*|\*/")
_PY_TOKEN = re.compile(r"""#|'''|\"\"\"|'|\"""")
_PY_STRING_END = {
    "'''": re.compile(r"(?:\\.|(?!''').)*'''", re.S),
    '"""': re.compile(r'(?:\\.|(?!""").)*"""', re.S),
    "'": re.compile(r"(?:\\.|[^'\\\n])*'"),
    '"': re.compile(r'(?:\\.|[^"\\\n])*"'),
}


def comment_spans(text: str, file: str) -> list | None:
    """The comments of a Rust (`//` to the end of the line, nested `/* */`) or Python (`#` to the
    end of the line) source as sorted (start, end) spans; None for other files."""
    rust = file.endswith(".rs")
    if not rust and not file.endswith(".py"):
        return None
    spans, i, n = [], 0, len(text)
    while (t := (_RS_TOKEN if rust else _PY_TOKEN).search(text, i)) is not None:
        s, tok = t.start(), t.group(0)
        if tok in ("//", "#"):
            e = text.find("\n", s)
            i = n if e < 0 else e
            spans.append((s, i))
        elif tok == "/*":
            depth, j = 1, t.end()
            while depth and (b := _RS_BLOCK.search(text, j)) is not None:
                depth += 1 if b.group(0) == "/*" else -1
                j = b.end()
            i = j if depth == 0 else n
            spans.append((s, i))
        elif rust and t.group(1) is not None:
            # A raw string: it ends at the first quote followed by as many hashes.
            e = text.find('"' + t.group(1), t.end())
            i = n if e < 0 else e + 1 + len(t.group(1))
        elif rust and tok.endswith('"'):
            e = _RS_STRING_END.match(text, t.end())
            i = e.end() if e else n
        elif rust:
            i = t.end()  # a character literal
        else:
            e = _PY_STRING_END[tok].match(text, t.end())
            i = e.end() if e else t.end()
    return spans


def anchor_problem(text: str, file: str, find: str, spans: list | None = None) -> str | None:
    """Why `find` is not an anchor in `text` (the source of `file`), or None: it must occur
    exactly once (a mutant replaces every occurrence), and not only inside a comment (a mutant
    there would edit the comment). `spans`: comment_spans(text, file), when known."""
    n = text.count(find)
    if n != 1:
        return f"find text occurs {n} times"
    if spans is None:
        spans = comment_spans(text, file) or []
    s = text.index(find)
    e = s + len(find)
    inside = [(a, b) for a, b in spans if a < e and b > s]
    code = (k for k, c in enumerate(text[s:e], s) if not c.isspace())
    if inside and all(any(a <= k < b for a, b in inside) for k in code):
        return "find text occurs once, inside a comment"
    return None


def fb_only_skipped(tree: Path, mutants: list) -> list:
    """The mutants among `mutants` that `tree` cannot hold: those of files under fb/ (internal
    tooling, `SrcMutant.fb_only`) that it lacks, in a tree that also lacks the export manifest
    fb/OSS_EXCLUDE.txt, which makes it an open-source export (the export drops fb/). In a tree
    with the manifest, the internal one, a missing fb/ file is a broken anchor like any other."""
    if (tree / "fb" / "OSS_EXCLUDE.txt").is_file():
        return []
    return [m for m in mutants if m.fb_only and not (tree / m.file).is_file()]


def broken_anchors(tree: Path, mutants: list) -> list:
    """(mutant, why) for every mutant whose find text is not an anchor in its file under `tree`
    (anchor_problem: the rule run_mutant_in_tree applies before it builds). Each file is read
    and scanned once per call. Also every fixture run whose case (a directory name under this
    checkout's fixtures/, as a run reads it) or variant does not exist: a run reads its
    expectation only after the mutant is built, and a missing one crashed the whole run (until
    2026-11-12 three mutants named their case `fixtures/<case>`)."""
    bad = []
    files = {}
    for m in mutants:
        if m.file not in files:
            f = tree / m.file
            files[m.file] = (f.read_text(), None) if f.is_file() else None
        if files[m.file] is None:
            bad.append((m, "no such file"))
            continue
        text, spans = files[m.file]
        if spans is None:
            spans = comment_spans(text, m.file) or []
            files[m.file] = (text, spans)
        why = anchor_problem(text, m.file, m.find, spans)
        if why:
            bad.append((m, why))
        for case, variant, front, _ in m.runs:
            try:
                expect_for(ROOT / "fixtures" / case, variant, front)
            except (OSError, KeyError, tomllib.TOMLDecodeError) as e:
                bad.append((m, f"run {case} {variant} {front}: {type(e).__name__} {e}"))
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, help="output directory (required unless --check-anchors)")
    ap.add_argument("--only", help="comma-separated mutant names")
    ap.add_argument("--check-anchors", action="store_true",
                    help="print the mutants whose find text does not occur exactly once in its file, "
                    "or only inside a comment, exit 1 if there is one; builds and runs nothing")
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
    # An open-source export has no fb/: its mutants are skipped there, with a count (a run
    # needs the files too). Every other mutant stays chosen; if none is left, that is an error.
    skipped = fb_only_skipped(ROOT if a.root is None else a.root, chosen)
    if skipped:
        print(f"srcmut: fb-only mutants skipped: {len(skipped)} (files under fb/, which this tree "
              "does not have)", file=sys.stderr)
        gone = {m.name for m in skipped}
        chosen = [m for m in chosen if m.name not in gone]
        if not chosen:
            print("srcmut: no mutant is chosen: every one is fb-only", file=sys.stderr)
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

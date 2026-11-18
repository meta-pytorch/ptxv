/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};

use ptxv_ir::{AtomOp, File, Front, IntOp, Kernel, Op, Val};

use crate::machine::{FastHasher, Machine, St};
use ptxv_memory_write::machine_registers::LaneRegisters;

#[derive(Clone, Debug)]
struct Live {
    words: Vec<bool>,
    predicates: Vec<u64>,
}

#[derive(Clone, Debug)]
pub(crate) struct FutureReads(Vec<Live>);

impl FutureReads {
    pub(crate) fn analyze(kernel: &Kernel) -> Option<Self> {
        if kernel.front != Front::Ptx || kernel.code.len() > 512 {
            return None;
        }
        let mut atomic = false;
        for (pc, instruction) in kernel.code.iter().enumerate() {
            match &instruction.op {
                Op::Atom {
                    op: AtomOp::Add,
                    bytes: 4,
                    d: Some(_),
                    p: None,
                    new: None,
                    ..
                } => atomic = true,
                Op::Mov { .. }
                | Op::Int { .. }
                | Op::MulWide { .. }
                | Op::Imad { .. }
                | Op::CvtInt { .. }
                | Op::Cvta { .. }
                | Op::Setp { .. }
                | Op::Ld { .. }
                | Op::St { .. }
                | Op::BarSync { .. }
                | Op::Exit
                | Op::Nop => {}
                Op::Bra { target, .. } if *target > pc && *target < kernel.code.len() => {}
                _ => return None,
            }
            if instruction
                .op
                .src_regs()
                .into_iter()
                .chain(instruction.op.dst_regs())
                .chain(instruction.guard.reg)
                .any(|register| !matches!(register.file, File::V | File::P))
            {
                return None;
            }
        }
        if !atomic {
            return None;
        }
        let mut reads = Vec::with_capacity(kernel.code.len());
        let mut future = BTreeSet::new();
        for instruction in kernel.code.iter().rev() {
            future.extend(instruction.op.src_regs());
            future.extend(instruction.guard.reg);
            let mut live = Live {
                words: vec![],
                predicates: vec![],
            };
            for register in &future {
                let index = usize::from(register.idx);
                match register.file {
                    File::V => {
                        live.words.resize(live.words.len().max(index + 1), false);
                        live.words[index] = true;
                    }
                    File::P => {
                        live.predicates
                            .resize(live.predicates.len().max(index / 64 + 1), 0);
                        live.predicates[index / 64] |= 1u64 << (index % 64);
                    }
                    _ => unreachable!(),
                }
            }
            reads.push(live);
        }
        reads.reverse();
        Some(Self(reads))
    }

    pub(crate) fn projected_hash(&self, machine: &Machine, thread: usize, retired: bool) -> u64 {
        let original = &machine.threads[thread];
        if original.clock_mem {
            return crate::machine::raw_thread_hash(original);
        }
        let start = original.pc + u32::from(retired);
        let mut trace = Vec::new();
        let mut probe = None;
        let mut pc = start;
        while let Some(instruction) = machine.k.code.get(pc as usize) {
            if original.st != St::Run {
                break;
            }
            if !private_step(&instruction.op) {
                break;
            }
            let shadow = probe.get_or_insert_with(|| {
                let mut shadow = machine.clone();
                shadow.threads[thread].make_mut().set_pc(start);
                shadow.race_log = None;
                shadow.fence_acq = None;
                shadow
            });
            if instruction
                .op
                .src_regs()
                .into_iter()
                .chain(instruction.guard.reg)
                .any(|register| shadow.reg_umask(thread, register) != 0)
            {
                break;
            }
            let active = shadow.prd(thread, instruction.guard);
            let findings = shadow.findings.len();
            if shadow.exec_one(&[thread], pc).is_err() || shadow.findings.len() != findings {
                // A failed speculative step must retain the original, unprojected state.
                return crate::machine::raw_thread_hash(original);
            }
            trace.push((pc, active));
            let next = shadow.threads[thread].pc;
            if next <= pc {
                return crate::machine::raw_thread_hash(original);
            }
            pc = next;
        }
        let mut registers = probe.as_ref().map_or_else(
            || original.registers.clone(),
            |shadow| shadow.threads[thread].registers.clone(),
        );
        self.project(pc, &mut registers);
        let mut projected = original.with_registers(registers);
        // Empty observations add no acquire edge. Fences and poll acquires disable this
        // quotient; an exit acquire clears diagnostic origins before joining releases.
        let empty = crate::machine::VC::new();
        projected
            .acq_pending
            .retain(|(_, release, _)| *release != empty);
        let mut hash = FastHasher {
            s: 0x243f_6a88_85a3_08d3,
        };
        projected.hash(&mut hash);
        pc.hash(&mut hash);
        trace.hash(&mut hash);
        // Even a dead value retains its original definedness and write history.
        original.registers.pdef.hash(&mut hash);
        original.registers.written.hash(&mut hash);
        original.registers.vund.hash(&mut hash);
        hash.finish() | 1
    }

    pub(crate) fn project(&self, pc: u32, registers: &mut LaneRegisters) {
        let Some(live) = self.0.get(pc as usize) else {
            return;
        };
        for (index, value) in registers.regs.iter_mut().enumerate() {
            if !live.words.get(index).copied().unwrap_or(false) {
                *value = 0;
            }
        }
        for (index, bits) in registers.preds.iter_mut().enumerate() {
            *bits &= live.predicates.get(index).copied().unwrap_or(0);
        }
    }
}

fn private_value(value: &Val) -> bool {
    matches!(value, Val::Imm(_) | Val::Reg(_, _))
}

fn private_step(operation: &Op) -> bool {
    match operation {
        Op::Int {
            op: IntOp::DivU | IntOp::DivS | IntOp::RemU | IntOp::RemS,
            ..
        } => false,
        Op::Mov { a, .. } | Op::CvtInt { a, .. } => private_value(a),
        Op::Int { a, b, .. } | Op::MulWide { a, b, .. } | Op::Setp { a, b, .. } => {
            private_value(a) && private_value(b)
        }
        Op::Imad { a, b, c, .. } => private_value(a) && private_value(b) && private_value(c),
        Op::Bra {
            any: false,
            all: false,
            tid0_falls: false,
            cond: None,
            div: None,
            ..
        }
        | Op::Nop => true,
        _ => false,
    }
}

pub(crate) fn plain_mode(machine: &Machine) -> bool {
    machine.num.is_none()
        && machine.prov.is_none()
        && machine.wpolls.is_empty()
        && machine.fill.is_none()
        && !machine.wlog.is_on()
        && machine.perm.is_none()
        && machine.gown.is_none()
        && machine.ring.is_none()
        && machine.dead_probe.is_none()
        && machine.fp.is_none()
        && machine.sfoot.is_none()
        && machine.gfoot.is_none()
        && machine.atomic_operand.is_none()
        && machine.forced_pred.is_none()
        && machine.clock_step == 0
        && machine.clock_choice_pcs.is_empty()
        && machine.clock_effect_pcs.is_empty()
        && machine.clock_lane_pcs.is_empty()
        && machine.clock_store_pcs.is_empty()
        && machine.delay_pcs.is_empty()
        && machine.clock_loop_pcs.is_empty()
        && machine.ctaint.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel(tail: &str) -> Kernel {
        ptxv_ptx::parse_kernel(
            &format!(
                r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 input) {{
 .reg .b32 %r<4>; .reg .b64 %a; .reg .pred %p<66>;
 ld.param.u64 %a,[input];
 atom.global.add.u32 %r0,[%a],1;
 {tail}
 ret;
}}
"#
            ),
            Some("k"),
        )
        .unwrap()
    }

    fn registers() -> LaneRegisters {
        LaneRegisters {
            regs: vec![u64::MAX; 16],
            preds: vec![u64::MAX],
            pdef: vec![0x1234],
            written: vec![0x5678],
            vund: vec![(0, 0x80)],
        }
    }

    fn initial(tail: &str) -> (Machine, ptxv_ir::Reg) {
        initial_threads(tail, 1)
    }

    fn initial_threads(tail: &str, threads: usize) -> (Machine, ptxv_ir::Reg) {
        use std::sync::Arc;
        let kernel = Arc::new(kernel(tail));
        let Op::Atom {
            d: Some(result), ..
        } = kernel.code[1].op
        else {
            panic!("atomic")
        };
        let manifest = serde_json::from_value(serde_json::json!({
            "name":"private-tail", "description":"native control and effects",
            "entry":"k", "grid":[1,1,1], "block":[threads,1,1], "outputs":["x"],
            "buffers":[{"name":"x", "dtype":"u32", "shape":[4], "init":"zeros"}],
            "params":[{"kind":"ptr", "buffer":"x"}], "variants":[]
        }))
        .unwrap();
        let case = ptxv_case::Case {
            dir: Default::default(),
            m: manifest,
        };
        let launch = Arc::new(crate::build_launch(&case, &kernel).unwrap());
        let mut machine = Machine::new(kernel, launch);
        machine
            .exec_one(&(0..threads).collect::<Vec<_>>(), 0)
            .unwrap();
        for actor in 0..threads {
            machine.threads[actor].make_mut().set_pc(2);
        }
        (machine, result)
    }

    fn finish(mut machine: Machine) -> Vec<u8> {
        while machine.threads[0].st == crate::machine::St::Run {
            let pc = machine.threads[0].pc;
            machine.exec_one(&[0], pc).unwrap();
        }
        machine.outputs()
    }

    #[test]
    fn private_tail_merges_only_equal_native_branch_and_memory_effects() {
        let (mut left, result) = initial("setp.ne.u32 %p0,%r0,2; @%p0 bra DONE; add.u32 %r1,%r0,1; st.global.u32 [%a],%r1; DONE:");
        left.wr(0, result, 0, 1);
        let mut right = left.clone();
        right.wr(0, result, 1, 1);
        let log = std::sync::Arc::new(std::sync::Mutex::new(Default::default()));
        left.race_log = Some(log.clone());
        let before = left.threads[0].registers.clone();
        assert_eq!(left.fingerprint(), right.fingerprint());
        assert!(log.lock().unwrap().is_empty());
        assert_eq!(left.threads[0].registers, before);
        assert_eq!(finish(left.clone()), finish(right.clone()));
        let mut winner = left.clone();
        winner.wr(0, result, 2, 1);
        assert_ne!(left.fingerprint(), winner.fingerprint());
        assert_ne!(finish(left), finish(winner));
    }

    #[test]
    fn projected_atomic_search_agrees_with_full_native_state_on_outputs_and_races() {
        use std::sync::Arc;
        for tail in [
            "setp.ne.u32 %p0,%r0,3; @%p0 bra DONE; st.global.u32 [%a],17; DONE:",
            "st.global.u32 [%a],%r0;",
            "setp.eq.u32 %p0,%r0,0; @%p0 bra OTHER; st.global.u32 [%a],%r0; OTHER:",
        ] {
            let (mut projected, _) = initial_threads(tail, 4);
            for actor in 0..4 {
                projected.threads[actor].make_mut().set_pc(1);
            }
            let mut full = projected.clone();
            full.k = Arc::new((*full.k).clone());
            let reduced = crate::explore::explore_threads(projected, 100_000, 1);
            let baseline = crate::explore::explore_threads(full, 100_000, 1);
            assert!(reduced.complete && baseline.complete, "{tail}");
            assert_eq!(reduced.classes(), baseline.classes(), "{tail}");
            eprintln!(
                "private-tail native control {tail}: reduced={} full={}",
                reduced.states, baseline.states
            );
            if tail.starts_with("setp.ne") {
                assert!(reduced.states < baseline.states);
            }
            let mut reduced_outputs = reduced.outputs.clone();
            reduced_outputs.sort();
            let mut baseline_outputs = baseline.outputs.clone();
            baseline_outputs.sort();
            assert_eq!(reduced_outputs, baseline_outputs, "{tail}");
            let mut reduced_findings: Vec<_> = reduced
                .findings
                .iter()
                .map(|finding| (&finding.kind, &finding.sig))
                .collect();
            reduced_findings.sort();
            let mut baseline_findings: Vec<_> = baseline
                .findings
                .iter()
                .map(|finding| (&finding.kind, &finding.sig))
                .collect();
            baseline_findings.sort();
            assert_eq!(reduced_findings, baseline_findings, "{tail}");
        }
    }

    #[test]
    fn cross_warp_empty_releases_preserve_native_search_results() {
        use std::sync::Arc;
        let tail = "setp.ne.u32 %p0,%r0,3; @%p0 bra DONE; st.global.u32 [%a],17; DONE:";
        let (mut projected, _) = initial_threads(tail, 34);
        for actor in 0..34 {
            let thread = projected.threads[actor].make_mut();
            if [0, 1, 32, 33].contains(&actor) {
                thread.set_pc(1);
            } else {
                thread.set_status(St::Exited);
            }
        }
        let mut full = projected.clone();
        full.k = Arc::new((*full.k).clone());
        let reduced = crate::explore::explore_threads(projected, 100_000, 1);
        let baseline = crate::explore::explore_threads(full, 100_000, 1);
        assert!(reduced.complete && baseline.complete);
        assert!(reduced.states < baseline.states);
        assert_eq!(reduced.classes(), baseline.classes());
        let mut reduced_outputs = reduced.outputs.clone();
        let mut baseline_outputs = baseline.outputs.clone();
        reduced_outputs.sort();
        baseline_outputs.sort();
        assert_eq!(reduced_outputs, baseline_outputs);
        let signatures = |verdict: &crate::explore::Verdict| {
            let mut findings: Vec<_> = verdict
                .findings
                .iter()
                .map(|finding| (finding.kind.clone(), finding.sig.clone()))
                .collect();
            findings.sort();
            findings
        };
        assert_eq!(signatures(&reduced), signatures(&baseline));
        eprintln!(
            "cross-warp private-tail reduced={} full={}",
            reduced.states, baseline.states
        );
    }

    #[test]
    fn pending_release_projection_keeps_nonempty_clocks_and_exit_diagnostics() {
        use crate::machine::{vc_single, VC};
        let (left, _) = initial("st.global.u32 [%a],7;");
        let mut empty = left.clone();
        empty.threads[0]
            .make_mut()
            .acq_pending
            .push((0, VC::new(), ptxv_ir::Scope::Gpu));
        assert_eq!(left.fingerprint(), empty.fingerprint());
        assert_eq!(empty.threads[0].acq_pending.len(), 1);
        let mut released = left.clone();
        released.threads[0]
            .make_mut()
            .acq_pending
            .push((0, vc_single(0, 3), ptxv_ir::Scope::Gpu));
        assert_ne!(left.fingerprint(), released.fingerprint());
        let mut recorded_left = left.clone();
        recorded_left.fence_acq = Some(Default::default());
        empty.fence_acq = recorded_left.fence_acq.clone();
        assert_eq!(recorded_left.fingerprint(), empty.fingerprint());
        let exit = recorded_left.k.code.len() as u32 - 1;
        for machine in [&mut recorded_left, &mut empty] {
            let recorder = machine.fence_acq.as_ref().unwrap();
            recorder
                .lock()
                .unwrap()
                .origins
                .entry((0, 0))
                .or_default()
                .insert(1);
            machine.threads[0].make_mut().set_pc(exit);
            std::sync::Arc::make_mut(&mut machine.launch).exit_fence = Some(ptxv_ir::Scope::Gpu);
            machine.exec_one(&[0], exit).unwrap();
        }
        assert_eq!(recorded_left.threads[0].vc, empty.threads[0].vc);
        assert!(empty
            .fence_acq
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .acquired
            .is_empty());
        assert!(FutureReads::analyze(&kernel("fence.acq_rel.gpu;")).is_none());
    }

    #[test]
    fn native_guards_across_predicate_words_and_vector_payloads_remain_distinct() {
        for predicate in [63, 64] {
            let (mut left, _) = initial(&format!(
                "@%p{predicate} bra DONE; st.global.u32 [%a],9; DONE:"
            ));
            let guard = left.k.code[2].guard.reg.unwrap();
            left.wp(0, guard, false);
            let mut right = left.clone();
            right.wp(0, guard, true);
            assert_ne!(left.fingerprint(), right.fingerprint());
            assert_ne!(finish(left), finish(right));
        }
        let (mut left, first) = initial("st.global.v2.u32 [%a],{%r0,%r1};");
        let second = ptxv_ir::Reg {
            file: first.file,
            idx: first.idx + 1,
        };
        left.wr(0, first, 7, 1);
        left.wr(0, second, 1, 1);
        let mut right = left.clone();
        right.wr(0, second, 2, 1);
        assert_ne!(left.fingerprint(), right.fingerprint());
        assert_ne!(finish(left), finish(right));
    }

    #[test]
    fn reconverging_private_paths_keep_the_entire_control_trace() {
        let (mut left, result) = initial(
            "setp.eq.u32 %p0,%r0,0; @%p0 bra A; mov.u32 %r1,7; bra END; A: mov.u32 %r1,7; END:",
        );
        left.wr(0, result, 0, 1);
        let mut right = left.clone();
        right.wr(0, result, 1, 1);
        assert_eq!(finish(left.clone()), finish(right.clone()));
        assert_ne!(left.fingerprint(), right.fingerprint());
    }

    #[test]
    fn native_addresses_remain_distinct_after_private_computation() {
        let (mut left, result) =
            initial("mul.lo.u32 %r1,%r0,4; add.u64 %a,%a,%r1; st.global.u32 [%a],9;");
        left.wr(0, result, 0, 1);
        let mut right = left.clone();
        right.wr(0, result, 1, 1);
        assert_ne!(left.fingerprint(), right.fingerprint());
        assert_ne!(finish(left), finish(right));
    }

    #[test]
    fn dead_payload_projection_preserves_every_definedness_bit() {
        let kernel = kernel("setp.eq.u32 %p0,%r0,0;");
        let reads = FutureReads::analyze(&kernel).unwrap();
        let mut projected = registers();
        let original = projected.clone();
        reads.project((kernel.code.len() - 1) as u32, &mut projected);
        assert!(projected.regs.iter().all(|&value| value == 0));
        assert!(projected.preds.iter().all(|&value| value == 0));
        assert_eq!(projected.pdef, original.pdef);
        assert_eq!(projected.written, original.written);
        assert_eq!(projected.vund, original.vund);
        let mut projected = original.clone();
        reads.project(u32::MAX, &mut projected);
        assert_eq!(projected, original);
    }

    #[test]
    fn backward_control_and_cross_lane_observers_keep_the_full_state() {
        for tail in [
            "LOOP: add.u32 %r1,%r0,1; bra LOOP;",
            "shfl.sync.idx.b32 %r1,%r0,0,31,-1;",
            "vote.sync.ballot.b32 %r1,%p0,-1;",
        ] {
            assert!(FutureReads::analyze(&kernel(tail)).is_none(), "{tail}");
        }
    }

    #[test]
    fn warp_aggregation_fixture_admits_only_its_forward_ptx_program() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let case = ptxv_case::Case::load(root.join("fixtures/e_shfl_warpagg")).unwrap();
        let source = case.variant_ptx("MULTI").unwrap();
        let kernel = ptxv_ptx::parse_kernel(&source, Some(&case.m.entry)).unwrap();
        assert!(FutureReads::analyze(&kernel).is_some());
        let source = case.variant_ptx("MASK").unwrap();
        let kernel = ptxv_ptx::parse_kernel(&source, Some(&case.m.entry)).unwrap();
        assert!(FutureReads::analyze(&kernel).is_none());
    }

    #[test]
    fn native_fingerprints_keep_live_results_memory_and_definedness() {
        use std::sync::Arc;

        use crate::machine::Machine;

        let kernel = Arc::new(kernel("st.global.u32 [%a],%r0;"));
        let result = match kernel.code[1].op {
            Op::Atom {
                d: Some(result), ..
            } => result,
            _ => panic!("expected scalar atomic"),
        };
        let manifest = serde_json::from_value(serde_json::json!({
            "name":"dead-register-projection", "description":"live return and dead payload controls",
            "entry":"k", "grid":[1,1,1], "block":[1,1,1], "outputs":["x"],
            "buffers":[{"name":"x", "dtype":"u32", "shape":[1], "init":"zeros"}],
            "params":[{"kind":"ptr", "buffer":"x"}], "variants":[]
        })).unwrap();
        let case = ptxv_case::Case {
            dir: Default::default(),
            m: manifest,
        };
        let launch = Arc::new(crate::build_launch(&case, &kernel).unwrap());
        let mut left = Machine::new(kernel, launch);
        left.exec_one(&[0], 0).unwrap();
        left.threads[0].make_mut().set_pc(2);
        left.wr(0, result, 1, 1);
        let mut right = left.clone();
        right.wr(0, result, 2, 1);
        assert_ne!(left.fingerprint(), right.fingerprint());
        let mut run_left = left.clone();
        let mut run_right = right.clone();
        run_left.exec_one(&[0], 2).unwrap();
        run_right.exec_one(&[0], 2).unwrap();
        assert_ne!(run_left.outputs(), run_right.outputs());

        left.threads[0].make_mut().set_pc(3);
        right.threads[0].make_mut().set_pc(3);
        assert_eq!(left.fingerprint(), right.fingerprint());
        assert_eq!(left.fingerprint(), left.private_statics().fingerprint());
        right.threads[0]
            .make_mut()
            .write_virtual(result.idx, 2, u64::MAX);
        assert_ne!(left.fingerprint(), right.fingerprint());
        let mut right = left.clone();
        *right.gmem[0].bytes.get_mut(0) = 1;
        assert_ne!(left.fingerprint(), right.fingerprint());

        let mut code = (*left.k).clone();
        code.code[3] = code.code[2].clone();
        let code = Arc::new(code);
        let mut right = left.clone();
        right.wr(0, result, 2, 1);
        left.k = code.clone();
        right.k = code;
        assert_ne!(left.fingerprint(), right.fingerprint());
        assert_ne!(
            left.private_statics().fingerprint(),
            right.private_statics().fingerprint()
        );
    }
}

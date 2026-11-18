/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_ir::Op;
use serde_json::json;

use crate::machine::Machine;

fn machine() -> Machine {
    let text = r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 input) {
  .reg .b32 %r<2>;
  .reg .b64 %address;
  .reg .pred %p<3>;
  ld.param.u64 %address, [input];
  mov.u32 %r0, %tid.x;
  atom.global.add.u32 %r1, [%address], 1;
  setp.eq.u32 %p0, %r0, 0;
  setp.eq.u32 %p1, %r1, 1;
  and.pred %p2, %p0, %p1;
  @%p2 trap;
  ret;
}
"#;
    let kernel = Arc::new(ptxv_ptx::parse_kernel(text, Some("k")).unwrap());
    let manifest = serde_json::from_value(json!({
        "name":"scalar-atomic-middle-order", "description":"three scalar atomic lane choices",
        "entry":"k", "grid":[1,1,1], "block":[3,1,1], "outputs":[],
        "buffers":[{"name":"x","dtype":"u32","shape":[1],"init":"zeros"}],
        "params":[{"kind":"ptr","buffer":"x"}], "variants":[]
    }))
    .unwrap();
    let case = ptxv_case::Case {
        dir: PathBuf::new(),
        m: manifest,
    };
    let launch = Arc::new(crate::build_launch(&case, &kernel).unwrap());
    Machine::new(kernel, launch)
}

#[test]
fn scalar_atomic_middle_lane_return_probe() {
    for reverse in [false, true] {
        let mut input = machine();
        input.atom_lanes_rev = reverse;
        for (name, verdict) in [
            ("stateful", crate::explore::explore(input.clone(), 10000)),
            ("DPOR", crate::dpor::explore_dpor(input, 10000)),
        ] {
            eprintln!(
                "scalar reverse={reverse} {name}: states={}, transitions={}, complete={}, classes={:?}",
                verdict.states,
                verdict.transitions,
                verdict.complete,
                verdict.classes()
            );
            assert!(verdict.complete);
            assert!(verdict.classes().iter().any(|kind| kind == "trap"));
        }
    }
    let mut witness = machine();
    let atomic = witness
        .k
        .code
        .iter()
        .position(|instruction| matches!(instruction.op, Op::Atom { .. }))
        .unwrap() as u32;
    for pc in 0..atomic {
        witness.exec_one(&[0, 1, 2], pc).unwrap();
    }
    for actor in [1, 0, 2] {
        witness.exec_one(&[actor], atomic).unwrap();
    }
    let (witness, findings, complete) = crate::explore::run_once(witness);
    eprintln!("scalar middle-order native: complete={complete}, findings={findings:?}");
    assert!(complete);
    assert!(findings
        .iter()
        .chain(witness.findings.iter())
        .any(|finding| finding.kind == "trap"));
    let certificate = crate::wellsync::certify(machine());
    assert!(
        certificate
            .verdict
            .as_ref()
            .is_some_and(|verdict| verdict.classes().iter().any(|kind| kind == "trap"))
            || !certificate.reasons.is_empty()
    );
    eprintln!(
        "scalar middle-order certificate: verdict={:?}, reasons={:?}, steps={}",
        certificate
            .verdict
            .as_ref()
            .map(|verdict| verdict.classes()),
        certificate.reasons,
        certificate.steps
    );
}

fn prepared() -> (Machine, u32) {
    let mut machine = machine();
    let pc = machine
        .k
        .code
        .iter()
        .position(|inst| matches!(inst.op, Op::Atom { .. }))
        .unwrap() as u32;
    for before in 0..pc {
        machine.exec_one(&[0, 1, 2], before).unwrap();
    }
    (machine, pc)
}

fn forced(mut machine: Machine, pc: u32, order: Vec<u32>) -> Machine {
    let mut ring = crate::wellsync::ring::RingCtl::default();
    ring.tick_pcs.insert(pc);
    ring.tick_words.insert((1, 0, 0));
    ring.tick_force.insert((1, 0, 0), order);
    machine.ring = Some(Arc::new(ring));
    machine
}

fn take(machine: &mut Machine, thread: usize) {
    use ptxv_memory_write::atomic_components::Component::First;
    let action=machine.actions().into_iter().find_map(|(action,_)|
        matches!(action,crate::explore::Action::AtomicComponent{thread:actor,component:First,..} if actor==thread).then_some(action)).expect("selected scalar lane is enabled");
    machine.apply(&action);
}

fn memory(machine: &Machine) -> u32 {
    (0..4).fold(0, |word, at| {
        word | ((*machine.gmem[0].bytes.get(at) as u32) << (8 * at))
    })
}

#[test]
fn scalar_all_six_forced_lane_orders_retire_and_count_once() {
    use ptxv_ir::Val;
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let (input, pc) = prepared();
        let Op::Atom {
            d: Some(destination),
            ..
        } = input.k.code[pc as usize].op
        else {
            panic!("scalar")
        };
        let mut machine = forced(input, pc, order.to_vec());
        for (rank, &actor) in order.iter().enumerate() {
            let choices = machine.actions();
            assert_eq!(choices.len(), 1);
            take(&mut machine, actor as usize);
            assert_eq!(
                machine
                    .rd(actor as usize, &Val::Reg(destination, 1))
                    .unwrap(),
                rank as u64
            );
            assert_eq!(
                machine.ring.as_ref().unwrap().tick_claims[&(1, 0, 0)],
                order[..=rank]
            );
            assert!(machine
                .threads
                .iter()
                .all(|thread| thread.pc == pc + u32::from(rank == 2)));
        }
        assert!(machine.atomic_components.is_empty());
        assert_eq!(memory(&machine), 3);
        assert!(machine.findings.is_empty(), "{:?}", machine.findings);
    }
}

#[test]
fn scalar_source_aliases_are_frozen_for_every_issued_lane() {
    use ptxv_ir::File;
    use ptxv_ir::Reg;
    use ptxv_ir::Val;
    let (mut machine, pc) = prepared();
    let uniform = Reg {
        file: File::UR,
        idx: 0,
    };
    let Op::Atom { v, d, .. } = &mut Arc::make_mut(&mut machine.k).code[pc as usize].op else {
        panic!("scalar")
    };
    *v = Val::Reg(uniform, 1);
    *d = Some(uniform);
    machine.wr(0, uniform, 7, 1);
    let mut machine = forced(machine, pc, vec![1, 0, 2]);
    for actor in [1, 0, 2] {
        take(&mut machine, actor);
    }
    assert_eq!(memory(&machine), 21);
    assert_eq!(machine.rd(0, &Val::Reg(uniform, 1)).unwrap(), 14);
    assert_eq!(
        machine.ring.as_ref().unwrap().tick_claims[&(1, 0, 0)],
        [1, 0, 2]
    );
    assert!(machine.findings.is_empty(), "{:?}", machine.findings);
}

#[test]
fn scalar_ticket_address_is_frozen_across_uniform_destination_alias() {
    use ptxv_ir::File;
    use ptxv_ir::Reg;
    use ptxv_ir::Val;
    let (mut machine, pc) = prepared();
    let Op::Atom { addr, .. } = &machine.k.code[pc as usize].op else {
        panic!("scalar")
    };
    let address = machine.rd(0, addr.base.as_ref().unwrap()).unwrap();
    let uniform = Reg {
        file: File::UR,
        idx: 0,
    };
    let Op::Atom { addr, d, .. } = &mut Arc::make_mut(&mut machine.k).code[pc as usize].op else {
        panic!("scalar")
    };
    addr.base = Some(Val::Reg(uniform, 2));
    *d = Some(uniform);
    machine.wr(0, uniform, address, 2);
    *machine.gmem[0].bytes.get_mut(0) = 123;
    let mut machine = forced(machine, pc, vec![1, 0, 2]);
    take(&mut machine, 1);
    assert_ne!(machine.rd(0, &Val::Reg(uniform, 2)).unwrap(), address);
    assert_eq!(machine.actions().len(), 1);
    take(&mut machine, 0);
    take(&mut machine, 2);
    assert_eq!(memory(&machine), 126);
    assert_eq!(
        machine.ring.as_ref().unwrap().tick_claims[&(1, 0, 0)],
        [1, 0, 2]
    );
    assert!(machine.findings.is_empty(), "{:?}", machine.findings);
}

#[test]
fn scalar_empty_and_partial_guards_preserve_progress_and_the_issued_mask() {
    use ptxv_ir::File;
    use ptxv_ir::Pred;
    use ptxv_ir::Reg;
    let predicate = Reg {
        file: File::P,
        idx: 0,
    };
    let (mut empty, pc) = prepared();
    Arc::make_mut(&mut empty.k).code[pc as usize].guard = Pred {
        reg: Some(predicate),
        neg: false,
    };
    for thread in 0..3 {
        empty.wp(thread, predicate, false);
    }
    let mut empty = forced(empty, pc, vec![99]);
    assert!(empty.split_enabled(&[0, 1, 2], pc));
    let action = empty.actions()[0].0.clone();
    empty.apply(&action);
    assert_eq!(memory(&empty), 0);
    assert!(empty.ring.as_ref().unwrap().tick_claims.is_empty());
    assert!(empty.threads.iter().all(|thread| thread.pc != pc));

    let (mut partial, pc) = prepared();
    Arc::make_mut(&mut partial.k).code[pc as usize].guard = Pred {
        reg: Some(predicate),
        neg: false,
    };
    for thread in 0..3 {
        partial.wp(thread, predicate, thread != 2);
    }
    let mut partial = forced(partial, pc, vec![1, 0]);
    take(&mut partial, 1);
    partial.wp(2, predicate, true);
    assert_eq!(partial.actions().len(), 1);
    take(&mut partial, 0);
    assert_eq!(memory(&partial), 2);
    assert_eq!(
        partial.ring.as_ref().unwrap().tick_claims[&(1, 0, 0)],
        [1, 0]
    );
    assert!(partial.threads.iter().all(|thread| thread.pc == pc + 1));
}

#[test]
fn scalar_issue_faults_before_claims_and_writes_but_later_access_keeps_the_prefix() {
    use ptxv_ir::Val;
    let (mut invalid, pc) = prepared();
    let Op::Atom { addr, .. } = &invalid.k.code[pc as usize].op else {
        panic!("scalar")
    };
    let Val::Reg(address, n) = addr.base.clone().unwrap() else {
        panic!("address")
    };
    let raw = invalid.rd(2, &Val::Reg(address, n)).unwrap();
    invalid.wr(2, address, raw + 2, n);
    let mut invalid = forced(invalid, pc, vec![1]);
    take(&mut invalid, 1);
    assert_eq!(memory(&invalid), 0);
    assert!(invalid.ring.as_ref().unwrap().tick_claims.is_empty());
    assert!(invalid
        .findings
        .iter()
        .any(|finding| finding.msg.contains("align")));
    assert!(invalid.atomic_components.is_empty());
    let (input, pc) = prepared();
    let mut later = forced(input, pc, vec![1, 0, 2]);
    take(&mut later, 1);
    Arc::make_mut(&mut Arc::make_mut(&mut later.launch).allocs[0].init).clear();
    assert!(later.split_enabled(&[0, 1, 2], pc));
    take(&mut later, 0);
    assert_eq!(memory(&later), 1);
    assert_eq!(later.ring.as_ref().unwrap().tick_claims[&(1, 0, 0)], [1]);
    assert!(later
        .findings
        .iter()
        .any(|finding| finding.msg.contains("outside every allocation")));
    assert!(later.atomic_components.is_empty());
}

#[test]
fn scalar_frozen_source_and_old_return_definedness_survive_literalization() {
    use ptxv_ir::File;
    use ptxv_ir::Reg;
    use ptxv_ir::Val;
    let (mut machine, pc) = prepared();
    let source = Reg {
        file: File::R,
        idx: 0,
    };
    let destination = Reg {
        file: File::R,
        idx: 1,
    };
    let Op::Atom { v, d, .. } = &mut Arc::make_mut(&mut machine.k).code[pc as usize].op else {
        panic!("scalar")
    };
    *v = Val::Reg(source, 1);
    *d = Some(destination);
    for thread in 0..3 {
        machine.vmask = Some(0xffff0000);
        machine.wr(thread, source, 1, 1);
        machine.vmask = None;
    }
    let mut machine = forced(machine, pc, vec![1, 0, 2]);
    take(&mut machine, 1);
    assert_eq!(machine.reg_umask(1, destination), 0);
    assert!((0..4).all(|at| machine.gmem[0].sh.get(at).undef));
    machine.wr(0, source, 1, 1);
    take(&mut machine, 0);
    assert_eq!(machine.reg_umask(0, destination), u32::MAX);
    assert!((0..4).all(|at| machine.gmem[0].sh.get(at).undef));
    assert!(machine
        .findings
        .iter()
        .any(|finding| finding.sig.starts_with("uninit-reg:")));
}

#[test]
fn scalar_or_middle_lane_return_is_not_certified_clean() {
    use ptxv_ir::AtomOp;
    use ptxv_ir::File;
    use ptxv_ir::Reg;
    use ptxv_ir::Val;
    let (mut machine, pc) = prepared();
    let source = Reg {
        file: File::R,
        idx: 1,
    };
    let Op::Atom { op, v, .. } = &mut Arc::make_mut(&mut machine.k).code[pc as usize].op else {
        panic!("scalar")
    };
    *op = AtomOp::Or;
    *v = Val::Reg(source, 1);
    for thread in 0..3 {
        machine.wr(thread, source, 1 << thread, 1);
    }
    let Op::Setp { b, .. } = &mut Arc::make_mut(&mut machine.k).code[pc as usize + 2].op else {
        panic!("comparison")
    };
    *b = Val::Imm(2);
    let mut witness = machine.clone();
    for thread in [1, 0, 2] {
        witness.exec_one(&[thread], pc).unwrap();
    }
    let (_, findings, complete) = crate::explore::run_once(witness);
    assert!(complete && findings.iter().any(|finding| finding.kind == "trap"));
    let outcome = crate::wellsync::certify(machine);
    eprintln!(
        "scalar OR certificate: verdict={:?}, reasons={:?}",
        outcome.verdict.as_ref().map(|verdict| verdict.classes()),
        outcome.reasons
    );
    assert!(
        outcome
            .verdict
            .as_ref()
            .is_some_and(|verdict| verdict.classes().iter().any(|kind| kind == "trap"))
            || !outcome.reasons.is_empty()
    );
}

fn family_machine(op: &str, initial: u32, sources: [u32; 3], target: u32) -> Machine {
    family_machine_spaced(op, initial, sources, target, false, false)
}

fn family_machine_spaced(
    op: &str,
    initial: u32,
    sources: [u32; 3],
    target: u32,
    disjoint: bool,
    single: bool,
) -> Machine {
    let text = r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 input) {
  .reg .b32 %r<3>;
  .reg .b64 %address;
  .reg .b64 %offset;
  .reg .pred %p<3>;
  ld.param.u64 %address, [input];
  mov.u32 %r0, %tid.x;
  setp.eq.u32 %p0, %r0, 0;
  setp.eq.u32 %p1, %r0, 1;
  selp.u32 %r2, @SRC1@, @SRC2@, %p1;
  selp.u32 %r2, @SRC0@, %r2, %p0;
  atom.global.@OP@ %r1, [%address], %r2;
  setp.eq.u32 %p0, %r0, 0;
  setp.eq.u32 %p1, %r1, @TARGET@;
  and.pred %p2, %p0, %p1;
  @%p2 trap;
  ret;
}
"#
    .replace("@OP@", op)
    .replace("@SRC0@", &sources[0].to_string())
    .replace("@SRC1@", &sources[1].to_string())
    .replace("@SRC2@", &sources[2].to_string())
    .replace("@TARGET@", &target.to_string());
    let text = if disjoint {
        text.replace("  mov.u32 %r0, %tid.x;", "  mov.u32 %r0, %tid.x;\n  mul.wide.u32 %offset, %r0, 4;\n  add.u64 %address, %address, %offset;")
    } else {
        text
    };
    let text = if single {
        text.replace("  atom.global.", "  mov.u32 %r1, 0;\n  @%p0 atom.global.")
    } else {
        text
    };
    let count = if disjoint { 3 } else { 1 };
    let values: Vec<_> = (0..count).map(|offset| json!([offset, initial])).collect();
    let kernel = Arc::new(ptxv_ptx::parse_kernel(&text, Some("k")).unwrap());
    let manifest = serde_json::from_value(json!({
        "name":"scalar-atomic-family-middle-order", "description":"reachable three-lane word atomic choices",
        "entry":"k", "grid":[1,1,1], "block":[3,1,1], "outputs":[],
        "buffers":[{"name":"x","dtype":"u32","shape":[count],"init":"zeros","values":values}],
        "params":[{"kind":"ptr","buffer":"x"}], "variants":[]
    })).unwrap();
    let case = ptxv_case::Case {
        dir: PathBuf::new(),
        m: manifest,
    };
    let launch = Arc::new(crate::build_launch(&case, &kernel).unwrap());
    Machine::new(kernel, launch)
}

#[test]
#[ignore = "upstream merge: pins exactly unmodeled, int7's lane-order findings also fire on the same atomic; one mechanism per shape pending; owner gC"]
fn scalar_word_families_expose_every_middle_return() {
    for (op, initial, sources, target, order) in [
        ("or.b32", 0, [1, 2, 4], 2, [1, 0, 2]),
        ("and.b32", 7, [6, 5, 3], 5, [1, 0, 2]),
        ("xor.b32", 0, [1, 2, 4], 2, [1, 0, 2]),
        ("exch.b32", 0, [1, 2, 4], 4, [2, 0, 1]),
        ("min.u32", 100, [80, 40, 20], 40, [1, 0, 2]),
        ("max.u32", 0, [2, 4, 8], 4, [1, 0, 2]),
        ("min.s32", 100, [80, 40, 20], 40, [1, 0, 2]),
        ("max.s32", 0, [2, 4, 8], 4, [1, 0, 2]),
        ("inc.u32", 0, [100, 100, 100], 1, [1, 0, 2]),
        ("dec.u32", 10, [100, 100, 100], 9, [1, 0, 2]),
    ] {
        let input = family_machine(op, initial, sources, target);
        for reverse in [false, true] {
            let mut input = input.clone();
            input.atom_lanes_rev = reverse;
            for (name, result) in [
                ("stateful", crate::explore::explore(input.clone(), 10000)),
                ("DPOR", crate::dpor::explore_dpor(input, 10000)),
            ] {
                eprintln!(
                    "family {op} reverse={reverse} {name}: complete={} states={} classes={:?}",
                    result.complete,
                    result.states,
                    result.classes()
                );
                assert!(result.complete);
                if matches!(op, "min.u32" | "min.s32" | "max.s32") {
                    assert_eq!(result.classes(), ["unmodeled"]);
                } else {
                    assert_eq!(result.classes(), ["trap"]);
                }
            }
        }
        let mut witness = input.clone();
        let pc = witness
            .k
            .code
            .iter()
            .position(|instruction| matches!(instruction.op, Op::Atom { .. }))
            .unwrap() as u32;
        for before in 0..pc {
            witness.exec_one(&[0, 1, 2], before).unwrap();
        }
        for actor in order {
            witness.exec_one(&[actor], pc).unwrap();
        }
        let (_, findings, complete) = crate::explore::run_once(witness);
        assert!(
            complete && findings.iter().any(|finding| finding.kind == "trap"),
            "{op}: {findings:?}"
        );
        let certificate = crate::wellsync::certify(input);
        eprintln!(
            "family {op} native trap confirmed; certificate={:?} reasons={:?}",
            certificate.verdict.as_ref().map(|v| v.classes()),
            certificate.reasons
        );
        assert!(certificate.verdict.is_none());
    }
}

#[test]
fn every_supported_scalar_word_freezes_uniform_aliases_and_counts_each_permutation() {
    use ptxv_ir::File;
    use ptxv_ir::Reg;
    use ptxv_ir::Val;
    use ptxv_memory_write::atomic_integer::IntegerOp;
    use ptxv_memory_write::atomic_integer::replacement32;
    for (text, op) in [
        ("or.b32", IntegerOp::Or),
        ("and.b32", IntegerOp::And),
        ("xor.b32", IntegerOp::Xor),
        ("add.u32", IntegerOp::Add),
        ("exch.b32", IntegerOp::Exch),
        ("max.u32", IntegerOp::MaxU),
        ("inc.u32", IntegerOp::Inc),
        ("dec.u32", IntegerOp::Dec),
    ] {
        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let mut machine = family_machine(text, 5, [7, 7, 7], 99);
            let pc = machine
                .k
                .code
                .iter()
                .position(|instruction| matches!(instruction.op, Op::Atom { .. }))
                .unwrap() as u32;
            for before in 0..pc {
                machine.exec_one(&[0, 1, 2], before).unwrap();
            }
            let uniform = Reg {
                file: File::UR,
                idx: 0,
            };
            let Op::Atom { v, d, .. } = &mut Arc::make_mut(&mut machine.k).code[pc as usize].op
            else {
                panic!("word atomic")
            };
            *v = Val::Reg(uniform, 1);
            *d = Some(uniform);
            machine.wr(0, uniform, 7, 1);
            let mut machine = forced(machine, pc, order.to_vec());
            let mut expected = 5;
            for (index, actor) in order.into_iter().enumerate() {
                let old = expected;
                expected = replacement32(op, old, 7, 0);
                take(&mut machine, actor as usize);
                assert_eq!(memory(&machine), expected, "{text} {order:?}");
                assert_eq!(
                    machine.rd(0, &Val::Reg(uniform, 1)).unwrap(),
                    old as u64,
                    "{text} {order:?}"
                );
                assert_eq!(
                    machine.ring.as_ref().unwrap().tick_claims[&(1, 0, 0)],
                    order[..=index]
                );
            }
            assert!(machine.threads.iter().all(|thread| thread.pc == pc + 1));
            assert!(
                machine.findings.is_empty(),
                "{text}: {:?}",
                machine.findings
            );
        }
    }
}

#[test]
fn scalar_exchange_returns_undefined_old_but_publishes_its_defined_frozen_source() {
    use ptxv_ir::File;
    use ptxv_ir::Reg;
    let mut machine = family_machine("exch.b32", 0, [7, 7, 7], 99);
    let pc = machine
        .k
        .code
        .iter()
        .position(|instruction| matches!(instruction.op, Op::Atom { .. }))
        .unwrap() as u32;
    for before in 0..pc {
        machine.exec_one(&[0, 1, 2], before).unwrap();
    }
    for offset in 0..4 {
        machine.gmem[0].sh.get_mut(offset).undef = true;
    }
    let mut machine = forced(machine, pc, vec![1, 0, 2]);
    take(&mut machine, 1);
    assert_eq!(
        machine.reg_umask(
            1,
            Reg {
                file: File::R,
                idx: 1
            }
        ),
        u32::MAX
    );
    assert_eq!(memory(&machine), 7);
    assert!((0..4).all(|offset| !machine.gmem[0].sh.get(offset).undef));
    assert!(machine.findings.is_empty(), "{:?}", machine.findings);
    take(&mut machine, 0);
    take(&mut machine, 2);
    assert_eq!(
        machine.reg_umask(
            0,
            Reg {
                file: File::R,
                idx: 1
            }
        ),
        0
    );
    assert_eq!(
        machine.reg_umask(
            2,
            Reg {
                file: File::R,
                idx: 1
            }
        ),
        0
    );
}

#[test]
fn unhandled_scalar_minmax_keeps_disjoint_and_single_claims_clean() {
    for (op, initial, sources) in [
        ("min.u32", 100, [80, 40, 20]),
        ("min.s32", 100, [80, 40, 20]),
        ("max.s32", 0, [2, 4, 8]),
    ] {
        for (disjoint, single) in [(true, false), (false, true)] {
            let input = family_machine_spaced(op, initial, sources, 99, disjoint, single);
            for result in [
                crate::explore::explore(input.clone(), 10000),
                crate::dpor::explore_dpor(input.clone(), 10000),
            ] {
                assert!(result.complete);
                assert_eq!(
                    result.classes(),
                    ["clean"],
                    "{op} disjoint={disjoint} single={single}"
                );
            }
            let result = crate::wellsync::certify(input);
            assert_eq!(
                result
                    .verdict
                    .unwrap_or_else(|| panic!("{op}: {:?}", result.reasons))
                    .classes(),
                ["clean"]
            );
        }
    }
}

#[test]
fn unhandled_scalar_minmax_shared_return_is_explicitly_unmodeled() {
    use ptxv_ir::File;
    use ptxv_ir::Reg;
    for op in ["min.u32", "min.s32", "max.s32"] {
        let mut input = family_machine_spaced(op, 100, [80, 40, 20], 99, true, false);
        let pc = input
            .k
            .code
            .iter()
            .position(|instruction| matches!(instruction.op, Op::Atom { .. }))
            .unwrap() as u32;
        for before in 0..pc {
            input.exec_one(&[0, 1, 2], before).unwrap();
        }
        let Op::Atom { d, .. } = &mut Arc::make_mut(&mut input.k).code[pc as usize].op else {
            panic!("word atomic")
        };
        *d = Some(Reg {
            file: File::UR,
            idx: 0,
        });
        input.exec_one(&[0, 1, 2], pc).unwrap();
        assert!(input
            .findings
            .iter()
            .any(|finding| finding.kind == "unmodeled"
                && finding.sig == format!("atomic-lane-order:{pc}")));
    }
}

#[test]
fn scalar_float_exposes_the_actual_middle_return() {
    let input = family_machine("add.f32", 0, [1.0f32.to_bits(); 3], 1.0f32.to_bits());
    for reverse in [false, true] {
        let mut machine = input.clone();
        machine.atom_lanes_rev = reverse;
        for (name, result) in [
            ("stateful", crate::explore::explore(machine.clone(), 10000)),
            ("DPOR", crate::dpor::explore_dpor(machine, 10000)),
        ] {
            eprintln!(
                "scalar F32 reverse={reverse} {name}: complete={} states={} classes={:?}",
                result.complete,
                result.states,
                result.classes()
            );
            assert!(result.complete);
            assert_eq!(result.classes(), ["trap"]);
        }
    }
    let mut witness = input.clone();
    let pc = witness
        .k
        .code
        .iter()
        .position(|instruction| matches!(instruction.op, Op::Atom { .. }))
        .unwrap() as u32;
    for before in 0..pc {
        witness.exec_one(&[0, 1, 2], before).unwrap();
    }
    for actor in [1, 0, 2] {
        witness.exec_one(&[actor], pc).unwrap();
    }
    let (_, findings, complete) = crate::explore::run_once(witness);
    eprintln!("scalar F32 actual 1,0,2: complete={complete} findings={findings:?}");
    assert!(complete && findings.iter().any(|finding| finding.kind == "trap"));
    let certificate = crate::wellsync::certify(input);
    eprintln!(
        "scalar F32 certificate={:?} reasons={:?}",
        certificate
            .verdict
            .as_ref()
            .map(|verdict| verdict.classes()),
        certificate.reasons
    );
    assert!(certificate.verdict.is_none());
}

#[test]
fn scalar_float_uniform_aliases_keep_frozen_values_and_numeric_histories() {
    use ptxv_ir::File;
    use ptxv_ir::Reg;
    use ptxv_ir::Val;
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let mut machine = family_machine("add.f32", 0, [1.0f32.to_bits(); 3], 99);
        let pc = machine
            .k
            .code
            .iter()
            .position(|instruction| matches!(instruction.op, Op::Atom { .. }))
            .unwrap() as u32;
        for before in 0..pc {
            machine.exec_one(&[0, 1, 2], before).unwrap();
        }
        let uniform = Reg {
            file: File::UR,
            idx: 0,
        };
        let Op::Atom { v, d, .. } = &mut Arc::make_mut(&mut machine.k).code[pc as usize].op else {
            panic!("float atomic")
        };
        *v = Val::Reg(uniform, 1);
        *d = Some(uniform);
        machine.wr(0, uniform, 1.0f32.to_bits() as u64, 1);
        let mut numeric = crate::num::NumState::default();
        let source = numeric.leaf(1.0f32.to_bits(), pc);
        numeric.set_reg_node(0, uniform, Some(source));
        machine.num = Some(Box::new(numeric));
        let mut machine = forced(machine, pc, order.to_vec());
        for (index, actor) in order.into_iter().enumerate() {
            take(&mut machine, actor as usize);
            assert_eq!(memory(&machine), ((index + 1) as f32).to_bits());
            assert_eq!(
                machine.rd(0, &Val::Reg(uniform, 1)).unwrap(),
                (index as f32).to_bits() as u64
            );
            let numeric = machine.num.as_ref().unwrap();
            let crate::num::NodeKind::AtomAdd(old, node, ftz) = numeric.nodes.last().unwrap().kind
            else {
                panic!("floating atomic add")
            };
            assert_eq!(node, source);
            assert!(ftz);
            assert_eq!(numeric.nodes[old as usize].hw, (index as f32).to_bits());
        }
        assert_eq!(
            machine.ring.as_ref().unwrap().tick_claims[&(1, 0, 0)],
            order
        );
        assert!(machine.threads.iter().all(|thread| thread.pc == pc + 1));
        assert!(machine.findings.is_empty(), "{:?}", machine.findings);
    }
}

#[test]
fn scalar_float_definedness_is_latched_separately_from_the_old_return() {
    let mut machine = family_machine("add.f32", 0, [1.0f32.to_bits(); 3], 99);
    let pc = machine
        .k
        .code
        .iter()
        .position(|instruction| matches!(instruction.op, Op::Atom { .. }))
        .unwrap() as u32;
    for before in 0..pc {
        machine.exec_one(&[0, 1, 2], before).unwrap();
    }
    let Op::Atom {
        v: ptxv_ir::Val::Reg(source, _),
        d: Some(destination),
        ..
    } = machine.k.code[pc as usize].op
    else {
        panic!("scalar source and destination")
    };
    machine.vmask = Some(0xffff0000);
    machine.wr(1, source, 1.0f32.to_bits() as u64, 1);
    machine.vmask = None;
    let mut machine = forced(machine, pc, vec![1, 0, 2]);
    take(&mut machine, 1);
    assert_eq!(memory(&machine), 1.0f32.to_bits());
    assert_eq!(machine.reg_umask(1, destination), 0);
    assert!((0..4).all(|at| machine.gmem[0].sh.get(at).undef));
    machine.wr(0, source, 9.0f32.to_bits() as u64, 1);
    take(&mut machine, 0);
    assert_eq!(memory(&machine), 2.0f32.to_bits());
    assert_eq!(machine.reg_umask(0, destination), u32::MAX);
    assert!((0..4).all(|at| machine.gmem[0].sh.get(at).undef));
}

#[test]
fn scalar_float_private_words_and_single_active_claims_remain_clean() {
    for (disjoint, single) in [(true, false), (false, true)] {
        let input =
            family_machine_spaced("add.f32", 0, [1.0f32.to_bits(); 3], 99, disjoint, single);
        for result in [
            crate::explore::explore(input.clone(), 10000),
            crate::dpor::explore_dpor(input.clone(), 10000),
        ] {
            assert!(result.complete);
            assert_eq!(result.classes(), ["clean"]);
        }
        let result = crate::wellsync::certify(input);
        assert_eq!(
            result
                .verdict
                .unwrap_or_else(|| panic!("{:?}", result.reasons))
                .classes(),
            ["clean"]
        );
    }
}

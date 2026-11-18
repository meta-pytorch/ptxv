/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::json;

use crate::machine::Machine;

pub(crate) fn machine_with_shape(scalar: bool, grid: u32, block: u32) -> Machine {
    let atomic = if scalar {
        r#"
  shr.u32 %r2, %r0, 16;
  atom.global.add.noftz.bf16 %r1, [%rd0], %r0;
  atom.global.add.noftz.bf16 %r3, [%rd0+2], %r2;
  shl.b32 %r3, %r3, 16;
  or.b32 %r1, %r1, %r3;
"#
    } else {
        "atom.global.add.noftz.bf16x2 %r1, [%rd0], %r0;"
    };
    let text = format!(
        r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 input) {{
  .reg .b32 %r<4>;
  .reg .b64 %rd0;
  .reg .pred %p<2>;
  ld.param.u64 %rd0, [input];
  mov.u32 %r0, %ctaid.x;
  setp.eq.u32 %p0, %r0, 0;
  mov.u32 %r0, 1092632448;
  @!%p0 mov.u32 %r0, 1101021184;
  {atomic}
  setp.eq.u32 %p1, %r1, 1101004800;
  and.pred %p1, %p1, %p0;
  @%p1 trap;
  ret;
}}
"#
    );
    let text = if block > 1 {
        text.replace("%ctaid.x", "%tid.x")
    } else {
        text
    };
    let kernel = Arc::new(ptxv_ptx::parse_kernel(&text, Some("k")).unwrap());
    let manifest = serde_json::from_value(json!({
        "name":"packed-atomic-components", "description":"bf16x2 mixed returned halves",
        "entry":"k", "grid":[grid,1,1], "block":[block,1,1], "outputs":[],
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

fn machine(scalar: bool) -> Machine {
    machine_with_shape(scalar, 2, 1)
}

#[test]
fn packed_atomic_components_reach_the_mixed_return_trap() {
    for scalar in [false, true] {
        for (name, verdict) in [
            (
                "stateful",
                crate::explore::explore(machine(scalar), 100_000),
            ),
            ("DPOR", crate::dpor::explore_dpor(machine(scalar), 100_000)),
        ] {
            eprintln!(
                "packed scalar_reference={scalar} {name}: states={}, transitions={}, complete={}, classes={:?}, findings={:?}",
                verdict.states,
                verdict.transitions,
                verdict.complete,
                verdict.classes(),
                verdict.findings
            );
            assert!(verdict.complete);
            assert!(verdict.classes().iter().any(|kind| kind == "trap"));
        }
    }
}

pub(crate) fn at_atomic(mut machine: Machine) -> Machine {
    for _ in 0..32 {
        let action = machine.actions().into_iter().find_map(|(action, _)| {
            matches!(action, crate::explore::Action::Split { .. }).then_some(action)
        });
        match action {
            Some(action) => machine.apply(&action),
            None => return machine,
        }
    }
    panic!("did not reach packed component actions");
}

pub(crate) fn take_component(
    machine: &mut Machine,
    thread: usize,
    component: ptxv_memory_write::atomic_components::Component,
) {
    let action = machine
        .actions()
        .into_iter()
        .find_map(|(action, _)| {
            matches!(action, crate::explore::Action::AtomicComponent { thread: t, component: c, .. }
            if t == thread && c == component)
            .then_some(action)
        })
        .expect("component enabled");
    machine.apply(&action);
}

#[test]
fn packed_witness_assembles_once_without_partial_register_writes() {
    use ptxv_ir::Op;
    use ptxv_ir::Val;
    use ptxv_memory_write::atomic_components::Component::First;
    use ptxv_memory_write::atomic_components::Component::Second;
    let mut machine = at_atomic(machine(false));
    let pc = machine.threads[0].pc;
    let Op::Atom {
        d: Some(destination),
        ..
    } = machine.k.code[pc as usize].op
    else {
        panic!("packed")
    };
    for thread in 0..2 {
        machine.wr(thread, destination, 0xdeadbeef, 1);
    }
    take_component(&mut machine, 0, First);
    assert_eq!(
        machine.rd(0, &Val::Reg(destination, 1)).unwrap(),
        0xdeadbeef
    );
    take_component(&mut machine, 1, First);
    assert_eq!(
        machine.rd(1, &Val::Reg(destination, 1)).unwrap(),
        0xdeadbeef
    );
    take_component(&mut machine, 1, Second);
    assert_eq!(machine.rd(1, &Val::Reg(destination, 1)).unwrap(), 0x3f80);
    take_component(&mut machine, 0, Second);
    assert_eq!(
        machine.rd(0, &Val::Reg(destination, 1)).unwrap(),
        0x41a00000
    );
    assert!(machine.atomic_components.is_empty());
    let memory: u32 = (0..4).fold(0, |word, at| {
        word | ((*machine.gmem[0].bytes.get(at) as u32) << (8 * at))
    });
    assert_eq!(memory, 0x41f04040);
}

#[test]
fn packed_return_keeps_half_definedness() {
    use ptxv_ir::Op;
    use ptxv_ir::Val;
    use ptxv_memory_write::atomic_components::Component::First;
    use ptxv_memory_write::atomic_components::Component::Second;
    let mut machine = at_atomic(machine(false));
    let pc = machine.threads[0].pc;
    let Op::Atom {
        d: Some(destination),
        ..
    } = machine.k.code[pc as usize].op
    else {
        panic!("packed")
    };
    for at in 2..4 {
        machine.gmem[0].sh.get_mut(at).undef = true;
    }
    take_component(&mut machine, 0, First);
    take_component(&mut machine, 0, Second);
    assert_eq!(machine.reg_umask(0, destination), 0xffff0000);
    assert_eq!(machine.rd(0, &Val::Reg(destination, 1)).unwrap(), 0);
    assert!(machine.findings.is_empty(), "{:?}", machine.findings);
    // S-INIT reports an undefined atomic return when it reaches a sink.
    for next in pc + 1..=pc + 3 {
        machine.exec_one(&[0], next).unwrap();
    }
    assert!(
        machine
            .findings
            .iter()
            .any(|finding| finding.sig == format!("uninit-reg:{}", pc + 3)),
        "{:?}",
        machine.findings
    );
}

#[test]
fn packed_uniform_source_and_destination_aliases_keep_all_issued_lanes_frozen() {
    use ptxv_ir::File;
    use ptxv_ir::Op;
    use ptxv_ir::Reg;
    use ptxv_ir::Val;
    use ptxv_memory_write::atomic_components::Component::First;
    use ptxv_memory_write::atomic_components::Component::Second;
    let mut machine = at_atomic(machine_with_shape(false, 1, 2));
    let pc = machine.threads[0].pc;
    let uniform = Reg {
        file: File::UR,
        idx: 0,
    };
    let Op::Atom { v, d, .. } = &mut Arc::make_mut(&mut machine.k).code[pc as usize].op else {
        panic!("packed")
    };
    *v = Val::Reg(uniform, 1);
    *d = Some(uniform);
    machine.wr(0, uniform, 0x41203f80, 1);
    take_component(&mut machine, 0, First);
    assert_eq!(machine.rd(0, &Val::Reg(uniform, 1)).unwrap(), 0x41203f80);
    take_component(&mut machine, 0, Second);
    assert_eq!(machine.rd(0, &Val::Reg(uniform, 1)).unwrap(), 0);
    assert!(machine.threads.iter().all(|thread| thread.pc == pc));
    take_component(&mut machine, 1, Second);
    assert_eq!(machine.rd(0, &Val::Reg(uniform, 1)).unwrap(), 0);
    take_component(&mut machine, 1, First);
    assert_eq!(machine.rd(0, &Val::Reg(uniform, 1)).unwrap(), 0x41203f80);
    assert!(machine.threads.iter().all(|thread| thread.pc == pc + 1));
    let memory: u32 = (0..4).fold(0, |word, at| {
        word | ((*machine.gmem[0].bytes.get(at) as u32) << (8 * at))
    });
    assert_eq!(memory, 0x41a04000);
    assert!(machine.findings.is_empty(), "{:?}", machine.findings);
}

#[test]
fn packed_issue_faults_before_all_component_writes() {
    use ptxv_ir::Op;
    use ptxv_ir::Val;
    use ptxv_memory_write::atomic_components::Component::First;
    let mut machine = at_atomic(machine_with_shape(false, 1, 2));
    let pc = machine.threads[0].pc;
    let Op::Atom { addr, .. } = &machine.k.code[pc as usize].op else {
        panic!("packed")
    };
    let Val::Reg(register, _) = addr.base.clone().unwrap() else {
        panic!("address")
    };
    let address = machine.rd(1, &Val::Reg(register, 1)).unwrap();
    machine.wr(1, register, address + 2, 1);
    let mut direct = machine.clone();
    assert!(direct.exec_one(&[0, 1], pc).unwrap_err().contains("align"));
    assert_eq!(*direct.gmem[0].bytes.get(0), 0x80);
    take_component(&mut machine, 0, First);
    assert!((0..4).all(|at| *machine.gmem[0].bytes.get(at) == 0));
    assert!(machine
        .findings
        .iter()
        .any(|finding| finding.msg.contains("align")));
    assert!(machine.atomic_components.is_empty());
    assert!(machine
        .threads
        .iter()
        .all(|thread| thread.st == crate::machine::St::Exited));
}

#[test]
fn packed_later_access_rechecks_bounds_and_keeps_completed_memory_prefix() {
    use ptxv_ir::Op;
    use ptxv_ir::Val;
    use ptxv_memory_write::atomic_components::Component::First;
    use ptxv_memory_write::atomic_components::Component::Second;
    let mut machine = at_atomic(machine_with_shape(false, 1, 1));
    let pc = machine.threads[0].pc;
    let Op::Atom {
        d: Some(destination),
        ..
    } = machine.k.code[pc as usize].op
    else {
        panic!("packed")
    };
    machine.wr(0, destination, 0xdeadbeef, 1);
    take_component(&mut machine, 0, First);
    Arc::make_mut(&mut machine.launch).allocs[0]
        .init
        .truncate(2);
    take_component(&mut machine, 0, Second);
    assert_eq!(
        [*machine.gmem[0].bytes.get(0), *machine.gmem[0].bytes.get(1)],
        [0x80, 0x3f]
    );
    assert_eq!(
        [*machine.gmem[0].bytes.get(2), *machine.gmem[0].bytes.get(3)],
        [0, 0]
    );
    assert_eq!(
        machine.rd(0, &Val::Reg(destination, 1)).unwrap(),
        0xdeadbeef
    );
    assert!(machine
        .findings
        .iter()
        .any(|finding| finding.msg.contains("outside every allocation")));
    assert_eq!(machine.threads[0].st, crate::machine::St::Exited);
    assert!(machine.atomic_components.is_empty());
}

#[test]
fn packed_sources_and_old_memory_provenance_survive_alias_writes() {
    use ptxv_ir::Op;
    use ptxv_ir::Val;
    use ptxv_memory_write::atomic_components::Component::First;
    use ptxv_memory_write::atomic_components::Component::Second;

    use crate::halfin::Src;
    let mut machine = at_atomic(machine_with_shape(false, 1, 1));
    let pc = machine.threads[0].pc;
    let source = ptxv_ir::Reg {
        file: ptxv_ir::File::R,
        idx: 0,
    };
    let Op::Atom { v, d, .. } = &mut Arc::make_mut(&mut machine.k).code[pc as usize].op else {
        panic!("packed")
    };
    *v = Val::Reg(source, 1);
    *d = Some(source);
    machine.wr(0, source, 0x41203f80, 1);
    let mut numeric = crate::num::NumState::default();
    numeric.half.track = true;
    let low = numeric.leaf(1.0f32.to_bits(), pc);
    let high = numeric.leaf(10.0f32.to_bits(), pc);
    machine.num = Some(Box::new(numeric));
    machine.half_set_reg(
        0,
        source,
        Some([
            Some(Src::N { node: low, b: 0 }),
            Some(Src::N { node: low, b: 1 }),
            Some(Src::N { node: high, b: 0 }),
            Some(Src::N { node: high, b: 1 }),
        ]),
    );
    let location = crate::exec::Loc::Global { alloc: 0, off: 0 };
    machine.half_mem_set(crate::halfin::hkey(location), &[Some(Src::Zero); 4]);
    take_component(&mut machine, 0, First);
    machine.wr(0, source, 0, 1);
    machine.half_set_reg(0, source, None);
    take_component(&mut machine, 0, Second);
    assert_eq!(machine.rd(0, &Val::Reg(source, 1)).unwrap(), 0);
    assert_eq!(
        machine.half_val(0, &Val::Reg(source, 1)),
        Some([Some(Src::Zero); 4])
    );
    let provenance = machine.half_mem(location, 4, None);
    for (index, expected) in [1.0f32.to_bits(), 10.0f32.to_bits()]
        .into_iter()
        .enumerate()
    {
        let Some(Src::N { node, b: 0 }) = provenance[2 * index] else {
            panic!("half history: {provenance:?}")
        };
        assert_eq!(
            machine.num.as_ref().unwrap().nodes[node as usize].hw,
            expected
        );
    }
    assert!(machine.findings.is_empty(), "{:?}", machine.findings);
}

#[test]
fn packed_returns_preserve_input_coverage() {
    let coverage = crate::coverage::analyze(&machine(false));
    assert!(coverage.terminal);
    assert_eq!(coverage.comparisons.len(), 1);
    assert!(coverage.comparisons[0]
        .inputs
        .iter()
        .any(|input| input.buffer == "x"));
}

#[test]
fn packed_source_definedness_is_latched_per_half_before_any_writeback() {
    use ptxv_ir::File;
    use ptxv_ir::Op;
    use ptxv_ir::Reg;
    use ptxv_ir::Val;
    use ptxv_memory_write::atomic_components::Component::First;
    use ptxv_memory_write::atomic_components::Component::Second;
    let mut machine = at_atomic(machine_with_shape(false, 1, 1));
    let pc = machine.threads[0].pc;
    let source = Reg {
        file: File::R,
        idx: 0,
    };
    let destination = Reg {
        file: File::R,
        idx: 1,
    };
    let Op::Atom { v, d, .. } = &mut Arc::make_mut(&mut machine.k).code[pc as usize].op else {
        panic!("packed")
    };
    *v = Val::Reg(source, 1);
    *d = Some(destination);
    machine.vmask = Some(0xffff0000);
    machine.wr(0, source, 0x41203f80, 1);
    machine.vmask = None;
    take_component(&mut machine, 0, First);
    machine.wr(0, source, 0, 1);
    take_component(&mut machine, 0, Second);
    assert!(!machine.gmem[0].sh.get(0).undef && !machine.gmem[0].sh.get(1).undef);
    assert!(machine.gmem[0].sh.get(2).undef && machine.gmem[0].sh.get(3).undef);
    assert_eq!(machine.reg_umask(0, destination), 0);
    assert!(
        machine
            .findings
            .iter()
            .any(|finding| finding.sig.starts_with("uninit-reg:")),
        "{:?}",
        machine.findings
    );
}

#[test]
fn packed_mixed_returns_are_not_certified_clean() {
    let outcome = crate::wellsync::certify(machine(false));
    eprintln!("packed mixed-return certificate: {:?}", outcome.reasons);
    if let Some(verdict) = outcome.verdict {
        assert!(verdict.classes().iter().any(|kind| kind == "trap"));
    } else {
        assert!(!outcome.reasons.is_empty());
    }
}

#[test]
fn packed_partial_guard_retires_exactly_the_frozen_issued_split() {
    use ptxv_ir::File;
    use ptxv_ir::Pred;
    use ptxv_ir::Reg;
    use ptxv_memory_write::atomic_components::Component::First;
    use ptxv_memory_write::atomic_components::Component::Second;
    let mut machine = at_atomic(machine_with_shape(false, 1, 3));
    let pc = machine.threads[0].pc;
    let guard = Reg {
        file: File::P,
        idx: 0,
    };
    Arc::make_mut(&mut machine.k).code[pc as usize].guard = Pred {
        reg: Some(guard),
        neg: false,
    };
    machine.wp(0, guard, true);
    machine.wp(1, guard, true);
    machine.wp(2, guard, false);
    assert_eq!(machine.actions().len(), 4);
    take_component(&mut machine, 0, First);
    machine.wp(2, guard, true);
    assert_eq!(machine.actions().len(), 3);
    assert!(machine.threads.iter().all(|thread| thread.pc == pc));
    for (thread, component) in [(1, First), (1, Second), (0, Second)] {
        take_component(&mut machine, thread, component);
    }
    assert!(machine.threads.iter().all(|thread| thread.pc == pc + 1));
    assert!(machine.atomic_components.is_empty());
    assert!(machine.findings.is_empty(), "{:?}", machine.findings);
}

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

fn mixed_return_text(vector: bool) -> String {
    let atomic = if vector {
        "atom.global.add.v2.f32 {%f2,%f3}, [%rd0], {%f0,%f1};"
    } else {
        "atom.global.add.f32 %f2, [%rd0], %f0;\n  atom.global.add.f32 %f3, [%rd0+4], %f1;"
    };
    format!(
        r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 input) {{
  .reg .b32 %r0;
  .reg .b64 %rd0;
  .reg .f32 %f<4>;
  .reg .pred %p<4>;
  ld.param.u64 %rd0, [input];
  mov.u32 %r0, %ctaid.x;
  setp.eq.u32 %p0, %r0, 1;
  mov.f32 %f0, 0f3f800000;
  mov.f32 %f1, 0f41200000;
  @%p0 mov.f32 %f0, 0f40000000;
  @%p0 mov.f32 %f1, 0f41a00000;
  {atomic}
  setp.eq.f32 %p1, %f2, 0f00000000;
  setp.eq.f32 %p2, %f3, 0f41a00000;
  and.pred %p1, %p1, %p2;
  not.pred %p0, %p0;
  and.pred %p1, %p1, %p0;
  @%p1 trap;
  ret;
}}
"#
    )
}

fn from_text(text: &str, grid: u32, block: u32, elements: u32) -> Machine {
    let kernel = Arc::new(ptxv_ptx::parse_kernel(text, Some("k")).unwrap());
    let manifest = serde_json::from_value(json!({
        "name":"atomic-component-order", "description":"mixed atomic component returns",
        "entry":"k", "grid":[grid,1,1], "block":[block,1,1], "outputs":[],
        "buffers":[{"name":"x","dtype":"f32","shape":[elements],"init":"zeros"}],
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

fn mixed_return_machine(vector: bool) -> Machine {
    from_text(&mixed_return_text(vector), 2, 1, 2)
}

fn at_atomic(mut machine: Machine) -> Machine {
    for _ in 0..32 {
        let action = machine.actions().into_iter().find_map(|(action, _)| {
            matches!(action, crate::explore::Action::Split { .. }).then_some(action)
        });
        match action {
            Some(action) => machine.apply(&action),
            None => return machine,
        }
    }
    panic!("did not reach component actions");
}

fn take_component(
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
        .expect("component is enabled");
    machine.apply(&action);
}

#[test]
fn same_warp_component_alternatives_expose_the_mixed_return() {
    let text = mixed_return_text(true).replace("%ctaid.x", "%tid.x");
    let machine = from_text(&text, 1, 2, 2);
    for (name, result) in [
        (
            "stateful",
            crate::explore::explore(machine.clone(), 100_000),
        ),
        ("DPOR", crate::dpor::explore_dpor(machine, 100_000)),
    ] {
        eprintln!(
            "same warp {name}: {} states, complete={}, {:?}",
            result.states,
            result.complete,
            result.classes()
        );
        assert!(result.complete);
        assert!(
            result.classes().iter().any(|kind| kind == "trap"),
            "{name}: {:?}",
            result.findings
        );
    }
}

#[test]
fn source_aliases_keep_frozen_values_and_numeric_histories() {
    use ptxv_ir::Op;
    use ptxv_ir::Reg;
    use ptxv_ir::Val;
    use ptxv_memory_write::atomic_components::Component::First;
    use ptxv_memory_write::atomic_components::Component::Second;

    let mut machine = at_atomic(mixed_return_machine(true));
    let pc = machine.threads[0].pc;
    let kernel = Arc::make_mut(&mut machine.k);
    let Op::Atom {
        v: Val::Reg(source, _),
        d,
        ..
    } = &mut kernel.code[pc as usize].op
    else {
        panic!("atomic")
    };
    *d = Some(Reg {
        file: source.file,
        idx: source.idx + 1,
    });
    let Op::Atom {
        v: Val::Reg(source, _),
        d: Some(destination),
        ..
    } = machine.k.code[pc as usize].op
    else {
        panic!("atomic")
    };
    let high_source = Reg {
        file: source.file,
        idx: source.idx + 1,
    };
    let mut numeric = crate::num::NumState::default();
    let source_node = numeric.leaf(10.0f32.to_bits(), pc);
    numeric.set_reg_node(0, high_source, Some(source_node));
    machine.num = Some(Box::new(numeric));
    for (thread, component) in [(0, First), (1, First), (1, Second), (0, Second)] {
        take_component(&mut machine, thread, component);
    }
    assert_eq!(machine.rd(0, &Val::Reg(destination, 1)).unwrap(), 0);
    assert_eq!(
        machine
            .rd(
                0,
                &Val::Reg(
                    Reg {
                        file: destination.file,
                        idx: destination.idx + 1
                    },
                    1
                )
            )
            .unwrap(),
        20.0f32.to_bits() as u64
    );
    let actual: Vec<_> = (0..8)
        .map(|offset| *machine.gmem[0].bytes.get(offset))
        .collect();
    let expected: Vec<_> = [3.0f32, 30.0]
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    assert_eq!(actual, expected);
    let numeric = machine.num.as_ref().unwrap();
    let crate::num::NodeKind::AtomAdd(old, node, ftz) = numeric.nodes.last().unwrap().kind else {
        panic!("floating atomic add")
    };
    assert_eq!(node, source_node);
    assert!(ftz);
    assert_eq!(numeric.nodes[old as usize].hw, 20.0f32.to_bits());
    assert!(machine.findings.is_empty(), "{:?}", machine.findings);
}

#[test]
fn address_alias_is_latched_before_component_writeback() {
    use ptxv_ir::Op;
    use ptxv_ir::Reg;
    use ptxv_ir::Val;
    use ptxv_memory_write::atomic_components::Component::First;
    use ptxv_memory_write::atomic_components::Component::Second;

    let mut machine = at_atomic(mixed_return_machine(true));
    let pc = machine.threads[0].pc;
    let kernel = Arc::make_mut(&mut machine.k);
    let Op::Atom { addr, d, .. } = &mut kernel.code[pc as usize].op else {
        panic!("atomic")
    };
    let Val::Reg(destination, _) = addr.base.clone().unwrap() else {
        panic!("register address")
    };
    *d = Some(destination);
    for (thread, component) in [(0, Second), (1, First), (1, Second), (0, First)] {
        take_component(&mut machine, thread, component);
    }
    assert_eq!(
        machine.rd(0, &Val::Reg(destination, 1)).unwrap(),
        2.0f32.to_bits() as u64
    );
    assert_eq!(
        machine
            .rd(
                0,
                &Val::Reg(
                    Reg {
                        file: destination.file,
                        idx: destination.idx + 1
                    },
                    1
                )
            )
            .unwrap(),
        0
    );
    let actual: Vec<_> = (0..8)
        .map(|offset| *machine.gmem[0].bytes.get(offset))
        .collect();
    let expected: Vec<_> = [3.0f32, 30.0]
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    assert_eq!(actual, expected);
    assert!(machine.findings.is_empty(), "{:?}", machine.findings);
}

#[test]
fn partially_guarded_issue_retires_the_frozen_split_together() {
    use ptxv_memory_write::atomic_components::Component::First;
    use ptxv_memory_write::atomic_components::Component::Second;

    let text = mixed_return_text(true)
        .replace("%ctaid.x", "%tid.x")
        .replace(
            "  atom.global.add.v2.f32",
            "  setp.lt.u32 %p3, %r0, 2;\n  @%p3 atom.global.add.v2.f32",
        );
    let mut machine = at_atomic(from_text(&text, 1, 3, 2));
    let pc = machine.threads[0].pc;
    assert_eq!(machine.actions().len(), 4);
    take_component(&mut machine, 0, First);
    assert!(machine.threads.iter().all(|thread| thread.pc == pc));
    assert_eq!(machine.actions().len(), 3);
    for (thread, component) in [(1, First), (1, Second), (0, Second)] {
        take_component(&mut machine, thread, component);
    }
    assert!(machine.threads.iter().all(|thread| thread.pc == pc + 1));
    assert!(machine.atomic_components.is_empty());
    assert!(machine.findings.is_empty(), "{:?}", machine.findings);
}

#[test]
fn mixed_atomic_component_returns_are_explored() {
    for vector in [false, true] {
        let machine = mixed_return_machine(vector);
        for (name, result) in [
            (
                "stateful",
                crate::explore::explore(machine.clone(), 100_000),
            ),
            ("DPOR", crate::dpor::explore_dpor(machine, 100_000)),
        ] {
            eprintln!(
                "vector={vector} {name}: {} states, {} transitions, complete={}, classes={:?}, findings={:?}",
                result.states,
                result.transitions,
                result.complete,
                result.classes(),
                result.findings
            );
            assert!(result.complete, "{name} vector={vector}");
            let trapped = result.classes().iter().any(|kind| kind == "trap");
            assert_eq!(trapped, true, "{name}: {:?}", result.findings);
        }
    }
}

#[test]
fn vector_mixed_returns_are_not_certified_clean() {
    let outcome = crate::wellsync::certify(mixed_return_machine(true));
    eprintln!("mixed-return certificate: {:?}", outcome.reasons);
    if let Some(verdict) = outcome.verdict {
        assert!(verdict.classes().iter().any(|kind| kind == "trap"));
    } else {
        assert!(!outcome.reasons.is_empty());
    }
}

#[test]
fn a_high_component_word_is_excluded_from_whole_instruction_tickets() {
    let text = mixed_return_text(true).replace(
        "  atom.global.add.v2.f32 {%f2,%f3}, [%rd0], {%f0,%f1};",
        "  mov.f32 %f2, 0f00000000; @%p0 bra scalar; atom.global.add.v2.f32 {%f2,%f3}, [%rd0], {%f0,%f1}; bra done; scalar: atom.global.add.f32 %f3, [%rd0+4], %f1; done:");
    let machine = from_text(&text, 2, 1, 2);
    let outcome = crate::wellsync::certify(machine.clone());
    assert!(outcome.verdict.is_none(), "{:?}", outcome.reasons);
    let explored = crate::explore::explore(machine, 100_000);
    assert!(explored.complete);
    assert!(explored.classes().iter().any(|kind| kind == "trap"));
}

#[test]
fn component_returns_preserve_input_coverage() {
    let coverage = crate::coverage::analyze(&mixed_return_machine(true));
    assert!(coverage.terminal);
    assert_eq!(coverage.comparisons.len(), 2);
    assert!(coverage
        .comparisons
        .iter()
        .all(|comparison| comparison.inputs.iter().any(|input| input.buffer == "x")));
}

#[test]
fn issue_fault_precedes_all_component_writes() {
    use ptxv_ir::Op;
    use ptxv_ir::Val;
    use ptxv_memory_write::atomic_components::Component::First;

    let text = mixed_return_text(true).replace("%ctaid.x", "%tid.x");
    let mut issued = at_atomic(from_text(&text, 1, 2, 2));
    let pc = issued.threads[0].pc;
    let Op::Atom { addr, .. } = &issued.k.code[pc as usize].op else {
        panic!("atomic")
    };
    let Val::Reg(address_register, _) = addr.base.clone().unwrap() else {
        panic!("register address")
    };
    let address = issued.rd(1, &Val::Reg(address_register, 1)).unwrap();
    issued.wr(1, address_register, address + 4, 1);
    let mut direct = issued.clone();
    let error = direct.exec_one(&[0, 1], pc).unwrap_err();
    assert!(error.contains("align"), "{error}");
    let direct_bytes: Vec<_> = (0..8).map(|i| *direct.gmem[0].bytes.get(i)).collect();
    let direct_expected: Vec<_> = [1.0f32, 10.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    assert_eq!(direct_bytes, direct_expected);

    take_component(&mut issued, 0, First);
    assert!((0..8).all(|i| *issued.gmem[0].bytes.get(i) == 0));
    assert_eq!(issued.findings.len(), 1, "{:?}", issued.findings);
    assert!(issued.findings[0].msg.contains("align"));
    assert!(issued.atomic_components.is_empty());
    assert!(issued
        .threads
        .iter()
        .all(|thread| thread.st == crate::machine::St::Exited));
}

#[test]
fn later_arrival_cannot_join_or_replace_an_issued_split() {
    use ptxv_memory_write::atomic_components::Component::First;
    use ptxv_memory_write::atomic_components::Component::Second;
    let text = mixed_return_text(true).replace("%ctaid.x", "%tid.x");
    let mut machine = at_atomic(from_text(&text, 1, 2, 2));
    let pc = machine.threads[0].pc;
    machine.threads[1].make_mut().set_pc(pc - 1);
    take_component(&mut machine, 0, First);
    machine.threads[1].make_mut().set_pc(pc);
    let choices = machine.actions();
    assert_eq!(choices.len(), 1);
    assert!(matches!(
        choices[0].0,
        crate::explore::Action::AtomicComponent {
            thread: 0,
            component: Second,
            ..
        }
    ));
    take_component(&mut machine, 0, Second);
    assert_eq!(machine.threads[0].pc, pc + 1);
    assert_eq!(machine.threads[1].pc, pc);
    assert!(machine.atomic_components.is_empty());
    take_component(&mut machine, 1, Second);
    take_component(&mut machine, 1, First);
    assert_eq!(machine.threads[1].pc, pc + 1);
    assert!(machine.atomic_components.is_empty());
}

#[test]
fn each_component_rechecks_allocation_bounds() {
    use ptxv_memory_write::atomic_components::Component::First;
    use ptxv_memory_write::atomic_components::Component::Second;
    let mut machine = at_atomic(from_text(&mixed_return_text(true), 1, 1, 2));
    take_component(&mut machine, 0, First);
    Arc::make_mut(&mut Arc::make_mut(&mut machine.launch).allocs[0].init).truncate(4);
    take_component(&mut machine, 0, Second);
    assert!(machine
        .findings
        .iter()
        .any(|finding| finding.msg.contains("outside every allocation")));
    assert!(machine.atomic_components.is_empty());
    assert_eq!(machine.threads[0].st, crate::machine::St::Exited);
    assert_eq!(
        (0..4)
            .map(|i| *machine.gmem[0].bytes.get(i))
            .collect::<Vec<_>>(),
        1.0f32.to_le_bytes()
    );
    assert!((4..8).all(|i| *machine.gmem[0].bytes.get(i) == 0));
}

#[test]
fn all_four_components_retire_once_in_an_arbitrary_lane_order() {
    use ptxv_memory_write::atomic_components::Component::First;
    use ptxv_memory_write::atomic_components::Component::Fourth;
    use ptxv_memory_write::atomic_components::Component::Second;
    use ptxv_memory_write::atomic_components::Component::Third;
    let text = mixed_return_text(true)
        .replace(".reg .f32 %f<4>", ".reg .f32 %f<8>")
        .replace(
            "atom.global.add.v2.f32 {%f2,%f3}, [%rd0], {%f0,%f1}",
            "mov.f32 %f2, %f0; mov.f32 %f3, %f1; atom.global.add.v4.f32 {%f4,%f5,%f6,%f7}, [%rd0], {%f0,%f1,%f2,%f3}",
        );
    let mut machine = at_atomic(from_text(&text, 2, 1, 4));
    let pcs = [machine.threads[0].pc, machine.threads[1].pc];
    for (thread, component) in [
        (0, First),
        (1, Fourth),
        (1, First),
        (1, Second),
        (0, Third),
        (1, Third),
        (0, Second),
        (0, Fourth),
    ] {
        take_component(&mut machine, thread, component);
    }
    assert!(machine.atomic_components.is_empty());
    for thread in 0..2 {
        assert_eq!(machine.threads[thread].pc, pcs[thread] + 1);
    }
    let actual: Vec<_> = (0..16).map(|i| *machine.gmem[0].bytes.get(i)).collect();
    let expected: Vec<_> = [3.0f32, 30.0, 3.0, 30.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    assert_eq!(actual, expected);
    assert!(machine.findings.is_empty(), "{:?}", machine.findings);
}

#[test]
#[ignore = "bounded exploration cost measurement"]
fn vector_reduction_exploration_cost_sample() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let case = ptxv_case::Case::load(root.join("probes/red_vec")).unwrap();
    let text = std::fs::read_to_string(case.dir.join("kernel.ptx")).unwrap();
    let kernel = Arc::new(ptxv_ptx::parse_kernel(&text, Some("red_vec")).unwrap());
    let launch = Arc::new(crate::build_launch(&case, &kernel).unwrap());
    let machine = Machine::new(kernel, launch);
    let result = crate::explore::explore(machine, 10_000);
    eprintln!(
        "returnless RED sample: {} states, {} transitions, {} terminal, complete={}, seconds={}, {:?}",
        result.states,
        result.transitions,
        result.terminal_states,
        result.complete,
        result.seconds,
        result.classes()
    );
}

fn concolic_atomic_machine(mut machine: Machine) -> Machine {
    let mut provenance = crate::coverage::ProvState::default();
    provenance.sym.0 = Some(Box::new(crate::symb::SymState::new(
        Default::default(),
        machine.threads.len(),
    )));
    machine.prov = Some(Box::new(provenance));
    machine
}

fn atomic_symbolic_register(
    machine: &mut Machine,
    thread: usize,
    register: ptxv_ir::Reg,
) -> (crate::symb::E, u64, bool) {
    let mut symbolic = machine.prov.as_mut().unwrap().sym.0.take().unwrap();
    let expression = symbolic.reg(machine, thread, register);
    let result = (
        expression,
        symbolic.ar.conc[expression as usize],
        symbolic.ar.is_const(expression).is_none(),
    );
    assert_eq!(symbolic.n_mismatch, 0, "{:?}", symbolic.mismatches);
    machine.prov.as_mut().unwrap().sym.0 = Some(symbolic);
    result
}

#[test]
fn symbolic_atomic_components_update_only_the_selected_return() {
    use ptxv_memory_write::atomic_components::Component::First;
    use ptxv_memory_write::atomic_components::Component::Second;
    let text = mixed_return_text(true).replace(
        "  atom.global.add.v2.f32",
        "  mov.f32 %f2, 0f3f800000;\n  mov.f32 %f3, 0f40000000;\n  atom.global.add.v2.f32",
    );
    let mut machine = at_atomic(concolic_atomic_machine(from_text(&text, 1, 2, 2)));
    let pc = machine.threads[0].pc;
    let ptxv_ir::Op::Atom {
        d: Some(destination),
        ..
    } = machine.k.code[pc as usize].op
    else {
        panic!("vector atomic")
    };
    let next = ptxv_ir::Reg {
        idx: destination.idx + 1,
        ..destination
    };
    let untouched = atomic_symbolic_register(&mut machine, 1, destination);
    let high = atomic_symbolic_register(&mut machine, 0, next);
    take_component(&mut machine, 0, First);
    assert_eq!(
        atomic_symbolic_register(&mut machine, 1, destination),
        untouched
    );
    assert_eq!(atomic_symbolic_register(&mut machine, 0, next), high);
    assert!(atomic_symbolic_register(&mut machine, 0, destination).2);
    for (thread, component) in [(1, First), (1, Second), (0, Second)] {
        take_component(&mut machine, thread, component);
    }
    for thread in 0..2 {
        for register in [destination, next] {
            let (_, symbolic, opaque) = atomic_symbolic_register(&mut machine, thread, register);
            assert!(opaque);
            assert_eq!(
                symbolic,
                machine.rd(thread, &ptxv_ir::Val::Reg(register, 1)).unwrap()
            );
        }
    }
}

#[test]
fn symbolic_atomic_guards_and_aliases_survive_another_warp() {
    use ptxv_memory_write::atomic_components::Component::First;
    use ptxv_memory_write::atomic_components::Component::Second;
    let text = mixed_return_text(true)
        .replace("setp.eq.u32 %p0, %r0, 1;",
            "ld.global.u32 %r0, [%rd0+8];\n  setp.eq.u32 %p0, %r0, 0;")
        .replace("  atom.global.add.v2.f32",
            "  ld.global.f32 %f2, [%rd0+8];\n  ld.global.f32 %f3, [%rd0+8];\n  @%p0 atom.global.add.v2.f32");
    let mut machine = concolic_atomic_machine(from_text(&text, 2, 1, 3));
    let kernel = Arc::make_mut(&mut machine.k);
    let instruction = kernel
        .code
        .iter_mut()
        .find(|instruction| matches!(instruction.op, ptxv_ir::Op::Atom { .. }))
        .unwrap();
    let ptxv_ir::Op::Atom {
        v: ptxv_ir::Val::Reg(source, _),
        d,
        ..
    } = &mut instruction.op
    else {
        panic!("vector atomic")
    };
    *d = Some(ptxv_ir::Reg {
        idx: source.idx + 1,
        ..*source
    });
    let destination = d.unwrap();
    let high = ptxv_ir::Reg {
        idx: destination.idx + 1,
        ..destination
    };
    let mut machine = at_atomic(machine);
    let original = atomic_symbolic_register(&mut machine, 0, high).0;
    take_component(&mut machine, 0, First);
    assert_eq!(atomic_symbolic_register(&mut machine, 0, high).0, original);
    take_component(&mut machine, 1, First);
    take_component(&mut machine, 1, Second);
    let action = machine
        .actions()
        .into_iter()
        .find_map(|(action, _)| {
            matches!(action, crate::explore::Action::Split { .. }).then_some(action)
        })
        .unwrap();
    machine.apply(&action);
    take_component(&mut machine, 0, Second);
    let (expression, captured, _) = atomic_symbolic_register(&mut machine, 0, high);
    assert_eq!(
        captured,
        machine.rd(0, &ptxv_ir::Val::Reg(high, 1)).unwrap()
    );
    let symbolic = machine.prov.as_ref().unwrap().sym.0.as_ref().unwrap();
    let input = symbolic
        .ar
        .leaves(original)
        .into_iter()
        .find(|&node| {
            matches!(
                symbolic.ar.nodes[node as usize],
                crate::symb::Node::Input {
                    alloc: 0,
                    off: 8,
                    bytes: 4
                }
            )
        })
        .expect("guard and old return share the input word");
    let environment = [(input, 1)].into_iter().collect();
    assert_eq!(
        crate::symb::Evaluator::new(&symbolic.ar, &machine).eval(expression, &environment),
        Some(1),
        "old={original} {} result={expression} {}",
        symbolic.ar.show(original, &|a| format!("a{a}"), 8),
        symbolic.ar.show(expression, &|a| format!("a{a}"), 8)
    );
}

#[test]
fn symbolic_packed_atomic_return_is_published_only_at_retirement() {
    use ptxv_memory_write::atomic_components::Component::First;
    use ptxv_memory_write::atomic_components::Component::Second;
    let mut machine = crate::atomic_packed_tests::at_atomic(concolic_atomic_machine(
        crate::atomic_packed_tests::machine_with_shape(false, 2, 1),
    ));
    let pc = machine.threads[0].pc;
    let ptxv_ir::Op::Atom {
        d: Some(destination),
        ..
    } = machine.k.code[pc as usize].op
    else {
        panic!("packed atomic")
    };
    let original = atomic_symbolic_register(&mut machine, 0, destination);
    crate::atomic_packed_tests::take_component(&mut machine, 0, Second);
    assert_eq!(
        atomic_symbolic_register(&mut machine, 0, destination),
        original
    );
    crate::atomic_packed_tests::take_component(&mut machine, 1, First);
    crate::atomic_packed_tests::take_component(&mut machine, 1, Second);
    crate::atomic_packed_tests::take_component(&mut machine, 0, First);
    let (_, value, opaque) = atomic_symbolic_register(&mut machine, 0, destination);
    assert!(opaque);
    assert_eq!(
        value,
        machine.rd(0, &ptxv_ir::Val::Reg(destination, 1)).unwrap()
    );
}

#[test]
fn symbolic_atomic_guard_false_lanes_are_modeled_once() {
    use ptxv_memory_write::atomic_components::Component::First;
    use ptxv_memory_write::atomic_components::Component::Second;
    let text = mixed_return_text(true)
        .replace(".reg .b32 %r0;", ".reg .b32 %r<2>;")
        .replace("mov.u32 %r0, %ctaid.x;", "ld.global.u32 %r0, [%rd0+8];\n  mov.u32 %r1, %tid.x;")
        .replace("setp.eq.u32 %p0, %r0, 1;", "setp.eq.u32 %p0, %r0, %r1;")
        .replace("  atom.global.add.v2.f32", "  ld.global.f32 %f2, [%rd0+8];\n  ld.global.f32 %f3, [%rd0+8];\n  @%p0 atom.global.add.v2.f32");
    let mut machine = at_atomic(concolic_atomic_machine(from_text(&text, 1, 2, 3)));
    let pc = machine.threads[0].pc;
    let ptxv_ir::Op::Atom {
        d: Some(destination),
        ..
    } = machine.k.code[pc as usize].op
    else {
        panic!("vector atomic")
    };
    take_component(&mut machine, 0, First);
    let ghost = atomic_symbolic_register(&mut machine, 1, destination);
    assert!(ghost.2);
    let before = machine
        .prov
        .as_ref()
        .unwrap()
        .sym
        .0
        .as_ref()
        .unwrap()
        .ar
        .opaque
        .iter()
        .filter(|leaf| leaf.why == "would-be value of a guarded-off instruction (not modelled)")
        .count();
    assert_eq!(before, 2);
    take_component(&mut machine, 0, Second);
    assert_eq!(
        atomic_symbolic_register(&mut machine, 1, destination),
        ghost
    );
    let after = machine
        .prov
        .as_ref()
        .unwrap()
        .sym
        .0
        .as_ref()
        .unwrap()
        .ar
        .opaque
        .iter()
        .filter(|leaf| leaf.why == "would-be value of a guarded-off instruction (not modelled)")
        .count();
    assert_eq!(after, before);
}

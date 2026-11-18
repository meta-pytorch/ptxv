/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_ir::File;
use ptxv_ir::Op;
use ptxv_ir::Reg;
use ptxv_ir::Scope;
use ptxv_memory_write::machine_control::Flow;
use ptxv_memory_write::machine_warp::MaskOp;
use serde_json::json;

use crate::machine::AsyncKind;
use crate::machine::AsyncOp;
use crate::machine::Cover;
use crate::machine::Footprint;
use crate::machine::Machine;
use crate::machine::MbRel;
use crate::machine::Resource;
use crate::machine::ResourceSpace;
use crate::machine::SFoot;
use crate::machine::St;
use crate::machine::TmapDesc;
use crate::machine::VC;
use crate::machine::vc_get;
use crate::machine::vc_join;
use crate::machine::vc_single;

fn empty_machine(text: &str) -> Machine {
    empty_machine_with_threads(text, 32)
}

fn empty_machine_with_threads(text: &str, threads: u32) -> Machine {
    let kernel = Arc::new(ptxv_ptx::parse_kernel(text, Some("k")).unwrap());
    let manifest = serde_json::from_value(json!({
        "name":"allocation-reduction", "description":"guarded allocation progress",
        "entry":"k", "grid":[1,1,1], "block":[threads,1,1], "outputs":[],
        "buffers":[], "params":[], "variants":[]
    }))
    .unwrap();
    let case = ptxv_case::Case {
        dir: PathBuf::new(),
        m: manifest,
    };
    let launch = Arc::new(crate::build_launch(&case, &kernel).unwrap());
    Machine::new(kernel, launch)
}

fn register_machine() -> Machine {
    let writes = (0..70)
        .map(|index| format!("mov.b64 %r{index}, 0;"))
        .collect::<Vec<_>>()
        .join("\n");
    empty_machine(&format!(
        ".version 8.8\n.target sm_103a\n.address_size 64\n\
         .visible .entry k() {{ .reg .b64 %r<70>; {writes} ret; }}"
    ))
}

#[test]
fn machine_register_writes_preserve_partial_definedness_and_snapshot_frames() {
    let mut machine = register_machine();
    let original = machine.clone();
    let reg = |idx| Reg { file: File::V, idx };
    machine.vmask = Some(0x1234);
    machine.wr(0, reg(65), 0x1122_3344_5566_7788, 1);
    machine.vmask = Some(0x5678);
    machine.wr(0, reg(63), 0xaabb_ccdd, 1);
    assert_eq!(machine.threads[0].vund, [(63, 0x5678), (65, 0x1234)]);
    machine.make_undefined(0, reg(65));
    assert_eq!(machine.reg_umask(0, reg(65)), u32::MAX);
    assert_eq!(machine.threads[0].regs[65], 0x1122_3344_5566_7788);
    machine.stale = Some(None);
    machine.wr(0, reg(65), 42, 1);
    assert_eq!(machine.threads[0].vund, [(63, 0x5678)]);
    machine.stale = None;
    machine.vmask = None;
    machine.wr(0, reg(63), 99, 1);
    assert!(machine.threads[0].vund.is_empty());
    assert_eq!(machine.reg_umask(0, reg(63)), 0);
    assert_eq!(machine.threads[1].regs, original.threads[1].regs);
    assert_eq!(original.threads[0].regs[63], 0);
    assert_eq!(original.threads[0].regs[65], 0);
    assert_eq!(machine.threads[0].pc, original.threads[0].pc);
    assert_eq!(machine.threads[0].st, original.threads[0].st);
    assert_eq!(machine.threads[0].vc, original.threads[0].vc);
}

#[test]
fn machine_shared_register_mutations_deliver_native_and_recorded_footprints() {
    for (footprint, recording) in [
        (None, false),
        (Some(false), false),
        (Some(false), true),
        (Some(true), true),
    ] {
        let mut machine = register_machine();
        let sfoot = Arc::new(SFoot::new(&[0]));
        machine.sfoot = Some(sfoot.clone());
        machine.set_footprint(footprint.map(|sync_only| {
            Box::new(Footprint {
                sync_only,
                ..Footprint::default()
            })
        }));
        if recording {
            assert!(machine.begin_action_effects());
        }
        let uniform = Reg {
            file: File::UR,
            idx: 63,
        };
        let predicate = Reg {
            file: File::UP,
            idx: 2,
        };
        machine.wr(0, uniform, 0x1122_3344_5566_7788, 2);
        machine.wp(1, predicate, true);
        machine.make_undefined(1, uniform);
        machine.make_undefined(1, predicate);
        assert_eq!(machine.warps[0].ur[63], 0x5566_7788);
        assert_eq!(machine.warps[0].ur[64], 0x1122_3344);
        assert_eq!(machine.reg_umask(0, uniform), u32::MAX);
        assert_eq!(
            machine.reg_umask(
                0,
                Reg {
                    file: File::UR,
                    idx: 64
                }
            ),
            0
        );
        assert_eq!(machine.warps[0].up & 4, 4);
        assert_eq!(machine.warps[0].up_def & 4, 0);
        assert!(sfoot.items.lock().unwrap().contains(&(17, 0, 0, 1, true)));
        if recording {
            machine.finish_action_effects(false);
        }
        if let Some(mut footprint) = machine.take_footprint() {
            footprint.normalize();
            assert_eq!(footprint.w, [(17, 0, 0, 1)]);
            assert!(footprint.r.is_empty());
            assert!(!footprint.all);
        }
    }
}

#[test]
fn machine_register_fill_preserves_written_values_and_undefinedness() {
    let mut machine = register_machine();
    let virtual_reg = Reg {
        file: File::V,
        idx: 0,
    };
    let uniform_reg = Reg {
        file: File::UR,
        idx: 0,
    };
    machine.wr(0, virtual_reg, 123, 1);
    machine.wr(0, uniform_reg, 456, 1);
    crate::fills::apply_fill(&mut machine, crate::fills::Fill::Word(0xaabb_ccdd));
    assert_eq!(machine.threads[0].regs[0], 123);
    assert_eq!(machine.threads[0].regs[1], 0xaabb_ccdd_aabb_ccdd);
    assert_eq!(machine.warps[0].ur[0], 456);
    assert_eq!(machine.warps[0].ur[1], 0xaabb_ccdd);
    assert_eq!(machine.reg_umask(0, virtual_reg), 0);
    assert_eq!(
        machine.reg_umask(
            0,
            Reg {
                file: File::V,
                idx: 1
            }
        ),
        u32::MAX
    );
    assert_eq!(machine.reg_umask(0, uniform_reg), 0);
    assert_eq!(
        machine.reg_umask(
            0,
            Reg {
                file: File::UR,
                idx: 1
            }
        ),
        u32::MAX
    );
}

#[test]
fn machine_control_transitions_preserve_registers_and_publish_one_acquire() {
    for status in [
        St::Run,
        St::Bar,
        St::ClusterWait,
        St::Unlaunched,
        St::WarpSync,
        St::Exited,
    ] {
        let mut machine = register_machine();
        let original = machine.clone();
        let thread = machine.threads[0].make_mut();
        let registers = thread.registers.clone();
        thread.set_status(status);
        thread.apply_flow(8, Flow::Next);
        assert_eq!(
            thread.pc,
            if status == St::Run {
                9
            } else {
                original.threads[0].pc
            }
        );
        assert_eq!(thread.st, status);
        thread.apply_flow(12, Flow::Jump(20));
        assert_eq!((thread.pc, thread.st), (20, status));
        thread.apply_flow(12, Flow::Stay);
        assert_eq!((thread.pc, thread.st), (20, status));
        thread.release(123);
        assert_eq!(
            (thread.pc, thread.st, thread.acq_bar),
            (21, St::Run, Some(123))
        );
        assert_eq!(thread.take_acquire(), Some(123));
        assert_eq!(thread.take_acquire(), None);
        assert_eq!(thread.registers, registers);
        assert_eq!(original.threads[0].st, St::Run);
        assert_eq!(original.threads[0].pc, 0);
        assert_eq!(original.threads[0].acq_bar, None);
    }
}

#[test]
fn machine_pending_acquire_clones_a_thread_only_when_consuming_a_key() {
    let mut machine = register_machine();
    let original = machine.clone();
    assert!(std::ptr::eq(&*machine.threads[0], &*original.threads[0]));
    assert_eq!(machine.take_thread_acquire(0), None);
    assert!(std::ptr::eq(&*machine.threads[0], &*original.threads[0]));

    machine.threads[0].make_mut().release(123);
    let pending = machine.clone();
    assert!(std::ptr::eq(&*machine.threads[0], &*pending.threads[0]));
    assert_eq!(machine.take_thread_acquire(0), Some(123));
    assert!(!std::ptr::eq(&*machine.threads[0], &*pending.threads[0]));
    assert_eq!(pending.threads[0].acq_bar, Some(123));
    assert_eq!(machine.threads[0].acq_bar, None);
    assert_eq!(machine.threads[0].pc, pending.threads[0].pc);
    assert_eq!(machine.threads[0].st, pending.threads[0].st);
    let consumed = machine.clone();
    assert_eq!(machine.take_thread_acquire(0), None);
    assert!(std::ptr::eq(&*machine.threads[0], &*consumed.threads[0]));
}

#[test]
fn machine_warp_mutations_keep_registers_barriers_and_owners_disjoint() {
    for recording in [false, true] {
        let mut machine = empty_machine_with_threads(
            ".version 8.8\n.target sm_103a\n.address_size 64\n.visible .entry k() { ret; }",
            64,
        );
        let original = machine.clone();
        machine.sfoot = Some(Arc::new(SFoot::new(&[0])));
        machine.set_footprint(Some(Box::default()));
        if recording {
            assert!(machine.begin_action_effects());
        }
        machine.wr(
            0,
            Reg {
                file: File::UR,
                idx: 1,
            },
            123,
            1,
        );
        machine.write_convergence_mask(0, 1, MaskOp::Insert(0x8000_0001));
        machine.write_convergence_mask(0, 1, MaskOp::Remove(1));
        machine.write_convergence_mask(0, 15, MaskOp::Replace(7));
        machine.write_convergence_mask(1, 1, MaskOp::Replace(3));
        machine.warps[0].make_mut().commit_async();
        assert_eq!(machine.warps[0].ur[1], 123);
        assert_eq!(machine.warps[1].ur, original.warps[1].ur);
        assert_eq!(machine.warps[0].bmask[1], 0x8000_0000);
        assert_eq!(machine.warps[0].bmask[15], 7);
        assert_eq!(machine.warps[1].bmask[1], 3);
        assert_eq!(machine.warps[0].cp_seq, 1);
        assert_eq!(machine.warps[1].cp_seq, 0);
        for index in 0..16 {
            if index != 1 && index != 15 {
                assert_eq!(machine.warps[0].bmask[index], 0);
            }
            if index != 1 {
                assert_eq!(machine.warps[1].bmask[index], 0);
            }
        }
        assert_eq!(original.warps[0].bmask, [0; 16]);
        assert_eq!(original.warps[1].bmask, [0; 16]);
        assert_eq!(original.warps[0].cp_seq, 0);
        assert_eq!(original.warps[0].ur[1], 0);
        assert_eq!(machine.threads[0].pc, original.threads[0].pc);
        assert_eq!(machine.threads[0].st, original.threads[0].st);
        assert_eq!(machine.threads[0].vc, original.threads[0].vc);
        if recording {
            machine.finish_action_effects(false);
        }
        let mut footprint = machine.take_footprint().unwrap();
        footprint.normalize();
        assert_eq!(
            footprint.w,
            [(17, 0, 0, 1), (19, 0, 1, 2), (19, 0, 15, 16), (19, 1, 1, 2)]
        );
        assert!(footprint.r.is_empty());
        assert!(!footprint.all);
        let items = &machine.sfoot.as_ref().unwrap().snapshot().1;
        for &(space, owner, start, end) in &footprint.w {
            assert!(items.contains(&(space, owner, start, end, true)));
        }
        for other in [(17, 0, 0, 1), (19, 0, 2, 3), (19, 1, 1, 2)] {
            let barrier = Footprint {
                w: vec![(19, 0, 1, 2)],
                ..Default::default()
            };
            let independent = Footprint {
                w: vec![other],
                ..Default::default()
            };
            assert!(!barrier.conflicts(&independent));
        }
    }
}

fn allocation_machine(enabled: bool, columns: &str) -> Machine {
    let predicate = u32::from(enabled);
    let text = format!(
        r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k() {{
  .reg .pred %p;
  .reg .b32 %n;
  .shared .align 4 .b32 handle;
  setp.eq.u32 %p, {predicate}, 1;
  @%p tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [handle], {columns};
  ret;
}}
"#
    );
    let mut machine = empty_machine(&text);
    for _ in 0..8 {
        let pc = machine.threads[0].pc;
        if matches!(machine.k.code[pc as usize].op, Op::TmemAlloc { .. }) {
            return machine;
        }
        let members: Vec<usize> = (0..32).collect();
        machine.exec(&members, pc).unwrap();
    }
    panic!("did not reach the allocation");
}

#[test]
fn disabled_allocation_is_enabled_and_advances_with_a_full_allocator() {
    for columns in ["32", "%n"] {
        let mut machine = allocation_machine(false, columns);
        machine.tmem[0].alloc = u16::MAX;
        machine.tmem[0].permit = true;
        machine.tmem[0].relinquished = true;
        let before = format!("{:?}", machine.tmem[0]);
        let members: Vec<usize> = (0..32).collect();
        let pc = machine.threads[0].pc;
        assert!(machine.split_enabled(&members, pc), "{columns}");
        machine.exec(&members, pc).unwrap();
        assert!(machine.threads.iter().all(|t| t.pc == pc + 1));
        assert_eq!(format!("{:?}", machine.tmem[0]), before);
        assert!(machine.findings.is_empty(), "{:?}", machine.findings);
    }
}

#[test]
fn active_allocation_waits_for_capacity_and_invalid_sizes_reach_execution() {
    let members: Vec<usize> = (0..32).collect();
    let mut machine = allocation_machine(true, "32");
    machine.tmem[0].alloc = u16::MAX;
    let pc = machine.threads[0].pc;
    assert!(!machine.split_enabled(&members, pc));
    machine.tmem[0].alloc = 0;
    assert!(machine.split_enabled(&members, pc));

    let mut invalid = allocation_machine(true, "17");
    invalid.tmem[0].alloc = u16::MAX;
    let pc = invalid.threads[0].pc;
    assert!(invalid.split_enabled(&members, pc));
    let error = invalid.exec(&members, pc).unwrap_err();
    assert!(error.contains("must be a power of two"), "{error}");
}

#[test]
fn plain_arrival_generation_order_preserves_the_reachable_race() {
    let text = include_str!("../../../fixtures/mbar_plain_generation_order/kernel.ptx");
    let kernel =
        Arc::new(ptxv_ptx::parse_kernel(text, Some("mbar_plain_generation_order")).unwrap());
    let manifest = serde_json::from_value(json!({
        "name":"plain-arrival-generation", "description":"plain arrival generation membership",
        "entry":"mbar_plain_generation_order", "grid":[1,1,1], "block":[128,1,1], "outputs":["out"],
        "buffers":[{"name":"out","dtype":"i32","shape":[1],"init":"zeros"}],
        "params":[{"kind":"ptr","buffer":"out"}], "variants":[]
    }))
    .unwrap();
    let case = ptxv_case::Case {
        dir: PathBuf::new(),
        m: manifest,
    };
    let launch = Arc::new(crate::build_launch(&case, &kernel).unwrap());
    let machine = Machine::new(kernel, launch);
    let result = crate::explore::explore(machine.clone(), 100_000);
    assert!(result.complete, "incomplete: {:?}", result.findings);
    assert!(
        result.classes().iter().any(|kind| kind == "race"),
        "{:?}",
        result.findings
    );
    let dpor = crate::dpor::explore_dpor(machine, 100_000);
    assert!(dpor.complete, "DPOR incomplete: {:?}", dpor.findings);
    assert!(
        dpor.classes().iter().any(|kind| kind == "race"),
        "DPOR: {:?}",
        dpor.findings
    );
    eprintln!(
        "plain generations: stateful {} states {:?}; DPOR {} states {:?}",
        result.states,
        result.classes(),
        dpor.states,
        dpor.classes()
    );
}

fn scalar_component_machine(lanes: u32, overlap: bool) -> Machine {
    let source = format!(
        r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 data, .param .u64 out) {{
  .reg .b32 %lane, %old;
  .reg .b64 %base, %result, %offset, %word;
  ld.param.u64 %base, [data];
  ld.param.u64 %result, [out];
  mov.u32 %lane, %tid.x;
  mul.wide.u32 %offset, %lane, 4;
  add.u64 %word, %base, {};
  atom.global.add.u32 %old, [%word], 1;
  add.u64 %result, %result, %offset;
  st.global.u32 [%result], %old;
  ret;
}}
"#,
        if overlap { "0" } else { "%offset" }
    );
    let kernel = Arc::new(ptxv_ptx::parse_kernel(&source, Some("k")).unwrap());
    let manifest = serde_json::from_value(json!({
        "name":"scalar-components", "description":"independent atomic lanes",
        "entry":"k", "grid":[1,1,1], "block":[lanes,1,1], "outputs":["out"],
        "buffers":[
            {"name":"data","dtype":"u32","shape":[lanes],"init":"zeros"},
            {"name":"out","dtype":"u32","shape":[lanes],"init":"zeros"}
        ],
        "params":[{"kind":"ptr","buffer":"data"},{"kind":"ptr","buffer":"out"}],
        "variants":[]
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
fn independent_scalar_atomic_components_stay_within_a_linear_state_budget() {
    let result = crate::explore::explore_threads(scalar_component_machine(16, false), 128, 1);
    assert!(
        result.complete,
        "{} states: {:?}",
        result.states, result.findings
    );
    assert_eq!(result.classes(), ["clean"]);
    assert_eq!(result.outputs, [vec![0; 16 * 4]]);
    eprintln!("independent atomic lanes: {} states", result.states);
}

#[test]
fn overlapping_scalar_components_retain_both_lane_orders() {
    let initial = scalar_component_machine(2, true);
    for result in [
        crate::explore::explore_threads(initial.clone(), 128, 1),
        crate::dpor::explore_dpor(initial, 128),
    ] {
        assert!(result.complete, "{} states", result.states);
        let mut outputs = result.outputs;
        outputs.sort();
        assert_eq!(
            outputs,
            [vec![0, 0, 0, 0, 1, 0, 0, 0], vec![1, 0, 0, 0, 0, 0, 0, 0]]
        );
    }
}

fn dead_return_fence_machine(edit: impl FnOnce(String) -> String) -> Machine {
    let case = ptxv_case::Case::load(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/e_fence_sc_lanes"),
    )
    .unwrap();
    let source = edit(case.variant_ptx("good").unwrap());
    let kernel = Arc::new(ptxv_ptx::parse_kernel(&source, Some(&case.m.entry)).unwrap());
    let launch = Arc::new(crate::build_launch(&case, &kernel).unwrap());
    Machine::new(kernel, launch)
}

#[test]
fn dead_atomic_clone_classes_keep_the_distinct_acquire_histories() {
    let initial = dead_return_fence_machine(|source| source);
    let classes = initial.dead_atomic_lanes.as_ref().unwrap().1;
    assert_ne!(classes[0], classes[1]);
    assert_ne!(classes[1], classes[2]);
    assert_eq!(classes[2], classes[31]);
    let (mut at, _, _, cut) = crate::explore::run_once_until(initial, &mut |machine| {
        machine
            .actions_bare()
            .iter()
            .any(|(action, _)| matches!(action, crate::explore::Action::AtomicComponent { .. }))
    });
    assert!(cut);
    assert!(vc_get(&at.threads[0].vc, 32) > vc_get(&at.threads[1].vc, 32));
    let claims = |machine: &Machine| {
        machine
            .actions_bare()
            .into_iter()
            .filter(|(action, _)| matches!(action, crate::explore::Action::AtomicComponent { .. }))
            .count()
    };
    assert_eq!(claims(&at), 3);
    let mut polls = at.clone();
    polls.vip = Some(Arc::new(crate::machine::VipLearn::default()));
    assert_eq!(claims(&polls), 3);
    let probed = crate::race_dead::probe_machine(&at, &[(2, 0)].into());
    assert_eq!(claims(&probed), 4);
    let mut replaced = at.clone();
    Arc::make_mut(&mut replaced.k).code[0]
        .text
        .push_str(" changed");
    assert_eq!(claims(&replaced), 32);
    assert_eq!(claims(&at.private_statics()), 3);
    at.dead_atomic_lanes = None;
    assert_eq!(claims(&at), 32);
}

#[test]
fn dead_atomic_symmetry_refuses_prior_claims_writers_loops_and_observed_returns() {
    for replacement in [
        "atom.global.add.u32 %r7, [%rd3], 1; atom.global.add.u32 %r5",
        "st.global.u32 [%rd3], 1; atom.global.add.u32 %r5",
    ] {
        let machine = dead_return_fence_machine(|source| {
            source.replace("atom.global.add.u32 %r5", replacement)
        });
        assert!(machine.dead_atomic_lanes.is_none(), "{replacement}");
    }
    for replacement in [
        "atom.global.add.u32 %r5, [%rd3], 1; @%p1 bra CONV;",
        "atom.global.add.u32 %r5, [%rd3], 1; mov.u32 %r7, %r5;",
    ] {
        let machine = dead_return_fence_machine(|source| {
            source.replace("atom.global.add.u32 %r5, [%rd3], 1;", replacement)
        });
        assert!(machine.dead_atomic_lanes.is_none(), "{replacement}");
    }

    let machine = dead_return_fence_machine(|source| {
        source.replace(
            "atom.global.add.u32 %r5",
            "setp.eq.u32 %p7, %r1, 2; @%p7 atom.global.add.u32 %r7, [%rd3], 1; atom.global.add.u32 %r5",
        )
    });
    let classes = machine.dead_atomic_lanes.as_ref().unwrap().1;
    assert_ne!(classes[2], classes[3]);
    assert_eq!(classes[3], classes[31]);
    let crowded = dead_return_fence_machine(|source| {
        source.replace("fence.sc.gpu;", &"fence.sc.gpu;\n".repeat(300))
    });
    assert!(crowded.dead_atomic_lanes.is_none());
}

#[test]
fn dead_atomic_fence_orders_complete_with_the_original_state_cap() {
    let initial = dead_return_fence_machine(|source| source);
    for verdict in [
        crate::explore::explore_threads(initial.clone(), 2_000_000, 1),
        crate::dpor::explore_dpor(initial, 2_000_000),
    ] {
        assert!(verdict.complete, "{} states", verdict.states);
        assert_eq!(verdict.classes(), ["race"]);
        eprintln!(
            "dead atomic fence: {} states in {:.3}s",
            verdict.states, verdict.seconds
        );
    }
}

fn before_plain_arrival(expected: u32) -> Machine {
    before_barrier_operation(expected, "mbarrier.arrive.shared::cta.b64 _, [bar];")
}

fn before_barrier_operation(expected: u32, instruction: &str) -> Machine {
    let text = format!(
        r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k() {{
  .reg .pred %p;
  .reg .b32 %tid;
  .shared .align 8 .b64 bar;
  .shared .align 16 .b32 response[4];
  mov.u32 %tid, %tid.x;
  setp.eq.u32 %p, %tid, 0;
  @%p mbarrier.init.shared::cta.b64 [bar], {expected};
  bar.sync 0;
  @%p {instruction}
  ret;
}}
"#
    );
    let mut machine = empty_machine(&text);
    let members: Vec<usize> = (0..32).collect();
    for _ in 0..8 {
        let pc = machine.threads[0].pc;
        if matches!(
            machine.k.code[pc as usize].op,
            Op::MbarArrive { .. } | Op::StAsync { .. }
        ) {
            return machine;
        }
        machine.exec(&members, pc).unwrap();
    }
    panic!("did not reach the arrival");
}

fn seeded_release(cta: u16, actor: u32, clock: u32) -> Vec<MbRel> {
    vec![MbRel {
        cta,
        scope: Scope::Cluster,
        rel: vc_single(actor, clock),
        rel_scope: Scope::Cluster,
        cover: Cover::All,
        asyncx: false,
    }]
}

fn released_clock(releases: &[MbRel]) -> VC {
    let mut clock = VC::new();
    for release in releases {
        vc_join(&mut clock, &release.rel);
    }
    clock
}

fn async_arrival(machine: &Machine, mut kind: AsyncKind) -> AsyncOp {
    let issuer = &machine.threads[0];
    if let AsyncKind::Remote { rels, land, n, .. } = &mut kind {
        *rels = machine.mbar_rels(0, Scope::Cluster, true, *n > 0);
        *land = vc_single(0, vc_get(&issuer.vc, 0));
    }
    AsyncOp {
        fifo: None,
        actor: 0,
        vc: issuer.vc.clone(),
        issuer: 0,
        pc: issuer.pc,
        kind,
        vis: issuer.vis.clone(),
        sh: issuer.vc_sh.clone(),
    }
}

fn complete_arrival(machine: &mut Machine, kind: AsyncKind) {
    let operation = async_arrival(machine, kind);
    machine.pending.push(operation);
    machine.complete_async(0);
}

#[test]
fn each_singleton_arrival_producer_records_a_phase_membership_read() {
    for producer in 0..4 {
        let mut machine = before_plain_arrival(2);
        let (&(cta, mbar), _) = machine.mbars.iter().next().unwrap();
        machine.set_footprint(Some(Box::new(Footprint {
            sync_only: true,
            ..Default::default()
        })));
        match producer {
            0 => {
                let members: Vec<usize> = (0..32).collect();
                let pc = machine.threads[0].pc;
                machine.exec(&members, pc).unwrap();
            }
            1 => complete_arrival(&mut machine, AsyncKind::Commit { cta, mbar }),
            2 => complete_arrival(
                &mut machine,
                AsyncKind::CpArrive {
                    cta,
                    mbar,
                    writes: vec![],
                    tx: false,
                },
            ),
            _ => complete_arrival(
                &mut machine,
                AsyncKind::Remote {
                    cta,
                    mbar,
                    tx: 0,
                    n: 1,
                    store: vec![],
                    fresh: false,
                    rels: vec![],
                    land: VC::new(),
                },
            ),
        }
        assert_eq!(
            machine.mbars[&(cta, mbar)].pending,
            1,
            "producer {producer}"
        );
        let key = (mbar as u64) << 20;
        assert!(
            machine
                .fp
                .as_ref()
                .unwrap()
                .r
                .contains(&(18, cta as u32, key, key + 1)),
            "producer {producer}: {:?}",
            machine.fp
        );
        assert!(
            !machine
                .fp
                .as_ref()
                .unwrap()
                .w
                .iter()
                .any(|&(space, owner, lo, hi)| {
                    space == 18 && owner == cta as u32 && lo <= key && key < hi
                })
        );
        assert!(
            machine.findings.is_empty(),
            "producer {producer}: {:?}",
            machine.findings
        );
    }
}

#[test]
fn remote_batch_records_every_generation_it_arrives_in() {
    let mut machine = before_plain_arrival(1);
    let (&(cta, mbar), _) = machine.mbars.iter().next().unwrap();
    machine.set_footprint(Some(Box::new(Footprint {
        sync_only: true,
        ..Default::default()
    })));
    complete_arrival(
        &mut machine,
        AsyncKind::Remote {
            cta,
            mbar,
            tx: 0,
            n: 3,
            store: vec![],
            fresh: false,
            rels: vec![],
            land: VC::new(),
        },
    );
    assert_eq!(machine.mbars[&(cta, mbar)].phase, 3);
    assert_eq!(machine.mbars[&(cta, mbar)].pending, 1);
    let footprint = machine.fp.as_ref().unwrap();
    for phase in 0..3 {
        let key = ((mbar as u64) << 20) | phase;
        assert!(
            footprint
                .w
                .iter()
                .any(|&(space, id, start, end)| space == 18
                    && id == cta as u32
                    && start <= key
                    && key < end),
            "{footprint:?}"
        );
    }
    assert!(machine.findings.is_empty(), "{:?}", machine.findings);
}

#[test]
fn all_arrival_producers_publish_to_the_old_parity_before_phase_wrap() {
    for producer in 0..4 {
        let mut machine = before_plain_arrival(1);
        let (&(cta, mbar), _) = machine.mbars.iter().next().unwrap();
        vc_join(&mut machine.threads[0].make_mut().vc, &vc_single(1000, 7));
        let before = machine.mbars.get_mut(&(cta, mbar)).unwrap();
        before.phase = u32::MAX;
        before.cur = seeded_release(cta, 1001, 11);
        before.done = [seeded_release(cta, 1002, 13), seeded_release(cta, 1003, 17)];
        machine.set_footprint(Some(Box::new(Footprint {
            sync_only: true,
            ..Default::default()
        })));
        match producer {
            0 => {
                let members: Vec<usize> = (0..32).collect();
                let pc = machine.threads[0].pc;
                machine.exec(&members, pc).unwrap();
            }
            1 => complete_arrival(&mut machine, AsyncKind::Commit { cta, mbar }),
            2 => complete_arrival(
                &mut machine,
                AsyncKind::CpArrive {
                    cta,
                    mbar,
                    writes: vec![],
                    tx: false,
                },
            ),
            _ => complete_arrival(
                &mut machine,
                AsyncKind::Remote {
                    cta,
                    mbar,
                    tx: 0,
                    n: 1,
                    store: vec![],
                    fresh: false,
                    rels: vec![],
                    land: VC::new(),
                },
            ),
        }
        let after = &machine.mbars[&(cta, mbar)];
        assert_eq!(
            (after.pending, after.tx, after.phase),
            (1, 0, 0),
            "producer {producer}"
        );
        assert_eq!(
            vc_get(&released_clock(&after.cur), 1000),
            0,
            "producer {producer}"
        );
        assert_eq!(
            vc_get(&released_clock(&after.cur), 1001),
            0,
            "producer {producer}"
        );
        assert_eq!(
            vc_get(&released_clock(&after.done[1]), 1000),
            7,
            "producer {producer}"
        );
        assert_eq!(
            vc_get(&released_clock(&after.done[1]), 1001),
            11,
            "producer {producer}"
        );
        assert_eq!(
            vc_get(&released_clock(&after.done[1]), 1003),
            0,
            "producer {producer}"
        );
        assert_eq!(
            vc_get(&released_clock(&after.done[0]), 1002),
            13,
            "producer {producer}"
        );
        let next_key = (mbar as u64) << 20;
        assert!(
            machine
                .fp
                .as_ref()
                .unwrap()
                .w
                .contains(&(7, cta as u32, next_key, next_key + 1)),
            "producer {producer}: {:?}",
            machine.fp
        );
        assert!(
            machine.findings.is_empty(),
            "producer {producer}: {:?}",
            machine.findings
        );
    }
}

#[test]
fn transaction_only_completions_publish_once_the_last_debt_is_paid() {
    for remote in [false, true] {
        let mut machine = before_plain_arrival(2);
        let (&(cta, mbar), _) = machine.mbars.iter().next().unwrap();
        vc_join(&mut machine.threads[0].make_mut().vc, &vc_single(1000, 7));
        let before = machine.mbars.get_mut(&(cta, mbar)).unwrap();
        before.pending = 0;
        before.tx = 2;
        before.cur = seeded_release(cta, 1001, 11);
        before.done = [vec![], seeded_release(cta, 1002, 13)];
        for payment in 1..=2 {
            let kind = if remote {
                AsyncKind::Remote {
                    cta,
                    mbar,
                    tx: -1,
                    n: 0,
                    store: vec![],
                    fresh: false,
                    rels: vec![],
                    land: VC::new(),
                }
            } else {
                AsyncKind::CpArrive {
                    cta,
                    mbar,
                    writes: vec![],
                    tx: true,
                }
            };
            complete_arrival(&mut machine, kind);
            let after = &machine.mbars[&(cta, mbar)];
            if payment == 1 {
                assert_eq!((after.pending, after.tx, after.phase), (0, 1, 0));
                assert_eq!(vc_get(&released_clock(&after.cur), 1000), 7);
                assert_eq!(vc_get(&released_clock(&after.done[0]), 1000), 0);
            } else {
                assert_eq!((after.pending, after.tx, after.phase), (2, 0, 1));
                assert_eq!(vc_get(&released_clock(&after.cur), 1000), 0);
                assert_eq!(vc_get(&released_clock(&after.done[0]), 1000), 7);
                assert_eq!(vc_get(&released_clock(&after.done[0]), 1001), 11);
                assert_eq!(vc_get(&released_clock(&after.done[1]), 1002), 13);
            }
        }
        assert!(machine.findings.is_empty(), "{:?}", machine.findings);
    }
}

#[test]
fn remote_overarrival_keeps_the_transaction_update_without_publishing() {
    let mut machine = before_plain_arrival(2);
    let (&(cta, mbar), _) = machine.mbars.iter().next().unwrap();
    vc_join(&mut machine.threads[0].make_mut().vc, &vc_single(1000, 7));
    let before = machine.mbars.get_mut(&(cta, mbar)).unwrap();
    before.pending = 0;
    before.tx = 5;
    before.cur = seeded_release(cta, 1001, 11);
    before.done = [seeded_release(cta, 1002, 13), seeded_release(cta, 1003, 17)];
    let done_before = format!("{:?}", before.done);
    let done_land_before = before.done_land.clone();
    complete_arrival(
        &mut machine,
        AsyncKind::Remote {
            cta,
            mbar,
            tx: -3,
            n: 1,
            store: vec![],
            fresh: false,
            rels: vec![],
            land: VC::new(),
        },
    );
    let after = &machine.mbars[&(cta, mbar)];
    assert_eq!((after.pending, after.tx, after.phase), (0, 2, 0));
    // The remote update lands its release before the arrival-count fault.
    assert_eq!(vc_get(&released_clock(&after.cur), 1000), 7);
    assert_eq!(vc_get(&released_clock(&after.cur), 1001), 11);
    assert_eq!(vc_get(&released_clock(&after.done[0]), 1002), 13);
    assert_eq!(vc_get(&released_clock(&after.done[1]), 1003), 17);
    assert_eq!(format!("{:?}", after.done), done_before);
    assert_eq!(after.done_land, done_land_before);
    assert_eq!(vc_get(&released_clock(&after.done[0]), 1000), 0);
    assert_eq!(vc_get(&released_clock(&after.done[1]), 1000), 0);
    assert!(
        machine
            .findings
            .iter()
            .any(|finding| finding.kind == "fault"),
        "{:?}",
        machine.findings
    );
}

#[test]
fn local_transaction_only_completion_releases_only_the_last_fence_clock() {
    let mut machine = before_barrier_operation(
        1,
        "mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [bar], 1;",
    );
    let (&(cta, mbar), _) = machine.mbars.iter().next().unwrap();
    vc_join(&mut machine.threads[0].make_mut().vc, &vc_single(1000, 7));
    vc_join(
        &mut machine.threads[0].make_mut().membar_vc,
        &vc_single(1001, 11),
    );
    let before = machine.mbars.get_mut(&(cta, mbar)).unwrap();
    before.pending = 0;
    before.tx = 1;
    before.cur = seeded_release(cta, 1002, 13);
    let members: Vec<usize> = (0..32).collect();
    let pc = machine.threads[0].pc;
    machine.exec(&members, pc).unwrap();
    let after = &machine.mbars[&(cta, mbar)];
    assert_eq!((after.pending, after.tx, after.phase), (1, 0, 1));
    assert_eq!(vc_get(&released_clock(&after.done[0]), 1000), 0);
    assert_eq!(vc_get(&released_clock(&after.done[0]), 1001), 11);
    assert_eq!(vc_get(&released_clock(&after.done[0]), 1002), 13);
    assert!(machine.findings.is_empty(), "{:?}", machine.findings);
}

#[test]
fn transaction_phase_membership_preserves_the_reachable_deadlock() {
    let text = include_str!("../../../fixtures/mbar_transaction_generation_order/kernel.ptx");
    let kernel =
        Arc::new(ptxv_ptx::parse_kernel(text, Some("mbar_transaction_generation_order")).unwrap());
    let manifest = serde_json::from_value(json!({
        "name":"transaction-generation", "description":"transaction phase membership",
        "entry":"mbar_transaction_generation_order", "grid":[1,1,1], "block":[96,1,1],
        "outputs":[], "buffers":[], "params":[], "variants":[]
    }))
    .unwrap();
    let case = ptxv_case::Case {
        dir: PathBuf::new(),
        m: manifest,
    };
    let launch = Arc::new(crate::build_launch(&case, &kernel).unwrap());
    let machine = Machine::new(kernel, launch);
    let stateful = crate::explore::explore(machine.clone(), 100_000);
    let dpor = crate::dpor::explore_dpor(machine, 100_000);
    assert!(stateful.complete && dpor.complete);
    assert_eq!(stateful.classes(), ["deadlock"]);
    assert_eq!(dpor.classes(), ["deadlock"]);
    assert!(stateful.terminal_states > 0 && dpor.terminal_states > 0);
    eprintln!(
        "transaction generations: stateful {} states, DPOR {} states",
        stateful.states, dpor.states
    );
}

#[test]
fn every_byte_completion_publishes_and_records_phase_membership() {
    for producer in 0..3 {
        let mut machine = if producer == 2 {
            before_barrier_operation(
                2,
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [response], 0, [bar];",
            )
        } else {
            before_plain_arrival(2)
        };
        let (&(cta, mbar), _) = machine.mbars.iter().next().unwrap();
        vc_join(&mut machine.threads[0].make_mut().vc, &vc_single(1000, 7));
        let before = machine.mbars.get_mut(&(cta, mbar)).unwrap();
        before.phase = u32::MAX;
        before.pending = 0;
        before.tx = if producer == 0 { 16 } else { 4 };
        before.cur = seeded_release(cta, 1001, 11);
        before.done = [seeded_release(cta, 1002, 13), vec![]];
        machine.set_footprint(Some(Box::new(Footprint {
            sync_only: true,
            ..Default::default()
        })));
        match producer {
            0 => {
                let operation = async_arrival(
                    &machine,
                    AsyncKind::Clc {
                        cta,
                        resp: 16,
                        mbar,
                        others: vec![],
                    },
                );
                machine.pending.push(operation);
                machine.complete_clc(0, None);
            }
            1 => complete_arrival(
                &mut machine,
                AsyncKind::Tma {
                    cta,
                    dst: 16,
                    mbar_cta: cta,
                    mbar,
                    coords: vec![1],
                    tmap: Arc::new(TmapDesc {
                        grid: 0,
                        param_off: 0,
                        buffer: String::new(),
                        base: 0,
                        esize: 4,
                        dims: vec![1],
                        strides: vec![],
                        swizzle: 0,
                        boxd: vec![1],
                        elem4: false,
                        fdtype: None,
                        smem_src: Some(cta),
                        im2col: None,
                    }),
                },
            ),
            _ => {
                let members: Vec<usize> = (0..32).collect();
                let pc = machine.threads[0].pc;
                machine.exec(&members, pc).unwrap();
            }
        }
        let after = &machine.mbars[&(cta, mbar)];
        assert_eq!(
            (after.pending, after.tx, after.phase),
            (2, 0, 0),
            "producer {producer}"
        );
        assert_eq!(
            vc_get(&released_clock(&after.cur), 1000),
            0,
            "producer {producer}"
        );
        assert_eq!(
            vc_get(&released_clock(&after.done[1]), 1000),
            7,
            "producer {producer}"
        );
        assert_eq!(
            vc_get(&released_clock(&after.done[1]), 1001),
            11,
            "producer {producer}"
        );
        assert_eq!(
            vc_get(&released_clock(&after.done[0]), 1002),
            13,
            "producer {producer}"
        );
        let next_key = (mbar as u64) << 20;
        assert!(
            machine
                .fp
                .as_ref()
                .unwrap()
                .w
                .contains(&(7, cta as u32, next_key, next_key + 1)),
            "producer {producer}: {:?}",
            machine.fp
        );
        let key = ((mbar as u64) << 20) | 0xf_ffff;
        assert!(
            machine
                .fp
                .as_ref()
                .unwrap()
                .w
                .iter()
                .any(|&(space, id, start, end)| space == 18
                    && id == cta as u32
                    && start <= key
                    && key < end),
            "producer {producer}: {:?}",
            machine.fp
        );
        assert!(
            machine.findings.is_empty(),
            "producer {producer}: {:?}",
            machine.findings
        );
    }
}

#[test]
fn zero_transaction_update_without_completion_does_not_write_phase_membership() {
    let mut machine = before_barrier_operation(
        2,
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [bar], 0;",
    );
    machine.set_footprint(Some(Box::new(Footprint {
        sync_only: true,
        ..Default::default()
    })));
    let members: Vec<usize> = (0..32).collect();
    let pc = machine.threads[0].pc;
    machine.exec(&members, pc).unwrap();
    let footprint = machine.fp.as_ref().unwrap();
    assert!(footprint.r.iter().any(|&(space, ..)| space == 18));
    assert!(!footprint.w.iter().any(|&(space, ..)| space == 18));
    assert!(machine.findings.is_empty(), "{:?}", machine.findings);
}

#[test]
fn completion_arrival_dependency_stays_with_its_barrier_across_phase_wrap() {
    for phase in [0, 1, 0xf_ffff, u32::MAX] {
        let mut machine = before_plain_arrival(1);
        let (&(cta, mbar), _) = machine.mbars.iter().next().unwrap();
        machine.mbars.get_mut(&(cta, mbar)).unwrap().phase = phase;
        machine.set_footprint(Some(Box::new(Footprint {
            sync_only: true,
            ..Default::default()
        })));
        complete_arrival(&mut machine, AsyncKind::Commit { cta, mbar });
        let after = machine.mbars[&(cta, mbar)].phase;
        let key = ((mbar as u64) << 20) | (after as u64 & 0xf_ffff);
        assert!(
            machine
                .fp
                .as_ref()
                .unwrap()
                .w
                .iter()
                .any(|&(space, id, start, end)| space == 7
                    && id == cta as u32
                    && start <= key
                    && key < end),
            "phase {phase:#x} next {after:#x}, missing next arrival key {key:#x}: {:?}",
            machine.fp
        );
    }
}

#[test]
fn invalid_barrier_wait_reports_its_error_in_both_explorers() {
    let text = include_str!("../../../fixtures/mbar_wait_uninitialized/kernel.ptx");
    let machine = empty_machine(text);
    let members: Vec<usize> = (0..32).collect();
    let pc = machine.threads[0].pc;
    assert!(machine.try_wait_result(0).is_err());
    let stateful = crate::explore::explore(machine.clone(), 1000);
    let dpor = crate::dpor::explore_dpor(machine.clone(), 1000);
    eprintln!(
        "invalid wait: stateful {} states {:?}/{}; DPOR {} states {:?}/{}",
        stateful.states,
        stateful.classes(),
        stateful.complete,
        dpor.states,
        dpor.classes(),
        dpor.complete
    );
    assert!(
        machine.split_enabled(&members, pc),
        "invalid wait was blocked instead of reaching its error"
    );
    assert!(stateful.complete && dpor.complete);
    for result in [&stateful, &dpor] {
        assert_eq!(result.classes(), ["deadlock", "violation"]);
        assert!(
            result
                .findings
                .iter()
                .any(|finding| finding.sig == format!("mbar-uninit:{pc}")
                    && finding.msg.contains("used before mbarrier.init"))
        );
        for finding in &result.findings {
            eprintln!("{} {}: {}", finding.kind, finding.sig, finding.msg);
        }
    }
}

#[test]
fn reported_wait_error_blocks_duplicates_and_later_completion_reenables_it() {
    for sync_only in [false, true] {
        let text = include_str!("../../../fixtures/mbar_wait_uninitialized/kernel.ptx");
        let mut machine = empty_machine(text);
        let members: Vec<usize> = (0..32).collect();
        let pc = machine.threads[0].pc;
        machine.set_footprint(Some(Box::new(Footprint {
            sync_only,
            ..Default::default()
        })));
        assert!(machine.split_enabled(&members, pc));
        machine.exec(&members, pc).unwrap();
        let fp = machine.fp.as_ref().unwrap();
        assert_eq!(fp.all, sync_only);
        if !sync_only {
            for space in [0, 6, 12] {
                assert!(fp.r.iter().any(|&(s, ..)| s == space), "{fp:?}");
            }
            assert!(
                fp.w.is_empty(),
                "an invalid query must not mutate shared state: {fp:?}"
            );
        }
        assert!(
            machine
                .threads
                .iter()
                .all(|thread| thread.wait_error == Some(pc))
        );
        assert!(!machine.split_enabled(&members, pc));
        assert_eq!(machine.findings.len(), 1);
        assert_eq!(machine.findings[0].sig, format!("mbar-uninit:{pc}"));
        let initialized = before_plain_arrival(1);
        let (&(cta, mbar), _) = initialized.mbars.iter().next().unwrap();
        machine.mbars = initialized.mbars;
        assert!(!machine.split_enabled(&members, pc));
        complete_arrival(&mut machine, AsyncKind::Commit { cta, mbar });
        assert!(machine.split_enabled(&members, pc));
        machine.exec(&members, pc).unwrap();
        assert!(
            machine
                .threads
                .iter()
                .all(|thread| thread.wait_error.is_none() && thread.pc == pc + 1)
        );
        assert!(
            machine
                .findings
                .iter()
                .any(|finding| finding.sig == format!("mbar-uninit:{pc}"))
        );
    }
}

#[test]
fn invalid_wait_address_keeps_a_universal_dependency() {
    let text = include_str!("../../../fixtures/mbar_wait_uninitialized/kernel.ptx")
        .replace("[bar]", "[bar+16]");
    let mut machine = empty_machine(&text);
    let pc = machine.threads[0].pc;
    let error = machine.try_wait_result(0).unwrap_err();
    assert!(!error.contains("used before mbarrier.init"), "{error}");
    machine.set_footprint(Some(Box::default()));
    machine.exec(&(0..32).collect::<Vec<_>>(), pc).unwrap();
    assert!(machine.fp.as_ref().unwrap().all);
    assert_eq!(machine.findings.len(), 1);
    assert!(machine.findings[0].msg.contains(&error));
}

#[test]
fn mixed_wait_split_keeps_the_pending_members_dependencies() {
    let text = r#"
.version 9.0
.target sm_100a
.address_size 64
.visible .entry k() {
  .reg .b32 %r;
  .reg .b64 %rd<2>;
  .reg .pred %p;
  .shared .align 8 .b64 bars[2];
  mov.u32 %r, %tid.x;
  mul.lo.u32 %r, %r, 8;
  mov.u64 %rd0, bars;
  cvt.u64.u32 %rd1, %r;
  add.u64 %rd0, %rd0, %rd1;
WAIT:
  mbarrier.try_wait.parity.shared::cta.b64 %p, [%rd0], 0;
  @!%p bra WAIT;
  ret;
}
"#;
    let mut machine = empty_machine_with_threads(text, 2);
    for pc in 0..5 {
        machine.exec_one(&[0, 1], pc).unwrap();
    }
    let initialized = before_plain_arrival(1);
    let (&(owner, bar), _) = initialized.mbars.iter().next().unwrap();
    machine.mbars = initialized.mbars;
    let pc = machine.threads[0].pc;
    assert!(machine.try_wait_result(0).unwrap().is_none());
    assert!(machine.try_wait_result(1).is_err());
    assert!(machine.split_enabled(&[0, 1], pc));
    machine.set_footprint(Some(Box::default()));
    machine.exec(&[0, 1], pc).unwrap();
    let fp = machine.fp.as_ref().unwrap();
    assert!(!fp.all);
    for off in [bar as u64, bar as u64 + 8] {
        for space in [0, 6] {
            assert!(
                fp.r.iter().any(|&(s, cta, lo, hi)| {
                    s == space && cta == owner as u32 && lo <= off && off < hi
                }),
                "missing query for offset {off}, space {space}: {fp:?}"
            );
        }
        let key = off << 20;
        assert!(
            fp.r.iter().any(|&(s, cta, lo, hi)| {
                s == 12 && cta == owner as u32 && lo <= key && key < hi
            }),
            "missing phase for offset {off}: {fp:?}"
        );
    }
    assert_eq!(machine.threads[0].wait_error, None);
    assert_eq!(machine.threads[1].wait_error, Some(pc));
    assert!(machine.threads.iter().all(|thread| thread.pc == pc));
}

#[test]
fn uninitialized_nonblocking_peek_is_parked_after_its_first_diagnostic() {
    let text = r#"
.version 9.0
.target sm_100a
.address_size 64
.visible .entry k() {
  .reg .pred %p;
  .shared .align 8 .b64 bar;
  mbarrier.try_wait.parity.shared::cta.b64 %p, [bar], 0;
  ret;
}
"#;
    let mut machine = empty_machine(text);
    let members: Vec<usize> = (0..32).collect();
    let pc = machine.threads[0].pc;
    assert!(machine.peeks.contains(&pc));
    assert!(machine.try_wait_result(0).is_err());
    assert!(machine.split_enabled(&members, pc));
    machine.exec(&members, pc).unwrap();
    assert!(
        machine
            .threads
            .iter()
            .all(|thread| thread.wait_error == Some(pc))
    );
    assert_eq!(machine.findings.len(), 1);
    assert_eq!(machine.findings[0].sig, format!("mbar-uninit:{pc}"));
    eprintln!(
        "nonblocking peek: {} findings, marker {:?}, still enabled {}",
        machine.findings.len(),
        machine.threads[0].wait_error,
        machine.split_enabled(&members, pc)
    );
    assert!(
        !machine.split_enabled(&members, pc),
        "diagnosed invalid peek remains enabled and repeats without progress"
    );
}

#[test]
fn uninitialized_forced_peek_reports_the_first_query_error() {
    let text = include_str!("../../../fixtures/mbar_forced_peek_uninitialized/kernel.ptx");
    let mut machine = empty_machine(text);
    let members: Vec<usize> = (0..32).collect();
    let pc = machine.threads[0].pc;
    assert!(machine.peeks.contains(&pc));
    let mut cfg = (*machine.bar_cfg).clone();
    cfg.peekwait = crate::explore::peekwait_set(&machine);
    machine.bar_cfg = Arc::new(cfg);
    let skip = crate::explore::peekskip_set(&machine);
    let mut cfg = (*machine.bar_cfg).clone();
    cfg.peekskip = skip;
    machine.bar_cfg = Arc::new(cfg);
    eprintln!("forced peek config: {:?}", machine.bar_cfg.peekskip);
    assert!(machine.bar_cfg.peekskip.contains_key(&pc));
    assert!(machine.try_wait_result(0).is_err());
    machine.exec(&members, pc).unwrap();
    eprintln!(
        "forced invalid peek: {:?} findings, next pc {}",
        machine.findings, machine.threads[0].pc
    );
    assert!(
        machine
            .findings
            .iter()
            .any(|finding| finding.sig == format!("mbar-uninit:{pc}")),
        "forced-failure routing bypassed the initial invalid query's diagnostic"
    );
}

#[test]
fn valid_pending_nonblocking_peek_advances_without_acquiring() {
    let text = include_str!("../../../fixtures/mbar_peek_uninitialized/kernel.ptx");
    let mut machine = empty_machine(text);
    let members: Vec<usize> = (0..32).collect();
    let pc = machine.threads[0].pc;
    let initialized = before_plain_arrival(1);
    machine.mbars = initialized.mbars;
    for t in 0..machine.threads.len() {
        machine.threads[t].make_mut().wait_error = Some(pc);
    }
    assert!(machine.try_wait_result(0).unwrap().is_none());
    assert!(machine.split_enabled(&members, pc));
    let clocks: Vec<_> = machine
        .threads
        .iter()
        .map(|thread| thread.vc.clone())
        .collect();
    let counters = machine.mbars.clone();
    machine.exec(&members, pc).unwrap();
    for (thread, before) in machine.threads.iter().zip(clocks) {
        assert_eq!(thread.pc, pc + 1);
        assert_eq!(thread.wait_error, None);
        assert_eq!(thread.vc, before);
    }
    assert!(machine.findings.is_empty());
    for (key, before) in counters.iter() {
        let after = &machine.mbars[key];
        assert_eq!(
            (after.pending, after.tx, after.phase),
            (before.pending, before.tx, before.phase)
        );
    }
}

#[test]
fn nonblocking_peek_error_finishes_with_a_diagnostic_in_both_explorers() {
    let text = include_str!("../../../fixtures/mbar_peek_uninitialized/kernel.ptx");
    let machine = empty_machine(text);
    for result in [
        crate::explore::explore(machine.clone(), 1000),
        crate::dpor::explore_dpor(machine, 1000),
    ] {
        eprintln!(
            "invalid nonblocking peek: {} states {:?}/{}",
            result.states,
            result.classes(),
            result.complete
        );
        assert!(result.complete);
        assert_eq!(result.classes(), ["deadlock", "violation"]);
        assert!(
            result
                .findings
                .iter()
                .any(|finding| finding.sig == "mbar-uninit:0")
        );
    }
}

#[test]
fn valid_forced_peek_still_advances_as_a_failed_test() {
    let text = include_str!("../../../fixtures/mbar_forced_peek_uninitialized/kernel.ptx");
    let mut machine = empty_machine(text);
    let members: Vec<usize> = (0..32).collect();
    let pc = machine.threads[0].pc;
    let mut cfg = (*machine.bar_cfg).clone();
    cfg.peekwait = crate::explore::peekwait_set(&machine);
    machine.bar_cfg = Arc::new(cfg);
    let skip = crate::explore::peekskip_set(&machine);
    let mut cfg = (*machine.bar_cfg).clone();
    cfg.peekskip = skip;
    machine.bar_cfg = Arc::new(cfg);
    assert!(machine.bar_cfg.peekskip.contains_key(&pc));
    let initialized = before_plain_arrival(1);
    let (&(cta, mbar), _) = initialized.mbars.iter().next().unwrap();
    machine.mbars = initialized.mbars;
    complete_arrival(&mut machine, AsyncKind::Commit { cta, mbar });
    assert!(machine.try_wait_result(0).unwrap().is_some());
    assert!(machine.split_enabled(&members, pc));
    machine.exec(&members, pc).unwrap();
    assert!(machine
        .threads
        .iter()
        .all(|thread| thread.pc == pc + 1 && thread.peekwin.iter().any(|window| window.3 == pc)));
    assert!(machine.findings.is_empty(), "{:?}", machine.findings);
}

#[test]
fn multiple_unit_arrivals_in_one_action_upgrade_phase_membership_to_write() {
    let mut machine = before_plain_arrival(40);
    let (&(cta, barrier), _) = machine.mbars.iter().next().unwrap();
    let pc = machine.threads[0].pc;
    Arc::make_mut(&mut machine.k).code[pc as usize].guard = ptxv_ir::Pred::TRUE;
    machine.set_footprint(Some(Box::default()));
    machine.exec(&(0..32).collect::<Vec<_>>(), pc).unwrap();
    let after = &machine.mbars[&(cta, barrier)];
    assert_eq!((after.pending, after.phase, after.tx), (8, 0, 0));
    let key = (barrier as u64) << 20;
    assert!(
        machine
            .fp
            .as_ref()
            .unwrap()
            .w
            .iter()
            .any(|&(space, owner, lo, hi)| {
                space == 18 && owner == cta as u32 && lo <= key && key < hi
            }),
        "{:?}",
        machine.fp
    );
    assert!(machine.findings.is_empty(), "{:?}", machine.findings);
}

#[test]
fn a_unit_arrival_with_one_pending_and_transaction_debt_is_dependent() {
    let source = r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k() {
 .reg .b32 %tid;
 .reg .pred %p;
 .shared .align 8 .b64 barrier;
 mov.u32 %tid, %tid.x;
 setp.eq.u32 %p, %tid, 0;
 @%p mbarrier.init.shared.b64 [barrier], 1;
 @%p mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [barrier], 4;
 bar.sync 0;
 setp.eq.u32 %p, %tid, 0;
 @%p bra A;
 setp.eq.u32 %p, %tid, 32;
 @%p bra B;
 ret;
A:
 mbarrier.arrive.shared.b64 _, [barrier];
 ret;
B:
 mbarrier.arrive.shared.b64 _, [barrier];
 ret;
}
"#;
    let kernel = Arc::new(ptxv_ptx::parse_kernel(source, Some("k")).unwrap());
    let manifest = serde_json::from_value(json!({
        "name":"unit-arrival-debt", "description":"one pending arrival with transaction debt",
        "entry":"k", "grid":[1,1,1], "block":[64,1,1], "outputs":[],
        "buffers":[], "params":[], "variants":[]
    }))
    .unwrap();
    let case = ptxv_case::Case {
        dir: PathBuf::new(),
        m: manifest,
    };
    let launch = Arc::new(crate::build_launch(&case, &kernel).unwrap());
    let mut initial = Machine::new(kernel, launch);
    let arrivals: Vec<_> = initial
        .k
        .code
        .iter()
        .enumerate()
        .filter_map(|(pc, inst)| {
            matches!(
                inst.op,
                Op::MbarArrive {
                    tx: None,
                    count: None,
                    ..
                }
            )
            .then_some(pc as u32)
        })
        .collect();
    assert_eq!(arrivals.len(), 2);
    for _ in 0..100 {
        let actions = initial.actions();
        let ready = actions
            .iter()
            .filter(|(action, _)| {
                matches!(action,
            crate::explore::Action::Split { pc, .. } if arrivals.contains(pc))
            })
            .count();
        if ready == 2 {
            break;
        }
        let action = actions
            .into_iter()
            .find(|(action, _)| {
                !matches!(action,
            crate::explore::Action::Split { pc, .. } if arrivals.contains(pc))
            })
            .unwrap()
            .0;
        initial.apply(&action);
    }
    assert!(initial.findings.is_empty(), "{:?}", initial.findings);
    let (&(cta, barrier), state) = initial.mbars.iter().next().unwrap();
    assert_eq!((state.pending, state.tx, state.phase), (1, 4, 0));
    let mut signatures = Vec::new();
    for order in [[0usize, 1usize], [1, 0]] {
        let mut machine = initial.clone();
        for (position, index) in order.into_iter().enumerate() {
            let action = machine
                .actions()
                .into_iter()
                .find(|(action, _)| {
                    matches!(action,
                crate::explore::Action::Split { pc, .. } if *pc == arrivals[index])
                })
                .unwrap()
                .0;
            machine.set_footprint(Some(Box::default()));
            machine.apply(&action);
            let key = (barrier as u64) << 20;
            assert!(
                machine
                    .fp
                    .as_ref()
                    .unwrap()
                    .w
                    .iter()
                    .any(|&(space, owner, lo, hi)| {
                        space == 18 && owner == cta as u32 && lo <= key && key < hi
                    })
            );
            if position == 0 {
                assert!(machine.findings.is_empty(), "{:?}", machine.findings);
                let state = &machine.mbars[&(cta, barrier)];
                assert_eq!((state.pending, state.tx, state.phase), (0, 4, 0));
            } else {
                let fault = machine
                    .findings
                    .iter()
                    .find(|finding| finding.kind == "fault")
                    .unwrap();
                assert_eq!(fault.sig, format!("fault:{}", arrivals[index]));
                signatures.push(fault.sig.clone());
            }
        }
    }
    assert_ne!(signatures[0], signatures[1]);
}

fn grouped_machine(expected: u32, tx: Option<u32>, threads: u32) -> Machine {
    let operation = tx.map_or_else(
        || "mbarrier.arrive.shared::cta.b64 _, [bar];".to_string(),
        |bytes| format!("mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], {bytes};"),
    );
    let text = format!(
        r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k() {{
 .reg .b32 %tid;
 .reg .pred %p;
 .shared .align 8 .b64 bar;
 mov.u32 %tid, %tid.x;
 setp.eq.u32 %p, %tid, 0;
 @%p mbarrier.init.shared::cta.b64 [bar], {expected};
 bar.sync 0;
 {operation}
 ret;
}}
"#
    );
    let kernel = Arc::new(ptxv_ptx::parse_kernel(&text, Some("k")).unwrap());
    let arrival_pc = kernel
        .code
        .iter()
        .position(|inst| matches!(inst.op, Op::MbarArrive { .. }))
        .unwrap() as u32;
    let manifest = serde_json::from_value(json!({
        "name":"grouped-arrival", "description":"whole-action footprint budget",
        "entry":"k", "grid":[1,1,1], "block":[threads,1,1], "outputs":[],
        "buffers":[], "params":[], "variants":[]
    }))
    .unwrap();
    let case = ptxv_case::Case {
        dir: PathBuf::new(),
        m: manifest,
    };
    let launch = Arc::new(crate::build_launch(&case, &kernel).unwrap());
    let mut machine = Machine::new(kernel, launch);
    let mut schedule = crate::explore::DetSched::default();
    for _ in 0..100 {
        if machine.threads.iter().all(|thread| thread.pc == arrival_pc) {
            return machine;
        }
        let action = schedule.pick(&machine).unwrap();
        machine.apply(&action);
    }
    panic!("did not reach grouped arrival");
}

fn arrival_action(machine: &Machine, warp: u32) -> crate::explore::Action {
    machine
        .actions()
        .into_iter()
        .find_map(|(action, _)| {
            matches!(action, crate::explore::Action::Split { warp: w, .. } if w == warp)
                .then_some(action)
        })
        .unwrap()
}

fn membership_item(machine: &Machine, write: bool) -> bool {
    let (&(cta, offset), _) = machine.mbars.iter().next().unwrap();
    let key = (offset as u64) << 20;
    let footprint = machine.fp.as_ref().unwrap();
    let items = if write { &footprint.w } else { &footprint.r };
    items
        .iter()
        .any(|&(space, owner, lo, hi)| space == 18 && owner == cta as u32 && lo <= key && key < hi)
}

fn release_frame(releases: &[crate::machine::MbRel]) -> Vec<serde_json::Value> {
    releases
        .iter()
        .map(|release| {
            json!({
                "cta": release.cta,
                "scope": format!("{:?}", release.scope),
                "rel": crate::machine::vc_entries(&release.rel),
                "rel_scope": format!("{:?}", release.rel_scope),
                "cover": format!("{:?}", release.cover),
                "asyncx": release.asyncx,
            })
        })
        .collect()
}

fn semantic_frame(machine: &Machine) -> serde_json::Value {
    use std::hash::Hash;
    use std::hash::Hasher;
    let mut state = std::collections::hash_map::DefaultHasher::new();
    machine.hash(&mut state);
    let clocks: Vec<_> = machine
        .threads
        .iter()
        .map(|thread| {
            (
                thread.pc,
                format!("{:?}", thread.st),
                crate::machine::vc_entries(&thread.vc),
            )
        })
        .collect();
    let barriers: Vec<_> = machine
        .mbars
        .iter()
        .map(|(key, barrier)| {
            (
                key,
                barrier.expected,
                barrier.pending,
                barrier.tx,
                barrier.phase,
                release_frame(&barrier.cur),
                barrier
                    .done
                    .iter()
                    .map(|releases| release_frame(releases))
                    .collect::<Vec<_>>(),
                crate::machine::vc_entries(&barrier.cur_land),
                barrier
                    .done_land
                    .iter()
                    .map(crate::machine::vc_entries)
                    .collect::<Vec<_>>(),
                format!("{:?}", barrier.atomics),
            )
        })
        .collect();
    json!({"state_hash":state.finish(), "clocks":clocks, "barriers":barriers,
        "findings":machine.findings, "outputs":machine.outputs()})
}

#[test]
fn audited_grouped_arrival_reads_only_within_the_whole_action_budget() {
    for sync_only in [false, true] {
        for (pending, tx, write) in [
            (128, None, false),
            (64, None, false),
            (63, None, true),
            (32, None, true),
            (31, None, true),
            (128, Some(1), true),
        ] {
            let initial = grouped_machine(pending, tx, 32);
            let action = arrival_action(&initial, 0);
            let mut unrecorded = initial.clone();
            unrecorded.apply(&action);
            let mut recorded = initial;
            recorded.set_footprint(Some(Box::new(Footprint {
                sync_only,
                ..Default::default()
            })));
            recorded.apply(&action);
            assert_eq!(semantic_frame(&recorded), semantic_frame(&unrecorded));
            assert!(!recorded.action_effects.is_recording());
            assert_eq!(
                membership_item(&recorded, true),
                write,
                "{pending}/{tx:?}/{sync_only}"
            );
            assert_eq!(
                membership_item(&recorded, false),
                !write,
                "{pending}/{tx:?}/{sync_only}"
            );
            if pending == 32 {
                assert!(
                    recorded
                        .fp
                        .as_ref()
                        .unwrap()
                        .w
                        .iter()
                        .any(|item| item.0 == 12)
                );
            }
        }
    }
}

#[test]
fn audited_grouped_swaps_preserve_native_clocks_histories_and_completion() {
    for pending in [64, 128] {
        let initial = grouped_machine(pending, None, 64);
        let mut outcomes = Vec::new();
        for order in [[0, 1], [1, 0]] {
            let mut machine = initial.clone();
            for (index, warp) in order.into_iter().enumerate() {
                let action = arrival_action(&machine, warp);
                machine.set_footprint(Some(Box::default()));
                machine.apply(&action);
                assert_eq!(membership_item(&machine, true), pending == 64 && index == 1);
                if pending == 64 && index == 1 {
                    let footprint = machine.fp.as_ref().unwrap();
                    assert!(footprint.w.iter().any(|item| item.0 == 12));
                    assert!(footprint.w.iter().any(|item| item.0 == 7));
                }
            }
            outcomes.push(semantic_frame(&machine));
        }
        assert_eq!(outcomes[0], outcomes[1]);
    }
}

#[test]
fn grouped_audit_fails_closed_without_a_fresh_action_boundary() {
    let mut direct = grouped_machine(128, None, 32);
    let pc = direct.threads[0].pc;
    direct.set_footprint(Some(Box::default()));
    direct.exec(&(0..32).collect::<Vec<_>>(), pc).unwrap();
    assert!(membership_item(&direct, true));

    let mut accumulated = grouped_machine(128, None, 64);
    accumulated.set_footprint(Some(Box::default()));
    let first = arrival_action(&accumulated, 0);
    accumulated.apply(&first);
    assert!(!membership_item(&accumulated, true));
    let second = arrival_action(&accumulated, 1);
    accumulated.apply(&second);
    assert!(membership_item(&accumulated, true));
}

#[test]
fn grouped_audit_poisoning_survives_duplicate_reports_raw_keys_and_union() {
    for poison in 0..6 {
        let mut machine = grouped_machine(128, None, 32);
        machine.report("fault", "repeated".into(), "existing diagnostic".into());
        machine.set_footprint(Some(Box::default()));
        assert!(machine.begin_action_effects());
        let pc = machine.threads[0].pc;
        machine.exec(&(0..32).collect::<Vec<_>>(), pc).unwrap();
        match poison {
            0 => machine.report("fault", "repeated".into(), "same diagnostic".into()),
            1 => machine.rec_sync(
                Resource {
                    space: ResourceSpace::MbarInitialization,
                    owner: 0,
                    start: 0,
                    end: 1,
                },
                true,
            ),
            2 => {
                let mut footprint = machine.take_footprint().unwrap();
                footprint.w.push((18, 0, 0, 1));
                machine.set_footprint(Some(footprint));
            }
            3 => {
                machine.finish_action_effects(true);
                let mut footprint = machine.take_footprint().unwrap();
                footprint.normalize();
                machine.set_footprint(Some(footprint));
            }
            4 => {
                machine.finish_action_effects(true);
                let mut footprint = machine.take_footprint().unwrap();
                footprint.union(&Footprint::default());
                machine.set_footprint(Some(footprint));
            }
            _ => {
                machine.finish_action_effects(true);
                let pending = machine.take_footprint().unwrap();
                let mut merged = Footprint::default();
                merged.union(&pending);
                machine.set_footprint(Some(Box::new(merged)));
            }
        }
        machine.finish_action_effects(false);
        assert!(membership_item(&machine, true), "poison {poison}");
        assert!(!machine.action_effects.is_recording());
    }
}

#[test]
fn grouped_sfoot_is_conservative_and_nic_namespace_is_disjoint() {
    let mut machine = grouped_machine(128, None, 32);
    machine.sfoot = Some(Arc::new(crate::machine::SFoot::new(&[1024])));
    let action = arrival_action(&machine, 0);
    machine.apply(&action);
    let (_, items) = machine.sfoot.as_ref().unwrap().snapshot();
    assert!(items.iter().any(|item| item.0 == 18 && item.4));
    let nic = Footprint {
        w: vec![(16, 0, 0, 1)],
        ..Default::default()
    };
    let membership = Footprint {
        w: vec![(18, 0, 0, 1)],
        ..Default::default()
    };
    assert!(!nic.conflicts(&membership));
}

#[test]
fn grouped_audit_mixed_lane_transactions_and_zero_count_are_conservative() {
    let mut mixed = grouped_machine(128, Some(0), 32);
    let pc = mixed.threads[0].pc;
    let Op::MbarArrive { tx, .. } = &mut Arc::make_mut(&mut mixed.k).code[pc as usize].op else {
        unreachable!()
    };
    *tx = Some(ptxv_ir::Val::SReg(ptxv_ir::SReg::Tid(0)));
    mixed.set_footprint(Some(Box::default()));
    let action = arrival_action(&mixed, 0);
    mixed.apply(&action);
    assert!(membership_item(&mixed, true));
    assert_eq!(mixed.mbars.iter().next().unwrap().1.tx, 496);

    let mut zero = before_barrier_operation(
        2,
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [bar], 0;",
    );
    zero.set_footprint(Some(Box::default()));
    let action = arrival_action(&zero, 0);
    zero.apply(&action);
    assert!(membership_item(&zero, true));
}

#[test]
fn grouped_audit_remote_phase_crossing_records_every_generation_as_write() {
    let mut machine = before_plain_arrival(1);
    let (&(cta, mbar), _) = machine.mbars.iter().next().unwrap();
    let operation = async_arrival(
        &machine,
        AsyncKind::Remote {
            cta,
            mbar,
            tx: 0,
            n: 3,
            store: vec![],
            fresh: false,
            rels: vec![],
            land: VC::new(),
        },
    );
    machine.pending.push(operation);
    machine.set_footprint(Some(Box::default()));
    machine.apply(&crate::explore::Action::Async(0));
    assert_eq!(machine.mbars[&(cta, mbar)].phase, 3);
    for phase in 0..3 {
        let key = ((mbar as u64) << 20) | phase;
        assert!(
            machine
                .fp
                .as_ref()
                .unwrap()
                .w
                .iter()
                .any(|&(space, owner, lo, hi)| {
                    space == 18 && owner == cta as u32 && lo <= key && key < hi
                })
        );
    }
    assert!(!machine.action_effects.is_recording());
}

#[test]
fn faulting_action_preserves_prior_counter_state_releases_and_accesses() {
    for sync_only in [false, true] {
        let mut initial = grouped_machine(128, Some(1), 32);
        let (&key, _) = initial.mbars.iter().next().unwrap();
        initial.mbars.get_mut(&key).unwrap().tx = i64::MAX - 1;
        let action = arrival_action(&initial, 0);
        let mut unrecorded = initial.clone();
        unrecorded.apply(&action);
        let mut recorded = initial;
        recorded.set_footprint(Some(Box::new(Footprint {
            sync_only,
            ..Default::default()
        })));
        recorded.apply(&action);
        assert_eq!(semantic_frame(&recorded), semantic_frame(&unrecorded));
        let barrier = &recorded.mbars[&key];
        assert_eq!((barrier.pending, barrier.tx), (127, i64::MAX));
        assert!(
            recorded
                .findings
                .iter()
                .any(|finding| finding.kind == "fault")
        );
        let footprint = recorded.fp.as_ref().unwrap();
        assert!(footprint.all);
        assert!(footprint.r.iter().any(|range| range.0 == 7));
        assert_eq!(footprint.rel.len(), 1);
        assert!(membership_item(&recorded, true));
        assert!(!recorded.action_effects.is_recording());
    }
}

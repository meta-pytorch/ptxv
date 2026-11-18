/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_ir::Kernel;
use ptxv_ir::Op;
use ptxv_sim::build_launch;
use ptxv_sim::explore::Action;
use ptxv_sim::explore::DetSched;
use ptxv_sim::explore::run_once_until;
use ptxv_sim::machine::Machine;
use ptxv_sim::machine::St;
use ptxv_sim::num::NumState;

const ACTION_CAP: usize = 10_000;

fn racing_poll_with_dummy_ctas(dummy_ctas: u32) -> Machine {
    racing_poll_setup(dummy_ctas, true, true)
}

fn remove_once(source: String, instruction: &str) -> String {
    assert_eq!(source.matches(instruction).count(), 1);
    source.replace(instruction, "")
}

fn racing_poll_setup(dummy_ctas: u32, publish: bool, progress: bool) -> Machine {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/e_det_poll_progress");
    let mut case = Case::load(dir).unwrap();
    case.m.launches[0].grid = [dummy_ctas, 1, 1];
    let mut source = case.variant_ptx("good").unwrap();
    if !publish {
        source = remove_once(source, "    st.relaxed.sys.global.u32 [%rd1], 1;\n");
    }
    if !progress {
        source = remove_once(source, "    add.u32 %r3, %r3, 1;\n");
    }
    let kernels = case
        .m
        .all_launches()
        .iter()
        .map(|launch| ptxv_ptx::parse_kernel(&source, Some(launch.entry)).unwrap())
        .collect();
    let kernel = Kernel::concat(kernels);
    let launch = build_launch(&case, &kernel).unwrap();
    let mut machine = Machine::new(Arc::new(kernel), Arc::new(launch));
    assert_eq!(machine.launch.grids.len(), 2);
    assert_eq!(machine.threads.len(), 2 + dummy_ctas as usize);
    assert_eq!(machine.threads[2].st, St::Unlaunched);
    assert_eq!(machine.outputs(), 2u32.to_le_bytes());
    let producer = machine.threads[1].warp;
    assert_ne!(producer, machine.threads[0].warp);
    let head = machine
        .wpolls
        .keys()
        .find(|&&pc| matches!(machine.k.code[pc as usize].op, Op::Ld { .. }))
        .copied()
        .expect("the racing loop must use the real warp-poll recognizer");
    assert!(!machine.polls.contains_key(&head));

    let prefix = machine
        .actions()
        .into_iter()
        .map(|(action, _)| action)
        .find(|action| matches!(action, Action::Split { warp, .. } if *warp == producer))
        .expect("the producer's initialization must actually be enabled");
    machine.apply(&prefix);
    assert_eq!(machine.outputs(), 0u32.to_le_bytes());
    assert_eq!(machine.threads[2].st, St::Unlaunched);
    let publication_pc = machine.threads[1].pc;
    if publish {
        assert!(matches!(
            machine.k.code[publication_pc as usize].op,
            Op::St { sync: Some(_), .. }
        ));
    }
    assert!(machine.actions().iter().any(|(action, _)| {
        matches!(action, Action::Split { warp, pc, .. }
            if *warp == producer && *pc == publication_pc)
    }));
    assert!(
        !machine
            .findings
            .iter()
            .any(|finding| finding.kind == "fault")
    );
    machine
}

fn racing_poll() -> Machine {
    racing_poll_with_dummy_ctas(1)
}

#[test]
fn full_scheduler_advances_a_producer_despite_a_racing_poll() {
    let mut steps = 0;
    let (machine, findings, terminal, stopped) = run_once_until(racing_poll(), &mut |_| {
        if steps == ACTION_CAP {
            return true;
        }
        steps += 1;
        false
    });
    assert!(
        terminal && !stopped,
        "producer starved after {steps} actions"
    );
    assert_eq!(machine.outputs(), 1u32.to_le_bytes());
    assert!(findings.iter().any(|finding| finding.kind == "race"));
    assert!(!findings.iter().any(|finding| finding.kind == "fault"));
    eprintln!("racing poll completed in {steps} actions without losing the race finding");
}

#[test]
fn full_scheduler_preserves_finite_no_split_priority() {
    let mut machine = racing_poll_with_dummy_ctas(2);
    let producer = machine.threads[1].warp;
    for _ in 0..16 {
        let actions = machine.actions();
        let action = actions
            .iter()
            .find(|(action, _)| matches!(action, Action::Split { warp, .. } if *warp == producer))
            .or_else(|| {
                actions
                    .iter()
                    .find(|(action, _)| matches!(action, Action::Split { .. }))
            });
        let Some((action, _)) = action else {
            break;
        };
        machine.apply(action);
    }
    assert!(
        machine
            .threads
            .iter()
            .take(2)
            .all(|thread| thread.st == St::Exited)
    );
    let actions = machine.actions();
    assert_eq!(actions.len(), 2);
    assert!(
        actions
            .iter()
            .all(|(action, _)| matches!(action, Action::Launch(_)))
    );
    let mut scheduler = DetSched::default();
    for _ in 0..ACTION_CAP {
        assert_eq!(scheduler.pick(&machine), Some(actions[0].0.clone()));
    }
}

fn assert_lowest_priority(machine: &Machine) {
    let expected = machine
        .actions()
        .into_iter()
        .map(|(action, _)| action)
        .find(|action| matches!(action, Action::Split { .. }))
        .unwrap();
    let mut scheduler = DetSched::default();
    for _ in 0..ACTION_CAP {
        assert_eq!(scheduler.pick(machine), Some(expected.clone()));
    }
}

#[test]
fn full_scheduler_preserves_numerical_priority() {
    let mut machine = racing_poll();
    machine.num = Some(Box::new(NumState::with_mma_eps(
        machine.launch.mma_model.eps(),
    )));
    assert_lowest_priority(&machine);
}

#[test]
fn full_scheduler_preserves_explicit_lowest_priority() {
    if std::env::var("PTXV_DET").as_deref() != Ok("lowest") {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "full_scheduler_preserves_explicit_lowest_priority",
                "--nocapture",
            ])
            .env("PTXV_DET", "lowest")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "lowest child failed: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    assert_lowest_priority(&racing_poll());
}

#[test]
fn delivered_race_context_allows_a_quiescent_poll_to_block() {
    let mut steps = 0;
    let (machine, findings, terminal, stopped) =
        run_once_until(racing_poll_setup(1, false, false), &mut |_| {
            if steps == ACTION_CAP {
                return true;
            }
            steps += 1;
            false
        });
    assert!(
        !terminal && !stopped,
        "stuttering poll ran for {steps} actions"
    );
    assert!(steps < ACTION_CAP);
    assert_eq!(machine.threads[1].st, St::Exited);
    let mut with_delivered_context = machine.clone();
    with_delivered_context.findings = findings.clone();
    assert!(with_delivered_context.actions().is_empty());
    assert_eq!(machine.outputs(), 0u32.to_le_bytes());
    assert_eq!(findings.iter().filter(|f| f.kind == "race").count(), 1);
    assert_eq!(findings.iter().filter(|f| f.kind == "deadlock").count(), 1);
    assert!(machine.findings.is_empty());
}

#[test]
fn stopped_runs_deliver_initial_and_later_findings_once() {
    for stop_on_race in [false, true] {
        let mut initial = racing_poll_setup(1, false, false);
        initial.report_pcs(
            "note",
            "initial-context".into(),
            "initial input diagnostic".into(),
            vec![],
        );
        let mut steps = 0;
        let (machine, findings, terminal, stopped) = run_once_until(initial, &mut |current| {
            if !stop_on_race || current.findings.iter().any(|f| f.kind == "race") {
                return true;
            }
            assert!(steps < ACTION_CAP, "the racing load was never reached");
            steps += 1;
            false
        });
        assert!(stopped && !terminal);
        assert_eq!(
            findings
                .iter()
                .filter(|f| f.sig == "initial-context")
                .count(),
            1
        );
        assert_eq!(
            findings.iter().filter(|f| f.kind == "race").count(),
            usize::from(stop_on_race)
        );
        assert!(machine.findings.is_empty());
    }
}

fn launch_source(source: &str, ctas: u32) -> Machine {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/e_det_poll_progress");
    let mut case = Case::load(dir).unwrap();
    case.m.grid = [ctas, 1, 1];
    case.m.launches.clear();
    let kernel = ptxv_ptx::parse_kernel(source, Some("main")).unwrap();
    let launch = build_launch(&case, &kernel).unwrap();
    Machine::new(Arc::new(kernel), Arc::new(launch))
}

#[test]
fn terminal_runs_deliver_exit_checks_and_clear_context() {
    let initial = launch_source(
        ".version 9.0\n.target sm_103a\n.address_size 64\n.visible .entry main(.param .u64 flag_p) {\n.reg .b64 %rd<2>;\n.reg .b32 %r<2>;\nld.param.u64 %rd1, [flag_p];\nst.global.u32 [%rd1], %r0;\nret;\n}",
        1,
    );
    let (machine, findings, terminal, stopped) = run_once_until(initial, &mut |_| false);
    assert!(terminal && !stopped);
    assert_eq!(
        findings
            .iter()
            .filter(|f| f.sig.starts_with("uninit-gmem-exit:"))
            .count(),
        1
    );
    assert!(machine.findings.is_empty());
}

#[test]
fn fault_threshold_observes_sixty_four_fresh_events() {
    let initial = launch_source(
        ".version 9.0\n.target sm_103a\n.address_size 64\n.visible .entry main(.param .u64 flag_p) {\n.reg .b64 %rd<2>;\n.reg .b32 %r<2>;\nld.param.u64 %rd1, [flag_p];\nld.global.u32 %r0, [%rd1+4];\nret;\n}",
        65,
    );
    assert_eq!(initial.threads.len(), 65);
    assert!(initial.threads.iter().all(|thread| thread.st == St::Run));
    let mut steps = 0;
    let (machine, findings, terminal, stopped) = run_once_until(initial, &mut |_| {
        assert!(steps < ACTION_CAP, "consecutive fault guard never stopped");
        steps += 1;
        false
    });
    assert!(!terminal && !stopped);
    assert_eq!(steps, 64);
    assert_eq!(
        machine
            .threads
            .iter()
            .filter(|thread| thread.st == St::Exited)
            .count(),
        64
    );
    assert_eq!(
        machine
            .threads
            .iter()
            .filter(|thread| thread.st == St::Run)
            .count(),
        1
    );
    assert_eq!(findings.len(), 64);
    assert!(findings.iter().all(|f| f.sig.starts_with("fault:")));
    assert!(findings.iter().all(|f| f.sig == findings[0].sig));
    assert!(machine.findings.is_empty());
}

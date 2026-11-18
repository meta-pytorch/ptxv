/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use ptxv_ir::AtomOp;
use ptxv_ir::File;
use ptxv_ir::Op;
use ptxv_ir::Reg;
use ptxv_ir::Val;
use serde_json::json;

use crate::machine::Machine;

fn compiled_machine(kind: &str) -> Machine {
    let source = match kind {
        "max_spin" => include_str!("atomic_order_cases/max_spin.ptx"),
        "max_spin_private" => include_str!("atomic_order_cases/max_spin_private.ptx"),
        "max_spin_single" => include_str!("atomic_order_cases/max_spin_single.ptx"),
        "unused_cas" => include_str!("atomic_order_cases/unused_cas.ptx"),
        _ => panic!("unknown CAS.SPIN test case {kind}"),
    };
    static NEXT_COMPILE: AtomicU64 = AtomicU64::new(0);
    let serial = NEXT_COMPILE.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "ptxv-cas-spin-{}-{serial}-{kind}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let source_path = directory.join("kernel.ptx");
    let path = directory.join("kernel.cubin");
    std::fs::write(&source_path, source).unwrap();
    ptxv_sass::ptxas(&source_path, &path, "sm_103a").unwrap();
    let kernel = Arc::new(ptxv_sass::load_cubin(&path, "k").unwrap());
    let manifest = serde_json::from_value(json!({
        "name":"compiled-predicate-only-cas-spin", "description":"actual ptxas max loop",
        "entry":"k", "grid":[1,1,1], "block":[3,1,1], "outputs":[],
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

#[test]
fn compiled_predicate_only_cas_spin_max_audit() {
    let certificate = crate::wellsync::certify(compiled_machine("max_spin"));
    eprintln!(
        "COMPILED_CAS_SPIN coverage certificate: {:?}, reasons={:?}",
        certificate.verdict.as_ref().map(|v| v.classes()),
        certificate.reasons
    );
    assert!(
        certificate
            .verdict
            .as_ref()
            .is_none_or(|v| v.classes().iter().any(|c| c == "unmodeled")),
        "predicate-only CAS.SPIN certificate must refuse missing lane coverage: {:?}",
        certificate.verdict.as_ref().map(|v| v.classes())
    );
    for reverse in [false, true] {
        let mut input = compiled_machine("max_spin");
        input.atom_lanes_rev = reverse;
        for (name, verdict) in [
            ("stateful", crate::explore::explore(input.clone(), 10000)),
            ("DPOR", crate::dpor::explore_dpor(input, 10000)),
        ] {
            eprintln!(
                "COMPILED_CAS_SPIN reverse={reverse} {name}: states={} transitions={} complete={} classes={:?}",
                verdict.states,
                verdict.transitions,
                verdict.complete,
                verdict.classes()
            );
            assert!(verdict.complete);
            assert_eq!(verdict.classes(), vec!["unmodeled"]);
        }
    }
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let mut witness = compiled_machine("max_spin");
        let atomic = witness
            .k
            .code
            .iter()
            .position(|i| matches!(i.op, Op::Atom { .. }))
            .unwrap() as u32;
        assert!(matches!(
            witness.k.code[atomic as usize].op,
            Op::Atom {
                op: AtomOp::CasSpin,
                bytes: 8,
                d: None,
                p: Some(Reg {
                    file: File::P,
                    idx: 0
                }),
                ..
            }
        ));
        eprintln!(
            "COMPILED_CAS_SPIN atomic pc={atomic} IR={:?}",
            witness.k.code[atomic as usize]
        );
        let mut prefix = Vec::new();
        for _ in 0..64 {
            if witness.threads.iter().all(|t| t.pc == atomic) {
                break;
            }
            let choices = witness.actions();
            assert_eq!(choices.len(), 1, "{choices:?}");
            prefix.push(choices[0].0.clone());
            witness.apply(&choices[0].0);
        }
        assert!(witness.threads.iter().all(|t| t.pc == atomic));
        assert!(witness.findings.is_empty(), "{:?}", witness.findings);
        eprintln!(
            "COMPILED_CAS_SPIN enabled prefix={prefix:?}, atomic choices={:?}",
            witness.actions()
        );
        for actor in order {
            witness.exec_one(&[actor], atomic).unwrap();
        }
        let (witness, findings, complete) = crate::explore::run_once(witness);
        let mut kinds = findings
            .iter()
            .chain(witness.findings.iter())
            .map(|f| f.kind.as_str())
            .collect::<Vec<_>>();
        kinds.sort_unstable();
        kinds.dedup();
        eprintln!("COMPILED_CAS_SPIN order={order:?} complete={complete} classes={kinds:?}");
        assert!(complete);
        assert_eq!(kinds, if order[0] == 1 { vec!["trap"] } else { vec![] });
        for actor in 0..3 {
            let old = witness
                .rd(
                    actor,
                    &Val::Reg(
                        Reg {
                            file: File::R,
                            idx: 4,
                        },
                        2,
                    ),
                )
                .unwrap();
            assert_eq!(old, u64::from(actor != order[0]));
        }
    }
}

fn clean_positive(kind: &str) {
    let sample = compiled_machine(kind);
    assert!(sample.k.code.iter().any(|i| matches!(
        i.op,
        Op::Atom {
            op: AtomOp::CasSpin,
            d: None,
            p: Some(Reg {
                file: File::P,
                idx: 0
            }),
            ..
        }
    )));
    for reverse in [false, true] {
        let mut input = compiled_machine(kind);
        input.atom_lanes_rev = reverse;
        for (name, verdict) in [
            ("stateful", crate::explore::explore(input.clone(), 10000)),
            ("DPOR", crate::dpor::explore_dpor(input, 10000)),
        ] {
            eprintln!(
                "CAS_SPIN_POSITIVE kind={kind} reverse={reverse} {name}: states={} transitions={} complete={} classes={:?}",
                verdict.states,
                verdict.transitions,
                verdict.complete,
                verdict.classes()
            );
            assert!(verdict.complete);
            assert_eq!(verdict.classes(), vec!["clean"]);
        }
    }
    let certificate = crate::wellsync::certify(compiled_machine(kind));
    eprintln!(
        "CAS_SPIN_POSITIVE kind={kind} certificate: verdict={:?} reasons={:?} steps={}",
        certificate.verdict.as_ref().map(|v| v.classes()),
        certificate.reasons,
        certificate.steps
    );
    assert!(certificate.reasons.is_empty());
    assert_eq!(certificate.verdict.unwrap().classes(), vec!["clean"]);
}
#[test]
fn compiled_distinct_words_predicate_cas_spin_stays_clean() {
    clean_positive("max_spin_private");
}
#[test]
fn compiled_single_active_predicate_cas_spin_stays_clean() {
    clean_positive("max_spin_single");
}

#[test]
fn compiled_unused_cas_memory_order_audit() {
    let certificate = crate::wellsync::certify(compiled_machine("unused_cas"));
    eprintln!(
        "UNUSED_CAS coverage certificate: {:?}, reasons={:?}",
        certificate.verdict.as_ref().map(|v| v.classes()),
        certificate.reasons
    );
    assert!(
        certificate
            .verdict
            .as_ref()
            .is_none_or(|v| v.classes().iter().any(|c| c == "unmodeled")),
        "discarded-result CAS certificate must refuse missing lane coverage: {:?}",
        certificate.verdict.as_ref().map(|v| v.classes())
    );
    for reverse in [false, true] {
        let mut input = compiled_machine("unused_cas");
        input.atom_lanes_rev = reverse;
        for (name, verdict) in [
            ("stateful", crate::explore::explore(input.clone(), 10000)),
            ("DPOR", crate::dpor::explore_dpor(input, 10000)),
        ] {
            eprintln!(
                "UNUSED_CAS reverse={reverse} {name}: states={} transitions={} complete={} classes={:?}",
                verdict.states,
                verdict.transitions,
                verdict.complete,
                verdict.classes()
            );
            assert!(verdict.complete);
            assert_eq!(verdict.classes(), vec!["unmodeled"]);
        }
    }
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let mut witness = compiled_machine("unused_cas");
        let atomic = witness
            .k
            .code
            .iter()
            .position(|i| matches!(i.op, Op::Atom { .. }))
            .unwrap() as u32;
        assert!(matches!(
            witness.k.code[atomic as usize].op,
            Op::Atom {
                op: AtomOp::Cas,
                bytes: 4,
                d: None,
                p: None,
                ..
            }
        ));
        for _ in 0..64 {
            if witness.threads.iter().all(|t| t.pc == atomic) {
                break;
            }
            let choices = witness.actions();
            assert_eq!(choices.len(), 1, "{choices:?}");
            witness.apply(&choices[0].0);
        }
        assert!(witness.threads.iter().all(|t| t.pc == atomic));
        assert!(witness.findings.is_empty(), "{:?}", witness.findings);
        eprintln!(
            "UNUSED_CAS pc={atomic} IR={:?} choices={:?}",
            witness.k.code[atomic as usize],
            witness.actions()
        );
        for actor in order {
            witness.exec_one(&[actor], atomic).unwrap();
        }
        let (witness, findings, complete) = crate::explore::run_once(witness);
        let mut kinds = findings
            .iter()
            .chain(witness.findings.iter())
            .map(|f| f.kind.as_str())
            .collect::<Vec<_>>();
        kinds.sort_unstable();
        kinds.dedup();
        eprintln!("UNUSED_CAS order={order:?} complete={complete} classes={kinds:?}");
        assert!(complete);
        assert_eq!(kinds, if order[0] == 1 { vec!["trap"] } else { vec![] });
        for actor in 0..3 {
            let observed = witness
                .rd(
                    actor,
                    &Val::Reg(
                        Reg {
                            file: File::R,
                            idx: 0,
                        },
                        1,
                    ),
                )
                .unwrap();
            assert_eq!(observed, order[0] as u64 + 1);
        }
    }
}

#[test]
fn predicate_owner_projection_distinguishes_constant_and_shared_returns() {
    let mut prepared = compiled_machine("max_spin_private");
    let pc = prepared
        .k
        .code
        .iter()
        .position(|inst| matches!(inst.op, Op::Atom { .. }))
        .unwrap() as u32;
    for _ in 0..64 {
        if prepared.threads.iter().all(|thread| thread.pc == pc) {
            break;
        }
        let actions = prepared.actions();
        assert_eq!(actions.len(), 1, "{actions:?}");
        prepared.apply(&actions[0].0);
    }
    assert!(prepared.threads.iter().all(|thread| thread.pc == pc));
    assert!(prepared.findings.is_empty());
    for (file, index, unsupported) in [
        (File::P, 7, false),
        (File::UP, 7, false),
        (File::UP, 0, true),
    ] {
        let mut input = prepared.clone();
        let mut kernel = (*input.k).clone();
        let Op::Atom { p, .. } = &mut kernel.code[pc as usize].op else {
            unreachable!()
        };
        *p = Some(Reg { file, idx: index });
        input.k = Arc::new(kernel);
        let uniform = input.warps[0].up;
        input.exec_one(&[0, 1, 2], pc).unwrap();
        assert_eq!(
            input
                .findings
                .iter()
                .any(|finding| finding.kind == "unmodeled"),
            unsupported,
            "{file:?}{index}: {:?}",
            input.findings
        );
        if index == 7 {
            assert_eq!(input.warps[0].up, uniform);
        }
    }
}

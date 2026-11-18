/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
use std::path::PathBuf;
use std::sync::Arc;

use ptxv_ir::AtomOp;
use ptxv_ir::Op;
use serde_json::json;

use crate::machine::Machine;

fn compiled_machine(kind: &str) -> Machine {
    let template = include_str!("atomic_order_cases/unused_exch.ptx");
    let source = if kind == "private" {
        template
            .replace(".reg .b64 %address;", ".reg .b64 %address, %offset;")
            .replace(
                "atom.global.exch",
                "mul.wide.u32 %offset,%lane,4; add.u64 %address,%address,%offset; atom.global.exch",
            )
            .replace(
                "setp.eq.u32 %middle,%value,2;",
                "setp.ne.u32 %middle,%value,%replacement;",
            )
    } else {
        template.to_string()
    };
    let directory = std::env::temp_dir().join(format!(
        "ptxv-unused-exchange-{}-{kind}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let source_path = directory.join("kernel.ptx");
    let path = directory.join("kernel.cubin");
    std::fs::write(&source_path, source).unwrap();
    ptxv_sass::ptxas(&source_path, &path, "sm_103a").unwrap();
    let kernel = Arc::new(ptxv_sass::load_cubin(&path, "k").unwrap());
    let manifest = serde_json::from_value(json!({
        "name":"compiled-unused-exchange", "description":"actual unused exchange local order",
        "entry":"k", "grid":[1,1,1], "block":[if kind=="single" {1}else{3},1,1], "outputs":[],
        "buffers":[{"name":"x","dtype":"u32","shape":[if kind=="private" {3}else{1}],"init":"zeros"}], "params":[{"kind":"ptr","buffer":"x"}], "variants":[]
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
fn compiled_unused_exchange_requires_local_order_coverage() {
    let certificate = crate::wellsync::certify(compiled_machine("collision"));
    eprintln!(
        "UNUSED_EXCH coverage certificate: {:?}, reasons={:?}",
        certificate.verdict.as_ref().map(|v| v.classes()),
        certificate.reasons
    );
    assert!(
        certificate
            .verdict
            .as_ref()
            .is_none_or(|v| v.classes().iter().any(|c| c == "unmodeled")),
        "unused EXCH certificate must refuse missing lane coverage"
    );
    for reverse in [false, true] {
        let mut input = compiled_machine("collision");
        input.atom_lanes_rev = reverse;
        for (name, verdict) in [
            ("stateful", crate::explore::explore(input.clone(), 10000)),
            ("DPOR", crate::dpor::explore_dpor(input, 10000)),
        ] {
            eprintln!(
                "UNUSED_EXCH reverse={reverse} {name}: states={} transitions={} complete={} classes={:?}",
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
        let mut witness = compiled_machine("collision");
        let atomic = witness
            .k
            .code
            .iter()
            .position(|i| matches!(i.op, Op::Atom { .. }))
            .unwrap() as u32;
        assert!(matches!(
            witness.k.code[atomic as usize].op,
            Op::Atom {
                op: AtomOp::Exch,
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
            "UNUSED_EXCH pc={atomic} IR={:?} choices={:?}",
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
        eprintln!("UNUSED_EXCH order={order:?} complete={complete} classes={kinds:?}");
        assert!(complete);
        assert_eq!(kinds, if order[2] == 1 { vec!["trap"] } else { vec![] });
        let observed = (0..4).fold(0u32, |word, at| {
            word | ((*witness.gmem[0].bytes.get(at) as u32) << (8 * at))
        });
        assert_eq!(observed, order[2] as u32 + 1);
    }
}

#[test]
fn compiled_unused_exchange_private_and_single_controls() {
    for kind in ["private", "single"] {
        for reverse in [false, true] {
            let mut input = compiled_machine(kind);
            input.atom_lanes_rev = reverse;
            for result in [
                crate::explore::explore(input.clone(), 10000),
                crate::dpor::explore_dpor(input, 10000),
            ] {
                eprintln!(
                    "EXCH_POSITIVE kind={kind} reverse={reverse} states={} complete={} classes={:?}",
                    result.states,
                    result.complete,
                    result.classes()
                );
                assert!(result.complete);
                assert_eq!(result.classes(), vec!["clean"]);
            }
        }
        let certificate = crate::wellsync::certify(compiled_machine(kind));
        eprintln!(
            "EXCH_POSITIVE kind={kind} certificate={:?} reasons={:?}",
            certificate.verdict.as_ref().map(|v| v.classes()),
            certificate.reasons
        );
        assert!(certificate.reasons.is_empty());
        assert_eq!(certificate.verdict.unwrap().classes(), vec!["clean"]);
    }
}

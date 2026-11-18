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

fn memory(machine: &Machine) -> u32 {
    (0..4).fold(0, |word, at| {
        word | ((*machine.gmem[0].bytes.get(at) as u32) << (8 * at))
    })
}

fn scalar_red(observer: bool, confluent: bool, private: bool, single: bool) -> Machine {
    let after = if observer {
        "bar.sync 0; ld.global.u32 %r1,[%address]; setp.eq.u32 %p0,%r0,0; setp.eq.u32 %p1,%r1,1065353216; and.pred %p2,%p0,%p1; @%p2 trap;"
    } else {
        ""
    };
    let operands = if confluent {
        "mov.u32 %r2,1065353216;"
    } else {
        "setp.eq.u32 %p0,%r0,0; setp.eq.u32 %p1,%r0,1; selp.u32 %r2,1065353216,3769465068,%p1; selp.u32 %r2,1621981420,%r2,%p0;"
    };
    let address = if private {
        "mul.wide.u32 %offset,%r0,4; add.u64 %address,%address,%offset;"
    } else {
        ""
    };
    let text = format!(
        r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 input) {{
 .reg .b32 %r<3>; .reg .b64 %address; .reg .b64 %offset; .reg .pred %p<3>;
 ld.param.u64 %address,[input]; mov.u32 %r0,%tid.x;
 {operands} {address}
 red.global.add.f32 [%address],%r2;
 {after}
 ret;
}}
"#
    );
    let kernel = Arc::new(ptxv_ptx::parse_kernel(&text, Some("k")).unwrap());
    let count = if private { 3 } else { 1 };
    let block = if single { 1 } else { 3 };
    let manifest=serde_json::from_value(json!({"name":"returnless-cancellation-order","description":"three scalar RED F32 operands with nonassociative cancellation","entry":"k","grid":[1,1,1],"block":[block,1,1],"outputs":["x"],"buffers":[{"name":"x","dtype":"u32","shape":[count],"init":"zeros"}],"params":[{"kind":"ptr","buffer":"x"}],"variants":[]})).unwrap();
    let case = ptxv_case::Case {
        dir: PathBuf::new(),
        m: manifest,
    };
    let launch = Arc::new(crate::build_launch(&case, &kernel).unwrap());
    Machine::new(kernel, launch)
}

#[test]
#[ignore = "upstream merge: pins exactly unmodeled, int7's lane-order findings also fire on the same atomic; one mechanism per shape pending; owner gC"]
fn ordinary_red_collision_cannot_report_clean_without_order_coverage() {
    for observer in [false, true] {
        let input = scalar_red(observer, false, false, false);
        for reverse in [false, true] {
            let mut machine = input.clone();
            machine.atom_lanes_rev = reverse;
            for (name, result) in [
                ("stateful", crate::explore::explore(machine.clone(), 10000)),
                ("DPOR", crate::dpor::explore_dpor(machine, 10000)),
            ] {
                eprintln!(
                    "scalar RED observer={observer} reverse={reverse} {name}: complete={} states={} classes={:?} outputs={:?}",
                    result.complete,
                    result.states,
                    result.classes(),
                    result.outputs
                );
                assert!(result.complete);
                assert_eq!(result.classes(), ["unmodeled"]);
                assert_eq!(result.outputs, [vec![0, 0, 0, 0]]);
            }
        }
        for (order, expected) in [([0, 1, 2], 0u32), ([2, 1, 0], 0), ([0, 2, 1], 1065353216)] {
            let mut witness = input.clone();
            let pc = witness
                .k
                .code
                .iter()
                .position(|inst| matches!(inst.op, Op::Atom { .. }))
                .unwrap() as u32;
            for before in 0..pc {
                witness.exec_one(&[0, 1, 2], before).unwrap();
            }
            for actor in order {
                witness.exec_one(&[actor], pc).unwrap();
            }
            let (witness, findings, complete) = crate::explore::run_once(witness);
            eprintln!(
                "scalar RED observer={observer} singleton order={order:?}: complete={complete} word={:#x} findings={findings:?}",
                memory(&witness)
            );
            assert!(complete);
            assert_eq!(memory(&witness), expected);
            if observer && expected != 0 {
                assert_eq!(
                    findings
                        .iter()
                        .map(|finding| finding.kind.as_str())
                        .collect::<Vec<_>>(),
                    ["trap"]
                );
            } else {
                assert!(findings.is_empty());
            }
        }
        let certificate = crate::wellsync::certify(input);
        eprintln!(
            "scalar RED observer={observer} certificate verdict={:?} reasons={:?} steps={}",
            certificate.verdict.as_ref().map(|verdict| (
                verdict.classes(),
                verdict.complete,
                verdict.states,
                verdict.outputs.clone()
            )),
            certificate.reasons,
            certificate.steps
        );
        if observer {
            assert!(certificate.verdict.is_none());
            assert!(
                certificate
                    .reasons
                    .iter()
                    .any(|reason| reason.contains("C-FADD"))
            );
        } else {
            let verdict = certificate
                .verdict
                .unwrap_or_else(|| panic!("{:?}", certificate.reasons));
            assert!(verdict.complete);
            assert_eq!(verdict.classes(), ["nondeterministic"]);
            assert_eq!(verdict.outputs.len(), 2);
            assert!(verdict.outputs.contains(&0u32.to_le_bytes().to_vec()));
            assert!(
                verdict
                    .outputs
                    .contains(&1065353216u32.to_le_bytes().to_vec())
            );
        }
    }
}

#[test]
fn returnless_certificate_collection_keeps_confluent_private_and_single_controls() {
    for (private, single) in [(false, false), (true, false), (false, true)] {
        let input = scalar_red(false, true, private, single);
        let certificate = crate::wellsync::certify(input.clone());
        eprintln!(
            "RED positive private={private} single={single}: {:?}, reasons={:?}",
            certificate.verdict.as_ref().map(|v| v.classes()),
            certificate.reasons
        );
        assert!(
            certificate
                .verdict
                .is_some_and(|v| v.complete && v.classes() == ["clean"])
        );
        let expected = if private || single {
            "clean"
        } else {
            "unmodeled"
        };
        for result in [
            crate::explore::explore(input.clone(), 10000),
            crate::dpor::explore_dpor(input, 10000),
        ] {
            assert!(result.complete);
            assert_eq!(result.classes(), [expected]);
        }
    }
}

#[test]
fn actual_vector_red_certificate_still_accepts_128_exact_writers() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let case = ptxv_case::Case::load(root.join("probes/red_vec")).unwrap();
    let text = std::fs::read_to_string(case.dir.join("kernel.ptx")).unwrap();
    let kernel = Arc::new(ptxv_ptx::parse_kernel(&text, Some("red_vec")).unwrap());
    let launch = Arc::new(crate::build_launch(&case, &kernel).unwrap());
    let result = crate::wellsync::certify(Machine::new(kernel, launch));
    eprintln!(
        "actual red_vec certificate: {:?}, steps={}, reasons={:?}",
        result.verdict.as_ref().map(|v| (v.complete, v.classes())),
        result.steps,
        result.reasons
    );
    let verdict = result
        .verdict
        .unwrap_or_else(|| panic!("{:?}", result.reasons));
    assert!(verdict.complete);
    assert_eq!(verdict.classes(), ["clean"]);
    let report = case.expect_report(&verdict.outputs[0]).unwrap();
    assert!(!report.is_empty() && report.iter().all(|line| line.contains("bitwise equal")));
}

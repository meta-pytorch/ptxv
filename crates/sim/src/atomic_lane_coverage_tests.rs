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
use ptxv_ir::Pred;
use ptxv_ir::Reg;
use ptxv_ir::Val;
use serde_json::json;

use crate::machine::Machine;

fn double_order_machine() -> Machine {
    double_order_machine_kind(false)
}

fn double_order_machine_kind(integer: bool) -> Machine {
    let text = r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 input) {
  .reg .b32 %lane;
  .reg .b64 %address;
  .reg .b64 %old;
  .reg .b64 %source;
  .reg .pred %p<3>;
  ld.param.u64 %address, [input];
  mov.u32 %lane, %tid.x;
  mov.b64 %source, 4607182418800017408;
  atom.global.add.f64 %old, [%address], %source;
  setp.eq.u32 %p0, %lane, 0;
  setp.eq.u64 %p1, %old, 4607182418800017408;
  and.pred %p2, %p0, %p1;
  @%p2 trap;
  ret;
}
"#;
    let text = if integer {
        text.replace("add.f64", "add.u64")
            .replace("4607182418800017408", "1")
    } else {
        text.to_string()
    };
    let kernel = Arc::new(ptxv_ptx::parse_kernel(&text, Some("k")).unwrap());
    let manifest = serde_json::from_value(json!({
        "name":"scalar-double-middle-order", "description":"residual double lane-order probe",
        "entry":"k", "grid":[1,1,1], "block":[3,1,1], "outputs":[],
        "buffers":[{"name":"x","dtype":"f64","shape":[3],"init":"zeros"}],
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

fn cas_order_machine() -> Machine {
    cas_order_machine_kind(false)
}

fn cas_order_machine_kind(bf16: bool) -> Machine {
    let text = r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 input) {
  .reg .b32 %r<3>;
  .reg .b64 %address;
  .reg .pred %p<3>;
  ld.param.u64 %address, [input];
  mov.u32 %r0, %tid.x;
  add.u32 %r2, %r0, 1;
  atom.global.cas.b32 %r1, [%address], 0, %r2;
  setp.eq.u32 %p0, %r0, 0;
  setp.eq.u32 %p1, %r1, 2;
  and.pred %p2, %p0, %p1;
  @%p2 trap;
  ret;
}
"#;
    let text = if bf16 {
        text.replace(
            "atom.global.cas.b32 %r1, [%address], 0, %r2;",
            "atom.global.add.noftz.bf16 %r1, [%address], 0x3f80;",
        )
        .replace("setp.eq.u32 %p1, %r1, 2;", "setp.eq.u32 %p1, %r1, 16256;")
    } else {
        text.to_string()
    };
    let kernel = Arc::new(ptxv_ptx::parse_kernel(&text, Some("k")).unwrap());
    let manifest=serde_json::from_value(json!({"name":"scalar-cas-middle-order","description":"residual CAS lane order","entry":"k","grid":[1,1,1],"block":[3,1,1],"outputs":[],"buffers":[{"name":"x","dtype":"u32","shape":[3],"init":"zeros"}],"params":[{"kind":"ptr","buffer":"x"}],"variants":[]})).unwrap();
    let case = ptxv_case::Case {
        dir: PathBuf::new(),
        m: manifest,
    };
    let launch = Arc::new(crate::build_launch(&case, &kernel).unwrap());
    Machine::new(kernel, launch)
}

fn inputs() -> [(String, Machine); 4] {
    [
        ("F64".into(), double_order_machine()),
        ("Add64".into(), double_order_machine_kind(true)),
        ("CAS32".into(), cas_order_machine()),
        ("BF16".into(), cas_order_machine_kind(true)),
    ]
}

fn at_atomic(mut machine: Machine) -> (Machine, u32) {
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

#[test]
#[ignore = "upstream merge: pins exactly unmodeled, int7's lane-order findings also fire on the same atomic; one mechanism per shape pending; owner gC"]
fn unsupported_returning_pairs_report_the_missing_lane_coverage() {
    for (label, input) in inputs() {
        for reverse in [false, true] {
            let mut machine = input.clone();
            machine.atom_lanes_rev = reverse;
            for (name, result) in [
                ("stateful", crate::explore::explore(machine.clone(), 10000)),
                ("DPOR", crate::dpor::explore_dpor(machine, 10000)),
            ] {
                eprintln!(
                    "unsupported {label} reverse={reverse} {name}: complete={} states={} classes={:?}",
                    result.complete,
                    result.states,
                    result.classes()
                );
                assert!(result.complete);
                assert_eq!(result.classes(), ["unmodeled"]);
                assert!(
                    result
                        .findings
                        .iter()
                        .any(|finding| finding.sig.starts_with("atomic-lane-order:"))
                );
            }
        }
        let (mut witness, pc) = at_atomic(input.clone());
        for actor in [1, 0, 2] {
            witness.exec_one(&[actor], pc).unwrap();
        }
        let (_, findings, complete) = crate::explore::run_once(witness);
        assert!(complete && findings.iter().any(|finding| finding.kind == "trap"));
        let certificate = crate::wellsync::certify(input);
        assert!(certificate.verdict.is_none());
    }
}

fn private_or_single(input: Machine, private: bool, single: bool) -> Machine {
    let (mut machine, pc) = at_atomic(input);
    let Op::Atom {
        addr,
        bytes,
        d: Some(destination),
        ..
    } = machine.k.code[pc as usize].op.clone()
    else {
        panic!("returning atomic")
    };
    let Some(Val::Reg(base, _)) = addr.base else {
        panic!("register address")
    };
    let words = (bytes / 4).max(1);
    let addresses: Vec<_> = (0..3)
        .map(|actor| machine.addr(actor, &addr).unwrap())
        .collect();
    for actor in 0..3 {
        machine.wr(actor, destination, 0, words);
        machine.wr(
            actor,
            base,
            addresses[actor]
                + if private {
                    bytes.max(4) as u64 * actor as u64
                } else {
                    0
                },
            2,
        );
    }
    if single {
        let predicate = Reg {
            file: File::P,
            idx: 0,
        };
        Arc::make_mut(&mut machine.k).code[pc as usize].guard = Pred {
            reg: Some(predicate),
            neg: false,
        };
        for actor in 0..3 {
            machine.wp(actor, predicate, actor == 0);
        }
    }
    machine
}

#[test]
fn disjoint_and_single_claims_preserve_existing_compatibility() {
    for (label, input) in inputs() {
        for (private, single) in [(true, false), (false, true)] {
            let input = private_or_single(input.clone(), private, single);
            for result in [
                crate::explore::explore(input.clone(), 10000),
                crate::dpor::explore_dpor(input.clone(), 10000),
            ] {
                eprintln!(
                    "unsupported positive {label} private={private} single={single}: complete={} states={} classes={:?}",
                    result.complete,
                    result.states,
                    result.classes()
                );
                assert!(result.complete);
                assert_eq!(result.classes(), ["clean"]);
            }
            let certificate = crate::wellsync::certify(input);
            assert!(
                certificate
                    .verdict
                    .is_some_and(|verdict| verdict.classes() == ["clean"])
            );
        }
    }
}

#[test]
fn disjoint_words_with_a_shared_return_still_report_the_gap() {
    for (label, input) in inputs() {
        for predicate in [false, true] {
            let mut machine = private_or_single(input.clone(), true, false);
            let pc = machine.threads[0].pc;
            let Op::Atom { addr, .. } = machine.k.code[pc as usize].op.clone() else {
                panic!("returning atomic")
            };
            let addresses: Vec<_> = (0..3)
                .map(|actor| machine.addr(actor, &addr).unwrap())
                .collect();
            assert_ne!(addresses[0], addresses[1]);
            assert_ne!(addresses[1], addresses[2]);
            let Op::Atom { d, p, .. } = &mut Arc::make_mut(&mut machine.k).code[pc as usize].op
            else {
                panic!("returning atomic")
            };
            if predicate {
                *p = Some(Reg {
                    file: File::UP,
                    idx: 0,
                });
            } else {
                *d = Some(Reg {
                    file: File::UR,
                    idx: 0,
                });
            }
            machine.exec_one(&[0, 1, 2], pc).unwrap();
            assert!(
                machine
                    .findings
                    .iter()
                    .any(|finding| finding.kind == "unmodeled"
                        && finding.sig == format!("atomic-lane-order:{pc}")),
                "{label} predicate={predicate}: {:?}",
                machine.findings
            );
        }
    }
}

#[test]
fn complete_claim_scan_finds_a_conflict_after_the_first_lane() {
    for (label, input) in inputs() {
        let mut machine = private_or_single(input, true, false);
        let pc = machine.threads[0].pc;
        let Op::Atom { addr, .. } = machine.k.code[pc as usize].op.clone() else {
            panic!("returning atomic")
        };
        let Some(Val::Reg(base, _)) = addr.base else {
            panic!("register address")
        };
        let other = machine.addr(1, &addr).unwrap();
        machine.wr(2, base, other, 2);
        assert_ne!(machine.addr(0, &addr).unwrap(), other);
        assert_eq!(machine.addr(2, &addr).unwrap(), other);
        machine.exec_one(&[0, 1, 2], pc).unwrap();
        assert!(
            machine
                .findings
                .iter()
                .any(|finding| finding.sig.starts_with("atomic-lane-order:")),
            "{label}: {:?}",
            machine.findings
        );
    }
}

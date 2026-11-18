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

fn machine(integer: bool, private: bool, single: bool) -> Machine {
    let (operation, source) = if integer {
        ("add.u32", 1)
    } else {
        ("add.f32", 1065353216)
    };
    let address = if private {
        "mul.wide.u32 %offset,%r0,4; add.u64 %address,%address,%offset;"
    } else {
        ""
    };
    let guard = if single {
        "setp.eq.u32 %p0,%r0,0; mov.u32 %r1,0; @%p0"
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
 mov.b32 %r2,{source}; {address}
 {guard} atom.global.{operation} %r1,[%address],%r2;
 setp.eq.u32 %p0,%r0,0; setp.eq.u32 %p1,%r1,{source};
 and.pred %p2,%p0,%p1; @%p2 trap; ret;
}}
"#
    );
    let dir = std::env::temp_dir().join(format!(
        "ptxv-sass-returned-word-{}-{integer}-{private}-{single}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let ptx = dir.join("kernel.ptx");
    let cubin = dir.join("kernel.cubin");
    std::fs::write(&ptx, text).unwrap();
    ptxv_sass::ptxas(&ptx, &cubin, "sm_103a").unwrap();
    let kernel = Arc::new(ptxv_sass::load_cubin(&cubin, "k").unwrap());
    assert_eq!(kernel.front, ptxv_ir::Front::Sass);
    let count = if private { 3 } else { 1 };
    let block = if single { 1 } else { 3 };
    let manifest=serde_json::from_value(json!({"name":"sass-returned-word-coverage","description":"actual compiled returned-word lane claims","entry":"k","grid":[1,1,1],"block":[block,1,1],"outputs":[],"buffers":[{"name":"x","dtype":"u32","shape":[count],"init":"zeros"}],"params":[{"kind":"ptr","buffer":"x"}],"variants":[]})).unwrap();
    let case = ptxv_case::Case {
        dir: PathBuf::new(),
        m: manifest,
    };
    let launch = Arc::new(crate::build_launch(&case, &kernel).unwrap());
    Machine::new(kernel, launch)
}

fn prepared(mut input: Machine) -> (Machine, u32) {
    let pc = input
        .k
        .code
        .iter()
        .position(|inst| matches!(inst.op, Op::Atom { .. }))
        .unwrap() as u32;
    let actors = (0..input.threads.len()).collect::<Vec<_>>();
    for before in 0..pc {
        input.exec_one(&actors, before).unwrap();
    }
    (input, pc)
}

#[test]
fn actual_sass_f32_certificate_cannot_erase_missing_lane_coverage() {
    let input = machine(false, false, false);
    let certificate = crate::wellsync::certify(input.clone());
    eprintln!(
        "SASS F32 coverage certificate verdict={:?} reasons={:?}",
        certificate
            .verdict
            .as_ref()
            .map(|verdict| verdict.classes()),
        certificate.reasons
    );
    assert!(
        certificate
            .verdict
            .as_ref()
            .is_none_or(|verdict| verdict.classes().iter().any(|kind| kind == "unmodeled")),
        "SASS F32 returned-word certificate must refuse missing lane coverage: {:?}",
        certificate
            .verdict
            .as_ref()
            .map(|verdict| verdict.classes())
    );
    for reverse in [false, true] {
        let mut machine = input.clone();
        machine.atom_lanes_rev = reverse;
        for (name, result) in [
            ("stateful", crate::explore::explore(machine.clone(), 10000)),
            ("DPOR", crate::dpor::explore_dpor(machine, 10000)),
        ] {
            eprintln!(
                "SASS F32 coverage reverse={reverse} {name}: complete={} states={} classes={:?}",
                result.complete,
                result.states,
                result.classes()
            );
            assert!(result.complete);
            assert_eq!(result.classes(), ["unmodeled"]);
        }
    }
    let (mut witness, pc) = prepared(input);
    assert_eq!(
        (0..3)
            .filter(|&actor| witness.prd(actor, witness.k.code[pc as usize].guard))
            .count(),
        3
    );
    for actor in [1, 0, 2] {
        witness.exec_one(&[actor], pc).unwrap();
    }
    let (_, findings, complete) = crate::explore::run_once(witness);
    assert!(complete);
    assert_eq!(
        findings
            .iter()
            .map(|finding| finding.kind.as_str())
            .collect::<Vec<_>>(),
        ["trap"]
    );
}

#[test]
fn actual_sass_private_single_and_aggregate_words_keep_their_certificates() {
    for (integer, private, single) in [
        (true, false, false),
        (false, true, false),
        (false, false, true),
    ] {
        let input = machine(integer, private, single);
        let (prepared, pc) = prepared(input.clone());
        let active = (0..prepared.threads.len())
            .filter(|&actor| prepared.prd(actor, prepared.k.code[pc as usize].guard))
            .count();
        assert_eq!(active, if private { 3 } else { 1 });
        let certificate = crate::wellsync::certify(input.clone());
        eprintln!(
            "SASS positive integer={integer} private={private} single={single}: active={active} certificate={:?} reasons={:?}",
            certificate
                .verdict
                .as_ref()
                .map(|verdict| verdict.classes()),
            certificate.reasons
        );
        assert!(
            certificate
                .verdict
                .as_ref()
                .is_some_and(|verdict| verdict.classes() == ["clean"])
        );
        for result in [
            crate::explore::explore(input.clone(), 10000),
            crate::dpor::explore_dpor(input, 10000),
        ] {
            assert!(result.complete);
            assert_eq!(result.classes(), ["clean"]);
        }
    }
}

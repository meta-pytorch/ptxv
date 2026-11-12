/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_ir::*;
use ptxv_sim::build_launch;
use ptxv_sim::machine::Machine;

#[test]
fn ieee_idioms_validate_fast_dataflow_and_complete_slow_helpers() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::temp_dir().join(format!("ptxv-ieee-helpers-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let case = Case {
        dir: Default::default(),
        m: serde_json::from_value(serde_json::json!({
            "name":"ieee-helpers", "description":"complete division and reciprocal contracts",
            "entry":"k", "grid":[1,1,1], "block":[1,1,1], "outputs":["out"],
            "params":[{"kind":"ptr","buffer":"x"},{"kind":"ptr","buffer":"out"}],
            "buffers":[{"name":"x","dtype":"f32","shape":[2],"init":"zeros","values":[[0,1],[1,2]]},
                {"name":"out","dtype":"f32","shape":[1],"init":"zeros"}], "variants":[]
        }))
        .unwrap(),
    };
    for kind in ["div", "rcp"] {
        let cubin = dir.join(format!("{kind}.cubin"));
        ptxv_sass::ptxas(
            &root.join(format!("fixtures/e_ieee_helpers/{kind}.ptx")),
            &cubin,
            "sm_103a",
        )
        .unwrap();
        let k = ptxv_sass::load_cubin(&cubin, "k").unwrap();
        let close = k
            .code
            .iter()
            .position(|i| matches!(i.op, Op::Bsync { .. }))
            .unwrap();
        let helper = k
            .labels
            .iter()
            .find(|(_, n)| n.contains("slowpath"))
            .unwrap()
            .0;
        let recognizes = |k: Kernel| {
            let launch = build_launch(&case, &k).unwrap();
            let m = Machine::new(Arc::new(k), Arc::new(launch));
            if kind == "div" {
                ptxv_sim::seg::divrn_at(&m, close as u32).is_some()
            } else {
                ptxv_sim::seg::rcprn_at(&m, close as u32).is_some()
            }
        };
        assert!(
            recognizes(k.clone()),
            "{kind}: unmodified compiler idiom refused"
        );
        for pc in 0..k.code.len() {
            if let Op::Ffma { a, b, c, neg, .. } = &k.code[pc].op {
                for operand in 0..4 {
                    let mut mutant = k.clone();
                    let Op::Ffma {
                        a: ma,
                        b: mb,
                        c: mc,
                        neg: mn,
                        ..
                    } = &mut mutant.code[pc].op
                    else {
                        unreachable!()
                    };
                    match operand {
                        0 => {
                            *ma = if *a == Val::Imm(0) {
                                Val::Imm(1)
                            } else {
                                Val::Imm(0)
                            }
                        }
                        1 => {
                            *mb = if *b == Val::Imm(0) {
                                Val::Imm(1)
                            } else {
                                Val::Imm(0)
                            }
                        }
                        2 => {
                            *mc = Some(if *c == Some(Val::Imm(0)) {
                                Val::Imm(1)
                            } else {
                                Val::Imm(0)
                            })
                        }
                        _ => mn[0] = !neg[0],
                    }
                    assert!(
                        !recognizes(mutant),
                        "{kind}: changed FMA at pc {pc}, operand {operand}, helper={helper}"
                    );
                }
            }
        }
        let mut mutant = k.clone();
        let pc = (0..helper)
            .find(|&p| matches!(k.code[p].op, Op::Mov { .. }))
            .unwrap_or(0);
        mutant.code[pc].op = Op::Bra {
            target: helper + 1,
            any: false,
            tid0_falls: false,
            all: false,
            cond: None,
            div: None,
        };
        assert!(
            !recognizes(mutant),
            "{kind}: external helper entry accepted"
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn captured_ieee_idioms_preserve_their_verified_bindings() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (name, division, reciprocal) in [
        ("c_rmsnorm_torch_f32", true, false),
        ("c_layernorm_torch_f32", true, true),
        ("r4_softmax_torch", true, false),
        ("c_silu", true, false),
        ("c_cve_sdpa_f32", false, true),
        ("c_cve_silumul_bf16_eager", true, false),
        ("r4_sdpa_flash_torch", false, true),
    ] {
        if ptxv_case::skip_missing_vendor_files(&root.join("fixtures"), &[name]) {
            continue;
        }
        let mut case = Case::load(root.join("fixtures").join(name)).unwrap();
        let k = ptxv_sass::load_cubin(
            &case.dir.join(case.m.cubin.as_ref().unwrap()),
            &case.m.entry,
        )
        .unwrap();
        // These helper contracts belong to the first launch of captured multi-launch cases.
        case.m.launches.clear();
        let launch = build_launch(&case, &k).unwrap();
        let m = Machine::new(Arc::new(k), Arc::new(launch));
        for (required, reciprocal) in [(division, false), (reciprocal, true)] {
            if !required {
                continue;
            }
            assert!(
                m.k.code.iter().enumerate().any(|(pc, i)| {
                    matches!(i.op, Op::Bsync { .. })
                        && if reciprocal {
                            ptxv_sim::seg::rcprn_at(&m, pc as u32).is_some()
                        } else {
                            ptxv_sim::seg::divrn_at(&m, pc as u32).is_some()
                        }
                }),
                "{name}: no verified {} idiom",
                if reciprocal { "reciprocal" } else { "division" }
            );
        }
    }
}

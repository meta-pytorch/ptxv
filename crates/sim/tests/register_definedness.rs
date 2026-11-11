/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use ptxv_case::Case;
use ptxv_ir::File;
use ptxv_ir::Op;
use ptxv_ir::Reg;
use ptxv_ir::Val;
use ptxv_sim::build_launch;
use ptxv_sim::explore::run_once;
use ptxv_sim::machine::Machine;

#[test]
fn restored_uniform_predicates_keep_undefined_source_bits() {
    let source = r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 output) {
    .reg .b64 %p;
    .reg .b32 %r;
    ld.param.u64 %p, [output];
    mov.b32 %r, 0; // restore
    mov.b32 %r, 0; // save
    st.global.b32 [%p], %r;
    ret;
}
"#;
    let base = ptxv_ptx::parse_kernel(source, Some("k")).unwrap();
    let case = Case {
        dir: Default::default(),
        m: serde_json::from_value(serde_json::json!({
            "name":"predicate-restore", "description":"uniform predicate source definedness",
            "entry":"k", "grid":[1,1,1], "block":[1,1,1], "outputs":["out"],
            "params":[{"kind":"ptr", "buffer":"out"}],
            "buffers":[{"name":"out", "dtype":"u32", "shape":[1], "init":"zeros"}],
            "variants":[]
        }))
        .unwrap(),
    };
    for shift in [0, 8, 16, 24] {
        for defined in [false, true] {
            let mut k = base.clone();
            let Op::Mov { d, .. } = k.code[2].op else {
                panic!("missing placeholder")
            };
            k.code[1].op = Op::R2p {
                a: Val::Reg(
                    Reg {
                        file: File::UR,
                        idx: if defined { 255 } else { 78 },
                    },
                    1,
                ),
                mask: 1,
                shift,
                uniform: true,
            };
            k.code[2].op = Op::P2r {
                d,
                a: Val::Imm(0),
                mask: 1,
                uniform: true,
                shift: 0,
            };
            let launch = build_launch(&case, &k).unwrap();
            let (m, findings, done) = run_once(Machine::new(Arc::new(k), Arc::new(launch)));
            assert!(done, "{findings:?}");
            assert_eq!(
                findings
                    .iter()
                    .any(|f| f.sig.starts_with("uninit-gmem-exit:")),
                !defined,
                "shift={shift} defined={defined}: {findings:?}"
            );
            if defined {
                assert_eq!(m.outputs(), [0; 4]);
            }
        }
    }
}

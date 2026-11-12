/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use ptxv_case::Case;
use ptxv_ir::File;
use ptxv_ir::IntOp;
use ptxv_ir::Op;
use ptxv_ir::Pred;
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

/// MISS 97, S-INIT: a PLOP3 `.SIGN` input (a register's bit 31; ptxas's add.sat.s32) that is
/// undefined makes the predicate undefined, so a select by it stores undefined bits; an
/// input with bit 31 defined (URZ) does not. Before, a `.SIGN` input counted as defined
/// (`Machine::pred_undef` took P / UP registers only) and the store was clean.
#[test]
fn plop3_sign_input_with_undefined_bit_31_is_undefined() {
    let source = r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 output) {
    .reg .b64 %p;
    .reg .b32 %r;
    .reg .pred %q;
    ld.param.u64 %p, [output];
    setp.ne.u32 %q, %r, 0; // PLOP3 placeholder
    selp.b32 %r, 1, 2, %q;
    st.global.b32 [%p], %r;
    ret;
}
"#;
    let base = ptxv_ptx::parse_kernel(source, Some("k")).unwrap();
    let case = Case {
        dir: Default::default(),
        m: serde_json::from_value(serde_json::json!({
            "name":"plop3-sign", "description":"PLOP3 .SIGN input definedness",
            "entry":"k", "grid":[1,1,1], "block":[1,1,1], "outputs":["out"],
            "params":[{"kind":"ptr", "buffer":"out"}],
            "buffers":[{"name":"out", "dtype":"u32", "shape":[1], "init":"zeros"}],
            "variants":[]
        }))
        .unwrap(),
    };
    for defined in [false, true] {
        let mut k = base.clone();
        let Op::Setp { d: q, .. } = k.code[1].op else {
            panic!("missing placeholder")
        };
        k.code[1].op = Op::Plop3 {
            d: q,
            a: Pred {
                reg: Some(Reg {
                    file: File::UR,
                    idx: if defined { 255 } else { 78 },
                }),
                neg: false,
            },
            b: Pred::TRUE,
            c: Pred::TRUE,
            lut: 0xf0,
            d2: None,
        };
        let launch = build_launch(&case, &k).unwrap();
        let (_, findings, done) = run_once(Machine::new(Arc::new(k), Arc::new(launch)));
        assert!(done, "{findings:?}");
        assert_eq!(
            findings
                .iter()
                .any(|f| f.sig.starts_with("uninit-gmem-exit:")),
            !defined,
            "defined={defined}: {findings:?}"
        );
    }
}

/// MISS 97, S-INIT: bit 31 of a `.SIGN` input's undefined mask decides the predicate's
/// definedness, whatever the other bits. An input whose bit 31 is undefined and low bits
/// defined (`x & 0x80000000`, x unwritten) makes the select undefined; one whose bit 31 is
/// defined and low bits undefined (`x & 0x7fffffff`) does not. A reader of another bit, or of
/// the whole mask as nonzero, fails one of the two.
#[test]
fn plop3_sign_input_definedness_is_bit_31_of_its_mask() {
    let source = r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 output) {
    .reg .b64 %p;
    .reg .b32 %r;
    .reg .pred %q;
    ld.param.u64 %p, [output];
    mov.b32 %r, 0; // AND placeholder
    setp.ne.u32 %q, %r, 0; // PLOP3 placeholder
    selp.b32 %r, 1, 2, %q;
    st.global.b32 [%p], %r;
    ret;
}
"#;
    let base = ptxv_ptx::parse_kernel(source, Some("k")).unwrap();
    let case = Case {
        dir: Default::default(),
        m: serde_json::from_value(serde_json::json!({
            "name":"plop3-sign-mask", "description":"PLOP3 .SIGN input, partly defined",
            "entry":"k", "grid":[1,1,1], "block":[1,1,1], "outputs":["out"],
            "params":[{"kind":"ptr", "buffer":"out"}],
            "buffers":[{"name":"out", "dtype":"u32", "shape":[1], "init":"zeros"}],
            "variants":[]
        }))
        .unwrap(),
    };
    let ur = |idx| Reg {
        file: File::UR,
        idx,
    };
    for (mask, undefined) in [(0x8000_0000u64, true), (0x7fff_ffff, false)] {
        let mut k = base.clone();
        let Op::Setp { d: q, .. } = k.code[2].op else {
            panic!("missing placeholder")
        };
        // UR5 = UR6 & mask, UR6 never written: the bits the mask keeps are undefined.
        k.code[1].op = Op::Int {
            op: IntOp::And,
            bits: 32,
            d: ur(5),
            a: Val::Reg(ur(6), 1),
            b: Val::Imm(mask),
        };
        k.code[2].op = Op::Plop3 {
            d: q,
            a: Pred {
                reg: Some(ur(5)),
                neg: false,
            },
            b: Pred::TRUE,
            c: Pred::TRUE,
            lut: 0xf0,
            d2: None,
        };
        let launch = build_launch(&case, &k).unwrap();
        let (m, findings, done) = run_once(Machine::new(Arc::new(k), Arc::new(launch)));
        assert!(done, "{findings:?}");
        assert_eq!(
            findings
                .iter()
                .any(|f| f.sig.starts_with("uninit-gmem-exit:")),
            undefined,
            "mask={mask:#x}: {findings:?}"
        );
        if !undefined {
            // Bit 31 is 0: the predicate is false and the select takes 2.
            assert_eq!(m.outputs(), 2u32.to_le_bytes());
        }
    }
}

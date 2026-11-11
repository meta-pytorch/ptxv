/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! SASS forms of the NVLS / CuTe dist kernels (docs/nvls.md): LDGMC address forms, the
//! grid-barrier `ATOMG.INC`, `REDG.ADD.S32`, and ptxas's `UP2UR.B3` around a fence.

use ptxv_ir::Op;

fn one(l: &str) -> Op {
    ptxv_sass::lower_text(l, None).unwrap_or_else(|e| panic!("{l}: {e}"))
}

#[test]
fn ldgmc_address_forms() {
    for (l, off) in [
        (
            "LDGMC.E.HPADD.BF16x8.RN.STRONG.SYS R16, [R24.64+URZ+-0xc00]",
            -0xc00,
        ),
        (
            "LDGMC.E.HPADD.BF16x8.RN.STRONG.SYS R16, [R24.64+URZ+0x1400]",
            0x1400,
        ),
        ("LDGMC.E.ADD.F32.RN.STRONG.SYS R4, [R2.64+URZ]", 0),
    ] {
        match one(l) {
            Op::LdReduce { addr, .. } => assert_eq!(addr.off, off, "{l}"),
            o => panic!("{l}: {o:?}"),
        }
    }
    assert!(ptxv_sass::lower_text("LDGMC.E.ADD.F32.RN.STRONG.GPU R4, [R2.64+URZ]", None).is_err());
}

#[test]
fn atomg_inc_dec_and_redg_add_s32() {
    use ptxv_ir::AtomOp;
    for (l, want) in [
        (
            "ATOMG.E.INC.STRONG.GPU PT, R9, desc[UR12][R4.64], R2",
            AtomOp::Inc,
        ),
        (
            "ATOMG.E.DEC.STRONG.GPU PT, R9, desc[UR12][R4.64], R2",
            AtomOp::Dec,
        ),
        (
            "REDG.E.ADD.S32.STRONG.SYS desc[UR12][R6.64], R3",
            AtomOp::Add,
        ),
    ] {
        match one(l) {
            Op::Atom { op, bytes, .. } => {
                assert_eq!(op, want, "{l}");
                assert_eq!(bytes, 4, "{l}");
            }
            o => panic!("{l}: {o:?}"),
        }
    }
    assert!(
        ptxv_sass::lower_text(
            "ATOMG.E.INC.64.STRONG.GPU PT, R9, desc[UR12][R4.64], R2",
            None
        )
        .is_err()
    );
}

#[test]
fn up2ur_byte_forms() {
    match one("UP2UR.B3 UR79, UPR, UR79, 0xff") {
        Op::P2r {
            mask,
            shift,
            uniform,
            ..
        } => {
            assert_eq!((mask, shift, uniform), (0xff, 24, true));
        }
        o => panic!("{o:?}"),
    }
    assert!(ptxv_sass::lower_text("UP2UR.B4 UR79, UPR, UR79, 0xff", None).is_err());
}

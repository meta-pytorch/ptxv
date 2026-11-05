/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! E-SPIN retry shapes from real binaries (spin::retry_shape), on hand-written SASS listings
//! lowered through the SASS front-end. The shapes come from the rung-3a CUTLASS grouped GEMM
//! (fixtures/r3_grouped_mm); the negative cases put an effect on the retry path.

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_ir::Op;
use ptxv_sim::build_launch;
use ptxv_sim::machine::Machine;

fn machine(body: &str) -> Machine {
    let mut dis = String::from("\t.text.k:\n");
    for l in body.lines().map(str::trim).filter(|l| !l.is_empty()) {
        if l.ends_with(':') {
            dis += &format!("{l}\n"); // a label
            continue;
        }
        let (a, ins) = l.split_once(' ').unwrap();
        // The high encoding word carries the control bits (no barriers, stall 15).
        dis += &format!(
            "        /*{a}*/                   {} ;  /* 0x0000000000000000 */\n                                                   /* 0x001fde0000000000 */\n",
            ins.trim()
        );
    }
    let k = ptxv_sass::lower(&dis, "k", &Default::default()).unwrap();
    // Any one-CTA case gives a launch; the listings never touch its buffers.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/e_spin_smem");
    let mut case = Case::load(root).unwrap();
    case.m.params.clear();
    let launch = build_launch(&case, &k).unwrap();
    Machine::new(Arc::new(k), Arc::new(launch))
}

/// A try_wait's pc and its `retry_shape`.
type Shape = (usize, Result<Option<(usize, usize)>, String>);

/// retry_shape of every try_wait in the listing (pc, result).
fn shapes(m: &Machine) -> Vec<Shape> {
    (0..m.k.code.len())
        .filter(|&w| matches!(m.k.code[w].op, Op::MbarTryWait { .. }))
        .map(|w| (w, ptxv_sim::spin::retry_shape(m, w)))
        .collect()
}

const PRE: &str = "
    0000 S2R R2, SR_TID.X
    0010 MOV R7, 0x0
    0020 UMOV UR5, 0x400
";

/// Shape A: wait, exit on success, retry by an unconditional back-branch to a prefix that
/// recomputes the parity and the address from unchanged registers.
#[test]
fn wait_then_exit_with_recomputing_back_branch() {
    let m = machine(&format!(
        "{PRE}
        .L_H:
        0030 SHF.L.U32 R3, R2, 0x1f, RZ
        0040 MOV R0, UR5
        0050 SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R0+URZ], R3
        0060 @!P0 NANOSLEEP.SYNCS 0x989680
        0070 @!P0 SYNCS.PHASECHK.TRANS64 P0, [R0+URZ], R3
        0080 @P0 EXIT
        0090 BRA `(.L_H)
        00a0 EXIT"
    ));
    let s = shapes(&m);
    assert_eq!(s.len(), 2);
    // Both polls retry through the prefix at index 3 (0x30) up to the TRYWAIT (index 5).
    for (w, r) in s {
        assert_eq!(r, Ok(Some((3, 5))), "try_wait at {w}");
    }
    assert!(ptxv_sim::spin::check(&m).is_empty());
}

/// Shape B: the retry branch goes back to a prefix that re-materializes the poll address.
#[test]
fn retry_rematerializes_address() {
    let m = machine(&format!(
        "{PRE}
        .L_H:
        0030 MOV R2, UR5
        0040 SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R2+URZ+0x40], R7
        0050 @!P0 NANOSLEEP.SYNCS 0x989680
        0060 @!P0 SYNCS.PHASECHK.TRANS64 P0, [R2+URZ+0x40], R7
        0070 @!P0 BRA `(.L_H)
        0080 BRA `(.L_C)
        .L_C:
        0090 EXIT"
    ));
    for (w, r) in shapes(&m) {
        assert_eq!(r, Ok(Some((3, 4))), "try_wait at {w}");
    }
    assert!(ptxv_sim::spin::check(&m).is_empty());
}

/// The same shapes with a store on the retry path are not blocking-equivalent.
#[test]
fn retry_with_side_effect_is_unmodeled() {
    let a = machine(&format!(
        "{PRE}
        .L_H:
        0030 SHF.L.U32 R3, R2, 0x1f, RZ
        0040 MOV R0, UR5
        0050 SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R0+URZ], R3
        0060 @P0 EXIT
        0070 STS [UR5+0x80], R3
        0080 BRA `(.L_H)
        0090 EXIT"
    ));
    let b = machine(&format!(
        "{PRE}
        .L_H:
        0030 STS [UR5+0x80], R7
        0040 SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR5+0x40], R7
        0050 @!P0 BRA `(.L_H)
        0060 EXIT"
    ));
    for m in [a, b] {
        let s = shapes(&m);
        assert!(s.iter().all(|(_, r)| r.is_err()), "{s:?}");
        assert!(
            ptxv_sim::spin::check(&m)
                .iter()
                .any(|f| f.kind == "unmodeled")
        );
    }
}

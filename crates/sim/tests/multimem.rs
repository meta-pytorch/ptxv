/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Replays recorded GB300 `multimem.ld_reduce` results (probes/multimem/results, 4 GPUs,
//! one NVLink domain, 2026-11-06) through the simulator's reduction (N-MULTIMEM,
//! N-MULTIMEM-ROUND16).

use std::path::PathBuf;

use ptxv_ir::McOp;
use ptxv_ir::McTy;
use ptxv_sim::multimem::reduce;

fn dir(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../probes/multimem/results")
        .join(sub)
}

fn read(sub: &str, name: &str) -> Vec<u8> {
    std::fs::read(dir(sub).join(name)).unwrap_or_else(|e| panic!("{sub}/{name}: {e}"))
}

struct Stats {
    accesses: usize,
    exact: usize,
    inexact: usize,
    lanes_off: usize,
    unmodeled: usize,
}

/// Reduce every `width`-byte access of the sample and compare with GB300.
fn replay(test: &str, op: McOp, ty: McTy, width: usize) -> Stats {
    let ins: Vec<Vec<u8>> = (0..4)
        .map(|m| read("ldr", &format!("{test}.in{m}.bin")))
        .collect();
    let out = read("ldr", &format!("{test}.out.bin"));
    let mut s = Stats {
        accesses: 0,
        exact: 0,
        inexact: 0,
        lanes_off: 0,
        unmodeled: 0,
    };
    let lw = ty.lane_bytes();
    for a in 0..out.len() / width {
        let r = a * width..(a + 1) * width;
        let members: Vec<Vec<u8>> = ins.iter().map(|m| m[r.clone()].to_vec()).collect();
        s.accesses += 1;
        match reduce(op, ty, &members) {
            Err(_) => s.unmodeled += 1,
            Ok(red) => {
                let gpu = &out[r.clone()];
                if red.bytes == gpu {
                    s.exact += 1;
                } else {
                    assert!(
                        red.inexact,
                        "{test} access {a}: mismatch on a lane the model calls exact"
                    );
                    // Every differing lane is the other neighbour (one unit in the last place).
                    for l in 0..width / lw {
                        let g = &gpu[l * lw..(l + 1) * lw];
                        let m = &red.bytes[l * lw..(l + 1) * lw];
                        let v = |b: &[u8]| b.iter().rev().fold(0u64, |x, &y| x << 8 | y as u64);
                        if g != m {
                            assert_eq!(v(g).abs_diff(v(m)), 1, "{test} access {a} lane {l}");
                            s.lanes_off += 1;
                        }
                    }
                }
                s.inexact += red.inexact as usize;
            }
        }
    }
    s
}

#[test]
fn integer_and_minmax_reductions_are_bitwise() {
    for (t, op, ty, w) in [
        ("u32add", McOp::Add, McTy::U32, 4),
        ("u64add", McOp::Add, McTy::U64, 8),
        ("s32min", McOp::Min, McTy::S32, 4),
        ("u32max", McOp::Max, McTy::U32, 4),
        ("xor", McOp::Xor, McTy::U32, 4),
        ("bf16x2min", McOp::Min, McTy::Bf16, 4),
        ("bf16x2max", McOp::Max, McTy::Bf16, 4),
        ("f16x2max", McOp::Max, McTy::F16, 4),
    ] {
        let s = replay(t, op, ty, w);
        assert_eq!(s.exact + s.unmodeled, s.accesses, "{t}");
        assert!(
            s.unmodeled * 1000 <= s.accesses,
            "{t}: {} unmodeled",
            s.unmodeled
        );
    }
}

#[test]
fn f32_sum_is_bitwise_except_flagged_lanes() {
    for (t, w) in [("f32", 4), ("f32w", 4), ("f32x4", 16)] {
        let s = replay(t, McOp::Add, McTy::F32, w);
        // NaN pairs are not modeled; everything else matches or is flagged inexact.
        assert!(
            s.unmodeled * 50 <= s.accesses,
            "{t}: {} unmodeled",
            s.unmodeled
        );
        assert!(
            s.exact * 100 >= 98 * s.accesses,
            "{t}: {} of {}",
            s.exact,
            s.accesses
        );
    }
}

#[test]
fn half_sums_are_one_of_the_two_neighbours() {
    // N-MULTIMEM-ROUND16: the calibrated rule; differing lanes are the other neighbour.
    for (t, ty) in [
        ("bf16x8a", McTy::Bf16),
        ("bf16x8", McTy::Bf16),
        ("f16x8a", McTy::F16),
        ("f16x8", McTy::F16),
    ] {
        let s = replay(t, McOp::Add, ty, 16);
        let lanes = (s.accesses - s.unmodeled) * 8;
        let floor = if ty == McTy::Bf16 { 99 } else { 95 };
        assert!(
            (lanes - s.lanes_off) * 100 >= floor * lanes,
            "{t}: {} of {lanes} lanes differ",
            s.lanes_off
        );
    }
}

#[test]
fn unprobed_types_are_not_modeled() {
    for (t, ty, w) in [
        ("f64", McTy::F64, 8),
        ("e4m3x16a", McTy::E4m3, 16),
        ("e5m2x16a", McTy::E5m2, 16),
    ] {
        let s = replay(t, McOp::Add, ty, w);
        assert_eq!(s.unmodeled, s.accesses, "{t}");
    }
}

#[test]
fn half_rounding_direction_follows_the_group_dither_on_sweeps() {
    // probes/multimem `mm sr3`: one lane at 1 + f/8 ulp, another lane's mantissa swept.
    for (t, ty) in [
        ("bf16x8a", McTy::Bf16),
        ("bf16x8", McTy::Bf16),
        ("f16x8a", McTy::F16),
    ] {
        let ins: Vec<Vec<u8>> = (0..4)
            .map(|m| read("sr3", &format!("{t}.in{m}.bin")))
            .collect();
        let out = read("sr3", &format!("{t}.r0.bin"));
        for a in 0..out.len() / 16 {
            let r = a * 16..(a + 1) * 16;
            let members: Vec<Vec<u8>> = ins.iter().map(|m| m[r.clone()].to_vec()).collect();
            let red = reduce(McOp::Add, ty, &members).unwrap();
            assert_eq!(red.bytes, &out[r], "{t} row {a}");
        }
    }
}

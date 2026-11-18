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
    /// Of `lanes_off`, the lanes the calibrated f16 / bf16 rule rounded (each the other rounding
    /// of its sum, on the side the model's mark gives).
    lanes_other: usize,
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
        lanes_other: 0,
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
                            // MISS 162: a lane the calibrated rule rounded differs by the other
                            // rounding of its sum, on the side the model's mark gives.
                            if let Some(&c) = red.calibrated.get(l) {
                                let next = c.unwrap_or_else(|| {
                                    panic!("{test} access {a} lane {l}: an exact lane differs")
                                });
                                let other = if next { v(m) + 1 } else { v(m) - 1 };
                                assert_eq!(v(g), other, "{test} access {a} lane {l}");
                                s.lanes_other += 1;
                            }
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
fn f32_sums_are_bitwise() {
    for (t, w) in [("f32", 4), ("f32w", 4), ("f32x4", 16)] {
        let s = replay(t, McOp::Add, McTy::F32, w);
        // NaN pairs are not modeled; every other access matches GB300 bitwise (MISS 225: until
        // then 98% did, the rest flagged inexact).
        assert!(
            s.unmodeled * 50 <= s.accesses,
            "{t}: {} unmodeled",
            s.unmodeled
        );
        assert_eq!(s.exact + s.unmodeled, s.accesses, "{t}");
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
        // Every differing lane is a calibrated one, and GB300 gave the other rounding of its sum.
        assert_eq!(s.lanes_other, s.lanes_off, "{t}");
        eprintln!(
            "{t}: {} of {lanes} lanes are the other rounding",
            s.lanes_other
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

/// Replays the GB300 `mm ldr` sweep directory `d` of `n` members (`<test>.in<i>.bin`,
/// `<test>.out.bin`) for `tests` (name, type, lane bytes, access bytes): every lane equals
/// GB300's, except that a lane the calibrated rule rounds may be the other rounding of its sum
/// (MISS 162). Returns the number of lanes compared.
fn replay_sweep(d: &std::path::Path, n: usize, tests: &[(&str, McTy, usize, usize)]) -> usize {
    let mut lanes = 0;
    for &(t, ty, lw, width) in tests {
        let Ok(out) = std::fs::read(d.join(format!("{t}.out.bin"))) else {
            continue;
        };
        let ins: Vec<Vec<u8>> = (0..n)
            .map(|i| std::fs::read(d.join(format!("{t}.in{i}.bin"))).unwrap())
            .collect();
        for a in 0..out.len() / width {
            let r = a * width..(a + 1) * width;
            let members: Vec<Vec<u8>> = ins.iter().map(|m| m[r.clone()].to_vec()).collect();
            let red = reduce(McOp::Add, ty, &members).unwrap();
            for l in 0..width / lw {
                let v = |b: &[u8]| b.iter().rev().fold(0u64, |x, &y| x << 8 | y as u64);
                let (g, m) = (
                    v(&out[r.start + l * lw..][..lw]),
                    v(&red.bytes[l * lw..][..lw]),
                );
                lanes += 1;
                if g == m {
                    continue;
                }
                let next = red.calibrated.get(l).copied().flatten();
                let other = next.map(|n| if n { m + 1 } else { m - 1 });
                assert_eq!(
                    Some(g),
                    other,
                    "{} {t} vector {a} lane {l}: GB300 {g:#x}, model {m:#x}",
                    d.display()
                );
            }
        }
    }
    lanes
}

fn sweeps() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../probes/multimem_add16_rules/sweep")
}

/// MISS 208 (probes/multimem_add16_rules/sweep, GB300 2026-11-12, `mm ldr` with MM_IN; the
/// generators gen1.py / gen2.py hold the lane kinds): a sum of zeros is +0 (f16, bf16; 2 and 4
/// members), and an f16 / bf16 sum whose leading bit lies below 2^(16 floor(eB / 16) - 23 -
/// bias), eB the largest member's biased exponent, is +0. s1: (B, -B, R, 0) for every eB with R
/// around that bound, (B, -B, R1, R2) with R2 below it and the sum above it (kept), 2-member
/// (B, -R) lanes whose group's other lanes hold zero mantissas (a zero dither), zero lanes; s2:
/// (B, -B, R1, R1 2^-6) with R1 above the bound and R1 2^-6 below it (kept).
#[test]
fn half_sums_follow_the_flush_sweeps() {
    let tests = [
        ("bf16x8", McTy::Bf16, 2, 16),
        ("bf16x8a", McTy::Bf16, 2, 16),
        ("f16x8", McTy::F16, 2, 16),
        ("f16x8a", McTy::F16, 2, 16),
    ];
    let lanes: usize = [("s1", 4), ("s1", 2), ("s2", 4)]
        .iter()
        .map(|&(s, n)| replay_sweep(&sweeps().join(s).join(format!("m{n}")), n, &tests))
        .sum();
    assert!(lanes > 20000, "{lanes} lanes");
}

/// MISS 225: f32 sums (probes/multimem_add16_rules/sweep s1, s2 and probes/multimem_f32_window,
/// GB300 2026-11-12, `mm ldr`): s1's zero lanes (every member -0 gives +0; 2 and 4 members),
/// s2's cancellations (B, -B, R, 0) with R 16 to 72 binades below B, eB 40 to 230 (3450 lanes
/// per form), and the probe's lanes: every lane of `mm ldr`'s random fill (MM_SEED 1, 2, 3,
/// 2359296 lanes) where the former window (63 bits below the largest member's last mantissa
/// bit) gave another value, with their 16-byte groups. Every lane equals GB300's: members
/// truncated to multiples of 2^(16 floor(eB / 16) - 207), +0 below 2^(16 floor(eB / 16) - 198).
#[test]
fn f32_sums_follow_the_window_sweeps() {
    let tests = [("f32x4", McTy::F32, 4, 16), ("f32", McTy::F32, 4, 4)];
    let mut lanes: usize = [("s1", 4), ("s1", 2), ("s2", 4)]
        .iter()
        .map(|&(s, n)| replay_sweep(&sweeps().join(s).join(format!("m{n}")), n, &tests))
        .sum();
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../probes/multimem_f32_window");
    let rows = std::fs::read_to_string(p.join("expected/lanes.csv")).unwrap();
    for n in [4usize, 2] {
        let ins: Vec<Vec<u8>> = (0..n)
            .map(|d| std::fs::read(p.join(format!("data/m{n}.in{d}.bin"))).unwrap())
            .collect();
        let out = std::fs::read(p.join(format!("expected/m{n}.out.bin"))).unwrap();
        // Each test's lanes are one section of the data, in this order; pad lanes (+0 members,
        // not measured) are skipped.
        let mut base = 0;
        for t in ["f32", "f32w", "f32x4"] {
            let mine: Vec<Vec<&str>> = rows
                .lines()
                .skip(1)
                .map(|l| l.split(',').collect::<Vec<_>>())
                .filter(|c| c[0] == n.to_string() && c[1] == t)
                .collect();
            for c in &mine {
                if c[5] == "pad" {
                    continue;
                }
                let k = base + c[2].parse::<usize>().unwrap();
                let members: Vec<Vec<u8>> =
                    ins.iter().map(|m| m[4 * k..4 * k + 4].to_vec()).collect();
                let red = reduce(McOp::Add, McTy::F32, &members).unwrap();
                assert_eq!(
                    red.bytes,
                    &out[4 * k..4 * k + 4],
                    "probe m{n} {t} lane {} ({} lane {})",
                    c[2],
                    c[3],
                    c[4]
                );
                lanes += 1;
            }
            base += mine.len();
        }
    }
    assert!(lanes > 9000, "{lanes} lanes");
}

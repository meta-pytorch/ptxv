/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv num` makes no error claim for outputs whose rounding history it dropped
//! (N-NUM-TRACK). Each fixture's expected/out.bin is its GB300 output (equal to the simulator
//! bitwise, `ptxv hw`), and every element differs from the fp64 value of the computation. So
//! for every element num analyzes, its error must equal |GB300 - fp64|; any other element must
//! be reported untracked.
//! - c_num_sumsq_f32: row sums of squares (FFMA chain; once max_err = 0, Thm 1 exact, no
//!   warning).
//! - c_num_sum_f32: row sums (FADD chain, tracked: the error is checked against fp64).
//! - c_num_max_f32: row maxima of x + y (exact `Max` nodes: tracked, checked against fp64).
//! - c_num_amax_f32: row maxima of |x + y| (FMNMX of |x| operands, untracked).
//! - c_num_bf16rt_f32: f32(bf16(x + y)) + z (the bf16 round trip dominates the error).
//! - c_num_sum_bf16: row sums stored as bf16 (read per 16-bit element, tracked).
//!
//! The invariant catches a taint drop at FFMA, FMNMX (c_num_amax_f32) or the PTX
//! cvt.f32.bf16 (MISS 36 mutants): the dropped value would become an exact constant and the claimed error too small.

use std::path::Path;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn f32s(p: &Path) -> Vec<f64> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect()
}

/// Runs `ptxv num`; its stdout and the (front, CSV text) pairs it wrote.
fn num(case: &str) -> (String, Vec<(&'static str, String)>) {
    let out = std::env::temp_dir().join(format!("ptxv-test-num-{case}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let r = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["num", root().join("fixtures").join(case).to_str().unwrap()])
        .arg("--out")
        .arg(&out)
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&r.stdout),
        String::from_utf8_lossy(&r.stderr)
    );
    assert!(r.status.success(), "{text}");
    let csvs = ["ptx", "sass"]
        .into_iter()
        .map(|f| {
            (
                f,
                std::fs::read_to_string(out.join(format!("good.{f}.num.csv"))).unwrap(),
            )
        })
        .collect();
    let _ = std::fs::remove_dir_all(&out);
    (text, csvs)
}

/// The fixture's buffer `name` (f32 elements).
fn buf(case: &str, name: &str) -> Vec<f64> {
    f32s(&root().join("fixtures").join(case).join(name))
}

/// Checks the invariant on an f32 output; returns the number of analyzed (front, element)
/// pairs.
fn check(case: &str, fp64: &[f64]) -> usize {
    check_hw(case, &buf(case, "expected/out.bin"), fp64)
}

/// The invariant against GB300 output values `hw`.
fn check_hw(case: &str, hw: &[f64], fp64: &[f64]) -> usize {
    assert_eq!(hw.len(), fp64.len());
    // The GB300 output carries rounding error in every element.
    assert!(hw.iter().zip(fp64).all(|(h, e)| h != e), "{case}");
    let (text, csvs) = num(case);
    let mut analyzed = 0;
    for (front, csv) in csvs {
        let mut n = 0;
        for line in csv.lines().skip(1) {
            let c: Vec<&str> = line.split(',').collect();
            let i: usize = c[0].parse().unwrap();
            n += 1;
            if c[11] == "true" {
                continue;
            }
            analyzed += 1;
            let abs_err: f64 = c[3].parse().unwrap();
            let want = (hw[i] - fp64[i]).abs();
            // fp64 itself rounds (up to 64 terms): agreement far below f32's ulp.
            assert!(
                (abs_err - want).abs() <= 1e-12 * fp64[i].abs().max(1.0),
                "{case} {front} element {i}: num claims |err| {abs_err}, GB300 - fp64 is {want}\n{text}"
            );
        }
        assert_eq!(n, hw.len(), "{case} {front}");
    }
    analyzed
}

fn rows(x: &[f64], k: usize, f: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    x.chunks_exact(k).map(f).collect()
}

#[test]
fn ffma_chain_is_not_analyzed() {
    let x = buf("c_num_sumsq_f32", "data/x.bin");
    let fp64 = rows(&x, 64, |r| r.iter().map(|v| v * v).sum());
    assert_eq!(check("c_num_sumsq_f32", &fp64), 0);
}

#[test]
fn fadd_chain_error_matches_gb300_minus_fp64() {
    let x = buf("c_num_sum_f32", "data/x.bin");
    let fp64 = rows(&x, 64, |r| r.iter().sum());
    // Tracked on both fronts: every element is a claim, and each matches.
    assert_eq!(check("c_num_sum_f32", &fp64), 2 * 256);
}

#[test]
fn fmnmx_of_rounded_sums_matches_gb300_minus_fp64() {
    let x = buf("c_num_max_f32", "data/x.bin");
    let y = buf("c_num_max_f32", "data/y.bin");
    let s: Vec<f64> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
    let fp64 = rows(&s, 64, |r| {
        r.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    });
    // Max nodes: every element is a claim on both fronts, and each matches.
    assert_eq!(check("c_num_max_f32", &fp64), 2 * 256);
}

#[test]
fn fmnmx_of_abs_is_not_claimed_exact() {
    let x = buf("c_num_amax_f32", "data/x.bin");
    let y = buf("c_num_amax_f32", "data/y.bin");
    let s: Vec<f64> = x.iter().zip(&y).map(|(a, b)| (a + b).abs()).collect();
    let fp64 = rows(&s, 64, |r| r.iter().copied().fold(0.0, f64::max));
    check("c_num_amax_f32", &fp64);
}

/// A bf16 output is read per 16-bit element; the error includes the final rounding.
#[test]
fn bf16_output_error_matches_gb300_minus_fp64() {
    let x = buf("c_num_sum_bf16", "data/x.bin");
    let fp64 = rows(&x, 64, |r| r.iter().sum());
    let hw: Vec<f64> = std::fs::read(root().join("fixtures/c_num_sum_bf16/expected/out.bin"))
        .unwrap()
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64)
        .collect();
    assert_eq!(check_hw("c_num_sum_bf16", &hw, &fp64), 2 * 256);
}

#[test]
fn bf16_round_trip_is_not_claimed_exact() {
    let [x, y, z] = ["x", "y", "z"].map(|n| buf("c_num_bf16rt_f32", &format!("data/{n}.bin")));
    let fp64: Vec<f64> = (0..x.len()).map(|i| x[i] + y[i] + z[i]).collect();
    check("c_num_bf16rt_f32", &fp64);
}

/// PTX `ld.shared.b16` + `cvt.f32.bf16` of a bf16 input element is the input itself: num's
/// claim on r0_tma_sum (FADD chain of widened bf16 inputs) matches the spec on both fronts.
#[test]
fn ptx_bf16_input_widening_is_tracked() {
    let (text, _) = num("r0_tma_sum");
    for f in ["ptx ", "sass"] {
        let line = text
            .lines()
            .find(|l| l.starts_with(&format!("num good {f}:")))
            .unwrap_or_else(|| panic!("{text}"));
        assert!(
            !line.contains("NOT ANALYZED") && line.contains("spec_match=Some(true)"),
            "{text}"
        );
    }
}

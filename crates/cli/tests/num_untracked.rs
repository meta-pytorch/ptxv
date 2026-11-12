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
//! - c_num_param_f32: roundings of kernel parameters, %tid and a MUFU of a parameter (no
//!   node, not input-derived: once constants taken as exact), next to exact constants.
//! - c_num_branch_f32: a branch and a store decided by a rounded value (the exact
//!   computation takes the other path): no claim at all.
//! - c_num_gather_f32: a gather, a shuffle and a scatter at an index computed from a rounded
//!   value (the exact computation accesses other elements): no claim at all.
//! - c_cat, c_cumsum (captured), c_num_index_f32: index arithmetic from constants ptxas
//!   materializes with HFMA2 and from the u32 division idiom (MUFU.RCP) is not input data:
//!   every element keeps a claim.
//! - c_num_vote_f32 (MISS 65): a branch on a warp ballot of other lanes' input data: no claim.
//! - c_num_match_f32 (MISS 95): a branch on a match of another lane's rounded value: no claim.
//! - c_num_atom_read_f32, c_num_atom_store_f32, c_num_atom_f64_f32, c_num_atom_cas_f32 (MISS
//!   114): atomics other than f32 adds: the values they return or write from a rounding are input
//!   data, an exchange moves the node, an f64 atomic sum deciding a branch is refused.
//! - c_num_ld16_f32, c_num_ldsm_trans_f32, c_num_st16_f32, c_num_tma16_f32, c_num_stsm8_f32 (MISS
//!   115): partial-width and rearranging accesses of words holding a rounding: input data.
//! - c_num_fold_f32, c_num_fold_add_f32, c_num_fold_ctl_f32 (MISS 108): roundings of constants
//!   ptxas folds into immediates: the SASS front claims only what the PTX front claims, with the
//!   same exact value.
//! - c_num_nonfinite_f32 (N-NUM-NONFINITE): overflows (inf, NaN) against finite exact values
//!   carry an infinite error and make max_err infinite; copies of inf / NaN have error 0;
//!   non-finite exact values the f64 expansions do not determine are not analyzed.
//! - c_num_nan2_f32 (N-NUM-NONFINITE): the independent review's overflow probe; every element
//!   analyzed, the even lanes' errors infinite.
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

/// Like `num`, with None for a front whose output carries no claim at all (no CSV).
fn num_opt(case: &str) -> (String, Vec<(&'static str, Option<String>)>) {
    let out = std::env::temp_dir().join(format!("ptxv-test-numo-{case}-{}", std::process::id()));
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
                std::fs::read_to_string(out.join(format!("good.{f}.num.csv"))).ok(),
            )
        })
        .collect();
    let _ = std::fs::remove_dir_all(&out);
    (text, csvs)
}

/// The invariant where some elements may equal their fp64 value: every analyzed element's
/// claimed error is |GB300 - fp64|. Returns the analyzed (front, element) pairs and stdout.
fn check_claims(case: &str, fp64: &[f64]) -> (Vec<(&'static str, usize)>, String) {
    let hw = buf(case, "expected/out.bin");
    assert_eq!(hw.len(), fp64.len());
    let (text, csvs) = num_opt(case);
    let mut analyzed = vec![];
    for (front, csv) in csvs {
        let Some(csv) = csv else { continue };
        for line in csv.lines().skip(1) {
            let c: Vec<&str> = line.split(',').collect();
            let i: usize = c[0].parse().unwrap();
            if c[11] == "true" {
                continue;
            }
            analyzed.push((front, i));
            let abs_err: f64 = c[3].parse().unwrap();
            let want = (hw[i] - fp64[i]).abs();
            assert!(
                (abs_err - want).abs() <= 1e-12 * fp64[i].abs().max(1.0),
                "{case} {front} element {i}: num claims |err| {abs_err}, GB300 - fp64 is {want}\n{text}"
            );
        }
    }
    (analyzed, text)
}

/// A rounding of values that come from no input buffer (f32 kernel parameters a = 0.1f,
/// b = 3, %tid, a MUFU of a parameter) gets no node; num once took its simulated value as an
/// exact constant (max_err 0, Thm 1 exact on both fronts). It is untracked now; the exact
/// constants (a itself, an exact f32(%tid)) stay claims.
#[test]
fn parameter_and_tid_roundings_are_not_claimed_exact() {
    let case = "c_num_param_f32";
    let x = buf(case, "data/x.bin");
    let a = f32::from_bits(0x3dcc_cccd) as f64;
    let b = 3.0;
    let mut fp64 = vec![0.0; 192];
    for t in 0..32 {
        let tf = t as f64;
        fp64[t] = x[t] + a * b;
        fp64[32 + t] = a * b;
        fp64[64 + t] = x[t] + tf * a;
        fp64[96 + t] = x[t] + a.exp2();
        fp64[128 + t] = x[t] + a;
        fp64[160 + t] = x[t] + tf;
    }
    let hw = buf(case, "expected/out.bin");
    // rn(a b) and ex2(a) differ from their exact values (every element of k = 0, 1, 3).
    assert!(
        (0..64).chain(96..128).all(|i| hw[i] != fp64[i]),
        "{case}: GB300 output"
    );
    let (analyzed, text) = check_claims(case, &fp64);
    for f in ["ptx", "sass"] {
        for i in 128..192 {
            assert!(
                analyzed.contains(&(f, i)),
                "{case} {f} element {i}: no claim\n{text}"
            );
        }
    }
}

/// A branch and a store decided by p = rn(x*y) >= 1 and q = rn(u*v) > 1: in the even lanes
/// this run (and GB300) takes the path the exact computation does not. num makes no claim.
#[test]
fn control_flow_on_a_rounded_value_is_not_analyzed() {
    let case = "c_num_branch_f32";
    let [x, y, z, u, v] = ["x", "y", "z", "u", "v"].map(|n| buf(case, &format!("data/{n}.bin")));
    let mut fp64 = vec![0.0; 96];
    for t in 0..32 {
        // A product of two f32 values is exact in f64.
        let p = x[t] * y[t] >= 1.0;
        let q = u[t] * v[t] > 1.0;
        fp64[t] = if p { 1.0 } else { -1.0 };
        fp64[32 + t] = if p { y[t] + z[t] } else { x[t] + z[t] };
        fp64[64 + t] = if q { 1.0 } else { 0.0 };
    }
    let hw = buf(case, "expected/out.bin");
    assert!(
        (0..32)
            .step_by(2)
            .all(|t| hw[t] != fp64[t] && hw[64 + t] != fp64[64 + t]),
        "{case}: GB300 output"
    );
    let (analyzed, text) = check_claims(case, &fp64);
    assert!(analyzed.is_empty(), "{case}: claims {analyzed:?}\n{text}");
    assert!(
        text.contains("control flow or an address depends on input data"),
        "{text}"
    );
}

/// A gather, a shuffle and a scatter at i = trunc(rn(x*y)): in the even lanes x*y rounds up to
/// 1, so this run (and GB300) reads and writes elements the exact computation (i = 0) does
/// not. num makes no claim.
#[test]
fn addresses_from_a_rounded_value_are_not_analyzed() {
    let case = "c_num_gather_f32";
    let [x, y, z] = ["x", "y", "z"].map(|n| buf(case, &format!("data/{n}.bin")));
    let mut fp64 = vec![0.0; 128];
    for t in 0..32 {
        // A product of two f32 values is exact in f64.
        let i = ((x[t] * y[t]).trunc() as usize).min(1);
        fp64[t] = z[(t + i) & 31];
        fp64[32 + t] = z[(t + i) & 31];
        fp64[64 + t + 32 * i] = x[t];
    }
    let hw = buf(case, "expected/out.bin");
    assert!(
        (0..32).step_by(2).all(|t| {
            hw[t] != fp64[t] && hw[32 + t] != fp64[32 + t] && hw[64 + t] != fp64[64 + t]
        }),
        "{case}: GB300 output"
    );
    let (analyzed, text) = check_claims(case, &fp64);
    assert!(analyzed.is_empty(), "{case}: claims {analyzed:?}\n{text}");
    assert!(
        text.contains("control flow or an address depends on input data"),
        "{text}"
    );
}

/// Roundings of values that are not input data, three more ways (N-NUM-TRACK; the independent
/// review's probes, GB300 outputs):
/// - c_num_mufu_imm_f32: MUFU.TANH of an immediate (ptxas keeps it; every SASS operand is an
///   immediate) is a rounding, not a constant: once claimed exact, tanh(0.1f) is 44 ulp off.
///   Elements k = 0..3 are not analyzed on either front. k = 4, ex2.approx of a constant, is not
///   analyzed on the PTX front; ptxas folds it to an immediate, which the SASS front took as a
///   constant: it claims nothing the PTX front does not (MISS 108).
/// - c_num_clobber_f32: rn(a + b) of parameters through an and drops its node: input data.
/// - c_num_atom_old_f32: the values atom.add.f32 returns (rounded partial sums) are input data;
///   the location's sum keeps its claim.
#[test]
fn roundings_through_immediates_bit_operations_and_atomics_are_not_exact() {
    let untracked = |case: &str, keep: &dyn Fn(&str, usize) -> bool| {
        let (text, csvs) = num_opt(case);
        for (front, csv) in csvs {
            let csv = csv.unwrap_or_else(|| panic!("{case} {front}: no CSV\n{text}"));
            for line in csv.lines().skip(1) {
                let c: Vec<&str> = line.split(',').collect();
                let i: usize = c[0].parse().unwrap();
                if !keep(front, i) {
                    assert_eq!(
                        c[11], "true",
                        "{case} {front} element {i}: claimed ({line})\n{text}"
                    );
                }
            }
        }
        text
    };
    untracked("c_num_mufu_imm_f32", &|_, _| false);
    untracked("c_num_clobber_f32", &|_, _| false);
    let text = untracked("c_num_atom_old_f32", &|_, _| false);
    for front in ["ptx", "sass"] {
        let acc = text
            .lines()
            .find(|l| l.starts_with(&format!("num good {front:<4}: output=acc")))
            .unwrap_or_else(|| panic!("{text}"));
        assert!(acc.contains("cor1a_ok=true"), "{acc}");
    }
}

/// N-NUM-UDIV: c_num_udiv_f32: div / rem .u32 by du = 7 and .s32 by ds = -9, which ptxas
/// lowers to I2F.RP, MUFU.RCP, F2I.TRUNC and an integer correction (.s32 on |n| and |d|, then
/// the signs; the remainder's correction partly by a selection), decide load addresses and a
/// store guard. Every quotient and remainder is exact: every element keeps a claim on both
/// fronts, each |GB300 - exact|.
#[test]
fn integer_division_idiom_keeps_its_claims() {
    let case = "c_num_udiv_f32";
    let [x, y] = ["x", "y"].map(|n| buf(case, &format!("data/{n}.bin")));
    let mut fp64 = vec![0.0; 640];
    for t in 0..128 {
        let n = (t as u32).wrapping_mul(0x01fc_07f1).wrapping_add(5);
        let sn = (t as i32 - 64).wrapping_mul(0x00ff_00ff);
        let (q, r) = (n / 7, n % 7);
        // Truncating, as div / rem .s32.
        let (sq, sr) = (sn / -9, sn % -9);
        fp64[t] = x[(q & 127) as usize];
        fp64[128 + t] = x[(r & 127) as usize];
        fp64[256 + t] = x[(sq & 127) as usize];
        fp64[384 + t] = x[(sr & 127) as usize];
        if q & 1 == 1 {
            fp64[512 + t] = x[t] + y[t];
        }
    }
    let (analyzed, text) = check_claims(case, &fp64);
    assert_eq!(analyzed.len(), 2 * fp64.len(), "{case}: {text}");
    // c_num_udiv_data_f32: the same idiom on input data (idx[t] / 7): exact, but the load
    // address depends on input data, so no claim.
    let case = "c_num_udiv_data_f32";
    let x = buf(case, "data/x.bin");
    let idx: Vec<u32> = std::fs::read(root().join("fixtures").join(case).join("data/idx.bin"))
        .unwrap()
        .chunks(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let fp64: Vec<f64> = idx.iter().map(|&n| x[((n / 7) & 127) as usize]).collect();
    let (analyzed, text) = check_claims(case, &fp64);
    assert!(analyzed.is_empty(), "{case}: claims {analyzed:?}\n{text}");
    assert!(text.contains("depends on input data"), "{case}: {text}");
}

/// num makes no claim for `case`, and GB300's output differs from the exact computation's
/// `fp64` at each of `differ` (the exact path differs from the run's).
fn refused_where_paths_differ(case: &str, fp64: &[f64], differ: &[usize]) {
    let hw = buf(case, "expected/out.bin");
    assert_eq!(hw.len(), fp64.len(), "{case}");
    for &i in differ {
        assert!(
            hw[i] != fp64[i],
            "{case} element {i}: GB300 {} = exact",
            hw[i]
        );
    }
    let (analyzed, text) = check_claims(case, fp64);
    assert!(analyzed.is_empty(), "{case}: claims {analyzed:?}\n{text}");
    assert!(
        text.contains("control flow or an address depends on input data or on a rounded value"),
        "{case}: {text}"
    );
}

/// Control decided by roundings of kernel parameters, %tid and constants (N-NUM-TRACK; the
/// independent review's probes, GB300 outputs). Each run takes another path, or reads other
/// elements, than the exact computation; num makes no claim on either front:
/// - c_num_ctl_branch_f32: branches on rn(a + b) >= c (an FADD node) and rn(a d) > 1 (no node);
/// - c_num_ctl_loop_f32: a float loop counter of parameters (3 iterations here, 4 exact);
/// - c_num_ctl_atom_f32: a branch on the sum of 32 atomic adds of 0.1f;
/// - c_num_ctl_guard_f32: a store guard rn(f32(t) 0.7f) >= 7;
/// - c_num_ctl_addr_f32: load addresses floor(rn(f32(t) 0.7f)) and floor(rn(f32(t) - 2^-25));
/// - c_num_ctl_shfl_f32: a shuffle lane floor(rn(f32(t) 0.7f)).
#[test]
fn control_from_rounded_values_is_refused() {
    let f = |b: u32| f32::from_bits(b) as f64;
    // Products and sums of two f32 values are exact in f64.
    let s7 = f(0x3f33_3333);
    let lane7 = |t: usize| (t as f64 * s7).floor() as usize;
    let case = "c_num_ctl_branch_f32";
    let [x, y, z] = ["x", "y", "z"].map(|n| buf(case, &format!("data/{n}.bin")));
    let (a, b, c, d) = (f(0x3dcc_cccd), f(0x3e4c_cccd), f(0x3e99_999a), 10.0);
    assert!(a + b < c && a * d > 1.0);
    let mut fp64 = vec![0.0; 64];
    for t in 0..32 {
        fp64[t] = x[t] + z[t];
        fp64[32 + t] = x[t] + y[t];
    }
    refused_where_paths_differ(case, &fp64, &(0..64).collect::<Vec<_>>());
    let case = "c_num_ctl_loop_f32";
    let x = buf(case, "data/x.bin");
    let (step, lim) = (f(0x3dcc_cccd), f(0x3e99_999a));
    let n = (0..).take_while(|&k| k as f64 * step < lim).count();
    assert_eq!(n, 4);
    let fp64: Vec<f64> = (0..32)
        .map(|t| (0..n).map(|k| x[(t + 32 * k) & 127]).sum())
        .collect();
    refused_where_paths_differ(case, &fp64, &(0..32).collect::<Vec<_>>());
    let case = "c_num_ctl_atom_f32";
    let [x, y, z] = ["x", "y", "z"].map(|n| buf(case, &format!("data/{n}.bin")));
    assert!(32.0 * f(0x3dcc_cccd) >= f(0x404c_cccd));
    let fp64: Vec<f64> = (0..32).map(|t| x[t] + y[t]).collect();
    let _ = z;
    refused_where_paths_differ(case, &fp64, &(0..32).collect::<Vec<_>>());
    let case = "c_num_ctl_guard_f32";
    let x = buf(case, "data/x.bin");
    let fp64: Vec<f64> = (0..32)
        .map(|t| if t as f64 * s7 >= 7.0 { x[t] } else { 0.0 })
        .collect();
    refused_where_paths_differ(case, &fp64, &[10]);
    let case = "c_num_ctl_addr_f32";
    let x = buf(case, "data/x.bin");
    let mut fp64 = vec![0.0; 64];
    for t in 0..32 {
        fp64[t] = x[lane7(t)];
        fp64[32 + t] = x[t.saturating_sub(1)];
    }
    refused_where_paths_differ(case, &fp64, &[10, 20, 30, 33, 63]);
    let case = "c_num_ctl_shfl_f32";
    let x = buf(case, "data/x.bin");
    let fp64: Vec<f64> = (0..32).map(|t| x[lane7(t)]).collect();
    refused_where_paths_differ(case, &fp64, &[10, 20, 30]);
}

/// MISS 65 (warp votes; the independent review's probe, GB300 outputs): in c_num_vote_f32 lanes
/// 16..31 branch on a ballot of lanes 0..15's predicate rn(x y) >= 1 (input data; x y < 1
/// exactly, so the exact ballot is 0 and the exact output x + z, where the run's ballot is
/// 0xffff and its output x + y). Each lane's ballot result carried only its own predicate's
/// taint, so lanes 16..31's branch looked input-independent; a warp-collective result now
/// carries every executing lane's. num makes no claim on either front.
#[test]
fn control_from_a_ballot_of_other_lanes_is_refused() {
    let case = "c_num_vote_f32";
    let [x, y, z] = ["x", "y", "z"].map(|n| buf(case, &format!("data/{n}.bin")));
    // Products of two f32 values are exact in f64.
    assert!((0..16).all(|t| x[t] * y[t] < 1.0));
    let fp64: Vec<f64> = (0..32)
        .map(|t| if t < 16 { 0.0 } else { x[t] + z[t] })
        .collect();
    refused_where_paths_differ(case, &fp64, &(16..32).collect::<Vec<_>>());
}

/// MISS 95 (a rounded value across lanes; GB300 outputs): in c_num_match_f32 lane 0 holds
/// v = rn(a + b) of the parameters a = 1, b = 2^-25 (rn(a + b) = 1, the exact sum 1 + 2^-25)
/// and lanes 1..31 the parameter c = 1; m = match.any(v). Lanes 1..31 store rn(x + 3) where
/// lane 0's v matches theirs (the run) and x where it does not (the exact computation). The
/// rounded-value rule looked at each lane's own operands, so both fronts claimed; a match, vote
/// or reduction now gives every executing lane a rounded operand of any lane. num makes no
/// claim on either front.
#[test]
fn control_from_a_match_of_another_lanes_rounded_value_is_refused() {
    let case = "c_num_match_f32";
    let x = buf(case, "data/x.bin");
    let fp64: Vec<f64> = (0..32).map(|t| if t == 0 { 0.0 } else { x[t] }).collect();
    refused_where_paths_differ(case, &fp64, &(1..32).collect::<Vec<_>>());
}

/// Index arithmetic keeps its claims. c_cat and c_cumsum (captured binaries): addresses use
/// constants ptxas materializes with `HFMA2 R, -RZ, RZ, hi, lo` (c_cat: x then y, copied;
/// c_cumsum: row prefix sums, exact in f64 for these inputs). c_num_index_f32: a loop counter
/// that starts at an HFMA2 constant is also data (f32(i) summed), and r = t / cols by a
/// parameter (SASS: the u32 division idiom I2F.U32.RP, MUFU.RCP, F2I.TRUNC, then an integer
/// correction) decides a load address, a store guard and an EXIT guard. None is input data,
/// and every element's claim is |GB300 - exact|.
#[test]
fn index_arithmetic_idioms_keep_their_claims() {
    let [x, y] = ["x", "y"].map(|n| buf("c_cat", &format!("data/{n}.bin")));
    let fp64: Vec<f64> = x.iter().chain(&y).copied().collect();
    let (analyzed, text) = check_claims("c_cat", &fp64);
    assert_eq!(analyzed.len(), fp64.len(), "c_cat: {text}");
    let x = buf("c_cumsum", "data/x.bin");
    let fp64: Vec<f64> = x
        .chunks(128)
        .flat_map(|row| {
            row.iter().scan(0.0, |s, v| {
                *s += v;
                Some(*s)
            })
        })
        .collect();
    let (analyzed, text) = check_claims("c_cumsum", &fp64);
    assert_eq!(analyzed.len(), fp64.len(), "c_cumsum: {text}");
    let case = "c_num_index_f32";
    let [x, y] = ["x", "y"].map(|n| buf(case, &format!("data/{n}.bin")));
    let mut fp64 = vec![0.0; 384];
    // x, y in [0.5, 2): every sum is exact in f64.
    for t in 0..128 {
        let (r, c) = (t / 12, t % 12);
        let j = (11 * c + r) & 127;
        if r < 8 {
            fp64[t] = x[j] + y[j];
        }
        if r < 9 {
            fp64[128 + t] = x[j];
        }
        fp64[256 + t] = (0..6).map(|i| x[(t + 32 * i) & 127] + i as f64).sum();
    }
    let (analyzed, text) = check_claims(case, &fp64);
    assert_eq!(analyzed.len(), 2 * fp64.len(), "{case}: {text}");
}

/// The independent review's probe (fixtures/c_num_nan2_f32, N-NUM-NONFINITE, MISS 63):
/// out[t] = x + y, out[32 + t] = (x + y) + (u + v). In the even lanes x + y overflows to +inf
/// (exact 6e38) and the second sum is NaN (exact 0): both errors are infinite. Once max_err
/// was 0.75 ulp while 32 of the 64 rows had a NaN error. Every element is analyzed; the odd
/// lanes carry |GB300 - exact|.
#[test]
fn overflowed_sums_carry_infinite_errors() {
    let case = "c_num_nan2_f32";
    let [x, y, u, v] = ["x", "y", "u", "v"].map(|n| buf(case, &format!("data/{n}.bin")));
    let hw = buf(case, "expected/out.bin");
    let (text, csvs) = num_opt(case);
    for (front, csv) in csvs {
        let csv = csv.unwrap_or_else(|| panic!("{case} {front}: no CSV\n{text}"));
        let mut rows = 0;
        for line in csv.lines().skip(1) {
            let c: Vec<&str> = line.split(',').collect();
            let i: usize = c[0].parse().unwrap();
            let t = i % 32;
            assert_eq!(
                c[11], "false",
                "{case} {front} element {i}: not analyzed ({line})"
            );
            // [1, 2) values and 3e38: every exact value is exact in f64.
            let exact = if i < 32 {
                x[t] + y[t]
            } else {
                (x[t] + y[t]) + (u[t] + v[t])
            };
            let want = if t.is_multiple_of(2) {
                f64::INFINITY
            } else {
                (hw[i] - exact).abs()
            };
            let abs_err: f64 = c[3].parse().unwrap();
            assert!(
                abs_err == want || (abs_err - want).abs() <= 1e-12 * want.max(1.0),
                "{case} {front} element {i}: num claims |err| {abs_err}, want {want}\n{text}"
            );
            rows += 1;
        }
        assert_eq!(rows, 64, "{case} {front}");
        let line = text
            .lines()
            .find(|l| l.starts_with(&format!("num good {front:<4}:")))
            .unwrap_or_else(|| panic!("{text}"));
        assert!(
            line.contains("max_err=inf") && line.contains("cor1a_ok=false"),
            "{line}"
        );
    }
}

/// Non-finite values (N-NUM-NONFINITE, MISS 63). Even lanes: (x + y) + (u + v) is NaN and x + y
/// overflows to inf against the finite exact values 0 and 6e38, so each carries an infinite
/// error and max_err is infinite (once NaN errors were skipped and max_err was finite). A copy
/// of +inf, -inf or NaN equals its exact value (error 0), and so does NaN + z. inf + z and the
/// overflowed (x + y) + -inf have a non-finite exact value the f64 expansions do not determine
/// (the exact computation's -inf has the expansion NaN): not analyzed. Other elements: error
/// |GB300 - exact|.
#[test]
fn nonfinite_errors_are_never_skipped() {
    let case = "c_num_nonfinite_f32";
    let [x, y, w, z, w2] = ["x", "y", "w", "z", "w2"].map(|n| buf(case, &format!("data/{n}.bin")));
    let hw = buf(case, "expected/out.bin");
    // The expected status per element: Some(error) analyzed, None not analyzed.
    let mut want: Vec<Option<f64>> = vec![None; 160];
    for t in 0..32 {
        // Odd lanes: u = -x, v = -y (exact 0); x + y and x + y + w2 exact in f64.
        let even = t % 2 == 0;
        let e = |i: usize, exact: f64| Some((hw[i] - exact).abs());
        want[t] = if even { Some(f64::INFINITY) } else { e(t, 0.0) };
        want[32 + t] = if even {
            Some(f64::INFINITY)
        } else {
            e(32 + t, x[t] + y[t])
        };
        want[64 + t] = Some(0.0);
        want[96 + t] = match t % 4 {
            0 | 1 => None,
            2 => Some(0.0),
            _ => e(96 + t, w[t] + z[t]),
        };
        want[128 + t] = if even {
            None
        } else {
            e(128 + t, x[t] + y[t] + w2[t])
        };
    }
    let (text, csvs) = num_opt(case);
    for (front, csv) in csvs {
        let csv = csv.unwrap_or_else(|| panic!("{case} {front}: no CSV\n{text}"));
        for line in csv.lines().skip(1) {
            let c: Vec<&str> = line.split(',').collect();
            let i: usize = c[0].parse().unwrap();
            let (untracked, nonfinite) = (c[11] == "true", c[12] == "true");
            match want[i] {
                None => assert!(
                    untracked && nonfinite,
                    "{case} {front} element {i}: analyzed ({line})\n{text}"
                ),
                Some(e) => {
                    assert!(
                        !untracked,
                        "{case} {front} element {i}: not analyzed ({line})\n{text}"
                    );
                    let abs_err: f64 = c[3].parse().unwrap();
                    assert!(
                        abs_err == e || (abs_err - e).abs() <= 1e-12 * e.max(1.0),
                        "{case} {front} element {i}: num claims |err| {abs_err}, want {e}\n{text}"
                    );
                }
            }
        }
        let line = text
            .lines()
            .find(|l| l.starts_with(&format!("num good {front:<4}:")))
            .unwrap_or_else(|| panic!("{text}"));
        assert!(
            line.contains("max_err=inf") && line.contains("cor1a_ok=false"),
            "{line}"
        );
    }
}

/// Atomics other than f32 adds (MISS 114; the independent review's probes and one more, GB300
/// outputs). Every claim is |GB300 - exact|; the elements below claim nothing:
/// - c_num_atom_read_f32: the old values that atom.or, atom.add.u32, atom.max.u32, a failing
///   atom.cas and atom.exch return from the sum of 32 atomic adds of 0.1f, and x + one of them,
///   were exact constants on both fronts (8 of 8 claimed, 0.25 ulp of the computed output,
///   against 4 ulp). None is claimed now.
/// - c_num_atom_store_f32: s = rn(t + 0.1f) through atom.exch, through st then red.or 0, and
///   through atom.cas: the exchanged and swapped words keep s's node (their claims are checked),
///   the red.or words (32..63) are input data.
/// - c_num_atom_f64_f32: a branch on an f64 atomic sum (0.9999999999999999 here, the exact sum
///   above 1) is refused.
/// - c_num_atom_cas_f32: a compare-and-swap that fails on the run's sum and swaps on the exact
///   one (0..2), and a bf16x2 atomic sum (3, 4), claim nothing; the f32 word a 64-bit exchange
///   writes keeps the exchanged node (5: |GB300 - exact| = 2^-30, an input element's node before).
/// - c_num_cas_after_or_f32: a compare-and-swap on a word that atom.or 0 made input data swaps in
///   the run and not in the exact computation; neither output is claimed (a CAS decided by input
///   data alone claimed out[0] exact, true error 2.2 absolute).
#[test]
fn atomics_other_than_f32_adds_keep_roundings_out_of_claims() {
    let tenth = f32::from_bits(0x3dcc_cccd) as f64;
    let sum = 32.0 * tenth;
    let case = "c_num_atom_read_f32";
    let x = buf(case, "data/x.bin");
    let mut fp64 = vec![sum; 8];
    fp64[7] = sum + x[0];
    let (analyzed, text) = check_claims(case, &fp64);
    assert!(analyzed.is_empty(), "{case}: claims {analyzed:?}\n{text}");

    let case = "c_num_atom_store_f32";
    let x = buf(case, "data/x.bin");
    let fp64: Vec<f64> = (0..128)
        .map(|i| (i % 32) as f64 + tenth + if i >= 96 { x[i % 32] } else { 0.0 })
        .collect();
    let (analyzed, text) = check_claims(case, &fp64);
    assert!(
        analyzed.iter().all(|&(_, i)| !(32..64).contains(&i)),
        "{case}: claims {analyzed:?}\n{text}"
    );

    let case = "c_num_atom_f64_f32";
    let x = buf(case, "data/x.bin");
    let fp64: Vec<f64> = x.iter().map(|v| v + 1.0).collect();
    let (analyzed, text) = check_claims(case, &fp64);
    assert!(analyzed.is_empty(), "{case}: claims {analyzed:?}\n{text}");
    assert!(
        text.contains("control flow or an address depends on input data"),
        "{text}"
    );

    let case = "c_num_atom_cas_f32";
    let x = buf(case, "data/x.bin");
    let bf = 32.0 * f32::from_bits(0x3dcd_0000) as f64;
    let fp64 = [1.0, sum, 1.0 + x[1], bf, bf + x[2], x[0] + 2f64.powi(-30)];
    let (analyzed, text) = check_claims(case, &fp64);
    assert_eq!(analyzed, [("ptx", 5), ("sass", 5)], "{case}\n{text}");

    let case = "c_num_cas_after_or_f32";
    let x = buf(case, "data/x.bin");
    let (analyzed, text) = check_claims(case, &[sum, sum + x[0]]);
    assert!(analyzed.is_empty(), "{case}: claims {analyzed:?}\n{text}");
}

/// Local memory (MISS 158; the independent review's probes, GB300 outputs). Every claim is
/// |GB300 - exact|:
/// - c_num_local_f32: s = rn(t + 0.1f) read back from a local word by two 16-bit loads (0..31)
///   and s's low half stored into a zero word (32..63, no exact value) claim nothing; s and
///   s2 = rn(t + 0.2f) through st/ld.local.b64 (64..127) keep their nodes, so their claims are
///   their rounding errors. Before, 0..31 and 64..127 were exact constants (93 of 128
///   under-claimed, up to 0.40 ulp of the computed output) and 32..63 claims without an exact
///   value.
/// - c_num_local_mix_f32: a constant 16-bit half (0..31) or byte (32..63) stored over a local word
///   holding s claims nothing (bit mixes, no exact value; before, exact constants); s read back
///   whole (64..95) keeps its claim.
/// - c_num_localctl_rnd_f32: a rounded atomic sum through st/ld.local.b64 decides a branch: the
///   output is refused, as for the register-direct twin (claimed 0.50 ulp before, true 1.0
///   absolute).
#[test]
fn local_memory_keeps_taint_and_nodes() {
    let (tenth, fifth) = (
        f32::from_bits(0x3dcc_cccd) as f64,
        f32::from_bits(0x3e4c_cccd) as f64,
    );
    let case = "c_num_local_f32";
    let fp64: Vec<f64> = (0..128)
        .map(|i| match i / 32 {
            0 | 2 => (i % 32) as f64 + tenth,
            3 => (i % 32) as f64 + fifth,
            _ => 0.0,
        })
        .collect();
    let (analyzed, text) = check_claims(case, &fp64);
    for front in ["ptx", "sass"] {
        let mine: Vec<usize> = analyzed
            .iter()
            .filter(|a| a.0 == front)
            .map(|a| a.1)
            .collect();
        assert_eq!(
            mine,
            (64..128).collect::<Vec<_>>(),
            "{case} {front}\n{text}"
        );
    }

    let case = "c_num_local_mix_f32";
    let fp64: Vec<f64> = (0..96)
        .map(|i| if i < 64 { 0.0 } else { (i % 32) as f64 + tenth })
        .collect();
    let (analyzed, text) = check_claims(case, &fp64);
    for front in ["ptx", "sass"] {
        let mine: Vec<usize> = analyzed
            .iter()
            .filter(|a| a.0 == front)
            .map(|a| a.1)
            .collect();
        assert_eq!(mine, (64..96).collect::<Vec<_>>(), "{case} {front}\n{text}");
    }

    let case = "c_num_localctl_rnd_f32";
    let x = buf(case, "data/x.bin");
    let fp64: Vec<f64> = x.iter().map(|v| v + 1.0).collect();
    let (analyzed, text) = check_claims(case, &fp64);
    assert!(analyzed.is_empty(), "{case}: claims {analyzed:?}\n{text}");
    assert!(
        text.contains("control flow or an address depends on input data"),
        "{text}"
    );
}

/// Partial-width and rearranging accesses of words holding a rounding (MISS 115; the independent
/// review's probes and three more, GB300 outputs). Every claim is |GB300 - exact|; the elements
/// below claim nothing:
/// - c_num_ld16_f32: s = rn(t + 0.1f) read back by two 16-bit loads and recombined (before: 64 of
///   64 claimed on both fronts, 0.50 ulp of the computed output, 51 under-claimed).
/// - c_num_ldsm_trans_f32: rounded words read by ldmatrix .trans and rebuilt by a shuffle and a
///   prmt (the lanes with l/4 even); the zero lanes keep their exact claims.
/// - c_num_st16_f32: a 16-bit store into a word holding s, and s's low half stored into a zero
///   word: words mixing bytes of a rounding with others, with no exact value.
/// - c_num_tma16_f32: s copied to shared memory by a TMA load of 2-byte elements.
/// - c_num_stsm8_f32: stmatrix .b8 writes bytes of s over zero words, and constant bytes over words
///   holding s (whose old nodes stayed before).
#[test]
fn partial_and_rearranging_accesses_of_rounded_words_are_not_exact() {
    let tenth = f32::from_bits(0x3dcc_cccd) as f64;
    let case = "c_num_ld16_f32";
    let x = buf(case, "data/x.bin");
    let fp64: Vec<f64> = (0..64)
        .map(|i| ((i % 32) ^ 1) as f64 + tenth + if i >= 32 { x[i % 32] } else { 0.0 })
        .collect();
    let (analyzed, text) = check_claims(case, &fp64);
    assert!(analyzed.is_empty(), "{case}: claims {analyzed:?}\n{text}");

    let case = "c_num_ldsm_trans_f32";
    let fp64: Vec<f64> = (0..32)
        .map(|l| {
            if (l / 4) % 2 == 0 {
                (8 * (l % 4) + l / 8) as f64 + tenth
            } else {
                0.0
            }
        })
        .collect();
    let (analyzed, text) = check_claims(case, &fp64);
    assert!(
        analyzed.iter().all(|&(_, l)| (l / 4) % 2 == 1),
        "{case}: claims {analyzed:?}\n{text}"
    );

    let case = "c_num_tma16_f32";
    let x = buf(case, "data/x.bin");
    let fp64: Vec<f64> = (0..64)
        .map(|i| (i % 32) as f64 + tenth + if i >= 32 { x[i % 32] } else { 0.0 })
        .collect();
    let (analyzed, text) = check_claims(case, &fp64);
    assert!(analyzed.is_empty(), "{case}: claims {analyzed:?}\n{text}");

    // Mixed bytes: no exact value, so any claim fails.
    for case in ["c_num_st16_f32", "c_num_stsm8_f32"] {
        let n = buf(case, "expected/out.bin").len();
        let (analyzed, text) = check_claims(case, &vec![f64::NAN; n]);
        assert!(analyzed.is_empty(), "{case}: claims {analyzed:?}\n{text}");
    }
}

/// Constants ptxas folds (MISS 108; GB300 outputs). ptxas folds roundings of immediates
/// (`mul.rn`, `add.rn`, `ex2.approx`) into immediates the SASS front would take as exact constants;
/// the PTX front has the roundings. The SASS front claims only elements the PTX front claims with
/// the same exact value, and no output the PTX front refuses:
/// - c_num_fold_f32 (the independent review's probe): mul.rn(0.1f, 3), mul.rn(0.1f, 0.1f) and
///   ex2.approx.ftz(0.1f). The SASS front claimed 192 of 192 (0.50 ulp of the computed output),
///   155 under-claimed; the PTX front claims none, and now neither front does.
/// - c_num_fold_add_f32: add.rn(0.1f, 1) and add.rn(1, 2^-24) (folded to 1.0). The PTX front
///   tracks the sums and claims every element, each |GB300 - exact|; the SASS front, whose exact
///   values miss the folded roundings, claims none.
/// - c_num_fold_ctl_f32: a branch on a literal against the folded mul.rn(0.1f, 3) (ptxas folds the
///   comparison away): the PTX front refuses the output, and so does the SASS front.
#[test]
fn sass_claims_need_a_ptx_claim_with_the_same_exact_value() {
    let tenth = f32::from_bits(0x3dcc_cccd) as f64;
    let case = "c_num_fold_f32";
    let x = buf(case, "data/x.bin");
    let c = [tenth * 3.0, tenth * tenth, tenth.exp2()];
    let fp64: Vec<f64> = (0..192).map(|i| x[i] + c[(i / 32) % 3]).collect();
    let (analyzed, text) = check_claims(case, &fp64);
    assert!(analyzed.is_empty(), "{case}: claims {analyzed:?}\n{text}");

    let case = "c_num_fold_add_f32";
    let x = buf(case, "data/x.bin");
    let c = [tenth + 1.0, 1.0 + 2f64.powi(-24)];
    let fp64: Vec<f64> = (0..128).map(|i| x[i] + c[(i / 32) % 2]).collect();
    let (analyzed, text) = check_claims(case, &fp64);
    assert!(
        analyzed.len() == 128 && analyzed.iter().all(|&(f, _)| f == "ptx"),
        "{case}: claims {analyzed:?}\n{text}"
    );

    let case = "c_num_fold_ctl_f32";
    let x = buf(case, "data/x.bin");
    let fp64: Vec<f64> = (0..32)
        .map(|t| x[t] + if t % 2 == 0 { 1.0 } else { 2.0 })
        .collect();
    let (analyzed, text) = check_claims(case, &fp64);
    assert!(analyzed.is_empty(), "{case}: claims {analyzed:?}\n{text}");
    assert!(text.contains("the PTX front makes no claim"), "{text}");
}

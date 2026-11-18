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
//! - c_num_stasync_f32 (MISS 205): a value stored by st.async over an input word's bits takes
//!   the stored value's node, not the word's earlier one.
//! - c_num_mufu_top_f32 (MISS 206): MUFU bounds at the 16 largest finite magnitudes.
//! - c_num_mufu_ftz_f32 (MISS 207): PTX .ftz approx results carry the MUFU bound.
//! - c_num_nic_write_f32 (MISS 224): an RDMA write moves the source words' nodes.
//! - c_num_nic_atomic_f32 (MISS 257): a NIC atomic keeps the nodes of the words it leaves
//!   unchanged, and words whose outcome a rounding or input data decides are input data.
//! - c_num_mufu_nan_f32 (MISS 258): sqrt, rsqrt and lg2 of negative numbers (negative
//!   subnormals the .ftz forms flush to -0, and -2) have no real value: not analyzed.
//! - c_num_mufu_inf_f32 (MISS 258): lg2, rsqrt and rcp of an exact 0 (a sum that is 1 in f32)
//!   have no finite value: not analyzed, also under max, min, ex2, rcp and add.
//! - c_num_mufu_nan_mma (MISS 258): such a MUFU result under an MMA factor (bf16 mma.sync):
//!   the products' outputs are not analyzed.
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
    num_tagged(case, "num")
}

/// `num` into an output directory of its own (`tag`: tests of one case run in parallel).
fn num_tagged(case: &str, tag: &str) -> (String, Vec<(&'static str, String)>) {
    let out = std::env::temp_dir().join(format!("ptxv-test-{tag}-{case}-{}", std::process::id()));
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

fn scalar_atomic_refused(text: &str, front: &str, output: &str) -> bool {
    let prefix = format!(
        "num good {front:<4}: output={output} NOT ANALYZED: the numerics run reported unmodeled: scalar atomic at "
    );
    text.lines().any(|line| {
        line.starts_with(&prefix)
            && line.ends_with(
                " has overlapping or shared-return warp-lane claims whose interleavings are not modeled",
            )
    })
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
            let Some(csv) = csv else {
                assert!(
                    case == "c_num_atom_old_f32" && scalar_atomic_refused(&text, front, "out"),
                    "{case} {front}: no CSV\n{text}"
                );
                continue;
            };
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
        assert!(
            acc.contains("cor1a_ok=true") || scalar_atomic_refused(&text, front, "acc"),
            "{acc}"
        );
    }
}

/// MISS 205: c_num_stasync_f32 (probes/stasync_num_shadow r5_stasync_co_c1): out[t] = add.rn(t,
/// 0.1f) stored by st.async over the input x[t] = rn(t + 0.1f), the same bits. The word takes the
/// stored value's node: every element's claimed error is |GB300 - (t + 0.1f)| (up to 3.8e-7, at
/// out[8]) on both fronts. Before, the word kept x's input node: abs_err 0 and bound 0 claimed at
/// the 31 elements with a rounding.
#[test]
fn stasync_words_take_the_stored_value_node() {
    let tenth = 0.1f32 as f64;
    let fp64: Vec<f64> = (0..32).map(|t| t as f64 + tenth).collect();
    let (analyzed, text) = check_claims("c_num_stasync_f32", &fp64);
    assert_eq!(analyzed.len(), 64, "{text}");
}

/// MISS 224: c_num_nic_write_f32 (probes/nic_num_shadow loop): one GPU, a loopback IBGDA queue
/// pair; dst holds a copy of the input c (c[i] = rn(x[i] + y[i])), the RDMA write of src[i] =
/// x[i] + y[i] replaces it with the same bits, and out = dst after the signal's completion. The
/// write moves src's nodes: every element's claimed error is |GB300 - (x + y)| (up to 1.0, at
/// out[3]) and its bound covers it, on both fronts. Before, out kept the copy's input node:
/// abs_err 0 and bound 0 at every element.
#[test]
fn nic_writes_move_numerics_nodes() {
    let case = "c_num_nic_write_f32";
    let x = [0.1f32, 1.0, 3.0, 1e8];
    let y = [0.2f32, 1e-8, 0.1, 1.0];
    let fp64: Vec<f64> = x
        .iter()
        .zip(&y)
        .map(|(&a, &b)| a as f64 + b as f64)
        .collect();
    let (analyzed, text) = check_claims(case, &fp64);
    assert_eq!(analyzed.len(), 8, "{text}");
    let hw = buf(case, "expected/out.bin");
    let (text, csvs) = num_opt(case);
    for (front, csv) in csvs {
        let csv = csv.unwrap_or_else(|| panic!("{front}: {text}"));
        for line in csv.lines().skip(1) {
            let c: Vec<&str> = line.split(',').collect();
            let i: usize = c[0].parse().unwrap();
            let bound: f64 = c[6].parse().unwrap();
            let err = (hw[i] - fp64[i]).abs();
            assert!(
                err > 0.0 && bound >= err,
                "{front} element {i}: bound {bound}, GB300 error {err}\n{text}"
            );
        }
    }
}

/// MISS 257: c_num_nic_atomic_f32: NIC atomics (one GPU, a loopback queue pair) on dst's 8-byte
/// words: W0..W3 = (rn(0.1f + 0.2f), rn(1 + 1e-8f)) (roundings of constants), W4 = (2.0f, 3.0f),
/// W5 = four bf16 elements of those roundings (16-bit stores), W6, W7 = the same roundings of
/// input data. out[0..16) = dst after the atomics, out[16..32) = the values they fetched, outh =
/// W5's bf16 elements. On each front every claim equals the element's exact error and its bound
/// covers it, and:
/// - W1, W5, W6 (a fetch-add of 0) and W3's low word (a fetch-add of 2^32) are unchanged whatever
///   the words held: they keep their nodes and 16-bit provenance (claimed on the PTX front; the
///   SASS front, where ptxas folds the constants' roundings, claims W6 only, as MISS 108's gate
///   allows);
/// - no claim at W0 (a compare-and-swap that fails on rounded words: the exact computation may
///   swap), W2, W7 (a fetch-add of 1: the low word changed, the high word's carry is the low
///   word's) and W3's high word (changed): input data;
/// - W4 (swapped in 7.0f, 5.0f) is claimed exact;
/// - the fetched values carry the old words' nodes.
///
/// Until MISS 257 every word lost its node without taint: W0..W3 were claimed exact on both
/// fronts (abs_err 0, bound 0), where the unchanged words' errors are 7.45e-9 and 1.0e-8.
#[test]
fn nic_atomics_keep_unchanged_words_and_taint_decided_ones() {
    #[derive(Clone, Copy, PartialEq)]
    enum Want {
        Claim,
        NoClaim,
        Either,
    }
    use Want::*;
    let case = "c_num_nic_atomic_f32";
    let r = [0.1f32 as f64 + 0.2f32 as f64, 1.0 + 1e-8f32 as f64];
    let h = [(0.1f32 + 0.2f32) as f64, (1.0f32 + 1e-8f32) as f64];
    // Per element: (exact, simulated), PTX front, SASS front.
    let rounded = |k: usize, ptx: Want, sass: Want| ((r[k], h[k]), ptx, sass);
    let exact = |v: f64| ((v, v), Claim, Claim);
    // A fetch-add of 1 on the word's bits: the next f32 up.
    let up = |v: f32| f32::from_bits(v.to_bits() + 1) as f64;
    // W5's words read as f32: two bf16 elements each, whose words hold roundings (input data,
    // MISS 115).
    let bf16_pairs = [f32::from_bits(0x3f80_3e9a), f32::from_bits(0x3e9a_3f80)];
    let w5 = |k: usize| ((f64::NAN, bf16_pairs[k] as f64), NoClaim, NoClaim);
    let mut out = vec![
        rounded(0, NoClaim, NoClaim),
        rounded(1, NoClaim, NoClaim),
        rounded(0, Claim, Either),
        rounded(1, Claim, Either),
        ((r[0], up(0.1f32 + 0.2f32)), NoClaim, NoClaim),
        rounded(1, NoClaim, NoClaim),
        rounded(0, Claim, Either),
        ((r[1], up(1.0f32 + 1e-8f32)), NoClaim, NoClaim),
        exact(7.0),
        exact(5.0),
        w5(0),
        w5(1),
        rounded(0, Claim, Claim),
        rounded(1, Claim, Claim),
        ((r[0], up(0.1f32 + 0.2f32)), NoClaim, NoClaim),
        rounded(1, NoClaim, NoClaim),
    ];
    for _ in 0..4 {
        out.extend([rounded(0, Claim, Either), rounded(1, Claim, Either)]);
    }
    out.extend([exact(2.0), exact(3.0), w5(0), w5(1)]);
    for _ in 0..2 {
        out.extend([rounded(0, Claim, Claim), rounded(1, Claim, Claim)]);
    }
    // bf16(rn(0.1f + 0.2f)) = 0x3e9a; bf16(1) = 1.
    let hb = [f32::from_bits(0x3e9a_0000) as f64, 1.0];
    let outh: Vec<_> = [0, 1, 1, 0]
        .into_iter()
        .map(|k| ((r[k], hb[k]), Claim, Either))
        .collect();
    let dir = std::env::temp_dir().join(format!("ptxv-test-num257-{case}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let p = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["num", root().join("fixtures").join(case).to_str().unwrap()])
        .arg("--out")
        .arg(&dir)
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&p.stdout),
        String::from_utf8_lossy(&p.stderr)
    );
    assert!(p.status.success(), "{text}");
    for front in ["ptx", "sass"] {
        for (sfx, want) in [("", &out), (".outh", &outh)] {
            let file = dir.join(format!("good.{front}.num{sfx}.csv"));
            let csv = std::fs::read_to_string(&file)
                .unwrap_or_else(|e| panic!("{}: {e}\n{text}", file.display()));
            let mut rows = 0;
            for line in csv.lines().skip(1) {
                let c: Vec<&str> = line.split(',').collect();
                let i: usize = c[0].parse().unwrap();
                let ((exact, hw), ptx, sass) = want[i];
                let w = if front == "ptx" { ptx } else { sass };
                let claimed = c[11] == "false" && c[12] == "false";
                let at = format!("{front}{sfx} element {i}: {line}");
                rows += 1;
                let got = c[1].parse::<f32>().unwrap() as f64;
                assert_eq!(got, hw, "{at}");
                assert!(w != Claim || claimed, "{at}: no claim\n{text}");
                assert!(w != NoClaim || !claimed, "{at}: claimed\n{text}");
                if claimed {
                    let abs_err: f64 = c[3].parse().unwrap();
                    let bound: f64 = c[6].parse().unwrap();
                    let err = (hw - exact).abs();
                    assert!(
                        (abs_err - err).abs() <= 1e-12 * exact.abs().max(1.0) && bound >= err,
                        "{at}: abs_err {abs_err}, bound {bound}, exact error {err}\n{text}"
                    );
                }
            }
            assert_eq!(rows, want.len(), "{front}{sfx}: {text}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// MISS 206 (defense in depth): a bound that an analyzed element of its output exceeds is no
/// bound. c_num_nonfinite_f32's overflowed elements carry infinite errors (N-NUM-NONFINITE), so
/// cor1a_ok and running_ok are false and every element of the output reports both bounds as
/// unknown, as does the summary's bound / error ratio; c_num_sum_f32 (no violation) keeps its
/// bounds. Until MISS 206 the bounds printed whatever they were (an overflowed MUFU bound:
/// 2^-126 against errors of 7.7e25, probes/mufu_bound_top).
#[test]
fn bounds_an_output_violates_are_unknown() {
    let (text, csvs) = num_tagged("c_num_nonfinite_f32", "num-unknown");
    for (front, csv) in csvs {
        for line in csv.lines().skip(1) {
            let c: Vec<&str> = line.split(',').collect();
            assert_eq!(
                (c[6], c[7]),
                ("unknown", "unknown"),
                "{front}: {line}\n{text}"
            );
        }
        let summary = text
            .lines()
            .find(|l| l.starts_with(&format!("num good {front:<4}:")))
            .unwrap_or_else(|| panic!("{text}"));
        assert!(
            summary.contains("cor1a_ok=false running_ok=false")
                && summary.contains("bound/err(mean)=unknown"),
            "{summary}"
        );
    }
    let (text, csvs) = num_tagged("c_num_sum_f32", "num-unknown");
    for (front, csv) in csvs {
        for line in csv.lines().skip(1) {
            let c: Vec<&str> = line.split(',').collect();
            assert!(
                c[11] == "true" || c[6].parse::<f64>().is_ok(),
                "{front}: {line}\n{text}"
            );
        }
    }
}

/// MISS 206: c_num_mufu_top_f32: PTX rcp.approx.f32 (no .ftz) of subnormals 2^-128 + k 2^-149
/// gives 0x7f7ffff8 / 0x7f7ffff0 at k = 1, 2, among the 16 largest finite magnitudes, where
/// |hw| (1 + 2^-20) rounds to infinity in f32. Every claimed element's bounds cover its GB300
/// error |GB300 - 1/x| (1/x in f64: a relative error of 2^-53, far below the margins), and the
/// output keeps its bounds (cor1a_ok). Before, the MUFU bound at those two elements was 2^-126
/// against errors of 7.7e25 and 3.1e26 (since the bound check of MISS 206, unknown for the whole
/// output). The SASS front claims none of these elements.
#[test]
fn mufu_bound_at_the_top_magnitudes_covers_gb300() {
    let case = "c_num_mufu_top_f32";
    let x = buf(case, "data/x.bin");
    let hw = buf(case, "expected/out.bin");
    let (text, csvs) = num_opt(case);
    let csv = csvs[0].1.as_ref().unwrap_or_else(|| panic!("{text}"));
    let mut claimed = vec![];
    for line in csv.lines().skip(1) {
        let c: Vec<&str> = line.split(',').collect();
        let i: usize = c[0].parse().unwrap();
        if c[11] == "true" {
            continue;
        }
        claimed.push(i);
        let te = (hw[i] - 1.0 / x[i]).abs();
        let slack = 1e-15 * (1.0 / x[i]).abs();
        for b in [c[6], c[7]] {
            let b: f64 = b
                .parse()
                .unwrap_or_else(|_| panic!("element {i}: {line}\n{text}"));
            assert!(
                b >= te - slack,
                "element {i}: bound {b} < |GB300 - 1/x| {te}\n{text}"
            );
        }
    }
    assert!(
        claimed.contains(&0) && claimed.contains(&1),
        "{claimed:?}\n{text}"
    );
    assert!(text.contains("num good ptx : outputs=32 thm1_exact=false spec_match=None cor1a_ok=true running_ok=true"), "{text}");
}

/// MISS 207: c_num_mufu_ftz_f32: PTX ex2.approx.ftz.f32 (oute), rcp.approx.ftz.f32 (outr) and
/// sqrt.approx.ftz.f32 (outs), a one-instruction slice node each in `ptxv num`, carry the swept
/// MUFU bound plus the flush of a subnormal operand (sqrt: 0 against sqrt(x) up to 2^-63). Every
/// claimed element's bounds on both fronts cover its GB300 error |GB300 - f(x)| (f(x) in f64), and
/// every output keeps its bounds. Before, the PTX front's bound was 0 at every element, below
/// errors up to 1.5e31 (ex2 of 128 - 2^-17), 4.7e-10 (rcp of 128 - 2^-17) and 5.4e-20 (sqrt of a
/// subnormal); since MISS 206's bound check, unknown for the whole output.
#[test]
fn ptx_ftz_approx_results_carry_the_sweep_bound() {
    let case = "c_num_mufu_ftz_f32";
    let (x, y) = (buf(case, "data/x.bin"), buf(case, "data/y.bin"));
    let out = std::env::temp_dir().join(format!("ptxv-test-numftz-{case}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let r = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["num", root().join("fixtures").join(case).to_str().unwrap()])
        .arg("--out")
        .arg(&out)
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&r.stdout).to_string();
    assert!(r.status.success(), "{text}");
    // (output, CSV name suffix, operands, the exact function of an operand)
    type Out = (&'static str, &'static str, Vec<f64>, fn(f64) -> f64);
    let outputs: [Out; 3] = [
        ("oute", "", x.clone(), f64::exp2),
        ("outr", ".outr", y, |v| 1.0 / v),
        ("outs", ".outs", x, f64::sqrt),
    ];
    for (name, sfx, arg, f) in outputs {
        let hw = buf(case, &format!("expected/{name}.bin"));
        for front in ["ptx", "sass"] {
            let csv =
                std::fs::read_to_string(out.join(format!("good.{front}.num{sfx}.csv"))).unwrap();
            let mut claimed = 0;
            for line in csv.lines().skip(1) {
                let c: Vec<&str> = line.split(',').collect();
                let i: usize = c[0].parse().unwrap();
                if c[11] == "true" {
                    continue;
                }
                claimed += 1;
                let te = (hw[i] - f(arg[i])).abs();
                let slack = 1e-15 * f(arg[i]).abs();
                for b in [c[6], c[7]] {
                    let b: f64 = b
                        .parse()
                        .unwrap_or_else(|_| panic!("{name} {front} {i}: {line}\n{text}"));
                    assert!(
                        b >= te - slack,
                        "{name} {front} {i}: bound {b} < {te}\n{text}"
                    );
                }
            }
            assert_eq!(claimed, 32, "{name} {front}\n{text}");
            let summary = text
                .lines()
                .find(|l| l.starts_with(&format!("num good {front:<4}: output={name} ")))
                .unwrap_or_else(|| panic!("{text}"));
            assert!(
                summary.contains("cor1a_ok=true running_ok=true"),
                "{summary}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&out);
}

/// MISS 258: c_num_mufu_nan_f32: out[32 k + t] = sqrt.approx.ftz, rsqrt.approx.ftz,
/// lg2.approx.ftz, ex2.approx.ftz(lg2.approx.ftz), max(lg2.approx.ftz, -200) and sqrt.approx (no
/// .ftz) of x[t]. At a negative operand (three negative subnormals, which the .ftz forms and the
/// bare MUFU units flush to -0, and -2) the true functions have no real value: on both fronts
/// every such element is not analyzed, nonfinite with its exact value and errors NaN where it
/// keeps a node (max(NaN, -200) at -2 keeps none, nor does the SASS front's sqrt.approx
/// sequence: untracked). Every other element is claimed (on the SASS front all but sqrt.approx),
/// every claim's bounds cover |GB300 - f(x)| (f in f64), and the output keeps its bounds. Until
/// MISS 258 num_mufu took the true value's NaN as the exact value 0: sqrt of a negative subnormal
/// was claimed with abs_err 0, ex2(lg2(.)) of it had exact 1 and abs_err 1, max(lg2(.), -200)
/// exact 0 and abs_err 200, and the NaN results at -2 carried infinite errors, so the output's
/// bounds were unknown.
#[test]
fn mufu_results_without_a_real_value_are_not_analyzed() {
    let case = "c_num_mufu_nan_f32";
    let x = buf(case, "data/x.bin");
    let hw = buf(case, "expected/out.bin");
    let f: [fn(f64) -> f64; 6] = [
        f64::sqrt,
        |v| 1.0 / v.sqrt(),
        f64::log2,
        |v| v.log2().exp2(),
        |v| v.log2().max(-200.0),
        f64::sqrt,
    ];
    let (text, csvs) = num_opt(case);
    for (front, csv) in csvs {
        let csv = csv.unwrap_or_else(|| panic!("{case} {front}: no CSV\n{text}"));
        let mut claimed = 0;
        for line in csv.lines().skip(1) {
            let c: Vec<&str> = line.split(',').collect();
            let i: usize = c[0].parse().unwrap();
            let (k, t) = (i / 32, i % 32);
            let (untracked, nonfinite) = (c[11] == "true", c[12] == "true");
            // No node: max(NaN, -200) at -2, the SASS front's sqrt.approx sequence.
            let dropped = (k == 4 && x[t] == -2.0) || (k == 5 && front == "sass");
            if x[t] < 0.0 {
                assert!(
                    untracked && (dropped || nonfinite && c[2] == "NaN" && c[3] == "NaN"),
                    "{front} element {i} (x = {:e}): analyzed ({line})\n{text}",
                    x[t]
                );
                continue;
            }
            if untracked {
                assert!(dropped, "{front} element {i}: no claim ({line})\n{text}");
                continue;
            }
            claimed += 1;
            let e = f[k](x[t]);
            let te = (hw[i] - e).abs();
            let slack = 1e-15 * e.abs();
            for b in [c[6], c[7]] {
                let b: f64 = b
                    .parse()
                    .unwrap_or_else(|_| panic!("{front} element {i}: {line}\n{text}"));
                assert!(
                    b >= te - slack,
                    "{front} element {i}: bound {b} < {te}\n{text}"
                );
            }
        }
        let want = if front == "ptx" { 6 * 16 } else { 5 * 16 };
        assert_eq!(claimed, want, "{front}\n{text}");
        let summary = text
            .lines()
            .find(|l| l.starts_with(&format!("num good {front:<4}: ")))
            .unwrap_or_else(|| panic!("{text}"));
        assert!(
            summary.contains("cor1a_ok=true running_ok=true"),
            "{summary}"
        );
    }
}

/// MISS 258: c_num_mufu_inf_f32: out[32 k + t] = chain_k(x[t]), x = ((a + b) + c) + e, for lg2,
/// rsqrt and rcp .approx, their .ftz forms, max(lg2f, -1000), min(rsqrtf, 1000),
/// min(rcpf, 1000), ex2f(lg2f), rcpf(rsqrtf), max(lg2, -1000) and add(lg2f, 1000). In lanes 0..15
/// x is 1 in f32 and exactly 0, where lg2, rsqrt and rcp have no finite true value (-inf, +inf,
/// a pole): on both fronts every such element is not analyzed, nonfinite with its exact value and
/// errors NaN where it keeps a node (the SASS front's non-ftz sequences keep none: untracked). In
/// lanes 16..31 (x = 2, 0.5, 3, 1.5, exact) every element is claimed (on the SASS front all but
/// the four chains of a non-ftz form), every claim's bounds cover |GB300 - f(x)| (f in f64), and
/// the output keeps its bounds. Until then num_mufu took the infinite true values as the exact
/// value 0: max(lg2.approx.ftz(x), -1000) was claimed exact with bound 43 where the true error is
/// 1000, and ex2(lg2.approx.ftz(x)) with error 0 where it is 1.
#[test]
fn mufu_results_without_a_finite_value_are_not_analyzed() {
    let case = "c_num_mufu_inf_f32";
    let [a, b, c, e] = ["a", "b", "c", "e"].map(|n| buf(case, &format!("data/{n}.bin")));
    let hw = buf(case, "expected/out.bin");
    let f: [fn(f64) -> f64; 13] = [
        f64::log2,
        |v| 1.0 / v.sqrt(),
        |v| 1.0 / v,
        f64::log2,
        |v| 1.0 / v.sqrt(),
        |v| 1.0 / v,
        |v| v.log2().max(-1000.0),
        |v| (1.0 / v.sqrt()).min(1000.0),
        |v| (1.0 / v).min(1000.0),
        |v| v,
        f64::sqrt,
        |v| v.log2().max(-1000.0),
        |v| v.log2() + 1000.0,
    ];
    // The chains of a non-ftz form: no node on the SASS front.
    let plain = [0, 1, 2, 11];
    let (text, csvs) = num_opt(case);
    for (front, csv) in csvs {
        let csv = csv.unwrap_or_else(|| panic!("{case} {front}: no CSV\n{text}"));
        let mut claimed = 0;
        for line in csv.lines().skip(1) {
            let col: Vec<&str> = line.split(',').collect();
            let i: usize = col[0].parse().unwrap();
            let (k, t) = (i / 32, i % 32);
            let (untracked, nonfinite) = (col[11] == "true", col[12] == "true");
            let dropped = front == "sass" && plain.contains(&k);
            // The exact operand (these sums are exact in f64).
            let x = a[t] + b[t] + c[t] + e[t];
            if x == 0.0 {
                assert!(
                    untracked && (dropped || nonfinite && col[2] == "NaN" && col[3] == "NaN"),
                    "{front} element {i} (exact operand 0): analyzed ({line})\n{text}"
                );
                continue;
            }
            if untracked {
                assert!(dropped, "{front} element {i}: no claim ({line})\n{text}");
                continue;
            }
            claimed += 1;
            let ex = f[k](x);
            let te = (hw[i] - ex).abs();
            let slack = 1e-15 * ex.abs();
            for bnd in [col[6], col[7]] {
                let bnd: f64 = bnd
                    .parse()
                    .unwrap_or_else(|_| panic!("{front} element {i}: {line}\n{text}"));
                assert!(
                    bnd >= te - slack,
                    "{front} element {i}: bound {bnd} < {te}\n{text}"
                );
            }
        }
        let want = if front == "ptx" { 13 * 16 } else { 9 * 16 };
        assert_eq!(claimed, want, "{front}\n{text}");
        let summary = text
            .lines()
            .find(|l| l.starts_with(&format!("num good {front:<4}: ")))
            .unwrap_or_else(|| panic!("{text}"));
        assert!(
            summary.contains("cor1a_ok=true running_ok=true"),
            "{summary}"
        );
    }
}

/// MISS 258: c_num_mufu_nan_mma: D = A B + 0 by mma.sync.m16n8k16 (bf16, f32 accumulate) with
/// A = bf16(sqrt.approx.ftz.f32(x[l])) for the lane l holding the element (x[0] = -3 2^-149, which
/// .ftz flushes to -0, where sqrt has no real value; x[l] = 1 in the other lanes) and B = bf16
/// ones. out[0..16) (rows 0 and 8 of D) hold lane 0's products: on both fronts not analyzed,
/// nonfinite with exact value and errors NaN. out[16..128) are 16 exactly: claimed with error 0.
/// Until then out_rec did not walk an MMA's factor nodes: out[0..16) were claimed with abs_err 0
/// and bound 6.79e-6 against the exact value made up for the MUFU (0).
#[test]
fn mufu_under_an_mma_factor_without_a_real_value_is_not_analyzed() {
    let case = "c_num_mufu_nan_mma";
    let hw = buf(case, "expected/out.bin");
    let (text, csvs) = num_opt(case);
    for (front, csv) in csvs {
        let csv = csv.unwrap_or_else(|| panic!("{case} {front}: no CSV\n{text}"));
        let mut claimed = 0;
        for line in csv.lines().skip(1) {
            let col: Vec<&str> = line.split(',').collect();
            let i: usize = col[0].parse().unwrap();
            let (untracked, nonfinite) = (col[11] == "true", col[12] == "true");
            if i < 16 {
                assert!(
                    untracked && nonfinite && col[2] == "NaN" && col[3] == "NaN",
                    "{front} element {i}: analyzed ({line})\n{text}"
                );
                continue;
            }
            assert!(!untracked, "{front} element {i}: no claim ({line})\n{text}");
            claimed += 1;
            assert_eq!(hw[i], 16.0, "{front} element {i}");
            let abs_err: f64 = col[3].parse().unwrap();
            assert_eq!(abs_err, 0.0, "{front} element {i}: {line}\n{text}");
        }
        assert_eq!(claimed, 112, "{front}\n{text}");
        let summary = text
            .lines()
            .find(|l| l.starts_with(&format!("num good {front:<4}: ")))
            .unwrap_or_else(|| panic!("{text}"));
        assert!(
            summary.contains("cor1a_ok=true running_ok=true"),
            "{summary}"
        );
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
        text.contains("control flow or an address depends on input data or on a rounded value")
            || case == "c_num_ctl_atom_f32"
                && ["ptx", "sass"]
                    .iter()
                    .all(|front| scalar_atomic_refused(&text, front, "out")),
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

/// The error in ulps at an output of +-FLT_MAX (MISS 161; fixtures/c_num_fltmax_f32, the
/// independent review's probe, GB300 output): add.rz(FLT_MAX, (t + 1) 2^99) is FLT_MAX in the
/// even lanes, (t + 1) 2^99 below the exact sum, (t + 1) / 32 of the top binade's ulp 2^104. Every
/// claim is |GB300 - exact|, and its ulp figure that over 2^104 (the odd lanes: over their own
/// ulp). Before, FLT_MAX's ulp was its distance to +inf and the even lanes' figure 0: max_err
/// printed 0.50 ulp (an odd lane's) against 31/32 ulp.
#[test]
fn ulp_figure_at_flt_max_uses_the_top_binade() {
    let case = "c_num_fltmax_f32";
    let x = buf(case, "data/x.bin");
    let fp64: Vec<f64> = (0..32).map(|t| x[t] + x[32 + t]).collect();
    let hw = buf(case, "expected/out.bin");
    let (analyzed, text) = check_claims(case, &fp64);
    for front in ["ptx", "sass"] {
        assert_eq!(
            analyzed.iter().filter(|a| a.0 == front).count(),
            32,
            "{case} {front}\n{text}"
        );
    }
    let (_, csvs) = num_opt(case);
    for (front, csv) in csvs {
        for line in csv.unwrap().lines().skip(1) {
            let c: Vec<&str> = line.split(',').collect();
            let i: usize = c[0].parse().unwrap();
            let (abs_err, ulps): (f64, f64) = (c[3].parse().unwrap(), c[5].parse().unwrap());
            let h = hw[i] as f32;
            let ulp = if h == f32::MAX {
                2f64.powi(104)
            } else {
                (f32::from_bits(h.to_bits() + 1) - h) as f64
            };
            assert_eq!(ulps, abs_err / ulp, "{case} {front} element {i}\n{text}");
            if i.is_multiple_of(2) {
                assert_eq!(ulps, (i + 1) as f64 / 32.0, "{case} {front} element {i}");
            }
        }
    }
    assert!(text.contains("(0.97 ulp)"), "{text}");
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
        text.contains("control flow or an address depends on input data")
            || ["ptx", "sass"]
                .iter()
                .all(|front| scalar_atomic_refused(&text, front, "out")),
        "{text}"
    );

    let case = "c_num_atom_cas_f32";
    let x = buf(case, "data/x.bin");
    let bf = 32.0 * f32::from_bits(0x3dcd_0000) as f64;
    let fp64 = [1.0, sum, 1.0 + x[1], bf, bf + x[2], x[0] + 2f64.powi(-30)];
    let (analyzed, text) = check_claims(case, &fp64);
    let expected = if scalar_atomic_refused(&text, "sass", "out") {
        vec![("ptx", 5)]
    } else {
        vec![("ptx", 5), ("sass", 5)]
    };
    assert_eq!(analyzed, expected, "{case}\n{text}");

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

/// A kernel with PTX the PTX front-end refuses (MISS 160; fixtures/c_num_fold_ptxref_f32, the
/// independent review's probe, GB300 output): x + mul.rn(0.1f, 3), the product folded by ptxas into
/// an immediate, next to atom.dec.u32 (variant good) or activemask.b32 (variant refused, which
/// the PTX front-end does not model). Neither variant gets a claim: in good the PTX front runs
/// (atom.dec is modeled) and claims nothing, so neither does the SASS front (MISS 108); in refused
/// the PTX front-end refuses the PTX, so no PTX claim bounds the SASS front's and it makes none.
/// Before, the PTX front-end refused both and the SASS front claimed 32 of 32 alone (0.50 ulp of
/// the computed output; 19 under-claimed, true error up to 0.5625 ulp).
#[test]
fn sass_claims_need_a_ptx_front_that_runs() {
    let tenth = f32::from_bits(0x3dcc_cccd) as f64;
    let case = "c_num_fold_ptxref_f32";
    let x = buf(case, "data/x.bin");
    let fp64: Vec<f64> = (0..32).map(|t| x[t] + tenth * 3.0).collect();
    let (analyzed, text) = check_claims(case, &fp64);
    assert!(analyzed.is_empty(), "{case}: claims {analyzed:?}\n{text}");
    assert!(!text.contains("not run"), "{text}");
    assert!(
        text.contains("no PTX claim with the same exact value"),
        "{text}"
    );

    let out = std::env::temp_dir().join(format!("ptxv-test-numv-{case}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let r = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["num", root().join("fixtures").join(case).to_str().unwrap()])
        .args(["--variant", "refused", "--out"])
        .arg(&out)
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&r.stdout).to_string();
    assert!(r.status.success(), "{text}");
    let _ = std::fs::remove_dir_all(&out);
    assert!(
        text.contains("num refused ptx : not run, the PTX front-end refuses it"),
        "{text}"
    );
    assert!(
        text.contains(
            "num refused sass: output=out NOT ANALYZED: the PTX front-end refuses this kernel's PTX"
        ),
        "{text}"
    );
}

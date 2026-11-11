/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The linearized tier's exact sites (N-ERRDIST-T2-OPEN, docs/errdist.md §9): exp2 and exp
//! slices opened into their argument rounding (a bounded uniform), the unit's error and the
//! constant's; and the cross-check's resolution rule for an estimate's own interval.

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run(name: &str, case: &str, spec: &str, mc: &str) -> serde_json::Value {
    let js =
        std::env::temp_dir().join(format!("ptxv-test-open-{name}-{}.json", std::process::id()));
    let scratch = js.with_extension("d");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args([
            "errdist",
            case,
            "--inputs",
            spec,
            "--mc",
            mc,
            "--sim-trials",
            "0",
        ])
        .args(["--json", js.to_str().unwrap()])
        .args(["--out", scratch.to_str().unwrap()])
        .current_dir(root())
        .env("CUDA_VISIBLE_DEVICES", "")
        .env_remove("PTXV_ERRDIST_LIN_TRAJ")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&js).unwrap()).unwrap();
    let _ = std::fs::remove_file(&js);
    let _ = std::fs::remove_dir_all(&scratch);
    j
}

fn estimate<'a>(j: &'a serde_json::Value, prefix: &str) -> &'a serde_json::Value {
    j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["method"].as_str().unwrap().starts_with(prefix))
        .unwrap_or_else(|| panic!("no estimate {prefix}"))
}

fn q999(e: &serde_json::Value) -> f64 {
    e["ulp"]["quantiles"]
        .as_array()
        .unwrap()
        .iter()
        .find(|q| q[0].as_f64() == Some(0.999))
        .unwrap()[1]
        .as_f64()
        .unwrap()
}

/// Naive softmax at N(50, 1): the exp2 argument's f32 rounding is 99.9% of the variance. As a
/// table law (the argument rounding averaged over the cell, normal shape) Tier 2 put P(differ)
/// at 0.978 and the 99.9% quantile at 70 ulps; Tier 3 gives 0.983 and 46 (GB300: 0.983).
#[test]
fn softmax_naive_exp2_argument_rounding_is_a_bounded_site() {
    let j = run(
        "softmax50",
        "fixtures/c_triton_softmax_naive",
        "fixtures/errdist/shift50.json",
        "20000",
    );
    let t3 = estimate(&j, "Monte Carlo");
    let t2 = estimate(&j, "linearized");
    let (p3, p2) = (
        t3["p_differ"][0].as_f64().unwrap(),
        t2["p_differ"][0].as_f64().unwrap(),
    );
    assert!(p3 > 0.98 && p3 < 0.99, "{p3}");
    assert!((p2 - p3).abs() < 0.005 * p3, "tier 2 {p2} vs tier 3 {p3}");
    let (a, b) = (q999(t2), q999(t3));
    assert!(
        (a - b).abs() < 3.0,
        "99.9% quantile: tier 2 {a} vs tier 3 {b}"
    );
    assert!(
        j["notes"].as_array().unwrap().iter().any(|n| n
            .as_str()
            .unwrap()
            .contains("slice exp2 (argument rounding)")),
        "{}",
        j["notes"]
    );
}

/// Stable softmax at x = k / 64 (k uniform on -256..=256): x - max is exact, and the exp
/// slice (exp(x - max) through ex2.approx of the product with log2(e), under the division's
/// range guards on the sum) carries 81% of the variance, most of it the product's f32
/// rounding. As a table law (moments of the cell, normal shape) Tier 2 put P(differ) 2.3% below
/// Tier 3 with its sd 3.6% above; opened, 0.2% above with the sd within 1%.
#[test]
fn softmax_stable_exp_slice_is_opened() {
    let j = run(
        "softmax_grid64",
        "fixtures/c_softmax_stable",
        "fixtures/errdist/grid64.json",
        "20000",
    );
    let t3 = estimate(&j, "Monte Carlo");
    let t2 = estimate(&j, "linearized");
    let (p3, p2) = (
        t3["p_differ"][0].as_f64().unwrap(),
        t2["p_differ"][0].as_f64().unwrap(),
    );
    assert!(p3 > 0.71 && p3 < 0.725, "{p3}");
    assert!((p2 - p3).abs() < 0.01 * p3, "tier 2 {p2} vs tier 3 {p3}");
    assert!(
        j["notes"].as_array().unwrap().iter().any(|n| n
            .as_str()
            .unwrap()
            .contains("slice exp (argument rounding)")),
        "{}",
        j["notes"]
    );
}

/// The f32-accumulator dot product with a bf16 output at N(0, 1): P(differ) about 7e-5. The
/// linearized tier's own interval over its inputs is about +-22%, so its point value is not
/// confirmed even where it falls inside Tier 3's tolerance.
#[test]
fn rare_event_estimate_needs_its_own_resolution() {
    let j = run(
        "dot32",
        "fixtures/c_triton_dot_f32acc",
        "fixtures/errdist/normal.json",
        "400000",
    );
    let t2 = estimate(&j, "linearized");
    assert_eq!(t2["applicable"], false, "{}", t2["applicability"]);
    assert!(
        t2["applicability"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a.as_str().unwrap().starts_with("estimate resolution")),
        "{}",
        t2["applicability"]
    );
}

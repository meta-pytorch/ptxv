/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Log-sum-exp (agent AN's demo): Triton's software log (an integer exponent extraction,
//! I2FP.F32.S32, and a polynomial) is a slice errdist builds and identifies as log.

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run(name: &str, case: &str, extra: &[&str]) -> (String, serde_json::Value) {
    let js = std::env::temp_dir().join(format!("ptxv-test-lse-{name}-{}.json", std::process::id()));
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["errdist", case, "--inputs", "fixtures/errdist/normal.json"])
        .args(extra)
        .args(["--json", js.to_str().unwrap()])
        .current_dir(root())
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "{text}");
    let j = serde_json::from_str(&std::fs::read_to_string(&js).unwrap()).unwrap();
    let _ = std::fs::remove_file(&js);
    (text, j)
}

fn p(j: &serde_json::Value, tier: u64) -> f64 {
    j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["tier"] == tier)
        .unwrap_or_else(|| panic!("no tier {tier}: {j}"))["p_differ"][0]
        .as_f64()
        .unwrap()
}

/// Since MISS 99 the log's range reduction (the argument's exponent, an integer step) is a
/// decision of the slice: 0.4% of the Tier 3 inputs take an exponent the identification's probes
/// never took, a share whose bound is within 10% of P at 4,000 inputs (at 1,000 it is not, and
/// errdist refuses).
#[test]
fn naive_lse_log_slice() {
    let (text, j) = run("naive", "fixtures/c_lse_naive", &["--mc", "4000"]);
    assert!(text.contains(": log("), "{text}");
    // Tier 3 ~ 0.071; the linearized tier 2 ~ 0.073.
    let (p3, p2) = (p(&j, 3), p(&j, 2));
    assert!(p3 > 0.04 && p3 < 0.11, "{p3}");
    assert!((p2 - p3).abs() < 0.2 * p3, "{p2} vs {p3}");
}

/// Captured-zero max shifts retain their input dependence without a repair retry, and the DAG
/// matches the simulator. Without Tier 3 (`--mc 0`): since MISS 99, 240 of 1,000 Tier 3 inputs
/// take exponents of the log's argument that the identification's probes never took (the sum
/// of the shifted exponentials spans more binades than the perturbed captured input), and
/// errdist refuses to report P(differ) (a precision cost; GB300's reference is 0.2084).
#[test]
fn stable_lse_keeps_max_shift_history() {
    let args = ["--mc", "0", "--sim-trials", "4"];
    let (text, j) = run("stable", "fixtures/c_lse_stable", &args);
    assert!(!text.contains("retrying in segment mode"), "{text}");
    assert!(text.contains(": log("), "{text}");
    let v = j["validation"].as_array().unwrap();
    assert!(!v.is_empty(), "{text}");
    for v in v {
        assert_eq!(v["dag_mismatch"], 0, "{v}");
    }
}

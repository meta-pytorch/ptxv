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

fn run(name: &str, case: &str) -> (String, serde_json::Value) {
    let js = std::env::temp_dir().join(format!("ptxv-test-lse-{name}-{}.json", std::process::id()));
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args([
            "errdist",
            case,
            "--inputs",
            "fixtures/errdist/normal.json",
            "--mc",
            "1000",
        ])
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

#[test]
fn naive_lse_log_slice() {
    let (text, j) = run("naive", "fixtures/c_lse_naive");
    assert!(text.contains(": log("), "{text}");
    // Tier 3 ~ 0.071; the linearized tier 2 ~ 0.073.
    let (p3, p2) = (p(&j, 3), p(&j, 2));
    assert!(p3 > 0.04 && p3 < 0.11, "{p3}");
    assert!((p2 - p3).abs() < 0.2 * p3, "{p2} vs {p3}");
}

#[test]
fn stable_lse_reextracts() {
    // The default extraction folds the max-shifted exp's argument; segment mode is right.
    let (text, j) = run("stable", "fixtures/c_lse_stable");
    assert!(text.contains("retrying in segment mode"), "{text}");
    assert!(text.contains(": log("), "{text}");
    for v in j["validation"].as_array().unwrap() {
        assert_eq!(v["dag_mismatch"], 0, "{v}");
    }
}

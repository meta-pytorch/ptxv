/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! errdist's guards against reporting from bad data (docs/errdist.md §1): NaN / Inf outputs
//! are out-of-domain mass, never dropped silently; a numerics DAG that disagrees with the
//! simulator is re-extracted in segment mode, and refused if that disagrees too.

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run(name: &str, path: &str, spec: &str) -> (String, serde_json::Value) {
    let js = std::env::temp_dir().join(format!(
        "ptxv-test-guard-{name}-{}.json",
        std::process::id()
    ));
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["errdist", path, "--inputs", spec, "--mc", "200"])
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

/// Row sums of N(3e37, 1e36) f32 inputs overflow: every output is Inf.
#[test]
fn nonfinite_outputs_are_out_of_domain() {
    let spec =
        std::env::temp_dir().join(format!("ptxv-test-guard-huge-{}.json", std::process::id()));
    std::fs::write(
        &spec,
        r#"{"version": 1, "seed": 1, "name": "huge", "default": {"dist": {"kind": "normal", "mean": 3e37, "std": 1e36}}}"#,
    )
    .unwrap();
    let (text, j) = run("inf", "fixtures/r4_sum_torch", spec.to_str().unwrap());
    let _ = std::fs::remove_file(&spec);
    let ood = &j["out_of_domain"];
    assert!(ood["mass"].as_f64().unwrap() > 0.99, "{ood}");
    assert!(
        ood["by_cause"]["non-finite output (NaN / Inf)"]
            .as_f64()
            .unwrap()
            > 0.99,
        "{ood}"
    );
    assert!(text.contains("TIER 3 COMPARED NO OUTPUTS"), "{text}");
    // The simulator's outputs are Inf too: the validation says so and does not agree.
    let v = j["validation"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["source"] == "sim")
        .unwrap();
    assert_eq!(v["agree"], false, "{v}");
}

/// Kahan compensation stays input-dependent even when captured as zero. The extracted DAG
/// must reproduce sampled simulator outputs without needing to repair a lost dependency.
#[test]
fn kahan_compensation_keeps_input_history() {
    let (text, j) = run(
        "kahan",
        "fixtures/c_sum_kahan",
        "fixtures/errdist/normal.json",
    );
    assert!(!text.contains("retrying in segment mode"), "{text}");
    for v in j["validation"].as_array().unwrap() {
        assert_eq!(v["dag_mismatch"], 0, "{v}");
    }
}

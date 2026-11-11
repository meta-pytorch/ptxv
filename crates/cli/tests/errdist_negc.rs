/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The DAG CLT (`errdist/tier2.rs`) on an FFMA with a negated addend: the addend's mean is
//! subtracted (fixture fixtures/errdist/negc, where the two sides nearly cancel).

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn dag_clt_subtracts_a_negated_addend() {
    let js = std::env::temp_dir().join(format!("ptxv-test-negc-{}.json", std::process::id()));
    let scratch = js.with_extension("d");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["errdist", "fixtures/errdist/negc"])
        .args(["--inputs", "fixtures/errdist/negc.json"])
        .args(["--mc", "4000", "--sim-trials", "0"])
        .args(["--json", js.to_str().unwrap()])
        .args(["--out", scratch.to_str().unwrap()])
        .current_dir(root())
        .env("CUDA_VISIBLE_DEVICES", "")
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
    let est = j["estimates"].as_array().unwrap();
    let i = est
        .iter()
        .position(|e| {
            e["method"]
                .as_str()
                .unwrap()
                .starts_with("CLT on the output's DAG")
        })
        .expect("DAG CLT estimate");
    let x = j["crosscheck"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["estimate"].as_u64() == Some(i as u64))
        .expect("DAG CLT cross-check");
    // Tier 3 is about 0.90; with the addend's mean added the DAG CLT gave 0.22.
    assert_eq!(x["agrees"], true, "{x}");
    let (p, p3) = (x["p"].as_f64().unwrap(), x["p3"].as_f64().unwrap());
    assert!((p - p3).abs() < 0.05 * p3, "DAG CLT {p} vs Tier 3 {p3}");
}

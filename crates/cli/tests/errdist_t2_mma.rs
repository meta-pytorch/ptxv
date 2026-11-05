/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tier 2's DAG CLT on a long bf16 MMA chain pair (K = 960, forward against reversed K): the
//! round-toward-zero means of the steps are correlated through the partial sums
//! (docs/errdist.md §5.3); without that term tier 2 was 0.3x of tier 3 here, and the gap grew
//! with K.

fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn correlated_rounding_means_on_a_long_mma_pair() {
    let js = std::env::temp_dir().join(format!("ptxv-test-t2mma-{}.json", std::process::id()));
    let scratch = js.with_extension("d");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args([
            "errdist",
            "fixtures/c_equiv_mm_k960/equiv.toml",
            "--inputs",
            "fixtures/errdist/normal.json",
            "--mc",
            "500",
            "--no-tier1",
            "--sim-trials",
            "0",
        ])
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
    let p3 = est.iter().find(|e| e["tier"] == 3).unwrap()["p_differ"][0]
        .as_f64()
        .unwrap();
    let t2 = est
        .iter()
        .find(|e| {
            e["tier"] == 2
                && e["method"]
                    .as_str()
                    .unwrap()
                    .starts_with("CLT on both DAGs")
        })
        .unwrap();
    let p2 = t2["p_differ"][0].as_f64().unwrap();
    // Tier 3 ~ 9e-4 (GB300 9.6e-4); tier 2 1.0e-3.
    assert!(p3 > 5e-4 && p3 < 1.5e-3, "{p3}");
    assert!(p2 / p3 > 0.8 && p2 / p3 < 1.3, "{p2} vs {p3}");
    // The two kernels' errors against the exact value are correlated (tier 3: 0.54).
    let corr = t2["pair"]["corr_rc"].as_f64().unwrap();
    assert!(corr > 0.3 && corr < 0.8, "{corr}");
}

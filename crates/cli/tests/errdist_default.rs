/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The default errdist policy (docs/errdist.md §3.2): tier 0, else tier 2 where its
//! applicability check and the Tier 3 cross-check (300 inputs by default) pass, else tier 3;
//! tier 1 only with --tier1.

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Third-party kernel binaries are not distributed (fixtures/VENDOR_FIXTURES.toml): true when
/// one these fixtures need is missing (the test then skips).
fn vendor_missing(needs: &[&str]) -> bool {
    ptxv_case::skip_missing_vendor_files(&root().join("fixtures"), needs)
}

fn run(name: &str, path: &str, spec: &str, extra: &[&str]) -> serde_json::Value {
    let js =
        std::env::temp_dir().join(format!("ptxv-test-dflt-{name}-{}.json", std::process::id()));
    let scratch = js.with_extension("d");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["errdist", path, "--inputs", spec, "--sim-trials", "0"])
        .args(["--json", js.to_str().unwrap()])
        .args(["--out", scratch.to_str().unwrap()])
        .args(extra)
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
    j
}

fn tiers(j: &serde_json::Value) -> Vec<u64> {
    j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["tier"].as_u64().unwrap())
        .collect()
}

fn best_tier(j: &serde_json::Value) -> u64 {
    j["estimates"][j["best"].as_u64().unwrap() as usize]["tier"]
        .as_u64()
        .unwrap()
}

#[test]
fn tier2_headline_when_it_agrees_with_the_cross_check() {
    let j = run(
        "sum",
        "fixtures/r4_sum_torch",
        "fixtures/errdist/normal.json",
        &[],
    );
    assert_eq!(best_tier(&j), 2);
    assert!(!tiers(&j).contains(&1), "tier 1 is opt-in: {:?}", tiers(&j));
    let n3 = j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["tier"] == 3)
        .unwrap()["n"]
        .as_u64()
        .unwrap();
    // 300 inputs to start, grown toward a 2% interval while cheap (P(differ) ~0.67).
    assert!(n3 > 300, "the cross-check grows: {n3}");
    let cc = j["crosscheck"].as_array().unwrap();
    assert!(
        !cc.is_empty() && cc.iter().all(|c| c["agrees"] == true),
        "{cc:?}"
    );
}

#[test]
fn heavy_tails_refuse_tier2_and_fall_back_to_tier3() {
    if vendor_missing(&["c_mm_bf16_nnt"]) {
        return;
    }
    // 2% N(0, 64^2) outliers: y's excess kurtosis ~ 300; tier 2 would say 8.7e-5 against
    // tier 3's 3.2e-5.
    let j = run(
        "outl",
        "fixtures/c_mm_bf16_nnt",
        "fixtures/errdist/outliers.json",
        &[],
    );
    assert_eq!(best_tier(&j), 3);
    let t2 = j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["tier"] == 2)
        .unwrap();
    assert_eq!(t2["applicable"], false);
    let why = t2["applicability"].to_string();
    assert!(why.contains("kurtosis"), "{why}");
}

#[test]
fn tier1_is_opt_in() {
    if vendor_missing(&["c_mm_bf16_nnt"]) {
        return;
    }
    let j = run(
        "t1",
        "fixtures/c_mm_bf16_nnt",
        "fixtures/errdist/normal.json",
        &["--tier1", "--mc", "100"],
    );
    assert!(tiers(&j).contains(&1), "{:?}", tiers(&j));
}

fn tier_est(j: &serde_json::Value, t: u64) -> &serde_json::Value {
    j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["tier"] == t)
        .unwrap_or_else(|| panic!("no tier {t}"))
}

#[test]
fn cross_check_grows_tier3_for_rare_events() {
    if vendor_missing(&["c_mm_bf16_nnt"]) {
        return;
    }
    // bf16 GEMM, normal inputs: P(differ) ~2.6e-5. At 300 inputs tier 3's interval is
    // +-30% and tier 2 (0.82x) passed; tier 3 grows to ~5000 inputs (+-10%) and refuses it.
    let j = run(
        "rare",
        "fixtures/c_mm_bf16_nnt",
        "fixtures/errdist/normal.json",
        &[],
    );
    assert!(tier_est(&j, 3)["n"].as_u64().unwrap() > 1000);
    assert_eq!(tier_est(&j, 2)["applicable"], false);
    assert_eq!(best_tier(&j), 3);
    // An explicit --mc keeps its input count; a tolerance above 20% of P(differ) does not
    // confirm the estimate.
    let j = run(
        "rare300",
        "fixtures/c_mm_bf16_nnt",
        "fixtures/errdist/normal.json",
        &["--mc", "300"],
    );
    assert_eq!(tier_est(&j, 3)["n"].as_u64().unwrap(), 300);
    let t2 = tier_est(&j, 2);
    assert_eq!(t2["applicable"], false);
    assert!(
        t2["applicability"].to_string().contains("resolution"),
        "{}",
        t2["applicability"]
    );
}

#[test]
fn atomic_pair_runs() {
    // The GPU ran its own order of the atomic adds: no bit-for-bit check against the capture.
    let j = run(
        "atomic",
        "fixtures/c_equiv_sum/equiv.toml",
        "fixtures/errdist/normal.json",
        &["--candidate", "2"],
    );
    assert!(tiers(&j).contains(&3), "{:?}", tiers(&j));
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tier 2 on block-scaled MMA chains (`tier2bs`, docs/errdist.md §5.3.1): applicable on
//! MXF8 under normal elements, refused by y's excess kurtosis under outliers (where it would
//! be 2.54x of Tier 3), and not run on pairs.

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Third-party kernel binaries are not distributed (fixtures/VENDOR_FIXTURES.toml).
fn vendor_missing(needs: &[&str]) -> bool {
    ptxv_case::skip_missing_vendor_files(&root().join("fixtures"), needs)
}

fn run(name: &str, args: &[&str]) -> serde_json::Value {
    let js =
        std::env::temp_dir().join(format!("ptxv-test-t2bs-{name}-{}.json", std::process::id()));
    let scratch = js.with_extension("d");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .arg("errdist")
        .args(args)
        .args(["--sim-trials", "0"])
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
    j
}

fn bs(j: &serde_json::Value) -> Option<&serde_json::Value> {
    j["estimates"].as_array().unwrap().iter().find(|e| {
        e["tier"] == 2
            && e["method"]
                .as_str()
                .unwrap()
                .starts_with("CLT on the MMA chain")
    })
}

fn p(e: &serde_json::Value) -> f64 {
    e["p_differ"][0].as_f64().unwrap()
}

/// tier2bs is 0.966x of Tier 3 here (§5.3.1: against a 20,000-input Tier 3), and its
/// cross-check takes 5% of p3 less Tier 3's half-width (MISS 91). Since MISS 139 Tier 3's
/// interval takes the input-level bound at any count: the default growth stops at 7033 inputs
/// (the 30 s budget) with a half-width of 1.9% of P, and the tolerance, 3.1%, is below
/// tier2bs's 3.5%, so the headline falls back to Tier 3 there (until MISS 139, 300 inputs and a
/// half-width of 1.5% over outputs left 3.5%: 1.50e-3 within 1.58e-3). At 20,000 inputs the
/// half-width is 0.7% and tier2bs stands.
#[test]
#[ignore = "heavy: 70 s"]
fn mxf8_normal_is_applicable() {
    if vendor_missing(&["r3_cute_mxf8"]) {
        return;
    }
    let j = run(
        "normal",
        &[
            "fixtures/r3_cute_mxf8",
            "--inputs",
            "fixtures/errdist/mxf8_gauss.json",
            "--mc",
            "20000",
        ],
    );
    let e = bs(&j).expect("tier2bs estimate");
    assert_eq!(e["applicable"], true, "{}", e["applicability"]);
    // 0.966x of a 20000-input Tier 3 (4.50e-2).
    let p3 = p(j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["tier"] == 3)
        .unwrap());
    let r = p(e) / p3;
    assert!((0.9..1.1).contains(&r), "tier2bs / tier 3 = {r}");
    assert!(
        j["assumptions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a == "N-ERRDIST-BSCLT")
    );
}

#[test]
#[ignore = "heavy: errdist Tier 3 subprocess; 251 s"]
fn mxf8_outliers_are_refused_by_the_gate() {
    if vendor_missing(&["r3_cute_mxf8"]) {
        return;
    }
    let j = run(
        "outl",
        &[
            "fixtures/r3_cute_mxf8",
            "--inputs",
            "fixtures/errdist/mxf8_outliers.json",
            "--mc",
            "100",
        ],
    );
    let e = bs(&j).expect("tier2bs estimate");
    assert_eq!(e["applicable"], false);
    let why = e["applicability"].to_string();
    assert!(why.contains("kurtosis") && why.contains("FAILS"), "{why}");
    assert_eq!(
        j["estimates"][j["best"].as_u64().unwrap() as usize]["tier"],
        3
    );
}

#[test]
#[ignore = "heavy: 66 s"]
fn mxf8_pairs_are_not_run() {
    if vendor_missing(&["r3_cute_mxf8"]) {
        return;
    }
    let j = run(
        "pair",
        &[
            "fixtures/c_equiv_mxf8_triton/equiv.toml",
            "--candidate",
            "1",
            "--inputs",
            "fixtures/errdist/mxf8_gauss.json",
            "--mc",
            "100",
        ],
    );
    assert!(bs(&j).is_none());
    let notes = j["notes"].to_string();
    assert!(notes.contains("pairs of block-scaled kernels"), "{notes}");
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A verdict is certified only when it is complete: ptxv also lists E-WELLSYNC among the
//! assumptions of incomplete verdicts (smid_slot: every enumerated %smid placement certified,
//! not every placement enumerated), and the JSON's certified flags (`two_level.sass_certified`,
//! `mutated_inputs[].certified`, compare's `check.certified`) and the report say so.

use std::path::PathBuf;

#[test]
fn an_incomplete_verdict_listing_e_wellsync_is_not_certified() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::temp_dir().join(format!("ptxv-certified-complete-{}", std::process::id()));
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .arg("check")
        .arg(root.join("fixtures/smid_slot"))
        .args([
            "--variant",
            "good",
            "--front",
            "sass",
            "--certify",
            "--json",
            "--no-input-coverage",
            "--out",
        ])
        .arg(&dir)
        .env_remove("PTXV_GB300")
        .env_remove("PTXV_HW_PROFILE")
        .env_remove("PTXV_TIME_BUDGET")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let report = String::from_utf8_lossy(&out.stderr).into_owned();
    let doc: serde_json::Value =
        serde_json::from_slice(&out.stdout).unwrap_or_else(|e| panic!("{e}: {report}"));
    let row = &doc["rows"][0];
    // The shape: incomplete, E-WELLSYNC among the assumptions, at both levels.
    assert_eq!(row["verdict"]["complete"], false, "{report}");
    let wellsync = |v: &serde_json::Value| {
        v.as_array()
            .is_some_and(|a| a.iter().any(|x| x == "E-WELLSYNC"))
    };
    assert!(wellsync(&row["verdict"]["assumptions"]), "{report}");
    let tl = &row["two_level"];
    assert_eq!(tl["sass_complete"], false, "{report}");
    assert_eq!(tl["sass_certified"], false, "{tl}");
    assert!(
        report
            .lines()
            .any(|l| l.contains("sass_gb300:") && l.contains("states, incomplete)")),
        "{report}"
    );
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv check --json` names the schedule explorer that produced its verdicts (`explorer`):
//! "dpor" under `PTXV_EXPLORER=dpor`, "stateful" otherwise, so a DPOR verdict cannot pass for
//! a default one (fb/kernel_db run.py `parse_check` refuses anything but "stateful").

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn explorer(env: Option<&str>) -> String {
    let case = root().join("fixtures/e_dsmem_exit");
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"));
    cmd.args([
        "check",
        case.to_str().unwrap(),
        "--variant",
        "good",
        "--front",
        "ptx",
        "--no-gb300",
        "--json",
    ])
    .env("CUDA_VISIBLE_DEVICES", "")
    .env_remove("PTXV_EXPLORER")
    .current_dir(std::env::temp_dir());
    if let Some(e) = env {
        cmd.env("PTXV_EXPLORER", e);
    }
    let out = cmd.output().unwrap();
    let doc: serde_json::Value = serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&out.stderr)));
    assert_eq!(doc["rows"][0]["got"], serde_json::json!(["clean"]), "{doc}");
    doc["explorer"].as_str().unwrap_or("<missing>").to_string()
}

#[test]
fn check_json_names_the_explorer() {
    assert_eq!(explorer(None), "stateful");
    assert_eq!(explorer(Some("dpor")), "dpor");
    // Any other value runs the stateful explorer, and says so.
    assert_eq!(explorer(Some("stateful")), "stateful");
    assert_eq!(explorer(Some("bogus")), "stateful");
}

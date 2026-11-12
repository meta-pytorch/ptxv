/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `--no-gb300` / `PTXV_GB300=0` turn off the GB300-profile re-run of the two-level verdict
//! (ASSUMPTIONS.md H-PROFILE) as well as the hardware-model classification, and the row says
//! the re-run was skipped (`two_level_skipped`), naming the flag or variable that did it; by
//! default the re-run runs. `--hw-profile none` / `PTXV_HW_PROFILE=none` turn the re-run off
//! the same way; any other name than `gb300` or `none` is an error, for `check` and `suite`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

/// `ptxv check fixtures/e_twolevel_bulk --variant good --front sass --certify --json` with
/// `args` and `env`: the JSON document and the report (stderr with --json).
fn check(args: &[&str], env: &[(&str, &str)]) -> (serde_json::Value, String) {
    let out = run(args, env);
    let report = String::from_utf8_lossy(&out.stderr).into_owned();
    let doc = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| panic!("{e}: {report}"));
    (doc, report)
}

/// The check's process output (a directory of its own for every run: the tests run at once).
fn run(args: &[&str], env: &[(&str, &str)]) -> std::process::Output {
    static RUNS: AtomicUsize = AtomicUsize::new(0);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::temp_dir().join(format!(
        "ptxv-two-level-gate-{}-{}",
        std::process::id(),
        RUNS.fetch_add(1, Ordering::Relaxed)
    ));
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .arg("check")
        .arg(root.join("fixtures/e_twolevel_bulk"))
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
        .args(args)
        .env_remove("PTXV_GB300")
        .env_remove("PTXV_HW_PROFILE")
        .envs(env.iter().copied())
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    out
}

#[test]
fn no_gb300_skips_the_profile_rerun() {
    // Default: the strict verdict has a contract finding (E-BULK-EXIT), so the SASS front is
    // checked again under the GB300 profile, which certifies it.
    let (doc, report) = check(&[], &[]);
    let row = &doc["rows"][0];
    assert_eq!(row["got"], serde_json::json!(["contract"]), "{report}");
    assert_eq!(row["two_level"]["sass_gb300"], "certified", "{report}");
    assert!(row.get("two_level_skipped").is_none(), "{report}");
    assert!(report.contains("sass_gb300: certified"), "{report}");
    for (args, env, cause) in [
        (&["--no-gb300"][..], &[][..], "--no-gb300 turned"),
        (&[][..], &[("PTXV_GB300", "0")][..], "PTXV_GB300=0 turned"),
    ] {
        let (doc, report) = check(args, env);
        let row = &doc["rows"][0];
        // The strict verdict is the same; the profile level is not checked, and says so.
        assert_eq!(
            row["got"],
            serde_json::json!(["contract"]),
            "{args:?} {env:?}"
        );
        assert!(row.get("two_level").is_none(), "{args:?} {env:?}: {report}");
        let why = row["two_level_skipped"].as_str().unwrap_or_default();
        assert!(
            why.contains("contract") && why.contains(cause),
            "{args:?} {env:?}: {why:?}"
        );
        assert!(
            report.contains("sass_gb300: not checked"),
            "{args:?} {env:?}: {report}"
        );
    }
}

/// `--hw-profile none` / `PTXV_HW_PROFILE=none` turn the GB300-profile re-run off as
/// `--no-gb300` does, and the row says so (`two_level_skipped`). Before, the row had neither
/// `two_level` nor `two_level_skipped`, and a missing `two_level` read like an older binary's.
#[test]
fn hw_profile_none_says_the_rerun_was_skipped() {
    for (args, env, cause) in [
        (
            &["--hw-profile", "none"][..],
            &[][..],
            "; --hw-profile none turned",
        ),
        (
            &[][..],
            &[("PTXV_HW_PROFILE", "none")][..],
            "; PTXV_HW_PROFILE=none turned",
        ),
    ] {
        let (doc, report) = check(args, env);
        let row = &doc["rows"][0];
        assert_eq!(
            row["got"],
            serde_json::json!(["contract"]),
            "{args:?} {env:?}"
        );
        assert!(row.get("two_level").is_none(), "{args:?} {env:?}: {report}");
        let why = row["two_level_skipped"].as_str().unwrap_or_default();
        assert!(
            why.contains("contract") && why.contains(cause),
            "{args:?} {env:?}: {why:?}"
        );
        assert!(
            report.contains("sass_gb300: not checked"),
            "{args:?} {env:?}: {report}"
        );
    }
}

/// A profile name other than `gb300` or `none` is an error; it used to turn the re-run off
/// without a word (a typo such as `GB300`).
#[test]
fn unknown_hw_profiles_are_refused() {
    for (args, env, says) in [
        (
            &["--hw-profile", "GB300"][..],
            &[][..],
            "unknown hardware profile \"GB300\" (--hw-profile)",
        ),
        (
            &[][..],
            &[("PTXV_HW_PROFILE", "gb3000")][..],
            "unknown hardware profile \"gb3000\" (PTXV_HW_PROFILE)",
        ),
    ] {
        let out = run(args, env);
        let text = String::from_utf8_lossy(&out.stdout).into_owned()
            + &String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "{args:?} {env:?}: {text}");
        assert!(text.contains(says), "{args:?} {env:?}: {text}");
    }
}

/// `ptxv suite` checks through the same path as `check`: an unknown profile name is refused
/// there too (it used to reach the check with no profile, which reported the re-run as turned
/// off by "--hw-profile none").
#[test]
fn suite_refuses_unknown_hw_profiles() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir =
        std::env::temp_dir().join(format!("ptxv-two-level-gate-suite-{}", std::process::id()));
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .arg("suite")
        .arg(root.join("fixtures/e_twolevel_bulk"))
        .arg("--out")
        .arg(&dir)
        .env_remove("PTXV_GB300")
        .env("PTXV_HW_PROFILE", "GB300")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{text}");
    assert!(
        text.contains("unknown hardware profile \"GB300\" (PTXV_HW_PROFILE)"),
        "{text}"
    );
    assert!(!text.contains("sass_gb300"), "{text}");
}

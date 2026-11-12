/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `--time-budget S` / `PTXV_TIME_BUDGET`: one wall-clock budget for the whole check. Every
//! phase it cuts leaves the verdict incomplete with a note, a cut certificate never holds, the
//! GB300-profile re-run does not start once the budget is spent (`two_level_skipped`) and
//! states the budget it starts with otherwise.

use std::io::Read;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

/// `ptxv check fixtures/<fixture> --variant <variant> --front sass --certify --json` with
/// `args` and `env`: the exit status, the JSON document (Null when there is none) and the
/// report (stderr with --json).
fn check(
    fixture: &str,
    variant: &str,
    args: &[&str],
    env: &[(&str, &str)],
) -> (bool, serde_json::Value, String) {
    check_within(fixture, variant, args, env, Duration::from_secs(1800))
}

/// `check`, killed after `limit` (a failure whose report says so).
fn check_within(
    fixture: &str,
    variant: &str,
    args: &[&str],
    env: &[(&str, &str)],
    limit: Duration,
) -> (bool, serde_json::Value, String) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::temp_dir().join(format!(
        "ptxv-time-budget-{}-{fixture}-{variant}-{}",
        std::process::id(),
        args.join("_").replace(['-', '.'], "") + &env.len().to_string()
    ));
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .arg("check")
        .arg(root.join("fixtures").join(fixture))
        .args([
            "--variant",
            variant,
            "--front",
            "sass",
            "--certify",
            "--json",
            "--no-input-coverage",
            "--out",
        ])
        .arg(&dir)
        .args(args)
        .env_remove("PTXV_TIME_BUDGET")
        .env_remove("PTXV_GB300")
        .env_remove("PTXV_HW_PROFILE")
        .env_remove("PTXV_DIST_DIR")
        .env_remove("PTXV_RANK")
        .env_remove("PTXV_RANKS")
        .env_remove("SLURM_PROCID")
        .env_remove("SLURM_NTASKS")
        .envs(env.iter().copied())
        .env("CUDA_VISIBLE_DEVICES", "")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let read = |mut r: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut b = Vec::new();
            let _ = r.read_to_end(&mut b);
            b
        })
    };
    let stdout = read(Box::new(child.stdout.take().unwrap()));
    let stderr = read(Box::new(child.stderr.take().unwrap()));
    let t0 = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break Some(s);
        }
        if t0.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let mut report = String::from_utf8_lossy(&stderr.join().unwrap()).into_owned();
    if status.is_none() {
        report.push_str(&format!("\n(killed after {limit:?})"));
    }
    let doc = serde_json::from_slice(&stdout.join().unwrap()).unwrap_or(serde_json::Value::Null);
    let _ = std::fs::remove_dir_all(&dir);
    (status.is_some_and(|s| s.success()), doc, report)
}

fn notes(row: &serde_json::Value) -> Vec<String> {
    row["verdict"]["notes"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|n| n.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn spent_budget_cuts_every_phase() {
    for (args, env) in [
        (&["--time-budget", "0"][..], &[][..]),
        (&[][..], &[("PTXV_TIME_BUDGET", "0")][..]),
    ] {
        // e_sb_mufu NOWAIT: a scoreboard violation on the first steps, so the strict verdict
        // wants the GB300-profile re-run even when everything else is cut.
        let (_, doc, report) = check("e_sb_mufu", "NOWAIT", args, env);
        let row = &doc["rows"][0];
        let ns = notes(row);
        let has = |s: &str| ns.iter().any(|n| n.contains(s));
        assert_eq!(row["got"], serde_json::json!(["violation"]), "{report}");
        assert_eq!(row["verdict"]["complete"], false, "{report}");
        assert!(
            has(
                "certificate failed (the certificate's run stopped at the time budget (--time-budget) after 0 steps)"
            ),
            "{env:?} {ns:?}"
        );
        assert!(
            has("exploration stopped at the time budget (--time-budget): incomplete"),
            "{ns:?}"
        );
        assert!(
            has("seed walks stopped at the time budget (--time-budget)"),
            "{ns:?}"
        );
        assert!(
            has("one deterministic run (stopped at the time budget (--time-budget))"),
            "{ns:?}"
        );
        assert!(row.get("two_level").is_none(), "{report}");
        let why = row["two_level_skipped"].as_str().unwrap_or_default();
        assert!(
            why.contains("violation")
                && why.contains(
                    "the time budget (--time-budget) was spent before the GB300-profile re-run"
                ),
            "{why:?}"
        );
        assert!(report.contains("sass_gb300: not checked"), "{report}");
        assert!(
            report.contains("time budget: 0 s for the whole check"),
            "{report}"
        );
    }
    // e_twolevel_bulk good certifies without a budget (two_level_gate.rs); with a spent one
    // the certificate does not hold and nothing is complete.
    let (_, doc, report) = check("e_twolevel_bulk", "good", &["--time-budget", "0"], &[]);
    let row = &doc["rows"][0];
    assert_eq!(row["verdict"]["complete"], false, "{report}");
    assert_ne!(row["got"], serde_json::json!(["clean"]), "{report}");
    let certified = row["verdict"]["assumptions"]
        .as_array()
        .is_some_and(|a| a.iter().any(|x| x == "E-WELLSYNC"));
    assert!(!certified, "{report}");
}

#[test]
fn rerun_states_the_budget_left() {
    let (_, doc, report) = check("e_twolevel_bulk", "good", &["--time-budget", "3600"], &[]);
    let row = &doc["rows"][0];
    // The strict verdict (E-BULK-EXIT contract) and the re-run's are those without a budget.
    assert_eq!(row["got"], serde_json::json!(["contract"]), "{report}");
    assert_eq!(row["two_level"]["sass_gb300"], "certified", "{report}");
    let left = row["two_level"]["budget_left_seconds"]
        .as_f64()
        .unwrap_or(-1.0);
    assert!(left > 0.0 && left <= 3600.0, "{left}: {report}");
    assert!(
        report.contains("GB300 profile: re-run with") && report.contains("of the time budget left"),
        "{report}"
    );
    // Without a budget the field is absent.
    let (_, doc, _) = check("e_twolevel_bulk", "good", &[], &[]);
    assert!(
        doc["rows"][0]["two_level"]
            .get("budget_left_seconds")
            .is_none()
    );
}

#[test]
fn bad_budgets_are_errors() {
    for (args, env) in [
        (&[][..], &[("PTXV_TIME_BUDGET", "-1")][..]),
        (&["--time-budget", "inf"][..], &[][..]),
        (&[][..], &[("PTXV_TIME_BUDGET", "soon")][..]),
    ] {
        let (ok, _, report) = check("e_twolevel_bulk", "good", args, env);
        assert!(!ok, "{args:?} {env:?}: {report}");
        assert!(
            report.contains("--time-budget") || report.contains("PTXV_TIME_BUDGET"),
            "{report}"
        );
    }
}

/// Multi-node mode (PTXV_DIST_DIR, more than one rank): every rank decided its own cuts, so the
/// ranks ended up in different phases and waited for each other (a hang, or a panic in the
/// exploration). The budget is refused before any rank connects (flag and environment); one
/// rank (PTXV_RANKS=1) is not multi-node and keeps it.
#[test]
fn multi_node_refuses_the_time_budget() {
    let dist = std::env::temp_dir().join(format!("ptxv-time-budget-dist-{}", std::process::id()));
    std::fs::create_dir_all(&dist).unwrap();
    let d = dist.to_str().unwrap();
    for (args, env) in [
        (
            &["--time-budget", "2"][..],
            &[("PTXV_RANK", "0"), ("PTXV_RANKS", "2")][..],
        ),
        (
            &[][..],
            &[
                ("PTXV_TIME_BUDGET", "2"),
                ("PTXV_RANK", "1"),
                ("PTXV_RANKS", "2"),
            ][..],
        ),
        (
            &["--time-budget", "2"][..],
            &[("SLURM_PROCID", "0"), ("SLURM_NTASKS", "4")][..],
        ),
    ] {
        let env: Vec<(&str, &str)> = env.iter().copied().chain([("PTXV_DIST_DIR", d)]).collect();
        // Before the fix rank 0 waited for the other ranks to connect, forever.
        let t0 = Instant::now();
        let (ok, _, report) = check_within(
            "e_cert_fasttrack",
            "good",
            args,
            &env,
            Duration::from_secs(60),
        );
        assert!(!ok, "{args:?} {env:?}: {report}");
        assert!(
            report.contains("not supported in multi-node mode (PTXV_DIST_DIR)"),
            "{report}"
        );
        assert!(
            t0.elapsed().as_secs() < 30,
            "refused only after {:?}",
            t0.elapsed()
        );
    }
    let env = [
        ("PTXV_DIST_DIR", d),
        ("PTXV_RANK", "0"),
        ("PTXV_RANKS", "1"),
    ];
    let (ok, doc, report) = check("e_cert_fasttrack", "good", &["--time-budget", "3600"], &env);
    let _ = std::fs::remove_dir_all(&dist);
    assert!(ok, "{report}");
    assert_eq!(
        doc["rows"][0]["got"],
        serde_json::json!(["clean"]),
        "{report}"
    );
}

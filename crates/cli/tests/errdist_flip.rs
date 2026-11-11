/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Branch flips at selections and maxima (N-ERRDIST-SEL, docs/errdist.md §3.2): Tier 3 counts
//! the inputs where the kernel passes on another child than the exact computation; the
//! linearized tiers, which follow one branch, stand only while the 95% upper bound on that
//! share is within 10% of P, with their P(differ) interval widened by it.

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run(name: &str, case: &str, spec: &str, mc: Option<&str>) -> serde_json::Value {
    run_env(name, case, spec, mc, &[])
}

fn run_env(
    name: &str,
    case: &str,
    spec: &str,
    mc: Option<&str>,
    env: &[(&str, &str)],
) -> serde_json::Value {
    let js =
        std::env::temp_dir().join(format!("ptxv-test-flip-{name}-{}.json", std::process::id()));
    let scratch = js.with_extension("d");
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"));
    cmd.args(["errdist", case, "--inputs", spec, "--sim-trials", "0"]);
    if let Some(n) = mc {
        cmd.args(["--mc", n]);
    }
    cmd.env_remove("PTXV_ERRDIST_LIN_TRAJ")
        .env_remove("PTXV_XCHECK_SECONDS");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd
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

/// The linearized estimates and their branch-flip applicability lines.
fn gates(j: &serde_json::Value) -> Vec<(bool, String)> {
    j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["method"].as_str().unwrap().starts_with("linearized"))
        .map(|e| {
            let line = e["applicability"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|a| a.as_str().unwrap().to_string())
                .find(|a| a.starts_with("branch flips"))
                .unwrap_or_default();
            (e["applicable"].as_bool().unwrap_or(true), line)
        })
        .collect()
}

/// y = max(t1, t2) of two products equal in exact arithmetic and apart by the kernel's roundings:
/// the maximum flips on most inputs, and the linearized tiers are refused for it.
#[test]
fn a_flipping_maximum_is_refused() {
    let j = run(
        "max",
        "fixtures/errdist/flip/max",
        "fixtures/errdist/normal.json",
        None,
    );
    // One maximum per element.
    let b = &j["branch_flips"];
    assert_eq!(
        (b["selections"].as_u64(), b["maxima"].as_u64()),
        (Some(0), Some(64)),
        "{b}"
    );
    assert!(
        b["flipped"].as_u64().unwrap() * 2 > b["inputs"].as_u64().unwrap(),
        "{b}"
    );
    let g = gates(&j);
    assert!(!g.is_empty());
    for (ok, line) in g {
        assert!(!ok && line.ends_with("FAILS"), "{line}");
    }
}

/// y = t1 > t2 ? x : w, the same comparison as a selection: refused on Tier 3's count. Each
/// element's two guarded stores are a selection each.
#[test]
fn a_flipping_selection_is_refused() {
    let j = run(
        "sel",
        "fixtures/errdist/flip/sel",
        "fixtures/errdist/normal.json",
        None,
    );
    let b = &j["branch_flips"];
    assert_eq!(
        (b["selections"].as_u64(), b["maxima"].as_u64()),
        (Some(128), Some(0)),
        "{b}"
    );
    assert!(
        b["flipped"].as_u64().unwrap() * 2 > b["inputs"].as_u64().unwrap(),
        "{b}"
    );
    for (ok, line) in gates(&j) {
        assert!(!ok && line.ends_with("FAILS"), "{line}");
    }
}

/// y = t1 > 0.5 ? t1 : t2 flips only where t1 and its exact value lie on either side of 0.5,
/// which no sample hits: with the default input count the bound (7e-3 at 525 inputs) is within
/// 10% of P (P about 0.26), the estimate stands with its P(differ) interval widened
/// by the bound, and it agrees with Tier 3 (the output's grid position follows the selected
/// child). At a fixed 100 inputs the bound (3.6e-2) is too wide and it is refused.
#[test]
fn a_rare_flip_is_bounded() {
    let j = run(
        "rare",
        "fixtures/errdist/flip/rare",
        "fixtures/errdist/normal.json",
        None,
    );
    let b = &j["branch_flips"];
    assert_eq!(
        (b["selections"].as_u64(), b["flipped"].as_u64()),
        (Some(64), Some(0)),
        "{b}"
    );
    let ub = b["input"][2].as_f64().unwrap();
    let est = j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["method"].as_str().unwrap().starts_with("linearized"))
        .collect::<Vec<_>>();
    assert!(!est.is_empty());
    for e in est {
        let line = e["applicability"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_str().unwrap())
            .find(|a| a.starts_with("branch flips"))
            .unwrap();
        assert!(line.ends_with(" ok"), "{line}");
        assert_eq!(e["applicable"], true, "{}", e["applicability"]);
        let p = &e["p_differ"];
        let (pp, lo, hi) = (
            p[0].as_f64().unwrap(),
            p[1].as_f64().unwrap(),
            p[2].as_f64().unwrap(),
        );
        assert!(pp - lo >= ub && hi - pp >= ub, "{p} vs bound {ub}");
    }
    let j = run(
        "rare100",
        "fixtures/errdist/flip/rare",
        "fixtures/errdist/normal.json",
        Some("100"),
    );
    assert_eq!(j["branch_flips"]["inputs"], 100);
    let g = gates(&j);
    assert!(!g.is_empty());
    for (ok, line) in g {
        assert!(!ok && line.ends_with("FAILS"), "{line}");
    }
}

/// With 4096 outputs per input Tier 3's interval is fine at 300 inputs, but the flip bound
/// counts inputs (1.2e-2 at 300 against a margin of about 4.6e-3 at P 0.057): with the default
/// input count Tier 3 grows until B is within FLIP_GROW (0.8) of the margin at its own P, the
/// same inputs on every run. A work budget too small for that leaves 300 inputs, a note, and
/// the gate refusing. (The linearized estimate itself is 21% low on this shape, a rounded
/// product added to a value of similar size, and its cross-check refuses it either way.)
#[test]
#[ignore = "heavy: 36 s"]
fn tier3_grows_for_the_flip_bound() {
    let case = "fixtures/errdist/flip/wide";
    let spec = "fixtures/errdist/normal.json";
    let notes = |j: &serde_json::Value| {
        j["notes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_str().unwrap().to_string())
            .filter(|n| n.starts_with("branch flips:"))
            .collect::<Vec<_>>()
    };
    let p3 = |j: &serde_json::Value| {
        j["estimates"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["tier"] == 3)
            .unwrap()["p_differ"][0]
            .as_f64()
            .unwrap()
    };
    let j = run("wide", case, spec, None);
    let b = &j["branch_flips"];
    let (n, ub) = (
        b["inputs"].as_u64().unwrap(),
        b["input"][2].as_f64().unwrap(),
    );
    assert_eq!(b["flipped"], 0, "{b}");
    assert!(n > 600, "{b}");
    assert!(ub <= 0.8 * 0.1 * p3(&j), "{b} at P {}", p3(&j));
    let nt = notes(&j);
    assert!(
        nt.iter()
            .any(|t| t.contains(&format!("grew to {n} inputs"))),
        "{nt:?}"
    );
    // The same inputs and estimates on every run.
    let again = run("wide2", case, spec, None);
    assert_eq!(again["branch_flips"], j["branch_flips"]);
    for (e, f) in j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .zip(again["estimates"].as_array().unwrap())
    {
        assert_eq!(e["p_differ"], f["p_differ"], "{}", e["method"]);
    }
    // A budget of 0.3 s of work units holds Tier 3 at 300 inputs.
    let j = run_env("wide3", case, spec, None, &[("PTXV_XCHECK_SECONDS", "0.3")]);
    assert_eq!(j["branch_flips"]["inputs"], 300, "{}", j["branch_flips"]);
    let nt = notes(&j);
    assert!(
        nt.iter()
            .any(|t| t.contains("more than the work budget allows")),
        "{nt:?}"
    );
    let g = gates(&j);
    assert!(!g.is_empty());
    for (ok, line) in g {
        assert!(!ok && line.ends_with("FAILS"), "{line}");
    }
}

/// The simulator's validation runs contradict flip_sel's linearized estimate (0 against about
/// 0.17 over 20 inputs; the default 4 leave a 99.9% interval of [0, 0.43]): a contradicted
/// estimate is refused whatever its other checks say ("sim validation ... DISAGREES FAILS").
#[test]
fn a_contradicted_estimate_is_refused() {
    let js = std::env::temp_dir().join(format!("ptxv-test-flip-val-{}.json", std::process::id()));
    let scratch = js.with_extension("d");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["errdist", "fixtures/errdist/flip/sel", "--sim-trials", "20"])
        .args(["--inputs", "fixtures/errdist/normal.json"])
        .args(["--json", js.to_str().unwrap()])
        .args(["--out", scratch.to_str().unwrap()])
        .current_dir(root())
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&js).unwrap()).unwrap();
    let _ = std::fs::remove_file(&js);
    let _ = std::fs::remove_dir_all(&scratch);
    let lin: Vec<_> = j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["method"].as_str().unwrap().starts_with("linearized"))
        .collect();
    assert!(!lin.is_empty());
    for e in lin {
        assert_eq!(e["applicable"], false);
        assert!(
            e["applicability"].as_array().unwrap().iter().any(|a| a
                .as_str()
                .unwrap()
                .starts_with("sim validation")
                && a.as_str().unwrap().ends_with("DISAGREES FAILS")),
            "{}",
            e["applicability"]
        );
    }
}

/// A pair counts the flips of both programs: R's selection flips nowhere, C's maximum on most
/// inputs.
#[test]
fn a_pair_counts_both_programs() {
    let j = run(
        "pair",
        "fixtures/errdist/flip/pair.toml",
        "fixtures/errdist/normal.json",
        None,
    );
    let b = &j["branch_flips"];
    assert_eq!(
        (b["selections"].as_u64(), b["maxima"].as_u64()),
        (Some(64), Some(64)),
        "{b}"
    );
    assert!(
        b["flipped"].as_u64().unwrap() * 2 > b["inputs"].as_u64().unwrap(),
        "{b}"
    );
    for (ok, line) in gates(&j) {
        assert!(!ok && line.ends_with("FAILS"), "{line}");
    }
}

/// A maximum of input elements (softmax's row max) decides alike in the kernel and in exact
/// arithmetic: no bound, no gate.
#[test]
fn exact_maxima_need_no_bound() {
    let j = run(
        "softmax",
        "fixtures/c_softmax_stable",
        "fixtures/errdist/normal.json",
        Some("300"),
    );
    assert!(j.get("branch_flips").is_none(), "{}", j["branch_flips"]);
    for (_, line) in gates(&j) {
        assert!(line.is_empty(), "{line}");
    }
}

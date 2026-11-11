/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! errdist's guards against reporting from bad data (docs/errdist.md §1): NaN / Inf outputs
//! are out-of-domain mass, never dropped silently; a numerics DAG that disagrees with the
//! simulator is re-extracted in segment mode, and refused if that disagrees too; a run whose
//! control, addresses or guards depend on input data where its DAG does not model them is
//! refused.

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

/// Run errdist where it must refuse: its output.
fn run_refused(path: &str, spec: &str) -> String {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["errdist", path, "--inputs", spec, "--mc", "200"])
        .current_dir(root())
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.status.success(), "{text}");
    text
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

/// MISS 53: an EXIT under a data-dependent guard whose predicate has no slice
/// (fixtures/errdist/miss53/exit, `bfind(bits(x)) == 30` on half of the threads). On the
/// captured input those threads exit, so their outputs had neither a node nor taint: errdist
/// skipped them, reported over the other half, and its validation compared only those. Every
/// input that stores there was evaluated as if it exited. The run is refused.
#[test]
fn data_dependent_exit_is_refused() {
    let text = run_refused(
        "fixtures/errdist/miss53/exit",
        "fixtures/errdist/uniform01.json",
    );
    assert!(
        text.contains("guard of") && text.contains("EXIT") && text.contains("refusing to report"),
        "{text}"
    );
}

/// MISS 53: a store address chosen by a select on input data (fixtures/errdist/miss53/scatter):
/// the DAG holds the captured address's values only. Refused.
#[test]
fn data_dependent_address_is_refused() {
    let text = run_refused(
        "fixtures/errdist/miss53/scatter",
        "fixtures/errdist/uniform01.json",
    );
    assert!(
        text.contains("an address / control operand") && text.contains("refusing to report"),
        "{text}"
    );
}

/// A guarded store whose guard holds for every thread of the captured run
/// (fixtures/errdist/miss53/sel): the default extraction tracks every output and keeps the guard
/// as a data-dependent precondition; segment mode models the store as a selection
/// (N-ERRDIST-SEL), so errdist reports, and its DAG agrees with the simulator on sampled inputs
/// where the guard fails as well.
#[test]
fn guarded_store_is_a_selection_not_a_refusal() {
    let (text, j) = run(
        "m53sel",
        "fixtures/errdist/miss53/sel",
        "fixtures/errdist/normal.json",
    );
    assert!(text.contains("N-ERRDIST-SEL"), "{text}");
    assert!(!text.contains("precondition: data-dependent"), "{text}");
    assert!(j["best"].is_number(), "{}", j["best"]);
    let v = j["validation"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["source"] == "sim")
        .unwrap();
    assert_eq!(v["dag_mismatch"], 0, "{v}");
}

/// MISS 53 in a pair (fixtures/errdist/miss53/pair.toml): the reference (sel) is clean in
/// segment mode, the candidate (exit) is not; the pair is refused on the candidate.
#[test]
fn data_dependent_candidate_is_refused() {
    let text = run_refused(
        "fixtures/errdist/miss53/pair.toml",
        "fixtures/errdist/normal.json",
    );
    assert!(
        text.contains("m53_exit") && text.contains("EXIT") && text.contains("refusing to report"),
        "{text}"
    );
}

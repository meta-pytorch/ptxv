/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

fn compare(
    case: &str,
    front: &str,
    segments: bool,
    certify: bool,
    separate_cases: bool,
) -> serde_json::Value {
    static N: AtomicUsize = AtomicUsize::new(0);
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::temp_dir().join(format!(
        "ptxv-soundness-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let base = root.join("fixtures").join(case);
    let (reference, candidate, rv, cv) = if separate_cases {
        (
            base.join("reference"),
            base.join("candidate"),
            "good",
            "good",
        )
    } else {
        (base.clone(), base, "reference", "candidate")
    };
    let config = dir.join("equiv.toml");
    std::fs::write(
        &config,
        format!(
            r#"
max_states = 10000
trials = 0
segments = {segments}
elementwise = false
reference = {{case = {reference:?}, variant = "{rv}", front = "{front}", certify = {certify}}}
candidates = [{{case = {candidate:?}, variant = "{cv}", front = "{front}", certify = {certify}}}]
"#
        ),
    )
    .unwrap();
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args([
            "equiv",
            config.to_str().unwrap(),
            "--no-hw",
            "--out",
            dir.join("result").to_str().unwrap(),
        ])
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("PTXV_NO_VCACHE", "1")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{case}/{front}/{segments}/{certify}: {}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let rows: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("result/equiv.json")).unwrap()).unwrap();
    let report = rows[0].clone();
    assert!(
        report["proven"] == false || report["strongest"] == "none",
        "{case}/{front}/{segments}/{certify}: {report}"
    );
    std::fs::remove_dir_all(dir).unwrap();
    report
}

#[test]
fn split_shuffle_uses_the_remote_lanes_history() {
    for segments in [false, true] {
        let r = compare("e_split_shuffle", "ptx", segments, true, false);
        assert_eq!(r["report"]["levels"][0]["holds"], false, "{r}");
    }
}

#[test]
fn uniform_register_history_is_shared_across_lanes() {
    for segments in [false, true] {
        let r = compare("e_uniform_history", "sass", segments, true, true);
        assert_eq!(r["report"]["levels"][0]["holds"], false, "{r}");
    }
}

#[test]
fn failed_polls_cannot_hide_live_register_or_memory_effects() {
    for case in ["e_poll_conditional_register", "e_mbar_retry_counter"] {
        for front in ["ptx", "sass"] {
            for certify in [false, true] {
                let r = compare(case, front, true, certify, false);
                let class = if case == "e_mbar_retry_counter" {
                    "unmodeled"
                } else {
                    "nondeterministic"
                };
                assert!(
                    r["pre_reference"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|p| p.as_str().unwrap().contains(class)),
                    "{r}"
                );
            }
        }
    }
}

#[test]
fn ring_poll_failures_cannot_hide_memory_effects() {
    for front in ["ptx", "sass"] {
        for certify in [false, true] {
            let r = compare("e_poll_conditional_store", front, true, certify, false);
            assert!(
                r["pre_reference"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|p| p.as_str().unwrap().contains("nondeterministic")),
                "{r}"
            );
        }
    }
}

#[test]
fn warp_poll_stutters_compare_uniform_state_and_wide_stores() {
    for case in ["e_poll_uniform", "e_poll_wide"] {
        for certify in [false, true] {
            let r = compare(case, "sass", true, certify, true);
            let class = if case == "e_poll_uniform" {
                "inconclusive"
            } else {
                "nondeterministic"
            };
            assert!(
                r["pre_reference"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|p| p.as_str().unwrap().contains(class)),
                "{r}"
            );
        }
    }
}

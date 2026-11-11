/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Inputs that carry an error (docs/errdist.md §2.1) and composition through errstate files
//! (§10): the linearized tier carries J e_in + e_own and agrees with Tier 3 on small errors,
//! refuses where the kernel is not linear at the error's scale, and an errstate written by one
//! kernel feeds the next as joint (value, error) pairs.

use std::path::Path;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ptxv-test-inerr-{name}-{}", std::process::id()))
}

fn spec(name: &str, x: serde_json::Value) -> PathBuf {
    let p = tmp(name).with_extension("json");
    let s = serde_json::json!({
        "version": 1, "seed": 1,
        "tensors": {"x": x},
        "default": {"dist": {"kind": "normal"}},
    });
    std::fs::write(&p, s.to_string()).unwrap();
    p
}

fn run(name: &str, case: &str, spec: &Path, extra: &[&str]) -> serde_json::Value {
    let js = tmp(name).with_extension("out.json");
    let scratch = tmp(name).with_extension("d");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["errdist", case, "--sim-trials", "0"])
        .args(["--inputs", spec.to_str().unwrap()])
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

fn est(j: &serde_json::Value, tier: u64) -> &serde_json::Value {
    j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["tier"] == tier)
        .unwrap_or_else(|| panic!("no tier {tier}: {j}"))
}

fn spread_ratio(j: &serde_json::Value) -> f64 {
    est(j, 2)["abs"]["mean_abs"].as_f64().unwrap() / est(j, 3)["abs"]["mean_abs"].as_f64().unwrap()
}

fn best_tier(j: &serde_json::Value) -> u64 {
    j["estimates"][j["best"].as_u64().unwrap() as usize]["tier"]
        .as_u64()
        .unwrap()
}

#[test]
#[ignore = "heavy: 53 s"]
fn small_input_errors_carried_by_tier2() {
    for (name, err) in [
        ("rel", serde_json::json!({"kind": "relative", "std": 1e-6})),
        (
            "bf16",
            serde_json::json!({"kind": "round", "format": "bf16"}),
        ),
    ] {
        let s = spec(
            name,
            serde_json::json!({"dist": {"kind": "normal"}, "error": err}),
        );
        let j = run(name, "fixtures/c_rmsnorm_triton_f32", &s, &[]);
        let _ = std::fs::remove_file(&s);
        let r = spread_ratio(&j);
        assert!(
            (0.9..1.1).contains(&r),
            "{name}: tier 2 / tier 3 mean abs error {r}"
        );
        assert_eq!(best_tier(&j), 2, "{name}");
        assert!(
            j["assumptions"]
                .as_array()
                .unwrap()
                .iter()
                .any(|a| a == "N-ERRDIST-INERR")
        );
    }
}

#[test]
#[ignore = "heavy: 71 s"]
fn large_input_error_through_exp_refused() {
    // Softmax of N(0, 4^2) logits with an absolute error of 0.5: exp is far from linear at
    // that scale (curvature ratio ~0.25).
    let s = spec(
        "big",
        serde_json::json!({"dist": {"kind": "normal", "std": 4.0},
                           "error": {"kind": "absolute", "std": 0.5}}),
    );
    let j = run("big", "fixtures/r4_softmax_triton", &s, &[]);
    let _ = std::fs::remove_file(&s);
    let t2 = est(&j, 2);
    assert_eq!(t2["applicable"], false, "{t2}");
    assert!(
        t2["applicability"].to_string().contains("second-order"),
        "{}",
        t2["applicability"]
    );
    assert_eq!(best_tier(&j), 3);
}

#[test]
#[ignore = "heavy: 138 s"]
fn errstate_feeds_the_next_kernel() {
    let s = spec(
        "up",
        serde_json::json!({"dist": {"kind": "normal"},
                           "error": {"kind": "relative", "std": 1e-6}}),
    );
    let es = tmp("up").with_extension("es.json");
    let j = run(
        "up",
        "fixtures/c_rmsnorm_triton_f32",
        &s,
        &["--errstate", es.to_str().unwrap()],
    );
    let _ = std::fs::remove_file(&s);
    assert_eq!(j["errstate"], es.to_str().unwrap());
    let e: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&es).unwrap()).unwrap();
    assert_eq!(e["kind"], "ptxv-errstate");
    assert_eq!(e["model"], "iid");
    let t = &e["tensors"]["out"];
    assert!(t["pairs"].as_array().unwrap().len() > 1000);
    assert!(t["corr"]["error_same_row"].is_number(), "{}", t["corr"]);
    // The next kernel reads rmsnorm's output with its error.
    let s2 = spec(
        "down",
        serde_json::json!({"dist": {"kind": "normal"},
                           "error": {"kind": "pairs", "file": es.to_str().unwrap()}}),
    );
    let j2 = run("down", "fixtures/r4_softmax_triton", &s2, &[]);
    let _ = std::fs::remove_file(&s2);
    let _ = std::fs::remove_file(&es);
    let r = spread_ratio(&j2);
    assert!(
        (0.9..1.1).contains(&r),
        "tier 2 / tier 3 mean abs error {r}"
    );
    assert_eq!(best_tier(&j2), 2);
}

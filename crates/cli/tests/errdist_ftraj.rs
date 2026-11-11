/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Float-trajectory linearization (N-ERRDIST-NL-FTRAJ, `PTXV_ERRDIST_LIN_TRAJ`). Kahan's
//! compensation c = (t - s) - v is identically 0 in real arithmetic, so the real-trajectory
//! linearized Tier 2 never sees the roundings of x - c that dominate the error (P(differ)
//! about 2x low under N(0, 1), ~5000x low under U[0, 1) against GB300). Expanded around
//! the computed trajectory those roundings become sites. GB300 (10,000 / 100,000 inputs,
//! the errdist refresh): 0.403 under N(0, 1), 0.00863 under U[0, 1).

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run(name: &str, spec: &str, traj: Option<&str>) -> serde_json::Value {
    let js = std::env::temp_dir().join(format!(
        "ptxv-test-ftraj-{name}-{}.json",
        std::process::id()
    ));
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"));
    cmd.args([
        "errdist",
        "fixtures/c_sum_kahan",
        "--inputs",
        spec,
        "--mc",
        "4000",
    ])
    .args(["--json", js.to_str().unwrap()])
    .current_dir(root())
    .env("CUDA_VISIBLE_DEVICES", "")
    .env_remove("PTXV_ERRDIST_LIN_TRAJ");
    if let Some(t) = traj {
        cmd.env("PTXV_ERRDIST_LIN_TRAJ", t);
    }
    let out = cmd.output().unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "{text}");
    let j = serde_json::from_str(&std::fs::read_to_string(&js).unwrap()).unwrap();
    let _ = std::fs::remove_file(&js);
    j
}

/// Which linearization trajectory an estimate's method names.
fn traj(m: &str) -> &'static str {
    if m.contains("per-site trajectory") {
        "both"
    } else if m.contains("float trajectory") {
        "float"
    } else {
        "real"
    }
}

/// (P(differ) of Tier 3, of the real-trajectory linearized Tier 2, of the float one, of the
/// per-site one).
fn ps(j: &serde_json::Value) -> (f64, Option<f64>, Option<f64>, Option<f64>) {
    let est = j["estimates"].as_array().unwrap();
    let p = |e: &serde_json::Value| e["p_differ"][0].as_f64().unwrap();
    let t3 = est.iter().find(|e| e["tier"] == 3).map(p).unwrap();
    let lin = |t: &str| {
        est.iter()
            .find(|e| {
                let m = e["method"].as_str().unwrap();
                e["tier"] == 2 && m.starts_with("linearized") && traj(m) == t
            })
            .map(p)
    };
    (t3, lin("real"), lin("float"), lin("both"))
}

#[test]
fn kahan_normal_float_trajectory_matches_tier3() {
    let j = run("normal", "fixtures/errdist/normal.json", Some("all"));
    let (t3, real, float, both) = ps(&j);
    let (real, float, both) = (real.unwrap(), float.unwrap(), both.unwrap());
    // Tier 3 ~ 0.40; real trajectory ~ 0.21; float trajectory ~ 0.40.
    assert!(t3 > 0.35 && t3 < 0.45, "{t3}");
    assert!(real < 0.6 * t3, "real trajectory {real} vs tier 3 {t3}");
    assert!(
        (float - t3).abs() < 0.05 * t3,
        "float trajectory {float} vs tier 3 {t3}"
    );
    // Per-site selection keeps the float trajectory's gain (Kahan's sites agree, or their
    // own-rounding-free sensitivities do).
    assert!(
        (both - t3).abs() < 0.05 * t3,
        "per-site trajectory {both} vs tier 3 {t3}"
    );
    assert!(
        j["assumptions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a == "N-ERRDIST-NL-FTRAJ"),
        "{}",
        j["assumptions"]
    );
}

#[test]
fn kahan_uniform01_float_trajectory_sees_the_compensation() {
    let j = run("u01", "fixtures/errdist/uniform01.json", Some("all"));
    let (t3, real, float, both) = ps(&j);
    let (real, float, both) = (real.unwrap(), float.unwrap(), both.unwrap());
    // Tier 3 ~ 8.4e-3; real trajectory ~ 1.7e-3 (the compensation is 0 in real arithmetic, so
    // the input-correction roundings are no sites); float trajectory ~ 8.7e-3. (Until the
    // output grid followed the deviation's sd, the two came out at 1.5e-6 and 6.5e-3.)
    assert!(t3 > 4e-3 && t3 < 1.5e-2, "{t3}");
    assert!(real < 0.5 * t3, "real trajectory {real} vs tier 3 {t3}");
    assert!(
        (float - t3).abs() < 0.1 * t3,
        "float trajectory {float} vs tier 3 {t3}"
    );
    assert!(
        both > 0.5 * t3 && both < 1.5 * t3,
        "per-site trajectory {both} vs tier 3 {t3}"
    );
}

#[test]
fn default_is_the_real_trajectory_only() {
    let j = run("default", "fixtures/errdist/normal.json", None);
    let (_, real, float, both) = ps(&j);
    assert!(real.is_some() && float.is_none() && both.is_none());
    let j = run("float", "fixtures/errdist/normal.json", Some("float"));
    let (_, real, float, both) = ps(&j);
    assert!(real.is_none() && float.is_some() && both.is_none());
    let j = run("both", "fixtures/errdist/normal.json", Some("both"));
    let (_, real, float, both) = ps(&j);
    assert!(real.is_none() && float.is_none() && both.is_some());
}

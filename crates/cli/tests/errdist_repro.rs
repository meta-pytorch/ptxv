/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Reproducible estimates: the same binary, case, inputs and options give the same estimates,
//! bit for bit (timings aside). Slice tables draw each binade from its own stream, maps
//! iterate in a fixed order, the input budgets of the linearized Tier 2 and of Tier 3's
//! cross-check growth are counted in work units, and Tier 3 merges its inputs in order.

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The estimates of one run, without their timings.
fn estimates(name: &str, case: &str, spec: &str, env: &[(&str, &str)]) -> serde_json::Value {
    let js = std::env::temp_dir().join(format!(
        "ptxv-test-repro-{name}-{}.json",
        std::process::id()
    ));
    let scratch = js.with_extension("d");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args([
            "errdist",
            case,
            "--inputs",
            spec,
            "--sim-trials",
            "0",
            "--tier1",
        ])
        .args(["--json", js.to_str().unwrap()])
        .args(["--out", scratch.to_str().unwrap()])
        .current_dir(root())
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("PTXV_QUIET", "1")
        .envs(env.iter().copied())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let mut j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&js).unwrap()).unwrap();
    let _ = std::fs::remove_file(&js);
    let _ = std::fs::remove_dir_all(&scratch);
    let mut e = j["estimates"].take();
    for x in e.as_array_mut().unwrap() {
        x.as_object_mut().unwrap().remove("seconds");
    }
    e
}

fn same_twice(name: &str, case: &str, spec: &str, env: &[(&str, &str)]) {
    let a = estimates(&format!("{name}-a"), case, spec, env);
    let b = estimates(&format!("{name}-b"), case, spec, env);
    assert!(a.as_array().unwrap().len() >= 2, "{a}");
    assert_eq!(a, b, "two runs of {case} differ");
}

/// Softmax at N(50, 1): slice tables (the division) and an opened exp2 slice; the linearized
/// Tier 2 stops on its work budget (about 80 inputs), not at a fixed count.
#[test]
fn single_kernel_estimates_are_reproducible() {
    same_twice(
        "softmax",
        "fixtures/c_triton_softmax_naive",
        "fixtures/errdist/shift50.json",
        &[],
    );
}

/// A pair of softmax kernels: matched roundings enter as joint sites, whose order came from a
/// map's iteration.
#[test]
#[ignore = "heavy: 2 errdist --tier1 subprocesses; 138 s"]
fn pair_estimates_are_reproducible() {
    same_twice(
        "pair",
        "fixtures/c_equiv_softmax/equiv.toml",
        "fixtures/errdist/normal.json",
        &[],
    );
}

/// Kahan summation at N(0, 1): Tier 3 grows from 300 inputs toward its cross-check
/// resolution until its budget (0.05 s of work: 873 inputs) runs out.
#[test]
fn tier3_growth_is_reproducible() {
    same_twice(
        "kahan",
        "fixtures/c_sum_kahan",
        "fixtures/errdist/normal.json",
        &[("PTXV_XCHECK_SECONDS", "0.05")],
    );
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv errdist` on non-chain programs (agent AG; docs/errdist.md §9): the analytic tiers
//! against Tier 3 on a fast subset of the validation matrix (no GPU).

fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn errdist(path: &str, spec: &str, extra: &[&str]) -> Result<serde_json::Value, String> {
    // Tests run in parallel and several share (path, spec): a counter keeps every call's
    // JSON and scratch directory distinct.
    static CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let js = std::env::temp_dir().join(format!(
        "ptxv-test-errdist-nl-{}-{}-{}-{}.json",
        path.replace('/', "_"),
        spec.replace('/', "_"),
        std::process::id(),
        CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let scratch = js.with_extension("d");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args([
            "errdist",
            path,
            "--inputs",
            spec,
            "--mc",
            "2000",
            "--sim-trials",
            "0",
            // These tests check the non-chain tier 1 (opt-in) beside tiers 0 and 2.
            "--tier1",
        ])
        .args(["--json", js.to_str().unwrap()])
        .args(["--out", scratch.to_str().unwrap()])
        .args(extra)
        .current_dir(root())
        .env("PTXV_QUIET", "1")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    if !out.status.success() {
        let _ = std::fs::remove_dir_all(&scratch);
        return Err(format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let j = serde_json::from_str(&std::fs::read_to_string(&js).unwrap()).unwrap();
    let _ = std::fs::remove_file(&js);
    let _ = std::fs::remove_dir_all(&scratch);
    Ok(j)
}

fn p_differ(e: &serde_json::Value) -> (f64, f64, f64) {
    let p = &e["p_differ"];
    (
        p[0].as_f64().unwrap(),
        p[1].as_f64().unwrap(),
        p[2].as_f64().unwrap(),
    )
}

/// The first estimate of `tier` whose method starts with `kind`.
fn find<'a>(j: &'a serde_json::Value, tier: u64, kind: &str) -> &'a serde_json::Value {
    j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["tier"] == tier && e["method"].as_str().unwrap().starts_with(kind))
        .unwrap_or_else(|| panic!("no tier {tier} {kind} estimate: {j}"))
}

/// P(differ) of `e` within max(5% relative, the CIs) of Tier 3's.
fn agrees(j: &serde_json::Value, e: &serde_json::Value) {
    let t3 = find(j, 3, "Monte Carlo");
    let (p3, lo3, hi3) = p_differ(t3);
    let (p, lo, hi) = p_differ(e);
    let tol = (0.05 * p3).max(hi3 - lo3).max(hi - lo);
    assert!(
        (p - p3).abs() <= tol,
        "{}: P(differ) {p} vs tier 3 {p3} [{lo3}, {hi3}]",
        e["method"]
    );
}

/// Softmax (torch): the linearized Tier 1 agrees with Tier 3 and is the headline.
#[test]
fn softmax_single_linearized() {
    let j = errdist(
        "fixtures/r4_softmax_torch",
        "fixtures/errdist/normal.json",
        &[],
    )
    .unwrap();
    let e = find(&j, 1, "linearized");
    agrees(&j, e);
    let best = &j["estimates"][j["best"].as_u64().unwrap() as usize];
    assert_ne!(best["tier"], 3, "{j}");
}

/// Elementwise pairs: exact enumeration on bf16 codes (Tier 0), stratified quadrature on f32
/// (Tier 1), and a Latin hypercube for silu(a) * b in bf16, each agreeing with Tier 3.
#[test]
fn elementwise_pairs() {
    let j = errdist(
        "fixtures/c_equiv_elementwise/gelu_bf16.toml",
        "fixtures/errdist/normal.json",
        &[],
    )
    .unwrap();
    let e = find(&j, 0, "elementwise");
    let (p, _, _) = p_differ(e);
    let (_, lo3, hi3) = p_differ(find(&j, 3, "Monte Carlo"));
    // Tier 3's interval at n = 2000 inputs (~1e5 outputs) is wide at p ~ 6e-5.
    assert!(p > 0.5 * lo3 && p < 1.5 * hi3, "{p} vs [{lo3}, {hi3}]");
    for t in [
        "fixtures/c_equiv_elementwise/silu.toml",
        "fixtures/errdist/cve/silumul_bf16.toml",
    ] {
        let j = errdist(t, "fixtures/errdist/normal.json", &[]).unwrap();
        let e = find(&j, 1, "elementwise");
        agrees(&j, e);
        let best = &j["estimates"][j["best"].as_u64().unwrap() as usize];
        assert!(
            best["method"].as_str().unwrap().starts_with("elementwise"),
            "{t}: {j}"
        );
    }
}

/// A pair of multi-launch regions (x @ (w1 + w2) against x @ w1 + x @ w2): extracted across
/// launches; Tier 1 agrees with Tier 3.
#[test]
fn multi_launch_pair() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    if ptxv_case::skip_missing_vendor_files(&root.join("fixtures"), &["fixtures/errdist/mm2.toml"]) {
        return;
    }
    let j = errdist(
        "fixtures/errdist/mm2.toml",
        "fixtures/errdist/normal.json",
        &[],
    )
    .unwrap();
    agrees(&j, find(&j, 1, "linearized"));
}

/// A region whose capture misses a device-to-device memcpy (inductor's addmm input copy) is
/// refused: its DAG disagrees with the captured GPU output.
#[test]
fn missing_memcpy_is_refused() {
    let e = errdist(
        "fixtures/errdist/c_mm2_addmm",
        "fixtures/errdist/normal.json",
        &[],
    )
    .unwrap_err();
    assert!(e.contains("differ from the captured GPU outputs"), "{e}");
}

/// Compile once, query many: a model written with --compile is loaded by --model for another
/// input distribution (and extended to its binades).
#[test]
fn model_compile_and_query() {
    let m = std::env::temp_dir().join(format!("ptxv-test-nl-model-{}.json", std::process::id()));
    let _ = std::fs::remove_file(&m);
    let ms = m.to_str().unwrap();
    let j = errdist(
        "fixtures/r4_softmax_torch",
        "fixtures/errdist/normal.json",
        &["--compile", ms],
    )
    .unwrap();
    assert!(j.to_string().contains("non-chain model written"), "{j}");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&m).unwrap()).unwrap();
    assert!(v["nonlin"]["tables"]["tables"].as_array().unwrap().len() > 1);
    let j = errdist(
        "fixtures/r4_softmax_torch",
        "fixtures/errdist/uniform12.json",
        &["--model", ms],
    )
    .unwrap();
    assert!(j.to_string().contains("loaded"), "{j}");
    agrees(&j, find(&j, 1, "linearized"));
    let _ = std::fs::remove_file(&m);
}

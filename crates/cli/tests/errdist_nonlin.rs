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

/// Softmax (torch): the linearized Tier 2 agrees with Tier 3 and is the headline. There is no
/// linearized Tier 1, with --tier1 too (retired 2026-11-10: within 0.5% of Tier 2 on every
/// scorecard row, never cheaper), and the site notes come from Tier 2.
#[test]
fn softmax_single_linearized() {
    let j = errdist(
        "fixtures/r4_softmax_torch",
        "fixtures/errdist/normal.json",
        &["--tier1"],
    )
    .unwrap();
    let e = find(&j, 2, "linearized");
    agrees(&j, e);
    let best = &j["estimates"][j["best"].as_u64().unwrap() as usize];
    assert_ne!(best["tier"], 3, "{j}");
    let lin: Vec<u64> = j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["method"].as_str().unwrap().starts_with("linearized"))
        .map(|e| e["tier"].as_u64().unwrap())
        .collect();
    assert_eq!(lin, [2], "linearized estimates by tier");
    let notes = j["notes"].to_string();
    assert!(
        notes.contains("variance budget of the pre-rounding deviation"),
        "{notes}"
    );
}

/// Elementwise pairs: exact enumeration on bf16 codes (Tier 0), stratified quadrature on f32
/// (Tier 1), and a Latin hypercube for silu(a) * b in bf16, each agreeing with Tier 3.
#[test]
#[ignore = "heavy: 3 errdist subprocesses at --mc 2000; 213 s"]
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

/// The elementwise law runs by default (its f32 quadrature is labelled Tier 1; --tier1 now
/// gates the chains' Tier 1 only): on single f32 silu it is the headline. Before 2026-11-10
/// --tier1 gated it, and the default headline was the linearized Tier 2 (0.317 against Tier
/// 3's 0.328).
#[test]
fn elementwise_law_runs_by_default() {
    let j = errdist("fixtures/c_silu", "fixtures/errdist/normal.json", &[]).unwrap();
    let e = find(&j, 1, "elementwise");
    agrees(&j, e);
    let best = &j["estimates"][j["best"].as_u64().unwrap() as usize];
    assert!(
        best["method"].as_str().unwrap().starts_with("elementwise"),
        "{j}"
    );
    let notes = j["notes"].to_string();
    assert!(!notes.contains("opt-in: --tier1"), "{notes}");
}

/// Single elementwise kernels (bf16 enumeration, Tier 0; f32 quadrature, Tier 1): the PMF is
/// on Tier 3's axis, output - RN(y) in output ulps, with all of the law's mass, and its mass
/// off 0 is P(differ). Until 2026-11-10 it was keyed by the error against y, and only tuples
/// whose error was a whole number of ulps entered it (c_gelu_bf16: total mass 1e-16).
fn single_pmf_has_all_the_mass(t: &str, tier: u64) {
    let j = errdist(t, "fixtures/errdist/normal.json", &[]).unwrap();
    let e = find(&j, tier, "elementwise");
    let pmf: Vec<(i64, f64)> = e["pmf_ulp"]
        .as_array()
        .unwrap_or_else(|| panic!("{t}: no pmf_ulp: {e}"))
        .iter()
        .map(|x| (x[0].as_i64().unwrap(), x[1].as_f64().unwrap()))
        .collect();
    let total: f64 = pmf.iter().map(|x| x.1).sum();
    let off: f64 = pmf.iter().filter(|x| x.0 != 0).map(|x| x.1).sum();
    let (p, _, _) = p_differ(e);
    assert!((total - 1.0).abs() < 1e-9, "{t}: PMF mass {total}: {pmf:?}");
    assert!(
        (off - p).abs() < 1e-9,
        "{t}: mass off 0 {off} vs P(differ) {p}"
    );
    // The PMF agrees with Tier 3's at +-1 ulp (2000 inputs).
    let t3 = find(&j, 3, "Monte Carlo");
    let k1 = |x: &serde_json::Value| -> f64 {
        x["pmf_ulp"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|y| y[0].as_i64().unwrap().abs() == 1)
            .map(|y| y[1].as_f64().unwrap())
            .sum()
    };
    let (a, b) = (k1(e), k1(t3));
    let (_, lo3, hi3) = p_differ(t3);
    assert!(
        (a - b).abs() <= (0.05 * b).max(hi3 - lo3),
        "{t}: P(|k| = 1) {a} vs Tier 3 {b}"
    );
}

#[test]
fn elementwise_single_pmf_has_all_the_mass_bf16() {
    single_pmf_has_all_the_mass("fixtures/c_gelu_bf16", 0);
}

#[test]
fn elementwise_single_pmf_has_all_the_mass_f32() {
    single_pmf_has_all_the_mass("fixtures/c_silu", 1);
}

/// An f32 pair whose C - R leaves the lattice across binades: no PMF, as Tier 3 (a partial
/// one would understate every level).
#[test]
fn elementwise_pair_off_the_lattice_has_no_partial_pmf() {
    let j = errdist(
        "fixtures/c_equiv_elementwise/gelu_tanh.toml",
        "fixtures/errdist/normal.json",
        &[],
    )
    .unwrap();
    let e = find(&j, 1, "elementwise");
    let total: f64 = e["pmf_ulp"]
        .as_array()
        .map_or(0.0, |v| v.iter().map(|x| x[1].as_f64().unwrap()).sum());
    assert!(
        total == 0.0 || (total - 1.0).abs() < 1e-9,
        "gelu_tanh pair: partial PMF of mass {total}"
    );
}

/// The f32 elementwise law on an input with a point mass (lin_gap/mix under vmix: v = 1 on 95%
/// of the elements, else N(0, 1); and the same law as 2 X of a mixture): the atom stays one
/// code, and the law agrees with Tier 3 and is the headline. Until 2026-11-10 the atom was
/// spread over its 1/256-binade cell like a density, values near 1 that round like N(0, 1)
/// draws: 0.26 against Tier 3's 0.013 (refused by the cross-check).
#[test]
fn elementwise_mixture_keeps_its_point_mass() {
    for spec in ["vmix", "vmix_affine"] {
        let j = errdist(
            "fixtures/errdist/lin_gap/mix",
            &format!("fixtures/errdist/lin_gap/{spec}.json"),
            &[],
        )
        .unwrap();
        let e = find(&j, 1, "elementwise");
        agrees(&j, e);
        let best = &j["estimates"][j["best"].as_u64().unwrap() as usize];
        assert!(
            best["method"].as_str().unwrap().starts_with("elementwise"),
            "{spec}: {j}"
        );
    }
}

/// A pair of multi-launch regions (x @ (w1 + w2) against x @ w1 + x @ w2): extracted across
/// launches; the linearized Tier 2 agrees with Tier 3.
#[test]
fn multi_launch_pair() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    if ptxv_case::skip_missing_vendor_files(&root.join("fixtures"), &["fixtures/errdist/mm2.toml"])
    {
        return;
    }
    let j = errdist(
        "fixtures/errdist/mm2.toml",
        "fixtures/errdist/normal.json",
        &[],
    )
    .unwrap();
    agrees(&j, find(&j, 2, "linearized"));
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
#[ignore = "heavy: 33 s"]
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
    agrees(&j, find(&j, 2, "linearized"));
    let _ = std::fs::remove_file(&m);
}

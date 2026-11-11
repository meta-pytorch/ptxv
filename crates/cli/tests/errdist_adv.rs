/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Adversarial rows for the errdist rules (ASSUMPTIONS.md N-ERRDIST-*): for each claimed
//! property, a case where breaking it changes the reported result. Each test is the target of a
//! source mutant in tools/mutate/srcmut.py (docs/soundness-harness.md, "Adversarial tests of
//! proof rules"); Tier 3 (bit-exact per sample, N-ERRDIST-MC) is the reference throughout.

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run(
    tag: &str,
    case: &str,
    spec: &str,
    extra: &[&str],
    env: &[(&str, &str)],
) -> serde_json::Value {
    let js = std::env::temp_dir().join(format!("ptxv-test-adv-{tag}-{}.json", std::process::id()));
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"));
    cmd.args(["errdist", case, "--inputs", spec])
        .args(extra)
        .args(["--json", js.to_str().unwrap()])
        .current_dir(root())
        .env("CUDA_VISIBLE_DEVICES", "")
        .env_remove("PTXV_ERRDIST_LIN_TRAJ")
        .env_remove("PTXV_XCHECK_SECONDS");
    for (k, v) in env {
        cmd.env(k, v);
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

fn ests(j: &serde_json::Value) -> &Vec<serde_json::Value> {
    j["estimates"].as_array().unwrap()
}

fn tier(j: &serde_json::Value, t: u64) -> Vec<&serde_json::Value> {
    ests(j).iter().filter(|e| e["tier"] == t).collect()
}

fn best(j: &serde_json::Value) -> &serde_json::Value {
    &ests(j)[j["best"].as_u64().unwrap() as usize]
}

fn p(e: &serde_json::Value) -> (f64, f64, f64) {
    let v = &e["p_differ"];
    (
        v[0].as_f64().unwrap(),
        v[1].as_f64().unwrap(),
        v[2].as_f64().unwrap(),
    )
}

fn method(e: &serde_json::Value) -> &str {
    e["method"].as_str().unwrap()
}

fn applicability(e: &serde_json::Value) -> String {
    e["applicability"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|x| x.as_str().unwrap().to_string())
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

/// The DAG CLT estimate (tier 2, not the linearized one).
fn dag_clt(j: &serde_json::Value) -> Option<&serde_json::Value> {
    tier(j, 2)
        .into_iter()
        .find(|e| method(e).starts_with("CLT on"))
}

/// Tier 0 certificate (tier0::certify): every truncation and rounding of the chain is exact
/// when the terms sit on a lattice 2^L and every partial sum is below 2^(L + 24). A bf16 GEMM
/// (K = 64) on integers -8..8 certifies, and Tier 3 sees no difference. With integers scaled by
/// 2^-6..2^6 the lattice is 2^-12 and the partial sums reach 1.7e7, so the certificate must
/// fail (on the lattice condition): Tier 3 sees P(differ) about 4e-5. The bound itself decides
/// only partial sums in [2^(L+24), 2^(L+26)): fixtures/errdist/adv_t0_window.json (768 or 1 per
/// element, L = 0, sums to 2^25.2) on the bf16 tcgen05 tile with f32 output, where Tier 3 sees
/// P(differ) 0.066; no certificate may claim it. Mutant t0-bound (the bound loosened) does.
#[test]
fn tier0_certificate_needs_the_24_bit_bound() {
    let j = run(
        "t0-int",
        "fixtures/c_mm_bf16_nnt",
        "fixtures/errdist/integers.json",
        &["--mc", "1000"],
        &[],
    );
    let t0 = tier(&j, 0);
    assert_eq!(t0.len(), 1, "{j}");
    assert!(
        method(t0[0]).starts_with("exactness certificate"),
        "{}",
        method(t0[0])
    );
    assert_eq!(best(&j)["tier"], 0);
    assert_eq!(p(tier(&j, 3)[0]).0, 0.0);
    let j = run(
        "t0-scaled",
        "fixtures/c_mm_bf16_nnt",
        "fixtures/errdist/ints_scaled.json",
        &["--mc", "1000"],
        &[],
    );
    assert!(tier(&j, 0).is_empty(), "no tier 0 may claim this: {j}");
    assert!(p(tier(&j, 3)[0]).1 > 0.0, "tier 3 sees differences: {j}");
    assert_ne!(best(&j)["tier"], 0);
    let j = run(
        "t0-window",
        "fixtures/r1a_tcgen05_tile",
        "fixtures/errdist/adv_t0_window.json",
        &["--mc", "1000"],
        &[],
    );
    assert!(p(tier(&j, 3)[0]).1 > 0.05, "tier 3 sees differences: {j}");
    for e in tier(&j, 0) {
        assert!(
            !method(e).starts_with("exactness certificate"),
            "certified exact: {}",
            method(e)
        );
    }
}

/// Tier 0 identical programs (tier0::identical_outputs): equal canonical DAGs give C = R. The
/// bf16 GEMM against Triton in the same K order is identical (Tier 3: no difference); against
/// Triton with K reversed it is not (Tier 3 about 4e-5), and no tier 0 may say D = 0. Mutant
/// t0-ident-noidx (leaves hashed without their element index) makes the reversed pair
/// "identical".
#[test]
fn tier0_identical_programs_needs_equal_dags() {
    let j = run(
        "t0-same",
        "fixtures/c_equiv_mm_bf16/equiv.toml",
        "fixtures/errdist/normal.json",
        &["--candidate", "0", "--mc", "1000"],
        &[],
    );
    let t0 = tier(&j, 0);
    assert_eq!(t0.len(), 1, "{j}");
    assert!(method(t0[0]).starts_with("identical programs"));
    assert_eq!(p(tier(&j, 3)[0]).0, 0.0);
    let j = run(
        "t0-rev",
        "fixtures/c_equiv_mm_bf16/equiv.toml",
        "fixtures/errdist/normal.json",
        &["--candidate", "1", "--mc", "1000"],
        &[],
    );
    assert!(tier(&j, 0).is_empty(), "no tier 0 may claim this: {j}");
    assert!(p(tier(&j, 3)[0]).1 > 0.0, "{j}");
}

/// Tier 2 structural gate (tier2::KURT_MAX): without Tier 3 (`--mc 0`) the DAG CLT's
/// applicability rests on its structural checks alone. A bf16 GEMM with 2% N(0, 64^2)
/// outliers has y's excess kurtosis ~300; the CLT would say 8.7e-5 against Tier 3's 3.2e-5
/// (docs/errdist.md §3.2), so it must be refused. Mutant t2-kurt (the bound raised) makes it
/// applicable.
#[test]
fn tier2_kurtosis_gate_holds_without_tier3() {
    let j = run(
        "t2-kurt",
        "fixtures/c_mm_bf16_nnt",
        "fixtures/errdist/outliers.json",
        &["--mc", "0"],
        &[],
    );
    let e = dag_clt(&j).expect("the DAG CLT runs");
    assert_eq!(e["applicable"], false, "{}", applicability(e));
    assert!(
        applicability(e).contains("excess kurtosis"),
        "{}",
        applicability(e)
    );
}

/// MISS 42 (docs/errdist.md §3.2): with `--mc 0` and every analytic tier refused, `best` falls back
/// to estimates[0] (`unwrap_or(0)`), so the refused CLT above (8.7e-5, 2.7x Tier 3) is the
/// summary's headline. Un-ignore when a refused estimate can no longer be the headline.
#[test]
fn refused_estimates_are_never_the_headline() {
    let j = run(
        "t2-kurt-best",
        "fixtures/c_mm_bf16_nnt",
        "fixtures/errdist/outliers.json",
        &["--mc", "0"],
        &[],
    );
    if !j["best"].is_null() {
        let b = best(&j);
        assert_ne!(b["applicable"], false, "refused headline: {}", method(b));
    }
}

/// The Tier 3 cross-check's tolerance (|p - p3| <= max(5% of p3 - h, h)): on the torch sum vs
/// Triton row-sum pair the DAG CLT says 0.744 against Tier 3's 0.701 (6% high) and must be
/// refused; the headline is Tier 3. Mutant xcheck-tol (50%) accepts it.
#[test]
fn crosscheck_refuses_a_six_percent_tier2() {
    let j = run(
        "xtol",
        "fixtures/c_equiv_sum/equiv.toml",
        "fixtures/errdist/normal.json",
        &["--candidate", "0"],
        &[],
    );
    let e = dag_clt(&j).expect("the DAG CLT runs");
    let (p2, p3) = (p(e).0, p(tier(&j, 3)[0]).0);
    assert!((p2 / p3 - 1.0).abs() > 0.05, "the row moved: {p2} vs {p3}");
    assert_eq!(e["applicable"], false, "{}", applicability(e));
    assert!(
        applicability(e).contains("Tier 3 cross-check"),
        "{}",
        applicability(e)
    );
    assert_eq!(best(&j)["tier"], 3);
}

/// The cross-check's resolution gate (XCHECK_MAX_REL): at a fixed 300 inputs a bf16 GEMM's
/// P(differ) of ~3e-5 has a +-30% Tier 3 interval, inside which a 0.82x Tier 2 passed the
/// tolerance; an agreement that wide confirms nothing, so the CLT is refused. Mutant xcheck-res
/// (no resolution gate) accepts it.
#[test]
fn crosscheck_needs_resolution() {
    let j = run(
        "xres",
        "fixtures/c_mm_bf16_nnt",
        "fixtures/errdist/normal.json",
        &["--mc", "300"],
        &[],
    );
    let e = dag_clt(&j).expect("the DAG CLT runs");
    assert_eq!(e["applicable"], false, "{}", applicability(e));
    assert!(
        applicability(e).contains("cross-check resolution"),
        "{}",
        applicability(e)
    );
    assert_eq!(best(&j)["tier"], 3);
}

/// N-ERRDIST-SEL: a store under a guard computed from input data is a selection.
/// fixtures/errdist/adv_sel: out = (s > 0) ? x0 + ... + x15 : x0, each branch half the time. The
/// decision input s is an input element, the same in the kernel and in exact arithmetic, so the
/// branch cannot flip: no flip bound applies (`branch_flips` absent), the linearized tier
/// follows the kernel's branch at each nominal input, and it stands on its Tier 3 cross-check
/// (0.240 against 0.241). Until 2026-11-10 every program with a selection refused it; decision
/// inputs that carry a rounding are covered by errdist_flip.rs.
#[test]
fn selection_with_exact_decision_inputs_needs_no_bound() {
    let j = run(
        "sel",
        "fixtures/errdist/adv_sel",
        "fixtures/errdist/adv_sel.json",
        &[],
        &[],
    );
    assert!(
        j["assumptions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a == "N-ERRDIST-SEL"),
        "{}",
        j["assumptions"]
    );
    assert!(j.get("branch_flips").is_none(), "{}", j["branch_flips"]);
    let (p3, lo, _) = p(tier(&j, 3)[0]);
    assert!(lo > 0.0 && p3 < 0.5, "{p3}");
    let t2 = tier(&j, 2);
    assert!(!t2.is_empty(), "{j}");
    for e in t2 {
        let a = applicability(e);
        assert!(!a.contains("branch flips"), "{a}");
        assert_eq!(e["applicable"], true, "{a}");
        assert!(
            a.contains("Tier 3 cross-check") && !a.contains("FAILS"),
            "{a}"
        );
        assert!((p(e).0 / p3 - 1.0).abs() < 0.05, "{} vs {p3}", p(e).0);
    }
    assert_eq!(best(&j)["tier"], 2);
}

/// MISS 41 (N-ERRDIST-CLT, docs/errdist.md §3.2): an f32 sum of integers -8..8 is exact (every
/// partial sum is an integer below 2^24), so P(differ) = 0. The DAG CLT takes f32 leaves as
/// smooth (`tier2::leaf_laws`, `laws::f32_smooth`: random low bits) and says 0.939; without
/// Tier 3 (`--mc 0`) it is the headline. Un-ignore when the leaf lattice (or a structural
/// refusal) lands.
#[test]
fn dag_clt_sees_the_lattice_of_integer_f32_inputs() {
    for spec in [
        "fixtures/errdist/integers.json",
        "fixtures/errdist/grid64.json",
    ] {
        let j = run("miss41", "fixtures/c_sum_naive", spec, &["--mc", "0"], &[]);
        if let Some(e) = dag_clt(&j) {
            assert!(
                e["applicable"] == false || p(e).0 == 0.0,
                "{spec}: DAG CLT applicable with P(differ) {} on an exact sum",
                p(e).0
            );
        }
        if !j["best"].is_null() {
            assert_eq!(p(best(&j)).0, 0.0, "{spec}: headline {}", best(&j));
        }
        // With Tier 3 the leaves' lattice shows: the CLT says 0 (1e-12 on grid64: its normal
        // node laws put that mass past 2^24) and stands on its cross-check.
        let j = run("miss41t3", "fixtures/c_sum_naive", spec, &[], &[]);
        let e = dag_clt(&j).expect("the DAG CLT runs");
        assert!(p(e).0 < 1e-9, "{spec}: {} {}", p(e).0, applicability(e));
        assert_eq!(e["applicable"], true, "{spec}: {}", applicability(e));
    }
}

/// MISS 43 (docs/errdist.md §3.2): without Tier 3 (`--mc 0`) the DAG CLT is applicable on its
/// structural checks (domain, y's excess kurtosis), which pass estimates 1.1-1.5x off: the bf16
/// tcgen05 tile with f32 output says 0.670 against Tier 3's 0.448 (5,000 inputs), the cuBLAS
/// bf16 GEMM's f32 accumulators (`--pre-round`) the same 0.670 against 0.447, and the Triton
/// bf16 dot product 1.07e-4 against GB300's 7.25e-5. Without Tier 3 only a Tier 0 certificate
/// may headline; un-ignore when the CLT is "unconfirmed (no Tier 3)" there.
#[test]
#[ignore = "heavy: 51 s"]
fn dag_clt_is_unconfirmed_without_tier3() {
    for (case, extra) in [
        ("fixtures/r1a_tcgen05_tile", None),
        ("fixtures/c_mm_bf16_nnt", Some("--pre-round")),
        ("fixtures/c_triton_dot_f32acc", None),
    ] {
        let mut args = vec!["--mc", "0"];
        args.extend(extra);
        let j = run("miss43", case, "fixtures/errdist/normal.json", &args, &[]);
        if let Some(e) = dag_clt(&j) {
            assert_eq!(
                e["applicable"],
                false,
                "{case}: DAG CLT applicable without Tier 3 at P(differ) {}",
                p(e).0
            );
        }
    }
}

/// MISS 44 (Tier 0 enumeration, docs/errdist.md §5.1): a bf16 tcgen05 tile with f32 output under
/// 768 (90%) or 1 (10%) per element (lattice 2^0, partial sums to 2^25.2: the certificate fails,
/// enumeration runs). Tier 0 says P(differ) 0.218, "exact", and headlines; Tier 3 says 0.0660
/// [0.0659, 0.0661] and the simulator agrees with Tier 3 (0.061 over 4 inputs). The means agree:
/// Tier 0 counts every nonzero (sub-ulp) error of the f32 output as a difference, Tier 3 counts
/// C != RN(exact). Un-ignore when Tier 0's f32 P(differ) is against RN(exact).
#[test]
fn tier0_enumeration_matches_tier3_on_f32_outputs() {
    let j = run(
        "miss44",
        "fixtures/r1a_tcgen05_tile",
        "fixtures/errdist/adv_t0_window.json",
        &["--mc", "20000"],
        &[],
    );
    let (p3, lo, hi) = p(tier(&j, 3)[0]);
    assert!(p3 > 0.05 && p3 < 0.08, "{p3}");
    assert!(!tier(&j, 0).is_empty(), "{j}");
    for e in tier(&j, 0) {
        // Tier 0 is checked against Tier 3's interval.
        assert!(
            applicability(e).contains("Tier 3 check"),
            "{}",
            applicability(e)
        );
        let p0 = p(e).0;
        let h = 0.5 * (hi - lo);
        assert!(
            (p0 - p3).abs() <= (0.01 * p3).max(3.0 * h),
            "tier 0 ({}) {p0} vs tier 3 {p3}",
            method(e)
        );
    }
}

/// `--uniform` adds the Tier 1 estimate under the uniform-residue heuristic after the Tier 3
/// cross-check: a comparison, never the headline. On the bf16 GEMM it says 6.59e-5 against
/// Tier 3's 2.85e-5, and until 2026-11-10 it was the summary's headline (the heuristic Tier 1
/// came before Tier 3 in the headline order).
#[test]
fn uniform_heuristic_is_never_the_headline() {
    let j = run(
        "uniform",
        "fixtures/c_mm_bf16_nnt",
        "fixtures/errdist/normal.json",
        &["--uniform"],
        &[],
    );
    let u: Vec<_> = ests(&j)
        .iter()
        .filter(|e| method(e).ends_with("[uniform-residue heuristic]"))
        .collect();
    assert!(!u.is_empty(), "{j}");
    for e in u {
        assert_eq!(e["applicable"], false, "{}", applicability(e));
    }
    assert!(
        !method(best(&j)).ends_with("[uniform-residue heuristic]"),
        "{}",
        method(best(&j))
    );
}

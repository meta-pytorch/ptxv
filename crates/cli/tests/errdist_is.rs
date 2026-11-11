/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Importance sampling for Tier 3 (`ptxv errdist --is`, docs/errdist.md §5.4.1): alpha = 1
//! repeats plain sampling bit for bit, tilted proposals agree with plain where plain resolves
//! P(differ), a tail probability plain sampling cannot see comes out within 2% of its closed
//! form, and a degenerate proposal is refused. The library tests in `errdist/is.rs` pin log w
//! (defensive term, per-element densities) and the unnormalized estimator.

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run(
    tag: &str,
    case: &str,
    spec: &str,
    prop: &str,
    n: usize,
    env: &[(&str, &str)],
) -> serde_json::Value {
    let js = std::env::temp_dir().join(format!("ptxv-test-is-{tag}-{}.json", std::process::id()));
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"));
    cmd.args(["errdist", case, "--inputs", spec, "--is", prop])
        .args(["--mc", &n.to_string()])
        .args(["--json", js.to_str().unwrap()])
        .current_dir(root())
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("PTXV_XCHECK_SECONDS", "0")
        .env("PTXV_NL_SECONDS", "0")
        .env_remove("PTXV_IS_FIRST");
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

fn tier3(j: &serde_json::Value) -> &serde_json::Value {
    j["estimates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["tier"] == 3)
        .unwrap()
}

fn triple(v: &serde_json::Value) -> (f64, f64, f64) {
    (
        v[0].as_f64().unwrap(),
        v[1].as_f64().unwrap(),
        v[2].as_f64().unwrap(),
    )
}

/// alpha = 1 on the plain run's own inputs (PTXV_IS_FIRST=0): the same P(differ) and the same
/// PMF bins, bit for bit (bins one input dominates are left out by the share gate, not
/// changed).
#[test]
fn is_alpha_one() {
    let j = run(
        "a1",
        "fixtures/c_sum_naive",
        "fixtures/errdist/uniform01.json",
        "fixtures/errdist/is/alpha_one.json",
        3000,
        &[("PTXV_IS_FIRST", "0")],
    );
    let is = &j["is_estimate"];
    assert_eq!(is["sampling"], "is", "{j}");
    assert_eq!(is["ess"].as_f64().unwrap(), 3000.0);
    let t3 = tier3(&j);
    assert_eq!(
        t3["p_differ"][0].as_f64().unwrap().to_bits(),
        is["p_differ"][0].as_f64().unwrap().to_bits()
    );
    let plain: std::collections::BTreeMap<i64, u64> = t3["pmf_ulp"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| (b[0].as_i64().unwrap(), b[1].as_f64().unwrap().to_bits()))
        .collect();
    let bins = is["pmf_ulp"].as_array().unwrap();
    assert!(bins.len() >= 10, "{bins:?}");
    for b in bins {
        let k = b[0].as_i64().unwrap();
        assert_eq!(
            Some(&b[1].as_f64().unwrap().to_bits()),
            plain.get(&k),
            "bin {k}"
        );
    }
}

/// Tilted proposals (alpha 0.2) against plain sampling on rows where plain resolves P(differ)
/// to about 2% at 20,000 inputs: the gate itself requires agreement within the combined
/// 99.9% interval (a disagreement is a refusal), and this test checks it again.
#[test]
fn is_vs_plain() {
    for (tag, case, spec, prop) in [
        (
            "softmax",
            "fixtures/c_softmax_stable",
            "fixtures/errdist/normal.json",
            "fixtures/errdist/is/softmax_perelem.json",
        ),
        (
            "dot",
            "fixtures/c_dot_bf16acc",
            "fixtures/errdist/uniform01.json",
            "fixtures/errdist/is/dot_hist.json",
        ),
        (
            "sum",
            "fixtures/c_sum_naive",
            "fixtures/errdist/uniform01.json",
            "fixtures/errdist/is/sum_hist.json",
        ),
    ] {
        let j = run(tag, case, spec, prop, 20_000, &[]);
        let is = &j["is_estimate"];
        assert!(is.is_object(), "{tag}: {}", j["is_refused"]);
        let (p, lo, hi) = triple(&tier3(&j)["p_differ"]);
        let (q, qlo, qhi) = triple(&is["p_differ"]);
        // Plain resolves it to about 2%.
        assert!((hi - lo) / 2.0 < 0.03 * p, "{tag}: plain {p} [{lo}, {hi}]");
        let (sp, sq) = ((hi - lo) / (2.0 * 1.96), (qhi - qlo) / (2.0 * 1.96));
        let tol = 3.29 * (sp * sp + sq * sq).sqrt();
        assert!((p - q).abs() <= tol, "{tag}: IS {q} vs plain {p} (+-{tol})");
        // The tilt moved the weights: not plain sampling in disguise.
        let n = is["n"].as_f64().unwrap();
        let ess = is["ess"].as_f64().unwrap();
        assert!(ess < 0.9 * n && ess > n / 50.0, "{tag}: ESS {ess} of {n}");
    }
}

/// out = ex2.approx.f32(x * log2 e) (Triton tl.exp, c_exp_one), x ~ N(80, 2): the output is
/// +inf exactly when RN(x * log2 e) >= 128, i.e. RN_f32(x) >= t, t the smallest such f32
/// (88.72284). P = 1 - Phi((m - 80) / 2), m the midpoint below t, about 6.46e-6: plain
/// sampling expects 0.13 events in 20,000 inputs. The IS estimate is within 2%.
#[test]
fn is_known_answer() {
    let c = f32::from_bits(0x3FB8_AA3B);
    let (mut a, mut b) = (88.0f32.to_bits(), 89.5f32.to_bits());
    while b - a > 1 {
        let m = a + (b - a) / 2;
        if f32::from_bits(m) * c >= 128.0 {
            b = m;
        } else {
            a = m;
        }
    }
    let mid = 0.5 * (f32::from_bits(a) as f64 + f32::from_bits(b) as f64);
    let want = 1.0 - ptxv_sim::errdist::special::phi((mid - 80.0) / 2.0);
    assert!(want > 6.4e-6 && want < 6.5e-6, "{want}");
    let j = run(
        "ka",
        "fixtures/c_exp_one",
        "fixtures/errdist/is/exp80.json",
        "fixtures/errdist/is/exp80_tail.json",
        20_000,
        &[],
    );
    let is = &j["is_estimate"];
    assert!(is.is_object(), "{}", j["is_refused"]);
    let (p, lo, hi) = triple(&is["p_nonfinite"]);
    assert!(
        (p - want).abs() < 0.02 * want,
        "IS P(non-finite) {p} [{lo}, {hi}] vs {want}"
    );
    assert!(lo <= want && want <= hi, "{want} outside [{lo}, {hi}]");
    // Plain sampling saw (almost surely) nothing.
    assert!(j["p_nonfinite"]["element"][0].as_f64().unwrap() < 1e-3);
}

/// A proposal concentrated in a 0.1-wide window (scale 0.05) with alpha 0.001: the
/// effective sample size collapses and the ESS gate refuses it (the share gate would too; the
/// test pins the ESS gate).
#[test]
fn is_refuse_degenerate() {
    let j = run(
        "deg",
        "fixtures/c_exp_one",
        "fixtures/errdist/is/exp80.json",
        "fixtures/errdist/is/exp80_degenerate.json",
        20_000,
        &[],
    );
    assert!(j["is_estimate"].is_null(), "{}", j["is_estimate"]);
    let r = j["is_refused"].as_str().unwrap();
    assert!(r.starts_with("ESS "), "{r}");
    assert!(
        j["notes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n.as_str().unwrap().starts_with("IS refused: ESS")),
        "{}",
        j["notes"]
    );
}

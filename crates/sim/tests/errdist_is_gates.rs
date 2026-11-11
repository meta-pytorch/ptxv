/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Adversarial rows for N-ERRDIST-IS: each gate of `errdist::is::summarize` (docs/errdist.md
//! §5.4.1) alone refuses a weighted sample built to pass every other gate, and the estimator is
//! the unnormalized sum w f / n. Each test is the target of a source mutant in
//! tools/mutate/srcmut.py (is-ess, is-share, is-agree, is-logw, is-selfnorm).

use ptxv_sim::errdist::is::IsRec;
use ptxv_sim::errdist::is::PlainRef;
use ptxv_sim::errdist::is::summarize;

fn rec(t: u64, logw: f64, differ: f64) -> IsRec {
    IsRec {
        t,
        logw,
        differ,
        nonfinite: 0.0,
        pmf: vec![(0, 1.0 - differ), (1, differ)],
    }
}

/// n inputs, weight 1, P(differ) 0.25 spread over every fourth input.
fn even(n: u64) -> Vec<IsRec> {
    (0..n)
        .map(|t| rec(t, 0.0, if t % 4 == 0 { 1.0 } else { 0.0 }))
        .collect()
}

#[test]
fn control_passes_every_gate() {
    let s = summarize(&even(4000), [None, None]).unwrap();
    assert_eq!(s.ess, 4000.0);
    assert!((s.p_differ.0 - 0.25).abs() < 1e-12, "{:?}", s.p_differ);
}

/// ESS >= max(200, n / 50): one input with weight e^12 among 4000 of weight 1 has ESS ~ 1.
/// Its differing output keeps the share gate from passing too, so the ESS gate must be the
/// reason.
#[test]
fn ess_gate_refuses_one_dominant_weight() {
    let mut r = even(4000);
    r[1] = rec(1, 12.0, 0.0);
    let e = summarize(&r, [None, None]).unwrap_err();
    assert!(e.starts_with("ESS"), "{e}");
}

/// Largest share <= 5%: 4000 inputs of weight 1 (ESS 4000), the only differing output in one
/// input: that input holds all of P(differ).
#[test]
fn share_gate_refuses_one_input_holding_p_differ() {
    let mut r: Vec<IsRec> = (0..4000).map(|t| rec(t, 0.0, 0.0)).collect();
    r[7] = rec(7, 0.0, 1.0);
    let e = summarize(&r, [None, None]).unwrap_err();
    assert!(e.contains("holds"), "{e}");
}

/// Agreement with plain where plain resolves the value: IS 0.25 against plain 0.20 +- 0.002
/// (|difference| 25 standard errors).
#[test]
fn agreement_gate_refuses_a_biased_estimate() {
    let plain = PlainRef { p: 0.2, se: 0.002 };
    let e = summarize(&even(4000), [Some(plain), None]).unwrap_err();
    assert!(e.contains("disagree"), "{e}");
    // Plain that does not resolve the value (95% half-width over 50%) is not a reference.
    let weak = PlainRef { p: 0.2, se: 0.2 };
    assert!(summarize(&even(4000), [Some(weak), None]).is_ok());
}

/// Finite log w: one NaN weight refuses.
#[test]
fn logw_gate_refuses_nan() {
    let mut r = even(4000);
    r[3].logw = f64::NAN;
    let e = summarize(&r, [None, None]).unwrap_err();
    assert!(e.starts_with("log w"), "{e}");
}

/// The estimate is sum w f / n, not the self-normalized sum w f / sum w: with every weight 1/2
/// (log w = -ln 2) the unnormalized estimate is half the self-normalized one.
#[test]
fn estimator_is_unnormalized() {
    let r: Vec<IsRec> = even(4000)
        .into_iter()
        .map(|mut x| {
            x.logw = -std::f64::consts::LN_2;
            x
        })
        .collect();
    let s = summarize(&r, [None, None]).unwrap();
    assert!((s.p_differ.0 - 0.125).abs() < 1e-12, "{:?}", s.p_differ);
}

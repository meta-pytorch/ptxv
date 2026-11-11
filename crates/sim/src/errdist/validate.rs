/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Predicted against observed: two-sample tests between an estimate's samples (Tier 3) or
//! CDF (analytic tiers) and errors observed on simulator / GPU runs of the same spec.

use super::mc::Obs;
use super::report::Validation;
use super::stats::*;

/// Compare a Monte Carlo prediction (its sample) with observed errors.
pub fn against_sample(
    source: &str,
    against: &str,
    pred: &Obs,
    obs: &Obs,
    mismatch: u64,
) -> Validation {
    let mut a: Vec<f64> = pred.sample.iter().map(|x| x.0).collect();
    let mut b: Vec<f64> = obs.sample.iter().map(|x| x.0).collect();
    // NaN errors (a NaN output, or inf - inf) have no place in the sorted sample.
    let before = a.len() + b.len();
    a.retain(|x| !x.is_nan());
    b.retain(|x| !x.is_nan());
    let nans = before - a.len() - b.len();
    sort(&mut a);
    sort(&mut b);
    let (ks_d, ks_p) = ks2(&a, &b, pred.inputs as f64, obs.inputs as f64);
    let (ad, ad_p) = ad2(&a, &b);
    let qq = QUANTILES
        .iter()
        .map(|&p| (p, quantile(&a, p), quantile(&b, p)))
        .collect();
    let pd = |o: &Obs| o.differ as f64 / o.compared.max(1) as f64;
    let ci = super::mc::p_interval(obs, 0.05);
    // Batch-means CI of the observed mean (u scale).
    let bm: Vec<f64> = obs.input_means.iter().map(|x| x.0).collect();
    let m = bm.iter().sum::<f64>() / bm.len().max(1) as f64;
    let v = bm.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (bm.len().max(2) - 1) as f64;
    let hw = 1.96 * (v / bm.len().max(1) as f64).sqrt();
    let mut notes = vec![];
    if nans > 0 {
        notes.push(format!(
            "{nans} NaN errors left out of the distribution tests"
        ));
    }
    // A NaN or infinite error (a kernel's NaN / inf output) has no mean: P(differ) and the
    // distribution tests carry the comparison.
    let finite = pred.u.mean().is_finite() && obs.u.mean().is_finite();
    if !finite {
        notes.push("mean (u scale) not compared: non-finite errors".into());
    }
    let mean_ok =
        !finite || (pred.u.mean() - obs.u.mean()).abs() <= hw + 1e-12 + 2.0 * pred_hw(pred);
    if !mean_ok {
        notes.push(format!(
            "mean (u scale) differs: predicted {:.4e}, observed {:.4e} +- {:.2e}",
            pred.u.mean(),
            obs.u.mean(),
            hw
        ));
    }
    // The predicted P(differ) must lie in the observed 99.9% interval (sampling noise of the
    // prediction included: its own interval must overlap).
    let pci = super::mc::p_interval(pred, 0.001);
    let oci = super::mc::p_interval(obs, 0.001);
    // Outputs of one input are correlated: also accept within 3.29 batch-means standard
    // errors over inputs (99.9%).
    let bm_ok = match (bm_se(&pred.input_differ), bm_se(&obs.input_differ)) {
        (Some(a), Some(b)) => (pd(pred) - pd(obs)).abs() <= 3.29 * (a * a + b * b).sqrt(),
        _ => false,
    };
    let differ_ok = (pci.0 <= oci.1 && oci.0 <= pci.1) || bm_ok;
    if !differ_ok {
        notes.push(format!(
            "P(differ): predicted {:.4e}, observed {:.4e} [{:.4e}, {:.4e}]",
            pd(pred),
            pd(obs),
            ci.0,
            ci.1
        ));
    }
    // NaN p-value counts as not rejected.
    let ks_ok = ks_p.partial_cmp(&1e-3) != Some(std::cmp::Ordering::Less);
    if !ks_ok {
        notes.push(format!(
            "KS rejects (D = {ks_d:.4}, p = {ks_p:.2e}, effective sizes = inputs)"
        ));
    }
    if mismatch > 0 {
        notes.push(format!(
            "{mismatch} outputs differ bitwise from the DAG evaluation"
        ));
    }
    Validation {
        source: source.into(),
        against: against.into(),
        inputs: obs.inputs,
        outputs: obs.compared,
        mean_pred: pred.u.mean(),
        mean_obs: obs.u.mean(),
        std_pred: pred.u.var().sqrt(),
        std_obs: obs.u.var().sqrt(),
        p_differ_pred: Some(pd(pred)),
        p_differ_obs: Some((pd(obs), ci.0, ci.1)),
        ks_d,
        ks_p,
        ad,
        ad_p,
        qq,
        pmf_tv: pmf_tv(&obs_pmf(pred), &obs_pmf(obs)),
        pmf_ulp: obs_pmf(obs),
        pmf_ulp_pred: obs_pmf(pred),
        dag_mismatch: mismatch,
        agree: mean_ok && differ_ok && ks_ok && mismatch == 0,
        notes,
    }
}

/// An observation's PMF on the output-ulp axis (empty when it is off the lattice).
pub fn obs_pmf(o: &Obs) -> Vec<(i64, f64)> {
    if o.off_lattice > 0 && o.off_lattice * 1000 > o.compared {
        return vec![];
    }
    let t = o.compared.max(1) as f64;
    o.pmf_ulp.iter().map(|(k, v)| (*k, *v as f64 / t)).collect()
}

/// Total-variation distance of two PMFs (None if either is empty).
pub fn pmf_tv(a: &[(i64, f64)], b: &[(i64, f64)]) -> Option<f64> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let mut m: std::collections::BTreeMap<i64, f64> = Default::default();
    for &(k, q) in a {
        *m.entry(k).or_default() += q;
    }
    for &(k, q) in b {
        *m.entry(k).or_default() -= q;
    }
    Some(0.5 * m.values().map(|x| x.abs()).sum::<f64>())
}

/// Batch-means standard error of a per-input statistic (None below 2 inputs).
fn bm_se(x: &[f64]) -> Option<f64> {
    if x.len() < 2 {
        return None;
    }
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    Some((x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt())
}

fn pred_hw(o: &Obs) -> f64 {
    let bm: Vec<f64> = o.input_means.iter().map(|x| x.0).collect();
    let m = bm.iter().sum::<f64>() / bm.len().max(1) as f64;
    let v = bm.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (bm.len().max(2) - 1) as f64;
    1.96 * (v / bm.len().max(1) as f64).sqrt()
}

/// Compare an analytic estimate (tier 0-2) with observed errors: P(differ) against the
/// observed 99.9% interval, the ulp-scale mean / std and quantiles side by side (no
/// two-sample test: the estimate carries no sample).
pub fn against_estimate(
    source: &str,
    est: &super::report::Estimate,
    obs: &Obs,
    mismatch: u64,
) -> Validation {
    let pd = obs.differ as f64 / obs.compared.max(1) as f64;
    let ci = super::mc::p_interval(obs, 0.05);
    let oci = super::mc::p_interval(obs, 0.001);
    let mut b: Vec<f64> = obs.sample.iter().map(|x| x.1).collect();
    b.retain(|x| !x.is_nan());
    sort(&mut b);
    let mut notes = vec![];
    let p = est.p_differ.map(|x| x.0);
    let bm = bm_se(&obs.input_differ);
    let differ_ok = p.is_some_and(|p| {
        (p >= oci.0 && p <= oci.1) || bm.is_some_and(|se| (p - pd).abs() <= 3.29 * se)
    });
    if let Some(se) = bm {
        notes.push(format!(
            "observed P(differ) batch-means standard error over {} inputs: {se:.3e}",
            obs.input_differ.len()
        ));
    }
    if let Some(p) = p {
        notes.push(format!(
            "P(differ): predicted {p:.4e}, observed {pd:.4e} [{:.4e}, {:.4e}] (99.9%: [{:.4e}, {:.4e}]){}",
            ci.0,
            ci.1,
            oci.0,
            oci.1,
            if pd > 0.0 { format!("; ratio {:.3}", p / pd) } else { String::new() }
        ));
    }
    if mismatch > 0 {
        notes.push(format!(
            "{mismatch} outputs differ bitwise from the DAG evaluation"
        ));
    }
    let um = est.ulp.as_ref();
    let qq = QUANTILES
        .iter()
        .map(|&q| {
            (
                q,
                um.and_then(|m| m.quantiles.iter().find(|x| x.0 == q).map(|x| x.1))
                    .unwrap_or(f64::NAN),
                quantile(&b, q),
            )
        })
        .collect();
    notes.push("mean / std / quantiles in output ulps of y's binade (ulp scale)".into());
    Validation {
        source: source.into(),
        against: format!("tier {}", est.tier),
        inputs: obs.inputs,
        outputs: obs.compared,
        mean_pred: um.map_or(f64::NAN, |m| m.mean),
        mean_obs: obs.ulp.mean(),
        std_pred: um.map_or(f64::NAN, |m| m.std),
        std_obs: obs.ulp.var().sqrt(),
        p_differ_pred: p,
        p_differ_obs: Some((pd, ci.0, ci.1)),
        ks_d: f64::NAN,
        ks_p: f64::NAN,
        ad: f64::NAN,
        ad_p: f64::NAN,
        qq,
        pmf_tv: pmf_tv(&est.pmf_ulp, &obs_pmf(obs)),
        pmf_ulp: obs_pmf(obs),
        pmf_ulp_pred: est.pmf_ulp.clone(),
        dag_mismatch: mismatch,
        agree: differ_ok && mismatch == 0,
        notes,
    }
}

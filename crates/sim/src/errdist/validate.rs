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
    let exact = p_contradicts(super::mc::p_interval(pred, 0.05), obs);
    let differ_ok = p_differ_agrees(pred, obs) && exact.is_none();
    if !differ_ok {
        notes.push(format!(
            "P(differ): predicted {:.4e}, observed {:.4e} [{:.4e}, {:.4e}]",
            pd(pred),
            pd(obs),
            ci.0,
            ci.1
        ));
    }
    if let Some((lo, hi)) = exact {
        notes.push(format!(
            "P(differ): the prediction's interval is outside the observed inputs' exact bounds [{lo:.4e}, {hi:.4e}] (99.9%, Clopper-Pearson over inputs)"
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

/// Whether a Monte Carlo prediction's P(differ) agrees with an observed run's: the predicted
/// P(differ) lies in the observed 99.9% interval with the prediction's own sampling noise (the
/// two intervals overlap), or, outputs of one input being correlated, the two are within 3.29
/// batch-means standard errors over inputs (99.9%).
pub fn p_differ_agrees(pred: &Obs, obs: &Obs) -> bool {
    let pd = |o: &Obs| o.differ as f64 / o.compared.max(1) as f64;
    let pci = super::mc::p_interval(pred, 0.001);
    let oci = super::mc::p_interval(obs, 0.001);
    let bm_ok = match (bm_se(&pred.input_differ), bm_se(&obs.input_differ)) {
        (Some(a), Some(b)) => (pd(pred) - pd(obs)).abs() <= 3.29 * (a * a + b * b).sqrt(),
        _ => false,
    };
    (pci.0 <= oci.1 && oci.0 <= pci.1) || bm_ok
}

/// Thresholds of the per-input fractions for `p_exact_bounds`: 1/64 .. 1, doubling.
const P_GRID: [f64; 7] = [
    1.0 / 64.0,
    1.0 / 32.0,
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
    1.0,
];

/// Exact bounds at level 1 - `alpha` on P(differ) from a run's inputs alone (the independent
/// unit; outputs of one input can differ together): P is the mean over inputs of the fraction
/// f of outputs that differ, E f = integral over c of Pr(f > c), bounded on each step of
/// `P_GRID` by Clopper-Pearson bounds on the share of inputs with f >= c_j (below) and
/// f > c_{j-1} (above), Bonferroni over the 2 x 7 bounds. Exact at any n, coarse (a step of
/// the grid): it catches a discrepancy confined to a few inputs, which the batch-means
/// interval (`p_differ_agrees`) passes at any n when 10 or fewer inputs carry it. None
/// without inputs.
pub fn p_exact_bounds(obs: &Obs, alpha: f64) -> Option<(f64, f64)> {
    let f = &obs.input_differ;
    let n = f.len() as u64;
    if n == 0 {
        return None;
    }
    let tail = 2.0 * alpha / (2.0 * P_GRID.len() as f64);
    let (mut lo, mut hi, mut prev) = (0.0, 0.0, 0.0);
    for &c in &P_GRID {
        let at_least = f.iter().filter(|&&x| x >= c).count() as u64;
        let above = f.iter().filter(|&&x| x > prev).count() as u64;
        lo += (c - prev) * super::special::clopper_pearson(at_least, n, tail).0;
        hi += (c - prev) * super::special::clopper_pearson(above, n, tail).1;
        prev = c;
    }
    Some((lo, hi.min(1.0)))
}

/// A claimed P(differ) interval that a run's inputs contradict: the claim lies outside the
/// run's exact 99.9% bounds (`p_exact_bounds`), which are returned. The check against a
/// headline's own interval that the batch-means test misses: a P(differ) of 0 against GB300
/// runs where every output of 4 of 400 inputs differs (an FTZ the DAG lacked) agreed.
pub fn p_contradicts(claim: (f64, f64), obs: &Obs) -> Option<(f64, f64)> {
    let (lo, hi) = p_exact_bounds(obs, 0.001)?;
    (claim.1 < lo || claim.0 > hi).then_some((lo, hi))
}

/// Whether a Monte Carlo prediction's error distribution and a run's agree at 99.9% with
/// inputs as the unit: the mean (u scale) within 3.29 batch-means standard errors, the KS test
/// at effective sizes = inputs, and the Anderson-Darling test on the per-input means (not on
/// outputs, whose dependence within an input makes its p-value optimistic). A weak test: with
/// the input counts as effective sizes the KS flags nothing when 1-2% of the inputs carry a
/// discrepancy, and the Anderson-Darling test's power on per-input means was not measured. A
/// discrepancy confined to few inputs is the exact bounds' to refuse (`p_contradicts`, which
/// `tier3_refusal` checks first); this test adds the error's distribution, no more.
pub fn moments_agree(pred: &Obs, obs: &Obs) -> bool {
    let means = |o: &Obs| -> Vec<f64> {
        let mut m: Vec<f64> = o
            .input_means
            .iter()
            .map(|x| x.0)
            .filter(|x| x.is_finite())
            .collect();
        sort(&mut m);
        m
    };
    let (ma, mb) = (means(pred), means(obs));
    if ma.len() < 2 || mb.len() < 2 {
        return true;
    }
    let mv = |m: &[f64]| {
        let n = m.len() as f64;
        let mu = m.iter().sum::<f64>() / n;
        let v = m.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0);
        (mu, v / n)
    };
    let ((mua, sa), (mub, sb)) = (mv(&ma), mv(&mb));
    let mean_ok = (mua - mub).abs() <= 3.29 * (sa + sb).sqrt() + 1e-12;
    let mut a: Vec<f64> = pred.sample.iter().map(|x| x.0).collect();
    let mut b: Vec<f64> = obs.sample.iter().map(|x| x.0).collect();
    a.retain(|x| !x.is_nan());
    b.retain(|x| !x.is_nan());
    sort(&mut a);
    sort(&mut b);
    let (_, ks_p) = ks2(&a, &b, pred.inputs as f64, obs.inputs as f64);
    let (_, ad_p) = ad2(&ma, &mb);
    let rejects = |p: f64| p.partial_cmp(&1e-3) == Some(std::cmp::Ordering::Less);
    mean_ok && !rejects(ks_p) && !rejects(ad_p)
}

/// Tier 3's verdict from a simulator or GPU validation run (`source`), on the criteria that
/// refuse an analytic estimate (`against_estimate`): the reason to refuse it when outputs differ
/// bit for bit from the DAG where no other order of an atomic chain explains the run's value
/// (`mismatch` counts every differing output, `atomic` the explained ones: GB300 adds those in
/// its own order, MISS 58), when its P(differ) and the run's disagree (`p_differ_agrees`), when
/// the run's inputs contradict Tier 3's own interval `claim` (`p_contradicts`), or, with
/// explained atomic mismatches, when the error distributions disagree (`moments_agree`, a weak
/// test: the exact bounds refuse what few inputs carry, the moments only add the error's
/// distribution).
/// Without any bitwise mismatch Tier 3 and the run sample one DAG and the moments can only
/// raise sampling alarms (the mean test runs at 95% on 4 default inputs); an atomic chain in
/// GB300's order is another DAG, and the moments count there. None when either side compared
/// no output (out of domain, outside these statistics). Until 2026-11-11 nothing refused Tier
/// 3, and the headline took it whatever a validation said (MISS 42's rule).
pub fn tier3_refusal(
    source: &str,
    pred: &Obs,
    obs: &Obs,
    mismatch: u64,
    atomic: u64,
    claim: (f64, f64),
) -> Option<String> {
    if pred.compared == 0 || obs.compared == 0 {
        return None;
    }
    let exact = mismatch.saturating_sub(atomic);
    if exact > 0 {
        return Some(format!(
            "{source} validation ({} inputs): {exact} outputs differ bit for bit from the DAG where no other order of an atomic chain explains it FAILS",
            obs.inputs
        ));
    }
    let pd = |o: &Obs| o.differ as f64 / o.compared.max(1) as f64;
    if !p_differ_agrees(pred, obs) {
        let (lo, hi) = super::mc::p_interval(obs, 0.001);
        return Some(format!(
            "{source} validation ({} inputs): Tier 3's P(differ) {:.3e} against observed {:.3e} [{lo:.3e}, {hi:.3e}] (99.9%) DISAGREES FAILS",
            obs.inputs,
            pd(pred),
            pd(obs)
        ));
    }
    if let Some((lo, hi)) = p_contradicts(claim, obs) {
        return Some(format!(
            "{source} validation ({} inputs): Tier 3's interval [{:.3e}, {:.3e}] against the observed inputs' exact bounds [{lo:.3e}, {hi:.3e}] (99.9%, Clopper-Pearson over inputs) DISAGREES FAILS",
            obs.inputs, claim.0, claim.1
        ));
    }
    if atomic > 0 && !moments_agree(pred, obs) {
        return Some(format!(
            "{source} validation ({} inputs): {atomic} outputs in the run's own order of atomic adds, and the error distributions disagree (mean, KS or Anderson-Darling over inputs, 99.9%) FAILS",
            obs.inputs
        ));
    }
    None
}

/// The GPU validation's verdict on the DAG (MISS 58): the refusal, if GB300's outputs differ
/// bit for bit from the DAG's at a tracked output where no other order of an atomic chain
/// explains GB300's value (`mismatch` counts every differing output, `atomic` the explained
/// ones, over `inputs` inputs; `AtomChain::order_explains`: an output whose value is an atomic
/// chain of two or more sources and three or more operands, within the reordering bound).
/// GB300 adds in its own order, the DAG in the simulator's (N-ERRDIST-DAG). Until 2026-11-10
/// a mismatch printed DISAGREE next to the headline; until 2026-11-11 every output whose DAG
/// held an atomic add was exempt, an FTZ the DAG lacked and errors upstream of the adds
/// included.
// ASSUME(N-ERRDIST-MC): Tier 3 is bit-exact per sample given the simulator and GPU validations.
pub fn gpu_dag_refusal(mismatch: u64, atomic: u64, inputs: u64) -> Option<String> {
    let exact = mismatch.saturating_sub(atomic);
    (exact > 0).then(|| {
        format!(
            "GB300 disagrees with the numerics DAG on {exact} tracked outputs over {inputs} validation inputs{}; refusing to report from it",
            if atomic > 0 {
                format!(
                    " (and on {atomic} outputs whose atomic adds GB300 ordered otherwise, within the reordering bound)"
                )
            } else {
                String::new()
            }
        )
    })
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
    // The estimate's own interval against the inputs' exact bounds as well (`p_contradicts`).
    let exact = est
        .p_differ
        .and_then(|(_, lo, hi)| p_contradicts((lo, hi), obs));
    let differ_ok = exact.is_none()
        && p.is_some_and(|p| {
            (p >= oci.0 && p <= oci.1) || bm.is_some_and(|se| (p - pd).abs() <= 3.29 * se)
        });
    if let Some((lo, hi)) = exact {
        notes.push(format!(
            "P(differ): the estimate's interval is outside the observed inputs' exact bounds [{lo:.4e}, {hi:.4e}] (99.9%, Clopper-Pearson over inputs)"
        ));
    }
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

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Importance sampling for Tier 3 (docs/errdist.md §5.4.1). A proposal q replaces the spec's
//! law p for some tensors' real draws (elements, axis scales, outlier channels); each input
//! carries log w = log p(draws) - log q(draws), both closed-form densities at the drawn point,
//! with q the defensive mixture alpha p + (1 - alpha) q_tilt (every weight <= 1 / alpha).
//! Quantization stays the spec's deterministic map, so no code pmf is needed. Estimates are
//! the unnormalized weighted means sum w f / n (p and q are normalized densities: unbiased),
//! reported only when the gates of `summarize` pass (fail closed).
//!
//! Proposal construction (the tail search) is not here: a `Proposal` comes from a file.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use serde::Deserialize;
use serde::Serialize;

use super::spec;
use super::spec::Dist;
use super::spec::InputError;
use super::spec::Layout;
use super::spec::Plan;
use super::spec::Sample;
use super::spec::Spec;
use super::special;

/// A proposal: per tensor, how its read elements and latents are drawn instead of the spec.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Proposal {
    pub name: String,
    /// Defensive mixture: with probability `alpha` a sample is drawn from the spec itself
    /// (0 < alpha <= 1; alpha = 1 is plain sampling). Bounds every weight by 1 / alpha.
    pub alpha: f64,
    #[serde(default)]
    pub description: String,
    pub tensors: BTreeMap<String, TensorProposal>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorProposal {
    #[serde(default)]
    pub elems: ElemProposal,
    /// Per `scales` entry (same order as the spec): the axis law with its own location and
    /// scale replaced (LogNormal: in the log domain, mu and sigma), or None (spec law).
    #[serde(default)]
    pub scales: Vec<Option<LocScale>>,
    /// Outlier-channel probability override, or None.
    #[serde(default)]
    pub outlier_prob: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElemProposal {
    /// The spec's law.
    #[default]
    Spec,
    /// Every read element: the spec law (Normal, Laplace, StudentT; LogNormal in the log
    /// domain) with location `loc` and scale `scale` (Normal: mean, std; Laplace: mean, scale;
    /// StudentT: mean, scale, same df; LogNormal: mu, sigma). Bounded laws: Histogram.
    LocScale(LocScale),
    /// Per read element index: (index, LocScale). Elements not listed: the spec law.
    PerElem(Vec<(u64, LocScale)>),
    /// Bounded laws (Uniform, Histogram): a piecewise-constant density covering the spec's
    /// support, edges strictly increasing, probs (each bin's probability) summing to 1.
    Histogram { edges: Vec<f64>, probs: Vec<f64> },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocScale {
    pub loc: f64,
    pub scale: f64,
}

impl Proposal {
    pub fn load(path: &std::path::Path) -> Result<Proposal, String> {
        let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let p: Proposal =
            serde_json::from_str(&s).map_err(|e| format!("{}: {e}", path.display()))?;
        if !(p.alpha > 0.0 && p.alpha <= 1.0) {
            return Err(format!("{}: alpha must be in (0, 1]", path.display()));
        }
        Ok(p)
    }

    /// One line: name, alpha, description.
    pub fn describe(&self) -> String {
        let mut s = format!("{} (alpha {})", self.name, self.alpha);
        if !self.description.is_empty() {
            s.push_str(": ");
            s.push_str(&self.description);
        }
        s
    }
}

const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

/// ln of the density of `d` at `x` in closed form; Err for laws without one (atoms).
pub fn ln_pdf(d: &Dist, x: f64) -> Result<f64, String> {
    Ok(match d {
        Dist::Normal { mean, std } => {
            let z = (x - mean) / std;
            -0.5 * z * z - std.ln() - LN_SQRT_2PI
        }
        Dist::Uniform { lo, hi } => {
            if *lo <= x && x < *hi {
                -(hi - lo).ln()
            } else {
                f64::NEG_INFINITY
            }
        }
        Dist::Laplace { mean, scale } => -(x - mean).abs() / scale - (2.0 * scale).ln(),
        Dist::StudentT { df, mean, scale } => {
            let z = (x - mean) / scale;
            special::ln_gamma(0.5 * (df + 1.0))
                - special::ln_gamma(0.5 * df)
                - 0.5 * (df * std::f64::consts::PI).ln()
                - scale.ln()
                - 0.5 * (df + 1.0) * (z * z / df).ln_1p()
        }
        Dist::LogNormal { mu, sigma, signed } => {
            let a = if *signed { x.abs() } else { x };
            if a <= 0.0 {
                return Ok(f64::NEG_INFINITY);
            }
            let z = (a.ln() - mu) / sigma;
            let half = if *signed { std::f64::consts::LN_2 } else { 0.0 };
            -0.5 * z * z - sigma.ln() - LN_SQRT_2PI - a.ln() - half
        }
        Dist::Histogram { edges, counts, .. } => {
            let tot: f64 = counts.iter().sum();
            match edges.partition_point(|e| *e <= x) {
                0 => f64::NEG_INFINITY,
                k if k >= edges.len() => f64::NEG_INFINITY,
                k => (counts[k - 1] / tot / (edges[k] - edges[k - 1])).ln(),
            }
        }
        Dist::Affine { a, b, dist } => {
            if *a == 0.0 {
                return Err("affine with a = 0 has no density".into());
            }
            ln_pdf(dist, (x - b) / a)? - a.abs().ln()
        }
        Dist::Mixture { components } => {
            let tot: f64 = components.iter().map(|c| c.weight).sum();
            let mut t = Vec::with_capacity(components.len());
            for c in components {
                if c.weight > 0.0 {
                    t.push((c.weight / tot).ln() + ln_pdf(&c.dist, x)?);
                }
            }
            let m = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if m == f64::NEG_INFINITY {
                m
            } else {
                m + t.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
            }
        }
        Dist::Discrete { .. }
        | Dist::Integers { .. }
        | Dist::Const { .. }
        | Dist::Empirical { .. }
        | Dist::Codes { .. } => {
            return Err(format!(
                "no closed-form density for a discrete law ({})",
                kind(d)
            ));
        }
    })
}

fn kind(d: &Dist) -> &'static str {
    match d {
        Dist::Normal { .. } => "normal",
        Dist::Uniform { .. } => "uniform",
        Dist::Laplace { .. } => "laplace",
        Dist::StudentT { .. } => "student_t",
        Dist::LogNormal { .. } => "log_normal",
        Dist::Mixture { .. } => "mixture",
        Dist::Discrete { .. } => "discrete",
        Dist::Integers { .. } => "integers",
        Dist::Const { .. } => "const",
        Dist::Empirical { .. } => "empirical",
        Dist::Histogram { .. } => "histogram",
        Dist::Codes { .. } => "codes",
        Dist::Affine { .. } => "affine",
    }
}

/// The spec law `d` with its location and scale replaced.
fn loc_scale_law(d: &Dist, ls: &LocScale) -> Result<Dist, String> {
    if !(ls.scale > 0.0 && ls.scale.is_finite() && ls.loc.is_finite()) {
        return Err(format!(
            "loc_scale: need a finite loc and scale > 0, got {ls:?}"
        ));
    }
    Ok(match d {
        Dist::Normal { .. } => Dist::Normal {
            mean: ls.loc,
            std: ls.scale,
        },
        Dist::Laplace { .. } => Dist::Laplace {
            mean: ls.loc,
            scale: ls.scale,
        },
        Dist::StudentT { df, .. } => Dist::StudentT {
            df: *df,
            mean: ls.loc,
            scale: ls.scale,
        },
        Dist::LogNormal { signed, .. } => Dist::LogNormal {
            mu: ls.loc,
            sigma: ls.scale,
            signed: *signed,
        },
        d => {
            return Err(format!(
                "loc_scale needs a normal, laplace, student_t or log_normal law, not {} (bounded laws: histogram)",
                kind(d)
            ));
        }
    })
}

/// A Histogram proposal on a bounded spec law, checked to cover every point where p > 0.
fn histogram_law(d: &Dist, edges: &[f64], probs: &[f64]) -> Result<Dist, String> {
    if !matches!(d, Dist::Uniform { .. } | Dist::Histogram { .. }) {
        return Err(format!(
            "histogram proposal on an unbounded or discrete law ({}): p > 0 outside its edges",
            kind(d)
        ));
    }
    if edges.len() != probs.len() + 1
        || probs.is_empty()
        || edges
            .windows(2)
            .any(|w| w[0].partial_cmp(&w[1]) != Some(std::cmp::Ordering::Less))
        || probs.iter().any(|p| p.is_nan() || *p < 0.0)
        || (probs.iter().sum::<f64>() - 1.0).abs() > 1e-9
    {
        return Err(
            "histogram proposal: increasing edges, probs.len() + 1 edges, probs >= 0 summing to 1"
                .into(),
        );
    }
    let (lo, hi) = (edges[0], edges[edges.len() - 1]);
    if d.cdf(lo, false) > 0.0 || d.cdf(hi, false) < 1.0 {
        return Err(format!(
            "histogram proposal: edges [{lo}, {hi}] do not cover the spec's support (q = 0 where p > 0)"
        ));
    }
    for (k, p) in probs.iter().enumerate() {
        if *p == 0.0 && d.mass(edges[k], true, edges[k + 1], false) > 0.0 {
            return Err(format!(
                "histogram proposal: bin [{}, {}) has probability 0 where the spec has mass",
                edges[k],
                edges[k + 1]
            ));
        }
    }
    Ok(Dist::Histogram {
        edges: edges.to_vec(),
        counts: probs.to_vec(),
        file: None,
    })
}

/// log w of the defensive mixture: -ln(alpha + (1 - alpha) e^lr), lr = ln q_tilt - ln p over
/// the tilted factors.
pub fn log_w(alpha: f64, lr: f64) -> f64 {
    if alpha >= 1.0 {
        return 0.0;
    }
    if lr.is_nan() {
        return f64::NAN;
    }
    if lr == f64::INFINITY {
        return f64::NEG_INFINITY;
    }
    if lr > 0.0 {
        -(lr + ((1.0 - alpha) + alpha * (-lr).exp()).ln())
    } else {
        -(alpha + (1.0 - alpha) * lr.exp()).ln()
    }
}

// ASSUME(N-ERRDIST-IS): w is the closed-form density ratio of the tilted draws.
/// Draw sample `idx` under the proposal: the sample (as `spec::sample_with` with `plan`) and
/// log w. Streams of the mixture choice and the tilted draws are disjoint from the spec's, so
/// alpha = 1 reproduces the plain sample bit for bit.
pub fn sample_is(
    spec: &Spec,
    prop: &Proposal,
    layouts: &[Layout],
    plan: &Plan,
    idx: u64,
) -> Result<(Sample, f64), String> {
    if !(prop.alpha > 0.0 && prop.alpha <= 1.0) {
        return Err("proposal: alpha must be in (0, 1]".into());
    }
    let tilt = prop.alpha < 1.0
        && spec::keyed(spec::stream_base(spec.seed, idx, "#is-mix"), 0).unit() >= prop.alpha;
    let scale_bufs: BTreeSet<String> = spec.scale_buffers().into_iter().map(|s| s.0).collect();
    let mut lr = 0.0f64;
    let mut over: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (name, tp) in &prop.tensors {
        let l = layouts
            .iter()
            .find(|l| &l.name == name)
            .ok_or_else(|| format!("proposal: {name} is not an input buffer"))?;
        if scale_bufs.contains(name) {
            return Err(format!(
                "proposal: {name} is a scale buffer (filled by its recipe, not tilted)"
            ));
        }
        let t = spec
            .tensor(name)
            .ok_or_else(|| format!("proposal: no distribution for {name}"))?;
        if matches!(t.error, Some(InputError::Pairs { .. })) {
            return Err(format!(
                "proposal: {name} draws its values from (value, error) pairs (not tilted)"
            ));
        }
        if tp.scales.len() > t.scales.len() {
            return Err(format!(
                "proposal: {name} has {} scale entries, the spec {}",
                tp.scales.len(),
                t.scales.len()
            ));
        }
        let shape = l.logical_shape(t);
        let strides: Vec<u64> = (0..shape.len())
            .map(|a| shape[a + 1..].iter().product())
            .collect();
        let coord = |i: u64, ax: usize| (i / strides[ax]) % shape[ax];
        let sparse = plan.elems.get(name);
        let m = sparse.map_or(l.elems(), |v| v.len());
        let elem = |j: usize| sparse.map_or(j as u64, |v| v[j]);
        let base = |tag: &str| spec::stream_base(spec.seed, idx, &format!("{name}{tag}"));
        let mut mult = vec![1.0f64; m];
        // Axis scales, in the spec's order (the product's order is the sampler's).
        for (k, s) in t.scales.iter().enumerate() {
            if s.axis >= shape.len() {
                continue;
            }
            let qd = tp
                .scales
                .get(k)
                .and_then(|x| x.as_ref())
                .map(|ls| loc_scale_law(&s.dist, ls))
                .transpose()
                .map_err(|e| format!("proposal {name} scale {k}: {e}"))?;
            let (pb, qb) = (base(&format!("#scale{k}")), base(&format!("#is-scale{k}")));
            let mut draws: BTreeMap<u64, f64> = BTreeMap::new();
            for (j, mj) in mult.iter_mut().enumerate() {
                let c = coord(elem(j), s.axis);
                let v = match draws.get(&c) {
                    Some(v) => *v,
                    None => {
                        let v = match (&qd, tilt) {
                            (Some(q), true) => q.sample(&mut spec::keyed(qb, c)),
                            _ => s.dist.sample(&mut spec::keyed(pb, c)),
                        };
                        if let Some(q) = &qd {
                            lr += ln_pdf(q, v)? - ln_pdf(&s.dist, v)?;
                        }
                        draws.insert(c, v);
                        v
                    }
                };
                *mj *= v;
            }
        }
        if let Some(o) = &t.outliers {
            if o.axis < shape.len() {
                let q = tp.outlier_prob;
                if let Some(q) = q {
                    if !(0.0..=1.0).contains(&q)
                        || (o.prob > 0.0 && q == 0.0)
                        || (o.prob < 1.0 && q == 1.0)
                    {
                        return Err(format!(
                            "proposal {name}: outlier_prob {q} gives q = 0 where p > 0 (spec {})",
                            o.prob
                        ));
                    }
                }
                let (pb, qb) = (base("#outliers"), base("#is-outliers"));
                let mut draws: BTreeMap<u64, bool> = BTreeMap::new();
                for (j, mj) in mult.iter_mut().enumerate() {
                    let c = coord(elem(j), o.axis);
                    let b = match draws.get(&c) {
                        Some(b) => *b,
                        None => {
                            let b = match (q, tilt) {
                                (Some(q), true) => spec::keyed(qb, c).unit() < q,
                                _ => spec::keyed(pb, c).unit() < o.prob,
                            };
                            if let Some(q) = q {
                                let (lq, lp) = if b {
                                    (q.ln(), o.prob.ln())
                                } else {
                                    ((1.0 - q).ln(), (1.0 - o.prob).ln())
                                };
                                lr += lq - lp;
                            }
                            draws.insert(c, b);
                            b
                        }
                    };
                    if b {
                        *mj *= o.factor;
                    }
                }
            }
        }
        // Element laws.
        let all: Option<Dist> = match &tp.elems {
            ElemProposal::Spec | ElemProposal::PerElem(_) => None,
            ElemProposal::LocScale(ls) => {
                Some(loc_scale_law(&t.dist, ls).map_err(|e| format!("proposal {name}: {e}"))?)
            }
            ElemProposal::Histogram { edges, probs } => Some(
                histogram_law(&t.dist, edges, probs)
                    .map_err(|e| format!("proposal {name}: {e}"))?,
            ),
        };
        let mut per: BTreeMap<u64, Dist> = BTreeMap::new();
        if let ElemProposal::PerElem(v) = &tp.elems {
            for (i, ls) in v {
                let drawn = match sparse {
                    Some(s) => s.binary_search(i).is_ok(),
                    None => (*i as usize) < m,
                };
                if !drawn {
                    return Err(format!(
                        "proposal {name}: per_elem names element {i}, which the program does not read"
                    ));
                }
                let q = loc_scale_law(&t.dist, ls).map_err(|e| format!("proposal {name}: {e}"))?;
                if per.insert(*i, q).is_some() {
                    return Err(format!("proposal {name}: per_elem names element {i} twice"));
                }
            }
        }
        let (pb, qb) = (base(""), base("#is"));
        let mut x = Vec::with_capacity(m);
        for (j, mj) in mult.iter().enumerate() {
            let i = elem(j);
            let qd = all.as_ref().or_else(|| per.get(&i));
            let v = match (qd, tilt) {
                (Some(q), true) => q.sample(&mut spec::keyed(qb, i)),
                _ => t.dist.sample(&mut spec::keyed(pb, i)),
            };
            if let Some(q) = qd {
                lr += ln_pdf(q, v)? - ln_pdf(&t.dist, v)?;
            }
            x.push(mj * v);
        }
        over.insert(name.clone(), x);
    }
    let smp = spec::sample_core(spec, layouts, idx, Some(plan), Some(&over))?;
    Ok((smp, log_w(prop.alpha, lr)))
}

// ---------------------------------------------------------------------------- estimators

/// One input's weighted contributions: log w and f per estimand (fractions of its outputs).
#[derive(Clone, Debug, Default)]
pub struct IsRec {
    /// The input sample index (records are summed in this order).
    pub t: u64,
    pub logw: f64,
    /// Fraction of compared outputs whose bits differ.
    pub differ: f64,
    /// Fraction of tracked outputs that are NaN / Inf.
    pub nonfinite: f64,
    /// Fraction of compared outputs per output-ulp multiple k.
    pub pmf: Vec<(i64, f64)>,
}

/// Weighted sums over n inputs, per component.
#[derive(Clone, Debug, Default)]
pub struct WObs {
    pub n: u64,
    pub sw: f64,
    pub sw2: f64,
    pub swf: Vec<f64>,
    pub sw2f2: Vec<f64>,
    /// max_i w_i f_i / sum_i w_i f_i (0 when the sum is 0).
    pub wmax_share: Vec<f64>,
}

impl WObs {
    /// From weights and per-input component values (f[i].len() == components).
    pub fn new(w: &[f64], f: &[Vec<f64>]) -> WObs {
        let k = f.first().map_or(0, |v| v.len());
        let mut o = WObs {
            n: w.len() as u64,
            swf: vec![0.0; k],
            sw2f2: vec![0.0; k],
            wmax_share: vec![0.0; k],
            ..Default::default()
        };
        let mut mx = vec![0.0f64; k];
        for (wi, fi) in w.iter().zip(f) {
            o.sw += wi;
            o.sw2 += wi * wi;
            for c in 0..k {
                let a = wi * fi[c];
                o.swf[c] += a;
                o.sw2f2[c] += a * a;
                mx[c] = mx[c].max(a);
            }
        }
        for ((share, m), s) in o.wmax_share.iter_mut().zip(&mx).zip(&o.swf) {
            if *s > 0.0 {
                *share = m / s;
            }
        }
        o
    }
    /// sum w f / n (unnormalized: p and q are normalized densities).
    pub fn est(&self, c: usize) -> f64 {
        self.swf[c] / self.n.max(1) as f64
    }
    /// Variance of `est`.
    pub fn var(&self, c: usize) -> f64 {
        let n = self.n as f64;
        if self.n < 2 {
            return f64::INFINITY;
        }
        let m = self.est(c);
        ((self.sw2f2[c] / n - m * m) / (n - 1.0)).max(0.0)
    }
    /// Effective sample size sum(w)^2 / sum(w^2).
    pub fn ess(&self) -> f64 {
        if self.sw2 > 0.0 {
            self.sw * self.sw / self.sw2
        } else {
            0.0
        }
    }
}

/// The plain estimate of a component, for the agreement gate.
#[derive(Clone, Copy, Debug)]
pub struct PlainRef {
    pub p: f64,
    /// Standard error.
    pub se: f64,
}

/// A gated IS estimate.
#[derive(Clone, Debug, Default)]
pub struct IsSummary {
    pub n: u64,
    pub ess: f64,
    /// (estimate, lo, hi) at 95% (normal on the estimator's variance).
    pub p_differ: (f64, f64, f64),
    pub p_nonfinite: (f64, f64, f64),
    /// Reported PMF bins (their wmax_share <= WMAX_SHARE) and the bins left out.
    pub pmf: Vec<(i64, f64)>,
    pub dropped_bins: Vec<i64>,
    /// wmax_share of P(differ).
    pub wmax_share: f64,
}

pub const WMAX_SHARE: f64 = 0.05;
const Z999: f64 = 3.290_526_731_491_926;

/// The weighted estimates and their gates (docs/errdist.md §5.4.1): Err("<reason>") when IS must
/// be refused. `plain`: P(differ) and P(non-finite) from plain sampling (None: not available).
pub fn summarize(recs: &[IsRec], plain: [Option<PlainRef>; 2]) -> Result<IsSummary, String> {
    let n = recs.len();
    if n < 2 {
        return Err("fewer than 2 inputs".into());
    }
    let mut recs: Vec<&IsRec> = recs.iter().collect();
    recs.sort_by_key(|r| r.t);
    if let Some(r) = recs
        .iter()
        .find(|r| r.logw.is_nan() || r.logw == f64::INFINITY)
    {
        return Err(format!("log w = {} at input {}", r.logw, r.t));
    }
    let keys: Vec<i64> = recs
        .iter()
        .flat_map(|r| r.pmf.iter().map(|x| x.0))
        .collect::<BTreeSet<i64>>()
        .into_iter()
        .collect();
    let w: Vec<f64> = recs.iter().map(|r| r.logw.exp()).collect();
    let f: Vec<Vec<f64>> = recs
        .iter()
        .map(|r| {
            let mut v = vec![r.differ, r.nonfinite];
            v.extend(
                keys.iter()
                    .map(|k| r.pmf.iter().find(|x| x.0 == *k).map_or(0.0, |x| x.1)),
            );
            v
        })
        .collect();
    let wo = WObs::new(&w, &f);
    let ess = wo.ess();
    let need = 200f64.max(n as f64 / 50.0);
    if ess.is_nan() || ess < need {
        return Err(format!("ESS {ess:.1} < {need:.0} (n = {n})"));
    }
    if wo.swf[0] <= 0.0 {
        return Err("no weighted differing output (IS saw no event)".into());
    }
    for (c, what) in [(0, "P(differ)"), (1, "P(non-finite)")] {
        if wo.wmax_share[c] > WMAX_SHARE {
            return Err(format!(
                "one input holds {:.3} of the weighted {what} (max share {WMAX_SHARE})",
                wo.wmax_share[c]
            ));
        }
    }
    for (c, what) in [(0, "P(differ)"), (1, "P(non-finite)")] {
        let Some(pr) = plain[c] else { continue };
        // Plain resolves it: 95% half-width <= 50% of the estimate.
        if pr.p > 0.0 && 1.96 * pr.se <= 0.5 * pr.p {
            let (e, v) = (wo.est(c), wo.var(c));
            let tol = Z999 * (v + pr.se * pr.se).sqrt();
            if (e - pr.p).abs() > tol {
                return Err(format!(
                    "IS {what} {e:.4e} and plain {:.4e} disagree beyond the combined 99.9% interval (+-{tol:.2e})",
                    pr.p
                ));
            }
        }
    }
    let ci = |c: usize| {
        let (e, hw) = (wo.est(c), 1.96 * wo.var(c).sqrt());
        (e, (e - hw).max(0.0), (e + hw).min(1.0))
    };
    let mut pmf = Vec::new();
    let mut dropped = Vec::new();
    for (j, k) in keys.iter().enumerate() {
        if wo.wmax_share[2 + j] <= WMAX_SHARE {
            pmf.push((*k, wo.est(2 + j)));
        } else {
            dropped.push(*k);
        }
    }
    Ok(IsSummary {
        n: n as u64,
        ess,
        p_differ: ci(0),
        p_nonfinite: ci(1),
        pmf,
        dropped_bins: dropped,
        wmax_share: wo.wmax_share[0],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec1(dist: serde_json::Value, n: u64) -> (Spec, Vec<Layout>, Plan) {
        let s = Spec::from_value(
            serde_json::json!({"version": 1, "seed": 7, "tensors": {"x": {"dist": dist}}}),
            std::path::Path::new("."),
        )
        .unwrap();
        let l = vec![Layout {
            name: "x".into(),
            dtype: "f32".into(),
            shape: vec![n],
            bytes: 4 * n as usize,
            fmt: spec::Format::F32,
            blocks: None,
        }];
        (s, l, Plan::default())
    }

    fn vals(s: &Sample) -> Vec<f64> {
        s.bytes["x"]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect()
    }

    /// alpha = 1 draws the spec's own sample, bit for bit, with log w = 0.
    #[test]
    fn is_alpha_one_sample_is_the_plain_sample() {
        let (s, l, plan) = spec1(serde_json::json!({"kind": "normal"}), 16);
        let p = Proposal {
            name: "t".into(),
            alpha: 1.0,
            description: String::new(),
            tensors: [(
                "x".to_string(),
                TensorProposal {
                    elems: ElemProposal::LocScale(LocScale {
                        loc: 3.0,
                        scale: 0.5,
                    }),
                    scales: vec![],
                    outlier_prob: None,
                },
            )]
            .into_iter()
            .collect(),
        };
        for idx in 0..50 {
            let (a, lw) = sample_is(&s, &p, &l, &plan, idx).unwrap();
            let b = spec::sample_with(&s, &l, idx, Some(&plan)).unwrap();
            assert_eq!(a.bytes, b.bytes);
            assert_eq!(lw, 0.0);
        }
    }

    /// log w against an independent computation from the drawn values: the defensive term and
    /// the per-element densities at the right elements (kills the "p / q_tilt" and "PerElem
    /// off by one" mutants deterministically).
    #[test]
    fn is_logw_is_the_closed_form_ratio() {
        let (s, l, plan) = spec1(serde_json::json!({"kind": "normal", "std": 2.0}), 8);
        let tilts = [(1u64, 1.5, 1.0), (2, -2.0, 3.0), (5, 0.7, 0.4)];
        let alpha = 0.3;
        let p = Proposal {
            name: "t".into(),
            alpha,
            description: String::new(),
            tensors: [(
                "x".to_string(),
                TensorProposal {
                    elems: ElemProposal::PerElem(
                        tilts
                            .iter()
                            .map(|&(i, loc, scale)| (i, LocScale { loc, scale }))
                            .collect(),
                    ),
                    scales: vec![],
                    outlier_prob: None,
                },
            )]
            .into_iter()
            .collect(),
        };
        let lnn = |x: f64, m: f64, sd: f64| {
            -0.5 * ((x - m) / sd).powi(2) - sd.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
        };
        let mut tilted = 0;
        for idx in 0..40 {
            let (smp, lw) = sample_is(&s, &p, &l, &plan, idx).unwrap();
            let x = vals(&smp);
            // The weight is on the real draws; f32 rounding of the stored value moves the
            // density by < 1e-6 relative.
            let mut lp = 0.0;
            let mut lq = 0.0;
            for &(i, loc, scale) in &tilts {
                lp += lnn(x[i as usize], 0.0, 2.0);
                lq += lnn(x[i as usize], loc, scale);
            }
            let want = lp - (alpha * lp.exp() + (1.0 - alpha) * lq.exp()).ln();
            assert!(
                (lw - want).abs() < 1e-4 * (1.0 + want.abs()),
                "idx {idx}: log w {lw} vs {want}"
            );
            tilted += (lw < -(alpha.ln()) - 1e-9) as u32;
        }
        assert!(tilted > 0);
    }

    /// The estimate is sum w f / n, not the self-normalized sum w f / sum w.
    #[test]
    fn is_estimate_is_unnormalized() {
        let recs: Vec<IsRec> = (0..400)
            .map(|t| IsRec {
                t,
                logw: if t % 2 == 0 {
                    0.5f64.ln()
                } else {
                    0.25f64.ln()
                },
                differ: if t % 4 == 0 { 1.0 } else { 0.0 },
                ..Default::default()
            })
            .collect();
        let w: Vec<f64> = recs.iter().map(|r| r.logw.exp()).collect();
        let f: Vec<Vec<f64>> = recs.iter().map(|r| vec![r.differ]).collect();
        let wo = WObs::new(&w, &f);
        // 100 events of weight 0.5 over 400 inputs.
        assert!((wo.est(0) - 0.125).abs() < 1e-15, "{}", wo.est(0));
    }

    #[test]
    fn is_refuses_discrete_laws_and_uncovered_histograms() {
        let (s, l, plan) = spec1(serde_json::json!({"kind": "integers", "lo": 0, "hi": 3}), 4);
        let mk = |e: ElemProposal| Proposal {
            name: "t".into(),
            alpha: 0.5,
            description: String::new(),
            tensors: [(
                "x".to_string(),
                TensorProposal {
                    elems: e,
                    scales: vec![],
                    outlier_prob: None,
                },
            )]
            .into_iter()
            .collect(),
        };
        let p = mk(ElemProposal::LocScale(LocScale {
            loc: 0.0,
            scale: 1.0,
        }));
        assert!(sample_is(&s, &p, &l, &plan, 0).is_err());
        let (s, l, plan) = spec1(
            serde_json::json!({"kind": "uniform", "lo": 0.0, "hi": 1.0}),
            4,
        );
        let p = mk(ElemProposal::Histogram {
            edges: vec![0.0, 0.5, 0.9],
            probs: vec![0.5, 0.5],
        });
        let e = sample_is(&s, &p, &l, &plan, 0).err().unwrap();
        assert!(e.contains("cover"), "{e}");
        let p = mk(ElemProposal::Histogram {
            edges: vec![0.0, 0.5, 1.0],
            probs: vec![0.0, 1.0],
        });
        assert!(sample_is(&s, &p, &l, &plan, 0).is_err());
        let p = mk(ElemProposal::PerElem(vec![(
            4,
            LocScale {
                loc: 0.0,
                scale: 1.0,
            },
        )]));
        let (s, l, plan) = spec1(serde_json::json!({"kind": "normal"}), 4);
        let e = sample_is(&s, &p, &l, &plan, 0).err().unwrap();
        assert!(e.contains("does not read"), "{e}");
    }

    /// Densities integrate to 1 (trapezoid) and match the samplers' laws.
    #[test]
    fn is_densities_are_normalized() {
        let laws = [
            serde_json::json!({"kind": "normal", "mean": 1.0, "std": 0.5}),
            serde_json::json!({"kind": "laplace", "mean": -1.0, "scale": 2.0}),
            serde_json::json!({"kind": "student_t", "df": 3.0, "scale": 1.5}),
            serde_json::json!({"kind": "log_normal", "mu": 0.0, "sigma": 0.5, "signed": true}),
            serde_json::json!({"kind": "uniform", "lo": -1.0, "hi": 2.0}),
            serde_json::json!({"kind": "affine", "a": -2.0, "b": 1.0, "dist": {"kind": "normal"}}),
        ];
        for j in laws {
            let d: Dist = serde_json::from_value(j.clone()).unwrap();
            let (a, b, k) = (-60.0, 60.0, 600_000);
            let h = (b - a) / k as f64;
            let s: f64 = (0..=k)
                .map(|i| ln_pdf(&d, a + i as f64 * h).unwrap().exp() * h)
                .sum();
            assert!((s - 1.0).abs() < 2e-3, "{j}: {s}");
        }
    }
}

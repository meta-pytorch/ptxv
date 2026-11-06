/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The errdist report, version 1 (docs/errdist.md §3): the JSON `ptxv errdist --json` writes.
//! Field names are stable within a version; new optional fields may be added.

use serde::Serialize;

pub const REPORT_VERSION: u32 = 1;

/// Summary of one error distribution in one unit.
#[derive(Clone, Debug, Serialize, Default)]
pub struct Moments {
    pub mean: f64,
    pub var: f64,
    pub std: f64,
    pub mean_abs: f64,
    /// P(error == 0) (exactly zero, not rounded).
    pub p_zero: f64,
    /// (p, x): the p-quantile.
    pub quantiles: Vec<(f64, f64)>,
    /// Largest |error| seen (Monte Carlo) or in the support (exact tiers).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_abs: Option<f64>,
}

/// Error bars of one estimate.
#[derive(Clone, Debug, Serialize, Default)]
pub struct ErrorBar {
    /// "exact", "dkw" (Monte Carlo CDF band), "tv" (total-variation bound of the residue
    /// model), "berry_esseen", "clt" (moments only), "none".
    pub kind: String,
    /// Sup over x of |predicted CDF - true CDF| (within the stated model), if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cdf_sup: Option<f64>,
    /// 95% half-widths of the mean and of the std (Monte Carlo), in the `abs` unit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_hw: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub std_rel_hw: Option<f64>,
    /// Probability mass whose error the model does not describe (added to every CDF bound).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_mass: Option<f64>,
    pub notes: Vec<String>,
}

/// One tier's estimate of the distribution (pooled over the outputs it covers).
#[derive(Clone, Debug, Serialize, Default)]
pub struct Estimate {
    pub tier: u8,
    pub method: String,
    /// Inputs (Tier 3) or coarse states (Tier 1) behind it; 0 for closed forms.
    pub n: u64,
    /// Output elements per input it covers.
    pub outputs: u64,
    /// Absolute error.
    pub abs: Moments,
    /// Error / (u * S(x)), u = 2^-24, S(x) = sum over the output's terms of |term| (the scale
    /// of the worst-case bound tau = k u S).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub u_scale: Option<Moments>,
    /// Error / ulp_out(y): ulps of the output format at the exact value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ulp: Option<Moments>,
    /// PMF over integer multiples of the output ulp (16-bit and narrower outputs), (k, p).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub pmf_ulp: Vec<(i64, f64)>,
    /// Histogram of error / (u S): (edges, probabilities).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hist_u: Option<(Vec<f64>, Vec<f64>)>,
    pub error_bar: ErrorBar,
    /// Pair mode: covariance of the two kernels' errors against the exact value, and each
    /// one's mean / variance (abs unit).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pair: Option<PairMoments>,
    /// Probability that an output's bits differ from the reference (pair) or from the
    /// correctly rounded exact value (single), with a 95% interval.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_differ: Option<(f64, f64, f64)>,
    pub seconds: f64,
    /// Whether the estimate passed its own applicability checks (analytic tiers); `best`
    /// only picks applicable estimates. None: no checks (tiers 0 and 3).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applicable: Option<bool>,
    /// The applicability checks and their outcomes.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub applicability: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Default)]
pub struct PairMoments {
    pub mean_r: f64,
    pub var_r: f64,
    pub mean_c: f64,
    pub var_c: f64,
    pub cov_rc: f64,
    pub corr_rc: f64,
}

/// Mass outside the modeled domain.
#[derive(Clone, Debug, Serialize, Default)]
pub struct OutOfDomain {
    /// Probability that an input sample has any out-of-domain event.
    pub mass: f64,
    /// Per cause: probability per input sample (or per element where stated).
    pub by_cause: std::collections::BTreeMap<String, f64>,
    pub method: String,
}

/// Predicted against observed on samples of the same spec.
#[derive(Clone, Debug, Serialize, Default)]
pub struct Validation {
    /// "sim" (bit-exact simulator) or "gpu" (GB300).
    pub source: String,
    /// The estimate compared against (tier, method).
    pub against: String,
    pub inputs: u64,
    pub outputs: u64,
    pub mean_pred: f64,
    pub mean_obs: f64,
    pub std_pred: f64,
    pub std_obs: f64,
    pub p_differ_pred: Option<f64>,
    pub p_differ_obs: Option<(f64, f64, f64)>,
    /// Two-sample Kolmogorov-Smirnov statistic between predicted and observed errors (u
    /// scale), with its asymptotic p-value (effective sample sizes).
    pub ks_d: f64,
    pub ks_p: f64,
    /// Anderson-Darling k-sample statistic (two samples) and its approximate p-value.
    pub ad: f64,
    pub ad_p: f64,
    /// QQ points: (p, predicted quantile, observed quantile), u scale.
    pub qq: Vec<(f64, f64, f64)>,
    /// Observed PMF on the output-ulp axis (as `Estimate::pmf_ulp`), and the compared
    /// estimate's, for overlays; their total-variation distance when both exist.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub pmf_ulp: Vec<(i64, f64)>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub pmf_ulp_pred: Vec<(i64, f64)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pmf_tv: Option<f64>,
    /// Outputs whose simulator bits differ from the DAG evaluation (must be 0).
    pub dag_mismatch: u64,
    pub agree: bool,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, serde::Deserialize, Default)]
pub struct KernelId {
    pub case: String,
    pub variant: String,
    pub front: String,
}

#[derive(Clone, Debug, Serialize, Default)]
pub struct Report {
    pub version: u32,
    pub kind: String,
    /// "single" (C - exact) or "pair" (C - R).
    pub mode: String,
    pub quantity: String,
    pub reference: KernelId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate: Option<KernelId>,
    pub spec: serde_json::Value,
    /// Output elements per input, their format, and the number of chain templates.
    pub outputs: u64,
    pub output_format: String,
    pub templates: u64,
    /// The tier whose estimate is the headline (most accurate that ran).
    pub best: usize,
    pub estimates: Vec<Estimate>,
    pub out_of_domain: OutOfDomain,
    pub validation: Vec<Validation>,
    /// Residue diagnostics: per rounding site, the largest |Fourier coefficient| at the ulp
    /// frequency and the uniform model's TV bound (docs/errdist.md §4).
    pub residue: Vec<ResidueSite>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub per_output: Vec<serde_json::Value>,
    pub assumptions: Vec<String>,
    pub notes: Vec<String>,
    /// The proven worst case beside the distribution (pairs: the equivalence proof's bound;
    /// single kernels: the kernel's own bound against the exact value).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worst_case: Option<WorstCase>,
    /// Each analytic estimate checked against this run's Tier 3 (docs/errdist.md §3.2).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub crosscheck: Vec<CrossCheck>,
    /// The output's errstate file (--errstate), the next kernel's input (docs/errdist.md §10).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub errstate: Option<String>,
}

/// An analytic estimate against Tier 3: agreement means |p - p3| <= max(5% of p3, half Tier
/// 3's interval) or p inside Tier 3's interval; a disagreeing estimate is not applicable.
#[derive(Clone, Debug, Serialize, serde::Deserialize, Default)]
pub struct CrossCheck {
    /// Index into `estimates`.
    pub estimate: usize,
    pub tier: u8,
    pub p: f64,
    pub p3: f64,
    /// Tier 3's interval and the tolerance used.
    pub p3_lo: f64,
    pub p3_hi: f64,
    pub tol: f64,
    /// Tier 3 inputs behind p3.
    pub inputs: u64,
    pub agrees: bool,
}

/// The rigorous per-output bound |e| <= rel |y| + k_u u S(x) + abs (u = 2^-24, y the exact
/// output, S(x) = sum of |terms|: the scale of `equiv`'s bound), its distribution under the
/// input spec, and the check that no observed difference exceeds it.
#[derive(Clone, Debug, Serialize, serde::Deserialize, Default)]
pub struct WorstCase {
    /// "pair" (|C - R|, the equivalence proof of `ptxv equiv`) or "single" (|C - y|, the
    /// kernel's own bound).
    pub kind: String,
    pub available: bool,
    /// Holds for every input without extra preconditions (else see `preconditions`).
    pub proven: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
    /// The strongest equivalence level every output reaches (pairs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<String>,
    pub outputs: usize,
    pub bounded: usize,
    /// "sum" (k u S + abs), "round" (16-bit outputs: rel |y| + k u S + abs), or another
    /// `equiv` form (softmax, norm, attn: formula only).
    pub form: String,
    /// Largest coefficients over the outputs. `scale_one`: the k_u term is k_u u (softmax:
    /// |C - R| <= rel y + k_u u + abs with y <= 1), else k_u u S(x).
    #[serde(default)]
    pub scale_one: bool,
    pub rel: f64,
    pub k_u: f64,
    pub abs: f64,
    pub formula: String,
    pub human: String,
    pub preconditions: Vec<String>,
    /// The bound in output ulps of y over the Tier 3 inputs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dist: Option<WorstDist>,
    /// Observed differences against the bound, per source (Tier 3 DAG evaluation,
    /// simulator, GPU).
    pub checks: Vec<WorstCheck>,
}

#[derive(Clone, Debug, Serialize, serde::Deserialize, Default)]
pub struct WorstDist {
    pub n: u64,
    pub max_ulp: f64,
    pub mean_ulp: f64,
    pub quantiles_ulp: Vec<(f64, f64)>,
    /// Outputs left out (exact value 0 or subnormal: the bound in ulps is unbounded).
    pub tiny: u64,
}

#[derive(Clone, Debug, Serialize, serde::Deserialize, Default)]
pub struct WorstCheck {
    pub source: String,
    pub checked: u64,
    /// Outputs whose observed |difference| exceeds the bound (must be 0).
    pub exceed: u64,
    /// Largest observed |difference| / bound.
    pub max_ratio: f64,
}

#[derive(Clone, Debug, Serialize, Default)]
pub struct ResidueSite {
    pub site: String,
    /// max over nonzero frequencies k of |phi(k)|, phi the residue's characteristic function.
    pub max_fourier: f64,
    /// TV distance bound between the residue law and uniform: (1/2) sum_k |phi(k)|.
    pub tv_uniform: f64,
    /// Whether the engine used the uniform model here (else the exact residue law).
    pub uniform_used: bool,
    pub weight: f64,
}

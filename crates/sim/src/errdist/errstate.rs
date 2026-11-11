/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A kernel's output as the next kernel's input (docs/errdist.md §10): per output tensor,
//! the value and error moments, the value-error covariance, the moment form of the error
//! given the value, and joint (value, error) samples from Tier 3.
//!
//! The error state is per element, independent across elements (`model: "iid"`,
//! N-ERRDIST-INERR). The file's `model` field names the form so richer states (a factor
//! model with latent terms shared by elements that read a common input, low-rank plus
//! diagonal covariance) can be added without changing the interface: a reader that does
//! not know a model refuses it.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde::Serialize;

pub const ERRSTATE_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ErrState {
    pub kind: String,
    pub version: u32,
    /// "iid": elements independent, each with the same joint law of (value, error).
    pub model: String,
    /// What produced it (kernel ids, input spec name).
    pub source: serde_json::Value,
    pub tensors: BTreeMap<String, TensorState>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct TensorState {
    pub shape: Vec<u64>,
    pub format: String,
    /// Moments over the sampled elements: value (the exact output), error (computed - exact),
    /// their covariance.
    pub value_mean: f64,
    pub value_var: f64,
    pub error_mean: f64,
    pub error_var: f64,
    pub cov: f64,
    /// e | value ~ N(alpha + beta value, s0^2 + s1^2 value^2): [alpha, beta, s0, s1].
    pub linear: [f64; 4],
    /// Elements behind the moments.
    pub n: u64,
    /// Joint (value, error) samples (a subsample of at most `MAX_PAIRS`).
    pub pairs: Vec<[f64; 2]>,
    /// What the iid model leaves out, measured on the same Tier 3 inputs: correlation
    /// coefficients between two elements of one input in the same row (last axis) and in
    /// different rows, of their errors, of their values, and of one's value with the other's
    /// error.
    #[serde(default)]
    pub corr: Option<CrossCorr>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CrossCorr {
    pub error_same_row: f64,
    pub error_other_row: f64,
    pub value_same_row: f64,
    pub value_other_row: f64,
    pub value_error_same_row: f64,
    /// Of the squared (standardized) values and errors: shared magnitudes (a row's scale)
    /// that zero linear correlation does not show.
    pub value_sq_same_row: f64,
    pub error_sq_same_row: f64,
    /// Element pairs behind the same-row / other-row coefficients.
    pub n_same_row: u64,
    pub n_other_row: u64,
}

/// Var(z^2) of standardized samples (normalizes the squared-value correlation).
fn k4(z: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut n) = (0.0f64, 0.0f64);
    for x in z {
        s += (x * x - 1.0).powi(2);
        n += 1.0;
    }
    (s / n.max(1.0)).max(1e-300)
}

/// Cross-element correlations of one input's sampled elements (`at`: element index, input).
pub fn cross_corr(pairs: &[(f64, f64)], at: &[(u32, u64)], row: u64) -> CrossCorr {
    let n = pairs.len().max(1) as f64;
    let mv = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let me = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let sv = (pairs.iter().map(|p| (p.0 - mv).powi(2)).sum::<f64>() / n)
        .sqrt()
        .max(1e-300);
    let se = (pairs.iter().map(|p| (p.1 - me).powi(2)).sum::<f64>() / n)
        .sqrt()
        .max(1e-300);
    let mut by: BTreeMap<u64, Vec<(u64, f64, f64)>> = BTreeMap::new();
    for (p, a) in pairs.iter().zip(at) {
        by.entry(a.1).or_default().push((
            a.0 as u64 / row.max(1),
            (p.0 - mv) / sv,
            (p.1 - me) / se,
        ));
    }
    // [same row, other row] x (ee, vv, ve, count, v2v2, e2e2)
    let mut acc = [[0.0f64; 6]; 2];
    for v in by.values() {
        for i in 0..v.len() {
            for j in i + 1..v.len() {
                let k = (v[i].0 != v[j].0) as usize;
                acc[k][0] += v[i].2 * v[j].2;
                acc[k][1] += v[i].1 * v[j].1;
                acc[k][2] += 0.5 * (v[i].1 * v[j].2 + v[j].1 * v[i].2);
                acc[k][3] += 1.0;
                acc[k][4] += (v[i].1 * v[i].1 - 1.0) * (v[j].1 * v[j].1 - 1.0);
                acc[k][5] += (v[i].2 * v[i].2 - 1.0) * (v[j].2 * v[j].2 - 1.0);
            }
        }
    }
    let c = |k: usize, m: usize| acc[k][m] / acc[k][3].max(1.0);
    CrossCorr {
        error_same_row: c(0, 0),
        error_other_row: c(1, 0),
        value_same_row: c(0, 1),
        value_other_row: c(1, 1),
        value_error_same_row: c(0, 2),
        value_sq_same_row: c(0, 4) / k4(pairs.iter().map(|p| (p.0 - mv) / sv)),
        error_sq_same_row: c(0, 5) / k4(pairs.iter().map(|p| (p.1 - me) / se)),
        n_same_row: acc[0][3] as u64,
        n_other_row: acc[1][3] as u64,
    }
}

pub const MAX_PAIRS: usize = 32768;

/// A tensor's state from Tier 3's (value, error) pairs.
pub fn tensor_state(
    pairs: &[(f64, f64)],
    at: &[(u32, u64)],
    shape: Vec<u64>,
    format: &str,
) -> TensorState {
    let row = *shape.last().unwrap_or(&1);
    // In (input sample, output) order, so the file does not depend on how the worker
    // threads' observations were merged.
    let mut ord: Vec<usize> = (0..pairs.len()).collect();
    if at.len() == pairs.len() {
        ord.sort_by_key(|&i| (at[i].1, at[i].0));
    }
    let pairs: Vec<(f64, f64)> = ord.iter().map(|&i| pairs[i]).collect();
    let at: Vec<(u32, u64)> = ord.iter().filter_map(|&i| at.get(i).copied()).collect();
    let n = pairs.len().max(1) as f64;
    let mv = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let me = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let vv = pairs.iter().map(|p| (p.0 - mv).powi(2)).sum::<f64>() / n;
    let ve = pairs.iter().map(|p| (p.1 - me).powi(2)).sum::<f64>() / n;
    let cv = pairs.iter().map(|p| (p.0 - mv) * (p.1 - me)).sum::<f64>() / n;
    let all: Vec<[f64; 2]> = pairs.iter().map(|p| [p.0, p.1]).collect();
    let step = all.len().div_ceil(MAX_PAIRS).max(1);
    TensorState {
        shape,
        format: format.into(),
        value_mean: mv,
        value_var: vv,
        error_mean: me,
        error_var: ve,
        cov: cv,
        linear: super::spec::fit_linear(&all),
        n: pairs.len() as u64,
        pairs: all.into_iter().step_by(step).collect(),
        corr: (at.len() == pairs.len()).then(|| cross_corr(&pairs, &at, row)),
    }
}

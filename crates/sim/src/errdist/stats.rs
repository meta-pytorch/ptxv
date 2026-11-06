/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Sample statistics: quantiles, two-sample Kolmogorov-Smirnov and Anderson-Darling, DKW.

use super::report::Moments;

pub const QUANTILES: [f64; 13] = [
    1e-4, 1e-3, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999,
];

/// Streaming moments (Welford is unnecessary at these sizes; sums in f64 with pairwise
/// merging per thread).
#[derive(Clone, Debug, Default)]
pub struct Acc {
    pub n: u64,
    pub s1: f64,
    pub s2: f64,
    pub sa: f64,
    pub zeros: u64,
    pub max_abs: f64,
}

impl Acc {
    #[inline]
    pub fn add(&mut self, x: f64) {
        self.n += 1;
        self.s1 += x;
        self.s2 += x * x;
        self.sa += x.abs();
        self.zeros += (x == 0.0) as u64;
        self.max_abs = self.max_abs.max(x.abs());
    }
    pub fn merge(&mut self, o: &Acc) {
        self.n += o.n;
        self.s1 += o.s1;
        self.s2 += o.s2;
        self.sa += o.sa;
        self.zeros += o.zeros;
        self.max_abs = self.max_abs.max(o.max_abs);
    }
    pub fn mean(&self) -> f64 {
        self.s1 / self.n.max(1) as f64
    }
    pub fn var(&self) -> f64 {
        (self.s2 / self.n.max(1) as f64 - self.mean().powi(2)).max(0.0)
    }
    /// Moments, with quantiles from `sorted` (a sample of the same distribution).
    pub fn moments(&self, sorted: &[f64]) -> Moments {
        Moments {
            mean: self.mean(),
            var: self.var(),
            std: self.var().sqrt(),
            mean_abs: self.sa / self.n.max(1) as f64,
            p_zero: self.zeros as f64 / self.n.max(1) as f64,
            quantiles: QUANTILES
                .iter()
                .map(|&p| (p, quantile(sorted, p)))
                .collect(),
            max_abs: Some(self.max_abs),
        }
    }
}

/// The p-quantile of a sorted sample (type 7, linear interpolation).
pub fn quantile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let h = (sorted.len() - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

pub fn sort(v: &mut [f64]) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// Two-sample KS statistic of sorted samples, and its asymptotic p-value with effective sizes
/// (n_eff, m_eff) (dependent samples: the number of independent units behind each).
pub fn ks2(a: &[f64], b: &[f64], n_eff: f64, m_eff: f64) -> (f64, f64) {
    // A NaN never compares equal: the merge below would not advance.
    if a.iter().chain(b).any(|x| x.is_nan()) {
        return (f64::NAN, f64::NAN);
    }
    if a.is_empty() || b.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let (mut i, mut j, mut d) = (0usize, 0usize, 0f64);
    while i < a.len() && j < b.len() {
        let x = a[i].min(b[j]);
        while i < a.len() && a[i] <= x {
            i += 1;
        }
        while j < b.len() && b[j] <= x {
            j += 1;
        }
        d = d.max((i as f64 / a.len() as f64 - j as f64 / b.len() as f64).abs());
    }
    let en = (n_eff * m_eff / (n_eff + m_eff)).sqrt();
    (d, super::special::kolmogorov_q((en + 0.12 + 0.11 / en) * d))
}

/// Anderson-Darling two-sample statistic (Scholz & Stephens 1987, A2_akN with ties) of sorted
/// samples, standardized, and an approximate p-value (their interpolation for k = 2). With
/// dependent samples the p-value is optimistic; `ks2`'s effective sizes are the safer read.
pub fn ad2(a: &[f64], b: &[f64]) -> (f64, f64) {
    // A NaN never compares equal: the merge below would not advance.
    if a.iter().chain(b).any(|x| x.is_nan()) {
        return (f64::NAN, f64::NAN);
    }
    let (n1, n2) = (a.len() as f64, b.len() as f64);
    if a.is_empty() || b.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let n = n1 + n2;
    if a[0] == a[a.len() - 1] && b[0] == b[b.len() - 1] && a[0] == b[0] {
        return (0.0, 1.0);
    }
    // Distinct pooled values with counts per sample.
    let (mut i, mut j) = (0usize, 0usize);
    let (mut m1, mut m2, mut b_) = (0f64, 0f64, 0f64); // counts strictly below
    let (mut s1, mut s2) = (0f64, 0f64);
    while i < a.len() || j < b.len() {
        let x = match (a.get(i), b.get(j)) {
            (Some(&p), Some(&q)) => p.min(q),
            (Some(&p), None) => p,
            (None, Some(&q)) => q,
            _ => break,
        };
        let (mut f1, mut f2) = (0f64, 0f64);
        while i < a.len() && a[i] == x {
            i += 1;
            f1 += 1.0;
        }
        while j < b.len() && b[j] == x {
            j += 1;
            f2 += 1.0;
        }
        let l = f1 + f2;
        if l > 0.5 * n {
            // One value holds most of the pooled sample: the tie-corrected statistic is
            // numerically meaningless; KS and P(differ) carry the comparison.
            return (f64::NAN, f64::NAN);
        }
        let bj = b_ + l / 2.0;
        let mij1 = m1 + f1 / 2.0;
        let mij2 = m2 + f2 / 2.0;
        let den = bj * (n - bj) - n * l / 4.0;
        if den > 1e-12 * n * n {
            s1 += l / n * (n * mij1 - n1 * bj).powi(2) / den;
            s2 += l / n * (n * mij2 - n2 * bj).powi(2) / den;
        }
        m1 += f1;
        m2 += f2;
        b_ += l;
    }
    let a2 = (n - 1.0) / n * (s1 / n1 + s2 / n2);
    // Standardize (k = 2): mean k - 1 = 1, variance from Scholz-Stephens.
    let h: f64 = (1..n as usize).map(|i| 1.0 / i as f64).sum();
    let hh = 1.0 / n1 + 1.0 / n2;
    let g: f64 = {
        let mut g = 0.0;
        let nn = n as usize;
        let mut inner = 0.0;
        for i in (1..nn.saturating_sub(1)).rev() {
            inner += 1.0 / (i + 1) as f64;
            g += inner / (nn - i) as f64;
        }
        g
    };
    let k = 2.0;
    let aa = (4.0 * g - 6.0) * (k - 1.0) + (10.0 - 6.0 * g) * hh;
    let bb = (2.0 * g - 4.0) * k * k + 8.0 * h * k + (2.0 * g - 14.0 * h - 4.0) * hh - 8.0 * h
        + 4.0 * g
        - 6.0;
    let cc = (6.0 * h + 2.0 * g - 2.0) * k * k
        + (4.0 * h - 4.0 * g + 6.0) * k
        + (2.0 * h - 6.0) * hh
        + 4.0 * h;
    let dd = (2.0 * h + 6.0) * k * k - 4.0 * h * k;
    let var = (aa * n.powi(3) + bb * n * n + cc * n + dd) / ((n - 1.0) * (n - 2.0) * (n - 3.0));
    let t = (a2 - (k - 1.0)) / var.max(1e-300).sqrt();
    // p-value: log-linear interpolation of Scholz-Stephens Table (m = 1).
    let tab: [(f64, f64); 5] = [
        (0.326, 0.25),
        (1.225, 0.10),
        (1.960, 0.05),
        (2.719, 0.025),
        (3.752, 0.01),
    ];
    let p = if t <= tab[0].0 {
        (0.25 + (tab[0].0 - t) * 0.25).min(1.0)
    } else if t >= tab[4].0 {
        0.01 * (-(t - tab[4].0)).exp()
    } else {
        let w = tab.windows(2).find(|w| t <= w[1].0).unwrap();
        let f = (t - w[0].0) / (w[1].0 - w[0].0);
        (w[0].1.ln() + f * (w[1].1.ln() - w[0].1.ln())).exp()
    };
    (t, p)
}

/// DKW: sup |F_n - F| <= eps with probability 1 - alpha.
pub fn dkw(n: f64, alpha: f64) -> f64 {
    ((2.0 / alpha).ln() / (2.0 * n)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errdist::special::Rng;

    #[test]
    fn two_sample_tests_separate_and_accept() {
        let mut r = Rng::new(3);
        let mut a: Vec<f64> = (0..4000).map(|_| r.normal()).collect();
        let mut b: Vec<f64> = (0..4000).map(|_| r.normal()).collect();
        let mut c: Vec<f64> = (0..4000).map(|_| 0.15 + r.normal()).collect();
        sort(&mut a);
        sort(&mut b);
        sort(&mut c);
        let (d, p) = ks2(&a, &b, 4000.0, 4000.0);
        assert!(p > 0.01, "{d} {p}");
        assert!(ks2(&a, &c, 4000.0, 4000.0).1 < 1e-3);
        assert!(ad2(&a, &b).1 > 0.01, "{:?}", ad2(&a, &b));
        assert!(ad2(&a, &c).1 < 0.01);
    }

    /// A NaN in both samples stopped the merge from advancing (an endless loop on a gelu
    /// pair under Student-t inputs, where the kernel's output is NaN).
    #[test]
    fn nan_samples_terminate() {
        let a = [-1.0, f64::NAN, 0.0, 1.0];
        let b = [-2.0, f64::NAN, 2.0];
        assert!(ks2(&a, &b, 4.0, 3.0).0.is_nan());
        assert!(ad2(&a, &b).0.is_nan());
    }
}

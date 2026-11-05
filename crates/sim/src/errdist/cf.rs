/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The exact sum's (y's) law per (sign, binade) cell by characteristic-function inversion
//! (docs/errdist.md §5.2). The Markov chain carries the partial sums as merged atom clouds,
//! which blur the density near 0 (the cancellation cells that dominate flip rates); the
//! characteristic function of y is exact (a product over the terms' laws), so the final cell
//! masses are taken from it and each cell's conditional error law from the chain.

use std::collections::HashMap;
use std::sync::Arc as Rc;

use super::fine::C64;
use super::markov::Cells;
use super::schema::ChainSchema;

/// phi of the sum of all terms of the chain at t = k dt, k = 0..=n: each distinct term law's
/// phi by a rotation recurrence per atom, combined per group as (sum_parts w phi)^n.
fn phi_grid(sch: &ChainSchema, dt: f64, n: usize, h: f64) -> Vec<C64> {
    let mut per_law: HashMap<usize, Vec<C64>> = HashMap::new();
    let mut logm = vec![0f64; n + 1];
    let mut arg = vec![0f64; n + 1];
    let mut zero = vec![false; n + 1];
    for st in &sch.steps {
        for g in &st.pop.groups {
            let mut s = vec![C64::new(0.0, 0.0); n + 1];
            for (w, tl) in &g.parts {
                let f = per_law.entry(Rc::as_ptr(tl) as usize).or_insert_with(|| {
                    let mut f = vec![C64::new(0.0, 0.0); n + 1];
                    // Atoms merged in bins of width h, each at its mean (the phase error is
                    // second order: <= (t h)^2 / 8).
                    let mut merged: HashMap<i64, (f64, f64)> = HashMap::new();
                    for a in &tl.atoms {
                        let key = (a.0 / h).round() as i64;
                        let e = merged.entry(key).or_default();
                        e.0 += a.0 * a.2;
                        e.1 += a.2;
                    }
                    for (sv, p) in merged.into_values() {
                        if p == 0.0 {
                            continue;
                        }
                        let (sn, cs) = (dt * sv / p).sin_cos();
                        let rot = C64::new(cs, sn);
                        let mut z = C64::new(p, 0.0);
                        for x in f.iter_mut() {
                            x.re += z.re;
                            x.im += z.im;
                            z = z.mul(rot);
                        }
                    }
                    f
                });
                for (x, y) in s.iter_mut().zip(f.iter()) {
                    x.re += w * y.re;
                    x.im += w * y.im;
                }
            }
            for k in 0..=n {
                let m = (s[k].re * s[k].re + s[k].im * s[k].im).sqrt();
                if m == 0.0 {
                    zero[k] = true;
                    continue;
                }
                logm[k] += g.n as f64 * m.ln();
                arg[k] += g.n as f64 * s[k].im.atan2(s[k].re);
            }
        }
    }
    (0..=n)
        .map(|k| {
            if zero[k] {
                return C64::new(0.0, 0.0);
            }
            let m = logm[k].exp();
            C64::new(m * arg[k].cos(), m * arg[k].sin())
        })
        .collect()
}

/// Per cell (index as `Cells`), P(y in the cell); None when the characteristic function does
/// not decay (lattice-like laws) or the moments are degenerate.
// ASSUME(N-ERRDIST-CF): exact cell masses; the chain keeps the conditional error laws.
pub fn cell_masses(sch: &ChainSchema, cells: &Cells) -> Option<Vec<f64>> {
    let (mut mean, mut m2) = (0.0, 0.0);
    for st in &sch.steps {
        for g in &st.pop.groups {
            let (mut e1, mut e2) = (0.0, 0.0);
            for (w, tl) in &g.parts {
                for a in &tl.atoms {
                    e1 += w * a.2 * a.0;
                    e2 += w * a.2 * a.0 * a.0;
                }
            }
            mean += g.n as f64 * e1;
            m2 += g.n as f64 * (e2 - e1 * e1);
        }
    }
    let sd = m2.max(0.0).sqrt();
    if sd <= 0.0 || !sd.is_finite() {
        return None;
    }
    let tmax = 12.0 / sd;
    let n = 1usize << 12;
    let dt = tmax / n as f64;
    // Decay check over [tmax, 3 tmax] (a lattice law's phi recurs).
    let h = sd / 4096.0;
    let far = phi_grid(sch, 3.0 * tmax / 256.0, 256, h);
    if far
        .iter()
        .skip(86)
        .any(|f| (f.re * f.re + f.im * f.im).sqrt() > 1e-9)
    {
        return None;
    }
    let ph = phi_grid(sch, dt, n, h);
    let simpson = |f: &dyn Fn(usize, f64) -> f64| -> f64 {
        let mut s = 0.0;
        for k in 0..=n {
            let w = if k == 0 || k == n {
                1.0
            } else if k % 2 == 1 {
                4.0
            } else {
                2.0
            };
            s += w * f(k, k as f64 * dt);
        }
        s * dt / 3.0
    };
    let big = mean.abs() + 12.0 * sd;
    // P(a < y <= b), finite a < b.
    let interval = |a: f64, b: f64| -> f64 {
        simpson(&|k, t| {
            if k == 0 {
                return b - a;
            }
            // Re[phi (e^{-ita} - e^{-itb}) / (it)]
            let (sa, ca) = (t * a).sin_cos();
            let (sb, cb) = (t * b).sin_cos();
            let d = C64::new(ca - cb, -sa + sb);
            let p = ph[k].mul(d);
            p.im / t
        }) / std::f64::consts::PI
    };
    // P(y > a).
    let above = |a: f64| -> f64 {
        0.5 + simpson(&|k, t| {
            if k == 0 {
                return mean - a;
            }
            let (sa, ca) = (t * a).sin_cos();
            let p = ph[k].mul(C64::new(ca, -sa));
            p.im / t
        }) / std::f64::consts::PI
    };
    let mut out = vec![0.0; cells.n()];
    for sign in [1.0f64, -1.0] {
        for b in cells.bmin..=cells.bmax {
            let lo = if b == cells.bmin { 0.0 } else { 2f64.powi(b) };
            let hi = if b == cells.bmax {
                f64::INFINITY
            } else {
                2f64.powi(b + 1)
            };
            let c = cells.of(sign * 2f64.powi(b) * 1.5);
            out[c] = if lo > big {
                0.0
            } else if hi > big {
                // The open-ended part: from the tail formula.
                if sign > 0.0 {
                    above(lo)
                } else {
                    1.0 - above(-lo)
                }
            } else if sign > 0.0 {
                interval(lo, hi)
            } else {
                interval(-hi, -lo)
            };
            out[c] = out[c].max(0.0);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc as Rc;

    use super::*;
    use crate::errdist::fine::Group;
    use crate::errdist::fine::Pop;
    use crate::errdist::fine::TermLaw;
    use crate::errdist::schema::StepSchema;

    #[test]
    fn cell_masses_of_a_uniform_sum() {
        // 64 terms uniform on a fine grid in [-1, 1]: y close to normal with sd sqrt(64 / 3).
        let k = 512;
        let atoms: Vec<(f64, i32, f64)> = (0..k)
            .map(|i| ((i as f64 + 0.5) / k as f64 * 2.0 - 1.0, 0, 1.0 / k as f64))
            .collect();
        let sch = ChainSchema {
            steps: vec![StepSchema {
                model: crate::errdist::chain::StepModel::Mma(crate::fp::DEFAULT_MMA),
                frac: 25,
                fp4: false,
                pop: Pop {
                    groups: vec![Group {
                        n: 64,
                        parts: vec![(1.0, Rc::new(TermLaw::new(atoms)))],
                    }],
                },
            }],
            round16: None,
            pruned: 0.0,
            notes: vec![],
        };
        let cells = Cells { bmin: -20, bmax: 6 };
        let m = cell_masses(&sch, &cells).unwrap();
        let tot: f64 = m.iter().sum();
        assert!((tot - 1.0).abs() < 1e-6, "{tot}");
        // Small binades: density at 0 times the width; f(0) with the Edgeworth term of the
        // uniform's excess kurtosis -6/5: f(0) = phi(0) / sd * (1 + kappa4 / (8 n)).
        let sd = (64.0f64 / 3.0).sqrt();
        let f0 = 1.0 / (sd * (2.0 * std::f64::consts::PI).sqrt()) * (1.0 - 1.2 / (8.0 * 64.0));
        for b in [-12, -8] {
            let c = cells.of(1.5 * 2f64.powi(b));
            let want = f0 * 2f64.powi(b);
            assert!(
                (m[c] / want - 1.0).abs() < 2e-4,
                "b {b}: {} vs {want}",
                m[c]
            );
        }
    }
}

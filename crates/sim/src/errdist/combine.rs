/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! From chain laws to estimates: one kernel (C - exact) and a pair (C - R) through their
//! shared final rounding (docs/errdist.md §6).
//!
//! Pair: both kernels end in a round toward zero of y + delta_K to the f32 grid of y's binade
//! (delta_K: the deviation before the final rounding). With U = frac(y / ulp) uniform and
//! independent of (delta_C, delta_R), D / ulp = floor(U + c) - floor(U + r) =
//! floor(d) + Bernoulli(frac(d)), d = (delta_C - delta_R) / ulp, c = delta_C / ulp,
//! r = delta_R / ulp; delta_C and delta_R are taken independent given y's cell
//! (N-ERRDIST-PAIR).

use std::collections::BTreeMap;

use super::markov::ChainLaw;
use super::markov::Hist;
use super::markov::ulp_exp;
use super::report::ErrorBar;
use super::report::Estimate;
use super::report::Moments;
use super::report::PairMoments;
use super::spec::Format;
use super::stats::QUANTILES;

/// A mixture law over real values: (value, prob), sorted, merged.
#[derive(Clone, Debug, Default)]
pub struct Law {
    pub a: Vec<(f64, f64)>,
}

impl Law {
    pub fn from(mut a: Vec<(f64, f64)>) -> Law {
        a.retain(|x| x.1 > 0.0);
        a.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        let mut o: Vec<(f64, f64)> = vec![];
        for (v, p) in a {
            match o.last_mut() {
                Some(l) if l.0 == v => l.1 += p,
                _ => o.push((v, p)),
            }
        }
        Law { a: o }
    }
    pub fn mass(&self) -> f64 {
        self.a.iter().map(|x| x.1).sum()
    }
    pub fn moments(&self) -> Moments {
        let m = self.mass().max(1e-300);
        let mean = self.a.iter().map(|x| x.0 * x.1).sum::<f64>() / m;
        let m2 = self.a.iter().map(|x| x.0 * x.0 * x.1).sum::<f64>() / m;
        let var = (m2 - mean * mean).max(0.0);
        Moments {
            mean,
            var,
            std: var.sqrt(),
            mean_abs: self.a.iter().map(|x| x.0.abs() * x.1).sum::<f64>() / m,
            p_zero: self
                .a
                .iter()
                .filter(|x| x.0 == 0.0)
                .map(|x| x.1)
                .sum::<f64>()
                / m,
            quantiles: QUANTILES.iter().map(|&p| (p, self.quantile(p))).collect(),
            max_abs: self
                .a
                .iter()
                .filter(|x| x.1 > 1e-12 * m)
                .map(|x| x.0.abs())
                .fold(None, |a: Option<f64>, b| Some(a.map_or(b, |a| a.max(b)))),
        }
    }
    pub fn quantile(&self, p: f64) -> f64 {
        let m = self.mass();
        let mut c = 0.0;
        for &(v, q) in &self.a {
            c += q / m;
            if c >= p - 1e-15 {
                return v;
            }
        }
        self.a.last().map_or(f64::NAN, |x| x.0)
    }
    /// The law restricted to [q(alpha), q(1 - alpha)] (renormalized), and the mass removed.
    pub fn trimmed(&self, alpha: f64) -> (Law, f64) {
        let (lo, hi) = (self.quantile(alpha), self.quantile(1.0 - alpha));
        let a: Vec<(f64, f64)> = self
            .a
            .iter()
            .copied()
            .filter(|x| x.0 >= lo && x.0 <= hi)
            .collect();
        let m = self.mass().max(1e-300);
        let kept: f64 = a.iter().map(|x| x.1).sum();
        (Law { a }, 1.0 - kept / m)
    }
    pub fn cdf(&self) -> Vec<(f64, f64)> {
        let m = self.mass().max(1e-300);
        let mut c = 0.0;
        self.a
            .iter()
            .map(|&(v, q)| {
                c += q / m;
                (v, c)
            })
            .collect()
    }
}

fn hist_atoms(h: &Hist, unit: f64, g: i64) -> Vec<(f64, f64)> {
    let mut v: Vec<(f64, f64)> =
        h.h.iter()
            .map(|(&i, &p)| (i as f64 / g as f64 * unit, p))
            .collect();
    v.push((0.0, h.z));
    v
}

/// One kernel: the error C - exact in ulps of the output's f32 binade (and absolute).
pub fn single(law: &ChainLaw, out_fmt: Format, round16: Option<Format>) -> Estimate {
    let mut ulp_atoms = vec![];
    let mut abs_atoms = vec![];
    let mut tail = 0.0;
    for (c, _p, h, _pre) in &law.fin {
        let b = law.cells.binade(*c);
        let ue = 2f64.powi(ulp_exp(b));
        ulp_atoms.extend(hist_atoms(h, 1.0, law.g));
        abs_atoms.extend(hist_atoms(h, ue, law.g));
        tail += h.tail;
    }
    let lu = Law::from(ulp_atoms);
    let la = Law::from(abs_atoms);
    if std::env::var("PTXV_ERRDIST_DEBUG").is_ok() {
        for (c, p, h, _) in &law.fin {
            let l = Law::from(hist_atoms(h, 1.0, law.g));
            let m = l.moments();
            eprintln!(
                "cell sign {} b {:>4}: p {:.4e} P0 {:.4} E|e| {:.3} sd {:.3}",
                law.cells.sign(*c),
                if *c == 0 { 0 } else { law.cells.binade(*c) },
                p,
                m.p_zero,
                m.mean_abs,
                m.std
            );
        }
    }
    let mut eb = ErrorBar {
        kind: "tv".into(),
        notes: vec!["residue model: see `residue` (TV of each residue law to uniform); coarse partial-sum laws merged to 384 atoms".into()],
        model_mass: Some(tail),
        ..Default::default()
    };
    eb.notes.extend(law.notes.iter().cloned());
    let est = Estimate {
        tier: 1,
        method: "residue-exact Markov chain on (cell, low bits) of the partial sums (fine residue layer, steptrans)"
            .into(),
        outputs: 1,
        error_bar: eb,
        ..Default::default()
    };
    single_laws(est, &lu, &la, out_fmt, round16)
}

/// Fill a single-kernel estimate from the error's law in output ulps (`lu`) and absolute
/// (`la`).
// ASSUME(N-ERRDIST-FLIP): y uniform on the 16-bit grid at the |e| scale.
pub fn single_laws(
    mut est: Estimate,
    lu: &Law,
    la: &Law,
    out_fmt: Format,
    round16: Option<Format>,
) -> Estimate {
    est.abs = la.moments();
    match round16.filter(|_| out_fmt != Format::F32) {
        None => {
            est.ulp = Some(ulp_moments(&est, lu));
            if est.tier != 0 {
                est.error_bar.notes.push("ulp-scale mean / std over the central 1 - 2e-6 of the law (cancellation tails excluded); quantiles and P(0) from the full law".into());
            }
            // C differs from RN(y) when |e| > 1/2 ulp; at exactly 1/2 (a tie of y) with
            // probability 1/2 (round to even).
            let p =
                lu.a.iter()
                    .map(|x| {
                        if x.0.abs() > 0.5 {
                            x.1
                        } else if x.0.abs() == 0.5 {
                            0.5 * x.1
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
                    / lu.mass().max(1e-300);
            let p = p.max(0.0) + 0.0;
            est.p_differ = Some((p, p, p));
            // Output - RN(y) in output ulps.
            est.pmf_ulp = round_pmf(lu);
        }
        Some(f) => {
            // Misrounding: P(rnd16(a) != rnd16(y)) = E[min(1, |e| / ulp16)], y's position on the
            // 16-bit grid uniform; e in f32 ulps, ulp16 = 2^(23 - mbits) f32 ulps.
            let k = 2f64.powi(23 - f.mbits());
            let m = lu.mass().max(1e-300);
            let p =
                lu.a.iter()
                    .map(|x| x.1 * (x.0.abs() / k).min(1.0))
                    .sum::<f64>()
                    / m;
            let p = p.max(0.0) + 0.0;
            est.p_differ = Some((p, p, p));
            // rnd16(a) - rnd16(y) in 16-bit ulps: the tent kernel over e / ulp16.
            let mut tp = TentPmf::default();
            for &(x, q) in &lu.a {
                tp.point(x / k, q, 1.0);
            }
            let (pmf, nn) = tp.finish(None);
            est.pmf_ulp = pmf;
            est.error_bar.notes.extend(nn);
            est.method += &format!(
                "; f32 accumulator error before the {f:?} output rounding, misrounding E[min(1, |e| / ulp16)]"
            );
            // `ulp` would be in output ulps; this law is the f32 accumulator's, so it goes
            // into a note.
            let m = lu.moments();
            est.error_bar.notes.push(format!(
                "f32 accumulator error before the {f:?} rounding, in f32 ulps of y: std {:.4e}, mean |e| {:.4e}, P(e = 0) {:.4}, q0.001 {:.2}, q0.999 {:.2}",
                m.std,
                m.mean_abs,
                m.p_zero,
                lu.quantile(0.001),
                lu.quantile(0.999)
            ));
            est.ulp = None;
        }
    }
    est
}

/// Ulp-scale moments: the analytic laws reach cancellation cells with |y| far below the
/// partial sums, where |e| / ulp(y) is astronomically large at tiny probability; the
/// variance would be dominated by them. Moments of the central 1 - 2e-6 (quantiles and P(0)
/// are unaffected); the excluded mass is in the method string.
pub(crate) fn ulp_moments(est: &Estimate, l: &Law) -> Moments {
    let full = l.moments();
    if est.tier == 0 {
        return full;
    }
    let (t, cut) = l.trimmed(1e-6);
    let mut m = t.moments();
    m.quantiles = full.quantiles;
    m.p_zero = full.p_zero;
    m.max_abs = full.max_abs;
    let _ = cut;
    m
}

fn cell_key(law: &ChainLaw, c: usize) -> (i8, i32) {
    (
        law.cells.sign(c) as i8,
        if c == 0 {
            i32::MIN
        } else {
            law.cells.binade(c)
        },
    )
}

/// D / ulp per cell under the PR-1 combination: d = delta_C - delta_R (pre-final deviations,
/// independent), D = floor(d) + Bernoulli(frac(d)) (y's position uniform); magnitude
/// direction.
fn pair_cell_indep(
    prec: &Hist,
    pc: f64,
    prer: &Hist,
    pr: f64,
    g: i64,
    neg: bool,
) -> Vec<(f64, f64)> {
    let a: Vec<(i64, f64)> = std::iter::once((0i64, prec.z / pc))
        .chain(prec.h.iter().map(|(&i, &p)| (i, p / pc)))
        .collect();
    let b: Vec<(i64, f64)> = std::iter::once((0i64, prer.z / pr))
        .chain(prer.h.iter().map(|(&i, &p)| (i, p / pr)))
        .collect();
    let mut d: BTreeMap<i64, f64> = BTreeMap::new();
    for &(i, p) in &a {
        for &(j, q) in &b {
            *d.entry(super::markov::quant(i - j)).or_default() += p * q;
        }
    }
    let mut out = vec![];
    for (i, p) in d {
        let v = if neg { -(i as f64) } else { i as f64 } / g as f64;
        let fl = v.floor();
        let fr = v - fl;
        out.push((fl, p * (1.0 - fr)));
        out.push((fl + 1.0, p * fr));
    }
    out
}

/// D / ulp per cell, magnitude direction, from the two kernels' errors after the final
/// rounding. With y = (N + U) ulp (magnitudes), a kernel's error is (m - U) ulp for an
/// integer m (its result N + m), so each kernel's law gives the joint law of (U, m); U is
/// common to both kernels. m_C and m_R are taken independent given U (and the cell):
/// D = m_C - m_R. Errors in the log-spaced bins (|e| beyond the exact range, cancellation
/// cells) have no usable U and fall back to `pair_cell_indep`'s rule against the other law.
fn pair_cell_cond(hc: &Hist, pc: f64, hr: &Hist, pr: f64, g: i64, neg: bool) -> Vec<(f64, f64)> {
    let w_exact = super::markov::G * super::markov::W_ULP;
    // Per kernel: U bin -> (m -> p), the far atoms (value in ulps, p), and the near mass.
    #[allow(clippy::type_complexity)]
    let split = |h: &Hist, pm: f64| -> (BTreeMap<i64, BTreeMap<i64, f64>>, Vec<(f64, f64)>, f64) {
        let mut near: BTreeMap<i64, BTreeMap<i64, f64>> = BTreeMap::new();
        let mut far = vec![];
        let mut nm = 0.0;
        for (i, p) in std::iter::once((0i64, h.z)).chain(h.h.iter().map(|(&i, &p)| (i, p))) {
            let p = p / pm;
            if p <= 0.0 {
                continue;
            }
            let im = if neg { -i } else { i };
            if im.abs() <= w_exact {
                let u = (-im).rem_euclid(g);
                let m = (im + u) / g;
                *near.entry(u).or_default().entry(m).or_default() += p;
                nm += p;
            } else {
                far.push((im as f64 / g as f64, p));
            }
        }
        (near, far, nm)
    };
    let (nc, fc, mc) = split(hc, pc);
    let (nr, fr, mr) = split(hr, pr);
    let marg = |n: &BTreeMap<i64, BTreeMap<i64, f64>>| -> BTreeMap<i64, f64> {
        let mut o: BTreeMap<i64, f64> = BTreeMap::new();
        for v in n.values() {
            for (&m, &p) in v {
                *o.entry(m).or_default() += p;
            }
        }
        let t: f64 = o.values().sum();
        o.values_mut().for_each(|p| *p /= t.max(1e-300));
        o
    };
    let (mc_marg, mr_marg) = (marg(&nc), marg(&nr));
    let mut d: BTreeMap<i64, f64> = BTreeMap::new();
    // Near x near: U from both kernels' laws (averaged), m's independent given U.
    let joint = mc * mr;
    if joint > 0.0 {
        let us: std::collections::BTreeSet<i64> = nc.keys().chain(nr.keys()).copied().collect();
        for u in us {
            let lc = nc.get(&u);
            let lr = nr.get(&u);
            let sc: f64 = lc.map_or(0.0, |v| v.values().sum());
            let sr: f64 = lr.map_or(0.0, |v| v.values().sum());
            let pu = 0.5 * (sc / mc.max(1e-300) + sr / mr.max(1e-300));
            let cc: Vec<(i64, f64)> = match lc {
                Some(v) => v.iter().map(|(&m, &p)| (m, p / sc)).collect(),
                None => mc_marg.iter().map(|(&m, &p)| (m, p)).collect(),
            };
            let cr: Vec<(i64, f64)> = match lr {
                Some(v) => v.iter().map(|(&m, &p)| (m, p / sr)).collect(),
                None => mr_marg.iter().map(|(&m, &p)| (m, p)).collect(),
            };
            for &(a, p) in &cc {
                for &(b, q) in &cr {
                    *d.entry(a - b).or_default() += joint * pu * p * q;
                }
            }
        }
    }
    let mut out: Vec<(f64, f64)> = d.into_iter().map(|(k, p)| (k as f64, p)).collect();
    // Far parts against the other kernel's whole law: D = floor(d) + Bernoulli(frac(d)).
    let all = |n: &BTreeMap<i64, BTreeMap<i64, f64>>, f: &[(f64, f64)]| -> Vec<(f64, f64)> {
        let mut v: Vec<(f64, f64)> = vec![];
        for (&u, ms) in n {
            for (&m, &p) in ms {
                v.push((m as f64 - u as f64 / g as f64, p));
            }
        }
        v.extend_from_slice(f);
        v
    };
    let mut fb = |xs: &[(f64, f64)], ys: &[(f64, f64)]| {
        for &(x, p) in xs {
            for &(y, q) in ys {
                let v = x - y;
                let fl = v.floor();
                let frc = v - fl;
                out.push((fl, p * q * (1.0 - frc)));
                out.push((fl + 1.0, p * q * frc));
            }
        }
    };
    if !fc.is_empty() {
        fb(&fc, &all(&nr, &fr));
    }
    if !fr.is_empty() {
        // R far against C near only (C far x R far counted above).
        let cn = all(&nc, &[]);
        fb(&cn, &fr);
    }
    out
}

/// A pair: D = C - R in ulps of y's f32 binade; for 16-bit outputs, the flip probability.
// ASSUME(N-ERRDIST-PAIR, N-ERRDIST-FLIP): the two kernels' result offsets independent
// given y's cell and y's position U in its ulp (N-ERRDIST-PAIR).
pub fn pair(r: &ChainLaw, c: &ChainLaw, round16: Option<Format>) -> Estimate {
    assert_eq!(r.g, c.g, "pair: both chain laws on one bin grid");
    let g = r.g;
    // The PR-1 combination (pre-final deviations independent, y's position uniform), for
    // comparison.
    let indep = std::env::var("PTXV_ERRDIST_PAIR_INDEP").is_ok();
    let mut dbg_pd = (0.0, 0.0);
    let mut rc: BTreeMap<(i8, i32), (f64, &Hist, &Hist)> = BTreeMap::new();
    for (k, p, h, pre) in &r.fin {
        rc.insert(cell_key(r, *k), (*p, h, pre));
    }
    let mut d_atoms: Vec<(f64, f64)> = vec![];
    let mut abs_atoms: Vec<(f64, f64)> = vec![];
    let mut dabs: Vec<(f64, f64)> = vec![];
    let (mut er, mut ec) = (vec![], vec![]);
    let mut unmatched = 0.0;
    for (k, pc, hc, prec) in &c.fin {
        let key = cell_key(c, *k);
        let Some(&(pr, hr, prer)) = rc.get(&key) else {
            unmatched += pc;
            continue;
        };
        if key.1 == i32::MIN {
            d_atoms.push((0.0, 0.5 * (pc + pr)));
            continue;
        }
        let w = 0.5 * (pc + pr);
        let ue = 2f64.powi(ulp_exp(key.1));
        er.extend(
            hist_atoms(hr, ue, r.g)
                .into_iter()
                .map(|(v, p)| (v, p / pr * w)),
        );
        ec.extend(
            hist_atoms(hc, ue, c.g)
                .into_iter()
                .map(|(v, p)| (v, p / pc * w)),
        );
        let neg = key.0 < 0;
        // D in ulps (magnitude direction: + is away from zero), as (value, prob) atoms.
        if std::env::var("PTXV_ERRDIST_DEBUG").is_ok() {
            let pi: f64 = pair_cell_indep(prec, *pc, prer, pr, g, neg)
                .iter()
                .filter(|x| x.0 != 0.0)
                .map(|x| x.1)
                .sum();
            let pq: f64 = pair_cell_cond(hc, *pc, hr, pr, g, neg)
                .iter()
                .filter(|x| x.0 != 0.0)
                .map(|x| x.1)
                .sum();
            dbg_pd.0 += w * pi;
            dbg_pd.1 += w * pq;
        }
        let dm: Vec<(f64, f64)> = if indep {
            pair_cell_indep(prec, *pc, prer, pr, g, neg)
        } else {
            pair_cell_cond(hc, *pc, hr, pr, g, neg)
        };
        for (k, q) in dm {
            if q <= 0.0 {
                continue;
            }
            let k = if neg { -k } else { k };
            d_atoms.push((k, w * q));
            dabs.push((k * ue, w * q));
            match round16 {
                // The outputs differ by one 16-bit ulp with probability min(1, |D| / ulp16).
                Some(f) => {
                    let pf = (k.abs() / 2f64.powi(23 - f.mbits())).min(1.0);
                    let u16_ = 2f64.powi(key.1.max(f.emin()) - f.mbits());
                    abs_atoms.push((u16_, w * q * pf / 2.0));
                    abs_atoms.push((-u16_, w * q * pf / 2.0));
                    abs_atoms.push((0.0, w * q * (1.0 - pf)));
                }
                None => abs_atoms.push((k * ue, w * q)),
            }
        }
    }
    if std::env::var("PTXV_ERRDIST_DEBUG").is_ok() {
        eprintln!(
            "pair P(D != 0): independent {:.4e}, given U {:.4e}",
            dbg_pd.0, dbg_pd.1
        );
    }
    let ld = Law::from(d_atoms);
    let la = Law::from(abs_atoms);
    let (lr, lc) = (Law::from(er), Law::from(ec));
    let (mr, mc, md) = (lr.moments(), lc.moments(), ld.moments());
    let cov = 0.5 * (mr.var + mc.var - Law::from(dabs).moments().var);
    let mut eb = ErrorBar {
        kind: "tv".into(),
        notes: vec![
            "D / ulp = floor(d) + Bernoulli(frac(d)): y's position in its ulp uniform, the two kernels' pre-final deviations independent given y's binade (N-ERRDIST-PAIR)".into(),
            "pair moments: each kernel's pre-final deviation (absolute); cov from Var(D) = Var(e_R) + Var(e_C) - 2 cov".into(),
        ],
        model_mass: Some(unmatched),
        ..Default::default()
    };
    eb.notes.extend(r.notes.iter().chain(&c.notes).cloned());
    let mut est = Estimate {
        tier: 1,
        method: "residue-exact chains of both kernels, combined given the shared exact value's ulp position".into(),
        outputs: 1,
        abs: la.moments(),
        error_bar: eb,
        pair: Some(PairMoments {
            mean_r: mr.mean,
            var_r: mr.var,
            mean_c: mc.mean,
            var_c: mc.var,
            cov_rc: cov,
            corr_rc: cov / (mr.var * mc.var).sqrt().max(1e-300),
        }),
        ..Default::default()
    };
    match round16 {
        None => {
            let p = 1.0 - md.p_zero;
            let p = p.max(0.0) + 0.0;
            est.p_differ = Some((p, p, p));
            est.pmf_ulp = ld.a.iter().map(|x| (x.0 as i64, x.1)).collect();
            est.ulp = Some(ulp_moments(&est, &ld));
            if est.tier != 0 {
                est.error_bar.notes.push("ulp-scale mean / std over the central 1 - 2e-6 of the law (cancellation tails excluded); quantiles and P(0) from the full law".into());
            }
            let _ = md;
        }
        Some(f) => {
            // A flip of the 16-bit outputs: P = E|D| / ulp16 (y's position on the 16-bit grid
            // uniform), |D| in f32 ulps.
            let k = 2f64.powi(23 - f.mbits());
            let m = ld.mass().max(1e-300);
            let p =
                ld.a.iter()
                    .map(|x| x.1 * (x.0.abs() / k).min(1.0))
                    .sum::<f64>()
                    / m;
            let p = p.max(0.0) + 0.0;
            est.p_differ = Some((p, p, p));
            // R16 - C16 in 16-bit ulps: the tent kernel over D / ulp16 (tails beyond one ulp
            // when |D| exceeds it).
            let mut tp = TentPmf::default();
            for &(x, q) in &ld.a {
                tp.point(x / k, q, 1.0);
            }
            let (pmf, nn) = tp.finish(None);
            est.pmf_ulp = pmf;
            est.error_bar.notes.extend(nn);
            est.method += &format!(
                "; {f:?} outputs flip with probability E[min(1, |D| / ulp16)], D the f32 difference"
            );
            est.ulp = Some(ulp_moments(
                &est,
                &Law::from(est.pmf_ulp.iter().map(|x| (x.0 as f64, x.1)).collect()),
            ));
            est.error_bar.notes.push("ulp-scale mean / std over the central 1 - 2e-6 of the PMF (tails from tiny |y| excluded)".into());
            est.error_bar.notes.push(format!(
                "f32 difference before the {f:?} rounding, in f32 ulps of y: std {:.4e}, mean |D| {:.4e}, P(D = 0) {:.4}, q0.001 {:.1}, q0.999 {:.1}",
                md.std,
                md.mean_abs,
                md.p_zero,
                ld.quantile(0.001),
                ld.quantile(0.999)
            ));
        }
    }
    est
}

// ------------------------------------------------------------- PMFs on the output-ulp axis
//
// Every tier reports `pmf_ulp` on one axis: k = (output - reference) / ulp_out(y), the
// reference being R (pair) or the correctly rounded exact value RN(y) (single). With y's
// position t in its output ulp uniform, an error x (output ulps, before the output
// rounding) gives k = Q(t + x) - Q(t), so P(k | x) = max(0, 1 - |x - k|) (a tent); a final
// round toward zero against RN(y) shifts x by -1/2 toward |y|.

/// Accumulates tent-kernel PMFs over a mixture of error laws.
#[derive(Default)]
pub struct TentPmf {
    m: BTreeMap<i64, f64>,
    /// Total weight passed in; weight whose spread exceeds the axis cap.
    total: f64,
    lost: f64,
}

/// Widest k range a single component may cover.
const TENT_SPAN: f64 = 4096.0;

impl TentPmf {
    /// x a point (output ulps, already shifted), recorded as sign * k.
    pub fn point(&mut self, x: f64, w: f64, sign: f64) {
        self.total += w;
        if !x.is_finite() || x.abs() > 1e12 {
            self.lost += w;
            return;
        }
        let k0 = x.floor();
        let f = x - k0;
        let sg = if sign < 0.0 { -1i64 } else { 1 };
        *self.m.entry(sg * k0 as i64).or_default() += w * (1.0 - f);
        if f > 0.0 {
            *self.m.entry(sg * (k0 as i64 + 1)).or_default() += w * f;
        }
    }

    /// x ~ N(m, s^2) (output ulps, shifted): P(k) = G(k-1) - 2 G(k) + G(k+1),
    /// G(a) = E[(x - a)+].
    pub fn normal(&mut self, m: f64, s: f64, w: f64, sign: f64) {
        if s.is_nan() || s <= 1e-9 {
            return self.point(m, w, sign);
        }
        self.total += w;
        if !m.is_finite() || m.abs() > 1e12 || 16.0 * s > TENT_SPAN {
            self.lost += w;
            return;
        }
        let g = |a: f64| {
            let z = (m - a) / s;
            (m - a) * super::special::phi(z) + s * super::special::phi_pdf(z)
        };
        let (k0, k1) = (
            (m - 8.0 * s).floor() as i64 - 1,
            (m + 8.0 * s).ceil() as i64 + 1,
        );
        let sg = if sign < 0.0 { -1i64 } else { 1 };
        for k in k0..=k1 {
            let kf = k as f64;
            let q = (g(kf - 1.0) - 2.0 * g(kf) + g(kf + 1.0)).max(0.0);
            if q > 0.0 {
                *self.m.entry(sg * k).or_default() += w * q;
            }
        }
    }

    /// Zero-inflated normal with overall mean m and variance v (output ulps), P(!= 0) p,
    /// shifted by `shift`.
    pub fn zi(&mut self, p: f64, m: f64, v: f64, shift: f64, w: f64, sign: f64) {
        let p = p.clamp(0.0, 1.0);
        if p < 1.0 {
            self.point(-shift, w * (1.0 - p), sign);
        }
        if p > 1e-300 {
            let mc = m / p;
            let vc = ((v + m * m) / p - mc * mc).max(0.0);
            self.normal(mc - shift, vc.sqrt(), w * p, sign);
        }
    }

    /// Weight passed in so far.
    pub fn weight(&self) -> f64 {
        self.total
    }

    /// The normalized PMF; with `p_differ`, the mass off 0 is rescaled to it (the tier's
    /// P(differ) may use a finer position law than the uniform one). Returns (pmf, notes).
    pub fn finish(self, p_differ: Option<f64>) -> (Vec<(i64, f64)>, Vec<String>) {
        let mut notes = vec![];
        let placed: f64 = self.m.values().sum();
        if placed <= 0.0 {
            return (vec![], notes);
        }
        let mut v: Vec<(i64, f64)> = self.m.into_iter().map(|(k, q)| (k, q / placed)).collect();
        if self.lost > 1e-9 * self.total {
            notes.push(format!(
                "pmf_ulp leaves out {:.3e} of the mass (error spread wider than {TENT_SPAN} output ulps; renormalized without it)",
                self.lost / self.total.max(1e-300)
            ));
        }
        if let Some(p) = p_differ {
            let off: f64 = v.iter().filter(|x| x.0 != 0).map(|x| x.1).sum();
            if off > 0.0 && (off - p).abs() > 1e-9 {
                if (off - p).abs() > 0.02 * p.max(1e-300) {
                    notes.push(format!(
                        "pmf_ulp: tent model with y's position uniform in its ulp gives P(k != 0) {off:.4e}; the off-zero mass is rescaled to the tier's P(differ) {p:.4e}"
                    ));
                }
                let s = p / off;
                for x in &mut v {
                    x.1 = if x.0 == 0 { 1.0 - p } else { x.1 * s };
                }
                if !v.iter().any(|x| x.0 == 0) {
                    v.push((0, 1.0 - p));
                    v.sort_by_key(|x| x.0);
                }
            }
        }
        // Below Tier 3's resolution and bulky in JSON.
        v.retain(|x| x.1 > 1e-12);
        (v, notes)
    }
}

/// k = round(e) for a law over the output error e (output ulps) against the exact value:
/// the output is on the grid, so y's position is fixed by e and C - RN(y) = round(e), ties
/// split (round to even at a tie of y).
pub fn round_pmf(l: &Law) -> Vec<(i64, f64)> {
    let mut m: BTreeMap<i64, f64> = BTreeMap::new();
    let tot = l.mass().max(1e-300);
    for &(e, q) in &l.a {
        if !e.is_finite() || e.abs() > 1e15 {
            continue;
        }
        let f = e - e.floor();
        if f == 0.5 {
            *m.entry(e.floor() as i64).or_default() += 0.5 * q / tot;
            *m.entry(e.floor() as i64 + 1).or_default() += 0.5 * q / tot;
        } else {
            *m.entry(e.round() as i64).or_default() += q / tot;
        }
    }
    m.into_iter().filter(|x| x.1 > 1e-12).collect()
}

#[cfg(test)]
mod tent_tests {
    use super::*;

    #[test]
    fn tent_pmf_matches_the_flip_formula_and_sums_to_one() {
        // Point: x = 1.25 -> k = 1 w.p. 0.75, k = 2 w.p. 0.25.
        let mut t = TentPmf::default();
        t.point(1.25, 1.0, 1.0);
        let (v, _) = t.finish(None);
        assert_eq!(v, vec![(1, 0.75), (2, 0.25)]);
        // Normal: mass 1, mean m, P(k != 0) = E[min(1, |x|)] (checked by quadrature).
        for &(m, s) in &[(0.0, 0.3), (0.4, 0.7), (-1.3, 2.5)] {
            let mut t = TentPmf::default();
            t.normal(m, s, 1.0, 1.0);
            let (v, _) = t.finish(None);
            let tot: f64 = v.iter().map(|x| x.1).sum();
            let mean: f64 = v.iter().map(|x| x.0 as f64 * x.1).sum();
            assert!((tot - 1.0).abs() < 1e-9, "{tot}");
            assert!((mean - m).abs() < 1e-6, "{mean} vs {m}");
            let n = 200_000;
            let (a, b) = (m - 10.0 * s, m + 10.0 * s);
            let h = (b - a) / n as f64;
            let mut q = 0.0;
            for i in 0..n {
                let x = a + (i as f64 + 0.5) * h;
                q += h * super::super::special::phi_pdf((x - m) / s) / s * x.abs().min(1.0);
            }
            let off: f64 = v.iter().filter(|x| x.0 != 0).map(|x| x.1).sum();
            assert!((off - q).abs() < 1e-6, "{off} vs {q}");
        }
        // RZ against RN: x = 0 shifted by 1/2 differs half the time, toward zero.
        let mut t = TentPmf::default();
        t.zi(0.0, 0.0, 0.0, 0.5, 1.0, -1.0);
        let (v, _) = t.finish(None);
        assert_eq!(v, vec![(0, 0.5), (1, 0.5)]);
    }
}

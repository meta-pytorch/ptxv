/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tier 2: closed-form CLT (docs/errdist.md §5.3). Partial sums are normal with the terms'
//! exact moments. Each step's local error is the round toward zero to ulp(S) plus the
//! truncation of each term at Q = 2^(b(S) - frac), taken with uniform residues
//! (N-ERRDIST-CLT). The error is the sum of the steps' local errors, a normal with the summed
//! mean and variance; the Berry-Esseen bound 0.56 sum E|X_k - mu_k|^3 / sigma^3 bounds the
//! CDF error of that normal against the sum of the modeled local errors.

use std::collections::HashMap;
use std::sync::Arc as Rc;

use super::combine::TentPmf;
use super::fine::TermLaw;
use super::report::ErrorBar;
use super::report::Estimate;
use super::report::Moments;
use super::report::PairMoments;
use super::schema::ChainSchema;
use super::spec::Format;
use super::special::phi;

/// Gaussian chain summary: mean and variance of the exact sum, of the error before the last
/// step's rounding, and of the full error; third absolute moments for Berry-Esseen.
#[derive(Clone, Debug, Default)]
pub struct Clt {
    pub y_mean: f64,
    pub y_var: f64,
    /// Error excluding the last step's rounding (in absolute units).
    pub pre_mean: f64,
    pub pre_var: f64,
    pub mean: f64,
    pub var: f64,
    pub abs3: f64,
    /// E[S], S = sum of |terms| (the scale of the worst-case bound).
    pub s_mean: f64,
    /// P(error != 0), P(pre-final error != 0) (steps independent).
    pub p_nz: f64,
    pub pre_p_nz: f64,
    /// Total-variation distance to uniform of y's position in its f32 ulp and in its 16-bit
    /// ulp (the CLT flip formulas assume uniform positions).
    pub tv_pos32: f64,
    pub tv_pos16: f64,
}

/// P(X in the binade b of sign s) for X ~ N(m, v).
pub(super) fn binade_prob(m: f64, sd: f64, b: i32, s: f64) -> f64 {
    let (lo, hi) = (2f64.powi(b), 2f64.powi(b + 1));
    if s > 0.0 {
        phi((hi - m) / sd) - phi((lo - m) / sd)
    } else {
        phi((-lo - m) / sd) - phi((-hi - m) / sd)
    }
}

/// The binades worth summing for N(m, v).
pub(super) fn binades(m: f64, sd: f64) -> std::ops::RangeInclusive<i32> {
    let top = (m.abs() + 10.0 * sd).log2().ceil() as i32;
    (top - 60)..=top
}

/// A term law's atoms by (exponent, lowest set bit): truncation at 2^qe drops nothing when
/// qe <= the lowest bit, the whole value when qe > the exponent, else computed per atom.
struct Buckets {
    /// (exponent, lowest bit) -> (atoms (v, p), sums of p sign(v) |v|, p v^2, p |v|^3).
    b: Vec<((i32, i32), Atoms, BucketSums)>,
}

/// A bucket's atoms (v, p).
type Atoms = Vec<(f64, f64)>;

/// A bucket's sums of p sign(v) |v|, p v^2, p |v|^3.
type BucketSums = (f64, f64, f64);

impl Buckets {
    fn new(t: &TermLaw) -> Buckets {
        let mut m: HashMap<(i32, i32), (Atoms, BucketSums)> = HashMap::new();
        for x in t.atoms.iter().filter(|x| x.0 != 0.0) {
            let k = (super::spec::exp_floor(x.0), super::fine::lowest_bit(x.0));
            let e = m.entry(k).or_default();
            e.0.push((x.0, x.2));
            let a = x.0.abs();
            e.1.0 += x.2 * x.0.signum() * a;
            e.1.1 += x.2 * a * a;
            e.1.2 += x.2 * a * a * a;
        }
        Buckets {
            b: m.into_iter().map(|(k, (v, s))| (k, v, s)).collect(),
        }
    }
    /// (E[sign(t) r], E[r^2], E[r^3]), r = |t| mod 2^qe.
    fn trunc_moments(&self, qe: i32, q: f64) -> (f64, f64, f64) {
        let mut e = (0.0, 0.0, 0.0);
        for ((ex, lb), atoms, sums) in &self.b {
            if qe <= *lb {
                continue;
            }
            if qe > *ex {
                e.0 += sums.0;
                e.1 += sums.1;
                e.2 += sums.2;
                continue;
            }
            for &(v, p) in atoms {
                let r = v.abs() % q;
                e.0 += p * v.signum() * r;
                e.1 += p * r * r;
                e.2 += p * r * r * r;
            }
        }
        e
    }
}

pub fn chain(sch: &ChainSchema) -> Clt {
    let mut c = Clt::default();
    // Per (term law, binade): (E[sign(t) r], E[r^2], E[r^3]), r = |t| mod Q_b.
    let mut cache: HashMap<(usize, i32), (f64, f64, f64)> = HashMap::new();
    let mut buckets: HashMap<usize, Rc<Buckets>> = HashMap::new();
    let mut sm = 0.0;
    // Coarsest power of two every value so far is a multiple of.
    let mut lat = i32::MAX;
    let ns = sch.steps.len();
    for (k, st) in sch.steps.iter().enumerate() {
        for g in &st.pop.groups {
            for (w, t) in &g.parts {
                sm += w * g.n as f64 * t.atoms.iter().map(|x| x.2 * x.0).sum::<f64>();
                c.s_mean += w * g.n as f64 * t.atoms.iter().map(|x| x.2 * x.0.abs()).sum::<f64>();
            }
        }
        lat = lat.min(st.pop.lattice());
        let mut var = 0.0;
        for s2 in sch.steps.iter().take(k + 1) {
            for g in &s2.pop.groups {
                let (mut e1, mut e2) = (0.0, 0.0);
                for (w, t) in &g.parts {
                    for x in &t.atoms {
                        e1 += w * x.2 * x.0;
                        e2 += w * x.2 * x.0 * x.0;
                    }
                }
                var += g.n as f64 * (e2 - e1 * e1);
            }
        }
        let sd = var.max(1e-300).sqrt();
        let (mut lm, mut lv, mut l3) = (0.0, 0.0, 0.0);
        let mut last_m = 0.0;
        let mut last_v = 0.0;
        for b in binades(sm, sd) {
            for s in [1.0f64, -1.0] {
                let p = binade_prob(sm, sd, b, s);
                if p < 1e-300 {
                    continue;
                }
                let ulp = 2f64.powi(super::markov::ulp_exp(b));
                let q = 2f64.powi(b - st.frac);
                // Round toward zero: -s x, x uniform on the multiples of g = 2^lattice in
                // [0, ulp) (0 when the lattice is at least the ulp).
                let g = 2f64.powi(lat.clamp(-1074, 1023)).min(ulp);
                let m = (ulp / g).round();
                let (x1, x2, x3) = (
                    g * (m - 1.0) / 2.0,
                    g * g * (m - 1.0) * (2.0 * m - 1.0) / 6.0,
                    g.powi(3) * ((m - 1.0) * m / 2.0).powi(2) / m,
                );
                let (rm, r2, r3) = (-s * x1, x2, x3);
                // Terms' truncations.
                let (mut tm, mut t2, mut t3) = (0.0, 0.0, 0.0);
                for g in &st.pop.groups {
                    for (w, t) in &g.parts {
                        let bk = buckets
                            .entry(Rc::as_ptr(t) as usize)
                            .or_insert_with(|| Rc::new(Buckets::new(t)))
                            .clone();
                        let e = *cache
                            .entry((Rc::as_ptr(t) as usize, b))
                            .or_insert_with(|| bk.trunc_moments(b - st.frac, q));
                        tm -= w * g.n as f64 * e.0;
                        t2 += w * g.n as f64 * e.1;
                        t3 += w * g.n as f64 * e.2;
                    }
                }
                lm += p * (rm + tm);
                lv += p * (r2 + t2);
                l3 += p * (r3 + t3);
                last_m += p * rm;
                last_v += p * r2;
            }
        }
        c.mean += lm;
        c.var += (lv - lm * lm).max(0.0);
        c.abs3 += l3;
        if k + 1 == ns {
            c.pre_mean = c.mean - last_m;
            c.pre_var = (c.var - (last_v - last_m * last_m)).max(0.0);
            c.y_mean = sm;
            c.y_var = var;
        }
    }
    c
}

/// P(all of a step's terms aligned <= e).
fn all_le(st: &super::schema::StepSchema, e: i32) -> f64 {
    let mut f = 1.0;
    for g in &st.pop.groups {
        let mut gp = 0.0;
        for (w, t) in &g.parts {
            let tot: f64 = t.atoms.iter().map(|a| a.2).sum();
            let le: f64 = t.atoms.iter().filter(|a| a.1 <= e).map(|a| a.2).sum();
            gp += w * (le / tot.max(1e-300)).powi(g.n as i32) * tot.powi(g.n as i32);
        }
        f *= gp;
    }
    f
}

/// Gauss-Legendre nodes on [-1, 1] (8 points).
const GL8: [(f64, f64); 8] = [
    (-0.960_289_856_497_536_3, 0.101_228_536_290_376_26),
    (-0.796_666_477_413_626_7, 0.222_381_034_453_374_47),
    (-0.525_532_409_916_329, 0.313_706_645_877_887_3),
    (-0.183_434_642_495_649_8, 0.362_683_783_378_362),
    (0.183_434_642_495_649_8, 0.362_683_783_378_362),
    (0.525_532_409_916_329, 0.313_706_645_877_887_3),
    (0.796_666_477_413_626_7, 0.222_381_034_453_374_47),
    (0.960_289_856_497_536_3, 0.101_228_536_290_376_26),
];

/// Tier 2's chain summary with exact local laws: per step, the local error's moments from
/// tier 1's residue-exact step law (`tier1::local_moments`), averaged over the step's
/// configurations (accumulator cell, emax, result cell) under normal partial sums with
/// emax independent of them; steps independent.
// ASSUME(N-ERRDIST-CLT): normal partial sums, independent steps, emax independent of the
// partial sums.
pub fn chain_exact(sch: &ChainSchema, threads: usize) -> Clt {
    let mut c = Clt::default();
    let ns = sch.steps.len();
    let (mut m_prev, mut v_prev) = (0.0, 0.0);
    let ulpe = |b: i32| 2f64.powi(super::markov::ulp_exp(b));
    for (k, st) in sch.steps.iter().enumerate() {
        // The block's moments.
        let (mut mb, mut vb) = (0.0, 0.0);
        for g in &st.pop.groups {
            let (mut e1, mut e2) = (0.0, 0.0);
            for (w, t) in &g.parts {
                for x in &t.atoms {
                    e1 += w * x.2 * x.0;
                    e2 += w * x.2 * x.0 * x.0;
                    c.s_mean += w * x.2 * x.0.abs() * g.n as f64;
                }
            }
            mb += g.n as f64 * e1;
            vb += g.n as f64 * (e2 - e1 * e1).max(0.0);
        }
        let (m_now, v_now) = (m_prev + mb, v_prev + vb);
        // emax law.
        let mut levels: Vec<i32> = st
            .pop
            .groups
            .iter()
            .flat_map(|g| g.parts.iter().flat_map(|p| p.1.atoms.iter().map(|a| a.1)))
            .filter(|&a| a > super::fine::NO_ALIGN)
            .collect();
        levels.sort();
        levels.dedup();
        let fe: Vec<f64> = levels.iter().map(|&e| all_le(st, e)).collect();
        let pe = |i: usize| {
            fe[i]
                - if i == 0 {
                    all_le(st, levels[0] - 1)
                } else {
                    fe[i - 1]
                }
        };
        // Configurations and weights.
        let mut cfgs: Vec<super::tier1::Cfg> = vec![];
        let mut wts: Vec<f64> = vec![];
        let sdb = vb.max(1e-300).sqrt();
        let push_bs = |bx: Option<i32>,
                       sx: f64,
                       x_nodes: &[(f64, f64)],
                       e: i32,
                       from_acc: bool,
                       w0: f64,
                       cfgs: &mut Vec<super::tier1::Cfg>,
                       wts: &mut Vec<f64>| {
            // Result cells given the accumulator at x (nodes with weights).
            let mut acc: std::collections::BTreeMap<(i32, i8), f64> = Default::default();
            for &(x, wx) in x_nodes {
                let m = x + mb;
                for b in binades(m, sdb) {
                    for s in [1.0f64, -1.0] {
                        let p = binade_prob(m, sdb, b, s);
                        if p > 1e-12 {
                            *acc.entry((b, s as i8)).or_default() += wx * p;
                        }
                    }
                }
            }
            for ((bs, ss), p) in acc {
                let w = w0 * p;
                if w > 1e-9 {
                    cfgs.push(super::tier1::Cfg {
                        bx,
                        sx,
                        e,
                        from_acc,
                        bs,
                        ss: ss as f64,
                    });
                    wts.push(w);
                }
            }
        };
        if k == 0 {
            for (i, &e) in levels.iter().enumerate() {
                let p = pe(i);
                if p > 1e-12 {
                    push_bs(None, 1.0, &[(0.0, 1.0)], e, false, p, &mut cfgs, &mut wts);
                }
            }
        } else {
            let sdp = v_prev.max(1e-300).sqrt();
            for bx in binades(m_prev, sdp) {
                for sx in [1.0f64, -1.0] {
                    let px = binade_prob(m_prev, sdp, bx, sx);
                    if px < 1e-10 {
                        continue;
                    }
                    // Nodes in the cell, weighted by the normal density (normalized to 1).
                    let (lo, hi) = (2f64.powi(bx), 2f64.powi(bx + 1));
                    let mut nodes: Vec<(f64, f64)> = GL8
                        .iter()
                        .map(|&(t, w)| {
                            let x = sx * (lo + (hi - lo) * (t + 1.0) / 2.0);
                            (x, w * (-0.5 * ((x - m_prev) / sdp).powi(2)).exp())
                        })
                        .collect();
                    let tw: f64 = nodes.iter().map(|x| x.1).sum();
                    if tw <= 0.0 {
                        continue;
                    }
                    nodes.iter_mut().for_each(|x| x.1 /= tw);
                    // emax from the accumulator, or from the terms above it.
                    let p_le = all_le(st, bx);
                    if p_le > 1e-12 {
                        push_bs(
                            Some(bx),
                            sx,
                            &nodes,
                            bx,
                            true,
                            px * p_le,
                            &mut cfgs,
                            &mut wts,
                        );
                    }
                    for (i, &e) in levels.iter().enumerate() {
                        if e > bx {
                            let p = pe(i);
                            if p > 1e-12 {
                                push_bs(
                                    Some(bx),
                                    sx,
                                    &nodes,
                                    e,
                                    false,
                                    px * p,
                                    &mut cfgs,
                                    &mut wts,
                                );
                            }
                        }
                    }
                }
            }
        }
        // Configurations below 1e-7 of the step's weight are dropped (renormalized).
        let tw0: f64 = wts.iter().sum::<f64>().max(1e-300);
        let keep: Vec<usize> = (0..cfgs.len()).filter(|&i| wts[i] > 1e-7 * tw0).collect();
        let cfgs: Vec<super::tier1::Cfg> = keep.iter().map(|&i| cfgs[i]).collect();
        let wts: Vec<f64> = keep.iter().map(|&i| wts[i]).collect();
        let tw: f64 = wts.iter().sum::<f64>().max(1e-300);
        let mo = super::tier1::local_moments(sch, k, &cfgs, false, threads);
        let (mut lm, mut l2, mut l3, mut pm, mut p2, mut pz, mut ppz) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for ((cf, w), m) in cfgs.iter().zip(&wts).zip(&mo) {
            let (w, u) = (w / tw, ulpe(cf.bs));
            lm += w * m[0] * u;
            l2 += w * m[1] * u * u;
            l3 += w * m[2] * u * u * u;
            pm += w * m[4] * u;
            p2 += w * m[5] * u * u;
            pz += w * m[3];
            ppz += w * m[6];
        }
        c.mean += lm;
        c.var += (l2 - lm * lm).max(0.0);
        c.abs3 += l3;
        if k + 1 == ns {
            c.pre_mean = c.mean - lm + pm;
            c.pre_var = (c.var - (l2 - lm * lm)).max(0.0) + (p2 - pm * pm).max(0.0);
            c.pre_p_nz = 1.0 - (1.0 - c.p_nz) * (1.0 - ppz.min(1.0));
            c.y_mean = m_now;
            c.y_var = v_now;
        }
        c.p_nz = 1.0 - (1.0 - c.p_nz) * (1.0 - pz.min(1.0));
        m_prev = m_now;
        v_prev = v_now;
    }
    // Positions of y on its grids at its most likely binade.
    let sd = c.y_var.max(1e-300).sqrt();
    let bmode = binades(c.y_mean, sd)
        .max_by(|&a, &b| {
            let pa = binade_prob(c.y_mean, sd, a, 1.0) + binade_prob(c.y_mean, sd, a, -1.0);
            let pb = binade_prob(c.y_mean, sd, b, 1.0) + binade_prob(c.y_mean, sd, b, -1.0);
            pa.partial_cmp(&pb).unwrap()
        })
        .unwrap_or(0);
    c.tv_pos32 = super::tier1::sum_position_tv(sch, super::markov::ulp_exp(bmode));
    c.tv_pos16 = super::tier1::sum_position_tv(sch, bmode - 7);
    c
}

/// Sum over y's (sign, binade) cells of P(cell) f(sign, ulp_f32, ulp_out).
fn over_y(c: &Clt, out: Format, f: impl Fn(f64, f64, f64) -> f64) -> f64 {
    let sd = c.y_var.max(1e-300).sqrt();
    let mut acc = 0.0;
    for b in binades(c.y_mean, sd) {
        for s in [1.0f64, -1.0] {
            let p = binade_prob(c.y_mean, sd, b, s);
            if p < 1e-300 {
                continue;
            }
            let u32_ = 2f64.powi(super::markov::ulp_exp(b));
            let uo = 2f64.powi(b.max(out.emin()) - out.mbits());
            acc += p * f(s, u32_, uo);
        }
    }
    acc
}

/// Visit y's binades and signs as `over_y` does: g(weight, sign, ulp32, ulp_out).
fn each_y(c: &Clt, out: Format, mut g: impl FnMut(f64, f64, f64, f64)) {
    let sd = c.y_var.max(1e-300).sqrt();
    for b in binades(c.y_mean, sd) {
        for s in [1.0f64, -1.0] {
            let p = binade_prob(c.y_mean, sd, b, s);
            if p < 1e-300 {
                continue;
            }
            let u32_ = 2f64.powi(super::markov::ulp_exp(b));
            let uo = 2f64.powi(b.max(out.emin()) - out.mbits());
            g(p, s, u32_, uo);
        }
    }
}

pub(super) fn moments(mean: f64, var: f64) -> Moments {
    let sd = var.max(0.0).sqrt();
    Moments {
        mean,
        var,
        std: sd,
        mean_abs: sd * (2.0 / std::f64::consts::PI).sqrt(),
        p_zero: 0.0,
        quantiles: super::stats::QUANTILES
            .iter()
            .map(|&p| (p, mean + sd * inv_phi(p)))
            .collect(),
        max_abs: None,
    }
}

/// Error / (u S) with S at its mean (S = sum of K |terms| concentrates).
pub(super) fn u_scale(m: f64, v: f64, s: f64) -> Moments {
    let k = 1.0 / (2f64.powi(-24) * s.max(1e-300));
    moments(m * k, v * k * k)
}

fn inv_phi(p: f64) -> f64 {
    let (mut lo, mut hi) = (-40.0, 40.0);
    for _ in 0..200 {
        let m = 0.5 * (lo + hi);
        if phi(m) < p {
            lo = m;
        } else {
            hi = m;
        }
    }
    0.5 * (lo + hi)
}

fn bar(cs: &[&Clt]) -> ErrorBar {
    let be: Vec<String> = cs
        .iter()
        .map(|c| format!("{:.3}", 0.56 * c.abs3 / c.var.max(1e-300).powf(1.5)))
        .collect();
    ErrorBar {
        kind: "berry_esseen".into(),
        cdf_sup: Some(cs.iter().map(|c| 0.56 * c.abs3 / c.var.max(1e-300).powf(1.5)).fold(0.0, f64::max).min(1.0)),
        notes: vec![
            format!("Berry-Esseen sup |F - Phi| <= {} (per kernel; against the modeled local errors, not the hardware)", be.join(", ")),
            "residue-exact local laws (tier 1's step law, accumulator residue from the previous terms), normal partial sums, emax independent of them, steps independent (N-ERRDIST-CLT)".into(),
        ],
        ..Default::default()
    }
}

/// E[f(X)] for X = 0 with probability 1 - p, else N(m_c, s_c^2), where (m_c, s_c) match
/// the overall mean m and variance v (the zero-inflated normal of a sparse error sum).
fn zi_expect(p: f64, m: f64, v: f64, f: impl Fn(f64) -> f64) -> f64 {
    let p = p.clamp(0.0, 1.0);
    if p <= 1e-300 {
        return f(0.0);
    }
    let mc = m / p;
    let vc = ((v + m * m) / p - mc * mc).max(0.0);
    let sc = vc.sqrt();
    // f's structure lives in |x| <= 2 (ulp units): a fine grid there, a coarse one outside.
    let normal = if sc <= 0.0 {
        f(mc)
    } else {
        let (lo, hi) = (mc - 8.0 * sc, mc + 8.0 * sc);
        let dens = |x: f64| {
            (-0.5 * ((x - mc) / sc).powi(2)).exp() / (sc * (2.0 * std::f64::consts::PI).sqrt())
        };
        let simpson = |a: f64, b: f64, h: f64| -> f64 {
            if b <= a {
                return 0.0;
            }
            let n = (((b - a) / h).ceil() as usize).clamp(2, 1 << 16);
            let n = n + (n & 1);
            let h = (b - a) / n as f64;
            let mut acc = 0.0;
            for i in 0..=n {
                let x = a + i as f64 * h;
                let w = if i == 0 || i == n {
                    1.0
                } else if i % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                acc += w * f(x) * dens(x);
            }
            acc * h / 3.0
        };
        let (c0, c1) = ((-2.0f64).clamp(lo, hi), 2.0f64.clamp(lo, hi));
        let hc = sc / 16.0;
        let hf = hc.min(1.0 / 256.0);
        simpson(lo, c0, hc) + simpson(c0, c1, hf) + simpson(c1, hi, hc)
    };
    (1.0 - p) * f(0.0) + p * normal
}

/// Applicability of the CLT flip formulas: y's position uniform on the grid they use.
fn check(est: &mut Estimate, cs: &[&Clt], sixteen: bool) {
    let tv = cs
        .iter()
        .map(|c| if sixteen { c.tv_pos16 } else { c.tv_pos32 })
        .fold(0.0, f64::max);
    let ok = tv <= 0.05;
    est.applicability.push(format!(
        "y's position in its {} ulp: TV to uniform {tv:.3} (needs <= 0.05) {}",
        if sixteen { "16-bit" } else { "f32" },
        if ok { "ok" } else { "FAILS" }
    ));
    let be = cs
        .iter()
        .map(|c| 0.56 * c.abs3 / c.var.max(1e-300).powf(1.5))
        .fold(0.0, f64::max);
    let sparse = cs.iter().map(|c| c.p_nz).fold(1.0, f64::min);
    est.applicability.push(format!(
        "error sum: P(error != 0) {sparse:.3}, Berry-Esseen bound {be:.3} on its normal part (informational)"
    ));
    est.applicable = Some(ok);
}

/// Tier 2 for one kernel.
// ASSUME(N-ERRDIST-CLT): normal partial sums, independent steps, residue-exact local laws.
pub fn single(sch: &ChainSchema, out_fmt: Format, threads: usize) -> Estimate {
    let c = chain_exact(sch, threads);
    let mut est = Estimate {
        tier: 2,
        method: "CLT: normal partial sums, residue-exact local error moments (tier 1 step laws), steps independent; zero-inflated normal error".into(),
        outputs: 1,
        abs: moments(c.mean, c.var),
        u_scale: Some(u_scale(c.mean, c.var, c.s_mean)),
        error_bar: bar(&[&c]),
        ..Default::default()
    };
    let sixteen = sch.round16.is_some() && out_fmt != Format::F32;
    let p = match sch.round16.filter(|_| out_fmt != Format::F32) {
        // Misrounding of the 16-bit output: E[min(1, |e| / ulp16(y))].
        Some(f) => over_y(&c, f, |_, _, uo| {
            zi_expect(c.p_nz, c.mean / uo, c.var / (uo * uo), |x| x.abs().min(1.0))
        }),
        // f32: RZ(y + e_pre) differs from RN(y) with probability E[min(1, |c - 1/2|)], c the
        // pre-final error toward |y| in ulps (y's position in its ulp uniform).
        None => over_y(&c, Format::F32, |s, u, _| {
            zi_expect(c.pre_p_nz, s * c.pre_mean / u, c.pre_var / (u * u), |x| {
                (x - 0.5).abs().min(1.0)
            })
        }),
    };
    // pmf_ulp: the same zero-inflated normal through the tent kernel (y's position uniform).
    let mut tp = TentPmf::default();
    match sch.round16.filter(|_| out_fmt != Format::F32) {
        Some(f) => each_y(&c, f, |w, _, _, uo| {
            tp.zi(c.p_nz, c.mean / uo, c.var / (uo * uo), 0.0, w, 1.0)
        }),
        None => each_y(&c, Format::F32, |w, s, u, _| {
            tp.zi(
                c.pre_p_nz,
                s * c.pre_mean / u,
                c.pre_var / (u * u),
                0.5,
                w,
                s,
            )
        }),
    }
    let (pmf, nn) = tp.finish(Some(p));
    est.pmf_ulp = pmf;
    est.error_bar.notes.extend(nn);
    check(&mut est, &[&c], sixteen);
    est.p_differ = Some((p, p, p));
    est
}

/// Tier 2 for a pair: D = delta_C - delta_R with independent deviations.
pub fn pair(r: &ChainSchema, c: &ChainSchema, out_fmt: Format, threads: usize) -> Estimate {
    let (a, b) = (chain_exact(r, threads), chain_exact(c, threads));
    let (dm, dv) = (b.pre_mean - a.pre_mean, a.pre_var + b.pre_var);
    let dp = 1.0 - (1.0 - a.pre_p_nz) * (1.0 - b.pre_p_nz);
    let (fm, fv) = (b.mean - a.mean, a.var + b.var);
    let fp = 1.0 - (1.0 - a.p_nz) * (1.0 - b.p_nz);
    let mut est = Estimate {
        tier: 2,
        method: "CLT for both kernels with residue-exact local moments; D = delta_C - delta_R zero-inflated normal, the deviations independent (N-ERRDIST-PAIR)".into(),
        outputs: 1,
        abs: moments(dm, dv),
        u_scale: Some(u_scale(dm, dv, a.s_mean)),
        error_bar: bar(&[&a, &b]),
        pair: Some(PairMoments {
            mean_r: a.mean,
            var_r: a.var,
            mean_c: b.mean,
            var_c: b.var,
            cov_rc: 0.0,
            corr_rc: 0.0,
        }),
        ..Default::default()
    };
    let r16 = r.round16.or(c.round16).filter(|_| out_fmt != Format::F32);
    let p = match r16 {
        Some(f) => {
            let flip = |uo: f64| zi_expect(fp, fm / uo, fv / (uo * uo), |x| x.abs().min(1.0));
            let p = over_y(&a, f, |_, _, uo| flip(uo));
            // C16 - R16 in 16-bit ulps: D / ulp16 through the tent kernel; moments in output
            // ulps from it, absolute from E[ulp16(y)^2 P(flip | y)].
            let mut tp = TentPmf::default();
            each_y(&a, f, |w, _, _, uo| {
                tp.zi(fp, fm / uo, fv / (uo * uo), 0.0, w, 1.0)
            });
            let (pmf, nn) = tp.finish(Some(p));
            est.error_bar.notes.extend(nn);
            est.ulp = Some(super::combine::ulp_moments(
                &est,
                &super::combine::Law::from(pmf.iter().map(|x| (x.0 as f64, x.1)).collect()),
            ));
            est.pmf_ulp = pmf;
            let v = over_y(&a, f, |_, _, uo| uo * uo * flip(uo));
            let ma = over_y(&a, f, |_, _, uo| uo * flip(uo));
            est.abs = Moments {
                mean: 0.0,
                var: v,
                std: v.sqrt(),
                mean_abs: ma,
                p_zero: 1.0 - p,
                quantiles: vec![],
                max_abs: None,
            };
            est.error_bar.notes.push(format!("f32 difference before the {f:?} rounding: zero-inflated normal, P(!= 0) {fp:.3e}, mean {fm:.4e}, std {:.4e} (absolute)", fv.sqrt()));
            p
        }
        None => {
            let p = over_y(&a, Format::F32, |_, u, _| {
                zi_expect(dp, dm / u, dv / (u * u), |x| x.abs().min(1.0))
            });
            let mut tp = TentPmf::default();
            each_y(&a, Format::F32, |w, _, u, _| {
                tp.zi(dp, dm / u, dv / (u * u), 0.0, w, 1.0)
            });
            let (pmf, nn) = tp.finish(Some(p));
            est.error_bar.notes.extend(nn);
            est.ulp = Some(super::combine::ulp_moments(
                &est,
                &super::combine::Law::from(pmf.iter().map(|x| (x.0 as f64, x.1)).collect()),
            ));
            est.pmf_ulp = pmf;
            p
        }
    };
    check(&mut est, &[&a, &b], r16.is_some());
    est.p_differ = Some((p, p, p));
    est
}

// ------------------------------------------------------------------ DAG chains (tier 2)
//
// Add / FMA / MMA DAGs over input leaves (reductions, split-K, atomics, FMA GEMMs, TF32
// MMA chains with f32 inputs): every node's rounding error and every product's
// input-quantization error (TF32 splits) as independent terms of known mean / variance,
// summed linearly (each node's error reaches the output with coefficient 1). Lattice-aware:
// a node is exact when the lattice of its operands (lowest set bit, a law over exponents)
// reaches its ulp.

use super::chain::Op;
use super::chain::Prog;
use super::laws::Pmf;

/// Exponent range of the lattice laws.
pub(super) const LMIN: i32 = -400;
pub(super) const LN: usize = 700;

/// A leaf buffer's law: values as read, and whether their low bits are random (f32 inputs,
/// `laws::f32_smooth`) or exactly the PMF's.
#[derive(Clone, Debug)]
pub struct LeafLaw {
    pub pmf: Pmf,
    pub smooth: bool,
}

/// Leaf laws for every input buffer the program reads.
pub fn leaf_laws(
    spec: &super::spec::Spec,
    fmts: &std::collections::BTreeMap<String, Format>,
) -> Result<std::collections::BTreeMap<String, LeafLaw>, String> {
    let mut out = std::collections::BTreeMap::new();
    let mut notes = vec![];
    for (name, fmt) in fmts {
        let t = spec
            .tensor(name)
            .ok_or_else(|| format!("no distribution for {name}"))?;
        let law = if *fmt == Format::F32 && matches!(t.quant, super::spec::Quant::Round) {
            LeafLaw {
                pmf: super::laws::f32_smooth(&super::laws::effective_dist(t, &mut notes)),
                smooth: true,
            }
        } else {
            match super::laws::tensor_law(t, *fmt, None, &mut notes)? {
                super::laws::TensorLaw::Iid(p) => LeafLaw {
                    pmf: p,
                    smooth: false,
                },
                _ => return Err(format!("{name}: block-scaled inputs outside an MMA chain")),
            }
        };
        if std::env::var("PTXV_T2DAG_DEBUG").is_ok() {
            eprintln!(
                "T2DAG leaf {name} {fmt:?} smooth {} atoms {} first {:?}",
                law.smooth,
                law.pmf.v.len(),
                &law.pmf.v[..law.pmf.v.len().min(3)]
            );
        }
        out.insert(name.clone(), law);
    }
    Ok(out)
}

/// P(lattice >= L) over L = LMIN.. (index L - LMIN).
type Lat = Rc<Vec<f64>>;

pub(super) fn lat_index(l: i32) -> usize {
    (l - LMIN).clamp(0, LN as i32 - 1) as usize
}

fn lat_of_law(l: &LeafLaw) -> Lat {
    let mut s = vec![0.0; LN];
    let tot: f64 = l.pmf.p.iter().sum::<f64>().max(1e-300);
    for (&v, &p) in l.pmf.v.iter().zip(&l.pmf.p) {
        let p = p / tot;
        if v == 0.0 {
            s.iter_mut().for_each(|x| *x += p);
            continue;
        }
        if l.smooth {
            let b = super::spec::exp_floor(v);
            let base = b - 23;
            for (i, x) in s.iter_mut().enumerate() {
                let li = i as i32 + LMIN;
                if li <= base {
                    *x += p;
                } else if li <= b {
                    *x += p * 2f64.powi(-(li - base));
                }
            }
        } else {
            let lb = super::fine::lowest_bit(v);
            for x in s.iter_mut().take(lat_index(lb) + 1) {
                *x += p;
            }
        }
    }
    Rc::new(s)
}

/// min of independent lattices: S = S_a S_b.
fn lat_min(a: &Lat, b: &Lat) -> Lat {
    Rc::new(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
}

/// Sum of independent lattice exponents (products).
fn lat_sum(a: &Lat, b: &Lat) -> Lat {
    let pa: Vec<f64> = (0..LN)
        .map(|i| a[i] - if i + 1 < LN { a[i + 1] } else { 0.0 })
        .collect();
    let pb: Vec<f64> = (0..LN)
        .map(|i| b[i] - if i + 1 < LN { b[i + 1] } else { 0.0 })
        .collect();
    let mut p = vec![0.0; LN];
    for (i, &x) in pa.iter().enumerate() {
        if x <= 1e-18 {
            continue;
        }
        for (j, &y) in pb.iter().enumerate() {
            if y <= 1e-18 {
                continue;
            }
            let l = (i as i32 + LMIN) + (j as i32 + LMIN);
            p[lat_index(l)] += x * y;
        }
    }
    // Zero values (S at the top) stay "no bit": keep their mass at the top index.
    let mut s = vec![0.0; LN];
    let mut acc = 0.0;
    for i in (0..LN).rev() {
        acc += p[i];
        s[i] = acc;
    }
    Rc::new(s)
}

/// Value moments of a node: mean, variance, E sum |terms|, and its lattice.
#[derive(Clone)]
struct Ns {
    m: f64,
    v: f64,
    s: f64,
    lat: Lat,
    /// The lattice of the node's exact value (no roundings).
    lx: Lat,
}

/// A factor of a product: its leaf, and which TF32 part of it (None: the value itself).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Fk {
    Plain,
    Hi,
    Lo,
    LoR,
    LoT,
}

/// The DAG summary: the output's error (sum of independent node / product errors).
/// A node error's law within one binade (and sign) of the node's value x: P(x there) and the
/// error's moments given x there.
#[derive(Clone, Copy, Debug)]
pub struct Part {
    pub b: i32,
    pub s: f64,
    pub p: f64,
    pub e1: f64,
    pub e2: f64,
    pub pn: f64,
    /// The rounding's ulp exponent, the law of the residue's grid bits k (index 1..=6;
    /// 7: finer / continuous; the rest: exact), and extra continuous variance (MMA term
    /// truncation).
    pub ue: i32,
    pub kw: [f64; 8],
    pub xv: f64,
}

/// One independent error term of the output: a node's rounding (with its value's law, for
/// conditioning on y) or a product group's TF32 split error (with its covariance with y).
#[derive(Clone, Debug, Default)]
pub struct NodeErr {
    pub h: u64,
    pub nm: f64,
    pub nv: f64,
    pub parts: Vec<Part>,
    pub mean: f64,
    pub m2: f64,
    pub pn: f64,
    pub cov_y: f64,
    pub rz: bool,
    /// The error's mean as a linear function of the node's value x: E[e | x] ~ beta x (a
    /// round toward zero: -sign(x) ulp(x) / 2 per binade). Partial sums share their terms, so
    /// these means are correlated across nodes (`DagClt::bias`).
    pub beta: f64,
    pub node: Option<u32>,
}

/// The DAG summary: the output's error as a sum of independent node / product errors.
#[derive(Clone, Debug, Default)]
pub struct DagClt {
    pub mean: f64,
    pub var: f64,
    pub p_nz: f64,
    pub y_mean: f64,
    pub y_var: f64,
    pub s_mean: f64,
    /// The final operation rounds toward zero (an MMA step) rather than to nearest.
    pub final_rz: bool,
    /// The output is rounded to this 16-bit format after the f32 value.
    pub round16: Option<Format>,
    /// P(y exactly on its f32 grid) (the flip formulas assume a uniform position).
    pub p_on_grid: f64,
    /// y's lattice before its own rounding, and for an MMA root the alignment (largest term
    /// exponent estimate, frac bits): the truncated sum sits on the 2^(max(b, et) - frac) grid.
    pub ylat: Option<Lat>,
    pub mma_q: Option<(i32, i32)>,
    /// The DAG has MMA steps / TF32 splits.
    pub has_mma: bool,
    pub has_tf32: bool,
    pub terms: Vec<NodeErr>,
    /// The output's own rounding term (its last node).
    pub root_h: Option<u64>,
    /// Per product term (canonical hashes of its two leaves): (c, v, y): the coefficient of
    /// the term in B = sum over nodes of beta x (the nodes' value-proportional error means),
    /// its variance, and its coefficient in y. Empty with TF32 splits.
    pub bias: Vec<BiasTerm>,
    /// y's excess kurtosis from the product terms (0 when not computed: TF32 splits).
    pub y_kurt: f64,
}

/// A product term of `DagClt::bias`: (its leaves' canonical hashes, (c, v, y)).
pub type BiasTerm = ((u64, u64), (f64, f64, f64));

/// Node terms conditioned on y individually: those with the largest share of y's variance.
const COND_NODES: usize = 4096;

/// The extra variance of the error given y from the correlation of the nodes' value-
/// proportional means (`DagClt::bias`): Var(B | y) for B = sum_g c_g t_g (pair: c_C - c_R),
/// less what the per-node laws already count (sum beta^2 Var(x | y) over the nodes `keep`
/// admits). Products independent; y = sum_g y_g t_g.
pub fn bias_extra(r: &DagClt, c: Option<&DagClt>, keep: &dyn Fn(u64) -> bool) -> f64 {
    let mut m: std::collections::BTreeMap<(u64, u64), (f64, f64, f64)> = Default::default();
    let sr = if c.is_some() { -1.0 } else { 1.0 };
    for (k, (cc, v, y)) in &r.bias {
        let e = m.entry(*k).or_insert((0.0, *v, *y));
        e.0 += sr * cc;
    }
    if let Some(c) = c {
        for (k, (cc, v, y)) in &c.bias {
            let e = m.entry(*k).or_insert((0.0, *v, *y));
            e.0 += cc;
        }
    }
    let (mut s2, mut sy, mut yy) = (0.0, 0.0, 0.0);
    for (cc, v, y) in m.values() {
        s2 += v * cc * cc;
        sy += v * cc * y;
        yy += v * y * y;
    }
    let var_b = (s2 - if yy > 0.0 { sy * sy / yy } else { 0.0 }).max(0.0);
    let mut already = 0.0;
    for d in std::iter::once(r).chain(c) {
        for t in d.terms.iter().filter(|t| keep(t.h)) {
            let rr = (t.nv / d.y_var.max(1e-300)).min(1.0);
            already += t.beta * t.beta * t.nv * (1.0 - rr);
        }
    }
    if std::env::var("PTXV_T2DAG_DEBUG").is_ok() {
        eprintln!(
            "T2DAG bias: Var(B | y) {var_b:.3e}, per-node {already:.3e}, terms {}",
            m.len()
        );
    }
    (var_b - already).max(0.0)
}

impl DagClt {
    /// Indices of the terms conditioned on y (the rest enter unconditionally).
    fn strong(&self) -> Vec<bool> {
        let mut r: Vec<(f64, usize)> = self
            .terms
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.parts.is_empty() && t.nv >= 1e-4 * self.y_var)
            .map(|(i, t)| (t.nv, i))
            .collect();
        r.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut out = vec![false; self.terms.len()];
        for &(_, i) in r.iter().take(COND_NODES) {
            out[i] = true;
        }
        out
    }

    /// Error (mean, variance, P(!= 0)) given y, over the terms `keep` admits; `pre` leaves out
    /// the output's own rounding.
    fn cond(
        &self,
        y: f64,
        strong: &[bool],
        keep: &dyn Fn(u64) -> bool,
        pre: bool,
    ) -> (f64, f64, f64) {
        let (mut m, mut v, mut lq) = (0.0, 0.0, 0.0);
        for (k, t) in self.terms.iter().enumerate() {
            if !keep(t.h) || (pre && Some(t.h) == self.root_h) {
                continue;
            }
            let (e1, e2, pn) = if strong[k] {
                // x | y ~ N(nm + (nv / vy)(y - my), nv (1 - nv / vy)) (x a sub-sum of y).
                let r = (t.nv / self.y_var.max(1e-300)).min(1.0);
                let mu = t.nm + r * (y - self.y_mean);
                let sd = (t.nv * (1.0 - r)).max(0.0).sqrt();
                let (mut w, mut a1, mut a2, mut ap) = (0.0, 0.0, 0.0, 0.0);
                for q in &t.parts {
                    let (lo, hi) = (2f64.powi(q.b), 2f64.powi(q.b + 1));
                    let pw = if sd <= 1e-12 * mu.abs().max(1e-300) {
                        let x = mu * q.s;
                        if x >= lo && x < hi { 1.0 } else { 0.0 }
                    } else if q.s > 0.0 {
                        phi((hi - mu) / sd) - phi((lo - mu) / sd)
                    } else {
                        phi((-lo - mu) / sd) - phi((-hi - mu) / sd)
                    };
                    w += pw;
                    a1 += pw * q.e1;
                    a2 += pw * q.e2;
                    ap += pw * q.pn;
                }
                if w > 1e-12 {
                    (a1 / w, a2 / w, ap / w)
                } else {
                    (t.mean, t.m2, t.pn)
                }
            } else if t.cov_y != 0.0 {
                let b = t.cov_y / self.y_var.max(1e-300);
                (
                    t.mean + b * (y - self.y_mean),
                    t.m2 - b * t.cov_y + (b * (y - self.y_mean)).powi(2),
                    t.pn,
                )
            } else {
                (t.mean, t.m2, t.pn)
            };
            m += e1;
            v += (e2 - e1 * e1).max(0.0);
            lq += (1.0 - pn.clamp(0.0, 1.0 - 1e-16)).ln();
        }
        (m, v, 1.0 - lq.exp())
    }
}

/// The output's error given y: a discrete law on the grid h (the strong nodes' residues,
/// convolved) plus an independent zero-inflated normal (the rest): 0 w.p. 1 - gp, else
/// N(gm, gv).
#[derive(Clone, Debug)]
pub struct Law {
    pub h: f64,
    pub off: i64,
    pub p: Vec<f64>,
    pub gm: f64,
    pub gv: f64,
    pub gp: f64,
}

/// Bins per ulp of y in the discrete error law; its largest size.
pub(super) const LAW_G: f64 = 16.0;
pub(super) const LAW_MAX: usize = 1 << 12;
/// Node errors kept discrete (the largest given y).
pub(super) const LAW_NODES: usize = 16;

impl Law {
    /// Atoms (value, P) with P > 1e-14.
    pub(super) fn atoms(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.p
            .iter()
            .enumerate()
            .filter(|x| *x.1 > 1e-14)
            .map(|(i, &q)| ((i as i64 + self.off) as f64 * self.h, q))
    }
}

/// Convolve a sparse law (bin, P) into a dense one.
pub(super) fn law_conv(acc: &mut (i64, Vec<f64>), add: &[(i64, f64)]) {
    if add.is_empty() {
        return;
    }
    let (lo, hi) = (
        add.iter().map(|x| x.0).min().unwrap(),
        add.iter().map(|x| x.0).max().unwrap(),
    );
    let mut out = vec![0.0; acc.1.len() + (hi - lo) as usize];
    for (i, &a) in acc.1.iter().enumerate() {
        if a <= 1e-16 {
            continue;
        }
        for &(j, b) in add {
            out[i + (j - lo) as usize] += a * b;
        }
    }
    acc.0 += lo;
    acc.1 = out;
    // Trim negligible ends.
    let first = acc.1.iter().position(|&x| x > 1e-15).unwrap_or(0);
    let last = acc.1.iter().rposition(|&x| x > 1e-15).unwrap_or(0);
    if first > 0 || last + 1 < acc.1.len() {
        acc.1 = acc.1[first..=last.max(first)].to_vec();
        acc.0 += first as i64;
    }
}

/// A rounding residue's error law at ulp u (grid bits k; 7 = continuous): (value / u, P).
pub(super) fn residue_atoms(k: usize, rz: bool, s: f64) -> Vec<(f64, f64)> {
    let (n, mid) = if k >= 7 {
        (32usize, 0.5)
    } else {
        (1usize << k, 0.0)
    };
    let w = 1.0 / n as f64;
    let mut out = Vec::with_capacity(n + 1);
    for j in 0..n {
        let f = (j as f64 + mid) / n as f64;
        if rz {
            out.push((-s * f, w));
        } else {
            // e in [-1/2, 1/2): j - n/2 over n; the tie -1/2 goes either way.
            let e = f - 0.5;
            if k < 7 && j == 0 {
                out.push((-0.5, w / 2.0));
                out.push((0.5, w / 2.0));
            } else {
                out.push((e, w));
            }
        }
    }
    out
}

impl DagClt {
    /// The error law given y in binade b.
    fn law(&self, y: f64, b: i32, strong: &[bool], keep: &dyn Fn(u64) -> bool, pre: bool) -> Law {
        let uy = 2f64.powi(super::markov::ulp_exp(b));
        let h = uy / LAW_G;
        let mut acc: (i64, Vec<f64>) = (0, vec![1.0]);
        let (mut gm, mut gv, mut lq) = (0.0, 0.0, 0.0);
        let gauss = |m: f64, v: f64, pn: f64, gm: &mut f64, gv: &mut f64, lq: &mut f64| {
            *gm += m;
            *gv += v.max(0.0);
            *lq += (1.0 - pn.clamp(0.0, 1.0 - 1e-16)).ln();
        };
        // (E e^2 given y, term, its parts' weights given y, their sum, E e, P(e != 0)).
        type Cand = (f64, usize, Vec<(usize, f64)>, f64, f64, f64);
        let mut cands: Vec<Cand> = vec![];
        for (k, t) in self.terms.iter().enumerate() {
            if !keep(t.h) || (pre && Some(t.h) == self.root_h) {
                continue;
            }
            if !strong[k] {
                let (e1, e2) = if t.cov_y != 0.0 {
                    let bb = t.cov_y / self.y_var.max(1e-300);
                    (
                        t.mean + bb * (y - self.y_mean),
                        t.m2 - bb * t.cov_y + (bb * (y - self.y_mean)).powi(2),
                    )
                } else {
                    (t.mean, t.m2)
                };
                gauss(e1, e2 - e1 * e1, t.pn, &mut gm, &mut gv, &mut lq);
                continue;
            }
            // x | y ~ N(nm + (nv / vy)(y - my), nv (1 - nv / vy)) (x a sub-sum of y).
            let r = (t.nv / self.y_var.max(1e-300)).min(1.0);
            let mu = t.nm + r * (y - self.y_mean);
            let sd = (t.nv * (1.0 - r)).max(0.0).sqrt();
            let mut ws: Vec<(usize, f64)> = vec![];
            let mut wt = 0.0;
            for (qi, q) in t.parts.iter().enumerate() {
                let (lo, hi) = (2f64.powi(q.b), 2f64.powi(q.b + 1));
                let pw = if sd <= 1e-12 * mu.abs().max(1e-300) {
                    let x = mu * q.s;
                    if x >= lo && x < hi { 1.0 } else { 0.0 }
                } else if q.s > 0.0 {
                    phi((hi - mu) / sd) - phi((lo - mu) / sd)
                } else {
                    phi((-lo - mu) / sd) - phi((-hi - mu) / sd)
                };
                if pw > 1e-12 {
                    ws.push((qi, pw));
                    wt += pw;
                }
            }
            if wt <= 1e-12 {
                gauss(
                    t.mean,
                    t.m2 - t.mean * t.mean,
                    t.pn,
                    &mut gm,
                    &mut gv,
                    &mut lq,
                );
                continue;
            }
            let (mut c1, mut c2, mut cp) = (0.0, 0.0, 0.0);
            for &(qi, pw) in &ws {
                let q = &t.parts[qi];
                c1 += pw / wt * q.e1;
                c2 += pw / wt * q.e2;
                cp += pw / wt * q.pn;
            }
            cands.push((c2, k, ws, wt, c1, cp));
        }
        // The largest node errors given y are discrete; the rest join the normal part.
        cands.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (ci, (c2, k, ws, wt, c1, cp)) in cands.into_iter().enumerate() {
            let t = &self.terms[k];
            if ci >= LAW_NODES {
                gauss(c1, c2 - c1 * c1, cp, &mut gm, &mut gv, &mut lq);
                continue;
            }
            // Parts at ulps of at least 2 bins: discrete; finer ones join the normal part.
            let mut sparse: std::collections::BTreeMap<i64, f64> = Default::default();
            let (mut cm, mut cv, mut cpn, mut cw) = (0.0, 0.0, 0.0, 0.0);
            for &(qi, pw) in &ws {
                let q = &t.parts[qi];
                let w = pw / wt;
                let u = 2f64.powi(q.ue);
                if u >= 2.0 * h {
                    let mut kz = 1.0;
                    for (kk, &kwv) in q.kw.iter().enumerate().skip(1) {
                        if kwv <= 0.0 {
                            continue;
                        }
                        kz -= kwv;
                        for (v, pa) in residue_atoms(kk, t.rz, q.s) {
                            *sparse.entry((v * u / h).round() as i64).or_insert(0.0) +=
                                w * kwv * pa;
                        }
                    }
                    *sparse.entry(0).or_insert(0.0) += w * kz.max(0.0);
                    gv += w * q.xv;
                } else {
                    *sparse.entry(0).or_insert(0.0) += w;
                    cm += w * q.e1;
                    cv += w * q.e2;
                    cpn += w * q.pn;
                    cw += w;
                }
            }
            if cw > 0.0 {
                gauss(cm, cv - cm * cm, cpn, &mut gm, &mut gv, &mut lq);
            }
            let add: Vec<(i64, f64)> = sparse.into_iter().filter(|x| x.1 > 1e-15).collect();
            let span = add.iter().map(|x| x.0).max().unwrap_or(0)
                - add.iter().map(|x| x.0).min().unwrap_or(0);
            if acc.1.len() + span as usize > LAW_MAX {
                let m = add.iter().map(|x| x.0 as f64 * h * x.1).sum::<f64>();
                let m2 = add
                    .iter()
                    .map(|x| (x.0 as f64 * h).powi(2) * x.1)
                    .sum::<f64>();
                let p0 = add.iter().filter(|x| x.0 == 0).map(|x| x.1).sum::<f64>();
                gauss(m, m2 - m * m, 1.0 - p0, &mut gm, &mut gv, &mut lq);
            } else {
                law_conv(&mut acc, &add);
            }
        }
        Law {
            h,
            off: acc.0,
            p: acc.1,
            gm,
            gv,
            gp: 1.0 - lq.exp(),
        }
    }
}

/// A law with extra variance in its normal part (the zero-inflation kept; P(!= 0) raised to
/// 1 when the part was empty).
fn with_var(mut l: Law, dv: f64) -> Law {
    if dv <= 0.0 {
        return l;
    }
    if l.gp > 1e-9 {
        l.gv += dv / l.gp;
    } else {
        l.gp = 1.0;
        l.gm = 0.0;
        l.gv = dv;
    }
    l
}

/// Mean and variance of a law.
pub(super) fn law_moments(l: &Law) -> (f64, f64) {
    let m0: f64 = l.atoms().map(|(x, q)| x * q).sum();
    let m2: f64 = l.atoms().map(|(x, q)| x * x * q).sum();
    let gm2 = l.gv + l.gm * l.gm;
    let m = m0 + l.gp * l.gm;
    (
        m,
        (m2 + 2.0 * m0 * l.gp * l.gm + l.gp * gm2 - m * m).max(0.0),
    )
}

/// P(error != 0) of a law.
pub(super) fn law_pnz(l: &Law) -> f64 {
    let p0: f64 = l.atoms().filter(|x| x.0 == 0.0).map(|x| x.1).sum();
    1.0 - p0 * (1.0 - l.gp)
}

/// Integrate g(y, s, b) P(y in binade b with sign s) over y ~ N(m, v), with y at the
/// binade's quarter points; g also gets each point's weight.
pub(super) fn over_y_pts_w(m: f64, v: f64, mut g: impl FnMut(f64, f64, i32, f64) -> f64) -> f64 {
    let sd = v.max(1e-300).sqrt();
    let mut acc = 0.0;
    for b in binades(m, sd) {
        for s in [1.0f64, -1.0] {
            let p = binade_prob(m, sd, b, s);
            if p < 1e-10 {
                continue;
            }
            let mut a = 0.0;
            for f in [1.125, 1.375, 1.625, 1.875] {
                a += g(s * f * 2f64.powi(b), s, b, p / 4.0);
            }
            acc += p * a / 4.0;
        }
    }
    acc
}

/// Rounding error moments, in ulp units, of a value whose residue mod ulp is uniform on the
/// grid of n = 2^k points (k >= 1): (E e, E e^2, P(e != 0)) for a positive value.
pub(super) fn grid_moments(k: i32, rz: bool) -> (f64, f64, f64) {
    if k > 40 {
        return if rz {
            (-0.5, 1.0 / 3.0, 1.0)
        } else {
            (0.0, 1.0 / 12.0, 1.0)
        };
    }
    let n = 2f64.powi(k);
    let pn = 1.0 - 1.0 / n;
    if rz {
        // e = -j/n, j uniform on 0..n.
        (
            -(n - 1.0) / (2.0 * n),
            (n - 1.0) * (2.0 * n - 1.0) / (6.0 * n * n),
            pn,
        )
    } else {
        // e = j/n for j in -n/2..n/2 (the tie j = -n/2 has |e| = 1/2, mean 0 over even/odd).
        let sq = |m: f64| m * (m + 1.0) * (2.0 * m + 1.0) / 6.0;
        let h = n / 2.0;
        (0.0, (sq(h) + sq(h - 1.0)) / (n * n * n), pn)
    }
}

/// Rounding of a value N(m, v) whose lattice law is `lat` (P(value a multiple of 2^L)) at
/// ulp 2^ulp_of(binade), the residue uniform on the lattice: the error per binade of the
/// value, and the rounded value's lattice (raised to the ulp).
fn round_moments(
    m: f64,
    v: f64,
    lat: &Lat,
    rz: bool,
    ulp_of: impl Fn(i32) -> i32,
) -> (Vec<Part>, Lat) {
    let sd = v.max(1e-300).sqrt();
    let mut parts = vec![];
    // P(ulp exponent <= L) for the output lattice.
    let mut ue_cdf = vec![0.0; LN];
    let mut ptot = 0.0;
    for b in binades(m, sd) {
        for s in [1.0f64, -1.0] {
            let p = binade_prob(m, sd, b, s);
            if p < 1e-300 {
                continue;
            }
            ptot += p;
            let ue = ulp_of(b);
            ue_cdf[lat_index(ue)] += p;
            let u = 2f64.powi(ue);
            let iu = lat_index(ue);
            let (mut e1, mut e2, mut pn) = (0.0, 0.0, 0.0);
            let mut kw = [0.0; 8];
            // P(lat = L) for L < ue: k = ue - L grid bits.
            for l in (0..iu).rev() {
                let pl = lat[l] - lat[l + 1];
                if pl <= 1e-300 {
                    continue;
                }
                let (gm, g2, gp) = grid_moments((iu - l) as i32, rz);
                kw[(iu - l).min(7)] += pl;
                e1 += pl * s * gm * u;
                e2 += pl * g2 * u * u;
                pn += pl * gp;
            }
            // Lattice mass below the table (index 0): fully random.
            let (gm, g2, gp) = grid_moments(99, rz);
            let p0 = 1.0 - lat[0];
            kw[7] += p0;
            e1 += p0 * s * gm * u;
            e2 += p0 * g2 * u * u;
            pn += p0 * gp;
            parts.push(Part {
                b,
                s,
                p,
                e1,
                e2,
                pn,
                ue,
                kw,
                xv: 0.0,
            });
        }
    }
    let mut out = vec![0.0; LN];
    let mut c = 0.0;
    for i in 0..LN {
        // L <= ue: always a multiple; L > ue: only if the value was (exactly representable).
        out[i] = (1.0 - c) * 1.0 + c * lat[i];
        out[i] = out[i].max(lat[i]).min(1.0);
        c += ue_cdf[i] / ptot.max(1e-300);
    }
    (parts, Rc::new(out))
}

/// A node term from its parts.
fn node_err(h: u64, nm: f64, nv: f64, mut parts: Vec<Part>, rz: bool) -> NodeErr {
    parts.retain(|q| q.p > 1e-10);
    let pt: f64 = parts.iter().map(|q| q.p).sum::<f64>().max(1e-300);
    let mean = parts.iter().map(|q| q.p * q.e1).sum::<f64>() / pt;
    // beta = Cov(E[e | x], x) / Var(x), E[e | x] constant per (binade, sign) part.
    let sd = nv.max(0.0).sqrt();
    let beta = if sd > 0.0 {
        let exs: f64 = parts
            .iter()
            .map(|q| {
                let (lo, hi) = if q.s > 0.0 {
                    (2f64.powi(q.b), 2f64.powi(q.b + 1))
                } else {
                    (-(2f64.powi(q.b + 1)), -(2f64.powi(q.b)))
                };
                let (a, b) = ((lo - nm) / sd, (hi - nm) / sd);
                // E[x; lo < x < hi] for x ~ N(nm, nv).
                let ex = nm * (phi(b) - phi(a))
                    + sd * (super::special::phi_pdf(a) - super::special::phi_pdf(b));
                q.e1 * ex
            })
            .sum::<f64>()
            / pt;
        (exs - mean * nm) / nv
    } else {
        0.0
    };
    let m2 = parts.iter().map(|q| q.p * q.e2).sum::<f64>() / pt;
    let pn = parts.iter().map(|q| q.p * q.pn).sum::<f64>() / pt;
    NodeErr {
        h,
        nm,
        nv,
        parts,
        mean,
        m2,
        pn,
        cov_y: 0.0,
        rz,
        beta,
        node: None,
    }
}

/// Moments of a leaf's TF32 noise given its law: (E x^2, E x l, E l^2, E x d, E l d, E d^2,
/// E l, E d, E x) with l = lo(x), d = loR(x) - lo(x) (or loT - lo).
fn tf32_noise(l: &LeafLaw, trunc_lo: bool) -> [f64; 9] {
    let tot: f64 = l.pmf.p.iter().sum::<f64>().max(1e-300);
    let mut r = [0.0; 9];
    for (&v, &p) in l.pmf.v.iter().zip(&l.pmf.p) {
        let p = p / tot;
        if v == 0.0 {
            continue;
        }
        let b = super::spec::exp_floor(v);
        let s = v.signum();
        // lo(x): uniform magnitude on [0, 2^(b-10)) (random low bits), sign of x.
        let w = 2f64.powi(b - 10);
        let (el, el2) = (s * w / 2.0, w * w / 3.0);
        // d = TF32(lo) - lo: lo (the low 13 bits of x) has its top bit at b - 11 - j with
        // P = 2^-(j+1); TF32 keeps 11 bits, so j = 0 drops 2 bits, j = 1 one bit, j >= 2
        // none. LoR (+0x1000 then truncate): rounds half away from zero; LoT truncates.
        let u0 = 2f64.powi(b - 21);
        let (ed, ed2) = if trunc_lo {
            (-s * u0 * 7.0 / 32.0, u0 * u0 * 15.0 / 128.0)
        } else {
            (s * u0 * 3.0 / 32.0, u0 * u0 * 7.0 / 128.0)
        };
        r[0] += p * v * v;
        r[1] += p * v * el;
        r[2] += p * el2;
        r[3] += p * v * ed;
        r[4] += p * el * ed;
        r[5] += p * ed2;
        r[6] += p * el;
        r[7] += p * ed;
        r[8] += p * v;
    }
    r
}

/// Coefficients of a factor F(x) in the basis (x, l, d): Plain x, Hi x - l, Lo l, LoR/LoT
/// l + d.
fn fk_coef(k: Fk) -> [f64; 3] {
    match k {
        Fk::Plain => [1.0, 0.0, 0.0],
        Fk::Hi => [1.0, -1.0, 0.0],
        Fk::Lo => [0.0, 1.0, 0.0],
        Fk::LoR | Fk::LoT => [0.0, 1.0, 1.0],
    }
}

/// Tier 2 summary of output `root`'s DAG. Err when the DAG is not a signed sum of products
/// of inputs.
// ASSUME(N-ERRDIST-CLT): node values normal with independent leaves, node errors
// independent, f32 inputs with random low bits.
pub fn dag(
    p: &Prog,
    root: u32,
    laws: &std::collections::BTreeMap<String, LeafLaw>,
) -> Result<DagClt, String> {
    // The cone, children first.
    let mut cone: Vec<u32> = vec![];
    {
        let mut seen = std::collections::HashSet::new();
        let mut st = vec![root];
        while let Some(i) = st.pop() {
            if !seen.insert(i) {
                continue;
            }
            cone.push(i);
            match &p.ops[i as usize] {
                Op::Leaf { .. } | Op::Const(_) => {}
                Op::Nl(n) => st.extend(n.args()),
                Op::Round { arg, .. } | Op::Tf32 { arg, .. } => st.push(*arg),
                Op::Add { a, b, .. } => st.extend([*a, *b]),
                Op::Fma { a, b, c, .. } => {
                    st.extend([*a, *b]);
                    st.extend(c.iter().copied());
                }
                Op::Mma {
                    acc, terms, scales, ..
                } => {
                    st.extend(acc.iter().copied());
                    for &(x, y) in terms.iter().chain(scales) {
                        st.extend([x, y]);
                    }
                }
            }
            if cone.len() > 2_000_000 {
                return Err("DAG too large".into());
            }
        }
        cone.sort();
    }
    let leaf_law = |i: u32| -> Result<&LeafLaw, String> {
        match &p.ops[i as usize] {
            Op::Leaf { buf, .. } => laws
                .get(&p.bufs[*buf as usize])
                .ok_or_else(|| format!("no law for {}", p.bufs[*buf as usize])),
            _ => Err("not a leaf".into()),
        }
    };
    // Resolve a product factor to (leaf or const, kind).
    let factor = |mut i: u32| -> Result<(u32, Fk), String> {
        let mut k = Fk::Plain;
        loop {
            match &p.ops[i as usize] {
                Op::Leaf { .. } | Op::Const(_) => return Ok((i, k)),
                Op::Round { arg, .. } => i = *arg,
                Op::Tf32 { arg, part } => {
                    if k != Fk::Plain {
                        return Err("nested TF32 parts".into());
                    }
                    k = match part {
                        crate::num::Tf32Part::Hi => Fk::Hi,
                        crate::num::Tf32Part::Lo => Fk::Lo,
                        crate::num::Tf32Part::LoR => Fk::LoR,
                        crate::num::Tf32Part::LoT => Fk::LoT,
                    };
                    i = *arg;
                }
                _ => return Err("a product of computed values".into()),
            }
        }
    };
    let mut memo = HashMap::new();
    let mut ns: HashMap<u32, Ns> = HashMap::new();
    let mut out = DagClt::default();
    let ulp32 = super::markov::ulp_exp;
    // Product groups: (leaf a, leaf b) -> list of (kind a, kind b, sign).
    type Group = Vec<(Fk, Fk, f64)>;
    let mut groups: std::collections::BTreeMap<(u32, u32), Group> = Default::default();
    let leaf_ns = |i: u32| -> Result<Ns, String> {
        match &p.ops[i as usize] {
            Op::Const(c) => {
                let l: Lat = Rc::new(
                    (0..LN)
                        .map(|k| {
                            if c.to_bits() == 0 || (k as i32 + LMIN) <= super::fine::lowest_bit(*c)
                            {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect(),
                );
                Ok(Ns {
                    m: *c,
                    v: 0.0,
                    s: c.abs(),
                    lat: l.clone(),
                    lx: l,
                })
            }
            _ => {
                let l = leaf_law(i)?;
                let (m, m2) = l.pmf.moments();
                let tot: f64 = l.pmf.p.iter().sum::<f64>().max(1e-300);
                let ea: f64 = l
                    .pmf
                    .v
                    .iter()
                    .zip(&l.pmf.p)
                    .map(|(v, p)| v.abs() * p)
                    .sum::<f64>()
                    / tot;
                let lt = lat_of_law(l);
                Ok(Ns {
                    m,
                    v: (m2 - m * m).max(0.0),
                    s: ea,
                    lat: lt.clone(),
                    lx: lt,
                })
            }
        }
    };
    let mut ns_leaf: HashMap<u32, Ns> = HashMap::new();
    let get_leaf = |i: u32, ns_leaf: &mut HashMap<u32, Ns>| -> Result<Ns, String> {
        if let Some(x) = ns_leaf.get(&i) {
            return Ok(x.clone());
        }
        let x = leaf_ns(i)?;
        ns_leaf.insert(i, x.clone());
        Ok(x)
    };
    let dbg = std::env::var("PTXV_T2DAG_DEBUG").is_ok();
    let mut pending: Vec<NodeErr> = vec![];
    let mut root_term: Option<u64> = None;
    let rootv = match &p.ops[root as usize] {
        Op::Round { arg, .. } => *arg,
        _ => root,
    };
    let mut root_pre: Option<Lat> = None;
    // Product terms: (entry node, leaf a, leaf b, sign, variance); TF32 parts: none.
    let mut prods: Vec<(u32, u32, u32, f64, f64)> = vec![];
    let mut prods_tf32 = false;
    if dbg {
        eprintln!(
            "T2DAG root {:?} arg {:?}",
            p.ops[root as usize], p.ops[rootv as usize]
        );
    }
    for &i in &cone {
        let me = match &p.ops[i as usize] {
            Op::Leaf { .. } | Op::Const(_) => get_leaf(i, &mut ns_leaf)?,
            Op::Nl(_) => {
                return Err("a non-linear op (slice, MUFU, max): the nonlin tiers' domain".into());
            }
            Op::Round { arg, f16 } => {
                let a = ns[arg].clone();
                if i == root {
                    out.round16 = Some(if *f16 { Format::F16 } else { Format::Bf16 });
                    a
                } else {
                    // An intermediate 16-bit rounding: an error of its own.
                    let f = if *f16 { Format::F16 } else { Format::Bf16 };
                    let (parts, l) =
                        round_moments(a.m, a.v, &a.lat, false, |b| b.max(f.emin()) - f.mbits());
                    pending.push(node_err(
                        super::tier0::canon(p, i, &mut memo),
                        a.m,
                        a.v,
                        parts,
                        false,
                    ));
                    pending.last_mut().unwrap().node = Some(i);
                    Ns { lat: l, ..a }
                }
            }
            Op::Tf32 { arg, part } => {
                out.has_tf32 = true;
                // Standalone TF32 parts are product factors; as values only hi(x) ~ x.
                let a = ns[arg].clone();
                if *part != crate::num::Tf32Part::Hi {
                    Ns {
                        m: 0.0,
                        v: 0.0,
                        s: 0.0,
                        lat: a.lat.clone(),
                        lx: Rc::new(vec![1.0; LN]),
                    }
                } else {
                    a
                }
            }
            Op::Add { a, b, rz } => {
                let (x, y) = (&ns[a], &ns[b]);
                let r = Ns {
                    m: x.m + y.m,
                    v: x.v + y.v,
                    s: x.s + y.s,
                    lat: lat_min(&x.lat, &y.lat),
                    lx: lat_min(&x.lx, &y.lx),
                };
                if i == rootv {
                    root_pre = Some(r.lat.clone());
                }
                let (parts, l) = round_moments(r.m, r.v, &r.lat, *rz, ulp32);
                let h = super::tier0::canon(p, i, &mut memo);
                pending.push(node_err(h, r.m, r.v, parts, *rz));
                pending.last_mut().unwrap().node = Some(i);
                if i == rootv {
                    root_term = Some(h);
                }
                Ns { lat: l, ..r }
            }
            Op::Fma {
                a, b, c, neg_ab, ..
            } => {
                let (fa, ka) = factor(*a)?;
                let (fb, kb) = factor(*b)?;
                let (x, y) = (get_leaf(fa, &mut ns_leaf)?, get_leaf(fb, &mut ns_leaf)?);
                let pm = x.m * y.m;
                let pv = ((x.v + x.m * x.m) * (y.v + y.m * y.m) - pm * pm).max(0.0);
                let sgn = if *neg_ab { -1.0 } else { 1.0 };
                if ka == Fk::Plain && kb == Fk::Plain {
                    prods.push((i, fa, fb, sgn, pv));
                } else {
                    prods_tf32 = true;
                }
                groups
                    .entry((fa.min(fb), fa.max(fb)))
                    .or_default()
                    .push(if fa <= fb {
                        (ka, kb, sgn)
                    } else {
                        (kb, ka, sgn)
                    });
                let is_lo = |k: Fk| matches!(k, Fk::Lo | Fk::LoR | Fk::LoT);
                let lxp = if is_lo(ka) || is_lo(kb) {
                    Rc::new(vec![1.0; LN])
                } else {
                    lat_sum(&x.lx, &y.lx)
                };
                let mut r = Ns {
                    m: sgn * pm,
                    v: pv,
                    s: x.s * y.s,
                    lat: lat_sum(&x.lat, &y.lat),
                    lx: lxp,
                };
                if let Some(c) = c {
                    let z = &ns[c];
                    r = Ns {
                        m: r.m + z.m,
                        v: r.v + z.v,
                        s: r.s + z.s,
                        lat: lat_min(&r.lat, &z.lat),
                        lx: lat_min(&r.lx, &z.lx),
                    };
                }
                if i == rootv {
                    root_pre = Some(r.lat.clone());
                }
                let (parts, l) = round_moments(r.m, r.v, &r.lat, false, ulp32);
                let h = super::tier0::canon(p, i, &mut memo);
                pending.push(node_err(h, r.m, r.v, parts, false));
                pending.last_mut().unwrap().node = Some(i);
                if i == rootv {
                    root_term = Some(h);
                }
                Ns { lat: l, ..r }
            }
            Op::Mma {
                model,
                acc,
                terms,
                scales,
                ..
            } => {
                if !scales.is_empty() {
                    return Err("block-scaled MMA in a DAG chain".into());
                }
                out.has_mma = true;
                let frac = match model {
                    super::chain::StepModel::Mma(m) => {
                        m.frac_bits.ok_or("an exact MMA model")? as i32
                    }
                    _ => return Err("narrow MMA in a DAG chain".into()),
                };
                let mut r = match acc {
                    Some(a) => ns[a].clone(),
                    None => Ns {
                        m: 0.0,
                        v: 0.0,
                        s: 0.0,
                        lat: Rc::new(vec![1.0; LN]),
                        lx: Rc::new(vec![1.0; LN]),
                    },
                };
                let (mut tm, mut tv, mut ts, mut tlat) = (0.0, 0.0, 0.0, Rc::new(vec![1.0; LN]));
                // Alignment exponent law: the largest of the terms' exponent fields
                // (e_a + e_b) and the accumulator's exponent, as a CDF over E.
                let mut ecdf: std::collections::BTreeMap<i32, f64> = Default::default();
                let mut tlaws: Vec<Vec<(i32, f64)>> = vec![];
                let mut tlx: Lat = Rc::new(vec![1.0; LN]);
                let mut tq: Vec<(f64, f64, Lat)> = vec![];
                for &(a, b) in terms {
                    let (fa, ka) = factor(a)?;
                    let (fb, kb) = factor(b)?;
                    let (x, y) = (get_leaf(fa, &mut ns_leaf)?, get_leaf(fb, &mut ns_leaf)?);
                    groups
                        .entry((fa.min(fb), fa.max(fb)))
                        .or_default()
                        .push(if fa <= fb {
                            (ka, kb, 1.0)
                        } else {
                            (kb, ka, 1.0)
                        });
                    // The used value's magnitude scale: lo parts are 2^-11 of x.
                    let sc = |k: Fk| {
                        if matches!(k, Fk::Lo | Fk::LoR | Fk::LoT) {
                            2f64.powi(-11)
                        } else {
                            1.0
                        }
                    };
                    let (mx, my) = (x.m * sc(ka), y.m * sc(kb));
                    let (vx, vy) = (
                        (x.v + x.m * x.m) * sc(ka).powi(2),
                        (y.v + y.m * y.m) * sc(kb).powi(2),
                    );
                    let pm = if matches!(ka, Fk::Lo | Fk::LoR | Fk::LoT)
                        || matches!(kb, Fk::Lo | Fk::LoR | Fk::LoT)
                    {
                        0.0
                    } else {
                        mx * my
                    };
                    let pv = (vx * vy - mx * mx * my * my).max(0.0);
                    if ka == Fk::Plain && kb == Fk::Plain {
                        prods.push((i, fa, fb, 1.0, pv));
                    } else {
                        prods_tf32 = true;
                    }
                    tm += pm;
                    tv += pv;
                    ts += x.s * y.s * sc(ka) * sc(kb);
                    {
                        let ex = |f: u32, k: Fk| -> Vec<(i32, f64)> {
                            let sh = if matches!(k, Fk::Lo | Fk::LoR | Fk::LoT) {
                                -11
                            } else {
                                0
                            };
                            match &p.ops[f as usize] {
                                Op::Const(c) => {
                                    if *c == 0.0 {
                                        vec![]
                                    } else {
                                        vec![(super::spec::exp_floor(*c) + sh, 1.0)]
                                    }
                                }
                                _ => {
                                    let l = leaf_law(f).unwrap();
                                    let tot: f64 = l.pmf.p.iter().sum::<f64>().max(1e-300);
                                    let mut m: std::collections::BTreeMap<i32, f64> =
                                        Default::default();
                                    for (&v, &q) in l.pmf.v.iter().zip(&l.pmf.p) {
                                        if v != 0.0 {
                                            *m.entry(super::spec::exp_floor(v) + sh)
                                                .or_insert(0.0) += q / tot;
                                        }
                                    }
                                    m.into_iter().collect()
                                }
                            }
                        };
                        let (xa, xb) = (ex(fa, ka), ex(fb, kb));
                        let mut m: std::collections::BTreeMap<i32, f64> = Default::default();
                        for &(e1, p1) in &xa {
                            for &(e2, p2) in &xb {
                                *m.entry(e1 + e2).or_insert(0.0) += p1 * p2;
                            }
                        }
                        tlaws.push(m.into_iter().filter(|x| x.1 > 1e-12).collect());
                    }
                    let lta = if matches!(ka, Fk::Plain) {
                        x.lat.clone()
                    } else {
                        tf32_lat(&x.lat, ka)
                    };
                    let ltb = if matches!(kb, Fk::Plain) {
                        y.lat.clone()
                    } else {
                        tf32_lat(&y.lat, kb)
                    };
                    let l = lat_sum(&lta, &ltb);
                    tlat = lat_min(&tlat, &l);
                    if !matches!(ka, Fk::Lo | Fk::LoR | Fk::LoT)
                        && !matches!(kb, Fk::Lo | Fk::LoR | Fk::LoT)
                    {
                        tlx = lat_min(&tlx, &lat_sum(&x.lx, &y.lx));
                    }
                    tq.push((pm, pv, l));
                }
                r = Ns {
                    m: r.m + tm,
                    v: r.v + tv,
                    s: r.s + ts,
                    lat: lat_min(&r.lat, &tlat),
                    lx: lat_min(&r.lx, &tlx),
                };
                // Truncation of each term at Q = 2^(b_S - frac) toward zero, and the result's
                // round toward zero (emax taken as the result's binade).
                let sd = r.v.max(1e-300).sqrt();
                let mut parts = vec![];
                // The alignment exponent: the accumulator's or the largest term's (exponent
                // fields: up to one binade below the product's), whichever is larger.
                // The accumulator's exponent law.
                let acc_law: Vec<(i32, f64)> = match acc {
                    Some(a) => {
                        let an = &ns[a];
                        let asd = an.v.max(1e-300).sqrt();
                        if an.v <= 0.0 && an.m == 0.0 {
                            vec![]
                        } else {
                            binades(an.m, asd)
                                .map(|b| {
                                    (
                                        b,
                                        binade_prob(an.m, asd, b, 1.0)
                                            + binade_prob(an.m, asd, b, -1.0),
                                    )
                                })
                                .filter(|x| x.1 > 1e-12)
                                .collect()
                        }
                    }
                    None => vec![],
                };
                let (elo, ehi) = tlaws
                    .iter()
                    .chain(std::iter::once(&acc_law))
                    .flat_map(|l| l.iter().map(|x| x.0))
                    .fold((i32::MAX, i32::MIN), |a, e| (a.0.min(e), a.1.max(e)));
                if elo <= ehi {
                    for e in elo..=ehi {
                        let mut c = 1.0;
                        for l in tlaws.iter().chain(std::iter::once(&acc_law)) {
                            let tot: f64 = l.iter().map(|x| x.1).sum();
                            let below: f64 =
                                l.iter().filter(|x| x.0 <= e).map(|x| x.1).sum::<f64>()
                                    + (1.0 - tot).max(0.0);
                            c *= below.min(1.0);
                        }
                        ecdf.insert(e, c);
                    }
                }
                let mut epmf: Vec<(i32, f64)> = vec![];
                let mut prev = 0.0;
                for (&e, &c) in &ecdf {
                    if c - prev > 1e-9 {
                        epmf.push((e, c - prev));
                    }
                    prev = c;
                }
                let et = epmf.iter().map(|x| x.0 as f64 * x.1).sum::<f64>().round() as i32;
                for bb in binades(r.m, sd) {
                    for s in [1.0f64, -1.0] {
                        let pr = binade_prob(r.m, sd, bb, s);
                        if pr < 1e-300 {
                            continue;
                        }
                        // E given the result's binade: the sum of n terms at most 2^(E + 2) each
                        // stays below 2^(E + 2 + log2 n); the result cannot exceed it.
                        let cap = bb - 2 - ((tq.len() + 1) as f64).log2().ceil() as i32;
                        // The accumulator's exponent given the result in this binade (the
                        // accumulator is a sub-sum of the result): E = max(the terms'
                        // exponents, the accumulator's).
                        let epmf_c: Vec<(i32, f64)> = match acc {
                            Some(a) if ns[a].v > 0.0 && r.v > 0.0 => {
                                let an = &ns[a];
                                let xr = s * 1.5 * 2f64.powi(bb);
                                let rr = (an.v / r.v).min(1.0);
                                let mu = an.m + rr * (xr - r.m);
                                let asd = (an.v * (1.0 - rr)).max(0.0).sqrt();
                                let acdf = |e: i32| -> f64 {
                                    if asd <= 1e-12 * mu.abs().max(1e-300) {
                                        return if mu == 0.0 || super::spec::exp_floor(mu) <= e {
                                            1.0
                                        } else {
                                            0.0
                                        };
                                    }
                                    let t = 2f64.powi(e + 1);
                                    (phi((t - mu) / asd) - phi((-t - mu) / asd)).clamp(0.0, 1.0)
                                };
                                let tcdf = |e: i32| -> f64 {
                                    let mut c = 1.0;
                                    for l in &tlaws {
                                        let tot: f64 = l.iter().map(|x| x.1).sum();
                                        let below: f64 =
                                            l.iter().filter(|x| x.0 <= e).map(|x| x.1).sum::<f64>()
                                                + (1.0 - tot).max(0.0);
                                        c *= below.min(1.0);
                                    }
                                    c
                                };
                                let mut out = vec![];
                                let mut prev = tcdf(elo - 1) * acdf(elo - 1);
                                for e in elo..=ehi.max(bb + 1) {
                                    let c = tcdf(e) * acdf(e);
                                    if c - prev > 1e-9 {
                                        out.push((e, c - prev));
                                    }
                                    prev = c;
                                }
                                out
                            }
                            _ => epmf.clone(),
                        };
                        let mut ew: Vec<(i32, f64)> =
                            epmf_c.iter().copied().filter(|x| x.0 >= cap).collect();
                        if ew.is_empty() {
                            ew = vec![(bb, 1.0)];
                        }
                        let wsum: f64 = ew.iter().map(|x| x.1).sum();
                        let ue = ulp32(bb);
                        let u = 2f64.powi(ue);
                        let fine = 1.0 - r.lat[lat_index(ue)].max(0.0);
                        let (mut e1, mut e2, mut pn, mut xv) = (0.0, 0.0, 0.0, 0.0);
                        let mut kw = [0.0; 8];
                        for (e, we) in ew {
                            let we = we / wsum;
                            let q = e - frac;
                            let qq = 2f64.powi(q);
                            let mut te2 = 0.0;
                            let mut texact = 1.0;
                            for (_, _, l) in &tq {
                                let below = 1.0 - l[lat_index(q)];
                                te2 += below * qq * qq / 3.0;
                                texact *= 1.0 - below;
                            }
                            // After truncation the sum sits on the grid of its lattice L (at
                            // least Q): k = ue - L bits below the ulp, the residue uniform on
                            // that grid (none when L >= ue).
                            let (mut gm, mut g2, mut gp) = (0.0, 0.0, 0.0);
                            if ue > q {
                                for k in 1..=(ue - q).min(60) {
                                    let lv = ue - k;
                                    let pk = if lv == q {
                                        1.0 - r.lat[lat_index(q + 1)]
                                    } else {
                                        r.lat[lat_index(lv)] - r.lat[lat_index(lv + 1)]
                                    }
                                    .max(0.0);
                                    if pk <= 1e-15 {
                                        continue;
                                    }
                                    let (a, b2, c) = grid_moments(k, true);
                                    gm += pk * a;
                                    g2 += pk * b2;
                                    gp += pk * c;
                                    kw[(k as usize).min(7)] += we * pk;
                                }
                            }
                            let _ = fine;
                            e1 += we * s * gm * u;
                            e2 += we * (te2 + g2 * u * u);
                            pn += we * (1.0 - texact * (1.0 - gp));
                            xv += we * te2;
                        }
                        parts.push(Part {
                            b: bb,
                            s,
                            p: pr,
                            e1,
                            e2,
                            pn,
                            ue,
                            kw,
                            xv,
                        });
                    }
                }
                let h = super::tier0::canon(p, i, &mut memo);
                pending.push(node_err(h, r.m, r.v, parts, true));
                pending.last_mut().unwrap().node = Some(i);
                if i == rootv {
                    root_term = Some(h);
                    out.mma_q = Some((et, frac));
                }
                out.final_rz = true;
                if i == rootv {
                    root_pre = Some(r.lat.clone());
                }
                let l = round_moments(r.m, r.v, &r.lat, true, ulp32).1;
                Ns { lat: l, ..r }
            }
        };
        if !matches!(p.ops[i as usize], Op::Mma { .. }) && i == root {
            out.final_rz = false;
        }
        ns.insert(i, me);
    }
    // The nodes' value-proportional error means, B = sum_j beta_j x_j = sum_g c_g t_g over
    // the product terms t_g: c_g sums beta over the nodes whose value contains t_g (its entry
    // node and every node above it), y_g its coefficient in y.
    if !prods_tf32 && !prods.is_empty() {
        let mut parents: HashMap<u32, Vec<u32>> = HashMap::new();
        for &i in &cone {
            let kids: Vec<u32> = match &p.ops[i as usize] {
                Op::Round { arg, .. } => vec![*arg],
                Op::Tf32 { arg, part } if *part == crate::num::Tf32Part::Hi => vec![*arg],
                Op::Add { a, b, .. } => vec![*a, *b],
                Op::Fma { c: Some(c), .. } => vec![*c],
                Op::Mma { acc: Some(a), .. } => vec![*a],
                _ => vec![],
            };
            for k in kids {
                parents.entry(k).or_default().push(i);
            }
        }
        let beta: HashMap<u32, f64> = pending
            .iter()
            .filter_map(|t| t.node.map(|n| (n, t.beta)))
            .collect();
        let (mut a_of, mut y_of): (HashMap<u32, f64>, HashMap<u32, f64>) =
            (HashMap::new(), HashMap::new());
        for &i in cone.iter().rev() {
            let mut a = beta.get(&i).copied().unwrap_or(0.0);
            let mut yc = if i == rootv { 1.0 } else { 0.0 };
            for par in parents.get(&i).map(|v| v.as_slice()).unwrap_or(&[]) {
                a += a_of.get(par).copied().unwrap_or(0.0);
                yc += y_of.get(par).copied().unwrap_or(0.0);
            }
            a_of.insert(i, a);
            y_of.insert(i, yc);
        }
        let mut bias: std::collections::BTreeMap<(u64, u64), (f64, f64, f64)> = Default::default();
        for &(e, fa, fb, sg, pv) in &prods {
            let (ha, hb) = (
                super::tier0::canon(p, fa, &mut memo),
                super::tier0::canon(p, fb, &mut memo),
            );
            let k = (ha.min(hb), ha.max(hb));
            let x = bias.entry(k).or_insert((0.0, pv, 0.0));
            x.0 += sg * a_of.get(&e).copied().unwrap_or(0.0);
            x.2 += sg * y_of.get(&e).copied().unwrap_or(0.0);
        }
        out.bias = bias.into_iter().collect();
        // y's excess kurtosis from the product terms (independent products, y = sum y_g
        // t_g): the applicability check's measure of how far y and its partial sums are
        // from normal.
        let raw = |i: u32| -> Result<[f64; 4], String> {
            match &p.ops[i as usize] {
                Op::Const(c) => Ok([*c, c * c, c.powi(3), c.powi(4)]),
                _ => {
                    let l = leaf_law(i)?;
                    let t = l.pmf.total().max(1e-300);
                    let mut r = [0.0; 4];
                    for (v, q) in l.pmf.v.iter().zip(&l.pmf.p) {
                        for (k, x) in r.iter_mut().enumerate() {
                            *x += q * v.powi(k as i32 + 1) / t;
                        }
                    }
                    Ok(r)
                }
            }
        };
        let (mut k2, mut k4) = (0.0, 0.0);
        for &(e, fa, fb, sg, _) in &prods {
            let c = sg * y_of.get(&e).copied().unwrap_or(0.0);
            let (a, b) = (raw(fa)?, raw(fb)?);
            let r: Vec<f64> = (0..4).map(|k| a[k] * b[k]).collect();
            let mu2 = r[1] - r[0] * r[0];
            let mu4 = r[3] - 4.0 * r[2] * r[0] + 6.0 * r[1] * r[0] * r[0] - 3.0 * r[0].powi(4);
            k2 += c * c * mu2;
            k4 += c.powi(4) * (mu4 - 3.0 * mu2 * mu2);
        }
        out.y_kurt = if k2 > 0.0 { k4 / (k2 * k2) } else { 0.0 };
    }
    // Product groups: the input-quantization error of TF32 splits (exact products: 0).
    for ((a, b), ks) in &groups {
        if ks.iter().all(|k| k.0 == Fk::Plain && k.1 == Fk::Plain) {
            continue;
        }
        let (la, lb) = (leaf_law(*a)?, leaf_law(*b)?);
        let trunc = ks.iter().any(|k| k.0 == Fk::LoT || k.1 == Fk::LoT);
        let (na, nb) = (tf32_noise(la, trunc), tf32_noise(lb, trunc));
        // e = sum_k s F_a G_b - exact, bilinear in (x, l, d) x (y, l', d').
        let mut c = [[0.0f64; 3]; 3];
        let mut exact = 0.0;
        for &(ka, kb, s) in ks {
            let (ca, cb) = (fk_coef(ka), fk_coef(kb));
            for i in 0..3 {
                for j in 0..3 {
                    c[i][j] += s * ca[i] * cb[j];
                }
            }
            if !matches!(ka, Fk::Lo | Fk::LoR | Fk::LoT)
                && !matches!(kb, Fk::Lo | Fk::LoR | Fk::LoT)
            {
                exact += s;
            }
        }
        c[0][0] -= exact;
        // E[X X'] matrices on each side.
        let ma = [
            [na[0], na[1], na[3]],
            [na[1], na[2], na[4]],
            [na[3], na[4], na[5]],
        ];
        let mb = [
            [nb[0], nb[1], nb[3]],
            [nb[1], nb[2], nb[4]],
            [nb[3], nb[4], nb[5]],
        ];
        let (ea, eb) = ([na[8], na[6], na[7]], [nb[8], nb[6], nb[7]]);
        let (mut m1, mut m2) = (0.0, 0.0);
        for i in 0..3 {
            for j in 0..3 {
                m1 += c[i][j] * ea[i] * eb[j];
                for k in 0..3 {
                    for l in 0..3 {
                        m2 += c[i][j] * c[k][l] * ma[i][k] * mb[j][l];
                    }
                }
            }
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        use std::hash::Hash;
        use std::hash::Hasher;
        (
            7u8,
            super::tier0::canon(p, *a, &mut memo),
            super::tier0::canon(p, *b, &mut memo),
        )
            .hash(&mut h);
        for k in ks {
            (k.0, k.1, k.2.to_bits()).hash(&mut h);
        }
        // Covariance with the group's contribution to y (exact * a b): TF32 truncation errors
        // scale with the product.
        let mut cv = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                cv += c[i][j] * ma[0][i] * mb[0][j];
            }
        }
        cv = exact * (cv - m1 * ea[0] * eb[0]);
        let pz = if la.smooth || lb.smooth { 1.0 } else { 0.5 };
        pending.push(NodeErr {
            h: h.finish(),
            mean: m1,
            m2: m2.max(0.0),
            pn: pz,
            cov_y: cv,
            ..Default::default()
        });
    }
    let y = &ns[&rootv];
    let ylat = root_pre.unwrap_or_else(|| y.lat.clone());
    out.y_mean = y.m;
    out.y_var = y.v;
    out.s_mean = cone
        .iter()
        .filter_map(|i| ns.get(i))
        .map(|n| 0.0 * n.s)
        .sum::<f64>()
        + y.s;
    out.root_h = root_term;
    for t in &pending {
        if dbg {
            eprintln!(
                "T2DAG term {:016x} mean {:.3e} rms {:.3e} p {:.3} nv/vy {:.3} cov_y {:.3e}",
                t.h,
                t.mean,
                t.m2.sqrt(),
                t.pn,
                t.nv / y.v.max(1e-300),
                t.cov_y
            );
        }
        out.mean += t.mean;
        out.var += (t.m2 - t.mean * t.mean).max(0.0);
        out.p_nz = 1.0 - (1.0 - out.p_nz) * (1.0 - t.pn.clamp(0.0, 1.0));
    }
    out.terms = pending;
    // The reported variance counts the correlation of the nodes' value-proportional means.
    let bs: f64 = out.bias.iter().map(|(_, (c, v, _))| v * c * c).sum();
    let bn: f64 = out.terms.iter().map(|t| t.beta * t.beta * t.nv).sum();
    out.var += (bs - bn).max(0.0);
    let sd = y.v.max(1e-300).sqrt();
    // The position of the exact y within its ulp: its own lattice (the computed operands'
    // lattices are those of rounded values, not of y).
    out.ylat = Some(y.lx.clone());
    out.p_on_grid = binades(y.m, sd)
        .map(|b| {
            (binade_prob(y.m, sd, b, 1.0) + binade_prob(y.m, sd, b, -1.0))
                * ylat[lat_index(ulp32(b))]
        })
        .sum();
    Ok(out)
}

/// The lattice law of a TF32 part of x given x's lattice: hi keeps 11 significant bits
/// (lowest bit >= its binade - 10), lo parts sit 11 bits lower.
fn tf32_lat(x: &Lat, k: Fk) -> Lat {
    let _ = x;
    // Without x's binade law here, approximate: the part's lattice is x's lattice raised by
    // the truncation (hi) or lowered (lo).
    let shift = match k {
        Fk::Hi | Fk::Plain => 13,
        _ => -11,
    };
    Rc::new(
        (0..LN)
            .map(|i| {
                let j = i as i32 - shift;
                if j < 0 {
                    1.0
                } else if j as usize >= LN {
                    0.0
                } else {
                    x[j as usize]
                }
            })
            .collect(),
    )
}

impl DagClt {
    /// Law of y's position within its f32 ulp at binade b: (grid bits k, weight); k = 0: on
    /// the grid, k = 99: continuous.
    fn pos_law(&self, b: i32) -> Vec<(i32, f64)> {
        let ue = super::markov::ulp_exp(b);
        let Some(l) = &self.ylat else {
            return vec![(99, 1.0)];
        };
        let at = |k: i32| l[lat_index(ue - k)];
        let mut out: Vec<(i32, f64)> = vec![];
        let mut push = |k: i32, w: f64| {
            if w > 1e-15 {
                match out.iter_mut().find(|x| x.0 == k) {
                    Some(x) => x.1 += w,
                    None => out.push((k, w)),
                }
            }
        };
        push(0, at(0));
        for k in 1..=6 {
            push(k, at(k) - at(k - 1));
        }
        push(99, 1.0 - at(6));
        out
    }
}

/// Positions t in [0, 1) of a value on a 2^k grid within its ulp (k = 99: continuous), with
/// weights.
pub(super) fn positions(k: i32) -> Vec<(f64, f64)> {
    let n = if k >= 7 { 32 } else { 1usize << k };
    let off = if k >= 7 { 0.5 } else { 0.0 };
    (0..n)
        .map(|j| ((j as f64 + off) / n as f64, 1.0 / n as f64))
        .collect()
}

/// P(N(m, sd^2) in [a, b)).
pub(super) fn pn_in(m: f64, sd: f64, a: f64, b: f64) -> f64 {
    if sd <= 1e-300 {
        return if m >= a && m < b { 1.0 } else { 0.0 };
    }
    (phi((b - m) / sd) - phi((a - m) / sd)).max(0.0)
}

/// E[min(1, |Z|)] for Z ~ N(m, s^2).
pub(super) fn emin1(m: f64, s: f64) -> f64 {
    if s <= 1e-300 {
        return m.abs().min(1.0);
    }
    let seg = |a: f64, b: f64| {
        // E[Z; a < Z < b]
        let (al, be) = ((a - m) / s, (b - m) / s);
        m * (phi(be) - phi(al)) + s * (super::special::phi_pdf(al) - super::special::phi_pdf(be))
    };
    let pin = phi((1.0 - m) / s) - phi((-1.0 - m) / s);
    (1.0 - pin) + seg(0.0, 1.0) - seg(-1.0, 0.0)
}

/// E[min(1, |x0 + g - c|)] with g = 0 w.p. 1 - gp, else N(gm, gs^2).
fn emin1_zi(x0: f64, gp: f64, gm: f64, gs: f64, c: f64) -> f64 {
    (1.0 - gp) * (x0 - c).abs().min(1.0) + gp * emin1(x0 + gm - c, gs)
}

/// P(t + x in [a, b)) for x = x0 + (0 w.p. 1 - gp, else N(gm, gs^2)).
pub(super) fn p_in(t: f64, x0: f64, gp: f64, gm: f64, gs: f64, a: f64, b: f64) -> f64 {
    let z = t + x0;
    let point = if z >= a && z < b { 1.0 } else { 0.0 };
    (1.0 - gp) * point + gp * pn_in(z + gm, gs, a, b)
}

/// P(the computed f32 equals RN(y)) for y at position t in [0, 1) ulps above the grid point
/// toward zero and pre-rounding error x0 + g (ulps, toward |y|). RN: RN(t + x) vs RN(t);
/// RZ: floor(t + x) vs RN(t); a tie (t = 1/2) rounds to even, either way with P 1/2.
fn same_pos(t: f64, x0: f64, gp: f64, gm: f64, gs: f64, rz: bool) -> f64 {
    let tie = (t - 0.5).abs() < 1e-12;
    let cs: &[(f64, f64)] = if tie {
        &[(0.0, 0.5), (1.0, 0.5)]
    } else if t < 0.5 {
        &[(0.0, 1.0)]
    } else {
        &[(1.0, 1.0)]
    };
    let mut acc = 0.0;
    for &(c, w) in cs {
        let (a, b) = if rz {
            (c, c + 1.0)
        } else if tie {
            // From the tie, RN(1/2 + x) is the even neighbour c for x in [-1, 0] (c = 0) or
            // [0, 1] (c = 1).
            if c == 0.0 {
                (-0.5, 0.5 + 1e-12)
            } else {
                (0.5, 1.5 + 1e-12)
            }
        } else {
            (c - 0.5, c + 0.5)
        };
        acc += w * p_in(t, x0, gp, gm, gs, a, b);
    }
    acc
}

/// P(differ) of one output from its error law given y.
fn flip_p(d: &DagClt, out16: Option<Format>, y: f64, b: i32, pre: &Law, full: &Law) -> f64 {
    let s = y.signum();
    match out16 {
        Some(f) => {
            // The f32 error flips the 16-bit rounding with P |e| / ulp16 (y uniform there).
            let uo = 2f64.powi(b.max(f.emin()) - f.mbits());
            let gs = full.gv.sqrt();
            full.atoms()
                .map(|(x, q)| {
                    let e = |m: f64| {
                        if gs <= 0.0 {
                            m.abs()
                        } else {
                            gs * (2.0 / std::f64::consts::PI).sqrt()
                                * (-0.5 * (m / gs).powi(2)).exp()
                                + m * (1.0 - 2.0 * phi(-m / gs))
                        }
                    };
                    q * (((1.0 - full.gp) * x.abs() + full.gp * e(x + full.gm)) / uo).min(1.0)
                })
                .sum()
        }
        None => {
            let u = 2f64.powi(super::markov::ulp_exp(b));
            let (gm, gs) = (s * pre.gm / u, pre.gv.max(0.0).sqrt() / u);
            let atoms: Vec<(f64, f64)> = pre.atoms().map(|(x, q)| (s * x / u, q)).collect();
            let mut same = 0.0;
            for (k, w) in d.pos_law(b) {
                // A continuous error spread of a third of an ulp washes out the position grid.
                if k >= 7 || gs >= 0.3 {
                    // Uniform position: P(differ | x) = min(1, |x - c|), c = 1/2 for RZ.
                    let c = if d.final_rz { 0.5 } else { 0.0 };
                    same += w
                        * (1.0
                            - atoms
                                .iter()
                                .map(|&(x0, q)| q * emin1_zi(x0, pre.gp, gm, gs, c))
                                .sum::<f64>());
                    continue;
                }
                for (t, wt) in positions(k) {
                    for &(x0, q) in &atoms {
                        same += w * wt * q * same_pos(t, x0, pre.gp, gm, gs, d.final_rz);
                    }
                }
            }
            (1.0 - same).clamp(0.0, 1.0)
        }
    }
}

/// Add a DAG error law (atoms plus, with probability gp, a normal) to a tent PMF: x scaled
/// to ulps, oriented, shifted; k recorded with `sign`.
fn tent_law(tp: &mut TentPmf, l: &Law, scale: f64, orient: f64, shift: f64, sign: f64, w: f64) {
    let (gm, gs) = (orient * l.gm / scale, l.gv.max(0.0).sqrt() / scale);
    for (x, q) in l.atoms() {
        let x0 = orient * x / scale - shift;
        if l.gp < 1.0 {
            tp.point(x0, w * q * (1.0 - l.gp), sign);
        }
        if l.gp > 0.0 {
            tp.normal(x0 + gm, gs, w * q * l.gp, sign);
        }
    }
}

/// Per rounding cell c: P(t + x in cell c) for x from a law (ulps, oriented), cells
/// [c + lo, c + hi).
fn cell_probs(
    t: f64,
    atoms: &[(f64, f64)],
    g: (f64, f64, f64),
    cell: (f64, f64),
    cs: (i64, i64),
) -> Vec<f64> {
    let (gp, gm, gs) = g;
    (cs.0..=cs.1)
        .map(|c| {
            atoms
                .iter()
                .map(|&(x0, q)| q * p_in(t, x0, gp, gm, gs, c as f64 + cell.0, c as f64 + cell.1))
                .sum()
        })
        .collect()
}

/// Largest excess kurtosis of y for which the DAG tier is applicable: the tier treats the
/// partial sums as normal given y, which heavy-tailed or scale-mixture terms break (a few
/// large terms dominate y and the partial sums before them are far smaller than the normal
/// law says).
pub const KURT_MAX: f64 = 1.0;

/// The DAG tier's structural applicability (before any Tier 3 cross-check): y close to
/// normal (excess kurtosis from the product terms at most `KURT_MAX`).
fn dag_applicable(d: &DagClt) -> (bool, String) {
    if d.y_kurt > KURT_MAX {
        (
            false,
            format!(
                "y's excess kurtosis {:.3} > {KURT_MAX}: heavy-tailed or scale-mixture terms, the partial sums are not normal given y FAILS",
                d.y_kurt
            ),
        )
    } else {
        (
            true,
            format!(
                "{} DAG ({} error terms), y's excess kurtosis {:.3} <= {KURT_MAX} ok",
                if d.has_mma { "MMA" } else { "add/FMA/TF32" },
                d.terms.len(),
                d.y_kurt
            ),
        )
    }
}

/// Tier 2 on a DAG output (single kernel).
pub fn dag_single(d: &DagClt, out_fmt: Format) -> Estimate {
    let mut est = Estimate {
        tier: 2,
        method: "CLT on the output's DAG: lattice-aware node rounding errors and TF32 split errors, independent given y".into(),
        outputs: 1,
        abs: moments(d.mean, d.var),
        u_scale: Some(u_scale(d.mean, d.var, d.s_mean)),
        error_bar: ErrorBar {
            kind: "clt".into(),
            notes: vec!["normal node values from independent leaves; node errors independent given y; f32 inputs have random low bits (N-ERRDIST-CLT)".into()],
            ..Default::default()
        },
        ..Default::default()
    };
    let strong = d.strong();
    let all = |_: u64| true;
    let out16 = d.round16.filter(|_| out_fmt != Format::F32);
    let dv = bias_extra(d, None, &all);
    if dv > 0.0 {
        est.error_bar.notes.push(format!(
            "correlated rounding means across partial sums (round toward zero): extra variance {dv:.3e} given y"
        ));
    }
    let mut tp = TentPmf::default();
    let p = over_y_pts_w(d.y_mean, d.y_var, |y, _, b, w| {
        let pre = with_var(d.law(y, b, &strong, &all, true), dv);
        let full = if out16.is_some() {
            with_var(d.law(y, b, &strong, &all, false), dv)
        } else {
            pre.clone()
        };
        match out16 {
            Some(f) => tent_law(
                &mut tp,
                &full,
                2f64.powi(b.max(f.emin()) - f.mbits()),
                1.0,
                0.0,
                1.0,
                w,
            ),
            None => {
                let s = y.signum();
                let c = if d.final_rz { 0.5 } else { 0.0 };
                tent_law(
                    &mut tp,
                    &pre,
                    2f64.powi(super::markov::ulp_exp(b)),
                    s,
                    c,
                    s,
                    w,
                );
            }
        }
        flip_p(d, out16, y, b, &pre, &full)
    });
    let (pmf, nn) = tp.finish(Some(p));
    est.pmf_ulp = pmf;
    est.error_bar.notes.extend(nn);
    if std::env::var("PTXV_T2DAG_DEBUG").is_ok() {
        eprintln!(
            "T2DAG single p {p:.4} p_nz {:.4} rz {} y {:.3e}/{:.3e} err {:.3e}/{:.3e} grid {:.3}",
            d.p_nz, d.final_rz, d.y_mean, d.y_var, d.mean, d.var, d.p_on_grid
        );
    }
    let (ok, why) = dag_applicable(d);
    est.applicability.push(why);
    est.applicable = Some(ok);
    est.p_differ = Some((p, p, p));
    est
}

/// Tier 2 on a pair of DAG outputs: D = delta_C - delta_R; identical sub-DAGs (equal
/// canonical hashes) carry the same error in both kernels and drop out of D.
pub fn dag_pair(r: &DagClt, c: &DagClt, out_fmt: Format) -> Estimate {
    let hr: std::collections::HashSet<u64> = r.terms.iter().map(|t| t.h).collect();
    let hc: std::collections::HashSet<u64> = c.terms.iter().map(|t| t.h).collect();
    let own = |h: u64| !(hr.contains(&h) && hc.contains(&h));
    let mut cov = 0.0;
    for t in &c.terms {
        if !own(t.h) {
            cov += (t.m2 - t.mean * t.mean).max(0.0);
        }
    }
    let fm = c.mean - r.mean;
    // The kernels' value-proportional means share the product terms: their covariance.
    let rb: HashMap<(u64, u64), f64> = r.bias.iter().map(|(k, (c, _, _))| (*k, *c)).collect();
    cov += c
        .bias
        .iter()
        .map(|(k, (cc, v, _))| v * cc * rb.get(k).copied().unwrap_or(0.0))
        .sum::<f64>();
    let fv = (c.var + r.var - 2.0 * cov).max(0.0);
    let mut est = Estimate {
        tier: 2,
        method: "CLT on both DAGs: lattice-aware node errors independent given y, identical sub-DAGs share their errors".into(),
        outputs: 1,
        abs: moments(fm, fv),
        u_scale: Some(u_scale(fm, fv, r.s_mean)),
        error_bar: ErrorBar {
            kind: "clt".into(),
            notes: vec![format!("shared error variance (identical sub-DAGs) {cov:.3e} of {:.3e} / {:.3e}", r.var, c.var)],
            ..Default::default()
        },
        pair: Some(PairMoments {
            mean_r: r.mean,
            var_r: r.var,
            mean_c: c.mean,
            var_c: c.var,
            cov_rc: cov,
            corr_rc: cov / (r.var * c.var).sqrt().max(1e-300),
        }),
        ..Default::default()
    };
    let (sr, sc) = (r.strong(), c.strong());
    let dv = bias_extra(r, Some(c), &own);
    if dv > 0.0 {
        est.error_bar.notes.push(format!(
            "correlated rounding means across partial sums (round toward zero), C - R: extra variance {dv:.3e} given y"
        ));
    }
    let out16 = r.round16.or(c.round16).filter(|_| out_fmt != Format::F32);
    let shared_h = |h: u64| !own(h);
    let rz = r.final_rz && c.final_rz;
    let mut tp = TentPmf::default();
    let p = over_y_pts_w(r.y_mean, r.y_var, |y, _, b, wy| {
        if let Some(f) = out16 {
            // D's flip of the 16-bit rounding: |D| / ulp16.
            let (lr, lc) = (
                r.law(y, b, &sr, &own, false),
                with_var(c.law(y, b, &sc, &own, false), dv),
            );
            let (mr, vr) = law_moments(&lr);
            let (mc, vc) = law_moments(&lc);
            let pz = 1.0 - (1.0 - law_pnz(&lr)) * (1.0 - law_pnz(&lc));
            let uo = 2f64.powi(b.max(f.emin()) - f.mbits());
            tp.zi(pz, (mc - mr) / uo, (vr + vc) / (uo * uo), 0.0, wy, 1.0);
            return zi_expect(pz, (mc - mr) / uo, (vr + vc) / (uo * uo), |x| {
                x.abs().min(1.0)
            });
        }
        // Both kernels round the same y (plus their shared error): the position within the
        // ulp and each kernel's own pre-rounding error decide.
        let (lr, lc) = (
            r.law(y, b, &sr, &own, true),
            with_var(c.law(y, b, &sc, &own, true), dv),
        );
        let u = 2f64.powi(super::markov::ulp_exp(b));
        let sg = y.signum();
        let orient = |l: &Law| -> (Vec<(f64, f64)>, f64, f64) {
            (
                l.atoms().map(|(x, q)| (sg * x / u, q)).collect(),
                sg * l.gm / u,
                l.gv.max(0.0).sqrt() / u,
            )
        };
        let (ar, gmr, gsr) = orient(&lr);
        let (ac, gmc, gsc) = orient(&lc);
        let spread = gsr.max(gsc);
        if spread > 16.0 {
            let (mr, vr) = law_moments(&lr);
            let (mc, vc) = law_moments(&lc);
            let pz = 1.0 - (1.0 - law_pnz(&lr)) * (1.0 - law_pnz(&lc));
            tp.zi(pz, (mc - mr) / u, (vr + vc) / (u * u), 0.0, wy, 1.0);
            return zi_expect(pz, (mc - mr) / u, (vr + vc) / (u * u), |x| x.abs().min(1.0));
        }
        // pmf_ulp: C - R = Q(t + aC) - Q(t + aR) with t uniform, through the tent kernel on
        // aC - aR (the four atom / normal combinations).
        if ar.len() * ac.len() > 1 << 20 {
            let (mr, vr) = law_moments(&lr);
            let (mc, vc) = law_moments(&lc);
            let pz = 1.0 - (1.0 - law_pnz(&lr)) * (1.0 - law_pnz(&lc));
            tp.zi(pz, (mc - mr) / u, (vr + vc) / (u * u), 0.0, wy, sg);
        } else {
            let (pr_, pc_) = (lr.gp, lc.gp);
            let gs_both = (gsr * gsr + gsc * gsc).sqrt();
            // aC - aR on the laws' grid first (one tent evaluation per difference, not per
            // pair of atoms).
            let hq = lr.h / u;
            let mut dd: std::collections::BTreeMap<i64, f64> = Default::default();
            for &(xr, qr) in &ar {
                for &(xc, qc) in &ac {
                    if qr * qc > 1e-16 {
                        *dd.entry(((xc - xr) / hq).round() as i64).or_insert(0.0) += qr * qc;
                    }
                }
            }
            for (&i, &qd) in &dd {
                let (x0, q) = (i as f64 * hq, wy * qd);
                tp.point(x0, q * (1.0 - pr_) * (1.0 - pc_), sg);
                if pr_ > 0.0 {
                    tp.normal(x0 - gmr, gsr, q * pr_ * (1.0 - pc_), sg);
                }
                if pc_ > 0.0 {
                    tp.normal(x0 + gmc, gsc, q * (1.0 - pr_) * pc_, sg);
                }
                if pr_ > 0.0 && pc_ > 0.0 {
                    tp.normal(x0 + gmc - gmr, gs_both, q * pr_ * pc_, sg);
                }
            }
        }
        let sh = r.cond(y, &sr, &shared_h, true);
        let law = if sh.1.sqrt() / u >= 0.25 && sh.2 > 0.5 {
            vec![(99, 1.0)]
        } else {
            r.pos_law(b)
        };
        let xmin = ar
            .iter()
            .chain(&ac)
            .map(|x| x.0)
            .fold(f64::INFINITY, f64::min)
            + gmr.min(gmc)
            - 8.0 * spread
            - 2.0;
        let xmax = ar
            .iter()
            .chain(&ac)
            .map(|x| x.0)
            .fold(f64::NEG_INFINITY, f64::max)
            + gmr.max(gmc)
            + 8.0 * spread
            + 2.0;
        let (lo_off, hi_off) = if rz { (0.0, 1.0) } else { (-0.5, 0.5) };
        let mut same = 0.0;
        for (k, w) in law {
            if k >= 7 || spread >= 0.3 {
                // Uniform position: P(differ | aR, aC) = min(1, |aC - aR|).
                let mut dd: std::collections::HashMap<i64, f64> = Default::default();
                let hq = lr.h / u;
                for &(xr, qr) in &ar {
                    for &(xc, qc) in &ac {
                        *dd.entry(((xc - xr) / hq).round() as i64).or_insert(0.0) += qr * qc;
                    }
                }
                let (pr_, pc_) = (lr.gp, lc.gp);
                let gs_both = (gsr * gsr + gsc * gsc).sqrt();
                let mut diff = 0.0;
                for (&i, &q) in &dd {
                    let x0 = i as f64 * hq;
                    diff += q
                        * ((1.0 - pr_) * (1.0 - pc_) * x0.abs().min(1.0)
                            + pr_ * (1.0 - pc_) * emin1(x0 - gmr, gsr)
                            + (1.0 - pr_) * pc_ * emin1(x0 + gmc, gsc)
                            + pr_ * pc_ * emin1(x0 + gmc - gmr, gs_both));
                }
                same += w * (1.0 - diff);
                continue;
            }
            for (t, wt) in positions(k) {
                let (c0, c1) = ((t + xmin).floor() as i64, (t + xmax).ceil() as i64);
                let vr = cell_probs(t, &ar, (lr.gp, gmr, gsr), (lo_off, hi_off), (c0, c1));
                let vc = cell_probs(t, &ac, (lc.gp, gmc, gsc), (lo_off, hi_off), (c0, c1));
                same += w * wt * vr.iter().zip(&vc).map(|(a, b)| a * b).sum::<f64>();
            }
        }
        (1.0 - same).clamp(0.0, 1.0)
    });
    let (pmf, nn) = tp.finish(Some(p));
    est.pmf_ulp = pmf;
    est.error_bar.notes.extend(nn);
    let (ok1, w1) = dag_applicable(r);
    let (ok2, w2) = dag_applicable(c);
    let ok = ok1 && ok2;
    est.applicability.push(if ok1 { w2 } else { w1 });
    est.applicable = Some(ok);
    est.p_differ = Some((p, p, p));
    est
}

/// Tier 2 over a sample of up to `max_out` outputs (evenly spaced; outputs differ in depth,
/// e.g. scans): per-output estimates averaged (P(differ), mixture moments).
pub fn dag_estimate(
    r: &Prog,
    c: Option<&Prog>,
    laws: &std::collections::BTreeMap<String, LeafLaw>,
    max_out: usize,
) -> Result<Estimate, String> {
    let idx: Vec<usize> = (0..r.outputs.len())
        .filter(|&i| {
            r.outputs[i].is_some_and(|o| r.reads_leaves(o))
                && c.is_none_or(|c| {
                    c.outputs
                        .get(i)
                        .copied()
                        .flatten()
                        .is_some_and(|o| c.reads_leaves(o))
                })
        })
        .collect();
    if idx.is_empty() {
        return Err("no input-dependent output".into());
    }
    let n = idx.len().min(max_out);
    let pick: Vec<usize> = (0..n).map(|k| idx[k * idx.len() / n]).collect();
    let one = |o: usize| -> Result<Estimate, String> {
        let dr = dag(r, r.outputs[o].unwrap(), laws)?;
        Ok(match c {
            None => dag_single(&dr, r.out_fmt),
            Some(c) => dag_pair(&dr, &dag(c, c.outputs[o].unwrap(), laws)?, r.out_fmt),
        })
    };
    let nt = std::env::var("PTXV_ERRDIST_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16usize)
        .max(1);
    let res: Vec<Result<Estimate, String>> = std::thread::scope(|sc| {
        let chunks: Vec<Vec<usize>> = (0..nt)
            .map(|t| pick.iter().copied().skip(t).step_by(nt).collect())
            .collect();
        let hs: Vec<_> = chunks
            .into_iter()
            .map(|ch| sc.spawn(move || ch.into_iter().map(|o| (o, one(o))).collect::<Vec<_>>()))
            .collect();
        let mut all: Vec<(usize, Result<Estimate, String>)> =
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect();
        all.sort_by_key(|x| x.0);
        all.into_iter().map(|x| x.1).collect()
    });
    let mut ests = vec![];
    for e in res {
        ests.push(e?);
    }
    let k = ests.len() as f64;
    let mix = |f: &dyn Fn(&Estimate) -> Option<Moments>| -> Option<Moments> {
        let ms: Vec<Moments> = ests.iter().filter_map(f).collect();
        if ms.is_empty() {
            return None;
        }
        let m = ms.iter().map(|x| x.mean).sum::<f64>() / ms.len() as f64;
        let s2 = ms
            .iter()
            .map(|x| x.std * x.std + x.mean * x.mean)
            .sum::<f64>()
            / ms.len() as f64;
        Some(moments(m, (s2 - m * m).max(0.0)))
    };
    let mut e = ests[0].clone();
    e.abs = mix(&|x| Some(x.abs.clone())).unwrap();
    e.u_scale = mix(&|x| x.u_scale.clone());
    let p = ests
        .iter()
        .map(|x| x.p_differ.map_or(0.0, |p| p.0))
        .sum::<f64>()
        / k;
    e.p_differ = Some((p, p, p));
    // pmf_ulp: the outputs' PMFs averaged.
    let mut pm: std::collections::BTreeMap<i64, f64> = Default::default();
    for x in &ests {
        for &(kk, q) in &x.pmf_ulp {
            *pm.entry(kk).or_default() += q / k;
        }
    }
    e.pmf_ulp = pm.into_iter().collect();
    let bad: Vec<usize> = pick
        .iter()
        .zip(&ests)
        .filter(|(_, x)| x.applicable == Some(false))
        .map(|(o, _)| *o)
        .collect();
    // Outputs outside the tier's domain matter in proportion to their P(differ).
    let pbad = ests
        .iter()
        .filter(|x| x.applicable == Some(false))
        .map(|x| x.p_differ.map_or(0.0, |p| p.0))
        .sum::<f64>()
        / k;
    let ok = pbad <= 0.05 * p.max(1e-300) || bad.is_empty();
    e.applicability = vec![format!(
        "{} of {} sampled outputs outside the DAG tier's domain, carrying {:.3e} of P(differ) {:.3e} {}",
        bad.len(),
        n,
        pbad,
        p,
        if ok { "ok" } else { "FAILS" }
    )];
    if let Some(x) = ests
        .iter()
        .find(|x| x.applicable == Some(false))
        .or(ests.first())
    {
        e.applicability
            .extend(x.applicability.iter().map(|s| format!("e.g. {s}")));
    }
    e.applicable = Some(ok);
    if n < idx.len() {
        e.error_bar.notes.push(format!(
            "{n} of {} outputs analyzed (evenly spaced)",
            idx.len()
        ));
    }
    Ok(e)
}

#[cfg(test)]
mod dag_tests {
    use super::*;

    /// The zero-inflated expectation resolves the narrow structure of f at |x| <= 2 even
    /// for a wide normal (a coarse grid point there once cost 1%).
    #[test]
    fn zi_expect_wide_normal() {
        let z = zi_expect(1.0, 0.0, 1e8, |x| (x - 0.5).abs().min(1.0));
        assert!((z - 1.0).abs() < 1e-3, "{z}");
        let z = zi_expect(1.0, 0.0, 1.0, |x| x.abs().min(1.0));
        // E min(1, |N(0,1)|) = 2(phi(0) - phi(1)) + P(|Z| > 1)
        let want = 2.0
            * (super::super::special::phi_pdf(0.0) - super::super::special::phi_pdf(1.0))
            + 2.0 * (1.0 - phi(1.0));
        assert!((z - want).abs() < 1e-6, "{z} {want}");
        assert!((emin1(0.0, 1.0) - want).abs() < 1e-9);
    }

    /// A reduction of n f32 leaves (sequential or pairwise), its Prog and its simulation.
    fn reduction(n: usize, tree: bool) -> (Prog, Vec<(usize, usize)>) {
        let mut ops = vec![];
        for i in 0..n {
            ops.push(Op::Leaf {
                buf: 0,
                idx: i as u64,
                fmt: Format::F32,
            });
        }
        let mut adds = vec![];
        let mut cur: Vec<u32> = (0..n as u32).collect();
        if tree {
            while cur.len() > 1 {
                let mut nx = vec![];
                for c in cur.chunks(2) {
                    if c.len() == 2 {
                        ops.push(Op::Add {
                            a: c[0],
                            b: c[1],
                            rz: false,
                        });
                        adds.push((c[0] as usize, c[1] as usize));
                        nx.push(ops.len() as u32 - 1);
                    } else {
                        nx.push(c[0]);
                    }
                }
                cur = nx;
            }
        } else {
            let mut acc = cur[0];
            for &x in &cur[1..] {
                ops.push(Op::Add {
                    a: acc,
                    b: x,
                    rz: false,
                });
                adds.push((acc as usize, x as usize));
                acc = ops.len() as u32 - 1;
            }
            cur = vec![acc];
        }
        let hw = vec![0; ops.len()];
        (
            Prog {
                ops,
                hw,
                outputs: vec![Some(cur[0])],
                out_fmt: Format::F32,
                bufs: vec!["x".into()],
            },
            adds,
        )
    }

    fn sim_reduction(
        n: usize,
        adds: &[(usize, usize)],
        d: &dyn Fn(&mut super::super::special::Rng) -> f64,
        trials: usize,
    ) -> f64 {
        let mut r = super::super::special::Rng::new(3);
        let mut differ = 0usize;
        for _ in 0..trials {
            let mut v: Vec<f32> = (0..n).map(|_| d(&mut r) as f32).collect();
            let exact: f64 = v.iter().map(|&x| x as f64).sum();
            for &(a, b) in adds {
                let s = v[a] + v[b];
                v.push(s);
            }
            if *v.last().unwrap() != exact as f32 {
                differ += 1;
            }
        }
        differ as f64 / trials as f64
    }

    /// Tier 2 on f32 reductions (sequential and pairwise; normal and same-sign uniform
    /// inputs) against simulation: P(computed != RN(exact)) within 5%.
    #[test]
    fn reductions_match_simulation() {
        use super::super::spec::Dist;
        type Draw = Box<dyn Fn(&mut super::super::special::Rng) -> f64>;
        let cases: Vec<(Dist, Draw)> = vec![
            (
                Dist::Normal {
                    mean: 0.0,
                    std: 1.0,
                },
                Box::new(|r| r.normal()),
            ),
            (
                Dist::Uniform { lo: 1.0, hi: 2.0 },
                Box::new(|r| 1.0 + r.unit()),
            ),
        ];
        for (dist, f) in cases {
            for tree in [false, true] {
                let n = 64;
                let (p, adds) = reduction(n, tree);
                let laws: std::collections::BTreeMap<String, LeafLaw> = [(
                    "x".to_string(),
                    LeafLaw {
                        pmf: super::super::laws::f32_smooth(&dist),
                        smooth: true,
                    },
                )]
                .into();
                let e = dag_estimate(&p, None, &laws, 1).unwrap();
                let pm = e.p_differ.unwrap().0;
                let ps = sim_reduction(n, &adds, &*f, 200_000);
                eprintln!("{dist:?} tree {tree}: tier 2 {pm:.4} sim {ps:.4}");
                assert!(
                    (pm - ps).abs() <= 0.05 * ps + 0.005,
                    "{dist:?} tree {tree}: {pm} vs {ps}"
                );
            }
        }
    }

    /// Grid residue moments: n = 2 (one bit) RN is a tie half the time (|e| = 1/2).
    #[test]
    fn grid_moments_small_grids() {
        let (m, e2, p) = grid_moments(1, false);
        assert_eq!((m, p), (0.0, 0.5));
        assert!((e2 - 0.125).abs() < 1e-12);
        let (m, e2, p) = grid_moments(2, true);
        assert!((m + 0.375).abs() < 1e-12 && (p - 0.75).abs() < 1e-12);
        assert!((e2 - (0.0 + 1.0 / 16.0 + 0.25 + 9.0 / 16.0) / 4.0).abs() < 1e-12);
        let (_, e2, _) = grid_moments(60, false);
        assert!((e2 - 1.0 / 12.0).abs() < 1e-12);
    }
}

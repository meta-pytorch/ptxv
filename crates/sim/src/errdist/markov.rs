/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tier 1: the error of one kernel's MMA accumulator chain as a Markov chain on the coarse
//! state (sign, binade) of the exact partial sum, with the exact-residue local error law of
//! each transition (docs/errdist.md §4-5).
//!
//! Coarse: the partial sums' laws are merged atom clouds (`laws::Cloud`), exact for small
//! supports; the joint law of (acc cell, emax, result cell) comes from them, emax by
//! inclusion-exclusion over the terms' alignment exponents.
//! Fine: given a transition, the step's local error is tau_acc + sum tau_terms + rho, rho =
//! -sign(S') (|S'| mod ulp) the final round toward zero, whose residue law on Z_r comes from
//! the terms' characteristic functions (`fine`). Assumed (N-ERRDIST-RESIDUE, -MARKOV): the
//! residues are independent of the coarse state given the transition, the accumulator's bits
//! above its ulp are uniform on the partial sums' lattice, and the coarse state is Markov.

use std::sync::Arc as Rc;

use super::fine::C64;
use super::fine::Pop;
use super::fine::TermLaw;
use super::fine::fft;
use super::laws::Cloud;
use super::schema::ChainSchema;
use super::schema::StepSchema;
use crate::errdist::det::HashMap;

/// Largest residue modulus computed exactly; larger ones are taken uniform.
pub const R_MAX: usize = 4096;
/// Sub-ulp resolution of the error histograms (bins per ulp of the state's binade).
pub const G: i64 = 16;
/// Histogram half-width in ulps.
pub const W_ULP: i64 = 64;
const W: i64 = G * W_ULP;
/// Bin maps (small integer keys).
pub type HMap = crate::errdist::det::HashMapWith<i64, f64, super::laws::MulHash>;

/// Tier 1 gives up beyond this many histogram entries (memory guard, ~1-2 GB).
pub(crate) const MAX_ENTRIES: usize = 40_000_000;

/// Beyond this many bins the mass is clipped (tail).
pub(crate) const W_MAX: i64 = 1 << 56;

/// Bin index kept exactly within +-W, beyond rounded to 4 significant bits (log-spaced bins
/// for the large deviations a cancellation produces in a small result binade).
#[inline]
pub fn quant(v: i64) -> i64 {
    let a = v.unsigned_abs();
    if a <= W as u64 {
        return v;
    }
    let sh = 63 - a.leading_zeros() as i32 - 3;
    let r = ((a + (1u64 << (sh - 1))) >> sh) << sh;
    if v < 0 { -(r as i64) } else { r as i64 }
}
const CAP: usize = 384;

/// Coarse cells: 0 = exactly zero; then (sign, binade) for binades bmin..=bmax.
#[derive(Clone, Copy, Debug)]
pub struct Cells {
    pub bmin: i32,
    pub bmax: i32,
}

impl Cells {
    pub fn nb(&self) -> usize {
        (self.bmax - self.bmin + 1) as usize
    }
    pub fn n(&self) -> usize {
        1 + 2 * self.nb()
    }
    pub fn of(&self, x: f64) -> usize {
        if x == 0.0 {
            return 0;
        }
        let b = super::spec::exp_floor(x).clamp(self.bmin, self.bmax);
        1 + if x < 0.0 { self.nb() } else { 0 } + (b - self.bmin) as usize
    }
    pub fn binade(&self, c: usize) -> i32 {
        self.bmin + ((c - 1) % self.nb()) as i32
    }
    pub fn sign(&self, c: usize) -> f64 {
        if c == 0 {
            0.0
        } else if c > self.nb() {
            -1.0
        } else {
            1.0
        }
    }
}

/// f32 ulp exponent of binade b (subnormal range: 2^-149).
pub fn ulp_exp(b: i32) -> i32 {
    b.max(-126) - 23
}

// ------------------------------------------------------------------------ fine statistics

/// Per (TermLaw, q, e): residues mod R_MAX (folded to smaller r on demand) and tau statistics.
struct TermFine {
    mass: f64,
    res: Vec<f64>,
    res0: Vec<f64>,
    tau1: f64,
    tau2: f64,
    tau0: f64,
    tau_hist: [f64; 33],
}

#[derive(Default)]
struct FineCache {
    terms: HashMap<(usize, i32, i32), Rc<TermFine>>,
    meas: HashMap<(usize, i32, usize, i32), Rc<Measure>>,
    taupw: HashMap<(usize, i32, i32, usize), Rc<Vec<f64>>>,
}

impl FineCache {
    fn term(&mut self, t: &Rc<TermLaw>, q: i32, e: i32) -> Rc<TermFine> {
        let key = (Rc::as_ptr(t) as usize, q, e);
        if let Some(x) = self.terms.get(&key) {
            return x.clone();
        }
        let st = super::fine::term_stats(t, q, R_MAX, e);
        let f = Rc::new(TermFine {
            mass: st.mass,
            res: st.res,
            res0: st.res0,
            tau1: st.tau1,
            tau2: st.tau2,
            tau0: st.tau0,
            tau_hist: st.tau_hist,
        });
        self.terms.insert(key, f.clone());
        f
    }

    /// The step's measure with all terms aligned <= e at quantum 2^q, modulus r: (mass,
    /// phi (unnormalized), E tau, E tau^2, P(all tau 0), tau histogram), as `fine::step_measure`.
    fn measure(&mut self, pop: &Pop, q: i32, r: usize, e: i32) -> Measure {
        let key = (pop as *const Pop as usize, q, r, e);
        if let Some(m) = self.meas.get(&key) {
            return (**m).clone();
        }
        let m = self.measure_new(pop, q, r, e);
        self.meas.insert(key, Rc::new(m.clone()));
        m
    }

    fn measure_new(&mut self, pop: &Pop, q: i32, r: usize, e: i32) -> Measure {
        let mut out = Measure {
            mass: 1.0,
            phi: vec![C64::new(1.0, 0.0); r],
            phi0: vec![C64::new(1.0, 0.0); r],
            tau1: 0.0,
            tau2: 0.0,
            tau0: 1.0,
            tau_hist: unit_hist(),
        };
        for g in &pop.groups {
            let mut gmass = 0.0;
            let mut gphi = vec![C64::new(0.0, 0.0); r];
            let mut gphi0 = vec![C64::new(0.0, 0.0); r];
            let (mut g1, mut g2, mut g0) = (0.0, 0.0, 0.0);
            let mut ghist: Vec<f64> = vec![0.0; out.tau_hist.len()];
            for (w, t) in &g.parts {
                let st = self.term(t, q, e);
                if st.mass <= 0.0 {
                    continue;
                }
                let m = st.mass;
                let mn = m.powi(g.n as i32);
                gmass += w * mn;
                let mut f: Vec<C64> = vec![C64::new(0.0, 0.0); r];
                for (k, p) in st.res.iter().enumerate() {
                    f[k % r].re += p / m;
                }
                fft(&mut f, false);
                for k in 0..r {
                    gphi[k] = gphi[k].add(f[k].powi(g.n).scale(w * mn));
                }
                let mut f0: Vec<C64> = vec![C64::new(0.0, 0.0); r];
                for (k, p) in st.res0.iter().enumerate() {
                    f0[k % r].re += p / m;
                }
                fft(&mut f0, false);
                for k in 0..r {
                    gphi0[k] = gphi0[k].add(f0[k].powi(g.n).scale(w * mn));
                }
                let (t1, t2) = (st.tau1 / m, st.tau2 / m);
                let n = g.n as f64;
                g1 += w * mn * n * t1;
                g2 += w * mn * (n * t2 + n * (n - 1.0) * t1 * t1);
                g0 += w * mn * (st.tau0 / m).powi(g.n as i32);
                let tk = (Rc::as_ptr(t) as usize, q, e, g.n);
                let pw = match self.taupw.get(&tk) {
                    Some(p) => p.clone(),
                    None => {
                        let one: Vec<f64> = st.tau_hist.iter().map(|x| x / m).collect();
                        let pz = st.tau0 / m;
                        let p = Rc::new(if g.n as f64 * (1.0 - pz) > 3.0 {
                            tau_clt(n * t1, n * (t2 - t1 * t1), pz.powi(g.n as i32))
                        } else {
                            tau_power(&one, g.n)
                        });
                        self.taupw.insert(tk, p.clone());
                        p
                    }
                };
                for (i, x) in pw.iter().enumerate() {
                    ghist[i] += w * mn * x;
                }
            }
            let (m0, t1_0, t2_0) = (out.mass, out.tau1, out.tau2);
            out.tau2 = t2_0 * gmass + 2.0 * t1_0 * g1 + m0 * g2;
            out.tau1 = t1_0 * gmass + m0 * g1;
            out.mass = m0 * gmass;
            for k in 0..r {
                out.phi[k] = out.phi[k].mul(gphi[k]);
                out.phi0[k] = out.phi0[k].mul(gphi0[k]);
            }
            out.tau0 *= g0;
            out.tau_hist = super::fine::conv_centered(&out.tau_hist, &ghist);
        }
        out
    }
}

/// tau histograms over +-TW/16 Q at 1/16 Q.
const TW: usize = 16 * 16;

fn unit_hist() -> Vec<f64> {
    let mut h = vec![0.0; 2 * TW + 1];
    h[TW] = 1.0;
    h
}

fn tau_power(one: &[f64], n: usize) -> Vec<f64> {
    let mut base = vec![0.0; 2 * TW + 1];
    for (i, x) in one.iter().enumerate() {
        base[TW + i - 16] += x;
    }
    let mut res = unit_hist();
    let mut k = n;
    let mut b = base;
    while k > 0 {
        if k & 1 == 1 {
            res = super::fine::conv_centered(&res, &b);
        }
        k >>= 1;
        if k > 0 {
            b = super::fine::conv_centered(&b, &b);
        }
    }
    res
}

/// A discretized normal (mean, var in Q units) on the tau grid, with an atom p0 at 0.
fn tau_clt(mean: f64, var: f64, p0: f64) -> Vec<f64> {
    let mut h = vec![0.0; 2 * TW + 1];
    h[TW] += p0;
    let sd = var.max(1e-12).sqrt();
    let mut w = vec![0.0; 2 * TW + 1];
    let mut tot = 0.0;
    for (i, x) in w.iter_mut().enumerate() {
        let v = (i as f64 - TW as f64) / 16.0;
        if i == TW {
            continue;
        }
        *x = (-0.5 * ((v - mean / (1.0 - p0).max(1e-12)) / sd).powi(2)).exp();
        tot += *x;
    }
    if tot > 0.0 {
        for (i, x) in w.iter().enumerate() {
            h[i] += (1.0 - p0) * x / tot;
        }
    }
    h
}

#[derive(Clone)]
struct Measure {
    mass: f64,
    phi: Vec<C64>,
    /// The same over the event "every tau = 0" (for the joint zero probability).
    phi0: Vec<C64>,
    tau1: f64,
    tau2: f64,
    tau0: f64,
    tau_hist: Vec<f64>,
}

impl Measure {
    fn sub(&self, o: &Measure) -> Measure {
        Measure {
            mass: self.mass - o.mass,
            phi: self
                .phi
                .iter()
                .zip(&o.phi)
                .map(|(a, b)| C64::new(a.re - b.re, a.im - b.im))
                .collect(),
            phi0: self
                .phi0
                .iter()
                .zip(&o.phi0)
                .map(|(a, b)| C64::new(a.re - b.re, a.im - b.im))
                .collect(),
            tau1: self.tau1 - o.tau1,
            tau2: self.tau2 - o.tau2,
            tau0: self.tau0 - o.tau0,
            tau_hist: self
                .tau_hist
                .iter()
                .zip(&o.tau_hist)
                .map(|(a, b)| a - b)
                .collect(),
        }
    }
}

// ------------------------------------------------------------------------ coarse clouds

/// Atom cap of a term's n-fold sum law (only steers the partial sums' binade law).
const POW_CAP: usize = 128;

/// A term law's atoms by alignment exponent, each slice merged to a small cloud of (v, p).
type Slices = Vec<(i32, Vec<(f64, f64)>)>;

#[derive(Default)]
struct CloudCache {
    powers: HashMap<(usize, i32, usize), Rc<(f64, Cloud)>>,
    /// Per term law: its atoms by alignment exponent, each slice merged to a small cloud.
    slices: HashMap<usize, Rc<Slices>>,
}

impl CloudCache {
    /// Law of n i.i.d. copies' sum of a term restricted to alignment <= e: (P(all <= e),
    /// normalized cloud).
    fn power(&mut self, t: &Rc<TermLaw>, e: i32, n: usize) -> Rc<(f64, Cloud)> {
        let key = (Rc::as_ptr(t) as usize, e, n);
        if let Some(x) = self.powers.get(&key) {
            return x.clone();
        }
        let sl = self
            .slices
            .entry(Rc::as_ptr(t) as usize)
            .or_insert_with(|| {
                let mut out: Vec<(i32, Vec<(f64, f64)>)> = vec![];
                for a in &t.atoms {
                    match out.last_mut() {
                        Some(l) if l.0 == a.1 => l.1.push((a.0, a.2)),
                        _ => out.push((a.1, vec![(a.0, a.2)])),
                    }
                }
                for l in out.iter_mut() {
                    let c = Cloud::from_atoms(std::mem::take(&mut l.1), 64);
                    l.1 = c.a;
                }
                Rc::new(out)
            })
            .clone();
        let atoms: Vec<(f64, f64)> = sl
            .iter()
            .filter(|x| x.0 <= e)
            .flat_map(|x| x.1.iter().copied())
            .collect();
        let m: f64 = atoms.iter().map(|a| a.1).sum();
        let res = if m <= 0.0 {
            Rc::new((0.0, Cloud::point(0.0)))
        } else {
            let c = Cloud::from_atoms(
                atoms.into_iter().map(|(v, p)| (v, p / m)).collect(),
                POW_CAP,
            );
            Rc::new((m.powi(n as i32), c.power(n, POW_CAP)))
        };
        self.powers.insert(key, res.clone());
        res
    }

    /// The (signed, unnormalized) law of the step's terms' sum with every term aligned <= e.
    fn block(&mut self, pop: &Pop, e: i32) -> Signed {
        let mut out = Signed {
            atoms: vec![(0.0, 1.0)],
            merged: false,
        };
        for g in &pop.groups {
            let mut gat: Vec<(f64, f64)> = vec![];
            for (w, t) in &g.parts {
                let pw = self.power(t, e, g.n);
                if pw.0 == 0.0 {
                    continue;
                }
                out.merged |= pw.1.merged;
                for &(v, p) in &pw.1.a {
                    gat.push((v, w * pw.0 * p));
                }
            }
            out.merged |= gat.len() > CAP;
            let gc = compact_signed(gat);
            let mut n = Vec::with_capacity(out.atoms.len() * gc.len());
            for &(x, p) in &out.atoms {
                for &(y, q) in &gc {
                    super::laws::push_sum(&mut n, x, y, p * q, out.merged);
                }
            }
            out.merged |= n.len() > CAP;
            out.atoms = compact_signed(n);
        }
        out
    }
}

/// A signed measure as atoms.
#[derive(Clone, Debug)]
struct Signed {
    atoms: Vec<(f64, f64)>,
    merged: bool,
}

/// Merge a signed atom list into at most CAP log bins (probability-weighted positions by
/// |weight|).
fn compact_signed(mut a: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    a.retain(|x| x.1 != 0.0);
    if a.len() <= CAP {
        // Merge equal values only.
        a.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        let mut out: Vec<(f64, f64)> = vec![];
        for (v, p) in a {
            match out.last_mut() {
                Some(l) if l.0 == v => l.1 += p,
                _ => out.push((v, p)),
            }
        }
        return out;
    }
    // Bin by sign and log2 at decreasing resolution until under CAP.
    let mut out: Vec<(f64, f64)> = super::laws::log_bins(&a, CAP)
        .into_iter()
        .filter(|b| b.w > 0.0)
        .map(|b| (b.s1 / b.w, b.signed))
        .collect();
    out.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    out
}

use super::laws::SMOOTH;

/// Spread mass w uniformly over [s - h, s + h] into the cells.
fn spread(cells: &Cells, s: f64, h: f64, w: f64, cm: &mut [f64]) {
    let (lo, hi) = (s - h, s + h);
    for sign in [1.0f64, -1.0] {
        // The part of [lo, hi] on this side, as magnitudes [a, b].
        let (a, b) = if sign > 0.0 {
            (lo.max(0.0), hi.max(0.0))
        } else {
            ((-hi).max(0.0), (-lo).max(0.0))
        };
        if b <= a {
            continue;
        }
        let mut top = b;
        let mut bin = super::spec::exp_floor(b);
        while top > a {
            let floor = 2f64.powi(bin).max(a);
            let len = top - floor;
            if bin <= cells.bmin {
                cm[cells.of(sign * 2f64.powi(cells.bmin))] += w * (top - a) / (2.0 * h);
                break;
            }
            cm[cells.of(sign * 2f64.powi(bin.min(cells.bmax)))] += w * len / (2.0 * h);
            top = floor;
            bin -= 1;
        }
    }
}

// --------------------------------------------------------------------- local error laws

/// A local error law in units of ulp(b_S)/G: P(exactly 0) and the nonzero part (index, prob).
#[derive(Clone, Debug, Default)]
pub struct Local {
    pub p0: f64,
    pub nz: Vec<(i64, f64)>,
    /// Exact moments (units of ulp(b_S)).
    pub m1: f64,
    pub m2: f64,
    /// Residue diagnostics: max |phi_k| (k != 0), TV to uniform, r.
    pub max_fourier: f64,
    pub tv: f64,
    pub r: usize,
}

/// Uniform-heuristic switch and options.
#[derive(Clone, Copy, Debug)]
pub struct Opts {
    /// Replace every residue law by the uniform law on Z_r (the "smooth density" heuristic).
    pub uniform: bool,
}

/// What the chain needs about the accumulator at a transition.
#[derive(Clone, Copy, Debug)]
struct AccCtx {
    /// None: no accumulator (first step or exactly zero).
    b: Option<i32>,
    sign: f64,
    /// Exponent of the lattice the exact partial sum lies on.
    lattice: i32,
}

/// `Engine::local`'s memo key: (step, acc binade, acc sign, emax, result binade, result sign,
/// last).
type LocalKey = (usize, Option<i32>, i8, i32, i32, i8, bool);

struct Engine<'a> {
    sch: &'a ChainSchema,
    opts: Opts,
    fine: FineCache,
    clouds: CloudCache,
    local: HashMap<LocalKey, Rc<Local>>,
    prev: HashMap<(usize, i32, u32), Vec<f64>>,
}

impl<'a> Engine<'a> {
    /// Local error law of step `si` for (acc, emax E, result binade bs, sign ss). `last`:
    /// without the final rounding (the pre-final deviation).
    // ASSUME(N-ERRDIST-RESIDUE): residues from the previous terms; components independent.
    fn local(&mut self, si: usize, acc: AccCtx, e: i32, bs: i32, ss: f64, last: bool) -> Rc<Local> {
        let key = (si, acc.b, acc.sign as i8, e, bs, ss as i8, last);
        if let Some(l) = self.local.get(&key) {
            return l.clone();
        }
        let st: &StepSchema = &self.sch.steps[si];
        // All values are multiples of 2^lattice: truncating finer than that is a no-op, so
        // the grid is 2^max(emax - frac, lattice) (exact).
        let q = (e - st.frac).max(acc.lattice);
        let ue = ulp_exp(bs);
        let r_full = if ue > q {
            1usize << (ue - q).min(40)
        } else {
            1
        };
        let r = r_full.min(R_MAX);
        // Terms: emax = E set by the accumulator (all terms <= E) or by a term (max = E).
        let meas = match acc.b {
            Some(b) if b == e => self.fine.measure(&st.pop, q, r, e),
            _ => {
                let a = self.fine.measure(&st.pop, q, r, e);
                let b = self.fine.measure(&st.pop, q, r, e - 1);
                a.sub(&b)
            }
        };
        let mut l = Local {
            r: r_full,
            ..Default::default()
        };
        if meas.mass <= 1e-300 {
            l.p0 = 1.0;
            let l = Rc::new(l);
            self.local.insert(key, l.clone());
            return l;
        }
        let m = meas.mass;
        // Accumulator: residue contribution (bits in [max(Q, g), ulp_S)) and truncation (bits
        // below Q), uniform on multiples of g = max(its ulp, the lattice).
        let mut phi: Vec<C64> = meas.phi.iter().map(|c| c.scale(1.0 / m)).collect();
        let mut phi0: Vec<C64> = meas.phi0.iter().map(|c| c.scale(1.0 / m)).collect();
        let (mut tx1, mut tx2, mut tx0) = (0.0, 0.0, 1.0);
        let mut tx_hist = unit_hist();
        if let Some(b) = acc.b {
            let g = ulp_exp(b).max(acc.lattice);
            // Residue: X mod ulp_S on the Q grid: X's bits in [max(g, q), ulp_S), which are the
            // previous terms' (truncated) sum's bits there (N-ERRDIST-RESIDUE).
            let gq = g.max(q);
            if gq < ue && r > 1 {
                let kb = ((ue - gq) as u32).min(R_MAX.trailing_zeros());
                let pr = if self.opts.uniform {
                    vec![1.0 / (1u64 << kb) as f64; 1 << kb]
                } else {
                    self.prev_res(si, gq, kb)
                };
                let sh = 1usize << (gq - q).min(40);
                let mut f = vec![C64::new(0.0, 0.0); r];
                for (i, p) in pr.iter().enumerate() {
                    f[(i * sh) % r].re += p;
                }
                fft(&mut f, false);
                for (c, x) in phi.iter_mut().zip(&f) {
                    *c = c.mul(*x);
                }
                for (c, x) in phi0.iter_mut().zip(&f) {
                    *c = c.mul(*x);
                }
            }
            // Truncation of X at Q: |X| mod Q, X's bits in [g, q).
            if g < q {
                let kb = (q - g).min(12) as u32;
                let cnt = 1usize << kb;
                let d = 1.0 / cnt as f64;
                let pr = if self.opts.uniform || q - g > 12 {
                    vec![d; cnt]
                } else {
                    let p = self.prev_res(si, g, kb);
                    // |X| for X < 0 reflects the residue.
                    if acc.sign < 0.0 {
                        (0..cnt).map(|i| p[(cnt - i) % cnt]).collect()
                    } else {
                        p
                    }
                };
                tx0 = if q - g > 12 { 0.0 } else { pr[0] };
                tx1 = 0.0;
                tx2 = 0.0;
                tx_hist = vec![0.0; 2 * TW + 1];
                for (i, p) in pr.iter().enumerate() {
                    let v = -acc.sign * i as f64 * d;
                    tx1 += p * v;
                    tx2 += p * v * v;
                    let bi =
                        ((v * 16.0).round() as i64 + TW as i64).clamp(0, 2 * TW as i64) as usize;
                    tx_hist[bi] += p;
                }
            }
        }
        // Residue law of |S'| mod ulp_S in Q units.
        let mut res = super::fine::inverse(&phi);
        for x in res.iter_mut() {
            *x = x.max(0.0);
        }
        let tot: f64 = res.iter().sum();
        for x in res.iter_mut() {
            *x /= tot.max(1e-300);
        }
        if r > 1 {
            let (tv, _, maxf) = super::fine::tv_uniform(&phi);
            l.tv = tv;
            l.max_fourier = maxf;
        }
        if self.opts.uniform || r_full > R_MAX {
            res = vec![1.0 / r as f64; r];
        }
        // R for |S'|: S' < 0 reflects the residue.
        let resa: Vec<f64> = if ss < 0.0 {
            (0..r).map(|j| res[(r - j) % r]).collect()
        } else {
            res
        };
        // Combine (independent): tau_x + tau_terms + rho, in units of ulp_S / G.
        let qg = (G as f64) / (r_full as f64); // one Q in bins (may be fractional)
        let th: Vec<f64> = meas.tau_hist.iter().map(|x| x / m).collect();
        let mut comb: HashMap<i64, f64> = HashMap::default();
        let p_tau0 = meas.tau0 / m;
        let p_r0 = if last || r_full == 1 { 1.0 } else { resa[0] };
        for (i, &pt) in th.iter().enumerate() {
            if pt.abs() < 1e-300 {
                continue;
            }
            let tv = (i as f64 - TW as f64) / 16.0; // Q units
            for (ix, &px) in tx_hist.iter().enumerate() {
                if px.abs() < 1e-300 {
                    continue;
                }
                let xv = (ix as f64 - TW as f64) / 16.0;
                let base = (tv + xv) * qg;
                if last || r_full == 1 {
                    *comb.entry(base.round() as i64).or_default() += pt * px;
                } else {
                    for (j, &pr) in resa.iter().enumerate() {
                        if pr <= 0.0 {
                            continue;
                        }
                        let rho = -ss * j as f64 * qg;
                        *comb.entry((base + rho).round() as i64).or_default() += pt * px * pr;
                    }
                }
            }
        }
        // Exact zero: every component zero (independence; exact cancellations ignored).
        // P(every tau = 0 and residue 0) jointly over the terms (both are "no small terms").
        let p_joint = if last || r_full == 1 {
            p_tau0
        } else if self.opts.uniform || r_full > R_MAX {
            p_tau0 * p_r0
        } else {
            super::fine::inverse(&phi0)[0].max(0.0)
        };
        l.p0 = (p_joint * tx0).clamp(0.0, 1.0);
        let z: f64 = comb.get(&0).copied().unwrap_or(0.0);
        let nz_at0 = (z - l.p0).max(0.0);
        comb.remove(&0);
        if nz_at0 > 0.0 {
            *comb.entry(1).or_default() += nz_at0 / 2.0;
            *comb.entry(-1).or_default() += nz_at0 / 2.0;
        }
        let mut nz: Vec<(i64, f64)> = comb.into_iter().filter(|x| x.1 > 0.0).collect();
        nz.sort_by_key(|x| x.0);
        let tot: f64 = nz.iter().map(|x| x.1).sum::<f64>() + l.p0;
        for x in nz.iter_mut() {
            x.1 /= tot;
        }
        l.p0 /= tot;
        l.nz = nz;
        // Exact moments in ulp_S units.
        let qu = 1.0 / r_full as f64;
        let (t1, t2) = (meas.tau1 / m * qu, meas.tau2 / m * qu * qu);
        let (x1, x2) = (tx1 * qu, tx2 * qu * qu);
        let (r1, r2) = if last || r_full == 1 {
            (0.0, 0.0)
        } else {
            let r1: f64 = resa
                .iter()
                .enumerate()
                .map(|(j, p)| p * j as f64)
                .sum::<f64>()
                * qu;
            let r2: f64 = resa
                .iter()
                .enumerate()
                .map(|(j, p)| p * (j * j) as f64)
                .sum::<f64>()
                * qu
                * qu;
            (-ss * r1, r2)
        };
        l.m1 = t1 + x1 + r1;
        l.m2 = t2 + x2 + r2 + 2.0 * (t1 * x1 + t1 * r1 + x1 * r1);
        let l = Rc::new(l);
        self.local.insert(key, l.clone());
        l
    }
}

impl Engine<'_> {
    /// PMF on Z_(2^kb) of the sum of the terms of steps before `si`, each truncated toward zero
    /// at 2^gq, in units of 2^gq.
    fn prev_res(&mut self, si: usize, gq: i32, kb: u32) -> Vec<f64> {
        let key = (si, gq, kb);
        if let Some(v) = self.prev.get(&key) {
            return v.clone();
        }
        let r = 1usize << kb;
        let mut phi = vec![C64::new(1.0, 0.0); r];
        for j in 0..si {
            let m = self
                .fine
                .measure(&self.sch.steps[j].pop, gq, r, i32::MAX / 2);
            for (c, x) in phi.iter_mut().zip(&m.phi) {
                *c = c.mul(x.scale(1.0 / m.mass));
            }
        }
        let mut p = super::fine::inverse(&phi);
        for x in p.iter_mut() {
            *x = x.max(0.0);
        }
        let t: f64 = p.iter().sum();
        for x in p.iter_mut() {
            *x /= t;
        }
        self.prev.insert(key, p.clone());
        p
    }
}

// ------------------------------------------------------------------------------ the chain

/// An error histogram in units of ulp(state binade) / G: exact-zero mass, nonzero bins.
#[derive(Clone, Debug, Default)]
pub struct Hist {
    pub z: f64,
    pub h: HMap,
    /// Mass beyond +-W bins (clipped), with its first moment (bins).
    pub tail: f64,
}

impl Hist {
    pub fn mass(&self) -> f64 {
        self.z + self.h.values().sum::<f64>() + self.tail
    }
    /// Rescale to a binade `k` above (k > 0: values shrink) or below.
    pub fn rescale(&self, k: i32) -> Hist {
        self.rescale_half(k, 0)
    }

    /// `rescale`, keeping inexact values off the tie bins (nonzero multiples of `half`, the
    /// half-ulp bin; 0: no such care).
    pub fn rescale_half(&self, k: i32, half: i64) -> Hist {
        if k == 0 {
            return self.clone();
        }
        let mut o = Hist {
            z: self.z,
            tail: self.tail,
            ..Default::default()
        };
        for (&i, &p) in &self.h {
            let j = if k > 0 {
                let d = 1i64 << k.min(62);
                quant(snap(i as f64 / d as f64, half))
            } else {
                quant(i.saturating_mul(1i64 << (-k).min(62)))
            };
            if j.abs() > W_MAX {
                o.tail += p;
            } else {
                *o.h.entry(j).or_default() += p;
            }
        }
        o
    }
}

/// Nearest bin to v, except that a nonzero v never maps to 0 and (with `half` > 0) an
/// inexact v never maps to a nonzero multiple of `half` (it moves one bin toward v).
#[inline]
pub fn snap(v: f64, half: i64) -> i64 {
    let k = v.round() as i64;
    let off = v - k as f64;
    if off == 0.0 {
        return k;
    }
    if k == 0 || (half > 0 && k % half == 0) {
        return k + if off > 0.0 { 1 } else { -1 };
    }
    k
}

/// Result of the chain: per final cell, its probability, the error law (with the final
/// rounding) and the pre-final deviation law, in units of ulp(final binade) / G.
pub struct ChainLaw {
    pub cells: Cells,
    pub fin: Vec<(usize, f64, Hist, Hist)>,
    /// Residue diagnostics: (step, max Fourier coefficient, TV to uniform, weight).
    pub residue: Vec<(usize, f64, f64, f64)>,
    pub notes: Vec<String>,
    /// Histogram bins per ulp (`G` here; tier 1 uses a finer grid).
    pub g: i64,
    /// Per step: P(error so far != 0), and the law of the result's tracked low bits (tier 1).
    pub trace: Vec<(f64, Vec<f64>)>,
}

/// Run the chain.
// ASSUME(N-ERRDIST-MARKOV): local errors independent of the past given the cell.
pub fn run(sch: &ChainSchema, opts: Opts) -> Result<ChainLaw, String> {
    let mut eng = Engine {
        sch,
        opts,
        fine: FineCache::default(),
        clouds: CloudCache::default(),
        local: HashMap::default(),
        prev: HashMap::default(),
    };
    // Binade range from the terms' magnitudes (with headroom for sums).
    let (mut lo, mut hi) = (i32::MAX, i32::MIN);
    for st in &sch.steps {
        for g in &st.pop.groups {
            for (_, t) in &g.parts {
                for a in &t.atoms {
                    if a.0 != 0.0 && a.2 > 1e-14 {
                        let b = super::spec::exp_floor(a.0);
                        lo = lo.min(b);
                        hi = hi.max(b);
                    }
                }
            }
        }
    }
    if lo == i32::MAX {
        return Err("every term is zero".into());
    }
    let terms: usize = sch
        .steps
        .iter()
        .map(|s| s.pop.terms())
        .sum::<usize>()
        .max(1);
    let cells = Cells {
        bmin: lo - 8,
        bmax: hi + (usize::BITS - terms.leading_zeros()) as i32 + 2,
    };
    let nc = cells.n();
    let mut notes = vec![];
    // State: per cell, probability and histogram; the accumulator cloud (exact partial sums).
    let mut acc_cloud = Cloud::point(0.0);
    let mut state: Vec<Hist> = vec![Hist::default(); nc];
    state[0].z = 1.0;
    let mut lattice = i32::MAX;
    let mut residue = vec![];
    let mut fin = vec![];
    let ns = sch.steps.len();
    let dbg = std::env::var("PTXV_ERRDIST_DEBUG").is_ok();
    let t0 = std::time::Instant::now();
    for (si, st) in sch.steps.iter().enumerate() {
        let last = si + 1 == ns;
        // Measures and local laws are per step (keys hold the step's population / index).
        eng.fine.meas.clear();
        eng.fine.terms.clear();
        eng.fine.taupw.clear();
        eng.local.clear();
        if dbg {
            eprintln!(
                "step {si}: {:.2}s, {} local laws, {} term stats, acc atoms {}",
                t0.elapsed().as_secs_f64(),
                eng.local.len(),
                eng.fine.terms.len(),
                acc_cloud.a.len()
            );
        }
        let (amin, amax) = (st.pop.min_align(), st.pop.max_align());
        // Blocks restricted to alignment <= e, for every e that can be emax.
        let mut blocks: HashMap<i32, Signed> = HashMap::default();
        let mut get_block = |eng: &mut Engine, e: i32| -> Signed {
            let e = e.clamp(amin - 1, amax);
            blocks
                .entry(e)
                .or_insert_with(|| eng.clouds.block(&sch.steps[si].pop, e))
                .clone()
        };
        // Accumulator atoms by cell.
        let mut by_cell: Vec<Vec<(f64, f64)>> = vec![vec![]; nc];
        for &(x, p) in &acc_cloud.a {
            by_cell[cells.of(x)].push((x, p));
        }
        let mut next: Vec<Hist> = vec![Hist::default(); nc];
        let mut dbg_nz = (0.0, 0.0, 0.0);
        let mut dbg_tab: std::collections::BTreeMap<(i32, i32), (f64, f64)> = Default::default();
        let mut fin_pre: Vec<Hist> = vec![Hist::default(); nc];
        for cx in 0..nc {
            let px: f64 = by_cell[cx].iter().map(|a| a.1).sum();
            if px <= 1e-14 || state[cx].mass() <= 0.0 {
                continue;
            }
            let bx = (cx != 0).then(|| cells.binade(cx));
            let sx = cells.sign(cx);
            // emax candidates.
            let es: Vec<i32> = match bx {
                Some(b) => std::iter::once(b).chain((b + 1)..=amax).collect(),
                None => ((amin)..=amax).collect(),
            };
            // Joint (E, result cell) masses.
            let mut joint: Vec<(i32, usize, f64)> = vec![];
            for &e in &es {
                let meas = match bx {
                    Some(b) if b == e => get_block(&mut eng, e),
                    _ => {
                        let a = get_block(&mut eng, e);
                        let b = get_block(&mut eng, e - 1);
                        Signed {
                            atoms: a
                                .atoms
                                .iter()
                                .copied()
                                .chain(b.atoms.iter().map(|&(v, p)| (v, -p)))
                                .collect(),
                            merged: a.merged || b.merged,
                        }
                    }
                };
                let mut cm = vec![0.0; nc];
                let smooth = meas.merged || acc_cloud.merged;
                for &(x, p) in &by_cell[cx] {
                    for &(y, w) in &meas.atoms {
                        let s = x + y;
                        let h = SMOOTH * (x.abs() + y.abs());
                        if smooth && s.abs() < 2.0 * h {
                            spread(&cells, s, h, p * w, &mut cm);
                        } else {
                            cm[cells.of(s)] += p * w;
                        }
                    }
                }
                for (cs, m) in cm.into_iter().enumerate() {
                    if m > 1e-15 * px {
                        joint.push((e, cs, m / px));
                    }
                }
            }
            let tot: f64 = joint.iter().map(|j| j.2).sum();
            if tot <= 0.0 {
                continue;
            }
            // Mix the local laws of every emax per result cell, then convolve once.
            let mut per_cs: HashMap<usize, (Local, Local)> = HashMap::default();
            let hm = state[cx].mass();
            for (e, cs, w) in joint {
                let w = w / tot;
                if w * hm < 1e-13 {
                    continue;
                }
                if cs == 0 {
                    // The exact sum is 0: the result is exactly 0 (the truncations of a sum
                    // that cancels exactly are taken as zero).
                    next[0].z += w * hm;
                    continue;
                }
                let bs = cells.binade(cs);
                let ss = cells.sign(cs);
                let acc = AccCtx {
                    b: bx,
                    sign: sx,
                    lattice: if lattice == i32::MAX {
                        st.pop.lattice()
                    } else {
                        lattice.min(st.pop.lattice())
                    },
                };
                let l = eng.local(si, acc, e, bs, ss, false);
                if l.r > 1 {
                    residue.push((si, l.max_fourier, l.tv, w * hm));
                }
                if dbg {
                    dbg_nz.0 += w * hm * (1.0 - l.p0);
                    dbg_nz.1 += w * hm;
                    dbg_nz.2 += w * hm * l.m2;
                    let key = (bx.map_or(-999, |b| bs - b), (bs - e));
                    let ent = dbg_tab.entry(key).or_insert((0.0, 0.0));
                    ent.0 += w * hm;
                    ent.1 += w * hm * (1.0 - l.p0);
                }
                let ent = per_cs.entry(cs).or_default();
                mix_into(&mut ent.0, &l, w);
                if last {
                    let lp = eng.local(si, acc, e, bs, ss, true);
                    mix_into(&mut ent.1, &lp, w);
                }
            }
            for (cs, (l, lp)) in per_cs {
                let k = cells.binade(cs) - bx.unwrap_or(cells.binade(cs));
                let h = state[cx].rescale(k);
                convolve_into(&mut next[cs], &h, &l, 1.0);
                if last {
                    convolve_into(&mut fin_pre[cs], &h, &lp, 1.0);
                }
            }
        }
        // Memory guard: the error histograms of all cells.
        let entries: usize = next
            .iter()
            .chain(fin_pre.iter())
            .map(|h| h.h.len())
            .sum::<usize>()
            + eng.local.len() * 64;
        if entries > MAX_ENTRIES {
            return Err(format!(
                "tier 1 state too large at step {si} ({entries} histogram entries > {MAX_ENTRIES}); use tier 3"
            ));
        }
        for h in next.iter_mut().chain(fin_pre.iter_mut()) {
            let m = h.mass();
            let mut drop = 0.0;
            h.h.retain(|_, p| {
                let keep = *p > 1e-13 * m;
                if !keep {
                    drop += *p;
                }
                keep
            });
            h.tail += drop;
        }
        if dbg {
            eprintln!(
                "step {si} model P(e != 0) = {:.4}, E e^2 = {:.4}",
                dbg_nz.0 / dbg_nz.1,
                dbg_nz.2 / dbg_nz.1
            );
            for (k, v) in &dbg_tab {
                if v.0 > 0.01 * dbg_nz.1 {
                    eprintln!(
                        "    bS-bX {:>4} bS-E {:>3}: w {:.3} P(e!=0) {:.3}",
                        k.0,
                        k.1,
                        v.0 / dbg_nz.1,
                        v.1 / v.0
                    );
                }
            }
        }
        // Advance the exact partial sums.
        let full = get_block(&mut eng, amax);
        if dbg {
            let m: f64 = full.atoms.iter().map(|a| a.1).sum();
            let v: f64 = full.atoms.iter().map(|a| a.0 * a.0 * a.1).sum::<f64>();
            let mut sq = 0.0;
            for g in &st.pop.groups {
                for (w, t) in &g.parts {
                    sq += w * g.n as f64 * t.atoms.iter().map(|a| a.0 * a.0 * a.2).sum::<f64>();
                }
            }
            eprintln!(
                "block: mass {m:.6} var {v:.4} exact var {sq:.4}; atoms {}",
                full.atoms.len()
            );
        }
        let was = acc_cloud.merged || full.merged;
        let mut sums = Vec::with_capacity(acc_cloud.a.len() * full.atoms.len());
        for &(x, p) in &acc_cloud.a {
            for &(y, w) in &full.atoms {
                super::laws::push_sum(&mut sums, x, y, p * w, was);
            }
        }
        acc_cloud = Cloud::from_atoms(sums, CAP);
        acc_cloud.merged |= was;
        acc_cloud.a.retain(|a| a.1 > 0.0);
        let m = acc_cloud.mass();
        for a in acc_cloud.a.iter_mut() {
            a.1 /= m;
        }
        lattice = lattice.min(st.pop.lattice());
        state = next;
        if last {
            for c in 0..nc {
                let p = state[c].mass();
                if p > 0.0 {
                    fin.push((c, p, state[c].clone(), fin_pre[c].clone()));
                }
            }
        }
    }
    // Final cell masses from y's exact characteristic function (N-ERRDIST-CF).
    if let Some(m) = super::cf::cell_masses(sch, &cells) {
        let mut worst = 0f64;
        for f in fin.iter_mut() {
            if f.0 == 0 || f.1 <= 0.0 {
                continue;
            }
            let k = m[f.0] / f.1;
            if m[f.0] > 1e-6 {
                worst = worst.max((k - 1.0).abs());
            }
            for h in [&mut f.2, &mut f.3] {
                h.z *= k;
                h.tail *= k;
                for p in h.h.values_mut() {
                    *p *= k;
                }
            }
            f.1 = m[f.0];
        }
        let tot: f64 = fin.iter().map(|f| f.1).sum();
        for f in fin.iter_mut() {
            let k = 1.0 / tot;
            f.1 *= k;
            for h in [&mut f.2, &mut f.3] {
                h.z *= k;
                h.tail *= k;
                for p in h.h.values_mut() {
                    *p *= k;
                }
            }
        }
        notes.push(format!("final (sign, binade) masses of y from its characteristic function (largest relative change {worst:.2} over cells above 1e-6)"));
    }
    let tail: f64 = fin.iter().map(|f| f.2.tail).sum();
    if tail > 1e-9 {
        notes.push(format!(
            "{tail:.2e} of the mass beyond 2^52 ulps or below 1e-15 of a cell (clipped)"
        ));
    }
    Ok(ChainLaw {
        cells,
        fin,
        residue,
        notes,
        g: G,
        trace: vec![],
    })
}

/// Add w times a local law to an (unnormalized) mixture.
pub(crate) fn mix_into(o: &mut Local, l: &Local, w: f64) {
    o.p0 += w * l.p0;
    let mut m: HashMap<i64, f64> = o.nz.drain(..).collect();
    for &(i, p) in &l.nz {
        *m.entry(i).or_default() += w * p;
    }
    o.nz = m.into_iter().collect();
    o.m1 += w * l.m1;
    o.m2 += w * l.m2;
}

pub(crate) fn convolve_into(o: &mut Hist, h: &Hist, l: &Local, w: f64) {
    // Exact zero stays zero only if the local error is zero.
    o.z += w * h.z * l.p0;
    o.tail += w * h.tail * (l.p0 + l.nz.iter().map(|x| x.1).sum::<f64>());
    for &(j, q) in &l.nz {
        let v = quant(j);
        if v.abs() > W_MAX {
            o.tail += w * h.z * q;
        } else {
            *o.h.entry(v).or_default() += w * h.z * q;
        }
    }
    for (&i, &p) in &h.h {
        // Nonzero + zero local error.
        *o.h.entry(i).or_default() += w * p * l.p0;
        for &(j, q) in &l.nz {
            let v = quant(i + j);
            if v == 0 {
                // An exact cancellation of two nonzero errors.
                o.z += w * p * q;
            } else if v.abs() > W_MAX {
                o.tail += w * p * q;
            } else {
                *o.h.entry(v).or_default() += w * p * q;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errdist::chain::StepModel;
    use crate::errdist::fine::Group;
    use crate::errdist::spec::Dist;
    use crate::errdist::spec::Format;
    use crate::errdist::special::Rng;

    pub(super) fn bf16_step(n: usize) -> (ChainSchema, crate::errdist::laws::Pmf) {
        let pm = crate::errdist::laws::code_pmf(
            &Dist::Normal {
                mean: 0.0,
                std: 1.0,
            },
            Format::Bf16,
            1.0,
        );
        let md = crate::fp::DEFAULT_MMA;
        let t = Rc::new(crate::errdist::schema::product_law(
            &pm,
            &pm,
            (1.0, 0),
            (1.0, 0),
            StepModel::Mma(md),
        ));
        let st = StepSchema {
            model: StepModel::Mma(md),
            frac: 25,
            pop: Pop {
                groups: vec![Group {
                    n,
                    parts: vec![(1.0, t)],
                }],
            },
            fp4: false,
        };
        (
            ChainSchema {
                steps: vec![st],
                round16: None,
                pruned: 0.0,
                notes: vec![],
            },
            pm,
        )
    }

    /// One bf16 step with no accumulator: P(local error != 0 | emax, result binade) from the
    /// engine against sampled steps of the model.
    #[test]
    #[ignore]
    fn first_step_local_law_matches_sampling() {
        let (sch, _) = bf16_step(16);
        let mut eng = Engine {
            sch: &sch,
            opts: Opts { uniform: false },
            fine: FineCache::default(),
            clouds: CloudCache::default(),
            local: HashMap::default(),
            prev: HashMap::default(),
        };
        let mut r = Rng::new(1);
        let mut tab: std::collections::BTreeMap<(i32, i32), (f64, f64)> = Default::default();
        let md = crate::fp::DEFAULT_MMA;
        let n = 400_000;
        for _ in 0..n {
            let t: Vec<crate::fp::Term> = (0..16)
                .map(|_| {
                    let a = Format::Bf16.round(r.normal(), false).0;
                    let b = Format::Bf16.round(r.normal(), false).0;
                    crate::fp::Term::prod(a, b, true)
                })
                .collect();
            let hw = md.apply_terms(0.0, &t) as f64;
            let ex: f64 = t.iter().map(|x| x.v).sum(); // exact enough to tell zero from not
            let mut acc = crate::errdist::chain::ExpAcc::default();
            for x in &t {
                acc.add(x.v);
            }
            acc.add(-hw);
            let e = t.iter().filter(|x| x.v != 0.0).map(|x| x.e).max().unwrap();
            let bs = crate::errdist::spec::exp_floor(ex);
            let ent = tab.entry((e, bs - e)).or_insert((0.0, 0.0));
            ent.0 += 1.0;
            ent.1 += (!acc.is_zero()) as u8 as f64;
        }
        for ((e, d), (c, nz)) in tab {
            if c < 5000.0 {
                continue;
            }
            let l = eng.local(
                0,
                AccCtx {
                    b: None,
                    sign: 0.0,
                    lattice: -200,
                },
                e,
                e + d,
                1.0,
                false,
            );
            eprintln!(
                "E {e:>3} bS-E {d:>3}: n {c:>7} MC P(e!=0) {:.4}  model {:.4}",
                nz / c,
                1.0 - l.p0
            );
        }
    }

    /// The second bf16 step (accumulator from the first): P(local error != 0 | acc binade,
    /// emax, result binade) against sampling.
    #[test]
    #[ignore]
    fn second_step_local_law_matches_sampling() {
        let (mut sch, _) = bf16_step(16);
        sch.steps.push(sch.steps[0].clone());
        let mut eng = Engine {
            sch: &sch,
            opts: Opts { uniform: false },
            fine: FineCache::default(),
            clouds: CloudCache::default(),
            local: HashMap::default(),
            prev: HashMap::default(),
        };
        let mut r = Rng::new(2);
        let mut tab: std::collections::BTreeMap<(i32, i32, i32, i8), (f64, f64)> =
            Default::default();
        let md = crate::fp::DEFAULT_MMA;
        let terms = |r: &mut Rng| -> Vec<crate::fp::Term> {
            (0..16)
                .map(|_| {
                    let a = Format::Bf16.round(r.normal(), false).0;
                    let b = Format::Bf16.round(r.normal(), false).0;
                    crate::fp::Term::prod(a, b, true)
                })
                .collect()
        };
        for _ in 0..600_000 {
            let t1 = terms(&mut r);
            let x = md.apply_terms(0.0, &t1);
            let t2 = terms(&mut r);
            let hw = md.apply_terms(x, &t2) as f64;
            let mut acc = crate::errdist::chain::ExpAcc::default();
            acc.add(x as f64);
            for t in &t2 {
                acc.add(t.v);
            }
            let ex = acc.approx();
            acc.add(-hw);
            if x == 0.0 || ex == 0.0 {
                continue;
            }
            let bx = crate::errdist::spec::exp_floor(x as f64);
            let e = t2
                .iter()
                .filter(|t| t.v != 0.0)
                .map(|t| t.e)
                .max()
                .unwrap()
                .max(bx);
            let bs = crate::errdist::spec::exp_floor(ex);
            let ent = tab
                .entry((bx, e, bs, (x as f64).signum() as i8 * ex.signum() as i8))
                .or_insert((0.0, 0.0));
            ent.0 += 1.0;
            ent.1 += (!acc.is_zero()) as u8 as f64;
        }
        for ((bx, e, bs, sg), (c, nz)) in tab {
            if c < 8000.0 {
                continue;
            }
            let l = eng.local(
                1,
                AccCtx {
                    b: Some(bx),
                    sign: 1.0,
                    lattice: -200,
                },
                e,
                bs,
                sg as f64,
                false,
            );
            eprintln!(
                "bX {bx:>3} E {e:>3} bS {bs:>3} sgn {sg:>2}: n {c:>7} MC P(e!=0) {:.4}  model {:.4}",
                nz / c,
                1.0 - l.p0
            );
        }
    }
}

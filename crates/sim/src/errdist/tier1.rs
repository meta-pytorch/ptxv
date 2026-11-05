/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tier 1, version 2 (docs/errdist.md §5.2): the error of an accumulation chain as a Markov
//! chain on the exact partial sum's (sign, binade) cell.
//!
//! - Coarse layer: the exact partial sums' law as point masses on a power-of-two grid (FFT
//!   convolutions; exact on lattices when the grid spacing is the lattice). The joint law of
//!   (accumulator cell, emax, result cell) per step comes from convolving the accumulator law
//!   restricted to each cell with the step's block law restricted to max alignment = emax
//!   (inclusion-exclusion over the order statistic).
//! - Fine layer (`steplaw`): the local error given (emax, result cell, accumulator cell) from
//!   each term's joint law of (truncated value mod r, dropped part) on Z_r x a dropped-part
//!   grid, exactly convolved over the step's terms; the accumulator's own residue from the
//!   previous steps' terms (N-ERRDIST-RESIDUE).
//! - The error state: per cell, a histogram in ulp / 16 of the cell, rescaled between cells.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::Arc as Rc;

use super::fine::C64;
use super::fine::NO_ALIGN;
use super::fine::TermLaw;
use super::fine::fft;
use super::markov::Cells;
use super::markov::ChainLaw;
/// Error bins per ulp: fine enough that an exact half-ulp error and one just beyond it (an
/// earlier, smaller error of the same sign) land in different bins.
pub const G: i64 = 64;
use super::markov::Hist;
use super::markov::Local;
use super::markov::snap;
use super::markov::ulp_exp;
use super::schema::ChainSchema;
use super::schema::StepSchema;
use super::spec::exp_floor;

#[derive(Clone, Copy, Debug, Default)]
pub struct Opts {
    /// Replace every residue law by the uniform law (the smooth-density heuristic).
    pub uniform: bool,
}

// ------------------------------------------------------------------------------------ grid

/// Point masses at x_i = (i - off) h, i in 0..n (n a power of two).
#[derive(Clone, Debug)]
pub struct Grid {
    pub h: f64,
    pub lh: i32,
    pub n: usize,
    pub off: i64,
    /// Every value of every term is a multiple of h: the grid is exact.
    pub exact: bool,
}

impl Grid {
    /// Pick the grid for a chain: the lattice when it resolves the final sum, else
    /// sd / 2^10; the range covers |mean| + 12 sd of the final sum (plus the largest term).
    pub fn for_chain(sch: &ChainSchema) -> Grid {
        let (mut mean, mut var, mut big) = (0.0, 0.0, 0f64);
        let mut lat = i32::MAX;
        for st in &sch.steps {
            for g in &st.pop.groups {
                let (mut e1, mut e2) = (0.0, 0.0);
                for (w, t) in &g.parts {
                    lat = lat.min(t.lattice);
                    for a in &t.atoms {
                        e1 += w * a.2 * a.0;
                        e2 += w * a.2 * a.0 * a.0;
                        if a.2 > 1e-12 {
                            big = big.max(a.0.abs());
                        }
                    }
                }
                mean += g.n as f64 * e1;
                var += g.n as f64 * (e2 - e1 * e1).max(0.0);
            }
        }
        let sd = var.max(1e-300).sqrt();
        let span = mean.abs() + 12.0 * sd + 2.0 * big;
        let cont = (sd / 1024.0).log2().floor() as i32;
        let (lh, exact) = if lat != i32::MAX && lat >= cont - 3 && span / 2f64.powi(lat) < 4e5 {
            (lat, true)
        } else {
            (cont, false)
        };
        let h = 2f64.powi(lh);
        let n = ((2.0 * span / h) as usize + 4)
            .next_power_of_two()
            .clamp(256, 1 << 18);
        Grid {
            h,
            lh,
            n,
            off: (n / 2) as i64,
            exact,
        }
    }
    fn x(&self, i: usize) -> f64 {
        (i as i64 - self.off) as f64 * self.h
    }
}

fn fwd(v: &[f64]) -> Vec<C64> {
    let mut f: Vec<C64> = v.iter().map(|&x| C64::new(x, 0.0)).collect();
    fft(&mut f, false);
    f
}

fn inv(f: &[C64]) -> Vec<f64> {
    let mut g = f.to_vec();
    fft(&mut g, true);
    let n = g.len() as f64;
    // The point masses are a convolution of shifted laws: undo the extra offset below.
    g.iter().map(|c| c.re / n).collect()
}

/// Map from grid index to cell, and the "tiny" region (cells narrower than 8 h on a
/// continuous grid): its mass is shared among the tiny cells by width.
struct CellMap {
    of: Vec<usize>,
    /// Binade below which cells are tiny (continuous grids); cells < bt get mass by width.
    bt: i32,
}

impl CellMap {
    fn new(grid: &Grid, cells: &Cells) -> CellMap {
        let bt = if grid.exact { i32::MIN } else { grid.lh + 3 };
        let of = (0..grid.n)
            .map(|i| {
                let x = grid.x(i);
                if x == 0.0 { 0 } else { cells.of(x) }
            })
            .collect();
        CellMap { of, bt }
    }
    /// Cell masses of a grid law (and the exact-zero mass `z` it holds at 0 on an exact grid).
    fn masses(&self, grid: &Grid, cells: &Cells, v: &[f64], zero: f64, out: &mut [f64]) {
        for x in out.iter_mut() {
            *x = 0.0;
        }
        if grid.exact {
            for (i, &p) in v.iter().enumerate() {
                out[self.of[i]] += p;
            }
            return;
        }
        // Continuous grid: points with |x| < 2^bt form the tiny region; `zero` of the mass at
        // 0 is exact zeros.
        let lim = 2f64.powi(self.bt);
        let (mut tp, mut tn) = (0.0, 0.0);
        let z0 = zero.min(v[grid.off as usize].max(0.0));
        out[0] += z0;
        for (i, &p) in v.iter().enumerate() {
            let x = grid.x(i);
            if x.abs() < lim {
                if x > 0.0 {
                    tp += p;
                } else if x < 0.0 {
                    tn += p;
                } else {
                    tp += (p - z0) / 2.0;
                    tn += (p - z0) / 2.0;
                }
            } else {
                out[self.of[i]] += p;
            }
        }
        // Share by width: cell b (b < bt) has width 2^b (the lowest cell 2^(bmin + 1)).
        let top = self.bt.min(cells.bmax + 1);
        for b in cells.bmin..top {
            let w = if b == cells.bmin {
                2f64.powi(b + 1)
            } else {
                2f64.powi(b)
            } / lim;
            let w = if b == cells.bmin { w.min(1.0) } else { w };
            out[cells.of(1.5 * 2f64.powi(b))] += tp * w;
            out[cells.of(-1.5 * 2f64.powi(b))] += tn * w;
        }
    }
}

// --------------------------------------------------------------------------- step laws

/// Low bits of the accumulator's magnitude (in its ulps) carried exactly in the chain state:
/// they decide the local error when the next result is in a higher binade, and they are
/// correlated with the error so far (the same fine terms set both).
pub const K_BITS: u32 = 2;
const NL: usize = 1 << K_BITS;

/// The accumulator's fine state for a step when it is finer than Q: its residue law (Q
/// units, mod the step's modulus) and its dropped part (below Q, signed bins).
#[derive(Clone, Debug)]
struct AccFine {
    res: Vec<f64>,
    tau: Vec<(i64, f64)>,
    tau0: f64,
}

/// How the accumulator enters a step's local law.
#[derive(Clone, Debug)]
enum AccIn {
    /// Exactly zero.
    Zero,
    /// On the Q grid: residue sx ((l mod 2^known) + 2^known H) shift (Q units, mod the
    /// modulus), l the tracked low bits of the magnitude, H (its next bits) ~ `upper[l]`
    /// (empty: H = 0).
    Lattice {
        sx: i8,
        shift: usize,
        known: u32,
        upper: Vec<Vec<f64>>,
    },
    /// Finer than Q (terms dominate): residue and dropped part from the previous terms.
    Fine(AccFine),
}

impl AccIn {
    fn key(&self) -> u64 {
        use std::hash::Hash;
        use std::hash::Hasher;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        match self {
            AccIn::Zero => 0u8.hash(&mut h),
            AccIn::Lattice {
                sx,
                shift,
                known,
                upper,
            } => {
                (1u8, *sx, *shift, *known).hash(&mut h);
                for u in upper {
                    u.len().hash(&mut h);
                    for x in u {
                        x.to_bits().hash(&mut h);
                    }
                }
            }
            AccIn::Fine(a) => {
                2u8.hash(&mut h);
                for x in &a.res {
                    x.to_bits().hash(&mut h);
                }
                for x in &a.tau {
                    (x.0, x.1.to_bits()).hash(&mut h);
                }
                a.tau0.to_bits().hash(&mut h);
            }
        }
        h.finish()
    }
}

/// Per (term law, q, modulus, r, e): the term's restricted mass, its law of
/// (a = trunc(t / Q) mod modulus, dropped part in Q / tsc(r) bins), and the residue law of
/// the terms with no dropped part.
struct TermFine {
    mass: f64,
    /// (a, tau bin) -> p
    joint: HashMap<(usize, i64), f64>,
    /// P(a, tau = 0)
    res0: Vec<f64>,
}

/// Dropped parts are binned in units of Q / tsc(r): Q / 4 while that is finer than ulp / G,
/// else ulp / G (nonzero parts below one bin keep their sign as +-1 bin).
fn tsc(r: usize) -> f64 {
    if r as i64 <= G / 4 {
        4.0
    } else {
        G as f64 / r as f64
    }
}

fn term_fine(t: &TermLaw, q: i32, m: usize, r: usize, e: i32) -> TermFine {
    let qq = 2f64.powi(q);
    let ts = tsc(r);
    let mut tf = TermFine {
        mass: 0.0,
        joint: HashMap::new(),
        res0: vec![0.0; m],
    };
    for a in t.atoms.iter().filter(|a| a.1 <= e) {
        tf.mass += a.2;
        let tr = (a.0 / qq).trunc();
        let tau = a.0 - tr * qq;
        let res = (tr as i64).rem_euclid(m as i64) as usize;
        if tau == 0.0 {
            tf.res0[res] += a.2;
            *tf.joint.entry((res, 0)).or_default() += a.2;
        } else {
            let mut b = (tau / qq * ts).round() as i64;
            if b == 0 {
                b = if tau > 0.0 { 1 } else { -1 };
            }
            *tf.joint.entry((res, b)).or_default() += a.2;
        }
    }
    tf
}

/// 2D transform shape: residue axis r (power of two), dropped-part axis tw.
#[derive(Clone, Copy)]
struct Shape {
    r: usize,
    tw: usize,
}

fn fft2(a: &mut [C64], s: Shape, invert: bool) {
    // rows (tau axis) then columns (residue axis); layout a[res * tw + tau]
    for row in a.chunks_mut(s.tw) {
        fft(row, invert);
    }
    let mut col = vec![C64::default(); s.r];
    for j in 0..s.tw {
        for i in 0..s.r {
            col[i] = a[i * s.tw + j];
        }
        fft(&mut col, invert);
        for i in 0..s.r {
            a[i * s.tw + j] = col[i];
        }
    }
}

fn pow_c(c: C64, n: usize) -> C64 {
    if n == 0 {
        return C64::new(1.0, 0.0);
    }
    let m = c.abs();
    if m == 0.0 {
        return C64::new(0.0, 0.0);
    }
    let a = c.im.atan2(c.re) * n as f64;
    let mm = m.powi(n as i32);
    C64::new(mm * a.cos(), mm * a.sin())
}

/// A block transform's key: (q, modulus, r, e, tw).
type BlockKey = (i32, usize, usize, i32, usize);
/// (joint transform, tau = 0 residue transform, mass)
type Block = Rc<(Vec<C64>, Vec<C64>, f64)>;
type TfMap = HashMap<(usize, i32, usize, usize, i32), Rc<TermFine>>;

/// The step's terms restricted to alignment <= e: transform of the joint (sum a mod m,
/// sum of dropped parts) law (unnormalized: total = P(all <= e)), and the transform of the
/// residue law restricted to every dropped part zero (P(all <= e, all tau = 0, a)).
fn block_fine(st: &StepSchema, k: BlockKey, tfc: &mut TfMap) -> Block {
    let (q, m, r, e, tw) = k;
    let s = Shape { r: m, tw };
    let len = s.r * s.tw;
    let mut tot = vec![C64::new(1.0, 0.0); len];
    let mut tot0 = vec![C64::new(1.0, 0.0); s.r];
    let mut mass = 1.0;
    for g in &st.pop.groups {
        let mut gt = vec![C64::default(); len];
        let mut g0 = vec![C64::default(); s.r];
        let mut gm = 0.0;
        for (w, t) in &g.parts {
            let tf = tfc
                .entry((Rc::as_ptr(t) as usize, q, m, r, e))
                .or_insert_with(|| Rc::new(term_fine(t, q, m, r, e)))
                .clone();
            if tf.mass <= 0.0 {
                continue;
            }
            let mut a = vec![C64::default(); len];
            for (&(res, b), &p) in &tf.joint {
                let j = b.rem_euclid(s.tw as i64) as usize;
                a[res * s.tw + j].re += p / tf.mass;
            }
            fft2(&mut a, s, false);
            let mn = tf.mass.powi(g.n as i32);
            for (x, y) in gt.iter_mut().zip(&a) {
                *x = x.add(pow_c(*y, g.n).scale(w * mn));
            }
            let mut f0: Vec<C64> = tf
                .res0
                .iter()
                .map(|&p| C64::new(p / tf.mass, 0.0))
                .collect();
            fft(&mut f0, false);
            for (x, y) in g0.iter_mut().zip(&f0) {
                *x = x.add(pow_c(*y, g.n).scale(w * mn));
            }
            gm += w * mn;
        }
        for (x, y) in tot.iter_mut().zip(&gt) {
            *x = x.mul(*y);
        }
        for (x, y) in tot0.iter_mut().zip(&g0) {
            *x = x.mul(*y);
        }
        mass *= gm;
    }
    Rc::new((tot, tot0, mass))
}

/// Largest residue modulus handled exactly; beyond, residues are folded (`r` capped).
const R_CAP: usize = 1024;

/// One step's local-law arguments: emax e (from the accumulator or the terms), the Q
/// exponent, the result cell (binade, sign) and the accumulator.
#[derive(Clone, Debug)]
struct StepArgs {
    e: i32,
    from_acc: bool,
    q: i32,
    bs: i32,
    ss: f64,
    acc: AccIn,
    /// Tracked low-bit values (NL, or 1: none).
    nl: usize,
}

/// (r = ulp_S / Q, modulus r 2^K, tau axis width, one tau bin in ulp_S / G bins).
fn shape_of(st: &StepSchema, q: i32, bs: i32, nl: usize) -> (usize, usize, usize, f64) {
    let ue = ulp_exp(bs);
    let r = if ue > q { 1usize << (ue - q) } else { 1 };
    let ts = tsc(r);
    let n_terms: usize = st.pop.groups.iter().map(|g| g.n).sum::<usize>() + 1;
    let tw = (2 * (n_terms * (ts.ceil() as usize + 1) + 4) + 1).next_power_of_two();
    (r, r * nl, tw, G as f64 / (r as f64 * ts))
}

fn block_keys(st: &StepSchema, a: &StepArgs) -> Vec<BlockKey> {
    let (r, m, tw, _) = shape_of(st, a.q, a.bs, a.nl);
    let mut v = vec![(a.q, m, r, a.e, tw)];
    if !a.from_acc {
        v.push((a.q, m, r, a.e - 1, tw));
    }
    v
}

/// A step's local laws per tracked accumulator bits l and result bits l': t[l][l'] = (error
/// with the final rounding, without it; units ulp_S / G), unnormalized (sum over l' = 1).
pub(crate) struct Trans {
    t: Vec<Vec<(Local, Local)>>,
}

fn steptrans(
    st: &StepSchema,
    a: &StepArgs,
    blocks: &HashMap<BlockKey, Block>,
    uniform: bool,
) -> Trans {
    let (r, m, tw, tbin) = shape_of(st, a.q, a.bs, a.nl);
    let nl = a.nl;
    let sh = Shape { r: m, tw };
    let hi = &blocks[&(a.q, m, r, a.e, tw)];
    let (mut tt, mut tt0, mut mass) = (hi.0.clone(), hi.1.clone(), hi.2);
    if !a.from_acc {
        let lo = &blocks[&(a.q, m, r, a.e - 1, tw)];
        for (x, y) in tt.iter_mut().zip(lo.0.iter()) {
            *x = x.add(y.scale(-1.0));
        }
        for (x, y) in tt0.iter_mut().zip(lo.1.iter()) {
            *x = x.add(y.scale(-1.0));
        }
        mass -= lo.2;
    }
    if mass <= 1e-300 {
        return Trans {
            t: (0..nl)
                .map(|l| {
                    (0..nl)
                        .map(|lp| {
                            let x = Local {
                                p0: (l == lp) as u8 as f64,
                                ..Default::default()
                            };
                            (x.clone(), x)
                        })
                        .collect()
                })
                .collect(),
        };
    }
    // The accumulator: one law shared by every l (shifted by `shifts[l]`), or one per l.
    let mut accs: Vec<(Vec<C64>, Vec<C64>)> = vec![];
    let mut shifts = vec![0usize; nl];
    let mut which = vec![0usize; nl];
    let point = |res: usize, a2: &mut Vec<C64>, a0: &mut Vec<C64>, p: f64| {
        a2[res * tw].re += p;
        a0[res].re += p;
    };
    match &a.acc {
        AccIn::Zero => {
            let (mut a2, mut a0) = (vec![C64::default(); m * tw], vec![C64::default(); m]);
            point(0, &mut a2, &mut a0, 1.0);
            accs.push((a2, a0));
        }
        AccIn::Lattice {
            sx,
            shift,
            known,
            upper,
        } => {
            let sg = *sx as i64;
            let mask = (1usize << known) - 1;
            if upper.is_empty() {
                let (mut a2, mut a0) = (vec![C64::default(); m * tw], vec![C64::default(); m]);
                point(0, &mut a2, &mut a0, 1.0);
                accs.push((a2, a0));
                for (l, s) in shifts.iter_mut().enumerate() {
                    *s = (sg * ((l & mask) as i64) * (*shift as i64)).rem_euclid(m as i64) as usize;
                }
            } else {
                // Per distinct l mod 2^known.
                let mut seen: HashMap<usize, usize> = HashMap::new();
                #[allow(clippy::needless_range_loop)] // `l` also keys `seen` and the shapes
                for l in 0..nl {
                    let lm = l & mask;
                    if let Some(&i) = seen.get(&lm) {
                        which[l] = i;
                        continue;
                    }
                    let (mut a2, mut a0) = (vec![C64::default(); m * tw], vec![C64::default(); m]);
                    for (h, &p) in upper[lm].iter().enumerate() {
                        if p > 0.0 {
                            let v = (lm as i64 + ((h as i64) << known)) % m as i64;
                            let res = (sg * v * (*shift as i64)).rem_euclid(m as i64) as usize;
                            point(res, &mut a2, &mut a0, p);
                        }
                    }
                    seen.insert(lm, accs.len());
                    which[l] = accs.len();
                    accs.push((a2, a0));
                }
            }
        }
        AccIn::Fine(af) => {
            let (mut a2, mut a0) = (vec![C64::default(); m * tw], vec![C64::default(); m]);
            for (res, &pr) in af.res.iter().enumerate() {
                if pr <= 0.0 {
                    continue;
                }
                for &(b, pt) in &af.tau {
                    let j = b.rem_euclid(tw as i64) as usize;
                    a2[res * tw + j].re += pr * pt;
                }
                a0[res].re += pr * af.tau0;
            }
            accs.push((a2, a0));
        }
    }
    let nn = (m * tw) as f64;
    let laws: Vec<(Vec<f64>, Vec<f64>)> = accs
        .into_iter()
        .map(|(mut a2, mut a0)| {
            fft2(&mut a2, sh, false);
            fft(&mut a0, false);
            let mut f: Vec<C64> = tt
                .iter()
                .zip(&a2)
                .map(|(x, y)| x.mul(*y).scale(1.0 / mass))
                .collect();
            fft2(&mut f, sh, true);
            let mut joint: Vec<f64> = f.iter().map(|c| (c.re / nn).max(0.0)).collect();
            let mut f0: Vec<C64> = tt0
                .iter()
                .zip(&a0)
                .map(|(x, y)| x.mul(*y).scale(1.0 / mass))
                .collect();
            fft(&mut f0, true);
            let mut z0: Vec<f64> = f0.iter().map(|c| (c.re / m as f64).max(0.0)).collect();
            if uniform && r > 1 {
                // The heuristic: the residue uniform, independent of the dropped parts.
                let mut tau = vec![0.0; tw];
                for res in 0..m {
                    for j in 0..tw {
                        tau[j] += joint[res * tw + j];
                    }
                }
                for res in 0..m {
                    for j in 0..tw {
                        joint[res * tw + j] = tau[j] / m as f64;
                    }
                }
                let z: f64 = z0.iter().sum();
                z0.iter_mut().for_each(|x| *x = z / m as f64);
            }
            (joint, z0)
        })
        .collect();
    let mut t = Vec::with_capacity(nl);
    for l in 0..nl {
        let s = shifts[l];
        let (joint, z0) = &laws[which[l]];
        let mut nz: Vec<BTreeMap<i64, f64>> = vec![BTreeMap::new(); nl];
        let mut nzp: Vec<BTreeMap<i64, f64>> = vec![BTreeMap::new(); nl];
        let (mut zall, mut zpre, mut p0) = (vec![0.0; nl], vec![0.0; nl], vec![0.0; nl]);
        for rr in 0..m {
            let r0 = (rr + m - s) % m;
            // Residues reflect for a negative sum: |S'| / Q mod m = -(S' / Q) mod m.
            let mg = if a.ss < 0.0 { (m - rr) % m } else { rr };
            let res = mg % r;
            let lp = (mg / r) % nl;
            if res == 0 {
                p0[lp] += z0[r0];
            }
            for j in 0..tw {
                let p = joint[r0 * tw + j];
                if p <= 1e-300 {
                    continue;
                }
                let tb = if j >= tw / 2 {
                    j as i64 - tw as i64
                } else {
                    j as i64
                };
                // Error e = -sum tau - ss res Q, in ulp_S / G units.
                let v = -(tb as f64) * tbin - a.ss * res as f64 * G as f64 / r as f64;
                if res == 0 && tb == 0 {
                    zall[lp] += p;
                } else {
                    // Exact unless a dropped part is below one bin (its sign kept).
                    let k = if v == 0.0 {
                        -tb.signum()
                    } else {
                        snap(v, G / 2)
                    };
                    let k = if k == 0 { -1 } else { k };
                    *nz[lp].entry(k).or_default() += p;
                }
                if tb == 0 {
                    zpre[lp] += p;
                } else {
                    *nzp[lp].entry(snap(-(tb as f64) * tbin, 0)).or_default() += p;
                }
            }
        }
        let tot: f64 = (0..nl)
            .map(|lp| zall[lp] + nz[lp].values().sum::<f64>())
            .sum::<f64>()
            .max(1e-300);
        let row: Vec<(Local, Local)> = (0..nl)
            .map(|lp| {
                // The (0, 0) cell holds exact zeros and nonzero parts that cancel in the bins.
                let p0l = p0[lp].min(zall[lp]);
                let extra = zall[lp] - p0l;
                let mut n = std::mem::take(&mut nz[lp]);
                if extra > 0.0 {
                    *n.entry(1).or_default() += extra / 2.0;
                    *n.entry(-1).or_default() += extra / 2.0;
                }
                let post = Local {
                    p0: p0l / tot,
                    nz: n.into_iter().map(|(k, p)| (k, p / tot)).collect(),
                    r,
                    ..Default::default()
                };
                let pre = Local {
                    p0: zpre[lp] / tot,
                    nz: std::mem::take(&mut nzp[lp])
                        .into_iter()
                        .map(|(k, p)| (k, p / tot))
                        .collect(),
                    r,
                    ..Default::default()
                };
                (post, pre)
            })
            .collect();
        t.push(row);
    }
    Trans { t }
}

// --------------------------------------------------------------------- error histograms

/// Dense range of the error histograms (bins of ulp / G): |i| <= W1 kept exactly in a
/// vector, beyond in log-spaced bins (`markov::quant`).
const W1: i64 = 1024;
static STATS_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static STATS: [std::sync::atomic::AtomicU64; 6] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 6];
const ND: usize = (2 * W1 + 1) as usize;

/// An error law: exact zeros, dense bins (index i + W1, the 0 bin unused), far bins, mass
/// clipped beyond `W_MAX`.
#[derive(Clone, Debug, Default)]
struct DHist {
    z: f64,
    d: Vec<f64>,
    far: super::markov::HMap,
    tail: f64,
}

/// A mixture of local laws (one emax case each), the dense part as a vector with its
/// touched index range.
struct Mix {
    p0: f64,
    d: Vec<f64>,
    lo: usize,
    hi: usize,
    sp: Vec<(i64, f64)>,
    any: bool,
    mass: f64,
}

impl Mix {
    fn new() -> Mix {
        Mix {
            p0: 0.0,
            d: vec![0.0; ND],
            lo: usize::MAX,
            hi: 0,
            sp: vec![],
            any: false,
            mass: 0.0,
        }
    }
    fn clear(&mut self) {
        if self.lo <= self.hi {
            self.d[self.lo..=self.hi].iter_mut().for_each(|x| *x = 0.0);
        }
        self.lo = usize::MAX;
        self.hi = 0;
        self.p0 = 0.0;
        self.sp.clear();
        self.any = false;
        self.mass = 0.0;
    }
    fn add(&mut self, l: &Local, w: f64) {
        self.any = true;
        self.p0 += w * l.p0;
        self.mass += w * l.p0;
        for &(j, q) in &l.nz {
            self.mass += w * q;
            if j.abs() < W1 {
                let k = (j + W1) as usize;
                self.d[k] += w * q;
                self.lo = self.lo.min(k);
                self.hi = self.hi.max(k);
            } else {
                self.sp.push((j, w * q));
            }
        }
    }
    /// Nonzero entries (j, p).
    fn entries(&self) -> Vec<(i64, f64)> {
        let mut v: Vec<(i64, f64)> = vec![];
        if self.lo <= self.hi {
            for k in self.lo..=self.hi {
                if self.d[k] != 0.0 {
                    v.push((k as i64 - W1, self.d[k]));
                }
            }
        }
        v.extend_from_slice(&self.sp);
        v
    }
    fn dense_fft(&self) -> Option<Vec<C64>> {
        if self.lo > self.hi {
            return None;
        }
        let mut a = vec![C64::default(); NF];
        for k in self.lo..=self.hi {
            let p = self.d[k];
            if p != 0.0 {
                a[(k as i64 - W1).rem_euclid(NF as i64) as usize].re = p;
            }
        }
        fft(&mut a, false);
        Some(a)
    }
}

/// Index range of a dense part's nonzero entries (empty: lo > hi).
fn support(d: &[f64]) -> (usize, usize) {
    match (
        d.iter().position(|&x| x != 0.0),
        d.iter().rposition(|&x| x != 0.0),
    ) {
        (Some(a), Some(b)) => (a, b),
        _ => (1, 0),
    }
}

impl DHist {
    fn mass(&self) -> f64 {
        self.z + self.d.iter().sum::<f64>() + self.far.values().sum::<f64>() + self.tail
    }
    fn dense(&mut self) -> &mut Vec<f64> {
        if self.d.is_empty() {
            self.d = vec![0.0; ND];
        }
        &mut self.d
    }
    /// Add mass p at bin i (i = 0: an exact cancellation, counted as exact zero).
    fn add(&mut self, i: i64, p: f64) {
        if i == 0 {
            self.z += p;
        } else if i.abs() <= W1 {
            self.dense()[(i + W1) as usize] += p;
        } else {
            let q = super::markov::quant(i);
            if q.abs() > super::markov::W_MAX {
                self.tail += p;
            } else {
                *self.far.entry(q).or_default() += p;
            }
        }
    }
    /// The law in the bins of a binade k above (k > 0: values shrink) or below.
    fn rescale(&self, k: i32) -> DHist {
        if k == 0 {
            return self.clone();
        }
        let mut o = DHist {
            z: self.z,
            tail: self.tail,
            ..Default::default()
        };
        let f = |i: i64| -> i64 {
            if k > 0 {
                snap(i as f64 / 2f64.powi(k), G / 2)
            } else {
                i.saturating_mul(1i64 << (-k).min(62))
            }
        };
        for (j, &p) in self.d.iter().enumerate() {
            if p != 0.0 {
                o.add(f(j as i64 - W1), p);
            }
        }
        for (&i, &p) in &self.far {
            o.add(f(i), p);
        }
        o
    }
    /// o += self convolved with the mixture l. Dense x dense: direct shifted adds when small,
    /// else accumulated in the Fourier domain (`facc`, size NF; `fh` the transform of self's
    /// dense part, computed on first use). `sup`: self's dense support.
    fn conv(
        &self,
        sup: (usize, usize),
        fh: &mut Option<Option<Vec<C64>>>,
        l: &Mix,
        o: &mut DHist,
        facc: &mut Option<Vec<C64>>,
    ) {
        o.z += self.z * l.p0;
        o.tail += self.tail * l.mass;
        let hn = if sup.0 <= sup.1 { sup.1 - sup.0 + 1 } else { 0 };
        let ln = if l.lo <= l.hi { l.hi - l.lo + 1 } else { 0 };
        if STATS_ON.load(std::sync::atomic::Ordering::Relaxed) {
            use std::sync::atomic::Ordering::Relaxed;
            STATS[0].fetch_add(1, Relaxed);
            STATS[1].fetch_add(hn as u64, Relaxed);
            STATS[2].fetch_add(ln as u64, Relaxed);
            STATS[3].fetch_add(self.far.len() as u64, Relaxed);
            STATS[4].fetch_add(l.sp.len() as u64, Relaxed);
            STATS[5].fetch_add((hn * ln < 60_000) as u64, Relaxed);
        }
        if hn * ln < 60_000 {
            // Direct: h p0, z nz, h nz.
            if hn > 0 && l.p0 != 0.0 {
                let od = o.dense();
                for (x, &y) in od[sup.0..=sup.1].iter_mut().zip(&self.d[sup.0..=sup.1]) {
                    *x += l.p0 * y;
                }
            }
            if ln > 0 {
                let mut outside: Vec<(i64, f64)> = vec![];
                if self.z != 0.0 {
                    let od = o.dense();
                    for (x, &y) in od[l.lo..=l.hi].iter_mut().zip(&l.d[l.lo..=l.hi]) {
                        *x += self.z * y;
                    }
                }
                if hn > 0 {
                    let od = o.dense();
                    for jk in l.lo..=l.hi {
                        let q = l.d[jk];
                        if q == 0.0 {
                            continue;
                        }
                        let j = jk as i64 - W1;
                        // Sources k (index) with target k + j in [0, ND).
                        let a = (sup.0 as i64).max(-j) as usize;
                        let b = (sup.1 as i64).min(ND as i64 - 1 - j);
                        if (a as i64) <= b {
                            let b = b as usize;
                            let src = &self.d[a..=b];
                            let dst = &mut od[(a as i64 + j) as usize..=(b as i64 + j) as usize];
                            for (x, y) in dst.iter_mut().zip(src) {
                                *x += q * y;
                            }
                        }
                        for k in (sup.0..a.min(sup.1 + 1))
                            .chain(((b + 1).max(sup.0 as i64) as usize)..=sup.1)
                        {
                            let p = self.d[k];
                            if p != 0.0 {
                                outside.push((k as i64 - W1 + j, q * p));
                            }
                        }
                    }
                }
                for (i, p) in outside {
                    o.add(i, p);
                }
            }
        } else {
            let fl = l.dense_fft();
            let fh = fh.get_or_insert_with(|| dense_fft(&self.d)).as_deref();
            let acc = facc.get_or_insert_with(|| vec![C64::default(); NF]);
            match (fh, &fl) {
                (Some(fh), Some(fl)) => {
                    for ((a, x), y) in acc.iter_mut().zip(fh).zip(fl) {
                        // h (p0 + nz) + z nz
                        let hy = x.mul(*y);
                        *a = C64::new(
                            a.re + hy.re + l.p0 * x.re + self.z * y.re,
                            a.im + hy.im + l.p0 * x.im + self.z * y.im,
                        );
                    }
                }
                (Some(fh), None) => {
                    for (a, x) in acc.iter_mut().zip(fh) {
                        *a = C64::new(a.re + l.p0 * x.re, a.im + l.p0 * x.im);
                    }
                }
                (None, Some(fl)) => {
                    for (a, y) in acc.iter_mut().zip(fl.iter()) {
                        *a = C64::new(a.re + self.z * y.re, a.im + self.z * y.im);
                    }
                }
                (None, None) => {}
            }
        }
        // Sparse local entries (|j| >= W1: targets in log-spaced bins of width >= 2^7)
        // against the dense part bucketed by 16 bins, and the exact zeros.
        if !l.sp.is_empty() {
            let mut bk: Vec<(i64, f64)> = vec![];
            if hn > 0 {
                let mut k = sup.0;
                while k <= sup.1 {
                    let e = (k + 16).min(sup.1 + 1);
                    let (mut m, mut c) = (0.0, 0.0);
                    for x in k..e {
                        m += self.d[x];
                        c += self.d[x] * (x as f64 - W1 as f64);
                    }
                    if m > 0.0 {
                        bk.push(((c / m).round() as i64, m));
                    }
                    k = e;
                }
            }
            for &(j, q) in &l.sp {
                o.add(j, self.z * q);
                for &(c, m) in &bk {
                    o.add(c + j, q * m);
                }
            }
        }
        // Far bins against the whole local law: local errors below half the bin width keep
        // the (log-spaced) bin.
        if !self.far.is_empty() {
            let nz = l.entries();
            for (&i, &p) in &self.far {
                let sh = 63 - i.unsigned_abs().leading_zeros() as i32 - 3;
                let hw = 1i64 << (sh - 1).max(0);
                let mut small = l.p0;
                let mut big = false;
                for &(j, q) in &nz {
                    if j.abs() < hw {
                        small += q;
                    } else {
                        big = true;
                    }
                }
                o.add(i, p * small);
                if big {
                    for &(j, q) in &nz {
                        if j.abs() >= hw {
                            o.add(super::markov::quant(i + j), p * q);
                        }
                    }
                }
            }
        }
    }
    /// Add an accumulated transform (inverse, |values| below 1e-17 dropped as round-off).
    fn add_dense_fft(&mut self, mut f: Vec<C64>) {
        fft(&mut f, true);
        let n = NF as f64;
        let tot: f64 = f.iter().map(|c| c.re.abs()).sum::<f64>() / n;
        let thr = (1e-14 * tot).max(1e-18);
        for (i, c) in f.iter().enumerate() {
            let v = c.re / n;
            if v <= thr {
                continue;
            }
            let t = if i >= NF / 2 {
                i as i64 - NF as i64
            } else {
                i as i64
            };
            self.add(t, v.max(0.0));
        }
    }

    /// Exact cancellations that landed in the dense 0 bin count as zero; tiny bins dropped.
    fn finish(&mut self) {
        if !self.d.is_empty() {
            self.z += std::mem::take(&mut self.d[W1 as usize]);
        }
        let m = self.mass();
        self.far.retain(|_, p| *p > 1e-14 * m);
        for x in self.d.iter_mut() {
            if *x <= 1e-14 * m {
                *x = 0.0;
            }
        }
        if !self.d.is_empty() && self.d.iter().all(|&x| x == 0.0) {
            self.d.clear();
        }
    }
    fn merge(&mut self, h: &DHist) {
        self.z += h.z;
        self.tail += h.tail;
        if !h.d.is_empty() {
            for (x, y) in self.dense().iter_mut().zip(&h.d) {
                *x += y;
            }
        }
        for (&i, &p) in &h.far {
            *self.far.entry(i).or_default() += p;
        }
    }
    fn to_hist(&self) -> Hist {
        let mut h = Hist {
            z: self.z,
            tail: self.tail,
            ..Default::default()
        };
        for (k, &p) in self.d.iter().enumerate() {
            if p > 0.0 {
                *h.h.entry(k as i64 - W1).or_default() += p;
            }
        }
        for (&i, &p) in &self.far {
            *h.h.entry(i).or_default() += p;
        }
        h
    }
}

/// Transform size for the dense convolutions: |i + j| < 2 W1.
const NF: usize = (4 * W1) as usize;

/// The transform of a dense part (index i + W1) at size NF, or None when empty.
fn dense_fft(d: &[f64]) -> Option<Vec<C64>> {
    if d.iter().all(|&x| x == 0.0) {
        return None;
    }
    let mut a = vec![C64::default(); NF];
    for (k, &p) in d.iter().enumerate() {
        if p != 0.0 {
            a[(k as i64 - W1).rem_euclid(NF as i64) as usize].re = p;
        }
    }
    fft(&mut a, false);
    Some(a)
}

/// Map `f` over 0..n on up to `threads` threads (order kept).
fn par_map<T: Send, F: Fn(usize) -> T + Sync>(n: usize, threads: usize, f: F) -> Vec<T> {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    if n == 0 {
        return vec![];
    }
    let th = threads.clamp(1, n);
    if th == 1 {
        return (0..n).map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let parts: Vec<Vec<(usize, T)>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..th)
            .map(|_| {
                s.spawn(|| {
                    let mut v = vec![];
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        v.push((i, f(i)));
                    }
                    v
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    for p in parts {
        for (i, t) in p {
            out[i] = Some(t);
        }
    }
    out.into_iter().map(|x| x.unwrap()).collect()
}

// --------------------------------------------------------------------------- the chain

/// Transform of the block law restricted to every term aligned <= e, on the grid.
fn block_coarse(st: &StepSchema, e: i32, grid: &Grid) -> Vec<C64> {
    let mut tot = vec![C64::new(1.0, 0.0); grid.n];
    for g in &st.pop.groups {
        let mut gt = vec![C64::default(); grid.n];
        for (w, t) in &g.parts {
            let mut v = vec![0.0; grid.n];
            let mut any = false;
            for a in t.atoms.iter().filter(|a| a.1 <= e) {
                // Values relative to the grid origin: index off is 0.
                let i = ((a.0 / grid.h).round() as i64).rem_euclid(grid.n as i64) as usize;
                v[i] += a.2;
                any = true;
            }
            if !any {
                continue;
            }
            let f = fwd(&v);
            for (x, y) in gt.iter_mut().zip(&f) {
                *x = x.add(pow_c(*y, g.n).scale(*w));
            }
        }
        for (x, y) in tot.iter_mut().zip(&gt) {
            *x = x.mul(*y);
        }
    }
    tot
}

/// Residue law of the previous steps' terms (truncated at 2^gq) mod 2^kb, in 2^gq units.
fn prev_res(
    sch: &ChainSchema,
    si: usize,
    gq: i32,
    kb: u32,
    cache: &mut HashMap<(usize, i32, u32), Vec<f64>>,
) -> Vec<f64> {
    if let Some(v) = cache.get(&(si, gq, kb)) {
        return v.clone();
    }
    let m = 1usize << kb;
    let qq = 2f64.powi(gq);
    let mut tot = vec![C64::new(1.0, 0.0); m];
    for st in &sch.steps[..si] {
        for g in &st.pop.groups {
            let mut gt = vec![C64::default(); m];
            for (w, t) in &g.parts {
                let mut v = vec![0.0; m];
                let mut mass = 0.0;
                for a in &t.atoms {
                    let tr = (a.0 / qq).trunc() as i64;
                    v[tr.rem_euclid(m as i64) as usize] += a.2;
                    mass += a.2;
                }
                if mass <= 0.0 {
                    continue;
                }
                let f = fwd(&v.iter().map(|x| x / mass).collect::<Vec<_>>());
                for (x, y) in gt.iter_mut().zip(&f) {
                    *x = x.add(pow_c(*y, g.n).scale(w * mass.powi(g.n as i32)));
                }
            }
            for (x, y) in tot.iter_mut().zip(&gt) {
                *x = x.mul(*y);
            }
        }
    }
    let mut p = inv(&tot);
    for x in p.iter_mut() {
        *x = x.max(0.0);
    }
    let s: f64 = p.iter().sum();
    for x in p.iter_mut() {
        *x /= s.max(1e-300);
    }
    cache.insert((si, gq, kb), p.clone());
    p
}

/// From the law of the low `known + nu` bits of the signed exact sum (`prev_res`), the law
/// of the magnitude's next `nu` bits given its low `known` bits (indexed by those bits;
/// impossible low bits get the marginal).
fn upper_given_low(v: &[f64], known: u32, nu: u32, sx: f64) -> Vec<Vec<f64>> {
    let n = 1usize << (known + nu);
    let (nl, nh) = (1usize << known, 1usize << nu);
    let mut out = vec![vec![0.0; nh]; nl];
    for (i, &p) in v.iter().enumerate().take(n) {
        let mg = if sx < 0.0 { (n - i) % n } else { i };
        out[mg & (nl - 1)][mg >> known] += p;
    }
    let mut marg = vec![0.0; nh];
    for o in &out {
        for (h, x) in o.iter().enumerate() {
            marg[h] += x;
        }
    }
    let t: f64 = marg.iter().sum::<f64>().max(1e-300);
    marg.iter_mut().for_each(|x| *x /= t);
    for o in out.iter_mut() {
        let t: f64 = o.iter().sum();
        if t > 1e-12 {
            o.iter_mut().for_each(|x| *x /= t);
        } else {
            o.clone_from(&marg);
        }
    }
    out
}

/// `prev_res` with a shared read-only cache and a local one (for worker threads).
fn prev_res_ro(
    sch: &ChainSchema,
    si: usize,
    gq: i32,
    kb: u32,
    shared: &HashMap<(usize, i32, u32), Vec<f64>>,
    local: &mut HashMap<(usize, i32, u32), Vec<f64>>,
) -> Vec<f64> {
    if let Some(v) = shared.get(&(si, gq, kb)) {
        return v.clone();
    }
    prev_res(sch, si, gq, kb, local)
}

/// The accumulator's fine state at step `si` for a result cell: residue mod ulp_S on the Q
/// grid and its dropped part below Q.
#[allow(clippy::too_many_arguments)]
fn acc_fine(
    sch: &ChainSchema,
    si: usize,
    bx: i32,
    sx: f64,
    q: i32,
    m: usize,
    top: i32,
    r: usize,
    lat: i32,
    shared: &HashMap<(usize, i32, u32), Vec<f64>>,
    cache: &mut HashMap<(usize, i32, u32), Vec<f64>>,
    uniform: bool,
) -> AccFine {
    let g = ulp_exp(bx).max(lat);
    // Residue: bits in [max(g, Q), 2^top), mod m in Q units.
    let mut res = vec![0.0; m];
    let gq = g.max(q);
    if gq < top && m > 1 {
        let kb = ((top - gq) as u32).min(12);
        let pr = if uniform {
            vec![1.0 / (1u64 << kb) as f64; 1 << kb]
        } else {
            prev_res_ro(sch, si, gq, kb, shared, cache)
        };
        let shift = 1usize << (gq - q).clamp(0, 30);
        for (i, p) in pr.iter().enumerate() {
            res[(i * shift) % m] += p;
        }
    } else {
        res[0] = 1.0;
    }
    // Dropped part: |x| mod Q, bits in [g, Q) (beyond 12 bits: the top 12, never zero).
    let (mut tau, mut tau0) = (vec![(0i64, 1.0)], 1.0);
    if g < q {
        let kb = ((q - g) as u32).min(12);
        let exact = q - g <= 12;
        let cnt = 1usize << kb;
        let pr = if uniform {
            vec![1.0 / cnt as f64; cnt]
        } else {
            prev_res_ro(sch, si, q - kb as i32, kb, shared, cache)
        };
        let mut mm: BTreeMap<i64, f64> = BTreeMap::new();
        tau0 = 0.0;
        for (i, &p) in pr.iter().enumerate() {
            let j = if sx < 0.0 { (cnt - i) % cnt } else { i };
            if j == 0 && exact {
                tau0 += p;
                *mm.entry(0).or_default() += p;
                continue;
            }
            let v = j as f64 / cnt as f64 * tsc(r) * sx;
            let mut b = v.round() as i64;
            if b == 0 {
                b = if sx > 0.0 { 1 } else { -1 };
            }
            *mm.entry(b).or_default() += p;
        }
        tau = mm.into_iter().collect();
    }
    AccFine { res, tau, tau0 }
}

/// Structural identity of a step (the same term laws, counts, weights and truncation):
/// steps with equal ids have equal local laws for equal arguments.
fn step_id(st: &StepSchema) -> u64 {
    use std::hash::Hash;
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (st.frac, st.fp4).hash(&mut h);
    for g in &st.pop.groups {
        g.n.hash(&mut h);
        for (w, t) in &g.parts {
            (w.to_bits(), Rc::as_ptr(t) as usize).hash(&mut h);
        }
    }
    h.finish()
}

/// One step configuration for `local_moments`: the accumulator's cell (None: zero), emax
/// (from the accumulator when `from_acc`), the result cell.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cfg {
    pub bx: Option<i32>,
    pub sx: f64,
    pub e: i32,
    pub from_acc: bool,
    pub bs: i32,
    pub ss: f64,
}

/// Moments of a local error in ulps of the result cell: (E e, E e^2, E |e|^3, P(e != 0),
/// E e_pre, E e_pre^2, P(e_pre != 0)) with e_pre the dropped parts only (no final rounding).
pub(crate) type LocalMoments = [f64; 7];

/// Exact local-law moments of step `si` for each configuration (tier 2's input): the
/// accumulator's residue from the previous terms (no tracked bits), the terms' exact
/// residue and dropped-part laws.
pub(crate) fn local_moments(
    sch: &ChainSchema,
    si: usize,
    cfgs: &[Cfg],
    uniform: bool,
    threads: usize,
) -> Vec<LocalMoments> {
    let st = &sch.steps[si];
    let lat = sch.steps[..=si]
        .iter()
        .map(|s| s.pop.lattice())
        .min()
        .unwrap_or(i32::MAX);
    let shared: HashMap<(usize, i32, u32), Vec<f64>> = HashMap::new();
    let mut lprc = HashMap::new();
    // Arguments.
    let args: Vec<StepArgs> = cfgs
        .iter()
        .map(|c| {
            let ue = ulp_exp(c.bs);
            let q = (c.e - st.frac)
                .max(lat)
                .max(ue - R_CAP.trailing_zeros() as i32);
            let r = if ue > q { 1usize << (ue - q) } else { 1 };
            let m = r;
            let top = ue;
            let acc = match c.bx {
                None => AccIn::Zero,
                Some(b) => {
                    let ux = ulp_exp(b);
                    if ux >= q || lat >= q {
                        let g = if ux >= q { ux } else { q };
                        let nu = ((top - g).max(0) as u32).min(12);
                        let upper = if nu == 0 {
                            vec![]
                        } else if uniform {
                            vec![vec![1.0 / (1u64 << nu) as f64; 1 << nu]]
                        } else {
                            upper_given_low(
                                &prev_res_ro(sch, si, g, nu, &shared, &mut lprc),
                                0,
                                nu,
                                c.sx,
                            )
                        };
                        AccIn::Lattice {
                            sx: c.sx as i8,
                            shift: 1usize << (g - q).clamp(0, 40),
                            known: 0,
                            upper,
                        }
                    } else {
                        AccIn::Fine(acc_fine(
                            sch, si, b, c.sx, q, m, top, r, lat, &shared, &mut lprc, uniform,
                        ))
                    }
                }
            };
            StepArgs {
                e: c.e,
                from_acc: c.from_acc,
                q,
                bs: c.bs,
                ss: c.ss,
                acc,
                nl: 1,
            }
        })
        .collect();
    let mut bk: Vec<BlockKey> = args.iter().flat_map(|a| block_keys(st, a)).collect();
    bk.sort();
    bk.dedup();
    let made = par_map(bk.len(), threads, |i| {
        let mut tfc = TfMap::new();
        block_fine(st, bk[i], &mut tfc)
    });
    let blocks: HashMap<BlockKey, Block> = bk.into_iter().zip(made).collect();
    par_map(args.len(), threads, |i| {
        let t = steptrans(st, &args[i], &blocks, uniform);
        let mut mo = [0.0; 7];
        for (post, pre) in &t.t[0] {
            for &(k, p) in &post.nz {
                let v = k as f64 / G as f64;
                mo[0] += p * v;
                mo[1] += p * v * v;
                mo[2] += p * v.abs().powi(3);
                mo[3] += p;
            }
            for &(k, p) in &pre.nz {
                let v = k as f64 / G as f64;
                mo[4] += p * v;
                mo[5] += p * v * v;
                mo[6] += p;
            }
        }
        mo
    })
}

/// Total-variation distance to uniform of the exact sum's position in a grid of spacing
/// 2^ue (64 bins: the sum's bits [ue - 6, ue) from every term, carries from below ignored).
pub(crate) fn sum_position_tv(sch: &ChainSchema, ue: i32) -> f64 {
    let mut c = HashMap::new();
    let v = prev_res(sch, sch.steps.len(), ue - 6, 6, &mut c);
    v.iter().map(|p| (p - 1.0 / 64.0).abs()).sum::<f64>() / 2.0
}

/// Run the chain (single kernel).
// ASSUME(N-ERRDIST-MARKOV, N-ERRDIST-RESIDUE): local errors independent of the past given
// the cell and the accumulator's low K_BITS bits; the accumulator's higher bits from the
// previous terms.
pub fn run(sch: &ChainSchema, opts: Opts) -> Result<ChainLaw, String> {
    let grid = Grid::for_chain(sch);
    let (mut lo, mut hi) = (i32::MAX, i32::MIN);
    for st in &sch.steps {
        for g in &st.pop.groups {
            for (_, t) in &g.parts {
                for a in &t.atoms {
                    if a.0 != 0.0 && a.2 > 1e-14 {
                        let b = exp_floor(a.0);
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
    let span = exp_floor(grid.h * grid.n as f64) + 1;
    let cells = Cells {
        bmin: (lo - 8).min(grid.lh - 24),
        bmax: span.max(hi + 2),
    };
    let nc = cells.n();
    let cmap = CellMap::new(&grid, &cells);
    let mut notes = vec![format!(
        "tier 1 grid: h = 2^{}, {} points, {}",
        grid.lh,
        grid.n,
        if grid.exact {
            "exact lattice"
        } else {
            "continuous"
        }
    )];
    let threads = std::env::var("PTXV_ERRDIST_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map_or(8, |n| n.get())
                .min(48)
        });
    // Exact partial sum law (point masses), and P(sum exactly 0) for continuous grids.
    let mut acc = vec![0.0; grid.n];
    acc[grid.off as usize] = 1.0;
    let mut pzero = 1.0;
    // Per cell, per tracked low bits of the accumulator: the error law.
    let mut state: Vec<Vec<DHist>> = vec![vec![DHist::default(); NL]; nc];
    state[0][0].z = 1.0;
    let mut lat = i32::MAX;
    let mut fin = vec![];
    let mut coarse_cache: HashMap<(u64, i32), Rc<Vec<C64>>> = HashMap::new();
    let mut blocks: HashMap<BlockKey, Block> = HashMap::new();
    let mut blocks_sid = 0u64;
    type TKey = (u64, i32, i32, bool, i32, i8, u64);
    let mut trans: HashMap<TKey, Rc<Trans>> = HashMap::new();
    let mut prc: HashMap<(usize, i32, u32), Vec<f64>> = HashMap::new();
    let ns = std::env::var("PTXV_T1_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .map_or(sch.steps.len(), |n: usize| n.min(sch.steps.len()));
    let dbg = std::env::var("PTXV_ERRDIST_DEBUG").is_ok();
    let t_all = std::time::Instant::now();
    STATS_ON.store(
        std::env::var("PTXV_ERRDIST_DEBUG").is_ok(),
        std::sync::atomic::Ordering::Relaxed,
    );
    let mut trace = vec![];
    for (si, st) in sch.steps.iter().enumerate().take(ns) {
        let t_step = std::time::Instant::now();
        let last = si + 1 == ns;
        lat = lat.min(st.pop.lattice());
        // Step-level: P(all terms zero) for the exact-zero bookkeeping.
        let mut p_all0 = 1.0;
        for g in &st.pop.groups {
            let mut gp = 0.0;
            for (w, t) in &g.parts {
                let z: f64 = t.atoms.iter().filter(|a| a.0 == 0.0).map(|a| a.2).sum();
                gp += w * z.powi(g.n as i32);
            }
            p_all0 *= gp;
        }
        // Candidate emax levels from the terms.
        let mut levels: Vec<i32> = st
            .pop
            .groups
            .iter()
            .flat_map(|g| g.parts.iter().flat_map(|p| p.1.atoms.iter().map(|a| a.1)))
            .filter(|&a| a > NO_ALIGN)
            .collect();
        levels.sort();
        levels.dedup();
        let (amin, amax) = (*levels.first().unwrap_or(&0), *levels.last().unwrap_or(&0));
        let sid = step_id(st);
        {
            // The coarse blocks this step needs, in parallel.
            let need: Vec<i32> = (amin - 1..=amax)
                .filter(|e| !coarse_cache.contains_key(&(sid, *e)))
                .collect();
            let made = par_map(need.len(), threads, |i| {
                Rc::new(block_coarse(st, need[i], &grid))
            });
            for (e, b) in need.into_iter().zip(made) {
                coarse_cache.insert((sid, e), b);
            }
        }
        let mut get_block = |e: i32| -> Rc<Vec<C64>> {
            let e = e.clamp(amin - 1, amax);
            coarse_cache
                .entry((sid, e))
                .or_insert_with(|| Rc::new(block_coarse(st, e, &grid)))
                .clone()
        };
        // D_e = B_e - B_{e-1}: the block with max alignment exactly e.
        let mut dmax: HashMap<i32, Rc<Vec<C64>>> = HashMap::new();
        for &e in &levels {
            let hi = get_block(e);
            let lo = get_block(e - 1);
            dmax.insert(
                e,
                Rc::new(
                    hi.iter()
                        .zip(lo.iter())
                        .map(|(a, b)| a.add(b.scale(-1.0)))
                        .collect(),
                ),
            );
        }
        let p_dead = get_block(amin - 1)[0].re;
        let full = get_block(amax);
        // Blocks with every term aligned <= e (emax from the accumulator).
        let ble: HashMap<i32, Rc<Vec<C64>>> =
            (amin - 1..=amax).map(|e| (e, get_block(e))).collect();
        // Accumulator law by cell.
        let mut acc_mass = vec![0.0; nc];
        cmap.masses(
            &grid,
            &cells,
            &acc,
            if grid.exact { 0.0 } else { pzero },
            &mut acc_mass,
        );
        let mut acc_cells: Vec<Vec<f64>> = vec![vec![]; nc];
        for i in 0..grid.n {
            let c = cmap.of[i];
            if acc[i] == 0.0 || acc_mass[c] <= 1e-15 {
                continue;
            }
            let tiny = c == 0 || (!grid.exact && cells.binade(c) < cmap.bt);
            if tiny || grid.x(i).abs() < 2f64.powi(cmap.bt.max(-1074)) {
                continue;
            }
            if acc_cells[c].is_empty() {
                acc_cells[c] = vec![0.0; grid.n];
            }
            acc_cells[c][i] = acc[i];
        }
        if blocks_sid != sid {
            blocks.clear();
            blocks_sid = sid;
        }
        // Phase 1 (per accumulator cell, parallel): the joint (emax case, result cell)
        // weights and each transition's local-law arguments.
        let hm_of: Vec<f64> = state
            .iter()
            .map(|v| v.iter().map(|h| h.mass()).sum())
            .collect();
        let t1 = std::time::Instant::now();
        let dt0 = t1.duration_since(t_step).as_secs_f64();
        #[allow(clippy::type_complexity)]
        let phase1: Vec<(
            Vec<(usize, f64, TKey, Option<StepArgs>)>,
            f64,
            HashMap<(usize, i32, u32), Vec<f64>>,
        )> = {
            let lat_now = lat;
            let prc_snapshot = &prc;
            par_map(nc, threads, |cx| {
                let px = acc_mass[cx];
                let hm = hm_of[cx];
                let mut out = vec![];
                let mut to_zero = 0.0;
                let mut lprc: HashMap<(usize, i32, u32), Vec<f64>> = HashMap::new();
                if px <= 1e-13 || hm <= 0.0 {
                    return (out, to_zero, lprc);
                }
                let bx = (cx != 0).then(|| cells.binade(cx));
                let sx = cells.sign(cx);
                let tiny = cx == 0 || (!grid.exact && bx.is_some_and(|b| b < cmap.bt));
                let fx: Option<Vec<C64>> =
                    (!tiny && !acc_cells[cx].is_empty()).then(|| fwd(&acc_cells[cx]));
                let xm: f64 = if tiny || acc_cells[cx].is_empty() {
                    px
                } else {
                    acc_cells[cx].iter().sum()
                };
                if xm <= 0.0 {
                    return (out, to_zero, lprc);
                }
                let mut cases: Vec<(i32, bool, Rc<Vec<C64>>)> = vec![];
                match bx {
                    Some(b) => {
                        cases.push((b, true, ble[&b.clamp(amin - 1, amax)].clone()));
                        for &e in levels.iter().filter(|&&e| e > b) {
                            cases.push((e, false, dmax[&e].clone()));
                        }
                    }
                    None => {
                        to_zero += p_dead;
                        for &e in &levels {
                            cases.push((e, false, dmax[&e].clone()));
                        }
                    }
                }
                let mut sm = vec![0.0; nc];
                for (e, from_acc, f) in cases {
                    let v: Vec<f64> = match &fx {
                        Some(fx) => {
                            let prod: Vec<C64> =
                                fx.iter().zip(f.iter()).map(|(a, b)| a.mul(*b)).collect();
                            inv(&prod)
                        }
                        None => {
                            let mut b = inv(&f);
                            b.rotate_right(grid.off as usize);
                            b.iter_mut().for_each(|x| *x *= px);
                            b
                        }
                    };
                    let tot: f64 = v.iter().sum();
                    if tot.abs() <= 1e-15 * px {
                        continue;
                    }
                    cmap.masses(&grid, &cells, &v, 0.0, &mut sm);
                    let q0 = (e - st.frac).max(lat_now);
                    #[allow(clippy::needless_range_loop)] // `cs` is also the cell index
                    for cs in 0..nc {
                        let w = sm[cs];
                        if w <= 1e-11 * px {
                            continue;
                        }
                        let w = w / xm;
                        if cs == 0 {
                            to_zero += w;
                            continue;
                        }
                        let bs = cells.binade(cs);
                        let ss = cells.sign(cs);
                        let ue = ulp_exp(bs);
                        let q = q0.max(ue - R_CAP.trailing_zeros() as i32);
                        let r = if ue > q { 1usize << (ue - q) } else { 1 };
                        let m = r * NL;
                        let top = ue + K_BITS as i32;
                        let accin = match bx {
                            None => AccIn::Zero,
                            Some(b) => {
                                let ux = ulp_exp(b);
                                if ux >= q {
                                    let needed = top - ux;
                                    if needed <= 0 {
                                        AccIn::Lattice {
                                            sx: sx as i8,
                                            shift: 1,
                                            known: 0,
                                            upper: vec![],
                                        }
                                    } else {
                                        let known = (K_BITS as i32).min(needed) as u32;
                                        let nu = ((needed - known as i32) as u32).min(12);
                                        let upper = if nu == 0 {
                                            vec![]
                                        } else if opts.uniform {
                                            vec![
                                                vec![1.0 / (1u64 << nu) as f64; 1 << nu];
                                                1 << known
                                            ]
                                        } else {
                                            let v = prev_res_ro(
                                                sch,
                                                si,
                                                ux,
                                                known + nu,
                                                prc_snapshot,
                                                &mut lprc,
                                            );
                                            upper_given_low(&v, known, nu, sx)
                                        };
                                        AccIn::Lattice {
                                            sx: sx as i8,
                                            shift: 1usize << (ux - q).clamp(0, 40),
                                            known,
                                            upper,
                                        }
                                    }
                                } else if lat_now >= q {
                                    let nu = ((top - q).max(0) as u32).min(12);
                                    let upper = if nu == 0 {
                                        vec![]
                                    } else if opts.uniform {
                                        vec![vec![1.0 / (1u64 << nu) as f64; 1 << nu]]
                                    } else {
                                        upper_given_low(
                                            &prev_res_ro(sch, si, q, nu, prc_snapshot, &mut lprc),
                                            0,
                                            nu,
                                            sx,
                                        )
                                    };
                                    AccIn::Lattice {
                                        sx: sx as i8,
                                        shift: 1,
                                        known: 0,
                                        upper,
                                    }
                                } else {
                                    AccIn::Fine(acc_fine(
                                        sch,
                                        si,
                                        b,
                                        sx,
                                        q,
                                        m,
                                        top,
                                        r,
                                        lat_now,
                                        prc_snapshot,
                                        &mut lprc,
                                        opts.uniform,
                                    ))
                                }
                            }
                        };
                        let key: TKey = (sid, q, e, from_acc, bs, ss as i8, accin.key());
                        let args = (!trans.contains_key(&key)).then_some(StepArgs {
                            e,
                            from_acc,
                            q,
                            bs,
                            ss,
                            acc: accin,
                            nl: NL,
                        });
                        out.push((cs, w, key, args));
                    }
                }
                // Normalize the transition weights of this cell to 1.
                let tw: f64 = out.iter().map(|x| x.1).sum::<f64>() + to_zero;
                if tw > 0.0 {
                    out.iter_mut().for_each(|x| x.1 /= tw);
                    to_zero /= tw;
                }
                (out, to_zero, lprc)
            })
        };
        for (_, _, l) in &phase1 {
            for (k, v) in l {
                prc.entry(*k).or_insert_with(|| v.clone());
            }
        }
        let dt1 = t1.elapsed().as_secs_f64();
        // Phase 2: missing local laws (blocks first), in parallel.
        let t2 = std::time::Instant::now();
        let mut need: HashMap<TKey, StepArgs> = HashMap::new();
        for (v, _, _) in &phase1 {
            for (_, _, k, a) in v {
                if let Some(a) = a {
                    need.entry(*k).or_insert_with(|| a.clone());
                }
            }
        }
        let mut bk: Vec<BlockKey> = need
            .values()
            .flat_map(|a| block_keys(st, a))
            .filter(|k| !blocks.contains_key(k))
            .collect();
        bk.sort();
        bk.dedup();
        let nb = bk.len();
        let newb = par_map(bk.len(), threads, |i| {
            let mut tfc = TfMap::new();
            block_fine(st, bk[i], &mut tfc)
        });
        for (k, b) in bk.into_iter().zip(newb) {
            blocks.insert(k, b);
        }
        let nk: Vec<(TKey, StepArgs)> = need.into_iter().collect();
        let newt = par_map(nk.len(), threads, |i| {
            Rc::new(steptrans(st, &nk[i].1, &blocks, opts.uniform))
        });
        let n_new = nk.len();
        for ((k, _), t) in nk.into_iter().zip(newt) {
            trans.insert(k, t);
        }
        let dt2 = t2.elapsed().as_secs_f64();
        // Phase 3: convolutions, per result cell (parallel).
        let t3 = std::time::Instant::now();
        let mut incoming: Vec<Vec<(usize, f64, Rc<Trans>)>> = vec![vec![]; nc];
        let mut zero_in = 0.0;
        let mut dtab: BTreeMap<(i32, i32), (f64, f64)> = BTreeMap::new();
        for (cx, (v, to_zero, _)) in phase1.iter().enumerate() {
            let hm = hm_of[cx];
            zero_in += hm * to_zero;
            for (cs, w, k, _) in v {
                let t = trans[k].clone();
                if dbg {
                    let bx = (cx != 0).then(|| cells.binade(cx));
                    let bs = cells.binade(*cs);
                    let key = (bx.map_or(-999, |b| bs - b), bs - k.2);
                    let d = dtab.entry(key).or_insert((0.0, 0.0));
                    let mut pnz = 0.0;
                    #[allow(clippy::needless_range_loop)] // `l` also indexes the transitions
                    for l in 0..NL {
                        let ml = state[cx][l].mass();
                        if ml > 0.0 {
                            pnz += ml * (1.0 - t.t[l].iter().map(|x| x.0.p0).sum::<f64>());
                        }
                    }
                    d.0 += w * hm;
                    d.1 += w * pnz;
                }
                incoming[*cs].push((cx, *w, t));
            }
        }
        let state_ref = &state;
        let results: Vec<(Vec<DHist>, Vec<DHist>)> = par_map(nc, threads, |cs| {
            let mut nx: Vec<DHist> = vec![DHist::default(); NL];
            let mut np: Vec<DHist> = vec![DHist::default(); NL];
            if incoming[cs].is_empty() {
                return (nx, np);
            }
            // Group by source cell: the mixture over emax cases of each (l, l') law.
            let mut by_cx: BTreeMap<usize, Vec<(f64, &Rc<Trans>)>> = BTreeMap::new();
            for (cx, w, t) in &incoming[cs] {
                by_cx.entry(*cx).or_default().push((*w, t));
            }
            // Accumulate the dense parts in the Fourier domain (size NF, linear convolution of
            // two |i| < W1 laws), the rest (exact zeros, far bins) directly.
            let mut fx: Vec<Option<Vec<C64>>> = vec![None; NL];
            let mut fp: Vec<Option<Vec<C64>>> = vec![None; NL];
            let mut mix = Mix::new();
            let mut mixp = Mix::new();
            for (cx, list) in by_cx {
                let k = cells.binade(cs)
                    - if cx == 0 {
                        cells.binade(cs)
                    } else {
                        cells.binade(cx)
                    };
                #[allow(clippy::needless_range_loop)] // `l` also indexes the transitions
                for l in 0..NL {
                    let src = &state_ref[cx][l];
                    if src.mass() <= 0.0 {
                        continue;
                    }
                    let sm = src.mass();
                    let mut h: Option<(DHist, (usize, usize))> = None;
                    let mut fh: Option<Option<Vec<C64>>> = None;
                    for lp in 0..NL {
                        // Contributions below 1e-13 are not resolved (kept as clipped mass).
                        let mm: f64 = list
                            .iter()
                            .map(|(w, t)| {
                                let a = &t.t[l][lp].0;
                                w * (a.p0 + a.nz.iter().map(|x| x.1).sum::<f64>())
                            })
                            .sum();
                        if mm <= 0.0 {
                            continue;
                        }
                        if sm * mm < 1e-13 {
                            nx[lp].tail += sm * mm;
                            if last {
                                np[lp].tail += sm * mm;
                            }
                            continue;
                        }
                        mix.clear();
                        mixp.clear();
                        for (w, t) in &list {
                            let (a, b) = &t.t[l][lp];
                            let m = a.p0 + a.nz.iter().map(|x| x.1).sum::<f64>();
                            if m <= 1e-15 {
                                continue;
                            }
                            mix.add(a, *w);
                            if last {
                                mixp.add(b, *w);
                            }
                        }
                        if !mix.any {
                            continue;
                        }
                        let (h, sup) = h.get_or_insert_with(|| {
                            let h = src.rescale(k);
                            let sup = support(&h.d);
                            (h, sup)
                        });
                        let sup = *sup;
                        h.conv(sup, &mut fh, &mix, &mut nx[lp], &mut fx[lp]);
                        if last {
                            h.conv(sup, &mut fh, &mixp, &mut np[lp], &mut fp[lp]);
                        }
                    }
                }
            }
            for lp in 0..NL {
                if let Some(f) = fx[lp].take() {
                    nx[lp].add_dense_fft(f);
                }
                if let Some(f) = fp[lp].take() {
                    np[lp].add_dense_fft(f);
                }
            }
            for h in nx.iter_mut().chain(np.iter_mut()) {
                h.finish();
            }
            (nx, np)
        });
        let mut next: Vec<Vec<DHist>> = Vec::with_capacity(nc);
        let mut fin_pre: Vec<Vec<DHist>> = Vec::with_capacity(nc);
        for (a, b) in results {
            next.push(a);
            fin_pre.push(b);
        }
        next[0][0].z += zero_in;
        let dt3 = t3.elapsed().as_secs_f64();
        // Advance the exact partial sums.
        let fa = fwd(&acc);
        let prod: Vec<C64> = fa.iter().zip(full.iter()).map(|(a, b)| a.mul(*b)).collect();
        acc = inv(&prod);
        for x in acc.iter_mut() {
            if *x < 0.0 {
                *x = 0.0;
            }
        }
        pzero *= p_all0;
        {
            let tot: f64 = next
                .iter()
                .flatten()
                .map(|h| h.mass())
                .sum::<f64>()
                .max(1e-300);
            let nz: f64 = next.iter().flatten().map(|h| h.mass() - h.z).sum();
            trace.push((
                nz / tot,
                (0..NL)
                    .map(|l| next.iter().map(|v| v[l].mass()).sum::<f64>() / tot)
                    .collect(),
            ));
        }
        if dbg {
            let st: Vec<u64> = STATS
                .iter()
                .map(|x| x.swap(0, std::sync::atomic::Ordering::Relaxed))
                .collect();
            let c = st[0].max(1) as f64;
            eprintln!(
                "  conv calls {} mean h support {:.0} mean l span {:.0} mean far {:.1} mean sparse {:.2} direct {:.2}",
                st[0],
                st[1] as f64 / c,
                st[2] as f64 / c,
                st[3] as f64 / c,
                st[4] as f64 / c,
                st[5] as f64 / c
            );
            let nz: f64 = next.iter().flatten().map(|h| h.mass() - h.z).sum();
            let tot: f64 = next.iter().flatten().map(|h| h.mass()).sum();
            let lb: Vec<String> = (0..NL)
                .map(|l| {
                    format!(
                        "{:.4}",
                        next.iter().map(|v| v[l].mass()).sum::<f64>() / tot.max(1e-300)
                    )
                })
                .collect();
            eprintln!(
                "tier1 step {si}: P(error != 0) {nz:.4} (mass {tot:.6}); low bits [{}]; {} new local laws, {nb} blocks; phases {dt0:.1}s {dt1:.1}s {dt2:.1}s {dt3:.1}s, step {:.1}s; hist entries {}",
                lb.join(" "),
                n_new,
                t_step.elapsed().as_secs_f64(),
                next.iter()
                    .flatten()
                    .map(|h| h.d.iter().filter(|&&x| x != 0.0).count() + h.far.len())
                    .sum::<usize>()
            );
            let dt: f64 = dtab.values().map(|x| x.0).sum();
            for (k, v) in &dtab {
                if v.0 > 0.01 * dt {
                    eprintln!(
                        "  t1 step {si} bS-bX {:>4} bS-E {:>3}: w {:.3} P(e!=0) {:.4}",
                        k.0,
                        k.1,
                        v.0 / dt,
                        v.1 / v.0
                    );
                }
            }
        }
        state = next;
        if last {
            for c in 0..nc {
                let p: f64 = state[c].iter().map(|h| h.mass()).sum();
                if p > 0.0 {
                    let mut h = DHist::default();
                    let mut hp = DHist::default();
                    for l in 0..NL {
                        h.merge(&state[c][l]);
                        hp.merge(&fin_pre[c][l]);
                    }
                    fin.push((c, p, h.to_hist(), hp.to_hist()));
                }
            }
        }
    }
    let tot: f64 = fin.iter().map(|f| f.1).sum();
    if (tot - 1.0).abs() > 1e-6 {
        notes.push(format!(
            "tier 1: total mass {tot:.6} (grid range / pruning)"
        ));
    }
    notes.push(format!(
        "tier 1: {} steps in {:.1}s ({threads} threads), low {K_BITS} accumulator bits tracked",
        ns,
        t_all.elapsed().as_secs_f64()
    ));
    Ok(ChainLaw {
        cells,
        fin,
        residue: vec![],
        notes,
        g: G,
        trace,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errdist::chain::StepModel;
    use crate::errdist::fine::Group;
    use crate::errdist::fine::Pop;
    use crate::errdist::spec::Dist;
    use crate::errdist::spec::Format;
    use crate::errdist::special::Rng;

    /// A chain of `steps` steps of `n` i.i.d. products of two codes with law `d` in `fmt`.
    fn chain(d: &Dist, fmt: Format, n: usize, steps: usize, lp: bool) -> ChainSchema {
        let pm = crate::errdist::laws::code_pmf(d, fmt, 1.0);
        let model = if lp {
            StepModel::Lp(crate::mma_lp::DEFAULT_LP_MMA)
        } else {
            StepModel::Mma(crate::fp::DEFAULT_MMA)
        };
        let t = Rc::new(crate::errdist::schema::product_law(
            &pm,
            &pm,
            (1.0, 0),
            (1.0, 0),
            model,
        ));
        let st = StepSchema {
            model,
            frac: 25,
            pop: Pop {
                groups: vec![Group {
                    n,
                    parts: vec![(1.0, t)],
                }],
            },
            fp4: false,
        };
        ChainSchema {
            steps: vec![st; steps],
            round16: None,
            pruned: 0.0,
            notes: vec![],
        }
    }

    /// Simulate the chain from its term laws (the step semantics: truncate each term at
    /// 2^(emax - 25) toward zero, add exactly, round toward zero to f32): per step
    /// P(error so far != 0) and the law of the result's low 2 bits; the final P(|e| > 1/2 ulp)
    /// (ties 1/2).
    fn simulate(sch: &ChainSchema, n_chains: usize, seed: u64) -> (Vec<(f64, [f64; 4])>, f64) {
        let t = &sch.steps[0].pop.groups[0].parts[0].1;
        let n = sch.steps[0].pop.groups[0].n;
        let mut cdf = vec![];
        let mut c = 0.0;
        for a in &t.atoms {
            c += a.2;
            cdf.push(c);
        }
        let tot = c;
        let mut r = Rng::new(seed);
        let ns = sch.steps.len();
        let mut out = vec![(0.0, [0.0; 4]); ns];
        let mut differ = 0.0;
        let mut tab: BTreeMap<(usize, i32, i32), (f64, f64)> = BTreeMap::new();
        for _ in 0..n_chains {
            let mut acc = 0f32;
            let mut ex = crate::errdist::chain::ExpAcc::default();
            for (si, o) in out.iter_mut().enumerate() {
                let mut ts: Vec<(f64, i32)> = vec![];
                for _ in 0..n {
                    let u = r.unit() * tot;
                    let k = cdf.partition_point(|&x| x < u).min(t.atoms.len() - 1);
                    let a = t.atoms[k];
                    if a.0 != 0.0 {
                        ts.push((a.0, a.1));
                    }
                    ex.add(a.0);
                }
                let a = acc as f64;
                if a != 0.0 {
                    ts.push((a, exp_floor(a)));
                }
                if !ts.is_empty() {
                    let emax = ts.iter().map(|x| x.1).max().unwrap();
                    let q = emax - 25;
                    let sc = 2f64.powi(-q);
                    let s: i128 = ts.iter().map(|x| (x.0 * sc).trunc() as i128).sum();
                    let mut loc = crate::errdist::chain::ExpAcc::default();
                    for x in &ts {
                        loc.add(x.0);
                    }
                    let exl = loc.approx();
                    acc = crate::fp::round_int_to_f32(s, q, true, 0);
                    loc.add(-(acc as f64));
                    if a != 0.0 && exl != 0.0 {
                        let k = (si, exp_floor(exl) - exp_floor(a), exp_floor(exl) - emax);
                        let c = tab.entry(k).or_insert((0.0, 0.0));
                        c.0 += 1.0;
                        c.1 += (!loc.is_zero()) as u8 as f64;
                    }
                }
                let mut d = ex.clone();
                d.add(-(acc as f64));
                o.0 += (!d.is_zero()) as u8 as f64;
                if acc != 0.0 {
                    let ul = 2f64.powi(ulp_exp(exp_floor(acc as f64)));
                    let lb = ((acc as f64).abs() / ul) as u64 & 3;
                    o.1[lb as usize] += 1.0;
                } else {
                    o.1[0] += 1.0;
                }
                if si + 1 == ns && acc != 0.0 {
                    let e = (-d.approx() / 2f64.powi(ulp_exp(exp_floor(acc as f64)))).abs();
                    differ += if e > 0.5 {
                        1.0
                    } else if e == 0.5 {
                        0.5
                    } else {
                        0.0
                    };
                }
            }
        }
        for si in 0..ns {
            let tot: f64 = tab.iter().filter(|x| x.0.0 == si).map(|x| x.1.0).sum();
            for (k, v) in tab.iter().filter(|x| x.0.0 == si) {
                if v.0 > 0.01 * tot {
                    eprintln!(
                        "  sim step {si} bS-bX {:>4} bS-E {:>3}: w {:.3} P(e!=0) {:.4}",
                        k.1,
                        k.2,
                        v.0 / tot,
                        v.1 / v.0
                    );
                }
            }
        }
        for o in out.iter_mut() {
            o.0 /= n_chains as f64;
            for x in o.1.iter_mut() {
                *x /= n_chains as f64;
            }
        }
        (out, differ / n_chains as f64)
    }

    fn compare(sch: &ChainSchema, n_chains: usize, tol_rel: f64) {
        let (sim, sim_pd) = simulate(sch, n_chains, 7);
        let law = run(sch, Opts::default()).unwrap();
        let est = crate::errdist::combine::single(&law, Format::F32, None);
        let pd = est.p_differ.unwrap().0;
        let mut worst = 0f64;
        for (si, ((p, lb), (q, lm))) in sim.iter().zip(&law.trace).enumerate() {
            let se = (p * (1.0 - p) / n_chains as f64).sqrt().max(1e-9);
            eprintln!(
                "step {si}: P(delta != 0) sim {p:.4} model {q:.4} ({:+.1} se); low bits sim {:.4?} model {:.4?}",
                (q - p) / se,
                lb,
                lm
            );
            if *p > 0.01 {
                worst = worst.max(((q - p) / p).abs() - 4.0 * se / p);
            }
        }
        let se = (sim_pd * (1.0 - sim_pd) / n_chains as f64).sqrt();
        eprintln!(
            "P(differ) sim {sim_pd:.4} model {pd:.4} ({:+.1} se)",
            (pd - sim_pd) / se.max(1e-9)
        );
        assert!(worst < tol_rel, "P(delta != 0) off by {worst}");
        assert!(
            (pd - sim_pd).abs() < tol_rel * sim_pd + 4.0 * se,
            "P(differ) {pd} vs {sim_pd}"
        );
    }

    /// The first step (accumulator 0): P(e != 0 | emax, result cell) against sampling.
    fn first_step(sch: &ChainSchema, n_samp: usize) -> f64 {
        let st = &sch.steps[0];
        let t = &st.pop.groups[0].parts[0].1;
        let n = st.pop.groups[0].n;
        let mut cdf = vec![];
        let mut c = 0.0;
        for a in &t.atoms {
            c += a.2;
            cdf.push(c);
        }
        let mut r = Rng::new(11);
        let mut tab: BTreeMap<(i32, i32, i8), (f64, f64)> = BTreeMap::new();
        for _ in 0..n_samp {
            let mut ts: Vec<(f64, i32)> = vec![];
            let mut ex = crate::errdist::chain::ExpAcc::default();
            for _ in 0..n {
                let u = r.unit() * c;
                let k = cdf.partition_point(|&y| y < u).min(t.atoms.len() - 1);
                let a = t.atoms[k];
                if a.0 != 0.0 {
                    ts.push((a.0, a.1));
                }
                ex.add(a.0);
            }
            if ts.is_empty() {
                continue;
            }
            let emax = ts.iter().map(|y| y.1).max().unwrap();
            let q = emax - 25;
            let sc = 2f64.powi(-q);
            let sm: i128 = ts.iter().map(|y| (y.0 * sc).trunc() as i128).sum();
            let res = crate::fp::round_int_to_f32(sm, q, true, 0) as f64;
            let exv = ex.approx();
            if exv == 0.0 {
                continue;
            }
            let mut d = ex.clone();
            d.add(-res);
            let ent = tab
                .entry((emax, exp_floor(exv), exv.signum() as i8))
                .or_insert((0.0, 0.0));
            ent.0 += 1.0;
            ent.1 += (!d.is_zero()) as u8 as f64;
        }
        let (mut wm, mut ws, mut tot) = (0.0, 0.0, 0.0);
        let mut worst = 0f64;
        for ((e, bs, ss), (cnt, nz)) in tab {
            if cnt < 5000.0 {
                continue;
            }
            let ue = ulp_exp(bs);
            let q = (e - 25).max(ue - 10);
            let a = StepArgs {
                e,
                from_acc: false,
                q,
                bs,
                ss: ss as f64,
                acc: AccIn::Zero,
                nl: NL,
            };
            let mut tfc = TfMap::new();
            let blocks: HashMap<BlockKey, Block> = block_keys(st, &a)
                .into_iter()
                .map(|k| (k, block_fine(st, k, &mut tfc)))
                .collect();
            let tr = steptrans(st, &a, &blocks, false);
            let md: f64 = tr.t[0]
                .iter()
                .map(|x| x.0.nz.iter().map(|y| y.1).sum::<f64>())
                .sum();
            let mc = nz / cnt;
            let se = (mc.max(3.0 / cnt) * (1.0 - mc) / cnt).sqrt();
            eprintln!(
                "E {e:>3} bS {bs:>3} s {ss:>2}: n {cnt:>7} MC {mc:.5} model {md:.5} ({:+.1} se)",
                (md - mc) / se
            );
            worst = worst.max(((md - mc) / se).abs());
            wm += cnt * md;
            ws += nz;
            tot += cnt;
        }
        eprintln!("weighted: MC {:.5} model {:.5}", ws / tot, wm / tot);
        worst
    }

    #[test]
    fn first_step_local_laws_match_sampling() {
        let w = first_step(
            &chain(
                &Dist::Normal {
                    mean: 0.0,
                    std: 1.0,
                },
                Format::Bf16,
                16,
                1,
                false,
            ),
            400_000,
        );
        assert!(w < 6.0, "bf16 worst {w} se");
        let w = first_step(
            &chain(
                &Dist::Normal {
                    mean: 0.0,
                    std: 64.0,
                },
                Format::E4m3,
                32,
                1,
                true,
            ),
            2_000_000,
        );
        assert!(w < 6.0, "e4m3 worst {w} se");
    }

    /// One step from a fixed accumulator: per (emax case, result cell), the law of
    /// (result low bits, error != 0) against `steptrans` with the accumulator's bits known.
    #[test]
    fn single_step_from_known_accumulator() {
        let sch = chain(
            &Dist::Normal {
                mean: 0.0,
                std: 64.0,
            },
            Format::E4m3,
            32,
            2,
            true,
        );
        let st = &sch.steps[1];
        let t = &st.pop.groups[0].parts[0].1;
        let mut cdf = vec![];
        let mut c = 0.0;
        for a in &t.atoms {
            c += a.2;
            cdf.push(c);
        }
        for &x in &[
            20001.0f32 * 1.0 + 0.0,
            -20003.0,
            9000.0 + 0.0009765625 * 3.0,
            70001.0,
        ] {
            let xv = x as f64;
            let bx = exp_floor(xv);
            let ux = ulp_exp(bx);
            let mx = (xv.abs() / 2f64.powi(ux)) as u64;
            let l = (mx & 3) as usize;
            let mut r = Rng::new(3);
            // (e, from_acc, bs, ss) -> (count, [low bits counts], nonzero count)
            type Row = (f64, [f64; 4], f64);
            let mut tab: BTreeMap<(i32, bool, i32, i8), Row> = BTreeMap::new();
            for _ in 0..400_000 {
                let mut ts: Vec<(f64, i32)> = vec![(xv, bx)];
                let mut ex = crate::errdist::chain::ExpAcc::default();
                ex.add(xv);
                let mut et = i32::MIN;
                for _ in 0..32 {
                    let u = r.unit() * c;
                    let k = cdf.partition_point(|&y| y < u).min(t.atoms.len() - 1);
                    let a = t.atoms[k];
                    if a.0 != 0.0 {
                        ts.push((a.0, a.1));
                        et = et.max(a.1);
                    }
                    ex.add(a.0);
                }
                let emax = ts.iter().map(|y| y.1).max().unwrap();
                let q = emax - 25;
                let sc = 2f64.powi(-q);
                let sm: i128 = ts.iter().map(|y| (y.0 * sc).trunc() as i128).sum();
                let res = crate::fp::round_int_to_f32(sm, q, true, 0) as f64;
                let exv = ex.approx();
                if res == 0.0 || exv == 0.0 {
                    continue;
                }
                let bs = exp_floor(exv);
                let e = emax;
                let from_acc = et <= bx;
                let mut d = ex.clone();
                d.add(-res);
                let lb = ((res.abs() / 2f64.powi(ulp_exp(exp_floor(res)))) as u64 & 3) as usize;
                let ent = tab
                    .entry((e, from_acc, bs, exv.signum() as i8))
                    .or_insert((0.0, [0.0; 4], 0.0));
                ent.0 += 1.0;
                ent.1[lb] += 1.0;
                ent.2 += (!d.is_zero()) as u8 as f64;
            }
            for ((e, from_acc, bs, ss), (n, lbs, nz)) in tab {
                if n < 20000.0 {
                    continue;
                }
                let ue = ulp_exp(bs);
                let q = (e - 25).max(ue - 10);
                let needed = ue + K_BITS as i32 - ux;
                let shift = 1usize << (ux - q).clamp(0, 40);
                let (known, upper) = if needed <= 0 {
                    (0, vec![])
                } else {
                    let known = (K_BITS as i32).min(needed) as u32;
                    let nu = (needed - known as i32).max(0) as u32;
                    let mut up = vec![0.0; 1 << nu];
                    if nu > 0 {
                        // The magnitude's bits above the known ones.
                        up[((mx >> known) as usize) & ((1 << nu) - 1)] = 1.0;
                    }
                    (known, if nu > 0 { vec![up; 1 << known] } else { vec![] })
                };
                let a = StepArgs {
                    e,
                    from_acc,
                    q,
                    bs,
                    ss: ss as f64,
                    acc: AccIn::Lattice {
                        sx: x.signum() as i8,
                        shift,
                        known,
                        upper,
                    },
                    nl: NL,
                };
                let mut tfc = TfMap::new();
                let blocks: HashMap<BlockKey, Block> = block_keys(st, &a)
                    .into_iter()
                    .map(|k| (k, block_fine(st, k, &mut tfc)))
                    .collect();
                let tr = steptrans(st, &a, &blocks, false);
                let row = &tr.t[l];
                let ml: Vec<f64> = row
                    .iter()
                    .map(|x| x.0.p0 + x.0.nz.iter().map(|y| y.1).sum::<f64>())
                    .collect();
                let mnz: f64 = row
                    .iter()
                    .map(|x| x.0.nz.iter().map(|y| y.1).sum::<f64>())
                    .sum();
                let sl: Vec<f64> = lbs.iter().map(|v| v / n).collect();
                eprintln!(
                    "x {x} l {l} e {e} acc {from_acc} bS-bX {} ss {ss}: n {n} low bits sim {:.4?} model {:.4?}; P(e!=0) sim {:.4} model {:.4}",
                    bs - bx,
                    sl,
                    ml,
                    nz / n,
                    mnz
                );
            }
        }
    }

    /// e4m3 codes of N(0, 64) (MX-like: coarse lattice, rare fine terms), Lp alignment.
    #[test]
    fn e4m3_chain_matches_simulation() {
        let sch = chain(
            &Dist::Normal {
                mean: 0.0,
                std: 64.0,
            },
            Format::E4m3,
            32,
            6,
            true,
        );
        compare(&sch, 400_000, 0.05);
    }

    /// bf16 normal products (fine residues everywhere).
    #[test]
    fn bf16_chain_matches_simulation() {
        let sch = chain(
            &Dist::Normal {
                mean: 0.0,
                std: 1.0,
            },
            Format::Bf16,
            16,
            4,
            false,
        );
        compare(&sch, 200_000, 0.05);
    }

    /// Small K: one e4m3 step (K = 32), where the residues are far from uniform.
    #[test]
    fn e4m3_single_step_matches_simulation() {
        let sch = chain(
            &Dist::Normal {
                mean: 0.0,
                std: 64.0,
            },
            Format::E4m3,
            32,
            1,
            true,
        );
        compare(&sch, 400_000, 0.05);
    }

    /// e5m2, heavy-tailed (Laplace) inputs.
    #[test]
    fn e5m2_laplace_chain_matches_simulation() {
        let sch = chain(
            &Dist::Laplace {
                mean: 0.0,
                scale: 16.0,
            },
            Format::E5m2,
            32,
            4,
            true,
        );
        compare(&sch, 400_000, 0.05);
    }

    /// bf16, Student-t (df 3) inputs: wide binade spread in every step.
    #[test]
    fn bf16_student_t_chain_matches_simulation() {
        let sch = chain(
            &Dist::StudentT {
                df: 3.0,
                mean: 0.0,
                scale: 1.0,
            },
            Format::Bf16,
            16,
            3,
            false,
        );
        compare(&sch, 200_000, 0.05);
    }
}

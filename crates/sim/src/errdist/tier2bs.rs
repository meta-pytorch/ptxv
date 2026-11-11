/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tier 2 on MMA accumulator chains, plain and block-scaled (docs/errdist.md §5.3.1).
//!
//! Each tensor-core step is one alignment to the step's largest exponent, a truncation of
//! every term (and of the accumulator) at Q = 2^(emax - frac), an exact sum and one round
//! toward zero to the f32 grid of the result (N-MMA, N-MMA-BS, N-MMA-F8). Its error has two
//! parts:
//!
//! - the final round toward zero: the sum after truncation lies on the grid
//!   2^max(Q, lattice), so the residue below the result's ulp has k = ulp - max(Q, lattice)
//!   grid bits (none when k <= 0). Its law is discrete, uniform on the 2^k grid;
//! - the truncation of the terms below Q (exact moments from the term atoms: block scale
//!   times element products, under the spec and the recipe) and of the accumulator.
//!
//! The decisive structure of narrow-format chains is the lattice: an output whose terms all
//! sit on a coarse grid is exact at every step, and the same output is on its f32 grid. The
//! tier therefore conditions on y's lattice class (the lowest set bit of all terms, against
//! y's ulp) as well as on y: given the class, every term's lowest bit is at least L, which
//! fixes each step's lattice law (the accumulator's lattice is the larger of the previous
//! terms' lattice and the previous result's ulp) and y's position within its ulp (on the
//! grid, or an odd multiple of 2^-k ulp). Partial sums are normal given y (sub-sums of y).
//! The largest step residues given y stay discrete (as the DAG CLT keeps its top 16); the
//! rest and the truncations are a zero-inflated normal. The PMF over output ulps follows
//! from y's position, the error law and the final rounding exactly; P(differ) = 1 - P(0).

use std::collections::BTreeMap;

use super::combine::TentPmf;
use super::fine::C64;
use super::markov::ulp_exp;
use super::report::ErrorBar;
use super::report::Estimate;
use super::report::PairMoments;
use super::schema::ChainSchema;
use super::spec::Format;
use super::tier2::LAW_NODES;
use super::tier2::LMIN;
use super::tier2::LN;
use super::tier2::Law;
use super::tier2::binade_prob;
use super::tier2::binades;
use super::tier2::grid_moments;
use super::tier2::lat_index;
use super::tier2::law_conv;
use super::tier2::law_moments;
use super::tier2::moments;
use super::tier2::over_y_pts_w;
use super::tier2::pn_in;
use super::tier2::u_scale;
use crate::errdist::det::HashMap;

/// Atoms (v, p).
type Atoms = Vec<(f64, f64)>;
/// A memo table shared across threads.
type Memo<K, V> = std::sync::Mutex<HashMap<K, V>>;

/// One term law's atoms by (exponent, lowest bit), and its mass by lowest bit.
struct PartLaw {
    w: f64,
    n: usize,
    /// ((exponent, lowest bit), atoms (v, p)).
    buckets: Vec<((i32, i32), Atoms)>,
    /// P(lowest bit >= x) for x = LMIN.. (zeros count as >= everything).
    ge: Vec<f64>,
    /// (alignment exponent, p), zeros included at their alignment.
    align: Vec<(i32, f64)>,
}

/// A step's precomputed laws.
struct Step {
    frac: i32,
    mb: f64,
    vb: f64,
    s_abs: f64,
    /// Groups of parts (independent groups; parts are a mixture over the group's context).
    groups: Vec<Vec<PartLaw>>,
    /// P(all of the step's terms have lowest bit >= x).
    own: Vec<f64>,
    /// Alignment levels and P(largest alignment <= level); `dead`: P(no live term).
    levels: Vec<i32>,
    fe: Vec<f64>,
    dead: f64,
    /// `trunc` results by (q, l); `cf` results by (ue, gr, qe, l).
    cache: Memo<(i32, i32), (f64, f64, f64)>,
    cf_cache: Memo<(i32, i32, i32, i32), Vec<C64>>,
}

impl Step {
    fn new(st: &super::schema::StepSchema) -> Step {
        let mut groups = vec![];
        let (mut mb, mut vb, mut s_abs) = (0.0, 0.0, 0.0);
        for g in &st.pop.groups {
            let (mut e1, mut e2) = (0.0, 0.0);
            let mut parts = vec![];
            for (w, t) in &g.parts {
                let mut bk: HashMap<(i32, i32), Vec<(f64, f64)>> = HashMap::default();
                let mut lbm = vec![0.0; LN];
                let mut zero = 0.0;
                let mut al: BTreeMap<i32, f64> = BTreeMap::new();
                for &(v, a, p) in &t.atoms {
                    e1 += w * p * v;
                    e2 += w * p * v * v;
                    s_abs += w * p * v.abs() * g.n as f64;
                    *al.entry(a).or_default() += p;
                    if v == 0.0 {
                        zero += p;
                        continue;
                    }
                    let lb = super::fine::lowest_bit(v);
                    lbm[lat_index(lb)] += p;
                    bk.entry((super::spec::exp_floor(v), lb))
                        .or_default()
                        .push((v, p));
                }
                let mut ge = vec![0.0; LN];
                let mut acc = 0.0;
                for i in (0..LN).rev() {
                    acc += lbm[i];
                    ge[i] = acc + zero;
                }
                parts.push(PartLaw {
                    w: *w,
                    n: g.n,
                    buckets: bk.into_iter().collect(),
                    ge,
                    align: al.into_iter().collect(),
                });
            }
            mb += g.n as f64 * e1;
            vb += g.n as f64 * (e2 - e1 * e1).max(0.0);
            groups.push(parts);
        }
        let mut own = vec![1.0; LN];
        for parts in &groups {
            for (i, o) in own.iter_mut().enumerate() {
                let gp: f64 = parts.iter().map(|p| p.w * p.ge[i].powi(p.n as i32)).sum();
                *o *= gp.clamp(0.0, 1.0);
            }
        }
        let mut levels: Vec<i32> = groups
            .iter()
            .flatten()
            .flat_map(|p| p.align.iter().map(|a| a.0))
            .filter(|&a| a > super::fine::NO_ALIGN)
            .collect();
        levels.sort();
        levels.dedup();
        let all_le = |e: i32| -> f64 {
            let mut f = 1.0;
            for parts in &groups {
                let gp: f64 = parts
                    .iter()
                    .map(|p| {
                        let le: f64 = p.align.iter().filter(|a| a.0 <= e).map(|a| a.1).sum();
                        p.w * le.powi(p.n as i32)
                    })
                    .sum();
                f *= gp.clamp(0.0, 1.0);
            }
            f
        };
        let fe: Vec<f64> = levels.iter().map(|&e| all_le(e)).collect();
        let dead = all_le(super::fine::NO_ALIGN);
        Step {
            frac: st.frac,
            mb,
            vb,
            s_abs,
            groups,
            own,
            levels,
            fe,
            dead,
            cache: Default::default(),
            cf_cache: Default::default(),
        }
    }

    /// The characteristic function (forward DFT, 2^(ue - gr) points, at most 64) of the
    /// step's terms truncated toward zero at 2^qe, floored to the 2^gr grid, mod 2^ue, given
    /// every lowest bit >= l.
    fn cf(&self, ue: i32, gr: i32, qe: i32, l: i32) -> Vec<C64> {
        let key = (ue, gr, qe, l);
        if let Some(r) = self.cf_cache.lock().unwrap().get(&key) {
            return r.clone();
        }
        let nb = 1usize << (ue - gr).clamp(0, 6);
        let li = lat_index(l);
        let mut tot = vec![C64::new(1.0, 0.0); nb];
        let g = 2f64.powi(gr);
        let qq = 2f64.powi(qe);
        for parts in &self.groups {
            let mut gsum = vec![C64::default(); nb];
            let mut pg = 0.0;
            for p in parts {
                let ml = p.ge[li];
                if ml <= 0.0 {
                    continue;
                }
                // Atoms with lowest bit >= ue (and zeros): multiples of the ulp, bin 0.
                let mut hist = vec![0.0; nb];
                hist[0] += p.ge[lat_index(ue.max(l))];
                for ((ex, lb), atoms) in &p.buckets {
                    if *lb < l || *lb >= ue {
                        continue;
                    }
                    for &(x, pr) in atoms {
                        let t = if *lb < qe {
                            if *ex < qe {
                                0.0
                            } else {
                                x.signum() * (x.abs() / qq).floor() * qq
                            }
                        } else {
                            x
                        };
                        let a = (t / g).floor() as i64;
                        hist[a.rem_euclid(nb as i64) as usize] += pr;
                    }
                }
                let mut f: Vec<C64> = hist.iter().map(|&h| C64::new(h / ml, 0.0)).collect();
                super::fine::fft(&mut f, false);
                let wg = p.w * ml.powi(p.n as i32);
                pg += wg;
                for (x, y) in gsum.iter_mut().zip(&f) {
                    *x = x.add(y.powi(p.n).scale(wg));
                }
            }
            if pg <= 0.0 {
                continue;
            }
            for (x, y) in tot.iter_mut().zip(&gsum) {
                *x = x.mul(y.scale(1.0 / pg));
            }
        }
        self.cf_cache.lock().unwrap().insert(key, tot.clone());
        tot
    }

    /// Truncation of the terms at 2^q given every lowest bit >= l: (E e, E e^2, P(e != 0)),
    /// e = -sum sign(t) (|t| mod 2^q).
    fn trunc(&self, q: i32, l: i32) -> (f64, f64, f64) {
        if let Some(r) = self.cache.lock().unwrap().get(&(q, l)) {
            return *r;
        }
        let r = self.trunc_raw(q, l);
        self.cache.lock().unwrap().insert((q, l), r);
        r
    }

    fn trunc_raw(&self, q: i32, l: i32) -> (f64, f64, f64) {
        let qq = 2f64.powi(q);
        let li = lat_index(l);
        let (mut m, mut v, mut lz) = (0.0, 0.0, 0.0);
        for parts in &self.groups {
            // Per term, conditional on the group's context mixture and lowest bit >= l.
            let mut pg = 0.0;
            let (mut a1, mut a2, mut ap) = (0.0, 0.0, 0.0);
            for p in parts {
                let ml = p.ge[li];
                if ml <= 0.0 {
                    continue;
                }
                let wg = p.w * ml.powi(p.n as i32);
                pg += wg;
                let (mut b1, mut b2, mut bp) = (0.0, 0.0, 0.0);
                for ((ex, lb), atoms) in &p.buckets {
                    if *lb < l || *lb >= q {
                        continue;
                    }
                    for &(x, pr) in atoms {
                        let r = if *ex < q { x.abs() } else { x.abs() % qq };
                        b1 += pr * x.signum() * r;
                        b2 += pr * r * r;
                        bp += pr * (r > 0.0) as u8 as f64;
                    }
                }
                a1 += wg * b1 / ml;
                a2 += wg * b2 / ml;
                ap += wg * bp / ml;
            }
            if pg <= 0.0 {
                continue;
            }
            let n = parts[0].n as f64;
            let (c1, c2, cp) = (a1 / pg, a2 / pg, (ap / pg).clamp(0.0, 1.0));
            m -= n * c1;
            v += n * (c2 - c1 * c1).max(0.0);
            lz += n * (1.0 - cp).max(1e-300).ln();
        }
        (m, v + m * m, 1.0 - lz.exp())
    }
}

/// Bins per ulp of y in the discrete error law; its largest size.
const G2: f64 = 256.0;
const G2_MAX: usize = 1 << 14;

/// A step's local error given its result cell and lattice class.
struct StepLaw {
    /// The final round toward zero: (error, P).
    disc: Vec<(f64, f64)>,
    /// The truncations: mean, E e^2, P(!= 0).
    m: f64,
    e2: f64,
    pnz: f64,
}

/// `step_law` cache key: (step, result binade, sign, lattice class, with the final rounding).
type StepKey = (usize, i32, i8, i32, bool);

/// A chain prepared for the tier.
pub struct ChainPre {
    steps: Vec<Step>,
    /// Cumulative means / variances of the exact partial sums.
    m: Vec<f64>,
    v: Vec<f64>,
    /// P(all terms up to and including step j have lowest bit >= x).
    upto: Vec<Vec<f64>>,
    pub round16: Option<Format>,
    pub s_mean: f64,
    cache: std::sync::Mutex<HashMap<StepKey, std::sync::Arc<StepLaw>>>,
}

impl ChainPre {
    pub fn new(sch: &ChainSchema) -> ChainPre {
        let steps: Vec<Step> = sch.steps.iter().map(Step::new).collect();
        let (mut m, mut v, mut upto) = (vec![], vec![], vec![]);
        let (mut cm, mut cv) = (0.0, 0.0);
        let mut cl = vec![1.0; LN];
        for s in &steps {
            cm += s.mb;
            cv += s.vb;
            m.push(cm);
            v.push(cv);
            cl = cl.iter().zip(&s.own).map(|(a, b)| a * b).collect();
            upto.push(cl.clone());
        }
        ChainPre {
            s_mean: steps.iter().map(|s| s.s_abs).sum(),
            steps,
            m,
            v,
            upto,
            round16: sch.round16,
            cache: Default::default(),
        }
    }

    fn y_mean(&self) -> f64 {
        *self.m.last().unwrap_or(&0.0)
    }
    fn y_var(&self) -> f64 {
        *self.v.last().unwrap_or(&0.0)
    }

    /// Binades (b, sign, p) of partial sum j given y.
    fn sum_given_y(&self, j: usize, y: f64) -> Vec<(i32, f64, f64)> {
        let n = self.steps.len();
        if j + 1 == n {
            let b = super::spec::exp_floor(y);
            return vec![(b, y.signum(), 1.0)];
        }
        let r = (self.v[j] / self.y_var().max(1e-300)).min(1.0);
        let mu = self.m[j] + r * (y - self.y_mean());
        let sd = (self.v[j] * (1.0 - r)).max(0.0).sqrt();
        if sd <= 1e-12 * mu.abs().max(1e-300) {
            if mu == 0.0 {
                return vec![];
            }
            return vec![(super::spec::exp_floor(mu), mu.signum(), 1.0)];
        }
        let mut out = vec![];
        for b in binades(mu, sd) {
            for s in [1.0f64, -1.0] {
                let p = binade_prob(mu, sd, b, s);
                if p > 1e-9 {
                    out.push((b, s, p));
                }
            }
        }
        out
    }

    /// The error law given y (binade b) and the lattice class "every term's lowest bit >=
    /// l": the steps' errors (`step_law` per binade of the partial sum given y), the last
    /// step's final rounding left out when `pre`. Discrete parts on a grid of ulp(y) / G2
    /// (a nonzero error below one bin keeps its sign as one bin).
    fn law(&self, y: f64, b: i32, l: i32, pre: bool) -> Law {
        let n = self.steps.len();
        let h = 2f64.powi(ulp_exp(b)) / G2;
        let mut disc: Vec<(f64, BTreeMap<i64, f64>)> = vec![];
        let (mut gm, mut gv, mut lq) = (0.0, 0.0, 0.0);
        for j in 0..n {
            let cur = self.sum_given_y(j, y);
            let tot: f64 = cur.iter().map(|x| x.2).sum::<f64>();
            if tot <= 0.0 {
                continue;
            }
            let rz = !(pre && j + 1 == n);
            let mut d: BTreeMap<i64, f64> = BTreeMap::new();
            let (mut nm, mut n2, mut np) = (0.0, 0.0, 0.0);
            for &(bs, ss, p) in &cur {
                let p = p / tot;
                let sl = self.step_law(j, bs, ss, l, rz);
                nm += p * sl.m;
                n2 += p * sl.e2;
                np += p * sl.pnz;
                for &(v, q) in &sl.disc {
                    let mut k = (v / h).round() as i64;
                    if k == 0 && v != 0.0 {
                        k = if v > 0.0 { 1 } else { -1 };
                    }
                    *d.entry(k).or_default() += p * q;
                }
            }
            gm += nm;
            gv += (n2 - nm * nm).max(0.0);
            lq += (1.0 - np.clamp(0.0, 1.0 - 1e-16)).ln();
            let dm: f64 = d.iter().map(|(k, q)| *k as f64 * h * q).sum();
            let d2: f64 = d.iter().map(|(k, q)| (*k as f64 * h).powi(2) * q).sum();
            if std::env::var("PTXV_T2BS_DEBUG2").is_ok() {
                eprintln!(
                    "  step {j} l {l} pre {pre}: disc mass {:.6} bins {} normal m {:.3e} sd {:.3e} p {:.3e}; cur {:?}",
                    d.values().sum::<f64>(),
                    d.len(),
                    nm,
                    (n2 - nm * nm).max(0.0).sqrt(),
                    np,
                    cur.iter()
                        .map(|c| (c.0, c.1 as i8, (c.2 * 1e3).round() / 1e3))
                        .collect::<Vec<_>>()
                );
            }
            if !d.is_empty() {
                disc.push((d2 - dm * dm, d));
            }
        }
        // The largest discrete step laws are convolved; the rest join the normal part.
        disc.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut acc: (i64, Vec<f64>) = (0, vec![1.0]);
        for (i, (_, d)) in disc.into_iter().enumerate() {
            let add: Vec<(i64, f64)> = d.into_iter().filter(|x| x.1 > 1e-15).collect();
            let span = add.iter().map(|x| x.0).max().unwrap_or(0)
                - add.iter().map(|x| x.0).min().unwrap_or(0);
            if i >= LAW_NODES || acc.1.len() + span as usize > G2_MAX {
                let m = add.iter().map(|x| x.0 as f64 * h * x.1).sum::<f64>();
                let m2 = add
                    .iter()
                    .map(|x| (x.0 as f64 * h).powi(2) * x.1)
                    .sum::<f64>();
                let p0 = add.iter().filter(|x| x.0 == 0).map(|x| x.1).sum::<f64>();
                gm += m;
                gv += (m2 - m * m).max(0.0);
                lq += p0.clamp(1e-16, 1.0).ln();
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

    /// Step j's local error given its result in binade bs (sign ss) and every lowest bit >=
    /// l: the final round toward zero as a discrete law (the residue of the truncated sum mod
    /// ulp from the terms' and the accumulator's characteristic functions; bits below 2^-6
    /// ulp uniform) when `rz`, and the truncations as a zero-inflated normal.
    fn step_law(&self, j: usize, bs: i32, ss: f64, l: i32, rz: bool) -> std::sync::Arc<StepLaw> {
        let key = (j, bs, ss as i8, l, rz);
        if let Some(x) = self.cache.lock().unwrap().get(&key) {
            return x.clone();
        }
        let x = std::sync::Arc::new(self.step_law_raw(j, bs, ss, l, rz));
        self.cache.lock().unwrap().insert(key, x.clone());
        x
    }

    fn step_law_raw(&self, j: usize, bs: i32, ss: f64, l: i32, rz: bool) -> StepLaw {
        let st = &self.steps[j];
        let ue = ulp_exp(bs);
        // The accumulator's binade given the result at the cell's middle.
        let prev: Vec<(i32, f64)> = if j == 0 {
            vec![]
        } else {
            let s = ss * 1.5 * 2f64.powi(bs);
            let r = (self.v[j - 1] / self.v[j].max(1e-300)).min(1.0);
            let mu = self.m[j - 1] + r * (s - self.m[j]);
            let sd = (self.v[j - 1] * (1.0 - r)).max(0.0).sqrt();
            let mut m: BTreeMap<i32, f64> = BTreeMap::new();
            if sd <= 1e-12 * mu.abs().max(1e-300) {
                if mu != 0.0 {
                    m.insert(super::spec::exp_floor(mu), 1.0);
                }
            } else {
                for b in binades(mu, sd) {
                    let p = binade_prob(mu, sd, b, 1.0) + binade_prob(mu, sd, b, -1.0);
                    if p > 1e-9 {
                        *m.entry(b).or_default() += p;
                    }
                }
            }
            let t: f64 = m.values().sum::<f64>().max(1e-300);
            m.into_iter().map(|(b, p)| (b, p / t)).collect()
        };
        // Alignment: e = max(largest term alignment, accumulator exponent); q = e - frac.
        // Combinations (q, accumulator binade) with weights.
        let mut combos: Vec<(i32, Option<i32>, f64)> = vec![];
        let mut elaw: Vec<(Option<i32>, f64)> = vec![];
        for (i, &e) in st.levels.iter().enumerate() {
            let p = st.fe[i] - if i == 0 { st.dead } else { st.fe[i - 1] };
            if p > 1e-12 {
                elaw.push((Some(e), p));
            }
        }
        if st.dead > 1e-12 {
            elaw.push((None, st.dead));
        }
        if prev.is_empty() {
            for &(e, p) in &elaw {
                if let Some(e) = e {
                    combos.push((e - st.frac, None, p));
                }
            }
        } else {
            for &(bp, pp) in &prev {
                for &(e, p) in &elaw {
                    let e = e.map_or(bp, |e| e.max(bp));
                    combos.push((e - st.frac, Some(bp), p * pp));
                }
            }
        }
        let ctot: f64 = combos.iter().map(|c| c.2).sum::<f64>().max(1e-300);
        let gr = l.max(ue - 6);
        let nb = 1usize << (ue - gr).clamp(0, 6);
        let mut disc: BTreeMap<i64, f64> = BTreeMap::new();
        let (mut m, mut e2, mut lz) = (0.0, 0.0, 0.0);
        let mut res: HashMap<(i32, i32, bool), (f64, Vec<f64>)> = HashMap::default();
        for &(q, bp, w) in &combos {
            let w = w / ctot;
            // Truncations (terms and the accumulator's bits below 2^q).
            let (a1, a2, ap) = st.trunc(q, l);
            let (mut c1, mut c2, mut cp) = (0.0, 0.0, 0.0);
            if let Some(bp) = bp {
                // The accumulator's lattice: max(previous terms' lattice, its ulp); its bits
                // below 2^q, uniform on that grid.
                let up = ulp_exp(bp);
                let qv = 2f64.powi(q);
                for lat in up.max(l)..q {
                    let pk = if lat == up.max(l) {
                        1.0 - self.before(j, lat + 1, l)
                    } else {
                        self.before(j, lat, l) - self.before(j, lat + 1, l)
                    };
                    if pk <= 1e-12 {
                        continue;
                    }
                    let (g1, g2, gp) = grid_moments((q - lat).min(40), true);
                    c1 += pk * ss * g1 * qv;
                    c2 += pk * g2 * qv * qv;
                    cp += pk * gp;
                }
            }
            let (t1, t2, tp) = (
                a1 + c1,
                a2 + c2 + 2.0 * a1 * c1,
                1.0 - (1.0 - ap) * (1.0 - cp),
            );
            let (mut s1, mut s2, mut sp) = (0.0, 0.0, 0.0);
            if rz && l < ue {
                // The residue of the truncated sum mod ulp on the 2^gr grid.
                let qe = q.clamp(gr, ue + 1);
                let ge = bp.map_or(ue + 1, |bp| ulp_exp(bp).max(q).clamp(gr, ue + 1));
                let sub = q.max(l) < gr;
                let (rw, r) = res
                    .entry((qe, ge, sub))
                    .or_insert_with(|| (0.0, self.residue(j, ue, gr, qe, ge, l)));
                *rw += w;
                let _ = r;
                // Bits below the grid (when the sum's lattice is finer): uniform within the
                // bin.
                if sub {
                    let g = 2f64.powi(gr);
                    let pf =
                        1.0 - self.upto[j][lat_index(gr)] / self.upto[j][lat_index(l)].max(1e-300);
                    s1 = -ss * g / 2.0 * pf;
                    s2 = g * g / 3.0 * pf;
                    sp = pf.clamp(0.0, 1.0);
                }
            }
            m += w * (t1 + s1);
            e2 += w * (t2 + s2 + 2.0 * t1 * s1);
            lz += w * (1.0 - (1.0 - tp) * (1.0 - sp)).clamp(0.0, 1.0);
        }
        for ((_, _, _), (w, r)) in res {
            let g = 2f64.powi(gr);
            for (i, &p) in r.iter().enumerate() {
                if p <= 1e-15 {
                    continue;
                }
                // Round toward zero: e = -(S mod ulp) for S > 0, +(|S| mod ulp) for S < 0.
                let e = if ss > 0.0 {
                    -(i as f64) * g
                } else {
                    ((nb - i) % nb) as f64 * g
                };
                *disc.entry(e.to_bits() as i64).or_default() += w * p;
            }
        }
        let mut disc: Vec<(f64, f64)> = disc
            .into_iter()
            .map(|(k, p)| (f64::from_bits(k as u64), p))
            .collect();
        // Combinations without a residue (the sum on the ulp grid): an exact rounding.
        let dm: f64 = disc.iter().map(|x| x.1).sum();
        if rz && dm < 1.0 {
            disc.push((0.0, 1.0 - dm));
        }
        // Mixture moments: the normal part's mean / E e^2 / P(!= 0) averaged over the
        // combinations (zero-inflated: P(!= 0) is the average).
        StepLaw {
            disc,
            m,
            e2,
            pnz: lz.clamp(0.0, 1.0),
        }
    }

    /// P(every term of the steps before j has lowest bit >= x | all >= l).
    fn before(&self, j: usize, x: i32, l: i32) -> f64 {
        if j == 0 {
            return 1.0;
        }
        self.upto[j - 1][lat_index(x.max(l))] / self.upto[j - 1][lat_index(l)].max(1e-300)
    }

    /// The law of (step j's terms truncated at 2^qe + the accumulator's bits from 2^ge) mod
    /// 2^ue on the 2^gr grid, given every lowest bit >= l (carries from below the grid
    /// ignored); P per bin (2^(ue - gr) bins).
    fn residue(&self, j: usize, ue: i32, gr: i32, qe: i32, ge: i32, l: i32) -> Vec<f64> {
        let nb = 1usize << (ue - gr).clamp(0, 6);
        let mut f = self.steps[j].cf(ue, gr, qe, l);
        if ge <= ue {
            for i in 0..j {
                let c = self.steps[i].cf(ue, ge.max(gr), ge.max(gr), l);
                let na = c.len();
                for (k, x) in f.iter_mut().enumerate() {
                    *x = x.mul(c[k % na]);
                }
            }
        }
        let mut v: Vec<C64> = f;
        super::fine::fft(&mut v, true);
        // Inverse DFT of the characteristic function (forward convention e^{-2 pi i}).
        v.iter().map(|c| (c.re / nb as f64).max(0.0)).collect()
    }
    /// y's lattice classes at binade b: (l, weight, positions of y in its ulp).
    fn classes(&self, b: i32) -> Vec<(i32, f64, Atoms)> {
        let ue = ulp_exp(b);
        let yl = self.upto.last().unwrap();
        let at = |x: i32| yl[lat_index(x)];
        let mut out = vec![(ue, at(ue), vec![(0.0, 1.0)])];
        for k in 1..=6 {
            let w = (at(ue - k) - at(ue - k + 1)).max(0.0);
            let n = 1usize << k;
            // An odd multiple of 2^-k ulp.
            let pos: Vec<(f64, f64)> = (0..n / 2)
                .map(|j| ((2 * j + 1) as f64 / n as f64, 2.0 / n as f64))
                .collect();
            out.push((ue - k, w, pos));
        }
        let w = (1.0 - at(ue - 6)).max(0.0);
        out.push((LMIN, w, super::tier2::positions(99)));
        out.retain(|c| c.1 > 1e-12);
        out
    }
}

/// Cells c of floor(t + x) for x from a law (ulps, oriented toward |y|): (first cell, P).
fn cells(t: f64, l: &Law, s: f64, u: f64) -> (i64, Vec<f64>) {
    let (gm, gs) = (s * l.gm / u, l.gv.max(0.0).sqrt() / u);
    let atoms: Vec<(f64, f64)> = l.atoms().map(|(x, q)| (s * x / u, q)).collect();
    if atoms.is_empty() {
        return (0, vec![1.0]);
    }
    let lo =
        atoms.iter().map(|a| a.0).fold(f64::INFINITY, f64::min) + t + gm.min(0.0) - 8.0 * gs - 1.0;
    let hi = atoms.iter().map(|a| a.0).fold(f64::NEG_INFINITY, f64::max)
        + t
        + gm.max(0.0)
        + 8.0 * gs
        + 1.0;
    let (c0, c1) = (lo.floor() as i64, hi.ceil() as i64);
    if c1 - c0 > 1 << 16 {
        // A spread far beyond the ulp: collapse to the mean cell.
        let m = t + atoms.iter().map(|a| a.0 * a.1).sum::<f64>() + l.gp * gm;
        return (m.floor() as i64, vec![1.0]);
    }
    let mut v = vec![0.0; (c1 - c0 + 1) as usize];
    for &(x0, q) in &atoms {
        let z = t + x0;
        if l.gp < 1.0 {
            let c = z.floor() as i64;
            if c >= c0 && c <= c1 {
                v[(c - c0) as usize] += q * (1.0 - l.gp);
            }
        }
        if l.gp > 0.0 {
            if gs <= 1e-12 {
                let c = (z + gm).floor() as i64;
                if c >= c0 && c <= c1 {
                    v[(c - c0) as usize] += q * l.gp;
                }
            } else {
                let a = ((z + gm - 8.0 * gs).floor() as i64).max(c0);
                let b = ((z + gm + 8.0 * gs).ceil() as i64).min(c1);
                for c in a..=b {
                    v[(c - c0) as usize] += q * l.gp * pn_in(z + gm, gs, c as f64, c as f64 + 1.0);
                }
            }
        }
    }
    (c0, v)
}

/// RN(t) over [0, 1): (cell, weight) (a tie rounds to even, either way).
fn rn_cells(t: f64) -> &'static [(i64, f64)] {
    if (t - 0.5).abs() < 1e-12 {
        &[(0, 0.5), (1, 0.5)]
    } else if t < 0.5 {
        &[(0, 1.0)]
    } else {
        &[(1, 1.0)]
    }
}

fn law_neg(l: &Law) -> Law {
    let mut p = l.p.clone();
    p.reverse();
    Law {
        h: l.h,
        off: -(l.off + l.p.len() as i64 - 1),
        p,
        gm: -l.gm,
        gv: l.gv,
        gp: l.gp,
    }
}

/// The law of a + b for independent laws on the same grid.
fn law_add(a: &Law, b: &Law) -> Law {
    let mut acc = (a.off, a.p.clone());
    let add: Vec<(i64, f64)> =
        b.p.iter()
            .enumerate()
            .filter(|x| *x.1 > 1e-15)
            .map(|(i, &q)| (i as i64 + b.off, q))
            .collect();
    law_conv(&mut acc, &add);
    // Zero-inflated normals: match the mean and variance of the sum.
    let (ma, va) = (
        a.gp * a.gm,
        a.gp * (a.gv + a.gm * a.gm) - (a.gp * a.gm).powi(2),
    );
    let (mb, vb) = (
        b.gp * b.gm,
        b.gp * (b.gv + b.gm * b.gm) - (b.gp * b.gm).powi(2),
    );
    let gp = 1.0 - (1.0 - a.gp) * (1.0 - b.gp);
    let (m, v) = (ma + mb, (va + vb).max(0.0));
    let (gm, gv) = if gp > 1e-300 {
        let mc = m / gp;
        (mc, ((v + m * m) / gp - mc * mc).max(0.0))
    } else {
        (0.0, 0.0)
    };
    Law {
        h: a.h,
        off: acc.0,
        p: acc.1,
        gm,
        gv,
        gp,
    }
}

/// Tier 2 on one or two MMA chains (single: C - exact; pair: C - R).
// ASSUME(N-ERRDIST-BSCLT): partial sums normal given y; steps independent given y and its
// lattice class; refused when y's excess kurtosis exceeds `tier2::KURT_MAX`.
pub fn estimate(r: &ChainPre, c: Option<&ChainPre>, out_fmt: Format) -> Estimate {
    let pair = c.is_some();
    let out16 = r
        .round16
        .or(c.and_then(|c| c.round16))
        .filter(|_| out_fmt != Format::F32);
    let mut pmf: BTreeMap<i64, f64> = BTreeMap::new();
    let mut tp = TentPmf::default();
    let mut wtot = 0.0;
    let (mut em, mut e2) = (0.0, 0.0);
    let dbg = std::env::var("PTXV_T2BS_DEBUG").is_ok();
    over_y_pts_w(r.y_mean(), r.y_var(), |y, s, b, wy| {
        let ue = ulp_exp(b);
        let u = 2f64.powi(ue);
        for (l, wc, pos) in r.classes(b) {
            let w = wy * wc;
            wtot += w;
            match (out16, c) {
                (Some(f), _) => {
                    let lr = r.law(y, b, l, false);
                    let d = match c {
                        None => lr,
                        Some(c) => law_add(&c.law(y, b, l, false), &law_neg(&lr)),
                    };
                    let (m, v) = law_moments(&d);
                    em += w * m;
                    e2 += w * (v + m * m);
                    let uo = 2f64.powi(b.max(f.emin()) - f.mbits());
                    let gs = d.gv.max(0.0).sqrt() / uo;
                    for (x, q) in d.atoms() {
                        let x0 = x / uo;
                        if d.gp < 1.0 {
                            tp.point(x0, w * q * (1.0 - d.gp), 1.0);
                        }
                        if d.gp > 0.0 {
                            tp.normal(x0 + d.gm / uo, gs, w * q * d.gp, 1.0);
                        }
                    }
                }
                (None, None) => {
                    let lr = r.law(y, b, l, true);
                    let (m, v) = law_moments(&lr);
                    em += w * m;
                    e2 += w * (v + m * m);
                    let mut same = 0.0;
                    for &(t, wt) in &pos {
                        let (c0, cv) = cells(t, &lr, s, u);
                        for &(base, wb) in rn_cells(t) {
                            for (i, &q) in cv.iter().enumerate() {
                                let k = (c0 + i as i64 - base) * s as i64;
                                *pmf.entry(k).or_default() += w * wt * wb * q;
                                if k == 0 {
                                    same += wt * wb * q;
                                }
                            }
                        }
                    }
                    if dbg {
                        eprintln!(
                            "T2BS y {y:.3e} b {b} class {} w {:.3e}: P(differ) {:.4e}, law gp {:.3e} gm {:.3e} gs {:.3e} atoms {:.80}",
                            ue - l,
                            w,
                            1.0 - same,
                            lr.gp,
                            lr.gm / u,
                            lr.gv.sqrt() / u,
                            lr.atoms()
                                .filter(|a| a.1 > 1e-6)
                                .map(|a| format!("{:.4}:{:.3e}", a.0 / u, a.1))
                                .collect::<Vec<_>>()
                                .join(" ")
                        );
                    }
                }
                (None, Some(c)) => {
                    let (lr, lc) = (r.law(y, b, l, true), c.law(y, b, l, true));
                    let (mr, vr) = law_moments(&lr);
                    let (mc, vc) = law_moments(&lc);
                    em += w * (mc - mr);
                    e2 += w * (vr + vc + (mc - mr).powi(2));
                    for &(t, wt) in &pos {
                        let (r0, rv) = cells(t, &lr, s, u);
                        let (c0, cvv) = cells(t, &lc, s, u);
                        for (i, &qr) in rv.iter().enumerate() {
                            if qr <= 1e-15 {
                                continue;
                            }
                            for (j, &qc) in cvv.iter().enumerate() {
                                let k = ((c0 + j as i64) - (r0 + i as i64)) * s as i64;
                                *pmf.entry(k).or_default() += w * wt * qr * qc;
                            }
                        }
                    }
                }
            }
        }
        0.0
    });
    let pmf: Vec<(i64, f64)> = if out16.is_some() {
        tp.finish(None).0
    } else {
        let t: f64 = pmf.values().sum::<f64>().max(1e-300);
        pmf.into_iter()
            .map(|(k, q)| (k, q / t))
            .filter(|x| x.1 > 1e-12)
            .collect()
    };
    let p = (1.0 - pmf.iter().filter(|x| x.0 == 0).map(|x| x.1).sum::<f64>()).clamp(0.0, 1.0);
    let wt = wtot.max(1e-300);
    let (m, v) = (em / wt, (e2 / wt - (em / wt).powi(2)).max(0.0));
    let mut est = Estimate {
        tier: 2,
        method: format!(
            "CLT on the MMA chain{}: one alignment truncation and one round toward zero per step, conditioned on y and on y's lattice class; step residues discrete, truncations normal",
            if pair { "s" } else { "" }
        ),
        outputs: 1,
        abs: moments(m, v),
        u_scale: Some(u_scale(m, v, r.s_mean)),
        error_bar: ErrorBar {
            kind: "clt".into(),
            notes: vec![
                "partial sums normal given y; every term's lowest bit >= y's lattice class; steps independent given y and the class (N-ERRDIST-CLT, N-ERRDIST-BSCLT)".into(),
            ],
            ..Default::default()
        },
        ..Default::default()
    };
    if pair {
        est.pair = Some(PairMoments::default());
        // Pair outputs are on the lattice: ulp-scale moments from the PMF.
        est.ulp = Some(super::combine::ulp_moments(
            &est,
            &super::combine::Law::from(pmf.iter().map(|x| (x.0 as f64, x.1)).collect()),
        ));
    }
    est.pmf_ulp = pmf;
    est.p_differ = Some((p, p, p));
    est
}

/// The excess kurtosis of y, the chain's exact sum, from its terms: independent groups,
/// each a mixture over its parts (a shared context: a block scale, an outlier channel) of a
/// sum of `n` i.i.d. terms. The tier treats the partial sums as normal given y; a heavy-tailed
/// or scale-mixture y breaks that, as for the DAG CLT (`tier2::KURT_MAX`).
pub fn y_kurtosis(sch: &ChainSchema) -> f64 {
    let (mut k2, mut k4) = (0.0, 0.0);
    for st in &sch.steps {
        for g in &st.pop.groups {
            // Raw moments of the group sum, mixed over the parts.
            let mut raw = [0.0f64; 5];
            let mut wt = 0.0;
            for (w, t) in &g.parts {
                let (mut m, mut c2, mut c3, mut c4, mut mass) = (0.0, 0.0, 0.0, 0.0, 0.0);
                for &(v, _, p) in &t.atoms {
                    m += p * v;
                    mass += p;
                }
                if mass <= 0.0 {
                    continue;
                }
                m /= mass;
                for &(v, _, p) in &t.atoms {
                    let d = v - m;
                    c2 += p * d * d / mass;
                    c3 += p * d * d * d / mass;
                    c4 += p * d.powi(4) / mass;
                }
                let n = g.n as f64;
                // Cumulants of the sum of n i.i.d. terms.
                let (q1, q2, q3, q4) = (n * m, n * c2, n * c3, n * (c4 - 3.0 * c2 * c2));
                // Raw moments from cumulants.
                let r1 = q1;
                let r2 = q2 + q1 * q1;
                let r3 = q3 + 3.0 * q2 * q1 + q1.powi(3);
                let r4 = q4 + 4.0 * q3 * q1 + 3.0 * q2 * q2 + 6.0 * q2 * q1 * q1 + q1.powi(4);
                for (i, r) in [1.0, r1, r2, r3, r4].into_iter().enumerate() {
                    raw[i] += w * r;
                }
                wt += w;
            }
            if wt <= 0.0 {
                continue;
            }
            let [_, r1, r2, r3, r4] = raw.map(|x| x / wt);
            let c2 = r2 - r1 * r1;
            let c4 = r4 - 4.0 * r3 * r1 + 6.0 * r2 * r1 * r1 - 3.0 * r1.powi(4) - 3.0 * c2 * c2;
            k2 += c2;
            k4 += c4;
        }
    }
    if k2 > 0.0 { k4 / (k2 * k2) } else { 0.0 }
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! From linearized local errors to output error distributions (Tiers 1 and 2 of non-chain
//! programs).
//!
//! Per output, the pre-rounding deviation in output ulps of y's binade is
//! d = (v^ - y) / ulp(y) = sum_k c_k delta_k / ulp(y) (`lin.rs`); for a pair the two kernels'
//! sites are merged by canonical hash (a sub-computation both kernels share has the same local
//! error in both), so D = d_C - d_R. The output's position u = frac(y / ulp) is uniform and
//! independent of the local errors (on y's lattice when y is a lattice value), so
//! - the integer offset K = y^ / ulp - RN(y) / ulp (pair: C - R) is floor(D') + Bernoulli(frac
//!   D') with D' = D plus the rounding modes' shifts, and P(differ) = P(K != 0);
//! - a single kernel's error is d + V, V uniform on the final rounding's cell (RN: (-1/2, 1/2]).
//!
//! A final slice adds its table cell's integer offsets from the correctly rounded value.
//!
//! Tier 2 takes d normal with the sites' exact means and variances. Tier 1 convolves the
//! dominant discrete site laws exactly (residue-aware rounding residuals, slice table cells)
//! and the rest as a normal.

use std::collections::BTreeMap;
use std::collections::HashMap;

/// A matched pair's integer law of (C - R) in units of its grid q, the grid q, and the
/// reference's own deviation variance there.
type PairLaw = (Vec<(i64, f64)>, f64, f64);

use super::super::mc::U;
use super::super::mc::ulp_out;
use super::super::report::ErrorBar;
use super::super::report::Estimate;
use super::super::report::Moments;
use super::super::report::PairMoments;
use super::super::spec::Format;
use super::super::spec::Layout;
use super::super::spec::Spec;
use super::lin::Final;
use super::lin::KModel;
use super::lin::Law;
use super::lin::State;
use super::table::Table;

static JDBG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Analytic tier configuration.
#[derive(Clone, Debug)]
pub struct Cfg {
    /// 1 (exact convolution of dominant sites) or 2 (normal).
    pub tier: u8,
    /// Nominal inputs (real arithmetic only) the conditional laws are averaged over.
    pub inputs: usize,
    /// Outputs per input (evenly spaced subset).
    pub outputs: usize,
    /// Dominant sites convolved exactly (tier 1).
    pub exact_sites: usize,
}

impl Default for Cfg {
    fn default() -> Cfg {
        Cfg {
            tier: 1,
            inputs: 48,
            outputs: 256,
            exact_sites: 6,
        }
    }
}

/// A law on a grid: probability of x0 + i h.
#[derive(Clone, Debug)]
struct Grid {
    x0: f64,
    h: f64,
    p: Vec<f64>,
}

impl Grid {
    fn point(x: f64, h: f64) -> Grid {
        Grid {
            x0: (x / h).round() * h,
            h,
            p: vec![1.0],
        }
    }
    /// Convolve with atoms (value, prob), linear interpolation onto the grid.
    fn conv_atoms(&self, atoms: &[(f64, f64)]) -> Grid {
        let lo = atoms.iter().map(|a| a.0).fold(f64::INFINITY, f64::min);
        let hi = atoms.iter().map(|a| a.0).fold(f64::NEG_INFINITY, f64::max);
        let h = self.h;
        let x0 = self.x0 + (lo / h).floor() * h;
        let n = self.p.len() + ((hi - lo) / h).ceil() as usize + 2;
        let mut p = vec![0.0; n];
        for &(a, w) in atoms {
            let off = (self.x0 + a - x0) / h;
            let i0 = off.floor();
            let f = off - i0;
            let i0 = i0 as usize;
            for (j, q) in self.p.iter().enumerate() {
                if *q == 0.0 {
                    continue;
                }
                p[i0 + j] += q * w * (1.0 - f);
                if f > 0.0 {
                    p[i0 + j + 1] += q * w * f;
                }
            }
        }
        let mut g = Grid { x0, h, p };
        g.trim();
        g
    }
    /// Convolve with N(m, v).
    fn conv_normal(&self, m: f64, v: f64) -> Grid {
        let h = self.h;
        let sd = v.sqrt();
        if sd < h / 4.0 {
            return self.conv_atoms(&[(m, 1.0)]);
        }
        let k = (6.0 * sd / h).ceil() as i64;
        let mut ker: Vec<(f64, f64)> = Vec::with_capacity((2 * k + 1) as usize);
        let mut tot = 0.0;
        for i in -k..=k {
            let x = i as f64 * h;
            let w = (-(x * x) / (2.0 * v)).exp();
            ker.push((m + x, w));
            tot += w;
        }
        for x in ker.iter_mut() {
            x.1 /= tot;
        }
        self.conv_atoms(&ker)
    }
    fn trim(&mut self) {
        let a = self.p.iter().position(|x| *x > 1e-15).unwrap_or(0);
        let b = self.p.iter().rposition(|x| *x > 1e-15).map_or(0, |x| x + 1);
        if a > 0 || b < self.p.len() {
            self.p = self.p[a..b.max(a)].to_vec();
            self.x0 += a as f64 * self.h;
        }
        if self.p.is_empty() {
            self.p.push(1.0);
        }
    }
    fn atoms(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.p
            .iter()
            .enumerate()
            .filter(|x| *x.1 > 0.0)
            .map(|(i, p)| (self.x0 + i as f64 * self.h, *p))
    }
}

/// Pooled log-scale histogram of a signed quantity (for quantiles in abs / u units).
#[derive(Clone, Debug, Default)]
struct LogHist {
    zero: f64,
    pos: BTreeMap<i32, f64>,
    neg: BTreeMap<i32, f64>,
}

const LB: f64 = 16.0; // bins per octave

impl LogHist {
    fn add(&mut self, x: f64, w: f64) {
        if w <= 0.0 || !x.is_finite() {
            return;
        }
        if x == 0.0 {
            self.zero += w;
            return;
        }
        let b = (x.abs().log2() * LB).floor() as i32;
        *if x > 0.0 {
            &mut self.pos
        } else {
            &mut self.neg
        }
        .entry(b)
        .or_default() += w;
    }
    fn quantile(&self, q: f64) -> f64 {
        let tot = self.zero + self.pos.values().sum::<f64>() + self.neg.values().sum::<f64>();
        let target = q * tot;
        let mut acc = 0.0;
        for (b, w) in self.neg.iter().rev() {
            if acc + w >= target {
                let f = (target - acc) / w;
                // Within the bin from its larger magnitude to its smaller.
                let lo = 2f64.powf((*b as f64 + 1.0) / LB);
                let hi = 2f64.powf(*b as f64 / LB);
                return -(lo + (hi - lo) * f);
            }
            acc += w;
        }
        if acc + self.zero >= target {
            return 0.0;
        }
        acc += self.zero;
        for (b, w) in &self.pos {
            if acc + w >= target {
                let f = (target - acc) / w;
                let lo = 2f64.powf(*b as f64 / LB);
                let hi = 2f64.powf((*b as f64 + 1.0) / LB);
                return lo + (hi - lo) * f;
            }
            acc += w;
        }
        self.pos
            .keys()
            .next_back()
            .map_or(0.0, |b| 2f64.powf((*b as f64 + 1.0) / LB))
    }
}

/// Mixture moments accumulator.
#[derive(Clone, Debug, Default)]
struct Mix {
    w: f64,
    s1: f64,
    s2: f64,
    sa: f64,
    zero: f64,
    max_abs: f64,
    hist: LogHist,
}

impl Mix {
    /// Add one output's law given as atoms (value, prob) with a uniform spread [a, b) around
    /// each atom (b == a: point masses), times `scale`.
    fn add_law(&mut self, atoms: &[(f64, f64)], a: f64, b: f64, scale: f64) {
        for &(x, p) in atoms {
            let (lo, hi) = ((x + a) * scale, (x + b) * scale);
            let m = 0.5 * (lo + hi);
            let v = (hi - lo).powi(2) / 12.0;
            self.w += p;
            self.s1 += p * m;
            self.s2 += p * (v + m * m);
            // Mean |.| of a uniform on [lo, hi].
            let ma = if lo >= 0.0 || hi <= 0.0 {
                m.abs()
            } else {
                (lo * lo + hi * hi) / (2.0 * (hi - lo))
            };
            self.sa += p * ma;
            self.max_abs = self.max_abs.max(lo.abs().max(hi.abs()));
            if lo == hi {
                if lo == 0.0 {
                    self.zero += p;
                }
                self.hist.add(lo, p);
            } else {
                let k = 8;
                for j in 0..k {
                    self.hist
                        .add(lo + (hi - lo) * (j as f64 + 0.5) / k as f64, p / k as f64);
                }
            }
        }
    }
    fn moments(&self) -> Moments {
        let w = self.w.max(1e-300);
        let mean = self.s1 / w;
        let var = (self.s2 / w - mean * mean).max(0.0);
        Moments {
            mean,
            var,
            std: var.sqrt(),
            mean_abs: self.sa / w,
            p_zero: self.zero / w,
            quantiles: super::super::stats::QUANTILES
                .iter()
                .map(|&q| (q, self.hist.quantile(q)))
                .collect(),
            max_abs: Some(self.max_abs),
        }
    }
}

/// The site class of an op (variance budget).
fn class(k: &KModel, i: u32) -> String {
    use super::super::chain::Op;
    use super::ops::NlOp;
    match &k.p.ops[i as usize] {
        Op::Add { .. } => "f32 add".into(),
        Op::Fma { c: None, .. } => "f32 mul".into(),
        Op::Fma { .. } => "f32 fma".into(),
        Op::Round { f16: true, .. } => "f16 rounding".into(),
        Op::Round { .. } => "bf16 rounding".into(),
        Op::Mma { .. } => "tensor-core step".into(),
        Op::Nl(NlOp::Seg { real: Some(f), .. }) => format!("slice {}", f.func.name()),
        Op::Nl(NlOp::Mufu { func, .. }) => format!("MUFU.{func:?}"),
        _ => "other".into(),
    }
}

/// One output's analysis at one input.
struct OutLaw {
    /// Single kernel, 16-bit output rounded from a continuous f32 value: the probability that
    /// the computed f32 value is a tie on the output grid (2^-(23 - mbits)), where RNE picks
    /// the even neighbour whatever side the exact value is on.
    tie: f64,
    /// The law of Z (pre-rounding deviation plus the rounding shifts) in output ulps:
    /// K = floor(w + Z) - floor(w), w uniform. A single kernel's d is Z - shift.
    d: Grid,
    shift: f64,
    /// Integer offsets of final slices (convolved into K), C minus R.
    off: Vec<(i64, f64)>,
    /// Lattice residues of y on the output grid (0: continuous position).
    r: u64,
    /// Single: the final rounding's error cell [a, b) in ulps.
    cell: (f64, f64),
    ulp: f64,
    s: f64,
    /// Identical results (shared final op): K = 0.
    identical: bool,
    /// Per-kernel moments against exact (ulp units): (mean_r, var_r, mean_c, var_c, cov).
    pm: (f64, f64, f64, f64, f64),
}

struct Site {
    key: u64,
    c: f64,
    law: Law,
    class: String,
    /// Kernel (0: reference, 1: candidate) and op.
    kern: u8,
    op: u32,
}

fn final_shift(f: Final) -> (f64, (f64, f64)) {
    match f {
        Final::Rn | Final::Slice { .. } | Final::SlicePre { .. } => (0.5, (-0.5, 0.5)),
        Final::Rz => (0.0, (-1.0, 0.0)),
        Final::Exact => (0.5, (0.0, 0.0)),
    }
}

/// Integer offsets law of a final slice cell, from its table.
fn slice_off(k: &KModel, f: Final) -> Vec<(i64, f64)> {
    match f {
        Final::Slice { tab, cell } => {
            let c = &k.tabs[tab as usize].cells[cell as usize];
            c.off
                .iter()
                .enumerate()
                .filter(|x| *x.1 > 0.0)
                .map(|(i, p)| (i as i64 - 4, *p as f64))
                .collect()
        }
        _ => vec![(0, 1.0)],
    }
}

fn conv_int(a: &[(i64, f64)], b: &[(i64, f64)]) -> Vec<(i64, f64)> {
    let mut m: BTreeMap<i64, f64> = BTreeMap::new();
    for &(x, p) in a {
        for &(y, q) in b {
            *m.entry(x + y).or_default() += p * q;
        }
    }
    m.into_iter().collect()
}

/// Matched roundings of a pair: ops of the two kernels that round the same real value with
/// different computations (canonical keys), found by equal real values at two inputs.
pub struct Matches {
    pub pairs: Vec<(u64, u64)>,
    pub by_r: HashMap<u64, usize>,
    pub by_c: HashMap<u64, usize>,
}

/// Per-input context of the linearized analysis.
struct Ctx<'a> {
    cfg: &'a Cfg,
    r: &'a KModel<'a>,
    sr: &'a State,
    /// `Trajectory::Both` (single kernels): the nominal leaves and the real-trajectory state,
    /// for the per-site sensitivity selection.
    both: Option<(&'a [crate::mma_lp::Operand], &'a State)>,
    c: Option<(&'a KModel<'a>, &'a State)>,
    m: &'a Matches,
    /// Slice tables computing the same function of their index value.
    same: &'a [(u32, u32)],
    /// Joint tables of slice pairs.
    ptabs: &'a [(u32, u32, Table)],
    /// An op of each kernel per canonical key (copies of a value share their key).
    op_r: &'a HashMap<u64, u32>,
    op_c: &'a HashMap<u64, u32>,
    /// Per matched pair: the integer law of (C - R) at it in units of its grid q, its q, and
    /// the reference's own deviation variance there.
    memo: HashMap<usize, std::rc::Rc<PairLaw>>,
    cones_r: HashMap<u32, std::rc::Rc<Vec<u32>>>,
    cones_c: HashMap<u32, std::rc::Rc<Vec<u32>>>,
    adj: Vec<f64>,
    depth: usize,
    /// The measured tie rate of 16-bit final roundings (single kernels).
    tie: f64,
}

/// `Trajectory::Both`, flagged sites: `PTXV_ERRDIST_BOTH_FLAGGED=real` takes the real
/// trajectory's sensitivity; the default (`loo`) the float one without the site's own rounding.
fn both_flagged_real() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("PTXV_ERRDIST_BOTH_FLAGGED").as_deref() == Ok("real"))
}

fn site_of(
    k: &KModel,
    s: &State,
    i: u32,
    g: f64,
    sign: f64,
    unit: f64,
) -> Result<Option<Site>, &'static str> {
    Ok(match &s.law[i as usize] {
        Law::Zero => None,
        Law::Missing => return Err("a local error law is missing"),
        l => Some(Site {
            key: k.canon[i as usize],
            c: sign * g / unit,
            law: l.clone(),
            class: class(k, i),
            kern: (sign > 0.0) as u8,
            op: i,
        }),
    })
}

impl<'a> Ctx<'a> {
    fn cone(&mut self, kern: u8, op: u32) -> std::rc::Rc<Vec<u32>> {
        let (k, m) = if kern == 0 {
            (self.r, &mut self.cones_r)
        } else {
            (self.c.unwrap().0, &mut self.cones_c)
        };
        m.entry(op)
            .or_insert_with(|| std::rc::Rc::new(k.cone(op)))
            .clone()
    }

    /// The sites of the deviation at (root_r, root_c) below the roots' own errors, in units
    /// of `unit`: unmatched sites of both kernels (the reference's with sign -1), and for
    /// each matched pair below, the joint site of (C - R) there and the reference's own
    /// deviation with the leftover adjoint.
    fn collect(
        &mut self,
        root_r: u32,
        root_c: Option<u32>,
        unit: f64,
        sign_r: f64,
    ) -> Result<Vec<Site>, &'static str> {
        let cone_r = self.cone(0, root_r);
        let cone_c = root_c.map(|o| self.cone(1, o));
        // The matched pairs present in both cones (not the roots).
        let mut cut_r: HashMap<u64, usize> = HashMap::new();
        let mut cut_c: HashMap<u64, usize> = HashMap::new();
        if let (Some((ck, _)), Some(cc)) = (self.c, cone_c.as_ref()) {
            if !self.m.pairs.is_empty() && self.depth < max_depth() {
                let kr: std::collections::HashSet<u64> = cone_r[1..]
                    .iter()
                    .map(|&i| self.r.canon[i as usize])
                    .filter(|k| self.m.by_r.contains_key(k))
                    .collect();
                let sc = self.c.unwrap().1;
                for &i in &cc[1..] {
                    let kc = ck.canon[i as usize];
                    if let Some(&p) = self.m.by_c.get(&kc) {
                        // The pair must hold the same real value at this input too.
                        let vr = self.sr.v[self.op_r[&self.m.pairs[p].0] as usize];
                        let vc = sc.v[i as usize];
                        if (vr - vc).abs() > 2f64.powi(-20) * vr.abs() {
                            continue;
                        }
                        if kr.contains(&self.m.pairs[p].0) {
                            cut_r.insert(self.m.pairs[p].0, p);
                            cut_c.insert(kc, p);
                        }
                    }
                }
            }
        }
        let mut sites = Vec::new();
        let mut adj = std::mem::take(&mut self.adj);
        let r = self.r;
        let (sr_list, cr_list) = r.adjoints_cut(self.sr, &cone_r, &mut adj, &|i| {
            cut_r.contains_key(&r.canon[i as usize])
        });
        // Trajectory::Both: the real trajectory's sensitivities, to flag the sites whose float
        // sensitivity differs.
        let real_g: Option<HashMap<u32, f64>> = match (self.both, self.c) {
            (Some((_, sreal)), None) => {
                let mut a2 = Vec::new();
                Some(r.adjoints(sreal, &cone_r, &mut a2).into_iter().collect())
            }
            _ => None,
        };
        for (i, g) in sr_list {
            if i == root_r {
                continue;
            }
            let g = match (&real_g, self.both) {
                (Some(rg), Some((wr, _))) if !matches!(self.sr.law[i as usize], Law::Zero) => {
                    let gr = rg.get(&i).copied().unwrap_or(0.0);
                    if (g - gr).abs() <= 2f64.powi(-10) * (g.abs() + gr.abs()) {
                        g
                    } else if both_flagged_real() {
                        gr
                    } else {
                        // Flagged: the adjoint with op i's own rounding left out.
                        let sk = r.state_keep(wr, i);
                        let mut a2 = Vec::new();
                        r.adjoints(&sk, &cone_r, &mut a2)
                            .into_iter()
                            .find(|x| x.0 == i)
                            .map_or(0.0, |x| x.1)
                    }
                }
                _ => g,
            };
            if let Some(x) = site_of(r, self.sr, i, g, sign_r, unit)? {
                sites.push(x);
            }
        }
        let mut gs: HashMap<usize, (f64, f64)> = HashMap::new();
        for (i, g) in cr_list {
            gs.entry(cut_r[&r.canon[i as usize]]).or_default().0 += g;
        }
        if let (Some((ck, sc)), Some(cc)) = (self.c, cone_c.as_ref()) {
            let (sc_list, cc_list) = ck.adjoints_cut(sc, cc, &mut adj, &|i| {
                cut_c.contains_key(&ck.canon[i as usize])
            });
            for (i, g) in sc_list {
                if Some(i) == root_c {
                    continue;
                }
                if let Some(x) = site_of(ck, sc, i, g, 1.0, unit)? {
                    sites.push(x);
                }
            }
            for (i, g) in cc_list {
                gs.entry(cut_c[&ck.canon[i as usize]]).or_default().1 += g;
            }
        }
        self.adj = adj;
        for (p, (g_r, g_c)) in gs {
            let jl = self.joint(p)?;
            let (kl, q, var_r) = (&jl.0, jl.1, jl.2);
            sites.push(Site {
                key: 0x6a01_75e5 ^ (p as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                c: g_c / unit,
                law: Law::Joint { q, k: kl.clone() },
                class: "matched roundings (joint)".into(),
                kern: 1,
                op: 0,
            });
            if g_c != g_r {
                sites.push(Site {
                    key: 0x51ed ^ (p as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f),
                    c: (g_c - g_r) / unit,
                    law: Law::Normal {
                        mean: 0.0,
                        var: var_r,
                    },
                    class: "matched roundings (reference part)".into(),
                    kern: 0,
                    op: 0,
                });
            }
        }
        Ok(sites)
    }

    // ASSUME(N-ERRDIST-NL-JOINT): matched roundings share their grid position.
    /// The law of (C - R) at matched pair p, in units of its rounding grid q.
    fn joint(&mut self, p: usize) -> Result<std::rc::Rc<PairLaw>, &'static str> {
        if let Some(x) = self.memo.get(&p) {
            return Ok(x.clone());
        }
        let (kr, kc) = self.m.pairs[p];
        let (or_, oc) = (self.op_r[&kr], self.op_c[&kc]);
        let (ck, sc) = self.c.unwrap();
        let lr = &self.sr.law[or_ as usize];
        let lc = &sc.law[oc as usize];
        // An exact op (no rounding at this input) is a rounding with no pre-rounding error.
        let pre = |l: &Law, v: f64, t: &[Table]| match l {
            Law::Zero => Some((ulp_out(v, Format::F32), vec![(0.0, 1.0)])),
            l => l.pre_rounding(t),
        };
        let (Some((q, er)), Some((qc, ec))) = (
            pre(lr, self.sr.v[or_ as usize], self.r.tabs),
            pre(lc, sc.v[oc as usize], ck.tabs),
        ) else {
            return Err("a matched rounding without a pre-rounding law");
        };
        let _ = qc;
        self.depth += 1;
        let mut sites = self.collect(or_, Some(oc), q, -1.0)?;
        self.depth -= 1;
        // The two roundings' own pre-rounding errors (grid units); they cancel when both are
        // the same function of their index value.
        let ident = match (lr, lc) {
            (Law::Cell { tab: a, .. }, Law::Cell { tab: b, .. }) => {
                a == b || self.same.contains(&((*a).min(*b), (*a).max(*b)))
            }
            // Two correctly rounded operations (add, mul, fma) of nearly the same exact value
            // share their position on the grid: the difference is the upstream deviation's
            // rounding, round(w + Z) - round(w), with no separate pre-rounding errors.
            (Law::Rn { .. } | Law::Zero, Law::Rn { .. } | Law::Zero) => {
                std::env::var("PTXV_NL_RN_INDEP").is_err()
            }
            _ => false,
        };
        // The candidate's slice may be bit-for-bit the reference's sub-DAG above the nodes
        // holding its arguments (a fused mean / eps / rsqrt against three launches): then
        // the reference's roundings in between are the slice's own and cancel with it.
        let mut ident = ident;
        let same_tab =
            matches!((lr, lc), (Law::Cell { tab: a, .. }, Law::Cell { tab: b, .. }) if a == b);
        if !same_tab {
            let ci = self.composite_ident(or_, oc);
            if std::env::var("PTXV_NL_COMP").is_ok() {
                eprintln!(
                    "composite r {} c {}: {}",
                    format!("{:?}", self.r.p.ops[or_ as usize])
                        .chars()
                        .take(60)
                        .collect::<String>(),
                    format!("{:?}", ck.p.ops[oc as usize])
                        .chars()
                        .take(60)
                        .collect::<String>(),
                    ci.as_ref().map_or(-1, |b| b.len() as i64)
                );
            }
            if let Some(between) = ci {
                ident = true;
                sites.retain(|s| !(s.kern == 0 && between.contains(&s.op)));
            }
        }
        // A joint table of the two slices: the integer law of g_C - g_R at equal real
        // arguments, convolved below with the upstream deviation's rounding.
        let mut pair_law: Option<Vec<(i64, f64)>> = None;
        if let (Law::Cell { tab: a, .. }, Law::Cell { tab: b, .. }) = (lr, lc) {
            if !ident {
                for (x, y, t) in self.ptabs {
                    let flip = if (*x, *y) == (*a, *b) {
                        false
                    } else if (*x, *y) == (*b, *a) {
                        true
                    } else {
                        continue;
                    };
                    let f = self.r.p.ops[or_ as usize].clone();
                    let tv = match &f {
                        super::super::chain::Op::Nl(super::ops::NlOp::Seg {
                            args,
                            real: Some(rf),
                            ..
                        }) => {
                            let av: Vec<f64> =
                                args.iter().map(|&j| self.sr.v[j as usize]).collect();
                            super::table::index_value(rf, &av)
                        }
                        _ => continue,
                    };
                    if let Some(ci) = t.lookup(tv) {
                        let c = &t.cells[ci];
                        pair_law = Some(
                            c.off
                                .iter()
                                .enumerate()
                                .filter(|z| *z.1 > 0.0)
                                .map(|(i, p)| {
                                    let k = i as i64 - 4;
                                    (if flip { -k } else { k }, *p as f64)
                                })
                                .collect(),
                        );
                    }
                    break;
                }
            }
        }
        if !ident && pair_law.is_none() {
            sites.push(Site {
                key: 0x7e7a ^ kr,
                c: -1.0,
                law: Law::Atoms(er),
                class: "pre-rounding".into(),
                kern: 0,
                op: or_,
            });
            sites.push(Site {
                key: 0x7e7b ^ kc,
                c: 1.0,
                law: Law::Atoms(ec),
                class: "pre-rounding".into(),
                kern: 1,
                op: oc,
            });
        }
        let var_r = lr.moments(self.r.tabs).1;
        if std::env::var("PTXV_NL_JOINT2").is_ok()
            && JDBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 6
        {
            for (kk, oo, st) in [(self.r, or_, self.sr), (ck, oc, sc)] {
                if let super::super::chain::Op::Nl(super::ops::NlOp::Seg {
                    args,
                    real: Some(rf),
                    ..
                }) = &kk.p.ops[oo as usize]
                {
                    let av: Vec<String> = args
                        .iter()
                        .map(|&j| {
                            format!("op{j}={:e} {:?}", st.v[j as usize], kk.p.ops[j as usize])
                                .chars()
                                .take(160)
                                .collect()
                        })
                        .collect();
                    eprintln!("  {} args {:?} pair_law {:?}", rf.show(), av, pair_law);
                }
            }
        }
        if std::env::var("PTXV_NL_JOINT").is_ok()
            && (std::env::var("PTXV_NL_JOINT").as_deref() == Ok("all")
                || matches!(lr, Law::Cell { .. }) && matches!(lc, Law::Cell { .. }))
            && JDBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 12
        {
            eprintln!(
                "joint pair {p} depth {} q {q:e} dv {:e} laws {:?} / {:?}:",
                self.depth,
                (sc.v[oc as usize] - self.sr.v[or_ as usize]) / q,
                lr,
                lc
            );
            for st in &sites {
                let (m, v) = st.law.moments(self.r.tabs);
                eprintln!(
                    "   k{} c {:.3e} {} m {:.3e} sd {:.3e}",
                    st.kern,
                    st.c,
                    st.class,
                    m,
                    v.sqrt()
                );
            }
        }
        let mut dummy = HashMap::new();
        // The two kernels' real values may differ slightly (f32 constants of the algorithms).
        let dv = (sc.v[oc as usize] - self.sr.v[or_ as usize]) / q;
        let z = self.z_law(sites, dv, &mut dummy);
        let mut kl: BTreeMap<i64, f64> = BTreeMap::new();
        for (x, pr) in z.atoms() {
            let f = x.floor();
            let fr = x - f;
            *kl.entry(f as i64).or_default() += pr * (1.0 - fr);
            if fr > 0.0 {
                *kl.entry(f as i64 + 1).or_default() += pr * fr;
            }
        }
        let mut kv: Vec<(i64, f64)> = kl.into_iter().collect();
        if let Some(pl) = pair_law {
            kv = conv_int(&kv, &pl);
        }
        let out = std::rc::Rc::new((kv, q, var_r));
        if std::env::var("PTXV_NL_JOINT").is_ok()
            && (std::env::var("PTXV_NL_JOINT").as_deref() == Ok("all")
                || matches!(lr, Law::Cell { .. }) && matches!(lc, Law::Cell { .. }))
            && JDBG.load(std::sync::atomic::Ordering::Relaxed) <= 12
        {
            eprintln!("   => K law {:?}", out.0);
        }
        self.memo.insert(p, out.clone());
        Ok(out)
    }

    /// Whether the candidate's op `oc` (a slice) computes, bit for bit, what the reference's
    /// sub-DAG above the nodes holding its argument values computes into `or_` (tested at 256
    /// points around this input). Returns the reference's ops strictly in between.
    fn composite_ident(&self, or_: u32, oc: u32) -> Option<std::collections::HashSet<u32>> {
        use super::super::chain::Op;
        let (ck, sc) = self.c?;
        let Op::Nl(super::ops::NlOp::Seg { prog, args, .. }) = &ck.p.ops[oc as usize] else {
            return None;
        };
        let cone = self.r.cone(or_);
        // Per argument: the reference node nearest the root with its value.
        let mut stops: Vec<u32> = Vec::new();
        for &a in args {
            let v = sc.v[a as usize];
            let j = cone.iter().copied().skip(1).find(|&j| {
                let w = self.sr.v[j as usize];
                w == v || (w - v).abs() <= 2f64.powi(-22) * v.abs()
            })?;
            stops.push(j);
        }
        let (sp, leaves, new) = super::enumerate::sub_prog_cut(self.r.p, or_, &stops);
        let new: Vec<u32> = new.into_iter().collect::<Option<Vec<u32>>>()?;
        if leaves.len() != new.iter().collect::<std::collections::HashSet<_>>().len() {
            return None;
        }
        let mut rng = super::super::special::Rng::new(0xc0f1 ^ or_ as u64);
        let mut w = vec![crate::mma_lp::Operand::from_value(0.0); sp.ops.len()];
        let mut scratch = Vec::new();
        for t in 0..256 {
            let bits: Vec<u32> = args
                .iter()
                .map(|&a| {
                    let v = sc.v[a as usize];
                    let f = if t == 0 {
                        1.0
                    } else {
                        (1.0 + (rng.unit() - 0.5) * 2f64.powi(-6))
                            * 2f64.powi(rng.below(5) as i32 - 2)
                    };
                    ((v * f) as f32).to_bits()
                })
                .collect();
            let g = prog.eval_with(&bits, &mut scratch).ok()?;
            for (k, &l) in new.iter().enumerate() {
                w[l as usize] = crate::mma_lp::Operand::from_value(f32::from_bits(bits[k]) as f64);
            }
            sp.eval_into(&mut w);
            let h = (w[sp.ops.len() - 1].v as f32).to_bits();
            if h != g && !(f32::from_bits(h).is_nan() && f32::from_bits(g).is_nan()) {
                return None;
            }
        }
        // The reference's ops above the stops.
        let mut between = std::collections::HashSet::new();
        let mut st = vec![or_];
        while let Some(i) = st.pop() {
            if stops.contains(&i) || !between.insert(i) {
                continue;
            }
            st.extend(super::lin::op_args(&self.r.p.ops[i as usize]));
        }
        Some(between)
    }

    /// Merge sites by key and build the law of shift + sum c delta (tier 1: dominant discrete
    /// sites exactly, the rest normal; tier 2: normal).
    fn z_law(&self, sites: Vec<Site>, shift: f64, budget: &mut HashMap<String, f64>) -> Grid {
        let mut merged: Vec<Site> = Vec::new();
        let mut idx: HashMap<u64, usize> = HashMap::new();
        for s in sites {
            match idx.get(&s.key) {
                Some(&j) => merged[j].c += s.c,
                None => {
                    idx.insert(s.key, merged.len());
                    merged.push(s);
                }
            }
        }
        merged.retain(|s| s.c != 0.0);
        let tabs = self.r.tabs;
        let mut mom: Vec<(f64, f64)> = merged
            .iter()
            .map(|s| {
                let (m, v) = s.law.moments(tabs);
                (s.c * m, s.c * s.c * v)
            })
            .collect();
        for (s, (_, v)) in merged.iter().zip(&mom) {
            *budget.entry(s.class.clone()).or_default() += v;
        }
        let tot_v: f64 = mom.iter().map(|x| x.1).sum();
        let tot_m: f64 = mom.iter().map(|x| x.0).sum();
        let sd = tot_v.sqrt();
        let h = if sd > 4.0 { sd / 16.0 } else { 1.0 / 64.0 };
        let mut d = Grid::point(shift, h);
        if self.cfg.tier == 1 && !merged.is_empty() {
            let mut order: Vec<usize> = (0..merged.len()).collect();
            order.sort_by(|a, b| mom[*b].1.total_cmp(&mom[*a].1));
            let mut taken = 0;
            for &j in &order {
                if taken >= self.cfg.exact_sites || mom[j].1.sqrt() < h / 2.0 {
                    break;
                }
                if let Some(at) = merged[j].law.atoms(tabs, 4096) {
                    let sc: Vec<(f64, f64)> =
                        at.iter().map(|(x, p)| (merged[j].c * x, *p)).collect();
                    d = d.conv_atoms(&sc);
                    mom[j] = (0.0, 0.0);
                    taken += 1;
                }
            }
            let (rm, rv) = mom.iter().fold((0.0, 0.0), |a, x| (a.0 + x.0, a.1 + x.1));
            d.conv_normal(rm, rv)
        } else {
            d.conv_normal(tot_m, tot_v)
        }
    }

    /// One output's law (output op of the reference, and of the candidate).
    fn out_law(
        &mut self,
        o_r: u32,
        o_c: Option<u32>,
        fmt: Format,
        budget: &mut HashMap<String, f64>,
    ) -> Result<OutLaw, &'static str> {
        let r = self.r;
        let sr = self.sr;
        let y = sr.v[o_r as usize];
        if !y.is_finite() {
            return Err("non-finite value");
        }
        let ulp = ulp_out(y, fmt);
        let (fr_i, fr) = r.final_of(sr, o_r);
        let single = o_c.is_none();
        let sign_r = if single { 1.0 } else { -1.0 };
        let (fc_i, fc) = match (self.c, o_c) {
            (Some((ck, sc)), Some(o)) => {
                let x = ck.final_of(sc, o);
                (Some(x.0), Some(x.1))
            }
            _ => (None, None),
        };
        let mut sites = self.collect(fr_i, fc_i, ulp, sign_r)?;
        // A final slice ending in a rounding: its pre-rounding error is a site.
        let push_pre = |sites: &mut Vec<Site>, k: &KModel, i: u32, f: Final, sign: f64| {
            if let Final::SlicePre { tab, cell, ulp: u } = f {
                sites.push(Site {
                    key: k.canon[i as usize],
                    c: sign / ulp,
                    law: Law::CellPre { tab, cell, ulp: u },
                    class: format!("{} (before its last rounding)", class(k, i)),
                    kern: (sign > 0.0) as u8,
                    op: i,
                });
            }
        };
        push_pre(&mut sites, r, fr_i, fr, sign_r);
        if let (Some((ck, _)), Some(i), Some(f)) = (self.c, fc_i, fc) {
            push_pre(&mut sites, ck, i, f, 1.0);
        }
        // Per-kernel moments against exact (ulp units) and their covariance.
        let mut pm = (0.0, 0.0, 0.0, 0.0, 0.0);
        {
            let mut by: HashMap<u64, (f64, f64, f64, f64)> = HashMap::new();
            for s in &sites {
                let (m, v) = s.law.moments(r.tabs);
                let e = by.entry(s.key).or_insert((0.0, 0.0, m, v));
                if s.kern == 0 || single {
                    e.0 += s.c * sign_r;
                } else {
                    e.1 += s.c;
                }
            }
            for (_, (a, b, m, v)) in by {
                pm.0 += a * m;
                pm.1 += a * a * v;
                pm.2 += b * m;
                pm.3 += b * b * v;
                pm.4 += a * b * v;
            }
        }
        let (sh_r, cell_r) = final_shift(fr);
        let mut off: Vec<(i64, f64)> = slice_off(r, fr)
            .into_iter()
            .map(|(k, p)| (if single { k } else { -k }, p))
            .collect();
        let (shift, cell, identical) = match (self.c, fc_i, fc) {
            (Some((ck, _)), Some(fci), Some(fc)) => {
                let (sh_c, _) = final_shift(fc);
                off = conv_int(&off, &slice_off(ck, fc));
                (
                    sh_c - sh_r,
                    (0.0, 0.0),
                    ck.canon[fci as usize] == r.canon[fr_i as usize],
                )
            }
            _ => (sh_r - 0.5, cell_r, false),
        };
        let d = self.z_law(sites, shift, budget);
        let rlat = sr.rlat[o_r as usize];
        let r_res = match (rlat, fc) {
            (Some(l), None) if l != i32::MAX => {
                let e = super::super::spec::exp_floor(ulp);
                if l >= e {
                    1
                } else if e - l <= 6 {
                    1u64 << (e - l)
                } else {
                    0
                }
            }
            _ => 0,
        };
        let tie = if single
            && r_res == 0
            && matches!(fr, Final::Rn)
            && matches!(fmt, Format::Bf16 | Format::F16)
        {
            self.tie
        } else {
            0.0
        };
        Ok(OutLaw {
            tie,
            d,
            shift,
            off,
            r: r_res,
            cell,
            ulp,
            s: y.abs().max(1e-300),
            identical,
            pm,
        })
    }
}

/// K's law (integer offsets) of an output.
fn k_law(ol: &OutLaw) -> Vec<(i64, f64)> {
    if ol.identical {
        return vec![(0, 1.0)];
    }
    let mut m: BTreeMap<i64, f64> = BTreeMap::new();
    for (x, p) in ol.d.atoms() {
        if ol.r > 0 {
            // y on a lattice of r residues per ulp: u = j / r.
            for j in 0..ol.r {
                let u = j as f64 / ol.r as f64;
                let k = (u + 0.5 + x).floor() - (u + 0.5).floor();
                *m.entry(k as i64).or_default() += p / ol.r as f64;
            }
        } else if ol.d.h > 0.5 && ol.d.p.len() > 1 {
            // A grid coarser than the ulp: each atom stands for its cell [x - h/2, x + h/2)
            // (as a point it would put its whole mass on one K: a constant P(K = 0) =
            // P(atom at 0) however wide the law). The tent kernel against a uniform cell:
            // P(K = k) is the cell's overlap with [k - 1/2, k + 1/2) (the position shift
            // averages out across the cell).
            // Integers beyond +-W ulps: their mass at its centroid (the PMF keeps |k| <= W
            // exactly, the moments the rest).
            const W: f64 = 32.0;
            let (a, b) = (x - ol.d.h / 2.0, x + ol.d.h / 2.0);
            for (lo, hi) in [(a, b.min(-W - 0.5)), (a.max(W + 0.5), b)] {
                if hi > lo {
                    let c = (0.5 * (lo + hi)).round() as i64;
                    *m.entry(c).or_default() += p * (hi - lo) / ol.d.h;
                }
            }
            let (a, b) = (a.max(-W - 0.5), b.min(W + 0.5));
            if b > a {
                let (k0, k1) = ((a + 0.5).floor() as i64, (b + 0.5).floor() as i64);
                for k in k0..=k1 {
                    let lo = a.max(k as f64 - 0.5);
                    let hi = b.min(k as f64 + 0.5);
                    if hi > lo {
                        *m.entry(k).or_default() += p * (hi - lo) / ol.d.h;
                    }
                }
            }
        } else {
            let f = x.floor();
            let fr = x - f;
            let pc = p * (1.0 - ol.tie);
            *m.entry(f as i64).or_default() += pc * (1.0 - fr);
            if fr > 0.0 {
                *m.entry(f as i64 + 1).or_default() += pc * fr;
            }
            if ol.tie > 0.0 {
                // The computed value at a tie: the even neighbour, either side with
                // probability 1/2; the exact value is |x| ulps away from it.
                let pt = p * ol.tie;
                let t = x.trunc();
                if x == 0.0 {
                    // |d| below the grid step, its sign either way (a continuous deviation).
                    *m.entry(0).or_default() += pt / 2.0;
                    *m.entry(1).or_default() += pt / 4.0;
                    *m.entry(-1).or_default() += pt / 4.0;
                } else {
                    *m.entry(t as i64).or_default() += pt / 2.0;
                    *m.entry((t + x.signum()) as i64).or_default() += pt / 2.0;
                }
            }
        }
    }
    conv_int(&m.into_iter().collect::<Vec<_>>(), &ol.off)
}

/// The analytic estimate of a single kernel (c None) or a pair.
#[allow(clippy::too_many_arguments)]
pub fn estimate(
    cfg: &Cfg,
    r: &KModel,
    c: Option<&KModel>,
    spec: &Spec,
    layouts: &[Layout],
    fmt: Format,
    same: &[(u32, u32)],
    ptabs: &[(u32, u32, Table)],
) -> Result<(Estimate, Vec<String>), String> {
    let t0 = std::time::Instant::now();
    let n_out = r.p.outputs.len();
    let pick: Vec<usize> = if n_out <= cfg.outputs {
        (0..n_out).collect()
    } else {
        (0..cfg.outputs).map(|i| i * n_out / cfg.outputs).collect()
    };
    let pick: Vec<usize> = pick
        .into_iter()
        .filter(|&o| r.p.outputs[o].is_some() && c.is_none_or(|c| c.p.outputs[o].is_some()))
        .collect();
    if pick.is_empty() {
        return Err("no tracked outputs".into());
    }
    // Matched roundings: ops of the two kernels with the same real value at two inputs.
    let mut matches: HashMap<u64, u64> = HashMap::new();
    if let Some(ck) = c {
        let st = |t: u64| -> Result<(State, State), String> {
            let smp = super::super::spec::sample(spec, layouts, (1u64 << 42) + t)?;
            Ok((
                r.state(&r.p.leaves(&smp.clean_bytes())?),
                ck.state(&ck.p.leaves(&smp.clean_bytes())?),
            ))
        };
        let (r0, c0) = st(0)?;
        let (r1, c1) = st(1)?;
        let rounding =
            |l: &Law| matches!(l, Law::Rn { .. } | Law::Cell { .. } | Law::CellPre { .. });
        if std::env::var("PTXV_NL_MDBG").is_ok() {
            for (nm, k, s0) in [("r", r, &r0), ("c", ck, &c0)] {
                for (i, op) in k.p.ops.iter().enumerate() {
                    if matches!(
                        op,
                        super::super::chain::Op::Nl(super::ops::NlOp::Seg { .. })
                    ) {
                        eprintln!(
                            "{nm} seg op{i} v {:e} law {:?}",
                            s0.v[i],
                            format!("{:?}", s0.law[i])
                                .chars()
                                .take(50)
                                .collect::<String>()
                        );
                    }
                }
            }
        }
        // Values within 2^-20 relative (algorithms differing in f32 constants, e.g.
        // Welford's 1/n factors, compute slightly different real values): bucket by the top
        // 20 mantissa bits and probe the neighbouring buckets.
        let key = |v: f64| (v.to_bits() >> 32, v.is_sign_negative());
        let mut by: HashMap<(u64, bool), Vec<u32>> = HashMap::new();
        for i in 0..ck.p.ops.len() {
            let v = c0.v[i];
            if rounding(&c0.law[i]) && v.is_finite() && v != 0.0 && v.abs() != 1.0 {
                by.entry(key(v)).or_default().push(i as u32);
            }
        }
        for i in 0..r.p.ops.len() {
            let v = r0.v[i];
            if !rounding(&r0.law[i]) || !v.is_finite() || v == 0.0 || v.abs() == 1.0 {
                continue;
            }
            let (kb, ks) = key(v);
            let mut cands: Vec<u32> = Vec::new();
            for d in [0u64, 1, u64::MAX] {
                if let Some(x) = by.get(&(kb.wrapping_add(d), ks)) {
                    cands.extend(x);
                }
            }
            {
                for &k in &cands {
                    let (a, b) = (r1.v[i], c1.v[k as usize]);
                    if (a - b).abs() <= 2f64.powi(-20) * a.abs() && rounding(&c1.law[k as usize]) {
                        matches.insert(r.canon[i], ck.canon[k as usize]);
                        break;
                    }
                }
            }
        }
    }
    let mut m = Matches {
        pairs: Vec::new(),
        by_r: HashMap::new(),
        by_c: HashMap::new(),
    };
    for (kr, kc) in &matches {
        if m.by_r.contains_key(kr) || m.by_c.contains_key(kc) {
            continue;
        }
        m.by_r.insert(*kr, m.pairs.len());
        m.by_c.insert(*kc, m.pairs.len());
        m.pairs.push((*kr, *kc));
    }
    let op_map = |k: &KModel| -> HashMap<u64, u32> {
        let mut h = HashMap::new();
        for (i, key) in k.canon.iter().enumerate() {
            h.entry(*key).or_insert(i as u32);
        }
        h
    };
    let op_r = op_map(r);
    let op_c = c.map(op_map).unwrap_or_default();
    let mut jdiag: BTreeMap<String, [f64; 4]> = BTreeMap::new();
    let mut adj = Vec::new();
    let mut kpmf: BTreeMap<i64, f64> = BTreeMap::new();
    let (mut ulp_m, mut abs_m, mut u_m) = (Mix::default(), Mix::default(), Mix::default());
    let mut per_input: Vec<f64> = Vec::new();
    let mut budget: HashMap<String, f64> = HashMap::new();
    let mut skipped: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut nonlinear = 0u64;
    // Input errors: second-order effect of the input error against its first-order effect
    // (N-ERRDIST-INERR), summed over the probed inputs: (|d+ + d-|^2, |d+ - d-|^2).
    let mut curv = (0.0f64, 0.0f64);
    let mut pmacc = [0.0f64; 6];
    let mut count = 0u64;
    // A single kernel's 16-bit outputs: how often the computed f32 value is a tie of the
    // output grid (RNE then picks the even neighbour). Its low bits need not be uniform (a
    // product with a 16-bit factor), so the rate is measured on bit-exact evaluations.
    let tie = if c.is_none() && matches!(fmt, Format::Bf16 | Format::F16) {
        let k = 23 - fmt.mbits();
        let n_in = (2e7 / r.p.ops.len().max(1) as f64).clamp(16.0, 4000.0) as u64;
        let (mut n, mut ties) = (0u64, 0u64);
        for t in 0..n_in {
            let smp = super::super::spec::sample(spec, layouts, (1u64 << 41) + t)?;
            let mut w = r.p.leaves(&smp.bytes)?;
            r.p.eval_into(&mut w);
            for o in r.p.outputs.iter().flatten() {
                if let super::super::chain::Op::Round { arg, .. } = &r.p.ops[*o as usize] {
                    let b = (w[*arg as usize].v as f32).to_bits();
                    n += 1;
                    ties += (b & ((1u32 << k) - 1) == 1u32 << (k - 1)) as u64;
                }
            }
        }
        // One pseudo-observation at the uniform-low-bits rate.
        let prior = 2f64.powi(-k);
        (ties as f64 + 1.0) / (n as f64 + 1.0 / prior)
    } else {
        0.0
    };
    // At least cfg.inputs nominal inputs; more while they fit the time budget (rare events,
    // such as 16-bit misrounding near zero outputs, need many configurations).
    let t_loop = std::time::Instant::now();
    let budget_s: f64 = std::env::var("PTXV_NL_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3.0);
    // Input errors: the estimate is cross-checked on the error's spread (mean |error|
    // within 10%), and 48 inputs leave it +-15% from seed to seed (a softmax fed by an
    // upstream error); 192 inputs at least.
    let (min_inputs, budget_s) = if spec.has_input_errors() {
        (cfg.inputs.max(192), 4.0 * budget_s)
    } else {
        (cfg.inputs, budget_s)
    };
    for t in 0..MAX_INPUTS {
        if t >= min_inputs as u64 {
            let el = t_loop.elapsed().as_secs_f64();
            if el * (t + 1) as f64 / t as f64 > budget_s {
                break;
            }
        }
        let smp = super::super::spec::sample(spec, layouts, (1u64 << 42) + t)?;
        // The nominal (error-free) values; input errors are the leaves' local laws.
        let wr = r.p.leaves(&smp.clean_bytes())?;
        let sr = r.state(&wr);
        let sreal =
            (r.traj == super::lin::Trajectory::Both && c.is_none()).then(|| r.state_real(&wr));
        if !r.in_err.is_empty() && t < 16 {
            let (a, b) = curvature(r, &sr, &wr, &pick, t);
            curv.0 += a;
            curv.1 += b;
        }
        let sc = match c {
            Some(c) => Some(c.state(&c.p.leaves(&smp.clean_bytes())?)),
            None => None,
        };
        let (mut pd, mut nd) = (0.0, 0u64);
        let mut ctx = Ctx {
            cfg,
            r,
            sr: &sr,
            both: sreal.as_ref().map(|s| (wr.as_slice(), s)),
            c: match (c, &sc) {
                (Some(ck), Some(s)) => Some((ck, s)),
                _ => None,
            },
            m: &m,
            same,
            ptabs,
            op_r: &op_r,
            op_c: &op_c,
            memo: HashMap::new(),
            cones_r: HashMap::new(),
            cones_c: HashMap::new(),
            adj: std::mem::take(&mut adj),
            depth: 0,
            tie,
        };
        for &o in &pick {
            let o_r = r.p.outputs[o].unwrap();
            let o_c = c.map(|ck| ck.p.outputs[o].unwrap());
            let res = ctx.out_law(o_r, o_c, fmt, &mut budget);
            let ol = match res {
                Ok(x) => x,
                Err(e) => {
                    *skipped.entry(e).or_default() += 1;
                    continue;
                }
            };
            // Applicability of the first-order model: perturbation << the value.
            let sd: f64 = ol.d.atoms().map(|(x, p)| p * x * x).sum::<f64>().sqrt();
            if sd * ol.ulp > 2f64.powi(-8) * ol.s {
                nonlinear += 1;
            }
            let k = k_law(&ol);
            let p0: f64 = k.iter().filter(|x| x.0 == 0).map(|x| x.1).sum();
            pd += 1.0 - p0;
            nd += 1;
            for (kk, p) in &k {
                *kpmf.entry(*kk).or_default() += p;
            }
            let ulp = ol.ulp;
            let us = ulp / (U * ol.s);
            if c.is_some() {
                let at: Vec<(f64, f64)> = k.iter().map(|(x, p)| (*x as f64, *p)).collect();
                ulp_m.add_law(&at, 0.0, 0.0, 1.0);
                abs_m.add_law(&at, 0.0, 0.0, ulp);
                u_m.add_law(&at, 0.0, 0.0, us);
            } else {
                // e / ulp = d + V, V uniform on the final rounding's cell.
                let mut at: Vec<(f64, f64)> =
                    ol.d.atoms().map(|(x, p)| (x - ol.shift, p)).collect();
                if ol.off.len() > 1 || ol.off.first().is_some_and(|x| x.0 != 0) {
                    let o: Vec<(f64, f64)> = ol.off.iter().map(|(x, p)| (*x as f64, *p)).collect();
                    let mut m = Vec::new();
                    for &(x, p) in &at {
                        for &(y, q) in &o {
                            m.push((x + y, p * q));
                        }
                    }
                    at = m;
                }
                let (a, b) = if ol.r == 1 && at.iter().all(|x| x.0 == 0.0) {
                    (0.0, 0.0)
                } else {
                    ol.cell
                };
                ulp_m.add_law(&at, a, b, 1.0);
                abs_m.add_law(&at, a, b, ulp);
                u_m.add_law(&at, a, b, us);
            }
            pmacc[0] += ol.pm.0 * ulp;
            pmacc[1] += ol.pm.1 * ulp * ulp + (ol.pm.0 * ulp).powi(2);
            pmacc[2] += ol.pm.2 * ulp;
            pmacc[3] += ol.pm.3 * ulp * ulp + (ol.pm.2 * ulp).powi(2);
            pmacc[4] += ol.pm.4 * ulp * ulp + ol.pm.0 * ol.pm.2 * ulp * ulp;
            count += 1;
        }
        if std::env::var("PTXV_NL_JDIAG").is_ok() && cfg.tier == 1 {
            if let Some(ck) = c {
                let mut wr2 = r.p.leaves(&smp.bytes)?;
                r.p.eval_into(&mut wr2);
                let mut wc2 = ck.p.leaves(&smp.bytes)?;
                ck.p.eval_into(&mut wc2);
                let agg = &mut jdiag;
                for p in 0..m.pairs.len() {
                    let (kr, kc) = m.pairs[p];
                    let (or_, oc) = (op_r[&kr], op_c[&kc]);
                    let Ok(jl) = ctx.joint(p) else { continue };
                    let q = jl.1;
                    let act = (wc2[oc as usize].v - wr2[or_ as usize].v) / q;
                    let pred: f64 = jl.0.iter().filter(|x| x.0 != 0).map(|x| x.1).sum();
                    if let (
                        super::super::chain::Op::Nl(super::ops::NlOp::Seg {
                            args: ar,
                            prog: pr,
                            ..
                        }),
                        super::super::chain::Op::Nl(super::ops::NlOp::Seg {
                            args: ac,
                            prog: pc,
                            ..
                        }),
                    ) = (&r.p.ops[or_ as usize], &ck.p.ops[oc as usize])
                    {
                        if ar.len() == 1
                            && ac.len() == 1
                            && JDBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 10
                            && std::env::var("PTXV_NL_JDIAG2").is_ok()
                        {
                            let (xa, xc) =
                                (wr2[ar[0] as usize].v as f32, wc2[ac[0] as usize].v as f32);
                            let g1 = pr.eval(&[xa.to_bits()]).unwrap();
                            let g2 = pc.eval(&[xa.to_bits()]).unwrap();
                            let (lr, lc) = (
                                &sr.law[or_ as usize],
                                &sc.as_ref().unwrap().law[oc as usize],
                            );
                            eprintln!(
                                "args {:e} {:e} out {:e} {:e} | at R's arg: {:e} {:e} laws {:?} {:?} jl {:?}",
                                xa,
                                xc,
                                wr2[or_ as usize].v,
                                wc2[oc as usize].v,
                                f32::from_bits(g1),
                                f32::from_bits(g2),
                                lr,
                                lc,
                                jl.0
                            );
                        }
                    }
                    let e = agg
                        .entry(format!("{} / {}", class(r, or_), class(ck, oc)))
                        .or_default();
                    e[0] += 1.0;
                    e[1] += (act != 0.0) as u8 as f64;
                    e[2] += pred;
                    e[3] += act.abs();
                }
            }
        }
        adj = std::mem::take(&mut ctx.adj);
        if nd > 0 {
            per_input.push(pd / nd as f64);
        }
    }
    for (k, e) in &jdiag {
        eprintln!(
            "matched {k}: n {} observed P(!=) {:.3} predicted {:.3} obs E|diff| {:.3}",
            e[0],
            e[1] / e[0],
            e[2] / e[0],
            e[3] / e[0]
        );
    }
    if count == 0 {
        return Err(format!("no output could be analyzed ({skipped:?})"));
    }
    let tot_skip: u64 = skipped.values().sum();
    let total = count + tot_skip;
    if tot_skip as f64 > 0.01 * total as f64 {
        return Err(format!(
            "does not apply: {tot_skip} of {total} output-inputs not modeled ({skipped:?})"
        ));
    }
    let n = per_input.len() as f64;
    let pmean = per_input.iter().sum::<f64>() / n;
    let pvar = per_input.iter().map(|x| (x - pmean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let hw = 1.96 * (pvar / n).sqrt();
    let ktot: f64 = kpmf.values().sum();
    let pmf: Vec<(i64, f64)> = kpmf.iter().map(|(k, p)| (*k, p / ktot)).collect();
    let mut notes = Vec::new();
    let btot: f64 = budget.values().sum();
    let mut b: Vec<(String, f64)> = budget.into_iter().collect();
    b.sort_by(|x, y| y.1.total_cmp(&x.1));
    let bs: Vec<String> = b
        .iter()
        .take(8)
        .map(|(k, v)| format!("{k} {:.1}%", 100.0 * v / btot.max(1e-300)))
        .collect();
    notes.push(format!(
        "variance budget of the pre-rounding deviation by site: {}",
        bs.join(", ")
    ));
    if !matches.is_empty() {
        notes.push(format!(
            "{} ops of the two kernels round the same real value (joint rounding model)",
            matches.len()
        ));
    }
    if nonlinear > 0 {
        notes.push(format!(
            "{nonlinear} of {count} output-inputs have a deviation above 2^-8 of |y| (first-order model strained)"
        ));
    }
    if tot_skip > 0 {
        notes.push(format!("{tot_skip} output-inputs not modeled: {skipped:?}"));
    }
    let cnt = count as f64;
    let pair = c.map(|_| {
        let (mr, mc) = (pmacc[0] / cnt, pmacc[2] / cnt);
        let (vr, vc) = (pmacc[1] / cnt - mr * mr, pmacc[3] / cnt - mc * mc);
        let cov = pmacc[4] / cnt - mr * mc;
        PairMoments {
            mean_r: mr,
            var_r: vr,
            mean_c: mc,
            var_c: vc,
            cov_rc: cov,
            corr_rc: cov / (vr * vc).sqrt().max(1e-300),
        }
    });
    let method = match cfg.tier {
        1 => {
            "linearized error propagation through the DAG; dominant site laws convolved exactly (residue-aware roundings, bit-exact slice / MUFU tables), the rest normal; conditional on nominal inputs"
        }
        _ => {
            "linearized error propagation through the DAG; normal deviation with exact site moments; conditional on nominal inputs"
        }
    };
    // Linearization of the input errors: the second-order term (d+ + d-) / 2 of a symmetric
    // probe against the first-order term (d+ - d-) / 2, in rms over outputs and inputs.
    let (applicable, applicability) = if curv.1 > 0.0 {
        let ratio = (curv.0 / curv.1).sqrt();
        let ok = ratio <= 0.1;
        (
            Some(ok),
            vec![format!(
                "input errors: second-order / first-order output effect {ratio:.3} (needs <= 0.1: the downstream curvature is small against the upstream error) {}",
                if ok { "ok" } else { "FAILS" }
            )],
        )
    } else {
        (None, vec![])
    };
    let est = Estimate {
        tier: cfg.tier,
        method: method.into(),
        n: per_input.len() as u64,
        outputs: n_out as u64,
        abs: abs_m.moments(),
        u_scale: Some(u_m.moments()),
        ulp: Some(ulp_m.moments()),
        pmf_ulp: pmf,
        hist_u: None,
        error_bar: ErrorBar {
            kind: "clt".into(),
            cdf_sup: None,
            mean_hw: None,
            std_rel_hw: None,
            model_mass: None,
            notes: vec![
                format!(
                    "P(differ) averaged over {} nominal inputs x {} outputs; its interval is the spread over inputs (95%), not the model error",
                    per_input.len(),
                    pick.len()
                ),
                "first-order in the local errors (N-ERRDIST-NL-LIN); site errors independent given the input except shared sub-computations (N-ERRDIST-NL-INDEP)".into(),
            ],
        },
        pair,
        p_differ: Some((pmean, (pmean - hw).max(0.0), (pmean + hw).min(1.0))),
        seconds: t0.elapsed().as_secs_f64(),
        applicable,
        applicability,
    };
    Ok((est, notes))
}

/// One symmetric probe of the input errors at a nominal input: each leaf with an error law
/// moved by +e and by -e (e drawn from its law), the outputs in real arithmetic: the sums
/// over the picked outputs of (d+ + d-)^2 and (d+ - d-)^2, d+- = y(x +- e) - y(x).
fn curvature(
    k: &KModel,
    s: &State,
    w: &[crate::mma_lp::Operand],
    pick: &[usize],
    t: u64,
) -> (f64, f64) {
    let mut rng = super::super::special::Rng::stream(0x1e57, t, "curvature");
    let mut e = vec![0.0f64; w.len()];
    for (i, op) in k.p.ops.iter().enumerate() {
        if !matches!(op, super::super::chain::Op::Leaf { .. }) {
            continue;
        }
        e[i] = match &s.law[i] {
            Law::Normal { mean, var } => mean + var.max(0.0).sqrt() * rng.normal(),
            Law::Rn { q, .. } => (rng.unit() - 0.5) * q,
            _ => 0.0,
        };
    }
    let at = |sg: f64| {
        let mut w2 = w.to_vec();
        for (i, d) in e.iter().enumerate() {
            if *d != 0.0 {
                w2[i].v += sg * d;
            }
        }
        super::real::eval(k.p, &w2)
    };
    let (y0, yp, ym) = (super::real::eval(k.p, w), at(1.0), at(-1.0));
    let (mut a, mut b) = (0.0, 0.0);
    for &o in pick {
        let Some(op) = k.p.outputs[o] else { continue };
        let i = op as usize;
        let (z, p, m) = (y0[i].0.v(), yp[i].0.v(), ym[i].0.v());
        if !(z.is_finite() && p.is_finite() && m.is_finite()) {
            continue;
        }
        let (dp, dm) = (p - z, m - z);
        a += (dp + dm).powi(2);
        b += (dp - dm).powi(2);
    }
    (a, b)
}

/// Most nominal inputs of the linearized tiers.
const MAX_INPUTS: u64 = 2000;

/// Recursion limit of the joint law through matched roundings (sequential reductions nest
/// one level per term).
fn max_depth() -> usize {
    std::env::var("PTXV_NL_DEPTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16)
}

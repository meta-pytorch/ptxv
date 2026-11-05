/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Exact laws of input elements and tensor-core terms under a spec: PMFs over codes (finite
//! support), the joint (scale, code) law of the quantization recipes as signed mixtures of
//! i.i.d. laws, and "clouds" (merged atom sets) for the coarse partial-sum laws.

use std::collections::BTreeMap;
use std::collections::HashMap;

use super::spec::Dist;
use super::spec::Format;
use super::spec::Quant;
use super::spec::ScaleRule;
use super::spec::TensorSpec;
use super::spec::exp_floor;

/// Atoms below this probability are dropped from element PMFs; their total is reported as
/// model mass (`Pmf::pruned`).
pub const PRUNE: f64 = 1e-16;

/// A PMF over the exact values of one element: value, probability, alignment exponent (the
/// format's exponent field, subnormals at the minimum; floor(log2 |v|) for f32 / bf16 / f16).
#[derive(Clone, Debug, Default)]
pub struct Pmf {
    pub v: Vec<f64>,
    pub p: Vec<f64>,
    pub ef: Vec<i32>,
    /// Mass dropped by pruning or outside the format (overflow handled as saturation).
    pub pruned: f64,
}

impl Pmf {
    pub fn total(&self) -> f64 {
        self.p.iter().sum()
    }
    pub fn normalize(&mut self) {
        let t = self.total();
        if t > 0.0 {
            for p in self.p.iter_mut() {
                *p /= t;
            }
        }
    }
    pub fn len(&self) -> usize {
        self.v.len()
    }
    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
    /// Mean and second moment.
    pub fn moments(&self) -> (f64, f64) {
        let t = self.total().max(1e-300);
        let m = self.v.iter().zip(&self.p).map(|(v, p)| v * p).sum::<f64>() / t;
        let m2 = self
            .v
            .iter()
            .zip(&self.p)
            .map(|(v, p)| v * v * p)
            .sum::<f64>()
            / t;
        (m, m2)
    }
    /// The atoms with |v| < hi (or <= with `incl`), unnormalized (their mass is P(|X| < hi)).
    pub fn below(&self, hi: f64) -> Pmf {
        let mut o = Pmf::default();
        for i in 0..self.v.len() {
            if self.v[i].abs() < hi {
                o.v.push(self.v[i]);
                o.p.push(self.p[i]);
                o.ef.push(self.ef[i]);
            }
        }
        o
    }
    /// The atoms with alignment exponent <= e (zeros always), unnormalized.
    pub fn align_le(&self, e: i32) -> Pmf {
        let mut o = Pmf::default();
        for i in 0..self.v.len() {
            if self.v[i] == 0.0 || self.ef[i] <= e {
                o.v.push(self.v[i]);
                o.p.push(self.p[i]);
                o.ef.push(self.ef[i]);
            }
        }
        o
    }
    /// P(alignment exponent <= e) (zeros count as below everything).
    pub fn cdf_align(&self, e: i32) -> f64 {
        self.v
            .iter()
            .zip(&self.p)
            .zip(&self.ef)
            .filter(|((v, _), ef)| **v == 0.0 || **ef <= e)
            .map(|((_, p), _)| p)
            .sum()
    }
}

/// Alignment exponent of a value of `fmt` (narrow formats: the exponent field).
pub fn align_exp(v: f64, fmt: Format) -> i32 {
    if v == 0.0 {
        return i32::MIN / 4;
    }
    match fmt {
        Format::F32 | Format::Bf16 | Format::F16 => exp_floor(v),
        _ => exp_floor(v).max(fmt.emin()),
    }
}

/// The code PMF of RNE(X / div) in `fmt` (saturating), X ~ `d`.
pub fn code_pmf(d: &Dist, fmt: Format, div: f64) -> Pmf {
    let mut acc: BTreeMap<u64, f64> = BTreeMap::new();
    let mut pruned = 0.0;
    add_dist(d, fmt, div, 1.0, &mut acc, &mut pruned);
    let mut o = Pmf {
        pruned,
        ..Default::default()
    };
    for (bits, p) in acc {
        if p < PRUNE {
            o.pruned += p;
            continue;
        }
        let v = f64::from_bits(bits);
        o.v.push(v);
        o.p.push(p);
        o.ef.push(align_exp(v, fmt));
    }
    o
}

fn add_dist(
    d: &Dist,
    fmt: Format,
    div: f64,
    w: f64,
    acc: &mut BTreeMap<u64, f64>,
    pruned: &mut f64,
) {
    if let Dist::Mixture { components } = d {
        let tot: f64 = components.iter().map(|c| c.weight).sum();
        for c in components {
            add_dist(&c.dist, fmt, div, w * c.weight / tot, acc, pruned);
        }
        return;
    }
    if let Some(atoms) = d.atoms() {
        for (x, p) in atoms {
            let (q, _) = fmt.round(x / div, true);
            *acc.entry(norm0(q).to_bits()).or_default() += w * p;
        }
        return;
    }
    // Continuous: CDF differences over the rounding intervals of every code.
    let mut grid: Vec<f64> = (0..(1u32 << fmt.width()))
        .map(|c| fmt.value(c))
        .filter(|v| v.is_finite() && *v >= 0.0)
        .collect();
    grid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    grid.dedup();
    // Positive half (and the negative half if signed), boundaries at midpoints; ties go to the
    // even code, which only matters for atoms (handled above).
    let n = grid.len();
    let put = |acc: &mut BTreeMap<u64, f64>, v: f64, lo: f64, hi: f64| {
        let m = d.mass(lo * div, false, hi * div, true);
        *acc.entry(norm0(v).to_bits()).or_default() += w * m;
    };
    for i in 0..n {
        let v = grid[i];
        let lo = if i == 0 { 0.0 } else { 0.5 * (grid[i - 1] + v) };
        let hi = if i + 1 == n {
            f64::INFINITY
        } else {
            0.5 * (v + grid[i + 1])
        };
        if fmt.signed() {
            if v != 0.0 {
                put(acc, v, lo, hi);
                put(acc, -v, -hi, -lo);
            } else {
                // The zero code takes (-h, h), h half the smallest positive value.
                let h = if n > 1 { 0.5 * grid[1] } else { f64::INFINITY };
                put(acc, 0.0, -h, h);
            }
        } else {
            // Unsigned formats read |x|.
            put(acc, v, lo, hi);
            put(acc, v, -hi, -lo);
        }
    }
    let _ = pruned;
}

fn norm0(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x }
}

// ----------------------------------------------------------------------------- tensor laws

/// The law of one tensor's elements given a block context.
#[derive(Clone, Debug)]
pub enum TensorLaw {
    /// i.i.d. elements with this PMF (values as the kernel reads them).
    Iid(Pmf),
    /// A block-scaled recipe: per block, a scale (value, exponent field) with probability,
    /// and given it the block's elements as a signed mixture of i.i.d. laws:
    /// law(block | scale) = sum_i w_i law_i^(x n) (the block-max coupling; weights may be
    /// negative and sum to 1).
    Blocked {
        block: usize,
        scales: Vec<BlockScale>,
    },
}

#[derive(Clone, Debug)]
pub struct BlockScale {
    pub value: f64,
    pub ef: i32,
    pub prob: f64,
    pub parts: Vec<(f64, Pmf)>,
}

/// Quadrature of a continuous scale distribution: `k` quantile midpoints (equal weights);
/// discrete scale laws exactly.
pub fn scale_nodes(d: &Dist, k: usize) -> Vec<(f64, f64)> {
    if let Some(a) = d.atoms() {
        return a;
    }
    // Bisection on the CDF for each midpoint quantile.
    let mut out = Vec::new();
    let (mut lo0, mut hi0) = (-1.0, 1.0);
    while d.cdf(lo0, true) > 1e-12 {
        lo0 *= 2.0;
    }
    while d.cdf(hi0, true) < 1.0 - 1e-12 {
        hi0 *= 2.0;
    }
    for i in 0..k {
        let q = (i as f64 + 0.5) / k as f64;
        let (mut lo, mut hi) = (lo0, hi0);
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if d.cdf(mid, true) < q {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        out.push((0.5 * (lo + hi), 1.0 / k as f64));
    }
    out
}

/// The element law of a tensor (the format it is read as: `fmt`) under its spec, averaged
/// over hierarchical scales (quadrature nodes, `notes` says when approximate).
// ASSUME(N-ERRDIST-IID): elements independent given their scales; block scales from the block.
/// The element law with hierarchical scales and outliers folded in (a scale mixture).
pub fn effective_dist(t: &TensorSpec, notes: &mut Vec<String>) -> Dist {
    let mut d = t.dist.clone();
    let mut mix: Vec<(f64, f64)> = vec![(1.0, 1.0)];
    for s in &t.scales {
        let nodes = scale_nodes(&s.dist, 32);
        if s.dist.atoms().is_none() {
            notes.push(
                "hierarchical scale integrated with 32 quantile nodes (analytic tiers)".into(),
            );
        }
        let mut m2 = Vec::new();
        for (a, pa) in &mix {
            for (b, pb) in &nodes {
                m2.push((a * b, pa * pb));
            }
        }
        mix = m2;
    }
    if let Some(o) = &t.outliers {
        mix = mix
            .into_iter()
            .flat_map(|(a, p)| [(a, p * (1.0 - o.prob)), (a * o.factor, p * o.prob)])
            .collect();
    }
    if mix.len() > 1 || mix[0].0 != 1.0 {
        d = Dist::Mixture {
            components: mix
                .into_iter()
                .map(|(a, p)| super::spec::Component {
                    weight: p,
                    dist: Dist::Affine {
                        a,
                        b: 0.0,
                        dist: Box::new(d.clone()),
                    },
                })
                .collect(),
        };
        notes.push("hierarchical scales: elements treated as independent with the scale-mixture marginal (analytic tiers)".into());
    }
    d
}

/// An f32 element law (continuous distributions rounded to f32): 16 equal-width bins per
/// binade with their probabilities, values at the bin midpoints (moments to ~0.1%), `ef` the
/// binade. For the CLT tier only (no residue structure: f32 values are taken to have random
/// low bits).
pub fn f32_smooth(d: &Dist) -> Pmf {
    let mut o = Pmf::default();
    if let Some(atoms) = d.atoms() {
        for (x, p) in atoms {
            let v = x as f32 as f64;
            o.v.push(v);
            o.p.push(p);
            o.ef.push(align_exp(v, Format::F32));
        }
        return o;
    }
    let z = d.mass(-1e-38, true, 1e-38, true);
    if z > 0.0 {
        o.v.push(0.0);
        o.p.push(z);
        o.ef.push(i32::MIN / 4);
    }
    for b in -126..128 {
        let (lo, hi) = (2f64.powi(b), 2f64.powi(b + 1));
        for s in [1.0f64, -1.0] {
            let m = if s > 0.0 {
                d.mass(lo, true, hi, false)
            } else {
                d.mass(-hi, false, -lo, true)
            };
            if m < 1e-15 {
                continue;
            }
            for k in 0..16 {
                let (a, c) = (
                    lo + (hi - lo) * k as f64 / 16.0,
                    lo + (hi - lo) * (k + 1) as f64 / 16.0,
                );
                let p = if s > 0.0 {
                    d.mass(a, true, c, false)
                } else {
                    d.mass(-c, false, -a, true)
                };
                if p > 1e-18 {
                    o.v.push(s * 0.5 * (a + c));
                    o.p.push(p);
                    o.ef.push(b);
                }
            }
        }
    }
    o
}

pub fn tensor_law(
    t: &TensorSpec,
    fmt: Format,
    scale_fmt: Option<Format>,
    notes: &mut Vec<String>,
) -> Result<TensorLaw, String> {
    let d = effective_dist(t, notes);
    Ok(match &t.quant {
        Quant::Round => {
            if matches!(fmt, Format::F32) {
                return Err("analytic tiers: f32 inputs are not supported (Tier 3 only)".into());
            }
            TensorLaw::Iid(code_pmf(&d, fmt, 1.0))
        }
        Quant::Cast { format } => TensorLaw::Iid(code_pmf(&d, *format, 1.0)),
        Quant::Codes { format } => {
            let f = format.unwrap_or(fmt);
            let atoms = d
                .atoms()
                .ok_or("quant codes needs a discrete distribution")?;
            let mut acc: BTreeMap<u64, f64> = BTreeMap::new();
            for (c, p) in atoms {
                let v = f.value(c as u32);
                if !v.is_finite() {
                    return Err(format!("code {c} of {f:?} is not finite (out of domain)"));
                }
                *acc.entry(norm0(v).to_bits()).or_default() += p;
            }
            let mut o = Pmf::default();
            for (b, p) in acc {
                let v = f64::from_bits(b);
                o.v.push(v);
                o.p.push(p);
                o.ef.push(align_exp(v, f));
            }
            TensorLaw::Iid(o)
        }
        Quant::Mx {
            elem,
            block,
            scale_rule,
            ..
        } => TensorLaw::Blocked {
            block: *block,
            scales: mx_scales(&d, *elem, *block, *scale_rule),
        },
        Quant::Nvfp4 {
            block,
            global_scale,
            ..
        } => {
            if *global_scale {
                notes.push(
                    "nvfp4 global scale taken at the median tensor amax (analytic tiers)".into(),
                );
            }
            let _ = scale_fmt;
            TensorLaw::Blocked {
                block: *block,
                scales: nvfp4_scales(&d, *block, *global_scale, t),
            }
        }
    })
}

/// P(|X| < t).
fn abs_cdf(d: &Dist, t: f64) -> f64 {
    d.mass(-t, false, t, false)
}

/// The MX joint law: scale exponent E from the block max (ceil / floor rule), elements
/// RNE(x / 2^E) given it, as F_hi^n mu_hi^n - F_lo^n mu_lo^n (normalized).
fn mx_scales(d: &Dist, elem: Format, n: usize, rule: ScaleRule) -> Vec<BlockScale> {
    let mut out = Vec::new();
    let max = elem.max();
    for e in -127..=127i32 {
        // amax range of exponent e.
        let (lo, hi) = match rule {
            ScaleRule::Ceil => (max * 2f64.powi(e - 1), max * 2f64.powi(e)),
            ScaleRule::Floor => {
                let k = e + exp_floor(max);
                (2f64.powi(k), 2f64.powi(k + 1))
            }
        };
        let (lo_incl_amax, hi_amax) = (lo, hi);
        // Ceil: amax in (lo, hi]; floor: [lo, hi). Atoms at the boundary matter only for
        // discrete laws; mass() with inclusions handles them.
        let (mut fl, fh) = match rule {
            ScaleRule::Ceil => (
                d.mass(-lo_incl_amax, true, lo_incl_amax, true),
                d.mass(-hi_amax, true, hi_amax, true),
            ),
            ScaleRule::Floor => (abs_cdf(d, lo_incl_amax), abs_cdf(d, hi_amax)),
        };
        if e == -127 {
            fl = 0.0; // every smaller amax (and 0) clamps here
        }
        let pe = fh.powi(n as i32) - fl.powi(n as i32);
        if pe < 1e-15 {
            continue;
        }
        let div = 2f64.powi(e);
        let part = |t: f64, incl: bool| -> Pmf {
            // Elements with |x| < t (<= t with incl), quantized at this scale; mass = P(.).
            let restricted = Restrict { d, t, incl };
            restricted.pmf(elem, div)
        };
        let (incl, f_hi, f_lo) = match rule {
            ScaleRule::Ceil => (true, fh, fl),
            ScaleRule::Floor => (false, fh, fl),
        };
        let mut mh = part(hi_amax, incl);
        let mut ml = part(lo_incl_amax, incl);
        mh.normalize();
        ml.normalize();
        let wh = f_hi.powi(n as i32) / pe;
        let wl = -f_lo.powi(n as i32) / pe;
        let mut parts = vec![(wh, mh)];
        if f_lo > 0.0 && wl.abs() > 1e-300 {
            parts.push((wl, ml));
        }
        out.push(BlockScale {
            value: div,
            ef: e,
            prob: pe,
            parts,
        });
    }
    out
}

/// NVFP4: scale s = RN_ue4m3(amax / (6 gs)); elements RNE_e2m1(x / (s gs)).
fn nvfp4_scales(d: &Dist, n: usize, global: bool, t: &TensorSpec) -> Vec<BlockScale> {
    let gs = if global {
        // The tensor's amax: its median over the tensor (many elements: concentrated).
        let elems = t
            .shape
            .as_ref()
            .map_or(32768.0, |s| s.iter().product::<u64>() as f64);
        let target = 0.5f64.powf(1.0 / elems); // P(|X| < a)^N = 1/2
        let (mut lo, mut hi) = (0.0, 1.0);
        while abs_cdf(d, hi) < target {
            hi *= 2.0;
        }
        for _ in 0..100 {
            let m = 0.5 * (lo + hi);
            if abs_cdf(d, m) < target {
                lo = m;
            } else {
                hi = m;
            }
        }
        ((0.5 * (lo + hi)) / (448.0 * 6.0)) as f32 as f64
    } else {
        1.0
    };
    let fmt = Format::Ue4m3;
    let mut grid: Vec<f64> = (0..127u32).map(|c| fmt.value(c)).collect();
    grid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut out = Vec::new();
    for (i, &s) in grid.iter().enumerate() {
        // amax / (6 gs) in the RN interval of s.
        let lo = if i == 0 { 0.0 } else { 0.5 * (grid[i - 1] + s) } * 6.0 * gs;
        let hi = if i + 1 == grid.len() {
            f64::INFINITY
        } else {
            0.5 * (s + grid[i + 1])
        } * 6.0
            * gs;
        let (fl, fh) = (abs_cdf(d, lo), abs_cdf(d, hi));
        let pe = fh.powi(n as i32) - fl.powi(n as i32);
        if pe < 1e-15 {
            continue;
        }
        let div = if s == 0.0 { 1.0 } else { s * gs };
        let mut mh = Restrict {
            d,
            t: hi,
            incl: false,
        }
        .pmf(Format::E2m1, div);
        let mut ml = Restrict {
            d,
            t: lo,
            incl: false,
        }
        .pmf(Format::E2m1, div);
        mh.normalize();
        ml.normalize();
        let mut parts = vec![(fh.powi(n as i32) / pe, mh)];
        if fl > 0.0 {
            parts.push((-fl.powi(n as i32) / pe, ml));
        }
        out.push(BlockScale {
            value: s,
            ef: if s == 0.0 {
                i32::MIN / 4
            } else {
                align_exp(s, fmt)
            },
            prob: pe,
            parts,
        });
    }
    out
}

/// X restricted to |X| < t (or <= t): its quantized PMF, unnormalized.
struct Restrict<'a> {
    d: &'a Dist,
    t: f64,
    incl: bool,
}

impl Restrict<'_> {
    fn pmf(&self, fmt: Format, div: f64) -> Pmf {
        // Atoms: filter; continuous: clip the CDF.
        if let Some(atoms) = self.d.atoms() {
            let mut acc: BTreeMap<u64, f64> = BTreeMap::new();
            for (x, p) in atoms {
                if x.abs() < self.t || (self.incl && x.abs() == self.t) {
                    let (q, _) = fmt.round(x / div, true);
                    *acc.entry(norm0(q).to_bits()).or_default() += p;
                }
            }
            let mut o = Pmf::default();
            for (b, p) in acc {
                let v = f64::from_bits(b);
                o.v.push(v);
                o.p.push(p);
                o.ef.push(align_exp(v, fmt));
            }
            return o;
        }
        let clipped = Clip {
            d: self.d.clone(),
            t: self.t,
        };
        code_pmf_fn(&clipped, fmt, div)
    }
}

/// A distribution clipped to |x| < t (mass outside removed, not renormalized).
struct Clip {
    d: Dist,
    t: f64,
}

fn code_pmf_fn(c: &Clip, fmt: Format, div: f64) -> Pmf {
    let mut grid: Vec<f64> = (0..(1u32 << fmt.width()))
        .map(|k| fmt.value(k))
        .filter(|v| v.is_finite() && *v >= 0.0)
        .collect();
    grid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    grid.dedup();
    let mass = |lo: f64, hi: f64| -> f64 {
        let (lo, hi) = (lo.max(-c.t), hi.min(c.t));
        if hi <= lo {
            0.0
        } else {
            c.d.mass(lo, false, hi, true)
        }
    };
    let mut acc: HashMap<u64, f64> = HashMap::new();
    let n = grid.len();
    for i in 0..n {
        let v = grid[i];
        let lo = if i == 0 { 0.0 } else { 0.5 * (grid[i - 1] + v) } * div;
        let hi = if i + 1 == n {
            f64::INFINITY
        } else {
            0.5 * (v + grid[i + 1])
        } * div;
        if v == 0.0 {
            let hi0 = if n > 1 {
                0.5 * grid[1] * div
            } else {
                f64::INFINITY
            };
            *acc.entry(0f64.to_bits()).or_default() += mass(-hi0, hi0);
            continue;
        }
        *acc.entry(v.to_bits()).or_default() += mass(lo, hi);
        if fmt.signed() {
            *acc.entry((-v).to_bits()).or_default() += mass(-hi, -lo);
        } else {
            *acc.entry(v.to_bits()).or_default() += mass(-hi, -lo);
        }
    }
    let mut o = Pmf::default();
    let mut keys: Vec<(u64, f64)> = acc.into_iter().collect();
    keys.sort_by(|a, b| {
        f64::from_bits(a.0)
            .partial_cmp(&f64::from_bits(b.0))
            .unwrap()
    });
    for (b, p) in keys {
        if p < PRUNE {
            o.pruned += p;
            continue;
        }
        let v = f64::from_bits(b);
        o.v.push(v);
        o.p.push(p);
        o.ef.push(align_exp(v, fmt));
    }
    o
}

// ------------------------------------------------------------------------------ clouds

/// A law on the reals as merged atoms: (value, probability), sorted by value. Atoms are
/// merged (probability-weighted mean) in bins of 1/RES binade per sign when there are more
/// than `cap`; exact while small (discrete inputs).
#[derive(Clone, Debug, Default)]
pub struct Cloud {
    pub a: Vec<(f64, f64)>,
    /// Some atoms were merged (the cloud approximates a law with a density).
    pub merged: bool,
}

pub const RES: f64 = 64.0;

/// Relative half-width of a merged atom: a sum of merged atoms closer to 0 than twice this is
/// an artifact of the merging; its mass is spread uniformly over [s - h, s + h].
pub const SMOOTH: f64 = 1.0 / 16.0;

/// Append the atom x + y (weight w); with `smooth`, near-cancellations are spread.
pub fn push_sum(out: &mut Vec<(f64, f64)>, x: f64, y: f64, w: f64, smooth: bool) {
    let s = x + y;
    let h = SMOOTH * (x.abs() + y.abs());
    if smooth && s.abs() < 2.0 * h && h > 0.0 {
        for k in [-0.75, -0.25, 0.25, 0.75] {
            out.push((s + k * h, w / 4.0));
        }
    } else {
        out.push((s, w));
    }
}

/// One log bin's sums: |weight|, sum v |w|, sum v^2 |w|, signed weight.
#[derive(Clone, Copy, Default)]
pub struct Bin {
    pub w: f64,
    pub s1: f64,
    pub s2: f64,
    pub signed: f64,
}

fn log_key(v: f64) -> (i8, i64) {
    if v == 0.0 {
        (0, 0)
    } else {
        (v.signum() as i8, (v.abs().log2() * RES).floor() as i64)
    }
}

/// Multiplicative hasher for small integer keys (the bins of `log_bins`).
#[derive(Default, Clone, Copy)]
pub struct MulHash(u64);

impl std::hash::Hasher for MulHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, b: &[u8]) {
        for &x in b {
            self.write_u64(x as u64);
        }
    }
    fn write_u8(&mut self, x: u8) {
        self.write_u64(x as u64);
    }
    fn write_i8(&mut self, x: i8) {
        self.write_u64(x as u64);
    }
    fn write_i64(&mut self, x: i64) {
        self.write_u64(x as u64);
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

type BinMap = HashMap<(i8, i64), Bin, std::hash::BuildHasherDefault<MulHash>>;

/// Bin atoms by (sign, floor(log2|v| * res)), res = RES halved (bins merged pairwise) until at
/// most `max_bins` bins or res < 1/64; sorted by key.
pub fn log_bins(a: &[(f64, f64)], max_bins: usize) -> Vec<Bin> {
    let mut m: BinMap = BinMap::with_capacity_and_hasher(4 * max_bins, Default::default());
    for &(v, p) in a {
        let b = m.entry(log_key(v)).or_default();
        let w = p.abs();
        b.w += w;
        b.s1 += v * w;
        b.s2 += v * v * w;
        b.signed += p;
    }
    let mut res = RES;
    while m.len() > max_bins && res >= 1.0 / 64.0 {
        res /= 2.0;
        let mut n: BinMap = BinMap::with_capacity_and_hasher(m.len() / 2 + 1, Default::default());
        for ((s, k), b) in m {
            let e = n
                .entry((s, if s == 0 { 0 } else { k.div_euclid(2) }))
                .or_default();
            e.w += b.w;
            e.s1 += b.s1;
            e.s2 += b.s2;
            e.signed += b.signed;
        }
        m = n;
    }
    let mut v: Vec<((i8, i64), Bin)> = m.into_iter().collect();
    v.sort_by_key(|x| (x.0.0, if x.0.0 < 0 { -x.0.1 } else { x.0.1 }));
    v.into_iter().map(|x| x.1).collect()
}

impl Cloud {
    pub fn point(x: f64) -> Cloud {
        Cloud {
            a: vec![(x, 1.0)],
            merged: false,
        }
    }
    pub fn mass(&self) -> f64 {
        self.a.iter().map(|x| x.1).sum()
    }
    pub fn from_atoms(mut a: Vec<(f64, f64)>, cap: usize) -> Cloud {
        a.retain(|x| x.1 > 0.0);
        let mut c = Cloud { a, merged: false };
        c.compact(cap);
        c
    }
    /// Sort, merge equal values, and merge into log bins while over `cap` atoms.
    pub fn compact(&mut self, cap: usize) {
        if self.a.len() <= 4 * cap {
            self.a.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            let mut out: Vec<(f64, f64)> = Vec::with_capacity(self.a.len());
            for &(v, p) in &self.a {
                match out.last_mut() {
                    Some(l) if l.0 == v => l.1 += p,
                    _ => out.push((v, p)),
                }
            }
            self.a = out;
            if self.a.len() <= cap {
                return;
            }
        }
        self.merged = true;
        // Bins of 1/res binade per sign, res halved until at most cap / 2 bins; each bin
        // becomes two atoms at mean +- sd (its mean and variance kept exactly).
        let bins = log_bins(&self.a, cap / 2);
        let mut out = Vec::with_capacity(bins.len() * 2);
        for b in bins {
            let w = b.w;
            if w <= 0.0 {
                continue;
            }
            let m = b.s1 / w;
            let sd = (b.s2 / w - m * m).max(0.0).sqrt();
            if sd <= 1e-12 * m.abs() {
                out.push((m, w));
            } else {
                out.push((m - sd, w / 2.0));
                out.push((m + sd, w / 2.0));
            }
        }
        out.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        self.a = out;
    }
    /// The law of X + Y, X ~ self, Y ~ o independent.
    pub fn conv(&self, o: &Cloud, cap: usize) -> Cloud {
        let mut a = Vec::with_capacity(self.a.len() * o.a.len());
        let sm = self.merged || o.merged;
        for &(x, p) in &self.a {
            for &(y, q) in &o.a {
                push_sum(&mut a, x, y, p * q, sm);
            }
        }
        let mut c = Cloud::from_atoms(a, cap);
        c.merged |= self.merged || o.merged;
        c
    }
    pub fn scale(&self, k: f64) -> Cloud {
        Cloud {
            a: self.a.iter().map(|&(v, p)| (v * k, p)).collect(),
            merged: self.merged,
        }
    }
    /// n-fold self-convolution by doubling.
    pub fn power(&self, n: usize, cap: usize) -> Cloud {
        let mut result: Option<Cloud> = None;
        let mut base = self.clone();
        let mut k = n;
        while k > 0 {
            if k & 1 == 1 {
                result = Some(match result {
                    None => base.clone(),
                    Some(r) => r.conv(&base, cap),
                });
            }
            k >>= 1;
            if k > 0 {
                base = base.conv(&base, cap);
            }
        }
        result.unwrap_or_else(|| Cloud::point(0.0))
    }
    pub fn mix(parts: &[(f64, Cloud)], cap: usize) -> Cloud {
        let mut a = Vec::new();
        for (w, c) in parts {
            for &(v, p) in &c.a {
                a.push((v, w * p));
            }
        }
        // Signed mixtures can leave tiny negative atoms after merging: clip.
        let mut c = Cloud { a, merged: false };
        c.compact(cap);
        c.a.retain(|x| x.1 > 1e-300);
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errdist::special::Rng;

    #[test]
    fn code_pmfs_match_sampling() {
        let d = Dist::Normal {
            mean: 0.0,
            std: 1.0,
        };
        let p = code_pmf(&d, Format::E4m3, 1.0 / 64.0);
        assert!((p.total() + p.pruned - 1.0).abs() < 1e-12);
        let mut r = Rng::new(5);
        let mut counts: HashMap<u64, f64> = HashMap::new();
        let n = 400_000;
        for _ in 0..n {
            let x = r.normal();
            let (q, _) = Format::E4m3.round(x * 64.0, true);
            *counts.entry(norm0(q).to_bits()).or_default() += 1.0 / n as f64;
        }
        for (v, pr) in p.v.iter().zip(&p.p) {
            let c = counts.get(&v.to_bits()).copied().unwrap_or(0.0);
            assert!(
                (c - pr).abs() < 5.0 * (pr / n as f64).sqrt() + 1e-5,
                "{v}: {c} vs {pr}"
            );
        }
    }

    #[test]
    fn mx_block_law_matches_sampling() {
        // Block of 4 (small, so the max coupling is strong): P(scale) and the element law
        // given the scale against sampling.
        let d = Dist::Normal {
            mean: 0.0,
            std: 1.0,
        };
        let sc = mx_scales(&d, Format::E4m3, 4, ScaleRule::Ceil);
        let tot: f64 = sc.iter().map(|s| s.prob).sum();
        assert!((tot - 1.0).abs() < 1e-9);
        let mut r = Rng::new(7);
        let n = 200_000;
        let mut pe: HashMap<i32, f64> = HashMap::new();
        let mut pz: HashMap<i32, f64> = HashMap::new(); // P(element code is the top binade | e)
        for _ in 0..n {
            let x: Vec<f64> = (0..4).map(|_| r.normal()).collect();
            let amax = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let e = ScaleRule::Ceil.exponent(amax, Format::E4m3);
            *pe.entry(e).or_default() += 1.0;
            let (q, _) = Format::E4m3.round(x[0] / 2f64.powi(e), true);
            *pz.entry(e).or_default() += (q.abs() >= 256.0) as u8 as f64;
        }
        for s in &sc {
            let obs = pe.get(&s.ef).copied().unwrap_or(0.0) / n as f64;
            assert!(
                (obs - s.prob).abs() < 4.0 * (s.prob / n as f64).sqrt() + 1e-4,
                "e {}: {obs} vs {}",
                s.ef,
                s.prob
            );
            if s.prob > 0.05 {
                let pred: f64 = s
                    .parts
                    .iter()
                    .map(|(w, m)| {
                        w * m
                            .v
                            .iter()
                            .zip(&m.p)
                            .filter(|(v, _)| v.abs() >= 256.0)
                            .map(|x| x.1)
                            .sum::<f64>()
                    })
                    .sum();
                let o = pz[&s.ef] / pe[&s.ef];
                assert!(
                    (o - pred).abs() < 0.01,
                    "e {}: top binade {o} vs {pred}",
                    s.ef
                );
            }
        }
    }

    #[test]
    fn cloud_power_keeps_moments() {
        let c = Cloud::from_atoms(vec![(-1.0, 0.25), (0.5, 0.5), (2.0, 0.25)], 64);
        let p = c.power(16, 256);
        let m: f64 = p.a.iter().map(|x| x.0 * x.1).sum();
        let v: f64 = p.a.iter().map(|x| x.0 * x.0 * x.1).sum::<f64>() - m * m;
        assert!((p.mass() - 1.0).abs() < 1e-12);
        assert!((m - 16.0 * 0.5).abs() < 1e-9);
        let v1 = 0.25 + 0.125 + 1.0 - 0.25;
        assert!((v - 16.0 * v1).abs() < 0.05 * 16.0 * v1, "{v}");
    }
}

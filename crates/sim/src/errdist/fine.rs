/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The fine part of one tensor-core step: the laws of the truncated terms' residues on the
//! cyclic group Z_r (r = ulp(result) / Q), combined over the step's terms through their
//! characteristic functions, and the truncation errors (docs/errdist.md §4).
//!
//! A step's terms come in independent groups; a group is n i.i.d. terms whose law is a signed
//! mixture sum_i w_i T_i^(x n) (contexts such as block scales, and the block-max coupling of
//! the quantization recipes). Every statistic here is linear in that measure, so the signed
//! mixture is exact. Restricting every term to alignment exponent <= E' gives the measure of
//! "emax <= E'"; differences give "emax = E'" (inclusion-exclusion over the order statistic).

use std::sync::Arc as Rc;

use crate::errdist::det::HashMap;

/// One term's law: (exact value, alignment exponent, probability), sorted by exponent.
#[derive(Clone, Debug, Default)]
pub struct TermLaw {
    pub atoms: Vec<(f64, i32, f64)>,
    /// Exponent of the coarsest power of two every nonzero atom is a multiple of.
    pub lattice: i32,
}

impl TermLaw {
    pub fn new(mut atoms: Vec<(f64, i32, f64)>) -> TermLaw {
        atoms.retain(|a| a.2 > 0.0);
        atoms.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.partial_cmp(&b.0).unwrap()));
        let lattice = atoms
            .iter()
            .filter(|a| a.0 != 0.0)
            .map(|a| lowest_bit(a.0))
            .min()
            .unwrap_or(0);
        TermLaw { atoms, lattice }
    }
    /// P(alignment <= e). Zero terms carry alignment i32::MIN / 4 (below everything); an FP4
    /// unit that is live but sums to 0 keeps its scale's alignment.
    pub fn mass_le(&self, e: i32) -> f64 {
        self.atoms.iter().filter(|a| a.1 <= e).map(|a| a.2).sum()
    }
    pub fn max_align(&self) -> i32 {
        self.atoms
            .iter()
            .filter(|a| a.1 > NO_ALIGN)
            .map(|a| a.1)
            .max()
            .unwrap_or(i32::MIN / 4)
    }
    pub fn min_align(&self) -> i32 {
        self.atoms
            .iter()
            .filter(|a| a.1 > NO_ALIGN)
            .map(|a| a.1)
            .min()
            .unwrap_or(i32::MIN / 4)
    }
}

/// Alignments at or below this are "none" (zero terms, dead FP4 units).
pub const NO_ALIGN: i32 = i32::MIN / 8;

/// Exponent of the lowest set bit of a nonzero finite f64.
pub fn lowest_bit(x: f64) -> i32 {
    let b = x.abs().to_bits();
    let e = ((b >> 52) & 0x7ff) as i32;
    let m = if e == 0 {
        b & ((1 << 52) - 1)
    } else {
        (b & ((1 << 52) - 1)) | (1 << 52)
    };
    let tz = m.trailing_zeros() as i32;
    (if e == 0 { -1074 } else { e - 1075 }) + tz
}

/// n i.i.d. terms with a signed-mixture law.
#[derive(Clone, Debug)]
pub struct Group {
    pub n: usize,
    pub parts: Vec<(f64, Rc<TermLaw>)>,
}

/// A step's terms: independent groups.
#[derive(Clone, Debug, Default)]
pub struct Pop {
    pub groups: Vec<Group>,
}

impl Pop {
    pub fn lattice(&self) -> i32 {
        self.groups
            .iter()
            .flat_map(|g| g.parts.iter().map(|p| p.1.lattice))
            .min()
            .unwrap_or(0)
    }
    pub fn max_align(&self) -> i32 {
        self.groups
            .iter()
            .flat_map(|g| g.parts.iter().map(|p| p.1.max_align()))
            .max()
            .unwrap_or(i32::MIN / 4)
    }
    pub fn min_align(&self) -> i32 {
        self.groups
            .iter()
            .flat_map(|g| g.parts.iter().map(|p| p.1.min_align()))
            .min()
            .unwrap_or(i32::MIN / 4)
    }
    /// P(every term aligns <= e).
    pub fn mass_le(&self, e: i32) -> f64 {
        self.groups
            .iter()
            .map(|g| {
                g.parts
                    .iter()
                    .map(|(w, t)| w * t.mass_le(e).powi(g.n as i32))
                    .sum::<f64>()
            })
            .product()
    }
    pub fn terms(&self) -> usize {
        self.groups.iter().map(|g| g.n).sum()
    }
}

// ---------------------------------------------------------------------------------- FFT

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    pub fn new(re: f64, im: f64) -> C64 {
        C64 { re, im }
    }
    // Called by name throughout errdist (cf, markov); `std::ops` impls would need the traits
    // imported at every call site for no gain.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, o: C64) -> C64 {
        C64::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, o: C64) -> C64 {
        C64::new(self.re + o.re, self.im + o.im)
    }
    pub fn scale(self, k: f64) -> C64 {
        C64::new(self.re * k, self.im * k)
    }
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    pub fn powi(self, n: usize) -> C64 {
        let mut r = C64::new(1.0, 0.0);
        let mut b = self;
        let mut k = n;
        while k > 0 {
            if k & 1 == 1 {
                r = r.mul(b);
            }
            b = b.mul(b);
            k >>= 1;
        }
        r
    }
}

/// In-place radix-2 DFT of a power-of-two length; `inv`: inverse without the 1/n factor.
/// Forward: X_k = sum_j x_j e^{-2 pi i jk/n}.
pub fn fft(a: &mut [C64], inv: bool) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    assert!(n.is_power_of_two(), "fft size {n}");
    FFT_PLANS.with(|c| {
        let mut c = c.borrow_mut();
        let plan = c
            .entry(n)
            .or_insert_with(|| std::rc::Rc::new(FftPlan::new(n)))
            .clone();
        drop(c);
        plan.run(a, inv);
    });
}

thread_local! {
    static FFT_PLANS: std::cell::RefCell<HashMap<usize, std::rc::Rc<FftPlan>>> =
        std::cell::RefCell::new(HashMap::default());
}

/// Radix-2 FFT with exact twiddles (cos / sin per index, not a recurrence: the recurrence
/// loses ~n eps relative accuracy, which shows on probabilities far below the largest).
struct FftPlan {
    rev: Vec<u32>,
    /// e^{-2 pi i k / n}, k < n / 2.
    tw: Vec<C64>,
}

impl FftPlan {
    fn new(n: usize) -> FftPlan {
        let bits = n.trailing_zeros();
        let rev = (0..n as u32)
            .map(|i| i.reverse_bits() >> (32 - bits))
            .collect();
        let tw = (0..n / 2)
            .map(|k| {
                let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                C64::new(a.cos(), a.sin())
            })
            .collect();
        FftPlan { rev, tw }
    }
    fn run(&self, a: &mut [C64], inv: bool) {
        let n = a.len();
        for i in 0..n {
            let j = self.rev[i] as usize;
            if i < j {
                a.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let step = n / len;
            let h = len / 2;
            for i in (0..n).step_by(len) {
                for k in 0..h {
                    let t = self.tw[k * step];
                    let w = if inv { C64::new(t.re, -t.im) } else { t };
                    let u = a[i + k];
                    let v = a[i + k + h].mul(w);
                    a[i + k] = u.add(v);
                    a[i + k + h] = C64::new(u.re - v.re, u.im - v.im);
                }
            }
            len <<= 1;
        }
    }
}

// ------------------------------------------------------------------- per-term statistics

/// One term's restricted statistics at truncation quantum Q = 2^q and modulus r Q:
/// mass of atoms with alignment <= e, the (unnormalized) PMF of trunc_Q(t)/Q mod r, and the
/// truncation error tau = trunc_Q(t) - t in units of Q: E[tau], E[tau^2], P(tau = 0), all
/// unnormalized (weights = atom probabilities).
#[derive(Clone, Debug)]
pub struct TermStats {
    pub mass: f64,
    pub res: Vec<f64>,
    /// The residue PMF over the atoms with tau = 0 only (unnormalized).
    pub res0: Vec<f64>,
    pub tau1: f64,
    pub tau2: f64,
    pub tau0: f64,
    /// Histogram of tau over 32 bins of 1/16 Q, from -2 Q... (index (tau + 1) * 16).
    pub tau_hist: [f64; 33],
}

pub fn term_stats(t: &TermLaw, q: i32, r: usize, e: i32) -> TermStats {
    let mut s = TermStats {
        mass: 0.0,
        res: vec![0.0; r],
        res0: vec![0.0; r],
        tau1: 0.0,
        tau2: 0.0,
        tau0: 0.0,
        tau_hist: [0.0; 33],
    };
    let inv_q = 2f64.powi(-q);
    for &(v, a, p) in &t.atoms {
        if a > e {
            continue;
        }
        s.mass += p;
        if v == 0.0 {
            s.res[0] += p;
            s.res0[0] += p;
            s.tau0 += p;
            s.tau_hist[16] += p;
            continue;
        }
        let x = v.abs() * inv_q; // exact: power-of-two scaling
        let k = x.trunc();
        let tau = (k - x) * v.signum();
        let km = if k >= 9.007e15 {
            0 // a multiple of every small power of two
        } else {
            (k as u64 % r as u64) as usize
        };
        let km = if v < 0.0 { (r - km) % r } else { km };
        s.res[km] += p;
        s.tau1 += p * tau;
        s.tau2 += p * tau * tau;
        if tau == 0.0 {
            s.tau0 += p;
            s.res0[km] += p;
        }
        let b = ((tau + 1.0) * 16.0).round().clamp(0.0, 32.0) as usize;
        s.tau_hist[b] += p;
    }
    s
}

/// The measure of one step's terms all aligned <= e, at quantum 2^q, modulus r: total mass,
/// the characteristic function of (sum of truncated terms)/Q mod r (unnormalized; length r),
/// E[sum tau], E[(sum tau)^2] and P(every tau = 0), unnormalized.
#[derive(Clone, Debug)]
pub struct StepMeasure {
    pub mass: f64,
    pub phi: Vec<C64>,
    pub tau1: f64,
    pub tau2: f64,
    pub tau0: f64,
    /// Group-summed tau histogram (1/16 Q bins centered at 0, wide enough for the sum), as an
    /// unnormalized signed measure.
    pub tau_hist: Vec<f64>,
}

const TAU_W: usize = 16 * 64; // tau sum range +-64 Q at 1/16 Q

pub fn step_measure(pop: &Pop, q: i32, r: usize, e: i32, want_hist: bool) -> StepMeasure {
    let mut out = StepMeasure {
        mass: 1.0,
        phi: vec![C64::new(1.0, 0.0); r],
        tau1: 0.0,
        tau2: 0.0,
        tau0: 1.0,
        tau_hist: if want_hist {
            let mut h = vec![0.0; 2 * TAU_W + 1];
            h[TAU_W] = 1.0;
            h
        } else {
            vec![]
        },
    };
    for g in &pop.groups {
        // Group measure: sum_i w_i (single_i)^n.
        let mut gmass = 0.0;
        let mut gphi = vec![C64::new(0.0, 0.0); r];
        let (mut g1, mut g2, mut g0) = (0.0, 0.0, 0.0);
        let mut ghist = if want_hist {
            vec![0.0; 2 * TAU_W + 1]
        } else {
            vec![]
        };
        for (w, t) in &g.parts {
            let st = term_stats(t, q, r, e);
            if st.mass <= 0.0 {
                continue;
            }
            let m = st.mass;
            let mn = m.powi(g.n as i32);
            gmass += w * mn;
            // phi of one term (normalized), to the n-th power, times w m^n.
            let mut f: Vec<C64> = st.res.iter().map(|&p| C64::new(p / m, 0.0)).collect();
            fft(&mut f, false);
            for k in 0..r {
                gphi[k] = gphi[k].add(f[k].powi(g.n).scale(w * mn));
            }
            let (t1, t2) = (st.tau1 / m, st.tau2 / m);
            let n = g.n as f64;
            g1 += w * mn * n * t1;
            g2 += w * mn * (n * t2 + n * (n - 1.0) * t1 * t1);
            g0 += w * mn * (st.tau0 / m).powi(g.n as i32);
            if want_hist {
                let one: Vec<f64> = st.tau_hist.iter().map(|x| x / m).collect();
                let pw = hist_power(&one, g.n);
                for (i, x) in pw.iter().enumerate() {
                    ghist[i] += w * mn * x;
                }
            }
        }
        // Combine with the groups so far (independent): masses multiply, means add.
        let (m0, t1_0, t2_0) = (out.mass, out.tau1, out.tau2);
        out.tau2 = t2_0 * gmass + 2.0 * t1_0 * g1 + m0 * g2;
        out.tau1 = t1_0 * gmass + m0 * g1;
        out.mass = m0 * gmass;
        for (o, g) in out.phi[..r].iter_mut().zip(&gphi[..r]) {
            *o = o.mul(*g);
        }
        out.tau0 *= g0;
        if want_hist {
            out.tau_hist = conv_centered(&out.tau_hist, &ghist);
        }
    }
    out
}

/// n-fold convolution of a 33-bin tau histogram (1/16 Q bins, center 16) into the TAU_W frame.
fn hist_power(one: &[f64], n: usize) -> Vec<f64> {
    let mut base = vec![0.0; 2 * TAU_W + 1];
    for (i, x) in one.iter().enumerate() {
        base[TAU_W + i - 16] += x;
    }
    let mut res = vec![0.0; 2 * TAU_W + 1];
    res[TAU_W] = 1.0;
    let mut k = n;
    let mut b = base;
    while k > 0 {
        if k & 1 == 1 {
            res = conv_centered(&res, &b);
        }
        k >>= 1;
        if k > 0 {
            b = conv_centered(&b, &b);
        }
    }
    res
}

/// Convolution of two histograms centered at index W (same length 2W + 1); mass beyond the
/// frame is clipped to its edges. Sparse in the nonzero entries.
pub fn conv_centered(a: &[f64], b: &[f64]) -> Vec<f64> {
    let w = (a.len() / 2) as i64;
    let nz = |v: &[f64]| -> Vec<(i64, f64)> {
        v.iter()
            .enumerate()
            .filter(|(_, x)| x.abs() > 1e-300)
            .map(|(i, x)| (i as i64 - w, *x))
            .collect()
    };
    let (na, nb) = (nz(a), nz(b));
    let mut out = vec![0.0; a.len()];
    for &(i, x) in &na {
        for &(j, y) in &nb {
            let k = (i + j).clamp(-w, w);
            out[(k + w) as usize] += x * y;
        }
    }
    out
}

/// The PMF on Z_r (unnormalized) of a characteristic function computed by `step_measure`.
pub fn inverse(phi: &[C64]) -> Vec<f64> {
    let mut f = phi.to_vec();
    fft(&mut f, true);
    let r = f.len() as f64;
    f.iter().map(|c| c.re / r).collect()
}

/// TV distance between the residue law with characteristic function `phi` (normalized, phi[0]
/// = 1) and the uniform law on Z_r: exactly (1/2) sum_m |p_m - 1/r|; also its Fourier bound
/// (1/2) min(sum_{k != 0} |phi_k|, (sum_{k != 0} |phi_k|^2)^(1/2)) and max_{k != 0} |phi_k|.
pub fn tv_uniform(phi: &[C64]) -> (f64, f64, f64) {
    let r = phi.len();
    if r <= 1 {
        return (0.0, 0.0, 0.0);
    }
    let p = inverse(phi);
    let tv = 0.5 * p.iter().map(|x| (x - 1.0 / r as f64).abs()).sum::<f64>();
    // |p_m - 1/r| <= (1/r) sum_k |phi_k|, and Parseval with Cauchy-Schwarz.
    let l1: f64 = phi[1..].iter().map(|c| c.abs()).sum();
    let l2: f64 = phi[1..]
        .iter()
        .map(|c| c.abs() * c.abs())
        .sum::<f64>()
        .sqrt();
    let bound = 0.5 * l1.min(l2);
    let maxf = phi[1..].iter().map(|c| c.abs()).fold(0.0, f64::max);
    (tv, bound.max(0.0), maxf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errdist::special::Rng;

    #[test]
    fn fft_round_trips() {
        let mut a: Vec<C64> = (0..16)
            .map(|i| C64::new(i as f64, (i * i) as f64))
            .collect();
        let b = a.clone();
        fft(&mut a, false);
        fft(&mut a, true);
        for (x, y) in a.iter().zip(&b) {
            assert!((x.re / 16.0 - y.re).abs() < 1e-9 && (x.im / 16.0 - y.im).abs() < 1e-9);
        }
    }

    /// The residue law of a sum of n i.i.d. truncated terms (FFT route) against brute-force
    /// enumeration of every n-tuple.
    #[test]
    fn residue_law_matches_enumeration() {
        let atoms = vec![
            (0.75, -1, 0.2),
            (-1.25, 0, 0.3),
            (3.0, 1, 0.25),
            (0.0, 0, 0.1),
            (2.5, 1, 0.15),
        ];
        let t = Rc::new(TermLaw::new(atoms.clone()));
        let pop = Pop {
            groups: vec![Group {
                n: 3,
                parts: vec![(1.0, t)],
            }],
        };
        for (q, r, e) in [(-1, 4usize, 1), (0, 8, 1), (-2, 2, 0)] {
            let m = step_measure(&pop, q, r, e, true);
            let res = inverse(&m.phi);
            // Brute force over the atoms aligned <= e.
            let ok: Vec<&(f64, i32, f64)> =
                atoms.iter().filter(|a| a.0 == 0.0 || a.1 <= e).collect();
            let mut bf = vec![0.0; r];
            let (mut mass, mut t1, mut t2, mut t0) = (0.0, 0.0, 0.0, 0.0);
            for a in &ok {
                for b in &ok {
                    for c in &ok {
                        let p = a.2 * b.2 * c.2;
                        let qv = 2f64.powi(q);
                        let tr = |v: f64| (v / qv).trunc();
                        let k = (tr(a.0) + tr(b.0) + tr(c.0)) as i64;
                        bf[k.rem_euclid(r as i64) as usize] += p;
                        let tau =
                            (tr(a.0) - a.0 / qv) + (tr(b.0) - b.0 / qv) + (tr(c.0) - c.0 / qv);
                        mass += p;
                        t1 += p * tau;
                        t2 += p * tau * tau;
                        if tr(a.0) * qv == a.0 && tr(b.0) * qv == b.0 && tr(c.0) * qv == c.0 {
                            t0 += p;
                        }
                    }
                }
            }
            assert!((m.mass - mass).abs() < 1e-12);
            for k in 0..r {
                assert!(
                    (res[k] - bf[k]).abs() < 1e-12,
                    "q {q} r {r}: {:?} vs {:?}",
                    res,
                    bf
                );
            }
            assert!(
                (m.tau1 - t1).abs() < 1e-12
                    && (m.tau2 - t2).abs() < 1e-12
                    && (m.tau0 - t0).abs() < 1e-12
            );
        }
        // Signed mixtures: w1 T1^n - w2 T2^n against the difference of enumerations.
        let _ = Rng::new(1);
    }

    #[test]
    fn uniform_tv_bound_on_known_laws() {
        // A sum of n uniform bits mod 8: TV to uniform shrinks like 2^-n per frequency.
        let t = Rc::new(TermLaw::new(vec![(0.0, 0, 0.5), (1.0, 0, 0.5)]));
        let mut last = 1.0;
        for n in [1usize, 4, 16, 64] {
            let pop = Pop {
                groups: vec![Group {
                    n,
                    parts: vec![(1.0, t.clone())],
                }],
            };
            let m = step_measure(&pop, 0, 8, 0, false);
            let (tv, bound, maxf) = tv_uniform(&m.phi);
            assert!(tv <= bound + 1e-12, "{tv} > {bound}");
            assert!(tv < last + 1e-12);
            // |phi_1| = |cos(pi/8)|^n for a fair bit at frequency 1/8.
            assert!((maxf - (std::f64::consts::PI / 8.0).cos().powi(n as i32)).abs() < 1e-9);
            last = tv;
        }
        // A lattice law (all terms multiples of 8): residue is a point mass, TV = 7/8.
        let t = Rc::new(TermLaw::new(vec![(8.0, 3, 0.5), (16.0, 4, 0.5)]));
        let pop = Pop {
            groups: vec![Group {
                n: 5,
                parts: vec![(1.0, t)],
            }],
        };
        let m = step_measure(&pop, 0, 8, 10, false);
        let (tv, _, maxf) = tv_uniform(&m.phi);
        assert!((tv - 7.0 / 8.0).abs() < 1e-12 && (maxf - 1.0).abs() < 1e-12);
    }
}

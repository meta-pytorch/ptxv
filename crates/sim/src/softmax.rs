/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Normalized exponentials in `ptxv equiv` (docs/equiv-reassoc.md, "Softmax").
//!
//! In segment mode (`num.rs`) a softmax output is a DAG of exact maxima (`Max`), sums and
//! products (`Add`, `Fma`) and instruction slices (`Seg`: an exp idiom, a reciprocal, ptxas's
//! division). This module puts each output into the normal form
//!
//!   y_i = exp(L_i) / sum_j c_j exp(L_j),   L = x - max(X) linear in input leaves and maxima,
//!
//! with, per kernel, an error record for every term: the slices' contracts (found by evaluating
//! each one-argument slice on every f32 argument of its domain, `sweep`), the rounding nodes on
//! the term's path, and the argument roundings. Online softmax's rescaling by exp(m_old - m_new)
//! telescopes in the normal form (exp(a) exp(b) = exp(a + b); the maxima cancel). Two kernels
//! compute the same function when their normal forms are equal; the bound is the softmax
//! theorem below, per kernel, added (fp_noise Thm 4).
//!
//! The theorem (docs/equiv-reassoc.md has the proof). With p_j = exp(L_j) / S, every
//! L_j <= 0, S >= 1 (the term of the maximum is exp(0)), each computed term
//! exp(L_j) phi_j + alpha_j, phi_j in [lo_j e^(-s|L_j|), up_j e^(s|L_j|)], |alpha_j| <= A_j, and
//! the division's relative error in [rho_lo, rho_up]:
//!   |y^ - y| <= K_rel y + K_arg + K_abs,
//!   K_rel = max(up_i rho_up / S_lo - 1, 1 - lo_i rho_lo / S_hi),
//!   K_arg = max(up_i rho_up / S_lo, 1) s / (e (1 - s)),
//!   S_lo = min lo (1 - s ln n) - A_S,   S_hi = max up (1 + s ln n e^(1024 s)) + n e^-1000 + A_S,
//! using sum_j p_j |L_j| <= ln n (the entropy of p) and exp(-t)(e^(st) - 1) <= s / (e (1 - s)).
//! ASSUME(N-SOFTMAX): see ASSUMPTIONS.md

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::sync::Mutex;

use ptxv_ir::*;
use serde::Serialize;

use crate::equiv::Leaf;
use crate::num::NodeKind;
use crate::num::NumState;
use crate::num::U32;
use crate::seg::SKind;
use crate::seg::SOp;
use crate::seg::SProg;
use crate::seg::SVal;

/// Buffer id of computed-score atoms (attention): `(SCORE_LEAF, hash of the exact polynomial)`.
pub const SCORE_LEAF: u16 = u16::MAX - 4;

/// An exact atom of exp arguments: an input leaf (or a computed score), or the maximum of a set
/// of them.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum Atom {
    In(Leaf),
    Max(std::sync::Arc<Vec<Leaf>>),
}

/// An exact linear form sum_a c_a a + k (k may be -inf).
#[derive(Clone, Debug, PartialEq, Default, Serialize)]
pub struct Lin {
    pub terms: BTreeMap<Atom, f64>,
    pub k: f64,
}

type LinKey = (Vec<(Atom, u64)>, u64);

impl Lin {
    fn key(&self) -> LinKey {
        (
            self.terms
                .iter()
                .filter(|(_, c)| **c != 0.0)
                .map(|(a, c)| (a.clone(), c.to_bits()))
                .collect(),
            self.k.to_bits(),
        )
    }
    fn add(&self, o: &Lin, sign: f64) -> Lin {
        let mut t = self.terms.clone();
        for (a, c) in &o.terms {
            *t.entry(a.clone()).or_default() += sign * c;
        }
        t.retain(|_, c| *c != 0.0);
        Lin {
            terms: t,
            k: self.k + sign * o.k,
        }
    }
    /// Provably <= 0 for every input: x - max(X) with x in X, max(A) - max(B) with A in B, a
    /// constant <= 0, or anything plus -inf.
    fn nonpos(&self) -> bool {
        if self.k == f64::NEG_INFINITY {
            return true;
        }
        if self.k > 0.0 || self.k.is_nan() {
            return false;
        }
        let pos: Vec<&Atom> = self
            .terms
            .iter()
            .filter(|(_, c)| **c == 1.0)
            .map(|x| x.0)
            .collect();
        let neg: Vec<&Atom> = self
            .terms
            .iter()
            .filter(|(_, c)| **c == -1.0)
            .map(|x| x.0)
            .collect();
        if pos.len() + neg.len() != self.terms.len() {
            return false;
        }
        match (pos.as_slice(), neg.as_slice()) {
            ([], []) => true,
            ([Atom::In(x)], [Atom::Max(s)]) => s.contains(x),
            ([Atom::Max(a)], [Atom::Max(b)]) => a.iter().all(|x| b.contains(x)),
            _ => false,
        }
    }
    pub fn show(&self, names: &[String]) -> String {
        let atom = |a: &Atom| match a {
            Atom::In((b, i)) if *b == SCORE_LEAF => format!("score#{i:x}"),
            Atom::In((b, i)) => format!("{}[{i}]", names[*b as usize]),
            Atom::Max(s) => format!("max({} values)", s.len()),
        };
        let mut v: Vec<String> = self
            .terms
            .iter()
            .map(|(a, c)| match *c {
                1.0 => format!("+{}", atom(a)),
                -1.0 => format!("-{}", atom(a)),
                c => format!("{c:+}*{}", atom(a)),
            })
            .collect();
        if self.k != 0.0 {
            v.push(format!("{:+}", self.k));
        }
        v.join(" ")
    }
}

/// Error record of one term c exp(L) of a computed sum: value exp(L) phi + (share of the
/// sum's absolute error), phi in [lo e^(-s|L|), up e^(s|L|)].
#[derive(Clone, Copy, Debug, Serialize)]
pub struct TermRec {
    pub coef: f64,
    pub up: f64,
    pub lo: f64,
    pub slope: f64,
}

/// A computed sum of exponentials: terms by their exact argument, and an absolute error.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ExpSum {
    pub terms: BTreeMap<LinKey, (Lin, TermRec)>,
    pub abs: f64,
}

impl ExpSum {
    /// An upper bound on the computed value's magnitude (each exp(L) <= 1).
    fn mag(&self) -> f64 {
        self.terms.values().map(|(_, r)| r.coef * r.up).sum::<f64>() + self.abs
    }
    fn round(mut self, eps: f64) -> ExpSum {
        for (_, r) in self.terms.values_mut() {
            r.up *= 1.0 + eps;
            r.lo *= 1.0 - eps;
        }
        self.abs *= 1.0 + eps;
        self
    }
    fn plus(mut self, o: &ExpSum) -> ExpSum {
        for (k, (l, r)) in &o.terms {
            match self.terms.get_mut(k) {
                Some((_, x)) => {
                    // Two occurrences of one term: their sum, the wider record.
                    *x = TermRec {
                        coef: x.coef + r.coef,
                        up: x.up.max(r.up),
                        lo: x.lo.min(r.lo),
                        slope: x.slope.max(r.slope),
                    }
                }
                None => {
                    self.terms.insert(k.clone(), (l.clone(), *r));
                }
            }
        }
        self.abs += o.abs;
        self
    }
    fn times(&self, o: &ExpSum) -> ExpSum {
        let mut out = ExpSum::default();
        for (la, ra) in self.terms.values() {
            for (lb, rb) in o.terms.values() {
                let l = la.add(lb, 1.0);
                let r = TermRec {
                    coef: ra.coef * rb.coef,
                    up: ra.up * rb.up,
                    lo: ra.lo * rb.lo,
                    // Both arguments <= 0: |La| + |Lb| = |La + Lb|.
                    slope: ra.slope.max(rb.slope),
                };
                let one = ExpSum {
                    terms: [(l.key(), (l, r))].into(),
                    abs: 0.0,
                };
                out = out.plus(&one);
            }
        }
        out.abs = self.abs * o.mag() + o.abs * self.mag() + self.abs * o.abs;
        out
    }
}

/// A one-argument slice's contract against its function f (exp or 1/x) over a domain:
/// |g(d) - f(d)| <= (r0 + r1 |d|) f(d) where f(d) >= `floor`, and <= a elsewhere.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Contract {
    pub func: &'static str,
    pub r0: f64,
    pub r1: f64,
    pub a: f64,
    pub gmin: f64,
    pub gmax: f64,
    pub values: u64,
}

/// Sweep results by slice key (shared by all outputs and both kernels).
static SWEEPS: Mutex<Option<HashMap<String, Result<Contract, String>>>> = Mutex::new(None);
/// A sweep ran since the cache was loaded (`save_sweeps` writes only then).
static SWEPT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Load sweep results saved by `save_sweeps` (a certificate cache: each entry is the result of
/// the exhaustive evaluation of the slice its key describes; `ptxv equiv --no-sweep-cache`
/// recomputes them).
pub fn load_sweeps(path: &std::path::Path) {
    let Ok(text) = std::fs::read_to_string(path) else {
        return;
    };
    let Ok(v) = serde_json::from_str::<Vec<(String, ContractFile)>>(&text) else {
        return;
    };
    let mut g = SWEEPS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    for (k, c) in v {
        let func = match c.func.as_str() {
            "exp" => "exp",
            "exp2" => "exp2",
            "rcp" => "rcp",
            "rsqrt" => "rsqrt",
            "sqrt" => "sqrt",
            _ => "range",
        };
        // Rounded up: JSON's decimal round trip may lose the last bit.
        let up = 1.0 + 2f64.powi(-48);
        m.entry(k).or_insert(Ok(Contract {
            func,
            r0: c.r0 * up,
            r1: c.r1 * up,
            a: c.a * up,
            gmin: c.gmin,
            gmax: c.gmax,
            values: c.values,
        }));
    }
}

/// Save every successful sweep (sorted by key).
pub fn save_sweeps(path: &std::path::Path) -> std::io::Result<()> {
    if !SWEPT.load(std::sync::atomic::Ordering::Relaxed) {
        return Ok(());
    }
    let g = SWEEPS.lock().unwrap();
    let mut v: Vec<(String, ContractFile)> = g
        .iter()
        .flatten()
        .filter_map(|(k, r)| {
            r.as_ref().ok().map(|c| {
                (
                    k.clone(),
                    ContractFile {
                        func: c.func.into(),
                        r0: c.r0,
                        r1: c.r1,
                        a: c.a,
                        gmin: c.gmin,
                        gmax: c.gmax,
                        values: c.values,
                    },
                )
            })
        })
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    std::fs::write(path, serde_json::to_string_pretty(&v).unwrap())
}

#[derive(Serialize, serde::Deserialize)]
struct ContractFile {
    func: String,
    r0: f64,
    r1: f64,
    a: f64,
    gmin: f64,
    gmax: f64,
    values: u64,
}

/// A key identifying a one-argument slice up to register names.
fn slice_key(p: &SProg, func: &str, dom: (u32, u32)) -> String {
    let mut s = format!("{func} {dom:?} ");
    for o in &p.ops {
        match &o.kind {
            SKind::Inst {
                op,
                dst,
                regs,
                cvals,
            } => {
                // Debug forms: the destination `d: Reg {..}`, value sources `Reg(Reg {..}, n)`,
                // predicates `Some(Reg {..})`.
                let mut t = format!("{op:?}").replace(&format!("d: {dst:?}"), "d: dst");
                for (r, v) in regs {
                    t = t
                        .replace(&format!("Reg({r:?}"), &format!("Reg({v:?}"))
                        .replace(&format!("Some({r:?})"), &format!("Some({v:?})"));
                }
                for (c, v) in cvals {
                    t = t.replace(&format!("{c:?}"), &format!("{v:#x}"));
                }
                s += &t;
            }
            k => s += &format!("{k:?}"),
        }
        s.push(';');
    }
    s + &format!("{:?}", p.out)
}

/// Evaluate a one-argument slice on every f32 in [lo, hi] (`func` also "rsqrt", "sqrt":
/// relative error over the whole domain) (bit patterns of the same sign,
/// walked in order; for negative values lo is the bits of -0 or the smallest magnitude) and fit
/// its contract against `func` ("exp" or "rcp").
pub fn sweep(p: &SProg, func: &'static str, dom: &[(u32, u32)]) -> Result<Contract, String> {
    let key = slice_key(p, func, (dom[0].0, dom[dom.len() - 1].1));
    if let Some(c) = SWEEPS
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .get(&key)
    {
        return c.clone();
    }
    if std::env::var("PTXV_SWEEP_DEBUG").is_ok() {
        eprintln!("sweep (not cached): {}", &key[..key.len().min(200)]);
    }
    let r = sweep_inner(p, func, dom);
    SWEPT.store(true, std::sync::atomic::Ordering::Relaxed);
    SWEEPS
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(key, r.clone());
    r
}

fn sweep_inner(p: &SProg, func: &'static str, dom: &[(u32, u32)]) -> Result<Contract, String> {
    // Work items: chunks of 2^20 bit patterns.
    let mut items = Vec::new();
    for &(lo, hi) in dom {
        let mut a = lo as u64;
        while a <= hi as u64 {
            let b = (a + (1 << 20) - 1).min(hi as u64);
            items.push((a as u32, b as u32));
            a = b + 1;
        }
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism()
        .map_or(8, |n| n.get())
        .min(128);
    // Per thread: (r0 over |d| <= 1, (rel - r0') candidates as (|d|, rel), a, gmin, gmax, n, err)
    type ThreadOut = (f64, Vec<(f64, f64)>, f64, f64, f64, u64);
    let results: Vec<Result<ThreadOut, String>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut vals = Vec::new();
                    let (mut r0, mut a, mut gmin, mut gmax, mut n) =
                        (0.0f64, 0.0f64, f64::INFINITY, 0.0f64, 0u64);
                    // Worst relative error per binade of |d| beyond 1 (for the slope).
                    // Per binade of |d| beyond 1: (smallest |d|, largest relative error).
                    let mut far: BTreeMap<i32, (f64, f64)> = BTreeMap::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&(lo, hi)) = items.get(i) else { break };
                        for bits in lo..=hi {
                            let g = f32::from_bits(p.eval_with(&[bits], &mut vals)?) as f64;
                            let d = f32::from_bits(bits) as f64;
                            if !g.is_finite() {
                                return Err(format!("not finite at {d}"));
                            }
                            n += 1;
                            gmin = gmin.min(g);
                            gmax = gmax.max(g);
                            let (f, rel_ok) = match func {
                                // e^d >= 2^-126 for d >= -87.
                                "exp" => (d.exp(), d >= -87.0),
                                "exp2" => (d.exp2(), d >= -126.0),
                                "rcp" => (1.0 / d, true),
                                "rsqrt" => (1.0 / d.sqrt(), true),
                                "sqrt" => (d.sqrt(), true),
                                // "range": only finiteness and the extremes.
                                _ => (g, true),
                            };
                            let err = (g - f).abs();
                            if rel_ok {
                                let rel = err / f;
                                if d.abs() <= 1.0 || matches!(func, "rcp" | "rsqrt" | "sqrt") {
                                    r0 = r0.max(rel);
                                } else {
                                    let e = d.abs().log2().floor() as i32;
                                    let x = far.entry(e).or_insert((f64::INFINITY, 0.0));
                                    x.0 = x.0.min(d.abs());
                                    x.1 = x.1.max(rel);
                                }
                            } else {
                                // f < 2^-126: absolute. The exact e^L is within a factor
                                // e^(87u) of e^d here (|d - L| <= u |L|): 2^-10 covers it.
                                a = a.max((err + f) * (1.0 + 2f64.powi(-10)));
                            }
                        }
                    }
                    Ok((r0, far.into_values().collect(), a, gmin, gmax, n))
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (mut r0, mut a, mut gmin, mut gmax, mut n) = (0.0f64, 0.0f64, f64::INFINITY, 0.0f64, 0u64);
    let mut far = Vec::new();
    for r in results {
        let (x0, f, xa, mi, ma, k) = r?;
        r0 = r0.max(x0);
        far.extend(f);
        a = a.max(xa);
        gmin = gmin.min(mi);
        gmax = gmax.max(ma);
        n += k;
    }
    // Slope: per binade the worst rel at that binade's smallest |d| (a lower bound on |d| in
    // the binade), so rel <= r0 + r1 |d| holds for every point of the binade.
    let r1 = far
        .iter()
        .map(|&(d, rel)| ((rel - r0) / d).max(0.0))
        .fold(0.0, f64::max);
    // f64 references: exp and 1/x are within 2^-52 relative.
    let slack = 2f64.powi(-50);
    Ok(Contract {
        func,
        r0: r0 + slack,
        r1,
        a: a * (1.0 + slack),
        gmin,
        gmax,
        values: n,
    })
}

/// The exp contract of a slice of one argument `d <= 0` (every f32 from -0 to -inf, and +0).
pub fn exp_contract(p: &SProg) -> Result<Contract, String> {
    let c = sweep(p, "exp", &[(0, 0), (0x8000_0000, 0xff80_0000)])?;
    if c.r0 > 2f64.powi(-10) || c.gmin < 0.0 {
        return Err(format!(
            "slice is not exp on d <= 0 (relative error {:.3e} on [-1, 0], minimum {})",
            c.r0, c.gmin
        ));
    }
    Ok(c)
}

/// The 2^d contract of a slice of one argument d <= 2^-7 (every f32 from -inf to 2^-7: the
/// shifted form's arguments may exceed 0 by the shift, at most 2^-8).
pub fn exp2_contract(p: &SProg) -> Result<Contract, String> {
    let c = sweep(p, "exp2", &[(0, 0x3c00_0000), (0x8000_0000, 0xff80_0000)])?;
    if c.r0 > 2f64.powi(-10) || c.gmin < 0.0 {
        return Err(format!(
            "slice is not 2^d on d <= 2^-7 (relative error {:.3e} on [-1, 2^-7], minimum {})",
            c.r0, c.gmin
        ));
    }
    Ok(c)
}

/// Is a one-argument slice finite on every finite f32 argument? (A product of it with an exact
/// zero is then exactly zero.)
pub fn finite_everywhere(p: &SProg) -> Result<(), String> {
    sweep(p, "range", &[(0, 0x7f7f_ffff), (0x8000_0000, 0xff7f_ffff)]).map(|_| ())
}

/// The 1/x contract of a slice on [lo, hi] (0 < lo).
pub fn rcp_contract(p: &SProg, lo: f64, hi: f64) -> Result<Contract, String> {
    let l = ((lo as f32).to_bits()).saturating_sub(1);
    let h = (hi as f32).to_bits() + 1;
    let c = sweep(p, "rcp", &[(l, h)])?;
    if c.r0 > 2f64.powi(-10) {
        return Err(format!(
            "slice is not 1/x on [{lo}, {hi}] (relative error {:.3e})",
            c.r0
        ));
    }
    Ok(c)
}

// ------------------------------------------------------------------ slice surgery

/// Which arguments each op depends on (bit i: argument i).
fn deps(p: &SProg) -> Vec<u64> {
    let mut d = vec![0u64; p.ops.len()];
    let of = |v: &SVal, d: &[u64]| match v {
        SVal::Arg(i) => 1u64 << i,
        SVal::Const(_) => 0,
        SVal::Op(i) => d[*i as usize],
    };
    for (i, o) in p.ops.iter().enumerate() {
        d[i] = srcs(o).iter().map(|v| of(v, &d)).fold(0, |a, b| a | b);
    }
    d
}

fn dep_of(p: &SProg, d: &[u64], v: SVal) -> u64 {
    let _ = p;
    match v {
        SVal::Arg(i) => 1 << i,
        SVal::Const(_) => 0,
        SVal::Op(i) => d[i as usize],
    }
}

fn srcs(o: &SOp) -> Vec<SVal> {
    match &o.kind {
        SKind::Inst { regs, .. } => regs.iter().map(|x| x.1).collect(),
        SKind::Guard { pred, new, old, .. } => vec![*pred, *new, *old],
        SKind::DivRn { a, b } => vec![*a, *b],
    }
}

fn map_srcs(o: &SOp, f: &dyn Fn(SVal) -> SVal) -> SOp {
    let kind = match &o.kind {
        SKind::Inst {
            op,
            dst,
            regs,
            cvals,
        } => SKind::Inst {
            op: op.clone(),
            dst: *dst,
            regs: regs.iter().map(|(r, v)| (*r, f(*v))).collect(),
            cvals: cvals.clone(),
        },
        SKind::Guard {
            pred,
            neg,
            new,
            old,
        } => SKind::Guard {
            pred: f(*pred),
            neg: *neg,
            new: f(*new),
            old: f(*old),
        },
        SKind::DivRn { a, b } => SKind::DivRn { a: f(*a), b: f(*b) },
    };
    SOp { pc: o.pc, kind }
}

/// Decide the predicates that compare argument `k` (with |x| / -x) against a constant over
/// the argument's range [lo, hi], and drop the guards and selects they decide.
fn simplify(p: &SProg, k: u16, lo: f64, hi: f64) -> SProg {
    let (l32, h32) = ((lo as f32).next_down_f32(), (hi as f32).next_up_f32());
    let mut alias: Vec<SVal> = Vec::with_capacity(p.ops.len());
    let mut ops: Vec<SOp> = Vec::new();
    for o in &p.ops {
        let o = map_srcs(o, &|v| match v {
            SVal::Op(i) => alias[i as usize],
            v => v,
        });
        let decided = match &o.kind {
            SKind::Guard {
                pred: SVal::Const(c),
                neg,
                new,
                old,
            } => Some(if (*c != 0) != *neg { *new } else { *old }),
            SKind::Inst {
                op:
                    Op::Fsetp {
                        cmp,
                        a,
                        b,
                        neg,
                        abs,
                        c,
                        bop: BoolOp::And,
                        ftz,
                        ..
                    },
                regs,
                ..
            } if c.reg.is_none() && !c.neg => {
                // Each operand: argument k or a constant.
                let side = |v: &Val| -> Option<Option<u32>> {
                    match v {
                        Val::Reg(r, _) => match regs.iter().find(|x| x.0 == *r)?.1 {
                            SVal::Arg(i) if i == k => Some(None),
                            SVal::Const(c) => Some(Some(c as u32)),
                            _ => None,
                        },
                        Val::Imm(c) => Some(Some(*c as u32)),
                        _ => None,
                    }
                };
                match (side(a), side(b)) {
                    (Some(sa), Some(sb)) if sa.is_none() || sb.is_none() => {
                        let at = |x: f32| {
                            let m = |s: Option<u32>, i: usize| {
                                let u = s.unwrap_or(x.to_bits());
                                let u = if abs[i] { u & 0x7fff_ffff } else { u };
                                if neg[i] { u ^ 0x8000_0000 } else { u }
                            };
                            crate::fp::f32_cmp(m(sa, 0), m(sb, 1), *cmp, *ftz)
                        };
                        // Monotone threshold in x > 0 (lo > 0): equal at both ends -> constant.
                        (lo > 0.0 && at(l32) == at(h32)).then(|| SVal::Const(at(l32) as u64))
                    }
                    _ => None,
                }
            }
            SKind::Inst {
                op: Op::Sel { a, b, p: pr, .. },
                regs,
                ..
            } => {
                let pv = pr
                    .reg
                    .and_then(|r| regs.iter().find(|x| x.0 == r))
                    .map(|x| x.1);
                let val = |v: &Val| match v {
                    Val::Reg(r, _) => regs.iter().find(|x| x.0 == *r).map(|x| x.1),
                    Val::Imm(c) => Some(SVal::Const(*c & 0xffff_ffff)),
                    _ => None,
                };
                match pv {
                    Some(SVal::Const(c)) => {
                        if (c != 0) != pr.neg {
                            val(a)
                        } else {
                            val(b)
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        match decided {
            Some(v) => alias.push(v),
            None => {
                ops.push(o);
                alias.push(SVal::Op(ops.len() as u32 - 1));
            }
        }
    }
    let out = match p.out {
        SVal::Op(i) => alias[i as usize],
        v => v,
    };
    SProg {
        args: p.args.clone(),
        ops,
        out,
        extra_pcs: p.extra_pcs.clone(),
    }
}

/// Decide the predicates that compare an argument with +-inf (finite arguments: the result is
/// the same for every finite value) and drop the selects they decide.
fn simplify_inf(p: &SProg) -> SProg {
    let mut q = p.clone();
    for k in 0..p.args.len() as u16 {
        let mut ops = Vec::new();
        let mut alias: Vec<SVal> = Vec::new();
        for o in &q.ops {
            let o = map_srcs(o, &|v| match v {
                SVal::Op(i) => alias[i as usize],
                v => v,
            });
            let decided = match &o.kind {
                SKind::Inst {
                    op:
                        Op::Fsetp {
                            cmp,
                            a,
                            b,
                            neg,
                            abs,
                            c,
                            bop: BoolOp::And,
                            ftz,
                            ..
                        },
                    regs,
                    ..
                } if c.reg.is_none() && !c.neg => {
                    let side = |v: &Val| -> Option<Option<u32>> {
                        match v {
                            Val::Reg(r, _) => match regs.iter().find(|x| x.0 == *r)?.1 {
                                SVal::Arg(i) if i == k => Some(None),
                                SVal::Const(c) => Some(Some(c as u32)),
                                _ => None,
                            },
                            Val::Imm(c) => Some(Some(*c as u32)),
                            _ => None,
                        }
                    };
                    let inf = |c: u32| c & 0x7fff_ffff == 0x7f80_0000;
                    match (side(a), side(b)) {
                        (Some(sa), Some(sb))
                            if (sa.is_none() && sb.is_some_and(inf))
                                || (sb.is_none() && sa.is_some_and(inf)) =>
                        {
                            let at = |x: u32| {
                                let m = |s: Option<u32>, i: usize| {
                                    let u = s.unwrap_or(x);
                                    let u = if abs[i] { u & 0x7fff_ffff } else { u };
                                    if neg[i] { u ^ 0x8000_0000 } else { u }
                                };
                                crate::fp::f32_cmp(m(sa, 0), m(sb, 1), *cmp, *ftz)
                            };
                            // The comparison with an infinity is the same for every finite x.
                            let v = [0xff7f_ffffu32, 0x8000_0000, 0, 0x7f7f_ffff].map(at);
                            v.iter()
                                .all(|&y| y == v[0])
                                .then(|| SVal::Const(v[0] as u64))
                        }
                        _ => None,
                    }
                }
                SKind::Inst {
                    op: Op::Sel { a, b, p: pr, .. },
                    regs,
                    ..
                } => {
                    let pv = pr
                        .reg
                        .and_then(|r| regs.iter().find(|x| x.0 == r))
                        .map(|x| x.1);
                    let val = |v: &Val| match v {
                        Val::Reg(r, _) => regs.iter().find(|x| x.0 == *r).map(|x| x.1),
                        Val::Imm(c) => Some(SVal::Const(*c & 0xffff_ffff)),
                        _ => None,
                    };
                    match pv {
                        Some(SVal::Const(c)) => {
                            if (c != 0) != pr.neg {
                                val(a)
                            } else {
                                val(b)
                            }
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            match decided {
                Some(v) => alias.push(v),
                None => {
                    ops.push(o);
                    alias.push(SVal::Op(ops.len() as u32 - 1));
                }
            }
        }
        let out = match q.out {
            SVal::Op(i) => alias[i as usize],
            v => v,
        };
        q = SProg {
            args: q.args.clone(),
            ops,
            out,
            extra_pcs: q.extra_pcs.clone(),
        };
    }
    // Ops no longer reached from the output (a decided predicate) are dropped.
    let mut keep = vec![false; q.ops.len()];
    let mut st = vec![q.out];
    while let Some(v) = st.pop() {
        if let SVal::Op(i) = v {
            if !keep[i as usize] {
                keep[i as usize] = true;
                st.extend(srcs(&q.ops[i as usize]));
            }
        }
    }
    let mut idx = vec![0u32; q.ops.len()];
    let mut ops = Vec::new();
    for (i, o) in q.ops.iter().enumerate() {
        if keep[i] {
            ops.push(map_srcs(o, &|v| match v {
                SVal::Op(j) => SVal::Op(idx[j as usize]),
                v => v,
            }));
            idx[i] = ops.len() as u32 - 1;
        }
    }
    let out = match q.out {
        SVal::Op(j) => SVal::Op(idx[j as usize]),
        v => v,
    };
    SProg {
        args: q.args,
        ops,
        out,
        extra_pcs: q.extra_pcs,
    }
}

/// The c of a one-op slice fl(Arg0 * c) (round to nearest, no modifiers, c > 0).
fn fmul_const_prog(p: &SProg) -> Option<f64> {
    if p.ops.len() != 1 || p.out != SVal::Op(0) {
        return None;
    }
    let SKind::Inst {
        op:
            Op::Ffma {
                a,
                b,
                c: None,
                neg: [false, false, false],
                abs: [false, false, false],
                rnd: Rnd::Rn,
                ftz: false,
                sat: false,
                ..
            },
        regs,
        cvals,
        ..
    } = &p.ops[0].kind
    else {
        return None;
    };
    let val = |v: &Val| -> Option<SVal> {
        match v {
            Val::Reg(r, _) => regs.iter().find(|x| x.0 == *r).map(|x| x.1),
            Val::Imm(c) => Some(SVal::Const(*c & 0xffff_ffff)),
            v => cvals
                .iter()
                .find(|x| x.0 == *v)
                .map(|x| SVal::Const(x.1 & 0xffff_ffff)),
        }
    };
    match (val(a)?, val(b)?) {
        (SVal::Arg(0), SVal::Const(c)) | (SVal::Const(c), SVal::Arg(0)) => {
            let c = f32::from_bits(c as u32) as f64;
            (c > 0.0 && c.is_finite()).then_some(c)
        }
        _ => None,
    }
}

/// Rewrite comparisons a != -b (or ==) of two arguments into S != 0 when another argument is
/// the node S = fl(a + b) (round to nearest, no FTZ): for finite a, b, fl(a + b) = 0 exactly
/// when a = -b (a nonzero exact sum is a multiple of 2^-149, never rounded to 0).
fn fold_sum_compare(p: &SProg, ns: &NumState) -> SProg {
    let mut q = p.clone();
    for o in q.ops.iter_mut() {
        let SKind::Inst {
            op:
                Op::Fsetp {
                    cmp,
                    a: Val::Reg(ra, _),
                    b: Val::Reg(rb, _),
                    neg,
                    abs: [false, false],
                    ftz: false,
                    ..
                },
            regs,
            ..
        } = &mut o.kind
        else {
            continue;
        };
        if !matches!(cmp, FCmp::Ne | FCmp::Neu | FCmp::Eq | FCmp::Equ) || neg[0] == neg[1] {
            continue;
        }
        let arg = |r: &Reg, regs: &Vec<(Reg, SVal)>| match regs.iter().find(|x| x.0 == *r) {
            Some((_, SVal::Arg(i))) => Some(*i),
            _ => None,
        };
        let (Some(i), Some(j)) = (arg(ra, regs), arg(rb, regs)) else {
            continue;
        };
        let (ni, nj) = (p.args[i as usize], p.args[j as usize]);
        let Some(k) = p.args.iter().position(|&s| {
            matches!(ns.nodes[s as usize].kind, NodeKind::Add(x, y, false) if (x, y) == (ni, nj) || (x, y) == (nj, ni))
        }) else {
            continue;
        };
        let (ra, rb) = (*ra, *rb);
        for (r, v) in regs.iter_mut() {
            if *r == ra {
                *v = SVal::Arg(k as u16);
            } else if *r == rb {
                *v = SVal::Const(0);
            }
        }
    }
    q
}

/// Drop operations the output no longer reads.
fn prune(q: &SProg) -> SProg {
    let mut keep = vec![false; q.ops.len()];
    let mut st = vec![q.out];
    while let Some(v) = st.pop() {
        if let SVal::Op(i) = v {
            if !keep[i as usize] {
                keep[i as usize] = true;
                st.extend(srcs(&q.ops[i as usize]));
            }
        }
    }
    let mut idx = vec![0u32; q.ops.len()];
    let mut ops = Vec::new();
    for (i, o) in q.ops.iter().enumerate() {
        if keep[i] {
            ops.push(map_srcs(o, &|v| match v {
                SVal::Op(j) => SVal::Op(idx[j as usize]),
                v => v,
            }));
            idx[i] = ops.len() as u32 - 1;
        }
    }
    let out = match q.out {
        SVal::Op(j) => SVal::Op(idx[j as usize]),
        v => v,
    };
    SProg {
        args: q.args.clone(),
        ops,
        out,
        extra_pcs: q.extra_pcs.clone(),
    }
}

trait NextF32 {
    fn next_down_f32(self) -> f32;
    fn next_up_f32(self) -> f32;
}
impl NextF32 for f32 {
    fn next_down_f32(self) -> f32 {
        f32::from_bits(self.to_bits() - 1)
    }
    fn next_up_f32(self) -> f32 {
        f32::from_bits(self.to_bits() + 1)
    }
}

/// The sub-slice computing `root`, which depends on argument `k` only, as a one-argument slice.
fn sub(p: &SProg, root: SVal, k: u16) -> SProg {
    let mut keep = vec![false; p.ops.len()];
    let mut stack = vec![root];
    while let Some(v) = stack.pop() {
        if let SVal::Op(i) = v {
            if !keep[i as usize] {
                keep[i as usize] = true;
                stack.extend(srcs(&p.ops[i as usize]));
            }
        }
    }
    let mut idx = vec![0u32; p.ops.len()];
    let mut ops = Vec::new();
    for (i, o) in p.ops.iter().enumerate() {
        if keep[i] {
            let o = map_srcs(o, &|v| match v {
                SVal::Op(j) => SVal::Op(idx[j as usize]),
                SVal::Arg(a) if a == k => SVal::Arg(0),
                v => v,
            });
            idx[i] = ops.len() as u32;
            ops.push(o);
        }
    }
    let out = match root {
        SVal::Op(j) => SVal::Op(idx[j as usize]),
        SVal::Arg(_) => SVal::Arg(0),
        v => v,
    };
    SProg {
        args: vec![p.args[k as usize]],
        ops,
        out,
        extra_pcs: BTreeSet::new(),
    }
}

// ------------------------------------------------------------------ forms of nodes

/// One kernel's numerics, with the buffer-id map of `equiv::Side`.
pub struct Ctx<'a> {
    pub ns: &'a NumState,
    pub buf_id: &'a [u16],
    sums: HashMap<u32, Result<ExpSum, String>>,
    lins: HashMap<u32, Result<(Lin, bool), String>>,
    wsums: HashMap<u32, Result<WSum, String>>,
    hashes: HashMap<u32, u64>,
    /// Computed scores used as exp-argument atoms (attention: q . k_j): by the hash of their
    /// exact polynomial, (structural hash, polynomial with Cor 1a multipliers).
    pub scores: HashMap<u64, (u64, BTreeMap<crate::equiv::Mono, crate::equiv::Term>)>,
    polys: crate::norm::Ctx<'a>,
    plain: HashMap<u32, bool>,
    /// Scaled scores (attention: c q.k_j, the softmax scale folded into an exp2 argument):
    /// some exp argument uses the unrounded c s_j (inside an FFMA) while the maxima are over
    /// the rounded fl(c s_k): the computed maximum then differs from the exact one by at most
    /// u max_k |c s_k| (`SHIFT_X`), a factor common to every term (N-ATTN, "shift").
    pub shifted: bool,
    /// A second structural hash accepted for a score atom (shifted kernels).
    score_alias: HashMap<u64, u64>,
    /// Per score atom used as the positive atom of an exp argument: rounded (fl(c s)) or not.
    term_rounded: HashMap<u64, bool>,
    /// The exp slices are base-2 (`exp2` contracts; arguments in log2 units).
    pub base2: Option<bool>,
}

/// Validity limit of the shifted form: every |c s_j| <= SHIFT_X (so the shift is <= u 2^16
/// = 2^-8 and the exp2 slices are swept up to 2^-7).
pub const SHIFT_X: f64 = 65536.0;

/// One output in normal form: exp(num) / den, with the division's relative error.
#[derive(Clone, Debug, Serialize)]
pub struct Quot {
    pub num: (Lin, TermRec),
    pub num_abs: f64,
    pub den: ExpSum,
    pub rho_up: f64,
    pub rho_lo: f64,
    pub div_abs: f64,
    /// Members of the maximum (the row): n.
    pub n: usize,
    pub members: Vec<Leaf>,
    pub contracts: Vec<Contract>,
    /// Shifted maxima (`Ctx::shifted`) and base-2 exponentials (`Ctx::base2`).
    pub shifted: bool,
    pub base2: bool,
}

impl<'a> Ctx<'a> {
    pub fn new(ns: &'a NumState, buf_id: &'a [u16]) -> Ctx<'a> {
        Ctx {
            ns,
            buf_id,
            sums: HashMap::new(),
            lins: HashMap::new(),
            wsums: HashMap::new(),
            hashes: HashMap::new(),
            scores: HashMap::new(),
            polys: crate::norm::Ctx::new(ns, buf_id),
            plain: HashMap::new(),
            shifted: false,
            score_alias: HashMap::new(),
            term_rounded: HashMap::new(),
            base2: None,
        }
    }

    /// fl(c x) of node x by a positive constant c (round to nearest, no modifiers): an FMUL
    /// node with a constant leaf, or a one-argument slice of one such FMUL. (x, c).
    fn fmul_const(&self, n: u32) -> Option<(u32, f64)> {
        let pos = |c: f64| (c > 0.0 && c.is_finite()).then_some(c);
        match self.kind(n) {
            NodeKind::Fma {
                a,
                b,
                c: None,
                neg_ab: false,
                ..
            } => match (self.konst(*a), self.konst(*b)) {
                (None, Some(c)) => Some((*a, pos(c)?)),
                (Some(c), None) => Some((*b, pos(c)?)),
                _ => None,
            },
            NodeKind::Seg(p)
                if p.0.args.len() == 1 && p.0.ops.len() == 1 && p.0.out == SVal::Op(0) =>
            {
                let SKind::Inst {
                    op:
                        Op::Ffma {
                            a,
                            b,
                            c: None,
                            neg: [false, false, false],
                            abs: [false, false, false],
                            rnd: Rnd::Rn,
                            ftz: false,
                            sat: false,
                            ..
                        },
                    regs,
                    cvals,
                    ..
                } = &p.0.ops[0].kind
                else {
                    return None;
                };
                let val = |v: &Val| -> Option<SVal> {
                    match v {
                        Val::Reg(r, _) => regs.iter().find(|x| x.0 == *r).map(|x| x.1),
                        Val::Imm(c) => Some(SVal::Const(*c & 0xffff_ffff)),
                        v => cvals
                            .iter()
                            .find(|x| x.0 == *v)
                            .map(|x| SVal::Const(x.1 & 0xffff_ffff)),
                    }
                };
                match (val(a)?, val(b)?) {
                    (SVal::Arg(0), SVal::Const(c)) | (SVal::Const(c), SVal::Arg(0)) => {
                        Some((p.0.args[0], pos(f32::from_bits(c as u32) as f64)?))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Register a computed score atom with exact polynomial `p` (its Cor 1a multipliers for
    /// the value the softmax uses) and structural hash `sh`; `unrounded`: the value is the
    /// exact c x of a rounded score x (inside an FFMA), where other uses round it (fl(c x),
    /// structural hash `sh_rounded`): the kernel is then `shifted`.
    fn register_score(
        &mut self,
        p: BTreeMap<crate::equiv::Mono, crate::equiv::Term>,
        sh: u64,
        alt: Option<(u64, bool)>,
    ) -> Result<Leaf, String> {
        use std::hash::Hash;
        use std::hash::Hasher;
        if p.keys().all(|m| m.is_empty()) {
            return Err("a constant".into());
        }
        let mut st = std::collections::hash_map::DefaultHasher::new();
        for (m, t) in &p {
            if t.coef != 0.0 {
                (m, t.coef.to_bits()).hash(&mut st);
            }
        }
        let key = st.finish() & 0x7fff_ffff_ffff_ffff;
        match self.scores.get(&key).cloned() {
            Some((h, _)) if h == sh || self.score_alias.get(&key) == Some(&sh) => {
                if alt.is_some_and(|a| a.1) {
                    self.scores.insert(key, (h, p));
                }
            }
            Some((h, _)) => match alt {
                // The unrounded and the rounded c x of the same x: the kernel is shifted;
                // the atom's polynomial is the unrounded value's (the exp terms use it).
                Some((other, unrounded)) if other == h => {
                    self.shifted = true;
                    self.score_alias.insert(key, sh);
                    if unrounded {
                        self.scores.insert(key, (h, p));
                    }
                }
                _ => return Err("one score computed with two different roundings".into()),
            },
            None => {
                self.scores.insert(key, (sh, p));
            }
        }
        Ok((SCORE_LEAF, key))
    }

    /// The scaled-score atom c x of score node x: `rounded` fl(c x), else the exact c x.
    fn scaled_score(&mut self, x: u32, c: f64, rounded: bool) -> Result<Leaf, String> {
        use std::hash::Hash;
        use std::hash::Hasher;
        if !self.input_poly(x) || matches!(self.kind(x), NodeKind::Input(_) | NodeKind::Leaf) {
            return Err("not a computed polynomial of the inputs".into());
        }
        let px = self.polys.poly(x)?;
        let mut p: BTreeMap<crate::equiv::Mono, crate::equiv::Term> = px
            .iter()
            .map(|(m, t)| {
                (
                    m.clone(),
                    crate::equiv::Term {
                        coef: t.coef * c,
                        mult: t.mult * c,
                        abs: t.abs * c,
                        gam: t.gam * c,
                    },
                )
            })
            .collect();
        if rounded {
            let mut q = BTreeMap::new();
            crate::equiv::add_poly_n(&mut q, &p, U32, 1);
            p = q;
        }
        let hx = self.shash(x);
        let h = |tag: &str| {
            let mut st = std::collections::hash_map::DefaultHasher::new();
            (tag, hx, c.to_bits()).hash(&mut st);
            st.finish()
        };
        let (sh, other) = if rounded {
            (h("fmulc"), h("fmac"))
        } else {
            (h("fmac"), h("fmulc"))
        };
        self.register_score(p, sh, Some((other, !rounded)))
    }

    fn kind(&self, n: u32) -> &NodeKind {
        &self.ns.nodes[n as usize].kind
    }

    fn konst(&self, n: u32) -> Option<f64> {
        match self.kind(n) {
            NodeKind::Leaf => Some(f32::from_bits(self.ns.nodes[n as usize].hw) as f64),
            _ => None,
        }
    }

    /// The exact linear form of node `n` and whether it was rounded once (an FADD of exact
    /// atoms: |d^ - L| <= u |L|).
    pub fn lin(&mut self, n: u32) -> Result<(Lin, bool), String> {
        if let Some(r) = self.lins.get(&n) {
            return r.clone();
        }
        let r = self.lin_inner(n);
        self.lins.insert(n, r.clone());
        r
    }

    fn lin_inner(&mut self, n: u32) -> Result<(Lin, bool), String> {
        match self.kind(n).clone() {
            NodeKind::Input((a, i)) => Ok((
                Lin {
                    terms: [(Atom::In((self.buf_id[a], i)), 1.0)].into(),
                    k: 0.0,
                },
                false,
            )),
            NodeKind::Leaf => Ok((
                Lin {
                    terms: BTreeMap::new(),
                    k: self.konst(n).unwrap(),
                },
                false,
            )),
            NodeKind::Max { min: false, .. } => {
                let mut set = BTreeSet::new();
                let mut stack = vec![n];
                while let Some(x) = stack.pop() {
                    match self.kind(x).clone() {
                        NodeKind::Max { a, b, min: false } => {
                            stack.push(a);
                            stack.push(b);
                        }
                        NodeKind::Input((a, i)) => {
                            set.insert((self.buf_id[a], i));
                        }
                        NodeKind::Leaf if self.ns.nodes[x as usize].hw == 0xff80_0000 => {}
                        k => match self.score(x) {
                            Ok(l) => {
                                set.insert(l);
                            }
                            Err(_) => return Err(format!("a maximum of a non-input ({k:?})")),
                        },
                    }
                }
                let set: Vec<Leaf> = set.into_iter().collect();
                let atom = if set.len() == 1 {
                    Atom::In(set[0])
                } else {
                    Atom::Max(set.into())
                };
                Ok((
                    Lin {
                        terms: [(atom, 1.0)].into(),
                        k: 0.0,
                    },
                    false,
                ))
            }
            NodeKind::Add(a, b, false) => self
                .lin_sum(a, false, b, false)
                .or_else(|e| self.score_lin(n).map_err(|_| e)),
            NodeKind::Fma {
                a,
                b,
                c: Some(c),
                neg_ab,
                neg_c,
            } if self.konst(b) == Some(1.0) => self
                .lin_sum(a, neg_ab, c, neg_c)
                .or_else(|e| self.score_lin(n).map_err(|_| e)),
            // FFMA(s, c, -m): the scaled score c s (unrounded) minus an exact linear form,
            // rounded once (attention with the softmax scale folded into an exp2 argument).
            NodeKind::Fma {
                a,
                b,
                c: Some(m),
                neg_ab: false,
                neg_c,
            } if self.unrounded_scaled(a, b).is_some() => {
                let (x, k) = self.unrounded_scaled(a, b).unwrap();
                let atom = self.scaled_score(x, k, false)?;
                let (lm, rm) = self.lin(m)?;
                if rm {
                    return Err("an exp argument rounded twice".into());
                }
                let mut l = Lin {
                    terms: [(Atom::In(atom), 1.0)].into(),
                    k: 0.0,
                }
                .add(&lm, if neg_c { -1.0 } else { 1.0 });
                let exact = l.k == f64::NEG_INFINITY;
                if exact {
                    l.terms.clear();
                }
                Ok((l, !exact))
            }
            // fl(c max(S)) of scores: max_k fl(c s_k) (rounding to nearest is monotone, c > 0).
            NodeKind::Seg(p)
                if p.0.args.len() == 1
                    && matches!(self.kind(p.0.args[0]), NodeKind::Max { min: false, .. }) =>
            {
                let q = simplify_inf(&p.0);
                let c = fmul_const_prog(&q).ok_or("a slice of a maximum that is not fl(c max)")?;
                let mut set = BTreeSet::new();
                let mut stack = vec![p.0.args[0]];
                while let Some(x) = stack.pop() {
                    match self.kind(x).clone() {
                        NodeKind::Max { a, b, min: false } => {
                            stack.push(a);
                            stack.push(b);
                        }
                        NodeKind::Leaf if self.ns.nodes[x as usize].hw == 0xff80_0000 => {}
                        _ => {
                            set.insert(self.scaled_score(x, c, true)?);
                        }
                    }
                }
                let set: Vec<Leaf> = set.into_iter().collect();
                let atom = if set.len() == 1 {
                    Atom::In(set[0])
                } else {
                    Atom::Max(set.into())
                };
                Ok((
                    Lin {
                        terms: [(atom, 1.0)].into(),
                        k: 0.0,
                    },
                    false,
                ))
            }
            k => self
                .score_lin(n)
                .map_err(|_| format!("not a linear form: {}", kind_name(&k))),
        }
    }

    /// (score node, c) when a * b is a score times a positive constant.
    fn unrounded_scaled(&mut self, a: u32, b: u32) -> Option<(u32, f64)> {
        for (x, k) in [(a, b), (b, a)] {
            if let Some(c) = self.konst(k) {
                if c > 0.0
                    && c != 1.0
                    && c.is_finite()
                    && !matches!(self.kind(x), NodeKind::Input(_) | NodeKind::Leaf)
                    && self.input_poly(x)
                {
                    return Some((x, c));
                }
            }
        }
        None
    }

    /// A computed score as an exact atom (its computed value: the kernel's softmax is exact in
    /// it, N-ATTN).
    fn score_lin(&mut self, n: u32) -> Result<(Lin, bool), String> {
        let l = self.score(n)?;
        Ok((
            Lin {
                terms: [(Atom::In(l), 1.0)].into(),
                k: 0.0,
            },
            false,
        ))
    }

    /// Does node `n` compute a rounded polynomial of input leaves (sums and products only)?
    fn input_poly(&mut self, n: u32) -> bool {
        if let Some(&b) = self.plain.get(&n) {
            return b;
        }
        let b = match self.kind(n).clone() {
            NodeKind::Input(_) | NodeKind::Leaf => true,
            NodeKind::Add(a, b, _) => self.input_poly(a) && self.input_poly(b),
            NodeKind::Fma { a, b, c, .. } => {
                self.input_poly(a) && self.input_poly(b) && c.is_none_or(|c| self.input_poly(c))
            }
            // A tensor-core step of input elements (q . k_j).
            NodeKind::Mma { acc, ops, sc, .. } => {
                sc.is_empty()
                    && acc.is_none_or(|a| self.input_poly(a))
                    && ops
                        .iter()
                        .all(|o| o.is_some_and(|(x, y)| self.input_poly(x) && self.input_poly(y)))
            }
            _ => false,
        };
        self.plain.insert(n, b);
        b
    }

    /// The atom of a computed score: (SCORE_LEAF, hash of its exact polynomial). Every node of
    /// one kernel with that polynomial must round alike (identical DAGs), so the atom is one
    /// value.
    fn score(&mut self, n: u32) -> Result<Leaf, String> {
        use std::hash::Hash;
        use std::hash::Hasher;
        // fl(c s) of a score s: a scaled-score atom.
        if let Some((x, c)) = self.fmul_const(n) {
            if !matches!(self.kind(x), NodeKind::Input(_) | NodeKind::Leaf) && self.input_poly(x) {
                return self.scaled_score(x, c, true);
            }
        }
        if matches!(self.kind(n), NodeKind::Input(_) | NodeKind::Leaf) || !self.input_poly(n) {
            return Err("not a computed polynomial of the inputs".into());
        }
        let p = self.polys.poly(n)?;
        if p.keys().all(|m| m.is_empty()) {
            return Err("a constant".into());
        }
        let mut st = std::collections::hash_map::DefaultHasher::new();
        for (m, t) in &p {
            if t.coef != 0.0 {
                (m, t.coef.to_bits()).hash(&mut st);
            }
        }
        let key = st.finish() & 0x7fff_ffff_ffff_ffff;
        let sh = self.shash(n);
        match self.scores.get(&key) {
            Some((h, _)) if *h != sh => {
                return Err("one score computed with two different roundings".into());
            }
            Some(_) => {}
            None => {
                self.scores.insert(key, (sh, p));
            }
        }
        Ok((SCORE_LEAF, key))
    }

    fn lin_sum(&mut self, a: u32, na: bool, b: u32, nb: bool) -> Result<(Lin, bool), String> {
        let (la, ra) = self.lin(a)?;
        let (lb, rb) = self.lin(b)?;
        if ra || rb {
            return Err("an exp argument rounded twice".into());
        }
        let sa = if na { -1.0 } else { 1.0 };
        let sb = if nb { -1.0 } else { 1.0 };
        let mut l = Lin::default().add(&la, sa).add(&lb, sb);
        // -inf - x is exactly -inf.
        let exact = l.k == f64::NEG_INFINITY;
        if exact {
            l.terms.clear();
        }
        Ok((l, !exact))
    }

    /// Node `n` as a computed sum of exponentials.
    pub fn sum(&mut self, n: u32) -> Result<ExpSum, String> {
        if let Some(r) = self.sums.get(&n) {
            return r.clone();
        }
        let r = self.sum_inner(n);
        self.sums.insert(n, r.clone());
        r
    }

    fn sum_inner(&mut self, n: u32) -> Result<ExpSum, String> {
        match self.kind(n).clone() {
            NodeKind::Leaf => {
                let c = self.konst(n).unwrap();
                if c == 0.0 {
                    return Ok(ExpSum::default());
                }
                if c < 0.0 || !c.is_finite() {
                    return Err(format!("a constant {c} in a sum of exponentials"));
                }
                let l = Lin::default();
                Ok(ExpSum {
                    terms: [(
                        l.key(),
                        (
                            l,
                            TermRec {
                                coef: c,
                                up: 1.0,
                                lo: 1.0,
                                slope: 0.0,
                            },
                        ),
                    )]
                    .into(),
                    abs: 0.0,
                })
            }
            NodeKind::Add(a, b, rz) => {
                let s = self.sum(a)?.plus(&self.sum(b)?);
                // Adding +0 is exact.
                let exact = self.konst(a) == Some(0.0) || self.konst(b) == Some(0.0);
                Ok(s.round(if exact {
                    0.0
                } else if rz {
                    2.0 * U32
                } else {
                    U32
                }))
            }
            NodeKind::Fma {
                a,
                b,
                c,
                neg_ab: false,
                neg_c: false,
            } => {
                // An exact zero factor times a finite value: the product is exactly zero
                // (online softmax's first rescaling of its initial sum 0).
                let zero = |x: u32, y: u32, me: &mut Self| {
                    me.konst(x) == Some(0.0)
                        && match me.kind(y).clone() {
                            NodeKind::Seg(p) if p.0.args.len() == 1 => {
                                finite_everywhere(&p.0).is_ok()
                            }
                            _ => me.sum(y).is_ok(),
                        }
                };
                if zero(a, b, self) || zero(b, a, self) {
                    return match c {
                        Some(c) => self.sum(c),
                        None => Ok(ExpSum::default()),
                    };
                }
                let p = self.sum(a)?.times(&self.sum(b)?);
                let s = match c {
                    Some(c) => p.plus(&self.sum(c)?),
                    None => p,
                };
                // Products of values <= n with exact-zero factors vanish exactly; otherwise one
                // rounding (the result is >= 2^-126 or absolute 2^-150 more).
                let mut s = s.round(U32);
                s.abs += 2f64.powi(-150);
                Ok(s)
            }
            NodeKind::Seg(p) => {
                let p = &p.0;
                if p.args.len() != 1 {
                    return Err(format!(
                        "a slice of {} arguments where an exponential was expected",
                        p.args.len()
                    ));
                }
                self.exp_term(p)
            }
            k => Err(format!("not a sum of exponentials: {}", kind_name(&k))),
        }
    }

    /// exp of a one-argument slice whose argument is a linear form <= 0.
    fn exp_term(&mut self, p: &SProg) -> Result<ExpSum, String> {
        let (l, rounded) = self.lin(p.args[0])?;
        if !l.nonpos() {
            return Err("an exp argument not provably <= 0".into());
        }
        // Scaled-score terms: the positive atom unrounded (FFMA(s, c, -m)) or rounded (fl(c s)).
        let unrounded = match self.kind(p.args[0]).clone() {
            NodeKind::Fma { a, b, .. } => self.unrounded_scaled(a, b).is_some(),
            _ => false,
        };
        for (a, c) in &l.terms {
            if let (Atom::In((SCORE_LEAF, k)), true) = (a, *c > 0.0) {
                if self
                    .term_rounded
                    .insert(*k, !unrounded)
                    .is_some_and(|r| r == unrounded)
                {
                    return Err("a score atom used both rounded and unrounded in exp terms".into());
                }
            }
        }
        // ASSUME(N-SOFTMAX): e^d, or 2^d for base-2 slices (MUFU.EX2 without the log2(e)
        // scaling: attention folds it into the scale); one base per kernel.
        let (c, base2) = match exp_contract(p) {
            Ok(c) => (c, false),
            Err(e) => match exp2_contract(p) {
                Ok(c) => (c, true),
                Err(_) => return Err(e),
            },
        };
        if self.base2.replace(base2).is_some_and(|b| b != base2) {
            return Err("exp slices of two bases (e and 2) in one kernel".into());
        }
        Ok(exp_sum_of(l, rounded, &c, base2))
    }

    /// The shifted-form and base-2 flags of this kernel, checked: a shifted kernel's exp terms
    /// must all use unrounded scaled scores.
    fn flags(&self) -> Result<(bool, bool), String> {
        if self.shifted && self.term_rounded.values().any(|&r| r) {
            return Err("shifted maxima with rounded scaled scores in exp terms".into());
        }
        Ok((self.shifted, self.base2 == Some(true)))
    }

    /// Output node `n` as exp(L_i) / S.
    pub fn quot(&mut self, n: u32) -> Result<Quot, String> {
        match self.kind(n).clone() {
            // Triton: (1/S slice) * (exp slice), one rounding.
            NodeKind::Fma {
                a,
                b,
                c: None,
                neg_ab: false,
                ..
            } => {
                let (sa, sb) = match (self.kind(a), self.kind(b)) {
                    (NodeKind::Seg(x), NodeKind::Seg(y)) => (x.0.clone(), y.0.clone()),
                    _ => return Err("a product that is not (slice) * (slice)".into()),
                };
                // The reciprocal: the slice of one argument that is a sum of exponentials.
                for (rcp, ex) in [(&sa, &sb), (&sb, &sa)] {
                    if rcp.args.len() != 1 {
                        continue;
                    }
                    let Ok(den) = self.sum(rcp.args[0]) else {
                        continue;
                    };
                    let (s_lo, s_hi, n_m, members) = self.den_range(&den)?;
                    let rc = rcp_contract(rcp, s_lo, s_hi)?;
                    let (num, num_abs, ec) = self.numerator(ex, rcp.args[0], (s_lo, s_hi))?;
                    check_num(&num.0, &members)?;
                    return Ok(Quot {
                        num,
                        num_abs,
                        den,
                        rho_up: (1.0 + rc.r0) * (1.0 + U32),
                        rho_lo: (1.0 - rc.r0) * (1.0 - U32),
                        div_abs: 2f64.powi(-150) + rc.a,
                        n: n_m,
                        members,
                        contracts: vec![ec, rc],
                        shifted: self.flags()?.0,
                        base2: self.flags()?.1,
                    });
                }
                Err("no factor is a reciprocal of a sum of exponentials".into())
            }
            // torch: one slice ending in div.rn(exp slice, S).
            NodeKind::Seg(p) => {
                let p = p.0.clone();
                let si = p
                    .args
                    .iter()
                    .position(|&a| self.sum(a).is_ok() && !matches!(self.kind(a), NodeKind::Seg(_)))
                    .ok_or("no argument is a sum of exponentials")?;
                let den = self.sum(p.args[si])?;
                let (s_lo, s_hi, n_m, members) = self.den_range(&den)?;
                let q = simplify(&p, si as u16, s_lo, s_hi);
                let SVal::Op(root) = q.out else {
                    return Err("the slice's value is not computed".into());
                };
                let SKind::DivRn { a, b } = q.ops[root as usize].kind.clone() else {
                    return Err("the slice does not end in a division".into());
                };
                if b != SVal::Arg(si as u16) {
                    return Err("the divisor is not the sum".into());
                }
                let d = deps(&q);
                let da = dep_of(&q, &d, a);
                if da.count_ones() != 1 || da & (1 << si) != 0 {
                    return Err("the dividend depends on more than its exp argument".into());
                }
                let k = da.trailing_zeros() as u16;
                let ex = sub(&q, a, k);
                let num_sum = self.exp_term(&ex)?;
                let ec = exp_contract(&ex)?;
                let (num, num_abs) = single(&num_sum)?;
                check_num(&num.0, &members)?;
                Ok(Quot {
                    num,
                    num_abs,
                    den,
                    // Correctly rounded: u relative, 2^-150 where the quotient is subnormal.
                    rho_up: 1.0 + U32,
                    rho_lo: 1.0 - U32,
                    div_abs: 2f64.powi(-150),
                    n: n_m,
                    members,
                    contracts: vec![ec],
                    shifted: self.flags()?.0,
                    base2: self.flags()?.1,
                })
            }
            k => Err(format!("output is not a quotient: {}", kind_name(&k))),
        }
    }

    /// The numerator slice: simplified over the divisor's range, one argument left, exp.
    fn numerator(
        &mut self,
        ex: &SProg,
        den_node: u32,
        (lo, hi): (f64, f64),
    ) -> Result<((Lin, TermRec), f64, Contract), String> {
        let mut q = ex.clone();
        if let Some(k) = q.args.iter().position(|&a| a == den_node) {
            q = simplify(&q, k as u16, lo, hi);
        }
        let d = deps(&q);
        let dv = dep_of(&q, &d, q.out);
        if dv.count_ones() != 1 {
            return Err(format!(
                "the numerator depends on {} arguments after the divisor's range",
                dv.count_ones()
            ));
        }
        let k = dv.trailing_zeros() as u16;
        let one = sub(&q, q.out, k);
        let s = self.exp_term(&one)?;
        let c = exp_contract(&one)?;
        let (t, a) = single(&s)?;
        Ok((t, a, c))
    }

    /// Range and structure of a softmax denominator: one term exp(x_j - max(X)) with
    /// coefficient 1 per member x_j of X. Returns (lower, upper bound of the computed sum, n,
    /// members).
    fn den_range(&self, s: &ExpSum) -> Result<(f64, f64, usize, Vec<Leaf>), String> {
        let mut members: Option<Vec<Leaf>> = None;
        let mut seen = BTreeSet::new();
        for (l, r) in s.terms.values() {
            if r.coef != 1.0 {
                return Err("a denominator term with a coefficient other than 1".into());
            }
            let pos: Vec<&Atom> = l
                .terms
                .iter()
                .filter(|(_, c)| **c == 1.0)
                .map(|x| x.0)
                .collect();
            let neg: Vec<&Atom> = l
                .terms
                .iter()
                .filter(|(_, c)| **c == -1.0)
                .map(|x| x.0)
                .collect();
            match (pos.as_slice(), neg.as_slice(), l.k == 0.0, l.terms.len()) {
                ([Atom::In(x)], [Atom::Max(m)], true, 2) => {
                    if *members.get_or_insert_with(|| (**m).clone()) != ***m {
                        return Err("denominator terms against different maxima".into());
                    }
                    seen.insert(*x);
                }
                _ => return Err("a denominator term is not exp(x - max)".into()),
            }
        }
        let members = members.ok_or("an empty denominator")?;
        if seen.iter().copied().collect::<Vec<_>>() != members {
            return Err("the denominator's terms are not one per member of the maximum".into());
        }
        let n = members.len();
        let (sl, su) = s.rel_range(n);
        let (lo, hi) = (sl - s.abs, n as f64 * su + s.abs);
        if lo <= 0.0 {
            return Err("the denominator is not provably positive".into());
        }
        Ok((lo, hi, n, members))
    }
}

impl ExpSum {
    /// S^ / S in [lo, hi] for a softmax denominator of n terms (the theorem's S_lo / S_hi
    /// without the absolute part).
    fn rel_range(&self, n: usize) -> (f64, f64) {
        let mut up: f64 = 1.0;
        let mut lo: f64 = f64::INFINITY;
        let mut s: f64 = 0.0;
        for (_, r) in self.terms.values() {
            up = up.max(r.up);
            lo = lo.min(r.lo);
            s = s.max(r.slope);
        }
        let ln = (n as f64).ln();
        let b_up = s * ln * (1024.0 * s).exp() + n as f64 * (-1000f64).exp();
        (lo * (1.0 - s * ln), up * (1.0 + b_up))
    }
}

/// The numerator must be exp(x_i - max(X)) with x_i in X, X the denominator's row.
fn check_num(num: &Lin, members: &[Leaf]) -> Result<(), String> {
    let ok = num.k == 0.0
        && num.terms.len() == 2
        && num
            .terms
            .iter()
            .any(|(a, c)| *c == -1.0 && *a == Atom::Max(members.to_vec().into()))
        && num
            .terms
            .iter()
            .any(|(a, c)| *c == 1.0 && matches!(a, Atom::In(x) if members.contains(x)));
    if ok {
        Ok(())
    } else {
        Err("the numerator is not exp(x_i - max) of the denominator's row".into())
    }
}

/// A single-term sum: the term and the absolute error.
fn single(s: &ExpSum) -> Result<((Lin, TermRec), f64), String> {
    if s.terms.len() != 1 {
        return Err(format!(
            "{} terms where one exponential was expected",
            s.terms.len()
        ));
    }
    let (l, r) = s.terms.values().next().unwrap();
    Ok(((l.clone(), *r), s.abs))
}

/// The term of exp(d^) for d^ = fl(L) (or exact): the slice's contract and the argument's
/// rounding as an error record (module doc).
fn exp_sum_of(l: Lin, rounded: bool, c: &Contract, base2: bool) -> ExpSum {
    if l.k == f64::NEG_INFINITY {
        // exp(-inf) = 0: the slice's value there is within its absolute bound.
        return ExpSum {
            terms: BTreeMap::new(),
            abs: c.a.max(c.gmax * 0.0),
        };
    }
    let u_arg = if rounded { U32 } else { 0.0 };
    // 1 - y >= e^(-y(1 + 2^-6)) for y <= 2^-6 (y = r1 |d| / (1 - r0) on |d| <= 87, or 126
    // for base 2). Base 2: the slope is per natural unit of the argument (L ln 2): the
    // contract's r1 |d| = (r1 / ln 2) |L ln 2|, the rounding's 2^(u |d|) = e^(u |L ln 2|).
    let r1 = if base2 {
        c.r1 / std::f64::consts::LN_2
    } else {
        c.r1
    };
    let slope = (r1 * (1.0 + u_arg) / (1.0 - c.r0) + u_arg) * (1.0 + 2f64.powi(-6));
    ExpSum {
        terms: [(
            l.key(),
            (
                l,
                TermRec {
                    coef: 1.0,
                    up: 1.0 + c.r0,
                    lo: 1.0 - c.r0,
                    slope,
                },
            ),
        )]
        .into(),
        abs: c.a,
    }
}

fn kind_name(k: &NodeKind) -> &'static str {
    match k {
        NodeKind::Leaf => "constant",
        NodeKind::Opaque => "untracked value",
        NodeKind::Sel { .. } => "data-dependent selection",
        NodeKind::Input(_) => "input",
        NodeKind::Add(..) => "add",
        NodeKind::AtomAdd(..) => "atomic add",
        NodeKind::Mma { .. } => "mma",
        NodeKind::Round { .. } => "16-bit rounding",
        NodeKind::Mufu { .. } => "mufu",
        NodeKind::Fma { .. } => "fma",
        NodeKind::Max { .. } => "max",
        NodeKind::Seg(_) => "slice",
        NodeKind::Tf32 { .. } => "tf32 part",
    }
}

/// One kernel's bound on |y^ - y| for an output: K_rel y + K_arg + K_abs (module doc).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct KBound {
    pub k_rel: f64,
    pub k_arg: f64,
    pub k_abs: f64,
}

/// The shifted form (`Ctx::shifted`): every computed term is 2^theta 2^(L*) phi with theta
/// common to the row (cancels in the quotient) and L* the exact argument against the true
/// maximum; the shift, |theta| <= THETA = u SHIFT_X (1 + u), also enters each term's
/// arguments: their magnitudes along a term's chain of exponentials (at most 2B + 1 shifts for
/// B rescalings, B <= 15) exceed |L*| by at most 31 THETA, so the records widen by
/// e^(32 THETA s) (s the largest slope), and the absolute parts (not scaled by 2^theta) by
/// 2^THETA. (factor on up / S_hi, divisor of lo / S_lo; absolute factor)
fn shift_factor(shifted: bool, s: f64) -> (f64, f64) {
    if !shifted {
        return (1.0, 1.0);
    }
    let th = U32 * SHIFT_X * (1.0 + U32);
    ((32.0 * th * s).exp(), th.exp2())
}

pub fn kbound(q: &Quot) -> KBound {
    let (sl, su) = q.den.rel_range(q.n);
    let (_, r) = &q.num;
    let s_all = q
        .den
        .terms
        .values()
        .map(|t| t.1.slope)
        .fold(r.slope, f64::max);
    let (fs, fa) = shift_factor(q.shifted, s_all);
    let s_lo = sl / fs - q.den.abs * fa;
    let s_hi = su * fs + q.den.abs * fa;
    let s = r.slope;
    let a = r.up * fs * q.rho_up / s_lo;
    let b = r.lo / fs * q.rho_lo / s_hi;
    let k_rel = (a - 1.0).max(1.0 - b);
    let k_arg = a.max(1.0) * s / (std::f64::consts::E * (1.0 - s));
    let k_abs = q.num_abs * fa * q.rho_up.max(1.0) / s_lo + q.div_abs;
    let up = 1.0 + 2f64.powi(-40);
    KBound {
        k_rel: k_rel * up,
        k_arg: k_arg * up,
        k_abs: k_abs * up,
    }
}

/// The two kernels compute the same normal form.
pub fn same(r: &Quot, c: &Quot) -> Result<(), String> {
    if r.num.0.key() != c.num.0.key() {
        return Err("different numerators".into());
    }
    let keys = |q: &Quot| -> Vec<(LinKey, u64)> {
        q.den
            .terms
            .iter()
            .map(|(k, (_, t))| (k.clone(), t.coef.to_bits()))
            .collect()
    };
    if keys(r) != keys(c) {
        return Err("different denominators".into());
    }
    Ok(())
}

// ------------------------------------------------------------------ weighted sums (attention)
// ASSUME(N-ATTN): see ASSUMPTIONS.md

/// `WSum::terms`: per (key of L_t, v_t), (L_t, v_t, term record).
pub type WSumTerms = BTreeMap<(LinKey, Option<Leaf>), (Lin, Option<Leaf>, TermRec)>;

/// A computed weighted sum of exponentials (attention's P.V): value
///   (sum_t c_t e^(L_t) v_t phi_t) [/ S^ if `div`] + alpha,
/// v_t an input leaf (or 1), phi_t in [lo e^(-s|L|), up e^(s|L|)] (the term record), and
/// |alpha| <= abs0 + absv max|v| (after the division, if any).
#[derive(Clone, Debug, Default, Serialize)]
pub struct WSum {
    pub terms: WSumTerms,
    pub abs0: f64,
    pub absv: f64,
    /// The common divisor (normalize-first form): (structural hash, node, the lower bound
    /// of S^, the exact S being >= 1).
    pub div: Option<(u64, u32, f64)>,
}

impl WSum {
    fn from_exp(e: &ExpSum) -> WSum {
        WSum {
            terms: e
                .terms
                .iter()
                .map(|(k, (l, r))| ((k.clone(), None), (l.clone(), None, *r)))
                .collect(),
            abs0: e.abs,
            absv: 0.0,
            div: None,
        }
    }
    fn has_leaf(&self) -> bool {
        self.terms.values().any(|t| t.1.is_some()) || self.absv > 0.0
    }
    /// Magnitude bound (each e^L <= 1, before any division): (constant part, max|v| part).
    fn mag(&self) -> (f64, f64) {
        // Divided by S^ >= s_lo (S >= 1) in the normalize-first form.
        let d = self.div.map_or(1.0, |x| 1.0 / x.2);
        let (mut m0, mut mv) = (0.0, 0.0);
        for (_, v, r) in self.terms.values() {
            let x = r.coef.abs() * r.up * d;
            if v.is_some() { mv += x } else { m0 += x }
        }
        (m0 + self.abs0, mv + self.absv)
    }
    fn round(mut self, eps: f64) -> WSum {
        for (_, _, r) in self.terms.values_mut() {
            r.up *= 1.0 + eps;
            r.lo *= 1.0 - eps;
        }
        self.abs0 *= 1.0 + eps;
        self.absv *= 1.0 + eps;
        self
    }
    fn plus(mut self, o: &WSum) -> Result<WSum, String> {
        if o.terms.is_empty() && o.abs0 == 0.0 && o.absv == 0.0 {
            return Ok(self);
        }
        if self.terms.is_empty() && self.abs0 == 0.0 && self.absv == 0.0 {
            return Ok(o.clone());
        }
        if self.div.map(|d| d.0) != o.div.map(|d| d.0) {
            return Err("a sum of terms with different divisors".into());
        }
        for (k, (l, v, r)) in &o.terms {
            match self.terms.get_mut(k) {
                Some((_, _, x)) => {
                    // Per-term factors do not survive cancellation: same signs only.
                    if (x.coef < 0.0) != (r.coef < 0.0) {
                        return Err("two occurrences of one term with opposite signs".into());
                    }
                    *x = TermRec {
                        coef: x.coef + r.coef,
                        up: x.up.max(r.up),
                        lo: x.lo.min(r.lo),
                        slope: x.slope.max(r.slope),
                    }
                }
                None => {
                    self.terms.insert(k.clone(), (l.clone(), *v, *r));
                }
            }
        }
        self.abs0 += o.abs0;
        self.absv += o.absv;
        Ok(self)
    }
    fn times(&self, o: &WSum) -> Result<WSum, String> {
        if self.div.is_some() && o.div.is_some() {
            return Err("a product of two normalized values".into());
        }
        if self.has_leaf() && o.has_leaf() {
            return Err("a product of two weights".into());
        }
        let mut out = WSum {
            div: self.div.or(o.div),
            ..Default::default()
        };
        for (la, va, ra) in self.terms.values() {
            for (lb, vb, rb) in o.terms.values() {
                let l = la.add(lb, 1.0);
                let v = va.or(*vb);
                let r = TermRec {
                    coef: ra.coef * rb.coef,
                    up: ra.up * rb.up,
                    lo: ra.lo * rb.lo,
                    slope: ra.slope.max(rb.slope),
                };
                let one = WSum {
                    terms: [((l.key(), v), (l, v, r))].into(),
                    div: out.div,
                    ..Default::default()
                };
                out = out.plus(&one)?;
            }
        }
        // |a b - A B| <= |alpha_a| |b| + |alpha_b| |a| + |alpha_a alpha_b|, one side weighted.
        let (a0, av) = self.mag();
        let (b0, bv) = o.mag();
        let (x0, xv) = (self.abs0, self.absv);
        let (y0, yv) = (o.abs0, o.absv);
        out.abs0 = x0 * b0 + y0 * a0 + x0 * y0;
        out.absv = x0 * bv + xv * b0 + y0 * av + yv * a0 + x0 * yv + xv * y0;
        Ok(out)
    }
}

/// One attention output in normal form: (sum_j e^(L_j) v_j phi_j) / S^, the division's
/// relative error (1 for the normalize-first form, whose terms carry it).
#[derive(Clone, Debug, Serialize)]
pub struct WQuot {
    pub num: WSum,
    pub den: ExpSum,
    pub rho_up: f64,
    pub rho_lo: f64,
    pub div_abs: f64,
    pub n: usize,
    pub members: Vec<Leaf>,
    pub contracts: Vec<Contract>,
    pub shifted: bool,
    pub base2: bool,
}

/// One kernel's attention bound: |o^ - o| <= k_rel V1 + k_arg Vmax + k_abs0 + k_absv Vmax,
/// V1 = sum_j p_j |v_j|, Vmax = max_j |v_j|.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct WBound {
    pub k_rel: f64,
    pub k_arg: f64,
    pub k_abs0: f64,
    pub k_absv: f64,
}

impl<'a> Ctx<'a> {
    /// A structural hash of node `n` (identical DAGs: bitwise equal values for every input).
    fn shash(&mut self, n: u32) -> u64 {
        use std::hash::Hash;
        use std::hash::Hasher;
        if let Some(&h) = self.hashes.get(&n) {
            return h;
        }
        let mut st = std::collections::hash_map::DefaultHasher::new();
        let nd = &self.ns.nodes[n as usize];
        match nd.kind.clone() {
            NodeKind::Leaf => ("leaf", nd.hw).hash(&mut st),
            NodeKind::Input(o) => ("in", self.buf_id[o.0], o.1).hash(&mut st),
            NodeKind::Add(a, b, rz) => {
                let mut c = [self.shash(a), self.shash(b)];
                c.sort();
                ("add", rz, c).hash(&mut st)
            }
            NodeKind::Fma {
                a,
                b,
                c,
                neg_ab,
                neg_c,
            } => {
                let mut ab = [self.shash(a), self.shash(b)];
                ab.sort();
                let hc = c.map(|c| self.shash(c));
                ("fma", ab, hc, neg_ab, neg_c).hash(&mut st)
            }
            NodeKind::Max { a, b, min } => {
                let mut c = [self.shash(a), self.shash(b)];
                c.sort();
                ("max", min, c).hash(&mut st)
            }
            NodeKind::Seg(p) => {
                (
                    "seg",
                    format!("{:?}", p.0.ops.iter().map(|o| &o.kind).collect::<Vec<_>>()),
                )
                    .hash(&mut st);
                format!("{:?}", p.0.out).hash(&mut st);
                for &a in &p.0.args {
                    self.shash(a).hash(&mut st);
                }
            }
            NodeKind::Mma { acc, ops, sc, .. } => {
                let ha = acc.map(|a| self.shash(a));
                let mut th: Vec<Option<[u64; 2]>> = ops
                    .iter()
                    .map(|o| {
                        o.map(|(x, y)| {
                            let mut p = [self.shash(x), self.shash(y)];
                            p.sort();
                            p
                        })
                    })
                    .collect();
                if th.iter().any(|t| t.is_none()) || !sc.is_empty() {
                    ("node", n).hash(&mut st);
                } else {
                    th.sort();
                    ("mma", ha, th).hash(&mut st);
                }
            }
            NodeKind::Round { arg, f16 } => ("round", f16, self.shash(arg)).hash(&mut st),
            _ => ("node", n).hash(&mut st),
        }
        let h = st.finish();
        self.hashes.insert(n, h);
        h
    }

    /// Node `n` as a weighted sum of exponentials.
    pub fn wsum(&mut self, n: u32) -> Result<WSum, String> {
        if let Some(r) = self.wsums.get(&n) {
            return r.clone();
        }
        let r = self.wsum_inner(n);
        self.wsums.insert(n, r.clone());
        r
    }

    fn wsum_inner(&mut self, n: u32) -> Result<WSum, String> {
        // A sum of exponentials (no weights): as before.
        if let Ok(e) = self.sum(n) {
            return Ok(WSum::from_exp(&e));
        }
        match self.kind(n).clone() {
            NodeKind::Input((a, i)) => {
                let leaf = (self.buf_id[a], i);
                let l = Lin::default();
                Ok(WSum {
                    terms: [(
                        (l.key(), Some(leaf)),
                        (
                            l,
                            Some(leaf),
                            TermRec {
                                coef: 1.0,
                                up: 1.0,
                                lo: 1.0,
                                slope: 0.0,
                            },
                        ),
                    )]
                    .into(),
                    ..Default::default()
                })
            }
            NodeKind::Leaf if self.konst(n) == Some(0.0) => Ok(WSum::default()),
            // ASSUME(N-MMA): a tensor-core step adds the accumulator and its products with an
            // error <= eps (|acc| + sum |products|): each child times its own (1 + theta),
            // |theta| <= eps (Cor 1a's form), i.e. every term record widened by (1 +- eps).
            NodeKind::Mma { sc, .. } if !sc.is_empty() => Err("a block-scaled MMA step".into()),
            NodeKind::Mma { acc, ops, eps, .. } => {
                let mut s = match acc {
                    Some(a) if self.konst(a) == Some(0.0) => WSum::default(),
                    Some(a) => self.wsum(a)?,
                    None => WSum::default(),
                };
                for o in &ops {
                    let (x, y) = o.ok_or("an MMA product of untracked origin")?;
                    let p = self.wsum(x)?.times(&self.wsum(y)?)?;
                    s = s.plus(&p)?;
                }
                Ok(s.round(eps))
            }
            // ASSUME(N-ROUND): a bf16 rounding (P before the P.V step): relative 2^-8, plus
            // 2^-134 absolute where the bf16 is subnormal.
            NodeKind::Round { arg, f16 } => {
                let (u16_, a16) = crate::equiv::round_unit(f16);
                let mut w = self.wsum(arg)?.round(u16_);
                w.abs0 += a16;
                Ok(w)
            }
            NodeKind::Add(a, b, rz) => {
                let s = self.wsum(a)?.plus(&self.wsum(b)?)?;
                let exact = self.konst(a) == Some(0.0) || self.konst(b) == Some(0.0);
                Ok(s.round(if exact {
                    0.0
                } else if rz {
                    2.0 * U32
                } else {
                    U32
                }))
            }
            NodeKind::Fma {
                a,
                b,
                c,
                neg_ab,
                neg_c,
            } => {
                // A normalized exponential (Triton: rcp(S) * exp slice): one term over S^.
                if c.is_none() && !neg_ab {
                    if let Ok(q) = self.quot(n) {
                        return self.wsum_of_quot(n, &q);
                    }
                }
                // An exact zero times a finite value is exactly zero (the accumulator's first
                // rescaling); the other factor must be finite for every input.
                let zero = |x: u32, y: u32, me: &mut Self| {
                    me.konst(x) == Some(0.0)
                        && match me.kind(y).clone() {
                            NodeKind::Seg(p) if p.0.args.len() == 1 => {
                                finite_everywhere(&p.0).is_ok()
                            }
                            _ => me.konst(y).is_some_and(f64::is_finite) || me.wsum(y).is_ok(),
                        }
                };
                let prod = if zero(a, b, self) || zero(b, a, self) {
                    WSum::default()
                } else {
                    let mut p = self.wsum(a)?.times(&self.wsum(b)?)?;
                    if neg_ab {
                        p.terms.values_mut().for_each(|t| t.2.coef = -t.2.coef);
                    }
                    p
                };
                let s = match c {
                    Some(c) => {
                        let mut w = self.wsum(c)?;
                        if neg_c {
                            w.terms.values_mut().for_each(|t| t.2.coef = -t.2.coef);
                        }
                        prod.plus(&w)?
                    }
                    None => prod,
                };
                let mut s = s.round(U32);
                // A product below 2^-126: 2^-150 absolute (after the division, for a
                // normalized value: |v| times that is in absv).
                s.absv += 2f64.powi(-150);
                s.abs0 += 2f64.powi(-150);
                Ok(s)
            }
            k => Err(format!("not a weighted sum: {}", kind_name(&k))),
        }
    }

    /// A normalized exponential exp(L) / S^ (1 + rho) as a one-term weighted sum over S^.
    fn wsum_of_quot(&mut self, n: u32, q: &Quot) -> Result<WSum, String> {
        // The divisor node: the reciprocal slice's argument.
        let den_node = match self.kind(n).clone() {
            NodeKind::Fma { a, b, .. } => [a, b]
                .into_iter()
                .find_map(|x| match self.kind(x).clone() {
                    NodeKind::Seg(p) if p.0.args.len() == 1 && self.sum(p.0.args[0]).is_ok() => {
                        Some(p.0.args[0])
                    }
                    _ => None,
                })
                .ok_or("no divisor")?,
            _ => return Err("no divisor".into()),
        };
        let h = self.shash(den_node);
        let (s_lo, _, _, _) = self.den_range(&q.den)?;
        let (l, r) = &q.num;
        Ok(WSum {
            terms: [(
                (l.key(), None),
                (
                    l.clone(),
                    None,
                    TermRec {
                        coef: r.coef,
                        up: r.up * q.rho_up,
                        lo: r.lo * q.rho_lo,
                        slope: r.slope,
                    },
                ),
            )]
            .into(),
            abs0: q.num_abs * q.rho_up / s_lo + q.div_abs,
            absv: 0.0,
            div: Some((h, den_node, s_lo)),
        })
    }

    /// A reciprocal slice of a softmax denominator: its arguments' sum of exponentials S (the
    /// first argument that is one), and after folding a != -b tests into S != 0 and deciding
    /// the guards over S's range, a slice of S alone. (S's node, that one-argument slice, S,
    /// S's range) or None.
    #[allow(clippy::type_complexity)]
    fn rcp_slice(
        &mut self,
        p: &SProg,
    ) -> Result<Option<(u32, SProg, ExpSum, (f64, f64, usize, Vec<Leaf>))>, String> {
        let q = fold_sum_compare(p, self.ns);
        let Some(si) = q
            .args
            .iter()
            .position(|&a| !matches!(self.kind(a), NodeKind::Seg(_)) && self.sum(a).is_ok())
        else {
            return Ok(None);
        };
        let den = self.sum(q.args[si])?;
        let range = self.den_range(&den)?;
        let q = prune(&simplify(&q, si as u16, range.0, range.1));
        let d = deps(&q);
        if dep_of(&q, &d, q.out) != 1 << si {
            return Ok(None);
        }
        Ok(Some((q.args[si], sub(&q, q.out, si as u16), den, range)))
    }

    /// A slice fl(rcp.rn(S) * N^) (after deciding its guards over S's range): the flash form
    /// with a correctly rounded reciprocal (None: not this shape).
    fn wquot_rcp_slice(&mut self, p: &SProg) -> Result<Option<WQuot>, String> {
        let q = fold_sum_compare(p, self.ns);
        let Some(si) = q
            .args
            .iter()
            .position(|&a| !matches!(self.kind(a), NodeKind::Seg(_)) && self.sum(a).is_ok())
        else {
            return Ok(None);
        };
        let den = self.sum(q.args[si])?;
        let (s_lo, s_hi, n_m, members) = self.den_range(&den)?;
        let q = simplify(&q, si as u16, s_lo, s_hi);
        let q = prune(&q);
        // [DivRn(1, S), FMUL(op0, N)] (either operand order).
        if q.ops.len() != 2 || q.out != SVal::Op(1) {
            return Ok(None);
        }
        let SKind::DivRn {
            a: SVal::Const(0x3f80_0000),
            b: SVal::Arg(b),
        } = q.ops[0].kind
        else {
            return Ok(None);
        };
        if b as usize != si {
            return Ok(None);
        }
        let SKind::Inst {
            op:
                Op::Ffma {
                    a,
                    b: bb,
                    c: None,
                    neg: [false, false, false],
                    abs: [false, false, false],
                    rnd: Rnd::Rn,
                    ftz: false,
                    sat: false,
                    ..
                },
            regs,
            ..
        } = &q.ops[1].kind
        else {
            return Ok(None);
        };
        let val = |v: &Val| match v {
            Val::Reg(r, _) => regs.iter().find(|x| x.0 == *r).map(|x| x.1),
            _ => None,
        };
        let nk = match (val(a), val(bb)) {
            (Some(SVal::Op(0)), Some(SVal::Arg(k))) | (Some(SVal::Arg(k)), Some(SVal::Op(0))) => k,
            _ => return Ok(None),
        };
        let num = self.wsum(q.args[nk as usize])?;
        if num.div.is_some() {
            return Err("a normalized numerator divided again".into());
        }
        check_wnum(&num, &members)?;
        Ok(Some(WQuot {
            num,
            den,
            // rcp.rn: correctly rounded (u); the product: u (2^-150 where subnormal).
            rho_up: (1.0 + U32) * (1.0 + U32),
            rho_lo: (1.0 - U32) * (1.0 - U32),
            div_abs: 2f64.powi(-150),
            n: n_m,
            members,
            contracts: vec![],
            shifted: self.flags()?.0,
            base2: self.flags()?.1,
        }))
    }

    /// Output node `n` as (sum_j e^(L_j) v_j phi_j) / S^.
    pub fn wquot(&mut self, n: u32) -> Result<WQuot, String> {
        // Flash form: rcp(S^) * N^ (div.full, or rcp.rn with torch's S != 0 guard), N^ a
        // weighted sum (a slice of it, the identity after the range guards, or the node).
        if let NodeKind::Fma {
            a,
            b,
            c: None,
            neg_ab: false,
            ..
        } = self.kind(n).clone()
        {
            for (x, y) in [(a, b), (b, a)] {
                let NodeKind::Seg(xp) = self.kind(x).clone() else {
                    continue;
                };
                let Some((den_node, rcp, den, (s_lo, s_hi, n_m, members))) =
                    self.rcp_slice(&xp.0)?
                else {
                    continue;
                };
                let rc = rcp_contract(&rcp, s_lo, s_hi)?;
                let num_node = match self.kind(y).clone() {
                    NodeKind::Seg(yp) => {
                        let mut q = yp.0.as_ref().clone();
                        if let Some(k) = q.args.iter().position(|&a| a == den_node) {
                            q = simplify(&q, k as u16, s_lo, s_hi);
                        }
                        // The numerator slice must be the identity after the range guards.
                        let SVal::Arg(k) = q.out else {
                            return Err("the numerator slice is not the weighted sum itself".into());
                        };
                        q.args[k as usize]
                    }
                    _ => y,
                };
                let num = self.wsum(num_node)?;
                if num.div.is_some() {
                    return Err("a normalized numerator divided again".into());
                }
                check_wnum(&num, &members)?;
                return Ok(WQuot {
                    num,
                    den,
                    rho_up: (1.0 + rc.r0) * (1.0 + U32),
                    rho_lo: (1.0 - rc.r0) * (1.0 - U32),
                    div_abs: 2f64.powi(-150) + rc.a,
                    n: n_m,
                    members,
                    contracts: vec![rc],
                    shifted: self.flags()?.0,
                    base2: self.flags()?.1,
                });
            }
        }
        // torch's flash epilogue: one slice o = fl(r N^), r = (S != 0 ? rcp.rn(S) : 1), the
        // test S != 0 compiled as a != -b for S = fl(a + b).
        if let NodeKind::Seg(p) = self.kind(n).clone() {
            if let Some(w) = self.wquot_rcp_slice(&p.0)? {
                return Ok(w);
            }
        }
        // Normalize-first form: a weighted sum of normalized exponentials over one S^.
        let w = self.wsum(n)?;
        let Some((_, dn, _)) = w.div else {
            return Err("output is neither rcp(S) * N nor a sum of normalized terms".into());
        };
        let den = self.sum(dn)?;
        let (_, _, n_m, members) = self.den_range(&den)?;
        check_wnum(&w, &members)?;
        Ok(WQuot {
            num: w,
            den,
            rho_up: 1.0,
            rho_lo: 1.0,
            div_abs: 0.0,
            n: n_m,
            members,
            contracts: vec![],
            shifted: self.flags()?.0,
            base2: self.flags()?.1,
        })
    }
}

/// Every numerator term must be exp(x_j - max(X)) v_j with x_j in X and v_j a leaf, one per
/// (x_j, v_j), coefficient 1.
fn check_wnum(w: &WSum, members: &[Leaf]) -> Result<(), String> {
    for (l, v, r) in w.terms.values() {
        if v.is_none() || r.coef != 1.0 {
            return Err("a numerator term without its weight or with a coefficient".into());
        }
        check_num(l, members)?;
    }
    Ok(())
}

/// The weighted theorem (docs/equiv-reassoc.md, "Attention"): with Ŝ/S in [S_lo, S_hi] and the
/// division's relative error in [rho_lo, rho_up], r = (1 + rho) S / Ŝ lies in
/// [rho_lo / S_hi, rho_up / S_lo], and
///   |o^ - o| <= K_rel V1 + up r_hi b_up Vmax + (abs0 + absv Vmax) (rho_up / S_lo if the
///   numerator is divided, else 1),
/// K_rel = max(up r_hi - 1, 1 - lo r_lo), b_up = s ln n e^(1024 s) + n e^-1000 (the bound on
/// sum_j p_j (e^(s|L_j|) - 1) of the softmax theorem).
pub fn wbound(q: &WQuot) -> WBound {
    let (sl, su) = q.den.rel_range(q.n);
    let (mut up, mut lo, mut s): (f64, f64, f64) = (1.0, f64::INFINITY, 0.0);
    for (_, _, r) in q.num.terms.values() {
        up = up.max(r.up);
        lo = lo.min(r.lo);
        s = s.max(r.slope);
    }
    let s_all = q.den.terms.values().map(|t| t.1.slope).fold(s, f64::max);
    let (fs, fa) = shift_factor(q.shifted, s_all);
    let s_lo = sl / fs - q.den.abs * fa;
    let s_hi = su * fs + q.den.abs * fa;
    up *= fs;
    lo /= fs;
    let r_hi = q.rho_up / s_lo;
    let r_lo = q.rho_lo / s_hi;
    let k_rel = (up * r_hi - 1.0).max(1.0 - lo * r_lo);
    let n = q.n as f64;
    let b_up = s * n.ln() * (1024.0 * s).exp() + n * (-1000f64).exp();
    let k_arg = up * r_hi * b_up;
    // The numerator's absolute part: divided (flash) or already after the division.
    let f = if q.num.div.is_some() {
        1.0
    } else {
        q.rho_up / s_lo
    };
    let upr = 1.0 + 2f64.powi(-40);
    WBound {
        k_rel: k_rel * upr,
        k_arg: k_arg * upr,
        k_abs0: (q.num.abs0 * fa * f + q.div_abs) * upr,
        k_absv: q.num.absv * fa * f * upr,
    }
}

/// The two kernels compute the same weighted normal form.
pub fn wsame(r: &WQuot, c: &WQuot) -> Result<(), String> {
    let keys = |w: &WSum| -> Vec<(LinKey, Option<Leaf>, u64)> {
        w.terms
            .iter()
            .map(|(k, (_, _, t))| (k.0.clone(), k.1, t.coef.to_bits()))
            .collect()
    };
    if keys(&r.num) != keys(&c.num) {
        return Err("different numerators".into());
    }
    let dk = |q: &WQuot| -> Vec<(LinKey, u64)> {
        q.den
            .terms
            .iter()
            .map(|(k, (_, t))| (k.clone(), t.coef.to_bits()))
            .collect()
    };
    if dk(r) != dk(c) {
        return Err("different denominators".into());
    }
    Ok(())
}

/// Bumped by the caller whenever the input that `AttnEval` methods evaluate at changes: the
/// score values at an input are cached per row (`ScoreRows`) for the current generation.
pub static ATTN_INPUT_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Computed scores among a row's members (`score_rows`): per score key, (monomial, c, a,
/// g_R, g_C).
pub type ScoreMap = HashMap<u64, Vec<(crate::equiv::Mono, f64, f64, f64, f64)>>;

/// `ScoreRows::cache`: the input generation and `AttnEval::score_vals_raw` at it.
type ScoreCache = Option<(u64, std::sync::Arc<(HashMap<u64, f64>, f64, f64)>)>;

/// The rows of a row's computed scores (`score_rows`), with a cache of their values at the
/// current input (`ATTN_INPUT_GEN`).
#[derive(Debug, Default)]
pub struct ScoreRows {
    pub rows: ScoreMap,
    cache: std::sync::Mutex<ScoreCache>,
}

impl ScoreRows {
    pub fn new(rows: ScoreMap) -> ScoreRows {
        ScoreRows {
            rows,
            cache: Default::default(),
        }
    }
}

impl std::ops::Deref for ScoreRows {
    type Target = ScoreMap;
    fn deref(&self) -> &Self::Target {
        &self.rows
    }
}

/// What evaluating an attention bound at an input needs.
#[derive(Clone, Debug, Default)]
pub struct AttnEval {
    /// The row (members of the maximum) and, per numerator term, (x_j, v_j).
    pub members: Vec<Leaf>,
    pub pairs: Vec<(Leaf, Leaf)>,
    pub r: (f64, f64, f64, f64),
    pub c: (f64, f64, f64, f64),
    /// Computed scores among the members: per score key, (monomial, c, a, g_R, g_C).
    /// Shared between the outputs of a row (the same members give the same rows).
    pub scores: std::sync::Arc<ScoreRows>,
    /// Base-2 exponentials: the atoms are in log2 units (o = sum_j 2^(x_j) v_j / sum_j 2^(x_j)),
    /// a score error Delta there is ln 2 Delta in the natural exponent.
    pub base2: bool,
    /// Some kernel has the shifted form: valid while every |score| <= SHIFT_X.
    pub shifted: bool,
    /// Both outputs are bf16 roundings of the forms (N-ROUND): 2^-8, else 0.
    pub out_round: f64,
}

/// The rows of each computed score in a row of members: exact coefficients and both kernels'
/// Cor 1a multipliers (the keys are exact polynomials, so the coefficients agree).
pub fn score_rows(members: &[Leaf], r: &Ctx, c: &Ctx) -> Result<ScoreMap, String> {
    let mut out = HashMap::new();
    for &(b, k) in members {
        if b != SCORE_LEAF {
            continue;
        }
        let (Some((_, pr)), Some((_, pc))) = (r.scores.get(&k), c.scores.get(&k)) else {
            return Err("a score atom only one kernel computes".into());
        };
        let mut rows = Vec::new();
        let keys: BTreeSet<&crate::equiv::Mono> = pr.keys().chain(pc.keys()).collect();
        for m in keys {
            let tr = pr.get(m).copied().unwrap_or_default();
            let tc = pc.get(m).copied().unwrap_or_default();
            if tr.coef != tc.coef {
                return Err("score polynomials differ".into());
            }
            let a = tr.abs.max(tc.abs).max(tr.coef.abs());
            let up = 1.0 + 2f64.powi(-40);
            rows.push((m.clone(), tr.coef, a, tr.gam * up, tc.gam * up));
        }
        out.insert(k, rows);
    }
    Ok(out)
}

impl AttnEval {
    pub fn new(q: &WQuot, r: WBound, c: WBound) -> AttnEval {
        let pairs = q
            .num
            .terms
            .values()
            .filter_map(|(l, v, _)| {
                let x = l.terms.iter().find_map(|(a, k)| match a {
                    Atom::In(x) if *k == 1.0 => Some(*x),
                    _ => None,
                })?;
                Some((x, (*v)?))
            })
            .collect();
        AttnEval {
            members: q.members.clone(),
            pairs,
            r: (r.k_rel, r.k_arg, r.k_abs0, r.k_absv),
            c: (c.k_rel, c.k_arg, c.k_abs0, c.k_absv),
            scores: Default::default(),
            base2: q.base2,
            shifted: q.shifted,
            out_round: 0.0,
        }
    }
    /// Is input x covered (the shifted form: every |score| + its error <= SHIFT_X)?
    pub fn covered(&self, x: &dyn Fn(Leaf) -> f64) -> bool {
        if !self.shifted {
            return true;
        }
        let (vals, dr, dc) = self.score_vals_raw(x);
        vals.values().all(|v| v.abs() + dr.max(dc) <= SHIFT_X)
    }
    /// The exact scores at x and each kernel's largest score error Delta_K (Cor 1a), the
    /// errors in natural exponent units.
    fn score_vals(&self, x: &dyn Fn(Leaf) -> f64) -> (HashMap<u64, f64>, f64, f64) {
        let (v, dr, dc) = self.score_vals_raw(x);
        let k = if self.base2 {
            std::f64::consts::LN_2
        } else {
            1.0
        };
        (v, dr * k, dc * k)
    }
    /// The same, errors in the atoms' own units (cached per row for the current input).
    fn score_vals_raw(&self, x: &dyn Fn(Leaf) -> f64) -> (HashMap<u64, f64>, f64, f64) {
        let g = ATTN_INPUT_GEN.load(std::sync::atomic::Ordering::Relaxed);
        if let Some((cg, v)) = &*self.scores.cache.lock().unwrap() {
            if *cg == g {
                return (v.0.clone(), v.1, v.2);
            }
        }
        let v = self.score_vals_compute(x);
        *self.scores.cache.lock().unwrap() = Some((g, std::sync::Arc::new(v.clone())));
        v
    }
    fn score_vals_compute(&self, x: &dyn Fn(Leaf) -> f64) -> (HashMap<u64, f64>, f64, f64) {
        let (mut vals, mut dr, mut dc) = (HashMap::new(), 0.0f64, 0.0f64);
        for (k, rows) in self.scores.iter() {
            let (mut s, mut er, mut ec) = (0.0, 0.0, 0.0);
            for (m, c, _, gr, gc) in rows {
                let v: f64 = m.iter().map(|&l| x(l)).product();
                s += c * v;
                er += gr * v.abs();
                ec += gc * v.abs();
            }
            vals.insert(*k, s);
            dr = dr.max(er);
            dc = dc.max(ec);
        }
        (vals, dr, dc)
    }
    /// Q(x) = max_j sum_m a_m |x_m| over the computed scores (0 without).
    pub fn q_at(&self, x: &dyn Fn(Leaf) -> f64) -> f64 {
        self.scores
            .values()
            .map(|rows| {
                rows.iter()
                    .map(|(m, _, a, _, _)| a * m.iter().map(|&l| x(l).abs()).product::<f64>())
                    .sum::<f64>()
            })
            .fold(0.0, f64::max)
    }
    /// (exact o, V1, Vmax) at x (scores at their exact values).
    pub fn exact(&self, x: &dyn Fn(Leaf) -> f64) -> (f64, f64, f64) {
        let (sv, _, _) = self.score_vals(x);
        self.exact_with(x, &sv)
    }
    fn exact_with(&self, x: &dyn Fn(Leaf) -> f64, sv: &HashMap<u64, f64>) -> (f64, f64, f64) {
        let xv = |l: Leaf| {
            if l.0 == SCORE_LEAF {
                sv.get(&l.1).copied().unwrap_or(f64::NAN)
            } else {
                x(l)
            }
        };
        let ex = |d: f64| if self.base2 { d.exp2() } else { d.exp() };
        let m = self
            .members
            .iter()
            .map(|&l| xv(l))
            .fold(f64::NEG_INFINITY, f64::max);
        let s: f64 = self.members.iter().map(|&l| ex(xv(l) - m)).sum();
        let (mut o, mut v1, mut vm) = (0.0, 0.0, 0.0f64);
        for &(xl, vl) in &self.pairs {
            let p = ex(xv(xl) - m) / s;
            let v = x(vl);
            o += p * v;
            v1 += p * v.abs();
            vm = vm.max(v.abs());
        }
        (o, v1, vm)
    }
    /// One kernel's bound against the exact o: its normal-form bound at its own computed
    /// scores (V1 there <= e^(2 Delta) V1) plus |o(s^) - o(s)| <= (e^(2 Delta) - 1) V1.
    /// With a bf16 output rounding (`ub` = 2^-8): (1 + ub) (that) + ub V1 + 2^-134.
    fn one(k: (f64, f64, f64, f64), v1: f64, vm: f64, delta: f64, ub: f64) -> f64 {
        let g = (2.0 * delta).exp();
        let b = k.0 * g * v1 + k.1 * vm + k.2 + k.3 * vm + (g - 1.0) * v1;
        let sb = if ub > 0.0 {
            crate::equiv::BF16_SUB
        } else {
            0.0
        };
        (b * (1.0 + ub) + ub * v1 + sb) * (1.0 + 1e-9) + 1e-300
    }
    /// (bound on |C - R|, exact o, each kernel's own bound) at x.
    pub fn at(&self, x: &dyn Fn(Leaf) -> f64) -> (f64, f64, f64, f64) {
        let (sv, dr, dc) = self.score_vals(x);
        let (o, v1, vm) = self.exact_with(x, &sv);
        let ub = self.out_round;
        let (br, bc) = (
            Self::one(self.r, v1, vm, dr, ub),
            Self::one(self.c, v1, vm, dc, ub),
        );
        (br + bc, o, br, bc)
    }
}

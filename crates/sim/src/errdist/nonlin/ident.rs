/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Which real function an instruction slice computes (for the exact value of a non-chain
//! output, and for the slices' local errors in the analytic tiers).
//!
//! A slice (`seg.rs`) is the bit-level computation of a value from its arguments: an exp
//! idiom's range reduction, a reciprocal's Newton steps, an rsqrt with its range guards. Its
//! real counterpart is not readable from the instructions (integer exponent tricks), so it is
//! *identified*: the slice is evaluated on argument tuples the kernel produces (the captured
//! input and perturbed inputs, `identify_prog`), and fitted against the forms
//!
//!   f(c0 + sum_i c_i a_i),  a_j * f(c0 + sum_{i != j} c_i a_i),  a_0 / a_1,  named(a_0)
//!
//! with f in {2^x, 1/x, 1/sqrt x, sqrt x, tanh x, x} (fitted on f^-1 of the outputs by least
//! squares) and the named elementwise functions (`elementwise::TRUE_FUNCS`). The fitted
//! constants are snapped to the closest of {1, log2 e, ln 2, sqrt 2, log2 e sqrt 2} x 2^k, the
//! slice's own f32 constants and those times log2 e or ln 2 (so exp2(x * 0x3fb8aa3b) is exp x,
//! the f32 rounding of log2 e being part of the kernel's error), and accepted when the slice
//! is within 2^-11 relative of the snapped form on every probe.
//! ASSUME(N-ERRDIST-IDENT): see ASSUMPTIONS.md

use std::collections::BTreeMap;

use serde::Deserialize;
use serde::Serialize;

use crate::errdist::det::HashMap;
use crate::seg::SKind;
use crate::seg::SProg;

/// The base function of a form.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Func {
    Exp2,
    Rcp,
    Rsqrt,
    Sqrt,
    Tanh,
    Ident,
    /// A polynomial of one argument (`RealFn::parg`): c0 + sum_i coef_i a^(i+1).
    Poly,
    /// arg0 / arg1.
    Div,
    /// A named elementwise function (`named`).
    Named(String),
}

/// Named one-argument functions: `elementwise::TRUE_FUNCS` and pieces of activations.
pub const NAMED: &[&str] = &[
    "gelu",
    "gelu_tanh",
    "silu",
    "sigmoid",
    "softplus",
    "erf",
    "erfc",
    "tanh",
    "exp",
    "log",
    "relu",
    "abs",
    "sin",
    "cos",
    // tanh(sqrt(2/pi) (x + 0.044715 x^3)): gelu_tanh's inner tanh.
    "tanh_gelu",
];

/// The value of a named function.
pub fn named(name: &str, x: f64) -> f64 {
    match name {
        "erfc" => crate::elementwise::erfc(x),
        "tanh_gelu" => ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh(),
        n => crate::elementwise::true_fn(n, x).unwrap_or(f64::NAN),
    }
}

impl Func {
    pub fn name(&self) -> String {
        match self {
            Func::Exp2 => "exp2".into(),
            Func::Rcp => "rcp".into(),
            Func::Rsqrt => "rsqrt".into(),
            Func::Sqrt => "sqrt".into(),
            Func::Tanh => "tanh".into(),
            Func::Ident => "affine".into(),
            Func::Poly => "poly".into(),
            Func::Div => "div".into(),
            Func::Named(n) => n.clone(),
        }
    }
    pub fn f(&self, z: f64) -> f64 {
        match self {
            Func::Exp2 => z.exp2(),
            Func::Rcp => 1.0 / z,
            Func::Rsqrt => 1.0 / z.sqrt(),
            Func::Sqrt => z.sqrt(),
            Func::Tanh => z.tanh(),
            Func::Ident | Func::Poly => z,
            Func::Div => f64::NAN,
            Func::Named(n) => named(n, z),
        }
    }
    /// d f / d z.
    pub fn df(&self, z: f64) -> f64 {
        match self {
            Func::Exp2 => std::f64::consts::LN_2 * z.exp2(),
            Func::Rcp => -1.0 / (z * z),
            Func::Rsqrt => -0.5 / (z * z.sqrt()),
            Func::Sqrt => 0.5 / z.sqrt(),
            Func::Tanh => 1.0 - z.tanh().powi(2),
            Func::Ident | Func::Poly => 1.0,
            Func::Div => f64::NAN,
            Func::Named(_) => {
                let h = 1e-5 * z.abs().max(1e-2);
                (self.f(z + h) - self.f(z - h)) / (2.0 * h)
            }
        }
    }
    fn inv(&self, g: f64) -> Option<f64> {
        let z = match self {
            Func::Exp2 if g > 0.0 => g.log2(),
            Func::Rcp if g != 0.0 => 1.0 / g,
            Func::Rsqrt if g > 0.0 => 1.0 / (g * g),
            Func::Sqrt if g >= 0.0 => g * g,
            Func::Tanh if g.abs() < 1.0 => g.atanh(),
            Func::Ident | Func::Poly => g,
            _ => return None,
        };
        z.is_finite().then_some(z)
    }
}

/// A slice's identified real function:
/// `outer.0 + outer.1 * [a_mul^pow *] f(c0 + sum_i coef_i a_i)` (Div: a_0 / a_1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealFn {
    pub func: Func,
    /// Per argument (the multiplier argument's coefficient is 0).
    pub coef: Vec<f64>,
    pub c0: f64,
    /// The argument multiplying f (power 1) or dividing it (power -1), if any.
    pub mul: Option<usize>,
    pub pow: i32,
    /// Outer affine map (alpha, beta) (named forms; else (0, 1)).
    pub outer: (f64, f64),
    /// The argument of a polynomial form.
    #[serde(default)]
    pub parg: usize,
    /// Every constant snapped to a named value (else some are least-squares fits).
    pub snapped: bool,
    /// Largest relative deviation of the slice from this form on the probes.
    pub fit_err: f64,
    pub probes: usize,
}

impl RealFn {
    fn plain(func: Func, n: usize) -> RealFn {
        RealFn {
            func,
            coef: vec![0.0; n],
            c0: 0.0,
            mul: None,
            pow: 1,
            outer: (0.0, 1.0),
            parg: 0,
            snapped: true,
            fit_err: 0.0,
            probes: 0,
        }
    }
    /// The inner argument z = c0 + sum coef_i a_i.
    pub fn z(&self, a: &[f64]) -> f64 {
        if self.func == Func::Poly {
            let x = a[self.parg];
            let mut p = 0.0;
            for c in self.coef.iter().rev() {
                p = (p + c) * x;
            }
            return self.c0 + p;
        }
        self.c0
            + self
                .coef
                .iter()
                .zip(a)
                .filter(|x| *x.0 != 0.0)
                .map(|(c, x)| c * x)
                .sum::<f64>()
    }
    fn factor(&self, a: &[f64]) -> f64 {
        match self.mul {
            Some(j) if self.pow < 0 => 1.0 / a[j],
            Some(j) => a[j],
            None => 1.0,
        }
    }
    pub fn eval(&self, a: &[f64]) -> f64 {
        if self.func == Func::Div {
            return a[0] / a[1];
        }
        self.outer.0 + self.outer.1 * self.factor(a) * self.func.f(self.z(a))
    }
    /// Partial derivatives with respect to each argument.
    pub fn grad(&self, a: &[f64]) -> Vec<f64> {
        if self.func == Func::Div {
            return vec![1.0 / a[1], -a[0] / (a[1] * a[1])];
        }
        if self.func == Func::Poly {
            let x = a[self.parg];
            let mut g = vec![0.0; a.len()];
            g[self.parg] = self
                .coef
                .iter()
                .enumerate()
                .map(|(i, c)| (i + 1) as f64 * c * x.powi(i as i32))
                .sum();
            return g;
        }
        let z = self.z(a);
        let (f, d) = (self.func.f(z), self.func.df(z));
        let m = self.factor(a) * self.outer.1;
        let mut g: Vec<f64> = self.coef.iter().map(|c| m * c * d).collect();
        if let Some(j) = self.mul {
            g[j] += self.outer.1 * if self.pow < 0 { -f / (a[j] * a[j]) } else { f };
        }
        g
    }
    pub fn show(&self) -> String {
        if self.func == Func::Div {
            return "a0 / a1".into();
        }
        if self.func == Func::Poly {
            let mut s = format!("{:e}", self.c0);
            for (i, c) in self.coef.iter().enumerate() {
                if *c != 0.0 {
                    s += &format!(" + {c:.9e} a{}^{}", self.parg, i + 1);
                }
            }
            return format!("poly({s})");
        }
        let mut s = String::new();
        if self.c0 != 0.0 {
            s += &format!("{:.6e}", self.c0);
        }
        for (i, c) in self.coef.iter().enumerate() {
            if *c != 0.0 {
                s += &format!(" + {c:.9e} a{i}");
            }
        }
        let mut f = format!("{}({})", self.func.name(), s.trim_start_matches(" + "));
        match self.mul {
            Some(j) if self.pow < 0 => f = format!("{f} / a{j}"),
            Some(j) => f = format!("a{j} * {f}"),
            None => {}
        }
        if self.outer != (0.0, 1.0) {
            f = format!("{} + {} * {f}", self.outer.0, self.outer.1);
        }
        f
    }
}

/// A key identifying a slice's computation (its ops and output; not its argument nodes).
pub fn slice_key(p: &SProg) -> String {
    let mut s = String::new();
    for o in &p.ops {
        s += &format!("{}:{:?};", o.pc, o.kind);
    }
    s + &format!("{:?}", p.out)
}

/// A key of a slice's function independent of its placement (no pcs, registers replaced by
/// the values they hold): equal keys compute the same function of their arguments (the form
/// of `softmax::slice_key`).
pub fn func_key(p: &SProg) -> String {
    let mut s = String::new();
    for o in &p.ops {
        match &o.kind {
            SKind::Inst {
                op,
                dst,
                regs,
                cvals,
            } => {
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

/// The f32 constants a slice reads (immediates and constant-bank values).
fn slice_consts(p: &SProg) -> Vec<f64> {
    let mut out = Vec::new();
    for o in &p.ops {
        if let SKind::Inst {
            op, cvals, regs, ..
        } = &o.kind
        {
            for (_, v) in cvals {
                out.push(f32::from_bits(*v as u32) as f64);
            }
            for (_, v) in regs {
                if let crate::seg::SVal::Const(c) = v {
                    out.push(f32::from_bits(*c as u32) as f64);
                }
            }
            let d = format!("{op:?}");
            for part in d.split("Imm(").skip(1) {
                let n: String = part.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(v) = n.parse::<u64>() {
                    out.push(f32::from_bits(v as u32) as f64);
                }
            }
        }
    }
    out.retain(|x| x.is_finite() && *x != 0.0 && x.abs() > 1e-30 && x.abs() < 1e30);
    out
}

/// Snap candidates (positive), with a priority: 0 for exact named constants
/// ({1, log2 e, ln 2, sqrt 2, ...} x 2^k, decimal epsilons), 1 for the slice's own f32
/// constants and their products with log2 e and ln 2.
fn snap_candidates(consts: &[f64]) -> Vec<(f64, u8)> {
    let l2e = std::f64::consts::LOG2_E;
    let ln2 = std::f64::consts::LN_2;
    let r2 = std::f64::consts::SQRT_2;
    let rpi = (2.0 / std::f64::consts::PI).sqrt();
    let mut out = Vec::new();
    for b in [
        1.0,
        l2e,
        ln2,
        r2,
        l2e * r2,
        ln2 * r2,
        3.0,
        1.0 / 3.0,
        5.0,
        rpi,
        rpi * 0.044715,
        0.044715,
    ] {
        for k in -40..=40 {
            out.push((b * 2f64.powi(k), 0));
        }
    }
    for e in [1e-5, 1e-6, 1e-8, 1e-12, 1e-10, 1e-3, 1e-4] {
        out.push((e, 0));
    }
    for &c in consts {
        let c = c.abs();
        for v in [c, c * l2e, c * ln2, c / l2e] {
            out.push((v, 1));
        }
    }
    out
}

fn snap(c: f64, cands: &[(f64, u8)], tol: f64) -> Option<f64> {
    if c.abs() <= tol {
        return Some(0.0);
    }
    let mut best: Option<(f64, u8, f64)> = None;
    for &(b, pr) in cands {
        let v = b.copysign(c);
        let d = (v - c).abs();
        if d <= tol && best.is_none_or(|x| (pr, d) < (x.1, x.2)) {
            best = Some((v, pr, d));
        }
    }
    best.map(|x| x.0)
}

/// Least squares z ~ [c0 +] sum c_i a_i over the columns in `cols`; None if singular.
fn lsq(rows: &[(Vec<f64>, f64)], cols: &[usize], intercept: bool) -> Option<(f64, Vec<f64>)> {
    let off = intercept as usize;
    let k = cols.len() + off;
    if k == 0 {
        return Some((0.0, vec![]));
    }
    let mut sc = vec![1.0f64; k];
    for (j, &c) in cols.iter().enumerate() {
        let m = rows.iter().map(|r| r.0[c].abs()).fold(0.0, f64::max);
        sc[j + off] = if m > 0.0 { m } else { 1.0 };
    }
    let mut a = vec![vec![0.0f64; k + 1]; k];
    for (x, z) in rows {
        let mut v = Vec::with_capacity(k);
        if intercept {
            v.push(1.0);
        }
        for (j, &c) in cols.iter().enumerate() {
            v.push(x[c] / sc[j + off]);
        }
        for i in 0..k {
            for j in 0..k {
                a[i][j] += v[i] * v[j];
            }
            a[i][k] += v[i] * z;
        }
    }
    let diag: f64 = (0..k).map(|i| a[i][i]).fold(0.0, f64::max);
    for i in 0..k {
        let p = (i..k).max_by(|&x, &y| a[x][i].abs().total_cmp(&a[y][i].abs()))?;
        if a[p][i].abs() <= 1e-13 * diag {
            return None;
        }
        a.swap(i, p);
        for r in 0..k {
            if r != i {
                let f = a[r][i] / a[i][i];
                #[allow(clippy::needless_range_loop)] // row r updated from row i of the same matrix
                for c in i..=k {
                    a[r][c] -= f * a[i][c];
                }
            }
        }
    }
    let sol: Vec<f64> = (0..k).map(|i| a[i][k] / a[i][i] / sc[i]).collect();
    if intercept {
        Some((sol[0], sol[1..].to_vec()))
    } else {
        Some((0.0, sol))
    }
}

/// Absolute floor of the relative deviation measure (where f is near 0 or underflows).
fn floor_of(f: &Func) -> f64 {
    match f {
        Func::Exp2 => 2f64.powi(-100),
        Func::Named(_) | Func::Tanh | Func::Ident | Func::Poly => 2f64.powi(-10),
        _ => 0.0,
    }
}

/// Largest relative deviation of the probes' outputs from the form.
fn max_err(r: &RealFn, probes: &[(Vec<f64>, f64)]) -> f64 {
    // A named form's floor is a quarter of its outer scale: 1 + tanh(z) built on MUFU.TANH
    // (2^-11 absolute) has large relative errors where it cancels.
    let floor = match r.func {
        Func::Named(_) => floor_of(&r.func).max(r.outer.1.abs() * 0.25),
        _ => floor_of(&r.func),
    };
    let mut w = 0.0f64;
    for (a, g) in probes {
        let p = r.eval(a);
        let d = (p - g).abs() / p.abs().max(g.abs()).max(floor).max(1e-300);
        if !d.is_finite() {
            if p == *g {
                continue;
            }
            return f64::INFINITY;
        }
        w = w.max(d);
    }
    w
}

pub const ACCEPT: f64 = 1.0 / 2048.0;

/// Snap a fitted form's constants (coefficients relative, c0 against the linear part, the
/// outer map), zeroing terms the value is insensitive to; keep only snaps that do not make the
/// fit worse.
fn snap_form(r: &RealFn, probes: &[(Vec<f64>, f64)], cands: &[(f64, u8)]) -> RealFn {
    let n = r.coef.len();
    // Sensitivity of log f to each term over the probes.
    let mut sens = vec![0.0f64; n + 1];
    for (a, _) in probes {
        let z = r.z(a);
        let (f, d) = (r.func.f(z), r.func.df(z));
        let l = if f != 0.0 { (d / f).abs() } else { d.abs() };
        if !l.is_finite() {
            continue;
        }
        for i in 0..n {
            sens[i] = sens[i].max((r.coef[i] * a[i]).abs() * l);
        }
        sens[n] = sens[n].max(r.c0.abs() * l);
    }
    let tol_rel = 2f64.powi(-14);
    let ok = |t: &RealFn| max_err(t, probes) <= 4.0 * r.fit_err + 2f64.powi(-23);
    let mut g = r.clone();
    let mut all = true;
    for i in 0..=n + 2 {
        let mut t = g.clone();
        let snapped = if i < n {
            if t.coef[i] == 0.0 {
                continue;
            }
            if sens[i] < 2f64.powi(-26) {
                Some(0.0)
            } else {
                snap(t.coef[i], cands, tol_rel * t.coef[i].abs())
            }
            .map(|v| t.coef[i] = v)
        } else if i == n {
            if t.c0 == 0.0 {
                continue;
            }
            if sens[n] < 2f64.powi(-26) {
                Some(0.0)
            } else {
                snap(t.c0, cands, tol_rel * t.c0.abs())
            }
            .map(|v| t.c0 = v)
        } else if i == n + 1 {
            if t.outer.0 == 0.0 {
                continue;
            }
            snap(
                t.outer.0,
                cands,
                2f64.powi(-14) * t.outer.0.abs().max(t.outer.1.abs()),
            )
            .map(|v| t.outer.0 = v)
        } else {
            if t.outer.1 == 1.0 {
                continue;
            }
            snap(t.outer.1, cands, tol_rel * t.outer.1.abs()).map(|v| t.outer.1 = v)
        };
        if snapped.is_some() && ok(&t) {
            g = t;
        } else {
            all = false;
        }
    }
    g.snapped = all;
    g.fit_err = max_err(&g, probes);
    g
}

/// Fit one form `[a_mul^pow *] f(c0 + sum c_i a_i)`.
#[allow(clippy::too_many_arguments)]
fn fit_form(
    p: &SProg,
    probes: &[(Vec<f64>, f64)],
    func: Func,
    mul: Option<usize>,
    pow: i32,
    nargs: usize,
    cands: &[(f64, u8)],
) -> Option<RealFn> {
    let cols: Vec<usize> = (0..nargs).filter(|i| Some(*i) != mul).collect();
    fit_form_cols(p, probes, func, mul, pow, nargs, cands, cols)
}

/// `fit_form` with the inner argument over the arguments `cols` only (the others' coefficients
/// 0): arguments that are linearly dependent on the probes (a sum and its two terms, both
/// arguments of a slice) leave the full least-squares problem singular.
#[allow(clippy::too_many_arguments)]
fn fit_form_cols(
    p: &SProg,
    probes: &[(Vec<f64>, f64)],
    func: Func,
    mul: Option<usize>,
    pow: i32,
    nargs: usize,
    cands: &[(f64, u8)],
    cols: Vec<usize>,
) -> Option<RealFn> {
    let h_of = |a: &[f64], g: f64| -> Option<f64> {
        match mul {
            Some(j) if a[j] != 0.0 && pow < 0 => Some(g * a[j]),
            Some(j) if a[j] != 0.0 => Some(g / a[j]),
            Some(_) => None,
            None => Some(g),
        }
    };
    let rows: Vec<(Vec<f64>, f64)> = probes
        .iter()
        .filter_map(|(a, g)| {
            let h = h_of(a, *g)?;
            // exp2: the underflow range carries no information on the exponent.
            if func == Func::Exp2 && h.abs() < 2f64.powi(-120) {
                return None;
            }
            func.inv(h).map(|z| (a.clone(), z))
        })
        .collect();
    if rows.len() < cols.len() + 2 || rows.len() * 2 < probes.len() {
        return None;
    }
    // The intercept from the slice at the origin (the other arguments 0, the multiplier 1),
    // which is exact where the fit's is not (eps << the linear part).
    let mut at0: Vec<u32> = vec![0; nargs];
    if let Some(j) = mul {
        at0[j] = 1f32.to_bits();
    }
    let c0_fixed = p
        .eval(&at0)
        .ok()
        .map(|b| f32::from_bits(b) as f64)
        .and_then(|g| func.inv(g));
    let mut best: Option<RealFn> = None;
    // A single-argument form also tries a zero intercept (1 / a, a slice whose value at the
    // origin is not on the form, e.g. a guarded reciprocal).
    let zero = (cols.len() == 1 && nargs > 1 && c0_fixed.is_none()).then_some(0.0);
    for fixed in [c0_fixed, zero, None] {
        if fixed.is_none() && zero.is_some() && best.is_some() {
            continue;
        }
        let (c0, cs) = match fixed {
            Some(c0) => {
                let rr: Vec<(Vec<f64>, f64)> =
                    rows.iter().map(|(a, z)| (a.clone(), z - c0)).collect();
                let (_, cs) = lsq(&rr, &cols, false)?;
                (c0, cs)
            }
            None => lsq(&rows, &cols, true)?,
        };
        let mut coef = vec![0.0; nargs];
        for (j, &c) in cols.iter().enumerate() {
            coef[c] = cs[j];
        }
        let mut r = RealFn {
            func: func.clone(),
            coef,
            c0,
            mul,
            pow,
            outer: (0.0, 1.0),
            parg: 0,
            snapped: false,
            fit_err: 0.0,
            probes: probes.len(),
        };
        r.fit_err = max_err(&r, probes);
        if r.fit_err.is_nan() || r.fit_err > ACCEPT {
            continue;
        }
        let s = snap_form(&r, probes, cands);
        if best
            .as_ref()
            .is_none_or(|b| (!s.snapped, s.fit_err) < (!b.snapped, b.fit_err))
        {
            best = Some(s);
        }
        if best.as_ref().is_some_and(|b| b.snapped) {
            break;
        }
    }
    best
}

/// Distinct values of argument `k` over the probes.
fn distinct(probes: &[(Vec<f64>, f64)], k: usize) -> usize {
    let mut v: Vec<u64> = probes.iter().map(|p| p.0[k].to_bits()).collect();
    v.sort_unstable();
    v.dedup();
    v.len()
}

/// Fewest distinct argument values for a fitted named or polynomial form: with only a few,
/// any two-parameter family fits them.
const MIN_DISTINCT: usize = 8;

/// Fit `alpha + beta * named(c a_0)` for one-argument slices.
fn fit_named(probes: &[(Vec<f64>, f64)], cands: &[(f64, u8)], k: usize) -> Vec<RealFn> {
    let mut out = Vec::new();
    if distinct(probes, k) < MIN_DISTINCT {
        return out;
    }
    let n = probes.first().map_or(1, |p| p.0.len());
    let rpi = (2.0 / std::f64::consts::PI).sqrt();
    for &name in NAMED {
        for c in [
            1.0,
            -1.0,
            std::f64::consts::FRAC_1_SQRT_2,
            -std::f64::consts::FRAC_1_SQRT_2,
            2.0,
            -2.0,
            0.5,
            rpi,
        ] {
            let rows: Vec<(Vec<f64>, f64)> = probes
                .iter()
                .filter_map(|(a, g)| {
                    let f = named(name, c * a[k]);
                    f.is_finite().then(|| (vec![f], *g))
                })
                .collect();
            if rows.len() * 10 < probes.len() * 9 {
                continue;
            }
            let Some((al, be)) = lsq(&rows, &[0], true) else {
                continue;
            };
            let mut r = RealFn::plain(Func::Named(name.into()), n);
            r.coef[k] = c;
            r.outer = (al, be[0]);
            r.probes = probes.len();
            r.fit_err = max_err(&r, probes);
            // MUFU.TANH-based forms are within 2^-10.99 relative.
            if r.fit_err > 2f64.powi(-9) {
                continue;
            }
            out.push(snap_form(&r, probes, cands));
        }
    }
    out
}

/// Fit a polynomial of degree <= 3 in argument k (pieces of activations: 0.044715 x^2).
fn fit_poly(probes: &[(Vec<f64>, f64)], cands: &[(f64, u8)], k: usize) -> Vec<RealFn> {
    let n = probes.first().map_or(1, |p| p.0.len());
    let mut out = Vec::new();
    for deg in (1..=3usize).filter(|d| distinct(probes, k) >= d + 3) {
        let rows: Vec<(Vec<f64>, f64)> = probes
            .iter()
            .map(|(a, g)| ((1..=deg).map(|i| a[k].powi(i as i32)).collect(), *g))
            .collect();
        let cols: Vec<usize> = (0..deg).collect();
        let Some((c0, cs)) = lsq(&rows, &cols, true) else {
            continue;
        };
        let mut r = RealFn::plain(Func::Poly, n);
        r.coef = cs;
        r.c0 = c0;
        r.parg = k;
        r.probes = probes.len();
        r.fit_err = max_err(&r, probes);
        if r.fit_err > ACCEPT {
            continue;
        }
        // Snap: coefficients relative, the rest of the poly's terms zero where tiny.
        let mut t = r.clone();
        let mut all = true;
        let scale: f64 = probes.iter().map(|p| p.1.abs()).fold(0.0, f64::max);
        let amax: f64 = probes.iter().map(|p| p.0[k].abs()).fold(0.0, f64::max);
        for i in 0..deg {
            if (t.coef[i] * amax.powi(i as i32 + 1)).abs() <= 2f64.powi(-24) * scale {
                t.coef[i] = 0.0;
                continue;
            }
            match snap(t.coef[i], cands, 2f64.powi(-14) * t.coef[i].abs()) {
                Some(v) => t.coef[i] = v,
                None => all = false,
            }
        }
        match snap(t.c0, cands, 2f64.powi(-20) * scale) {
            Some(v) => t.c0 = v,
            None => all = false,
        }
        t.fit_err = max_err(&t, probes);
        if t.fit_err <= 4.0 * r.fit_err + 2f64.powi(-23) {
            t.snapped = all;
            out.push(t);
        } else {
            out.push(r);
        }
        break;
    }
    out
}

/// Identify one slice from its probes (argument tuples and outputs, f64).
pub fn identify(p: &SProg, probes: &[(Vec<f64>, f64)]) -> Result<RealFn, String> {
    let n = p.args.len();
    let probes: Vec<(Vec<f64>, f64)> = probes
        .iter()
        .filter(|(a, g)| g.is_finite() && a.iter().all(|x| x.is_finite()))
        .cloned()
        .collect();
    if std::env::var("PTXV_IDENT_DEBUG").is_ok() {
        eprintln!(
            "slice @{} {} args, {} ops:",
            p.ops.first().map_or(0, |o| o.pc),
            n,
            p.ops.len()
        );
        for (a, g) in probes.iter().take(6) {
            eprintln!("    probe {a:?} -> {g:e}");
        }
    }
    if probes.len() < 4 {
        return Err(format!("{} finite probes", probes.len()));
    }
    let cands = snap_candidates(&slice_consts(p));
    if n == 2 {
        let mut r = RealFn::plain(Func::Div, 2);
        r.probes = probes.len();
        r.fit_err = max_err(&r, &probes);
        if r.fit_err <= 2f64.powi(-20) {
            return Ok(r);
        }
    }
    let mut found: Vec<RealFn> = Vec::new();
    for f in [
        Func::Exp2,
        Func::Rcp,
        Func::Rsqrt,
        Func::Sqrt,
        Func::Tanh,
        Func::Ident,
    ] {
        if let Some(r) = fit_form(p, &probes, f.clone(), None, 1, n, &cands) {
            found.push(r);
        }
        if n >= 2 {
            for j in 0..n {
                for pow in [1, -1] {
                    if let Some(r) = fit_form(p, &probes, f.clone(), Some(j), pow, n, &cands) {
                        found.push(r);
                    }
                }
            }
        }
    }
    // One argument at a time (dependent arguments).
    if n >= 2 && !found.iter().any(|r| r.snapped) {
        for f in [
            Func::Exp2,
            Func::Rcp,
            Func::Rsqrt,
            Func::Sqrt,
            Func::Tanh,
            Func::Ident,
        ] {
            for k in 0..n {
                if let Some(r) = fit_form_cols(p, &probes, f.clone(), None, 1, n, &cands, vec![k]) {
                    found.push(r);
                }
            }
        }
    }
    if !found.iter().any(|r| r.snapped) {
        for k in 0..n {
            found.extend(fit_named(&probes, &cands, k));
            found.extend(fit_poly(&probes, &cands, k));
        }
    }
    // Prefer snapped forms, then the smallest deviation, then the simplest.
    found.sort_by(|a, b| {
        let w = |r: &RealFn| {
            (
                !r.snapped as u8,
                r.fit_err.max(2f64.powi(-26)),
                r.mul.is_some() as u8 + matches!(r.func, Func::Named(_)) as u8,
            )
        };
        let (x, y) = (w(a), w(b));
        x.0.cmp(&y.0).then(x.1.total_cmp(&y.1)).then(x.2.cmp(&y.2))
    });
    found
        .into_iter()
        .next()
        .ok_or_else(|| format!("no form fits the slice ({n} args, {} probes)", probes.len()))
}

/// One slice function's identification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IdentInfo {
    pub result: Result<RealFn, String>,
    /// Ops computing it, and the pc of the first one's first instruction.
    pub count: usize,
    pub pc: u32,
    pub prog: std::sync::Arc<SProg>,
    /// The probes it was identified on.
    pub probes: Vec<(Vec<f64>, f64)>,
}

/// Identification results of a program's slices, by `func_key`.
pub type Idents = BTreeMap<String, IdentInfo>;

/// Identify every slice of a program, probing it on the captured input (`base`, its leaf
/// values) and `n` perturbed
/// inputs (each leaf times exp(N(0, 1/2)) with a random sign, rounded to its format), and
/// store the functions in the ops. Returns the per-slice results.
pub fn identify_prog(
    p: &mut super::super::chain::Prog,
    base: &super::super::chain::LeafVals,
    n: usize,
    seed: u64,
) -> Idents {
    use super::super::chain::Op;
    use super::ops::NlOp;
    let mut groups: HashMap<String, Vec<u32>> = HashMap::default();
    for (i, op) in p.ops.iter().enumerate() {
        if let Op::Nl(NlOp::Seg { prog, .. }) = op {
            groups.entry(func_key(prog)).or_default().push(i as u32);
        }
    }
    let mut out = Idents::new();
    if groups.is_empty() {
        return out;
    }
    let mut probes: HashMap<String, Vec<(Vec<f64>, f64)>> = HashMap::default();
    let mut rng = super::super::special::Rng::new(seed ^ 0x1de7);
    let cap = 6000usize;
    for t in 0..=n {
        let mut w = base.clone();
        if t > 0 {
            for (i, op) in p.ops.iter().enumerate() {
                if let Op::Leaf { fmt, .. } = op {
                    let v = w[i].v;
                    let f = (0.5 * rng.normal()).exp() * if rng.below(2) == 0 { 1.0 } else { -1.0 };
                    let x = if v == 0.0 { rng.normal() } else { v * f };
                    let r = match fmt {
                        super::super::spec::Format::F32 => x as f32 as f64,
                        super::super::spec::Format::Bf16 | super::super::spec::Format::F16 => {
                            fmt.round(x, false).0
                        }
                        _ => continue,
                    };
                    w[i] = crate::mma_lp::Operand::from_value(r);
                }
            }
        }
        p.eval_into(&mut w);
        for (k, ids) in &groups {
            let e = probes.entry(k.clone()).or_default();
            // A random subset of the group's instances per input.
            let per = (cap / (n + 1)).max(1);
            for _ in 0..per.min(ids.len()) {
                let i = ids[rng.below(ids.len() as u64) as usize];
                if let Op::Nl(NlOp::Seg { args, .. }) = &p.ops[i as usize] {
                    let a: Vec<f64> = args
                        .iter()
                        .map(|&j| w[j as usize].v as f32 as f64)
                        .collect();
                    e.push((a, w[i as usize].v as f32 as f64));
                }
            }
        }
    }
    for (k, ids) in &groups {
        let Op::Nl(NlOp::Seg { prog, .. }) = &p.ops[ids[0] as usize] else {
            continue;
        };
        let prog = prog.clone();
        let r = identify(&prog, &probes[k]);
        if let Ok(f) = &r {
            for &i in ids {
                if let Op::Nl(NlOp::Seg { real, .. }) = &mut p.ops[i as usize] {
                    *real = Some(f.clone());
                }
            }
        }
        out.insert(
            k.clone(),
            IdentInfo {
                result: r,
                count: ids.len(),
                pc: prog.ops.first().map_or(0, |o| o.pc),
                prog,
                probes: probes.remove(k).unwrap_or_default(),
            },
        );
    }
    out
}

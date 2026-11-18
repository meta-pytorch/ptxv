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
    /// Exact maximum in the ordered, non-NaN real interpretation of a checked slice.
    Max,
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
            Func::Max => "max".into(),
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
            Func::Div | Func::Max => f64::NAN,
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
            Func::Div | Func::Max => f64::NAN,
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
    /// The argument of a polynomial form, or the selected argument on a maximum tie.
    #[serde(default)]
    pub parg: usize,
    /// Every constant snapped to a named value (else some are least-squares fits).
    pub snapped: bool,
    /// Largest relative deviation of the slice from this form on the probes.
    pub fit_err: f64,
    pub probes: usize,
    /// The data-dependent decisions the probes took (MISS 57). None: the slice takes none,
    /// or the probes took every pattern of them.
    #[serde(default)]
    pub outcomes: Option<Outcomes>,
    /// The roundings that act as steps on the probes, with the lanes that do (op index, bit
    /// mask; `SProg::rounding_steps`): their bits are decisions too (MISS 99 follow-up).
    #[serde(default)]
    pub steps: Vec<(u32, u32)>,
    /// Forms of another shape (`shape`: function, multiplier, arguments read) with exact
    /// constants that fit the probes about as well as this one, within EXACT_VETO of its
    /// deviation plus a rounding (MISS 147): the probes do not tell them apart, so where they
    /// disagree beyond that margin the slice's exact value is not established (`agrees`,
    /// `real::eval_table`).
    #[serde(default)]
    pub alts: Vec<RealFn>,
}

/// The patterns of a slice's data-dependent decisions (`SProg::decisions`: the predicates it
/// reads that depend on its arguments) that its identification's probes took (MISS 57). The
/// identified function is the slice's real counterpart only where the slice takes one of
/// them: a guarded instruction or a select whose other outcome no probe took is another
/// function there (ASSUME(N-ERRDIST-IDENT)).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Outcomes {
    /// Decisions per pattern.
    pub n: u32,
    /// The patterns taken, 64 decisions per word, sorted.
    pub seen: Vec<Vec<u64>>,
}

impl Outcomes {
    fn pack(d: &[bool]) -> Vec<u64> {
        let mut w = vec![0u64; d.len().div_ceil(64)];
        for (i, &b) in d.iter().enumerate() {
            w[i / 64] |= (b as u64) << (i % 64);
        }
        w
    }

    /// The outcomes of the patterns `seen` of `n` decisions (None when there is no decision
    /// or every pattern was taken: the function holds wherever the slice goes).
    pub fn of(n: usize, seen: &[Vec<bool>]) -> Option<Outcomes> {
        let mut w: Vec<Vec<u64>> = seen
            .iter()
            .filter(|d| d.len() == n)
            .map(|d| Self::pack(d))
            .collect();
        w.sort();
        w.dedup();
        if n == 0 || (n < 64 && w.len() as u64 == 1u64 << n) {
            return None;
        }
        Some(Outcomes {
            n: n as u32,
            seen: w,
        })
    }

    /// Whether decisions `d` are a pattern the probes took.
    pub fn covers(&self, d: &[bool]) -> bool {
        d.len() == self.n as usize && self.seen.binary_search(&Self::pack(d)).is_ok()
    }

    /// How many of the 2^n patterns the probes took, as text.
    pub fn show(&self) -> String {
        if self.n < 64 {
            format!("{} of {}", self.seen.len(), 1u64 << self.n)
        } else {
            format!("{} of 2^{}", self.seen.len(), self.n)
        }
    }
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
            outcomes: None,
            steps: vec![],
            alts: vec![],
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
    pub(crate) fn maximum_arg(&self, x: f64, y: f64) -> usize {
        if x > y {
            0
        } else if y > x {
            1
        } else {
            self.parg
        }
    }

    pub fn eval(&self, a: &[f64]) -> f64 {
        if self.func == Func::Max {
            return if a[0].is_nan() || a[1].is_nan() {
                f64::NAN
            } else {
                a[self.maximum_arg(a[0], a[1])]
            };
        }
        if self.func == Func::Div {
            return a[0] / a[1];
        }
        self.outer.0 + self.outer.1 * self.factor(a) * self.func.f(self.z(a))
    }
    /// Whether every alternative (`alts`, MISS 147) agrees with this form's value `v` at the
    /// arguments `a`: the same value, or within EXACT_VETO times the two forms' deviations on
    /// the probes plus a rounding (relative, with `max_err`'s floor). On the probes they agree
    /// within the sum of the deviations; beyond them a curve that is linear on the probed
    /// range (exp2, tanh) leaves the line it matched there. NaN from both (an argument whose
    /// exact value is not determined) agrees: the output is left out by that cause.
    pub fn agrees(&self, a: &[f64], v: f64) -> bool {
        self.alts.iter().all(|alt| {
            let w = alt.eval(a);
            if w == v || (w.is_nan() && v.is_nan()) {
                return true;
            }
            let floor = form_floor(self).max(form_floor(alt));
            let d = (w - v).abs() / v.abs().max(w.abs()).max(floor).max(1e-300);
            d <= EXACT_VETO * (self.fit_err + alt.fit_err) + 2f64.powi(-23)
        })
    }
    /// Partial derivatives with respect to each argument.
    pub fn grad(&self, a: &[f64]) -> Vec<f64> {
        if self.func == Func::Max {
            return if a[0].is_nan() || a[1].is_nan() {
                vec![f64::NAN, f64::NAN]
            } else {
                let mut gradient = vec![0.0, 0.0];
                gradient[self.maximum_arg(a[0], a[1])] = 1.0;
                gradient
            };
        }
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
        if self.func == Func::Max {
            return "max(a0, a1)".into();
        }
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

/// The f32 constants a slice reads (immediates and constant-bank values), finite and nonzero.
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
    out.retain(|x| x.is_finite() && *x != 0.0);
    out
}

/// Snap candidates (positive), with a priority: 0 for exact named constants
/// ({1, log2 e, ln 2, sqrt 2, ...} x 2^k, decimal epsilons), 1 for the slice's own f32
/// constants and their products with log2 e and ln 2, 2 for products of two of them (the
/// real arithmetic multiplies them: y = (x 2^100) C + 2 is 2 + (2^100 C) x). Since MISS 138
/// a form with fitted constants is no exact reference, and such a slope did not snap.
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
        if c <= 1e-30 || c >= 1e30 {
            continue;
        }
        for v in [c, c * l2e, c * ln2, c / l2e] {
            out.push((v, 1));
        }
    }
    let mut cs: Vec<f64> = consts.iter().map(|c| c.abs()).collect();
    cs.sort_by(f64::total_cmp);
    cs.dedup();
    cs.truncate(32);
    for (i, &a) in cs.iter().enumerate() {
        for &b in &cs[i..] {
            let v = a * b;
            if v > 1e-30 && v < 1e30 {
                out.push((v, 2));
            }
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

/// The floor of a form's relative deviation (`max_err`, `RealFn::agrees`). A named form's is a
/// quarter of its outer scale: 1 + tanh(z) built on MUFU.TANH (2^-11 absolute) has large
/// relative errors where it cancels.
fn form_floor(r: &RealFn) -> f64 {
    match r.func {
        Func::Named(_) => floor_of(&r.func).max(r.outer.1.abs() * 0.25),
        _ => floor_of(&r.func),
    }
}

/// Largest relative deviation of the probes' outputs from the form.
fn max_err(r: &RealFn, probes: &[(Vec<f64>, f64)]) -> f64 {
    let floor = form_floor(r);
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
            outcomes: None,
            steps: vec![],
            alts: vec![],
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

/// Whether a form reads argument `k` (`arg_dependence`).
fn reads_arg(r: &RealFn, k: usize) -> bool {
    match r.func {
        Func::Div => k < 2,
        Func::Poly => r.parg == k && r.coef.iter().any(|c| *c != 0.0),
        _ => r.mul == Some(k) || r.coef.get(k).is_some_and(|c| *c != 0.0),
    }
}

/// The arguments the slice's output depends on over the probes (MISS 138): argument k is one
/// when giving a probe another probe's value of it changes the slice's output bits. A form
/// that leaves such an argument out drops a term of the slice however small the term is
/// against the output: the acceptance bound (2^-11 relative) is thousands of output ulps, and
/// `y = RM(floor(x / 1024) + u)` (a FADD.RM floor, u near 2^23) was fitted as y = u, the floor
/// 1-11 ulps on the probes and up to 2000 on inputs beyond them.
fn arg_dependence(p: &SProg, probes: &[(Vec<f64>, f64)]) -> Vec<bool> {
    let m = probes.len();
    let mut dep = vec![false; p.args.len()];
    if m < 2 {
        return dep;
    }
    let same =
        |x: u32, y: u32| x == y || (f32::from_bits(x).is_nan() && f32::from_bits(y).is_nan());
    for (k, d) in dep.iter_mut().enumerate() {
        // Each of the first probes against one from the other half.
        for i in 0..m.min(512) {
            let (a, b) = (&probes[i].0, &probes[(i + m / 2 + 1) % m].0);
            let v = (b[k] as f32).to_bits();
            if (a[k] as f32).to_bits() == v {
                continue;
            }
            let mut x: Vec<u32> = a.iter().map(|&y| (y as f32).to_bits()).collect();
            let Ok(y0) = p.eval(&x) else { continue };
            x[k] = v;
            let Ok(y1) = p.eval(&x) else { continue };
            if !same(y0, y1) {
                *d = true;
                break;
            }
        }
    }
    dep
}

/// Whether every instruction of the slice is an FP add, multiply or fused multiply-add (FADD, FMUL,
/// FFMA, FFMA2; a guard picks between such values): every rounding the identity, its real function
/// is then a polynomial in its arguments on each decision pattern (MISS 147), and a curved form
/// (exp2, tanh, a reciprocal, a root, a quotient, a named function) can match it only where the
/// curve is linear on the probes, as tanh(2^-64 x) matched an exact scaling y = RN(x 2^-64).
fn polynomial(p: &SProg) -> bool {
    p.ops.iter().all(|o| match &o.kind {
        SKind::Inst { op, .. } => matches!(
            op,
            ptxv_ir::Op::Ffma { .. } | ptxv_ir::Op::Fadd { .. } | ptxv_ir::Op::Ffma2 { .. }
        ),
        SKind::Guard { .. } => true,
        SKind::DivRn { .. } => false,
    })
}

/// Keep a fitted form if it reads every argument the slice depends on (`arg_dependence`);
/// else remember the best such form for the refusal's text.
fn admit(r: RealFn, dep: &[bool], found: &mut Vec<RealFn>, dropped: &mut Option<RealFn>) {
    if (0..dep.len()).all(|k| !dep[k] || reads_arg(&r, k)) {
        found.push(r);
    } else if dropped
        .as_ref()
        .is_none_or(|d| (!r.snapped, r.fit_err) < (!d.snapped, d.fit_err))
    {
        *dropped = Some(r);
    }
}

/// Identify one slice from its probes (argument tuples and outputs, f64).
pub fn identify(p: &SProg, probes: &[(Vec<f64>, f64)]) -> Result<RealFn, String> {
    if crate::seg::real_maximum(p).is_some() {
        let mut form = RealFn::plain(Func::Max, 2);
        form.parg = match p.eval(&[0, 0x8000_0000])? {
            0 => 0,
            0x8000_0000 => 1,
            _ => return Err("checked maximum did not select an argument on a zero tie".into()),
        };
        return Ok(form);
    }
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
    // MISS 147: a slice of FP adds, multiplies and FMAs takes polynomial forms only.
    let poly = polynomial(p);
    let funcs: Vec<Func> = if poly {
        vec![Func::Ident]
    } else {
        vec![
            Func::Exp2,
            Func::Rcp,
            Func::Rsqrt,
            Func::Sqrt,
            Func::Tanh,
            Func::Ident,
        ]
    };
    let pows: &[i32] = if poly { &[1] } else { &[1, -1] };
    if n == 2 && !poly {
        let mut r = RealFn::plain(Func::Div, 2);
        r.probes = probes.len();
        r.fit_err = max_err(&r, &probes);
        if r.fit_err <= 2f64.powi(-20) {
            return Ok(r);
        }
    }
    // MISS 138: a form must read every argument the slice depends on.
    let dep = arg_dependence(p, &probes);
    let (mut found, mut dropped): (Vec<RealFn>, Option<RealFn>) = (Vec::new(), None);
    for f in &funcs {
        if let Some(r) = fit_form(p, &probes, f.clone(), None, 1, n, &cands) {
            admit(r, &dep, &mut found, &mut dropped);
        }
        if n >= 2 {
            for j in 0..n {
                for &pow in pows {
                    if let Some(r) = fit_form(p, &probes, f.clone(), Some(j), pow, n, &cands) {
                        admit(r, &dep, &mut found, &mut dropped);
                    }
                }
            }
        }
    }
    // Over the arguments the slice depends on: where some are functionally dependent on the
    // others (a sum and its terms), the fit over all of them is singular, and one argument at
    // a time (below) leaves the others out.
    let dcols: Vec<usize> = (0..n).filter(|&k| dep[k]).collect();
    if n >= 2 && !dcols.is_empty() && dcols.len() < n && !found.iter().any(|r| r.snapped) {
        for f in &funcs {
            if let Some(r) = fit_form_cols(p, &probes, f.clone(), None, 1, n, &cands, dcols.clone())
            {
                admit(r, &dep, &mut found, &mut dropped);
            }
            for &j in &dcols {
                let cols: Vec<usize> = dcols.iter().copied().filter(|&k| k != j).collect();
                for &pow in pows {
                    if let Some(r) =
                        fit_form_cols(p, &probes, f.clone(), Some(j), pow, n, &cands, cols.clone())
                    {
                        admit(r, &dep, &mut found, &mut dropped);
                    }
                }
            }
        }
    }
    // One argument at a time (dependent arguments).
    if n >= 2 && !found.iter().any(|r| r.snapped) {
        for f in &funcs {
            for k in 0..n {
                if let Some(r) = fit_form_cols(p, &probes, f.clone(), None, 1, n, &cands, vec![k]) {
                    admit(r, &dep, &mut found, &mut dropped);
                }
            }
        }
    }
    if !found.iter().any(|r| r.snapped) {
        for k in 0..n {
            let named = if poly {
                Vec::new()
            } else {
                fit_named(&probes, &cands, k)
            };
            for r in named.into_iter().chain(fit_poly(&probes, &cands, k)) {
                admit(r, &dep, &mut found, &mut dropped);
            }
        }
    }
    choose(found, dropped.as_ref(), &dep, probes.len())
}

/// The identified form among the fitted ones (`found`, each reading every argument the slice
/// depends on, `dep`): the closest with exact (snapped) constants, unless one with fitted
/// constants fits EXACT_VETO times better (MISS 138). Until MISS 138 any snapped form ranked
/// above every fitted one, so `y = u` (2.76e-6 relative) won over `u + c x` (about 2.5e-7) and
/// dropped a floor of x; and a fitted form stood where no snapped one fit, though a fit is not
/// an exact reference. `dropped`: the closest form that leaves out a dependent argument, for
/// the refusal's text.
fn choose(
    mut found: Vec<RealFn>,
    dropped: Option<&RealFn>,
    dep: &[bool],
    probes: usize,
) -> Result<RealFn, String> {
    let n = dep.len();
    // The smallest deviation, then the simplest.
    found.sort_by(|a, b| {
        let w = |r: &RealFn| {
            (
                r.fit_err.max(2f64.powi(-26)),
                r.mul.is_some() as u8 + matches!(r.func, Func::Named(_)) as u8,
            )
        };
        let (x, y) = (w(a), w(b));
        x.0.total_cmp(&y.0).then(x.1.cmp(&y.1))
    });
    let show = |r: &RealFn| format!("{} [max rel dev {:.2e}]", r.show(), r.fit_err);
    let left_out = |r: &RealFn| -> String {
        (0..n)
            .filter(|&k| dep[k] && !reads_arg(r, k))
            .map(|k| format!("a{k}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let best = found.iter().find(|r| r.snapped);
    let fitted = found.iter().find(|r| !r.snapped);
    let also = dropped.map_or(String::new(), |d| {
        format!(
            "; {} leaves out {}, on which the slice depends",
            show(d),
            left_out(d)
        )
    });
    match (best, fitted) {
        (Some(b), Some(f)) if EXACT_VETO * f.fit_err + 2f64.powi(-23) < b.fit_err => Err(format!(
            "{} (constants fitted) fits the slice better than any form with exact constants, {}: that form drops a term of the slice, and a fit is not an exact reference{also} (MISS 138)",
            show(f),
            show(b)
        )),
        (Some(b), _) => {
            let mut b = b.clone();
            b.alts = alternatives(&found, &b, n);
            Ok(b)
        }
        (None, Some(f)) => Err(format!(
            "no form with exact constants fits the slice ({n} args, {probes} probes); the closest, {} with fitted constants, is not an exact reference{also} (MISS 138)",
            show(f)
        )),
        (None, None) => Err(match dropped {
            Some(d) => format!(
                "every form that fits leaves out an argument the slice depends on ({}: another probe's value changes its output), e.g. {} (MISS 138)",
                left_out(d),
                show(d)
            ),
            None => format!("no form fits the slice ({n} args, {probes} probes)"),
        }),
    }
}

/// A form's shape apart from its constants (MISS 147): function, multiplier and power, the
/// polynomial's argument, the arguments read (of `n`).
fn shape(r: &RealFn, n: usize) -> (String, Option<usize>, i32, usize, Vec<bool>) {
    (
        r.func.name(),
        r.mul,
        r.pow,
        if r.func == Func::Poly { r.parg } else { 0 },
        (0..n).map(|k| reads_arg(r, k)).collect(),
    )
}

/// The alternatives of the identified form `b` among the fitted forms `found` (sorted, best
/// first; each reads every argument the slice depends on): the best form with exact constants
/// of each other shape that fits the probes within EXACT_VETO of `b`'s deviation plus a
/// rounding, at most MAX_ALTS (MISS 147). Until MISS 147 the first of equally good forms stood
/// alone, in the candidates' order: tanh(2^-64 x) for y = RN(x 2^-64), which beyond the probes
/// reported P(differ) 1 for 0. Only forms that could be the exact reference count: one with
/// fitted constants cannot (MISS 138), and least squares gives such a form a coefficient of
/// rounding size on an argument the output does not move with (-1.6e-11 a0 beside 8.84e-2 a1
/// in a kernel_db attention backward), which left out every output where that argument is
/// large.
fn alternatives(found: &[RealFn], b: &RealFn, n: usize) -> Vec<RealFn> {
    let tol = EXACT_VETO * b.fit_err + 2f64.powi(-23);
    let mut seen = vec![shape(b, n)];
    let mut out = Vec::new();
    for r in found.iter().filter(|r| r.snapped && r.fit_err <= tol) {
        let s = shape(r, n);
        if seen.contains(&s) {
            continue;
        }
        seen.push(s);
        let mut a = r.clone();
        a.alts.clear();
        out.push(a);
        if out.len() == MAX_ALTS {
            break;
        }
    }
    out
}

/// At most this many alternatives per identified form (`alternatives`).
const MAX_ALTS: usize = 4;

/// A form with exact constants is not the exact reference when one with fitted constants fits
/// the slice this many times better (beyond 2^-23 relative, a rounding's worth): the margin
/// `snap_form` allows a snap.
const EXACT_VETO: f64 = 4.0;

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
    // Per group: the argument bits of each finite probe (the probes `identify` fits), whose
    // decisions the identified function holds on.
    let mut finite: HashMap<String, Vec<Vec<u32>>> = HashMap::default();
    let mut scratch = Vec::new();
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
                    let g = w[i as usize].v as f32 as f64;
                    if g.is_finite() && a.iter().all(|x| x.is_finite()) {
                        let bits: Vec<u32> = a.iter().map(|&x| (x as f32).to_bits()).collect();
                        finite.entry(k.clone()).or_default().push(bits);
                    }
                    e.push((a, g));
                }
            }
        }
    }
    for (k, ids) in &groups {
        let Op::Nl(NlOp::Seg { prog, .. }) = &p.ops[ids[0] as usize] else {
            continue;
        };
        let prog = prog.clone();
        let mut r = identify(&prog, &probes[k]);
        if let Ok(f) = r.as_mut() {
            // The roundings that act as steps on these probes join the decisions (MISS 99
            // follow-up), then the patterns the probes took.
            let fin = finite.get(k).map_or(&[][..], |v| v.as_slice());
            f.steps = prog.rounding_steps(fin);
            let taken: Vec<Vec<bool>> = fin
                .iter()
                .filter_map(|b| prog.decisions(b, &f.steps, &mut scratch))
                .collect();
            f.outcomes = Outcomes::of(prog.decision_count(&f.steps), &taken);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn form(snapped: bool, fit_err: f64) -> RealFn {
        let mut r = RealFn::plain(Func::Ident, 1);
        r.coef[0] = if snapped { 1.0 } else { 1.000122 };
        r.snapped = snapped;
        r.fit_err = fit_err;
        r
    }

    /// MISS 138: a form with exact constants that a fitted one fits EXACT_VETO times better
    /// (beyond a rounding's 2^-23) is no exact reference: it drops a term of the slice (rE_fs3:
    /// y = u at 2.76e-6 against u + c x at about 2.5e-7). Within the margin it stands.
    #[test]
    fn a_much_better_fitted_form_vetoes_an_exact_one() {
        let dep = [true];
        let r = choose(
            vec![form(true, 2.76e-6), form(false, 2.5e-7)],
            None,
            &dep,
            100,
        );
        assert!(
            r.as_ref().is_err_and(|e| e.contains("drops a term")),
            "{r:?}"
        );
        // A rounding's worth: 4 x 1e-7 + 2^-23 is above 2.4e-7.
        let r = choose(vec![form(true, 2.4e-7), form(false, 1e-7)], None, &dep, 100);
        assert!(r.is_ok_and(|f| f.snapped));
        let r = choose(vec![form(true, 1e-6), form(false, 3e-7)], None, &dep, 100);
        assert!(r.is_ok_and(|f| f.snapped));
    }

    /// MISS 147: the best form of each other shape that fits within EXACT_VETO of the
    /// identified one, plus a rounding, is its alternative, and where they disagree beyond that
    /// margin the value is not established: tanh(2^-64 x) and 2^-64 x take the same values near
    /// x ~ 1 and not at x = 2^64. A form of the same shape is no second alternative, nor is one
    /// that fits more than EXACT_VETO times worse.
    #[test]
    fn forms_the_probes_cannot_tell_apart_are_alternatives() {
        let c = 2f64.powi(-64);
        let mut t = RealFn::plain(Func::Tanh, 1);
        t.coef[0] = c;
        let mut l = RealFn::plain(Func::Ident, 1);
        l.coef[0] = c;
        let mut same = l.clone();
        same.fit_err = 1e-9;
        let f = choose(vec![t.clone(), l, same], None, &[true], 100).unwrap();
        assert_eq!((f.func.clone(), f.alts.len()), (Func::Tanh, 1));
        assert!(f.agrees(&[1.0], f.eval(&[1.0])));
        let x = 2f64.powi(64);
        assert!(!f.agrees(&[x], f.eval(&[x])));
        assert!(f.agrees(&[f64::NAN], f64::NAN));
        t.fit_err = 1e-7;
        let mut e = RealFn::plain(Func::Exp2, 1);
        e.coef[0] = 1.0;
        e.fit_err = 1e-5;
        let f = choose(vec![t.clone(), e], None, &[true], 100).unwrap();
        assert!(f.alts.is_empty());
        // A form with fitted constants, as close, is no alternative: it cannot be the exact
        // reference.
        let mut fitted = RealFn::plain(Func::Ident, 1);
        fitted.coef[0] = c * 1.000001;
        fitted.snapped = false;
        fitted.fit_err = 1e-7;
        let f = choose(vec![t, fitted], None, &[true], 100).unwrap();
        assert!(f.alts.is_empty(), "{:?}", f.alts);
    }

    /// MISS 138: a form with fitted constants alone is not an exact reference.
    #[test]
    fn a_fitted_form_alone_is_no_exact_reference() {
        let r = choose(vec![form(false, 2e-7)], None, &[true], 100);
        assert!(
            r.as_ref()
                .is_err_and(|e| e.contains("not an exact reference")),
            "{r:?}"
        );
    }
}

#[cfg(test)]
mod exact_comparison_maximum_tests {
    use std::sync::Arc;

    use ptxv_ir::FCmp;

    use super::*;
    use crate::errdist::chain::Op;
    use crate::errdist::chain::Prog;
    use crate::errdist::nonlin;
    use crate::errdist::nonlin::lin::Final;
    use crate::errdist::nonlin::lin::KModel;
    use crate::errdist::nonlin::lin::Law;
    use crate::errdist::nonlin::ops::NlOp;
    use crate::errdist::spec::Format;

    fn program(comparison: FCmp) -> SProg {
        let (args, ops, out) =
            crate::seg::comparison_program_parts(crate::seg::ComparisonContext {
                plan: ptxv_ir::comparison_select::Plan {
                    comparison,
                    negate: false,
                    swap: false,
                },
                left: 0,
                right: 1,
                comparison_pc: 3,
                selection_pc: 5,
            });
        SProg {
            args,
            ops,
            out,
            extra_pcs: Default::default(),
        }
    }

    #[test]
    fn maximum_is_identified_structurally_without_fitting_arbitrary_slices() {
        let prog = program(FCmp::Gt);
        let form = identify(&prog, &[]).unwrap();
        assert_eq!(form.func, Func::Max);
        for args in [[-2.0, 3.0], [3.0, -2.0], [2.0, 2.0]] {
            assert_eq!(form.eval(&args), args[0].max(args[1]));
        }
        assert_eq!(form.grad(&[3.0, -2.0]), [1.0, 0.0]);
        assert_eq!(form.grad(&[-2.0, 3.0]), [0.0, 1.0]);
        assert_eq!(form.grad(&[2.0, 2.0]), [0.0, 1.0]);
        assert_eq!(
            identify(&program(FCmp::Ge), &[]).unwrap().grad(&[2.0, 2.0]),
            [1.0, 0.0]
        );
        assert!(form.eval(&[f64::NAN, 3.0]).is_nan());
        assert!(form.eval(&[3.0, f64::NAN]).is_nan());
        assert_eq!(form.eval(&[f64::NEG_INFINITY, 3.0]), 3.0);
        assert!(identify(&program(FCmp::Lt), &[]).is_err());
        assert!(identify(&program(FCmp::Eq), &[]).is_err());
        let mut malformed = prog.clone();
        malformed.extra_pcs.insert(9);
        assert!(identify(&malformed, &[]).is_err());
    }

    #[test]
    fn maximum_keeps_exact_slice_bits_and_serializes_its_real_projection() {
        let prog = program(FCmp::Gt);
        let form = identify(&prog, &[]).unwrap();
        let op = NlOp::Seg {
            prog: Arc::new(prog),
            args: vec![0, 1],
            real: Some(form),
        };
        let restored: NlOp = serde_json::from_str(&serde_json::to_string(&op).unwrap()).unwrap();
        for (x, y) in [
            (0, 0x8000_0000),
            (0x8000_0000, 0),
            (0x7fc0_1234, 0x3f80_0000),
        ] {
            for slice in [&op, &restored] {
                let value =
                    slice.eval(&|i| f32::from_bits(if i == 0 { x } else { y }), &mut vec![]);
                assert_eq!(value.to_bits(), y);
            }
        }
        assert_eq!(op.shape_key(), restored.shape_key());
        let NlOp::Seg {
            real: Some(real), ..
        } = restored
        else {
            panic!()
        };
        assert_eq!(real.func, Func::Max);
    }

    #[test]
    fn maximum_uses_no_sampled_table_and_preserves_the_selected_rounding_site() {
        let prog = Arc::new(program(FCmp::Gt));
        let form = identify(&prog, &[]).unwrap();
        let slice = NlOp::Seg {
            prog: prog.clone(),
            args: vec![0, 1],
            real: Some(form.clone()),
        };
        let p = Prog {
            ops: vec![Op::Const(2.0), Op::Const(5.0), Op::Nl(slice)],
            hw: vec![2f32.to_bits(), 5f32.to_bits(), 5f32.to_bits()],
            outputs: vec![Some(2)],
            out_fmt: Format::F32,
            bufs: vec![],
            dep_untracked: vec![],
            atomic: vec![],
            atomic_src: vec![],
            atomic_free: vec![],
            pre_rounded: vec![],
        };
        let ids = [(
            func_key(&prog),
            IdentInfo {
                result: Ok(form),
                count: 1,
                pc: 3,
                prog,
                probes: vec![],
            },
        )]
        .into();
        let spec = serde_json::from_value(serde_json::json!({"version": 1})).unwrap();
        let tables = nonlin::compile(&[(&p, &ids)], &spec, &[]).unwrap();
        assert!(tables.tables.is_empty());
        let indices = nonlin::op_tables(&p, &tables);
        assert_eq!(indices, [None, None, None]);
        let model = KModel::new(&p, &tables.tables, indices);
        let state = model.state(&[]);
        assert_eq!(state.v[2], 5.0);
        assert!(matches!(state.law[2], Law::Zero));
        assert_eq!(model.final_of(&state, 2), (1, Final::Exact));
        let gradients = model.adjoints(&state, &model.cone(2), &mut vec![]);
        assert!(gradients.contains(&(1, 1.0)));
        assert!(!gradients.iter().any(|&(i, _)| i == 0));
        let values = nonlin::real::eval(&p, &[]);
        assert_eq!(values[2], values[1]);
    }

    #[test]
    fn maximum_ties_preserve_the_actual_selected_rounding_site() {
        let value = 1.1f32 as f64;
        for (comparison, selected, final_kind) in
            [(FCmp::Gt, 2, Final::Exact), (FCmp::Ge, 1, Final::Rn)]
        {
            let program = Arc::new(program(comparison));
            let form = identify(&program, &[]).unwrap();
            let p = Prog {
                ops: vec![
                    Op::Const(value),
                    Op::Round { arg: 0, f16: false },
                    Op::Const(value),
                    Op::Nl(NlOp::Seg {
                        prog: program,
                        args: vec![1, 2],
                        real: Some(form),
                    }),
                ],
                hw: vec![],
                outputs: vec![Some(3)],
                out_fmt: Format::F32,
                bufs: vec![],
                dep_untracked: vec![],
                atomic: vec![],
                atomic_src: vec![],
                atomic_free: vec![],
                pre_rounded: vec![],
            };
            let model = KModel::new(&p, &[], vec![None; 4]);
            let state = model.state(&[]);
            assert_eq!(state.v[1], state.v[2]);
            assert!(matches!(state.law[1], Law::Rn { .. }));
            assert!(matches!(state.law[2], Law::Zero));
            assert_eq!(state.lat[3], state.lat[selected as usize]);
            assert_eq!(model.final_of(&state, 3), (selected, final_kind));
            let gradients = model.adjoints(&state, &model.cone(3), &mut vec![]);
            assert!(gradients.contains(&(selected, 1.0)));
            assert!(!gradients.iter().any(|&(i, _)| i == 3 - selected));
        }
    }

    #[test]
    fn maximum_real_evaluation_uses_double_double_low_bits_before_ties() {
        let program = Arc::new(program(FCmp::Gt));
        let form = identify(&program, &[]).unwrap();
        let p = Prog {
            ops: vec![
                Op::Const(1e20),
                Op::Const(1.0),
                Op::Add {
                    a: 0,
                    b: 1,
                    rz: false,
                    ftz: false,
                },
                Op::Const(1e20),
                Op::Nl(NlOp::Seg {
                    prog: program,
                    args: vec![2, 3],
                    real: Some(form),
                }),
            ],
            hw: vec![],
            outputs: vec![Some(4)],
            out_fmt: Format::F32,
            bufs: vec![],
            dep_untracked: vec![],
            atomic: vec![],
            atomic_src: vec![],
            atomic_free: vec![],
            pre_rounded: vec![],
        };
        let values = nonlin::real::eval(&p, &[]);
        assert_eq!(values[2].0.hi, values[3].0.hi);
        assert_eq!(values[2].0.lo, 1.0);
        assert_eq!(values[3].0.lo, 0.0);
        assert_eq!(values[4], values[2]);
    }
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Normalizations in `ptxv equiv` (docs/equiv-reassoc.md, "Layer norm and RMS norm").
//!
//! In segment mode an output of a layer norm or an RMS norm is a polynomial (sums and products
//! of input leaves, rounded) in the inputs and one nonlinear scalar per row, rho = f(A), with A
//! itself a rounded polynomial of the inputs (the variance plus eps, or the mean square plus
//! eps) and f a power (a^-1/2: rsqrt, 1 / sqrt; also a^-1, a^1/2). Each kernel computes rho as an
//! instruction slice g (an approximate rsqrt, a sqrt and a division, with their range guards);
//! its polynomial prefix (div.rn by a constant, + eps, FFMA) is folded into A here
//! (`Ctx::poly`), and the rest, a slice of one argument, is identified with f by evaluating it on
//! every positive normal f32 (`softmax::sweep`): |g(a) - f(a)| <= r0 f(a).
//!
//! Atomized, an output is P(x, rho) = sum_m c_m x_m rho^(e_m), e_m in {0, 1}. Cor 1a, with the
//! computed rho^ as an exact leaf, bounds each kernel's |K - P_K(x, rho^_K)| by
//! sum_m |x_m| |rho^_K|^e_m g_K,m. The two kernels' A polynomials and output polynomials may
//! differ in their coefficients (Welford's update uses the f32 constant 1/3, a two-pass mean
//! the exact 1/128): the reference's exact polynomials are the pivot, and every coefficient
//! difference is charged as an error. With rho* = A_R(x)^p, the theorem (docs) is
//!
//!   |C - R| <= |P1_R(x)| rho* (eta_R + eta_C) + sum_m |x_m| rho*^e_m H^e_m (g_R,m + g_C,m + |dc_m|)
//!
//! where P1_R is the rho-linear part of R's polynomial, eta_K <= r0_K + (1 + r0_K) c1 e_K is
//! rho^_K's relative error (slice contract and A's error e_K = (sum_m |x_m| (g^A_K,m +
//! |dc^A_m|)) / A_R(x) <= k_A,K u kappa(x), kappa = S_A / A_R, S_A = sum_m |c^A_R,m| |x_m|),
//! valid while e_K <= E_MAX, and H bounds rho^ / rho*. ASSUME(N-NORM): see ASSUMPTIONS.md

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;

use ptxv_ir::*;
use serde::Serialize;

use crate::equiv::Leaf;
use crate::equiv::Mono;
use crate::equiv::Term;
use crate::equiv::add_poly_n;
use crate::equiv::poly_mul;
use crate::num::NodeKind;
use crate::num::NumState;
use crate::num::U32;
use crate::seg::SKind;
use crate::seg::SOp;
use crate::seg::SProg;
use crate::seg::SVal;
use crate::softmax::Contract;

/// Buffer id of the nonlinear scalar rho in atomized polynomials.
pub const ATOM_LEAF: u16 = u16::MAX - 3;
/// The largest relative error of A the compact (kappa-linear) bound covers: it fixes kappa_max.
pub const E_MAX: f64 = 1.0 / 1024.0;
/// The largest relative error of A the exact per-input bound (`NormEval::at`) covers.
pub const E_EVAL: f64 = 0.5;

/// rho^'s largest relative error |rho^ / rho* - 1| for A's relative error e and the slice's
/// contract r0: the extreme corners of (1 +- r0)(1 +- e)^p.
fn eta_exact(p: f64, r0: f64, e: f64) -> f64 {
    [
        (1.0 + r0) * (1.0 - e).powf(p) - 1.0,
        (1.0 + r0) * (1.0 + e).powf(p) - 1.0,
        1.0 - (1.0 - r0) * (1.0 - e).powf(p),
        1.0 - (1.0 - r0) * (1.0 + e).powf(p),
    ]
    .into_iter()
    .fold(0.0, f64::max)
        * (1.0 + 2f64.powi(-40))
}

type Poly = BTreeMap<Mono, Term>;

/// One nonlinear scalar: rho^ = g(A^), g a one-argument slice with g ~ A^p.
#[derive(Clone, Debug)]
pub struct NAtom {
    pub g: SProg,
    pub p: f64,
    pub arg: Poly,
    pub contract: Contract,
}

/// One kernel's numerics (as `softmax::Ctx`).
pub struct Ctx<'a> {
    pub ns: &'a NumState,
    pub buf_id: &'a [u16],
    memo: HashMap<u32, Result<Poly, String>>,
    pub atoms: Vec<NAtom>,
}

#[derive(Clone)]
enum Sym {
    P(Poly),
    /// Not a polynomial of the slice's arguments.
    N,
}

fn konst(v: f64) -> Poly {
    let mut p = Poly::new();
    if v != 0.0 {
        p.insert(
            vec![],
            Term {
                coef: v,
                abs: v.abs(),
                ..Default::default()
            },
        );
    }
    p
}

fn as_const(p: &Poly) -> Option<f64> {
    match p.len() {
        0 => Some(0.0),
        1 => p.get(&vec![]).filter(|t| t.gam == 0.0).map(|t| t.coef),
        _ => None,
    }
}

fn scaled(p: &Poly, k: f64) -> Poly {
    p.iter()
        .map(|(m, t)| {
            (
                m.clone(),
                Term {
                    coef: t.coef * k,
                    mult: t.mult * k.abs(),
                    abs: t.abs * k.abs(),
                    gam: t.gam * k.abs(),
                },
            )
        })
        .collect()
}

fn sum(a: &Poly, b: &Poly) -> Poly {
    let mut out = a.clone();
    add_poly_n(&mut out, b, 0.0, 0);
    out.retain(|_, t| t.coef != 0.0 || t.gam != 0.0);
    out
}

fn round(p: &Poly, eps: f64) -> Poly {
    let mut out = Poly::new();
    add_poly_n(&mut out, p, eps, 1);
    out
}

/// Is a power of two (division by it exact unless the quotient is subnormal)?
fn pow2(c: f64) -> bool {
    c != 0.0 && c.is_finite() && {
        let b = c.abs().to_bits();
        b & ((1u64 << 52) - 1) == 0
    }
}

impl<'a> Ctx<'a> {
    pub fn new(ns: &'a NumState, buf_id: &'a [u16]) -> Ctx<'a> {
        Ctx {
            ns,
            buf_id,
            memo: HashMap::new(),
            atoms: vec![],
        }
    }

    /// Node `n` as a polynomial in the input leaves and atoms (`ATOM_LEAF`, index into
    /// `atoms`), with Cor 1a multipliers (the atom's computed value an exact leaf).
    pub fn poly(&mut self, n: u32) -> Result<Poly, String> {
        if let Some(r) = self.memo.get(&n) {
            return r.clone();
        }
        let r = self.poly_inner(n);
        self.memo.insert(n, r.clone());
        r
    }

    fn poly_inner(&mut self, n: u32) -> Result<Poly, String> {
        let nd = &self.ns.nodes[n as usize];
        match nd.kind.clone() {
            NodeKind::Leaf => Ok(konst(f32::from_bits(nd.hw) as f64)),
            NodeKind::Input((a, i)) => Ok([(vec![(self.buf_id[a], i)], Term::one())].into()),
            NodeKind::Add(a, b, rz) => {
                let (pa, pb) = (self.poly(a)?, self.poly(b)?);
                let exact = as_const(&pa) == Some(0.0) || as_const(&pb) == Some(0.0);
                let eps = if exact {
                    0.0
                } else if rz {
                    2.0 * U32
                } else {
                    U32
                };
                Ok(round(&sum(&pa, &pb), eps))
            }
            NodeKind::Fma {
                a,
                b,
                c,
                neg_ab,
                neg_c,
            } => {
                let (pa, pb) = (self.poly(a)?, self.poly(b)?);
                let mut prod = poly_mul(&pa, &pb);
                if neg_ab {
                    prod = scaled(&prod, -1.0);
                }
                if let Some(c) = c {
                    let pc = self.poly(c)?;
                    prod = sum(&prod, &if neg_c { scaled(&pc, -1.0) } else { pc });
                }
                Ok(round(&prod, U32))
            }
            NodeKind::Seg(p) => self.seg(&p.0.clone()),
            // ASSUME(N-MMA): one tensor-core step, its contract eps on every child.
            NodeKind::Mma { sc, .. } if !sc.is_empty() => Err("a block-scaled MMA step".into()),
            NodeKind::Mma { acc, ops, eps, .. } => {
                let mut s = match acc {
                    Some(a) => self.poly(a)?,
                    None => Poly::new(),
                };
                for o in &ops {
                    let (x, y) = o.ok_or("an MMA product of untracked origin")?;
                    let (px, py) = (self.poly(x)?, self.poly(y)?);
                    s = sum(&s, &poly_mul(&px, &py));
                }
                Ok(round(&s, eps))
            }
            NodeKind::Round { arg, f16 } => {
                // Inside a polynomial only when no product scales its subnormal error.
                let pa = self.poly(arg)?;
                Ok(round(&pa, crate::equiv::round_unit(f16).0))
            }
            k => Err(format!("{k:?} in a normalization")),
        }
    }

    /// A slice: its polynomial when every operation is a rounded sum / product or a division
    /// by a constant; otherwise an atom g(A) with A the one polynomial value its
    /// non-polynomial part reads.
    fn seg(&mut self, p: &SProg) -> Result<Poly, String> {
        let p = &cse(p);
        let mut args = Vec::new();
        for &a in &p.args {
            args.push(self.poly(a).map(Sym::P).unwrap_or(Sym::N));
        }
        let mut vals: Vec<Sym> = Vec::with_capacity(p.ops.len());
        let get = |v: SVal, vals: &[Sym]| -> Sym {
            match v {
                SVal::Arg(i) => args[i as usize].clone(),
                SVal::Const(c) => Sym::P(konst(f32::from_bits(c as u32) as f64)),
                SVal::Op(i) => vals[i as usize].clone(),
            }
        };
        for o in &p.ops {
            let r = match &o.kind {
                SKind::Guard {
                    pred: SVal::Const(c),
                    neg,
                    new,
                    old,
                } => get(if (*c != 0) != *neg { *new } else { *old }, &vals),
                SKind::Guard { .. } => Sym::N,
                SKind::DivRn { a, b } => match (get(*a, &vals), get(*b, &vals)) {
                    (Sym::P(pa), Sym::P(pb)) => match as_const(&pb) {
                        Some(c) if c != 0.0 && c.is_finite() => {
                            // A power of two: exact (no underflow, N-NORM); otherwise one
                            // rounding, and 2^-52 for 1/c in f64.
                            let e = if pow2(c) { 0.0 } else { U32 + 2f64.powi(-52) };
                            Sym::P(round(&scaled(&pa, 1.0 / c), e))
                        }
                        _ => Sym::N,
                    },
                    _ => Sym::N,
                },
                SKind::Inst {
                    op, regs, cvals, ..
                } => {
                    let val = |v: &Val, vals: &[Sym]| -> Sym {
                        match v {
                            Val::Reg(r, _) => match regs.iter().find(|x| x.0 == *r) {
                                Some((_, s)) => get(*s, vals),
                                None => Sym::N,
                            },
                            Val::Imm(c) => Sym::P(konst(f32::from_bits(*c as u32) as f64)),
                            v => match cvals.iter().find(|x| x.0 == *v) {
                                Some((_, c)) => Sym::P(konst(f32::from_bits(*c as u32) as f64)),
                                None => Sym::N,
                            },
                        }
                    };
                    let sgn = |p: Sym, neg: bool| match p {
                        Sym::P(p) if neg => Sym::P(scaled(&p, -1.0)),
                        s => s,
                    };
                    match op {
                        Op::Ffma {
                            a,
                            b,
                            c,
                            neg,
                            abs,
                            rnd: Rnd::Rn,
                            ftz: false,
                            sat: false,
                            ..
                        } if !abs.iter().any(|x| *x) => {
                            let (x, y) = (sgn(val(a, &vals), neg[0]), sgn(val(b, &vals), neg[1]));
                            let z = c.as_ref().map(|c| sgn(val(c, &vals), neg[2]));
                            match (x, y, z) {
                                (Sym::P(x), Sym::P(y), None) => {
                                    let one =
                                        as_const(&x) == Some(1.0) || as_const(&y) == Some(1.0);
                                    Sym::P(round(&poly_mul(&x, &y), if one { 0.0 } else { U32 }))
                                }
                                (Sym::P(x), Sym::P(y), Some(Sym::P(z))) => {
                                    Sym::P(round(&sum(&poly_mul(&x, &y), &z), U32))
                                }
                                _ => Sym::N,
                            }
                        }
                        Op::Fadd {
                            a,
                            b,
                            neg,
                            abs,
                            rnd: Rnd::Rn,
                            ftz: false,
                            sat: false,
                            ..
                        } if !abs.iter().any(|x| *x) => {
                            match (sgn(val(a, &vals), neg[0]), sgn(val(b, &vals), neg[1])) {
                                (Sym::P(x), Sym::P(y)) => {
                                    let z = as_const(&x) == Some(0.0) || as_const(&y) == Some(0.0);
                                    Sym::P(round(&sum(&x, &y), if z { 0.0 } else { U32 }))
                                }
                                _ => Sym::N,
                            }
                        }
                        Op::Mov { a, .. } => val(a, &vals),
                        _ => Sym::N,
                    }
                }
            };
            vals.push(r);
        }
        if let Sym::P(q) = get(p.out, &vals) {
            return Ok(q);
        }
        // The polynomial values the non-polynomial part reads.
        let srcs_of = |o: &SOp| -> Vec<SVal> {
            match &o.kind {
                SKind::Inst { regs, .. } => regs.iter().map(|x| x.1).collect(),
                SKind::Guard { pred, new, old, .. } => vec![*pred, *new, *old],
                SKind::DivRn { a, b } => vec![*a, *b],
            }
        };
        let mut front: BTreeSet<(u8, u32)> = BTreeSet::new();
        let key = |v: SVal| match v {
            SVal::Arg(i) => Some((0u8, i as u32)),
            SVal::Op(i) => Some((1u8, i)),
            SVal::Const(_) => None,
        };
        for (i, o) in p.ops.iter().enumerate() {
            if matches!(vals[i], Sym::N) {
                for s in srcs_of(o) {
                    if matches!(get(s, &vals), Sym::P(_)) {
                        if let Some(k) = key(s) {
                            front.insert(k);
                        }
                    }
                }
            }
        }
        // The root: a polynomial value every other one the nonlinear part reads is computed
        // from (a range guard's scaled copy of A): the slice from it reads no argument.
        let mut found = None;
        for &(kind, idx) in &front {
            let root = if kind == 0 {
                SVal::Arg(idx as u16)
            } else {
                SVal::Op(idx)
            };
            if let Ok(g) = one_arg(p, root, &srcs_of) {
                found = Some((root, g));
                break;
            }
        }
        if std::env::var("PTXV_NORM_DEBUG").is_ok() && found.is_none() {
            for (i, o) in p.ops.iter().enumerate() {
                let k = match &o.kind {
                    SKind::Inst { op, regs, .. } => format!("{op:?} {regs:?}"),
                    k => format!("{k:?}"),
                };
                eprintln!(
                    "  op{i} {} {k}",
                    if matches!(vals[i], Sym::P(_)) {
                        "P"
                    } else {
                        "N"
                    }
                );
            }
            eprintln!("  out {:?} front {front:?}", p.out);
        }
        let Some((root, g)) = found else {
            return Err(format!(
                "a slice whose nonlinear part reads {} polynomial values, not computed from one",
                front.len()
            ));
        };
        let Sym::P(arg) = get(root, &vals) else {
            unreachable!()
        };
        let (pw, contract) = identify(&g)?;
        self.atoms.push(NAtom {
            g,
            p: pw,
            arg,
            contract,
        });
        let id = (self.atoms.len() - 1) as u64;
        Ok([(vec![(ATOM_LEAF, id)], Term::one())].into())
    }
}

/// `p` with structurally equal operations merged (the slicer repeats an instruction read at
/// two trace positions).
fn cse(p: &SProg) -> SProg {
    let mut seen: HashMap<String, u32> = HashMap::new();
    let mut alias: Vec<SVal> = Vec::with_capacity(p.ops.len());
    let mut ops: Vec<SOp> = Vec::new();
    let m = |v: SVal, alias: &[SVal]| match v {
        SVal::Op(i) => alias[i as usize],
        v => v,
    };
    for o in &p.ops {
        let kind = match &o.kind {
            SKind::Inst {
                op,
                dst,
                regs,
                cvals,
            } => SKind::Inst {
                op: op.clone(),
                dst: *dst,
                regs: regs.iter().map(|(r, v)| (*r, m(*v, &alias))).collect(),
                cvals: cvals.clone(),
            },
            SKind::Guard {
                pred,
                neg,
                new,
                old,
            } => SKind::Guard {
                pred: m(*pred, &alias),
                neg: *neg,
                new: m(*new, &alias),
                old: m(*old, &alias),
            },
            SKind::DivRn { a, b } => SKind::DivRn {
                a: m(*a, &alias),
                b: m(*b, &alias),
            },
        };
        let key = format!("{kind:?}");
        match seen.get(&key) {
            Some(&i) => alias.push(SVal::Op(i)),
            None => {
                ops.push(SOp { pc: o.pc, kind });
                let i = ops.len() as u32 - 1;
                seen.insert(key, i);
                alias.push(SVal::Op(i));
            }
        }
    }
    SProg {
        args: p.args.clone(),
        out: m(p.out, &alias),
        ops,
        extra_pcs: p.extra_pcs.clone(),
    }
}

/// The part of `p` computing its output from `root` alone, as a one-argument slice.
fn one_arg(p: &SProg, root: SVal, srcs_of: &dyn Fn(&SOp) -> Vec<SVal>) -> Result<SProg, String> {
    let mut keep = vec![false; p.ops.len()];
    let mut stack = vec![p.out];
    while let Some(v) = stack.pop() {
        if v == root {
            continue;
        }
        if let SVal::Op(i) = v {
            if !keep[i as usize] {
                keep[i as usize] = true;
                stack.extend(srcs_of(&p.ops[i as usize]));
            }
        }
    }
    let mut idx_map = vec![0u32; p.ops.len()];
    let mut ops = Vec::new();
    let map = |v: SVal, idx_map: &[u32]| -> Result<SVal, String> {
        if v == root {
            return Ok(SVal::Arg(0));
        }
        match v {
            SVal::Op(j) => Ok(SVal::Op(idx_map[j as usize])),
            SVal::Arg(_) => Err("the nonlinear part reads a second argument".into()),
            c => Ok(c),
        }
    };
    for (i, o) in p.ops.iter().enumerate() {
        if !keep[i] {
            continue;
        }
        let kind = match &o.kind {
            SKind::Inst {
                op,
                dst,
                regs,
                cvals,
            } => SKind::Inst {
                op: op.clone(),
                dst: *dst,
                regs: regs
                    .iter()
                    .map(|(r, v)| map(*v, &idx_map).map(|v| (*r, v)))
                    .collect::<Result<_, _>>()?,
                cvals: cvals.clone(),
            },
            SKind::Guard {
                pred,
                neg,
                new,
                old,
            } => SKind::Guard {
                pred: map(*pred, &idx_map)?,
                neg: *neg,
                new: map(*new, &idx_map)?,
                old: map(*old, &idx_map)?,
            },
            SKind::DivRn { a, b } => SKind::DivRn {
                a: map(*a, &idx_map)?,
                b: map(*b, &idx_map)?,
            },
        };
        idx_map[i] = ops.len() as u32;
        ops.push(SOp { pc: o.pc, kind });
    }
    let out = map(p.out, &idx_map)?;
    Ok(SProg {
        args: vec![u32::MAX],
        ops,
        out,
        extra_pcs: BTreeSet::new(),
    })
}

/// The power a slice of one argument approximates on positive normal f32 (from g(4) / g(1)),
/// and its contract there, from the exhaustive sweep.
fn identify(g: &SProg) -> Result<(f64, Contract), String> {
    let at = |x: f32| g.eval(&[x.to_bits()]).map(|b| f32::from_bits(b) as f64);
    let ratio = at(4.0)? / at(1.0)?;
    let (p, func) = [(-0.5, "rsqrt"), (-1.0, "rcp"), (0.5, "sqrt")]
        .into_iter()
        .find(|(p, _)| (ratio / 4f64.powf(*p) - 1.0).abs() < 1e-3)
        .ok_or_else(|| format!("a slice that is not a^p (g(4) / g(1) = {ratio})"))?;
    // Every positive normal f32: [2^-126, f32::MAX].
    let c = crate::softmax::sweep(g, func, &[(0x0080_0000, 0x7f7f_ffff)])?;
    if c.r0 > 2f64.powi(-10) {
        return Err(format!("slice is not {func} (relative error {:.3e})", c.r0));
    }
    Ok((p, c))
}

/// One kernel's output, atomized: its polynomial with the atom renumbered to 0 (if any).
struct Side1 {
    poly: Poly,
    atom: Option<NAtom>,
}

fn side1(ctx: &mut Ctx, n: u32) -> Result<Side1, String> {
    let poly = ctx.poly(n)?;
    let mut ids = BTreeSet::new();
    for m in poly.keys() {
        let k = m.iter().filter(|l| l.0 == ATOM_LEAF).count();
        if k > 1 {
            return Err("a monomial with more than one factor rho".into());
        }
        ids.extend(m.iter().filter(|l| l.0 == ATOM_LEAF).map(|l| l.1));
    }
    if ids.len() > 1 {
        return Err(format!("{} nonlinear scalars in one output", ids.len()));
    }
    let atom = ids.first().map(|&i| ctx.atoms[i as usize].clone());
    let poly = poly
        .into_iter()
        .map(|(m, t)| {
            let m: Mono = m
                .into_iter()
                .map(|l| if l.0 == ATOM_LEAF { (ATOM_LEAF, 0) } else { l })
                .collect();
            (m, t)
        })
        .collect();
    Ok(Side1 { poly, atom })
}

/// The per-output norm bound (module doc), and what `bound_at` needs to evaluate it.
#[derive(Clone, Debug, Serialize)]
pub struct NormBound {
    /// |C - R| <= (rel_u + kappa_u kappa) u |Y1| + add_u u T(x) for kappa(x) <= kappa_max.
    pub rel_u: f64,
    pub kappa_u: f64,
    pub add_u: f64,
    pub kappa_max: f64,
    /// kappa <= 1 for every input (A's monomials are squares with positive coefficients).
    pub kappa_le_1: bool,
    /// Each kernel's share (rel, kappa, add) in units of u.
    pub ref_part: (f64, f64, f64),
    pub cand_part: (f64, f64, f64),
    /// Largest relative coefficient difference (A and output polynomials) against the pivot.
    pub coef_diff: f64,
    pub p: f64,
    pub contracts: Vec<Contract>,
    /// Both outputs 16-bit roundings of the form's values (`rounded`): f16 if true.
    pub out_round: Option<bool>,
    #[serde(skip)]
    pub eval: NormEval,
}

/// Data for the bound's exact evaluation at an input.
#[derive(Clone, Debug, Default)]
pub struct NormEval {
    pub p: f64,
    pub r0: (f64, f64),
    pub h: f64,
    pub c1: f64,
    /// A: (monomial, c_R, a, g_R, g_C + |dc|).
    pub a: Vec<(Mono, f64, f64, f64, f64)>,
    /// Output: (monomial without rho, e, c_R, a, g_R + g_C + |dc|, g_R, g_C + |dc|).
    pub out: Vec<(Mono, u8, f64, f64, f64, f64, f64)>,
    /// Both outputs are 16-bit roundings (f16: `Some(true)`) of the form's values: the bounds
    /// are of the roundings (`equiv::round16_gap`).
    pub out_round: Option<bool>,
}

impl NormEval {
    /// (bound, |Y1|, T, kappa) at input x; None outside the covered inputs (e > E_MAX).
    pub fn at(&self, x: &dyn Fn(Leaf) -> f64) -> Option<(f64, f64, f64, f64)> {
        let r = self.at_pre(x)?;
        match self.out_round {
            None => Some(r),
            Some(f16) => {
                let (y, br, bc) = self.own_pre(x)?;
                let g = crate::equiv::round16_gap(y, br, bc, f16)?;
                Some((g.0, r.1, r.2, r.3))
            }
        }
    }

    /// The bound in 16-bit steps at x (16-bit outputs).
    pub fn steps_at(&self, x: &dyn Fn(Leaf) -> f64) -> Option<u64> {
        let f16 = self.out_round?;
        let (y, br, bc) = self.own_pre(x)?;
        crate::equiv::round16_gap(y, br, bc, f16).map(|g| g.1)
    }

    fn at_pre(&self, x: &dyn Fn(Leaf) -> f64) -> Option<(f64, f64, f64, f64)> {
        let mv = |m: &Mono| m.iter().map(|&l| x(l)).product::<f64>();
        let (mut astar, mut sa, mut er, mut ec) = (0.0, 0.0, 0.0, 0.0);
        for (m, c, a, gr, gc) in &self.a {
            let v = mv(m);
            astar += c * v;
            sa += a * v.abs();
            er += gr * v.abs();
            ec += gc * v.abs();
        }
        // Covered: A's error at most E_EVAL relative, A* >= 2^-125 (so A^ is a positive
        // normal f32, inside the swept domain).
        if astar < 2f64.powi(-125) {
            return None;
        }
        let (er, ec) = (er / astar, ec / astar);
        if er.max(ec) > E_EVAL {
            return None;
        }
        let rho = astar.powf(self.p);
        let (eta_r, eta_c) = (
            eta_exact(self.p, self.r0.0, er),
            eta_exact(self.p, self.r0.1, ec),
        );
        let h = 1.0 + eta_r.max(eta_c);
        let (mut p1, mut t, mut rest) = (0.0, 0.0, 0.0);
        for (m, e, c, a, g, _, _) in &self.out {
            let v = mv(m);
            let s = if *e == 1 { rho } else { 1.0 };
            if *e == 1 {
                p1 += c * v;
            }
            t += a * v.abs() * s;
            rest += g * v.abs() * s * if *e == 1 { h } else { 1.0 };
        }
        let y1 = (p1 * rho).abs();
        let b = y1 * (eta_r + eta_c) + rest;
        Some((b * (1.0 + 1e-9) + 1e-300, y1, t, sa / astar))
    }
}

impl NormEval {
    /// The reference's exact value P_R(x, rho*) and each kernel's own bound against it (the
    /// candidate's includes its coefficient differences); of the 16-bit roundings with
    /// `out_round`.
    pub fn own(&self, x: &dyn Fn(Leaf) -> f64) -> Option<(f64, f64, f64)> {
        let (y, br, bc) = self.own_pre(x)?;
        Some(match self.out_round {
            None => (y, br, bc),
            Some(f16) => (
                y,
                crate::equiv::round16_own(y, br, f16),
                crate::equiv::round16_own(y, bc, f16),
            ),
        })
    }

    fn own_pre(&self, x: &dyn Fn(Leaf) -> f64) -> Option<(f64, f64, f64)> {
        let mv = |m: &Mono| m.iter().map(|&l| x(l)).product::<f64>();
        let (mut astar, mut er, mut ec) = (0.0, 0.0, 0.0);
        for (m, c, _, gr, gc) in &self.a {
            let v = mv(m);
            astar += c * v;
            er += gr * v.abs();
            ec += gc * v.abs();
        }
        if astar < 2f64.powi(-125) || er.max(ec) / astar > E_EVAL {
            return None;
        }
        let (er, ec) = (er / astar, ec / astar);
        let rho = astar.powf(self.p);
        let (eta_r, eta_c) = (
            eta_exact(self.p, self.r0.0, er),
            eta_exact(self.p, self.r0.1, ec),
        );
        let hh = 1.0 + eta_r.max(eta_c);
        let (mut exact, mut p1, mut br, mut bc) = (0.0, 0.0, 0.0, 0.0);
        for (m, e, c, _, _, gr, gc) in &self.out {
            let v = mv(m);
            let (s, h) = if *e == 1 { (rho, hh) } else { (1.0, 1.0) };
            exact += c * v * s;
            if *e == 1 {
                p1 += c * v;
            }
            br += gr * v.abs() * s * h;
            bc += gc * v.abs() * s * h;
        }
        let y1 = (p1 * rho).abs();
        let slack = |b: f64| b * (1.0 + 1e-9) + 1e-300;
        Some((exact, slack(y1 * eta_r + br), slack(y1 * eta_c + bc)))
    }
}

/// Bound output `rn` of R against `cn` of C, both atomized.
pub fn bound(r: &mut Ctx, rn: u32, c: &mut Ctx, cn: u32) -> Result<NormBound, String> {
    let sr = side1(r, rn)?;
    let sc = side1(c, cn)?;
    let (ar, ac) = match (&sr.atom, &sc.atom) {
        (Some(a), Some(b)) => (a, b),
        _ => return Err("no nonlinear scalar rho on both sides".into()),
    };
    if ar.p != ac.p {
        return Err(format!("different powers ({} vs {})", ar.p, ac.p));
    }
    let up = 1.0 + 2f64.powi(-40);
    // Weight of a monomial (any positive weights make the bound rigorous): the larger sum of
    // |coefficient| over its occurrences in the two kernels' DAGs. A Welford reduction's
    // partial sums cancel (a pair of one thread's elements enters with weight ~1/4, the
    // variance's coefficient is -2/n^2), so the multipliers are relative to these.
    let weight = |m: &Mono, pr: &Poly, pc: &Poly| -> Result<f64, String> {
        let a = [pr.get(m), pc.get(m)]
            .into_iter()
            .flatten()
            .map(|t| t.abs.max(t.coef.abs()))
            .fold(0.0, f64::max);
        if a > 0.0 {
            Ok(a)
        } else {
            Err("a monomial without weight".into())
        }
    };
    // A.
    let mut a_rows = Vec::new();
    let (mut ka_r, mut ka_c, mut coef_diff) = (0.0f64, 0.0f64, 0.0f64);
    let mut kappa_le_1 = true;
    let mut c0 = 0.0;
    let keys: BTreeSet<&Mono> = ar.arg.keys().chain(ac.arg.keys()).collect();
    for m in keys {
        let tr = ar.arg.get(m).copied().unwrap_or_default();
        let tc = ac.arg.get(m).copied().unwrap_or_default();
        let a = weight(m, &ar.arg, &ac.arg)?;
        let dc = (tc.coef - tr.coef).abs();
        coef_diff = coef_diff.max(dc / a);
        ka_r = ka_r.max(tr.gam / a * up);
        ka_c = ka_c.max((tc.gam + dc) / a * up);
        if m.is_empty() {
            c0 = tr.coef;
        }
        let square = m.chunks(2).all(|w| w.len() == 2 && w[0] == w[1]);
        kappa_le_1 &= tr.coef > 0.0 && square && a == tr.coef;
        if std::env::var("PTXV_NORM_DEBUG").is_ok() && tr.gam / a / U32 > 1000.0 {
            eprintln!(
                "A {m:?}: cR {:e} absR {:e} gR {:e} cC {:e} gC {:e}",
                tr.coef,
                tr.abs,
                tr.gam / U32,
                tc.coef,
                tc.gam / U32
            );
        }
        a_rows.push((m.clone(), tr.coef, a, tr.gam * up, (tc.gam + dc) * up));
    }
    if c0 <= 0.0 {
        return Err("A has no positive constant term (eps)".into());
    }
    if coef_diff > 2f64.powi(-10) {
        return Err(format!(
            "different arguments of rho (relative coefficient difference {coef_diff:.2e})"
        ));
    }
    let (ka_r, ka_c) = (ka_r / U32, ka_c / U32);
    let kappa_max = E_MAX / (ka_r.max(ka_c) * U32);
    // c1: |(1 -+ e)^p - 1| <= c1 e on [0, E_MAX] (convexity, module doc).
    let p = ar.p;
    let c1 = ((1.0 - E_MAX).powf(p) - 1.0)
        .abs()
        .max(((1.0 + E_MAX).powf(p) - 1.0).abs())
        / E_MAX
        * up;
    let (r0r, r0c) = (ar.contract.r0, ac.contract.r0);
    let h = (1.0 + r0r.max(r0c)) * (1.0 + c1 * E_MAX) * up;
    // Output polynomial.
    let mut out_rows = Vec::new();
    let (mut kt, mut kt_r, mut kt_c) = (0.0f64, 0.0f64, 0.0f64);
    let keys: BTreeSet<&Mono> = sr.poly.keys().chain(sc.poly.keys()).collect();
    for m in keys {
        let tr = sr.poly.get(m).copied().unwrap_or_default();
        let tc = sc.poly.get(m).copied().unwrap_or_default();
        let a = weight(m, &sr.poly, &sc.poly)?;
        let dc = (tc.coef - tr.coef).abs();
        coef_diff = coef_diff.max(dc / a);
        let e = m.iter().filter(|l| l.0 == ATOM_LEAF).count() as u8;
        let hh = if e == 1 { h } else { 1.0 };
        let g = tr.gam + tc.gam + dc;
        kt = kt.max(g * hh / a * up);
        kt_r = kt_r.max(tr.gam * hh / a * up);
        kt_c = kt_c.max((tc.gam + dc) * hh / a * up);
        let mono: Mono = m.iter().copied().filter(|l| l.0 != ATOM_LEAF).collect();
        out_rows.push((mono, e, tr.coef, a, g * up, tr.gam * up, (tc.gam + dc) * up));
    }
    if coef_diff > 2f64.powi(-10) {
        return Err(format!(
            "different output polynomials (relative coefficient difference {coef_diff:.2e})"
        ));
    }
    let kr = c1 * (1.0 + r0r) * ka_r;
    let kc = c1 * (1.0 + r0c) * ka_c;
    Ok(NormBound {
        rel_u: (r0r + r0c) / U32 * up,
        kappa_u: (kr + kc) * up,
        add_u: kt / U32,
        kappa_max,
        kappa_le_1,
        ref_part: (r0r / U32, kr, kt_r / U32),
        cand_part: (r0c / U32, kc, kt_c / U32),
        coef_diff,
        p,
        contracts: vec![ar.contract, ac.contract],
        out_round: None,
        eval: NormEval {
            p,
            r0: (r0r * up, r0c * up),
            h,
            c1,
            a: a_rows,
            out: out_rows,
            out_round: None,
        },
    })
}

impl NormBound {
    /// Both outputs are 16-bit roundings of the form's values (`f16`: f16): the compact
    /// bound becomes (1 + u16) (bound) + 2 u16 |y| with |y| <= T(x), each kernel's (1 + u16)
    /// (own) + u16 |y| (`equiv::round_unit`; the absolute 2 a16 is in the output's abs_extra).
    pub fn rounded(mut self, f16: bool) -> NormBound {
        let (u16_, _) = crate::equiv::round_unit(f16);
        let g = 1.0 + u16_;
        self.rel_u *= g;
        self.kappa_u *= g;
        self.add_u = self.add_u * g + 2.0 * u16_ / U32;
        let own = |p: (f64, f64, f64)| (p.0 * g, p.1 * g, p.2 * g + u16_ / U32);
        self.ref_part = own(self.ref_part);
        self.cand_part = own(self.cand_part);
        self.eval.out_round = Some(f16);
        self.out_round = Some(f16);
        self
    }
}

impl NormBound {
    /// The coefficient of u T(x) valid for every covered input: |Y1| <= T, kappa <= 1 when
    /// proven, else kappa_max.
    pub fn total_u(&self) -> f64 {
        let k = if self.kappa_le_1 { 1.0 } else { self.kappa_max };
        self.rel_u + self.kappa_u * k + self.add_u
    }
}

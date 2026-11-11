/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Elementwise equivalence (docs/equiv-elementwise.md): kernels whose every output depends
//! on one input element through a short per-element computation (gelu, silu, exp, ...).
//!
//! In segment mode (`num.rs`, `seg.rs`) such an output's numerics node is a small DAG over a
//! single `Input` leaf: instruction slices (`Seg`, with their data-dependent guards as both
//! outcomes), and possibly tracked products / sums of the leaf with slices (`Fma`, `Add`).
//! Control flow and addresses do not depend on input data (the taint check, N-EQUIV), so the
//! DAG is output i's function for every input. `extract` turns each output's DAG into an
//! `ETree` (the element function) and the input element it reads; outputs with the same
//! function share one `ETree` (structural key, leaf identity ignored).
//!
//! `sweep` then evaluates a reference and a candidate element function on every value of the
//! input type (all 2^32 f32 bit patterns, or all 2^16 bf16 / f16 ones) and reports the exact
//! largest difference between them (absolute, relative, ulps) and of each against the true
//! function in f64, when one is named or identified (`TRUE_FUNCS`). This is a proof for every
//! input of that element type, as far as the slice semantics are exec.rs's (N-SEG) and the
//! true function's f64 evaluation is accurate (N-ELEMENTWISE). ASSUME(N-ELEMENTWISE): see
//! ASSUMPTIONS.md

use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::Mutex;

use ptxv_ir::Op;
use ptxv_ir::Val;
use serde::Serialize;

use crate::fp;
use crate::num::NodeKind;
use crate::num::NumState;
use crate::num::Orig;
use crate::seg::SKind;
use crate::seg::SProg;
use crate::seg::SVal;

/// Element type of an input leaf or an output word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
pub enum EType {
    #[default]
    F32,
    Bf16,
    F16,
}

impl EType {
    pub fn parse(s: &str) -> Option<EType> {
        match s {
            "f32" => Some(EType::F32),
            "bf16" => Some(EType::Bf16),
            "f16" => Some(EType::F16),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            EType::F32 => "f32",
            EType::Bf16 => "bf16",
            EType::F16 => "f16",
        }
    }
    /// Number of bit patterns.
    pub fn count(self) -> u64 {
        match self {
            EType::F32 => 1 << 32,
            _ => 1 << 16,
        }
    }
    /// The value of bit pattern `b` as f32 bits (bf16 and f16 are exact in f32).
    pub fn to_f32(self, b: u32) -> u32 {
        match self {
            EType::F32 => b,
            EType::Bf16 => b << 16,
            EType::F16 => f16_to_f32_bits(b as u16),
        }
    }
    /// The value of bit pattern `b` as f64.
    pub fn to_f64(self, b: u32) -> f64 {
        f32::from_bits(self.to_f32(b)) as f64
    }
    /// Spacing of this format at magnitude |v| (the ulp of the binade holding |v|).
    pub fn ulp(self, v: f64) -> f64 {
        let (p, emin) = match self {
            EType::F32 => (24, -126),
            EType::Bf16 => (8, -126),
            EType::F16 => (11, -14),
        };
        let a = v.abs();
        let e = if a == 0.0 || !a.is_finite() {
            emin
        } else {
            (a.log2().floor() as i32).max(emin)
        };
        2f64.powi(e - (p - 1))
    }
    /// Largest finite value.
    pub fn max(self) -> f64 {
        match self {
            EType::F32 => f32::MAX as f64,
            EType::Bf16 => f32::from_bits(0x7f7f_0000) as f64,
            EType::F16 => 65504.0,
        }
    }
    /// Distance in representable values between two patterns of this format (ordered
    /// integers; +0 and -0 are one apart).
    pub fn ulps_between(self, a: u32, b: u32) -> u64 {
        let w = match self {
            EType::F32 => 32,
            _ => 16,
        };
        let key = |x: u32| -> i64 {
            let sign = 1u32 << (w - 1);
            if x & sign != 0 {
                -((x & (sign - 1)) as i64) - 1
            } else {
                x as i64
            }
        };
        (key(a) - key(b)).unsigned_abs()
    }
    fn is_nan(self, b: u32) -> bool {
        f32::from_bits(self.to_f32(b)).is_nan()
    }
    /// A normal (finite, nonzero, not subnormal) value of this format.
    pub fn normal(self, b: u32) -> bool {
        let v = self.to_f64(b).abs();
        v.is_finite()
            && v >= self.ulp(0.0)
                * 2f64.powi(match self {
                    EType::F32 => 23,
                    EType::Bf16 => 7,
                    EType::F16 => 10,
                })
    }
}

/// f16 bits to f32 bits (exact).
pub fn f16_to_f32_bits(h: u16) -> u32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    match (e, m) {
        (0, 0) => s,
        (0, _) => {
            // Subnormal: m * 2^-24.
            let v = m as f32 * 2f32.powi(-24);
            s | v.to_bits()
        }
        (31, 0) => s | 0x7f80_0000,
        (31, _) => s | 0x7fc0_0000 | (m << 13),
        _ => s | ((e + 112) << 23) | (m << 13),
    }
}

/// An element function: the computation of one output from its input element `X`.
#[derive(Clone, Debug)]
pub enum ETree {
    /// The input element (as f32 bits).
    X,
    /// An input-independent constant (f32 bits).
    K(u32),
    /// Input element k of an output word reading several (resolved by `bind`).
    L(usize),
    /// The low or high 16 bits of a packed word.
    Half(Box<ETree>, bool),
    /// A bf16 (`true`: f16) rounding (N-ROUND) of an f32 value: its 16 bits (a 16-bit output
    /// element).
    Round(Box<ETree>, bool),
    /// An instruction slice of its arguments (`seg.rs`).
    Seg(Arc<SProg>, Vec<ETree>),
    /// fp32 FMA / FMUL, round to nearest: (-1)^neg_ab a b + (-1)^neg_c c.
    Fma {
        a: Box<ETree>,
        b: Box<ETree>,
        c: Option<Box<ETree>>,
        neg_ab: bool,
        neg_c: bool,
    },
    /// fp32 add, to nearest or toward zero.
    Add(Box<ETree>, Box<ETree>, bool),
}

impl ETree {
    /// Value at input `x` as f32 bits: a bf16 rounding widened (`eval` gives its 16 bits).
    pub fn eval_f32(&self, x: u32, scratch: &mut Vec<u64>) -> Result<u32, String> {
        let v = self.eval(x, scratch)?;
        Ok(match self {
            ETree::Round(_, false) => v << 16,
            ETree::Round(_, true) => f16_to_f32_bits(v as u16),
            _ => v,
        })
    }

    /// Value at input `x` (f32 bits; a top-level `Round`: its 16 bits).
    pub fn eval(&self, x: u32, scratch: &mut Vec<u64>) -> Result<u32, String> {
        Ok(match self {
            ETree::X => x,
            ETree::K(k) => *k,
            ETree::L(_) => return Err("unbound leaf".into()),
            ETree::Half(t, hi) => (t.eval(x, scratch)? >> if *hi { 16 } else { 0 }) & 0xffff,
            ETree::Round(t, false) => fp::f32_to_bf16_rn(t.eval_f32(x, scratch)?) as u32,
            ETree::Round(t, true) => fp::f32_to_f16_rn(t.eval_f32(x, scratch)?) as u32,
            ETree::Seg(p, args) => {
                let a: Vec<u32> = args
                    .iter()
                    .map(|t| t.eval_f32(x, scratch))
                    .collect::<Result<_, _>>()?;
                p.eval_with(&a, scratch)?
            }
            ETree::Fma {
                a,
                b,
                c,
                neg_ab,
                neg_c,
            } => {
                let av = a.eval_f32(x, scratch)? ^ if *neg_ab { 0x8000_0000 } else { 0 };
                let bv = b.eval_f32(x, scratch)?;
                let cv = match c {
                    Some(c) => Some(c.eval_f32(x, scratch)? ^ if *neg_c { 0x8000_0000 } else { 0 }),
                    None => None,
                };
                fp::f32_fma(av, bv, cv, ptxv_ir::Rnd::Rn, false)
            }
            ETree::Add(a, b, rz) => {
                let (av, bv) = (a.eval_f32(x, scratch)?, b.eval_f32(x, scratch)?);
                if *rz {
                    fp::f32_add_rz(av, bv)
                } else {
                    fp::f32_add_rn(av, bv)
                }
            }
        })
    }

    /// Structural key: equal keys compute the same function of `X`.
    pub fn key(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.hash_into(&mut h);
        h.finish()
    }

    fn hash_into(&self, h: &mut impl Hasher) {
        match self {
            ETree::X => 0u8.hash(h),
            ETree::K(k) => {
                1u8.hash(h);
                k.hash(h);
            }
            ETree::L(k) => {
                5u8.hash(h);
                k.hash(h);
            }
            ETree::Round(t, f16) => {
                // bf16 keeps its key (committed sweep caches).
                if *f16 { 9u8 } else { 7u8 }.hash(h);
                t.hash_into(h);
            }
            ETree::Half(t, hi) => {
                6u8.hash(h);
                hi.hash(h);
                t.hash_into(h);
            }
            ETree::Seg(p, args) => {
                2u8.hash(h);
                // The slice's instructions and wiring: argument node ids, pcs and register
                // names excluded (each op's registers renamed in order of appearance; `regs`
                // maps them to the values they hold, so the renaming keeps the semantics).
                for o in &p.ops {
                    canon_op(&o.kind).hash(h);
                }
                format!("{:?}", p.out).hash(h);
                for a in args {
                    a.hash_into(h);
                }
            }
            ETree::Fma {
                a,
                b,
                c,
                neg_ab,
                neg_c,
            } => {
                3u8.hash(h);
                a.hash_into(h);
                b.hash_into(h);
                c.is_some().hash(h);
                if let Some(c) = c {
                    c.hash_into(h);
                }
                (neg_ab, neg_c).hash(h);
            }
            ETree::Add(a, b, rz) => {
                4u8.hash(h);
                a.hash_into(h);
                b.hash_into(h);
                rz.hash(h);
            }
        }
    }

    /// Canonical text (debugging): slices with registers renamed.
    pub fn show(&self, ind: usize) -> String {
        let pad = "  ".repeat(ind);
        match self {
            ETree::X => format!("{pad}X\n"),
            ETree::K(k) => format!("{pad}K {:e}\n", f32::from_bits(*k)),
            ETree::L(k) => format!("{pad}L{k}\n"),
            ETree::Half(t, hi) => format!("{pad}half hi {hi}\n{}", t.show(ind + 1)),
            ETree::Round(t, f16) => format!(
                "{pad}round {}\n{}",
                if *f16 { "f16" } else { "bf16" },
                t.show(ind + 1)
            ),
            ETree::Seg(p, args) => {
                let mut s = format!("{pad}slice -> {:?}\n", p.out);
                for o in &p.ops {
                    s += &format!("{pad}  | {}\n", canon_op(&o.kind));
                }
                for a in args {
                    s += &a.show(ind + 1);
                }
                s
            }
            ETree::Fma {
                a,
                b,
                c,
                neg_ab,
                neg_c,
            } => {
                let mut s = format!("{pad}fma neg_ab {neg_ab} neg_c {neg_c}\n");
                s += &a.show(ind + 1);
                s += &b.show(ind + 1);
                if let Some(c) = c {
                    s += &c.show(ind + 1);
                }
                s
            }
            ETree::Add(a, b, rz) => {
                format!("{pad}add rz {rz}\n{}{}", a.show(ind + 1), b.show(ind + 1))
            }
        }
    }

    /// Number of slice instructions (for the report).
    pub fn size(&self) -> usize {
        match self {
            ETree::X | ETree::K(_) | ETree::L(_) => 0,
            ETree::Half(t, _) | ETree::Round(t, _) => t.size(),
            ETree::Seg(p, args) => p.ops.len() + args.iter().map(|a| a.size()).sum::<usize>(),
            ETree::Fma { a, b, c, .. } => {
                1 + a.size() + b.size() + c.as_ref().map_or(0, |c| c.size())
            }
            ETree::Add(a, b, _) => 1 + a.size() + b.size(),
        }
    }
}

/// A slice operation with register names abstracted: each source register is named by its
/// position in `regs` (which gives the value it holds), the destination `D` (its name only
/// matters where an operand refers to it, e.g. LOP3's predicate output). Two operations with
/// equal texts compute the same value from the same source values.
fn canon_op(k: &crate::seg::SKind) -> String {
    use crate::seg::SKind;
    let SKind::Inst {
        op,
        dst,
        regs,
        cvals,
    } = k
    else {
        return format!("{k:?}");
    };
    let dtext = format!("{dst:?}");
    let mut s = format!("{op:?}");
    // The op's own destination field.
    let df = format!("d: {dtext}");
    if let Some(i) = s.find(&df) {
        s.replace_range(i..i + df.len(), "d: D");
    }
    for (j, (r, _)) in regs.iter().enumerate() {
        s = s.replace(&format!("{r:?}"), &format!("s{j}"));
    }
    s = s.replace(&dtext, "D");
    let wiring: Vec<String> = regs.iter().map(|(_, v)| format!("{v:?}")).collect();
    format!("{s} <- [{}] {:?}", wiring.join(", "), cvals)
}

/// Output `index` is `fns[f]` of input element `leaf`.
#[derive(Clone, Debug, Serialize)]
pub struct ElemOut {
    pub index: usize,
    pub leaf: Orig,
    pub f: usize,
    /// Product outputs: the output is `rn(fns[f](leaf) * factor)`, one f32 multiply (round
    /// to nearest) by another input element (fused `silu(a) * b` against eager's silu kernel
    /// followed by a mul kernel).
    pub factor: Option<Orig>,
    /// Product outputs stored as bf16: the output is `bf16(rn(fns[f](leaf) * factor))`.
    pub round16: bool,
}

#[derive(Clone, Debug)]
pub struct Extract {
    pub outs: Vec<ElemOut>,
    pub fns: Vec<ETree>,
}

/// The element function of node `n` over the input elements it reads (`leaves`; leaf k is
/// `ETree::L(k)` until resolved).
fn tree(ns: &NumState, n: u32, leaves: &mut Vec<Orig>, depth: usize) -> Result<ETree, String> {
    if depth > 64 {
        return Err("element function too deep".into());
    }
    let node = &ns.nodes[n as usize];
    let sub = |m: u32, leaves: &mut Vec<Orig>| tree(ns, m, leaves, depth + 1).map(Box::new);
    Ok(match &node.kind {
        NodeKind::Input(o) => {
            let k = leaves.iter().position(|l| l == o).unwrap_or_else(|| {
                leaves.push(*o);
                leaves.len() - 1
            });
            ETree::L(k)
        }
        NodeKind::Leaf => ETree::K(node.hw),
        NodeKind::Seg(p) => {
            let args =
                p.0.args
                    .iter()
                    .map(|&a| tree(ns, a, leaves, depth + 1))
                    .collect::<Result<_, _>>()?;
            ETree::Seg(p.0.clone(), args)
        }
        NodeKind::Fma {
            a,
            b,
            c,
            neg_ab,
            neg_c,
        } => ETree::Fma {
            a: sub(*a, leaves)?,
            b: sub(*b, leaves)?,
            c: match c {
                Some(c) => Some(sub(*c, leaves)?),
                None => None,
            },
            neg_ab: *neg_ab,
            neg_c: *neg_c,
        },
        NodeKind::Add(a, b, rz) => ETree::Add(sub(*a, leaves)?, sub(*b, leaves)?, *rz),
        NodeKind::Round { arg, f16 } => ETree::Round(sub(*arg, leaves)?, *f16),
        k => {
            return Err(format!(
                "a {} node is not an element function",
                match k {
                    NodeKind::Opaque => "untracked (opaque)",
                    NodeKind::Max { .. } => "max",
                    NodeKind::Mufu { .. } => "MUFU",
                    NodeKind::Mma { .. } => "tensor-core",
                    NodeKind::AtomAdd(..) | NodeKind::AtomAdd16(..) => "atomic add",
                    _ => "other",
                }
            ));
        }
    })
}

impl ETree {
    /// Leaves this tree reads.
    fn leaves(&self, out: &mut Vec<usize>) {
        match self {
            ETree::L(k) => {
                if !out.contains(k) {
                    out.push(*k)
                }
            }
            ETree::X | ETree::K(_) => {}
            ETree::Seg(_, a) => a.iter().for_each(|t| t.leaves(out)),
            ETree::Fma { a, b, c, .. } => {
                a.leaves(out);
                b.leaves(out);
                if let Some(c) = c {
                    c.leaves(out);
                }
            }
            ETree::Add(a, b, _) => {
                a.leaves(out);
                b.leaves(out);
            }
            ETree::Half(t, _) | ETree::Round(t, _) => t.leaves(out),
        }
    }

    /// Leaf `k` becomes the argument `X`, every other leaf the constant 0 (sound only where
    /// the result does not depend on them).
    fn bind(&self, k: usize) -> ETree {
        let b = |t: &ETree| Box::new(t.bind(k));
        match self {
            ETree::L(j) if *j == k => ETree::X,
            ETree::L(_) => ETree::K(0),
            ETree::X => ETree::X,
            ETree::K(c) => ETree::K(*c),
            ETree::Seg(p, a) => ETree::Seg(p.clone(), a.iter().map(|t| t.bind(k)).collect()),
            ETree::Fma {
                a,
                b: bb,
                c,
                neg_ab,
                neg_c,
            } => ETree::Fma {
                a: b(a),
                b: b(bb),
                c: c.as_ref().map(|c| b(c)),
                neg_ab: *neg_ab,
                neg_c: *neg_c,
            },
            ETree::Add(x, y, rz) => ETree::Add(b(x), b(y), *rz),
            ETree::Half(t, hi) => ETree::Half(b(t), *hi),
            ETree::Round(t, f) => ETree::Round(b(t), *f),
        }
    }
}

/// Arguments (indices into `p.args`) that value `v` of slice `p` depends on.
fn slice_deps(p: &SProg, v: SVal, memo: &mut HashMap<u32, Vec<u16>>) -> Vec<u16> {
    match v {
        SVal::Arg(i) => vec![i],
        SVal::Const(_) => vec![],
        SVal::Op(i) => {
            if let Some(d) = memo.get(&i) {
                return d.clone();
            }
            let srcs: Vec<SVal> = match &p.ops[i as usize].kind {
                // FFMA2: only the registers of the lane `dst` holds.
                SKind::Inst {
                    op: Op::Ffma2 { d, a, b, c, .. },
                    dst,
                    regs,
                    ..
                } => {
                    let lane = dst.idx.wrapping_sub(d.idx);
                    let used: Vec<ptxv_ir::Reg> = [a, b, c]
                        .iter()
                        .filter_map(|(v, l)| match v {
                            Val::Reg(r, _) => Some(ptxv_ir::Reg {
                                file: r.file,
                                idx: r.idx
                                    + match l {
                                        ptxv_ir::F2Lanes::Bcast => 0,
                                        ptxv_ir::F2Lanes::Pair => lane,
                                        ptxv_ir::F2Lanes::Swap => 1 - lane,
                                    },
                            }),
                            _ => None,
                        })
                        .collect();
                    regs.iter()
                        .filter(|(r, _)| used.contains(r))
                        .map(|(_, s)| *s)
                        .collect()
                }
                SKind::Inst { regs, .. } => regs.iter().map(|(_, s)| *s).collect(),
                SKind::Guard { pred, new, old, .. } => vec![*pred, *new, *old],
                SKind::DivRn { a, b } => vec![*a, *b],
            };
            let mut out: Vec<u16> = Vec::new();
            for s in srcs {
                for d in slice_deps(p, s, memo) {
                    if !out.contains(&d) {
                        out.push(d);
                    }
                }
            }
            memo.insert(i, out.clone());
            out
        }
    }
}

/// The 16-bit half `hi` of a word whose tree is `t`, as a function of one leaf: the word's
/// last operation packs two f32 values (`F2FP.BF16/F16.F32.PACK_AB d, a, b`: a high, b low);
/// the half's operand must depend on a single leaf. Returns (leaf, element function).
fn half_of(t: &ETree, hi: bool) -> Result<(usize, ETree), String> {
    let ETree::Seg(p, args) = t else {
        return Err("a 16-bit output word not computed by a slice".into());
    };
    let SVal::Op(o) = p.out else {
        return Err("a 16-bit output word is not computed".into());
    };
    let SKind::Inst { op, regs, .. } = &p.ops[o as usize].kind else {
        return Err("a 16-bit output word not packed by F2FP".into());
    };
    let operand = match op {
        Op::F2fpBf16 { a, b, .. }
        | Op::F2fpF16 {
            a, b, rz: false, ..
        } => {
            if hi {
                a
            } else {
                b
            }
        }
        _ => return Err(format!("a 16-bit output word packed by {op:?}, not F2FP")),
    };
    let deps: Vec<u16> = match operand {
        Val::Reg(r, _) => match regs.iter().find(|(x, _)| x == r) {
            Some((_, s)) => slice_deps(p, *s, &mut HashMap::new()),
            None => vec![],
        },
        _ => vec![],
    };
    let mut ls = Vec::new();
    for d in deps {
        args[d as usize].leaves(&mut ls);
    }
    match ls.as_slice() {
        [k] => Ok((*k, ETree::Half(Box::new(t.bind(*k)), hi))),
        [] => Err("a 16-bit output does not depend on any input".into()),
        _ => Err("a 16-bit output depends on two input elements".into()),
    }
}

/// `t = rn(f * L(k))` (an f32 multiply by input element k, round to nearest, no addend) with
/// `f` a function of one other input element: (k, f).
fn product_of(t: &ETree) -> Option<(usize, &ETree)> {
    let ETree::Fma {
        a,
        b,
        c: None,
        neg_ab: false,
        ..
    } = t
    else {
        return None;
    };
    let one = |f: &ETree, k: usize| {
        let mut l = Vec::new();
        f.leaves(&mut l);
        l.len() == 1 && l[0] != k
    };
    match (a.as_ref(), b.as_ref()) {
        (ETree::L(k), f) | (f, ETree::L(k)) if one(f, *k) => Some((*k, f)),
        _ => None,
    }
}

/// The leaves the value of `t` depends on: through a slice, only the arguments its output
/// reads (`slice_deps`; a slice of a packed word of two elements depends on one of them).
fn dep_leaves(t: &ETree, out: &mut Vec<usize>) {
    match t {
        ETree::Round(x, _) => dep_leaves(x, out),
        ETree::Seg(p, args) => {
            for d in slice_deps(p, p.out, &mut HashMap::new()) {
                dep_leaves(&args[d as usize], out);
            }
        }
        _ => t.leaves(out),
    }
}

/// Every output's element function. `outs[w]` = (bits, node) of output word w; with `half`,
/// each word holds two 16-bit outputs (2w low, 2w + 1 high).
pub fn extract(ns: &NumState, outs: &[(u32, Option<u32>)], half: bool) -> Result<Extract, String> {
    let mut fns: Vec<ETree> = Vec::new();
    let mut keys: HashMap<u64, usize> = HashMap::new();
    let mut eo = Vec::with_capacity(outs.len() * if half { 2 } else { 1 });
    let mut factors: HashMap<usize, Orig> = HashMap::new();
    let mut rounded: std::collections::HashSet<usize> = Default::default();
    for (w, (bits, n)) in outs.iter().enumerate() {
        let Some(n) = n else {
            return Err(format!(
                "output word {w} has no dataflow history (bits {bits:#x})"
            ));
        };
        let mut leaves = Vec::new();
        let t = tree(ns, *n, &mut leaves, 0).map_err(|e| format!("output word {w}: {e}"))?;
        let parts: Vec<(usize, Orig, ETree)> = if half {
            let mut v = Vec::new();
            for (h, idx) in [(false, 2 * w), (true, 2 * w + 1)] {
                let (k, f) = half_of(&t, h).map_err(|e| format!("output {idx}: {e}"))?;
                v.push((idx, leaves[k], f));
            }
            v
        } else {
            // A bf16 rounding of a slice: the leaves the slice's output reads.
            let mut dl = Vec::new();
            if matches!(t, ETree::Round(..)) {
                dep_leaves(&t, &mut dl);
            }
            match (leaves.as_slice(), dl.as_slice()) {
                (_, [k]) => vec![(w, leaves[*k], t.bind(*k))],
                ([l], _) => vec![(w, *l, t.bind(0))],
                ([], _) => return Err(format!("output {w} does not depend on any input")),
                ([_, _], _) if product_of(&t).is_some() => {
                    let (k, f) = product_of(&t).unwrap();
                    factors.insert(w, leaves[k]);
                    let mut fl = Vec::new();
                    f.leaves(&mut fl);
                    vec![(w, leaves[fl[0]], f.bind(fl[0]))]
                }
                // bf16(rn(f(x) * y)): a product output rounded to bf16 when stored.
                ([_, _], _) if matches!(&t, ETree::Round(p, _) if product_of(p).is_some()) => {
                    let ETree::Round(p, _) = &t else {
                        unreachable!()
                    };
                    let (k, f) = product_of(p).unwrap();
                    factors.insert(w, leaves[k]);
                    rounded.insert(w);
                    let mut fl = Vec::new();
                    f.leaves(&mut fl);
                    vec![(w, leaves[fl[0]], f.bind(fl[0]))]
                }
                ([a, b, ..], _) => {
                    return Err(format!(
                        "output {w} depends on two input elements ({}[{}] and {}[{}])",
                        a.0, a.1, b.0, b.1
                    ));
                }
            }
        };
        for (index, leaf, f) in parts {
            let k = f.key();
            if std::env::var("PTXV_EW_DEBUG").is_ok() && !keys.contains_key(&k) {
                eprintln!("function {k:016x} (output {index}):\n{}", f.show(1));
            }
            let fi = *keys.entry(k).or_insert_with(|| {
                fns.push(f);
                fns.len() - 1
            });
            eo.push(ElemOut {
                index,
                leaf,
                f: fi,
                factor: factors.get(&index).copied(),
                round16: rounded.contains(&index),
            });
        }
    }
    Ok(Extract { outs: eo, fns })
}

// ------------------------------------------------------------------ true functions

/// erf in f64: Taylor series below 3 (relative error < 1e-13 there), else 1 - erfc.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let a = x.abs();
    let r = if a < 3.0 {
        erf_series(a)
    } else {
        1.0 - erfc_cf(a)
    };
    r.copysign(x)
}

/// erfc in f64 (accurate relative to erfc, also for large positive x).
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x < 3.0 {
        1.0 - erf_series(x)
    } else {
        erfc_cf(x)
    }
}

fn erf_series(x: f64) -> f64 {
    // erf(x) = 2/sqrt(pi) sum_n (-1)^n x^(2n+1) / (n! (2n+1))
    let x2 = x * x;
    let mut term = x; // (-1)^n x^(2n+1) / n!
    let mut sum = x;
    let mut n = 0u32;
    loop {
        n += 1;
        term *= -x2 / n as f64;
        let t = term / (2 * n + 1) as f64;
        sum += t;
        if t.abs() < 1e-17 * sum.abs() || n > 200 {
            break;
        }
    }
    sum * std::f64::consts::FRAC_2_SQRT_PI
}

fn erfc_cf(x: f64) -> f64 {
    // erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + 1/2/(x + 1/(x + 3/2/(x + 2/(x + ...)))))
    // evaluated with the modified Lentz method on K(a_k / (x)), a_k = k/2.
    let tiny = 1e-300;
    let mut f = x;
    let mut c = x;
    let mut d = 0.0;
    for k in 1..2000 {
        let a = k as f64 / 2.0;
        d = x + a * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = x + a / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    (-x * x).exp() / (f * std::f64::consts::PI.sqrt())
}

/// Named real functions an elementwise kernel may compute.
pub const TRUE_FUNCS: &[&str] = &[
    "identity",
    "neg",
    "abs",
    "relu",
    "exp",
    "exp2",
    "log",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "tanh",
    "sigmoid",
    "silu",
    "gelu",
    "gelu_tanh",
    "erf",
    "sin",
    "cos",
    "softplus",
];

/// The true value of `func` at x in f64.
pub fn true_fn(func: &str, x: f64) -> Option<f64> {
    let sigmoid = |x: f64| {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    };
    Some(match func {
        "identity" => x,
        "neg" => -x,
        "abs" => x.abs(),
        "relu" => x.max(0.0),
        "exp" => x.exp(),
        "exp2" => x.exp2(),
        "log" => x.ln(),
        "sqrt" => x.sqrt(),
        "rsqrt" => 1.0 / x.sqrt(),
        "reciprocal" => 1.0 / x,
        "tanh" => x.tanh(),
        "sigmoid" => sigmoid(x),
        "silu" => x * sigmoid(x),
        "erf" => erf(x),
        // 0.5 x (1 + erf(x / sqrt 2)) = 0.5 x erfc(-x / sqrt 2) (no cancellation for x < 0).
        "gelu" => 0.5 * x * erfc(-x * std::f64::consts::FRAC_1_SQRT_2),
        // 0.5 x (1 + tanh(u)) = x / (1 + exp(-2u)), u = sqrt(2/pi) (x + 0.044715 x^3).
        "gelu_tanh" => {
            let u = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
            x * sigmoid(2.0 * u)
        }
        "sin" => x.sin(),
        "cos" => x.cos(),
        // torch's softplus (beta 1, threshold 20): x above 20, else log1p(exp(x)).
        "softplus" => {
            if x > 20.0 {
                x
            } else {
                x.exp().ln_1p()
            }
        }
        _ => return None,
    })
}

/// The library function closest to both element functions on a sample of inputs (largest
/// deviation relative to max(|true value|, 2^-10)), if within 2^-5 and every other function
/// is at least 8 times farther (gelu and gelu_tanh differ by 0.2 on this measure).
pub fn identify(fs: &[&ETree], ty: EType, out: EType) -> Option<(&'static str, f64)> {
    let mut xs: Vec<f64> = Vec::new();
    for i in -400..=400 {
        xs.push(i as f64 / 20.0);
    }
    for e in -20..6 {
        for m in [1.0, 1.37, 1.9] {
            xs.push(m * 2f64.powi(e));
            xs.push(-m * 2f64.powi(e));
        }
    }
    let mut scratch = Vec::new();
    let mut all: Vec<(&'static str, f64)> = Vec::new();
    for &name in TRUE_FUNCS {
        let mut worst = 0.0f64;
        for &x in &xs {
            let xb = (x as f32).to_bits();
            let xb = match ty {
                EType::F32 => xb,
                EType::Bf16 => xb & 0xffff_0000,
                EType::F16 => (x as f32).to_bits(),
            };
            let xv = f32::from_bits(xb) as f64;
            let Some(t) = true_fn(name, xv) else { break };
            if !t.is_finite() || t.abs() > 1e30 {
                continue;
            }
            for f in fs {
                let y = if out == EType::F32 {
                    f.eval_f32(xb, &mut scratch)
                } else {
                    f.eval(xb, &mut scratch)
                };
                let Ok(y) = y else {
                    return None;
                };
                let y = out.to_f64(y);
                let d = if y.is_nan() {
                    f64::INFINITY
                } else {
                    (y - t).abs() / t.abs().max(2f64.powi(-10))
                };
                worst = worst.max(d);
            }
        }
        if std::env::var("PTXV_EW_DEBUG").is_ok() {
            eprintln!("identify: {name}: worst deviation {worst:e}");
        }
        all.push((name, worst));
    }
    all.sort_by(|a, b| a.1.total_cmp(&b.1));
    match all.as_slice() {
        [b, n, ..] if b.1 < 2f64.powi(-5) && n.1 >= 8.0 * b.1 => Some(*b),
        _ => None,
    }
}

// ------------------------------------------------------------------ sweeps

/// One kernel against the true function, over every finite input.
#[derive(Clone, Debug, Default, Serialize)]
pub struct VsTrue {
    pub max_abs: f64,
    pub at_abs: f64,
    pub max_rel: f64,
    pub at_rel: f64,
    /// |K - T| / ulp(T) in the output format (ulp at the binade of the true value).
    pub max_ulps: f64,
    pub at_ulps: f64,
    /// The same where |true value| >= 2^-10 (away from underflow and cancellation tails,
    /// where a tiny absolute error is many ulps).
    pub max_ulps_big: f64,
    pub at_ulps_big: f64,
    /// Inputs where the true value is beyond the output format's range (excluded).
    pub overflow: u64,
    /// Inputs with a finite true value where the kernel returned inf or NaN.
    pub non_finite: u64,
    pub first_non_finite: Option<f64>,
}

/// A reference and a candidate element function over every input value.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PairSweep {
    pub input: EType,
    pub output: EType,
    pub inputs: u64,
    pub bitwise_equal: u64,
    /// Both NaN (any payloads) but not bitwise equal.
    pub both_nan: u64,
    /// One NaN / infinite and the other not, or infinities of different signs.
    pub special_mismatch: u64,
    pub first_special_mismatch: Option<(f64, f64, f64)>,
    /// Over inputs where both are finite.
    pub max_abs: f64,
    pub at_abs: f64,
    pub values_at_abs: (f64, f64),
    pub max_rel: f64,
    pub at_rel: f64,
    pub max_ulps: u64,
    pub at_ulps: f64,
    pub values_at_ulps: (f64, f64),
    /// Outputs for x = +inf, -inf, NaN (reference, candidate).
    pub specials: Vec<(String, f64, f64)>,
    pub true_fn: Option<String>,
    pub reference_vs_true: Option<VsTrue>,
    pub candidate_vs_true: Option<VsTrue>,
    pub seconds: f64,
}

/// Output bits of an element function, preserving encodings for a same-format copy.
pub fn out_bits(
    f: &ETree,
    ty: EType,
    out: EType,
    b: u32,
    scratch: &mut Vec<u64>,
) -> Result<u32, String> {
    if matches!(f, ETree::X) && ty == out {
        return Ok(b);
    }
    Ok(match out {
        EType::F32 => f.eval_f32(ty.to_f32(b), scratch)?,
        // A 16-bit output: its element function ends in `Half` or `Round`.
        _ => f.eval(ty.to_f32(b), scratch)? & 0xffff,
    })
}

fn out_f64(out: EType, y: u32) -> f64 {
    out.to_f64(y)
}

#[derive(Clone, Default)]
struct Acc {
    s: PairSweep,
    vr: VsTrue,
    vc: VsTrue,
}

fn vs_true(v: &mut VsTrue, out: EType, x: f64, y: f64, t: f64) {
    if !t.is_finite() || t.abs() > out.max() * (1.0 + out.ulp(out.max()) / out.max() / 2.0) {
        v.overflow += 1;
        return;
    }
    if !y.is_finite() {
        v.non_finite += 1;
        if v.first_non_finite.is_none() {
            v.first_non_finite = Some(x);
        }
        return;
    }
    let d = (y - t).abs();
    if d > v.max_abs {
        v.max_abs = d;
        v.at_abs = x;
    }
    if t != 0.0 {
        let r = d / t.abs();
        if r > v.max_rel {
            v.max_rel = r;
            v.at_rel = x;
        }
    }
    let u = d / out.ulp(t);
    if u > v.max_ulps {
        v.max_ulps = u;
        v.at_ulps = x;
    }
    if t.abs() >= 2f64.powi(-10) && u > v.max_ulps_big {
        v.max_ulps_big = u;
        v.at_ulps_big = x;
    }
}

fn merge_true(a: &mut VsTrue, b: &VsTrue) {
    if b.max_abs > a.max_abs {
        a.max_abs = b.max_abs;
        a.at_abs = b.at_abs;
    }
    if b.max_rel > a.max_rel {
        a.max_rel = b.max_rel;
        a.at_rel = b.at_rel;
    }
    if b.max_ulps > a.max_ulps {
        a.max_ulps = b.max_ulps;
        a.at_ulps = b.at_ulps;
    }
    if b.max_ulps_big > a.max_ulps_big {
        a.max_ulps_big = b.max_ulps_big;
        a.at_ulps_big = b.at_ulps_big;
    }
    a.overflow += b.overflow;
    a.non_finite += b.non_finite;
    if a.first_non_finite.is_none() {
        a.first_non_finite = b.first_non_finite;
    }
}

/// Sweep cache: (reference key, candidate key, input, output, true function) -> result.
static CACHE: Mutex<Option<HashMap<String, PairSweep>>> = Mutex::new(None);

pub fn load_cache(path: &std::path::Path) {
    let m: HashMap<String, PairSweep> = std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<HashMap<String, serde_json::Value>>(&s).ok())
        .map(|m| {
            m.into_iter()
                .filter_map(|(k, v)| from_json(&v).map(|p| (k, p)))
                .collect()
        })
        .unwrap_or_default();
    *CACHE.lock().unwrap() = Some(m);
}

/// Save the sweep cache, sorted by key (a committed file: stable across runs).
pub fn save_cache(path: &std::path::Path) -> std::io::Result<()> {
    let g = CACHE.lock().unwrap();
    if let Some(m) = g.as_ref() {
        let sorted: std::collections::BTreeMap<_, _> = m.iter().collect();
        std::fs::write(path, serde_json::to_string_pretty(&sorted).unwrap())?;
    }
    Ok(())
}

fn from_json(v: &serde_json::Value) -> Option<PairSweep> {
    let f = |k: &str| v[k].as_f64();
    let u = |k: &str| v[k].as_u64();
    let ty = |k: &str| EType::parse(v[k].as_str()?.to_lowercase().as_str());
    let vt = |k: &str| -> Option<VsTrue> {
        let w = &v[k];
        if w.is_null() {
            return None;
        }
        Some(VsTrue {
            max_abs: w["max_abs"].as_f64()?,
            at_abs: w["at_abs"].as_f64()?,
            max_rel: w["max_rel"].as_f64()?,
            at_rel: w["at_rel"].as_f64()?,
            max_ulps: w["max_ulps"].as_f64()?,
            at_ulps: w["at_ulps"].as_f64()?,
            max_ulps_big: w["max_ulps_big"].as_f64()?,
            at_ulps_big: w["at_ulps_big"].as_f64()?,
            overflow: w["overflow"].as_u64()?,
            non_finite: w["non_finite"].as_u64()?,
            first_non_finite: w["first_non_finite"].as_f64(),
        })
    };
    let pair = |k: &str| -> Option<(f64, f64)> { Some((v[k][0].as_f64()?, v[k][1].as_f64()?)) };
    Some(PairSweep {
        input: ty("input")?,
        output: ty("output")?,
        inputs: u("inputs")?,
        bitwise_equal: u("bitwise_equal")?,
        both_nan: u("both_nan")?,
        special_mismatch: u("special_mismatch")?,
        first_special_mismatch: v["first_special_mismatch"].as_array().and_then(|a| {
            Some((
                a.first()?.as_f64()?,
                a.get(1)?.as_f64()?,
                a.get(2)?.as_f64()?,
            ))
        }),
        max_abs: f("max_abs")?,
        at_abs: f("at_abs")?,
        values_at_abs: pair("values_at_abs")?,
        max_rel: f("max_rel")?,
        at_rel: f("at_rel")?,
        max_ulps: u("max_ulps")?,
        at_ulps: f("at_ulps")?,
        values_at_ulps: pair("values_at_ulps")?,
        specials: v["specials"]
            .as_array()?
            .iter()
            .filter_map(|s| {
                Some((
                    s[0].as_str()?.to_string(),
                    s[1].as_f64().unwrap_or(f64::NAN),
                    s[2].as_f64().unwrap_or(f64::NAN),
                ))
            })
            .collect(),
        true_fn: v["true_fn"].as_str().map(|s| s.to_string()),
        reference_vs_true: vt("reference_vs_true"),
        candidate_vs_true: vt("candidate_vs_true"),
        seconds: f("seconds").unwrap_or(0.0),
    })
}

/// Evaluate `r` and `c` on every input pattern of `ty` (outputs in format `out`); with
/// `func`, each against the true function too.
pub fn sweep(
    r: &ETree,
    c: &ETree,
    ty: EType,
    out: EType,
    func: Option<&str>,
) -> Result<PairSweep, String> {
    let policy =
        if ty == out && ty != EType::F32 && (matches!(r, ETree::X) || matches!(c, ETree::X)) {
            "copy-bits-v2-"
        } else {
            ""
        };
    let key = format!(
        "{policy}{:016x}-{:016x}-{}-{}-{}",
        r.key(),
        c.key(),
        ty.name(),
        out.name(),
        func.unwrap_or("-")
    );
    if let Some(m) = CACHE.lock().unwrap().as_ref() {
        if let Some(p) = m.get(&key) {
            return Ok(p.clone());
        }
    }
    let t0 = std::time::Instant::now();
    let total = ty.count();
    let chunk: u64 = 1 << 20;
    let items = total.div_ceil(chunk);
    let next = std::sync::atomic::AtomicU64::new(0);
    let threads = std::thread::available_parallelism()
        .map_or(8, |n| n.get())
        .min(128);
    let results: Vec<Result<Acc, String>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| -> Result<Acc, String> {
                    let mut a = Acc::default();
                    let mut scratch = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= items {
                            break;
                        }
                        let lo = i * chunk;
                        let hi = ((i + 1) * chunk).min(total);
                        for b in lo..hi {
                            let b = b as u32;
                            let x = ty.to_f64(b);
                            let yr = out_bits(r, ty, out, b, &mut scratch)?;
                            let yc = out_bits(c, ty, out, b, &mut scratch)?;
                            let (vr, vc) = (out_f64(out, yr), out_f64(out, yc));
                            let s = &mut a.s;
                            s.inputs += 1;
                            if yr == yc {
                                s.bitwise_equal += 1;
                            } else if out.is_nan(yr) && out.is_nan(yc) {
                                s.both_nan += 1;
                            } else if !(vr.is_finite() && vc.is_finite()) {
                                if vr != vc {
                                    s.special_mismatch += 1;
                                    if s.first_special_mismatch.is_none() {
                                        s.first_special_mismatch = Some((x, vr, vc));
                                    }
                                }
                            } else {
                                let d = (vr - vc).abs();
                                if d > s.max_abs {
                                    s.max_abs = d;
                                    s.at_abs = x;
                                    s.values_at_abs = (vr, vc);
                                }
                                if vr != 0.0 {
                                    let rel = d / vr.abs();
                                    if rel > s.max_rel {
                                        s.max_rel = rel;
                                        s.at_rel = x;
                                    }
                                }
                                let u = out.ulps_between(yr, yc);
                                if u > s.max_ulps {
                                    s.max_ulps = u;
                                    s.at_ulps = x;
                                    s.values_at_ulps = (vr, vc);
                                }
                            }
                            if let Some(fname) = func {
                                if x.is_finite() {
                                    if let Some(t) = true_fn(fname, x) {
                                        if !t.is_nan() {
                                            vs_true(&mut a.vr, out, x, vr, t);
                                            vs_true(&mut a.vc, out, x, vc, t);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Ok(a)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut s = PairSweep {
        input: ty,
        output: out,
        ..Default::default()
    };
    let (mut vr, mut vc) = (VsTrue::default(), VsTrue::default());
    for a in results {
        let a = a?;
        let p = a.s;
        s.inputs += p.inputs;
        s.bitwise_equal += p.bitwise_equal;
        s.both_nan += p.both_nan;
        s.special_mismatch += p.special_mismatch;
        if s.first_special_mismatch.is_none() {
            s.first_special_mismatch = p.first_special_mismatch;
        }
        if p.max_abs > s.max_abs {
            s.max_abs = p.max_abs;
            s.at_abs = p.at_abs;
            s.values_at_abs = p.values_at_abs;
        }
        if p.max_rel > s.max_rel {
            s.max_rel = p.max_rel;
            s.at_rel = p.at_rel;
        }
        if p.max_ulps > s.max_ulps {
            s.max_ulps = p.max_ulps;
            s.at_ulps = p.at_ulps;
            s.values_at_ulps = p.values_at_ulps;
        }
        merge_true(&mut vr, &a.vr);
        merge_true(&mut vc, &a.vc);
    }
    let mut scratch = Vec::new();
    let specials: Vec<(&str, u32)> = match ty {
        EType::F32 => vec![
            ("+inf", 0x7f80_0000),
            ("-inf", 0xff80_0000),
            ("nan", 0x7fc0_0000),
        ],
        EType::Bf16 => vec![("+inf", 0x7f80), ("-inf", 0xff80), ("nan", 0x7fc0)],
        EType::F16 => vec![("+inf", 0x7c00), ("-inf", 0xfc00), ("nan", 0x7e00)],
    };
    for (n, b) in specials {
        let yr = out_bits(r, ty, out, b, &mut scratch)?;
        let yc = out_bits(c, ty, out, b, &mut scratch)?;
        s.specials
            .push((n.to_string(), out_f64(out, yr), out_f64(out, yc)));
    }
    if let Some(f) = func {
        s.true_fn = Some(f.to_string());
        s.reference_vs_true = Some(vr);
        s.candidate_vs_true = Some(vc);
    }
    s.seconds = t0.elapsed().as_secs_f64();
    if let Some(m) = CACHE.lock().unwrap().as_mut() {
        m.insert(key, s.clone());
    }
    Ok(s)
}

/// Every (x, y) pair of a 16-bit product output `bf16(rn(f(x) * y))` (`ElemOut::round16`):
/// both kernels' outputs at all 2^32 pairs of 16-bit x and y.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ProductSweep {
    pub pairs: u64,
    pub bitwise_equal: u64,
    /// Both NaN (any payload) but not bitwise equal.
    pub both_nan: u64,
    /// One finite and the other not, or infinities of different signs.
    pub special_mismatch: u64,
    /// Over pairs where both are finite.
    pub max_abs: f64,
    pub max_ulps: u64,
    /// (x, y, reference, candidate) at max_ulps.
    pub at_ulps: (f64, f64, f64, f64),
    /// Pairs whose outputs are 1 ulp apart.
    pub one_ulp: u64,
    /// max_ulps over the pairs where f_R(x), f_C(x) and both outputs are normal (no
    /// underflow: the roundings are relative there).
    pub max_ulps_normal: u64,
    pub at_ulps_normal: (f64, f64, f64, f64),
    pub seconds: f64,
}

/// The 16-bit output of `rn(fv * y)` (fv, y f32 bits) in format `out`.
fn round_prod(fv: u32, y: u32, out: EType) -> u32 {
    let p = fp::f32_fma(fv, y, None, ptxv_ir::Rnd::Rn, false);
    match out {
        EType::Bf16 => fp::f32_to_bf16_rn(p) as u32,
        EType::F16 => fp::f32_to_f16_rn(p) as u32,
        EType::F32 => p,
    }
}

/// `ProductSweep` of reference `r` and candidate `c` (each `f` of a `round16` product output):
/// x of type `tx`, y of type `ty`, both 16-bit; outputs in `out` (bf16 or f16).
pub fn product_sweep(
    r: &ETree,
    c: &ETree,
    tx: EType,
    ty: EType,
    out: EType,
) -> Result<ProductSweep, String> {
    if tx == EType::F32 || ty == EType::F32 || out == EType::F32 {
        return Err("the pair sweep needs 16-bit x, y and output".into());
    }
    let t0 = std::time::Instant::now();
    let n = tx.count() as usize;
    let mut scratch = Vec::new();
    let mut fr = Vec::with_capacity(n);
    let mut fc = Vec::with_capacity(n);
    for b in 0..n as u32 {
        fr.push(r.eval_f32(tx.to_f32(b), &mut scratch)?);
        fc.push(c.eval_f32(tx.to_f32(b), &mut scratch)?);
    }
    let ys: Vec<u32> = (0..ty.count() as u32).map(|b| ty.to_f32(b)).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism()
        .map_or(8, |n| n.get())
        .min(128);
    let parts: Vec<ProductSweep> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut a = ProductSweep::default();
                    loop {
                        let x = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if x >= n {
                            break;
                        }
                        a.pairs += ys.len() as u64;
                        // Equal f: the same rounding of the same product for every y.
                        if fr[x] == fc[x] {
                            a.bitwise_equal += ys.len() as u64;
                            continue;
                        }
                        let normal32 = |v: u32| {
                            let e = (v >> 23) & 0xff;
                            e != 0 && e != 0xff
                        };
                        let f_normal = normal32(fr[x]) && normal32(fc[x]);
                        for &y in &ys {
                            let (or, oc) = (round_prod(fr[x], y, out), round_prod(fc[x], y, out));
                            if or == oc {
                                a.bitwise_equal += 1;
                                continue;
                            }
                            let (vr, vc) = (out.to_f64(or), out.to_f64(oc));
                            if out.is_nan(or) && out.is_nan(oc) {
                                a.both_nan += 1;
                            } else if !(vr.is_finite() && vc.is_finite()) {
                                if vr != vc {
                                    a.special_mismatch += 1;
                                }
                            } else {
                                a.max_abs = a.max_abs.max((vr - vc).abs());
                                let u = out.ulps_between(or, oc);
                                if u == 1 {
                                    a.one_ulp += 1;
                                }
                                let at = (tx.to_f64(x as u32), f32::from_bits(y) as f64, vr, vc);
                                if u > a.max_ulps {
                                    a.max_ulps = u;
                                    a.at_ulps = at;
                                }
                                if f_normal
                                    && u > a.max_ulps_normal
                                    && out.normal(or)
                                    && out.normal(oc)
                                {
                                    a.max_ulps_normal = u;
                                    a.at_ulps_normal = at;
                                }
                            }
                        }
                    }
                    a
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut s = ProductSweep::default();
    for p in parts {
        s.pairs += p.pairs;
        s.bitwise_equal += p.bitwise_equal;
        s.both_nan += p.both_nan;
        s.special_mismatch += p.special_mismatch;
        s.one_ulp += p.one_ulp;
        s.max_abs = s.max_abs.max(p.max_abs);
        if p.max_ulps > s.max_ulps {
            s.max_ulps = p.max_ulps;
            s.at_ulps = p.at_ulps;
        }
        if p.max_ulps_normal > s.max_ulps_normal {
            s.max_ulps_normal = p.max_ulps_normal;
            s.at_ulps_normal = p.at_ulps_normal;
        }
    }
    s.seconds = t0.elapsed().as_secs_f64();
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_known_values() {
        // erf(0.5), erf(1), erf(2), erfc(3), erfc(5), erfc(10) (A&S / mpmath to 17 digits).
        let close = |a: f64, b: f64| (a - b).abs() <= 2e-14 * b.abs();
        assert!(close(erf(0.5), 0.520_499_877_813_046_5));
        assert!(close(erf(1.0), 0.842_700_792_949_714_9));
        assert!(close(erf(2.0), 0.995_322_265_018_952_7));
        assert!(close(erfc(3.0), 2.209_049_699_858_544e-5));
        assert!(close(erfc(5.0), 1.537_459_794_428_034_8e-12));
        assert!(close(erfc(10.0), 2.088_487_583_762_544_7e-45));
        assert!(close(erfc(-1.0), 1.842_700_792_949_715));
        assert_eq!(erf(-0.0).to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn identity_tree_sweeps_bitwise() {
        let s = sweep(
            &ETree::X,
            &ETree::X,
            EType::Bf16,
            EType::F32,
            Some("identity"),
        )
        .unwrap();
        assert_eq!(s.inputs, 1 << 16);
        assert_eq!(s.bitwise_equal, 1 << 16);
        assert_eq!(s.reference_vs_true.unwrap().max_abs, 0.0);
    }

    #[test]
    fn ulps_cross_zero() {
        assert_eq!(EType::F32.ulps_between(0x0000_0000, 0x8000_0000), 1);
        assert_eq!(EType::F32.ulps_between(0x0000_0001, 0x8000_0001), 3);
        assert_eq!(EType::Bf16.ulps_between(0x3f80, 0x3f81), 1);
    }
}

#[cfg(test)]
mod bit_identity_tests {
    use super::*;

    #[test]
    fn copies_preserve_every_half_encoding() {
        for ty in [EType::Bf16, EType::F16] {
            let mut scratch = Vec::new();
            for bits in 0..=u16::MAX as u32 {
                assert_eq!(
                    out_bits(&ETree::X, ty, ty, bits, &mut scratch).unwrap(),
                    bits
                );
            }
        }
    }

    #[test]
    fn leaves_above_255_remain_distinct() {
        let mut state = NumState::default();
        let mut leaves = Vec::new();
        for index in 0..=256 {
            state.nodes.push(crate::num::Node {
                kind: NodeKind::Input((0, index)),
                hw: 0,
                exact: vec![],
                err: vec![],
                pc: 0,
            });
            let extracted = tree(&state, index as u32, &mut leaves, 0).unwrap();
            let mut dependencies = Vec::new();
            extracted.leaves(&mut dependencies);
            assert_eq!(dependencies, vec![index as usize]);
            assert!(matches!(extracted.bind(index as usize), ETree::X));
            if index == 256 {
                assert!(matches!(extracted.bind(0), ETree::K(0)));
            }
        }
    }

    #[test]
    fn copy_sweeps_do_not_reuse_old_rounding_cache_entries() {
        let reference = ETree::X;
        let candidate = ETree::Round(Box::new(ETree::X), false);
        let ty = EType::Bf16;
        let legacy_key = format!(
            "{:016x}-{:016x}-bf16-bf16--",
            reference.key(),
            candidate.key()
        );
        let stale = PairSweep {
            inputs: ty.count(),
            bitwise_equal: ty.count(),
            ..Default::default()
        };
        CACHE
            .lock()
            .unwrap()
            .get_or_insert_with(HashMap::new)
            .insert(legacy_key.clone(), stale);
        let sweep = sweep(&reference, &candidate, ty, ty, None).unwrap();
        CACHE.lock().unwrap().as_mut().unwrap().remove(&legacy_key);
        assert!(
            sweep.bitwise_equal < sweep.inputs,
            "copies preserve NaN encodings that rounding changes"
        );
    }
}

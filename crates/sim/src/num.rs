/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Numerics shadow execution.
//!
//! Along one (deterministic) execution, every fp32 value that carries rounding history gets
//! a node in a dataflow DAG. A node stores the hardware result, the *exact* value the same
//! algorithm would have produced in real arithmetic, and its own exact local rounding error
//! e_v = fl(a_v) - a_v. Values without a node are exact by construction (inputs, exact
//! conversions such as bf16 -> f32, integer bit manipulation of exact values).
//!
//! This is front-end independent: SASS converts bf16 to f32 with `SHF.L.U32 R, R, 0x10`,
//! PTX with `cvt.f32.bf16`; both are exact, so neither creates a node.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;

use ptxv_ir::*;
use serde::Serialize;

use crate::exact::Expansion;
use crate::exact::{self};
use crate::exec::Loc;
use crate::fp;
use crate::machine::Machine;

/// Where an input value came from: (allocation, element index).
pub type Orig = (usize, u64);

#[derive(Clone, Debug, Serialize)]
pub enum NodeKind {
    Leaf,
    /// Segment mode: an input-derived value without a node or a slice (untracked; `Leaf`s
    /// are then input-independent constants).
    Opaque,
    /// An element of an input buffer as a load read it (allocation, element index): exact,
    /// the input itself. Moves (registers, shuffles, shared memory) keep the node, so a pure
    /// data-movement kernel's outputs are these leaves.
    Input(Orig),
    /// fp32 add of two children; `rz`: rounded toward zero (else to nearest).
    Add(u32, u32, bool),
    /// fp32 atomic add (`red/atom.add.f32`: round to nearest, FTZ) of a value (second child)
    /// into a location holding the first child. The chain of them on one location adds its
    /// values in a schedule-dependent order (`equiv` bounds every order).
    AtomAdd(u32, u32),
    /// One tensor-core step (tcgen05.mma K-slice or HMMA): accumulator child (None = 0) plus
    /// exact product terms, each with the nodes of its two bf16 factors when they are known
    /// (`ops[i]` for `terms[i]`; input elements are `Input` leaves, rounded values `Round`
    /// nodes). Products of a provably zero factor are left out. `eps`: the calibrated model's
    /// contract for this step (`MmaModel::eps_terms` of the terms it can add):
    /// |hw - (acc + sum terms)| <= eps (|acc| + sum |terms|).
    Mma {
        acc: Option<u32>,
        terms: Vec<f64>,
        ops: Vec<Option<(u32, u32)>>,
        eps: f64,
        /// Block-scaled kinds: per term the nodes of its two scale factors (empty: unscaled);
        /// the term is ops.0 * ops.1 * sc.0 * sc.1.
        sc: Vec<(u32, u32)>,
        /// FP4 kinds (model `u16-f35-live-sfields`): per term its 16-element unit, whose exact
        /// sum the model truncates as one; empty where the model is order-invariant within the
        /// step.
        units: Vec<u8>,
    },
    /// The child rounded to bf16 (`f16`: to f16), to nearest even (`F2FP.BF16/F16.F32.PACK_AB`):
    /// the node's value is the 16-bit value widened to f32. |hw - child| <= u16 |child| + a16:
    /// bf16 2^-8 and 2^-134, f16 2^-11 and 2^-25 (`equiv::round_unit`; f16: no overflow).
    Round {
        arg: u32,
        f16: bool,
    },
    /// A MUFU / PTX approx op of one child (operand modifiers applied to it: `-|x|`, `-x`,
    /// `|x|`). Its `exact` is the true function of the child's exact value and its `err` the
    /// unit's error at the hardware input, both rounded to f64 (transcendentals are not
    /// expansions): Thm 1 is no longer an identity above such a node (`OutRec.n_mufu`).
    Mufu {
        arg: u32,
        func: MufuFunc,
        ptx: bool,
        neg: bool,
        abs: bool,
    },
    /// Segment mode: an fp32 FMA / FMUL (round to nearest, no saturation or FTZ) of tracked
    /// values: `(-1)^neg_ab * a * b + (-1)^neg_c * c`, one rounding.
    Fma {
        a: u32,
        b: u32,
        c: Option<u32>,
        neg_ab: bool,
        neg_c: bool,
    },
    /// Segment mode: the maximum (or minimum) of two tracked values, exact (`FMNMX`, or an
    /// `FSETP` + `SEL` pair choosing between its own operands).
    Max {
        a: u32,
        b: u32,
        min: bool,
    },
    /// Segment mode: a value computed from tracked nodes by an instruction slice (`seg.rs`).
    Seg(SegRef),
    /// A TF32 part of the f32 value `arg` (3xTF32 split, N-TF32): the high part the TF32 MMA
    /// reads, or a low-part factor of a split.
    Tf32 {
        arg: u32,
        part: Tf32Part,
    },
}

/// Which TF32 part of an f32 value x (its node's hw value) a `Tf32` node is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, serde::Deserialize)]
pub enum Tf32Part {
    /// hi(x) = x with the low 13 mantissa bits cleared: `LOP3 d, x, 0xffffe000`, or a TF32
    /// MMA operand read from x (the MMA truncates it).
    Hi,
    /// lo(x) = x - hi(x), exact (`FADD d, x, -hi`).
    Lo,
    /// The bits of lo(x) plus 0x1000 (CUTLASS's round-half-away conversion); as a TF32 MMA
    /// operand, lo(x) rounded half away from zero to 11 significant bits. The node's hw is
    /// that truncated value.
    LoR,
    /// lo(x) read by a TF32 MMA: truncated to 11 significant bits.
    LoT,
}

impl Tf32Part {
    /// For the low parts: their relative error delta against lo(x) as a TF32 MMA factor
    /// (|factor - lo| <= delta |lo|, lo normal or zero).
    pub fn lo_delta(self) -> Option<f64> {
        match self {
            Tf32Part::LoR => Some(2f64.powi(-11)),
            Tf32Part::LoT => Some(2f64.powi(-10)),
            Tf32Part::Hi | Tf32Part::Lo => None,
        }
    }
}

/// A shared slice (serialized as its size only).
#[derive(Clone, Debug)]
pub struct SegRef(pub std::sync::Arc<crate::seg::SProg>);

impl Serialize for SegRef {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&format!("slice of {} ops", self.0.ops.len()))
    }
}

/// Segment mode: what one executed instruction left in one destination register.
#[derive(Clone, Copy, Debug)]
pub struct Def {
    pub val: u64,
    pub tainted: bool,
    /// The tainted bits (bitwise LOP3 taint; every bit of a tainted register otherwise).
    pub tmask: u32,
    pub node: Option<u32>,
}

/// The true real function a MUFU op approximates (`ptx`: sin/cos take radians, else
/// revolutions), in f64.
// ASSUME(N-MUFU): see ASSUMPTIONS.md
pub fn mufu_true(func: MufuFunc, ptx: bool, x: f64) -> f64 {
    // An angle in revolutions: reduce exactly (x - round(x) is exact for |x| < 2^52).
    let rev = |x: f64| std::f64::consts::TAU * (x - x.round());
    match func {
        MufuFunc::Ex2 => x.exp2(),
        MufuFunc::Rcp | MufuFunc::Rcp64h => 1.0 / x,
        MufuFunc::Rsq | MufuFunc::Rsq64h => 1.0 / x.sqrt(),
        MufuFunc::Lg2 => x.log2(),
        MufuFunc::Sqrt => x.sqrt(),
        MufuFunc::Tanh => x.tanh(),
        MufuFunc::Sin if ptx => x.sin(),
        MufuFunc::Cos if ptx => x.cos(),
        MufuFunc::Sin => rev(x).sin(),
        MufuFunc::Cos => rev(x).cos(),
    }
}

/// d f / d x of `mufu_true`.
pub fn mufu_true_deriv(func: MufuFunc, ptx: bool, x: f64) -> f64 {
    let k = if ptx { 1.0 } else { std::f64::consts::TAU };
    match func {
        MufuFunc::Ex2 => std::f64::consts::LN_2 * x.exp2(),
        MufuFunc::Rcp | MufuFunc::Rcp64h => -1.0 / (x * x),
        MufuFunc::Rsq | MufuFunc::Rsq64h => -0.5 / (x * x.sqrt()),
        MufuFunc::Lg2 => 1.0 / (x * std::f64::consts::LN_2),
        MufuFunc::Sqrt => 0.5 / x.sqrt(),
        MufuFunc::Tanh => 1.0 - x.tanh().powi(2),
        MufuFunc::Sin => k * mufu_true(MufuFunc::Cos, ptx, x),
        MufuFunc::Cos => -k * mufu_true(MufuFunc::Sin, ptx, x),
    }
}

/// A-priori bound of one MUFU node's own error at operand `x` and hardware result `hw`
/// (ASSUME(N-MUFU)). ulp(f(x)) is taken just above |hw|, which covers f(x) in the next binade.
pub fn mufu_local_bound(func: MufuFunc, ptx: bool, x: f32, hw: f32) -> f64 {
    let b = crate::mufu::error_bound(func, ptx, x);
    let above = (hw.abs() as f64 * (1.0 + 2f64.powi(-20))) as f32;
    b.rel_ulps * ulp_f32(above.max(hw.abs())) + b.abs
}

#[derive(Clone, Debug, Serialize)]
pub struct Node {
    pub kind: NodeKind,
    /// Hardware value (f32 bits).
    pub hw: u32,
    /// Exact value of the same algorithm in real arithmetic.
    pub exact: Expansion,
    /// Exact local rounding error e_v = hw - (sum of the children's hw values).
    pub err: Expansion,
    pub pc: u32,
}

#[derive(Default, Debug, Clone)]
pub struct NumState {
    /// Error contract of one MMA step (|e_v| <= eps * sum |terms|), from the MMA model.
    pub mma_eps: f64,
    pub nodes: Vec<Node>,
    reg: HashMap<(u32, u8, u16), u32>,
    mem: HashMap<(u8, u64, u64), u32>,
    /// Origin of each shared-memory element a TMA load brought from an input buffer.
    orig: HashMap<(u16, u32), Orig>,
    /// Allocations whose elements are not inputs (outputs): no origin.
    pub not_inputs: BTreeSet<usize>,
    /// Input allocations of f32 elements: a 4-byte load of one of their words, not written
    /// since and still holding its initial bits, is an `Input` leaf.
    pub f32_inputs: BTreeSet<usize>,
    /// bf16 / f16 input allocations and the byte provenance of registers loaded from them
    /// (`halfin.rs`: widened elements become `Input` leaves).
    pub half: crate::halfin::HalfState,
    /// Global words (allocation, byte offset) written during the run.
    written: std::collections::HashSet<(u64, u64)>,
    /// fp32 atomic adds seen (their order is schedule-dependent).
    pub atomic_adds: usize,
    /// Other atomics, and fp32 atomic adds whose old value is returned.
    pub other_atomics: usize,
    pub taint: Taint,
    pub warnings: BTreeSet<String>,
    /// Segment mode (equiv runs that ask for it): products, maxima and instruction slices
    /// get nodes (`Fma`, `Max`, `Seg`); MUFU results do not (they are part of slices).
    pub segments: bool,
    /// Segment mode: per thread, the pc of every instruction it issued (`seg::GUARD_FALSE`
    /// set when its guard was false).
    pub trace: HashMap<u32, Vec<u32>>,
    /// Segment mode: (thread, trace position, register) -> what it left there.
    pub defs: HashMap<(u32, u32, u8, u16), Def>,
    /// Registers a numerics hook set during the current instruction.
    touched: Vec<(u32, u8, u16)>,
    /// Segment mode: per (thread, predicate), the FSETP that wrote it: (trace position,
    /// comparison, operand nodes).
    pred_src: HashMap<(u32, u16), (usize, FCmp, u32, u32)>,
    /// Instructions covered by materialized slices (their data-dependent guards are part of
    /// the slices), and why some value could not be materialized.
    pub seg_pcs: BTreeSet<u32>,
    /// Instructions of div.rn sequences (computed on first use).
    idiom_pcs: Option<BTreeSet<u32>>,
    pub seg_fails: BTreeMap<u32, String>,
    /// Leaves of input-independent constants (N-FINITE).
    const_leaves: std::collections::HashSet<u32>,
    /// `fin` per node.
    fin_memo: HashMap<u32, bool>,
}

/// A slice value for every input with finite elements and no overflow (N-FINITE): finite,
/// one constant (possibly an infinity), or unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FinVal {
    Fin,
    K(u64),
    Top,
}

/// Whether a slice's value is finite whenever its arguments are (`fin_args`) and no operation
/// overflows (N-FINITE): `seg_val` is finite or a finite constant.
// ASSUME(N-FINITE): see ASSUMPTIONS.md
pub(crate) fn seg_fin(p: &crate::seg::SProg, fin_args: &[bool]) -> bool {
    match seg_val(p, fin_args) {
        FinVal::Fin => true,
        FinVal::K(c) => f32::from_bits(c as u32).is_finite(),
        FinVal::Top => false,
    }
}

/// A slice's value for admissible inputs (N-FINITE), by abstract evaluation of its operations:
/// operations whose sources are all constants are evaluated (exec.rs's semantics); FADD /
/// FFMA (FMUL) / FMNMX / FSEL / SEL / MOV / IMAD.MOV / MUFU.EX2, TANH, SIN, COS of finite
/// values and finite constants are finite (FMNMX: or its identity infinity); an FADD of an
/// infinity and a finite value is that infinity (exact in every rounding mode); a selection
/// by a constant predicate is the chosen value, otherwise the join of both; everything else
/// (integer bits) is unknown, which only predicates may consume.
// ASSUME(N-FINITE): see ASSUMPTIONS.md
pub(crate) fn seg_val(p: &crate::seg::SProg, fin_args: &[bool]) -> FinVal {
    use FinVal::*;

    use crate::seg::SKind;
    use crate::seg::SVal;
    let join = |a: FinVal, b: FinVal| -> FinVal {
        let fin = |x: FinVal| match x {
            Fin => true,
            K(c) => f32::from_bits(c as u32).is_finite(),
            Top => false,
        };
        if a == b {
            a
        } else if fin(a) && fin(b) {
            Fin
        } else {
            Top
        }
    };
    let mut f: Vec<FinVal> = Vec::with_capacity(p.ops.len());
    let val = |v: SVal, f: &[FinVal]| match v {
        SVal::Arg(i) => {
            if fin_args.get(i as usize).copied().unwrap_or(false) {
                Fin
            } else {
                Top
            }
        }
        SVal::Const(c) => K(c),
        SVal::Op(i) => f[i as usize],
    };
    for o in &p.ops {
        let r = match &o.kind {
            SKind::Guard {
                pred,
                neg,
                new,
                old,
            } => match val(*pred, &f) {
                K(c) => {
                    if (c != 0) != *neg {
                        val(*new, &f)
                    } else {
                        val(*old, &f)
                    }
                }
                _ => join(val(*new, &f), val(*old, &f)),
            },
            SKind::DivRn { .. } => Top,
            SKind::Inst {
                op,
                dst,
                regs,
                cvals,
            } => {
                let srcs: Vec<(Reg, FinVal)> =
                    regs.iter().map(|(r, s)| (*r, val(*s, &f))).collect();
                let all_k = srcs.iter().all(|(_, a)| matches!(a, K(_)));
                // An operand's value: a register's abstract value, an immediate or a constant
                // bank word.
                let av = |v: &Val| -> FinVal {
                    match v {
                        Val::Reg(r, _) => srcs.iter().find(|(x, _)| x == r).map_or(Top, |x| x.1),
                        Val::Imm(c) => K(*c),
                        c => cvals.iter().find(|(x, _)| x == c).map_or(Top, |x| K(x.1)),
                    }
                };
                let pk = |pr: &Pred| -> Option<bool> {
                    match pr.reg {
                        None => Some(!pr.neg),
                        Some(r) => match srcs.iter().find(|(x, _)| *x == r).map(|x| x.1) {
                            Some(K(c)) => Some((c != 0) != pr.neg),
                            _ => None,
                        },
                    }
                };
                let is_fin = |a: FinVal| match a {
                    Fin => true,
                    K(c) => f32::from_bits(c as u32).is_finite(),
                    Top => false,
                };
                if all_k {
                    let rv = |v: &Val| -> u64 {
                        match av(v) {
                            K(c) => c,
                            _ => 0,
                        }
                    };
                    let rp = |pr: &Pred| pk(pr).unwrap_or(false);
                    match crate::seg::eval_inst(op, *dst, &rv, &rp) {
                        Ok(x) => K(x),
                        Err(_) => Top,
                    }
                } else if matches!(dst.file, File::P | File::UP) {
                    Top
                } else {
                    match op {
                        Op::Fadd {
                            a,
                            b,
                            neg,
                            abs,
                            sat,
                            ..
                        } => {
                            let (x, y) = (av(a), av(b));
                            let inf = |v: FinVal, i: usize| match v {
                                K(c) => {
                                    let m = fmod_bits(c as u32, neg[i], abs[i]);
                                    f32::from_bits(m).is_infinite().then_some(m)
                                }
                                _ => None,
                            };
                            match (inf(x, 0), inf(y, 1)) {
                                (Some(m), None) if is_fin(y) => {
                                    K((if *sat { fp::f32_sat(m) } else { m }) as u64)
                                }
                                (None, Some(m)) if is_fin(x) => {
                                    K((if *sat { fp::f32_sat(m) } else { m }) as u64)
                                }
                                _ if is_fin(x) && is_fin(y) => Fin,
                                _ => Top,
                            }
                        }
                        Op::Ffma { a, b, c, .. } => {
                            if is_fin(av(a))
                                && is_fin(av(b))
                                && c.as_ref().is_none_or(|c| is_fin(av(c)))
                            {
                                Fin
                            } else {
                                Top
                            }
                        }
                        Op::Fmnmx {
                            a,
                            b,
                            c,
                            min,
                            neg,
                            abs,
                            nan: false,
                            ..
                        } => {
                            // The identity of a max (min) with a constant choice.
                            let ident = match pk(min) {
                                Some(true) => Some(0x7f80_0000u32),
                                Some(false) => Some(0xff80_0000u32),
                                None => None,
                            };
                            let ops: Vec<(FinVal, usize)> = [Some(a), Some(b), c.as_ref()]
                                .into_iter()
                                .enumerate()
                                .filter_map(|(i, v)| v.map(|v| (av(v), i)))
                                .collect();
                            let ok = |(v, i): (FinVal, usize)| {
                                is_fin(v)
                                    || matches!(v, K(k) if Some(fmod_bits(k as u32, neg[i], abs[i])) == ident)
                            };
                            if ops.iter().all(|&x| ok(x)) && ops.iter().any(|&(v, _)| is_fin(v)) {
                                Fin
                            } else {
                                Top
                            }
                        }
                        Op::Fsel { a, b, p, .. } | Op::Sel { a, b, p, .. } => {
                            // FSEL's |x| and -x keep finiteness; a constant becomes another
                            // constant (not tracked: finite stays finite).
                            let m = |v: FinVal| match (op, v) {
                                (Op::Fsel { neg, abs, .. }, K(c))
                                    if neg[0] || neg[1] || abs[0] || abs[1] =>
                                {
                                    if f32::from_bits(c as u32).is_finite() {
                                        Fin
                                    } else {
                                        Top
                                    }
                                }
                                _ => v,
                            };
                            match pk(p) {
                                Some(true) => m(av(a)),
                                Some(false) => m(av(b)),
                                None => join(m(av(a)), m(av(b))),
                            }
                        }
                        Op::Mov { a, .. } => av(a),
                        // `IMAD.MOV d, RZ, RZ, c`: a copy of c.
                        Op::Imad {
                            a,
                            b,
                            c,
                            neg_c: false,
                            not_c: false,
                            wide: false,
                            hi: false,
                            cin: None,
                            cout: None,
                            ..
                        } if matches!(a, Val::Imm(0)) || matches!(b, Val::Imm(0)) => av(c),
                        Op::Mufu {
                            func,
                            fmt: MufuFmt::F32,
                            a,
                            ..
                        } if matches!(
                            func,
                            MufuFunc::Ex2 | MufuFunc::Tanh | MufuFunc::Sin | MufuFunc::Cos
                        ) && is_fin(av(a)) =>
                        {
                            Fin
                        }
                        _ => Top,
                    }
                }
            }
        };
        f.push(r);
    }
    val(p.out, &f)
}

/// Bitwise taint of `LOP3 d = lut(a, b, c)` (`vals`, `masks`: a, b, c and d's old value and
/// tainted bits; `keep`: d may keep its old value, a data-dependent guard): (the result bits
/// that can take both values as the tainted bits vary, the bits that are 1 for every input).
pub(crate) fn lop3_taint(lut: u8, vals: [u32; 4], masks: [u32; 4], keep: bool) -> (u32, u32) {
    let (mut m, mut one) = (0u32, 0u32);
    for i in 0..32 {
        let cand = |j: usize| -> &'static [u32] {
            if masks[j] >> i & 1 == 1 {
                &[0, 1]
            } else if vals[j] >> i & 1 == 1 {
                &[1]
            } else {
                &[0]
            }
        };
        let mut outs = 0u8;
        for &x in cand(0) {
            for &y in cand(1) {
                for &z in cand(2) {
                    outs |= 1 << ((lut >> ((x << 2) | (y << 1) | z)) & 1);
                }
            }
        }
        if keep {
            for &o in cand(3) {
                outs |= 1 << o;
            }
        }
        if outs == 3 {
            m |= 1 << i;
        } else if outs == 2 {
            one |= 1 << i;
        }
    }
    (m, one)
}

/// Bits of an f32 operand after its `|x|` and `-x` modifiers.
fn fmod_bits(u: u32, neg: bool, abs: bool) -> u32 {
    let u = if abs { u & 0x7fff_ffff } else { u };
    if neg { u ^ 0x8000_0000 } else { u }
}

/// CUTLASS's float atomic max (`atomicMaxFloat`, N-FMAX): `ISETP.GE P, v, RZ` (or `.GT P, v,
/// -1`), straight-line to `@[!]P BRA T`; one side `ATOMS.MAX.S32 RZ, [a], v`, the other
/// `ATOMS.MIN RZ, [a], v` (unsigned), the same address and value, reconverging right after:
/// the signed max where v's sign bit is clear, the unsigned min where it is set. Returns the
/// two atomics' pcs for the branch at `bra`.
// ASSUME(N-FMAX): see ASSUMPTIONS.md
pub fn fmax_diamond(code: &[Inst], bra: usize) -> Option<(usize, usize)> {
    let b = code.get(bra)?;
    let Op::Bra {
        target,
        any: false,
        tid0_falls: false,
        cond: None,
        div: None,
    } = b.op
    else {
        return None;
    };
    let p = b.guard.reg?;
    let atom = |i: usize| match &code.get(i)?.op {
        Op::Atom {
            op,
            space: Space::Shared,
            bytes: 4,
            addr,
            v: Val::Reg(v, 1),
            d: None,
            p: None,
            ..
        } if code[i].guard.is_always() && matches!(op, AtomOp::MaxS | AtomOp::MinU) => {
            Some((*op, addr.clone(), *v))
        }
        _ => None,
    };
    let (op1, a1, v1) = atom(bra + 1)?;
    let Op::Bra {
        target: join,
        any: false,
        tid0_falls: false,
        cond: None,
        div: None,
    } = code.get(bra + 2)?.op
    else {
        return None;
    };
    if !code[bra + 2].guard.is_always() || target != bra + 3 || join != bra + 4 {
        return None;
    }
    let (op2, a2, v2) = atom(bra + 3)?;
    if op1 == op2 || a1 != a2 || v1 != v2 {
        return None;
    }
    // The predicate: the last write of P before the branch, in straight-line code that writes
    // neither P nor v after it and that no branch enters.
    let mut i = bra;
    let setp = loop {
        i = i.checked_sub(1)?;
        let inst = &code[i];
        if inst.op.dst_regs().contains(&p) {
            break i;
        }
        if inst.op.dst_regs().contains(&v1)
            || matches!(
                inst.op,
                Op::Bra { .. } | Op::Exit | Op::Call { .. } | Op::Ret { .. }
            )
        {
            return None;
        }
    };
    let entered = code
        .iter()
        .any(|x| matches!(x.op, Op::Bra { target, .. } if target > setp && target <= bra));
    if entered || !code[setp].guard.is_always() {
        return None;
    }
    let Op::Setp {
        d,
        cmp,
        bits: 32,
        a: Val::Reg(sa, _),
        b: ref sb,
        and,
        or: false,
        xor: false,
        ex: None,
    } = code[setp].op
    else {
        return None;
    };
    let k = match sb {
        Val::Imm(x) => *x as u32,
        _ => return None,
    };
    // P = (v as i32 >= 0).
    let sign_clear = match (cmp, k) {
        (Cmp::GeS, 0) | (Cmp::GtS, 0xffff_ffff) => true,
        _ => return None,
    };
    if d != p || sa != v1 || !and.is_always() || !sign_clear {
        return None;
    }
    // Taken where the sign bit is clear (P) unless the guard is negated: then the target must
    // be the signed max.
    let taken_clear = !b.guard.neg;
    let (max_pc, min_pc) = if (op2 == AtomOp::MaxS) == taken_clear {
        (bra + 3, bra + 1)
    } else {
        return None;
    };
    Some((max_pc, min_pc))
}

/// Bits of an operand before its `-x` modifier (`vals` hold them after).
fn fmod_sign(u: u32, neg: bool) -> u32 {
    if neg { u ^ 0x8000_0000 } else { u }
}

fn file_tag(f: File) -> u8 {
    f as u8
}

fn mem_key(loc: Loc) -> (u8, u64, u64) {
    match loc {
        Loc::Shared { cta, off } => (0, cta as u64, off as u64),
        Loc::Global { alloc, off } => (1, alloc as u64, off),
        Loc::Param { grid, off } => (2, grid as u64, off as u64),
    }
}

impl NumState {
    /// A readable rendering of node `n`'s DAG (debugging: `PTXV_EQUIV_DUMP`), `depth` levels.
    pub fn dump(&self, n: u32, depth: usize) -> String {
        let nd = &self.nodes[n as usize];
        let v = f32::from_bits(nd.hw);
        if depth == 0 {
            return format!("#{n}={v}");
        }
        let d = |x: u32| self.dump(x, depth - 1);
        match &nd.kind {
            NodeKind::Leaf => format!("{v}"),
            NodeKind::Opaque => format!("opaque({v})"),
            NodeKind::Input((a, i)) => format!("in{a}[{i}]"),
            NodeKind::Add(a, b, _) => format!("({} + {})", d(*a), d(*b)),
            NodeKind::AtomAdd(a, b) => format!("atom({} + {})", d(*a), d(*b)),
            NodeKind::Fma { a, b, c, .. } => format!(
                "fma({} * {}{})",
                d(*a),
                d(*b),
                c.map(|c| format!(" + {}", d(c))).unwrap_or_default()
            ),
            NodeKind::Max { a, b, min } => {
                format!("{}({}, {})", if *min { "min" } else { "max" }, d(*a), d(*b))
            }
            NodeKind::Seg(p) => format!(
                "slice#{n}[{} ops]({})",
                p.0.ops.len(),
                p.0.args
                    .iter()
                    .map(|&x| d(x))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            NodeKind::Mma { acc, ops, .. } => format!(
                "mma#{n}[{}; {}]",
                acc.map(&d).unwrap_or_else(|| "0".into()),
                ops.iter()
                    .map(|o| match o {
                        Some((x, y)) => format!("{}*{}", d(*x), d(*y)),
                        None => "?".into(),
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            NodeKind::Round { arg, f16 } => {
                format!("{}({})", if *f16 { "f16" } else { "bf16" }, d(*arg))
            }
            NodeKind::Mufu { func, arg, .. } => format!("{func:?}({})", d(*arg)),
            NodeKind::Tf32 { arg, part } => format!("tf32{part:?}({})", d(*arg)),
        }
    }

    pub fn with_mma_eps(eps: f64) -> NumState {
        NumState {
            mma_eps: eps,
            ..Default::default()
        }
    }

    fn node_of(&self, t: usize, v: &Val) -> Option<u32> {
        match v {
            Val::Reg(r, _) => self.reg.get(&(t as u32, file_tag(r.file), r.idx)).copied(),
            _ => None,
        }
    }
    pub(crate) fn node_of_reg(&self, t: usize, r: Reg) -> Option<u32> {
        self.reg.get(&(t as u32, file_tag(r.file), r.idx)).copied()
    }
    pub(crate) fn set_reg_node(&mut self, t: usize, d: Reg, n: Option<u32>) {
        self.set(t, d, n);
    }
    fn set(&mut self, t: usize, d: Reg, n: Option<u32>) {
        let k = (t as u32, file_tag(d.file), d.idx);
        self.touched.push(k);
        match n {
            Some(n) => {
                self.reg.insert(k, n);
            }
            None => {
                self.reg.remove(&k);
            }
        }
    }
    pub(crate) fn leaf(&mut self, bits: u32, pc: u32) -> u32 {
        self.nodes.push(Node {
            kind: NodeKind::Leaf,
            hw: bits,
            exact: exact::from(f32::from_bits(bits) as f64),
            err: vec![],
            pc,
        });
        (self.nodes.len() - 1) as u32
    }
    /// A `Tf32` node of `arg` with hardware value `hw`: hi(x) has x's exact value (its own
    /// error is the truncation), the low parts are errors (exact value 0).
    fn tf32_node(&mut self, arg: u32, part: Tf32Part, hw: u32, pc: u32) -> u32 {
        let a = &self.nodes[arg as usize];
        let (exact, err) = match part {
            Tf32Part::Hi => (
                a.exact.clone(),
                exact::sub(
                    &exact::from(f32::from_bits(hw) as f64),
                    &exact::from(f32::from_bits(a.hw) as f64),
                ),
            ),
            _ => (vec![], exact::from(f32::from_bits(hw) as f64)),
        };
        self.nodes.push(Node {
            kind: NodeKind::Tf32 { arg, part },
            hw,
            exact,
            err,
            pc,
        });
        (self.nodes.len() - 1) as u32
    }
    /// An exact max / min node of two nodes.
    fn max_node(&mut self, a: u32, b: u32, min: bool, pc: u32) -> u32 {
        let (ea, eb) = (&self.nodes[a as usize].exact, &self.nodes[b as usize].exact);
        let a_big = exact::approx(&exact::sub(ea, eb)) >= 0.0;
        let pick_a = a_big != min;
        let (ha, hb) = (
            f32::from_bits(self.nodes[a as usize].hw),
            f32::from_bits(self.nodes[b as usize].hw),
        );
        let hw = if (ha >= hb) != min { ha } else { hb };
        let exact = if pick_a { ea.clone() } else { eb.clone() };
        self.nodes.push(Node {
            kind: NodeKind::Max { a, b, min },
            hw: hw.to_bits(),
            exact,
            err: vec![],
            pc,
        });
        (self.nodes.len() - 1) as u32
    }
    /// An `Mma` node: acc + products (`ops`: their factors' provenance; zero products are
    /// left out), result `r`, the model's contract for the terms it can add.
    #[allow(clippy::too_many_arguments)]
    fn push_mma(
        &mut self,
        acc_node: Option<u32>,
        acc: f32,
        prods: &[f64],
        ops: &[crate::halfin::HalfOp],
        r: f32,
        pc: u32,
        eps_of: &dyn Fn(usize, bool) -> f64,
        scales: Option<&[crate::halfin::HalfOp]>,
        unit: usize,
    ) -> u32 {
        use crate::halfin::HalfOp;
        let mut ex = acc_node
            .map(|n| self.nodes[n as usize].exact.clone())
            .unwrap_or_default();
        let mut a_v = if acc_node.is_some() {
            exact::from(acc as f64)
        } else {
            vec![]
        };
        let (mut terms, mut tops, mut tsc, mut tun) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, &p) in prods.iter().enumerate() {
            let (oa, ob) = (ops[2 * i], ops[2 * i + 1]);
            let (sa, sb) = scales.map_or((HalfOp::Unknown, HalfOp::Unknown), |s| {
                (s[2 * i], s[2 * i + 1])
            });
            if [oa, ob].contains(&HalfOp::Zero)
                || scales.is_some() && [sa, sb].contains(&HalfOp::Zero)
            {
                continue;
            }
            a_v = exact::add(&a_v, &exact::from(p));
            let (o, s) = match (oa, ob, sa, sb) {
                (HalfOp::Node(x), HalfOp::Node(y), _, _) if scales.is_none() => {
                    ex = exact::add(
                        &ex,
                        &exact::mul(&self.nodes[x as usize].exact, &self.nodes[y as usize].exact),
                    );
                    (Some((x, y)), (0, 0))
                }
                (HalfOp::Node(x), HalfOp::Node(y), HalfOp::Node(u), HalfOp::Node(v)) => {
                    let e = |k: u32| &self.nodes[k as usize].exact;
                    ex = exact::add(
                        &ex,
                        &exact::mul(&exact::mul(e(x), e(y)), &exact::mul(e(u), e(v))),
                    );
                    (Some((x, y)), (u, v))
                }
                _ => {
                    ex = exact::add(&ex, &exact::from(p));
                    (None, (0, 0))
                }
            };
            terms.push(p);
            tops.push(o);
            tsc.push(s);
            if let Some(q) = i.checked_div(unit) {
                tun.push(q as u8);
            }
        }
        if scales.is_none() {
            tsc.clear();
        }
        // An accumulator that is the constant +0 adds nothing.
        let acc_counts = acc_node.is_some_and(|a| {
            !(matches!(self.nodes[a as usize].kind, NodeKind::Leaf)
                && self.nodes[a as usize].hw == 0)
        });
        let eps = eps_of(terms.len(), acc_counts);
        let err = exact::sub(&exact::from(r as f64), &a_v);
        self.nodes.push(Node {
            kind: NodeKind::Mma {
                acc: acc_node,
                terms,
                ops: tops,
                eps,
                sc: tsc,
                units: tun,
            },
            hw: r.to_bits(),
            exact: ex,
            err,
            pc,
        });
        (self.nodes.len() - 1) as u32
    }
    /// A leaf for operand `v` of thread `t` without a node: in segment mode `Opaque` when it
    /// is input-derived, else a constant `Leaf`.
    fn leaf_for(&mut self, t: usize, v: &Val, bits: u32, pc: u32) -> u32 {
        let n = self.leaf(bits, pc);
        if matches!(v, Val::Reg(r, _) if self.taint.reg_t(t, *r)) {
            if self.segments {
                self.nodes[n as usize].kind = NodeKind::Opaque;
            }
        } else {
            self.const_leaves.insert(n);
        }
        n
    }

    /// A leaf of an input-independent constant.
    fn const_leaf(&mut self, bits: u32, pc: u32) -> u32 {
        let n = self.leaf(bits, pc);
        self.const_leaves.insert(n);
        n
    }

    /// Whether node `n` is finite for every input with finite elements on which no operation
    /// of the DAG overflows (N-FINITE): inputs, finite constants, and sums, products, MMA steps,
    /// roundings, exp2 / tanh / sin / cos and TF32 parts of such nodes; a max (min) of one
    /// such node and a constant other than +inf (-inf).
    // ASSUME(N-FINITE): see ASSUMPTIONS.md
    pub(crate) fn fin(&mut self, root: u32) -> bool {
        let mut stack = vec![(root, false)];
        while let Some((n, done)) = stack.pop() {
            if self.fin_memo.contains_key(&n) {
                continue;
            }
            let nd = &self.nodes[n as usize];
            let kids: Vec<u32> = match &nd.kind {
                NodeKind::Leaf | NodeKind::Opaque | NodeKind::Input(_) => vec![],
                NodeKind::Seg(p) => p.0.args.clone(),
                NodeKind::Add(a, b, _) | NodeKind::AtomAdd(a, b) => vec![*a, *b],
                NodeKind::Fma { a, b, c, .. } => {
                    [Some(*a), Some(*b), *c].into_iter().flatten().collect()
                }
                NodeKind::Max { a, b, .. } => vec![*a, *b],
                NodeKind::Mma { acc, ops, .. } => acc
                    .iter()
                    .copied()
                    .chain(ops.iter().flatten().flat_map(|&(x, y)| [x, y]))
                    .collect(),
                NodeKind::Round { arg, .. }
                | NodeKind::Mufu { arg, .. }
                | NodeKind::Tf32 { arg, .. } => {
                    vec![*arg]
                }
            };
            if !done {
                stack.push((n, true));
                stack.extend(
                    kids.iter()
                        .filter(|k| !self.fin_memo.contains_key(k))
                        .map(|&k| (k, false)),
                );
                continue;
            }
            let f = |k: &u32| self.fin_memo.get(k).copied().unwrap_or(false);
            let konst = |k: u32| {
                matches!(self.nodes[k as usize].kind, NodeKind::Leaf)
                    && self.const_leaves.contains(&k)
            };
            let v = match &nd.kind {
                NodeKind::Input(_) => true,
                NodeKind::Leaf => konst(n) && f32::from_bits(nd.hw).is_finite(),
                NodeKind::Opaque => false,
                // A slice without arguments is the same value for every input.
                NodeKind::Seg(p) if p.0.args.is_empty() => {
                    p.0.eval(&[]).is_ok_and(|v| f32::from_bits(v).is_finite())
                }
                NodeKind::Seg(p) => {
                    let args: Vec<bool> = p.0.args.iter().map(f).collect();
                    seg_fin(&p.0, &args)
                }
                NodeKind::Mma { terms, ops, .. } => {
                    kids.iter().all(f)
                        && ops.iter().zip(terms).all(|(o, t)| o.is_some() || *t == 0.0)
                }
                NodeKind::Max { a, b, min } => {
                    let bad = if *min { 0xff80_0000 } else { 0x7f80_0000 };
                    let ok_const = |k: u32| konst(k) && self.nodes[k as usize].hw != bad;
                    (f(a) && f(b)) || (f(a) && ok_const(*b)) || (f(b) && ok_const(*a))
                }
                NodeKind::Mufu { func, .. } => {
                    matches!(
                        func,
                        MufuFunc::Ex2 | MufuFunc::Tanh | MufuFunc::Sin | MufuFunc::Cos
                    ) && kids.iter().all(f)
                }
                _ => kids.iter().all(f),
            };
            self.fin_memo.insert(n, v);
        }
        self.fin_memo[&root]
    }
    pub fn mem_node(&self, loc: Loc) -> Option<u32> {
        self.mem.get(&mem_key(loc)).copied()
    }
    pub(crate) fn forget_orig(&mut self, cta: u16, off: u32, n: usize) {
        if self.orig.is_empty() {
            return;
        }
        // Elements are 2-byte aligned bf16 (the only origins recorded so far).
        for o in (off & !1..off + n as u32).step_by(2) {
            self.orig.remove(&(cta, o));
        }
    }
    pub fn orig_at(&self, cta: u16, off: u32) -> Option<Orig> {
        self.orig.get(&(cta, off)).copied()
    }
    fn note_write(&mut self, k: (u8, u64, u64)) {
        if k.0 == 1 {
            self.written.insert((k.1, k.2));
        }
    }
}

impl Machine {
    /// Segment mode: the node of operand `v` of thread `t`, materializing an input-derived
    /// register without one as a slice (`seg.rs`). None: a constant, or no slice (recorded in
    /// `seg_fails`; the caller then treats it as untracked).
    pub(crate) fn num_node_mat(&mut self, t: usize, v: &Val) -> Option<u32> {
        let ns = self.num.as_ref()?;
        if let Some(n) = ns.node_of(t, v) {
            return Some(n);
        }
        let Val::Reg(r, _) = v else { return None };
        if !ns.segments || !ns.taint.reg_t(t, *r) {
            return None;
        }
        let pc =
            ns.trace.get(&(t as u32)).and_then(|v| v.last()).copied()? & !crate::seg::GUARD_FALSE;
        match self.seg_slice(t, *r) {
            Ok(p) => {
                let bits = self.rd(t, &Val::Reg(*r, 1)).ok()? as u32;
                let ns = self.num.as_mut().unwrap();
                ns.seg_pcs.extend(p.pcs());
                ns.nodes.push(Node {
                    kind: NodeKind::Seg(SegRef(std::sync::Arc::new(p))),
                    hw: bits,
                    exact: exact::from(f32::from_bits(bits) as f64),
                    err: vec![],
                    pc,
                });
                let id = (ns.nodes.len() - 1) as u32;
                // Later reads of the register reuse it.
                let k = (t as u32, file_tag(r.file), r.idx);
                ns.reg.insert(k, id);
                Some(id)
            }
            Err(e) => {
                let ns = self.num.as_mut().unwrap();
                ns.seg_fails.entry(pc).or_insert(e);
                None
            }
        }
    }

    /// Segment mode: FFMA / FMUL `d = (-1)^na a * (-1)^nb b + (-1)^nc c` (round to nearest,
    /// no FTZ / saturation / |x|; `vals`: the operands' bits after the modifiers, `r` the
    /// result). A node when it combines at least two tracked values, or a tracked value with
    /// a slice of other arguments (a rescaling); otherwise part of a slice (no node).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn num_ffma(
        &mut self,
        t: usize,
        d: Reg,
        ops: (&Val, &Val, Option<&Val>),
        neg: [bool; 3],
        vals: [u32; 3],
        r: u32,
        pc: u32,
    ) {
        if self.num.is_none() {
            return;
        }
        match self.num_ffma_node(t, ops, neg, vals, r, pc) {
            Some(id) => self.num.as_mut().unwrap().set(t, d, Some(id)),
            None => {
                let srcs: Vec<&Val> = [Some(ops.0), Some(ops.1), ops.2]
                    .into_iter()
                    .flatten()
                    .collect();
                self.num_clobber(t, d, &srcs, pc);
            }
        }
    }

    /// `num_ffma`'s node, without writing a register (None: the destination is clobbered).
    #[allow(clippy::too_many_arguments)]
    fn num_ffma_node(
        &mut self,
        t: usize,
        ops: (&Val, &Val, Option<&Val>),
        neg: [bool; 3],
        vals: [u32; 3],
        r: u32,
        pc: u32,
    ) -> Option<u32> {
        let ns = self.num.as_ref()?;
        let srcs: Vec<&Val> = [Some(ops.0), Some(ops.1), ops.2]
            .into_iter()
            .flatten()
            .collect();
        if !ns.segments || !self.num_anchor(t, &srcs, pc) {
            return None;
        }
        let mut nodes = Vec::new();
        for (i, v) in srcs.iter().enumerate() {
            let n = match self.num_node_mat(t, v) {
                Some(n) => n,
                None => {
                    let ns = self.num.as_mut().unwrap();
                    ns.leaf_for(t, v, fmod_sign(vals[i], neg[i]), pc)
                }
            };
            nodes.push(n);
        }
        let ns = self.num.as_mut().unwrap();
        let c = ops.2.map(|_| nodes[2]);
        let (a, b) = (nodes[0], nodes[1]);
        let mut ex = exact::mul(&ns.nodes[a as usize].exact, &ns.nodes[b as usize].exact);
        if neg[0] != neg[1] {
            ex = exact::neg(&ex);
        }
        let mut av = exact::mul(
            &exact::from(f32::from_bits(vals[0]) as f64),
            &exact::from(f32::from_bits(vals[1]) as f64),
        );
        if let Some(c) = c {
            let ce = &ns.nodes[c as usize].exact;
            ex = exact::add(&ex, &if neg[2] { exact::neg(ce) } else { ce.clone() });
            av = exact::add(&av, &exact::from(f32::from_bits(vals[2]) as f64));
        }
        let err = exact::sub(&exact::from(f32::from_bits(r) as f64), &av);
        ns.nodes.push(Node {
            kind: NodeKind::Fma {
                a,
                b,
                c,
                neg_ab: neg[0] != neg[1],
                neg_c: neg[2],
            },
            hw: r,
            exact: ex,
            err,
            pc,
        });
        Some((ns.nodes.len() - 1) as u32)
    }

    /// Segment mode: does an FADD / FFMA of these operands get a node? When two of its
    /// non-constant operands depend on different tracked values (a node's set is itself, an
    /// input-derived value without one: its slice's arguments), and it is not inside a div.rn
    /// sequence (`seg::divrn_at`, one opaque quotient). Otherwise it is part of a slice: an exp
    /// idiom's range reduction depends on its argument alone, Newton steps on the divisor.
    pub(crate) fn num_anchor(&mut self, t: usize, srcs: &[&Val], pc: u32) -> bool {
        let Some(ns) = self.num.as_mut() else {
            return false;
        };
        if ns.idiom_pcs.is_none() {
            let k = self.k.clone();
            let mut set = BTreeSet::new();
            for (q, i) in k.code.iter().enumerate() {
                if matches!(i.op, Op::Bsync { .. }) {
                    if crate::seg::rcprn_at(self, q as u32).is_some() {
                        // The fast path's Newton step (not instructions interleaved with it).
                        set.extend(q as u32 - 4..=q as u32);
                    }
                    if let Some((_, b, _)) = crate::seg::divrn_at(self, q as u32) {
                        // From the MUFU.RCP of the divisor to the BSYNC.
                        let lo = (0..q)
                            .rev()
                            .take(20)
                            .find(|&p| {
                                matches!(&k.code[p].op, Op::Mufu { func: MufuFunc::Rcp, a: Val::Reg(x, _), .. } if *x == b)
                            })
                            .unwrap_or(q);
                        set.extend(lo as u32..=q as u32);
                    }
                }
            }
            self.num.as_mut().unwrap().idiom_pcs = Some(set);
        }
        let ns = self.num.as_ref().unwrap();
        if ns.idiom_pcs.as_ref().unwrap().contains(&pc) {
            return false;
        }
        let mut sets: Vec<Option<Vec<u32>>> = Vec::new();
        let mut direct = 0;
        for v in srcs {
            let ns = self.num.as_ref().unwrap();
            if let Some(n) = ns.node_of(t, v) {
                sets.push(Some(vec![n]));
                direct += 1;
                continue;
            }
            match v {
                Val::Reg(r, _) if ns.taint.reg_t(t, *r) => {
                    let mut a = self.seg_slice(t, *r).ok().map(|p| p.args);
                    if let Some(a) = a.as_mut() {
                        a.sort();
                    }
                    sets.push(a);
                }
                _ => {}
            }
        }
        // Two operands that are tracked nodes themselves (the same one twice: a square x * x,
        // as in sums of squares) make a node; an exp idiom's steps read one node and constants.
        // An operand that cannot be sliced counts as different (it stays untracked anyway).
        direct >= 2
            || sets.windows(2).any(|w| w[0] != w[1] || w[0].is_none())
            || sets.len() >= 2 && sets.iter().all(|s| s.is_none())
    }

    /// Segment mode: FADD `d = (-1)^na a + (-1)^nb b` (round to nearest or toward zero, no
    /// FTZ / saturation / |x|): an `Add` node, a negated form as `Fma` with the constant 1,
    /// or part of a slice (`num_anchor`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn num_fadd_seg(
        &mut self,
        t: usize,
        d: Reg,
        (a, b): (&Val, &Val),
        neg: [bool; 2],
        (x, y, r): (u32, u32, u32),
        rz: bool,
        pc: u32,
    ) {
        if !self.num_anchor(t, &[a, b], pc) {
            self.num_clobber(t, d, &[a, b], pc);
            return;
        }
        if !neg[0] && !neg[1] {
            self.num_fadd(t, d, a, b, (x, y, r), rz, pc);
            return;
        }
        if rz {
            self.num_clobber(t, d, &[a, b], pc);
            return;
        }
        // -a + b etc.: a * 1 + b with the signs as modifiers (the product is exact).
        let one = Val::Imm(0x3f80_0000);
        self.num_ffma(
            t,
            d,
            (a, &one, Some(b)),
            [neg[0], false, neg[1]],
            [x, 0x3f80_0000, y],
            r,
            pc,
        );
    }

    /// Segment mode: FMNMX `d = min/max(a, b[, c])` of tracked values (no modifiers, FTZ or
    /// NaN propagation; an input-independent min/max predicate): exact `Max` nodes.
    pub(crate) fn num_fmnmx(&mut self, t: usize, d: Reg, ops: &[&Val], min: Option<bool>, pc: u32) {
        let Some(ns) = self.num.as_ref() else { return };
        if !ns.segments || min.is_none() {
            self.num_clobber(t, d, ops, pc);
            return;
        }
        let mut nodes = Vec::new();
        for v in ops {
            match self.num_node_mat(t, v) {
                Some(n) => nodes.push(n),
                None => {
                    // A constant operand (Triton's running maximum starts at -inf).
                    let ns = self.num.as_mut().unwrap();
                    if matches!(v, Val::Reg(r, _) if ns.taint.reg_t(t, *r)) {
                        self.num_clobber(t, d, ops, pc);
                        return;
                    }
                    let bits = self.rd(t, v).unwrap_or(0) as u32;
                    let ns = self.num.as_mut().unwrap();
                    nodes.push(ns.const_leaf(bits, pc));
                }
            }
        }
        let min = min.unwrap();
        let ns = self.num.as_mut().unwrap();
        // Only input-independent constants (a fully masked row's -inf): a constant, no node.
        let untainted = ops
            .iter()
            .all(|v| !matches!(v, Val::Reg(r, _) if ns.taint.reg_t(t, *r)));
        if untainted
            && nodes.iter().all(|&n| {
                matches!(ns.nodes[n as usize].kind, NodeKind::Leaf) && ns.const_leaves.contains(&n)
            })
        {
            ns.set(t, d, None);
            return;
        }
        let mut acc = nodes[0];
        for &n in &nodes[1..] {
            acc = ns.max_node(acc, n, min, pc);
        }
        ns.set(t, d, Some(acc));
    }

    /// Segment mode: FSETP `P = a cmp b` (combined with PT, no modifiers): remembered, so a
    /// following SEL choosing between a and b by P is a `Max`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn num_fsetp(&mut self, t: usize, d: Reg, cmp: FCmp, a: &Val, b: &Val, plain: bool) {
        let Some(ns) = self.num.as_mut() else { return };
        if !ns.segments {
            return;
        }
        let key = (t as u32, d.idx);
        ns.pred_src.remove(&key);
        if !plain {
            return;
        }
        let (Some(na), Some(nb)) = (ns.node_of(t, a), ns.node_of(t, b)) else {
            return;
        };
        let pos = ns
            .trace
            .get(&(t as u32))
            .map_or(0, |v| v.len().saturating_sub(1));
        ns.pred_src.insert(key, (pos, cmp, na, nb));
    }

    pub(crate) fn num_mov(&mut self, t: usize, d: Reg, a: &Val) {
        let Some(ns) = self.num.as_mut() else { return };
        let n = ns.node_of(t, a);
        ns.set(t, d, n);
        // A 64-bit move into 32-bit registers (SASS `LDC.64 R10`) also overwrites R11.
        let slots = match a {
            Val::Reg(_, k) => *k,
            Val::Const(_, b) => (*b / 4).max(1),
            _ => 1,
        };
        if d.file != File::V {
            for i in 1..slots as u16 {
                ns.set(
                    t,
                    Reg {
                        file: d.file,
                        idx: d.idx + i,
                    },
                    None,
                );
            }
        }
    }

    /// A non-fp operation wrote `d`. If a source carried rounding history, the numerics
    /// lose track of it: we say so instead of guessing.
    pub(crate) fn num_clobber(&mut self, t: usize, d: Reg, srcs: &[&Val], pc: u32) {
        let Some(ns) = self.num.as_mut() else { return };
        for s in srcs {
            if let Some(n) = ns.node_of(t, s) {
                if !exact::is_zero(&exact::sub(
                    &exact::from(f32::from_bits(ns.nodes[n as usize].hw) as f64),
                    &ns.nodes[n as usize].exact,
                )) {
                    ns.warnings.insert(format!(
                        "value with rounding error consumed by a non-fp instruction at {}",
                        self.k.describe(pc as usize)
                    ));
                }
            }
        }
        ns.set(t, d, None);
    }

    /// FFMA2: lane i of `d` is a*b + c. Tracked only in the form x*1 + 0 (cuBLAS's alpha = 1,
    /// beta = 0 epilogue), with the 1 and the +0 constants (no history, not input-derived):
    /// that is exactly x + 0, an `Add` with a +0 leaf (exact; it only turns -0 into +0).
    /// Anything else loses track, as `num_clobber`.
    pub(crate) fn num_ffma2(
        &mut self,
        t: usize,
        d: Reg,
        ops: [&(Val, F2Lanes); 3],
        neg: [bool; 3],
        vals: [[u32; 2]; 3],
        pc: u32,
    ) {
        if self.num.as_ref().is_some_and(|n| n.segments) {
            self.num_ffma2_seg(t, d, ops, neg, vals, pc);
            return;
        }
        let Some(ns) = self.num.as_mut() else { return };
        let lane_val = |(v, l): &(Val, F2Lanes), i: usize| -> Val {
            match (v, l) {
                (Val::Reg(r, _), F2Lanes::Pair | F2Lanes::Swap) => {
                    let k = if (*l == F2Lanes::Swap) == (i == 0) {
                        1
                    } else {
                        0
                    };
                    Val::Reg(
                        Reg {
                            file: r.file,
                            idx: r.idx + k,
                        },
                        1,
                    )
                }
                (Val::Reg(r, _), F2Lanes::Bcast) => Val::Reg(*r, 1),
                _ => v.clone(),
            }
        };
        let mut out = [None, None];
        let mut lost = false;
        for (i, o) in out.iter_mut().enumerate() {
            let v = ops.map(|op| lane_val(op, i));
            let konst = |k: usize| {
                ns.node_of(t, &v[k]).is_none()
                    && !matches!(&v[k], Val::Reg(r, _) if ns.taint.reg_t(t, *r))
            };
            let one = 1.0f32.to_bits();
            let x = if vals[0][i] == one && konst(0) {
                Some(1)
            } else if vals[1][i] == one && konst(1) {
                Some(0)
            } else {
                None
            };
            match x {
                Some(k) if vals[2][i] == 0 && konst(2) && !neg[k] => {
                    let nx = ns
                        .node_of(t, &v[k])
                        .unwrap_or_else(|| ns.leaf(vals[k][i], pc));
                    *o = Some((nx, vals[k][i]));
                }
                _ => lost |= v.iter().any(|s| ns.node_of(t, s).is_some()),
            }
        }
        for (i, o) in out.into_iter().enumerate() {
            let di = Reg {
                file: d.file,
                idx: d.idx + i as u16,
            };
            let Some((nx, xv)) = o else {
                ns.set(t, di, None);
                continue;
            };
            let z = ns.leaf(0, pc);
            let r = f32::from_bits(xv) + 0.0;
            ns.nodes.push(Node {
                kind: NodeKind::Add(nx, z, false),
                hw: r.to_bits(),
                exact: ns.nodes[nx as usize].exact.clone(),
                err: exact::sub(
                    &exact::from(r as f64),
                    &exact::from(f32::from_bits(xv) as f64),
                ),
                pc,
            });
            let id = (ns.nodes.len() - 1) as u32;
            ns.set(t, di, Some(id));
        }
        if lost {
            ns.warnings.insert(format!(
                "FFMA2 other than x*1 + 0 drops rounding history at {}",
                self.k.describe(pc as usize)
            ));
        }
    }

    /// Segment mode: FFMA2 / FMUL2 / FADD2 lane by lane, as two FFMA (FADD2 = a*1 + c: two
    /// FADD) on the lanes' registers.
    fn num_ffma2_seg(
        &mut self,
        t: usize,
        d: Reg,
        ops: [&(Val, F2Lanes); 3],
        neg: [bool; 3],
        vals: [[u32; 2]; 3],
        pc: u32,
    ) {
        let lane_val = |(v, l): &(Val, F2Lanes), i: usize| -> Val {
            match (v, l) {
                (Val::Reg(r, _), F2Lanes::Pair | F2Lanes::Swap) => {
                    let k = if (*l == F2Lanes::Swap) == (i == 0) {
                        1
                    } else {
                        0
                    };
                    Val::Reg(
                        Reg {
                            file: r.file,
                            idx: r.idx + k,
                        },
                        1,
                    )
                }
                (Val::Reg(r, _), F2Lanes::Bcast) => Val::Reg(*r, 1),
                _ => v.clone(),
            }
        };
        let lanes: Vec<[Val; 3]> = (0..2).map(|i| ops.map(|o| lane_val(o, i))).collect();
        // Lane 1 must not read a register lane 0 writes (d + 0).
        let clash = lanes[1]
            .iter()
            .any(|v| matches!(v, Val::Reg(r, _) if *r == d));
        // Materialize every source first (lane 0's write must not change lane 1's sources).
        let mut r = [0u32; 2];
        for (i, ri) in r.iter_mut().enumerate() {
            *ri = fp::f32_fma(vals[0][i], vals[1][i], Some(vals[2][i]), Rnd::Rn, false);
        }
        let di = |i: usize| Reg {
            file: d.file,
            idx: d.idx + i as u16,
        };
        // A clash (`FFMA2 R10, R11.F32, R6.F32, R10.F32`): both lanes read their sources
        // before either writes, so make both nodes first, then write them.
        if clash && !lanes.iter().any(|v| matches!(v[1], Val::Imm(0x3f80_0000))) {
            let mut ids = [None; 2];
            for (i, v) in lanes.iter().enumerate() {
                ids[i] = self.num_ffma_node(
                    t,
                    (&v[0], &v[1], Some(&v[2])),
                    neg,
                    [vals[0][i], vals[1][i], vals[2][i]],
                    r[i],
                    pc,
                );
            }
            if let [Some(a), Some(b)] = ids {
                let ns = self.num.as_mut().unwrap();
                ns.set(t, di(0), Some(a));
                ns.set(t, di(1), Some(b));
                return;
            }
        }
        for (i, v) in lanes.iter().enumerate() {
            let di = di(i);
            if clash {
                self.num_clobber(t, di, &[&v[0], &v[1], &v[2]], pc);
                continue;
            }
            if matches!(v[1], Val::Imm(0x3f80_0000)) {
                self.num_fadd_seg(
                    t,
                    di,
                    (&v[0], &v[2]),
                    [neg[0], neg[2]],
                    (vals[0][i], vals[2][i], r[i]),
                    false,
                    pc,
                );
            } else {
                self.num_ffma(
                    t,
                    di,
                    (&v[0], &v[1], Some(&v[2])),
                    neg,
                    [vals[0][i], vals[1][i], vals[2][i]],
                    r[i],
                    pc,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn num_fadd(
        &mut self,
        t: usize,
        d: Reg,
        a: &Val,
        b: &Val,
        (x, y, r): (u32, u32, u32),
        rz: bool,
        pc: u32,
    ) {
        if self.num.is_none() {
            return;
        }
        let ma = self.num_node_mat(t, a);
        let mb = self.num_node_mat(t, b);
        let ns = self.num.as_mut().unwrap();
        let na = ma.unwrap_or_else(|| ns.leaf_for(t, a, x, pc));
        let nb = mb.unwrap_or_else(|| ns.leaf_for(t, b, y, pc));
        let ex = exact::add(&ns.nodes[na as usize].exact, &ns.nodes[nb as usize].exact);
        let a_v = exact::add(
            &exact::from(f32::from_bits(x) as f64),
            &exact::from(f32::from_bits(y) as f64),
        );
        let err = exact::sub(&exact::from(f32::from_bits(r) as f64), &a_v);
        ns.nodes.push(Node {
            kind: NodeKind::Add(na, nb, rz),
            hw: r,
            exact: ex,
            err,
            pc,
        });
        let id = (ns.nodes.len() - 1) as u32;
        ns.set(t, d, Some(id));
    }

    /// A MUFU / PTX approx op wrote `d` = f(operand), the operand being register value `raw`
    /// with modifiers `neg` / `abs` applied. Tracked for f32 operands that carry rounding
    /// history or come from input data; anything else (integer-division reciprocals, 16-bit
    /// forms) is handled like `num_clobber`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn num_mufu(
        &mut self,
        t: usize,
        d: Reg,
        a: &Val,
        (raw, r): (u32, u32),
        (func, fmt, ptx): (MufuFunc, MufuFmt, bool),
        (neg, abs): (bool, bool),
        pc: u32,
    ) {
        let Some(ns) = self.num.as_mut() else { return };
        let na = ns.node_of(t, a);
        let tainted = matches!(a, Val::Reg(rg, _) if ns.taint.reg_t(t, *rg));
        // The 64H units work on f64 high words, which the f32 numerics do not track.
        let f64_unit = matches!(func, MufuFunc::Rcp64h | MufuFunc::Rsq64h);
        // Segment mode: part of a slice (the exp / reciprocal idioms around the unit).
        if ns.segments || fmt != MufuFmt::F32 || f64_unit || (na.is_none() && !tainted) {
            self.num_clobber(t, d, &[a], pc);
            return;
        }
        let ns = self.num.as_mut().unwrap();
        let na = na.unwrap_or_else(|| ns.leaf(raw, pc));
        let mut x = raw;
        if abs {
            x &= 0x7fff_ffff;
        }
        if neg {
            x ^= 0x8000_0000;
        }
        let mut xe = exact::approx(&ns.nodes[na as usize].exact);
        if abs {
            xe = xe.abs();
        }
        if neg {
            xe = -xe;
        }
        let ex = mufu_true(func, ptx, xe);
        let loc = f32::from_bits(r) as f64 - mufu_true(func, ptx, f32::from_bits(x) as f64);
        let fin = |v: f64| {
            if v.is_finite() {
                exact::from(v)
            } else {
                vec![]
            }
        };
        ns.nodes.push(Node {
            kind: NodeKind::Mufu {
                arg: na,
                func,
                ptx,
                neg,
                abs,
            },
            hw: r,
            exact: fin(ex),
            err: fin(loc),
            pc,
        });
        let id = (ns.nodes.len() - 1) as u32;
        ns.set(t, d, Some(id));
    }

    /// A load of `bytes` (`n` vector elements) into `d`, `data` being the bytes read. Each
    /// 4-byte word that lands in one 32-bit register takes the node of the memory word, or
    /// becomes an `Input` leaf if it is an unwritten word of an f32 input still holding its
    /// initial bits; every other register written loses its node.
    // ASSUME(N-LEAF): see ASSUMPTIONS.md
    pub(crate) fn num_ld(
        &mut self,
        t: usize,
        d: Reg,
        loc: Loc,
        (bytes, n, sext): (u8, u8, bool),
        data: &[u8],
    ) {
        if self.num.is_none() {
            return;
        }
        self.half_on_ld(t, d, loc, (bytes, n, sext), data);
        let launch = self.launch.clone();
        let ns = self.num.as_mut().unwrap();
        let reg = |i: u8| Reg {
            file: d.file,
            idx: d.idx + i as u16,
        };
        // (register, byte offset of its word in the access, or None: not one f32 word).
        let regs: Vec<(Reg, Option<u64>)> = if n > 1 {
            let es = bytes / n;
            (0..n)
                .map(|i| (reg(i), (es == 4).then_some(4 * i as u64)))
                .collect()
        } else if d.file == File::V {
            // One PTX register per access: 32-bit, or a 64-bit / 128-bit (two-slot) value.
            if bytes == 4 {
                vec![(d, Some(0))]
            } else {
                (0..(bytes / 8).max(1)).map(|i| (reg(i), None)).collect()
            }
        } else if bytes >= 4 {
            (0..bytes / 4)
                .map(|i| (reg(i), Some(4 * i as u64)))
                .collect()
        } else {
            vec![(d, None)]
        };
        for (r, w) in regs {
            // This register's own bytes (a whole-word register: its word; else the access).
            let (sp, a, off) = mem_key(loc);
            let (lo, n) = w.map_or((0, bytes as u64), |w| (w, 4));
            let rt = (lo..lo + n).any(|b| ns.taint.mem.contains(&(sp, a, off + b)));
            ns.taint.ld_read.push((t, r, rt));
            let node = w.and_then(|w| {
                let (sp, a, off) = mem_key(loc);
                let k = (sp, a, off + w);
                if let Some(&nd) = ns.mem.get(&k) {
                    return Some(nd);
                }
                let alloc = a as usize;
                let off = off + w;
                let bits = u32::from_le_bytes(data[w as usize..w as usize + 4].try_into().ok()?);
                let init = launch
                    .allocs
                    .get(alloc)?
                    .init
                    .get(off as usize..off as usize + 4)?;
                if sp != 1
                    || off % 4 != 0
                    || !ns.f32_inputs.contains(&alloc)
                    || ns.written.contains(&(a, off))
                    || init != bits.to_le_bytes()
                {
                    return None;
                }
                ns.nodes.push(Node {
                    kind: NodeKind::Input((alloc, off / 4)),
                    hw: bits,
                    exact: exact::from(f32::from_bits(bits) as f64),
                    err: vec![],
                    pc: 0,
                });
                let id = (ns.nodes.len() - 1) as u32;
                ns.mem.insert(k, id);
                Some(id)
            });
            ns.set(t, r, node);
        }
    }

    /// `d` of thread `t` now holds a value moved from elsewhere with node `n` (a shuffle).
    pub(crate) fn num_put(&mut self, t: usize, d: Reg, n: Option<u32>) {
        if let Some(ns) = self.num.as_mut() {
            ns.set(t, d, n);
        }
    }

    /// SEL / FSEL wrote `d` = the chosen operand `v` (no modifier). With a predicate that
    /// does not depend on input data, the choice is the same for every input, so `d` keeps
    /// `v`'s node; otherwise it is a data-dependent selection the numerics do not model.
    pub(crate) fn num_sel(&mut self, t: usize, d: Reg, v: &Val, p: Pred, pc: u32) {
        self.num_sel2(t, d, v, None, p, pc)
    }

    /// SEL with both operands known (`other`: the one not chosen): in segment mode a
    /// data-dependent choice between the two operands of the FSETP that wrote the predicate is
    /// their max or min.
    pub(crate) fn num_sel2(
        &mut self,
        t: usize,
        d: Reg,
        v: &Val,
        other: Option<(&Val, bool)>,
        p: Pred,
        pc: u32,
    ) {
        let Some(ns) = self.num.as_mut() else { return };
        let data_dep = p.reg.is_some_and(|r| ns.taint.reg_t(t, r));
        if !data_dep {
            let n = ns.node_of(t, v);
            ns.set(t, d, n);
            return;
        }
        // other = (the operand not chosen, whether `v` is the SEL's first operand).
        if let (true, Some(r), Some((o, v_first))) = (ns.segments, p.reg, other) {
            if let (Some(&(pos, cmp, x, y)), Some(nv), Some(no)) = (
                ns.pred_src.get(&(t as u32, r.idx)),
                ns.node_of(t, v),
                ns.node_of(t, o),
            ) {
                // The predicate still holds that FSETP's result (nothing wrote it since).
                let tr = &ns.trace[&(t as u32)];
                let code = &self.k.code;
                let fresh = tr[pos + 1..tr.len() - 1].iter().all(|e| {
                    !code[(e & !crate::seg::GUARD_FALSE) as usize]
                        .op
                        .dst_regs()
                        .contains(&r)
                });
                // SEL d, a, b, P: d = P ? a : b, P = (x cmp y) ^ neg.
                let (a, b) = if v_first { (nv, no) } else { (no, nv) };
                use FCmp::*;
                let gt = matches!(cmp, Gt | Ge | Gtu | Geu);
                let lt = matches!(cmp, Lt | Le | Ltu | Leu);
                // "P true when x is the larger" after the negation.
                let big = if p.neg { lt } else { gt };
                let small = if p.neg { gt } else { lt };
                let kind = if (a, b) == (x, y) {
                    if big {
                        Some(false)
                    } else if small {
                        Some(true)
                    } else {
                        None
                    }
                } else if (a, b) == (y, x) {
                    if big {
                        Some(true)
                    } else if small {
                        Some(false)
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let (true, Some(min)) = (fresh, kind) {
                    let n = ns.max_node(x, y, min, pc);
                    ns.set(t, d, Some(n));
                    return;
                }
            }
        }
        self.num_clobber(t, d, &[v], pc);
    }

    /// An atomic on memory at `loc`. fp32 adds (`fadd`: old, added and new bits) extend the
    /// location's atomic chain; any other atomic leaves an untracked value. The returned old
    /// value depends on the schedule: it has no node.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn num_atom(
        &mut self,
        t: usize,
        v: &Val,
        loc: Loc,
        fadd: Option<(u32, u32, u32)>,
        d: Option<Reg>,
        pc: u32,
    ) {
        if fadd.is_some() {
            self.num_node_mat(t, v);
        }
        let Some(ns) = self.num.as_mut() else { return };
        let k = mem_key(loc);
        ns.note_write(k);
        if fadd.is_none() || d.is_some() {
            ns.other_atomics += 1;
        }
        match fadd {
            Some((o, x, r)) => {
                let no = ns.mem.get(&k).copied().unwrap_or_else(|| ns.leaf(o, pc));
                let nx = ns.node_of(t, v).unwrap_or_else(|| ns.leaf_for(t, v, x, pc));
                let ex = exact::add(&ns.nodes[no as usize].exact, &ns.nodes[nx as usize].exact);
                let a_v = exact::add(
                    &exact::from(f32::from_bits(o) as f64),
                    &exact::from(f32::from_bits(x) as f64),
                );
                let err = exact::sub(&exact::from(f32::from_bits(r) as f64), &a_v);
                ns.nodes.push(Node {
                    kind: NodeKind::AtomAdd(no, nx),
                    hw: r,
                    exact: ex,
                    err,
                    pc,
                });
                ns.atomic_adds += 1;
                let id = (ns.nodes.len() - 1) as u32;
                ns.mem.insert(k, id);
            }
            None => {
                ns.mem.remove(&k);
            }
        }
        if let Some(d) = d {
            ns.set(t, d, None);
        }
    }

    pub(crate) fn num_st(&mut self, t: usize, v: &Val, loc: Loc, bytes: u8) {
        self.half_on_st(t, v, loc, bytes);
        // Registers holding 16-bit values with provenance (packed bf16) are moved as bytes:
        // nothing to materialize.
        let halves = match v {
            Val::Reg(r, _) if r.file == File::R => (0..(bytes as u16).div_ceil(4)).all(|i| {
                self.half_val(
                    t,
                    &Val::Reg(
                        Reg {
                            file: r.file,
                            idx: r.idx + i,
                        },
                        1,
                    ),
                )
                .is_some_and(|b| b.iter().all(|x| x.is_some()))
            }),
            _ => false,
        };
        if halves {
        } else if bytes == 4 {
            self.num_node_mat(t, v);
        } else if bytes.is_multiple_of(4) {
            // A vector store of 32-bit registers (STS.64 of a quotient and a sum): each word's
            // input-derived value is materialized as for a 4-byte store.
            if let Val::Reg(r, n) = v {
                if r.file == File::R || *n >= 2 {
                    for i in 0..(bytes / 4) as u16 {
                        let w = Val::Reg(
                            Reg {
                                file: r.file,
                                idx: r.idx + i,
                            },
                            1,
                        );
                        self.num_node_mat(t, &w);
                    }
                }
            }
        }
        let Some(ns) = self.num.as_mut() else { return };
        // ASSUME(N-EQUIV): a store of 32-bit registers: word i carries the taint of its own register (the
        // access was tainted whole if any value register of any thread was).
        if let (Val::Reg(r, _), false, 0) = (v, ns.taint.position, bytes % 4) {
            if r.file == File::R {
                let (sp, a, off) = mem_key(loc);
                if sp <= 1 {
                    for i in 0..(bytes / 4) as u16 {
                        let w = Reg {
                            file: r.file,
                            idx: r.idx + i,
                        };
                        let tw = ns.taint.reg_t(t, w);
                        for b in 0..4u64 {
                            let k = (sp, a, off + 4 * i as u64 + b);
                            if tw {
                                ns.taint.mem.insert(k);
                            } else {
                                ns.taint.mem.remove(&k);
                            }
                        }
                    }
                }
            }
        }
        // Vector stores (STS.128, st.v2/v4.b32): word i comes from register r + i (consecutive
        // PTX registers after lowering, as in exec).
        let split = bytes > 8 && bytes.is_multiple_of(4)
            || bytes > 4
                && bytes.is_multiple_of(4)
                && matches!(v, Val::Reg(r, n) if r.file == File::R || *n >= 2);
        // Every word the store covers loses its old node (a 64-bit PTX register: both).
        let words = (bytes as u16).div_ceil(4);
        for i in 0..words {
            let n = match v {
                Val::Reg(r, _) if split => ns.node_of(
                    t,
                    &Val::Reg(
                        Reg {
                            file: r.file,
                            idx: r.idx + i,
                        },
                        1,
                    ),
                ),
                _ if bytes == 4 => ns.node_of(t, v),
                _ => None,
            };
            let (sp, a, off) = mem_key(loc);
            let k = (sp, a, off + 4 * i as u64);
            ns.note_write(k);
            match n {
                Some(n) => {
                    ns.mem.insert(k, n);
                }
                None => {
                    ns.mem.remove(&k);
                }
            }
        }
    }

    /// STL of `v` to local word `addr` of thread `t` (private memory): the word keeps its node.
    pub(crate) fn num_local_st(&mut self, t: usize, v: &Val, addr: u32) {
        self.half_local_st(t, v, addr);
        let n = self.num_node_mat(t, v);
        let Some(ns) = self.num.as_mut() else { return };
        // Taint follows the value through local memory (spills).
        let tainted = matches!(v, Val::Reg(r, _) if ns.taint.reg_t(t, *r));
        for b in 0..4u64 {
            let k = (4u8, t as u64, addr as u64 + b);
            if tainted {
                ns.taint.mem.insert(k);
            } else {
                ns.taint.mem.remove(&k);
            }
        }
        match n {
            Some(n) => {
                ns.mem.insert((4, t as u64, addr as u64), n);
            }
            None => {
                ns.mem.remove(&(4, t as u64, addr as u64));
            }
        }
    }

    /// LDL of local word `addr` of thread `t` into `d`.
    pub(crate) fn num_local_ld(&mut self, t: usize, d: Reg, addr: u32) {
        self.half_local_ld(t, d, addr);
        let Some(ns) = self.num.as_mut() else { return };
        if (0..4u64).any(|b| ns.taint.mem.contains(&(4u8, t as u64, addr as u64 + b))) {
            ns.taint.cur_read = true;
        }
        let n = ns.mem.get(&(4, t as u64, addr as u64)).copied();
        ns.set(t, d, n);
    }

    /// tcgen05.st of register `v` into a TMEM word: the word carries v's provenance.
    pub(crate) fn num_tmem_st(&mut self, t: usize, v: &Val, cta: u16, lane: usize, col: usize) {
        if self.num.as_ref().is_some_and(|n| n.half.is_on()) {
            let b = self.half_val(t, v).unwrap_or([None; 4]);
            self.half_mem_set(crate::halfin::tkey(cta, lane, col, 0), &b);
        }
        let Some(ns) = self.num.as_mut() else { return };
        let key = (3u8, cta as u64, crate::machine::tmem_idx(lane, col) as u64);
        match ns.node_of(t, v) {
            Some(n) => {
                ns.mem.insert(key, n);
            }
            None => {
                ns.mem.remove(&key);
            }
        }
    }

    pub(crate) fn num_tmem_ld(&mut self, t: usize, d: Reg, cta: u16, lane: usize, col: usize) {
        if self.num.as_ref().is_some_and(|n| n.half.is_on()) {
            let b = self.half_tmem(cta, lane, col);
            self.half_set_reg(t, d, Some(b));
        }
        let Some(ns) = self.num.as_mut() else { return };
        let n = ns
            .mem
            .get(&(3, cta as u64, crate::machine::tmem_idx(lane, col) as u64))
            .copied();
        ns.set(t, d, n);
    }

    /// A tcgen05.mma step wrote TMEM word (lane, col) of `cta`: `r` = acc (if `enable`) plus
    /// the exact products `prods`, with operand provenance `ops` (zero products left out).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn num_mma(
        &mut self,
        cta: u16,
        lane: usize,
        col: usize,
        enable: bool,
        acc: f32,
        (prods, ops): (&[f64], &[crate::halfin::HalfOp]),
        r: f32,
        pc: u32,
        eps_of: &dyn Fn(usize, bool) -> f64,
        scales: Option<&[crate::halfin::HalfOp]>,
        unit: usize,
    ) {
        if self.num.as_ref().is_some_and(|n| n.half.is_on()) {
            self.half_mem_set(crate::halfin::tkey(cta, lane, col, 0), &[None; 4]);
        }
        let Some(ns) = self.num.as_mut() else { return };
        let key = (3u8, cta as u64, crate::machine::tmem_idx(lane, col) as u64);
        let acc_node = if enable {
            Some(
                ns.mem
                    .get(&key)
                    .copied()
                    .unwrap_or_else(|| ns.leaf(acc.to_bits(), pc)),
            )
        } else {
            None
        };
        let id = ns.push_mma(acc_node, acc, prods, ops, r, pc, eps_of, scales, unit);
        ns.mem.insert(key, id);
    }

    /// The node of an accumulator operand `v` (bits `bits`): its node (segment mode:
    /// materialized), else a leaf (`Opaque` when input-derived in segment mode).
    pub(crate) fn num_acc_node(&mut self, t: usize, v: &Val, bits: u32, pc: u32) -> Option<u32> {
        self.num.as_ref()?;
        if let Some(n) = self.num_node_mat(t, v) {
            return Some(n);
        }
        let ns = self.num.as_mut().unwrap();
        Some(ns.leaf_for(t, v, bits, pc))
    }

    /// HMMA.16816.F32.BF16 (`tf32`: HMMA.1688.F32.TF32) of the warp: lane `t`'s result
    /// register `dr` = `r` from accumulator node `acc_node` (value `acc`) and the products of
    /// its row and column.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn num_hmma(
        &mut self,
        t: usize,
        dr: Reg,
        (acc_node, acc): (Option<u32>, f32),
        (prods, ops): (&[f64], &[crate::halfin::HalfOp]),
        r: f32,
        pc: u32,
        tf32: bool,
    ) {
        let Some(ns) = self.num.as_mut() else { return };
        // No accumulator register (RZ): the input-independent constant.
        let accn = acc_node.unwrap_or_else(|| ns.const_leaf(acc.to_bits(), pc));
        let model = if tf32 {
            crate::fp::TMMA_MODEL
        } else {
            crate::fp::HMMA_MODEL
        };
        let id = ns.push_mma(
            Some(accn),
            acc,
            prods,
            ops,
            r,
            pc,
            &|n, a| model.eps_terms(n + a as usize),
            None,
            0,
        );
        ns.set(t, dr, Some(id));
    }

    /// `FSETP d = (a cmp b) bop c` whose outcome is the same for every input with finite
    /// elements and no overflow (N-FINITE): one operand a node `fin` proves finite, the other
    /// an input-independent infinity or NaN (the finiteness test `|x| != +INF`, a comparison
    /// with a running maximum's -inf start), `c` and the instruction's guard input-independent.
    /// Then `d` is not input data (its taint is cleared after the instruction).
    // ASSUME(N-FINITE): see ASSUMPTIONS.md
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn num_fin_setp(
        &mut self,
        t: usize,
        d: Reg,
        cmp: FCmp,
        (a, b): (&Val, &Val),
        (neg, abs): ([bool; 2], [bool; 2]),
        (c, ftz): (Pred, bool),
        pc: u32,
    ) {
        let Some(ns) = self.num.as_ref() else { return };
        if ns.taint.position {
            return;
        }
        let guard = self.k.code[pc as usize].guard;
        let tainted = |r: &Reg| ns.taint.reg_t(t, *r);
        if guard.reg.as_ref().is_some_and(tainted) || c.reg.as_ref().is_some_and(tainted) {
            return;
        }
        // Each operand: Some(node) if tainted (must be finite), else its constant bits.
        let mut fin_ops = Vec::new();
        let mut konst = Vec::new();
        for (i, v) in [a, b].into_iter().enumerate() {
            match v {
                Val::Reg(r, _) if tainted(r) => match ns.node_of(t, v) {
                    Some(n) => fin_ops.push(n),
                    None => return,
                },
                _ => match self.rd(t, v) {
                    Ok(x) => konst.push((i, fmod_bits(x as u32, neg[i], abs[i]))),
                    Err(_) => return,
                },
            }
        }
        let ([n], [(ki, kb)]) = (fin_ops.as_slice(), konst.as_slice()) else {
            return;
        };
        let (n, ki, kb) = (*n, *ki, *kb);
        if f32::from_bits(kb).is_finite() {
            return;
        }
        let ns = self.num.as_mut().unwrap();
        if !ns.fin(n) || !f32::from_bits(ns.nodes[n as usize].hw).is_finite() {
            return;
        }
        // Against an infinity or NaN every finite x compares alike.
        let outcome = |x: u32| {
            let (p, q) = if ki == 0 { (kb, x) } else { (x, kb) };
            fp::f32_cmp(p, q, cmp, ftz)
        };
        let r = outcome(0);
        if [
            0x8000_0000u32,
            0x3f80_0000,
            0xbf80_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
            1,
        ]
        .iter()
        .any(|&x| outcome(x) != r)
        {
            return;
        }
        let tn = &mut ns.taint;
        tn.fin_clear.push((t, d));
        tn.fin_pcs.insert(pc);
    }

    /// An atomic of a float atomic max diamond (N-FMAX) updated shared word `loc` from `old`
    /// to `new` with value `x` (register `v`): the word's node becomes max(old, v). Both must
    /// be non-NaN floats for every admissible input (`fin`, or a non-NaN constant), and `new`
    /// the float maximum; otherwise the diamond is a violation.
    // ASSUME(N-FMAX): see ASSUMPTIONS.md
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn num_fmax_atom(
        &mut self,
        t: usize,
        v: &Val,
        loc: Loc,
        (old, x, new): (u32, u32, u32),
        pc: u32,
    ) -> bool {
        let Some(ns) = self.num.as_ref() else {
            return false;
        };
        let in_diamond = (pc as usize)
            .checked_sub(1)
            .into_iter()
            .chain((pc as usize).checked_sub(3))
            .any(|b| {
                fmax_diamond(&self.k.code, b)
                    .is_some_and(|(x1, x2)| x1 == pc as usize || x2 == pc as usize)
            });
        if !in_diamond || ns.taint.position {
            return false;
        }
        let describe = self.k.describe(pc as usize);
        let k = mem_key(loc);
        let ns = self.num.as_mut().unwrap();
        ns.note_write(k);
        ns.other_atomics += 1;
        // Whether this thread's read found input-derived bytes.
        let loc_tainted = ns.taint.last_read;
        let nan = |b: u32| f32::from_bits(b).is_nan();
        let on = match ns.mem.get(&k).copied() {
            Some(n) => Some(n),
            None if !loc_tainted && !nan(old) => Some(ns.const_leaf(old, pc)),
            None => None,
        };
        let mat = self.num_node_mat(t, v);
        let ns = self.num.as_mut().unwrap();
        let vn = match mat {
            Some(n) => Some(n),
            None if !matches!(v, Val::Reg(r, _) if ns.taint.reg_t(t, *r)) && !nan(x) => {
                Some(ns.const_leaf(x, pc))
            }
            None => None,
        };
        let ok_node = |ns: &mut NumState, n: u32| {
            let konst = matches!(ns.nodes[n as usize].kind, NodeKind::Leaf)
                && ns.const_leaves.contains(&n)
                && !nan(ns.nodes[n as usize].hw);
            konst || ns.fin(n)
        };
        let fmax = fp::f32_minmax(old, x, false, false, false);
        let why = match (on, vn) {
            (None, _) => Some("the shared word has no node"),
            (_, None) => Some("the value has no node"),
            (Some(a), Some(b)) => {
                if !ok_node(ns, a) || !ok_node(ns, b) {
                    Some("an operand is not a non-NaN float for every admissible input")
                } else if fmax != new {
                    Some("the result is not the float maximum")
                } else {
                    let m = ns.max_node(a, b, false, pc);
                    ns.mem.insert(k, m);
                    None
                }
            }
        };
        if let Some(why) = why {
            ns.mem.remove(&k);
            ns.taint
                .violations
                .insert(format!("float atomic max at {describe}: {why}"));
        }
        true
    }

    /// The node of a register operand `v` of thread `t` for the TF32 split hooks (segment
    /// mode: materialized), None for a non-register or an untracked one.
    fn tf32_src(&mut self, t: usize, v: &Val) -> Option<u32> {
        matches!(v, Val::Reg(..)).then_some(())?;
        self.num_node_mat(t, v)
    }

    /// Whether operand `v` of thread `t` is the same for every input (not a tainted register),
    /// and its 32-bit value.
    pub(crate) fn tf32_const(&self, t: usize, v: &Val) -> Option<u32> {
        let ns = self.num.as_ref()?;
        if matches!(v, Val::Reg(r, _) if ns.taint.reg_t(t, *r)) {
            return None;
        }
        self.rd(t, v).ok().map(|x| x as u32)
    }

    /// `LOP3 d = lut(a, b, c)` with one tracked register x and two input-independent operands,
    /// computing x & 0xffffe000 bit by bit for every x: d is hi(x), a `Tf32 Hi` node.
    // ASSUME(N-TF32): see ASSUMPTIONS.md
    pub(crate) fn num_tf32_hi(
        &mut self,
        t: usize,
        d: Reg,
        ops: [&Val; 3],
        lut: u8,
        pc: u32,
    ) -> bool {
        if self.num.is_none() {
            return false;
        }
        // One operand register that is not input-independent (or holds a node), the others
        // constants; the LUT is checked before the operand is materialized.
        let mut x = None;
        let mut k = [0u32; 3];
        for (i, v) in ops.iter().enumerate() {
            let has_node = self.num.as_ref().unwrap().node_of(t, v).is_some();
            match self.tf32_const(t, v) {
                Some(c) if !(has_node && x.is_none()) => k[i] = c,
                _ if x.is_none() && matches!(v, Val::Reg(..)) => x = Some(i),
                _ => return false,
            }
        }
        let Some(xi) = x else { return false };
        // Result bit j as a function of x's bit j, the others fixed: must be x_j for j >= 13
        // and 0 below.
        for j in 0..32 {
            let out = |xb: u32| {
                let mut bits = [(k[0] >> j) & 1, (k[1] >> j) & 1, (k[2] >> j) & 1];
                bits[xi] = xb;
                (lut >> ((bits[0] << 2) | (bits[1] << 1) | bits[2])) & 1
            };
            let want1 = if j >= 13 { 1 } else { 0 };
            if out(0) != 0 || out(1) != want1 {
                return false;
            }
        }
        let Some(n) = self.tf32_src(t, ops[xi]) else {
            return false;
        };
        let ns = self.num.as_mut().unwrap();
        let hw = fp::tf32_in(ns.nodes[n as usize].hw);
        let id = ns.tf32_node(n, Tf32Part::Hi, hw, pc);
        ns.set(t, d, Some(id));
        true
    }

    /// `FADD d, x, -h` (either order; round to nearest, no other modifiers) with h = hi(x) of
    /// the same node: d = lo(x), exact (x and hi(x) share sign and exponent, so the difference
    /// is the dropped bits). A `Tf32 Lo` node.
    // ASSUME(N-TF32): see ASSUMPTIONS.md
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn num_tf32_lo(
        &mut self,
        t: usize,
        d: Reg,
        (a, b): (&Val, &Val),
        neg: [bool; 2],
        plain: bool,
        r: u32,
        pc: u32,
    ) -> bool {
        if self.num.is_none() || !plain || neg[0] == neg[1] {
            return false;
        }
        let (xv, hv) = if neg[1] { (a, b) } else { (b, a) };
        let (Some(nx), Some(nh)) = (self.tf32_src(t, xv), self.tf32_src(t, hv)) else {
            return false;
        };
        let ns = self.num.as_mut().unwrap();
        let NodeKind::Tf32 {
            arg,
            part: Tf32Part::Hi,
        } = ns.nodes[nh as usize].kind
        else {
            return false;
        };
        let xb = ns.nodes[nx as usize].hw;
        if arg != nx || !f32::from_bits(xb).is_finite() {
            return false;
        }
        let lo = f32::from_bits(xb) as f64 - f32::from_bits(fp::tf32_in(xb)) as f64;
        if (lo as f32).to_bits() != r && !(lo == 0.0 && r == 0) {
            return false;
        }
        let id = ns.tf32_node(nx, Tf32Part::Lo, r, pc);
        ns.set(t, d, Some(id));
        true
    }

    /// An integer add `d = s + k` (IMAD s * 1 + k, LEA s + k, IADD3 s + k + 0) of a register s
    /// holding lo(x) and an input-independent k = 0x1000: the `Tf32 LoR` node of x (its hw is
    /// the value the TF32 MMA reads, the sum truncated). `terms`: the added operands, each with
    /// whether it is the integer 1 multiplier slot (ignored).
    // ASSUME(N-TF32): see ASSUMPTIONS.md
    pub(crate) fn num_tf32_lor(
        &mut self,
        t: usize,
        d: Reg,
        terms: &[&Val],
        r: u32,
        pc: u32,
    ) -> bool {
        if self.num.is_none() || terms.len() != 2 {
            return false;
        }
        for (s, k) in [(terms[0], terms[1]), (terms[1], terms[0])] {
            let Some(ns_) = self.tf32_src(t, s) else {
                continue;
            };
            if self.tf32_const(t, k) != Some(0x1000) {
                continue;
            }
            let ns = self.num.as_mut().unwrap();
            let NodeKind::Tf32 {
                arg,
                part: Tf32Part::Lo,
            } = ns.nodes[ns_ as usize].kind
            else {
                continue;
            };
            if ns.nodes[ns_ as usize].hw.wrapping_add(0x1000) != r {
                continue;
            }
            let id = ns.tf32_node(arg, Tf32Part::LoR, fp::tf32_in(r), pc);
            ns.set(t, d, Some(id));
            return true;
        }
        false
    }

    /// The factor a TF32 MMA operand register `v` (word `bits`) of thread `t` is: hi(x) of a
    /// tracked value x (the MMA truncates it), a low-part factor of a split, a constant, or
    /// unknown.
    // ASSUME(N-TF32): see ASSUMPTIONS.md
    pub(crate) fn tf32_factor(
        &mut self,
        t: usize,
        v: &Val,
        bits: u32,
        pc: u32,
    ) -> crate::halfin::HalfOp {
        use crate::halfin::HalfOp;
        if self.num.is_none() {
            return HalfOp::Unknown;
        }
        let tb = fp::tf32_in(bits);
        match self.num_node_mat(t, v) {
            Some(n) => {
                let ns = self.num.as_mut().unwrap();
                match ns.nodes[n as usize].kind {
                    NodeKind::Tf32 {
                        part: Tf32Part::Hi | Tf32Part::LoR,
                        ..
                    } if ns.nodes[n as usize].hw == tb => HalfOp::Node(n),
                    NodeKind::Tf32 {
                        arg,
                        part: Tf32Part::Lo,
                    } => HalfOp::Node(ns.tf32_node(arg, Tf32Part::LoT, tb, pc)),
                    NodeKind::Tf32 { .. } => HalfOp::Unknown,
                    _ => HalfOp::Node(ns.tf32_node(n, Tf32Part::Hi, tb, pc)),
                }
            }
            None => match self.tf32_const(t, v) {
                Some(_) if tb & 0x7fff_ffff == 0 => HalfOp::Zero,
                Some(_) => {
                    let ns = self.num.as_mut().unwrap();
                    HalfOp::Node(ns.const_leaf(tb, pc))
                }
                None => HalfOp::Unknown,
            },
        }
    }

    /// F2FP.BF16.F32.PACK_AB `d` = (bf16(a) << 16) | bf16(b), round to nearest even: the halves
    /// are `Round` nodes of the operands' nodes (16-bit provenance `N`); a constant operand
    /// gives a constant half (zero bytes for 0), an untracked one unknown bytes.
    pub(crate) fn num_f2fp_bf16(
        &mut self,
        t: usize,
        d: Reg,
        (a, b): (&Val, &Val),
        (x, y): (u32, u32),
        pc: u32,
    ) {
        self.num_f2fp16(t, d, (a, b), (x, y), pc, false)
    }

    /// `F2FP.BF16.F32.PACK_AB` (`f16`: `.F16`): a rounding node per half, its bytes `N{node}`.
    pub(crate) fn num_f2fp16(
        &mut self,
        t: usize,
        d: Reg,
        (a, b): (&Val, &Val),
        (x, y): (u32, u32),
        pc: u32,
        f16: bool,
    ) {
        if self.num.is_none() {
            return;
        }
        let mut halves = [[None; 2]; 2];
        for (i, (v, bits)) in [(b, y), (a, x)].into_iter().enumerate() {
            let rb = if f16 {
                crate::fp::f32_to_f16_rn(bits) as u32
            } else {
                crate::fp::f32_to_bf16_rn_gpu(bits) as u32
            };
            // The 16-bit value widened to f32.
            let wide = if f16 {
                crate::elementwise::f16_to_f32_bits(rb as u16)
            } else {
                rb << 16
            };
            let n = self.num_node_mat(t, v);
            let ns = self.num.as_mut().unwrap();
            let tainted = matches!(v, Val::Reg(r, _) if ns.taint.reg_tainted(t, *r));
            let n = match n {
                Some(n) => n,
                None if !tainted => {
                    // A constant: its bf16 rounding is a constant too.
                    if rb == 0 {
                        halves[i] = [Some(crate::halfin::Src::Zero); 2];
                    } else {
                        let l = ns.leaf(wide, pc);
                        if f16 {
                            ns.half.f16_nodes.insert(l);
                        }
                        halves[i] = [
                            Some(crate::halfin::Src::N { node: l, b: 0 }),
                            Some(crate::halfin::Src::N { node: l, b: 1 }),
                        ];
                    }
                    continue;
                }
                None => continue,
            };
            let hw = wide;
            let arg_hw = ns.nodes[n as usize].hw;
            ns.nodes.push(Node {
                kind: NodeKind::Round { arg: n, f16 },
                hw,
                exact: ns.nodes[n as usize].exact.clone(),
                err: exact::sub(
                    &exact::from(f32::from_bits(hw) as f64),
                    &exact::from(f32::from_bits(arg_hw) as f64),
                ),
                pc,
            });
            let id = (ns.nodes.len() - 1) as u32;
            if f16 {
                ns.half.f16_nodes.insert(id);
            }
            halves[i] = [
                Some(crate::halfin::Src::N { node: id, b: 0 }),
                Some(crate::halfin::Src::N { node: id, b: 1 }),
            ];
        }
        let ns = self.num.as_mut().unwrap();
        ns.set(t, d, None);
        let bytes = [halves[0][0], halves[0][1], halves[1][0], halves[1][1]];
        self.half_set_reg(t, d, Some(bytes));
    }

    /// A TMEM word written by an operation the numerics shadow does not follow (an MMA outside
    /// its model, an f16 accumulator): it loses its node.
    pub(crate) fn num_tmem_forget(&mut self, cta: u16, lane: usize, col: usize) {
        let Some(ns) = self.num.as_mut() else { return };
        ns.mem
            .remove(&(3u8, cta as u64, crate::machine::tmem_idx(lane, col) as u64));
    }

    /// A 4-byte copy between memory locations (TMA store): the value keeps its node.
    pub(crate) fn num_copy(&mut self, from: Loc, to: Loc) {
        let Some(ns) = self.num.as_mut() else { return };
        ns.note_write(mem_key(to));
        match ns.mem.get(&mem_key(from)).copied() {
            Some(n) => {
                ns.mem.insert(mem_key(to), n);
            }
            None => {
                ns.mem.remove(&mem_key(to));
            }
        }
    }

    /// A cp.async copy into shared memory at (`cta`, `off`) completed with `data`, read from
    /// global (allocation, offset) `src` (None: zero fill). Each 4-byte word takes the global
    /// word's node, or becomes an `Input` leaf if it is an unwritten word of an f32 input still
    /// holding its initial bits (as a load would, N-LEAF); other words lose their node.
    // ASSUME(N-LEAF): see ASSUMPTIONS.md
    pub(crate) fn num_cp_async(
        &mut self,
        src: Option<(usize, u64)>,
        cta: u16,
        off: u32,
        data: &[u8],
    ) {
        if self.num.is_none() {
            return;
        }
        self.half_copy(
            src.map(|(alloc, off)| Loc::Global { alloc, off }),
            Loc::Shared { cta, off },
            data.len(),
            data,
        );
        // Taint: the copy completes in a later instruction (the wait), so the destination takes
        // the source bytes' taint here (zero fill: none).
        {
            let ns = self.num.as_mut().unwrap();
            for b in 0..data.len() as u64 {
                let t = src.is_some_and(|(a, o)| ns.taint.mem.contains(&(1, a as u64, o + b)));
                let k = (0u8, cta as u64, off as u64 + b);
                if t {
                    ns.taint.mem.insert(k);
                } else {
                    ns.taint.mem.remove(&k);
                }
            }
        }
        let launch = self.launch.clone();
        let ns = self.num.as_mut().unwrap();
        for w in 0..data.len() / 4 {
            let to = mem_key(Loc::Shared {
                cta,
                off: off + 4 * w as u32,
            });
            ns.note_write(to);
            let node = src.and_then(|(alloc, goff)| {
                let goff = goff + 4 * w as u64;
                let k = mem_key(Loc::Global { alloc, off: goff });
                if let Some(&n) = ns.mem.get(&k) {
                    return Some(n);
                }
                let bits = u32::from_le_bytes(data[4 * w..4 * w + 4].try_into().ok()?);
                let init = launch
                    .allocs
                    .get(alloc)?
                    .init
                    .get(goff as usize..goff as usize + 4)?;
                if !goff.is_multiple_of(4)
                    || !ns.f32_inputs.contains(&alloc)
                    || ns.written.contains(&(alloc as u64, goff))
                    || init != bits.to_le_bytes()
                {
                    return None;
                }
                ns.nodes.push(Node {
                    kind: NodeKind::Input((alloc, goff / 4)),
                    hw: bits,
                    exact: exact::from(f32::from_bits(bits) as f64),
                    err: vec![],
                    pc: 0,
                });
                let id = (ns.nodes.len() - 1) as u32;
                ns.mem.insert(k, id);
                Some(id)
            });
            match node {
                Some(n) => {
                    ns.mem.insert(to, n);
                }
                None => {
                    ns.mem.remove(&to);
                }
            }
        }
    }

    /// A TMA load wrote the shared-memory element at `off` from global element `src`
    /// (None: out of bounds, zero-filled).
    pub(crate) fn num_tma_elem(&mut self, cta: u16, off: u32, src: Option<(usize, u64)>) {
        let Some(ns) = self.num.as_mut() else { return };
        ns.mem.remove(&mem_key(Loc::Shared { cta, off }));
        match src.filter(|(a, _)| !ns.not_inputs.contains(a)) {
            Some(o) => {
                ns.orig.insert((cta, off), o);
            }
            None => {
                ns.orig.remove(&(cta, off));
            }
        }
    }
}

/// Per-output numerics record (one f32 output element).
#[derive(Clone, Debug, Serialize)]
pub struct OutRec {
    pub index: usize,
    pub hw: f32,
    pub exact: f64,
    /// hw - exact, computed exactly then rounded once to f64.
    pub signed_err: f64,
    pub abs_err: f64,
    pub rel_err: f64,
    pub err_ulps: f64,
    /// fp_noise Cor 1a a-priori bound: sum_i |t_i| ((1+u)^d_i - 1).
    pub bound_cor1a: f64,
    /// Running (a-posteriori) bound: sum over rounding nodes of u * |hw_v|.
    pub bound_running: f64,
    /// fp_noise Thm 1, checked exactly: hw - exact == sum of local errors. False when the
    /// tree contains MUFU nodes (`n_mufu`), where the identity is only first-order.
    pub thm1_exact: bool,
    /// MUFU / approx nodes in the tree: each contributes its own error bound (N-MUFU) plus
    /// its operand's bound times |f'| (first order).
    pub n_mufu: usize,
    pub n_leaves: usize,
    pub n_roundings: usize,
    /// Sum of |leaf terms| (|products| for dot products): the natural error scale.
    pub abs_sum: f64,
    /// Local errors e_v along the tree, in evaluation order (for plots).
    pub local_errs: Vec<f64>,
    /// |partial sum| at each rounding node, in evaluation order.
    pub partials: Vec<f64>,
}

pub const U32: f64 = 1.0 / (1u64 << 24) as f64;

pub fn ulp_f32(x: f32) -> f64 {
    let a = x.abs();
    if a == 0.0 || !a.is_finite() {
        return f32::from_bits(1) as f64;
    }
    let next = f32::from_bits(a.to_bits() + 1);
    (next - a) as f64
}

impl NumState {
    /// Analyze the tree rooted at `node` (or an exact value if `node` is None).
    // ASSUME(N-BOUND): see ASSUMPTIONS.md
    pub fn out_rec(&self, index: usize, hw_bits: u32, node: Option<u32>) -> OutRec {
        let hw = f32::from_bits(hw_bits);
        let Some(root) = node else {
            return OutRec {
                index,
                hw,
                exact: hw as f64,
                signed_err: 0.0,
                abs_err: 0.0,
                rel_err: 0.0,
                err_ulps: 0.0,
                bound_cor1a: 0.0,
                bound_running: 0.0,
                thm1_exact: true,
                n_mufu: 0,
                n_leaves: 0,
                n_roundings: 0,
                abs_sum: hw.abs() as f64,
                local_errs: vec![],
                partials: vec![],
            };
        };
        // Walk the tree: leaves with their growth factor (the product of (1 + eps_v) over the
        // rounding nodes on the path to the root), rounding nodes in evaluation (node id) order.
        let mut stack = vec![(root, 1.0f64)];
        let mut rnodes = Vec::new();
        let mut leaf_terms: Vec<(f64, f64)> = Vec::new();
        let mut mufu: Vec<(u32, f64)> = Vec::new();
        let mut nonlin = 0usize;
        let hwabs = |n: u32| (f32::from_bits(self.nodes[n as usize].hw) as f64).abs();
        while let Some((n, g)) = stack.pop() {
            match &self.nodes[n as usize].kind {
                NodeKind::Leaf | NodeKind::Opaque | NodeKind::Input(_) => {
                    leaf_terms.push((hwabs(n), g));
                }
                NodeKind::Add(a, b, rz) => {
                    rnodes.push(n);
                    let e = if *rz { 2.0 * U32 } else { U32 };
                    stack.push((*a, g * (1.0 + e)));
                    stack.push((*b, g * (1.0 + e)));
                }
                NodeKind::AtomAdd(a, b) => {
                    rnodes.push(n);
                    stack.push((*a, g * (1.0 + U32)));
                    stack.push((*b, g * (1.0 + U32)));
                }
                NodeKind::Mma {
                    acc,
                    terms,
                    ops,
                    eps,
                    ..
                } => {
                    rnodes.push(n);
                    // Products of rounded (non-input) factors: their factors' errors are not
                    // analyzed here (equiv does), and Thm 1 is not an identity through them.
                    if ops.iter().flatten().any(|&(x, y)| {
                        !matches!(
                            self.nodes[x as usize].kind,
                            NodeKind::Input(_) | NodeKind::Leaf
                        ) || !matches!(
                            self.nodes[y as usize].kind,
                            NodeKind::Input(_) | NodeKind::Leaf
                        )
                    }) {
                        nonlin += 1;
                    }
                    for t in terms {
                        leaf_terms.push((t.abs(), g * (1.0 + eps)));
                    }
                    if let Some(a) = acc {
                        stack.push((*a, g * (1.0 + eps)));
                    }
                }
                NodeKind::Round { arg, f16 } => {
                    rnodes.push(n);
                    stack.push((*arg, g * (1.0 + crate::equiv::round_unit(*f16).0)));
                }
                NodeKind::Mufu { .. } => {
                    // Its value is a term of the tree above; its own error and its operand's
                    // propagated error are added below (`mufu_extra`).
                    rnodes.push(n);
                    mufu.push((n, g));
                    leaf_terms.push((hwabs(n), g));
                }
                NodeKind::Fma { .. }
                | NodeKind::Max { .. }
                | NodeKind::Seg(_)
                | NodeKind::Tf32 { .. } => {
                    // Segment mode (equiv) only (and TF32 parts): a term of the tree above, its own error not
                    // analyzed here (Thm 1 is not an identity through it).
                    nonlin += 1;
                    leaf_terms.push((hwabs(n), g));
                }
            }
        }
        rnodes.sort();
        let ex = &self.nodes[root as usize].exact;
        let exact_v = exact::approx(ex);
        let diff = exact::sub(&exact::from(hw as f64), ex);
        let mut sum_e: Expansion = vec![];
        for &n in &rnodes {
            sum_e = exact::add(&sum_e, &self.nodes[n as usize].err);
        }
        let thm1 = mufu.is_empty() && nonlin == 0 && exact::is_zero(&exact::sub(&diff, &sum_e));
        let abs_err = exact::approx(&diff).abs();
        // ASSUME(N-MUFU): a MUFU node adds its own a-priori error bound and, to first order,
        // |f'(x)| times its operand's bounds; both are carried through the rounding nodes
        // above it.
        let mut mufu_extra = (0.0, 0.0);
        for &(n, grow) in &mufu {
            let nd = &self.nodes[n as usize];
            let NodeKind::Mufu {
                arg,
                func,
                ptx,
                neg,
                abs,
            } = nd.kind
            else {
                unreachable!()
            };
            let child = &self.nodes[arg as usize];
            let mut x = f32::from_bits(child.hw) as f64;
            if abs {
                x = x.abs();
            }
            if neg {
                x = -x;
            }
            let sub = self.out_rec(index, child.hw, Some(arg));
            let fp = mufu_true_deriv(func, ptx, x).abs();
            let own = mufu_local_bound(func, ptx, x as f32, f32::from_bits(nd.hw));
            mufu_extra.0 += (own + fp * sub.bound_cor1a) * grow;
            mufu_extra.1 += (own + fp * sub.bound_running) * grow;
        }
        let cor1a: f64 = leaf_terms.iter().map(|&(m, g)| m * (g - 1.0)).sum::<f64>() + mufu_extra.0;
        let running: f64 = rnodes
            .iter()
            .map(|&n| {
                let nd = &self.nodes[n as usize];
                match &nd.kind {
                    NodeKind::Mma {
                        acc, terms, eps, ..
                    } => {
                        let a = acc
                            .map(|a| (f32::from_bits(self.nodes[a as usize].hw) as f64).abs())
                            .unwrap_or(0.0);
                        eps * (a + terms.iter().map(|t| t.abs()).sum::<f64>())
                    }
                    NodeKind::Round { arg, f16 } => {
                        let (u16_, a16) = crate::equiv::round_unit(*f16);
                        u16_ * hwabs(*arg) + a16
                    }
                    NodeKind::Mufu { .. } => 0.0, // in mufu_extra
                    _ => U32 * (f32::from_bits(nd.hw) as f64).abs(),
                }
            })
            .sum::<f64>()
            + mufu_extra.1;
        OutRec {
            index,
            hw,
            exact: exact_v,
            signed_err: exact::approx(&diff),
            abs_err,
            rel_err: if exact_v != 0.0 {
                abs_err / exact_v.abs()
            } else {
                0.0
            },
            err_ulps: abs_err / ulp_f32(hw),
            bound_cor1a: cor1a,
            bound_running: running,
            thm1_exact: thm1,
            n_mufu: mufu.len(),
            n_leaves: leaf_terms.len(),
            abs_sum: leaf_terms.iter().map(|t| t.0).sum(),
            n_roundings: rnodes.len(),
            local_errs: rnodes
                .iter()
                .map(|&n| exact::approx(&self.nodes[n as usize].err))
                .collect(),
            partials: rnodes
                .iter()
                .map(|&n| (f32::from_bits(self.nodes[n as usize].hw) as f64).abs())
                .collect(),
        }
    }
}

// ---------------------------------------------------------------- input-data taint

/// Register-only operations (no memory, synchronization or control): every source is a
/// value, and the result is tainted if any source is.
fn pure_op(op: &Op) -> bool {
    matches!(
        op,
        Op::Mov { .. }
            | Op::Int { .. }
            | Op::MulWide { .. }
            | Op::Imad { .. }
            | Op::Iadd3 { .. }
            | Op::CvtInt { .. }
            | Op::R2Ur { .. }
            | Op::VoteAny { .. }
            | Op::VoteAll { .. }
            | Op::Redux { .. }
            | Op::Flo { .. }
            | Op::Lea { .. }
            | Op::LeaHi { .. }
            | Op::Shf { .. }
            | Op::Lop3 { .. }
            | Op::Plop3 { .. }
            | Op::Setp { .. }
            | Op::Fadd { .. }
            | Op::Ffma2 { .. }
            | Op::Hfma2 { .. }
            | Op::CvtF32Bf16 { .. }
            | Op::Cvta { .. }
            | Op::Bfe { .. }
            | Op::I2F { .. }
            | Op::F2I { .. }
            | Op::Pack64 { .. }
            | Op::Unpack64 { .. }
            | Op::Mufu { .. }
            | Op::Sel { .. }
            | Op::Prmt { .. }
            | Op::Brev { .. }
            | Op::AddMinMax { .. }
            | Op::IntMnmx3 { .. }
            | Op::P2r { .. }
            | Op::LdConst { .. }
            | Op::Ffma { .. }
            | Op::Fmnmx { .. }
            | Op::Fsetp { .. }
            | Op::Hsetp2Bf16 { .. }
            | Op::Fset { .. }
            | Op::Fchk { .. }
            | Op::Fsel { .. }
            | Op::R2p { .. }
            | Op::F2fpBf16 { .. }
            | Op::F2fpF16 { .. }
            | Op::F2fF16 { .. }
            | Op::CvtF32F16 { .. }
            | Op::CvtPack { .. }
            | Op::CvtRsX4 { .. }
            | Op::CvtUnpack { .. }
            | Op::Nop
    )
}

/// The value operands of a memory / synchronization operation: the data it moves. Every
/// other register it reads (addresses, coordinates, descriptors, counts, lane indices) decides
/// where or whether something happens.
/// Position mode: registers of `op` that may depend on the CTA's position (global addresses,
/// TMA coordinates), and whether they are a generic address (must resolve to global).
fn pos_free_regs(op: &Op) -> (Vec<Reg>, bool) {
    let mut out = Vec::new();
    let mut v = |x: &Val| {
        if let Val::Reg(r, n) = x {
            for i in 0..(*n).max(1) as u16 {
                out.push(Reg {
                    file: r.file,
                    idx: r.idx + if r.file == File::V { 0 } else { i },
                });
                if r.file == File::V {
                    break;
                }
            }
        }
    };
    let mut generic = false;
    match op {
        Op::Ld { space, addr, .. } | Op::St { space, addr, .. } | Op::Atom { space, addr, .. }
            if matches!(space, Space::Global | Space::Generic) =>
        {
            generic = *space == Space::Generic;
            addr.base.iter().chain(addr.base2.iter()).for_each(&mut v);
        }
        Op::TmaLoad { coords, .. } | Op::TmaStore { coords, .. } => coords.iter().for_each(&mut v),
        _ => {}
    }
    (out, generic)
}

fn value_regs(op: &Op) -> Vec<Reg> {
    let mut out = Vec::new();
    let mut v = |x: &Val| {
        if let Val::Reg(r, n) = x {
            for i in 0..(*n).max(1) as u16 {
                out.push(Reg {
                    file: r.file,
                    idx: r.idx + if r.file == File::V { 0 } else { i },
                });
                if r.file == File::V {
                    break;
                }
            }
        }
    };
    match op {
        Op::St { v: x, .. }
        | Op::Atom { v: x, .. }
        | Op::StAsync { v: x, .. }
        | Op::StLocal { v: x, .. } => v(x),
        Op::Shfl { a, .. } => v(a),
        // A warp-collective MMA: every operand is data (any lane's feeds every lane's result).
        Op::Hmma { a, b, c, .. } => a.iter().chain(b).chain(c).for_each(v),
        Op::StMatrix { v: xs, .. } | Op::TmemSt { v: xs, .. } => xs.iter().for_each(v),
        _ => {}
    }
    out
}

/// A LOP3 member before it executes (`Taint::lop_pre`): (thread, operand and old destination
/// values, their taint masks, whether the guard is input data).
type LopPre = (usize, [u32; 4], [u32; 4], bool, bool);

/// Taint shadow: which registers and memory bytes hold values derived from input buffers.
/// Proves the equivalence precondition "control flow and addresses do not depend on input
/// data" for one execution: a tainted guard, branch condition, address, coordinate,
/// descriptor or count is reported. (Race-freedom makes every schedule equivalent to it.)
// ASSUME(N-EQUIV): see ASSUMPTIONS.md
#[derive(Default, Debug, Clone)]
pub struct Taint {
    reg: BTreeSet<(u32, u8, u16)>,
    mem: std::collections::HashSet<(u8, u64, u64)>,
    /// The current instruction / async completion: a value operand is tainted ...
    cur_src: bool,
    /// ... or it read a tainted byte / TMEM word.
    cur_read: bool,
    /// Whether the last read (this thread's access) found a tainted byte.
    last_read: bool,
    /// The current load: per thread and destination register, whether the bytes it got
    /// were tainted.
    ld_read: Vec<(usize, Reg, bool)>,
    /// The current shuffle: per thread, whether the source lane's value was tainted.
    pub(crate) shfl_src: Vec<(usize, bool)>,
    /// Register-only instruction: per active thread, whether one of its sources is tainted
    /// (taken before the instruction overwrites them).
    cur_pure: Option<Vec<(usize, bool)>>,
    /// Where control or an address depended on input data.
    pub violations: BTreeSet<String>,
    /// N-FINITE: predicates of the current instruction whose value is fixed by finite inputs
    /// and no overflow (cleared after it), and the pcs of such comparisons.
    fin_clear: Vec<(usize, Reg)>,
    pub fin_pcs: BTreeSet<u32>,
    /// N-FMAX: branches of float atomic max diamonds.
    pub fmax_pcs: BTreeSet<u32>,
    /// Bitwise taint of registers in `reg` that are tainted in some bits only (LOP3 with
    /// input-independent operands: flag words); absent: every bit.
    bits: HashMap<(u32, u8, u16), u32>,
    /// The current LOP3, per member: (thread, operand and old destination values, their
    /// taint masks, whether the guard is input data).
    lop_pre: Vec<LopPre>,
    /// The instruction of each guard violation (segment mode excuses those inside slices).
    pub violation_pcs: BTreeMap<String, u32>,
    /// Position mode (ASSUME(E-CTA-INDEP) cluster symmetry): the taint source is `%ctaid`
    /// instead of the input buffers, and global addresses / TMA coordinates may depend on it.
    pub position: bool,
    /// The current instruction is a generic access whose address depends on the position:
    /// it must resolve to global memory.
    cur_gen_addr: bool,
    /// Position mode: per thread, every position-dependent control value it used (pc, guard
    /// outcome or operand value), in order. Clusters are symmetric when corresponding threads
    /// have equal traces.
    pub trace: std::collections::HashMap<u32, Vec<(u32, u64)>>,
}

impl Taint {
    pub fn taint_alloc(&mut self, alloc: usize, bytes: usize) {
        for b in 0..bytes as u64 {
            self.mem.insert((1, alloc as u64, b));
        }
    }
    fn reg_t(&self, t: usize, r: Reg) -> bool {
        self.reg.contains(&(t as u32, r.file as u8, r.idx))
    }
    pub(crate) fn reg_tainted(&self, t: usize, r: Reg) -> bool {
        self.reg_t(t, r)
    }
    /// The tainted bits of register `r` of thread `t`.
    fn reg_mask(&self, t: usize, r: Reg) -> u32 {
        let k = (t as u32, r.file as u8, r.idx);
        if !self.reg.contains(&k) {
            return 0;
        }
        self.bits.get(&k).copied().unwrap_or(u32::MAX)
    }
    fn set_mask(&mut self, t: usize, r: Reg, m: u32) {
        let k = (t as u32, r.file as u8, r.idx);
        self.bits.remove(&k);
        if m == 0 {
            self.reg.remove(&k);
        } else {
            self.reg.insert(k);
            if m != u32::MAX && !matches!(r.file, File::P | File::UP) {
                self.bits.insert(k, m);
            }
        }
    }
    pub(crate) fn begin_async(&mut self) {
        self.cur_src = false;
        self.cur_read = false;
    }
    fn check_gen(&mut self, key: (u8, u64)) {
        if self.cur_gen_addr && key.0 != 1 {
            self.violations
                .insert("a generic access whose address depends on the CTA's position resolves outside global memory".into());
        }
    }
    pub(crate) fn read(&mut self, key: (u8, u64), off: u64, n: u64) {
        self.check_gen(key);
        self.last_read = (off..off + n).any(|b| self.mem.contains(&(key.0, key.1, b)));
        if self.last_read {
            self.cur_read = true;
        }
    }
    pub(crate) fn write(&mut self, key: (u8, u64), off: u64, n: u64) {
        self.check_gen(key);
        let t = self.cur_src || self.cur_read;
        for b in off..off + n {
            if t {
                self.mem.insert((key.0, key.1, b));
            } else {
                self.mem.remove(&(key.0, key.1, b));
            }
        }
    }
}

impl Machine {
    /// Before an instruction of `members` (active: guard true): check its control and address
    /// operands, and note whether it moves tainted data.
    pub(crate) fn taint_pre(&mut self, members: &[usize], active: &[usize], pc: u32) {
        if self.num.is_none() {
            return;
        }
        let half_only = self.eff_active(active, pc);
        let active = half_only.as_deref().unwrap_or(active);
        let k = self.k.clone();
        let inst = &k.code[pc as usize];
        let srcs = inst.op.src_regs();
        let vals = value_regs(&inst.op);
        let position = self.num.as_ref().unwrap().taint.position;
        // Position mode: global addresses and TMA coordinates may differ between clusters.
        let free = if position {
            pos_free_regs(&inst.op)
        } else {
            (vec![], false)
        };
        let ctl: Vec<Reg> = srcs
            .iter()
            .copied()
            .filter(|r| !vals.contains(r) && !free.0.contains(r))
            .collect();
        // Position mode: control may depend on the position if every cluster decides the
        // same: the guard outcomes and control operand values that depend on it go into the
        // thread's `trace`, compared between clusters afterwards.
        let mut traced: Vec<(usize, u64)> = Vec::new();
        if position {
            let tn = &self.num.as_ref().unwrap().taint;
            if let Some(g) = inst.guard.reg {
                for &t in members.iter().filter(|&&t| tn.reg_t(t, g)) {
                    traced.push((t, self.prd(t, inst.guard) as u64));
                }
            }
            if !pure_op(&inst.op) {
                for &t in active {
                    for &r in ctl.iter().filter(|&&r| tn.reg_t(t, r)) {
                        let v = if matches!(r.file, File::P | File::UP) {
                            self.prd(
                                t,
                                Pred {
                                    reg: Some(r),
                                    neg: false,
                                },
                            ) as u64
                        } else {
                            self.rd(t, &Val::Reg(r, 1)).unwrap_or(u64::MAX)
                        };
                        traced.push((t, v));
                    }
                }
            }
        }
        // A LOP3's operands, old destination and their taint masks (bitwise taint).
        let mut lop_pre = Vec::new();
        if let Op::Lop3 { d, a, b, c, pq, .. } = &inst.op {
            if !position {
                let tn = &self.num.as_ref().unwrap().taint;
                let g = inst.guard.reg;
                for &t in members {
                    let mut vals = [0u32; 4];
                    let mut masks = [0u32; 4];
                    for (i, v) in [a, b, c, &Val::Reg(*d, 1)].into_iter().enumerate() {
                        vals[i] = self.rd(t, v).unwrap_or(0) as u32;
                        if let Val::Reg(r, _) = v {
                            masks[i] = tn.reg_mask(t, *r);
                        }
                    }
                    // The predicate operand's taint, before P (which may be Pq) is written.
                    let qt = pq.is_some_and(|(q, _)| q.reg.is_some_and(|r| tn.reg_t(t, r)));
                    lop_pre.push((t, vals, masks, g.is_some_and(|g| tn.reg_t(t, g)), qt));
                }
            }
        }
        // ASSUME(N-EQUIV): a selection by an input-independent predicate is its chosen operand.
        let sel_taint = match &inst.op {
            Op::Sel { a, b, p, .. } | Op::Fsel { a, b, p, .. } if !position => {
                let tn = &self.num.as_ref().unwrap().taint;
                let srcs_t = |t: usize| srcs.iter().any(|&r| tn.reg_t(t, r));
                Some(
                    active
                        .iter()
                        .map(|&t| {
                            let x = if p.reg.is_some_and(|r| tn.reg_t(t, r)) {
                                srcs_t(t)
                            } else {
                                let v = if self.prd(t, *p) { a } else { b };
                                matches!(v, Val::Reg(r, _) if tn.reg_t(t, *r))
                            };
                            (t, x)
                        })
                        .collect::<Vec<_>>(),
                )
            }
            _ => None,
        };
        let describe = k.describe(pc as usize);
        let ns = self.num.as_mut().unwrap();
        ns.taint.lop_pre = lop_pre;
        ns.touched.clear();
        if ns.segments {
            for &t in members {
                let f = if active.contains(&t) {
                    0
                } else {
                    crate::seg::GUARD_FALSE
                };
                ns.trace.entry(t as u32).or_default().push(pc | f);
            }
        }
        let tn = &mut ns.taint;
        for (t, v) in traced {
            tn.trace.entry(t as u32).or_default().push((pc, v));
        }
        tn.cur_read = false;
        tn.ld_read.clear();
        tn.shfl_src.clear();
        tn.cur_gen_addr = false;
        // Position mode: reading %ctaid is the source.
        let seed = position
            && inst
                .op
                .src_vals()
                .iter()
                .any(|v| matches!(v, Val::SReg(SReg::CtaId(_))));
        if let Some(g) = inst.guard.reg {
            if !position && members.iter().any(|&t| tn.reg_t(t, g)) {
                // ASSUME(N-FMAX): both sides of a float atomic max diamond compute the same
                // update (the numerics check its operands are non-NaN floats).
                if fmax_diamond(&k.code, pc as usize).is_some() {
                    tn.fmax_pcs.insert(pc);
                } else {
                    let v = format!("guard of {describe} depends on input data");
                    tn.violation_pcs.insert(v.clone(), pc);
                    tn.violations.insert(v);
                }
            }
        }
        if pure_op(&inst.op) {
            tn.cur_pure = Some(
                active
                    .iter()
                    .map(|&t| (t, seed || srcs.iter().any(|&r| tn.reg_t(t, r))))
                    .collect(),
            );
            // A reduction across the warp: every lane's result depends on every lane's
            // operand.
            if matches!(inst.op, Op::Redux { .. }) {
                if let Some(per) = tn.cur_pure.as_mut() {
                    let any = per.iter().any(|x| x.1);
                    per.iter_mut().for_each(|x| x.1 = any);
                }
            }
            // SEL / FSEL with an input-independent predicate: the result is the chosen operand
            // for every input, so it carries that operand's taint only (a causal mask's
            // `FSEL R, R, -INF, P` leaves the input-independent -inf).
            if let Some(per) = sel_taint {
                tn.cur_pure = Some(per);
            }
            return;
        }
        tn.cur_pure = None;
        tn.cur_src = seed || active.iter().any(|&t| vals.iter().any(|&r| tn.reg_t(t, r)));
        if free.1
            && active
                .iter()
                .any(|&t| free.0.iter().any(|&r| tn.reg_t(t, r)))
        {
            tn.cur_gen_addr = true;
        }
        if !position && active.iter().any(|&t| ctl.iter().any(|&r| tn.reg_t(t, r))) {
            tn.violations.insert(format!(
                "an address / control operand of {describe} depends on input data"
            ));
        }
    }

    /// After it: its destination registers carry the taint of what it read.
    /// The threads of `active` that execute instruction `pc`: a half-warp tcgen05.ld
    /// (`.16dp32bit_t0_t15` / `_t16_t31`) writes only its half's registers; the other lanes
    /// keep theirs (and their nodes and taint), as if their guard were false.
    fn eff_active(&self, active: &[usize], pc: u32) -> Option<Vec<usize>> {
        if let Op::TmemLd { half: Some(h), .. } = self.k.code[pc as usize].op {
            Some(
                active
                    .iter()
                    .copied()
                    .filter(|&t| self.threads[t].lane / 16 == h as u32)
                    .collect(),
            )
        } else {
            None
        }
    }

    pub(crate) fn taint_post(&mut self, members: &[usize], active: &[usize], pc: u32) {
        let half_only = self.eff_active(active, pc);
        let active = half_only.as_deref().unwrap_or(active);
        self.half_post(active, pc);
        let Some(ns) = self.num.as_mut() else { return };
        let inst = &self.k.code[pc as usize];
        // A destination no numerics hook set loses its node (as `num_clobber`). Segment mode:
        // under a guard that depends on input data, every member's destinations lose theirs,
        // executed or not (other inputs decide otherwise; slices cover both outcomes).
        let data_guard = ns.segments
            && inst
                .guard
                .reg
                .is_some_and(|g| members.iter().any(|&t| ns.taint.reg_t(t, g)));
        // ASSUME(N-EQUIV): the members under a data-dependent guard: for a per-thread register operation, those
        // whose own guard depends on input data (another member's guard is fixed for every
        // input, so its destination is as executed); for warp-collective ones, every member.
        let per_thread = pure_op(&inst.op)
            && !inst.uniform
            && !matches!(
                inst.op,
                Op::VoteAny { .. } | Op::VoteAll { .. } | Op::Redux { .. }
            );
        let dg: Vec<usize> = match inst.guard.reg {
            Some(g) if data_guard && per_thread => members
                .iter()
                .copied()
                .filter(|&t| ns.taint.reg_t(t, g))
                .collect(),
            _ if data_guard => members.to_vec(),
            _ => vec![],
        };
        let mut touched_ts: Vec<usize> = active.to_vec();
        touched_ts.extend(dg.iter().copied().filter(|t| !active.contains(t)));
        for r in inst.op.dst_regs() {
            for &t in &touched_ts {
                let k = (t as u32, file_tag(r.file), r.idx);
                if dg.contains(&t) || !ns.touched.contains(&k) {
                    ns.reg.remove(&k);
                }
            }
        }
        let tn = &mut ns.taint;
        let t_any = tn.cur_src || tn.cur_read;
        let mut per = tn.cur_pure.take();
        // ASSUME(N-EQUIV): a shuffle: each destination carries the taint of the lane it read.
        if per.is_none() && matches!(inst.op, Op::Shfl { .. }) && !tn.shfl_src.is_empty() {
            per = Some(std::mem::take(&mut tn.shfl_src));
        }
        // ASSUME(N-EQUIV): a load: each thread's destination register carries the taint of the bytes it got.
        let ld_read = if per.is_none() && matches!(inst.op, Op::Ld { .. }) {
            std::mem::take(&mut tn.ld_read)
        } else {
            vec![]
        };
        for r in inst.op.dst_regs() {
            for &t in &touched_ts {
                tn.bits.remove(&(t as u32, r.file as u8, r.idx));
            }
            for &t in active {
                let k = (t as u32, r.file as u8, r.idx);
                let tainted = match &per {
                    Some(p) => p.iter().any(|&(u, x)| u == t && x),
                    None => ld_read
                        .iter()
                        .find(|x| x.0 == t && x.1 == r)
                        .map_or(t_any, |x| x.2),
                };
                if tainted {
                    tn.reg.insert(k);
                } else {
                    tn.reg.remove(&k);
                }
            }
            // Under a data-dependent guard the member's destination depends on input data.
            for &t in &dg {
                tn.reg.insert((t as u32, r.file as u8, r.idx));
            }
        }
        // ASSUME(N-FINITE): a comparison fixed by finiteness is the same for every input.
        for (t, r) in std::mem::take(&mut tn.fin_clear) {
            tn.reg.remove(&(t as u32, r.file as u8, r.idx));
        }
        // Bitwise taint of a LOP3: each result bit is the LUT of the operands' same bits; it is
        // input data where it can take both values as the tainted operand bits (and, under a
        // data-dependent guard in segment mode, the old destination bit) vary.
        // ASSUME(N-EQUIV): see ASSUMPTIONS.md
        if let Op::Lop3 { d, lut, p, .. } = &inst.op {
            let lop = std::mem::take(&mut tn.lop_pre);
            for (t, vals, masks, gt, qt) in lop {
                let runs = active.contains(&t);
                let data_guard = gt && ns.segments;
                if !runs && !data_guard {
                    // Its own guard is false for every input: the register is unchanged (the
                    // blanket rule above tainted every member's destination).
                    tn.set_mask(t, *d, masks[3]);
                    continue;
                }
                let (m, one) = lop3_taint(*lut, vals, masks, data_guard);
                tn.set_mask(t, *d, m);
                // P = (result != 0): input-independent if an input-independent bit is 1.
                // A tainted predicate operand (L-LOP3-P) taints P.
                if let (Some(p), true) = (p, runs && !data_guard) {
                    tn.set_mask(t, *p, if one == 0 && m != 0 || qt { 1 } else { 0 });
                }
            }
        }
        if !ns.segments {
            return;
        }
        // Segment mode: what each destination now holds, for slices.
        let k = self.k.clone();
        let mut defs = Vec::new();
        for r in k.code[pc as usize].op.dst_regs() {
            for &t in active {
                let val = if matches!(r.file, File::P | File::UP) {
                    self.prd(
                        t,
                        Pred {
                            reg: Some(r),
                            neg: false,
                        },
                    ) as u64
                } else {
                    self.rd(t, &Val::Reg(r, 1)).unwrap_or(0) & 0xffff_ffff
                };
                defs.push((t, r, val));
            }
        }
        let ns = self.num.as_mut().unwrap();
        for (t, r, val) in defs {
            let pos = ns
                .trace
                .get(&(t as u32))
                .map_or(0, |v| v.len().saturating_sub(1));
            let d = Def {
                val,
                tainted: ns.taint.reg_t(t, r),
                tmask: ns.taint.reg_mask(t, r),
                node: ns.node_of_reg(t, r),
            };
            ns.defs
                .insert((t as u32, pos as u32, file_tag(r.file), r.idx), d);
        }
    }
}

#[cfg(test)]
mod fin_tests {
    use super::*;
    use crate::seg::SKind;
    use crate::seg::SOp;
    use crate::seg::SProg;
    use crate::seg::SVal;

    /// (op, destination, source registers and their values) of one instruction.
    type InstSpec = (Op, Reg, Vec<(Reg, SVal)>);

    fn prog(ops: Vec<InstSpec>) -> SProg {
        let n = ops.len() as u32;
        SProg {
            args: vec![0],
            ops: ops
                .into_iter()
                .map(|(op, dst, regs)| SOp {
                    pc: 0,
                    kind: SKind::Inst {
                        op,
                        dst,
                        regs,
                        cvals: vec![],
                    },
                })
                .collect(),
            out: SVal::Op(n - 1),
            extra_pcs: Default::default(),
        }
    }
    fn r(idx: u16) -> Reg {
        Reg { file: File::R, idx }
    }
    fn fadd(a: Val, b: Val, neg: [bool; 2]) -> Op {
        Op::Fadd {
            d: r(3),
            a,
            b,
            rnd: Rnd::Rn,
            neg,
            abs: [false, false],
            ftz: false,
            sat: false,
        }
    }

    /// N-FINITE: -inf - m with m finite is -inf for every admissible input, and exp2 of it
    /// the constant 0; an unknown m, a finite start, or -inf times m are not fixed.
    #[test]
    fn infinity_minus_finite_is_fixed() {
        let ninf = 0xff80_0000u64;
        let sub = fadd(Val::Reg(r(1), 1), Val::Reg(r(2), 1), [false, true]);
        let regs = vec![(r(1), SVal::Const(ninf)), (r(2), SVal::Arg(0))];
        let p = prog(vec![(sub.clone(), r(3), regs.clone())]);
        assert_eq!(seg_val(&p, &[true]), FinVal::K(ninf));
        assert_eq!(seg_val(&p, &[false]), FinVal::Top);
        assert!(!seg_fin(&p, &[true]));
        let ex2 = Op::Mufu {
            d: r(4),
            a: Val::Reg(r(3), 1),
            func: MufuFunc::Ex2,
            fmt: MufuFmt::F32,
            hi: false,
            neg: false,
            abs: false,
            ptx: None,
        };
        let p2 = prog(vec![
            (sub.clone(), r(3), regs.clone()),
            (ex2.clone(), r(4), vec![(r(3), SVal::Op(0))]),
        ]);
        assert_eq!(seg_val(&p2, &[true]), FinVal::K(0));
        assert!(seg_fin(&p2, &[true]));
        // A finite start: finite, not fixed.
        let p3 = prog(vec![(
            sub.clone(),
            r(3),
            vec![(r(1), SVal::Const(0xc000_0000)), (r(2), SVal::Arg(0))],
        )]);
        assert_eq!(seg_val(&p3, &[true]), FinVal::Fin);
        // -inf * m: NaN for m = 0.
        let mul = Op::Ffma {
            d: r(3),
            a: Val::Reg(r(1), 1),
            b: Val::Reg(r(2), 1),
            c: None,
            neg: [false; 3],
            abs: [false; 3],
            rnd: Rnd::Rn,
            ftz: false,
            sat: false,
        };
        let p4 = prog(vec![(mul, r(3), regs.clone())]);
        assert_eq!(seg_val(&p4, &[true]), FinVal::Top);
        // A selection by a constant predicate: the chosen operand only (a causal mask's
        // -inf where it is not chosen); by an unknown one: the join.
        let p0 = Reg {
            file: File::P,
            idx: 0,
        };
        let fsel = Op::Fsel {
            d: r(5),
            a: Val::Reg(r(2), 1),
            b: Val::Imm(ninf),
            p: Pred {
                reg: Some(p0),
                neg: false,
            },
            neg: [false; 2],
            abs: [false; 2],
        };
        let ps = |pv: SVal| {
            prog(vec![(
                fsel.clone(),
                r(5),
                vec![(r(2), SVal::Arg(0)), (p0, pv)],
            )])
        };
        assert_eq!(seg_val(&ps(SVal::Const(1)), &[true]), FinVal::Fin);
        assert_eq!(seg_val(&ps(SVal::Const(0)), &[true]), FinVal::K(ninf));
        assert_eq!(seg_val(&ps(SVal::Arg(0)), &[true]), FinVal::Top);
    }
}

#[cfg(test)]
mod fmax_tests {
    /// N-FMAX: for non-NaN floats L and v, the signed max where v's sign bit is clear and the
    /// unsigned min where it is set is FMNMX's maximum (special values and random patterns).
    #[test]
    fn atomic_max_float_is_the_float_max() {
        let special = [
            0u32,
            0x8000_0000,
            1,
            0x8000_0001,
            0x007f_ffff,
            0x807f_ffff,
            0x0080_0000,
            0x8080_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x7f80_0000,
            0xff80_0000,
        ];
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut vals: Vec<u32> = special.to_vec();
        for _ in 0..4000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            vals.push(s as u32);
        }
        vals.retain(|&b| !f32::from_bits(b).is_nan());
        for &l in &vals {
            for &v in &vals {
                let r = if v >> 31 == 0 {
                    (l as i32).max(v as i32) as u32
                } else {
                    l.min(v)
                };
                assert_eq!(
                    r,
                    crate::fp::f32_minmax(l, v, false, false, false),
                    "{l:#x} {v:#x}"
                );
            }
        }
    }
}

#[cfg(test)]
mod lop3_taint_tests {
    use super::lop3_taint;

    /// CUTLASS's flag word (fixtures/c_cve_sdpa_f32 R108): `@P LOP3 R, R, 0x1, RZ, 0xfc`
    /// (R |= 1) under a data-dependent P taints bit 0 only; a test of bit 1 (`LOP3 P, RZ, R,
    /// 0x2, RZ, 0xc0`) stays input-independent, a test of bit 0 does not; AND with a constant
    /// clears the taint of the bits it zeroes; a data bit in an XOR stays data.
    #[test]
    fn flag_bits_keep_their_own_taint() {
        // R = R | 0x1 under a data guard; R's old value 0x6, untainted.
        let (m, _) = lop3_taint(0xfc, [0x6, 0x1, 0, 0x6], [0, 0, 0, 0], true);
        assert_eq!(m, 0x1);
        // P = (R & 0x2) != 0: no tainted bit survives the mask, bit 1 is 1.
        let (m2, one) = lop3_taint(0xc0, [0x7, 0x2, 0, 0], [m, 0, 0, 0], false);
        assert_eq!((m2, one), (0, 0x2));
        // P = (R & 0x1) != 0: data.
        let (m3, one3) = lop3_taint(0xc0, [0x6, 0x1, 0, 0], [m, 0, 0, 0], false);
        assert_eq!((m3, one3), (0x1, 0));
        // R & 0xfffffffe of a fully tainted R: every bit but 0.
        let (m4, _) = lop3_taint(0xc0, [5, 0xffff_fffe, 0, 0], [u32::MAX, 0, 0, 0], false);
        assert_eq!(m4, 0xffff_fffe);
        // R ^ x with x tainted in bit 3: bit 3.
        let (m5, _) = lop3_taint(0x3c, [0, 0x8, 0, 0], [0, 0x8, 0, 0], false);
        assert_eq!(m5, 0x8);
    }
}

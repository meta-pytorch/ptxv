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
    /// An input-derived value without a node or a slice (`Leaf`s are constants).
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
    /// Segment mode: a choice made by input data. `pred` is the slice computing the predicate
    /// (0 / 1) from its argument nodes; the value is `a` where `pred ^ neg` holds, else `b`.
    /// A store under a data-dependent guard (`@P STS [x], v`) leaves `Sel(P, v, old)` in every
    /// word it covers, whichever way this input went (N-ERRDIST-SEL).
    Sel {
        pred: SegRef,
        neg: bool,
        a: u32,
        b: u32,
    },
    /// A TF32 part of the f32 value `arg` (3xTF32 split, N-TF32): the high part the TF32 MMA
    /// reads, or a low-part factor of a split.
    Tf32 {
        arg: u32,
        part: Tf32Part,
    },
}

/// What a 16-bit bf16 operand half holds for the numerics (`Machine::bf16_half`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bf16Half {
    Node(u32),
    Const(u16),
    Unknown,
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
    pred_src: HashMap<(u32, Reg), (usize, FCmp, u32, u32)>,
    /// Instructions covered by materialized slices (their data-dependent guards are part of
    /// the slices), and why some value could not be materialized.
    pub seg_pcs: BTreeSet<u32>,
    /// Instructions of div.rn sequences (computed on first use).
    idiom_pcs: Option<BTreeSet<u32>>,
    pub seg_fails: BTreeMap<u32, String>,
    /// Stores under a data-dependent guard modeled as selections (`NodeKind::Sel`): their
    /// guards are part of the dataflow, not a different one.
    pub sel_pcs: BTreeSet<u32>,
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
    let atom = |i: usize| fmax_atom(code, i);
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
    if !sign_setp(code, bra, p, v1) {
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

/// Whether P = (v as i32 >= 0) at the branch `first`: the last write of P before it is
/// `ISETP.GE P, v, RZ` (or `.GT P, v, -1`), unguarded, in straight-line code that writes
/// neither P nor v after it and that no branch enters.
fn sign_setp(code: &[Inst], first: usize, p: Reg, v: Reg) -> bool {
    let mut i = first;
    let setp = loop {
        let Some(j) = i.checked_sub(1) else {
            return false;
        };
        i = j;
        let inst = &code[i];
        if inst.op.dst_regs().contains(&p) {
            break i;
        }
        if inst.op.dst_regs().contains(&v)
            || matches!(
                inst.op,
                Op::Bra { .. } | Op::Exit | Op::Call { .. } | Op::Ret { .. }
            )
        {
            return false;
        }
    };
    let entered = code
        .iter()
        .any(|x| matches!(x.op, Op::Bra { target, .. } if target > setp && target <= first));
    if entered || !code[setp].guard.is_always() {
        return false;
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
        return false;
    };
    let k = match sb {
        Val::Imm(x) => *x as u32,
        _ => return false,
    };
    let sign_clear = matches!((cmp, k), (Cmp::GeS, 0) | (Cmp::GtS, 0xffff_ffff));
    d == p && sa == v && and.is_always() && sign_clear
}

/// An unguarded shared 4-byte `ATOMS.MAX.S32` / `ATOMS.MIN` (unsigned) of a register with no
/// destination at `i`: (op, address, value register).
fn fmax_atom(code: &[Inst], i: usize) -> Option<(AtomOp, Addr, Reg)> {
    match &code.get(i)?.op {
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
    }
}

/// The sequential form of the float atomic max (N-FMAX, CuTeDSL's `smem_fmax`): two skips on
/// one sign predicate, `@P BRA A1; ATOMS.MIN RZ, [a], v; A1: ...; @!P BRA A2;
/// ATOMS.MAX.S32 RZ, [a], v; A2:`, with only `NOP` / `BSYNC` / `BSSY` between the first atomic
/// and the second branch, in either order. Exactly one atomic executes: the unsigned min
/// where v's sign bit is set, the signed max where it is clear. Returns the two atomics' pcs
/// (max, min) for either branch `bra`.
// ASSUME(N-FMAX): see ASSUMPTIONS.md
pub fn fmax_seq(code: &[Inst], bra: usize) -> Option<(usize, usize)> {
    // A skip: `@[!]P BRA bra+2` over an atomic; (P, negated, atomic).
    let skip = |b: usize| {
        let inst = code.get(b)?;
        let Op::Bra {
            target,
            any: false,
            tid0_falls: false,
            cond: None,
            div: None,
        } = inst.op
        else {
            return None;
        };
        let p = inst.guard.reg?;
        if target != b + 2 {
            return None;
        }
        let (op, a, v) = fmax_atom(code, b + 1)?;
        Some((p, inst.guard.neg, op, a, v))
    };
    let filler = |i: usize| {
        matches!(code[i].op, Op::Nop | Op::Bsync { .. } | Op::Bssy { .. })
            && code[i].guard.is_always()
    };
    let (p, neg, op, a, v) = skip(bra)?;
    // The partner: after this skip's atomic, or before this branch.
    let mut partner = None;
    let mut j = bra + 2;
    while j < code.len() && filler(j) {
        j += 1;
    }
    if let Some(x) = skip(j) {
        partner = Some((bra, j, x));
    }
    if partner.is_none() {
        let mut j = bra;
        while j > 0 && filler(j - 1) {
            j -= 1;
        }
        // code[j - 1] is the partner's atomic, code[j - 2] its branch.
        if let Some(b0) = j.checked_sub(2) {
            if let Some(x) = skip(b0) {
                partner = Some((b0, bra, x));
            }
        }
    }
    let (first, second, (p2, neg2, op2, a2, v2)) = partner?;
    let ((pf, negf, opf), (ps, negs, ops)) = if first == bra {
        ((p, neg, op), (p2, neg2, op2))
    } else {
        ((p2, neg2, op2), (p, neg, op))
    };
    if pf != ps || negf == negs || opf == ops || a != a2 || v != v2 {
        return None;
    }
    // No branch enters between the first skip's atomic and the second branch, other than the
    // first skip's own target.
    let entered = code.iter().enumerate().any(|(i, x)| {
        matches!(x.op, Op::Bra { target, .. }
            if target > first + 1 && target <= second && !(i == first && target == first + 2))
    });
    if entered || !sign_setp(code, first, pf, v) {
        return None;
    }
    // Each atomic runs where its skip's guard is false: the min where the sign bit is set
    // (P false), so the min's skip is `@P` and the max's `@!P`.
    let (max_pc, min_pc) = if opf == AtomOp::MinU {
        (second + 1, first + 1)
    } else {
        (first + 1, second + 1)
    };
    let min_neg = if opf == AtomOp::MinU { negf } else { negs };
    if min_neg {
        return None;
    }
    Some((max_pc, min_pc))
}

/// A branch of a float atomic max (N-FMAX), either form: the atomics' pcs (max, min).
pub fn fmax_branch(code: &[Inst], bra: usize) -> Option<(usize, usize)> {
    fmax_diamond(code, bra).or_else(|| fmax_seq(code, bra))
}

/// Bits of an operand before its `-x` modifier (`vals` hold them after).
fn fmod_sign(u: u32, neg: bool) -> u32 {
    if neg { u ^ 0x8000_0000 } else { u }
}

fn file_tag(f: File) -> u8 {
    f as u8
}

/// Numerics pseudo-index bit of the high half of a 64-bit PTX register: lane 1 of a packed
/// f32x2 value (`mov.b64 %rd, {lo, hi}`, `fma.rn.f32x2`, `ld.b64`) is tracked under
/// `idx | V_HI`. Never an executable register: the PTX front-end keeps indices below it.
pub(crate) const V_HI: u16 = 0x8000;

/// The register holding lane `i` (0 / 1) of a packed f32x2 value whose low lane is `r`: the
/// next SASS register, or the high half of a PTX register (`V_HI`).
pub(crate) fn lane_reg(r: Reg, i: usize) -> Reg {
    match (r.file, i) {
        (_, 0) => r,
        (File::V, _) => Reg {
            file: File::V,
            idx: r.idx | V_HI,
        },
        _ => Reg {
            file: r.file,
            idx: r.idx + i as u16,
        },
    }
}

/// Lane `i` of a packed f32x2 operand (`F2Lanes`), as a 32-bit value.
fn f2_lane_val((v, l): &(Val, F2Lanes), i: usize) -> Val {
    match (v, l) {
        (Val::Reg(r, _), F2Lanes::Pair | F2Lanes::Swap) => {
            let k = usize::from((*l == F2Lanes::Swap) == (i == 0));
            Val::Reg(lane_reg(*r, k), 1)
        }
        (Val::Reg(r, _), F2Lanes::Bcast) => Val::Reg(*r, 1),
        _ => v.clone(),
    }
}

/// A PTX register's high-half pseudo-register (`V_HI`).
fn is_v_hi(r: Reg) -> bool {
    r.file == File::V && r.idx & V_HI != 0
}

/// A store under a data-dependent guard, before it executes (`Machine::num_gst_pre`).
pub(crate) struct GstPre {
    keys: Vec<(u8, u64, u64)>,
    old: Vec<Option<u32>>,
    pred: Result<crate::seg::SProg, String>,
    neg: bool,
}

/// The 32-bit word at byte `at` past `loc` (shared or global memory).
fn word_at(m: &Machine, loc: Loc, at: u64) -> u32 {
    let b: Vec<u8> = match loc {
        Loc::Shared { cta, off } => (0..4)
            .map(|i| {
                *m.smem[cta as usize]
                    .bytes
                    .get(off as usize + at as usize + i)
            })
            .collect(),
        Loc::Global { alloc, off } => (0..4)
            .map(|i| *m.gmem[alloc].bytes.get((off + at) as usize + i))
            .collect(),
        Loc::Param { .. } => vec![0; 4],
    };
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
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
            NodeKind::Sel { a, b, neg, .. } => {
                format!(
                    "sel({}p ? {} : {})",
                    if *neg { "!" } else { "" },
                    d(*a),
                    d(*b)
                )
            }
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
        // A write of a whole PTX register also replaces its high half (a lane-1 node is set
        // after its lane 0).
        if d.file == File::V && !is_v_hi(d) {
            let h = (t as u32, file_tag(d.file), d.idx | V_HI);
            self.touched.push(h);
            self.reg.remove(&h);
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
        let hw = f32::from_bits(fp::f32_minmax(
            ha.to_bits(),
            hb.to_bits(),
            min,
            false,
            false,
        ));
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
    fn leaf_with_origin(&mut self, bits: u32, pc: u32, input_derived: bool) -> u32 {
        let n = self.leaf(bits, pc);
        self.nodes[n as usize].kind = if input_derived {
            NodeKind::Opaque
        } else {
            NodeKind::Leaf
        };
        if !input_derived {
            self.const_leaves.insert(n);
        }
        n
    }

    /// A missing operand history stays opaque when the operand depends on input data.
    fn leaf_for(&mut self, t: usize, v: &Val, bits: u32, pc: u32) -> u32 {
        let input_derived = matches!(v, Val::Reg(r, _) if self.taint.reg_t(t, *r));
        self.leaf_with_origin(bits, pc, input_derived)
    }

    /// A leaf of an input-independent constant.
    fn const_leaf(&mut self, bits: u32, pc: u32) -> u32 {
        let n = self.leaf(bits, pc);
        self.const_leaves.insert(n);
        n
    }

    /// Whether node `n` is non-NaN for every admissible input: a non-NaN constant leaf, or
    /// `fin` (N-FINITE).
    pub(crate) fn non_nan(&mut self, n: u32) -> bool {
        let konst = matches!(self.nodes[n as usize].kind, NodeKind::Leaf)
            && self.const_leaves.contains(&n)
            && !f32::from_bits(self.nodes[n as usize].hw).is_nan();
        konst || self.fin(n)
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
                NodeKind::Sel { pred, a, b, .. } => {
                    pred.0.args.iter().copied().chain([*a, *b]).collect()
                }
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
                // The predicate's arguments do not reach the value.
                NodeKind::Sel { a, b, .. } => f(a) && f(b),
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
    pub(crate) fn note_write_span(&mut self, key: (u8, u64), off: u64, bytes: u64) {
        if key.0 == 1 {
            for word in (off & !3..off + bytes).step_by(4) {
                self.written.insert((key.1, word));
            }
        }
    }
    fn note_write(&mut self, k: (u8, u64, u64)) {
        if k.0 == 1 {
            self.written.insert((k.1, k.2));
        }
    }
}

#[derive(Clone, Debug, Hash)]
pub struct ReadSnapshot {
    origins: Vec<bool>,
    nodes: Vec<Option<u32>>,
    half: Vec<Option<crate::halfin::Src>>,
}

pub(crate) struct StoreOrigins {
    words: [bool; 64],
    count: usize,
    guard: bool,
    coverage: bool,
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
                        // The sequence's own instructions (its fast path's Newton step; not
                        // instructions interleaved with it).
                        set.extend(crate::seg::rcprn_pcs(self, q as u32));
                    }
                    if let Some(lo) = crate::seg::divrn_lo(self, q as u32) {
                        // From the MUFU.RCP of the divisor (constant divisor: the first
                        // quotient FFMA) to the BSYNC.
                        set.extend(lo..=q as u32);
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
    /// `nan` (`FMNMX.NAN`): NaN-propagating, the same maximum where no operand is NaN; a node
    /// only when every operand is non-NaN for every admissible input (N-FINITE).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn num_fmnmx(
        &mut self,
        t: usize,
        d: Reg,
        ops: &[&Val],
        bits: &[u32],
        min: Option<bool>,
        nan: bool,
        pc: u32,
    ) {
        let Some(ns) = self.num.as_ref() else { return };
        if !ns.segments || min.is_none() {
            self.num_clobber(t, d, ops, pc);
            return;
        }
        let mut nodes = Vec::new();
        for (v, &b) in ops.iter().zip(bits) {
            match self.num_node_mat(t, v) {
                Some(n) => nodes.push(n),
                None => {
                    // A constant operand (Triton's running maximum starts at -inf), as read
                    // before the instruction wrote its destination.
                    let ns = self.num.as_mut().unwrap();
                    if matches!(v, Val::Reg(r, _) if ns.taint.reg_t(t, *r)) {
                        self.num_clobber(t, d, ops, pc);
                        return;
                    }
                    nodes.push(ns.const_leaf(b, pc));
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
        if nan && !nodes.clone().into_iter().all(|n| ns.non_nan(n)) {
            self.num_clobber(t, d, ops, pc);
            return;
        }
        let ns = self.num.as_mut().unwrap();
        let mut acc = nodes[0];
        for &n in &nodes[1..] {
            acc = ns.max_node(acc, n, min, pc);
        }
        ns.set(t, d, Some(acc));
    }

    /// Segment mode: `R2UR d, a` (a value uniform over the active lanes): the uniform register holds the
    /// first active lane's value, so every active lane's `d` gets that lane's node (each lane
    /// reads the uniform register).
    pub(crate) fn num_r2ur(&mut self, active: &[usize], d: Reg, a: &Val) {
        let Some(&t0) = active.first() else { return };
        if !self.num.as_ref().is_some_and(|n| n.segments) {
            return;
        }
        let n = self.num_node_mat(t0, a);
        let ns = self.num.as_mut().unwrap();
        for &t in active {
            ns.set(t, d, n);
        }
    }

    /// Segment mode: `CREDUX.MAX|MIN.F32[.NAN]` (redux.sync.max|min[.NaN].f32, no `.abs`)
    /// over the active lanes: the maximum (minimum) of their operands, exact and
    /// order-independent, as a tree of `Max` nodes (the input-independent lanes folded into
    /// one constant). The uniform destination is read by every lane: each active lane gets the
    /// node. `.NAN` requires every operand non-NaN for every admissible input (N-FINITE),
    /// where it is the same maximum. Otherwise the destination is untracked.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn num_redux_f(
        &mut self,
        active: &[usize],
        vals: &[u32],
        d: Reg,
        a: &Val,
        min: bool,
        nan: bool,
        pc: u32,
    ) {
        if !self.num.as_ref().is_some_and(|n| n.segments) {
            return;
        }
        let mut nodes = Vec::new();
        let mut konst: Option<u32> = None;
        for (&t, &bits) in active.iter().zip(vals) {
            match self.num_node_mat(t, a) {
                Some(n) => nodes.push(n),
                None => {
                    let ns = self.num.as_ref().unwrap();
                    if matches!(a, Val::Reg(r, _) if ns.taint.reg_t(t, *r)) {
                        return;
                    }
                    konst = Some(match konst {
                        Some(k) => fp::f32_minmax(k, bits, min, nan, false),
                        None => bits,
                    });
                }
            }
        }
        let ns = self.num.as_mut().unwrap();
        if nodes.is_empty() {
            return;
        }
        if let Some(k) = konst {
            if f32::from_bits(k).is_nan() {
                return;
            }
            nodes.push(ns.const_leaf(k, pc));
        }
        if nan && !nodes.clone().into_iter().all(|n| ns.non_nan(n)) {
            ns.seg_fails
                .entry(pc)
                .or_insert_with(|| "CREDUX.NAN of a value not provably non-NaN (no node)".into());
            return;
        }
        let mut acc = nodes[0];
        for &n in &nodes[1..] {
            acc = ns.max_node(acc, n, min, pc);
        }
        for &t in active {
            ns.set(t, d, Some(acc));
        }
    }

    /// Segment mode: FSETP `P = a cmp b` (combined with PT, no modifiers): remembered, so a
    /// following SEL choosing between a and b by P is a `Max`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn num_fsetp(&mut self, t: usize, d: Reg, cmp: FCmp, a: &Val, b: &Val, plain: bool) {
        let Some(ns) = self.num.as_mut() else { return };
        if !ns.segments {
            return;
        }
        let key = (t as u32, d);
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
        // A PTX register's high half (an f32x2 lane) moves with it.
        if let (File::V, Val::Reg(r, _)) = (d.file, a) {
            if r.file == File::V {
                let h = ns.node_of_reg(t, lane_reg(*r, 1));
                ns.set(t, lane_reg(d, 1), h);
            }
        }
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

    /// PTX `mov.b64 d, {lo, hi}`: the halves keep their nodes as d's lanes (exact moves).
    pub(crate) fn num_pack64(&mut self, t: usize, d: Reg, lo: &Val, hi: &Val) {
        let Some(ns) = self.num.as_mut() else { return };
        let (a, b) = (ns.node_of(t, lo), ns.node_of(t, hi));
        ns.set(t, d, a);
        if d.file == File::V {
            ns.set(t, lane_reg(d, 1), b);
        }
    }

    /// PTX `mov.b64 {lo, hi}, a`: a's lanes keep their nodes (exact moves).
    pub(crate) fn num_unpack64(&mut self, t: usize, lo: Option<Reg>, hi: Option<Reg>, a: &Val) {
        let Some(ns) = self.num.as_mut() else { return };
        let (na, nb) = match a {
            Val::Reg(r, _) if r.file == File::V => {
                (ns.node_of_reg(t, *r), ns.node_of_reg(t, lane_reg(*r, 1)))
            }
            _ => (None, None),
        };
        if let Some(r) = lo {
            ns.set(t, r, na);
        }
        if let Some(r) = hi {
            ns.set(t, r, nb);
        }
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
        let lane_val = f2_lane_val;
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
                        .unwrap_or_else(|| ns.leaf_for(t, &v[k], vals[k][i], pc));
                    *o = Some((nx, vals[k][i]));
                }
                _ => lost |= v.iter().any(|s| ns.node_of(t, s).is_some()),
            }
        }
        for (i, o) in out.into_iter().enumerate() {
            let di = lane_reg(d, i);
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
        let lane_val = f2_lane_val;
        let lanes: Vec<[Val; 3]> = (0..2).map(|i| ops.map(|o| lane_val(o, i))).collect();
        // PTX registers: a tainted lane without a node would need an instruction slice of
        // half a 64-bit register, which seg.rs does not build. Fail closed.
        let ns = self.num.as_ref().unwrap();
        if lanes.iter().flatten().any(|v| {
            matches!(v, Val::Reg(r, _) if r.file == File::V
                && ns.node_of(t, v).is_none() && ns.taint.reg_t(t, *r))
        }) {
            let ns = self.num.as_mut().unwrap();
            ns.seg_fails
                .entry(pc)
                .or_insert_with(|| "f32x2 lane of a PTX register without a node (no slice)".into());
            for (i, v) in lanes.iter().enumerate() {
                self.num_clobber(t, lane_reg(d, i), &[&v[0], &v[1], &v[2]], pc);
            }
            return;
        }
        // Lane 1 must not read a register lane 0 writes (d + 0; a PTX register: either half,
        // since writing lane 0 replaces the whole register).
        let clash = lanes.iter().enumerate().any(|(i, l)| {
            l.iter().any(|v| match v {
                Val::Reg(r, _) if d.file == File::V => r.file == File::V && r.idx & !V_HI == d.idx,
                Val::Reg(r, _) => i == 1 && *r == d,
                _ => false,
            })
        });
        // Materialize every source first (lane 0's write must not change lane 1's sources).
        let mut r = [0u32; 2];
        for (i, ri) in r.iter_mut().enumerate() {
            *ri = fp::f32_fma(vals[0][i], vals[1][i], Some(vals[2][i]), Rnd::Rn, false);
        }
        let di = |i: usize| lane_reg(d, i);
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
        let na = na.unwrap_or_else(|| ns.leaf_for(t, a, raw, pc));
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
        // (register, byte offset of its word in the access or None: not one f32 word, the
        // register's own bytes for its taint or None: a high-half lane, no taint of its own).
        type Slot = (Reg, Option<u64>, Option<(u64, u64)>);
        let word = |r: Reg, w: Option<u64>| -> Slot {
            (r, w, Some(w.map_or((0, bytes as u64), |w| (w, 4))))
        };
        // 64-bit PTX registers: two f32 lanes (`lane_reg`), the register's taint over both.
        let pair = |i: u8| -> [Slot; 2] {
            let r = reg(i);
            let o = 8 * i as u64;
            [
                (r, Some(o), Some((o, 8))),
                (lane_reg(r, 1), Some(o + 4), None),
            ]
        };
        let regs: Vec<Slot> = if n > 1 {
            let es = bytes / n;
            if d.file == File::V && es == 8 {
                (0..n).flat_map(pair).collect()
            } else {
                (0..n)
                    .map(|i| word(reg(i), (es == 4).then_some(4 * i as u64)))
                    .collect()
            }
        } else if d.file == File::V {
            // One PTX register per access: 32-bit, or a 64-bit / 128-bit (two-slot) value.
            if bytes == 4 {
                vec![word(d, Some(0))]
            } else if bytes % 8 == 0 {
                (0..bytes / 8).flat_map(pair).collect()
            } else {
                vec![word(d, None)]
            }
        } else if bytes >= 4 {
            (0..bytes / 4)
                .map(|i| word(reg(i), Some(4 * i as u64)))
                .collect()
        } else {
            vec![word(d, None)]
        };
        for (r, w, own) in regs {
            // This register's own bytes (a whole-word register: its word; else the access).
            let (sp, a, off) = mem_key(loc);
            if let Some((lo, n)) = own {
                let rt = (lo..lo + n).any(|b| ns.taint.mem_has((sp, a, off + b)));
                ns.taint.ld_read.push((t, r, rt));
            }
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
                    || !off.is_multiple_of(4)
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
                ns.pred_src.get(&(t as u32, r)),
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
        old_input_derived: bool,
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
                let no = ns
                    .mem
                    .get(&k)
                    .copied()
                    .unwrap_or_else(|| ns.leaf_with_origin(o, pc, old_input_derived));
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

    pub(crate) fn num_store_origins(
        &self,
        t: usize,
        v: &Val,
        bytes: u8,
        guard: Option<Reg>,
    ) -> StoreOrigins {
        let Some(ns) = self.num.as_ref() else {
            return StoreOrigins {
                words: [false; 64],
                count: 0,
                guard: false,
                coverage: false,
            };
        };
        let guard = guard.is_some_and(|reg| ns.taint.reg_t(t, reg));
        let count = (bytes as usize).div_ceil(4);
        let mut words = [false; 64];
        for i in 0..(bytes as u16).div_ceil(4) {
            let source = match v {
                Val::Reg(r, n) if r.file == File::V && bytes / (*n).max(1) >= 8 => Some(Reg {
                    file: r.file,
                    idx: r.idx + i / 2,
                }),
                Val::Reg(r, n) if bytes > 8 || bytes > 4 && (r.file == File::R || *n >= 2) => {
                    Some(Reg {
                        file: r.file,
                        idx: r.idx + i,
                    })
                }
                Val::Reg(r, _) => Some(*r),
                _ => None,
            };
            words[i as usize] = ns.taint.store_origin(t, source, guard);
        }
        StoreOrigins {
            words,
            count,
            guard,
            coverage: ns.taint.cur_src || ns.taint.cur_read,
        }
    }

    pub(crate) fn num_st(
        &mut self,
        t: usize,
        v: &Val,
        loc: Loc,
        data: &[u8],
        origins: &StoreOrigins,
        pc: u32,
    ) {
        let bytes = data.len() as u8;
        if self.num.is_none() {
            return;
        }
        let (store_space, store_allocation, store_offset) = mem_key(loc);
        let partial = !store_offset.is_multiple_of(4) || !bytes.is_multiple_of(4);
        let partial_words = if partial {
            let memory = match loc {
                Loc::Global { alloc, .. } => &self.gmem[alloc],
                Loc::Shared { cta, .. } => &self.smem[cta as usize],
                Loc::Param { .. } => return,
            };
            let first = store_offset & !3;
            let end = store_offset + bytes as u64;
            (first..end)
                .step_by(4)
                .map(|off| {
                    let bits = u32::from_le_bytes(std::array::from_fn(|byte| {
                        let index = off as usize + byte;
                        if index < memory.bytes.len() {
                            *memory.bytes.get(index)
                        } else {
                            0
                        }
                    }));
                    (off, bits)
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        if origins.guard {
            self.half_mem_set(
                (store_space, store_allocation, store_offset),
                &vec![None; data.len()],
            );
        } else {
            self.half_on_st(t, v, loc, bytes);
        }
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
        // The register holding word i of a store of PTX registers with 64-bit elements
        // (`st.b64`, `st.v2.b64`, `st.b128`): lane i % 2 of element i / 2.
        let v_pair = |i: u16| -> Option<Reg> {
            match v {
                Val::Reg(r, n) if r.file == File::V && bytes / (*n).max(1) >= 8 => Some(lane_reg(
                    Reg {
                        file: File::V,
                        idx: r.idx + i / 2,
                    },
                    (i % 2) as usize,
                )),
                _ => None,
            }
        };
        if halves {
        } else if bytes == 4 {
            self.num_node_mat(t, v);
        } else if bytes.is_multiple_of(4) && v_pair(0).is_some() {
            // Lanes of PTX registers: only their nodes (no slices; num_ffma2_seg).
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
        if !ns.taint.position && store_space <= 1 {
            for (i, &input_derived) in origins.words[..origins.count].iter().enumerate() {
                if input_derived == origins.coverage {
                    continue;
                }
                let off = store_offset + 4 * i as u64;
                let end = (off + 4).min(store_offset + bytes as u64);
                ns.taint
                    .assign_memory(store_space, store_allocation, off, end, input_derived);
            }
        }
        if partial {
            for (off, bits) in partial_words {
                let key = (store_space, store_allocation, off);
                ns.note_write(key);
                ns.mem.remove(&key);
                if ns
                    .taint
                    .memory_intersects(store_space, store_allocation, off, off + 4)
                {
                    let node = ns.leaf_with_origin(bits, pc, true);
                    ns.mem.insert(key, node);
                }
            }
            return;
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
            let mut n = match v {
                _ if bytes.is_multiple_of(4) && v_pair(i).is_some() => {
                    ns.node_of_reg(t, v_pair(i).unwrap())
                }
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
            if origins.guard || n.is_none() && origins.words[i as usize] {
                let bits = u32::from_le_bytes(
                    data[4 * i as usize..4 * i as usize + 4]
                        .try_into()
                        .expect("complete store word"),
                );
                n = Some(ns.leaf_with_origin(bits, pc, true));
            }
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

    /// Segment mode, before a store whose guard depends on input data (thread `t`, executed or
    /// not): the words it covers and their nodes before it (`None`: input-derived without a
    /// node), and the slice of its guard predicate. None when not such a store, or when the
    /// store is outside what `num_gst_post` models (it then stays a taint violation).
    pub(crate) fn num_gst_pre(&mut self, t: usize, inst: &Inst, pc: u32) -> Option<GstPre> {
        let ns = self.num.as_ref()?;
        let g = inst.guard.reg?;
        if !ns.segments || ns.taint.position || !ns.taint.reg_t(t, g) {
            return None;
        }
        let Op::St {
            space,
            bytes,
            addr,
            v,
            ..
        } = &inst.op
        else {
            return None;
        };
        let words_ok = match v {
            Val::Reg(r, _) => r.file == File::R,
            Val::Imm(0) => true,
            _ => false,
        };
        if !words_ok || !bytes.is_multiple_of(4) || *bytes > 16 {
            return None;
        }
        let a = self.addr(t, addr).ok()?;
        let loc = self.resolve(t, *space, a, *bytes as usize).ok()?;
        let (sp, al, off) = mem_key(loc);
        if sp > 1 {
            return None;
        }
        let pred = self.seg_slice(t, g);
        let mut keys = Vec::new();
        let mut old = Vec::new();
        for i in 0..(*bytes / 4) as u64 {
            let k = (sp, al, off + 4 * i);
            let bits = word_at(self, loc, 4 * i);
            let ns = self.num.as_mut().unwrap();
            let n = match ns.mem.get(&k) {
                Some(&n) => Some(n),
                None if (0..4).any(|b| ns.taint.mem.contains(&(sp, al, off + 4 * i + b))) => None,
                None => Some(ns.const_leaf(bits, pc)),
            };
            keys.push(k);
            old.push(n);
        }
        Some(GstPre {
            keys,
            old,
            pred,
            neg: inst.guard.neg,
        })
    }

    /// After the store of `num_gst_pre`: word i becomes `Sel(P, new_i, old_i)` (the guard's
    /// predicate, the stored value's node, the node before), the same node whether or not this
    /// input executed it; a word whose value or old value is input-derived without a node, or
    /// whose predicate has no slice, loses its node and the guard stays a violation.
    pub(crate) fn num_gst_post(
        &mut self,
        t: usize,
        inst: &Inst,
        pc: u32,
        executed: bool,
        g: GstPre,
    ) {
        let Op::St { v, .. } = &inst.op else { return };
        let mut ok = true;
        let mut sels = Vec::new();
        for (i, (&k, &old)) in g.keys.iter().zip(&g.old).enumerate() {
            let w = match v {
                Val::Reg(r, _) => Val::Reg(
                    Reg {
                        file: r.file,
                        idx: r.idx + i as u16,
                    },
                    1,
                ),
                x => x.clone(),
            };
            let nv = match self.num_node_mat(t, &w) {
                Some(n) => Some(n),
                None => {
                    let ns = self.num.as_ref().unwrap();
                    if matches!(&w, Val::Reg(r, _) if ns.taint.reg_t(t, *r)) {
                        None
                    } else {
                        let bits = self.rd(t, &w).unwrap_or(0) as u32;
                        Some(self.num.as_mut().unwrap().const_leaf(bits, pc))
                    }
                }
            };
            sels.push((k, nv, old));
            ok &= nv.is_some() && old.is_some();
        }
        let ns = self.num.as_mut().unwrap();
        let pred = match g.pred {
            Ok(p) if ok => p,
            Ok(_) => {
                for (k, ..) in sels {
                    ns.mem.remove(&k);
                }
                ns.seg_fails.entry(pc).or_insert_with(|| {
                    "store under a data-dependent guard of an input-derived value without a node"
                        .into()
                });
                return;
            }
            Err(e) => {
                for (k, ..) in sels {
                    ns.mem.remove(&k);
                }
                ns.seg_fails
                    .entry(pc)
                    .or_insert_with(|| format!("guard predicate of a store: {e}"));
                return;
            }
        };
        let pred = SegRef(std::sync::Arc::new(pred));
        ns.seg_pcs.extend(pred.0.pcs());
        for (k, nv, old) in sels {
            let (a, b) = (nv.unwrap(), old.unwrap());
            let pick_a = executed;
            let src = if pick_a { a } else { b };
            let (hw, exact) = (
                ns.nodes[src as usize].hw,
                ns.nodes[src as usize].exact.clone(),
            );
            ns.nodes.push(Node {
                kind: NodeKind::Sel {
                    pred: pred.clone(),
                    neg: g.neg,
                    a,
                    b,
                },
                hw,
                exact,
                err: vec![],
                pc,
            });
            let id = (ns.nodes.len() - 1) as u32;
            ns.note_write(k);
            ns.mem.insert(k, id);
        }
        ns.sel_pcs.insert(pc);
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
            ns.taint.mem_set(k, tainted);
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
        if (0..4u64).any(|b| ns.taint.mem_has((4u8, t as u64, addr as u64 + b))) {
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
        old_input_derived: bool,
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
                    .unwrap_or_else(|| ns.leaf_with_origin(acc.to_bits(), pc, old_input_derived)),
            )
        } else {
            None
        };
        let id = ns.push_mma(acc_node, acc, prods, ops, r, pc, eps_of, scales, unit);
        ns.mem.insert(key, id);
    }

    /// tcgen05.mma scale-input-d reads accumulator cell (`lane`, `col`) as D * 2^-s
    /// (`scaled`). The numerics do not model the scaling: the cell becomes a new leaf of the
    /// scaled value (segment mode: `Opaque`), with a warning when its node carried rounding
    /// error, which is then no longer tracked.
    pub(crate) fn num_tmem_rescale(
        &mut self,
        cta: u16,
        lane: usize,
        col: usize,
        scaled: f32,
        pc: u32,
    ) {
        let Some(ns) = self.num.as_mut() else { return };
        let key = (3u8, cta as u64, crate::machine::tmem_idx(lane, col) as u64);
        if let Some(n) = ns.mem.get(&key).copied() {
            let node = &ns.nodes[n as usize];
            if !exact::is_zero(&exact::sub(
                &exact::from(f32::from_bits(node.hw) as f64),
                &node.exact,
            )) {
                ns.warnings.insert(format!(
                    "accumulator with rounding error scaled by tcgen05.mma scale-input-d at {} (not tracked)",
                    self.k.describe(pc as usize)
                ));
            }
        }
        let leaf = ns.leaf(scaled.to_bits(), pc);
        if ns.segments {
            ns.nodes[leaf as usize].kind = NodeKind::Opaque;
        }
        ns.mem.insert(key, leaf);
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
                fmax_branch(&self.k.code, b)
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

    /// What 16-bit half `h` (0 = low) of operand `v` holds as a bf16 value: a node (an input
    /// element, a rounding), a constant (an immediate or provably zero bytes), or unknown.
    fn bf16_half(&mut self, t: usize, v: &Val, h: u32) -> Bf16Half {
        if let Val::Imm(c) = v {
            return Bf16Half::Const((*c >> (16 * h)) as u16);
        }
        let Some(b) = self.half_val(t, v) else {
            return Bf16Half::Unknown;
        };
        let k = 2 * h as usize;
        match self.half_op(b[k], b[k + 1]) {
            crate::halfin::HalfOp::Zero => Bf16Half::Const(0),
            crate::halfin::HalfOp::Node(n) => Bf16Half::Node(n),
            crate::halfin::HalfOp::Unknown => Bf16Half::Unknown,
        }
    }

    /// One lane of a bf16 FMA (`Op::Hfma2Bf16`, the bf16 atomic adds) as numerics nodes: a
    /// `Round` (to bf16) of an f32 `Add` (b = 1.0: HADD2), `Fma` with the constant 1 (a
    /// negated sum) or `Fma` without addend (c = -0: HMUL2). These are the hardware's single
    /// rounding: a sum or product of two bf16 values rounded first to f32 and then to bf16 is
    /// the one rounding to bf16 (f32 has at least 2 x 8 + 2 bits; the product of two bf16 is
    /// exact in f32 down to where its bf16 rounding cannot change). The f32 intermediate is
    /// checked against the hardware lane `hw`. A general FMA (a nonzero addend) is not one of
    /// them (its f32 rounding can move a bf16 tie): None, as is any unknown operand.
    /// ASSUME(N-BF16X2): see ASSUMPTIONS.md
    fn bf16_lane_node(
        &mut self,
        ops: [Bf16Half; 3],
        neg: [bool; 3],
        hw: u16,
        pc: u32,
    ) -> Option<u32> {
        let ns = self.num.as_mut()?;
        if ops.contains(&Bf16Half::Unknown) {
            return None;
        }
        let bits = |o: Bf16Half, ns: &NumState| -> u32 {
            match o {
                Bf16Half::Node(n) => ns.nodes[n as usize].hw,
                Bf16Half::Const(c) => (c as u32) << 16,
                Bf16Half::Unknown => 0,
            }
        };
        let node = |o: Bf16Half, ns: &mut NumState| -> u32 {
            match o {
                Bf16Half::Node(n) => n,
                _ => {
                    let b = bits(o, ns);
                    ns.const_leaf(b, pc)
                }
            }
        };
        // Values after the operand sign flips.
        let v: Vec<u32> = (0..3)
            .map(|k| bits(ops[k], ns) ^ ((neg[k] as u32) << 31))
            .collect();
        let one = Bf16Half::Const(0x3f80);
        let (kind, r, ex, av) = if ops[1] == one && !neg[1] {
            let (na, nc) = (node(ops[0], ns), node(ops[2], ns));
            let r = fp::f32_fma(v[0], 0x3f80_0000, Some(v[2]), Rnd::Rn, false);
            let sg = |e: &Expansion, n: bool| if n { exact::neg(e) } else { e.clone() };
            let ex = exact::add(
                &sg(&ns.nodes[na as usize].exact, neg[0]),
                &sg(&ns.nodes[nc as usize].exact, neg[2]),
            );
            let av = exact::add(
                &exact::from(f32::from_bits(v[0]) as f64),
                &exact::from(f32::from_bits(v[2]) as f64),
            );
            let kind = if !neg[0] && !neg[2] {
                NodeKind::Add(na, nc, false)
            } else {
                let n1 = ns.const_leaf(0x3f80_0000, pc);
                NodeKind::Fma {
                    a: na,
                    b: n1,
                    c: Some(nc),
                    neg_ab: neg[0],
                    neg_c: neg[2],
                }
            };
            (kind, r, ex, av)
        } else if bits(ops[2], ns) ^ ((neg[2] as u32) << 31) == 0x8000_0000 {
            let (na, nb) = (node(ops[0], ns), node(ops[1], ns));
            let r = fp::f32_fma(v[0], v[1], None, Rnd::Rn, false);
            let mut ex = exact::mul(&ns.nodes[na as usize].exact, &ns.nodes[nb as usize].exact);
            if neg[0] != neg[1] {
                ex = exact::neg(&ex);
            }
            let av = exact::mul(
                &exact::from(f32::from_bits(v[0]) as f64),
                &exact::from(f32::from_bits(v[1]) as f64),
            );
            let kind = NodeKind::Fma {
                a: na,
                b: nb,
                c: None,
                neg_ab: neg[0] != neg[1],
                neg_c: false,
            };
            (kind, r, ex, av)
        } else {
            return None;
        };
        if fp::f32_to_bf16_rn_gpu(r) != hw {
            ns.warnings.insert(format!(
                "bf16 lane at pc {pc}: the f32 intermediate does not round to the hardware result"
            ));
            return None;
        }
        let err = exact::sub(&exact::from(f32::from_bits(r) as f64), &av);
        ns.nodes.push(Node {
            kind,
            hw: r,
            exact: ex,
            err,
            pc,
        });
        let f = (ns.nodes.len() - 1) as u32;
        let wide = (hw as u32) << 16;
        ns.nodes.push(Node {
            kind: NodeKind::Round { arg: f, f16: false },
            hw: wide,
            exact: ns.nodes[f as usize].exact.clone(),
            err: exact::sub(
                &exact::from(f32::from_bits(wide) as f64),
                &exact::from(f32::from_bits(r) as f64),
            ),
            pc,
        });
        Some((ns.nodes.len() - 1) as u32)
    }

    /// `Op::Hfma2Bf16` wrote `d` = `r` from operand words `v`: each lane a `bf16_lane_node`
    /// (its bytes `N{node}`), or unknown bytes (with `num_clobber`'s warning for operands that
    /// carried rounding error). `.RELU` lanes are not followed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn num_hfma2_bf16(
        &mut self,
        t: usize,
        d: Reg,
        ops: [&Val; 3],
        _v: [u32; 3],
        (sel, neg, relu): ([HSel; 3], [bool; 3], bool),
        r: u32,
        pc: u32,
    ) {
        if self.num.is_none() {
            return;
        }
        let mut bytes: crate::halfin::Bytes = [None; 4];
        let mut lost = false;
        for lane in 0..2u32 {
            let mut h = [Bf16Half::Unknown; 3];
            for k in 0..3 {
                let half = match sel[k] {
                    HSel::Pair => lane,
                    HSel::H0 => 0,
                    HSel::H1 => 1,
                };
                h[k] = self.bf16_half(t, ops[k], half);
            }
            let n = if relu {
                None
            } else {
                self.bf16_lane_node(h, neg, (r >> (16 * lane)) as u16, pc)
            };
            match n {
                Some(n) => {
                    bytes[2 * lane as usize] = Some(crate::halfin::Src::N { node: n, b: 0 });
                    bytes[2 * lane as usize + 1] = Some(crate::halfin::Src::N { node: n, b: 1 });
                }
                None => lost = true,
            }
        }
        if lost {
            let srcs: Vec<&Val> = ops.to_vec();
            self.num_clobber(t, d, &srcs, pc);
        } else {
            self.num.as_mut().unwrap().set(t, d, None);
        }
        self.half_set_reg(t, d, Some(bytes));
    }

    /// `Op::FfmaBf16` wrote `d` = `r` (f32): with round to nearest and no saturation, an f32
    /// `Add` (b = 1.0, no sign flips: FHADD) or `Fma` node of the bf16 halves' nodes (input
    /// elements, roundings, constants) and c's node; else, or with an unknown half, as
    /// `num_clobber`. ASSUME(N-BF16X2): see ASSUMPTIONS.md
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn num_ffma_bf16(
        &mut self,
        t: usize,
        d: Reg,
        ops: [&Val; 3],
        v: [u32; 3],
        (hi, neg, rnd, sat): ([bool; 2], [bool; 3], Rnd, bool),
        r: u32,
        pc: u32,
    ) {
        if self.num.is_none() {
            return;
        }
        let ha = self.bf16_half(t, ops[0], hi[0] as u32);
        let hb = self.bf16_half(t, ops[1], hi[1] as u32);
        if rnd != Rnd::Rn || sat || ha == Bf16Half::Unknown || hb == Bf16Half::Unknown {
            self.num_clobber(t, d, &ops, pc);
            return;
        }
        let nc = self.num_node_mat(t, ops[2]);
        let ns = self.num.as_mut().unwrap();
        let node = |h: Bf16Half, ns: &mut NumState| match h {
            Bf16Half::Node(n) => n,
            Bf16Half::Const(c) => ns.const_leaf((c as u32) << 16, pc),
            Bf16Half::Unknown => unreachable!(),
        };
        let (na, nb) = (node(ha, ns), node(hb, ns));
        let nc = nc.unwrap_or_else(|| ns.leaf_for(t, ops[2], v[2], pc));
        let bits = |n: u32, ng: bool, ns: &NumState| ns.nodes[n as usize].hw ^ ((ng as u32) << 31);
        let (x, y, z) = (
            bits(na, neg[0], ns),
            bits(nb, neg[1], ns),
            bits(nc, neg[2], ns),
        );
        let sg = |e: &Expansion, n: bool| if n { exact::neg(e) } else { e.clone() };
        let ex = exact::add(
            &sg(
                &exact::mul(&ns.nodes[na as usize].exact, &ns.nodes[nb as usize].exact),
                neg[0] != neg[1],
            ),
            &sg(&ns.nodes[nc as usize].exact, neg[2]),
        );
        let av = exact::add(
            &exact::mul(
                &exact::from(f32::from_bits(x) as f64),
                &exact::from(f32::from_bits(y) as f64),
            ),
            &exact::from(f32::from_bits(z) as f64),
        );
        let add = hb == Bf16Half::Const(0x3f80) && !neg.iter().any(|&n| n);
        let kind = if add {
            NodeKind::Add(na, nc, false)
        } else {
            NodeKind::Fma {
                a: na,
                b: nb,
                c: Some(nc),
                neg_ab: neg[0] != neg[1],
                neg_c: neg[2],
            }
        };
        let err = exact::sub(&exact::from(f32::from_bits(r) as f64), &av);
        ns.nodes.push(Node {
            kind,
            hw: r,
            exact: ex,
            err,
            pc,
        });
        let id = (ns.nodes.len() - 1) as u32;
        ns.set(t, d, Some(id));
    }

    /// A bf16 atomic add (`AtomOp::AddBf16x2` / `AddBf16`, `bytes` 4 / 2) at `loc` of operand
    /// `v` (bits `x`) to the old value `o`, leaving `n`: each lane a `bf16_lane_node` of the
    /// location's old bytes and the operand's half, written as the location's byte provenance
    /// (unknown when untracked); `d` gets the old bytes. One schedule's order: certification
    /// refuses unordered bf16 adds of one location (C-ORDER).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn num_atom_bf16(
        &mut self,
        t: usize,
        v: &Val,
        loc: Loc,
        bytes: u8,
        n: u32,
        d: Option<Reg>,
        pc: u32,
    ) {
        let Some(ns) = self.num.as_mut() else { return };
        let k = mem_key(loc);
        ns.note_write(k);
        ns.other_atomics += 1;
        ns.mem.remove(&k);
        let old = self.half_mem(loc, bytes as usize, None);
        let mut out: Vec<Option<crate::halfin::Src>> = vec![None; bytes as usize];
        for lane in 0..(bytes as u32 / 2) {
            let i = 2 * lane as usize;
            let a = match self.half_op(old[i], old[i + 1]) {
                crate::halfin::HalfOp::Zero => Bf16Half::Const(0),
                crate::halfin::HalfOp::Node(n) => Bf16Half::Node(n),
                crate::halfin::HalfOp::Unknown => Bf16Half::Unknown,
            };
            let c = self.bf16_half(t, v, lane);
            let node = self.bf16_lane_node(
                [a, Bf16Half::Const(0x3f80), c],
                [false; 3],
                (n >> (16 * lane)) as u16,
                pc,
            );
            if let Some(nd) = node {
                out[i] = Some(crate::halfin::Src::N { node: nd, b: 0 });
                out[i + 1] = Some(crate::halfin::Src::N { node: nd, b: 1 });
            }
        }
        self.half_mem_set(crate::halfin::hkey(loc), &out);
        if let Some(d) = d {
            self.num.as_mut().unwrap().set(t, d, None);
            let mut b: crate::halfin::Bytes = [None; 4];
            for (i, s) in old.iter().enumerate().take(4) {
                b[i] = *s;
            }
            if bytes == 2 {
                b[2] = Some(crate::halfin::Src::Zero);
                b[3] = Some(crate::halfin::Src::Zero);
            }
            self.half_set_reg(t, d, Some(b));
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

    /// One word of a bulk f32 add-reduction: global `to` (old bits o) += shared `from` (bits
    /// x), result r. As `num_atom`'s atomic add, with the shared word's node (or a leaf) as the
    /// addend.
    pub(crate) fn num_bulk_add(
        &mut self,
        from: Loc,
        to: Loc,
        (o, x, r): (u32, u32, u32),
        old_input_derived: bool,
        pc: u32,
    ) {
        let Some(ns) = self.num.as_mut() else { return };
        let k = mem_key(to);
        ns.note_write(k);
        let no = ns
            .mem
            .get(&k)
            .copied()
            .unwrap_or_else(|| ns.leaf_with_origin(o, pc, old_input_derived));
        let source_input_derived = ns.taint.mem_has(mem_key(from));
        let nx = ns
            .mem
            .get(&mem_key(from))
            .copied()
            .unwrap_or_else(|| ns.leaf_with_origin(x, pc, source_input_derived));
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

    /// A word written with a value the graph does not track: it loses its node.
    pub(crate) fn num_forget(&mut self, loc: Loc) {
        let Some(ns) = self.num.as_mut() else { return };
        let (space, allocation, off) = mem_key(loc);
        let key = (space, allocation, off & !3);
        ns.note_write(key);
        ns.mem.remove(&key);
    }

    pub(crate) fn num_read_snapshot(
        &mut self,
        from: Option<Loc>,
        data: &[u8],
        pc: u32,
    ) -> Option<ReadSnapshot> {
        let ns = self.num.as_ref()?;
        let half = if ns.half.is_on() {
            match from {
                Some(loc) => self.half_mem(loc, data.len(), Some(data)),
                None => vec![Some(crate::halfin::Src::Zero); data.len()],
            }
        } else {
            Vec::new()
        };
        let launch = self.launch.clone();
        let ns = self.num.as_mut().unwrap();
        let origins = match from {
            Some(loc) => {
                let (space, allocation, off) = mem_key(loc);
                ns.taint
                    .capture_origins((space, allocation), off, data.len())
            }
            None => vec![false; data.len()],
        };
        let mut nodes = Vec::with_capacity(data.len() / 4);
        for word in 0..data.len() / 4 {
            let bits = u32::from_le_bytes(
                data[4 * word..4 * word + 4]
                    .try_into()
                    .expect("complete copy word"),
            );
            let mut node = from.and_then(|loc| {
                let (space, allocation, off) = mem_key(loc);
                let off = off + 4 * word as u64;
                let key = (space, allocation, off);
                if let Some(&node) = ns.mem.get(&key) {
                    return Some(node);
                }
                let alloc = allocation as usize;
                let init = launch
                    .allocs
                    .get(alloc)?
                    .init
                    .get(off as usize..off as usize + 4)?;
                if space != 1
                    || !off.is_multiple_of(4)
                    || !ns.f32_inputs.contains(&alloc)
                    || ns.written.contains(&(allocation, off))
                    || init != bits.to_le_bytes()
                {
                    return None;
                }
                ns.nodes.push(Node {
                    kind: NodeKind::Input((alloc, off / 4)),
                    hw: bits,
                    exact: exact::from(f32::from_bits(bits) as f64),
                    err: vec![],
                    pc,
                });
                let node = (ns.nodes.len() - 1) as u32;
                ns.mem.insert(key, node);
                Some(node)
            });
            if node.is_none() && origins[4 * word..4 * word + 4].iter().any(|&origin| origin) {
                node = Some(ns.leaf_with_origin(bits, pc, true));
            }
            nodes.push(node);
        }
        Some(ReadSnapshot {
            origins,
            nodes,
            half,
        })
    }

    pub(crate) fn num_copy_snapshot(
        &mut self,
        to: Loc,
        data: &[u8],
        snapshot: Option<&ReadSnapshot>,
        pc: u32,
    ) {
        if self.num.is_none() {
            return;
        }
        let (space, allocation, off) = mem_key(to);
        let partial = !off.is_multiple_of(4) || !data.len().is_multiple_of(4);
        let partial_words = if partial {
            let memory = match to {
                Loc::Global { alloc, .. } => &self.gmem[alloc],
                Loc::Shared { cta, .. } => &self.smem[cta as usize],
                Loc::Param { .. } => return,
            };
            (off & !3..off + data.len() as u64)
                .step_by(4)
                .map(|offset| {
                    let bits = u32::from_le_bytes(std::array::from_fn(|byte| {
                        let index = offset as usize + byte;
                        if index < memory.bytes.len() {
                            *memory.bytes.get(index)
                        } else {
                            0
                        }
                    }));
                    (offset, bits)
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        if let Some(snapshot) = snapshot {
            if !snapshot.half.is_empty() {
                self.half_mem_set((space, allocation, off), &snapshot.half);
            }
        }
        let ns = self.num.as_mut().unwrap();
        match snapshot {
            Some(snapshot) => ns
                .taint
                .restore_origins((space, allocation), off, &snapshot.origins),
            None => ns
                .taint
                .assign_memory(space, allocation, off, off + data.len() as u64, true),
        }
        if partial {
            for (offset, bits) in partial_words {
                let key = (space, allocation, offset);
                ns.note_write(key);
                ns.mem.remove(&key);
                if ns
                    .taint
                    .memory_intersects(space, allocation, offset, offset + 4)
                {
                    let node = ns.leaf_with_origin(bits, pc, true);
                    ns.mem.insert(key, node);
                }
            }
        } else {
            for word in 0..data.len() / 4 {
                let key = (space, allocation, off + 4 * word as u64);
                ns.note_write(key);
                let node = snapshot
                    .and_then(|snapshot| snapshot.nodes[word])
                    .or_else(|| {
                        if snapshot.is_none() {
                            let bits = u32::from_le_bytes(
                                data[4 * word..4 * word + 4]
                                    .try_into()
                                    .expect("complete copy word"),
                            );
                            Some(ns.leaf_with_origin(bits, pc, true))
                        } else {
                            None
                        }
                    });
                if let Some(node) = node {
                    ns.mem.insert(key, node);
                } else {
                    ns.mem.remove(&key);
                }
            }
        }
    }

    /// A TMA load wrote the shared-memory element at `off` from global element `src`
    /// (None: out of bounds, zero-filled).
    pub(crate) fn num_tma_elem(&mut self, cta: u16, off: u32, src: Option<(usize, u64)>) {
        let Some(ns) = self.num.as_mut() else { return };
        ns.mem.remove(&mem_key(Loc::Shared { cta, off: off & !3 }));
        match src.filter(|(a, element)| {
            !ns.not_inputs.contains(a) && !ns.written.contains(&(*a as u64, (element * 2) & !3))
        }) {
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
                | NodeKind::Sel { .. }
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
pub(crate) fn pure_op(op: &Op) -> bool {
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
            | Op::MatchAny { .. }
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
            | Op::Hfma2Bf16 { .. }
            | Op::FfmaBf16 { .. }
            | Op::Hmnmx2Bf16 { .. }
            | Op::Vhmnmx { .. }
            | Op::CvtF32Bf16 { .. }
            | Op::Cvta { .. }
            | Op::Bfe { .. }
            | Op::Bfi { .. }
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
            | Op::CvtTf32 { .. }
            | Op::CvtF32F16 { .. }
            | Op::CvtPack { .. }
            | Op::CvtRsX4 { .. }
            | Op::CvtUnpack { .. }
            | Op::CvtS2f6x2 { .. }
            | Op::Nop
    )
}

/// The value operands of a memory / synchronization operation: the data it moves. Every
/// other register it reads (addresses, coordinates, descriptors, counts, lane indices) decides
/// where or whether something happens.
/// Position mode: registers of `op` that may depend on the CTA's position (global addresses,
/// TMA coordinates), and whether they are a generic address (must resolve to global).
pub(crate) fn pos_free_regs(op: &Op) -> (Vec<Reg>, bool) {
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
        Op::Ld256 { addr, .. } | Op::St256 { addr, .. } => {
            addr.base.iter().chain(addr.base2.iter()).for_each(&mut v);
        }
        Op::LdReduce { addr, .. } => addr.base.iter().chain(addr.base2.iter()).for_each(&mut v),
        Op::TmaLoad { coords, .. } | Op::TmaStore { coords, .. } => coords.iter().for_each(&mut v),
        _ => {}
    }
    (out, generic)
}

fn value_vals(op: &Op) -> Vec<Val> {
    match op {
        Op::St { v, .. } | Op::Atom { v, .. } | Op::StAsync { v, .. } | Op::StLocal { v, .. } => {
            vec![v.clone()]
        }
        Op::St256 { lo, hi, .. } => vec![Val::Reg(*lo, 4), Val::Reg(*hi, 4)],
        Op::Shfl { a, .. } => vec![a.clone()],
        Op::Hmma { a, b, c, .. } => a.iter().chain(b).chain(c).cloned().collect(),
        Op::StMatrix { v, .. } | Op::TmemSt { v, .. } => v.clone(),
        _ => vec![],
    }
}

fn operand_regs(values: &[Val]) -> Vec<Reg> {
    values
        .iter()
        .flat_map(|value| match value {
            Val::Reg(r, slots) => (0..if r.file == File::V {
                1
            } else {
                (*slots).max(1)
            })
                .map(|i| Reg {
                    file: r.file,
                    idx: r.idx + i as u16,
                })
                .collect(),
            _ => Vec::new(),
        })
        .collect()
}

pub(crate) fn value_regs(op: &Op) -> Vec<Reg> {
    let mut out = operand_regs(&value_vals(op));
    if let Some((r, k)) = op.vector_src() {
        if !matches!(op, Op::ClcQuery { .. }) {
            out.extend((1..k).map(|i| Reg {
                file: r.file,
                idx: r.idx + i,
            }));
        }
    }
    out
}

fn control_regs(op: &Op) -> Vec<Reg> {
    let mut operands = op.src_vals();
    for value in value_vals(op) {
        if let Some(index) = operands.iter().position(|operand| *operand == value) {
            operands.remove(index);
        }
    }
    operand_regs(&operands)
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
    /// Every global byte is a source (E-CTA-REP: whatever a CTA loads from global memory
    /// may differ between CTAs) until written with an untainted value (`clean_global`).
    pub all_global: bool,
    clean_global: std::collections::HashSet<(u64, u64)>,
}

impl Taint {
    fn memory_intersects(&self, space: u8, allocation: u64, start: u64, end: u64) -> bool {
        (start..end).any(|off| self.mem_has((space, allocation, off)))
    }

    fn assign_memory(&mut self, space: u8, allocation: u64, start: u64, end: u64, origin: bool) {
        for off in start..end {
            let key = (space, allocation, off);
            self.mem_set(key, origin);
        }
    }

    fn capture_origins(&self, key: (u8, u64), off: u64, count: usize) -> Vec<bool> {
        (0..count)
            .map(|byte| self.memory_unit_input_derived((key.0, key.1, off + byte as u64)))
            .collect()
    }

    fn restore_origins(&mut self, key: (u8, u64), off: u64, origins: &[bool]) {
        for (byte, &origin) in origins.iter().enumerate() {
            let offset = off + byte as u64;
            self.assign_memory(key.0, key.1, offset, offset + 1, origin);
        }
    }

    fn store_origin(&self, thread: usize, source: Option<Reg>, guard_input_derived: bool) -> bool {
        guard_input_derived || source.is_some_and(|source| self.reg_t(thread, source))
    }

    pub fn last_read_input_derived(&self) -> bool {
        self.last_read
    }

    pub fn memory_unit_input_derived(&self, key: (u8, u64, u64)) -> bool {
        self.mem_has(key)
    }

    pub fn taint_alloc(&mut self, alloc: usize, bytes: usize) {
        for b in 0..bytes as u64 {
            self.mem.insert((1, alloc as u64, b));
        }
    }
    fn reg_t(&self, t: usize, r: Reg) -> bool {
        // A PTX register's high half (V_HI) has the whole register's taint.
        let idx = if is_v_hi(r) { r.idx & !V_HI } else { r.idx };
        self.reg.contains(&(t as u32, r.file as u8, idx))
    }
    pub(crate) fn reg_tainted(&self, t: usize, r: Reg) -> bool {
        self.reg_t(t, r)
    }
    /// Whether memory byte (space, object, offset) is tainted (`all_global`: every global
    /// byte not since written untainted).
    pub(crate) fn mem_has(&self, k: (u8, u64, u64)) -> bool {
        self.mem.contains(&k)
            || self.all_global && k.0 == 1 && !self.clean_global.contains(&(k.1, k.2))
    }
    pub(crate) fn mem_set(&mut self, k: (u8, u64, u64), tainted: bool) {
        if tainted {
            self.mem.insert(k);
        } else {
            self.mem.remove(&k);
        }
        if self.all_global && k.0 == 1 {
            if tainted {
                self.clean_global.remove(&(k.1, k.2));
            } else {
                self.clean_global.insert((k.1, k.2));
            }
        }
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
        self.last_read = (off..off + n).any(|b| self.mem_has((key.0, key.1, b)));
        if self.last_read {
            self.cur_read = true;
        }
    }
    /// Whether memory byte (space, allocation, offset) holds input-derived data.
    pub fn byte_tainted(&self, k: (u8, u64, u64)) -> bool {
        self.mem.contains(&k)
    }

    pub(crate) fn write(&mut self, key: (u8, u64), off: u64, n: u64) {
        self.check_gen(key);
        let t = self.cur_src || self.cur_read;
        for b in off..off + n {
            self.mem_set((key.0, key.1, b), t);
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
        let ctl: Vec<Reg> = control_regs(&inst.op)
            .into_iter()
            .filter(|r| !free.0.contains(r))
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
                if fmax_branch(&k.code, pc as usize).is_some() {
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
        tn.cur_src = seed
            || active.iter().any(|&t| {
                let sources = if position { &vals } else { &srcs };
                sources.iter().any(|&r| tn.reg_t(t, r))
                    || inst.guard.reg.is_some_and(|guard| tn.reg_t(t, guard))
            });
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
        let data_guard = inst
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
                Op::VoteAny { .. } | Op::VoteAll { .. } | Op::Redux { .. } | Op::MatchAny { .. }
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
                if dg.contains(&t) {
                    ns.half.forget_register(t, r);
                }
                if dg.contains(&t) || !ns.touched.contains(&k) {
                    ns.reg.remove(&k);
                }
                // The high half of a PTX register (V_HI) too.
                if r.file == File::V {
                    let h = (t as u32, file_tag(r.file), r.idx | V_HI);
                    if dg.contains(&t) || !ns.touched.contains(&h) {
                        ns.reg.remove(&h);
                    }
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
                let data_guard = gt;
                if !runs && !data_guard {
                    // Its own guard is false for every input: the register is unchanged (the
                    // blanket rule above tainted every member's destination).
                    tn.set_mask(t, *d, masks[3]);
                    continue;
                }
                let (mut m, one) = lop3_taint(*lut, vals, masks, data_guard);
                if data_guard && d.file == File::V {
                    m = u32::MAX;
                }
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

#[cfg(test)]
mod fmax_seq_tests {
    use super::fmax_branch;
    use super::fmax_seq;

    fn kernel(body: &str) -> ptxv_ir::Kernel {
        // Each instruction with a zero control word (`nvdisasm -hex` layout).
        let mut text = String::from(".text.k:\n");
        for l in body.lines() {
            text.push_str(l);
            text.push('\n');
            if l.trim_start().starts_with("/*") {
                text.push_str("        /* 0x000fc00000000000 */\n");
            }
        }
        ptxv_sass::lower(&text, "k", &ptxv_sass::CubinMeta::default()).expect("lower")
    }

    /// CuTeDSL's `smem_fmax` (FA4 decode, sass 0x3e40..0x3ec0): two skips on one sign
    /// predicate, the unsigned min where the sign bit is set, the signed max where it is clear,
    /// with a BSYNC between them. Both branches are recognized; swapping the atomics (the max
    /// where the sign bit is set) or reading another value is not.
    #[test]
    fn sequential_float_atomic_max() {
        let good = "
        /*0000*/                   ISETP.GE.AND P1, PT, R4, RZ, PT ;
        /*0010*/               @P1 BRA `(.L_x_58) ;
        /*0020*/                   ATOMS.MIN RZ, [R23+0x10], R4 ;
.L_x_58:
        /*0030*/                   NOP ;
        /*0040*/                   BSYNC.RECONVERGENT B0 ;
        /*0050*/              @!P1 BRA `(.L_x_56) ;
        /*0060*/                   ATOMS.MAX.S32 RZ, [R23+0x10], R4 ;
.L_x_56:
        /*0070*/                   NOP ;
        /*0080*/                   EXIT ;
";
        let k = kernel(good);
        assert_eq!(fmax_seq(&k.code, 1), Some((6, 2)));
        assert_eq!(fmax_seq(&k.code, 5), Some((6, 2)));
        assert_eq!(fmax_branch(&k.code, 5), Some((6, 2)));
        let swapped = good
            .replace("ATOMS.MIN RZ", "ATOMS.TMP RZ")
            .replace("ATOMS.MAX.S32 RZ", "ATOMS.MIN RZ")
            .replace("ATOMS.TMP RZ", "ATOMS.MAX.S32 RZ");
        let k = kernel(&swapped);
        assert_eq!(fmax_seq(&k.code, 1), None);
        assert_eq!(fmax_seq(&k.code, 5), None);
        let other = good.replace(
            "ATOMS.MAX.S32 RZ, [R23+0x10], R4",
            "ATOMS.MAX.S32 RZ, [R23+0x10], R5",
        );
        let k = kernel(&other);
        assert_eq!(fmax_seq(&k.code, 1), None);
        let rewritten = good.replace(
            "        /*0040*/                   BSYNC.RECONVERGENT B0 ;",
            "        /*0040*/                   MOV R4, RZ ;",
        );
        let k = kernel(&rewritten);
        assert_eq!(fmax_seq(&k.code, 1), None);
    }
}

#[cfg(test)]
mod untracked_producer_tests {
    use std::sync::Arc;

    use super::*;
    use crate::equiv::Side;
    use crate::equiv::compare;

    fn run_body_ir(body: &str, input: f64, use_lop3_ir: bool) -> (NumState, (u32, Option<u32>)) {
        run_body_finish_ir(
            body,
            input,
            use_lop3_ir,
            "add.rn.f32 %f1, %f0, 0f00000000; st.global.f32 [%p1], %f1;",
        )
    }

    pub(super) fn run_body_finish_ir(
        body: &str,
        input: f64,
        use_lop3_ir: bool,
        finish: &str,
    ) -> (NumState, (u32, Option<u32>)) {
        let mut machine = run_machine_body_finish_ir(body, input, use_lop3_ir, finish, false);
        let word = u32::from_le_bytes(std::array::from_fn(|byte| *machine.gmem[1].bytes.get(byte)));
        let state = *machine.num.take().unwrap();
        let node = state.mem.get(&(1, 1, 0)).copied();
        if let Some(node) = node {
            assert_eq!(state.nodes[node as usize].hw, word);
        }
        (state, (word, node))
    }

    fn run_machine_body_finish_ir(
        body: &str,
        input: f64,
        use_lop3_ir: bool,
        finish: &str,
        half_inputs: bool,
    ) -> Machine {
        run_machine_body_transform(body, input, use_lop3_ir, finish, half_inputs, false, None)
    }

    fn run_machine_body_transform(
        body: &str,
        input: f64,
        use_lop3_ir: bool,
        finish: &str,
        half_inputs: bool,
        segments: bool,
        transform: Option<fn(&mut ptxv_ir::Kernel)>,
    ) -> Machine {
        let ptx = format!(
            r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 input, .param .u64 output) {{
  .reg .b64 %p<2>;
  .reg .b32 %r<2>;
  .reg .f32 %f<2>;
  .reg .f64 %fd<2>;
  .reg .pred %q;
  .reg .b64 %w;
  ld.param.u64 %p0, [input];
  ld.param.u64 %p1, [output];
  ld.global.u32 %r0, [%p0];
  {body}
  {finish}
  ret;
}}
"#
        );
        let mut kernel = ptxv_ptx::parse_kernel(&ptx, Some("k")).unwrap();
        if let Some(transform) = transform {
            transform(&mut kernel);
        }
        if use_lop3_ir {
            // Exercise the SASS LOP3 path as well as PTX AND.
            for instruction in &mut kernel.code {
                if let Op::Int {
                    op: IntOp::And,
                    d,
                    a: Val::Imm(0),
                    b: Val::Imm(0),
                    ..
                } = instruction.op
                {
                    instruction.op = Op::Lop3 {
                        d,
                        a: Val::Imm(0),
                        b: Val::Imm(0),
                        c: Val::Imm(0),
                        lut: 0,
                        p: None,
                        pq: None,
                    };
                }
            }
        }
        let kernel = Arc::new(kernel);
        let manifest = serde_json::from_value(serde_json::json!({
            "name": "untracked-zero", "description": "missing integer history",
            "entry": "k", "grid": [1,1,1], "block": [1,1,1], "outputs": ["out"],
            "buffers": [
                {"name":"x", "dtype":if half_inputs { "bf16" } else { "f32" }, "shape":[if half_inputs || transform.is_some() { 2 } else { 1 }], "init":"zeros", "values":if transform.is_some() { vec![(0u64,input),(1u64,2.0*input)] } else { vec![(0u64,input)] }},
                {"name":"out", "dtype":"f32", "shape":[1], "init":"zeros"}
            ],
            "params": [{"kind":"ptr","buffer":"x"}, {"kind":"ptr","buffer":"out"}],
            "variants": []
        })).unwrap();
        let case = ptxv_case::Case {
            dir: std::path::PathBuf::new(),
            m: manifest,
        };
        let launch = Arc::new(crate::build_launch(&case, &kernel).unwrap());
        let mut machine = Machine::new(kernel, launch);
        let mut state = NumState {
            segments,
            ..Default::default()
        };
        state.not_inputs.insert(1);
        if half_inputs {
            state.half.inputs.insert(0, crate::halfin::Half::Bf16);
        } else {
            state.f32_inputs.insert(0);
        }
        state
            .taint
            .taint_alloc(0, if transform.is_some() { 8 } else { 4 });
        machine.num = Some(Box::new(state));
        let (machine, findings, completed) = crate::explore::run_once(machine);
        assert!(completed, "{findings:?}");
        assert!(findings.is_empty(), "{findings:?}");
        machine
    }

    fn run_body(body: &str, input: f64) -> (NumState, (u32, Option<u32>)) {
        run_body_ir(body, input, false)
    }

    fn shifted_input(shift: u8, input: f64) -> (NumState, (u32, Option<u32>)) {
        run_body(
            &format!("shr.u32 %r1, %r0, {shift}; mov.b32 %f0, %r1;"),
            input,
        )
    }

    #[test]
    fn distinct_input_computations_at_zero_stay_untracked() {
        let (reference, routput) = shifted_input(1, 0.0);
        let (candidate, coutput) = shifted_input(2, 0.0);
        assert_eq!(routput.0, 0);
        assert_eq!(coutput.0, 0);
        for state in [&reference, &candidate] {
            assert!(
                state
                    .nodes
                    .iter()
                    .any(|node| matches!(node.kind, NodeKind::Opaque))
            );
        }
        let report = compare(
            &Side {
                ns: &reference,
                outs: vec![routput],
                buf_id: vec![0, 1],
            },
            &Side {
                ns: &candidate,
                outs: vec![coutput],
                buf_id: vec![0, 1],
            },
            &["x".into(), "out".into()],
            None,
        );
        assert!(!report.levels[0].holds);
        assert!(!report.levels[1].holds);
        let (_, rwitness) = shifted_input(1, 1.0);
        let (_, cwitness) = shifted_input(2, 1.0);
        assert_ne!(rwitness.0, cwitness.0);
    }

    #[test]
    fn initial_output_zero_keeps_a_constant_atomic_base() {
        let body = "mov.b32 %f0, %r0; red.global.add.f32 [%p1], %f0; ld.global.f32 %f0, [%p1];";
        let (state, _) = run_body(body, 0.0);
        let base = state
            .nodes
            .iter()
            .find_map(|node| match node.kind {
                NodeKind::AtomAdd(base, _) => Some(base),
                _ => None,
            })
            .unwrap();
        assert!(matches!(state.nodes[base as usize].kind, NodeKind::Leaf));
        assert_eq!(state.nodes[base as usize].hw, 0);
        assert!(
            !state
                .nodes
                .iter()
                .any(|node| matches!(node.kind, NodeKind::Opaque))
        );
    }

    #[test]
    fn input_derived_old_atomic_zero_remains_opaque() {
        let body = "shr.u32 %r1, %r0, 1; mov.b32 %f0, %r1; st.global.f32 [%p1], %f0; \
            mov.b32 %f0, %r0; red.global.add.f32 [%p1], %f0; ld.global.f32 %f0, [%p1];";
        let (state, output) = run_body(body, 0.0);
        assert_eq!(output.0, 0);
        let base = state
            .nodes
            .iter()
            .find_map(|node| match node.kind {
                NodeKind::AtomAdd(base, _) => Some(base),
                _ => None,
            })
            .unwrap();
        assert!(matches!(state.nodes[base as usize].kind, NodeKind::Opaque));
    }
    #[test]
    fn inactive_input_guard_preserves_possible_old_values() {
        let body = "mov.b32 %f0, %r0; setp.ne.f32 %q, %f0, 0f00000000; \
            mov.b32 %r1, 0; @%q mov.b32 %r1, 0x3f800000; mov.b32 %f0, %r1;";
        let (state, output) = run_body(body, 0.0);
        assert_eq!(output.0, 0);
        assert!(
            state
                .nodes
                .iter()
                .any(|node| matches!(node.kind, NodeKind::Opaque))
        );
        assert!(!state.taint.violations.is_empty());
        let (_, witness) = run_body(body, 1.0);
        assert_eq!(witness.0, 1.0f32.to_bits());
    }

    #[test]
    fn unclassified_float_arithmetic_cannot_clear_input_taint() {
        let body = "mov.b32 %f0, %r0; add.rn.f32 %f1, %f0, 0f3f800000; div.full.f32 %f0, %f0, %f1;";
        let (state, output) = run_body(body, 0.0);
        assert_eq!(output.0, 0);
        assert!(
            state
                .nodes
                .iter()
                .any(|node| matches!(node.kind, NodeKind::Opaque))
        );
        assert!(!state.taint.violations.is_empty());
        let (_, witness) = run_body(body, 1.0);
        assert_eq!(witness.0, 0.5f32.to_bits());
    }

    #[test]
    fn guarded_low_word_lop3_cannot_clear_retained_high_word_taint() {
        let body = "mov.b32 %f0, %r0; setp.ne.f32 %q, %f0, 0f00000000; \
            mov.b64 %w, 0x100000000; @%q and.b32 %w, 0, 0; \
            shr.u64 %w, %w, 32; mov.b32 %f0, %w;";
        for use_lop3_ir in [false, true] {
            let (state, retained) = run_body_ir(body, 0.0, use_lop3_ir);
            assert!(
                state
                    .nodes
                    .iter()
                    .any(|node| matches!(node.kind, NodeKind::Opaque))
            );
            assert!(!state.taint.violations.is_empty());
            let (_, replaced) = run_body_ir(body, 1.0, use_lop3_ir);
            assert_ne!(retained.0, replaced.0);
        }
    }

    #[test]
    fn direct_untracked_zero_store_has_opaque_history() {
        let direct = |shift, input| {
            run_body_finish_ir(
                &format!("shr.u32 %r1, %r0, {shift}; mov.b32 %f0, %r1;"),
                input,
                false,
                "st.global.f32 [%p1], %f0;",
            )
        };
        let (reference, routput) = direct(1, 0.0);
        let (candidate, coutput) = direct(2, 0.0);
        for (state, output) in [(&reference, routput), (&candidate, coutput)] {
            assert_eq!(output.0, 0);
            assert!(matches!(
                state.nodes[output.1.unwrap() as usize].kind,
                NodeKind::Opaque
            ));
        }
        let report = compare(
            &Side {
                ns: &reference,
                outs: vec![routput],
                buf_id: vec![0, 1],
            },
            &Side {
                ns: &candidate,
                outs: vec![coutput],
                buf_id: vec![0, 1],
            },
            &["x".into(), "out".into()],
            None,
        );
        assert!(!report.levels[0].holds && !report.levels[1].holds);
        assert_ne!(direct(1, 1.0).1.0, direct(2, 1.0).1.0);
    }

    #[test]
    fn guarded_direct_store_preserves_guard_origin() {
        let body = "mov.b32 %f0, %r0; setp.eq.f32 %q, %f0, 0f00000000;             st.global.u32 [%p1], 0x3f800000; @%q st.global.u32 [%p1], 0;";
        let (state, output) = run_body_finish_ir(body, 0.0, false, "");
        assert_eq!(output.0, 0);
        assert!(matches!(
            state.nodes[output.1.unwrap() as usize].kind,
            NodeKind::Opaque
        ));
        assert!(state.taint.memory_unit_input_derived((1, 1, 0)));
        assert_eq!(
            run_body_finish_ir(body, 1.0, false, "").1.0,
            1.0f32.to_bits()
        );
    }

    #[test]
    fn partial_and_unaligned_stores_invalidate_old_word_history() {
        for (offset, width) in [(0, "u16"), (1, "u8")] {
            let body = format!(
                "mov.b32 %f0, %r0; st.global.f32 [%p1], %f0;                 add.u64 %p1, %p1, {offset}; st.global.{width} [%p1], 0; sub.u64 %p1, %p1, {offset};"
            );
            let (state, output) = run_body_finish_ir(&body, 0.0, false, "");
            assert_eq!(output.0, 0);
            assert!(matches!(
                state.nodes[output.1.unwrap() as usize].kind,
                NodeKind::Opaque
            ));
            assert_ne!(run_body_finish_ir(&body, 1.0, false, "").1.0, 0);
        }
    }

    #[test]
    fn async_copy_preserves_issue_epoch_at_captured_zero() {
        let body = ".shared .align 4 .b8 scratch[4]; mov.u64 %w, scratch;             cp.async.ca.shared.global [%w], [%p0], 4; cp.async.commit_group;             st.global.u32 [%p0], 0; cp.async.wait_group 0; ld.shared.u32 %r1, [%w]; mov.b32 %f0, %r1;";
        let (state, output) = run_body_finish_ir(body, 0.0, false, "st.global.f32 [%p1], %f0;");
        assert_eq!(output.0, 0);
        assert_eq!(
            run_body_finish_ir(body, 1.0, false, "st.global.f32 [%p1], %f0;")
                .1
                .0,
            1.0f32.to_bits()
        );
        assert!(
            output.1.is_some(),
            "captured input history was lost at completion"
        );
        assert!(
            state.taint.memory_unit_input_derived((1, 1, 0)),
            "captured input origin was re-read from overwritten source"
        );
        assert_eq!(
            run_body_finish_ir(body, 1.0, false, "st.global.f32 [%p1], %f0;")
                .1
                .0,
            1.0f32.to_bits()
        );
    }

    #[test]
    fn async_copy_preserves_half_bytes_at_issue() {
        let body = ".shared .align 4 .b8 scratch[4]; mov.u64 %w, scratch;             cp.async.ca.shared.global [%w], [%p0], 4; cp.async.commit_group;             st.global.u32 [%p0], 0; cp.async.wait_group 0;";
        let machine = run_machine_body_finish_ir(body, 0.0, false, "", true);
        let dst = machine
            .k
            .code
            .iter()
            .find_map(|instruction| match &instruction.op {
                Op::CpAsync { dst, .. } => Some(dst),
                _ => None,
            })
            .unwrap();
        let location = machine
            .resolve(0, ptxv_ir::Space::Shared, machine.addr(0, dst).unwrap(), 4)
            .unwrap();
        let bytes = machine.half_mem(location, 4, None);
        for (byte, origin) in bytes.into_iter().enumerate() {
            assert_eq!(
                origin,
                Some(crate::halfin::Src::E {
                    alloc: 0,
                    elem: byte as u64 / 2,
                    b: (byte % 2) as u8
                })
            );
        }
        let (space, allocation, off) = mem_key(location);
        assert!(
            machine
                .num
                .as_ref()
                .unwrap()
                .taint
                .memory_unit_input_derived((space, allocation, off))
        );
    }

    #[test]
    fn async_copy_exit_preserves_issue_snapshot() {
        let body = ".shared .align 4 .b8 scratch[4]; mov.u64 %w, scratch; \
            cp.async.ca.shared.global [%w], [%p0], 4; cp.async.commit_group; \
            st.global.u32 [%p0], 0; cp.async.wait_group 0;";
        let waited = run_machine_body_finish_ir(body, 0.0, false, "", true);
        let mut kernel = (*waited.k).clone();
        for instruction in &mut kernel.code {
            if matches!(instruction.op, Op::CpAsyncWait { .. }) {
                instruction.op = Op::Nop;
            }
        }
        let mut machine = Machine::new(Arc::new(kernel), waited.launch.clone());
        let mut state = NumState::default();
        state.not_inputs.insert(1);
        state.half.inputs.insert(0, crate::halfin::Half::Bf16);
        state.taint.taint_alloc(0, 4);
        machine.num = Some(Box::new(state));
        let (machine, findings, completed) = crate::explore::run_once(machine);
        assert!(completed, "{findings:?}");
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert_eq!(findings[0].kind, "contract");
        assert!(findings[0].sig.starts_with("cp-async-exit:"));
        let dst = machine
            .k
            .code
            .iter()
            .find_map(|instruction| match &instruction.op {
                Op::CpAsync { dst, .. } => Some(dst),
                _ => None,
            })
            .unwrap();
        let location = machine
            .resolve(0, ptxv_ir::Space::Shared, machine.addr(0, dst).unwrap(), 4)
            .unwrap();
        assert_eq!(
            machine.half_mem(location, 4, None),
            waited.half_mem(location, 4, None)
        );
        let key = mem_key(location);
        assert!(
            machine
                .num
                .as_ref()
                .unwrap()
                .taint
                .memory_unit_input_derived(key)
        );
    }

    #[test]
    fn tma_does_not_relabel_written_source_as_original_input() {
        let mut unchanged = run_machine_body_finish_ir("", 0.0, false, "", true);
        unchanged.num_tma_elem(0, 0, Some((0, 0)));
        assert!(unchanged.num.as_ref().unwrap().orig_at(0, 0).is_some());
        let mut written =
            run_machine_body_finish_ir("st.global.u16 [%p0], 0;", 0.0, false, "", true);
        written.num_tma_elem(0, 0, Some((0, 0)));
        assert!(written.num.as_ref().unwrap().orig_at(0, 0).is_none());
    }
    #[test]
    fn atomic_write_must_invalidate_half_input_origins() {
        let machine =
            run_machine_body_finish_ir("atom.global.exch.b32 %r1, [%p0], 0;", 0.0, false, "", true);
        let origins = machine.half_mem(Loc::Global { alloc: 0, off: 0 }, 2, Some(&[0, 0]));
        assert!(
            !origins
                .iter()
                .any(|origin| matches!(origin, Some(crate::halfin::Src::E { .. }))),
            "an atomic rewrite must not retain original input identities: {origins:?}"
        );
    }

    #[test]
    fn predicate_files_must_not_alias_comparison_histories() {
        fn namespaces(kernel: &mut ptxv_ir::Kernel) {
            kernel.nregs[File::P as usize] = 1;
            for inst in &mut kernel.code {
                match &mut inst.op {
                    Op::Fsetp { d, cmp, .. } => {
                        *d = Reg {
                            file: if *cmp == FCmp::Lt { File::P } else { File::UP },
                            idx: 0,
                        };
                    }
                    Op::Sel { p, .. } => {
                        p.reg = Some(Reg {
                            file: File::P,
                            idx: 0,
                        })
                    }
                    _ => {}
                }
            }
        }
        let body = ".reg .pred %other; mov.b32 %f0, %r0; \
            ld.global.f32 %f1, [%p0+4]; \
            setp.lt.f32 %q, %f0, %f1; setp.gt.f32 %other, %f0, %f1; \
            selp.f32 %f1, %f0, %f1, %q;";
        let machine = run_machine_body_transform(
            body,
            1.0,
            false,
            "st.global.f32 [%p1], %f1;",
            false,
            true,
            Some(namespaces),
        );
        let output =
            u32::from_le_bytes(std::array::from_fn(|byte| *machine.gmem[1].bytes.get(byte)));
        assert_eq!(output, 1.0f32.to_bits());
        let state = machine.num.as_ref().unwrap();
        let node = state.mem_node(Loc::Global { alloc: 1, off: 0 }).unwrap();
        assert_eq!(
            state.nodes[node as usize].hw, output,
            "P0 and UP0 must keep distinct comparison histories"
        );
        assert!(
            matches!(
                state.nodes[node as usize].kind,
                NodeKind::Max { min: true, .. }
            ),
            "kind={:?}, comparisons={:?}, trace={:?}, code={:?}",
            state.nodes[node as usize].kind,
            state.pred_src,
            state.trace,
            machine
                .k
                .code
                .iter()
                .map(|inst| &inst.op)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn bare_partial_write_invalidates_float_word_history() {
        let mut machine = run_machine_body_finish_ir(
            "mov.b32 %f0, %r0;",
            1.0,
            false,
            "st.global.f32 [%p1], %f0;",
            false,
        );
        let word = Loc::Global { alloc: 1, off: 0 };
        assert!(machine.num.as_ref().unwrap().mem_node(word).is_some());
        let partial = Loc::Global { alloc: 1, off: 2 };
        let vc = machine.threads[0].vc.clone();
        machine.access(0, &vc, 0, partial, 2, Some(&[0, 0]));
        machine.num_forget(partial);
        let value =
            u32::from_le_bytes(std::array::from_fn(|byte| *machine.gmem[1].bytes.get(byte)));
        assert_eq!(value, 0);
        assert!(machine.num.as_ref().unwrap().mem_node(word).is_none());
    }
    fn half_data_registers(kernel: &mut ptxv_ir::Kernel) {
        fn register(reg: &mut Reg) {
            if reg.file == File::V && reg.idx >= 2 {
                reg.file = File::R;
            }
        }
        fn value(val: &mut Val) {
            if let Val::Reg(reg, _) = val {
                register(reg);
            }
        }
        kernel.nregs[File::R as usize] = kernel.nregs[File::V as usize];
        for inst in &mut kernel.code {
            match &mut inst.op {
                Op::Ld {
                    d,
                    space: Space::Global,
                    ..
                } => register(d),
                Op::Mov { d, a } => {
                    register(d);
                    value(a);
                }
                Op::Int { d, a, b, .. } => {
                    register(d);
                    value(a);
                    value(b);
                }
                Op::Fsetp { a, b, .. } => {
                    value(a);
                    value(b);
                }
                Op::St { v, .. } => value(v),
                _ => {}
            }
        }
    }

    #[test]
    fn input_guard_cannot_leave_half_registers_as_unconditional_copies() {
        let body = "mov.b32 %r1, %r0; mov.b32 %f0, %r0; shl.b32 %f0, %f0, 16; \
            setp.ne.f32 %q, %f0, 0f00000000; @%q mov.b32 %r1, 0;";
        for input in [0.0, 1.0] {
            let machine = run_machine_body_transform(
                body,
                input,
                false,
                "st.global.u16 [%p1], %r1;",
                true,
                false,
                Some(half_data_registers),
            );
            let origins = machine.half_mem(Loc::Global { alloc: 1, off: 0 }, 2, None);
            assert!(origins.iter().all(Option::is_none), "{origins:?}");
        }
    }

    #[test]
    fn input_guard_cannot_leave_half_stores_as_unconditional_copies() {
        let body = "mov.b32 %r1, %r0; mov.b32 %f0, %r0; shl.b32 %f0, %f0, 16; \
            setp.gt.f32 %q, %f0, 0f00000000;";
        let machine = run_machine_body_transform(
            body,
            1.0,
            false,
            "@%q st.global.u16 [%p1], %r1;",
            true,
            false,
            Some(half_data_registers),
        );
        let origins = machine.half_mem(Loc::Global { alloc: 1, off: 0 }, 2, None);
        assert!(origins.iter().all(Option::is_none), "{origins:?}");
    }
}

#[cfg(test)]
mod control_dependency_tests {
    use super::untracked_producer_tests::run_body_finish_ir;
    use super::*;

    #[test]
    fn address_value_alias_still_reports_control_dependency() {
        let body = "cvt.u64.u32 %w, %r0; add.u64 %p1, %p1, %w; st.global.u32 [%p1], %p1;";
        let (state, output) = run_body_finish_ir(body, 0.0, false, "");
        assert_eq!(output.0, 0);
        assert!(
            state
                .taint
                .violations
                .iter()
                .any(|message| message.contains("address / control"))
        );
    }
    #[test]
    fn source_validity_and_branch_mask_are_control_inputs() {
        let predicate = Reg {
            file: File::P,
            idx: 0,
        };
        let mask = Reg {
            file: File::UR,
            idx: 0,
        };
        let address = Addr {
            base: None,
            base2: None,
            off: 0,
        };
        let copy = Op::CpAsync {
            dst: address.clone(),
            src: address,
            bytes: 4,
            src_ok: Pred {
                reg: Some(predicate),
                neg: false,
            },
            src_size: None,
            zfill: false,
        };
        assert!(control_regs(&copy).contains(&predicate));
        let branch = Op::Bra {
            target: 0,
            any: false,
            tid0_falls: false,
            cond: Some(Pred {
                reg: Some(predicate),
                neg: false,
            }),
            div: Some(Val::Reg(mask, 1)),
        };
        let controls = control_regs(&branch);
        assert!(controls.contains(&predicate));
        assert!(controls.contains(&mask));
    }
}

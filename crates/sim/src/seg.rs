/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Segments: the instruction slices between tracked numerics nodes (docs/equiv-reassoc.md).
//!
//! The numerics shadow (`num.rs`) keeps an exact dataflow node for sums, products, maxima
//! and input leaves. Everything else a kernel does to an input-derived value (an exp
//! idiom's range reduction with `FFMA.RM`, `SHF` exponent bits, `MUFU.EX2`, a guarded
//! `FMUL` for the subnormal range, a division's Newton steps) has no node. In equiv runs
//! (`NumState::segments`) such a value is *materialized* when a tracked operation consumes it
//! or it is stored: a backward slice along the thread's executed instruction trace, from the
//! tracked nodes it depends on (its arguments) and input-independent constants. Control flow
//! does not depend on input data (N-EQUIV's taint check), so the trace is the same for every
//! input, except where a guard depends on input data: such an instruction is part of the
//! slice whatever its guard did in the run (`Guard`). A slice is thus the computation for
//! every input, as a function of its arguments. Its value at any
//! argument is `eval`; `equiv` identifies one-argument slices by exhaustive evaluation
//! (`softmax.rs`).
//!
//! ptxas's IEEE division (`div.rn.f32`: `MUFU.RCP`, two Newton steps, `FCHK` and a call of
//! the slow path) is one opaque operation (`DivRn`), whose result PTX defines as the
//! correctly rounded quotient. ASSUME(N-SEG): see ASSUMPTIONS.md

use std::collections::BTreeSet;
use std::collections::HashMap;

use ptxv_ir::*;
use serde::Deserialize;
use serde::Serialize;

use crate::fp;
use crate::machine::Machine;

/// A value in a slice: an argument (index into `SProg::args`), a constant (32-bit pattern or
/// predicate 0 / 1), or the result of an operation (index into `SProg::ops`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SVal {
    Arg(u16),
    Const(u64),
    Op(u32),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SKind {
    /// The instruction at `pc`, computing register `dst`, with each register / predicate
    /// source given by `regs` and each constant-bank operand's value by `cvals`.
    Inst {
        op: Op,
        dst: Reg,
        regs: Vec<(Reg, SVal)>,
        cvals: Vec<(Val, u64)>,
    },
    /// A guarded instruction whose guard depends on input data: `pred ^ neg ? new : old`.
    Guard {
        pred: SVal,
        neg: bool,
        new: SVal,
        old: SVal,
    },
    /// ptxas's `div.rn.f32` sequence: the correctly rounded a / b.
    DivRn { a: SVal, b: SVal },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SOp {
    pub pc: u32,
    pub kind: SKind,
}

/// One materialized slice.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SProg {
    /// Numerics nodes of the arguments.
    pub args: Vec<u32>,
    /// In dependence order (each op's sources come before it).
    pub ops: Vec<SOp>,
    pub out: SVal,
    /// Instructions of div.rn sequences the slice stands for.
    pub extra_pcs: BTreeSet<u32>,
}

impl SProg {
    /// Instructions the slice covers (their data-dependent guards are part of it).
    pub fn pcs(&self) -> BTreeSet<u32> {
        self.ops
            .iter()
            .map(|o| o.pc)
            .chain(self.extra_pcs.iter().copied())
            .collect()
    }

    /// The value at the given argument bits.
    pub fn eval(&self, args: &[u32]) -> Result<u32, String> {
        let mut vals = Vec::with_capacity(self.ops.len());
        self.eval_with(args, &mut vals)
    }

    /// `eval` with a scratch buffer (sweeps).
    pub fn eval_with(&self, args: &[u32], vals: &mut Vec<u64>) -> Result<u32, String> {
        vals.clear();
        let get = |v: SVal, vals: &[u64]| -> u64 {
            match v {
                SVal::Arg(i) => args[i as usize] as u64,
                SVal::Const(c) => c,
                SVal::Op(i) => vals[i as usize],
            }
        };
        for o in &self.ops {
            let r = match &o.kind {
                SKind::Guard {
                    pred,
                    neg,
                    new,
                    old,
                } => {
                    if (get(*pred, vals) != 0) != *neg {
                        get(*new, vals)
                    } else {
                        get(*old, vals)
                    }
                }
                SKind::DivRn { a, b } => {
                    let (a, b) = (get(*a, vals) as u32, get(*b, vals) as u32);
                    div_rn(a, b) as u64
                }
                SKind::Inst {
                    op,
                    dst,
                    regs,
                    cvals,
                } => {
                    let rv = |v: &Val| -> u64 {
                        match v {
                            Val::Reg(r, _) => regs
                                .iter()
                                .find(|(x, _)| x == r)
                                .map_or(0, |(_, s)| get(*s, vals)),
                            Val::Imm(c) => *c,
                            v => cvals.iter().find(|(x, _)| x == v).map_or(0, |(_, c)| *c),
                        }
                    };
                    let rp = |p: &Pred| -> bool {
                        let v = match p.reg {
                            None => true,
                            Some(r) => regs
                                .iter()
                                .find(|(x, _)| *x == r)
                                .is_none_or(|(_, s)| get(*s, vals) != 0),
                        };
                        v != p.neg
                    };
                    eval_inst(op, *dst, &rv, &rp)?
                }
            };
            vals.push(r);
        }
        Ok(get(self.out, vals) as u32)
    }
}

/// The correctly rounded f32 quotient (`div.rn.f32`: IEEE 754 binary32 division, nearest
/// even, subnormals kept; NaN results are the canonical NaN, N-FALU).
pub fn div_rn(a: u32, b: u32) -> u32 {
    let q = f32::from_bits(a) / f32::from_bits(b);
    if q.is_nan() { 0x7fff_ffff } else { q.to_bits() }
}

fn fmod(u: u32, neg: bool, abs: bool) -> u32 {
    let u = if abs { u & 0x7fff_ffff } else { u };
    if neg { u ^ 0x8000_0000 } else { u }
}

/// One instruction's result (`dst`; a predicate is 0 / 1), for the operations slices use.
/// The semantics are exec.rs's (N-FALU, N-MUFU, N-FCHK).
pub(crate) fn eval_inst(
    op: &Op,
    dst: Reg,
    rv: &dyn Fn(&Val) -> u64,
    rp: &dyn Fn(&Pred) -> bool,
) -> Result<u64, String> {
    let w = |v: &Val| rv(v) as u32;
    Ok(match op {
        Op::Ffma {
            a,
            b,
            c,
            neg,
            abs,
            rnd,
            ftz,
            sat,
            ..
        } => {
            let x = fmod(w(a), neg[0], abs[0]);
            let y = fmod(w(b), neg[1], abs[1]);
            let z = c.as_ref().map(|c| fmod(w(c), neg[2], abs[2]));
            let r = fp::f32_fma(x, y, z, *rnd, *ftz);
            (if *sat { fp::f32_sat(r) } else { r }) as u64
        }
        Op::Fadd {
            a,
            b,
            rnd,
            neg,
            abs,
            ftz,
            sat,
            ..
        } => {
            let x = fmod(w(a), neg[0], abs[0]);
            let y = fmod(w(b), neg[1], abs[1]);
            let r = fp::f32_fma(x, 0x3f80_0000, Some(y), *rnd, *ftz);
            (if *sat { fp::f32_sat(r) } else { r }) as u64
        }
        Op::Mufu {
            a,
            func,
            fmt,
            hi,
            neg,
            abs,
            ptx,
            ..
        } => {
            // The f64 forms (high-word units) are not slice material.
            if *fmt == ptxv_ir::MufuFmt::F64 {
                return Err(format!("{func:?} (F64) in a slice"));
            }
            let x = fmod(w(a), *neg, *abs);
            match ptx {
                None => crate::mufu::mufu(*func, *fmt, *hi, x),
                Some(f) => crate::mufu::ptx_approx(*func, *fmt, *f, x),
            }
            .ok_or_else(|| format!("{func:?} not modeled"))? as u64
        }
        Op::Fsetp {
            cmp,
            a,
            b,
            neg,
            abs,
            c,
            bop,
            ftz,
            ..
        } => {
            let x = fmod(w(a), neg[0], abs[0]);
            let y = fmod(w(b), neg[1], abs[1]);
            let r = fp::f32_cmp(x, y, *cmp, *ftz);
            let q = rp(c);
            (match bop {
                BoolOp::And => r && q,
                BoolOp::Or => r || q,
                BoolOp::Xor => r != q,
            }) as u64
        }
        Op::Hsetp2Bf16 {
            q: qd,
            cmp,
            a,
            b,
            sel,
            neg,
            abs,
            c,
            bop,
            f16,
            ..
        } => {
            let lane = (Some(dst) == *qd) as u32;
            let r = fp::hsetp2_lane((w(a), w(b)), *sel, (*neg, *abs), *f16, *cmp, lane);
            let q = rp(c);
            (match bop {
                BoolOp::And => r && q,
                BoolOp::Or => r || q,
                BoolOp::Xor => r != q,
            }) as u64
        }
        Op::Fchk { a, b, .. } => fp::fchk(w(a), w(b)) as u64,
        // Predicate logic (torch's rsqrtf combines the subnormal test with the bounds test:
        // `PLOP3.LUT P0, PT, PT, PT, PT, 0x80, 0x8` then `@!P6 PLOP3.LUT P0, PT, P1, ...`).
        Op::Plop3 {
            a, b, c, lut, d2, ..
        } => {
            let i = ((rp(a) as u8) << 2) | ((rp(b) as u8) << 1) | rp(c) as u8;
            let lut = match d2 {
                Some((r, lut2)) if *r == dst => lut2,
                _ => lut,
            };
            ((lut >> i) & 1) as u64
        }
        Op::Sel { a, b, p, .. } => (if rp(p) { w(a) } else { w(b) }) as u64,
        Op::Fsel {
            a, b, p, neg, abs, ..
        } => {
            let (v, k) = if rp(p) { (a, 0) } else { (b, 1) };
            fmod(w(v), neg[k], abs[k]) as u64
        }
        Op::Fmnmx {
            a,
            b,
            min,
            ftz,
            nan,
            xorsign: true,
            ..
        } => fp::f32_minmax_xorsign(w(a), w(b), rp(min), *nan, *ftz) as u64,
        Op::Fmnmx {
            a,
            b,
            c,
            neg,
            abs,
            min,
            ftz,
            nan,
            ..
        } => {
            let x = fmod(w(a), neg[0], abs[0]);
            let y = fmod(w(b), neg[1], abs[1]);
            let lo = rp(min);
            let mut r = fp::f32_minmax(x, y, lo, *nan, *ftz);
            if let Some(c) = c {
                r = fp::f32_minmax(r, fmod(w(c), neg[2], abs[2]), lo, *nan, *ftz);
            }
            r as u64
        }
        Op::Hfma2Bf16 {
            a,
            b,
            c,
            sel,
            neg,
            abs,
            relu,
            ..
        } => fp::hfma2_bf16([w(a), w(b), w(c)], *sel, (*neg, *abs), *relu) as u64,
        Op::FfmaBf16 {
            a,
            b,
            c,
            hi,
            neg,
            rnd,
            sat,
            ..
        } => fp::ffma_bf16([w(a), w(b), w(c)], *hi, *neg, *rnd, *sat) as u64,
        Op::Hmnmx2Bf16 {
            a,
            b,
            sel,
            min,
            nan,
            xorsign,
            ..
        } => fp::hmnmx2_bf16((w(a), w(b)), *sel, rp(min), *nan, *xorsign) as u64,
        Op::Vhmnmx {
            a,
            b,
            c,
            sel,
            abs,
            min,
            nan,
            f16,
            ..
        } => fp::vhmnmx([w(a), w(b), w(c)], *sel, *abs, (rp(min), *nan, *f16)) as u64,
        Op::Shf {
            lo,
            sh,
            hi,
            right,
            signed,
            high,
            ..
        } => {
            let v = (rv(hi) << 32) | (rv(lo) & 0xffff_ffff);
            let s = (rv(sh) & 63) as u32;
            let r = if *right {
                if *signed {
                    ((v as i64) >> s) as u64
                } else {
                    v >> s
                }
            } else {
                v << s
            };
            if *high { r >> 32 } else { r & 0xffff_ffff }
        }
        Op::Lop3 {
            a,
            b,
            c,
            lut,
            p,
            pq,
            ..
        } => {
            let (x, y, z) = (rv(a), rv(b), rv(c));
            let mut r = 0u64;
            for i in 0..8 {
                if (lut >> i) & 1 == 1 {
                    let fa = if i & 4 != 0 { x } else { !x };
                    let fb = if i & 2 != 0 { y } else { !y };
                    let fc = if i & 1 != 0 { z } else { !z };
                    r |= fa & fb & fc;
                }
            }
            let r = r & 0xffff_ffff;
            if *p == Some(dst) {
                // L-LOP3-P: P = (r != 0) || Pq, or && Pq (`.PAND`).
                (match pq {
                    None => r != 0,
                    Some((q, true)) => r != 0 && rp(q),
                    Some((q, false)) => r != 0 || rp(q),
                }) as u64
            } else {
                r
            }
        }
        Op::Imad {
            a,
            b,
            c,
            neg_c,
            not_c,
            wide: false,
            hi: false,
            cin: None,
            cout: None,
            ..
        } => {
            let mut c0 = rv(c) & 0xffff_ffff;
            if *not_c {
                c0 = !c0 & 0xffff_ffff;
            }
            if *neg_c {
                c0 = c0.wrapping_neg();
            }
            (rv(a) & 0xffff_ffff)
                .wrapping_mul(rv(b) & 0xffff_ffff)
                .wrapping_add(c0)
                & 0xffff_ffff
        }
        Op::Iadd3 {
            a,
            b,
            c,
            neg,
            not,
            wide: false,
            cout,
            cin,
            ..
        } if cout.is_empty() && cin.is_empty() => {
            let mut s = 0u64;
            for ((v, n), m) in [a, b, c].iter().zip(neg).zip(not) {
                let mut x = rv(v) & 0xffff_ffff;
                if *m {
                    x = !x & 0xffff_ffff;
                }
                if *n {
                    x = x.wrapping_neg();
                }
                s = s.wrapping_add(x);
            }
            s & 0xffff_ffff
        }
        Op::Mov { a, .. } => rv(a) & 0xffff_ffff,
        // ptxas's rcp.rn exponent filter as `VIADD`.
        Op::Int {
            op: IntOp::Add,
            bits: 32,
            a,
            b,
            ..
        } => rv(a).wrapping_add(rv(b)) & 0xffff_ffff,
        // 32-bit integer compares (exec.rs `Op::Setp`): predicates saved by `P2R` and read
        // back with `ISETP.NE P, R, RZ`.
        Op::Setp {
            cmp,
            bits: 32,
            a,
            b,
            and,
            or,
            xor,
            ex: None,
            ..
        } => {
            let (x, y) = (rv(a) & 0xffff_ffff, rv(b) & 0xffff_ffff);
            let (sx, sy) = (x as u32 as i32, y as u32 as i32);
            let r = match cmp {
                Cmp::Eq => x == y,
                Cmp::Ne => x != y,
                Cmp::LtU => x < y,
                Cmp::LeU => x <= y,
                Cmp::GtU => x > y,
                Cmp::GeU => x >= y,
                Cmp::LtS => sx < sy,
                Cmp::LeS => sx <= sy,
                Cmp::GtS => sx > sy,
                Cmp::GeS => sx >= sy,
            };
            let q = rp(and);
            (if *xor {
                r ^ q
            } else if *or {
                r || q
            } else {
                r && q
            }) as u64
        }
        // `P2R d, PR, a, mask`: d = (a & !mask) | (bit i = Pi, i in mask).
        Op::P2r {
            a,
            mask,
            uniform,
            shift,
            ..
        } => {
            let mut pr = 1u64 << 7; // PT / UPT
            for i in 0..7u16 {
                if mask >> i & 1 == 1 {
                    let p = Pred {
                        reg: Some(Reg {
                            file: if *uniform { File::UP } else { File::P },
                            idx: i,
                        }),
                        neg: false,
                    };
                    pr |= (rp(&p) as u64) << i;
                }
            }
            let (m, pr) = ((*mask as u64) << shift, pr << shift);
            (rv(a) & !m & 0xffff_ffff) | (pr & m)
        }
        // FFMA2 / FMUL2 / FADD2: the lane `dst` holds (exec.rs: lanes of a 64-bit operand are
        // its two registers; `.F32` broadcasts the low one).
        Op::Ffma2 {
            d,
            a,
            b,
            c,
            neg,
            abs,
            rnd,
        } => {
            let lane = dst.idx.wrapping_sub(d.idx);
            if lane > 1 {
                return Err("FFMA2 lane".into());
            }
            let opv = |(v, l): &(Val, F2Lanes)| -> u32 {
                let i = match l {
                    F2Lanes::Bcast => 0,
                    F2Lanes::Pair => lane,
                    F2Lanes::Swap => 1 - lane,
                };
                match v {
                    Val::Reg(r, _) => rv(&Val::Reg(
                        Reg {
                            file: r.file,
                            idx: r.idx + i,
                        },
                        1,
                    )) as u32,
                    Val::Imm(x) => (if i == 0 { *x } else { *x >> 32 }) as u32,
                    v => rv(v) as u32,
                }
            };
            let n = |k: usize| if neg[k] { 0x8000_0000 } else { 0 };
            let m = |k: usize| if abs[k] { 0x7fff_ffff } else { !0 };
            fp::f32_fma(
                opv(a) & m(0) ^ n(0),
                opv(b) & m(1) ^ n(1),
                Some(opv(c) & m(2) ^ n(2)),
                *rnd,
                false,
            ) as u64
        }
        Op::CvtF32Bf16 { a, .. } => (rv(a) & 0xffff) << 16,
        // Integer <-> f32 conversions of exponent fields (Triton's log: `I2FP.F32.S32` of the
        // extracted exponent), as exec.rs; 64-bit (register pair) forms are not modeled.
        Op::I2F {
            a,
            signed,
            rnd,
            bits,
            hi,
            f64: false,
            ..
        } if *bits != 64 => {
            let x = rv(a) >> if *hi { 16 } else { 0 };
            let x = match (*bits, *signed) {
                (16, true) => x as u16 as i16 as i64 as u64,
                (16, false) => x & 0xffff,
                (8, true) => x as u8 as i8 as i64 as u64,
                (8, false) => x & 0xff,
                (_, true) => x as u32 as i32 as i64 as u64,
                (_, false) => x & 0xffff_ffff,
            };
            fp::int_to_f32(x, *signed, *rnd).to_bits() as u64
        }
        Op::F2I {
            a,
            signed,
            bits,
            rnd,
            ftz,
            f64: false,
            ..
        } if *bits == 32 => {
            let u = if *ftz { fp::f32_ftz(w(a)) } else { w(a) };
            let x = f32::from_bits(u);
            let x = match rnd {
                Rnd::Rz => x,
                Rnd::Rn => x.round_ties_even(),
                Rnd::Rm => x.floor(),
                Rnd::Rp => x.ceil(),
            };
            (if *signed { x as i32 as u32 } else { x as u32 }) as u64
        }
        Op::F2fpBf16 { a, b, .. } => {
            ((fp::f32_to_bf16_rn_gpu(w(a)) as u64) << 16) | fp::f32_to_bf16_rn_gpu(w(b)) as u64
        }
        Op::F2fpF16 { a, b, .. } => {
            ((fp::f32_to_f16_rn(w(a)) as u64) << 16) | fp::f32_to_f16_rn(w(b)) as u64
        }
        Op::Prmt { a, sel, b, .. } => {
            let x = (rv(a) & 0xffff_ffff) | ((rv(b) & 0xffff_ffff) << 32);
            let s = rv(sel);
            let mut out = 0u64;
            for i in 0..4 {
                let n = (s >> (4 * i)) & 0xf;
                let byte = (x >> (8 * (n & 7))) & 0xff;
                let v = if n & 8 != 0 {
                    if byte & 0x80 != 0 { 0xff } else { 0 }
                } else {
                    byte
                };
                out |= v << (8 * i);
            }
            out
        }
        _ => return Err(format!("{op:?} is not modeled in a segment")),
    })
}

/// `Slicer::memo`'s key: (register, position, decided guards, demanded bits).
type MemoKey = (Reg, usize, Vec<(SVal, bool)>, u32);

/// Builds one slice (`Machine::seg_slice`).
struct Slicer<'a> {
    m: &'a Machine,
    t: usize,
    /// The thread's issued instructions (`NumState::trace`).
    trace: &'a [u32],
    args: Vec<u32>,
    ops: Vec<SOp>,
    /// (register, position, decided guards, demanded bits) -> value.
    memo: HashMap<MemoKey, SVal>,
    depth: usize,
    extra: BTreeSet<u32>,
    /// Data-dependent guard predicates decided on the current path: inside the `new` side of
    /// a `Guard` its predicate is known (a second instruction under the same guard takes its
    /// own `new` side; erf's `@P MUFU.EX2 R; @P FADD R, -R, 1` reads R only where P holds).
    assume: Vec<(SVal, bool)>,
}

const MAX_OPS: usize = 1024;

/// The 0 / 1 values bit i of a LOP3 operand can take: an immediate's bit, else both.
fn lop3_bits(v: &Val, i: u32) -> &'static [u64] {
    match v {
        Val::Imm(x) if x >> i & 1 == 1 => &[1],
        Val::Imm(_) => &[0],
        _ => &[0, 1],
    }
}

/// Whether `op` is a `LOP3 r = lut(..)` whose result equals its operand `r` (the same
/// register; the other operands immediates) in every bit of `mask`, for both values of that
/// bit. Each result bit is the LUT of the operands' same bits, so then the demanded bits are
/// what they were before, whether the instruction executes or not.
// ASSUME(N-SEG): see ASSUMPTIONS.md
fn lop3_keeps(op: &Op, r: Reg, mask: u32) -> bool {
    let Op::Lop3 {
        d, a, b, c, lut, p, ..
    } = op
    else {
        return false;
    };
    if *d != r || *p == Some(r) {
        return false;
    }
    let ops = [a, b, c];
    let own: Vec<usize> = (0..3)
        .filter(|&k| matches!(ops[k], Val::Reg(x, _) if *x == r))
        .collect();
    let [k] = own[..] else { return false };
    if (0..3).any(|q| q != k && !matches!(ops[q], Val::Imm(_))) {
        return false;
    }
    (0..32).filter(|i| mask >> i & 1 == 1).all(|i| {
        [0u64, 1].iter().all(|&x| {
            let bit = |q: usize| if q == k { x } else { lop3_bits(ops[q], i)[0] };
            let idx = (bit(0) << 2) | (bit(1) << 1) | bit(2);
            (*lut as u64 >> idx) & 1 == x
        })
    })
}

/// The bits of source `s` that instruction `op` computing `dst` reads, when the bits of
/// `mask` of its result are demanded: for a LOP3, the bits where some value of the other
/// operands (an immediate's own bits) lets bit i of `s` change result bit i (every bit for
/// the predicate output, `result != 0`); every bit for other instructions.
// ASSUME(N-SEG): see ASSUMPTIONS.md
fn lop3_demand(op: &Op, dst: Reg, s: Reg, mask: u32) -> u32 {
    let Op::Lop3 {
        d, a, b, c, lut, p, ..
    } = op
    else {
        return u32::MAX;
    };
    let out = if *p == Some(dst) {
        u32::MAX
    } else if *d == dst {
        mask
    } else {
        return u32::MAX;
    };
    let ops = [a, b, c];
    let mut m = 0u32;
    for k in (0..3).filter(|&k| matches!(ops[k], Val::Reg(x, _) if *x == s)) {
        for i in (0..32).filter(|i| out >> i & 1 == 1) {
            let others: Vec<usize> = (0..3).filter(|&q| q != k).collect();
            let dep = lop3_bits(ops[others[0]], i).iter().any(|&y| {
                lop3_bits(ops[others[1]], i).iter().any(|&z| {
                    let at = |x: u64| {
                        let mut v = [0u64; 3];
                        v[k] = x;
                        v[others[0]] = y;
                        v[others[1]] = z;
                        (*lut as u64 >> ((v[0] << 2) | (v[1] << 1) | v[2])) & 1
                    };
                    at(0) != at(1)
                })
            });
            if dep {
                m |= 1 << i;
            }
        }
    }
    // The same register in two operand positions, or not a LUT operand: all its bits.
    if ops
        .iter()
        .filter(|v| matches!(v, Val::Reg(x, _) if *x == s))
        .count()
        != 1
    {
        return u32::MAX;
    }
    m
}
/// Trace entry flag: issued with its guard false.
pub const GUARD_FALSE: u32 = 1 << 31;

impl Slicer<'_> {
    fn arg(&mut self, n: u32) -> SVal {
        let i = match self.args.iter().position(|&a| a == n) {
            Some(i) => i,
            None => {
                self.args.push(n);
                self.args.len() - 1
            }
        };
        SVal::Arg(i as u16)
    }

    fn push(&mut self, pc: u32, kind: SKind) -> Result<SVal, String> {
        if self.ops.len() >= MAX_OPS {
            return Err(format!("more than {MAX_OPS} operations"));
        }
        self.ops.push(SOp { pc, kind });
        Ok(SVal::Op(self.ops.len() as u32 - 1))
    }

    fn describe(&self, pc: u32) -> String {
        self.m.k.describe(pc as usize)
    }

    /// The value register `r` holds just before trace position `pos` (predicates: 0 / 1).
    fn value(&mut self, r: Reg, pos: usize) -> Result<SVal, String> {
        self.value_m(r, pos, u32::MAX)
    }

    /// `value` where only the bits of `mask` are read (a flag word's bit tested by `LOP3`):
    /// the result is exact in those bits; the others are unspecified.
    fn value_m(&mut self, r: Reg, pos: usize, mask: u32) -> Result<SVal, String> {
        let mask = if matches!(r.file, File::P | File::UP) {
            u32::MAX
        } else {
            mask
        };
        if mask == 0 {
            // No bit is read.
            return Ok(SVal::Const(0));
        }
        // The value in every bit serves any demanded bits.
        for m in [mask, u32::MAX] {
            if let Some(v) = self.memo.get(&(r, pos, self.assume.clone(), m)) {
                return Ok(*v);
            }
        }
        self.depth += 1;
        if self.depth > 400 {
            return Err("slice too deep".into());
        }
        let v = self.value_inner(r, pos, mask);
        self.depth -= 1;
        let v = v?;
        self.memo.insert((r, pos, self.assume.clone(), mask), v);
        Ok(v)
    }

    fn value_inner(&mut self, r: Reg, pos: usize, mask: u32) -> Result<SVal, String> {
        let code = &self.m.k.code;
        let mut j = pos;
        while j > 0 {
            j -= 1;
            let e = self.trace[j];
            let pc = e & !GUARD_FALSE;
            let inst = &code[pc as usize];
            // The close of ptxas's rcp.rn sequence: one opaque reciprocal (1 / b rounded).
            if matches!(inst.op, Op::Bsync { .. }) {
                if let Some((b, q, add)) = rcprn_at(self.m, pc) {
                    let f = (0..j)
                        .rev()
                        .find(|&x| self.trace[x] & !GUARD_FALSE == add)
                        .ok_or("rcp.rn without its exponent filter in the trace")?;
                    if q == r {
                        let bv = self.value(b, f)?;
                        self.extra.extend(rcprn_pcs(self.m, pc));
                        return self.push(
                            pc,
                            SKind::DivRn {
                                a: SVal::Const(0x3f80_0000),
                                b: bv,
                            },
                        );
                    }
                    if rcprn_clobbers(self.m, pc).contains(&r) {
                        return Err(format!(
                            "{} is written by the slow path of the rcp.rn sequence closing at {}",
                            reg_name(r),
                            self.describe(pc)
                        ));
                    }
                    let bra = (f..j)
                        .find(|&x| {
                            let i = &code[(self.trace[x] & !GUARD_FALSE) as usize];
                            matches!(i.op, Op::Bra { .. }) && i.guard.reg.is_some()
                        })
                        .ok_or("rcp.rn without its branch in the trace")?;
                    j = bra;
                    continue;
                }
            }
            // The close of ptxas's div.rn sequence: one opaque quotient.
            if matches!(inst.op, Op::Bsync { .. }) {
                if let Some((a, b, q)) = divrn_at(self.m, pc) {
                    if q == r {
                        let f = (0..j)
                            .rev()
                            .find(|&x| {
                                matches!(
                                    code[(self.trace[x] & !GUARD_FALSE) as usize].op,
                                    Op::Fchk { .. }
                                )
                            })
                            .ok_or("div.rn without its FCHK in the trace")?;
                        let av = self.value(a, f)?;
                        let bv = match b {
                            Val::Reg(b, _) => self.value(b, f)?,
                            Val::Imm(x) => SVal::Const(x as u32 as u64),
                            _ => return Err("div.rn divisor operand".into()),
                        };
                        let lo = self.trace[f] & !GUARD_FALSE;
                        self.extra.extend(lo.saturating_sub(4)..=pc);
                        return self.push(pc, SKind::DivRn { a: av, b: bv });
                    }
                    // Another register across the sequence: the slow path may not write it
                    // (both paths share what comes before the branch).
                    if divrn_clobbers(self.m, pc).contains(&r) {
                        return Err(format!(
                            "{} is written by the slow path of the div.rn sequence closing at {}",
                            reg_name(r),
                            self.describe(pc)
                        ));
                    }
                    let bra = (0..j)
                        .rev()
                        .find(|&x| {
                            let i = &code[(self.trace[x] & !GUARD_FALSE) as usize];
                            matches!(i.op, Op::Bra { target, .. } if target as u32 == pc)
                        })
                        .ok_or("div.rn without its branch in the trace")?;
                    j = bra;
                    continue;
                }
            }
            // A branch whose guard depends on input data: other inputs take another path.
            // ASSUME(N-FMAX): a float atomic max diamond's two sides write no register and
            // join right after, so every register is the same on both.
            if let (Op::Bra { .. }, Some(g)) = (&inst.op, inst.guard.reg) {
                if crate::num::fmax_branch(code, pc as usize).is_none()
                    && !matches!(self.value(g, j)?, SVal::Const(_))
                {
                    return Err(format!(
                        "data-dependent branch at {} before the value is read",
                        self.describe(pc)
                    ));
                }
            }
            if !inst.op.dst_regs().contains(&r) {
                continue;
            }
            // A LOP3 that leaves the demanded bits as they were, executed or not (setting or
            // clearing other bits of a flag word): look further back.
            if lop3_keeps(&inst.op, r, mask) {
                continue;
            }
            // One value per write (a guard predicate read at two places is one `SVal`).
            let key = (r, usize::MAX - j, self.assume.clone(), mask);
            for m in [mask, u32::MAX] {
                if let Some(v) = self.memo.get(&(r, usize::MAX - j, self.assume.clone(), m)) {
                    return Ok(*v);
                }
            }
            let v = self.writer(r, j, mask)?;
            self.memo.insert(key, v);
            return Ok(v);
        }
        // Never written in the trace: an initial register (only untainted ones are values).
        Err(format!(
            "{} is read before any write in the trace",
            reg_name(r)
        ))
    }

    /// Register `r` as the instruction at trace position `j` left it.
    fn writer(&mut self, r: Reg, j: usize, mask: u32) -> Result<SVal, String> {
        let ns = self.m.num.as_ref().unwrap();
        let e = self.trace[j];
        let pc = e & !GUARD_FALSE;
        let inst = self.m.k.code[pc as usize].clone();
        // The guard's value does not depend on the guards decided on this path.
        let saved = std::mem::take(&mut self.assume);
        let gv = match inst.guard.reg {
            Some(g) => self.value(g, j).map(Some),
            None => Ok(None),
        };
        self.assume = saved;
        let gv = gv?;
        let data_guard = matches!(gv, Some(SVal::Op(_)) | Some(SVal::Arg(_)));
        if !data_guard {
            if e & GUARD_FALSE != 0 {
                // Not executed, for every input: the value from before.
                return self.value_m(r, j, mask);
            }
            if let Some(d) = ns
                .defs
                .get(&(self.t as u32, j as u32, r.file as u8, r.idx))
                .copied()
            {
                if let Some(n) = d.node {
                    return Ok(self.arg(n));
                }
                // The demanded bits are the same for every input.
                if !d.tainted || d.tmask & mask == 0 {
                    return Ok(SVal::Const(d.val));
                }
            }
            return self.inst(&inst, pc, r, j, mask);
        }
        // The guard depends on input data: both outcomes (one, if the path decided it).
        let p = gv.unwrap();
        let holds = !inst.guard.neg; // the guard holds iff (p != 0) == holds
        if let Some(&(_, pv)) = self.assume.iter().find(|(s, _)| *s == p) {
            return if pv == holds {
                self.inst(&inst, pc, r, j, mask)
            } else {
                self.value_m(r, j, mask)
            };
        }
        self.assume.push((p, holds));
        let new = self.inst(&inst, pc, r, j, mask);
        self.assume.pop();
        let new = new?;
        self.assume.push((p, !holds));
        let old = self.value_m(r, j, mask);
        self.assume.pop();
        let old = old?;
        self.push(
            pc,
            SKind::Guard {
                pred: gv.unwrap(),
                neg: inst.guard.neg,
                new,
                old,
            },
        )
    }

    /// The instruction at trace position `j` (pc `pc`) computing `dst`, its sources sliced.
    fn inst(
        &mut self,
        inst: &Inst,
        pc: u32,
        dst: Reg,
        j: usize,
        mask: u32,
    ) -> Result<SVal, String> {
        if matches!(
            inst.op,
            Op::Shfl { .. }
                | Op::Ld { .. }
                | Op::LdReduce { .. }
                | Op::Atom { .. }
                | Op::TmemLd { .. }
        ) {
            return Err(format!(
                "{} (moves an input-derived value without a node)",
                self.describe(pc)
            ));
        }
        let mut regs = Vec::new();
        for s in inst.op.src_regs() {
            if regs.iter().any(|(x, _)| *x == s) {
                continue;
            }
            let v = self.value_m(s, j, lop3_demand(&inst.op, dst, s, mask))?;
            regs.push((s, v));
        }
        let mut cvals = Vec::new();
        for v in inst.op.src_vals() {
            if let Val::Const(..) = v {
                let x = self.m.rd(self.t, &v).map_err(|e| e.to_string())?;
                cvals.push((v, x & 0xffff_ffff));
            }
        }
        // Check the op is one `eval` covers (fail here, with the instruction, not later).
        eval_inst(&inst.op, dst, &|_| 0, &|_| true)
            .map_err(|e| format!("{e} ({})", self.describe(pc)))?;
        self.push(
            pc,
            SKind::Inst {
                op: inst.op.clone(),
                dst,
                regs,
                cvals,
            },
        )
    }
}

/// Is the instruction at `pc` the `BSYNC` closing ptxas's `div.rn.f32` sequence? Shapes:
///   BSSY; MUFU.RCP r0, b; FCHK P, a, b; FFMA x5 (two Newton steps, the last writing q);
///   @!P BRA L; (slow path:) ...; CALL <*div_rn*slowpath*>; MOV q, res; L: BSYNC
/// or, for a divisor known at compile time (`FCHK P, a, 6`; the reciprocal y of b refined from
/// constants, possibly hoisted out of the sequence):
///   FCHK P, a, b; FFMA q0, a, y, RZ; FFMA r, q0, -b, a; FFMA q, y, r, q0; @!P BRA L; ...
/// Returns (a, b, q); b is a register or an immediate.
pub fn divrn_at(m: &Machine, pc: u32) -> Option<(Reg, Val, Reg)> {
    divrn_full(m, pc).map(|(a, b, q, _)| (a, b, q))
}

/// The first instruction of the div.rn sequence closing at `pc` that belongs to the quotient's
/// computation (the MUFU.RCP of the divisor, or the first quotient FFMA of the constant form).
pub fn divrn_lo(m: &Machine, pc: u32) -> Option<u32> {
    divrn_full(m, pc).map(|x| x.3)
}

fn divrn_full(m: &Machine, pc: u32) -> Option<(Reg, Val, Reg, u32)> {
    let code = &m.k.code;
    if !matches!(code.get(pc as usize)?.op, Op::Bsync { .. }) {
        return None;
    }
    let bra = (pc.saturating_sub(12)..pc).find(|&p| {
        let i = &code[p as usize];
        matches!(i.op, Op::Bra { target, .. } if target as u32 == pc) && i.guard.neg
    })?;
    let call = (bra + 1..pc).find(|&p| matches!(code[p as usize].op, Op::Call { .. }))?;
    let Op::Call { target } = code[call as usize].op else {
        return None;
    };
    let name =
        m.k.labels
            .iter()
            .find(|(p, _)| *p == target)
            .map(|(_, n)| n.as_str())
            .unwrap_or_default();
    if !(name.contains("div_rn") && name.contains("slowpath")) {
        return None;
    }
    let p = code[bra as usize].guard.reg?;
    // ptxas interleaves other work into the sequence (mslk quantize_nvfp4: 28 instructions
    // from the FCHK to the branch); the FFMA count below still pins the Newton steps.
    let lo = bra.saturating_sub(40);
    // The FCHK closest to the branch (an earlier sequence's FCHK may write the same P).
    let fchk = (lo..bra)
        .rev()
        .find(|&q| matches!(&code[q as usize].op, Op::Fchk { d, .. } if *d == p))?;
    let Op::Fchk {
        a: Val::Reg(a, _),
        b: bval,
        ..
    } = &code[fchk as usize].op
    else {
        return None;
    };
    let (q, first) = match bval {
        Val::Reg(b, _) => {
            let rcp = (lo..bra).rev().find(|&q| {
                matches!(&code[q as usize].op, Op::Mufu { func: MufuFunc::Rcp, a: Val::Reg(x, _), neg: false, abs: false, .. } if x == b)
            })?;
            let ffmas: Vec<u32> = (rcp + 1..bra)
                .filter(|&q| matches!(code[q as usize].op, Op::Ffma { .. }))
                .collect();
            if ffmas.len() != 5 {
                return None;
            }
            let Op::Ffma { d: q, .. } = code[*ffmas.last()? as usize].op else {
                return None;
            };
            (q, rcp)
        }
        Val::Imm(bb) => {
            let (q, first) = divrn_const_tail(code, *a, *bb as u32, fchk, bra)?;
            (q, first)
        }
        _ => return None,
    };
    let mov = (call + 1..pc).any(|x| code[x as usize].op.dst_regs().contains(&q))
        // torch's vectorized kernels: the slow path writes the quotient register itself.
        || (target..code.len())
            .take_while(|&x| !matches!(code[x].op, Op::Ret { .. }))
            .any(|x| code[x].op.dst_regs().contains(&q));
    mov.then_some((*a, bval.clone(), q, first))
}

/// The quotient tail of a div.rn by the constant b (f32 bits) between the FCHK and the branch:
/// `FFMA q0, a, y, RZ; FFMA r, q0, -b, a; FFMA q, y, r, q0`, each the last write of its
/// destination before the next, all plain round-to-nearest. Returns (q, pc of the q0 FFMA).
fn divrn_const_tail(code: &[Inst], a: Reg, b: u32, fchk: u32, bra: u32) -> Option<(Reg, u32)> {
    let plain = |pc: u32| -> Option<(Reg, Val, Val, Val)> {
        match &code[pc as usize].op {
            Op::Ffma {
                d,
                a,
                b,
                c: Some(c),
                neg: [false, nb, false],
                abs: [false, false, false],
                rnd: Rnd::Rn,
                ftz: false,
                sat: false,
            } if code[pc as usize].guard.is_always() => {
                let b = match (b, nb) {
                    (Val::Imm(x), true) => Val::Imm(*x ^ 0x8000_0000),
                    (v, false) => v.clone(),
                    _ => return None,
                };
                Some((*d, a.clone(), b, c.clone()))
            }
            _ => None,
        }
    };
    // The last writer of `r` in [lo, hi).
    let writer = |r: Reg, lo: u32, hi: u32| {
        (lo..hi)
            .rev()
            .find(|&x| code[x as usize].op.dst_regs().contains(&r))
    };
    let last = (fchk + 1..bra)
        .rev()
        .find(|&x| matches!(code[x as usize].op, Op::Ffma { .. }))?;
    let (q, y, rr, q0) = plain(last)?;
    let (Val::Reg(y, _), Val::Reg(rr, _), Val::Reg(q0, _)) = (y, rr, q0) else {
        return None;
    };
    let pr = writer(rr, fchk.saturating_sub(8), last)?;
    let (_, x0, nb, xa) = plain(pr)?;
    let is = |v: &Val, r: Reg| matches!(v, Val::Reg(x, _) if *x == r);
    if !is(&x0, q0) || nb != Val::Imm((b ^ 0x8000_0000) as u64) || !is(&xa, a) {
        return None;
    }
    let pq = writer(q0, fchk.saturating_sub(8), pr)?;
    let (_, u, v, z) = plain(pq)?;
    let ok_ay = (is(&u, a) && is(&v, y)) || (is(&u, y) && is(&v, a));
    if !ok_ay || z != Val::Imm(0) {
        return None;
    }
    // y and a are not rewritten between the quotient's FFMAs.
    if writer(y, pq + 1, last).is_some() || writer(a, pq + 1, pr).is_some() {
        return None;
    }
    Some((q, pq))
}

/// Is the instruction at `pc` the `BSYNC` closing ptxas's `rcp.rn.f32` sequence (an IEEE
/// reciprocal, e.g. CUDA's `1.f / x`)? Shape:
///   IADD3 y, b, 0x1800000; LOP3 x, y, 0x7f800000 (and); ISETP.GT.U32 P, x, 0x1ffffff;
///   ...; @P BRA F; MOVs; CALL <slow path writing q>; BRA L;
///   F: MUFU.RCP r, b; FFMA t, b, r, -1; FADD.FTZ t, -t, -RZ; FFMA q, r, t, r; L: BSYNC
/// The fast path is checked exhaustively against the correctly rounded 1 / b for every b it
/// takes (`rcp_fast_ok`); the slow path's result is PTX's `rcp.rn.f32` (N-SEG). Returns
/// (b, q, pc of the IADD3); `rcprn_pcs` lists the sequence's own instructions.
pub fn rcprn_at(m: &Machine, pc: u32) -> Option<(Reg, Reg, u32)> {
    rcprn_full(m, pc).map(|(b, q, pcs)| (b, q, pcs[0]))
}

/// The instructions of the rcp.rn sequence closing at `pc` (the exponent filter, the branch,
/// the slow-path call and moves, the fast path, the BSYNC), in order.
pub fn rcprn_pcs(m: &Machine, pc: u32) -> Vec<u32> {
    rcprn_full(m, pc).map(|x| x.2).unwrap_or_default()
}

fn rcprn_full(m: &Machine, pc: u32) -> Option<(Reg, Reg, Vec<u32>)> {
    let code = &m.k.code;
    let p = pc as usize;
    if p < 8 || !matches!(code.get(p)?.op, Op::Bsync { .. }) {
        return None;
    }
    // The join label: the BSYNC, or `NOP`s right before it (CuTeDSL).
    let mut e = p;
    while e > 8 && matches!(code[e - 1].op, Op::Nop) && code[e - 1].guard.is_always() {
        e -= 1;
    }
    let plain = |i: usize| code[i].guard.reg.is_none() && !code[i].guard.neg;
    let Op::Ffma {
        d: q,
        a: Val::Reg(r1, _),
        b: Val::Reg(t2, _),
        c: Some(Val::Reg(r2, _)),
        neg: [false, false, false],
        abs: [false, false, false],
        rnd: Rnd::Rn,
        ftz: false,
        sat: false,
        ..
    } = code[e - 1].op
    else {
        return None;
    };
    let Op::Fadd {
        d: t1,
        a: Val::Reg(t0, _),
        b: Val::Imm(0),
        neg: [true, true],
        abs: [false, false],
        rnd: Rnd::Rn,
        ftz: true,
        sat: false,
        ..
    } = code[e - 2].op
    else {
        return None;
    };
    let Op::Ffma {
        d: t,
        a: Val::Reg(b, _),
        b: Val::Reg(r3, _),
        c: Some(Val::Imm(0xbf80_0000)),
        neg: [false, false, false],
        abs: [false, false, false],
        rnd: Rnd::Rn,
        ftz: false,
        sat: false,
        ..
    } = code[e - 3].op
    else {
        return None;
    };
    let Op::Mufu {
        d: r,
        a: Val::Reg(b2, _),
        func: MufuFunc::Rcp,
        fmt: MufuFmt::F32,
        neg: false,
        abs: false,
        ptx: None,
        ..
    } = code[e - 4].op
    else {
        return None;
    };
    // q may be b's or r's register (`FFMA R0, R2, R0, R2`): b is read at the filter.
    if !(r1 == r && r2 == r && r3 == r && t2 == t1 && t1 == t && t0 == t && b2 == b)
        || !(e - 4..p).all(plain)
    {
        return None;
    }
    if !matches!(code[e - 5].op, Op::Bra { target, .. } if target == e) || !plain(e - 5) {
        return None;
    }
    // CALL <slow path>, optionally followed by MOV q, <its result>.
    let call = (p.saturating_sub(8)..e - 5)
        .rev()
        .find(|&x| matches!(code[x].op, Op::Call { .. }))?;
    let Op::Call { target } = code[call].op else {
        return None;
    };
    // The slow path (up to its return, or the move after the call) writes q.
    if !(call + 1..e - 5).any(|x| code[x].op.dst_regs().contains(&q))
        && !(target..code.len())
            .take_while(|&x| !matches!(code[x].op, Op::Ret { .. }))
            .any(|x| code[x].op.dst_regs().contains(&q))
    {
        return None;
    }
    let bra = (p.saturating_sub(16)..call).rev().find(|&x| {
        matches!(code[x].op, Op::Bra { target, .. } if target == e - 4)
            && code[x].guard.reg.is_some()
            && !code[x].guard.neg
    })?;
    let pr = code[bra].guard.reg?;
    // The exponent filter computing P from b (the last writers before the branch).
    let last = |reg: Reg, before: usize| {
        (before.saturating_sub(40)..before)
            .rev()
            .find(|&x| code[x].op.dst_regs().contains(&reg))
    };
    let isetp = last(pr, bra)?;
    let Op::Setp {
        cmp: Cmp::GtU,
        bits: 32,
        a: Val::Reg(x, _),
        b: Val::Imm(0x1ff_ffff),
        and: Pred {
            reg: None,
            neg: false,
        },
        or: false,
        ex: None,
        ..
    } = &code[isetp].op
    else {
        return None;
    };
    let lop = last(*x, isetp)?;
    let Op::Lop3 {
        a: Val::Reg(y, _),
        b: Val::Imm(0x7f80_0000),
        c: Val::Imm(0),
        lut: 0xc0,
        p: None,
        ..
    } = &code[lop].op
    else {
        return None;
    };
    let add = last(*y, lop)?;
    // `IADD3 y, b, 0x1800000, RZ` or `VIADD y, b, 0x1800000`.
    let b3 = match &code[add].op {
        Op::Iadd3 {
            a: Val::Reg(b3, _),
            b: Val::Imm(0x180_0000),
            c: Val::Imm(0),
            neg: [false, false, false],
            not: [false, false, false],
            wide: false,
            cout,
            cin,
            ..
        } if cout.is_empty() && cin.is_empty() => b3,
        Op::Int {
            op: IntOp::Add,
            bits: 32,
            a: Val::Reg(b3, _),
            b: Val::Imm(0x180_0000),
            ..
        } => b3,
        _ => return None,
    };
    // b must not change between the filter and the fast path.
    if *b3 != b || (add + 1..e - 4).any(|x| code[x].op.dst_regs().contains(&b)) {
        return None;
    }
    if ![isetp, lop, add].iter().all(|&i| plain(i)) || !rcp_fast_ok() {
        return None;
    }
    let mut pcs: Vec<u32> = vec![add as u32, lop as u32, isetp as u32];
    pcs.extend(bra as u32..=pc);
    Some((b, q, pcs))
}

/// The fast path of ptxas's rcp.rn.f32 (MUFU.RCP, one Newton step with FFMA / FADD.FTZ /
/// FFMA) equals the correctly rounded 1 / b for every f32 b its exponent filter
/// ((b + 0x1800000) & 0x7f800000 > 0x1ffffff, unsigned) sends there: checked once, over all
/// 2^32 bit patterns, with exec.rs's semantics of the four instructions (N-MUFU, N-FALU).
pub fn rcp_fast_ok() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| {
        let threads = std::thread::available_parallelism()
            .map_or(8, |n| n.get())
            .min(128);
        let next = std::sync::atomic::AtomicU64::new(0);
        let bad = std::sync::atomic::AtomicU64::new(0);
        std::thread::scope(|sc| {
            for _ in 0..threads {
                sc.spawn(|| {
                    loop {
                        let c = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if c >= 1 << 12 {
                            break;
                        }
                        for i in 0..1u64 << 20 {
                            let b = ((c << 20) | i) as u32;
                            if (b.wrapping_add(0x180_0000) & 0x7f80_0000) <= 0x1ff_ffff {
                                continue;
                            }
                            let r = crate::mufu::mufu(MufuFunc::Rcp, MufuFmt::F32, false, b)
                                .unwrap_or(0);
                            let t = fp::f32_fma(b, r, Some(0xbf80_0000), Rnd::Rn, false);
                            let t = fp::f32_fma(
                                t ^ 0x8000_0000,
                                0x3f80_0000,
                                Some(0x8000_0000),
                                Rnd::Rn,
                                true,
                            );
                            let q = fp::f32_fma(r, t, Some(r), Rnd::Rn, false);
                            let want = div_rn(0x3f80_0000, b);
                            let nan = |x: u32| f32::from_bits(x).is_nan();
                            if q != want && !(nan(q) && nan(want)) {
                                bad.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    }
                });
            }
        });
        bad.load(std::sync::atomic::Ordering::Relaxed) == 0
    })
}

/// Registers either path of the rcp.rn sequence closing at `pc` may write after its branch
/// (the slow path's moves and call, the callee up to its return, the fast path).
// ASSUME(N-SEG): see ASSUMPTIONS.md
pub fn rcprn_clobbers(m: &Machine, pc: u32) -> BTreeSet<Reg> {
    let code = &m.k.code;
    let mut out = BTreeSet::new();
    let pcs = rcprn_pcs(m, pc);
    let Some(&bra) = pcs.get(3) else {
        return out;
    };
    for p in bra + 1..pc {
        out.extend(code[p as usize].op.dst_regs());
        if let Op::Call { target } = code[p as usize].op {
            let mut q = target;
            while q < code.len() && !matches!(code[q].op, Op::Ret { .. }) {
                out.extend(code[q].op.dst_regs());
                q += 1;
            }
        }
    }
    // Writes to RZ / URZ are discarded.
    out.retain(|r| !(matches!(r.file, File::R | File::UR) && r.idx == 255));
    out
}

/// Registers the slow path of the div.rn sequence closing at `pc` may write (after its
/// branch: the call's argument moves, the result move, and the slow-path function up to its
/// return).
pub fn divrn_clobbers(m: &Machine, pc: u32) -> BTreeSet<Reg> {
    let code = &m.k.code;
    let mut out = BTreeSet::new();
    let lo = (0..pc)
        .rev()
        .find(|&p| matches!(code[p as usize].op, Op::Bra { target, .. } if target as u32 == pc))
        .unwrap_or(pc);
    for p in lo + 1..pc {
        out.extend(code[p as usize].op.dst_regs());
        if let Op::Call { target } = code[p as usize].op {
            let mut q = target;
            while q < code.len() && !matches!(code[q].op, Op::Ret { .. }) {
                out.extend(code[q].op.dst_regs());
                q += 1;
            }
        }
    }
    out
}

fn reg_name(r: Reg) -> String {
    format!("{:?}{}", r.file, r.idx)
}

impl Machine {
    /// The slice computing register `r` of thread `t` as read by the instruction being
    /// executed (the last entry of its trace).
    pub(crate) fn seg_slice(&self, t: usize, r: Reg) -> Result<SProg, String> {
        let ns = self.num.as_ref().unwrap();
        let trace = ns.trace.get(&(t as u32)).map_or(&[][..], |v| &v[..]);
        let mut s = Slicer {
            m: self,
            t,
            trace,
            args: vec![],
            ops: vec![],
            memo: HashMap::new(),
            assume: Vec::new(),
            depth: 0,
            extra: BTreeSet::new(),
        };
        let out = s.value(r, trace.len().saturating_sub(1))?;
        Ok(SProg {
            args: s.args,
            ops: s.ops,
            out,
            extra_pcs: s.extra,
        })
    }
}

/// Every register the slow path of an executed div.rn sequence may write, other than the
/// quotient, must be dead after the sequence (written before it is read again, along each
/// thread's trace): otherwise inputs that take the slow path would see other values there.
pub fn divrn_dead(m: &Machine) -> Vec<String> {
    let Some(ns) = m.num.as_ref() else {
        return vec![];
    };
    let code = &m.k.code;
    let mut out = BTreeSet::new();
    let mut cache: HashMap<u32, Option<(Reg, BTreeSet<Reg>)>> = HashMap::new();
    for tr in ns.trace.values() {
        for (j, &e) in tr.iter().enumerate() {
            let pc = e & !GUARD_FALSE;
            if !matches!(code[pc as usize].op, Op::Bsync { .. }) {
                continue;
            }
            let info = cache.entry(pc).or_insert_with(|| {
                divrn_at(m, pc)
                    .map(|(_, _, q)| (q, divrn_clobbers(m, pc)))
                    .or_else(|| rcprn_at(m, pc).map(|(_, q, _)| (q, rcprn_clobbers(m, pc))))
            });
            let Some((q, clob)) = info else { continue };
            let mut live: BTreeSet<Reg> = clob.iter().copied().filter(|r| r != q).collect();
            for &f in &tr[j + 1..] {
                if live.is_empty() {
                    break;
                }
                let i = &code[(f & !GUARD_FALSE) as usize];
                let reads: Vec<Reg> =
                    i.op.src_regs()
                        .into_iter()
                        .chain(i.guard.reg)
                        .filter(|r| live.contains(r))
                        .collect();
                for r in reads {
                    out.insert(format!(
                        "{} is written by the div.rn slow path closing at {} and read later at {}",
                        reg_name(r),
                        m.k.describe(pc as usize),
                        m.k.describe((f & !GUARD_FALSE) as usize)
                    ));
                    live.remove(&r);
                }
                if f & GUARD_FALSE == 0 {
                    for r in i.op.dst_regs() {
                        live.remove(&r);
                    }
                }
            }
        }
    }
    out.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(idx: u16) -> Reg {
        Reg { file: File::R, idx }
    }
    fn p(idx: u16) -> Reg {
        Reg { file: File::P, idx }
    }
    fn lop(d: Reg, a: Val, b: Val, c: Val, lut: u8, pr: Option<Reg>) -> Op {
        Op::Lop3 {
            d,
            a,
            b,
            c,
            lut,
            p: pr,
            pq: None,
        }
    }

    /// A flag word's other bits: `R |= 0x8` and `R &= ~0x2` leave bit 0 as it was (skipped
    /// whether they run or not); the bit they write, or an operand that is not an immediate,
    /// is not skipped.
    #[test]
    fn flag_word_writes_keep_other_bits() {
        let w = r(108);
        let or8 = lop(w, Val::Reg(w, 1), Val::Imm(0x8), Val::Imm(0), 0xfc, None);
        let clr2 = lop(
            w,
            Val::Reg(w, 1),
            Val::Imm(0xffff_fffd),
            Val::Imm(0),
            0xc0,
            None,
        );
        assert!(lop3_keeps(&or8, w, 0x1));
        assert!(lop3_keeps(&clr2, w, 0x1 | 0x8));
        assert!(!lop3_keeps(&or8, w, 0x8));
        assert!(!lop3_keeps(&clr2, w, 0x2));
        assert!(!lop3_keeps(&or8, w, u32::MAX));
        // The other operand a register: unknown bits.
        let orr = lop(
            w,
            Val::Reg(w, 1),
            Val::Reg(r(3), 1),
            Val::Imm(0),
            0xfc,
            None,
        );
        assert!(!lop3_keeps(&orr, w, 0x1));
        // Another destination.
        assert!(!lop3_keeps(&or8, r(7), 0x1));
    }

    /// `P = (R & 0x200) != 0` reads bit 9 of R only; `R2 = R | 0x8` demanded in bit 3 reads
    /// nothing of R, in bit 0 bit 0; a register in two operand positions: every bit.
    #[test]
    fn flag_word_reads_demand_their_bits() {
        let w = r(108);
        let test9 = lop(
            r(255),
            Val::Reg(w, 1),
            Val::Imm(0x200),
            Val::Imm(0),
            0xc0,
            Some(p(2)),
        );
        assert_eq!(lop3_demand(&test9, p(2), w, u32::MAX), 0x200);
        let or8 = lop(r(5), Val::Reg(w, 1), Val::Imm(0x8), Val::Imm(0), 0xfc, None);
        assert_eq!(lop3_demand(&or8, r(5), w, 0x8), 0);
        assert_eq!(lop3_demand(&or8, r(5), w, 0x9), 0x1);
        let twice = lop(
            r(5),
            Val::Reg(w, 1),
            Val::Reg(w, 1),
            Val::Imm(0),
            0x80,
            None,
        );
        assert_eq!(lop3_demand(&twice, r(5), w, 0x1), u32::MAX);
        // A register other operand: both of its values.
        let and_r = lop(
            r(5),
            Val::Reg(w, 1),
            Val::Reg(r(3), 1),
            Val::Imm(0),
            0xc0,
            None,
        );
        assert_eq!(lop3_demand(&and_r, r(5), w, 0xf0), 0xf0);
        // Not a LOP3: every bit.
        let mov = Op::Mov {
            d: r(5),
            a: Val::Reg(w, 1),
        };
        assert_eq!(lop3_demand(&mov, r(5), w, 0x1), u32::MAX);
    }

    /// `P2R`, `ISETP` and `VIADD` in slices evaluate as exec.rs does.
    #[test]
    fn predicate_save_ops_evaluate() {
        let vals = |x: u64, y: u64| {
            move |v: &Val| match v {
                Val::Reg(q, _) if q.idx == 1 => x,
                Val::Reg(q, _) if q.idx == 2 => y,
                Val::Imm(c) => *c,
                _ => 0,
            }
        };
        let preds = |pr: &Pred| match pr.reg {
            None => !pr.neg,
            Some(q) => (q.idx == 2 || q.idx == 5) != pr.neg,
        };
        // P2R R, PR, R1, 0x24: bits 2 and 5 from P2, P5, the rest from R1.
        let p2r = Op::P2r {
            d: r(9),
            a: Val::Reg(r(1), 1),
            mask: 0x24,
            uniform: false,
            shift: 0,
        };
        let f = vals(0xffff_ff00, 0);
        assert_eq!(eval_inst(&p2r, r(9), &f, &preds).unwrap(), 0xffff_ff24);
        let f = vals(0x24, 0);
        let p2r6 = Op::P2r {
            d: r(9),
            a: Val::Reg(r(1), 1),
            mask: 0x40,
            uniform: false,
            shift: 0,
        };
        assert_eq!(eval_inst(&p2r6, r(9), &f, &preds).unwrap(), 0x24);
        let setp = |cmp, and| Op::Setp {
            d: p(0),
            cmp,
            bits: 32,
            a: Val::Reg(r(1), 1),
            b: Val::Reg(r(2), 1),
            and,
            or: false,
            xor: false,
            ex: None,
        };
        let pt = Pred {
            reg: None,
            neg: false,
        };
        let f = vals(0xffff_fffe, 3);
        assert_eq!(eval_inst(&setp(Cmp::LtS, pt), p(0), &f, &preds).unwrap(), 1);
        assert_eq!(eval_inst(&setp(Cmp::LtU, pt), p(0), &f, &preds).unwrap(), 0);
        assert_eq!(eval_inst(&setp(Cmp::Ne, pt), p(0), &f, &preds).unwrap(), 1);
        let not_p2 = Pred {
            reg: Some(p(2)),
            neg: true,
        };
        assert_eq!(
            eval_inst(&setp(Cmp::Ne, not_p2), p(0), &f, &preds).unwrap(),
            0
        );
        let viadd = Op::Int {
            op: IntOp::Add,
            bits: 32,
            d: r(3),
            a: Val::Reg(r(1), 1),
            b: Val::Imm(0x180_0000),
        };
        let f = vals(0xff00_0000, 0);
        assert_eq!(eval_inst(&viadd, r(3), &f, &preds).unwrap(), 0x0080_0000);
    }
}

#[cfg(test)]
mod divrn_const_tests {
    use ptxv_ir::*;

    use super::divrn_const_tail;

    fn kernel(body: &str) -> Kernel {
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

    /// ptxas's div.rn by the constant 6 (mslk quantize_nvfp4, sass 0x0d70..0x0e00): the
    /// reciprocal refined from constants, then q0 = a y, r = a - 6 q0, q = y r + q0. A tail
    /// whose remainder uses another divisor, or whose first product reads another register, is
    /// not the sequence.
    #[test]
    fn constant_divisor_tail() {
        let good = "
        /*0000*/                   MOV R0, 0x3e2aaaab ;
        /*0010*/                   IMAD.MOV.U32 R3, RZ, RZ, 0x40c00000 ;
        /*0020*/                   FCHK P0, R44, 6 ;
        /*0030*/                   FFMA R3, R0, -R3, 1 ;
        /*0040*/                   FFMA R22, R3, R0, 0.16666667163372039795 ;
        /*0050*/                   FFMA R45, R44, R22, RZ ;
        /*0060*/                   FFMA R0, R45, -6, R44 ;
        /*0070*/                   FFMA R45, R22, R0, R45 ;
        /*0080*/              @!P0 BRA `(.L_x_3) ;
        /*0090*/                   NOP ;
.L_x_3:
        /*00a0*/                   EXIT ;
";
        let six = 6f32.to_bits();
        let r = |i: u16| Reg {
            file: File::R,
            idx: i,
        };
        let k = kernel(good);
        assert_eq!(
            divrn_const_tail(&k.code, r(44), six, 2, 8),
            Some((r(45), 5))
        );
        let k = kernel(&good.replace("FFMA R0, R45, -6, R44", "FFMA R0, R45, -5, R44"));
        assert_eq!(divrn_const_tail(&k.code, r(44), six, 2, 8), None);
        let k = kernel(&good.replace("FFMA R45, R44, R22, RZ", "FFMA R45, R43, R22, RZ"));
        assert_eq!(divrn_const_tail(&k.code, r(44), six, 2, 8), None);
    }
}

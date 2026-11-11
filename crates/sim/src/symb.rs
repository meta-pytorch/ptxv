// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Concolic input expressions (`--concolic`): along the input-coverage run (coverage.rs), every
//! input-derived register, shared and global byte carries an expression over input leaves
//! next to its concrete value, and every input-dependent predicate a thread computes or obeys
//! (comparisons and tests, guards of instructions, selectors) appends `cond == observed` to
//! that thread's path condition. solve.rs turns the untaken sides of A into goals on these
//! expressions and replays every witness.
//!
//! Expressions (`Node`) are hash-consed bit-vector terms of width 1..=64 over `Input` leaves
//! (bytes of a launch allocation as a load read them) and `Opaque` leaves (values the layer
//! does not model, fixed at their captured value, with a reason):
//! - integer: add, sub, mul, high multiplies, division, shifts, bitwise ops and three-input
//!   LUTs (LOP3 / PLOP3), extension, extraction, concatenation (PRMT, BFE, SHF, pairs),
//!   min / max, comparisons and selects;
//! - floating point: f32 values as their 32-bit encodings through the simulator's own
//!   functions (fp.rs `f32_fma` with its rounding mode and FTZ, `f32_minmax`, `f32_cmp`,
//!   mufu.rs units): the expression is the hardware function, not real arithmetic;
//! - `Inst`: any other register-only instruction, evaluated by running that instruction in
//!   the simulator on the argument values (exact by construction, opaque to the structural
//!   solvers).
//!
//! Every node built for a destination register is checked against the simulator's value on
//! the captured input; a mismatch falls back to `Inst` (then to an opaque leaf), so a modelling
//! slip costs precision, never a wrong expression on the captured input. The input provenance
//! of coverage.rs (`Prov`) is the leaf set of an expression (`Arena::leaves`); registers whose
//! provenance is clean hold no expression.
//!
//! Never a constant of the captured value, whatever the provenance:
//! - a write under an input-dependent guard: `ite(guard, new, old)`, for the threads the
//!   guard holds for and (registers) for those it fails for (the would-be value from the
//!   model, else opaque); a store's bytes likewise for the threads that ran it (stores of
//!   guard-false threads are not followed: their bytes stay as they were);
//! - a value read through an input-dependent address: `AtAddr`, defined only at the captured
//!   address;
//! - a register read before any write: an opaque leaf (`UNDEF`, not input);
//! - a shuffle: the source lane's expression (an input-dependent lane: opaque);
//! - each element of a vector load its own expression; bytes moved by copy units
//!   (cp.async, TMA) keep their sources.
//!
//! The f16 / bf16 element widened to f32 has one form (`Un::H2F`, `Un::B2F`): shifts by 16,
//! masks with 0xffff0000, byte permutes and zero-concatenations that place a 16-bit element
//! in the high half normalize to `B2F`. `SymState::fp_obs` lists, per pc, the
//! input-dependent floating-point operations with their operands after modifiers (FFMA2:
//! one `Fma` per lane).
//!
//! Size cap (`Config::cap`, tree size): a larger value becomes an opaque leaf. Atomic results
//! other than the old value of an add of a constant, votes, reductions, TMEM, tensor-core
//! results, float atomics and memory written by units the layer does not follow are opaque.
//!
//! Off by default: `ProvState::sym` is None and coverage.rs's hooks test only that.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;

use ptxv_ir::Cmp;
use ptxv_ir::FCmp;
use ptxv_ir::FFmt;
use ptxv_ir::File;
use ptxv_ir::HSel;
use ptxv_ir::Lp;
use ptxv_ir::LpRnd;
use ptxv_ir::MufuFmt;
use ptxv_ir::MufuFunc;
use ptxv_ir::Op;
use ptxv_ir::Pred;
use ptxv_ir::Reg;
use ptxv_ir::Rnd;
use ptxv_ir::SReg;
use ptxv_ir::Val;
use serde::Serialize;

use crate::exec::Loc;
use crate::machine::Machine;

/// An expression: index into `Arena::nodes`. Children always have smaller indices.
pub type E = u32;

/// One-operand operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Un {
    Not,
    Neg,
    Popc,
    Brev,
    /// f32 sign flip / clear (FADD / FFMA / FSETP operand modifiers `-x`, `|x|`).
    FNeg,
    FAbs,
    /// f32 saturation to [0, 1] (`.SAT`, fp.rs `f32_sat`).
    FSat,
    /// f16 (16 bits) widened exactly to f32 (32 bits; a NaN becomes 0x7fffffff, as HADD2.F32).
    H2F,
    /// bf16 (16 bits) widened exactly to f32 (32 bits: the encoding shifted left by 16). The
    /// canonical form of `concat(h, 0)`, `zext(h) << 16`, `x & 0xffff0000`, `cvt.f32.bf16`.
    B2F,
    /// f64 sign flip / clear that leaves a NaN as it is (DFMA operand modifiers).
    DNegK,
    DAbsK,
}

/// Two-operand bit-vector operations (both operands and the result of width `w`). Shifts:
/// an amount of `w` or more gives 0 (Shl, LShr) or the sign (AShr), as the simulator's `Int`.
/// Division by zero has no value (evaluation fails).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Bin {
    Add,
    Sub,
    Mul,
    MulHiU,
    MulHiS,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
    UDiv,
    SDiv,
    URem,
    SRem,
    UMin,
    UMax,
    SMin,
    SMax,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Node {
    Const {
        w: u8,
        v: u64,
    },
    /// `bytes` bytes of input allocation `alloc` at byte offset `off`, little-endian: an
    /// element of the launch's input as the kernel read it (width 8 * bytes).
    Input {
        alloc: u32,
        off: u64,
        bytes: u8,
    },
    /// A value the layer does not model, fixed at its captured value (`Arena::opaque[id]`).
    Opaque {
        w: u8,
        id: u32,
    },
    /// A special register whose value depends on the CTA's or warp's placement (`%ctaid`,
    /// `%clusterid`, `%cluster_ctaid`, `%cluster_ctarank`, `%warpid`, shared-window addresses
    /// with the cluster rank): one leaf per CTA (`key`; per warp for `%warpid`), never folded
    /// into a constant; solvers keep it at its captured value (64 bits wide).
    Sreg {
        s: SReg,
        key: u32,
    },
    Un {
        op: Un,
        w: u8,
        a: E,
    },
    Bin {
        op: Bin,
        w: u8,
        a: E,
        b: E,
    },
    /// Bitwise three-input function: result bit = `lut` bit (a << 2 | b << 1 | c).
    Lop3 {
        lut: u8,
        w: u8,
        a: E,
        b: E,
        c: E,
    },
    /// Integer comparison of two values of one width (1 bit).
    Cmp {
        cmp: Cmp,
        a: E,
        b: E,
    },
    Ite {
        c: E,
        a: E,
        b: E,
    },
    Zext {
        w: u8,
        a: E,
    },
    Sext {
        w: u8,
        a: E,
    },
    Extract {
        lo: u8,
        w: u8,
        a: E,
    },
    Concat {
        hi: E,
        lo: E,
    },
    /// f32 `a * b (+ c)` rounded once (`fp::f32_fma`; FADD is `a * 1.0 + b`, FMUL `c` None).
    Fma {
        a: E,
        b: E,
        c: Option<E>,
        rnd: Rnd,
        ftz: bool,
    },
    /// f32 min / max (`fp::f32_minmax`).
    FMinMax {
        min: bool,
        nan: bool,
        ftz: bool,
        a: E,
        b: E,
    },
    /// f32 comparison (`fp::f32_cmp`, 1 bit).
    FCmp {
        cmp: FCmp,
        ftz: bool,
        a: E,
        b: E,
    },
    /// MUFU / PTX approx unit on a 32-bit operand (`mufu::mufu`, `mufu::ptx_approx`).
    Mufu {
        func: MufuFunc,
        fmt: MufuFmt,
        hi: bool,
        ptx: Option<bool>,
        a: E,
    },
    /// f64 `a * b (+ c)` rounded once (`fp::f64_fma`, DFMA / DADD / DMUL).
    DFma {
        a: E,
        b: E,
        c: Option<E>,
        rnd: Rnd,
    },
    /// f64 comparison (`fp::f64_cmp`, 1 bit).
    DCmp {
        cmp: FCmp,
        a: E,
        b: E,
    },
    /// One 16-bit lane of HFMA2: `a * b + c` rounded once to nearest even, f16
    /// (`fp::f16_fma_rn`) or bf16 (`fp::bf16_fma`).
    HFma {
        bf16: bool,
        a: E,
        b: E,
        c: E,
    },
    /// f32 narrowed to bf16 (GPU converter, `fp::f32_to_bf16_rn_gpu`) or f16
    /// (`fp::f32_to_f16_rn`), round to nearest even (F2FP.PACK_AB lanes; 16 bits).
    Narrow {
        bf16: bool,
        a: E,
    },
    /// Float-to-float conversion between IEEE formats (`fp::fcvt`; FRND with `int`); the
    /// operand is the source encoding (`from` bits), the result the target's.
    FCvt {
        from: FFmt,
        to: FFmt,
        rnd: Rnd,
        ftz: bool,
        int: bool,
        a: E,
    },
    /// SASS F2F between f32 and f64 (`to64`: f32 -> f64 exactly; else f64 -> f32 per `rnd`).
    F2F {
        to64: bool,
        rnd: Rnd,
        a: E,
    },
    /// Integer (64-bit operand, already extended) to f32 / f64 (`fp::int_to_f32` / `_f64`).
    I2F {
        signed: bool,
        rnd: Rnd,
        f64: bool,
        a: E,
    },
    /// f32 / f64 to an integer of `bits` bits, saturating, NaN as the simulator's F2I
    /// (32-bit result; 64 for `bits` 64).
    F2I {
        signed: bool,
        bits: u8,
        rnd: Rnd,
        ftz: bool,
        f64: bool,
        a: E,
    },
    /// `cvt` / F2FP narrow pack of two or four lanes (`lowp::pack`, 32 bits).
    Pack {
        half: Option<Lp>,
        to: Lp,
        rnd: LpRnd,
        sat: bool,
        relu: bool,
        a: E,
        b: E,
        c: E,
    },
    /// The value `val` a load read through the input-dependent address `addr` (captured value
    /// `at`): defined only while `addr` evaluates to `at` (elsewhere the bytes of another
    /// address, unknown here: evaluation fails).
    AtAddr {
        addr: E,
        at: u64,
        val: E,
    },
    /// Destination `out` (`Op::dst_regs` index) of instruction `pc` run by thread `t` on
    /// `args` (one per `Op::src_regs` entry), evaluated by the simulator.
    Inst {
        pc: u32,
        t: u32,
        out: u8,
        w: u8,
        args: Vec<E>,
    },
}

/// Why a value is opaque.
#[derive(Clone, Debug, Serialize)]
pub struct OpaqueLeaf {
    pub why: String,
    pub pc: u32,
}

/// Opaque reason of a register read before any write: never a constant, whatever the
/// provenance says.
pub const UNDEF: &str = "register read before any write (undefined)";

pub fn mask(w: u8) -> u64 {
    if w >= 64 { u64::MAX } else { (1u64 << w) - 1 }
}

fn sx(v: u64, w: u8) -> i64 {
    let w = w.clamp(1, 64) as u32;
    ((v << (64 - w)) as i64) >> (64 - w)
}

pub fn cmp_eval(c: Cmp, w: u8, x: u64, y: u64) -> bool {
    let (x, y) = (x & mask(w), y & mask(w));
    let (a, b) = (sx(x, w), sx(y, w));
    match c {
        Cmp::Eq => x == y,
        Cmp::Ne => x != y,
        Cmp::LtU => x < y,
        Cmp::LeU => x <= y,
        Cmp::GtU => x > y,
        Cmp::GeU => x >= y,
        Cmp::LtS => a < b,
        Cmp::LeS => a <= b,
        Cmp::GtS => a > b,
        Cmp::GeS => a >= b,
    }
}

fn lop3(lut: u8, x: u64, y: u64, z: u64) -> u64 {
    let mut r = 0u64;
    for i in 0..8 {
        if (lut >> i) & 1 == 1 {
            let fa = if i & 4 != 0 { x } else { !x };
            let fb = if i & 2 != 0 { y } else { !y };
            let fc = if i & 1 != 0 { z } else { !z };
            r |= fa & fb & fc;
        }
    }
    r
}

pub fn bin_eval(op: Bin, w: u8, x: u64, y: u64) -> Option<u64> {
    let m = mask(w);
    let (x, y) = (x & m, y & m);
    let ww = w as u64;
    let r = match op {
        Bin::Add => x.wrapping_add(y),
        Bin::Sub => x.wrapping_sub(y),
        Bin::Mul => x.wrapping_mul(y),
        Bin::MulHiU => ((x as u128 * y as u128) >> w) as u64,
        Bin::MulHiS => ((sx(x, w) as i128 * sx(y, w) as i128) >> w) as u64,
        Bin::And => x & y,
        Bin::Or => x | y,
        Bin::Xor => x ^ y,
        Bin::Shl => {
            if y >= ww {
                0
            } else {
                x << y
            }
        }
        Bin::LShr => {
            if y >= ww {
                0
            } else {
                x >> y
            }
        }
        Bin::AShr => (sx(x, w) >> y.min(ww - 1)) as u64,
        Bin::UDiv | Bin::SDiv | Bin::URem | Bin::SRem if y == 0 => return None,
        Bin::UDiv => x / y,
        Bin::URem => x % y,
        Bin::SDiv => sx(x, w).wrapping_div(sx(y, w)) as u64,
        Bin::SRem => sx(x, w).wrapping_rem(sx(y, w)) as u64,
        Bin::UMin => x.min(y),
        Bin::UMax => x.max(y),
        Bin::SMin => sx(x, w).min(sx(y, w)) as u64,
        Bin::SMax => sx(x, w).max(sx(y, w)) as u64,
    };
    Some(r & m)
}

pub fn un_eval(op: Un, w: u8, x: u64) -> u64 {
    let m = mask(w);
    let x = x & m;
    (match op {
        Un::Not => !x,
        Un::Neg => x.wrapping_neg(),
        Un::Popc => x.count_ones() as u64,
        Un::Brev => x.reverse_bits() >> (64 - w as u32),
        Un::FNeg => x ^ 0x8000_0000,
        Un::FAbs => x & 0x7fff_ffff,
        Un::FSat => crate::fp::f32_sat(x as u32) as u64,
        Un::H2F => {
            let v = crate::fp::f16_to_f64(x as u16);
            if v.is_nan() {
                0x7fff_ffff
            } else {
                (v as f32).to_bits() as u64
            }
        }
        Un::B2F => (x & 0xffff) << 16,
        Un::DNegK | Un::DAbsK if x & !(1 << 63) > 0x7ff0_0000_0000_0000 => x,
        Un::DNegK => x ^ 1 << 63,
        Un::DAbsK => x & !(1 << 63),
    }) & m
}

/// The simulator's F2I (both the f32 and the f64 forms).
pub fn f2i_eval(v: u64, signed: bool, bits: u8, rnd: Rnd, ftz: bool, f64: bool) -> u64 {
    if f64 {
        let x = f64::from_bits(v);
        let x = match rnd {
            Rnd::Rz => x.trunc(),
            Rnd::Rn => x.round_ties_even(),
            Rnd::Rm => x.floor(),
            Rnd::Rp => x.ceil(),
        };
        return match (bits, signed) {
            (64, _) if x.is_nan() => 1 << 63,
            (_, _) if x.is_nan() => 1 << 31,
            (64, true) => x as i64 as u64,
            (64, false) => x as u64,
            (_, true) => x as i32 as u32 as u64,
            (_, false) => x as u32 as u64,
        };
    }
    let mut u = v as u32;
    if ftz {
        u = crate::fp::f32_ftz(u);
    }
    let x = f32::from_bits(u);
    let x = match rnd {
        Rnd::Rz => x,
        Rnd::Rn => x.round_ties_even(),
        Rnd::Rm => x.floor(),
        Rnd::Rp => x.ceil(),
    };
    match (bits, signed) {
        (64, _) if x.is_nan() => 1 << 63,
        (64, true) => x as i64 as u64,
        (64, false) => x as u64,
        (16, true) => x as i16 as i32 as u32 as u64,
        (16, false) => x as u16 as u64,
        (_, true) => x as i32 as u32 as u64,
        (_, false) => x as u32 as u64,
    }
}

pub fn mufu_eval(func: MufuFunc, fmt: MufuFmt, hi: bool, ptx: Option<bool>, x: u32) -> Option<u32> {
    match ptx {
        None => crate::mufu::mufu(func, fmt, hi, x),
        Some(ftz) => crate::mufu::ptx_approx(func, fmt, ftz, x),
    }
}

/// The value of `n` from its children's values (`get`), None for leaves, `Inst` and
/// undefined results.
pub fn eval_node(n: &Node, width: &dyn Fn(E) -> u8, get: &dyn Fn(E) -> u64) -> Option<u64> {
    Some(match *n {
        Node::Const { v, .. } => v,
        Node::Input { .. } | Node::Opaque { .. } | Node::Sreg { .. } | Node::Inst { .. } => {
            return None;
        }
        Node::Un { op, w, a } => un_eval(op, w, get(a)),
        Node::Bin { op, w, a, b } => bin_eval(op, w, get(a), get(b))?,
        Node::Lop3 { lut, w, a, b, c } => lop3(lut, get(a), get(b), get(c)) & mask(w),
        Node::Cmp { cmp, a, b } => cmp_eval(cmp, width(a), get(a), get(b)) as u64,
        Node::Ite { c, a, b } => {
            if get(c) & 1 == 1 {
                get(a)
            } else {
                get(b)
            }
        }
        Node::Zext { a, .. } => get(a),
        Node::Sext { w, a } => (sx(get(a), width(a)) as u64) & mask(w),
        Node::Extract { lo, w, a } => (get(a) >> lo) & mask(w),
        Node::Concat { hi, lo } => {
            let wl = width(lo);
            if wl >= 64 {
                get(lo)
            } else {
                (get(hi) << wl) | get(lo)
            }
        }
        Node::Fma { a, b, c, rnd, ftz } => crate::fp::f32_fma(
            get(a) as u32,
            get(b) as u32,
            c.map(|c| get(c) as u32),
            rnd,
            ftz,
        ) as u64,
        Node::FMinMax {
            min,
            nan,
            ftz,
            a,
            b,
        } => crate::fp::f32_minmax(get(a) as u32, get(b) as u32, min, nan, ftz) as u64,
        Node::FCmp { cmp, ftz, a, b } => {
            crate::fp::f32_cmp(get(a) as u32, get(b) as u32, cmp, ftz) as u64
        }
        Node::Mufu {
            func,
            fmt,
            hi,
            ptx,
            a,
        } => mufu_eval(func, fmt, hi, ptx, get(a) as u32)? as u64,
        Node::DFma { a, b, c, rnd } => crate::fp::f64_fma(get(a), get(b), c.map(get), rnd),
        Node::DCmp { cmp, a, b } => crate::fp::f64_cmp(get(a), get(b), cmp) as u64,
        Node::HFma { bf16, a, b, c } => {
            let (x, y, z) = (get(a) as u16, get(b) as u16, get(c) as u16);
            (if bf16 {
                crate::fp::bf16_fma(x, y, z)
            } else {
                crate::fp::f16_fma_rn(x, y, z)
            }) as u64
        }
        Node::Narrow { bf16, a } => {
            (if bf16 {
                crate::fp::f32_to_bf16_rn_gpu(get(a) as u32)
            } else {
                crate::fp::f32_to_f16_rn(get(a) as u32)
            }) as u64
        }
        Node::FCvt {
            from,
            to,
            rnd,
            ftz,
            int,
            a,
        } => crate::fp::fcvt(get(a), from, to, rnd, ftz, int),
        Node::F2F { to64, rnd, a } => {
            if to64 {
                (f32::from_bits(get(a) as u32) as f64).to_bits()
            } else {
                crate::fp::f64_to_f32(f64::from_bits(get(a)), rnd).to_bits() as u64
            }
        }
        Node::I2F {
            signed,
            rnd,
            f64,
            a,
        } => {
            if f64 {
                crate::fp::int_to_f64(get(a), signed, rnd).to_bits()
            } else {
                crate::fp::int_to_f32(get(a), signed, rnd).to_bits() as u64
            }
        }
        Node::F2I {
            signed,
            bits,
            rnd,
            ftz,
            f64,
            a,
        } => f2i_eval(get(a), signed, bits, rnd, ftz, f64),
        Node::Pack {
            half,
            to,
            rnd,
            sat,
            relu,
            a,
            b,
            c,
        } => crate::lowp::pack(
            get(a) as u32,
            get(b) as u32,
            get(c) as u32,
            half,
            to,
            rnd,
            sat,
            relu,
        ) as u64,
        Node::AtAddr { addr, at, val } => {
            if get(addr) != at {
                return None;
            }
            get(val)
        }
    })
}

/// Children of a node.
pub fn kids(n: &Node) -> Vec<E> {
    match n {
        Node::Const { .. } | Node::Input { .. } | Node::Opaque { .. } | Node::Sreg { .. } => vec![],
        Node::Un { a, .. }
        | Node::Zext { a, .. }
        | Node::Sext { a, .. }
        | Node::Extract { a, .. } => vec![*a],
        Node::Mufu { a, .. } | Node::Narrow { a, .. } => vec![*a],
        Node::FCvt { a, .. } | Node::F2F { a, .. } | Node::I2F { a, .. } | Node::F2I { a, .. } => {
            vec![*a]
        }
        Node::DCmp { a, b, .. } => vec![*a, *b],
        Node::AtAddr { addr, val, .. } => vec![*addr, *val],
        Node::HFma { a, b, c, .. } | Node::Pack { a, b, c, .. } => vec![*a, *b, *c],
        Node::DFma { a, b, c, .. } => {
            let mut v = vec![*a, *b];
            v.extend(*c);
            v
        }
        Node::Bin { a, b, .. }
        | Node::Cmp { a, b, .. }
        | Node::FMinMax { a, b, .. }
        | Node::FCmp { a, b, .. } => {
            vec![*a, *b]
        }
        Node::Concat { hi, lo } => vec![*hi, *lo],
        Node::Lop3 { a, b, c, .. } | Node::Ite { c: a, a: b, b: c } => vec![*a, *b, *c],
        Node::Fma { a, b, c, .. } => {
            let mut v = vec![*a, *b];
            v.extend(*c);
            v
        }
        Node::Inst { args, .. } => args.clone(),
    }
}

/// Hash-consed expressions with their captured values.
#[derive(Clone, Debug)]
pub struct Arena {
    pub nodes: Vec<Node>,
    pub width: Vec<u8>,
    /// The value on the captured input.
    pub conc: Vec<u64>,
    size: Vec<u32>,
    /// The cone holds a placement-dependent special register (`Node::Sreg`).
    pub placed: Vec<bool>,
    /// The cone holds an input or opaque leaf (input-derived).
    pub dep: Vec<bool>,
    /// The cone holds the value of a register read before any write (`UNDEF`).
    pub undef: Vec<bool>,
    /// The cone holds an input leaf.
    pub inp: Vec<bool>,
    /// An input leaf reaches the value through data (not only through the condition of an
    /// `Ite`, i.e. a select or guard).
    pub dinp: Vec<bool>,
    map: HashMap<Node, E>,
    pub opaque: Vec<OpaqueLeaf>,
    /// Tree size above which a new node becomes an opaque leaf.
    pub cap: u32,
    /// The pc the current node is built for (opaque reasons).
    pub pc: u32,
}

impl Default for Arena {
    fn default() -> Self {
        Arena {
            nodes: vec![],
            width: vec![],
            conc: vec![],
            size: vec![],
            placed: vec![],
            dep: vec![],
            undef: vec![],
            inp: vec![],
            dinp: vec![],
            map: HashMap::new(),
            opaque: vec![],
            cap: 2048,
            pc: 0,
        }
    }
}

impl Arena {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn push(&mut self, n: Node, w: u8, v: u64, size: u32) -> E {
        if let Some(&e) = self.map.get(&n) {
            return e;
        }
        let e = self.nodes.len() as E;
        let placed =
            matches!(n, Node::Sreg { .. }) || kids(&n).iter().any(|&k| self.placed[k as usize]);
        self.placed.push(placed);
        let dep = match n {
            Node::Input { .. } => true,
            // A register read before any write is not input (its provenance is clean).
            Node::Opaque { id, .. } => self.opaque[id as usize].why != UNDEF,
            _ => kids(&n).iter().any(|&k| self.dep[k as usize]),
        };
        self.dep.push(dep);
        let undef = matches!(n, Node::Opaque { id, .. } if self.opaque[id as usize].why == UNDEF)
            || kids(&n).iter().any(|&k| self.undef[k as usize]);
        self.undef.push(undef);
        let inp = matches!(n, Node::Input { .. }) || kids(&n).iter().any(|&k| self.inp[k as usize]);
        self.inp.push(inp);
        let dinp = match n {
            Node::Input { .. } => true,
            Node::Ite { a, b, .. } => self.dinp[a as usize] || self.dinp[b as usize],
            _ => kids(&n).iter().any(|&k| self.dinp[k as usize]),
        };
        self.dinp.push(dinp);
        self.nodes.push(n.clone());
        self.width.push(w);
        self.conc.push(v & mask(w));
        self.size.push(size);
        self.map.insert(n, e);
        e
    }

    pub fn konst(&mut self, w: u8, v: u64) -> E {
        let v = v & mask(w);
        self.push(Node::Const { w, v }, w, v, 1)
    }

    pub fn is_const(&self, e: E) -> Option<u64> {
        match self.nodes[e as usize] {
            Node::Const { v, .. } => Some(v),
            _ => None,
        }
    }

    pub fn input(&mut self, alloc: u32, off: u64, bytes: u8, v: u64) -> E {
        self.push(Node::Input { alloc, off, bytes }, 8 * bytes, v, 1)
    }

    pub fn opaque(&mut self, w: u8, v: u64, why: impl Into<String>) -> E {
        let id = self.opaque.len() as u32;
        self.opaque.push(OpaqueLeaf {
            why: why.into(),
            pc: self.pc,
        });
        self.push(Node::Opaque { w, id }, w, v, 1)
    }

    pub fn sreg(&mut self, s: SReg, key: u32, v: u64) -> E {
        self.push(Node::Sreg { s, key }, 64, v, 1)
    }

    pub fn inst(&mut self, pc: u32, t: u32, out: u8, w: u8, args: Vec<E>, v: u64) -> E {
        let size = 1 + args
            .iter()
            .map(|&a| self.size[a as usize])
            .fold(0u32, |x, y| x.saturating_add(y));
        if size > self.cap {
            return self.opaque(w, v, "expression size cap");
        }
        self.push(
            Node::Inst {
                pc,
                t,
                out,
                w,
                args,
            },
            w,
            v,
            size,
        )
    }

    /// The node, its value computed from its children's captured values (constants folded).
    fn mk(&mut self, n: Node, w: u8) -> E {
        let ks = kids(&n);
        if ks.iter().all(|&k| self.is_const(k).is_some()) && !ks.is_empty() {
            if let Some(v) = eval_node(&n, &|e| self.width[e as usize], &|e| self.conc[e as usize])
            {
                return self.konst(w, v);
            }
        }
        if let Some(&e) = self.map.get(&n) {
            return e;
        }
        let v = match eval_node(&n, &|e| self.width[e as usize], &|e| self.conc[e as usize]) {
            Some(v) => v,
            // An undefined result (division by zero) on the captured input: opaque.
            None => return self.opaque(w, 0, "undefined result"),
        };
        let size = 1 + ks
            .iter()
            .map(|&a| self.size[a as usize])
            .fold(0u32, |x, y| x.saturating_add(y));
        if size > self.cap {
            return self.opaque(w, v, "expression size cap");
        }
        self.push(n, w, v, size)
    }

    /// A node of the kinds without a dedicated constructor (f64, 16-bit lanes, narrowing).
    pub fn mk_pub(&mut self, n: Node, w: u8) -> E {
        self.mk(n, w)
    }

    pub fn w(&self, e: E) -> u8 {
        self.width[e as usize]
    }

    pub fn un(&mut self, op: Un, a: E) -> E {
        let w = self.w(a);
        if matches!(op, Un::H2F | Un::B2F) {
            let h = self.extract(0, 16, a);
            if op == Un::B2F {
                if let Some(x) = self.is_const(h) {
                    return self.konst(32, x << 16);
                }
            }
            return self.mk(Node::Un { op, w: 32, a: h }, 32);
        }
        if let Node::Un { op: o2, a: x, .. } = self.nodes[a as usize] {
            if o2 == op && matches!(op, Un::Not | Un::Neg | Un::FNeg) {
                return x;
            }
        }
        self.mk(Node::Un { op, w, a }, w)
    }

    pub fn bin(&mut self, op: Bin, a: E, b: E) -> E {
        let w = self.w(a);
        debug_assert_eq!(w, self.w(b), "{op:?}");
        let (ca, cb) = (self.is_const(a), self.is_const(b));
        let m = mask(w);
        // Identities.
        match (op, ca, cb) {
            (
                Bin::Add | Bin::Or | Bin::Xor | Bin::Sub | Bin::Shl | Bin::LShr | Bin::AShr,
                _,
                Some(0),
            ) => return a,
            (Bin::Add | Bin::Or | Bin::Xor, Some(0), _) => return b,
            (Bin::Mul, _, Some(1)) | (Bin::UDiv | Bin::SDiv, _, Some(1)) => return a,
            (Bin::Mul, Some(1), _) => return b,
            (Bin::Mul | Bin::And, _, Some(0)) => return self.konst(w, 0),
            (Bin::Mul | Bin::And, Some(0), _) => return self.konst(w, 0),
            (Bin::And, _, Some(x)) if x == m => return a,
            (Bin::And, Some(x), _) if x == m => return b,
            _ => {}
        }
        // A bf16 element widened to f32 (`B2F`): `x << 16`, `x * 0x10000`, `x & 0xffff0000`.
        if w == 32 {
            match (op, ca, cb) {
                (Bin::Shl, _, Some(16)) | (Bin::Mul, _, Some(0x10000)) => {
                    let h = self.extract(0, 16, a);
                    return self.un(Un::B2F, h);
                }
                (Bin::Mul, Some(0x10000), _) => {
                    let h = self.extract(0, 16, b);
                    return self.un(Un::B2F, h);
                }
                (Bin::And, _, Some(0xffff_0000)) => {
                    let h = self.extract(16, 16, a);
                    return self.un(Un::B2F, h);
                }
                (Bin::And, Some(0xffff_0000), _) => {
                    let h = self.extract(16, 16, b);
                    return self.un(Un::B2F, h);
                }
                _ => {}
            }
        }
        // Constants second for commutative operations.
        let (a, b) = match op {
            Bin::Add | Bin::Mul | Bin::And | Bin::Or | Bin::Xor if ca.is_some() && cb.is_none() => {
                (b, a)
            }
            _ => (a, b),
        };
        self.mk(Node::Bin { op, w, a, b }, w)
    }

    pub fn lop3(&mut self, lut: u8, a: E, b: E, c: E) -> E {
        let w = self.w(a);
        let m = mask(w);
        // Fold constant all-zero / all-one inputs into the table, then recognize one- and
        // two-input functions.
        let mut lut = lut;
        let ins = [a, b, c];
        let mut used = [true; 3];
        for (k, &x) in ins.iter().enumerate() {
            let bit = 2 - k; // a: index bit 2, b: bit 1, c: bit 0
            let v = match self.is_const(x) {
                Some(0) => false,
                Some(v) if v == m => true,
                _ => continue,
            };
            let mut nl = 0u8;
            for i in 0..8u8 {
                let j = if v { i | (1 << bit) } else { i & !(1 << bit) };
                if (lut >> j) & 1 == 1 {
                    nl |= 1 << i;
                }
            }
            lut = nl;
            used[k] = false;
        }
        // Inputs the table does not depend on.
        for (k, u) in used.iter_mut().enumerate() {
            let bit = 2 - k;
            if (0..8u8).all(|i| ((lut >> i) & 1) == ((lut >> (i ^ (1 << bit))) & 1)) {
                *u = false;
            }
        }
        let live: Vec<usize> = (0..3).filter(|&k| used[k]).collect();
        // The function on the live inputs, as a truth table over their bits.
        let f = |vals: &[u8]| -> u8 {
            let mut i = 0u8;
            for (j, &k) in live.iter().enumerate() {
                i |= vals[j] << (2 - k);
            }
            (lut >> i) & 1
        };
        match live.len() {
            0 => return self.konst(w, if lut & 1 == 1 { m } else { 0 }),
            1 => {
                let x = ins[live[0]];
                return match (f(&[0]), f(&[1])) {
                    (0, 1) => x,
                    (1, 0) => self.un(Un::Not, x),
                    (0, 0) => self.konst(w, 0),
                    _ => self.konst(w, m),
                };
            }
            2 => {
                let (x, y) = (ins[live[0]], ins[live[1]]);
                let t = [f(&[0, 0]), f(&[0, 1]), f(&[1, 0]), f(&[1, 1])];
                match t {
                    [0, 0, 0, 1] => return self.bin(Bin::And, x, y),
                    [0, 1, 1, 1] => return self.bin(Bin::Or, x, y),
                    [0, 1, 1, 0] => return self.bin(Bin::Xor, x, y),
                    [0, 0, 1, 0] => {
                        let ny = self.un(Un::Not, y);
                        return self.bin(Bin::And, x, ny);
                    }
                    [0, 1, 0, 0] => {
                        let nx = self.un(Un::Not, x);
                        return self.bin(Bin::And, nx, y);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        self.mk(Node::Lop3 { lut, w, a, b, c }, w)
    }

    pub fn cmp(&mut self, cmp: Cmp, a: E, b: E) -> E {
        self.mk(Node::Cmp { cmp, a, b }, 1)
    }

    pub fn ite(&mut self, c: E, a: E, b: E) -> E {
        if a == b {
            return a;
        }
        if let Some(v) = self.is_const(c) {
            return if v & 1 == 1 { a } else { b };
        }
        let w = self.w(a);
        if w == 1 {
            if let (Some(1), Some(0)) = (self.is_const(a), self.is_const(b)) {
                return c;
            }
        }
        self.mk(Node::Ite { c, a, b }, w)
    }

    pub fn zext(&mut self, w: u8, a: E) -> E {
        let wa = self.w(a);
        if wa == w {
            return a;
        }
        if wa > w {
            return self.extract(0, w, a);
        }
        if let Node::Zext { a: x, .. } = self.nodes[a as usize] {
            return self.zext(w, x);
        }
        self.mk(Node::Zext { w, a }, w)
    }

    pub fn sext(&mut self, w: u8, a: E) -> E {
        let wa = self.w(a);
        if wa == w {
            return a;
        }
        if wa > w {
            return self.extract(0, w, a);
        }
        if let Node::Sext { a: x, .. } = self.nodes[a as usize] {
            return self.sext(w, x);
        }
        self.mk(Node::Sext { w, a }, w)
    }

    pub fn extract(&mut self, lo: u8, w: u8, a: E) -> E {
        let wa = self.w(a);
        if lo == 0 && w == wa {
            return a;
        }
        if lo >= wa {
            return self.konst(w, 0);
        }
        if lo as u32 + w as u32 > wa as u32 {
            // Bits above the value: zero.
            let x = self.extract(lo, wa - lo, a);
            return self.zext(w, x);
        }
        match self.nodes[a as usize].clone() {
            Node::Extract { lo: l2, a: x, .. } => return self.extract(lo + l2, w, x),
            Node::Zext { a: x, .. } => {
                let wx = self.w(x);
                if lo + w <= wx {
                    return self.extract(lo, w, x);
                }
                if lo >= wx {
                    return self.konst(w, 0);
                }
                let y = self.extract(lo, wx - lo, x);
                return self.zext(w, y);
            }
            Node::Sext { a: x, .. } => {
                let wx = self.w(x);
                if lo + w <= wx {
                    return self.extract(lo, w, x);
                }
                if lo == 0 {
                    return self.sext(w, x);
                }
            }
            Node::Concat { hi, lo: l } => {
                let wl = self.w(l);
                if lo + w <= wl {
                    return self.extract(lo, w, l);
                }
                if lo >= wl {
                    return self.extract(lo - wl, w, hi);
                }
                let a1 = self.extract(lo, wl - lo, l);
                let a2 = self.extract(0, w - (wl - lo), hi);
                return self.concat(a2, a1);
            }
            // Low bits of ring and bitwise operations depend on the operands' low bits only.
            Node::Bin { op, a: x, b: y, .. }
                if lo == 0
                    && matches!(
                        op,
                        Bin::Add | Bin::Sub | Bin::Mul | Bin::And | Bin::Or | Bin::Xor
                    ) =>
            {
                let (x, y) = (self.extract(0, w, x), self.extract(0, w, y));
                return self.bin(op, x, y);
            }
            Node::Bin { op, a: x, b: y, .. } if matches!(op, Bin::And | Bin::Or | Bin::Xor) => {
                let (x, y) = (self.extract(lo, w, x), self.extract(lo, w, y));
                return self.bin(op, x, y);
            }
            // The low bits of a shift left by a constant: the shift of the low bits.
            Node::Bin {
                op: Bin::Shl,
                a: x,
                b: y,
                ..
            } if lo == 0 && self.is_const(y).is_some_and(|k| k < w as u64) => {
                let k = self.is_const(y).unwrap_or(0);
                let x = self.extract(0, w, x);
                let k = self.konst(w, k);
                return self.bin(Bin::Shl, x, k);
            }
            Node::Un {
                op: op @ (Un::Not | Un::Neg),
                a: x,
                ..
            } if lo == 0 || op == Un::Not => {
                let x = self.extract(lo, w, x);
                return self.un(op, x);
            }
            Node::Lop3 {
                lut,
                a: x,
                b: y,
                c: z,
                ..
            } => {
                let (x, y, z) = (
                    self.extract(lo, w, x),
                    self.extract(lo, w, y),
                    self.extract(lo, w, z),
                );
                return self.lop3(lut, x, y, z);
            }
            Node::Ite { c, a: x, b: y } => {
                let (x, y) = (self.extract(lo, w, x), self.extract(lo, w, y));
                return self.ite(c, x, y);
            }
            // `B2F(x)` is `concat(x, 0:16)`.
            Node::Un {
                op: Un::B2F, a: x, ..
            } => {
                if lo + w <= 16 {
                    return self.konst(w, 0);
                }
                if lo >= 16 {
                    return self.extract(lo - 16, w, x);
                }
                let h = self.extract(0, w - (16 - lo), x);
                let z = self.konst(16 - lo, 0);
                return self.concat(h, z);
            }
            _ => {}
        }
        self.mk(Node::Extract { lo, w, a }, w)
    }

    pub fn concat(&mut self, hi: E, lo: E) -> E {
        let (wh, wl) = (self.w(hi), self.w(lo));
        let w = wh + wl;
        if self.is_const(hi) == Some(0) {
            return self.zext(w, lo);
        }
        // Constant low pieces gather at the right: concat(a, concat(b, k)) = concat(concat(a, b), k).
        if let Node::Concat { hi: h2, lo: l2 } = self.nodes[lo as usize].clone() {
            if self.is_const(l2).is_some() && self.is_const(h2).is_none() {
                let x = self.concat(hi, h2);
                return self.concat(x, l2);
            }
        }
        if wh == 16 && wl == 16 && self.is_const(lo) == Some(0) {
            return self.un(Un::B2F, hi);
        }
        // Adjacent pieces of one value.
        if let (
            Node::Extract { lo: l1, a: x, .. },
            Node::Extract {
                lo: l2,
                w: w2,
                a: y,
            },
        ) = (
            self.nodes[hi as usize].clone(),
            self.nodes[lo as usize].clone(),
        ) {
            if x == y && l1 == l2 + w2 {
                return self.extract(l2, w, x);
            }
        }
        if let Node::Extract { lo: l1, a: x, .. } = self.nodes[hi as usize].clone() {
            if l1 == wl && x == lo {
                return self.extract(0, w, x);
            }
        }
        // A copy of the low piece's sign: sign extension.
        if let Node::Sext { a: s, .. } = self.nodes[hi as usize].clone() {
            if let Node::Extract { lo: lb, w: 1, a: x } = self.nodes[s as usize].clone() {
                if lb + 1 == wl && x == lo {
                    return self.sext(w, lo);
                }
                if let Node::Extract {
                    lo: l0,
                    w: w0,
                    a: y,
                } = self.nodes[lo as usize].clone()
                {
                    if x == y && lb == l0 + w0 - 1 {
                        return self.sext(w, lo);
                    }
                }
            }
        }
        self.mk(Node::Concat { hi, lo }, w)
    }

    pub fn fma(&mut self, a: E, b: E, c: Option<E>, rnd: Rnd, ftz: bool) -> E {
        self.mk(Node::Fma { a, b, c, rnd, ftz }, 32)
    }

    pub fn fminmax(&mut self, min: bool, nan: bool, ftz: bool, a: E, b: E) -> E {
        self.mk(
            Node::FMinMax {
                min,
                nan,
                ftz,
                a,
                b,
            },
            32,
        )
    }

    pub fn fcmp(&mut self, cmp: FCmp, ftz: bool, a: E, b: E) -> E {
        self.mk(Node::FCmp { cmp, ftz, a, b }, 1)
    }

    pub fn mufu(&mut self, func: MufuFunc, fmt: MufuFmt, hi: bool, ptx: Option<bool>, a: E) -> E {
        self.mk(
            Node::Mufu {
                func,
                fmt,
                hi,
                ptx,
                a,
            },
            32,
        )
    }

    pub fn bnot(&mut self, a: E) -> E {
        self.un(Un::Not, a)
    }

    /// The nodes of `e`'s cone in topological order (children first).
    pub fn cone(&self, e: E) -> Vec<E> {
        let mut seen = HashSet::new();
        let mut st = vec![e];
        while let Some(x) = st.pop() {
            if seen.insert(x) {
                st.extend(kids(&self.nodes[x as usize]));
            }
        }
        let mut v: Vec<E> = seen.into_iter().collect();
        v.sort_unstable();
        v
    }

    /// The leaves (`Input`, `Opaque`) of `e`.
    pub fn leaves(&self, e: E) -> Vec<E> {
        self.cone(e)
            .into_iter()
            .filter(|&x| {
                matches!(
                    self.nodes[x as usize],
                    Node::Input { .. } | Node::Opaque { .. } | Node::Sreg { .. }
                )
            })
            .collect()
    }

    /// The input allocations `e` derives from (coverage.rs `Prov` mask).
    pub fn alloc_mask(&self, e: E) -> u64 {
        self.leaves(e)
            .iter()
            .fold(0, |m, &x| match self.nodes[x as usize] {
                Node::Input { alloc, .. } => m | 1u64 << alloc.min(63),
                Node::Sreg { .. } => m,
                Node::Opaque { id, .. } if self.opaque[id as usize].why == UNDEF => m,
                _ => m | 1 << 63,
            })
    }

    /// Text form (depth-limited), for notes and debugging.
    pub fn show(&self, e: E, names: &dyn Fn(u32) -> String, depth: usize) -> String {
        if depth == 0 {
            return "...".into();
        }
        let s = |x: E| self.show(x, names, depth - 1);
        match &self.nodes[e as usize] {
            Node::Const { w, v } => {
                if *w == 32 && *v > 0xffff {
                    format!("{v:#x}")
                } else {
                    format!("{v}")
                }
            }
            Node::Input { alloc, off, bytes } => {
                let b = *bytes as u64;
                if off % b == 0 {
                    format!("{}[{}]", names(*alloc), off / b)
                } else {
                    format!("{}[byte {off:#x}:{bytes}]", names(*alloc))
                }
            }
            Node::Opaque { id, .. } => format!("opaque#{id}({})", self.opaque[*id as usize].why),
            Node::Sreg { s, key } => format!("{s:?}@{key}"),
            Node::Un { op, a, .. } => format!("{op:?}({})", s(*a)),
            Node::Bin { op, a, b, .. } => format!("{op:?}({}, {})", s(*a), s(*b)),
            Node::Lop3 { lut, a, b, c, .. } => {
                format!("lop3_{lut:#x}({}, {}, {})", s(*a), s(*b), s(*c))
            }
            Node::Cmp { cmp, a, b } => format!("{cmp:?}({}, {})", s(*a), s(*b)),
            Node::Ite { c, a, b } => format!("ite({}, {}, {})", s(*c), s(*a), s(*b)),
            Node::Zext { w, a } => format!("zext{w}({})", s(*a)),
            Node::Sext { w, a } => format!("sext{w}({})", s(*a)),
            Node::Extract { lo, w, a } => format!("{}[{lo}+:{w}]", s(*a)),
            Node::Concat { hi, lo } => format!("concat({}, {})", s(*hi), s(*lo)),
            Node::Fma { a, b, c, rnd, ftz } => format!(
                "fma.{rnd:?}{}({}, {}, {})",
                if *ftz { ".ftz" } else { "" },
                s(*a),
                s(*b),
                c.map_or("-".into(), s)
            ),
            Node::FMinMax { min, a, b, .. } => format!(
                "f{}({}, {})",
                if *min { "min" } else { "max" },
                s(*a),
                s(*b)
            ),
            Node::FCmp { cmp, a, b, .. } => format!("f{cmp:?}({}, {})", s(*a), s(*b)),
            Node::Mufu { func, a, .. } => format!("{func:?}({})", s(*a)),
            Node::DFma { a, b, c, rnd } => format!(
                "dfma.{rnd:?}({}, {}, {})",
                s(*a),
                s(*b),
                c.map_or("-".into(), s)
            ),
            Node::DCmp { cmp, a, b } => format!("d{cmp:?}({}, {})", s(*a), s(*b)),
            Node::HFma { bf16, a, b, c } => format!(
                "{}fma({}, {}, {})",
                if *bf16 { "bf16" } else { "f16" },
                s(*a),
                s(*b),
                s(*c)
            ),
            Node::Narrow { bf16, a } => {
                format!("to_{}({})", if *bf16 { "bf16" } else { "f16" }, s(*a))
            }
            Node::AtAddr { addr, at, val } => format!("at({} == {at:#x}: {})", s(*addr), s(*val)),
            Node::Pack { to, a, b, c, .. } => {
                format!("pack_{to:?}({}, {}, {})", s(*a), s(*b), s(*c))
            }
            Node::FCvt { from, to, a, .. } => format!("cvt_{from:?}_{to:?}({})", s(*a)),
            Node::F2F { to64, a, .. } => {
                format!("{}({})", if *to64 { "f2f64" } else { "f2f32" }, s(*a))
            }
            Node::I2F { signed, f64, a, .. } => format!(
                "i2f{}{}({})",
                if *signed { "s" } else { "u" },
                if *f64 { 64 } else { 32 },
                s(*a)
            ),
            Node::F2I {
                signed, bits, a, ..
            } => format!("f2{}{bits}({})", if *signed { "s" } else { "u" }, s(*a)),
            Node::Inst { pc, out, args, .. } => format!(
                "inst@{pc}.{out}({})",
                args.iter().map(|&x| s(x)).collect::<Vec<_>>().join(", ")
            ),
        }
    }
}

/// Where a memory byte's value comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Src {
    /// Not written since the launch, not input (shared memory): the loaded value.
    Init,
    /// Input byte `off` of allocation `alloc` (also a copy of it).
    In(u32, u64),
    /// A constant byte.
    K(u8),
    /// Byte `i` of expression `e`.
    B(E, u8),
    /// An opaque byte (`Arena::opaque` reason index).
    O(u32),
    /// A byte stored under an input-dependent guard (`SymState::gbytes` index): the stored
    /// byte while the guard holds, the previous one otherwise.
    G(u32),
}

/// Kinds of path-condition atoms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum AtomKind {
    /// A comparison or test that writes a predicate (an input-coverage site).
    Site,
    /// The guard of an instruction (or a branch condition).
    Guard,
    /// A select's predicate.
    Select,
}

/// `cond == val` for one thread at `pc`.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Atom {
    pub pc: u32,
    pub cond: E,
    pub val: bool,
    pub kind: AtomKind,
}

#[derive(Clone, Debug)]
pub struct Config {
    /// Expression tree-size cap.
    pub cap: u32,
    /// Path-condition atoms kept per thread.
    pub max_atoms: usize,
    /// Observations kept per site and outcome.
    pub max_examples: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            cap: 2048,
            max_atoms: 1024,
            max_examples: 64,
        }
    }
}

/// One observation at a site: thread `t` produced its predicate with path atom `atom`.
#[derive(Clone, Copy, Debug)]
pub struct Obs {
    pub t: u32,
    pub warp: u32,
    /// Index into the thread's path (`SymState::path[t]`).
    pub atom: u32,
}

/// One input-dependent floating-point operation of thread `t` (lane `lane` of the
/// instruction's operations, low lane first): `node` is the result, `operands` its operands
/// after modifiers (`Arena::nodes[node]`'s children, with rounding mode / FTZ / function in
/// the node itself).
#[derive(Clone, Debug)]
pub struct FpObs {
    pub t: u32,
    pub warp: u32,
    pub lane: u8,
    pub node: E,
    pub operands: Vec<E>,
}

/// What `post` does for one thread.
#[derive(Clone, Debug)]
enum Pend {
    /// Compute the destinations from the operation and the operands read before it ran
    /// (None: no operand has an expression).
    Op(Option<Vec<(Reg, E)>>),
    /// A load: the bytes the thread read (`acc`), shared among the destinations; `addr`: the
    /// address operands' expression when input-dependent.
    Load {
        bytes: u32,
        sext: bool,
        addr: Option<E>,
    },
    /// A shuffle: the source lane's operand (or why not known).
    Shfl(Result<E, String>),
    /// A local-memory load: the bytes per destination register (from this thread's stores).
    Local(Vec<Vec<Src>>),
    /// Destinations not modelled (input-derived ones become opaque).
    Opaque(String),
    /// Every destination opaque, whatever its provenance (float atomics: the result depends on
    /// the order of other threads' updates).
    ForceOpaque(String),
    /// An atomic add of a non-negative constant: the old value is the word's initial value
    /// plus the other threads' adds before it (an opaque, non-negative part).
    AtomAdd { bytes: u8 },
}

/// `ProvState::sym`: a clone (a trial execution's scratch machine) has none.
#[derive(Debug, Default)]
pub struct SymSlot(pub Option<Box<SymState>>);

impl Clone for SymSlot {
    fn clone(&self) -> Self {
        SymSlot(None)
    }
}

/// The symbolic shadow of one run (`ProvState::sym`).
#[derive(Clone, Debug, Default)]
pub struct SymState {
    pub ar: Arena,
    pub cfg: Config,
    regs: HashMap<(u32, u8, u16), E>,
    mem: HashMap<(u8, u64, u64), Src>,
    local: HashMap<(usize, u32), Src>,
    /// Byte sources each actor read during the current instruction (async units: since
    /// their last write).
    acc: HashMap<u32, Vec<Src>>,
    /// The bytes each actor's store of the current instruction writes.
    wdata: HashMap<u32, Vec<Src>>,
    pending: Vec<(usize, Pend)>,
    nth: usize,
    /// What the current instruction is, for opaque bytes it writes (`OPQ_MARK`).
    wreason: String,
    /// Uniform registers given an expression by some member during the current instruction.
    uset: HashSet<(u32, u8, u16)>,
    /// The first member whose guard holds (R2UR takes its value).
    first: usize,
    /// Input-dependent guard of each active thread of the current instruction (stores).
    sguard: HashMap<u32, E>,
    /// Bytes stored under an input-dependent guard: (guard, stored, previous).
    gbytes: Vec<(E, Src, Src)>,
    /// Per thread of the current instruction: its input-dependent guard and the destinations'
    /// expressions before it ran (a write becomes `ite(guard, new, old)`).
    gold: HashMap<usize, (E, Vec<(Reg, E)>)>,
    /// Model guard-false members without recording path atoms.
    quiet: bool,
    /// Guard-false members of an input-dependent guard (`gold`), modelled in `post`.
    ghost: Vec<(usize, Pend)>,
    /// cp.async copies in flight: the bytes each will land, by (issuing thread, CTA, shared
    /// offset); the landing (at a wait, possibly another lane's) writes them.
    cp_stash: HashMap<(u32, u64, u64), Vec<Src>>,
    /// Path conditions per thread.
    pub path: HashMap<u32, Vec<Atom>>,
    /// Threads whose path reached `max_atoms`.
    pub truncated: HashSet<u32>,
    /// Threads whose path holds only part of the conditions before its atoms (a target's
    /// per-element threads seeded with a bounded tail of their core's path).
    pub partial: HashSet<u32>,
    /// Observations per site pc and outcome.
    pub sites: BTreeMap<u32, [Vec<Obs>; 2]>,
    /// Input-dependent floating-point operations per pc (`record_fp`).
    pub fp_obs: BTreeMap<u32, Vec<FpObs>>,
    /// Node-vs-simulator mismatches on the captured input (pc, register, text); each fell back.
    pub mismatches: Vec<(u32, String)>,
    pub n_mismatch: usize,
    /// Destinations whose provenance (coverage.rs `Prov`) was clean while input leaves reach
    /// their expression through data (a provenance miss; the expression is kept; dependence
    /// through a select or guard condition only is not counted): count and examples.
    pub n_prov_gap: usize,
    pub prov_gaps: Vec<(u32, String)>,
    /// Destinations given `Inst` nodes / opaque leaves, by reason.
    pub n_inst: usize,
    pub n_opaque: BTreeMap<String, usize>,
}

fn reg_key(m: &Machine, t: usize, r: Reg) -> (u32, u8, u16) {
    let owner = match r.file {
        File::UR | File::UP => 1u32 << 31 | m.threads[t].warp,
        _ => t as u32,
    };
    (owner, r.file as u8, r.idx)
}

fn loc_key(loc: Loc) -> (u8, u64, u64) {
    match loc {
        Loc::Shared { cta, off } => (0, cta as u64, off as u64),
        Loc::Global { alloc, off } => (1, alloc as u64, off),
        Loc::Param { .. } => (2, 0, 0),
    }
}

/// The constant-true predicate (SASS PT = P7, UPT = UP7; PTX has none).
fn is_pt(m: &Machine, r: Reg) -> bool {
    r.idx == 7 && (r.file == File::UP || r.file == File::P && m.k.front == ptxv_ir::Front::Sass)
}

/// Width of a register's value.
fn reg_w(r: Reg) -> u8 {
    match r.file {
        File::V => 64,
        File::R | File::UR => 32,
        File::P | File::UP => 1,
    }
}

/// The simulator's value of register `r` of thread `t`.
pub fn reg_val(m: &Machine, t: usize, r: Reg) -> u64 {
    match r.file {
        File::P | File::UP => m.prd(
            t,
            Pred {
                reg: Some(r),
                neg: false,
            },
        ) as u64,
        _ => m.rd(t, &Val::Reg(r, 1)).unwrap_or(0),
    }
}

/// Register-only operations whose result for one thread is a function of its own operands
/// (the ones `Inst` may re-run).
pub fn per_thread_pure(op: &Op) -> bool {
    let collective = matches!(
        op,
        Op::Shfl { .. }
            | Op::VoteAny { .. }
            | Op::VoteAll { .. }
            | Op::MatchAny { .. }
            | Op::Redux { .. }
            | Op::MovMatrix { .. }
            | Op::Hmma { .. }
            | Op::R2Ur { .. }
            | Op::LdConst { .. }
            | Op::P2r { uniform: true, .. }
    );
    !collective
        && (crate::num::pure_op(op)
            || matches!(
                op,
                Op::Dsetp { .. }
                    | Op::Dfma { .. }
                    | Op::DsetpMinMax { .. }
                    | Op::Rsqrt64Approx { .. }
                    | Op::FCvt { .. }
                    | Op::F2F { .. }
                    | Op::Hset2 { .. }
                    | Op::Dp2a { .. }
                    | Op::Fchk { .. }
                    | Op::Fsetp { .. }
                    | Op::Fsel { .. }
                    | Op::Fset { .. }
                    | Op::Hsetp2Bf16 { .. }
                    | Op::R2p { .. }
                    | Op::CvtPack { .. }
                    | Op::CvtUnpack { .. }
                    | Op::F2fpBf16 { .. }
                    | Op::F2fpF16 { .. }
                    | Op::CvtF32F16 { .. }
                    | Op::F2fF16 { .. }
                    | Op::CvtTf32 { .. }
                    | Op::DivFull { .. }
                    | Op::DivSqrt64 { .. }
            ))
}

/// Whether coverage.rs counts `op`'s predicate destinations as sites.
fn site_op(op: &Op) -> bool {
    matches!(
        op,
        Op::Setp { .. }
            | Op::Fsetp { .. }
            | Op::Dsetp { .. }
            | Op::Fchk { .. }
            | Op::Lop3 { .. }
            | Op::R2p { .. }
            | Op::VoteAll { .. }
            | Op::VoteAny { .. }
            | Op::MatchAny { .. }
            | Op::Hsetp2Bf16 { .. }
    )
}

impl SymState {
    pub fn new(cfg: Config, nth: usize) -> SymState {
        let mut s = SymState {
            cfg,
            nth,
            ..Default::default()
        };
        s.ar.cap = s.cfg.cap;
        s
    }

    /// The expression of register `r` of thread `t` (a constant when it holds none).
    pub fn reg(&mut self, m: &Machine, t: usize, r: Reg) -> E {
        if is_pt(m, r) {
            return self.ar.konst(1, 1);
        }
        let k = reg_key(m, t, r);
        if let Some(&e) = self.regs.get(&k) {
            return e;
        }
        let v = reg_val(m, t, r);
        if m.reg_umask(t, r) != 0 {
            // Read before any write (or holding undefined bits): never a constant.
            let e = self.ar.opaque(reg_w(r), v, UNDEF);
            self.regs.insert(k, e);
            return e;
        }
        self.ar.konst(reg_w(r), v)
    }

    /// The expression of `bytes` bytes of global allocation `alloc` at `off` (their current
    /// sources), `v` their value.
    pub fn global_expr(&mut self, m: &Machine, alloc: u32, off: u64, bytes: u8, v: u64) -> E {
        let src: Vec<Src> = (0..bytes as u64)
            .map(|i| {
                self.mem
                    .get(&(1, alloc as u64, off + i))
                    .copied()
                    .unwrap_or(Src::In(alloc, off + i))
            })
            .collect();
        load_expr(self, m, &src, v)
    }

    /// An expression is held for `r` (input-derived, placement-dependent or undefined).
    pub fn has(&self, m: &Machine, t: usize, r: Reg) -> bool {
        self.regs.contains_key(&reg_key(m, t, r))
    }

    fn set(&mut self, m: &Machine, t: usize, r: Reg, e: E) {
        if is_pt(m, r) {
            return;
        }
        let k = reg_key(m, t, r);
        if self.ar.is_const(e).is_some() {
            self.regs.remove(&k);
        } else {
            self.regs.insert(k, e);
        }
    }

    fn push_atom(
        &mut self,
        m: &Machine,
        t: usize,
        pc: u32,
        cond: E,
        val: bool,
        kind: AtomKind,
    ) -> Option<u32> {
        if !self.ar.dep[cond as usize] || self.quiet {
            return None;
        }
        let max = self.cfg.max_atoms;
        let p = self.path.entry(t as u32).or_default();
        // The same condition again (a guard read by consecutive instructions): once.
        if let Some(i) = p.iter().rposition(|a| a.cond == cond && a.val == val) {
            if p.len() - i <= 8 {
                // A guard is the strongest kind (control flow follows it).
                if kind == AtomKind::Guard {
                    p[i].kind = AtomKind::Guard;
                }
                return Some(i as u32);
            }
        }
        if p.len() >= max {
            self.truncated.insert(t as u32);
            return None;
        }
        p.push(Atom {
            pc,
            cond,
            val,
            kind,
        });
        let _ = m;
        Some((p.len() - 1) as u32)
    }
}

/// The 64-bit operand value of `v` from the registers captured before the instruction ran.
fn opnd(s: &mut SymState, m: &Machine, t: usize, snap: &[(Reg, E)], v: &Val) -> E {
    let get = |s: &mut SymState, r: Reg| -> E {
        match snap.iter().find(|x| x.0 == r) {
            Some(x) => x.1,
            None => s.reg(m, t, r),
        }
    };
    match v {
        Val::Reg(r, n) => match r.file {
            File::V => get(s, *r),
            File::R | File::UR => {
                let g = |s: &mut SymState, i: u16| {
                    if i >= 255 {
                        s.ar.konst(32, 0)
                    } else {
                        get(
                            s,
                            Reg {
                                file: r.file,
                                idx: i,
                            },
                        )
                    }
                };
                let lo = g(s, r.idx);
                if *n >= 2 {
                    let hi = g(s, r.idx + 1);
                    s.ar.concat(hi, lo)
                } else {
                    s.ar.zext(64, lo)
                }
            }
            _ => {
                let p = get(s, *r);
                s.ar.zext(64, p)
            }
        },
        Val::SReg(sr) if placed_sreg(*sr) => {
            let x = m.rd(t, v).unwrap_or(0);
            let key = match sr {
                SReg::WarpId => m.threads[t].warp,
                _ => m.threads[t].cta as u32,
            };
            s.ar.sreg(*sr, key, x)
        }
        _ => {
            let x = m.rd(t, v).unwrap_or(0);
            s.ar.konst(64, x)
        }
    }
}

/// Special registers whose value can depend on placement (which CTA, cluster, warp slot or
/// SM, or time): leaves, never constants. Only those known to be fixed for a thread by the
/// launch configuration are constants; anything else defaults to placement-dependent.
// TODO: use the IR's `SReg::position_dependent` classification once it exists.
pub fn placed_sreg(s: SReg) -> bool {
    !matches!(
        s,
        SReg::Tid(_)
            | SReg::NTid(_)
            | SReg::NCtaId(_)
            | SReg::LaneId
            | SReg::ClusterNCtaRank
            | SReg::ClusterNCtaId(_)
            | SReg::NClusterId(_)
            | SReg::LaneMaskLt
            | SReg::LaneMaskLe
            | SReg::LaneMaskGe
            | SReg::LaneMaskGt
            | SReg::LaneMaskEq
            | SReg::GlobalAddr(_)
    )
}

fn pred_e(s: &mut SymState, m: &Machine, t: usize, snap: &[(Reg, E)], p: Pred) -> E {
    let e = match p.reg {
        None => s.ar.konst(1, 1),
        Some(r) => match snap.iter().find(|x| x.0 == r) {
            Some(x) => x.1,
            None => s.reg(m, t, r),
        },
    };
    if p.neg { s.ar.bnot(e) } else { e }
}

/// `access_x` hook (coverage.rs `ProvState::on_access`).
pub(crate) fn on_access(s: &mut SymState, actor: u32, loc: Loc, n: usize, write: bool) {
    let (sp, o, off) = loc_key(loc);
    if sp == 2 {
        return;
    }
    let read: Vec<Src> = (0..n as u64)
        .map(|i| {
            let k = (sp, o, off + i);
            match s.mem.get(&k) {
                Some(b) => *b,
                None if sp == 1 => Src::In(o as u32, off + i),
                None => Src::Init,
            }
        })
        .collect();
    if !write {
        s.acc.entry(actor).or_default().extend(read);
        return;
    }
    let landed = match s.cp_stash.remove(&(actor, o, off)) {
        Some(d) if sp == 0 && d.len() == n => Some(d),
        _ => None,
    };
    let data: Vec<Src> = match s.wdata.get(&actor) {
        _ if landed.is_some() => landed.unwrap_or_default(),
        Some(d) if d.len() == 1 && d[0] == Src::O(OPQ_MARK) => {
            let id = s.ar.opaque.len() as u32;
            let why = format!("memory written by {}", s.wreason);
            s.ar.opaque.push(OpaqueLeaf { why, pc: s.ar.pc });
            vec![Src::O(id); n]
        }
        Some(d) if d.len() == n => d.clone(),
        Some(d) if d.len() == 1 => vec![d[0]; n],
        _ => {
            // A unit copying what it read (cp.async, TMA, bulk copies): the bytes in order,
            // when exactly as many were read since its last write as it writes now (a chunk
            // moved whole; out-of-bounds fills and reordering within a chunk are writes
            // without matching reads: opaque). A shared byte never written has no known value
            // when copied: opaque.
            let acc = s.acc.entry(actor).or_default();
            if acc.len() == n && !acc.contains(&Src::Init) {
                std::mem::take(acc)
            } else {
                acc.clear();
                let id = s.ar.opaque.len() as u32;
                s.ar.opaque.push(OpaqueLeaf {
                    why: "memory written by an operation the expressions do not follow".into(),
                    pc: s.ar.pc,
                });
                vec![Src::O(id); n]
            }
        }
    };
    let guard = s.sguard.get(&actor).copied();
    for (i, b) in data.into_iter().enumerate().take(n) {
        let b = match guard {
            // Stored under an input-dependent guard: the previous byte where it fails.
            Some(g) if b != read[i] && !matches!(b, Src::O(_)) => {
                s.gbytes.push((g, b, read[i]));
                Src::G(s.gbytes.len() as u32 - 1)
            }
            _ => b,
        };
        s.mem.insert((sp, o, off + i as u64), b);
    }
    if actor as usize >= s.nth {
        s.acc.remove(&actor);
    }
}

/// `wdata` marker: the bytes the thread writes get a fresh opaque leaf (`SymState::wreason`).
const OPQ_MARK: u32 = u32::MAX;

/// Bytes `bytes` (sources and locations) as an expression of width 8 * len, `v` the value the
/// simulator loaded (unknown bytes take theirs).
fn bytes_expr(s: &mut SymState, m: &Machine, src: &[Src], v: u64) -> E {
    let mut acc: Option<E> = None;
    for (i, &b) in src.iter().enumerate() {
        let byte_v = (v >> (8 * i)) & 0xff;
        let piece = src_byte(s, m, b, Some(byte_v), 0);
        acc = Some(match acc {
            None => piece,
            Some(lo) => s.ar.concat(piece, lo),
        });
    }
    acc.unwrap_or_else(|| s.ar.konst(8, 0))
}

/// The expression of one byte from source `b`; `v`: its value when known (the byte a load
/// read), None for the unselected side of a guarded store.
fn src_byte(s: &mut SymState, m: &Machine, b: Src, v: Option<u64>, depth: u32) -> E {
    let init = |a: u32, o: u64| {
        m.launch
            .allocs
            .get(a as usize)
            .and_then(|x| x.init.get(o as usize))
            .copied()
            .unwrap_or(0)
    };
    match b {
        Src::In(a, o) => {
            let x = v.unwrap_or_else(|| init(a, o) as u64);
            s.ar.input(a, o, 1, x)
        }
        Src::K(x) => s.ar.konst(8, x as u64),
        Src::B(e, j) => s.ar.extract(8 * j, 8, e),
        Src::Init => match v {
            Some(x) => s.ar.konst(8, x),
            None => {
                s.ar.opaque(8, 0, "shared memory not written before a guarded store")
            }
        },
        Src::O(id) => {
            let why = s.ar.opaque[id as usize].why.clone();
            s.ar.opaque(8, v.unwrap_or(0), why)
        }
        Src::G(id) => {
            let (g, new, old) = s.gbytes[id as usize];
            if depth > 8 {
                return s.ar.opaque(8, v.unwrap_or(0), "nested guarded stores");
            }
            // The guard held on the captured input (the store ran): `v` is the new byte's.
            let hold = s.ar.conc[g as usize] & 1 == 1;
            let n = src_byte(s, m, new, if hold { v } else { None }, depth + 1);
            let o = src_byte(s, m, old, if hold { None } else { v }, depth + 1);
            s.ar.ite(g, n, o)
        }
    }
}

/// Merge runs of input bytes of one allocation into one wider input leaf: a word load is
/// one leaf, not a concatenation of bytes.
fn input_word(s: &mut SymState, src: &[Src], v: u64) -> Option<E> {
    let Src::In(a0, o0) = *src.first()? else {
        return None;
    };
    if src.len() > 8 {
        return None;
    }
    for (i, &b) in src.iter().enumerate() {
        if b != Src::In(a0, o0 + i as u64) {
            return None;
        }
    }
    Some(s.ar.input(a0, o0, src.len() as u8, v))
}

fn load_expr(s: &mut SymState, m: &Machine, src: &[Src], v: u64) -> E {
    match input_word(s, src, v) {
        Some(e) => e,
        None => bytes_expr(s, m, src, v),
    }
}

/// The store data of `op` as bytes (`wdata`), from the registers before it ran.
fn store_bytes(s: &mut SymState, m: &Machine, t: usize, op: &Op) -> Option<Vec<Src>> {
    let Op::St { v, bytes, .. } = op else {
        return None;
    };
    let n = *bytes as usize;
    // The data as pieces of at most 64 bits, low bytes first: (expression, bytes).
    let pieces: Vec<(E, usize)> = match v {
        Val::Reg(r, k) if r.file == File::V && *k >= 2 => {
            // st.vN: one register per element.
            let per = (n / *k as usize).max(1);
            (0..*k as u16)
                .map(|i| {
                    (
                        s.reg(
                            m,
                            t,
                            Reg {
                                file: r.file,
                                idx: r.idx + i,
                            },
                        ),
                        per,
                    )
                })
                .collect()
        }
        Val::Reg(r, k) if r.file != File::V && n > 8 => {
            // STS.128 / wide stores: consecutive 32-bit registers.
            (0..(n / 4).max(*k as usize) as u16)
                .map(|i| {
                    (
                        s.reg(
                            m,
                            t,
                            Reg {
                                file: r.file,
                                idx: r.idx + i,
                            },
                        ),
                        4,
                    )
                })
                .collect()
        }
        _ => vec![(opnd(s, m, t, &[], v), n.min(8))],
    };
    let mut out = Vec::with_capacity(n);
    for (e, per) in pieces {
        let w = s.ar.w(e) as usize;
        for j in 0..per {
            if 8 * j >= w {
                // Bytes above the register (a narrow element in a wide slot): zero.
                out.push(Src::K(0));
            } else if let Some(c) = s.ar.is_const(e) {
                out.push(Src::K((c >> (8 * j)) as u8));
            } else {
                out.push(Src::B(e, j as u8));
            }
        }
    }
    (out.len() == n).then_some(out)
}

/// Before `pc` runs for `members` (coverage.rs `pre`); `active`: the members whose guard holds.
pub(crate) fn pre(s: &mut SymState, m: &Machine, members: &[usize], active: &[usize], pc: u32) {
    let k = m.k.clone();
    let inst = &k.code[pc as usize];
    s.ar.pc = pc;
    s.pending.clear();
    s.wdata.clear();
    s.uset.clear();
    s.sguard.clear();
    s.gold.clear();
    s.ghost.clear();
    s.nth = m.threads.len();
    // Guards and branch conditions on input-derived predicates.
    let cond = match &inst.op {
        Op::Bra { cond, .. } => *cond,
        _ => None,
    };
    let dsts: Vec<Reg> = inst.op.dst_regs();
    let mut ghosts = Vec::new();
    for &t in members {
        for p in [Some(inst.guard), cond].into_iter().flatten() {
            if p.reg.is_some() {
                let g = pred_e(s, m, t, &[], p);
                let v = m.prd(t, p);
                s.push_atom(m, t, pc, g, v, AtomKind::Guard);
            }
        }
        // An input-dependent guard: what it writes becomes `ite(guard, new, old)`, for the
        // members it holds for and (registers only) for those it fails for.
        if inst.guard.reg.is_some() && !matches!(inst.op, Op::Bra { .. }) {
            let g = pred_e(s, m, t, &[], inst.guard);
            if s.ar.dep[g as usize] {
                let old: Vec<(Reg, E)> = dsts
                    .iter()
                    .filter(|r| !(matches!(r.file, File::R | File::UR) && r.idx >= 255))
                    .map(|&r| (r, s.reg(m, t, r)))
                    .collect();
                if active.contains(&t) {
                    s.sguard.insert(t as u32, g);
                } else if !dsts.iter().any(|r| r.is_uniform()) {
                    ghosts.push(t);
                }
                s.gold.insert(t, (g, old));
            }
        }
    }
    s.wreason = op_reason(&inst.op);
    s.first = active.first().copied().unwrap_or(usize::MAX);
    let copies = matches!(
        inst.op,
        Op::St { .. } | Op::CpAsync { .. } | Op::StLocal { .. }
    );
    for &t in active {
        s.acc.remove(&(t as u32));
        if !copies {
            // Memory a thread writes with anything but a store or a copy (atomics, matrix
            // stores, reductions): opaque.
            s.wdata.insert(t as u32, vec![Src::O(OPQ_MARK)]);
        }
        let pend = pend_of(s, m, t, &inst.op, active);
        s.pending.push((t, pend));
    }
    // Guard-false members of an input-dependent guard: their destinations' would-be values
    // (no memory effect is followed for them).
    for t in ghosts {
        let pend = match &inst.op {
            Op::Ld { .. }
            | Op::LdLocal { .. }
            | Op::Atom { .. }
            | Op::St { .. }
            | Op::StLocal { .. }
            | Op::Shfl { .. } => Pend::Opaque("guarded-off memory or warp operation".into()),
            _ => {
                let snap: Vec<(Reg, E)> = inst
                    .op
                    .src_regs()
                    .iter()
                    .map(|&r| (r, s.reg(m, t, r)))
                    .collect();
                Pend::Op(Some(snap))
            }
        };
        s.ghost.push((t, pend));
    }
}

/// What `post` does for active thread `t` (`pre`).
fn pend_of(s: &mut SymState, m: &Machine, t: usize, op: &Op, active: &[usize]) -> Pend {
    let inst_op = op;
    {
        let srcs = inst_op.src_regs();
        let any = srcs
            .iter()
            .any(|&r| s.has(m, t, r) || m.reg_umask(t, r) != 0)
            || inst_op
                .src_vals()
                .iter()
                .any(|v| matches!(v, Val::SReg(x) if placed_sreg(*x)));
        let pend = match inst_op {
            Op::Ld {
                bytes, sext, addr, ..
            } => Pend::Load {
                bytes: *bytes as u32,
                sext: *sext,
                addr: addr_expr(s, m, t, addr),
            },
            Op::Shfl { .. } => Pend::Shfl(shfl_src(s, m, t, inst_op, active)),
            Op::Atom {
                op: ptxv_ir::AtomOp::Add,
                bytes: b @ (4 | 8),
                v,
                d: Some(_),
                ..
            } if !matches!(v, Val::Reg(r, _) if s.has(m, t, *r)) && {
                let x = m.rd(t, v).unwrap_or(u64::MAX);
                if *b == 8 {
                    (x as i64) >= 0
                } else {
                    (x as u32 as i32) >= 0
                }
            } =>
            {
                Pend::AtomAdd { bytes: *b }
            }
            Op::Atom {
                op:
                    ptxv_ir::AtomOp::FAddFtz
                    | ptxv_ir::AtomOp::FAdd64
                    | ptxv_ir::AtomOp::AddBf16x2
                    | ptxv_ir::AtomOp::AddBf16,
                ..
            } => Pend::ForceOpaque("float atomic result".into()),
            Op::St { .. } => {
                if let Some(d) = store_bytes(s, m, t, inst_op) {
                    s.wdata.insert(t as u32, d);
                }
                Pend::Opaque("store".into())
            }
            Op::StLocal { bytes, addr, v } => {
                if let Ok(a) = m.local_addr(t, addr) {
                    let e = opnd(s, m, t, &[], v);
                    for i in 0..*bytes {
                        let b = if i >= 8 {
                            Src::O(local_opaque(s))
                        } else if let Some(c) = s.ar.is_const(e) {
                            Src::K((c >> (8 * i)) as u8)
                        } else {
                            Src::B(e, i as u8)
                        };
                        s.local.insert((t, a + i), b);
                    }
                }
                Pend::Opaque("local store".into())
            }
            Op::LdLocal { bytes, addr, .. } => match m.local_addr(t, addr) {
                Ok(a) => {
                    let dsts = inst_op.dst_regs();
                    let per = (*bytes as usize / dsts.len().max(1)).max(1);
                    let srcs: Option<Vec<Vec<Src>>> = (0..dsts.len())
                        .map(|j| {
                            (0..per)
                                .map(|i| s.local.get(&(t, a + (j * per + i) as u32)).copied())
                                .collect::<Option<Vec<Src>>>()
                        })
                        .collect();
                    match srcs {
                        Some(v) => Pend::Local(v),
                        None => {
                            Pend::Opaque("local load of bytes this thread did not store".into())
                        }
                    }
                }
                Err(_) => Pend::Opaque("local address".into()),
            },
            _ if !any => Pend::Op(None),
            _ => {
                let snap: Vec<(Reg, E)> = srcs.iter().map(|&r| (r, s.reg(m, t, r))).collect();
                Pend::Op(Some(snap))
            }
        };
        pend
    }
}

/// The address operands of a load (their sum) when input-dependent.
fn addr_expr(s: &mut SymState, m: &Machine, t: usize, a: &ptxv_ir::Addr) -> Option<E> {
    let mut sum: Option<E> = None;
    let mut dep = false;
    for v in [a.base.as_ref(), a.base2.as_ref()].into_iter().flatten() {
        if !matches!(v, Val::Reg(..)) {
            continue;
        }
        let e = opnd(s, m, t, &[], v);
        dep |= s.ar.dep[e as usize];
        sum = Some(match sum {
            None => e,
            Some(x) => s.ar.bin(Bin::Add, x, e),
        });
    }
    sum.filter(|_| dep)
}

/// The operand a shuffle moves to thread `t`: the source lane's expression (the lane
/// computation of exec.rs `Op::Shfl`), or why it is not known.
fn shfl_src(
    s: &mut SymState,
    m: &Machine,
    t: usize,
    op: &Op,
    active: &[usize],
) -> Result<E, String> {
    let Op::Shfl {
        a,
        lane,
        clamp,
        mode,
        ..
    } = op
    else {
        return Err("shuffle".into());
    };
    for v in [lane, clamp] {
        if let Val::Reg(r, _) = v {
            if s.regs
                .get(&reg_key(m, t, *r))
                .is_some_and(|&e| s.ar.dep[e as usize])
            {
                return Err("shuffle with an input-dependent lane".into());
            }
        }
    }
    let (Ok(c), Ok(b)) = (m.rd(t, clamp), m.rd(t, lane)) else {
        return Err("shuffle".into());
    };
    let c = c as u32;
    let me = m.threads[t].lane;
    let seg = (c >> 8) & 31;
    let min_lane = me & seg;
    let max_lane = min_lane | (c & 31 & !seg);
    let b = b as u32 & 31;
    let (j, ok) = match mode {
        ptxv_ir::ShflMode::Idx => {
            let j = min_lane | (b & !seg);
            (j, j <= max_lane)
        }
        ptxv_ir::ShflMode::Up => (me.wrapping_sub(b), me >= max_lane + b),
        ptxv_ir::ShflMode::Down => (me + b, me + b <= max_lane),
        ptxv_ir::ShflMode::Bfly => ((me ^ b) & 31, (me ^ b) <= max_lane),
    };
    let src = if ok { j } else { me };
    let warp = m.threads[t].warp;
    let Some(&st) = active
        .iter()
        .find(|&&u| m.threads[u].warp == warp && m.threads[u].lane == src)
    else {
        return Err("shuffle from a lane not executing it".into());
    };
    Ok(opnd(s, m, st, &[], a))
}

fn local_opaque(s: &mut SymState) -> u32 {
    let id = s.ar.opaque.len() as u32;
    s.ar.opaque.push(OpaqueLeaf {
        why: "wide local store".into(),
        pc: s.ar.pc,
    });
    id
}

/// After `pc` ran (coverage.rs `post`, after the provenance update): `clean(t, r)` says
/// whether the provenance of destination `r` is free of input data.
pub(crate) fn post(s: &mut SymState, m: &Machine, pc: u32, clean: &dyn Fn(usize, Reg) -> bool) {
    let k = m.k.clone();
    let inst = &k.code[pc as usize];
    s.ar.pc = pc;
    // RZ / URZ destinations discard the value.
    let dsts: Vec<Reg> = inst
        .op
        .dst_regs()
        .into_iter()
        .filter(|r| !(matches!(r.file, File::R | File::UR) && r.idx >= 255))
        .collect();
    let all_dsts = inst.op.dst_regs();
    let pending = std::mem::take(&mut s.pending);
    if matches!(inst.op, Op::CpAsync { .. }) {
        for &(t, _) in &pending {
            cp_issue(s, m, t, pc);
        }
    }
    for (t, pend) in pending {
        // R2UR without a predicate: the first active lane's value.
        if matches!(inst.op, Op::R2Ur { p: None, .. }) && t != s.first {
            continue;
        }
        let mut out: Vec<(Reg, E)> = Vec::new();
        let mut forced: Vec<Reg> = Vec::new();
        match pend {
            Pend::Op(snap) => {
                if let Some(snap) = snap {
                    out = model(s, m, t, pc, &inst.op, &snap);
                    // Destinations the model left out: re-run the instruction.
                    out.retain(|x| {
                        dsts.contains(&x.0) || matches!(x.0.file, File::P | File::UP | File::V)
                    });
                    for (j, &d) in all_dsts.iter().enumerate() {
                        if !dsts.contains(&d) || out.iter().any(|x| x.0 == d) || clean(t, d) {
                            continue;
                        }
                        let e = inst_node(s, m, t, pc, &inst.op, &snap, j, d);
                        out.push((d, e));
                    }
                    // Check every node against the simulator; a mismatch re-runs the
                    // instruction instead.
                    for (j, x) in out.iter_mut().enumerate() {
                        let want = reg_val(m, t, x.0) & mask(reg_w(x.0));
                        if s.ar.conc[x.1 as usize] != want {
                            s.n_mismatch += 1;
                            if s.mismatches.len() < 64 {
                                s.mismatches.push((
                                    pc,
                                    format!(
                                        "{} t{t} {:?}{}: node {:#x} simulator {:#x}: {}",
                                        m.k.describe(pc as usize),
                                        x.0.file,
                                        x.0.idx,
                                        s.ar.conc[x.1 as usize],
                                        want,
                                        s.ar.show(x.1, &|a| format!("a{a}"), 6)
                                    ),
                                ));
                            }
                            let jj = all_dsts.iter().position(|&d| d == x.0).unwrap_or(j);
                            x.1 = inst_node(s, m, t, pc, &inst.op, &snap, jj, x.0);
                        }
                    }
                }
            }
            Pend::Shfl(src) => {
                let Op::Shfl { d, .. } = &inst.op else {
                    unreachable!()
                };
                let v = reg_val(m, t, *d) & mask(reg_w(*d));
                let e = match src {
                    Ok(e) => {
                        let x = write(s, *d, e, 1)[0].1;
                        if s.ar.conc[x as usize] == v {
                            x
                        } else {
                            s.n_mismatch += 1;
                            s.ar.opaque(reg_w(*d), v, "shuffle not matched to the source lane")
                        }
                    }
                    Err(why) if !clean(t, *d) => {
                        *s.n_opaque.entry(why.clone()).or_default() += 1;
                        s.ar.opaque(reg_w(*d), v, why)
                    }
                    Err(_) => s.ar.konst(reg_w(*d), v),
                };
                if dsts.contains(d) {
                    out.push((*d, e));
                }
            }
            Pend::Load { bytes, sext, addr } => {
                let acc = s.acc.remove(&(t as u32)).unwrap_or_default();
                let n = dsts.len().max(1);
                let per = (bytes as usize / n).max(1);
                if acc.len() == bytes as usize {
                    for (i, &d) in dsts.iter().enumerate() {
                        let part = &acc[(i * per).min(acc.len())..((i + 1) * per).min(acc.len())];
                        let v = reg_val(m, t, d);
                        let e = load_expr(s, m, part, v);
                        let w = reg_w(d);
                        let e = if s.ar.w(e) < w {
                            if sext {
                                s.ar.sext(w, e)
                            } else {
                                s.ar.zext(w, e)
                            }
                        } else {
                            s.ar.extract(0, w, e)
                        };
                        let want = v & mask(w);
                        let e = if s.ar.conc[e as usize] != want {
                            s.n_mismatch += 1;
                            if s.mismatches.len() < 64 {
                                let kinds: Vec<String> = part
                                    .iter()
                                    .map(|b| match b {
                                        Src::Init => "init".into(),
                                        Src::In(a, o) => format!("in{a}:{o}"),
                                        Src::K(k) => format!("k{k:x}"),
                                        Src::B(e, j) => format!(
                                            "{}.{j}",
                                            s.ar.show(*e, &|a| format!("a{a}"), 2)
                                        ),
                                        Src::O(_) => "o".into(),
                                        Src::G(i) => format!("g{i}"),
                                    })
                                    .collect();
                                s.mismatches.push((
                                    pc,
                                    format!(
                                        "{} t{t}: load bytes {:#x} vs {:#x}: {}",
                                        m.k.describe(pc as usize),
                                        s.ar.conc[e as usize],
                                        want,
                                        kinds.join(" ")
                                    ),
                                ));
                            }
                            s.ar.opaque(w, want, "load not matched to the bytes read")
                        } else {
                            e
                        };
                        // Read through an input-dependent address: defined at that address only.
                        let e = match addr {
                            Some(a) => {
                                let at = s.ar.conc[a as usize];
                                s.ar.mk_pub(
                                    Node::AtAddr {
                                        addr: a,
                                        at,
                                        val: e,
                                    },
                                    w,
                                )
                            }
                            _ => e,
                        };
                        out.push((d, e));
                    }
                } else {
                    for &d in &dsts {
                        if !clean(t, d) {
                            let v = reg_val(m, t, d);
                            let e =
                                s.ar.opaque(reg_w(d), v, "load of bytes the run did not record");
                            out.push((d, e));
                        }
                    }
                }
            }
            Pend::AtomAdd { bytes } => {
                let acc = s.acc.remove(&(t as u32)).unwrap_or_default();
                let d = dsts.first().copied();
                if let (Some(d), Some(Src::In(a, o))) = (d, acc.first().copied()) {
                    let contiguous = acc.len() == bytes as usize
                        && acc
                            .iter()
                            .enumerate()
                            .all(|(i, b)| *b == Src::In(a, o + i as u64));
                    let old = reg_val(m, t, d) & mask(8 * bytes);
                    if contiguous {
                        let init = &m.launch.allocs[a as usize].init;
                        let mut iv = 0u64;
                        for i in 0..bytes as usize {
                            iv |= (*init.get(o as usize + i).unwrap_or(&0) as u64) << (8 * i);
                        }
                        let leaf = s.ar.input(a, o, bytes, iv);
                        let delta = s.ar.opaque(
                            8 * bytes,
                            old.wrapping_sub(iv),
                            "other threads' atomic adds before this one",
                        );
                        let e = s.ar.bin(Bin::Add, leaf, delta);
                        let e = s.ar.zext(reg_w(d), e);
                        out.push((d, e));
                    }
                }
                for &d in &dsts {
                    if !clean(t, d) && !out.iter().any(|x| x.0 == d) {
                        let v = reg_val(m, t, d);
                        let e = s.ar.opaque(reg_w(d), v, "atomic result");
                        out.push((d, e));
                    }
                }
            }
            Pend::Local(srcs) => {
                for (i, &d) in dsts.iter().enumerate() {
                    let v = reg_val(m, t, d);
                    let w = reg_w(d);
                    let e = match srcs.get(i) {
                        Some(src) => {
                            let e = bytes_expr(s, m, src, v);
                            let e = s.ar.zext(w, e);
                            if s.ar.conc[e as usize] == v & mask(w) {
                                e
                            } else {
                                s.ar.opaque(w, v, "local load")
                            }
                        }
                        None => s.ar.opaque(w, v, "local load"),
                    };
                    out.push((d, e));
                }
            }
            Pend::ForceOpaque(why) => {
                for &d in dsts
                    .iter()
                    .filter(|d| !matches!(d.file, File::P | File::UP))
                {
                    let v = reg_val(m, t, d);
                    *s.n_opaque.entry(why.clone()).or_default() += 1;
                    let e = s.ar.opaque(reg_w(d), v, why.clone());
                    out.push((d, e));
                    forced.push(d);
                }
            }
            Pend::Opaque(why) => {
                for &d in &dsts {
                    if !clean(t, d) {
                        let v = reg_val(m, t, d);
                        let why = if why.is_empty() {
                            op_reason(&inst.op)
                        } else {
                            why.clone()
                        };
                        *s.n_opaque.entry(why.clone()).or_default() += 1;
                        let e = s.ar.opaque(reg_w(d), v, why);
                        out.push((d, e));
                    }
                }
            }
        }
        // An input-dependent guard: `ite(guard, new, old)` (the guard held here).
        if let Some((g, old)) = s.gold.get(&t).cloned() {
            for (d, o) in old {
                let w = reg_w(d);
                let new = match out.iter().find(|x| x.0 == d) {
                    Some(&(_, e)) => s.ar.zext(w, e),
                    None => s.ar.konst(w, reg_val(m, t, d)),
                };
                let e = s.ar.ite(g, new, o);
                out.retain(|x| x.0 != d);
                out.push((d, e));
                forced.push(d);
            }
        }
        record_fp(s, m, t, pc, &out);
        // Clean provenance: no expression (a constant) whatever the model built. A uniform
        // destination (one per warp) that another member gave an expression in this
        // instruction keeps it (a warp reduction is input-derived when any lane's operand is).
        // Kept whatever the provenance: values with input leaves (the provenance can miss a
        // path, e.g. bytes a cp.async landed during a wait), under an input-dependent guard
        // or address, and undefined ones.
        for &d in &dsts {
            let k = reg_key(m, t, d);
            let shared = d.is_uniform() && s.uset.contains(&k);
            if let Some(&(_, e)) = out.iter().find(|x| x.0 == d) {
                if s.ar.dinp[e as usize]
                    && clean(t, d)
                    && !forced.contains(&d)
                    && !matches!(s.ar.nodes[e as usize], Node::AtAddr { .. })
                {
                    s.n_prov_gap += 1;
                    if s.prov_gaps.len() < 16 {
                        let txt = format!(
                            "{} t{t} {:?}{}: {}",
                            m.k.describe(pc as usize),
                            d.file,
                            d.idx,
                            s.ar.show(e, &|a| format!("a{a}"), 4)
                        );
                        s.prov_gaps.push((pc, txt));
                    }
                }
            }
            match out.iter().find(|x| x.0 == d) {
                Some(&(_, e))
                    if !clean(t, d)
                        || s.ar.is_const(e).is_some()
                        || s.ar.placed[e as usize]
                        || s.ar.undef[e as usize]
                        || s.ar.inp[e as usize]
                        || forced.contains(&d) && s.ar.dep[e as usize]
                        || matches!(s.ar.nodes[e as usize], Node::AtAddr { .. }) =>
                {
                    if s.ar.is_const(e).is_some() && shared {
                        continue;
                    }
                    s.set(m, t, d, e);
                    if d.is_uniform() && s.ar.is_const(e).is_none() {
                        s.uset.insert(k);
                    }
                }
                _ if shared => {}
                _ => {
                    s.regs.remove(&k);
                }
            }
        }
        // Path atoms: comparison / test predicates (sites) and selectors.
        if site_op(&inst.op) {
            for &d in &dsts {
                if !matches!(d.file, File::P | File::UP) || is_pt(m, d) {
                    continue;
                }
                let Some(&e) = s.regs.get(&reg_key(m, t, d)) else {
                    continue;
                };
                let v = reg_val(m, t, d) == 1;
                if let Some(i) = s.push_atom(m, t, pc, e, v, AtomKind::Site) {
                    let warp = m.threads[t].warp;
                    let max = s.cfg.max_examples;
                    let ex = s.sites.entry(pc).or_default();
                    let l = &mut ex[v as usize];
                    if l.len() < max && !l.iter().any(|o| o.t == t as u32) {
                        l.push(Obs {
                            t: t as u32,
                            warp,
                            atom: i,
                        });
                    }
                }
            }
        }
    }
    // Guard-false members of an input-dependent guard: `ite(guard, would-be, old)` (the
    // would-be value from the model, without path atoms; an opaque leaf when not modelled).
    let ghosts = std::mem::take(&mut s.ghost);
    s.quiet = true;
    for (t, pend) in ghosts {
        let Some((g, old)) = s.gold.get(&t).cloned() else {
            continue;
        };
        let mut out = match pend {
            Pend::Op(Some(snap)) => model(s, m, t, pc, &inst.op, &snap),
            _ => vec![],
        };
        out.retain(|x| dsts.contains(&x.0));
        for (d, o) in old {
            let w = reg_w(d);
            let new = match out.iter().find(|x| x.0 == d) {
                Some(&(_, e)) => s.ar.zext(w, e),
                None => s.ar.opaque(
                    w,
                    0,
                    "would-be value of a guarded-off instruction (not modelled)",
                ),
            };
            let mut e = s.ar.ite(g, new, o);
            let want = reg_val(m, t, d) & mask(w);
            if s.ar.conc[e as usize] != want {
                s.n_mismatch += 1;
                e = s.ar.opaque(w, want, "guarded-off destination not matched");
            }
            s.set(m, t, d, e);
        }
    }
    s.quiet = false;
}

/// After thread `t` issued a cp.async at `pc`: the bytes its copies will land (what it read
/// now; zeros for the zero-filled tail), keyed by their shared destination.
fn cp_issue(s: &mut SymState, m: &Machine, t: usize, pc: u32) {
    let reads = s.acc.remove(&(t as u32)).unwrap_or_default();
    let th = &m.threads[t];
    let cta = th.cta as u64;
    let mut q = th.cp_pending.iter().rev();
    let Some(last) = q.next().filter(|c| c.pc == pc) else {
        return;
    };
    let data = if last.src.is_none() {
        s.cp_stash.insert(
            (t as u32, cta, last.off as u64),
            vec![Src::K(0); last.data.len()],
        );
        q.next()
            .filter(|c| c.pc == pc && c.src.is_some() && c.off + c.data.len() as u32 == last.off)
    } else {
        Some(last)
    };
    if let Some(c) = data {
        if c.data.len() == reads.len() {
            s.cp_stash.insert((t as u32, cta, c.off as u64), reads);
        }
    }
}

/// Input-dependent floating-point operations of `out` (implicit-site observations): per
/// lane, the operation node, its operands after modifiers and its result.
fn record_fp(s: &mut SymState, m: &Machine, t: usize, pc: u32, out: &[(Reg, E)]) {
    let max = s.cfg.max_examples;
    if s.fp_obs.get(&pc).is_some_and(|v| v.len() >= 4 * max) {
        return;
    }
    let mut found: Vec<E> = Vec::new();
    for &(_, e) in out {
        let mut st = vec![(e, 0u32)];
        while let Some((x, depth)) = st.pop() {
            if !s.ar.dep[x as usize] || depth > 6 || found.contains(&x) {
                continue;
            }
            match &s.ar.nodes[x as usize] {
                Node::Fma { .. }
                | Node::FMinMax { .. }
                | Node::FCmp { .. }
                | Node::Mufu { .. }
                | Node::DFma { .. }
                | Node::DCmp { .. }
                | Node::HFma { .. }
                | Node::Narrow { .. }
                | Node::FCvt { .. }
                | Node::F2F { .. }
                | Node::I2F { .. }
                | Node::F2I { .. }
                | Node::Pack { .. }
                | Node::Un {
                    op: Un::FSat | Un::H2F | Un::B2F,
                    ..
                } => found.push(x),
                // Lane assembly, selects, predicate combination: look through (low lane first).
                Node::Concat { hi, lo } => {
                    st.push((*hi, depth + 1));
                    st.push((*lo, depth + 1));
                }
                Node::Ite { a, b, .. } => {
                    st.push((*b, depth + 1));
                    st.push((*a, depth + 1));
                }
                Node::Bin {
                    op: Bin::And | Bin::Or | Bin::Xor,
                    w: 1,
                    a,
                    b,
                } => {
                    st.push((*b, depth + 1));
                    st.push((*a, depth + 1));
                }
                Node::Extract { a, .. } | Node::Zext { a, .. } | Node::Sext { a, .. } => {
                    st.push((*a, depth + 1))
                }
                _ => {}
            }
        }
    }
    if found.is_empty() {
        return;
    }
    let warp = m.threads[t].warp;
    let v = s.fp_obs.entry(pc).or_default();
    for (lane, &x) in found.iter().enumerate() {
        let operands = kids(&s.ar.nodes[x as usize]);
        v.push(FpObs {
            t: t as u32,
            warp,
            lane: lane as u8,
            node: x,
            operands,
        });
    }
}

/// Select-predicate atoms (called from `post` per thread before the destinations change):
/// recorded in `model`.
fn select_atom(s: &mut SymState, m: &Machine, t: usize, pc: u32, p: E) {
    if s.ar.dep[p as usize] {
        let v = s.ar.conc[p as usize] & 1 == 1;
        s.push_atom(m, t, pc, p, v, AtomKind::Select);
    }
}

fn op_reason(op: &Op) -> String {
    let d = format!("{op:?}");
    let name = d.split([' ', '{', '(']).next().unwrap_or("").to_string();
    match op {
        Op::Shfl { .. } => "shuffle".into(),
        Op::VoteAny { .. } | Op::VoteAll { .. } | Op::MatchAny { .. } | Op::Redux { .. } => {
            "warp-collective result".into()
        }
        Op::Atom { .. } => "atomic result".into(),
        Op::TmemLd { .. } => "TMEM".into(),
        _ => format!("not modelled ({name})"),
    }
}

/// An `Inst` node for destination `j` (`d`) of `op`, or an opaque leaf when `op` is not a
/// per-thread register operation.
#[allow(clippy::too_many_arguments)]
fn inst_node(
    s: &mut SymState,
    m: &Machine,
    t: usize,
    pc: u32,
    op: &Op,
    snap: &[(Reg, E)],
    j: usize,
    d: Reg,
) -> E {
    let v = reg_val(m, t, d);
    let w = reg_w(d);
    if !per_thread_pure(op) {
        let why = op_reason(op);
        *s.n_opaque.entry(why.clone()).or_default() += 1;
        return s.ar.opaque(w, v, why);
    }
    s.n_inst += 1;
    let args: Vec<E> = snap.iter().map(|x| x.1).collect();
    s.ar.inst(pc, t as u32, j as u8, w, args, v)
}

/// Writes of value `e` to `d` over `slots` 32-bit registers (a virtual register: one).
fn write(s: &mut SymState, d: Reg, e: E, slots: u8) -> Vec<(Reg, E)> {
    match d.file {
        File::V => {
            let x = s.ar.zext(64, e);
            vec![(d, x)]
        }
        File::P | File::UP => {
            let x = s.ar.extract(0, 1, e);
            vec![(d, x)]
        }
        _ => {
            let x = s.ar.zext(64, e);
            (0..slots.max(1) as u16)
                .map(|i| {
                    let y = s.ar.extract(32 * i as u8, 32, x);
                    (
                        Reg {
                            file: d.file,
                            idx: d.idx + i,
                        },
                        y,
                    )
                })
                .collect()
        }
    }
}

/// The 16 bits lane `lane` of a 64-bit operand value supplies under selector `sel`.
fn half_lane(s: &mut SymState, v: E, sel: HSel, lane: u32) -> E {
    let h = match sel {
        HSel::Pair => lane,
        HSel::H0 => 0,
        HSel::H1 => 1,
    };
    s.ar.extract(16 * h as u8, 16, v)
}

fn fmod(s: &mut SymState, x: E, neg: bool, abs: bool) -> E {
    let x = if abs { s.ar.un(Un::FAbs, x) } else { x };
    if neg { s.ar.un(Un::FNeg, x) } else { x }
}

/// The destinations of `op` the model computes natively (the rest: `Inst`).
fn model(
    s: &mut SymState,
    m: &Machine,
    t: usize,
    pc: u32,
    op: &Op,
    snap: &[(Reg, E)],
) -> Vec<(Reg, E)> {
    macro_rules! o {
        ($v:expr) => {
            opnd(s, m, t, snap, $v)
        };
    }
    macro_rules! p {
        ($v:expr) => {
            pred_e(s, m, t, snap, $v)
        };
    }
    let lo32 = |s: &mut SymState, e: E| s.ar.extract(0, 32, e);
    match op {
        Op::Mov { d, a } if !matches!(a, Val::Const(_, 16)) => {
            let e = o!(a);
            let slots = match a {
                Val::Reg(_, n) => *n,
                Val::Const(_, b) => (*b / 4).max(1),
                _ => 1,
            };
            write(s, *d, e, slots)
        }
        Op::Int {
            op: iop,
            bits,
            d,
            a,
            b,
        } => {
            let (xa, xb) = (o!(a), o!(b));
            let w = *bits;
            let x = s.ar.extract(0, w, xa);
            let y = s.ar.extract(0, w, xb);
            use ptxv_ir::IntOp as I;
            let r = match iop {
                I::Add => s.ar.bin(Bin::Add, x, y),
                I::Sub => s.ar.bin(Bin::Sub, x, y),
                I::Mul => s.ar.bin(Bin::Mul, x, y),
                I::And => s.ar.bin(Bin::And, x, y),
                I::Or => s.ar.bin(Bin::Or, x, y),
                I::Xor => s.ar.bin(Bin::Xor, x, y),
                I::Shl => s.ar.bin(Bin::Shl, x, y),
                I::ShrU => s.ar.bin(Bin::LShr, x, y),
                I::ShrS => s.ar.bin(Bin::AShr, x, y),
                I::DivU => s.ar.bin(Bin::UDiv, x, y),
                I::DivS => s.ar.bin(Bin::SDiv, x, y),
                I::RemU => s.ar.bin(Bin::URem, x, y),
                I::RemS => s.ar.bin(Bin::SRem, x, y),
                I::MinU => s.ar.bin(Bin::UMin, x, y),
                I::MaxU => s.ar.bin(Bin::UMax, x, y),
                I::MinS => s.ar.bin(Bin::SMin, x, y),
                I::MaxS => s.ar.bin(Bin::SMax, x, y),
                I::MulHiU => s.ar.bin(Bin::MulHiU, x, y),
                I::MulHiS => s.ar.bin(Bin::MulHiS, x, y),
                I::Popc => s.ar.un(Un::Popc, x),
                I::Abs => {
                    let z = s.ar.konst(w, 0);
                    let neg = s.ar.cmp(Cmp::LtS, x, z);
                    let nx = s.ar.un(Un::Neg, x);
                    s.ar.ite(neg, nx, x)
                }
                I::Add16x2 if w == 32 => {
                    let (xl, yl) = (s.ar.extract(0, 16, x), s.ar.extract(0, 16, y));
                    let (xh, yh) = (s.ar.extract(16, 16, x), s.ar.extract(16, 16, y));
                    let l = s.ar.bin(Bin::Add, xl, yl);
                    let h = s.ar.bin(Bin::Add, xh, yh);
                    s.ar.concat(h, l)
                }
                _ => return vec![],
            };
            write(s, *d, r, (*bits / 32).max(1))
        }
        Op::MulWide {
            d,
            a,
            b,
            signed,
            bits,
        } => {
            if *bits != 32 {
                // mul.wide.{s,u}16: not modelled here; the node is re-run (exact).
                return vec![];
            }
            let (xa, xb) = (o!(a), o!(b));
            let (x, y) = (lo32(s, xa), lo32(s, xb));
            let (x, y) = if *signed {
                (s.ar.sext(64, x), s.ar.sext(64, y))
            } else {
                (s.ar.zext(64, x), s.ar.zext(64, y))
            };
            let r = s.ar.bin(Bin::Mul, x, y);
            write(s, *d, r, 2)
        }
        Op::Imad {
            d,
            a,
            b,
            c,
            neg_c,
            not_c,
            wide,
            hi,
            signed,
            cin,
            ..
        } => {
            let (xa, xb) = (o!(a), o!(b));
            let (x, y) = (lo32(s, xa), lo32(s, xb));
            let mut c0 = o!(c);
            if *not_c {
                let l = lo32(s, c0);
                let n = s.ar.bnot(l);
                c0 = s.ar.zext(64, n);
            }
            if *neg_c {
                c0 = s.ar.un(Un::Neg, c0);
            }
            let ci = match cin {
                Some(p) => {
                    let q = p!(*p);
                    s.ar.zext(64, q)
                }
                None => s.ar.konst(64, 0),
            };
            if *wide || *hi {
                let (x, y) = if *signed {
                    (s.ar.sext(64, x), s.ar.sext(64, y))
                } else {
                    (s.ar.zext(64, x), s.ar.zext(64, y))
                };
                let p = s.ar.bin(Bin::Mul, x, y);
                let s1 = s.ar.bin(Bin::Add, p, c0);
                let s2 = s.ar.bin(Bin::Add, s1, ci);
                if *hi {
                    let h = s.ar.extract(32, 32, s2);
                    write(s, *d, h, 1)
                } else {
                    write(s, *d, s2, 2)
                }
            } else {
                let p = s.ar.bin(Bin::Mul, x, y);
                let c32 = lo32(s, c0);
                let ci32 = lo32(s, ci);
                let s1 = s.ar.bin(Bin::Add, p, c32);
                let s2 = s.ar.bin(Bin::Add, s1, ci32);
                write(s, *d, s2, 1)
            }
        }
        Op::Iadd3 {
            d,
            a,
            b,
            c,
            neg,
            not,
            wide,
            cin,
            ..
        } => {
            let w = if *wide { 64 } else { 32 };
            let mut sum = s.ar.konst(w, 0);
            for ((v, n), c) in [a, b, c].iter().zip(neg).zip(not) {
                let xv = o!(v);
                let mut x = s.ar.extract(0, w, xv);
                if *c {
                    x = s.ar.bnot(x);
                }
                if *n {
                    x = s.ar.un(Un::Neg, x);
                }
                sum = s.ar.bin(Bin::Add, sum, x);
            }
            for p in cin {
                let q = p!(*p);
                let q = s.ar.zext(w, q);
                sum = s.ar.bin(Bin::Add, sum, q);
            }
            write(s, *d, sum, if *wide { 2 } else { 1 })
        }
        Op::CvtInt {
            d,
            a,
            from_bits,
            to_bits,
            signed,
        } => {
            let xa = o!(a);
            let v = s.ar.extract(0, *from_bits, xa);
            let v = if *signed {
                s.ar.sext(64, v)
            } else {
                s.ar.zext(64, v)
            };
            let r = s.ar.extract(0, *to_bits, v);
            write(s, *d, r, (*to_bits / 32).max(1))
        }
        Op::Sel { d, a, b, p } => {
            let c = p!(*p);
            select_atom(s, m, t, pc, c);
            let (xa, xb) = (o!(a), o!(b));
            let r = s.ar.ite(c, xa, xb);
            let r = if d.file == File::V { r } else { lo32(s, r) };
            write(s, *d, r, 1)
        }
        Op::Fsel {
            d,
            a,
            b,
            p,
            neg,
            abs,
        } => {
            let c = p!(*p);
            select_atom(s, m, t, pc, c);
            let (xa, xb) = (o!(a), o!(b));
            let (x, y) = (lo32(s, xa), lo32(s, xb));
            let x = fmod(s, x, neg[0], abs[0]);
            let y = fmod(s, y, neg[1], abs[1]);
            let r = s.ar.ite(c, x, y);
            write(s, *d, r, 1)
        }
        Op::Prmt { d, a, sel, b } => {
            let sv = o!(sel);
            let Some(sc) = s.ar.is_const(sv) else {
                return vec![];
            };
            let (xa, xb) = (o!(a), o!(b));
            let (la, lb) = (lo32(s, xa), lo32(s, xb));
            let x = s.ar.concat(lb, la);
            let mut acc: Option<E> = None;
            for i in 0..4 {
                let n = ((sc >> (4 * i)) & 0xf) as u8;
                let byte = if n & 8 != 0 {
                    let sb = s.ar.extract(8 * (n & 7) + 7, 1, x);
                    s.ar.sext(8, sb)
                } else {
                    s.ar.extract(8 * (n & 7), 8, x)
                };
                acc = Some(match acc {
                    None => byte,
                    Some(lo) => s.ar.concat(byte, lo),
                });
            }
            let r = acc.unwrap();
            write(s, *d, r, 1)
        }
        Op::Bfe {
            d,
            a,
            pos,
            len,
            signed,
            bits,
        } => {
            if *bits != 32 {
                // bfe.{u,s}64: not modelled here; the node is re-run (exact).
                return vec![];
            }
            let (pv, lv) = (o!(pos), o!(len));
            let (Some(p), Some(l)) = (s.ar.is_const(pv), s.ar.is_const(lv)) else {
                return vec![];
            };
            let (p, l) = ((p & 0xff) as u8, (l & 0xff) as u8);
            let xa = o!(a);
            let x = lo32(s, xa);
            let r = if l == 0 {
                s.ar.konst(32, 0)
            } else if p >= 32 {
                if *signed {
                    let sb = s.ar.extract(31, 1, x);
                    s.ar.sext(32, sb)
                } else {
                    s.ar.konst(32, 0)
                }
            } else {
                let width = l.min(32 - p);
                let f = s.ar.extract(p, width, x);
                if *signed && l < 32 {
                    s.ar.sext(32, f)
                } else {
                    s.ar.zext(32, f)
                }
            };
            write(s, *d, r, 1)
        }
        Op::Shf {
            d,
            lo,
            sh,
            hi,
            right,
            signed,
            high,
            wide,
            wrap,
        } => {
            let (xl, xh, xs) = (o!(lo), o!(hi), o!(sh));
            let (l, h) = (lo32(s, xl), lo32(s, xh));
            let v = s.ar.concat(h, l);
            let s32 = lo32(s, xs);
            let n = if *wrap {
                let k = s.ar.konst(32, 31);
                s.ar.bin(Bin::And, s32, k)
            } else {
                let k = s.ar.konst(32, if *wide { 64 } else { 32 });
                s.ar.bin(Bin::UMin, s32, k)
            };
            let n = s.ar.zext(64, n);
            let r = match (*right, *signed) {
                (true, true) => s.ar.bin(Bin::AShr, v, n),
                (true, false) => s.ar.bin(Bin::LShr, v, n),
                (false, _) => s.ar.bin(Bin::Shl, v, n),
            };
            let r = if *high {
                s.ar.extract(32, 32, r)
            } else {
                lo32(s, r)
            };
            write(s, *d, r, 1)
        }
        Op::Lea {
            d,
            a,
            b,
            shift,
            neg_a,
            neg_b,
            ..
        } => {
            let (xa, xb) = (o!(a), o!(b));
            let x = lo32(s, xa);
            let k = s.ar.konst(32, *shift as u64);
            let mut sa = s.ar.bin(Bin::Shl, x, k);
            if *neg_a {
                sa = s.ar.un(Un::Neg, sa);
            }
            let mut bv = lo32(s, xb);
            if *neg_b {
                bv = s.ar.un(Un::Neg, bv);
            }
            let r = s.ar.bin(Bin::Add, sa, bv);
            write(s, *d, r, 1)
        }
        Op::Lop3 {
            d,
            a,
            b,
            c,
            lut,
            p,
            pq,
        } => {
            let (xa, xb, xc) = (o!(a), o!(b), o!(c));
            let (x, y, z) = (lo32(s, xa), lo32(s, xb), lo32(s, xc));
            let r = s.ar.lop3(*lut, x, y, z);
            let mut out = write(s, *d, r, 1);
            if let Some(p) = p {
                let zero = s.ar.konst(32, 0);
                let nz = s.ar.cmp(Cmp::Ne, r, zero);
                let v = match pq {
                    None => nz,
                    Some((q, and)) => {
                        let qe = p!(*q);
                        s.ar.bin(if *and { Bin::And } else { Bin::Or }, nz, qe)
                    }
                };
                out.push((*p, v));
            }
            out
        }
        Op::Plop3 {
            d,
            a,
            b,
            c,
            lut,
            d2,
        } => {
            let (x, y, z) = (p!(*a), p!(*b), p!(*c));
            let r = s.ar.lop3(*lut, x, y, z);
            let mut out = vec![(*d, r)];
            if let Some((d2, l2)) = d2 {
                let r2 = s.ar.lop3(*l2, x, y, z);
                out.push((*d2, r2));
            }
            out
        }
        Op::Setp {
            d,
            cmp,
            bits,
            a,
            b,
            and,
            or,
            xor,
            ex,
        } => {
            let (xa, xb) = (o!(a), o!(b));
            let x = s.ar.extract(0, *bits, xa);
            let y = s.ar.extract(0, *bits, xb);
            let mut r = s.ar.cmp(*cmp, x, y);
            if let Some(p) = ex {
                let pe = p!(*p);
                let eq = s.ar.cmp(Cmp::Eq, x, y);
                r = s.ar.ite(eq, pe, r);
            }
            let q = p!(*and);
            let v = if *xor {
                s.ar.bin(Bin::Xor, r, q)
            } else if *or {
                s.ar.bin(Bin::Or, r, q)
            } else {
                s.ar.bin(Bin::And, r, q)
            };
            vec![(*d, v)]
        }
        Op::Fsetp {
            d,
            cmp,
            a,
            b,
            neg,
            abs,
            c,
            bop,
            ftz,
        } => {
            let (xa, xb) = (o!(a), o!(b));
            let (x, y) = (lo32(s, xa), lo32(s, xb));
            let x = fmod(s, x, neg[0], abs[0]);
            let y = fmod(s, y, neg[1], abs[1]);
            let r = s.ar.fcmp(*cmp, *ftz, x, y);
            let q = p!(*c);
            let v = match bop {
                ptxv_ir::BoolOp::And => s.ar.bin(Bin::And, r, q),
                ptxv_ir::BoolOp::Or => s.ar.bin(Bin::Or, r, q),
                ptxv_ir::BoolOp::Xor => s.ar.bin(Bin::Xor, r, q),
            };
            vec![(*d, v)]
        }
        Op::Fadd {
            d,
            a,
            b,
            rnd,
            neg,
            abs,
            ftz,
            sat,
        } => {
            let (xa, xb) = (o!(a), o!(b));
            let (x, y) = (lo32(s, xa), lo32(s, xb));
            let x = fmod(s, x, neg[0], abs[0]);
            let y = fmod(s, y, neg[1], abs[1]);
            let one = s.ar.konst(32, 0x3f80_0000);
            let mut r = s.ar.fma(x, one, Some(y), *rnd, *ftz);
            if *sat {
                r = s.ar.un(Un::FSat, r);
            }
            write(s, *d, r, 1)
        }
        Op::Ffma {
            d,
            a,
            b,
            c,
            neg,
            abs,
            rnd,
            ftz,
            sat,
        } => {
            let (xa, xb) = (o!(a), o!(b));
            let (x, y) = (lo32(s, xa), lo32(s, xb));
            let x = fmod(s, x, neg[0], abs[0]);
            let y = fmod(s, y, neg[1], abs[1]);
            let z = match c {
                Some(c) => {
                    let xc = o!(c);
                    let z = lo32(s, xc);
                    Some(fmod(s, z, neg[2], abs[2]))
                }
                None => None,
            };
            let mut r = s.ar.fma(x, y, z, *rnd, *ftz);
            if *sat {
                r = s.ar.un(Un::FSat, r);
            }
            write(s, *d, r, 1)
        }
        Op::Fmnmx {
            d,
            a,
            b,
            c,
            neg,
            abs,
            min,
            ftz,
            nan,
            xorsign: false,
        } => {
            let lo = p!(*min);
            if s.ar.is_const(lo).is_none() {
                select_atom(s, m, t, pc, lo);
            }
            let (xa, xb) = (o!(a), o!(b));
            let (x, y) = (lo32(s, xa), lo32(s, xb));
            let x = fmod(s, x, neg[0], abs[0]);
            let y = fmod(s, y, neg[1], abs[1]);
            let z = match c {
                Some(c) => {
                    let xc = o!(c);
                    let z = lo32(s, xc);
                    Some(fmod(s, z, neg[2], abs[2]))
                }
                None => None,
            };
            let build = |s: &mut SymState, mn: bool| {
                let r = s.ar.fminmax(mn, *nan, *ftz, x, y);
                match z {
                    Some(z) => s.ar.fminmax(mn, *nan, *ftz, r, z),
                    None => r,
                }
            };
            let r = match s.ar.is_const(lo) {
                Some(v) => build(s, v & 1 == 1),
                None => {
                    let rmin = build(s, true);
                    let rmax = build(s, false);
                    s.ar.ite(lo, rmin, rmax)
                }
            };
            write(s, *d, r, 1)
        }
        Op::Mufu {
            d,
            a,
            func,
            fmt,
            hi,
            neg,
            abs,
            ptx,
        } if *fmt != MufuFmt::F64 => {
            let xa = o!(a);
            let x = lo32(s, xa);
            let x = fmod(s, x, *neg, *abs);
            let r = s.ar.mufu(*func, *fmt, *hi, *ptx, x);
            write(s, *d, r, 1)
        }
        Op::Ffma2 {
            d,
            a,
            b,
            c,
            neg,
            abs,
            rnd,
            ftz,
        } => {
            if *ftz {
                // .FTZ: not modelled here; the node is re-run (exact).
                return vec![];
            }
            // Two f32 FMAs, one per lane (exec.rs `Op::Ffma2`).
            let mut lanes = [[0 as E; 2]; 3];
            for (k, (v, l)) in [a, b, c].into_iter().enumerate() {
                let x = o!(v);
                let (lo, hi) = (s.ar.extract(0, 32, x), s.ar.extract(32, 32, x));
                let pair = match l {
                    ptxv_ir::F2Lanes::Bcast => [lo, lo],
                    ptxv_ir::F2Lanes::Pair => [lo, hi],
                    ptxv_ir::F2Lanes::Swap => [hi, lo],
                };
                lanes[k] = pair.map(|y| fmod(s, y, neg[k], abs[k]));
            }
            let r0 =
                s.ar.fma(lanes[0][0], lanes[1][0], Some(lanes[2][0]), *rnd, false);
            let r1 =
                s.ar.fma(lanes[0][1], lanes[1][1], Some(lanes[2][1]), *rnd, false);
            if d.file == File::V {
                let r = s.ar.concat(r1, r0);
                return vec![(*d, r)];
            }
            vec![
                (*d, r0),
                (
                    Reg {
                        file: d.file,
                        idx: d.idx + 1,
                    },
                    r1,
                ),
            ]
        }
        Op::CvtF32Bf16 { d, a } => {
            let x = o!(a);
            let r = s.ar.un(Un::B2F, x);
            write(s, *d, r, 1)
        }
        Op::CvtF32F16 { d, a, hi } => {
            let x = o!(a);
            let h = s.ar.extract(if *hi { 16 } else { 0 }, 16, x);
            let r = s.ar.un(Un::H2F, h);
            write(s, *d, r, 1)
        }
        Op::R2Ur { d, a, p: None, .. } => {
            let x = o!(a);
            let x = lo32(s, x);
            write(s, *d, x, 1)
        }
        Op::Pack64 { d, lo, hi } => {
            let (xl, xh) = (o!(lo), o!(hi));
            let (l, h) = (lo32(s, xl), lo32(s, xh));
            let r = s.ar.concat(h, l);
            write(s, *d, r, 2)
        }
        Op::Unpack64 { lo, hi, a } => {
            let x = o!(a);
            let mut out = Vec::new();
            if let Some(r) = lo {
                let l = lo32(s, x);
                out.extend(write(s, *r, l, 1));
            }
            if let Some(r) = hi {
                let h = s.ar.extract(32, 32, x);
                out.extend(write(s, *r, h, 1));
            }
            out
        }
        Op::Dfma {
            d,
            a,
            b,
            c,
            neg,
            abs,
            rnd,
        } => {
            let m2 = |s: &mut SymState, x: E, n: bool, ab: bool| {
                let x = if ab { s.ar.un(Un::DAbsK, x) } else { x };
                if n { s.ar.un(Un::DNegK, x) } else { x }
            };
            let (xa, xb) = (o!(a), o!(b));
            let x = m2(s, xa, neg[0], abs[0]);
            let y = m2(s, xb, neg[1], abs[1]);
            let z = match c {
                Some(c) => {
                    let xc = o!(c);
                    Some(m2(s, xc, neg[2], abs[2]))
                }
                None => None,
            };
            let r = s.ar.mk_pub(
                Node::DFma {
                    a: x,
                    b: y,
                    c: z,
                    rnd: *rnd,
                },
                64,
            );
            write(s, *d, r, 2)
        }
        Op::Dsetp {
            d,
            cmp,
            a,
            b,
            neg,
            abs,
            c,
            bop,
        } => {
            let m2 = |s: &mut SymState, x: E, n: bool, ab: bool| {
                let x = if ab {
                    let k = s.ar.konst(64, !(1u64 << 63));
                    s.ar.bin(Bin::And, x, k)
                } else {
                    x
                };
                if n {
                    let k = s.ar.konst(64, 1u64 << 63);
                    s.ar.bin(Bin::Xor, x, k)
                } else {
                    x
                }
            };
            let (xa, xb) = (o!(a), o!(b));
            let x = m2(s, xa, neg[0], abs[0]);
            let y = m2(s, xb, neg[1], abs[1]);
            let r = s.ar.mk_pub(
                Node::DCmp {
                    cmp: *cmp,
                    a: x,
                    b: y,
                },
                1,
            );
            let q = p!(*c);
            let v = match bop {
                ptxv_ir::BoolOp::And => s.ar.bin(Bin::And, r, q),
                ptxv_ir::BoolOp::Or => s.ar.bin(Bin::Or, r, q),
                ptxv_ir::BoolOp::Xor => s.ar.bin(Bin::Xor, r, q),
            };
            vec![(*d, v)]
        }
        Op::Hfma2 {
            d,
            a,
            b,
            c,
            abs_a,
            neg_a,
            neg_c,
            sel,
        } => {
            let v = [o!(a), o!(b), o!(c)];
            let mut lanes = Vec::new();
            for lane in 0..2u32 {
                let mut h = [0 as E; 3];
                for k in 0..3 {
                    let mut x = half_lane(s, v[k], sel[k], lane);
                    if k == 0 && *abs_a {
                        let km = s.ar.konst(16, 0x7fff);
                        x = s.ar.bin(Bin::And, x, km);
                    }
                    if (k == 0 && *neg_a) || (k == 2 && *neg_c) {
                        let km = s.ar.konst(16, 0x8000);
                        x = s.ar.bin(Bin::Xor, x, km);
                    }
                    h[k] = x;
                }
                lanes.push(s.ar.mk_pub(
                    Node::HFma {
                        bf16: false,
                        a: h[0],
                        b: h[1],
                        c: h[2],
                    },
                    16,
                ));
            }
            let r = s.ar.concat(lanes[1], lanes[0]);
            write(s, *d, r, 1)
        }
        Op::Hfma2Bf16 {
            d,
            a,
            b,
            c,
            sel,
            neg,
            abs,
            relu,
        } => {
            let v = [o!(a), o!(b), o!(c)];
            let mut lanes = Vec::new();
            for lane in 0..2u32 {
                let mut h = [0 as E; 3];
                for k in 0..3 {
                    let mut x = half_lane(s, v[k], sel[k], lane);
                    if abs[k] {
                        let km = s.ar.konst(16, 0x7fff);
                        x = s.ar.bin(Bin::And, x, km);
                    }
                    if neg[k] {
                        let km = s.ar.konst(16, 0x8000);
                        x = s.ar.bin(Bin::Xor, x, km);
                    }
                    h[k] = x;
                }
                let mut x = s.ar.mk_pub(
                    Node::HFma {
                        bf16: true,
                        a: h[0],
                        b: h[1],
                        c: h[2],
                    },
                    16,
                );
                if *relu {
                    let nan = s.ar.konst(16, 0x7fff);
                    let sb = s.ar.konst(16, 0x8000);
                    let z = s.ar.konst(16, 0);
                    let c1 = s.ar.cmp(Cmp::Ne, x, nan);
                    let xs = s.ar.bin(Bin::And, x, sb);
                    let c2 = s.ar.cmp(Cmp::Ne, xs, z);
                    let both = s.ar.bin(Bin::And, c1, c2);
                    x = s.ar.ite(both, z, x);
                }
                lanes.push(x);
            }
            let r = s.ar.concat(lanes[1], lanes[0]);
            write(s, *d, r, 1)
        }
        Op::Hsetp2Bf16 {
            d,
            q,
            cmp,
            a,
            b,
            sel,
            neg,
            abs,
            c,
            bop,
            f16,
        } => {
            let (xa, xb) = (o!(a), o!(b));
            let qin = p!(*c);
            let mut out = Vec::new();
            for (lane, p) in [(0u32, Some(*d)), (1, *q)] {
                let Some(p) = p else { continue };
                let mut w = [0 as E; 2];
                for (k, xv) in [xa, xb].into_iter().enumerate() {
                    let mut h = half_lane(s, xv, sel[k], lane);
                    if abs[k] {
                        let km = s.ar.konst(16, 0x7fff);
                        h = s.ar.bin(Bin::And, h, km);
                    }
                    if neg[k] {
                        let km = s.ar.konst(16, 0x8000);
                        h = s.ar.bin(Bin::Xor, h, km);
                    }
                    w[k] = if *f16 {
                        s.ar.un(Un::H2F, h)
                    } else {
                        let z = s.ar.konst(16, 0);
                        s.ar.concat(h, z)
                    };
                }
                let r = s.ar.fcmp(*cmp, false, w[0], w[1]);
                let v = match bop {
                    ptxv_ir::BoolOp::And => s.ar.bin(Bin::And, r, qin),
                    ptxv_ir::BoolOp::Or => s.ar.bin(Bin::Or, r, qin),
                    ptxv_ir::BoolOp::Xor => s.ar.bin(Bin::Xor, r, qin),
                };
                out.push((p, v));
            }
            out
        }
        Op::F2fpBf16 { d, a, b } | Op::F2fpF16 { d, a, b, rz: false } => {
            let bf16 = matches!(op, Op::F2fpBf16 { .. });
            let (xa, xb) = (o!(a), o!(b));
            let (x, y) = (lo32(s, xa), lo32(s, xb));
            let hx = s.ar.mk_pub(Node::Narrow { bf16, a: x }, 16);
            let hy = s.ar.mk_pub(Node::Narrow { bf16, a: y }, 16);
            let r = s.ar.concat(hx, hy);
            write(s, *d, r, 1)
        }
        Op::CvtPack {
            d,
            a,
            b,
            c,
            half,
            to,
            rnd,
            sat,
            relu,
        } => {
            let (xa, xb, xc) = (o!(a), o!(b), o!(c));
            let (x, y, z) = (lo32(s, xa), lo32(s, xb), lo32(s, xc));
            let r = s.ar.mk_pub(
                Node::Pack {
                    half: *half,
                    to: *to,
                    rnd: *rnd,
                    sat: *sat,
                    relu: *relu,
                    a: x,
                    b: y,
                    c: z,
                },
                32,
            );
            write(s, *d, r, 1)
        }
        Op::FCvt {
            d,
            a,
            from,
            to,
            rnd,
            ftz,
            int,
            hi,
        } => {
            let xa = o!(a);
            let x =
                s.ar.extract(if *hi { 16 } else { 0 }, from.bits() as u8, xa);
            let r = s.ar.mk_pub(
                Node::FCvt {
                    from: *from,
                    to: *to,
                    rnd: *rnd,
                    ftz: *ftz,
                    int: *int,
                    a: x,
                },
                to.bits() as u8,
            );
            write(s, *d, r, if *to == FFmt::F64 { 2 } else { 1 })
        }
        Op::F2F { d, a, to64, rnd } => {
            let xa = o!(a);
            let x = if *to64 { lo32(s, xa) } else { xa };
            let r = s.ar.mk_pub(
                Node::F2F {
                    to64: *to64,
                    rnd: *rnd,
                    a: x,
                },
                if *to64 { 64 } else { 32 },
            );
            write(s, *d, r, if *to64 { 2 } else { 1 })
        }
        Op::I2F {
            d,
            a,
            signed,
            rnd,
            bits,
            hi,
            f64,
        } => {
            let xa = o!(a);
            let x = if *hi { s.ar.extract(16, 48, xa) } else { xa };
            let b = match *bits {
                64 => 64,
                16 => 16,
                8 => 8,
                _ => 32,
            };
            let x = s.ar.extract(0, b, x);
            let x = if *signed {
                s.ar.sext(64, x)
            } else {
                s.ar.zext(64, x)
            };
            let r = s.ar.mk_pub(
                Node::I2F {
                    signed: *signed,
                    rnd: *rnd,
                    f64: *f64,
                    a: x,
                },
                if *f64 { 64 } else { 32 },
            );
            write(s, *d, r, if *f64 { 2 } else { 1 })
        }
        Op::F2I {
            d,
            a,
            signed,
            bits,
            rnd,
            ftz,
            f64,
            neg,
        } => {
            // (A negated source is lifted only for an f32 source.)
            if *neg && *f64 {
                return vec![];
            }
            let xa = o!(a);
            let x = if *f64 {
                xa
            } else {
                let x = lo32(s, xa);
                fmod(s, x, *neg, false)
            };
            let w = if *bits == 64 { 64 } else { 32 };
            let r = s.ar.mk_pub(
                Node::F2I {
                    signed: *signed,
                    bits: *bits,
                    rnd: *rnd,
                    ftz: *ftz,
                    f64: *f64,
                    a: x,
                },
                w,
            );
            write(s, *d, r, (*bits / 32).max(1))
        }
        Op::Brev { d, a } => {
            let xa = o!(a);
            let x = lo32(s, xa);
            let r = s.ar.un(Un::Brev, x);
            write(s, *d, r, 1)
        }
        _ => vec![],
    }
}

/// How `Inst` nodes are evaluated: by the machine whose run built them (GB300: `run_inst` on
/// a scratch copy of the run's final machine; other targets bring their own machine).
pub trait InstEval {
    /// Destination `out` of instruction `pc` run by thread `t` on `args` (the node's
    /// arguments, in the target's order); None when it has no value there.
    fn run_inst(&mut self, pc: u32, t: u32, out: u8, args: &[u64]) -> Option<u64>;
}

impl InstEval for Machine {
    fn run_inst(&mut self, pc: u32, t: u32, out: u8, args: &[u64]) -> Option<u64> {
        run_inst(self, pc, t as usize, out as usize, args)
    }
}

/// Evaluation of expressions at other leaf values (solve.rs). `Inst` nodes run their
/// instruction on a scratch copy of the run's final machine (`InstEval`).
pub struct Evaluator<'a> {
    pub ar: &'a Arena,
    scratch: Box<dyn InstEval + 'a>,
}

impl<'a> Evaluator<'a> {
    pub fn new(ar: &'a Arena, m: &Machine) -> Self {
        let mut scratch = m.clone();
        scratch.prov = None;
        scratch.num = None;
        Evaluator {
            ar,
            scratch: Box::new(scratch),
        }
    }

    /// An evaluator whose `Inst` nodes run on `scratch` (a target other than GB300).
    pub fn with(ar: &'a Arena, scratch: Box<dyn InstEval + 'a>) -> Self {
        Evaluator { ar, scratch }
    }

    /// The value of `e` with the leaves in `env` replaced (others at their captured values);
    /// None when some node has no value there (division by zero, an unmodelled unit input).
    pub fn eval(&mut self, e: E, env: &HashMap<E, u64>) -> Option<u64> {
        let cone = self.ar.cone(e);
        self.eval_cone(&cone, env)
    }

    /// `cone`: `Arena::cone` of the root (its last element).
    pub fn eval_cone(&mut self, cone: &[E], env: &HashMap<E, u64>) -> Option<u64> {
        let mut val: HashMap<E, u64> = HashMap::with_capacity(cone.len());
        for &x in cone {
            let n = &self.ar.nodes[x as usize];
            let v = match n {
                Node::Input { .. } | Node::Opaque { .. } | Node::Sreg { .. } => {
                    env.get(&x).copied().unwrap_or(self.ar.conc[x as usize])
                        & mask(self.ar.width[x as usize])
                }
                Node::Const { v, .. } => *v,
                Node::Inst {
                    pc,
                    t,
                    out,
                    args,
                    w,
                } => {
                    let a: Vec<u64> = args.iter().map(|k| val[k]).collect();
                    self.scratch.run_inst(*pc, *t, *out, &a)? & mask(*w)
                }
                _ => {
                    let ar = self.ar;
                    eval_node(n, &|k| ar.width[k as usize], &|k| val[&k])?
                }
            };
            val.insert(x, v);
        }
        cone.last().map(|r| val[r])
    }
}

/// Run instruction `pc` for thread `t` of `m` with its source registers set to `args`
/// (`Op::src_regs` order); destination `out`'s value.
pub fn run_inst(m: &mut Machine, pc: u32, t: usize, out: usize, args: &[u64]) -> Option<u64> {
    let k = m.k.clone();
    let inst = &k.code[pc as usize];
    let srcs = inst.op.src_regs();
    for (r, &v) in srcs.iter().zip(args) {
        match r.file {
            File::P | File::UP => m.wp(t, *r, v & 1 == 1),
            _ => m.wr(t, *r, v, 1),
        }
    }
    if !m.exec_value(t, pc) {
        return None;
    }
    let d = *inst.op.dst_regs().get(out)?;
    Some(reg_val(m, t, d))
}

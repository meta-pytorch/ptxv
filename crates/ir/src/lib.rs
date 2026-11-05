/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The one IR that both front-ends (PTX text and SASS from a cubin) lower into.
//!
//! Design rules:
//! * Operations carry PTX-level *meaning* (an mbarrier arrive is `MbarArrive`, whatever its
//!   SASS spelling), so every checker rule is written once.
//! * Register files keep front-end identity: PTX virtual registers hold up to 64 bits in one
//!   slot (`File::V`); SASS registers are 32-bit slots and 64-bit values use consecutive pairs.
//! * Each instruction keeps its source line and text so that every report points at the
//!   input the user gave us (a PTX line or a SASS address).

use serde::Deserialize;
use serde::Serialize;

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize
)]
pub enum File {
    /// PTX virtual register (one slot holds the whole value, up to 64 bits).
    V,
    /// SASS per-thread register (32-bit slot).
    R,
    /// SASS uniform register (32-bit slot, one copy per warp).
    UR,
    /// Per-thread predicate (PTX `.pred` registers map here too).
    P,
    /// SASS uniform predicate (one copy per warp).
    UP,
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize
)]
pub struct Reg {
    pub file: File,
    pub idx: u16,
}

impl Reg {
    pub fn is_uniform(&self) -> bool {
        matches!(self.file, File::UR | File::UP)
    }
}

/// `REDUX` / `CREDUX` reduction (32-bit lanes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReduxOp {
    And,
    Or,
    Xor,
    Sum,
    MinU,
    MaxU,
    MinS,
    MaxS,
}

/// Special (read-only) registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SReg {
    Tid(u8),
    NTid(u8),
    CtaId(u8),
    NCtaId(u8),
    LaneId,
    WarpId,
    /// Rank of this CTA in its cluster (`%cluster_ctarank`, SASS `SR_CgaCtaId`).
    ClusterCtaRank,
    /// Number of CTAs in the cluster (`%cluster_nctarank`, SASS `SR_CgaSize`).
    ClusterNCtaRank,
    /// `%lanemask_lt` (SASS `SR_LTMASK`): the lanes below this one.
    LaneMaskLt,
    /// `%lanemask_le` / `_ge` / `_gt` / `_eq` (SASS `SR_LEMASK` / `GEMASK` / `GTMASK` /
    /// `EQMASK`; GB300: fixtures/s_warpcoll).
    LaneMaskLe,
    LaneMaskGe,
    LaneMaskGt,
    LaneMaskEq,
    /// Address of a `.shared` symbol at this offset: `(cluster rank << 24) | offset`, the
    /// executing CTA's window inside the cluster's shared window (SASS: `LEA R, SR_CgaCtaId,
    /// off, 0x18`). ASSUME(S-DSMEM): see ASSUMPTIONS.md
    SharedAddr(u32),
    /// SASS `SR_SWINHI`: the high word of the generic shared window. A generic address
    /// `(SWINHI << 32) | (rank << 24) | offset` is that shared::cluster address
    /// (`cvta.shared`, generic `mapa`). ASSUME(S-SWIN): see ASSUMPTIONS.md
    SharedWindowHi,
}

/// A value operand.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Val {
    /// Register read. `n` = number of 32-bit slots for R/UR (1 or 2); ignored for V.
    Reg(Reg, u8),
    Imm(u64),
    /// Read from constant bank 0 (`c[0x0][off]`), `bytes` wide.
    Const(u32, u8),
    SReg(SReg),
}

/// A predicate operand. `reg == None` is the constant true predicate (PT / UPT).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Pred {
    pub reg: Option<Reg>,
    pub neg: bool,
}

impl Pred {
    pub const TRUE: Pred = Pred {
        reg: None,
        neg: false,
    };
    pub fn is_always(&self) -> bool {
        self.reg.is_none() && !self.neg
    }
}

/// Address = base + base2 + off (SASS allows `[R2+UR8+0x10]`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Addr {
    pub base: Option<Val>,
    pub base2: Option<Val>,
    pub off: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Space {
    /// `.shared::cta`: the low 24 bits of an address are the offset in this CTA's window.
    Shared,
    /// `.shared::cluster`: bits 24.. select the CTA rank in the cluster.
    SharedCluster,
    Global,
    Param,
    Generic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShflMode {
    Idx,
    Up,
    Down,
    Bfly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntOp {
    Add,
    Sub,
    Mul,
    Shl,
    ShrU,
    ShrS,
    And,
    Or,
    Xor,
    DivU,
    DivS,
    RemU,
    RemS,
    MinU,
    MinS,
    MaxU,
    MaxS,
    /// |a| (signed; `b` is ignored).
    Abs,
    /// High half of the double-width product (`mul.hi`).
    MulHiU,
    MulHiS,
    /// Population count (`popc.b32`, SASS POPC; `b` is ignored).
    Popc,
    /// `bmsk.clamp.b32 d, a, b` (SASS `BMSK` / `UBMSK`): b bits set from bit a; a and b clamp
    /// at 32.
    Bmsk,
    /// `bmsk.wrap.b32` (SASS `BMSK.W`): a and b taken modulo 32.
    BmskWrap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Cmp {
    Eq,
    Ne,
    LtU,
    LeU,
    GtU,
    GeU,
    LtS,
    LeS,
    GtS,
    GeS,
}

/// Floating-point comparison of SASS `FSETP` / PTX `setp.f32`. The ordered forms are false
/// when either operand is NaN; the `U` (unordered) forms are true then. `Num` / `Nan`: both
/// operands are numbers / at least one is NaN.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FCmp {
    F,
    Lt,
    Eq,
    Le,
    Gt,
    Ne,
    Ge,
    Num,
    Nan,
    Ltu,
    Equ,
    Leu,
    Gtu,
    Neu,
    Geu,
    T,
}

/// How a compare combines its result with a predicate operand (`.AND` / `.OR` / `.XOR`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BoolOp {
    And,
    Or,
    Xor,
}

/// How a packed f32x2 operand supplies its two lanes (SASS `FFMA2` operand suffixes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum F2Lanes {
    /// `.F32`: one 32-bit value for both lanes.
    Bcast,
    /// `.F32x2.HI_LO`: lane 0 from the low register, lane 1 from the high one.
    Pair,
    /// `.F32x2.LO_HI`: swapped.
    Swap,
}

/// Function of the SASS multi-function unit (`MUFU.<func>`), or of a PTX `.approx` op.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize
)]
pub enum MufuFunc {
    Ex2,
    Rcp,
    Rsq,
    Lg2,
    /// sin(2 pi x) for SASS (x in revolutions); sin(x) for PTX `sin.approx`.
    Sin,
    Cos,
    Sqrt,
    Tanh,
    /// `MUFU.RCP64H` / `MUFU.RSQ64H`: on the high word of an f64 (sign, exponent, top 20
    /// mantissa bits), the high word of an approximate 1/x, 1/sqrt(x) (low word: 0).
    Rcp64h,
    Rsq64h,
}

/// Operand / result format of a MUFU op: f32 (also the 32-bit high word for the 64H
/// functions), a 16-bit value in the low half (the high half of the result is zero), two
/// packed 16-bit lanes, or a PTX f64.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MufuFmt {
    F32,
    F16,
    Bf16,
    F16x2,
    Bf16x2,
    /// PTX `rcp/rsqrt.approx.ftz.f64`: a 64-bit register, the 64H unit on its high word,
    /// result `unit << 32`.
    F64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Rnd {
    Rn,
    Rz,
    Rm,
    Rp,
}

/// Narrow float formats of `cvt` / SASS `F2FP` and of tensor-core operands: OCP FP8
/// (e4m3, e5m2), FP6 (e2m3, e3m2), FP4 (e2m1), the unsigned scale formats ue8m0 / ue4m3, and
/// f16 / bf16 (the packed `cvt.rs` forms). Arithmetic: `ptxv_sim::lowp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Lp {
    E4m3,
    E5m2,
    E2m3,
    E3m2,
    E2m1,
    Ue8m0,
    Ue4m3,
    F16,
    Bf16,
}

/// Rounding of a narrow conversion: `.rn`, `.rz`, `.rp` (ue8m0 only) or `.rs` (stochastic,
/// random bits from an operand).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LpRnd {
    Rn,
    Rz,
    Rp,
    Rs,
}

/// `tcgen05.mma` kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MmaKind {
    F16,
    F8f6f4,
    /// `.kind::mxf8f6f4.block_scale`: ue8m0 scales, one per 32 K elements.
    Mxf8f6f4,
    /// `.kind::mxf4.block_scale`: packed e2m1, ue8m0 scales per 32 (`scale_vec::2X`).
    Mxf4,
    /// `.kind::mxf4nvf4.block_scale`: packed e2m1, scales per 32 (`scale_vec::2X`) or per 16
    /// (`x4`: `scale_vec::4X`), ue8m0 or ue4m3 per the instruction descriptor.
    Mxf4nvf4 {
        x4: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AtomOp {
    Or,
    And,
    Xor,
    Add,
    Exch,
    /// f32 add, round to nearest, subnormals flushed (`ATOM.E.ADD.F32.FTZ.RN`,
    /// `atom.add.f32`).
    FAddFtz,
    /// Compare-and-swap, SASS `.CAST.SPIN`: `v` is the register pair {compare, new}; the
    /// result (`d` = 1 / 0, `p`) says whether the swap happened.
    CasSpin,
    /// Compare-and-swap returning the old value (SASS `ATOM.E.CAS` / `ATOMG.E.CAS`, PTX
    /// `atom.cas`): `v` is the register pair {compare, new}. A failed compare only reads.
    Cas,
    /// 32-bit integer minimum / maximum, signed (`.S32`) or unsigned (SASS `ATOMS.MIN`,
    /// `ATOMS.MAX.S32`: CUTLASS's float max through integer atomics).
    MinU,
    MaxU,
    MinS,
    MaxS,
}

/// PTX memory-model scope of a strong operation or fence.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize
)]
pub enum Scope {
    Cta,
    Cluster,
    Gpu,
    Sys,
}

/// Memory-ordering semantics of a strong operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sem {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
}

/// A strong (non-plain) memory access: `ld/st.relaxed|acquire|release.<scope>`,
/// `ld/st.volatile` (= relaxed.sys), `atom` / `red` (default relaxed.gpu). Strong accesses that
/// are morally strong (each one's scope includes the other's thread) do not race, and a
/// release observed by an acquire synchronizes (S-SCOPE).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemSync {
    pub sem: Sem,
    pub scope: Scope,
}

impl MemSync {
    pub fn acquires(&self) -> bool {
        matches!(self.sem, Sem::Acquire | Sem::AcqRel)
    }
    pub fn releases(&self) -> bool {
        matches!(self.sem, Sem::Release | Sem::AcqRel)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FenceKind {
    /// `fence.mbarrier_init.release.cluster`
    MbarInit,
    /// A full memory fence with release and acquire semantics at a scope (PTX `fence.acq_rel`
    /// / `fence.sc`). What a later relaxed cluster arrive releases.
    Membar(Scope),
    /// Release side only (SASS `MEMBAR.*.<scope>`; ptxas emits it before release stores).
    Release(Scope),
    /// Acquire side only (SASS `CCTL.IVALL`, which ptxas emits after acquire loads and after
    /// `MEMBAR` for `fence.acq_rel`): earlier strong loads' observations take effect.
    Acquire,
    /// `fence.proxy.async[.shared::cta|.global]`, SASS `FENCE.VIEW.ASYNC.*`
    ProxyAsync,
    /// `tcgen05.fence::before_thread_sync`: prior tcgen05 ops are released by the next sync.
    TcBefore,
    /// `tcgen05.fence::after_thread_sync`: later tcgen05 ops are ordered after prior syncs.
    TcAfter,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Op {
    Mov {
        d: Reg,
        a: Val,
    },
    Int {
        op: IntOp,
        bits: u8,
        d: Reg,
        a: Val,
        b: Val,
    },
    /// 32x32 -> 64 (`mul.wide`).
    MulWide {
        d: Reg,
        a: Val,
        b: Val,
        signed: bool,
    },
    /// `d = a*b + c (+ carry-in)`; `wide`: 32x32+64 -> 64 (SASS IMAD.WIDE, `d`/`c` are pairs).
    /// `hi` (IMAD.HI): `d` = bits 32..63 of `a*b + c`, `c` a 64-bit pair.
    Imad {
        d: Reg,
        a: Val,
        b: Val,
        c: Val,
        /// `c` is negated (32-bit forms only).
        neg_c: bool,
        /// `c` is bitwise complemented (`~R`, 32-bit forms only).
        not_c: bool,
        wide: bool,
        hi: bool,
        /// `a`, `b` are signed (IMAD.WIDE / IMAD.HI without .U32).
        signed: bool,
        cin: Option<Pred>,
        /// Carry out of bit 64 of `a*b + c` (IMAD.WIDE.U32 / IMAD.HI.U32 `Rd, Pc, ...`).
        cout: Option<Reg>,
    },
    /// `d = a + b + c (+ carry-ins)`, each input optionally negated. `wide`: 64-bit pairs
    /// (UIADD3.64). `cout`: carry-out predicate. `cin`: carry-in predicates (IADD3.X).
    Iadd3 {
        d: Reg,
        a: Val,
        b: Val,
        c: Val,
        neg: [bool; 3],
        /// Input bitwise complemented (`~R`).
        not: [bool; 3],
        wide: bool,
        /// Carry-out predicates; with two, the carry (0..2) is unary: `cout[0]` = carry >= 1,
        /// `cout[1]` = carry >= 2 (so that `.X` adding both as carry-ins adds the carry).
        cout: Vec<Reg>,
        cin: Vec<Pred>,
    },
    /// Integer conversion (`cvt.u64.u32` etc.).
    CvtInt {
        d: Reg,
        a: Val,
        from_bits: u8,
        to_bits: u8,
        signed: bool,
    },
    /// Per-thread register -> uniform register; the value must be uniform over active lanes.
    R2Ur {
        d: Reg,
        a: Val,
        /// `R2UR P, URd, Ra` (ptxas's uniformization loop): URd = the first active lane's
        /// value, and per lane P = (its value differs), i.e. "not covered yet".
        p: Option<Reg>,
    },
    /// Warp-collective ballot: `d` (UR/R) = mask of active lanes whose `a` holds.
    VoteAny {
        d: Reg,
        a: Pred,
    },
    /// Find leading one: bit index of the most significant set bit, or 0xffffffff. `sh`
    /// (FLO.SH): 31 - that index (the left shift that normalizes `a`), 0xffffffff for 0.
    Flo {
        d: Reg,
        a: Val,
        sh: bool,
    },
    /// Bit reverse (BREV, `brev.b32`).
    Brev {
        d: Reg,
        a: Val,
    },
    /// SASS `VIADDMNMX d, a, b, c, PT`: d = min(a + b, c) (`max`: !PT), 32-bit.
    AddMinMax {
        d: Reg,
        a: Val,
        b: Val,
        c: Val,
        signed: bool,
        max: bool,
        /// `-Rb` (DeepEP: `VIADDMNMX R5, R23, -R45, 0x20, PT`).
        neg_b: bool,
    },
    /// SASS `VIMNMX3 d, a, b, c, sel`: the minimum (sel PT) or maximum (!PT) of three 32-bit
    /// integers, signed unless `.U32`.
    IntMnmx3 {
        d: Reg,
        a: Val,
        b: Val,
        c: Val,
        signed: bool,
        max: bool,
    },
    /// SASS `P2R d, PR, a, mask`: d = (a & !mask) | (bit i = Pi, for i in mask).
    P2r {
        d: Reg,
        a: Val,
        mask: u32,
    },
    /// Load from constant bank `bank` at `base + off` (SASS `LDC R, c[0x2][R6+0xc]`). Banks
    /// other than 0 are the cubin's `.nv.constant<bank>.<kernel>` section, copied into `data`
    /// (ptxas jump tables: function-relative code addresses).
    LdConst {
        d: Reg,
        bank: u8,
        base: Val,
        off: u32,
        bytes: u8,
        data: Vec<u8>,
    },
    /// Warp-collective reduction of `a` over active lanes into uniform `d` (SASS `REDUX`
    /// = AND, `.OR`, `.XOR`, `.SUM`; `CREDUX.MIN/.MAX[.S32]`).
    Redux {
        d: Reg,
        a: Val,
        op: ReduxOp,
    },
    /// Atomic read-modify-write on memory; `d` gets the old value.
    Atom {
        op: AtomOp,
        space: Space,
        bytes: u8,
        addr: Addr,
        v: Val,
        d: Option<Reg>,
        sync: MemSync,
        /// Predicate result: a generic `ATOM` sets it when the address was global (the
        /// operation was performed; ptxas then skips its shared / local fallbacks), `.CAST.SPIN`
        /// when the swap happened.
        p: Option<Reg>,
        /// Compare-and-swap (`Cas`, `CasSpin`): the new value, with `v` the compare value.
        /// None: the new value is the register (pair, for 8 bytes) right after `v` (SASS
        /// `ATOM.E.CAS PT, Rd, [Ra], Rb, Rb+1`).
        new: Option<Val>,
    },
    /// SASS `QSPC.E.{S,D} Pd, RZ, [Ra]`: does the generic address lie in the shared::cta (S) /
    /// shared::cluster (D) window? (ptxas's generic-atomic fallbacks.)
    Qspc {
        p: Reg,
        addr: Addr,
    },
    /// `d = (a << shift) + b`; `cout`: carry out of the 32-bit add (`LEA Rd, Pc, ...`).
    /// `neg_a` (`LEA Rd, Pc, -Ra, Rb, s`): `d = b + ~(a << s) + 1`, the low half of a 64-bit
    /// subtraction (the carry feeds `IADD3.X ..., ~hi, ..., Pc`).
    Lea {
        d: Reg,
        a: Val,
        b: Val,
        shift: u8,
        cout: Option<Reg>,
        neg_a: bool,
    },
    /// Funnel shift of the 64-bit value `hi:lo`; result is the high or low 32 bits.
    Shf {
        d: Reg,
        lo: Val,
        sh: Val,
        hi: Val,
        right: bool,
        signed: bool,
        high: bool,
    },
    /// Three-input bitwise LUT (a = 0xF0, b = 0xCC, c = 0xAA convention).
    Lop3 {
        d: Reg,
        a: Val,
        b: Val,
        c: Val,
        lut: u8,
        /// `LOP3.LUT Pp, Rd, ...`: also P = (result != 0).
        p: Option<Reg>,
        /// The predicate operand `Pq` and its mode: P = (result != 0) || Pq (the default), or
        /// && Pq (`.PAND`). None: P = (result != 0) (`!PT` in the default mode).
        /// ASSUME(L-LOP3-P): see ASSUMPTIONS.md
        pq: Option<(Pred, bool)>,
    },
    /// Three-input predicate LUT, same convention.
    Plop3 {
        d: Reg,
        a: Pred,
        b: Pred,
        c: Pred,
        lut: u8,
    },
    Setp {
        d: Reg,
        cmp: Cmp,
        bits: u8,
        a: Val,
        b: Val,
        /// Combined with the comparison: `d = cmp && and`, or `cmp || and` when `or`.
        and: Pred,
        or: bool,
        /// `.XOR` combine: `d = cmp ^ and` (DeepEP: `UISETP.EQ.XOR.EX`).
        xor: bool,
        /// ISETP.EX: `a`, `b` are the high words of a 64-bit compare whose low-word result
        /// (same comparison, unsigned) is this predicate: it decides when the high words are
        /// equal.
        ex: Option<Pred>,
    },
    /// Warp-collective: `d = all active threads have a` (VOTE.ALL / VOTEU.ALL). `uni`: `d =
    /// all active threads agree on a` instead (VOTE.EQ, `vote.sync.uni.pred`).
    VoteAll {
        d: Reg,
        a: Pred,
        uni: bool,
    },
    /// Warp-collective: exactly one active thread (the lowest lane) gets `d = true`.
    Elect {
        d: Reg,
        lane: Option<Reg>,
    },
    /// f32 add. Each input's sign is first cleared (`abs`, `|R|`) then flipped (`neg`, `-R`).
    /// `ftz` (`.FTZ`): subnormal inputs and results are flushed to zero of the same sign;
    /// `sat` (`.SAT`): the result is clamped to [+0, 1], NaN to +0.
    Fadd {
        d: Reg,
        a: Val,
        b: Val,
        rnd: Rnd,
        neg: [bool; 2],
        abs: [bool; 2],
        ftz: bool,
        sat: bool,
    },
    /// f32 FMA `d = a*b + c` with one rounding (FFMA; FMUL: `c` = None, the rounded
    /// product). Each input's sign is first cleared (`abs`, `|R|`) then flipped (`neg`, `-R`).
    /// `ftz` as for Fadd; `sat` (`.SAT`): the result is clamped to [+0, 1], NaN to +0.
    Ffma {
        d: Reg,
        a: Val,
        b: Val,
        c: Option<Val>,
        neg: [bool; 3],
        abs: [bool; 3],
        rnd: Rnd,
        ftz: bool,
        sat: bool,
    },
    /// SASS `FMNMX d, a, b, p` (`min.f32` / `max.f32`): d = p ? min(a, b) : max(a, b); a NaN
    /// operand is ignored unless both are NaN. `c` (FMNMX3): the minimum / maximum of three.
    /// `nan` (`.NAN`, `min.NaN.f32`): a NaN operand makes the result NaN. Operand modifiers
    /// as for Ffma.
    Fmnmx {
        d: Reg,
        a: Val,
        b: Val,
        c: Option<Val>,
        neg: [bool; 3],
        abs: [bool; 3],
        min: Pred,
        ftz: bool,
        nan: bool,
    },
    /// SASS `FSETP.<cmp>.<bop> Pd, PT, a, b, Pc`: Pd = (a cmp b) bop Pc. Operand modifiers as
    /// for Fadd; `ftz`: subnormal operands compare as zero.
    Fsetp {
        d: Reg,
        cmp: FCmp,
        a: Val,
        b: Val,
        neg: [bool; 2],
        abs: [bool; 2],
        c: Pred,
        bop: BoolOp,
        ftz: bool,
    },
    /// SASS `HSETP2.BF16_V2.<cmp>.<bop> Pd, PT, a.Hx_Hx, b.Hy_Hy, Pc` (`setp.<cmp>.bf16`): one
    /// bf16 lane of each operand (`hi`: the high half), compared as the exact f32 widening;
    /// Pd = (a cmp b) bop Pc. An immediate pair is `b` with `hi[1] = false`.
    Hsetp2Bf16 {
        d: Reg,
        cmp: FCmp,
        a: Val,
        b: Val,
        hi: [bool; 2],
        c: Pred,
        bop: BoolOp,
    },
    /// SASS `FSET.BF.cmp.bop d, a, b, c`: Fsetp's predicate as a register, 1.0f when true
    /// (`.BF`), else 0.
    Fset {
        d: Reg,
        cmp: FCmp,
        a: Val,
        b: Val,
        neg: [bool; 2],
        abs: [bool; 2],
        c: Pred,
        bop: BoolOp,
        ftz: bool,
    },
    /// SASS `FSEL d, a, b, p` with operand modifiers: d = p ? a : b after `|x|` / `-x` (sign
    /// bit only; FSEL without modifiers is `Sel`).
    Fsel {
        d: Reg,
        a: Val,
        b: Val,
        p: Pred,
        neg: [bool; 2],
        abs: [bool; 2],
    },
    /// SASS `FCHK Pd, a, b`: Pd = the quotient a / b needs the slow path of ptxas's IEEE
    /// division sequence (see `fp::fchk`).
    Fchk {
        d: Reg,
        a: Val,
        b: Val,
    },
    /// SASS `R2P PR, a, mask`: predicate Pi = bit i of a, for i in mask (inverse of P2R).
    R2p {
        a: Val,
        mask: u32,
    },
    /// SASS `F2FP.BF16.F32.PACK_AB d, a, b`: d = bf16(a) << 16 | bf16(b), round to nearest even.
    F2fpBf16 {
        d: Reg,
        a: Val,
        b: Val,
    },
    /// Packed f32x2 FMA, round to nearest (`fma.rn.f32x2`, SASS FFMA2): `d+i = a_i*b_i + c_i`
    /// for lanes i = 0, 1. `neg`: operand a / b / c negated (sign bit flipped per lane,
    /// `-R16.F32`).
    Ffma2 {
        d: Reg,
        a: (Val, F2Lanes),
        b: (Val, F2Lanes),
        c: (Val, F2Lanes),
        neg: [bool; 3],
    },
    /// Packed half2 FMA `d = a*b + c` (per 16-bit half).
    Hfma2 {
        d: Reg,
        a: Val,
        b: Val,
        c: Val,
        neg_a: bool,
    },
    /// Two lanes into a narrow format: SASS `F2FP[.SATFINITE][.RELU].<to>.F32.PACK_AB[_MERGE_C]
    /// [.RZ|.RP|.RS] Rd, Ra, Rb[, Rc]` and the PTX `cvt.<rnd>[.satfinite][.relu].<to>x2.f32 d, a,
    /// b` forms (a -> high lane). `half`: the source is one packed f16x2 / bf16x2 register `a`
    /// instead (`F2FP.<to>.<F16|BF16>.UNPACK_B_MERGE_C Rd, Ra, Rc`, `cvt...x2.f16x2`). Lanes
    /// are 8 bits (4 for e2m1), 16 for f16 / bf16; d = c << (2 lanes) | a' << lane | b'. With
    /// `.RS`, c also supplies the random bits (see `lowp::rs_bits`). N-CVT-LP.
    CvtPack {
        d: Reg,
        a: Val,
        b: Val,
        c: Val,
        half: Option<Lp>,
        to: Lp,
        rnd: LpRnd,
        sat: bool,
        relu: bool,
    },
    /// PTX `cvt.rs.satfinite[.relu].<to>x4.f32 d, {a, b, e, f}, rbits`: four lanes, a in the
    /// most significant (ptxas emits two `F2FP...PACK_AB_MERGE_C.RS`; N-CVT-LP).
    CvtRsX4 {
        d: Reg,
        src: [Val; 4],
        rbits: Val,
        to: Lp,
        relu: bool,
    },
    /// Two narrow codes in the low bits of `a` (`hi`: in its high half) -> f16x2 / bf16x2:
    /// SASS `F2FP[.RELU].<F16|BF16>.<from>.UNPACK_B Rd, Ra[.H1]`, PTX `cvt.rn[.relu].f16x2.<from>x2`
    /// / `cvt.rn.bf16x2.ue8m0x2`. Exact; N-CVT-LP.
    CvtUnpack {
        d: Reg,
        a: Val,
        hi: bool,
        from: Lp,
        to: Lp,
        relu: bool,
    },
    /// bf16 -> f32 (exact).
    CvtF32Bf16 {
        d: Reg,
        a: Val,
    },
    /// `cvta[.to].space`: generic <-> space address conversion.
    Cvta {
        d: Reg,
        a: Val,
        space: Space,
        to_generic: bool,
    },
    Ld {
        d: Reg,
        n: u8,
        space: Space,
        bytes: u8,
        addr: Addr,
        /// None: a plain (weak) load.
        sync: Option<MemSync>,
        /// Sign-extend a 1- or 2-byte load (`LDG.E.S8`, `ld.global.s8`).
        sext: bool,
    },
    St {
        space: Space,
        bytes: u8,
        addr: Addr,
        v: Val,
        sync: Option<MemSync>,
    },
    /// Branch. `any`: the whole active split branches if any active thread's guard holds.
    Bra {
        target: usize,
        any: bool,
        /// SASS `BRA.U.NOT_TID0`: a warp-uniform branch taken by every warp except the one
        /// holding thread 0 of the CTA, which falls through (GB300: probes/not_tid0; cuBLAS
        /// guards its once-per-CTA TMEM free with it).
        tid0_falls: bool,
        /// SASS `@P0 BRA P1, target`: a second, per-thread condition; taken where the guard
        /// and `cond` both hold.
        cond: Option<Pred>,
        /// SASS `BRA.DIV URm, target` (ptxas's collective lowering of a run-time member
        /// mask): warp-collective; every lane of the executing split branches unless the guard
        /// holds on all of them and their lane mask equals URm (GB300: fixtures/s_warpcoll).
        div: Option<Val>,
    },
    /// SASS `BRX Ra -off` / `BRXU URa -off` (ptxas jump tables): jump to the code address
    /// `Ra + next + off` within this kernel (16 bytes per instruction), per thread. `targets`:
    /// the instructions nvdisasm lists as `BRANCH_TARGETS`; any other address is an error.
    Brx {
        base: Val,
        off: i64,
        targets: Vec<usize>,
    },
    Exit,
    Nop,
    /// Call a local function (return address is managed by the code, SASS CALL.REL.NOINC).
    Call {
        target: usize,
    },
    /// Call the function whose address `target` holds (SASS `CALL.ABS.NOINC`). `syms`: the
    /// (synthetic address, name) of each function / object a constant bank refers to; only
    /// `__assertfail` is modeled, as a trap.
    CallAbs {
        target: Val,
        syms: Vec<(u64, String)>,
    },
    /// Return to the code address held in `addr` (SASS RET.REL.NODEC).
    Ret {
        addr: Val,
    },
    /// Abort the kernel (SASS BPT.TRAP). `why` = enclosing function label (ptxas guardrails).
    Trap {
        why: String,
    },
    /// `bar.warp.sync` / WARPSYNC: the lanes in `mask` rendezvous.
    WarpSync {
        mask: Val,
        /// SASS `WARPSYNC.COLLECTIVE Rm, end` (the slow path of a run-time member mask): lanes
        /// naming different masks run the region up to `ENDCOLLECTIVE` one mask group at a
        /// time (GB300: fixtures/s_warpcoll).
        collective: bool,
    },
    /// SASS `ENDCOLLECTIVE`: closes the region opened by the `WARPSYNC.COLLECTIVE` at `start`;
    /// the next waiting mask group there may enter.
    EndCollective {
        start: usize,
    },
    /// SASS `BSSY Bb, target`: convergence barrier `b` records the lanes executing it.
    Bssy {
        b: u8,
    },
    /// SASS `BSYNC Bb`: the lanes `BSSY Bb` recorded (not exited) rendezvous, like
    /// `bar.warp.sync` with that mask (`__syncwarp` after a divergent region compiles to it).
    Bsync {
        b: u8,
    },
    /// SASS `BREAK Bb`: the executing lanes leave convergence barrier `b` (a loop exited early
    /// skips its `BSYNC`); lanes parked at that BSYNC no longer wait for them.
    Break {
        b: u8,
    },
    /// `bar.sync id[, count]`; no count = all threads of the CTA. `arrive`: `bar.arrive id,
    /// count` (SASS `BAR.ARV`): the thread counts toward (and releases into) the barrier's
    /// current generation without waiting; `count` is then always present.
    BarSync {
        id: Val,
        count: Option<Val>,
        arrive: bool,
    },
    /// `mbarrier.init`. `encoded`: `count` is the raw 64-bit state word (SASS SYNCS.EXCH.64).
    MbarInit {
        addr: Addr,
        count: Val,
        encoded: bool,
    },
    /// `mbarrier.arrive[.expect_tx]` (one arrival; `tx` = bytes added to the tx-count first).
    MbarArrive {
        addr: Addr,
        tx: Option<Val>,
        /// `.shared::cluster` address (SASS `SYNCS.ARRIVE.*.RED`): the rank byte selects the
        /// CTA whose mbarrier receives the arrival.
        remote: bool,
        /// PTX `mbarrier.arrive [a], count` (None: one arrival); 1 and 0 (no arrival: SASS
        /// `.A0TR` = `mbarrier.expect_tx`, `.A0TX` = `mbarrier.complete_tx`) are modeled.
        count: Option<Val>,
        /// `tx` completes transaction bytes instead of expecting them (`mbarrier.complete_tx`).
        complete: bool,
    },
    /// `clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128
    /// [resp], [mbar]` (SASS `UGETNEXTWORKID.SELFCAST [resp], [mbar]`): ask to cancel a CTA that
    /// has not launched yet; the 16-byte response lands in `resp` and completes 16 transaction
    /// bytes on `mbar` (E-CLC).
    ClcTryCancel {
        resp: Val,
        mbar: Val,
        /// `.multicast::cluster::all` (SASS `.BROADCAST`): the response goes to every CTA of
        /// the cluster. Parsed but not simulated: executing it is an error.
        multicast: bool,
    },
    /// `clusterlaunchcontrol.query_cancel`: `is_canceled` into a predicate (`ctaid` = None), or
    /// `get_first_ctaid::x|y` (dimension) into a register, from a .b128 response held in two
    /// 64-bit registers (`q`, `q+1`).
    ClcQuery {
        d: Reg,
        q: Reg,
        ctaid: Option<u8>,
    },
    /// `mapa.shared::cluster.u32 d, a, rank`: the same offset in CTA `rank`'s window.
    Mapa {
        d: Reg,
        a: Val,
        rank: Val,
    },
    /// `stmatrix.sync.aligned.m8n8.xN[.trans].shared.b16 [addr], {v0..vN-1}` (warp-collective):
    /// matrix i's row j goes to the address held by lane 8i + j; its 8 b16 elements come from
    /// lanes 4j..4j+3, register v_i (two elements each, low half first). `.trans` stores the
    /// transpose.
    StMatrix {
        addr: Addr,
        v: Vec<Val>,
        trans: bool,
    },
    /// `shfl.sync.idx.b32 d, a, lane, clamp, 0xffffffff` (warp-collective): d = a of lane
    /// `lane` within the lane's segment (`clamp` = 0x1f: the whole warp).
    /// Other modes (`mode`, SASS SHFL.UP/DOWN/BFLY): the source lane is lane - b, lane + b,
    /// lane ^ b, within the segment and clamp of `clamp` (PTX `c`); out of range: own `a`.
    Shfl {
        d: Reg,
        a: Val,
        lane: Val,
        clamp: Val,
        mode: ShflMode,
        /// Predicate result (`shfl.sync ... d|p`, SASS `SHFL.UP P0, ...`): the source lane was
        /// in range.
        p: Option<Reg>,
    },
    /// `bfe d, a, pos, len`: bit-field extract (sign-extended if `signed`).
    Bfe {
        d: Reg,
        a: Val,
        pos: Val,
        len: Val,
        signed: bool,
    },
    /// `tcgen05.st.sync.aligned.32x32b.xN [taddr + off], {v..}`: TMEM lane (taddr lane +
    /// this lane), columns col..col+N-1. Asynchronous until `tcgen05.wait::st`.
    TmemSt {
        taddr: Val,
        off: u32,
        v: Vec<Val>,
        /// PTX: an asynchronous tcgen05 op (tcgen05 actor). SASS `STTM`: ordered like an
        /// ordinary store (as LDTM; scoreboards).
        tracked: bool,
        /// SASS `STTM.16dp32bit_t0_t15` / `_t16_t31`: as `TmemLd::half`.
        half: Option<u8>,
    },
    /// SASS `LEA.HI d, a, b, c, s`: d = b + high word of ({c:a} << s). `sx32`
    /// (`LEA.HI.SX32 d, a, b, s`): the upper word is a's sign instead of c.
    LeaHi {
        d: Reg,
        a: Val,
        b: Val,
        c: Val,
        shift: u8,
        sx32: bool,
        /// `-Rb`: b enters as its two's complement.
        neg_b: bool,
        /// `LEA.HI.X`: carry-in added to the result.
        cin: Option<Pred>,
        /// `LEA.HI d, P, a, b, c, s`: carry-out of the 32-bit add.
        cout: Option<Reg>,
    },
    /// Integer -> f32 (`I2F[.RP|.RM|.RZ]`, `cvt.rn.f32.s32`).
    I2F {
        d: Reg,
        a: Val,
        signed: bool,
        rnd: Rnd,
        /// Source width: 32, or 64 (`I2F.U64`, a register pair), or 16 / 8 (`I2F.S16`: the
        /// low bits, or the high half with `hi`: `I2F.U16 Rd, Ra.H1`).
        bits: u8,
        hi: bool,
    },
    /// f32 -> integer, truncating (`F2I.TRUNC`, `cvt.rzi.u32.f32`); out of range saturates.
    F2I {
        d: Reg,
        a: Val,
        signed: bool,
        /// Destination width: 32, or 64 (`F2I.U64.TRUNC`, a register pair).
        bits: u8,
    },
    /// PTX `mov.b64 %d, {lo, hi}`: d = hi << 32 | lo.
    Pack64 {
        d: Reg,
        lo: Val,
        hi: Val,
    },
    /// PTX `mov.b64 {lo, hi}, %a` (`_` discards a half).
    Unpack64 {
        lo: Option<Reg>,
        hi: Option<Reg>,
        a: Val,
    },
    /// SASS `MUFU.<func>[.F16|.BF16|.F16x2|.BF16x2]`, or a PTX `<func>.approx` op (N-MUFU).
    Mufu {
        d: Reg,
        a: Val,
        func: MufuFunc,
        fmt: MufuFmt,
        /// Scalar 16-bit formats: the operand is the high half of `a` (SASS `R.H1`).
        hi: bool,
        /// f32 operand modifiers `-R` / `|R|` (applied as `-|a|` when both).
        neg: bool,
        abs: bool,
        /// A PTX op: `Some(ftz)`. Its semantics include ptxas's expansion around the unit
        /// (denormal scaling, the 1/(2 pi) range reduction of sin/cos). `None`: a bare MUFU.
        ptx: Option<bool>,
    },
    /// `tcgen05.wait::st`
    TmemWaitSt,
    /// `fence.proxy.tensormap::generic.acquire` on the 128-byte tensor map at `addr` (SASS
    /// `UTMACCTL.IV [URa]`): later TMA ops of this thread see its current contents.
    TmapFence {
        addr: Addr,
    },
    /// `mbarrier.inval`: the mbarrier object ends; its 8 bytes are plain memory again.
    MbarInval {
        addr: Addr,
    },
    /// `griddepcontrol.wait` (SASS `ACQBULK`): block until every prerequisite grid has
    /// completed, then see all of its memory operations (E-PDL).
    GridDepWait,
    /// `griddepcontrol.launch_dependents` (SASS `PREEXIT`): this CTA allows the dependent grid
    /// to launch. No memory ordering (E-PDL).
    GridDepLaunch,
    /// `d = p ? a : b` (SASS `SEL` / `USEL`).
    Sel {
        d: Reg,
        a: Val,
        b: Val,
        p: Pred,
    },
    /// SASS `PRMT d, a, sel, b`: byte i of d = byte `sel[4i..4i+3] & 7` of {a, b} (b = bytes
    /// 4..7); bit 3 of the selector nibble replicates that byte's sign bit.
    Prmt {
        d: Reg,
        a: Val,
        sel: Val,
        b: Val,
    },
    /// `barrier.cluster.arrive[.release|.relaxed]` (SASS `UCGABAR_ARV`, relaxed). A relaxed
    /// arrive releases only what the thread's last fence released (E-CLUSTER-BAR).
    ClusterArrive {
        release: bool,
    },
    /// `barrier.cluster.wait[.acquire]` (SASS `UCGABAR_WAIT`).
    ClusterWait,
    /// `mbarrier.try_wait.parity`. `bit31`: parity is bit 31 of the operand (SASS encoding).
    MbarTryWait {
        d: Reg,
        addr: Addr,
        parity: Val,
        bit31: bool,
    },
    Fence {
        kind: FenceKind,
    },
    /// `cp.async.bulk.tensor.Nd.shared::cluster.global.mbarrier::complete_tx::bytes`
    TmaLoad {
        dims: u8,
        dst: Val,
        /// Byte offset added to `dst` (`[%r15+33792]`).
        off: u32,
        mbar: Val,
        /// Byte offset added to `mbar`.
        mbar_off: u32,
        tmap: Val,
        /// `dims` coordinates; `tile::gather4` (dims = 2): 5, the column then four rows.
        coords: Vec<Val>,
        /// `.multicast::cluster` ctaMask: the box lands at the same offsets in every CTA rank
        /// in the mask, each completing the transaction on its own mbarrier at `mbar`'s offset.
        mask: Option<Val>,
        /// `.cta_group::2` (SASS `.2CTA`): the mbarrier may be in the peer CTA of the pair.
        pair: bool,
    },
    /// cp.async.bulk.tensor (shared::cta -> global), bulk_group completion.
    TmaStore {
        dims: u8,
        src: Val,
        /// Byte offset added to `src`.
        off: u32,
        tmap: Val,
        coords: Vec<Val>,
    },
    /// cp.async.bulk.commit_group: close the issuing thread's open bulk group.
    BulkCommit,
    /// cp.async.bulk.wait_group[.read] n: block until at most n committed groups are pending.
    BulkWait {
        n: u32,
        read: bool,
    },
    /// `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [dst], ncols` (warp-wide).
    TmemAlloc {
        dst: Addr,
        ncols: Val,
    },
    /// SASS `UTCATOMSWS.FIND_AND_SET.ALIGN UPok, URd, URn`: allocate `units` 32-column units;
    /// blocks while the allocation cannot succeed (ptxas spins on failure).
    TmemAllocRaw {
        d: Reg,
        ok: Reg,
        units: Val,
        /// `UTCATOMSWS.2CTA`: the even CTA allocates the same columns in both CTAs of its pair.
        pair: bool,
    },
    /// `st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [addr], v, [mbar]` (SASS
    /// `STAS [Ra.64], Rv`: Ra = address, Ra+1 = mbarrier): store 4 bytes into (another CTA's)
    /// shared memory and complete 4 transaction bytes on the mbarrier there.
    StAsync {
        addr: Val,
        mbar: Val,
        v: Val,
    },
    /// SASS `UTCATOMSWS.AND URZ, URmask`: allocator bitmap &= mask.
    TmemFreeRaw {
        mask: Val,
    },
    /// `tcgen05.relinquish_alloc_permit` (SASS `UVIRTCOUNT.DEALLOC.SMPOOL`).
    TmemRelinquish,
    /// `tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, ncols` (warp-wide).
    TmemDealloc {
        taddr: Val,
        ncols: Val,
    },
    /// `tcgen05.mma.cta_group::1.kind::f16 [d], adesc, bdesc, idesc, enable_input_d`.
    Mma {
        d: Val,
        adesc: Val,
        bdesc: Val,
        idesc: Val,
        enable_d: Pred,
        /// `cta_group::2` (SASS `UTCHMMA.2CTA`): issued by the even CTA of a pair; M is split by
        /// rows (A, D) and N in halves (B) over the two CTAs, at the same addresses in each.
        pair: bool,
        /// PTX `disable-output-lane` mask registers (empty: none); only all-zero is modeled.
        disable: Vec<Val>,
        /// A in TMEM (`tcgen05.mma ... [d], [a], bdesc, ...`, SASS `UTCHMMA tmem[URa], ...`):
        /// `adesc` is then A's TMEM address, laid out like D (row i at D's lane for row i),
        /// two 16-bit elements per 32-bit column, the even K index in the low half.
        a_tmem: bool,
        /// `.kind::` (SASS: UTCHMMA = f16, UTCQMMA = f8f6f4, UTCOMMA = the block-scaled kinds).
        kind: MmaKind,
        /// Block-scaled kinds: the TMEM addresses of the A and B scale factors
        /// (`[scale_a], [scale_b]`).
        scale: Option<(Val, Val)>,
    },
    /// `tcgen05.cp.cta_group::1.32x128b.warpx4 [taddr], sdesc` (SASS
    /// `UTCCP.T.S.4x32dp128bit`): copies a 32-row x 16-byte shared-memory matrix (K-major
    /// descriptor) into TMEM columns taddr..taddr+3, row r to lane 32w + r of every
    /// sub-partition w. Asynchronous, in the issuing thread's tcgen05 pipe (ordered with its
    /// MMAs and commits). `pair` (`cta_group::2`, SASS `.2CTA`): issued by the even CTA of the
    /// pair, each CTA's shared memory to its own TMEM at the same addresses (E-PAIR).
    TcCp {
        taddr: Val,
        /// Added to `taddr` (SASS `tmem[URa+off]`).
        off: u32,
        sdesc: Val,
        pair: bool,
    },
    /// `tcgen05.commit...mbarrier::arrive::one` (SASS UTCBAR): arrive when all prior
    /// tcgen05 async ops of this thread complete.
    TcCommit {
        mbar: Val,
        /// Byte offset added to `mbar`.
        off: u32,
        /// `.multicast::cluster` ctaMask: arrive on the mbarrier at the same offset in every
        /// CTA rank in the mask.
        mask: Option<Val>,
    },
    /// `tcgen05.ld.32x32b.xN` (SASS LDTM.xN): N consecutive columns from this warp's lanes.
    /// `tracked`: results are undefined until `tcgen05.wait::ld` (PTX semantics); SASS relies
    /// on register scoreboards instead.
    TmemLd {
        d: Reg,
        n: u8,
        taddr: Val,
        off: u32,
        tracked: bool,
        /// `.16x64b.xN` (SASS LDTM.16dp64bit.xN): 16 lanes x 2N columns. Lane l of the warp
        /// reads TMEM lane l/4 + 8(l%2), register j column 2j + (l/2)%2 (GB300:
        /// probes/tmem_16x64b).
        s16x64b: bool,
        /// SASS `LDTM.16dp32bit_t0_t15` / `_t16_t31` (half of PTX `.16x32bx2`): only lanes
        /// 16h..16h+15 take part; lane l reads TMEM lane (address lane) + l%16, N columns.
        half: Option<u8>,
        /// `.16x256b.xN` (SASS `LDTM.16dp256bit`): 16 lanes x 8N columns, 4N registers; like
        /// an mma accumulator fragment: register 4j + 2h + c of lane l is TMEM lane
        /// l/4 + 8h, column 8j + 2(l%4) + c.
        s16x256b: bool,
        /// `.16x128b.xN` (SASS `LDTM.16dp128bit`): 16 lanes x 4N columns, 2N registers:
        /// register 2j + h of lane l is TMEM lane l/4 + 8h, column 4j + l%4 (GB300:
        /// probes/tmem_16x128b).
        s16x128b: bool,
    },
    /// `tcgen05.wait::ld`
    TmemWaitLd,
    /// `dp2a.{lo,hi}.{u32,s32}.{u32,s32} d, a, b, c` (SASS `IDP.2A.{LO,HI}.{U16,S16}.{U8,S8}`):
    /// d = c + a.h0 * b.byte(2hi) + a.h1 * b.byte(2hi+1), halves / bytes zero- or
    /// sign-extended, modulo 2^32.
    Dp2a {
        d: Reg,
        a: Val,
        b: Val,
        c: Val,
        hi: bool,
        a_signed: bool,
        b_signed: bool,
    },
    /// SASS `DFMA[.RZ|.RM|.RP] Rd, Ra, Rb, Rc` / `DMUL` (`c` None) / `DADD` (b = 1.0): f64 on
    /// register pairs, one rounding; operand modifiers `-x` / `|x|` per source.
    Dfma {
        d: Reg,
        a: Val,
        b: Val,
        c: Option<Val>,
        neg: [bool; 3],
        abs: [bool; 3],
        rnd: Rnd,
    },
    /// PTX `div.{rn,rz,rm,rp}.f64 d, a, b` (`b` Some) / `sqrt.{rn,..}.f64 d, a` (`b` None):
    /// IEEE, correctly rounded (ptxas's MUFU.RCP64H / RSQ64H + DFMA expansions are; the NaN
    /// results are measured: probes/f64_ptx).
    DivSqrt64 {
        d: Reg,
        a: Val,
        b: Option<Val>,
        rnd: Rnd,
    },
    /// SASS `DSETP.{MIN,MAX}.AND Pd, Pq, Ra, Rb, PT` (ptxas's min / max.f64): Pd = select a
    /// (a is the min / max: a not NaN and b NaN, or a strictly below / above b, -0 below +0),
    /// Pq = both are NaN (the result is then b, quieted).
    DsetpMinMax {
        d: Reg,
        q: Option<Reg>,
        a: Val,
        b: Val,
        max: bool,
    },
    /// SASS `DSETP.<cmp>.<AND|OR|XOR> Pd, PT, Ra, Rb, Pc` (f64 compare).
    Dsetp {
        d: Reg,
        cmp: FCmp,
        a: Val,
        b: Val,
        neg: [bool; 2],
        abs: [bool; 2],
        c: Pred,
        bop: BoolOp,
    },
    /// SASS `F2F.F32.F64 Rd, Ra` (f64 pair -> f32, rounded per `rnd`) / `F2F.F64.F32 Rd, Ra`
    /// (f32 -> f64 pair, exact).
    F2F {
        d: Reg,
        a: Val,
        to64: bool,
        rnd: Rnd,
    },
    /// SASS `HADD2.F32 Rd, -RZ, Ra.H0_H0` / `.H1_H1`: f16 half of `a` -> f32 (exact; adding -0).
    CvtF32F16 {
        d: Reg,
        a: Val,
        hi: bool,
    },
    /// SASS `F2F.F16.F32 Rd, Ra`: d = f16(a), round to nearest (upper half zero). `bf16`:
    /// `F2F.BF16.F32` (cvt.rn.bf16.f32 into a 16-bit register) instead. Both give the
    /// canonical NaN 0x7fff (GB300: fixtures/s_ptx_cute_ops).
    F2fF16 {
        d: Reg,
        a: Val,
        bf16: bool,
    },
    /// SASS `F2FP.F16.F32.PACK_AB Rd, Ra, Rb`: d = f16(a) << 16 | f16(b), round to nearest.
    F2fpF16 {
        d: Reg,
        a: Val,
        b: Val,
    },
    /// PTX `div.full.f32 d, a, b`, as ptxas expands it: if |b| > 2^126 both operands are
    /// scaled by 1/4, if |b| < 2^-126 by 2^24 (FMUL, nearest), then d = MUFU.RCP(b) * a.
    /// `rn`: `div.rn.f32` instead, the IEEE quotient (round to nearest even, subnormals kept;
    /// NaN results canonical).
    DivFull {
        d: Reg,
        a: Val,
        b: Val,
        rn: bool,
    },
    /// SASS `LDG.E.ENL2.256 Rhi, Rlo, [Ra.64]` (`ld.global.v8.b32`): a 32-byte global load;
    /// bytes 0..16 go to Rlo..Rlo+3, bytes 16..32 to Rhi..Rhi+3 (checked against ptxas).
    Ld256 {
        hi: Reg,
        lo: Reg,
        addr: Addr,
        sync: Option<MemSync>,
    },
    /// SASS `LDL[.LU] Rd, [Ra+off]` (`.64` / `.128`: consecutive registers): thread-private
    /// local memory (register spills). `LU` (last use) is only a cache hint.
    LdLocal {
        d: Reg,
        bytes: u32,
        addr: Addr,
    },
    /// SASS `STL [Ra+off], Rv`.
    StLocal {
        bytes: u32,
        addr: Addr,
        v: Val,
    },
    /// `ldmatrix.sync.aligned.m8n8.xN[.trans].shared.b16 {d..d+N-1}, [addr]` (SASS
    /// `LDSM.16.M88.N` / `.MT88.N`, warp-collective), the inverse of `StMatrix`: matrix i's
    /// row j is read from the address held by lane 8i + j; register d+i of lane l gets
    /// elements 2(l%4), 2(l%4)+1 of row l/4 (`.trans`: column l/4 of rows 2(l%4), 2(l%4)+1).
    LdMatrix {
        d: Reg,
        n: u8,
        addr: Addr,
        trans: bool,
    },
    /// `cp.async.cg.shared.global [dst], [src], bytes` (SASS `LDGSTS.E.BYPASS[.LTC128B].128
    /// [Rd+off], desc[UR][Ra.64]`): an asynchronous copy from global to this CTA's shared
    /// memory, part of the thread's open cp.async group.
    CpAsync {
        dst: Addr,
        src: Addr,
        bytes: u32,
        /// Source-valid predicate (SASS third operand; absent = PT): when false (src-size 0),
        /// zeros are written and global memory is not read (Triton's masked prefetch past the
        /// end: `!PT`).
        src_ok: Pred,
    },
    /// `cp.async.commit_group` (SASS `LDGDEPBAR`): close the open group.
    CpAsyncCommit,
    /// The asynchronous half of `cp.async.mbarrier.arrive[.noinc].shared.b64 [addr]` (SASS
    /// `ARRIVES.LDGSTSBAR.64.TRANSCNT` / `.ARVCNT`): once every cp.async this thread issued
    /// so far has landed, complete one transaction unit (`tx`; the non-`.noinc` form, whose
    /// synchronous half `SYNCS.ARRIVE.TRANS64.RED.A0T1` expects it) or arrive once (`.noinc`).
    CpAsyncMbarArrive {
        addr: Addr,
        tx: bool,
    },
    /// `cp.async.wait_group n` (SASS `DEPBAR.LE SBk, n` on LDGDEPBAR's scoreboard): block until
    /// at most n committed groups of this thread are incomplete.
    CpAsyncWait {
        n: u32,
    },
    /// `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (SASS
    /// `HMMA.16816.F32.BF16 Rd, Ra, Rb, Rc`, warp-collective): D (16x8 f32, 4 registers per
    /// lane) = A (16x16 bf16, 4 registers) x B (16x8 bf16, 2 registers) + C (4 registers;
    /// empty: zero). Fragments as in the PTX ISA (group g = lane/4, t = lane%4): a0 = A[g][2t,
    /// 2t+1], a1 = A[g+8][2t..], a2 = A[g][2t+8..], a3 = A[g+8][2t+8..]; b0 = B[2t, 2t+1][g],
    /// b1 = B[2t+8, 2t+9][g]; c0, c1 = C[g][2t, 2t+1], c2, c3 = C[g+8][2t, 2t+1].
    /// `tf32`: `mma.sync.m16n8k8...tf32` (SASS `HMMA.1688.F32.TF32`): A 16x8 and B 8x8, one
    /// f32 word per element: a0 = A[g][t], a1 = A[g+8][t], a2 = A[g][t+4], a3 = A[g+8][t+4];
    /// b0 = B[t][g], b1 = B[t+4][g]; C / D as above.
    Hmma {
        d: Reg,
        a: Vec<Val>,
        b: Vec<Val>,
        c: Vec<Val>,
        tf32: bool,
    },
}

impl Op {
    /// Visible operations are the scheduling points of the model checker: those that block
    /// or read/modify synchronization state other actors can observe. Everything else
    /// (plain and TMEM loads/stores, fences, issuing async work) runs eagerly: under DRF-SC
    /// its interleaving cannot change the outcome, and races between such accesses are found
    /// by happens-before tracking instead. Completing async work is a separate scheduler
    /// choice.
    // ASSUME(E-DRF): see ASSUMPTIONS.md
    pub fn is_visible(&self) -> bool {
        matches!(
            self,
            Op::Ld { sync: Some(_), .. }
                | Op::St { sync: Some(_), .. }
                | Op::BarSync { .. }
                | Op::MbarInit { .. }
                | Op::MbarArrive { .. }
                | Op::MbarTryWait { .. }
                | Op::ClusterArrive { .. }
                | Op::ClusterWait
                | Op::BulkWait { .. }
                | Op::Exit
                | Op::Atom { .. }
                | Op::WarpSync { .. }
                | Op::EndCollective { .. }
                | Op::Bsync { .. }
                | Op::Break { .. }
                | Op::TmemAlloc { .. }
                | Op::TmemAllocRaw { .. }
                | Op::StAsync { .. }
                | Op::ClcTryCancel { .. }
                | Op::GridDepWait
                | Op::GridDepLaunch
                | Op::TmemFreeRaw { .. }
                | Op::TmemRelinquish
                | Op::TmemDealloc { .. }
                | Op::Trap { .. }
        )
    }

    /// PTX `.sync.aligned` instructions: every lane of the warp must execute them together.
    pub fn is_warp_aligned(&self) -> bool {
        matches!(
            self,
            Op::StMatrix { .. }
                | Op::Shfl { .. }
                // elect.sync waits for every thread of its membermask (all lanes: -1).
                | Op::Elect { .. }
                | Op::TmemSt { .. }
                | Op::TmemWaitSt
                | Op::TmemAlloc { .. }
                | Op::TmemDealloc { .. }
                | Op::TmemRelinquish
                | Op::TmemLd { tracked: true, .. }
                | Op::TmemWaitLd
        )
    }

    /// Register values read by this operation, predicates included (not the guard). Exhaustive:
    /// used by "read before ready" checks and by the E-SPIN liveness check.
    pub fn src_regs(&self) -> Vec<Reg> {
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
        let vals = self.src_vals();
        vals.iter().for_each(&mut v);
        out
    }

    /// Operand values read by this operation (registers, immediates, constants, special
    /// registers), predicates included (not the guard).
    pub fn src_vals(&self) -> Vec<Val> {
        let pr = |p: &Pred| p.reg.map(|r| Val::Reg(r, 1)).unwrap_or(Val::Imm(0));
        let ad = |x: &Addr| {
            [x.base.clone(), x.base2.clone()]
                .into_iter()
                .flatten()
                .collect::<Vec<Val>>()
        };
        match self {
            Op::Mov { a, .. }
            | Op::CvtF32Bf16 { a, .. }
            | Op::Cvta { a, .. }
            | Op::CvtInt { a, .. }
            | Op::R2Ur { a, .. }
            | Op::Flo { a, .. }
            | Op::Brev { a, .. }
            | Op::I2F { a, .. }
            | Op::F2I { a, .. }
            | Op::Mufu { a, .. }
            | Op::F2F { a, .. }
            | Op::CvtF32F16 { a, .. }
            | Op::F2fF16 { a, .. }
            | Op::R2p { a, .. }
            | Op::CvtUnpack { a, .. }
            | Op::Redux { a, .. } => vec![a.clone()],
            Op::CvtPack { a, b, c, .. } => vec![a.clone(), b.clone(), c.clone()],
            Op::CvtRsX4 { src, rbits, .. } => {
                let mut x = src.to_vec();
                x.push(rbits.clone());
                x
            }
            Op::Int { a, b, .. }
            | Op::MulWide { a, b, .. }
            | Op::Lea { a, b, .. }
            | Op::Mapa { a, rank: b, .. }
            | Op::F2fpBf16 { a, b, .. }
            | Op::Fchk { a, b, .. }
            | Op::DivFull { a, b, .. }
            | Op::F2fpF16 { a, b, .. }
            | Op::Fadd { a, b, .. } => vec![a.clone(), b.clone()],
            Op::Fmnmx { a, b, c, min, .. } => {
                let mut x = vec![a.clone(), b.clone()];
                x.extend(c.iter().cloned());
                x.push(pr(min));
                x
            }
            Op::Fsetp { a, b, c, .. }
            | Op::Fset { a, b, c, .. }
            | Op::Hsetp2Bf16 { a, b, c, .. } => {
                vec![a.clone(), b.clone(), pr(c)]
            }
            Op::LdConst { base, .. } => vec![base.clone()],
            Op::P2r { a, mask, .. } => {
                let mut x = vec![a.clone()];
                x.extend((0..7).filter(|i| mask >> i & 1 == 1).map(|i| {
                    Val::Reg(
                        Reg {
                            file: File::P,
                            idx: i,
                        },
                        1,
                    )
                }));
                x
            }
            Op::Sel { a, b, p, .. } | Op::Fsel { a, b, p, .. } => vec![a.clone(), b.clone(), pr(p)],
            Op::Pack64 { lo, hi, .. } => vec![lo.clone(), hi.clone()],
            Op::Ffma2 { a, b, c, .. } => vec![a.0.clone(), b.0.clone(), c.0.clone()],
            Op::Unpack64 { a, .. } => vec![a.clone()],
            Op::Setp { a, b, and, ex, .. } => {
                let mut x = vec![a.clone(), b.clone(), pr(and)];
                x.extend(ex.iter().map(pr));
                x
            }
            Op::Imad { a, b, c, cin, .. } => {
                let mut x = vec![a.clone(), b.clone(), c.clone()];
                x.extend(cin.iter().map(pr));
                x
            }
            Op::Iadd3 { a, b, c, cin, .. } => {
                let mut x = vec![a.clone(), b.clone(), c.clone()];
                x.extend(cin.iter().map(pr));
                x
            }
            Op::Dfma { a, b, c, .. } => {
                let mut x = vec![a.clone(), b.clone()];
                x.extend(c.iter().cloned());
                x
            }
            Op::Dsetp { a, b, c, .. } => vec![a.clone(), b.clone(), pr(c)],
            Op::DsetpMinMax { a, b, .. } => vec![a.clone(), b.clone()],
            Op::DivSqrt64 { a, b, .. } => {
                let mut x = vec![a.clone()];
                x.extend(b.iter().cloned());
                x
            }
            Op::Ffma { a, b, c, .. } => {
                let mut x = vec![a.clone(), b.clone()];
                x.extend(c.iter().cloned());
                x
            }
            Op::Lop3 { a, b, c, pq, .. } => {
                let mut x = vec![a.clone(), b.clone(), c.clone()];
                x.extend(pq.iter().map(|(q, _)| pr(q)));
                x
            }
            Op::Dp2a { a, b, c, .. }
            | Op::Hfma2 { a, b, c, .. }
            | Op::AddMinMax { a, b, c, .. }
            | Op::IntMnmx3 { a, b, c, .. }
            | Op::Prmt {
                a, sel: b, b: c, ..
            } => vec![a.clone(), b.clone(), c.clone()],
            Op::LeaHi { a, b, c, cin, .. } => {
                let mut x = vec![a.clone(), b.clone(), c.clone()];
                x.extend(cin.iter().map(pr));
                x
            }
            Op::Shf { lo, sh, hi, .. } => vec![lo.clone(), sh.clone(), hi.clone()],
            Op::Plop3 { a, b, c, .. } => vec![pr(a), pr(b), pr(c)],
            Op::VoteAny { a, .. } | Op::VoteAll { a, .. } => vec![pr(a)],
            Op::Bfe { a, pos, len, .. } => vec![a.clone(), pos.clone(), len.clone()],
            Op::Shfl { a, lane, clamp, .. } => vec![a.clone(), lane.clone(), clamp.clone()],
            Op::StMatrix { addr, v: vs, .. } => {
                let mut x = ad(addr);
                x.extend(vs.iter().cloned());
                x
            }
            Op::TmemSt { taddr, v: vs, .. } => {
                let mut x = vec![taddr.clone()];
                x.extend(vs.iter().cloned());
                x
            }
            Op::TmapFence { addr } => ad(addr),
            Op::LdLocal { addr, .. }
            | Op::LdMatrix { addr, .. }
            | Op::Qspc { addr, .. }
            | Op::Ld256 { addr, .. } => ad(addr),
            Op::StLocal { addr, v: y, .. } => {
                let mut x = ad(addr);
                x.push(y.clone());
                x
            }
            Op::CpAsync { dst, src, .. } => {
                let mut x = ad(dst);
                x.extend(ad(src));
                x
            }
            Op::CpAsyncMbarArrive { addr, .. } => ad(addr),
            Op::Hmma { a, b, c, .. } => a.iter().chain(b).chain(c).cloned().collect(),
            Op::Ld { addr, .. } | Op::MbarInval { addr } | Op::TmemAlloc { dst: addr, .. } => {
                let mut x = ad(addr);
                if let Op::TmemAlloc { ncols, .. } = self {
                    x.push(ncols.clone());
                }
                x
            }
            Op::St { addr, v: y, .. } => {
                let mut x = ad(addr);
                x.push(y.clone());
                x
            }
            Op::Atom {
                addr, v: y, new, ..
            } => {
                let mut x = ad(addr);
                x.push(y.clone());
                x.extend(new.iter().cloned());
                x
            }
            Op::MbarInit { addr, count, .. } => {
                let mut x = ad(addr);
                x.push(count.clone());
                x
            }
            Op::MbarArrive {
                addr, tx, count, ..
            } => {
                let mut x = ad(addr);
                x.extend(tx.iter().cloned());
                x.extend(count.iter().cloned());
                x
            }
            Op::MbarTryWait { addr, parity, .. } => {
                let mut x = ad(addr);
                x.push(parity.clone());
                x
            }
            Op::Ret { addr } => vec![addr.clone()],
            Op::Brx { base, .. } => vec![base.clone()],
            Op::CallAbs { target, .. } => vec![target.clone()],
            Op::Bra { cond: Some(c), .. } => vec![pr(c)],
            Op::WarpSync { mask, .. } | Op::TmemFreeRaw { mask } => vec![mask.clone()],
            Op::BarSync { id, count, .. } => {
                let mut x = vec![id.clone()];
                x.extend(count.iter().cloned());
                x
            }
            Op::ClcTryCancel { resp, mbar, .. } => vec![resp.clone(), mbar.clone()],
            Op::ClcQuery { q, .. } => vec![Val::Reg(*q, 4)],
            Op::TmaLoad {
                dst,
                mbar,
                tmap,
                coords,
                mask,
                ..
            } => {
                let mut x = vec![dst.clone(), mbar.clone(), tmap.clone()];
                x.extend(coords.iter().cloned());
                x.extend(mask.iter().cloned());
                x
            }
            Op::TmaStore {
                src, tmap, coords, ..
            } => {
                let mut x = vec![src.clone(), tmap.clone()];
                x.extend(coords.iter().cloned());
                x
            }
            Op::TmemAllocRaw { units, .. } => vec![units.clone()],
            Op::StAsync { addr, mbar, v: y } => vec![addr.clone(), mbar.clone(), y.clone()],
            Op::TmemDealloc { taddr, ncols } => vec![taddr.clone(), ncols.clone()],
            Op::TcCp { taddr, sdesc, .. } => vec![taddr.clone(), sdesc.clone()],
            Op::Mma {
                d,
                adesc,
                bdesc,
                idesc,
                enable_d,
                disable,
                scale,
                ..
            } => {
                let mut x = vec![
                    d.clone(),
                    adesc.clone(),
                    bdesc.clone(),
                    idesc.clone(),
                    pr(enable_d),
                ];
                x.extend(disable.iter().cloned());
                if let Some((sa, sb)) = scale {
                    x.push(sa.clone());
                    x.push(sb.clone());
                }
                x
            }
            Op::TcCommit { mbar, mask, .. } => {
                let mut x = vec![mbar.clone()];
                x.extend(mask.iter().cloned());
                x
            }
            Op::TmemLd { taddr, .. } => vec![taddr.clone()],
            Op::Bra { div: Some(m), .. } => vec![m.clone()],
            Op::Elect { .. }
            | Op::EndCollective { .. }
            | Op::Bssy { .. }
            | Op::Bsync { .. }
            | Op::Break { .. }
            | Op::Bra { .. }
            | Op::Exit
            | Op::Nop
            | Op::Call { .. }
            | Op::Trap { .. }
            | Op::TmemWaitSt
            | Op::GridDepWait
            | Op::GridDepLaunch
            | Op::ClusterArrive { .. }
            | Op::ClusterWait
            | Op::Fence { .. }
            | Op::BulkCommit
            | Op::BulkWait { .. }
            | Op::TmemRelinquish
            | Op::CpAsyncCommit
            | Op::CpAsyncWait { .. }
            | Op::TmemWaitLd => vec![],
        }
    }

    /// Registers this operation writes (predicates included). Exhaustive.
    pub fn dst_regs(&self) -> Vec<Reg> {
        let span = |d: Reg, n: u16| {
            (0..n.max(1))
                .map(|i| Reg {
                    file: d.file,
                    idx: d.idx + if d.file == File::V { 0 } else { i },
                })
                .collect::<Vec<_>>()
        };
        match self {
            Op::Mov { d, a } => match a {
                Val::Reg(_, n) => span(*d, *n as u16),
                // `LDC.64 R10, c[..]` / `LDCU.128`: one register per 4 bytes.
                Val::Const(_, b) => span(*d, (*b as u16 / 4).max(1)),
                _ => span(*d, 1),
            },
            Op::Imad { d, wide, cout, .. } => {
                let mut x = span(*d, if *wide { 2 } else { 1 });
                x.extend(*cout);
                x
            }
            Op::Iadd3 { d, wide, cout, .. } => {
                let mut x = span(*d, if *wide { 2 } else { 1 });
                x.extend(cout.iter().copied());
                x
            }
            Op::Lea { d, cout, .. } => {
                let mut x = vec![*d];
                x.extend(*cout);
                x
            }
            Op::I2F { d, .. } => vec![*d],
            Op::F2I { d, bits, .. } => span(*d, *bits as u16 / 32),
            Op::LdConst { d, bytes, .. } => span(*d, (*bytes as u16 / 4).max(1)),
            Op::MulWide { d, .. } => span(*d, 2),
            Op::Int { d, bits, .. }
            | Op::CvtInt {
                d, to_bits: bits, ..
            } => span(*d, (*bits as u16 / 32).max(1)),
            Op::Cvta { d, .. } => span(*d, 2),
            Op::R2Ur { d, p, .. } => {
                let mut x = vec![*d];
                x.extend(*p);
                x
            }
            Op::Lop3 { d, p, .. } => {
                let mut x = vec![*d];
                x.extend(*p);
                x
            }
            Op::Elect { d, lane } => {
                let mut x = vec![*d];
                x.extend(*lane);
                x
            }
            Op::VoteAny { d, .. }
            | Op::Flo { d, .. }
            | Op::Brev { d, .. }
            | Op::AddMinMax { d, .. }
            | Op::IntMnmx3 { d, .. }
            | Op::P2r { d, .. }
            | Op::Ffma { d, .. }
            | Op::F2fpBf16 { d, .. }
            | Op::Redux { d, .. }
            | Op::Shf { d, .. }
            | Op::Plop3 { d, .. }
            | Op::Setp { d, .. }
            | Op::VoteAll { d, .. }
            | Op::Fadd { d, .. }
            | Op::Hfma2 { d, .. }
            | Op::CvtF32Bf16 { d, .. }
            | Op::CvtPack { d, .. }
            | Op::CvtRsX4 { d, .. }
            | Op::CvtUnpack { d, .. }
            | Op::Shfl { d, p: None, .. }
            | Op::Bfe { d, .. }
            | Op::LeaHi { d, cout: None, .. }
            | Op::Mufu { d, .. }
            | Op::Fmnmx { d, .. }
            | Op::Fsetp { d, .. }
            | Op::Hsetp2Bf16 { d, .. }
            | Op::Fset { d, .. }
            | Op::Fchk { d, .. }
            | Op::Fsel { d, .. }
            | Op::Sel { d, .. }
            | Op::Prmt { d, .. }
            | Op::MbarTryWait { d, .. } => vec![*d],
            Op::Mapa { d, .. } | Op::Ffma2 { d, .. } => span(*d, 2),
            Op::LeaHi {
                d, cout: Some(c), ..
            } => vec![*d, *c],
            Op::Pack64 { d, .. } => vec![*d],
            Op::R2p { mask, .. } => (0..7)
                .filter(|i| mask >> i & 1 == 1)
                .map(|i| Reg {
                    file: File::P,
                    idx: i,
                })
                .collect(),
            Op::Unpack64 { lo, hi, .. } => lo.iter().chain(hi.iter()).copied().collect(),
            Op::Atom {
                d, bytes, p, op, ..
            } => {
                let w = if *op == AtomOp::CasSpin {
                    1
                } else {
                    (*bytes as u16 / 4).max(1)
                };
                let mut x = d.map(|d| span(d, w)).unwrap_or_default();
                x.extend(*p);
                x
            }
            Op::Qspc { p, .. } => vec![*p],
            Op::Ld { d, bytes, .. } => span(*d, (*bytes as u16 / 4).max(1)),
            Op::ClcQuery { d, .. } => span(*d, 4),
            Op::TmemAllocRaw { d, ok, .. } => vec![*d, *ok],
            Op::TmemLd { d, n, .. } => span(*d, *n as u16),
            Op::LdLocal { d, bytes, .. } => span(*d, (*bytes as u16 / 4).max(1)),
            Op::Ld256 { hi, lo, .. } => {
                let mut x = span(*lo, 4);
                x.extend(span(*hi, 4));
                x
            }
            Op::Shfl { d, p: Some(p), .. } => vec![*d, *p],
            Op::DivFull { d, .. }
            | Op::Dp2a { d, .. }
            | Op::CvtF32F16 { d, .. }
            | Op::F2fF16 { d, .. }
            | Op::F2fpF16 { d, .. } => vec![*d],
            Op::F2F { d, to64, .. } => span(*d, if *to64 { 2 } else { 1 }),
            Op::Dfma { d, .. } => span(*d, 2),
            Op::DivSqrt64 { d, .. } => vec![*d],
            Op::Dsetp { d, .. } => vec![*d],
            Op::DsetpMinMax { d, q, .. } => {
                let mut x = vec![*d];
                x.extend(*q);
                x
            }
            Op::LdMatrix { d, n, .. } => span(*d, *n as u16),
            Op::Hmma { d, .. } => span(*d, 4),
            Op::St { .. }
            | Op::Bra { .. }
            | Op::EndCollective { .. }
            | Op::Exit
            | Op::Nop
            | Op::Call { .. }
            | Op::Ret { .. }
            | Op::Brx { .. }
            | Op::CallAbs { .. }
            | Op::Trap { .. }
            | Op::WarpSync { .. }
            | Op::Bssy { .. }
            | Op::Bsync { .. }
            | Op::Break { .. }
            | Op::BarSync { .. }
            | Op::MbarInit { .. }
            | Op::MbarArrive { .. }
            | Op::ClcTryCancel { .. }
            | Op::StMatrix { .. }
            | Op::TmemSt { .. }
            | Op::TmemWaitSt
            | Op::MbarInval { .. }
            | Op::TmapFence { .. }
            | Op::GridDepWait
            | Op::GridDepLaunch
            | Op::ClusterArrive { .. }
            | Op::ClusterWait
            | Op::Fence { .. }
            | Op::TmaLoad { .. }
            | Op::TmaStore { .. }
            | Op::BulkCommit
            | Op::BulkWait { .. }
            | Op::TmemAlloc { .. }
            | Op::StAsync { .. }
            | Op::TmemFreeRaw { .. }
            | Op::TmemRelinquish
            | Op::TmemDealloc { .. }
            | Op::Mma { .. }
            | Op::TcCp { .. }
            | Op::TcCommit { .. }
            | Op::StLocal { .. }
            | Op::CpAsync { .. }
            | Op::CpAsyncMbarArrive { .. }
            | Op::CpAsyncCommit
            | Op::CpAsyncWait { .. }
            | Op::TmemWaitLd => vec![],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Front {
    Ptx,
    Sass,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Inst {
    pub guard: Pred,
    pub op: Op,
    /// Executed once per warp-split (SASS uniform datapath), not once per thread.
    pub uniform: bool,
    /// PTX: 1-based source line. SASS: byte address of the instruction.
    pub loc: u32,
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub offset: u32,
    pub size: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Kernel {
    pub name: String,
    pub front: Front,
    pub code: Vec<Inst>,
    /// Number of registers per file, indexed by `File as usize`.
    pub nregs: [u16; 5],
    pub params: Vec<Param>,
    /// Offset of the parameter block in constant bank 0.
    pub param_cbank: u32,
    /// Total static shared memory, including the system-reserved prefix.
    pub smem_bytes: u32,
    /// Human-readable register names (PTX virtual registers), indexed like `nregs[V]`.
    pub vreg_names: Vec<String>,
    /// Code labels (instruction index, name), sorted; used to name trap sites.
    pub labels: Vec<(usize, String)>,
    /// Findings that hold for the instruction stream itself, independent of any schedule
    /// (SASS: scoreboard hazards). Reported alongside the model checker's findings.
    pub static_findings: Vec<StaticFinding>,
    /// SASS: instructions whose control word waits on the bulk-group read scoreboard
    /// (`UTMACMDFLUSH`'s rbar), sorted. Each first waits as `cp.async.bulk.wait_group.read 0`.
    pub bulk_read_waits: Vec<usize>,
    /// Kernels of a multi-kernel launch (`Kernel::concat`), in launch order; empty for a
    /// single kernel. Their code is concatenated in `code`.
    pub parts: Vec<KernelPart>,
}

/// One kernel of a concatenated multi-kernel launch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelPart {
    pub name: String,
    /// First instruction of this kernel in the combined `code`.
    pub entry: usize,
    pub params: Vec<Param>,
    pub param_cbank: u32,
    pub smem_bytes: u32,
    pub vreg_names: Vec<String>,
}

/// SASS control-bit edit used to build binary-only mutants (applied by `ptxv_sass::scoreboard`).
/// One edit of an instruction's control bits, as written in a case manifest:
/// `at` selects the first instruction of `kernel` whose disassembly contains it, or the
/// instruction at an address written `@0x0a70`.
#[derive(Clone, Debug, Default, serde::Deserialize, Serialize)]
pub struct CtrlPatch {
    pub at: String,
    /// Replacement wait mask (bit k = SBk).
    pub wait: Option<u8>,
    /// Replacement write / read scoreboard (7 = none).
    pub wbar: Option<u8>,
    pub rbar: Option<u8>,
    pub stall: Option<u8>,
    /// Instruction edits (rung 4a mutants of library kernels): replace the instruction by
    /// the kernel's NOP padding, keeping its control bits ...
    #[serde(default)]
    pub nop: bool,
    /// ... or set bit fields of the 128-bit encoding: `[start bit, width, value]` (e.g. the
    /// guard predicate is bits 12..16, a 32-bit immediate bits 32..64).
    #[serde(default)]
    pub bits: Vec<[u64; 3]>,
    /// The patched instruction's expected disassembly, guard included (checked after
    /// patching).
    #[serde(default)]
    pub expect: Option<String>,
    /// The kernel to patch (default: the case's `entry`), e.g. one of several kernels of a
    /// captured library module (rung 4b).
    #[serde(default)]
    pub kernel: Option<String>,
    /// The cubin to patch: a launch's own cubin (default: the case's `cubin`).
    #[serde(default)]
    pub cubin: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StaticFinding {
    /// Same vocabulary as dynamic findings ("violation", ...).
    pub kind: String,
    pub sig: String,
    pub msg: String,
    /// Instructions involved (indices into `code`): producer, the consumer the message names,
    /// then further consumers.
    pub pcs: Vec<usize>,
}

impl Kernel {
    /// Name of the function/label region containing `pc` (nearest preceding label).
    pub fn region(&self, pc: usize) -> Option<&str> {
        self.labels
            .iter()
            .rev()
            .find(|(i, _)| *i <= pc)
            .map(|(_, n)| n.as_str())
    }

    pub fn describe(&self, pc: usize) -> String {
        let k = match self.part_of(pc) {
            Some(p) => format!("{} ", p.name),
            None => String::new(),
        };
        match self.code.get(pc) {
            Some(i) => match self.front {
                Front::Ptx => format!("{k}ptx:{} `{}`", i.loc, i.text),
                Front::Sass => format!("{k}sass:{:#06x} `{}`", i.loc, i.text),
            },
            None => format!("pc {pc} (end)"),
        }
    }

    /// The kernel (of a concatenated launch) whose code contains `pc`.
    pub fn part_of(&self, pc: usize) -> Option<&KernelPart> {
        self.parts.iter().rev().find(|p| p.entry <= pc)
    }

    /// First instruction of the kernel containing `pc` (0 for a single kernel).
    pub fn entry_of(&self, pc: usize) -> usize {
        self.part_of(pc).map_or(0, |p| p.entry)
    }

    /// Register names for the kernel containing `pc`.
    pub fn vreg_names_at(&self, pc: usize) -> &[String] {
        self.part_of(pc).map_or(&self.vreg_names, |p| &p.vreg_names)
    }

    /// Concatenate the kernels of a multi-kernel launch (same front-end) into one code
    /// stream: branch targets, labels and static findings are relocated; register counts are
    /// the maximum (each thread has its own registers). Kernel-level fields are the first
    /// kernel's; `parts` keeps every kernel's own.
    pub fn concat(ks: Vec<Kernel>) -> Kernel {
        assert!(!ks.is_empty());
        let mut out = ks[0].clone();
        out.code.clear();
        out.labels.clear();
        out.static_findings.clear();
        out.bulk_read_waits.clear();
        out.name = ks
            .iter()
            .map(|k| k.name.as_str())
            .collect::<Vec<_>>()
            .join("+");
        for k in &ks {
            assert!(k.front == out.front && k.parts.is_empty());
            let base = out.code.len();
            out.parts.push(KernelPart {
                name: k.name.clone(),
                entry: base,
                params: k.params.clone(),
                param_cbank: k.param_cbank,
                smem_bytes: k.smem_bytes,
                vreg_names: k.vreg_names.clone(),
            });
            for f in 0..5 {
                out.nregs[f] = out.nregs[f].max(k.nregs[f]);
            }
            out.smem_bytes = out.smem_bytes.max(k.smem_bytes);
            for i in &k.code {
                let mut i = i.clone();
                match &mut i.op {
                    Op::Bra { target, .. } | Op::Call { target } => *target += base,
                    Op::Brx { targets, .. } => targets.iter_mut().for_each(|t| *t += base),
                    _ => {}
                }
                out.code.push(i);
            }
            out.labels
                .extend(k.labels.iter().map(|(i, n)| (i + base, n.clone())));
            out.bulk_read_waits
                .extend(k.bulk_read_waits.iter().map(|i| i + base));
            for f in &k.static_findings {
                let mut f = f.clone();
                f.pcs.iter_mut().for_each(|p| *p += base);
                out.static_findings.push(f);
            }
        }
        out
    }
}

/// Constant-bank-0 offsets the driver fills in (Blackwell, validated by differential runs).
pub mod cbank {
    /// Generic address of the kernel parameter block (used by TMA tensor-map operands).
    pub const PARAM_GENERIC_BASE: u32 = 0x348;
}

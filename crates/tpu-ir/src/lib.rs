/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The IR that both TPU front-ends (libtpu LLO `final_bundles` and `original` listings) lower
//! into.
//!
//! Unlike `ptxv-ir`, which models SIMT threads with 32-bit register slots, this models one
//! TensorCore: a single VLIW instruction stream (`Bundle`s) driving 8x128x32-bit vector
//! registers, FIFO-coupled units (EUP, XLU, MXU) and asynchronous DMA engines. Every `Inst`
//! keeps the listing text and bundle address it came from, so reports point at the listing.
//! Semantics of each operation: fb/targets/tpu/notes/sim.md (clean room: inferred from
//! listings and hardware runs).

use serde::Deserialize;
use serde::Serialize;

/// Which listing a kernel was lifted from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Front {
    /// Scheduled VLIW bundles, physical registers (the SASS analogue).
    Final,
    /// Unscheduled SSA, one op per step, structured regions and phis (the PTX analogue).
    Original,
}

/// Register files. `idx` is the physical register in final listings and the SSA number in
/// original listings.
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
    S,
    P,
    V,
    Vm,
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
    pub idx: u32,
}

/// Memory spaces.
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
pub enum Space {
    Smem,
    Vmem,
    Hbm,
    Sflag,
}

/// A static allocation (`#allocationN` of the original listing's header, or a spill slot the
/// register allocator added in the final listing).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Alloc {
    pub name: String,
    pub space: Space,
    /// Bytes.
    pub size: u64,
    /// Fixed absolute byte address (`fixed_abs`), if any.
    pub fixed: Option<u64>,
    /// The listing's `tag = '...'` (e.g. `barrier sync flag number`, `dummy sync flag`).
    pub tag: String,
}

/// One term of an address sum.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Term {
    Reg(Reg),
    /// Index into `Kernel::allocs`.
    Alloc(u32),
    Imm(i64),
}

/// An address: the sum of its terms, in the unit of its space (SMEM words, VMEM rows, HBM
/// granules, flag words).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Addr {
    pub terms: Vec<Term>,
}

/// A value operand. Vector operands that are not registers broadcast the scalar value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Val {
    Reg(Reg),
    Imm(u32),
    Addr(Addr),
}

/// An instruction guard.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Guard {
    Reg { reg: Reg, neg: bool },
    Const(bool),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SBin {
    Add,
    Sub,
    Mul,
    MulHiU,
    And,
    Or,
    Xor,
    Shl,
    Shrl,
    Shra,
    MinU,
    MaxU,
    MinS,
    MaxS,
    DivU,
    RemU,
    DivS,
    RemS,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Cond {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CmpTy {
    S32,
    U32,
    F32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PBin {
    Or,
    And,
    Nand,
    Nor,
    Xor,
}

/// Floating-point VPU ops (f32 lanes, or packed bf16 halves).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FOp {
    Add,
    Sub,
    Mul,
    Max,
    Min,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    Shrl,
    Shra,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MOp {
    And,
    Or,
    Xor,
    Not,
}

/// EUP (transcendental) functions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EupFn {
    Rcp,
    Exp2,
    Log2,
    Rsqrt,
    Tanh,
    Sqrt,
    Exp,
    Log,
}

/// Result FIFOs of a core.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Fifo {
    /// Sync-flag values (`vsyncmov` -> `vpop.sfrf`).
    Sfrf,
    Eup,
    /// XLU unit results (`vpop.xlane/permute/trf.xluN`).
    Xlu(u8),
    /// Vector to scalar (`vpush` -> `spop`).
    V2s,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum XlaneOp {
    Add,
    Max,
    Min,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum XposePhase {
    Start,
    Cont,
    End,
}

/// DMA kinds (source -> destination space).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DmaKind {
    pub src: Space,
    pub dst: Space,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Op {
    /// No effect (pseudo-ops of final listings: `int_to_ptr`, `sphi`, `vphi`, `sfence`, ...).
    Nop,
    /// `int_to_ptr` in an original listing: a copy.
    Copy(Val),
    /// `inlined_call_operand.<space>`: the address of operand `index`.
    Operand {
        index: u32,
    },
    SMov(Val),
    PMov(Val),
    SBin {
        op: SBin,
        a: Val,
        b: Val,
    },
    SCmp {
        cond: Cond,
        ty: CmpTy,
        a: Val,
        b: Val,
    },
    SSelect {
        p: Val,
        t: Val,
        f: Val,
    },
    PNeg(Val),
    PBin {
        op: PBin,
        a: Val,
        b: Val,
    },
    SCvtSF(Val),
    SCvtFS(Val),
    SLd(Addr),
    SSt {
        addr: Addr,
        val: Val,
    },
    /// `shalt.err`: halt with an error (bounds checks, the exit semaphore check).
    HaltErr {
        msg: String,
    },
    Halt,
    /// `sbr.rel`: jump to bundle `target` after `delay` more bundles; `stops` has bit k set
    /// when bundle `k` of the window (1-based) holds a guarded sync-flag op, which ends the
    /// window at run time if its guard holds. Delay and stops are resolved by the front-end.
    /// Taken: the next `delay` bundles execute, then control moves to bundle `target`.
    Branch {
        target: u32,
        delay: u8,
    },
    /// Original listings: SSA phi of a loop header (first operand from the pre-header,
    /// second from the back edge).
    Phi {
        init: Val,
        back: Val,
        vector: bool,
    },
    SyncAdd {
        flag: Val,
        val: Val,
    },
    /// `vsyncpa flag, k`: set the flag.
    SyncSet {
        flag: Val,
        val: Val,
    },
    SyncMov {
        flag: Val,
    },
    /// `vwait.ge` / `vwait.eq` / `dma.done.wait`: block until the flag satisfies the test.
    Wait {
        flag: Val,
        val: Val,
        eq: bool,
    },
    /// Original listings: `dma.done f, n` = wait for n then subtract n.
    DmaDone {
        flag: Val,
        val: Val,
    },
    RemoteSyncAdd {
        addr: Val,
        val: Val,
    },
    Dma {
        kind: DmaKind,
        src: Val,
        dst: Val,
        size: Val,
        flag: Val,
        stride: Option<[Val; 3]>,
    },
    DmaGeneral {
        kind: DmaKind,
        src: Val,
        dst: Val,
        size: Val,
        dflag: Val,
        sflag: Val,
        overrides: Val,
    },
    VLd {
        addr: Addr,
        sm: u8,
        stride: Option<(u32, u32)>,
    },
    VSt {
        addr: Addr,
        sm: u8,
        vm: Option<Reg>,
        src: Val,
    },
    VMov(Val),
    VStv(Val),
    VF32 {
        op: FOp,
        a: Val,
        b: Val,
    },
    VBf16 {
        op: FOp,
        a: Val,
        b: Val,
    },
    VInt {
        op: IOp,
        a: Val,
        b: Val,
    },
    VCmp {
        cond: Cond,
        ty: CmpTy,
        a: Val,
        b: Val,
    },
    VMask {
        op: MOp,
        a: Reg,
        b: Option<Reg>,
    },
    VSel {
        m: Reg,
        t: Val,
        f: Val,
    },
    VTrunc(Val),
    VCvtFS(Val),
    VCvtSF(Val),
    VCvtTZ(Val),
    /// `vpack.c.f32.bf16 hi, lo` (`int`: `vpack.c.b32.b16`, truncating).
    VPack {
        hi: Val,
        lo: Val,
        int: bool,
    },
    /// `vunpack.c.{l,h/u}.bf16.f32` (`int`: `.s16.s32`, sign-extending).
    VUnpack {
        a: Val,
        high: bool,
        int: bool,
    },
    /// `vunpack.c.{l,u}.s8.bf16`: sublanes 0..3 (l) / 4..7 (u) hold 16 int8 rows (row 4s+i in
    /// byte i of sublane s); result = those rows as bf16, (2,1)-packed.
    VUnpackS8 {
        a: Val,
        high: bool,
    },
    VCombine {
        a: Val,
        b: Val,
        high: bool,
    },
    VLaneSeq,
    VRotSlane {
        a: Val,
        k: Val,
    },
    /// `vperm.slane a, pattern` (sublane permute; only the broadcast patterns observed).
    VPermSlane {
        a: Val,
        pat: Val,
    },
    /// `vunpack.c.<q>.s8.s32`: int8 -> int32, quarter `q`.
    VUnpackS8S32 {
        a: Val,
        q: u8,
    },
    VRotLane {
        unit: u8,
        a: Val,
        k: Val,
    },
    XLane {
        unit: u8,
        op: XlaneOp,
        a: Val,
    },
    VXpose {
        unit: u8,
        phase: XposePhase,
        a: Val,
        width: u32,
    },
    VCmask(Val),
    VPush(Val),
    VTos(Val),
    Eup {
        f: EupFn,
        a: Val,
        fused_pop: bool,
    },
    Pop(Fifo),
    MxuSubr {
        unit: u8,
        a: Val,
    },
    MxuPush {
        unit: u8,
        a: Val,
    },
    MxuPrep {
        unit: u8,
        a: Val,
    },
    MxuMatmul {
        unit: u8,
        a: Val,
        slot: u32,
        f32: bool,
        latch: bool,
    },
    MxuPop {
        unit: u8,
        slot: u32,
    },
    /// Original-listing MXU (FIFO form): pushes of f32 (8 K-rows) or bf16 (16 K-rows) vregs,
    /// `hi` = the N 128..255 half.
    OMxuPush {
        unit: u8,
        a: Val,
        hi: bool,
        bf16: bool,
    },
    OMxuMatmul {
        unit: u8,
        a: Val,
        f32: bool,
    },
    OMxuPop {
        unit: u8,
    },
    /// An op the front-end did not lower (`Kernel::unlowered[idx]`); halts if executed.
    Unlowered {
        idx: u32,
    },
}

impl Op {
    /// Sync-flag ops: they end a branch delay window (TPU-E-BR-SYNCSTOP).
    pub fn is_sync(&self) -> bool {
        matches!(
            self,
            Op::SyncAdd { .. }
                | Op::SyncSet { .. }
                | Op::SyncMov { .. }
                | Op::Wait { .. }
                | Op::DmaDone { .. }
                | Op::RemoteSyncAdd { .. }
        )
    }

    /// Ops whose order relative to other cores and DMA engines can matter: the explorer
    /// schedules around them.
    pub fn is_visible(&self) -> bool {
        self.is_sync()
            || matches!(
                self,
                Op::Dma { .. } | Op::DmaGeneral { .. } | Op::HaltErr { .. } | Op::Halt
            )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Inst {
    pub guard: Option<Guard>,
    pub def: Option<Reg>,
    pub op: Op,
    /// Mnemonic as printed.
    pub mn: String,
    /// Listing text of the op.
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bundle {
    pub insts: Vec<Inst>,
    /// Address printed in the listing (final) or op index (original).
    pub addr: u32,
    /// Loop depth markers (`>`).
    pub depth: u8,
}

/// An operand of the kernel (`inlined_call_operand`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperandDecl {
    pub index: u32,
    /// `vmem`, `hbm`, `smem` or `none` (passed in a register).
    pub space: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub output: bool,
}

/// An op the front-end could not lower.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Unlowered {
    pub bundle: u32,
    pub mn: String,
    pub text: String,
    pub why: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Kernel {
    pub name: String,
    pub front: Front,
    pub bundles: Vec<Bundle>,
    pub allocs: Vec<Alloc>,
    pub operands: Vec<OperandDecl>,
    /// Ops left unlowered (they halt the machine if reached).
    pub unlowered: Vec<Unlowered>,
    /// Where the branch delays came from: `none` (no branches / original listing),
    /// `binary` (decoded from the executable), `inferred`, or `mixed`.
    pub delay_source: String,
    /// Assumption IDs the lifted program rests on beyond the always-on ones.
    pub assumptions: Vec<String>,
    /// Front-end notes (branch target cross-checks, alignment quality).
    pub notes: Vec<String>,
}

impl Kernel {
    pub fn describe(&self, pc: usize) -> String {
        match self.bundles.get(pc) {
            Some(b) => format!(
                "{} {:#x}: {}",
                self.name,
                b.addr,
                b.insts
                    .iter()
                    .map(|i| i.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ;; ")
            ),
            None => format!("{} end", self.name),
        }
    }

    /// Number of ops.
    pub fn n_insts(&self) -> usize {
        self.bundles.iter().map(|b| b.insts.len()).sum()
    }
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Input coverage (A) and branch-flipping input mutation (B), `ptxv check`.
//!
//! A verdict holds for the captured input bytes only: a path that no thread takes on that
//! input is not checked under any schedule (a bucket-fill kernel: lanes
//! with ok = 0 store distinct values to one sentinel-row address, a race that no lane runs
//! on an all-valid capture).
//!
//! A (`analyze`): one deterministic run (`explore::run_once`) with a provenance shadow. Every
//! global byte the launch starts with is input; values carry the input bytes they derive from
//! (`Prov`), through registers, shared and global memory. Every comparison that writes a
//! predicate (`setp` / ISETP / FSETP / DSETP) from input-derived operands is a site; the
//! predicate values the run's threads produced there are recorded. A site where only one
//! value was seen has an untaken side: an input-dependent path not exercised on this input.
//! Comparisons, not branches, are the sites: Triton turns masks into predicated stores and
//! selects, and a bounds mask can cover the other side of a combined predicate (tail lanes)
//! while the input-dependent component never flips.
//!
//! B (`plan`, `apply`): for an untaken side whose comparison reads one input word directly
//! (a load of input bytes, then moves / integer widening) against a value that does not
//! depend on the input (an immediate, a constant-bank word, a register computed from
//! neither), write a value that flips it into K elements (threads in different warps where
//! possible) and check again. Results on mutated inputs are their own evidence, never part
//! of the verdict on the captured input. Anything else is refused with a named reason.
//!
//! The same run records every instruction other than register arithmetic whose guard, address,
//! control or synchronization operand derives from input data (`input_dependent`). None, on a
//! run that finished: the control path, the accesses and the synchronization do not depend on
//! the input bytes, so they are the same for every input of the captured shape and parameters
//! (the argument of Volta's structured CTAs and of N-EQUIV's taint precondition); the parts of
//! a verdict that compare values (same-value writes, determinism, the GPU output) still hold
//! for the captured input only. It also reports the instructions no thread reached and the
//! reached branch or predicated instruction in front of each (`unexecuted`): behind an
//! input-dependent condition, or behind one free of input data (parameters, special registers:
//! another shape or launch configuration, which no input mutation reaches).
//!
//! Declared input invariants (`Invariant`, case `buffers[].invariant`) are checked on the
//! captured bytes, never trusted (`Domain::new`); B writes only values they admit, and a check
//! whose every written element is admitted is labelled `within declared invariants`.
//!
//! One schedule: a site whose outcomes depend on the schedule (atomic counters) may take
//! the other side in another one. The provenance is data provenance: a value read through
//! an input-dependent address, an atomic's result, a shuffle, TMEM, or a spilled value that
//! was input-derived counts as input-derived but not as a single input word.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use ptxv_ir::Cmp;
use ptxv_ir::File;
use ptxv_ir::Op;
use ptxv_ir::Pred;
use ptxv_ir::Reg;
use ptxv_ir::Val;
use serde::Serialize;

use crate::exec::Loc;
use crate::machine::Machine;

/// Where a register value comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Prov {
    /// No input byte.
    Clean,
    /// Exactly one input word: `bytes` bytes of allocation `alloc` at `off`, loaded directly
    /// (zero- or sign-extended, `sext`), possibly moved or widened since.
    Word {
        alloc: u32,
        off: u64,
        bytes: u8,
        sext: bool,
        /// The old value of an atomic add of a non-negative constant to the word: at least
        /// the word's initial value (other threads' adds come on top).
        atom: bool,
    },
    /// Derived from input bytes of these allocations (bit i: allocation i; bit 63: 63 and
    /// above), not as a single word.
    Mixed(u64),
}

impl Prov {
    fn mask(self) -> u64 {
        match self {
            Prov::Clean => 0,
            Prov::Word { alloc, .. } => 1u64 << (alloc.min(63)),
            Prov::Mixed(m) => m,
        }
    }
    fn of_mask(m: u64) -> Prov {
        if m == 0 { Prov::Clean } else { Prov::Mixed(m) }
    }
}

/// Where a memory byte comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum MByte {
    Clean,
    /// Input byte `off` of allocation `alloc` (also a copy of it).
    In(u32, u64),
    Mask(u64),
}

impl MByte {
    fn mask(self) -> u64 {
        match self {
            MByte::Clean => 0,
            MByte::In(a, _) => 1u64 << a.min(63),
            MByte::Mask(m) => m,
        }
    }
}

/// One thread's observation at a comparison site whose result B could flip.
#[derive(Clone, Debug)]
struct Example {
    warp: u32,
    alloc: u32,
    off: u64,
    bytes: u8,
    sext: bool,
    /// The word is an atomic add's old value (`Prov::Word::atom`).
    atom: bool,
    /// The input word is operand `a` (else `b`).
    word_is_a: bool,
    /// The other operand's value.
    other: u64,
    cmp: Cmp,
    bits: u8,
    /// The combining predicate's value and mode.
    and_val: bool,
    or: bool,
    xor: bool,
}

#[derive(Clone, Debug, Default)]
struct Site {
    /// bit 0: false seen, bit 1: true seen.
    seen: u8,
    /// Thread observations per outcome (index = outcome).
    n: [u64; 2],
    /// Input allocations and byte ranges the compared operands derive from (words only).
    words: BTreeMap<u32, (u64, u64)>,
    /// Input allocations the operands derive from (any provenance).
    allocs: u64,
    /// Examples per outcome (index = outcome), for B.
    examples: [Vec<Example>; 2],
    /// Why some observation was not a simple single-word comparison (the first reason).
    refuse: Option<String>,
}

/// A comparison's plan: its predicate destination, the provenance of its result, and the
/// observation (`CmpObs`).
type CmpPlan = (
    Reg,
    Prov,
    Option<Example>,
    Option<String>,
    BTreeMap<u32, (u64, u64)>,
    u64,
);

/// What `post` does for one thread.
#[derive(Clone, Debug)]
enum Plan {
    /// Every destination gets this provenance.
    All(Prov),
    /// A load: destinations from the bytes read (`acc`), unless the address is
    /// input-derived (`addr` mask) or the access is atomic.
    Load {
        addr: u64,
        sext: bool,
        bytes: u8,
        atomic: bool,
        extra: u64,
    },
    /// A comparison: record the predicate's value (destination `d`), then `All(p)`.
    Cmp(Box<CmpPlan>),
}

/// The data a thread's stores of the current instruction write: one value for every byte, or
/// one per byte.
#[derive(Clone, Debug, Default)]
enum WData {
    #[default]
    Unset,
    One(MByte),
    Bytes(Vec<MByte>),
}

impl WData {
    fn mask(&self) -> u64 {
        match self {
            WData::Unset => 0,
            WData::One(b) => b.mask(),
            WData::Bytes(v) => v.iter().fold(0, |x, b| x | b.mask()),
        }
    }
}

/// What `pre` needs of an instruction, computed once per pc.
#[derive(Debug)]
struct Info {
    /// Register-only arithmetic (`data_only`).
    pure: bool,
    /// Source registers other than stored / exchanged values (for a non-arithmetic
    /// instruction): control, address and synchronization operands.
    ctl: Vec<Reg>,
    /// `Op::src_vals`.
    srcv: Vec<Val>,
    /// `Op::src_regs`.
    srcr: Vec<Reg>,
    /// Register arithmetic whose results come from its own lane's operands only (`pure`,
    /// and not a warp-collective operation).
    local: bool,
}

/// The first access of one run to a global allocation (`first_access`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct FirstAcc {
    /// It is a thread's plain write.
    write: bool,
    actor: u32,
    /// The first access's epoch (actor, clock), for a first read.
    rd: (u32, u32),
    /// A later plain write that the first read does not happen-before.
    late: bool,
}

/// Does the read `rd` (actor, own clock) happen-before a write by `w` at clock `vc`, in every
/// order? Happens-before, and for two lanes of one warp more: the SASS front joins the lanes
/// of a split instruction by instruction (E-LOCKSTEP), also after a divergence, where the
/// explorer runs the paths in either order; so lanes of one warp count as ordered only while
/// the warp is converged at the write and has not diverged since the read (as E-WARP-ORDER's
/// labels). Conservative: an ordered pair it rejects only makes a scratch buffer produced.
fn read_orders(m: &Machine, rd: (u32, u32), w: u32, vc: &crate::machine::VC) -> bool {
    if rd.1 > crate::machine::vc_get(vc, rd.0) {
        return false;
    }
    let (r, wi) = (rd.0 as usize, w as usize);
    if r == wi || r >= m.threads.len() || wi >= m.threads.len() {
        return true;
    }
    let warp = m.threads[wi].warp;
    m.threads[r].warp != warp
        || m.threads[r].cta != m.threads[wi].cta
        || (!m.warp_div(warp) && rd.1 >= m.div_clk.get(r).copied().unwrap_or(0))
}

/// Input words by allocation: (first, last) byte offset.
type Words = BTreeMap<u32, (u64, u64)>;

/// The provenance shadow of one run (`Machine::prov`). Not part of the state hash.
#[derive(Debug, Default)]
pub struct ProvState {
    /// The placeholder a look-ahead copy (`Machine::scratch`) carries instead of the shadow:
    /// its instructions record nothing.
    inert: bool,
    /// A shadow riding on another run (`tap_attach`) that passed its own cap ("time" /
    /// "memory") stops recording; the run goes on.
    detached: Option<&'static str>,
    /// That cap, and when the run started.
    cap: Option<(Budget, std::time::Instant)>,
    /// Instructions recorded (for checking the cap now and then).
    steps: u64,
    /// Registers: per thread (V, R, P) and per warp (UR, UP), packed `Prov`s (`Packed`).
    regs: Vec<[Vec<u64>; 3]>,
    uregs: Vec<[Vec<u64>; 2]>,
    /// Per thread / per warp: how many of those registers are not clean.
    dirty: Vec<u32>,
    wdirty: Vec<u32>,
    /// Memory bytes written during the run, packed `MByte`s in pages (`Packed`); a byte never
    /// written is its own input byte (global) or clean (shared).
    mem: FxMap<(u8, u64, u64), Box<[u64]>>,
    /// Per allocation bit: the written bytes whose provenance has it (`MByte::mask`).
    mem_bits: Vec<u64>,
    packed: Packed,
    /// Bytes read by each thread during the current instruction (by thread).
    acc: Vec<Vec<MByte>>,
    /// Asynchronous actors (TMA, bulk copies): the inputs of the bytes read since their last
    /// write.
    aacc: FxMap<u32, u64>,
    /// Per thread: the data its stores of the current instruction write (`WData`), valid
    /// when its generation is `gen` (the thread runs the current instruction), and whether its
    /// access is a read-modify-write (atomics).
    wd: Vec<(u32, bool, WData)>,
    /// The current instruction's generation (`wd`).
    gen: u32,
    /// Per instruction (by pc), what `pre` needs of it (`Info`).
    info: Vec<Option<Arc<Info>>>,
    /// Threads that spilled an input-derived value to local memory.
    local: HashSet<usize>,
    /// Some TMEM word holds input-derived data.
    tmem: u64,
    /// cp.async copies in flight: per shared destination (CTA, byte offset), the bytes the
    /// issue read from global memory. The copy lands at a wait (wait_group, the warp's
    /// DEPBAR, or an asynchronous unit after cp.async.mbarrier.arrive), which reads nothing
    /// itself: its write takes these bytes (`on_access`).
    cp: HashMap<(u64, u64), Vec<MByte>>,
    pending: Vec<(usize, Plan)>,
    sites: BTreeMap<u32, Site>,
    nth: usize,
    /// Bytes each actor read (not overwrote) during the current instruction: the data of
    /// stores whose value registers do not say it (cp.async, bulk copies).
    racc: Vec<u64>,
    /// Inputs any predicate register was derived from (a CTA-wide barrier reduction's result).
    pred_any: u64,
    /// Instructions some thread reached (guard true or false), by pc.
    reached: Vec<bool>,
    /// Conditional control: per branch / predicated instruction, the outcomes seen (bit 0:
    /// not taken / guard false, bit 1: taken / guard true) and the inputs the condition
    /// derives from.
    gates: Vec<Option<(u8, u64)>>,
    /// The current branch's condition inputs (`pre` to `post`).
    bra_mask: u64,
    /// Per global allocation, the run's first access to it (`first_access`, S-INIT scratch
    /// buffers).
    first: BTreeMap<u64, FirstAcc>,
    /// Control, address and synchronization operands (and guards of non-arithmetic
    /// instructions) derived from input data: per instruction, what (bit 0: guard, bit 1:
    /// operand), the inputs, and the input words.
    ctl: BTreeMap<u32, (u8, u64, Words)>,
    /// The concolic expressions (symb.rs, `--concolic`); None: off. Not copied into clones
    /// (trial executions on scratch machines).
    pub(crate) sym: crate::symb::SymSlot,
}

/// A copy of a machine carrying the shadow (a look-ahead: `Machine::scratch`, the
/// certificate's peek walks, the poll-loop trials) gets an inert placeholder: those copies run
/// instructions that are not part of the recorded run, and copying the shadow made each of
/// them cost its size. The run's own machine is never a copy (`explore::run_once_until` and
/// the certificate's run check that it keeps a live shadow; a run that ends with a placeholder
/// reports nothing: `report`).
impl Clone for ProvState {
    fn clone(&self) -> Self {
        if !crate::machine::prov_clone_full() {
            return ProvState {
                inert: true,
                ..Default::default()
            };
        }
        ProvState {
            inert: self.inert,
            detached: self.detached,
            cap: self.cap,
            steps: self.steps,
            regs: self.regs.clone(),
            uregs: self.uregs.clone(),
            dirty: self.dirty.clone(),
            wdirty: self.wdirty.clone(),
            mem: self.mem.clone(),
            mem_bits: self.mem_bits.clone(),
            packed: self.packed.clone(),
            acc: self.acc.clone(),
            aacc: self.aacc.clone(),
            cp: self.cp.clone(),
            wd: self.wd.clone(),
            gen: self.gen,
            info: self.info.clone(),
            local: self.local.clone(),
            tmem: self.tmem,
            pending: self.pending.clone(),
            sites: self.sites.clone(),
            nth: self.nth,
            racc: self.racc.clone(),
            pred_any: self.pred_any,
            reached: self.reached.clone(),
            gates: self.gates.clone(),
            bra_mask: self.bra_mask,
            first: self.first.clone(),
            ctl: self.ctl.clone(),
            sym: self.sym.clone(),
        }
    }
}

/// FxHash (the shadow's maps are keyed by small integers; SipHash dominated A's profile).
#[derive(Default, Clone, Copy)]
struct Fx(u64);

impl std::hash::Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u8(&mut self, i: u8) {
        self.write_u64(i as u64);
    }
    fn write_u16(&mut self, i: u16) {
        self.write_u64(i as u64);
    }
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64);
    }
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

type FxMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<Fx>>;

/// A register file's place in `ProvState::regs` (false) or `uregs` (true, per warp).
fn slot(f: File) -> (bool, usize) {
    match f {
        File::V => (false, 0),
        File::R => (false, 1),
        File::P => (false, 2),
        File::UR => (true, 0),
        File::UP => (true, 1),
    }
}

/// Bytes per memory page of the shadow.
const PAGE: u64 = 1024;

/// Provenance values packed into one word: 0 is `Prov::Clean` (registers) or "never written"
/// (memory); tag 1 (bits 62-63) a `Word` / `In` whose fields fit; tag 2 a mask of allocations
/// 0..61 (`Prov::Mixed`; `MByte`: mask + 1, 0 for `MByte::Clean`); tag 3 an index into the
/// table of the other values (masks naming allocation 62 or above, words with fields too
/// wide). Exact: every value unpacks to the one packed.
#[derive(Clone, Debug, Default)]
struct Packed {
    provs: Vec<Prov>,
    prov_ix: FxMap<Prov, u64>,
    bytes: Vec<MByte>,
    byte_ix: FxMap<MByte, u64>,
}

const TAG1: u64 = 1 << 62;
const TAG2: u64 = 2 << 62;
const TAG3: u64 = 3 << 62;
const LOW62: u64 = (1 << 62) - 1;

impl Packed {
    fn prov(&mut self, p: Prov) -> u64 {
        match p {
            Prov::Clean => 0,
            // alloc 14 bits, off 42, bytes 4, sext, atom.
            Prov::Word {
                alloc,
                off,
                bytes,
                sext,
                atom,
            } if alloc < 1 << 14 && off < 1 << 42 && bytes < 16 => {
                TAG1 | (alloc as u64) << 48
                    | off << 6
                    | (bytes as u64) << 2
                    | (sext as u64) << 1
                    | atom as u64
            }
            Prov::Mixed(m) if m <= LOW62 => TAG2 | m,
            p => {
                let n = self.provs.len() as u64;
                let i = *self.prov_ix.entry(p).or_insert(n);
                if i == n {
                    self.provs.push(p);
                }
                TAG3 | i
            }
        }
    }
    fn unprov(&self, w: u64) -> Prov {
        match w >> 62 {
            0 => Prov::Clean,
            1 => Prov::Word {
                alloc: ((w >> 48) & 0x3fff) as u32,
                off: (w >> 6) & ((1 << 42) - 1),
                bytes: ((w >> 2) & 0xf) as u8,
                sext: (w >> 1) & 1 != 0,
                atom: w & 1 != 0,
            },
            2 => Prov::Mixed(w & LOW62),
            _ => self.provs[(w & LOW62) as usize],
        }
    }
    fn byte(&mut self, b: MByte) -> u64 {
        match b {
            // alloc 18 bits, off 44.
            MByte::In(a, o) if a < 1 << 18 && o < 1 << 44 => TAG1 | (a as u64) << 44 | o,
            MByte::Clean => TAG2,
            MByte::Mask(m) if m < LOW62 => TAG2 | (m + 1),
            b => {
                let n = self.bytes.len() as u64;
                let i = *self.byte_ix.entry(b).or_insert(n);
                if i == n {
                    self.bytes.push(b);
                }
                TAG3 | i
            }
        }
    }
    fn unbyte(&self, w: u64) -> MByte {
        match w >> 62 {
            1 => MByte::In(((w >> 44) & 0x3ffff) as u32, w & ((1 << 44) - 1)),
            2 if w & LOW62 == 0 => MByte::Clean,
            2 => MByte::Mask((w & LOW62) - 1),
            _ => self.bytes[(w & LOW62) as usize],
        }
    }
}

fn loc_key(loc: Loc) -> (u8, u64, u64) {
    match loc {
        Loc::Shared { cta, off } => (0, cta as u64, off as u64),
        Loc::Global { alloc, off } => (1, alloc as u64, off),
        Loc::Param { .. } => (2, 0, 0),
    }
}

impl ProvState {
    fn reg(&self, m: &Machine, t: usize, r: Reg) -> Prov {
        if r.is_pt(m.k.front) {
            return Prov::Clean;
        }
        let w = match slot(r.file) {
            (false, i) => self.regs.get(t).map(|f| &f[i]),
            (true, i) => self.uregs.get(m.threads[t].warp as usize).map(|f| &f[i]),
        }
        .and_then(|v| v.get(r.idx as usize).copied())
        .unwrap_or(0);
        self.packed.unprov(w)
    }

    fn set_reg(&mut self, m: &Machine, t: usize, r: Reg, p: Prov) {
        let w = self.packed.prov(p);
        let (v, n) = match slot(r.file) {
            (false, i) => {
                if self.regs.len() <= t {
                    self.regs.resize_with(t + 1, Default::default);
                    self.dirty.resize(t + 1, 0);
                }
                (&mut self.regs[t][i], &mut self.dirty[t])
            }
            (true, i) => {
                let wi = m.threads[t].warp as usize;
                if self.uregs.len() <= wi {
                    self.uregs.resize_with(wi + 1, Default::default);
                    self.wdirty.resize(wi + 1, 0);
                }
                (&mut self.uregs[wi][i], &mut self.wdirty[wi])
            }
        };
        let i = r.idx as usize;
        if v.len() <= i {
            if w == 0 {
                return;
            }
            v.resize(i + 1, 0);
        }
        match (v[i] != 0, w != 0) {
            (false, true) => *n += 1,
            (true, false) => *n -= 1,
            _ => {}
        }
        v[i] = w;
    }

    /// The provenance of operand `v` of thread `t` (a register pair: the union).
    fn val(&self, m: &Machine, t: usize, v: &Val) -> Prov {
        match v {
            Val::Reg(r, n) if r.file != File::V && *n > 1 => {
                let mut p = self.reg(m, t, *r);
                for i in 1..*n as u16 {
                    let q = self.reg(
                        m,
                        t,
                        Reg {
                            file: r.file,
                            idx: r.idx + i,
                        },
                    );
                    if q.mask() != 0 {
                        p = Prov::of_mask(p.mask() | q.mask());
                    }
                }
                p
            }
            Val::Reg(r, _) => self.reg(m, t, *r),
            _ => Prov::Clean,
        }
    }

    /// `reg(..).mask()` without unpacking.
    fn reg_mask(&self, m: &Machine, t: usize, r: Reg) -> u64 {
        if matches!(r.file, File::P | File::UP) && r.idx == 7 {
            return 0;
        }
        let w = match slot(r.file) {
            (false, i) => self.regs.get(t).map(|f| &f[i]),
            (true, i) => self.uregs.get(m.threads[t].warp as usize).map(|f| &f[i]),
        }
        .and_then(|v| v.get(r.idx as usize).copied())
        .unwrap_or(0);
        match w >> 62 {
            0 => 0,
            1 => 1u64 << ((w >> 48) & 0x3fff).min(63),
            2 => w & LOW62,
            _ => self.packed.unprov(w).mask(),
        }
    }

    /// `val(..).mask()` without unpacking.
    fn val_mask(&self, m: &Machine, t: usize, v: &Val) -> u64 {
        match v {
            Val::Reg(r, n) if r.file != File::V && *n > 1 => (0..*n as u16).fold(0, |x, i| {
                x | self.reg_mask(
                    m,
                    t,
                    Reg {
                        file: r.file,
                        idx: r.idx + i,
                    },
                )
            }),
            Val::Reg(r, _) => self.reg_mask(m, t, *r),
            _ => 0,
        }
    }

    fn pred(&self, m: &Machine, t: usize, p: Pred) -> Prov {
        p.reg.map_or(Prov::Clean, |r| self.reg(m, t, r))
    }

    /// Bytes `off..off + n` of memory (space, id).
    fn mem_read(&self, (s, o, off): (u8, u64, u64), n: usize, out: &mut Vec<MByte>) {
        let end = off + n as u64;
        let mut a = off;
        while a < end {
            let pg = a / PAGE;
            let b = end.min((pg + 1) * PAGE);
            match self.mem.get(&(s, o, pg)) {
                Some(page) => out.extend((a..b).map(|x| match page[(x % PAGE) as usize] {
                    0 if s == 1 => MByte::In(o as u32, x),
                    0 => MByte::Clean,
                    w => self.packed.unbyte(w),
                })),
                None if s == 1 => out.extend((a..b).map(|x| MByte::In(o as u32, x))),
                None => out.extend((a..b).map(|_| MByte::Clean)),
            }
            a = b;
        }
    }

    /// Writes `data` at `off` of memory (space, id).
    fn mem_write(&mut self, (s, o, off): (u8, u64, u64), data: &[MByte]) {
        if self.mem_bits.is_empty() {
            self.mem_bits = vec![0; 64];
        }
        let mut i = 0usize;
        while i < data.len() {
            let a = off + i as u64;
            let pg = a / PAGE;
            let n = ((pg + 1) * PAGE - a).min((data.len() - i) as u64) as usize;
            let page = self
                .mem
                .entry((s, o, pg))
                .or_insert_with(|| vec![0u64; PAGE as usize].into_boxed_slice());
            for (j, &b) in data[i..i + n].iter().enumerate() {
                let x = ((a % PAGE) as usize) + j;
                let old = page[x];
                if old != 0 {
                    let mut k = self.packed.unbyte(old).mask();
                    while k != 0 {
                        self.mem_bits[k.trailing_zeros() as usize] -= 1;
                        k &= k - 1;
                    }
                }
                let mut k = b.mask();
                while k != 0 {
                    self.mem_bits[k.trailing_zeros() as usize] += 1;
                    k &= k - 1;
                }
                page[x] = self.packed.byte(b);
            }
            i += n;
        }
    }

    /// The inputs of every byte written during the run.
    fn mem_mask(&self) -> u64 {
        (0..self.mem_bits.len())
            .filter(|&i| self.mem_bits[i] != 0)
            .fold(0, |x, i| x | 1 << i)
    }

    /// A riding shadow's cap (`tap_attach`), checked every 4096 instructions (the resident
    /// set every 65536): past it, the shadow detaches.
    fn over_cap(&mut self) -> bool {
        let Some((b, t0)) = self.cap else {
            return false;
        };
        self.steps += 1;
        if !self.steps.is_multiple_of(4096) {
            return false;
        }
        if b.seconds.is_some_and(|s| t0.elapsed().as_secs_f64() > s) {
            self.detached = Some("time");
        } else if self.steps.is_multiple_of(65536)
            && b.rss.is_some_and(|r| crate::membudget::rss() > r)
        {
            self.detached = Some("memory");
        }
        if self.detached.is_some() {
            // What it holds is no longer needed (the report uses only the sites and
            // control records gathered so far).
            self.regs = Vec::new();
            self.uregs = Vec::new();
            self.mem = Default::default();
            self.acc = Vec::new();
            self.wd = Vec::new();
        }
        self.detached.is_some()
    }

    /// Per-thread vectors for `n` threads.
    fn grow(&mut self, n: usize) {
        if self.acc.len() < n {
            self.acc.resize_with(n, Vec::new);
            self.racc.resize(n, 0);
            self.wd.resize_with(n, Default::default);
        }
    }

    /// `Info` of instruction `pc`.
    fn info(&mut self, k: &ptxv_ir::Kernel, pc: u32) -> Arc<Info> {
        let i = pc as usize;
        if self.info.len() <= i {
            self.info.resize(k.code.len().max(i + 1), None);
        }
        self.info[i]
            .get_or_insert_with(|| {
                let op = &k.code[i].op;
                let pure = data_only(op);
                let ctl = if pure {
                    vec![]
                } else {
                    let vals = crate::num::value_regs(op);
                    op.src_regs()
                        .into_iter()
                        .filter(|r| !vals.contains(r))
                        .collect()
                };
                let collective = matches!(
                    op,
                    Op::Shfl { .. }
                        | Op::VoteAny { .. }
                        | Op::VoteAll { .. }
                        | Op::MatchAny { .. }
                        | Op::Redux { .. }
                        | Op::MovMatrix { .. }
                        | Op::Hmma { .. }
                        | Op::BarResult { .. }
                );
                Arc::new(Info {
                    pure,
                    ctl,
                    srcv: op.src_vals(),
                    srcr: op.src_regs(),
                    local: pure && !collective,
                })
            })
            .clone()
    }

    /// The outcomes seen at conditional control `pc` (`gates`).
    fn gate(&mut self, pc: u32) -> &mut (u8, u64) {
        let i = pc as usize;
        if self.gates.len() <= i {
            self.gates.resize(i + 1, None);
        }
        self.gates[i].get_or_insert((0, 0))
    }

    /// Instruction `pc` was reached.
    fn reach(&mut self, pc: u32) {
        let i = pc as usize;
        if self.reached.len() <= i {
            self.reached.resize(i + 1, false);
        }
        self.reached[i] = true;
    }

    fn is_reached(&self, pc: u32) -> bool {
        self.reached.get(pc as usize).copied().unwrap_or(false)
    }

    /// `tmem_block`: `actor` writes TMEM words. Every TMEM writer joins what it read into the
    /// TMEM shadow (one mask for all of TMEM): a thread's `tcgen05.st` its registers (at issue),
    /// an asynchronous unit (the MMA, `tcgen05.cp`) the bytes it read since its last write, and
    /// all input-derived shared memory when it read none that the shadow knows (never clean).
    /// Until 2026-11-08 only `tcgen05.st` and the MMA joined: `tcgen05.cp` of input data left
    /// the words clean and a branch on a `tcgen05.ld` of them was input-independent.
    pub(crate) fn on_tmem_write(&mut self, actor: u32) {
        if (actor as usize) < self.nth {
            return;
        }
        let read = self.aacc.get(&actor).copied().unwrap_or(0);
        let m = if read != 0 {
            read
        } else {
            self.mem
                .iter()
                .filter(|(k, _)| k.0 == 0)
                .flat_map(|(_, page)| page.iter())
                .filter(|&&w| w != 0)
                .fold(0, |x, &w| x | self.packed.unbyte(w).mask())
        };
        self.tmem |= m;
    }

    /// `access_x`: the bytes an actor reads (and, for a write, overwrites) and writes.
    /// `rmw`: the access is an atomic's (it reads the location too). `vc`: the actor's clock.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn on_access(
        &mut self,
        m: &Machine,
        actor: u32,
        loc: Loc,
        n: usize,
        write: bool,
        rmw: bool,
        vc: &crate::machine::VC,
    ) {
        if self.inert {
            return;
        }
        if let Some(sy) = self.sym.0.as_mut() {
            crate::symb::on_access(sy, actor, loc, n, write);
        }
        let (s, o, off) = loc_key(loc);
        if s == 1 {
            // An async actor (TMA, bulk copy or reduction) does not say whether it reads: its
            // accesses count as reads.
            let plain_write = write && (actor as usize) < self.nth && !rmw;
            match self.first.get_mut(&o) {
                None => {
                    let clock = crate::machine::vc_get(vc, actor);
                    self.first.insert(
                        o,
                        FirstAcc {
                            write: plain_write,
                            actor,
                            rd: (actor, clock),
                            late: false,
                        },
                    );
                }
                // A plain write the first read does not happen-before: in some order it
                // comes first.
                Some(f) if !f.write && plain_write && !read_orders(m, f.rd, actor, vc) => {
                    f.late = true;
                }
                Some(_) => {}
            }
        }
        if self.detached.is_some() || s == 2 {
            return;
        }
        let a = actor as usize;
        let mut read: Vec<MByte> = Vec::with_capacity(n);
        self.mem_read((s, o, off), n, &mut read);
        if a < self.nth {
            self.grow(a + 1);
            self.acc[a].extend(read.iter().copied());
        } else {
            *self.aacc.entry(actor).or_default() |= read.iter().fold(0, |x, b| x | b.mask());
        }
        if !write {
            if a < self.nth {
                self.racc[a] |= read.iter().fold(0, |x, b| x | b.mask());
            }
            return;
        }
        let live = a < self.nth && self.wd[a].0 == self.gen;
        let data: Vec<MByte> = match live.then(|| &self.wd[a]) {
            Some((_, _, WData::Bytes(d))) if d.len() == n => d.clone(),
            Some((_, _, WData::One(b))) if n == 1 => vec![*b],
            Some((_, rmw, d)) => {
                let mut m = d.mask();
                if *rmw {
                    m |= read.iter().fold(0, |x, b| x | b.mask());
                }
                m |= self.racc[a];
                vec![if m == 0 { MByte::Clean } else { MByte::Mask(m) }; n]
            }
            None => {
                // An asynchronous unit (TMA, bulk copy): the union of what it read.
                let m = if (actor as usize) < self.nth {
                    0
                } else {
                    self.aacc.get(&actor).copied().unwrap_or(0)
                };
                vec![if m == 0 { MByte::Clean } else { MByte::Mask(m) }; n]
            }
        };
        // A cp.async copy landing (`cp`): until 2026-11-08 its write had no data and no read
        // of its own, so the copied input bytes came out clean.
        let landed = if s == 0 {
            self.cp.remove(&(o, off)).filter(|c| c.len() == n)
        } else {
            None
        };
        let data: Vec<MByte> = data
            .into_iter()
            .enumerate()
            .take(n)
            .map(|(i, b)| match landed.as_ref().map(|c| c[i]) {
                Some(c) if b.mask() == 0 => c,
                Some(c) if c.mask() != 0 => MByte::Mask(b.mask() | c.mask()),
                _ => b,
            })
            .collect();
        self.mem_write((s, o, off), &data[..n.min(data.len())]);
        if actor as usize >= self.nth {
            self.aacc.remove(&actor);
        }
    }
}

/// A comparison observation: the example B can flip (if simple), why not (if not), the
/// input words read, and the input allocations the operands derive from.
type CmpObs = (
    Option<Example>,
    Option<String>,
    BTreeMap<u32, (u64, u64)>,
    u64,
);

fn simple_cmp(ps: &ProvState, m: &Machine, t: usize, op: &Op) -> CmpObs {
    let mut words = BTreeMap::new();
    let mut add = |p: Prov| {
        if let Prov::Word {
            alloc, off, bytes, ..
        } = p
        {
            let e = words.entry(alloc).or_insert((off, off + bytes as u64 - 1));
            e.0 = e.0.min(off);
            e.1 = e.1.max(off + bytes as u64 - 1);
        }
    };
    match op {
        Op::Setp {
            cmp,
            bits,
            a,
            b,
            and,
            or,
            xor,
            ex,
            ..
        } => {
            let (pa, pb) = (ps.val(m, t, a), ps.val(m, t, b));
            add(pa);
            add(pb);
            let allocs = pa.mask() | pb.mask() | ps.pred(m, t, *and).mask();
            let why = if ex.is_some() {
                Some("a 64-bit comparison (ISETP.EX): not a single-word comparison".to_string())
            } else if ps.pred(m, t, *and).mask() != 0 {
                Some("the combining predicate is input-derived".to_string())
            } else {
                match (pa, pb) {
                    (Prov::Word { .. }, Prov::Clean) | (Prov::Clean, Prov::Word { .. }) => None,
                    (Prov::Word { .. }, Prov::Word { .. }) => {
                        Some("both operands are input words".to_string())
                    }
                    _ => Some(
                        "an operand is derived from input data, not a single input word (arithmetic, an atomic result other than the old value of an add of a constant, an input-dependent address, a shuffle or TMEM)"
                            .to_string(),
                    ),
                }
            };
            if why.is_some() {
                return (None, why, words, allocs);
            }
            let word_is_a = matches!(pa, Prov::Word { .. });
            let (w, o) = if word_is_a { (pa, b) } else { (pb, a) };
            let Prov::Word {
                alloc,
                off,
                bytes,
                sext,
                atom,
            } = w
            else {
                unreachable!()
            };
            let other = m.rd(t, o).unwrap_or(0);
            let ex = Example {
                warp: m.threads[t].warp,
                alloc,
                off,
                bytes,
                sext,
                atom,
                word_is_a,
                other,
                cmp: *cmp,
                bits: *bits,
                and_val: m.prd(t, *and),
                or: *or,
                xor: *xor,
            };
            (Some(ex), None, words, allocs)
        }
        Op::Fsetp { a, b, c, .. } | Op::Dsetp { a, b, c, .. } => {
            let (pa, pb) = (ps.val(m, t, a), ps.val(m, t, b));
            add(pa);
            add(pb);
            (
                None,
                Some("a floating-point comparison: mutation not handled".to_string()),
                words,
                pa.mask() | pb.mask() | ps.pred(m, t, *c).mask(),
            )
        }
        _ => (None, None, words, 0),
    }
}

/// Before `pc` runs for `members` (`Machine::exec_inner`).
pub(crate) fn pre(m: &mut Machine, members: &[usize], pc: u32) {
    let Some(mut ps) = m.prov.take() else {
        return;
    };
    if ps.inert {
        m.prov = Some(ps);
        return;
    }
    if ps.detached.is_some() || ps.over_cap() {
        m.prov = Some(ps);
        return;
    }
    let k = m.k.clone();
    let inst = &k.code[pc as usize];
    ps.nth = m.threads.len();
    ps.grow(ps.nth);
    ps.pending.clear();
    // A new generation: no thread's store data of the last instruction is current.
    ps.gen = ps.gen.wrapping_add(1);
    if ps.gen == 0 {
        for w in &mut ps.wd {
            w.0 = 0;
        }
        ps.gen = 1;
    }
    let active: Vec<usize> = members
        .iter()
        .copied()
        .filter(|&t| m.prd(t, inst.guard))
        .collect();
    ps.reach(pc);
    if let Some(sy) = ps.sym.0.as_mut() {
        crate::symb::pre(sy, m, members, &active, pc);
    }
    let info = ps.info(&k, pc);
    let pure = info.pure;
    // Conditional control: a guarded instruction (or a branch with a second condition).
    let cond = match &inst.op {
        Op::Bra { cond, .. } => *cond,
        _ => None,
    };
    if let Op::Bra { div, .. } = &inst.op {
        // Sides from where the threads go (`post`); the condition's inputs here.
        ps.bra_mask = members.iter().fold(0, |x, &t| {
            x | ps.pred(m, t, inst.guard).mask()
                | cond.map_or(0, |c| ps.pred(m, t, c).mask())
                | div.as_ref().map_or(0, |d| ps.val(m, t, d).mask())
        });
    } else if inst.guard.reg.is_some() {
        let mut sides = 0u8;
        let mut mask = 0u64;
        for &t in members {
            let taken = m.prd(t, inst.guard) && cond.is_none_or(|c| m.prd(t, c));
            sides |= 1 << taken as u8;
            mask |= ps.pred(m, t, inst.guard).mask() | cond.map_or(0, |c| ps.pred(m, t, c).mask());
        }
        let g = ps.gate(pc);
        g.0 |= sides;
        g.1 |= mask;
    }
    if let Op::Brx { base, .. } = &inst.op {
        let mask = active.iter().fold(0, |x, &t| x | ps.val(m, t, base).mask());
        let g = ps.gate(pc);
        g.0 |= 2;
        g.1 |= mask;
    }
    // Control, address and synchronization operands of anything but arithmetic (the
    // operands `num::value_regs` does not name as stored / exchanged data), and the guards
    // of those instructions: input-derived ones make the run's accesses or synchronization
    // depend on the input.
    if !pure {
        let ctl = &info.ctl;
        for &t in members {
            let g = ps.pred(m, t, inst.guard);
            let c = cond.map_or(Prov::Clean, |c| ps.pred(m, t, c));
            let mut what = 0u8;
            let mut mask = 0u64;
            let mut words = BTreeMap::new();
            let mut add = |p: Prov, bit: u8| {
                if p.mask() == 0 {
                    return;
                }
                what |= bit;
                mask |= p.mask();
                if let Prov::Word {
                    alloc, off, bytes, ..
                } = p
                {
                    let e = words.entry(alloc).or_insert((off, off + bytes as u64 - 1));
                    e.0 = e.0.min(off);
                    e.1 = e.1.max(off + bytes as u64 - 1);
                }
            };
            add(g, 1);
            add(c, 1);
            if m.prd(t, inst.guard) {
                for &r in ctl {
                    add(ps.reg(m, t, r), 2);
                }
            }
            if what != 0 {
                let e = ps.ctl.entry(pc).or_insert((0, 0, BTreeMap::new()));
                e.0 |= what;
                e.1 |= mask;
                for (a, (lo, hi)) in words {
                    let w = e.2.entry(a).or_insert((lo, hi));
                    w.0 = w.0.min(lo);
                    w.1 = w.1.max(hi);
                }
            }
        }
    }
    let mut coll: Option<u64> = None;
    for &t in &active {
        // Register arithmetic of a thread none of whose registers (nor its warp's uniform
        // ones) holds input-derived data leaves every destination clean, as it found it:
        // nothing to record.
        if info.local
            && ps.dirty.get(t).is_none_or(|&n| n == 0)
            && ps
                .wdirty
                .get(m.threads[t].warp as usize)
                .is_none_or(|&n| n == 0)
        {
            continue;
        }
        ps.acc[t].clear();
        ps.racc[t] = 0;
        let guard = ps.pred(m, t, inst.guard).mask();
        // Every register read, each element of a PTX vector operand included (`src_regs`).
        let srcs = info
            .srcr
            .iter()
            .fold(guard, |x, &r| x | ps.reg_mask(m, t, r));
        // Memory written by an instruction other than st / atom (stmatrix, st.v256, cp.async,
        // bulk copies issued by the thread): its registers, plus what it read (`racc`).
        ps.wd[t] = (ps.gen, false, WData::One(MByte::Mask(srcs)));
        let plan = match &inst.op {
            Op::Ld {
                addr, sext, bytes, ..
            } => {
                let a = [addr.base.clone(), addr.base2.clone()]
                    .iter()
                    .flatten()
                    .fold(guard, |x, v| x | ps.val(m, t, v).mask());
                Plan::Load {
                    addr: a,
                    sext: *sext,
                    bytes: *bytes,
                    atomic: false,
                    extra: 0,
                }
            }
            Op::Atom { op, bytes, v, .. } => {
                ps.wd[t] = (ps.gen, true, WData::One(MByte::Mask(srcs)));
                // atom.add of a non-negative constant: the old value is at least the word's
                // initial value (`Prov::Word::atom`).
                let x = m.rd(t, v).unwrap_or(u64::MAX);
                let nonneg = if *bytes == 8 {
                    (x as i64) >= 0
                } else {
                    (x as u32 as i32) >= 0
                };
                let add_const =
                    *op == ptxv_ir::AtomOp::Add && ps.val(m, t, v).mask() == 0 && nonneg;
                Plan::Load {
                    addr: 0,
                    sext: false,
                    bytes: *bytes,
                    atomic: !add_const,
                    extra: if add_const { 1 << 62 } else { srcs },
                }
            }
            Op::St { v, bytes, .. } => {
                // PTX `st.vN`: one virtual register per element.
                let p = match v {
                    Val::Reg(r, n) if r.file == File::V && *n >= 2 => {
                        Prov::of_mask((0..*n as u16).fold(0, |x, i| {
                            x | ps
                                .reg(
                                    m,
                                    t,
                                    Reg {
                                        file: r.file,
                                        idx: r.idx + i,
                                    },
                                )
                                .mask()
                        }))
                    }
                    _ => ps.val(m, t, v),
                };
                let d: Vec<MByte> = match p {
                    Prov::Word {
                        alloc,
                        off,
                        bytes: wb,
                        atom: false,
                        ..
                    } if wb == *bytes && guard == 0 => (0..*bytes as u64)
                        .map(|i| MByte::In(alloc, off + i))
                        .collect(),
                    _ => {
                        let mm = p.mask() | guard;
                        vec![if mm == 0 {
                            MByte::Clean
                        } else {
                            MByte::Mask(mm)
                        }]
                    }
                };
                ps.wd[t].2 = if d.len() == 1 {
                    WData::One(d[0])
                } else {
                    WData::Bytes(d)
                };
                Plan::All(Prov::Clean)
            }
            Op::StLocal { .. } => {
                if srcs != 0 {
                    ps.local.insert(t);
                }
                Plan::All(Prov::Clean)
            }
            Op::LdLocal { .. } => Plan::All(if ps.local.contains(&t) {
                Prov::Mixed(1 << 63)
            } else {
                Prov::Clean
            }),
            Op::TmemSt { .. } => {
                ps.tmem |= srcs;
                Plan::All(Prov::Clean)
            }
            Op::TmemLd { .. } => Plan::All(Prov::of_mask(ps.tmem | guard)),
            // The copy's shared-memory reads join at its completion (`on_tmem_write`); its
            // operands (TMEM address, descriptor) at issue.
            Op::TcCp { .. } => {
                ps.tmem |= srcs;
                Plan::All(Prov::Clean)
            }
            Op::Mma { .. } => {
                // The MMA reads shared memory and TMEM: input-derived if any of either is.
                let sm = ps.mem_mask();
                ps.tmem |= sm | srcs;
                Plan::All(Prov::Clean)
            }
            Op::Mov { a, .. } if guard == 0 => Plan::All(ps.val(m, t, a)),
            Op::CvtInt {
                a,
                from_bits,
                signed,
                ..
            } if guard == 0 => {
                let p = ps.val(m, t, a);
                let fb = *from_bits as u32;
                Plan::All(match p {
                    // Wider source: the word and its extension pass through.
                    Prov::Word { bytes, .. } if fb > 8 * bytes as u32 => p,
                    // The word's width or narrower (whole bytes): its low bytes, extended as
                    // the conversion says.
                    Prov::Word {
                        alloc,
                        off,
                        bytes,
                        atom: false,
                        ..
                    } if fb.is_multiple_of(8) && fb >= 8 && fb <= 8 * bytes as u32 => Prov::Word {
                        alloc,
                        off,
                        bytes: (fb / 8) as u8,
                        sext: *signed,
                        atom: false,
                    },
                    p => Prov::of_mask(p.mask()),
                })
            }
            // Byte extraction from a packed input word (ptxas loads eight u8 flags with one
            // LDG.E.64 and takes each with `PRMT Rd, Ra, 0x777k, RZ`).
            Op::Prmt { a, sel, b, .. } if guard == 0 => {
                let (pa, pb, psel) = (ps.val(m, t, a), ps.val(m, t, b), ps.val(m, t, sel));
                let r = if psel.mask() == 0 && pb.mask() == 0 {
                    prmt_word(
                        pa,
                        m.rd(t, sel).unwrap_or(0) as u32,
                        m.rd(t, b).unwrap_or(0) as u32,
                    )
                } else {
                    None
                };
                Plan::All(r.unwrap_or(Prov::of_mask(pa.mask() | pb.mask() | psel.mask())))
            }
            Op::Bfe {
                a,
                pos,
                len,
                signed,
                ..
            } if guard == 0 => {
                let pa = ps.val(m, t, a);
                let known = ps.val(m, t, pos).mask() == 0 && ps.val(m, t, len).mask() == 0;
                let (po, ln) = (
                    m.rd(t, pos).unwrap_or(0) as u32 & 0xff,
                    m.rd(t, len).unwrap_or(0) as u32 & 0xff,
                );
                Plan::All(match pa {
                    Prov::Word {
                        alloc,
                        off,
                        bytes,
                        atom: false,
                        ..
                    } if known
                        && po % 8 == 0
                        && ln % 8 == 0
                        && ln > 0
                        && (po + ln) <= 8 * bytes as u32 =>
                    {
                        Prov::Word {
                            alloc,
                            off: off + (po / 8) as u64,
                            bytes: (ln / 8) as u8,
                            sext: *signed,
                            atom: false,
                        }
                    }
                    p => Prov::of_mask(p.mask() | srcs),
                })
            }
            Op::Int { op, a, b, .. } if guard == 0 => {
                let (pa, pb) = (ps.val(m, t, a), ps.val(m, t, b));
                let r = match (op, pa, pb) {
                    // word & 0xff.. (low bytes), word >> 8k (zero-extended: the bytes above).
                    (ptxv_ir::IntOp::And, Prov::Word { .. }, Prov::Clean) => {
                        low_bytes(pa, m.rd(t, b).unwrap_or(0))
                    }
                    (ptxv_ir::IntOp::And, Prov::Clean, Prov::Word { .. }) => {
                        low_bytes(pb, m.rd(t, a).unwrap_or(0))
                    }
                    (
                        ptxv_ir::IntOp::ShrU,
                        Prov::Word {
                            alloc,
                            off,
                            bytes,
                            sext: false,
                            atom: false,
                        },
                        Prov::Clean,
                    ) => {
                        let sh = m.rd(t, b).unwrap_or(0);
                        (sh.is_multiple_of(8) && sh > 0 && sh / 8 < bytes as u64).then(|| {
                            Prov::Word {
                                alloc,
                                off: off + sh / 8,
                                bytes: bytes - (sh / 8) as u8,
                                sext: false,
                                atom: false,
                            }
                        })
                    }
                    _ => None,
                };
                Plan::All(r.unwrap_or(Prov::of_mask(srcs)))
            }
            Op::Sel { a, b, p, .. } | Op::Fsel { a, b, p, .. } if guard == 0 => {
                let pp = ps.pred(m, t, *p);
                let chosen = if m.prd(t, *p) { a } else { b };
                let c = ps.val(m, t, chosen);
                Plan::All(if pp.mask() == 0 {
                    c
                } else {
                    Prov::of_mask(c.mask() | pp.mask())
                })
            }
            Op::Shfl { .. }
            | Op::VoteAny { .. }
            | Op::VoteAll { .. }
            | Op::MatchAny { .. }
            | Op::Redux { .. }
            | Op::MovMatrix { .. }
            | Op::Hmma { .. } => {
                // Warp-collective: every lane's result depends on every lane's operands
                // (the same union for every lane: computed once).
                let all = *coll.get_or_insert_with(|| {
                    active.iter().fold(0, |x, &u| {
                        x | info.srcv.iter().fold(0, |y, v| y | ps.val_mask(m, u, v))
                    })
                });
                Plan::All(Prov::of_mask(all | guard))
            }
            Op::Setp { d, .. } | Op::Fsetp { d, .. } | Op::Dsetp { d, .. } => {
                let (ex, why, words, allocs) = simple_cmp(&ps, m, t, &inst.op);
                Plan::Cmp(Box::new((*d, Prov::of_mask(srcs), ex, why, words, allocs)))
            }
            // A CTA-wide barrier reduction of predicates.
            Op::BarResult { .. } => Plan::All(Prov::of_mask(srcs | ps.pred_any)),
            _ => Plan::All(Prov::of_mask(srcs)),
        };
        ps.pending.push((t, plan));
    }
    m.prov = Some(ps);
}

/// The operation's name (`Op` variant), for refusal reasons.
fn op_name(op: &Op) -> String {
    let d = format!("{op:?}");
    d.split([' ', '{', '(']).next().unwrap_or("").to_string()
}

/// Register-only arithmetic (`num::pure_op`, and the f64 / conversion / warp-MMA operations
/// it leaves out): every source is a value.
fn data_only(op: &Op) -> bool {
    crate::num::pure_op(op)
        || matches!(
            op,
            Op::Dsetp { .. }
                | Op::Dfma { .. }
                | Op::DsetpMinMax { .. }
                | Op::Rsqrt64Approx { .. }
                | Op::DivSqrt64 { .. }
                | Op::FCvt { .. }
                | Op::F2F { .. }
                | Op::DivFull { .. }
                | Op::Hset2 { .. }
                | Op::Dp2a { .. }
                | Op::Hmma { .. }
                | Op::MovMatrix { .. }
        )
}

/// `word & mask` for a mask of whole low bytes (0xff, 0xffff, 0xffffff): the word's low bytes,
/// zero-extended.
fn low_bytes(p: Prov, mask: u64) -> Option<Prov> {
    let Prov::Word {
        alloc,
        off,
        bytes,
        atom: false,
        ..
    } = p
    else {
        return None;
    };
    let n = match mask {
        0xff => 1u8,
        0xffff => 2,
        0xff_ffff => 3,
        _ => return None,
    };
    (n <= bytes).then_some(Prov::Word {
        alloc,
        off,
        bytes: n,
        sext: false,
        atom: false,
    })
}

/// PRMT (`prmt.b32 d, a, sel, b`) of an input word `a` with a constant selector and a constant
/// `b`: result byte i is byte (sel nibble i & 7) of {a, b} (bit 3: that byte's sign
/// replicated). A run of consecutive bytes of the word from byte 0 of the result, the rest
/// zero (or the run's sign), is a narrower word.
fn prmt_word(pa: Prov, sel: u32, bv: u32) -> Option<Prov> {
    let Prov::Word {
        alloc,
        off,
        bytes,
        sext,
        atom: false,
    } = pa
    else {
        return None;
    };
    let nib = |i: u32| (sel >> (4 * i)) & 0xf;
    let k = nib(0);
    if k & 8 != 0 || k >= bytes as u32 {
        return None;
    }
    let mut len = 1u32;
    while len < 4 && nib(len) == k + len && k + len < bytes as u32 {
        len += 1;
    }
    // The remaining result bytes: zero (b's zero bytes, or a's extension bytes when a is
    // zero-extended) or the sign of the run's top byte.
    let zero_byte = |n: u32| {
        if n & 8 != 0 {
            return false;
        }
        let n = n & 7;
        if n >= 4 {
            (bv >> (8 * (n - 4))) & 0xff == 0
        } else {
            n >= bytes as u32 && !sext
        }
    };
    let top = k + len - 1;
    let all_zero = (len..4).all(|i| zero_byte(nib(i)));
    let all_sign = (len..4).all(|i| nib(i) == (top | 8));
    if all_zero {
        Some(Prov::Word {
            alloc,
            off: off + k as u64,
            bytes: len as u8,
            sext: false,
            atom: false,
        })
    } else if all_sign && len < 4 {
        Some(Prov::Word {
            alloc,
            off: off + k as u64,
            bytes: len as u8,
            sext: true,
            atom: false,
        })
    } else {
        None
    }
}

fn word_from(bytes: &[MByte], sext: bool) -> Option<Prov> {
    let MByte::In(a, o) = *bytes.first()? else {
        return None;
    };
    for (i, b) in bytes.iter().enumerate() {
        if *b != MByte::In(a, o + i as u64) {
            return None;
        }
    }
    Some(Prov::Word {
        alloc: a,
        off: o,
        bytes: bytes.len() as u8,
        sext,
        atom: false,
    })
}

/// After `pc` ran (`Machine::exec_inner`).
pub(crate) fn post(m: &mut Machine, members: &[usize], pc: u32) {
    let Some(mut ps) = m.prov.take() else {
        return;
    };
    if ps.inert {
        m.prov = Some(ps);
        return;
    }
    if ps.detached.is_some() {
        m.prov = Some(ps);
        return;
    }
    let k = m.k.clone();
    let inst = &k.code[pc as usize];
    // A branch's sides: where its threads went. Unconditional ones are not gates.
    if let Op::Bra {
        target,
        any,
        tid0_falls,
        cond,
        div,
    } = &inst.op
    {
        if inst.guard.reg.is_some() || cond.is_some() || div.is_some() || *any || *tid0_falls {
            let mut sides = 0u8;
            for &t in members {
                let taken = m.threads[t].pc as usize == *target && *target != pc as usize + 1;
                sides |= 1 << taken as u8;
            }
            let bm = ps.bra_mask;
            let g = ps.gate(pc);
            g.0 |= sides;
            g.1 |= bm;
        }
    }
    // cp.async: the bytes the issue read, for the write that lands them (`ProvState::cp`).
    if let Op::CpAsync { .. } = inst.op {
        for &t in members.iter().filter(|&&t| m.prd(t, inst.guard)) {
            let th = &m.threads[t];
            if let Some(c) = th
                .cp_pending
                .iter()
                .rev()
                .find(|c| c.pc == pc && c.src.is_some())
            {
                let read = ps.acc.get(t).cloned().unwrap_or_default();
                if read.len() == c.data.len() {
                    ps.cp.insert((th.cta as u64, c.off as u64), read);
                } else {
                    // Not one read per copied byte: every byte gets all of it.
                    let mk = read.iter().fold(0, |x, b| x | b.mask());
                    let b = if mk == 0 {
                        MByte::Clean
                    } else {
                        MByte::Mask(mk)
                    };
                    ps.cp
                        .insert((th.cta as u64, c.off as u64), vec![b; c.data.len()]);
                }
            }
        }
    }
    let dsts = inst.op.dst_regs();
    // (Its buffer goes back after: `pre` refills it for every instruction.)
    let mut pending = std::mem::take(&mut ps.pending);
    for (t, plan) in pending.drain(..) {
        match plan {
            Plan::All(p) => {
                // Registers loaded from memory by an instruction not handled above (ldmatrix,
                // ld.v256, multimem ld_reduce): what it read.
                let r = ps.racc.get(t).copied().unwrap_or(0);
                let p = if r == 0 {
                    p
                } else {
                    Prov::of_mask(p.mask() | r)
                };
                for &d in &dsts {
                    if matches!(d.file, File::P | File::UP) {
                        ps.pred_any |= p.mask();
                        // A predicate from input data written by a test other than a
                        // comparison (FCHK's division slow-path test, LOP3's bit test, R2P,
                        // a vote, a packed half comparison): a site B does not flip.
                        // Carry-outs of address arithmetic (LEA, IADD3, IMAD) and predicate
                        // logic of other sites' results (PLOP3) are not sites.
                        let test = matches!(
                            inst.op,
                            Op::Fchk { .. }
                                | Op::Lop3 { .. }
                                | Op::R2p { .. }
                                | Op::VoteAll { .. }
                                | Op::VoteAny { .. }
                                | Op::MatchAny { .. }
                                | Op::Hsetp2Bf16 { .. }
                        );
                        if test && p.mask() != 0 && !d.is_pt(m.k.front) {
                            let v = m.prd(
                                t,
                                Pred {
                                    reg: Some(d),
                                    neg: false,
                                },
                            );
                            let s = ps.sites.entry(pc).or_default();
                            s.seen |= 1 << v as u8;
                            s.n[v as usize] += 1;
                            s.allocs |= p.mask();
                            if s.refuse.is_none() {
                                s.refuse = Some(format!(
                                    "the predicate is not a comparison ({})",
                                    op_name(&inst.op)
                                ));
                            }
                        }
                    }
                    ps.set_reg(m, t, d, p);
                }
            }
            Plan::Load {
                addr,
                sext,
                bytes,
                atomic,
                extra,
            } => {
                // (The next instruction of `t` starts its list anew: `pre`.)
                let acc = std::mem::take(&mut ps.acc[t]);
                let all = acc.iter().fold(0, |x, b| x | b.mask());
                let n = dsts.len().max(1);
                // Each destination register gets its share of the bytes.
                let w = (bytes as usize / n).max(1);
                // An input-dependent address still reads one input word: B mutates that word and
                // leaves the address's inputs alone. A load of bytes the shadow does not track
                // (constant / parameter space: a lookup table) carries its address's inputs;
                // until 2026-11-08 it came out clean.
                let untracked = acc.is_empty() && addr != 0;
                for (i, &d) in dsts.iter().enumerate() {
                    let p = if untracked {
                        Prov::of_mask(addr)
                    } else if atomic {
                        Prov::of_mask(all | extra)
                    } else {
                        let part = acc.get(i * w..(i + 1) * w).unwrap_or(&[]);
                        match word_from(part, sext && w < 4) {
                            Some(Prov::Word {
                                alloc,
                                off,
                                bytes,
                                sext,
                                ..
                            }) => Prov::Word {
                                alloc,
                                off,
                                bytes,
                                sext,
                                atom: extra == 1 << 62,
                            },
                            Some(p) => p,
                            None => Prov::of_mask(part.iter().fold(0, |x, b| x | b.mask())),
                        }
                    };
                    ps.set_reg(m, t, d, p);
                }
                ps.acc[t] = acc;
            }
            Plan::Cmp(c) => {
                let (d, p, ex, why, words, allocs) = *c;
                for &dd in &dsts {
                    ps.set_reg(m, t, dd, p);
                }
                ps.pred_any |= p.mask();
                if allocs == 0 {
                    continue;
                }
                let v = m.prd(
                    t,
                    Pred {
                        reg: Some(d),
                        neg: false,
                    },
                );
                let s = ps.sites.entry(pc).or_default();
                s.seen |= 1 << v as u8;
                s.n[v as usize] += 1;
                s.allocs |= allocs;
                for (a, (lo, hi)) in words {
                    let e = s.words.entry(a).or_insert((lo, hi));
                    e.0 = e.0.min(lo);
                    e.1 = e.1.max(hi);
                }
                if s.refuse.is_none() {
                    s.refuse = why;
                }
                if let Some(e) = ex {
                    if s.examples[v as usize].len() < 4096 {
                        s.examples[v as usize].push(e);
                    }
                }
            }
        }
    }
    ps.pending = pending;
    if let Some(mut sy) = ps.sym.0.take() {
        crate::symb::post(&mut sy, m, pc, &|t, r| ps.reg(m, t, r).mask() == 0);
        ps.sym.0 = Some(sy);
    }
    // Threads that exited: their registers are never read again (a warp-collective operation
    // reads its active lanes only); their shadow goes.
    if matches!(inst.op, Op::Exit | Op::Ret { .. }) {
        for &t in members {
            if m.threads[t].st == crate::machine::St::Exited {
                if let Some(f) = ps.regs.get_mut(t) {
                    *f = Default::default();
                    ps.dirty[t] = 0;
                }
                if let Some(a) = ps.acc.get_mut(t) {
                    *a = Vec::new();
                }
            }
        }
    }
    m.prov = Some(ps);
}

/// An input-dependent comparison side that no thread took.
#[derive(Clone, Debug, Serialize)]
pub struct Untaken {
    pub pc: u32,
    /// The instruction (`Kernel::describe`).
    pub at: String,
    /// The predicate value never produced.
    pub never: bool,
    /// The input buffers (byte ranges for single words) the operands derive from.
    pub sources: String,
    /// `sources` as data: single-word inputs first (with their byte range), then buffers the
    /// operands derive from otherwise (`bytes` null; `<local/TMEM>` for input-derived local
    /// or TMEM data).
    pub inputs: Vec<InputRef>,
    /// The note's text.
    pub note: String,
    /// Why B cannot flip it (None: some thread's comparison read one input word).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refuse: Option<String>,
}

/// An input buffer an untaken comparison's operands derive from.
#[derive(Clone, Debug, Serialize)]
pub struct InputRef {
    pub buffer: String,
    /// `[first, last]` byte offsets read as single input words (null: derived otherwise).
    pub bytes: Option<[u64; 2]>,
}

/// An input-dependent comparison and its outcomes in the run.
#[derive(Clone, Debug, Serialize)]
pub struct Comparison {
    pub pc: u32,
    pub at: String,
    pub n_true: u64,
    pub n_false: u64,
    pub inputs: Vec<InputRef>,
}

/// An instruction whose control, address or synchronization depended on input data.
#[derive(Clone, Debug, Serialize)]
pub struct InputDep {
    pub pc: u32,
    pub at: String,
    /// "guard", "operand" or "guard, operand".
    pub what: String,
    pub inputs: Vec<InputRef>,
}

/// Consecutive instructions no thread reached.
#[derive(Clone, Debug, Serialize)]
pub struct Gap {
    pub first: u32,
    pub last: u32,
    pub first_at: String,
    pub last_at: String,
    /// Source locations (PTX line, SASS address) of `first` / `last`, and the kernel of a
    /// concatenated launch they belong to.
    pub first_loc: u64,
    pub last_loc: u64,
    pub kernel: Option<String>,
    pub instructions: usize,
    /// The reached branch or predicated instruction whose untaken side leads here without
    /// passing a reached instruction (None: none found, e.g. code after an unconditional
    /// exit or a function never called).
    pub gate_pc: Option<u32>,
    pub gate_at: Option<String>,
    /// "input" (the gate's condition derives from input data), "parameter" (it does not),
    /// "divergence" (BRA.DIV / BRA.CONV: the collective fallback of a warp that is not
    /// converged), "none".
    pub gate: String,
}

/// The result of A on one case.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Coverage {
    /// Comparisons of input-derived values that ran.
    pub sites: usize,
    pub untaken: Vec<Untaken>,
    /// Every input-dependent comparison: thread observations per outcome.
    pub comparisons: Vec<Comparison>,
    /// Instructions whose guard (non-arithmetic instructions), address, control or
    /// synchronization operand derived from input data in the run.
    pub input_dependent: Vec<InputDep>,
    /// The run finished and `input_dependent` is empty: every branch, guard, address and
    /// synchronization operand was free of input data, so another input of this shape with
    /// these parameters takes the same control path and makes the same accesses (one
    /// schedule's provenance).
    pub input_independent: bool,
    /// Instructions no thread reached, as runs of consecutive instructions with the
    /// conditional control in front of them.
    pub unexecuted: Vec<Gap>,
    pub unexecuted_instructions: usize,
    /// Of those, behind a condition derived from input data (`gate` = "input") and behind
    /// a condition free of it (parameters, special registers: "parameter").
    pub behind_input: usize,
    pub behind_parameter: usize,
    /// The run reached a terminal state.
    pub terminal: bool,
    /// The run stopped at A's own budget (`Budget`): "time" or "memory". Then only
    /// `input_dependent` and the sites seen are reported (each holds for the part that ran);
    /// no side is claimed untaken, no code unexecuted, and the run is not input-independent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stopped: Option<String>,
    pub seconds: f64,
    /// `--concolic` (solve.rs): the outcome per untaken side and the per-solver counts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concolic: Option<crate::solve::Report>,
    #[serde(skip)]
    examples: BTreeMap<u32, [Vec<Example>; 2]>,
    /// Instructions reached, by pc (`mutated_pass`).
    #[serde(skip)]
    reached: Vec<bool>,
}

fn inputs(m: &Machine, s: &Site) -> Vec<InputRef> {
    inputs_of(m, &s.words, s.allocs)
}

fn inputs_of(m: &Machine, words: &BTreeMap<u32, (u64, u64)>, allocs: u64) -> Vec<InputRef> {
    let name = |a: u32| {
        m.launch
            .allocs
            .get(a as usize)
            .map_or("?".to_string(), |x| x.name.clone())
    };
    let mut v: Vec<InputRef> = words
        .iter()
        .map(|(a, (lo, hi))| InputRef {
            buffer: name(*a),
            bytes: Some([*lo, *hi]),
        })
        .collect();
    for a in 0..64u32 {
        if allocs >> a & 1 == 1 && !words.contains_key(&a) {
            v.push(InputRef {
                buffer: if a == 63 {
                    "<local/TMEM>".to_string()
                } else {
                    name(a)
                },
                bytes: None,
            });
        }
    }
    v
}

fn sources(v: &[InputRef]) -> String {
    v.iter()
        .map(|r| match (r.buffer.as_str(), r.bytes) {
            ("<local/TMEM>", _) => "input-derived local / TMEM data".to_string(),
            (b, Some([lo, hi])) => format!("{b}[bytes {lo:#x}..{hi:#x}]"),
            (b, None) => b.to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// A's own caps (`analyze_budget`): wall seconds, and the process's resident set in bytes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Budget {
    pub seconds: Option<f64>,
    pub rss: Option<u64>,
}

/// A: one deterministic run of `m0` with the provenance shadow; the input-dependent
/// comparison sides it never took.
pub fn analyze(m0: &Machine) -> Coverage {
    analyze_budget(m0, Budget::default())
}

/// `analyze` within `b`: the run stops (`Coverage::stopped`) when it passes either cap.
pub fn analyze_budget(m0: &Machine, b: Budget) -> Coverage {
    analyze_sym(m0, b, None).0
}

/// A, with the concolic expressions when `sym` is given (symb.rs): the coverage, the
/// expressions and the run's final machine (for `symb::Evaluator`).
pub fn analyze_with(
    m0: &Machine,
    sym: Option<crate::symb::Config>,
) -> (Coverage, Option<(Box<crate::symb::SymState>, Machine)>) {
    analyze_sym(m0, Budget::default(), sym)
}

/// `analyze_with` within `b`.
pub fn analyze_sym(
    m0: &Machine,
    b: Budget,
    sym: Option<crate::symb::Config>,
) -> (Coverage, Option<(Box<crate::symb::SymState>, Machine)>) {
    let t0 = std::time::Instant::now();
    let mut m = m0.clone();
    let nth = m.threads.len();
    m.prov = Some(Box::new(ProvState {
        nth,
        sym: crate::symb::SymSlot(sym.map(|c| Box::new(crate::symb::SymState::new(c, nth)))),
        ..Default::default()
    }));
    let mut why: Option<&'static str> = None;
    let mut last_rss = t0;
    let (mut m, _, ok, stopped) = crate::explore::run_once_until(m, &mut |_| {
        let now = std::time::Instant::now();
        if b.seconds.is_some_and(|s| (now - t0).as_secs_f64() > s) {
            why = Some("time");
            return true;
        }
        // The resident set at most every 100 ms (a /proc read).
        if let Some(r) = b.rss {
            if (now - last_rss).as_millis() >= 100 {
                last_rss = now;
                if crate::membudget::rss() > r {
                    why = Some("memory");
                    return true;
                }
            }
        }
        false
    });
    let mut ps = *m.prov.take().expect("provenance shadow");
    let sym_out = ps.sym.0.take();
    let c = report(
        &m,
        ps,
        ok,
        stopped.then(|| why.unwrap_or("time").to_string()),
        t0,
    );
    (c, sym_out.map(|s| (s, m)))
}

/// The coverage report of a finished (or stopped) provenance run: `m` is its final machine.
fn report(
    m: &Machine,
    ps: ProvState,
    ok: bool,
    stopped: Option<String>,
    t0: std::time::Instant,
) -> Coverage {
    // A copy adopted as the run's state records nothing from there on: no claims at all.
    if ps.inert {
        eprintln!(
            "input coverage: the run ended on a copy of its machine (placeholder shadow); nothing reported"
        );
        // Tests: a run must never continue on a copy (`live_shadow`).
        if cfg!(debug_assertions) || STRICT.get() {
            panic!("input coverage: the run ended on a copy of its machine");
        }
    }
    if std::env::var_os("PTXV_COVERAGE_STATS").is_some() {
        let regs: usize = ps
            .regs
            .iter()
            .map(|f| f.iter().map(|v| v.capacity()).sum::<usize>())
            .sum();
        let peak_len: usize = ps
            .regs
            .iter()
            .map(|f| f.iter().map(|v| v.len()).max().unwrap_or(0))
            .max()
            .unwrap_or(0);
        eprintln!(
            "coverage stats: {} threads, {} register words (longest file {}), {} memory pages, {} acc bytes, {} interned provs, {} interned bytes",
            ps.regs.len(),
            regs,
            peak_len,
            ps.mem.len(),
            ps.acc.iter().map(|a| a.capacity()).sum::<usize>(),
            ps.packed.provs.len(),
            ps.packed.bytes.len()
        );
    }
    let stopped = stopped
        .or_else(|| ps.detached.map(String::from))
        .or_else(|| ps.inert.then(|| "copy".to_string()));
    let ok = ok && stopped.is_none();
    let mut c = Coverage {
        sites: ps.sites.len(),
        terminal: ok,
        stopped: stopped.clone(),
        ..Default::default()
    };
    let stopped = stopped.is_some();
    for (&pc, s) in &ps.sites {
        // A stopped run claims no side untaken: the rest of the run may take it.
        if s.seen == 3 || stopped {
            continue;
        }
        let never = s.seen == 1; // only false seen: true never
        let flippable = !s.examples[!never as usize].is_empty();
        let at = m.k.describe(pc as usize);
        let inp = inputs(m, s);
        let src = sources(&inp);
        c.untaken.push(Untaken {
            pc,
            note: format!(
                "input-coverage: comparison at {at} (predicate from {src}) never {}: that side was not exercised on this input (one schedule)",
                if never { "true" } else { "false" }
            ),
            at,
            never,
            sources: src,
            inputs: inp,
            refuse: if flippable { None } else { s.refuse.clone() },
        });
        c.examples.insert(pc, s.examples.clone());
    }
    for (&pc, s) in &ps.sites {
        c.comparisons.push(Comparison {
            pc,
            at: m.k.describe(pc as usize),
            n_true: s.n[1],
            n_false: s.n[0],
            inputs: inputs(m, s),
        });
    }
    for (&pc, (what, mask, words)) in &ps.ctl {
        c.input_dependent.push(InputDep {
            pc,
            at: m.k.describe(pc as usize),
            what: match what {
                1 => "guard",
                2 => "operand",
                _ => "guard, operand",
            }
            .to_string(),
            inputs: inputs_of(m, words, *mask),
        });
    }
    c.input_independent = ok && ps.ctl.is_empty();
    if !stopped {
        gaps(m, &ps, &mut c);
    }
    c.reached = ps.reached;
    c.seconds = t0.elapsed().as_secs_f64();
    c
}

/// Input coverage riding on the certificate's run (`ptxv check --certify`): the run the
/// certificate makes anyway gets the provenance shadow, instead of a second run of the same
/// launch. Requested before the check (`tap_request`), attached to the first certificate run
/// (`tap_attach`, wellsync.rs), reported at its end (`tap_finish`), taken after the check
/// (`tap_take`).
enum Tap {
    Off,
    Want(Budget),
    Running,
    Done(Box<Coverage>),
}

thread_local! {
    static TAP: std::cell::RefCell<Tap> = const { std::cell::RefCell::new(Tap::Off) };
    /// `set_strict`.
    static STRICT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Tests: a run that ends on a copy of its machine (a placeholder shadow) panics instead of
/// reporting nothing, on this thread.
pub fn set_strict(on: bool) {
    STRICT.set(on);
}

/// Asks for input coverage from the next certificate run on this thread, within `b`.
pub fn tap_request(b: Budget) {
    TAP.with_borrow_mut(|t| *t = Tap::Want(b));
}

/// The coverage of the run the request rode on (None: no certificate run made one, or that
/// run was not the plain deterministic run: the caller runs `analyze` instead). Clears the
/// request.
pub fn tap_take() -> Option<Coverage> {
    TAP.with_borrow_mut(|t| match std::mem::replace(t, Tap::Off) {
        Tap::Done(c) => Some(*c),
        _ => None,
    })
}

/// The machine of a recorded run must carry a live shadow, never a copy's placeholder (debug
/// builds check it after every step of `explore::run_once_until` and the certificate's run).
pub(crate) fn live_shadow(m: &Machine) -> bool {
    m.prov.as_ref().is_none_or(|p| !p.inert)
}

/// A certificate run starts on `m` (wellsync.rs `run_logged`): the requested shadow rides
/// on it. Not on a launch whose CTAs a cluster launch control assigns (the certificate's run
/// launches them differently from `explore::run_once`).
pub(crate) fn tap_attach(m: &mut Machine) {
    TAP.with_borrow_mut(|t| {
        let Tap::Want(b) = *t else {
            return;
        };
        if m.prov.is_some()
            || m.k
                .code
                .iter()
                .any(|i| matches!(i.op, Op::ClcTryCancel { .. } | Op::ClcQuery { .. }))
        {
            *t = Tap::Off;
            return;
        }
        m.prov = Some(Box::new(ProvState {
            nth: m.threads.len(),
            cap: Some((b, std::time::Instant::now())),
            ..Default::default()
        }));
        *t = Tap::Running;
    });
}

/// The certificate run on `m` ended (`ok`: terminal; `whole`: it ran to its end, as
/// `explore::run_once` would have, not stopped by a certificate-only condition).
pub(crate) fn tap_finish(m: &mut Machine, ok: bool, whole: bool) {
    let Some(ps) = m.prov.take() else {
        return;
    };
    TAP.with_borrow_mut(|t| {
        if !matches!(t, Tap::Running) {
            return;
        }
        if ps.inert && (cfg!(debug_assertions) || STRICT.get()) {
            panic!("input coverage: the certificate's run ended on a copy of its machine");
        }
        if !whole || ps.inert {
            *t = Tap::Off;
            return;
        }
        let t0 = ps.cap.map_or_else(std::time::Instant::now, |c| c.1);
        *t = Tap::Done(Box::new(report(m, *ps, ok, None, t0)));
    });
}

/// S-INIT scratch buffers (`build_launch`): for each allocation of `want`, whether one
/// deterministic run shows it produced inside the case (true) or read as an input (false),
/// and the first accessor. Produced: the first access is a thread's plain write (not an
/// atomic, not an async actor), or a plain write later in the run is not ordered after the
/// first read (some other order runs it first). Input: the first access is a read that
/// happens-before every plain write of the run, so it comes first in every order. Only the
/// accesses are recorded (the shadow starts detached: no provenance); the run stops once
/// every allocation of `want` is decided as produced. An allocation the run never reaches is
/// absent.
pub fn first_access(m0: &Machine, want: &[usize]) -> BTreeMap<usize, (bool, String)> {
    let mut m = m0.clone();
    m.prov = Some(Box::new(ProvState {
        nth: m.threads.len(),
        detached: Some("first-access"),
        ..Default::default()
    }));
    let decided = |m: &Machine| {
        m.prov.as_ref().is_some_and(|p| {
            want.iter()
                .all(|&a| p.first.get(&(a as u64)).is_some_and(|f| f.write || f.late))
        })
    };
    let (mut m, _, _, _) = crate::explore::run_once_until(m, &mut |m| decided(m));
    let first = m.prov.take().map(|p| p.first).unwrap_or_default();
    want.iter()
        .filter_map(|&a| {
            first
                .get(&(a as u64))
                .map(|f| (a, (f.write || f.late, m.actor_name(f.actor))))
        })
        .collect()
}

/// Successors of instruction `pc` in the code (for the unexecuted-code walk).
fn succs(m: &Machine, pc: usize) -> Vec<usize> {
    let inst = &m.k.code[pc];
    let next = pc + 1;
    match &inst.op {
        Op::Bra {
            target,
            cond,
            tid0_falls,
            any,
            ..
        } => {
            if inst.guard.reg.is_none() && cond.is_none() && !tid0_falls && !any {
                vec![*target]
            } else {
                vec![*target, next]
            }
        }
        Op::Brx { targets, .. } => targets.clone(),
        Op::Exit | Op::Ret { .. } | Op::Trap { .. } if inst.guard.reg.is_none() => vec![],
        Op::Call { target } => vec![*target, next],
        _ => vec![next],
    }
}

/// The unexecuted instructions and the conditions in front of them: from each reached
/// gate's untaken side, the instructions reachable through unreached ones.
fn gaps(m: &Machine, ps: &ProvState, c: &mut Coverage) {
    let n = m.k.code.len();
    // Per unreached instruction: the gate (pc, input-derived) that leads to it; an input
    // gate wins over a parameter gate.
    let mut by: HashMap<usize, (u32, bool)> = HashMap::new();
    for (g, &(sides, mask)) in ps
        .gates
        .iter()
        .enumerate()
        .filter_map(|(i, x)| Some((i as u32, x.as_ref()?)))
    {
        if sides == 3 {
            continue;
        }
        let gi = g as usize;
        let starts: Vec<usize> = match &m.k.code[gi].op {
            Op::Bra { target, .. } => vec![if sides & 2 == 0 { *target } else { gi + 1 }],
            Op::Brx { targets, .. } => targets.clone(),
            // A predicated instruction whose guard was never true: itself.
            _ if sides & 2 == 0 => vec![gi],
            _ => vec![],
        };
        let mut stack: Vec<usize> = starts
            .into_iter()
            .filter(|&x| x < n && (x == gi || !ps.is_reached(x as u32)))
            .collect();
        let mut seen = HashSet::new();
        while let Some(x) = stack.pop() {
            if !seen.insert(x) {
                continue;
            }
            let e = by.entry(x).or_insert((g, mask != 0));
            if mask != 0 && !e.1 {
                *e = (g, true);
            }
            for y in succs(m, x) {
                if y < n && !ps.is_reached(y as u32) {
                    stack.push(y);
                }
            }
        }
    }
    let unreached =
        |x: usize| !ps.is_reached(x as u32) || by.get(&x).is_some_and(|b| b.0 as usize == x);
    let mut x = 0;
    while x < n {
        if !unreached(x) {
            x += 1;
            continue;
        }
        let key = by.get(&x).copied();
        let mut y = x;
        let part = |i: usize| m.k.part_of(i).map(|p| p.name.as_str());
        while y + 1 < n
            && unreached(y + 1)
            && by.get(&(y + 1)).copied() == key
            && part(y + 1) == part(x)
        {
            y += 1;
        }
        let cnt = y - x + 1;
        c.unexecuted_instructions += cnt;
        // BRA.DIV / BRA.CONV: ptxas's fallback for a warp that is not converged.
        let label = match key {
            Some((_, true)) => "input",
            Some((g, false)) if matches!(m.k.code[g as usize].op, Op::Bra { div: Some(_), .. }) => {
                "divergence"
            }
            Some(_) => "parameter",
            None => "none",
        };
        match label {
            "input" => c.behind_input += cnt,
            "parameter" => c.behind_parameter += cnt,
            _ => {}
        }
        c.unexecuted.push(Gap {
            first: x as u32,
            last: y as u32,
            first_at: m.k.describe(x),
            last_at: m.k.describe(y),
            first_loc: m.k.code[x].loc as u64,
            last_loc: m.k.code[y].loc as u64,
            kernel: m.k.part_of(x).map(|p| p.name.clone()),
            instructions: cnt,
            gate_pc: key.map(|k| k.0),
            gate_at: key.map(|k| m.k.describe(k.0 as usize)),
            gate: label.to_string(),
        });
        x = y + 1;
    }
}

/// One input mutation of B: elements of input buffers set to values that flip an untaken
/// comparison side.
#[derive(Clone, Debug, Serialize)]
pub struct Mutation {
    pub pc: u32,
    pub never: bool,
    /// (allocation, byte offset, bytes, value) per element.
    #[serde(skip)]
    pub writes: Vec<(usize, u64, u8, u64)>,
    /// `buffer[element]=value` per element, e.g. `ok[5]=0, ok[37]=0`.
    pub label: String,
    /// Every element written lies in a buffer whose declared invariant (checked on the
    /// capture) admits the new value: evidence class `on-mutated-input (within declared
    /// invariants)`.
    pub within_invariants: bool,
}

/// The deterministic pass on a mutated input before its check (`mutated_pass`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    /// The mutation's comparison took the side it was to make taken.
    Flipped,
    /// The pass finished without taking that side and reached exactly the instructions the
    /// captured input's run reached: the mutation opens no code on that pass.
    Unchanged,
    /// Neither shown (a mutation given by hand, a run that did not finish, a budget).
    Undecided,
}

/// B's pre-pass: the provenance run on the mutated input `mm`, stopped as soon as the
/// mutation's comparison takes the side it was planned to make taken (usually early).
pub fn mutated_pass(mm: &Machine, base: &Coverage, mu: &Mutation, b: Budget) -> Pass {
    if mu.pc == u32::MAX || !base.terminal || base.stopped.is_some() {
        return Pass::Undecided;
    }
    let t0 = std::time::Instant::now();
    let want: u8 = if mu.never { 2 } else { 1 };
    let flipped = |m: &Machine| {
        m.prov
            .as_ref()
            .and_then(|p| p.sites.get(&mu.pc))
            .is_some_and(|s| s.seen & want != 0)
    };
    let mut m = mm.clone();
    m.prov = Some(Box::new(ProvState {
        nth: m.threads.len(),
        ..Default::default()
    }));
    let mut over = false;
    let (m, _, ok, stopped) = crate::explore::run_once_until(m, &mut |m| {
        if b.seconds.is_some_and(|s| t0.elapsed().as_secs_f64() > s) {
            over = true;
            return true;
        }
        flipped(m)
    });
    if flipped(&m) {
        return Pass::Flipped;
    }
    if over || stopped || !ok {
        return Pass::Undecided;
    }
    let r = &m.prov.as_ref().expect("provenance shadow").reached;
    let n = r.len().max(base.reached.len());
    let at = |v: &Vec<bool>, i: usize| v.get(i).copied().unwrap_or(false);
    if (0..n).all(|i| at(r, i) == at(&base.reached, i)) {
        Pass::Unchanged
    } else {
        Pass::Undecided
    }
}

/// A side B does not flip, and why.
#[derive(Clone, Debug, Serialize)]
pub struct Refusal {
    pub pc: u32,
    pub reason: String,
}

fn cmp_eval(c: Cmp, bits: u8, x: u64, y: u64) -> bool {
    let w = bits.clamp(1, 64) as u32;
    let msk = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
    let (x, y) = (x & msk, y & msk);
    let sx = ((x << (64 - w)) as i64) >> (64 - w);
    let sy = ((y << (64 - w)) as i64) >> (64 - w);
    match c {
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
    }
}

/// A word value (stored in `e.bytes` bytes) whose comparison gives `want` and that `ok`
/// admits, or None. `extra`: candidates tried first (a declared invariant's values and bounds).
fn flip_value(e: &Example, want: bool, extra: &[u64], admits: &dyn Fn(u64) -> bool) -> Option<u64> {
    // The combined predicate must follow the comparison.
    let target = if e.xor {
        want ^ e.and_val
    } else if e.or {
        if e.and_val {
            return None;
        }
        want
    } else {
        if !e.and_val {
            return None;
        }
        want
    };
    let wb = 8 * e.bytes as u32;
    let wm = if wb >= 64 { u64::MAX } else { (1u64 << wb) - 1 };
    let o = e.other;
    let base = [
        o,
        o.wrapping_add(1),
        o.wrapping_sub(1),
        0,
        1,
        u64::MAX,
        wm,
        wm >> 1,
        (wm >> 1) + 1,
        o.wrapping_add(2),
    ];
    let cands: Vec<u64> = extra.iter().chain(base.iter()).copied().collect();
    if e.atom && matches!(e.cmp, Cmp::Eq | Cmp::Ne) {
        return None;
    }
    for &reg in &cands {
        // The register value the word gives when loaded: extended from `bytes`.
        let word = reg & wm;
        if e.atom {
            // The comparison sees the word's value or more: it must hold for every value from
            // the candidate up.
            let ok = [0u64, 1, 64, 4096].iter().all(|&dv| {
                let x = word.wrapping_add(dv) & wm;
                let (a, b) = if e.word_is_a { (x, o) } else { (o, x) };
                x >= word && cmp_eval(e.cmp, e.bits, a, b) == target
            });
            if ok && admits(word) {
                return Some(word);
            }
            continue;
        }
        let ext = if e.sext && wb < 64 {
            (((word << (64 - wb)) as i64) >> (64 - wb)) as u64
        } else {
            word
        };
        let (x, y) = if e.word_is_a { (ext, o) } else { (o, ext) };
        if cmp_eval(e.cmp, e.bits, x, y) == target && admits(word) {
            return Some(word);
        }
    }
    None
}

/// A declared input invariant (case `[buffers.invariant]`): every element of `buffer` is one
/// of `values` and within [`min`, `max`]. Checked on the captured bytes (`Domain::new`), never
/// trusted; B writes only values it admits.
#[derive(Clone, Debug, Serialize)]
pub struct Invariant {
    pub buffer: String,
    pub values: Option<Vec<i64>>,
    pub min: Option<i64>,
    pub max: Option<i64>,
    /// Element size in bytes and signedness (integer dtypes only).
    pub elem: u8,
    pub signed: bool,
}

impl Invariant {
    pub fn holds(&self, x: i64) -> bool {
        self.values.as_ref().is_none_or(|v| v.contains(&x))
            && self.min.is_none_or(|lo| x >= lo)
            && self.max.is_none_or(|hi| x <= hi)
    }

    pub fn describe(&self) -> String {
        let mut p = Vec::new();
        if let Some(v) = &self.values {
            p.push(format!(
                "in {{{}}}",
                v.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if let Some(lo) = self.min {
            p.push(format!(">= {lo}"));
        }
        if let Some(hi) = self.max {
            p.push(format!("<= {hi}"));
        }
        format!("{} {}", self.buffer, p.join(", "))
    }

    fn decode(&self, b: &[u8]) -> i64 {
        let mut x = 0u64;
        for (i, &y) in b.iter().enumerate().take(8) {
            x |= (y as u64) << (8 * i);
        }
        let w = 8 * b.len().min(8) as u32;
        if self.signed && w < 64 {
            ((x << (64 - w)) as i64) >> (64 - w)
        } else {
            x as i64
        }
    }

    /// Candidate words: the declared values and bounds (and their neighbours).
    fn cands(&self) -> Vec<u64> {
        let mut v: Vec<i64> = self.values.clone().unwrap_or_default();
        for b in [self.min, self.max].into_iter().flatten() {
            v.extend([b, b.wrapping_add(1), b.wrapping_sub(1)]);
        }
        v.into_iter().map(|x| x as u64).collect()
    }
}

/// The declared invariants that hold on the captured input, by allocation.
#[derive(Clone, Debug, Default)]
pub struct Domain {
    by_alloc: HashMap<usize, Invariant>,
}

impl Domain {
    /// Each invariant checked on the captured bytes: those that hold are used, the others are
    /// reported (`notes`) and ignored.
    pub fn new(m: &Machine, invs: Vec<Invariant>) -> (Domain, Vec<String>) {
        let mut d = Domain::default();
        let mut notes = Vec::new();
        for inv in invs {
            let Some(a) = m.launch.allocs.iter().position(|x| x.name == inv.buffer) else {
                notes.push(format!(
                    "input invariant {}: no such buffer",
                    inv.describe()
                ));
                continue;
            };
            let init = &m.launch.allocs[a].init;
            let eb = inv.elem.max(1) as usize;
            let bad: Vec<(usize, i64)> = init
                .chunks_exact(eb)
                .enumerate()
                .map(|(i, c)| (i, inv.decode(c)))
                .filter(|&(_, x)| !inv.holds(x))
                .collect();
            if let Some(&(i, x)) = bad.first() {
                notes.push(format!(
                    "input invariant {} does not hold on the captured input ({} of {} elements; {}[{i}] = {x}): not used",
                    inv.describe(),
                    bad.len(),
                    init.len() / eb,
                    inv.buffer
                ));
            } else {
                notes.push(format!(
                    "input invariant {} holds on the captured input ({} elements)",
                    inv.describe(),
                    init.len() / eb
                ));
                d.by_alloc.insert(a, inv);
            }
        }
        (d, notes)
    }

    /// Whether writing `v` into `bytes` bytes at `off` of allocation `a` keeps its element in
    /// the declared invariant (None: no invariant for `a`).
    pub fn admits(&self, m: &Machine, a: usize, off: u64, bytes: u8, v: u64) -> Option<bool> {
        self.admits_in(
            m.launch.allocs.get(a).map(|x| &x.init[..]),
            a,
            off,
            bytes,
            v,
        )
    }

    /// `admits` for allocation `a` whose captured bytes are `init` (targets other than GB300).
    pub fn admits_in(
        &self,
        init: Option<&[u8]>,
        a: usize,
        off: u64,
        bytes: u8,
        v: u64,
    ) -> Option<bool> {
        let inv = self.by_alloc.get(&a)?;
        let eb = inv.elem.max(1) as u64;
        let start = off - off % eb;
        let init = init?;
        let mut el: Vec<u8> = (start..start + eb)
            .map(|i| init.get(i as usize).copied().unwrap_or(0))
            .collect();
        for i in 0..bytes as u64 {
            let j = off + i;
            if j >= start + eb {
                // A write over two elements: both must be checked; refuse to judge.
                return Some(false);
            }
            el[(j - start) as usize] = (v >> (8 * i)) as u8;
        }
        Some(inv.holds(inv.decode(&el)))
    }

    /// Candidate words for allocation `a` from its declared invariant (values, bounds).
    pub fn cands(&self, a: usize) -> Vec<u64> {
        self.by_alloc.get(&a).map(|i| i.cands()).unwrap_or_default()
    }

    /// Every write lands in a buffer with a declared invariant that admits it.
    pub fn within(&self, m: &Machine, writes: &[(usize, u64, u8, u64)]) -> bool {
        !writes.is_empty()
            && writes
                .iter()
                .all(|&(a, off, b, v)| self.admits(m, a, off, b, v) == Some(true))
    }
}

/// B: up to `n` mutations, each writing a flipping value into `k` elements of different
/// threads (different warps first), and the refusals. One untaken side per input buffer
/// first (unrolled copies of one comparison flip the same buffer), then the rest in order.
pub fn plan(
    m: &Machine,
    c: &Coverage,
    n: usize,
    k: usize,
    dom: &Domain,
) -> (Vec<Mutation>, Vec<Refusal>) {
    let pick = |e: &Example, want: bool| {
        flip_value(e, want, &dom.cands(e.alloc as usize), &|v| {
            dom.admits(m, e.alloc as usize, e.off, e.bytes, v) != Some(false)
        })
    };
    let mut cands: Vec<Mutation> = Vec::new();
    let mut refused = Vec::new();
    for u in &c.untaken {
        let want = u.never;
        // The threads that produced the other value are the ones to flip (those whose
        // comparison read one input word; other threads' observations do not matter).
        let ex = c
            .examples
            .get(&u.pc)
            .map(|e| e[!want as usize].clone())
            .unwrap_or_default();
        if ex.is_empty() {
            refused.push(Refusal {
                pc: u.pc,
                reason: u.refuse.clone().unwrap_or_else(|| {
                    "no thread's observation was a simple single-word comparison".into()
                }),
            });
            continue;
        }
        // Spread: elements of threads in different warps first, then evenly over the rest.
        let mut chosen: Vec<&Example> = Vec::new();
        let step = (ex.len() / k.max(1)).max(1);
        for pass in 0..2 {
            for e in ex.iter().step_by(step).chain(ex.iter()) {
                if chosen.len() >= k {
                    break;
                }
                if chosen.iter().any(|c| c.alloc == e.alloc && c.off == e.off) {
                    continue;
                }
                if pass == 0 && chosen.iter().any(|c| c.warp == e.warp) {
                    continue;
                }
                if pick(e, want).is_some() {
                    chosen.push(e);
                }
            }
        }
        if chosen.is_empty() {
            refused.push(Refusal {
                pc: u.pc,
                reason: if ex.iter().any(|e| dom.by_alloc.contains_key(&(e.alloc as usize))) {
                    "no value of the input word that its declared invariant admits flips the comparison".into()
                } else {
                    "no value of the input word flips the comparison (the combining predicate decides it)".into()
                },
            });
            continue;
        }
        let mut writes = Vec::new();
        let mut label = Vec::new();
        for e in &chosen {
            let v = pick(e, want).unwrap();
            writes.push((e.alloc as usize, e.off, e.bytes, v));
            let name = m
                .launch
                .allocs
                .get(e.alloc as usize)
                .map_or("?".to_string(), |x| x.name.clone());
            let shown = if e.sext && e.bytes < 8 && (v >> (8 * e.bytes as u32 - 1)) & 1 == 1 {
                format!("{}", (v | !((1u64 << (8 * e.bytes as u32)) - 1)) as i64)
            } else {
                format!("{v}")
            };
            if e.off % e.bytes as u64 == 0 {
                label.push(format!("{name}[{}]={shown}", e.off / e.bytes as u64));
            } else {
                label.push(format!("{name}[byte {:#x}]={shown}", e.off));
            }
        }
        cands.push(Mutation {
            pc: u.pc,
            never: u.never,
            within_invariants: dom.within(m, &writes),
            writes,
            label: label.join(", "),
        });
    }
    // One per input buffer first, then the rest; never the same writes twice.
    let mut muts: Vec<Mutation> = Vec::new();
    let mut bufs = HashSet::new();
    for pass in 0..2 {
        for mu in &cands {
            if muts.len() >= n {
                break;
            }
            let b = mu.writes.first().map(|w| w.0);
            if pass == 0 && !bufs.insert(b) {
                continue;
            }
            if muts.iter().any(|x| x.writes == mu.writes) {
                continue;
            }
            muts.push(mu.clone());
        }
    }
    (muts, refused)
}

/// A mutation given by hand (`--mutate-input 'ok[0]=0,ok[432]=0'`): `name[i]=v` per
/// element, `elem` the element size of buffer `name` (None: no such buffer).
pub fn parse_mutation(
    m: &Machine,
    spec: &str,
    elem: &dyn Fn(&str) -> Option<usize>,
    dom: &Domain,
) -> Result<Mutation, String> {
    let mut writes = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|x| !x.is_empty()) {
        let (lhs, v) = part
            .split_once('=')
            .ok_or_else(|| format!("{part}: expected name[i]=value"))?;
        let (name, idx) = lhs
            .trim()
            .trim_end_matches(']')
            .split_once('[')
            .ok_or_else(|| format!("{part}: expected name[i]=value"))?;
        let i: u64 = idx
            .trim()
            .parse()
            .map_err(|_| format!("{part}: bad index"))?;
        let v = v.trim();
        let val: u64 = if let Some(h) = v.strip_prefix("0x") {
            u64::from_str_radix(h, 16).map_err(|_| format!("{part}: bad value"))?
        } else {
            v.parse::<i64>().map_err(|_| format!("{part}: bad value"))? as u64
        };
        let a = m
            .launch
            .allocs
            .iter()
            .position(|x| x.name == name.trim())
            .ok_or_else(|| format!("{part}: no buffer {name}"))?;
        let eb = elem(name.trim()).ok_or_else(|| format!("{part}: no buffer {name}"))?;
        writes.push((a, i * eb as u64, eb as u8, val));
    }
    Ok(Mutation {
        pc: u32::MAX,
        never: false,
        within_invariants: dom.within(m, &writes),
        writes,
        label: spec.to_string(),
    })
}

/// The machine for `m0`'s kernel on the input with `mu`'s elements written.
pub fn apply(m0: &Machine, mu: &Mutation) -> Machine {
    let mut l = (*m0.launch).clone();
    for &(a, off, bytes, v) in &mu.writes {
        if let Some(al) = l.allocs.get_mut(a) {
            for i in 0..bytes as usize {
                if let Some(b) = al.init.get_mut(off as usize + i) {
                    *b = (v >> (8 * i)) as u8;
                }
            }
        }
    }
    Machine::new(m0.k.clone(), Arc::new(l))
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `TpuMachine`: one or more tpu7x TensorCores executing a lifted LLO kernel.
//!
//! State per core: scalar/predicate/vector/mask registers, SMEM (words), VMEM (rows of 128
//! words), sync flags, result FIFOs (sync flags, EUP, XLU, vector-to-scalar), two MXUs; shared:
//! HBM (granules of 8 words), in-flight DMAs and remote signals. Bundles execute with
//! read-old/commit-at-end semantics. Deterministic mode (`run`) mirrors tools/llosim.py
//! exactly; the explorer (`model.rs`) drives the same state with DMA completion and remote
//! delivery as nondeterministic actions.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::OnceLock;

use ptxv_sim::machine::FastHasher;
use ptxv_sim::machine::Finding;
use ptxv_sim::machine::VC;
use ptxv_sim::machine::vc_get;
use ptxv_sim::machine::vc_join;
use ptxv_sim::machine::vc_single;
use ptxv_sim::machine::vc_tick;
use ptxv_tpu_ir::*;

use crate::num::CANON_NAN;
use crate::num::Numerics;
use crate::num::VpuOp;
use crate::num::ftz;

/// One vector register: 8 sublanes x 128 lanes, index `sublane * 128 + lane`.
pub type V = Arc<[u32; 1024]>;

const PAGE: usize = 4096;
/// 64 MiB VMEM per TensorCore (public tpu7x figure).
pub const VMEM_WORDS: usize = 64 * 1024 * 1024 / 4;
/// 1 MiB SMEM (fixed addresses up to 0xfff8c seen).
pub const SMEM_WORDS: usize = 1 << 18;

fn zero_v() -> V {
    Arc::new([0u32; 1024])
}

fn new_v(f: impl Fn(usize) -> u32) -> V {
    let mut a = [0u32; 1024];
    for (i, x) in a.iter_mut().enumerate() {
        *x = f(i);
    }
    Arc::new(a)
}

// ------------------------------------------------------------------ errors

#[derive(Clone, Debug, PartialEq)]
pub enum SimErr {
    /// The core halted (`shalt.err`, an empty FIFO pop, a step limit).
    Halt(String),
    /// An op or form the machine does not model.
    Unimpl(String),
    Deadlock(String),
}

impl fmt::Display for SimErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimErr::Halt(m) => write!(f, "Halt: {m}"),
            SimErr::Unimpl(m) => write!(f, "Unimpl: {m}"),
            SimErr::Deadlock(m) => write!(f, "Deadlock: {m}"),
        }
    }
}

impl std::error::Error for SimErr {}

impl SimErr {
    pub fn status(&self) -> &'static str {
        match self {
            SimErr::Halt(_) => "Halt",
            SimErr::Unimpl(_) => "Unimpl",
            SimErr::Deadlock(_) => "Deadlock",
        }
    }
}

type R<T> = Result<T, SimErr>;

fn unimpl<T>(m: impl Into<String>) -> R<T> {
    Err(SimErr::Unimpl(m.into()))
}

// ------------------------------------------------------------------ paged memory

struct PageData {
    w: Vec<u32>,
    h: OnceLock<u64>,
}

impl Clone for PageData {
    fn clone(&self) -> Self {
        PageData {
            w: self.w.clone(),
            h: OnceLock::new(),
        }
    }
}

impl PageData {
    fn hash(&self) -> u64 {
        *self.h.get_or_init(|| {
            let mut h = FastHasher::default();
            for &x in &self.w {
                h.write_u32(x);
            }
            h.finish()
        })
    }
}

/// Copy-on-write word memory; untouched pages read as zero.
#[derive(Clone)]
pub struct Mem {
    pages: Vec<Option<Arc<PageData>>>,
    words: usize,
}

impl Mem {
    pub fn new(words: usize) -> Self {
        Mem {
            pages: vec![None; words.div_ceil(PAGE)],
            words,
        }
    }

    pub fn len(&self) -> usize {
        self.words
    }

    pub fn is_empty(&self) -> bool {
        self.words == 0
    }

    pub fn get(&self, i: usize) -> u32 {
        match self.pages.get(i / PAGE) {
            Some(Some(p)) => p.w[i % PAGE],
            _ => 0,
        }
    }

    fn page_mut(&mut self, p: usize) -> &mut Vec<u32> {
        let slot = &mut self.pages[p];
        if slot.is_none() {
            *slot = Some(Arc::new(PageData {
                w: vec![0; PAGE],
                h: OnceLock::new(),
            }));
        }
        let pd = Arc::make_mut(slot.as_mut().expect("just set"));
        pd.h = OnceLock::new();
        &mut pd.w
    }

    pub fn set(&mut self, i: usize, v: u32) {
        self.page_mut(i / PAGE)[i % PAGE] = v;
    }

    pub fn read(&self, at: usize, n: usize) -> R<Vec<u32>> {
        if at + n > self.words {
            return Err(SimErr::Halt(format!(
                "memory read out of range: {at:#x}+{n}"
            )));
        }
        Ok((at..at + n).map(|i| self.get(i)).collect())
    }

    pub fn write(&mut self, at: usize, data: &[u32]) -> R<()> {
        if at + data.len() > self.words {
            return Err(SimErr::Halt(format!(
                "memory write out of range: {at:#x}+{}",
                data.len()
            )));
        }
        let mut i = 0;
        while i < data.len() {
            let a = at + i;
            let (p, o) = (a / PAGE, a % PAGE);
            let n = (PAGE - o).min(data.len() - i);
            if data[i..i + n].iter().all(|&x| x == 0) && self.pages[p].is_none() {
                i += n;
                continue;
            }
            self.page_mut(p)[o..o + n].copy_from_slice(&data[i..i + n]);
            i += n;
        }
        Ok(())
    }

    fn hash_into(&self, h: &mut FastHasher) {
        for (i, p) in self.pages.iter().enumerate() {
            if let Some(p) = p {
                let ph = p.hash();
                if ph != ZERO_PAGE_HASH.with(|z| *z) {
                    h.write_usize(i);
                    h.write_u64(ph);
                }
            }
        }
    }
}

thread_local! {
    static ZERO_PAGE_HASH: u64 = PageData { w: vec![0; PAGE], h: OnceLock::new() }.hash();
}

// ------------------------------------------------------------------ core state

#[derive(Clone, Default)]
struct Regs {
    s: Vec<u32>,
    p: Vec<bool>,
    v: Vec<Option<V>>,
    vm: Vec<Option<V>>,
}

#[derive(Clone, Default)]
struct Mxu {
    /// Final listings (notes/numerics.md (e)): `vmatprep.subr` FIFO, each entry paired with the
    /// next push; pushes since the last latch (rows, paired subr rows) and their form (1 = push1,
    /// 3 = push3, 4 = push1 .xpose, 5 = push3 .xpose); latched weights (form, K, K x 128 lo, K x 128 hi).
    stage: VecDeque<Vec<u32>>,
    pend: Vec<(Vec<u32>, Option<Vec<u32>>)>,
    pend_mode: u8,
    gmr: Option<Arc<(u8, usize, Vec<u32>, Vec<u32>)>>,
    /// Original listings: staged weights and the result FIFO.
    stage_lo: Vec<Vec<u32>>,
    stage_hi: Vec<Vec<u32>>,
    /// Latched weights: (k, k x 256 bf16 bits).
    w: Option<Arc<(usize, Vec<u32>)>>,
    prep: Option<V>,
    mrb: BTreeMap<u32, V>,
    fifo: VecDeque<V>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Pending {
    remaining: u8,
    target: u32,
}

#[derive(Clone)]
pub struct Core {
    pub id: usize,
    regs: Regs,
    pub smem: Mem,
    pub vmem: Mem,
    pub flags: BTreeMap<u32, i64>,
    /// Flags the kernel writes with vsyncpa/vsyncset (re-initialized by the kernel itself: the
    /// compiler emits `vsyncpa f, 0` at entry and `vsyncpa f, 1` at exit).
    pub set_flags: std::collections::BTreeSet<u32>,
    /// DMAs this core has issued (names its DMA actors).
    pub dma_seq: u32,
    sfrf: VecDeque<u32>,
    v2s: VecDeque<u32>,
    eup: VecDeque<V>,
    xlu: [VecDeque<V>; 2],
    xpose: [Vec<V>; 2],
    mxu: [Mxu; 2],
    pub pc: usize,
    branch: Option<Pending>,
    pub halted: Option<String>,
    pub done: bool,
    pub steps: u64,
    pub blocked: Option<String>,
    backedge: bool,
    phi_snap: Option<Box<Regs>>,
    operands: Vec<u32>,
    /// Explorer: vector clock and the flags' clocks.
    pub vc: VC,
    flag_vc: BTreeMap<u32, VC>,
}

impl Core {
    fn new(id: usize) -> Self {
        Core {
            id,
            regs: Regs::default(),
            smem: Mem::new(SMEM_WORDS),
            vmem: Mem::new(VMEM_WORDS),
            flags: BTreeMap::new(),
            set_flags: Default::default(),
            dma_seq: 0,
            sfrf: VecDeque::new(),
            v2s: VecDeque::new(),
            eup: VecDeque::new(),
            xlu: Default::default(),
            xpose: Default::default(),
            mxu: Default::default(),
            pc: 0,
            branch: None,
            halted: None,
            done: false,
            steps: 0,
            blocked: None,
            backedge: false,
            phi_snap: None,
            operands: Vec::new(),
            vc: vc_single(id as u32, 1),
            flag_vc: BTreeMap::new(),
        }
    }

    fn flag(&self, f: u32) -> i64 {
        self.flags.get(&f).copied().unwrap_or(0)
    }

    /// Software add (`vsyncadd`, a wait's decrement, a remote signal): int32, overflow halts.
    /// ASSUME(TPU-S-SEM-INT32, TPU-S-SEM-OVF-TRAP)
    fn flag_add_checked(&mut self, f: u32, v: i64) -> Result<(), String> {
        let cur = self.flag(f);
        let n = cur + v;
        if n < i32::MIN as i64 || n > i32::MAX as i64 {
            return Err(format!("sync flag {f} overflows int32 ({cur} + {v})"));
        }
        self.flags.insert(f, n);
        Ok(())
    }

    /// DMA completion add: wraps modulo 2^32. ASSUME(TPU-S-SEM-OVF-TRAP)
    fn flag_add_wrap(&mut self, f: u32, v: i64) {
        let n = (self.flag(f) as i32).wrapping_add(v as i32);
        self.flags.insert(f, n as i64);
    }

    fn hash_into(&self, h: &mut FastHasher) {
        h.write_usize(self.pc);
        self.branch.hash(h);
        self.done.hash(h);
        self.halted.hash(h);
        let r = &self.regs;
        r.s.hash(h);
        r.p.hash(h);
        for f in [&r.v, &r.vm] {
            for (i, x) in f.iter().enumerate() {
                if let Some(x) = x {
                    h.write_usize(i);
                    x.as_slice().hash(h);
                }
            }
        }
        self.flags.hash(h);
        self.set_flags.hash(h);
        self.dma_seq.hash(h);
        // happens-before state: merging states whose clocks differ could drop a race
        self.vc.hash(h);
        self.flag_vc.hash(h);
        self.sfrf.hash(h);
        self.v2s.hash(h);
        for q in [&self.eup, &self.xlu[0], &self.xlu[1]] {
            h.write_usize(q.len());
            for x in q {
                x.as_slice().hash(h);
            }
        }
        for m in &self.mxu {
            m.stage.hash(h);
            m.pend.hash(h);
            m.pend_mode.hash(h);
            m.gmr.as_ref().map(|g| (g.0, g.1, &g.2, &g.3)).hash(h);
            m.stage_lo.hash(h);
            m.stage_hi.hash(h);
            m.w.as_ref().map(|w| (w.0, &w.1)).hash(h);
            m.prep.as_ref().map(|p| p.as_slice()).hash(h);
            for (k, v) in &m.mrb {
                (k, v.as_slice()).hash(h);
            }
            h.write_usize(m.fifo.len());
            for v in &m.fifo {
                v.as_slice().hash(h);
            }
        }
        for x in &self.xpose {
            h.write_usize(x.len());
            for v in x {
                v.as_slice().hash(h);
            }
        }
        self.phi_snap.is_some().hash(h);
        self.backedge.hash(h);
        self.smem.hash_into(h);
        self.vmem.hash_into(h);
    }
}

/// An in-flight DMA: an atomic copy at completion plus one flag add of size/32.
/// ASSUME(TPU-S-DMA-REDUCE, TPU-S-DMA-DONE)
#[derive(Clone)]
pub struct Dma {
    pub core: usize,
    pub kind: DmaKind,
    pub src: u32,
    pub dst: u32,
    pub nbytes: u64,
    pub flag: u32,
    pub dst_core: usize,
    pub src_flag: Option<u32>,
    pub stride: Option<(u32, u32, u32)>,
    pub pc: usize,
    pub actor: u32,
    pub vc: VC,
    /// Source data latched at the send-done step (remote copies with a send flag), and its
    /// hash; `None` until then.
    pub latched: Option<Arc<(Vec<Vec<u32>>, u64)>>,
}

/// An in-flight remote sync-flag add (explorer mode; may overtake earlier remote DMAs).
/// ASSUME(TPU-S-REM-SIGNAL)
#[derive(Clone)]
pub struct RemoteMsg {
    pub from: usize,
    pub to: usize,
    pub flag: u32,
    pub n: i64,
    pub pc: usize,
    pub vc: VC,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DmaPolicy {
    Eager,
    Lazy,
    Random(u64),
}

/// Where an operand lives.
#[derive(Clone, Debug)]
pub struct Placed {
    pub core: usize,
    pub index: u32,
    pub space: String,
    pub addr: u32,
    pub decl: OperandDecl,
    /// Words placed (the operand's size in its layout).
    pub nwords: usize,
}

// ------------------------------------------------------------------ race shadow

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum MemKind {
    Vmem,
    Smem,
    Hbm,
}

/// One recorded access: rows [lo, hi) of one memory, by one actor at one epoch.
#[derive(Clone, Debug)]
struct Rec {
    kind: MemKind,
    owner: u16,
    lo: u32,
    hi: u32,
    write: bool,
    actor: u32,
    clock: u32,
    who: Arc<str>,
}

/// Race shadow: the accesses not yet known to be ordered before every later one, as row
/// intervals (a 64 MiB DMA is one record). Shared copy-on-write between explorer states.
#[derive(Clone, Default)]
pub struct RaceState {
    recs: Arc<Vec<Rec>>,
}

// ------------------------------------------------------------------ machine

/// Effects of one op, committed in op order at the end of its bundle; every op of a bundle
/// reads the state from before the bundle. ASSUME(TPU-E-VLIW-READ-OLD)
enum Eff {
    Reg(Reg, Value),
    Vmem {
        row: u32,
        mask: Option<Box<[bool; 1024]>>,
        val: V,
    },
    Smem(u32, u32),
    FlagSet(u32, i64),
    FlagAdd(u32, i64),
    SfrfMov(u32),
    PushV(Fifo, V),
    PushS(Fifo, u32),
    Xpose(u8, XposePhase, V),
    Remote {
        to: usize,
        flag: u32,
        n: i64,
    },
    MxuStage {
        unit: u8,
        hi: bool,
        rows: Vec<u32>,
    },
    MxuSubrF(u8, Vec<u32>),
    MxuPushF(u8, Vec<u32>, u8),
    MxuPrep(u8, V),
    MxuMatmul {
        unit: u8,
        x: V,
        slot: u32,
        f32: bool,
        latch: bool,
    },
    OMxuMatmul {
        unit: u8,
        x: V,
        f32: bool,
    },
}

#[derive(Clone)]
enum Value {
    S(u32),
    P(bool),
    V(V),
}

#[derive(Clone)]
pub struct TpuMachine {
    pub kernel: Arc<Kernel>,
    pub cores: Vec<Core>,
    pub hbm: BTreeMap<u32, Mem>,
    next_hbm: u32,
    pub pending: Vec<Dma>,
    pub remote: Vec<RemoteMsg>,
    pub devmap: HashMap<(u32, u32), usize>,
    alloc_addr: Arc<Vec<u32>>,
    /// Sync-flag allocations: index ranges and kinds.
    pub flag_ranges: Arc<Vec<(u32, u32, FlagKind)>>,
    vmem_next: u32,
    vnext: Vec<u32>,
    snext: Vec<u32>,
    pub placed: Vec<Placed>,
    pub num: &'static dyn Numerics,
    dma_count: u32,
    /// Explorer mode: remote signals in flight, vector clocks and race shadow.
    pub explore: bool,
    pub race: Option<Box<RaceState>>,
    pub max_steps: u64,
    rng: u64,
}

/// Address of each allocation (in its space's unit) and the first free VMEM row after them.
fn assign_allocs(k: &Kernel) -> (Vec<u32>, u32) {
    let (mut vm, mut sm, mut sf) = (0u32, 0x1000u32 / 4, 16u32);
    let mut addr = vec![0u32; k.allocs.len()];
    let declared = |a: &Alloc| a.name.bytes().all(|c| c.is_ascii_digit());
    let mut ids: Vec<usize> = (0..k.allocs.len())
        .filter(|&i| declared(&k.allocs[i]))
        .collect();
    ids.sort_by_key(|&i| k.allocs[i].name.parse::<u64>().unwrap_or(0));
    for i in ids {
        let a = &k.allocs[i];
        if let Some(off) = a.fixed {
            addr[i] = (off / if a.space == Space::Vmem { 512 } else { 4 }) as u32;
            continue;
        }
        match a.space {
            Space::Vmem => {
                addr[i] = vm;
                vm += a.size.div_ceil(512) as u32;
            }
            Space::Smem => {
                addr[i] = sm;
                sm += a.size.div_ceil(4) as u32 + 8;
            }
            Space::Sflag => {
                addr[i] = sf;
                // `u32[]{:S(2)}` copy-start outputs print size 2^64-1: one flag
                sf += if a.size >= 1 << 32 {
                    1
                } else {
                    (a.size / 4).max(1) as u32
                };
            }
            Space::Hbm => addr[i] = 0,
        }
    }
    let mut vnext = vm + 64;
    for (i, a) in k.allocs.iter().enumerate().filter(|(_, a)| !declared(a)) {
        if a.space == Space::Vmem {
            addr[i] = vnext;
            vnext += 8;
        } else {
            addr[i] = 0x20000 + i as u32 * 8;
        }
    }
    (addr, vnext)
}

/// What a sync flag is for (from its allocation's tag).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlagKind {
    Scratch,
    /// `get_barrier_semaphore`: one per core, shared by every collective id, no exit check.
    /// ASSUME(TPU-S-BARRIER)
    Barrier,
    /// `dummy sync flag` (flag 0): completion target of DMAs nobody waits for.
    Dummy,
}

/// Flag ranges `[lo, hi)` of the kernel's sync-flag allocations.
fn flag_ranges(k: &Kernel, addr: &[u32]) -> Vec<(u32, u32, FlagKind)> {
    k.allocs
        .iter()
        .enumerate()
        .filter(|(_, a)| a.space == Space::Sflag)
        .map(|(i, a)| {
            let n = if a.size >= 1 << 32 {
                1
            } else {
                (a.size / 4).max(1) as u32
            };
            let kind = if a.tag.contains("barrier") {
                FlagKind::Barrier
            } else if a.tag.contains("dummy") {
                FlagKind::Dummy
            } else {
                FlagKind::Scratch
            };
            (addr[i], addr[i] + n, kind)
        })
        .collect()
}

impl TpuMachine {
    pub fn new(
        kernel: Arc<Kernel>,
        ncores: usize,
        devmap: HashMap<(u32, u32), usize>,
        num: &'static dyn Numerics,
    ) -> Self {
        let (alloc_addr, vmem_next) = assign_allocs(&kernel);
        let flag_ranges = Arc::new(flag_ranges(&kernel, &alloc_addr));
        let mut cores: Vec<Core> = (0..ncores).map(Core::new).collect();
        for c in &mut cores {
            // partition id, replica id, core index. ASSUME(TPU-L-DEVICE-IDS)
            c.smem.set(0xfff88 / 4, c.id as u32);
            c.smem.set(0xfff8c / 4, 0);
            c.smem.set(1, 0);
        }
        TpuMachine {
            kernel,
            cores,
            hbm: BTreeMap::new(),
            next_hbm: 1 << 20,
            pending: Vec::new(),
            remote: Vec::new(),
            devmap,
            alloc_addr: Arc::new(alloc_addr),
            flag_ranges,
            vmem_next,
            vnext: Vec::new(),
            snext: Vec::new(),
            placed: Vec::new(),
            num,
            dma_count: 0,
            explore: false,
            race: None,
            max_steps: 5_000_000,
            rng: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// Allocate an HBM buffer (granule address), padded to 4 KiB.
    pub fn hbm_alloc(&mut self, nwords: usize) -> u32 {
        let base = self.next_hbm;
        let g = nwords.div_ceil(1024) * 128;
        self.hbm.insert(base, Mem::new(g * 8));
        self.next_hbm += g as u32 + 4096;
        base
    }

    fn hbm_seg(&self, g: u32, n: usize) -> R<(u32, usize)> {
        if let Some((&b, m)) = self.hbm.range(..=g).next_back() {
            let off = (g - b) as usize * 8;
            if off + n <= m.len() {
                return Ok((b, off));
            }
        }
        Err(SimErr::Halt(format!(
            "HBM access outside any buffer: granule {g:#x} +{n} words"
        )))
    }

    pub fn hbm_read(&self, g: u32, n: usize) -> R<Vec<u32>> {
        let (b, off) = self.hbm_seg(g, n)?;
        self.hbm[&b].read(off, n)
    }

    pub fn hbm_write(&mut self, g: u32, data: &[u32]) -> R<()> {
        let (b, off) = self.hbm_seg(g, data.len())?;
        self.hbm.get_mut(&b).expect("found").write(off, data)
    }

    /// Place operand data for core `c` (inputs: their words; outputs: zeros), in index order:
    /// VMEM operands after the allocations (8-row gaps), SMEM operands from word 0x2000
    /// (16-word gaps), HBM buffers from the shared allocator. Scalar operands take `scalar`.
    pub fn place(
        &mut self,
        c: usize,
        decl: &OperandDecl,
        words: &[u32],
        scalar: Option<u32>,
    ) -> R<()> {
        while self.vnext.len() <= c {
            self.vnext.push(self.vmem_next);
            self.snext.push(0x8000 / 4);
        }
        let idx = decl.index as usize;
        if self.cores[c].operands.len() <= idx {
            self.cores[c].operands.resize(idx + 1, 0);
        }
        let space = decl.space.as_str();
        let addr = match space {
            "<no memory space>" | "none" => {
                self.cores[c].operands[idx] = scalar.unwrap_or(0);
                return Ok(());
            }
            "vmem" => {
                let at = self.vnext[c];
                self.cores[c].vmem.write(at as usize * 128, words)?;
                self.vnext[c] += words.len().div_ceil(128) as u32 + 8;
                at
            }
            "smem" => {
                let at = self.snext[c];
                self.cores[c].smem.write(at as usize, words)?;
                self.snext[c] += words.len() as u32 + 16;
                at
            }
            "hbm" => {
                let b = self.hbm_alloc(words.len());
                self.hbm_write(b, words)?;
                b
            }
            s => return unimpl(format!("operand space {s}")),
        };
        self.cores[c].operands[idx] = addr;
        self.placed.push(Placed {
            core: c,
            index: decl.index,
            space: space.to_string(),
            addr,
            decl: decl.clone(),
            nwords: words.len(),
        });
        Ok(())
    }

    /// Words of a placed operand.
    pub fn operand_words(&self, p: &Placed) -> R<Vec<u32>> {
        let n = p.nwords;
        match p.space.as_str() {
            "vmem" => self.cores[p.core].vmem.read(p.addr as usize * 128, n),
            "smem" => self.cores[p.core].smem.read(p.addr as usize, n),
            _ => self.hbm_read(p.addr, n),
        }
    }

    // -------------------------------------------------------------- operands

    fn reg_get(&self, c: usize, r: Reg) -> Value {
        let regs = match &self.cores[c].phi_snap {
            Some(s) if PHI_READ.with(|p| p.get()) => s.as_ref(),
            _ => &self.cores[c].regs,
        };
        let i = r.idx as usize;
        match r.file {
            File::S => Value::S(regs.s.get(i).copied().unwrap_or(0)),
            File::P => Value::P(regs.p.get(i).copied().unwrap_or(false)),
            File::V => Value::V(regs.v.get(i).cloned().flatten().unwrap_or_else(zero_v)),
            File::Vm => Value::V(regs.vm.get(i).cloned().flatten().unwrap_or_else(zero_v)),
        }
    }

    fn addr(&self, c: usize, a: &Addr) -> u32 {
        let mut t: i64 = 0;
        for x in &a.terms {
            t += match x {
                Term::Imm(v) => *v,
                Term::Alloc(i) => self.alloc_addr[*i as usize] as i64,
                Term::Reg(r) => match self.reg_get(c, *r) {
                    Value::S(v) => v as i64,
                    Value::P(b) => b as i64,
                    Value::V(_) => 0,
                },
            };
        }
        t as u32
    }

    fn sval(&self, c: usize, v: &Val) -> R<u32> {
        Ok(match v {
            Val::Imm(x) => *x,
            Val::Addr(a) => self.addr(c, a),
            Val::Reg(r) => match self.reg_get(c, *r) {
                Value::S(x) => x,
                Value::P(b) => b as u32,
                Value::V(_) => return unimpl("vector register in scalar context"),
            },
        })
    }

    fn vval(&self, c: usize, v: &Val) -> R<V> {
        if let Val::Reg(r) = v
            && matches!(r.file, File::V | File::Vm)
        {
            let Value::V(x) = self.reg_get(c, *r) else {
                unreachable!()
            };
            return Ok(x);
        }
        let s = self.sval(c, v)?;
        Ok(Arc::new([s; 1024]))
    }

    fn mask(&self, c: usize, r: Reg) -> V {
        match self.reg_get(c, r) {
            Value::V(x) => x,
            Value::S(x) => Arc::new([(x != 0) as u32; 1024]),
            Value::P(b) => Arc::new([b as u32; 1024]),
        }
    }

    fn guard(&self, c: usize, g: &Option<Guard>) -> bool {
        match g {
            None => true,
            Some(Guard::Const(b)) => *b,
            Some(Guard::Reg { reg, neg }) => {
                let v = match self.reg_get(c, *reg) {
                    Value::S(x) => x != 0,
                    Value::P(b) => b,
                    Value::V(_) => true,
                };
                v != *neg
            }
        }
    }

    // -------------------------------------------------------------- stepping

    /// Can core `c` execute its next bundle (not done, not halted, waits satisfied)?
    pub fn can_step(&self, c: usize) -> bool {
        let core = &self.cores[c];
        if core.done || core.halted.is_some() {
            return false;
        }
        let Some(b) = self.kernel.bundles.get(core.pc) else {
            return true;
        };
        if self.window_sync(c, b) {
            return true;
        }
        b.insts.iter().all(|i| self.wait_ok(c, i))
    }

    /// A wait whose condition fails stalls its whole bundle (no op of it executes).
    /// ASSUME(TPU-E-WAIT-BUNDLE)
    fn wait_ok(&self, c: usize, i: &Inst) -> bool {
        let (flag, val, eq) = match &i.op {
            Op::Wait { flag, val, eq } => (flag, val, *eq),
            Op::DmaDone { flag, val } => (flag, val, false),
            _ => return true,
        };
        if !self.guard(c, &i.guard) {
            return true;
        }
        let (Ok(f), Ok(n)) = (self.sval(c, flag), self.sval(c, val)) else {
            return true;
        };
        let cur = self.cores[c].flag(f);
        let need = n as i32 as i64;
        if eq { cur == need } else { cur >= need }
    }

    /// The wait core `c` is blocked on, if any: (flag, needed value, eq).
    pub fn blocked_wait(&self, c: usize) -> Option<(u32, i64, bool)> {
        let b = self.kernel.bundles.get(self.cores[c].pc)?;
        for i in &b.insts {
            if self.wait_ok(c, i) {
                continue;
            }
            let (flag, val, eq) = match &i.op {
                Op::Wait { flag, val, eq } => (flag, val, *eq),
                Op::DmaDone { flag, val } => (flag, val, false),
                _ => continue,
            };
            let (Ok(f), Ok(n)) = (self.sval(c, flag), self.sval(c, val)) else {
                continue;
            };
            return Some((f, n as i32 as i64, eq));
        }
        None
    }

    /// Largest value flag `f` of core `c` can reach from the work in flight alone (pending DMA
    /// completion and send adds, positive remote signals).
    pub fn flag_potential(&self, c: usize, f: u32) -> i64 {
        let mut v = self.cores[c].flag(f);
        for d in &self.pending {
            let g = (d.nbytes / 32) as i64;
            if d.dst_core == c && d.flag == f {
                v += g;
            }
            if d.core == c && d.src_flag == Some(f) && d.latched.is_none() {
                v += g;
            }
        }
        for r in &self.remote {
            if r.to == c && r.flag == f && r.n > 0 {
                v += r.n;
            }
        }
        v
    }

    /// A scalar operand that does not depend on registers (immediate or allocation address).
    pub fn const_val(&self, v: &Val) -> Option<u32> {
        match v {
            Val::Imm(x) => Some(*x),
            Val::Addr(a) if a.terms.iter().all(|t| !matches!(t, Term::Reg(_))) => {
                Some(self.addr(0, a))
            }
            _ => None,
        }
    }

    /// Does the next bundle of core `c` execute an op other cores or DMA engines can observe
    /// (sync flags, DMA issue, halts)?
    pub fn next_visible(&self, c: usize) -> bool {
        let core = &self.cores[c];
        match self.kernel.bundles.get(core.pc) {
            Some(b) => b
                .insts
                .iter()
                .any(|i| i.op.is_visible() && self.guard(c, &i.guard)),
            None => false,
        }
    }

    /// A sync-flag op about to execute inside a taken branch's delay window (faults the core
    /// on hardware: brsync probe). ASSUME(TPU-E-BR-SYNC)
    fn window_sync(&self, c: usize, b: &Bundle) -> bool {
        self.cores[c].branch.is_some()
            && b.insts
                .iter()
                .any(|i| i.op.is_sync() && self.guard(c, &i.guard))
    }

    /// Execute one bundle of core `c`. Ok(false): blocked on a wait.
    pub fn step(&mut self, c: usize, findings: &mut Vec<Finding>) -> R<bool> {
        let kernel = self.kernel.clone();
        let pc = self.cores[c].pc;
        let Some(b) = kernel.bundles.get(pc) else {
            self.cores[c].done = true;
            return Ok(true);
        };
        if self.window_sync(c, b) {
            let msg = format!(
                "sync-flag op inside a branch delay window: {}",
                kernel.describe(pc)
            );
            findings.push(Finding {
                kind: "violation".into(),
                sig: format!("delay-window-sync:core{c}@{:#x}", b.addr),
                msg: format!("core {c} faults: {msg}"),
                trace: Vec::new(),
                pcs: vec![pc as u32],
                gb300: None,
                perm_rows: vec![],
            });
            self.cores[c].halted = Some(msg);
            return Ok(true);
        }
        for i in &b.insts {
            if !self.wait_ok(c, i) {
                self.cores[c].blocked = Some(i.text.clone());
                return Ok(false);
            }
        }
        self.cores[c].blocked = None;
        let orig = kernel.front == Front::Original;
        let mut effs: Vec<Eff> = Vec::new();
        let mut nxt: Option<(u32, u8)> = None;
        for i in &b.insts {
            if !self.guard(c, &i.guard) {
                continue;
            }
            if let Op::Branch { target, delay } = i.op {
                nxt = Some((target, delay));
            }
            self.exec(c, i, &mut effs, findings)?;
            if self.cores[c].halted.is_some() {
                return Ok(true);
            }
        }
        for e in effs {
            self.commit(c, e)?;
        }
        let core = &mut self.cores[c];
        core.steps += 1;
        if let Some(p) = core.branch {
            if p.remaining == 0 {
                core.pc = p.target as usize;
                core.branch = None;
                if nxt.is_some() {
                    return unimpl("branch in a branch delay slot");
                }
                return Ok(true);
            }
            core.branch = Some(Pending {
                remaining: p.remaining - 1,
                ..p
            });
        }
        if let Some((target, delay)) = nxt {
            if delay == 0 {
                core.phi_snap = None;
                core.backedge = target as usize <= core.pc;
                core.pc = target as usize;
                return Ok(true);
            }
            core.branch = Some(Pending {
                remaining: delay - 1,
                target,
            });
        }
        if orig && b.insts.iter().any(|i| matches!(i.op, Op::Branch { .. })) {
            core.backedge = false;
            core.phi_snap = None;
        }
        core.pc += 1;
        if core.pc >= kernel.bundles.len() && core.branch.is_none() {
            core.done = true;
        }
        Ok(true)
    }

    fn def(i: &Inst, v: Value, effs: &mut Vec<Eff>) {
        if let Some(r) = i.def {
            effs.push(Eff::Reg(r, v));
        }
    }

    fn exec(
        &mut self,
        c: usize,
        i: &Inst,
        effs: &mut Vec<Eff>,
        findings: &mut Vec<Finding>,
    ) -> R<()> {
        let num = self.num;
        macro_rules! s {
            ($v:expr) => {
                self.sval(c, $v)?
            };
        }
        macro_rules! v {
            ($v:expr) => {
                self.vval(c, $v)?
            };
        }
        match &i.op {
            Op::Nop => {}
            Op::Unlowered { idx } => {
                let u = &self.kernel.unlowered[*idx as usize];
                return unimpl(format!("opcode {}: {} ({})", u.mn, u.text, u.why));
            }
            Op::Copy(a) | Op::SMov(a) => Self::def(i, Value::S(s!(a)), effs),
            Op::PMov(a) => Self::def(i, Value::P(s!(a) != 0), effs),
            Op::Operand { index } => {
                let v = self.cores[c]
                    .operands
                    .get(*index as usize)
                    .copied()
                    .unwrap_or(0);
                Self::def(i, Value::S(v), effs)
            }
            Op::SBin { op, a, b } => {
                let (a, b) = (s!(a), s!(b));
                let (sa, sb) = (a as i32 as i64, b as i32 as i64);
                let r: u32 = match op {
                    SBin::Add => a.wrapping_add(b),
                    SBin::Sub => a.wrapping_sub(b),
                    SBin::Mul => a.wrapping_mul(b),
                    SBin::MulHiU => ((a as u64 * b as u64) >> 32) as u32,
                    SBin::And => a & b,
                    SBin::Or => a | b,
                    SBin::Xor => a ^ b,
                    SBin::Shl => {
                        if b < 32 {
                            a << b
                        } else {
                            0
                        }
                    }
                    SBin::Shrl => {
                        if b < 32 {
                            a >> b
                        } else {
                            0
                        }
                    }
                    SBin::Shra => ((a as i32) >> b.min(31)) as u32,
                    SBin::MinU => a.min(b),
                    SBin::MaxU => a.max(b),
                    SBin::MinS => sa.min(sb) as u32,
                    SBin::MaxS => sa.max(sb) as u32,
                    SBin::DivU => {
                        if b == 0 {
                            u32::MAX
                        } else {
                            a / b
                        }
                    }
                    SBin::RemU => {
                        if b == 0 {
                            a
                        } else {
                            a % b
                        }
                    }
                    SBin::DivS => {
                        if sb == 0 {
                            u32::MAX
                        } else {
                            (sa / sb) as u32
                        }
                    }
                    SBin::RemS => {
                        if sb == 0 {
                            a
                        } else {
                            (sa % sb) as u32
                        }
                    }
                };
                Self::def(i, Value::S(r), effs)
            }
            Op::SCmp { cond, ty, a, b } => {
                let (a, b) = (s!(a), s!(b));
                let r = match ty {
                    CmpTy::S32 => cmp(*cond, a as i32, b as i32),
                    CmpTy::U32 => cmp(*cond, a, b),
                    CmpTy::F32 => fcmp(*cond, f32::from_bits(a), f32::from_bits(b)),
                };
                Self::def(i, Value::P(r), effs)
            }
            Op::SSelect { p, t, f } => {
                let v = if s!(p) != 0 { s!(t) } else { s!(f) };
                Self::def(i, Value::S(v), effs)
            }
            Op::PNeg(a) => Self::def(i, Value::P(s!(a) == 0), effs),
            Op::PBin { op, a, b } => {
                let (a, b) = (s!(a) != 0, s!(b) != 0);
                let r = match op {
                    PBin::Or => a || b,
                    PBin::And => a && b,
                    PBin::Nand => !(a && b),
                    PBin::Nor => !(a || b),
                    PBin::Xor => a != b,
                };
                Self::def(i, Value::P(r), effs)
            }
            Op::SCvtSF(a) => Self::def(i, Value::S((s!(a) as i32 as f32).to_bits()), effs),
            Op::SCvtFS(a) => {
                let v = f32::from_bits(s!(a)) as f64;
                let t = if v.is_nan() { 0.0 } else { v.trunc() };
                Self::def(
                    i,
                    Value::S(t.clamp(-2147483648.0, 2147483647.0) as i64 as u32),
                    effs,
                )
            }
            Op::SLd(a) => {
                let at = self.addr(c, a) as usize;
                self.access(c, MemKind::Smem, c, at as u32, 1, false, findings);
                Self::def(i, Value::S(self.cores[c].smem.get(at)), effs)
            }
            Op::SSt { addr, val } => {
                let at = self.addr(c, addr);
                self.access(c, MemKind::Smem, c, at, 1, true, findings);
                effs.push(Eff::Smem(at, s!(val)))
            }
            Op::HaltErr { msg } => {
                let m = format!(
                    "shalt.err {}",
                    &msg[msg.find("shalt.err").unwrap_or(0)..]
                        .chars()
                        .take(200)
                        .collect::<String>()
                );
                let (kind, sig) = if msg.contains("BoundsCheck") {
                    ("violation", "bounds-check")
                } else if msg.contains("Semaphore") {
                    ("violation", "exit-check")
                } else {
                    ("violation", "halt")
                };
                // ASSUME(TPU-S-EXIT)
                findings.push(Finding {
                    kind: kind.into(),
                    sig: format!(
                        "{sig}:core{c}@{:#x}",
                        self.kernel.bundles[self.cores[c].pc].addr
                    ),
                    msg: format!("core {c} halts: {}", self.kernel.describe(self.cores[c].pc)),
                    trace: Vec::new(),
                    pcs: vec![self.cores[c].pc as u32],
                    gb300: None,
                    perm_rows: vec![],
                });
                self.cores[c].halted = Some(m);
            }
            Op::Halt => self.cores[c].done = true,
            Op::Branch { .. } => {}
            Op::Phi { init, back, vector } => {
                if self.cores[c].phi_snap.is_none() {
                    let snap = self.cores[c].regs.clone();
                    self.cores[c].phi_snap = Some(Box::new(snap));
                }
                let src = if self.cores[c].backedge { back } else { init };
                PHI_READ.with(|p| p.set(true));
                let v = if *vector {
                    self.vval(c, src).map(Value::V)
                } else {
                    self.sval(c, src).map(Value::S)
                };
                PHI_READ.with(|p| p.set(false));
                let v = v?;
                let v = match (i.def.map(|d| d.file), v) {
                    (Some(File::P), Value::S(x)) => Value::P(x != 0),
                    (_, v) => v,
                };
                Self::def(i, v, effs)
            }
            Op::SyncAdd { flag, val } => {
                let f = s!(flag);
                self.check_flag(c, f, &i.mn, findings);
                effs.push(Eff::FlagAdd(f, s!(val) as i32 as i64))
            }
            Op::SyncSet { flag, val } => {
                let f = s!(flag);
                self.check_flag(c, f, &i.mn, findings);
                effs.push(Eff::FlagSet(f, s!(val) as i32 as i64))
            }
            Op::SyncMov { flag } => {
                let f = s!(flag);
                self.check_flag(c, f, &i.mn, findings);
                effs.push(Eff::SfrfMov(f))
            }
            Op::Wait { flag, .. } => {
                let f = s!(flag);
                self.check_flag(c, f, &i.mn, findings);
                self.acquire(c, f);
            }
            Op::DmaDone { flag, val } => {
                let f = s!(flag);
                self.check_flag(c, f, &i.mn, findings);
                self.acquire(c, f);
                effs.push(Eff::FlagAdd(f, -(s!(val) as i32 as i64)))
            }
            Op::RemoteSyncAdd { addr, val } => {
                // [13:0] flag, bit 14 core in chip, bit 15 set, [30:17] chip.
                // ASSUME(TPU-L-REMOTE-FLAG-ADDR)
                let a = s!(addr);
                let (flag, core, chip) = (a & 0x3FFF, (a >> 14) & 1, (a >> 17) & 0x3FFF);
                let Some(&to) = self.devmap.get(&(chip, core)) else {
                    return unimpl(format!("remote flag address {a:#x}"));
                };
                self.check_flag(c, flag, &i.mn, findings);
                effs.push(Eff::Remote {
                    to,
                    flag,
                    n: s!(val) as i32 as i64,
                })
            }
            Op::Dma {
                kind,
                src,
                dst,
                size,
                flag,
                stride,
            } => {
                let st = match stride {
                    Some([a, b, k]) => Some((s!(a), s!(b), s!(k))),
                    None => None,
                };
                let (s0, d0, n, f) = (s!(src), s!(dst), s!(size), s!(flag));
                self.check_flag(c, f, &i.mn, findings);
                self.issue_dma(c, *kind, s0, d0, n, f, None, c, st);
            }
            Op::DmaGeneral {
                kind,
                src,
                dst,
                size,
                dflag,
                sflag,
                overrides,
            } => {
                let (s0, d0, n, df, sf, ov) = (
                    s!(src),
                    s!(dst),
                    s!(size),
                    s!(dflag),
                    s!(sflag),
                    s!(overrides),
                );
                // dst_syncflagno bit 14 = remote, bit 13 = core in chip; overrides low bits =
                // chip. ASSUME(TPU-L-DMA-GENERAL-DEV)
                let to = if df & 0x4000 != 0 {
                    let (core, chip) = ((df >> 13) & 1, ov & 0x3FFF);
                    match self.devmap.get(&(chip, core)) {
                        Some(&t) => t,
                        None => {
                            return unimpl(format!(
                                "dma.general remote target chip {chip} core {core}"
                            ));
                        }
                    }
                } else {
                    c
                };
                self.check_flag(c, df & 0x1FFF, &i.mn, findings);
                self.check_flag(c, sf, &i.mn, findings);
                self.issue_dma(c, *kind, s0, d0, n, df & 0x1FFF, Some(sf), to, None);
            }
            Op::VLd { addr, sm, stride } => {
                let row = self.addr(c, addr) as usize;
                let rows: Vec<usize> = match stride {
                    // strided load. ASSUME(TPU-L-VLD-STRIDE)
                    Some((ss, sps)) => (0..8)
                        .map(|j| row + (j / *sps as usize) * *ss as usize + j % *sps as usize)
                        .collect(),
                    None => (row..row + 8).collect(),
                };
                let core = &self.cores[c];
                let mut a = [0u32; 1024];
                for (s, &r) in rows.iter().enumerate() {
                    // masked-off sublanes read as 0 (unobserved). ASSUME(TPU-S-VLD-MASKED)
                    if sm & (1 << s) != 0 {
                        for l in 0..128 {
                            a[s * 128 + l] = core.vmem.get(r * 128 + l);
                        }
                    }
                }
                for (s, &r) in rows.iter().enumerate() {
                    if sm & (1 << s) != 0 {
                        self.access(c, MemKind::Vmem, c, r as u32, 1, false, findings);
                    }
                }
                Self::def(i, Value::V(Arc::new(a)), effs)
            }
            Op::VSt { addr, sm, vm, src } => {
                let row = self.addr(c, addr);
                let val = v!(src);
                let mut m = Box::new([false; 1024]);
                let lane = vm.map(|r| self.mask(c, r));
                for s in 0..8 {
                    for l in 0..128 {
                        m[s * 128 + l] =
                            sm & (1 << s) != 0 && lane.as_ref().is_none_or(|x| x[s * 128 + l] != 0);
                    }
                }
                for s in 0..8 {
                    if sm & (1 << s) != 0 {
                        self.access(c, MemKind::Vmem, c, row + s as u32, 1, true, findings);
                    }
                }
                let mask = if m.iter().all(|&x| x) { None } else { Some(m) };
                effs.push(Eff::Vmem { row, mask, val })
            }
            Op::VMov(a) => Self::def(i, Value::V(v!(a)), effs),
            Op::VStv(a) => {
                let x = s!(a);
                Self::def(i, Value::V(Arc::new([x; 1024])), effs)
            }
            Op::VF32 { op, a, b } => {
                let (a, b) = (v!(a), v!(b));
                let op = vpu(*op);
                Self::def(i, Value::V(new_v(|k| num.vpu_f32(op, a[k], b[k]))), effs)
            }
            Op::VBf16 { op, a, b } => {
                let (a, b) = (v!(a), v!(b));
                let op = vpu(*op);
                let r = new_v(|k| {
                    let mut out = 0;
                    for sh in [0, 16] {
                        out |= (num.vpu_bf16(op, (a[k] >> sh) & 0xFFFF, (b[k] >> sh) & 0xFFFF)
                            & 0xFFFF)
                            << sh;
                    }
                    out
                });
                Self::def(i, Value::V(r), effs)
            }
            Op::VInt { op, a, b } => {
                let (a, b) = (v!(a), v!(b));
                let r = new_v(|k| {
                    let (x, y) = (a[k], b[k]);
                    match op {
                        IOp::Add => x.wrapping_add(y),
                        IOp::Sub => x.wrapping_sub(y),
                        IOp::Mul => x.wrapping_mul(y),
                        IOp::And => x & y,
                        IOp::Or => x | y,
                        IOp::Xor => x ^ y,
                        IOp::Shl => x << (y & 31),
                        IOp::Shrl => x >> (y & 31),
                        IOp::Shra => ((x as i32) >> ((y as i32) & 31)) as u32,
                    }
                });
                Self::def(i, Value::V(r), effs)
            }
            Op::VCmp { cond, ty, a, b } => {
                let (a, b) = (v!(a), v!(b));
                let r = new_v(|k| {
                    (match ty {
                        CmpTy::F32 => {
                            fcmp(*cond, f32::from_bits(ftz(a[k])), f32::from_bits(ftz(b[k])))
                        }
                        CmpTy::S32 => cmp(*cond, a[k] as i32, b[k] as i32),
                        CmpTy::U32 => cmp(*cond, a[k], b[k]),
                    }) as u32
                });
                Self::def(i, Value::V(r), effs)
            }
            Op::VMask { op, a, b } => {
                let a = self.mask(c, *a);
                let b = b.map(|r| self.mask(c, r));
                let r = new_v(|k| {
                    let (x, y) = (a[k] != 0, b.as_ref().is_some_and(|b| b[k] != 0));
                    (match op {
                        MOp::And => x && y,
                        MOp::Or => x || y,
                        MOp::Xor => x != y,
                        MOp::Not => !x,
                    }) as u32
                });
                Self::def(i, Value::V(r), effs)
            }
            Op::VSel { m, t, f } => {
                let (m, t, f) = (self.mask(c, *m), v!(t), v!(f));
                Self::def(
                    i,
                    Value::V(new_v(|k| if m[k] != 0 { t[k] } else { f[k] })),
                    effs,
                )
            }
            Op::VTrunc(a) => {
                let a = v!(a);
                let r = new_v(|k| {
                    let x = f32::from_bits(ftz(a[k]));
                    if x.is_nan() {
                        CANON_NAN
                    } else {
                        x.trunc().to_bits()
                    }
                });
                Self::def(i, Value::V(r), effs)
            }
            Op::VCvtFS(a) | Op::VCvtTZ(a) => {
                let a = v!(a);
                let tz = matches!(i.op, Op::VCvtTZ(_));
                let r = new_v(|k| {
                    let x = f32::from_bits(ftz(a[k])) as f64;
                    let x = if x.is_nan() {
                        0.0
                    } else if tz {
                        x.trunc()
                    } else {
                        x
                    };
                    let x = x.clamp(-2147483648.0, 2147483647.0);
                    (x as i64) as i32 as u32
                });
                Self::def(i, Value::V(r), effs)
            }
            Op::VCvtSF(a) => {
                let a = v!(a);
                Self::def(i, Value::V(new_v(|k| (a[k] as i32 as f32).to_bits())), effs)
            }
            Op::VPack { hi, lo, int } => {
                let (hi, lo) = (v!(hi), v!(lo));
                let r = new_v(|k| {
                    let (s, l) = (k / 128, k % 128);
                    let src = if s < 4 { &lo } else { &hi };
                    let s2 = (s % 4) * 2;
                    let rnd = |x: u32| if *int { x & 0xFFFF } else { num.bf16_round(x) };
                    rnd(src[s2 * 128 + l]) | (rnd(src[(s2 + 1) * 128 + l]) << 16)
                });
                Self::def(i, Value::V(r), effs)
            }
            Op::VUnpack { a, high, int } => {
                let a = v!(a);
                let base = if *high { 4 } else { 0 };
                let r = new_v(|k| {
                    let (s, l) = (k / 128, k % 128);
                    let w = a[(base + s / 2) * 128 + l];
                    let h = if s % 2 == 0 { w & 0xFFFF } else { w >> 16 };
                    if *int {
                        h as u16 as i16 as i32 as u32
                    } else {
                        num.bf16_unpack(h)
                    }
                });
                Self::def(i, Value::V(r), effs)
            }
            Op::VUnpackS8 { a, high } => {
                // int8 -> bf16 is exact (|x| <= 128). ASSUME(TPU-L-S8-PACK)
                let a = v!(a);
                let base = if *high { 4 } else { 0 };
                let row = |r: usize, l: usize| {
                    let b = (a[(base + r / 4) * 128 + l] >> (8 * (r % 4))) as u8 as i8;
                    (b as f32).to_bits() >> 16
                };
                let r = new_v(|k| {
                    let (s, l) = (k / 128, k % 128);
                    row(2 * s, l) | (row(2 * s + 1, l) << 16)
                });
                Self::def(i, Value::V(r), effs)
            }
            Op::VCombine { a, b, high } => {
                let (a, b) = (v!(a), v!(b));
                let off = if *high { 512 } else { 0 };
                Self::def(
                    i,
                    Value::V(new_v(|k| {
                        if k < 512 {
                            a[off + k]
                        } else {
                            b[off + k - 512]
                        }
                    })),
                    effs,
                )
            }
            Op::VLaneSeq => Self::def(i, Value::V(new_v(|k| k as u32)), effs),
            Op::VPermSlane { a, pat } => {
                // pattern bytes j = (j << 3) | src: broadcast of sublane src (review_mxu_ops3 bat/lay
                // bitwise; reading src as a half-sublane index gives 384 wrong). Other patterns are
                // unobserved. ASSUME(TPU-L-VPERM-SLANE)
                let (a, p) = (v!(a), s!(pat));
                let b = p.to_le_bytes();
                let src = (b[0] & 7) as usize;
                if (0..4).any(|j| b[j] != (8 * j as u8) | src as u8) {
                    return unimpl(format!("vperm.slane pattern {p:#x}"));
                }
                Self::def(i, Value::V(new_v(|x| a[src * 128 + x % 128])), effs)
            }
            Op::VUnpackS8S32 { a, q } => {
                // int8 rows 8q..8q+7 of the T(32,128)(4,1) packing (row 4s+i in byte i of sublane
                // s) sign-extended to one s32 vreg (review_mxu_ops3 lay bitwise; byte q of every
                // sublane gives 6,616 wrong). ASSUME(TPU-L-S8-UNPACK32)
                let (a, q) = (v!(a), *q as usize);
                let r = new_v(|x| {
                    let (r, l) = (x / 128, x % 128);
                    let (s, byte) = (2 * q + r / 4, r % 4);
                    (a[s * 128 + l] >> (8 * byte)) as u8 as i8 as i32 as u32
                });
                Self::def(i, Value::V(r), effs)
            }
            Op::VRotSlane { a, k } => {
                // out[s] = in[(s + k) mod 8]: opposite to jnp.roll (roll probe). ASSUME(TPU-L-ROT-DIR)
                let (a, k) = (v!(a), s!(k) as usize % 8);
                Self::def(
                    i,
                    Value::V(new_v(|x| a[(((x / 128) + k) % 8) * 128 + x % 128])),
                    effs,
                )
            }
            Op::VRotLane { unit, a, k } => {
                let (a, k) = (v!(a), s!(k) as usize % 128);
                let r = new_v(|x| a[(x / 128) * 128 + (x % 128 + 128 - k) % 128]);
                effs.push(Eff::PushV(Fifo::Xlu(*unit), r))
            }
            Op::XLane { unit, op, a } => {
                let a = v!(a);
                let sums: Vec<u32> = (0..8)
                    .map(|s| num.xlane(*op, &a[s * 128..s * 128 + 128]))
                    .collect();
                effs.push(Eff::PushV(Fifo::Xlu(*unit), new_v(|k| sums[k / 128])))
            }
            Op::VXpose {
                unit,
                phase,
                a,
                width,
            } => {
                if *width != 128 {
                    return unimpl("vxpose width");
                }
                effs.push(Eff::Xpose(*unit, *phase, v!(a)))
            }
            Op::VCmask(a) => {
                // imm = l_end<<13 | s_end<<10 | l_start<<3 | s_start. ASSUME(TPU-L-VCMASK-ENC)
                let imm = s!(a);
                let (s0, l0, s1, l1) = (
                    imm & 7,
                    (imm >> 3) & 127,
                    (imm >> 10) & 7,
                    (imm >> 13) & 127,
                );
                let r = new_v(|k| {
                    let (s, l) = ((k / 128) as u32, (k % 128) as u32);
                    (s >= s0 && s <= s1 && l >= l0 && l <= l1) as u32
                });
                Self::def(i, Value::V(r), effs)
            }
            Op::VPush(a) => {
                // element (0,0). ASSUME(TPU-L-VPUSH-ELT)
                let x = v!(a)[0];
                effs.push(Eff::PushS(Fifo::V2s, x))
            }
            Op::VTos(a) => {
                let x = v!(a)[0];
                Self::def(i, Value::S(x), effs)
            }
            Op::Eup { f, a, fused_pop } => {
                let a = v!(a);
                let r = if i.mn.contains(".bf16") {
                    // packed bf16: per 16-bit half. ASSUME(TPU-N-EUP-BF16)
                    let mut out = [0u32; 1024];
                    for (k, o) in out.iter_mut().enumerate() {
                        for sh in [0, 16] {
                            let Some(h) = num.eup_bf16(*f, (a[k] >> sh) & 0xFFFF) else {
                                return unimpl(format!("{}: no bitwise model", i.mn));
                            };
                            *o |= (h & 0xFFFF) << sh;
                        }
                    }
                    Arc::new(out)
                } else {
                    if !num.eup_modeled(*f) {
                        // EUP f32 functions without a bitwise model. ASSUME(TPU-N-EUP-F32)
                        return unimpl(format!(
                            "{}: no bitwise model of this EUP f32 function",
                            i.mn
                        ));
                    }
                    new_v(|k| num.eup(*f, a[k]))
                };
                if *fused_pop {
                    Self::def(i, Value::V(r), effs)
                } else {
                    effs.push(Eff::PushV(Fifo::Eup, r))
                }
            }
            Op::Pop(f) => {
                let core = &mut self.cores[c];
                let v = match f {
                    Fifo::Sfrf => core.sfrf.pop_front().map(Value::S),
                    Fifo::V2s => core.v2s.pop_front().map(Value::S),
                    Fifo::Eup => core.eup.pop_front().map(Value::V),
                    Fifo::Xlu(u) => core.xlu[*u as usize].pop_front().map(Value::V),
                };
                let Some(v) = v else {
                    return Err(SimErr::Halt(format!("{} on an empty FIFO", i.mn)));
                };
                Self::def(i, v, effs)
            }
            Op::MxuSubr { unit, a } => effs.push(Eff::MxuSubrF(*unit, unpack_bf16_rows(&v!(a)))),
            Op::MxuPush { unit, a } => {
                // form from the mnemonic: vmatpush1 / vmatpush3 [.xpose]. ASSUME(TPU-N-MXU-COMBINE)
                let mode = if i.mn.contains(".xpose") {
                    if i.mn.starts_with("vmatpush3") { 5 } else { 4 }
                } else if i.mn.starts_with("vmatpush3") {
                    3
                } else {
                    1
                };
                effs.push(Eff::MxuPushF(*unit, unpack_bf16_rows(&v!(a)), mode))
            }
            Op::MxuPrep { unit, a } => effs.push(Eff::MxuPrep(*unit, v!(a))),
            Op::MxuMatmul {
                unit,
                a,
                slot,
                f32,
                latch,
            } => effs.push(Eff::MxuMatmul {
                unit: *unit,
                x: v!(a),
                slot: *slot,
                f32: *f32,
                latch: *latch,
            }),
            Op::MxuPop { unit, slot } => {
                // returns the entry (canonical NaN) and empties it; popping an empty entry faults
                let Some(v) = self.cores[c].mxu[*unit as usize].mrb.remove(slot) else {
                    return Err(SimErr::Halt(format!("{}: empty result slot", i.mn)));
                };
                let v = new_v(|k| {
                    if f32::from_bits(v[k]).is_nan() {
                        CANON_NAN
                    } else {
                        v[k]
                    }
                });
                Self::def(i, Value::V(v), effs)
            }
            Op::OMxuPush { unit, a, hi, bf16 } => {
                let rows = if *bf16 {
                    match a {
                        Val::Imm(_) => vec![0u32; 16 * 128],
                        _ => unpack_bf16_rows(&v!(a)),
                    }
                } else {
                    v!(a).iter().map(|&x| num.bf16_round(x)).collect()
                };
                effs.push(Eff::MxuStage {
                    unit: *unit,
                    hi: *hi,
                    rows,
                })
            }
            Op::OMxuMatmul { unit, a, f32 } => effs.push(Eff::OMxuMatmul {
                unit: *unit,
                x: v!(a),
                f32: *f32,
            }),
            Op::OMxuPop { unit } => {
                let Some(v) = self.cores[c].mxu[*unit as usize].fifo.pop_front() else {
                    return Err(SimErr::Halt(format!("{}: empty result FIFO", i.mn)));
                };
                Self::def(i, Value::V(v), effs)
            }
        }
        Ok(())
    }

    fn commit(&mut self, c: usize, e: Eff) -> R<()> {
        let num = self.num;
        match e {
            Eff::Reg(r, v) => {
                let regs = &mut self.cores[c].regs;
                let i = r.idx as usize;
                match (r.file, v) {
                    (File::S, Value::S(x)) => set_at(&mut regs.s, i, x),
                    (File::S, Value::P(b)) => set_at(&mut regs.s, i, b as u32),
                    (File::P, Value::P(b)) => set_at(&mut regs.p, i, b),
                    (File::P, Value::S(x)) => set_at(&mut regs.p, i, x != 0),
                    (File::V, Value::V(x)) => set_at(&mut regs.v, i, Some(x)),
                    (File::Vm, Value::V(x)) => set_at(&mut regs.vm, i, Some(x)),
                    (f, _) => {
                        return unimpl(format!("write of the wrong kind to a {f:?} register"));
                    }
                }
            }
            Eff::Vmem { row, mask, val } => {
                let m = &mut self.cores[c].vmem;
                let base = row as usize * 128;
                let data: Vec<u32> = match &mask {
                    None => val.to_vec(),
                    Some(mk) => (0..1024)
                        .map(|k| if mk[k] { val[k] } else { m.get(base + k) })
                        .collect(),
                };
                m.write(base, &data)?;
            }
            Eff::Smem(a, v) => {
                if a as usize >= SMEM_WORDS {
                    return Err(SimErr::Halt(format!("SMEM store out of range {a:#x}")));
                }
                self.cores[c].smem.set(a as usize, v)
            }
            Eff::FlagSet(f, v) => {
                self.release(c, f);
                self.cores[c].flags.insert(f, v as i32 as i64);
                self.cores[c].set_flags.insert(f);
            }
            Eff::FlagAdd(f, v) => {
                self.release(c, f);
                if let Err(m) = self.cores[c].flag_add_checked(f, v) {
                    let at = self
                        .kernel
                        .bundles
                        .get(self.cores[c].pc)
                        .map(|b| b.addr)
                        .unwrap_or(0);
                    return Err(SimErr::Halt(format!("sem-overflow core {c}@{at:#x}: {m}")));
                }
            }
            Eff::SfrfMov(f) => {
                self.acquire(c, f);
                let v = self.cores[c].flag(f) as u32;
                self.cores[c].sfrf.push_back(v)
            }
            Eff::PushV(f, v) => match f {
                Fifo::Eup => self.cores[c].eup.push_back(v),
                Fifo::Xlu(u) => self.cores[c].xlu[u as usize].push_back(v),
                _ => return unimpl("vector push to a scalar FIFO"),
            },
            Eff::PushS(f, v) => match f {
                Fifo::V2s => self.cores[c].v2s.push_back(v),
                Fifo::Sfrf => self.cores[c].sfrf.push_back(v),
                _ => return unimpl("scalar push to a vector FIFO"),
            },
            Eff::Xpose(u, phase, v) => {
                let core = &mut self.cores[c];
                let st = &mut core.xpose[u as usize];
                if phase == XposePhase::Start {
                    st.clear();
                }
                st.push(v);
                if phase == XposePhase::End {
                    let n = st.len();
                    if n != 16 {
                        return unimpl(format!("vxpose of {n} vregs"));
                    }
                    let mtx: Vec<u32> = st.iter().flat_map(|x| x.iter().copied()).collect();
                    st.clear();
                    for blk in 0..16 {
                        let r = new_v(|k| {
                            let (s, l) = (blk * 8 + k / 128, k % 128);
                            mtx[l * 128 + s]
                        });
                        core.xlu[u as usize].push_back(r);
                    }
                }
            }
            Eff::Remote { to, flag, n } => {
                if self.explore {
                    let vc = self.cores[c].vc.clone();
                    vc_tick(&mut self.cores[c].vc, c as u32);
                    self.remote.push(RemoteMsg {
                        from: c,
                        to,
                        flag,
                        n,
                        pc: self.cores[c].pc,
                        vc,
                    });
                } else {
                    // delivered at issue (deterministic mode, as llosim.py); overflow of a
                    // remote add is untested: modelled as the local trap. ASSUME(TPU-S-REM-OVF)
                    if let Err(m) = self.cores[to].flag_add_checked(flag, n) {
                        return Err(SimErr::Halt(format!(
                            "sem-overflow remote add from core {c}: {m}"
                        )));
                    }
                }
            }
            Eff::MxuStage { unit, hi, rows } => {
                let m = &mut self.cores[c].mxu[unit as usize];
                if hi {
                    m.stage_hi.push(rows)
                } else {
                    m.stage_lo.push(rows)
                }
            }
            Eff::MxuSubrF(u, rows) => self.cores[c].mxu[u as usize].stage.push_back(rows),
            Eff::MxuPushF(u, rows, mode) => {
                let m = &mut self.cores[c].mxu[u as usize];
                if m.pend_mode != 0 && m.pend_mode != mode {
                    return unimpl("mixed vmatpush forms in one weight generation");
                }
                m.pend_mode = mode;
                let hi = m.stage.pop_front();
                m.pend.push((rows, hi));
            }
            Eff::MxuPrep(u, v) => self.cores[c].mxu[u as usize].prep = Some(v),
            Eff::MxuMatmul {
                unit,
                x,
                slot,
                f32,
                latch,
            } => {
                // notes/numerics.md (e). ASSUME(TPU-L-MXU-DATAFLOW, TPU-N-MXU-COMBINE)
                let m = &mut self.cores[c].mxu[unit as usize];
                if latch {
                    latch_gmr(m)?;
                }
                let Some(g) = m.gmr.clone() else {
                    return unimpl("matmul before any weights were latched");
                };
                let (mode, k, lo, hi) = (g.0, g.1, &g.2, &g.3);
                let rows = if f32 { 8 } else { 16 };
                let lhs = |v: Option<&V>| -> Vec<u32> {
                    match v {
                        None => vec![0; rows * 128],
                        Some(v) if f32 => v.iter().map(|&b| num.bf16_round(b)).collect(),
                        Some(v) => unpack_bf16_rows(v),
                    }
                };
                let (x0, x1) = (lhs(Some(&x)), lhs(m.prep.as_ref()));
                m.prep = None;
                let parts = if mode == 3 {
                    if k > 128 {
                        return unimpl("vmatpush3 generation with K > 128");
                    }
                    let cut = |x: &[u32]| -> Vec<u32> {
                        (0..rows)
                            .flat_map(|r| x[r * 128..r * 128 + k].iter().copied())
                            .collect()
                    };
                    [
                        num.mxu_pass(&cut(&x0), lo, rows, k, 128),
                        num.mxu_pass(&cut(&x1), hi, rows, k, 128),
                    ]
                } else {
                    let xx: Vec<u32> = (0..rows)
                        .flat_map(|r| (0..k).map(move |c| (r, c)))
                        .map(|(r, c)| {
                            if c < 128 {
                                x0[r * 128 + c]
                            } else {
                                x1[r * 128 + c - 128]
                            }
                        })
                        .collect();
                    [
                        num.mxu_pass(&xx, lo, rows, k, 128),
                        num.mxu_pass(&xx, hi, rows, k, 128),
                    ]
                };
                let mut writes: Vec<(u32, V)> = Vec::new();
                if f32 {
                    writes.push((slot, new_v(|i| parts[0][i])));
                    writes.push((slot + 1, new_v(|i| parts[1][i])));
                } else {
                    writes.push((slot, new_v(|i| parts[0][i])));
                    writes.push((slot + 1, new_v(|i| parts[1][i])));
                    writes.push((slot + 2, new_v(|i| parts[0][1024 + i])));
                    writes.push((slot + 3, new_v(|i| parts[1][1024 + i])));
                }
                for (e, v) in writes {
                    // a valid (unpopped) entry accumulates with one f32 RNE add
                    let nv = match m.mrb.get(&e) {
                        Some(old) => new_v(|i| num.f32_add(old[i], v[i])),
                        None => v,
                    };
                    m.mrb.insert(e, nv);
                }
            }
            Eff::OMxuMatmul { unit, x, f32 } => {
                let m = &mut self.cores[c].mxu[unit as usize];
                if !m.stage_lo.is_empty() {
                    latch_weights(m);
                }
                let Some(w) = m.w.clone() else {
                    return unimpl("matmul before any weights were pushed");
                };
                let (k, wd) = (w.0, &w.1);
                if f32 {
                    let rnd = |v: &V, s: usize, kk: usize| num.mxu_lhs_f32(v[s * 128 + kk]);
                    let lhs: Vec<u32> = (0..8)
                        .flat_map(|s| (0..k).map(move |kk| (s, kk)))
                        .map(|(s, kk)| {
                            if kk < 128 {
                                rnd(&x, s, kk)
                            } else {
                                m.prep.as_ref().map(|p| rnd(p, s, kk - 128)).unwrap_or(0)
                            }
                        })
                        .collect();
                    let a16: Vec<u32> = lhs.iter().map(|&b| b >> 16).collect();
                    let (wl, wh) = split_w(wd, k);
                    let (p0, p1) = (
                        num.mxu_pass(&a16, &wl, 8, k, 128),
                        num.mxu_pass(&a16, &wh, 8, k, 128),
                    );
                    m.fifo.push_back(new_v(|i| p0[i]));
                    m.fifo.push_back(new_v(|i| p1[i]));
                } else {
                    let r = mxu_bf16(num, &x, m.prep.as_ref(), k, wd);
                    m.fifo.extend(r);
                }
                m.prep = None;
            }
        }
        Ok(())
    }

    // -------------------------------------------------------------- DMA

    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the DMA descriptor fields"
    )]
    fn issue_dma(
        &mut self,
        c: usize,
        kind: DmaKind,
        src: u32,
        dst: u32,
        granules: u32,
        flag: u32,
        src_flag: Option<u32>,
        dst_core: usize,
        stride: Option<(u32, u32, u32)>,
    ) {
        // Actor ids are per issuing core and its own issue count, so states reached by
        // different interleavings name the same DMA the same way.
        let actor = 1000 + (c as u32) * 1_000_000 + self.cores[c].dma_seq;
        self.cores[c].dma_seq += 1;
        self.dma_count += 1;
        let mut vc = VC::new();
        if self.explore {
            vc = self.cores[c].vc.clone();
            vc_join(&mut vc, &vc_single(actor, 1));
            vc_tick(&mut self.cores[c].vc, c as u32);
        }
        self.pending.push(Dma {
            core: c,
            kind,
            src,
            dst,
            nbytes: granules as u64 * 32,
            flag,
            dst_core,
            src_flag,
            stride,
            pc: self.cores[c].pc,
            actor,
            vc,
            latched: None,
        });
    }

    /// The DMA's (src, dst, words) pieces. (src_stride, dst_stride, steps_per_stride): chunks
    /// of steps x 512 B, chunk i from src + i*src_stride to dst + i*dst_stride (granules).
    /// ASSUME(TPU-L-DMA-STRIDE)
    fn dma_pieces(r: &Dma) -> Vec<(u32, u32, usize)> {
        match r.stride {
            Some((ss, ds, steps)) => {
                let chunk = steps as usize * 16;
                let cnt = ((r.nbytes / 32) as usize).div_ceil(chunk.max(1));
                (0..cnt)
                    .map(|k| (r.src + k as u32 * ss, r.dst + k as u32 * ds, chunk * 8))
                    .collect()
            }
            None => vec![(r.src, r.dst, (r.nbytes / 4) as usize)],
        }
    }

    fn dma_who(&self, r: &Dma) -> Arc<str> {
        format!(
            "dma(core{}@{:#x})",
            r.core,
            self.kernel.bundles.get(r.pc).map(|b| b.addr).unwrap_or(0)
        )
        .into()
    }

    /// Read the DMA's source (all pieces) and record the reads.
    fn dma_read(&mut self, r: &Dma, findings: &mut Vec<Finding>) -> R<Vec<Vec<u32>>> {
        let who = self.dma_who(r);
        let mut out = Vec::new();
        for (s, _, n) in Self::dma_pieces(r) {
            out.push(match r.kind.src {
                Space::Hbm => self.hbm_read(s, n)?,
                Space::Vmem => self.cores[r.core].vmem.read(s as usize * 8, n)?,
                Space::Smem => self.cores[r.core].smem.read(s as usize, n)?,
                Space::Sflag => return unimpl("DMA from sync flags"),
            });
            self.dma_access(r, r.kind.src, r.core, s, n, false, &who, findings);
        }
        Ok(out)
    }

    /// Is in-flight DMA `i` waiting for its send-done step (remote copy with a send flag)?
    pub fn dma_needs_send(&self, i: usize) -> bool {
        let r = &self.pending[i];
        r.src_flag.is_some() && r.latched.is_none()
    }

    /// Send-done of in-flight DMA `i`: every source granule has been read (latched) and the
    /// send flag on the issuing core rises by the size (wrapping). After it the source is free.
    /// ASSUME(TPU-S-DMA-SRC-LIVE, TPU-S-DMA-REDUCE)
    pub fn dma_send(&mut self, i: usize, findings: &mut Vec<Finding>) -> R<()> {
        let r = self.pending[i].clone();
        let data = self.dma_read(&r, findings)?;
        let mut h = FastHasher::default();
        data.hash(&mut h);
        let hv = h.finish();
        self.pending[i].latched = Some(Arc::new((data, hv)));
        if let Some(sf) = r.src_flag {
            self.cores[r.core].flag_add_wrap(sf, (r.nbytes / 32) as i64);
            if self.explore {
                join_flag(&mut self.cores[r.core], sf, &r.vc);
            }
        }
        Ok(())
    }

    /// Complete in-flight DMA `i`: (send-done first if due), the destination writes, then the
    /// destination flag add (wrapping). Data moves per 32 B granule in any order on hardware
    /// (TPU-S-DMA-GRANULE); the explorer moves a DMA's data in one step because any access that
    /// could observe a partial state is unordered with the DMA and reported as a race.
    /// Completion order across DMAs is free (TPU-S-DMA-NO-ORDER); the receive flag implies the
    /// data (TPU-S-RDMA-RECV, TPU-S-DMA-DONE). DMA addresses: HBM and VMEM in 32 B granules,
    /// SMEM in words; vector ops address VMEM in 512 B rows, scalar ops SMEM in words; flags by
    /// index. ASSUME(TPU-L-ADDR-UNITS, TPU-S-DMA-REDUCE, TPU-S-DMA-NO-ORDER, TPU-S-RDMA-RECV)
    pub fn complete_dma(&mut self, i: usize, findings: &mut Vec<Finding>) -> R<()> {
        if self.dma_needs_send(i) {
            self.dma_send(i, findings)?;
        }
        let r = self.pending.remove(i);
        let data = match &r.latched {
            Some(l) => l.0.clone(),
            None => self.dma_read(&r, findings)?,
        };
        let who = self.dma_who(&r);
        for ((_, d, n), data) in Self::dma_pieces(&r).into_iter().zip(data) {
            match r.kind.dst {
                Space::Hbm => self.hbm_write(d, &data)?,
                Space::Vmem => self.cores[r.dst_core].vmem.write(d as usize * 8, &data)?,
                Space::Smem => self.cores[r.dst_core].smem.write(d as usize, &data)?,
                Space::Sflag => return unimpl("DMA to sync flags"),
            }
            self.dma_access(&r, r.kind.dst, r.dst_core, d, n, true, &who, findings);
        }
        self.cores[r.dst_core].flag_add_wrap(r.flag, (r.nbytes / 32) as i64);
        if self.explore {
            join_flag(&mut self.cores[r.dst_core], r.flag, &r.vc);
        }
        Ok(())
    }

    /// Sync flags left nonzero at exit, per core: (core, flag, value, kind). Flags persist
    /// across launches, so a leftover corrupts later programs even when the kernel's own exit
    /// check is compiled out (`disable_semaphore_checks`). Dummy flags are excluded.
    /// ASSUME(TPU-S-SEM-PERSIST)
    pub fn unbalanced_flags(&self) -> Vec<(usize, u32, i64, FlagKind)> {
        let mut out = Vec::new();
        for c in &self.cores {
            for (&f, &v) in &c.flags {
                // flags the kernel sets explicitly (vsyncpa convention) are excluded.
                // ASSUME(TPU-S-SYNCPA)
                if v == 0 || c.set_flags.contains(&f) {
                    continue;
                }
                let kind = self.flag_kind(f).unwrap_or(FlagKind::Scratch);
                if kind != FlagKind::Dummy {
                    out.push((c.id, f, v, kind));
                }
            }
        }
        out
    }

    /// Kind of flag `f`, or `None` if no sync-flag allocation covers it.
    pub fn flag_kind(&self, f: u32) -> Option<FlagKind> {
        self.flag_ranges
            .iter()
            .find(|(lo, hi, _)| *lo <= f && f < *hi)
            .map(|x| x.2)
    }

    /// Semaphore indices are not bounds-checked by hardware: an index outside every sync-flag
    /// allocation silently addresses another flag; the verifier reports it.
    /// ASSUME(TPU-S-SEM-INDEX)
    fn check_flag(&self, c: usize, f: u32, what: &str, findings: &mut Vec<Finding>) {
        if self.flag_ranges.is_empty() || self.flag_kind(f).is_some() {
            return;
        }
        let pc = self.cores[c].pc;
        let addr = self.kernel.bundles.get(pc).map(|b| b.addr).unwrap_or(0);
        let sig = format!("sem-index:core{c}@{addr:#x}");
        if findings.iter().any(|x| x.sig == sig) {
            return;
        }
        findings.push(Finding {
            kind: "violation".into(),
            sig,
            msg: format!(
                "core {c}: {what} uses sync flag {f}, outside every sync-flag allocation (not bounds-checked by hardware): {}",
                self.kernel.describe(pc)
            ),
            trace: Vec::new(),
            pcs: vec![pc as u32],
            gb300: None,
            perm_rows: vec![],
        });
    }

    /// Deliver in-flight remote signal `i` (explorer mode). ASSUME(TPU-S-REM-OVF)
    pub fn deliver_remote(&mut self, i: usize) -> R<()> {
        let m = self.remote.remove(i);
        if let Err(e) = self.cores[m.to].flag_add_checked(m.flag, m.n) {
            return Err(SimErr::Halt(format!(
                "sem-overflow remote add from core {}: {e}",
                m.from
            )));
        }
        join_flag(&mut self.cores[m.to], m.flag, &m.vc);
        Ok(())
    }

    fn next_rand(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    /// Run to completion with a DMA completion policy (the loop of tools/llosim.py).
    pub fn run(&mut self, policy: DmaPolicy) -> R<()> {
        if let DmaPolicy::Random(s) = policy {
            self.rng = s.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        }
        let mut sink = Vec::new();
        loop {
            let live: Vec<usize> = (0..self.cores.len())
                .filter(|&c| !self.cores[c].done)
                .collect();
            if live.is_empty() {
                return Ok(());
            }
            let mut progressed = false;
            for c in live {
                if self.step(c, &mut sink)? {
                    progressed = true;
                }
                if let Some(h) = &self.cores[c].halted {
                    return Err(SimErr::Halt(format!("core {c}: {h}")));
                }
                if self.cores[c].steps > self.max_steps {
                    return Err(SimErr::Halt(format!("core {c}: step limit")));
                }
            }
            if !self.pending.is_empty() {
                let go = match policy {
                    DmaPolicy::Eager => true,
                    DmaPolicy::Lazy => !progressed,
                    DmaPolicy::Random(_) => !progressed || self.next_rand() % 10 < 3,
                };
                if go {
                    let i = match policy {
                        DmaPolicy::Random(_) => {
                            (self.next_rand() % self.pending.len() as u64) as usize
                        }
                        _ => 0,
                    };
                    self.complete_dma(i, &mut sink)?;
                    progressed = true;
                }
            }
            if !progressed && self.pending.is_empty() && self.cores.iter().any(|c| !c.done) {
                let why: Vec<String> = self
                    .cores
                    .iter()
                    .filter(|c| !c.done)
                    .map(|c| {
                        format!(
                            "core {} at {:#x}: {}",
                            c.id,
                            self.kernel.bundles.get(c.pc).map(|b| b.addr).unwrap_or(0),
                            c.blocked.clone().unwrap_or_default()
                        )
                    })
                    .collect();
                return Err(SimErr::Deadlock(format!(
                    "all cores blocked: {}",
                    why.join("; ")
                )));
            }
        }
    }

    // -------------------------------------------------------------- explorer support

    /// Wait / flag read: join the flag's clock into the core's.
    fn acquire(&mut self, c: usize, f: u32) {
        if !self.explore {
            return;
        }
        let core = &mut self.cores[c];
        if let Some(v) = core.flag_vc.get(&f).cloned() {
            vc_join(&mut core.vc, &v);
        }
    }

    /// Flag write: the flag's clock learns the core's.
    fn release(&mut self, c: usize, f: u32) {
        if !self.explore {
            return;
        }
        let core = &mut self.cores[c];
        let v = core.vc.clone();
        join_flag(core, f, &v);
        vc_tick(&mut core.vc, c as u32);
    }

    #[expect(clippy::too_many_arguments, reason = "access description")]
    fn record(
        &mut self,
        kind: MemKind,
        owner: usize,
        first: u32,
        n: u32,
        write: bool,
        actor: u32,
        clock: u32,
        vc: &VC,
        who: Arc<str>,
        findings: &mut Vec<Finding>,
    ) {
        let Some(rs) = self.race.as_mut() else { return };
        let (lo, hi, owner) = (first, first + n, owner as u16);
        let hb = |r: &Rec| r.actor == actor || vc_get(vc, r.actor) >= r.clock;
        for r in rs.recs.iter() {
            if r.kind != kind
                || r.owner != owner
                || r.hi <= lo
                || hi <= r.lo
                || !(write || r.write)
                || hb(r)
            {
                continue;
            }
            let (x, y) = if *r.who < *who {
                (r.who.to_string(), who.to_string())
            } else {
                (who.to_string(), r.who.to_string())
            };
            let sig = format!("race:{x}:{y}");
            if !findings.iter().any(|f| f.sig == sig) {
                let row = lo.max(r.lo);
                findings.push(Finding {
                    kind: "race".into(),
                    sig,
                    msg: format!(
                        "{kind:?} row {row:#x} of core {owner}: {who} ({}) unordered with {} ({})",
                        if write { "write" } else { "read" },
                        r.who,
                        if r.write { "write" } else { "read" }
                    ),
                    trace: Vec::new(),
                    pcs: Vec::new(),
                    gb300: None,
                    perm_rows: vec![],
                });
            }
        }
        let recs = Arc::make_mut(&mut rs.recs);
        // A write ordered after an access to rows it covers supersedes it; an access repeated
        // by the same actor keeps only its latest epoch.
        recs.retain(|r| {
            let same = r.kind == kind && r.owner == owner;
            let covered = same && lo <= r.lo && r.hi <= hi;
            !(covered && ((write && hb(r)) || (r.actor == actor && r.write == write)))
        });
        recs.push(Rec {
            kind,
            owner,
            lo,
            hi,
            write,
            actor,
            clock,
            who,
        });
    }

    #[expect(clippy::too_many_arguments, reason = "access description")]
    fn access(
        &mut self,
        c: usize,
        kind: MemKind,
        owner: usize,
        at: u32,
        n: u32,
        write: bool,
        findings: &mut Vec<Finding>,
    ) {
        if self.race.is_none() {
            return;
        }
        let vc = self.cores[c].vc.clone();
        let clock = vc_get(&vc, c as u32);
        let who: Arc<str> = format!(
            "core{c}@{:#x}",
            self.kernel
                .bundles
                .get(self.cores[c].pc)
                .map(|b| b.addr)
                .unwrap_or(0)
        )
        .into();
        self.record(
            kind, owner, at, n, write, c as u32, clock, &vc, who, findings,
        );
    }

    #[expect(clippy::too_many_arguments, reason = "access description")]
    fn dma_access(
        &mut self,
        r: &Dma,
        space: Space,
        owner: usize,
        at: u32,
        nwords: usize,
        write: bool,
        who: &Arc<str>,
        findings: &mut Vec<Finding>,
    ) {
        if self.race.is_none() {
            return;
        }
        let (kind, owner, first, n) = match space {
            // rows of 128 words; DMA VMEM addresses are granules (8 words)
            Space::Vmem => (
                MemKind::Vmem,
                owner,
                at / 16,
                (at as usize * 8 + nwords).div_ceil(128) as u32 - at / 16,
            ),
            Space::Smem => (MemKind::Smem, owner, at, nwords as u32),
            Space::Hbm => (
                MemKind::Hbm,
                0xFFFF,
                at / 16,
                (at as usize * 8 + nwords).div_ceil(128) as u32 - at / 16,
            ),
            Space::Sflag => return,
        };
        self.record(
            kind,
            owner,
            first,
            n,
            write,
            r.actor,
            1,
            &r.vc.clone(),
            who.clone(),
            findings,
        );
    }

    /// Drop race-shadow records every live accessor (each core, each in-flight DMA; future DMAs
    /// start from their issuer's clock) is already ordered after: they can never race again.
    /// Keeps the shadow, and so the fingerprint, canonical.
    pub fn prune_race(&mut self) {
        let Some(rs) = self.race.as_mut() else { return };
        let vcs: Vec<&VC> = self
            .cores
            .iter()
            .map(|c| &c.vc)
            .chain(self.pending.iter().map(|d| &d.vc))
            .collect();
        if rs
            .recs
            .iter()
            .any(|r| vcs.iter().all(|v| vc_get(v, r.actor) >= r.clock))
        {
            Arc::make_mut(&mut rs.recs)
                .retain(|r| !vcs.iter().all(|v| vc_get(v, r.actor) >= r.clock));
        }
    }

    /// 128-bit fingerprint of everything that affects the future, including the
    /// happens-before state (core, flag, DMA and message clocks, and the race shadow): two
    /// states with equal data but different clocks can differ in which later accesses race.
    pub fn fingerprint(&self) -> u128 {
        let mut h1 = FastHasher::default();
        for c in &self.cores {
            c.hash_into(&mut h1);
        }
        // in-flight work as multisets: issue order does not matter
        let mut items: Vec<u64> = self
            .pending
            .iter()
            .map(|d| {
                let mut h = FastHasher::default();
                (
                    d.core, d.src, d.dst, d.nbytes, d.flag, d.dst_core, d.src_flag, d.stride, d.pc,
                )
                    .hash(&mut h);
                d.latched.as_ref().map(|l| l.1).hash(&mut h);
                (d.actor, &d.vc).hash(&mut h);
                h.finish()
            })
            .collect();
        items.sort_unstable();
        items.hash(&mut h1);
        let mut items: Vec<u64> = self
            .remote
            .iter()
            .map(|m| {
                let mut h = FastHasher::default();
                (m.from, m.to, m.flag, m.n, m.pc, &m.vc).hash(&mut h);
                h.finish()
            })
            .collect();
        items.sort_unstable();
        items.hash(&mut h1);
        for (b, m) in &self.hbm {
            h1.write_u32(*b);
            m.hash_into(&mut h1);
        }
        // race shadow, canonical (sorted) so the order accesses were recorded in does not matter
        if let Some(rs) = &self.race {
            let mut recs: Vec<_> = rs
                .recs
                .iter()
                .map(|r| (r.kind, r.owner, r.lo, r.hi, r.write, r.actor, r.clock))
                .collect();
            recs.sort_unstable();
            recs.hash(&mut h1);
        }
        let a = h1.finish();
        let mut h2 = FastHasher::default();
        h2.write_u64(a ^ 0x5bd1_e995);
        h2.write_usize(self.pending.len() * 31 + self.remote.len());
        ((a as u128) << 64) | h2.finish() as u128
    }
}

thread_local! {
    static PHI_READ: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn join_flag(core: &mut Core, f: u32, v: &VC) {
    let e = core.flag_vc.entry(f).or_default();
    vc_join(e, v);
}

fn set_at<T: Clone + Default>(v: &mut Vec<T>, i: usize, x: T) {
    if v.len() <= i {
        v.resize(i + 1, T::default());
    }
    v[i] = x;
}

fn vpu(op: FOp) -> VpuOp {
    match op {
        FOp::Add => VpuOp::Add,
        FOp::Sub => VpuOp::Sub,
        FOp::Mul => VpuOp::Mul,
        FOp::Max => VpuOp::Max,
        FOp::Min => VpuOp::Min,
    }
}

fn cmp<T: PartialOrd>(c: Cond, a: T, b: T) -> bool {
    match c {
        Cond::Eq => a == b,
        Cond::Ne => a != b,
        Cond::Lt => a < b,
        Cond::Le => a <= b,
        Cond::Gt => a > b,
        Cond::Ge => a >= b,
    }
}

fn fcmp(c: Cond, a: f32, b: f32) -> bool {
    cmp(c, a, b)
}

/// bf16 vreg (word = rows 2s | 2s+1 << 16) -> 16 x 128 bf16 bits.
fn unpack_bf16_rows(v: &V) -> Vec<u32> {
    let mut out = vec![0u32; 16 * 128];
    for s in 0..8 {
        for l in 0..128 {
            let w = v[s * 128 + l];
            out[(2 * s) * 128 + l] = w & 0xFFFF;
            out[(2 * s + 1) * 128 + l] = w >> 16;
        }
    }
    out
}

/// Latch the pushes since the last latch into the weights (notes/numerics.md (e)).
fn latch_gmr(m: &mut Mxu) -> R<()> {
    let pend = std::mem::take(&mut m.pend);
    let mode = std::mem::replace(&mut m.pend_mode, 0);
    if mode == 4 {
        // .xpose: push i = 16 output columns x 128 K (B^T), its subr the next 128 K of the same
        // columns: W[:, 16i..16i+16] = [push_i^T ; subr_i^T] (256 K)
        if pend.len() > 16 {
            return unimpl("more than 16 .xpose pushes in one generation");
        }
        let mut w = vec![0u32; 256 * 256];
        for (i, (p, q)) in pend.iter().enumerate() {
            for col in 0..16 {
                for kk in 0..128 {
                    w[kk * 256 + 16 * i + col] = p[col * 128 + kk];
                    w[(128 + kk) * 256 + 16 * i + col] =
                        q.as_ref().map_or(0, |q| q[col * 128 + kk]);
                }
            }
        }
        let lo: Vec<u32> = (0..256)
            .flat_map(|r| w[r * 256..r * 256 + 128].to_vec())
            .collect();
        let hi: Vec<u32> = (0..256)
            .flat_map(|r| w[r * 256 + 128..r * 256 + 256].to_vec())
            .collect();
        m.gmr = Some(Arc::new((1, 256, lo, hi)));
        return Ok(());
    }
    if mode == 5 {
        // vmatpush3 .xpose: push i = 16 output columns x 128 K (B^T tile) of pass a, its subr the
        // same for pass b: W_a[:, 16i..16i+16] = push_i^T, W_b[:, 16i..16i+16] = subr_i^T
        // (notes/numerics.md (e); review_sched nt, review_mxu_ops xposeA/B). ASSUME(TPU-N-MXU-COMBINE)
        if pend.len() > 8 {
            return unimpl(format!(
                "{} pushes in one push3 .xpose generation",
                pend.len()
            ));
        }
        let mut wa = vec![0u32; 128 * 128];
        let mut wb = vec![0u32; 128 * 128];
        for (i, (p, q)) in pend.iter().enumerate() {
            for col in 0..16 {
                for kk in 0..128 {
                    wa[kk * 128 + 16 * i + col] = p[col * 128 + kk];
                    wb[kk * 128 + 16 * i + col] = q.as_ref().map_or(0, |q| q[col * 128 + kk]);
                }
            }
        }
        m.gmr = Some(Arc::new((3, 128, wa, wb)));
        return Ok(());
    }
    if pend.len() > if mode == 3 { 8 } else { 16 } {
        return unimpl(format!(
            "{} pushes in one push{mode} generation",
            pend.len()
        ));
    }
    let k = pend.len() * 16;
    let lo: Vec<u32> = pend.iter().flat_map(|(p, _)| p.iter().copied()).collect();
    let hi: Vec<u32> = pend
        .iter()
        .flat_map(|(_, q)| q.clone().unwrap_or_else(|| vec![0; 16 * 128]))
        .collect();
    m.gmr = Some(Arc::new((if mode == 0 { 1 } else { mode }, k, lo, hi)));
    Ok(())
}

fn latch_weights(m: &mut Mxu) {
    let lo: Vec<u32> = m.stage_lo.concat();
    let k = lo.len() / 128;
    let mut hi: Vec<u32> = m.stage_hi.concat();
    hi.resize(k * 128, 0);
    let mut w = vec![0u32; k * 256];
    for r in 0..k {
        w[r * 256..r * 256 + 128].copy_from_slice(&lo[r * 128..r * 128 + 128]);
        w[r * 256 + 128..r * 256 + 256].copy_from_slice(&hi[r * 128..r * 128 + 128]);
    }
    m.w = Some(Arc::new((k, w)));
    m.stage_lo.clear();
    m.stage_hi.clear();
}

/// bf16 LHS: 16 rows x K 0..127 from `x`, K 128..255 from `prep`; results (rows 0-7, N lo),
/// (rows 0-7, N hi), (rows 8-15, N lo), (rows 8-15, N hi).
/// (K x 256) weights -> the (K x 128) halves.
fn split_w(w: &[u32], k: usize) -> (Vec<u32>, Vec<u32>) {
    let lo = (0..k)
        .flat_map(|r| w[r * 256..r * 256 + 128].to_vec())
        .collect();
    let hi = (0..k)
        .flat_map(|r| w[r * 256 + 128..r * 256 + 256].to_vec())
        .collect();
    (lo, hi)
}

/// Original listings, bf16 LHS: X = [x | prep] (16 rows x K) against both weight halves (one
/// pass each); blocks (rows 0-7, N lo), (0-7, N hi), (8-15, N lo), (8-15, N hi).
fn mxu_bf16(num: &dyn Numerics, x: &V, prep: Option<&V>, k: usize, w: &[u32]) -> Vec<V> {
    let rows = unpack_bf16_rows(x);
    let khi = prep
        .map(unpack_bf16_rows)
        .unwrap_or_else(|| vec![0; 16 * 128]);
    let a16: Vec<u32> = (0..16)
        .flat_map(|r| (0..k).map(move |kk| (r, kk)))
        .map(|(r, kk)| {
            if kk < 128 {
                rows[r * 128 + kk]
            } else {
                khi[r * 128 + kk - 128]
            }
        })
        .collect();
    let (wl, wh) = split_w(w, k);
    let (p0, p1) = (
        num.mxu_pass(&a16, &wl, 16, k, 128),
        num.mxu_pass(&a16, &wh, 16, k, 128),
    );
    vec![
        new_v(|i| p0[i]),
        new_v(|i| p1[i]),
        new_v(|i| p0[1024 + i]),
        new_v(|i| p1[1024 + i]),
    ]
}

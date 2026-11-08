/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Well-synchronization certificate (ASSUMPTIONS.md E-WELLSYNC): decide a kernel's verdict
//! for *every* schedule from *one* deterministic run, or say which synchronization point is
//! not determined (the caller then explores).
//!
//! The run (`run_once`'s schedule) records a log of every synchronization event with the
//! vector clock it happened at. The certificate holds when every event's outcome is a
//! function of the events that happen-before it, and every event that can block or be
//! "overtaken" is ordered by happens-before (HB) against everything that could change it:
//!
//! * **C-PHASE** (mbarriers): every operation that lands in phase k (arrive, expect_tx,
//!   complete_tx from TMA / tcgen05.commit / st.async / CLC) is HB-after every operation of
//!   phase k-1; every expect_tx that is not itself an arrival is HB-before some arrival of its
//!   phase (so the phase cannot complete with tx-count 0 before all bytes are expected); a
//!   final phase whose arrivals all came has a non-negative tx-count.
//! * **C-WAIT** (try_wait / test_wait): a successful wait observed the completion of phase
//!   k; every operation of phase k-1 is HB-before it (it cannot see phase k-2's completion of
//!   the same parity). Upper bound: let h be the first phase from k+1 on that has an operation
//!   after the wait (HB-after the waiter, or on SASS after any lane that ran the wait in the
//!   same step) or does not complete; in another schedule the wait may arrive in any phase up
//!   to h and observes k + 2j (more happens-before only), so it must not arrive at h with h
//!   of its own parity: h - k odd (usually h = k+1, the empty/full handshake). Failed blocking
//!   polls are invisible (E-SPIN). A peek (try_wait without the retry shape) must return the
//!   same result in every schedule, unless its only use (directly or through PLOP3 / ISETP /
//!   VOTE copies, recorded with their other-world values) is one branch that skips a blocking
//!   wait on the same mbarrier and parity (CUTLASS "peek, work, then wait": `skip_equiv`
//!   runs both sides on copies; they must meet with no other effect, and every register they
//!   leave different must be dead there, by a static liveness walk); such a peek is that wait
//!   placed at the branch. A failed peek that is never read is harmless.
//! * **C-GEN** (bar.sync / bar.arrive generations, barrier.cluster generations): every
//!   arrival in generation g is HB-after every arrival of generation g-1 (for barrier.cluster
//!   it suffices that the thread waited on g-1 itself).
//! * **C-ORDER** (values): every pair of conflicting strong accesses (ld/st with a
//!   scope, atomics; found by a sweep in run order, `order.rs`) is HB-ordered, except two atomics of the same commutative operation
//!   (add/or/and/xor, not exch) whose results are unused and which do not acquire; plain
//!   accesses are covered by the run's race detector (any race fails the certificate); the
//!   lanes of one SASS atomic step are ordered by lane order (E-LOCKSTEP; C-WARP keeps them
//!   together);
//!   TMEM allocations / frees / relinquish of one allocator are totally HB-ordered; no CLC
//!   response, no try_cancel.
//! * **C-POLL** (flag polls, E-SPIN poll loops): a poll load may read the flag's HB-latest
//!   write before it or any strong write concurrent with it (the run's read is one of them).
//!   Its loop body is replayed for each candidate on a copy of the machine before the load: an
//!   iteration that failed in the run must fail on every candidate; one that exited must fail
//!   or exit at the same pc with the same live registers and at least the run's clock and
//!   pending releases. The run itself continues with the least clock of the exiting candidates
//!   it has seen when the load runs (a poll that read a later release in the run may read an
//!   earlier one elsewhere). Atomic or partially overlapping flag writes are not checked.
//!   Liveness: for an exiting iteration, every candidate it fails on must be HB-before
//!   (hence coherence-before) an exiting concurrent candidate. Otherwise the failing write
//!   may be coherence-latest and the poll may spin forever (fixture e_poll_any `dl`).
//! * **C-CAS** (compare-and-swap semaphores: E-SPIN poll loops whose poll operation is a CAS,
//!   torch symmetric memory's `while (cas(&w, 0, 1) != 0);` put and `while (cas(&w, 1, 0) !=
//!   1);` wait). A successful CAS (a *link*) reads exactly its compare value and writes its
//!   swap value; which write it read from is what decides its synchronization. The links of
//!   one word that are not already all HB-ordered (those are C-ORDER's) must form a chain
//!   whose order is forced by the values: (1) every link changes the value (compare ≠ swap);
//!   (2) one *kind* (compare → swap) per compare value; (3) the links of one kind are totally
//!   HB-ordered; (4) in the run nothing else writes the word between the first and the last
//!   link (each link read the bytes and synchronization source the previous one left), and
//!   the word's write before the first link (race shadow) is HB-before every link; (5) each
//!   link's iteration, replayed on a copy of the machine before it, leaves the loop on its
//!   compare value and fails on every other value of the chain (the value before it and each
//!   swap value). A CAS-poll attempt that failed in the run must fail its iteration too (its
//!   read is not a link) and must observe no release the thread did not already know (in
//!   another schedule the attempt is not made). Other accesses to the word stay under
//!   C-ORDER (strong) and the race detector (plain). Then in every execution the links
//!   succeed in the run's order: after links 1..i-1 the word holds link i's compare value
//!   (no other write in between); only link i's kind can succeed on it (2), and of that kind
//!   only the earliest link not yet done (3: the later ones are HB-after it); failed attempts
//!   are invisible (E-SPIN, (5)). So link i reads from link i-1 (or the same first write) as
//!   in the run, its compare-and-swap leaves the same registers, and the synchronization
//!   source it reads is the same function (release clock joined along the RMW chain) of
//!   clocks that contain the run's: it synchronizes with at least the run's clocks. Progress:
//!   when E has done links 1..i-1, link i is enabled (the word holds its compare value), so
//!   the R-first event E has not executed can run.
//! * **C-RING** (polls of monotone counters whose observed value steers the program: a ring
//!   buffer's consumer chunks its slots by the tail it sees). A counter is written only by
//!   strong stores of one thread with nondecreasing values. A poll's candidates (every write
//!   from the coherence bound on) are replayed on copies; exiting ones are grouped into
//!   classes by the state they leave; exits must be upward closed (no lost wakeup) and failed
//!   iterations invisible. A poll of one class is decided by the run (its clock lowered to
//!   the earliest member's); a poll of several classes is a choice, and the certificate runs
//!   once per sequence of classes (forcing reads, `Machine::ring`), every run must certify,
//!   and all must leave the same outputs (chunking-obliviousness). See `ring.rs`.
//! * **C-WARP** (lane membership): an instruction whose effect depends on which lanes run it
//!   together is run by lanes that became ready at that pc in one event (one instruction or
//!   one barrier release), and, if more lanes joining would change it (SASS lane collectives
//!   and uniform operations, which run once per lane group; PTX elect / vote / redux /
//!   tcgen05.alloc), lanes of the same warp that run that pc in another group are HB-ordered
//!   with them. SASS per-lane memory / sync instructions only need the first part (the
//!   E-LOCKSTEP join; more lanes only add happens-before); SASS lane-local arithmetic,
//!   branches and BREAK (each lane clears its own mask bit; see `membership`) need neither. BSYNC only parks lanes BSSY recorded; a WARPSYNC/BSYNC release
//!   waits only for lanes whose exit is HB-before every arrival (exits do not re-check
//!   warpsync releases).
//! * The run reports no finding except exit-time `contract` findings (E-BULK-EXIT, a
//!   happens-before check at the end: every schedule's clocks contain the run's, so the run
//!   reports the union). A race breaks the value argument; the rest is left to the explorer.
//!
//! Soundness sketch (the long form is ASSUMPTIONS.md E-WELLSYNC). Let R be the run and E any
//! execution of the same launch. By induction over E's steps: E's next event of each actor is
//! R's next event of that actor, with the same outcome and a vector clock containing R's (a
//! wait may observe a later phase, C-WAIT; every check is monotone in the clocks). Values: a
//! plain read reads the HB-latest write (R race-free and HB at least R's ⇒ no race in E: the
//! detector is precise, since the shadow keeps every access a later write does not cover,
//! `Sh::hw`; ASSUMPTIONS.md E-WELLSYNC "Detector precision"), a strong read is ordered with every write
//! (C-ORDER), atomics' results are unused or ordered, CAS-poll links read from the run's
//! predecessor (C-CAS), warp collectives see the same lanes (C-WARP). Synchronization: C-PHASE makes every operation land in the phase it landed in R
//! (it cannot run before phase k-1 completes, and phase k cannot complete without it), so
//! phase completions join the same clocks; C-WAIT makes every wait observe the same phase
//! (or a later one of the same parity, never blocking where the run did not);
//! C-GEN the same for barrier generations. Progress: every blocking event's enabling
//! condition depends only on events before it in R, and every event that could disable it
//! (a later phase completing) is HB-after it, so the R-first event E has not executed is
//! enabled: E can always continue until it has executed exactly R's events, reaching R's
//! final state. Hence every schedule terminates with R's outputs, findings and HB graph, and
//! the explorer would visit only states equivalent to R's prefixes.
//!
//! HB queries use FastTrack epochs with one refinement: a strong store does not tick its
//! thread's clock, so an event in the same clock window after such a store is "ambiguous" (a
//! reader that acquired the store would look ordered after it) and needs a strictly larger
//! clock.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use ptxv_ir::*;

use crate::exec::Loc;
use crate::explore::Action;
use crate::explore::Verdict;
use crate::machine::*;

mod order;
pub mod ring;

/// The log slot on `Machine` (`Machine::wlog`): empty except in a certificate run. Clones are
/// empty, so speculative copies (poll / retry-stub lookahead, explorer states) record nothing.
#[derive(Default, Debug)]
pub struct LogSlot(pub Option<Box<SyncLog>>);

impl Clone for LogSlot {
    fn clone(&self) -> Self {
        LogSlot(None)
    }
}

impl LogSlot {
    #[inline]
    pub fn is_on(&self) -> bool {
        self.0.is_some()
    }
}

/// One event: who, at which own clock, and its position in the run (program order for one
/// actor).
#[derive(Clone, Copy, Debug)]
struct Ev {
    actor: u32,
    clock: u32,
    /// A strong store earlier in the same clock window (see the module doc).
    amb: bool,
    pc: u32,
    seq: u64,
}

impl Ev {
    /// Does this event happen-before an event of `actor` at run position `seq` whose clock
    /// (before it) is `vc`?
    #[inline]
    fn hb(&self, actor: u32, seq: u64, vc: &VC) -> bool {
        if self.actor == actor {
            self.seq < seq
        } else {
            vc_get(vc, self.actor) >= self.clock + self.amb as u32
        }
    }
}

/// "Everything in this set happens-before": the latest event per actor.
#[derive(Default, Clone, Debug)]
struct Req(BTreeMap<u32, Ev>);

impl Req {
    fn add(&mut self, e: Ev) {
        let need = |x: &Ev| (x.clock + x.amb as u32, x.seq);
        match self.0.get(&e.actor) {
            Some(o) if need(o) >= need(&e) => {}
            _ => {
                self.0.insert(e.actor, e);
            }
        }
    }
    /// The first event of the set that is not HB-before (actor, seq, vc).
    fn missing(&self, actor: u32, seq: u64, vc: &VC) -> Option<Ev> {
        self.0.values().find(|e| !e.hb(actor, seq, vc)).copied()
    }
}

#[derive(Clone, Debug)]
struct OpRec {
    ev: Ev,
    vc: VC,
    darr: i64,
    dtx: i64,
}

#[derive(Clone, Debug)]
struct WaitRec {
    ev: Ev,
    vc: VC,
    /// Witnesses (actor, clock): an event of another actor that knows one of them is after
    /// the wait in every schedule. The waiter after its tick; on SASS also every lane that
    /// ran the instruction with it (C-WARP: they always run it in one step).
    wit: Vec<(u32, u32)>,
    /// Success: the phase whose completion it observed. Failed peek: the pending phase.
    phase: i64,
    success: bool,
    peek: bool,
    /// Peek: index into `SyncLog::peeks`.
    pid: Option<usize>,
}

/// How a peek's result was used before its predicate died.
#[derive(Clone, Debug)]
enum PeekUse {
    /// Never read.
    Unused,
    /// Read once, by a branch that skips a blocking wait on the same mbarrier and parity
    /// (checked by lookahead, `skip_equiv`); `need`: the clock an event of another actor
    /// must know of the peeking thread to be HB-after that branch.
    Skip {
        wit: Vec<(u32, u32)>,
    },
    Bad(String),
}

#[derive(Default, Clone, Debug)]
struct PhaseRec {
    ops: Vec<OpRec>,
    completed: bool,
}

#[derive(Clone, Debug)]
struct MbObj {
    init: Epoch,
    phase0: i64,
    expected: u32,
    phases: BTreeMap<i64, PhaseRec>,
    waits: Vec<WaitRec>,
    inval: Option<(Ev, VC)>,
}

#[derive(Clone, Debug)]
struct Stash {
    cta: u16,
    off: u32,
    ev: Ev,
    vc: VC,
    pending: u32,
    tx: i64,
}

#[derive(Clone, Debug)]
struct Arr {
    ev: Ev,
    vc: VC,
    /// barrier.cluster: the thread waited on the previous generation itself.
    po_ok: bool,
}

/// Lanes of one warp that ran one pc together, consecutive instances with the same lanes.
#[derive(Clone, Debug)]
struct Run {
    mask: u64,
    ts: Vec<usize>,
    first_vc: Vec<VC>,
    first_seq: u64,
    last: Vec<Ev>,
}

#[derive(Clone, Debug)]
struct SAcc {
    key: (u8, u32),
    lo: u64,
    hi: u64,
    ev: Ev,
    vc: VC,
    write: bool,
    /// Atomic: (operation, result used, acquires, key max). Key max: `MaxS` of an operand
    /// with the sign bit clear or `MinU` of one with it set; both are the maximum under one
    /// total order of 32-bit words (sign clear: the word as signed; set: smaller words
    /// first), so any mix of them commutes (CUTLASS's float atomic max).
    rmw: Option<(AtomOp, bool, bool, bool)>,
    /// fp32 atomic adds (C-FADD): the added operand (bits).
    fadd: Option<u32>,
    /// Writes (C-POLL): the bytes written and those they replaced, and the location's
    /// synchronization source before and after (the latter filled in by `post_inst`).
    val: Vec<u8>,
    old: Vec<u8>,
    old_cell: Option<SyncCell>,
    cell: Option<SyncCell>,
    /// Writes: the race shadow's last writes of the bytes before this one (C-CAS: the writes
    /// a CAS chain starts from).
    prev_w: Vec<Epoch>,
    /// The successful compare-and-swap of a CAS poll (C-CAS): the machine before its
    /// instruction, to replay the loop on other values of the word.
    at: Option<Snap>,
    /// SASS atomics: the instruction step (`inst_tok`). Lanes of one step are performed in
    /// lane order (E-LOCKSTEP), and C-WARP keeps them together in every schedule.
    step: Option<u64>,
}

/// A machine copy kept by the log (not printed).
#[derive(Clone)]
struct Snap(Arc<Machine>);

impl std::fmt::Debug for Snap {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Snap")
    }
}

/// One execution of a poll load (E-SPIN flag poll) by one thread, for C-POLL: where it reads,
/// and a copy of the machine just before it, to try the loop on other values.
#[derive(Clone)]
struct PollRec {
    t: usize,
    pc: u32,
    key: (u8, u32),
    lo: u64,
    hi: u64,
    ev: Ev,
    vc: VC,
    at: Arc<Machine>,
    /// Exiting: the thread's clock and pending releases after the load as the run continues
    /// with them (`poll_post`): the least the iteration can synchronize with.
    sync: Option<PSync>,
}

impl std::fmt::Debug for PollRec {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "PollRec(t{} pc{} {:?})", self.t, self.pc, self.ev)
    }
}

#[derive(Clone, Debug)]
struct WaitPre {
    t: usize,
    cta: u16,
    off: u32,
    phase: i64,
    ev: Ev,
    vc: VC,
    peek: bool,
    d: Reg,
    parity: u32,
    init: Epoch,
    phase0: i64,
    expected: u32,
}

/// A thread's peek taint (`SyncLog::ptaint`): the peek, and each register derived from it
/// with its value had the peek returned the other result.
type PeekTaint = (usize, Vec<(Reg, u64)>);

/// A lane before a CAS-poll instruction (`SyncLog::cas_pre`): (thread, pc, clock, pending
/// releases).
type CasPre = (usize, u32, VC, Vec<(u32, VC, Scope)>);

/// The event log of one certificate run.
#[derive(Debug, Default)]
pub struct SyncLog {
    seq: u64,
    nthreads: usize,
    tok: Vec<u64>,
    next_tok: u64,
    /// `next_tok` when the current instruction started.
    inst_tok: u64,
    /// Per thread: own clock at its last strong (non-ticking) store, or u32::MAX.
    last_nt: Vec<u32>,
    mbars: HashMap<(u16, u32), Vec<MbObj>>,
    stash: Option<Stash>,
    pre_waits: Vec<WaitPre>,
    /// Per thread: the peek whose result it holds, and the registers that hold it or a value
    /// derived from it (predicates, or a GPR token `SEL Rd, RZ, 0x1, !P`), each with its
    /// value had the peek returned the other result.
    ptaint: Vec<Option<PeekTaint>>,
    /// Per peek: (mbarrier, parity, how it was used, instruction instance).
    peeks: Vec<((u16, u32), u32, PeekUse, u64)>,
    /// Threads whose peek taint the current instruction ends.
    pre_clear: Vec<usize>,
    /// Tainted registers the current instruction overwrites / derives: (thread, register,
    /// value in the other world).
    pre_kill: Vec<(usize, Reg)>,
    pre_add: Vec<(usize, Reg, u64)>,
    /// Named barrier arrivals: (cta, id) -> generation -> arrivals.
    bars: HashMap<(u16, u32), BTreeMap<u32, Vec<Arr>>>,
    /// Bar arrivals of the current instruction: (thread, cta, id, generation).
    pre_bars: Vec<(usize, u16, u32, u32)>,
    clusters: HashMap<u32, BTreeMap<u32, Vec<Arr>>>,
    /// Per thread: (last barrier.cluster generation arrived in, waited on it).
    cl_last: Vec<Option<(u32, bool)>>,
    runs: HashMap<(u32, u32), Vec<Run>>,
    strong: Vec<SAcc>,
    tmem: BTreeMap<u16, Vec<(Ev, VC)>>,
    polls: Vec<PollRec>,
    /// `seq` when the current instruction started.
    inst_seq: u64,
    /// C-CAS: the machine before the current CAS-poll instruction, and per lane that runs it
    /// (thread, clock, pending releases) before it, to check a failed attempt gains nothing.
    cas_snap: Option<Snap>,
    cas_pre: Vec<CasPre>,
    cas_snaps: usize,
    /// Immediate failures, by key (first message kept).
    fails: BTreeMap<String, String>,
    /// C-RING: poll-loop entries and their iterations (ring.rs).
    ring: ring::RingLog,
}

/// Runs whose lanes change more often than this at one (warp, pc) fail the certificate
/// (the pairwise check would be quadratic).
const MAX_RUNS: usize = 256;
/// Poll executions checked by C-POLL before giving up (each keeps a machine copy).
const MAX_POLLS: usize = 4096;
/// Unordered overlapping strong-access pairs visited before giving up (C-ORDER, order.rs).
const MAX_PAIRS: u64 = 200_000_000;
/// C-ORDER conflicts collected for the dead-read probe (`dead_reads`) before giving up.
const MAX_DEAD_PAIRS: usize = 1 << 14;

impl SyncLog {
    pub fn new(m: &Machine) -> SyncLog {
        let n = m.threads.len();
        SyncLog {
            nthreads: n,
            tok: vec![0; n],
            next_tok: 1,
            last_nt: vec![u32::MAX; n],
            cl_last: vec![None; n],
            ptaint: vec![None; n],
            ..Default::default()
        }
    }

    fn fail(&mut self, key: String, msg: String) {
        self.fails.entry(key).or_insert(msg);
    }

    fn ev(&mut self, actor: u32, vc: &VC, pc: u32) -> Ev {
        self.seq += 1;
        let clock = vc_get(vc, actor);
        let amb = (actor as usize) < self.nthreads && self.last_nt[actor as usize] == clock;
        Ev {
            actor,
            clock,
            amb,
            pc,
            seq: self.seq,
        }
    }

    /// The incarnation of the mbarrier at (cta, off) whose init is `init` (created on first
    /// sight, with the phase it is in then: no completion can precede its first operation).
    fn mbobj(&mut self, cta: u16, off: u32, init: Epoch, phase0: i64, expected: u32) -> &mut MbObj {
        let v = self.mbars.entry((cta, off)).or_default();
        if v.last().is_none_or(|o| o.init != init) {
            v.push(MbObj {
                init,
                phase0,
                expected,
                phases: BTreeMap::new(),
                waits: vec![],
                inval: None,
            });
        }
        v.last_mut().unwrap()
    }
}

fn take(m: &mut Machine) -> Box<SyncLog> {
    m.wlog.0.take().expect("wellsync log")
}

/// Which checks an instruction's lane membership needs: (arrived-together, no concurrent
/// other group).
fn membership(m: &Machine, pc: u32) -> (bool, bool) {
    let op = &m.k.code[pc as usize].op;
    match op {
        Op::WarpSync { .. } | Op::Bsync { .. } => (false, false),
        // SASS BREAK clears each active lane's own bit of the warp's convergence mask and then
        // releases lanes parked at that barrier's BSYNC once every lane still in the mask is
        // parked. Bit l changes only by lane l's own BSSY / BREAK or by a release that lane l
        // is parked in, and the release fires on the first event after which the set is
        // complete, joining only the parked lanes' clocks: the mask, the release point and
        // its clocks are the same however the breaking lanes are grouped (E-BSYNC).
        Op::Break { .. } if m.k.front == Front::Sass => (false, false),
        // SASS per-lane instructions without effects: who runs them together is irrelevant.
        _ if m.k.front == Front::Sass && !inst_uniform(m, pc) && sass_lane_local(op) => {
            (false, false)
        }
        // Lane collectives and uniform operations (run once per lane group): exactly the
        // run's lanes must run them together.
        Op::Elect { .. }
        | Op::VoteAny { .. }
        | Op::MatchAny { .. }
        | Op::VoteAll { .. }
        | Op::Redux { .. }
        | Op::R2Ur { .. }
        | Op::Shfl { .. }
        | Op::StMatrix { .. }
        | Op::Bssy { .. }
        | Op::Break { .. }
        | Op::BMov { .. }
        | Op::Bra { any: true, .. }
        | Op::Bra { div: Some(_), .. }
        | Op::EndCollective { .. }
        | Op::TmemAlloc { .. }
        | Op::TmemAllocRaw { .. }
        | Op::TmemDealloc { .. }
        | Op::TmemFreeRaw { .. }
        | Op::TmemRelinquish
        | Op::TmemLd { .. }
        | Op::TmemSt { .. }
            if m.k.front == Front::Sass =>
        {
            (true, true)
        }
        _ if m.k.front == Front::Sass && inst_uniform(m, pc) => (true, true),
        // Per-lane memory and synchronization: the run's lanes must stay together (the
        // E-LOCKSTEP join of a memory access; a try_wait's lanes witness each other, C-WAIT);
        // more lanes joining only adds happens-before.
        _ if m.k.front == Front::Sass => (true, false),
        // PTX `redux.sync` / `shfl.sync` / `vote.sync` with a subset membermask: each mask
        // group runs on its own once every live lane of its mask has arrived (split_enabled,
        // exec.rs), reading and writing only that group's lanes; which other groups share
        // the step does not change its effect or its violation check (E-WARPCOLL).
        _ if op.member_mask().is_some() => (false, false),
        // PTX `.sync.aligned`: the explorer runs it only with every live lane (split_enabled).
        _ if op.is_warp_aligned() => (false, true),
        Op::Elect { .. }
        | Op::VoteAny { .. }
        | Op::MatchAny { .. }
        | Op::VoteAll { .. }
        | Op::Redux { .. }
        | Op::R2Ur { .. }
        | Op::Bssy { .. }
        | Op::Break { .. }
        | Op::BMov { .. }
        | Op::TmemAlloc { .. }
        | Op::TmemDealloc { .. }
        | Op::TmemRelinquish => (true, true),
        _ => (false, false),
    }
}

fn inst_uniform(m: &Machine, pc: u32) -> bool {
    m.k.code[pc as usize].uniform
}

/// SASS instructions that only read and write the lane's own registers.
fn sass_lane_local(op: &Op) -> bool {
    matches!(
        op,
        Op::Bra {
            any: false,
            div: None,
            ..
        } | Op::Exit
            | Op::Nop
            | Op::Mov { .. }
            | Op::Int { .. }
            | Op::MulWide { .. }
            | Op::Imad { .. }
            | Op::Iadd3 { .. }
            | Op::CvtInt { .. }
            | Op::Flo { .. }
            | Op::Lea { .. }
            | Op::LeaHi { .. }
            | Op::Shf { .. }
            | Op::Lop3 { .. }
            | Op::Plop3 { .. }
            | Op::Setp { .. }
            | Op::Fadd { .. }
            | Op::Ffma { .. }
            | Op::Fmnmx { .. }
            | Op::Fsetp { .. }
            | Op::Hsetp2Bf16 { .. }
            | Op::Fset { .. }
            | Op::Fchk { .. }
            | Op::Fsel { .. }
            | Op::Mufu { .. }
            | Op::R2p { .. }
            | Op::Ffma2 { .. }
            | Op::Hfma2 { .. }
            | Op::Hfma2Bf16 { .. }
            | Op::FfmaBf16 { .. }
            | Op::Hmnmx2Bf16 { .. }
            | Op::Vhmnmx { .. }
            | Op::CvtF32Bf16 { .. }
            | Op::F2fpBf16 { .. }
            | Op::Bfe { .. }
            | Op::I2F { .. }
            | Op::F2I { .. }
            | Op::Pack64 { .. }
            | Op::Unpack64 { .. }
            | Op::Sel { .. }
            | Op::Prmt { .. }
            | Op::Brev { .. }
            | Op::AddMinMax { .. }
            | Op::IntMnmx3 { .. }
            | Op::P2r { .. }
    )
}

/// Hook: before `members` execute the instruction at `pc` (exec.rs `exec`).
pub fn pre_inst(m: &mut Machine, members: &[usize], pc: u32) {
    let mut log = take(m);
    pre_inst_log(m, &mut log, members, pc);
    // An exit releases everything the thread did (the next grid on its GPU joins its final
    // clock): end the window of a strong store that did not tick (module doc), or that store
    // would look unordered with the next grid.
    // Likewise a lane's arrival at a convergence barrier (BSYNC / WARPSYNC): the release
    // joins its current clock into every released lane (E-BSYNC), which is after the store.
    if matches!(
        m.k.code[pc as usize].op,
        Op::Exit | Op::Bsync { .. } | Op::WarpSync { .. }
    ) {
        let g = m.k.code[pc as usize].guard;
        for &t in members {
            let c = vc_get(&m.threads[t].vc, t as u32);
            if m.prd(t, g) && log.last_nt[t] == c {
                vc_tick(&mut m.threads[t].make_mut().vc, t as u32);
            }
        }
    }
    m.wlog.0 = Some(log);
}

fn pre_inst_log(m: &Machine, log: &mut SyncLog, members: &[usize], pc: u32) {
    let k = m.k.clone();
    let inst = &k.code[pc as usize];
    let mut active: Vec<usize> = members
        .iter()
        .copied()
        .filter(|&t| m.prd(t, inst.guard))
        .collect();
    // A uniform instruction runs once, for the first active lane (exec.rs).
    if inst.uniform {
        active.truncate(1);
    }
    log.inst_tok = log.next_tok;
    let (tokc, pairc) = membership(m, pc);
    if tokc && members.iter().any(|&t| log.tok[t] != log.tok[members[0]]) {
        let w = m.threads[members[0]].warp;
        log.fail(
            format!("conv:{pc}"),
            format!(
                "C-WARP: lanes of warp {w} that reached {} by different paths run it together; whether they meet there depends on the schedule",
                k.describe(pc as usize)
            ),
        );
    }
    if pairc && !members.is_empty() {
        let w = m.threads[members[0]].warp;
        let base = m.warp_threads(w).start;
        let mut mask = 0u64;
        for &t in members {
            mask |= 1 << ((t - base) as u64 & 63);
        }
        let evs: Vec<Ev> = members
            .iter()
            .map(|&t| log.ev(t as u32, &m.threads[t].vc, pc))
            .collect();
        let seq = log.seq;
        let runs = log.runs.entry((w, pc)).or_default();
        match runs.last_mut() {
            Some(r) if r.mask == mask => r.last = evs,
            _ => {
                if runs.len() >= MAX_RUNS {
                    log.fail(
                        format!("runs:{pc}"),
                        format!(
                            "C-WARP: warp {w} runs {} with more than {MAX_RUNS} changes of lane groups (not checked)",
                            k.describe(pc as usize)
                        ),
                    );
                } else {
                    runs.push(Run {
                        mask,
                        ts: members.to_vec(),
                        first_vc: members.iter().map(|&t| m.threads[t].vc.clone()).collect(),
                        first_seq: seq,
                        last: evs,
                    });
                }
            }
        }
    }
    peek_uses(m, log, members, &active, pc);
    ring::pre(m, log, members, &active, pc);
    log.inst_seq = log.seq;
    if m.polls.contains_key(&pc) && !active.is_empty() {
        poll_pre(m, log, &active, pc);
        cas_pre(m, log, &active, pc);
    }
    match &inst.op {
        Op::BarSync { id, packed, .. } => {
            for &t in &active {
                let Ok(idv) = m.rd(t, id) else { continue };
                let idv = ptxv_ir::bar_args(idv, None, *packed).0 as u64;
                let cta = m.threads[t].cta;
                let g = m.bar_gen.get(&(cta, idv as u32)).copied().unwrap_or(0);
                let vc = m.threads[t].vc.clone();
                let ev = log.ev(t as u32, &vc, pc);
                log.bars
                    .entry((cta, idv as u32))
                    .or_default()
                    .entry(g)
                    .or_default()
                    .push(Arr {
                        ev,
                        vc,
                        po_ok: false,
                    });
                log.pre_bars.push((t, cta, idv as u32, g));
            }
        }
        Op::ClusterArrive { .. } => {
            for &t in &active {
                if m.threads[t].cl_arrived.is_some() {
                    continue; // reported by the machine
                }
                let cid = m.launch.cluster_of(m.threads[t].cta as u32).0;
                let g = m.clusters.get(&cid).map_or(0, |c| c.gen);
                let po_ok = match log.cl_last[t] {
                    None => g == 0,
                    Some((pg, waited)) => waited && pg + 1 == g,
                };
                log.cl_last[t] = Some((g, false));
                let vc = m.threads[t].vc.clone();
                let ev = log.ev(t as u32, &vc, pc);
                log.clusters
                    .entry(cid)
                    .or_default()
                    .entry(g)
                    .or_default()
                    .push(Arr { ev, vc, po_ok });
            }
        }
        Op::ClusterWait => {
            for &t in &active {
                if let (Some(g), Some((lg, w))) = (m.threads[t].cl_arrived, log.cl_last[t].as_mut())
                {
                    if *lg == g {
                        *w = true;
                    }
                }
            }
        }
        Op::MbarTryWait {
            d,
            addr,
            parity,
            bit31,
        } => {
            for &t in &active {
                let Ok(a) = m.addr(t, addr) else { continue };
                let Ok(Loc::Shared { cta, off }) = m.resolve(t, Space::Shared, a, 8) else {
                    continue;
                };
                let Ok(p) = m.rd(t, parity) else { continue };
                let par = (if *bit31 { (p >> 31) & 1 } else { p & 1 }) as u32;
                let mb = match m.mbars.get(&(cta, off)) {
                    Some(mb) => mb.clone(),
                    None => match m.mbar_decode_mem(cta, off) {
                        Some(mb) => mb,
                        None => continue, // reported by the machine
                    },
                };
                let vc = m.threads[t].vc.clone();
                let ev = log.ev(t as u32, &vc, pc);
                log.pre_waits.push(WaitPre {
                    t,
                    cta,
                    off,
                    phase: mb.phase as i64,
                    ev,
                    vc,
                    peek: m.peeks.contains(&pc),
                    d: *d,
                    parity: par,
                    init: mb.init,
                    phase0: mb.phase as i64,
                    expected: mb.expected,
                });
            }
        }
        Op::TmemAlloc { .. } | Op::TmemDealloc { .. } | Op::TmemRelinquish => {
            if let Some(&t) = active.first() {
                let vc = m.threads[t].vc.clone();
                let ev = log.ev(t as u32, &vc, pc);
                log.tmem.entry(m.threads[t].cta).or_default().push((ev, vc));
            }
        }
        // The lanes of one instruction run in one atomic step, in a fixed order: one event.
        Op::TmemAllocRaw { pair, .. } => {
            if let Some(&t) = active.first() {
                let vc = m.threads[t].vc.clone();
                let ev = log.ev(t as u32, &vc, pc);
                for c in m.alloc_group(m.threads[t].cta, *pair).unwrap_or_default() {
                    log.tmem.entry(c).or_default().push((ev, vc.clone()));
                }
            }
        }
        Op::TmemFreeRaw { .. } => {
            if let Some(&t) = active.first() {
                let vc = m.threads[t].vc.clone();
                let ev = log.ev(t as u32, &vc, pc);
                log.tmem.entry(m.threads[t].cta).or_default().push((ev, vc));
            }
        }
        Op::ClcTryCancel { .. } | Op::ClcQuery { .. } if !active.is_empty() => {
            log.fail(
                format!("clc:{pc}"),
                format!(
                    "C-ORDER: cluster launch control at {}: its response is a scheduler choice",
                    k.describe(pc as usize)
                ),
            );
        }
        Op::Bsync { b } => {
            if let Some(&t0) = active.first() {
                let w = &m.warps[m.threads[t0].warp as usize];
                let base = m.warp_threads(m.threads[t0].warp).start;
                let own = active
                    .iter()
                    .fold(0u32, |x, &t| x | 1 << ((t - base) as u32 & 31));
                if own & !w.bmask[*b as usize] != 0 {
                    log.fail(
                        format!("bsync:{pc}"),
                        format!(
                            "C-WARP: lanes BSSY did not record reach {} (the convergence mask depends on which lanes arrive together)",
                            k.describe(pc as usize)
                        ),
                    );
                }
            }
        }
        _ => {}
    }
}

/// Peek predicates: a read must be the one branch that skips a blocking wait on the peek's
/// mbarrier and parity (CUTLASS's "peek, work, then wait"); anything else marks the peek.
fn peek_uses(m: &Machine, log: &mut SyncLog, members: &[usize], active: &[usize], pc: u32) {
    let k = m.k.clone();
    let inst = &k.code[pc as usize];
    let srcs = inst.op.src_regs();
    let dsts = inst.op.dst_regs();
    let reads_of = |regs: &[(Reg, u64)]| {
        regs.iter()
            .any(|(r, _)| inst.guard.reg == Some(*r) || srcs.contains(r))
    };
    let readers: Vec<usize> = members
        .iter()
        .copied()
        .filter(|&t| {
            log.ptaint[t]
                .as_ref()
                .is_some_and(|(_, regs)| reads_of(regs))
        })
        .collect();
    // Overwritten tainted registers.
    for &t in active {
        if let Some((_, regs)) = &log.ptaint[t] {
            for (r, _) in regs {
                // A try_wait overwrites it too: a blocking one with its own (always true)
                // result, a peek with a fresh taint set after the instruction.
                if dsts.contains(r) {
                    log.pre_kill.push((t, *r));
                }
            }
        }
    }
    if readers.is_empty() {
        return;
    }
    let pid_of = |log: &SyncLog, t: usize| log.ptaint[t].as_ref().unwrap().0;
    let p0 = pid_of(log, readers[0]);
    let (obj, parity, inst0) = (log.peeks[p0].0, log.peeks[p0].1, log.peeks[p0].3);
    // The readers' peeks: one mbarrier and parity, unread so far, and (SASS, or more than
    // one lane) one instruction instance, whose lanes always see the same result.
    let one_peek = readers.iter().all(|&t| {
        let q = &log.peeks[pid_of(log, t)];
        q.0 == obj
            && q.1 == parity
            && matches!(q.2, PeekUse::Unused)
            && (q.3 == inst0 || (k.front == Front::Ptx && readers.len() == 1))
    });
    let bad = |log: &mut SyncLog, why: String| {
        for &t in &readers {
            let pid = pid_of(log, t);
            log.peeks[pid].2 = PeekUse::Bad(why.clone());
            log.pre_clear.push(t);
        }
    };
    let derives = (matches!(
        inst.op,
        Op::Plop3 { .. } | Op::Setp { .. } | Op::VoteAny { .. } | Op::VoteAll { .. }
    ) && !dsts.is_empty()
        && dsts.iter().all(|r| matches!(r.file, File::P | File::UP)))
        // A token register (CUTLASS's producer try-acquire: `SEL R4, RZ, 0x1, !P1`, read
        // at the next iteration's `ISETP.NE.AND P1, PT, R4, 0x1`).
        || matches!(inst.op, Op::Sel { .. }) && dsts.len() == 1 && dsts[0].file == File::R;
    if derives {
        // A predicate computed from the peek result: record its value in the other world
        // (the instruction rerun on a copy with the tainted registers flipped).
        if !one_peek || (k.front == Front::Ptx && readers.len() > 1) {
            bad(
                log,
                format!(
                    "its result is combined across lanes at {}",
                    k.describe(pc as usize)
                ),
            );
            return;
        }
        let mut c = m.clone();
        c.fp = None;
        c.num = None;
        c.gfoot = None;
        c.gown = None;
        for &t in &readers {
            for &(r, v) in &log.ptaint[t].as_ref().unwrap().1 {
                set_taint(&mut c, t, r, v);
            }
        }
        if c.exec_one(members, pc).is_err() {
            bad(
                log,
                format!("deriving from it at {} faults", k.describe(pc as usize)),
            );
            return;
        }
        for &t in &readers {
            for &r in &dsts {
                let v = if matches!(r.file, File::P | File::UP) {
                    c.prd(
                        t,
                        Pred {
                            reg: Some(r),
                            neg: false,
                        },
                    ) as u64
                } else {
                    c.rd(t, &Val::Reg(r, 1)).unwrap_or(0) & 0xffff_ffff
                };
                log.pre_add.push((t, r, v));
            }
        }
        return;
    }
    if !matches!(
        inst.op,
        Op::Bra {
            any: false,
            div: None,
            ..
        }
    ) {
        bad(
            log,
            format!("its result is read by {}", k.describe(pc as usize)),
        );
        return;
    }
    if !one_peek || readers.len() != members.len() {
        bad(
            log,
            format!(
                "its result is read again, or with other lanes, by {}",
                k.describe(pc as usize)
            ),
        );
        return;
    }
    let flips: Vec<(usize, Vec<(Reg, u64)>)> = readers
        .iter()
        .map(|&t| (t, log.ptaint[t].as_ref().unwrap().1.clone()))
        .collect();
    let r = if m.k.front == Front::Ptx {
        // PTX lanes are independent threads: each side of the branch, lane by lane.
        flips
            .iter()
            .try_for_each(|f| skip_equiv(m, &[f.0], pc, std::slice::from_ref(f), obj, parity))
    } else {
        skip_equiv(m, members, pc, &flips, obj, parity)
    };
    // On SASS the lanes that reached the branch together run it as one step.
    let together =
        m.k.front == Front::Sass && members.iter().all(|&t| log.tok[t] == log.tok[members[0]]);
    for &t in &readers {
        let pid = pid_of(log, t);
        log.peeks[pid].2 = match &r {
            Ok(()) => PeekUse::Skip {
                wit: if together {
                    members
                        .iter()
                        .map(|&u| (u as u32, need_after(log, m, u)))
                        .collect()
                } else {
                    vec![(t as u32, need_after(log, m, t))]
                },
            },
            Err(e) => PeekUse::Bad(e.clone()),
        };
        log.pre_clear.push(t);
    }
}

/// Write a peek-tainted register's other-world value on a copy.
fn set_taint(c: &mut Machine, t: usize, r: Reg, v: u64) {
    if matches!(r.file, File::P | File::UP) {
        c.wp(t, r, v != 0);
    } else {
        c.wr(t, r, v, 1);
    }
}

/// Memory or synchronization effects (not allowed where the two sides of a peek branch
/// differ).
fn effectful(op: &Op) -> bool {
    op.is_visible()
        || matches!(
            op,
            Op::Ld { .. }
                | Op::LdReduce { .. }
                | Op::St { .. }
                | Op::Ld256 { .. }
                | Op::St256 { .. }
                | Op::Atom { .. }
                | Op::StAsync { .. }
                | Op::TmaLoad { .. }
                | Op::TmaStore { .. }
                | Op::Mma { .. }
                | Op::TcCp { .. }
                | Op::TcCommit { .. }
                | Op::TmemLd { .. }
                | Op::TmemSt { .. }
                | Op::StMatrix { .. }
                | Op::BulkCommit
                | Op::Fence { .. }
                | Op::MbarInit { .. }
                | Op::MbarInval { .. }
                | Op::ClcQuery { .. }
                | Op::TmapFence { .. }
                | Op::Trap { .. }
                | Op::Call { .. }
                | Op::CallAbs { .. }
                // Lane collectives: their result depends on who runs them together.
                | Op::Elect { .. }
                | Op::VoteAny { .. }
                | Op::MatchAny { .. }
                | Op::VoteAll { .. }
                | Op::Redux { .. }
                | Op::Shfl { .. }
                | Op::R2Ur { .. }
                | Op::Bssy { .. }
                | Op::Break { .. }
                | Op::BMov { .. }
        )
}

const LOOKAHEAD: usize = 4096;

/// One instruction of `members` on a lookahead copy. A blocking try_wait on the peek's
/// mbarrier and parity counts as passed (the condition it waits for is the peek's, checked by
/// C-WAIT); `visible`: other effects are allowed (both copies run the same instruction).
fn look_step(
    c: &mut Machine,
    members: &[usize],
    obj: (u16, u32),
    parity: u32,
    visible: bool,
) -> Result<(), String> {
    let k = c.k.clone();
    let pc = c.threads[members[0]].pc;
    if members
        .iter()
        .any(|&t| c.threads[t].pc != pc || c.threads[t].st != St::Run)
    {
        return Err("the lanes part ways".into());
    }
    let inst = &k.code[pc as usize];
    if let Op::MbarTryWait {
        d,
        addr,
        parity: pv,
        bit31,
    } = &inst.op
    {
        let ours = members.iter().all(|&t| {
            !c.prd(t, inst.guard)
                || (c
                    .addr(t, addr)
                    .ok()
                    .and_then(|a| c.resolve(t, Space::Shared, a, 8).ok())
                    == Some(Loc::Shared {
                        cta: obj.0,
                        off: obj.1,
                    })
                    && c.rd(t, pv)
                        .ok()
                        .map(|p| (if *bit31 { (p >> 31) & 1 } else { p & 1 }) as u32)
                        == Some(parity))
        });
        if ours && !c.peeks.contains(&pc) {
            for &t in members {
                if c.prd(t, inst.guard) {
                    c.wp(t, *d, true);
                }
                c.threads[t].make_mut().pc = pc + 1;
            }
            return Ok(());
        }
    }
    // A predicated-off instruction does nothing (`@!P0 NANOSLEEP` and the `@!P0` re-poll of a
    // retry stub after its try_wait passed).
    let live = members.iter().any(|&t| c.prd(t, inst.guard));
    if live && effectful(&inst.op) && !visible {
        return Err(format!("{} runs on one side only", k.describe(pc as usize)));
    }
    if !c.split_enabled(members, pc) {
        return Err(format!("{} blocks", k.describe(pc as usize)));
    }
    c.exec_one(members, pc)
        .map_err(|e| format!("{e} at {}", k.describe(pc as usize)))?;
    if members.iter().any(|&t| c.threads[t].st != St::Run) {
        return Err(format!("{} stops the lanes", k.describe(pc as usize)));
    }
    Ok(())
}

/// Register locations: (0 per-thread slot | 1 uniform | 2 predicate | 3 uniform predicate,
/// thread or warp, index).
type RegLoc = (u8, usize, u16);

/// Register locations of `members` (and their warp) that differ between two copies.
fn reg_diff(a: &Machine, b: &Machine, members: &[usize]) -> Result<Vec<RegLoc>, String> {
    let mut out = Vec::new();
    for &t in members {
        let (x, y) = (&a.threads[t], &b.threads[t]);
        if x.pc != y.pc || x.pending_ld != y.pending_ld || x.tma_count != y.tma_count {
            return Err("the two sides differ in more than registers".into());
        }
        for i in 0..x.regs.len() {
            let wb = |th: &Thread| th.written[i / 64] >> (i % 64) & 1;
            if x.regs[i] != y.regs[i] || wb(x) != wb(y) {
                out.push((0, t, i as u16));
            }
        }
        for i in 0..(x.preds.len() * 64) {
            if (x.preds[i / 64] ^ y.preds[i / 64] | x.pdef[i / 64] ^ y.pdef[i / 64]) >> (i % 64) & 1
                != 0
            {
                out.push((2, t, i as u16));
            }
        }
        let w = x.warp as usize;
        let (u, v) = (&a.warps[w], &b.warps[w]);
        for i in 0..u.ur.len() {
            if u.ur[i] != v.ur[i]
                || (u.ur_written[i / 64] ^ v.ur_written[i / 64]) >> (i % 64) & 1 != 0
            {
                out.push((1, w, i as u16));
            }
        }
        for i in 0..8 {
            if (u.up ^ v.up | u.up_def ^ v.up_def) >> i & 1 != 0 {
                out.push((3, w, i));
            }
        }
        if u.bmask != v.bmask {
            return Err("the two sides differ in convergence state".into());
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

/// The branch at `pc` reads a peek predicate `p`. Run both outcomes on copies: the paths
/// must meet again with no effect but the blocking wait on the peek's (mbarrier, parity)
/// on one side, and every register in which they then differ must be overwritten before it
/// is read (both copies run on in lockstep).
fn skip_equiv(
    m: &Machine,
    members: &[usize],
    pc: u32,
    flips: &[(usize, Vec<(Reg, u64)>)],
    obj: (u16, u32),
    parity: u32,
) -> Result<(), String> {
    let k = m.k.clone();
    let Op::Bra { target, .. } = k.code[pc as usize].op else {
        unreachable!()
    };
    let fresh = |x: &Machine| {
        let mut c = x.clone();
        c.fp = None;
        c.num = None;
        c.gfoot = None;
        c.gown = None;
        c
    };
    let mut a = fresh(m);
    a.exec_one(members, pc)?;
    // The other world: the peek returned the other result, so every register derived from
    // it holds its other value, and the branch goes the other way.
    let mut b = fresh(m);
    for (t, regs) in flips {
        for &(r, v) in regs {
            set_taint(&mut b, *t, r, v);
        }
    }
    b.exec_one(members, pc)?;
    let want: Vec<u32> = members
        .iter()
        .map(|&t| {
            if a.threads[t].pc == target as u32 {
                pc + 1
            } else {
                target as u32
            }
        })
        .collect();
    if members
        .iter()
        .zip(&want)
        .any(|(&t, &w)| b.threads[t].pc != w)
    {
        return Err("the branch does not go the other way when the peek is flipped".into());
    }
    // The pcs one side passes; the other runs until it reaches one of them.
    let mut seen: HashMap<u32, usize> = HashMap::new();
    {
        let mut c = fresh(&a);
        for i in 0..LOOKAHEAD {
            seen.entry(c.threads[members[0]].pc).or_insert(i);
            if look_step(&mut c, members, obj, parity, false).is_err() {
                break;
            }
        }
    }
    let mut joined = false;
    for _ in 0..LOOKAHEAD {
        if seen.contains_key(&b.threads[members[0]].pc) {
            joined = true;
            break;
        }
        look_step(&mut b, members, obj, parity, false)?;
    }
    if !joined {
        return Err("the two sides of the branch do not meet".into());
    }
    let j = b.threads[members[0]].pc;
    for _ in 0..seen[&j] {
        look_step(&mut a, members, obj, parity, false)?;
    }
    // Every register in which the two sides differ must be dead where they meet: on every
    // control-flow path from there it is overwritten before it is read (static: the values
    // other threads write meanwhile could steer the path anywhere).
    let diff = reg_diff(&a, &b, members)?;
    let mut regs: Vec<Reg> = diff
        .iter()
        .map(|&(kind, _, idx)| Reg {
            file: match kind {
                0 if k.front == Front::Ptx => File::V,
                0 => File::R,
                1 => File::UR,
                2 => File::P,
                _ => File::UP,
            },
            idx,
        })
        .collect();
    regs.sort_by_key(|r| (r.file as u8, r.idx));
    regs.dedup();
    walk_dead(&a, members, obj, parity, j as usize, regs)
}

/// After a peek branch's two sides meet at `start`: are the registers `regs` they leave
/// different dead? The lanes are walked on a copy (`at`), whose register values are the
/// run's, through ALU ops and branches; memory / sync effects with no register result are
/// stepped over (they cannot steer the lanes, and on the copy they could fault or block, the
/// run's other threads not having caught up). A register written with a result from memory
/// (loads, try_wait, TMEM) or from an unknown input becomes unknown; a branch on an unknown
/// predicate forks the walk. A register of `regs` read on any path fails; written (guard known
/// true) it is dead on that path. Where the walk cannot go on (budget, divergent lanes,
/// calls), the static walk decides (`live_at`: the values other threads write could steer
/// the path anywhere). CUTLASS's producer loops need this: the peek's token is rewritten by
/// the next iteration's peek, after loads that steer branches in between.
fn walk_dead(
    a: &Machine,
    members: &[usize],
    obj: (u16, u32),
    parity: u32,
    start: usize,
    regs: Vec<Reg>,
) -> Result<(), String> {
    let k = a.k.clone();
    let fresh = |m: &Machine| {
        let mut c = m.clone();
        c.wlog = LogSlot(None);
        c.fp = None;
        c.num = None;
        c.gfoot = None;
        c.gown = None;
        c
    };
    let real = |r: &Reg| match r.file {
        File::R | File::UR => r.idx != 255,
        File::P | File::UP => r.idx != 7,
        _ => true,
    };
    let fallback = |at: usize, diff: &[Reg]| -> Result<(), String> {
        for &r in diff {
            if let Some(q) = live_at(&k.code, at, r) {
                return Err(format!(
                    "where the paths meet ({}), a register they leave different ({r:?}) may be read at {}",
                    k.describe(start),
                    k.describe(q)
                ));
            }
        }
        Ok(())
    };
    let set_pc = |c: &mut Machine, pc: usize| {
        for &t in members {
            c.threads[t].make_mut().pc = pc as u32;
        }
    };
    // Which bits of each differing register differ (all, at the meeting point): byte
    // permutes, constant shifts and moves carry them bit by bit, and a result in which none
    // differ is the same on both sides. CuTe's `PRMT R0, R15, 0x7632, R0` keeps the old R0's
    // upper half, which the next `SHF.L.U32 R5, R0, 0x10, RZ` shifts out.
    let masks0: Vec<(Reg, u32)> = regs.iter().map(|&r| (r, !0u32)).collect();
    // (copy, unknown registers, registers still to kill, their differing bits, pc)
    type Walk = (Machine, Vec<Reg>, Vec<Reg>, Vec<(Reg, u32)>, usize);
    let mut stack: Vec<Walk> = vec![(fresh(a), vec![], regs, masks0, start)];
    let mut budget = 4 * LOOKAHEAD;
    while let Some((mut c, mut unk, mut diff, mut dm, mut at)) = stack.pop() {
        loop {
            if diff.is_empty() {
                break;
            }
            if budget == 0 || at >= k.code.len() {
                fallback(at, &diff)?;
                break;
            }
            budget -= 1;
            let inst = &k.code[at];
            let srcs = inst.op.src_regs();
            let g = inst.guard.reg;
            let known = g.is_none_or(|r| !unk.contains(&r));
            let on: Vec<bool> = members.iter().map(|&t| c.prd(t, inst.guard)).collect();
            let all_on = known && on.iter().all(|&x| x);
            let all_off = known && on.iter().all(|&x| !x);
            if let Some(r) = diff.iter().find(|r| srcs.contains(r) || g == Some(**r)) {
                let m = |v: &Val| -> Option<u32> {
                    match v {
                        // A 64-bit operand: not tracked (its second register may differ).
                        Val::Reg(q, n) if *n > 1 && q.file != File::V => {
                            let hit = (0..*n as u16).any(|i| {
                                diff.contains(&Reg {
                                    file: q.file,
                                    idx: q.idx + i,
                                })
                            });
                            if hit { None } else { Some(0) }
                        }
                        Val::Reg(q, _) if diff.contains(q) => {
                            if !matches!(q.file, File::R | File::UR) {
                                return None;
                            }
                            Some(dm.iter().find(|x| x.0 == *q).map_or(!0, |x| x.1))
                        }
                        _ => Some(0),
                    }
                };
                let carried: Option<(Reg, u32)> = match &inst.op {
                    Op::Mov { d, a } => m(a).map(|x| (*d, x)),
                    Op::Prmt {
                        d,
                        a,
                        sel: Val::Imm(sl),
                        b,
                    } => match (m(a), m(b)) {
                        (Some(ma), Some(mb)) => {
                            let src = ma as u64 | (mb as u64) << 32;
                            let mut out = 0u32;
                            for i in 0..4 {
                                let n = (sl >> (4 * i)) & 0xf;
                                let byte = (src >> (8 * (n & 7))) & 0xff;
                                let bm = if n & 8 != 0 {
                                    if byte & 0x80 != 0 { 0xff } else { 0 }
                                } else {
                                    byte
                                };
                                out |= (bm as u32) << (8 * i);
                            }
                            Some((*d, out))
                        }
                        _ => None,
                    },
                    Op::Shf {
                        d,
                        lo,
                        sh: Val::Imm(n),
                        hi,
                        right,
                        signed,
                        high,
                        wide,
                        wrap,
                    } => match (m(lo), m(hi)) {
                        (Some(ml), Some(mh)) => {
                            let v = (mh as u64) << 32 | ml as u64;
                            let r = crate::exec::shf_value(
                                v,
                                crate::exec::shf_amount(*n, *wide, *wrap),
                                *right,
                                *signed,
                            );
                            Some((*d, (if *high { r >> 32 } else { r & 0xffff_ffff }) as u32))
                        }
                        _ => None,
                    },
                    _ => None,
                };
                let Some((d, mask)) = carried.filter(|_| {
                    all_on && g.is_none_or(|q| !diff.contains(&q) && !unk.contains(&q))
                }) else {
                    return Err(format!(
                        "where the paths meet ({}), {r:?}, which they leave different, is read at {}",
                        k.describe(start),
                        k.describe(at)
                    ));
                };
                let unknown = srcs.iter().any(|q| unk.contains(q));
                if look_step(&mut c, members, obj, parity, true).is_err() {
                    fallback(at, &diff)?;
                    break;
                }
                diff.retain(|q| *q != d);
                dm.retain(|x| x.0 != d);
                unk.retain(|q| *q != d);
                if mask != 0 && real(&d) {
                    diff.push(d);
                    dm.push((d, mask));
                }
                if unknown {
                    unk.push(d);
                }
                at = c.threads[members[0]].pc as usize;
                continue;
            }
            if known && !all_on && !all_off && matches!(inst.op, Op::Bra { .. } | Op::Exit) {
                fallback(at, &diff)?; // the lanes part ways
                break;
            }
            let dsts: Vec<Reg> = inst.op.dst_regs().into_iter().filter(real).collect();
            if all_on {
                diff.retain(|r| !dsts.contains(r));
                if diff.is_empty() {
                    break;
                }
            }
            match &inst.op {
                Op::Exit if all_on => break, // the lanes end
                Op::Exit if !known => {
                    stack.push((fresh(&c), unk.clone(), diff.clone(), dm.clone(), at + 1));
                    break;
                }
                Op::Bra {
                    target,
                    any: false,
                    tid0_falls: false,
                    cond: None,
                    div: None,
                } if !known => {
                    let mut t = fresh(&c);
                    set_pc(&mut t, *target);
                    stack.push((t, unk.clone(), diff.clone(), dm.clone(), *target));
                    at += 1;
                    set_pc(&mut c, at);
                    continue;
                }
                Op::Call { .. } | Op::CallAbs { .. } | Op::Ret { .. } | Op::Brx { .. } => {
                    fallback(at, &diff)?;
                    break;
                }
                _ => {}
            }
            let from_memory = matches!(
                inst.op,
                Op::Ld { .. }
                    | Op::LdReduce { .. }
                    | Op::Ld256 { .. }
                    | Op::LdLocal { .. }
                    | Op::LdMatrix { .. }
                    | Op::Atom { .. }
                    | Op::MbarTryWait { .. }
                    | Op::TmemLd { .. }
                    | Op::ClcQuery { .. }
                    | Op::Qspc { .. }
            );
            if from_memory || !known || srcs.iter().any(|r| unk.contains(r)) {
                if matches!(inst.op, Op::Bra { .. }) {
                    fallback(at, &diff)?;
                    break;
                }
                if !all_off {
                    unk.extend(dsts.iter().copied());
                }
                at += 1;
                set_pc(&mut c, at);
                continue;
            }
            if effectful(&inst.op) && dsts.is_empty() || all_off {
                at += 1;
                set_pc(&mut c, at);
                continue;
            }
            unk.retain(|r| !dsts.contains(r));
            if look_step(&mut c, members, obj, parity, true).is_err() {
                fallback(at, &diff)?;
                break;
            }
            at = c.threads[members[0]].pc as usize;
        }
    }
    Ok(())
}

/// Is `r` read before it is overwritten on some path from `pc`? Returns a reading pc (or a
/// pc where the control flow is not followed).
fn live_at(code: &[Inst], pc: usize, r: Reg) -> Option<usize> {
    let mut seen = vec![false; code.len() + 1];
    let mut work = vec![pc];
    let mut calls = false;
    while let Some(p) = work.pop() {
        if p >= code.len() || seen[p] {
            continue;
        }
        seen[p] = true;
        let inst = &code[p];
        if inst.guard.reg == Some(r) || inst.op.src_regs().contains(&r) {
            return Some(p);
        }
        let always = inst.guard.is_always();
        if always && inst.op.dst_regs().contains(&r) {
            continue;
        }
        match &inst.op {
            Op::Bra {
                target,
                cond,
                tid0_falls,
                div,
                ..
            } => {
                work.push(*target);
                if !always || cond.is_some() || *tid0_falls || div.is_some() {
                    work.push(p + 1);
                }
            }
            Op::Exit if always => {}
            // A call continues in the callee and, after its return, right after the call
            // (pushed here, so a return itself ends the path; a kill inside the callee is
            // ignored: conservative).
            Op::Call { target } => {
                calls = true;
                work.push(*target);
                work.push(p + 1);
            }
            // A return not reached through a call followed here goes to an unknown caller.
            Op::Ret { .. } if !calls => return Some(p),
            Op::Ret { .. } => {}
            Op::Brx { targets, .. } => work.extend(targets.iter().copied()),
            Op::CallAbs { .. } => return Some(p),
            _ => work.push(p + 1),
        }
    }
    None
}

/// The clock another actor must know of thread `t` (now) to be after its current point.
fn need_after(log: &SyncLog, m: &Machine, t: usize) -> u32 {
    let c = vc_get(&m.threads[t].vc, t as u32);
    c + (log.last_nt[t] == c) as u32
}

/// Is `op` (actor, seq, clock) after the point the witnesses mark?
fn after_wit(wit: &[(u32, u32)], me: u32, seq: u64, actor: u32, oseq: u64, vc: &VC) -> bool {
    if actor == me {
        return oseq > seq;
    }
    wit.iter().any(|&(a, n)| {
        if a == actor {
            oseq > seq
        } else {
            vc_get(vc, a) >= n
        }
    })
}

/// Hook: after the instruction at `pc` ran (`ok`: no fault).
pub fn post_inst(m: &mut Machine, members: &[usize], pc: u32, _ok: bool) {
    let mut log = take(m);
    let nt = log.next_tok;
    log.next_tok += 1;
    for &t in members {
        // Lanes a barrier released during this instruction keep the release's token.
        if log.tok[t] < log.inst_tok {
            log.tok[t] = nt;
        }
    }
    // Only the threads edited here can have been emptied (O(edits), not O(threads)).
    let mut touched = Vec::new();
    for (t, r) in std::mem::take(&mut log.pre_kill) {
        if let Some((_, regs)) = log.ptaint[t].as_mut() {
            regs.retain(|x| x.0 != r);
            touched.push(t);
        }
    }
    for (t, r, v) in std::mem::take(&mut log.pre_add) {
        if let Some((_, regs)) = log.ptaint[t].as_mut() {
            regs.retain(|x| x.0 != r);
            regs.push((r, v));
        }
    }
    for t in std::mem::take(&mut log.pre_clear) {
        log.ptaint[t] = None;
    }
    for t in touched {
        if log.ptaint[t].as_ref().is_some_and(|x| x.1.is_empty()) {
            log.ptaint[t] = None;
        }
    }
    let inst_id = log.seq;
    let from = log.inst_seq;
    poll_post(m, &mut log, from);
    cas_post(m, &mut log, from);
    for a in log.strong.iter_mut().rev() {
        if a.ev.seq <= from {
            break;
        }
        if a.write {
            a.cell = m.syncs.get(&(a.key.0, a.key.1, a.lo)).cloned();
        }
    }
    // SASS: the lanes ran this instruction as one step (C-WARP checks they always do).
    let together = m.k.front == Front::Sass && membership(m, pc).0;
    for w in std::mem::take(&mut log.pre_waits) {
        let th = &m.threads[w.t];
        let success = if w.peek {
            m.prd(
                w.t,
                Pred {
                    reg: Some(w.d),
                    neg: false,
                },
            )
        } else {
            th.pc != pc
        };
        if !success && !w.peek {
            continue;
        }
        let wit = if together {
            members
                .iter()
                .map(|&u| (u as u32, need_after(&log, m, u)))
                .collect()
        } else {
            vec![(w.t as u32, need_after(&log, m, w.t))]
        };
        let pid = w.peek.then(|| {
            log.peeks
                .push(((w.cta, w.off), w.parity, PeekUse::Unused, inst_id));
            log.ptaint[w.t] = Some((log.peeks.len() - 1, vec![(w.d, !success as u64)]));
            log.peeks.len() - 1
        });
        let obj = log.mbobj(w.cta, w.off, w.init, w.phase0, w.expected);
        obj.waits.push(WaitRec {
            ev: w.ev,
            vc: w.vc,
            wit,
            phase: if success { w.phase - 1 } else { w.phase },
            success,
            peek: w.peek,
            pid,
        });
    }
    ring::post(m, &mut log, members, pc);
    // A warp's bar.sync arrivals must all land in the generation they started in.
    for (t, cta, id, g) in std::mem::take(&mut log.pre_bars) {
        let now = m.bar_gen.get(&(cta, id)).copied().unwrap_or(0);
        let landed_later = now > g + 1 || (now == g + 1 && m.threads[t].st == St::Bar);
        if landed_later {
            log.fail(
                format!("bargen:{pc}"),
                format!(
                    "C-GEN: arrivals of one instruction at {} span two generations of named barrier {id}",
                    m.k.describe(pc as usize)
                ),
            );
        }
    }
    m.wlog.0 = Some(log);
}

/// Hook (exec.rs `mbar_atomic`): an operation on the mbarrier at (cta, off) by `actor` at
/// clock `vc` is about to change it (arrive / complete_tx) or observe it (wait, inval).
pub fn mbar_op(m: &mut Machine, actor: u32, vc: &VC, pc: u32, cta: u16, off: u32) {
    let Some(mb) = m.mbars.get(&(cta, off)) else {
        return;
    };
    let (init, phase, expected, pending, tx) =
        (mb.init, mb.phase as i64, mb.expected, mb.pending, mb.tx);
    let thread_op = (actor as usize) < m.threads.len();
    let k = m.k.clone();
    let kind = &k.code[pc as usize].op;
    let mut log = take(m);
    let ev = log.ev(actor, vc, pc);
    match kind {
        Op::MbarTryWait { .. } if thread_op => {}
        Op::MbarInval { .. } if thread_op => {
            log.mbobj(cta, off, init, phase, expected).inval = Some((ev, vc.clone()));
        }
        _ => {
            log.mbobj(cta, off, init, phase, expected);
            log.stash = Some(Stash {
                cta,
                off,
                ev,
                vc: vc.clone(),
                pending,
                tx,
            });
        }
    }
    m.wlog.0 = Some(log);
}

/// Hook (start of exec.rs `mbar_maybe_complete`): the stashed operation changed the mbarrier.
pub fn mbar_update(m: &mut Machine, cta: u16, off: u32) {
    let mb = &m.mbars[&(cta, off)];
    let (init, phase, expected, pending, tx) =
        (mb.init, mb.phase as i64, mb.expected, mb.pending, mb.tx);
    let mut log = take(m);
    match log.stash.take() {
        Some(s) if s.cta == cta && s.off == off => {
            let obj = log.mbobj(cta, off, init, phase, expected);
            let p = obj.phases.entry(phase).or_default();
            p.ops.push(OpRec {
                ev: s.ev,
                vc: s.vc,
                darr: s.pending as i64 - pending as i64,
                dtx: tx - s.tx,
            });
            if pending == 0 && tx == 0 {
                p.completed = true;
            }
        }
        _ => log.fail(
            format!("mbar-unlogged:{cta}:{off}"),
            format!("mbarrier cta{cta}@smem{off:#x} changed by an operation the log missed"),
        ),
    }
    m.wlog.0 = Some(log);
}

/// Hook (exec.rs barrier releases): `threads` were released together (one new token).
pub fn released(m: &mut Machine, threads: &[usize]) {
    let mut log = take(m);
    let nt = log.next_tok;
    log.next_tok += 1;
    for &t in threads {
        log.tok[t] = nt;
    }
    m.wlog.0 = Some(log);
}

/// Hook (exec.rs `warpsync_release`, before it changes anything): `lanes` (live lanes of the
/// mask, all parked) are released. An exited lane of the mask need not have exited
/// HB-before the arrivals: in an order where it exits after them, its exit re-checks the
/// release (`exit_warpsync_release`, E-BSYNC) and frees the same lanes with the same joined
/// clock (an exit adds none), so the order changes no value or happens-before edge.
pub fn warpsync_released(m: &mut Machine, _warp: u32, _mask: u32, lanes: &[usize]) {
    let mut log = take(m);
    let nt = log.next_tok;
    log.next_tok += 1;
    for &t in lanes {
        log.tok[t] = nt;
    }
    m.wlog.0 = Some(log);
}

/// Hook (exec.rs `access_x`): a strong access.
pub fn strong_access(
    m: &mut Machine,
    actor: u32,
    vc: &VC,
    pc: u32,
    loc: Loc,
    n: usize,
    data: Option<&[u8]>,
) {
    let write = data.is_some();
    // Poll loads are C-POLL's (`poll_pre`) and C-RING's (ring.rs), including the reads of
    // blocked polls.
    if !write && (m.polls.contains_key(&pc) || m.ring_head(pc).is_some()) {
        return;
    }
    let (key, lo) = match loc {
        Loc::Shared { cta, off } => ((0u8, cta as u32), off as u64),
        Loc::Global { alloc, off } => ((1u8, alloc as u32), off),
        Loc::Param { .. } => return,
    };
    let (old, old_cell) = if write {
        (
            mem_bytes(m, loc, n),
            m.syncs.get(&(key.0, key.1, lo)).cloned(),
        )
    } else {
        (vec![], None)
    };
    let thread = (actor as usize) < m.threads.len();
    let (rmw, fadd) = match &m.k.code[pc as usize].op {
        Op::Atom { op, d, sync, v, .. } if thread => {
            let x = m.rd(actor as usize, v);
            let sign = x.as_ref().map(|x| x & 0x8000_0000 != 0);
            let key = matches!(
                (op, sign),
                (AtomOp::MaxS, Ok(false)) | (AtomOp::MinU, Ok(true))
            );
            let fadd = (*op == AtomOp::FAddFtz && n == 4)
                .then(|| x.ok().map(|x| x as u32))
                .flatten();
            (Some((*op, d.is_some(), sync.acquires(), key)), fadd)
        }
        _ => (None, None),
    };
    let prev_w: Vec<Epoch> = if write {
        (0..n)
            .filter_map(|i| match loc {
                Loc::Shared { cta, off } => m.smem[cta as usize].sh.get(off as usize + i).w,
                Loc::Global { alloc, off } => m.gmem[alloc].sh.get(off as usize + i).w,
                Loc::Param { .. } => None,
            })
            .collect()
    } else {
        vec![]
    };
    let cas_poll =
        write && matches!(rmw, Some((AtomOp::Cas, _, _, _))) && m.polls.contains_key(&pc);
    let mut log = take(m);
    let at = if cas_poll { log.cas_snap.clone() } else { None };
    let ev = log.ev(actor, vc, pc);
    if write && thread && rmw.is_none() {
        log.last_nt[actor as usize] = ev.clock;
    }
    log.strong.push(SAcc {
        key,
        lo,
        hi: lo + n as u64,
        ev,
        vc: vc.clone(),
        write,
        rmw,
        fadd,
        val: data.map(|d| d.to_vec()).unwrap_or_default(),
        old,
        old_cell,
        cell: None,
        prev_w,
        at,
        step: (m.k.front == Front::Sass && rmw.is_some()).then_some(log.inst_tok),
    });
    m.wlog.0 = Some(log);
}

/// The current bytes at `loc` (no race check).
fn mem_bytes(m: &Machine, loc: Loc, n: usize) -> Vec<u8> {
    match loc {
        Loc::Shared { cta, off } => (0..n)
            .map(|i| *m.smem[cta as usize].bytes.get(off as usize + i))
            .collect(),
        Loc::Global { alloc, off } => (0..n)
            .map(|i| *m.gmem[alloc].bytes.get(off as usize + i))
            .collect(),
        Loc::Param { .. } => vec![],
    }
}

fn set_mem_bytes(m: &mut Machine, loc: Loc, v: &[u8]) {
    let mem = match loc {
        Loc::Shared { cta, off: _ } => &mut m.smem[cta as usize],
        Loc::Global { alloc, off: _ } => &mut m.gmem[alloc],
        Loc::Param { .. } => return,
    };
    let base = match loc {
        Loc::Shared { off, .. } => off as usize,
        Loc::Global { off, .. } => off as usize,
        Loc::Param { .. } => unreachable!(),
    };
    for (i, &b) in v.iter().enumerate() {
        *mem.bytes.get_mut(base + i) = b;
    }
}

// ---------------------------------------------------------------- analysis

/// `s` with thread numbers folded (`t37` -> `t*`), as a deduplication key.
fn lanes_folded(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        let boundary = i == 0 || !(b[i - 1] as char).is_ascii_alphanumeric();
        if b[i] == b't' && boundary && i + 1 < b.len() && b[i + 1].is_ascii_digit() {
            out.push_str("t*");
            i += 1;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            continue;
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

fn describe_ev(m: &Machine, e: &Ev) -> String {
    format!(
        "{} at {}",
        m.actor_name(e.actor),
        m.k.describe(e.pc as usize)
    )
}

/// Check the log against the final machine; returns the reasons the certificate fails.
fn analyze(m: &Machine, log: &SyncLog) -> Vec<String> {
    FADD_NONDET.with(|f| f.borrow_mut().clear());
    let mut why: Vec<String> = log.fails.values().cloned().collect();
    // One reason per condition and instruction: lanes of one warp fail alike.
    let mut seen: std::collections::HashSet<String> = Default::default();
    let mut add = |s: String| {
        if (why.len() < 64 || std::env::var("PTXV_WS_ALL").is_ok()) && seen.insert(lanes_folded(&s))
        {
            why.push(s);
        }
    };
    // C-PHASE / C-WAIT.
    let mut keys: Vec<&(u16, u32)> = log.mbars.keys().collect();
    keys.sort();
    for key in keys {
        let (cta, off) = *key;
        for (inc, o) in log.mbars[key].iter().enumerate() {
            let name = format!(
                "mbarrier cta{cta}@smem{off:#x}{}",
                if inc > 0 {
                    format!(" (incarnation {inc})")
                } else {
                    String::new()
                }
            );
            let reqs: BTreeMap<i64, Req> = o
                .phases
                .iter()
                .map(|(&k, p)| {
                    let mut r = Req::default();
                    for op in &p.ops {
                        r.add(op.ev);
                    }
                    (k, r)
                })
                .collect();
            for (&k, p) in &o.phases {
                if let Some(r) = reqs.get(&(k - 1)) {
                    for op in &p.ops {
                        if let Some(e) = r.missing(op.ev.actor, op.ev.seq, &op.vc) {
                            add(format!(
                                "C-PHASE: {name} phase {k}: {} is not ordered after {} (phase {}): it could land in phase {} in another schedule",
                                describe_ev(m, &op.ev),
                                describe_ev(m, &e),
                                k - 1,
                                k - 1
                            ));
                            break;
                        }
                    }
                } else if k > o.phase0 {
                    add(format!(
                        "C-PHASE: {name} phase {k} has operations but phase {} has none in the log",
                        k - 1
                    ));
                }
                for x in p.ops.iter().filter(|x| x.dtx > 0 && x.darr == 0) {
                    let before = p
                        .ops
                        .iter()
                        .any(|a| a.darr > 0 && x.ev.hb(a.ev.actor, a.ev.seq, &a.vc));
                    if !before {
                        add(format!(
                            "C-PHASE: {name} phase {k}: expect_tx by {} is not ordered before any arrival of the phase (the phase could complete before the bytes are expected)",
                            describe_ev(m, &x.ev)
                        ));
                    }
                }
                let arr: i64 = p.ops.iter().map(|x| x.darr).sum();
                let tx: i64 = p.ops.iter().map(|x| x.dtx).sum();
                if !p.completed && arr >= o.expected as i64 && tx < 0 {
                    add(format!(
                        "C-PHASE: {name} phase {k} ends with every arrival and a negative tx-count ({tx}): another order completes it"
                    ));
                }
            }
            let completed = |k: i64| o.phases.get(&k).is_some_and(|p| p.completed);
            for w in &o.waits {
                let k = w.phase;
                let what = if w.peek {
                    if w.success {
                        "successful peek"
                    } else {
                        "failed peek"
                    }
                } else {
                    "wait"
                };
                // Lower bound: the phase before the one it observed / found pending.
                let lo = k - 1;
                if lo >= o.phase0 {
                    match reqs.get(&lo) {
                        Some(r) if completed(lo) => {
                            if let Some(e) = r.missing(w.ev.actor, w.ev.seq, &w.vc) {
                                add(format!(
                                    "C-WAIT: {name}: {what} by {} (observing phase {k}) is not ordered after {} (phase {lo}): it could observe an older phase of the same parity",
                                    describe_ev(m, &w.ev),
                                    describe_ev(m, &e)
                                ));
                            }
                        }
                        _ => add(format!(
                            "C-WAIT: {name}: {what} by {} relies on phase {lo}, which did not complete in the log",
                            describe_ev(m, &w.ev)
                        )),
                    }
                }
                // Upper bound. The wait may arrive while the current phase is any c up to h,
                // the first phase from `first` on that cannot complete without it (an
                // operation HB-after it) or does not complete at all. A blocking wait arriving
                // at c returns at the first phase of the other parity from c on: it observes
                // a phase k + 2j (only more happens-before than in the run, harmless: every
                // check is monotone in the clocks), and it must not block at h (c = h of
                // its own parity: nobody completes h). A peek's result must not change.
                // A peek whose only use is a branch skipping a blocking wait on the same
                // (mbarrier, parity) is that wait, placed at the branch (a failed one: the run
                // took the wait; in a schedule where it succeeds, the skip is the same wait
                // done early, with more happens-before). A failed peek never read changes
                // nothing; a successful one acquires, so it must be determined.
                let usage = w.pid.map(|i| &log.peeks[i].2);
                let failed_wit = vec![(w.ev.actor, w.ev.clock + 1)];
                let (check, as_wait, wit) = match (w.peek, usage) {
                    (false, _) => (true, true, &w.wit),
                    (true, Some(PeekUse::Skip { wit })) => (w.success, true, wit),
                    (true, Some(PeekUse::Unused)) if !w.success => (false, false, &w.wit),
                    _ => (true, false, if w.success { &w.wit } else { &failed_wit }),
                };
                if !check {
                    continue;
                }
                let first = if w.success { k + 1 } else { k };
                let mut h = first;
                while completed(h)
                    && !o.phases[&h].ops.iter().any(|op| {
                        after_wit(wit, w.ev.actor, w.ev.seq, op.ev.actor, op.ev.seq, &op.vc)
                    })
                {
                    h += 1;
                }
                let bad = if as_wait {
                    (h - k) % 2 == 0
                } else {
                    h != first
                };
                if bad {
                    let why = match usage {
                        Some(PeekUse::Bad(e)) => format!(" ({e})"),
                        _ => String::new(),
                    };
                    add(format!(
                        "C-WAIT: {name}: {what} by {} (phase {k}) is not ordered before any operation of phase {first}{}: in another schedule it {}{why}",
                        describe_ev(m, &w.ev),
                        if h > first {
                            format!("..{}", h - 1)
                        } else {
                            String::new()
                        },
                        if as_wait {
                            "blocks on a phase that cannot complete without it"
                        } else {
                            "returns the other result"
                        }
                    ));
                }
            }
            if let Some((iev, ivc)) = &o.inval {
                let late = o
                    .phases
                    .values()
                    .flat_map(|p| p.ops.iter().map(|x| x.ev))
                    .chain(o.waits.iter().map(|w| w.ev))
                    .find(|e| !e.hb(iev.actor, iev.seq, ivc));
                if let Some(e) = late {
                    add(format!(
                        "C-PHASE: {name}: mbarrier.inval by {} is not ordered after {}",
                        describe_ev(m, iev),
                        describe_ev(m, &e)
                    ));
                }
            }
        }
    }
    // C-GEN: named barriers.
    let mut bkeys: Vec<&(u16, u32)> = log.bars.keys().collect();
    bkeys.sort();
    for key in bkeys {
        let gens = &log.bars[key];
        for (&g, arrs) in gens {
            let Some(prev) = g.checked_sub(1).and_then(|p| gens.get(&p)) else {
                continue;
            };
            let mut r = Req::default();
            for a in prev {
                r.add(a.ev);
            }
            for a in arrs {
                if let Some(e) = r.missing(a.ev.actor, a.ev.seq, &a.vc) {
                    add(format!(
                        "C-GEN: named barrier {} of CTA {} generation {g}: arrival by {} is not ordered after arrival by {} (generation {}): it could land there",
                        key.1,
                        key.0,
                        describe_ev(m, &a.ev),
                        describe_ev(m, &e),
                        g - 1
                    ));
                    break;
                }
            }
        }
    }
    // C-GEN: cluster barriers.
    let mut ckeys: Vec<&u32> = log.clusters.keys().collect();
    ckeys.sort();
    for cid in ckeys {
        let gens = &log.clusters[cid];
        for (&g, arrs) in gens {
            let prev = g.checked_sub(1).and_then(|p| gens.get(&p));
            let mut r = Req::default();
            for a in prev.into_iter().flatten() {
                r.add(a.ev);
            }
            for a in arrs.iter().filter(|a| !a.po_ok) {
                if let Some(e) = r.missing(a.ev.actor, a.ev.seq, &a.vc) {
                    add(format!(
                        "C-GEN: barrier.cluster of cluster {cid} generation {g}: arrival by {} is not ordered after arrival by {}",
                        describe_ev(m, &a.ev),
                        describe_ev(m, &e)
                    ));
                    break;
                }
            }
        }
    }
    // C-ORDER: TMEM allocators.
    for (cta, evs) in &log.tmem {
        for w in evs.windows(2) {
            let (a, (b, bvc)) = (&w[0].0, &w[1]);
            if !a.hb(b.actor, b.seq, bvc) {
                add(format!(
                    "C-ORDER: TMEM allocator of CTA {cta}: {} and {} are not ordered (allocation results depend on the order)",
                    describe_ev(m, a),
                    describe_ev(m, b)
                ));
                break;
            }
        }
    }
    // C-ORDER: strong accesses.
    for s in poll_check(m, log) {
        add(s);
    }
    // C-RING: polls of monotone counters (ring.rs).
    let (ring_why, choices, counters) = ring::check(m, log);
    for s in ring_why {
        add(s);
    }
    // Words plain writes of equal bytes may share unordered (order.rs `pair`).
    let polled: HashSet<((u8, u32), u64)> = log
        .polls
        .iter()
        .flat_map(|p| (p.lo / 4..=(p.hi.max(p.lo + 1) - 1) / 4).map(move |w| (p.key, w)))
        .collect();
    let forced = m.ring.as_ref().is_some_and(|r| !r.force.is_empty());
    let same_ok = (choices.is_empty() && !forced).then_some(&polled);
    RING_OUT.with(|r| *r.borrow_mut() = (choices, counters));
    let (cas_why, cas_ok) = cas_check(m, log);
    for s in cas_why {
        add(s);
    }
    for s in strong_conflicts(m, &log.strong, &cas_ok, same_ok) {
        add(s);
    }
    // C-WARP: concurrent groups at one pc.
    let mut rkeys: Vec<&(u32, u32)> = log.runs.keys().collect();
    rkeys.sort();
    'keys: for key in rkeys {
        let runs = &log.runs[key];
        if runs.len() < 2 {
            continue;
        }
        for (i, a) in runs.iter().enumerate() {
            for b in &runs[i + 1..] {
                if a.mask == b.mask {
                    continue;
                }
                for (xi, &x) in a.ts.iter().enumerate() {
                    for (yi, &y) in b.ts.iter().enumerate() {
                        if x == y || (b.ts.contains(&x) && a.ts.contains(&y)) {
                            continue;
                        }
                        let ex = a.last[xi];
                        if !ex.hb(y as u32, b.first_seq, &b.first_vc[yi]) {
                            add(format!(
                                "C-WARP: warp {} runs {} with different lane groups that are not ordered ({} and {}): in another schedule they run it together",
                                key.0,
                                m.k.describe(key.1 as usize),
                                m.actor_name(x as u32),
                                m.actor_name(y as u32)
                            ));
                            continue 'keys;
                        }
                    }
                }
            }
        }
    }
    why
}

/// `a` synchronizes with no more than `b`.
fn psync_le(m: &Machine, t: usize, a: &PSync, b: &PSync) -> bool {
    vc_le(&a.0, &b.0)
        && a.1.iter().all(|(w, rel, _)| {
            b.1.iter()
                .any(|(w2, rel2, _)| covers(m, t, *w, *w2) && vc_le(rel, rel2))
        })
}

/// A pending release by `w2` reaches thread `t` through every acquire fence that one by
/// `w` does (fences filter pending releases by the writer, `acquire_pending`).
fn covers(m: &Machine, t: usize, w: u32, w2: u32) -> bool {
    w == w2
        || [Scope::Cta, Scope::Cluster, Scope::Gpu, Scope::Sys]
            .iter()
            .all(|&s| !m.scope_includes(s, t as u32, w) || m.scope_includes(s, t as u32, w2))
}

/// `a` is contained in `b`.
fn vc_le(a: &VC, b: &VC) -> bool {
    let mut x = b.clone();
    vc_join(&mut x, a);
    x == *b
}

fn loc_of(key: (u8, u32), lo: u64) -> Loc {
    if key.0 == 0 {
        Loc::Shared {
            cta: key.1 as u16,
            off: lo as u32,
        }
    } else {
        Loc::Global {
            alloc: key.1 as usize,
            off: lo,
        }
    }
}

/// Hook (`pre_inst`): `active` run the poll load at `pc`. Recorded for C-POLL with a copy
/// of the machine.
fn poll_pre(m: &Machine, log: &mut SyncLog, active: &[usize], pc: u32) {
    if log.polls.len() + active.len() > MAX_POLLS {
        log.fail(
            "polls".into(),
            format!("C-POLL: more than {MAX_POLLS} poll executions (not checked)"),
        );
        return;
    }
    let Op::Ld {
        space, bytes, addr, ..
    } = &m.k.code[pc as usize].op
    else {
        return;
    };
    let at = Arc::new(m.clone());
    for &t in active {
        let Ok(loc) = m
            .addr(t, addr)
            .and_then(|a| m.resolve(t, *space, a, *bytes as usize))
        else {
            continue; // the step reports it
        };
        let (key, lo) = match loc {
            Loc::Shared { cta, off } => ((0u8, cta as u32), off as u64),
            Loc::Global { alloc, off } => ((1u8, alloc as u32), off),
            Loc::Param { .. } => continue,
        };
        let vc = m.threads[t].vc.clone();
        let ev = log.ev(t as u32, &vc, pc);
        log.polls.push(PollRec {
            t,
            pc,
            key,
            lo,
            hi: lo + *bytes as u64,
            ev,
            vc,
            at: at.clone(),
            sync: None,
        });
    }
}

/// Hook (`post_inst`): the poll loads this instruction ran (events after `from`). An exiting
/// read may have read a write that releases more than an earlier one it could read in
/// another schedule; the run continues with the least of the clocks the exiting candidates
/// known so far give (concurrent writes before it in the run, the flag's value before them),
/// when one is contained in all the others, so every later check holds for the weakest
/// reading. C-POLL checks the rest (writes after it in the run) against that clock.
fn poll_post(m: &mut Machine, log: &mut SyncLog, from: u64) {
    let n = log.polls.len();
    let first = log
        .polls
        .iter()
        .rposition(|r| r.ev.seq <= from)
        .map_or(0, |i| i + 1);
    for i in first..n {
        let r = log.polls[i].clone();
        let t = r.t;
        // The iteration as the run takes it (only its load has run yet). Candidates are
        // compared after the body (a relaxed poll's acquire fence may be in it); the least
        // one's state after the load is what the run continues with.
        let Ok(Some((run, load))) = poll_try(&r, None) else {
            continue; // it failed (a split where another lane exited); C-POLL checks it
        };
        let exit_pc = run.threads[t].pc;
        let post = |x: &Machine| (x.threads[t].vc.clone(), x.threads[t].acq_pending.clone());
        let mut best = (post(&run), load);
        // Only candidates that leave the run's live registers count (C-RING: one class).
        let code = &m.k.code;
        let (lh, lb) = m.polls[&r.pc];
        let mut written: Vec<Reg> = Vec::new();
        for inst in &code[lh as usize..=lb as usize] {
            written.extend(inst.op.dst_regs());
            if let Op::Ld { d, n, .. } = &inst.op {
                for k in 0..(*n).max(1) as u16 {
                    written.push(Reg {
                        file: d.file,
                        idx: d.idx + k,
                    });
                }
            }
        }
        let val_of = |x: &Machine, g: Reg| -> Option<u64> {
            if matches!(g.file, File::P | File::UP) {
                Some(x.prd(
                    t,
                    Pred {
                        reg: Some(g),
                        neg: false,
                    },
                ) as u64)
            } else {
                x.rd(t, &Val::Reg(g, 1)).ok()
            }
        };
        let live: Vec<Reg> = written
            .iter()
            .copied()
            .filter(|&g| live_at(code, exit_pc as usize, g).is_some())
            .collect();
        let conc = poll_conc(log, &r, |a| a.ev.seq < r.ev.seq);
        let Ok(conc) = conc else {
            continue; // C-POLL reports it
        };
        for (_, v, cell) in poll_cands(&r, &conc, log) {
            let Ok(Some((y, yload))) = poll_try(&r, Some((&v, &cell))) else {
                continue;
            };
            if y.threads[t].pc != exit_pc || live.iter().any(|&g| val_of(&y, g) != val_of(&run, g))
            {
                continue;
            }
            let c = post(&y);
            if psync_le(m, t, &c, &best.0) {
                best = (c, yload);
            }
        }
        let th = m.threads[t].make_mut();
        th.vc = best.1.0;
        th.acq_pending = best.1.1;
        let best = best.0;
        log.polls[i].sync = Some(best);
    }
}

/// The strong writes of `r`'s flag concurrent with it (filtered by `keep`), or why C-POLL
/// does not check them.
fn poll_conc<'a>(
    log: &'a SyncLog,
    r: &PollRec,
    keep: impl Fn(&SAcc) -> bool,
) -> Result<Vec<&'a SAcc>, (Ev, &'static str)> {
    let mut conc = Vec::new();
    for a in log.strong.iter() {
        if !a.write || a.key != r.key || a.hi <= r.lo || r.hi <= a.lo || !keep(a) {
            continue;
        }
        if a.ev.hb(r.t as u32, r.ev.seq, &r.vc) || r.ev.hb(a.ev.actor, a.ev.seq, &a.vc) {
            continue;
        }
        if a.rmw.is_some() {
            return Err((a.ev, "an atomic"));
        }
        // A wider write starting at the flag (ptxas merges a data store into the flag's
        // STS.64) is a candidate for the flag's bytes; the synchronization source is kept per
        // start offset, so other overlaps are not checked.
        if a.lo != r.lo || a.hi < r.hi {
            return Err((a.ev, "a write that overlaps it differently"));
        }
        conc.push(a);
    }
    Ok(conc)
}

/// The values `r` may read besides the run's: (description, bytes, synchronization source).
/// The HB-latest write before the poll is what the first concurrent write the run did
/// before it replaced (with none, it is the current contents, which `poll_try(None)` reads).
fn poll_cands(
    r: &PollRec,
    conc: &[&SAcc],
    _log: &SyncLog,
) -> Vec<(Option<Ev>, Vec<u8>, Option<SyncCell>)> {
    let mut cands = Vec::new();
    let n = (r.hi - r.lo) as usize;
    if let Some(a) = conc
        .iter()
        .filter(|a| a.ev.seq < r.ev.seq)
        .min_by_key(|a| a.ev.seq)
    {
        cands.push((None, a.old[..n].to_vec(), a.old_cell.clone()));
    }
    for a in conc {
        cands.push((Some(a.ev), a.val[..n].to_vec(), a.cell.clone()));
    }
    cands
}

/// Run poll `r`'s loop body (load to back branch) for its thread alone on a copy of the
/// machine before the load, with the flag replaced by `src` (bytes and synchronization
/// source) if given. Some(copy after the body) if the iteration exits the loop.
type PSync = (VC, Vec<(u32, VC, Scope)>);

fn poll_try(
    r: &PollRec,
    src: Option<(&[u8], &Option<SyncCell>)>,
) -> Result<Option<(Machine, PSync)>, String> {
    let (h, b) = r.at.polls[&r.pc];
    let mut m = (*r.at).clone();
    m.fp = None;
    m.num = None;
    if let Some((v, cell)) = src {
        m.ring = None;
        let loc = loc_of(r.key, r.lo);
        set_mem_bytes(&mut m, loc, v);
        let k = (r.key.0, r.key.1, r.lo);
        match cell {
            Some(c) => {
                m.syncs.insert(k, c.clone());
            }
            None => {
                m.syncs.remove(&k);
            }
        }
    }
    let mut p = r.pc;
    let mut at_load = None;
    loop {
        m.exec(&[r.t], p)?;
        if p == r.pc {
            let th = &m.threads[r.t];
            at_load = Some((th.vc.clone(), th.acq_pending.clone()));
        }
        if p == b {
            break;
        }
        p += 1;
    }
    Ok((m.threads[r.t].pc != h).then(|| (m, at_load.unwrap())))
}

/// C-POLL: every execution of a poll load reads the flag's HB-latest write before it (unique:
/// two unordered writes are a race or fail C-ORDER) or a strong write concurrent with it. For
/// each such candidate, the iteration must end as in the run: a failing iteration fails on
/// all of them (the thread cannot leave the loop earlier); an exiting one fails or exits with
/// the same live registers and at least the run's clock and pending releases (a later write
/// may release more; reading it only adds happens-before). Then the poll's outcome, as seen by
/// the rest of the program, is the run's in every schedule (failed iterations are invisible,
/// E-SPIN).
fn poll_check(m: &Machine, log: &SyncLog) -> Vec<String> {
    let mut out = Vec::new();
    let mut done: std::collections::HashSet<(u32, String)> = Default::default();
    for r in &log.polls {
        // A monotone counter's polls are C-RING's (ring.rs).
        let hs = ring::hist(m, log, (r.key.0, r.key.1, r.lo), (r.hi - r.lo) as usize);
        if hs.why_not.is_none() && !hs.writes.is_empty() {
            continue;
        }
        let name = |e: &Ev| describe_ev(m, e);
        let mut fail = |out: &mut Vec<String>, s: String| {
            if out.len() < 8 && done.insert((r.pc, lanes_folded(&s))) {
                out.push(s);
            }
        };
        let conc = match poll_conc(log, r, |_| true) {
            Ok(c) => c,
            Err((e, what)) => {
                fail(
                    &mut out,
                    format!(
                        "C-POLL: the poll at {} may read {}, {what} (not checked)",
                        name(&r.ev),
                        name(&e)
                    ),
                );
                continue;
            }
        };
        let raw = poll_cands(r, &conc, log);
        let srcs: Vec<Option<Ev>> = raw.iter().map(|c| c.0).collect();
        let cands: Vec<(String, Vec<u8>, Option<SyncCell>)> = raw
            .into_iter()
            .map(|(e, v, c)| {
                let what = match e {
                    Some(e) => name(&e),
                    None => "the flag's value before the concurrent writes".into(),
                };
                (what, v, c)
            })
            .collect();
        if cands.is_empty() {
            continue;
        }
        let run = match poll_try(r, None) {
            Ok(x) => x.map(|x| x.0),
            Err(e) => {
                fail(
                    &mut out,
                    format!("C-POLL: replaying the poll at {}: {e}", name(&r.ev)),
                );
                continue;
            }
        };
        let code = &m.k.code;
        let (h, b) = m.polls[&r.pc];
        let mut written: Vec<Reg> = Vec::new();
        for inst in &code[h as usize..=b as usize] {
            written.extend(inst.op.dst_regs());
            if let Op::Ld { d, n, .. } = &inst.op {
                for k in 0..(*n).max(1) as u16 {
                    written.push(Reg {
                        file: d.file,
                        idx: d.idx + k,
                    });
                }
            }
        }
        written.sort_by_key(|r| (r.file as u8, r.idx));
        written.dedup();
        let t = r.t;
        let val_of = |x: &Machine, g: Reg| -> Option<u64> {
            if matches!(g.file, File::P | File::UP) {
                Some(x.prd(
                    t,
                    Pred {
                        reg: Some(g),
                        neg: false,
                    },
                ) as u64)
            } else {
                x.rd(t, &Val::Reg(g, 1)).ok()
            }
        };
        // Whether each candidate's iteration leaves the loop (for the liveness check below).
        let mut exits: Vec<Option<bool>> = vec![None; cands.len()];
        for (ci, (what, v, cell)) in cands.iter().enumerate() {
            let alt = match poll_try(r, Some((v, cell))) {
                Ok(x) => x.map(|x| x.0),
                Err(e) => {
                    fail(
                        &mut out,
                        format!("C-POLL: the poll at {} reading {what}: {e}", name(&r.ev)),
                    );
                    continue;
                }
            };
            exits[ci] = Some(alt.is_some());
            let why = match (&run, &alt) {
                (_, None) => None,
                (None, Some(_)) => {
                    Some("exits the loop where the run's iteration failed".to_string())
                }
                (Some(x), Some(y)) => {
                    let pc = x.threads[t].pc as usize;
                    let mine = (x.threads[t].vc.clone(), x.threads[t].acq_pending.clone());
                    let sync = r.sync.as_ref().unwrap_or(&mine);
                    let theirs = (y.threads[t].vc.clone(), y.threads[t].acq_pending.clone());
                    if y.threads[t].pc as usize != pc {
                        Some("leaves the loop elsewhere".into())
                    } else if let Some(g) = written
                        .iter()
                        .find(|&&g| live_at(code, pc, g).is_some() && val_of(x, g) != val_of(y, g))
                    {
                        Some(format!("leaves {g:?} (read later) different"))
                    } else if !psync_le(m, t, sync, &theirs) {
                        Some("synchronizes with less than the run's read".into())
                    } else {
                        None
                    }
                }
            };
            if let Some(why) = why {
                fail(
                    &mut out,
                    format!(
                        "C-POLL: the poll at {} may read {what} instead (not ordered with it), and then {why}",
                        name(&r.ev)
                    ),
                );
            }
        }
        // Liveness: an iteration that fails on a candidate is invisible only if the poll
        // can still read a value it leaves on. It can be stuck when the failing write may
        // be coherence-latest: every write after it in coherence order must eventually be
        // read (E-SPIN), so we require each failing candidate to be HB-before (hence
        // coherence-before) an exiting concurrent write. The coherence-latest candidate of
        // any execution then exits. The value before the concurrent writes is the poll's
        // HB-latest write (its own past): the race shadow's last write before the first
        // concurrent write the run did before the poll, else before the poll; none is the
        // initial contents (coherence-first).
        if run.is_some() {
            let exiting: Vec<&SAcc> = conc
                .iter()
                .copied()
                .filter(|a| {
                    srcs.iter()
                        .zip(&exits)
                        .any(|(s, x)| s.is_some_and(|s| s.seq == a.ev.seq) && *x == Some(true))
                })
                .collect();
            for (ci, s) in srcs.iter().enumerate() {
                if exits[ci] != Some(false) {
                    continue;
                }
                let later = match s {
                    Some(f) => exiting.iter().any(|a| f.hb(a.ev.actor, a.ev.seq, &a.vc)),
                    None => {
                        let before: Vec<Epoch> = match conc
                            .iter()
                            .filter(|a| a.ev.seq < r.ev.seq)
                            .min_by_key(|a| a.ev.seq)
                        {
                            Some(a) => a.prev_w.clone(),
                            None => {
                                let loc = loc_of(r.key, r.lo);
                                (0..(r.hi - r.lo) as usize)
                                    .filter_map(|i| match loc {
                                        Loc::Shared { cta, off } => {
                                            r.at.smem[cta as usize].sh.get(off as usize + i).w
                                        }
                                        Loc::Global { alloc, off } => {
                                            r.at.gmem[alloc].sh.get(off as usize + i).w
                                        }
                                        Loc::Param { .. } => None,
                                    })
                                    .collect()
                            }
                        };
                        before.iter().all(|e| exiting.iter().any(|a| e.hb(&a.vc)))
                    }
                };
                if !later {
                    fail(
                        &mut out,
                        format!(
                            "C-POLL: the poll at {} may read {} last (no write it leaves on is ordered after it), and then never leaves the loop",
                            name(&r.ev),
                            cands[ci].0
                        ),
                    );
                }
            }
        }
    }
    out
}

/// A word (location, first byte, end) of CAS-poll chains.
type CasWord = ((u8, u32), u64, u64);

/// Hook (`pre_inst`): `active` run the CAS poll at `pc` (an E-SPIN poll loop whose poll
/// operation is a compare-and-swap). Keep a copy of the machine for C-CAS's replays and each
/// lane's clock and pending releases, for `cas_post`.
fn cas_pre(m: &Machine, log: &mut SyncLog, active: &[usize], pc: u32) {
    if !matches!(
        m.k.code[pc as usize].op,
        Op::Atom {
            op: AtomOp::Cas,
            ..
        }
    ) {
        return;
    }
    if log.cas_snaps >= MAX_POLLS {
        log.fail(
            "cas-snaps".into(),
            format!("C-CAS: more than {MAX_POLLS} CAS poll steps (not checked)"),
        );
        return;
    }
    log.cas_snaps += 1;
    log.cas_snap = Some(Snap(Arc::new(m.clone())));
    for &t in active {
        let th = &m.threads[t];
        log.cas_pre
            .push((t, pc, th.vc.clone(), th.acq_pending.clone()));
    }
}

/// Hook (`post_inst`): a lane whose CAS failed ran an attempt E-SPIN treats as invisible; in
/// a schedule where the word already held its compare value it does not run it. So the
/// attempt must not synchronize: its read observes no release beyond what the lane knew.
fn cas_post(m: &Machine, log: &mut SyncLog, from: u64) {
    let snap = log.cas_snap.take();
    for (t, pc, vc0, pend0) in std::mem::take(&mut log.cas_pre) {
        let wrote = log
            .strong
            .iter()
            .rev()
            .take_while(|a| a.ev.seq > from)
            .any(|a| a.write && a.ev.actor == t as u32);
        if wrote {
            continue;
        }
        // The attempt's read is not logged (C-CAS only checks successful links), so it must
        // not decide anything: the iteration fails (E-SPIN then blocks on it).
        if let Some(snap) = &snap {
            let at = &snap.0;
            let leaves = match &at.k.code[pc as usize].op {
                Op::Atom {
                    space,
                    bytes,
                    addr,
                    d: Some(d),
                    ..
                } => at
                    .addr(t, addr)
                    .and_then(|a| at.resolve(t, *space, a, *bytes as usize))
                    .ok()
                    .and_then(|loc| match loc {
                        Loc::Shared { cta, off } => Some(((0u8, cta as u32), off as u64)),
                        Loc::Global { alloc, off } => Some(((1u8, alloc as u32), off)),
                        Loc::Param { .. } => None,
                    })
                    .and_then(|(key, lo)| {
                        // The value it read: the old value it returned.
                        let old = m.rd(t, &Val::Reg(*d, 1)).ok()?;
                        let v: Vec<u8> = (0..*bytes).map(|i| (old >> (8 * i)) as u8).collect();
                        Some(cas_replay(at, t, pc, key, lo, Some(&v)))
                    }),
                _ => None,
            };
            if !matches!(leaves, Some(Ok(false))) {
                log.fail(
                    format!("cas-failexit:{pc}"),
                    format!(
                        "C-CAS: the CAS poll at {} by {} leaves its loop after a failed compare (or cannot be replayed): its read decides the outcome",
                        m.k.describe(pc as usize),
                        m.actor_name(t as u32)
                    ),
                );
            }
        }
        let th = &m.threads[t];
        let me = t as u32;
        let mut base = vc0.clone();
        while vc_get(&base, me) < vc_get(&th.vc, me) {
            vc_tick(&mut base, me);
        }
        let gained = !vc_le(&th.vc, &base)
            || th.acq_pending.iter().any(|(w, rel, sc)| {
                !vc_le(rel, &vc0)
                    && !pend0
                        .iter()
                        .any(|(w0, r0, s0)| w0 == w && s0 == sc && vc_le(rel, r0))
            });
        if gained {
            log.fail(
                format!("cas-failacq:{pc}"),
                format!(
                    "C-CAS: a failed attempt of the CAS poll at {} by {} observes a release the thread did not know (failed attempts are invisible, E-SPIN: another schedule does not make it)",
                    m.k.describe(pc as usize),
                    m.actor_name(me)
                ),
            );
        }
    }
}

/// Replay a CAS poll iteration (CAS to back branch) for thread `t` alone on a copy of `at`,
/// with the word's bytes replaced by `v` if given: does the iteration leave the loop?
fn cas_replay(
    at: &Machine,
    t: usize,
    pc: u32,
    key: (u8, u32),
    lo: u64,
    v: Option<&[u8]>,
) -> Result<bool, String> {
    let (h, b) = at.polls[&pc];
    let mut m = at.clone();
    m.fp = None;
    m.num = None;
    if let Some(v) = v {
        set_mem_bytes(&mut m, loc_of(key, lo), v);
    }
    let mut p = pc;
    loop {
        m.exec(&[t], p)?;
        if p == b {
            break;
        }
        p += 1;
    }
    Ok(m.threads[t].pc != h)
}

fn cell_eq(a: &Option<SyncCell>, b: &Option<SyncCell>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.writer == b.writer
                && a.scope == b.scope
                && a.rel_scope == b.rel_scope
                && a.rel == b.rel
        }
        _ => false,
    }
}

/// C-CAS: the successful compare-and-swaps of CAS polls on one word (`g`, in run order) form
/// a chain whose order is the same in every schedule (module doc). Ok, or why not.
fn cas_chain(m: &Machine, log: &SyncLog, g: &[&SAcc]) -> Result<(), String> {
    let name = |a: &SAcc| describe_ev(m, &a.ev);
    let hex = |v: &[u8]| {
        let mut x = 0u64;
        for (i, b) in v.iter().enumerate().take(8) {
            x |= (*b as u64) << (8 * i);
        }
        format!("{x:#x}")
    };
    // One kind (compare -> swap) per compare value, and a kind changes the value.
    let mut kind: BTreeMap<&[u8], &SAcc> = BTreeMap::new();
    for &a in g {
        if a.old == a.val {
            return Err(format!(
                "{} swaps in the value it compares ({}): CAS polls of that kind may succeed in any order",
                name(a),
                hex(&a.old)
            ));
        }
        if let Some(&o) = kind.get(a.old.as_slice()) {
            if o.val != a.val {
                return Err(format!(
                    "{} ({} -> {}) and {} ({} -> {}) compare the same value: which one consumes it depends on the schedule",
                    name(o),
                    hex(&o.old),
                    hex(&o.val),
                    name(a),
                    hex(&a.old),
                    hex(&a.val)
                ));
            }
        }
        kind.insert(&a.old, a);
    }
    // The CAS polls of one kind are totally ordered by happens-before.
    let mut last: BTreeMap<&[u8], &SAcc> = BTreeMap::new();
    for &a in g {
        if let Some(&p) = last.get(a.old.as_slice()) {
            if !p.ev.hb(a.ev.actor, a.ev.seq, &a.vc) {
                return Err(format!(
                    "{} and {} both swap {} -> {} and are not ordered: either may consume the value",
                    name(p),
                    name(a),
                    hex(&a.old),
                    hex(&a.val)
                ));
            }
        }
        last.insert(&a.old, a);
    }
    // Only the chain writes the word between its first and last link.
    for w in g.windows(2) {
        let (p, a) = (w[0], w[1]);
        if a.old != p.val || !cell_eq(&a.old_cell, &p.cell) {
            return Err(format!(
                "another write changes the word between {} and {}",
                name(p),
                name(a)
            ));
        }
    }
    let (first, lastl) = (g[0], g[g.len() - 1]);
    if let Some(b) = log.strong.iter().find(|b| {
        b.write
            && b.key == first.key
            && b.lo < first.hi
            && first.lo < b.hi
            && b.ev.seq > first.ev.seq
            && b.ev.seq < lastl.ev.seq
            && !(b.at.is_some() && (b.lo, b.hi) == (first.lo, first.hi))
    }) {
        return Err(format!(
            "{} writes the word between CAS polls of its chain",
            describe_ev(m, &b.ev)
        ));
    }
    // The value the chain starts from is written before every link.
    for e in &first.prev_w {
        if let Some(a) = g.iter().find(|a| !e.hb(&a.vc)) {
            return Err(format!(
                "the word's write before the chain ({} at {}) is not ordered before {}",
                m.actor_name(e.actor),
                m.k.describe(e.pc as usize),
                name(a)
            ));
        }
    }
    // Each link leaves its loop on success and fails on every other value the word holds
    // while the chain runs (the value before it and every value a link writes).
    let mut vals: Vec<&[u8]> = vec![&first.old];
    vals.extend(g.iter().map(|a| a.val.as_slice()));
    vals.sort();
    vals.dedup();
    for &a in g {
        let at = a.at.as_ref().unwrap();
        let t = a.ev.actor as usize;
        match cas_replay(&at.0, t, a.ev.pc, a.key, a.lo, Some(&a.old)) {
            Ok(true) => {}
            Ok(false) => {
                return Err(format!("{} swaps but does not leave its loop", name(a)));
            }
            Err(e) => return Err(format!("replaying {}: {e}", name(a))),
        }
        for &v in &vals {
            if v == a.old.as_slice() {
                continue;
            }
            match cas_replay(&at.0, t, a.ev.pc, a.key, a.lo, Some(v)) {
                Ok(false) => {}
                Ok(true) => {
                    return Err(format!(
                        "{} leaves its loop after a failed compare against {} (the word holds it at some point)",
                        name(a),
                        hex(v)
                    ));
                }
                Err(e) => return Err(format!("replaying {} on {}: {e}", name(a), hex(v))),
            }
        }
    }
    Ok(())
}

/// C-CAS over every word with CAS-poll links that are not all HB-ordered (ordered ones are
/// C-ORDER's): the reasons it fails, and the words whose chains it determined.
fn cas_check(m: &Machine, log: &SyncLog) -> (Vec<String>, HashSet<CasWord>) {
    let mut groups: BTreeMap<CasWord, Vec<&SAcc>> = BTreeMap::new();
    for a in &log.strong {
        if a.at.is_some() {
            groups.entry((a.key, a.lo, a.hi)).or_default().push(a);
        }
    }
    let (mut why, mut ok) = (Vec::new(), HashSet::new());
    for (w, g) in groups {
        let ordered = g.iter().enumerate().all(|(i, a)| {
            g[i + 1..]
                .iter()
                .all(|b| a.ev.hb(b.ev.actor, b.ev.seq, &b.vc))
        });
        if ordered {
            continue;
        }
        match cas_chain(m, log, &g) {
            Ok(()) => {
                ok.insert(w);
            }
            Err(s) => why.push(format!(
                "C-CAS: CAS polls on {} {} byte {:#x}: {s}",
                if w.0.0 == 0 {
                    "shared memory of CTA"
                } else {
                    "global allocation"
                },
                w.0.1,
                w.1
            )),
        }
    }
    (why, ok)
}

/// Conflicting strong accesses not ordered by happens-before (C-ORDER).
fn strong_conflicts(
    m: &Machine,
    acc: &[SAcc],
    cas_ok: &HashSet<CasWord>,
    same_ok: order::SameOk,
) -> Vec<String> {
    let mut out = Vec::new();
    // Debugging: PTXV_WS_WORD=space:alloc:offset prints the strong accesses of one word.
    if let Ok(w) = std::env::var("PTXV_WS_WORD") {
        let f: Vec<u64> = w
            .split(':')
            .filter_map(|x| u64::from_str_radix(x.trim_start_matches("0x"), 16).ok())
            .collect();
        if let [sp, al, off] = f[..] {
            for a in acc
                .iter()
                .filter(|a| a.key == (sp as u8, al as u32) && a.lo <= off && off < a.hi)
            {
                eprintln!(
                    "WORD seq={} {} write={} val={:02x?} rmw={:?} clock={} amb={} vc={:?}",
                    a.ev.seq,
                    describe_ev(m, &a.ev),
                    a.write,
                    a.val,
                    a.rmw,
                    a.ev.clock,
                    a.ev.amb,
                    acc.iter()
                        .filter(|b| b.key == a.key
                            && b.lo <= off
                            && off < b.hi
                            && b.ev.seq < a.ev.seq)
                        .map(|b| (
                            b.ev.seq,
                            b.ev.hb(a.ev.actor, a.ev.seq, &a.vc),
                            vc_get(&a.vc, b.ev.actor)
                        ))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
    // The pairs are found by a sweep in run order (order.rs), not enumerated.
    let all = std::env::var("PTXV_WS_ALL").is_ok();
    let found = order::search(acc, cas_ok, MAX_DEAD_PAIRS, MAX_PAIRS, same_ok);
    // Pairs of a write and a plain (non-atomic) read by a thread: the dead-read candidates
    // (`dead_reads`): if the value read reaches nothing, the order does not matter.
    let mut cand: Option<DeadCand> =
        (!found.capped && found.conflicts.len() < MAX_DEAD_PAIRS).then(Default::default);
    for (i, &(xi, yi)) in found.conflicts.iter().enumerate() {
        let (x, y) = (&acc[xi], &acc[yi]);
        let msg = format!(
            "C-ORDER: strong accesses to {} {} byte {:#x} by {} and {} are not ordered (the value read or left depends on the schedule)",
            if x.key.0 == 0 {
                "shared memory of CTA"
            } else {
                "global allocation"
            },
            x.key.1,
            x.lo.max(y.lo),
            describe_ev(m, &x.ev),
            describe_ev(m, &y.ev)
        );
        let read = |a: &SAcc| {
            (!a.write && a.rmw.is_none() && (a.ev.actor as usize) < m.threads.len())
                .then_some((a.ev.actor, a.ev.pc))
        };
        match (cand.as_mut(), read(x).xor(read(y))) {
            (Some(c), Some(r)) => {
                c.0.insert(r);
                if i < 8 || all {
                    c.1.push(msg.clone());
                }
            }
            _ => cand = None,
        }
        if i < 8 || all {
            if x.key.0 == 0 {
                CORDER_SMEM.with(|c| c.borrow_mut().push((x.ev.pc, y.ev.pc, msg.clone())));
            }
            out.push(msg);
        }
    }
    DEAD_CAND.with(|d| *d.borrow_mut() = cand.filter(|c| !c.0.is_empty()));
    if found.capped {
        out.push(format!(
            "C-ORDER: more than {MAX_PAIRS} unordered overlapping strong-access pairs (not checked)"
        ));
        return out;
    }
    if out.len() >= 8 && !all {
        return out;
    }
    let fwords = found.fwords;
    for w in fwords.keys() {
        if let Err(e) = fadd_word(m, acc, *w) {
            out.push(e);
            if out.len() >= 8 && std::env::var("PTXV_WS_ALL").is_err() {
                return out;
            }
        }
    }
    out
}

thread_local! {
    /// C-ORDER pairs on shared memory seen since `corder_take`: (earlier pc, later pc,
    /// reason). Root causes reported with a racy or nondeterministic verdict.
    static CORDER_SMEM: std::cell::RefCell<Vec<(u32, u32, String)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The shared-memory C-ORDER pairs recorded since the last call (and clear them).
fn corder_take() -> Vec<(u32, u32, String)> {
    CORDER_SMEM.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

/// A verdict that has a race or nondeterministic outputs, from a launch whose certificate run
/// had unordered strong shared-memory pairs (C-ORDER under S-SMEM-STRONG: not data races):
/// report each pair first, as a finding of kind `unordered` (not a verdict class), so the
/// order the outcome depends on is named next to its consequence (a global race downstream,
/// say).
fn add_root_pairs(v: &mut Verdict, pairs: Vec<(u32, u32, String)>) {
    let consequence = v.distinct_outputs > 1
        || v.findings
            .iter()
            .any(|f| f.kind == "race" || f.kind == "nondeterministic");
    if !consequence {
        return;
    }
    let mut roots = Vec::new();
    for (a, b, msg) in pairs {
        let sig = format!("unordered-smem:{a}:{b}");
        if roots.iter().any(|f: &Finding| f.sig == sig) || v.findings.iter().any(|f| f.sig == sig) {
            continue;
        }
        roots.push(Finding {
            kind: "unordered".into(),
            sig,
            msg: format!(
                "possible root cause: {}; with LDS/STS strong (S-SMEM-STRONG) this is not a data race, but the verdict's race / nondeterministic outcome may depend on this order (a barrier between the accesses orders it)",
                msg.trim_start_matches("C-ORDER: ")
            ),
            trace: vec![],
            pcs: vec![a, b],
            gb300: None,
        });
    }
    if !roots.is_empty() {
        v.notes.push(format!(
            "{} unordered shared-memory pair(s) from the certificate's run listed first (kind `unordered`, not a verdict class)",
            roots.len()
        ));
        roots.append(&mut v.findings);
        v.findings = roots;
    }
}

thread_local! {
    /// C-FADD words whose order-dependent values reach only the outputs (allocation, first
    /// byte, reachable values), from the last analysis.
    static FADD_NONDET: std::cell::RefCell<Vec<(usize, u64, Vec<u32>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Most linear extensions C-FADD enumerates per group of unordered fp32 adds.
const MAX_FADD_ORDERS: usize = 1 << 16;

/// C-FADD: fp32 atomic adds of one word (unused results, not acquiring) that happens-before
/// does not order. Every other access of the word is ordered with each of them (C-ORDER for
/// strong ones, the race detector for plain ones), so it lies before or after a whole
/// component of the "unordered" relation, and the value the component starts from is the
/// run's. Every schedule applies the component in some linear extension of happens-before;
/// all of them must leave the run's value (the fold is the executor's own add, N-ATOMIC).
fn fadd_word(m: &Machine, acc: &[SAcc], w: ((u8, u32), u64, u64)) -> Result<(), String> {
    let (key, lo, hi) = w;
    let mut ws: Vec<&SAcc> = acc
        .iter()
        .filter(|a| {
            a.write
                && a.key == key
                && (a.lo, a.hi) == (lo, hi)
                && a.fadd.is_some()
                && matches!(a.rmw, Some((AtomOp::FAddFtz, false, false, _)))
        })
        .collect();
    ws.sort_by_key(|a| a.ev.seq);
    let n = ws.len();
    let hb = |i: usize, j: usize| {
        let (a, b) = (ws[i], ws[j]);
        a.ev.hb(b.ev.actor, b.ev.seq, &b.vc)
    };
    // Components of the unordered relation.
    let mut comp: Vec<usize> = (0..n).collect();
    fn root(c: &mut [usize], i: usize) -> usize {
        let mut r = i;
        while c[r] != r {
            r = c[r];
        }
        c[i] = r;
        r
    }
    for i in 0..n {
        for j in i + 1..n {
            if !hb(i, j) && !hb(j, i) {
                let (a, b) = (root(&mut comp, i), root(&mut comp, j));
                comp[a.max(b)] = a.min(b);
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        let r = root(&mut comp, i);
        groups.entry(r).or_default().push(i);
    }
    let where_ = format!(
        "{} {} byte {lo:#x}",
        if key.0 == 0 {
            "shared memory of CTA"
        } else {
            "global allocation"
        },
        key.1
    );
    for g in groups.values().filter(|g| g.len() > 1) {
        let word = |b: &[u8]| -> u32 {
            b.iter()
                .take(4)
                .enumerate()
                .fold(0u32, |x, (i, &v)| x | (v as u32) << (8 * i))
        };
        let v0 = word(&ws[g[0]].old);
        let want = word(&ws[*g.last().unwrap()].val);
        // Depth-first over linear extensions of happens-before restricted to the group.
        let k = g.len();
        let mut seen: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        let mut count = 0usize;
        let mut stack: Vec<(Vec<bool>, u32, Vec<usize>)> = vec![(vec![false; k], v0, vec![])];
        while let Some((used, acc_v, order)) = stack.pop() {
            if order.len() == k {
                count += 1;
                seen.entry(acc_v).or_insert(order);
                if count > MAX_FADD_ORDERS {
                    return Err(format!(
                        "C-FADD: more than {MAX_FADD_ORDERS} orders of the {k} unordered fp32 atomic adds to {where_} (not checked)"
                    ));
                }
                continue;
            }
            for x in 0..k {
                if used[x] {
                    continue;
                }
                // x may come next if every member happens-before it has been applied.
                if (0..k).any(|y| !used[y] && y != x && hb(g[y], g[x])) {
                    continue;
                }
                let mut u = used.clone();
                u[x] = true;
                let v = crate::fp::f32_fma(
                    acc_v,
                    0x3f80_0000,
                    Some(ws[g[x]].fadd.unwrap()),
                    Rnd::Rn,
                    true,
                );
                let mut o = order.clone();
                o.push(x);
                stack.push((u, v, o));
            }
        }
        // Nondeterministic in the output only: the component holds the word's last writes, of
        // an output buffer, and nothing reads the word after them (the final race shadow
        // has no reader since the last write, which is one of the adds; no strong write
        // follows). Then the order changes only the final bytes of that word: every value of
        // `seen` is reachable and nothing else differs between schedules.
        if seen.len() > 1 && seen.contains_key(&want) && key.0 == 1 {
            let alloc = key.1 as usize;
            let last_seq = g.iter().map(|&i| ws[i].ev.seq).max().unwrap_or(0);
            let later_write = acc
                .iter()
                .any(|a| a.write && a.key == key && a.lo < hi && lo < a.hi && a.ev.seq > last_seq);
            let quiet = (lo..hi).all(|b| {
                let sh = m.gmem[alloc].sh.get(b as usize);
                sh.r.is_empty()
                    && sh.w.is_some_and(|w| {
                        g.iter()
                            .any(|&i| (ws[i].ev.actor, ws[i].ev.pc) == (w.actor, w.pc))
                    })
            });
            if !later_write && quiet && m.launch.outputs.contains(&alloc) {
                FADD_NONDET.with(|f| {
                    f.borrow_mut()
                        .push((alloc, lo, seen.keys().copied().collect()))
                });
                continue;
            }
        }
        if seen.len() > 1 || !seen.contains_key(&want) {
            let vals: Vec<String> = seen.keys().map(|v| format!("{v:#010x}")).collect();
            return Err(format!(
                "C-FADD: the {k} unordered fp32 atomic adds to {where_} (first {}) leave {} different values depending on their order ({}; the run left {want:#010x})",
                describe_ev(m, &ws[g[0]].ev),
                seen.len(),
                vals.join(", ")
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- driver

/// Result of a certificate attempt.
pub struct Outcome {
    /// Some: certified; the verdict holds for every schedule.
    pub verdict: Option<Verdict>,
    /// Why the certificate does not hold (empty when certified).
    pub reasons: Vec<String>,
    pub steps: usize,
    pub seconds: f64,
    /// Races the certificate's own (unforced) runs reported: each is a race in a feasible
    /// execution of the model. The exploration that replaces a failed certificate decides
    /// the verdict, but may run out of budget before it reaches them (`run_races`).
    pub races: Vec<Finding>,
}

/// The races of a failed certificate's runs that stand in the reference (LDS/STS strong)
/// model: all of them, unless the certified kernel had LDS/STS plain (`weak`: whether the
/// strong kernel has acquiring LDS), where those a strong re-check may dismiss
/// (`smem::strong_race`) are dropped. Printed now (before the exploration, which may run out
/// of budget or time) and returned for `merge_run_races`.
fn run_races(o: &Outcome, weak: Option<bool>, what: &str) -> Vec<Finding> {
    let rs: Vec<Finding> = o
        .races
        .iter()
        .filter(|f| weak.is_none_or(|acq| !crate::smem::strong_race(&f.sig, acq)))
        .cloned()
        .collect();
    for f in &rs {
        eprintln!(
            "certificate run race ({what}): {}: {}",
            f.sig,
            f.msg.lines().next().unwrap_or("")
        );
    }
    rs
}

/// An exploration that did not complete keeps the races the certificate's runs found (a
/// feasible execution reaches each), with a note.
fn merge_run_races(v: &mut Verdict, races: Vec<Finding>) {
    if v.complete {
        return;
    }
    for f in races {
        if !v.findings.iter().any(|x| x.sig == f.sig) {
            v.notes.push(format!(
                "race {} found by the certificate's run (a feasible execution), not reached by the exploration",
                f.sig
            ));
            v.findings.push(f);
        }
    }
}

/// One logged run under one leader policy: (final machine, findings, terminal, steps,
/// reasons the certificate fails).
fn run_logged(mut m: Machine) -> (Machine, Vec<Finding>, bool, usize, Vec<String>) {
    // The dead-read probe reruns from the start (`dead_reads`).
    let m0 = m.dead_probe.is_none().then(|| m.clone());
    DEAD_CAND.with(|d| *d.borrow_mut() = None);
    // E-RACE-DEAD: the run's racing reads (race_dead.rs).
    let race_log = m0.as_ref().and_then(|_| crate::race_dead::new_log());
    m.race_log = race_log.clone();
    m.wlog = LogSlot(Some(Box::new(SyncLog::new(&m))));
    if std::env::var("PTXV_WS_NOLOG").is_ok() {
        m.wlog = LogSlot(None);
    }
    let mut fs = Vec::new();
    let mut steps = 0usize;
    // PTXV_CERT_PROGRESS=N: print the step count every N steps (stderr), for timing runs that
    // do not finish.
    let progress = std::env::var("PTXV_CERT_PROGRESS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0);
    let t0 = std::time::Instant::now();
    let mut extra = Vec::new();
    let mut sched = crate::explore::DetSched::default();
    let mut faults = crate::explore::FaultLoop::default();
    while let Some(a) = sched.pick(&m) {
        if matches!(a, Action::Clc(..)) {
            extra.push(
                "C-ORDER: a cluster launch control response is pending (a scheduler choice)".into(),
            );
            break;
        }
        m.apply(&a);
        steps += 1;
        if progress.is_some_and(|n| steps.is_multiple_of(n)) {
            eprintln!(
                "[certificate] {steps} steps {:.1}s",
                t0.elapsed().as_secs_f64()
            );
        }
        // `certify_inner` launched every CTA up front (E-CLC): a try_cancel that executes
        // ends the run (its fault would repeat), and the certificate starts over lazily.
        if m.clc_eager
            && m.findings
                .iter()
                .any(|f| f.msg.contains("E-CLC eager launch"))
        {
            EAGER_FAILED.with(|e| e.set(true));
            fs.append(&mut m.findings);
            break;
        }
        if let Some(why) = faults.repeats(&m.findings) {
            extra.push(why);
            fs.append(&mut m.findings);
            break;
        }
        fs.append(&mut m.findings);
    }
    let ok = m.terminal();
    RING_PRUNED.with(|p| *p.borrow_mut() = None);
    if ok {
        m.exit_checks();
        fs.append(&mut m.findings);
    } else if let Some(why) = ring::stuck_infeasible(&m).filter(|_| extra.is_empty()) {
        // C-RING: a forced read waits for a write that never comes while the poll would exit
        // on a value already written: no execution makes this run's choices.
        RING_PRUNED.with(|p| *p.borrow_mut() = Some(why));
    } else if extra.is_empty() {
        let (sig, msg) = m.deadlock_report();
        fs.push(Finding {
            kind: "deadlock".into(),
            sig,
            msg,
            trace: vec![],
            ..Default::default()
        });
    }
    let mut why = extra;
    match m.wlog.0.take() {
        Some(log) => {
            // ASSUME(S-SAME-VALUE): the run's written values stand for every schedule's only
            // when every read is determined: no strong accesses (atomics, flags, C-RING
            // counters) and no polls, whose values may differ in another schedule
            // (fixtures/e_ring_warp W1: equal in the run, a race elsewhere).
            if fs.iter().any(|f| f.sig.starts_with("same-value:"))
                && (!log.strong.is_empty() || !log.polls.is_empty())
            {
                why.push(
                    "S-SAME-VALUE: writes of equal values in a run with strong accesses or polls, whose values may depend on the schedule"
                        .into(),
                );
            }
            why.extend(analyze(&m, &log));
            CARRY.with(|p| *p.borrow_mut() = carry(&log));
        }
        None => why.push("no log (PTXV_WS_NOLOG)".into()),
    }
    if let Some(m0) = m0 {
        // Dead reads: C-ORDER conflicts against plain loads, and plain races, whose loaded
        // values reach nothing (one rerun with the loads undefined decides; `probe_run`).
        let corder = DEAD_CAND.with(|d| d.borrow_mut().take());
        let races = race_log
            .map(|l| crate::race_dead::candidates(&l, &fs))
            .unwrap_or_default();
        let probe = |reads: &BTreeSet<(u32, u32)>| probe_run(&m0, reads, &fs, ok);
        let (corder_dead, dead_races) = match &corder {
            Some((reads, _)) => {
                let mut all = reads.clone();
                all.extend(races.iter().flat_map(|c| c.1.iter().copied()));
                if probe(&all) {
                    (true, races.iter().map(|c| c.0.clone()).collect())
                } else if races.is_empty() {
                    (false, BTreeSet::new())
                } else {
                    (probe(reads), crate::race_dead::decide(&races, probe))
                }
            }
            None => (false, crate::race_dead::decide(&races, probe)),
        };
        if let (true, Some((reads, msgs))) = (corder_dead, corder) {
            why.retain(|w| !msgs.contains(w));
            let first = reads.iter().next().copied().unwrap_or_default();
            fs.push(Finding {
                kind: "note".into(),
                sig: format!("dead-read:{}", first.1),
                msg: format!(
                    "C-ORDER: {} load(s) unordered with a write read values that reach nothing (rerun with them undefined, S-INIT), first at {}",
                    reads.len(),
                    m0.k.describe(first.1 as usize)
                ),
                trace: vec![],
                ..Default::default()
            });
        }
        crate::race_dead::mark(&mut fs, &dead_races);
    }
    (m, fs, ok, steps, why)
}

/// Dead reads (C-ORDER conflicts against plain loads, E-RACE-DEAD races): rerun the launch
/// with the loads `reads` (thread, pc) leaving their registers undefined (`Machine::dead_probe`,
/// S-INIT): undefined bits propagate through register-only operations and are reported where
/// they reach memory, an address, a predicate or control flow, or a synchronization operand.
/// True if the rerun reports no use the run did not and ends as the run did: no read value
/// reaches anything along the run's path, so a read of any other value leaves the same events
/// (BlackwellScoreSummary: warps that skip the named barrier read the TMEM address before it
/// is stored and never use it).
fn probe_run(m0: &Machine, reads: &BTreeSet<(u32, u32)>, fs: &[Finding], ok: bool) -> bool {
    let m = crate::race_dead::probe_machine(m0, reads);
    // The rerun is a logged run too: keep the thread-local results of this one.
    let carry = CARRY.with(|p| p.borrow().clone());
    let ring_out = RING_OUT.with(|r| std::mem::take(&mut *r.borrow_mut()));
    let pruned = RING_PRUNED.with(|p| p.borrow_mut().take());
    let eager = EAGER_FAILED.with(|e| e.get());
    let corder = CORDER_SMEM.with(|c| c.borrow().clone());
    let (_, fs1, ok1, _, _) = run_logged(m);
    CORDER_SMEM.with(|c| *c.borrow_mut() = corder);
    CARRY.with(|p| *p.borrow_mut() = carry);
    RING_OUT.with(|r| *r.borrow_mut() = ring_out);
    RING_PRUNED.with(|p| *p.borrow_mut() = pruned);
    EAGER_FAILED.with(|e| e.set(eager));
    ok1 == ok && crate::race_dead::no_new_use(fs, &fs1)
}

/// A global byte range (allocation, first byte, end) and an access to it.
pub type Poison = (usize, u64, u64, Epoch);

/// What a later launch group needs from a certified group's log (`groups.rs`): global words
/// written by unordered strong writes, and every strong read of global memory.
#[derive(Default, Clone, Debug)]
pub struct Carry {
    pub poison: Vec<Poison>,
    pub reads: Vec<Poison>,
}

/// C-ORDER dead-read candidates: the plain reads (thread, pc) and the reasons they account for.
type DeadCand = (BTreeSet<(u32, u32)>, Vec<String>);

thread_local! {
    /// The last logged run executed a try_cancel with every CTA launched up front.
    static EAGER_FAILED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The last logged run's `carry` (for `certify_keep`).
    static CARRY: std::cell::RefCell<Carry> = std::cell::RefCell::new(Carry::default());
    /// The last logged run was a C-RING run whose choices no execution makes (why).
    static RING_PRUNED: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
    /// The last analysis' C-ORDER dead-read candidates: the reads (thread, pc) and the
    /// reasons they account for, when every C-ORDER conflict is a write against such a read.
    static DEAD_CAND: std::cell::RefCell<Option<DeadCand>> =
        const { std::cell::RefCell::new(None) };
    /// The last analysis' C-RING choices (run order) and the counters it relied on.
    static RING_OUT: std::cell::RefCell<(Vec<ring::ChoiceRec>, BTreeSet<String>)> =
        std::cell::RefCell::new(Default::default());
}

/// `Carry` of a certificate run.
///
/// * `poison`: global words written by strong writes that happens-before does not order
///   (allowed by C-ORDER's commutative atomics, C-FADD and C-CAS). The race shadow keeps only
///   the run's last writer of a word, so a later group must not touch such a word.
/// * `reads`: every strong read of global memory (C-POLL's flag polls included), as an epoch
///   (strict after a strong store in the same window). The shadow drops the reads before a
///   byte's last write; a strong read need not be ordered before that write (a poll that
///   read an earlier value), so a later group's write must be checked against it directly.
fn carry(log: &SyncLog) -> Carry {
    let epoch = |e: &Ev| Epoch {
        actor: e.actor,
        clock: e.clock + e.amb as u32,
        pc: e.pc,
        strong: 0,
        sv: false,
    };
    let mut by: BTreeMap<u32, Vec<&SAcc>> = BTreeMap::new();
    let mut out = Carry::default();
    for a in &log.strong {
        if a.key.0 != 1 {
            continue;
        }
        if a.write {
            by.entry(a.key.1).or_default().push(a);
        } else {
            out.reads.push((a.key.1 as usize, a.lo, a.hi, epoch(&a.ev)));
        }
    }
    for r in &log.polls {
        if r.key.0 == 1 {
            out.reads.push((r.key.1 as usize, r.lo, r.hi, epoch(&r.ev)));
        }
    }
    for (alloc, lo, hi, ev) in ring::global_reads(log) {
        out.reads.push((alloc as usize, lo, hi, epoch(&ev)));
    }
    for (alloc, mut v) in by {
        v.sort_by_key(|a| (a.lo, a.ev.seq));
        for i in 0..v.len() {
            for b in &v[i + 1..] {
                let a = v[i];
                if b.lo >= a.hi {
                    break;
                }
                if a.ev.actor == b.ev.actor {
                    continue;
                }
                let (x, y) = if a.ev.seq < b.ev.seq {
                    (a, *b)
                } else {
                    (*b, a)
                };
                if !x.ev.hb(y.ev.actor, y.ev.seq, &y.vc) {
                    out.poison
                        .push((alloc as usize, a.lo.max(b.lo), a.hi.min(b.hi), epoch(&x.ev)));
                }
            }
        }
    }
    out
}

/// `certify`, keeping the run's final machine and its `carry` when it holds
/// (one leader policy only).
pub fn certify_keep(m0: Machine) -> (Outcome, Option<(Machine, Carry)>) {
    let mut keep = None;
    let o = certify_inner(m0, Some(&mut keep));
    (o, keep)
}

/// Try the well-synchronization certificate on `m0` (a whole launch, or a slice prepared like
/// `explore::explore_slice` does). ASSUME(E-WELLSYNC): see ASSUMPTIONS.md
pub fn certify(m0: Machine) -> Outcome {
    certify_inner(m0, None)
}

fn certify_inner(m0: Machine, keep: Option<&mut Option<(Machine, Carry)>>) -> Outcome {
    // ASSUME(E-CLC): every CTA launched up front when no try_cancel executes. Instead of a
    // separate trial run (`explore::clc_eager`), the certificate's own runs decide it: a run
    // that executes a try_cancel stops, and the certificate starts over on `m0` (lazy).
    match crate::explore::clc_eager_launch(&m0) {
        Some(e) => {
            let lazy = m0;
            let mut kept = None;
            let o = certify_runs(e, keep.is_some().then_some(&mut kept));
            if EAGER_FAILED.with(|f| f.replace(false)) {
                return certify_runs(lazy, keep);
            }
            if let Some(k) = keep {
                *k = kept;
            }
            o
        }
        None => certify_runs(m0, keep),
    }
}

fn certify_runs(m0: Machine, keep: Option<&mut Option<(Machine, Carry)>>) -> Outcome {
    let t0 = Instant::now();
    // ASSUME(E-RESIDENCY): the certificate's progress argument has every CTA resident and
    // every allocation placed first-fit; limited residency and the TMEM co-tenant are decided
    // by residency::analyze / tenant::check instead.
    if m0.launch.env.residency.is_some() || m0.launch.env.tmem_cotenant {
        return Outcome {
            verdict: None,
            reasons: vec![
                "limited residency / TMEM co-tenant mode: the certificate assumes every CTA resident (E-RESIDENCY); see residency::analyze".into(),
            ],
            steps: 0,
            seconds: 0.0,
            races: vec![],
        };
    }
    // S-NIC-*: the certificate's conditions do not cover NIC queue pairs (nic.rs).
    if m0.nic.is_some() {
        return Outcome {
            verdict: None,
            reasons: vec!["NIC queue pairs: decided by exploration (docs/nic.md)".into()],
            steps: 0,
            seconds: 0.0,
            races: vec![],
        };
    }
    let m0 = crate::explore::clc_eager(m0);
    let elects =
        m0.k.front == Front::Ptx && m0.k.code.iter().any(|i| matches!(i.op, Op::Elect { .. }));
    if elects && std::env::var("PTXV_ELECT").as_deref() == Ok("all") {
        return Outcome {
            verdict: None,
            reasons: vec![
                "PTXV_ELECT=all: every lane may be elected (not a fixed function)".into(),
            ],
            steps: 0,
            seconds: t0.elapsed().as_secs_f64(),
            races: vec![],
        };
    }
    // ASSUME(E-ELECT): certify each leader policy separately (as `per_leader_policy`).
    let policies: &[bool] = if elects { &[false, true] } else { &[false] };
    let mut findings: Vec<Finding> = crate::explore::static_findings(&m0);
    findings.extend(crate::spin::check(&m0));
    let mut outputs: Vec<Vec<u8>> = Vec::new();
    let mut reasons = Vec::new();
    let mut steps = 0;
    let mut terminal = 0;
    let mut fadd_note: Option<String> = None;
    let mut fadd_total = 0usize;
    // C-RING: one run per sequence of classes at the choices (ring.rs), as a depth-first
    // search over forced reads; a run whose choices no execution makes is dropped.
    let max_runs: usize = std::env::var("PTXV_RING_MAX_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let mut ring_runs = 0usize;
    let mut run_race_list: Vec<Finding> = Vec::new();
    let mut ring_pruned = 0usize;
    let mut ring_counters: BTreeSet<String> = BTreeSet::new();
    let mut ring_choices = 0usize;
    type Forcing = BTreeMap<(u32, u32, u32), ring::Forced>;
    let mut work: Vec<(bool, Option<Forcing>)> =
        policies.iter().map(|&hi| (hi, None)).rev().collect();
    let mut first_out: Option<(Vec<u8>, String)> = None;
    // Final machines and carries of the runs (launch groups continue from their meet).
    let mut leaves: Vec<(Machine, Carry)> = Vec::new();
    while let Some((hi, force)) = work.pop() {
        if ring_runs >= max_runs {
            reasons.push(format!(
                "C-RING: more than {max_runs} runs (one per sequence of choices; PTXV_RING_MAX_RUNS)"
            ));
            break;
        }
        ring_runs += 1;
        let mut m = m0.clone();
        m.elect_high = hi;
        let forced_desc = force.as_ref().map(|f| f.len()).unwrap_or(0);
        m.ring = Some(Arc::new(ring::RingCtl::new(
            force.clone().unwrap_or_default(),
        )));
        let (m, fs, ok, n, why) = run_logged(m);
        steps += n;
        if EAGER_FAILED.with(|f| f.get()) {
            return Outcome {
                verdict: None,
                reasons: vec![
                    "E-CLC: a try_cancel executed with every CTA launched up front".into(),
                ],
                steps,
                seconds: t0.elapsed().as_secs_f64(),
                races: vec![],
            };
        }
        let pruned = RING_PRUNED.with(|p| p.borrow_mut().take());
        let (choices, counters) = RING_OUT.with(|r| std::mem::take(&mut *r.borrow_mut()));
        ring_counters.extend(counters);
        let tag = match (hi, forced_desc) {
            (true, 0) => " (leader policy: highest active lane)".to_string(),
            (false, 0) => String::new(),
            (true, k) => {
                format!(" (leader policy: highest active lane; C-RING run with {k} forced read(s))")
            }
            (false, k) => format!(" (C-RING run with {k} forced read(s))"),
        };
        for w in why {
            reasons.push(format!("{w}{tag}"));
        }
        // Children: every other class at every choice this run did not inherit, with the
        // earlier choices of several classes pinned to this run's class.
        if std::env::var("PTXV_RING_DEBUG").is_ok() {
            eprintln!(
                "C-RING run {ring_runs} ({forced_desc} forced): {} choice(s)",
                choices.len()
            );
            for c in &choices {
                eprintln!(
                    "  {:?} {} classes {:?} run class {} forced {}",
                    c.id,
                    c.what,
                    c.classes
                        .iter()
                        .map(|x| (x.0, x.1.clone()))
                        .collect::<Vec<_>>(),
                    c.run_class,
                    c.forced
                );
            }
        }
        let mut pin = force.clone().unwrap_or_default();
        for c in &choices {
            if c.forced || c.classes.len() < 2 {
                continue;
            }
            ring_choices += 1;
            let forced_to = |x: usize| ring::Forced {
                word: c.word,
                n: c.n,
                idx: c.classes[x].0,
                bytes: c.classes[x].1.clone(),
                cell: c.classes[x].2.clone(),
            };
            for x in 0..c.classes.len() {
                if x == c.run_class {
                    continue;
                }
                let mut f = pin.clone();
                for &(lane, ord) in &c.lanes {
                    f.insert((lane, c.id.1, ord), forced_to(x));
                }
                work.push((hi, Some(f)));
            }
            for &(lane, ord) in &c.lanes {
                pin.insert((lane, c.id.1, ord), forced_to(c.run_class));
            }
        }
        if pruned.is_some() {
            ring_pruned += 1;
            if !reasons.is_empty() {
                break;
            }
            continue;
        }
        if std::env::var("PTXV_WS_DEBUG").is_ok() {
            for f in &fs {
                eprintln!("run finding [{}] {}", f.kind, f.msg);
            }
        }
        // Exit-time contract findings (E-BULK-EXIT) are happens-before checks at the end:
        // every schedule's clocks contain the run's (C-WAIT), so the run reports the union.
        // So are the two-level check's havoc probes (kind "probe", l1.rs), and notes (kind
        // "note": S-SAME-VALUE's equal-value writes, which change no value). So is a tcgen05
        // shared-memory descriptor's version field (E-DESC-VERSION, "desc-version:", a
        // `contract` finding for tcgen05.mma, a violation for tcgen05.cp): a check of a
        // descriptor value, which the certificate (schedule-independent values) makes the
        // same in every schedule; the explorer adds nothing, at 35 GB per kernel on a
        // 4096 x 4096 x 8192 tile.
        let (contract, fs): (Vec<Finding>, Vec<Finding>) = fs.into_iter().partition(|f| {
            f.kind == "contract"
                || f.kind == "probe"
                || f.kind == "note"
                || f.kind == "race-dead"
                || f.sig.starts_with("desc-version:")
        });
        for f in contract {
            if !findings.iter().any(|x| x.sig == f.sig) {
                findings.push(f);
            }
        }
        // Races of an unforced run (C-RING runs with forced reads may not be executions).
        if forced_desc == 0 {
            for f in fs.iter().filter(|f| f.kind == "race") {
                if !run_race_list.iter().any(|x| x.sig == f.sig) {
                    run_race_list.push(f.clone());
                }
            }
        }
        if !fs.is_empty() {
            let mut kinds: Vec<&str> = fs.iter().map(|f| f.kind.as_str()).collect();
            kinds.sort();
            kinds.dedup();
            reasons.push(format!(
                "the run reported findings ({}); the explorer decides them: first: {}{tag}",
                kinds.join(", "),
                fs[0].msg.lines().next().unwrap_or("")
            ));
        }
        if ok {
            terminal += 1;
            let o = m.outputs();
            // C-RING: every sequence of choices must leave the same outputs (chunking-
            // obliviousness, checked semantically).
            match &first_out {
                Some((x, what)) if *x != o && forced_desc + ring_choices > 0 => {
                    reasons.push(format!(
                        "C-RING: not chunking-oblivious: the run{tag} leaves other outputs than {what} (the observed counter values reach the results)"
                    ));
                    break;
                }
                Some(_) => {}
                None => first_out = Some((o.clone(), format!("the first run{tag}"))),
            }
            let nd = FADD_NONDET.with(|f| std::mem::take(&mut *f.borrow_mut()));
            if !nd.is_empty() {
                // Every combination of the words' reachable values (enumerated up to 256).
                let mut base_off = std::collections::HashMap::new();
                let mut off = 0usize;
                for &i in &m.launch.outputs {
                    base_off.insert(i, off);
                    off += m.gmem[i].bytes.len();
                }
                let total = nd
                    .iter()
                    .fold(1usize, |p, (_, _, v)| p.saturating_mul(v.len()));
                fadd_note = Some(format!(
                    "C-FADD: {} output word(s) take order-dependent values from unordered fp32 atomic adds, {} reachable output(s){}",
                    nd.len(),
                    total,
                    if total > 256 { " (256 enumerated)" } else { "" }
                ));
                fadd_total = total;
                let mut combo = vec![0usize; nd.len()];
                for _ in 0..total.min(256) {
                    let mut x = o.clone();
                    for (k, (alloc, lo, vals)) in nd.iter().enumerate() {
                        let at = base_off[alloc] + *lo as usize;
                        x[at..at + 4].copy_from_slice(&vals[combo[k]].to_le_bytes());
                    }
                    if !outputs.contains(&x) {
                        outputs.push(x);
                    }
                    for k in 0..combo.len() {
                        combo[k] += 1;
                        if combo[k] < nd[k].2.len() {
                            break;
                        }
                        combo[k] = 0;
                    }
                }
            } else if !outputs.contains(&o) {
                outputs.push(o);
            }
        }
        if keep.is_some() && policies.len() == 1 && ok {
            leaves.push((m, CARRY.with(|p| std::mem::take(&mut *p.borrow_mut()))));
        }
        if !reasons.is_empty() {
            break;
        }
    }
    let seconds = t0.elapsed().as_secs_f64();
    if !reasons.is_empty() {
        return Outcome {
            verdict: None,
            reasons,
            steps,
            seconds,
            races: run_race_list,
        };
    }
    let mut keep_note = None;
    if let Some(k) = keep {
        *k = match leaves.len() {
            0 => None,
            1 => leaves.pop(),
            _ => match crate::groups::meet_runs(leaves) {
                Ok(x) => Some(x),
                Err(why) => {
                    keep_note = Some(format!(
                        "C-RING: no continuation for later launch groups: {why}"
                    ));
                    None
                }
            },
        };
    }
    let mut assumptions = crate::assumptions::for_machine(&m0);
    assumptions.push("E-WELLSYNC".into());
    findings.sort_by(|a, b| (&a.kind, &a.sig).cmp(&(&b.kind, &b.sig)));
    Outcome {
        verdict: Some(Verdict {
            states: steps,
            transitions: steps,
            complete: true,
            terminal_states: terminal,
            distinct_outputs: outputs.len().max(fadd_total),
            findings,
            assumptions,
            notes: vec![format!(
                "well-synchronized: one run covers all schedules (E-WELLSYNC, {steps} steps{})",
                if policies.len() > 1 {
                    ", one run per leader policy"
                } else {
                    ""
                }
            )],
            seconds,
            outputs,
        })
        .map(|mut v| {
            v.notes.extend(fadd_note);
            v.notes.extend(keep_note);
            // ASSUME(E-RESIDENCY): the certified run has every CTA resident; a cross-CTA wait
            // is decided only under that assumption (residency::analyze lifts it).
            v.notes.extend(crate::residency::reliance_note(&m0));
            if !ring_counters.is_empty() {
                v.notes.push(format!(
                    "C-RING: {} poll(s) of monotone counters decided; {} choice(s) of several classes, {} run(s) ({} dropped as infeasible); relies on E-RESIDENCY (producer and consumer CTAs are resident at the same time){}; counters: {}",
                    ring_counters.len(),
                    ring_choices,
                    ring_runs,
                    ring_pruned,
                    if m0.launch.poll_acquire { " and E-POLL-ACQ (warp-level polls acquire)" } else { "" },
                    ring_counters.iter().cloned().collect::<Vec<_>>().join("; ")
                ));
                for a in ["E-RESIDENCY", "E-POLL-ACQ"] {
                    if (a != "E-POLL-ACQ" || m0.launch.poll_acquire)
                        && !v.assumptions.iter().any(|x| x == a)
                    {
                        v.assumptions.push(a.into());
                    }
                }
            }
            v
        }),
        reasons: vec![],
        steps,
        seconds,
        races: vec![],
    }
}

/// `m0` restricted to cluster `c`, as `explore::explore_slice` prepares it: every CTA of
/// another cluster (of this grid or a later one) counts as exited.
pub fn slice_machine(mut m0: Machine, c: u32) -> Machine {
    let cg = (c >> 20) as usize;
    for t in 0..m0.threads.len() {
        let cta = m0.threads[t].cta as u32;
        if m0.launch.cluster_of(cta).0 != c && m0.launch.grid_index(cta) >= cg {
            m0.threads[t].make_mut().st = St::Exited;
        }
    }
    let live: Vec<usize> = (0..m0.threads.len())
        .filter(|&t| m0.threads[t].st != St::Exited)
        .collect();
    m0.live = match (live.first(), live.last()) {
        (Some(&a), Some(&b)) => a..b + 1,
        _ => 0..0,
    };
    m0
}

/// `ptxv check --certify`: the certificate, else exhaustive exploration (printing why;
/// PTXV_CERTIFY_ONLY=1: an incomplete verdict instead of exploring).
pub fn check_certified(m0: Machine, max_states: usize) -> Verdict {
    let base = m0.clone();
    let mut v = check_certified_inner(m0, max_states);
    // S-INIT-FILLS: undefined values that reach only output data.
    crate::fills::check(&base, &mut v);
    v
}

fn check_certified_inner(m0: Machine, max_states: usize) -> Verdict {
    // ASSUME(S-SMEM-STRONG): the certificate on the weak kernel (LDS/STS plain) first, as
    // `smem::two_pass` does for exploration. If that fails, the certificate on the kernel
    // itself (LDS/STS relaxed strong at CTA scope, the hardware-faithful model): a race
    // between plain LDS/STS (a shared-memory relay such as DeepEP combine's tail) is a
    // strong-access pair there, which C-ORDER / C-POLL decide. Either certificate alone is
    // a verdict for every schedule of its model; the strong model is the reference one.
    if let Some(w) = crate::smem::weak(&m0.k) {
        let w = std::sync::Arc::new(w);
        let mut mw = m0.clone();
        mw.k = w;
        let ow = certify(mw);
        if let Some(v) = ow.verdict {
            return v;
        }
        corder_take();
        let os = certify(m0.clone());
        let roots = corder_take();
        if let Some(mut v) = os.verdict {
            v.notes.push(format!(
                "certified with LDS/STS as relaxed .cta strong accesses (S-SMEM-STRONG); with them plain it was not ({})",
                ow.reasons.first().map(String::as_str).unwrap_or("?")
            ));
            return v;
        }
        if std::env::var("PTXV_CERTIFY_ONLY").is_ok() {
            report_fallback("launch (LDS/STS plain)", &ow);
            report_fallback("launch (LDS/STS strong)", &os);
            let mut reasons = ow.reasons.clone();
            reasons.extend(os.reasons.iter().map(|r| format!("(LDS/STS strong) {r}")));
            return not_certified(&Outcome { reasons, ..os });
        }
        report_fallback("launch (LDS/STS plain)", &ow);
        report_fallback("launch (LDS/STS strong)", &os);
        let acq = crate::smem::acquiring_lds(&m0.k);
        let mut races = run_races(&ow, Some(acq), "launch (LDS/STS plain)");
        races.extend(run_races(&os, None, "launch (LDS/STS strong)"));
        let first = ow.reasons.first().cloned().unwrap_or_default();
        let mut v = crate::smem::two_pass(m0, |m| crate::explore::explore(m, max_states));
        v.notes
            .push(format!("certificate failed ({first}), explored instead"));
        merge_run_races(&mut v, races);
        add_root_pairs(&mut v, roots);
        return v;
    }
    check_certified_one(m0, max_states)
}

fn not_certified(o: &Outcome) -> Verdict {
    Verdict {
        states: o.steps,
        transitions: o.steps,
        complete: false,
        terminal_states: 0,
        distinct_outputs: 0,
        findings: vec![],
        assumptions: vec![],
        notes: o
            .reasons
            .iter()
            .map(|r| format!("not certified: {r}"))
            .collect(),
        seconds: o.seconds,
        outputs: vec![],
    }
}

fn check_certified_one(m0: Machine, max_states: usize) -> Verdict {
    let o = certify(m0.clone());
    match o.verdict {
        Some(v) => v,
        None if std::env::var("PTXV_CERTIFY_ONLY").is_ok() => {
            report_fallback("launch", &o);
            Verdict {
                states: o.steps,
                transitions: o.steps,
                complete: false,
                terminal_states: 0,
                distinct_outputs: 0,
                findings: vec![],
                assumptions: vec![],
                notes: o
                    .reasons
                    .iter()
                    .map(|r| format!("not certified: {r}"))
                    .collect(),
                seconds: o.seconds,
                outputs: vec![],
            }
        }
        None => {
            report_fallback("launch", &o);
            let weak = (m0.weak_stop > 0).then_some(m0.weak_stop == 2);
            let races = run_races(&o, weak, "launch");
            let mut v = crate::explore::explore(m0, max_states);
            v.notes.push(format!(
                "certificate failed ({}), explored instead",
                o.reasons.first().map(String::as_str).unwrap_or("?")
            ));
            merge_run_races(&mut v, races);
            v
        }
    }
}

fn report_fallback(what: &str, o: &Outcome) {
    eprintln!(
        "certificate for {what} does not hold ({} steps, {:.1}s); exploring instead. Reasons:",
        o.steps, o.seconds
    );
    for r in o.reasons.iter().take(12) {
        eprintln!("  - {r}");
    }
    if o.reasons.len() > 12 {
        eprintln!("  ... {} more", o.reasons.len() - 12);
    }
}

/// `ptxv check --certify --slices`: per cluster, the certificate or else `explore_slice`;
/// then the cross-cluster global-memory check of `explore::explore_sliced` (E-CTA-INDEP).
/// A certified slice's run touched exactly the bytes any of its schedules touches.
pub fn check_sliced(
    make: &dyn Fn() -> Machine,
    max_states: usize,
    per_slice: &mut dyn FnMut(u32, &Verdict),
) -> Verdict {
    crate::smem::two_pass_make(make, &mut |mk| check_sliced_one(mk, max_states, per_slice))
}

fn check_sliced_one(
    make: &dyn Fn() -> Machine,
    max_states: usize,
    per_slice: &mut dyn FnMut(u32, &Verdict),
) -> Verdict {
    let t0 = Instant::now();
    let m = make();
    let launch = m.launch.clone();
    // ASSUME(S-CGA-SHAPE): a launch with a preferred cluster dimension runs each group of
    // CTAs as one preferred cluster or as regular clusters, a hardware choice per group. Both
    // shapes' clusters are sliced; the cross-cluster check pairs every two clusters that can
    // coexist (disjoint CTAs), and each group must write the same bytes either way, so every
    // mix ends with the same memory.
    let shapes: Vec<Arc<crate::machine::Launch>> = std::iter::once(launch.clone())
        .chain(launch.preferred_shape().map(Arc::new))
        .collect();
    let out_init = m.outputs();
    let out_allocs: Vec<(usize, usize)> = {
        let mut at = 0;
        launch
            .outputs
            .iter()
            .map(|&a| {
                let o = at;
                at += m.gmem[a].bytes.len();
                (a, o)
            })
            .collect()
    };
    drop(m);
    let only: Option<Vec<u32>> = std::env::var("PTXV_SLICE_ONLY")
        .ok()
        .map(|only| only.split(',').filter_map(|x| x.parse().ok()).collect());
    let mut all = Verdict {
        states: 0,
        transitions: 0,
        complete: true,
        terminal_states: 0,
        distinct_outputs: 0,
        findings: vec![],
        assumptions: vec![],
        notes: vec![],
        seconds: 0.0,
        outputs: vec![],
    };
    let (mut certified, mut explored, mut nslices) = (0usize, 0usize, 0usize);
    // A slice's global footprint, sparse: (allocation, 64-byte word, read bits, written bits;
    // fp32 atomic adds count as writes here, no N-ATOMIC relaxation), sorted.
    type Bits = Vec<(u32, u32, u64, u64)>;
    let sparse = |snap: &[(Vec<u64>, Vec<u64>, Vec<u64>)]| -> Bits {
        let mut v = Vec::new();
        for (a, (r, w, x)) in snap.iter().enumerate() {
            for (k, ((&r, &w), &x)) in r.iter().zip(w).zip(x).enumerate() {
                if r | w | x != 0 {
                    v.push((a as u32, k as u32, r, w | x));
                }
            }
        }
        v
    };
    // (shape, cluster, its CTAs, footprint)
    let mut foots: Vec<(usize, u32, Vec<u32>, Bits)> = Vec::new();
    // Per shape: the outputs of the whole launch, assembled from the slices' writes (None
    // when some slice did not end with one output, or only some clusters ran).
    // Single-grid launches only (a slice of a later grid also runs the earlier grids, whose
    // writes its footprint does not record).
    let assemble = only.is_none() && launch.grids.len() == 1;
    let mut assembled: Vec<Option<Vec<u8>>> =
        vec![assemble.then(|| out_init.clone()); shapes.len()];
    for (si, shape) in shapes.iter().enumerate() {
        let mk = || {
            let m = make();
            if si == 0 {
                m
            } else {
                m.with_cluster_shape(shape.clone())
            }
        };
        let mut ids = crate::explore::clusters(&mk());
        if let Some(keep) = &only {
            ids.retain(|c| keep.contains(c));
        }
        if si > 0 {
            eprintln!(
                "slices with every cluster in the preferred shape (SR_CgaSize 1, S-CGA-SHAPE):"
            );
            all.notes.push(format!(
                "preferred cluster shape: {} more slices with every cluster preferred (SR_CgaSize 1, S-CGA-SHAPE)",
                ids.len()
            ));
        }
        // The slices' certificates run in parallel (PTXV_SLICE_THREADS, default 8; each run is
        // independent and single-threaded, and holds its own machine: mm_4k_8k's slices take
        // about 3 GB each); results are taken in cluster order.
        let nthreads = std::env::var("PTXV_SLICE_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8)
            .clamp(1, ids.len().max(1));
        type Done = (u32, Arc<GFoot>, Outcome, Option<Machine>);
        let mut done: BTreeMap<usize, Done> = BTreeMap::new();
        let (tx_res, rx_res) = std::sync::mpsc::channel::<(usize, Done)>();
        let (tx_job, rx_job) =
            std::sync::mpsc::sync_channel::<(usize, u32, Machine, Arc<GFoot>)>(1);
        let rx_job = std::sync::Mutex::new(rx_job);
        std::thread::scope(|scope| {
            for _ in 0..nthreads {
                let tx_res = tx_res.clone();
                let rx_job = &rx_job;
                scope.spawn(move || {
                    loop {
                        let job = rx_job.lock().unwrap().recv();
                        let Ok((i, c, m, g)) = job else { break };
                        let o = certify(slice_machine(m.clone(), c));
                        // The machine is kept only for the exploration that replaces a
                        // failed certificate.
                        let keep = o.verdict.is_none().then_some(m);
                        if tx_res.send((i, (c, g, o, keep))).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(tx_res);
            let mut next = 0usize;
            let mut fed = ids.iter().enumerate();
            let mut feeding = true;
            while next < ids.len() {
                if feeding {
                    match fed.next() {
                        Some((i, &c)) => {
                            let mut m = mk();
                            let sizes: Vec<usize> = m.gmem.iter().map(|g| g.bytes.len()).collect();
                            let mut g = GFoot::new(&sizes);
                            g.cluster = Some(c);
                            let g = Arc::new(g);
                            m.gfoot = Some(g.clone());
                            tx_job.send((i, c, m, g)).unwrap();
                        }
                        None => feeding = false,
                    }
                }
                while let Ok((i, d)) = if feeding {
                    rx_res.try_recv().map_err(|_| ())
                } else {
                    rx_res.recv().map_err(|_| ())
                } {
                    done.insert(i, d);
                    if !feeding {
                        break;
                    }
                }
                while let Some((c, g, o, keep)) = done.remove(&next) {
                    next += 1;
                    nslices += 1;
                    let v = match o.verdict {
                        Some(v) => {
                            certified += 1;
                            v
                        }
                        None => {
                            let m = keep.expect("kept for exploration");
                            report_fallback(&format!("cluster {c}"), &o);
                            explored += 1;
                            let weak = (m.weak_stop > 0).then_some(m.weak_stop == 2);
                            let races = run_races(&o, weak, &format!("cluster {c}"));
                            let mut v = crate::explore::explore_slice(m, max_states, Some(c));
                            v.notes.push(format!(
                                "cluster {c}: certificate failed ({}), explored",
                                o.reasons.first().map(String::as_str).unwrap_or("?")
                            ));
                            merge_run_races(&mut v, races);
                            v
                        }
                    };
                    per_slice(c, &v);
                    all.states += v.states;
                    all.transitions += v.transitions;
                    all.complete &= v.complete;
                    all.terminal_states += v.terminal_states;
                    all.distinct_outputs = all.distinct_outputs.max(v.distinct_outputs);
                    let snap = g.snapshot();
                    match (&mut assembled[si], v.outputs.as_slice()) {
                        (Some(out), [o]) if o.len() == out.len() => {
                            for &(a, at) in &out_allocs {
                                let (_, w, x) = &snap[a];
                                for (k, (&wk, &xk)) in w.iter().zip(x).enumerate() {
                                    let mut bits = wk | xk;
                                    while bits != 0 {
                                        let b = 64 * k + bits.trailing_zeros() as usize;
                                        bits &= bits - 1;
                                        if let Some(&byte) = o.get(at + b) {
                                            out[at + b] = byte;
                                        }
                                    }
                                }
                            }
                        }
                        (slot, _) => *slot = None,
                    }
                    for f in v.findings {
                        if !all.findings.iter().any(|x| x.sig == f.sig) {
                            all.findings.push(f);
                        }
                    }
                    for a in v.assumptions {
                        if !all.assumptions.contains(&a) {
                            all.assumptions.push(a);
                        }
                    }
                    for n in v.notes {
                        if !n.starts_with("well-synchronized") && !all.notes.contains(&n) {
                            all.notes.push(n);
                        }
                    }
                    foots.push((si, c, shape.cluster_ctas(c), sparse(&snap)));
                }
            }
            drop(tx_job);
        });
    }
    all.notes.insert(
        0,
        format!(
            "one cluster at a time ({nslices} clusters): {certified} certified well-synchronized (E-WELLSYNC, one run each), {explored} explored"
        ),
    );
    if shapes.len() > 1 && !all.assumptions.iter().any(|a| a == "S-CGA-SHAPE") {
        all.assumptions.push("S-CGA-SHAPE".into());
    }
    if shapes.len() > 1 && only.is_none() {
        // Each preferred cluster and the regular clusters in its CTAs write the same bytes.
        let writes = |f: &Bits| -> BTreeMap<(u32, u32), u64> {
            let mut m = BTreeMap::new();
            for &(a, k, _, w) in f {
                if w != 0 {
                    *m.entry((a, k)).or_insert(0) |= w;
                }
            }
            m
        };
        for (_, cp, ctas_p, fp) in foots.iter().filter(|f| f.0 == 1) {
            let mut u: BTreeMap<(u32, u32), u64> = BTreeMap::new();
            for f in foots
                .iter()
                .filter(|f| f.0 == 0 && f.2.iter().all(|c| ctas_p.contains(c)))
            {
                for (key, w) in writes(&f.3) {
                    *u.entry(key).or_insert(0) |= w;
                }
            }
            if u != writes(fp) {
                all.distinct_outputs = all.distinct_outputs.max(2);
                all.notes.push(format!(
                    "preferred cluster {cp} (CTAs {ctas_p:?}) writes other global bytes than the regular clusters of those CTAs: a launch mixing cluster shapes ends differently (S-CGA-SHAPE)"
                ));
            }
        }
    }
    match assembled.as_slice() {
        [Some(a)] => all.outputs = vec![a.clone()],
        [Some(a), Some(b)] => {
            if a == b {
                all.outputs = vec![a.clone()];
            } else {
                all.outputs = vec![a.clone(), b.clone()];
                all.distinct_outputs = all.distinct_outputs.max(2);
                all.notes.push(
                    "the regular and the preferred cluster shape end with different outputs (S-CGA-SHAPE)".into(),
                );
            }
        }
        [_, _] if assemble => all.findings.push(Finding {
            kind: "unmodeled".into(),
            sig: "cga-shape-outputs".into(),
            msg: "preferred cluster shape: a slice did not end with one output, so the two shapes' outputs could not be compared (S-CGA-SHAPE)".into(),
            trace: vec![],
            ..Default::default()
        }),
        _ => {}
    }
    // Cross-cluster conflicts, as in `explore_sliced`.
    let ordered = |gi: usize, gj: usize| {
        let mut g = gj;
        while g > gi {
            if launch.grids[g].pdl {
                return false;
            }
            match launch.prev_grid(g) {
                Some(p) => g = p,
                None => return false,
            }
        }
        g == gi
    };
    // Indexed by word: only words some slice writes can conflict, and only slices that touch
    // such a word are compared (the pairwise scan of dense bitsets was quadratic in launch
    // size: hours at 384 slices of 160 MB).
    // (allocation, word) -> (slice, read bits, written bits) of every slice touching it
    type Touch = Vec<(usize, u64, u64)>;
    let mut written: HashMap<(u32, u32), Touch> = HashMap::new();
    for f in &foots {
        for &(a, k, _, w) in &f.3 {
            if w != 0 {
                written.entry((a, k)).or_default();
            }
        }
    }
    for (i, f) in foots.iter().enumerate() {
        for &(a, k, r, w) in &f.3 {
            if let Some(v) = written.get_mut(&(a, k)) {
                v.push((i, r, w));
            }
        }
    }
    // (slice i, slice j, allocation) -> first conflicting byte
    let mut hits: BTreeMap<(usize, usize, u32), u64> = BTreeMap::new();
    for (&(a, k), v) in &written {
        for (x, &(i, ri, wi)) in v.iter().enumerate() {
            for &(j, rj, wj) in &v[x + 1..] {
                let y = (wi & (rj | wj)) | (wj & (ri | wi));
                if y == 0 {
                    continue;
                }
                let ((si, ci, ctas_i, _), (sj, cj, ctas_j, _)) = (&foots[i], &foots[j]);
                // Clusters of the two shapes that share CTAs are alternatives, never both.
                if si != sj && ctas_i.iter().any(|c| ctas_j.contains(c)) {
                    continue;
                }
                let (gi, gj) = ((ci >> 20) as usize, (cj >> 20) as usize);
                if gi != gj && (ordered(gi, gj) || ordered(gj, gi)) {
                    continue;
                }
                let b = 64 * k as u64 + y.trailing_zeros() as u64;
                let e = hits.entry((i.min(j), i.max(j), a)).or_insert(b);
                *e = (*e).min(b);
            }
        }
    }
    for ((i, j, alloc), b) in hits {
        let (ci, cj) = (foots[i].1, foots[j].1);
        all.findings.push(Finding {
            kind: "race".into(),
            sig: format!("cross-cluster:{alloc}"),
            msg: format!(
                "clusters {ci} and {cj} both access global byte {b:#x} of allocation {alloc}, at least one writing, with no synchronization between clusters"
            ),
            trace: vec![],
            ..Default::default()
        });
    }
    all.findings
        .sort_by(|a, b| (&a.kind, &a.sig).cmp(&(&b.kind, &b.sig)));
    all.seconds = t0.elapsed().as_secs_f64();
    all
}

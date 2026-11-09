/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! C-RING (ASSUMPTIONS.md E-WELLSYNC): polls of monotone counters whose observed value may
//! steer the rest of the program.
//!
//! A ring buffer's consumer polls the producer's tail and copies the slots between its head
//! and the tail it saw; the producer polls the consumer's head before reusing slots. Which
//! tail value a poll sees depends on the schedule, so the consumer's sequence of events (how
//! it chunks the slots) does too, and the plain certificate (one run, every schedule repeats
//! its events) does not apply. C-RING decides such polls:
//!
//! * **Counters.** A word is a counter when every write to it in the run is a strong store
//!   of one thread, the values never decrease (unsigned), and nothing else (no plain write,
//!   no atomic) writes it. The values a thread writes are the same in every schedule the
//!   certificate covers (induction), so a check on the run's values is a check on all.
//! * **Candidates.** A poll's exiting read may read the write it read in the run, or any
//!   other write of the counter from the latest one that happens-before the read on (the
//!   counter's earlier values are excluded by coherence; later ones may be read in another
//!   schedule). Writes that happen-after the read are kept too (over-approximation: a forced
//!   run on such a value gets stuck and is dropped, see below).
//! * **Classes.** The exiting iteration is replayed on a copy of the machine from its start
//!   (the snapshot taken at the loop head, or before the load of a straight poll loop) with
//!   the counter's bytes and synchronization source replaced by each candidate's. A
//!   candidate either fails (back at the head: every register the loop writes that is live at
//!   the head must be the snapshot's, so the failed iteration is invisible) or exits; exiting
//!   candidates are grouped by their continuation: exit pcs and every register the loop
//!   writes that is live there. Candidates in one class leave the same program state except
//!   for the synchronization they acquire, which grows along the counter's sequence (one
//!   writer); the earliest member acquires the least.
//! * **No lost wakeup.** In candidate order, once a candidate exits every later one exits:
//!   a wait on a monotone counter, once satisfied, stays satisfied.
//! * **Choices.** A poll with one class is decided by the run (its clock lowered to the
//!   earliest member's, as C-POLL does). A poll with several classes is a *choice*: the
//!   certificate is run once per sequence of classes (`certify_inner`'s search), forcing
//!   each choice's read to the earliest member of its class (`RingCtl`). Every execution
//!   reads, at each choice, some member of some class; the search covers every class at
//!   every choice reached, and the forced run for a class sequence is a certificate run whose
//!   every other condition is checked as usual. A forced run that gets stuck because a
//!   forced value is never written while the poll would exit on a value already written is
//!   infeasible (no execution makes that choice) and is dropped.
//! * **Chunking-obliviousness** is then checked semantically: every run of the search must
//!   certify, and all must agree on the outputs. A consumer whose stored values, addresses or
//!   control depend on how it chunked (a threshold on the batch size, a batch counter) gives
//!   different outputs for different class sequences and is refused.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

use ptxv_ir::*;

use super::Ev;
use super::SyncLog;
use super::describe_ev;
use super::loc_of;
use super::psync_le;
use super::set_mem_bytes;
use crate::exec::Loc;
use crate::machine::*;

/// A counter word: (space 0 smem / 1 gmem, CTA or allocation, first byte).
pub type Word = (u8, u32, u64);

/// A write of a counter word: its index in the word's write sequence, or None for the value
/// the word held before its first write in the run.
pub type WIdx = Option<usize>;

/// A choice point: (first lane that read the counter, poll-loop head, how many times that
/// lane had left this loop before).
pub type Choice = (u32, u32, u32);

/// One forced read: the exiting read of a poll loop returns write `idx` of `word`.
#[derive(Clone, Debug)]
pub struct Forced {
    pub word: Word,
    pub n: usize,
    pub idx: WIdx,
    /// The value the choice was made on (checked against this run's history).
    pub bytes: Vec<u8>,
    /// For `idx` None: the synchronization source before the first write.
    pub cell: Option<SyncCell>,
}

/// Strong writes to a forced word, in order: bytes after the write, synchronization source.
pub type WriteHist = Vec<(Vec<u8>, Option<SyncCell>)>;

/// The forced reads of one C-RING run (`Machine::ring`), and what the machine needs to apply
/// them: the forced words' write histories and how often each lane left each poll loop.
#[derive(Clone, Debug, Default)]
pub struct RingCtl {
    /// (lane, loop head, exits of that lane from the loop so far) -> forced read.
    pub force: BTreeMap<(u32, u32, u32), Forced>,
    /// Strong writes to forced words, in order: bytes after the write, synchronization source.
    pub hist: BTreeMap<Word, WriteHist>,
    /// Width of each forced word.
    pub width: BTreeMap<Word, usize>,
    /// Exits per (lane, loop head) so far.
    pub ord: BTreeMap<(u32, u32), u32>,
    /// A forced write index whose bytes differ from the choice's (the counter's history
    /// depends on the choices).
    pub mismatch: Option<String>,
    /// C-TICKET (ticket.rs): the instructions that write ticket words, which this run gates.
    pub tick_pcs: BTreeSet<u32>,
    /// C-TICKET: the ticket words.
    pub tick_words: BTreeSet<Word>,
    /// C-TICKET: the forced claimant prefix of each ticket word.
    pub tick_force: super::ticket::TickForce,
    /// C-TICKET: the writers so far per word, in order.
    pub tick_claims: BTreeMap<Word, Vec<u32>>,
}

impl RingCtl {
    pub fn new(force: BTreeMap<(u32, u32, u32), Forced>) -> RingCtl {
        let mut c = RingCtl {
            force,
            ..Default::default()
        };
        for f in c.force.values() {
            c.hist.entry(f.word).or_default();
            c.width.insert(f.word, f.n);
        }
        c
    }
}

impl Machine {
    /// The poll loop (head pc) whose load `pc` is: a straight poll load (`polls`) or any pc
    /// inside a warp-level poll loop (`wpolls`).
    pub fn ring_head(&self, pc: u32) -> Option<u32> {
        if let Some(&(h, _)) = self.polls.get(&pc) {
            return Some(h);
        }
        self.wpolls
            .range(..=pc)
            .next_back()
            .filter(|(_, &b)| pc <= b)
            .map(|(&h, _)| h)
    }

    /// The forced read (bytes, synchronization source) of thread `t`'s load at `pc` from
    /// `loc`, if this C-RING run forces it and the forced write exists.
    pub fn ring_value(
        &mut self,
        t: usize,
        pc: u32,
        loc: Loc,
    ) -> Option<(Vec<u8>, Option<SyncCell>)> {
        let r = self.ring.as_ref()?;
        let h = self.ring_head(pc)?;
        let k = r.ord.get(&(t as u32, h)).copied().unwrap_or(0);
        let f = r.force.get(&(t as u32, h, k))?;
        let w = word_of(loc)?;
        if w != f.word {
            return None;
        }
        let got = match f.idx {
            None => Some((f.bytes.clone(), f.cell.clone())),
            Some(j) => r.hist.get(&w).and_then(|v| v.get(j)).cloned(),
        };
        let (bytes, cell) = got?;
        if bytes != f.bytes && r.mismatch.is_none() {
            let msg = format!(
                "C-RING: write {} of the counter at {} holds {:?} in this run but {:?} where the choice was made (the counter's values depend on the choices)",
                f.idx.map_or(-1, |j| j as i64),
                self.k.describe(pc as usize),
                bytes,
                f.bytes
            );
            Arc::make_mut(self.ring.as_mut().unwrap()).mismatch = Some(msg);
        }
        Some((bytes, cell))
    }

    /// May `members` enter the poll loop at `head` (or run its straight poll load)? Not while
    /// a member's forced write has not happened yet.
    pub fn ring_ready(&self, members: &[usize], head: u32) -> bool {
        let Some(r) = self.ring.as_ref() else {
            return true;
        };
        members.iter().all(|&t| {
            let k = r.ord.get(&(t as u32, head)).copied().unwrap_or(0);
            match r.force.get(&(t as u32, head, k)) {
                Some(Forced {
                    idx: Some(j), word, ..
                }) => r.hist.get(word).is_some_and(|v| v.len() > *j),
                _ => true,
            }
        })
    }

    /// Record a strong write to a forced word (after the write and its `sync_publish`).
    pub fn ring_wrote(&mut self, key: (u8, u32, u64)) {
        let Some(r) = self.ring.as_ref() else {
            return;
        };
        let Some(&n) = r.width.get(&key) else {
            return;
        };
        let loc = loc_of((key.0, key.1), key.2);
        let bytes = mem_now(self, loc, n);
        let cell = self.syncs.get(&key).cloned();
        Arc::make_mut(self.ring.as_mut().unwrap())
            .hist
            .get_mut(&key)
            .unwrap()
            .push((bytes, cell));
    }
}

fn word_of(loc: Loc) -> Option<Word> {
    match loc {
        Loc::Shared { cta, off } => Some((0, cta as u32, off as u64)),
        Loc::Global { alloc, off } => Some((1, alloc as u32, off)),
        Loc::Param { .. } => None,
    }
}

fn mem_now(m: &Machine, loc: Loc, n: usize) -> Vec<u8> {
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

// ---------------------------------------------------------------- recording

/// A load by a member of a poll loop during one of its real (not blocked) iterations.
#[derive(Clone, Debug)]
pub(super) struct RRead {
    t: usize,
    pc: u32,
    word: Word,
    n: usize,
    ev: Ev,
    vc: VC,
    /// Iteration of the loop entry (index into `snaps`).
    iter: usize,
    /// The bytes it read.
    bytes: Vec<u8>,
}

/// A loop entry by a split: one snapshot per real iteration (taken at its start) and the
/// loads.
#[derive(Clone, Debug)]
pub(super) struct Entry {
    h: u32,
    b: u32,
    /// Warp-level loop (`wpolls`); else a straight poll loop (`polls`), whose iteration is
    /// replayed from its load.
    warp: bool,
    members: Vec<usize>,
    snaps: Vec<super::Snap>,
    reads: Vec<RRead>,
    /// Exits of the first member from this loop before (the choice ordinal).
    ords: Vec<u32>,
}

/// C-RING's part of the certificate log.
#[derive(Debug, Default)]
pub(super) struct RingLog {
    /// Open warp-level loop entries, by warp.
    open: BTreeMap<u32, Entry>,
    /// Straight poll loops: the current iteration's snapshot (taken before the load) per thread.
    straight: BTreeMap<usize, Entry>,
    /// Loop entries that exited.
    pub(super) done: Vec<Entry>,
    pub(super) fails: BTreeMap<String, String>,
    snaps: usize,
}

/// Snapshots kept per run before C-RING gives up.
const MAX_SNAPS: usize = 4096;

fn fail(rl: &mut RingLog, key: String, msg: String) {
    rl.fails.entry(key).or_insert(msg);
}

/// Hook (`pre_inst`): iteration starts and loads of poll loops.
pub(super) fn pre(m: &Machine, log: &mut SyncLog, members: &[usize], active: &[usize], pc: u32) {
    let code = &m.k.code;
    let inst = &code[pc as usize];
    // A warp-level loop iteration starts at the head.
    if let Some(&b) = m.wpolls.get(&pc) {
        let w = m.threads[members[0]].warp;
        let rl = &mut log.ring;
        if rl.snaps >= MAX_SNAPS {
            fail(
                rl,
                "snaps".into(),
                format!("C-RING: more than {MAX_SNAPS} poll iterations (not checked)"),
            );
            return;
        }
        rl.snaps += 1;
        let snap = super::Snap(Arc::new(m.clone()));
        let ord = |t: usize| {
            m.ring
                .as_ref()
                .and_then(|r| r.ord.get(&(t as u32, pc)).copied())
                .unwrap_or(0)
        };
        match rl.open.get_mut(&w) {
            Some(e) if e.h == pc && e.members == members => e.snaps.push(snap),
            _ => {
                if let Some(e) = rl.open.remove(&w) {
                    fail(
                        rl,
                        format!("ring-split:{}", e.h),
                        format!(
                            "C-RING: warp {w} re-enters the poll loop at {} with other lanes before leaving it",
                            m.k.describe(e.h as usize)
                        ),
                    );
                }
                rl.open.insert(
                    w,
                    Entry {
                        h: pc,
                        b,
                        warp: true,
                        members: members.to_vec(),
                        snaps: vec![snap],
                        reads: vec![],
                        ords: members.iter().map(|&t| ord(t)).collect(),
                    },
                );
            }
        }
    }
    let Op::Ld {
        space, bytes, addr, ..
    } = &inst.op
    else {
        return;
    };
    let straight = m.polls.get(&pc).copied();
    let in_warp_loop = straight.is_none()
        && m.wpolls
            .range(..=pc)
            .next_back()
            .is_some_and(|(_, &b)| pc <= b);
    if straight.is_none() && !in_warp_loop {
        return;
    }
    let w = m.threads[members[0]].warp;
    for &t in active {
        let Ok(loc) = m
            .addr(t, addr)
            .and_then(|a| m.resolve(t, *space, a, *bytes as usize))
        else {
            continue; // the step reports it
        };
        let Some(word) = word_of(loc) else { continue };
        let vc = m.threads[t].vc.clone();
        let ev = log.ev(t as u32, &vc, pc);
        let mut bytes_now = mem_now(m, loc, *bytes as usize);
        // A forced read returns the forced bytes (the executor applies it).
        if let Some(r) = m.ring.as_ref() {
            let h = m.ring_head(pc).unwrap();
            let k = r.ord.get(&(t as u32, h)).copied().unwrap_or(0);
            if let Some(f) = r.force.get(&(t as u32, h, k)).filter(|f| f.word == word) {
                bytes_now = f.bytes.clone();
            }
        }
        let rl = &mut log.ring;
        if let Some((h, b)) = straight {
            if rl.snaps >= MAX_SNAPS {
                fail(
                    rl,
                    "snaps".into(),
                    format!("C-RING: more than {MAX_SNAPS} poll iterations (not checked)"),
                );
                return;
            }
            rl.snaps += 1;
            let ord = m
                .ring
                .as_ref()
                .and_then(|r| r.ord.get(&(t as u32, h)).copied())
                .unwrap_or(0);
            rl.straight.insert(
                t,
                Entry {
                    h,
                    b,
                    warp: false,
                    members: vec![t],
                    snaps: vec![super::Snap(Arc::new(m.clone()))],
                    reads: vec![RRead {
                        t,
                        pc,
                        word,
                        n: *bytes as usize,
                        ev,
                        vc,
                        iter: 0,
                        bytes: bytes_now,
                    }],
                    ords: vec![ord],
                },
            );
        } else {
            match rl.open.get_mut(&w) {
                Some(e) if e.members.contains(&t) => {
                    let iter = e.snaps.len() - 1;
                    e.reads.push(RRead {
                        t,
                        pc,
                        word,
                        n: *bytes as usize,
                        ev,
                        vc,
                        iter,
                        bytes: bytes_now,
                    });
                }
                // A load of a warp-level poll loop by a thread that did not enter it at its
                // head: C-ORDER no longer sees it (strong_access), so it must not go unchecked.
                _ => fail(
                    rl,
                    format!("ring-entry:{pc}"),
                    format!(
                        "C-RING: {} runs the poll load at {} without entering its loop at the head (not checked)",
                        m.actor_name(t as u32),
                        m.k.describe(pc as usize)
                    ),
                ),
            }
        }
    }
}

/// Hook (`post_inst`): members that left a poll loop with this instruction close their loop
/// entry; the exiting read's clock is lowered to the earliest write of its class known so
/// far (natural reads only: a forced read is the earliest member already).
pub(super) fn post(m: &mut Machine, log: &mut SyncLog, members: &[usize], pc: u32) {
    post_straight(m, log, members, pc);
    let Some((h, b)) = m
        .wpolls
        .range(..=pc)
        .next_back()
        .filter(|(_, &b)| pc <= b)
        .map(|(&h, &b)| (h, b))
    else {
        return;
    };
    let left: Vec<usize> = members
        .iter()
        .copied()
        .filter(|&t| {
            let th = &m.threads[t];
            th.st == St::Exited || th.pc < h || th.pc > b
        })
        .collect();
    if left.is_empty() {
        return;
    }
    let w = m.threads[members[0]].warp;
    let Some(e) = log.ring.open.remove(&w) else {
        return;
    };
    if e.h != h || left != e.members {
        fail(
            &mut log.ring,
            format!("ring-part:{h}"),
            format!(
                "C-RING: only some lanes of warp {w} leave the poll loop at {} at {} (not checked)",
                m.k.describe(h as usize),
                m.k.describe(pc as usize)
            ),
        );
        return;
    }
    bump(m, &e.members, h);
    lower(m, log, &e);
    log.ring.done.push(e);
}

/// Straight poll loops: a thread that leaves the loop (its back branch falls through, or an
/// exit) closes the entry its last load opened.
fn post_straight(m: &mut Machine, log: &mut SyncLog, members: &[usize], pc: u32) {
    for &t in members {
        let Some(e) = log.ring.straight.get(&t) else {
            continue;
        };
        if pc < e.h || pc > e.b {
            continue;
        }
        let th = &m.threads[t];
        if th.st != St::Exited && th.pc >= e.h && th.pc <= e.b {
            continue;
        }
        let e = log.ring.straight.remove(&t).unwrap();
        bump(m, &[t], e.h);
        log.ring.done.push(e);
    }
}

fn bump(m: &mut Machine, members: &[usize], h: u32) {
    if let Some(r) = m.ring.as_mut() {
        let r = Arc::make_mut(r);
        for &t in members {
            *r.ord.entry((t as u32, h)).or_default() += 1;
        }
    }
}

/// The counter writes of `word` in the run (strong writes with the same bytes, in order) and
/// whether the word is a counter (else why not).
pub(super) struct Hist<'a> {
    pub writes: Vec<&'a super::SAcc>,
    pub why_not: Option<String>,
}

pub(super) fn hist<'a>(m: &Machine, log: &'a SyncLog, word: Word, n: usize) -> Hist<'a> {
    let (key, lo) = ((word.0, word.1), word.2);
    let mut writes: Vec<&super::SAcc> = log
        .strong
        .iter()
        .filter(|a| a.write && a.key == key && a.lo < lo + n as u64 && lo < a.hi)
        .collect();
    writes.sort_by_key(|a| a.ev.seq);
    let mut why = None;
    if let Some(a) = writes.iter().find(|a| a.lo != lo || a.hi != lo + n as u64) {
        why = Some(format!(
            "{} writes it with another width or offset",
            describe_ev(m, &a.ev)
        ));
    } else if let Some(a) = writes.iter().find(|a| a.rmw.is_some()) {
        why = Some(format!(
            "{} writes it with an atomic",
            describe_ev(m, &a.ev)
        ));
    } else if let Some(a) = writes.iter().find(|a| a.ev.actor != writes[0].ev.actor) {
        why = Some(format!(
            "it has two writers ({} and {})",
            describe_ev(m, &writes[0].ev),
            describe_ev(m, &a.ev)
        ));
    } else {
        let val = |b: &[u8]| b.iter().rev().fold(0u64, |x, &y| (x << 8) | y as u64);
        let mut prev = writes.first().map(|a| val(&a.old));
        for a in &writes {
            let v = val(&a.val);
            if prev.is_some_and(|p| v < p) {
                why = Some(format!(
                    "{} decreases it ({} after {})",
                    describe_ev(m, &a.ev),
                    v,
                    prev.unwrap()
                ));
                break;
            }
            prev = Some(v);
        }
    }
    // A plain write would end the value sequence (and the release sequence): not a counter.
    if why.is_none() {
        if let Some(sh) = plain_writer(m, word, n, &writes) {
            why = Some(sh);
        }
    }
    Hist {
        writes,
        why_not: why,
    }
}

/// A plain write to the word after the first strong one, from the race shadow's last writer
/// (a later plain write would be the shadow's last write).
fn plain_writer(m: &Machine, word: Word, n: usize, writes: &[&super::SAcc]) -> Option<String> {
    let loc = loc_of((word.0, word.1), word.2);
    let last = writes.last()?;
    for i in 0..n {
        let w = match loc {
            Loc::Shared { cta, off } => m.smem[cta as usize].sh.get(off as usize + i).w,
            Loc::Global { alloc, off } => m.gmem[alloc].sh.get(off as usize + i).w,
            Loc::Param { .. } => None,
        };
        if let Some(w) = w {
            if w.strong == 0 && (w.actor != last.ev.actor || w.clock > last.ev.clock) {
                return Some(format!(
                    "a plain write by {} at {} ends its sequence",
                    m.actor_name(w.actor),
                    m.k.describe(w.pc as usize)
                ));
            }
        }
    }
    None
}

/// The write a C-RING read read: the forced one, or the counter's last write before it in
/// the run.
fn read_index(m: &Machine, e: &Entry, x: &RRead, hs: &Hist) -> WIdx {
    let i = e.members.iter().position(|&t| t == x.t).unwrap_or(0);
    if let Some(f) = m
        .ring
        .as_ref()
        .and_then(|c| c.force.get(&(x.t as u32, e.h, e.ords[i])))
        .filter(|f| f.word == x.word)
    {
        return f.idx;
    }
    hs.writes
        .iter()
        .filter(|a| a.ev.seq < x.ev.seq)
        .count()
        .checked_sub(1)
}

/// The earliest write a read may read (coherence): the latest write that happens-before it,
/// and no earlier than any write a read that happens-before it (its own earlier polls
/// included) read. A relaxed counter store after a release fence is not itself
/// happens-before its reader, so the first bound alone would let a poll read back a value
/// older than one it already saw.
fn coh_lb(m: &Machine, log: &SyncLog, hs: &Hist, word: Word, t: usize, ev: &Ev, vc: &VC) -> WIdx {
    let mut lb = hs
        .writes
        .iter()
        .rposition(|a| a.ev.hb(t as u32, ev.seq, vc));
    for e in &log.ring.done {
        for x in &e.reads {
            if x.word != word || x.ev.seq >= ev.seq || !x.ev.hb(t as u32, ev.seq, vc) {
                continue;
            }
            lb = lb.max(read_index(m, e, x, hs));
        }
    }
    lb
}

/// Is `r` live at `pc` in the model? As wellsync's `live_at`, with two refinements:
/// * `CALL.ABS` is what the model executes: a no-op that reads its target (vprintf, E-CLOCK)
///   or a trap (`__assertfail`), not a call that may read every register (DeepEP's timeout
///   handlers call vprintf, which otherwise made every register live wherever one is
///   reachable);
/// * a write under a guard kills `r` for later reads under the same guard until the guard's
///   predicate is rewritten (`@P0 MOV R4, R64; @P0 MOV R5, R65; @P0 STG [R4.64]`, as
///   spin.rs `live_after`).
fn live_at(code: &[Inst], pc: usize, r: Reg) -> Option<usize> {
    type Cond = BTreeSet<(Reg, bool)>;
    let mut seen: BTreeSet<(usize, Cond)> = BTreeSet::new();
    let mut work: Vec<(usize, Cond)> = vec![(pc, Cond::new())];
    let mut calls = false;
    while let Some((p, mut cond)) = work.pop() {
        if p >= code.len() || !seen.insert((p, cond.clone())) {
            continue;
        }
        let inst = &code[p];
        let killed = inst
            .guard
            .reg
            .is_some_and(|g| g != r && cond.contains(&(g, inst.guard.neg)));
        if inst.guard.reg == Some(r) || (!killed && inst.op.src_regs().contains(&r)) {
            return Some(p);
        }
        let always = inst.guard.is_always();
        let mut dst = inst.op.dst_regs();
        if let Op::Ld { d, n, .. } = &inst.op {
            for k in 0..(*n).max(1) as u16 {
                dst.push(Reg {
                    file: d.file,
                    idx: d.idx + if d.file == File::V { 0 } else { k },
                });
            }
        }
        if dst.contains(&r) {
            match inst.guard.reg {
                _ if always => continue,
                Some(g) => {
                    cond.insert((g, inst.guard.neg));
                }
                None => {}
            }
        }
        cond.retain(|(g, _)| !dst.contains(g));
        match &inst.op {
            Op::Bra {
                target,
                cond: c,
                tid0_falls,
                div,
                ..
            } => {
                work.push((*target, cond.clone()));
                if !always || c.is_some() || *tid0_falls || div.is_some() {
                    work.push((p + 1, cond));
                }
            }
            Op::Exit if always => {}
            Op::Call { target } => {
                calls = true;
                work.push((*target, cond.clone()));
                work.push((p + 1, cond));
            }
            Op::Ret { .. } if !calls => return Some(p),
            Op::Ret { .. } => {}
            Op::Brx { targets, .. } => work.extend(targets.iter().map(|&x| (x, cond.clone()))),
            _ => work.push((p + 1, cond)),
        }
    }
    None
}

/// Registers a poll loop writes (loads' destinations included).
fn loop_written(code: &[Inst], h: u32, b: u32) -> Vec<Reg> {
    let mut out: Vec<Reg> = Vec::new();
    for inst in &code[h as usize..=b as usize] {
        out.extend(inst.op.dst_regs());
        if let Op::Ld { d, n, .. } = &inst.op {
            for k in 0..(*n).max(1) as u16 {
                out.push(Reg {
                    file: d.file,
                    idx: d.idx + if d.file == File::V { 0 } else { k },
                });
            }
        }
    }
    out.sort_by_key(|r| (r.file as u8, r.idx));
    out.dedup();
    out
}

fn reg_val(x: &Machine, t: usize, g: Reg) -> Option<u64> {
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
}

/// The program state an exit leaves, as far as the rest of the run can tell: per member its
/// pc (u32::MAX: exited) and the loop-written registers live there.
type Key = Vec<(u32, Vec<(Reg, Option<u64>)>)>;

fn exit_key(x: &Machine, members: &[usize], written: &[Reg]) -> Key {
    let code = &x.k.code;
    members
        .iter()
        .map(|&t| {
            let th = &x.threads[t];
            if th.st == St::Exited {
                return (u32::MAX, vec![]);
            }
            let pc = th.pc;
            let regs = written
                .iter()
                .filter(|&&g| live_at(code, pc as usize, g).is_some())
                .map(|&g| (g, reg_val(x, t, g)))
                .collect();
            (pc, regs)
        })
        .collect()
}

/// Is `r` read, on some path of one iteration (from the head `h`, staying inside [h, b]),
/// before the iteration writes it? Then a failed iteration's value of `r` reaches the next.
fn loop_live(code: &[Inst], h: u32, b: u32, r: Reg) -> Option<usize> {
    let mut seen = vec![false; code.len() + 1];
    let mut work = vec![h as usize];
    while let Some(p) = work.pop() {
        if p < h as usize || p > b as usize || seen[p] {
            continue;
        }
        seen[p] = true;
        let inst = &code[p];
        let mut reads = inst.op.src_regs();
        reads.extend(inst.guard.reg);
        if let Op::Plop3 { a, b, c, .. } = &inst.op {
            reads.extend([a.reg, b.reg, c.reg].into_iter().flatten());
        }
        if let Op::Setp { and, .. } = &inst.op {
            reads.extend(and.reg);
        }
        if reads.contains(&r) {
            return Some(p);
        }
        if inst.guard.is_always() && inst.op.dst_regs().contains(&r) {
            continue;
        }
        if let Op::Ld { d, n, .. } = &inst.op {
            let k = (*n).max(1) as u16;
            if inst.guard.is_always() && d.file == r.file && r.idx >= d.idx && r.idx < d.idx + k {
                continue;
            }
        }
        match &inst.op {
            Op::Bra {
                target,
                cond,
                tid0_falls,
                div,
                ..
            } => {
                // The back edge to the head starts the next iteration: stop there.
                if *target != h as usize {
                    work.push(*target);
                }
                if !inst.guard.is_always() || cond.is_some() || *tid0_falls || div.is_some() {
                    work.push(p + 1);
                }
            }
            Op::Exit if inst.guard.is_always() => {}
            _ => work.push(p + 1),
        }
    }
    None
}

/// A loop-written register that one iteration carries to the next and that differs between
/// two states at the head.
fn carried_diff(
    a: &Machine,
    x: &Machine,
    members: &[usize],
    h: u32,
    b: u32,
    written: &[Reg],
) -> Option<String> {
    let code = &a.k.code;
    for &t in members {
        for &g in written {
            if let Some(p) = loop_live(code, h, b, g) {
                if reg_val(a, t, g) != reg_val(x, t, g) {
                    return Some(format!(
                        "{g:?} of {} (read at {})",
                        a.actor_name(t as u32),
                        a.k.describe(p)
                    ));
                }
            }
        }
    }
    None
}

/// Outcome of replaying one iteration.
enum Out {
    Exit(Box<Machine>),
    Fail(Box<Machine>),
}

/// Replay the iteration that starts in `snap` (loop head of a warp-level loop, or the load of
/// a straight one) with `word` holding `bytes` and synchronization source `cell`.
fn replay(
    e: &Entry,
    snap: &Machine,
    word: Word,
    bytes: &[u8],
    cell: &Option<SyncCell>,
) -> Result<Out, String> {
    let mut m = snap.clone();
    m.fp = None;
    m.num = None;
    m.ring = None;
    set_mem_bytes(&mut m, loc_of((word.0, word.1), word.2), bytes);
    match cell {
        Some(c) => {
            m.syncs.insert(word, c.clone());
        }
        None => {
            m.syncs.remove(&word);
        }
    }
    let initial = m.clone();
    let mut changed_memory = false;
    let members = &e.members;
    if !e.warp {
        // Straight: from the load to the back branch (spin::poll_loops shape).
        let t = members[0];
        let mut p = m.threads[t].pc;
        loop {
            m.exec(&[t], p)?;
            if p == e.b {
                break;
            }
            p += 1;
        }
        return Ok(if m.threads[t].pc == e.h {
            Out::Fail(Box::new(m))
        } else {
            Out::Exit(Box::new(m))
        });
    }
    // Warp-level: run the members from the head until all are back at the head (failed) or
    // all left the loop (as spin::wpoll_dry).
    let (h, b) = (e.h, e.b);
    let mut back = vec![false; members.len()];
    let mut out = vec![false; members.len()];
    for _ in 0..100_000 {
        let run: Vec<usize> = (0..members.len())
            .filter(|&i| !back[i] && !out[i] && m.threads[members[i]].st == St::Run)
            .collect();
        let Some(pc) = run.iter().map(|&i| m.threads[members[i]].pc).min() else {
            break;
        };
        let group: Vec<usize> = run
            .iter()
            .map(|&i| members[i])
            .filter(|&t| m.threads[t].pc == pc)
            .collect();
        let stores = crate::spin::store_spans(&m, &group, pc)?;
        m.exec(&group, pc)?;
        changed_memory |= stores
            .iter()
            .any(|&(loc, n)| !crate::spin::same_bytes(&m, &initial, loc, n));
        for &t in &group {
            let i = members.iter().position(|&u| u == t).unwrap();
            let th = &m.threads[t];
            if th.st == St::Exited || th.pc < h || th.pc > b {
                out[i] = true;
            } else if pc == b && th.pc == h {
                back[i] = true;
            } else if th.st == St::Run && th.pc <= pc {
                return Err(format!(
                    "the iteration loops inside at {}",
                    m.k.describe(pc as usize)
                ));
            }
        }
    }
    if out.iter().all(|&x| x) {
        Ok(Out::Exit(Box::new(m)))
    } else if back.iter().all(|&x| x) {
        if changed_memory {
            return Err("a failed poll iteration changes memory".into());
        }
        Ok(Out::Fail(Box::new(m)))
    } else {
        Err("its lanes do not all leave the loop or all return to the head".into())
    }
}

/// Synchronization state of the members after an exit.
fn psyncs(x: &Machine, members: &[usize]) -> Vec<super::PSync> {
    members
        .iter()
        .map(|&t| (x.threads[t].vc.clone(), x.threads[t].acq_pending.clone()))
        .collect()
}

/// The counter read of an entry's exiting iteration that C-RING decides: the reads of words
/// some strong write is concurrent with. Err if there are several such words.
fn ring_read<'a>(m: &Machine, log: &SyncLog, e: &'a Entry) -> Result<Option<&'a RRead>, String> {
    let last = e.snaps.len() - 1;
    let mut found: Option<&RRead> = None;
    for r in e.reads.iter().filter(|r| r.iter == last) {
        let (key, lo) = ((r.word.0, r.word.1), r.word.2);
        let conc = log.strong.iter().any(|a| {
            a.write
                && a.key == key
                && a.lo < lo + r.n as u64
                && lo < a.hi
                && !a.ev.hb(r.t as u32, r.ev.seq, &r.vc)
        });
        if !conc {
            continue;
        }
        match found {
            Some(f) if f.word != r.word => {
                return Err(format!(
                    "the exiting iteration at {} reads two counters ({} and {})",
                    m.k.describe(e.h as usize),
                    describe_ev(m, &f.ev),
                    describe_ev(m, &r.ev)
                ));
            }
            Some(_) => {}
            None => found = Some(r),
        }
    }
    Ok(found)
}

/// In-run lowering at an exit (natural reads): the earliest write of the read's class among
/// the writes so far gives the least synchronization; continue with it.
fn lower(m: &mut Machine, log: &mut SyncLog, e: &Entry) {
    let Ok(Some(r)) = ring_read(m, log, e) else {
        return;
    };
    let i = e.members.iter().position(|&t| t == r.t).unwrap_or(0);
    if m.ring
        .as_ref()
        .is_some_and(|c| c.force.contains_key(&(r.t as u32, e.h, e.ords[i])))
    {
        return; // forced: exact
    }
    let hs = hist(m, log, r.word, r.n);
    if hs.why_not.is_some() {
        return; // refused at the end
    }
    let snap = e.snaps.last().unwrap().0.clone();
    let written = loop_written(&m.k.code, e.h, e.b);
    let mine = exit_key(m, &e.members, &written);
    let lb = coh_lb(m, log, &hs, r.word, r.t, &r.ev, &r.vc);
    let upto = hs.writes.iter().filter(|a| a.ev.seq < r.ev.seq).count();
    let mut best: Option<Vec<super::PSync>> = None;
    let mut cands: Vec<(Vec<u8>, Option<SyncCell>)> = Vec::new();
    if lb.is_none() {
        if let Some(a) = hs.writes.first() {
            cands.push((a.old.clone(), a.old_cell.clone()));
        }
    }
    for a in &hs.writes[lb.unwrap_or(0)..upto] {
        cands.push((a.val.clone(), a.cell.clone()));
    }
    for (bytes, cell) in cands {
        let Ok(Out::Exit(y)) = replay(e, &snap, r.word, &bytes, &cell) else {
            continue;
        };
        if exit_key(&y, &e.members, &written) != mine {
            continue;
        }
        best = Some(psyncs(&y, &e.members));
        break; // the earliest in sequence order acquires the least (one writer)
    }
    if let Some(best) = best {
        let cur = psyncs(m, &e.members);
        if best
            .iter()
            .zip(&cur)
            .zip(&e.members)
            .all(|((b, c), &t)| psync_le(m, t, b, c))
        {
            for ((vc, ap), &t) in best.into_iter().zip(&e.members) {
                let th = m.threads[t].make_mut();
                th.vc = vc;
                th.acq_pending = ap;
            }
        }
    }
}

// ---------------------------------------------------------------- analysis

/// One choice of a run: the read, its classes (earliest member of each, with its bytes and,
/// for the initial value, synchronization source) and the run's class.
#[derive(Clone, Debug)]
pub struct ChoiceRec {
    pub id: Choice,
    /// Lanes that read the counter in the exiting iteration, with their exit ordinals.
    pub lanes: Vec<(u32, u32)>,
    pub word: Word,
    pub n: usize,
    pub classes: Vec<(WIdx, Vec<u8>, Option<SyncCell>)>,
    pub run_class: usize,
    pub forced: bool,
    pub what: String,
}

/// C-RING's analysis of a finished run: reasons it fails, the run's choices (in run order),
/// and the counters it relied on (for the verdict).
pub(super) fn check(m: &Machine, log: &SyncLog) -> (Vec<String>, Vec<ChoiceRec>, BTreeSet<String>) {
    let mut out: Vec<String> = log.ring.fails.values().cloned().collect();
    let mut choices = Vec::new();
    let mut counters = BTreeSet::new();
    if let Some(msg) = m.ring.as_ref().and_then(|r| r.mismatch.clone()) {
        out.push(msg);
    }
    let mut seen = BTreeSet::new();
    let mut add = |out: &mut Vec<String>, s: String| {
        if out.len() < 16 && seen.insert(super::lanes_folded(&s)) {
            out.push(s);
        }
    };
    let code = &m.k.code;
    for e in &log.ring.done {
        // A straight poll on a word that is not a counter stays C-POLL's.
        let r = match ring_read(m, log, e) {
            Ok(Some(r)) => r,
            Ok(None) => continue, // nothing concurrent: determined
            Err(why) => {
                add(&mut out, format!("C-RING: {why}"));
                continue;
            }
        };
        let hs = hist(m, log, r.word, r.n);
        if let Some(why) = &hs.why_not {
            if e.warp {
                add(
                    &mut out,
                    format!(
                        "C-RING: the poll at {} reads a word that is not a monotone counter: {why}",
                        describe_ev(m, &r.ev)
                    ),
                );
            }
            continue;
        }
        let written = loop_written(code, e.h, e.b);
        // Failed real iterations are invisible: consecutive iteration starts agree on what
        // the loop carries.
        for i in 1..e.snaps.len() {
            if let Some(g) = carried_diff(
                &e.snaps[i - 1].0,
                &e.snaps[i].0,
                &e.members,
                e.h,
                e.b,
                &written,
            ) {
                add(
                    &mut out,
                    format!(
                        "C-RING: a failed iteration of the poll loop at {} changes {g}, which a later iteration reads",
                        m.k.describe(e.h as usize)
                    ),
                );
            }
        }
        // Other words read in the loop must read the same bytes in every real iteration.
        for x in &e.reads {
            if x.word == r.word {
                continue;
            }
            if let Some(y) = e
                .reads
                .iter()
                .find(|y| y.word == x.word && y.t == x.t && y.pc == x.pc && y.bytes != x.bytes)
            {
                add(
                    &mut out,
                    format!(
                        "C-RING: the poll loop at {} also reads {}, which changes between its iterations",
                        m.k.describe(e.h as usize),
                        describe_ev(m, &y.ev)
                    ),
                );
            }
        }
        // Candidates: from the latest write that happens-before the loop's first read of
        // the counter on.
        let first = e
            .reads
            .iter()
            .filter(|x| x.word == r.word)
            .min_by_key(|x| x.ev.seq)
            .unwrap();
        let lb = coh_lb(m, log, &hs, r.word, first.t, &first.ev, &first.vc);
        type Cand = (WIdx, Vec<u8>, Option<SyncCell>, Option<Ev>);
        let mut cands: Vec<Cand> = Vec::new();
        match lb {
            None => {
                let (bytes, cell) = match hs.writes.first() {
                    Some(a) => (a.old.clone(), a.old_cell.clone()),
                    None => (r.bytes.clone(), None),
                };
                cands.push((None, bytes, cell, None));
                for (j, a) in hs.writes.iter().enumerate() {
                    cands.push((Some(j), a.val.clone(), a.cell.clone(), Some(a.ev)));
                }
            }
            Some(l) => {
                for (j, a) in hs.writes.iter().enumerate().skip(l) {
                    cands.push((Some(j), a.val.clone(), a.cell.clone(), Some(a.ev)));
                }
            }
        }
        // The write the run's exiting read read.
        let forced = m.ring.as_ref().and_then(|c| {
            c.force
                .get(&(
                    r.t as u32,
                    e.h,
                    e.ords[e.members.iter().position(|&t| t == r.t).unwrap_or(0)],
                ))
                .cloned()
        });
        let read_idx: WIdx = match &forced {
            Some(f) => f.idx,
            None => {
                let k = hs.writes.iter().filter(|a| a.ev.seq < r.ev.seq).count();
                k.checked_sub(1)
            }
        };
        let snap = &e.snaps.last().unwrap().0;
        let mut keys: Vec<Option<Key>> = Vec::new();
        let mut states: Vec<Option<Vec<super::PSync>>> = Vec::new();
        let mut bad = false;
        // States a failed iteration may leave at the head (checked below).
        let mut fails: Vec<(String, Box<Machine>)> = Vec::new();
        for (idx, bytes, cell, ev) in &cands {
            let what = match ev {
                Some(ev) => describe_ev(m, ev),
                None => "the value before its first write".into(),
            };
            match replay(e, snap, r.word, bytes, cell) {
                Ok(Out::Exit(y)) => {
                    keys.push(Some(exit_key(&y, &e.members, &written)));
                    states.push(Some(psyncs(&y, &e.members)));
                }
                Ok(Out::Fail(y)) => {
                    if let Some(g) = carried_diff(snap, &y, &e.members, e.h, e.b, &written) {
                        add(
                            &mut out,
                            format!(
                                "C-RING: the poll at {} reading {what} fails its iteration but changes {g}, which the next iteration reads",
                                describe_ev(m, &r.ev)
                            ),
                        );
                        bad = true;
                    }
                    fails.push((what.clone(), y));
                    keys.push(None);
                    states.push(None);
                }
                Err(why) => {
                    add(
                        &mut out,
                        format!(
                            "C-RING: replaying the poll at {} on {what}: {why}",
                            describe_ev(m, &r.ev)
                        ),
                    );
                    bad = true;
                    keys.push(None);
                    states.push(None);
                }
            }
            for earlier in &e.snaps[..e.snaps.len() - 1] {
                if let Err(why) = replay(e, &earlier.0, r.word, bytes, cell) {
                    add(
                        &mut out,
                        format!("C-RING: replaying an earlier poll iteration on {what}: {why}"),
                    );
                    bad = true;
                }
            }
            let _ = idx;
        }
        if bad {
            continue;
        }
        // No lost wakeup: exits are upward closed in the counter's sequence.
        if let Some(i) = keys.iter().position(|k| k.is_some()) {
            if let Some(j) = (i..keys.len()).find(|&j| keys[j].is_none()) {
                let name = |c: &(WIdx, Vec<u8>, Option<SyncCell>, Option<Ev>)| match &c.3 {
                    Some(ev) => format!("{} (value {:?})", describe_ev(m, ev), c.1),
                    None => format!("the value before its first write ({:?})", c.1),
                };
                add(
                    &mut out,
                    format!(
                        "C-RING: the wait at {} exits on {} but not on the later {}: it may miss its wakeup",
                        describe_ev(m, &r.ev),
                        name(&cands[i]),
                        name(&cands[j])
                    ),
                );
                continue;
            }
        }
        // Classes by exit state, each with its earliest member.
        let mut classes: Vec<(Key, usize)> = Vec::new();
        for (i, k) in keys.iter().enumerate() {
            if let Some(k) = k {
                if !classes.iter().any(|(c, _)| c == k) {
                    classes.push((k.clone(), i));
                }
            }
        }
        // Failed iterations leave no trace in what the exit leaves: from every state a failed
        // iteration may leave at the head (the run's earlier real iterations, and each failing
        // candidate's), each class's exit leaves the same state as from the snapshot.
        let mut starts: Vec<(String, &Machine)> = e.snaps[..e.snaps.len() - 1]
            .iter()
            .map(|x| ("an earlier iteration of the run".to_string(), &*x.0))
            .collect();
        starts.extend(
            fails
                .iter()
                .map(|(w, y)| (format!("a failed iteration on {w}"), &**y)),
        );
        'starts: for (what, x) in &starts {
            for (key, ci) in &classes {
                let c = &cands[*ci];
                let got = replay(e, x, r.word, &c.1, &c.2);
                let same = match &got {
                    Ok(Out::Exit(y)) => exit_key(y, &e.members, &written) == *key,
                    _ => false,
                };
                if !same && std::env::var("PTXV_RING_DEBUG").is_ok() {
                    match &got {
                        Ok(Out::Exit(y)) => {
                            eprintln!(
                                "C-RING debug: from {what}: {:?}\n  vs snapshot: {:?}",
                                exit_key(y, &e.members, &written),
                                key
                            );
                            for (pc, regs) in key {
                                for (g, _) in regs {
                                    if let Some(p) = live_at(code, *pc as usize, *g) {
                                        eprintln!("  {g:?} read at {}", m.k.describe(p));
                                    }
                                }
                            }
                        }
                        Ok(Out::Fail(_)) => eprintln!("C-RING debug: from {what}: fails"),
                        Err(w) => eprintln!("C-RING debug: from {what}: {w}"),
                    }
                }
                if !same {
                    add(
                        &mut out,
                        format!(
                            "C-RING: after {what}, the poll at {} leaves its loop differently on the same value (the loop carries state from one iteration to the next)",
                            describe_ev(m, &r.ev)
                        ),
                    );
                    bad = true;
                    break 'starts;
                }
            }
        }
        if bad {
            continue;
        }
        let Some(ri) = cands.iter().position(|c| c.0 == read_idx) else {
            add(
                &mut out,
                format!(
                    "C-RING: the poll at {} read a write older than the latest one that happens-before it",
                    describe_ev(m, &r.ev)
                ),
            );
            continue;
        };
        let Some(run_key) = keys[ri].clone() else {
            add(
                &mut out,
                format!(
                    "C-RING: replaying the poll at {} on the value it read does not exit",
                    describe_ev(m, &r.ev)
                ),
            );
            continue;
        };
        let rc = classes.iter().position(|(c, _)| *c == run_key).unwrap();
        // The run continued with the earliest member's synchronization (lowered at the exit,
        // or forced); every member of the class acquires at least that.
        let base = states[classes[rc].1].clone().unwrap();
        for (i, k) in keys.iter().enumerate() {
            if k.as_ref() != Some(&run_key) {
                continue;
            }
            let s = states[i].as_ref().unwrap();
            if !base
                .iter()
                .zip(s)
                .zip(&e.members)
                .all(|((a, b), &t)| psync_le(m, t, a, b))
            {
                add(
                    &mut out,
                    format!(
                        "C-RING: the poll at {} may read a later write that synchronizes with less (not checked)",
                        describe_ev(m, &r.ev)
                    ),
                );
            }
        }
        counters.insert(format!(
            "{} ({} write(s))",
            describe_counter(m, &hs),
            hs.writes.len()
        ));
        let lanes: Vec<(u32, u32)> = e
            .reads
            .iter()
            .filter(|x| x.word == r.word && x.iter == e.snaps.len() - 1)
            .map(|x| {
                let i = e.members.iter().position(|&t| t == x.t).unwrap_or(0);
                (x.t as u32, e.ords[i])
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let i0 = e.members.iter().position(|&t| t == r.t).unwrap_or(0);
        choices.push(ChoiceRec {
            id: (r.t as u32, e.h, e.ords[i0]),
            lanes,
            word: r.word,
            n: r.n,
            classes: classes
                .iter()
                .map(|(_, i)| (cands[*i].0, cands[*i].1.clone(), cands[*i].2.clone()))
                .collect(),
            run_class: rc,
            forced: forced.is_some(),
            what: describe_ev(m, &r.ev),
        });
    }
    // The run order of choices is the order of their exits (entries close in run order).
    (out, choices, counters)
}

fn describe_counter(m: &Machine, hs: &Hist) -> String {
    match hs.writes.first() {
        Some(a) => format!("counter written by {}", describe_ev(m, &a.ev)),
        None => "counter".into(),
    }
}

/// Global words read by C-RING polls (for a later launch group's obligations).
pub(super) fn global_reads(log: &SyncLog) -> Vec<(u32, u64, u64, Ev)> {
    let mut v = Vec::new();
    for e in &log.ring.done {
        for r in &e.reads {
            if r.word.0 == 1 {
                v.push((r.word.1, r.word.2, r.word.2 + r.n as u64, r.ev));
            }
        }
    }
    for e in log.ring.open.values().chain(log.ring.straight.values()) {
        for r in &e.reads {
            if r.word.0 == 1 {
                v.push((r.word.1, r.word.2, r.word.2 + r.n as u64, r.ev));
            }
        }
    }
    v
}

/// Is a stuck forced run infeasible? Some thread waits for a forced write that has not
/// happened while its poll would exit on the counter's current value: then no execution
/// makes that choice (the thread reads a written value eventually, E-SPIN fairness).
pub(super) fn stuck_infeasible(m: &Machine) -> Option<String> {
    let r = m.ring.as_ref()?;
    for (&(t, h, k), f) in &r.force {
        let t = t as usize;
        let th = &m.threads[t];
        if th.st == St::Exited || r.ord.get(&(t as u32, h)).copied().unwrap_or(0) != k {
            continue;
        }
        let Some(j) = f.idx else { continue };
        if r.hist.get(&f.word).is_some_and(|v| v.len() > j) {
            continue;
        }
        // Waiting at the loop: does an iteration on the current value exit?
        let pc = th.pc;
        if m.ring_head(pc) != Some(h) && pc != h {
            continue;
        }
        let mut x = m.clone();
        x.ring = None;
        x.fp = None;
        x.num = None;
        let warp = th.warp;
        let members: Vec<usize> = x
            .warp_threads(warp)
            .filter(|&u| x.threads[u].pc == pc && x.threads[u].st == St::Run)
            .collect();
        if members.is_empty() {
            continue;
        }
        let exits = if let Some(&b) = x.wpolls.get(&pc) {
            crate::spin::wpoll_dry(&x, &members, pc, b).is_none()
        } else {
            x.split_enabled(&members, pc)
        };
        if exits {
            return Some(format!(
                "{} waits for write {} of a counter that never comes, but would leave the loop on its current value",
                m.actor_name(t as u32),
                j
            ));
        }
    }
    None
}

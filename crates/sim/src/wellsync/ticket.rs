/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! C-TICKET (ASSUMPTIONS.md E-WELLSYNC): atomic counters whose results steer the program
//! (a persistent kernel's tile scheduler: `t = atomicInc(&next, limit)`, work on tile `t`,
//! claim again; an arrival counter whose last arriver resets it).
//!
//! C-ORDER refuses two unordered atomics whose results are used: which thread gets which
//! value depends on the schedule. For a ticket word the only freedom is the order of its
//! writes (its coherence order), so the certificate enumerates those orders instead:
//!
//! * **Ticket words.** Some atomic's result is used, and every strong access of the word in
//!   the run is a write of the same bytes by one lane of a thread (one lane per instruction
//!   step): a relaxed read-modify-write (not acquiring, not a compare-and-swap, which C-CAS
//!   decides) or a strong store. No load reads the word (a poll is C-POLL's); plain accesses
//!   are races for the race detector. Given the order of its writes, every value an atomic
//!   returns is determined.
//! * **Forcing.** A run forces a prefix of each word's writer sequence (`RingCtl::tick_force`):
//!   a lane at a gated write waits until the writes before its forced position are done
//!   (`Machine::ticket_ready`). Later writes are free (the deterministic schedule). Waiting
//!   only delays a write, so a forced run that ends is an execution. A run that gets stuck
//!   with a lane waiting for a forced writer that never writes is no execution and is dropped
//!   (that writer cannot write before the others).
//! * **Branches**: dynamic partial-order reduction over the writes of the word. Everything else
//!   that conflicts is ordered by happens-before (the certificate's other conditions, checked
//!   on every run), so the writes of a ticket word are the only dependent events left
//!   unordered. For each write c_i past the forced prefix and each other thread u, let c_j be
//!   u's first later write: if c_j happens-after none of c_i .. c_{j-1}, a run forces the
//!   prefix up to c_i with u in place of c_i's writer. This is a superset of the backtracking
//!   points of Flanagan-Godefroid DPOR (which reverses only the last such pair); a write
//!   that happens-after one of them is reached by reversing an earlier pair. Each forced
//!   prefix runs once.
//! * **Order-created edges.** A claim followed by an acquire fence of its thread (a last-block
//!   reduction: `atom.add` ticket, then `fence.acq_rel` before reading the partials;
//!   `SAcc::facq`) synchronizes with the claims before it in the coherence order, so the
//!   run's happens-before does have edges that this order created. The enumeration stays
//!   complete: the happens-before it consults is a subset of Flanagan-Godefroid's (program
//!   order plus every dependent pair in trace order), since each memory-model edge comes from
//!   a pair of conflicting accesses whose order in the trace it follows (an order-created one:
//!   a claim reading the value an earlier claim of the word left), and fewer edges only add
//!   backtracking points. Precondition: every forced run computes its own happens-before
//!   from its own execution, and its branches and its certificate use only that run's clocks;
//!   no run reuses another run's (a base run's) clocks or log. (C-POLL-ATOM did: until
//!   2026-11-07 it reconstructed a poll's base from the adds of the one run.) An acquiring
//!   `atom` (by its own .sem) is still refused below.
//! * **Equivalence** is checked semantically: every run must certify on its own (every other
//!   condition as usual) and all runs must leave the same outputs. A kernel whose results
//!   depend on which CTA processed which ticket is refused.
//!
//! Refusals carry the prefix `C-TICKET:`.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use ptxv_ir::*;

use super::SAcc;
use super::SyncLog;
use super::describe_ev;
use super::ring::Word;
use crate::machine::*;

/// A forced claimant prefix per ticket word.
pub type TickForce = BTreeMap<Word, Vec<u32>>;

/// What one logged run found about ticket words.
#[derive(Default, Debug)]
pub struct TickOut {
    /// Ticket words claimed by at least two threads: (word, the pcs of its atomics).
    pub words: BTreeMap<Word, BTreeSet<u32>>,
    /// Words that look like ticket counters but are not (why, with the `C-TICKET:` prefix).
    pub bad: Vec<String>,
    /// Forced prefixes to run next (dynamic partial-order reduction).
    pub children: Vec<TickForce>,
    /// Claims per word, in order: claimant threads.
    pub seq: BTreeMap<Word, Vec<u32>>,
}

fn word_of(a: &SAcc) -> Word {
    (a.key.0, a.key.1, a.lo)
}

/// An atomic whose result may make a word a ticket word (compare-and-swap is C-CAS's).
fn is_ticket_op(op: AtomOp) -> bool {
    !matches!(op, AtomOp::Cas | AtomOp::CasSpin)
}

/// The ticket words of a run (`words`, `bad`) and, when `force` is the run's forcing, the
/// branches (`children`). `claims`: the claims the machine counted (`RingCtl::tick_claims`),
/// checked against the log.
pub(super) fn analyze(
    m: &Machine,
    log: &SyncLog,
    force: &TickForce,
    claims: &BTreeMap<Word, Vec<u32>>,
) -> TickOut {
    let mut out = TickOut::default();
    // Candidate words: global or shared words written by an atomic Inc / Add with a used
    // result, by at least two threads.
    let nthreads = m.threads.len() as u32;
    let mut cand: BTreeMap<Word, (u64, Vec<&SAcc>)> = BTreeMap::new();
    for a in &log.strong {
        if let Some((op, used, _, _)) = a.rmw {
            // Declared C-TICKET-ADDR counters are perm.rs's.
            let perm =
                a.key.0 == 1 && super::PERM_OK.with(|p| p.borrow().contains(&(a.key.1 as usize)));
            if a.write && used && is_ticket_op(op) && a.ev.actor < nthreads && !perm {
                cand.entry(word_of(a)).or_insert((a.hi, Vec::new()));
            }
        }
    }
    if cand.is_empty() {
        return out;
    }
    // Every strong access overlapping a candidate word.
    for a in &log.strong {
        for (w, (hi, v)) in cand.iter_mut() {
            if a.key == (w.0, w.1) && a.lo < *hi && w.2 < a.hi {
                v.push(a);
            }
        }
    }
    'words: for (w, (_, accs)) in &cand {
        let writes: Vec<&SAcc> = accs.iter().copied().filter(|a| a.write).collect();
        let threads: BTreeSet<u32> = writes.iter().map(|a| a.ev.actor).collect();
        if threads.len() < 2 {
            continue;
        }
        let first = writes[0];
        let what = |m: &Machine, a: &SAcc| describe_ev(m, &a.ev);
        let gated = super::TICKET_OK.with(|t| t.borrow().contains(w));
        for a in accs {
            // A read-modify-write is logged as its read and its write.
            let ok = a.ev.actor < nthreads
                && (a.lo, a.hi) == (first.lo, first.hi)
                && match a.rmw {
                    Some((op, _, acq, _)) => !acq && !matches!(op, AtomOp::Cas | AtomOp::CasSpin),
                    None => a.write,
                };
            if !ok {
                // Not a ticket word (C-ORDER decides it), unless this search gates it.
                if gated {
                    out.bad.push(format!(
                        "C-TICKET: the counter at {} {} byte {:#x} (claimed at {}) is also accessed by {}, which is not a thread's relaxed atomic or strong store of the same bytes",
                        if w.0 == 0 { "shared memory of CTA" } else { "global allocation" },
                        w.1,
                        w.2,
                        what(m, first),
                        what(m, a)
                    ));
                }
                continue 'words;
            }
        }
        // One lane per step.
        let mut steps = BTreeSet::new();
        for a in &writes {
            if let Some(s) = a.step {
                if !steps.insert(s) {
                    out.bad.push(format!(
                        "C-TICKET: several lanes of one instruction claim the counter at {} (lane order)",
                        what(m, a)
                    ));
                    continue 'words;
                }
            }
        }
        out.words
            .insert(*w, writes.iter().map(|a| a.ev.pc).collect());
        out.seq
            .insert(*w, writes.iter().map(|a| a.ev.actor).collect());
        // The machine's count of claims must be the log's (when this run gated the word).
        if let Some(c) = claims.get(w) {
            if *c != out.seq[w] {
                out.bad.push(format!(
                    "C-TICKET: the claims counted at the atomics of {} differ from the logged ones",
                    what(m, first)
                ));
                continue 'words;
            }
        }
        // Branches past the forced prefix.
        let p = force.get(w).map_or(0, |v| v.len());
        for i in p..writes.len() {
            let ci = writes[i];
            let mut seen = BTreeSet::new();
            for j in i + 1..writes.len() {
                let cj = writes[j];
                let u = cj.ev.actor;
                if u == ci.ev.actor || !seen.insert(u) {
                    continue;
                }
                // u's first claim after position i: free of happens-before from the claims
                // between?
                let after = (i..j).any(|k| writes[k].ev.hb(cj.ev.actor, cj.ev.seq, &cj.vc));
                if after {
                    continue;
                }
                let mut f = force.clone();
                let mut pre: Vec<u32> = writes[..i].iter().map(|a| a.ev.actor).collect();
                pre.push(u);
                f.insert(*w, pre);
                out.children.push(f);
            }
        }
    }
    out
}

impl Machine {
    /// C-TICKET: the word thread `t`'s memory instruction at `pc` accesses.
    fn ticket_word(&self, t: usize, pc: u32) -> Option<Word> {
        let (space, bytes, addr) = match &self.k.code[pc as usize].op {
            Op::Atom {
                space, bytes, addr, ..
            }
            | Op::St {
                space, bytes, addr, ..
            } => (*space, *bytes, addr),
            _ => return None,
        };
        let a = self.addr(t, addr).ok()?;
        match self.resolve(t, space, a, bytes as usize).ok()? {
            crate::exec::Loc::Shared { cta, off } => Some((0, cta as u32, off as u64)),
            crate::exec::Loc::Global { alloc, off } => Some((1, alloc as u32, off)),
            crate::exec::Loc::Param { .. } => None,
        }
    }

    /// C-TICKET: may `members` run the gated write at `pc`? Not while an active lane writing
    /// a ticket word is not the forced writer of that word's next write.
    pub fn ticket_ready(&self, members: &[usize], pc: u32) -> bool {
        let Some(r) = self.ring.as_ref() else {
            return true;
        };
        let inst = &self.k.code[pc as usize];
        members
            .iter()
            .filter(|&&t| self.prd(t, inst.guard))
            .all(|&t| {
                let Some(w) = self.ticket_word(t, pc) else {
                    return true;
                };
                let k = r.tick_claims.get(&w).map_or(0, |v| v.len());
                r.tick_force
                    .get(&w)
                    .and_then(|v| v.get(k))
                    .is_none_or(|&f| f == t as u32)
            })
    }

    /// C-TICKET: count the writes of ticket words by `members` at the gated `pc` (before it
    /// runs), in order.
    pub fn ticket_count(&mut self, members: &[usize], pc: u32) {
        let Some(r) = self.ring.as_ref() else {
            return;
        };
        let inst = &self.k.code[pc as usize];
        let lanes: Vec<(Word, u32)> = members
            .iter()
            .filter(|&&t| self.prd(t, inst.guard))
            .filter_map(|&t| {
                self.ticket_word(t, pc)
                    .filter(|w| r.tick_words.contains(w))
                    .map(|w| (w, t as u32))
            })
            .collect();
        if lanes.is_empty() {
            return;
        }
        let r = std::sync::Arc::make_mut(self.ring.as_mut().unwrap());
        for (w, t) in lanes {
            r.tick_claims.entry(w).or_default().push(t);
        }
    }
}

/// C-TICKET: the run is stuck with a lane waiting at a ticket atomic for a forced claimant
/// that never claims (no execution makes this claim order).
pub(super) fn stuck_infeasible(m: &Machine) -> Option<String> {
    let r = m.ring.as_ref()?;
    if r.tick_pcs.is_empty() {
        return None;
    }
    for (t, th) in m.threads.iter().enumerate() {
        if th.st != St::Run || !r.tick_pcs.contains(&th.pc) {
            continue;
        }
        let Some(w) = m.ticket_word(t, th.pc) else {
            continue;
        };
        let k = r.tick_claims.get(&w).map_or(0, |v| v.len());
        if let Some(&f) = r.tick_force.get(&w).and_then(|v| v.get(k)) {
            if f != t as u32 {
                return Some(format!(
                    "C-TICKET: {} waits at {} for claim {k} by {}, which never comes",
                    m.actor_name(t as u32),
                    m.k.describe(th.pc as usize),
                    m.actor_name(f)
                ));
            }
        }
    }
    None
}

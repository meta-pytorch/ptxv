/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! C-ORDER pair search (ASSUMPTIONS.md E-WELLSYNC): every pair of overlapping strong
//! accesses of two actors, at least one a write, must be ordered by happens-before or exempt
//! (CAS-chain links, lanes of one SASS atomic step, commutative atomics with unused results,
//! fp32 atomic adds left to C-FADD).
//!
//! The pairs are not enumerated. The accesses are swept in run order; each 4-byte word keeps,
//! per actor, that actor's earlier reads and writes of the word in program order. For an
//! access y and an actor A, A's accesses that happen-before y form a prefix of A's list
//! (program order is in happens-before, which is transitive; in the clock representation the
//! needed clock `clock + amb` never decreases along one actor's events), so a walk from the
//! end of the list stops at the first access ordered before y: only the unordered pairs are
//! visited, plus one ordered access per (word, actor). A read y needs only A's writes; a write
//! y needs both lists. A pair that overlaps in several words is decided at the first one. An actor whose needed clocks do decrease (not expected) falls back to every
//! pair of that word for that actor, so the result is the pairwise one in every case.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;

use ptxv_ir::AtomOp;

use super::CasWord;
use super::SAcc;

/// How a pair of strong accesses (x before y in the run) stands under C-ORDER.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Pair {
    /// Ordered, not conflicting, or exempt.
    Fine,
    /// fp32 atomic adds of one word left to C-FADD: (key, lo, hi).
    Fadd((u8, u32), u64, u64),
    /// Not ordered, not exempt.
    Conflict,
}

/// Words (key, byte / 4) where plain writes of equal bytes need no order: None when the run
/// has C-RING choices or forced reads; otherwise the words no poll reads.
pub(super) type SameOk<'a> = Option<&'a HashSet<((u8, u32), u64)>>;

/// The C-ORDER decision for x (earlier in the run) and y, which overlap.
pub(super) fn pair(x: &SAcc, y: &SAcc, cas_ok: &HashSet<CasWord>, same_ok: SameOk) -> Pair {
    if !(x.write || y.write) || x.ev.actor == y.ev.actor {
        return Pair::Fine;
    }
    if x.ev.hb(y.ev.actor, y.ev.seq, &y.vc) {
        return Pair::Fine;
    }
    // Two links of a CAS chain that C-CAS determined.
    if x.at.is_some()
        && y.at.is_some()
        && (x.lo, x.hi) == (y.lo, y.hi)
        && cas_ok.contains(&(x.key, x.lo, x.hi))
    {
        return Pair::Fine;
    }
    // Lanes of one SASS atomic step: performed in lane order (E-LOCKSTEP).
    if x.step.is_some() && x.step == y.step {
        return Pair::Fine;
    }
    // Two plain writes of the same bytes to the overlap (Triton's all-reduce result stored
    // by every warp to one shared address with `STS.128`): either order leaves those bytes.
    // The values are the run's in every schedule, by induction along happens-before: a write's
    // value comes from reads before it, each ordered with every write it conflicts with (or
    // a poll's exit value, C-POLL; a CAS chain's, C-CAS), so it reads the run's value. A read
    // unordered with one of the writes is its own pair; a read after both already has both
    // in its clock, whichever it reads. Not on words a poll reads (it may exit on either write
    // and synchronize with a different writer), and not with C-RING choices or forced reads,
    // where the run's values are one choice among several (fixtures/e_ring_warp W1).
    if same_ok.is_some_and(|polled| {
        x.write
            && y.write
            && x.rmw.is_none()
            && y.rmw.is_none()
            && same_bytes(x, y)
            && (x.lo.max(y.lo) / 4..=(x.hi.min(y.hi) - 1) / 4)
                .all(|w| !polled.contains(&(x.key, w)))
    }) {
        return Pair::Fine;
    }
    // C-TICKET: writes of a ticket word (atomics and strong stores), whose orders the
    // certificate enumerates (ticket.rs).
    if (x.lo, x.hi) == (y.lo, y.hi)
        && (x.write || x.rmw.is_some())
        && (y.write || y.rmw.is_some())
        && super::TICKET_OK.with(|t| t.borrow().contains(&(x.key.0, x.key.1, x.lo)))
    {
        return Pair::Fine;
    }
    // C-POLL-ATOM coherence: a strong store by a thread after its exiting poll of a counter of
    // adds is coherence-after every add the exit needed (the read saw the full sum, written by
    // the coherence-last of them; the store is program-order-after the read: CoRW), so the
    // value left is the store's in every execution.
    if (x.lo, x.hi) == (y.lo, y.hi)
        && x.rmw.is_some()
        && y.write
        && y.rmw.is_none()
        && super::poll_atom_before(x, y)
    {
        return Pair::Fine;
    }
    // C-TICKET-ADDR: relaxed fetch-adds of a declared counter word, whose results only place
    // rows (perm.rs checks it after the run, or the certificate refuses). A claim observes no
    // synchronization in the certificate's runs (exec.rs), so no fence after it acquires
    // through it: a fence-acquired-read test on these pairs would never fire and is not needed.
    if (x.lo, x.hi) == (y.lo, y.hi)
        && x.key.0 == 1
        && matches!(x.rmw, Some((AtomOp::Add, _, false, _)))
        && matches!(y.rmw, Some((AtomOp::Add, _, false, _)))
        && super::PERM_OK.with(|p| p.borrow().contains(&(x.key.1 as usize)))
    {
        return Pair::Fine;
    }
    // Commutative atomics with unused results that do not acquire, by themselves or through
    // a later acquire fence of their thread (`SAcc::facq`): either makes the order decide
    // which releases the thread synchronizes with (until 2026-11-08 `red.relaxed` followed by
    // `fence.acq_rel` certified with the edge of the run's order only).
    if let (Some((ox, ux, qx, kx)), Some((oy, uy, qy, ky))) = (x.rmw, y.rmw) {
        let (qx, qy) = (qx || x.facq, qy || y.facq);
        let comm = !matches!(
            ox,
            AtomOp::Exch
                | AtomOp::FAddFtz
                | AtomOp::FAdd64
                | AtomOp::AddBf16x2
                | AtomOp::AddBf16
                | AtomOp::CasSpin
                | AtomOp::Cas
        );
        if (ox == oy || kx && ky) && comm && !ux && !uy && !qx && !qy {
            return Pair::Fine;
        }
        // fp32 adds of one word: their order may change the bits (C-FADD).
        if ox == AtomOp::FAddFtz
            && oy == AtomOp::FAddFtz
            && !ux
            && !uy
            && !qx
            && !qy
            && x.fadd.is_some()
            && y.fadd.is_some()
            && (x.lo, x.hi) == (y.lo, y.hi)
        {
            return Pair::Fadd(x.key, x.lo, x.hi);
        }
    }
    Pair::Conflict
}

/// Two writes store the same bytes where they overlap.
fn same_bytes(x: &SAcc, y: &SAcc) -> bool {
    let (lo, hi) = (x.lo.max(y.lo), x.hi.min(y.hi));
    let at = |a: &SAcc, b: u64| a.val.get((b - a.lo) as usize).copied();
    lo < hi && (lo..hi).all(|b| at(x, b).is_some() && at(x, b) == at(y, b))
}

/// What the search found: conflicting pairs (indices into the access list, earlier first,
/// in the order found), the C-FADD words, and whether it stopped at `max_unordered`.
pub(super) struct Found {
    pub conflicts: Vec<(usize, usize)>,
    pub fwords: BTreeMap<((u8, u32), u64, u64), ()>,
    pub capped: bool,
}

/// The access's needed clock: an event of another actor is HB-after it iff that actor's
/// clock entry is at least this (`Ev::hb`).
#[inline]
fn need(a: &SAcc) -> u64 {
    a.ev.clock as u64 + a.ev.amb as u64
}

#[derive(Default)]
struct PerActor {
    reads: Vec<u32>,
    writes: Vec<u32>,
    /// The needed clocks were not monotone: check every earlier access of this actor.
    flat: bool,
}

/// Every C-ORDER conflict among `acc` (any order; pairs are formed by run position), stopping
/// after `stop_after` conflicts or `max_unordered` unordered pairs visited.
pub(super) fn search(
    acc: &[SAcc],
    cas_ok: &HashSet<CasWord>,
    stop_after: usize,
    max_unordered: u64,
    same_ok: SameOk,
) -> Found {
    let mut idx: Vec<usize> = (0..acc.len()).collect();
    idx.sort_by_key(|&i| acc[i].ev.seq);
    let mut words: HashMap<((u8, u32), u64), HashMap<u32, PerActor>> = HashMap::new();
    let mut out = Found {
        conflicts: Vec::new(),
        fwords: BTreeMap::new(),
        capped: false,
    };
    let mut unordered = 0u64;
    for &yi in &idx {
        let y = &acc[yi];
        if y.hi <= y.lo {
            continue;
        }
        let (w0, w1) = (y.lo / 4, (y.hi - 1) / 4);
        for w in w0..=w1 {
            let per = words.entry((y.key, w)).or_default();
            for (&actor, pa) in per.iter() {
                if actor == y.ev.actor {
                    continue;
                }
                let lists: [&Vec<u32>; 2] = [&pa.writes, &pa.reads];
                for (li, list) in lists.iter().enumerate() {
                    if li == 1 && !y.write {
                        break;
                    }
                    for &xi in list.iter().rev() {
                        let x = &acc[xi as usize];
                        if x.ev.hb(y.ev.actor, y.ev.seq, &y.vc) {
                            if pa.flat {
                                continue;
                            }
                            break;
                        }
                        if x.hi <= y.lo || y.hi <= x.lo {
                            continue;
                        }
                        // A pair that shares several words is decided at the first of them.
                        if x.lo.max(y.lo) / 4 != w {
                            continue;
                        }
                        unordered += 1;
                        if unordered > max_unordered {
                            out.capped = true;
                            return out;
                        }
                        match pair(x, y, cas_ok, same_ok) {
                            Pair::Fine => {}
                            Pair::Fadd(k, lo, hi) => {
                                out.fwords.insert((k, lo, hi), ());
                            }
                            Pair::Conflict => {
                                out.conflicts.push((xi as usize, yi));
                                if out.conflicts.len() >= stop_after {
                                    return out;
                                }
                            }
                        }
                    }
                }
            }
            let pa = per.entry(y.ev.actor).or_default();
            let list = if y.write {
                &mut pa.writes
            } else {
                &mut pa.reads
            };
            let down = list
                .last()
                .is_some_and(|&l| need(&acc[l as usize]) > need(y));
            list.push(yi as u32);
            pa.flat |= down;
        }
    }
    out
}

/// The pairwise reference: every overlapping pair (tests).
#[cfg(test)]
pub(super) fn search_pairwise(acc: &[SAcc], cas_ok: &HashSet<CasWord>, same_ok: SameOk) -> Found {
    let mut out = Found {
        conflicts: Vec::new(),
        fwords: BTreeMap::new(),
        capped: false,
    };
    for (i, a) in acc.iter().enumerate() {
        for (j, b) in acc.iter().enumerate() {
            if a.key != b.key || a.ev.seq >= b.ev.seq || a.hi <= b.lo || b.hi <= a.lo {
                continue;
            }
            match pair(a, b, cas_ok, same_ok) {
                Pair::Fine => {}
                Pair::Fadd(k, lo, hi) => {
                    out.fwords.insert((k, lo, hi), ());
                }
                Pair::Conflict => out.conflicts.push((i, j)),
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::Ev;
    use super::*;
    use crate::machine::VC;
    use crate::machine::vc_join;
    use crate::machine::vc_tick;

    /// A random run of `actors` actors over a few words: each event is a read, a write, a
    /// commutative atomic or an fp32 add, and some events synchronize (the actor joins
    /// another's clock, as an acquire of a release would).
    fn run(seed: u64, actors: u32, events: usize) -> Vec<SAcc> {
        let mut s = seed | 1;
        let mut rnd = move |n: u64| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s % n
        };
        let mut vcs: Vec<VC> = (0..actors)
            .map(|a| {
                let mut v = VC::new();
                vc_tick(&mut v, a);
                v
            })
            .collect();
        let mut amb = vec![false; actors as usize];
        let mut out = Vec::new();
        for seq in 0..events as u64 {
            let a = rnd(actors as u64) as u32;
            match rnd(8) {
                0 => {
                    // acquire: join another actor's clock
                    let b = rnd(actors as u64) as usize;
                    let o = vcs[b].clone();
                    vc_join(&mut vcs[a as usize], &o);
                    vc_tick(&mut vcs[a as usize], a);
                    amb[a as usize] = false;
                    continue;
                }
                1 => {
                    vc_tick(&mut vcs[a as usize], a);
                    amb[a as usize] = false;
                    continue;
                }
                _ => {}
            }
            let kind = rnd(5);
            let word = rnd(3);
            let (lo, hi) = match rnd(3) {
                0 => (word * 4, word * 4 + 4),
                1 => (word * 4 + rnd(4), word * 4 + 4),
                _ => (word * 4, word * 4 + 8),
            };
            let vc = vcs[a as usize].clone();
            let clock = crate::machine::vc_get(&vc, a);
            let write = kind != 0;
            let rmw = match kind {
                2 => Some((AtomOp::Add, false, false, false)),
                3 => Some((AtomOp::FAddFtz, false, false, false)),
                _ => None,
            };
            out.push(SAcc {
                key: (0, 0),
                lo,
                hi,
                ev: Ev {
                    actor: a,
                    clock,
                    amb: amb[a as usize],
                    pc: 0,
                    seq,
                },
                vc,
                write,
                rmw,
                fadd: (kind == 3).then_some(0x3f80_0000),
                // Two byte values, so equal-value writes occur.
                val: vec![(seq % 2) as u8; (hi - lo) as usize],
                old: vec![],
                old_cell: None,
                cell: None,
                prev_w: vec![],
                at: None,
                step: None,
                facq: false,
            });
            if write && rmw.is_none() {
                amb[a as usize] = true;
            }
        }
        out
    }

    #[test]
    fn sweep_finds_exactly_the_pairwise_conflicts() {
        for seed in 1..400u64 {
            for actors in [2u32, 3, 5] {
                let acc = run(seed * 7919 + actors as u64, actors, 60);
                let none = HashSet::new();
                let polled: HashSet<((u8, u32), u64)> = [((0, 0), 1)].into();
                let same = (seed % 2 == 0).then_some(&polled);
                let a = search(&acc, &none, usize::MAX, u64::MAX, same);
                let b = search_pairwise(&acc, &none, same);
                let mut x = a.conflicts.clone();
                let mut y = b.conflicts.clone();
                x.sort();
                y.sort();
                assert_eq!(x, y, "seed {seed} actors {actors}");
                assert_eq!(
                    a.fwords.keys().collect::<Vec<_>>(),
                    b.fwords.keys().collect::<Vec<_>>(),
                    "seed {seed} actors {actors}"
                );
            }
        }
    }
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! C-TICKET-ADDR (ASSUMPTIONS.md E-WELLSYNC): slot assignment by fetch-add. A counter word
//! hands each claimant `old = atomicAdd(&cnt[e], c)` and the claimant stores its row at an
//! address computed from `old` (a MoE dispatch: the token's row within its expert's segment).
//! The order of the adds decides which claimant gets which row, so the outputs differ between
//! schedules by a permutation of rows within each counter's segment, and C-ORDER refuses the
//! adds (their results are used). The case names the counter buffer and the row buffers
//! (`[[permuted]]`); the declaration is a hint, every condition below is checked, and any
//! failure refuses the certificate with the reason.
//!
//! On the certificate's run, C-ORDER leaves relaxed, non-acquiring adds of a declared counter
//! word unordered (order.rs), and a claim synchronizes with nothing: an acquire fence after it
//! does not acquire from the claimant whose add it read from (exec.rs; that edge exists in some
//! orders only, and a run certified without it holds in every order, which only adds edges);
//! then:
//!
//! 1. **Counter.** Every strong access to the word in the run is such an add of one nonzero
//!    constant `c`; the word's writes before the first add (race shadow) happen-before every
//!    add; the values the adds returned are `v0, v0 + c, ...` (`V_w`). Every execution makes
//!    the same claims (2: no claim's result reaches control flow, guards or synchronization,
//!    and every other value is the run's by the certificate), so in any coherence order the
//!    claims receive a bijection onto `V_w`.
//! 2. **Taint.** For each claim, a rerun with its result register undefined (S-INIT taint:
//!    undefined bits propagate through register-only operations). The only uses allowed are
//!    address operands of the claimant thread's own stores into the declared row buffers;
//!    anything else (a branch, guard, predicate, load or atomic address, stored data, a
//!    synchronization operand, another thread's instruction, an operation S-INIT does not
//!    propagate through) refuses. The rerun must end as the run did.
//! 3. **Replays.** For each claim and each `t` in `V_w`, a rerun with its result forced to
//!    `t`, recording the claimant's stores into the row buffers after the claim: `A_i(t)`.
//!    Every claim of a word must give the same `A(t)` (the addresses depend on the ticket
//!    alone), the sets must be disjoint across tickets and across words (no two claims share
//!    a byte whatever the assignment), and the segment of a word is the union over `V_w`.
//! 4. **Order.** Every other access to a segment byte in the run happens-before every claim
//!    store: no access after them or unordered with them (the rows are outputs only).
//!
//! Then any execution has the run's events, happens-before and values except the claim stores'
//! addresses, and leaves the run's final memory up to a permutation of the per-ticket rows
//! within each segment: the verdict class `clean-up-to-permutation` (never `clean`).

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use ptxv_ir::*;

use super::SAcc;
use super::SyncLog;
use crate::exec::Loc;
use crate::machine::*;

/// Byte ranges (allocation, first byte, end).
pub type Ranges = Vec<(usize, u64, u64)>;

/// Per counter word (allocation, byte): its claims (`claims`).
pub type Words = BTreeMap<(usize, u64), Word>;

/// One claim: (thread, pc, counter word (allocation, byte), occurrence of that triple).
pub type ClaimId = (u32, u32, (usize, u64), u32);

/// What a C-TICKET-ADDR rerun does at its claim.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    /// Record accesses to the row buffers (condition 4).
    #[default]
    Observe,
    /// The claim's result register is undefined (condition 2).
    Undef,
    /// The claim returns this value (condition 3).
    Force(u64),
}

/// An access to a row buffer in an observing run.
#[derive(Clone, Debug)]
pub struct RowAcc {
    pub actor: u32,
    pub pc: u32,
    pub write: bool,
    pub alloc: usize,
    pub lo: u64,
    pub hi: u64,
    pub vc: VC,
}

/// State of a C-TICKET-ADDR rerun (`Machine::perm`).
#[derive(Clone, Debug, Default)]
pub struct PermRun {
    pub mode: Mode,
    pub claim: Option<ClaimId>,
    /// Row buffers (allocations).
    pub rows: BTreeSet<usize>,
    /// Occurrences of the claim's (thread, pc, word) so far.
    pub seen: u32,
    /// The claim has executed.
    pub after: bool,
    /// Undef: the allowed uses (thread, pc): store addresses into row buffers.
    pub uses: BTreeSet<(u32, u32)>,
    /// Undef: forbidden uses (descriptions).
    pub bad: Vec<String>,
    /// The claimant's stores into row buffers after the claim (allocation, first byte, end).
    pub stores: Vec<(usize, u64, u64)>,
    /// Observe: every access to a row buffer.
    pub acc: Vec<RowAcc>,
}

impl Machine {
    /// The counter word an atomic of thread `t` at `pc` accesses (before it executes).
    pub(crate) fn perm_word(&self, t: usize, op: &Op) -> Option<(usize, u64)> {
        let Op::Atom {
            space, bytes, addr, ..
        } = op
        else {
            return None;
        };
        let a = self.addr(t, addr).ok()?;
        match self.resolve(t, *space, a, *bytes as usize).ok()? {
            Loc::Global { alloc, off } => Some((alloc, off)),
            _ => None,
        }
    }

    /// Hook (exec.rs `exec_thread`, after an atomic of thread `t` at `pc` on `word`): the
    /// rerun's claim gets an undefined or a forced result.
    pub(crate) fn perm_claim(&mut self, t: usize, pc: u32, word: (usize, u64), op: &Op) {
        let Op::Atom {
            d: Some(d), bytes, ..
        } = op
        else {
            return;
        };
        let Some(p) = self.perm.as_mut() else {
            return;
        };
        let Some((ct, cpc, cw, occ)) = p.claim else {
            return;
        };
        if (t as u32, pc, word) != (ct, cpc, cw) {
            return;
        }
        let k = p.seen;
        p.seen += 1;
        if k != occ {
            return;
        }
        p.after = true;
        match p.mode.clone() {
            Mode::Undef => self.make_undefined(t, *d),
            Mode::Force(v) => {
                let slots = (*bytes / 4).max(1);
                self.wr(t, *d, v, slots);
            }
            Mode::Observe => {}
        }
    }

    /// Is an undefined register `regs` of thread `t` at `pc` an allowed use (condition 2):
    /// address operands of the claimant's store into a row buffer?
    pub(crate) fn perm_allowed(&self, t: usize, pc: u32, regs: &[Reg]) -> bool {
        let Some(p) = self.perm.as_ref() else {
            return false;
        };
        if p.claim.is_none_or(|c| c.0 != t as u32) {
            return false;
        }
        let Op::St {
            space, bytes, addr, ..
        } = &self.k.code[pc as usize].op
        else {
            return false;
        };
        let areg: Vec<Reg> = [addr.base.as_ref(), addr.base2.as_ref()]
            .into_iter()
            .flatten()
            .flat_map(|v| match v {
                Val::Reg(r, n) if r.file != File::V => (0..(*n).max(1) as u16)
                    .map(|i| Reg {
                        file: r.file,
                        idx: r.idx + i,
                    })
                    .collect::<Vec<_>>(),
                Val::Reg(r, _) => vec![*r],
                _ => vec![],
            })
            .collect();
        if !regs.iter().all(|r| areg.contains(r)) {
            return false;
        }
        let Ok(a) = self.addr(t, addr) else {
            return false;
        };
        matches!(
            self.resolve(t, *space, a, *bytes as usize),
            Ok(Loc::Global { alloc, .. }) if p.rows.contains(&alloc)
        )
    }

    /// Hook (exec.rs, before an instruction, Undef reruns): a use of undefined bits by an
    /// operation that is not register-only (or by a guard of one) must be an allowed use.
    pub(crate) fn perm_taint_pre(&mut self, members: &[usize], pc: u32, reg_only: bool) {
        let inst = self.k.code[pc as usize].clone();
        let mut bad = Vec::new();
        let mut ok = Vec::new();
        for &t in members {
            if let Some(r) = self.pred_undef(t, inst.guard) {
                if !reg_only {
                    bad.push(format!(
                        "{} guard {:?}{} at {}",
                        self.actor_name(t as u32),
                        r.file,
                        r.idx,
                        self.k.describe(pc as usize)
                    ));
                }
                continue;
            }
            if reg_only || !self.prd(t, inst.guard) {
                continue;
            }
            let regs: Vec<Reg> = crate::exec::src_regs_at(&self.k, &inst, pc)
                .iter()
                .copied()
                .filter(|r| self.reg_umask(t, *r) != 0)
                .collect();
            if regs.is_empty() {
                continue;
            }
            if self.perm_allowed(t, pc, &regs) {
                ok.push((t as u32, pc));
            } else {
                bad.push(format!(
                    "{} uses it at {}",
                    self.actor_name(t as u32),
                    self.k.describe(pc as usize)
                ));
            }
        }
        if let Some(p) = self.perm.as_mut() {
            p.uses.extend(ok);
            p.bad.extend(bad);
        }
    }

    /// Hook (exec.rs `report_uninit`, Undef reruns): an S-INIT report is a use; allowed only
    /// as a row store's address.
    pub(crate) fn perm_report(&mut self, t: usize, r: Reg, pc: u32) {
        let allowed = self.perm_allowed(t, pc, &[r]);
        let what = format!(
            "{} uses it at {}",
            self.actor_name(t as u32),
            self.k.describe(pc as usize)
        );
        if let Some(p) = self.perm.as_mut() {
            if allowed {
                p.uses.insert((t as u32, pc));
            } else {
                p.bad.push(what);
            }
        }
    }

    /// Hook (exec.rs `access_x`): the claimant's row stores after the claim; every row access
    /// when observing.
    pub(crate) fn perm_access(
        &mut self,
        actor: u32,
        vc: &VC,
        pc: u32,
        loc: Loc,
        n: usize,
        write: bool,
    ) {
        let Some(p) = self.perm.as_mut() else {
            return;
        };
        let Loc::Global { alloc, off } = loc else {
            return;
        };
        if !p.rows.contains(&alloc) {
            return;
        }
        if write && p.after && p.claim.is_some_and(|c| c.0 == actor) {
            p.stores.push((alloc, off, off + n as u64));
        }
        if p.mode == Mode::Observe {
            p.acc.push(RowAcc {
                actor,
                pc,
                write,
                alloc,
                lo: off,
                hi: off + n as u64,
                vc: vc.clone(),
            });
        }
    }
}

/// The claims of one counter word in the run.
#[derive(Clone, Debug, Default)]
pub struct Word {
    pub claims: Vec<ClaimId>,
    /// `V_w`: the values handed out, in increasing order of k (v0 + k c).
    pub values: Vec<u64>,
    pub bytes: u8,
}

/// The run's claims on declared counters (condition 1), or why the word fails it.
pub(super) fn claims(m: &Machine, log: &SyncLog) -> Result<Words, String> {
    let counters: BTreeSet<usize> = m.launch.perm.iter().map(|p| p.0).collect();
    let mut by: BTreeMap<(usize, u64), Vec<&SAcc>> = BTreeMap::new();
    for a in &log.strong {
        if a.key.0 == 1 && counters.contains(&(a.key.1 as usize)) {
            by.entry((a.key.1 as usize, a.lo)).or_default().push(a);
        }
    }
    let what = |a: &SAcc| super::describe_ev(m, &a.ev);
    let mut out = BTreeMap::new();
    for (w, accs) in by {
        // Words whose atomics' results are all unused are C-ORDER's (commutative atomics).
        if !accs
            .iter()
            .any(|a| a.write && matches!(a.rmw, Some((_, true, _, _))))
        {
            continue;
        }
        let first = accs[0];
        let mut words = Word::default();
        let mut c: Option<u64> = None;
        let mut olds = Vec::new();
        let mut occ: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        for a in &accs {
            let ok = (a.lo, a.hi) == (first.lo, first.hi)
                && matches!(a.rmw, Some((AtomOp::Add, _, false, _)))
                && (a.ev.actor as usize) < m.threads.len();
            if !ok {
                return Err(format!(
                    "C-TICKET-ADDR: the counter at global allocation {} byte {:#x} is accessed by {}, which is not a thread's relaxed fetch-add of the same bytes",
                    w.0,
                    w.1,
                    what(a)
                ));
            }
            if !a.write {
                continue;
            }
            let n = (a.hi - a.lo) as usize;
            let le = |b: &[u8]| b.iter().rev().fold(0u64, |x, &y| x << 8 | y as u64);
            let (old, new) = (le(&a.old), le(&a.val));
            let mask = if n >= 8 {
                u64::MAX
            } else {
                (1u64 << (8 * n)) - 1
            };
            let d = new.wrapping_sub(old) & mask;
            if d == 0 || c.is_some_and(|c| c != d) {
                return Err(format!(
                    "C-TICKET-ADDR: the adds of the counter at global allocation {} byte {:#x} do not all add one nonzero constant ({})",
                    w.0,
                    w.1,
                    what(a)
                ));
            }
            c = Some(d);
            // The word's writes before the first add happen-before every add.
            for e in &first.prev_w {
                if !e.hb(&a.vc) && e.actor != a.ev.actor {
                    return Err(format!(
                        "C-TICKET-ADDR: the counter at global allocation {} byte {:#x} is initialized by an access that does not happen-before the add by {}",
                        w.0,
                        w.1,
                        what(a)
                    ));
                }
            }
            let k = occ.entry((a.ev.actor, a.ev.pc)).or_insert(0);
            words.claims.push((a.ev.actor, a.ev.pc, w, *k));
            *k += 1;
            olds.push(old);
            words.bytes = n as u8;
        }
        // V_w: v0, v0 + c, ... in the run's coherence order.
        let c = c.unwrap_or(1);
        let n = (first.hi - first.lo) as usize;
        let mask = if n >= 8 {
            u64::MAX
        } else {
            (1u64 << (8 * n)) - 1
        };
        for (k, &o) in olds.iter().enumerate() {
            if o != olds[0].wrapping_add(c.wrapping_mul(k as u64)) & mask {
                return Err(format!(
                    "C-TICKET-ADDR: the counter at global allocation {} byte {:#x} returned {o:#x} to its claim {k}, not v0 + k * c",
                    w.0, w.1
                ));
            }
        }
        words.values = olds;
        out.insert(w, words);
    }
    Ok(out)
}

/// A rerun of `m0` in `mode` at `claim`: (ended as a terminal state, the rerun's state).
fn rerun(
    m0: &Machine,
    rows: &BTreeSet<usize>,
    mode: Mode,
    claim: Option<ClaimId>,
) -> (bool, PermRun) {
    let mut m = m0.clone();
    m.perm = Some(Box::new(PermRun {
        mode,
        claim,
        rows: rows.clone(),
        ..Default::default()
    }));
    let (mut m, _, ok) = crate::explore::run_once(m);
    (ok, *m.perm.take().unwrap_or_default())
}

/// What C-TICKET-ADDR established: per counter word, its segment (sorted byte ranges).
pub struct PermOut {
    /// Per counter word: (claims, segment).
    pub words: BTreeMap<(usize, u64), (usize, Ranges)>,
    /// Per word (same order), per ticket: the row byte ranges.
    pub tickets: Vec<Vec<Ranges>>,
    pub runs: usize,
}

/// Conditions 2-4 for the run's claims (`claims`) from the start machine `m0`, whose run ended
/// as a terminal state.
pub(super) fn check(m0: &Machine, words: &Words) -> Result<PermOut, String> {
    let rows: BTreeSet<usize> = m0
        .launch
        .perm
        .iter()
        .filter(|p| words.keys().any(|w| w.0 == p.0))
        .flat_map(|p| p.1.iter().copied())
        .collect();
    let max_runs: usize = std::env::var("PTXV_PERM_MAX_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
    let total: usize = words
        .values()
        .map(|w| w.claims.len() * (1 + w.values.len()))
        .sum();
    if total > max_runs {
        return Err(format!(
            "C-TICKET-ADDR: {total} reruns needed, more than {max_runs} (PTXV_PERM_MAX_RUNS)"
        ));
    }
    let name = |t: u32| m0.actor_name(t);
    let desc = |c: &ClaimId| {
        format!(
            "the claim by {} at {} (global allocation {} byte {:#x}, occurrence {})",
            name(c.0),
            m0.k.describe(c.1 as usize),
            c.2.0,
            c.2.1,
            c.3
        )
    };
    let mut runs = 0usize;
    // Condition 2: taint.
    let mut uses: BTreeSet<(u32, u32)> = BTreeSet::new();
    for w in words.values() {
        for c in &w.claims {
            let (ok, p) = rerun(m0, &rows, Mode::Undef, Some(*c));
            runs += 1;
            if !p.after {
                return Err(format!(
                    "C-TICKET-ADDR: {} did not recur in the rerun",
                    desc(c)
                ));
            }
            if let Some(b) = p.bad.first() {
                return Err(format!(
                    "C-TICKET-ADDR: the result of {} reaches more than its own row stores' addresses: {b}",
                    desc(c)
                ));
            }
            if !ok {
                return Err(format!(
                    "C-TICKET-ADDR: the rerun with the result of {} undefined does not end",
                    desc(c)
                ));
            }
            uses.extend(p.uses);
        }
    }
    // Condition 3: replays over V_w.
    let mut seg: BTreeMap<(usize, u64), (usize, Ranges)> = BTreeMap::new();
    let mut owner: BTreeMap<(usize, u64), ((usize, u64), u64)> = BTreeMap::new();
    let mut tickets = Vec::new();
    for (wk, w) in words {
        let mut at: BTreeMap<u64, Vec<(usize, u64, u64)>> = BTreeMap::new();
        for c in &w.claims {
            for &v in &w.values {
                let (ok, p) = rerun(m0, &rows, Mode::Force(v), Some(*c));
                runs += 1;
                if !ok || !p.after {
                    return Err(format!(
                        "C-TICKET-ADDR: the rerun with {} returning {v:#x} does not end",
                        desc(c)
                    ));
                }
                let mut s = p.stores;
                s.sort();
                s.dedup();
                match at.get(&v) {
                    Some(x) if *x != s => {
                        return Err(format!(
                            "C-TICKET-ADDR: with the value {v:#x}, {} stores its row elsewhere than another claim of the same counter (the row depends on more than the ticket)",
                            desc(c)
                        ));
                    }
                    Some(_) => {}
                    None => {
                        at.insert(v, s);
                    }
                }
            }
        }
        let mut all = Vec::new();
        for (&v, s) in &at {
            for &(a, lo, hi) in s {
                for b in lo..hi {
                    if let Some(&(ow, ov)) = owner.get(&(a, b)) {
                        return Err(format!(
                            "C-TICKET-ADDR: global allocation {a} byte {b:#x} is a row of ticket {ov:#x} of the counter at allocation {} byte {:#x} and of ticket {v:#x} of the counter at allocation {} byte {:#x}",
                            ow.0, ow.1, wk.0, wk.1
                        ));
                    }
                    owner.insert((a, b), (*wk, v));
                }
                all.push((a, lo, hi));
            }
        }
        all.sort();
        seg.insert(*wk, (w.claims.len(), all));
        tickets.push(at.into_values().collect());
    }
    // Condition 4: every other access to a segment byte happens-before every claim store.
    let (_, obs) = rerun(m0, &rows, Mode::Observe, None);
    runs += 1;
    let in_seg = |a: &RowAcc| (a.lo..a.hi).any(|b| owner.contains_key(&(a.alloc, b)));
    let is_claim = |a: &RowAcc| a.write && uses.contains(&(a.actor, a.pc));
    let claim_st: Vec<(usize, &RowAcc)> = obs
        .acc
        .iter()
        .enumerate()
        .filter(|(_, a)| is_claim(a))
        .collect();
    for (i, a) in obs
        .acc
        .iter()
        .enumerate()
        .filter(|(_, a)| in_seg(a) && !is_claim(a))
    {
        // Happens-before as the race detector's epochs (`Epoch::hb`); one actor: run order.
        let clock = vc_get(&a.vc, a.actor);
        for &(j, s) in &claim_st {
            let before = if s.actor == a.actor {
                i < j
            } else {
                clock <= vc_get(&s.vc, a.actor)
            };
            if !before {
                return Err(format!(
                    "C-TICKET-ADDR: the {} of global allocation {} byte {:#x} by {} at {} does not happen-before the row store by {} at {} (the rows must be outputs only)",
                    if a.write { "write" } else { "read" },
                    a.alloc,
                    a.lo,
                    name(a.actor),
                    m0.k.describe(a.pc as usize),
                    name(s.actor),
                    m0.k.describe(s.pc as usize)
                ));
            }
        }
    }
    Ok(PermOut {
        words: seg,
        tickets,
        runs,
    })
}

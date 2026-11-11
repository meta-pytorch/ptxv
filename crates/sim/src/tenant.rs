/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The TMEM co-tenant. ASSUME(E-TMEM-COTENANT, E-TMEM-PERMIT): see ASSUMPTIONS.md.
//!
//! `tcgen05.alloc` takes columns of the SM's 512-column tensor memory, which other CTAs on
//! the same SM (of this kernel, of a PDL dependent, of another stream) share. The default
//! model gives each CTA a whole SM. Co-tenant mode (`Env::tmem_cotenant`, case
//! `tmem_cotenant`, `ptxv check --tmem-cotenant`) adds an adversary that holds any columns
//! for an arbitrary (finite) time:
//! - *Placement.* An allocation may land at any free, naturally aligned column run (the
//!   adversary held the runs below it while the CTA allocated). Explored as a choice of the
//!   allocating split. A kernel that assumes column 0 instead of the returned address reads
//!   or writes columns it does not own.
//! - *Waiting.* An allocation that does not fit blocks (E-ALLOC) until the adversary releases;
//!   the adversary always does, so waiting alone never deadlocks. It does deadlock when the
//!   co-tenant is a CTA that cannot release before this CTA progresses: CTA A holds columns
//!   (or, E-TMEM-PERMIT, its allocation permit: allocated and not yet relinquished) while it
//!   waits on CTA B, and B, placed on A's SM, cannot allocate. Pair placements
//!   (`Env::tmem_share`) put two CTAs on one SM: their allocations share the columns and, with
//!   the permit model, one cannot allocate while the other holds its permit. Explored.
//! - *Happens-before analysis* (`hb_findings`, any size): one recorded run gives, per CTA,
//!   its allocations, frees, relinquish and exit with their vector clocks. For CTAs A and B
//!   of different clusters on one GPU: B's allocation β can come after A's allocation α
//!   (not β →hb α), and A cannot release before β (β →hb A's release point: the free of α's
//!   columns, or the relinquish / exit for the permit). Then placing A and B on one SM
//!   deadlocks: when α comes first, β blocks and A waits for β forever. Columns: only when
//!   the two allocations exceed 512 columns. With a certificate (E-WELLSYNC) the run's
//!   happens-before is every schedule's, so this is exact; otherwise it describes the run.
//! - *Missing relinquish.* A CTA that allocated and exits without
//!   `tcgen05.relinquish_alloc_permit` keeps every co-tenant of its SM from allocating until
//!   it exits (E-TMEM-PERMIT): reported as a `contract` finding.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use ptxv_ir::Op;

use crate::explore::Action;
use crate::explore::Verdict;
use crate::machine::*;

/// `m` in co-tenant mode with the given SM-sharing pairs.
pub fn with_tenant(mut m: Machine, share: Vec<(u16, u16)>) -> Machine {
    let mut l = (*m.launch).clone();
    l.env.tmem_cotenant = true;
    l.env.tmem_share = share;
    m.launch = std::sync::Arc::new(l);
    m
}

/// Free, naturally aligned runs of `units` 32-column units in `busy`.
pub fn free_runs(busy: u16, units: u32) -> Vec<u32> {
    if units == 0 || units > 16 {
        return vec![];
    }
    let mask = ((1u32 << units) - 1) as u16;
    (0..16u32)
        .step_by(units as usize)
        .filter(|&i| busy & (mask << i) == 0)
        .collect()
}

impl Machine {
    /// The CTA sharing `cta`'s SM (`Env::tmem_share`).
    pub fn tmem_partner(&self, cta: u16) -> Option<u16> {
        self.launch.env.tmem_share.iter().find_map(|&(a, b)| {
            if a == cta {
                Some(b)
            } else if b == cta {
                Some(a)
            } else {
                None
            }
        })
    }

    /// Columns of `cta`'s SM that are allocated: its own and its SM partner's.
    pub fn tmem_busy(&self, cta: u16) -> u16 {
        let own = self.tmem[cta as usize].alloc;
        match self.tmem_partner(cta) {
            Some(p) => own | self.tmem[p as usize].alloc,
            None => own,
        }
    }

    fn cta_finished(&self, cta: u16) -> bool {
        self.threads
            .iter()
            .filter(|t| t.cta == cta)
            .all(|t| t.st == St::Exited)
    }

    /// ASSUME(E-TMEM-PERMIT): `cta` cannot allocate while its SM partner holds its
    /// allocation permit (allocated, not relinquished, not exited).
    pub fn tmem_permit_blocked(&self, cta: u16) -> bool {
        if !self.launch.env.tmem_permit {
            return false;
        }
        let Some(p) = self.tmem_partner(cta) else {
            return false;
        };
        let tp = &self.tmem[p as usize];
        tp.permit && !tp.relinquished && !self.cta_finished(p)
    }

    /// (CTAs, units) of the allocation the split at `pc` performs, if it is one and runs.
    fn alloc_at(&self, members: &[usize], pc: u32) -> Option<(Vec<u16>, u32)> {
        let inst = &self.k.code[pc as usize];
        let t = members[0];
        match &inst.op {
            Op::TmemAlloc { ncols, .. } => {
                let n = self.rd(t, ncols).ok()? as u32;
                (n >= 32 && n.is_power_of_two() && n <= 512)
                    .then(|| (vec![self.threads[t].cta], n / 32))
            }
            Op::TmemAllocRaw { units, pair, .. } => {
                if !members.iter().any(|&m| self.prd(m, inst.guard)) {
                    return None;
                }
                let n = self.rd(t, units).ok()? as u32;
                let ctas = self.alloc_group(self.threads[t].cta, *pair).ok()?;
                (n >= 1 && n.is_power_of_two() && n <= 16).then_some((ctas, n))
            }
            _ => None,
        }
    }

    /// ASSUME(E-TMEM-COTENANT): the column units an enabled allocation may land at (None:
    /// not co-tenant mode, not an allocation, or a single place).
    pub fn tenant_alloc_choices(&self, members: &[usize], pc: u32) -> Option<Vec<u32>> {
        if !self.launch.env.tmem_cotenant {
            return None;
        }
        let (ctas, units) = self.alloc_at(members, pc)?;
        let busy = ctas.iter().fold(0u16, |b, &c| b | self.tmem_busy(c));
        let mut v = free_runs(busy, units);
        // The lowest and the highest free run (`PTXV_TMEM_PLACE=all`: every one). Placement
        // changes addresses only; the two ends catch code that assumes column 0.
        if v.len() > 2 && std::env::var("PTXV_TMEM_PLACE").as_deref() != Ok("all") {
            v = vec![v[0], *v.last().unwrap()];
        }
        (v.len() > 1).then_some(v)
    }

    /// The column unit chosen for the allocation being executed (the action's choice, see
    /// `tenant_alloc_choices`), if it is free for every CTA of the group.
    pub fn tenant_place(&mut self, ctas: &[u16], units: u32) -> Option<u32> {
        if !self.launch.env.tmem_cotenant {
            return None;
        }
        let p = self.elect_choice.take()? as u32;
        let busy = ctas.iter().fold(0u16, |b, &c| b | self.tmem_busy(c));
        free_runs(busy, units).contains(&p).then_some(p)
    }
}

// ------------------------------------------------------------------ recorded run

#[derive(Clone, Debug)]
enum Kind {
    Alloc(u16),
    Free(u16),
    Relinquish,
}

#[derive(Clone, Debug)]
struct Ev {
    cta: u16,
    pc: u32,
    kind: Kind,
    thread: usize,
    /// The thread's clock before the instruction.
    before: VC,
    /// The thread's own clock right after it (the recording run ticks it there, so that
    /// whatever the thread does later carries at least this clock).
    at: u32,
}

impl Ev {
    /// Does `vc` know this event (FastTrack epoch test)?
    fn known_by(&self, vc: &VC) -> bool {
        vc_get(vc, self.thread as u32) >= self.at
    }
}

fn is_tmem_op(op: &Op) -> bool {
    matches!(
        op,
        Op::TmemAlloc { .. }
            | Op::TmemAllocRaw { .. }
            | Op::TmemDealloc { .. }
            | Op::TmemFreeRaw { .. }
            | Op::TmemRelinquish
    )
}

/// One deterministic run (`run_once`'s policy) recording every TMEM allocator event.
fn record(m0: &Machine) -> (Vec<Ev>, Machine, bool) {
    let mut m = m0.clone();
    m.fp = None;
    let mut evs = Vec::new();
    loop {
        let acts = m.actions();
        let pick = acts
            .iter()
            .find(|(a, _)| matches!(a, Action::Split { .. }))
            .or(acts.first());
        let Some((a, _)) = pick.cloned() else { break };
        let snap = match &a {
            Action::Split { warp, pc, .. } if is_tmem_op(&m.k.code[*pc as usize].op) => {
                let inst = &m.k.code[*pc as usize];
                let members = m
                    .splits()
                    .into_iter()
                    .find(|(w, p, _)| w == warp && p == pc)
                    .map(|(_, _, v)| v)
                    .unwrap_or_default();
                let t = members
                    .iter()
                    .copied()
                    .find(|&t| m.prd(t, inst.guard))
                    .or(members.first().copied());
                t.map(|t| {
                    let st: Vec<(u16, bool)> = (0..m.launch.ctas() as usize)
                        .map(|c| (m.tmem[c].alloc, m.tmem[c].relinquished))
                        .collect();
                    (*pc, t, m.threads[t].vc.clone(), st)
                })
            }
            _ => None,
        };
        m.apply(&a);
        m.findings.clear();
        if let Some((pc, t, before, st)) = snap {
            vc_tick(&mut m.threads[t].make_mut().vc, t as u32);
            let at = vc_get(&m.threads[t].vc, t as u32);
            for (c, &(a0, r0)) in st.iter().enumerate() {
                let (a1, r1) = (m.tmem[c].alloc, m.tmem[c].relinquished);
                let mut push = |kind| {
                    evs.push(Ev {
                        cta: c as u16,
                        pc,
                        kind,
                        thread: t,
                        before: before.clone(),
                        at,
                    })
                };
                if a1 & !a0 != 0 {
                    push(Kind::Alloc(a1 & !a0));
                }
                if a0 & !a1 != 0 {
                    push(Kind::Free(a0 & !a1));
                }
                if r1 && !r0 {
                    push(Kind::Relinquish);
                }
            }
        }
    }
    let ok = m.terminal();
    (evs, m, ok)
}

/// What a CTA did with tensor memory in the recorded run.
struct CtaTmem {
    allocs: Vec<Ev>,
    /// Release point of each allocation's columns (the clock before the free, else exit).
    col_release: Vec<VC>,
    /// Release point of the permit (before the first relinquish, else exit).
    permit_release: VC,
    relinquished: bool,
}

/// The happens-before analysis and the missing-relinquish check (module doc), on one
/// recorded run of `m0`. Returns findings and notes.
pub fn hb_findings(m0: &Machine) -> (Vec<Finding>, Vec<String>) {
    let (evs, fin, ok) = record(m0);
    let mut notes = Vec::new();
    if !ok {
        notes.push("TMEM co-tenant analysis: the recorded run did not terminate".into());
    }
    let ncta = fin.launch.ctas() as usize;
    let exit_vc: Vec<VC> = (0..ncta)
        .map(|c| {
            let mut v = VC::new();
            for t in fin.threads.iter().filter(|t| t.cta as usize == c) {
                vc_join(&mut v, &t.vc);
            }
            v
        })
        .collect();
    let mut per: BTreeMap<u16, CtaTmem> = BTreeMap::new();
    for c in 0..ncta as u16 {
        let mine: Vec<&Ev> = evs.iter().filter(|e| e.cta == c).collect();
        let allocs: Vec<Ev> = mine
            .iter()
            .filter(|e| matches!(e.kind, Kind::Alloc(_)))
            .map(|e| (*e).clone())
            .collect();
        if allocs.is_empty() {
            continue;
        }
        let col_release = mine
            .iter()
            .enumerate()
            .filter_map(|(i, e)| match e.kind {
                Kind::Alloc(m) => Some(
                    mine[i + 1..]
                        .iter()
                        .find(|f| matches!(f.kind, Kind::Free(x) if x & m != 0))
                        .map_or(exit_vc[c as usize].clone(), |f| f.before.clone()),
                ),
                _ => None,
            })
            .collect();
        let rel = mine.iter().find(|e| matches!(e.kind, Kind::Relinquish));
        per.insert(
            c,
            CtaTmem {
                allocs,
                col_release,
                permit_release: rel.map_or(exit_vc[c as usize].clone(), |e| e.before.clone()),
                relinquished: rel.is_some(),
            },
        );
    }
    let mut out: Vec<Finding> = Vec::new();
    let mut seen = BTreeSet::new();
    let loc = |pc: u32| fin.k.describe(pc as usize);
    // Missing relinquish.
    for (&c, t) in &per {
        if !t.relinquished {
            let pc = t.allocs[0].pc;
            let sig = format!("tmem-relinquish:{pc}");
            if seen.insert(sig.clone()) {
                out.push(Finding {
                    kind: "contract".into(),
                    sig,
                    msg: format!(
                        "TMEM co-tenant (E-TMEM-PERMIT): {} allocates at {} and exits without tcgen05.relinquish_alloc_permit: no other CTA on its SM (of this kernel, a PDL dependent or another stream) can allocate tensor memory until it exits",
                        fin.cta_name(c),
                        loc(pc)
                    ),
                    trace: vec![],
                    ..Default::default()
                });
            }
        }
    }
    // Pairs that may share an SM: different clusters, same GPU.
    for (&a, ta) in &per {
        for (&b, tb) in &per {
            if a == b
                || fin.launch.cluster_of(a as u32).0 == fin.launch.cluster_of(b as u32).0
                || fin.launch.g(a as u32).device != fin.launch.g(b as u32).device
            {
                continue;
            }
            for beta in &tb.allocs {
                let Kind::Alloc(bm) = beta.kind else {
                    continue;
                };
                // Permit: A holds it from its first allocation to its relinquish / exit.
                let a1 = &ta.allocs[0];
                if m0.launch.env.tmem_permit
                    && !beta.known_by(&a1.before)
                    && beta.known_by(&ta.permit_release)
                {
                    let sig = format!("tmem-cotenant:permit:{}:{}", a1.pc, beta.pc);
                    if seen.insert(sig.clone()) {
                        out.push(Finding {
                            kind: "deadlock".into(),
                            sig,
                            msg: format!(
                                "TMEM co-tenant deadlock (E-TMEM-COTENANT, E-TMEM-PERMIT): if {} and {} share an SM, {} holds its allocation permit from {} until it {}, which happens after {}'s allocation at {}; that allocation blocks while the permit is held",
                                fin.cta_name(a),
                                fin.cta_name(b),
                                fin.cta_name(a),
                                loc(a1.pc),
                                if ta.relinquished { "relinquishes" } else { "exits (no relinquish)" },
                                fin.cta_name(b),
                                loc(beta.pc)
                            ),
                            trace: vec![],
                            ..Default::default()
                        });
                    }
                }
                // Columns: A's allocation and B's do not fit together.
                for (i, alpha) in ta.allocs.iter().enumerate() {
                    let Kind::Alloc(am) = alpha.kind else {
                        continue;
                    };
                    if am.count_ones() + bm.count_ones() <= 16 {
                        continue;
                    }
                    if !beta.known_by(&alpha.before) && beta.known_by(&ta.col_release[i]) {
                        let sig = format!("tmem-cotenant:columns:{}:{}", alpha.pc, beta.pc);
                        if seen.insert(sig.clone()) {
                            out.push(Finding {
                                kind: "deadlock".into(),
                                sig,
                                msg: format!(
                                    "TMEM co-tenant deadlock (E-TMEM-COTENANT): if {} and {} share an SM, {} holds {} columns from {} until after {}'s allocation of {} columns at {}, and the two do not fit in 512",
                                    fin.cta_name(a),
                                    fin.cta_name(b),
                                    fin.cta_name(a),
                                    am.count_ones() * 32,
                                    loc(alpha.pc),
                                    fin.cta_name(b),
                                    bm.count_ones() * 32,
                                    loc(beta.pc)
                                ),
                                trace: vec![],
                                ..Default::default()
                            });
                        }
                    }
                }
            }
        }
    }
    let users: Vec<String> = per
        .iter()
        .map(|(&c, t)| {
            let cols: u32 = t
                .allocs
                .iter()
                .map(|e| match e.kind {
                    Kind::Alloc(m) => m.count_ones() * 32,
                    _ => 0,
                })
                .max()
                .unwrap_or(0);
            format!(
                "{} {cols} col{}",
                fin.cta_name(c),
                if t.relinquished {
                    ""
                } else {
                    " (no relinquish)"
                }
            )
        })
        .take(8)
        .collect();
    if !per.is_empty() {
        notes.push(format!(
            "TMEM users in the recorded run: {}{}",
            users.join(", "),
            if per.len() > 8 { ", ..." } else { "" }
        ));
    }
    (out, notes)
}

fn residency_certified(m: &Machine) -> bool {
    crate::residency::certified(m)
}

/// One run with every allocation at its lowest (`high` false) or highest free column run.
fn run_placed(m0: &Machine, high: bool) -> (Machine, Vec<Finding>, bool) {
    let mut m = with_tenant(m0.clone(), vec![]);
    m.fp = None;
    let mut fs = Vec::new();
    loop {
        let acts = m.actions();
        let Some(i) = acts
            .iter()
            .position(|(a, _)| matches!(a, Action::Split { .. }))
            .or((!acts.is_empty()).then_some(0))
        else {
            break;
        };
        let mut a = acts[i].0.clone();
        if high {
            if let Action::Split { warp, pc, .. } = a {
                if let Some((b, _)) = acts.iter().rev().find(
                    |(b, _)| matches!(b, Action::Split { warp: w, pc: p, .. } if *w == warp && *p == pc),
                ) {
                    a = b.clone();
                }
            }
        }
        m.apply(&a);
        fs.append(&mut m.findings);
    }
    let ok = m.terminal();
    if ok {
        m.exit_checks();
        fs.append(&mut m.findings);
    }
    (m, fs, ok)
}

fn check_certified(
    m0: &Machine,
    plain: &Machine,
    hb: Vec<Finding>,
    mut notes: Vec<String>,
    t0: std::time::Instant,
) -> Verdict {
    let (lo, flo, oklo) = run_placed(m0, false);
    let (hi, fhi, okhi) = run_placed(m0, true);
    let mut findings: Vec<Finding> = Vec::new();
    for f in flo.into_iter().chain(fhi) {
        if !findings.iter().any(|g| g.sig == f.sig) {
            findings.push(f);
        }
    }
    let mut outputs = vec![];
    if oklo {
        outputs.push(lo.outputs());
    }
    if okhi && !outputs.contains(&hi.outputs()) {
        outputs.push(hi.outputs());
    }
    notes.push(format!(
        "certified in the default model (E-WELLSYNC): the happens-before analysis is exact; placement checked by two runs (allocations at the lowest and at the highest free columns){}",
        if outputs.len() > 1 { ": their outputs differ" } else { "" }
    ));
    for f in hb {
        if !findings.iter().any(|g| g.sig == f.sig) {
            findings.push(f);
        }
    }
    findings.sort_by(|a, b| (&a.kind, &a.sig).cmp(&(&b.kind, &b.sig)));
    let mut assumptions = crate::assumptions::for_machine(plain);
    for a in ["E-WELLSYNC", "E-TMEM-COTENANT", "E-TMEM-PERMIT"] {
        if !assumptions.iter().any(|x| x == a) {
            assumptions.push(a.into());
        }
    }
    Verdict {
        states: 2,
        transitions: 0,
        complete: oklo && okhi,
        terminal_states: oklo as usize + okhi as usize,
        distinct_outputs: outputs.len(),
        findings,
        assumptions,
        notes,
        seconds: t0.elapsed().as_secs_f64(),
        outputs,
    }
}

/// CTA pairs worth exploring on one SM: TMEM users of different clusters on one GPU.
pub fn candidate_pairs(m: &Machine) -> Vec<(u16, u16)> {
    let uses = |c: u16| {
        m.grid_code(c as u32)
            .iter()
            .any(|i| matches!(i.op, Op::TmemAlloc { .. } | Op::TmemAllocRaw { .. }))
    };
    let n = m.launch.ctas() as u16;
    let live = |c: u16| m.threads.iter().any(|t| t.cta == c && t.st != St::Exited);
    let mut v = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            if uses(a)
                && uses(b)
                && live(a)
                && live(b)
                && m.launch.cluster_of(a as u32).0 != m.launch.cluster_of(b as u32).0
                && m.launch.g(a as u32).device == m.launch.g(b as u32).device
            {
                v.push((a, b));
            }
        }
    }
    v
}

/// Co-tenant check of a launch: the happens-before analysis, exploration with placement
/// choices, and each candidate pair explored on one SM (at most `max_pairs`).
pub fn check(m0: &Machine, max_states: usize, max_pairs: usize) -> Verdict {
    let t0 = std::time::Instant::now();
    let allocates = (0..m0.launch.grids.len()).any(|gi| {
        m0.grid_threads(gi).any(|t| m0.threads[t].st != St::Exited)
            && m0
                .grid_code(m0.launch.grids[gi].cta0)
                .iter()
                .any(|i| matches!(i.op, Op::TmemAlloc { .. } | Op::TmemAllocRaw { .. }))
    });
    if !allocates {
        return Verdict {
            states: 0,
            transitions: 0,
            complete: true,
            terminal_states: 0,
            distinct_outputs: 0,
            findings: vec![],
            assumptions: vec!["E-TMEM-COTENANT".into()],
            notes: vec![
                "no tcgen05.alloc in the live kernels: a TMEM co-tenant cannot affect them (the default verdict applies)".into(),
            ],
            seconds: t0.elapsed().as_secs_f64(),
            outputs: vec![],
        };
    }
    let (hb, mut notes) = hb_findings(m0);
    // A certified launch (E-WELLSYNC, checked in the default model) has one happens-before
    // for every schedule, so the analysis above is exact for it; exploration in co-tenant
    // mode (no certificate there) would be the whole state space. Placement is then checked
    // by two runs, every allocation at its lowest and at its highest free run.
    let plain = {
        let mut p = m0.clone();
        let mut l = (*p.launch).clone();
        l.env.tmem_cotenant = false;
        l.env.tmem_share.clear();
        p.launch = std::sync::Arc::new(l);
        p
    };
    // Small launches (at most 1024 threads) are explored all the same.
    if m0.threads.len() > 1024 && residency_certified(&plain) {
        return check_certified(m0, &plain, hb, notes, t0);
    }
    let mut v = crate::explore::explore(with_tenant(m0.clone(), vec![]), max_states);
    notes.push(format!(
        "placement choices explored: {} states{}",
        v.states,
        if v.complete { "" } else { " (incomplete)" }
    ));
    let pairs = candidate_pairs(m0);
    if pairs.len() > max_pairs {
        notes.push(format!(
            "{} CTA pairs could share an SM; exploring the first {max_pairs} (the happens-before analysis covers all)",
            pairs.len()
        ));
    }
    for &(a, b) in pairs.iter().take(max_pairs) {
        let w = crate::explore::explore(with_tenant(m0.clone(), vec![(a, b)]), max_states);
        v.states += w.states;
        v.transitions += w.transitions;
        v.complete &= w.complete;
        notes.push(format!(
            "{} and {} on one SM: {:?}, {} states{}",
            m0.cta_name(a),
            m0.cta_name(b),
            w.classes(),
            w.states,
            if w.complete { "" } else { " (incomplete)" }
        ));
        for mut f in w.findings {
            if f.kind == "deadlock" {
                f.sig = format!("tmem-cotenant:pair:{}", f.sig);
                f.msg = format!(
                    "TMEM co-tenant deadlock with {} and {} on one SM (E-TMEM-COTENANT{}): {}",
                    m0.cta_name(a),
                    m0.cta_name(b),
                    if m0.launch.env.tmem_permit {
                        ", E-TMEM-PERMIT"
                    } else {
                        ""
                    },
                    f.msg
                );
            }
            if !v.findings.iter().any(|g| g.sig == f.sig) {
                v.findings.push(f);
            }
        }
        for o in w.outputs {
            if !v.outputs.contains(&o) {
                v.outputs.push(o);
            }
        }
    }
    for f in hb {
        if !v.findings.iter().any(|g| g.sig == f.sig) {
            v.findings.push(f);
        }
    }
    v.distinct_outputs = v.outputs.len();
    for a in ["E-TMEM-COTENANT", "E-TMEM-PERMIT"] {
        if !v.assumptions.iter().any(|x| x == a) {
            v.assumptions.push(a.into());
        }
    }
    v.notes.extend(notes);
    v.findings
        .sort_by(|a, b| (&a.kind, &a.sig).cmp(&(&b.kind, &b.sig)));
    v.seconds = t0.elapsed().as_secs_f64();
    v
}

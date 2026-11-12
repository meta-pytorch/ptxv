/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Dynamic partial-order reduction (Flanagan & Godefroid, POPL 2005) with sleep sets: a
//! stateless search that explores one schedule per Mazurkiewicz trace instead of every
//! interleaving.
//!
//! Processes are splits (named by their lowest thread; a split's elect choices are explored
//! together), in-order async pipes (tcgen05 MMA/commit FIFOs) and individual unordered async
//! operations (TMA loads/stores). Splits of one warp diverge and merge, so a split's program
//! order is the union of its threads' histories, and the warp's merge-sensitive pcs
//! (`Machine::merge_sensitive`) are a resource (footprint space 8): a split that stops at or
//! runs through such a pc while another split of its warp waits there is dependent on it,
//! since the two would merge. Two transitions are dependent when their recorded
//! footprints conflict (`Footprint::conflicts`): overlapping memory/TMEM bytes with at least one
//! write, the same mbarrier object with at least one arrive (polls read it; arrivals into one
//! phase commute unless one completes it; a kernel where a phase gets more arrivals than it
//! expects goes to the stateful explorer, `OVER_ARRIVAL`), the TMEM allocator, CTA exit state vs. async
//! completion, bulk-group state, or a trap. Named-barrier arrivals commute except at a
//! generation boundary of a barrier that `bar.arrive` or more threads than its count use
//! (footprint space 14, E-BAR-ARRIVE / E-BAR-OVER).
//!
//! A transition that disables another process's next transition (consumes its async op, exits
//! its threads, or makes its CTA launch impossible) is dependent on it; the disabled process
//! is added to the frame's backtrack set when the transition is taken (MISS 136, MISS 146).
//! Consuming a remote op and landing it write one key of its pipe (exec.rs `complete_remote`),
//! so a drain also conflicts with the op's own landing.
//!
//! Memory: every frame keeps, for each alternative, its successor `Machine` (`Alt::succ`, a
//! full clone with the transition applied) besides its pid, footprint and op. RSS grows with
//! depth x alternatives x machine size: about 78 MB per frame on a 32-CTA x 8-warp capture
//! (mslk cutlass.f8f8bf16_conv), so the first schedule of a full-grid capture reaches the
//! memory budget before any backtracking. Proposed: keep only (pid, footprint, op, desc,
//! created, disables) per alternative and the frame's own state, and recompute the successor
//! (clone and apply) when an alternative is picked.
//!
//! What is preserved (relative to `explore`): every reachable deadlock, every reachable terminal
//! output, and every finding a transition can produce, because every Mazurkiewicz trace of
//! the dependency relation is represented. The cross-check is `tests/fixtures.rs`, which runs
//! both explorers on every fixture and requires identical verdicts.

use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::time::Instant;

use crate::explore::Action;
use crate::explore::Verdict;
use crate::explore::static_findings;
use crate::machine::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Pid {
    /// Launching an unlaunched CTA (E-CLC).
    Launch(u16),
    /// A split, named by its lowest thread. Splits of a warp diverge and merge; each thread's
    /// program order is carried by `Exec::threads`.
    Split(u32),
    Fifo(u64),
    Op(u32, u32),
}

/// Identity of an async op: (actor, actor clock at issue), unique and stable.
fn op_id(o: &AsyncOp) -> (u32, u32) {
    (o.actor, vc_get(&o.vc, o.actor))
}

fn split_threads(m: &Machine, warp: u32, pc: u32) -> Vec<u32> {
    m.threads
        .iter()
        .enumerate()
        .filter(|(_, t)| t.warp == warp && t.pc == pc && t.st == St::Run)
        .map(|(i, _)| i as u32)
        .collect()
}

fn pid_of(m: &Machine, a: &Action) -> (Pid, Option<(u32, u32)>) {
    match a {
        Action::Split { warp, pc, .. } => (Pid::Split(split_threads(m, *warp, *pc)[0]), None),
        Action::Launch(c) => (Pid::Launch(*c), None),
        Action::Clc(i, _) | Action::Async(i) => {
            let o = &m.pending[*i];
            let id = op_id(o);
            (
                o.fifo.map(Pid::Fifo).unwrap_or(Pid::Op(id.0, id.1)),
                Some(id),
            )
        }
    }
}

struct Alt {
    pid: Pid,
    /// Threads of a split (empty for async processes).
    threads: Vec<u32>,
    op: Option<(u32, u32)>,
    desc: String,
    succ: Machine,
    fp: Footprint,
    findings: Vec<Finding>,
    created: Vec<(u32, u32)>,
    /// Other processes enabled at the frame whose next transition this one disables (see
    /// `successors`); taking it adds them to the frame's backtrack set.
    disables: Vec<Pid>,
    /// It consumes a FIFO's op behind the head while the head stays pending (a backtrack the
    /// head-only rule would miss; counted in `NONHEAD_BACKTRACKS` when taken).
    nonhead: bool,
}

/// Transitions taken that consumed a non-head FIFO op (their FIFO is backtracked although its
/// head stays pending). Instrumentation for the cross-check tests (fixtures/e_dpor_nonhead).
pub static NONHEAD_BACKTRACKS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Pid of the process async op `o` belongs to.
fn op_pid(o: &AsyncOp) -> Pid {
    let id = op_id(o);
    o.fifo.map(Pid::Fifo).unwrap_or(Pid::Op(id.0, id.1))
}

/// The transition taken from a frame.
struct Exec {
    pid: Pid,
    threads: Vec<u32>,
    fp: Footprint,
    /// DPOR happens-before clock over processes: pid -> highest trace index that
    /// happens-before (or is) this transition.
    cv: HashMap<Pid, usize>,
    desc: String,
    created: Vec<(u32, u32)>,
}

struct Frame {
    alts: Vec<Alt>,
    backtrack: BTreeSet<Pid>,
    done: BTreeSet<Pid>,
    todo: Vec<usize>,
    sleep: Vec<(Pid, Footprint)>,
    exec: Option<Exec>,
}

fn successors(m: &Machine) -> Vec<Alt> {
    let before: HashSet<(u32, u32)> = m.pending.iter().map(op_id).collect();
    let mut alts: Vec<Alt> = m
        .actions()
        .into_iter()
        .map(|(a, desc)| {
            let (pid, op) = pid_of(m, &a);
            let threads = match a {
                Action::Split { warp, pc, .. } => split_threads(m, warp, pc),
                Action::Async(_) | Action::Clc(..) | Action::Launch(_) => vec![],
            };
            let mut n = m.clone();
            n.fp = Some(Box::default());
            n.apply(&a);
            let mut fp = *n.fp.take().unwrap();
            fp.normalize();
            let findings = std::mem::take(&mut n.findings);
            let created = n
                .pending
                .iter()
                .map(op_id)
                .filter(|id| !before.contains(id))
                .collect();
            Alt {
                pid,
                threads,
                op,
                desc,
                succ: n,
                fp,
                findings,
                created,
                disables: vec![],
                nonhead: false,
            }
        })
        .collect();
    // A transition that disables another process's next transition (MISS 136, MISS 146): that
    // process has no next transition in the successor, so FG never compares one with this
    // transition and never finds the dependent pair. Each such process is enabled here, so
    // FG's rule adds it to this frame's backtrack set (the explore loop does, from
    // `disables`). Its dependencies on earlier transitions were found on arriving at this
    // frame: it is among this frame's next transitions. Three ways to disable one:
    // - consuming async ops (drain_remote: a cluster-scope release, CGAERRBAR, a non-.cta
    //   fence or MEMBAR.ALL.GPU; deliver_fifo_before): every op pending before and gone after,
    //   other than the transition's own, disables its process, its FIFO's head or not (a
    //   non-head op can land only after the ops ahead of it, which this transition may also
    //   have to follow);
    // - exiting other threads (a trap or __assertfail exits every thread): a split whose
    //   threads are all Exited after;
    // - a CTA launch that is no longer possible after (its threads exited by a trap).
    let pids: Vec<Pid> = alts.iter().map(|a| a.pid).collect();
    for k in 0..alts.len() {
        let a = &alts[k];
        let live: HashSet<(u32, u32)> = a.succ.pending.iter().map(op_id).collect();
        let mut d: Vec<Pid> = Vec::new();
        let mut nonhead = false;
        let mut push = |p: Pid| {
            if p != a.pid && !d.contains(&p) {
                d.push(p);
            }
        };
        for o in &m.pending {
            let id = op_id(o);
            if Some(id) != a.op && !live.contains(&id) {
                let p = op_pid(o);
                nonhead |= matches!(p, Pid::Fifo(_))
                    && alts
                        .iter()
                        .any(|b| b.pid == p && b.op.is_some_and(|h| live.contains(&h)));
                push(p);
            }
        }
        for b in &alts {
            let gone = match b.pid {
                Pid::Split(_) => {
                    !b.threads.is_empty()
                        && !b.threads.iter().any(|t| a.threads.contains(t))
                        && b.threads
                            .iter()
                            .all(|&t| a.succ.threads[t as usize].st == St::Exited)
                }
                Pid::Launch(c) => !a
                    .succ
                    .threads
                    .iter()
                    .any(|t| t.cta == c && t.st == St::Unlaunched),
                Pid::Fifo(_) | Pid::Op(..) => false,
            };
            if gone {
                push(b.pid);
            }
        }
        // A disabled process that is not enabled here (a FIFO whose head is not an
        // alternative): FG's rule then takes every enabled process.
        if d.iter().any(|p| !pids.contains(p)) {
            d = pids.iter().copied().filter(|p| *p != a.pid).collect();
        }
        alts[k].disables = d;
        alts[k].nonhead = nonhead;
    }
    alts
}

/// Footprints of the next transitions of blocked warps (their blocking operation), so that
/// a transition that will later unblock or race with them is recognized as dependent.
fn blocked_footprints(m: &Machine, enabled: &BTreeSet<Pid>) -> Vec<(Pid, Footprint, Vec<u32>)> {
    let mut out: Vec<(Pid, Footprint, Vec<u32>)> = Vec::new();
    for (w, pc, members) in m.splits() {
        let pid = Pid::Split(members[0] as u32);
        if enabled.contains(&pid) {
            continue;
        }
        let t = members[0];
        let inst = &m.k.code[pc as usize];
        out.push((
            pid,
            Footprint::default(),
            members.iter().map(|&u| u as u32).collect(),
        ));
        let f = &mut out.last_mut().unwrap().1;
        // Its action will start here (a split of the same warp arriving would merge).
        if m.merge_sensitive(pc) {
            f.add(8, w, pc as u64, pc as u64 + 1, true);
        }
        // Blocked at an aligned barrier: whether it passes depends on where the warp's other
        // splits are (`Machine::rec_pc`, the warp's shared aligned-barrier resource).
        if inst.op.is_aligned_bar() {
            f.add(8, w, 1 << 40, (1 << 40) + 1, true);
        }
        // ASSUME(E-HW-SPLIT-UR): held back by another split's uniform-register window.
        if m.launch.hw_split_ur && !m.ur_window.is_empty() && m.ur_window_blocks(&members, pc) {
            f.add(8, w, 1 << 41, (1 << 41) + 1, true);
        }
        // A barrier released during the warp's last action is acquired by its next one.
        for &u in &members {
            if let Some(key) = m.threads[u].acq_bar {
                f.acq.push(key);
            }
        }
        // A blocked warp-level poll (spin::wpoll_loops) reads its flags.
        if let Some(&b) = m.wpolls.get(&pc) {
            for (_, _, loc, bytes, _) in
                crate::spin::wpoll_dry(m, &members, pc, b).unwrap_or_default()
            {
                match loc {
                    crate::exec::Loc::Shared { cta, off } => {
                        f.add(0, cta as u32, off as u64, off as u64 + bytes as u64, false)
                    }
                    crate::exec::Loc::Global { alloc, off } => {
                        f.add(1, alloc as u32, off, off + bytes as u64, false)
                    }
                    _ => {}
                }
            }
        }
        // A control-word bulk read wait blocks like Op::BulkWait.
        if m.k
            .bulk_read_waits
            .binary_search_by_key(&(pc as usize), |w| w.0)
            .is_ok()
        {
            for &u in &members {
                f.add(5, u as u32, 0, 1, false)
            }
        }
        match &inst.op {
            // A blocked poll reads the mbarrier and its phase state, which the completion that
            // unblocks it (or, completing the next phase first, keeps it blocked) writes.
            ptxv_ir::Op::MbarTryWait { addr, .. } => {
                if let Ok(a) = m.addr(t, addr) {
                    let off = a & 0xff_ffff;
                    f.add(0, m.threads[t].cta as u32, off, off + 8, false);
                    f.add(6, m.threads[t].cta as u32, off, off + 8, false);
                }
            }
            ptxv_ir::Op::BulkWait { .. } => {
                for &u in &members {
                    f.add(5, u as u32, 0, 1, false)
                }
            }
            // A blocked poll (strong or plain, spin::poll_loops) reads its location: a write
            // there may unblock it.
            ptxv_ir::Op::Ld {
                space, bytes, addr, ..
            }
            | ptxv_ir::Op::Atom {
                space, bytes, addr, ..
            } => {
                for &u in &members {
                    let loc = m
                        .addr(u, addr)
                        .and_then(|a| m.resolve(u, *space, a, *bytes as usize));
                    match loc {
                        Ok(crate::exec::Loc::Shared { cta, off }) => {
                            f.add(0, cta as u32, off as u64, off as u64 + *bytes as u64, false)
                        }
                        Ok(crate::exec::Loc::Global { alloc, off }) => {
                            f.add(1, alloc as u32, off, off + *bytes as u64, false)
                        }
                        _ => {}
                    }
                }
            }
            ptxv_ir::Op::GridDepWait => {
                let gi = m.launch.grid_index(m.threads[t].cta as u32);
                if let Some(p) = m.launch.prev_grid(gi) {
                    let g = &m.launch.grids[p];
                    for c in g.cta0..g.cta0 + g.ctas() {
                        f.add(9, c, 0, u64::MAX, false);
                    }
                }
            }
            ptxv_ir::Op::TmemAlloc { .. } | ptxv_ir::Op::TmemAllocRaw { .. } => {
                f.add(3, m.threads[t].cta as u32, 0, 1, true)
            }
            _ => {}
        }
    }
    // A later grid's CTAs that may not launch yet wait on the previous grid (E-PDL).
    let mut gated: Vec<u16> = m
        .threads
        .iter()
        .filter(|t| t.st == St::Unlaunched && m.launch.grid_index(t.cta as u32) > 0)
        .map(|t| t.cta)
        .collect();
    gated.dedup();
    for c in gated {
        let pid = Pid::Launch(c);
        if enabled.contains(&pid) {
            continue;
        }
        let mut f = Footprint::default();
        if let Some(p) = m.launch.prev_grid(m.launch.grid_index(c as u32)) {
            let g = &m.launch.grids[p];
            for q in g.cta0..g.cta0 + g.ctas() {
                f.add(9, q, 0, u64::MAX, false);
            }
        }
        out.push((pid, f, vec![]));
    }
    for (_, f, _) in out.iter_mut() {
        f.normalize();
    }
    out
}

pub fn explore_dpor(m0: Machine, max_states: usize) -> Verdict {
    // Limited residency and the TMEM co-tenant (residency.rs, tenant.rs): the stateful
    // explorer only (the footprints do not name residency slots or allocation choices).
    // NIC queue pairs (nic.rs): the stateful explorer only.
    if m0.launch.env.residency.is_some() || m0.launch.env.tmem_cotenant || m0.nic.is_some() {
        return crate::explore::explore(m0, max_states);
    }
    let base = m0.clone();
    let mut v = crate::smem::two_pass(m0, |m0| {
        crate::explore::per_leader_policy(m0, 1, |m, _| explore_dpor_one(m, max_states))
    });
    // An over-arrival: the stateful explorer decides (`OVER_ARRIVAL`).
    if v.notes.iter().any(|n| n == OVER_ARRIVAL) {
        let mut v = crate::explore::explore(base, max_states);
        v.notes.push(
            "DPOR saw an mbarrier update that does not commute with the other updates of its \
             phase (an over-arrival: more arrivals than expected before a completion that \
             orders them): checked by the stateful explorer instead"
                .into(),
        );
        return v;
    }
    // S-INIT-FILLS: undefined values that reach only output data.
    crate::fills::check(&base, &mut v);
    v
}

/// Marker note of a DPOR search that stopped at an mbarrier update writing footprint space 25
/// (`rec_late` / `rec_count` in exec.rs): an update that lands after a completion its actor
/// does not know of, or an arrival of more than the pending arrivals at once. Such an update
/// makes the dependency of arrivals depend on the state: an arrival that commuted with the
/// others where it ran would land late after them, and the sleep sets and the last-conflict
/// backtracking (which assume a transition's footprint does not change while it waits) can
/// miss that order (`fixtures/mbar_over_arrival`; `fixtures/mbar_count_order` and its
/// variants, which all over-arrive). Every execution of such a kernel has some update that
/// writes the key (the arrivals beyond the expected count are not ordered after the completion,
/// in whichever order they come), so the first DPOR execution that meets one stops the search,
/// and the stateful explorer redoes it.
const OVER_ARRIVAL: &str = "dpor: over-arrival";

fn explore_dpor_one(m0: Machine, max_states: usize) -> Verdict {
    let t0 = Instant::now();
    let mut findings: Vec<Finding> = static_findings(&m0);
    findings.extend(crate::spin::check(&m0));
    let assumptions = crate::assumptions::for_machine(&m0);
    let mut seen_sig: HashMap<String, usize> = findings
        .iter()
        .enumerate()
        .map(|(i, f)| (f.sig.clone(), i))
        .collect();
    let mut outputs: Vec<Vec<u8>> = Vec::new();
    let mut out_seen: HashSet<Vec<u8>> = HashSet::new();
    let (mut states, mut transitions, mut terminal) = (1usize, 0usize, 0usize);
    let mut complete = true;

    let debug = std::env::var("PTXV_DPOR_DEBUG").is_ok();
    let mut why: HashMap<String, usize> = HashMap::new();
    let mut frames: Vec<Frame> = Vec::new();
    // Start: frame 0 holds the initial state's alternatives.
    let new_frame = |m: &Machine, sleep: Vec<(Pid, Footprint)>| Frame {
        alts: successors(m),
        backtrack: BTreeSet::new(),
        done: BTreeSet::new(),
        todo: vec![],
        sleep,
        exec: None,
    };
    frames.push(new_frame(&m0, vec![]));
    let mut cur: Machine = m0;
    let over = |f: &Frame| {
        f.alts.iter().any(|a| {
            a.fp.w
                .iter()
                .any(|x| x.0 == crate::machine::SPACE_MBAR_OVER)
        })
    };
    let mut over_arrival = over(&frames[0]);

    // Called on reaching a new frame (index n = frames.len()-1, whose state is `cur`).
    // Returns false if the state is terminal/deadlocked/sleep-blocked (nothing to explore).
    let mut progress = crate::explore::Progress::new(
        format!("{}dpor", if cur.elect_high { "leader=high " } else { "" }),
        max_states,
    );
    loop {
        progress.tick(states, terminal, frames.len(), &findings);
        let n = frames.len() - 1;
        // ---- arriving at frame n for the first time: FG backtrack update + initial choice
        if frames[n].exec.is_none()
            && frames[n].done.is_empty()
            && frames[n].todo.is_empty()
            && frames[n].backtrack.is_empty()
        {
            let enabled: BTreeSet<Pid> = frames[n].alts.iter().map(|a| a.pid).collect();
            if enabled.is_empty() {
                if cur.terminal() {
                    cur.exit_checks();
                    for mut f in cur.findings.drain(..) {
                        if !seen_sig.contains_key(&f.sig) {
                            f.trace = trace(&frames);
                            seen_sig.insert(f.sig.clone(), findings.len());
                            findings.push(f);
                        }
                    }
                    terminal += 1;
                    let o = cur.outputs();
                    if out_seen.insert(o.clone()) {
                        outputs.push(o);
                    }
                } else {
                    let (sig, msg) = cur.deadlock_report();
                    if !seen_sig.contains_key(&sig) {
                        seen_sig.insert(sig.clone(), findings.len());
                        findings.push(Finding {
                            kind: "deadlock".into(),
                            sig,
                            msg,
                            trace: trace(&frames),
                            ..Default::default()
                        });
                    }
                }
            } else {
                // Next transitions of every process: enabled (union over alternatives) and blocked.
                // (footprint, op, threads) of a process's next transition.
                type Next = (Footprint, Option<(u32, u32)>, Vec<u32>);
                let mut nexts: HashMap<Pid, Next> = HashMap::new();
                for a in &frames[n].alts {
                    let e = nexts.entry(a.pid).or_insert((
                        Footprint::default(),
                        a.op,
                        a.threads.clone(),
                    ));
                    e.0.union(&a.fp);
                }
                for (p, f, th) in blocked_footprints(&cur, &enabled) {
                    nexts.insert(p, (f, None, th));
                }
                for (p, (fp, op, th)) in &nexts {
                    // HB clock of p's next transition: p's history, its issuer (async ops),
                    // and whatever the transition itself acquires (a poll acquires the phase
                    // completion it observes, so it cannot be reordered before it).
                    let mut pred = pred_cv(&frames, *p, *op, th);
                    if !fp.acq.is_empty() {
                        for f in &frames[..n] {
                            let e = f.exec.as_ref().unwrap();
                            if fp.acquires_from(&e.fp) {
                                for (q, &k) in &e.cv {
                                    let v = pred.entry(*q).or_insert(0);
                                    *v = (*v).max(k);
                                }
                            }
                        }
                    }
                    for j in (0..n).rev() {
                        let e = frames[j].exec.as_ref().unwrap();
                        if e.pid == *p || !e.fp.conflicts(fp) {
                            continue;
                        }
                        if pred.get(&e.pid).is_some_and(|&k| k >= j) {
                            continue; // already ordered before p's next transition
                        }
                        if debug {
                            let a = frames[j]
                                .exec
                                .as_ref()
                                .unwrap()
                                .desc
                                .split(" runs from ")
                                .last()
                                .unwrap_or("")
                                .to_string();
                            let b = frames[n]
                                .alts
                                .iter()
                                .find(|x| x.pid == *p)
                                .map(|x| {
                                    x.desc.split(" runs from ").last().unwrap_or("").to_string()
                                })
                                .unwrap_or(format!("{p:?} (blocked)"));
                            *why.entry(format!("{a}  <->  {b}")).or_insert(0usize) += 1;
                        }
                        let fj = &mut frames[j];
                        // p, or the split at j that p's lead thread then belonged to.
                        let lead = match p {
                            Pid::Split(t) => Some(*t),
                            _ => None,
                        };
                        if fj.alts.iter().any(|a| a.pid == *p) {
                            fj.backtrack.insert(*p);
                        } else if let Some(a) = fj
                            .alts
                            .iter()
                            .find(|a| lead.is_some_and(|t| a.threads.contains(&t)))
                        {
                            let q = a.pid;
                            fj.backtrack.insert(q);
                        } else {
                            let all: Vec<Pid> = fj.alts.iter().map(|a| a.pid).collect();
                            fj.backtrack.extend(all);
                        }
                        break;
                    }
                }
                // Initial choice: any enabled process that is not asleep.
                let asleep: BTreeSet<Pid> = frames[n].sleep.iter().map(|s| s.0).collect();
                if let Some(p) = enabled.iter().find(|p| !asleep.contains(p)) {
                    frames[n].backtrack.insert(*p);
                }
            }
        }
        // ---- pick the next alternative to explore from frame n
        let next_alt = loop {
            let f = &mut frames[n];
            if let Some(i) = f.todo.pop() {
                break Some(i);
            }
            let asleep: BTreeSet<Pid> = f.sleep.iter().map(|s| s.0).collect();
            let pick = f
                .backtrack
                .iter()
                .find(|p| !f.done.contains(p) && !asleep.contains(p))
                .copied();
            match pick {
                Some(p) => {
                    f.done.insert(p);
                    f.todo = f
                        .alts
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| a.pid == p)
                        .map(|(i, _)| i)
                        .collect();
                    f.todo.reverse();
                }
                None => break None,
            }
        };
        let Some(ai) = next_alt else {
            // Frame exhausted: backtrack.
            frames.pop();
            if frames.is_empty() {
                break;
            }
            let k = frames.len() - 1;
            // The parent's explored process goes to sleep for its remaining siblings.
            let e = frames[k].exec.take().unwrap();
            frames[k].sleep.push((e.pid, e.fp));
            // `cur` is only read when a frame is first reached, so it need not be restored.
            continue;
        };
        if states >= max_states || crate::membudget::exceeded() || crate::membudget::past_deadline()
        {
            complete = false;
            break;
        }
        // ---- execute alternative ai of frame n
        let (pid, threads, fp, desc, created, succ, fnds, op) = {
            let a = &frames[n].alts[ai];
            (
                a.pid,
                a.threads.clone(),
                a.fp.clone(),
                a.desc.clone(),
                a.created.clone(),
                a.succ.clone(),
                a.findings.clone(),
                a.op,
            )
        };
        transitions += 1;
        // MISS 136 / MISS 146: the processes this transition disables (`successors`).
        let disables = frames[n].alts[ai].disables.clone();
        if frames[n].alts[ai].nonhead {
            NONHEAD_BACKTRACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        frames[n].backtrack.extend(disables);
        let mut cv = pred_cv(&frames, pid, op, &threads);
        for f in &frames[..n] {
            let e = f.exec.as_ref().unwrap();
            if e.pid != pid && (e.fp.conflicts(&fp) || fp.acquires_from(&e.fp)) {
                for (p, &k) in &e.cv {
                    let v = cv.entry(*p).or_insert(0);
                    *v = (*v).max(k);
                }
            }
        }
        cv.insert(pid, n);
        let child_sleep: Vec<(Pid, Footprint)> = frames[n]
            .sleep
            .iter()
            .filter(|(q, f)| *q != pid && !f.conflicts(&fp))
            .cloned()
            .collect();
        frames[n].exec = Some(Exec {
            pid,
            threads,
            fp,
            cv,
            desc,
            created,
        });
        for mut f in fnds {
            if !seen_sig.contains_key(&f.sig) {
                f.trace = trace(&frames);
                seen_sig.insert(f.sig.clone(), findings.len());
                findings.push(f);
            }
        }
        states += 1;
        cur = succ;
        frames.push(new_frame(&cur, child_sleep));
        if over(frames.last().unwrap()) {
            over_arrival = true;
            break;
        }
    }
    if debug {
        let mut v: Vec<_> = why.into_iter().collect();
        v.sort_by_key(|e| std::cmp::Reverse(e.1));
        for (k, c) in v.iter().take(15) {
            eprintln!("dpor backtrack {c:>7}  {k}");
        }
    }
    if over_arrival {
        return Verdict {
            notes: vec![OVER_ARRIVAL.into()],
            states,
            transitions,
            complete: false,
            terminal_states: terminal,
            distinct_outputs: 0,
            findings,
            assumptions,
            seconds: t0.elapsed().as_secs_f64(),
            outputs: vec![],
        };
    }
    Verdict {
        notes: if complete {
            vec![]
        } else {
            crate::membudget::note()
                .into_iter()
                .chain(crate::membudget::deadline_note())
                .collect()
        },
        states,
        transitions,
        complete,
        terminal_states: terminal,
        distinct_outputs: outputs.len(),
        findings,
        assumptions,
        seconds: t0.elapsed().as_secs_f64(),
        outputs,
    }
}

/// HB clock that the next transition of `p` inherits: p's own last transition, plus (for an
/// async op) the transition that issued it.
fn pred_cv(
    frames: &[Frame],
    p: Pid,
    op: Option<(u32, u32)>,
    threads: &[u32],
) -> HashMap<Pid, usize> {
    let mut cv: HashMap<Pid, usize> = HashMap::new();
    let mut join = |c: &HashMap<Pid, usize>| {
        for (q, &k) in c {
            let v = cv.entry(*q).or_insert(0);
            *v = (*v).max(k);
        }
    };
    if threads.is_empty() {
        for f in frames.iter().rev() {
            if let Some(e) = &f.exec {
                if e.pid == p {
                    join(&e.cv);
                    break;
                }
            }
        }
    } else {
        // Program order of every member thread: the last transition that ran it.
        let mut left: Vec<u32> = threads.to_vec();
        for f in frames.iter().rev() {
            if left.is_empty() {
                break;
            }
            if let Some(e) = &f.exec {
                if !e.threads.is_empty() && left.iter().any(|t| e.threads.contains(t)) {
                    join(&e.cv);
                    left.retain(|t| !e.threads.contains(t));
                }
            }
        }
    }
    if let Some(id) = op {
        for f in frames.iter().rev() {
            if let Some(e) = &f.exec {
                if e.created.contains(&id) {
                    join(&e.cv);
                    break;
                }
            }
        }
    }
    cv
}

fn trace(frames: &[Frame]) -> Vec<String> {
    frames
        .iter()
        .filter_map(|f| f.exec.as_ref().map(|e| e.desc.clone()))
        .collect()
}

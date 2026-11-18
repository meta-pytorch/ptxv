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

use crate::explore::static_findings;
use crate::explore::Action;
use crate::explore::Verdict;
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
        Action::AtomicComponent { warp, pc, .. } => (
            Pid::Split(m.atomic_component_members(*warp, *pc)[0] as u32),
            None,
        ),
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
                Action::AtomicComponent { warp, pc, .. } => m
                    .atomic_component_members(warp, pc)
                    .into_iter()
                    .map(|t| t as u32)
                    .collect(),
                Action::Async(_) | Action::Clc(..) | Action::Launch(_) => vec![],
            };
            let mut n = m.clone();
            n.set_footprint(Some(Box::default()));
            n.apply(&a);
            let mut fp = *n.take_footprint().unwrap();
            fp.normalize();
            // Dry poll checks need the path's observed signatures to recognize repeats.
            let findings = n.findings.clone();
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
        let shared = ptxv_memory_write::footprint::shared_register_range(w);
        f.add(shared.0, shared.1, shared.2, shared.3, false);
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
                for &u in &members {
                    if !m.prd(u, inst.guard) {
                        continue;
                    }
                    match m
                        .addr(u, addr)
                        .and_then(|a| m.resolve(u, ptxv_ir::Space::Shared, a, 8))
                    {
                        Ok(crate::exec::Loc::Shared { cta, off }) => {
                            let off = off as u64;
                            f.add(0, cta as u32, off, off + 8, false);
                            f.add(6, cta as u32, off, off + 8, false);
                            f.add(12, cta as u32, off << 20, (off + 1) << 20, false);
                        }
                        _ => f.all = true,
                    }
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
        let gi = m.launch.grid_index(c as u32);
        for p in m
            .launch
            .prev_grid(gi)
            .into_iter()
            .chain(m.launch.grids[gi].waits.iter().copied())
        {
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

fn select_stateful_remote_barriers(machine: &Machine) -> bool {
    machine.k.front == ptxv_ir::Front::Ptx
        && machine.launch.grids.len() == 1
        && machine.launch.grids[0]
            .cluster
            .iter()
            .any(|&extent| extent > 1)
        && machine
            .k
            .code
            .iter()
            .any(|inst| matches!(inst.op, ptxv_ir::Op::StAsync { .. }))
        && machine
            .k
            .code
            .iter()
            .any(|inst| matches!(inst.op, ptxv_ir::Op::MbarTryWait { .. }))
        && !machine
            .k
            .code
            .iter()
            .any(|inst| matches!(inst.op, ptxv_ir::Op::Atom { .. }))
}

fn select_stateful_tmem(machine: &Machine) -> bool {
    use ptxv_ir::AtomOp;
    use ptxv_ir::Front;
    use ptxv_ir::Op;
    use ptxv_ir::Space;

    machine.launch.grids.len() == 1
        && machine.launch.grids[0].grid == [1, 1, 1]
        && machine.launch.grids[0].cluster == [1, 1, 1]
        && machine.k.code.iter().any(|inst| {
            matches!(
                (machine.k.front, &inst.op),
                (Front::Ptx, Op::TmemAlloc { .. }) | (Front::Sass, Op::TmemAllocRaw { .. })
            )
        })
        && machine.k.code.iter().any(|inst| {
            matches!(
                (machine.k.front, &inst.op),
                (Front::Ptx, Op::TmemDealloc { .. }) | (Front::Sass, Op::TmemFreeRaw { .. })
            )
        })
        && machine
            .k
            .code
            .iter()
            .any(|inst| matches!(inst.op, Op::TmemLd { .. }))
        && machine
            .k
            .code
            .iter()
            .any(|inst| matches!(inst.op, Op::TmemSt { .. }))
        && machine
            .k
            .code
            .iter()
            .any(|inst| matches!(inst.op, Op::MbarTryWait { .. }))
        && machine.k.code.iter().all(|inst| match inst.op {
            Op::Atom {
                op: AtomOp::And | AtomOp::Or,
                space: Space::Shared,
                bytes: 4,
                d: None,
                p: None,
                new: None,
                ..
            } => true,
            Op::Atom { .. } => false,
            _ => true,
        })
}

pub fn explore_dpor(m0: Machine, max_states: usize) -> Verdict {
    // Limited residency and the TMEM co-tenant (residency.rs, tenant.rs): the stateful
    // explorer only (the footprints do not name residency slots or allocation choices).
    // NIC queue pairs (nic.rs): the stateful explorer only.
    if m0.launch.env.residency.is_some() || m0.launch.env.tmem_cotenant || m0.nic.is_some() {
        return crate::explore::explore(m0, max_states);
    }
    if select_stateful_remote_barriers(&m0) {
        let mut verdict = crate::explore::explore(m0, max_states);
        verdict.notes.push(format!(
            "Stateful explorer selected for PTX cluster st.async/mbarrier code without atomics (state cap {max_states})"
        ));
        return verdict;
    }
    if select_stateful_tmem(&m0) {
        let front = match m0.k.front {
            ptxv_ir::Front::Ptx => "PTX",
            ptxv_ir::Front::Sass => "SASS",
        };
        let mut verdict = crate::explore::explore(m0, max_states);
        verdict.notes.push(format!(
            "Stateful explorer selected for single-CTA {front} TMEM lifecycle/mbarrier code with only returnless shared32 AND/OR atomics (state cap {max_states})"
        ));
        return verdict;
    }
    if m0.private_register_projection().is_some() {
        let mut verdict = crate::explore::explore(m0, max_states);
        verdict.notes.push(format!(
            "Stateful explorer selected for forward PTX integer-atomic private continuations (state cap {max_states})"
        ));
        return verdict;
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

#[cfg(test)]
mod uniform_blocked_footprint_tests {
    use super::*;

    #[test]
    fn blocked_wait_tracks_every_lanes_resolved_barrier_and_phase() {
        let text = r#"
.version 9.0
.target sm_100a
.address_size 64
.visible .entry k() {
  .reg .b32 %r;
  .reg .b64 %rd<2>;
  .reg .pred %p;
  .shared .align 8 .b64 bars[2];
  mov.u32 %r, %tid.x;
  mul.lo.u32 %r, %r, 8;
  mov.u64 %rd0, bars;
  cvt.u64.u32 %rd1, %r;
  add.u64 %rd0, %rd0, %rd1;
WAIT:
  mbarrier.try_wait.parity.shared::cta.b64 %p, [%rd0], 0;
  @!%p bra WAIT;
  ret;
}
"#;
        let kernel = ptxv_ptx::parse_kernel(text, Some("k")).unwrap();
        let case = ptxv_case::Case {
            dir: std::path::PathBuf::new(),
            m: serde_json::from_value(serde_json::json!({
                "name":"blocked-wait-lanes", "description":"per-lane query dependencies",
                "entry":"k", "grid":[1,1,1], "block":[2,1,1],
                "outputs":[], "buffers":[], "params":[], "variants":[]
            }))
            .unwrap(),
        };
        let launch = crate::build_launch(&case, &kernel).unwrap();
        let mut machine = Machine::new(std::sync::Arc::new(kernel), std::sync::Arc::new(launch));
        for pc in 0..5 {
            machine.exec_one(&[0, 1], pc).unwrap();
        }
        let pc = machine.threads[0].pc;
        assert!(matches!(
            machine.k.code[pc as usize].op,
            ptxv_ir::Op::MbarTryWait { .. }
        ));
        for t in 0..2 {
            machine.threads[t].make_mut().wait_error = Some(pc);
        }
        assert!(!machine.split_enabled(&[0, 1], pc));
        let blocked = blocked_footprints(&machine, &BTreeSet::new());
        assert_eq!(blocked.len(), 1);
        let read = &blocked[0].1;
        assert!(!read.all);
        let ptxv_ir::Op::MbarTryWait { addr, .. } = &machine.k.code[pc as usize].op else {
            unreachable!()
        };
        let mut targets = Vec::new();
        for t in 0..2 {
            let crate::exec::Loc::Shared { cta, off } = machine
                .resolve(t, ptxv_ir::Space::Shared, machine.addr(t, addr).unwrap(), 8)
                .unwrap()
            else {
                unreachable!()
            };
            targets.push((cta as u32, off as u64));
        }
        assert_eq!(targets[0].0, targets[1].0);
        assert_eq!(targets[0].1 + 8, targets[1].1);
        for (cta, off) in targets {
            for (space, lo, hi) in [
                (0, off, off + 8),
                (6, off, off + 8),
                (12, off << 20, (off + 1) << 20),
            ] {
                let mut write = Footprint::default();
                write.add(space, cta, lo, hi, true);
                assert!(
                    read.conflicts(&write),
                    "lane target {off}, space {space}: {read:?}"
                );
            }
        }
    }

    #[test]
    fn blocked_split_keeps_its_uniform_read_dependency() {
        let instructions = [
            "S2UR UR0, SR_CTAID.X",
            "S2R R0, SR_LANEID",
            "ISETP.EQ.U32.AND P0, PT, R0, RZ, PT",
            "@P0 SYNCS.EXCH.64 URZ, [RZ], 0x7ffff800001ffffe",
            "@P0 BRA `(.L_wait)",
            "UMOV UR0, 0x80000000",
            "EXIT",
            "SYNCS.PHASECHK.TRANS64 P1, [RZ], UR0",
            "@!P1 BRA `(.L_wait)",
            "BPT.TRAP 0x1",
            "EXIT",
        ];
        let mut text = String::from(".text.k:\n");
        for (pc, instruction) in instructions.iter().enumerate() {
            if pc == 7 {
                text.push_str(".L_wait:\n");
            }
            let control = ptxv_sass::scoreboard::Ctrl {
                stall: 15,
                yld: false,
                wbar: matches!(pc, 0 | 1 | 7).then_some(0),
                rbar: None,
                wait: if matches!(pc, 1 | 2 | 8) { 1 } else { 0 },
                reuse: 0,
            };
            text.push_str(&format!(
                "/*{:04x}*/ {instruction} ;\n/* 0x{:016x} */\n",
                pc * 16,
                control.encode(0)
            ));
        }
        let kernel = ptxv_sass::lower(&text, "k", &ptxv_sass::CubinMeta::default()).unwrap();
        assert!(
            kernel.static_findings.is_empty(),
            "{:?}",
            kernel.static_findings
        );
        let manifest = serde_json::from_value(serde_json::json!({
            "name":"uniform-blocked-wait", "description":"shared warp register dependencies",
            "entry":"k", "grid":[1,1,1], "block":[32,1,1], "dynamic_smem":16,
            "outputs":[], "buffers":[], "params":[], "variants":[]
        }))
        .unwrap();
        let case = ptxv_case::Case {
            dir: std::path::PathBuf::new(),
            m: manifest,
        };
        let launch = crate::build_launch(&case, &kernel).unwrap();
        let mut machine = Machine::new(std::sync::Arc::new(kernel), std::sync::Arc::new(launch));
        for pc in 0..5 {
            machine.exec_one(&(0..32).collect::<Vec<_>>(), pc).unwrap();
        }
        assert!(!machine.split_enabled(&[0], 7));
        let enabled = machine
            .actions()
            .iter()
            .map(|(action, _)| pid_of(&machine, action).0)
            .collect();
        let blocked = blocked_footprints(&machine, &enabled);
        let (_, read, _) = blocked
            .iter()
            .find(|(pid, _, _)| *pid == Pid::Split(0))
            .unwrap();
        machine.set_footprint(Some(Box::default()));
        machine.wr(
            1,
            ptxv_ir::Reg {
                file: ptxv_ir::File::UR,
                idx: 0,
            },
            0x80000000,
            1,
        );
        let mut write = *machine.take_footprint().unwrap();
        write.normalize();
        assert!(write.conflicts(read));
        assert!(machine.split_enabled(&[0], 7));
    }
}

#[cfg(test)]
mod poll_finding_history_tests {
    use std::path::Path;
    use std::sync::Arc;

    use ptxv_case::Case;

    use super::*;

    #[test]
    fn repeated_poll_races_block_after_their_first_report() {
        let case =
            Case::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/l1_litmus"))
                .unwrap();
        let text = case.variant_ptx("pr3").unwrap();
        let kernel = ptxv_ptx::parse_kernel(&text, Some(&case.m.entry)).unwrap();
        let launch = crate::build_launch(&case, &kernel).unwrap();
        let result = explore_dpor(Machine::new(Arc::new(kernel), Arc::new(launch)), 1000);
        assert!(
            result.complete,
            "{} states: {:?}",
            result.states,
            result.classes()
        );
        let mut classes = result.classes();
        classes.retain(|class| class != "nondeterministic");
        assert_eq!(classes, ["race", "stale"]);
        let signatures: HashSet<_> = result.findings.iter().map(|finding| &finding.sig).collect();
        assert_eq!(signatures.len(), result.findings.len());
    }
}

#[cfg(test)]
mod remote_barrier_selection_tests {
    use std::path::Path;
    use std::sync::Arc;

    use ptxv_case::Case;
    use ptxv_ir::Front;
    use ptxv_ir::Op;
    use serde_json::json;

    use super::*;

    fn machine() -> Machine {
        let case = Case::load(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/e_rmsnorm_cluster_drain"),
        )
        .unwrap();
        let mut text = case.variant_ptx("good").unwrap();
        for (from, to, count) in [
            ("%r14, 2;", "%r14, 1;", 1),
            (
                "mbarrier.try_wait.parity.shared.b64",
                "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64",
                4,
            ),
            (
                "mbarrier.arrive.shared::cluster.b64",
                "mbarrier.arrive.release.cluster.shared::cluster.b64",
                1,
            ),
            (
                "mbarrier.arrive.expect_tx.shared.b64",
                "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64",
                1,
            ),
        ] {
            assert_eq!(
                text.matches(from).count(),
                count,
                "remote selector fixture changed: {from}"
            );
            text = text.replace(from, to);
        }
        let kernel = ptxv_ptx::parse_kernel(&text, Some(&case.m.entry)).unwrap();
        let launch = crate::build_launch(&case, &kernel).unwrap();
        Machine::new(Arc::new(kernel), Arc::new(launch))
    }

    fn snapshot(verdict: &Verdict) -> serde_json::Value {
        json!({"states":verdict.states,"transitions":verdict.transitions,
            "terminal_states":verdict.terminal_states,
            "complete":verdict.complete,"outputs":verdict.outputs,
            "distinct_outputs":verdict.distinct_outputs,"findings":verdict.findings,
            "assumptions":verdict.assumptions,"notes":verdict.notes})
    }

    fn remove_selection_note(verdict: &mut Verdict, cap: usize) {
        assert_eq!(
            verdict.notes.pop().unwrap(),
            format!(
                "Stateful explorer selected for PTX cluster st.async/mbarrier code without atomics (state cap {cap})"
            )
        );
    }

    #[test]
    fn selection_requires_the_remote_barrier_shape_and_excludes_every_atomic() {
        let initial = machine();
        assert!(select_stateful_remote_barriers(&initial));
        assert!(!ptxv_ir::atomic_cancel::certificate_candidate(
            initial.k.front,
            &initial.k.code
        ));
        let mut wrong_front = initial.clone();
        Arc::make_mut(&mut wrong_front.k).front = Front::Sass;
        assert!(!select_stateful_remote_barriers(&wrong_front));
        for remove_store in [false, true] {
            let mut missing = initial.clone();
            for inst in &mut Arc::make_mut(&mut missing.k).code {
                if matches!(inst.op, Op::StAsync { .. }) && remove_store
                    || matches!(inst.op, Op::MbarTryWait { .. }) && !remove_store
                {
                    inst.op = Op::Nop;
                }
            }
            assert!(!select_stateful_remote_barriers(&missing));
        }
        let mut no_cluster = initial.clone();
        Arc::make_mut(&mut no_cluster.launch).grids[0].cluster = [1, 1, 1];
        assert!(!select_stateful_remote_barriers(&no_cluster));
        let mut large_cluster = initial.clone();
        Arc::make_mut(&mut large_cluster.launch).grids[0].cluster = [65_536, 65_536, 1];
        assert!(select_stateful_remote_barriers(&large_cluster));
        let mut multi_grid = initial.clone();
        let grid = multi_grid.launch.grids[0].clone();
        Arc::make_mut(&mut multi_grid.launch).grids.push(grid);
        assert!(!select_stateful_remote_barriers(&multi_grid));
        let case = Case::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../probes/generic8"))
            .unwrap();
        let text = case.variant_ptx("good").unwrap();
        let kernel = ptxv_ptx::parse_kernel(&text, Some(&case.m.entry)).unwrap();
        for atom in kernel
            .code
            .iter()
            .filter(|inst| matches!(inst.op, Op::Atom { .. }))
        {
            let mut with_atomic = initial.clone();
            Arc::make_mut(&mut with_atomic.k).code[0].op = atom.op.clone();
            assert!(!select_stateful_remote_barriers(&with_atomic));
        }
    }

    #[test]
    fn selection_preserves_direct_stateful_observations_and_assumptions() {
        let initial = machine();
        let direct = crate::explore::explore(initial.clone(), 2_000_000);
        let mut selected = explore_dpor(initial, 2_000_000);
        assert!(direct.complete);
        remove_selection_note(&mut selected, 2_000_000);
        assert_eq!(snapshot(&selected), snapshot(&direct));
    }

    #[test]
    fn selection_preserves_initial_findings_and_low_budget_incompleteness() {
        let mut terminal = machine();
        for thread in terminal.threads.iter_mut() {
            thread.make_mut().set_status(St::Exited);
        }
        terminal.report(
            "violation",
            "selector-initial".into(),
            "pre-existing finding".into(),
        );
        let direct = crate::explore::explore(terminal.clone(), 16);
        let mut selected = explore_dpor(terminal, 16);
        assert!(direct
            .findings
            .iter()
            .any(|finding| finding.sig == "selector-initial"));
        remove_selection_note(&mut selected, 16);
        assert_eq!(snapshot(&selected), snapshot(&direct));
        let initial = machine();
        let direct = crate::explore::explore(initial.clone(), 1);
        let mut selected = explore_dpor(initial, 1);
        assert!(!direct.complete);
        assert!(!selected.complete);
        remove_selection_note(&mut selected, 1);
        assert_eq!(snapshot(&selected), snapshot(&direct));
    }

    #[test]
    fn inner_dpor_backtracks_a_nonhead_consumed_op() {
        let case =
            Case::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/e_dpor_nonhead"))
                .unwrap();
        let text = case.variant_ptx("NH").unwrap();
        let kernel = ptxv_ptx::parse_kernel(&text, Some(&case.m.entry)).unwrap();
        let launch = crate::build_launch(&case.for_variant("NH").unwrap(), &kernel).unwrap();
        let initial = Machine::new(Arc::new(kernel), Arc::new(launch));
        assert!(select_stateful_remote_barriers(&initial));
        let before = NONHEAD_BACKTRACKS.load(std::sync::atomic::Ordering::Relaxed);
        let verdict = explore_dpor_one(initial, 2_000_000);
        let after = NONHEAD_BACKTRACKS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            after > before,
            "NH/ptx: no inner non-head backtrack; states={}, complete={}, notes={:?}",
            verdict.states,
            verdict.complete,
            verdict.notes
        );
    }
}

#[cfg(test)]
mod tmem_selection_tests {
    use std::sync::Arc;

    use ptxv_ir::Addr;
    use ptxv_ir::AtomOp;
    use ptxv_ir::File;
    use ptxv_ir::Front;
    use ptxv_ir::MemSync;
    use ptxv_ir::Op;
    use ptxv_ir::Reg;
    use ptxv_ir::Space;
    use ptxv_ir::Val;
    use serde_json::json;

    use super::*;

    fn machine(front: Front) -> Machine {
        let mut machine = crate::dpor_uniform_tests::machine("UMOV UR0, 0x0", "NOP", &["NOP"], 4);
        // Unreachable markers exercise admission independently of the parsed FA4 fixture.
        let code = &mut Arc::make_mut(&mut machine.k).code;
        let reg = Reg {
            file: File::R,
            idx: 0,
        };
        let mut add = |op| {
            let mut inst = code.last().unwrap().clone();
            inst.op = op;
            code.push(inst);
        };
        add(Op::TmemAllocRaw {
            d: reg,
            ok: reg,
            units: Val::Imm(1),
            pair: false,
        });
        add(Op::TmemFreeRaw { mask: Val::Imm(0) });
        add(Op::TmemLd {
            d: reg,
            n: 1,
            taddr: Val::Imm(0),
            off: 0,
            tracked: false,
            s16x64b: false,
            half: None,
            s16x256b: false,
            s16x128b: false,
            red: None,
            regs: None,
        });
        add(Op::TmemSt {
            taddr: Val::Imm(0),
            off: 0,
            v: vec![Val::Imm(0)],
            tracked: false,
            half: None,
            s16x256b: false,
            s16x128b: false,
        });
        add(Op::MbarTryWait {
            addr: Addr {
                base: None,
                base2: None,
                off: 0,
            },
            parity: Val::Imm(0),
            bit31: false,
            d: reg,
            scope: ptxv_ir::Scope::Cluster,
            acquire: true,
        });
        if front == Front::Ptx {
            for inst in code {
                inst.op = match &inst.op {
                    Op::TmemAllocRaw { .. } => Op::TmemAlloc {
                        dst: Addr {
                            base: None,
                            base2: None,
                            off: 0,
                        },
                        ncols: Val::Imm(32),
                    },
                    Op::TmemFreeRaw { .. } => Op::TmemDealloc {
                        taddr: Val::Imm(0),
                        ncols: Val::Imm(32),
                    },
                    _ => inst.op.clone(),
                };
            }
            Arc::make_mut(&mut machine.k).front = front;
        }
        machine
    }

    fn atom(op: AtomOp) -> Op {
        Op::Atom {
            op,
            space: Space::Shared,
            bytes: 4,
            addr: Addr {
                base: None,
                base2: None,
                off: 0,
            },
            v: Val::Imm(0),
            d: None,
            p: None,
            new: None,
            sync: MemSync {
                sem: ptxv_ir::Sem::Relaxed,
                scope: ptxv_ir::Scope::Cta,
            },
        }
    }

    fn snapshot(verdict: &Verdict) -> serde_json::Value {
        json!({"states":verdict.states,"transitions":verdict.transitions,
            "terminal_states":verdict.terminal_states,
            "complete":verdict.complete,"outputs":verdict.outputs,
            "distinct_outputs":verdict.distinct_outputs,"findings":verdict.findings,
            "assumptions":verdict.assumptions,"notes":verdict.notes})
    }

    fn remove_note(verdict: &mut Verdict, cap: usize, front: &str) {
        assert_eq!(
            verdict.notes.pop().unwrap(),
            format!(
                "Stateful explorer selected for single-CTA {front} TMEM lifecycle/mbarrier code with only returnless shared32 AND/OR atomics (state cap {cap})"
            )
        );
    }

    #[test]
    fn selection_requires_lifecycle_single_cta_and_returnless_shared_bitwise_atoms() {
        for front in [Front::Ptx, Front::Sass] {
            let initial = machine(front);
            assert!(select_stateful_tmem(&initial));
            let mut wrong = initial.clone();
            Arc::make_mut(&mut wrong.k).front = if front == Front::Ptx {
                Front::Sass
            } else {
                Front::Ptx
            };
            assert!(!select_stateful_tmem(&wrong));
            let mut wrong = initial.clone();
            Arc::make_mut(&mut wrong.launch).grids[0].grid = [2, 1, 1];
            assert!(!select_stateful_tmem(&wrong));
            let mut wrong = initial.clone();
            Arc::make_mut(&mut wrong.launch).grids[0].cluster = [2, 1, 1];
            assert!(!select_stateful_tmem(&wrong));
            let mut wrong = initial.clone();
            let grid = wrong.launch.grids[0].clone();
            Arc::make_mut(&mut wrong.launch).grids.push(grid);
            assert!(!select_stateful_tmem(&wrong));
            for pc in initial.k.code.iter().enumerate().filter_map(|(pc, inst)| {
                matches!(
                    inst.op,
                    Op::TmemAllocRaw { .. }
                        | Op::TmemFreeRaw { .. }
                        | Op::TmemAlloc { .. }
                        | Op::TmemDealloc { .. }
                        | Op::TmemLd { .. }
                        | Op::TmemSt { .. }
                        | Op::MbarTryWait { .. }
                )
                .then_some(pc)
            }) {
                let mut wrong = initial.clone();
                Arc::make_mut(&mut wrong.k).code[pc].op = Op::Nop;
                assert!(!select_stateful_tmem(&wrong));
            }
            for operation in [AtomOp::And, AtomOp::Or] {
                let mut good = initial.clone();
                let mut inst = good.k.code[0].clone();
                inst.op = atom(operation);
                Arc::make_mut(&mut good.k).code.push(inst);
                assert!(select_stateful_tmem(&good));
            }
            for operation in [
                AtomOp::Add,
                AtomOp::Cas,
                AtomOp::CasSpin,
                AtomOp::Exch,
                AtomOp::FAddFtz,
            ] {
                let mut wrong = initial.clone();
                let mut inst = wrong.k.code[0].clone();
                inst.op = atom(operation);
                Arc::make_mut(&mut wrong.k).code.push(inst);
                assert!(!select_stateful_tmem(&wrong));
            }
            let reg = Reg {
                file: File::R,
                idx: 0,
            };
            for field in 0..6 {
                let mut op = atom(AtomOp::Or);
                if let Op::Atom {
                    space,
                    bytes,
                    d,
                    p,
                    new,
                    ..
                } = &mut op
                {
                    match field {
                        0 => *space = Space::Global,
                        1 => *space = Space::Generic,
                        2 => *bytes = 8,
                        3 => *d = Some(reg),
                        4 => *p = Some(reg),
                        _ => *new = Some(Val::Imm(1)),
                    }
                }
                let mut wrong = initial.clone();
                let mut inst = wrong.k.code[0].clone();
                inst.op = op;
                Arc::make_mut(&mut wrong.k).code.push(inst);
                assert!(!select_stateful_tmem(&wrong));
            }
        }
    }

    #[test]
    fn selection_preserves_full_stateful_result_and_initial_findings() {
        for front in [Front::Ptx, Front::Sass] {
            let mut initial = machine(front);
            initial.report(
                "violation",
                "tmem-selection-initial".into(),
                "pre-existing finding".into(),
            );
            let direct = crate::explore::explore(initial.clone(), 1000);
            let mut selected = explore_dpor(initial, 1000);
            assert!(direct.complete);
            assert!(direct
                .findings
                .iter()
                .any(|f| f.sig == "tmem-selection-initial"));
            remove_note(
                &mut selected,
                1000,
                if front == Front::Ptx { "PTX" } else { "SASS" },
            );
            assert_eq!(snapshot(&selected), snapshot(&direct));
        }
    }

    #[test]
    fn selection_keeps_the_same_low_cap_incomplete() {
        for front in [Front::Ptx, Front::Sass] {
            let initial = machine(front);
            let direct = crate::explore::explore(initial.clone(), 1);
            let mut selected = explore_dpor(initial, 1);
            assert!(!direct.complete);
            assert!(!selected.complete);
            remove_note(
                &mut selected,
                1,
                if front == Front::Ptx { "PTX" } else { "SASS" },
            );
            assert_eq!(snapshot(&selected), snapshot(&direct));
        }
    }
}

#[cfg(test)]
mod private_continuation_selection_tests {
    use super::*;
    use std::sync::Arc;

    fn machine(tail: &str) -> Machine {
        let source = format!(".version 8.8\n.target sm_103a\n.address_size 64\n.visible .entry k(.param .u64 input) {{ .reg .b32 %r0; .reg .b64 %a; .reg .pred %p0; ld.param.u64 %a,[input]; atom.global.add.u32 %r0,[%a],1; {tail} ret; }}");
        let kernel = ptxv_ptx::parse_kernel(&source, Some("k")).unwrap();
        let case = ptxv_case::Case {
            dir: Default::default(),
            m: serde_json::from_value(serde_json::json!({
                "name":"private-continuations", "description":"direct stateless cross-check",
                "entry":"k", "grid":[1,1,1], "block":[4,1,1], "outputs":["x"],
                "buffers":[{"name":"x", "dtype":"u32", "shape":[4], "init":"zeros"}],
                "params":[{"kind":"ptr", "buffer":"x"}], "variants":[]
            }))
            .unwrap(),
        };
        let launch = crate::build_launch(&case, &kernel).unwrap();
        Machine::new(Arc::new(kernel), Arc::new(launch))
    }

    #[test]
    fn selected_search_matches_unselected_stateless_outputs_and_races() {
        for tail in [
            "setp.ne.u32 %p0,%r0,3; @%p0 bra DONE; st.global.u32 [%a+8],17; DONE:",
            "st.global.u32 [%a+8],%r0;",
        ] {
            let initial = machine(tail);
            assert!(initial.private_register_projection().is_some());
            let selected = explore_dpor(initial.clone(), 100_000);
            let raw = explore_dpor_one(initial, 100_000);
            assert!(selected.complete && raw.complete);
            assert!(selected
                .notes
                .iter()
                .any(|note| note.contains("private continuations")));
            assert_eq!(selected.classes(), raw.classes());
            let mut selected_outputs = selected.outputs;
            let mut raw_outputs = raw.outputs;
            selected_outputs.sort();
            raw_outputs.sort();
            assert_eq!(selected_outputs, raw_outputs);
            let signatures = |findings: Vec<Finding>| {
                let mut signatures: Vec<_> = findings
                    .into_iter()
                    .map(|finding| (finding.kind, finding.sig))
                    .collect();
                signatures.sort();
                signatures
            };
            assert_eq!(signatures(selected.findings), signatures(raw.findings));
        }
    }

    #[test]
    fn selection_keeps_the_cap_and_requires_the_pinned_plain_machine() {
        let initial = machine("st.global.u32 [%a+8],%r0;");
        let capped = explore_dpor(initial.clone(), 1);
        assert!(!capped.complete);
        assert!(capped
            .notes
            .iter()
            .any(|note| note.contains("state cap 1)")));
        let mut replaced = initial.clone();
        replaced.k = Arc::new((*replaced.k).clone());
        assert!(replaced.private_register_projection().is_none());
        let unselected = explore_dpor(replaced, 100_000);
        assert!(unselected.complete);
        assert!(!unselected
            .notes
            .iter()
            .any(|note| note.contains("private continuations")));
        let mut recorded = initial;
        recorded.set_footprint(Some(Default::default()));
        assert!(recorded.private_register_projection().is_none());
    }
}

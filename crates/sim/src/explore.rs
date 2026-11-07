/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Exhaustive exploration of all schedules (TLC-style), plus a deterministic single run.
//!
//! Actions are (a) running one enabled warp-split to its next visible instruction and
//! (b) completing one pending async operation. States are cached by a 128-bit fingerprint
//! of everything that affects the future, so the search is exhaustive *for the given launch*
//! while visiting each distinct state once.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::hash::BuildHasherDefault;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use serde::Serialize;

use crate::machine::*;

#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// `leader`: for a split that starts at `elect.sync` on the PTX front-end, which active
    /// thread is elected (the PTX contract allows any; GB300 picks the lowest lane).
    Split {
        warp: u32,
        pc: u32,
        leader: Option<usize>,
    },
    Async(usize),
    /// Launch a CTA that has not launched (E-CLC).
    Launch(u16),
    /// Complete try_cancel response `.0`, cancelling CTA `.1` (None: the request fails).
    Clc(usize, Option<u16>),
}

struct PathNode {
    parent: Option<Arc<PathNode>>,
    desc: String,
    /// The action taken (multi-node work stealing sends schedules, `DistRound`).
    act: Action,
    /// Steps from the initial state (the progress line's deepest schedule).
    depth: u32,
}

fn path_depth(p: &Option<Arc<PathNode>>) -> u32 {
    p.as_ref().map_or(0, |n| n.depth)
}

// Unlink iteratively: paths are as long as the schedule, too deep to drop recursively.
impl Drop for PathNode {
    fn drop(&mut self) {
        let mut cur = self.parent.take();
        while let Some(n) = cur {
            match Arc::try_unwrap(n) {
                Ok(mut n) => cur = n.parent.take(),
                Err(_) => break,
            }
        }
    }
}

fn path_vec(p: &Option<Arc<PathNode>>) -> Vec<String> {
    let mut v = Vec::new();
    let mut cur = p.clone();
    while let Some(n) = cur {
        v.push(n.desc.clone());
        cur = n.parent.clone();
    }
    v.reverse();
    v
}

fn lanes(m: &Machine, ts: &[usize]) -> String {
    let mut ids: Vec<u32> = ts.iter().map(|&t| m.threads[t].tid[0]).collect();
    ids.sort();
    let mut parts = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        let s = ids[i];
        let mut e = s;
        while i + 1 < ids.len() && ids[i + 1] == e + 1 {
            i += 1;
            e = ids[i];
        }
        parts.push(if s == e {
            format!("t{s}")
        } else {
            format!("t{s}-{e}")
        });
        i += 1;
    }
    parts.join(",")
}

impl Machine {
    pub fn actions(&self) -> Vec<(Action, String)> {
        let mut out = Vec::new();
        for (w, pc, members) in self.splits() {
            if self.split_enabled(&members, pc) {
                let desc = format!(
                    "warp {w} [{}] runs from {}",
                    lanes(self, &members),
                    self.k.describe(pc as usize)
                );
                let inst = &self.k.code[pc as usize];
                // ASSUME(E-TMEM-COTENANT): the allocation lands at any free column run.
                if let Some(cands) = self.tenant_alloc_choices(&members, pc) {
                    for p in cands {
                        out.push((
                            Action::Split {
                                warp: w,
                                pc,
                                leader: Some(p as usize),
                            },
                            format!("{desc}, allocating at column {}", p * 32),
                        ));
                    }
                    continue;
                }
                // ASSUME(E-ELECT): PTX elects any active thread; SASS (hardware) the lowest lane.
                if matches!(inst.op, ptxv_ir::Op::Elect { .. })
                    && self.k.front == ptxv_ir::Front::Ptx
                {
                    let active: Vec<usize> = members
                        .iter()
                        .copied()
                        .filter(|&t| self.prd(t, inst.guard))
                        .collect();
                    if active.len() > 1 {
                        // ASSUME(E-ELECT): the leader is a fixed function of the active set
                        // (PTX ISA: "deterministically"); `explore` runs once with the lowest
                        // and once with the highest active lane (PTXV_ELECT=all: every lane at
                        // every elect).
                        let reps: Vec<usize> = if elect_all() {
                            active.clone()
                        } else if self.elect_high {
                            vec![*active.last().unwrap()]
                        } else {
                            vec![active[0]]
                        };
                        for &t in &reps {
                            out.push((
                                Action::Split {
                                    warp: w,
                                    pc,
                                    leader: Some(t),
                                },
                                format!("{desc}, electing t{}", self.threads[t].tid[0]),
                            ));
                        }
                        continue;
                    }
                }
                out.push((
                    Action::Split {
                        warp: w,
                        pc,
                        leader: None,
                    },
                    desc,
                ));
            }
        }
        // ASSUME(E-CLC): unlaunched CTAs launch in any order, and a try_cancel response either
        // fails or cancels any CTA that has not launched (GB300: probes/clc).
        // ASSUME(E-PDL): a later grid's CTAs launch only once the previous grid allows it.
        let mut unlaunched: Vec<u16> = self
            .threads
            .iter()
            .filter(|t| t.st == St::Unlaunched)
            .map(|t| t.cta)
            .collect();
        unlaunched.dedup();
        // Gates of the grids with unlaunched CTAs only (each costs a scan of a grid).
        let mut gates: Vec<Option<bool>> = vec![None; self.launch.grids.len()];
        // ASSUME(E-RESIDENCY): under limited residency a cluster launches as a unit, when its
        // GPU has room for it.
        let counts = (self.launch.env.residency.is_some() && !unlaunched.is_empty())
            .then(|| self.residency_counts());
        for &c in &unlaunched {
            let g = self.launch.grid_index(c as u32);
            if *gates[g].get_or_insert_with(|| self.grid_gate_open(g))
                && counts.as_ref().is_none_or(|n| self.residency_admits(n, c))
            {
                out.push((Action::Launch(c), format!("{} launches", self.cta_name(c))));
            }
        }
        for (i, op) in self.pending.iter().enumerate() {
            if let AsyncKind::Clc { .. } = op.kind {
                let issued = self.k.describe(op.pc as usize);
                out.push((
                    Action::Clc(i, None),
                    format!("try_cancel issued at {issued} fails"),
                ));
                let gi = self
                    .launch
                    .grid_index(self.threads[op.issuer as usize].cta as u32);
                // A cluster launch cancels whole clusters: the victim is a cluster's rank-0 CTA
                // whose cluster is entirely unlaunched (probes/clc_mcast). ASSUME(E-CLC)
                for &c in unlaunched.iter().filter(|&&c| {
                    self.launch.grid_index(c as u32) == gi
                        && self.launch.cluster_of(c as u32).1 == 0
                        && (0..self.launch.cluster_size(c as u32)).all(|r| {
                            self.launch
                                .cta_of_rank(c as u32, r)
                                .is_some_and(|u| unlaunched.contains(&(u as u16)))
                        })
                }) {
                    out.push((
                        Action::Clc(i, Some(c)),
                        format!("try_cancel issued at {issued} cancels {}", self.cta_name(c)),
                    ));
                }
                continue;
            }
            // In-order pipes (tcgen05): only the oldest op of each pipe can complete.
            if let Some(f) = op.fifo {
                if self.pending[..i].iter().any(|o| o.fifo == Some(f)) {
                    continue;
                }
            }
            let what = match &op.kind {
                AsyncKind::Tma { .. } => format!("{} completes", self.actor_name(op.actor)),
                AsyncKind::TmaStore { .. } => {
                    format!("{} (store) completes", self.actor_name(op.actor))
                }
                AsyncKind::Mma { .. } => format!(
                    "tcgen05.mma issued at {} completes",
                    self.k.describe(op.pc as usize)
                ),
                AsyncKind::Cp { .. } => format!(
                    "tcgen05.cp issued at {} completes",
                    self.k.describe(op.pc as usize)
                ),
                AsyncKind::Commit { .. } => format!(
                    "tcgen05.commit issued at {} arrives",
                    self.k.describe(op.pc as usize)
                ),
                AsyncKind::CpArrive { .. } => format!(
                    "cp.async.mbarrier.arrive issued at {} completes",
                    self.k.describe(op.pc as usize)
                ),
                AsyncKind::Remote { cta, .. } => format!(
                    "remote operation issued at {} by {} lands in CTA {cta}",
                    self.k.describe(op.pc as usize),
                    self.actor_name(op.issuer)
                ),
                AsyncKind::Clc { .. } => unreachable!(),
            };
            out.push((Action::Async(i), what));
        }
        out
    }

    pub fn apply(&mut self, a: &Action) {
        match a {
            Action::Split { warp, pc, leader } => {
                self.elect_choice = *leader;
                self.step_split(*warp, *pc);
                self.elect_choice = None;
            }
            Action::Async(i) => self.complete_async(*i),
            Action::Launch(c) => self.launch_unit(*c),
            Action::Clc(i, v) => self.complete_clc(*i, *v),
        }
    }

    pub fn terminal(&self) -> bool {
        self.pending.is_empty() && self.live_threads().all(|(_, t)| t.st == St::Exited)
    }

    /// Describe why nothing can move.
    pub fn deadlock_report(&self) -> (String, String) {
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, t) in self.live_threads() {
            if t.st == St::Exited {
                continue;
            }
            let why = match t.st {
                St::Bar => format!("waiting at barrier {}", self.k.describe(t.pc as usize)),
                St::ClusterWait => format!(
                    "waiting at barrier.cluster {}",
                    self.k.describe(t.pc as usize)
                ),
                St::Unlaunched if self.residency_why(t.cta).is_some() => {
                    self.residency_why(t.cta).unwrap()
                }
                St::Unlaunched => {
                    let gi = self.launch.grid_index(t.cta as u32);
                    format!(
                        "{} never launches: kernel {gi} waits for kernel {} to {}",
                        self.cta_name(t.cta),
                        self.launch.prev_grid(gi).unwrap_or(0),
                        if self.launch.grids[gi].pdl {
                            "trigger (griddepcontrol.launch_dependents) or exit in every CTA"
                        } else {
                            "complete"
                        }
                    )
                }
                _ => format!("blocked at {}", self.k.describe(t.pc as usize)),
            };
            groups.entry(why).or_default().push(i);
        }
        let resid = self.launch.env.residency.is_some()
            && self
                .live_threads()
                .any(|(_, t)| t.st == St::Unlaunched && self.residency_why(t.cta).is_some());
        let mut msg = String::from(if resid {
            "residency deadlock (E-RESIDENCY): every resident CTA is blocked and a CTA has no slot.\n"
        } else {
            "deadlock: no thread or async unit can make progress.\n"
        });
        let mut sig = String::from(if resid {
            "deadlock:residency"
        } else {
            "deadlock"
        });
        for (why, ts) in &groups {
            msg += &format!("    [{}] {why}\n", lanes(self, ts));
            sig += &format!(":{}", self.threads[ts[0]].pc);
        }
        for ((cta, off), mb) in &self.mbars {
            msg += &format!(
                "    mbarrier cta{cta}@smem{off:#x}: phase {} pending arrivals {}/{} tx-count {} bytes\n",
                mb.phase, mb.pending, mb.expected, mb.tx
            );
        }
        (sig, msg)
    }
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct Verdict {
    pub states: usize,
    pub transitions: usize,
    pub complete: bool,
    pub terminal_states: usize,
    /// Distinct output byte-vectors over all terminal states (1 = deterministic).
    pub distinct_outputs: usize,
    pub findings: Vec<Finding>,
    /// ASSUMPTIONS.md IDs this verdict depends on.
    pub assumptions: Vec<String>,
    /// How the verdict was reached, when not by one plain exploration (slices, symmetry).
    pub notes: Vec<String>,
    pub seconds: f64,
    #[serde(skip)]
    pub outputs: Vec<Vec<u8>>,
}

impl Verdict {
    pub fn classes(&self) -> Vec<String> {
        // Havoc probes (kind "probe", the two-level check's level 2: l1.rs, twolevel.rs) are
        // witnesses for classification, not findings of the check; `unordered` root-cause
        // pairs (wellsync.rs `add_root_pairs`) explain a race / nondeterminism found anyway;
        // notes (kind "note", S-SAME-VALUE) are not findings either.
        let mut c: Vec<String> = self
            .findings
            .iter()
            .filter(|f| f.kind != "probe" && f.kind != "unordered" && f.kind != "note")
            .map(|f| f.kind.clone())
            .collect();
        if self.distinct_outputs > 1 {
            c.push("nondeterministic".into());
        }
        c.sort();
        c.dedup();
        if c.is_empty() {
            c.push("clean".into());
        }
        c
    }

    /// Does the verdict meet the expectation? A complete exploration must produce exactly the
    /// expected classes. An incomplete one never proves "clean"; for a bug variant it passes
    /// once every expected class was found (later classes could still appear).
    pub fn meets(&self, expect: &[String]) -> bool {
        let got = self.classes();
        if self.complete {
            got == expect
        } else {
            !expect.iter().any(|e| e == "clean") && expect.iter().all(|e| got.contains(e))
        }
    }
}

/// Schedule-independent findings of the kernel (SASS scoreboard hazards), as findings.
pub(crate) fn static_findings(m: &Machine) -> Vec<Finding> {
    m.k.static_findings
        .iter()
        .map(|f| Finding {
            kind: f.kind.clone(),
            sig: f.sig.clone(),
            msg: f.msg.clone(),
            trace: f
                .pcs
                .iter()
                .take(2)
                .map(|&pc| format!("  static: {}", m.k.code[pc].text))
                .collect(),
            ..Default::default()
        })
        .collect()
}

fn elect_all() -> bool {
    std::env::var("PTXV_ELECT").as_deref() == Ok("all")
}

/// Worker threads for the stateful explorer: PTXV_THREADS, else every core, but at most one
/// per 10k states of the cap. Under a small cap many workers search breadth-first in effect
/// and may stop before any of them reaches a terminal state; a bounded search for bugs wants
/// depth, as the serial search gives it.
pub fn threads(max_states: usize) -> usize {
    std::env::var("PTXV_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or_else(|| {
            let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
            cores.min(max_states / 10_000).max(1)
        })
}

/// Stack size of explorer threads (instruction semantics recurse a little; be generous).
const STACK: usize = 64 << 20;

/// ASSUME(E-ELECT): run `f` once per leader policy (lowest / highest active lane) when the
/// PTX kernel elects, and merge the verdicts. `f` gets a thread budget: with more than one
/// thread the two (independent) searches run side by side, splitting it.
pub fn per_leader_policy(
    m0: Machine,
    threads: usize,
    f: impl Fn(Machine, usize) -> Verdict + Sync,
) -> Verdict {
    let elects = m0.k.front == ptxv_ir::Front::Ptx
        && !elect_all()
        && m0
            .k
            .code
            .iter()
            .any(|i| matches!(i.op, ptxv_ir::Op::Elect { .. }));
    if !elects {
        return f(m0, threads);
    }
    let mut hi = m0.clone();
    hi.elect_high = true;
    // Multi-node: searches are collectives, run one at a time (in the same order everywhere).
    let multi = crate::dist::get().is_some();
    let (mut v, w) = if threads > 1 && !multi {
        std::thread::scope(|s| {
            let h = std::thread::Builder::new()
                .stack_size(STACK)
                .spawn_scoped(s, || f(hi, threads / 2))
                .expect("spawn explorer thread");
            let v = f(m0, threads - threads / 2);
            (v, h.join().unwrap())
        })
    } else {
        let t = if multi { threads } else { 1 };
        (f(m0, t), f(hi, t))
    };
    v.states += w.states;
    v.transitions += w.transitions;
    v.complete &= w.complete;
    v.terminal_states += w.terminal_states;
    v.seconds = if threads > 1 && !multi {
        v.seconds.max(w.seconds)
    } else {
        v.seconds + w.seconds
    };
    for mut x in w.findings {
        if !v.findings.iter().any(|y| y.sig == x.sig) {
            x.msg = format!("{} (leader policy: highest active lane)", x.msg);
            v.findings.push(x);
        }
    }
    for o in w.outputs {
        if !v.outputs.contains(&o) {
            v.outputs.push(o);
        }
    }
    v.distinct_outputs = v.outputs.len();
    v
}

/// ASSUME(E-CTA-INDEP): can the clusters (single CTAs without clusters) of this launch only
/// interact through plain global memory? Then exploring one cluster at a time (`cta_ample`)
/// is exact. Cluster-scope operations (DSMEM, remote arrivals, multicast, cta_group::2) only
/// name CTAs of the issuer's own cluster; try_cancel is allowed when it cannot execute
/// (`clc_eager`: executing it is reported).
pub fn ctas_independent(m: &Machine) -> bool {
    use ptxv_ir::Op;
    use ptxv_ir::Space;
    // ASSUME(E-TMEM-COTENANT): CTAs sharing an SM share its tensor memory allocator.
    if std::env::var("PTXV_NO_CTA_POR").is_ok()
        || !m.launch.env.tmem_share.is_empty()
        || m.launch.grids.len() != 1
        || m.threads.iter().any(|t| t.st == St::Unlaunched)
    {
        return false;
    }
    !m.k.code.iter().any(|i| match &i.op {
        Op::Atom { space, .. } => *space != Space::Shared,
        Op::Ld { space, sync, .. } | Op::St { space, sync, .. } => {
            sync.is_some() && !matches!(space, Space::Shared | Space::SharedCluster)
        }
        Op::Ld256 { sync, .. } | Op::St256 { sync, .. } => sync.is_some(),
        Op::ClcTryCancel { .. } | Op::ClcQuery { .. } => !m.clc_eager,
        // griddepcontrol is a no-op in a single-grid launch (E-PDL).
        _ => false,
    })
}

/// `m0` with every CTA launched up front (`clc_eager` without its trial run), when eager
/// launch applies (one grid, no limited residency, unlaunched CTAs, no PTXV_NO_CLC_EAGER).
/// Executing a try_cancel on it is a fault naming "E-CLC eager launch".
pub fn clc_eager_launch(m0: &Machine) -> Option<Machine> {
    if m0.launch.grids.len() != 1
        || m0.launch.env.residency.is_some()
        || std::env::var("PTXV_NO_CLC_EAGER").is_ok()
        || !m0.threads.iter().any(|t| t.st == St::Unlaunched)
    {
        return None;
    }
    let mut m = m0.clone();
    m.clc_eager = true;
    for c in 0..m.launch.ctas() {
        if m.cta_threads(c as u16)
            .any(|t| m.threads[t].st == St::Unlaunched)
        {
            m.launch_cta(c as u16);
        }
    }
    m.findings.clear();
    Some(m)
}

/// ASSUME(E-CLC): a kernel containing try_cancel starts with every CTA unlaunched, which
/// multiplies the schedules by the launch orders. When no try_cancel executes, launch times
/// do not matter (a late launch is a schedule in which the CTA's threads have not stepped
/// yet), so the CTAs can all be launched up front. A deterministic run with eager launch
/// decides whether to try (cuBLAS disables its try_cancel with a launch parameter); the
/// exploration then reports any try_cancel that does execute ("unmodeled"), so a wrong guess
/// cannot hide schedules.
pub fn clc_eager(m0: Machine) -> Machine {
    let Some(m) = clc_eager_launch(&m0) else {
        return m0;
    };
    // The deterministic run of `run_once`, stopped at the first try_cancel (a faulting step
    // may repeat forever) or after a step budget (then: lazy, the safe default).
    let mut probe = m.clone();
    probe.fp = None;
    let mut sched = DetSched::default();
    let mut faults = FaultLoop::default();
    for _ in 0..50_000_000u64 {
        let Some(a) = sched.pick(&probe) else {
            return m;
        };
        probe.apply(&a);
        // A step that faults forever: undecided, so lazy (the safe default).
        if faults.repeats(&probe.findings).is_some() {
            return m0;
        }
        if probe
            .findings
            .drain(..)
            .any(|f| f.msg.contains("E-CLC eager launch"))
        {
            return m0;
        }
    }
    m0
}

fn has_clc(m: &Machine) -> bool {
    m.k.code.iter().any(|i| {
        matches!(
            i.op,
            ptxv_ir::Op::ClcTryCancel { .. } | ptxv_ir::Op::ClcQuery { .. }
        )
    })
}

/// The CTA an action belongs to.
fn action_cta(m: &Machine, a: &Action) -> Option<u16> {
    match a {
        Action::Split { warp, .. } => m.threads.iter().find(|t| t.warp == *warp).map(|t| t.cta),
        Action::Async(i) => Some(m.threads[m.pending[*i].issuer as usize].cta),
        Action::Launch(_) | Action::Clc(..) => None,
    }
}

/// Ample set for independent clusters: only the actions of the lowest cluster (CTA without
/// clusters) that has any. Actions of different clusters commute (they share no
/// synchronization object), and a race on global memory between them is found by vector
/// clocks in either order (E-DRF), so this keeps every terminal state, deadlock and race up
/// to reordering. The state graph is acyclic (every action ticks a clock or retires an async
/// op), so no cycle proviso is needed.
fn cta_ample(m: &Machine, acts: Vec<(Action, String)>) -> Vec<(Action, String)> {
    let ctas: Vec<Option<u32>> = acts
        .iter()
        .map(|(a, _)| action_cta(m, a).map(|c| m.launch.cluster_of(c as u32).0))
        .collect();
    let Some(lo) = ctas.iter().flatten().min().copied() else {
        return acts;
    };
    if ctas.iter().any(|c| c.is_none()) {
        return acts;
    }
    acts.into_iter()
        .zip(ctas)
        .filter(|(_, c)| *c == Some(lo))
        .map(|(a, _)| a)
        .collect()
}

/// Ample sets for actions that commute with every other action (ASSUME(E-DRF)):
/// - a `bar.sync` arrival: it only adds the thread to the barrier and ends the step; the last
///   arrival releases everyone, whatever the order; likewise barrier.cluster arrive and wait;
/// - a whole-warp `cp.async.bulk.wait_group` (enabled only once its stores completed), and
///   a whole-warp exit with none of its async operations pending, and a whole-warp
///   visible operation guarded off in every lane (a no-op);
/// - a split that starts at an invisible instruction (it resumed after a wait, or stopped at
///   a convergence point) when it holds every live thread of its warp: it runs register,
///   plain-memory and async-issue operations only, whose conflicts are races found by vector
///   clocks in either order. (With other splits of the warp live, moving its pc could change
///   which splits merge, so it is not independent of them.)
/// - a whole-warp plain `mbarrier.arrive` that cannot complete the phase;
/// - the completion of a TMA store or of an MMA (data movement only).
///
/// One such action alone is an ample set; the state graph is acyclic, so no cycle proviso.
fn local_ample(m: &Machine, acts: Vec<(Action, String)>) -> Vec<(Action, String)> {
    if acts.len() < 2 || std::env::var("PTXV_NO_LOCAL_POR").is_ok() {
        return acts;
    }
    // ASSUME(E-RESIDENCY): under limited residency, the enabled launches together are an
    // ample set (eager launch). A launch commutes with every thread and async action, and
    // only another launch can disable it (by taking the slot), which is in the set. Launching
    // a CTA later than its slot frees is the same execution with the CTA idle until then.
    if m.launch.env.residency.is_some() && std::env::var("PTXV_NO_LAUNCH_POR").is_err() {
        let n = acts
            .iter()
            .filter(|(a, _)| matches!(a, Action::Launch(_)))
            .count();
        if n > 0 && n < acts.len() && !has_clc(m) {
            return acts
                .into_iter()
                .filter(|(a, _)| matches!(a, Action::Launch(_)))
                .collect();
        }
    }
    let pos = acts.iter().position(|(a, _)| match a {
        Action::Split {
            warp,
            pc,
            leader: None,
        } => {
            let op = &m.k.code[*pc as usize].op;
            // ASSUME(E-CLUSTER-BAR): barrier.cluster arrive / wait commute like bar.sync
            // arrivals: the barrier completes once every live thread arrived, releasing the
            // join of their releases whatever the order; a wait only parks or acquires.
            // bar.arrive does not (E-BAR-ARRIVE): an arrival that does not wait can land in
            // the next generation, so arrival order decides who is released by whom; with
            // any bar.arrive in the kernel, bar.sync arrivals are not commuted here either.
            // Nor at a barrier more threads than its count may still arrive at (E-BAR-OVER):
            // which threads share a generation then depends on arrival order.
            if matches!(
                op,
                ptxv_ir::Op::ClusterArrive { .. } | ptxv_ir::Op::ClusterWait
            ) || matches!(op, ptxv_ir::Op::BarSync { arrive: false, .. })
                && !m
                    .k
                    .code
                    .iter()
                    .any(|i| matches!(i.op, ptxv_ir::Op::BarSync { arrive: true, .. }))
                && !bar_spills(m, *warp, *pc)
            {
                return true;
            }
            // An unguarded exit of part of a warp, none of whose threads has an async
            // operation pending: exited threads never merge with a split again, and an exit
            // commutes with barrier arrivals (all-thread bar.sync, barrier.cluster, warpsync
            // complete once every live thread arrived, in either order).
            if matches!(op, ptxv_ir::Op::Exit) && m.k.code[*pc as usize].guard.is_always() {
                let split: Vec<u32> = m
                    .warp_threads(*warp)
                    .map(|t| t as u32)
                    .filter(|&t| {
                        let th = &m.threads[t as usize];
                        th.warp == *warp && th.pc == *pc && th.st == St::Run
                    })
                    .collect();
                if !m.pending.iter().any(|o| split.contains(&o.issuer)) {
                    return true;
                }
            }
            if !op.is_visible()
                && std::env::var("PTXV_NO_EXIT_BOUND").is_err()
                && exit_bound(m, *warp, *pc)
            {
                return true;
            }
            // ASSUME(E-BSYNC): an arrival at `BSYNC Bb` by lanes all of which execute it, when
            // Bb's recorded mask covers every live lane and every lane already parked waits at
            // this same BSYNC: it parks the lanes (or, as the last arrival, releases every
            // live lane), the same whichever order the lanes arrive in, and touches nothing
            // outside its warp.
            if let ptxv_ir::Op::Bsync { b } = op {
                let guard = m.k.code[*pc as usize].guard;
                let members: Vec<usize> = m
                    .warp_threads(*warp)
                    .filter(|&t| {
                        let th = &m.threads[t];
                        th.warp == *warp && th.pc == *pc && th.st == St::Run
                    })
                    .collect();
                let live = m
                    .warp_threads(*warp)
                    .map(|t| &m.threads[t])
                    .filter(|t| t.warp == *warp && t.st != St::Exited)
                    .fold(0u32, |a, t| a | 1 << t.lane);
                let bm = m.warps[*warp as usize].bmask[*b as usize];
                if std::env::var("PTXV_NO_BSYNC_POR").is_err()
                    && !members.is_empty()
                    && members.iter().all(|&t| m.prd(t, guard))
                    && bm & live == live
                    && m.warp_threads(*warp)
                        .map(|t| &m.threads[t])
                        .all(|t| t.warp != *warp || t.st != St::WarpSync || t.pc == *pc)
                {
                    return true;
                }
            }
            let wt = m.warp_threads(*warp);
            let whole = wt
                .clone()
                .map(|t| &m.threads[t])
                .filter(|t| t.st != St::Exited)
                .all(|t| t.pc == *pc && t.st == St::Run);
            // Not whole, but the rest of the warp is parked at a BSYNC only this split releases.
            let whole_rel = !whole && whole_or_parked(m, *warp, *pc);
            if whole_rel && !op.is_visible() {
                return true;
            }
            let members = || wt.clone().map(|i| i as u32);
            let guard = m.k.code[*pc as usize].guard;
            // A visible operation guarded off in every member is a no-op.
            let off = || members().all(|t| !m.prd(t as usize, guard));
            whole
                && (off()
                    || match op {
                        // Enabled once its own stores completed; nothing can disable it.
                        ptxv_ir::Op::BulkWait { .. } => true,
                        // Exit conflicts only with the completion of the exiting threads' own
                        // async operations (async-after-exit).
                        ptxv_ir::Op::Exit => {
                            !m.pending.iter().any(|o| members().any(|t| t == o.issuer))
                        }
                        ptxv_ir::Op::Elect { .. } => false,
                        // A plain arrival that cannot complete the phase only decrements the
                        // pending count: it commutes with the other arrivals and changes nothing
                        // a try_wait reads.
                        ptxv_ir::Op::MbarArrive {
                            addr,
                            tx: None,
                            remote: false,
                            ..
                        } => {
                            let active: Vec<usize> = members()
                                .map(|t| t as usize)
                                .filter(|&t| m.threads[t].pc == *pc && m.prd(t, guard))
                                .collect();
                            let mut per: std::collections::BTreeMap<(u16, u32), u32> =
                                Default::default();
                            let mut ok = true;
                            for &t in &active {
                                match m
                                    .addr(t, addr)
                                    .and_then(|a| m.resolve(t, ptxv_ir::Space::Shared, a, 8))
                                {
                                    Ok(crate::exec::Loc::Shared { cta, off }) => {
                                        *per.entry((cta, off)).or_insert(0) += 1
                                    }
                                    _ => ok = false,
                                }
                            }
                            ok && per
                                .iter()
                                .all(|(k, n)| m.mbars.get(k).is_some_and(|b| b.pending > *n))
                        }
                        op => !op.is_visible(),
                    })
        }
        // ASSUME(E-PDL, E-CLC): launching a CTA whose gate is open (it is enabled) commutes with
        // every other action, and nothing disables it: the gate only opens (the previous grid
        // completed, or every CTA of it triggered or exited); the launch only sets the CTA's
        // threads running at their entry pc and joins the previous grid's completion into
        // their clocks, which no other thread's step reads (an unlaunched thread counts as
        // live and not arrived at every barrier, like a launched one that has not stepped;
        // its shared memory is alive either way, `check_alive`). Only a try_cancel response
        // reads which CTAs have launched, so not in kernels with CLC.
        // ASSUME(E-RESIDENCY): not under limited residency, where a launch takes a slot that
        // another launch needs.
        Action::Launch(_) => {
            std::env::var("PTXV_NO_LAUNCH_POR").is_err()
                && m.launch.env.residency.is_none()
                && !m.k.code.iter().any(|i| {
                    matches!(
                        i.op,
                        ptxv_ir::Op::ClcTryCancel { .. } | ptxv_ir::Op::ClcQuery { .. }
                    )
                })
        }
        // A TMA store or MMA completion only moves data (shared/global memory, TMEM): its
        // conflicts are races, found in either order; wait_group / commit observe it later.
        Action::Async(i) => matches!(
            m.pending[*i].kind,
            AsyncKind::TmaStore { .. } | AsyncKind::Mma { .. } | AsyncKind::Cp { .. }
        ),
        _ => false,
    });
    match pos {
        Some(i) => acts.into_iter().nth(i).into_iter().collect(),
        None => acts,
    }
}

/// ASSUME(E-BSYNC): lanes of `warp` parked at a reconvergence point (`St::WarpSync`) that
/// only the warp's running lanes can release: every one is parked at the same `BSYNC Bb`
/// whose recorded mask (`Warp::bmask`) covers every live lane of the warp. The release then
/// happens exactly when the last live lane arrives there, whichever order the lanes arrive
/// in, and frees every live lane (`warpsync_release`). `PTXV_NO_BSYNC_POR` turns the
/// reductions that rely on it off.
fn bsync_parked(m: &Machine, warp: u32) -> Option<u32> {
    if std::env::var("PTXV_NO_BSYNC_POR").is_ok() {
        return None;
    }
    let mut at: Option<u32> = None;
    let mut live = 0u32;
    for t in m.warp_threads(warp).map(|t| &m.threads[t]) {
        if t.warp != warp || t.st == St::Exited {
            continue;
        }
        live |= 1 << t.lane;
        if t.st == St::WarpSync && *at.get_or_insert(t.pc) != t.pc {
            return None;
        }
    }
    let p = at?;
    let ptxv_ir::Op::Bsync { b } = m.k.code[p as usize].op else {
        return None;
    };
    let bm = m.warps[warp as usize].bmask[b as usize];
    (bm & live == live).then_some(p)
}

/// Does the split of `warp` at `pc` hold every live thread of its warp that can move: the
/// others (if any) are parked at a reconvergence point only it can release (`bsync_parked`)?
fn whole_or_parked(m: &Machine, warp: u32, pc: u32) -> bool {
    let mut parked = false;
    let mut run = false;
    for t in m.warp_threads(warp).map(|t| &m.threads[t]) {
        if t.warp != warp || t.st == St::Exited {
            continue;
        }
        if t.st == St::Run && t.pc == pc {
            run = true;
        } else if t.st == St::WarpSync {
            parked = true;
        } else {
            return false;
        }
    }
    run && (!parked || bsync_parked(m, warp).is_some())
}

/// Is the split of `warp` at `pc` on its way out: does every path its threads can take from
/// `pc` reach an unguarded exit through invisible instructions only (visible ones guarded off
/// in every member are no-ops), writing no warp-shared (uniform) register, with none of the
/// split's async operations pending? Then running it commutes with every other action, also
/// with the other splits of its warp: a split it would merge with on the way runs the same
/// invisible instructions from there, and merging only regroups invisible steps of threads
/// that exit (the partial-exit case above). Guards are evaluated on the current predicates,
/// so the walk gives up at a guarded instruction once the path wrote a predicate. (cuBLAS's
/// producer drains its stages lane by lane, then all lanes exit: without this, every lane's
/// position on the way out multiplied the states.)
fn exit_bound(m: &Machine, warp: u32, pc: u32) -> bool {
    use ptxv_ir::File;
    use ptxv_ir::Op;
    let members: Vec<usize> = m
        .warp_threads(warp)
        .filter(|&t| {
            let th = &m.threads[t];
            th.warp == warp && th.pc == pc && th.st == St::Run
        })
        .collect();
    if members.is_empty()
        || m.pending
            .iter()
            .any(|o| members.contains(&(o.issuer as usize)))
    {
        return false;
    }
    let code = &m.k.code;
    let (mut at, mut wrote_pred) = (pc as usize, false);
    for _ in 0..64 {
        let Some(inst) = code.get(at) else {
            return false;
        };
        let g = inst.guard;
        // The guard's value in every member: Some(on) when uniform and still current.
        let on = if g.is_always() {
            Some(true)
        } else if wrote_pred {
            None
        } else {
            let v = m.prd(members[0], g);
            members.iter().all(|&t| m.prd(t, g) == v).then_some(v)
        };
        let Some(on) = on else {
            return false;
        };
        if !on {
            at += 1;
            continue;
        }
        match &inst.op {
            Op::Exit => return true,
            Op::Bra {
                target,
                any: false,
                tid0_falls: false,
                cond: None,
                div: None,
            } => {
                at = *target;
                continue;
            }
            // Control transfers we do not follow, and warp-collective operations (they read
            // the other lanes, which other splits move).
            op if op.is_visible()
                || op.is_warp_aligned()
                || matches!(
                    op,
                    Op::Bra { .. }
                        | Op::Call { .. }
                        | Op::Ret { .. }
                        | Op::Brx { .. }
                        | Op::Trap { .. }
                        | Op::VoteAll { .. }
                        | Op::VoteAny { .. }
                        | Op::MatchAny { .. }
                        | Op::Redux { .. }
                ) =>
            {
                return false;
            }
            op => {
                for r in op.dst_regs() {
                    match r.file {
                        File::UR | File::UP => return false,
                        File::P => wrote_pred = true,
                        _ => {}
                    }
                }
            }
        }
        at += 1;
    }
    false
}

/// Who performs an action, for the synchronization reduction (`sync_ample`): a warp, or the
/// asynchronous units working for a warp (its TMA, tcgen05 and bulk operations).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Ent {
    Warp(u32),
    Async(u32),
}

fn action_ent(m: &Machine, a: &Action) -> Option<Ent> {
    match a {
        Action::Split { warp, .. } => Some(Ent::Warp(*warp)),
        Action::Async(i) => Some(Ent::Async(m.threads[m.pending[*i].issuer as usize].warp)),
        Action::Launch(_) | Action::Clc(..) => None,
    }
}

/// Can the synchronization reduction run on this kernel? Not with synchronization whose
/// objects the footprints do not all name: CLC (unless it cannot execute, `clc_eager`), PDL
/// across grids (single-grid launches only), st.async, plain shared::cluster loads/stores (their liveness check is not recorded).
/// Cluster synchronization is named: remote arrivals, multicast TMA and commits touch the
/// target CTA's mbarrier keys; barrier.cluster arrive/wait write the cluster's key (space 13).
/// Exits need no key: an exit commutes with barrier.cluster arrivals (the barrier completes
/// once every live thread arrived, in either order), and a remote access racing a target's
/// exit is reported order-independently at terminal states (S-CLUSTER-LIFE, `exit_checks`).
/// Multi-grid launches (rung 3a: a prepare kernel, then the GEMM) are checked per state
/// (`sync_por_state_ok`): the reduction runs only once every live thread and pending async
/// operation belongs to one grid and nothing is left to launch.
fn sync_por_ok(m: &Machine) -> bool {
    use ptxv_ir::Op;
    use ptxv_ir::Space;
    // ASSUME(E-TMEM-COTENANT): the footprints do not name an SM partner's allocator.
    if std::env::var("PTXV_NO_SYNC_POR").is_ok()
        || !m.launch.env.tmem_share.is_empty()
        || (m.launch.grids.len() != 1 || m.threads.iter().any(|t| t.st == St::Unlaunched))
            && std::env::var("PTXV_NO_GRID_POR").is_ok()
    {
        return false;
    }
    !m.k.code.iter().any(|i| match &i.op {
        Op::ClcTryCancel { .. } | Op::ClcQuery { .. } => !m.clc_eager,
        // griddepcontrol is a no-op in a single-grid launch (E-PDL).
        Op::StAsync { .. } => true,
        Op::Ld { space, .. } | Op::St { space, .. } => *space == Space::SharedCluster,
        _ => false,
    })
}

/// ASSUME(E-PDL): the synchronization reduction in a state of a multi-grid launch. Sound once
/// the launch is down to one grid: no CTA is left to launch (no `Action::Launch`, which has
/// no entity and is not learned), and every live thread and pending async operation belongs
/// to one grid. The earlier grids then completed (a grid launches only after its predecessor
/// completed or triggered, and every thread of the earlier grids has exited with none of its
/// async operations pending), so their launch gates (space 9) can no longer change: the state
/// behaves as a single-grid launch of that grid, whose initial memory is what the earlier
/// grids left (ordered before it by the grid boundary).
fn sync_por_state_ok(m: &Machine) -> bool {
    // ASSUME(E-RESIDENCY): launches under limited residency depend on exits (slots); the
    // reduction does not learn them, so it waits until every CTA has launched.
    if m.launch.env.residency.is_some() && m.live_threads().any(|(_, t)| t.st == St::Unlaunched) {
        return false;
    }
    if m.launch.grids.len() == 1 {
        return true;
    }
    // ASSUME(E-PDL): several grids live at once (grids on different GPUs, launched
    // independently; launch groups). Sound once nothing is left to launch and no live grid
    // can still wait for another one's completion (`griddepcontrol.wait` of a PDL grid whose
    // predecessor runs): the state is then a closed set of CTAs whose only contact is global
    // memory, named in the footprints (strong accesses, spaces 10/11) or a race (E-DRF), as
    // between the CTAs of one grid. `PTXV_NO_MULTIGRID_POR` keeps the one-grid rule.
    if std::env::var("PTXV_NO_MULTIGRID_POR").is_err() {
        let mut live_grids: Vec<usize> = Vec::new();
        for (_, t) in m.live_threads() {
            match t.st {
                St::Exited => {}
                St::Unlaunched => return false,
                _ => {
                    let gi = m.launch.grid_index(t.cta as u32);
                    if live_grids.last() != Some(&gi) && !live_grids.contains(&gi) {
                        live_grids.push(gi);
                    }
                }
            }
        }
        return live_grids.iter().all(|&gi| {
            !m.launch.grids[gi].pdl || m.launch.prev_grid(gi).is_none_or(|p| m.grid_complete(p))
        });
    }
    let mut g: Option<usize> = None;
    let mut same = |cta: u16| {
        let gi = m.launch.grid_index(cta as u32);
        *g.get_or_insert(gi) == gi
    };
    for (_, t) in m.live_threads() {
        match t.st {
            St::Exited => {}
            St::Unlaunched => return false,
            _ => {
                if !same(t.cta) {
                    return false;
                }
            }
        }
    }
    m.pending
        .iter()
        .all(|o| same(m.threads[o.issuer as usize].cta))
}

/// pcs reachable from each pc (itself included) in the control-flow graph, as bitsets.
/// Conservative: a call may return (to the next pc), and a return may go to the pc after any
/// call (ASSUME(E-RET), checked on every executed return; `PTXV_NO_RET_SITES`: anywhere).
/// `cut`: bit 0 stops at barrier.cluster waits, bit 1 at all-thread `bar.sync` (the barrier
/// pc itself is reachable, nothing past it): what a warp can reach before a barrier that
/// still needs another warp's arrival completes.
fn reach_sets(code: &[ptxv_ir::Inst], cut: u8, cfg: &BarCfg) -> Vec<Vec<u64>> {
    let n = code.len();
    let words = n.div_ceil(64);
    let succ = cfg_succ(code, cut, &cfg.fold, &cfg.rets);
    let mut r: Vec<Vec<u64>> = (0..n)
        .map(|pc| {
            let mut v = vec![0u64; words];
            v[pc / 64] |= 1 << (pc % 64);
            v
        })
        .collect();
    loop {
        let mut changed = false;
        for pc in (0..n).rev() {
            for &q in &succ[pc] {
                if q == pc {
                    continue;
                }
                let (a, b) = if pc < q {
                    let (x, y) = r.split_at_mut(q);
                    (&mut x[pc], &y[0])
                } else {
                    let (x, y) = r.split_at_mut(pc);
                    (&mut y[0], &x[q])
                };
                for (wa, wb) in a.iter_mut().zip(b) {
                    let nw = *wa | *wb;
                    if nw != *wa {
                        *wa = nw;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            return r;
        }
    }
}

/// Control-flow successors of each pc (`reach_sets`, same `cut`).
fn cfg_succ(
    code: &[ptxv_ir::Inst],
    cut: u8,
    fold: &[Option<bool>],
    rets: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    use ptxv_ir::Op;
    let n = code.len();
    code.iter()
        .enumerate()
        .map(|(pc, i)| {
            let next = (pc + 1 < n).then_some(pc + 1);
            let always = i.guard.is_always();
            match &i.op {
                Op::ClusterWait if cut & 1 != 0 => vec![],
                Op::BarSync {
                    count: None,
                    arrive: false,
                    ..
                } if cut & 2 != 0 && always => vec![],
                Op::Bra {
                    target,
                    tid0_falls: false,
                    div: None,
                    ..
                } if always => vec![*target],
                Op::Bra { target, .. } if fold.get(pc).copied().flatten().is_some() => {
                    if fold[pc] == Some(true) {
                        vec![*target]
                    } else {
                        next.into_iter().collect()
                    }
                }
                Op::Bra { target, .. } => [Some(*target), next].into_iter().flatten().collect(),
                Op::Call { target } => [Some(*target), next].into_iter().flatten().collect(),
                // (No table: anywhere.)
                Op::Ret { .. } if always => {
                    rets.get(pc).cloned().unwrap_or_else(|| (0..n).collect())
                }
                Op::Ret { .. } => rets
                    .get(pc)
                    .cloned()
                    .unwrap_or_else(|| (0..n).collect())
                    .into_iter()
                    .chain(next)
                    .collect(),
                Op::Brx { targets, .. } => targets.clone(),
                Op::Exit | Op::Trap { .. } if always => vec![],
                _ => next.into_iter().collect(),
            }
        })
        .collect()
}

/// Static control flow for `Machine::bar_oversub` (built once per machine).
#[derive(Clone, Debug, Default)]
pub struct BarCfg {
    /// Per pc, the named barriers (bit = id; all bits for a non-immediate id or an indirect
    /// call) that a thread at that pc may still arrive at (`bar.sync` / `bar.arrive`).
    pub reach: Vec<u32>,
    succ: Vec<Vec<usize>>,
    /// Registers each pc writes.
    dsts: Vec<Vec<ptxv_ir::Reg>>,
    /// Branches decided by launch constants (`const_guards`).
    pub fold: Vec<Option<bool>>,
    /// Where each return may land (`ret_table`).
    pub rets: Vec<Vec<usize>>,
    /// Named barriers (bit = id; all bits for a non-immediate id) some `bar.arrive` of which
    /// may be followed by another arrival there (`exec.rs` `Op::BarSync`, E-BAR-ARRIVE).
    pub arv_repeat: u32,
    /// Named barriers (bit = id; all bits for a non-immediate id) whose `bar.sync` /
    /// `bar.arrive` sites do not all name the same immediate thread count: the first arrival
    /// of a generation fixes its count, so arrival order matters there (E-BAR-ARRIVE).
    pub arv_mixed: u32,
    /// Peeks run as blocking waits (`peekwait_set`): filled in by the explorer only.
    pub peekwait: BTreeSet<u32>,
    /// Peeks run as failed tests (`peekskip_set`): peek pc -> (pcs where its window ends,
    /// the blocking waits it skips). Filled in by the explorer only.
    pub peekskip: BTreeMap<u32, (BTreeSet<u32>, BTreeSet<u32>)>,
    /// Some `bar.sync` / `bar.arrive` names its barrier with a register (E-BAR-OVER's
    /// value-tracking walk is needed only then).
    pub regid: bool,
}

/// Where each return may land (ASSUME(E-RET)), per pc (empty for other instructions): right
/// after a call of a function whose body (walked from its entry without entering nested
/// calls, which are taken to return to their next pc) reaches the return.
/// `PTXV_NO_RET_SITES=1`: anywhere.
pub(crate) fn ret_table(k: &ptxv_ir::Kernel) -> Vec<Vec<usize>> {
    use ptxv_ir::Op;
    let code = &k.code;
    let n = code.len();
    let mut out: Vec<Vec<usize>> = vec![vec![]; n];
    if std::env::var("PTXV_NO_RET_SITES").is_ok() {
        for (pc, i) in code.iter().enumerate() {
            if matches!(i.op, Op::Ret { .. }) {
                out[pc] = (0..n).collect();
            }
        }
        return out;
    }
    let mut callers: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for (pc, i) in code.iter().enumerate() {
        if let Op::Call { target } = i.op {
            if pc + 1 < n {
                callers.entry(target).or_default().push(pc + 1);
            }
        }
    }
    for (f, sites) in callers {
        let mut seen = vec![false; n];
        let mut stack = vec![f];
        while let Some(pc) = stack.pop() {
            if pc >= n || std::mem::replace(&mut seen[pc], true) {
                continue;
            }
            let i = &code[pc];
            let always = i.guard.is_always();
            match &i.op {
                Op::Ret { .. } => {
                    out[pc].extend(&sites);
                    if !always {
                        stack.push(pc + 1);
                    }
                }
                Op::Bra {
                    target,
                    tid0_falls: false,
                    div: None,
                    ..
                } if always => stack.push(*target),
                Op::Bra { target, .. } => stack.extend([*target, pc + 1]),
                Op::Brx { targets, .. } => stack.extend(targets),
                Op::Exit | Op::Trap { .. } if always => {}
                _ => stack.push(pc + 1),
            }
        }
    }
    for v in &mut out {
        v.sort_unstable();
        v.dedup();
    }
    out
}

/// ASSUME(E-CONST-BRA): branches decided by launch constants. Per pc, `Some(v)` when the
/// guard of a plain branch (`Bra` without condition operand, `.ANY` or `NOT_TID0`) has the
/// value v in every thread of every grid that can reach it, by forward constant propagation
/// from each grid's entry over launch constants: constant-bank reads (`Val::Const`, the
/// grid's parameters and launch shape) and immediates, through `Mov` and `Setp` only (every
/// other register write, and everything after a call, is unknown; registers start unknown).
/// The explorer's control-flow graph (`cfg_succ`) drops the other edge of such a branch;
/// `exec.rs` checks the guard on every execution of it (reported as `unmodeled` if it
/// differs). `PTXV_NO_CONST_BRA=1` turns it off. (CUTLASS: `c[0x0][0x36c] == 1` picks the
/// cluster barrier over `bar.sync 0` after the mbarrier inits.)
pub(crate) fn const_guards(m: &Machine, rets: &[Vec<usize>]) -> Vec<Option<bool>> {
    use ptxv_ir::Cmp;
    use ptxv_ir::File;
    use ptxv_ir::Op;
    use ptxv_ir::Pred;
    use ptxv_ir::Reg;
    use ptxv_ir::Val;
    let code = &m.k.code;
    let n = code.len();
    // Per pc: None = no grid reaches it; Some(None) = differs or unknown; Some(Some(v)).
    let mut out: Vec<Option<Option<bool>>> = vec![None; n];
    if std::env::var("PTXV_NO_CONST_BRA").is_ok() || n == 0 {
        return vec![None; n];
    }
    type Env = std::collections::BTreeMap<Reg, u64>;
    for (gi, g) in m.launch.grids.iter().enumerate() {
        let Some(t0) = m
            .threads
            .iter()
            .position(|t| m.launch.grid_index(t.cta as u32) == gi)
        else {
            continue;
        };
        let val = |env: &Env, v: &Val| -> Option<u64> {
            match v {
                Val::Imm(x) => Some(*x),
                Val::Const(off, b) => m.cbank(t0, *off, *b).ok(),
                Val::Reg(r, 1) if r.file != File::V => env.get(r).copied(),
                Val::Reg(r, _) if r.file == File::V => env.get(r).copied(),
                _ => None,
            }
        };
        let pred = |env: &Env, p: Pred| -> Option<bool> {
            let Some(r) = p.reg else {
                return Some(!p.neg);
            };
            // SASS PT / UPT read true.
            if r.idx == 7
                && (r.file == File::UP || m.k.front == ptxv_ir::Front::Sass && r.file == File::P)
            {
                return Some(!p.neg);
            }
            env.get(&r).map(|&v| (v != 0) ^ p.neg)
        };
        let mut st: Vec<Option<Env>> = vec![None; n];
        let entry = g.entry as usize;
        if entry >= n {
            continue;
        }
        st[entry] = Some(Env::new());
        let mut work = vec![entry];
        let mut guards: Vec<Option<Option<bool>>> = vec![None; n];
        while let Some(pc) = work.pop() {
            let env = st[pc].clone().unwrap();
            let i = &code[pc];
            let gv = pred(&env, i.guard);
            guards[pc] = Some(gv);
            // The register state after `pc`.
            let mut next = env.clone();
            let dsts = i.op.dst_regs();
            let mut set: Option<(Reg, u64)> = None;
            match &i.op {
                Op::Mov { d, a } => {
                    let ok = match a {
                        Val::Const(_, b) => d.file == File::V || *b <= 4,
                        Val::Reg(_, k) => d.file == File::V || *k == 1,
                        _ => true,
                    };
                    if let (true, Some(v)) = (ok, val(&env, a)) {
                        let v = if d.file == File::V {
                            v
                        } else {
                            v & 0xffff_ffff
                        };
                        set = Some((*d, v));
                    }
                }
                Op::Setp {
                    d,
                    cmp,
                    bits,
                    a,
                    b,
                    and,
                    or,
                    ex: None,
                    xor: false,
                } => {
                    if let (Some(x), Some(y), Some(c)) =
                        (val(&env, a), val(&env, b), pred(&env, *and))
                    {
                        let w = (*bits as u32).clamp(1, 64);
                        let mk = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
                        let (x, y) = (x & mk, y & mk);
                        let (sx, sy) = (
                            ((x << (64 - w)) as i64) >> (64 - w),
                            ((y << (64 - w)) as i64) >> (64 - w),
                        );
                        let r = match cmp {
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
                        };
                        set = Some((*d, (if *or { r || c } else { r && c }) as u64));
                    }
                }
                _ => {}
            }
            for r in &dsts {
                // A write under a guard that may be off keeps the old value as a possibility.
                match (gv, set) {
                    (Some(false), _) => {}
                    (Some(true), Some((d, v))) if d == *r => {
                        next.insert(d, v);
                    }
                    (None, Some((d, v))) if d == *r && env.get(&d) == Some(&v) => {}
                    _ => {
                        next.remove(r);
                    }
                }
            }
            if gv == Some(false) {
                next = env.clone();
            }
            let fall = (pc + 1 < n).then_some(pc + 1);
            let succ: Vec<(usize, bool)> = match &i.op {
                Op::Bra {
                    target,
                    any: false,
                    tid0_falls: false,
                    cond: None,
                    div: None,
                } => match gv {
                    Some(true) => vec![(*target, false)],
                    Some(false) => fall.map(|f| (f, false)).into_iter().collect(),
                    None => [Some(*target), fall]
                        .into_iter()
                        .flatten()
                        .map(|q| (q, false))
                        .collect(),
                },
                Op::Bra { target, .. } => [Some(*target), fall]
                    .into_iter()
                    .flatten()
                    .map(|q| (q, false))
                    .collect(),
                // After a call nothing is known (the callee writes registers).
                Op::Call { target } => {
                    let mut v = vec![(*target, false)];
                    v.extend(fall.map(|f| (f, true)));
                    v
                }
                Op::Ret { .. } | Op::CallAbs { .. } => rets[pc]
                    .iter()
                    .map(|&q| (q, true))
                    .chain(fall.map(|f| (f, true)))
                    .collect(),
                Op::Brx { targets, .. } => targets.iter().map(|&q| (q, false)).collect(),
                Op::Exit | Op::Trap { .. } if gv == Some(true) => vec![],
                _ => fall.map(|f| (f, false)).into_iter().collect(),
            };
            for (q, clear) in succ {
                let inc = if clear { Env::new() } else { next.clone() };
                let merged = match &st[q] {
                    None => inc,
                    Some(old) => old
                        .iter()
                        .filter(|(r, v)| inc.get(r) == Some(v))
                        .map(|(r, v)| (*r, *v))
                        .collect(),
                };
                if st[q].as_ref() != Some(&merged) {
                    st[q] = Some(merged);
                    work.push(q);
                }
            }
        }
        for pc in 0..n {
            if let Some(gv) = guards[pc] {
                out[pc] = Some(match out[pc] {
                    None => gv,
                    Some(prev) if prev == gv => gv,
                    Some(_) => None,
                });
            }
        }
    }
    let folds: Vec<Option<bool>> = code
        .iter()
        .zip(out)
        .map(|(i, g)| match (&i.op, g) {
            (
                Op::Bra {
                    any: false,
                    tid0_falls: false,
                    cond: None,
                    div: None,
                    ..
                },
                Some(Some(v)),
            ) if !i.guard.is_always() => Some(v),
            _ => None,
        })
        .collect();
    if std::env::var("PTXV_POR_DEBUG").is_ok() {
        for (pc, f) in folds.iter().enumerate() {
            if let Some(v) = f {
                eprintln!(
                    "CONSTBRA {} {}",
                    m.k.describe(pc),
                    if *v { "taken" } else { "falls" }
                );
            }
        }
    }
    folds
}

pub(crate) fn bar_cfg(m: &Machine) -> BarCfg {
    use ptxv_ir::Op;
    use ptxv_ir::Val;
    let code = &m.k.code;
    let rets = ret_table(&m.k);
    let fold = const_guards(m, &rets);
    let succ = cfg_succ(code, 0, &fold, &rets);
    let mut r: Vec<u32> = code
        .iter()
        .map(|i| match &i.op {
            Op::BarSync {
                id: Val::Imm(v), ..
            } if *v < 32 => 1 << v,
            // An indirect call may run any code.
            Op::BarSync { .. } | Op::CallAbs { .. } => u32::MAX,
            _ => 0,
        })
        .collect();
    loop {
        let mut changed = false;
        for pc in (0..code.len()).rev() {
            let m = succ[pc].iter().fold(r[pc], |a, &q| a | r[q]);
            if m != r[pc] {
                r[pc] = m;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut arv_repeat = 0u32;
    for (pc, i) in code.iter().enumerate() {
        if let Op::BarSync {
            id, arrive: true, ..
        } = &i.op
        {
            let bit = match id {
                Val::Imm(v) if *v < 32 => 1u32 << v,
                _ => u32::MAX,
            };
            let after = succ[pc].iter().fold(0, |a, &q| a | r[q]);
            if after & bit != 0 {
                arv_repeat |= bit;
            }
        }
    }
    let mut arv_mixed = 0u32;
    let mut counts: BTreeMap<u32, Option<u32>> = BTreeMap::new();
    for i in code {
        if let Op::BarSync { id, count, .. } = &i.op {
            let c = match count {
                None => None,
                Some(Val::Imm(c)) => Some(*c as u32),
                Some(_) => Some(u32::MAX),
            };
            match id {
                Val::Imm(v) if *v < 32 => {
                    let v = *v as u32;
                    if c == Some(u32::MAX) || counts.get(&v).is_some_and(|o| *o != c) {
                        arv_mixed |= 1 << v;
                    }
                    counts.insert(v, c);
                }
                _ => arv_mixed = u32::MAX,
            }
        }
    }
    BarCfg {
        reach: r,
        succ,
        dsts: code.iter().map(|i| i.op.dst_regs()).collect(),
        fold,
        rets,
        arv_repeat,
        arv_mixed,
        peekwait: BTreeSet::new(),
        peekskip: BTreeMap::new(),
        regid: code.iter().any(|i| {
            matches!(&i.op, ptxv_ir::Op::BarSync { id, .. } if !matches!(id, ptxv_ir::Val::Imm(_)))
        }),
    }
}

/// Register-only operations with no effect but their destinations (the peek-wait shape).
fn reg_only(op: &ptxv_ir::Op) -> bool {
    use ptxv_ir::Op;
    matches!(
        op,
        Op::Mov { .. }
            | Op::Int { .. }
            | Op::MulWide { .. }
            | Op::Imad { .. }
            | Op::Iadd3 { .. }
            | Op::CvtInt { .. }
            | Op::Flo { .. }
            | Op::Brev { .. }
            | Op::AddMinMax { .. }
            | Op::IntMnmx3 { .. }
            | Op::P2r { .. }
            | Op::Lea { .. }
            | Op::LeaHi { .. }
            | Op::Shf { .. }
            | Op::Lop3 { .. }
            | Op::Plop3 { .. }
            | Op::Setp { .. }
            | Op::Sel { .. }
            | Op::Prmt { .. }
            | Op::Bfe { .. }
            | Op::Pack64 { .. }
            | Op::Unpack64 { .. }
            | Op::Nop
    )
}

/// Is `r` read before it is overwritten on some path from `pc`? A reading pc, or a pc where
/// the control flow is not followed (as `wellsync::live_at`).
fn reg_live_at(code: &[ptxv_ir::Inst], pc: usize, r: ptxv_ir::Reg) -> Option<usize> {
    use ptxv_ir::Op;
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
            Op::Call { target } => {
                calls = true;
                work.push(*target);
                work.push(p + 1);
            }
            Op::Ret { .. } if !calls => return Some(p),
            Op::Ret { .. } => {}
            Op::Brx { targets, .. } => work.extend(targets.iter().copied()),
            Op::CallAbs { .. } => return Some(p),
            _ => work.push(p + 1),
        }
    }
    None
}

/// ASSUME(E-SPIN): peeks that are one blocking wait (`PTXV_NO_PEEKWAIT` turns this off). A
/// peek `d = try_wait [A], P` (unguarded, `Machine::peeks`) whose only use is a branch on `d`
/// that skips a blocking wait of the same mbarrier and parity:
/// - from the peek to the branch, register-only instructions that neither read `d` nor
///   write `d` or the poll's operands;
/// - on the failure side, register-only instructions (or unconditional branches) that do not
///   read `d` nor write the poll's operands, then an unguarded try_wait with the same
///   operands and the retry shape (`spin::retry_shape`, so it blocks until it succeeds,
///   E-SPIN), then `@!d bra` to its failure path, falling through to where the success side
///   goes (or `@d bra` there);
/// - every register the failure side wrote (but `d`, true on both sides) is dead there.
///
/// A failed peek then leads, with no effect, to a wait that blocks until the phase
/// completes and joins the success side with the same live registers. So the peek is run as
/// that wait: it blocks while the phase is pending, and a failed peek is invisible, as a
/// failed blocking try_wait is under E-SPIN.
pub(crate) fn peekwait_set(m: &Machine) -> BTreeSet<u32> {
    use ptxv_ir::Op;
    let code = &m.k.code;
    let mut out = BTreeSet::new();
    'peek: for &p in m.peeks.iter() {
        let p = p as usize;
        let inst = &code[p];
        let Op::MbarTryWait {
            d,
            addr,
            parity,
            bit31,
        } = &inst.op
        else {
            continue;
        };
        if !inst.guard.is_always() {
            continue;
        }
        let operands = inst.op.src_regs();
        let ok_pure = |i: &ptxv_ir::Inst, writes_d: bool| {
            reg_only(&i.op)
                && i.guard.reg != Some(*d)
                && !i.op.src_regs().contains(d)
                && i.op
                    .dst_regs()
                    .iter()
                    .all(|r| !operands.contains(r) && (writes_d || r != d))
        };
        // Up to the branch on d.
        let mut q = p + 1;
        let (succ, fail) = loop {
            let Some(i) = code.get(q) else {
                continue 'peek;
            };
            if let Op::Bra {
                target,
                any: false,
                tid0_falls: false,
                cond: None,
                div: None,
            } = i.op
            {
                if i.guard.reg == Some(*d) {
                    break if i.guard.neg {
                        (q + 1, target)
                    } else {
                        (target, q + 1)
                    };
                }
            }
            if !ok_pure(i, false) || q > p + 16 {
                continue 'peek;
            }
            q += 1;
        };
        // The failure side, up to the blocking wait.
        let mut f = fail;
        let mut clob: BTreeSet<ptxv_ir::Reg> = BTreeSet::new();
        let mut hops = 0;
        let w = loop {
            hops += 1;
            let Some(i) = code.get(f) else {
                continue 'peek;
            };
            if hops > 32 || f == p {
                continue 'peek;
            }
            if let Op::MbarTryWait {
                d: d2,
                addr: a2,
                parity: p2,
                bit31: b2,
            } = &i.op
            {
                if d2 == d
                    && a2 == addr
                    && p2 == parity
                    && b2 == bit31
                    && i.guard.is_always()
                    && !m.peeks.contains(&(f as u32))
                    && crate::spin::retry_shape(m, f).is_ok()
                {
                    break f;
                }
                continue 'peek;
            }
            if let Op::Bra {
                target,
                any: false,
                tid0_falls: false,
                cond: None,
                div: None,
            } = i.op
            {
                if i.guard.is_always() {
                    f = target;
                    continue;
                }
                continue 'peek;
            }
            if !ok_pure(i, true) {
                continue 'peek;
            }
            clob.extend(i.op.dst_regs());
            f += 1;
        };
        // Its continuation.
        let cont = match code.get(w + 1) {
            Some(ptxv_ir::Inst {
                op:
                    Op::Bra {
                        target,
                        any: false,
                        tid0_falls: false,
                        cond: None,
                        div: None,
                    },
                guard,
                ..
            }) if guard.reg == Some(*d) => {
                if guard.neg {
                    w + 2
                } else {
                    *target
                }
            }
            _ => continue,
        };
        if cont != succ {
            continue;
        }
        clob.remove(d);
        if clob.iter().any(|&r| reg_live_at(code, succ, r).is_some()) {
            continue;
        }
        out.insert(p as u32);
    }
    out
}

/// A predicate's value in one of the two worlds of `peekskip_set`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
enum Tri {
    F,
    T,
    U,
}

/// ASSUME(E-SPIN): peeks run as failed tests (`PTXV_NO_PEEKWAIT` / `PTXV_NO_PEEKSKIP` turn
/// this off). CUTLASS's "test now, work, then wait": a peek `d = try_wait [A], P` whose
/// result only decides whether a later blocking wait on the same mbarrier and parity runs.
/// Two worlds are walked in lockstep from the peek, `d` true in one (success) and false in the
/// other, tracking which registers may differ (`D`, with each world's known predicate value):
/// - instructions that read nothing in `D` run in both worlds alike (whatever their effect);
///   branches on other values are followed both ways;
/// - predicates derived from `D` (PLOP3, ISETP's combining predicate, VOTE) are evaluated in
///   each world;
/// - a branch on `D` must go the same way in both worlds, or the success world goes one known
///   way and the failure world, alone, through register-only code and one blocking try_wait
///   with the retry shape (`spin::retry_shape`), joins it; every register still differing
///   there must be dead (`reg_live_at`);
/// - anything else that reads `D` (or a call, return, indirect branch) rejects the peek.
///
/// Then every behaviour of the success world is one of the failure world (it waits where the
/// success world skipped the wait, and the wait passes at once), provided (checked at run
/// time, `exec.rs`) the skipped wait is on the peek's mbarrier and parity, and the mbarrier
/// does not complete a phase while the window between the peek and that wait is open with the
/// peek's result true (ABA: the success world would have gone on, the failure world waits).
/// So the peek is run as a failed test, which reads no phase: it no longer depends on the
/// arrival that completes the phase. Its footprint reads only the key of the completion
/// that would be such an ABA, so that order is explored and the check sees it. Returns, per
/// peek, the pcs where the window ends (the worlds have joined on every path through them)
/// and the waits it skips.
pub(crate) fn peekskip_set(m: &Machine) -> BTreeMap<u32, (BTreeSet<u32>, BTreeSet<u32>)> {
    use ptxv_ir::Op;
    use ptxv_ir::Pred;
    use ptxv_ir::Reg;
    type Dm = BTreeMap<Reg, (Tri, Tri)>;
    /// Visited (pc, flattened `Dm`) pairs.
    type Seen = HashSet<(usize, Vec<(Reg, Tri, Tri)>)>;
    let code = &m.k.code;
    let sass = m.k.front == ptxv_ir::Front::Sass;
    let blocking = |q: usize| {
        let i = &code[q];
        matches!(i.op, Op::MbarTryWait { .. })
            && i.guard.is_always()
            && !m.peeks.contains(&(q as u32))
            && crate::spin::retry_shape(m, q).is_ok()
    };
    let mut out = BTreeMap::new();
    'peek: for &p in m.peeks.iter() {
        let p = p as usize;
        if m.bar_cfg.peekwait.contains(&(p as u32)) {
            continue;
        }
        let Op::MbarTryWait { d, .. } = &code[p].op else {
            continue;
        };
        let guarded = !code[p].guard.is_always();
        let dbg = std::env::var("PTXV_POR_DEBUG").is_ok();
        macro_rules! fail {
            ($q:expr, $why:expr) => {{
                if dbg {
                    eprintln!(
                        "PORPEEKSKIP no {:#x}: {} at {:#x}",
                        code[p].loc,
                        $why,
                        code.get($q).map_or(0, |i| i.loc)
                    );
                }
                continue 'peek;
            }};
        }
        // A predicate's value in world `w` (0 success, 1 failure): None if it is not in D
        // (the same in both worlds, unknown).
        let tri = |dm: &Dm, q: Pred, w: usize| -> Option<Tri> {
            let r = q.reg?;
            let v = dm.get(&r)?;
            let x = if w == 0 { v.0 } else { v.1 };
            Some(match (x, q.neg) {
                (Tri::U, _) => Tri::U,
                (Tri::T, false) | (Tri::F, true) => Tri::T,
                _ => Tri::F,
            })
        };
        let konst = |q: Pred| -> Option<bool> {
            match q.reg {
                None => Some(!q.neg),
                Some(r)
                    if r.idx == 7
                        && (r.file == ptxv_ir::File::UP || r.file == ptxv_ir::File::P && sass) =>
                {
                    Some(!q.neg)
                }
                _ => None,
            }
        };
        // Possible values of a predicate operand in world w, as a Tri, where operands not in
        // D are unknown (but equal across worlds, which these derivations do not need).
        let val = |dm: &Dm, q: Pred, w: usize| -> Tri {
            if let Some(b) = konst(q) {
                return if b { Tri::T } else { Tri::F };
            }
            tri(dm, q, w).unwrap_or(Tri::U)
        };
        let in_d = |dm: &Dm, i: &ptxv_ir::Inst| -> bool {
            i.guard.reg.is_some_and(|r| dm.contains_key(&r))
                || i.op.src_regs().iter().any(|r| dm.contains_key(r))
        };
        let mut d0: Dm = BTreeMap::new();
        d0.insert(*d, (Tri::T, Tri::F));
        let mut work: Vec<(usize, Dm)> = vec![(p + 1, d0)];
        let mut seen: Seen = HashSet::new();
        let mut joins: BTreeSet<u32> = BTreeSet::new();
        let mut open: BTreeSet<u32> = BTreeSet::new();
        let mut ws: BTreeSet<u32> = BTreeSet::new();
        let mut steps = 0;
        while let Some((pc, mut dm)) = work.pop() {
            dm.retain(|r, v| !(v.0 == v.1 && v.0 != Tri::U) && reg_live_at(code, pc, *r).is_some());
            if dm.is_empty() {
                joins.insert(pc as u32);
                continue;
            }
            steps += 1;
            if steps > 4000 {
                fail!(pc, "too long");
            }
            let key = (
                pc,
                dm.iter().map(|(r, v)| (*r, v.0, v.1)).collect::<Vec<_>>(),
            );
            if !seen.insert(key) {
                continue;
            }
            if pc == p {
                // The peek again: the walk from it is this one.
                continue;
            }
            let Some(i) = code.get(pc) else {
                fail!(pc, "end of code");
            };
            open.insert(pc as u32);
            if in_d(&dm, i) {
                match &i.op {
                    Op::Bra {
                        target,
                        any: false,
                        tid0_falls: false,
                        cond: None,
                        div: None,
                    } => {
                        let (ds, df) = (val(&dm, i.guard, 0), val(&dm, i.guard, 1));
                        let s_next = match ds {
                            Tri::T => *target,
                            Tri::F => pc + 1,
                            Tri::U => fail!(pc, "branch unknown in the success world"),
                        };
                        let f_nexts: Vec<usize> = match df {
                            Tri::T => vec![*target],
                            Tri::F => vec![pc + 1],
                            Tri::U => vec![*target, pc + 1],
                        };
                        for f in f_nexts {
                            if f == s_next {
                                work.push((s_next, dm.clone()));
                                continue;
                            }
                            // The failure world alone, to the join: register-only
                            // code, branches its own predicate values decide, blocking
                            // try_waits (their continuation once they succeed), and peeks
                            // (both results).
                            let mut fstack: Vec<(usize, Dm, usize)> = vec![(f, dm.clone(), 0)];
                            let mut fseen: Seen = HashSet::new();
                            while let Some((q, mut fd, hops)) = fstack.pop() {
                                if q == s_next {
                                    fd.retain(|_, v| !(v.0 == v.1 && v.0 != Tri::U));
                                    if let Some(r) =
                                        fd.keys().find(|&&r| reg_live_at(code, s_next, r).is_some())
                                    {
                                        fail!(s_next, format!("{r:?} live where the worlds join"));
                                    }
                                    joins.insert(s_next as u32);
                                    continue;
                                }
                                let Some(j) = code.get(q) else {
                                    fail!(q, "end of code");
                                };
                                // (The failure world runs its own values: reading D is
                                // fine; the waits' mbarrier and parity are checked at run
                                // time.)
                                if hops > 64 || q == p {
                                    fail!(q, "failure path too long (or loops)");
                                }
                                if !fseen
                                    .insert((q, fd.iter().map(|(r, v)| (*r, v.0, v.1)).collect()))
                                {
                                    continue;
                                }
                                open.insert(q as u32);
                                if let Op::Bra {
                                    target,
                                    any: false,
                                    tid0_falls: false,
                                    cond: None,
                                    div: None,
                                } = j.op
                                {
                                    match val(&fd, j.guard, 1) {
                                        Tri::T => fstack.push((target, fd, hops + 1)),
                                        Tri::F => fstack.push((q + 1, fd, hops + 1)),
                                        Tri::U => {
                                            fail!(q, "a branch the failure world cannot decide")
                                        }
                                    }
                                    continue;
                                }
                                if let Op::MbarTryWait { d: dw, .. } = &j.op {
                                    let peek_here = m.peeks.contains(&(q as u32))
                                        && !m.bar_cfg.peekwait.contains(&(q as u32));
                                    if !j.guard.is_always() || !(peek_here || blocking(q)) {
                                        fail!(
                                            q,
                                            "a guarded or unrecognized try_wait on the failure path"
                                        );
                                    }
                                    ws.insert(q as u32);
                                    let sv = fd.get(dw).map_or(Tri::U, |v| v.0);
                                    if peek_here {
                                        let mut fd2 = fd.clone();
                                        fd2.insert(*dw, (sv, Tri::F));
                                        fstack.push((q + 1, fd2, hops + 1));
                                    }
                                    fd.insert(*dw, (sv, Tri::T));
                                    // Its continuation once it succeeds (d true): what
                                    // `@!d` guards does not run (back-off, re-polls, the
                                    // retry branch); branches taken on d are followed.
                                    let mut c = q + 1;
                                    let mut n = 0;
                                    loop {
                                        n += 1;
                                        let Some(k) = code.get(c) else {
                                            fail!(c, "end of code");
                                        };
                                        if n > 16 {
                                            fail!(c, "no continuation after the wait");
                                        }
                                        if k.guard.reg == Some(*dw) && k.guard.neg {
                                            c += 1;
                                            continue;
                                        }
                                        if let Op::Bra {
                                            target,
                                            any: false,
                                            tid0_falls: false,
                                            cond: None,
                                            div: None,
                                        } = k.op
                                        {
                                            if k.guard.is_always()
                                                || k.guard.reg == Some(*dw) && !k.guard.neg
                                            {
                                                c = target;
                                                continue;
                                            }
                                        }
                                        break;
                                    }
                                    fstack.push((c, fd, hops + 1));
                                    continue;
                                }
                                if !(reg_only(&j.op) || matches!(j.op, Op::Nop))
                                    || j.guard.reg.is_some_and(|r| !fd.contains_key(&r))
                                        && !j.guard.is_always()
                                {
                                    fail!(q, "an effect on the failure path");
                                }
                                // A guarded register op on the failure path: its guard's
                                // value there decides (unknown: both, as differing).
                                let runs = !matches!(val(&fd, j.guard, 1), Tri::F);
                                if runs && !matches!(j.op, Op::Nop) {
                                    for r in j.op.dst_regs() {
                                        fd.insert(r, (Tri::U, Tri::U));
                                    }
                                }
                                fstack.push((q + 1, fd, hops + 1));
                            }
                        }
                    }
                    Op::Plop3 {
                        d: pd,
                        a,
                        b,
                        c,
                        lut,
                        d2: None,
                    } if i.guard.is_always() => {
                        let mut v = [Tri::U; 2];
                        for (w, vw) in v.iter_mut().enumerate() {
                            let ins = [val(&dm, *a, w), val(&dm, *b, w), val(&dm, *c, w)];
                            let mut seen_v = [false; 2];
                            for x in 0..8u8 {
                                let ok = (0..3).all(|k| {
                                    let bit = (x >> (2 - k)) & 1 == 1;
                                    match ins[k] {
                                        Tri::T => bit,
                                        Tri::F => !bit,
                                        Tri::U => true,
                                    }
                                });
                                if ok {
                                    seen_v[((lut >> x) & 1) as usize] = true;
                                }
                            }
                            *vw = match seen_v {
                                [true, false] => Tri::F,
                                [false, true] => Tri::T,
                                _ => Tri::U,
                            };
                        }
                        dm.insert(*pd, (v[0], v[1]));
                        work.push((pc + 1, dm));
                    }
                    Op::Setp {
                        d: sd,
                        a,
                        b,
                        and,
                        or,
                        ex: None,
                        ..
                    } if i.guard.is_always()
                        && [a, b].iter().all(|x| match x {
                            ptxv_ir::Val::Reg(r, _) => !dm.contains_key(r),
                            _ => true,
                        }) =>
                    {
                        let f = |x: Tri| match (x, *or) {
                            (Tri::T, true) => Tri::T,
                            (Tri::F, false) => Tri::F,
                            _ => Tri::U,
                        };
                        dm.insert(*sd, (f(val(&dm, *and, 0)), f(val(&dm, *and, 1))));
                        work.push((pc + 1, dm));
                    }
                    Op::VoteAny { d: vd, a }
                    | Op::VoteAll {
                        d: vd,
                        a,
                        uni: false,
                    } if i.guard.is_always() && !guarded => {
                        dm.insert(*vd, (val(&dm, *a, 0), val(&dm, *a, 1)));
                        work.push((pc + 1, dm));
                    }
                    _ => fail!(pc, "D read"),
                }
                continue;
            }
            // Runs alike in both worlds.
            let always = i.guard.is_always();
            for r in i.op.dst_regs() {
                if always {
                    dm.remove(&r);
                } else if let Some(v) = dm.get_mut(&r) {
                    *v = (Tri::U, Tri::U);
                }
            }
            match &i.op {
                Op::Bra {
                    target,
                    cond,
                    tid0_falls,
                    ..
                } => {
                    work.push((*target, dm.clone()));
                    if !always || cond.is_some() || *tid0_falls {
                        work.push((pc + 1, dm));
                    }
                }
                Op::Exit | Op::Trap { .. } if always => {
                    joins.insert(pc as u32);
                }
                Op::Call { .. } | Op::Ret { .. } | Op::Brx { .. } | Op::CallAbs { .. } => {
                    fail!(pc, "call, return or indirect branch")
                }
                _ => work.push((pc + 1, dm)),
            }
        }
        // A peek whose result skips no wait (it is dead, or only steers code that runs alike)
        // keeps its plain semantics: its success also acquires, which a failed test would not.
        if ws.is_empty() {
            fail!(p, "it skips no wait");
        }
        let joins: BTreeSet<u32> = joins.difference(&open).copied().collect();
        out.insert(p as u32, (joins, ws));
    }
    out
}

/// A value `Machine::thread_may_arrive` read: a predicate, or an operand.
#[derive(Clone, Debug, PartialEq)]
enum BarKey {
    Pred(ptxv_ir::Reg),
    Val(ptxv_ir::Val),
}
type BarRead = (BarKey, u64);

/// Is the bar.sync at `pc` an arrival at a barrier that may receive more arrivals than its
/// count (`Machine::bar_oversub`), for any of warp `w`'s threads there? Operands that cannot
/// be read: conservatively yes.
fn bar_spills(m: &Machine, w: u32, pc: u32) -> bool {
    let ptxv_ir::Op::BarSync {
        id, count, packed, ..
    } = &m.k.code[pc as usize].op
    else {
        return false;
    };
    let Some(count) = count else {
        return false;
    };
    m.warp_threads(w)
        .filter(|&t| m.threads[t].pc == pc && m.threads[t].st == St::Run)
        .any(|t| match (m.rd(t, id).ok(), m.rd(t, count).ok()) {
            (Some(i), Some(c)) => {
                let (i, c) = ptxv_ir::bar_args(i, Some(c), *packed);
                m.bar_oversub(m.threads[t].cta, i, c)
            }
            _ => true,
        })
}

impl Machine {
    /// ASSUME(E-BAR-OVER): can barrier `id` of `cta` (thread count `count`) receive more
    /// arrivals than `count` from here on? Every live thread of the CTA that may still reach
    /// a `bar.sync` / `bar.arrive` on it counts (those parked there included). With at most
    /// `count` of them, a thread arriving once per generation (bar.sync waits), the next
    /// generation completes exactly when all of them arrived, so arrivals commute; with more,
    /// which threads share a generation depends on arrival order. All-thread barriers
    /// (`count` None) never spill. Monotone: the count only shrinks along an execution
    /// (`thread_may_arrive`).
    /// ASSUME(E-BAR-ARRIVE): does `bar.arrive` by thread `t` on barrier `id` of `cta` land
    /// in the current generation whatever the order of the other threads' steps
    /// (`PTXV_NO_ARVGEN` turns this off)? Yes when no thread can arrive twice in one
    /// generation (`t` has not arrived in it, and no `bar.arrive` on `id` is followed by
    /// another arrival there, `BarCfg::arv_repeat`), every site on `id` names the same
    /// immediate thread count (`BarCfg::arv_mixed`: the first arrival fixes the generation's
    /// count, so with mixed counts order matters), and the generation cannot complete
    /// without `t`: for an all-thread barrier always; for a counted one, the arrivals so far
    /// plus every other thread that may still arrive (`thread_may_arrive`) stay below the
    /// count. Then its arrival, like a `bar.sync` one, only reads the generation's key.
    pub(crate) fn arv_current(&self, t: usize, cta: u16, id: u32, count: Option<u32>) -> bool {
        if std::env::var("PTXV_NO_ARVGEN").is_ok() {
            return false;
        }
        let bit = if id < 32 { 1u32 << id } else { u32::MAX };
        if (self.bar_cfg.arv_repeat | self.bar_cfg.arv_mixed) & bit != 0 {
            return false;
        }
        let Some(bar) = self.bars.get(&(cta, id)) else {
            return false;
        };
        if bar.arrivers.binary_search(&(t as u32)).is_ok()
            || bar.arrived.binary_search(&(t as u32)).is_ok()
        {
            return false;
        }
        let Some(c) = count else {
            return true;
        };
        let mut n = bar.arrived.len() + bar.arrivers.len();
        // Arrivals so far and threads that may still arrive, per warp (E-BAR-EXIT: a warp
        // that arrives counts the warp size).
        let mut per_warp: HashMap<u32, usize> = HashMap::new();
        for &a in bar.arrived.iter().chain(&bar.arrivers) {
            *per_warp.entry(self.threads[a as usize].warp).or_default() += 1;
        }
        let mut memo: Vec<(u32, Vec<BarRead>, bool)> = Vec::new();
        for (u, th) in self.live_threads() {
            if u == t
                || th.cta != cta
                || th.st == St::Exited
                || self.bar_cfg.reach[th.pc as usize] & bit == 0
                || bar.arrived.binary_search(&(u as u32)).is_ok()
                || bar.arrivers.binary_search(&(u as u32)).is_ok()
            {
                continue;
            }
            let hit = memo
                .iter()
                .find(|(pc, reads, _)| {
                    *pc == th.pc && reads.iter().all(|r| self.bar_read(u, &r.0) == Some(r.1))
                })
                .map(|e| e.2);
            let may = hit.unwrap_or_else(|| {
                let mut reads = Vec::new();
                let may = self.thread_may_arrive(u, bit, &mut reads);
                memo.push((th.pc, reads, may));
                may
            });
            if may {
                n += 1;
                *per_warp.entry(th.warp).or_default() += 1;
                if n + 1 > c as usize {
                    return false;
                }
            }
        }
        // ASSUME(E-BAR-EXIT): before `t` arrives, a warp with an arrival counts at most the
        // warp size (one less for `t`'s own warp: `t` is live and has not arrived), its
        // exited lanes included.
        let tw = self.threads[t].warp;
        let bound: usize = per_warp
            .iter()
            .map(|(&w, &k)| k.max(if w == tw { 31 } else { 32 }))
            .sum();
        bound < c as usize
    }

    pub fn bar_oversub(&self, cta: u16, id: u32, count: Option<u32>) -> bool {
        let Some(c) = count else {
            return false;
        };
        let bit = if id < 32 { 1u32 << id } else { u32::MAX };
        let mut n = 0;
        let mut per_warp: HashMap<u32, usize> = HashMap::new();
        // The walk is a function of the pc and the values it read: threads that agree on
        // them (typically the lanes of a warp) share one walk.
        let mut memo: Vec<(u32, Vec<BarRead>, bool)> = Vec::new();
        for (t, th) in self.live_threads() {
            if th.cta != cta || th.st == St::Exited || self.bar_cfg.reach[th.pc as usize] & bit == 0
            {
                continue;
            }
            let hit = memo
                .iter()
                .find(|(pc, reads, _)| {
                    *pc == th.pc && reads.iter().all(|r| self.bar_read(t, &r.0) == Some(r.1))
                })
                .map(|e| e.2);
            let may = hit.unwrap_or_else(|| {
                let mut reads = Vec::new();
                let may = self.thread_may_arrive(t, bit, &mut reads);
                memo.push((th.pc, reads, may));
                may
            });
            if may {
                n += 1;
                *per_warp.entry(th.warp).or_default() += 1;
                if n > c as usize {
                    return true;
                }
            }
        }
        // ASSUME(E-BAR-EXIT): a warp that arrives counts the warp size.
        per_warp.values().map(|&k| k.max(32)).sum::<usize>() > c as usize
    }

    /// Can thread `t` still reach a barrier in `bit`? A walk of the control-flow graph from
    /// its pc that decides guarded branches and exits where it can: by predicates a
    /// comparison on the walked path set from values it can compute, or else by the thread's
    /// current predicate values. A register value is only used when no instruction in the
    /// walked region writes it (it keeps its value while the thread stays there, and the
    /// thread can leave only by edges walked); one found written is treated as unknown and the
    /// walk redone.
    fn bar_read(&self, t: usize, k: &BarKey) -> Option<u64> {
        match k {
            BarKey::Pred(r) => Some(self.prd(
                t,
                ptxv_ir::Pred {
                    reg: Some(*r),
                    neg: false,
                },
            ) as u64),
            BarKey::Val(v) => self.rd(t, v).ok(),
        }
    }

    /// `thread_may_arrive_walk`, memoized per worker thread across states: the walk is a
    /// function of the code (`bar_cfg`), the pc, `bit` and the values it read, so an earlier
    /// walk from the same pc applies to every thread reading the same values there.
    fn thread_may_arrive(&self, t: usize, bit: u32, reads: &mut Vec<BarRead>) -> bool {
        type Walks = HashMap<(usize, u32, u32), Vec<(Vec<BarRead>, bool)>>;
        thread_local! {
            static WALKS: std::cell::RefCell<Walks> = Default::default();
        }
        let key = (Arc::as_ptr(&self.bar_cfg) as usize, self.threads[t].pc, bit);
        let hit = WALKS.with(|w| {
            w.borrow().get(&key).and_then(|es| {
                es.iter()
                    .find(|(rs, _)| rs.iter().all(|r| self.bar_read(t, &r.0) == Some(r.1)))
                    .cloned()
            })
        });
        if let Some((rs, may)) = hit {
            *reads = rs;
            return may;
        }
        let may = self.thread_may_arrive_walk(t, bit, reads);
        // One entry per value read (a walk reads the same register on many paths).
        let mut seen_k: Vec<BarKey> = Vec::new();
        reads.retain(|(k, _)| {
            if seen_k.contains(k) {
                false
            } else {
                seen_k.push(k.clone());
                true
            }
        });
        WALKS.with(|w| {
            let mut w = w.borrow_mut();
            if w.len() > 1 << 16 {
                w.clear();
            }
            let es = w.entry(key).or_default();
            if es.len() >= 64 {
                es.remove(0);
            }
            es.push((reads.clone(), may));
        });
        may
    }

    fn thread_may_arrive_walk(&self, t: usize, bit: u32, reads: &mut Vec<BarRead>) -> bool {
        // Barrier ids in registers need the value-tracking walk; kernels whose `bar.sync`
        // ids are all immediates keep the plain walk (same answers there, cheaper memo).
        let regid = self.bar_cfg.regid;
        if regid {
            self.thread_may_arrive_walk_regs(t, bit, reads)
        } else {
            self.thread_may_arrive_walk_plain(t, bit, reads)
        }
    }

    fn thread_may_arrive_walk_plain(&self, t: usize, bit: u32, reads: &mut Vec<BarRead>) -> bool {
        use ptxv_ir::Cmp;
        use ptxv_ir::Op;
        use ptxv_ir::Pred;
        use ptxv_ir::Reg;
        use ptxv_ir::Val;
        type Env = Vec<(Reg, bool)>;
        let cfg = &self.bar_cfg;
        let mut unknown: Vec<Reg> = Vec::new();
        loop {
            let mut seen: HashSet<(usize, Env)> = HashSet::new();
            let mut stack: Vec<(usize, Env)> = vec![(self.threads[t].pc as usize, vec![])];
            let mut used: Vec<Reg> = Vec::new();
            let mut written: Vec<Reg> = Vec::new();
            let mut hit = false;
            while let Some((pc, env)) = stack.pop() {
                if pc >= cfg.reach.len() || cfg.reach[pc] & bit == 0 {
                    continue;
                }
                if !seen.insert((pc, env.clone())) {
                    continue;
                }
                let i = &self.k.code[pc];
                if match &i.op {
                    Op::BarSync {
                        id: Val::Imm(v), ..
                    } => *v >= 32 || (1u32 << v) & bit != 0,
                    Op::BarSync { .. } | Op::CallAbs { .. } => true,
                    _ => false,
                } {
                    hit = true;
                    break;
                }
                written.extend(cfg.dsts[pc].iter().copied());
                let pred =
                    |p: Pred, used: &mut Vec<Reg>, reads: &mut Vec<BarRead>| -> Option<bool> {
                        let Some(r) = p.reg else {
                            return Some(true);
                        };
                        if let Some(&(_, v)) = env.iter().find(|e| e.0 == r) {
                            return Some(v ^ p.neg);
                        }
                        if unknown.contains(&r) {
                            return None;
                        }
                        used.push(r);
                        let k = BarKey::Pred(r);
                        let v = self.bar_read(t, &k);
                        reads.push((k, v.unwrap_or(0)));
                        v.map(|v| (v != 0) ^ p.neg)
                    };
                let guard = pred(i.guard, &mut used, reads);
                let mut next = env.clone();
                next.retain(|e| !cfg.dsts[pc].contains(&e.0));
                if let (
                    Some(true),
                    Op::Setp {
                        d,
                        cmp,
                        bits,
                        a,
                        b,
                        and,
                        or,
                        ex: None,
                        xor: false,
                    },
                ) = (guard, &i.op)
                {
                    let val = |v: &Val, used: &mut Vec<Reg>, reads: &mut Vec<BarRead>| {
                        if let Val::Reg(r, _) = v {
                            if unknown.contains(r) {
                                return None;
                            }
                            used.push(*r);
                        }
                        let k = BarKey::Val(v.clone());
                        let x = self.bar_read(t, &k);
                        // (A failed read is recorded as 0: a thread reading 0 there reuses a
                        // walk that knew less, which is conservative.)
                        reads.push((k, x.unwrap_or(0)));
                        x
                    };
                    let x = val(a, &mut used, reads);
                    let y = val(b, &mut used, reads);
                    let c = pred(*and, &mut used, reads);
                    // (SASS PT / UPT: writes are discarded.)
                    let pt = self.k.front == ptxv_ir::Front::Sass && d.idx == 7;
                    if let (Some(x), Some(y), Some(c), false) = (x, y, c, pt) {
                        let w = (*bits as u32).clamp(1, 64);
                        let mk = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
                        let (x, y) = (x & mk, y & mk);
                        let (sx, sy) = (
                            ((x << (64 - w)) as i64) >> (64 - w),
                            ((y << (64 - w)) as i64) >> (64 - w),
                        );
                        let r = match cmp {
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
                        };
                        next.push((*d, if *or { r || c } else { r && c }));
                        next.sort();
                    }
                }
                let decided = match &i.op {
                    Op::Bra {
                        cond: None,
                        any: false,
                        tid0_falls: false,
                        div: None,
                        ..
                    }
                    | Op::Exit
                    | Op::Trap { .. } => guard,
                    _ => None,
                };
                match (decided, &i.op) {
                    (Some(true), Op::Bra { target, .. }) => stack.push((*target, next)),
                    (Some(true), _) => {}
                    (Some(false), _) => stack.push((pc + 1, next)),
                    (None, _) => stack.extend(cfg.succ[pc].iter().map(|&q| (q, next.clone()))),
                }
            }
            if hit {
                return true;
            }
            let bad: Vec<_> = used.into_iter().filter(|r| written.contains(r)).collect();
            if bad.is_empty() {
                return false;
            }
            unknown.extend(bad);
        }
    }

    fn thread_may_arrive_walk_regs(&self, t: usize, bit: u32, reads: &mut Vec<BarRead>) -> bool {
        use ptxv_ir::Cmp;
        use ptxv_ir::Op;
        use ptxv_ir::Pred;
        use ptxv_ir::Reg;
        use ptxv_ir::Val;
        type Env = Vec<(Reg, bool)>;
        // Register values computed on the walked path from thread constants (special
        // registers, immediates, launch constants) through MOV / SEL: exact on that path, so
        // a barrier id recomputed from %tid in a loop body is known (DeepEP's
        // `bar.sync (tid >= 384), 384` halves).
        // Each tracked register holds its source: an immediate, special register, constant,
        // or a register's value at the walk's start. Evaluated (and recorded in `reads`)
        // only when a decision uses it, so lanes that differ only in an unused %tid share
        // the walk's memo.
        type REnv = Vec<(Reg, Val)>;
        let cfg = &self.bar_cfg;
        // Registers some walked path from the thread's pc may write before each pc (forward
        // may-write dataflow over the region that can still reach a barrier in `bit`). A
        // register outside both this and the path's `REnv` still holds its current value.
        let start = self.threads[t].pc as usize;
        // Bitsets over the registers the region writes (`MwSets::ix`).
        struct MwSets {
            ix: HashMap<Reg, usize>,
            at: HashMap<usize, Vec<u64>>,
        }
        impl MwSets {
            fn may(&self, pc: usize, r: &Reg) -> bool {
                let Some(set) = self.at.get(&pc) else {
                    return true;
                };
                match self.ix.get(r) {
                    Some(&i) => set.get(i / 64).is_some_and(|w| w >> (i % 64) & 1 == 1),
                    None => false,
                }
            }
        }
        type Mw = Arc<MwSets>;
        thread_local! {
            // The dataflow depends only on the code, the start pc and `bit`.
            static MW: std::cell::RefCell<HashMap<(usize, usize, u32), Mw>> = Default::default();
        }
        let key = (Arc::as_ptr(&self.bar_cfg) as usize, start, bit);
        let mw: Mw = match MW.with(|c| c.borrow().get(&key).cloned()) {
            Some(m) => m,
            None => {
                let mut ix: HashMap<Reg, usize> = HashMap::new();
                let mut at: HashMap<usize, Vec<u64>> = HashMap::new();
                let mut work = vec![start];
                at.insert(start, Vec::new());
                while let Some(pc) = work.pop() {
                    if pc >= cfg.reach.len() || cfg.reach[pc] & bit == 0 {
                        continue;
                    }
                    let mut out = at[&pc].clone();
                    for &d in &cfg.dsts[pc] {
                        let n = ix.len();
                        let i = *ix.entry(d).or_insert(n);
                        if out.len() <= i / 64 {
                            out.resize(i / 64 + 1, 0);
                        }
                        out[i / 64] |= 1 << (i % 64);
                    }
                    for &q in &cfg.succ[pc] {
                        let fresh = !at.contains_key(&q);
                        let e = at.entry(q).or_default();
                        if e.len() < out.len() {
                            e.resize(out.len(), 0);
                        }
                        let mut grew = false;
                        for (x, y) in e.iter_mut().zip(&out) {
                            if *x | *y != *x {
                                *x |= *y;
                                grew = true;
                            }
                        }
                        if fresh || grew {
                            work.push(q);
                        }
                    }
                }
                let m = Arc::new(MwSets { ix, at });
                MW.with(|c| {
                    let mut c = c.borrow_mut();
                    if c.len() > 4096 {
                        c.clear();
                    }
                    c.insert(key, m.clone());
                });
                m
            }
        };
        let mut unknown: Vec<Reg> = Vec::new();
        loop {
            let mut seen: HashSet<(usize, Env, REnv)> = HashSet::new();
            let mut stack: Vec<(usize, Env, REnv)> =
                vec![(self.threads[t].pc as usize, vec![], vec![])];
            let mut used: Vec<Reg> = Vec::new();
            let mut written: HashSet<Reg> = HashSet::new();
            let mut hit = false;
            while let Some((pc, env, renv)) = stack.pop() {
                if pc >= cfg.reach.len() || cfg.reach[pc] & bit == 0 {
                    continue;
                }
                if !seen.insert((pc, env.clone(), renv.clone())) {
                    continue;
                }
                let i = &self.k.code[pc];
                // A value operand: from the path's environment, a thread constant (recorded
                // in `reads`, as the walk's result depends on it), or a register the walked
                // region must not write (`used`).
                let rval =
                    |v: &Val, used: &mut Vec<Reg>, reads: &mut Vec<BarRead>| -> Option<u64> {
                        match v {
                            Val::Reg(r, _) => {
                                // Path-sensitive: a register not written on the path from the
                                // thread's pc still holds its current value.
                                if let Some((_, src)) = renv.iter().find(|e| e.0 == *r) {
                                    let k = BarKey::Val(src.clone());
                                    let x = self.bar_read(t, &k);
                                    reads.push((k, x.unwrap_or(0)));
                                    return x;
                                }
                                if mw.may(pc, r) {
                                    return None;
                                }
                                let _ = &used;
                            }
                            Val::Imm(x) => return Some(*x),
                            _ => {}
                        }
                        let k = BarKey::Val(v.clone());
                        let x = self.bar_read(t, &k);
                        reads.push((k, x.unwrap_or(0)));
                        x
                    };
                if match &i.op {
                    Op::BarSync {
                        id: Val::Imm(v), ..
                    } => *v >= 32 || (1u32 << v) & bit != 0,
                    Op::BarSync { packed: true, .. } => true,
                    Op::BarSync { id, .. } if bit != u32::MAX => match rval(id, &mut used, reads) {
                        Some(v) => v >= 32 || (1u32 << v) & bit != 0,
                        None => true,
                    },
                    Op::BarSync { .. } | Op::CallAbs { .. } => true,
                    _ => false,
                } {
                    hit = true;
                    break;
                }
                written.extend(cfg.dsts[pc].iter().copied());
                let pred =
                    |p: Pred, used: &mut Vec<Reg>, reads: &mut Vec<BarRead>| -> Option<bool> {
                        let Some(r) = p.reg else {
                            return Some(true);
                        };
                        if let Some(&(_, v)) = env.iter().find(|e| e.0 == r) {
                            return Some(v ^ p.neg);
                        }
                        if unknown.contains(&r) {
                            return None;
                        }
                        used.push(r);
                        let k = BarKey::Pred(r);
                        let v = self.bar_read(t, &k);
                        reads.push((k, v.unwrap_or(0)));
                        v.map(|v| (v != 0) ^ p.neg)
                    };
                let guard = pred(i.guard, &mut used, reads);
                let mut next = env.clone();
                next.retain(|e| !cfg.dsts[pc].contains(&e.0));
                let mut rnext = renv.clone();
                rnext.retain(|e| !cfg.dsts[pc].contains(&e.0));
                if guard == Some(true) {
                    // The source of an operand, without evaluating it.
                    let src = |v: &Val| -> Option<Val> {
                        match v {
                            Val::Reg(r, _) => match renv.iter().find(|e| e.0 == *r) {
                                Some((_, s)) => Some(s.clone()),
                                None if mw.may(pc, r) => None,
                                None => Some(v.clone()),
                            },
                            Val::SReg(ptxv_ir::SReg::SharedAddr(_)) => None,
                            _ => Some(v.clone()),
                        }
                    };
                    let v = match &i.op {
                        Op::Mov { d, a } => src(a).map(|x| (*d, x)),
                        Op::Sel { d, a, b, p } => match pred(*p, &mut used, reads) {
                            Some(true) => src(a).map(|x| (*d, x)),
                            Some(false) => src(b).map(|x| (*d, x)),
                            None => None,
                        },
                        _ => None,
                    };
                    if let Some((d, x)) = v {
                        if matches!(
                            d.file,
                            ptxv_ir::File::R | ptxv_ir::File::UR | ptxv_ir::File::V
                        ) {
                            rnext.push((d, x));
                            rnext.sort_by_key(|e| (e.0.file as u8, e.0.idx));
                        }
                    }
                }
                if let (
                    Some(true),
                    Op::Setp {
                        d,
                        cmp,
                        bits,
                        a,
                        b,
                        and,
                        or,
                        ex: None,
                        xor: false,
                    },
                ) = (guard, &i.op)
                {
                    // (A failed read is recorded as 0: a thread reading 0 there reuses a walk
                    // that knew less, which is conservative.)
                    let val = |v: &Val, used: &mut Vec<Reg>, reads: &mut Vec<BarRead>| {
                        rval(v, used, reads)
                    };
                    let x = val(a, &mut used, reads);
                    let y = val(b, &mut used, reads);
                    let c = pred(*and, &mut used, reads);
                    // (SASS PT / UPT: writes are discarded.)
                    let pt = self.k.front == ptxv_ir::Front::Sass && d.idx == 7;
                    if let (Some(x), Some(y), Some(c), false) = (x, y, c, pt) {
                        let w = (*bits as u32).clamp(1, 64);
                        let mk = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
                        let (x, y) = (x & mk, y & mk);
                        let (sx, sy) = (
                            ((x << (64 - w)) as i64) >> (64 - w),
                            ((y << (64 - w)) as i64) >> (64 - w),
                        );
                        let r = match cmp {
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
                        };
                        next.push((*d, if *or { r || c } else { r && c }));
                        next.sort();
                    }
                }
                let decided = match &i.op {
                    Op::Bra {
                        cond: None,
                        any: false,
                        tid0_falls: false,
                        div: None,
                        ..
                    }
                    | Op::Exit
                    | Op::Trap { .. } => guard,
                    _ => None,
                };
                match (decided, &i.op) {
                    (Some(true), Op::Bra { target, .. }) => stack.push((*target, next, rnext)),
                    (Some(true), _) => {}
                    (Some(false), _) => stack.push((pc + 1, next, rnext)),
                    (None, _) => stack.extend(
                        cfg.succ[pc]
                            .iter()
                            .map(|&q| (q, next.clone(), rnext.clone())),
                    ),
                }
            }
            if hit {
                return true;
            }
            used.sort_by_key(|r| (r.file as u8, r.idx));
            used.dedup();
            let bad: Vec<_> = used.into_iter().filter(|r| written.contains(r)).collect();
            if bad.is_empty() {
                return false;
            }
            unknown.extend(bad);
        }
    }
}

/// What each entity was seen to touch, per pc: the synchronization footprint (spaces 3-5, 7,
/// 9-12) of every action of a warp that started at that pc, and of every completion of an
/// async operation issued there, plus what a split blocked at that pc waits on
/// (`blocked_sync`). Polls are widened to every phase of their mbarrier. Gathered over every
/// transition and state of the search.
type Sets = HashMap<(Ent, u32), Footprint>;

type FpRange = (u8, u32, u64, u64);

type Choice = (Ent, Vec<FpRange>, Vec<FpRange>);

/// The synchronization reduction's knowledge, shared by the worker threads. The sets only
/// grow; each growth publishes a new copy of them (rare after the first few thousand
/// states), which workers pick up through `version` (`LearnedView`).
#[derive(Default)]
struct Learned {
    /// `reach_sets` with cut 0..4.
    reach: [Vec<Vec<u64>>; 4],
    /// `cfg_succ` with cut 0..4 (`gate_reach`).
    succ: [Vec<Vec<usize>>; 4],
    /// Bitset of the pcs of `mbarrier.init` (`forget_dead_sync`).
    inits: Vec<u64>,
    /// Multi-node: growth is broadcast to the other ranks (search `call`).
    net: Option<(Arc<crate::dist::Dist>, u64)>,
    /// Bumped, under `inner`, whenever a set grows.
    version: AtomicU64,
    inner: Mutex<LearnedInner>,
}

#[derive(Default)]
struct LearnedInner {
    by: Arc<Sets>,
    /// A set grew, after an ample choice, by something that conflicts with what a chosen
    /// action (of another entity) touched: that choice may have been wrong, and the search
    /// must run again with the larger sets.
    dirty: bool,
    /// The distinct (entity, footprint) of every ample choice so far.
    chosen: HashSet<Choice>,
}

/// What an ample choice relied on: its entity and footprint, with the mbarrier arrival ranges
/// of the parked warps it took as released by it (`sync_ample`) added as writes, so that
/// anyone else later seen arriving there conflicts with the choice.
fn choice_key(e: Ent, fp: &Footprint, deps: Vec<FpRange>) -> Choice {
    let mut w = fp.w.clone();
    w.extend(deps);
    w.sort_unstable();
    w.dedup();
    (e, fp.r.clone(), w)
}

/// `cur` grown by `fp`, if that changes it.
fn grown(cur: Option<&Footprint>, fp: &Footprint) -> Option<Footprint> {
    let mut g = cur.cloned().unwrap_or_default();
    g.union(&Footprint {
        r: fp.r.clone(),
        w: fp.w.clone(),
        all: fp.all,
        ..Default::default()
    });
    match cur {
        Some(c) if (&c.r, &c.w, c.all) == (&g.r, &g.w, g.all) => None,
        _ => Some(g),
    }
}

/// One worker's view of `Learned`: a snapshot of the sets, and the choices it knows are
/// recorded in `chosen`.
struct LearnedView<'a> {
    l: &'a Learned,
    by: Arc<Sets>,
    version: u64,
    chosen: HashSet<Choice>,
}

impl<'a> LearnedView<'a> {
    fn new(l: &'a Learned) -> Self {
        let g = l.inner.lock().unwrap();
        LearnedView {
            l,
            by: g.by.clone(),
            version: l.version.load(Ordering::Acquire),
            chosen: HashSet::new(),
        }
    }

    fn refresh(&mut self) {
        if self.l.version.load(Ordering::Acquire) != self.version {
            let g = self.l.inner.lock().unwrap();
            self.by = g.by.clone();
            self.version = self.l.version.load(Ordering::Acquire);
        }
    }

    fn learn(&mut self, e: Ent, pc: u32, fp: &Footprint) {
        // A poll's phase key depends on the phase, which the chosen action may change (by
        // completing it): what a future poll of M can read is every phase of M.
        let mut fp = fp.clone();
        for x in fp.r.iter_mut().filter(|x| x.0 == 12) {
            x.2 &= !0xf_ffff;
            x.3 = x.2 + (1 << 20);
        }
        fp.normalize();
        let fp = &fp;
        // The sets only grow: if the snapshot already covers `fp`, so do the current sets.
        if fp.is_empty() || grown(self.by.get(&(e, pc)), fp).is_none() {
            return;
        }
        let mut g = self.l.inner.lock().unwrap();
        // Drop our snapshot so that `make_mut` copies the sets only if another worker holds one.
        self.by = Default::default();
        if let Some(cur) = grown(g.by.get(&(e, pc)), fp) {
            if let Some((d, call)) = &self.l.net {
                d.broadcast(&crate::dist::Msg {
                    call: *call,
                    round: 0,
                    body: crate::dist::Body::Learn(encode_learn(e, pc, &cur)),
                });
            }
            Arc::make_mut(&mut g.by).insert((e, pc), cur);
            if !g.dirty {
                g.dirty = g.chosen.iter().any(|(c, r, w)| {
                    *c != e
                        && fp.conflicts(&Footprint {
                            r: r.clone(),
                            w: w.clone(),
                            ..Default::default()
                        })
                });
            }
            self.l.version.fetch_add(1, Ordering::AcqRel);
        }
        self.by = g.by.clone();
        self.version = self.l.version.load(Ordering::Acquire);
    }

    /// `sync_ample` on the current sets, recording the choice. A choice already recorded needs
    /// no lock: a set that grows after our snapshot was taken is checked against it by `learn`.
    /// A new one is (re)made and recorded atomically with respect to `learn`.
    fn ample(
        &mut self,
        m: &Machine,
        acts: &[(Action, String)],
        fps: &[Footprint],
    ) -> Option<usize> {
        self.refresh();
        let (i, deps) = sync_ample(m, acts, fps, &self.by, &self.l.reach, &self.l.succ)?;
        let Some(e) = action_ent(m, &acts[i].0) else {
            return Some(i);
        };
        let key = choice_key(e, &fps[i], deps);
        if self.chosen.contains(&key) {
            return Some(i);
        }
        let mut g = self.l.inner.lock().unwrap();
        let (i, key) = if self.l.version.load(Ordering::Acquire) != self.version {
            self.by = g.by.clone();
            self.version = self.l.version.load(Ordering::Acquire);
            let (i, deps) = sync_ample(m, acts, fps, &self.by, &self.l.reach, &self.l.succ)?;
            match action_ent(m, &acts[i].0) {
                Some(e) => (i, choice_key(e, &fps[i], deps)),
                None => return Some(i),
            }
        } else {
            (i, key)
        };
        g.chosen.insert(key.clone());
        self.chosen.insert(key);
        Some(i)
    }
}

/// The synchronization footprint of the next action of each blocked split (a disabled
/// process's next transition, as in DPOR): what may have disabled it, or will enable it. An
/// action of another entity that disables it (completes the phase it polls, writes the flag it
/// spins on, takes the TMEM it waits for) conflicts with this, although the disabled action
/// itself is never executed after it.
fn blocked_sync(m: &Machine, all: &[(Action, String)]) -> Vec<(Ent, u32, Footprint)> {
    use ptxv_ir::Op;
    use ptxv_ir::Space;

    use crate::exec::Loc;
    let mut out = Vec::new();
    for (w, pc, members) in m.splits() {
        if all
            .iter()
            .any(|(a, _)| matches!(a, Action::Split { warp, pc: p, .. } if *warp == w && *p == pc))
        {
            continue;
        }
        let mut f = Footprint {
            sync_only: true,
            ..Default::default()
        };
        // A blocked warp-level poll (spin::wpoll_loops) reads its flags.
        if let Some(&b) = m.wpolls.get(&pc) {
            for (_, _, loc, bytes, _) in
                crate::spin::wpoll_dry(m, &members, pc, b).unwrap_or_default()
            {
                match loc {
                    Loc::Shared { cta, off } => {
                        f.add(10, cta as u32, off as u64, off as u64 + bytes as u64, false)
                    }
                    Loc::Global { alloc, off } => {
                        f.add(11, alloc as u32, off, off + bytes as u64, false)
                    }
                    _ => {}
                }
            }
        }
        match &m.k.code[pc as usize].op {
            Op::MbarTryWait { addr, .. } => {
                for &t in &members {
                    if let Ok(Loc::Shared { cta, off }) = m
                        .addr(t, addr)
                        .and_then(|a| m.resolve(t, Space::Shared, a, 8))
                    {
                        let base = (off as u64) << 20;
                        f.add(12, cta as u32, base, base + (1 << 20), false);
                    }
                }
            }
            // A blocked poll, strong or plain (spin::poll_loops), or a CAS poll.
            Op::Ld {
                space, bytes, addr, ..
            }
            | Op::Atom {
                space, bytes, addr, ..
            } => {
                for &t in &members {
                    match m
                        .addr(t, addr)
                        .and_then(|a| m.resolve(t, *space, a, *bytes as usize))
                    {
                        Ok(Loc::Shared { cta, off }) => f.add(
                            10,
                            cta as u32,
                            off as u64,
                            off as u64 + *bytes as u64,
                            false,
                        ),
                        Ok(Loc::Global { alloc, off }) => {
                            f.add(11, alloc as u32, off, off + *bytes as u64, false)
                        }
                        _ => {}
                    }
                }
            }
            Op::TmemAlloc { .. } | Op::TmemAllocRaw { .. } => {
                f.add(3, m.threads[members[0]].cta as u32, 0, 1, true)
            }
            _ => {}
        }
        f.normalize();
        out.push((Ent::Warp(w), pc, f));
    }
    out
}

/// The pc an action starts at, as `Learned` keys it.
fn action_pc(m: &Machine, a: &Action) -> u32 {
    match a {
        Action::Split { pc, .. } => *pc,
        Action::Async(i) => m.pending[*i].pc,
        _ => 0,
    }
}

/// Warps that cannot take any step before warp `w` arrives at a barrier: every live thread
/// is parked in a barrier.cluster wait of a generation `w` has not arrived at (same
/// cluster), or in an all-threads `bar.sync` that `w` has not arrived at (same CTA). `w`'s
/// next action comes before its arrival, so nothing such a warp will do can precede it.
fn frozen_by(m: &Machine, w: u32) -> HashSet<u32> {
    let mine: Vec<usize> = (0..m.threads.len())
        .filter(|&t| m.threads[t].warp == w && m.threads[t].st != St::Exited)
        .collect();
    let Some(&t0) = mine.first() else {
        return HashSet::new();
    };
    let my_cta = m.threads[t0].cta;
    let my_cid = m.launch.cluster_of(my_cta as u32).0;
    let blocked_on_me = |u: usize| -> bool {
        let th = &m.threads[u];
        match th.st {
            St::ClusterWait => {
                m.launch.cluster_of(th.cta as u32).0 == my_cid
                    && mine
                        .iter()
                        .all(|&t| m.threads[t].cl_arrived != th.cl_arrived)
            }
            St::Bar => {
                th.cta == my_cta
                    && m.bars.iter().any(|((c, _), b)| {
                        *c == my_cta
                            && b.count.is_none()
                            && b.arrived.binary_search(&(u as u32)).is_ok()
                            && mine
                                .iter()
                                .all(|&t| b.arrived.binary_search(&(t as u32)).is_err())
                    })
            }
            _ => false,
        }
    };
    let mut ok: HashMap<u32, bool> = HashMap::new();
    for (u, th) in m.live_threads() {
        if th.st == St::Exited || th.warp == w {
            continue;
        }
        let e = ok.entry(th.warp).or_insert(true);
        *e = *e && blocked_on_me(u);
    }
    ok.into_iter().filter(|(_, f)| *f).map(|(w, _)| w).collect()
}

static EMPTY_BARS: Vec<(u32, u64)> = Vec::new();

/// Synchronization reduction (persistent sets, ASSUME(E-DRF)): a whole-warp split `a` whose
/// synchronization footprint conflicts with nothing any other entity can still do is
/// expanded alone. "Can still do": what the other warps were seen to do at pcs reachable from
/// where their threads are, and what async operations were seen to do when issued at such a
/// pc or at the pc of one already pending (`a`'s own warp's async operations included). Those
/// actions then commute with `a` (plain memory and TMEM conflicts are races, found by vector
/// clocks in either order), and nothing they do enables or disables it, so every terminal
/// state, deadlock and race is still reached. The sets are learned during the search;
/// `explore_one` repeats the search until no set grows after a reduction relied on them.
/// Executed footprints alone would not do: a step that `a` disables (a poll of the phase `a`
/// completes) never runs after `a`, so blocked splits contribute what they wait on, and polls
/// count for every phase (ASSUMPTIONS.md, "Synchronization reduction"). Mbarrier
/// polls conflict only with the completion of the phase they observe (space 12), so a
/// consumer's poll is independent of a producer that will not complete that phase again.
fn sync_ample(
    m: &Machine,
    acts: &[(Action, String)],
    fps: &[Footprint],
    by: &Sets,
    reach: &[Vec<Vec<u64>>; 4],
    succ: &[Vec<Vec<usize>>; 4],
) -> Option<(usize, Vec<FpRange>)> {
    // Per warp and cut (`reach_sets`): the pcs its live threads can still reach, and the
    // issue pcs of its pending async operations.
    let words = reach[0].first().map_or(0, |v| v.len());
    let mut fut: HashMap<u32, [Vec<u64>; 4]> = HashMap::new();
    let mut wcta: HashMap<u32, u16> = HashMap::new();
    for (_, t) in m.live_threads().filter(|(_, t)| t.st != St::Exited) {
        wcta.insert(t.warp, t.cta);
        let f = fut
            .entry(t.warp)
            .or_insert_with(|| std::array::from_fn(|_| vec![0; words]));
        for (c, fc) in f.iter_mut().enumerate() {
            for (a, b) in fc.iter_mut().zip(&reach[c][t.pc as usize]) {
                *a |= *b;
            }
        }
    }
    let mut pend: HashSet<(u32, u32)> = HashSet::new();
    for o in &m.pending {
        pend.insert((m.threads[o.issuer as usize].warp, o.pc));
    }
    // ASSUME(E-DRF): a warp's steps past a barrier that the candidate warp still has to
    // arrive at cannot come before the candidate's step: an all-thread bar.sync of its own
    // CTA (a running candidate has not arrived at any), a barrier.cluster wait of its cluster
    // when the candidate's threads have not arrived at the current generation.
    let mut not_arrived: HashMap<u32, bool> = HashMap::new();
    for (_, t) in m.live_threads().filter(|(_, t)| t.st != St::Exited) {
        *not_arrived.entry(t.warp).or_insert(true) &= t.cl_arrived.is_none();
    }
    let cut_for = |cand: u32, e: u32| -> usize {
        let (Some(&ca), Some(&ea)) = (wcta.get(&cand), wcta.get(&e)) else {
            return 0;
        };
        let same_cluster = m.launch.cluster_of(ca as u32).0 == m.launch.cluster_of(ea as u32).0;
        let cl = same_cluster && not_arrived.get(&cand).copied().unwrap_or(false);
        let bs = ca == ea;
        (cl as usize) | ((bs as usize) << 1)
    };
    let live = |cand: u32, e: Ent, pc: u32| -> bool {
        let hit = |w: u32| {
            let c = cut_for(cand, w);
            fut.get(&w)
                .is_some_and(|f| f[c][pc as usize / 64] >> (pc % 64) & 1 == 1)
        };
        match e {
            Ent::Warp(w) => hit(w),
            Ent::Async(w) => hit(w) || pend.contains(&(w, pc)),
        }
    };
    // ASSUME(E-DRF): phase-aware footprints (`PTXV_NO_PHASE_CLIP` turns this off). The learned
    // footprints merge every phase of an mbarrier, and every operation issued at a pc. A
    // conflict found with them is re-checked against what the entity can still do from this
    // state:
    // - mbarrier phases only grow until the next init / inval of that mbarrier, so the
    //   future arrivals (space 7 reads), polls and phase completions (space 12) of an mbarrier
    //   whose current phase is k touch keys >= k, and future completions write arrival keys
    //   >= k + 1. Not for an mbarrier that some live footprint may still (re)initialize (a
    //   write of all its phase keys), nor one not materialized yet (`mbars`);
    // - an async unit whose warp can no longer issue at that pc (or whose warp is the
    //   candidate's: it issues nothing before the candidate) only completes the operations
    //   already pending there, which arrive at the mbarriers they name.
    let clip_on = std::env::var("PTXV_NO_PHASE_CLIP").is_err();
    let reinit: HashSet<(u32, u64)> = if clip_on {
        let live0 = |e: Ent, q: u32| -> bool {
            let hit = |w: u32| {
                fut.get(&w)
                    .is_some_and(|f| f[0][q as usize / 64] >> (q % 64) & 1 == 1)
            };
            match e {
                Ent::Warp(w) => hit(w),
                Ent::Async(w) => hit(w) || pend.contains(&(w, q)),
            }
        };
        by.iter()
            .filter(|((e, q), _)| live0(*e, *q))
            .flat_map(|(_, f)| {
                f.w.iter()
                    .filter(|x| x.0 == 15)
                    .flat_map(|x| (x.2..x.3.min(x.2 + 4096)).map(move |b| (x.1, b)))
            })
            .collect()
    } else {
        HashSet::new()
    };
    let mut pend_bars: HashMap<(u32, u32), Vec<(u32, u64)>> = HashMap::new();
    for o in &m.pending {
        if let AsyncKind::Clc { others, mbar, .. } = &o.kind {
            pend_bars
                .entry((m.threads[o.issuer as usize].warp, o.pc))
                .or_default()
                .extend(others.iter().map(|&c| (c as u32, *mbar as u64)));
        }
        let bar = match &o.kind {
            AsyncKind::Tma { mbar_cta, mbar, .. } => Some((*mbar_cta as u32, *mbar as u64)),
            AsyncKind::Commit { cta, mbar }
            | AsyncKind::Clc { cta, mbar, .. }
            | AsyncKind::Remote { cta, mbar, .. }
            | AsyncKind::CpArrive { cta, mbar, .. } => Some((*cta as u32, *mbar as u64)),
            AsyncKind::Mma { .. } | AsyncKind::Cp { .. } | AsyncKind::TmaStore { .. } => None,
        };
        let e = pend_bars
            .entry((m.threads[o.issuer as usize].warp, o.pc))
            .or_default();
        e.extend(bar);
    }
    // Mbarriers whose phase a refinement relied on: recorded with the choice as writes of
    // their (re)initialization key (space 15), so that an init / inval of one learned later
    // forces another round.
    let clipped: std::cell::RefCell<Vec<(u32, u64)>> = Default::default();
    // `f` (learned for entity `e` at pc `q`) as it can still occur, for candidate warp `cand`
    // (u32::MAX: an async candidate).
    let refine_with =
        |cand: u32, e: Ent, q: u32, f: &Footprint, over: Option<&Vec<(u32, u64)>>| -> Footprint {
            let only_pending: Option<&Vec<(u32, u64)>> = match e {
                _ if over.is_some() => over,
                Ent::Async(w) => {
                    let c = cut_for(cand, w);
                    let issue = w != cand
                        && fut
                            .get(&w)
                            .is_some_and(|f| f[c][q as usize / 64] >> (q % 64) & 1 == 1);
                    if issue {
                        None
                    } else {
                        Some(pend_bars.get(&(w, q)).unwrap_or(&EMPTY_BARS))
                    }
                }
                Ent::Warp(_) => None,
            };
            let mut g = Footprint {
                all: f.all,
                ..Default::default()
            };
            for (write, items) in [(false, &f.r), (true, &f.w)] {
                for &(sp, id, lo, hi) in items {
                    if sp != 7 && sp != 12 || hi <= lo {
                        g.add(sp, id, lo, hi, write);
                        continue;
                    }
                    let (b0, b1) = (lo >> 20, (hi - 1) >> 20);
                    if b1 - b0 > 64 {
                        g.add(sp, id, lo, hi, write);
                        continue;
                    }
                    for b in b0..=b1 {
                        let (sl, sh) = (lo.max(b << 20), hi.min((b + 1) << 20));
                        if only_pending.is_some_and(|v| !v.contains(&(id, b))) {
                            continue;
                        }
                        let phase = m
                            .mbars
                            .get(&(id as u16, b as u32))
                            .map(|x| x.phase as u64)
                            .filter(|_| !reinit.contains(&(id, b)));
                        if phase.is_some() {
                            clipped.borrow_mut().push((id, b));
                        }
                        let lo2 = match phase {
                            Some(k) => {
                                let kmin = k + (sp == 7 && write) as u64;
                                if kmin > 0xf_ffff {
                                    sl
                                } else {
                                    sl.max((b << 20) | kmin)
                                }
                            }
                            None => sl,
                        };
                        if lo2 < sh {
                            g.add(sp, id, lo2, sh, write);
                        }
                    }
                }
            }
            g.normalize();
            g
        };
    let refine = |cand: u32, e: Ent, q: u32, f: &Footprint| refine_with(cand, e, q, f, None);
    // ASSUME(E-DRF, E-SPIN): gated reachability (`PTXV_NO_GATE` turns it off). What a warp
    // can do before the candidate warp's step, refined by the mbarrier polls on its way: a
    // blocking poll (not a peek) whose mbarrier, parity and guard are fixed (read from
    // registers no instruction it may execute before then writes, so they keep their current
    // values), whose awaited phase has not completed, and whose mbarrier no one can arrive
    // at, initialize or invalidate before the candidate, stops the warp there. "No one":
    // no learned footprint of a live entity names it (space 7, or space 15), where a warp is
    // live at the pcs of its gated region (the candidate's warp: never, its next step is the
    // candidate), an async unit at the issue pcs in its warp's gated region (not the
    // candidate's) and at its pending operations (their own mbarriers). Least fixpoint:
    // regions grow as mbarriers become completable. The mbarriers that stopped a warp are
    // recorded with the choice (arrival and init keys), so that an arrival there learned
    // later forces another round. Argument: take the first step, in an execution from here
    // without the candidate, that leaves a gated region; it passes a stopping poll, so that
    // mbarrier completed a phase, so someone arrived at (or re-initialized) it earlier, at a
    // pc in its region: then the mbarrier was completable.
    let gate_on = std::env::var("PTXV_NO_GATE").is_err();
    let valtrack_on = std::env::var("PTXV_NO_VALTRACK").is_err();
    let la_debug = std::env::var("PTXV_LA_DEBUG").is_ok();
    // Per warp: the pcs of its gated region, and the polls it stops at; and the mbarriers
    // that stopped some warp.
    type Gate = (HashMap<u32, (Vec<u64>, Vec<u64>)>, Vec<(u32, u64)>);
    // Keyed by the candidate: its warp, or (u32::MAX, its pending-op index).
    let gates: std::cell::RefCell<HashMap<(u32, usize), std::rc::Rc<Gate>>> = Default::default();
    let gate_used: std::cell::RefCell<Vec<(u32, u64)>> = Default::default();
    let bars_of = |f: &Footprint, only: Option<&Vec<(u32, u64)>>, out: &mut HashSet<(u32, u64)>| {
        for &(sp, id, lo, hi) in f.r.iter().chain(&f.w) {
            let (b0, b1) = match sp {
                7 if hi > lo => (lo >> 20, (hi - 1) >> 20),
                15 if hi > lo => (lo, hi - 1),
                _ => continue,
            };
            if b1 - b0 > 4096 {
                // Too wide to list: every mbarrier of that CTA.
                out.insert((id, u64::MAX));
                continue;
            }
            for b in b0..=b1 {
                if only.is_none_or(|v| v.contains(&(id, b))) {
                    out.insert((id, b));
                }
            }
        }
    };
    // `cand`: the candidate's warp (u32::MAX for an async candidate); `skip`: the candidate's
    // pending-op index (an async candidate), which cannot come before itself.
    let compute_gate = |cand: u32, skip: usize| -> Gate {
        let words = reach[0].first().map_or(0, |v| v.len());
        // Live warps other than the candidate's, with the pcs their threads are at.
        let mut starts: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut threads_of: HashMap<u32, Vec<usize>> = HashMap::new();
        for (t, th) in m.live_threads() {
            if th.st == St::Exited || th.warp == cand {
                continue;
            }
            starts.entry(th.warp).or_default().push(th.pc);
            threads_of.entry(th.warp).or_default().push(t);
        }
        // The mbarriers of the pending operations, but the candidate.
        let mut pbars: HashMap<(u32, u32), Vec<(u32, u64)>> = HashMap::new();
        for (k, o) in m.pending.iter().enumerate() {
            let e = pbars
                .entry((m.threads[o.issuer as usize].warp, o.pc))
                .or_default();
            if k == skip {
                continue;
            }
            if let AsyncKind::Clc { others, mbar, .. } = &o.kind {
                e.extend(others.iter().map(|&c| (c as u32, *mbar as u64)));
            }
            match &o.kind {
                AsyncKind::Tma { mbar_cta, mbar, .. } => e.push((*mbar_cta as u32, *mbar as u64)),
                AsyncKind::Commit { cta, mbar }
                | AsyncKind::Clc { cta, mbar, .. }
                | AsyncKind::Remote { cta, mbar, .. }
                | AsyncKind::CpArrive { cta, mbar, .. } => e.push((*cta as u32, *mbar as u64)),
                AsyncKind::Mma { .. } | AsyncKind::Cp { .. } | AsyncKind::TmaStore { .. } => {}
            }
        }
        let mut completable: HashSet<(u32, u64)> = HashSet::new();
        loop {
            let mut regions: HashMap<u32, (Vec<u64>, Vec<u64>)> = HashMap::new();
            let mut stopped: Vec<(u32, u64)> = Vec::new();
            for (&w, pcs) in &starts {
                let c = cut_for(cand, w);
                let ts = &threads_of[&w];
                let mut unknown: HashSet<ptxv_ir::Reg> = HashSet::new();
                loop {
                    let mut seen = vec![0u64; words];
                    let mut stops = vec![0u64; words];
                    let mut written: HashSet<ptxv_ir::Reg> = HashSet::new();
                    let mut used: Vec<ptxv_ir::Reg> = Vec::new();
                    let mut stop_here: Vec<(u32, u64)> = Vec::new();
                    let mut stack: Vec<usize> = pcs.iter().map(|&p| p as usize).collect();
                    while let Some(x) = stack.pop() {
                        if x >= m.k.code.len() || seen[x / 64] >> (x % 64) & 1 == 1 {
                            continue;
                        }
                        seen[x / 64] |= 1 << (x % 64);
                        let inst = &m.k.code[x];
                        if let ptxv_ir::Op::MbarTryWait {
                            addr,
                            parity,
                            bit31,
                            ..
                        } = &inst.op
                        {
                            if !m.peeks.contains(&(x as u32))
                                || m.bar_cfg.peekwait.contains(&(x as u32))
                            {
                                let mut regs = inst.op.src_regs();
                                regs.extend(inst.guard.reg);
                                if regs.iter().all(|r| !unknown.contains(r)) {
                                    used.extend(regs);
                                    // The mbarrier every thread would wait on there, if all
                                    // would block.
                                    let mut bars: Vec<(u32, u64)> = Vec::new();
                                    let blocks = ts.iter().all(|&t| {
                                        if !m.prd(t, inst.guard) {
                                            return false;
                                        }
                                        let Ok(a) = m.addr(t, addr) else {
                                            return false;
                                        };
                                        let Ok(crate::exec::Loc::Shared { cta, off }) =
                                            m.resolve(t, ptxv_ir::Space::Shared, a, 8)
                                        else {
                                            return false;
                                        };
                                        let Ok(p) = m.rd(t, parity) else {
                                            return false;
                                        };
                                        let p = if *bit31 { (p >> 31) & 1 } else { p & 1 } as u32;
                                        let key = (cta as u32, off as u64);
                                        let ok = m.mbars.get(&(cta, off)).is_some_and(|mb| {
                                            mb.phase % 2 == p
                                                && !completable.contains(&key)
                                                && !completable.contains(&(key.0, u64::MAX))
                                        });
                                        if ok {
                                            bars.push(key);
                                        }
                                        ok
                                    });
                                    if blocks {
                                        stops[x / 64] |= 1 << (x % 64);
                                        stop_here.extend(bars);
                                        continue;
                                    }
                                }
                            }
                        }
                        written.extend(inst.op.dst_regs());
                        stack.extend(succ[c][x].iter().copied());
                    }
                    let bad: Vec<ptxv_ir::Reg> =
                        used.into_iter().filter(|r| written.contains(r)).collect();
                    if bad.is_empty() {
                        regions.insert(w, (seen, stops));
                        stopped.extend(stop_here);
                        break;
                    }
                    unknown.extend(bad);
                }
            }
            // Mbarriers someone live can arrive at (or re-initialize) before the candidate.
            let mut newc: HashSet<(u32, u64)> = HashSet::new();
            let at = |w: u32, q: u32| {
                regions.get(&w).is_some_and(|(r, st)| {
                    r[q as usize / 64] >> (q % 64) & 1 == 1
                        && st[q as usize / 64] >> (q % 64) & 1 == 0
                })
            };
            for ((e, q), f) in by.iter() {
                match *e {
                    Ent::Warp(w) => {
                        if at(w, *q) {
                            bars_of(f, None, &mut newc);
                        }
                    }
                    Ent::Async(w) => {
                        if w != cand && at(w, *q) {
                            bars_of(f, None, &mut newc);
                        } else if let Some(v) = pbars.get(&(w, *q)) {
                            bars_of(f, Some(v), &mut newc);
                        }
                    }
                }
            }
            if newc.iter().all(|b| completable.contains(b)) {
                stopped.sort_unstable();
                stopped.dedup();
                return (regions, stopped);
            }
            completable.extend(newc);
        }
    };
    // ASSUME(E-DRF, E-SPIN, E-BSYNC): the value-tracking look-ahead (`lookahead`), computed
    // on demand per candidate and intersected with the gated regions (both sound): per warp,
    // the pcs it may execute (None: no opinion), and the mbarriers that held it back.
    let las: std::cell::RefCell<HashMap<(u32, usize), std::rc::Rc<LaResult>>> = Default::default();
    let compute_la = |cand: u32, skip: usize, g: &Gate| -> LaResult {
        let regions = &g.0;
        let wild_bars = |w: u32, out: &mut HashSet<(u32, u64)>| {
            let at = |q: u32| {
                regions
                    .get(&w)
                    .is_none_or(|(r, _)| r[q as usize / 64] >> (q % 64) & 1 == 1)
            };
            for ((e, q), f) in by.iter() {
                let mine = match *e {
                    Ent::Warp(x) => x == w,
                    Ent::Async(x) => x == w && w != cand,
                };
                if mine && at(*q) {
                    bars_of(f, None, out);
                }
            }
        };
        let pc_bars = |w: u32, pc: u32, out: &mut HashSet<(u32, u64)>| {
            for e in [Ent::Warp(w), Ent::Async(w)] {
                if let Some(f) = by.get(&(e, pc)) {
                    bars_of(f, None, out);
                }
            }
        };
        let la = lookahead(m, cand, skip, &cut_for, &wild_bars, &pc_bars);
        if la_debug {
            for y in &la.why {
                eprintln!("PORLA {y}");
            }
        }
        la
    };
    // Can entity `e` take a step at pc `q` before the candidate (gated)? A warp stopped at
    // a poll there takes no step there: the poll stays disabled until the candidate, and the
    // candidate does not disable it (its awaited phase is the incomplete one).
    let gate_live = |cand: u32, skip: usize, e: Ent, q: u32| -> bool {
        if !gate_on {
            return true;
        }
        let g = gates
            .borrow_mut()
            .entry((cand, skip))
            .or_insert_with(|| std::rc::Rc::new(compute_gate(cand, skip)))
            .clone();
        let at = |w: u32| {
            g.0.get(&w).is_none_or(|(r, st)| {
                r[q as usize / 64] >> (q % 64) & 1 == 1 && st[q as usize / 64] >> (q % 64) & 1 == 0
            })
        };
        let pending = |w: u32| {
            m.pending
                .iter()
                .enumerate()
                .any(|(k, o)| k != skip && o.pc == q && m.threads[o.issuer as usize].warp == w)
        };
        let mut live = match e {
            Ent::Warp(w) => w != cand && at(w),
            Ent::Async(w) => pending(w) || (w != cand && at(w)),
        };
        let mut la_used = None;
        if live && valtrack_on {
            let w = match e {
                Ent::Warp(w) | Ent::Async(w) => w,
            };
            if w != cand && !(matches!(e, Ent::Async(_)) && pending(w)) {
                let la = las
                    .borrow_mut()
                    .entry((cand, skip))
                    .or_insert_with(|| std::rc::Rc::new(compute_la(cand, skip, &g)))
                    .clone();
                if let Some(Some(ex)) = la.exec.get(&w) {
                    if ex[q as usize / 64] >> (q % 64) & 1 == 0 {
                        live = false;
                        la_used = Some(la.clone());
                        if la_debug {
                            eprintln!("PORLA prune {e:?} at {}", m.k.code[q as usize].loc);
                        }
                    }
                }
            }
        }
        if !live {
            gate_used.borrow_mut().extend(g.1.iter().copied());
            if let Some(la) = la_used {
                gate_used.borrow_mut().extend(la.stopped.iter().copied());
            }
        }
        live
    };
    // ASSUME(E-DRF, E-SPIN, E-BSYNC): upper phase clipping (with the look-ahead). Before a
    // warp candidate's step, mbarrier B completes at most C more phases (`LaResult::bound`):
    // arrivals, polls and completions by anyone else touch phase keys at most phase + C.
    // The mbarriers this relied on are recorded with the choice (arrival and init keys), as
    // for the gate: an arrival there learned later forces another round.
    let upper = |cand: u32, f: Footprint| -> Footprint {
        if !valtrack_on || !gate_on || cand == u32::MAX {
            return f;
        }
        let la = {
            let g = gates
                .borrow_mut()
                .entry((cand, usize::MAX))
                .or_insert_with(|| std::rc::Rc::new(compute_gate(cand, usize::MAX)))
                .clone();
            las.borrow_mut()
                .entry((cand, usize::MAX))
                .or_insert_with(|| std::rc::Rc::new(compute_la(cand, usize::MAX, &g)))
                .clone()
        };
        let mut out = Footprint {
            all: f.all,
            ..Default::default()
        };
        let mut used = Vec::new();
        for (write, items) in [(false, &f.r), (true, &f.w)] {
            for &(sp, id, lo, hi) in items {
                if sp != 7 && sp != 12 || hi <= lo || (lo >> 20) != ((hi - 1) >> 20) {
                    out.add(sp, id, lo, hi, write);
                    continue;
                }
                let b = lo >> 20;
                let Some(mb) = m.mbars.get(&(id as u16, b as u32)) else {
                    out.add(sp, id, lo, hi, write);
                    continue;
                };
                if reinit.contains(&(id, b)) {
                    out.add(sp, id, lo, hi, write);
                    continue;
                }
                let c = la.bound(m, (id, b));
                let top = (mb.phase as u64).saturating_add(c);
                if top >= 0xf_ffff {
                    out.add(sp, id, lo, hi, write);
                    continue;
                }
                // Completing phase j writes its phase key j (space 12) and the arrival key
                // j + 1 (space 7); arrivals and polls in phase j read key j.
                let hi2 = hi.min((b << 20) | (top + (!(write && sp == 12)) as u64));
                if hi2 < hi {
                    used.push((id, b));
                }
                if lo < hi2 {
                    out.add(sp, id, lo, hi2, write);
                }
            }
        }
        if !used.is_empty() {
            out.normalize();
            gate_used.borrow_mut().extend(used);
        }
        out
    };
    // Does candidate footprint `a` conflict with what `e` may still do at `q` before the
    // candidate (`refine`, `upper`)?
    let conflicts = |cand: u32, a: &Footprint, e: Ent, q: u32, f: &Footprint| -> bool {
        a.conflicts(f)
            && (!clip_on || {
                let g = refine(cand, e, q, f);
                a.conflicts(&g) && a.conflicts(&upper(cand, g))
            })
    };
    let clip_deps = || -> Vec<FpRange> {
        let mut v: Vec<FpRange> = clipped
            .borrow()
            .iter()
            .map(|&(id, b)| (15, id, b, b + 1))
            .collect();
        for &(id, b) in gate_used.borrow().iter() {
            v.push((7, id, b << 20, (b + 1) << 20));
            v.push((15, id, b, b + 1));
        }
        v.sort_unstable();
        v.dedup();
        v
    };
    // Warps all of whose live threads are parked at an mbarrier poll (spin model: blocked
    // until the phase completes), with the mbarriers they poll.
    let mut parked: HashMap<u32, Vec<(u16, u32)>> = HashMap::new();
    {
        let mut not_parked: HashSet<u32> = HashSet::new();
        for (w, pc, members) in m.splits() {
            let enabled = m.split_enabled(&members, pc);
            let mut bars = Vec::new();
            if let (false, ptxv_ir::Op::MbarTryWait { addr, .. }) =
                (enabled, &m.k.code[pc as usize].op)
            {
                for &t in &members {
                    match m
                        .addr(t, addr)
                        .and_then(|a| m.resolve(t, ptxv_ir::Space::Shared, a, 8))
                    {
                        Ok(crate::exec::Loc::Shared { cta, off }) => bars.push((cta, off)),
                        _ => {
                            not_parked.insert(w);
                        }
                    }
                }
            } else {
                not_parked.insert(w);
            }
            parked.entry(w).or_default().extend(bars);
        }
        for (_, t) in m.live_threads() {
            if t.st != St::Exited && t.st != St::Run {
                not_parked.insert(t.warp);
            }
        }
        parked.retain(|w, b| !not_parked.contains(w) && !b.is_empty());
        for b in parked.values_mut() {
            b.sort_unstable();
            b.dedup();
        }
    }
    let arrives_at = |f: &Footprint, (cta, off): (u16, u32)| {
        let (lo, hi) = ((off as u64) << 20, (off as u64 + 1) << 20);
        f.all
            || f.r
                .iter()
                .chain(&f.w)
                .any(|x| x.0 == 7 && x.1 == cta as u32 && x.2 < hi && lo < x.3)
    };
    // ASSUME(E-DRF): parked warps that only the candidate's side can release. A warp parked
    // at a poll of B takes no step before B's current phase completes; if no entity outside
    // `allowed` (the candidate's own later work) can still arrive at B, that comes after the
    // candidate, and so does everything the parked warp will do. Returns those warps and the
    // mbarrier arrival ranges the judgement relied on (recorded with the choice, so that a
    // later-learned arrival there by anyone else triggers another round).
    let released_by =
        |cand: u32, me: Ent, allowed: &dyn Fn(Ent, u32) -> bool| -> (HashSet<u32>, Vec<FpRange>) {
            let mut ws = HashSet::new();
            let mut rel = Vec::new();
            if std::env::var("PTXV_NO_RELEASED").is_ok() {
                return (ws, rel);
            }
            for (x, bars) in &parked {
                if me == Ent::Warp(*x) {
                    continue;
                }
                let ok = bars.iter().all(|&b| {
                    !by.iter().any(|((e, q), f)| {
                        *e != Ent::Warp(*x)
                            && !allowed(*e, *q)
                            && live(cand, *e, *q)
                            && arrives_at(f, b)
                            && (!clip_on || arrives_at(&refine(cand, *e, *q, f), b))
                    })
                });
                if ok {
                    ws.insert(*x);
                    for &(cta, off) in bars {
                        rel.push((7, cta as u32, (off as u64) << 20, (off as u64 + 1) << 20));
                    }
                }
            }
            (ws, rel)
        };
    // A tcgen05 pipe op (MMA, commit) completes in issue order within its pipe.
    let fifo_pc = |q: u32| {
        matches!(
            m.k.code[q as usize].op,
            ptxv_ir::Op::Mma { .. } | ptxv_ir::Op::TcCp { .. } | ptxv_ir::Op::TcCommit { .. }
        )
    };
    let mut deps: Vec<FpRange> = Vec::new();
    let pos = acts.iter().enumerate().position(|(i, (a, _))| {
        // The completion of a tcgen05 pipe op, when it is its entity's only enabled action:
        // the entity's later pipe completions come after it (in-order pipe), so only other
        // entities, and this entity's non-pipe ops, can conflict.
        if let Action::Async(j) = a {
            let op = &m.pending[*j];
            if fps[i].all {
                return false;
            }
            let me = Ent::Async(m.threads[op.issuer as usize].warp);
            // ASSUME(E-DRF): the completion of a TMA load (data into shared memory, then
            // complete_tx on its mbarrier) as a candidate (`PTXV_NO_TMA_POR` turns it off).
            // The entity's other pending operations complete in any order, so they are
            // checked like any other entity's (as `refine` sees them: their own mbarriers,
            // phases from here on); this operation itself is not one of them.
            if op.fifo.is_none() {
                // ASSUME(S-REMOTE-INFLIGHT): a remote arrival / st.async delivery likewise (data,
                // then arrivals on its mbarrier; its exit-key read conflicts with the target's
                // last exit).
                if !matches!(op.kind, AsyncKind::Tma { .. } | AsyncKind::Remote { .. })
                    || !clip_on
                    || std::env::var("PTXV_NO_TMA_POR").is_ok()
                {
                    return false;
                }
                let w0 = m.threads[op.issuer as usize].warp;
                // Parked warps only this completion can release: no other operation pending
                // at its pc, and none can still be issued there.
                let alone = !m.pending.iter().enumerate().any(|(k, o)| {
                    k != *j && o.pc == op.pc && m.threads[o.issuer as usize].warp == w0
                }) && !fut.get(&w0).is_some_and(|f| {
                    f[cut_for(u32::MAX, w0)][op.pc as usize / 64] >> (op.pc % 64) & 1 == 1
                });
                let pc0 = op.pc;
                let (released, rel) =
                    released_by(u32::MAX, me, &|e, q| alone && e == me && q == pc0);
                let bad = by.iter().find(|((e, q), f)| {
                    if !(!matches!(e, Ent::Warp(w) if released.contains(w))
                        && live(u32::MAX, *e, *q)
                        && fps[i].conflicts(f))
                    {
                        return false;
                    }
                    if !gate_live(u32::MAX, *j, *e, *q) {
                        return false;
                    }
                    let g = if *e == me && *q == op.pc {
                        // The other operations pending at this pc.
                        let mut others: Vec<(u32, u64)> = Vec::new();
                        let mut issue = false;
                        for (k, o) in m.pending.iter().enumerate() {
                            if k != *j && o.pc == op.pc && m.threads[o.issuer as usize].warp == w0 {
                                match &o.kind {
                                    AsyncKind::Tma { mbar_cta, mbar, .. } => {
                                        others.push((*mbar_cta as u32, *mbar as u64))
                                    }
                                    _ => issue = true,
                                }
                            }
                        }
                        let c = cut_for(u32::MAX, w0);
                        issue |= fut
                            .get(&w0)
                            .is_some_and(|f| f[c][op.pc as usize / 64] >> (op.pc % 64) & 1 == 1);
                        if issue {
                            refine(u32::MAX, *e, *q, f)
                        } else if others.is_empty() {
                            return false;
                        } else {
                            refine_with(u32::MAX, *e, *q, f, Some(&others))
                        }
                    } else {
                        refine(u32::MAX, *e, *q, f)
                    };
                    fps[i].conflicts(&g)
                });
                if let Some(((e, q), f)) = bad {
                    if std::env::var("PTXV_POR_DEBUG").is_ok() {
                        eprintln!(
                            "PORA {me:?} {} blocked by {e:?} at {}: a={:?} {:?} / other r={:?} w={:?}",
                            m.k.describe(op.pc as usize),
                            m.k.describe(*q as usize),
                            fps[i].r,
                            fps[i].w,
                            f.r,
                            f.w
                        );
                    }
                }
                if bad.is_none() {
                    deps = rel;
                    deps.extend(clip_deps());
                }
                clipped.borrow_mut().clear();
                gate_used.borrow_mut().clear();
                return bad.is_none();
            }
            if acts
                .iter()
                .filter(|(b, _)| action_ent(m, b) == Some(me))
                .count()
                > 1
            {
                return false;
            }
            let (released, _) = released_by(u32::MAX, me, &|e, q| e == me && fifo_pc(q));
            let bad = by.iter().find(|((e, q), f)| {
                !(*e == me && fifo_pc(*q))
                    && !matches!(e, Ent::Warp(w) if released.contains(w))
                    && live(u32::MAX, *e, *q)
                    && conflicts(u32::MAX, &fps[i], *e, *q, f)
                    && gate_live(u32::MAX, *j, *e, *q)
            });
            if let Some(((e, q), f)) = bad {
                if std::env::var("PTXV_POR_DEBUG").is_ok() {
                    eprintln!(
                        "PORA {me:?} {} blocked by {e:?} at {}: a={:?} {:?} / other r={:?} w={:?}",
                        m.k.describe(op.pc as usize),
                        m.k.describe(*q as usize),
                        fps[i].r,
                        fps[i].w,
                        f.r,
                        f.w
                    );
                }
            }
            if bad.is_none() {
                deps = released_by(u32::MAX, me, &|e, q| e == me && fifo_pc(q)).1;
                deps.extend(clip_deps());
            }
            clipped.borrow_mut().clear();
            gate_used.borrow_mut().clear();
            return bad.is_none();
        }
        let Action::Split { warp, pc, leader } = a else {
            return false;
        };
        // One action for this warp (an elect with several leader choices is not one).
        if leader.is_some()
            && acts
                .iter()
                .filter(|(b, _)| matches!(b, Action::Split { warp: w2, .. } if w2 == warp))
                .count()
                > 1
        {
            return false;
        }
        // (Lanes parked at a BSYNC that only this split releases come after it, like the
        // warp's own later steps.)
        let whole = whole_or_parked(m, *warp, *pc);
        if !whole || fps[i].all {
            return false;
        }
        let me = Ent::Warp(*warp);
        let frozen = frozen_by(m, *warp);
        let w0 = *warp;
        let mine = |e: Ent, q: u32| e == Ent::Async(w0) && !pend.contains(&(w0, q));
        let (released, rel) = released_by(w0, me, &mine);
        // The candidate's own warp issues no async operation before the candidate runs (it is
        // the warp's next step): of its async units, only the operations already pending can
        // come first (`PTXV_NO_OWN_ASYNC`: all reachable issue pcs, as before).
        let own_async = std::env::var("PTXV_NO_OWN_ASYNC").is_err();
        let bad = by.iter().find(|((e, q), f)| {
            *e != me
                && !(own_async && mine(*e, *q))
                && !matches!(e, Ent::Warp(w) if frozen.contains(w) || released.contains(w))
                && live(*warp, *e, *q)
                && conflicts(*warp, &fps[i], *e, *q, f)
                && gate_live(*warp, usize::MAX, *e, *q)
        });
        if bad.is_none() {
            deps = rel;
            deps.extend(clip_deps());
        }
        clipped.borrow_mut().clear();
        gate_used.borrow_mut().clear();
        if let Some(((e, q), f)) = bad {
            if std::env::var("PTXV_POR_DEBUG").is_ok() {
                let at: Vec<String> = match e {
                    Ent::Warp(w) | Ent::Async(w) => {
                        let mut v: Vec<String> = m
                            .live_threads()
                            .filter(|(_, t)| t.warp == *w && t.st != St::Exited)
                            .map(|(_, t)| format!("{:x}{:?}", m.k.code[t.pc as usize].loc, t.st))
                            .collect();
                        v.dedup();
                        v
                    }
                };
                if valtrack_on && gate_on {
                    if let Some(la) = las.borrow().get(&(*warp, usize::MAX)) {
                        let w = match e {
                            Ent::Warp(w) | Ent::Async(w) => *w,
                        };
                        let ex = la.exec.get(&w).map(|x| {
                            x.as_ref().map(|v| v[*q as usize / 64] >> (*q % 64) & 1 == 1)
                        });
                        let why: Vec<&String> =
                            la.why.iter().filter(|y| y.starts_with(&format!("w{w}"))).collect();
                        let mut bs = HashSet::new();
                        for x in fps[i].r.iter().chain(&fps[i].w).chain(&f.r).chain(&f.w) {
                            if (x.0 == 7 || x.0 == 12) && x.3 > x.2 && (x.3 - x.2) < 1 << 20 {
                                bs.insert((x.1, x.2 >> 20));
                            }
                        }
                        let bounds: Vec<String> = bs
                            .iter()
                            .map(|&k| {
                                format!(
                                    "{:x?}:ph{:?}+{}",
                                    k,
                                    m.mbars.get(&(k.0 as u16, k.1 as u32)).map(|x| x.phase),
                                    la.bound(m, k)
                                )
                            })
                            .collect();
                        eprintln!(
                            "PORLAI exec={ex:?} why={why:?} bounds={bounds:?} wild={:?} inf={:x?} {:?} arr={:x?}",
                            la.why, la.inf, la.inf_cta, la.arr
                        );
                    }
                }
                eprintln!(
                    "POR w{warp} {} blocked by {e:?}{at:?} at {}: a={:?} {:?} / other r={:?} w={:?}",
                    m.k.describe(*pc as usize),
                    m.k.describe(*q as usize),
                    fps[i].r,
                    fps[i].w,
                    f.r,
                    f.w
                );
            }
        }
        bad.is_none()
    });
    pos.map(|i| (i, deps))
}

pub fn explore(m0: Machine, max_states: usize) -> Verdict {
    let note = crate::residency::reliance_note(&m0);
    let mut v = crate::smem::two_pass(m0, |m| explore_slice(m, max_states, None));
    v.notes.extend(note);
    v
}

/// Clusters (CTAs without clusters) of a single-grid launch, by id.
pub fn clusters(m: &Machine) -> Vec<u32> {
    let mut v: Vec<u32> = (0..m.launch.ctas())
        .map(|c| m.launch.cluster_of(c).0)
        .collect();
    // Ids repeat out of order when clusters span grid rows (cluster y > 1).
    v.sort_unstable();
    v.dedup();
    v
}

/// `explore`, restricted to one cluster when `cluster` is Some: every other CTA counts as
/// already exited and never runs. Sound for the whole launch only together with
/// `cross_slice_conflicts` (E-CTA-INDEP): see `ptxv check --slices`.
pub fn explore_slice(m0: Machine, max_states: usize, cluster: Option<u32>) -> Verdict {
    let mut m0 = m0;
    if let Some(c) = cluster {
        // The grids before this cluster's run too: it starts from what they leave in memory
        // (a multi-kernel launch, rung 3a: the GEMM reads the workspace a prepare kernel
        // wrote). Their accesses are ordered before the slice's by the grid boundary.
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
    }
    let m0 = clc_eager(m0);
    let mut v = per_leader_policy(m0, threads(max_states), |m, n| {
        explore_one(m, max_states, n)
    });
    // Which trace finds a bug first depends on the schedule of the search threads; the
    // order of the findings should not.
    v.findings
        .sort_by(|a, b| (&a.kind, &a.sig).cmp(&(&b.kind, &b.sig)));
    v
}

/// ASSUME(E-CTA-INDEP): exhaustive exploration of a launch whose clusters are independent,
/// one cluster at a time (`explore_slice` on a fresh machine from `make` each), then the
/// cross-cluster check: no global byte that one slice wrote is read or written by another
/// (`GFoot`, every explored transition). With that, the slices' schedules interleave
/// freely: every terminal state, deadlock and race of the launch is one of a slice, or a
/// cross-slice conflict. `per_slice` sees each slice's verdict as it finishes.
pub fn explore_sliced(
    make: &dyn Fn() -> Machine,
    max_states: usize,
    per_slice: &mut dyn FnMut(u32, &Verdict),
) -> Verdict {
    let m = make();
    let mut ids = clusters(&m);
    drop(m);
    // Debugging: explore only these clusters (comma-separated ids); the verdict then does
    // not cover the launch.
    if let Ok(only) = std::env::var("PTXV_SLICE_ONLY") {
        let keep: Vec<u32> = only.split(',').filter_map(|x| x.parse().ok()).collect();
        ids.retain(|c| keep.contains(c));
    }
    let t0 = Instant::now();
    let mut all = Verdict {
        states: 0,
        transitions: 0,
        complete: true,
        terminal_states: 0,
        distinct_outputs: 0,
        findings: vec![],
        assumptions: vec![],
        notes: vec![format!(
            "explored one cluster at a time ({} clusters)",
            ids.len()
        )],
        seconds: 0.0,
        outputs: vec![],
    };
    type Bits = Vec<(Vec<u64>, Vec<u64>, Vec<u64>)>;
    // Multi-node: slices are independent searches, so rank r explores slices r, r + N, ...
    // on its own (`dist::set_local`), and rank 0 gathers verdicts and footprints.
    let net = crate::dist::get();
    let mine: Vec<u32> = match &net {
        Some(d) => ids
            .iter()
            .enumerate()
            .filter(|(i, _)| i % d.size == d.rank)
            .map(|(_, c)| *c)
            .collect(),
        None => ids.clone(),
    };
    let mut done: Vec<(u32, Verdict, Bits)> = Vec::new();
    crate::dist::set_local(net.is_some());
    for &c in &mine {
        let mut m = make();
        let sizes: Vec<usize> = m.gmem.iter().map(|g| g.bytes.len()).collect();
        let mut g = crate::machine::GFoot::new(&sizes);
        g.cluster = Some(c);
        let g = Arc::new(g);
        m.gfoot = Some(g.clone());
        let v = explore_slice(m, max_states, Some(c));
        per_slice(c, &v);
        done.push((c, v, g.snapshot()));
    }
    crate::dist::set_local(false);
    const GATHER: u64 = u64::MAX;
    if let Some(d) = &net {
        let sink = Arc::new(crate::dist::MailSink::default());
        d.set_round(GATHER, 0, Some(sink.clone()));
        if d.rank != 0 {
            let b = serde_json::to_vec(&done).expect("slice report");
            d.send(
                0,
                Msg {
                    call: GATHER,
                    round: 0,
                    body: Body::Report(b),
                },
            );
            let b = sink.0.wait(|q| {
                q.get(&0).and_then(|v| {
                    v.iter().find_map(|x| match x {
                        Body::Decision(b) => Some(b.clone()),
                        _ => None,
                    })
                })
            });
            d.set_round(GATHER, 0, None);
            return serde_json::from_slice(&b).expect("slice decision");
        }
        let others: Vec<Vec<u8>> = sink.0.wait(|q| {
            let got: Vec<Vec<u8>> = (1..d.size)
                .filter_map(|r| {
                    q.get(&r)?.iter().find_map(|x| match x {
                        Body::Report(b) => Some(b.clone()),
                        _ => None,
                    })
                })
                .collect();
            (got.len() == d.size - 1).then_some(got)
        });
        for b in others {
            let theirs: Vec<(u32, Verdict, Bits)> =
                serde_json::from_slice(&b).expect("slice report");
            for (c, v, f) in theirs {
                per_slice(c, &v);
                done.push((c, v, f));
            }
        }
        done.sort_by_key(|x| x.0);
    }
    let mut foots: Vec<(u32, Bits)> = Vec::new();
    for (c, v, f) in done {
        all.states += v.states;
        all.transitions += v.transitions;
        all.complete &= v.complete;
        all.terminal_states += v.terminal_states;
        all.distinct_outputs = all.distinct_outputs.max(v.distinct_outputs);
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
        foots.push((c, f));
    }
    // Grid `j` starts after grid `i` completed: stream order on one GPU, no PDL in between.
    let launch = make().launch.clone();
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
    // ASSUME(N-ATOMIC): fp32 atomic adds (no returned value) of two clusters into one byte
    // do not conflict; that byte's final value then depends on their order.
    let mut red_shared: Option<(usize, u64)> = None;
    for (i, (ci, fi)) in foots.iter().enumerate() {
        for (cj, fj) in &foots[i + 1..] {
            let (gi, gj) = ((ci >> 20) as usize, (cj >> 20) as usize);
            if gi != gj && (ordered(gi, gj) || ordered(gj, gi)) {
                continue;
            }
            for (alloc, ((ri, wi, xi), (rj, wj, xj))) in fi.iter().zip(fj).enumerate() {
                let hit = |w: &Vec<u64>, o1: &Vec<u64>, o2: &Vec<u64>, o3: &Vec<u64>| {
                    w.iter().enumerate().find_map(|(k, &x)| {
                        let g = |o: &Vec<u64>| o.get(k).copied().unwrap_or(0);
                        let y = x & (g(o1) | g(o2) | g(o3));
                        (y != 0).then(|| 64 * k as u64 + y.trailing_zeros() as u64)
                    })
                };
                let none = Vec::new();
                if red_shared.is_none() {
                    red_shared = hit(xi, xj, &none, &none).map(|b| (alloc, b));
                }
                if let Some(b) = hit(wi, rj, wj, xj)
                    .or_else(|| hit(wj, ri, wi, xi))
                    .or_else(|| hit(xi, rj, &none, &none))
                    .or_else(|| hit(xj, ri, &none, &none))
                {
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
            }
        }
    }
    if let Some((a, b)) = red_shared {
        // Several clusters' reduction orders: more than one output in general.
        all.distinct_outputs = all.distinct_outputs.max(2);
        all.notes.push(format!(
            "global byte {b:#x} of allocation {a} receives fp32 atomic adds from several clusters: its final value depends on their order (N-ATOMIC)"
        ));
    }
    all.findings
        .sort_by(|a, b| (&a.kind, &a.sig).cmp(&(&b.kind, &b.sig)));
    all.seconds = t0.elapsed().as_secs_f64();
    if let Some(d) = &net {
        if net.is_some() && d.size > 1 {
            let b = serde_json::to_vec(&all).expect("slice decision");
            d.broadcast(&Msg {
                call: GATHER,
                round: 0,
                body: Body::Decision(b),
            });
            d.set_round(GATHER, 0, None);
        }
    }
    all
}

/// ASSUME(E-CTA-INDEP), cluster symmetry: explore one cluster and let it stand for all, when
/// one deterministic run of the whole launch shows that
/// - no guard, branch, shared / TMEM / mbarrier address or other control operand depends on
///   `%ctaid` (position taint, the input-data taint with `%ctaid` as its source; global
///   addresses and TMA coordinates may depend on it), and
/// - no global byte that one actor writes is accessed by another actor at all (so no two
///   accesses of a cluster, or of two clusters, can conflict; addresses do not depend on the
///   schedule, because they do not depend on loaded data: the input taint).
///
/// Then every cluster takes the same steps in every schedule as cluster 0, with its own
/// global addresses, so cluster 0's exhaustive verdict (races, deadlocks, determinism) is every
/// cluster's. Otherwise falls back to `explore_sliced`.
pub fn explore_symmetric(
    make: &dyn Fn() -> Machine,
    max_states: usize,
    per_slice: &mut dyn FnMut(u32, &Verdict),
) -> Verdict {
    crate::smem::two_pass_make(make, &mut |mk| {
        explore_symmetric_one(mk, max_states, per_slice)
    })
}

fn explore_symmetric_one(
    make: &dyn Fn() -> Machine,
    max_states: usize,
    per_slice: &mut dyn FnMut(u32, &Verdict),
) -> Verdict {
    // Debugging: PTXV_SLICE_ONLY skips the symmetry check (see explore_sliced).
    if std::env::var("PTXV_SLICE_ONLY").is_ok() {
        return explore_sliced(make, max_states, per_slice);
    }
    let t0 = Instant::now();
    let mut m = clc_eager(make());
    let mut ns = crate::num::NumState::with_mma_eps(m.launch.mma_model.eps());
    ns.taint.position = true;
    m.num = Some(Box::new(ns));
    let sizes: Vec<usize> = m.gmem.iter().map(|g| g.bytes.len()).collect();
    m.gown = Some(Box::new(crate::machine::GOwn::new(&sizes)));
    let ids = clusters(&m);
    let (m, run_findings, ok) = run_once(m);
    let mut why: Vec<String> = Vec::new();
    if !ok {
        why.push("the deterministic run did not terminate".into());
    }
    if let Some(ns) = m.num.as_ref() {
        why.extend(ns.taint.violations.iter().cloned());
        // Corresponding threads (CTA rank in the cluster, thread index in the CTA) of every
        // cluster must have used the same position-dependent control values.
        // Per (CTA rank, thread index): its position-taint trace.
        type Traces<'a> = BTreeMap<(u32, u32), &'a [(u32, u64)]>;
        let mut by_cluster: BTreeMap<u32, Traces> = BTreeMap::new();
        for (t, th) in m.threads.iter().enumerate() {
            let (cid, rank) = m.launch.cluster_of(th.cta as u32);
            let b = m.launch.g(th.cta as u32).block;
            let lin = th.tid[0] + b[0] * (th.tid[1] + b[1] * th.tid[2]);
            let tr = ns.taint.trace.get(&(t as u32)).map_or(&[][..], |v| &v[..]);
            by_cluster.entry(cid).or_default().insert((rank, lin), tr);
        }
        let mut it = by_cluster.iter();
        if let Some((c0, first)) = it.next() {
            for (c, other) in it {
                if let Some(((rank, lin), _)) =
                    first.iter().find(|(key, tr)| other.get(key) != Some(tr))
                {
                    let pc = first[&(*rank, *lin)]
                        .iter()
                        .zip(other.get(&(*rank, *lin)).copied().unwrap_or(&[]).iter())
                        .find(|(a, b)| a != b)
                        .map(|(a, _)| m.k.describe(a.0 as usize))
                        .unwrap_or_else(|| "(trace length)".into());
                    why.push(format!(
                        "cluster {c} decides differently from cluster {c0} (CTA rank {rank}, thread {lin}, first at {pc})"
                    ));
                    break;
                }
            }
        }
    }
    if let Some((a, o)) = m.gown.as_ref().and_then(|g| g.conflict()) {
        why.push(format!(
            "global byte {o:#x} of allocation {a} is written by one actor and accessed by another"
        ));
    }
    if ids.len() < 2 || !why.is_empty() {
        for w in &why {
            eprintln!("clusters not symmetric: {w}");
        }
        let mut v = explore_sliced(make, max_states, per_slice);
        for w in why {
            v.notes.push(format!("clusters not symmetric: {w}"));
        }
        return v;
    }
    let mut v = explore_slice(make(), max_states, Some(ids[0]));
    per_slice(ids[0], &v);
    // The slice's outputs hold only cluster ids[0]'s part of the launch's output: they are not
    // launch outputs (none are kept, as in `explore_sliced`; distinct_outputs counts the slice's).
    v.outputs.clear();
    for f in run_findings {
        if !v.findings.iter().any(|x| x.sig == f.sig) {
            v.findings.push(f);
        }
    }
    v.findings
        .sort_by(|a, b| (&a.kind, &a.sig).cmp(&(&b.kind, &b.sig)));
    v.notes.push(format!(
        "cluster {} explored; it stands for all {} clusters (position-independent control, exclusive global writes)",
        ids[0],
        ids.len()
    ));
    v.seconds = t0.elapsed().as_secs_f64();
    v
}

/// Progress on stderr while exploring: one line every few seconds (states, terminal states,
/// frontier, rate, ETA to the state cap, finding kinds so far). PTXV_QUIET=1 turns it off.
pub(crate) struct Progress {
    label: String,
    t0: Instant,
    last: Instant,
    last_states: usize,
    calls: u32,
    on: bool,
    max_states: usize,
    /// The deepest state expanded (stateful explorer; 0: not shown).
    depth: usize,
}

impl Progress {
    pub(crate) fn new(label: String, max_states: usize) -> Self {
        let now = Instant::now();
        Progress {
            label,
            t0: now,
            last: now,
            last_states: 0,
            calls: 0,
            on: std::env::var("PTXV_QUIET").is_err(),
            max_states,
            depth: 0,
        }
    }

    pub(crate) fn tick(
        &mut self,
        states: usize,
        terminal: usize,
        frontier: usize,
        findings: &[Finding],
    ) {
        self.calls += 1;
        if !self.calls.is_multiple_of(256) {
            return;
        }
        self.report(states, terminal, frontier, findings);
    }

    /// Is a progress line due?
    pub(crate) fn due(&self) -> bool {
        self.on && self.last.elapsed().as_secs_f64() >= 5.0
    }

    pub(crate) fn report(
        &mut self,
        states: usize,
        terminal: usize,
        frontier: usize,
        findings: &[Finding],
    ) {
        if !self.on {
            return;
        }
        let now = Instant::now();
        let dt = now.duration_since(self.last).as_secs_f64();
        if dt < 5.0 {
            return;
        }
        let rate = (states - self.last_states) as f64 / dt;
        let total = now.duration_since(self.t0).as_secs_f64();
        let eta = if rate > 0.0 {
            let left = self.max_states.saturating_sub(states) as f64 / rate;
            format!("{:.0}m to the {} cap", left / 60.0, self.max_states)
        } else {
            "-".into()
        };
        let mut kinds: Vec<&str> = findings.iter().map(|f| f.kind.as_str()).collect();
        kinds.sort();
        kinds.dedup();
        eprintln!(
            "  [{}] {:.0}s  states={states} terminal={terminal} frontier={frontier}{}  {rate:.0}/s  eta {eta}  rss={:.1}G  findings={kinds:?}",
            self.label,
            total,
            if self.depth > 0 {
                format!(" depth={}", self.depth)
            } else {
                String::new()
            },
            rss_bytes() as f64 / (1u64 << 30) as f64
        );
        self.last = now;
        self.last_states = states;
    }
}

/// Resident set size of this process (Linux `/proc/self/status`; 0 elsewhere), for the
/// progress line.
fn rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            let l = s.lines().find(|l| l.starts_with("VmRSS:"))?;
            l.split_whitespace().nth(1)?.parse::<u64>().ok()
        })
        .map_or(0, |kb| kb * 1024)
}

/// ASSUME(E-DRF): forget synchronization bookkeeping no future step can read
/// (`PTXV_NO_DEAD_SYNC` turns it off), so that states that differ only in it are one state:
/// - an mbarrier's `atomics` (the arrivals' epochs) are read only by a later
///   `mbarrier.init` of it (init-vs-arrival race check): cleared once no live thread can
///   reach any `mbarrier.init` pc;
/// - a named barrier's completed-generation count and last release clock (`bar_gen`,
///   `bar_prev`) are read only by later arrivals at it, and its (empty) generation record
///   by later arrivals or by exits releasing a generation in progress: dropped once no live
///   thread of its CTA may still arrive at it (`BarCfg::reach`) and no generation is in
///   progress.
///
/// Reachability is the control-flow graph's (E-RET, E-CONST-BRA), from every live thread's
/// pc (unlaunched ones from their entry). Nothing else reads these fields (exit checks,
/// outputs and deadlock reports do not), so the merged states have the same futures,
/// terminal outputs and findings.
fn forget_dead_sync(m: &mut Machine, reach0: &[Vec<u64>], inits: &[u64]) {
    if reach0.is_empty() || std::env::var("PTXV_NO_DEAD_SYNC").is_ok() {
        return;
    }
    let words = inits.len();
    let mut pcs: Vec<u32> = Vec::new();
    let mut may_arrive: HashMap<u16, u32> = HashMap::new();
    for (_, t) in m.live_threads() {
        if t.st == St::Exited {
            continue;
        }
        pcs.push(t.pc);
        *may_arrive.entry(t.cta).or_insert(0) |= m
            .bar_cfg
            .reach
            .get(t.pc as usize)
            .copied()
            .unwrap_or(u32::MAX);
    }
    pcs.sort_unstable();
    pcs.dedup();
    let init_reachable = pcs.iter().any(|&pc| {
        reach0
            .get(pc as usize)
            .is_none_or(|r| (0..words).any(|w| r[w] & inits[w] != 0))
    });
    if !init_reachable && m.mbars.values().any(|b| !b.atomics.is_empty()) {
        for b in m.mbars.values_mut() {
            b.atomics.clear();
        }
    }
    let dead = |&(c, id): &(u16, u32)| -> bool {
        let bit = if id < 32 { 1u32 << id } else { u32::MAX };
        may_arrive.get(&c).copied().unwrap_or(0) & bit == 0
    };
    let idle = |m: &Machine, k: &(u16, u32)| {
        m.bars
            .get(k)
            .is_none_or(|b| b.arrived.is_empty() && b.arrivers.is_empty())
    };
    let gone: Vec<(u16, u32)> = m
        .bar_gen
        .keys()
        .chain(m.bar_prev.keys())
        .chain(m.bars.keys())
        .copied()
        .filter(|k| dead(k) && idle(m, k))
        .collect();
    for k in gone {
        m.bar_gen.remove(&k);
        m.bar_prev.remove(&k);
        m.bars.remove(&k);
    }
}

/// Debugging (`PTXV_CHECK_WALKS=N`): the end states (terminal or deadlocked) the search
/// reached, to compare with N random complete executions run without any reduction.
static CHECK_TERMINALS: Mutex<Option<HashSet<u128>>> = Mutex::new(None);

/// `PTXV_CHECK_WALKS=N`: after a complete search, run N random executions of `m0` (every
/// enabled action possible at each step, no reduction) and report any end state or finding
/// the search did not reach (an empirical check of the reductions).
/// `plain`: `m0` is the model without the explorer's peek changes (`peekwait_set`,
/// `peekskip_set`), whose end states differ from the search's in dead registers and in where
/// a deadlocked thread waits: then terminal outputs and findings are compared instead of end
/// states (and the look-ahead is not checked).
fn check_walks(m0: &Machine, v: &Verdict, threads: usize, by: &Sets, plain: bool) {
    let Some(n) = std::env::var("PTXV_CHECK_WALKS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    else {
        return;
    };
    let ends = CHECK_TERMINALS.lock().unwrap().clone().unwrap_or_default();
    // End states are compared after the same canonicalization the search applies.
    let por = sync_por_ok(m0);
    let reach0 = if por {
        reach_sets(&m0.k.code, 0, &m0.bar_cfg)
    } else {
        Vec::new()
    };
    let mut inits = vec![0u64; m0.k.code.len().div_ceil(64)];
    for (pc, i) in m0.k.code.iter().enumerate() {
        if matches!(i.op, ptxv_ir::Op::MbarInit { .. }) {
            inits[pc / 64] |= 1 << (pc % 64);
        }
    }
    let (reach0, inits) = (&reach0, &inits);
    let sigs: HashSet<String> = v.findings.iter().map(|f| f.sig.clone()).collect();
    let next = AtomicUsize::new(0);
    let missing = AtomicUsize::new(0);
    let new_sigs: Mutex<HashSet<String>> = Default::default();
    let valtrack = std::env::var("PTXV_NO_VALTRACK").is_err();
    let la_bad: Mutex<(usize, usize, usize, Option<String>)> = Default::default();
    let la_bad = &la_bad;
    std::thread::scope(|s| {
        for _ in 0..threads.clamp(1, 64) {
            s.spawn(|| {
                let like = m0.private_statics();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let mut r = (0x9e37_79b9_7f4a_7c15u64
                        ^ (i as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9))
                        | 1;
                    let mut m = m0.unshared(&like);
                    let mut fs: Vec<Finding> = Vec::new();
                    let mut la_checks = 0;
                    loop {
                        let acts = m.actions();
                        if acts.is_empty() {
                            break;
                        }
                        r ^= r << 13;
                        r ^= r >> 7;
                        r ^= r << 17;
                        let (a, _) = &acts[(r % acts.len() as u64) as usize];
                        if !plain && la_checks < 8 && valtrack && (r >> 20).is_multiple_of(64) {
                            la_checks += 1;
                            check_lookahead(&m, a, r, la_bad, by);
                        }
                        m.apply(a);
                        fs.append(&mut m.findings);
                    }
                    if plain {
                        if m.terminal() && !v.outputs.contains(&m.outputs()) {
                            missing.fetch_add(1, Ordering::Relaxed);
                        }
                    } else {
                        let mut c = m.clone();
                        forget_dead_sync(&mut c, reach0, inits);
                        if !ends.contains(&c.fingerprint()) {
                            missing.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    if m.terminal() {
                        m.exit_checks();
                        fs.append(&mut m.findings);
                    } else {
                        fs.push(Finding {
                            kind: "deadlock".into(),
                            sig: m.deadlock_report().0,
                            msg: String::new(),
                            trace: vec![],
                            ..Default::default()
                        });
                    }
                    for f in fs {
                        if !sigs.contains(&f.sig) {
                            new_sigs
                                .lock()
                                .unwrap()
                                .insert(format!("{}: {}", f.kind, f.sig));
                        }
                    }
                }
            });
        }
    });
    let new_sigs = new_sigs.into_inner().unwrap();
    if plain {
        eprintln!(
            "CHECKWALKS plain model (no peek changes), {n} random executions: {} terminal outputs not among the search's ({} distinct); findings it did not report: {:?}",
            missing.load(Ordering::Relaxed),
            v.outputs.len(),
            new_sigs
        );
        return;
    }
    if valtrack {
        let (n, held, bad, ex) = la_bad.lock().unwrap().clone();
        eprintln!(
            "CHECKWALKS look-ahead: {n} checks ({held} with a warp held back), {bad} violations{}",
            ex.map(|e| format!(" (first: {e})")).unwrap_or_default()
        );
    }
    eprintln!(
        "CHECKWALKS {n} random executions: {} end states the search had ({} distinct), {} not reached by the search; findings it did not report: {:?}",
        n - missing.load(Ordering::Relaxed),
        ends.len(),
        missing.load(Ordering::Relaxed),
        new_sigs
    );
}

/// `PTXV_CHECK_WALKS` (with the look-ahead on): from state `m` of a random execution, with
/// the warp of action `a` as the candidate, check the look-ahead's claim (`lookahead`, every
/// cut off and every arrival of a warp it gives up on unbounded) on a random continuation
/// without that warp: every step a tracked warp takes starts at a pc the look-ahead lets it
/// execute. Counts (checks, checks with a warp held back, violations, first violation) in
/// `bad`.
fn check_lookahead(
    m: &Machine,
    a: &Action,
    mut r: u64,
    bad: &Mutex<(usize, usize, usize, Option<String>)>,
    by: &Sets,
) {
    let Action::Split { warp: cand, .. } = *a else {
        return;
    };
    // The mbarriers a footprint arrives at or (re)initializes (as `bars_of`).
    let bars = |f: &Footprint, out: &mut HashSet<(u32, u64)>| {
        for &(sp, id, lo, hi) in f.r.iter().chain(&f.w) {
            let (b0, b1) = match sp {
                7 if hi > lo => (lo >> 20, (hi - 1) >> 20),
                15 if hi > lo => (lo, hi - 1),
                _ => continue,
            };
            if b1 - b0 > 4096 {
                out.insert((id, u64::MAX));
            } else {
                out.extend((b0..=b1).map(|b| (id, b)));
            }
        }
    };
    // The search's final learned footprints stand in for the gated regions: everything a
    // warp (or its async unit) was ever seen to do.
    let wild = |w: u32, out: &mut HashSet<(u32, u64)>| {
        for ((e, _), f) in by.iter() {
            if matches!(*e, Ent::Warp(x) | Ent::Async(x) if x == w) {
                bars(f, out);
            }
        }
    };
    let at_pc = |w: u32, pc: u32, out: &mut HashSet<(u32, u64)>| {
        for e in [Ent::Warp(w), Ent::Async(w)] {
            if let Some(f) = by.get(&(e, pc)) {
                bars(f, out);
            }
        }
    };
    let la = lookahead(m, cand, usize::MAX, &|_, _| 0, &wild, &at_pc);
    let mut x = m.clone();
    let mut viol = None;
    for _ in 0..2000 {
        let acts: Vec<Action> = x
            .actions()
            .into_iter()
            .map(|(a, _)| a)
            .filter(|a| !matches!(a, Action::Split { warp, .. } if *warp == cand))
            .collect();
        if acts.is_empty() {
            break;
        }
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        let a = &acts[(r % acts.len() as u64) as usize];
        if let Action::Split { warp, pc, .. } = *a {
            if let Some(Some(ex)) = la.exec.get(&warp) {
                if ex[pc as usize / 64] >> (pc % 64) & 1 == 0 {
                    viol = Some(format!(
                        "candidate warp {cand}: warp {warp} ran from {}",
                        m.k.describe(pc as usize)
                    ));
                    break;
                }
            }
        }
        x.apply(a);
        // Completions so far stay within the look-ahead's bounds (`upper`).
        if viol.is_none() {
            for (k, mb) in m.mbars.iter() {
                let key = (k.0 as u32, k.1 as u64);
                if let Some(now) = x.mbars.get(k) {
                    let done = now.phase.saturating_sub(mb.phase) as u64;
                    if now.phase >= mb.phase && done > la.bound(m, key) {
                        viol = Some(format!(
                            "candidate warp {cand}: mbarrier {key:x?} completed {done} phases (bound {})",
                            la.bound(m, key)
                        ));
                    }
                }
            }
        }
    }
    let mut b = bad.lock().unwrap();
    b.0 += 1;
    b.1 += !la.stopped.is_empty() as usize;
    if let Some(v) = viol {
        b.2 += 1;
        b.3.get_or_insert(v);
    }
}

fn explore_one(m0: Machine, max_states: usize, threads: usize) -> Verdict {
    let v = explore_one_model(m0.clone(), max_states, threads, true, 2);
    // A racy kernel's outcomes depend on orders of plain accesses that the reductions do
    // not keep (E-DRF); the peek changes remove branching that used to expose some of them.
    // So a search that finds a race is run again in the plain model. Likewise for
    // value-insensitive polls (E-POLL-ANY), which acquire nothing: a race found with them may
    // need an edge they dropped.
    let peek_model = std::env::var("PTXV_NO_PEEKWAIT").is_err()
        && (!peekwait_set(&m0).is_empty()
            || std::env::var("PTXV_NO_PEEKSKIP").is_err() && !peekskip_set(&m0).is_empty());
    let vip_model = m0.vip.is_none() && std::env::var("PTXV_NO_VIP").is_err();
    // Debugging: PTXV_NO_RACE_RERUN keeps the first search's verdict.
    if (peek_model || vip_model)
        && v.findings.iter().any(|f| f.kind == "race")
        && std::env::var("PTXV_NO_RACE_RERUN").is_err()
    {
        // First without the acquire loads among the value-insensitive polls, which is what
        // usually needed the edge; then without any of the changes.
        if vip_model {
            let w = explore_one_model(m0.clone(), max_states, threads, true, 1);
            if !w.findings.iter().any(|f| f.kind == "race") {
                let mut w = w;
                w.notes.push(format!(
                    "a race was found: explored again with value-insensitive polls (E-POLL-ANY) on relaxed loads only ({} states with acquire loads too)",
                    v.states
                ));
                return w;
            }
        }
        let mut w = explore_one_model(m0, max_states, threads, false, 0);
        w.notes.push(format!(
            "a race was found: explored again without the peek changes (E-SPIN) and value-insensitive polls (E-POLL-ANY) ({} states with them)",
            v.states
        ));
        return w;
    }
    v
}

fn explore_one_model(
    mut m0: Machine,
    max_states: usize,
    threads: usize,
    peeks: bool,
    // E-POLL-ANY: 0 off, 1 relaxed loads, 2 acquire loads too.
    vip: u8,
) -> Verdict {
    let t0 = Instant::now();
    // The original model, for the soundness walks (`check_walks`).
    let m_orig = m0.clone();
    if peeks && std::env::var("PTXV_NO_PEEKWAIT").is_err() {
        let pw = peekwait_set(&m0);
        if std::env::var("PTXV_POR_DEBUG").is_ok() {
            let locs: Vec<String> = pw
                .iter()
                .map(|&p| format!("{:#x}", m0.k.code[p as usize].loc))
                .collect();
            eprintln!(
                "PORPEEKWAIT {} of {} peeks: {locs:?}",
                pw.len(),
                m0.peeks.len()
            );
        }
        if !pw.is_empty() {
            let mut cfg = (*m0.bar_cfg).clone();
            cfg.peekwait = pw;
            m0.bar_cfg = Arc::new(cfg);
        }
        if std::env::var("PTXV_NO_PEEKSKIP").is_err() {
            let mut ps = peekskip_set(&m0);
            // Debugging: leave these peek locations (hex) out.
            if let Ok(v) = std::env::var("PTXV_PEEKSKIP_EXCLUDE") {
                let locs: Vec<u32> = v
                    .split(',')
                    .filter_map(|x| u32::from_str_radix(x.trim_start_matches("0x"), 16).ok())
                    .collect();
                ps.retain(|&p, _| !locs.contains(&m0.k.code[p as usize].loc));
            }
            if std::env::var("PTXV_POR_DEBUG").is_ok() {
                let locs: Vec<String> = ps
                    .iter()
                    .map(|(&p, (j, w))| {
                        let l = |x: &u32| format!("{:#x}", m0.k.code[*x as usize].loc);
                        format!(
                            "{} joins {:?} waits {:?}",
                            l(&p),
                            j.iter().map(l).collect::<Vec<_>>(),
                            w.iter().map(l).collect::<Vec<_>>()
                        )
                    })
                    .collect();
                eprintln!("PORPEEKSKIP {locs:?}");
            }
            if !ps.is_empty() {
                let mut cfg = (*m0.bar_cfg).clone();
                cfg.peekskip = ps;
                m0.bar_cfg = Arc::new(cfg);
            }
        }
    }
    if std::env::var("PTXV_CHECK_WALKS").is_ok() {
        *CHECK_TERMINALS.lock().unwrap() = Some(HashSet::new());
    }
    // E-POLL-ANY: value-insensitive polls, learned over the rounds (`PTXV_NO_VIP` off).
    if vip > 0 && m0.vip.is_none() && std::env::var("PTXV_NO_VIP").is_err() {
        m0.vip = Some(Arc::new(crate::machine::VipLearn {
            acquire: vip > 1 && std::env::var("PTXV_VIP_NO_ACQ").is_err(),
            ..Default::default()
        }));
    }
    let por = sync_por_ok(&m0);
    // Multi-node: every rank runs the same program, so the n-th search is the same on all.
    let net = crate::dist::get().map(|d| {
        let call = {
            let mut c = d.calls.lock().unwrap();
            *c += 1;
            *c
        };
        (d, call)
    });
    let learned = Arc::new(Learned {
        reach: if por {
            [0, 1, 2, 3].map(|c| reach_sets(&m0.k.code, c, &m0.bar_cfg))
        } else {
            Default::default()
        },
        succ: if por {
            [0, 1, 2, 3].map(|c| cfg_succ(&m0.k.code, c, &m0.bar_cfg.fold, &m0.bar_cfg.rets))
        } else {
            Default::default()
        },
        inits: {
            let mut v = vec![0u64; m0.k.code.len().div_ceil(64)];
            for (pc, i) in m0.k.code.iter().enumerate() {
                if matches!(i.op, ptxv_ir::Op::MbarInit { .. }) {
                    v[pc / 64] |= 1 << (pc % 64);
                }
            }
            v
        },
        net: net.clone(),
        ..Default::default()
    });
    if let Some((d, call)) = &net {
        d.set_learn(*call, Some(Arc::new(LearnSink(learned.clone()))));
    }
    let mut prev: Option<Verdict> = None;
    if por {
        seed_learned(&m0, &learned, threads);
    }
    for round in 1.. {
        let mut v = explore_round(
            m0.clone(),
            max_states,
            por.then_some(&*learned),
            round,
            threads,
            net.clone(),
        );
        // Every transition of every round is a real execution step: keep what earlier rounds
        // found.
        if let Some(p) = prev.take() {
            for f in p.findings {
                if !v.findings.iter().any(|g| g.sig == f.sig) {
                    v.findings.push(f);
                }
            }
            for o in p.outputs {
                if !v.outputs.contains(&o) {
                    v.outputs.push(o);
                }
            }
            v.distinct_outputs = v.outputs.len();
        }
        v.seconds = t0.elapsed().as_secs_f64();
        // E-POLL-ANY: a location or value learned this round that earlier decisions did not
        // know: run the round again (the last round decides with every value).
        let vip_dirty = m0
            .vip
            .as_ref()
            .is_some_and(|x| x.dirty.swap(false, Ordering::Relaxed));
        let mut g = learned.inner.lock().unwrap();
        if (!por || !g.dirty) && !(vip_dirty && v.complete) || !v.complete {
            if let Some((d, call)) = &net {
                d.set_learn(*call, None);
            }
            if v.complete {
                let by = g.by.clone();
                drop(g);
                check_walks(&m0, &v, threads, &by, false);
                if !m0.bar_cfg.peekwait.is_empty() || !m0.bar_cfg.peekskip.is_empty() {
                    check_walks(&m_orig, &v, threads, &by, true);
                }
            }
            return v;
        }
        g.dirty = false;
        g.chosen.clear();
        prev = Some(v);
    }
    unreachable!()
}

/// Seed the synchronization reduction's sets with random complete executions, so that the
/// first round rarely learns something new after relying on the sets (which costs a second
/// round). Deterministic (fixed seed). With one thread the walks draw from one random stream;
/// with more, each walk has its own (so they can run side by side: on large kernels the walks
/// take as long as the parallel search itself).
fn seed_learned(m0: &Machine, learned: &Learned, threads: usize) {
    let walks: usize = std::env::var("PTXV_POR_SEED_WALKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let seed: u64 = 0x9e37_79b9_7f4a_7c15;
    if threads == 1 {
        let (mut view, mut rng) = (LearnedView::new(learned), seed);
        for _ in 0..walks {
            seed_walk(m0.clone(), &mut view, &mut rng);
        }
        return;
    }
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.min(walks) {
            let next = &next;
            std::thread::Builder::new()
                .stack_size(STACK)
                .spawn_scoped(s, move || {
                    let like = m0.private_statics();
                    let mut view = LearnedView::new(learned);
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= walks {
                            break;
                        }
                        // splitmix64 of the walk index: independent, never-zero xorshift seeds.
                        let mut z = seed.wrapping_add((i as u64 + 1).wrapping_mul(seed));
                        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                        let mut r = (z ^ (z >> 31)) | 1;
                        seed_walk(m0.unshared(&like), &mut view, &mut r);
                    }
                })
                .expect("spawn explorer thread");
        }
    });
}

/// One random complete execution from `m`, learning the footprint of every step.
fn seed_walk(mut m: Machine, view: &mut LearnedView, rng: &mut u64) {
    for _ in 0..1_000_000 {
        let acts = m.actions();
        for (e, pc, f) in blocked_sync(&m, &acts) {
            view.learn(e, pc, &f);
        }
        if acts.is_empty() {
            break;
        }
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        let (a, _) = &acts[(*rng % acts.len() as u64) as usize];
        let (ent, pc) = (action_ent(&m, a), action_pc(&m, a));
        m.fp = Some(Box::new(Footprint {
            sync_only: true,
            ..Default::default()
        }));
        m.apply(a);
        m.findings.clear();
        let mut fp = m.fp.take().map(|b| *b).unwrap_or_default();
        fp.normalize();
        if let Some(e) = ent {
            view.learn(e, pc, &fp);
        }
    }
}

/// Fingerprints are already well mixed: hash a u128 by folding it.
#[derive(Default)]
struct FoldHasher(u64);

impl Hasher for FoldHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, b: &[u8]) {
        for &x in b {
            self.0 = (self.0.rotate_left(8) ^ x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        }
    }
    fn write_u128(&mut self, x: u128) {
        self.0 = (x as u64) ^ (x >> 64) as u64;
    }
}

/// The visited set, sharded by fingerprint so that threads rarely contend.
struct Visited {
    shards: Vec<Mutex<HashSet<u128, BuildHasherDefault<FoldHasher>>>>,
}

impl Visited {
    fn new(threads: usize) -> Self {
        let n = if threads == 1 { 1 } else { 4096 };
        Visited {
            shards: (0..n).map(|_| Default::default()).collect(),
        }
    }
    fn insert(&self, f: u128) -> bool {
        // The shard from the high bits; the in-shard hash uses all of them.
        let i = ((f >> 96) as usize) % self.shards.len();
        self.shards[i].lock().unwrap().insert(f)
    }
}

type Item = (Machine, Option<Arc<PathNode>>);

/// One worker's stack, padded to its own cache lines: the owner pushes and pops at the
/// back (depth first), idle workers steal the oldest (shallowest) state at the front. `len`
/// mirrors the length so that thieves skip empty stacks without taking their locks.
#[repr(align(128))]
#[derive(Default)]
struct Deque {
    q: Mutex<VecDeque<Item>>,
    len: AtomicUsize,
}

impl Deque {
    fn pop(&self, back: bool) -> Option<Item> {
        if self.len.load(Ordering::Relaxed) == 0 {
            return None;
        }
        let mut q = self.q.lock().unwrap();
        let it = if back { q.pop_back() } else { q.pop_front() };
        self.len.store(q.len(), Ordering::Relaxed);
        it
    }
    fn push(&self, items: Vec<Item>) {
        let mut q = self.q.lock().unwrap();
        q.extend(items);
        self.len.store(q.len(), Ordering::Relaxed);
    }
}

/// What a round's workers share.
struct Search<'a> {
    max_states: usize,
    indep: bool,
    stats: bool,
    /// `Machine::weak_stop` of the root: stop at the first race that sends the weak pass's
    /// verdict to the strong pass (smem.rs).
    weak_stop: u8,
    learned: Option<&'a Learned>,
    sh: Arc<Shared>,
    /// Multi-node: this rank's part of the round.
    dist: Option<Arc<DistRound>>,
    terminal: AtomicUsize,
    /// The deepest state expanded so far (steps from the initial state), for the progress
    /// line: a search whose depth keeps growing past any execution's length has a cycle
    /// that changes the state.
    depth: AtomicUsize,
    /// Debugging (`PTXV_DFS_ESTIMATE`, one worker only): per depth on the current path, how
    /// many new children its parent had (`done_estimate`).
    widths: Option<Mutex<Vec<u32>>>,
    found: Mutex<Found>,
    progress: Mutex<Progress>,
    hist: Mutex<HashMap<String, usize>>,
}

/// What a search shares with its multi-node message handler (`DistRound`).
struct Shared {
    visited: Visited,
    deques: Vec<Deque>,
    /// Distinct states found (the size of `visited`).
    states: AtomicUsize,
    /// States queued or being expanded (multi-node: also waiting for their owner's answer);
    /// the search is over when it drops to 0 (multi-node: on every rank, with no work in
    /// flight).
    pending: AtomicUsize,
    /// The state cap was hit.
    stop: AtomicBool,
}

impl std::ops::Deref for Search<'_> {
    type Target = Shared;
    fn deref(&self) -> &Shared {
        &self.sh
    }
}

/// Findings (first trace per sig) and distinct terminal outputs, in discovery order.
struct Found {
    findings: Vec<Finding>,
    seen_sig: HashSet<String>,
    outputs: Vec<Vec<u8>>,
    out_seen: HashSet<Vec<u8>>,
}

/// One worker's private state: what it already knows is recorded globally, and counters.
struct Local<'a> {
    me: usize,
    sigs: HashSet<String>,
    outs: HashSet<Vec<u8>>,
    view: Option<LearnedView<'a>>,
    transitions: usize,
    expanded: usize,
    steals: usize,
    rng: u64,
    /// Our own copies of the static parts of a machine (`Machine::unshared`).
    statics: Option<Machine>,
}

impl Search<'_> {
    /// Record a finding unless its sig is known; `path` is the trace to it.
    fn finding(&self, loc: &mut Local, mut f: Finding, path: &Option<Arc<PathNode>>) {
        if loc.sigs.contains(&f.sig) {
            return;
        }
        let mut g = self.found.lock().unwrap();
        if g.seen_sig.insert(f.sig.clone()) {
            f.trace = path_vec(path);
            loc.sigs.insert(f.sig.clone());
            let stop = self.weak_stop > 0
                && f.kind == "race"
                && crate::smem::strong_race(&f.sig, self.weak_stop == 2);
            g.findings.push(f);
            if stop {
                self.weak_stopped();
            }
        } else {
            loc.sigs.insert(f.sig);
        }
    }

    /// Debugging (`PTXV_DFS_ESTIMATE`): the fraction of the depth-first search done, if
    /// sibling subtrees were the same size (Knuth-style; deduplication makes later subtrees
    /// smaller, so the true fraction is larger). At depth d the path's state has `w[d]`
    /// children, `r[d]` of them still queued and one on the path.
    fn done_estimate(&self) -> (f64, String) {
        let w = self.widths.as_ref().unwrap().lock().unwrap().clone();
        let mut r = vec![0u32; w.len().max(1)];
        {
            let q = self.deques[0].q.lock().unwrap();
            for (_, p) in q.iter() {
                let d = path_depth(p) as usize;
                if d < r.len() {
                    r[d] += 1;
                }
            }
        }
        let (mut f, mut scale) = (0.0f64, 1.0f64);
        let mut top = Vec::new();
        for d in 1..w.len() {
            let c = w[d].max(1) as f64;
            let left = r[d] as f64;
            f += scale * ((c - left - 1.0).max(0.0) / c);
            scale /= c;
            if top.len() < 12 && w[d] > 1 {
                top.push(format!("d{d}:{}/{}", r[d], w[d]));
            }
        }
        (f, top.join(" "))
    }

    /// The weak pass found a race that the strong pass decides: its verdict will be
    /// discarded, so stop (multi-node: every rank, as rank 0 does at the state cap).
    fn weak_stopped(&self) {
        if !self.stop.swap(true, Ordering::AcqRel) {
            eprintln!("  weak LDS/STS pass: stopped at a race that the strong pass decides");
            if let Some(d) = &self.dist {
                d.d.broadcast(&d.msg(Body::Control(1)));
            }
        }
    }

    fn output(&self, loc: &mut Local, o: Vec<u8>) {
        if loc.outs.contains(&o) {
            return;
        }
        let mut g = self.found.lock().unwrap();
        if g.out_seen.insert(o.clone()) {
            g.outputs.push(o.clone());
        }
        loc.outs.insert(o);
    }

    /// The next state to expand: our own newest, else (a private copy of) the oldest of
    /// another worker.
    fn next(&self, loc: &mut Local) -> Option<Item> {
        if let Some(it) = self.deques[loc.me].pop(true) {
            return Some(it);
        }
        if let Some(it) = self.dist.as_ref().and_then(|d| d.inbox.pop(true)) {
            return Some(it);
        }
        let n = self.deques.len();
        loc.rng ^= loc.rng << 13;
        loc.rng ^= loc.rng >> 7;
        loc.rng ^= loc.rng << 17;
        let start = loc.rng as usize % n;
        for k in 0..n {
            let v = (start + k) % n;
            if v == loc.me {
                continue;
            }
            if let Some((m, path)) = self.deques[v].pop(false) {
                loc.steals += 1;
                // A deep copy of a whole-grid machine (slices: every other cluster exited)
                // costs more than the reference-count traffic it saves; it still gets our
                // own static parts (the kernel's count is touched on every step).
                let m = match &loc.statics {
                    Some(like) if m.threads.len() > 8192 => m.with_statics(like),
                    Some(like) => m.unshared(like),
                    None => m,
                };
                return Some((m, path));
            }
        }
        None
    }

    fn worker(&self, me: usize, m0: &Machine) -> usize {
        let mut loc = Local {
            me,
            sigs: HashSet::new(),
            outs: HashSet::new(),
            view: self.learned.map(LearnedView::new),
            transitions: 0,
            expanded: 0,
            steals: 0,
            statics: (self.deques.len() > 1).then(|| m0.private_statics()),
            rng: 0x9e37_79b9_7f4a_7c15 ^ (me as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9),
        };
        let mut idle = 0u32;
        let mut idle_since: Option<Instant> = None;
        let mut idle_time = 0.0;
        while !self.stop.load(Ordering::Relaxed) {
            match self.next(&mut loc) {
                Some((m, path)) => {
                    idle = 0;
                    if let Some(t) = idle_since.take() {
                        idle_time += t.elapsed().as_secs_f64();
                    }
                    self.expand(&mut loc, m, path);
                    self.pending.fetch_sub(1, Ordering::AcqRel);
                }
                None => {
                    if self.pending.load(Ordering::Acquire) == 0
                        && self
                            .dist
                            .as_ref()
                            .is_none_or(|d| d.done.load(Ordering::Acquire))
                    {
                        break;
                    }
                    idle += 1;
                    if self.stats && idle_since.is_none() {
                        idle_since = Some(Instant::now());
                    }
                    if idle < 64 {
                        std::thread::yield_now();
                    } else {
                        std::thread::sleep(Duration::from_micros(50));
                    }
                }
            }
        }
        if self.stats {
            eprintln!(
                "STAT worker {me}: expanded {} stole {} idle {idle_time:.2}s",
                loc.expanded, loc.steals
            );
        }
        loc.transitions
    }

    fn expand(&self, loc: &mut Local, m: Machine, path: Option<Arc<PathNode>>) {
        self.depth
            .fetch_max(path_depth(&path) as usize, Ordering::Relaxed);
        if self.stats && loc.expanded.is_multiple_of(97) {
            // Control configuration of a sampled state (debugging state-space blowup).
            let mut conf = String::new();
            let mut last = u32::MAX;
            for (_, t) in m.live_threads().filter(|(_, t)| t.st != St::Exited) {
                if t.warp != last {
                    conf += &format!(" w{}:{:x}{:?}", t.warp, t.pc, t.st);
                    last = t.warp;
                }
            }
            let pend: Vec<u32> = m.pending.iter().map(|o| o.pc).collect();
            eprintln!(
                "CONF{conf} | pending {pend:?} | maps mbars {} syncs {} bars {}",
                m.mbars.len(),
                m.syncs.len(),
                m.bars.len()
            );
        }
        let acts = m.actions();
        if let Some(v) = loc.view.as_mut() {
            for (e, pc, f) in blocked_sync(&m, &acts) {
                v.learn(e, pc, &f);
            }
        }
        let acts = if self.indep {
            cta_ample(&m, acts)
        } else {
            acts
        };
        let acts = local_ample(&m, acts);
        if acts.is_empty() {
            if let Some(t) = CHECK_TERMINALS.lock().unwrap().as_mut() {
                t.insert(m.fingerprint());
            }
            if m.terminal() {
                let mut m = m;
                m.exit_checks();
                for f in std::mem::take(&mut m.findings) {
                    self.finding(loc, f, &path);
                }
                crate::groups::terminal_sink(&m);
                self.terminal.fetch_add(1, Ordering::Relaxed);
                self.output(loc, m.outputs());
            } else {
                let (sig, msg) = m.deadlock_report();
                let f = Finding {
                    kind: "deadlock".into(),
                    sig,
                    msg,
                    trace: vec![],
                    ..Default::default()
                };
                self.finding(loc, f, &path);
            }
            return;
        }
        // Every successor (all of them are real steps, so their findings count), with its
        // synchronization footprint when the reduction is on.
        let mut succ: Vec<(Machine, Option<Arc<PathNode>>, Footprint)> = Vec::new();
        for (a, desc) in &acts {
            let mut n = m.clone();
            if loc.view.is_some() {
                n.fp = Some(Box::new(Footprint {
                    sync_only: true,
                    ..Default::default()
                }));
            }
            n.apply(a);
            loc.transitions += 1;
            let mut fp = n.fp.take().map(|b| *b).unwrap_or_default();
            fp.normalize();
            if let (Some(v), Some(e)) = (loc.view.as_mut(), action_ent(&m, a)) {
                v.learn(e, action_pc(&m, a), &fp);
            }
            let np = Some(Arc::new(PathNode {
                parent: path.clone(),
                desc: desc.clone(),
                act: a.clone(),
                depth: path_depth(&path) + 1,
            }));
            for f in std::mem::take(&mut n.findings) {
                self.finding(loc, f, &np);
            }
            succ.push((n, np, fp));
        }
        if succ.len() > 1 && sync_por_state_ok(&m) {
            if let Some(v) = loc.view.as_mut() {
                let fps: Vec<Footprint> = succ.iter().map(|s| s.2.clone()).collect();
                if let Some(i) = v.ample(&m, &acts, &fps) {
                    succ = vec![succ.swap_remove(i)];
                } else if std::env::var("PTXV_POR_DEBUG").is_ok() {
                    // States the reduction could not cut: what was enabled.
                    let mut d: Vec<String> = acts
                        .iter()
                        .map(|(a, _)| match a {
                            Action::Split { warp, pc, .. } => {
                                format!("w{warp}@{:x}", m.k.code[*pc as usize].loc)
                            }
                            Action::Async(j) => format!(
                                "A{}@{:x}",
                                m.threads[m.pending[*j].issuer as usize].warp,
                                m.k.code[m.pending[*j].pc as usize].loc
                            ),
                            _ => "other".into(),
                        })
                        .collect();
                    d.sort();
                    eprintln!("PORFULL {}", d.join(" "));
                }
            }
        }
        if self.stats && succ.len() > 1 && loc.expanded.is_multiple_of(101) {
            let d: Vec<&str> = acts.iter().map(|(_, d)| d.as_str()).collect();
            eprintln!("BRANCH {}", d.join(" || "));
        }
        if self.stats && succ.len() > 1 {
            let mut hist = self.hist.lock().unwrap();
            for (_, d) in &acts {
                let key: String = d
                    .split(" runs from ")
                    .last()
                    .unwrap_or(d)
                    .chars()
                    .take(70)
                    .collect();
                *hist.entry(key).or_insert(0usize) += 1;
            }
        }
        let mut new: Vec<Item> = Vec::new();
        for (mut n, np, _) in succ {
            if let Some(l) = self.learned {
                forget_dead_sync(&mut n, &l.reach[0], &l.inits);
            }
            let f = n.fingerprint();
            if let Some(d) = &self.dist {
                if d.d.owner(f) != d.d.rank {
                    d.ask(f, (n, np), loc.me);
                    continue;
                }
            }
            if self.visited.insert(f) {
                // Multi-node: rank 0 applies the cap to the sum over ranks.
                if self.states.fetch_add(1, Ordering::Relaxed) + 1 > self.max_states
                    && self.dist.is_none()
                {
                    self.stop.store(true, Ordering::Relaxed);
                    break;
                }
                new.push((n, np));
            }
        }
        if let Some(w) = &self.widths {
            // One worker, depth first: the new states are the children (at depth d + 1) of the
            // state on the current path at depth d.
            let d = path_depth(&path) as usize + 1;
            let mut w = w.lock().unwrap();
            w.resize(d + 1, 0);
            w[d] = new.len() as u32;
        }
        if !new.is_empty() {
            self.pending.fetch_add(new.len(), Ordering::AcqRel);
            self.deques[loc.me].push(new);
        }
        loc.expanded += 1;
        // Expansions of large kernels can take seconds each: look often (the line itself
        // is printed at most every 5 s).
        if loc.expanded.is_multiple_of(4) && self.dist.is_none() {
            if let Ok(mut p) = self.progress.try_lock() {
                if p.due() {
                    p.depth = self.depth.load(Ordering::Relaxed);
                    if self.widths.is_some() {
                        let (f, top) = self.done_estimate();
                        eprintln!(
                            "  [dfs estimate] done ~{f:.3e} of the tree; top levels (pending/children): {top}"
                        );
                    }
                    let f = self.found.lock().unwrap();
                    p.report(
                        self.states.load(Ordering::Relaxed),
                        self.terminal.load(Ordering::Relaxed),
                        self.pending.load(Ordering::Relaxed),
                        &f.findings,
                    );
                }
            }
        }
        let states = self.states.load(Ordering::Relaxed);
        if self.stats && states.is_multiple_of(20000) {
            let mut hist = self.hist.lock().unwrap();
            if !hist.is_empty() {
                let mut v: Vec<_> = hist.iter().collect();
                v.sort_by(|a, b| b.1.cmp(a.1));
                for (k, c) in v.iter().take(25) {
                    eprintln!("STAT {c:8} {k}");
                }
                eprintln!("STAT ---- {states}");
                hist.clear();
            }
        }
    }
}

/// One exhaustive search, depth first, by `threads` workers that steal work from each other
/// (with one thread: a plain depth-first search, in the same order every time).
fn explore_round(
    m0: Machine,
    max_states: usize,
    learned: Option<&Learned>,
    round: usize,
    threads: usize,
    net: Option<(Arc<crate::dist::Dist>, u64)>,
) -> Verdict {
    let t0 = Instant::now();
    let progress = Progress::new(
        format!(
            "{}{}{}",
            if m0.elect_high { "leader=high " } else { "" },
            "stateful",
            if round > 1 {
                format!(" round {round}")
            } else {
                String::new()
            }
        ),
        max_states,
    );
    let mut findings: Vec<Finding> = static_findings(&m0);
    findings.extend(crate::spin::check(&m0));
    let assumptions = crate::assumptions::for_machine(&m0);
    let f0 = m0.fingerprint();
    // Multi-node: the root belongs to its owner; the other ranks start idle and steal.
    let root_here = net.as_ref().is_none_or(|(d, _)| d.owner(f0) == d.rank);
    let sh = Arc::new(Shared {
        visited: Visited::new(threads),
        deques: (0..threads).map(|_| Deque::default()).collect(),
        states: AtomicUsize::new(root_here as usize),
        pending: AtomicUsize::new(root_here as usize),
        stop: AtomicBool::new(false),
    });
    let dist = net.map(|(d, call)| {
        let r = Arc::new(DistRound::new(
            d.clone(),
            call,
            round as u32,
            sh.clone(),
            &m0,
            max_states,
        ));
        d.set_round(
            call,
            round as u32,
            Some(r.clone() as Arc<dyn crate::dist::Sink>),
        );
        r
    });
    let search = Search {
        max_states,
        indep: ctas_independent(&m0),
        stats: std::env::var("PTXV_STATS").is_ok(),
        weak_stop: m0.weak_stop,
        learned,
        sh: sh.clone(),
        dist: dist.clone(),
        terminal: AtomicUsize::new(0),
        depth: AtomicUsize::new(0),
        widths: (threads == 1 && dist.is_none() && std::env::var("PTXV_DFS_ESTIMATE").is_ok())
            .then(|| Mutex::new(Vec::new())),
        found: Mutex::new(Found {
            seen_sig: findings.iter().map(|f| f.sig.clone()).collect(),
            findings,
            outputs: Vec::new(),
            out_seen: HashSet::new(),
        }),
        progress: Mutex::new(progress),
        hist: Mutex::new(HashMap::new()),
    };
    if root_here {
        search.visited.insert(f0);
        search.deques[0].push(vec![(m0.clone(), None)]);
    }
    let transitions: usize = if threads == 1 && dist.is_none() {
        search.worker(0, &m0)
    } else {
        std::thread::scope(|s| {
            let hs: Vec<_> = (1..threads)
                .map(|i| {
                    let (search, m0) = (&search, &m0);
                    std::thread::Builder::new()
                        .stack_size(STACK)
                        .spawn_scoped(s, move || search.worker(i, m0))
                        .expect("spawn explorer thread")
                })
                .collect();
            let comm = dist.as_ref().map(|d| {
                let (d, search) = (d.clone(), &search);
                s.spawn(move || d.comm(search))
            });
            let t = search.worker(0, &m0);
            let t = t + hs.into_iter().map(|h| h.join().unwrap()).sum::<usize>();
            if let Some(c) = comm {
                c.join().unwrap();
            }
            t
        })
    };
    let complete = !search.stop.load(Ordering::Relaxed);
    let terminal = search.terminal.load(Ordering::Relaxed);
    let states = search.states.load(Ordering::Relaxed);
    let found = search.found.into_inner().unwrap();
    let mut v = Verdict {
        notes: vec![],
        states,
        transitions,
        complete,
        terminal_states: terminal,
        distinct_outputs: found.outputs.len(),
        findings: found.findings,
        assumptions,
        seconds: t0.elapsed().as_secs_f64(),
        outputs: found.outputs,
    };
    if let Some(d) = dist {
        d.finish(&mut v, learned);
    }
    v
}

/// The scheduler of the deterministic runs (`run_once`, the certificate's `run_logged`,
/// `clc_eager`). Any schedule is a valid run; this one is chosen for cost: per step it looks
/// at one warp, not at every thread of the launch (`Machine::actions`, which also formats a
/// description of every action, costs O(threads) per step and dominated 128-CTA runs).
///
/// Policy (numerics runs: the original order, below): keep running the current warp's first
/// enabled split (lowest pc) for up to
/// `DET_STREAK` steps; when it has none, complete the first async operation that can
/// complete (in issue order, respecting the in-order pipes), else move to the next warp, in
/// cyclic order, that has an enabled split; when nothing of that kind can move (or CTAs are
/// unlaunched, a TMEM co-tenant chooses placements, or a cluster launch control response is
/// pending), the first action of `Machine::actions`.
#[derive(Default)]
pub struct DetSched {
    warp: u32,
    streak: u32,
    /// The live warps `lo..hi` (from `Machine::live`), computed on first use.
    warps: Option<(u32, u32)>,
    /// The machine has unlaunched CTAs or a TMEM co-tenant: use `actions` throughout.
    full: Option<bool>,
}

/// Steps one warp may take in a row before the deterministic scheduler moves on (so that a
/// warp that never blocks cannot starve the others).
const DET_STREAK: u32 = 4096;

impl DetSched {
    pub fn pick(&mut self, m: &Machine) -> Option<Action> {
        let full = *self.full.get_or_insert_with(|| {
            m.launch.env.tmem_cotenant || m.live_threads().any(|(_, t)| t.st == St::Unlaunched)
        });
        if full {
            // Unlaunched CTAs may launch at any step: keep the original policy.
            let acts = m.actions();
            return acts
                .iter()
                .find(|(a, _)| matches!(a, Action::Split { .. }))
                .or(acts.first())
                .map(|(a, _)| a.clone());
        }
        let (lo, hi) = *self.warps.get_or_insert_with(|| {
            if m.live.is_empty() {
                (0, 0)
            } else {
                (
                    m.threads[m.live.start].warp,
                    m.threads[m.live.end - 1].warp + 1,
                )
            }
        });
        if hi == lo {
            return None;
        }
        if self.warp < lo || self.warp >= hi {
            self.warp = lo;
        }
        // Numerics runs (`m.num`) keep the original order (lowest enabled warp, async only
        // when no split can move): the N-FMAX / N-FINITE bookkeeping depends on it
        // (fixtures/c_equiv_online loses its float atomic max diamond under the cyclic order).
        // PTXV_DET=lowest forces it everywhere.
        if m.num.is_some() || std::env::var("PTXV_DET").as_deref() == Ok("lowest") {
            for w in lo..hi {
                if let Some(a) = det_split(m, w) {
                    return Some(a);
                }
            }
            if let Some(i) = det_async(m) {
                if m.pending[..i]
                    .iter()
                    .all(|o| !matches!(o.kind, AsyncKind::Clc { .. }))
                {
                    return Some(Action::Async(i));
                }
            }
            return m.actions().first().map(|(a, _)| a.clone());
        }
        if self.streak < DET_STREAK {
            if let Some(a) = det_split(m, self.warp) {
                self.streak += 1;
                return Some(a);
            }
        }
        if let Some(i) = det_async(m) {
            return Some(Action::Async(i));
        }
        let n = hi - lo;
        for d in 1..=n {
            let w = lo + (self.warp - lo + d) % n;
            if let Some(a) = det_split(m, w) {
                self.warp = w;
                self.streak = 1;
                return Some(a);
            }
        }
        let acts = m.actions();
        acts.first().map(|(a, _)| a.clone())
    }
}

/// The first enabled split of warp `w` (lowest pc), as `Machine::actions` would offer it
/// first (PTX elect: the leader of the machine's policy).
fn det_split(m: &Machine, w: u32) -> Option<Action> {
    let wt = m.warp_threads(w);
    let mut pcs: Vec<u32> = wt
        .clone()
        .filter(|&i| m.threads[i].st == St::Run)
        .map(|i| m.threads[i].pc)
        .collect();
    pcs.sort_unstable();
    pcs.dedup();
    for pc in pcs {
        let members: Vec<usize> = wt
            .clone()
            .filter(|&i| m.threads[i].st == St::Run && m.threads[i].pc == pc)
            .collect();
        if !m.split_enabled(&members, pc) {
            continue;
        }
        let inst = &m.k.code[pc as usize];
        let mut leader = None;
        // ASSUME(E-ELECT): as in `actions`.
        if matches!(inst.op, ptxv_ir::Op::Elect { .. }) && m.k.front == ptxv_ir::Front::Ptx {
            let active: Vec<usize> = members
                .iter()
                .copied()
                .filter(|&t| m.prd(t, inst.guard))
                .collect();
            if active.len() > 1 {
                leader = Some(if m.elect_high && !elect_all() {
                    *active.last().unwrap()
                } else {
                    active[0]
                });
            }
        }
        return Some(Action::Split {
            warp: w,
            pc,
            leader,
        });
    }
    None
}

/// The first pending async operation that can complete (not a cluster launch control
/// response; in-order pipes: only the oldest of each), as `Machine::actions` lists them.
fn det_async(m: &Machine) -> Option<usize> {
    for (i, op) in m.pending.iter().enumerate() {
        if let AsyncKind::Clc { .. } = op.kind {
            continue;
        }
        if let Some(f) = op.fifo {
            if m.pending[..i].iter().any(|o| o.fifo == Some(f)) {
                continue;
            }
        }
        return Some(i);
    }
    None
}

/// A deterministic run's guard against a step that faults again and again: a faulting step
/// leaves its warp where it was, so a scheduler that keeps picking it never ends.
#[derive(Default)]
pub struct FaultLoop {
    last: Option<(String, u32)>,
}

impl FaultLoop {
    /// After a step with findings `fs`: Some(why) once the same fault came 64 steps in a row.
    pub fn repeats(&mut self, fs: &[Finding]) -> Option<String> {
        match fs.iter().find(|f| f.sig.starts_with("fault:")) {
            None => self.last = None,
            Some(f) => match &mut self.last {
                Some((s, n)) if *s == f.sig => {
                    *n += 1;
                    if *n >= 64 {
                        return Some(format!(
                            "a step faults again and again ({}); the run stops there",
                            f.msg.lines().next().unwrap_or("")
                        ));
                    }
                }
                _ => self.last = Some((f.sig.clone(), 1)),
            },
        }
        None
    }
}

/// One deterministic execution (`DetSched`). Returns the final machine and the findings
/// seen along the way.
pub fn run_once(mut m: Machine) -> (Machine, Vec<Finding>, bool) {
    let mut fs = Vec::new();
    // Debugging: PTXV_RUN_ONCE_PROGRESS=n prints every n-th action.
    let every: usize = std::env::var("PTXV_RUN_ONCE_PROGRESS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut n = 0usize;
    let mut sched = DetSched::default();
    let mut faults = FaultLoop::default();
    let mut stuck = false;
    while let Some(a) = sched.pick(&m) {
        n += 1;
        if every > 0 && n.is_multiple_of(every) {
            let what = match &a {
                Action::Split { warp, pc, .. } => {
                    format!("warp {warp} runs from {}", m.k.describe(*pc as usize))
                }
                _ => format!("{a:?}"),
            };
            eprintln!("run_once step {n}: {what}");
        }
        m.apply(&a);
        if every > 0 && n.is_multiple_of(every) {
            for f in &m.findings {
                eprintln!("  finding: {}", f.msg.lines().next().unwrap_or(""));
            }
        }
        stuck = faults.repeats(&m.findings).is_some();
        fs.append(&mut m.findings);
        if stuck {
            break;
        }
    }
    let ok = m.terminal();
    if ok {
        m.exit_checks();
        fs.append(&mut m.findings);
    } else if !stuck {
        let (sig, msg) = m.deadlock_report();
        fs.push(Finding {
            kind: "deadlock".into(),
            sig,
            msg,
            trace: vec![],
            ..Default::default()
        });
    }
    (m, fs, ok)
}

// ---------------------------------------------------------------- multi-node (dist.rs)

use crate::dist::Body;
use crate::dist::Dist;
use crate::dist::Msg;
use crate::dist::R;
use crate::dist::W;

fn put_action(w: &mut W, a: &Action) {
    match a {
        Action::Split { warp, pc, leader } => {
            w.u8(1);
            w.u32(*warp);
            w.u32(*pc);
            w.u8(leader.is_some() as u8);
            w.u64(leader.unwrap_or(0) as u64);
        }
        Action::Async(i) => {
            w.u8(2);
            w.u64(*i as u64);
        }
        Action::Launch(c) => {
            w.u8(3);
            w.u32(*c as u32);
        }
        Action::Clc(i, c) => {
            w.u8(4);
            w.u64(*i as u64);
            w.u8(c.is_some() as u8);
            w.u32(c.unwrap_or(0) as u32);
        }
    }
}

fn get_action(r: &mut R) -> Action {
    match r.u8() {
        1 => {
            let (warp, pc) = (r.u32(), r.u32());
            let has = r.u8() != 0;
            let l = r.u64() as usize;
            Action::Split {
                warp,
                pc,
                leader: has.then_some(l),
            }
        }
        2 => Action::Async(r.u64() as usize),
        3 => Action::Launch(r.u32() as u16),
        _ => {
            let i = r.u64() as usize;
            let has = r.u8() != 0;
            let c = r.u32() as u16;
            Action::Clc(i, has.then_some(c))
        }
    }
}

fn put_ranges(w: &mut W, v: &[FpRange]) {
    w.u64(v.len() as u64);
    for &(a, b, c, d) in v {
        w.u8(a);
        w.u32(b);
        w.u64(c);
        w.u64(d);
    }
}

fn get_ranges(r: &mut R) -> Vec<FpRange> {
    let n = r.u64() as usize;
    (0..n)
        .map(|_| (r.u8(), r.u32(), r.u64(), r.u64()))
        .collect()
}

fn encode_learn(e: Ent, pc: u32, f: &Footprint) -> Vec<u8> {
    let mut w = W::default();
    match e {
        Ent::Warp(x) => {
            w.u8(0);
            w.u32(x);
        }
        Ent::Async(x) => {
            w.u8(1);
            w.u32(x);
        }
    }
    w.u32(pc);
    put_ranges(&mut w, &f.r);
    put_ranges(&mut w, &f.w);
    w.u8(f.all as u8);
    w.0
}

/// A learned set that grew on another rank.
struct LearnSink(Arc<Learned>);

impl crate::dist::Sink for LearnSink {
    fn handle(&self, _from: usize, m: Msg) {
        let Body::Learn(b) = m.body else { return };
        let mut r = R(&b);
        let e = match r.u8() {
            0 => Ent::Warp(r.u32()),
            _ => Ent::Async(r.u32()),
        };
        let pc = r.u32();
        let fp = Footprint {
            r: get_ranges(&mut r),
            w: get_ranges(&mut r),
            all: r.u8() != 0,
            ..Default::default()
        };
        // As `LearnedView::learn`, without broadcasting again.
        let l = &self.0;
        let mut g = l.inner.lock().unwrap();
        if let Some(cur) = grown(g.by.get(&(e, pc)), &fp) {
            Arc::make_mut(&mut g.by).insert((e, pc), cur);
            if !g.dirty {
                g.dirty = g.chosen.iter().any(|(c, r, w)| {
                    *c != e
                        && fp.conflicts(&Footprint {
                            r: r.clone(),
                            w: w.clone(),
                            ..Default::default()
                        })
                });
            }
            l.version.fetch_add(1, Ordering::AcqRel);
        }
    }
}

fn put_findings(w: &mut W, fs: &[Finding]) {
    w.u64(fs.len() as u64);
    for f in fs {
        w.str(&f.kind);
        w.str(&f.sig);
        w.str(&f.msg);
        w.u64(f.trace.len() as u64);
        for t in &f.trace {
            w.str(t);
        }
    }
}

fn get_findings(r: &mut R) -> Vec<Finding> {
    let n = r.u64() as usize;
    (0..n)
        .map(|_| Finding {
            kind: r.str(),
            sig: r.str(),
            msg: r.str(),
            trace: (0..r.u64() as usize).map(|_| r.str()).collect(),
            ..Default::default()
        })
        .collect()
}

/// One round's numbers (a rank's report to rank 0, or rank 0's merged decision).
struct RoundSum {
    dirty: bool,
    complete: bool,
    states: u64,
    transitions: u64,
    terminal: u64,
    findings: Vec<Finding>,
    outputs: Vec<Vec<u8>>,
}

impl RoundSum {
    fn encode(&self) -> Vec<u8> {
        let mut w = W::default();
        w.u8(self.dirty as u8);
        w.u8(self.complete as u8);
        w.u64(self.states);
        w.u64(self.transitions);
        w.u64(self.terminal);
        put_findings(&mut w, &self.findings);
        w.u64(self.outputs.len() as u64);
        for o in &self.outputs {
            w.bytes(o);
        }
        w.0
    }
    fn decode(b: &[u8]) -> RoundSum {
        let mut r = R(b);
        RoundSum {
            dirty: r.u8() != 0,
            complete: r.u8() != 0,
            states: r.u64(),
            transitions: r.u64(),
            terminal: r.u64(),
            findings: get_findings(&mut r),
            outputs: (0..r.u64() as usize).map(|_| r.bytes()).collect(),
        }
    }
}

/// Coordinator view of the ranks (rank 0): the latest status of each.
#[derive(Default)]
struct Coord {
    status: HashMap<usize, (bool, u64, u64, u64, u64, u64)>,
    /// The previous all-idle wave: totals sent / received, and each rank's wave number.
    prev: Option<(u64, u64, Vec<u64>)>,
}

/// One rank's part of one round of a multi-node search (see dist.rs).
struct DistRound {
    d: Arc<Dist>,
    call: u64,
    round: u32,
    sh: Arc<Shared>,
    m0: Machine,
    max_states: usize,
    /// States waiting for their owner's answer, by token, with the worker that found them.
    tokens: Mutex<HashMap<u64, (Item, usize)>>,
    next_token: AtomicU64,
    /// Questions not yet sent, per owner.
    outq: Vec<Mutex<Vec<(u128, u64)>>>,
    /// Stolen states.
    inbox: Deque,
    /// Work messages (non-empty steals) sent / received.
    sent: AtomicU64,
    recv: AtomicU64,
    steal_out: AtomicBool,
    /// Rank 0 said the round's search is over.
    done: AtomicBool,
    coord: Mutex<Coord>,
    mail: crate::dist::Mailbox,
}

impl DistRound {
    fn new(
        d: Arc<Dist>,
        call: u64,
        round: u32,
        sh: Arc<Shared>,
        m0: &Machine,
        max_states: usize,
    ) -> DistRound {
        let n = d.size;
        DistRound {
            d,
            call,
            round,
            sh,
            m0: m0.clone(),
            max_states,
            tokens: Mutex::new(HashMap::new()),
            next_token: AtomicU64::new(0),
            outq: (0..n).map(|_| Mutex::new(Vec::new())).collect(),
            inbox: Deque::default(),
            sent: AtomicU64::new(0),
            recv: AtomicU64::new(0),
            steal_out: AtomicBool::new(false),
            done: AtomicBool::new(false),
            coord: Mutex::new(Coord::default()),
            mail: Default::default(),
        }
    }

    fn msg(&self, body: Body) -> Msg {
        Msg {
            call: self.call,
            round: self.round,
            body,
        }
    }

    /// Keep `it` until the owner of `f` says whether it is new; then it goes back on the
    /// stack of `worker`, which found it (depth first: a shared queue of answered states
    /// turned the search breadth first, and its frontier ran out of memory).
    fn ask(&self, f: u128, it: Item, worker: usize) {
        let t = self.next_token.fetch_add(1, Ordering::Relaxed);
        self.sh.pending.fetch_add(1, Ordering::AcqRel);
        self.tokens.lock().unwrap().insert(t, (it, worker));
        let o = self.d.owner(f);
        let full = {
            let mut q = self.outq[o].lock().unwrap();
            q.push((f, t));
            q.len() >= 4096
        };
        if full {
            self.flush(o);
        }
    }

    fn flush(&self, o: usize) {
        let v = std::mem::take(&mut *self.outq[o].lock().unwrap());
        if !v.is_empty() {
            self.d.send(o, self.msg(Body::Query(v)));
        }
    }

    /// Rebuild a stolen state from its schedule.
    fn replay(&self, b: &[u8]) -> Item {
        let mut r = R(b);
        let n = r.u64() as usize;
        let mut m = self.m0.clone();
        let mut path: Option<Arc<PathNode>> = None;
        for _ in 0..n {
            let a = get_action(&mut r);
            let desc = m
                .actions()
                .into_iter()
                .find(|(b, _)| *b == a)
                .map(|(_, d)| d)
                .unwrap_or_else(|| format!("{a:?}"));
            m.apply(&a);
            m.findings.clear();
            let depth = path_depth(&path) + 1;
            path = Some(Arc::new(PathNode {
                parent: path,
                desc,
                act: a,
                depth,
            }));
        }
        (m, path)
    }

    /// The schedule that reaches a queued state.
    fn schedule(p: &Option<Arc<PathNode>>) -> Vec<u8> {
        let mut acts = Vec::new();
        let mut cur = p.clone();
        while let Some(n) = cur {
            acts.push(n.act.clone());
            cur = n.parent.clone();
        }
        acts.reverse();
        let mut w = W::default();
        w.u64(acts.len() as u64);
        for a in &acts {
            put_action(&mut w, a);
        }
        w.0
    }

    /// Give part of our queued work to an idle rank.
    fn give(&self, to: usize) {
        let mut items: Vec<Item> = Vec::new();
        let busiest = self
            .sh
            .deques
            .iter()
            .chain(std::iter::once(&self.inbox))
            .max_by_key(|q| q.len.load(Ordering::Relaxed));
        if let Some(q) = busiest {
            let n = q.len.load(Ordering::Relaxed);
            if n >= 2 {
                for _ in 0..(n / 2).min(16) {
                    match q.pop(false) {
                        Some(it) => items.push(it),
                        None => break,
                    }
                }
            }
        }
        let paths: Vec<Vec<u8>> = items.iter().map(|(_, p)| Self::schedule(p)).collect();
        if !paths.is_empty() {
            self.sent.fetch_add(1, Ordering::AcqRel);
            self.sh.pending.fetch_sub(paths.len(), Ordering::AcqRel);
        }
        self.d.send(to, self.msg(Body::Work(paths)));
    }

    /// The rank's communication loop during the round: send questions, report status, ask
    /// for work when idle; on rank 0 also decide termination and the state cap.
    fn comm(&self, search: &Search) {
        let mut wave = 0u64;
        let mut last_status = Instant::now();
        let mut last_steal = Instant::now();
        let mut fails = 0u32;
        let mut rng = 0x2545_f491_4f6c_dd1du64 ^ (self.d.rank as u64 + 1);
        let mut progress_due = Instant::now();
        while !self.done.load(Ordering::Acquire) && !self.sh.stop.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(1));
            for o in 0..self.d.size {
                if o != self.d.rank {
                    self.flush(o);
                }
            }
            let idle = self.sh.pending.load(Ordering::Acquire) == 0;
            if last_status.elapsed() >= Duration::from_millis(20) {
                last_status = Instant::now();
                wave += 1;
                self.d.send(
                    0,
                    self.msg(Body::Status {
                        idle,
                        sent: self.sent.load(Ordering::Acquire),
                        recv: self.recv.load(Ordering::Acquire),
                        states: self.sh.states.load(Ordering::Relaxed) as u64,
                        terminal: search.terminal.load(Ordering::Relaxed) as u64,
                        wave,
                    }),
                );
            }
            if idle
                && !self.steal_out.load(Ordering::Acquire)
                && last_steal.elapsed() >= Duration::from_millis((2u64 << fails.min(4)).min(20))
            {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let mut v = (rng % (self.d.size as u64 - 1)) as usize;
                if v >= self.d.rank {
                    v += 1;
                }
                self.steal_out.store(true, Ordering::Release);
                last_steal = Instant::now();
                fails += 1;
                self.d.send(v, self.msg(Body::StealReq));
            } else if !idle {
                fails = 0;
            }
            if self.d.rank == 0 {
                self.coordinate(search, &mut progress_due);
            }
        }
    }

    fn coordinate(&self, search: &Search, progress_due: &mut Instant) {
        let mut c = self.coord.lock().unwrap();
        if c.status.len() < self.d.size {
            return;
        }
        let states: u64 = c.status.values().map(|s| s.3).sum();
        let terminal: u64 = c.status.values().map(|s| s.4).sum();
        if progress_due.elapsed() >= Duration::from_secs(5) {
            *progress_due = Instant::now();
            if let Ok(mut p) = search.progress.try_lock() {
                let f = search.found.lock().unwrap();
                let busy = c.status.values().filter(|s| !s.0).count();
                // This rank's deepest state (the others' are not reported).
                p.depth = search.depth.load(Ordering::Relaxed);
                p.report(states as usize, terminal as usize, busy, &f.findings);
            }
        }
        if states as usize > self.max_states {
            drop(c);
            self.d.broadcast(&self.msg(Body::Control(1)));
            self.sh.stop.store(true, Ordering::Release);
            return;
        }
        let all_idle = c.status.values().all(|s| s.0);
        let sent: u64 = c.status.values().map(|s| s.1).sum();
        let recv: u64 = c.status.values().map(|s| s.2).sum();
        let waves: Vec<u64> = (0..self.d.size).map(|r| c.status[&r].5).collect();
        if !all_idle || sent != recv {
            c.prev = None;
            return;
        }
        match &c.prev {
            Some((ps, pr, pw)) if *ps == sent && *pr == recv => {
                if waves.iter().zip(pw).all(|(w, p)| w > p) {
                    drop(c);
                    self.d.broadcast(&self.msg(Body::Control(2)));
                    self.done.store(true, Ordering::Release);
                }
            }
            _ => c.prev = Some((sent, recv, waves)),
        }
    }

    /// End of the round: wait until every rank's learned-set messages arrived, send our
    /// numbers to rank 0, and adopt the merged result (so every rank continues alike).
    fn finish(&self, v: &mut Verdict, learned: Option<&Learned>) {
        self.d.broadcast(&self.msg(Body::Marker));
        let peers = self.d.size - 1;
        self.mail.wait(|q| {
            (q.values()
                .filter(|b| b.iter().any(|x| matches!(x, Body::Marker)))
                .count()
                >= peers)
                .then_some(())
        });
        let dirty = learned.is_some_and(|l| l.inner.lock().unwrap().dirty);
        let mine = RoundSum {
            dirty,
            complete: v.complete,
            states: v.states as u64,
            transitions: v.transitions as u64,
            terminal: v.terminal_states as u64,
            findings: std::mem::take(&mut v.findings),
            outputs: std::mem::take(&mut v.outputs),
        };
        self.d.send(0, self.msg(Body::Report(mine.encode())));
        if self.d.rank == 0 {
            let reports: Vec<RoundSum> = self.mail.wait(|q| {
                let n = q
                    .values()
                    .filter(|b| b.iter().any(|x| matches!(x, Body::Report(_))))
                    .count();
                (n >= self.d.size).then(|| {
                    (0..self.d.size)
                        .map(|r| {
                            let b = q[&r]
                                .iter()
                                .find_map(|x| match x {
                                    Body::Report(b) => Some(b.clone()),
                                    _ => None,
                                })
                                .unwrap();
                            RoundSum::decode(&b)
                        })
                        .collect()
                })
            });
            let mut all = RoundSum {
                dirty: false,
                complete: true,
                states: 0,
                transitions: 0,
                terminal: 0,
                findings: Vec::new(),
                outputs: Vec::new(),
            };
            for r in reports {
                all.dirty |= r.dirty;
                all.complete &= r.complete;
                all.states += r.states;
                all.transitions += r.transitions;
                all.terminal += r.terminal;
                for f in r.findings {
                    if !all.findings.iter().any(|g| g.sig == f.sig) {
                        all.findings.push(f);
                    }
                }
                for o in r.outputs {
                    if !all.outputs.contains(&o) {
                        all.outputs.push(o);
                    }
                }
            }
            let b = all.encode();
            self.d.broadcast(&self.msg(Body::Decision(b.clone())));
            self.mail.put(0, Body::Decision(b));
        }
        let dec = self.mail.wait(|q| {
            q.get(&0).and_then(|v| {
                v.iter().find_map(|x| match x {
                    Body::Decision(b) => Some(b.clone()),
                    _ => None,
                })
            })
        });
        let s = RoundSum::decode(&dec);
        v.complete = s.complete;
        v.states = s.states as usize;
        v.transitions = s.transitions as usize;
        v.terminal_states = s.terminal as usize;
        v.findings = s.findings;
        v.outputs = s.outputs;
        v.distinct_outputs = v.outputs.len();
        if let Some(l) = learned {
            l.inner.lock().unwrap().dirty = s.dirty;
        }
        self.d.set_round(self.call, self.round, None);
    }
}

impl crate::dist::Sink for DistRound {
    fn handle(&self, from: usize, m: Msg) {
        match m.body {
            Body::Query(v) => {
                let mut out = Vec::with_capacity(v.len());
                for (f, t) in v {
                    let new = self.sh.visited.insert(f);
                    if new {
                        self.sh.states.fetch_add(1, Ordering::Relaxed);
                    }
                    out.push((t, new));
                }
                self.d.send(from, self.msg(Body::Reply(out)));
            }
            Body::Reply(v) => {
                let mut new: HashMap<usize, Vec<Item>> = HashMap::new();
                let mut old = 0;
                {
                    let mut toks = self.tokens.lock().unwrap();
                    for (t, n) in v {
                        let (it, w) = toks.remove(&t).expect("dist: unknown token");
                        if n {
                            new.entry(w).or_default().push(it);
                        } else {
                            old += 1;
                        }
                    }
                }
                // New states stay pending (now queued here); the others are done.
                for (w, items) in new {
                    self.sh.deques[w].push(items);
                }
                self.sh.pending.fetch_sub(old, Ordering::AcqRel);
            }
            Body::StealReq => self.give(from),
            Body::Work(paths) => {
                if !paths.is_empty() {
                    let items: Vec<Item> = paths.iter().map(|p| self.replay(p)).collect();
                    self.sh.pending.fetch_add(items.len(), Ordering::AcqRel);
                    self.inbox.push(items);
                    self.recv.fetch_add(1, Ordering::AcqRel);
                }
                self.steal_out.store(false, Ordering::Release);
            }
            Body::Status {
                idle,
                sent,
                recv,
                states,
                terminal,
                wave,
            } => {
                self.coord
                    .lock()
                    .unwrap()
                    .status
                    .insert(from, (idle, sent, recv, states, terminal, wave));
            }
            Body::Control(1) => self.sh.stop.store(true, Ordering::Release),
            Body::Control(_) => self.done.store(true, Ordering::Release),
            b @ (Body::Marker | Body::Report(_) | Body::Decision(_)) => self.mail.put(from, b),
            Body::Learn(_) => {}
        }
    }
}

// ------------------------------------------------------------------ value-tracking look-ahead

/// Abstract value of a register in the look-ahead (`lookahead`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Av {
    /// Known, the same in every lane.
    K(u64),
    /// Known per lane (index into the warp's interned lane vectors, `LaWarp::lv`).
    L(u32),
    /// Unknown, but the same in every lane.
    U,
    /// Unknown.
    N,
    /// A predicate true in exactly one lane of those running (`ELECT`).
    E,
}

/// Register values of a configuration: a small sorted map (cheap to copy).
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
struct LaEnv(Vec<(ptxv_ir::Reg, Av)>);

impl LaEnv {
    fn get(&self, r: &ptxv_ir::Reg) -> Option<&Av> {
        self.0
            .binary_search_by(|x| x.0.cmp(r))
            .ok()
            .map(|i| &self.0[i].1)
    }
    fn insert(&mut self, r: ptxv_ir::Reg, v: Av) {
        match self.0.binary_search_by(|x| x.0.cmp(&r)) {
            Ok(i) => self.0[i].1 = v,
            Err(i) => self.0.insert(i, (r, v)),
        }
    }
}

/// A fast hasher for the look-ahead's configuration sets (FxHash).
#[derive(Default)]
struct LaHasher(u64);

impl Hasher for LaHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, b: &[u8]) {
        for c in b.chunks(8) {
            let mut x = [0u8; 8];
            x[..c.len()].copy_from_slice(c);
            self.write_u64(u64::from_le_bytes(x));
        }
    }
    fn write_u8(&mut self, x: u8) {
        self.write_u64(x as u64)
    }
    fn write_u16(&mut self, x: u16) {
        self.write_u64(x as u64)
    }
    fn write_u32(&mut self, x: u32) {
        self.write_u64(x as u64)
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    fn write_usize(&mut self, x: usize) {
        self.write_u64(x as u64)
    }
}

/// One abstract configuration of a warp: every live lane at `pc`, or, inside an elected
/// body, the lanes running it while the rest wait at the BSYNC `body.0`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct LaCfg {
    pc: u32,
    /// Registers whose value differs from their start value (`LaWarp::init`).
    env: LaEnv,
    /// Per mbarrier: a lower bound on the phase completions this configuration observed.
    obs: BTreeMap<(u32, u64), u64>,
    /// Upper bound on the lanes running.
    lanes: u32,
    /// Inside an elected body: the BSYNC pc where the warp rejoins, and the per-thread
    /// registers the body wrote (lane-dependent after the rejoin).
    body: Option<(u32, BTreeSet<ptxv_ir::Reg>)>,
    /// The merge-point representatives and forked configurations this one descends from
    /// (`LaWarp::summarize`); ignored by comparisons.
    hist: LaHist,
}

/// Ancestry of a configuration (sorted ids); equal to every other, hashes to nothing.
#[derive(Clone, Debug, Default)]
struct LaHist(Vec<u64>);

impl PartialEq for LaHist {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for LaHist {}

impl std::hash::Hash for LaHist {
    fn hash<H: Hasher>(&self, _: &mut H) {}
}

impl LaHist {
    fn has(&self, id: u64) -> bool {
        self.0.binary_search(&id).is_ok()
    }
    fn add(&mut self, id: u64) {
        if let Err(i) = self.0.binary_search(&id) {
            self.0.insert(i, id);
        }
    }
    fn union(&mut self, o: &LaHist) {
        for &x in &o.0 {
            self.add(x);
        }
    }
}

/// What the look-ahead found for one candidate.
struct LaResult {
    /// Per live warp (not the candidate's): the pcs it may execute before the candidate, or
    /// None when the look-ahead gave up on it.
    exec: HashMap<u32, Option<Vec<u64>>>,
    /// Upper bounds on the arrivals at each mbarrier before the candidate.
    arr: HashMap<(u32, u64), u64>,
    /// Mbarriers (and CTAs, all of whose mbarriers) with unbounded arrivals.
    inf: HashSet<(u32, u64)>,
    inf_cta: HashSet<u32>,
    /// Mbarriers that held some configuration back at the end.
    stopped: Vec<(u32, u64)>,
    /// Why warps were given up on (`PTXV_LA_DEBUG`).
    why: Vec<String>,
}

impl LaResult {
    /// Upper bound on the phase completions of mbarrier `k` before the candidate.
    fn bound(&self, m: &Machine, k: (u32, u64)) -> u64 {
        if self.inf.contains(&k) || self.inf_cta.contains(&k.0) {
            return u64::MAX;
        }
        let Some(mb) = m.mbars.get(&(k.0 as u16, k.1 as u32)) else {
            return u64::MAX;
        };
        let a = self.arr.get(&k).copied().unwrap_or(0);
        if a == u64::MAX || mb.expected == 0 {
            return u64::MAX;
        }
        // Transaction bytes are ignored (an upper bound). With no arrival pending, the phase
        // can complete on its transaction bytes alone.
        if a < mb.pending as u64 {
            0
        } else {
            1 + (a - mb.pending as u64) / mb.expected as u64
        }
    }

    fn arrive(&mut self, k: (u32, u64), n: u64) {
        let e = self.arr.entry(k).or_insert(0);
        *e = e.saturating_add(n);
    }
}

/// What one step of a configuration led to.
enum LaStep {
    Next,
    Jump(u32),
    /// The configuration ends (exit, trap, or a barrier cut by the candidate).
    End,
    /// Blocked at a poll of this mbarrier awaiting this parity.
    Wait((u32, u64), u64),
    /// Alternatives (a branch on a uniform unknown value), by pc; None: falls through and
    /// also ends (a uniform unknown exit).
    Fork(Vec<Option<u32>>),
    Wild(String),
}

/// A representative configuration at a merge point and its id (`LaWarp::at`).
type LaRep = (LaCfg, u64);

/// The mbarriers a footprint names (`bars_of`).
type BarSet = HashSet<(u32, u64)>;

/// One warp's look-ahead.
struct LaWarp {
    w: u32,
    /// Live lanes (thread indices); `ts[0]` is the representative.
    ts: Vec<usize>,
    /// Start values, computed on demand.
    init: HashMap<ptxv_ir::Reg, Av>,
    lv: Vec<Vec<u64>>,
    lvi: HashMap<Vec<u64>, u32>,
    work: Vec<LaCfg>,
    waiting: Vec<(LaCfg, (u32, u64), u64)>,
    seen: HashMap<LaCfg, u64, BuildHasherDefault<LaHasher>>,
    /// Next configuration id (`LaHist`).
    ids: u64,
    exec: Vec<u64>,
    steps: usize,
    wild: Option<String>,
    wild_done: bool,
    cut: usize,
    /// Per merge point (and body rejoin point): the configurations run from there, each the
    /// join of compatible ones that reached it (`LaWarp::compatible`).
    at: HashMap<(u32, Option<u32>), Vec<LaRep>>,
    /// Debugging (`PTXV_LA_TRACE=cand,warp`): print this warp's polls and arrivals.
    trace: bool,
    /// Per pc, its strongly connected component of the control-flow graph (u32::MAX: on no
    /// cycle); the components summarized by a join (`summarize`); the mbarriers each arrival
    /// pc arrived at.
    scc: std::rc::Rc<Vec<u32>>,
    loops: HashSet<u32>,
    arr_at: HashMap<u32, HashSet<(u32, u64)>>,
}

/// Configurations kept apart per merge point; beyond that, new ones join the first.
const LA_REPS: usize = 32;

fn la_join_av(a: Av, b: Av) -> Av {
    if a == b {
        a
    } else if matches!(a, Av::K(_) | Av::U) && matches!(b, Av::K(_) | Av::U) {
        Av::U
    } else {
        Av::N
    }
}

fn la_rd(m: &Machine, t: usize, r: ptxv_ir::Reg) -> u64 {
    use ptxv_ir::File;
    match r.file {
        File::P | File::UP => m.prd(
            t,
            ptxv_ir::Pred {
                reg: Some(r),
                neg: false,
            },
        ) as u64,
        _ => m.rd(t, &ptxv_ir::Val::Reg(r, 1)).unwrap_or(0),
    }
}

fn la_uniform_file(r: ptxv_ir::Reg) -> bool {
    matches!(r.file, ptxv_ir::File::UR | ptxv_ir::File::UP)
}

impl LaWarp {
    fn mk(&mut self, v: Vec<u64>) -> Av {
        if v.iter().all(|&x| x == v[0]) {
            return Av::K(v[0]);
        }
        if let Some(&i) = self.lvi.get(&v) {
            return Av::L(i);
        }
        let i = self.lv.len() as u32;
        self.lv.push(v.clone());
        self.lvi.insert(v, i);
        Av::L(i)
    }

    fn lane(&self, a: Av, i: usize) -> Option<u64> {
        match a {
            Av::K(x) => Some(x),
            Av::L(j) => Some(self.lv[j as usize][i]),
            _ => None,
        }
    }

    fn lanes_of(&self, a: Av) -> Option<Vec<u64>> {
        (0..self.ts.len()).map(|i| self.lane(a, i)).collect()
    }

    fn get(&mut self, m: &Machine, c: &LaCfg, r: ptxv_ir::Reg) -> Av {
        use ptxv_ir::File;
        if r.idx == 7
            && (r.file == File::UP || r.file == File::P && m.k.front == ptxv_ir::Front::Sass)
        {
            return Av::K(1);
        }
        if matches!(r.file, File::R | File::UR) && r.idx >= 255 {
            return Av::K(0);
        }
        if let Some(v) = c.env.get(&r) {
            return *v;
        }
        if let Some(v) = self.init.get(&r) {
            return *v;
        }
        let v = if la_uniform_file(r) {
            Av::K(la_rd(m, self.ts[0], r))
        } else {
            let v: Vec<u64> = self.ts.iter().map(|&t| la_rd(m, t, r)).collect();
            self.mk(v)
        };
        self.init.insert(r, v);
        v
    }

    fn val(&mut self, m: &Machine, c: &LaCfg, v: &ptxv_ir::Val) -> Av {
        use ptxv_ir::SReg;
        use ptxv_ir::Val;
        match v {
            Val::Imm(x) => Av::K(*x),
            Val::SReg(
                SReg::Tid(_)
                | SReg::LaneId
                | SReg::LaneMaskLt
                | SReg::LaneMaskLe
                | SReg::LaneMaskGe
                | SReg::LaneMaskGt
                | SReg::LaneMaskEq,
            ) => {
                let x: Result<Vec<u64>, String> = self.ts.iter().map(|&t| m.rd(t, v)).collect();
                x.map_or(Av::N, |x| self.mk(x))
            }
            Val::Const(..) | Val::SReg(_) => m.rd(self.ts[0], v).map_or(Av::N, Av::K),
            Val::Reg(r, n) => {
                if r.file == ptxv_ir::File::V || *n <= 1 {
                    return self.get(m, c, *r);
                }
                let lo = self.get(m, c, *r);
                let hi = self.get(
                    m,
                    c,
                    ptxv_ir::Reg {
                        file: r.file,
                        idx: r.idx + 1,
                    },
                );
                match (self.lanes_of(lo), self.lanes_of(hi)) {
                    (Some(a), Some(b)) => {
                        let v = a
                            .iter()
                            .zip(&b)
                            .map(|(a, b)| (a & 0xffff_ffff) | (b << 32))
                            .collect();
                        self.mk(v)
                    }
                    _ if matches!(lo, Av::K(_) | Av::U) && matches!(hi, Av::K(_) | Av::U) => Av::U,
                    _ => Av::N,
                }
            }
        }
    }

    fn pred(&mut self, m: &Machine, c: &LaCfg, p: ptxv_ir::Pred) -> Av {
        let Some(r) = p.reg else {
            return Av::K(!p.neg as u64);
        };
        match self.get(m, c, r) {
            Av::K(v) => Av::K(((v != 0) ^ p.neg) as u64),
            Av::L(j) => {
                let v = self.lv[j as usize]
                    .iter()
                    .map(|&x| ((x != 0) ^ p.neg) as u64)
                    .collect();
                self.mk(v)
            }
            Av::E if !p.neg => Av::E,
            Av::U => Av::U,
            _ => Av::N,
        }
    }

    /// Are the operands known (Ok(true): the same in every lane; Ok(false): per lane), or
    /// not (the join of the unknown ones: U or N)?
    fn kind(&mut self, m: &Machine, c: &LaCfg, vals: &[ptxv_ir::Val]) -> Result<bool, Av> {
        let mut uni = true;
        let mut bad = None;
        for v in vals {
            match self.val(m, c, v) {
                Av::K(_) => {}
                Av::L(_) => uni = false,
                Av::U => bad = Some(bad.unwrap_or(Av::U)),
                _ => bad = Some(Av::N),
            }
        }
        bad.map_or(Ok(uni), Err)
    }

    /// Load lane `i`'s values of the registers among `vals` into the scratch machine.
    fn load(&mut self, m: &Machine, c: &LaCfg, s: &mut Machine, vals: &[ptxv_ir::Val], i: usize) {
        use ptxv_ir::File;
        let t = self.ts[i];
        for v in vals {
            let ptxv_ir::Val::Reg(r, n) = v else {
                continue;
            };
            let k = if r.file == File::V {
                1
            } else {
                (*n).max(1) as u16
            };
            for j in 0..k {
                let rr = ptxv_ir::Reg {
                    file: r.file,
                    idx: r.idx + j,
                };
                if rr.idx >= 255 && matches!(rr.file, File::R | File::UR) {
                    continue;
                }
                let a = self.get(m, c, rr);
                if let Some(x) = self.lane(a, i) {
                    match rr.file {
                        File::P | File::UP => {
                            if !(rr.idx == 7
                                && (rr.file == File::UP || m.k.front == ptxv_ir::Front::Sass))
                            {
                                s.wp(t, rr, x != 0)
                            }
                        }
                        _ => s.wr(t, rr, x, 1),
                    }
                }
            }
        }
    }

    /// Run `f` on the scratch machine for each lane in `which` (None: once, on the
    /// representative, the operands being the same in every lane), with the operands
    /// `vals` loaded. None if the operands are not known.
    fn per_lane<T>(
        &mut self,
        m: &Machine,
        c: &LaCfg,
        s: &mut Machine,
        vals: &[ptxv_ir::Val],
        f: &mut dyn FnMut(&mut Machine, usize) -> Option<T>,
    ) -> Option<Vec<Option<T>>> {
        match self.kind(m, c, vals) {
            Ok(true) => {
                self.load(m, c, s, vals, 0);
                let t = self.ts[0];
                Some(vec![f(s, t)])
            }
            Ok(false) => {
                let mut out = Vec::with_capacity(self.ts.len());
                for i in 0..self.ts.len() {
                    self.load(m, c, s, vals, i);
                    out.push(f(s, self.ts[i]));
                }
                Some(out)
            }
            Err(_) => None,
        }
    }

    /// Write `v` to `r` (lane-dependent writes inside a body are noted).
    fn set(&mut self, c: &mut LaCfg, r: ptxv_ir::Reg, v: Av) {
        let uf = la_uniform_file(r);
        if let Some((_, w)) = c.body.as_mut() {
            if !uf {
                w.insert(r);
            }
        }
        let v = if uf && matches!(v, Av::L(_) | Av::N | Av::E) {
            Av::U
        } else {
            v
        };
        c.env.insert(r, v);
    }

    /// Write `v` to `r` where guard value `g` holds.
    fn guarded(&mut self, m: &Machine, c: &mut LaCfg, r: ptxv_ir::Reg, v: Av, g: Av) {
        let nv = match g {
            Av::K(0) => return,
            Av::K(_) => v,
            _ => {
                let old = self.get(m, c, r);
                if old == v {
                    v
                } else if let (Av::L(_), Some(o), Some(n)) =
                    (g, self.lanes_of(old), self.lanes_of(v))
                {
                    let gs = self.lanes_of(g).unwrap();
                    let x = (0..self.ts.len())
                        .map(|i| if gs[i] != 0 { n[i] } else { o[i] })
                        .collect();
                    self.mk(x)
                } else if g == Av::U
                    && matches!(old, Av::K(_) | Av::U)
                    && matches!(v, Av::K(_) | Av::U)
                {
                    Av::U
                } else {
                    Av::N
                }
            }
        };
        self.set(c, r, nv);
    }

    /// May `a` and `b` (same pc and body) be joined: no register known in both with
    /// different values? (Configurations of different loop iterations, say, stay apart, so
    /// stage indices and phase parities stay known.)
    fn compatible(&mut self, m: &Machine, a: &LaCfg, b: &LaCfg) -> bool {
        if a.env == b.env {
            return true;
        }
        let known = |x: Av| matches!(x, Av::K(_) | Av::L(_));
        let (x, y) = (&a.env.0, &b.env.0);
        let (mut i, mut j) = (0, 0);
        while i < x.len() || j < y.len() {
            let (r, va, vb) = if j >= y.len() || i < x.len() && x[i].0 < y[j].0 {
                i += 1;
                (x[i - 1].0, Some(x[i - 1].1), None)
            } else if i >= x.len() || y[j].0 < x[i].0 {
                j += 1;
                (y[j - 1].0, None, Some(y[j - 1].1))
            } else {
                i += 1;
                j += 1;
                (x[i - 1].0, Some(x[i - 1].1), Some(y[j - 1].1))
            };
            let va = va.unwrap_or_else(|| self.get(m, a, r));
            let vb = vb.unwrap_or_else(|| self.get(m, b, r));
            if va != vb && known(va) && known(vb) {
                return false;
            }
        }
        true
    }

    /// A configuration covering both `a` and `b` (same pc and body rejoin point).
    fn join(&mut self, m: &Machine, a: &LaCfg, b: &LaCfg) -> LaCfg {
        let mut j = a.clone();
        if a.env != b.env {
            let (x, y) = (&a.env.0, &b.env.0);
            let mut out = Vec::with_capacity(x.len().max(y.len()));
            let (mut i, mut k) = (0, 0);
            while i < x.len() || k < y.len() {
                let (r, va, vb) = if k >= y.len() || i < x.len() && x[i].0 < y[k].0 {
                    i += 1;
                    (x[i - 1].0, Some(x[i - 1].1), None)
                } else if i >= x.len() || y[k].0 < x[i].0 {
                    k += 1;
                    (y[k - 1].0, None, Some(y[k - 1].1))
                } else {
                    i += 1;
                    k += 1;
                    (x[i - 1].0, Some(x[i - 1].1), Some(y[k - 1].1))
                };
                let va = va.unwrap_or_else(|| self.get(m, a, r));
                let vb = vb.unwrap_or_else(|| self.get(m, b, r));
                out.push((r, la_join_av(va, vb)));
            }
            j.env = LaEnv(out);
        }
        j.obs = a
            .obs
            .iter()
            .filter_map(|(k, &x)| b.obs.get(k).map(|&y| (*k, x.min(y))))
            .collect();
        j.lanes = a.lanes.max(b.lanes);
        if let (Some((_, wa)), Some((_, wb))) = (j.body.as_mut(), b.body.as_ref()) {
            wa.extend(wb.iter().copied());
        }
        j
    }

    fn mark(&mut self, pc: u32) {
        self.exec[pc as usize / 64] |= 1 << (pc % 64);
    }

    /// Lanes taking part, per lane (`g`: the guard): Some(bits) when known per lane.
    fn active(&self, g: Av) -> Option<Vec<bool>> {
        match g {
            Av::K(x) => Some(vec![x != 0; self.ts.len()]),
            Av::L(j) => Some(self.lv[j as usize].iter().map(|&x| x != 0).collect()),
            _ => None,
        }
    }

    /// Add arrivals at the mbarriers `targets` (per lane in `ts` order, or one entry for every
    /// lane); `cap`: at most this many lanes arrive.
    fn tally(
        &mut self,
        pc: u32,
        targets: &[Option<(u32, u64)>],
        g: Av,
        cap: u64,
        mult: u64,
        res: &mut LaResult,
    ) -> bool {
        let act = self.active(g);
        let mut per: HashMap<(u32, u64), u64> = HashMap::new();
        if targets.len() == 1 {
            let Some(k) = targets[0] else {
                return false;
            };
            let n = match &act {
                Some(a) => a.iter().filter(|&&x| x).count() as u64,
                None => self.ts.len() as u64,
            };
            if n > 0 {
                per.insert(k, n);
            }
        } else {
            for (i, k) in targets.iter().enumerate() {
                if act.as_ref().is_some_and(|a| !a[i]) {
                    continue;
                }
                let Some(k) = k else {
                    return false;
                };
                *per.entry(*k).or_insert(0) += 1;
            }
        }
        // In a loop summarized by a join, an arrival may repeat any number of times.
        let lp = self.scc[pc as usize] != u32::MAX && self.loops.contains(&self.scc[pc as usize]);
        for (k, n) in per {
            self.arr_at.entry(pc).or_default().insert(k);
            if lp {
                res.inf.insert(k);
            } else {
                res.arrive(k, n.min(cap).saturating_mul(mult));
            }
        }
        true
    }

    /// A join at `pc` (a configuration covered by, or merged into, one that ran from there):
    /// if `pc` is on a control-flow cycle, the abstract run no longer counts the iterations
    /// of that cycle, so every arrival on it becomes unbounded.
    fn summarize(&mut self, pc: u32, res: &mut LaResult) {
        let id = self.scc[pc as usize];
        if id == u32::MAX || !self.loops.insert(id) {
            return;
        }
        for (q, bars) in &self.arr_at {
            if self.scc[*q as usize] == id {
                res.inf.extend(bars.iter().copied());
            }
        }
    }

    /// One step of configuration `c`.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        m: &Machine,
        s: &mut Machine,
        c: &mut LaCfg,
        res: &mut LaResult,
        pc_bars: &dyn Fn(u32, u32, &mut BarSet),
        changed: &mut bool,
    ) -> LaStep {
        use ptxv_ir::Op;
        let code = &m.k.code;
        let pc = c.pc;
        if pc as usize >= code.len() {
            return LaStep::End;
        }
        // The warp rejoins at the BSYNC that ends an elected body.
        if c.body.as_ref().is_some_and(|b| b.0 == pc) {
            let (_, wr) = c.body.take().unwrap();
            for r in wr {
                c.env.insert(r, Av::N);
            }
            c.lanes = self.ts.len() as u32;
        }
        let inst = &code[pc as usize];
        let g = self.pred(m, c, inst.guard);
        let n_ts = self.ts.len() as u64;
        // At most this many lanes take part in one execution.
        let cap = if inst.uniform || g == Av::E {
            1
        } else {
            c.lanes as u64
        };
        let (w, loc) = (self.w, inst.loc);
        let wild = |why: &str| LaStep::Wild(format!("w{w} at {loc:#x}: {why}"));
        // Unknown arrival targets: unbounded arrivals wherever the learned footprints of
        // this pc arrive.
        let unknown_target = |res: &mut LaResult, changed: &mut bool| {
            let mut bars = HashSet::new();
            pc_bars(w, pc, &mut bars);
            for (id, b) in bars {
                if b == u64::MAX {
                    *changed |= res.inf_cta.insert(id);
                } else {
                    *changed |= res.inf.insert((id, b));
                }
            }
        };
        if !matches!(inst.op, Op::MbarTryWait { .. }) {
            self.mark(pc);
        }
        match &inst.op {
            Op::MbarTryWait {
                d,
                addr,
                parity,
                bit31,
            } => {
                if g == Av::K(0) {
                    self.mark(pc);
                    return LaStep::Next;
                }
                let peek = m.peeks.contains(&pc) && !m.bar_cfg.peekwait.contains(&pc);
                // The mbarrier and parity, if known and the same for every lane.
                let mut vals: Vec<ptxv_ir::Val> = [addr.base.clone(), addr.base2.clone()]
                    .into_iter()
                    .flatten()
                    .collect();
                vals.push(parity.clone());
                let b31 = *bit31;
                let tg = self.per_lane(m, c, s, &vals, &mut |s: &mut Machine, t: usize| {
                    let a = s.addr(t, addr).ok()?;
                    let crate::exec::Loc::Shared { cta, off } =
                        s.resolve(t, ptxv_ir::Space::Shared, a, 8).ok()?
                    else {
                        return None;
                    };
                    let p = s.rd(t, parity).ok()?;
                    let p = if b31 { (p >> 31) & 1 } else { p & 1 };
                    Some(((cta as u32, off as u64), p))
                });
                let target = tg.and_then(|v| {
                    let f = v[0]?;
                    v.iter().all(|x| *x == Some(f)).then_some(f)
                });
                if self.trace {
                    eprintln!(
                        "LATRACE w{w} {loc:#x} poll target={target:x?} g={g:?} obs={:x?}",
                        c.obs
                    );
                }
                if peek && m.bar_cfg.peekskip.contains_key(&pc) {
                    // Run as a failed test (`peekskip_set`; where its lanes disagree on the
                    // guard it runs as a peek: either result).
                    self.mark(pc);
                    let v = if matches!(g, Av::K(_)) {
                        Av::K(0)
                    } else {
                        Av::U
                    };
                    self.guarded(m, c, *d, v, g);
                    return LaStep::Next;
                }
                if peek {
                    // A peek: fails for sure if no phase of the other parity can have
                    // completed (the configuration is also kept waiting, to be run again if
                    // the bound grows), else either.
                    self.mark(pc);
                    let mut v = Av::U;
                    if let Some((k, p)) = target {
                        if let Some(mb) = m.mbars.get(&(k.0 as u16, k.1 as u32)) {
                            let lo = c.obs.get(&k).copied().unwrap_or(0);
                            let first = if (mb.phase as u64 + lo) % 2 != p {
                                lo
                            } else {
                                lo + 1
                            };
                            if first > res.bound(m, k) {
                                self.waiting.push((c.clone(), k, p));
                                v = Av::K(0);
                            }
                        }
                    }
                    self.guarded(m, c, *d, v, g);
                    return LaStep::Next;
                }
                match (target, g) {
                    (Some((k, p)), Av::K(1)) if m.mbars.contains_key(&(k.0 as u16, k.1 as u32)) => {
                        let mb = &m.mbars[&(k.0 as u16, k.1 as u32)];
                        let hi = res.bound(m, k);
                        let lo = c.obs.get(&k).copied().unwrap_or(0);
                        let first = if (mb.phase as u64 + lo) % 2 != p {
                            lo
                        } else {
                            lo + 1
                        };
                        if self.trace {
                            eprintln!("LATRACE   phase={} lo={lo} hi={hi} first={first}", mb.phase);
                        }
                        if first > hi {
                            return LaStep::Wait(k, p);
                        }
                        c.obs.insert(k, first);
                    }
                    _ => {}
                }
                self.mark(pc);
                // A blocking poll goes on only once it succeeds.
                self.guarded(m, c, *d, Av::K(1), g);
                LaStep::Next
            }
            Op::MbarArrive {
                addr,
                count,
                remote,
                ..
            } => {
                if g == Av::K(0) {
                    return LaStep::Next;
                }
                let n = match count {
                    None => 1,
                    Some(v) => match self.val(m, c, v) {
                        Av::K(x) => x,
                        _ => 1,
                    },
                };
                if n == 0 {
                    return LaStep::Next;
                }
                let vals: Vec<ptxv_ir::Val> = [addr.base.clone(), addr.base2.clone()]
                    .into_iter()
                    .flatten()
                    .collect();
                let space = if *remote {
                    ptxv_ir::Space::SharedCluster
                } else {
                    ptxv_ir::Space::Shared
                };
                let tg = self.per_lane(m, c, s, &vals, &mut |s: &mut Machine, t: usize| {
                    let a = s.addr(t, addr).ok()?;
                    match s.resolve(t, space, a, 8).ok()? {
                        crate::exec::Loc::Shared { cta, off } => Some((cta as u32, off as u64)),
                        _ => None,
                    }
                });
                if self.trace {
                    eprintln!(
                        "LATRACE w{w} {loc:#x} arrive targets={tg:x?} g={g:?} cap={cap} n={n}"
                    );
                }
                if !tg.is_some_and(|v| self.tally(pc, &v, g, cap, n, res)) {
                    unknown_target(res, changed);
                }
                *changed = true;
                LaStep::Next
            }
            Op::TcCommit { mbar, off, mask } => {
                if g == Av::K(0) {
                    return LaStep::Next;
                }
                let mut vals = vec![mbar.clone()];
                vals.extend(mask.iter().cloned());
                let off = *off;
                let tg = self.per_lane(m, c, s, &vals, &mut |s: &mut Machine, t: usize| {
                    let a = s.rd(t, mbar).ok()?.wrapping_add(off as u64);
                    let me = s.threads[t].cta;
                    let addrs: Vec<u64> = match mask {
                        None => vec![a],
                        Some(mv) => {
                            let mv = s.rd(t, mv).ok()? & 0xffff;
                            (0..16u64)
                                .filter(|r| mv >> r & 1 == 1)
                                .map(|r| (a & 0xff_ffff) | (r << 24))
                                .collect()
                        }
                    };
                    addrs
                        .into_iter()
                        .map(|x| {
                            s.rank_cta(me, x)
                                .ok()
                                .map(|cta| (cta as u32, x & 0xff_ffff))
                        })
                        .collect::<Option<Vec<_>>>()
                });
                let ok = tg.is_some_and(|v| {
                    // One commit per target, per lane.
                    let k = v
                        .iter()
                        .map(|x| x.as_ref().map_or(0, |y| y.len()))
                        .max()
                        .unwrap_or(0);
                    (0..k).all(|j| {
                        let col: Vec<Option<(u32, u64)>> = v
                            .iter()
                            .map(|x| x.as_ref().and_then(|y| y.get(j).copied()))
                            .collect();
                        self.tally(pc, &col, g, cap, 1, res)
                    }) && v.iter().all(|x| x.is_some())
                });
                if !ok {
                    unknown_target(res, changed);
                }
                *changed = true;
                LaStep::Next
            }
            Op::MbarInit { .. } | Op::MbarInval { .. } if g != Av::K(0) => {
                wild("mbarrier (re)initialization")
            }
            Op::ClcTryCancel { .. } | Op::CallAbs { .. } if g != Av::K(0) => {
                wild("unsupported operation")
            }
            // An asynchronous arrival (`.noinc`); the transaction form is ignored like bytes.
            Op::CpAsyncMbarArrive { tx: false, .. } if g != Av::K(0) => {
                wild("cp.async.mbarrier.arrive.noinc")
            }
            Op::BarSync {
                count: None,
                arrive: false,
                ..
            } if self.cut & 2 != 0 && inst.guard.is_always() => LaStep::End,
            Op::ClusterWait if self.cut & 1 != 0 => LaStep::End,
            Op::Exit | Op::Trap { .. } => match g {
                Av::K(0) => LaStep::Next,
                _ if c.body.is_some() => wild("exit in an elected body"),
                Av::K(_) => LaStep::End,
                Av::U => LaStep::Fork(vec![Some(pc + 1), None]),
                // Some lanes leave; the rest go on (counted as before: an upper bound).
                _ => LaStep::Next,
            },
            // BRA.DIV (E-COLLECTIVE): the whole split goes one way; it falls through only when
            // the guard holds in every lane and the lane mask equals the operand.
            Op::Bra {
                target,
                div: Some(dm),
                ..
            } => {
                if c.body.is_some() {
                    return wild("BRA.DIV in an elected body");
                }
                let target = *target as u32;
                let lanes = self
                    .ts
                    .iter()
                    .fold(0u32, |a, &t| a | 1 << m.threads[t].lane);
                let mv = self.val(m, c, dm);
                match (g, mv) {
                    (Av::K(0), _) => LaStep::Next,
                    (Av::K(_), Av::K(x)) if x as u32 == lanes => LaStep::Next,
                    (Av::K(_), Av::K(_)) | (Av::L(_), _) => LaStep::Jump(target),
                    _ => LaStep::Fork(vec![Some(target), Some(pc + 1)]),
                }
            }
            Op::Bra {
                target,
                any,
                tid0_falls,
                cond,
                div: None,
            } => {
                let target = *target as u32;
                let t = if let Some(v) = m.bar_cfg.fold.get(pc as usize).copied().flatten() {
                    // ASSUME(E-CONST-BRA), as the control-flow graph.
                    Av::K(v as u64)
                } else if *any {
                    match g {
                        Av::K(x) => Av::K((x != 0) as u64),
                        Av::L(_) | Av::E => Av::K(1),
                        _ => Av::U,
                    }
                } else {
                    let th = &m.threads[self.ts[0]];
                    let b = m.launch.g(th.cta as u32).block;
                    let falls =
                        *tid0_falls && th.tid[0] + b[0] * (th.tid[1] + b[1] * th.tid[2]) < 32;
                    let cv = match cond {
                        Some(p) => self.pred(m, c, *p),
                        None => Av::K(1),
                    };
                    if falls {
                        Av::K(0)
                    } else {
                        match (g, cv) {
                            (Av::K(0), _) | (_, Av::K(0)) => Av::K(0),
                            (Av::K(_), x) | (x, Av::K(_)) => x,
                            (Av::L(_), Av::L(_)) => {
                                let (a, b) =
                                    (self.lanes_of(g).unwrap(), self.lanes_of(cv).unwrap());
                                let v = a
                                    .iter()
                                    .zip(&b)
                                    .map(|(x, y)| (*x & *y != 0) as u64)
                                    .collect();
                                self.mk(v)
                            }
                            (Av::U, Av::U) => Av::U,
                            _ => Av::N,
                        }
                    }
                };
                match t {
                    Av::K(0) => LaStep::Next,
                    Av::K(_) => LaStep::Jump(target),
                    Av::U => LaStep::Fork(vec![Some(target), Some(pc + 1)]),
                    _ => {
                        // Lane-dependent: only an elected body (E-BSYNC): the lanes going one
                        // way wait at a BSYNC there while the others run the other way.
                        if c.body.is_some() {
                            return wild("divergence in an elected body");
                        }
                        let bs = |q: u32| {
                            matches!(code.get(q as usize).map(|i| &i.op), Some(Op::Bsync { .. }))
                        };
                        let ones = match t {
                            Av::L(j) => {
                                self.lv[j as usize].iter().filter(|&&x| x != 0).count() as u32
                            }
                            Av::E => 1,
                            _ => c.lanes,
                        };
                        let zeros = match t {
                            Av::L(j) => {
                                self.lv[j as usize].iter().filter(|&&x| x == 0).count() as u32
                            }
                            _ => c.lanes,
                        };
                        // `@!P BRA L` with P elected: one lane runs the body.
                        let elected_body = cond.is_none()
                            && !*any
                            && inst.guard.neg
                            && inst.guard.reg.is_some_and(|r| self.get(m, c, r) == Av::E);
                        if bs(target) {
                            c.lanes = if elected_body { 1 } else { zeros.min(c.lanes) };
                            c.body = Some((target, BTreeSet::new()));
                            LaStep::Next
                        } else if bs(pc + 1) {
                            c.lanes = ones.min(c.lanes);
                            c.body = Some((pc + 1, BTreeSet::new()));
                            LaStep::Jump(target)
                        } else {
                            wild("divergent branch")
                        }
                    }
                }
            }
            Op::Brx { base, off, targets } => match self.val(m, c, base) {
                Av::K(b) => {
                    let a = (b as i64).wrapping_add(*off);
                    let to = (a >> 4) as usize + m.k.entry_of(pc as usize);
                    if a & 15 != 0 || !targets.contains(&to) {
                        return wild("BRX target");
                    }
                    LaStep::Jump(to as u32)
                }
                Av::U => LaStep::Fork(targets.iter().map(|&q| Some(q as u32)).collect()),
                _ => wild("divergent BRX"),
            },
            Op::Call { target } => LaStep::Jump(*target as u32),
            Op::Ret { addr } => {
                let rets = m.bar_cfg.rets.get(pc as usize).cloned().unwrap_or_default();
                match self.val(m, c, addr) {
                    Av::K(a) => {
                        let to = (a >> 4) as usize + m.k.entry_of(pc as usize);
                        if !m.bar_cfg.rets.is_empty() && !rets.contains(&to) {
                            return wild("RET target");
                        }
                        LaStep::Jump(to as u32)
                    }
                    Av::U if !rets.is_empty() => {
                        LaStep::Fork(rets.iter().map(|&q| Some(q as u32)).collect())
                    }
                    _ => wild("unknown RET"),
                }
            }
            Op::Elect { d, lane } => {
                self.guarded(m, c, *d, Av::E, g);
                if let Some(l) = lane {
                    self.guarded(m, c, *l, Av::U, g);
                }
                LaStep::Next
            }
            Op::R2Ur { d, a, p, or } => {
                let v = match self.val(m, c, a) {
                    Av::K(x) => Av::K(x),
                    _ => Av::U,
                };
                let pv = if matches!(v, Av::K(_)) && !*or {
                    Av::K(0)
                } else {
                    Av::N
                };
                self.guarded(m, c, *d, v, g);
                if let Some(p) = p {
                    self.guarded(m, c, *p, pv, g);
                }
                LaStep::Next
            }
            op => {
                let dsts = op.dst_regs();
                if dsts.is_empty() || g == Av::K(0) {
                    return LaStep::Next;
                }
                let pure = matches!(
                    op,
                    Op::Mov { .. }
                        | Op::Int { .. }
                        | Op::MulWide { .. }
                        | Op::Imad { .. }
                        | Op::Iadd3 { .. }
                        | Op::CvtInt { .. }
                        | Op::Flo { .. }
                        | Op::Brev { .. }
                        | Op::AddMinMax { .. }
                        | Op::IntMnmx3 { .. }
                        | Op::P2r { .. }
                        | Op::Lea { .. }
                        | Op::LeaHi { .. }
                        | Op::Shf { .. }
                        | Op::Lop3 { .. }
                        | Op::Plop3 { .. }
                        | Op::Setp { .. }
                        | Op::Sel { .. }
                        | Op::Prmt { .. }
                        | Op::Bfe { .. }
                        | Op::Pack64 { .. }
                        | Op::Unpack64 { .. }
                        | Op::LdConst { .. }
                        | Op::Mapa { .. }
                        | Op::Cvta { .. }
                        | Op::Hfma2 { .. }
                        | Op::Hfma2Bf16 { .. }
                        | Op::FfmaBf16 { .. }
                        | Op::Hmnmx2Bf16 { .. }
                        | Op::Vhmnmx { .. }
                );
                let vals = op.src_vals();
                let out: Vec<(ptxv_ir::Reg, Av)> = if pure {
                    let dd = dsts.clone();
                    let r = self.per_lane(m, c, s, &vals, &mut |s: &mut Machine, t: usize| {
                        s.exec_pure(t, pc)
                            .then(|| dd.iter().map(|&r| la_rd(s, t, r)).collect::<Vec<u64>>())
                    });
                    match r {
                        Some(v) if v.iter().all(|x| x.is_some()) => {
                            let v: Vec<Vec<u64>> = v.into_iter().map(|x| x.unwrap()).collect();
                            dsts.iter()
                                .enumerate()
                                .map(|(j, &r)| {
                                    let col: Vec<u64> = if v.len() == 1 {
                                        vec![v[0][j]; n_ts as usize]
                                    } else {
                                        v.iter().map(|x| x[j]).collect()
                                    };
                                    (r, self.mk(col))
                                })
                                .collect()
                        }
                        Some(_) => dsts.iter().map(|&r| (r, Av::N)).collect(),
                        None => {
                            let k = self.kind(m, c, &vals).err().unwrap_or(Av::N);
                            dsts.iter().map(|&r| (r, k)).collect()
                        }
                    }
                } else {
                    let k = match op {
                        // Lockstep: a uniform address reads one value.
                        Op::Ld { .. } => match self.kind(m, c, &vals) {
                            Ok(true) | Err(Av::U) => Av::U,
                            _ => Av::N,
                        },
                        // The same value in every lane taking part.
                        Op::VoteAll { .. } | Op::VoteAny { .. } | Op::Redux { .. } => Av::U,
                        // A broadcast: every lane reads the same lane.
                        Op::Shfl { a, lane, mode, .. } => {
                            match (self.val(m, c, a), mode, self.val(m, c, lane)) {
                                (Av::K(x), _, _) => Av::K(x),
                                (_, ptxv_ir::ShflMode::Idx, Av::K(_))
                                    if c.body.is_none() && n_ts == 32 =>
                                {
                                    Av::U
                                }
                                _ => Av::N,
                            }
                        }
                        _ => Av::N,
                    };
                    dsts.iter().map(|&r| (r, k)).collect()
                };
                for (r, v) in out {
                    self.guarded(m, c, r, v, g);
                }
                LaStep::Next
            }
        }
    }
}

/// Strongly connected components of the control-flow graph (as `cfg_succ` with no cut,
/// calls and returns included), per pc; u32::MAX for a pc on no cycle.
fn la_scc(m: &Machine) -> Vec<u32> {
    let code = &m.k.code;
    let n = code.len();
    let rets = &m.bar_cfg.rets;
    let succ: Vec<Vec<usize>> = (0..n)
        .map(|pc| {
            use ptxv_ir::Op;
            let i = &code[pc];
            let next = (pc + 1 < n).then_some(pc + 1);
            let always = i.guard.is_always();
            let v: Vec<usize> = match &i.op {
                Op::Bra {
                    target,
                    tid0_falls: false,
                    cond: None,
                    div: None,
                    ..
                } if always => vec![*target],
                Op::Bra { target, .. } | Op::Call { target } => {
                    [Some(*target), next].into_iter().flatten().collect()
                }
                Op::Ret { .. } => match rets.get(pc) {
                    Some(r) if !r.is_empty() => {
                        r.iter().copied().chain(next.filter(|_| !always)).collect()
                    }
                    _ => (0..n).collect(),
                },
                Op::Brx { targets, .. } => targets.clone(),
                Op::CallAbs { .. } => (0..n).collect(),
                Op::Exit | Op::Trap { .. } if always => vec![],
                _ => next.into_iter().collect(),
            };
            v.into_iter().filter(|&q| q < n).collect()
        })
        .collect();
    // Iterative Tarjan.
    let mut index = vec![u32::MAX; n];
    let mut low = vec![0u32; n];
    let mut on = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut comp = vec![u32::MAX; n];
    let mut next_index = 0u32;
    let mut ncomp = 0u32;
    for root in 0..n {
        if index[root] != u32::MAX {
            continue;
        }
        let mut call: Vec<(usize, usize)> = vec![(root, 0)];
        index[root] = next_index;
        low[root] = next_index;
        next_index += 1;
        stack.push(root);
        on[root] = true;
        while let Some(&mut (v, ref mut k)) = call.last_mut() {
            if *k < succ[v].len() {
                let w = succ[v][*k];
                *k += 1;
                if index[w] == u32::MAX {
                    index[w] = next_index;
                    low[w] = next_index;
                    next_index += 1;
                    stack.push(w);
                    on[w] = true;
                    call.push((w, 0));
                } else if on[w] {
                    low[v] = low[v].min(index[w]);
                }
            } else {
                call.pop();
                if let Some(&(u, _)) = call.last() {
                    low[u] = low[u].min(low[v]);
                }
                if low[v] == index[v] {
                    let mut members = Vec::new();
                    loop {
                        let x = stack.pop().unwrap();
                        on[x] = false;
                        members.push(x);
                        if x == v {
                            break;
                        }
                    }
                    let cyclic = members.len() > 1 || succ[v].contains(&v);
                    if cyclic {
                        for x in members {
                            comp[x] = ncomp;
                        }
                        ncomp += 1;
                    }
                }
            }
        }
    }
    comp
}

/// A warp (live lanes `ts`) split in two: the lanes at one pc run, every other lane is at
/// (parked in, or about to run) one WARPSYNC / BSYNC whose mask covers every live lane, so
/// they wait there for the running ones. Returns (running pc, sync pc, running lanes).
fn la_rejoin(m: &Machine, w: u32, ts: &[usize]) -> Option<(u32, u32, u32)> {
    use ptxv_ir::Op;
    if std::env::var("PTXV_NO_BSYNC_POR").is_ok() {
        return None;
    }
    let live = ts.iter().fold(0u32, |a, &t| a | 1 << m.threads[t].lane);
    let covers = |pc: u32| -> bool {
        let t = ts[0];
        match &m.k.code[pc as usize].op {
            Op::WarpSync {
                mask,
                collective: false,
                ..
            } => m.rd(t, mask).is_ok_and(|x| x as u32 & live == live),
            Op::Bsync { b } => m.warps[w as usize].bmask[*b as usize] & live == live,
            _ => false,
        }
    };
    let mut groups: BTreeMap<u32, (u32, bool)> = BTreeMap::new();
    for &t in ts {
        let th = &m.threads[t];
        let e = groups.entry(th.pc).or_insert((0, false));
        e.0 += 1;
        e.1 |= th.st == St::WarpSync;
    }
    if groups.len() != 2 {
        return None;
    }
    let g: Vec<(u32, (u32, bool))> = groups.into_iter().collect();
    for (a, b) in [(0, 1), (1, 0)] {
        let ((r, (n, parked_r)), (l, _)) = (g[a], g[b]);
        if !parked_r && covers(l) {
            return Some((r, l, n));
        }
    }
    None
}

/// ASSUME(E-DRF, E-SPIN, E-BSYNC): the value-tracking look-ahead (`PTXV_NO_VALTRACK` turns
/// it off). From the current state, run every live warp but the candidate's abstractly, as
/// far as it can get without the candidate:
/// - registers hold abstract values (`Av`): known (the same in every lane, or per lane),
///   computed from the current registers through register-only instructions by the
///   executor itself on a scratch copy (`Machine::exec_pure`); or unknown (loads, ...),
///   either the same in every lane or not;
/// - a blocking mbarrier poll (spin model) whose mbarrier and parity are known passes only if
///   a phase of the other parity can have completed by then: between what this
///   configuration already observed and the upper bound on completions. The bound counts
///   arrivals per lane and per execution, from every configuration of every warp, plus the
///   pending commits; transaction bytes are ignored. Waiting configurations are retried as
///   the bounds grow (a least fixpoint). Unknown mbarrier or parity, peeks, named barriers
///   and cluster barriers (but those cut by the candidate, `cut_for`) pass;
/// - a branch on a known value goes one way (or the way `fold` decided, E-CONST-BRA); on a
///   value unknown but the same in every lane, both ways (alternative configurations); on a
///   lane-dependent value, only the elected-body shape is followed (the lanes going one way
///   wait at the BSYNC there, E-BSYNC, while the others run to it); anything else gives up;
/// - an arrival or commit at an unknown mbarrier counts as unbounded arrivals at every
///   mbarrier the learned footprints of its pc name.
///
/// A warp given up on (budget, divergence, (re)initialization) gets the gated region, with
/// unbounded arrivals at every mbarrier its learned footprints there name (`wild_bars`).
/// Soundness, by induction over any execution from here without the candidate: each warp's
/// steps are those of one of its configurations (same pc, known values equal, observed
/// completions a lower bound), and every arrival so far is counted in the bounds (the
/// learned footprints being complete for the mbarriers that held a configuration back,
/// which are recorded with the choice, as for the gate), so a poll that waits in the
/// look-ahead has not succeeded in the execution either.
fn lookahead(
    m: &Machine,
    cand: u32,
    skip: usize,
    cut_for: &dyn Fn(u32, u32) -> usize,
    wild_bars: &dyn Fn(u32, &mut BarSet),
    pc_bars: &dyn Fn(u32, u32, &mut BarSet),
) -> LaResult {
    use ptxv_ir::Op;
    let budget: usize = std::env::var("PTXV_LA_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4000);
    let code = &m.k.code;
    let words = code.len().div_ceil(64);
    let mut res = LaResult {
        exec: HashMap::new(),
        arr: HashMap::new(),
        inf: HashSet::new(),
        inf_cta: HashSet::new(),
        stopped: Vec::new(),
        why: Vec::new(),
    };
    for (k, o) in m.pending.iter().enumerate() {
        if k == skip {
            continue;
        }
        if let AsyncKind::Clc { others, mbar, .. } = &o.kind {
            for &c in others {
                res.arrive((c as u32, *mbar as u64), 1);
            }
        }
        if let AsyncKind::Commit { cta, mbar }
        | AsyncKind::Clc { cta, mbar, .. }
        | AsyncKind::CpArrive {
            cta,
            mbar,
            tx: false,
            ..
        } = &o.kind
        {
            res.arrive((*cta as u32, *mbar as u64), 1);
        }
    }
    let mut lanes_of: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (t, th) in m.live_threads() {
        if th.st != St::Exited && th.warp != cand {
            lanes_of.entry(th.warp).or_default().push(t);
        }
    }
    let scc = std::rc::Rc::new(la_scc(m));
    let mut sims: Vec<LaWarp> = Vec::new();
    for (&w, ts) in &lanes_of {
        let mut sim = LaWarp {
            w,
            ts: ts.clone(),
            init: Default::default(),
            lv: Vec::new(),
            lvi: HashMap::new(),
            work: Vec::new(),
            waiting: Vec::new(),
            seen: Default::default(),
            ids: 0,
            exec: vec![0; words],
            steps: 0,
            wild: None,
            wild_done: false,
            cut: cut_for(cand, w),
            at: HashMap::new(),
            trace: std::env::var("PTXV_LA_TRACE").is_ok_and(|v| v == format!("{cand},{w}")),
            scc: scc.clone(),
            loops: HashSet::new(),
            arr_at: HashMap::new(),
        };
        // Lanes blocked at a CTA or cluster barrier sit at it (they run it again here: they
        // stop there if it is cut, else pass).
        let odd = ts.iter().any(|&t| {
            let th = &m.threads[t];
            match th.st {
                St::Run | St::WarpSync => false,
                St::Bar => !matches!(code[th.pc as usize].op, Op::BarSync { .. }),
                St::ClusterWait => !matches!(code[th.pc as usize].op, Op::ClusterWait),
                _ => true,
            }
        });
        let pcs: BTreeSet<u32> = ts
            .iter()
            .filter(|&&t| m.threads[t].st != St::WarpSync)
            .map(|&t| m.threads[t].pc)
            .collect();
        let any_parked = ts.iter().any(|&t| m.threads[t].st == St::WarpSync);
        let cfg = |pc: u32, lanes: u32, body| LaCfg {
            pc,
            env: LaEnv::default(),
            obs: BTreeMap::new(),
            lanes,
            body,
            hist: LaHist::default(),
        };
        if odd {
            sim.wild = Some(format!("w{w}: blocked or unlaunched lanes"));
        } else if pcs.len() == 1 && !any_parked {
            sim.work
                .push(cfg(pcs.first().copied().unwrap(), ts.len() as u32, None));
        } else if let Some((r, l, running)) = la_rejoin(m, w, ts) {
            // One running split; the rest of the warp waits (or is about to wait) at a
            // WARPSYNC / BSYNC that needs it: an elected body (E-BSYNC).
            sim.work.push(cfg(r, running, Some((l, BTreeSet::new()))));
        } else if let (1, Some(l)) = (pcs.len(), bsync_parked(m, w)) {
            // A running split with the rest parked at a BSYNC only it releases: an elected
            // body (E-BSYNC).
            let running = ts
                .iter()
                .filter(|&&t| m.threads[t].st != St::WarpSync)
                .count();
            sim.work.push(cfg(
                pcs.first().copied().unwrap(),
                running as u32,
                Some((l, BTreeSet::new())),
            ));
        } else {
            sim.wild = Some(format!(
                "w{w}: starts diverged {pcs:x?} parked={any_parked}"
            ));
        }
        sims.push(sim);
    }
    // Merge points: branch targets, and the pcs after branches and polls.
    let mut merge = vec![0u64; words];
    for (pc, i) in code.iter().enumerate() {
        let mut set = |q: usize| {
            if q < code.len() {
                merge[q / 64] |= 1 << (q % 64);
            }
        };
        match &i.op {
            Op::Bra { target, .. } | Op::Call { target } => {
                set(*target);
                set(pc + 1);
            }
            Op::Brx { targets, .. } => targets.iter().for_each(|&q| set(q)),
            Op::Ret { .. } | Op::Exit => set(pc + 1),
            _ => {}
        }
    }
    for r in &m.bar_cfg.rets {
        for &q in r {
            if q < code.len() {
                merge[q / 64] |= 1 << (q % 64);
            }
        }
    }
    let mut scratch = m.clone();
    let mut rounds = 0usize;
    loop {
        rounds += 1;
        let mut changed = false;
        for sim in sims.iter_mut() {
            if sim.wild.is_some() {
                continue;
            }
            // Waiting configurations whose mbarrier may now have completed enough phases.
            let waiting = std::mem::take(&mut sim.waiting);
            let mut retry: Vec<LaCfg> = Vec::new();
            for (c, k, p) in waiting {
                let mb = &m.mbars[&(k.0 as u16, k.1 as u32)];
                let lo = c.obs.get(&k).copied().unwrap_or(0);
                let first = if (mb.phase as u64 + lo) % 2 != p {
                    lo
                } else {
                    lo + 1
                };
                if first <= res.bound(m, k) {
                    retry.push(c);
                } else {
                    sim.waiting.push((c, k, p));
                }
            }
            'cfg: while let Some((mut c, mut again)) = retry
                .pop()
                .map(|c| (c, true))
                .or_else(|| sim.work.pop().map(|c| (c, false)))
            {
                loop {
                    // (A configuration run again after waiting is not covered by its own
                    // earlier run.)
                    if std::mem::take(&mut again) {
                    } else if merge[c.pc as usize / 64] >> (c.pc % 64) & 1 == 1 {
                        let key = (c.pc, c.body.as_ref().map(|b| b.0));
                        let mut reps = sim.at.remove(&key).unwrap_or_default();
                        let pos = reps.iter().position(|r| sim.compatible(m, &r.0, &c));
                        let pos = match pos {
                            None if reps.len() >= LA_REPS => Some(0),
                            p => p,
                        };
                        let covered = match pos {
                            Some(i) => {
                                // Back at a representative it descends from: a cycle.
                                if c.hist.has(reps[i].1) {
                                    sim.summarize(c.pc, &mut res);
                                }
                                let mut nj = sim.join(m, &reps[i].0, &c);
                                if nj == reps[i].0 {
                                    true
                                } else {
                                    nj.hist.union(&c.hist);
                                    nj.hist.add(reps[i].1);
                                    c = nj.clone();
                                    reps[i].0 = nj;
                                    false
                                }
                            }
                            None => {
                                let id = sim.ids;
                                sim.ids += 1;
                                c.hist.add(id);
                                reps.push((c.clone(), id));
                                false
                            }
                        };
                        sim.at.insert(key, reps);
                        if covered {
                            // Covered by what already ran from here.
                            break;
                        }
                    }
                    sim.steps += 1;
                    if sim.steps > budget {
                        sim.wild = Some(format!(
                            "w{}: budget at {:#x} (work {}, seen {}, env {})",
                            sim.w,
                            code.get(c.pc as usize).map_or(0, |i| i.loc),
                            sim.work.len(),
                            sim.seen.len(),
                            c.env.0.len()
                        ));
                        break 'cfg;
                    }
                    match sim.step(m, &mut scratch, &mut c, &mut res, pc_bars, &mut changed) {
                        LaStep::Next => c.pc += 1,
                        LaStep::Jump(q) => c.pc = q,
                        LaStep::End => break,
                        LaStep::Wait(k, p) => {
                            sim.waiting.push((c, k, p));
                            break;
                        }
                        LaStep::Fork(alts) => {
                            for q in alts.into_iter().flatten() {
                                let mut x = c.clone();
                                x.pc = q;
                                match sim.seen.get(&x) {
                                    Some(&id) => {
                                        if c.hist.has(id) {
                                            sim.summarize(q, &mut res);
                                        }
                                    }
                                    None => {
                                        let id = sim.ids;
                                        sim.ids += 1;
                                        x.hist.add(id);
                                        sim.seen.insert(x.clone(), id);
                                        sim.work.push(x);
                                    }
                                }
                            }
                            break;
                        }
                        LaStep::Wild(why) => {
                            sim.wild = Some(why);
                            break 'cfg;
                        }
                    }
                }
            }
        }
        for sim in sims.iter_mut() {
            if sim.wild.is_some() && !sim.wild_done {
                sim.wild_done = true;
                sim.work.clear();
                sim.waiting.clear();
                let mut bars = HashSet::new();
                wild_bars(sim.w, &mut bars);
                for (id, b) in bars {
                    if b == u64::MAX {
                        res.inf_cta.insert(id);
                    } else {
                        res.inf.insert((id, b));
                    }
                }
            }
        }
        let more = sims.iter().any(|s| {
            s.wild.is_none()
                && (!s.work.is_empty()
                    || s.waiting.iter().any(|(c, k, p)| {
                        let mb = &m.mbars[&(k.0 as u16, k.1 as u32)];
                        let lo = c.obs.get(k).copied().unwrap_or(0);
                        let first = if (mb.phase as u64 + lo) % 2 != *p {
                            lo
                        } else {
                            lo + 1
                        };
                        first <= res.bound(m, *k)
                    }))
        });
        if !more {
            break;
        }
        if rounds > 100_000 {
            // Not converged: no opinion on any warp.
            res.exec = sims.iter().map(|s| (s.w, None)).collect();
            res.why.push("rounds".into());
            return res;
        }
    }
    if std::env::var("PTXV_LA_STATS").is_ok() {
        let v: Vec<String> = sims
            .iter()
            .map(|s| {
                format!(
                    "w{}:{}{}/{}",
                    s.w,
                    s.steps,
                    if s.wild.is_some() { "W" } else { "" },
                    s.seen.len()
                )
            })
            .collect();
        eprintln!("PORLASTAT cand={cand} rounds={rounds} {}", v.join(" "));
    }
    for sim in sims {
        if let Some(why) = sim.wild {
            res.why.push(why);
            res.exec.insert(sim.w, None);
        } else {
            for (_, k, _) in &sim.waiting {
                res.stopped.push(*k);
            }
            res.exec.insert(sim.w, Some(sim.exec));
        }
    }
    res.stopped.sort_unstable();
    res.stopped.dedup();
    res
}

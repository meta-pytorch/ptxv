/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! ASSUME(E-SLICE): thread slices. Exhaustive exploration of a launch (group) one set of warps
//! at a time, for launches whose warps fall into groups that share nothing. The generalization
//! of cluster slices (E-CTA-INDEP, `explore::explore_sliced`) to warps of several CTAs on
//! several GPUs: DeepEP's dispatch is one (source rank, destination rank) flow per slice, the
//! sender warps of one GPU and the receiver warps of the other.
//!
//! - **Partition** (`partition`): one deterministic run of the launch with the full footprint
//!   of every step (`Footprint`, all spaces) attributed to the warp that took it (an async
//!   operation to its issuer's warp). Two warps are joined when they touch a common item and
//!   one writes it: a byte of shared or global memory, an mbarrier, a TMEM word or allocator,
//!   a cluster barrier, a named barrier (any two users of one `(CTA, id)`), or an all-thread
//!   `bar.sync` (every warp of its CTA). The slices are the connected components. This only
//!   proposes the partition; the check below decides.
//! - **Exploration** (`restrict`): each slice is explored on the launch's machine with every
//!   thread outside the slice exited, recording every footprint item of every explored
//!   transition (`SFoot`, `GFoot`; dry-run look-ahead steps included).
//! - **Cross-slice check** (`conflict`): no item that one slice wrote is read or written by
//!   another; an all-thread barrier or cluster barrier of one slice is in a CTA (cluster) with
//!   no thread of another slice. A conflict makes the verdict incomplete (the slices were not
//!   independent; it is not by itself a race).
//!
//! Argument (ASSUMPTIONS.md E-SLICE): take any execution E of the whole launch and its first
//! step e of slice A whose behaviour depends on another slice B, i.e. e accesses an item that
//! a B step w earlier in E wrote (or e writes an item a B step read; or e is an all-thread
//! barrier of a CTA with B threads). Before e, every step of A and of B depended on its own
//! slice only, so A's steps up to e and B's steps up to w are executions of A alone and of B
//! alone, which the per-slice explorations cover; so e and w are both recorded, and the check
//! fails. If the check passes, no step of E depends on another slice: E is an interleaving of
//! one execution per slice, each reachable alone, and every interleaving of such executions is
//! an execution. Races, deadlocks, faults and final states of the launch are then exactly the
//! combinations of the slices' (a deadlock of the launch is a state where every slice is
//! terminal or deadlocked, at least one deadlocked).

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::explore::Action;
use crate::explore::Verdict;
use crate::machine::*;

/// A footprint item: (space, id, lo, hi, write).
pub type Item = (u8, u32, u64, u64, bool);

/// Items of one step, with memory ranges: global (space 1, id = allocation) and shared
/// (space 0, id = CTA) included.
fn step_items(fp: &Footprint, sf: &SFoot) -> Vec<Item> {
    let mut v: Vec<Item> = Vec::new();
    for (list, w) in [(&fp.r, false), (&fp.w, true)] {
        for &(s, id, lo, hi) in list {
            match s {
                0 | 1 => v.push((s, id, lo, hi, w)),
                // 10 / 11 are the bytes of 0 / 1 again; 4, 5, 8, 9 are not shared (SFoot::add).
                4 | 5 | 8 | 9 | 10 | 11 => {}
                // Every use of a barrier generation or mbarrier phase key conflicts.
                14 | 7 | 12 => v.push((s, id, lo, hi, true)),
                _ => v.push((s, id, lo, hi, w)),
            }
        }
    }
    for &(s, id, lo, hi, w) in sf.items.lock().unwrap().iter() {
        if s == SPACE_CTA_ALL || s == 13 {
            v.push((s, id, lo, hi, w));
        }
    }
    v
}

/// Union-find over warp ids.
struct Uf(HashMap<u32, u32>);

impl Uf {
    fn find(&mut self, x: u32) -> u32 {
        let p = *self.0.entry(x).or_insert(x);
        if p == x {
            return x;
        }
        let r = self.find(p);
        self.0.insert(x, r);
        r
    }
    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.0.insert(ra.max(rb), ra.min(rb));
        }
    }
}

/// Pairs of owners whose intervals of one (space, id) overlap with at least one write.
fn overlaps(mut iv: Vec<(u64, u64, bool, u32)>, mut hit: impl FnMut(u32, u32)) {
    iv.sort_unstable_by_key(|x| (x.0, x.1));
    let mut active: Vec<(u64, u64, bool, u32)> = Vec::new();
    for x in iv {
        active.retain(|a| a.1 > x.0);
        for a in &active {
            if a.3 != x.3 && (a.2 || x.2) && a.0 < x.1 {
                hit(a.3, x.3);
            }
        }
        // Many readers of one read-only range: keep one reader per owner and access kind.
        if !active
            .iter()
            .any(|a| a.3 == x.3 && a.2 == x.2 && a.0 <= x.0 && a.1 >= x.1)
        {
            active.push(x);
        }
    }
}

/// The warps (global warp ids) of each proposed slice, from one deterministic run of `m`
/// (`run_once`'s schedule). Warps with no live thread are left out; one slice means no split.
pub fn partition(m: &Machine) -> Vec<Vec<u32>> {
    let mut m = m.clone();
    m.gfoot = None;
    m.sfoot = None;
    // (space, id) -> accesses as (start, end, write, warp).
    type Accesses = BTreeMap<(u8, u32), Vec<(u64, u64, bool, u32)>>;
    let mut by: Accesses = BTreeMap::new();
    let mut live: Vec<u32> = m
        .live_threads()
        .filter(|(_, t)| t.st != St::Exited)
        .map(|(_, t)| t.warp)
        .collect();
    live.dedup();
    let cta_warps: BTreeMap<u16, Vec<u32>> = {
        let mut c: BTreeMap<u16, Vec<u32>> = BTreeMap::new();
        for (_, t) in m.live_threads().filter(|(_, t)| t.st != St::Exited) {
            let v = c.entry(t.cta).or_default();
            if v.last() != Some(&t.warp) {
                v.push(t.warp);
            }
        }
        c
    };
    let mut uf = Uf(HashMap::new());
    let mut steps = 0usize;
    loop {
        let acts = m.actions();
        let pick = acts
            .iter()
            .find(|(a, _)| matches!(a, Action::Split { .. }))
            .or(acts.first());
        let Some((a, _)) = pick.cloned() else { break };
        let warp = match &a {
            Action::Split { warp, .. } => Some(*warp),
            Action::Async(j) => Some(m.threads[m.pending[*j].issuer as usize].warp),
            _ => None,
        };
        let sf = Arc::new(SFoot::new(&[]));
        m.sfoot = Some(sf.clone());
        m.fp = Some(Box::default());
        m.apply(&a);
        m.findings.clear();
        let mut fp = m.fp.take().map(|b| *b).unwrap_or_default();
        m.sfoot = None;
        fp.normalize();
        steps += 1;
        let Some(w) = warp else { continue };
        for (s, id, lo, hi, wr) in step_items(&fp, &sf) {
            if s == SPACE_CTA_ALL {
                // An all-thread barrier joins every warp of its CTA.
                for &x in cta_warps.get(&(id as u16)).into_iter().flatten() {
                    uf.union(w, x);
                }
                continue;
            }
            by.entry((s, id)).or_default().push((lo, hi, wr, w));
        }
    }
    let _ = steps;
    for (_, iv) in by {
        overlaps(iv, |a, b| uf.union(a, b));
    }
    let mut comps: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for w in live {
        let r = uf.find(w);
        comps.entry(r).or_default().push(w);
    }
    let mut v: Vec<Vec<u32>> = comps.into_values().collect();
    v.sort_by_key(|s| s[0]);
    v
}

/// `m` with every live thread outside `warps` exited, and slice footprints on. Fails when a
/// live grid's stream predecessor is still live (a grid boundary would join the slices).
pub fn restrict(m: &Machine, warps: &[u32]) -> Result<Machine, String> {
    let mut m = m.clone();
    for g in 0..m.launch.grids.len() {
        let live = |m: &Machine, g: usize| m.grid_threads(g).any(|t| m.threads[t].st != St::Exited);
        if live(&m, g) {
            if let Some(p) = m.launch.prev_grid(g) {
                if live(&m, p) {
                    return Err(format!(
                        "launch {g} waits for launch {p}, which is live in the same slice run"
                    ));
                }
            }
        }
    }
    let keep: std::collections::BTreeSet<u32> = warps.iter().copied().collect();
    for t in 0..m.threads.len() {
        if m.threads[t].st != St::Exited && !keep.contains(&m.threads[t].warp) {
            m.threads[t].make_mut().st = St::Exited;
        }
    }
    let live: Vec<usize> = (0..m.threads.len())
        .filter(|&t| m.threads[t].st != St::Exited)
        .collect();
    m.live = match (live.first(), live.last()) {
        (Some(&a), Some(&b)) => a..b + 1,
        _ => 0..0,
    };
    let sizes: Vec<usize> = m.gmem.iter().map(|g| g.bytes.len()).collect();
    m.gfoot = Some(Arc::new(GFoot::new(&sizes)));
    let ss: Vec<usize> = (0..m.smem.len()).map(|c| m.smem[c].bytes.len()).collect();
    let sf = SFoot::new(&ss);
    // The CTAs and clusters this slice has threads in: an all-thread barrier or cluster
    // barrier of another slice there would depend on these threads' exits.
    for &t in &live {
        let cta = m.threads[t].cta;
        sf.items
            .lock()
            .unwrap()
            .insert((SPACE_CTA_ALL, cta as u32, 0, 1, false));
        let cid = m.launch.cluster_of(cta as u32).0;
        sf.items.lock().unwrap().insert((13, cid, 0, 1, false));
    }
    m.sfoot = Some(Arc::new(sf));
    Ok(m)
}

/// What one slice's exploration touched: global (read, write) bitsets per allocation, shared
/// (read, write) per CTA, other items.
#[derive(Clone, Default)]
pub struct Touched {
    pub g: Vec<(Vec<u64>, Vec<u64>)>,
    pub s: Vec<(Vec<u64>, Vec<u64>)>,
    pub items: Vec<Item>,
}

impl Touched {
    pub fn of(gfoot: Option<&GFoot>, sfoot: Option<&SFoot>) -> Touched {
        let g = gfoot
            .map(|g| {
                g.snapshot()
                    .into_iter()
                    .map(|(r, w, x)| (r, w.iter().zip(&x).map(|(a, b)| a | b).collect()))
                    .collect()
            })
            .unwrap_or_default();
        let (s, items) = sfoot.map(|s| s.snapshot()).unwrap_or_default();
        Touched { g, s, items }
    }
}

/// The first conflict between two slices' footprints, described.
pub fn conflict(a: &Touched, b: &Touched) -> Option<String> {
    let bits = |x: &[(Vec<u64>, Vec<u64>)], y: &[(Vec<u64>, Vec<u64>)]| -> Option<(usize, u64)> {
        for (i, ((rx, wx), (ry, wy))) in x.iter().zip(y).enumerate() {
            for k in 0..wx.len().max(wy.len()) {
                let g = |v: &Vec<u64>| v.get(k).copied().unwrap_or(0);
                let c = (g(wx) & (g(ry) | g(wy))) | (g(wy) & g(rx));
                if c != 0 {
                    return Some((i, 64 * k as u64 + c.trailing_zeros() as u64));
                }
            }
        }
        None
    };
    if let Some((al, o)) = bits(&a.g, &b.g) {
        return Some(format!("global byte {o:#x} of allocation {al}"));
    }
    if let Some((c, o)) = bits(&a.s, &b.s) {
        return Some(format!("shared memory byte {o:#x} of CTA {c}"));
    }
    for x in &a.items {
        for y in &b.items {
            if x.0 == y.0 && x.1 == y.1 && x.2 < y.3 && y.2 < x.3 && (x.4 || y.4) {
                let what = match x.0 {
                    SPACE_CTA_ALL => format!("an all-thread bar.sync of CTA {}", x.1),
                    13 => format!("barrier.cluster of cluster {}", x.1),
                    14 => format!("named barrier {} of CTA {}", x.2.max(y.2) >> 32, x.1),
                    s => format!("footprint space {s}, id {}, index {:#x}", x.1, x.2.max(y.2)),
                };
                return Some(what);
            }
        }
    }
    None
}

/// Explore each slice of `m` (`explore` = the caller's exploration of one machine, returning
/// its verdict and whatever else it collects), then the cross-slice check. Returns the slices'
/// warps, each verdict and result, and the conflict if any.
#[allow(clippy::type_complexity)]
pub fn explore_slices<R>(
    m: &Machine,
    slices: &[Vec<u32>],
    explore: &mut dyn FnMut(Machine) -> (Verdict, R),
    report: &mut dyn FnMut(usize, &[u32], &Verdict),
) -> Result<(Vec<(Verdict, R)>, Option<String>), String> {
    let mut out = Vec::new();
    let mut touched: Vec<Touched> = Vec::new();
    for (i, s) in slices.iter().enumerate() {
        let ms = restrict(m, s)?;
        let t0 = Instant::now();
        let (gf, sf) = (ms.gfoot.clone(), ms.sfoot.clone());
        let (mut v, r) = explore(ms);
        if v.seconds == 0.0 {
            v.seconds = t0.elapsed().as_secs_f64();
        }
        report(i, s, &v);
        touched.push(Touched::of(gf.as_deref(), sf.as_deref()));
        out.push((v, r));
    }
    let mut bad = None;
    'o: for i in 0..touched.len() {
        for j in i + 1..touched.len() {
            if let Some(c) = conflict(&touched[i], &touched[j]) {
                bad = Some(format!("slices {i} and {j} both touch {c}, one writing"));
                break 'o;
            }
        }
    }
    Ok((out, bad))
}

/// Describe a slice: its warps per CTA (`k<launch>.cta<c>.w<a>-<b>,<c>`, warp indices in the
/// CTA).
pub fn describe(m: &Machine, warps: &[u32]) -> String {
    let mut by: BTreeMap<u16, (u32, Vec<u32>)> = BTreeMap::new();
    for t in 0..m.threads.len() {
        let th = &m.threads[t];
        let e = by.entry(th.cta).or_insert((th.warp, Vec::new()));
        if warps.contains(&th.warp) && e.1.last() != Some(&th.warp) {
            e.1.push(th.warp);
        }
    }
    let mut parts = Vec::new();
    for (cta, (first, ws)) in by {
        if ws.is_empty() {
            continue;
        }
        let g = m.launch.grid_index(cta as u32);
        let mut rs: Vec<String> = Vec::new();
        let mut i = 0;
        while i < ws.len() {
            let mut j = i;
            while j + 1 < ws.len() && ws[j + 1] == ws[j] + 1 {
                j += 1;
            }
            let (a, b) = (ws[i] - first, ws[j] - first);
            rs.push(if a == b {
                format!("{a}")
            } else {
                format!("{a}-{b}")
            });
            i = j + 1;
        }
        let c = cta as u32 - m.launch.grids[g].cta0;
        parts.push(format!("k{g}.cta{c}.w{}", rs.join(",")));
    }
    parts.join(" + ")
}

/// `groups::explore_meet` one thread slice at a time (E-SLICE), when `partition` splits `cur`:
/// the slices' verdicts combined, and the meet of every combination of their terminal states
/// (`Meet::compose`, from one deterministic run of the whole group). Falls back to exploring
/// `cur` whole when it does not split; a cross-slice conflict makes the verdict incomplete.
pub fn explore_meet_sliced(
    cur: &Machine,
    max_states: usize,
    report: &mut dyn FnMut(usize, &str, &Verdict),
) -> (Verdict, crate::groups::Meet) {
    let mut parts = partition(cur);
    // Debugging: PTXV_SLICE_ONLY=i,j explores only these slices (the verdict is incomplete).
    let only: Option<Vec<usize>> = std::env::var("PTXV_SLICE_ONLY")
        .ok()
        .map(|x| x.split(',').filter_map(|y| y.parse().ok()).collect());
    if let Some(o) = &only {
        let mut i = 0;
        parts.retain(|_| {
            i += 1;
            o.contains(&(i - 1))
        });
    }
    if parts.len() < 2 && only.is_none() {
        let (mut v, meet) = crate::groups::explore_meet(cur.clone(), max_states);
        v.notes
            .push("thread slices: the group does not split (one slice)".into());
        return (v, meet);
    }
    let names: Vec<String> = parts.iter().map(|s| describe(cur, s)).collect();
    eprintln!("thread slices (E-SLICE): {}", names.len());
    for (i, n) in names.iter().enumerate() {
        eprintln!("  slice {i}: {n}");
    }
    let r = explore_slices(
        cur,
        &parts,
        &mut |m| crate::groups::explore_meet(m, max_states),
        &mut |i, _, v| report(i, &names[i], v),
    );
    let (res, bad) = match r {
        Ok(x) => x,
        Err(e) => {
            let (mut v, meet) = crate::groups::explore_meet(cur.clone(), max_states);
            v.notes.push(format!("thread slices not applicable: {e}"));
            return (v, meet);
        }
    };
    let mut all = Verdict {
        states: 0,
        transitions: 0,
        complete: true,
        terminal_states: 1,
        distinct_outputs: 1,
        findings: vec![],
        assumptions: vec![],
        notes: vec![format!(
            "explored one thread slice at a time (E-SLICE, {} slices: {})",
            names.len(),
            names.join("; ")
        )],
        seconds: 0.0,
        outputs: vec![],
    };
    let mut meets = Vec::new();
    for (v, meet) in res {
        all.states += v.states;
        all.transitions += v.transitions;
        all.complete &= v.complete;
        all.terminal_states = all.terminal_states.saturating_mul(v.terminal_states);
        all.distinct_outputs = all.distinct_outputs.saturating_mul(v.distinct_outputs);
        all.seconds += v.seconds;
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
            if !all.notes.contains(&n) {
                all.notes.push(n);
            }
        }
        meets.push(meet);
    }
    all.assumptions.push("E-SLICE".into());
    if only.is_some() {
        all.complete = false;
        all.notes
            .push("PTXV_SLICE_ONLY: some slices not explored".into());
    }
    if let Some(c) = bad {
        all.complete = false;
        all.notes.push(format!(
            "thread slices are not independent ({c}): the verdict covers each slice alone"
        ));
    }
    let (t0, fs, ok) = crate::explore::run_once(cur.clone());
    if !ok {
        for f in fs {
            if !all.findings.iter().any(|x| x.sig == f.sig) {
                all.findings.push(f);
            }
        }
    }
    if ok && all.distinct_outputs == 1 {
        all.outputs = vec![t0.outputs()];
    }
    all.findings
        .sort_by(|a, b| (&a.kind, &a.sig).cmp(&(&b.kind, &b.sig)));
    let meet = crate::groups::Meet::compose(cur, &t0, meets);
    (all, meet)
}

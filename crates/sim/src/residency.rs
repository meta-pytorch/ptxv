/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Limited residency. ASSUME(E-RESIDENCY): see ASSUMPTIONS.md.
//!
//! The default model makes every CTA of a launch resident at once (E-GRID). On a real GPU
//! a non-cooperative launch gets no such guarantee: other kernels may hold SMs, and CTAs
//! that do not fit wait until earlier ones exit. A kernel whose CTAs wait on each other
//! (grid-wide spin barriers, persistent producer/consumer CTAs, DeepEP's sender and
//! receiver CTAs) then deadlocks.
//!
//! The residency model (`Env::residency = Some(R)`): at most R CTAs per GPU hold a residency
//! slot. Every CTA starts unlaunched; a cluster becomes resident as a unit (its CTAs are
//! co-scheduled by the hardware), in any order, when R slots minus the resident CTAs leave
//! room for it; a CTA frees its slot when all its threads exited and none of its async
//! operations is pending. Exploration (`explore::explore` on `with_residency(m, R)`) is then
//! exhaustive over schedules *and* dispatch orders. A state where every resident CTA is
//! blocked and a CTA is left without a slot is a deadlock ("never becomes resident").
//!
//! `analyze` computes, per launch, the residency bounds that deadlock and the smallest R from
//! which every bound is deadlock-free:
//! - kernels with a well-synchronization certificate (E-WELLSYNC): the *launch-set search*.
//!   Every schedule performs the certified run's events with its happens-before, so the
//!   events a set of launched CTAs can perform, run until nothing moves, do not depend on
//!   the order (each event is enabled by earlier events only, and nothing disables it). The
//!   state after launching a set L of clusters is therefore a function of L. The search
//!   enumerates the reachable sets (adding one cluster at a time), and for each the
//!   bottleneck b(L): the fewest slots with which L can be reached. L deadlocks at R when
//!   b(L) <= R, it is not terminal, and no unlaunched cluster fits next to its resident
//!   CTAs. This is a proof for every R.
//! - other kernels: exploration with each bound (exhaustive when complete). When it is not,
//!   the launch-set search over deterministic runs still finds real deadlocks (every state
//!   it visits is reached by an execution), but its "deadlock-free" is not a proof.

use std::collections::BTreeMap;

use anyhow::Result;
use ptxv_ir::File;
use ptxv_ir::Kernel;
use ptxv_ir::Op;
use ptxv_ir::Space;
use serde::Serialize;

use crate::explore::Action;
use crate::machine::*;

/// SMs of a GB300 on the dev box (`multi_processor_count`, CUDA 13.0, 2026-11-04).
pub const GB300_SMS: u32 = 152;
/// Per-SM limits of sm_103a (CUDA occupancy calculator values, checked against the device
/// properties of the dev box): shared memory, registers, threads, resident CTAs.
pub const SM_SMEM: u32 = 233_472;
pub const CTA_SMEM_RESERVED: u32 = 1024;
pub const SM_REGS: u32 = 65_536;
pub const SM_THREADS: u32 = 2048;
pub const SM_CTAS: u32 = 32;

/// Launch environment: limited residency and the TMEM co-tenant (tenant.rs). Static.
#[derive(Clone, Debug, Default)]
pub struct Env {
    /// At most this many CTAs per GPU resident at once (None: all, E-RESIDENCY default).
    pub residency: Option<u32>,
    /// The bound applies to this GPU only (None: to every GPU).
    pub residency_device: Option<u32>,
    /// The launch is cooperative (case `cooperative`): every CTA is co-resident or the
    /// launch fails. Reported, not modeled.
    pub cooperative: bool,
    /// E-TMEM-COTENANT: allocations may land at any free column run (tenant.rs).
    pub tmem_cotenant: bool,
    /// E-TMEM-PERMIT: a CTA sharing an SM (`tmem_share`) cannot allocate while its partner
    /// holds the allocation permit.
    pub tmem_permit: bool,
    /// CTA pairs placed on one SM: they share its 512 TMEM columns.
    pub tmem_share: Vec<(u16, u16)>,
}

impl Env {
    pub fn from_case(m: &ptxv_case::Manifest) -> Result<Env> {
        let residency = match std::env::var("PTXV_RESIDENCY").ok() {
            Some(v) if v == "none" || v.is_empty() => None,
            Some(v) => Some(
                v.parse()
                    .map_err(|_| anyhow::anyhow!("PTXV_RESIDENCY: not a number: {v}"))?,
            ),
            None => m.residency,
        };
        if residency == Some(0) {
            anyhow::bail!("residency must be at least 1");
        }
        Ok(Env {
            residency,
            residency_device: None,
            cooperative: m.cooperative,
            tmem_cotenant: match std::env::var("PTXV_TMEM_COTENANT").as_deref() {
                Ok("1") => true,
                Ok("0") => false,
                _ => m.tmem_cotenant,
            },
            tmem_permit: m.tmem_permit.unwrap_or(true),
            tmem_share: vec![],
        })
    }
}

/// `m` with residency bound `r` (CTAs per GPU): every CTA that has not started yet becomes
/// unlaunched (later grids already are, E-PDL).
pub fn with_residency(m: Machine, r: u32) -> Machine {
    with_residency_on(m, r, None)
}

/// `with_residency` with the bound on GPU `device` only (None: every GPU).
pub fn with_residency_on(mut m: Machine, r: u32, device: Option<u32>) -> Machine {
    let mut l = (*m.launch).clone();
    l.env.residency = Some(r);
    l.env.residency_device = device;
    m.launch = std::sync::Arc::new(l);
    for t in 0..m.threads.len() {
        if m.threads[t].st == St::Run
            && m.threads[t].pc == m.launch.g(m.threads[t].cta as u32).entry
        {
            m.threads[t].make_mut().st = St::Unlaunched;
        }
    }
    m
}

impl Machine {
    /// Resident CTAs per GPU: launched, with a live thread or a pending async operation.
    pub fn residency_counts(&self) -> Vec<u32> {
        let n = self.launch.ctas() as usize;
        let (mut launched, mut live) = (vec![false; n], vec![false; n]);
        for (_, t) in self.live_threads() {
            let c = t.cta as usize;
            match t.st {
                St::Unlaunched => {}
                St::Exited => launched[c] = true,
                _ => {
                    launched[c] = true;
                    live[c] = true;
                }
            }
        }
        for o in &self.pending {
            live[self.threads[o.issuer as usize].cta as usize] = true;
        }
        let ndev = self
            .launch
            .grids
            .iter()
            .map(|g| g.device)
            .max()
            .unwrap_or(0) as usize
            + 1;
        let mut out = vec![0; ndev];
        for c in 0..n {
            if launched[c] && live[c] {
                out[self.launch.g(c as u32).device as usize] += 1;
            }
        }
        out
    }

    /// ASSUME(E-RESIDENCY): may CTA `cta` (with its cluster) become resident now, given the
    /// per-GPU counts `counts` (`residency_counts`)? Only a cluster's rank-0 CTA has a launch
    /// action; it launches the whole cluster (`launch_unit`).
    pub fn residency_admits(&self, counts: &[u32], cta: u16) -> bool {
        let Some(r) = self.launch.env.residency else {
            return true;
        };
        if self.launch.cluster_of(cta as u32).1 != 0 {
            return false;
        }
        let g = self.launch.g(cta as u32);
        self.launch
            .env
            .residency_device
            .is_some_and(|d| d != g.device)
            || counts[g.device as usize] + g.cluster_size() <= r
    }

    /// Launch CTA `cta`; under limited residency its whole cluster.
    pub fn launch_unit(&mut self, cta: u16) {
        if self.launch.env.residency.is_none() {
            self.launch_cta(cta);
            return;
        }
        let cs = self.launch.cluster_size(cta as u32);
        for r in 0..cs {
            let Some(c) = self.launch.cta_of_rank(cta as u32, r) else {
                continue;
            };
            let c = c as u16;
            if self
                .threads
                .iter()
                .any(|t| t.cta == c && t.st == St::Unlaunched)
            {
                self.launch_cta(c);
            }
        }
    }

    /// Why an unlaunched CTA stays so in a deadlock, under limited residency (its grid may
    /// launch, but no slot is free).
    pub fn residency_why(&self, cta: u16) -> Option<String> {
        let r = self.launch.env.residency?;
        let gi = self.launch.grid_index(cta as u32);
        if !self.grid_gate_open(gi) {
            return None;
        }
        let g = &self.launch.grids[gi];
        if self
            .launch
            .env
            .residency_device
            .is_some_and(|d| d != g.device)
        {
            return None;
        }
        let counts = self.residency_counts();
        Some(format!(
            "{} never becomes resident (E-RESIDENCY): {} of at most {r} CTAs resident on GPU {}, all blocked{}",
            self.cta_name(cta),
            counts[g.device as usize],
            g.device,
            if g.cluster_size() > 1 {
                format!(", and its cluster needs {} slots", g.cluster_size())
            } else {
                String::new()
            }
        ))
    }
}

/// Loads (pcs) of poll loops on global or generic memory: waits on other CTAs (or GPUs).
pub fn cross_cta_waits(k: &Kernel) -> Vec<usize> {
    let global = |op: &Op| match op {
        Op::Ld { space, .. } | Op::Atom { space, .. } => {
            matches!(space, Space::Global | Space::Generic)
        }
        _ => false,
    };
    let mut v: Vec<usize> = crate::spin::poll_loops(&k.code, k.front)
        .into_iter()
        .filter(|(l, r)| r.is_ok() && global(&k.code[*l].op))
        .map(|(l, _)| l)
        .collect();
    for (h, b) in crate::spin::wpoll_loops(&k.code, k.front) {
        if let Some(l) = (h..=b.min(k.code.len() - 1)).find(|&p| global(&k.code[p].op)) {
            v.push(l);
        }
    }
    v.sort();
    v.dedup();
    v
}

/// `cross_cta_waits` plus loads on global / generic memory inside loops whose poll shape
/// E-SPIN did not accept (they may still be waits).
pub fn cross_cta_waits_or_suspect(k: &Kernel) -> Vec<usize> {
    let mut v = cross_cta_waits(k);
    for (l, r) in crate::spin::poll_loops(&k.code, k.front) {
        if r.is_err()
            && matches!(&k.code[l].op, Op::Ld { space, .. } | Op::Atom { space, .. }
                if matches!(space, Space::Global | Space::Generic))
        {
            v.push(l);
        }
    }
    v.sort();
    v.dedup();
    v
}

/// The note a verdict carries when it relies on every CTA being co-resident: the kernel
/// waits on other CTAs and some GPU runs more than one launch unit.
pub fn reliance_note(m: &Machine) -> Option<String> {
    if m.launch.env.residency.is_some() {
        return None;
    }
    let waits = cross_cta_waits(&m.k);
    if waits.is_empty() {
        return None;
    }
    let mut per_dev: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
    for g in &m.launch.grids {
        let e = per_dev.entry(g.device).or_default();
        e.0 += g.ctas();
        e.1 = e.1.max(g.cluster_size());
    }
    let multi: Vec<String> = per_dev
        .iter()
        .filter(|(_, (n, cs))| n > cs)
        .map(|(d, (n, _))| format!("GPU {d}: {n} CTAs"))
        .collect();
    if multi.is_empty() {
        return None;
    }
    let locs: Vec<String> = waits
        .iter()
        .take(6)
        .map(|&p| format!("{:#x}", m.k.code[p].loc))
        .collect();
    Some(format!(
        "relies on E-RESIDENCY: every CTA is assumed co-resident ({}); poll loop(s) at {}{} wait on other CTAs, and a non-cooperative launch does not guarantee co-residency. `ptxv check --residency auto` computes the residency the kernel needs",
        multi.join(", "),
        locs.join(", "),
        if waits.len() > 6 { ", ..." } else { "" }
    ))
}

// ------------------------------------------------------------------------- occupancy

/// What a GPU can hold of one grid at once.
#[derive(Clone, Debug, Serialize)]
pub struct Occupancy {
    pub sms: u32,
    /// CTAs per SM (None: unknown, registers not known on the PTX front-end).
    pub per_sm: Option<u32>,
    /// CTAs per SM allowed by every known limit (an upper bound when `per_sm` is None).
    pub upper: u32,
    pub limiter: String,
}

/// Registers per thread of a SASS grid: the highest R register named, plus one, rounded up
/// to 8 (ptxas's allocation granularity). None on the PTX front-end.
fn regs_per_thread(m: &Machine, gi: usize) -> Option<u32> {
    if m.k.front != ptxv_ir::Front::Sass {
        return None;
    }
    let g = &m.launch.grids[gi];
    let code = m.grid_code(g.cta0);
    let mut hi = 0u32;
    for i in code {
        for r in i.op.src_regs().into_iter().chain(i.op.dst_regs()) {
            if r.file == File::R && r.idx < 255 {
                hi = hi.max(r.idx as u32 + 1);
            }
        }
    }
    Some((hi + 2).div_ceil(8) * 8)
}

pub fn occupancy(m: &Machine, gi: usize) -> Occupancy {
    let g = &m.launch.grids[gi];
    let threads = g.threads_per_cta();
    let warps = threads.div_ceil(32);
    let mut lim: Vec<(u32, String)> = vec![
        (SM_CTAS, "CTAs per SM".into()),
        (
            SM_THREADS / (warps * 32).max(1),
            format!("threads ({threads})"),
        ),
        (
            SM_SMEM / (g.smem_bytes + CTA_SMEM_RESERVED),
            format!("shared memory ({} B)", g.smem_bytes),
        ),
    ];
    let regs = regs_per_thread(m, gi);
    if let Some(r) = regs {
        // Registers are allocated per warp in units of 256.
        let per_warp = (r * 32).div_ceil(256) * 256;
        lim.push((
            SM_REGS / (per_warp * warps).max(1),
            format!("registers ({r}/thread)"),
        ));
    }
    let (n, why) = lim.into_iter().min_by_key(|x| x.0).unwrap();
    Occupancy {
        sms: GB300_SMS,
        per_sm: regs.map(|_| n),
        upper: n,
        limiter: if regs.is_some() {
            why
        } else {
            format!("{why}; registers unknown on the PTX front-end")
        },
    }
}

// ------------------------------------------------------------------------- analysis

/// One GPU's part of the analyzed launches, and what its residency bound needs. Each GPU
/// is analyzed with the others unlimited.
#[derive(Clone, Debug, Serialize)]
pub struct DeviceReport {
    pub device: u32,
    pub ctas: u32,
    /// CTAs per cluster (a launch unit; the largest of the GPU's launches).
    pub unit: u32,
    pub occupancy: Occupancy,
    /// Bounds R (CTAs resident on this GPU) that deadlock, each with an example.
    pub deadlocks: BTreeMap<u32, String>,
    /// Smallest R from which every bound up to `ctas` is deadlock-free; None: deadlocks even
    /// with every CTA resident.
    pub min: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub devices: Vec<DeviceReport>,
    pub method: String,
    /// The minima are proven (certificate + complete search, or complete explorations).
    pub proven: bool,
    pub states: usize,
    pub cooperative: bool,
    pub notes: Vec<String>,
}

impl Report {
    /// The verdict lines: what each GPU needs and whether the launch guarantees it.
    pub fn lines(&self) -> Vec<String> {
        let mut v = vec![format!(
            "residency: {} ({}), {} states",
            self.method,
            if self.proven {
                "proof"
            } else {
                "not a proof; the deadlocks listed are real"
            },
            self.states
        )];
        for d in &self.devices {
            let head = format!("GPU {}: {} CTAs (units of {})", d.device, d.ctas, d.unit);
            let Some(n) = d.min else {
                v.push(format!("  {head}: deadlocks even with every CTA resident"));
                continue;
            };
            let below: Vec<u32> = d.deadlocks.keys().copied().collect();
            let need_txt = if below.is_empty() && !self.proven {
                "no residency deadlock found (bounds not all decided)".to_string()
            } else if below.is_empty() {
                format!(
                    "deadlock-free at every residency (>= {n} co-resident CTAs, one launch unit)"
                )
            } else if !self.proven {
                format!(
                    "residency deadlock at R = {below:?}: needs at least {n} co-resident CTAs (higher bounds not all decided)"
                )
            } else {
                let iff = (d.unit..n).all(|r| d.deadlocks.contains_key(&r));
                format!(
                    "deadlock-free {} >= {n} co-resident CTAs (residency deadlock at R = {below:?})",
                    if iff { "iff" } else { "for" }
                )
            };
            let cap = d.occupancy.per_sm.map(|p| p * d.occupancy.sms);
            let g = if !self.proven {
                format!("needs at least {n} (lower bound)")
            } else if n <= d.unit {
                "guaranteed (one launch unit: a cluster is co-scheduled)".to_string()
            } else if self.cooperative {
                match cap {
                    Some(c) if d.ctas > c => format!(
                        "cooperative launch, but {} CTAs exceed the capacity {c}: the launch fails",
                        d.ctas
                    ),
                    _ => "guaranteed by the cooperative launch".to_string(),
                }
            } else {
                match cap {
                    Some(c) if n > c => format!(
                        "cannot be met: capacity {} SMs x {} CTAs/SM = {c} ({}); deadlocks even on an idle GPU",
                        d.occupancy.sms,
                        d.occupancy.per_sm.unwrap(),
                        d.occupancy.limiter
                    ),
                    Some(c) => format!(
                        "NOT guaranteed: non-cooperative launch; fits an idle GPU (capacity {} SMs x {} CTAs/SM = {c}, limit: {}), but other kernels holding SMs can leave fewer than {n} resident",
                        d.occupancy.sms,
                        d.occupancy.per_sm.unwrap(),
                        d.occupancy.limiter
                    ),
                    None => format!(
                        "NOT guaranteed: non-cooperative launch (capacity at most {} SMs x {} CTAs/SM, limit: {})",
                        d.occupancy.sms, d.occupancy.upper, d.occupancy.limiter
                    ),
                }
            };
            v.push(format!("  {head}: {need_txt}; {g}"));
            for (r, ex) in &d.deadlocks {
                v.push(format!(
                    "    R = {r}: {}",
                    ex.lines().take(6).collect::<Vec<_>>().join(" | ")
                ));
            }
        }
        for n in &self.notes {
            v.push(format!("  note: {n}"));
        }
        v
    }

    /// The single-GPU minimum (a case's `min_residency`: 0 for "deadlocks anyway").
    pub fn min_of(&self, device: u32) -> Option<u32> {
        self.devices
            .iter()
            .find(|d| d.device == device)
            .map(|d| d.min.unwrap_or(0))
    }
}

/// A cluster of unlaunched CTAs (leader = rank 0).
#[derive(Clone, Debug)]
struct Unit {
    leader: u16,
    size: u32,
    device: u32,
    grid: usize,
}

fn units(m: &Machine) -> Vec<Unit> {
    let mut v = Vec::new();
    for c in 0..m.launch.ctas() {
        let c16 = c as u16;
        if m.launch.cluster_of(c).1 != 0 {
            continue;
        }
        if !m
            .threads
            .iter()
            .any(|t| t.cta == c16 && t.st == St::Unlaunched)
        {
            continue;
        }
        let gi = m.launch.grid_index(c);
        let g = &m.launch.grids[gi];
        v.push(Unit {
            leader: c16,
            size: g.cluster_size(),
            device: g.device,
            grid: gi,
        });
    }
    v
}

/// Run the deterministic policy of `run_once` (splits first, then async completions), never
/// launching, until nothing moves.
fn quiesce(m: &mut Machine, fs: &mut Vec<Finding>) {
    for _ in 0..200_000_000u64 {
        let acts = m.actions();
        let pick = acts
            .iter()
            .find(|(a, _)| matches!(a, Action::Split { .. }))
            .or_else(|| acts.iter().find(|(a, _)| !matches!(a, Action::Launch(_))));
        let Some((a, _)) = pick.cloned() else {
            return;
        };
        m.apply(&a);
        fs.append(&mut m.findings);
    }
}

/// Result of the launch-set search: per GPU, the bounds that deadlock.
pub struct Search {
    pub states: usize,
    pub complete: bool,
    pub deadlocks: BTreeMap<u32, BTreeMap<u32, String>>,
    pub findings: Vec<Finding>,
}

/// The launch-set search (module doc), for every GPU at once: each set L of launched
/// clusters carries, per GPU d, the bottleneck b_d(L) (the fewest slots on d with which L
/// can be reached while the other GPUs are unlimited). L deadlocks for GPU d at R when
/// b_d(L) <= R, L is not terminal, and every cluster that could launch is on d and does not
/// fit. `m0`: the machine before any CTA of the analyzed launches started. At most `cap`
/// launch sets.
pub fn launch_search(m0: &Machine, cap: usize) -> Search {
    let m0 = with_residency(m0.clone(), u32::MAX);
    let us = units(&m0);
    let ndev = m0.launch.grids.iter().map(|g| g.device).max().unwrap_or(0) as usize + 1;
    let mut per_dev = vec![0u32; ndev];
    for u in &us {
        per_dev[u.device as usize] += u.size;
    }
    let mut out = Search {
        states: 0,
        complete: true,
        deadlocks: BTreeMap::new(),
        findings: Vec::new(),
    };
    if us.len() > 127 {
        out.complete = false;
        return out;
    }
    let mut start = m0;
    quiesce(&mut start, &mut out.findings);
    let mut level: BTreeMap<u128, (Vec<u32>, Machine)> = BTreeMap::new();
    level.insert(0, (vec![0; ndev], start));
    while !level.is_empty() {
        let mut next: BTreeMap<u128, (Vec<u32>, Machine)> = BTreeMap::new();
        for (mask, (b, m)) in level {
            out.states += 1;
            if m.terminal() {
                continue;
            }
            let counts = m.residency_counts();
            let mut gates: BTreeMap<usize, bool> = BTreeMap::new();
            let open: Vec<(usize, u32)> = us
                .iter()
                .enumerate()
                .filter(|(i, u)| {
                    mask >> i & 1 == 0
                        && *gates
                            .entry(u.grid)
                            .or_insert_with(|| m.grid_gate_open(u.grid))
                })
                .map(|(i, u)| (i, counts[u.device as usize] + u.size))
                .collect();
            for d in 0..ndev {
                if per_dev[d] == 0 {
                    continue;
                }
                if open.iter().any(|&(i, _)| us[i].device as usize != d) {
                    continue; // another GPU can still launch
                }
                for r in b[d].max(1)..=per_dev[d] {
                    let e = out.deadlocks.entry(d as u32).or_default();
                    if open.iter().all(|&(_, need)| need > r) && !e.contains_key(&r) {
                        let mut shown = m.clone();
                        let mut l = (*shown.launch).clone();
                        l.env.residency = Some(r);
                        l.env.residency_device = Some(d as u32);
                        shown.launch = std::sync::Arc::new(l);
                        let (_, msg) = shown.deadlock_report();
                        e.insert(
                            r,
                            format!(
                                "{} of {} launch unit(s) launched, {} CTA(s) resident on GPU {d}, all blocked; {}",
                                mask.count_ones(),
                                us.len(),
                                counts[d],
                                msg.lines().skip(1).collect::<Vec<_>>().join("\n")
                            ),
                        );
                    }
                }
            }
            for (i, need) in open {
                let child = mask | 1u128 << i;
                let mut nb = b.clone();
                let d = us[i].device as usize;
                nb[d] = nb[d].max(need);
                if let Some(e) = next.get_mut(&child) {
                    for (x, y) in e.0.iter_mut().zip(&nb) {
                        *x = (*x).min(*y);
                    }
                    continue;
                }
                if out.states + next.len() >= cap {
                    out.complete = false;
                    continue;
                }
                let mut c = m.clone();
                c.launch_unit(us[i].leader);
                c.findings.clear();
                quiesce(&mut c, &mut out.findings);
                next.insert(child, (nb, c));
            }
        }
        level = next;
    }
    out
}

/// Is the machine's launch decided by the certificate (weak LDS/STS model, then strong)?
pub fn certified(m: &Machine) -> bool {
    if let Some(w) = crate::smem::weak(&m.k) {
        let mut mw = m.clone();
        mw.k = std::sync::Arc::new(w);
        if crate::wellsync::certify(mw).verdict.is_some() {
            return true;
        }
    }
    crate::wellsync::certify(m.clone()).verdict.is_some()
}

fn devices(m: &Machine) -> Vec<DeviceReport> {
    let mut per: BTreeMap<u32, (u32, u32, usize)> = BTreeMap::new();
    for (gi, g) in m.launch.grids.iter().enumerate() {
        if !m.grid_threads(gi).any(|t| m.threads[t].st != St::Exited) {
            continue;
        }
        let e = per.entry(g.device).or_insert((0, 1, gi));
        e.0 += g.ctas();
        e.1 = e.1.max(g.cluster_size());
    }
    per.into_iter()
        .map(|(d, (n, u, gi))| DeviceReport {
            device: d,
            ctas: n,
            unit: u,
            occupancy: occupancy(m, gi),
            deadlocks: BTreeMap::new(),
            min: None,
        })
        .collect()
}

/// The residency analysis of a machine before any CTA of its live launches started (module
/// doc): per GPU, with the other GPUs unlimited. `max_states` bounds each exploration, `cap`
/// the launch-set search.
pub fn analyze(m0: &Machine, max_states: usize, cap: usize) -> Report {
    let mut notes = Vec::new();
    let mut devs = devices(m0);
    let (proven, method, states);
    let live_code = |m: &Machine| -> Vec<usize> {
        // Grids with live threads only (held-back launch groups do not run).
        let mut v = Vec::new();
        for (gi, g) in m.launch.grids.iter().enumerate() {
            if m.grid_threads(gi).any(|t| m.threads[t].st != St::Exited) {
                let lo = g.entry as usize;
                let hi = m
                    .launch
                    .grids
                    .get(gi + 1)
                    .map_or(m.k.code.len(), |n| n.entry as usize);
                v.extend(lo..hi);
            }
        }
        v
    };
    let code = live_code(m0);
    let waits: Vec<usize> = cross_cta_waits_or_suspect(&m0.k)
        .into_iter()
        .filter(|p| code.contains(p))
        .collect();
    if waits.is_empty() {
        // No CTA can block on another one outside its cluster: every wait is on its own
        // threads, its cluster (co-scheduled), its async operations, its SM's tensor memory or
        // an earlier grid that launched before it (E-PDL). A launch unit that becomes resident
        // therefore runs to completion as it does with every CTA resident.
        let (_, _, ok) = crate::explore::run_once(m0.clone());
        if !ok {
            notes.push(
                "the deterministic run does not terminate even with every CTA resident".into(),
            );
            for d in devs.iter_mut() {
                d.deadlocks
                    .insert(d.ctas, "deadlock with every CTA resident".into());
            }
        }
        proven = ok;
        method =
            "no cross-CTA waits (no poll loop on global memory in the live kernels)".to_string();
        states = 1;
    } else if certified(m0) {
        let s = launch_search(m0, cap);
        for d in devs.iter_mut() {
            d.deadlocks = s.deadlocks.get(&d.device).cloned().unwrap_or_default();
        }
        proven = s.complete;
        if !s.complete {
            notes.push(format!(
                "launch-set search stopped at {cap} sets: the deadlocks listed are real, the minimum is not proven"
            ));
        }
        method = "certified launch-set search (E-WELLSYNC)".to_string();
        states = s.states;
    } else {
        // Deadlocks the launch-set search finds over deterministic runs are real; exploration
        // decides the other bounds.
        let s = launch_search(m0, cap);
        let (mut complete, mut st) = (true, s.states);
        // With every CTA of a GPU allowed, a late launch is a schedule in which the CTA has
        // not stepped yet: the bound is the default model, explored once with its reductions.
        let mut full: Option<crate::explore::Verdict> = None;
        for d in devs.iter_mut() {
            let found = s.deadlocks.get(&d.device).cloned().unwrap_or_default();
            for r in d.unit..=d.ctas {
                if let Some(e) = found.get(&r) {
                    d.deadlocks.insert(r, e.clone());
                    continue;
                }
                let v = if r == d.ctas {
                    full.get_or_insert_with(|| crate::explore::explore(m0.clone(), max_states))
                        .clone()
                } else {
                    crate::explore::explore(
                        with_residency_on(m0.clone(), r, Some(d.device)),
                        max_states,
                    )
                };
                st += v.states;
                complete &= v.complete;
                if let Some(f) = v.findings.iter().find(|f| f.kind == "deadlock") {
                    d.deadlocks
                        .insert(r, f.msg.lines().skip(1).collect::<Vec<_>>().join("\n"));
                }
            }
        }
        proven = complete;
        method = if complete {
            "exploration per bound (deadlocks first searched over deterministic runs)".to_string()
        } else {
            notes.push(format!(
                "exploration incomplete at {max_states} states for some bound: the deadlocks listed are real, the minimum is not proven"
            ));
            "exploration per bound (incomplete) + launch-set search over runs".to_string()
        };
        states = st;
    }
    for d in devs.iter_mut() {
        d.deadlocks.retain(|&r, _| r >= d.unit);
        d.min = if d.deadlocks.contains_key(&d.ctas) {
            None
        } else {
            Some(d.deadlocks.keys().max().map_or(d.unit, |r| r + 1))
        };
    }
    Report {
        devices: devs,
        method,
        proven,
        states,
        cooperative: m0.launch.env.cooperative,
        notes,
    }
}

/// Machines at the start of each launch group: the earlier groups run once each
/// (`explore::run_once`, as `groups::Opts::run_first`), later groups held back. A group
/// whose predecessor did not terminate gets the error.
pub fn group_machines(full: &Machine, groups: &[Vec<usize>]) -> Vec<Result<Machine, String>> {
    let hold = |mut m: Machine, keep: &[usize]| {
        for g in 0..m.launch.grids.len() {
            if !keep.contains(&g) {
                for t in m.grid_threads(g) {
                    m.threads[t].make_mut().st = St::Exited;
                }
            }
        }
        let live: Vec<usize> = (0..m.threads.len())
            .filter(|&t| m.threads[t].st != St::Exited)
            .collect();
        m.live = match (live.first(), live.last()) {
            (Some(&a), Some(&b)) => a..b + 1,
            _ => 0..0,
        };
        m
    };
    let mut out = Vec::new();
    let mut cur = Some(hold(full.clone(), &groups[0]));
    for i in 0..groups.len() {
        let Some(m) = cur.take() else {
            out.push(Err(format!("group {} did not terminate in one run", i)));
            continue;
        };
        out.push(Ok(m.clone()));
        if i + 1 < groups.len() {
            let (fin, _, ok) = crate::explore::run_once(m);
            if ok {
                cur = Some(crate::groups::boundary(fin, full, &groups[i + 1], &[]));
            }
        }
    }
    out
}

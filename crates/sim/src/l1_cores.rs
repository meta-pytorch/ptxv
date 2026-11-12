/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Co-resident kernels and outside agents in the L1 model (opt-in; E-L1-CORES, E-L1-HAVOC).
//!
//! E-HW-L1 (l1.rs) models GB300's non-coherent per-SM L1 for the case's own CTAs, each on its
//! own SM, and nothing else on the GPU (E-L1-PLACE, E-L1-CLOSED). A stale read needs a stale
//! copy of the line on the reader's SM, and in a closed world only the case's own loads make
//! one. This module drops those two assumptions, separately:
//!
//! - **Co-residency** (case `l1_coresident`, `PTXV_L1_CORES=1`, E-L1-CORES). Launches that
//!   are not stream-ordered (`after = -1`, other streams) run at the same time, and their
//!   CTAs may sit on one SM and share its lines. Which CTAs share an SM is a nondeterministic
//!   choice: `explore_placements` runs the explorer once per placement (`placements`) and
//!   merges the verdicts. A CTA's launch drops the lines other grids left on its SM (the
//!   kernel boundary, probes/l1_kbound), and a line filled by a co-resident grid after that
//!   hits for it. DPOR orders fills against writes of the line and against invalidations of
//!   the SM (footprint spaces 20 / 21), so it reaches schedules in which a co-resident load
//!   lands between a reader's invalidation and a remote write.
//! - **Havoc** (case `l1_havoc`, `PTXV_L1_HAVOC=1`, E-L1-HAVOC). Any agent outside the case
//!   may fill any line of a case buffer on any SM at any time (a legal co-resident reader of
//!   other bytes of the line does exactly that). A weak load R is reported
//!   (`stale-l1-havoc:<pc>`) when the last write W of bytes it reads comes from another SM, W's
//!   grid is not stream-ordered before R's and W does not happen-before R's launch (the kernel
//!   boundary would drop the copy), and no
//!   L1 invalidation on R's SM is ordered after W and before R (W -> I -> R in
//!   happens-before). Then a fill placed after the last such invalidation and before W lands
//!   leaves a copy without W's bytes that R hits. Like the predictive check, this covers every
//!   schedule with the run's happens-before order, without modelling the scheduler.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::explore::Verdict;
use crate::l1::L1;
use crate::l1::L1Ev;
use crate::l1::LINE;
use crate::machine::*;

/// DPOR footprint space: L1 line contents (id = allocation, range = the line's bytes). Loads
/// read it, stores and atomics write it.
pub const SPACE_LINE: u8 = 20;
/// DPOR footprint space: an SM's L1 (id = SM, index = allocation << 40 | line). Loads read
/// their line's key; invalidations, atomics and CTA launches write every key of the SM.
pub const SPACE_SM: u8 = 21;

/// GB300 per-SM residency limits (CUDA occupancy calculator, sm_100 family).
pub const MAX_CTAS_PER_SM: u32 = 32;
pub const MAX_THREADS_PER_SM: u32 = 2048;
pub const MAX_SMEM_PER_SM: u32 = 228 * 1024;
/// Shared memory the driver reserves per resident CTA.
pub const SMEM_RESERVED_PER_CTA: u32 = 1024;
/// Placements above this count are refused (use `PTXV_L1_PLACE=pairs`).
pub const MAX_PLACEMENTS: usize = 4096;

/// Is grid `a` stream-ordered before grid `b` (every CTA of `b` starts after `a` completed)?
/// Follows the predecessor links from `b` (`Launch::hb_before`); a PDL edge does not order
/// (its successor may overlap), so a chain through one counts as unordered (more sharing:
/// never a miss). ASSUME(E-LAUNCH-ORDER): only with a recorded or declared launch order; an
/// older capture's list may serialize grids that overlapped (unordered: more sharing).
pub fn grids_ordered(l: &Launch, a: usize, b: usize) -> bool {
    l.order_recorded && l.hb_before(a, b)
}

/// May grids `a` and `b` be resident at the same time?
pub fn grids_overlap(l: &Launch, a: usize, b: usize) -> bool {
    a == b || (!grids_ordered(l, a, b) && !grids_ordered(l, b, a))
}

/// Can CTAs `a` and `b` (global ids) be resident on one SM at once? Same GPU, grids not
/// stream-ordered, and the SM's thread / CTA / shared-memory limits hold. Registers are not
/// known here and not checked (that only adds placements: never a miss).
pub fn can_share(l: &Launch, ctas: &[u32]) -> bool {
    let gs: Vec<usize> = ctas.iter().map(|&c| l.grid_index(c)).collect();
    let dev = l.grids[gs[0]].device;
    if gs.iter().any(|&g| l.grids[g].device != dev) {
        return false;
    }
    for i in 0..gs.len() {
        for j in i + 1..gs.len() {
            if !grids_overlap(l, gs[i], gs[j]) {
                return false;
            }
        }
    }
    let threads: u32 = gs.iter().map(|&g| l.grids[g].threads_per_cta()).sum();
    let smem: u32 = gs
        .iter()
        .map(|&g| l.grids[g].smem_bytes + SMEM_RESERVED_PER_CTA)
        .sum();
    ctas.len() as u32 <= MAX_CTAS_PER_SM && threads <= MAX_THREADS_PER_SM && smem <= MAX_SMEM_PER_SM
}

/// SM placements to explore (CTA -> SM id; CTAs not listed are alone on their SM).
///
/// `pairs` (the default): every CTA alone, plus each pair of CTAs that `can_share`. This
/// decides whether *some* placement reaches a stale read. Take a schedule under any placement
/// and its first stale read R, which hits a line that CTA F filled on R's SM (F may be R's own
/// CTA: the all-alone placement covers that). Run the same schedule with only F and R's CTA
/// on that SM and every other CTA alone. Up to R, a CTA's loads see a subset of the
/// invalidations, stores and fills they saw before on their SM, plus nothing new except
/// fills by F or R's CTA; so each earlier load returns what it did, or is itself stale
/// (an earlier first stale read), and if none is, F's fill survives to R as before. Either
/// way some stale read is reached. It does not list every stale read of every placement.
/// `all`: every partition of each GPU's CTAs into SMs within the limits (exponential; capped
/// at `MAX_PLACEMENTS`).
pub fn placements(l: &Launch, mode: &str) -> anyhow::Result<Vec<BTreeMap<u32, u32>>> {
    let n = l.ctas();
    let mut out = vec![BTreeMap::new()];
    match mode {
        "pairs" => {
            for a in 0..n {
                for b in a + 1..n {
                    if can_share(l, &[a, b]) {
                        let sm = (1 << 20) + a;
                        out.push([(a, sm), (b, sm)].into_iter().collect());
                    }
                }
            }
        }
        "all" => {
            // Set partitions, CTA by CTA: each joins an existing block it can share with, or
            // opens a new one.
            fn rec(
                l: &Launch,
                c: u32,
                n: u32,
                blocks: &mut Vec<Vec<u32>>,
                out: &mut Vec<BTreeMap<u32, u32>>,
            ) -> anyhow::Result<()> {
                if out.len() > MAX_PLACEMENTS {
                    anyhow::bail!(
                        "more than {MAX_PLACEMENTS} L1 placements; use PTXV_L1_PLACE=pairs"
                    );
                }
                if c == n {
                    let mut m = BTreeMap::new();
                    for b in blocks.iter().filter(|b| b.len() > 1) {
                        for &x in b {
                            m.insert(x, (1 << 20) + b[0]);
                        }
                    }
                    out.push(m);
                    return Ok(());
                }
                for i in 0..blocks.len() {
                    let mut t = blocks[i].clone();
                    t.push(c);
                    if can_share(l, &t) {
                        blocks[i].push(c);
                        rec(l, c + 1, n, blocks, out)?;
                        blocks[i].pop();
                    }
                }
                blocks.push(vec![c]);
                rec(l, c + 1, n, blocks, out)?;
                blocks.pop();
                Ok(())
            }
            out.clear();
            rec(l, 0, n, &mut Vec::new(), &mut out)?;
        }
        m => anyhow::bail!("PTXV_L1_PLACE: unknown mode {m} (pairs, all, none)"),
    }
    Ok(out)
}

/// ASSUME(E-L1-CORES): run `run` once per SM placement (`placements`, mode
/// `PTXV_L1_PLACE`, default `pairs`) and merge the verdicts: findings of every placement
/// (the message names the placement), complete only if every run was, outputs pooled.
/// Without co-residency (or with havoc, which covers every placement's fills) this is one
/// plain run. Inside each `%smid` placement (ASSUME(E-SMID), `smid::explore_smid`).
pub fn explore_placements(m0: Machine, run: &mut dyn FnMut(Machine) -> Verdict) -> Verdict {
    crate::smid::explore_smid(m0, &mut |m| explore_l1_placements(m, run))
}

fn explore_l1_placements(m0: Machine, run: &mut dyn FnMut(Machine) -> Verdict) -> Verdict {
    if !m0.launch.l1_cores || m0.launch.l1_havoc || m0.l1.is_none() {
        // ASSUME(E-L1-PLACE): without co-residency, the one placement of `L1::new`.
        return run(m0);
    }
    let mode = std::env::var("PTXV_L1_PLACE").unwrap_or_else(|_| "pairs".into());
    if mode == "none" {
        // One run with `L1::new`'s placement (every CTA alone, or PTXV_L1_COLOCATE).
        return run(m0);
    }
    let ps = match placements(&m0.launch, &mode) {
        Ok(p) => p,
        Err(e) => {
            let mut v = run(m0);
            v.complete = false;
            v.notes.push(format!("E-L1-CORES: {e}"));
            return v;
        }
    };
    let t0 = std::time::Instant::now();
    let mut merged: Option<Verdict> = None;
    let mut outs: std::collections::BTreeSet<Vec<u8>> = Default::default();
    for p in &ps {
        let mut m = m0.clone();
        // A fresh L1 with this placement; keeps the two-level check's settings (stats,
        // havoc_probe: twolevel.rs).
        let base = m0.l1.as_ref().unwrap();
        let mut l1 = L1 {
            stats: base.stats.clone(),
            havoc_probe: base.havoc_probe,
            ..L1::new()
        };
        l1.colocate = Arc::new(p.clone());
        m.l1 = Some(Arc::new(l1));
        let name = placement_name(&m0, p);
        let mut v = run(m);
        for f in &mut v.findings {
            f.msg = format!("[L1 placement: {name}] {}", f.msg);
        }
        outs.extend(v.outputs.iter().cloned());
        match &mut merged {
            None => merged = Some(v),
            Some(a) => {
                a.states += v.states;
                a.transitions += v.transitions;
                a.complete &= v.complete;
                a.terminal_states += v.terminal_states;
                for f in v.findings {
                    if !a.findings.iter().any(|x| x.sig == f.sig) {
                        a.findings.push(f);
                    }
                }
                for n in v.notes {
                    if !a.notes.contains(&n) {
                        a.notes.push(n);
                    }
                }
            }
        }
    }
    let mut v = merged.expect("at least one placement");
    v.outputs = outs.into_iter().collect();
    v.distinct_outputs = v.outputs.len();
    v.seconds = t0.elapsed().as_secs_f64();
    v.notes.push(format!(
        "E-L1-CORES: {} SM placement(s) explored (mode {mode}); a finding names its placement",
        ps.len()
    ));
    v
}

/// "alone" or "k1.cta0+k0.cta1; ...".
pub fn placement_name(m: &Machine, p: &BTreeMap<u32, u32>) -> String {
    let mut by: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (&c, &s) in p {
        by.entry(s).or_default().push(c);
    }
    if by.is_empty() {
        return "every CTA alone".into();
    }
    by.values()
        .map(|cs| {
            cs.iter()
                .map(|&c| m.cta_name(c as u16))
                .collect::<Vec<_>>()
                .join(" + ")
        })
        .collect::<Vec<_>>()
        .join("; ")
}

impl Machine {
    /// May a line filled by grid `fill` hit for a load of grid `reader` (on the same SM)?
    /// Without co-residency only within one grid (kernel boundary, E-HW-L1); with it, also
    /// between grids that may be resident at once (a CTA's launch dropped what was there
    /// before, `l1_cta_launch`).
    pub(crate) fn l1_shared(&self, fill: u32, reader: u32) -> bool {
        fill == reader
            || (self.launch.l1_cores && grids_overlap(&self.launch, fill as usize, reader as usize))
    }

    /// ASSUME(E-L1-CORES): a CTA's launch drops the lines that other grids left on its SM
    /// (the kernel boundary of probes/l1_kbound, taken as happening at launch on each SM).
    pub(crate) fn l1_cta_launch(&mut self, cta: u16) {
        if self.l1.is_none() || !self.launch.l1_cores {
            return;
        }
        let sm = self.l1_sm_of_cta(cta as u32);
        let gi = self.launch.grid_index(cta as u32) as u32;
        self.l1_fp_drop(sm);
        let l1 = Arc::make_mut(self.l1.as_mut().unwrap());
        l1.lines.retain(|k, l| k.0 != sm || l.fill.grid == gi);
    }

    fn l1_fp_on(&self) -> bool {
        self.launch.l1_cores && self.fp.as_ref().is_some_and(|f| !f.sync_only)
    }

    /// DPOR (E-L1-CORES): a weak load of `line` on `sm` reads the line and the SM's copy.
    pub(crate) fn l1_fp_load(&mut self, sm: u32, alloc: usize, line: u64) {
        if self.l1_fp_on() {
            let key = ((alloc as u64) << 40) | line;
            self.rec(SPACE_LINE, alloc as u32, line, line + LINE, false);
            self.rec(SPACE_SM, sm, key, key + 1, false);
        }
    }

    /// DPOR (E-L1-CORES): a store or atomic changes the lines it touches.
    pub(crate) fn l1_fp_store(&mut self, alloc: usize, off: u64, n: usize) {
        if self.l1_fp_on() {
            let lo = off / LINE * LINE;
            let hi = (off + n.max(1) as u64).div_ceil(LINE) * LINE;
            self.rec(SPACE_LINE, alloc as u32, lo, hi, true);
        }
    }

    /// DPOR (E-L1-CORES): an invalidation, atomic or CTA launch changes the SM's copies.
    pub(crate) fn l1_fp_drop(&mut self, sm: u32) {
        if self.l1_fp_on() {
            self.rec(SPACE_SM, sm, 0, u64::MAX, true);
        }
    }

    /// ASSUME(E-L1-HAVOC): could an outside fill make the weak load by `t` at `pc` of bytes
    /// from `lo` in allocation `alloc` (one line, on `sm`) stale? `ws` are the last writes of
    /// those bytes by other SMs. ASSUME(E-L1-CLOSED): without havoc, only the case's own
    /// loads fill lines, and nothing is reported here.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn l1_havoc_check(
        &self,
        t: usize,
        pc: u32,
        sm: u32,
        alloc: usize,
        lo: u64,
        ws: &[Epoch],
        me: &L1Ev,
    ) -> Option<(String, String, Vec<u32>)> {
        let l1 = self.l1.as_ref()?;
        if !self.launch.l1_havoc && !l1.havoc_probe {
            return None;
        }
        let empty = Vec::new();
        let invals: &[L1Ev] = l1.invals.get(&sm).map(|v| &v[..]).unwrap_or(&empty);
        for w in ws {
            let wt = self.actor_thread(w.actor);
            let wg = self.launch.grid_index(self.threads[wt].cta as u32);
            let rg = me.grid as usize;
            if wg != rg && grids_ordered(&self.launch, wg, rg) {
                continue; // the reader's launch dropped every copy older than W
            }
            // W happens-before the reader's launch (it completed a grid the reader's grid
            // waits for): a fill that survives into the reader's grid came after W.
            // ASSUME(E-LAUNCH-ORDER): only with a recorded or declared launch order.
            if self.launch.order_recorded
                && self.launch.full_preds(rg).any(|p| {
                    self.done_vcs[p]
                        .as_ref()
                        .is_some_and(|d| w.clock <= vc_get(d, w.actor))
                })
            {
                continue;
            }
            let killed = invals.iter().any(|i| {
                grids_overlap(&self.launch, i.grid as usize, me.grid as usize)
                    && w.hb(&i.vc)
                    && i.clock <= vc_get(&me.vc, i.actor)
            });
            if killed {
                continue;
            }
            let msg = format!(
                "{} at {} may read stale bytes (allocation {}, offset {lo:#x}, line {:#x}): the last write, by {} at {}, comes from another SM, and no L1 invalidation on the reader's SM is ordered after that write and before the read; an agent on the reader's SM that loads any byte of this line after the reader's last invalidation and before the write lands leaves a copy without the written bytes, which this load hits (E-L1-HAVOC; the write {} the read)",
                self.actor_name(t as u32),
                self.k.describe(pc as usize),
                self.launch.allocs[alloc].name,
                lo / LINE * LINE,
                self.actor_name(w.actor),
                self.k.describe(w.pc as usize),
                if w.hb(&me.vc) {
                    "happens-before"
                } else {
                    "is not ordered before"
                },
            );
            return Some((format!("stale-l1-havoc:{pc}"), msg, vec![pc, w.pc]));
        }
        None
    }
}

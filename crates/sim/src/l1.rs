/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Opt-in hardware-faithful L1 model (E-HW-L1, case `hw_l1`, `PTXV_HW_L1=0|1`).
//!
//! GB300's per-SM L1 is not coherent (docs/poll-acq-litmus.md, docs/deepep-repro.md): a weak
//! global load (`LD.E`, `LDG.E`, `LDG.E.CONSTANT`) that hits a line filled earlier on the same
//! SM returns the line's bytes even if another SM or GPU has since written memory. Lines are
//! dropped by `CCTL.IVALL` (the acquire half of `ld.acquire` / `fence.acq_rel`, PTX
//! `fence.acq_rel.gpu|sys` and acquire loads at gpu / sys scope) and at kernel boundaries
//! (probes/l1_kbound). Strong loads (`.STRONG.*`, volatile) neither hit nor fill; the SM's own
//! stores keep its copy current; its atomics drop the line.
//!
//! Two checks, both reported as kind "stale":
//! - *operational* (`stale-l1:<pc>`): in the explored schedule, a weak load hit a line whose
//!   bytes differ from memory. The load returns the line's bytes (as the hardware does).
//! - *predictive* (`stale-l1-pred:<R>:<F>`): for a weak load R of bytes whose last write W (in
//!   the explored schedule) is by another SM, some weak load F of the same line on R's SM could have
//!   filled the line before W: W does not happen-before F, and no L1 invalidation I on that SM
//!   is forced between them (F -> I -> R or W -> I -> R in happens-before). Like a
//!   happens-before race check, this holds for every schedule with the same happens-before
//!   order, so one run covers them; it is approximate (pairwise reasoning, capped logs, see
//!   ASSUMPTIONS.md E-HW-L1).
//!
//! An SM is a CTA by default (each CTA on its own SM). `PTXV_L1_COLOCATE="0,1;2,3"` puts the
//! listed global CTA ids on one SM; lines filled by another grid never survive (kernel
//! boundary, probed for consecutive kernels on one stream). Co-resident launches, explored
//! placements and outside fills (E-L1-CORES, E-L1-HAVOC) are in l1_cores.rs.
//!
//! Representation (for exploration speed; the semantics are those above). The explorer clones
//! a machine per successor, and every weak load changes the L1 state (a fill or the load log).
//! All per-line state is therefore kept per SM behind its own `Arc` (`SmMap`), and each line's
//! load log behind another: cloning the L1 copies one pointer per SM, and the first change in
//! a successor copies only the touched SM's map of pointers and the touched log. Each SM's
//! lines carry a cached hash, so hashing a state rehashes only the SMs that changed. Updates
//! that change nothing (a store or atomic to a line not in L1, an invalidation of an empty
//! L1's lines) no longer copy anything.

use std::collections::BTreeMap;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;

use crate::exec::Loc;
use crate::machine::*;

/// L1 line size (bytes).
pub const LINE: u64 = 128;
/// Per (SM, line, actor): the first and last loads kept for the predictive check.
const KEEP_LOADS: usize = 2;
/// Per (SM, actor): the last invalidations kept.
const KEEP_INVALS: usize = 4;

/// A weak load or an invalidation, with its thread's clock.
#[derive(Clone, Debug)]
pub struct L1Ev {
    pub actor: u32,
    pub clock: u32,
    pub pc: u32,
    pub grid: u32,
    pub vc: VC,
}

impl L1Ev {
    /// Does this event happen-before a thread whose clock is `v`?
    fn hb(&self, v: &VC) -> bool {
        self.clock <= vc_get(v, self.actor)
    }
}

#[derive(Clone, Debug)]
pub struct L1Line {
    pub bytes: Vec<u8>,
    pub fill: L1Ev,
}

/// Key of an L1 line: (SM, allocation, line address).
pub type LineKey = (u32, u32, u64);

/// One SM's entries of an `SmMap`, with a lazily computed hash of its keys and the bytes
/// `hash_val` extracts (reset by every change).
#[derive(Clone, Debug)]
pub struct SmNode<V> {
    map: BTreeMap<(u32, u64), V>,
    hash: OnceLock<u64>,
}

impl<V> Default for SmNode<V> {
    fn default() -> Self {
        SmNode {
            map: BTreeMap::new(),
            hash: OnceLock::new(),
        }
    }
}

/// A map (SM, allocation, line) -> V shared copy-on-write per SM. Offers the subset of
/// `BTreeMap<LineKey, V>`'s interface the model uses, with the same key type and ordering.
#[derive(Clone, Debug)]
pub struct SmMap<V: Clone> {
    sms: BTreeMap<u32, Arc<SmNode<V>>>,
}

impl<V: Clone> Default for SmMap<V> {
    fn default() -> Self {
        SmMap {
            sms: BTreeMap::new(),
        }
    }
}

impl<V: Clone> SmMap<V> {
    pub fn get(&self, k: &LineKey) -> Option<&V> {
        self.sms.get(&k.0)?.map.get(&(k.1, k.2))
    }
    pub fn contains_key(&self, k: &LineKey) -> bool {
        self.get(k).is_some()
    }
    fn node_mut(&mut self, sm: u32) -> &mut SmNode<V> {
        let n = Arc::make_mut(self.sms.entry(sm).or_default());
        n.hash.take();
        n
    }
    pub fn get_mut(&mut self, k: &LineKey) -> Option<&mut V> {
        if !self.contains_key(k) {
            return None;
        }
        self.node_mut(k.0).map.get_mut(&(k.1, k.2))
    }
    pub fn insert(&mut self, k: LineKey, v: V) -> Option<V> {
        self.node_mut(k.0).map.insert((k.1, k.2), v)
    }
    pub fn remove(&mut self, k: &LineKey) -> Option<V> {
        if !self.contains_key(k) {
            return None;
        }
        let n = self.node_mut(k.0);
        let r = n.map.remove(&(k.1, k.2));
        if n.map.is_empty() {
            self.sms.remove(&k.0);
        }
        r
    }
    /// `BTreeMap::entry(k).or_default()`.
    pub fn get_or_default(&mut self, k: LineKey) -> &mut V
    where
        V: Default,
    {
        self.node_mut(k.0).map.entry((k.1, k.2)).or_default()
    }
    /// Drop every entry of SM `sm` (cheap: no copy).
    pub fn clear_sm(&mut self, sm: u32) {
        self.sms.remove(&sm);
    }
    /// Does SM `sm` have any entry?
    pub fn has_sm(&self, sm: u32) -> bool {
        self.sms.contains_key(&sm)
    }
    pub fn retain(&mut self, mut f: impl FnMut(&LineKey, &mut V) -> bool) {
        let sms: Vec<u32> = self.sms.keys().copied().collect();
        for sm in sms {
            // Copy an SM's entries only if something of it goes.
            let keep_all = {
                let n = &self.sms[&sm];
                n.map
                    .iter()
                    .all(|(&(a, l), v)| f(&(sm, a, l), &mut v.clone()))
            };
            if keep_all {
                continue;
            }
            let n = self.node_mut(sm);
            n.map.retain(|&(a, l), v| f(&(sm, a, l), v));
            if n.map.is_empty() {
                self.sms.remove(&sm);
            }
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = (LineKey, &V)> {
        self.sms
            .iter()
            .flat_map(|(&sm, n)| n.map.iter().map(move |(&(a, l), v)| ((sm, a, l), v)))
    }
    pub fn keys(&self) -> impl Iterator<Item = LineKey> + '_ {
        self.iter().map(|(k, _)| k)
    }
    pub fn len(&self) -> usize {
        self.sms.values().map(|n| n.map.len()).sum()
    }
    pub fn is_empty(&self) -> bool {
        self.sms.is_empty() || self.sms.values().all(|n| n.map.is_empty())
    }
}

impl<V: Clone> SmMap<V> {
    /// Hash: per SM, its keys and what `f` hashes of each value (cached per SM).
    fn hash_by<H: Hasher>(
        &self,
        h: &mut H,
        f: impl Fn(&V, &mut std::collections::hash_map::DefaultHasher),
    ) {
        for (sm, n) in &self.sms {
            if n.map.is_empty() {
                continue;
            }
            let v = *n.hash.get_or_init(|| {
                let mut s = std::collections::hash_map::DefaultHasher::new();
                for (k, l) in &n.map {
                    k.hash(&mut s);
                    f(l, &mut s);
                }
                s.finish()
            });
            sm.hash(h);
            v.hash(h);
        }
    }
}

/// Per-instruction counts of the L1's weak loads over a whole exploration, shared by every
/// state's L1 (two-level check, twolevel.rs: why a race is not observable on GB300).
#[derive(Debug, Default)]
pub struct L1Stats {
    shards: [Mutex<BTreeMap<u32, PcStat>>; 16],
}

#[derive(Clone, Debug, Default)]
pub struct PcStat {
    /// Weak loads at this pc (per thread and line).
    pub lookups: u64,
    /// L1 hits among them.
    pub hits: u64,
    /// Hits whose bytes differed from memory (operational stale reads).
    pub stale: u64,
    /// The first line looked up: (SM, allocation, line address).
    pub line: Option<LineKey>,
    /// The last invalidation on that SM before that lookup (its pc), if any in this grid.
    pub last_inval: Option<u32>,
    /// The pc of the load that filled the line, for the first hit.
    pub hit_fill: Option<u32>,
}

impl L1Stats {
    pub fn get(&self, pc: u32) -> Option<PcStat> {
        self.shards[pc as usize % 16]
            .lock()
            .unwrap()
            .get(&pc)
            .cloned()
    }
    fn note(&self, pc: u32, f: impl FnOnce(&mut PcStat)) {
        f(self.shards[pc as usize % 16]
            .lock()
            .unwrap()
            .entry(pc)
            .or_default());
    }
}

#[derive(Clone, Debug, Default)]
pub struct L1 {
    /// Operational state: (SM, allocation, line address) -> the line as filled.
    pub lines: SmMap<L1Line>,
    /// Predictive check: weak loads per (SM, allocation, line), in load order.
    pub loads: SmMap<Arc<Vec<L1Ev>>>,
    /// Predictive check: invalidations per SM.
    pub invals: BTreeMap<u32, Arc<Vec<L1Ev>>>,
    /// CTA -> SM (PTXV_L1_COLOCATE); absent: the CTA is its own SM.
    pub colocate: Arc<BTreeMap<u32, u32>>,
    /// Per-pc counts (two-level check only; shared by all clones, not hashed).
    pub stats: Option<Arc<L1Stats>>,
    /// Hash the predictive check's logs too (their (thread, pc) entries), so that states
    /// differing only in which loads could have filled a line stale are not merged. Set by
    /// the two-level check's hardware model (twolevel.rs), which must not miss a predictive
    /// stale read through state deduplication; off in a case's own `hw_l1` runs.
    pub hash_logs: bool,
    /// Run the havoc check (E-L1-HAVOC, `l1_havoc_check`) and report its `stale-l1-havoc`
    /// witnesses without opting the launch into havoc (placements stay explored). Set only by
    /// the two-level check's hardware model (twolevel.rs), to tell findings latent in any world
    /// from findings latent only under E-L1-CLOSED. Not hashed: it is the same for all states.
    pub havoc_probe: bool,
}

impl Hash for L1 {
    fn hash<H: Hasher>(&self, h: &mut H) {
        // Only the operational state decides behavior; the logs feed findings only.
        self.lines.hash_by(h, |l, s| l.bytes.hash(s));
        if self.hash_logs {
            let evs = |v: &[L1Ev], s: &mut std::collections::hash_map::DefaultHasher| {
                for e in v {
                    (e.actor, e.pc, e.grid).hash(s);
                }
            };
            self.loads.hash_by(h, |v, s| evs(v, s));
            for (sm, v) in &self.invals {
                sm.hash(h);
                let mut s = std::collections::hash_map::DefaultHasher::new();
                evs(v, &mut s);
                s.finish().hash(h);
            }
        }
    }
}

impl L1 {
    pub fn new() -> L1 {
        let mut colocate = BTreeMap::new();
        if let Ok(s) = std::env::var("PTXV_L1_COLOCATE") {
            for (gi, grp) in s.split(';').enumerate() {
                for c in grp.split(',').filter(|c| !c.trim().is_empty()) {
                    if let Ok(c) = c.trim().parse::<u32>() {
                        colocate.insert(c, (1 << 20) + gi as u32);
                    }
                }
            }
        }
        L1 {
            colocate: Arc::new(colocate),
            ..Default::default()
        }
    }
}

impl Machine {
    pub(crate) fn l1_sm_of_cta(&self, cta: u32) -> u32 {
        let l1 = self.l1.as_ref().unwrap();
        *l1.colocate.get(&cta).unwrap_or(&cta)
    }

    pub(crate) fn l1_ev(&self, t: usize, pc: u32) -> L1Ev {
        let vc = self.threads[t].vc.clone();
        let cta = self.threads[t].cta as u32;
        L1Ev {
            actor: t as u32,
            clock: vc_get(&vc, t as u32),
            pc,
            grid: self.launch.grid_index(cta) as u32,
            vc,
        }
    }

    /// ASSUME(E-HW-L1): a weak global load of `data.len()` bytes at `loc` by thread `t`.
    /// `data` holds memory's bytes; on an L1 hit it is replaced by the line's bytes.
    pub(crate) fn l1_load(&mut self, t: usize, pc: u32, loc: Loc, data: &mut [u8]) {
        let Loc::Global { alloc, off } = loc else {
            return;
        };
        let n = data.len() as u64;
        let sm = self.l1_sm_of_cta(self.threads[t].cta as u32);
        let me = self.l1_ev(t, pc);
        let mut findings: Vec<(String, String, Vec<u32>)> = Vec::new();
        let first = off / LINE;
        let last = (off + n.max(1) - 1) / LINE;
        for line in first..=last {
            let key = (sm, alloc as u32, line * LINE);
            let lo = off.max(line * LINE);
            let hi = (off + n).min(line * LINE + LINE);
            // Predictive check: the last cross-SM writes W of the bytes read. W need not
            // happen-before R: the strict model's relaxed polls give no happens-before, yet on
            // GB300 the poll exit does order later L1 misses (docs/poll-acq-litmus.md).
            let mut ws: Vec<Epoch> = Vec::new();
            for a in lo..hi {
                if let Some(w) = self.gmem[alloc].sh.get(a as usize).w {
                    let wt = self.actor_thread(w.actor);
                    let wsm = self.l1_sm_of_cta(self.threads[wt].cta as u32);
                    if wsm != sm && !ws.iter().any(|x| x.actor == w.actor && x.clock == w.clock) {
                        ws.push(w);
                    }
                }
            }
            if !ws.is_empty() {
                let l1 = self.l1.as_ref().unwrap();
                let empty = Vec::new();
                let invals: &[L1Ev] = l1.invals.get(&sm).map(|v| &v[..]).unwrap_or(&empty);
                'w: for w in &ws {
                    for f in l1.loads.get(&key).into_iter().flat_map(|v| v.iter()) {
                        if !self.l1_shared(f.grid, me.grid) {
                            continue; // kernel boundary (l1_cores.rs: or co-resident grids)
                        }
                        if w.hb(&f.vc) {
                            continue; // F is ordered after W: it fills fresh bytes
                        }
                        let killed = invals.iter().any(|i| {
                            self.l1_shared(i.grid, me.grid)
                                && (f.hb(&i.vc) || w.hb(&i.vc))
                                && i.hb(&me.vc)
                        });
                        if killed {
                            continue;
                        }
                        let msg = format!(
                            "{} at {} may read stale bytes (allocation {}, offset {lo:#x}): the weak load by {} at {} on the same SM can fill the L1 line before the write by {} at {} lands, and no L1 invalidation on that SM is ordered between them (E-HW-L1, predictive; the write {} the read)",
                            self.actor_name(t as u32),
                            self.k.describe(pc as usize),
                            self.launch.allocs[alloc].name,
                            self.actor_name(f.actor),
                            self.k.describe(f.pc as usize),
                            self.actor_name(w.actor),
                            self.k.describe(w.pc as usize),
                            if w.hb(&me.vc) {
                                "happens-before"
                            } else {
                                "is not ordered before"
                            },
                        );
                        findings.push((
                            format!("stale-l1-pred:{pc}:{}", f.pc),
                            msg,
                            vec![pc, f.pc, w.pc],
                        ));
                        break 'w;
                    }
                }
            }
            // E-L1-HAVOC / E-L1-CORES (l1_cores.rs): outside fills; DPOR line footprints.
            if let Some(x) = self.l1_havoc_check(t, pc, sm, alloc, lo, &ws, &me) {
                findings.push(x);
            }
            self.l1_fp_load(sm, alloc, line * LINE);
            // Operational state.
            let ldat = &mut data[(lo - off) as usize..(hi - off) as usize];
            let l1 = self.l1.as_ref().unwrap();
            let hit = l1
                .lines
                .get(&key)
                .filter(|l| self.l1_shared(l.fill.grid, me.grid))
                .map(|l| {
                    let o = (lo - line * LINE) as usize;
                    (l.bytes[o..o + ldat.len()].to_vec(), l.fill.clone())
                });
            if let Some(st) = l1.stats.clone() {
                let last_inval = l1
                    .invals
                    .get(&sm)
                    .and_then(|v| v.iter().rev().find(|i| self.l1_shared(i.grid, me.grid)))
                    .map(|i| i.pc);
                let (is_hit, stale, fpc) = match &hit {
                    Some((b, f)) => (true, b[..] != ldat[..], Some(f.pc)),
                    None => (false, false, None),
                };
                st.note(pc, |s| {
                    s.lookups += 1;
                    s.hits += is_hit as u64;
                    s.stale += stale as u64;
                    if s.line.is_none() {
                        s.line = Some(key);
                        s.last_inval = last_inval;
                    }
                    if is_hit && s.hit_fill.is_none() {
                        s.hit_fill = fpc;
                    }
                });
            }
            match hit {
                Some((bytes, fill)) => {
                    if bytes != ldat {
                        let msg = format!(
                            "{} at {} read stale bytes from L1 (allocation {}, offset {lo:#x}: {:02x?} in L1, {:02x?} in memory); the line was filled by {} at {} (E-HW-L1)",
                            self.actor_name(t as u32),
                            self.k.describe(pc as usize),
                            self.launch.allocs[alloc].name,
                            bytes,
                            ldat,
                            self.actor_name(fill.actor),
                            self.k.describe(fill.pc as usize),
                        );
                        findings.push((format!("stale-l1:{pc}"), msg, vec![pc, fill.pc]));
                        ldat.copy_from_slice(&bytes);
                    }
                }
                None => {
                    let mlen = self.gmem[alloc].bytes.len() as u64;
                    let base = line * LINE;
                    let bytes: Vec<u8> = (base..(base + LINE).min(mlen))
                        .map(|a| *self.gmem[alloc].bytes.get(a as usize))
                        .chain(std::iter::repeat(0))
                        .take(LINE as usize)
                        .collect();
                    let l1 = Arc::make_mut(self.l1.as_mut().unwrap());
                    l1.lines.insert(
                        key,
                        L1Line {
                            bytes,
                            fill: me.clone(),
                        },
                    );
                }
            }
            // Log the load (first and last KEEP_LOADS per actor).
            let l1 = Arc::make_mut(self.l1.as_mut().unwrap());
            let v = Arc::make_mut(l1.loads.get_or_default(key));
            let mut mine = 0;
            let mut third = None;
            for (i, e) in v.iter().enumerate() {
                if e.actor == me.actor {
                    if mine == KEEP_LOADS {
                        third = Some(i);
                    }
                    mine += 1;
                }
            }
            if mine >= 2 * KEEP_LOADS {
                v.remove(third.unwrap());
            }
            v.push(me.clone());
        }
        for (sig, msg, pcs) in findings {
            // The two-level check's havoc probe (`L1::havoc_probe`, the launch not in havoc
            // mode) reports kind "probe": a witness for classification, not a finding of the
            // check (explorers and launch groups do not act on it).
            let kind = if sig.starts_with("stale-l1-havoc:") && !self.launch.l1_havoc {
                "probe"
            } else {
                "stale"
            };
            self.report_pcs(kind, sig, msg, pcs);
        }
    }

    /// ASSUME(E-HW-L1): a store by thread `t` keeps its own SM's copy current.
    pub(crate) fn l1_store(&mut self, t: usize, loc: Loc, data: &[u8]) {
        let Loc::Global { alloc, off } = loc else {
            return;
        };
        let sm = self.l1_sm_of_cta(self.threads[t].cta as u32);
        self.l1_fp_store(alloc, off, data.len());
        if data.is_empty() || !self.l1.as_ref().unwrap().lines.has_sm(sm) {
            return;
        }
        let first = off / LINE;
        let last = (off + data.len() as u64 - 1) / LINE;
        for line in first..=last {
            let key = (sm, alloc as u32, line * LINE);
            if !self.l1.as_ref().unwrap().lines.contains_key(&key) {
                continue;
            }
            let lo = off.max(line * LINE);
            let hi = (off + data.len() as u64).min(line * LINE + LINE);
            let src = &data[(lo - off) as usize..(hi - off) as usize];
            let o = (lo - line * LINE) as usize;
            let l1 = self.l1.as_ref().unwrap();
            if l1.lines.get(&key).unwrap().bytes[o..o + src.len()] == *src {
                continue; // already current
            }
            let l1 = Arc::make_mut(self.l1.as_mut().unwrap());
            let l = l1.lines.get_mut(&key).unwrap();
            l.bytes[o..o + src.len()].copy_from_slice(src);
        }
    }

    /// ASSUME(E-HW-L1): an atomic by thread `t` drops its own SM's copy of the line.
    pub(crate) fn l1_atom(&mut self, t: usize, loc: Loc, n: usize) {
        let Loc::Global { alloc, off } = loc else {
            return;
        };
        let sm = self.l1_sm_of_cta(self.threads[t].cta as u32);
        self.l1_fp_store(alloc, off, n);
        self.l1_fp_drop(sm);
        for line in off / LINE..=(off + n.max(1) as u64 - 1) / LINE {
            let key = (sm, alloc as u32, line * LINE);
            if self.l1.as_ref().unwrap().lines.contains_key(&key) {
                Arc::make_mut(self.l1.as_mut().unwrap()).lines.remove(&key);
            }
        }
    }

    /// ASSUME(E-HW-L1): `CCTL.IVALL` (or a PTX acquire at gpu / sys scope) by thread `t`
    /// drops every line of its SM. Call after the acquire took effect (the clock it records
    /// is what the invalidation is ordered after).
    pub(crate) fn l1_inval(&mut self, t: usize, pc: u32) {
        let sm = self.l1_sm_of_cta(self.threads[t].cta as u32);
        let me = self.l1_ev(t, pc);
        self.l1_fp_drop(sm);
        let l1 = Arc::make_mut(self.l1.as_mut().unwrap());
        l1.lines.clear_sm(sm);
        let v = Arc::make_mut(l1.invals.entry(sm).or_default());
        let mut n = 0;
        let mut oldest = None;
        for (i, e) in v.iter().enumerate() {
            if e.actor == me.actor {
                oldest.get_or_insert(i);
                n += 1;
            }
        }
        if n >= KEEP_INVALS {
            v.remove(oldest.unwrap());
        }
        v.push(me);
    }
}

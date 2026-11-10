/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The TPU machine as an explorer `Model` (ptxv_sim::model): actions are "core c runs up to
//! and including its next visible bundle" (sync-flag op, DMA issue, halt), "DMA i completes"
//! and "remote signal i arrives". Local bundles between visible ones commute with every other
//! action unless they touch memory an unordered DMA also touches, which the vector-clock race
//! detector reports, so interleaving only at visible bundles loses no outcome of a race-free
//! kernel.
//!
//! Findings: `deadlock` (no action enabled, not finished), `race` (a core access and a DMA
//! access, or two DMAs, to the same 512 B VMEM row / SMEM word / HBM row with no
//! happens-before through sync flags), `violation` exit-check (the kernel's "semaphore nonzero
//! at exit" halt), bounds-check (`BoundsCheck` halts), delay-window-sync (a sync-flag op inside
//! a branch delay window), `unmodeled` (an op or form the machine refuses).

use std::fmt;

use ptxv_sim::machine::Finding;
use ptxv_sim::model::Model;

use crate::machine::SimErr;
use crate::machine::TpuMachine;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    Core(usize),
    /// Send-done of a copy with a send flag: source read, send flag up.
    DmaSend(usize),
    Dma(usize),
    Remote(usize),
}

impl fmt::Display for Act {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Act::Core(c) => write!(f, "core {c}"),
            Act::DmaSend(i) => write!(f, "dma {i} send-done"),
            Act::Dma(i) => write!(f, "dma {i}"),
            Act::Remote(i) => write!(f, "remote {i}"),
        }
    }
}

/// Assumptions every TPU verdict rests on (fb/targets/tpu/ASSUMPTIONS.md).
pub const BASE_ASSUMPTIONS: &[&str] = &[
    "TPU-E-VLIW-READ-OLD",
    "TPU-E-BR-DELAY",
    "TPU-E-BR-SYNC",
    "TPU-E-WAIT-BUNDLE",
    "TPU-S-DMA-REDUCE",
    "TPU-S-DMA-GRANULE",
    "TPU-S-DMA-DONE",
    "TPU-S-DMA-NO-ORDER",
    "TPU-S-DMA-SRC-LIVE",
    "TPU-S-RDMA-RECV",
    "TPU-S-REM-SIGNAL",
    "TPU-S-REM-OVF",
    "TPU-S-SEM-INT32",
    "TPU-S-SEM-OVF-TRAP",
    "TPU-S-WAIT-GE",
    "TPU-S-SEM-PERSIST",
    "TPU-S-SEM-INDEX",
    "TPU-S-BARRIER",
    "TPU-L-ADDR-UNITS",
    "TPU-L-REMOTE-FLAG-ADDR",
    "TPU-L-DMA-GENERAL-DEV",
    "TPU-L-DEVICE-IDS",
];

#[derive(Clone)]
pub struct TpuModel {
    pub m: TpuMachine,
    /// Bundles one core may run in one action before the run is cut (non-terminating loop).
    pub run_limit: u64,
    err: Option<SimErr>,
    /// Flags whose value a core reads (non-exit-check `vsyncmov`, `vwait.eq`); `None`: the
    /// reduction is off (some such read has a run-time flag operand, or disabled).
    observed: Option<std::sync::Arc<std::collections::BTreeSet<u32>>>,
}

impl TpuModel {
    /// Explorer mode on `m`: remote signals in flight, vector clocks and the race shadow on.
    pub fn new(mut m: TpuMachine) -> Self {
        m.explore = true;
        m.race = Some(Box::default());
        let observed = observed_flags(&m).map(std::sync::Arc::new);
        TpuModel {
            m,
            run_limit: 1_000_000,
            err: None,
            observed,
        }
    }

    /// Explore every completion order (no ample reduction).
    pub fn without_reduction(mut self) -> Self {
        self.observed = None;
        self
    }

    pub fn reduced(&self) -> bool {
        self.observed.is_some()
    }

    /// No core can step now and none ever will: every blocked core waits for more than its flag
    /// can reach from the work in flight (`ge` waits; an `eq` wait counts as satisfiable).
    fn unsatisfiable(&self) -> Option<String> {
        let m = &self.m;
        if (0..m.cores.len()).any(|c| m.can_step(c)) {
            return None;
        }
        let mut why = Vec::new();
        for c in m.cores.iter().filter(|c| !c.done && c.halted.is_none()) {
            let (f, need, eq) = m.blocked_wait(c.id)?;
            let pot = m.flag_potential(c.id, f);
            if eq || pot >= need {
                return None;
            }
            why.push(format!(
                "core {} waits for flag {f} >= {need}, at most {pot} can arrive",
                c.id
            ));
        }
        (!why.is_empty()).then(|| why.join("; "))
    }

    fn halted(&self) -> bool {
        self.err.is_some() || self.m.cores.iter().any(|c| c.halted.is_some())
    }

    fn run_core(&mut self, c: usize, f: &mut Vec<Finding>) {
        let mut n = 0u64;
        loop {
            if !self.m.can_step(c) {
                return;
            }
            let vis = self.m.next_visible(c);
            match self.m.step(c, f) {
                Ok(true) => {}
                Ok(false) => return,
                Err(e) => {
                    self.err = Some(e);
                    return;
                }
            }
            let core = &self.m.cores[c];
            if vis || core.done || core.halted.is_some() {
                return;
            }
            n += 1;
            if n > self.run_limit {
                self.err = Some(SimErr::Halt(format!(
                    "core {c}: {} bundles without a visible op",
                    self.run_limit
                )));
                return;
            }
        }
    }
}

impl Model for TpuModel {
    type Action = Act;

    fn enabled(&self) -> Vec<Act> {
        if self.halted() {
            return Vec::new();
        }
        if self.unsatisfiable().is_some() {
            return Vec::new();
        }
        // Ample set: a DMA step or a positive remote signal whose flag no core reads by value
        // commutes with every other action up to race findings (reported in either order) and
        // with `>=` waits (adds only raise the flag, so waits pass no later): take it alone.
        // ASSUME(TPU-S-AMPLE)
        if let Some(obs) = &self.observed {
            for (i, d) in self.m.pending.iter().enumerate() {
                let send = self.m.dma_needs_send(i);
                let f = if send {
                    d.src_flag.unwrap_or(d.flag)
                } else {
                    d.flag
                };
                if !obs.contains(&f) {
                    return vec![if send { Act::DmaSend(i) } else { Act::Dma(i) }];
                }
            }
            for (i, r) in self.m.remote.iter().enumerate() {
                if r.n > 0 && !obs.contains(&r.flag) {
                    return vec![Act::Remote(i)];
                }
            }
        }
        let mut v: Vec<Act> = (0..self.m.cores.len())
            .filter(|&c| self.m.can_step(c))
            .map(Act::Core)
            .collect();
        for i in 0..self.m.pending.len() {
            v.push(if self.m.dma_needs_send(i) {
                Act::DmaSend(i)
            } else {
                Act::Dma(i)
            });
        }
        v.extend((0..self.m.remote.len()).map(Act::Remote));
        v
    }

    fn step(&mut self, a: &Act) -> Vec<Finding> {
        let mut f = Vec::new();
        match *a {
            Act::Core(c) => self.run_core(c, &mut f),
            Act::Dma(i) => {
                if let Err(e) = self.m.complete_dma(i, &mut f) {
                    self.err = Some(e);
                }
            }
            Act::DmaSend(i) => {
                if let Err(e) = self.m.dma_send(i, &mut f) {
                    self.err = Some(e);
                }
            }
            Act::Remote(i) => {
                if let Err(e) = self.m.deliver_remote(i) {
                    self.err = Some(e);
                }
            }
        }
        self.m.prune_race();
        f
    }

    fn fingerprint(&self) -> u128 {
        self.m.fingerprint() ^ (self.err.is_some() as u128)
    }

    fn is_terminal(&self) -> bool {
        !self.halted()
            && self.m.cores.iter().all(|c| c.done)
            && self.m.pending.is_empty()
            && self.m.remote.is_empty()
    }

    fn stuck(&self) -> Option<Finding> {
        let finding = |kind: &str, sig: String, msg: String| Finding {
            kind: kind.into(),
            sig,
            msg,
            trace: Vec::new(),
            pcs: Vec::new(),
            gb300: None,
            perm_rows: vec![],
        };
        if let Some(e) = &self.err {
            let kind = if matches!(e, SimErr::Unimpl(_)) {
                "unmodeled"
            } else {
                "violation"
            };
            let text = e.to_string();
            let sig = match e {
                // "sem-overflow core 0@0x11: ..." -> "sem-overflow:core 0@0x11"
                SimErr::Halt(m) if m.starts_with("sem-overflow") => format!(
                    "sem-overflow:{}",
                    m["sem-overflow ".len()..].split(':').next().unwrap_or("")
                ),
                _ => format!("{kind}:{}", text.chars().take(80).collect::<String>()),
            };
            return Some(finding(kind, sig, text));
        }
        if self.halted() {
            return None;
        }
        if let Some(why) = self.unsatisfiable() {
            let pcs: Vec<String> = self
                .m
                .cores
                .iter()
                .filter(|c| !c.done)
                .map(|c| {
                    format!(
                        "{}@{:#x}",
                        c.id,
                        self.m.kernel.bundles.get(c.pc).map(|b| b.addr).unwrap_or(0)
                    )
                })
                .collect();
            return Some(finding(
                "deadlock",
                format!("deadlock:{}", pcs.join(",")),
                format!("wait for more than can arrive: {why}"),
            ));
        }
        if self.is_terminal() {
            // reported even when the kernel's own exit check is compiled out
            // (disable_semaphore_checks). ASSUME(TPU-S-SEM-PERSIST)
            let u = self.m.unbalanced_flags();
            if u.is_empty() {
                return None;
            }
            let list: Vec<String> = u
                .iter()
                .map(|(c, f, v, k)| format!("core {c} flag {f} ({k:?}) = {v}"))
                .collect();
            let sig: Vec<String> = u.iter().map(|(c, f, _, _)| format!("{c}.{f}")).collect();
            return Some(finding(
                "violation",
                format!("sem-unbalanced:{}", sig.join(",")),
                format!(
                    "sync flags nonzero at exit (they persist into the next launch): {}",
                    list.join("; ")
                ),
            ));
        }
        let k = &self.m.kernel;
        let who: Vec<String> = self
            .m
            .cores
            .iter()
            .filter(|c| !c.done)
            .map(|c| format!("core {} at {}", c.id, k.describe(c.pc)))
            .collect();
        let pcs: Vec<String> = self
            .m
            .cores
            .iter()
            .filter(|c| !c.done)
            .map(|c| {
                format!(
                    "{}@{:#x}",
                    c.id,
                    k.bundles.get(c.pc).map(|b| b.addr).unwrap_or(0)
                )
            })
            .collect();
        Some(finding(
            "deadlock",
            format!("deadlock:{}", pcs.join(",")),
            format!("no core can proceed: {}", who.join("; ")),
        ))
    }

    fn outputs(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for p in self.m.placed.iter().filter(|p| p.decl.output) {
            if let Ok(w) = self.m.operand_words(p) {
                for x in w {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        out
    }

    fn describe(&self, a: &Act) -> String {
        match *a {
            Act::Core(c) => {
                let core = &self.m.cores[c];
                format!(
                    "core {c} from {}",
                    self.m
                        .kernel
                        .describe(core.pc)
                        .chars()
                        .take(120)
                        .collect::<String>()
                )
            }
            Act::DmaSend(i) | Act::Dma(i) => {
                let d = &self.m.pending[i];
                format!(
                    "dma from core {} @{:#x} {} ({:?}->{:?}, {} B, flag {} on core {})",
                    d.core,
                    self.m.kernel.bundles.get(d.pc).map(|b| b.addr).unwrap_or(0),
                    if matches!(a, Act::DmaSend(_)) {
                        "send-done"
                    } else {
                        "completes"
                    },
                    d.kind.src,
                    d.kind.dst,
                    d.nbytes,
                    d.flag,
                    d.dst_core
                )
            }
            Act::Remote(i) => {
                let r = &self.m.remote[i];
                format!(
                    "remote signal core {} -> core {} flag {} += {} arrives",
                    r.from, r.to, r.flag, r.n
                )
            }
        }
    }

    fn assumptions(&self) -> Vec<String> {
        let mut v: Vec<String> = BASE_ASSUMPTIONS.iter().map(|s| s.to_string()).collect();
        v.extend(self.m.kernel.assumptions.iter().cloned());
        v
    }
}

/// Flags some core reads by value: `vsyncmov` (except the exit check's: a `Semaphore` halt within
/// the next four bundles) and `vwait.eq`. `None` if any such operand is computed at run time.
fn observed_flags(m: &TpuMachine) -> Option<std::collections::BTreeSet<u32>> {
    use ptxv_tpu_ir::Op;
    let k = &m.kernel;
    let mut out = std::collections::BTreeSet::new();
    for (bi, b) in k.bundles.iter().enumerate() {
        for i in &b.insts {
            let flag = match &i.op {
                Op::SyncMov { flag } => {
                    let exit_check = k.bundles[bi..(bi + 5).min(k.bundles.len())]
                        .iter()
                        .flat_map(|b| &b.insts)
                        .any(|i| matches!(&i.op, Op::HaltErr { msg } if msg.contains("Semaphore")));
                    if exit_check {
                        continue;
                    }
                    flag
                }
                Op::Wait { flag, eq: true, .. } => flag,
                _ => continue,
            };
            out.insert(m.const_val(flag)?);
        }
    }
    Some(out)
}

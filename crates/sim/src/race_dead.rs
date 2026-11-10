/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Races whose racing loads' values are dead (class `race-dead`, ASSUMPTIONS.md
//! E-RACE-DEAD). A read-write race is a plain load by a thread that happens-before does not
//! order with a write; the value it returns depends on the schedule. When the loaded value
//! reaches nothing, no outcome depends on it. That is shown, not assumed: the check is run
//! again with every racing load (thread, pc) of the race leaving its destination registers
//! undefined (`Machine::dead_probe`), and S-INIT reports any undefined bit that reaches a
//! store, an address, a predicate or branch, an atomic or synchronization operand. If the
//! rerun reports nothing the first run did not, the race is kept as a finding of kind
//! `race-dead` (no verdict class, GB300 label kept); otherwise it stays `race`.
//!
//! Every racing read must be known: `Machine::race_log` records each one (exec.rs, before
//! the per-access deduplication of findings) over every explored transition, shared by all
//! copies of the machine, so a complete exploration has seen them all. A race with any other
//! kind of read side (a write-write race, an atomic, an acquiring load, an asynchronous unit)
//! has no entry and stays `race`. A tensormap proxy fence's read (`Op::TmapFence`) is a racing
//! read whose value is the snapshot later TMA ops compare against; the probe leaves the
//! snapshot undefined (exec.rs), so a TMA op that relies on it reports a stale map.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::machine::Finding;
use crate::machine::Machine;
use crate::machine::RaceLog;

/// A fresh, empty log (`Machine::race_log`); None when disabled (PTXV_NO_RACE_DEAD).
pub fn new_log() -> Option<RaceLog> {
    std::env::var("PTXV_NO_RACE_DEAD")
        .is_err()
        .then(|| Arc::new(std::sync::Mutex::new(Default::default())))
}

/// A failing finding other than a race: the check is not about to be clean, and a run that
/// already reports undefined values cannot show that the racing loads add none (an
/// undefined use of a never-written location has the same signature as one of a probed load).
fn failing(f: &Finding) -> bool {
    matches!(
        f.kind.as_str(),
        "violation" | "fault" | "deadlock" | "unmodeled"
    )
}

/// The races of `fs` whose every racing read is a thread's non-acquiring load, with those
/// reads (thread, pc). None when `fs` has a failing finding other than a race. `facq`: reads
/// a later acquire fence of their thread made acquiring (`Machine::fence_acq`); a race with
/// one of them stays `race` (the value read decided a synchronization, dead or not).
pub fn candidates(
    log: &RaceLog,
    fs: &[Finding],
    facq: &BTreeSet<(u32, u32)>,
) -> Vec<(String, BTreeSet<(u32, u32)>)> {
    if fs.iter().any(failing) {
        return Vec::new();
    }
    let g = log.lock().unwrap();
    fs.iter()
        .filter(|f| f.kind == "race")
        .filter_map(|f| match g.get(&f.sig) {
            Some(Some(r)) if !r.is_empty() && r.is_disjoint(facq) => {
                Some((f.sig.clone(), r.clone()))
            }
            _ => None,
        })
        .collect()
}

/// The rerun reports no failing finding (the first run had none, `candidates`), and no
/// contract finding the first run did not.
pub fn no_new_use(before: &[Finding], after: &[Finding]) -> bool {
    !after
        .iter()
        .any(|f| failing(f) || f.kind == "contract" && !before.iter().any(|g| g.sig == f.sig))
}

/// Turn the races `sigs` of `fs` into `race-dead` findings. ASSUME(E-RACE-DEAD)
pub fn mark(fs: &mut [Finding], sigs: &BTreeSet<String>) {
    for f in fs.iter_mut() {
        if f.kind == "race" && sigs.contains(&f.sig) {
            f.kind = "race-dead".into();
            f.msg = format!(
                "value dead (rerun with the racing loads undefined reports no use, S-INIT; E-RACE-DEAD): {}",
                f.msg
            );
        }
    }
}

/// Which candidate races are dead, given `probe(reads)`: true when a rerun with `reads`
/// undefined reports nothing new. All together first; when that fails, each alone (at most
/// `PER_RACE` of them).
pub fn decide(
    cands: &[(String, BTreeSet<(u32, u32)>)],
    mut probe: impl FnMut(&BTreeSet<(u32, u32)>) -> bool,
) -> BTreeSet<String> {
    const PER_RACE: usize = 8;
    if cands.is_empty() {
        return BTreeSet::new();
    }
    let all: BTreeSet<(u32, u32)> = cands.iter().flat_map(|c| c.1.iter().copied()).collect();
    if probe(&all) {
        return cands.iter().map(|c| c.0.clone()).collect();
    }
    if cands.len() == 1 {
        return BTreeSet::new();
    }
    cands
        .iter()
        .take(PER_RACE)
        .filter(|c| probe(&c.1))
        .map(|c| c.0.clone())
        .collect()
}

/// `m0` with the loads `reads` left undefined and no race log (a probe run).
pub fn probe_machine(m0: &Machine, reads: &BTreeSet<(u32, u32)>) -> Machine {
    let mut m = m0.clone();
    let mut set = m.dead_probe.as_deref().cloned().unwrap_or_default();
    set.extend(reads.iter().copied());
    m.dead_probe = Some(Arc::new(set));
    m.race_log = None;
    m.fence_acq = None;
    m
}

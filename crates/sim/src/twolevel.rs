/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Two-level verdicts: PTX-model findings, classified against the GB300 hardware model.
//!
//! Level 1 is the check as configured (the strict PTX memory model unless the case opts into
//! something). When it reports a race or a stale read, or relies on a refuted assumption, the
//! same check runs again under the hardware model (level 2, E-HW-L1):
//!
//! - every SM has the non-coherent L1 of `l1.rs` (`Launch::hw_l1`);
//! - behind the L1s memory is sequentially consistent (`Launch::hw_order`): a strong write
//!   passes on its writer's history and a strong read that observes it acquires that history,
//!   at any scope. This is the ordering half of E-POLL-ACQ (a miss after a poll exit returns
//!   the new data), which the GB300 litmus supports; its acquire half (that stale lines go
//!   away) is not assumed: the L1 decides that. Store reordering by a missing release fence is
//!   not in the model (GB300 never showed it: r4b B33 / B39 are latent there).
//!
//! Each level-1 race / stale finding is then classified (`Finding::gb300`):
//!
//! - **observable**: reachable under the hardware model: the same race remains (the accesses
//!   are unordered even with sequentially consistent memory, so either order can happen), or
//!   a weak read involved can return stale L1 bytes (an operational or predictive `stale`
//!   finding of level 2 at that read);
//! - **latent**: a PTX-model violation not reachable on GB300, with the reason (what orders
//!   the accesses, and why the read cannot hit a stale line: no fill of the line on that SM
//!   since the last invalidation, a strong load that bypasses L1, ...), and no outside fill
//!   could make it observable either;
//! - **latent-closed-world**: not reachable in the hardware model, which is closed-world
//!   (E-L1-CLOSED: only the case's own loads fill L1 lines), but a weak read involved is
//!   stale if any agent outside the case fills its line on the reader's SM between the
//!   reader's last invalidation and the write (E-L1-HAVOC). A legal co-resident kernel that
//!   reads other bytes of the line does that: on GB300 these are observable with such a
//!   neighbour (DeepEP notify_dispatch, docs/deepep-coreader.md). The witness is the havoc
//!   check's `stale-l1-havoc:<pc>`;
//! - **undecided**: no level-2 match, but level 2 was not exhaustive; also every alias-proxy
//!   race (`race:alias:`, S-MC-ALIAS), whose order this model does not represent;
//! - **latent-needs-divergence** (set when the race is reported, not here; PTX front only):
//!   two lanes of one warp that stays converged between the accesses, both plain memory
//!   instructions. GB300 issues them in program order (E-LOCKSTEP, probes/warp_order), so the
//!   race shows only if the warp diverges between them or the compiler reorders them.
//!
//! The havoc check (`l1_cores.rs`, `l1_havoc_check`) runs inside the level-2 run as a probe
//! (`L1::havoc_probe`): it is predictive (one run covers every schedule with the same
//! happens-before order) and changes nothing in the exploration; its witnesses are findings
//! of kind "probe", used only here.
//!
//! The classification is extra information: a latent finding is still a PTX-model bug and the
//! verdict's classes do not change.

use std::sync::Arc;
use std::time::Instant;

use serde::Deserialize;
use serde::Serialize;

use crate::explore::Verdict;
use crate::l1::L1Stats;
use crate::machine::Finding;
use crate::machine::Launch;
use crate::machine::Machine;

/// Class of an intra-warp PTX race in a converged warp (E-LOCKSTEP), set when the race is
/// reported (exec.rs), not by the hardware-model run.
pub const LATENT_DIV: &str = "latent-needs-divergence";

/// The GB300 classification of one PTX-model finding.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gb300 {
    /// "observable" | "latent" | "latent-closed-world" | "undecided" |
    /// "latent-needs-divergence" (`LATENT_DIV`).
    pub class: String,
    pub reason: String,
    /// The level-2 finding that shows it: observable, its signature; latent-closed-world, the
    /// havoc check's (`stale-l1-havoc:<pc>`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub witness: Option<String>,
}

/// The level-2 run and its tally.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Gb300Check {
    /// What level 2 assumes.
    pub model: String,
    /// Why it ran.
    pub why: String,
    pub states: usize,
    pub complete: bool,
    pub seconds: f64,
    pub observable: usize,
    pub latent: usize,
    /// Latent only under E-L1-CLOSED (observable with an outside fill, E-L1-HAVOC).
    #[serde(default)]
    pub latent_closed_world: usize,
    pub undecided: usize,
    /// Level-2 stale reads that match no level-1 finding (e.g. hidden by a refuted
    /// assumption level 1 was allowed to use): observable on GB300 (closed-world hardware
    /// model), or latent-closed-world (only the havoc probe, E-L1-HAVOC, reports the read).
    pub extra: Vec<Finding>,
}

pub const MODEL: &str = "E-HW-L1: non-coherent per-SM L1 (CCTL.IVALL and kernel boundaries invalidate) over sequentially consistent memory (a strong read that observes a strong write is ordered after the writer's history, at any scope); closed world (E-L1-CLOSED), with the havoc check (E-L1-HAVOC) as a probe for outside fills; no refuted assumption";

/// Does level 1's verdict call for the hardware model? Some(reason) if so.
pub fn wanted(v: &Verdict, launch: &Launch) -> Option<String> {
    let n = v.findings.iter().filter(|f| classifiable(f)).count();
    let mut why = Vec::new();
    if n > 0 {
        why.push(format!("{n} race / stale finding(s) to classify"));
    }
    for a in &launch.refuted_opt_in {
        why.push(format!("the verdict relies on {a} (refuted, opted in)"));
    }
    if why.is_empty() {
        None
    } else {
        Some(why.join("; "))
    }
}

/// Race / stale findings without a label yet (an intra-warp race in a converged warp is
/// labelled when reported, `LATENT_DIV`).
fn classifiable(f: &Finding) -> bool {
    (f.kind == "race" || f.kind == "stale") && f.gb300.is_none()
}

/// The hardware model's launch for a level-1 launch.
pub fn hw_launch(l: &Launch) -> Launch {
    let mut h = l.clone();
    h.hw_l1 = true;
    h.hw_order = true;
    h.mc_alias_hw = true; // E-MC-ALIAS-HW
    h.poll_acquire = false;
    h.refuted_opt_in.clear();
    // Closed world (E-L1-CLOSED): outside fills are probed separately (`hw_machine`), so a
    // level-1 havoc opt-in does not turn the level-2 model into havoc mode (which would also
    // drop the explored placements, l1_cores.rs).
    h.l1_havoc = false;
    h
}

/// The hardware model's machine for a level-1 machine (same kernel; fresh initial state),
/// its L1 reporting per-pc statistics to `st` (shared by all its clones).
pub fn hw_machine(m1: &Machine, st: &Arc<L1Stats>) -> Machine {
    let mut m = Machine::new(m1.k.clone(), Arc::new(hw_launch(&m1.launch)));
    if let Some(l1) = m.l1.as_mut() {
        let l1 = Arc::make_mut(l1);
        l1.stats = Some(st.clone());
        l1.hash_logs = true;
        // E-L1-HAVOC as a probe: witnesses of kind "probe" (latent vs latent-closed-world).
        l1.havoc_probe = true;
    }
    m
}

/// Level 2 for a check that takes one machine: runs `check` on the hardware-model machine
/// of `m1` and classifies `v1`'s findings in place.
pub fn run(
    v1: &mut Verdict,
    m1: &Machine,
    why: String,
    check: &mut dyn FnMut(Machine) -> Verdict,
) -> Gb300Check {
    let t0 = Instant::now();
    let st = Arc::new(L1Stats::default());
    let m2 = hw_machine(m1, &st);
    let v2 = check(m2.clone());
    let mut c = classify(v1, &v2, &st, &m2);
    c.why = why;
    c.seconds = t0.elapsed().as_secs_f64();
    c
}

/// The read side(s) of a finding, and whether it is a global-memory access.
fn reads_of(f: &Finding) -> (Vec<u32>, bool, bool) {
    // sig: race:<space>:<earlier pc>:<kind> / race-ldsts:... / race:tmem|mbar|proxy:...
    // stale-l1:<R> / stale-l1-pred:<R>:<F>
    let parts: Vec<&str> = f.sig.split(':').collect();
    if f.kind == "stale" {
        let r = f
            .pcs
            .first()
            .copied()
            .or_else(|| parts.get(1).and_then(|p| p.parse().ok()));
        return (r.into_iter().collect(), true, false);
    }
    let global = parts.get(1) == Some(&"gmem");
    let kind = parts.last().copied().unwrap_or("");
    let earlier = f
        .pcs
        .first()
        .copied()
        .or_else(|| parts.get(2).and_then(|p| p.parse().ok()));
    let later = f.pcs.get(1).copied();
    let r = match kind {
        "write-read" => later.into_iter().collect(),
        "read-write" => earlier.into_iter().collect(),
        _ => vec![],
    };
    (r, global, kind == "write-write")
}

/// The read pc of a level-2 stale finding.
fn stale_read(f: &Finding) -> Option<u32> {
    if f.kind != "stale" {
        return None;
    }
    f.pcs
        .first()
        .copied()
        .or_else(|| f.sig.split(':').nth(1).and_then(|p| p.parse().ok()))
}

/// The read pc of a level-2 havoc probe (`stale-l1-havoc:<R>`).
fn probe_read(f: &Finding) -> Option<u32> {
    if f.kind != "probe" {
        return None;
    }
    f.pcs
        .first()
        .copied()
        .or_else(|| f.sig.split(':').nth(1).and_then(|p| p.parse().ok()))
}

/// Classify `v1`'s race / stale findings against level 2's verdict `v2`.
pub fn classify(v1: &mut Verdict, v2: &Verdict, st: &L1Stats, m: &Machine) -> Gb300Check {
    // Exhaustive (a certified verdict is complete too). Callers that checked only part of
    // level 2 on purpose (launch groups up to where level 1 stopped) set `complete`.
    let decided = v2.complete;
    let mut used = vec![false; v2.findings.len()];
    let mut c = Gb300Check {
        model: MODEL.into(),
        states: v2.states,
        complete: decided,
        ..Default::default()
    };
    let d = |pc: u32| m.k.describe(pc as usize);
    for f in v1.findings.iter_mut().filter(|f| classifiable(f)) {
        // Alias-proxy races (S-MC-ALIAS) are outside this model, which orders memory by
        // address; E-MC-ALIAS-HW (`mc_alias = "hw"`) is the measured GB300 alternative.
        if f.sig.starts_with("race:alias:") {
            f.gb300 = Some(Gb300 {
                class: "undecided".into(),
                reason: "alias-proxy order (multicast vs unicast view) is not part of the hardware model; rerun with mc_alias = \"hw\" (E-MC-ALIAS-HW) for the measured GB300 behavior".into(),
                witness: None,
            });
            c.undecided += 1;
            continue;
        }
        let (reads, global, ww) = reads_of(f);
        // 1. The same finding under the hardware model.
        if let Some(i) = v2
            .findings
            .iter()
            .position(|g| g.sig == f.sig && g.kind != "probe")
        {
            used[i] = true;
            let reason = if f.kind == "stale" {
                "the stale L1 read is reachable under the hardware model".to_string()
            } else {
                "the accesses are unordered even with GB300's sequentially consistent memory: either order is reachable, so the outcome depends on timing".to_string()
            };
            f.gb300 = Some(Gb300 {
                class: "observable".into(),
                reason,
                witness: Some(f.sig.clone()),
            });
            c.observable += 1;
            continue;
        }
        // 2. A stale L1 read at one of its reads.
        if let Some(i) = v2
            .findings
            .iter()
            .position(|g| stale_read(g).is_some_and(|r| reads.contains(&r)))
        {
            used[i] = true;
            let g = &v2.findings[i];
            f.gb300 = Some(Gb300 {
                class: "observable".into(),
                reason: format!(
                    "the read at {} can return stale L1 bytes: {}",
                    d(stale_read(g).unwrap()),
                    g.msg.lines().next().unwrap_or("")
                ),
                witness: Some(g.sig.clone()),
            });
            c.observable += 1;
            continue;
        }
        // 3. Not reachable: why.
        let mut why = Vec::new();
        if f.kind == "race" {
            why.push(
                "with GB300's ordering (sequentially consistent memory behind the L1s: a relaxed poll that observed a signal is ordered after everything its writer did before the signal) the accesses are ordered"
                    .to_string(),
            );
        }
        if ww {
            why.push("both accesses are writes, and L1 is write-through".into());
        } else if !global && f.kind == "race" {
            why.push("not a global-memory access, so L1 does not apply".into());
        } else {
            for &r in &reads {
                why.push(match st.get(r) {
                    None => format!(
                        "the read at {} does not go through L1 (strong / volatile / bulk load){}",
                        d(r),
                        if decided { "" } else { ", or was not reached" }
                    ),
                    Some(s) => {
                        let line = match s.line {
                            Some((sm, a, l)) => format!(
                                "line {}+{l:#x} on SM {}",
                                m.launch
                                    .allocs
                                    .get(a as usize)
                                    .map(|x| x.name.as_str())
                                    .unwrap_or("?"),
                                if sm >= 1 << 20 {
                                    format!("group {}", sm - (1 << 20))
                                } else {
                                    format!("{sm} (CTA {sm})")
                                }
                            ),
                            None => "its line".into(),
                        };
                        let since = match s.last_inval {
                            Some(p) => format!("the L1 invalidation at {}", d(p)),
                            None => "the kernel started".into(),
                        };
                        if s.hits == 0 {
                            format!(
                                "no fill of {line} between {since} and the read at {}: all {} weak loads there missed L1 and read memory",
                                d(r),
                                s.lookups
                            )
                        } else {
                            format!(
                                "every fill of {line} between {since} and the read at {} comes after the write landed: of {} weak loads there, {} hit a line filled after the write (first by {}), none a stale one",
                                d(r),
                                s.lookups,
                                s.hits,
                                s.hit_fill.map(d).unwrap_or_else(|| "?".into())
                            )
                        }
                    }
                });
            }
        }
        // 4. Closed world or not: a havoc witness (E-L1-HAVOC) at one of its reads.
        let havoc = v2
            .findings
            .iter()
            .position(|g| probe_read(g).is_some_and(|r| reads.contains(&r)));
        let mut witness = None;
        if let Some(i) = havoc {
            used[i] = true;
            let g = &v2.findings[i];
            let r = probe_read(g).unwrap();
            if decided {
                why.insert(
                    0,
                    "latent only under E-L1-CLOSED (no agent outside the case fills lines of the case's buffers)".into(),
                );
                why.push(format!(
                    "OBSERVABLE if a co-resident kernel fills the line (E-L1-HAVOC, {}): {}",
                    g.sig,
                    g.msg.lines().next().unwrap_or("")
                ));
                witness = Some(g.sig.clone());
            } else {
                why.push(format!(
                    "with an outside fill (E-L1-HAVOC) the read at {} can be stale ({})",
                    d(r),
                    g.sig
                ));
            }
        }
        let (class, n) = if decided && witness.is_some() {
            ("latent-closed-world", &mut c.latent_closed_world)
        } else if decided {
            ("latent", &mut c.latent)
        } else {
            why.insert(
                0,
                format!(
                    "not reached in the hardware-model check, which was not exhaustive ({} states)",
                    v2.states
                ),
            );
            ("undecided", &mut c.undecided)
        };
        *n += 1;
        f.gb300 = Some(Gb300 {
            class: class.into(),
            reason: why.join("; "),
            witness,
        });
    }
    for (i, g) in v2.findings.iter().enumerate() {
        if !used[i] && g.kind == "stale" {
            let mut g = g.clone();
            g.gb300 = Some(Gb300 {
                class: "observable".into(),
                reason: "a stale L1 read under the hardware model that the level-1 model does not report".into(),
                witness: Some(g.sig.clone()),
            });
            c.extra.push(g);
        }
    }
    // Havoc witnesses at reads no finding of either level covers (e.g. a race level 1 did not
    // report because it relied on a refuted assumption): stale only with an outside fill.
    let covered: Vec<u32> = v1
        .findings
        .iter()
        .filter(|f| f.kind == "race" || f.kind == "stale")
        .flat_map(|f| reads_of(f).0)
        .chain(v2.findings.iter().filter_map(stale_read))
        .collect();
    let mut seen = std::collections::BTreeSet::new();
    for (i, g) in v2.findings.iter().enumerate() {
        let Some(r) = probe_read(g) else { continue };
        if used[i] || covered.contains(&r) || !seen.insert(r) {
            continue;
        }
        let mut g = g.clone();
        g.kind = "stale".into();
        g.gb300 = Some(Gb300 {
            class: "latent-closed-world".into(),
            reason: format!(
                "neither model reports a stale read at {} in a closed world (E-L1-CLOSED), but it is stale if an agent outside the case fills the line (E-L1-HAVOC, {})",
                d(r),
                g.sig
            ),
            witness: Some(g.sig.clone()),
        });
        c.extra.push(g);
    }
    c
}

/// One line per classified finding, for the text report.
pub fn summary_line(c: &Gb300Check) -> String {
    format!(
        "GB300 model (E-HW-L1, {}): {} observable, {} latent, {} latent-closed-world (latent only under E-L1-CLOSED){}{}; {} states{} {:.2}s",
        c.why,
        c.observable,
        c.latent,
        c.latent_closed_world,
        if c.undecided > 0 {
            format!(", {} undecided", c.undecided)
        } else {
            String::new()
        },
        {
            let cw = c
                .extra
                .iter()
                .filter(|g| g.gb300.as_ref().is_some_and(|x| x.class != "observable"))
                .count();
            let ob = c.extra.len() - cw;
            let mut s = String::new();
            if ob > 0 {
                s += &format!(", {ob} more stale read(s) only the hardware model sees");
            }
            if cw > 0 {
                s += &format!(
                    ", {cw} more stale read(s) only with an outside fill (latent-closed-world)"
                );
            }
            s
        },
        c.states,
        if c.complete { "" } else { " INCOMPLETE" },
        c.seconds
    )
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Hardware profiles: every probe-backed hardware assumption of one GPU, turned on at once,
//! for the second level of the two-level verdict (`ptxv check`, `ptx_contract` /
//! `sass_gb300`). A profile assumption is either a model switch (the run itself changes) or
//! a discharge (a `contract` finding whose deviation the probe showed the hardware tolerates
//! stops counting). Each entry names its probe; ASSUMPTIONS.md has the evidence. Nothing
//! that is only argued goes in.

use std::sync::Arc;

use crate::explore::Verdict;
use crate::machine::Launch;
use crate::machine::Machine;

/// One assumption of a profile.
#[derive(Clone, Copy, Debug)]
pub struct Item {
    /// ASSUMPTIONS.md id.
    pub id: &'static str,
    /// The probe (or litmus) that backs it.
    pub probe: &'static str,
    /// `contract` finding signature prefixes it discharges; empty: a model switch.
    pub discharges: &'static [&'static str],
}

/// GB300 (sm_103a, CUDA 13.0): the probe-backed assumptions.
pub const GB300: &[Item] = &[
    Item {
        id: "E-HW-L1",
        probe: "docs/poll-acq-litmus.md (non-coherent per-SM L1, stale hits after a fenceless poll)",
        discharges: &[],
    },
    Item {
        id: "E-MC-ALIAS-HW",
        probe: "probes/multimem mm litmus, probes/l2alias (multicast write then unicast read; unicast write then multimem.ld_reduce; write-after-read not probed)",
        discharges: &[],
    },
    Item {
        id: "E-REMOTE-FIFO",
        probe: "probes/remote_arrive_exit (order.cu: remote arrivals never out of happens-before issue order)",
        discharges: &[],
    },
    Item {
        id: "E-BULK-EXIT",
        probe: "probes/tma_store_exit",
        discharges: &["bulk-store-at-exit:"],
    },
    Item {
        id: "E-TC-EXIT",
        probe: "probes/tc_commit_exit",
        discharges: &["tc-commit-at-exit:"],
    },
    Item {
        id: "E-DESC-VERSION",
        probe: "probes/desc_version",
        discharges: &["desc-version:"],
    },
    Item {
        id: "S-TMAP-TAIL",
        probe: "probes/tmap_tail",
        discharges: &["tmap-tail-undef:"],
    },
];

/// The profile named `name` ("gb300"), or None ("none" / "off" / unknown).
pub fn named(name: &str) -> Option<&'static [Item]> {
    match name {
        "gb300" => Some(GB300),
        _ => None,
    }
}

/// The launch under the profile: its model switches on (ASSUME(E-HW-L1),
/// ASSUME(E-MC-ALIAS-HW), ASSUME(E-REMOTE-FIFO)), refuted opt-ins off (a profile is the hardware, not an opt-in).
pub fn launch(l: &Launch, p: &[Item]) -> Launch {
    let mut h = l.clone();
    for it in p {
        match it.id {
            "E-HW-L1" => h.hw_l1 = true,
            "E-MC-ALIAS-HW" => h.mc_alias_hw = true,
            "E-REMOTE-FIFO" => h.remote_fifo = true,
            _ => {}
        }
    }
    h.poll_acquire = false;
    h.refuted_opt_in.clear();
    h
}

/// A fresh machine for `m1`'s kernel under the profile.
pub fn machine(m1: &Machine, p: &[Item]) -> Machine {
    Machine::new(m1.k.clone(), Arc::new(launch(&m1.launch, p)))
}

/// Removes the contract findings the profile discharges from `v` and returns the profile
/// assumptions the verdict used: the model switches that apply to the launch (E-HW-L1
/// always; E-MC-ALIAS-HW with multicast objects; E-REMOTE-FIFO when the kernel has remote
/// mbarrier arrivals) and each assumption that discharged a finding, with the discharged
/// signatures.
pub fn discharge(v: &mut Verdict, m: &Machine, p: &[Item]) -> Vec<(String, Vec<String>)> {
    let l = &m.launch;
    let mut used: Vec<(String, Vec<String>)> = Vec::new();
    for it in p {
        let applies = match it.id {
            "E-HW-L1" => true,
            "E-MC-ALIAS-HW" => !l.mcast.is_empty(),
            "E-REMOTE-FIFO" => {
                m.k.code
                    .iter()
                    .any(|i| matches!(i.op, ptxv_ir::Op::MbarArrive { remote: true, .. }))
            }
            _ => false,
        };
        let sigs: Vec<String> = v
            .findings
            .iter()
            .filter(|f| f.kind == "contract" && it.discharges.iter().any(|d| f.sig.starts_with(d)))
            .map(|f| f.sig.clone())
            .collect();
        if applies || !sigs.is_empty() {
            used.push((it.id.to_string(), sigs));
        }
    }
    v.findings.retain(|f| {
        !(f.kind == "contract"
            && p.iter()
                .any(|it| it.discharges.iter().any(|d| f.sig.starts_with(d))))
    });
    used
}

/// Whether a level-1 verdict needs the second level: any outcome the strict model calls
/// undefined or a deviation (a race, a stale read, a violation, a contract deviation,
/// undefined values, a deadlock, several outcomes), not only an unmodeled step or a bounded
/// search that found nothing (`inconclusive`).
pub fn wanted(v: &Verdict) -> bool {
    v.classes()
        .iter()
        .any(|c| c != "clean" && c != "unmodeled" && c != "timeout" && c != "inconclusive")
}

/// The second level's one-word result: "certified" (an E-WELLSYNC certificate with no
/// finding left), "clean" (exhaustively explored, no finding left), "incomplete", or the
/// remaining classes joined by ','.
pub fn result(v: &Verdict) -> String {
    let c = v.classes();
    if c.iter().all(|x| x == "clean") {
        if v.assumptions.iter().any(|a| a == "E-WELLSYNC") {
            "certified".into()
        } else if v.complete {
            "clean".into()
        } else {
            "incomplete".into()
        }
    } else {
        c.join(",")
    }
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Hardware profiles: every hardware-measured assumption of one GPU (a probe, or a captured
//! kernel's GB300 output where no probe exists yet: E-HW-SB-QUEUE), turned on at once,
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

/// GB300 (sm_103a, CUDA 13.0): the hardware-measured assumptions (each item names its
/// evidence).
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
        id: "E-HW-SPLIT-UR",
        probe: "probes/ur_window (0 split interleavings across a stalled R2UR..BRA.DIV window in 268,022 warp observations), probes/split_yield",
        discharges: &[],
    },
    Item {
        id: "E-BULK-EXIT",
        probe: "probes/tma_store_exit",
        discharges: &["bulk-store-at-exit:"],
    },
    Item {
        id: "E-CPASYNC-EXIT",
        probe: "probes/cpasync_exit",
        discharges: &["cp-async-exit:"],
    },
    Item {
        id: "E-TC-EXIT",
        probe: "probes/tc_commit_exit",
        discharges: &["tc-commit-at-exit:"],
    },
    Item {
        id: "E-HW-BSYNC-COLLECTIVE",
        probe: "fixtures/e_bsync_parked_collective (a BSYNC passes while the rest of its mask waits at a WARPSYNC.COLLECTIVE)",
        discharges: &[],
    },
    Item {
        id: "E-HW-BSYNC-YIELD",
        probe: "probes/mbar_init_diverge late_sleep_reconv (lanes pass a plain BSYNC while the elected lane sleeps: 10,000 of 10,000 runs), probes/bsync_coll, probes/split_yield (no switch without a YIELD or a blocking instruction)",
        discharges: &[],
    },
    Item {
        id: "E-HW-STS-LOWEST",
        probe: "fixtures/e_lane_order (32 lanes of one shared store, different values: lane 0's value in 500 of 500 runs of each of six variants), fixtures/e_sts_shapes (19 widths and lane sets, 500 of 500 runs each: the lowest active lane of the last 32 / (bytes / 4)-lane phase), fixtures/e_sts_keep and e_sts_keep_wide (traps on that lane's value in every launch)",
        discharges: &[],
    },
    Item {
        id: "E-HW-STG-LOWEST",
        probe: "fixtures/e_stg_keep (one 32-bit STG by every lane, by lanes 8..23, by the odd lanes; plain, .STRONG.SM, .GPU, .SYS, .EF, .NA, after a MEMBAR, split over two words: the lowest active lane's value in 500 of 500 runs of each of 16 variants; a trap on lane 0's value in 5 of 5 launches, 20 of 20 in review; STG.E.MMIO.SYS keeps the highest active lane's value and is explored)",
        discharges: &[],
    },
    Item {
        id: "E-DESC-VERSION",
        probe: "probes/desc_version",
        discharges: &["desc-version:"],
    },
    // Not E-TMEM-QUEUE: its probe (probes/tmem_queue) is a measured null without a
    // discriminating control, so `tmem-queue` contract findings stay contracts here too.
    // ASSUME(E-HW-PROXY-WAR): discharges the strict level's S-PROXY-WAR contract findings.
    Item {
        id: "E-HW-PROXY-WAR",
        probe: "probes/handoff_window v=16/17 (a generic read, happens-before without a proxy fence, then a peer's bulk copy over the bytes: 0 / 6.08e9 and 0 / 3.04e9 under contention; positive control v=15, the same read with no happens-before: up to 7678 / 7.6e8)",
        discharges: &["proxy-war:"],
    },
    // ASSUME(E-HW-SB-QUEUE): discharges the strict level's scoreboard findings that only the
    // GB300 queue rules clear (scoreboard.rs `HW_QUEUE_SIG`).
    Item {
        id: "E-HW-SB-QUEUE",
        probe: "probes/redux_queue COVER (a REDUX read after a wait on a later REDUX's scoreboard only: 1000 / 1000 correct; positive control NOWAIT wrong 1000 / 1000); probes/f2i64_f64, i2f64_s16 COVER (F2I.S64.F64.TRUNC by F2I.F64.TRUNC, I2F.F64.S16 by I2F.F64: 0 / 1000, NOWAIT 1000 / 1000; the other f64 conversion pairs, the f64 pipe and POPC / FLO across opcodes, the conversion / f64 operand-read order and UGETNEXTWORKID's reads at issue are sb-unmeasured: contracts, MISS 170), sb_mio_order, stg_guard_war (memory-queue reads), frnd_same COVER (FRND by FRND.FLOOR, FRND.F64 by FRND.F64.FLOOR: 0 / 1000, NOWAIT controls 1000 / 1000; FRND.F64 by an f32 FRND refuted: frnd64, MISS 156), shared_queue COVER (an LDS covered by a later SHFL), smem_queue_w1 COVER (a conflicted LDSM.16.M88.4 by a later LDS / SHFL.BFLY, an LDS by a later LDSM.16.M88.4 / MT88.4, 2 to 8 stall cycles apart: 0 / 1000, NOWAIT / DELAY / MUFU control 1000 / 1000; the broader queue orders are sb-unmeasured: contracts, not discharged), cvt64_queue F2I32_COVER (a 32-bit F2I by a later F2I.S64), sb_mio_order rd_on_ldg / rd_on_lds (read order across memory-queue opcodes), wcov (write cover across opcodes), qwar_lds / qwar_ldg (a load after another opcode's read); TMA store reads: the CUTLASS FMHA varlen capture's output = GB300, no probe yet; tcgen05 reads: fixtures/c_fa4_causal and the UTCCP runs",
        discharges: &["sb-hwq:"],
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
            "E-HW-BSYNC-COLLECTIVE" => h.hw_bsync_coll = true,
            "E-HW-SPLIT-UR" => h.hw_split_ur = true,
            "E-HW-STS-LOWEST" => h.hw_sts_lowest = true,
            "E-HW-STG-LOWEST" => h.hw_stg_lowest = true,
            "E-HW-BSYNC-YIELD" => h.hw_bsync_yield = true,
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
            "E-HW-SPLIT-UR" => !m.ur_window.is_empty(),
            "E-HW-STS-LOWEST" => m.k.code.iter().any(|i| {
                matches!(
                    i.op,
                    ptxv_ir::Op::St {
                        space: ptxv_ir::Space::Shared,
                        ..
                    }
                )
            }),
            "E-HW-STG-LOWEST" => l.mcast.is_empty() && m.k.code.iter().any(|i| {
                matches!(
                    i.op,
                    ptxv_ir::Op::St {
                        space: ptxv_ir::Space::Global,
                        bytes: 4,
                        ..
                    }
                ) && !i.text.contains(".MMIO")
            }),
            "E-REMOTE-FIFO" => {
                m.k.code
                    .iter()
                    .any(|i| matches!(i.op, ptxv_ir::Op::MbarArrive { remote: true, .. }))
            }
            "E-HW-BSYNC-YIELD" => m.k.code.iter().any(|i| {
                matches!(i.op, ptxv_ir::Op::Bsync { form, .. } if !crate::exec::bsync_holds(form))
            }),
            // Named when it released a BSYNC in the run (its note).
            "E-HW-BSYNC-COLLECTIVE" => v.findings.iter().any(|f| f.sig.starts_with("bsync-coll:")),
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

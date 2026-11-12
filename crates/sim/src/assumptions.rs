/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The assumption registry (long form: ASSUMPTIONS.md). Every verdict lists the IDs that
//! apply to its kernel; `tests/assumptions.rs` keeps this table, the document and the
//! `ASSUME(ID)` tags in the code in sync.

use ptxv_ir::FenceKind;
use ptxv_ir::Front;
use ptxv_ir::Kernel;
use ptxv_ir::Op;

pub struct Assumption {
    pub id: &'static str,
    /// "code" (tagged in the source) or "process" (how we test; not tagged).
    pub kind: &'static str,
    pub status: &'static str,
    pub title: &'static str,
    /// Does a verdict on this kernel depend on it?
    pub applies: fn(&Kernel) -> bool,
}

fn any(k: &Kernel, f: impl Fn(&Op) -> bool) -> bool {
    k.code.iter().any(|i| f(&i.op))
}
fn sass(k: &Kernel) -> bool {
    k.front == Front::Sass
}
fn always(_: &Kernel) -> bool {
    true
}
fn never(_: &Kernel) -> bool {
    false
}
fn tmem(k: &Kernel) -> bool {
    any(k, |o| {
        matches!(
            o,
            Op::TmemAlloc { .. }
                | Op::TmemAllocRaw { .. }
                | Op::TmemLd { .. }
                | Op::TmemDealloc { .. }
                | Op::Mma { .. }
        )
    })
}
/// Uses another CTA of its cluster (DSMEM, multicast, cluster barrier, cluster rank).
fn cluster(k: &Kernel) -> bool {
    any(k, |o| {
        matches!(
            o,
            Op::Mapa { .. }
                | Op::ClusterArrive { .. }
                | Op::ClusterWait
                | Op::MbarArrive { remote: true, .. }
                | Op::TmaLoad { mask: Some(_), .. }
        )
    })
}
fn mbar(k: &Kernel) -> bool {
    any(k, |o| {
        matches!(
            o,
            Op::MbarInit { .. } | Op::MbarArrive { .. } | Op::MbarTryWait { .. }
        )
    })
}

pub const REGISTRY: &[Assumption] = &[
    Assumption {
        id: "E-SPIN",
        kind: "code",
        status: "enforced",
        title: "a failed try_wait / flag poll blocks until it would succeed (loop shape checked)",
        applies: |k| {
            any(k, |o| matches!(o, Op::MbarTryWait { .. }))
                || !crate::spin::poll_loops(&k.code, k.front).is_empty()
                || !crate::spin::wpoll_loops(&k.code, k.front).is_empty()
        },
    },
    Assumption {
        id: "E-CLOCK",
        kind: "code",
        status: "argued",
        title: "clock reads (%clock64 / %globaltimer, CS2R SR_CLOCKLO / SR_GLOBALTIMERLO) return 0 (opt-in PTXV_CLOCK_STEP: a per-thread counter): timeouts never fire, vprintf is a no-op",
        applies: |k| {
            k.code.iter().any(|i| {
                i.text.contains("SR_CLOCKLO")
                    || i.text.contains("SR_GLOBALTIMERLO")
                    || i.text.contains("vprintf")
                    || i.op
                        .src_vals()
                        .iter()
                        .any(|v| matches!(v, ptxv_ir::Val::SReg(s) if s.is_clock()))
            })
        },
    },
    Assumption {
        id: "E-ALLOC",
        kind: "code",
        status: "argued",
        title: "a TMEM alloc that cannot succeed blocks until columns are freed",
        applies: |k| {
            any(k, |o| {
                matches!(o, Op::TmemAlloc { .. } | Op::TmemAllocRaw { .. })
            })
        },
    },
    Assumption {
        id: "E-RACE-DEAD",
        kind: "code",
        status: "argued",
        title: "a race whose racing loads' values reach no S-INIT sink does not change the outcome",
        applies: |k| any(k, |o| matches!(o, Op::Ld { .. })),
    },
    Assumption {
        id: "E-CTA-REP",
        kind: "code",
        status: "argued",
        title: "a CTA whose replayed position-dependent control values equal a certified representative's takes its path",
        // Added to a verdict only when `--rep` decided it (rep.rs).
        applies: never,
    },
    Assumption {
        id: "E-INPUT-INDEP",
        kind: "code",
        status: "argued",
        title: "input coverage's provenance covers every flow of input bytes into guards, branch conditions, addresses and synchronization operands of its run",
        // Added to a verdict only when a reduction relies on it.
        applies: never,
    },
    Assumption {
        id: "C-EXH-PATH",
        kind: "code",
        status: "argued",
        title: "inputs that agree on every control decision before position i make the same decision at i (deterministic coverage run)",
        // Listed by an exhaustive concolic expansion record (cert.rs), not by verdicts.
        applies: never,
    },
    Assumption {
        id: "C-EXH-CONTROL",
        kind: "code",
        status: "argued",
        title: "comparisons and selects reach control only through a guard or branch condition, an atom of its own",
        // Listed by an exhaustive concolic expansion record (cert.rs), not by verdicts.
        applies: never,
    },
    Assumption {
        id: "C-EXH-DEP",
        kind: "code",
        status: "argued",
        title: "every input dependence reaching a decision or operand is in the expressions or the provenance (cross-checked)",
        // Listed by an exhaustive concolic expansion record (cert.rs), not by verdicts.
        applies: never,
    },
    Assumption {
        id: "C-EXH-UNSAT",
        kind: "code",
        status: "validated",
        title: "a region is infeasible when two SMT solvers answer unsat for an encoding that over-approximates the simulator",
        // Listed by an exhaustive concolic expansion record (cert.rs), not by verdicts.
        applies: never,
    },
    Assumption {
        id: "C-EXH-TRANSFER",
        kind: "code",
        status: "argued",
        title: "a path's representative verdict holds for every input on the path when every input-dependent operand is fixed by the path",
        // Listed by an exhaustive concolic expansion record (cert.rs), not by verdicts.
        applies: never,
    },
    Assumption {
        id: "C-EXH-LAUNCH",
        kind: "code",
        status: "enforced",
        title: "every byte the launch writes goes through the shadow (NIC writers block)",
        // Listed by an exhaustive concolic expansion record (cert.rs), not by verdicts.
        applies: never,
    },
    Assumption {
        id: "E-DRF",
        kind: "code",
        status: "argued",
        title: "DRF-SC: plain and TMEM accesses are not scheduling points",
        applies: always,
    },
    Assumption {
        id: "E-ELECT",
        kind: "code",
        status: "argued",
        title: "elect: SASS lowest lane (GB300 probe); PTX fixed leader, lowest and highest explored",
        applies: |k| any(k, |o| matches!(o, Op::Elect { .. })),
    },
    Assumption {
        id: "E-LOCKSTEP",
        kind: "code",
        status: "argued",
        title: "SASS: lanes executing together in a split are ordered instruction by instruction (STSM / LDSM included)",
        applies: sass,
    },
    Assumption {
        id: "E-STSM-DUP",
        kind: "code",
        status: "argued",
        title: "lanes of one stmatrix writing a shared word with the same value write it once; different values are a violation",
        applies: |k| any(k, |o| matches!(o, Op::StMatrix { .. })),
    },
    Assumption {
        id: "S-REMOTE-RESERVED",
        kind: "code",
        status: "argued",
        title: "remote arrivals on reserved-prefix mbarriers (ptxas's tcgen05.alloc cta_group::2 handshake at 0x48) are ptxas's protocol: lifetime findings about them are contract findings",
        applies: never,
    },
    Assumption {
        id: "S-INIT-FILLS",
        kind: "code",
        status: "argued",
        title: "undef-insensitive: undefined values that reach only output data leave the outputs bitwise equal over 6 fills (empirical evidence, not a certificate)",
        applies: never,
    },
    Assumption {
        id: "S-INIT-DEAD",
        kind: "code",
        status: "trusted",
        title: "a buffer declared dead workspace (init = \"undef\", scratch = true, not an output) is never read after the case: undefined data left in it at the end is a note, not an S-INIT violation",
        applies: never,
    },
    Assumption {
        id: "S-TMAP-TAIL",
        kind: "code",
        status: "validated",
        title: "GB300's TMA ignores bytes 0x40..0x7f of a tensor map in global memory (undefined bytes there: contract finding)",
        applies: never,
    },
    Assumption {
        id: "E-WARP-ORDER",
        kind: "code",
        status: "validated",
        title: "a converged warp's shared-memory instructions issue in program order (SASS: lockstep join; PTX: intra-warp races labelled latent-needs-divergence)",
        applies: sass,
    },
    Assumption {
        id: "E-ALIGNED",
        kind: "code",
        status: "validated",
        title: ".sync.aligned instructions and aligned barriers are warp convergence points",
        applies: |k| any(k, |o| o.is_warp_aligned() || o.is_aligned_bar()),
    },
    Assumption {
        id: "E-WELLSYNC",
        kind: "code",
        status: "argued",
        title: "a well-synchronized run (every phase, wait, generation, value determined) stands for every schedule",
        // Added to a verdict only when `ptxv check --certify` decided it (wellsync.rs).
        applies: never,
    },
    Assumption {
        id: "E-LAUNCH-GROUPS",
        kind: "code",
        status: "argued",
        title: "launch groups closed under stream order, checked one after another with a cross-group race check, stand for the whole case",
        // Added to a verdict only when `ptxv check --groups` decided it (groups.rs).
        applies: never,
    },
    Assumption {
        id: "E-MC-ALIAS-HW",
        kind: "code",
        status: "validated",
        title: "opt-in: multicast and unicast views of NVLS memory need no fence.proxy.alias (GB300 litmus: multicast write / unicast read and unicast write / multimem.ld_reduce)",
        // Added to a verdict when the case sets mc_alias = "hw" (for_machine).
        applies: never,
    },
    Assumption {
        id: "E-EXIT-FENCE",
        kind: "code",
        status: "argued",
        title: "kernel completion is a fence.acq_rel at the case's exit_fence scope (opt-in)",
        // Added to a verdict when the case sets `exit_fence` (explore.rs).
        applies: never,
    },
    Assumption {
        id: "E-POLL-ACQ",
        kind: "code",
        status: "refuted",
        title: "leaving a warp-level poll loop acquires what its relaxed polls observed (opt-in)",
        // Added to a verdict when the case sets `poll_acquire`.
        applies: never,
    },
    Assumption {
        id: "E-SMEM-CUMUL",
        kind: "code",
        status: "validated",
        title: "a SASS shared store releases, at CTA scope, what its thread acquired (opt-in)",
        // Added to a verdict when the case sets `smem_cumulative`.
        applies: never,
    },
    Assumption {
        id: "E-HW-CORW",
        kind: "code",
        status: "argued",
        title: "GB300 hardware model (two-level only): a write after the thread's strong read of the last write's bytes is coherence-after every earlier write of them",
        // Part of the level-2 hardware model (twolevel.rs MODEL), not of level-1 verdicts.
        applies: never,
    },
    Assumption {
        id: "E-HW-REMOTE-MEMBAR",
        kind: "code",
        status: "probed",
        title: "GB300 hardware model (two-level only): MEMBAR.ALL.GPU / SYS delivers the thread's in-flight remote shared::cluster stores",
        // Part of the level-2 hardware model (twolevel.rs MODEL), not of level-1 verdicts.
        applies: never,
    },
    Assumption {
        id: "E-HW-WEAK-FLAG",
        kind: "code",
        status: "validated",
        title: "GB300 hardware model (two-level only): an aligned weak global store of <= 8 bytes after a sys fence is a relaxed sys store releasing the fence's clock",
        // Part of the level-2 hardware model (twolevel.rs MODEL), not of level-1 verdicts.
        applies: never,
    },
    Assumption {
        id: "E-HW-L1",
        kind: "code",
        status: "probed",
        title: "opt-in non-coherent per-SM L1: weak global loads may hit stale lines; CCTL.IVALL and kernel boundaries invalidate",
        // Added to a verdict when the case sets `hw_l1`.
        applies: never,
    },
    Assumption {
        id: "E-L1-CLOSED",
        kind: "code",
        status: "argued",
        title: "closed world for L1: no agent outside the case loads case buffers on an SM the case uses",
        // Added to a verdict when `hw_l1` is on and `l1_havoc` is off.
        applies: never,
    },
    Assumption {
        id: "E-L1-PLACE",
        kind: "code",
        status: "unvalidated",
        title: "L1 placement: each CTA on its own SM (or PTXV_L1_COLOCATE), no co-resident CTAs",
        // Added to a verdict when `hw_l1` is on and neither `l1_coresident` nor `l1_havoc`.
        applies: never,
    },
    Assumption {
        id: "E-L1-CORES",
        kind: "code",
        status: "argued",
        title: "co-resident launches share SM L1s; pair placements within SM limits decide reachability; a CTA launch drops other grids' lines",
        // Added to a verdict when `l1_coresident` is on (and `l1_havoc` off).
        applies: never,
    },
    Assumption {
        id: "E-L1-HAVOC",
        kind: "code",
        status: "argued",
        title: "any outside agent may fill any case line on any SM at any time (predictive check)",
        // Added to a verdict when `l1_havoc` is on.
        applies: never,
    },
    Assumption {
        id: "E-SLICE",
        kind: "code",
        status: "argued",
        title: "warps that share nothing are explored one slice at a time, across CTAs and GPUs",
        // Added to a verdict explored by thread slices.
        applies: never,
    },
    Assumption {
        id: "E-POLL-ANY",
        kind: "code",
        status: "argued",
        title: "a strong load whose step ends alike for every value it can read commutes with its location's writes",
        applies: |k| any(k, |o| matches!(o, Op::Ld { sync: Some(_), .. })),
    },
    Assumption {
        id: "E-READ-DONE",
        kind: "code",
        status: "argued",
        title: "a used read is ordered before writes that depend on a later strong store of its thread (opt-in)",
        // Added to a verdict when the case sets `read_done`.
        applies: never,
    },
    Assumption {
        id: "E-HW-BSYNC-COLLECTIVE",
        kind: "code",
        status: "validated",
        title: "GB300 profile: a BSYNC releases its parked lanes while the rest of its lane mask waits at a WARPSYNC.COLLECTIVE",
        // Named by the GB300 profile (hwprofile.rs) when it released a BSYNC in the run.
        applies: never,
    },
    Assumption {
        id: "E-REMOTE-FIFO",
        kind: "code",
        status: "validated",
        title: "remote mbarrier arrivals from one CTA to one CTA land in happens-before issue order (opt-in)",
        // Added to a verdict when the case sets `remote_fifo`.
        applies: never,
    },
    Assumption {
        id: "E-CTA-INDEP",
        kind: "code",
        status: "argued",
        title: "CTAs with no shared synchronization object are explored one at a time",
        applies: |k| !cluster(k),
    },
    Assumption {
        id: "E-WARPCOLL",
        kind: "code",
        status: "argued",
        title: "stmatrix / shfl exchange data among the lanes of a converged warp in one step",
        applies: |k| any(k, |o| matches!(o, Op::StMatrix { .. } | Op::Shfl { .. })),
    },
    Assumption {
        id: "E-COLLECTIVE",
        kind: "code",
        status: "validated",
        title: "BRA.DIV moves the whole split; WARPSYNC.COLLECTIVE regions run one member-mask group at a time",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::Bra { div: Some(_), .. }
                        | Op::WarpSync {
                            collective: true,
                            ..
                        }
                )
            })
        },
    },
    Assumption {
        id: "E-COLLECTIVE-ALL",
        kind: "code",
        status: "validated",
        title: "WARPSYNC.COLLECTIVE.ALL waits for every live lane of the warp at any such gate (non-aligned barrier.sync slow path)",
        applies: |k| any(k, |o| matches!(o, Op::WarpSync { all: true, .. })),
    },
    Assumption {
        id: "E-BAR-EXIT",
        kind: "code",
        status: "validated",
        title: "bar.sync without count waits for live threads only; counted bar.sync counts arrivals",
        applies: |k| any(k, |o| matches!(o, Op::BarSync { .. })),
    },
    Assumption {
        id: "E-BAR-ARRIVE",
        kind: "code",
        status: "validated",
        title: "bar.arrive counts and releases without waiting; a re-arrival is a violation, counted",
        applies: |k| any(k, |o| matches!(o, Op::BarSync { arrive: true, .. })),
    },
    Assumption {
        id: "E-BAR-OVER",
        kind: "code",
        status: "validated",
        title: "a counted named barrier completes a generation every count arrivals; surplus arrivals start the next",
        applies: |k| any(k, |o| matches!(o, Op::BarSync { count: Some(_), .. })),
    },
    Assumption {
        id: "E-BAR-CUT",
        kind: "code",
        status: "argued",
        title: "a thread that reaches a counted barrier only through an all-thread barrier does not arrive in its open generation",
        applies: |k| {
            any(k, |o| matches!(o, Op::BarSync { count: Some(_), .. }))
                && any(k, |o| {
                    matches!(
                        o,
                        Op::BarSync {
                            count: None,
                            arrive: false,
                            ..
                        }
                    )
                })
        },
    },
    Assumption {
        id: "E-BAR-RED",
        kind: "code",
        status: "validated",
        title: "bar.red reduces the waiters' predicates of one barrier generation; each waiter reads the result",
        applies: |k| {
            any(k, |o| {
                matches!(o, Op::BarSync { red: Some(_), .. } | Op::BarResult { .. })
            })
        },
    },
    Assumption {
        id: "E-GRID",
        kind: "code",
        status: "argued",
        title: "exhaustive for the captured launch only (its grid, block and cluster shape)",
        applies: always,
    },
    Assumption {
        id: "E-RESIDENCY",
        kind: "code",
        status: "argued",
        title: "every CTA co-resident (default) or at most R per GPU, resident in any order as slots free (--residency)",
        // Kernels that wait on other CTAs (poll loops on global memory); added to every
        // verdict under limited residency (`for_machine`).
        applies: |k| !crate::residency::cross_cta_waits(k).is_empty(),
    },
    Assumption {
        id: "E-TMEM-COTENANT",
        kind: "code",
        status: "argued",
        title: "opt-in: a TMEM co-tenant holds any columns for a finite time; allocations land anywhere free; CTA pairs may share an SM",
        // Added to a verdict in co-tenant mode (`for_machine`, tenant::check).
        applies: never,
    },
    Assumption {
        id: "E-TMEM-PERMIT",
        kind: "code",
        status: "unvalidated",
        title: "co-tenant mode: a CTA's allocation permit (allocated, not relinquished, not exited) blocks its SM partner's allocations",
        applies: never,
    },
    Assumption {
        id: "E-CLUSTER-BAR",
        kind: "code",
        status: "argued",
        title: "barrier.cluster over live threads; relaxed arrive releases only fenced history / inits",
        applies: |k| {
            any(k, |o| {
                matches!(o, Op::ClusterArrive { .. } | Op::ClusterWait)
            })
        },
    },
    Assumption {
        id: "E-CLC",
        kind: "code",
        status: "validated",
        title: "CLC: CTAs launch in any order; try_cancel fails or cancels any unlaunched CTA",
        applies: |k| any(k, |o| matches!(o, Op::ClcTryCancel { .. })),
    },
    Assumption {
        id: "E-BULK-EXIT",
        kind: "code",
        status: "validated",
        title: "a CTA's shared memory outlives its exit until its bulk (TMA) stores have read it",
        applies: |k| any(k, |o| matches!(o, Op::TmaStore { .. })),
    },
    Assumption {
        id: "E-CPASYNC-EXIT",
        kind: "code",
        status: "validated",
        title: "a cp.async no thread waited for lands before the CTA's shared memory is reused",
        applies: |k| any(k, |o| matches!(o, Op::CpAsync { .. })),
    },
    Assumption {
        id: "E-TC-EXIT",
        kind: "code",
        status: "validated",
        title: "a tcgen05.commit no thread waited for arrives before the CTA's shared memory is released",
        applies: |k| any(k, |o| matches!(o, Op::TcCommit { .. })),
    },
    Assumption {
        id: "E-TC-FENCE-NOP",
        kind: "code",
        status: "argued",
        title: "a PTX race the tcgen05 thread-sync fences would order is a contract deviation: GB300 runs the fences as NOPs",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::TmemLd { .. }
                        | Op::TmemSt { .. }
                        | Op::Mma { .. }
                        | Op::TcCp { .. }
                        | Op::TmemAlloc { .. }
                )
            })
        },
    },
    Assumption {
        id: "E-CPASYNC-MBAR",
        kind: "code",
        status: "validated",
        title: "cp.async.mbarrier.arrive completes the thread's earlier cp.async writes (at their issue clocks), then one transaction unit (or arrival)",
        applies: |k| any(k, |o| matches!(o, Op::CpAsyncMbarArrive { .. })),
    },
    Assumption {
        id: "S-TMA-GATHER4",
        kind: "code",
        status: "validated",
        title: "tile::gather4: four rows at one column land as a 4-row box; the column must be 16-byte aligned",
        applies: |k| {
            any(
                k,
                |o| matches!(o, Op::TmaLoad { dims: 2, coords, .. } if coords.len() == 5),
            )
        },
    },
    Assumption {
        id: "E-DESC-VERSION",
        kind: "code",
        status: "validated",
        title: "tcgen05.mma ignores shared-memory descriptor bits 46-48 (version) on GB300: a value other than 0b001 is a contract finding",
        applies: never,
    },
    Assumption {
        id: "E-CPASYNC",
        kind: "code",
        status: "validated",
        title: "cp.async reads global at issue and writes shared memory when wait_group retires its group (an async actor at the issue clock; copies in flight together race)",
        applies: |k| any(k, |o| matches!(o, Op::CpAsync { .. })),
    },
    Assumption {
        id: "E-PDL",
        kind: "code",
        status: "validated",
        title: "multi-kernel launch: later grids wait for the previous grid's trigger (PDL) or completion",
        applies: |k| {
            !k.parts.is_empty() || any(k, |o| matches!(o, Op::GridDepWait | Op::GridDepLaunch))
        },
    },
    Assumption {
        id: "E-ASYNC-STEP",
        kind: "code",
        status: "argued",
        title: "a TMA box / store / MMA tile / commit completes in one step; wait_group.read orders only the bulk group's shared-memory reads (PTX)",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::TmaLoad { .. } | Op::TmaStore { .. } | Op::Mma { .. } | Op::TcCommit { .. }
                )
            })
        },
    },
    Assumption {
        id: "E-NANOSLEEP",
        kind: "code",
        status: "validated",
        title: "nanosleep is a no-op",
        applies: |k| {
            k.code
                .iter()
                .any(|i| i.text.to_lowercase().contains("nanosleep"))
        },
    },
    Assumption {
        id: "E-HW-SPLIT-UR",
        kind: "code",
        status: "validated",
        title: "GB300 profile: no switch between a warp's splits inside a uniform-register window",
        // Added to a verdict when the launch sets `hw_split_ur` (the GB300 profile) and the
        // kernel has a window.
        applies: never,
    },
    Assumption {
        id: "E-SETMAXNREG",
        kind: "code",
        status: "argued",
        title: "setmaxnreg / USETMAXREG is a no-op and an increase always completes (GB300 hangs when the CTA's register pool cannot satisfy it)",
        applies: |k| {
            k.code.iter().any(|i| {
                let t = i.text.to_lowercase();
                t.contains("setmaxnreg") || t.contains("usetmaxreg")
            })
        },
    },
    Assumption {
        id: "E-RET",
        kind: "code",
        status: "enforced",
        title: "a SASS RET lands right after a CALL (control-flow graph of the reductions); checked on every return",
        applies: |k| any(k, |o| matches!(o, Op::Ret { .. })),
    },
    Assumption {
        id: "E-CONST-BRA",
        kind: "code",
        status: "enforced",
        title: "branches decided by launch constants (constant propagation) go that way; checked on every execution",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::Bra {
                        any: false,
                        tid0_falls: false,
                        cond: None,
                        ..
                    }
                )
            })
        },
    },
    Assumption {
        id: "S-MBAR",
        kind: "code",
        status: "validated",
        title: "mbarrier phase / parity / tx-count semantics",
        applies: mbar,
    },
    Assumption {
        id: "S-MBAR-SCOPE",
        kind: "code",
        status: "argued",
        title: "an mbarrier wait synchronizes only with the phase's updates it is morally strong with; SASS SYNCS / PHASECHK scopes from ptxas's lowering",
        applies: |k| cluster(k) && mbar(k),
    },
    Assumption {
        id: "S-MBAR-MEM",
        kind: "code",
        status: "validated",
        title: "an mbarrier can be initialized by an STS.64 of its encoded state",
        applies: |k| sass(k) && mbar(k),
    },
    Assumption {
        id: "S-TCGEN05",
        kind: "code",
        status: "argued",
        title: "tcgen05 ordering: ld view advances at fence::after; MMA/commit in-order pipe",
        applies: |k| !sass(k) && tmem(k),
    },
    Assumption {
        id: "S-TC-COLLECTOR",
        kind: "code",
        status: "validated",
        title: "tcgen05.mma .collector::a use / lastuse read A from memory and require a still-valid fill of the same A (else a contract finding)",
        applies: |k| {
            any(
                k,
                |o| matches!(o, Op::Mma { coll, .. } if *coll != ptxv_ir::MmaCollector::Discard),
            )
        },
    },
    Assumption {
        id: "S-TMA",
        kind: "code",
        status: "validated",
        title: "TMA load/store: N-D box, none/32/64/128B swizzle, OOB zero fill (OOB unvalidated)",
        applies: |k| any(k, |o| matches!(o, Op::TmaLoad { .. } | Op::TmaStore { .. })),
    },
    Assumption {
        id: "S-TMA-2CTA-MCAST",
        kind: "code",
        status: "unvalidated",
        title: "a multicast 2-CTA TMA load completes each destination's tx on its pair's mbarrier (peer bit of the operand)",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::TmaLoad {
                        mask: Some(_),
                        pair: true,
                        ..
                    }
                )
            })
        },
    },
    Assumption {
        id: "S-PROXY",
        kind: "code",
        status: "validated",
        title: "generic smem writes reach the async proxy only via a fence.proxy.async that the write HB and that HB the async op (any thread's: probes/proxy_fence)",
        applies: |k| any(k, |o| matches!(o, Op::TmaStore { .. } | Op::Mma { .. })),
    },
    Assumption {
        id: "S-PROXY-WAR",
        kind: "code",
        status: "argued",
        title: "strict level: a generic smem read before an async-proxy write of the same bytes (a consumer releasing a TMA stage) needs a fence.proxy.async that the read HB and that HB the async op (PTX ISA 8.9.5); without one a contract finding (proxy-war), discharged under the GB300 profile (E-HW-PROXY-WAR)",
        applies: |k| any(k, |o| matches!(o, Op::TmaLoad { .. })),
    },
    Assumption {
        id: "E-TMEM-QUEUE",
        kind: "code",
        status: "measured, non-discriminating",
        title: "A thread's STTM (tcgen05.st) is ordered after its own earlier LDTMs, waited or not, and after its own earlier STTMs to the same words, fenced or not (one in-order TMEM queue): not assumed at any level; the strict level and the GB300 profile report such orders as tmem-queue contract findings (probes/tmem_queue: measured nulls, no discriminating control)",
        // No verdict relies on it: the orders it names are contract findings at both levels.
        applies: never,
    },
    Assumption {
        id: "E-HW-PROXY-WAR",
        kind: "code",
        status: "probed",
        title: "GB300 profile: a generic smem read that happens-before an async-proxy write of the same bytes is ordered before it without a proxy fence (probes/handoff_window v=16/17: 0 under contention, positive control v=15); discharges proxy-war contract findings",
        // Added to a verdict by the GB300 profile when it discharged a `proxy-war:` finding.
        applies: never,
    },
    Assumption {
        id: "S-GPROXY",
        kind: "code",
        status: "argued",
        title: "an async-proxy (TMA / bulk) access of global memory is ordered after a generic access, or another CTA's async access, only through a global proxy fence (fence.proxy.async[.global]) on the happens-before path",
        applies: |k| any(k, |o| matches!(o, Op::TmaLoad { .. } | Op::TmaStore { .. })),
    },
    Assumption {
        id: "S-MULTIMEM",
        kind: "code",
        status: "argued",
        title: "a multimem access (multimem.* / LDGMC, and STG / REDG on a multicast address) is one access per bound GPU with the instruction's semantics, all at once",
        // Added to a verdict when the case has multicast objects (for_machine).
        applies: never,
    },
    Assumption {
        id: "S-VA-ALIAS",
        kind: "code",
        status: "argued",
        title: "accesses through different virtual mappings of the same bytes (multicast / unicast, a second cuMemMap, a peer's import) are ordered only by a fence.proxy.alias or a kernel boundary on the happens-before path (PTX alias proxy); mappings used from different address spaces are undecided by default",
        applies: never,
    },
    Assumption {
        id: "E-CPASYNC-PROXY",
        kind: "code",
        status: "unvalidated",
        title: "cp.async writes reach a later async-proxy read (TMA store, tcgen05.mma) without fence.proxy.async; reported as a contract finding",
        applies: |k| {
            any(k, |o| matches!(o, Op::CpAsync { .. }))
                && any(k, |o| matches!(o, Op::TmaStore { .. } | Op::Mma { .. }))
        },
    },
    Assumption {
        id: "S-INIT",
        kind: "code",
        status: "enforced",
        title: "unwritten registers / smem are undefined; undefined values propagate through register-only operations and are reported where they reach a sink (memory, address, control flow, synchronization)",
        applies: always,
    },
    Assumption {
        id: "S-INIT-PRED",
        kind: "code",
        status: "enforced",
        title: "predicates start undefined (PT/UPT are constants) and are undefined when computed from undefined values; reported at sinks",
        applies: always,
    },
    Assumption {
        id: "E-WARPID",
        kind: "code",
        status: "validated",
        title: "%warpid is the warp's SM slot, not determined by the program: its value is undefined (S-INIT reports uses at sinks)",
        applies: |k| any(k, ptxv_ir::reads_warpid),
    },
    Assumption {
        id: "S-DSMEM",
        kind: "code",
        status: "validated",
        title: "shared address = (cluster rank << 24) | offset; shared::cluster selects the CTA",
        applies: cluster,
    },
    Assumption {
        id: "S-SWIN",
        kind: "code",
        status: "validated",
        title: "generic shared window: (SR_SWINHI << 32) | shared::cluster address",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::Mov {
                        a: ptxv_ir::Val::SReg(ptxv_ir::SReg::SharedWindowHi),
                        ..
                    } | Op::Cvta {
                        space: ptxv_ir::Space::Shared | ptxv_ir::Space::SharedCluster,
                        ..
                    }
                )
            })
        },
    },
    Assumption {
        id: "E-PAIR",
        kind: "code",
        status: "validated",
        title: "cta_group::2: per-CTA alloc; pair MMA splits M by rows and N in halves, each CTA's own scale factors; pair tcgen05.cp: each CTA's smem to its own TMEM",
        applies: |k| {
            any(k, |o| {
                matches!(o, Op::Mma { pair: true, .. } | Op::TcCp { pair: true, .. })
            })
        },
    },
    Assumption {
        id: "S-RESERVED",
        kind: "code",
        status: "validated",
        title: "reserved smem: 0, except the driver's cta_group::2 handshake state (0x48 mbarrier, 0x50 phase)",
        applies: |k| {
            k.front == Front::Sass
                && any(k, |o| {
                    matches!(
                        o,
                        Op::TmemAllocRaw { pair: true, .. } | Op::Mma { pair: true, .. }
                    )
                })
        },
    },
    Assumption {
        id: "S-RED-ASYNC",
        kind: "code",
        status: "probed",
        title: "red.async.global.add / UREDGR: a completed red that releases at its scope (probes/redasync_litmus)",
        applies: |k| {
            k.code
                .iter()
                .any(|i| i.text.contains("red.async") || i.text.contains("UREDGR"))
        },
    },
    Assumption {
        id: "S-SMEM-STRONG",
        kind: "code",
        status: "validated",
        title: "SASS LDS/STS are relaxed .cta strong; an LDS waited on before the next memory instruction acquires",
        applies: |k| {
            k.front == ptxv_ir::Front::Sass
                && any(k, |o| {
                    matches!(
                        o,
                        Op::Ld {
                            space: ptxv_ir::Space::Shared,
                            ..
                        } | Op::St {
                            space: ptxv_ir::Space::Shared,
                            ..
                        }
                    )
                })
        },
    },
    Assumption {
        id: "S-SCOPE",
        kind: "code",
        status: "validated",
        title: "strong accesses and fences synchronize by PTX scope rules",
        applies: |k| {
            cluster(k)
                || any(k, |o| {
                    matches!(
                        o,
                        Op::Ld { sync: Some(_), .. }
                            | Op::St { sync: Some(_), .. }
                            | Op::Ld256 { sync: Some(_), .. }
                            | Op::St256 { sync: Some(_), .. }
                            | Op::Atom { .. }
                            | Op::Fence {
                                kind: FenceKind::Membar(_)
                                    | FenceKind::Release(_)
                                    | FenceKind::Acquire
                                    | FenceKind::AcquireAt(_)
                            }
                    )
                })
        },
    },
    Assumption {
        id: "S-CLUSTER-LIFE",
        kind: "code",
        status: "argued",
        title: "remote smem accesses must happen-before some exit of the target CTA, or land on an mbarrier phase it waited on",
        applies: cluster,
    },
    Assumption {
        id: "S-NIC-DB",
        kind: "code",
        status: "argued",
        title: "an IBGDA doorbell is a strong .sys store; WQE and payload stores must be released at GPU scope or wider before it",
        applies: never,
    },
    Assumption {
        id: "S-NIC-ORDER",
        kind: "code",
        status: "validated",
        title: "a QP's WQEs execute one at a time in posting order; QPs are unordered with each other; only IBTA ordering",
        applies: never,
    },
    Assumption {
        id: "S-NIC-SIGNAL",
        kind: "code",
        status: "validated",
        title: "an RDMA atomic is placed after the same QP's earlier RDMA writes and releases them",
        applies: never,
    },
    Assumption {
        id: "S-NIC-WRITE",
        kind: "code",
        status: "argued",
        title: "RDMA write data lands as plain writes of the NIC: nothing synchronizes with it directly",
        applies: never,
    },
    Assumption {
        id: "S-NIC-WORD",
        kind: "code",
        status: "validated",
        title: "each naturally aligned 8-byte word of RDMA write data lands single-copy atomic (a relaxed .sys write)",
        applies: never,
    },
    Assumption {
        id: "S-NIC-FLAG",
        kind: "code",
        status: "validated",
        title: "a lone aligned 4- or 8-byte RDMA write is one strong .sys write that releases the QP's earlier WQEs",
        applies: never,
    },
    Assumption {
        id: "S-NIC-ATOMIC",
        kind: "code",
        status: "validated",
        title: "NIC atomics: strong .sys RMW, little-endian 64-bit values, atomic only with other NIC atomics",
        applies: never,
    },
    Assumption {
        id: "S-NIC-CQE",
        kind: "code",
        status: "validated",
        title: "a CQE is written after its WQE completed (RDMA write: placed at the peer) and releases what the WQE did",
        applies: never,
    },
    Assumption {
        id: "S-NIC-DOMAIN",
        kind: "code",
        status: "argued",
        title: "GPUs in different NVLink domains reach each other's memory only through the NIC",
        applies: never,
    },
    Assumption {
        id: "S-REMOTE-INFLIGHT",
        kind: "code",
        status: "argued",
        title: "remote mbarrier arrivals and st.async are in flight until delivered, unordered, drained by the issuer's cluster-scope releases",
        applies: |k| {
            any(k, |o| {
                matches!(o, Op::MbarArrive { remote: true, .. } | Op::StAsync { .. })
            })
        },
    },
    Assumption {
        id: "S-ALIGN",
        kind: "code",
        status: "validated",
        title: "ld/st/atom of n bytes (power of two) must be n-aligned, else a fault",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::Ld { .. }
                        | Op::St { .. }
                        | Op::Ld256 { .. }
                        | Op::St256 { .. }
                        | Op::Atom { .. }
                )
            })
        },
    },
    Assumption {
        id: "S-GLOBAL",
        kind: "code",
        status: "argued",
        title: "global memory is touched only by this kernel",
        applies: always,
    },
    Assumption {
        id: "S-GLOBAL-INIT",
        kind: "code",
        status: "argued",
        title: "a module global without a case buffer starts from its ELF initial value",
        applies: |k| !k.globals.is_empty() || k.parts.iter().any(|p| !p.globals.is_empty()),
    },
    Assumption {
        id: "S-CBANK",
        kind: "code",
        status: "validated",
        title: "driver constant bank layout (dims, param base, stack)",
        applies: sass,
    },
    Assumption {
        id: "S-SAME-VALUE",
        kind: "code",
        status: "validated",
        title: "unordered plain writes of the byte value already there conflict with nothing",
        applies: |k| any(k, |o| matches!(o, Op::St { .. })),
    },
    Assumption {
        id: "S-SMEM-ADDR32",
        kind: "code",
        status: "validated",
        title: "shared-memory addresses are 32-bit: address operand parts add modulo 2^32",
        applies: |k| {
            mbar(k)
                || any(k, |o| {
                    use ptxv_ir::Space::Shared;
                    use ptxv_ir::Space::SharedCluster;
                    matches!(
                        o,
                        Op::Ld {
                            space: Shared | SharedCluster,
                            ..
                        } | Op::St {
                            space: Shared | SharedCluster,
                            ..
                        } | Op::Atom {
                            space: Shared | SharedCluster,
                            ..
                        }
                    )
                })
        },
    },
    Assumption {
        id: "S-CGA-SHAPE",
        kind: "code",
        status: "validated",
        title: "SR_CgaSize is the cluster's shape index (0 regular, 1 preferred); preferred launches mix shapes per CTA group",
        applies: |k| {
            k.code.iter().any(|i| {
                i.op.src_vals()
                    .iter()
                    .any(|v| matches!(v, ptxv_ir::Val::SReg(ptxv_ir::SReg::CgaShape)))
            })
        },
    },
    Assumption {
        id: "E-SMID",
        kind: "exec",
        status: "unvalidated",
        title: "%smid: fixed per CTA, in [0, %nsmid = 152); verdicts cover the enumerated SM placements (PTXV_SMID), not every assignment",
        applies: |k| {
            crate::smid::reads_smid(k)
                || any(k, |o| {
                    o.src_vals().iter().any(|v| {
                        matches!(
                            v,
                            ptxv_ir::Val::SReg(ptxv_ir::SReg::NSmId)
                                | ptxv_ir::Val::Const(0x2d0, _)
                        )
                    })
                })
        },
    },
    Assumption {
        id: "E-BSYNC",
        kind: "exec",
        status: "argued",
        title: "BSYNC is a warp rendezvous of the lanes its BSSY recorded, ordering their memory",
        applies: |k| any(k, |o| matches!(o, Op::Bsync { .. })),
    },
    Assumption {
        id: "S-TMAP",
        kind: "code",
        status: "validated",
        title: "tensor maps: the reverse-engineered CUtensorMap layout (device-built maps, and captured parameter maps' geometry); fence.proxy.tensormap per using thread",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::TmapFence { .. } | Op::TmaLoad { .. } | Op::TmaStore { .. }
                )
            })
        },
    },
    Assumption {
        id: "S-NC",
        kind: "code",
        status: "argued",
        title: "a non-coherent load (ld.global.nc, LDG .CONSTANT) is exact for bytes nothing writes while its kernel runs; a write of a byte it reads during the kernel is reported",
        applies: |k| {
            any(k, |o| {
                matches!(o, Op::Ld { nc: true, .. } | Op::Ld256 { nc: true, .. })
            })
        },
    },
    Assumption {
        id: "E-ELECT-2POL",
        kind: "code",
        status: "argued",
        title: "PTX elect explored with the lowest and highest active lane only, where only the strict leader-sensitivity gate fires (lane-varying synchronization, a call or indexed branch; machine-level, for_machine)",
        applies: |_| false,
    },
    Assumption {
        id: "E-UNI",
        kind: "code",
        status: "argued",
        title: "a .uni branch whose executing lanes disagree on its guard or index is a contract finding",
        applies: |k| any(k, |o| matches!(o, Op::Uniform { .. })),
    },
    Assumption {
        id: "S-BRA-COND",
        kind: "code",
        status: "argued",
        title: "`@P0 BRA P1, target` is taken where the guard and P1 both hold",
        applies: |k| any(k, |o| matches!(o, Op::Bra { cond: Some(_), .. })),
    },
    Assumption {
        id: "S-TMEM",
        kind: "code",
        status: "validated",
        title: "TMEM: per CTA, 128x512, 32-column units; one CTA per SM",
        applies: tmem,
    },
    Assumption {
        id: "S-TCCP",
        kind: "code",
        status: "validated",
        title: "tcgen05.cp 32x128b.warpx4: 32 x 16 B smem rows -> 4 TMEM columns, row r at lane 32w + r of every sub-partition (r3_cute_mxf8; pair: probes/tc_pair_bs/cp)",
        applies: |k| any(k, |o| matches!(o, Op::TcCp { .. })),
    },
    Assumption {
        id: "L-ALIAS-FENCE",
        kind: "code",
        status: "validated",
        title: "SASS: a MEMBAR.SC inside ptxas's UP2UR.B3 / UR2UP bracket is fence.proxy.alias; driver word c[0x0][0x2d8] = 0 (GB300 probe)",
        applies: never,
    },
    Assumption {
        id: "L-NVDISASM",
        kind: "code",
        status: "trusted",
        title: "nvdisasm prints what the bits encode",
        applies: sass,
    },
    Assumption {
        id: "L-ROSETTA",
        kind: "code",
        status: "validated",
        title: "SASS lowering table (seen forms only; others are errors)",
        applies: sass,
    },
    Assumption {
        id: "L-RPCMOV",
        kind: "code",
        status: "validated",
        title: "RPCMOV.32 Rpc.LO: a per-thread slot only RPCMOV touches (probes/rpcmov)",
        applies: |k| k.code.iter().any(|i| i.text.contains("RPCMOV")),
    },
    Assumption {
        id: "L-PTX-HINT",
        kind: "code",
        status: "argued",
        title: "PTX cp.async .L2::cache_hint (and its policy operand) / .L2::{64,128,256}B are performance hints only",
        applies: |k| k.front == Front::Ptx && k.code.iter().any(|i| i.text.contains(".L2::")),
    },
    Assumption {
        id: "PTX-FMAD",
        kind: "code",
        status: "validated",
        title: "PTX mul + add / sub contracted as ptxas 13.0's default --fmad=true: without rounding modifiers, and .f32x2 of equal modes (probes/fmad, probes/fmad2); ambiguous shapes refused",
        applies: |k| {
            k.front == Front::Ptx
                && k.code.iter().any(|i| {
                    // add.f64 alone is a DFMA a * 1.0 + b (ptx lower_inst).
                    let one = ptxv_ir::Val::Imm(1f64.to_bits());
                    let t = i.text.trim_start();
                    (matches!(i.op, Op::Ffma { c: Some(_), .. })
                        || matches!(&i.op, Op::Dfma { c: Some(_), b, .. } if *b != one)
                        // add / sub.f32x2 alone is FFMA2 a * 1.0 + b; contracted: FFMA2 a * b + c.
                        || matches!(&i.op, Op::Ffma2 { b, .. }
                            if b.0 != ptxv_ir::Val::Imm(0x3f80_0000)))
                        && (t.starts_with("add.") || t.starts_with("sub."))
                })
        },
    },
    Assumption {
        id: "PTX-FMAD-X2-VERSION",
        kind: "code",
        status: "validated",
        title: "PTX .f32x2 contraction follows the case's ptxas version: 13.0 any equal modes, 13.1 / 13.3 plain pairs only; unknown versions refuse where they differ (probes/fmad2_v133)",
        applies: |k| {
            k.front == Front::Ptx
                && k.code.iter().any(|i| {
                    let t = i.text.trim_start();
                    matches!(&i.op, Op::Ffma2 { b, .. } if b.0 != ptxv_ir::Val::Imm(0x3f80_0000))
                        && (t.starts_with("add.") || t.starts_with("sub."))
                })
        },
    },
    Assumption {
        id: "PTX-FMUL-SCALE",
        kind: "code",
        status: "validated",
        title: "a mul.f32 without a rounding modifier read only by a mul by 0.125 .. 8 is one FMUL.D8 .. .M8 (one rounding of the scaled product) as ptxas 13.0 / 13.1 / 13.3 build it; other candidate shapes and versions refuse (probes/fmul_scale, MISS 55)",
        applies: |k| k.front == Front::Ptx && any(k, |o| matches!(o, Op::FmulScale { .. })),
    },
    Assumption {
        id: "L-LOP3-P",
        kind: "code",
        status: "validated",
        title: "LOP3 predicate operand: Pd = (r != 0) || Pq, or && Pq with .PAND (probes/lop3_pmode)",
        applies: |k| any(k, |o| matches!(o, Op::Lop3 { pq: Some(_), .. })),
    },
    Assumption {
        id: "L-SHF",
        kind: "code",
        status: "validated",
        title: "SHF / shf / shl / shr: u32 amount clamped to 32 (.U32/.S32) or 64 (.U64/.S64), mod 32 with .W (fixtures/s_shf)",
        applies: |k| any(k, |o| matches!(o, Op::Shf { .. })),
    },
    Assumption {
        id: "L-CTRL",
        kind: "code",
        status: "validated",
        title: "control word layout hi>>41 (stall/yield/wbar/rbar/wait/reuse)",
        applies: sass,
    },
    Assumption {
        id: "L-SB-ORDER",
        kind: "code",
        status: "validated",
        title: "a memory-pipe read scoreboard covers every earlier memory-pipe operand read (probes/sb_mio_order)",
        applies: sass,
    },
    Assumption {
        id: "L-OPREAD",
        kind: "code",
        status: "validated",
        title: "operand-read timing rules (memory ops late + in order, UR at issue)",
        applies: sass,
    },
    Assumption {
        id: "L-STALL",
        kind: "code",
        status: "trusted",
        title: "fixed-latency hazards are covered by stall counts (unchecked)",
        applies: sass,
    },
    Assumption {
        id: "L-MEMSB",
        kind: "code",
        status: "trusted",
        title: "memory-ordering scoreboards (fence completion) are trusted",
        applies: sass,
    },
    Assumption {
        id: "L-OPCLASS",
        kind: "code",
        status: "validated",
        title: "variable-latency / async-reader opcode sets are complete",
        applies: sass,
    },
    Assumption {
        id: "N-FADD",
        kind: "code",
        status: "validated",
        title: "FADD is IEEE RN, no FTZ, canonical NaN",
        applies: |k| any(k, |o| matches!(o, Op::Fadd { .. })),
    },
    Assumption {
        id: "N-F2I-F64",
        kind: "code",
        status: "validated",
        title: "F2I.*.F64 / cvt.r*i.*.f64: round, saturate, NaN 2^31 / 2^63 (fixtures/s_f2i_f64)",
        applies: |k| any(k, |o| matches!(o, Op::F2I { f64: true, .. })),
    },
    Assumption {
        id: "N-FALU",
        kind: "code",
        status: "validated",
        title: "f32 ALU forms (FADD/FMUL/FFMA[2] all modes, FTZ, SAT, FMNMX, FSETP) = IEEE + measured FTZ/NaN rules",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::Fadd { .. }
                        | Op::Ffma { .. }
                        | Op::FmulScale { .. }
                        | Op::Ffma2 { .. }
                        | Op::Fmnmx { .. }
                        | Op::Fsetp { .. }
                        | Op::Hsetp2Bf16 { .. }
                        | Op::Fset { .. }
                )
            })
        },
    },
    Assumption {
        id: "N-FCHK",
        kind: "code",
        status: "validated",
        title: "FCHK = exponent-range test measured on GB300 (probes/fchk)",
        applies: |k| any(k, |o| matches!(o, Op::Fchk { .. })),
    },
    Assumption {
        id: "N-BF16X2",
        kind: "code",
        status: "validated",
        title: "packed / mixed bf16 arithmetic and bf16 atomics: one rounding, NaN 0x7fff (probes/bf16x2, bf16_ext)",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::Hfma2Bf16 { .. }
                        | Op::FfmaBf16 { f16: false, .. }
                        | Op::Hmnmx2Bf16 { .. }
                        | Op::Vhmnmx { .. }
                        | Op::Atom {
                            op: ptxv_ir::AtomOp::AddBf16x2 | ptxv_ir::AtomOp::AddBf16,
                            ..
                        }
                )
            })
        },
    },
    Assumption {
        id: "N-QMUL4",
        kind: "code",
        status: "validated",
        title: "QMUL4.E4M3x4.E2M1x4.E4M3x4: e2m1 x e4m3 per byte, rounded to e4m3 (nearest even), overflow / NaN 0x7f (probes/qmul4)",
        applies: |k| any(k, |o| matches!(o, Op::Qmul4 { .. })),
    },
    Assumption {
        id: "N-F2FP-RZ",
        kind: "code",
        status: "validated",
        title: "F2FP.F16.F32.PACK_AB.RZ: f32 -> f16 toward zero, overflow to 0x7bff, NaN 0x7fff (probes/f2fp_rz, every f32 per lane)",
        applies: |k| any(k, |o| matches!(o, Op::F2fpF16 { rz: true, .. })),
    },
    Assumption {
        id: "N-CVT-F16",
        kind: "code",
        status: "validated",
        title: "cvt f32 <-> f16 / bf16 (.rn, .sat, packed .satfinite / .relu, the widening forms) = GB300; other modifier lists refused (probes/cvt_f16)",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::F2fpF16 { rz: false, .. }
                        | Op::F2fpBf16 { .. }
                        | Op::CvtF32F16 { .. }
                        | Op::CvtF32Bf16 { .. }
                )
            })
        },
    },
    Assumption {
        id: "N-FHADD",
        kind: "code",
        status: "validated",
        title: "FHADD / FHFMA without .BF16 (add / sub / fma .f32.f16): f16 halves widened exactly, one f32 rounding in the mode (probes/fhadd)",
        applies: |k| any(k, |o| matches!(o, Op::FfmaBf16 { f16: true, .. })),
    },
    Assumption {
        id: "N-HMNMX2",
        kind: "code",
        status: "validated",
        title: "HMNMX2 (f16x2 / f16 min / max [.NaN] [.xorsign.abs]): FMNMX's rules per f16 lane, NaN 0x7fff (probes/hmnmx2)",
        applies: |k| any(k, |o| matches!(o, Op::Hmnmx2F16 { .. })),
    },
    Assumption {
        id: "N-XORSIGN",
        kind: "code",
        status: "validated",
        title: "FMNMX.XORSIGN / HMNMX2.XORSIGN: min/max of |a|, |b| with sign(a) XOR sign(b) (probes/fmnmx_xorsign)",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::Fmnmx { xorsign: true, .. }
                        | Op::Hmnmx2Bf16 { xorsign: true, .. }
                        | Op::Hmnmx2F16 { xorsign: true, .. }
                )
            })
        },
    },
    Assumption {
        id: "N-FCVT",
        kind: "code",
        status: "validated",
        title: "FRND / F2F between f64, f32, f16, bf16: IEEE rounding, NaN payload rules (probes/fcvt)",
        applies: |k| any(k, |o| matches!(o, Op::FCvt { .. })),
    },
    Assumption {
        id: "N-F2I16",
        kind: "code",
        status: "validated",
        title: "F2I.S16 / .U16: round, saturate to 16 bits, NaN 0, extended in the register (probes/f2i16)",
        applies: |k| any(k, |o| matches!(o, Op::F2I { bits: 16, .. })),
    },
    Assumption {
        id: "N-INT16X2",
        kind: "code",
        status: "validated",
        title: "VIMNMX / VIMNMX3 .U16x2 / .S16x2: per 16-bit lane min / max (probes/int16x2)",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::IntMnmx3 { x16: true, .. }
                        | Op::IntMnmx3 { relu: true, .. }
                        | Op::AddMinMax { relu: true, .. }
                )
            })
        },
    },
    Assumption {
        id: "N-BULK-RED",
        kind: "code",
        status: "validated",
        title: "UBLKRED.G.S.ADD.F32.RN: per-word atomic f32 add, RN, no flush, canonical NaN (probes/bulk_red)",
        applies: |k| any(k, |o| matches!(o, Op::TmaStore { red: true, .. })),
    },
    Assumption {
        id: "N-RED16",
        kind: "code",
        status: "validated",
        title: "unordered f16 / bf16 bulk reductions into one element: some sequential order, one 16-bit rounding per add; nondeterministic, bounded over every order (probes/tma_red_order)",
        applies: |k| {
            any(
                k,
                |o| matches!(o, Op::TmaStore { red: true, dims, .. } if *dims > 0),
            )
        },
    },
    Assumption {
        id: "N-MUFU",
        kind: "code",
        status: "validated",
        title: "MUFU / PTX approx ops = the units reconstructed from exhaustive GB300 sweeps",
        applies: |k| any(k, |o| matches!(o, Op::Mufu { .. })),
    },
    Assumption {
        id: "N-EQUIV",
        kind: "code",
        status: "argued",
        title: "equivalence: one tainted-clean run covers every input; canonical DAGs round alike",
        // Applies to `ptxv equiv` results, not to check verdicts.
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-IID",
        kind: "code",
        status: "argued",
        title: "errdist: the input law is the spec's (elements independent given their scales)",
        // Applies to `ptxv errdist` estimates, not to check verdicts.
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-TEMPLATE",
        kind: "code",
        status: "argued",
        title: "errdist tiers 0-2 analyze output 0's chain as the template for every output",
        // Applies to `ptxv errdist` estimates, not to check verdicts.
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-RESIDUE",
        kind: "code",
        status: "argued",
        title: "errdist tier 1: an accumulator's bits below its result's ulp follow the previous terms' residue law",
        // Applies to `ptxv errdist` estimates, not to check verdicts.
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-MARKOV",
        kind: "code",
        status: "argued",
        title: "errdist tier 1: the chain is Markov in the partial sums' (sign, binade) and 2 tracked low bits",
        // Applies to `ptxv errdist` estimates, not to check verdicts.
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-CF",
        kind: "code",
        status: "argued",
        title: "errdist tier 1: final cell masses from the exact characteristic function, conditional error laws from the chain",
        // Applies to `ptxv errdist` estimates, not to check verdicts.
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-CLT",
        kind: "code",
        status: "argued",
        title: "errdist tier 2: normal node values, lattice-aware node errors independent given y",
        // Applies to `ptxv errdist` estimates, not to check verdicts.
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-DAG",
        kind: "code",
        status: "argued",
        title: "errdist exact value of DAG outputs: TF32 hi(x) stands for x, atomics in the simulator's order",
        // Applies to `ptxv errdist` estimates, not to check verdicts.
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-PAIR",
        kind: "code",
        status: "argued",
        title: "errdist pairs: deviations independent given the exact y's position in its ulp (shared sub-DAGs shared)",
        // Applies to `ptxv errdist` estimates, not to check verdicts.
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-FLIP",
        kind: "code",
        status: "argued",
        title: "errdist: a 16-bit output flips with probability min(1, |D| / ulp16), y's position on the 16-bit grid uniform",
        // Applies to `ptxv errdist` estimates, not to check verdicts.
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-MC",
        kind: "code",
        status: "argued",
        title: "errdist tier 3: per-sample exact; output-level CIs treat outputs as independent",
        // Applies to `ptxv errdist` estimates, not to check verdicts.
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-BSCLT",
        kind: "code",
        status: "argued",
        title: "errdist tier 2 on block-scaled chains: normal partial sums given y and its lattice class",
        // Applies to `ptxv errdist` estimates, not to check verdicts.
        applies: |_| false,
    },
    // Agent AG: errdist for non-chain programs (softmax, norms, attention, elementwise).
    Assumption {
        id: "N-ERRDIST-IDENT",
        kind: "code",
        status: "argued",
        title: "errdist: an instruction slice computes the real function it was fitted to",
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-REAL",
        kind: "code",
        status: "argued",
        title: "errdist: the exact value of a non-chain output is its algorithm in real arithmetic",
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-NL-LIN",
        kind: "code",
        status: "argued",
        title: "errdist non-chain linearized tier: first-order propagation of local errors at nominal inputs",
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-IS",
        kind: "code",
        status: "argued",
        title: "errdist --is: importance-sampled Tier 3 is unbiased (closed-form weights of the drawn values, defensive mixture, gated)",
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-NL-FTRAJ",
        kind: "code",
        status: "argued",
        title: "errdist non-chain linearized tier (opt-in): linearize around the computed float trajectory",
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-T2-OPEN",
        kind: "code",
        status: "argued",
        title: "errdist linearized tiers: an exp2 / exp slice opened into argument rounding, unit error and constant",
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-NL-INDEP",
        kind: "code",
        status: "argued",
        title: "errdist non-chain: local errors independent given their cells; tables from probes",
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-NL-JOINT",
        kind: "code",
        status: "argued",
        title: "errdist non-chain pairs: matched roundings share their grid position",
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-NL-ELEM",
        kind: "code",
        status: "argued",
        title: "errdist elementwise law: exact on 16-bit codes, stratified quadrature for f32",
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-INTFIX",
        kind: "code",
        status: "argued",
        title: "errdist holds integer input buffers at their captured values (not taint sources)",
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-FINCAP",
        kind: "code",
        status: "argued",
        title: "errdist extracts the DAG with non-finite captured float input elements set to 0 (the input distributions are finite)",
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-SEL",
        kind: "code",
        status: "argued",
        title: "a store under a data-dependent guard is a selection P ? v : old (re-evaluated per input by Tier 3; linearized / CLT tiers refused)",
        applies: |_| false,
    },
    Assumption {
        id: "N-ERRDIST-INERR",
        kind: "code",
        status: "argued",
        title: "errdist input errors and composition: per-element errors independent across elements, first order through the kernel",
        // Applies to `ptxv errdist` estimates, not to check verdicts.
        applies: |_| false,
    },
    Assumption {
        id: "N-LEAF",
        kind: "code",
        status: "argued",
        title: "input leaves keep their identity through loads, shuffles, moves and selects",
        // Applies to `ptxv equiv` results, not to check verdicts.
        applies: |_| false,
    },
    Assumption {
        id: "N-REASSOC",
        kind: "code",
        status: "argued",
        title: "equiv bound: rigorous Cor 1a per kernel, combined by Thm 4",
        applies: |_| false,
    },
    Assumption {
        id: "N-EXACT",
        kind: "code",
        status: "argued",
        title: "an FMUL of two narrow leaves, or a 16-bit rounding of one, is exact when the format holds every value the operands can take",
        applies: |_| false,
    },
    Assumption {
        id: "N-ATOMIC",
        kind: "code",
        status: "argued",
        title: "fp32 atomic add = one RN FTZ add; atomic adds commute up to their order",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::Atom {
                        op: ptxv_ir::AtomOp::FAddFtz,
                        ..
                    }
                )
            })
        },
    },
    Assumption {
        id: "N-SEG",
        kind: "code",
        status: "argued",
        title: "equiv segment mode: instruction slices re-execute the trace; div.rn = IEEE quotient",
        applies: |_| false,
    },
    Assumption {
        id: "N-ELEMENTWISE",
        kind: "code",
        status: "validated",
        title: "elementwise mode: a single-leaf output DAG is that element's function; swept over every input value",
        applies: |_| false,
    },
    Assumption {
        id: "N-HALF-LEAF",
        kind: "code",
        status: "argued",
        title: "a bf16 / f16 input element widened to f32 by byte moves is an exact Input leaf",
        applies: |_| false,
    },
    Assumption {
        id: "N-SOFTMAX",
        kind: "code",
        status: "argued",
        title: "equiv softmax bound from exhaustive slice contracts and the normal form",
        applies: |_| false,
    },
    Assumption {
        id: "N-NORM",
        kind: "code",
        status: "argued",
        title: "equiv norm bound: polynomial in the inputs and one swept power rho = A^p",
        applies: |_| false,
    },
    Assumption {
        id: "N-ATTN",
        kind: "code",
        status: "argued",
        title: "equiv attention bound: weighted normal form sum exp(L) v / sum exp(L)",
        applies: |_| false,
    },
    Assumption {
        id: "N-MULTIMEM",
        kind: "code",
        status: "validated",
        title: "multimem.ld_reduce arithmetic: integers exact; f32 add = window-truncated exact sum, round to nearest even; f16/bf16 min/max = minNum/maxNum (GB300 bitwise, probes/multimem)",
        applies: |k| any(k, |o| matches!(o, Op::LdReduce { .. })),
    },
    Assumption {
        id: "N-MULTIMEM-ROUND16",
        kind: "code",
        status: "unvalidated",
        title: "f16/bf16 multimem.ld_reduce add: the rounding direction of a non-representable sum follows a dither from the other lanes of its 8-byte group (99.7% bf16 / 96.7% f16 of random lanes); output bytes holding such a lane compared within one ulp",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::LdReduce {
                        op: ptxv_ir::McOp::Add,
                        ty: ptxv_ir::McTy::F16 | ptxv_ir::McTy::Bf16,
                        ..
                    }
                )
            })
        },
    },
    Assumption {
        id: "N-MMA",
        kind: "code",
        status: "enforced",
        title: "MMA arithmetic = g16-f25-rz-accin-tzero-einputs (finite normal inputs)",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::Mma {
                        kind: ptxv_ir::MmaKind::F16,
                        ..
                    }
                )
            })
        },
    },
    Assumption {
        id: "N-MMA-SPECIAL",
        kind: "code",
        status: "validated",
        title: "tcgen05.mma.kind::f16 special values: subnormal f16 / bf16 operands exact at the field minimum, NaN -> 0x7fffffff, Inf rules (probes/tc_special: 139,264 GB300 outputs)",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::Mma {
                        kind: ptxv_ir::MmaKind::F16,
                        ..
                    }
                )
            })
        },
    },
    Assumption {
        id: "N-HMMA",
        kind: "code",
        status: "validated",
        title: "HMMA.16816.F32[.BF16] = the N-MMA model per instruction (probes/hmma, probes/hmma_f16: 98,304 GB300 outputs each)",
        applies: |k| any(k, |o| matches!(o, Op::Hmma { s8: false, .. })),
    },
    Assumption {
        id: "N-IMMA",
        kind: "code",
        status: "validated",
        title: "IMMA.16832.S8.S8.SAT = exact integer C + A B saturated to s32 (probes/imma_s8, GB300 bitwise)",
        applies: |k| any(k, |o| matches!(o, Op::Hmma { s8: true, .. })),
    },
    Assumption {
        id: "N-TF32",
        kind: "code",
        status: "argued",
        title: "TF32 MMA operands: hi(x) within 2^-10 |x|; a 3xTF32 split's three products are a b (1 + theta), |theta| <= 2^-19 (equiv)",
        applies: |k| any(k, |o| matches!(o, Op::Hmma { tf32: true, .. })),
    },
    Assumption {
        id: "N-FINITE",
        kind: "code",
        status: "argued",
        title: "comparisons with an infinity / NaN of values finite inputs keep finite (no overflow) are input-independent (equiv precondition)",
        applies: |_| false,
    },
    Assumption {
        id: "N-FMAX",
        kind: "code",
        status: "argued",
        title: "CUTLASS's atomicMaxFloat branch (signed max / unsigned min by the sign bit) is one float max on either side (equiv precondition)",
        applies: |_| false,
    },
    Assumption {
        id: "N-ROUND",
        kind: "code",
        status: "argued",
        title: "F2FP.BF16 / bf16 output stores round to nearest even: relative 2^-8, 2^-134 absolute (equiv)",
        applies: |_| false,
    },
    Assumption {
        id: "N-MMA-F8",
        kind: "code",
        status: "validated",
        title: "kind::f8f6f4 = g32-f25-rz-accin-tzero-efields (f16 D: rn), K-major smem layout (probes/mma_f8)",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::Mma { kind, .. } if *kind != ptxv_ir::MmaKind::F16
                )
            })
        },
    },
    Assumption {
        id: "N-MMA-BS",
        kind: "code",
        status: "validated",
        title: "block-scaled kinds: mxf8f6f4 = N-MMA-F8 with scaled terms; mxf4 / mxf4nvf4 = u16-f35-live-sfields; SF TMEM layout (probes/mma_bs)",
        applies: |k| any(k, |o| matches!(o, Op::Mma { scale: Some(_), .. })),
    },
    Assumption {
        id: "N-MMA-K96",
        kind: "code",
        status: "validated",
        title: "kind::mxf4nvf4 block16 K = 96 (idesc bit 31): one 96-wide u16-f35-live-sfields step; 48-byte rows, two-word scale layout, absolute-LBO mode (probes/mma_k96)",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::Mma {
                        kind: ptxv_ir::MmaKind::Mxf4nvf4 { x4: true },
                        ..
                    }
                )
            })
        },
    },
    Assumption {
        id: "N-CVT-LP",
        kind: "code",
        status: "validated",
        title: "narrow-format cvt / F2FP (FP8/FP6/FP4/ue8m0, .rs packs) = lowp (probes/cvt_lp GB300 sweep)",
        applies: |k| {
            any(k, |o| {
                matches!(
                    o,
                    Op::CvtPack { .. }
                        | Op::CvtRsX4 { .. }
                        | Op::CvtUnpack { .. }
                        | Op::CvtS2f6x2 { .. }
                )
            })
        },
    },
    Assumption {
        id: "N-BOUND",
        kind: "code",
        status: "argued",
        title: "Cor 1a bound: first-order, per-node epsilon",
        applies: never,
    },
    Assumption {
        id: "N-NUM-TRACK",
        kind: "code",
        status: "enforced",
        title: "num: an element whose tree holds an input-derived value or a rounding without a tracked derivation is reported untracked, and a run whose control flow or addresses depend on input data makes no error claim",
        applies: never,
    },
    Assumption {
        id: "N-NUM-UDIV",
        kind: "code",
        status: "enforced",
        title: "num: the quotient and remainder of ptxas's 32-bit integer division by a run-time divisor (I2F.RP, MUFU.RCP, F2I.TRUNC, integer correction), matched structurally and checked per thread against n / d, are exact (rests on MUFU.RCP's measured accuracy: gb300)",
        applies: never,
    },
    Assumption {
        id: "N-NUM-NONFINITE",
        kind: "code",
        status: "enforced",
        title: "num: a non-finite simulated value against a finite exact value carries an infinite error, an identical non-finite value error 0, any other non-finite element is not analyzed; max_err skips no analyzed element",
        applies: never,
    },
    Assumption {
        id: "H-PROFILE",
        kind: "process",
        status: "argued",
        title: "two-level verdicts: a hardware profile turns on only probe-backed assumptions; the strict (contract) verdict is always reported next to the profile's SASS verdict",
        applies: never,
    },
    Assumption {
        id: "H-RUNS",
        kind: "process",
        status: "argued",
        title: "N clean hardware runs are evidence, not proof",
        applies: never,
    },
    Assumption {
        id: "H-SCRUB",
        kind: "process",
        status: "argued",
        title: "the scrubber reaches every SM's shared memory",
        applies: never,
    },
    Assumption {
        id: "H-DELAY",
        kind: "process",
        status: "argued",
        title: "witness delays recompile the kernel",
        applies: never,
    },
    Assumption {
        id: "H-GPU",
        kind: "process",
        status: "argued",
        title: "only GPUs without foreign processes; GB300 / CUDA 13.0",
        applies: never,
    },
];

/// IDs a verdict on `k` depends on, in registry order.
pub fn for_kernel(k: &Kernel) -> Vec<String> {
    REGISTRY
        .iter()
        .filter(|a| (a.applies)(k))
        .map(|a| a.id.to_string())
        .collect()
}

/// The kernel's assumptions plus those its launch opts into (E-EXIT-FENCE).
pub fn for_machine(m: &crate::machine::Machine) -> Vec<String> {
    let mut v = for_kernel(&m.k);
    // NIC queue pairs and NVLink domains (nic.rs, docs/nic.md).
    if let Some(n) = &m.launch.nic {
        if !n.qps.is_empty() {
            for id in [
                "S-NIC-DB",
                "S-NIC-ORDER",
                "S-NIC-SIGNAL",
                "S-NIC-WRITE",
                "S-NIC-WORD",
                "S-NIC-FLAG",
                "S-NIC-ATOMIC",
                "S-NIC-CQE",
            ] {
                v.push(id.into());
            }
        }
        if !n.domains.is_empty() {
            v.push("S-NIC-DOMAIN".into());
        }
    }
    if m.launch.exit_fence.is_some() {
        v.push("E-EXIT-FENCE".into());
    }
    // A declared dead workspace downgrades undefined data left in it to a note.
    if m.launch.allocs.iter().any(|a| a.dead) {
        v.push("S-INIT-DEAD".into());
    }
    if crate::explore::elect_two_policy_note(m).is_some() {
        v.push("E-ELECT-2POL".into());
    }
    if m.launch.poll_acquire {
        v.push("E-POLL-ACQ".into());
    }
    if m.launch.smem_cumulative {
        v.push("E-SMEM-CUMUL".into());
    }
    if m.launch.hw_l1 {
        v.push("E-HW-L1".into());
        if m.launch.l1_havoc {
            v.push("E-L1-HAVOC".into());
        } else {
            v.push("E-L1-CLOSED".into());
            if m.launch.l1_cores {
                v.push("E-L1-CORES".into());
            } else {
                v.push("E-L1-PLACE".into());
            }
        }
    }
    if m.launch.env.residency.is_some() && !v.iter().any(|a| a == "E-RESIDENCY") {
        v.push("E-RESIDENCY".into());
    }
    if m.launch.env.tmem_cotenant {
        v.push("E-TMEM-COTENANT".into());
        if m.launch.env.tmem_permit && !m.launch.env.tmem_share.is_empty() {
            v.push("E-TMEM-PERMIT".into());
        }
    }
    if m.launch.read_done {
        v.push("E-READ-DONE".into());
    }
    if m.launch.remote_fifo {
        v.push("E-REMOTE-FIFO".into());
    }
    if m.launch.hw_split_ur && !m.ur_window.is_empty() {
        v.push("E-HW-SPLIT-UR".into());
    }
    let mc = !m.launch.mcast.is_empty();
    if mc {
        v.push("S-MULTIMEM".into());
        if m.launch.mc_alias_hw {
            v.push("E-MC-ALIAS-HW".into());
        }
    }
    if !m.launch.views.is_empty() || (mc && !m.launch.mc_alias_hw) {
        v.push("S-VA-ALIAS".into());
    }
    if (mc || !m.launch.views.is_empty()) && m.k.front == Front::Sass {
        v.push("L-ALIAS-FENCE".into());
    }
    v
}

pub fn get(id: &str) -> Option<&'static Assumption> {
    // S-MC-ALIAS became S-VA-ALIAS (every virtual alias, 2026-11-10): old verdicts name it.
    let id = if id == "S-MC-ALIAS" { "S-VA-ALIAS" } else { id };
    REGISTRY.iter().find(|a| a.id == id)
}

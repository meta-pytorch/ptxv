/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! E-WELLSYNC: the well-synchronization certificate against the fixture suite. Wherever it
//! certifies, its one-run verdict must be the exhaustive one (the variants' expected classes,
//! which `tests/fixtures.rs` checks against the explorers); every schedule-dependent bug must
//! fail to certify or be found by the run. On small kernels the certified outputs are also
//! compared with the explorer's.

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::machine::Machine;
use ptxv_sim::wellsync::certify;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Third-party kernel binaries are not distributed (fixtures/VENDOR_FIXTURES.toml): true when
/// one these fixtures need is missing (the test then skips).
fn vendor_missing(needs: &[&str]) -> bool {
    ptxv_case::skip_missing_vendor_files(&root().join("fixtures"), needs)
}

/// The kernel of a variant through one front-end (as tests/fixtures.rs builds it).
fn kernel(case: &Case, variant: &str, front: &str) -> ptxv_ir::Kernel {
    // Per call: tests run in parallel on the same variants.
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "ptxv-wstest-{}-{variant}-{}-{n}",
        case.m.name,
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    if let Some(bin) = &case.m.cubin {
        // Binary-only case: the captured cubin plus the variant's control-bit patch.
        let c = dir.join("k.cubin");
        std::fs::copy(case.dir.join(bin), &c).unwrap();
        // Patches of the case's cubin, and of launches' own cubins (`CtrlPatch::cubin`).
        let patch = &case.variant(variant).unwrap().patch;
        let (own, main): (Vec<_>, Vec<_>) = patch.iter().cloned().partition(|p| p.cubin.is_some());
        ptxv_sass::scoreboard::patch_file(&c, &case.m.entry, &main).unwrap();
        let mut patched = std::collections::BTreeMap::new();
        for p in &own {
            let name = p.cubin.clone().unwrap();
            let dst = patched.entry(name.clone()).or_insert_with(|| {
                let d = dir.join(&name);
                std::fs::copy(case.dir.join(&name), &d).unwrap();
                d
            });
            let entry = p.kernel.clone().unwrap_or_default();
            ptxv_sass::scoreboard::patch_file(dst, &entry, std::slice::from_ref(p)).unwrap();
        }
        let ks: Vec<_> = case
            .m
            .all_launches()
            .iter()
            .map(|l| match l.cubin {
                Some(own) if patched.contains_key(own) => {
                    ptxv_sass::load_cubin(&patched[own], l.entry).unwrap()
                }
                Some(own) => ptxv_sass::load_cubin(&case.dir.join(own), l.entry).unwrap(),
                None => ptxv_sass::load_cubin(&c, l.entry).unwrap(),
            })
            .collect();
        return if ks.len() == 1 {
            ks.into_iter().next().unwrap()
        } else {
            ptxv_ir::Kernel::concat(ks)
        };
    }
    let ptx = case.variant_ptx(variant).unwrap();
    let cubin = || {
        let p = dir.join("k.ptx");
        std::fs::write(&p, &ptx).unwrap();
        let c = dir.join("k.cubin");
        ptxv_sass::ptxas(&p, &c, &case.m.arch).unwrap();
        c
    };
    if !case.m.launches.is_empty() {
        let c = (front == "sass").then(cubin);
        let ks = case
            .m
            .all_launches()
            .iter()
            .map(|l| match &c {
                None => ptxv_ptx::parse_kernel(&ptx, Some(l.entry)).unwrap(),
                Some(c) => ptxv_sass::load_cubin(c, l.entry).unwrap(),
            })
            .collect();
        return ptxv_ir::Kernel::concat(ks);
    }
    match front {
        "ptx" => ptxv_ptx::parse_kernel(&ptx, Some(&case.m.entry)).unwrap(),
        _ => {
            let c = cubin();
            ptxv_sass::scoreboard::patch_file(
                &c,
                &case.m.entry,
                &case.variant(variant).unwrap().patch,
            )
            .unwrap();
            ptxv_sass::load_cubin(&c, &case.m.entry).unwrap()
        }
    }
}

fn machine(case: &Case, variant: &str, front: &str) -> Machine {
    let k = kernel(case, variant, front);
    // The variant's launch-order edits (`Variant::after`) belong to the launch: without them a
    // stream-order mutant (s_globals `unordered`) certified as the ordered original.
    let launch = build_launch(&case.for_variant(variant).unwrap(), &k).unwrap();
    Machine::new(Arc::new(k), Arc::new(launch))
}

/// Binary-only or very large launches (their certificate runs take minutes: see
/// `ptxv check --certify --slices`).
const SKIP: &[&str] = &[
    "r2_cublas_gemm",
    "r3_grouped_mm",
    "r4b_moe",
    // r3_grouped_mm's mutants: tma_store_wait_counts_its_own_scoreboard.
    "e_gmm_epilogue_reuse",
    // A bounded-search fixture (bounded_search_never_reports_clean): too wide to explore.
    "e_deep_race",
];

/// Variants the certificate must keep proving (a regression guard on its strength).
const MUST_CERTIFY: &[(&str, &str, &str)] = &[
    ("r0_tma_sum", "good", "ptx"),
    ("r0_tma_sum", "good", "sass"),
    ("r1a_tcgen05_tile", "good", "ptx"),
    ("r1a_tcgen05_tile", "good", "sass"),
    ("r1b_pipeline", "good", "ptx"),
    ("r1b_pipeline", "good", "sass"),
    ("r1c_cluster", "good", "sass"),
    ("r1c_2sm", "good", "sass"),
    // cta_group::2 M = 128 (Layout B) and M = 128 N = 8 (probes/tc_m128).
    // PTX e_mma_pair_m128: an S-MBAR-SCOPE contract finding (the leader's commit waited at
    // .cta scope in the peer), which the certificate refuses.
    ("e_mma_pair_m128", "good", "sass"),
    ("e_mma_m128_n8", "good", "ptx"),
    ("e_mma_m128_n8", "good", "sass"),
    ("r1e_pdl", "good", "sass"),
    ("e_bararv", "good", "sass"),
    ("e_barover", "good", "sass"),
    ("e_elect", "good", "ptx"),
    ("r2_cute_small", "good", "sass"),
    // Rung 3b: the whole MoE (5 launches, an 8-CTA cluster); gemm1 with stages wrapping
    // (try-acquire peeks, `walk_dead`).
    ("r3b_trtllm_moe", "fixed", "sass"),
    ("r3b_trtllm_bmm_k2048", "good", "sass"),
    // Flag polls (C-POLL).
    ("e_spin_smem", "good", "ptx"),
    ("e_spin_smem", "warp", "ptx"),
    ("r1f_pingpong", "good", "ptx"),
    ("r1f_pingpong", "fenced", "sass"),
    ("r1g_allreduce", "good", "ptx"),
    ("r1g_allreduce", "good", "sass"),
    ("e_poll_any", "good", "ptx"),
    ("e_poll_any", "good", "sass"),
    // Ring buffers on monotone counters (C-RING).
    ("e_ring", "good", "ptx"),
    ("e_ring", "good", "sass"),
    ("e_ring2", "good", "ptx"),
    ("e_ring2", "good", "sass"),
    ("e_ring_warp", "good", "ptx"),
    ("e_ring_warp", "good", "sass"),
    // CAS semaphores (C-CAS).
    ("s_cas", "good", "sass"),
    ("e_cas_sem", "good", "sass"),
    ("r4a_oneshot", "good", "sass"),
    ("r4a_twoshot", "good", "sass"),
    // Integer min / max and CAS atomics, CuTeDSL's PTX forms (cp.async, bulk copies, votes).
    ("s_atom_ext", "good", "ptx"),
    ("s_atom_ext", "good", "sass"),
    ("s_ptx_cute_ops", "good", "ptx"),
    ("s_ptx_cute_ops", "good", "sass"),
];

/// Soundness-harness miss 1: warp 0's plain init, then its own `red.add` / `ATOMS` on the
/// same word, then the other warps' atomics with no barrier. The certificate run must report
/// the race (it used to certify clean: the shadow kept only the writer's strong write), on
/// both front-ends and through `check_certified` (weak, then strong LDS/STS model, where the
/// SASS race shows as nondeterministic outputs); the good kernel still certifies.
#[test]
fn plain_write_hidden_by_own_atomic_is_a_race() {
    let case = Case::load(root().join("fixtures/e_cert_fasttrack")).unwrap();
    for front in ["ptx", "sass"] {
        let v = ptxv_sim::wellsync::check_certified(machine(&case, "nosync", front), 100_000);
        let want = case.variant("nosync").unwrap().expect_for(front);
        assert_eq!(v.classes(), want, "nosync/{front}: {:#?}", v.findings);
        let o = certify(machine(&case, "good", front));
        let v = o.verdict.expect("good certifies");
        assert!(v.complete && v.classes() == ["clean"], "good/{front}");
    }
}

/// Soundness-harness miss 2: a branch on a predicate nothing wrote (the cuBLAS tile
/// scheduler's try-wait NOPed, `BRA !P4` after it) is an `uninit-reg` violation of the run,
/// so the certificate refuses; the controls, where every lane runs the retry stub or skips the
/// wait, still certify (S-INIT-PRED: no false alarm on ptxas's `PT` / `ISETP` idioms).
#[test]
fn branch_on_unwritten_predicate_is_reported() {
    if vendor_missing(&["c_mm_bf16_mcast"]) {
        return;
    }
    let case = Case::load(root().join("fixtures/e_uninit_pred")).unwrap();
    let o = certify(machine(&case, "nowait", "sass"));
    assert!(o.verdict.is_none(), "nowait certified");
    assert!(
        o.reasons
            .iter()
            .any(|r| r.contains("reads register P4") && r.contains("0x1b80")),
        "{:?}",
        o.reasons
    );
    for v in ["good", "nowait_stub", "nowait_nobra", "stub"] {
        let o = certify(machine(&case, v, "sass"));
        let r = o.verdict.unwrap_or_else(|| panic!("{v}: {:?}", o.reasons));
        assert!(
            r.complete && r.classes() == ["contract"],
            "{v}: {:#?}",
            r.findings
        );
    }
}

/// Soundness-harness miss 3: the CUTLASS grouped GEMM's epilogue with its staging-buffer
/// wait removed (the `BAR.SYNC 0x1, 0x80` after the bulk-read wait, or the
/// `DEPBAR.LE SB0, 0x1` itself) rewrites a buffer a TMA store may still be reading. A
/// control-word wait on another scoreboard (the `BSYNC`'s SB3) used to drain every bulk
/// group and hide it; the certificate run must now report the race. The good kernel still
/// certifies (contract-only: E-BULK-EXIT).
/// A failed certificate whose own run found a race: when the exploration that replaces it
/// runs out of budget first, the verdict still reports the race (a feasible execution
/// reaches it; tools/mutate classifies r3_grouped_mm's miss-3 mutants by it). The deterministic
/// run that follows a search with no terminal state (explore.rs) may report it first.
#[test]
fn certificate_run_race_survives_an_incomplete_exploration() {
    let case = Case::load(root().join("fixtures/r0_tma_sum")).unwrap();
    for front in ["ptx", "sass"] {
        let v = ptxv_sim::wellsync::check_certified(machine(&case, "B02", front), 2);
        assert!(!v.complete, "{front}");
        assert!(
            v.classes().contains(&"race".to_string()),
            "{front}: {:#?}",
            v
        );
        assert!(
            v.notes
                .iter()
                .any(|n| n.contains("found by the certificate's run")
                    || n.contains("one deterministic run")),
            "{front}: {:?}",
            v.notes
        );
    }
}

#[test]
fn tma_store_wait_counts_its_own_scoreboard() {
    if vendor_missing(&["r3_grouped_mm"]) {
        return;
    }
    let case = Case::load(root().join("fixtures/e_gmm_epilogue_reuse")).unwrap();
    std::thread::scope(|s| {
        for v in ["nobar", "nodepbar", "good"] {
            let case = &case;
            s.spawn(move || {
                let o = certify(machine(case, v, "sass"));
                if v == "good" {
                    let r = o.verdict.unwrap_or_else(|| panic!("good: {:?}", o.reasons));
                    // E-BULK-EXIT: the last store is still reading at exit (contract only).
                    assert!(
                        r.complete && r.classes() == ["contract"],
                        "good: {:#?}",
                        r.findings
                    );
                } else {
                    assert!(o.verdict.is_none(), "{v} certified");
                    assert!(
                        o.reasons
                            .iter()
                            .any(|r| r.contains("race") && r.contains("UTMASTG")),
                        "{v}: {:?}",
                        o.reasons
                    );
                }
            });
        }
    });
}

#[test]
fn certified_verdicts_match_the_fixture_suite() {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root().join("fixtures"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("case.toml").exists())
        .collect();
    dirs.sort();
    let mut jobs = Vec::new();
    let mut absent = Vec::new();
    for d in dirs {
        let name = d.file_name().unwrap().to_str().unwrap().to_string();
        if vendor_missing(&[&name]) {
            absent.push(name);
            continue;
        }
        let case = Arc::new(Case::load(&d).unwrap());
        if SKIP.contains(&case.m.name.as_str()) {
            continue;
        }
        for v in case.m.variants.iter().map(|v| v.id.clone()) {
            for front in ["ptx", "sass"] {
                if front == "ptx" && case.m.cubin.is_some() {
                    continue;
                }
                let expect = case.variant(&v).unwrap().expect_for(front);
                if expect == ["unlifted"] {
                    continue; // a lifter coverage target (tests/fixtures.rs r4_kernel_classes)
                }
                jobs.push((case.clone(), v.clone(), front, expect));
            }
        }
    }
    // Independent single runs: spread them over the cores.
    let next = std::sync::atomic::AtomicUsize::new(0);
    let (certified, refused, errors) = (
        std::sync::Mutex::new(Vec::new()),
        std::sync::atomic::AtomicUsize::new(0),
        std::sync::Mutex::new(Vec::new()),
    );
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get().min(32));
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((case, v, front, expect)) = jobs.get(i) else {
                        break;
                    };
                    let o = certify(machine(case, v, front));
                    // C-TICKET-ADDR: `clean-up-to-permutation` stands for the explorer's
                    // `nondeterministic` (the outputs differ by a certified permutation of rows;
                    // tests/fixtures.rs `e_ticket_addr_permutation_certificate` compares them).
                    let same = |c: Vec<String>| {
                        let mut c: Vec<String> = c
                            .into_iter()
                            .map(|x| {
                                if x == "clean-up-to-permutation" {
                                    "nondeterministic".into()
                                } else {
                                    x
                                }
                            })
                            .collect();
                        c.sort();
                        c == *expect
                    };
                    match o.verdict {
                        Some(verdict) if !verdict.complete || !same(verdict.classes()) => {
                            errors.lock().unwrap().push(format!(
                                "{}/{v}/{front}: certified verdict {:?} (complete: {}) differs \
                                 from the exhaustive one {expect:?}: {:#?}",
                                case.m.name,
                                verdict.classes(),
                                verdict.complete,
                                verdict.findings
                            ));
                        }
                        Some(_) => {
                            certified
                                .lock()
                                .unwrap()
                                .push((case.m.name.clone(), v.clone(), *front))
                        }
                        None => {
                            if o.reasons.is_empty() {
                                errors.lock().unwrap().push(format!(
                                    "{}/{v}/{front}: refused without a reason",
                                    case.m.name
                                ));
                            }
                            refused.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
            });
        }
    });
    let errors = errors.into_inner().unwrap();
    assert!(errors.is_empty(), "{}", errors.join("\n"));
    let (certified, refused) = (certified.into_inner().unwrap(), refused.into_inner());
    eprintln!(
        "certificate: {} certified, {refused} refused (explored instead)",
        certified.len()
    );
    for (c, v, f) in MUST_CERTIFY {
        if absent.iter().any(|a| a == c) {
            continue;
        }
        assert!(
            certified.iter().any(|(a, b, g)| a == c && b == v && g == f),
            "{c}/{v}/{f} no longer certifies"
        );
    }
}

/// Small kernels: the certified outputs are the explorer's (every terminal state).
#[test]
fn certified_outputs_match_exploration() {
    for (c, v, f) in [
        ("r0_tma_sum", "good", "ptx"),
        ("r0_tma_sum", "good", "sass"),
        ("r1a_tcgen05_tile", "good", "ptx"),
        ("r1a_tcgen05_tile", "good", "sass"),
        ("e_bararv", "good", "sass"),
        ("e_barover", "good", "ptx"),
        ("e_elect", "good", "ptx"),
        ("e_sym", "good", "sass"),
        ("r1e_pdl", "good", "ptx"),
        // (r1c_cluster PTX: now an S-MBAR-SCOPE contract finding, refused by the certificate.)
        ("s_alu", "good", "sass"),
        ("e_spin_smem", "good", "ptx"),
        ("r1f_pingpong", "good", "sass"),
        ("r1g_allreduce", "good", "ptx"),
        ("s_cas", "good", "sass"),
        ("e_cas_sem", "good", "sass"),
        ("e_ring", "good", "ptx"),
        ("e_ring", "good", "sass"),
        ("e_ring2", "good", "ptx"),
        ("e_ring2", "good", "sass"),
        ("e_ring_warp", "good", "ptx"),
        ("e_ring_warp", "good", "sass"),
        ("e_poll_any", "good", "ptx"),
        ("e_poll_any", "good", "sass"),
        // A lane exits inside a BSSY region while the others converge (E-BSYNC exit order).
        ("e_bsync_exit", "good", "sass"),
    ] {
        let case = Case::load(root().join("fixtures").join(c)).unwrap();
        let m = machine(&case, v, f);
        let cv = certify(m.clone())
            .verdict
            .unwrap_or_else(|| panic!("{c}/{v}/{f} does not certify"));
        let ev = ptxv_sim::explore::explore(m, 2_000_000);
        assert!(ev.complete, "{c}/{v}/{f}: exploration incomplete");
        assert_eq!(cv.classes(), ev.classes(), "{c}/{v}/{f}");
        let (mut a, mut b) = (cv.outputs.clone(), ev.outputs.clone());
        a.sort();
        b.sort();
        assert_eq!(
            a, b,
            "{c}/{v}/{f}: certified outputs differ from the explorer's"
        );
    }
}

/// C-POLL liveness: a poll that exits in the run but may read a concurrent write it fails on
/// last (no write it leaves on is ordered after it) can block forever. e_poll_any `dl`:
/// thread 0 sets the flag to 1, a barrier, then polls it; thread 32 stores 0 after the barrier.
/// The run's poll reads 1 and exits; with the store first it spins forever.
#[test]
fn poll_that_may_read_a_blocking_write_last_is_refused() {
    let case = Case::load(root().join("fixtures/e_poll_any")).unwrap();
    for f in ["ptx", "sass"] {
        let o = certify(machine(&case, "dl", f));
        assert!(o.verdict.is_none(), "dl/{f} certified");
        // Which reason depends on the run's schedule (explore.rs `DetSched`): a run in which
        // the poll reads 1 and exits is refused by C-POLL; a run in which the store comes
        // first deadlocks, and the explorer decides it.
        assert!(
            o.reasons.iter().any(|r| (r.starts_with("C-POLL:")
                && r.ends_with("never leaves the loop"))
                || r.contains("first: deadlock")),
            "dl/{f}: {:#?}",
            o.reasons
        );
    }
}

/// A poll whose exit may read either of two writes, when the value is used, is refused: the
/// flag is a monotone counter (0, 1, 2 by one thread), so C-RING runs both choices, and they
/// leave different outputs.
#[test]
fn poll_reading_either_write_is_refused() {
    let case = Case::load(root().join("fixtures/e_spin_smem")).unwrap();
    let o = certify(machine(&case, "SP6", "ptx"));
    assert!(o.verdict.is_none());
    assert!(
        o.reasons
            .iter()
            .any(|r| r.starts_with("C-RING: not chunking-oblivious")),
        "{:#?}",
        o.reasons
    );
}

/// C-RING (ring buffers with monotone head and tail counters, fixtures e_ring, e_ring2 and
/// e_ring_warp): every mutant is refused for the right reason or certified with its
/// exhaustive verdict (it is not: each mutant's bug shows in some run of the search).
#[test]
fn ring_mutants_are_refused_with_their_reason() {
    for (fx, v, want) in [
        ("e_ring", "R1", "race"),
        ("e_ring", "R2", "race"),
        ("e_ring", "R3", "race"),
        ("e_ring", "R4", "C-RING: not chunking-oblivious"),
        ("e_ring", "R5", "race"),
        ("e_ring", "R6", "race"),
        ("e_ring", "R7", "C-RING: not chunking-oblivious"),
        ("e_ring2", "R1", "race"),
        ("e_ring2", "R2", "race"),
        ("e_ring2", "R3", "race"),
        ("e_ring2", "R4", "C-RING: not chunking-oblivious"),
        ("e_ring2", "R5", "race"),
        ("e_ring2", "R6", "race"),
        ("e_ring2", "R7", "C-RING: not chunking-oblivious"),
        ("e_ring_warp", "R1", "race"),
        ("e_ring_warp", "R2", "race"),
        ("e_ring_warp", "R4", "C-RING: not chunking-oblivious"),
        ("e_ring_warp", "R7", "C-RING: not chunking-oblivious"),
        // W1's racing writes store equal values in the certificate's run, but the values come
        // through polls, so S-SAME-VALUE refuses rather than accepting the note.
        ("e_ring_warp", "W1", "S-SAME-VALUE"),
    ] {
        let case = Case::load(root().join("fixtures").join(fx)).unwrap();
        for front in ["ptx", "sass"] {
            let o = certify(machine(&case, v, front));
            assert!(o.verdict.is_none(), "{fx}/{v}/{front} certified");
            let hit = |r: &String| {
                if want == "race" {
                    r.contains("reported findings (race")
                } else {
                    r.starts_with(want)
                }
            };
            assert!(
                o.reasons.iter().any(hit),
                "{fx}/{v}/{front}: {:#?}",
                o.reasons
            );
        }
    }
}

/// C-RING decides the good rings with one run per sequence of choices: several runs, one of
/// them dropped as infeasible (the consumer cannot see the last tail first: the producer
/// needs the head first), and the verdict notes the reliance on E-RESIDENCY.
#[test]
fn ring_search_runs_every_choice() {
    let case = Case::load(root().join("fixtures/e_ring")).unwrap();
    for front in ["ptx", "sass"] {
        let v = certify(machine(&case, "good", front))
            .verdict
            .unwrap_or_else(|| panic!("e_ring/good/{front} does not certify"));
        let note = v
            .notes
            .iter()
            .find(|n| n.starts_with("C-RING"))
            .unwrap_or_else(|| panic!("no C-RING note: {:?}", v.notes));
        assert!(
            note.contains("4 run(s) (1 dropped as infeasible)"),
            "{note}"
        );
        assert!(note.contains("E-RESIDENCY"), "{note}");
        assert!(v.assumptions.iter().any(|a| a == "E-RESIDENCY"));
    }
}

/// C-POLL-ATOM: the poller's reset of the counter after its exit is ordered after the adds
/// the exit needed (coherence), so the certificate holds; without the poll it would be a
/// C-ORDER conflict.
#[test]
fn poll_atom_reset_certifies() {
    let case = Case::load(root().join("fixtures/e_poll_atom")).unwrap();
    for front in ["ptx", "sass"] {
        let o = certify(machine(&case, "reset", front));
        assert!(
            o.verdict.is_some(),
            "e_poll_atom/reset/{front}: {:#?}",
            o.reasons
        );
    }
}

/// C-TICKET: a persistent tile scheduler (atom.inc ticket counter, arrival counter reset by
/// the last arriver) certifies with one run per claim order; outputs that depend on the
/// claimant, and tiles that share an output word, are refused with their reasons.
#[test]
fn ticket_counter_orders_are_enumerated() {
    let case = Case::load(root().join("fixtures/e_ticket")).unwrap();
    for front in ["ptx", "sass"] {
        let v = certify(machine(&case, "good", front))
            .verdict
            .unwrap_or_else(|| panic!("e_ticket/good/{front} does not certify"));
        let note = v
            .notes
            .iter()
            .find(|n| n.starts_with("C-TICKET"))
            .unwrap_or_else(|| panic!("no C-TICKET note: {:?}", v.notes));
        assert!(note.contains("2 ticket counter(s)"), "{note}");
        for (var, want) in [
            ("ctaid", "C-TICKET: the claim order reaches the outputs"),
            ("race", "reported findings (race"),
        ] {
            let o = certify(machine(&case, var, front));
            assert!(o.verdict.is_none(), "e_ticket/{var}/{front} certified");
            assert!(
                o.reasons.iter().any(|r| r.contains(want)),
                "e_ticket/{var}/{front}: {:#?}",
                o.reasons
            );
        }
    }
}

/// The first call of a two-call multi-GPU capture (as tests/fixtures.rs `first_call`): each
/// GPU's first launch, the outputs named in `keep`; `expect = false` drops the GPU bytes.
fn first_call(mut case: Case, keep: &[&str], expect: bool) -> Case {
    let mut seen = vec![0u32];
    case.m.launches.retain(|l| {
        let first = !seen.contains(&l.device);
        seen.push(l.device);
        first
    });
    case.m.outputs.retain(|o| keep.contains(&o.as_str()));
    case.m
        .replicas
        .retain(|g| g.iter().all(|o| keep.contains(&o.as_str())));
    if !expect {
        for b in &mut case.m.buffers {
            b.expect = None;
        }
    }
    case
}

/// The rung-4a cases (torch symmetric-memory all-reduce on 3 GPUs, docs/rung4.md): (label,
/// case, whether its GPU bytes are this case's final state).
fn r4a_cases() -> Vec<(String, Case, bool)> {
    let one = || Case::load(root().join("fixtures/r4a_oneshot")).unwrap();
    let two = || Case::load(root().join("fixtures/r4a_twoshot")).unwrap();
    vec![
        ("one-shot, 2 calls".into(), one(), true),
        ("two-shot, 2 calls".into(), two(), true),
        (
            "one-shot, 1 call".into(),
            first_call(one(), &["o1_r0", "o1_r1", "o1_r2"], true),
            true,
        ),
        (
            "two-shot, 1 call".into(),
            first_call(two(), &["buf_r0", "buf_r1", "buf_r2"], false),
            false,
        ),
    ]
}

/// C-CAS: torch's one-shot and two-shot all-reduce certify, one call and two back-to-back
/// calls (the latter take billions of states to explore). Under the case's E-EXIT-FENCE the
/// certified outputs equal every rank's GPU bytes.
#[test]
fn r4a_cas_signals_certify() {
    if std::env::var("PTXV_EXIT_FENCE").is_ok() {
        return; // the case's own exit_fence is what is tested
    }
    for (what, case, gpu) in r4a_cases() {
        let o = certify(machine(&case, "good", "sass"));
        let v = o
            .verdict
            .unwrap_or_else(|| panic!("{what}: does not certify: {:#?}", o.reasons));
        assert_eq!(v.classes(), ["clean"], "{what}: {:#?}", v.findings);
        assert_eq!(v.outputs.len(), 1, "{what}");
        if gpu {
            let report = case.expect_report(&v.outputs[0]).unwrap();
            assert!(
                !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
                "{what}: {report:?}"
            );
        }
    }
}

/// One call explores to completion: the certified verdict and output are the explorer's, with
/// and without E-EXIT-FENCE (the strict model is clean for one call).
#[test]
fn r4a_certified_first_call_matches_exploration() {
    if std::env::var("PTXV_EXIT_FENCE").is_ok() {
        return;
    }
    for (what, case, _) in r4a_cases().into_iter().filter(|c| c.0.contains("1 call")) {
        for strict in [false, true] {
            let mut case = case.clone();
            if strict {
                case.m.exit_fence = None;
            }
            let m = machine(&case, "good", "sass");
            let cv = certify(m.clone())
                .verdict
                .unwrap_or_else(|| panic!("{what} (strict: {strict}) does not certify"));
            let ev = ptxv_sim::explore::explore(m, 4_000_000);
            assert!(ev.complete, "{what}: exploration incomplete");
            assert_eq!(cv.classes(), ev.classes(), "{what} (strict: {strict})");
            assert_eq!(cv.outputs, ev.outputs, "{what} (strict: {strict})");
        }
    }
}

/// Without E-EXIT-FENCE the one-shot's closing barrier does not order a peer's reads of call 1
/// before this rank's restaging in call 2 (T4a): the certificate must not call it clean.
#[test]
fn r4a_strict_model_does_not_certify_the_reuse_race() {
    let mut case = Case::load(root().join("fixtures/r4a_oneshot")).unwrap();
    case.m.exit_fence = None;
    let o = certify(machine(&case, "good", "sass"));
    match o.verdict {
        None => assert!(
            o.reasons.iter().any(|r| r.contains("race")),
            "{:#?}",
            o.reasons
        ),
        Some(v) => assert!(
            v.classes().contains(&"race".to_string()),
            "{:?}",
            v.classes()
        ),
    }
}

/// The rung-4a mutants (B57-B62), one and two calls: refused, or certified with their bug.
#[test]
fn r4a_mutants_are_not_certified_clean() {
    for (what, case, _) in r4a_cases() {
        let vs: &[&str] = if case.m.name == "r4a_oneshot" {
            &["B57", "B58", "B59", "B60", "B61"]
        } else {
            &["B62"]
        };
        for v in vs {
            let m = machine(&case, v, "sass");
            let o = certify(m.clone());
            let Some(cv) = o.verdict else { continue };
            if what.contains("2 calls") {
                let expect = case.variant(v).unwrap().expect_for("sass");
                assert!(
                    cv.meets(&expect),
                    "{what}/{v}: certified {:?}, want {expect:?}",
                    cv.classes()
                );
            } else {
                // The variants' bugs are about reuse by the second call; one call explores
                // to completion and must agree.
                let ev = ptxv_sim::explore::explore(m, 4_000_000);
                assert!(ev.complete, "{what}/{v}: exploration incomplete");
                assert_eq!(cv.classes(), ev.classes(), "{what}/{v}");
                assert_eq!(cv.outputs, ev.outputs, "{what}/{v}");
            }
        }
    }
}

/// CAS patterns whose outcome depends on the schedule (fixtures/e_cas_sem, other entry
/// points of its cubin): the explorer shows each bug, and the certificate refuses each, for
/// the reason given.
#[test]
fn cas_patterns_that_must_be_refused() {
    for (entry, why, bug) in [
        // Two lock acquisitions (0 -> 1) by different threads: either goes first.
        (
            "cas_lock",
            "either may consume the value",
            "nondeterministic",
        ),
        // 0 -> 1 and 0 -> 2 compete for the same value: a deadlock when 2 goes first.
        ("cas_two_kinds", "compare the same value", "deadlock"),
        // A put loop that also exits on a failed compare (against 2).
        (
            "cas_exit_fail",
            "leaves its loop after a failed compare",
            "nondeterministic",
        ),
        // A strong store to the signal word, unordered with the CAS polls.
        ("cas_store", "C-ORDER: strong accesses", "deadlock"),
    ] {
        let mut case = Case::load(root().join("fixtures/e_cas_sem")).unwrap();
        case.m.entry = entry.into();
        let m = machine(&case, "good", "sass");
        let ev = ptxv_sim::explore::explore(m.clone(), 2_000_000);
        assert!(ev.complete, "{entry}: exploration incomplete");
        assert!(
            ev.classes().contains(&bug.to_string()),
            "{entry}: explorer {:?}",
            ev.classes()
        );
        let o = certify(m);
        assert!(
            o.verdict.is_none(),
            "{entry} certified: {:?}",
            o.verdict.map(|v| v.classes())
        );
        assert!(
            o.reasons.iter().any(|r| r.contains(why)),
            "{entry}: {:#?}",
            o.reasons
        );
    }
}

/// C-FADD: fp32 atomic adds whose order changes the bits. Triton's row sum (4 programs per
/// row, `REDG.E.ADD.F32`) writes only the output words, which nothing reads afterwards, so
/// the certificate reports every reachable output (nondeterministic in the output only),
/// and the GPU's captured bytes are one of them. Every output a bounded exploration reaches
/// is among them.
#[test]
fn fadd_orders_are_reported_as_output_nondeterminism() {
    let case = Case::load(root().join("fixtures/c_triton_rowsum_atomic")).unwrap();
    let v = certify(machine(&case, "good", "sass"))
        .verdict
        .expect("certified");
    assert_eq!(v.classes(), ["nondeterministic"]);
    assert_eq!(v.distinct_outputs, 9);
    assert!(
        v.notes.iter().any(|n| n.starts_with("C-FADD")),
        "{:?}",
        v.notes
    );
    assert!(
        v.outputs.iter().any(|o| case
            .expect_report(o)
            .unwrap()
            .iter()
            .all(|l| l.contains("bitwise equal"))),
        "the GPU's output is not among the certified ones"
    );
    let e = ptxv_sim::explore::explore(machine(&case, "good", "sass"), 200_000);
    assert!(!e.outputs.is_empty());
    for o in &e.outputs {
        assert!(v.outputs.contains(o), "an explored output is not certified");
    }
}

/// C-FADD negative: the same order-dependent adds, but the word is not a launch output (so
/// a later launch may read it): the certificate must refuse, naming the different values.
#[test]
fn fadd_orders_outside_the_outputs_are_refused() {
    let case = Case::load(root().join("fixtures/c_triton_rowsum_atomic")).unwrap();
    let k = kernel(&case, "good", "sass");
    let mut launch = build_launch(&case, &k).unwrap();
    launch.outputs.clear();
    let o = certify(Machine::new(Arc::new(k), Arc::new(launch)));
    assert!(
        o.verdict.is_none(),
        "certified: {:?}",
        o.verdict.map(|v| v.classes())
    );
    assert!(
        o.reasons
            .iter()
            .any(|r| r.contains("C-FADD") && r.contains("different values")),
        "{:#?}",
        o.reasons
    );
}

/// E-DESC-VERSION: Triton 3.5 leaves a tcgen05 shared-memory descriptor's fixed bits 46-48 at
/// 0. The finding checks a descriptor value, which the certificate makes the same in every
/// schedule, so it passes through the certificate: the kernel certifies (one run for all
/// schedules) and the finding is still reported, as a `contract` finding.
#[test]
fn desc_version_finding_does_not_break_the_certificate() {
    let case = Case::load(root().join("fixtures/c_triton_mm_bf16")).unwrap();
    let v = ptxv_sim::wellsync::check_certified(machine(&case, "good", "sass"), 100_000);
    assert!(
        v.notes.iter().any(|n| n.contains("well-synchronized")),
        "{:?}",
        v.notes
    );
    assert!(
        v.findings
            .iter()
            .any(|f| f.sig.starts_with("desc-version:")),
        "{:?}",
        v.findings.iter().map(|f| &f.sig).collect::<Vec<_>>()
    );
    // A contract finding (GB300's MMA ignores the bits, probes/desc_version), no violation.
    assert!(
        v.findings
            .iter()
            .all(|f| !f.sig.starts_with("desc-version:") || f.kind == "contract"),
        "descriptor finding is a contract one"
    );
    assert!(
        v.findings.iter().all(|f| f.kind != "violation"),
        "no violation"
    );
}

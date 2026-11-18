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

/// The kernel of a variant through one front-end (as tests/fixtures.rs builds it), or why it
/// does not load (a front-end that refuses the variant, ptxas, a missing file).
fn try_kernel(case: &Case, variant: &str, front: &str) -> anyhow::Result<ptxv_ir::Kernel> {
    // Per call: tests run in parallel on the same variants.
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "ptxv-wstest-{}-{variant}-{}-{n}",
        case.m.name,
        std::process::id()
    ));
    std::fs::create_dir_all(&dir)?;
    if let Some(bin) = &case.m.cubin {
        // Binary-only case: the captured cubin plus the variant's control-bit patch.
        let c = dir.join("k.cubin");
        std::fs::copy(case.dir.join(bin), &c)?;
        // Patches of the case's cubin, and of launches' own cubins (`CtrlPatch::cubin`).
        let patch = &case.variant(variant)?.patch;
        let (own, main): (Vec<_>, Vec<_>) = patch.iter().cloned().partition(|p| p.cubin.is_some());
        ptxv_sass::scoreboard::patch_file(&c, &case.m.entry, &main)?;
        let mut patched = std::collections::BTreeMap::new();
        for p in &own {
            let name = p.cubin.clone().unwrap_or_default();
            if !patched.contains_key(&name) {
                let d = dir.join(&name);
                std::fs::copy(case.dir.join(&name), &d)?;
                patched.insert(name.clone(), d);
            }
            let entry = p.kernel.clone().unwrap_or_default();
            ptxv_sass::scoreboard::patch_file(&patched[&name], &entry, std::slice::from_ref(p))?;
        }
        let ks = case
            .m
            .all_launches()
            .iter()
            .map(|l| match l.cubin {
                Some(own) if patched.contains_key(own) => {
                    ptxv_sass::load_cubin(&patched[own], l.entry)
                }
                Some(own) => ptxv_sass::load_cubin(&case.dir.join(own), l.entry),
                None => ptxv_sass::load_cubin(&c, l.entry),
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        return Ok(if ks.len() == 1 {
            ks.into_iter().next().unwrap()
        } else {
            ptxv_ir::Kernel::concat(ks)
        });
    }
    let ptx = case.variant_ptx(variant)?;
    let cubin = || -> anyhow::Result<PathBuf> {
        let p = dir.join("k.ptx");
        std::fs::write(&p, &ptx)?;
        let c = dir.join("k.cubin");
        ptxv_sass::ptxas(&p, &c, &case.m.arch)?;
        Ok(c)
    };
    let toolkit = case
        .m
        .ptxas_toolkit
        .as_deref()
        .and_then(ptxv_ptx::toolkit_version);
    if !case.m.launches.is_empty() {
        let c = (front == "sass").then(cubin).transpose()?;
        let ks = case
            .m
            .all_launches()
            .iter()
            .map(|l| match &c {
                None => ptxv_ptx::parse_kernel_for(&ptx, Some(l.entry), toolkit),
                Some(c) => ptxv_sass::load_cubin(c, l.entry),
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        return Ok(ptxv_ir::Kernel::concat(ks));
    }
    match front {
        "ptx" => ptxv_ptx::parse_kernel_for(&ptx, Some(&case.m.entry), toolkit),
        _ => {
            let c = cubin()?;
            ptxv_sass::scoreboard::patch_file(&c, &case.m.entry, &case.variant(variant)?.patch)?;
            ptxv_sass::load_cubin(&c, &case.m.entry)
        }
    }
}

/// [`try_kernel`] for tests whose variants must load.
fn kernel(case: &Case, variant: &str, front: &str) -> ptxv_ir::Kernel {
    try_kernel(case, variant, front)
        .unwrap_or_else(|e| panic!("{}/{variant}/{front}: {e:#}", case.m.name))
}

/// The machine of a variant through one front-end, or why its kernel or launch does not load.
fn try_machine(case: &Case, variant: &str, front: &str) -> anyhow::Result<Machine> {
    let k = try_kernel(case, variant, front)?;
    // The variant's launch-order edits (`Variant::after`) belong to the launch: without them a
    // stream-order mutant (s_globals `unordered`) certified as the ordered original.
    let launch = build_launch(&case.for_variant(variant)?, &k)?;
    Ok(Machine::new(Arc::new(k), Arc::new(launch)))
}

fn machine(case: &Case, variant: &str, front: &str) -> Machine {
    try_machine(case, variant, front)
        .unwrap_or_else(|e| panic!("{}/{variant}/{front}: {e:#}", case.m.name))
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
    // The certificate certifies its SASS clean where the exploration is nondeterministic (MISS
    // 150: a warp's splits are not independent through its uniform registers); pinned in
    // tests/fixtures.rs e_split_ur_strict_level_switches_inside_a_window.
    "e_split_ur",
];

/// Variants the certificate must keep proving (a regression guard on its strength).
const MUST_CERTIFY: &[(&str, &str, &str)] = &[
    ("r0_tma_sum", "good", "ptx"),
    ("r0_tma_sum", "good", "sass"),
    ("r1a_tcgen05_tile", "good", "ptx"),
    // r1a_tcgen05_tile, r1c_cluster and r1c_2sm SASS: refused since MISS 81 (C-WARP: the
    // lanes leave ptxas's BSYNC without .RECONVERGENT apart and meet again at a TRYWAIT).
    ("r1b_pipeline", "good", "ptx"),
    ("r1b_pipeline", "good", "sass"),
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
    // S-FENCE-SC: the publication certifies with the forced Fence-SC order (C-TICKET).
    ("e_fence_sc_publish", "good", "ptx"),
    ("e_fence_sc_publish", "good", "sass"),
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
#[ignore = "heavy: 43 s"]
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

/// Runs `run` over `jobs` on `workers` threads and returns the failures, each `name: why`.
///
/// A job that panics fails under its own name, and no job starts after it (the running ones
/// finish): a panic in a worker thread is not lost while the others keep the sweep going. A
/// job running longer than `limit` calls `stall` once with the names and ages of the running
/// jobs (a certificate or explorer run that does not end is a failure, not a hang); the
/// fixture suite's `stall` exits the test process. With `trace`, each job's start and time go
/// to stderr unbuffered (visible under captured output too).
fn run_sweep<J: Sync>(
    jobs: &[J],
    workers: usize,
    name: &(dyn Fn(&J) -> String + Sync),
    run: &(dyn Fn(&J) -> Result<(), String> + Sync),
    limit: Option<std::time::Duration>,
    stall: &(dyn Fn(&str) + Sync),
    trace: bool,
) -> Vec<String> {
    use std::io::Write;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Instant;
    let say = |msg: String| {
        // Unbuffered and past the test harness's capture.
        let _ = std::io::stderr().write_all(format!("{msg}\n").as_bytes());
    };
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let alive = AtomicUsize::new(workers);
    let errors = Mutex::new(Vec::new());
    let running: Mutex<Vec<Option<(usize, Instant)>>> = Mutex::new(vec![None; workers]);
    let times = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for w in 0..workers {
            let (next, stop, alive, errors, running, times, say) =
                (&next, &stop, &alive, &errors, &running, &times, &say);
            s.spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(i) else { break };
                    let t = Instant::now();
                    running.lock().unwrap_or_else(|e| e.into_inner())[w] = Some((i, t));
                    if trace {
                        say(format!("sweep job {i}/{} {}: start", jobs.len(), name(job)));
                    }
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(job)));
                    running.lock().unwrap_or_else(|e| e.into_inner())[w] = None;
                    let secs = t.elapsed().as_secs_f64();
                    times
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push((secs, i));
                    if trace {
                        say(format!(
                            "sweep job {i}/{} {}: {secs:.1} s",
                            jobs.len(),
                            name(job)
                        ));
                    }
                    let why = match r {
                        Ok(Ok(())) => continue,
                        Ok(Err(why)) => why,
                        Err(p) => {
                            stop.store(true, Ordering::Relaxed);
                            let msg = p
                                .downcast_ref::<String>()
                                .map(String::as_str)
                                .or_else(|| p.downcast_ref::<&str>().copied())
                                .unwrap_or("(no message)");
                            format!("panicked: {msg}")
                        }
                    };
                    errors
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(format!("{}: {why}", name(job)));
                }
                alive.fetch_sub(1, Ordering::Relaxed);
            });
        }
        let Some(limit) = limit else { return };
        let (alive, running, errors) = (&alive, &running, &errors);
        s.spawn(move || {
            let tick = (limit / 8).clamp(
                std::time::Duration::from_millis(20),
                std::time::Duration::from_secs(5),
            );
            while alive.load(Ordering::Relaxed) > 0 {
                std::thread::sleep(tick);
                let late: Vec<String> = running
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .iter()
                    .flatten()
                    .filter(|(_, t)| t.elapsed() > limit)
                    .map(|&(i, t)| {
                        format!("{} ({:.0} s)", name(&jobs[i]), t.elapsed().as_secs_f64())
                    })
                    .collect();
                if !late.is_empty() {
                    let mut errors = errors.lock().unwrap_or_else(|e| e.into_inner());
                    let mut msg = format!(
                        "job(s) still running after the {} s limit: {}",
                        limit.as_secs_f64(),
                        late.join(", ")
                    );
                    // The failures so far: `stall` may end the process before they are seen.
                    if !errors.is_empty() {
                        msg += &format!("; failed before: {}", errors.join("\n"));
                    }
                    errors.push(msg.clone());
                    drop(errors);
                    stall(&msg);
                    return;
                }
            }
        });
    });
    let mut times = times.into_inner().unwrap_or_else(|e| e.into_inner());
    times.sort_by(|a, b| b.0.total_cmp(&a.0));
    let slowest: Vec<String> = times
        .iter()
        .take(5)
        .map(|&(t, i)| format!("{} {t:.1} s", name(&jobs[i])))
        .collect();
    eprintln!(
        "sweep: {} of {} jobs ran; slowest: {}",
        times.len(),
        jobs.len(),
        slowest.join(", ")
    );
    errors.into_inner().unwrap_or_else(|e| e.into_inner())
}

/// A panicking job fails the sweep under its name, promptly: the jobs queued after it do not
/// run (the sweep used to lose the panic in its worker and run on).
#[test]
fn sweep_reports_a_panicking_job_by_name() {
    let ran = std::sync::atomic::AtomicUsize::new(0);
    let jobs: Vec<usize> = (0..100).collect();
    let errors = run_sweep(
        &jobs,
        1,
        &|j| format!("job{j}"),
        &|&j| {
            ran.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            match j {
                3 => Err("wrong verdict".into()),
                5 => panic!("unsupported PTX instruction"),
                _ => Ok(()),
            }
        },
        None,
        &|_| unreachable!(),
        false,
    );
    assert_eq!(
        errors,
        [
            "job3: wrong verdict",
            "job5: panicked: unsupported PTX instruction"
        ]
    );
    assert_eq!(ran.into_inner(), 6, "jobs ran after the panic");
}

/// A job that does not end trips the limit, which names it (the fixture suite then exits the
/// test process: a run that cannot be interrupted must not hold the test forever).
#[test]
fn sweep_reports_a_job_past_its_limit_by_name() {
    let tripped = std::sync::atomic::AtomicBool::new(false);
    let jobs = ["quick", "stuck"];
    let t = std::time::Instant::now();
    let errors = run_sweep(
        &jobs,
        2,
        &|j| j.to_string(),
        &|&j| {
            // The stuck job runs until the limit has been reported (or a minute: the test fails).
            while j == "stuck"
                && !tripped.load(std::sync::atomic::Ordering::Relaxed)
                && t.elapsed().as_secs() < 60
            {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            Ok(())
        },
        Some(std::time::Duration::from_millis(200)),
        &|msg| {
            assert!(msg.contains("stuck (") && !msg.contains("quick"), "{msg}");
            tripped.store(true, std::sync::atomic::Ordering::Relaxed);
        },
        false,
    );
    assert!(tripped.into_inner());
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(t.elapsed().as_secs() < 30);
}

/// MISS 235: the certificate's run stops at a step that stops threads at an error, and the
/// certificate does not hold. r4c_nowrap B36 (SASS): a dispatch warp's bulk copy faults, and
/// past it the other warps of its named barrier went round BAR.SYNC forever (the attempt never
/// ended; its memory grew). The certificate runs on a thread of its own: a run that does not
/// end fails the test after 120 s instead of holding it.
#[test]
fn cert_run_stops_at_a_thread_stopping_error() {
    if vendor_missing(&["r4c_nowrap"]) {
        return;
    }
    let case = Case::load(root().join("fixtures/r4c_nowrap")).unwrap();
    let m = machine(&case, "B36", "sass");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(certify(m));
    });
    let o = match rx.recv_timeout(std::time::Duration::from_secs(120)) {
        Ok(o) => o,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("the certificate's run did not end within 120 s (MISS 235)")
        }
        Err(e) => panic!("the certificate's thread ended without an outcome: {e}"),
    };
    assert!(o.verdict.is_none(), "certified: {:#?}", o.verdict);
    assert!(
        o.reasons
            .iter()
            .any(|r| r.contains("stopped its threads at an error (fault: ")),
        "{:#?}",
        o.reasons
    );
    // About 4.5k steps (3.2k with LDS/STS plain); the runaway took 770 a second without end.
    assert!(o.steps < 6_000, "{} steps", o.steps);
}

/// MISS 235 on a small fixture: e_atom_half exact_size, whose 16-bit atomic emulation reads past
/// the allocation (`fault`). The certificate is refused, its reason naming the fault, and
/// `check_certified` (exploration after the refusal) keeps the fault: the fixture's verdict.
#[test]
fn cert_refuses_a_faulting_run_and_keeps_the_fault() {
    let case = Case::load(root().join("fixtures/e_atom_half")).unwrap();
    for front in ["ptx", "sass"] {
        let o = certify(machine(&case, "exact_size", front));
        assert!(
            o.verdict.is_none(),
            "{front}: certified {:?}",
            o.verdict.map(|v| v.classes())
        );
        assert!(
            o.reasons.iter().any(|r| {
                r.contains("stopped its threads at an error (fault: ") && r.contains("outside")
            }),
            "{front}: {:#?}",
            o.reasons
        );
        let v = ptxv_sim::wellsync::check_certified(machine(&case, "exact_size", front), 200_000);
        assert_eq!(
            v.classes(),
            case.variant("exact_size").unwrap().expect_for(front),
            "{front}: {:#?}",
            v.findings
        );
    }
}

/// The front-ends' refusals of a form they do not take: a variant whose front expects
/// `unlifted` must fail to load with one of them.
const FRONT_REFUSALS: &[&str] = &[
    "unsupported PTX instruction",
    "unsupported SASS opcode",
    "ptxas error",
    // The SASS front-end's modifier audit (modset.rs): a modifier outside the audited set is
    // refused before the opcode is lifted (e_lane_order EXCH, e_lane_order_ticket EXCHLIVE:
    // `ATOMS.EXCH`).
    "not in the audited set",
];

/// Runs whose certified verdict differs from the fixture's expectation for a registered miss
/// not fixed yet (the expectation is the correct verdict): (case, variant, front, miss, the
/// certified classes while it is open), 18 runs. The certificate must give exactly these
/// classes; another verdict, a refusal, or the expected verdict (fixed: drop the row) fails the
/// sweep, and so does a row that names no run of the suite.
/// - MISS 256: writes over a live mbarrier by atomics, cp.async and TMA are not checked.
/// - MISS 265: two STS.64 inits of one object by two warps are clean on SASS.
/// - MISS 266: an STS.64-initialized object revives after its inval (SASS).
/// - MISS 272: a poll whose result is discarded is a NOP on SASS.
const OPEN_MISSES: &[(&str, &str, &str, u32, &[&str])] = &[
    ("e_mbar_async_live", "ATOM", "ptx", 256, &["clean"]),
    ("e_mbar_async_live", "RED", "ptx", 256, &["clean"]),
    ("e_mbar_async_live", "CPASYNC", "ptx", 256, &["clean"]),
    ("e_mbar_async_live", "BULK", "ptx", 256, &["clean"]),
    ("e_mbar_async_live", "ATOM", "sass", 256, &["clean"]),
    ("e_mbar_async_live", "RED", "sass", 256, &["clean"]),
    ("e_mbar_async_live", "CPASYNC", "sass", 256, &["clean"]),
    ("e_mbar_async_live", "BULK", "sass", 256, &["clean"]),
    ("e_mbar_double_sts_init", "good", "sass", 265, &["clean"]),
    ("e_mbar_double_sts_init", "PREINV", "sass", 265, &["clean"]),
    ("e_mbar_resurrect", "INV2", "sass", 266, &["clean"]),
    ("e_mbar_discarded_poll", "good", "sass", 272, &["clean"]),
    ("e_mbar_discarded_poll", "INV", "sass", 272, &["clean"]),
    ("e_mbar_inval_vis", "TWAIT", "sass", 272, &["clean"]),
    ("e_mbar_inval_vis", "TWAITS", "sass", 272, &["clean"]),
    ("e_mbar_inval_vis", "TRYW", "sass", 272, &["clean"]),
    ("e_mbar_inval_vis", "TRYWS", "sass", 272, &["clean"]),
    ("e_mbar_resurrect", "WAIT", "sass", 272, &["clean"]),
];

/// One certificate run of the fixture suite: a variant through one front-end and its expected
/// (exhaustive) classes.
type SuiteJob = (Arc<Case>, String, &'static str, Vec<String>);

/// The parts of `certified_verdicts_match_the_fixture_suite`, each a test of its own, so that
/// the suite's lanes (8 threads, 50G) run them apart: part k < CERT_PARTS takes the jobs whose
/// index in the sorted job list, without those of CERT_OWN, is k modulo CERT_PARTS; part
/// CERT_PARTS + g runs the jobs of CERT_OWN[g], all at once.
const CERT_PARTS: usize = 2;

/// The fewest jobs a part k < CERT_PARTS may have (a part cannot silently go empty).
const MIN_JOBS: usize = 800;

/// The sweep's slowest jobs (case, variant, front), in parts of their own, eight each, which a
/// lane's eight workers run at once: the r4c_* SASS mutants, whose certificate attempts take 110
/// to 320 s each before they are refused, and r3b_trtllm_moe's fixed, B71 and B72 on SASS (about
/// 170 s and 11 GB each).
const CERT_OWN: &[&[(&str, &str, &str)]] = &[
    &[
        ("r4c_nowrap", "B33", "sass"),
        ("r4c_dwrap", "B43", "sass"),
        ("r4c_dwrap", "B41", "sass"),
        ("r4c_chunk2", "B33", "sass"),
        ("r4c_dwrap", "B42", "sass"),
        ("r4c_chunk2", "B43", "sass"),
        ("r4c_chunk2", "B44", "sass"),
        ("r3b_trtllm_moe", "fixed", "sass"),
    ],
    &[
        ("r4c_dwrap", "B33", "sass"),
        ("r4c_nowrap", "B43", "sass"),
        ("r4c_nowrap", "B37", "sass"),
        ("r4c_chunk2", "B36", "sass"),
        ("r4c_nowrap", "B42", "sass"),
        ("r4c_chunk2", "B42", "sass"),
        ("r3b_trtllm_moe", "B71", "sass"),
        ("r3b_trtllm_moe", "B72", "sass"),
    ],
];

/// Every fixture variant on both fronts (the cubin fixtures on SASS only): a certificate that
/// holds gives the explorer's classes, complete; a refusal gives a reason; an `unlifted`
/// variant keeps failing to load with a front-end's refusal; the variants of MUST_CERTIFY
/// certify. This is part `part` of the jobs (CERT_PARTS, CERT_OWN).
fn certified_verdicts_match_the_fixture_suite(part: usize) {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root().join("fixtures"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("case.toml").exists())
        .collect();
    dirs.sort();
    let mut all: Vec<SuiteJob> = Vec::new();
    let mut absent = Vec::new();
    for d in dirs {
        let name = d.file_name().unwrap().to_str().unwrap().to_string();
        // A vendor fixture that is not here has no runs (and no MUST_CERTIFY check).
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
                all.push((case.clone(), v.clone(), front, expect));
            }
        }
    }
    let in_own = |g: &[(&str, &str, &str)], c: &str, v: &str, f: &str| {
        g.iter().any(|&(a, b, d)| a == c && b == v && d == f)
    };
    let mut jobs: Vec<SuiteJob> = if part < CERT_PARTS {
        all.iter()
            .filter(|(c, v, f, _)| !CERT_OWN.iter().any(|g| in_own(g, &c.m.name, v, f)))
            .enumerate()
            .filter(|(k, _)| k % CERT_PARTS == part)
            .map(|(_, j)| j.clone())
            .collect()
    } else {
        // Vendor fixtures whose kernel binaries are absent (the open-source export) are
        // skipped, as everywhere in the sweep.
        let g = CERT_OWN[part - CERT_PARTS];
        let gone: Vec<_> = g
            .iter()
            .filter(|(c, _, _)| absent.iter().any(|a| a == c))
            .collect();
        if !gone.is_empty() {
            eprintln!(
                "part {part}: skipping {} of its {} jobs, their vendor kernel binaries are absent \
                 (fixtures/VENDOR_FIXTURES.toml): {gone:?}",
                gone.len(),
                g.len()
            );
        }
        all.iter()
            .filter(|(c, v, f, _)| in_own(g, &c.m.name, v, f))
            .cloned()
            .collect()
    };
    // PTXV_WELLSYNC_ONLY=a,b: only the runs whose `case/variant/front` contains one of them
    // (a debugging aid; MUST_CERTIFY is checked for the names it keeps).
    let only: Option<Vec<String>> = std::env::var("PTXV_WELLSYNC_ONLY")
        .ok()
        .map(|s| s.split(',').map(str::to_string).collect());
    let kept = |n: &str| {
        only.as_ref()
            .is_none_or(|o| o.iter().any(|x| n.contains(x.as_str())))
    };
    jobs.retain(|(c, v, f, _)| kept(&format!("{}/{v}/{f}", c.m.name)));
    let (certified, refused, unlifted) = (
        std::sync::Mutex::new(Vec::new()),
        std::sync::atomic::AtomicUsize::new(0),
        std::sync::atomic::AtomicUsize::new(0),
    );
    let run = |(case, v, front, expect): &SuiteJob| -> Result<(), String> {
        // `unlifted`: the front-end refuses the variant by design (a lifter coverage target,
        // tests/fixtures.rs r4_kernel_classes; a form the PTX front-end does not model, MISS
        // 160): it must keep refusing, with a front-end's refusal (FRONT_REFUSALS), not any
        // load error (a find text that no longer matches, a missing file). Any other variant
        // must load.
        let m = match try_machine(case, v, front) {
            Ok(_) if *expect == ["unlifted"] => {
                return Err("loads now; update its `unlifted` expectation".into());
            }
            Ok(m) => m,
            Err(e) if *expect == ["unlifted"] => {
                let why = format!("{e:#}");
                if !FRONT_REFUSALS.iter().any(|r| why.contains(r)) {
                    return Err(format!(
                        "expects `unlifted`, but does not load for another reason: {why}"
                    ));
                }
                unlifted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(());
            }
            Err(e) => return Err(format!("does not load: {e:#}")),
        };
        let o = certify(m);
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
        let open = OPEN_MISSES
            .iter()
            .find(|r| (r.0, r.1, r.2) == (case.m.name.as_str(), v.as_str(), *front));
        match o.verdict {
            Some(verdict) if open.is_some() => {
                let (.., miss, pin) = open.unwrap();
                if verdict.complete && verdict.classes() == *pin {
                    Ok(())
                } else {
                    Err(format!(
                        "open MISS {miss}: certified verdict {:?} (complete: {}), pinned {pin:?} \
                         (expected when fixed: {expect:?}; then drop its OPEN_MISSES row)",
                        verdict.classes(),
                        verdict.complete
                    ))
                }
            }
            Some(verdict) if !verdict.complete || !same(verdict.classes()) => Err(format!(
                "certified verdict {:?} (complete: {}) differs from the exhaustive one \
                 {expect:?}: {:#?}",
                verdict.classes(),
                verdict.complete,
                verdict.findings
            )),
            Some(_) => {
                certified
                    .lock()
                    .unwrap()
                    .push((case.m.name.clone(), v.clone(), *front));
                Ok(())
            }
            None if open.is_some() => Err(format!(
                "open MISS {}: refused ({:?}), pinned {:?}",
                open.unwrap().3,
                o.reasons,
                open.unwrap().4
            )),
            None if o.reasons.is_empty() => Err("refused without a reason".into()),
            // A refusal at the memory budget did not check the run (RMK277 F3: a latched budget
            // refused every later certificate of the process at step 0, and the sweep counted
            // them as refusals).
            None if o.reasons.iter().any(|r| r.contains("memory budget")) => {
                Err(format!("refused at the memory budget: {:?}", o.reasons))
            }
            None => {
                refused.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
        }
    };
    // Independent single runs, as many at once as the run has explorer threads: PTXV_THREADS
    // (a suite lane: 8 threads, 50G), else 8, not every core. A run past
    // PTXV_WELLSYNC_JOB_SECS (default 1800 s; 0: no limit; single runs take up to about 345 s on
    // 8 cores) fails the test: it would otherwise hold the tier for hours without a verdict.
    // PTXV_WELLSYNC_TRACE=1 prints every run's time.
    let workers = std::env::var("PTXV_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or(8);
    let limit = match std::env::var("PTXV_WELLSYNC_JOB_SECS") {
        Ok(s) => s.parse::<u64>().expect("PTXV_WELLSYNC_JOB_SECS: seconds"),
        Err(_) => 1800,
    };
    let errors = run_sweep(
        &jobs,
        workers,
        &|(case, v, front, _)| format!("{}/{v}/{front}", case.m.name),
        &run,
        (limit > 0).then(|| std::time::Duration::from_secs(limit)),
        &|msg| {
            use std::io::Write;
            let _ = std::io::stderr().write_all(
                format!("certified_verdicts_match_the_fixture_suite part {part} FAILED: {msg}\n")
                    .as_bytes(),
            );
            std::process::exit(101);
        },
        std::env::var_os("PTXV_WELLSYNC_TRACE").is_some(),
    );
    let mut errors = errors;
    let (certified, refused) = (certified.into_inner().unwrap(), refused.into_inner());
    eprintln!(
        "part {part}: {} of {} jobs, {} certified, {refused} refused (explored instead), {} \
         unlifted as expected",
        jobs.len(),
        all.len(),
        certified.len(),
        unlifted.into_inner()
    );
    // A part cannot silently go empty, nor lose a job of its own (unless PTXV_WELLSYNC_ONLY).
    if only.is_none() {
        // Every OPEN_MISSES row names a run of the suite (any part).
        for r in OPEN_MISSES {
            if !all
                .iter()
                .any(|(c, v, f, _)| (c.m.name.as_str(), v.as_str(), *f) == (r.0, r.1, r.2))
            {
                errors.push(format!("OPEN_MISSES row {r:?} names no run of the suite"));
            }
        }
        if part < CERT_PARTS {
            if jobs.len() < MIN_JOBS {
                errors.push(format!("part {part}: only {} jobs", jobs.len()));
            }
        } else {
            let g = CERT_OWN[part - CERT_PARTS];
            let present = g
                .iter()
                .filter(|(c, _, _)| !absent.iter().any(|a| a == c))
                .count();
            if jobs.len() != present {
                errors.push(format!("part {part}: a job of {g:?} is no longer one"));
            }
        }
    }
    // The guard on the certificate's strength, reported with the other failures. Every part
    // checks that each entry is still a job of the sweep (a renamed variant, a case in SKIP or
    // a typo fails); the part that runs it checks that it certifies. Only absent vendor
    // fixtures and the names PTXV_WELLSYNC_ONLY leaves out are not checked.
    for (c, v, f) in MUST_CERTIFY {
        let name = format!("{c}/{v}/{f}");
        if absent.iter().any(|a| a == c) || !kept(&name) {
            continue;
        }
        let is = |(a, b, g, _): &SuiteJob| a.m.name == *c && b == v && g == f;
        if !all.iter().any(is) {
            errors.push(format!("{name}: is no job of the sweep (MUST_CERTIFY)"));
        } else if jobs.iter().any(is)
            && !certified.iter().any(|(a, b, g)| a == c && b == v && g == f)
        {
            errors.push(format!("{name}: no longer certifies (MUST_CERTIFY)"));
        }
    }
    assert!(errors.is_empty(), "part {part}: {}", errors.join("\n"));
}

/// MISS 277: a step that reports a violation and leaves the machine as it was (an mbarrier poll
/// on an object that is not live, `mbar-uninit`: the thread stays at the poll) is picked again
/// and again by the certificate's deterministic run, which had no bound but the whole check's
/// time budget: with none set it never ended and its findings grew without limit
/// (certified_verdicts_match_the_fixture_suite_part0 killed at 16G). Each job must come back
/// within the deadline, its certificate refused or giving the fixture's classes; the test
/// stops the process (exit 101) at the deadline or once the resident set grew by 3 GiB, so it
/// cannot run out of memory itself.
#[test]
fn certificate_run_stops_a_step_that_repeats_without_progress() {
    const JOBS: &[(&str, &str)] = &[
        ("e_mbar_discarded_poll", "good"),
        ("e_mbar_discarded_poll", "INV"),
        ("e_mbar_discarded_poll", "USED"),
        ("e_mbar_inval_vis", "TRYWS"),
        ("e_mbar_inval_vis", "TWAITS"),
        ("e_mbar_resurrect", "WAIT"),
    ];
    for &(name, v) in JOBS {
        let case = Case::load(root().join("fixtures").join(name)).unwrap();
        let expect = case.variant(v).unwrap().expect_for("ptx");
        let m = machine(&case, v, "ptx");
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(certify(m));
        });
        let base = ptxv_sim::membudget::rss();
        let t0 = std::time::Instant::now();
        let o = loop {
            match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(o) => break o,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(e) => panic!("{name}/{v}/ptx: {e}"),
            }
            let grown = ptxv_sim::membudget::rss().saturating_sub(base);
            if t0.elapsed().as_secs() >= 120 || grown >= 3 << 30 {
                // Past the test harness's capture (exit skips its output).
                use std::io::Write;
                let _ = std::io::stderr().write_all(
                    format!(
                        "certificate_run_stops_a_step_that_repeats_without_progress FAILED: \
                         {name}/{v}/ptx: no certificate result after {:.0} s, resident set \
                         grown by {:.1} GiB\n",
                        t0.elapsed().as_secs_f64(),
                        grown as f64 / (1u64 << 30) as f64
                    )
                    .as_bytes(),
                );
                std::process::exit(101);
            }
        };
        match o.verdict {
            Some(ver) => assert_eq!(ver.classes(), expect, "{name}/{v}/ptx: certified"),
            None => assert!(
                !o.reasons.is_empty(),
                "{name}/{v}/ptx: refused without a reason"
            ),
        }
    }
}

#[test]
#[ignore = "heavy: half of the certify sweep's other jobs; 90 s"]
fn certified_verdicts_match_the_fixture_suite_part0() {
    certified_verdicts_match_the_fixture_suite(0);
}

#[test]
#[ignore = "heavy: half of the certify sweep's other jobs; 76 s"]
fn certified_verdicts_match_the_fixture_suite_part1() {
    certified_verdicts_match_the_fixture_suite(1);
}

#[test]
#[ignore = "heavy: eight of the certify sweep's slowest jobs at once; 330 s"]
fn certified_verdicts_match_the_fixture_suite_part2() {
    certified_verdicts_match_the_fixture_suite(2);
}

#[test]
#[ignore = "heavy: eight of the certify sweep's slowest jobs at once; 320 s"]
fn certified_verdicts_match_the_fixture_suite_part3() {
    certified_verdicts_match_the_fixture_suite(3);
}

/// E-WELLSYNC cross-check (`PTXV_CERT_CROSSCHECK`): random schedules never refute a
/// certificate the fixture suite relies on (no false refutation: every finding, the
/// termination and the outputs of each random run agree with the certified run).
#[test]
fn cert_crosscheck_keeps_certified_fixtures() {
    let jobs: Vec<_> = MUST_CERTIFY
        .iter()
        .filter(|(c, _, _)| !c.starts_with("r3b_") && !c.starts_with("r2_"))
        .filter(|(c, _, _)| !vendor_missing(&[c]))
        .collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let errors = std::sync::Mutex::new(Vec::new());
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get().min(16));
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((c, v, f)) = jobs.get(i) else {
                        break;
                    };
                    let case = Case::load(root().join("fixtures").join(c)).unwrap();
                    let o = ptxv_sim::wellsync::certify_crosschecked(machine(&case, v, f), 4);
                    if o.verdict.is_none() {
                        errors
                            .lock()
                            .unwrap()
                            .push(format!("{c}/{v}/{f}: {:?}", o.reasons));
                    }
                }
            });
        }
    });
    let errors = errors.into_inner().unwrap();
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

/// Cross-check and E-RACE-DEAD: a race the certificate marks `race-dead` (after its run,
/// race_dead::mark) is the certified race when a random schedule reports it unmarked, and from
/// the other side (the read first: `read-write` where the certified run had `write-read`, a
/// signature naming the other pc). Before, every walk refuted the good variants of both
/// E-RACE-DEAD fixtures on both front-ends.
#[test]
fn cert_crosscheck_race_dead_is_the_certified_race() {
    for c in ["e_race_dead", "e_race_dead_tmap"] {
        let case = Case::load(root().join("fixtures").join(c)).unwrap();
        for front in ["ptx", "sass"] {
            let o = ptxv_sim::wellsync::certify_crosschecked(machine(&case, "good", front), 4);
            let v = o
                .verdict
                .unwrap_or_else(|| panic!("{c} {front}: {:#?}", o.reasons));
            assert!(
                v.findings.iter().any(|f| f.kind == "race-dead"),
                "{c} {front}"
            );
        }
    }
}

/// MISS 52's false deadlock, gone with MISS 81 (probes/warpsync_pairs good, SASS): a schedule
/// that runs the even lanes' `VOTEU.ANY UR5` between the odd lanes' own VOTEU and the `LOP3`
/// that reads UR5 (an STS sits between them) sent the odd lanes past `@!P0 BRA.DIV` to
/// `BSYNC B0`, where they waited for the even lanes parked in `WARPSYNC.COLLECTIVE`. That
/// BSYNC has no .RECONVERGENT, so since MISS 81 it holds no lane and no walk deadlocks there;
/// the certificate refuses instead, at the strict level and under the GB300 profile (C-WARP:
/// the lanes leave the BSYNC apart and meet again at the next uniform instructions).
#[test]
fn cert_crosscheck_deadlocking_schedule_refutes() {
    let case = Case::load(root().join("probes/warpsync_pairs")).unwrap();
    let m = machine(&case, "good", "sass");
    let g = ptxv_sim::hwprofile::machine(&m, ptxv_sim::hwprofile::GB300);
    for (level, m) in [("strict", m), ("gb300", g)] {
        let o = ptxv_sim::wellsync::certify_crosschecked(m, 4);
        assert!(o.verdict.is_none(), "{level}");
        assert!(
            !o.reasons.is_empty() && o.reasons.iter().all(|r| r.starts_with("C-WARP")),
            "{level}: {:#?}",
            o.reasons
        );
        assert!(
            !o.witnesses.iter().any(|f| f.kind == "deadlock"),
            "{level}: {:#?}",
            o.witnesses
        );
    }
}

/// MISS 52 under the GB300 profile: a split inside a uniform-register window runs to its read
/// before another split of its warp moves (E-HW-SPLIT-UR, urwin.rs; probes/ur_window and
/// probes/split_yield: no switch without a YIELD or a blocking instruction). On
/// probes/warpsync_pairs good, SASS, the windows after each `VOTEU.ANY UR5` reach the `LOP3`
/// that reads it across the STS between them. Until MISS 81 the certificate then held under the
/// cross-check, as GB300 (bitwise); since then it refuses at both levels
/// (`cert_crosscheck_deadlocking_schedule_refutes`), and the windows are what is checked here.
#[test]
fn cert_crosscheck_warpsync_pairs_certifies_under_gb300() {
    let case = Case::load(root().join("probes/warpsync_pairs")).unwrap();
    let m = machine(&case, "good", "sass");
    let w = ptxv_sim::urwin::windows(&m.k);
    // The divergent fast paths' VOTEU windows (an STS between the write and the read).
    let votes: Vec<usize> = (0..m.k.code.len())
        .filter(|&pc| m.k.code[pc].text.contains("VOTEU.ANY"))
        .collect();
    assert!(votes.len() >= 4, "{votes:?}");
    for &pc in &votes {
        assert!(w[pc + 1] && w[pc + 2], "window after {}", m.k.code[pc].text);
    }
    let g = ptxv_sim::hwprofile::machine(&m, ptxv_sim::hwprofile::GB300);
    assert!(g.launch.hw_split_ur);
}

/// Cross-check: a walk cut before it terminates proves nothing by itself; the deterministic
/// scheduler continues it, and the certificate holds when the continuation ends like the
/// certified run (a note on the verdict). A continuation that does not terminate within its
/// budget refutes. Before, any walk still running at its cap refuted the certificate.
#[test]
fn cert_crosscheck_cut_walks_are_continued() {
    let case = Case::load(root().join("probes/warpsync_pairs")).unwrap();
    let m = machine(&case, "good", "ptx");
    let o = ptxv_sim::wellsync::certify_crosschecked_capped(m.clone(), 4, Some(10), None);
    let v = o.verdict.unwrap_or_else(|| panic!("{:#?}", o.reasons));
    let cut = v
        .notes
        .iter()
        .filter(|n| n.contains("cut at 10 steps"))
        .count();
    assert_eq!(cut, 4, "{:#?}", v.notes);
    let o = ptxv_sim::wellsync::certify_crosschecked_capped(m, 4, Some(10), Some(10));
    assert!(o.verdict.is_none());
    assert!(
        o.reasons
            .iter()
            .any(|r| r.contains("within 10 more deterministic steps")),
        "{:#?}",
        o.reasons
    );
}

/// Small kernels: the certified outputs are the explorer's (every terminal state).
#[test]
fn certified_outputs_match_exploration() {
    for (c, v, f) in [
        ("r0_tma_sum", "good", "ptx"),
        ("r0_tma_sum", "good", "sass"),
        ("r1a_tcgen05_tile", "good", "ptx"),
        // (r1a_tcgen05_tile SASS: refused since MISS 81, see MUST_CERTIFY.)
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
///
/// Rung 4a (docs/rung4.md, formerly also fixtures.rs `r4a_first_call_exhaustive`, which ran
/// three of these four explorations again): one call of torch's one-shot and two-shot
/// all-reduce on 3 GPUs, captured by tools/capture_dist.py, explored exhaustively, is clean
/// with one output; the one-shot outputs equal every rank's GPU bytes, with and without
/// E-EXIT-FENCE.
#[test]
#[ignore = "heavy: 4 certify + 4M-state explores; 193 s"]
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
            // The former r4a_first_call_exhaustive's checks: the one-shot with and without
            // E-EXIT-FENCE, the two-shot with it.
            let one = case.m.name == "r4a_oneshot";
            if one || !strict {
                assert_eq!(
                    ev.classes(),
                    ["clean"],
                    "{what} (strict: {strict}): {:#?}",
                    ev.findings
                );
            }
            if one {
                assert_eq!(ev.outputs.len(), 1, "{what} (strict: {strict})");
                let report = case.expect_report(&ev.outputs[0]).unwrap();
                assert!(
                    report.len() == 3 && report.iter().all(|l| l.contains("bitwise equal")),
                    "{what} (strict: {strict}): {report:?}"
                );
            }
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
#[ignore = "heavy: 38 s"]
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

/// C-FADD in shared memory (MISS 142): two warps' unordered `red.shared.add.f32` of the
/// subnormals 3 and 5 (x 2^-149) into one +0 word. A shared-memory add keeps subnormals, so
/// both orders leave 8, as the run does, and the certificate holds with that output; the fold
/// over the orders adds as the executor does (a flushing fold would leave 0 in both orders and
/// refuse it). Before, the executor flushed too and the output was 0 (GB300: 8, N-ATOMIC).
#[test]
fn fadd_orders_in_shared_memory_keep_subnormals() {
    let ptx = ".version 8.8\n.target sm_103a\n.address_size 64\n\
               .visible .entry k(.param .u64 out) {\n\
               .reg .b32 %r<8>; .reg .f32 %f<2>; .reg .b64 %rd<2>; .reg .pred %p<4>;\n\
               .shared .align 4 .b32 s[1];\n\
               mov.u32 %r1, %tid.x; mov.u32 %r2, s; setp.eq.u32 %p1, %r1, 0;\n\
               @%p1 st.shared.u32 [%r2], 0; bar.sync 0;\n\
               and.b32 %r3, %r1, 31; setp.ne.u32 %p2, %r3, 0; @%p2 bra SKIP;\n\
               setp.lt.u32 %p3, %r1, 32; selp.b32 %r5, 3, 5, %p3; mov.b32 %f1, %r5;\n\
               red.shared.add.f32 [%r2], %f1;\n\
               SKIP: bar.sync 0;\n\
               ld.param.u64 %rd1, [out]; cvta.to.global.u64 %rd1, %rd1;\n\
               @%p1 ld.shared.u32 %r6, [%r2]; @%p1 st.global.u32 [%rd1], %r6; ret; }";
    let k = ptxv_ptx::parse_kernel(ptx, Some("k")).unwrap();
    let case = Case {
        dir: Default::default(),
        m: serde_json::from_value(serde_json::json!({
            "name":"fadd_shared", "description":"C-FADD in shared memory",
            "entry":"k", "grid":[1,1,1], "block":[64,1,1], "outputs":["out"],
            "buffers":[{"name":"out", "dtype":"u32", "shape":[1], "init":"zeros"}],
            "params":[{"kind":"ptr", "buffer":"out"}],
            "variants":[]
        }))
        .unwrap(),
    };
    let launch = build_launch(&case, &k).unwrap();
    let o = certify(Machine::new(Arc::new(k), Arc::new(launch)));
    let v = o
        .verdict
        .unwrap_or_else(|| panic!("not certified: {:#?}", o.reasons));
    assert_eq!(v.classes(), ["clean"], "{:#?}", v.findings);
    assert_eq!(v.outputs.len(), 1);
    assert_eq!(v.outputs[0][..4], 8u32.to_le_bytes(), "{:?}", v.outputs);
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

/// MISS 30 (C-WARP, found by the adversarial harness; GB300: 16 words differ in run 0 of
/// every process): lanes that leave a loop after lane-dependent trip counts and are never
/// reconverged (the BSYNC NOPed) were merged by the model at the next pc in the certificate's
/// run; the merged split's lane-local instruction gave them one token, so the following STS /
/// LDS exchange looked joined in every schedule. The certificate must refuse NOBSYNC and
/// NOBSSY, which the explorer finds nondeterministic; good (ptxas's BSSY / BSYNC) certifies.
#[test]
fn merged_splits_are_not_converged() {
    let case = Case::load(root().join("fixtures/e_lockstep_nobsync")).unwrap();
    for v in ["good", "NOBSYNC", "NOBSSY"] {
        let o = certify(machine(&case, v, "sass"));
        match (v, o.verdict) {
            ("good", Some(r)) => assert_eq!(r.classes(), ["clean"]),
            ("good", None) => panic!("good does not certify: {:?}", o.reasons),
            (_, Some(r)) => panic!("{v} certified: {:?}", r.classes()),
            (_, None) => assert!(
                o.reasons.iter().any(|r| r.contains("C-WARP")),
                "{v}: {:?}",
                o.reasons
            ),
        }
        let e = ptxv_sim::explore::explore(machine(&case, v, "sass"), 1_000_000);
        let want: Vec<String> = case.variant(v).unwrap().expect.clone();
        assert!(e.complete && e.classes() == want, "{v}: {:?}", e.classes());
    }
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Regression: every fixture variant must get exactly its expected verdict classes, through
//! both front-ends. A mutant that stops being caught fails CI.

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::explore::explore;
use ptxv_sim::machine::Machine;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Third-party kernel binaries are not distributed (fixtures/VENDOR_FIXTURES.toml): true when
/// one these fixtures need is missing (the test then skips).
fn vendor_missing(needs: &[&str]) -> bool {
    ptxv_case::skip_missing_vendor_files(&root().join("fixtures"), needs)
}

fn kernel(case: &Case, variant: &str, front: &str) -> ptxv_ir::Kernel {
    let ptx = case.variant_ptx(variant).unwrap();
    if !case.m.launches.is_empty() {
        // Multi-kernel launch: one combined kernel.
        let ks = case
            .m
            .all_launches()
            .iter()
            .map(|l| match front {
                "ptx" => ptxv_ptx::parse_kernel(&ptx, Some(l.entry)).unwrap(),
                _ => {
                    let dir =
                        std::env::temp_dir().join(format!("ptxv-test-{}-{variant}", case.m.name));
                    std::fs::create_dir_all(&dir).unwrap();
                    let p = dir.join("k.ptx");
                    std::fs::write(&p, &ptx).unwrap();
                    let c = dir.join("k.cubin");
                    ptxv_sass::ptxas(&p, &c, &case.m.arch).unwrap();
                    ptxv_sass::load_cubin(&c, l.entry).unwrap()
                }
            })
            .collect();
        return ptxv_ir::Kernel::concat(ks);
    }
    match front {
        "ptx" => ptxv_ptx::parse_kernel(&ptx, Some(&case.m.entry)).unwrap(),
        _ => {
            let dir = std::env::temp_dir().join(format!("ptxv-test-{}-{variant}", case.m.name));
            std::fs::create_dir_all(&dir).unwrap();
            let p = dir.join("k.ptx");
            std::fs::write(&p, &ptx).unwrap();
            let c = dir.join("k.cubin");
            ptxv_sass::ptxas(&p, &c, &case.m.arch).unwrap();
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

/// Does every output's exact dataflow value equal the independent spec?
fn spec_match(case: &str, variant: &str, front: &str) -> bool {
    use ptxv_sim::exact;
    let case = Case::load(root().join("fixtures").join(case)).unwrap();
    let k = kernel(&case, variant, front);
    let spec = ptxv_sim::spec_exact(&case)
        .unwrap()
        .expect("case has a spec");
    let launch = build_launch(&case, &k).unwrap();
    let mut m = Machine::new(Arc::new(k), Arc::new(launch));
    m.num = Some(Box::new(ptxv_sim::num::NumState::with_mma_eps(
        m.launch.mma_model.eps(),
    )));
    let (m, _, ok) = ptxv_sim::explore::run_once(m);
    assert!(ok);
    let ns = m.num.as_ref().unwrap();
    let oi = m.launch.outputs[0];
    (0..spec.len()).all(|i| {
        let node = ns
            .mem_node(ptxv_sim::exec::Loc::Global {
                alloc: oi,
                off: (i * 4) as u64,
            })
            .unwrap();
        exact::is_zero(&exact::sub(&ns.nodes[node as usize].exact, &spec[i]))
    })
}

fn check_dir(dir: &str, case: &str, variant: &str, front: &str) {
    let case = Case::load(root().join(dir).join(case)).unwrap();
    let k = kernel(&case, variant, front);
    let launch = build_launch(&case, &k).unwrap();
    let m0 = Machine::new(Arc::new(k), Arc::new(launch));
    let v = explore(m0.clone(), 2_000_000);
    assert!(v.complete, "{variant}/{front}: exploration incomplete");
    assert_eq!(
        v.classes(),
        case.variant(variant).unwrap().expect_for(front),
        "{variant}/{front}: {:#?}",
        v.findings
    );
    // Cross-check: partial-order reduction must reach the same verdict and outputs.
    let d = ptxv_sim::dpor::explore_dpor(m0, 2_000_000);
    assert!(d.complete, "{variant}/{front}: DPOR incomplete");
    // A race left after the strong re-check (S-SMEM-STRONG, e.g. a TMA write against an LDS):
    // reductions are exact for race-free kernels only (E-DRF), so the explorers must agree
    // on the race, not on the outcomes it leaves.
    let rechecked =
        |v: &ptxv_sim::explore::Verdict| v.notes.iter().any(|n| n.contains("S-SMEM-STRONG"));
    if rechecked(&v) && v.classes().iter().any(|c| c == "race") {
        let drop_nd = |c: Vec<String>| -> Vec<String> {
            c.into_iter().filter(|x| x != "nondeterministic").collect()
        };
        assert_eq!(
            drop_nd(d.classes()),
            drop_nd(v.classes()),
            "{variant}/{front}: DPOR verdict differs"
        );
        return;
    }
    let mut a: Vec<_> = v.outputs.clone();
    let mut b: Vec<_> = d.outputs.clone();
    a.sort();
    b.sort();
    assert_eq!(
        d.classes(),
        v.classes(),
        "{variant}/{front}: DPOR verdict differs: {:#?}",
        d.findings
    );
    assert!(
        a == b,
        "{variant}/{front}: DPOR reached different outputs ({} vs {})",
        b.len(),
        a.len()
    );
    eprintln!(
        "{variant}/{front}: stateful {} states, DPOR {} states",
        v.states, d.states
    );
}

fn check(case: &str, variant: &str, front: &str) {
    check_dir("fixtures", case, variant, front)
}

#[test]
fn r0_good() {
    check("r0_tma_sum", "good", "ptx");
    check("r0_tma_sum", "good", "sass");
}
#[test]
fn r0_b01_expect_tx_deadlock() {
    check("r0_tma_sum", "B01", "ptx");
    check("r0_tma_sum", "B01", "sass");
}
#[test]
fn r0_b02_wrong_parity() {
    check("r0_tma_sum", "B02", "sass");
}
#[test]
#[ignore = "slow (~1 min): PTX front-end explores 29k states"]
fn r0_b02_wrong_parity_ptx() {
    check("r0_tma_sum", "B02", "ptx");
}
#[test]
fn r0_b03_init_race() {
    check("r0_tma_sum", "B03", "ptx");
    check("r0_tma_sum", "B03", "sass");
}

fn both(case: &str, variant: &str) {
    check(case, variant, "ptx");
    check(case, variant, "sass");
}

#[test]
fn r1a_good() {
    both("r1a_tcgen05_tile", "good");
}
#[test]
fn r1a_b05_no_wait_ld() {
    both("r1a_tcgen05_tile", "B05");
}
#[test]
fn r1a_b06_no_tc_fence() {
    both("r1a_tcgen05_tile", "B06");
}
#[test]
fn r1a_b07_dealloc_race() {
    both("r1a_tcgen05_tile", "B07");
}
#[test]
fn r1a_b41_tmem_leak() {
    both("r1a_tcgen05_tile", "B41");
}
#[test]
fn r1a_b42_acc_uninit() {
    both("r1a_tcgen05_tile", "B42");
}
#[test]
fn r1a_b16_partial_warp_alloc() {
    both("r1a_tcgen05_tile", "B16");
}
#[test]
fn r1a_b43_commit_wrong_mbar() {
    both("r1a_tcgen05_tile", "B43");
}

#[test]
fn calibrated_mma_model_matches_hardware_probe() {
    // One recorded GB300 output (probe_align style): 1 + fifteen products of 2^-30 truncate
    // away entirely under the calibrated model (25 bits kept below the largest exponent).
    let m = ptxv_sim::fp::DEFAULT_MMA;
    let mut terms = vec![ptxv_sim::fp::Term::prod(1.0, 1.0, true)];
    for _ in 0..15 {
        terms.push(ptxv_sim::fp::Term::prod(2f64.powi(-30), 1.0, true));
    }
    assert_eq!(m.apply_terms(0.0, &terms), 1.0);
}
#[test]
fn r1a_b44_sass_missing_wait() {
    both("r1a_tcgen05_tile", "B44");
}
#[test]
fn r1a_b45_sass_missing_scoreboard() {
    both("r1a_tcgen05_tile", "B45");
}

/// ASSUME(N-MMA) is enforced: inputs outside the calibrated domain make the verdict
/// inconclusive ("unmodeled") instead of silently producing a modeled answer.
#[test]
fn r1a_mma_outside_calibrated_domain_is_unmodeled() {
    let mut case = Case::load(root().join("fixtures").join("r1a_tcgen05_tile")).unwrap();
    case.m
        .buffers
        .iter_mut()
        .find(|b| b.name == "A")
        .unwrap()
        .init = "nan".into();
    let k =
        ptxv_ptx::parse_kernel(&case.variant_ptx("good").unwrap(), Some(&case.m.entry)).unwrap();
    let launch = build_launch(&case, &k).unwrap();
    let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 2_000_000);
    assert_eq!(
        v.classes(),
        vec!["unmodeled".to_string()],
        "{:#?}",
        v.findings
    );
}

// ---- assumption fixtures and probes (ASSUMPTIONS.md) ----

#[test]
fn s_init_partial_producer() {
    both("s_init", "good");
    both("s_init", "B48");
}
#[test]
fn r0_b47_uninit_accumulator() {
    both("r0_tma_sum", "B47");
    // ptxas reused R0 (= %tid.x) as the undefined accumulator: the binary reads no unwritten
    // register, but its dataflow no longer matches the spec. Numerics catch what execution can't.
    assert!(spec_match("r0_tma_sum", "good", "sass"));
    assert!(!spec_match("r0_tma_sum", "B47", "sass"));
}
#[test]
fn e_elect_leader_choice() {
    both("e_elect", "good");
    both("e_elect", "B46");
}
/// A plain write followed by its own thread's atomic, then other threads' unordered atomics
/// (soundness-harness miss 1): the race is found by exploration and by the certificate run
/// (`wellsync.rs` `plain_write_hidden_by_own_atomic_is_a_race`).
#[test]
fn e_cert_fasttrack_plain_then_own_atomic() {
    both("e_cert_fasttrack", "good");
    // Stateful explorer only: DPOR also reports what the race leaves (an add before the init
    // reads unwritten shared memory), and reductions agree only on race-free kernels (E-DRF).
    check_stateful("e_cert_fasttrack", "nosync", "ptx", 100_000);
    check_stateful("e_cert_fasttrack", "nosync", "sass", 100_000);
}
/// bar.arrive / BAR.ARV (E-BAR-ARRIVE): release without waiting, count mismatch, re-arrival.
#[test]
fn e_bararv_named_barrier_arrive() {
    for v in ["good", "AB1", "AB3", "AB4"] {
        both("e_bararv", v);
    }
    // AB2 (no bar.sync at all): the stateful explorer reports the race in one order (E-DRF);
    // DPOR also reverses it, where the consumer reads shared memory nothing wrote yet (S-INIT
    // violation). Both must report every expected class (SASS: no race, S-SMEM-STRONG).
    for front in ["ptx", "sass"] {
        check_stateful("e_bararv", "AB2", front, 100_000);
        let case = Case::load(root().join("fixtures").join("e_bararv")).unwrap();
        let k = kernel(&case, "AB2", front);
        let launch = build_launch(&case, &k).unwrap();
        let d = ptxv_sim::dpor::explore_dpor(Machine::new(Arc::new(k), Arc::new(launch)), 100_000);
        let expect = case.variant("AB2").unwrap().expect_for(front);
        assert!(
            d.complete && expect.iter().all(|e| d.classes().contains(e)),
            "AB2/{front}: DPOR got {:?}, want at least {expect:?}",
            d.classes()
        );
    }
}
/// A named barrier used by more threads than its count (E-BAR-OVER): which warps share a
/// generation depends on arrival order, so those arrivals must not be commuted.
#[test]
fn e_barover_oversubscribed_named_barrier() {
    both("e_barover", "good");
    // As for e_bararv AB2: the stateful explorer reports each race in one order (E-DRF); DPOR
    // also reverses it (the consumer then reads shared memory nothing wrote yet: S-INIT
    // violation, and a second output). Both must find every expected class.
    for v in ["BO1", "BO2", "BO3"] {
        for front in ["ptx", "sass"] {
            check_stateful("e_barover", v, front, 100_000);
            let case = Case::load(root().join("fixtures").join("e_barover")).unwrap();
            let k = kernel(&case, v, front);
            let launch = build_launch(&case, &k).unwrap();
            let m0 = Machine::new(Arc::new(k), Arc::new(launch));
            let d = ptxv_sim::dpor::explore_dpor(m0, 100_000);
            let expect = case.variant(v).unwrap().expect_for(front);
            assert!(
                d.complete && expect.iter().all(|e| d.classes().contains(e)),
                "{v}/{front}: DPOR got {:?}, want at least {expect:?}",
                d.classes()
            );
        }
    }
}
/// The model agrees with GB300 on every probe variant (probes/bar_exit, run 2026-10-30).
#[test]
fn probe_bar_exit_matches_hardware() {
    for v in ["whole_warp", "half_warp", "counted", "control"] {
        check_dir("probes", "bar_exit", v, "ptx");
        check_dir("probes", "bar_exit", v, "sass");
    }
}
/// L-SB-ORDER (probes/sb_mio_order, GB300 2026-11-05): the static scoreboard check accepts the
/// windows where a later memory-pipe read scoreboard covers an earlier STS's address read
/// (0 of 1000 runs wrong, also with an LDG in between: quack rmsnorm_bwd N=256's shape) and
/// flags the two that fail on every run.
#[test]
fn probe_sb_mio_order_matches_hardware() {
    let case = Case::load(root().join("probes/sb_mio_order/fixture")).unwrap();
    for v in [
        "good",
        "inorder",
        "inorder_ldg",
        "rd_on_ldg",
        "rd_on_a",
        "nowait",
    ] {
        let k = kernel(&case, v, "sass");
        let launch = build_launch(&case, &k).unwrap();
        let r = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)));
        let verdict = r.verdict.expect("certifies");
        assert_eq!(
            verdict.classes(),
            case.variant(v).unwrap().expect_for("sass"),
            "{v}: {:#?}",
            verdict.findings
        );
    }
}
/// E-BAR-EXIT on partly exited warps: a counted barrier counts a warp's arrival as the warp
/// size, its exited or nonexistent lanes included (probes/bar_exit_warp, run 2026-11-05: half,
/// one-lane and two half warps complete `bar.sync 1, 64`, 16 lanes complete `bar.sync 1, 32`
/// alone, a 16-lane partial warp counts 32; 2 warps short of `bar.sync 1, 96` hang).
#[test]
fn probe_bar_exit_warp_matches_hardware() {
    for v in [
        "half",
        "both_half",
        "one_lane",
        "half_alone",
        "half_short",
        "half_full96",
        "half_arrive",
        "control",
    ] {
        check_dir("probes", "bar_exit_warp", v, "ptx");
        check_dir("probes", "bar_exit_warp", v, "sass");
    }
    check_dir("probes", "bar_exit_warp/partial", "full", "ptx");
    check_dir("probes", "bar_exit_warp/partial", "full", "sass");
}
/// The model agrees with GB300 on every probe variant (probes/bar_arrive, run 2026-11-02;
/// `mismatch` traps on hardware, reported as a violation).
#[test]
fn probe_bar_arrive_matches_hardware() {
    for v in ["arrive", "sync", "twice", "twice_alone", "mismatch"] {
        check_dir("probes", "bar_arrive", v, "ptx");
        check_dir("probes", "bar_arrive", v, "sass");
    }
}
/// E-SPIN on shared-memory flags: ptxas compiles the poll to a plain LDS loop, which used to
/// run inside one transition until the local-step limit (a racy one reported a race per
/// iteration). Blocked polls give bounded verdicts on both front-ends.
#[test]
fn e_spin_smem_flag_polls() {
    for v in ["good", "warp", "relaxed_fence", "SP1", "SP2", "SP4", "SP5"] {
        both("e_spin_smem", v);
    }
    check("e_spin_smem", "SP3", "ptx");
    // SASS SP3 has no poll left (ptxas deletes the plain spin) and is racy: DPOR also finds
    // the second output of the race, the stateful reduction need not (E-DRF). Stateful only.
    let case = Case::load(root().join("fixtures/e_spin_smem")).unwrap();
    let k = kernel(&case, "SP3", "sass");
    let launch = build_launch(&case, &k).unwrap();
    let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 2_000_000);
    assert!(v.complete);
    assert_eq!(
        v.classes(),
        case.variant("SP3").unwrap().expect_for("sass"),
        "{:#?}",
        v.findings
    );
}

#[test]
fn r0_b49_spin_with_side_effects() {
    both("r0_tma_sum", "B49");
}

/// Rung 1b is too large for the stateful explorer: DPOR alone, which `check_dir` has
/// cross-checked against it on every smaller fixture.
fn check_dpor(case: &str, variant: &str, front: &str, max_states: usize) {
    let case = Case::load(root().join("fixtures").join(case)).unwrap();
    let k = kernel(&case, variant, front);
    let launch = build_launch(&case, &k).unwrap();
    let m0 = Machine::new(Arc::new(k), Arc::new(launch));
    let d = ptxv_sim::dpor::explore_dpor(m0, max_states);
    let expect = case.variant(variant).unwrap().expect_for(front);
    assert!(
        d.meets(&expect),
        "{variant}/{front}: got {:?} (complete: {}), want {expect:?}: {:#?}",
        d.classes(),
        d.complete,
        d.findings
    );
}

#[test]
fn r1b_good() {
    check_dpor("r1b_pipeline", "good", "ptx", 2_000_000);
    check_dpor("r1b_pipeline", "good", "sass", 2_000_000);
}

#[test]
fn r1b_ptx_mutants() {
    for v in ["B04", "B11", "B50"] {
        check_dpor("r1b_pipeline", v, "ptx", 2_000_000);
    }
}

/// Bugs whose consequences blow up the state space: the expected classes must show up within
/// a bounded search (`Verdict::meets`).
#[test]
fn r1b_b09_ptx() {
    check_dpor("r1b_pipeline", "B09", "ptx", 20_000);
}
#[test]
fn r1b_b09_sass() {
    check_dpor("r1b_pipeline", "B09", "sass", 20_000);
}
#[test]
fn r1b_b10_ptx() {
    check_dpor("r1b_pipeline", "B10", "ptx", 20_000);
}
#[test]
fn r1b_b10_sass() {
    check_dpor("r1b_pipeline", "B10", "sass", 20_000);
}

// Rung 1c: two CTAs make the stateful explorer too slow (> 15 min); DPOR only, as for 1b.
#[test]
fn r1c_good() {
    check_dpor("r1c_cluster", "good", "ptx", 2_000_000);
    check_dpor("r1c_cluster", "good", "sass", 2_000_000);
}
#[test]
fn r1c_b13_cluster_lifetime() {
    check_dpor("r1c_cluster", "B13", "ptx", 2_000_000);
    check_dpor("r1c_cluster", "B13", "sass", 2_000_000);
}
#[test]
fn r1c_b14_no_mbarrier_init_fence() {
    check_dpor("r1c_cluster", "B14", "ptx", 2_000_000);
    check_dpor("r1c_cluster", "B14", "sass", 2_000_000);
}

// Rung 1c(ii): cta_group::2; the SASS front includes ptxas's paired-alloc handshake.
#[test]
fn r1c_2sm_good() {
    check_dpor("r1c_2sm", "good", "ptx", 2_000_000);
    check_dpor("r1c_2sm", "good", "sass", 2_000_000);
}
#[test]
fn r1c_2sm_b12_commit_mask() {
    check_dpor("r1c_2sm", "B12", "ptx", 2_000_000);
    check_dpor("r1c_2sm", "B12", "sass", 2_000_000);
}

// tcgen05.mma shapes measured with probes/tc_m128: cta_group::2 with M = 128 (PTX ISA Layout B:
// each CTA's 64 rows in lanes 0..63 / 64..127 for the two halves of N) and cta_group::1 M = 128
// with N = 8. Verdicts on both front-ends (explorer and DPOR), then the certified output against
// GB300's TMEM dump, bit for bit (sentinel cells included).
#[test]
fn e_mma_pair_m128_verdicts() {
    for v in ["good", "B12", "n8"] {
        both("e_mma_pair_m128", v);
    }
}
#[test]
fn e_mma_m128_n8_verdicts() {
    for v in ["good", "n4"] {
        both("e_mma_m128_n8", v);
    }
    // A race: the explorers agree on the verdict, not on the outcomes it leaves (E-DRF).
    check_stateful("e_mma_m128_n8", "B05", "ptx", 2_000_000);
    check_stateful("e_mma_m128_n8", "B05", "sass", 2_000_000);
}
// The PTX forms lifted for the mslk decode kernels: redux.sync (f32 and integer), neg.f32,
// setp.<cmp>.f32 (bitwise against GB300 in e_mma_m128_shapes_match_hardware).
#[test]
fn e_ptx_redux_verdicts() {
    both("e_ptx_redux", "good");
}
#[test]
fn e_mma_m128_shapes_match_hardware() {
    for case in ["e_mma_pair_m128", "e_mma_m128_n8", "e_ptx_redux"] {
        let case = Case::load(root().join("fixtures").join(case)).unwrap();
        for front in ["ptx", "sass"] {
            let k = kernel(&case, "good", front);
            let launch = build_launch(&case, &k).unwrap();
            let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
                .verdict
                .unwrap_or_else(|| panic!("{} {front} certifies", case.m.name));
            assert_eq!(v.classes(), ["clean"], "{:#?}", v.findings);
            let report = case.expect_report(&v.outputs[0]).unwrap();
            assert!(
                !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
                "{} {front}: {report:?}",
                case.m.name
            );
        }
    }
}

// Rung 1d: CLC. The launch/cancel dependency is new, so the good kernel is also cross-checked
// against the stateful explorer (PTX; 430k states, about a minute).
#[test]
fn r1d_good() {
    check_dir("fixtures", "r1d_clc", "good", "ptx");
    check_dpor("r1d_clc", "good", "sass", 2_000_000);
}
#[test]
fn r1d_b17_unwaited_response() {
    check_dpor("r1d_clc", "B17", "ptx", 2_000_000);
    check_dpor("r1d_clc", "B17", "sass", 2_000_000);
}
#[test]
fn r1d_b53_parity() {
    // Unbounded (the CTA keeps re-requesting): bounded search, Verdict::meets.
    check_dpor("r1d_clc", "B53", "ptx", 20_000);
    check_dpor("r1d_clc", "B53", "sass", 20_000);
}
#[test]
fn r1d_b54_lost_tile() {
    check_dpor("r1d_clc", "B54", "ptx", 2_000_000);
    check_dpor("r1d_clc", "B54", "sass", 2_000_000);
}

// Rung 1e: PDL. Small enough for both explorers (check_dir cross-checks them).
#[test]
fn r1e_pdl() {
    for v in ["good", "B18", "B55"] {
        for f in ["ptx", "sass"] {
            check_dir("fixtures", "r1e_pdl", v, f);
            check_dir("fixtures", "r1e_serial", v, f);
        }
    }
}

// Rung 1f: two GPUs, flag ping-pong over peer memory (both explorers).
#[test]
fn r1f_pingpong() {
    for v in ["good", "fenced", "B19", "B20", "B56"] {
        for f in ["ptx", "sass"] {
            check_dir("fixtures", "r1f_pingpong", v, f);
        }
    }
}

/// Stateful explorer only, must complete (r1g: DPOR revisits too many traces here).
fn check_stateful(case: &str, variant: &str, front: &str, max_states: usize) {
    let case = Case::load(root().join("fixtures").join(case)).unwrap();
    let k = kernel(&case, variant, front);
    let launch = build_launch(&case, &k).unwrap();
    let m0 = Machine::new(Arc::new(k), Arc::new(launch));
    let v = explore(m0, max_states);
    let expect = case.variant(variant).unwrap().expect_for(front);
    assert!(
        v.complete && v.meets(&expect),
        "{variant}/{front}: got {:?} (complete: {}), want {expect:?}",
        v.classes(),
        v.complete
    );
}

// Rung 1g: 3-GPU all-reduce, two calls. PTX is exhaustive; the SASS good kernel needs tens of
// millions of states (docs/rung1g.md), so SASS checks the mutants only.
#[test]
fn r1g_allreduce() {
    for v in ["good", "B21", "B22", "B24"] {
        check_stateful("r1g_allreduce", v, "ptx", 2_000_000);
    }
    for v in ["B21", "B22"] {
        check_stateful("r1g_allreduce", v, "sass", 2_000_000);
    }
    check_dpor("r1g_allreduce", "B24", "sass", 20_000);
}

/// Stateful explorer, bounded: a mutant must show its expected findings within `max_states`.
fn check_bounded(case: &str, variant: &str, front: &str, max_states: usize) {
    let case = Case::load(root().join("fixtures").join(case)).unwrap();
    let k = kernel(&case, variant, front);
    let launch = build_launch(&case, &k).unwrap();
    let m0 = Machine::new(Arc::new(k), Arc::new(launch));
    let v = explore(m0, max_states);
    let expect = case.variant(variant).unwrap().expect_for(front);
    assert!(
        v.meets(&expect),
        "{variant}/{front}: got {:?} (complete: {}), want {expect:?}",
        v.classes(),
        v.complete
    );
}

// Rung 2: Triton 3.5 persistent tcgen05 GEMM (docs/rung2.md). SASS is exhaustive for the good
// kernel, the fixed kernel and M3/M4; M1/M2 (missing mbarrier waits) explode and are bounded.
#[test]
fn r2_triton_sass() {
    for v in ["good", "fenced", "M3", "M4"] {
        check_stateful("r2_triton_gemm", v, "sass", 2_000_000);
    }
    for v in ["M1", "M2"] {
        check_bounded("r2_triton_gemm", v, "sass", 5_000);
    }
}

#[test]
#[ignore = "slow (~10 min): PTX good/fenced explore 50k states each"]
fn r2_triton_ptx() {
    for v in ["good", "fenced"] {
        check_stateful("r2_triton_gemm", v, "ptx", 2_000_000);
    }
    for v in ["M1", "M2", "M3", "M4"] {
        check_bounded("r2_triton_gemm", v, "ptx", 20_000);
    }
}

// Rung 2: CuTeDSL GEMM at one tile per CTA (6 warps, warp-specialized). Exhaustive except the
// PTX of CM1 / CM4, whose bug lets the pipeline run off into millions of states.
#[test]
fn r2_cute_small() {
    if vendor_missing(&["r2_cute_small"]) {
        return;
    }
    for v in ["good", "fenced", "CM1", "CM2", "CM3", "CM4"] {
        check_stateful("r2_cute_small", v, "sass", 2_000_000);
    }
    for v in ["good", "fenced", "CM2", "CM3"] {
        check_stateful("r2_cute_small", v, "ptx", 2_000_000);
    }
    // CM1's deadlock needs the MMA warp to run ahead: ~1M states (a few seconds in parallel).
    check_bounded("r2_cute_small", "CM1", "ptx", 1_000_000);
    check_bounded("r2_cute_small", "CM4", "ptx", 50_000);
}

// Rung 3b (CuTeDSL half): MXF8 block-scaled GEMM (TMA, tcgen05.cp of the scale factors,
// tcgen05.mma.kind::mxf8f6f4.block_scale). Exhaustive verdicts; both front-ends bitwise equal
// to the recorded GB300 outputs (good and B30); B30 (scale factors of the wrong K-block)
// fails the exact MXF8 spec while the good kernel's dataflow equals it.
#[test]
fn r3_cute_mxf8() {
    if vendor_missing(&["r3_cute_mxf8"]) {
        return;
    }
    for v in ["good", "B30"] {
        for front in ["ptx", "sass"] {
            check_stateful("r3_cute_mxf8", v, front, 2_000_000);
            let case = Case::load(root().join("fixtures/r3_cute_mxf8")).unwrap();
            let k = kernel(&case, v, front);
            let launch = build_launch(&case, &k).unwrap();
            let (m, _, _) =
                ptxv_sim::explore::run_once(Machine::new(Arc::new(k), Arc::new(launch)));
            let hw = std::fs::read(case.dir.join(if v == "good" {
                "gb300_C.bin"
            } else {
                "gb300_C_B30.bin"
            }))
            .unwrap();
            assert!(
                m.outputs() == hw,
                "{v}/{front}: simulator output differs from GB300"
            );
            assert_eq!(
                spec_match("r3_cute_mxf8", v, front),
                v == "good",
                "{v}/{front}: spec"
            );
        }
    }
}

// The same CuTeDSL GEMM for MXF4 (e2m1 codes through 4-bit tensor maps, kind::mxf4nvf4
// block32): exhaustive verdicts; both front-ends bitwise equal to the recorded GB300 output.
#[test]
fn r3_cute_mxf4() {
    if vendor_missing(&["r3_cute_mxf4"]) {
        return;
    }
    for front in ["ptx", "sass"] {
        check_stateful("r3_cute_mxf4", "good", front, 2_000_000);
        let case = Case::load(root().join("fixtures/r3_cute_mxf4")).unwrap();
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let (m, _, _) = ptxv_sim::explore::run_once(Machine::new(Arc::new(k), Arc::new(launch)));
        let hw = std::fs::read(case.dir.join("gb300_C.bin")).unwrap();
        assert!(
            m.outputs() == hw,
            "{front}: simulator output differs from GB300"
        );
    }
}

// The same CuTeDSL GEMM for NVFP4 (e2m1 codes, ue4m3 scales per 16, kind::mxf4nvf4 block16):
// exhaustive verdicts; both front-ends bitwise equal to the recorded GB300 output.
#[test]
fn r3_cute_nvf4() {
    if vendor_missing(&["r3_cute_nvf4"]) {
        return;
    }
    for front in ["ptx", "sass"] {
        check_stateful("r3_cute_nvf4", "good", front, 2_000_000);
        let case = Case::load(root().join("fixtures/r3_cute_nvf4")).unwrap();
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let (m, _, _) = ptxv_sim::explore::run_once(Machine::new(Arc::new(k), Arc::new(launch)));
        let hw = std::fs::read(case.dir.join("gb300_C.bin")).unwrap();
        assert!(
            m.outputs() == hw,
            "{front}: simulator output differs from GB300"
        );
    }
}

// Rung 2: CuTeDSL GEMM with a 2-stage A/B ring and one accumulator buffer (CUTE_STAGES=2,1,2),
// K = 256: both rings wrap, so phase parity is exercised (CM5 / CM6 drop a phase flip).
#[test]
#[ignore = "slow (~20 min on all cores): good/fenced SASS explore 6.5M states each"]
fn r2_cute_wrap() {
    for v in ["good", "fenced"] {
        check_stateful("r2_cute_wrap", v, "sass", 20_000_000);
    }
    for v in ["CM1", "CM2", "CM3", "CM4", "CM5", "CM6"] {
        check_bounded("r2_cute_wrap", v, "sass", 2_000_000);
    }
}

/// Cluster symmetry (`explore_symmetric`): the good kernel lets CTA 0 stand for all four; each
/// bug only shows in CTA 3, so symmetry must be refused and every CTA explored.
#[test]
fn e_sym_symmetry() {
    let case = Case::load(root().join("fixtures/e_sym")).unwrap();
    for front in ["ptx", "sass"] {
        for variant in ["good", "SYM1", "SYM2"] {
            let k = Arc::new(kernel(&case, variant, front));
            let launch = Arc::new(build_launch(&case, &k).unwrap());
            let make = || Machine::new(k.clone(), launch.clone());
            let v = ptxv_sim::explore::explore_symmetric(&make, 100_000, &mut |_, _| {});
            assert!(v.complete, "{variant}/{front}: incomplete");
            assert_eq!(
                v.classes(),
                case.variant(variant).unwrap().expect_for(front),
                "{variant}/{front}: {:#?}",
                v.findings
            );
            let symmetric = v.notes.iter().any(|n| n.contains("stands for all"));
            assert_eq!(
                symmetric,
                variant == "good",
                "{variant}/{front}: {:?}",
                v.notes
            );
        }
    }
}

#[test]
fn e_exitbound_races_on_the_way_out() {
    let case = Case::load(root().join("fixtures/e_exitbound")).unwrap();
    for front in ["ptx", "sass"] {
        for variant in ["good", "XB1", "XB2"] {
            let k = Arc::new(kernel(&case, variant, front));
            let launch = Arc::new(build_launch(&case, &k).unwrap());
            let v = ptxv_sim::explore::explore(Machine::new(k.clone(), launch.clone()), 100_000);
            assert!(v.complete, "{variant}/{front}: incomplete");
            assert_eq!(
                v.classes(),
                case.variant(variant).unwrap().expect_for(front),
                "{variant}/{front}: {:#?}",
                v.findings
            );
            if variant == "good" {
                assert_eq!(v.distinct_outputs, 1, "{front}");
            }
        }
    }
}

/// Captured cuBLAS GEMMs (`torch.mm` vs `(b.t() @ a.t()).t()`, bf16): the TTN one stores an
/// 8 x 16 box with SWIZZLE_64B (smem rows padded to the span, probes/tma_narrow_box); the
/// mcast one (cluster of 8) multicasts 4-row slices of a 128B-swizzled tile to 512-byte
/// offsets that are not 1024-aligned (probes/tma_mcast_swizzle).
/// c_cve_sdpa_f32 (CUTLASS fmha, f32 on TF32 tensor cores) certifies in one run; its
/// float-max atomics interleave past 200k states when explored.
const CERTIFY_ONLY: &[&str] = &[
    // Two CTAs of 128 threads (beyond 200k states on SASS; e_mma_pair_m128_verdicts explores it).
    "e_mma_pair_m128",
    "r3b_trtllm_bmm",
    "r3b_trtllm_bmm_k2048",
    "s_cpasync_mbar",
    // 64 threads of independent cp.async / bulk copies (beyond 200k states).
    "s_ptx_cute_ops",
    "c_mm_bf16_nnt",
    "c_mm_bf16_ttn",
    "c_mm_bf16_mcast",
    "c_cve_sdpa_f32",
    // FlashAttention-4 forward: 16 warps; certified in one run (E-WELLSYNC).
    "c_fa4_fwd",
    // FA4 forward, head dim 128 (AL's fa_hd128): its retry stub polls the same mbarrier
    // through an address computed after the wait (E-SPIN stub parity).
    "c_fa4_hd128",
    // FA4 forward with causal / local (64, 0) masks: mask words 0xffffffff >> n, n up to 128
    // (SHF clamp, L-SHF; fixtures/s_shf).
    "c_fa4_causal",
    "c_fa4_local",
    // fixtures/c_equiv_online: four warps' float-max atomics on one word per row.
    "c_online_ok",
    "c_online_bad_mp",
    "c_online_bad_sel",
    "c_online_bad_shfl",
    "c_online_bad_word",
];

/// Captured cases (tools/capture.py, `c_*`): the good kernel is clean on every schedule and
/// its one output equals the bytes the GPU produced when the case was captured (PTX cases:
/// from both front-ends).
#[test]
fn captured_outputs_match_the_gpu() {
    let mut n = 0;
    for e in std::fs::read_dir(root().join("fixtures")).unwrap() {
        let d = e.unwrap().path();
        if !d.join("case.toml").exists() {
            continue;
        }
        if vendor_missing(&[d.file_name().unwrap().to_str().unwrap()]) {
            continue;
        }
        let case = Case::load(&d).unwrap();
        if !case.m.buffers.iter().any(|b| b.expect.is_some()) {
            continue;
        }
        // Multi-GPU captures (rung 4a) need millions of states: r4a_first_call_exhaustive.
        // s_warpcoll does not finish either: s_warpcoll_matches_hardware.
        // r3b_trtllm_bmm_partial reads unwritten shared memory (tests/r3b.rs); r3b_trtllm_moe
        // races in gemm2 (T3b-2, tests/r3b_moe.rs).
        if case.m.ndevices() > 1
            || case.m.name == "s_warpcoll"
            || case.m.name == "r3b_trtllm_bmm_partial"
            || case.m.name == "r3b_trtllm_moe"
        {
            continue;
        }
        // The good variant's expectation: "clean" ones must reproduce the GPU bytes; rung-4
        // kernels that stop at an unmodelled MUFU expect "unmodeled", and coverage targets the
        // lifter cannot take yet expect "unlifted" (loading must fail, not mis-lift).
        let expect = case.variant("good").unwrap().expect_for("sass");
        let name = &case.m.name;
        // PTX cases (Triton / CuTeDSL JIT): both front-ends must reproduce the GPU bytes.
        let ks: Vec<(&str, ptxv_ir::Kernel)> = match &case.m.cubin {
            Some(bin) => {
                let ks: Result<Vec<_>, _> = case
                    .m
                    .all_launches()
                    .iter()
                    .map(|l| {
                        let c = case.dir.join(l.cubin.unwrap_or(bin));
                        ptxv_sass::load_cubin(&c, l.entry)
                    })
                    .collect();
                if expect == ["unlifted"] {
                    assert!(ks.is_err(), "{name}: lifts now; update its expectation");
                    continue;
                }
                let mut ks = ks.unwrap();
                let k = if ks.len() == 1 {
                    ks.pop().unwrap()
                } else {
                    ptxv_ir::Kernel::concat(ks)
                };
                vec![("sass", k)]
            }
            None => ["sass", "ptx"]
                .into_iter()
                .map(|f| (f, kernel(&case, "good", f)))
                .collect(),
        };
        for (front, k) in ks {
            let launch = build_launch(&case, &k).unwrap();
            let m = Machine::new(Arc::new(k), Arc::new(launch));
            // Launches too large to explore whole (cuBLAS, 8 CTAs in clusters of 4) are decided
            // by the certificate (E-WELLSYNC), which must hold.
            let v = if CERTIFY_ONLY.contains(&name.as_str()) {
                ptxv_sim::wellsync::certify(m)
                    .verdict
                    .unwrap_or_else(|| panic!("{name}: certificate does not hold"))
            } else {
                explore(m, 200_000)
            };
            assert_eq!(v.classes(), expect, "{name}/{front}: {:#?}", v.findings);
            // Runs that stop early (unmodeled) or have several outputs are not compared; the
            // others are, clean or not (r4_attn_triton: a contract violation, bitwise = GB300).
            if expect
                .iter()
                .any(|e| e == "unmodeled" || e == "nondeterministic")
            {
                continue;
            }
            assert!(v.complete, "{name}/{front}: incomplete");
            assert_eq!(v.outputs.len(), 1, "{name}/{front}");
            let report = case.expect_report(&v.outputs[0]).unwrap();
            assert!(
                !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
                "{name}/{front}: {report:?}"
            );
        }
        n += 1;
    }
    assert!(n > 0, "no captured fixtures");
}

/// A binary-only case's variant: its cubin plus the variant's control-bit patch.
fn binary_kernel(case: &Case, variant: &str) -> anyhow::Result<ptxv_ir::Kernel> {
    let dir = std::env::temp_dir().join(format!("ptxv-test-{}-{variant}", case.m.name));
    std::fs::create_dir_all(&dir)?;
    let c = dir.join("k.cubin");
    std::fs::copy(case.dir.join(case.m.cubin.as_ref().unwrap()), &c)?;
    ptxv_sass::scoreboard::patch_file(&c, &case.m.entry, &case.variant(variant)?.patch)?;
    ptxv_sass::load_cubin(&c, &case.m.entry)
}

/// Rung 4 (softmax, layer/rms norm, row reductions, attention; Triton and torch's own
/// kernels): every variant gets its expected classes (MUFU per N-MUFU; their outputs are
/// checked against GB300 by `captured_outputs_match_the_gpu`). "unlifted" cases are coverage
/// targets: loading them must fail (see docs/lift-coverage.md).
#[test]
fn r4_kernel_classes() {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root().join("fixtures"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_str().unwrap().starts_with("r4_"))
        .collect();
    dirs.sort();
    assert!(!dirs.is_empty());
    for d in dirs {
        if vendor_missing(&[d.file_name().unwrap().to_str().unwrap()]) {
            continue;
        }
        let case = Case::load(&d).unwrap();
        for v in case.m.variants.iter().map(|v| v.id.clone()) {
            let expect = case.variant(&v).unwrap().expect_for("sass");
            let name = format!("{}/{v}", case.m.name);
            let k = binary_kernel(&case, &v);
            if expect == ["unlifted"] {
                assert!(k.is_err(), "{name}: lifts now; update its expectation");
                continue;
            }
            let k = k.unwrap();
            let launch = build_launch(&case, &k).unwrap();
            let r = explore(Machine::new(Arc::new(k), Arc::new(launch)), 200_000);
            assert_eq!(r.classes(), expect, "{name}: {:#?}", r.findings);
            // A control-bit mutant must be caught by the scoreboard check itself (r4_attn's
            // good variant already has a violation: Triton's descriptor bits).
            if !case.variant(&v).unwrap().patch.is_empty() {
                assert!(
                    r.findings.iter().any(|f| f.sig.starts_with("sb:")),
                    "{name}: no scoreboard finding: {:#?}",
                    r.findings
                );
            }
        }
    }
}

fn fnv(b: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in b {
        h = (h ^ *x as u64).wrapping_mul(0x100000001b3);
    }
    h
}

/// One certified run (E-WELLSYNC) of a binary case; its output's FNV-1a hash must be the one
/// GB300 wrote (`ptxv hw` prints it as output_hash).
fn certified_hash(dir: &str, case: &str, gb300: u64) {
    let case = Case::load(root().join(dir).join(case)).unwrap();
    let k = binary_kernel(&case, "good").unwrap();
    let launch = build_launch(&case, &k).unwrap();
    let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
        .verdict
        .unwrap_or_else(|| panic!("{} does not certify", case.m.name));
    assert_eq!(v.classes(), ["clean"], "{:#?}", v.findings);
    assert_eq!(v.outputs.len(), 1);
    assert_eq!(
        fnv(&v.outputs[0]),
        gb300,
        "{}: output differs from GB300's",
        case.m.name
    );
}

/// N-F2I-F64: f64 -> integer conversions (fixtures/s_f2i_f64: cvt.r{z,n,m,p}i.{s,u}{32,64}.f64 =
/// SASS F2I[.S64|.U64|.U32].F64[.TRUNC|.FLOOR|.CEIL], 256 special and random operands) agree
/// with GB300 bitwise on both front-ends (run 2026-11-05, GPU 3, 100/100 runs).
#[test]
fn s_f2i_f64_matches_hardware() {
    let case = Case::load(root().join("fixtures/s_f2i_f64")).unwrap();
    for front in ["ptx", "sass"] {
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
            .verdict
            .expect("s_f2i_f64 certifies");
        assert_eq!(v.classes(), ["clean"], "{front}: {:#?}", v.findings);
        assert_eq!(
            fnv(&v.outputs[0]),
            0x484a8cbc44fc8165,
            "{front}: differs from GB300"
        );
    }
}

/// L-SHF: shifts by a run-time amount (fixtures/s_shf: shl / shr .b32 / .u64 / .s64,
/// shf.{l,r}.{clamp,wrap}; SASS SHF.{L,R}[.W].{U32,S32,U64,S64}[.HI]) with amounts 0..33, at the
/// 32 / 64 boundaries, >= 64 and >= 2^31 agree with GB300 bitwise on both front-ends (run
/// 2026-11-05, GPU 0, 20/20 runs). Amounts >= 64 used to wrap mod 64 in SHF, which made
/// FA4's causal / local mask words wrong (fixtures/c_fa4_causal, c_fa4_local).
#[test]
fn s_shf_matches_hardware() {
    let case = Case::load(root().join("fixtures/s_shf")).unwrap();
    for front in ["ptx", "sass"] {
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
            .verdict
            .expect("s_shf certifies");
        assert_eq!(v.classes(), ["clean"], "{front}: {:#?}", v.findings);
        assert_eq!(
            fnv(&v.outputs[0]),
            0x3f8e417548beb899,
            "{front}: differs from GB300"
        );
    }
}

/// The rung-4 f32 ALU forms (N-FALU, N-FCHK, fixtures/s_alu2: 720,896 output words) agree
/// with GB300 bitwise (run 2026-11-03, GPU 0).
#[test]
fn s_alu2_matches_hardware() {
    certified_hash("fixtures", "s_alu2", 0x6c4edfa944cadc91);
}

/// gatherTopK's warp collectives (E-COLLECTIVE; fixtures/s_warpcoll: REDUX / CREDUX forms,
/// SR_*MASK, BRA.DIV and WARPSYNC.COLLECTIVE regions with 1, 2 and 3 member-mask groups)
/// agree with GB300 bitwise (run 2026-11-03, GPU 3). Lanes leave its shrinking-mask loop at
/// different iterations, which the certificate does not take (C-WARP), so: a bounded
/// exploration with no finding, one distinct output, equal to GB300's bytes.
#[test]
fn s_warpcoll_matches_hardware() {
    let case = Case::load(root().join("fixtures/s_warpcoll")).unwrap();
    let k = binary_kernel(&case, "good").unwrap();
    assert!(
        k.code
            .iter()
            .any(|i| matches!(i.op, ptxv_ir::Op::EndCollective { .. })),
        "no collective region lowered"
    );
    let launch = build_launch(&case, &k).unwrap();
    let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 200_000);
    assert!(v.findings.is_empty(), "{:#?}", v.findings);
    assert_eq!(v.outputs.len(), 1);
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
        "{report:?}"
    );
}

/// FCHK's predicate (N-FCHK): probes/fchk, a cubin patched to run FCHK on every pair of the
/// s_alu2 operands, against GB300 (run 2026-11-03).
#[test]
fn probe_fchk_matches_hardware() {
    certified_hash("probes", "fchk", 0xa83533de36c447d4);
}

/// LOP3's predicate operand (L-LOP3-P): probes/lop3_pmode, the ptxas LOP3 patched into the
/// `.PAND` and default forms with P1 / !P1, through the SASS front-end against GB300's outputs
/// (run 2026-11-04).
#[test]
fn probe_lop3_pmode_matches_hardware() {
    let case = Case::load(root().join("probes/lop3_pmode")).unwrap();
    for form in ["base", "pand_p1", "pand_np1", "por_p1", "por_np1"] {
        let cubin = case.dir.join(format!("cubins/{form}.cubin"));
        let k = ptxv_sass::load_cubin(&cubin, &case.m.entry).unwrap();
        let pand = k.code.iter().any(|i| {
            matches!(
                i.op,
                ptxv_ir::Op::Lop3 {
                    pq: Some((_, true)),
                    ..
                }
            )
        });
        assert_eq!(pand, form.starts_with("pand"), "{form}");
        let launch = build_launch(&case, &k).unwrap();
        let (m, f, _) = ptxv_sim::explore::run_once(Machine::new(Arc::new(k), Arc::new(launch)));
        assert!(f.is_empty(), "{form}: {f:?}");
        let hw = std::fs::read(case.dir.join(format!("expected/{form}.bin"))).unwrap();
        assert!(
            m.outputs() == hw,
            "{form}: simulator output differs from GB300"
        );
    }
}

/// S-TMEM: tcgen05.ld.16x128b .x1 / .x2 / .x4 (probes/tmem_16x128b; SASS LDTM.16dp128bit),
/// both front-ends against GB300's output (run 2026-11-04).
#[test]
fn probe_tmem_16x128b_matches_hardware() {
    let case = Case::load(root().join("probes/tmem_16x128b")).unwrap();
    let hw = std::fs::read(case.dir.join("expected/out.bin")).unwrap();
    for front in ["ptx", "sass"] {
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let (m, f, _) = ptxv_sim::explore::run_once(Machine::new(Arc::new(k), Arc::new(launch)));
        assert!(f.is_empty(), "{front}: {f:?}");
        assert!(
            m.outputs() == hw,
            "{front}: simulator output differs from GB300"
        );
    }
}

/// N-HMMA: probes/hmma (mma.sync m16n8k16 bf16, 4 accumulating steps per 16x8 tile) against
/// GB300's output for its default inputs (the calibration over 98,304 outputs is
/// `ptxv calib-mma probes/hmma`).
#[test]
fn probe_hmma_matches_hardware() {
    let case = Case::load(root().join("probes/hmma")).unwrap();
    let k = binary_kernel(&case, "good").unwrap();
    let launch = build_launch(&case, &k).unwrap();
    let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
        .verdict
        .expect("probes/hmma certifies");
    assert_eq!(v.classes(), ["clean"], "{:#?}", v.findings);
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
        "{report:?}"
    );
}

/// N-HMMA, TF32: probes/tmma (mma.sync m16n8k8 tf32 on raw f32 words, 8 accumulating steps
/// per 16x8 tile) against GB300's output for its default inputs (the calibration over 98,304
/// outputs is `ptxv calib-mma probes/tmma`).
#[test]
fn probe_tmma_matches_hardware() {
    let case = Case::load(root().join("probes/tmma")).unwrap();
    let k = binary_kernel(&case, "good").unwrap();
    let launch = build_launch(&case, &k).unwrap();
    let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
        .verdict
        .expect("probes/tmma certifies");
    assert_eq!(v.classes(), ["clean"], "{:#?}", v.findings);
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
        "{report:?}"
    );
}

/// The forms lifted for the compile-vs-eager gaps (fixtures/s_lift_k: FFMA2 with negated
/// operands, VIMNMX3, FSET.BF, FCHK with an immediate, ATOMS.MIN / .MAX[.S32] including the
/// float-max mix, LDL / STL .U8 / .U16; 1,050,624 output bytes) agree with GB300 bitwise (run
/// 2026-11-03, GPU 0; 1000/1000 runs).
#[test]
fn s_lift_k_matches_hardware() {
    certified_hash("fixtures", "s_lift_k", 0xaa290e508e347383);
}

/// S-PROXY: a fence.proxy.async by any thread that observed the writes and happens-before the
/// async op covers them (probes/proxy_fence: GB300 1000/1000 correct with the fence after the
/// barrier, by all threads or by the issuer only; 1000/1000 wrong without).
#[test]
fn probe_proxy_fence_matches_hardware() {
    for v in ["good", "after_all", "issuer", "none"] {
        check_dir("probes", "proxy_fence", v, "ptx");
        check_dir("probes", "proxy_fence", v, "sass");
    }
}

/// PTX f64 front-end (probes/f64_ptx): div / sqrt.f64 in every rounding mode and add / sub /
/// mul / fma.f64 on every pair of the s_f64 operands equal GB300's output (PTX front only:
/// the SASS expansion needs MUFU.RCP64H / RSQ64H).
#[test]
fn probe_f64_ptx_matches_hardware() {
    let case = Case::load(root().join("probes/f64_ptx")).unwrap();
    let k = kernel(&case, "good", "ptx");
    let launch = build_launch(&case, &k).unwrap();
    let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
        .verdict
        .expect("probes/f64_ptx certifies");
    assert_eq!(v.classes(), ["clean"], "{:#?}", v.findings);
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
        "{report:?}"
    );
}

/// A binary case with several launches (rung 4a: torch's symmetric-memory all-reduce on three
/// GPUs): every launch's kernel, from the variant's patched cubin, combined.
fn binary_launches(case: &Case, variant: &str) -> ptxv_ir::Kernel {
    // Per call: tests run in parallel on the same variants.
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "ptxv-test-{}-{variant}-{}-{n}",
        case.m.name,
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let c = dir.join("k.cubin");
    std::fs::copy(case.dir.join(case.m.cubin.as_ref().unwrap()), &c).unwrap();
    // Patches of the case's cubin, and of launches' own cubins (`CtrlPatch::cubin`, rung 4b).
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
    let mut ks: Vec<_> = case
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
    if ks.len() == 1 {
        ks.pop().unwrap()
    } else {
        ptxv_ir::Kernel::concat(ks)
    }
}

/// The first call of a two-call multi-GPU capture: each GPU's first launch, and the outputs
/// (and replica groups) named in `keep`. `expect = false` drops the GPU bytes (they are the
/// state after both calls).
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

fn r4a_machine(case: &Case, variant: &str) -> Machine {
    let k = binary_launches(case, variant);
    let launch = build_launch(case, &k).unwrap();
    Machine::new(Arc::new(k), Arc::new(launch))
}

/// Rung 4a (docs/rung4.md): one call of torch's one-shot and two-shot all-reduce on 3 GPUs,
/// captured by tools/capture_dist.py, explored exhaustively; the one-shot outputs equal every
/// rank's GPU bytes. The strict PTX model (no E-EXIT-FENCE) is clean too for one call.
#[test]
fn r4a_first_call_exhaustive() {
    if std::env::var("PTXV_EXIT_FENCE").is_ok() {
        return; // the case's own exit_fence is what is tested
    }
    for strict in [false, true] {
        let mut case = first_call(
            Case::load(root().join("fixtures/r4a_oneshot")).unwrap(),
            &["o1_r0", "o1_r1", "o1_r2"],
            true,
        );
        if strict {
            case.m.exit_fence = None;
        }
        let v = explore(r4a_machine(&case, "good"), 4_000_000);
        assert!(v.complete, "one-shot: incomplete at {} states", v.states);
        assert_eq!(v.classes(), ["clean"], "{:#?}", v.findings);
        assert_eq!(v.outputs.len(), 1);
        let report = case.expect_report(&v.outputs[0]).unwrap();
        assert!(
            report.len() == 3 && report.iter().all(|l| l.contains("bitwise equal")),
            "{report:?}"
        );
    }
    let case = first_call(
        Case::load(root().join("fixtures/r4a_twoshot")).unwrap(),
        &["buf_r0", "buf_r1", "buf_r2"],
        false,
    );
    let v = explore(r4a_machine(&case, "good"), 4_000_000);
    assert!(v.complete, "two-shot: incomplete at {} states", v.states);
    assert_eq!(v.classes(), ["clean"], "{:#?}", v.findings);
}

/// Rung 4a: without E-EXIT-FENCE (the strict PTX model) the one-shot's closing barrier does
/// not order a peer's reads of call 1 before this rank's restaging in call 2: the waiting side
/// of torch's closing barrier has no acquire (`sync_remote_blocks<memory_order_release>`).
#[test]
fn r4a_oneshot_strict_model_reports_the_reuse_race() {
    if std::env::var("PTXV_EXIT_FENCE").is_ok() {
        return;
    }
    let mut case = Case::load(root().join("fixtures/r4a_oneshot")).unwrap();
    case.m.exit_fence = None;
    let v = explore(r4a_machine(&case, "good"), 3_000_000);
    assert!(
        v.classes().contains(&"race".to_string()),
        "{:?}",
        v.classes()
    );
    assert!(
        v.findings
            .iter()
            .any(|f| f.kind == "race" && f.msg.contains("sass:0x02b0")),
        "the race is call 2's restaging store: {:#?}",
        v.findings
    );
}

/// Rung 4a mutants (B57-B62): each shows its expected findings within a bounded search of the
/// full two-call case (the good kernels' two-call verdicts are in docs/rung4.md: exhaustive,
/// hundreds of millions of states).
#[test]
fn r4a_mutants() {
    for (c, vs) in [
        ("r4a_oneshot", &["B57", "B58", "B59", "B60", "B61"][..]),
        ("r4a_twoshot", &["B62"][..]),
    ] {
        let case = Case::load(root().join("fixtures").join(c)).unwrap();
        for v in vs {
            let r = explore(r4a_machine(&case, v), 3_000_000);
            let expect = case.variant(v).unwrap().expect_for("sass");
            assert!(
                r.meets(&expect),
                "{c}/{v}: got {:?} (complete: {}), want {expect:?}",
                r.classes(),
                r.complete
            );
        }
    }
}

/// SASS compare-and-swap (fixtures/s_cas): generic and global CAS, a CAS signal handoff, and
/// its relaxed-put / lost-signal mutants; outputs = GB300's.
#[test]
fn s_cas_variants() {
    let case = Case::load(root().join("fixtures/s_cas")).unwrap();
    for v in ["good", "relaxed_put", "lost_signal"] {
        let r = explore(r4a_machine(&case, v), 200_000);
        let expect = case.variant(v).unwrap().expect_for("sass");
        assert!(r.complete && r.meets(&expect), "{v}: {:?}", r.classes());
        if v == "good" {
            let report = case.expect_report(&r.outputs[0]).unwrap();
            assert!(report.len() == 3 && report.iter().all(|l| l.contains("bitwise equal")));
        }
    }
}

// ---------------------------------------------------------------- rung 4b (docs/rung4.md)

/// Rung 4b: one deterministic schedule of the whole MoE layer (2 GPUs, 28 launches: DeepEP
/// dispatch, torch gather + CUTLASS grouped GEMM + index_add, DeepEP combine). The bounded and
/// Multi-node explorations are in docs/rung4.md; here the run's findings and outputs.
fn r4b_run(case: &Case, variant: &str) -> (Machine, Vec<ptxv_sim::machine::Finding>, bool) {
    let vcase = case.for_variant(variant).unwrap();
    ptxv_sim::explore::run_once(r4a_machine(&vcase, variant))
}

/// The good layer under the strict PTX model (the case no longer opts into the refuted
/// E-POLL-ACQ): the E-BULK-EXIT contracts and barrier_block's T4b races, and both ranks'
/// outputs equal the GPU's. With the old opt-in, explicitly allowed, only the contracts.
#[test]
fn r4b_moe_run_matches_the_gpu() {
    if vendor_missing(&["r4b_moe"]) {
        return;
    }
    if std::env::var("PTXV_POLL_ACQUIRE").is_ok() || std::env::var("PTXV_HW_MODEL").is_ok() {
        return;
    }
    let mut case = Case::load(root().join("fixtures/r4b_moe")).unwrap();
    assert!(!case.m.poll_acquire);
    let (m, fs, ok) = r4b_run(&case, "good");
    assert!(ok, "not terminal");
    let kinds: std::collections::BTreeSet<_> = fs.iter().map(|f| f.kind.as_str()).collect();
    assert_eq!(kinds, ["contract", "race"].into_iter().collect(), "{fs:?}");
    let report = case.expect_report(&m.outputs()).unwrap();
    assert!(
        report.len() == 2 && report.iter().all(|l| l.contains("bitwise equal")),
        "{report:?}"
    );
    case.m.poll_acquire = true;
    case.m.assume_refuted = vec!["E-POLL-ACQ".into()];
    let (m, fs, ok) = r4b_run(&case, "good");
    assert!(ok, "not terminal");
    assert_eq!(m.launch.refuted_opt_in, ["E-POLL-ACQ"]);
    let kinds: std::collections::BTreeSet<_> = fs.iter().map(|f| f.kind.as_str()).collect();
    assert_eq!(kinds, ["contract"].into_iter().collect(), "{fs:?}");
}

/// B32-B40: each mutant's expected class shows up in the deterministic run (races are
/// happens-before facts, so one schedule that reaches both accesses reports them).
#[test]
fn r4b_moe_mutants() {
    if vendor_missing(&["r4b_moe"]) {
        return;
    }
    let case = Case::load(root().join("fixtures/r4b_moe")).unwrap();
    for v in [
        "B32", "B33", "B34", "B35", "B36", "B37", "B38", "B39", "B40",
    ] {
        let (_, fs, ok) = r4b_run(&case, v);
        let mut kinds: Vec<String> = fs.iter().map(|f| f.kind.clone()).collect();
        if !ok {
            kinds.push("deadlock".into());
        }
        for e in &case.variant(v).unwrap().expect {
            assert!(kinds.contains(e), "{v}: want {e}, got {kinds:?}");
        }
    }
}

/// The dependency claim: output token t of rank s depends exactly on its top-k experts.
/// Perturbing expert e's weights changes exactly the rows of the tokens routed to e, on
/// every rank (and the outputs stay a deterministic function of the inputs).
#[test]
fn r4b_moe_token_depends_exactly_on_its_experts() {
    if vendor_missing(&["r4b_moe"]) {
        return;
    }
    let case = Case::load(root().join("fixtures/r4b_moe")).unwrap();
    let (base, _, ok) = r4b_run(&case, "good");
    assert!(ok);
    let base = base.outputs();
    let (t, h, k, e_per_rank) = (8usize, 128usize, 2usize, 2usize);
    let topk = |r: usize| -> Vec<i64> {
        let b = case.buffer(&format!("topk_r{r}")).unwrap().init_bytes();
        b.chunks(8)
            .take(t * k)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let dir = std::env::temp_dir().join(format!("ptxv-r4b-dep-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for e in 0..4usize {
        let (owner, local) = (e / e_per_rank, e % e_per_rank);
        let mut c = case.clone();
        let name = format!("Wl_r{owner}");
        let b = c.m.buffers.iter_mut().find(|b| b.name == name).unwrap();
        let mut w = b.init_bytes();
        // Expert `local`'s [H, H] bf16 slice: replace by a different deterministic pattern.
        let (lo, hi) = (local * h * h * 2, (local + 1) * h * h * 2);
        for (i, x) in w[lo..hi].chunks_mut(2).enumerate() {
            let v: u16 = 0x3c00 + ((i * 37 + e * 11) % 255) as u16; // bf16 in [0.0078, 0.03)
            x.copy_from_slice(&v.to_le_bytes());
        }
        let f = dir.join(format!("{name}_e{e}.bin"));
        std::fs::write(&f, &w).unwrap();
        b.file = Some(f.to_str().unwrap().to_string());
        let (m, _, ok) = r4b_run(&c, "good");
        assert!(ok);
        let out = m.outputs();
        // outputs = y_r0 ++ y_r1, each [T, H] bf16.
        for r in 0..2 {
            let tk = topk(r);
            for tok in 0..t {
                let off = (r * t + tok) * h * 2;
                let changed = out[off..off + h * 2] != base[off..off + h * 2];
                let routed = tk[tok * k..tok * k + k].contains(&(e as i64));
                assert_eq!(
                    changed, routed,
                    "expert {e}: rank {r} token {tok} changed={changed} routed={routed}"
                );
            }
        }
    }
}

/// T4b: without the opt-ins (strict PTX model), DeepEP's relaxed barrier polls and its
/// shared-memory tail relay are races; the outputs still equal the GPU's.
#[test]
fn r4b_moe_strict_model_reports_deepep_races() {
    if vendor_missing(&["r4b_moe"]) {
        return;
    }
    let mut case = Case::load(root().join("fixtures/r4b_moe")).unwrap();
    case.m.poll_acquire = false;
    case.m.smem_cumulative = false;
    if std::env::var("PTXV_POLL_ACQUIRE").is_ok() || std::env::var("PTXV_SMEM_CUMULATIVE").is_ok() {
        return;
    }
    let (m, fs, ok) = r4b_run(&case, "good");
    assert!(ok);
    let races: Vec<_> = fs.iter().filter(|f| f.kind == "race").collect();
    // notify_dispatch's counts (barrier_block) and combine's tail relay.
    assert!(
        races.iter().any(|f| f.msg.contains("notify_dispatch")),
        "{races:?}"
    );
    assert!(
        races
            .iter()
            .any(|f| f.msg.contains("sass:0x41a0") || f.msg.contains("sass:0x4190")),
        "{races:?}"
    );
    let report = case.expect_report(&m.outputs()).unwrap();
    assert!(
        report.iter().all(|l| l.contains("bitwise equal")),
        "{report:?}"
    );
}

/// barrier.cluster generations are per cluster: cuBLAS's nvjet bf16 mm (clusters of 4, CTAs
/// launched in any order, E-CLC) under random schedules. A cluster's barrier completing used
/// to wake the other cluster's waiters of the same generation, which then ran past their own
/// barrier (false mbarrier-init races, and the wrong cluster's clock acquired).
#[test]
fn cluster_barrier_wakes_only_its_cluster() {
    if vendor_missing(&["c_mm_bf16_nnt"]) {
        return;
    }
    let case = Case::load(root().join("fixtures/c_mm_bf16_nnt")).unwrap();
    let m0 = r4a_machine(&case, "good");
    for seed in 0..8u64 {
        let mut m = m0.clone();
        let mut rng = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut findings = Vec::new();
        loop {
            let acts = m.actions();
            if acts.is_empty() {
                break;
            }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let pick = acts[(rng % acts.len() as u64) as usize].0.clone();
            m.apply(&pick);
            findings.extend(m.findings.drain(..).filter(|f| f.kind != "contract"));
        }
        assert!(
            m.terminal(),
            "seed {seed}: no terminal state; {findings:#?}"
        );
        assert!(findings.is_empty(), "seed {seed}: {findings:#?}");
    }
}

/// Rung 4b launch groups (E-LAUNCH-GROUPS): DeepEP's notify / dispatch, the torch expert
/// computation, DeepEP's cached notify / combine.
fn r4b_groups() -> Vec<Vec<usize>> {
    let torch: Vec<usize> = (6..=18).chain(21..=25).collect();
    vec![
        vec![0, 1],
        vec![2, 3],
        vec![4, 5],
        torch,
        vec![19, 26],
        vec![20, 27],
    ]
}

/// Checks groups of an r4b variant with `opts`; returns the per-group verdicts and the total.
fn r4b_check_groups(
    variant: &str,
    groups: &[Vec<usize>],
    opts: &ptxv_sim::groups::Opts,
    max_states: usize,
) -> (Vec<ptxv_sim::explore::Verdict>, ptxv_sim::explore::Verdict) {
    let case = Case::load(root().join("fixtures/r4b_moe")).unwrap();
    let vcase = case.for_variant(variant).unwrap();
    let m = r4a_machine(&vcase, variant);
    let mut each = Vec::new();
    let v = ptxv_sim::groups::check_groups_opts(
        m,
        groups,
        max_states,
        &mut |_, v| each.push(v.clone()),
        opts,
    );
    (each, v)
}

/// The layout kernels certify, and DeepEP's notify_dispatch on both GPUs, explored from
/// their end state with the multi-grid reduction, is exhaustive. Under the strict PTX model
/// (the case no longer opts into the refuted E-POLL-ACQ) it has barrier_block's races (T4b);
/// under the GB300 hardware model (two-level check) it is clean with one final memory, so each
/// race is latent.
#[test]
fn r4b_notify_group_is_exhaustive() {
    if vendor_missing(&["r4b_moe"]) {
        return;
    }
    if std::env::var("PTXV_POLL_ACQUIRE").is_ok() || std::env::var("PTXV_HW_MODEL").is_ok() {
        return;
    }
    let g = r4b_groups();
    let groups = vec![g[0].clone(), g[1].clone(), g[2..].concat()];
    let opts = ptxv_sim::groups::Opts {
        upto: Some(2),
        ..Default::default()
    };
    let (each, _) = r4b_check_groups("good", &groups, &opts, 1_000_000);
    assert_eq!(each.len(), 2);
    assert!(
        each[0].assumptions.iter().any(|a| a == "E-WELLSYNC"),
        "layout not certified"
    );
    assert!(
        each[1].complete,
        "notify incomplete at {} states",
        each[1].states
    );
    assert_eq!(each[1].classes(), ["race"], "{:#?}", each[1].findings);
    assert!(
        each[1]
            .findings
            .iter()
            .all(|f| f.msg.contains("notify_dispatch"))
    );
    // The hardware model.
    let case = Case::load(root().join("fixtures/r4b_moe")).unwrap();
    let vcase = case.for_variant("good").unwrap();
    let m1 = r4a_machine(&vcase, "good");
    let st = Arc::new(ptxv_sim::l1::L1Stats::default());
    let mut hw = Vec::new();
    ptxv_sim::groups::check_groups_opts(
        ptxv_sim::twolevel::hw_machine(&m1, &st),
        &groups,
        1_000_000,
        &mut |_, v| hw.push(v.clone()),
        &opts,
    );
    assert_eq!(hw.len(), 2);
    assert!(hw[1].complete, "hardware model: notify incomplete");
    assert_eq!(hw[1].classes(), ["clean"], "{:#?}", hw[1].findings);
    assert_eq!(hw[1].distinct_outputs, 1);
    // The two-level classification (twolevel.rs). Closed world, every race is latent; but the
    // havoc probe (E-L1-HAVOC) makes the peer-count reads stale with an outside fill, as
    // GB300 shows with a legal co-resident reader (docs/deepep-coreader.md): those are
    // latent-closed-world. The write-write races stay latent (L1 is write-through).
    let m2 = ptxv_sim::twolevel::hw_machine(&m1, &st);
    let mut v1 = each[1].clone();
    let c = ptxv_sim::twolevel::classify(&mut v1, &hw[1], &st, &m2);
    assert!(c.complete);
    let sass = |pc: u32| {
        let d = m2.k.describe(pc as usize);
        let i = d.find("sass:0x").expect("a sass address");
        d[i + 5..i + 11].to_string()
    };
    let mut got: Vec<(String, String, String)> = v1
        .findings
        .iter()
        .map(|f| {
            let g = f.gb300.as_ref().expect("classified");
            if g.class == "latent-closed-world" {
                assert!(g.reason.contains("E-L1-CLOSED"), "{}", g.reason);
                let w = g.witness.as_deref().unwrap();
                assert_eq!(w, format!("stale-l1-havoc:{}", f.pcs[1]), "{}", f.sig);
            }
            (sass(f.pcs[0]), sass(f.pcs[1]), g.class.clone())
        })
        .collect();
    got.sort();
    got.dedup();
    let cw = "latent-closed-world".to_string();
    let lat = "latent".to_string();
    let want: Vec<(String, String, String)> = vec![
        ("0x0730".into(), "0x15f0".into(), cw.clone()),
        ("0x0730".into(), "0x1600".into(), cw.clone()),
        ("0x0730".into(), "0x1630".into(), lat.clone()),
        ("0x0f80".into(), "0x16d0".into(), cw.clone()),
        ("0x0f80".into(), "0x16f0".into(), cw.clone()),
        ("0x0f80".into(), "0x1d50".into(), lat.clone()),
    ];
    assert_eq!(got, want);
    assert_eq!((c.observable, c.latent_closed_world), (0, 4));
}

/// The torch expert computation of both ranks (gather, CUTLASS grouped GEMM with its
/// second-tile peeks, routing-weight scaling, `index_add_`'s fp32 atomics under C-FADD,
/// casts) certifies as one group, from the end state of one deterministic run of the DeepEP
/// kernels before it (the test's shortcut for groups 1-3).
#[test]
fn r4b_torch_group_certifies() {
    if vendor_missing(&["r4b_moe"]) {
        return;
    }
    let g = r4b_groups();
    let groups = vec![g[..3].concat(), g[3].clone(), g[4..].concat()];
    let opts = ptxv_sim::groups::Opts {
        run_first: 1,
        upto: Some(2),
        ..Default::default()
    };
    let (each, _) = r4b_check_groups("good", &groups, &opts, 1);
    assert_eq!(each.len(), 1);
    assert!(
        each[0].assumptions.iter().any(|a| a == "E-WELLSYNC"),
        "{:?}",
        each[0].notes
    );
    // The grouped GEMM exits with its last TMA store still reading shared memory: its last
    // waits are `DEPBAR.LE SB0, 0x1` and a control-word wait on SB2, another flush's
    // scoreboard, which waits for none of SB0's groups (E-ASYNC-STEP, soundness-harness miss
    // 3). That is an E-BULK-EXIT contract finding only.
    assert_eq!(each[0].classes(), ["contract"], "{:#?}", each[0].findings);
    assert!(
        each[0]
            .findings
            .iter()
            .all(|f| f.sig.starts_with("bulk-store-at-exit:")),
        "{:#?}",
        each[0].findings
    );
}

/// B32 (the expert gather is ordered after nothing on its stream) in group mode: the torch
/// group reads what dispatch, an earlier group, wrote, with nothing ordering the two. The
/// demoted race shadow at the group boundary reports it.
#[test]
fn r4b_groups_catch_b32_at_the_boundary() {
    if vendor_missing(&["r4b_moe"]) {
        return;
    }
    let g = r4b_groups();
    let groups = vec![g[..3].concat(), g[3].clone(), g[4..].concat()];
    let opts = ptxv_sim::groups::Opts {
        run_first: 1,
        upto: Some(2),
        ..Default::default()
    };
    let (each, _) = r4b_check_groups("B32", &groups, &opts, 20_000);
    assert!(
        each[0].findings.iter().any(|f| f.kind == "race"),
        "{:?}",
        each[0].classes()
    );
}

/// E-HW-L1 litmus (fixtures/l1_litmus, docs/deepep-repro.md): DeepEP barrier_block shapes
/// with the payload line's L1 state varied. Stale exactly where GB300 saw stale reads.
#[test]
fn l1_litmus() {
    if std::env::var("PTXV_HW_L1").is_ok() {
        return;
    }
    let case = Case::load(root().join("fixtures/l1_litmus")).unwrap();
    for v in [
        "pr0",
        "pr0-payld",
        "pr0-lineshare",
        "pr1",
        "pr2",
        "pr3",
        "neg",
        "neg-payld",
    ] {
        for f in ["ptx", "sass"] {
            let k = kernel(&case, v, f);
            let launch = build_launch(&case, &k).unwrap();
            let m0 = Machine::new(Arc::new(k), Arc::new(launch));
            let want = case.variant(v).unwrap().expect_for(f);
            // The stateful explorer does not schedule weak loads (E-DRF): the predictive
            // check decides.
            let sv = explore(m0.clone(), 100_000);
            assert!(sv.complete);
            assert_eq!(sv.classes(), want, "{v}/{f}: {:#?}", sv.findings);
            // DPOR orders loads against conflicting stores, so it also reaches schedules in
            // which the early load fills the line before the producer's write: an operational
            // stale hit, whose stale output makes the outputs differ.
            let dv = ptxv_sim::dpor::explore_dpor(m0, 100_000);
            assert!(dv.complete);
            let stale = want.iter().any(|c| c == "stale");
            let op = dv.findings.iter().any(|x| x.sig.starts_with("stale-l1:"));
            let mut dc = dv.classes();
            dc.retain(|c| c != "nondeterministic");
            assert_eq!(dc, want, "{v}/{f} (DPOR): {:#?}", dv.findings);
            assert!(
                !op || stale,
                "{v}/{f}: operational stale hit in a clean variant"
            );
            eprintln!("{v}/{f}: DPOR operational stale hit: {op}");
        }
    }
}

/// E-L1-CORES / E-L1-HAVOC (fixtures/l1_coreader, docs/deepep-coreader.md): DeepEP's
/// barrier_block handoff plus a co-resident reader kernel that legally reads a never-written
/// word of the payload's line. The reader is legal: no race in any mode (the `racy-reader`
/// variant, which reads the payload word itself, is a race). In a closed world with one CTA
/// per SM the handoff is clean; with co-residency the placement that puts the reader next to
/// the consumer makes the consumer's read stale (predictively in the stateful explorer, with
/// an operational hit and the stale output under DPOR); havoc flags the read without the
/// reader. The fixed victim (acquire fence after the poll) is clean in every mode.
#[test]
fn l1_coreader() {
    for var in [
        "PTXV_HW_L1",
        "PTXV_L1_CORES",
        "PTXV_L1_HAVOC",
        "PTXV_POLL_ACQUIRE",
    ] {
        if std::env::var(var).is_ok() {
            return;
        }
    }
    let case0 = Case::load(root().join("fixtures/l1_coreader")).unwrap();
    let run = |cores: bool, havoc: bool, v: &str, f: &str, dpor: bool| {
        let mut case = case0.clone();
        case.m.l1_coresident = cores;
        case.m.l1_havoc = havoc;
        let k = kernel(&case, v, f);
        let launch = build_launch(&case, &k).unwrap();
        let m0 = Machine::new(Arc::new(k), Arc::new(launch));
        let vd = ptxv_sim::l1_cores::explore_placements(m0, &mut |m| {
            if dpor {
                ptxv_sim::dpor::explore_dpor(m, 100_000)
            } else {
                explore(m, 100_000)
            }
        });
        assert!(vd.complete, "{v}/{f}");
        vd
    };
    for f in ["ptx", "sass"] {
        // Default flags: the case's expectations (good / fixed clean, racy-reader a race),
        // and the verdict names the closed-world and placement assumptions.
        for v in ["good", "racy-reader", "fixed"] {
            let vd = run(false, false, v, f, false);
            assert_eq!(
                vd.classes(),
                case0.variant(v).unwrap().expect_for(f),
                "{v}/{f}"
            );
            for a in ["E-HW-L1", "E-L1-CLOSED", "E-L1-PLACE"] {
                assert!(vd.assumptions.iter().any(|x| x == a), "{v}/{f}: {a}");
            }
        }
        // Co-residency: the legal reader makes the consumer's read stale, in the placement
        // that shares the consumer's SM; no race.
        let vd = run(true, false, "good", f, false);
        assert_eq!(vd.classes(), ["stale"], "good/{f}: {:#?}", vd.findings);
        assert!(
            vd.findings
                .iter()
                .all(|x| x.msg.contains("kernel 0 CTA 1 + kernel 1 CTA 0"))
        );
        assert!(vd.assumptions.iter().any(|x| x == "E-L1-CORES"));
        assert!(!vd.assumptions.iter().any(|x| x == "E-L1-PLACE"));
        // DPOR orders the reader's fill between the entry fence and the producer's write:
        // an operational hit, and the consumer returns the stale 0 instead of 1.
        let dv = run(true, false, "good", f, true);
        assert!(
            dv.findings.iter().any(|x| x.sig.starts_with("stale-l1:")),
            "good/{f} DPOR: {:#?}",
            dv.findings
        );
        assert!(!dv.findings.iter().any(|x| x.kind == "race"));
        let outs: Vec<u32> = dv
            .outputs
            .iter()
            .map(|o| u32::from_le_bytes(o[..4].try_into().unwrap()))
            .collect();
        assert!(outs.contains(&0) && outs.contains(&1), "{outs:?}");
        // Havoc: vulnerable without the reader; no closed-world assumption listed.
        let vd = run(false, true, "good", f, false);
        assert_eq!(vd.classes(), ["stale"], "good/{f} havoc");
        assert!(
            vd.findings
                .iter()
                .any(|x| x.sig.starts_with("stale-l1-havoc:"))
        );
        assert!(!vd.assumptions.iter().any(|x| x == "E-L1-CLOSED"));
        // The two-level check (twolevel.rs). Closed world the level-1 verdict is clean (it
        // opts into E-POLL-ACQ, so level 2 runs); the hardware model is clean too, and its
        // havoc probe reports the consumer's read as stale with an outside fill:
        // latent-closed-world. Havoc at level 1: its finding is latent-closed-world.
        // Co-residency is real hardware: observable.
        let two = |cores: bool, havoc: bool, v: &str| {
            let mut case = case0.clone();
            case.m.l1_coresident = cores;
            case.m.l1_havoc = havoc;
            let k = kernel(&case, v, f);
            let launch = build_launch(&case, &k).unwrap();
            let m1 = Machine::new(Arc::new(k), Arc::new(launch));
            let mut v1 =
                ptxv_sim::l1_cores::explore_placements(m1.clone(), &mut |m| explore(m, 100_000));
            let why = ptxv_sim::twolevel::wanted(&v1, &m1.launch).expect("level 2");
            let c = ptxv_sim::twolevel::run(&mut v1, &m1, why, &mut |m| {
                ptxv_sim::l1_cores::explore_placements(m, &mut |m| explore(m, 100_000))
            });
            assert!(c.complete, "{v}/{f}");
            (v1, c)
        };
        let (v1, c) = two(false, false, "good");
        assert_eq!(v1.classes(), ["clean"]);
        assert_eq!(c.extra.len(), 1, "good/{f}: {:#?}", c.extra);
        let g = c.extra[0].gb300.as_ref().unwrap();
        assert_eq!(g.class, "latent-closed-world");
        assert!(c.extra[0].sig.starts_with("stale-l1-havoc:"));
        let (v1, c) = two(false, true, "good");
        assert_eq!(c.latent_closed_world, 1, "{:#?}", v1.findings);
        assert!(v1.findings.iter().all(|x| {
            x.gb300
                .as_ref()
                .is_some_and(|g| g.class == "latent-closed-world")
        }));
        let (v1, c) = two(true, false, "good");
        assert!(c.observable >= 1 && c.latent_closed_world == 0);
        assert!(
            v1.findings
                .iter()
                .all(|x| x.gb300.as_ref().is_some_and(|g| g.class == "observable"))
        );
        let (_, c) = two(false, false, "fixed");
        assert!(c.extra.is_empty(), "fixed/{f}: {:#?}", c.extra);
        // The fix: clean with co-residency (both explorers) and under havoc.
        assert_eq!(run(true, false, "fixed", f, false).classes(), ["clean"]);
        assert_eq!(run(true, false, "fixed", f, true).classes(), ["clean"]);
        assert_eq!(run(false, true, "fixed", f, false).classes(), ["clean"]);
    }
}

/// E-HW-L1 on r1f_pingpong: B20 (relaxed poll, no acquire) is wrong on every GB300 run with
/// out = {2, 1}. The L1 model reads round 0's value from rank 1's L1 in round 2 and ends with
/// exactly those bytes; the correct variants never read a stale line.
#[test]
fn l1_model_r1f_pingpong() {
    if std::env::var("PTXV_HW_L1").is_ok() {
        return;
    }
    let mut case = Case::load(root().join("fixtures/r1f_pingpong")).unwrap();
    case.m.hw_l1 = true;
    for v in ["good", "fenced", "B19", "B20", "B56"] {
        for f in ["ptx", "sass"] {
            let k = kernel(&case, v, f);
            let launch = build_launch(&case, &k).unwrap();
            let m0 = Machine::new(Arc::new(k), Arc::new(launch));
            assert!(m0.l1.is_some());
            let vd = explore(m0.clone(), 100_000);
            assert!(vd.complete);
            let stale = vd.findings.iter().any(|x| x.kind == "stale");
            assert_eq!(stale, v == "B20", "{v}/{f}: {:#?}", vd.findings);
            if v == "B20" {
                assert!(
                    vd.findings.iter().any(|x| x.sig.starts_with("stale-l1:")),
                    "{v}/{f}: operational stale hit expected"
                );
                let (m, _, ok) = ptxv_sim::explore::run_once(m0);
                assert!(ok);
                assert_eq!(
                    m.outputs(),
                    [2u32, 1]
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<u8>>()
                );
            }
        }
    }
    // Off by default: the same variant has no L1 state and no stale finding.
    let case = Case::load(root().join("fixtures/r1f_pingpong")).unwrap();
    if std::env::var("PTXV_HW_L1").is_err() {
        let k = kernel(&case, "B20", "sass");
        let launch = build_launch(&case, &k).unwrap();
        let m0 = Machine::new(Arc::new(k), Arc::new(launch));
        assert!(m0.l1.is_none());
        let vd = explore(m0, 100_000);
        assert!(!vd.findings.iter().any(|x| x.kind == "stale"));
    }
}

/// E-HW-L1 on the captured DeepEP layer (rung 4b), under the GB300 hardware model of the
/// two-level check (twolevel.rs: L1 + sequentially consistent ordering, no E-POLL-ACQ): no
/// weak read of the 28 launches can hit a stale L1 line (predictive check, all schedules with
/// this run's happens-before order), no race remains, and the outputs still equal the GPU's.
/// With the L1 but the strict PTX ordering the predictive check flags notify_dispatch's
/// post-barrier reads, because nothing orders the peer's writes before them.
#[test]
fn r4b_moe_l1_model() {
    if vendor_missing(&["r4b_moe"]) {
        return;
    }
    if std::env::var("PTXV_HW_L1").is_ok()
        || std::env::var("PTXV_POLL_ACQUIRE").is_ok()
        || std::env::var("PTXV_HW_MODEL").is_ok()
    {
        return;
    }
    let case = Case::load(root().join("fixtures/r4b_moe")).unwrap();
    let vcase = case.for_variant("good").unwrap();
    let m1 = r4a_machine(&vcase, "good");
    let st = Arc::new(ptxv_sim::l1::L1Stats::default());
    let m = ptxv_sim::twolevel::hw_machine(&m1, &st);
    assert!(m.launch.hw_order && m.l1.is_some());
    let (m, fs, ok) = ptxv_sim::explore::run_once(m);
    assert!(ok, "not terminal");
    let kinds: std::collections::BTreeSet<_> = fs
        .iter()
        .filter(|f| f.kind != "probe")
        .map(|f| f.kind.as_str())
        .collect();
    assert_eq!(kinds, ["contract"].into_iter().collect(), "{fs:?}");
    // The havoc probe (two-level check, E-L1-HAVOC): with an outside fill, notify_dispatch's
    // post-barrier reads of the peer's counts are stale (docs/deepep-coreader.md). In this one
    // schedule rank 0's CTA 0 reads after its peer's writes (rank 1's reads 0x15f0 / 0x16d0
    // are the other order: the exhaustive notify group test covers both).
    let probes: std::collections::BTreeSet<String> = fs
        .iter()
        .filter(|f| f.kind == "probe")
        .map(|f| {
            let d = m.k.describe(f.pcs[0] as usize);
            assert!(d.contains("notify_dispatch"), "{d}");
            let i = d.find("sass:0x").unwrap();
            d[i + 5..i + 11].to_string()
        })
        .collect();
    assert_eq!(
        probes,
        ["0x1600", "0x16f0"].into_iter().map(String::from).collect()
    );
    let report = case.expect_report(&m.outputs()).unwrap();
    assert!(
        report.iter().all(|l| l.contains("bitwise equal")),
        "{report:?}"
    );
    let mut case = case;
    case.m.hw_l1 = true;
    let (_, fs, _) = r4b_run(&case, "good");
    assert!(
        fs.iter()
            .any(|f| f.kind == "stale" && f.msg.contains("notify_dispatch")),
        "{fs:?}"
    );
}

/// E-L1-CORES / E-L1-HAVOC on DeepEP (fixtures/r4b_coreader: r4b_moe's layout and
/// notify_dispatch launches plus a legal co-resident reader per GPU; docs/deepep-coreader.md).
/// Closed world: clean, and the readers race with nothing. Havoc and co-residency both make
/// exactly notify_dispatch CTA 0's post-barrier reads of the peer's counts stale (rank 0:
/// 0x1600, 0x16f0; rank 1: 0x15f0, 0x16d0, 0x1900), the latter only in the placements that
/// put a reader next to that rank's CTA 0.
#[test]
fn r4b_coreader_notify() {
    if vendor_missing(&["r4b_coreader"]) {
        return;
    }
    for var in [
        "PTXV_HW_L1",
        "PTXV_L1_CORES",
        "PTXV_L1_HAVOC",
        "PTXV_POLL_ACQUIRE",
    ] {
        if std::env::var(var).is_ok() {
            return;
        }
    }
    let case0 = Case::load(root().join("fixtures/r4b_coreader")).unwrap();
    let run = |cores: bool, havoc: bool| {
        let mut case = case0.clone();
        case.m.l1_coresident = cores;
        case.m.l1_havoc = havoc;
        let m0 = r4a_machine(&case, "good");
        ptxv_sim::l1_cores::explore_placements(m0, &mut |m| {
            ptxv_sim::groups::check_spec(m, "2,3,4,5", 2_000_000, &mut |_, _, _| {})
        })
    };
    let want: std::collections::BTreeSet<String> =
        ["0x1600", "0x16f0", "0x15f0", "0x16d0", "0x1900"]
            .into_iter()
            .map(String::from)
            .collect();
    let stale_pcs = |v: &ptxv_sim::explore::Verdict| -> std::collections::BTreeSet<String> {
        v.findings
            .iter()
            .filter(|f| f.kind == "stale")
            .map(|f| {
                let i = f
                    .msg
                    .find("notify_dispatch")
                    .expect("a notify_dispatch read");
                let j = f.msg[i..].find("sass:").unwrap() + i + 5;
                f.msg[j..j + 6].to_string()
            })
            .collect()
    };
    let v = run(false, false);
    assert!(v.complete);
    assert_eq!(v.classes(), ["clean"], "{:#?}", v.findings);
    assert!(v.assumptions.iter().any(|a| a == "E-L1-CLOSED"));
    let v = run(false, true);
    assert!(v.complete);
    assert_eq!(v.classes(), ["stale"], "{:#?}", v.findings);
    assert_eq!(stale_pcs(&v), want);
    let v = run(true, false);
    assert!(v.complete);
    assert_eq!(v.classes(), ["stale"], "{:#?}", v.findings);
    assert_eq!(stale_pcs(&v), want);
    for f in &v.findings {
        assert!(
            f.msg.contains("kernel 2 CTA 0 + kernel 4 CTA 0")
                || f.msg.contains("kernel 3 CTA 0 + kernel 5 CTA 0"),
            "{}",
            f.msg
        );
        assert!(f.msg.contains("coreader sass:0x0060"), "{}", f.msg);
    }
}

/// E-POLL-ANY: a poll whose every possible value passes commutes with the flag's store; a
/// poll that would block on a value written later is not reduced (its deadlock is found once
/// a later round has learned the value), and a race after a reduced poll is still found.
#[test]
fn e_poll_any_keeps_value_sensitive_polls() {
    for v in ["good", "dl", "race"] {
        both("e_poll_any", v);
    }
}

/// Rung 4c (docs/rung4.md): `case` (variant `variant`) with only the launches in `keep` live.
fn r4c_machine(case: &Case, variant: &str, keep: &[usize]) -> Machine {
    let vcase = case.for_variant(variant).unwrap();
    let mut m = r4a_machine(&vcase, variant);
    for g in 0..m.launch.grids.len() {
        if keep.contains(&g) {
            continue;
        }
        for t in m.grid_threads(g) {
            m.threads[t].make_mut().st = ptxv_sim::machine::St::Exited;
        }
    }
    m
}

/// The race findings of one deterministic run of `m`, and whether it terminated.
fn r4c_run(m: Machine) -> (Vec<String>, bool) {
    let (_, fs, ok) = ptxv_sim::explore::run_once(m);
    (
        fs.iter()
            .filter(|f| f.kind == "race")
            .map(|f| f.msg.clone())
            .collect(),
        ok,
    )
}

/// E-SLICE: DeepEP dispatch on 2 GPUs splits into its 4 (source, destination) flows: the
/// sender warps of one rank for one destination with the receiver warps of the other rank
/// for that source (one deterministic run of layout and notify first).
#[test]
fn r4c_dispatch_splits_into_flows() {
    if vendor_missing(&["r4c_nowrap"]) {
        return;
    }
    let case = Case::load(root().join("fixtures/r4c_nowrap")).unwrap();
    let full = r4c_machine(&case, "good", &(0..10).collect::<Vec<_>>());
    let m = r4c_machine(&case, "good", &[0, 1, 2, 3]);
    let (fin, _, ok) = ptxv_sim::explore::run_once(m);
    assert!(ok);
    let cur = ptxv_sim::groups::boundary(fin, &full, &[4, 5], &[]);
    let parts = ptxv_sim::slices::partition(&cur);
    let names: Vec<String> = parts
        .iter()
        .map(|s| ptxv_sim::slices::describe(&cur, s))
        .collect();
    assert_eq!(
        names,
        [
            "k4.cta0.w0-11 + k4.cta1.w0-11",
            "k4.cta0.w12-23 + k5.cta1.w0-11",
            "k4.cta1.w12-23 + k5.cta0.w0-11",
            "k5.cta0.w12-23 + k5.cta1.w12-23",
        ]
    );
}

/// E-SLICE in launch-group mode: notify_dispatch on 2 GPUs splits into the barrier CTA pair
/// and 4 single CTAs; explored slice by slice it is exhaustive and clean, with the same 99
/// terminal states as explored whole (rung 4b). Without E-READ-DONE, as in rung 4b: its read
/// entries tell more terminal states apart (207).
#[test]
fn r4c_notify_group_by_slices() {
    if vendor_missing(&["r4c_nowrap"]) {
        return;
    }
    let mut case = Case::load(root().join("fixtures/r4c_nowrap")).unwrap();
    case.m.read_done = false;
    let m = r4c_machine(&case, "good", &(0..10).collect::<Vec<_>>());
    let groups = vec![vec![0, 1], vec![2, 3], (4..10).collect()];
    let opts = ptxv_sim::groups::Opts {
        upto: Some(2),
        slices: true,
        ..Default::default()
    };
    let mut each = Vec::new();
    ptxv_sim::groups::check_groups_opts(
        m,
        &groups,
        200_000,
        &mut |_, v| each.push(v.clone()),
        &opts,
    );
    let v = &each[1];
    assert!(v.complete, "{:?}", v.notes);
    assert_eq!(v.classes(), ["clean"], "{:#?}", v.findings);
    assert_eq!(v.terminal_states, 99);
    assert!(v.assumptions.iter().any(|a| a == "E-SLICE"));
}

/// T4c: DeepEP reuses a ring slot once the receiver's relaxed head store says it was
/// consumed. Under the strict PTX model the receiver's slot reads race with the sender's
/// overwrite (a 2-slot ring that wraps, layout to dispatch); E-READ-DONE orders them.
#[test]
fn r4c_ring_reuse_is_load_buffering_under_the_strict_model() {
    if vendor_missing(&["r4c_dwrap"]) {
        return;
    }
    let mut case = Case::load(root().join("fixtures/r4c_dwrap")).unwrap();
    case.m.read_done = false;
    let (races, ok) = r4c_run(r4c_machine(&case, "good", &[0, 1, 2, 3, 4, 5]));
    assert!(ok);
    assert!(
        races
            .iter()
            .any(|r| r.contains("read-write race") && r.contains("dispatch")),
        "{races:#?}"
    );
    case.m.read_done = true;
    let (races, ok) = r4c_run(r4c_machine(&case, "good", &[0, 1, 2, 3, 4, 5]));
    assert!(ok);
    assert!(races.is_empty(), "{races:#?}");
}

/// The new ring mutants (rung 4c), layout to dispatch: B41 (the receiver publishes its head
/// before its warps finished reading the slots) races in the wrapping ring; B42 (the sender
/// publishes its tail before every warp wrote its tokens) races.
#[test]
fn r4c_ring_mutants() {
    if vendor_missing(&["r4c_dwrap", "r4c_nowrap"]) {
        return;
    }
    let dwrap = Case::load(root().join("fixtures/r4c_dwrap")).unwrap();
    let (races, _) = r4c_run(r4c_machine(&dwrap, "B41", &[0, 1, 2, 3, 4, 5]));
    assert!(!races.is_empty(), "B41 not caught");
    let nowrap = Case::load(root().join("fixtures/r4c_nowrap")).unwrap();
    let (races, _) = r4c_run(r4c_machine(&nowrap, "B42", &[0, 1, 2, 3, 4, 5]));
    assert!(!races.is_empty(), "B42 not caught");
}

// ---------------------------------------------------------------- two-level verdicts

/// Two-level verdicts (twolevel.rs) on the E-HW-L1 litmus: under the GB300 hardware model a
/// stale read is reachable exactly in the variants where GB300 read stale data, and each
/// level-1 finding is classified: barrier_block's race with nothing loading the line (pr0, the
/// DeepEP shape) or with the line not in L1 (pr1) is latent; the stale reads of pr0-payld /
/// pr2 / pr3 are observable; the races on the early loads of neg / neg-payld are observable
/// timing races (the loads run concurrently with the producer) without any stale read.
#[test]
fn twolevel_l1_litmus() {
    if std::env::var("PTXV_HW_L1").is_ok() || std::env::var("PTXV_HW_MODEL").is_ok() {
        return;
    }
    let case = Case::load(root().join("fixtures/l1_litmus")).unwrap();
    for (v, gb300_stale) in [
        ("pr0", false),
        ("pr0-payld", true),
        ("pr1", false),
        ("pr2", true),
        ("pr3", true),
        ("neg", false),
        ("neg-payld", false),
        ("pr0-lineshare", true),
    ] {
        for f in ["ptx", "sass"] {
            let k = kernel(&case, v, f);
            let launch = build_launch(&case, &k).unwrap();
            let m1 = Machine::new(Arc::new(k), Arc::new(launch));
            let mut v1 = explore(m1.clone(), 100_000);
            let why = ptxv_sim::twolevel::wanted(&v1, &m1.launch).expect("findings");
            let mut v2 = None;
            let c = ptxv_sim::twolevel::run(&mut v1, &m1, why, &mut |m| {
                let r = explore(m, 100_000);
                v2 = Some(r.clone());
                r
            });
            let v2 = v2.unwrap();
            assert!(c.complete, "{v}/{f}");
            assert_eq!(
                v2.findings.iter().any(|x| x.kind == "stale"),
                gb300_stale,
                "{v}/{f}: {:#?}",
                v2.findings
            );
            // The classes do not change; every race / stale finding is classified.
            assert_eq!(v1.classes(), case.variant(v).unwrap().expect_for(f));
            for x in &v1.findings {
                let g = x.gb300.as_ref().expect("classified");
                let want = match (v, x.kind.as_str()) {
                    // Closed world latent; stale with an outside fill of the payload line
                    // (pr0-lineshare is that fill, by the kernel itself).
                    ("pr0" | "pr1", _) => "latent-closed-world",
                    ("pr0-payld" | "pr2" | "neg" | "neg-payld", _) => "observable",
                    // pr3: the reload in the poll loop and the read after it.
                    _ => "observable",
                };
                assert_eq!(g.class, want, "{v}/{f}: {} {:?}", x.sig, g);
            }
            if v == "pr0" {
                let g = v1.findings[0].gb300.as_ref().unwrap();
                assert!(g.reason.contains("no fill of line pay+0x0"), "{}", g.reason);
                assert!(g.reason.contains("E-L1-CLOSED"), "{}", g.reason);
                assert!(
                    g.witness
                        .as_deref()
                        .is_some_and(|w| w.starts_with("stale-l1-havoc:")),
                    "{g:?}"
                );
            }
            assert!(c.extra.is_empty(), "{v}/{f}: {:#?}", c.extra);
        }
    }
}

/// The hardware model on r1f_pingpong: B20 (relaxed poll, no acquire; GB300 wrong 1000/1000)
/// has an observable stale read; the correct variants have none.
#[test]
fn twolevel_r1f_pingpong() {
    if std::env::var("PTXV_HW_L1").is_ok() || std::env::var("PTXV_HW_MODEL").is_ok() {
        return;
    }
    let case = Case::load(root().join("fixtures/r1f_pingpong")).unwrap();
    for v in ["good", "fenced", "B19", "B20", "B56"] {
        let k = kernel(&case, v, "sass");
        let launch = build_launch(&case, &k).unwrap();
        let m1 = Machine::new(Arc::new(k), Arc::new(launch));
        let st = Arc::new(ptxv_sim::l1::L1Stats::default());
        let vd = explore(ptxv_sim::twolevel::hw_machine(&m1, &st), 100_000);
        assert!(vd.complete);
        let stale = vd.findings.iter().any(|x| x.kind == "stale");
        assert_eq!(stale, v == "B20", "{v}: {:#?}", vd.findings);
        if v == "B20" {
            let mut v1 = explore(m1.clone(), 100_000);
            let why = ptxv_sim::twolevel::wanted(&v1, &m1.launch).unwrap();
            ptxv_sim::twolevel::run(&mut v1, &m1, why, &mut |m| explore(m, 100_000));
            assert!(
                v1.findings
                    .iter()
                    .any(|x| x.gb300.as_ref().is_some_and(|g| g.class == "observable")),
                "{:#?}",
                v1.findings
            );
        }
    }
}

/// E-POLL-ACQ is refuted: a case that sets `poll_acquire` is refused unless it opts in
/// explicitly (`assume_refuted`), and the launch then names the opt-in. Only refuted IDs can be
/// opted into.
#[test]
fn refuted_assumption_needs_explicit_opt_in() {
    if std::env::var("PTXV_POLL_ACQUIRE").is_ok() || std::env::var("PTXV_ASSUME_REFUTED").is_ok() {
        return;
    }
    let mut case = Case::load(root().join("fixtures/l1_litmus")).unwrap();
    let k = kernel(&case, "pr0", "ptx");
    let l = build_launch(&case, &k).unwrap();
    assert!(!l.poll_acquire && l.refuted_opt_in.is_empty());
    case.m.poll_acquire = true;
    let e = build_launch(&case, &k).unwrap_err().to_string();
    assert!(
        e.contains("E-POLL-ACQ") && e.contains("assume_refuted"),
        "{e}"
    );
    case.m.assume_refuted = vec!["E-POLL-ACQ".into()];
    let l = build_launch(&case, &k).unwrap();
    assert!(l.poll_acquire);
    assert_eq!(l.refuted_opt_in, ["E-POLL-ACQ"]);
    let m = Machine::new(Arc::new(k.clone()), Arc::new(l));
    let v = explore(m, 100_000);
    assert!(v.assumptions.iter().any(|a| a == "E-POLL-ACQ"));
    case.m.assume_refuted = vec!["E-SMEM-CUMUL".into()];
    assert!(build_launch(&case, &k).is_err());
}

// ------------------------------------------- integer min / max and CAS atomics, CuTeDSL forms

/// Explore `case`/`variant` on `front` exhaustively (`certified`: by the E-WELLSYNC
/// certificate, which must hold); expect a clean verdict whose single output equals the
/// captured GB300 bytes.
fn clean_and_bitwise(case: &str, variant: &str, front: &str, certified: bool) {
    let case = Case::load(root().join("fixtures").join(case)).unwrap();
    let k = kernel(&case, variant, front);
    let launch = build_launch(&case, &k).unwrap();
    let m = Machine::new(Arc::new(k), Arc::new(launch));
    let v = if certified {
        ptxv_sim::wellsync::certify(m)
            .verdict
            .unwrap_or_else(|| panic!("{variant}/{front}: the certificate does not hold"))
    } else {
        explore(m, 2_000_000)
    };
    assert!(
        v.complete && v.classes() == ["clean"],
        "{variant}/{front}: {:?} (complete: {})",
        v.classes(),
        v.complete
    );
    assert_eq!(v.outputs.len(), 1, "{variant}/{front}");
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
        "{variant}/{front}: {report:?}"
    );
}

/// red/atom .min/.max (u32, s32, u64, s64), atom.cas (b32, b64) on shared, global and generic
/// addresses, and CuTeDSL's smem_fmax: GB300's bytes on both front-ends (SASS ATOMS.MIN/MAX,
/// ATOMS.CAS[.64], ATOMS.CAST.SPIN.64, ATOMG/REDG MIN/MAX, ATOMG/ATOM CAS[.64]).
#[test]
fn s_atom_ext_bitwise() {
    for certified in [false, true] {
        clean_and_bitwise("s_atom_ext", "good", "ptx", certified);
        clean_and_bitwise("s_atom_ext", "good", "sass", certified);
    }
}

/// CuTeDSL's PTX forms (cp.async family, non-tensor cp.async.bulk, mad.lo.s64, vote.sync,
/// bar.warp.sync, div.rn.f32, cvt.rn.bf16.f32, cvt.rzi, mov.b32 halves): GB300's bytes.
#[test]
fn s_ptx_cute_ops_bitwise() {
    // 64 threads with independent async copies: the certificate decides it (exploring all
    // schedules exceeds 2M states).
    clean_and_bitwise("s_ptx_cute_ops", "good", "ptx", true);
    clean_and_bitwise("s_ptx_cute_ops", "good", "sass", true);
}

/// The FA4 decode running-max fold bug with the exact smem_fmax asm (red.min.u32 / red.max.s32): the pre-fix read of
/// the shared running max races with the next tile's fold; the fix is clean.
#[test]
fn e_fmax_fold() {
    // The racy pre-fix: the explorers agree only up to the race (E-DRF), so stateful only.
    both("e_fmax_fold", "good");
    for v in ["prefix", "good_3tiles", "prefix_3tiles"] {
        check_stateful("e_fmax_fold", v, "ptx", 2_000_000);
    }
    check_stateful("e_fmax_fold", "good_3tiles", "sass", 2_000_000);
    // SASS pre-fix: LDS vs ATOMS is a race kept by the strong re-check (S-SMEM-STRONG); its
    // schedules exceed 2M states, so the test asks only for the finding.
    for v in ["prefix", "prefix_3tiles"] {
        check_finds("e_fmax_fold", v, "sass", 200_000);
    }
}

/// An incomplete exploration that must still report every expected (non-clean) class.
fn check_finds(case: &str, variant: &str, front: &str, max_states: usize) {
    let case = Case::load(root().join("fixtures").join(case)).unwrap();
    let k = kernel(&case, variant, front);
    let launch = build_launch(&case, &k).unwrap();
    let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), max_states);
    let expect = case.variant(variant).unwrap().expect_for(front);
    assert!(
        !expect.iter().any(|e| e == "clean") && v.meets(&expect),
        "{variant}/{front}: got {:?}, want {expect:?}",
        v.classes()
    );
}

/// The signal-pad channel bug with the barrier's atom.cas spin loops (E-SPIN CAS polls): ring flags sharing
/// the CAS barrier's channel deadlock at ws=2 (two ring calls) and ws=3; the fix is clean.
#[test]
fn e_rs_chan_ws2() {
    for v in ["good", "prefix"] {
        check_stateful("e_rs_chan_ws2", v, "ptx", 2_000_000);
        check_stateful("e_rs_chan_ws2", v, "sass", 2_000_000);
    }
}
#[test]
fn e_rs_chan_ws3() {
    for v in ["good", "prefix"] {
        check_stateful("e_rs_chan_ws3", v, "ptx", 2_000_000);
    }
    check_stateful("e_rs_chan_ws3", "prefix", "sass", 2_000_000);
}
#[test]
#[ignore = "slow (~1 min on all cores): the SASS fix explores 2.8M states"]
fn e_rs_chan_ws3_sass_good() {
    check_stateful("e_rs_chan_ws3", "good", "sass", 4_000_000);
}

/// The block-sparsity bug's real captured CuTeDSL kernel (mad.lo.s64, vote.sync.any.pred on the PTX
/// front-end): the pre-fix reduction-buffer reuse is caught, the fix is clean and equals the
/// GPU's bytes.
#[test]
fn c_block_sparsity() {
    clean_and_bitwise("c_block_sparsity", "good", "ptx", false);
    clean_and_bitwise("c_block_sparsity", "good", "sass", false);
    check_stateful("c_block_sparsity", "prefix", "ptx", 2_000_000);
    check_stateful("c_block_sparsity", "prefix", "sass", 2_000_000);
}

/// SASS forms of real CuTeDSL kernels (fixtures/s_lift_l: CREDUX.{MIN,MAX}[ABS].F32[.NAN] over
/// 32 / 16 / 1 lanes, PLOP3 with a second predicate output, R2P PR, Rx.Bk, LEA.HI with -b;
/// 1,048,576 output bytes) agree with GB300 bitwise (run 2026-11-05, GPU 3; 20/20 runs).
#[test]
fn s_lift_l_matches_hardware() {
    // The 16- and 1-lane reductions split warps (no certificate, C-WARP): explored.
    let case = Case::load(root().join("fixtures/s_lift_l")).unwrap();
    let k = binary_kernel(&case, "good").unwrap();
    let launch = build_launch(&case, &k).unwrap();
    let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 200_000);
    assert!(v.complete, "s_lift_l: incomplete");
    assert_eq!(v.classes(), ["clean"], "{:#?}", v.findings);
    assert_eq!(v.distinct_outputs, 1);
    assert_eq!(
        fnv(&v.outputs[0]),
        6231483402960366304,
        "s_lift_l: output differs from GB300's"
    );
}

/// tcgen05.ld.red (LDTM.STAT.xN.MIN|MAX[ABS].F32[.NAN]) and a negative TMEM offset
/// (`tmem[UR+-0x10]`): probes/tmem_ldred agrees with GB300 bitwise (2026-11-05, 100/100).
#[test]
fn probe_tmem_ldred_matches_hardware() {
    certified_hash("probes", "tmem_ldred", 16475847427938539045);
}

/// E-COLLECTIVE-ALL: ptxas's BRA.DIV / WARPSYNC.COLLECTIVE.ALL / BAR.SYNC R, R / ENDCOLLECTIVE
/// path of a non-aligned barrier.sync. The two kernels GB300 completes are clean with one
/// output equal to GB300's (hw 20/20 bitwise); the two it hangs on deadlock.
#[test]
fn bardiv_matches_hardware() {
    for (name, hash) in [
        ("s_bardiv", 699766608600163605u64),
        ("s_bardiv_mixed", 14485688438926170181),
    ] {
        let case = Case::load(root().join("fixtures").join(name)).unwrap();
        let k = binary_kernel(&case, "good").unwrap();
        assert!(
            k.code
                .iter()
                .any(|i| matches!(i.op, ptxv_ir::Op::WarpSync { all: true, .. })),
            "{name}: no WARPSYNC.COLLECTIVE.ALL lowered"
        );
        let launch = build_launch(&case, &k).unwrap();
        let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 200_000);
        assert!(v.complete, "{name}: incomplete");
        assert_eq!(v.classes(), ["clean"], "{name}: {:#?}", v.findings);
        assert_eq!(v.distinct_outputs, 1, "{name}");
        assert_eq!(
            fnv(&v.outputs[0]),
            hash,
            "{name}: output differs from GB300's"
        );
    }
    for name in ["e_bardiv_hang", "e_bardiv_short"] {
        let case = Case::load(root().join("fixtures").join(name)).unwrap();
        let k = binary_kernel(&case, "good").unwrap();
        let launch = build_launch(&case, &k).unwrap();
        let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 200_000);
        assert!(v.complete, "{name}: incomplete");
        assert_eq!(v.classes(), ["deadlock"], "{name}: {:#?}", v.findings);
    }
}

/// F2I with RN / FLOOR / CEIL (libdevice sinf / cosf: F2I.FTZ.NTZ) and I2F.F64 from 64- and
/// 32-bit integers in every rounding (fixtures/s_f2i; 1,572,864 output bytes) agree with GB300
/// bitwise (2026-11-05, GPU 3, 20/20 runs).
#[test]
fn s_f2i_matches_hardware() {
    certified_hash("fixtures", "s_f2i", 8370786828746058149);
}
/// PTX `st.async ... mbarrier::complete_tx::bytes` lowers to StAsync (its store completes 4
/// transaction bytes on the target's mbarrier), as SASS STAS does. It used to lower to a plain
/// st, leaving the good kernel a spurious deadlock on the PTX front-end.
#[test]
fn e_stasync_complete_tx() {
    for v in ["good", "SA1", "SA3"] {
        both("e_stasync", v);
    }
    for front in ["ptx", "sass"] {
        check_stateful("e_stasync", "SA2", front, 100_000);
    }
}

/// 64-bit selp (b64 / u64 / s64, register and immediate operands) keeps all 64 bits, on both
/// front-ends; fixtures/s_selp64's expected bytes match GB300 (ptxv hw, 2026-11-05).
#[test]
fn s_selp64_keeps_64_bits() {
    let case = Case::load(root().join("fixtures/s_selp64")).unwrap();
    for front in ["ptx", "sass"] {
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 10_000);
        assert_eq!(v.classes(), ["clean"], "{front}: {:#?}", v.findings);
        assert_eq!(v.outputs.len(), 1);
        let r = case.expect_report(&v.outputs[0]).unwrap();
        assert!(
            r.iter().all(|l| l.contains("bitwise equal")),
            "{front}: {r:?}"
        );
    }
}

/// LDS vs ATOMS (S-SMEM-STRONG): on SASS an unsynchronized plain shared load against a
/// shared-memory atomic of another warp is a schedule-dependent outcome, not a race (both are
/// strong at CTA scope); on PTX it is a data race. The strong re-check used to take LDS/STS
/// pairs only, and E-POLL-ANY treated the LDS as value-insensitive although the same step
/// stored the loaded value, which lost the second outcome.
#[test]
fn e_lds_atoms_strong_recheck() {
    both("e_lds_atoms", "good");
    // PTX: the stateful explorer reports the race in one order (E-DRF); DPOR also reverses it.
    check_stateful("e_lds_atoms", "LA1", "ptx", 100_000);
    check("e_lds_atoms", "LA1", "sass");
}

/// E-ALIGNED for named barriers: two halves of a warp at two different aligned bar.sync
/// (SASS BAR) is a divergent aligned barrier (GB300 hangs); the non-aligned barrier.sync is
/// legal there (GB300 completes; SASS: ptxas's BRA.DIV / WARPSYNC.COLLECTIVE.ALL path).
#[test]
fn e_aligned_bar_divergence() {
    for v in ["good", "AB1", "AB2", "AB3"] {
        both("e_aligned_bar", v);
    }
}

/// The FA4 forward warp-role barrier case: the per-warp divergence probe
/// is a violation, the cross-warp pre-fix protocol and the fix are clean.
#[test]
fn e_fa4_warp_roles_fa4_aligned_barriers() {
    for v in [
        "good",
        "prefix",
        "prefix_tmem",
        "prefix_stats",
        "probe_intrawarp",
        "good_nofence",
        "prefix_nofence",
    ] {
        both("e_fa4_warp_roles", v);
    }
    // Racy controls: the stateful explorer reports each race in one order (E-DRF).
    for v in ["ctl_nowait", "ctl_prefix_nosync", "ctl_nofreewait"] {
        for front in ["ptx", "sass"] {
            check_stateful("e_fa4_warp_roles", v, front, 200_000);
        }
    }
    both("e_fa4_warp_roles", "unaligned");
    // SASS: the BRA.DIV / WARPSYNC.COLLECTIVE.ALL slow path in both halves leaves DPOR
    // incomplete at 2M states; the stateful explorer takes ~36k.
    check("e_fa4_warp_roles", "probe_intrawarp_unaligned", "ptx");
    check_stateful(
        "e_fa4_warp_roles",
        "probe_intrawarp_unaligned",
        "sass",
        200_000,
    );
}

/// The FusedRMSNormBwd cluster exit-drain case (S-REMOTE-INFLIGHT): the
/// pre-fix epilogue drains only the last stage's empty barrier, so the peer's earlier
/// `.release.cta` remote arrive on the other stage can land after the exit. It used to be
/// clean (remote arrivals landed at issue); the fix and the no-drain control are as before.
#[test]
fn e_rmsnorm_cluster_drain_remote_arrive_exit_drain() {
    for v in ["good", "bug", "good_k3", "bug_k3", "nodrain"] {
        both("e_rmsnorm_cluster_drain", v);
    }
}

/// S-ALIGN: misaligned ld / st / atom of every width are faults on both front-ends (GB300
/// traps: misaligned address, fixtures/e_misaligned).
#[test]
fn e_misaligned_accesses_fault() {
    for v in ["good", "MA1", "MA2", "MA3", "MA4"] {
        both("e_misaligned", v);
    }
}

/// st.async of every width (.b32 / .b64 / .v2 / .v4; SASS STAS / STAS.64 / STAS.128) completes
/// its byte count; GB300 matches the good variant bitwise and hangs on SW1.
#[test]
fn e_stasync_wide_widths() {
    both("e_stasync_wide", "good");
    both("e_stasync_wide", "SW1");
}

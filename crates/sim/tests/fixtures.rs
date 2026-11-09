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
                    let dir = scratch_dir(&case.m.name, variant);
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
            let dir = scratch_dir(&case.m.name, variant);
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
/// mov.b128 q, {lo, hi} / mov.b128 {lo, hi}, q (CuTe DSL's CLC response from .v2.b64).
#[test]
fn r1d_v2pack() {
    check_dpor("r1d_clc", "v2pack", "ptx", 2_000_000);
    check_dpor("r1d_clc", "v2pack", "sass", 2_000_000);
}
/// CLC on a 3-D grid: get_first_ctaid::z and .v4 (ctaid.z in the high half of word 1).
#[test]
fn r1d_clc3d() {
    for v in ["good", "v4"] {
        check_dpor("r1d_clc3d", v, "ptx", 2_000_000);
        check_dpor("r1d_clc3d", v, "sass", 2_000_000);
    }
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

// Rung 1h: NVLS multicast all-reduce on 2 GPUs (multimem.red barrier, multimem.ld_reduce;
// S-MULTIMEM, S-MC-ALIAS), both explorers.
#[test]
fn r1h_nvls() {
    for v in ["good", "B23", "B63", "B64", "B65"] {
        for f in ["ptx", "sass"] {
            // B65 is PTX undefined behavior; in SASS its STG is a multimem store whose two
            // ranks' bytes collide (race + nondeterministic: the explorers disagree on the
            // second class), so only the PTX verdict is checked.
            if v == "B65" && f == "sass" {
                continue;
            }
            check_dir("fixtures", "r1h_nvls", v, f);
        }
    }
}

// Rung 5a: put-with-signal over an IBGDA queue pair between two NVLink domains (nic.rs,
// docs/nic.md).
#[test]
fn r5a_ibgda_putsig() {
    for v in [
        "good", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "T", "N9",
    ] {
        for f in ["ptx", "sass"] {
            check_dir("fixtures", "r5a_ibgda_putsig", v, f);
        }
    }
}

/// E-MC-ALIAS-HW (opt-in): without the alias-proxy check the missing fence.proxy.alias (B23)
/// is clean; the scope / ordering mutants still race (B64 in SASS: the alias fence's
/// MEMBAR.SC.SYS releases, see the case).
#[test]
fn r1h_nvls_alias_hw() {
    for (v, f, want) in [
        ("good", "ptx", "clean"),
        ("good", "sass", "clean"),
        ("B23", "ptx", "clean"),
        ("B23", "sass", "clean"),
        ("B63", "ptx", "race"),
        ("B63", "sass", "race"),
        ("B64", "ptx", "race"),
        ("B64", "sass", "clean"),
    ] {
        let case = Case::load(root().join("fixtures/r1h_nvls")).unwrap();
        let k = kernel(&case, v, f);
        let mut launch = build_launch(&case, &k).unwrap();
        launch.mc_alias_hw = true;
        let m0 = Machine::new(Arc::new(k), Arc::new(launch));
        let r = explore(m0, 2_000_000);
        assert!(r.complete, "{v}/{f}: incomplete");
        assert_eq!(
            r.classes(),
            vec![want.to_string()],
            "{v}/{f}: {:#?}",
            r.findings
        );
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
            // The front's own expectation (a case's `expect_sass` differs from its PTX one).
            let expect = case.variant("good").unwrap().expect_for(front);
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

/// A scratch directory unique to this process and call: tests that share a case name
/// (e.g. `r1h_nvls` and `r1h_nvls_alias_hw`) and concurrent suites must not share files.
fn scratch_dir(name: &str, variant: &str) -> std::path::PathBuf {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "ptxv-test-{}-{n}-{name}-{variant}",
        std::process::id()
    ))
}

/// A binary-only case's variant: its cubin plus the variant's control-bit patch.
fn binary_kernel(case: &Case, variant: &str) -> anyhow::Result<ptxv_ir::Kernel> {
    let dir = scratch_dir(&case.m.name, variant);
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

/// L-ROSETTA, E-SETMAXNREG: the PTX forms of the mslk decode kernels (fixtures/s_ptx_packed:
/// fma / mul / add / sub .rn.f32x2, cvt.rn.{f16x2,bf16x2}.f32, div.s32, cp.async.ca / .cg,
/// ld / st .v2.b64, vector operands whose registers are not consecutive, setmaxnreg) agree with
/// GB300 bitwise on both front-ends (run 2026-11-06, GPU 3, 50/50 runs), including subnormal,
/// infinite and NaN lanes.
#[test]
fn s_ptx_packed_matches_hardware() {
    let case = Case::load(root().join("fixtures/s_ptx_packed")).unwrap();
    let expected = std::fs::read(root().join("fixtures/s_ptx_packed/expected/out.bin")).unwrap();
    assert_eq!(fnv(&expected), 0xee08a5ce622b0a7b);
    for front in ["ptx", "sass"] {
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
            .verdict
            .expect("s_ptx_packed certifies");
        assert_eq!(v.classes(), ["clean"], "{front}: {:#?}", v.findings);
        assert_eq!(v.outputs[0], expected, "{front}: differs from GB300");
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

/// Shared-memory address operands add in 32 bits (S-SMEM-ADDR32): probes/smem_addr_wrap,
/// `LDS R7, [R4+UR4+0x28]` with R4 = 0xffffffd8 in clusters of 2, against GB300 (run
/// 2026-11-06, 20 runs). A 64-bit sum names CTA rank 256 and faults.
#[test]
fn probe_smem_addr_wrap_matches_hardware() {
    let case = Case::load(root().join("probes/smem_addr_wrap")).unwrap();
    let k = binary_kernel(&case, "good").unwrap();
    let launch = build_launch(&case, &k).unwrap();
    let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 200_000);
    assert!(v.complete && v.findings.is_empty(), "{:#?}", v.findings);
    assert_eq!(v.outputs.len(), 1);
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
        "{report:?}"
    );
}

/// Unordered plain stores of one value (S-SAME-VALUE): probes/same_value, 2 CTAs x 4 warps
/// storing v to one shared and one global word, GB300 out = v in 1000 / 1000 runs (16 CTAs x
/// 8 warps: 200 / 200), 2026-11-06. `good` is clean with notes and GB300's output; `differ`
/// (warp w stores v + w) stays a race on both front-ends.
#[test]
fn probe_same_value_matches_hardware() {
    let case = Case::load(root().join("probes/same_value")).unwrap();
    // The stateful explorer only: DPOR does not finish on this probe's cross-CTA global
    // stores within the test's time.
    for f in ["ptx", "sass"] {
        let k = kernel(&case, "differ", f);
        let launch = build_launch(&case, &k).unwrap();
        let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 200_000);
        assert!(v.complete, "differ/{f}: incomplete");
        assert_eq!(v.classes(), ["nondeterministic", "race"], "differ/{f}");
    }
    for f in ["ptx", "sass"] {
        let k = kernel(&case, "good", f);
        let launch = build_launch(&case, &k).unwrap();
        let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 200_000);
        assert!(v.complete, "good/{f}: incomplete");
        assert_eq!(v.classes(), ["clean"], "good/{f}");
        assert!(v.findings.iter().any(|x| x.kind == "note"), "{f}: no note");
        assert_eq!(v.outputs.len(), 1);
        let report = case.expect_report(&v.outputs[0]).unwrap();
        assert!(
            !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
            "{f}: {report:?}"
        );
    }
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

/// S-TMEM: tcgen05.st.16x256b / .16x128b (probes/tmem_st16; SASS STTM.16dp256bit /
/// .16dp128bit) write the cells the same-shape loads read: both front-ends = GB300.
#[test]
fn probe_tmem_st16_matches_hardware() {
    let case = Case::load(root().join("probes/tmem_st16")).unwrap();
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

/// S-TMAP: a tensor map built on the device with tensormap.replace (derived words 0x40 / 0x48
/// left 0), published with tensormap.cp_fenceproxy and used by a TMA load = GB300
/// (probes/tmap_device: the box and the published map bytes) on both front-ends.
#[test]
fn probe_tmap_device_matches_hardware() {
    probe_bitwise("probes/tmap_device", "good", "ptx");
    probe_bitwise("probes/tmap_device", "good", "sass");
}

/// S-TMAP: tensormap.replace on every field / ordinal / enum value over random map bytes, and
/// tensormap.cp_fenceproxy (probes/tmap_replace) = GB300 on both front-ends.
#[test]
fn probe_tmap_replace_matches_hardware() {
    probe_bitwise("probes/tmap_replace", "good", "ptx");
    probe_bitwise("probes/tmap_replace", "good", "sass");
}

/// S-TMAP: a driver-encoded map copied out of the kernel parameters (its raw bytes), patched
/// with tensormap.replace, published and used by a TMA load (probes/tmap_param_copy) on both
/// front-ends.
#[test]
fn probe_tmap_param_copy_matches_hardware() {
    probe_bitwise("probes/tmap_param_copy", "good", "ptx");
    probe_bitwise("probes/tmap_param_copy", "good", "sass");
}

/// L-ROSETTA: atom.global.inc.u32 (SASS ATOMG.E.INC) = GB300 (probes/atom_inc), both
/// front-ends.
#[test]
fn probe_atom_inc_matches_hardware() {
    probe_bitwise("probes/atom_inc", "good", "ptx");
    probe_bitwise("probes/atom_inc", "good", "sass");
}

/// L-ROSETTA: LDSM.8.MT1616[.2] (ldmatrix.m16n16.x1 / .x2.trans.b8; probes/ldsm_b8) = GB300
/// on both fronts.
#[test]
fn probe_ldsm_b8_matches_hardware() {
    probe_bitwise("probes/ldsm_b8", "good", "sass");
    probe_bitwise("probes/ldsm_b8", "good", "ptx");
}

/// S-TMEM: tcgen05.st / ld .16x32bx2 with a split offset (probes/tmem_x2; SASS STTM / LDTM
/// .16dp32bit_t0_t15 + _t16_t31) = GB300, both front-ends.
#[test]
fn probe_tmem_x2_matches_hardware() {
    probe_bitwise("probes/tmem_x2", "good", "ptx");
    probe_bitwise("probes/tmem_x2", "good", "sass");
}

/// S-TMEM: LDTM.STAT.16dp32bit_t0_t15 / _t16_t31 .xN.MIN|MAX[ABS].F32[.NAN]
/// (tcgen05.ld.red.16x32bx2; probes/tmem_ldred16) = GB300 on both front-ends (the PTX one
/// lowers kernel.ptx's tcgen05.ld.red to TmemLd with `red`).
#[test]
fn probe_tmem_ldred16_matches_hardware() {
    probe_bitwise_cubin("probes/tmem_ldred16", Some("kernel.cubin"));
    let case = Case::load(root().join("probes/tmem_ldred16")).unwrap();
    let v = certify_ptx_file(&case);
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
        "ptx: {report:?}"
    );
}

/// tcgen05.ld.red forms beyond 32x32b / 16x32bx2 min / max .f32 are refused.
#[test]
fn tmem_ldred_ptx_other_forms_refused() {
    let src = std::fs::read_to_string(root().join("probes/tmem_ldred/kernel.ptx")).unwrap();
    for (from, to) in [
        ("32x32b.x32.max.f32", "16x64b.x32.max.f32"),
        ("32x32b.x32.max.f32", "32x32b.x32.max.min.f32"),
        ("32x32b.x32.max.f32", "32x32b.x32.max.u32"),
    ] {
        let bad = src.replacen(from, to, 1);
        assert_ne!(bad, src);
        assert!(
            ptxv_ptx::parse_kernel(&bad, Some("probe_tmem_ldred")).is_err(),
            "{to}"
        );
    }
}

/// Certify a cubin-only case's kernel.ptx on the PTX front-end (clean, one output).
fn certify_ptx_file(case: &Case) -> ptxv_sim::explore::Verdict {
    let src = std::fs::read_to_string(case.dir.join("kernel.ptx")).unwrap();
    let k = ptxv_ptx::parse_kernel(&src, Some(&case.m.entry)).unwrap();
    let launch = build_launch(case, &k).unwrap();
    let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
        .verdict
        .unwrap_or_else(|| panic!("{} ptx certifies", case.m.name));
    assert_eq!(v.classes(), ["clean"], "{:#?}", v.findings);
    assert_eq!(v.outputs.len(), 1);
    v
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

/// PTX mma.sync m16n8k16 bf16 / m16n8k8 tf32 (lowered to `Op::Hmma`) on probes/hmma and
/// probes/tmma = GB300 (their kernel.ptx assembled by ptxas: the same output, 20/20 scrubbed).
#[test]
fn probe_mma_sync_ptx_matches_hardware() {
    probe_bitwise("probes/hmma", "good", "ptx");
    probe_bitwise("probes/tmma", "good", "ptx");
}

/// N-HMMA, m16n8k8 bf16 (SASS HMMA.1688.F32.BF16): probes/bmma8 through both front-ends =
/// GB300 for the default inputs (`ptxv calib-mma probes/bmma8 --step 8`: the k16 model matches
/// all 98,304 outputs).
#[test]
fn probe_bmma8_matches_hardware() {
    probe_bitwise_cubin("probes/bmma8", Some("kernel.cubin"));
    probe_bitwise("probes/bmma8", "good", "ptx");
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
/// barrier, by all threads or by the issuer only; 1000/1000 wrong without). `global`: a
/// fence.proxy.async.global (FENCE.VIEW.ASYNC.G) orders global memory only: a proxy race.
#[test]
fn probe_proxy_fence_matches_hardware() {
    for v in ["good", "after_all", "issuer", "none", "global"] {
        check_dir("probes", "proxy_fence", v, "ptx");
        check_dir("probes", "proxy_fence", v, "sass");
    }
}

/// E-CPASYNC-PROXY: cp.async writes then a TMA store with no fence.proxy.async is a contract
/// finding, st.shared writers a proxy race (probes/proxy_cpasync; GB300 1000/1000 correct in
/// every variant).
#[test]
fn probe_proxy_cpasync_verdicts() {
    for v in ["good", "none", "st_none"] {
        check_dir("probes", "proxy_cpasync", v, "ptx");
        check_dir("probes", "proxy_cpasync", v, "sass");
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

/// A probe's variant certifies clean on front-end `front` and every output equals the GB300
/// bytes recorded in its `expected/`.
fn probe_bitwise(dir: &str, variant: &str, front: &str) {
    let case = Case::load(root().join(dir)).unwrap();
    let k = kernel(&case, variant, front);
    let launch = build_launch(&case, &k).unwrap();
    let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
        .verdict
        .unwrap_or_else(|| panic!("{dir} {variant} {front} certifies"));
    assert_eq!(v.classes(), ["clean"], "{dir} {front}: {:#?}", v.findings);
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
        "{dir} {variant} {front}: {report:?}"
    );
}

/// N-BF16X2: packed bf16 arithmetic (HMUL2 / HADD2 / HFMA2 .BF16_V2: lane selectors, -RZ
/// addend, immediate pairs, .RELU) and bf16 atomics (ATOMG / REDG .ADD.BF16x2.RN, the scalar
/// form's CAS loop) on edge / tie / random operands = GB300 (probes/bf16x2). The PTX replay
/// uses the `.rn` spelling of the non-`.rn` lines (variant `rn`; same cubin bits).
#[test]
fn probe_bf16x2_matches_hardware() {
    probe_bitwise("probes/bf16x2", "good", "sass");
    probe_bitwise("probes/bf16x2", "rn", "ptx");
}

/// N-BF16X2, N-FALU: FHFMA / FHADD .BF16 in every rounding mode and .SAT, HMNMX2.BF16_V2
/// [.NAN] [.XORSIGN], FFMA2 / FMUL2 / FADD2 .RZ / .RM / .RP = GB300 (probes/bf16_ext).
#[test]
fn probe_bf16_ext_matches_hardware() {
    probe_bitwise("probes/bf16_ext", "good", "ptx");
    probe_bitwise("probes/bf16_ext", "good", "sass");
}

/// N-XORSIGN: min / max [.NaN] [.ftz] .xorsign.abs.f32 (FMNMX.XORSIGN with |R|, immediate
/// and |UR| operands) on every pair of 32 edge patterns = GB300 (probes/fmnmx_xorsign).
#[test]
fn probe_fmnmx_xorsign_matches_hardware() {
    probe_bitwise("probes/fmnmx_xorsign", "good", "ptx");
    probe_bitwise("probes/fmnmx_xorsign", "good", "sass");
}

/// REDUX.SUM.S32 and the other redux.sync forms, ATOMG / REDG .ADD.S32 (probes/redux_s32);
/// with partial active-lane masks behind BRA.DIV, SASS only (probes/redux_s32/masks).
#[test]
fn probe_redux_s32_matches_hardware() {
    probe_bitwise("probes/redux_s32", "good", "ptx");
    probe_bitwise("probes/redux_s32", "good", "sass");
    probe_bitwise("probes/redux_s32/masks", "good", "sass");
}

/// sqrt.rn.f32 on 4096 patterns (probes/sqrt_rn): GB300 = IEEE (NaN 0x7fffffff); the PTX front
/// and ptxas's MUFU.RSQ / Newton / slow-path code on the SASS front both match.
#[test]
fn probe_sqrt_rn_matches_hardware() {
    probe_bitwise("probes/sqrt_rn", "good", "ptx");
    probe_bitwise("probes/sqrt_rn", "good", "sass");
}

/// brx.idx through .branchtargets (probes/brx_idx; `good` = GB300 on both front-ends; `oob`:
/// an index outside the list, a fault on both).
#[test]
fn probe_brx_idx_verdicts() {
    probe_bitwise("probes/brx_idx", "good", "ptx");
    probe_bitwise("probes/brx_idx", "good", "sass");
    check_dir("probes", "brx_idx", "oob", "ptx");
    check_dir("probes", "brx_idx", "oob", "sass");
}

/// 256-bit global accesses with L2 eviction hints (LDG / STG .ELL2.256 / .EFL2.256;
/// probes/ldst256_l2) = GB300 on both front-ends.
#[test]
fn probe_ldst256_l2_matches_hardware() {
    probe_bitwise("probes/ldst256_l2", "good", "ptx");
    probe_bitwise("probes/ldst256_l2", "good", "sass");
}

/// STTM with a negative TMEM offset (probes/tmem_stneg) = GB300 on both front-ends.
#[test]
fn probe_tmem_stneg_matches_hardware() {
    probe_bitwise("probes/tmem_stneg", "good", "ptx");
    probe_bitwise("probes/tmem_stneg", "good", "sass");
}

/// bar.warp.sync on per-lane pair masks (probes/warpsync_pairs) = GB300 on both front-ends: the
/// PTX front's plain WARPSYNC releases each mask group on its own; on the SASS front (ptxas
/// 13.0: WARPSYNC.COLLECTIVE per call site) the two lanes of a pair at different collective
/// instructions enter together (E-COLLECTIVE).
#[test]
fn probe_warpsync_pairs_verdicts() {
    probe_bitwise("probes/warpsync_pairs", "good", "ptx");
    probe_bitwise("probes/warpsync_pairs", "good", "sass");
}

/// PTX-FMAD: plain mul + add / sub contracted as ptxas 13.0 does, and the forms it keeps apart
/// (probes/fmad) = GB300 on both front-ends.
#[test]
fn probe_fmad_matches_hardware() {
    probe_bitwise("probes/fmad", "good", "ptx");
    probe_bitwise("probes/fmad", "good", "sass");
}

/// Collectives whose pair mask is reached at different instructions (probes/coll_split): GB300
/// combines them. SASS: the model runs the identical WARPSYNC.COLLECTIVE regions as one; PTX:
/// each mask group runs with its lanes at other shfl.sync / vote.sync instructions (E-WARPCOLL).
#[test]
fn probe_coll_split_matches_hardware() {
    probe_bitwise_cubin("probes/coll_split", Some("kernel.cubin"));
    probe_bitwise("probes/coll_split", "good", "ptx");
}

/// PTX shfl.sync / vote.sync / redux.sync with partial member masks (probes/shfl_mask): half-warp
/// and quad masks, mask groups split across instructions = GB300 on the PTX front (the SASS
/// front refuses ptxas's WARPSYNC.EXCLUSIVE); lanes outside the mask executing it: violation.
#[test]
fn probe_shfl_mask_verdicts() {
    probe_bitwise("probes/shfl_mask", "good", "ptx");
    check_dir("probes", "shfl_mask", "good", "ptx");
    check_dir("probes", "shfl_mask", "skip_half", "ptx");
}

/// E-COLLECTIVE: `BRA.DIV 0xffffffff` falling through orders the converged lanes as the slow
/// path's WARPSYNC.COLLECTIVE does (probes/syncwarp_fast, DeepEP's `__syncwarp()` + elect
/// shape) = GB300 on both front-ends.
#[test]
fn probe_syncwarp_fast_matches_hardware() {
    probe_bitwise("probes/syncwarp_fast", "good", "ptx");
    probe_bitwise("probes/syncwarp_fast", "good", "sass");
}

/// PTX copysign.f32 (probes/copysign: NaN payloads, signed zeros, Inf, subnormals) = GB300
/// on both front-ends.
#[test]
fn probe_copysign_matches_hardware() {
    probe_bitwise("probes/copysign", "good", "ptx");
    probe_bitwise("probes/copysign", "good", "sass");
}

/// cvt.rn / .rz [.relu] [.satfinite] .tf32.f32 (F2FP [.SATFINITE] [.RELU] .TF32.F32.PACK_B
/// [.RZ]; probes/cvt_tf32_rn, 65,536 words x 8 forms) = GB300 on both front-ends.
#[test]
fn probe_cvt_tf32_rn_matches_hardware() {
    probe_bitwise("probes/cvt_tf32_rn", "good", "ptx");
    probe_bitwise("probes/cvt_tf32_rn", "good", "sass");
}

/// PTX cvt.rna.tf32.f32 (probes/cvt_tf32: ties, NaN payloads, Inf, subnormals) = GB300 on
/// both front-ends.
#[test]
fn probe_cvt_tf32_matches_hardware() {
    probe_bitwise("probes/cvt_tf32", "good", "ptx");
    probe_bitwise("probes/cvt_tf32", "good", "sass");
}

/// S-INIT: discard.global.L2 makes the line indeterminate (probes/discard_l2): rewritten
/// before the read = GB300 on the PTX front; read after the discard: a violation.
#[test]
fn probe_discard_l2_verdicts() {
    probe_bitwise("probes/discard_l2", "good", "ptx");
    check_dir("probes", "discard_l2", "read_after", "ptx");
}

/// tcgen05.mma cta_group::2 M = 128 with A in TMEM (probes/tc_pair_atmem) = GB300 on both
/// front-ends.
#[test]
fn probe_tc_pair_atmem_matches_hardware() {
    probe_bitwise("probes/tc_pair_atmem", "good", "ptx");
    probe_bitwise("probes/tc_pair_atmem", "good", "sass");
}

/// tcgen05.mma kind::tf32 MN-major through SWIZZLE_128B_BASE32B (probes/tc_tf32_mn) = GB300 on
/// both front-ends.
#[test]
fn probe_tc_tf32_mn_matches_hardware() {
    probe_bitwise("probes/tc_tf32_mn", "good", "ptx");
    probe_bitwise("probes/tc_tf32_mn", "good", "sass");
}

/// tcgen05.mma finite overflow to Inf (probes/tc_mma_ovf) = GB300 on both front-ends.
#[test]
fn probe_tc_mma_ovf_matches_hardware() {
    probe_bitwise("probes/tc_mma_ovf", "good", "ptx");
    probe_bitwise("probes/tc_mma_ovf", "good", "sass");
}

/// HMMA bf16 on subnormal / Inf / NaN operands and finite overflow (probes/hmma_special) =
/// GB300 on both front-ends.
#[test]
fn probe_hmma_special_matches_hardware() {
    probe_bitwise("probes/hmma_special", "good", "ptx");
    probe_bitwise("probes/hmma_special", "good", "sass");
}

/// mma.sync m16n8k16 f16 / HMMA.16816.F32 (probes/hmma_f16) = GB300 on both front-ends.
#[test]
fn probe_hmma_f16_matches_hardware() {
    probe_bitwise("probes/hmma_f16", "good", "ptx");
    probe_bitwise("probes/hmma_f16", "good", "sass");
}

/// cvt.bf16.f16 / cvt.f16.bf16 without a rounding modifier on every pattern
/// (probes/cvt_bf16_f16) = GB300 on both front-ends.
#[test]
fn probe_cvt_bf16_f16_matches_hardware() {
    probe_bitwise("probes/cvt_bf16_f16", "good", "ptx");
    probe_bitwise("probes/cvt_bf16_f16", "good", "sass");
}

/// rsqrt.approx.f64 without .ftz (probes/rsqrt_f64) = GB300 on both front-ends.
#[test]
fn probe_rsqrt_f64_matches_hardware() {
    probe_bitwise("probes/rsqrt_f64", "good", "ptx");
    probe_bitwise("probes/rsqrt_f64", "good", "sass");
}

/// min / max / neg / abs .f64 (probes/f64_misc) = GB300 on both front-ends.
#[test]
fn probe_f64_misc_matches_hardware() {
    probe_bitwise("probes/f64_misc", "good", "ptx");
    probe_bitwise("probes/f64_misc", "good", "sass");
}

/// div.rn.ftz.f32 / sqrt.rn.ftz.f32 (probes/ftz_div_sqrt) = GB300 on both front-ends.
#[test]
fn probe_ftz_div_sqrt_matches_hardware() {
    probe_bitwise("probes/ftz_div_sqrt", "good", "ptx");
    probe_bitwise("probes/ftz_div_sqrt", "good", "sass");
}

/// 64-bit and vector ld / st .local (probes/local_vec) = GB300 on both front-ends.
#[test]
fn probe_local_vec_matches_hardware() {
    probe_bitwise("probes/local_vec", "good", "ptx");
    probe_bitwise("probes/local_vec", "good", "sass");
}

/// Cluster special registers (probes/cluster_sreg) = GB300 on both front-ends.
#[test]
fn probe_cluster_sreg_matches_hardware() {
    probe_bitwise("probes/cluster_sreg", "good", "ptx");
    probe_bitwise("probes/cluster_sreg", "good", "sass");
}

/// Byte packing / unpacking moves (probes/mov_pack) = GB300 on both front-ends.
#[test]
fn probe_mov_pack_matches_hardware() {
    probe_bitwise("probes/mov_pack", "good", "ptx");
    probe_bitwise("probes/mov_pack", "good", "sass");
}

/// tcgen05.mma with an immediate enable-input-d (probes/tc_mma_imm) = GB300 on both
/// front-ends.
#[test]
fn probe_tc_mma_imm_matches_hardware() {
    probe_bitwise("probes/tc_mma_imm", "good", "ptx");
    probe_bitwise("probes/tc_mma_imm", "good", "sass");
}

/// bfi.b32 / clz.b32 (probes/bfi_clz) = GB300 on both front-ends.
#[test]
fn probe_bfi_clz_matches_hardware() {
    probe_bitwise("probes/bfi_clz", "good", "ptx");
    probe_bitwise("probes/bfi_clz", "good", "sass");
}

/// cvt.ftz.sat.f32.f32 / cvt.ftz.f32.f32 (probes/cvt_ftz) = GB300 on both front-ends.
#[test]
fn probe_cvt_ftz_matches_hardware() {
    probe_bitwise("probes/cvt_ftz", "good", "ptx");
    probe_bitwise("probes/cvt_ftz", "good", "sass");
}

/// Decimal / cross-width float literals (probes/float_lit) = GB300 on both front-ends.
#[test]
fn probe_float_lit_matches_hardware() {
    probe_bitwise("probes/float_lit", "good", "ptx");
    probe_bitwise("probes/float_lit", "good", "sass");
}

/// mul.f32x2 contracted into add / sub.f32x2 of the same rounding mode, .rn included
/// (probes/fmad2) = GB300 on both front-ends.
#[test]
fn probe_fmad2_matches_hardware() {
    probe_bitwise("probes/fmad2", "good", "ptx");
    probe_bitwise("probes/fmad2", "good", "sass");
}

/// mma.sync m16n8k32 s8 satfinite / IMMA.16832.S8.S8.SAT (probes/imma_s8) = GB300 on both
/// front-ends, saturating on both s32 limits.
#[test]
fn probe_imma_s8_matches_hardware() {
    probe_bitwise("probes/imma_s8", "good", "ptx");
    probe_bitwise("probes/imma_s8", "good", "sass");
}

/// Plain mul.bf16 / .bf16x2 reaching mma / cvt through packs, prmt and shfl
/// (probes/bf16_sink) = GB300 on both front-ends.
#[test]
fn probe_bf16_sink_matches_hardware() {
    probe_bitwise("probes/bf16_sink", "good", "ptx");
    probe_bitwise("probes/bf16_sink", "good", "sass");
}

/// tcgen05.mma.kind::f16 with f16 operands and kind::tf32 (instruction descriptor A / B
/// format 0 / 2): probes/tc_f16 and probes/tc_tf32 = GB300 on both front-ends (the models:
/// `ptxv calib-mma` on each, N-MMA).
#[test]
fn probe_tc_f16_tf32_match_hardware() {
    for p in ["probes/tc_f16", "probes/tc_tf32"] {
        probe_bitwise(p, "good", "ptx");
        probe_bitwise(p, "good", "sass");
    }
}

/// mbarrier.arrive with an arrival count (SASS SYNCS.ARRIVE.TRANS64.ART0 / .RED.ART0;
/// probes/mbar_count; `good` = GB300 on both front-ends; `over`: a count above the pending
/// count, a fault, as on GB300).
#[test]
fn probe_mbar_count_verdicts() {
    probe_bitwise("probes/mbar_count", "good", "ptx");
    probe_bitwise("probes/mbar_count", "good", "sass");
    check_dir("probes", "mbar_count", "over", "ptx");
    check_dir("probes", "mbar_count", "over", "sass");
}

/// redux.sync with membermasks naming a subset of the warp, executed by exactly those lanes
/// (probes/redux_mask; `good` = GB300 on both front-ends; `mask_mismatch`: every lane runs a
/// redux whose mask is 0xaaaaaaaa, a PTX violation; ptxas's REDUX has no mask, so the SASS
/// verdict is clean, and GB300 gives every lane the full-warp sum).
#[test]
fn probe_redux_mask_verdicts() {
    probe_bitwise("probes/redux_mask", "good", "ptx");
    probe_bitwise("probes/redux_mask", "good", "sass");
    check_dir("probes", "redux_mask", "mask_mismatch", "ptx");
    check_dir("probes", "redux_mask", "mask_mismatch", "sass");
}

/// IMAD.WIDE[.U32] with a negated 64-bit addend, POPC read after a wait on a later POPC's
/// scoreboard (probes/int_misc) = GB300.
#[test]
fn probe_int_misc_matches_hardware() {
    probe_bitwise("probes/int_misc", "good", "ptx");
    probe_bitwise("probes/int_misc", "good", "sass");
}

/// 32-byte global loads and stores (ld / ld.nc / st .global.v8.u32 as LDG / STG .E.ENL2.256):
/// operand order and register quads (probes/ldst256) = GB300.
#[test]
fn probe_ldst256_matches_hardware() {
    probe_bitwise("probes/ldst256", "good", "ptx");
    probe_bitwise("probes/ldst256", "good", "sass");
}

/// A cubin probe (no PTX source) certifies clean on the SASS front-end and equals `expected/`.
fn probe_bitwise_cubin(dir: &str, cubin: Option<&str>) {
    let case = Case::load(root().join(dir)).unwrap();
    let k = match cubin {
        Some(c) => ptxv_sass::load_cubin(&case.dir.join(c), &case.m.entry).unwrap(),
        None => binary_kernel(&case, "good").unwrap(),
    };
    let launch = build_launch(&case, &k).unwrap();
    let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
        .verdict
        .unwrap_or_else(|| panic!("{dir} {cubin:?} certifies"));
    assert_eq!(v.classes(), ["clean"], "{dir} {cubin:?}: {:#?}", v.findings);
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
        "{dir} {cubin:?}: {report:?}"
    );
}

/// N-FCVT: FRND (f32 [.FTZ], .F64, .F16, .BF16 x four roundings), F2F.{BF16,F16}.F64[.R*],
/// F2F.F64.{BF16,F16} [.H1], F2F.BF16.F16 / F2F.F16.BF16 [.R*] [.H1] on edge, tie and random
/// inputs = GB300 on both front-ends (probes/fcvt).
#[test]
fn probe_fcvt_matches_hardware() {
    probe_bitwise("probes/fcvt", "good", "ptx");
    probe_bitwise("probes/fcvt", "good", "sass");
}

/// N-INT16X2: VIMNMX / VIMNMX3 .U16x2 / .S16x2 (register and immediate) and the 64-bit
/// `(sext(a) << s) - y` = GB300 (probes/int16x2), both front-ends; the same kernel with its
/// `IMAD.X ~Ry` high words rewritten into `LEA.HI.X.SX32 Rd, Ra, ~Ry, s, P` (cubins/lea.cubin)
/// gives the same bytes.
#[test]
fn probe_int16x2_matches_hardware() {
    probe_bitwise("probes/int16x2", "good", "ptx");
    probe_bitwise("probes/int16x2", "good", "sass");
    probe_bitwise_cubin("probes/int16x2", Some("cubins/lea.cubin"));
}

/// N-BULK-RED: cp.reduce.async.bulk ... .add.f32 (UBLKRED.G.S.ADD.F32.RN) on edge pairs, and
/// four CTAs' concurrent bulk adds into the same words (atomic per word: no race, one result)
/// = GB300 (probes/bulk_red), both front-ends.
#[test]
fn probe_bulk_red_matches_hardware() {
    // Explored, not certified (the certificate does not cover asynchronous reductions): every
    // order of the four concurrent bulk adds, one output, equal to GB300's.
    let case = Case::load(root().join("probes/bulk_red")).unwrap();
    for front in ["ptx", "sass"] {
        let k = Arc::new(kernel(&case, "good", front));
        let launch = Arc::new(build_launch(&case, &k).unwrap());
        let v = ptxv_sim::explore::explore(Machine::new(k, launch), 200_000);
        assert!(v.complete, "{front}: incomplete");
        assert_eq!(v.classes(), ["clean"], "{front}: {:#?}", v.findings);
        assert_eq!(v.distinct_outputs, 1, "{front}");
        let report = case.expect_report(&v.outputs[0]).unwrap();
        assert!(
            !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
            "{front}: {report:?}"
        );
    }
}

/// N-F2I16: F2I[.FTZ].{S16,U16}[.TRUNC|.FLOOR|.CEIL].NTZ (16-bit and widened results) and
/// the CCTL.E.PF1 / PF2 prefetch hints = GB300 (probes/f2i16), both front-ends.
#[test]
fn probe_f2i16_matches_hardware() {
    probe_bitwise("probes/f2i16", "good", "ptx");
    probe_bitwise("probes/f2i16", "good", "sass");
}

/// E-BSYNC: `BREAK.RELIABLE Pc, Bb` in ptxas's div.rn.ftz.f32 slow path (probes/div_ftz, every
/// pair of 32 edge patterns) = GB300 (SASS; the BREAK exists only there).
#[test]
fn probe_div_ftz_matches_hardware() {
    probe_bitwise("probes/div_ftz", "good", "sass");
}

/// N-INT16X2: VIMNMX.RELU, VIMNMX.S16x2.RELU, VIADDMNMX.RELU (min / max.relu, an add fused
/// with min.relu) = GB300 (probes/imnmx_relu), both front-ends.
#[test]
fn probe_imnmx_relu_matches_hardware() {
    probe_bitwise("probes/imnmx_relu", "good", "ptx");
    probe_bitwise("probes/imnmx_relu", "good", "sass");
}

/// N-CVT-LP: F2FP[.RELU].F16.E2M1.UNPACK_B Rd, Ra.B1 / .B2 / .B3 (cvt.rn[.relu].f16x2.e2m1x2
/// of byte k of a register) = GB300 (probes/e2m1_bytes), both front-ends.
#[test]
fn probe_e2m1_bytes_matches_hardware() {
    probe_bitwise("probes/e2m1_bytes", "good", "ptx");
    probe_bitwise("probes/e2m1_bytes", "good", "sass");
}

/// N-BULK-RED: UTMAREDG.2D.ADD on bf16, f16 and f32 tensors (every pair of 64 edge patterns)
/// = GB300 (probes/tma_red), both front-ends.
#[test]
fn probe_tma_red_matches_hardware() {
    let case = Case::load(root().join("probes/tma_red")).unwrap();
    for front in ["ptx", "sass"] {
        let k = Arc::new(kernel(&case, "good", front));
        let launch = Arc::new(build_launch(&case, &k).unwrap());
        let v = ptxv_sim::explore::explore(Machine::new(k, launch), 200_000);
        assert!(v.complete, "{front}: incomplete");
        assert_eq!(v.classes(), ["clean"], "{front}: {:#?}", v.findings);
        let report = case.expect_report(&v.outputs[0]).unwrap();
        assert!(
            !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
            "{front}: {report:?}"
        );
    }
}

/// S-TMA: UBLKCP.S.S (cp.async.bulk.shared::cluster.shared::cta) to the peer CTA and to the
/// issuing CTA of a 2-CTA cluster = GB300 (probes/bulk_s2s), both front-ends.
#[test]
fn probe_bulk_s2s_matches_hardware() {
    let case = Case::load(root().join("probes/bulk_s2s")).unwrap();
    for front in ["ptx", "sass"] {
        let k = Arc::new(kernel(&case, "good", front));
        let launch = Arc::new(build_launch(&case, &k).unwrap());
        let v = ptxv_sim::explore::explore(Machine::new(k, launch), 500_000);
        assert!(v.complete, "{front}: incomplete");
        assert_eq!(v.classes(), ["clean"], "{front}: {:#?}", v.findings);
        let report = case.expect_report(&v.outputs[0]).unwrap();
        assert!(
            !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
            "{front}: {report:?}"
        );
    }
}

/// L-RPCMOV: `RPCMOV.32 Rpc.LO, Ra` then `RPCMOV.32 Rd, Rpc.LO` restore each thread's own
/// value = GB300 (probes/rpcmov, a patched cubin).
#[test]
fn probe_rpcmov_matches_hardware() {
    probe_bitwise_cubin("probes/rpcmov", None);
}

/// N-FCHK: `FCHK P, -Ra, Rb` (probes/fchk_neg, probes/fchk's cubin with the dividend's negate
/// bit set) gives probes/fchk's GB300 output hash: the sign is ignored.
#[test]
fn probe_fchk_neg_matches_hardware() {
    certified_hash("probes", "fchk_neg", 0xa83533de36c447d4);
}

/// N-FCHK: `FCHK P, |Ra|, |Rb|` (probes/fchk_abs, probes/fchk's cubin with both abs bits set)
/// = GB300: the same output as probes/fchk_neg.
#[test]
fn probe_fchk_abs_matches_hardware() {
    certified_hash("probes", "fchk_abs", 0xa83533de36c447d4);
}

/// The f64 pipe completes in order: an unscoreboarded DFMA covered by a later DADD's
/// scoreboard, and a DMUL overwriting a register an earlier DADD reads (probes/f64_queue) =
/// GB300.
#[test]
fn probe_f64_queue_matches_hardware() {
    probe_bitwise("probes/f64_queue", "good", "ptx");
    probe_bitwise("probes/f64_queue", "good", "sass");
    // L-OPCLASS on sm_100a: the same kernel built for sm_100a (ptxas 13.0, cubins/sm100a.cubin)
    // covers its f64 results with stall counts only (fixed-latency f64 pipe); the outputs are
    // IEEE-exact, so GB300's expected/ applies.
    probe_bitwise_cubin("probes/f64_queue", Some("cubins/sm100a.cubin"));
}

/// L-OPCLASS: DSETP in the f64 queue (a DADD with no write scoreboard read after a wait on a
/// later DSETP's), PTX setp.<cmp>.f64 (probes/f64_dsetp) = GB300 on both front-ends.
#[test]
fn probe_f64_dsetp_matches_hardware() {
    probe_bitwise("probes/f64_dsetp", "good", "ptx");
    probe_bitwise("probes/f64_dsetp", "good", "sass");
}

/// The f64 / conversion queue reads operands in issue order: a later `F2F.F64.F32` overwrites
/// the source of an earlier one that has a read scoreboard, with no wait on it (variant
/// `rd_sb`; the form ptxas emits in CuTe top-k for sm_103a and sm_100a) = GB300
/// (probes/xu_war).
#[test]
fn probe_xu_war_matches_hardware() {
    for variant in ["good", "rd_sb"] {
        let case = Case::load(root().join("probes/xu_war")).unwrap();
        let k = binary_kernel(&case, variant).unwrap();
        let launch = build_launch(&case, &k).unwrap();
        let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
            .verdict
            .unwrap_or_else(|| panic!("xu_war {variant} certifies"));
        assert_eq!(v.classes(), ["clean"], "{variant}: {:#?}", v.findings);
        let report = case.expect_report(&v.outputs[0]).unwrap();
        assert!(
            !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
            "{variant}: {report:?}"
        );
    }
}

/// A TMEM load overwrites the registers of the TMEM store before it with no wait; the queue
/// reads them first (probes/sttm_ldtm, GB300 1000/1000).
#[test]
fn probe_sttm_ldtm_matches_hardware() {
    probe_bitwise("probes/sttm_ldtm", "good", "ptx");
    probe_bitwise("probes/sttm_ldtm", "good", "sass");
    // tcgen05.ld destinations listed out of order (CuTe DSL: descending): PTX front.
    probe_bitwise("probes/sttm_ldtm", "permuted", "ptx");
    probe_bitwise("probes/sttm_ldtm", "permuted", "sass");
}

/// abs.bf16x2 / abs.bf16 as HFMA2.BF16_V2 -RZ, RZ, |R| (probes/bf16_abs) = GB300.
#[test]
fn probe_bf16_abs_matches_hardware() {
    probe_bitwise("probes/bf16_abs", "good", "ptx");
    probe_bitwise("probes/bf16_abs", "good", "sass");
}

/// f16 arithmetic: HFMA2 / HMUL2 / HADD2 with half selectors and immediate pairs, PTX
/// fma / mul / add / sub .rn .f16[x2] and an uncontractible mul.f16 (probes/f16_arith) = GB300.
#[test]
fn probe_f16_arith_matches_hardware() {
    probe_bitwise("probes/f16_arith", "good", "ptx");
    probe_bitwise("probes/f16_arith", "good", "sass");
}

/// PTX ldmatrix x1 / x2.trans / x4 / x4.trans (probes/ldmatrix_ptx) = GB300.
#[test]
fn probe_ldmatrix_ptx_matches_hardware() {
    probe_bitwise("probes/ldmatrix_ptx", "good", "ptx");
    probe_bitwise("probes/ldmatrix_ptx", "good", "sass");
}

/// HSETP2 / HSETP2.BF16_V2: both predicate outputs, all comparisons, scalar halves, immediate
/// pairs with distinct halves (probes/hsetp2) = GB300.
#[test]
fn probe_hsetp2_matches_hardware() {
    probe_bitwise("probes/hsetp2", "good", "ptx");
    probe_bitwise("probes/hsetp2", "good", "sass");
}

/// VHMNMX / VHMNMX.BF16_V2 [.NAN]: three-input half min / max, f16 and bf16, |x| operands
/// (probes/vhmnmx) = GB300. SASS front only (no PTX f16 min / max).
#[test]
fn probe_vhmnmx_matches_hardware() {
    probe_bitwise("probes/vhmnmx", "good", "sass");
}

/// MATCH.ANY[.U64]: full membermask on both front-ends; divergent lanes with partial masks
/// (WARPSYNC Rmask / WARPSYNC.COLLECTIVE + MATCH.ANY) on the SASS front (probes/match_any,
/// probes/match_any_div) = GB300.
#[test]
fn probe_match_any_matches_hardware() {
    probe_bitwise("probes/match_any", "good", "ptx");
    probe_bitwise("probes/match_any", "good", "sass");
    probe_bitwise("probes/match_any_div", "good", "sass");
}

/// Cache-hint and scope tokens of the DeepEP / DeepCP kernels (LDG.NA, ST.EF, LD/ST/ATOM
/// .STRONG.{SM,GPU,SYS}, STG.E.64.MMIO.SYS, UBLKPF.L2 in ptxas's uniformizing loop with
/// R2UR.OR) (probes/memhints) = GB300.
#[test]
fn probe_memhints_matches_hardware() {
    probe_bitwise("probes/memhints", "good", "ptx");
    probe_bitwise("probes/memhints", "good", "sass");
}

/// FLO and POPC share an in-order queue: an unscoreboarded FLO.U32 read after a wait on a
/// later POPC's scoreboard (probes/flo_popc, GB300 500/500).
#[test]
fn probe_flo_popc_matches_hardware() {
    probe_bitwise("probes/flo_popc", "good", "ptx");
    probe_bitwise("probes/flo_popc", "good", "sass");
}

/// cp.async with a src-size operand (register 16 / 0, registers 0..16, immediate 4; 16-, 8- and
/// 4-byte copies; SASS LDGSTS[.ZFILL]): the first src-size bytes copied, the rest zero
/// (probes/cpasync_srcsize) = GB300.
#[test]
fn probe_cpasync_srcsize_matches_hardware() {
    probe_bitwise("probes/cpasync_srcsize", "good", "ptx");
    probe_bitwise("probes/cpasync_srcsize", "good", "sass");
}

/// mul / add / sub .f32x2 with no rounding modifier where ptxas cannot contract (a mul of a
/// mul converted to bf16x2, an add / sub of loads; FMUL2 / FADD2) = .rn (probes/f32x2_norn) =
/// GB300.
#[test]
fn probe_f32x2_norn_matches_hardware() {
    probe_bitwise("probes/f32x2_norn", "good", "ptx");
    probe_bitwise("probes/f32x2_norn", "good", "sass");
}

/// Plain mul.f32x2 read as an fma addend, unpacked into scalar ops, or repacked across lanes
/// into an add.f32x2 (probes/f32x2_unpack): ptxas keeps FMUL2 apart; = GB300 on both fronts.
#[test]
fn probe_f32x2_unpack_matches_hardware() {
    probe_bitwise("probes/f32x2_unpack", "good", "ptx");
    probe_bitwise("probes/f32x2_unpack", "good", "sass");
}

/// The cp.async scoreboard counts the warp's commits: copies issued on two divergent paths,
/// one converged DEPBAR, each lane reads its neighbour's slot (probes/cpasync_warp; `good` =
/// GB300; `w1`: the warp's second group is still pending).
#[test]
fn probe_cpasync_warp_verdicts() {
    probe_bitwise("probes/cpasync_warp", "good", "ptx");
    probe_bitwise("probes/cpasync_warp", "good", "sass");
    check_dir("probes", "cpasync_warp", "w1", "ptx");
    check_dir("probes", "cpasync_warp", "w1", "sass");
}

/// abs.f32 / abs.ftz.f32 (FADD[.FTZ] |a|, -RZ: canonical NaN) and abs.s32 (IABS) on special
/// patterns (probes/abs_f32) = GB300.
#[test]
fn probe_abs_f32_matches_hardware() {
    probe_bitwise("probes/abs_f32", "good", "ptx");
    probe_bitwise("probes/abs_f32", "good", "sass");
}

/// cvt.rn.f32.f64 / cvt.f64.f32 (F2F), cvt.sat.f32.f32 (FADD.SAT RZ, a), cvt.r{n,m,p}i[.ftz]
/// .{s,u}32.f32 (F2I) on special and random inputs (probes/cvt_misc) = GB300.
#[test]
fn probe_cvt_misc_matches_hardware() {
    probe_bitwise("probes/cvt_misc", "good", "ptx");
    probe_bitwise("probes/cvt_misc", "good", "sass");
}

/// PTX prmt.b32 (default mode) = SASS PRMT d, a, c, b (probes/prmt_ptx) = GB300.
#[test]
fn probe_prmt_ptx_matches_hardware() {
    probe_bitwise("probes/prmt_ptx", "good", "ptx");
    probe_bitwise("probes/prmt_ptx", "good", "sass");
}

/// %lanemask_*, a `%`-less inline-asm scope register, `[%rd+-N]`, and a vector operand list
/// continued after a comma (probes/ptx_syntax) = GB300.
#[test]
fn probe_ptx_syntax_matches_hardware() {
    probe_bitwise("probes/ptx_syntax", "good", "ptx");
    probe_bitwise("probes/ptx_syntax", "good", "sass");
}

/// add / sub / mul .bf16x2 and .bf16 with no rounding modifier where ptxas cannot contract
/// (probes/bf16_norn; SASS HADD2.BF16_V2 / HFMA2.BF16_V2) = GB300.
#[test]
fn probe_bf16_norn_matches_hardware() {
    probe_bitwise("probes/bf16_norn", "good", "ptx");
    probe_bitwise("probes/bf16_norn", "good", "sass");
}

/// rcp.rn.f32 = the IEEE 1 / a, NaN canonical (probes/rcp_rn; ptxas's div.rn expansion with
/// its slow-path subroutine) = GB300.
#[test]
fn probe_rcp_rn_matches_hardware() {
    probe_bitwise("probes/rcp_rn", "good", "ptx");
    probe_bitwise("probes/rcp_rn", "good", "sass");
}

/// cvt.{rn,rz,rm,rp}.f32.f64 = F2F.F32.F64[.RZ|.RM|.RP] (probes/cvt_f64_rnd) = GB300.
#[test]
fn probe_cvt_f64_rnd_matches_hardware() {
    probe_bitwise("probes/cvt_f64_rnd", "good", "ptx");
    probe_bitwise("probes/cvt_f64_rnd", "good", "sass");
}

/// bar.red.{popc,and,or} = BAR.RED + B2R.RESULT, CTA-wide and on counted barriers reached by
/// two warps (probes/bar_red) = GB300.
#[test]
fn probe_bar_red_matches_hardware() {
    probe_bitwise("probes/bar_red", "good", "ptx");
    probe_bitwise("probes/bar_red", "good", "sass");
}

/// cvt.{rn,rz,rm,rp}.{f32,f64}.{s,u}{16,32,64} = I2F[.F64][.S16|.U16|.S64|.U64][.RZ|.RM|.RP]
/// (probes/cvt_i2f) = GB300.
#[test]
fn probe_cvt_i2f_matches_hardware() {
    probe_bitwise("probes/cvt_i2f", "good", "ptx");
    probe_bitwise("probes/cvt_i2f", "good", "sass");
}

/// abs.s64, brev.b32, rcp.{rn,rz,rm,rp}.f64 (probes/ptx_misc2) = GB300.
#[test]
fn probe_ptx_misc2_matches_hardware() {
    probe_bitwise("probes/ptx_misc2", "good", "ptx");
    probe_bitwise("probes/ptx_misc2", "good", "sass");
}

/// tcgen05.mma `[taddr + off]` D and A-in-TMEM operands (probes/tc_mma_off) = GB300.
#[test]
fn probe_tc_mma_off_matches_hardware() {
    probe_bitwise("probes/tc_mma_off", "good", "ptx");
    probe_bitwise("probes/tc_mma_off", "good", "sass");
}

/// FFMA2 with `-URn.F32` / `-URn.F32x2.HI_LO` operands (probes/ffma2_ur) = GB300.
#[test]
fn probe_ffma2_ur_matches_hardware() {
    probe_bitwise("probes/ffma2_ur", "good", "ptx");
    probe_bitwise("probes/ffma2_ur", "good", "sass");
}

/// cp.async.bulk.prefetch.tensor (SASS UTMAPF.L2.2D) as a cache hint (probes/tma_prefetch)
/// = GB300.
#[test]
fn probe_tma_prefetch_matches_hardware() {
    probe_bitwise("probes/tma_prefetch", "good", "ptx");
    probe_bitwise("probes/tma_prefetch", "good", "sass");
}

/// atom.add.f64 / red.add.f64 = ATOMG / REDG .E.ADD.F64.RN (probes/atom_f64) = GB300. The
/// red adds to shared words do not commute for the reductions (f64 order changes bits), so the
/// stateful explorer checks every order: one output, GB300's.
#[test]
fn probe_atom_f64_matches_hardware() {
    for front in ["ptx", "sass"] {
        let case = Case::load(root().join("probes/atom_f64")).unwrap();
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 2_000_000);
        assert!(v.complete, "{front}: exploration incomplete");
        assert_eq!(v.classes(), ["clean"], "{front}: {:#?}", v.findings);
        assert_eq!(v.outputs.len(), 1, "{front}: one output");
        let report = case.expect_report(&v.outputs[0]).unwrap();
        assert!(
            !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
            "{front}: {report:?}"
        );
    }
}

/// Generic byte accesses (LD.E.U8 / .S8, ST.E.U8) and CTA-scope global atomics (ATOMG / REDG
/// .STRONG.SM; probes/generic8) = GB300 on both front-ends (explored: the atomic's returned
/// value depends on the order; every order stores the same output).
#[test]
fn probe_generic8_matches_hardware() {
    for front in ["ptx", "sass"] {
        let case = Case::load(root().join("probes/generic8")).unwrap();
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 2_000_000);
        assert!(v.complete, "{front}: exploration incomplete");
        assert_eq!(v.classes(), ["clean"], "{front}: {:#?}", v.findings);
        assert_eq!(v.outputs.len(), 1, "{front}: one output");
        let report = case.expect_report(&v.outputs[0]).unwrap();
        assert!(
            !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
            "{front}: {report:?}"
        );
    }
}

/// Vector float reductions red.global.add.v4 / .v2.f32 (REDG.E.ADD.F32x4 / F32x2; probes/red_vec)
/// = GB300 on both front-ends (explored; the sums are exact in every order).
#[test]
fn probe_red_vec_matches_hardware() {
    for front in ["ptx", "sass"] {
        let case = Case::load(root().join("probes/red_vec")).unwrap();
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 2_000_000);
        assert!(v.complete, "{front}: exploration incomplete");
        assert_eq!(v.classes(), ["clean"], "{front}: {:#?}", v.findings);
        assert_eq!(v.outputs.len(), 1, "{front}: one output");
        let report = case.expect_report(&v.outputs[0]).unwrap();
        assert!(
            !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
            "{front}: {report:?}"
        );
    }
}

/// Vector float atomics with a result, atom.global.add.v4 / .v2.f32 (ATOMG.E.ADD.F32x4 /
/// F32x2.FTZ.RN; probes/atom_vec) = GB300 on both front-ends, returned old values included.
#[test]
fn probe_atom_vec_matches_hardware() {
    probe_bitwise("probes/atom_vec", "good", "ptx");
    probe_bitwise("probes/atom_vec", "good", "sass");
}

/// A vector atom whose destination overlaps a later source element is refused.
#[test]
fn atom_vec_overlap_refused() {
    let src = std::fs::read_to_string(root().join("probes/atom_vec/kernel.ptx")).unwrap();
    let bad = src.replace(
        "{%f5, %f6, %f7, %f8}, [%rd6], {%f1, %f2, %f3, %f4}",
        "{%f2, %f3, %f4, %f5}, [%rd6], {%f1, %f2, %f3, %f4}",
    );
    assert_ne!(bad, src);
    let err = ptxv_ptx::parse_kernel(&bad, Some("atom_vec")).unwrap_err();
    assert!(format!("{err:#}").contains("overlapping"), "{err:#}");
}

/// popc.b64 (two POPC + IADD3; probes/popc64) = GB300 on both front-ends.
#[test]
fn probe_popc64_matches_hardware() {
    probe_bitwise("probes/popc64", "good", "ptx");
    probe_bitwise("probes/popc64", "good", "sass");
}

/// Masked 32-byte loads (LDG.E.ENL2.256 ..., mask: ld.global.v4.b64 / .v8.b32 with some
/// elements unused) and st.global.v4.b64 (probes/ldg256_mask) = GB300 on both front-ends.
#[test]
fn probe_ldg256_mask_matches_hardware() {
    probe_bitwise("probes/ldg256_mask", "good", "ptx");
    probe_bitwise("probes/ldg256_mask", "good", "sass");
}

/// abs / neg .f16x2 / .f16 (HADD2 |R| / -R, -RZ.H0_H0; probes/f16_absneg, every f16 pattern)
/// = GB300 on both front-ends.
#[test]
fn probe_f16_absneg_matches_hardware() {
    probe_bitwise("probes/f16_absneg", "good", "ptx");
    probe_bitwise("probes/f16_absneg", "good", "sass");
}

/// Narrowing integer cvt to a signed type sign-extends into the destination register
/// (probes/cvt_s8_narrow) = GB300 on both front-ends.
#[test]
fn probe_cvt_s8_narrow_matches_hardware() {
    probe_bitwise("probes/cvt_s8_narrow", "good", "ptx");
    probe_bitwise("probes/cvt_s8_narrow", "good", "sass");
}

/// div.full / div.approx .f32 by a constant = FMUL by RN(1/C), contracted like a plain mul
/// (probes/div_const) = GB300 on both front-ends.
#[test]
fn probe_div_const_matches_hardware() {
    probe_bitwise("probes/div_const", "good", "ptx");
    probe_bitwise("probes/div_const", "good", "sass");
}

/// Signed loads into 64-bit registers sign-extend to 64 bits (probes/ld_sext64) = GB300 on
/// both front-ends.
#[test]
fn probe_ld_sext64_matches_hardware() {
    probe_bitwise("probes/ld_sext64", "good", "ptx");
    probe_bitwise("probes/ld_sext64", "good", "sass");
}

/// membar.{cta,gl,sys} / fence.acquire.{cta,cluster,gpu,sys}: every variant's verdict on both
/// front-ends (probes/fence_scopes; good = GB300).
#[test]
fn probe_fence_scopes_verdicts() {
    for v in [
        "good",
        "sys",
        "acq_sys",
        "acq_cta",
        "acq_cluster",
        "rel_cta",
        "acq_cta_local",
    ] {
        check_dir("probes", "fence_scopes", v, "ptx");
        check_dir("probes", "fence_scopes", v, "sass");
    }
}

/// E-CLOCK on both front-ends: clocks 0, vprintf a no-op noted per site, cvta.local
/// (probes/printf_clock) = GB300.
#[test]
fn probe_printf_clock_matches_hardware() {
    for front in ["ptx", "sass"] {
        let case = Case::load(root().join("probes/printf_clock")).unwrap();
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 2_000_000);
        assert!(v.complete, "{front}");
        assert_eq!(v.classes(), ["clean"], "{front}: {:#?}", v.findings);
        assert!(
            v.findings
                .iter()
                .any(|f| f.kind == "note" && f.msg.contains("vprintf reachable")),
            "{front}"
        );
        let report = case.expect_report(&v.outputs[0]).unwrap();
        assert!(
            report.iter().all(|l| l.contains("bitwise equal")),
            "{front}: {report:?}"
        );
    }
}

/// PTX-FMAD for a plain mul.f32 with several readers (probes/fmad_multi): all or nothing per
/// mul, adds of two products = GB300 on both fronts.
#[test]
fn probe_fmad_multi_matches_hardware() {
    probe_bitwise("probes/fmad_multi", "good", "ptx");
    probe_bitwise("probes/fmad_multi", "good", "sass");
}

/// min / max [.NaN] [.ftz] .abs.f32 with three sources (FMNMX3 |a|, |b|, |c|) on every triple
/// of 16 edge patterns = GB300 on both fronts (probes/fmnmx3_abs).
#[test]
fn probe_fmnmx3_abs_matches_hardware() {
    probe_bitwise("probes/fmnmx3_abs", "good", "ptx");
    probe_bitwise("probes/fmnmx3_abs", "good", "sass");
}

/// mbarrier.expect_tx / complete_tx (probes/mbar_expect_tx): tx-count steps = GB300 on both
/// fronts; without a fence the relaxed complete_tx publishes nothing (a race on the PTX front).
#[test]
fn probe_mbar_expect_tx_matches_hardware() {
    probe_bitwise("probes/mbar_expect_tx", "good", "ptx");
    probe_bitwise("probes/mbar_expect_tx", "good", "sass");
    check_dir("probes", "mbar_expect_tx", "relaxed", "ptx");
    check_dir("probes", "mbar_expect_tx", "relaxed", "sass");
}

/// PTX-FMAD for plain mul.f16 (probes/fmad_f16): HFMA2 into every add / sub reader, HMUL2 +
/// HADD2 when the product is also converted or packed = GB300 on both fronts.
#[test]
fn probe_fmad_f16_matches_hardware() {
    probe_bitwise("probes/fmad_f16", "good", "ptx");
    probe_bitwise("probes/fmad_f16", "good", "sass");
}

/// PTX-FMAD: plain mul.f32x2 products no add / sub.f32x2 reads directly (selp, fma addend,
/// mov copy, loop-carried and unpacked; probes/f32x2_sinks) = GB300 on both fronts.
#[test]
fn probe_f32x2_sinks_matches_hardware() {
    probe_bitwise("probes/f32x2_sinks", "good", "ptx");
    probe_bitwise("probes/f32x2_sinks", "good", "sass");
}

/// S-DSMEM: mapa with a rank outside the cluster gives an undefined address, not a fault
/// (probes/mapa_lanes): good = GB300 on both fronts; a store through it is reported (GB300
/// fails the launch).
#[test]
fn probe_mapa_lanes_matches_hardware() {
    probe_bitwise("probes/mapa_lanes", "good", "ptx");
    probe_bitwise("probes/mapa_lanes", "good", "sass");
    check_dir("probes", "mapa_lanes", "oob", "ptx");
    check_dir("probes", "mapa_lanes", "oob", "sass");
}

/// cvta.to.local of a stack buffer's generic address (probes/local_cvta) = GB300 on both
/// fronts; a generic address outside the local window converted and stored through faults.
#[test]
fn probe_local_cvta_matches_hardware() {
    probe_bitwise("probes/local_cvta", "good", "ptx");
    probe_bitwise("probes/local_cvta", "good", "sass");
    let case = Case::load(root().join("probes/local_cvta")).unwrap();
    let src = std::fs::read_to_string(case.dir.join("kernel.ptx")).unwrap();
    let bad = src.replace(
        "cvta.to.local.u64 %rd5, %rd3;",
        "cvta.to.local.u64 %rd5, %rd2;",
    );
    assert_ne!(bad, src);
    let k = ptxv_ptx::parse_kernel(&bad, Some("local_cvta")).unwrap();
    let launch = build_launch(&case, &k).unwrap();
    let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 200_000);
    assert_eq!(v.classes(), ["fault"], "{:#?}", v.findings);
    assert!(
        v.findings
            .iter()
            .any(|f| f.msg.contains("outside the local window")),
        "{:#?}",
        v.findings
    );
}

/// S-GLOBAL-INIT on both front-ends: module .global variables with and without initializers
/// (probes/ptx_globals) = GB300; the zero-initialized ones are noted by name.
#[test]
fn probe_ptx_globals_matches_hardware() {
    for front in ["ptx", "sass"] {
        let case = Case::load(root().join("probes/ptx_globals")).unwrap();
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 200_000);
        assert!(v.complete, "{front}");
        assert_eq!(v.classes(), ["clean"], "{front}: {:#?}", v.findings);
        assert!(
            v.findings
                .iter()
                .any(|f| f.kind == "note" && f.msg.ends_with(": scratch, zero")),
            "{front}: {:#?}",
            v.findings
        );
        let report = case.expect_report(&v.outputs[0]).unwrap();
        assert!(
            report.iter().all(|l| l.contains("bitwise equal")),
            "{front}: {report:?}"
        );
    }
}

/// elect.sync with a register membermask (probes/elect_regmask) = GB300; a partial register
/// mask is refused on PTX when executed.
#[test]
fn probe_elect_regmask_matches_hardware() {
    probe_bitwise("probes/elect_regmask", "good", "ptx");
    probe_bitwise("probes/elect_regmask", "good", "sass");
    check_dir("probes", "elect_regmask", "partial", "ptx");
    check_dir("probes", "elect_regmask", "partial", "sass");
}

/// cp.async.bulk with .L2::cache_hint both ways (probes/bulk_hint) = GB300.
#[test]
fn probe_bulk_hint_matches_hardware() {
    probe_bitwise("probes/bulk_hint", "good", "ptx");
    probe_bitwise("probes/bulk_hint", "good", "sass");
}

/// call.uni __assertfail (probes/assertfail): not taken = GB300; taken, a trap on both fronts.
#[test]
fn probe_assertfail_matches_hardware() {
    probe_bitwise("probes/assertfail", "good", "ptx");
    probe_bitwise("probes/assertfail", "good", "sass");
    check_dir("probes", "assertfail", "taken", "ptx");
    check_dir("probes", "assertfail", "taken", "sass");
}

/// plain mul.bf16 through selp into stmatrix / tcgen05.st (probes/bf16_selp) = GB300.
#[test]
fn probe_bf16_selp_matches_hardware() {
    probe_bitwise("probes/bf16_selp", "good", "ptx");
    probe_bitwise("probes/bf16_selp", "good", "sass");
}

/// mul.f32x2 halves repacked next to other values, integer reads of the product
/// (probes/f32x2_repack) = GB300; the identity repack {p.lo, p.hi} (FFMA2) is refused on PTX.
#[test]
fn probe_f32x2_repack_matches_hardware() {
    probe_bitwise("probes/f32x2_repack", "good", "ptx");
    probe_bitwise("probes/f32x2_repack", "good", "sass");
    let src = std::fs::read_to_string(root().join("probes/f32x2_repack/kernel.ptx")).unwrap();
    let ident = src.replace("{%r14, %r2}; // @K", "{%r14, %r15}; // @K");
    assert_ne!(ident, src);
    let e = ptxv_ptx::parse_kernel(&ident, Some("f32x2_repack")).unwrap_err();
    assert!(format!("{e:#}").contains("mul.f32x2"), "{e:#}");
}

/// mad.wide.{s,u}16 with an immediate multiplier (probes/mad_wide16) = GB300.
#[test]
fn probe_mad_wide16_matches_hardware() {
    probe_bitwise("probes/mad_wide16", "good", "ptx");
    probe_bitwise("probes/mad_wide16", "good", "sass");
}

/// set.<cmp>.{u32,bf16x2,f16x2}.{bf16x2,f16x2} = HSET2[.BF16_V2][.BF] (probes/hset2) = GB300.
#[test]
fn probe_hset2_matches_hardware() {
    probe_bitwise("probes/hset2", "good", "ptx");
    probe_bitwise("probes/hset2", "good", "sass");
}

/// movmatrix.sync.aligned.m8n8.trans.b16 = MOVM.16.MT88 (probes/movm) = GB300.
#[test]
fn probe_movm_matches_hardware() {
    probe_bitwise("probes/movm", "good", "ptx");
    probe_bitwise("probes/movm", "good", "sass");
}

/// Multicast try_cancel (UGETNEXTWORKID.BROADCAST) in a cluster-2 launch: both CTAs get the
/// response, whole clusters are cancelled (probes/clc_mcast); every schedule is clean.
#[test]
fn probe_clc_mcast_follows_hardware() {
    let case = Case::load(root().join("probes/clc_mcast")).unwrap();
    for front in ["ptx", "sass"] {
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 2_000_000);
        assert!(v.complete, "{front}: incomplete");
        assert_eq!(
            v.classes(),
            ["nondeterministic"],
            "{front}: {:#?}",
            v.findings
        );
        let (mut cancels, mut fails) = (0, 0);
        for o in &v.outputs {
            let w: Vec<u32> = o
                .chunks(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let cta = |c: usize| &w[8 * c..8 * c + 8];
            for cl in 0..2 {
                let (a, b) = (cta(2 * cl), cta(2 * cl + 1));
                // Whole clusters run or are cancelled; both CTAs get the same response.
                assert_eq!(a[0], b[0], "{front}: cluster {cl} half ran: {w:?}");
                if a[0] == 0 {
                    continue;
                }
                assert_eq!(a[1..7], b[1..7], "{front}: responses differ: {w:?}");
                assert_eq!((a[7], b[7]), (0, 1), "{front}: ranks: {w:?}");
                if a[1] == 1 {
                    cancels += 1;
                    let x = a[2] as usize;
                    assert!(
                        x.is_multiple_of(2) && x / 2 != cl,
                        "{front}: first_ctaid {x}: {w:?}"
                    );
                    assert_eq!(a[4..7], [0, 1, 0], "{front}: raw: {w:?}");
                    assert_eq!((cta(x)[0], cta(x + 1)[0]), (0, 0), "{front}: ran: {w:?}");
                } else {
                    fails += 1;
                    assert_eq!(a[2..7], [0, 0, 0, 0, 0], "{front}: failure: {w:?}");
                }
            }
        }
        assert!(
            cancels > 0 && fails > 0,
            "{front}: {cancels} cancels, {fails} failures"
        );
    }
}

/// add.u16x2 / add.s16x2 as VIADD.16x2, register and immediate b: two 16-bit lanes, no carry
/// between them (probes/viadd16x2) = GB300.
#[test]
fn probe_viadd16x2_matches_hardware() {
    probe_bitwise("probes/viadd16x2", "good", "ptx");
    probe_bitwise("probes/viadd16x2", "good", "sass");
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

/// Module data globals (fixtures/s_globals, S-GLOBAL-INIT): three launches of one module share
/// `counter` (a case buffer, `symbol = "counter"`) and `table` (automatic, from
/// .nv.global.init); outputs = GB300's. Unordered launches race on the globals.
#[test]
fn s_globals_variants() {
    let case = Case::load(root().join("fixtures/s_globals")).unwrap();
    for v in ["good", "unordered"] {
        let vcase = case.for_variant(v).unwrap();
        let m = r4a_machine(&vcase, v);
        let names: Vec<&str> = m.launch.allocs.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["out", "counter", "global:table"]);
        let r = explore(m, 200_000);
        let expect = case.variant(v).unwrap().expect_for("sass");
        assert!(r.complete && r.meets(&expect), "{v}: {:?}", r.classes());
        assert!(r.assumptions.iter().any(|a| a == "S-GLOBAL-INIT"));
        if v == "good" {
            let report = case.expect_report(&r.outputs[0]).unwrap();
            assert!(
                report.len() == 1 && report[0].contains("bitwise equal"),
                "{report:?}"
            );
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
    // Notes (S-SAME-VALUE) are not findings.
    let kinds: std::collections::BTreeSet<_> = fs
        .iter()
        .filter(|f| f.kind != "note")
        .map(|f| f.kind.as_str())
        .collect();
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
            .filter(|f| f.kind != "note")
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
        .filter(|f| f.kind != "note")
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
            .filter(|f| f.kind != "note")
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
            // DPOR does not run the E-RACE-DEAD probe: a race the stateful explorer marked
            // dead (neg-payld's masked stress load) stays a race there, with the same sig.
            let dead: Vec<&str> = sv
                .findings
                .iter()
                .filter(|x| x.kind == "race-dead")
                .map(|x| x.sig.as_str())
                .collect();
            if !dead.is_empty() {
                assert!(
                    dead.iter().all(|s| dv.findings.iter().any(|x| x.sig == *s)),
                    "{v}/{f} (DPOR): {:#?}",
                    dv.findings
                );
                dc.retain(|c| c != "race");
                if dc.is_empty() {
                    dc.push("clean".into());
                }
            }
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
    for f in v.findings.iter().filter(|f| f.kind != "note") {
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

/// E-HW-WEAK-FLAG: a weak flag store after fence.sc.sys (msl DeepEP nvlink barrier) races in
/// the strict model and is latent under the hardware model; without the writer's fence the
/// race stays observable.
#[test]
fn twolevel_weak_flag() {
    if std::env::var("PTXV_HW_L1").is_ok() || std::env::var("PTXV_HW_MODEL").is_ok() {
        return;
    }
    let case = Case::load(root().join("fixtures/e_weak_flag")).unwrap();
    for f in ["ptx", "sass"] {
        for v in ["good", "nofence"] {
            let k = kernel(&case, v, f);
            let launch = build_launch(&case, &k).unwrap();
            let m1 = Machine::new(Arc::new(k), Arc::new(launch));
            let mut v1 = explore(m1.clone(), 100_000);
            assert!(v1.findings.iter().any(|x| x.kind == "race"), "{v} {f}");
            let why = ptxv_sim::twolevel::wanted(&v1, &m1.launch).unwrap();
            ptxv_sim::twolevel::run(&mut v1, &m1, why, &mut |m| explore(m, 100_000));
            let obs = v1
                .findings
                .iter()
                .any(|x| x.gb300.as_ref().is_some_and(|g| g.class == "observable"));
            assert_eq!(obs, v == "nofence", "{v} {f}: {:#?}", v1.findings);
        }
    }
}

/// E-HW-CORW: a plain store that clears a word its thread polled strongly (DeepEP hybrid
/// dispatch clearing NIC-written slots, docs/nic.md finding L) races in the strict model; on
/// GB300 the clear is coherence-after the store it read (latent). The store/poll race stays
/// observable; a weak poll or no poll leaves the write-write race observable.
#[test]
fn twolevel_hw_corw() {
    if std::env::var("PTXV_HW_L1").is_ok() || std::env::var("PTXV_HW_MODEL").is_ok() {
        return;
    }
    let case = Case::load(root().join("fixtures/e_hw_corw")).unwrap();
    for f in ["ptx", "sass"] {
        for v in ["good", "weakpoll", "noread"] {
            let k = kernel(&case, v, f);
            let launch = build_launch(&case, &k).unwrap();
            let m1 = Machine::new(Arc::new(k), Arc::new(launch));
            let mut v1 = explore(m1.clone(), 100_000);
            let ww =
                |x: &ptxv_sim::machine::Finding| x.kind == "race" && x.msg.contains("write-write");
            assert!(v1.findings.iter().any(ww), "{v} {f}: {:#?}", v1.findings);
            let why = ptxv_sim::twolevel::wanted(&v1, &m1.launch).unwrap();
            ptxv_sim::twolevel::run(&mut v1, &m1, why, &mut |m| explore(m, 100_000));
            for x in v1.findings.iter().filter(|x| ww(x)) {
                let c = &x.gb300.as_ref().unwrap().class;
                assert_eq!(c == "observable", v != "good", "{v} {f}: {:#?}", x);
            }
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
/// (`tmem[UR+-0x10]`): probes/tmem_ldred agrees with GB300 bitwise (2026-11-05, 100/100), on
/// the PTX front-end too (tcgen05.ld.red.32x32b).
#[test]
fn probe_tmem_ldred_matches_hardware() {
    certified_hash("probes", "tmem_ldred", 16475847427938539045);
    let case = Case::load(root().join("probes/tmem_ldred")).unwrap();
    let v = certify_ptx_file(&case);
    assert_eq!(
        fnv(&v.outputs[0]),
        16475847427938539045,
        "ptx: output differs from GB300's"
    );
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

/// S-INIT through FFMA2 (fixtures/e_uninit_f32x2): each lane of a packed f32x2 FMA reads only
/// the same lane of its operands, so an undefined lane 1 leaves lane 0 defined (SASS clean;
/// storing the undefined lane is reported).
#[test]
fn ffma2_lanes_keep_their_own_undefined_bits() {
    both("e_uninit_f32x2", "good");
    both("e_uninit_f32x2", "hi");
}

/// S-INIT per lane of a PTX f32x2 virtual register (fixtures/e_uninit_f32x2_lanes): a
/// `mov.b64 {a, b}` pack of a defined and an undefined lane keeps the defined lane defined
/// through add.f32x2 and the unpack; storing the undefined lane is reported.
#[test]
fn ptx_f32x2_register_lanes_keep_their_own_undefined_bits() {
    both("e_uninit_f32x2_lanes", "good");
    both("e_uninit_f32x2_lanes", "store_hi");
}

/// S-TCGEN05 (fixtures/s_tc_st_mma): a thread's own tcgen05.st, wait::st, then its
/// tcgen05.mma with no tcgen05.fence in that thread is ordered; the MMA issued before the
/// wait::st races with the store.
#[test]
fn own_tcgen05_st_wait_orders_later_mma() {
    both("s_tc_st_mma", "good");
    // The race leaves outcomes DPOR also enumerates: the stateful verdict only.
    check_stateful("s_tc_st_mma", "early", "ptx", 100_000);
    check_stateful("s_tc_st_mma", "early", "sass", 100_000);
}

/// Input coverage and input mutation (fixtures/e_sentinel_fill, the shape of a
/// bucket-fill kernel): on the all-valid input the check is clean with one untaken
/// input-dependent side (`ok != 0` never false); setting ok = 0 in two lanes of different warps
/// gives the race on the sentinel word; the fixed shape (only valid lanes store) stays clean;
/// a comparison of a value computed from the input word is refused.
#[test]
fn input_coverage_and_mutation_find_the_sentinel_race() {
    let case = Case::load(root().join("fixtures/e_sentinel_fill")).unwrap();
    for front in ["ptx", "sass"] {
        for (v, want) in [("good", "race"), ("fixed", "clean")] {
            both("e_sentinel_fill", v);
            let k = kernel(&case, v, front);
            let launch = build_launch(&case, &k).unwrap();
            let m = Machine::new(Arc::new(k), Arc::new(launch));
            let c = ptxv_sim::coverage::analyze(&m);
            assert_eq!(c.untaken.len(), 1, "{v}/{front}: {:#?}", c.untaken);
            assert!(
                !c.untaken[0].never && c.untaken[0].sources.starts_with("ok["),
                "{:#?}",
                c.untaken
            );
            // Control and addresses depend on the input (the flag guards the slot store,
            // the bucket id is its address).
            assert!(!c.input_independent && !c.input_dependent.is_empty());
            assert_eq!(c.comparisons.len(), 1);
            assert_eq!((c.comparisons[0].n_true, c.comparisons[0].n_false), (64, 0));
            // Declared invariants (ok in {0, 1}, bucket in [0, 63]) hold on the capture.
            let inv =
                |buffer: &str, values: Option<Vec<i64>>, min, max| ptxv_sim::coverage::Invariant {
                    buffer: buffer.into(),
                    values,
                    min,
                    max,
                    elem: 4,
                    signed: false,
                };
            let (dom, notes) = ptxv_sim::coverage::Domain::new(
                &m,
                vec![
                    inv("ok", Some(vec![0, 1]), None, None),
                    inv("bucket", None, Some(0), Some(63)),
                ],
            );
            assert!(
                notes
                    .iter()
                    .all(|n| n.contains("holds on the captured input")),
                "{notes:#?}"
            );
            let (muts, refused) = ptxv_sim::coverage::plan(&m, &c, 4, 2, &dom);
            assert!(refused.is_empty(), "{refused:#?}");
            assert_eq!(muts.len(), 1);
            assert_eq!(muts[0].label, "ok[0]=0, ok[32]=0");
            assert!(muts[0].within_invariants);
            // Unconstrained (no invariants): the same mutation, not within declared invariants.
            let (m2, _) = ptxv_sim::coverage::plan(&m, &c, 4, 2, &Default::default());
            assert!(m2[0].label == muts[0].label && !m2[0].within_invariants);
            // ok in {1} admits no flipping value; ok in {2} does not hold on the capture.
            let (d1, _) =
                ptxv_sim::coverage::Domain::new(&m, vec![inv("ok", Some(vec![1]), None, None)]);
            let (m1, r1) = ptxv_sim::coverage::plan(&m, &c, 4, 2, &d1);
            assert!(
                m1.is_empty() && r1[0].reason.contains("declared invariant"),
                "{r1:#?}"
            );
            let (_, n2) =
                ptxv_sim::coverage::Domain::new(&m, vec![inv("ok", Some(vec![2]), None, None)]);
            assert!(
                n2[0].contains("does not hold") && n2[0].contains("64 of 64"),
                "{n2:#?}"
            );
            let w = ptxv_sim::wellsync::check_certified(
                ptxv_sim::coverage::apply(&m, &muts[0]),
                100_000,
            );
            assert_eq!(
                w.classes(),
                vec![want.to_string()],
                "{v}/{front}: {:#?}",
                w.findings
            );
        }
        let k = kernel(&case, "derived", front);
        let launch = build_launch(&case, &k).unwrap();
        let m = Machine::new(Arc::new(k), Arc::new(launch));
        let c = ptxv_sim::coverage::analyze(&m);
        let (muts, refused) = ptxv_sim::coverage::plan(&m, &c, 4, 2, &Default::default());
        assert!(
            muts.is_empty() && refused.len() == 1,
            "{front}: {muts:#?} {refused:#?}"
        );
        assert!(
            refused[0].reason.contains("not a single input word"),
            "{refused:#?}"
        );
    }
}

/// Input coverage beyond comparisons: a kernel whose control, addresses and synchronization
/// do not touch input data is input-independent (fixtures/s_init); an input-derived FCHK
/// (division slow-path test, captured silu, fixtures/c_silu) is a site B does not flip, and
/// the slow path behind it is unexecuted code behind an input-dependent condition.
#[test]
fn input_independence_and_predicate_sites() {
    let run = |name: &str, front: &str| {
        let case = Case::load(root().join("fixtures").join(name)).unwrap();
        let k = match &case.m.cubin {
            Some(c) => ptxv_sass::load_cubin(&case.dir.join(c), &case.m.entry).unwrap(),
            None => kernel(&case, "good", front),
        };
        let launch = build_launch(&case, &k).unwrap();
        let m = Machine::new(Arc::new(k), Arc::new(launch));
        ptxv_sim::coverage::analyze(&m)
    };
    for front in ["ptx", "sass"] {
        let c = run("s_init", front);
        assert!(
            c.input_independent && c.input_dependent.is_empty() && c.untaken.is_empty(),
            "{front}: {c:#?}"
        );
    }
    let c = run("c_silu", "sass");
    assert!(!c.input_independent);
    assert_eq!(c.untaken.len(), 8, "{:#?}", c.untaken);
    assert!(
        c.untaken.iter().all(|u| u.at.contains("FCHK") && u.never),
        "{:#?}",
        c.untaken
    );
    assert!(c.untaken.iter().all(|u| {
        u.refuse
            .as_deref()
            .is_some_and(|r| r.contains("not a comparison"))
    }));
    assert!(c.behind_input > 0 && c.unexecuted.iter().any(|g| g.gate == "input"));
}

/// S-INIT lane precision for narrow packs (fixtures/e_uninit_lanes): F2FP pack / unpack and
/// HADD2.F32 of one half keep undefined bits per lane, so a defined low lane stays defined.
#[test]
fn narrow_pack_lanes_keep_their_own_undefined_bits() {
    both("e_uninit_lanes", "good");
    both("e_uninit_lanes", "hi");
}

/// S-INIT through a warp reduction (fixtures/e_uninit_redux): redux.sync carries undefined
/// operand bits to its result; warp 1's undefined sum, dropped: clean (was reported at the
/// REDUX, Triton cross-warp reduction dash/5819); stored: reported.
#[test]
fn undefined_values_through_redux() {
    both("e_uninit_redux", "good");
    both("e_uninit_redux", "store_all");
}

/// S-INIT for tcgen05.ld of never-written TMEM (fixtures/e_uninit_tmem): the registers are
/// undefined and the load is a note (`tmem-uninit-ld`); unused: clean (was a violation,
/// Triton peeled prologue dash/5713); stored: reported.
#[test]
fn never_written_tmem_load_is_a_note() {
    both("e_uninit_tmem", "good");
    both("e_uninit_tmem", "use");
    let case = Case::load(root().join("fixtures/e_uninit_tmem")).unwrap();
    for front in ["ptx", "sass"] {
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let r = explore(Machine::new(Arc::new(k), Arc::new(launch)), 10_000);
        assert!(
            r.findings
                .iter()
                .any(|f| f.kind == "note" && f.sig.starts_with("tmem-uninit-ld:")),
            "{front}: {:#?}",
            r.findings
        );
    }
}

/// S-INIT-FILLS (fixtures/e_uninit_fills): undefined registers reach the output only as
/// data; min(x, -inf) gives the same outputs in every fill: `undef-insensitive` (empirical);
/// x + 1 does not: still a violation.
#[test]
fn multi_fill_rerun_separates_value_insensitive_outputs() {
    both("e_uninit_fills", "good");
    both("e_uninit_fills", "add");
}

/// The second level of the two-level verdict: the SASS front under the GB300 hardware
/// profile (hwprofile.rs), certificate first, profile-discharged contract findings removed.
fn sass_gb300(case: &str, variant: &str) -> String {
    let case = Case::load(root().join("fixtures").join(case)).unwrap();
    let k = kernel(&case, variant, "sass");
    let launch = build_launch(&case, &k).unwrap();
    let m = Machine::new(Arc::new(k), Arc::new(launch));
    let p = ptxv_sim::hwprofile::GB300;
    let m2 = ptxv_sim::hwprofile::machine(&m, p);
    let mut v = ptxv_sim::wellsync::check_certified(m2.clone(), 100_000);
    ptxv_sim::hwprofile::discharge(&mut v, &m2, p);
    ptxv_sim::hwprofile::result(&v)
}

/// Two-level verdict (fixtures/e_twolevel_bulk): no wait_group.read before exit is a contract
/// deviation at level 1 and certifies under the GB300 profile (E-BULK-EXIT); without the
/// proxy fence both levels report the race; with the wait both are clean.
#[test]
fn contract_only_kernel_certifies_under_the_gb300_profile() {
    for v in ["good", "waited", "nofence"] {
        both("e_twolevel_bulk", v);
    }
    assert_eq!(sass_gb300("e_twolevel_bulk", "good"), "certified");
    assert_eq!(sass_gb300("e_twolevel_bulk", "waited"), "certified");
    assert_eq!(sass_gb300("e_twolevel_bulk", "nofence"), "race");
}

/// A bounded search never reports `clean` (fixtures/e_deep_race): a race deep in a wide state
/// space, under a 20k-state cap, at 1, 8, 32 and 64 worker threads, is the race (found by the
/// search or by the deterministic run to the end that follows a search with no terminal
/// state); the race-free variant is `inconclusive`. Before, 8+ workers spent the cap
/// breadth-first and reported `clean` with no terminal state.
#[test]
fn bounded_search_never_reports_clean() {
    let case = Case::load(root().join("fixtures/e_deep_race")).unwrap();
    for front in ["ptx", "sass"] {
        for workers in [1, 8, 32, 64] {
            for (v, want) in [("good", "race"), ("one", "inconclusive")] {
                let k = kernel(&case, v, front);
                let launch = build_launch(&case, &k).unwrap();
                let r = ptxv_sim::explore::explore_threads(
                    Machine::new(Arc::new(k), Arc::new(launch)),
                    20_000,
                    workers,
                );
                let c = r.classes();
                assert!(
                    !r.complete && !c.iter().any(|x| x == "clean"),
                    "{v}/{front}/{workers}: {c:?}"
                );
                assert!(c.iter().any(|x| x == want), "{v}/{front}/{workers}: {c:?}");
            }
        }
    }
}

/// S-INIT through shared memory (fixtures/e_uninit_smem): an STS of undefined data marks the
/// bytes undefined, an LDS of them gives undefined registers; discarded: clean; stored to
/// global memory: reported.
#[test]
fn undefined_values_through_shared_memory() {
    both("e_uninit_smem", "good");
    both("e_uninit_smem", "store_all");
}

/// S-INIT per byte for vector stores (fixtures/e_uninit_vec): a v4 store with one undefined
/// register marks only its 4 bytes; reading the defined lanes, or overwriting the undefined
/// global lane before exit, is clean; reading the undefined lane, or leaving it, is reported.
#[test]
fn vector_store_with_one_undefined_lane() {
    for v in ["good", "read_undef", "keep_undef"] {
        both("e_uninit_vec", v);
    }
}

/// Root pairs (S-SMEM-STRONG, fixtures/e_smem_root): a cross-warp STS / LDS pair with no
/// barrier is no race in SASS, but the output depends on it; the certified check lists the
/// pair first as an `unordered` finding (not a verdict class) next to the nondeterminism.
#[test]
fn unordered_smem_pair_is_listed_as_root_cause() {
    check("e_smem_root", "good", "sass");
    check_stateful("e_smem_root", "good", "ptx", 10_000);
    let case = Case::load(root().join("fixtures/e_smem_root")).unwrap();
    let k = kernel(&case, "good", "sass");
    let launch = build_launch(&case, &k).unwrap();
    let v =
        ptxv_sim::wellsync::check_certified(Machine::new(Arc::new(k), Arc::new(launch)), 10_000);
    assert_eq!(v.classes(), ["nondeterministic"], "{:#?}", v.findings);
    let f = v.findings.first().expect("a root-cause finding");
    assert_eq!(f.kind, "unordered", "{:#?}", v.findings);
    assert!(
        f.sig.starts_with("unordered-smem:") && f.pcs.len() == 2,
        "{f:#?}"
    );
}

/// S-INIT poison propagation (fixtures/e_uninit_discard): undefined registers of lanes that
/// skipped a predicated load flow through a 64-bit add (SASS: a carry-out predicate) into a
/// store under the same predicate: clean. An unpredicated store or a branch on the sum is a
/// sink: reported (the store: global bytes still undefined when the launch ends). The SASS good
/// kernel used to be reported at the IADD3 with the carry-out.
#[test]
fn undefined_values_discarded_by_predicated_store_are_clean() {
    for v in ["good", "store_all", "branch"] {
        both("e_uninit_discard", v);
    }
    let case = Case::load(root().join("fixtures/e_uninit_discard")).unwrap();
    // The sink: the store (`ST`), or the branch (SASS: ptxas folds it into `@P0 EXIT`).
    for (v, at) in [("store_all", &["ST"][..]), ("branch", &["BRA", "EXIT"][..])] {
        for front in ["ptx", "sass"] {
            let k = kernel(&case, v, front);
            let launch = build_launch(&case, &k).unwrap();
            let r = explore(Machine::new(Arc::new(k), Arc::new(launch)), 10_000);
            assert!(
                r.findings.iter().any(|f| (f.sig.starts_with("uninit-reg:")
                    || f.sig.starts_with("uninit-gmem-exit:"))
                    && at.iter().any(|a| f
                        .msg
                        .rsplit(" at ")
                        .next()
                        .unwrap_or("")
                        .to_ascii_uppercase()
                        .contains(a))),
                "{v}/{front}: {:#?}",
                r.findings
            );
        }
    }
}

/// A warp-local stmatrix -> ld.shared exchange (fixtures/e_stsm_warp): ptxas drops the
/// `bar.warp.sync -1`, so on SASS the LDS is ordered after the other lanes' STSM by lockstep
/// alone (E-LOCKSTEP); it used to be a write-read race on the SASS front-end. Without the
/// `bar.warp.sync` the PTX race between lanes of the converged warp is reported and labelled
/// latent-needs-divergence; a store inside a lane-dependent branch is not labelled.
#[test]
fn e_stsm_warp_lockstep() {
    both("e_stsm_warp", "good");
    both("e_stsm_warp", "nosync");
    // Overlapping stmatrix rows (E-STSM-DUP): same value once, different values a violation.
    both("e_stsm_warp", "dup");
    both("e_stsm_warp", "dupdiff");
    // divst/ptx: a race left in; the stateful explorer reports one order and DPOR also the
    // reversed one (nondeterministic), as in e_lds_atoms (E-DRF): checked below, not by `both`.
    check("e_stsm_warp", "divst", "sass");
    let case = Case::load(root().join("fixtures/e_stsm_warp")).unwrap();
    let label = |variant: &str| -> Vec<Option<String>> {
        let k = kernel(&case, variant, "ptx");
        let launch = build_launch(&case, &k).unwrap();
        let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 10_000);
        v.findings
            .iter()
            .filter(|f| f.kind == "race")
            .map(|f| f.gb300.as_ref().map(|g| g.class.clone()))
            .collect()
    };
    let l = label("nosync");
    assert!(
        !l.is_empty()
            && l.iter()
                .all(|c| c.as_deref() == Some(ptxv_sim::twolevel::LATENT_DIV)),
        "{l:?}"
    );
    let l = label("divst");
    assert!(!l.is_empty() && l.iter().all(|c| c.is_none()), "{l:?}");
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

/// E-REMOTE-FIFO (opt-in): remote arrivals from one CTA to one CTA land in happens-before
/// issue order (FA4 2-CTA teardown shape): clean with the option, a violation without it,
/// and still a violation when no barrier orders the two issues or the first comes after.
#[test]
fn e_remote_fifo_orders_hb_ordered_arrivals() {
    for v in ["good", "nobar", "late"] {
        both("e_remote_fifo", v);
    }
    both("e_remote_fifo_strict", "good");
}

/// C-POLL-ATOM (E-WELLSYNC): an arrival counter of atomic adds polled to its full count is
/// certified; a poll that exits on a partial count is refused and races.
#[test]
fn e_poll_atom_counter_poll() {
    for v in ["good", "reset", "ge1"] {
        both("e_poll_atom", v);
    }
}

/// S-SAME-VALUE with the reader's own store (msl DeepEP DeviceSyncer currentCountIdx): two
/// CTAs store the same value and read it back: clean. A different value, a reader without its
/// own store, or a different-valued store that happens before the read but is unordered with
/// the reader's store: race.
#[test]
fn s_sv_own_store() {
    both("s_sv_own", "good");
    // Racing variants: the stateful explorer's verdict only (DPOR also enumerates the
    // nondeterministic outputs the race leaves).
    stateful_classes("s_sv_own", &["diff", "noown", "xhb"]);
}

/// E-HW-CORW fixture classes at level 1 (the level-2 classification: twolevel_hw_corw).
#[test]
fn e_hw_corw_classes() {
    stateful_classes("e_hw_corw", &["good", "weakpoll", "noread"]);
}

/// C-TICKET fixture classes (the certificate side: tests/wellsync.rs).
#[test]
fn e_ticket_classes() {
    both("e_ticket", "good");
    stateful_classes("e_ticket", &["ctaid", "race"]);
}

/// The stateful explorer's classes of `variants` on both front-ends equal the expected ones.
fn stateful_classes(case: &str, variants: &[&str]) {
    let case = Case::load(root().join("fixtures").join(case)).unwrap();
    for &v in variants {
        for f in ["ptx", "sass"] {
            let k = kernel(&case, v, f);
            let launch = build_launch(&case, &k).unwrap();
            let r = explore(Machine::new(Arc::new(k), Arc::new(launch)), 100_000);
            assert_eq!(
                r.classes(),
                case.variant(v).unwrap().expect_for(f),
                "{v} {f}"
            );
        }
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

/// `MATCH.ANY` / `MATCH.ANY.U64` (full warp and in a divergent branch) and `UBLKPF.L2`
/// (fixtures/s_match) agree with GB300 bitwise (200 runs, 2026-11-06).
#[test]
fn s_match_matches_hardware() {
    let case = Case::load(root().join("fixtures/s_match")).unwrap();
    let k = binary_kernel(&case, "good").unwrap();
    let n = k
        .code
        .iter()
        .filter(|i| matches!(i.op, ptxv_ir::Op::MatchAny { .. }))
        .count();
    assert_eq!(n, 3, "MATCH.ANY forms lowered");
    let launch = build_launch(&case, &k).unwrap();
    let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 100_000);
    assert!(v.findings.is_empty(), "{:#?}", v.findings);
    assert_eq!(v.outputs.len(), 1);
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
        "{report:?}"
    );
}

/// `HFMA2.BF16_V2` (register and pair-immediate forms), `HADD2.BF16_V2`, `HMUL2.BF16_V2` and
/// `FHADD.BF16` over normals, subnormals, near-maximum values, zeros, infinities, NaNs and ties
/// (fixtures/s_hfma2_bf16) agree with GB300 bitwise (20 runs, 2026-11-06).
#[test]
fn s_hfma2_bf16_matches_hardware() {
    let case = Case::load(root().join("fixtures/s_hfma2_bf16")).unwrap();
    let k = binary_kernel(&case, "good").unwrap();
    let n = k
        .code
        .iter()
        .filter(|i| matches!(i.op, ptxv_ir::Op::Hfma2Bf16 { .. }))
        .count();
    assert_eq!(n, 4, "bf16 forms lowered");
    let n = k
        .code
        .iter()
        .filter(|i| matches!(i.op, ptxv_ir::Op::FfmaBf16 { .. }))
        .count();
    assert_eq!(n, 4, "FHADD.BF16 forms lowered");
    let launch = build_launch(&case, &k).unwrap();
    let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 100_000);
    assert!(v.findings.is_empty(), "{:#?}", v.findings);
    assert_eq!(v.outputs.len(), 1);
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
        "{report:?}"
    );
}

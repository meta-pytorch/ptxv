/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Search and certificate reductions: each reduced check must keep its verdicts. The good
//! kernel of a fixture certifies with the explorer's classes and outputs; its mutants, whose
//! bug sits in or right after the reduced region, are still refused by the certificate (or
//! found by its run) and found by the explorer.

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::explore::explore;
use ptxv_sim::machine::Machine;
use ptxv_sim::wellsync::certify;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Third-party kernel binaries are not distributed (fixtures/VENDOR_FIXTURES.toml): true, with a
/// message on stderr, when a file of a vendor fixture the test needs is missing (it then skips).
fn vendor_missing(needs: &[&str]) -> bool {
    ptxv_case::skip_missing_vendor_files(&root().join("fixtures"), needs)
}

/// A PTX fixture's variant through one front-end.
fn machine(case: &Case, variant: &str, front: &str) -> Machine {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "ptxv-redtest-{}-{variant}-{}-{n}",
        case.m.name,
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let ptx = case.variant_ptx(variant).unwrap();
    let k = match front {
        "ptx" => ptxv_ptx::parse_kernel_for(
            &ptx,
            Some(&case.m.entry),
            case.m
                .ptxas_toolkit
                .as_deref()
                .and_then(ptxv_ptx::toolkit_version),
        )
        .unwrap(),
        _ => {
            let p = dir.join("k.ptx");
            std::fs::write(&p, &ptx).unwrap();
            let c = dir.join("k.cubin");
            ptxv_sass::ptxas(&p, &c, &case.m.arch).unwrap();
            ptxv_sass::load_cubin(&c, &case.m.entry).unwrap()
        }
    };
    let launch = build_launch(case, &k).unwrap();
    Machine::new(Arc::new(k), Arc::new(launch))
}

/// SASS BREAK (E-BSYNC, C-WARP): in e_brk the lanes reach `BREAK.RELIABLE B1` by two paths
/// (ptxas's branch chain for predicated shared stores). The certificate no longer requires the
/// breaking lanes to have arrived together. The good kernel certifies with the explorer's
/// verdict and outputs; BRK1 (two lanes write one word inside the branch chain) and BRK2 (no
/// barrier before the neighbour read) are refused and found.
#[test]
fn break_reached_by_two_paths_keeps_verdicts() {
    let case = Case::load(root().join("fixtures/e_brk")).unwrap();
    let m = machine(&case, "good", "sass");
    let brk = (0..m.k.code.len())
        .filter(|&pc| matches!(m.k.code[pc].op, ptxv_ir::Op::Break { .. }))
        .count();
    assert!(brk > 0, "e_brk: ptxas emitted no BREAK");
    let o = certify(m.clone());
    let cv = o
        .verdict
        .unwrap_or_else(|| panic!("e_brk good does not certify: {:#?}", o.reasons));
    let ev = explore(m, 2_000_000);
    assert!(ev.complete, "e_brk good: exploration incomplete");
    assert_eq!(cv.classes(), ev.classes());
    assert_eq!(cv.classes(), ["clean"]);
    let (mut a, mut b) = (cv.outputs.clone(), ev.outputs.clone());
    a.sort();
    b.sort();
    assert_eq!(
        a, b,
        "e_brk good: certified outputs differ from the explorer's"
    );
    for v in ["BRK1", "BRK2"] {
        let m = machine(&case, v, "sass");
        let o = certify(m.clone());
        if let Some(cv) = &o.verdict {
            assert_ne!(cv.classes(), ["clean"], "{v}: certified clean");
        }
        let ev = explore(m, 2_000_000);
        assert_eq!(
            ev.classes(),
            case.variant(v).unwrap().expect_for("sass"),
            "{v}: {:#?}",
            ev.findings
        );
    }
}

/// E-SPIN peeks (spin.rs `is_peek`) through a warp vote: the test's result reaches its branch
/// through `vote.any` (trtllm-gen's `VOTEU.ANY UP0, P1`); the test is still a peek (VOTE,
/// clean, no E-SPIN finding). A loop that repeats while the vote is false is a spin
/// (VOTESPIN, unmodeled), as without the vote (SPIN).
#[test]
fn peek_through_a_warp_vote() {
    let case = Case::load(root().join("fixtures/e_peek_loop")).unwrap();
    for v in ["good", "SPIN", "VOTE", "VOTESPIN"] {
        for front in ["ptx", "sass"] {
            let ev = explore(machine(&case, v, front), 200_000);
            assert_eq!(
                ev.classes(),
                case.variant(v).unwrap().expect_for(front),
                "{v}/{front}: {:#?}",
                ev.findings
            );
        }
    }
}

/// E-SPIN recognition (spin.rs `poll_loops`): a counted loop with a volatile load and an
/// unconditional back branch is explored as ordinary steps when no exit depends on the load
/// (e_vol_loop good, like the dispatch kernel's pipelined main loop); a loop whose exit tests
/// the loaded value keeps the `unmodeled` finding (SPIN, SASS), also when it reaches the exit
/// only through the counter carried around the loop (CARRY).
#[test]
fn counted_loop_with_volatile_load_is_not_a_poll() {
    let case = Case::load(root().join("fixtures/e_vol_loop")).unwrap();
    for (v, front) in [
        ("good", "ptx"),
        ("good", "sass"),
        ("SPIN", "ptx"),
        ("SPIN", "sass"),
        ("CARRY", "ptx"),
        ("CARRY", "sass"),
    ] {
        let m = machine(&case, v, front);
        let ev = explore(m, 200_000);
        assert_eq!(
            ev.classes(),
            case.variant(v).unwrap().expect_for(front),
            "{v}/{front}: {:#?}",
            ev.findings
        );
        if front == "sass" && v != "good" {
            assert!(
                ev.findings
                    .iter()
                    .any(|f| f.kind == "unmodeled" && f.msg.contains("E-SPIN")),
                "{:#?}",
                ev.findings
            );
        }
    }
}

/// S-TMAP: a tensor-map parameter's element size, extents, strides and box come from the
/// captured descriptor, not from the dtype the case gives the buffer behind it
/// (FusedDoubleRMSNormBwd: an f32 map over storage captured as f64 made every TMA copy count
/// twice its bytes toward complete_tx). c_mm_bf16_nnt's bf16 operand b, declared f32 here,
/// still gets 2-byte elements.
#[test]
fn tensormap_param_geometry_comes_from_the_descriptor() {
    if vendor_missing(&["c_mm_bf16_nnt"]) {
        return;
    }
    let mut case = Case::load(root().join("fixtures/c_mm_bf16_nnt")).unwrap();
    let k = ptxv_sass::load_cubin(
        &root().join("fixtures/c_mm_bf16_nnt/k0.cubin"),
        &case.m.entry,
    )
    .unwrap();
    let b = case.m.buffers.iter_mut().find(|b| b.name == "b").unwrap();
    assert_eq!((b.dtype.as_str(), &b.shape[..]), ("bf16", &[64u64, 64][..]));
    b.dtype = "f32".into();
    b.shape = vec![64, 32];
    let launch = build_launch(&case, &k).unwrap();
    let t = launch
        .tensormaps
        .iter()
        .find(|t| t.buffer == "b")
        .expect("no tensor map over b");
    assert_eq!(t.esize, 2);
    assert_eq!(t.dims, [64, 64, 1]);
    // The extent-1 outer dim's stride is 0 in the descriptor (unused).
    assert_eq!(t.strides, [128, 0]);
    assert_eq!(t.boxd, [64, 16, 1]);
}

/// C-ORDER dead reads (wellsync.rs `dead_reads`, S-INIT): in e_dead_read warp 1 loads a
/// shared word unordered with thread 0's store. The good kernel never uses the value and
/// certifies (with LDS/STS strong, S-SMEM-STRONG) with the explorer's verdict and outputs;
/// DR1 (stores the value) and DR2 (branches on it) are refused and found.
#[test]
fn unordered_dead_read_certifies_and_used_reads_do_not() {
    let case = Case::load(root().join("fixtures/e_dead_read")).unwrap();
    let m = machine(&case, "good", "sass");
    let o = certify(m.clone());
    let cv = o
        .verdict
        .unwrap_or_else(|| panic!("e_dead_read good does not certify: {:#?}", o.reasons));
    assert!(
        cv.findings.iter().any(|f| f.sig.starts_with("dead-read:")),
        "{:#?}",
        cv.findings
    );
    let ev = explore(m, 200_000);
    assert!(ev.complete);
    assert_eq!(cv.classes(), ev.classes());
    assert_eq!(cv.classes(), ["clean"]);
    let (mut a, mut b) = (cv.outputs.clone(), ev.outputs.clone());
    a.sort();
    b.sort();
    assert_eq!(
        a, b,
        "e_dead_read good: certified outputs differ from the explorer's"
    );
    for v in ["DR1", "DR2"] {
        let m = machine(&case, v, "sass");
        let o = certify(m.clone());
        assert!(o.verdict.is_none(), "{v}: certified");
        let ev = explore(m, 200_000);
        assert_eq!(
            ev.classes(),
            case.variant(v).unwrap().expect_for("sass"),
            "{v}: {:#?}",
            ev.findings
        );
    }
}

/// C-ORDER equal-value writes (order.rs `pair`): in e_same_strong two warps store one value to
/// one shared word without order. With a strong global load in the run the plain-LDS/STS
/// certificate refuses (S-SAME-VALUE); the strong one holds, with the explorer's verdict and
/// outputs. SV1 (different values) is refused and found nondeterministic.
#[test]
fn equal_value_strong_writes_certify_and_different_ones_do_not() {
    let case = Case::load(root().join("fixtures/e_same_strong")).unwrap();
    let m = machine(&case, "good", "sass");
    let o = certify(m.clone());
    let cv = o
        .verdict
        .unwrap_or_else(|| panic!("e_same_strong good does not certify: {:#?}", o.reasons));
    let ev = explore(m, 200_000);
    assert!(ev.complete);
    assert_eq!(cv.classes(), ev.classes());
    assert_eq!(cv.classes(), ["clean"]);
    let (mut a, mut b) = (cv.outputs.clone(), ev.outputs.clone());
    a.sort();
    b.sort();
    assert_eq!(a, b);
    let m = machine(&case, "SV1", "sass");
    let o = certify(m.clone());
    assert!(o.verdict.is_none(), "SV1: certified");
    let ev = explore(m, 200_000);
    assert_eq!(
        ev.classes(),
        case.variant("SV1").unwrap().expect_for("sass"),
        "{:#?}",
        ev.findings
    );
}

/// E-RACE-DEAD (race_dead.rs): in e_race_dead every thread loads a word thread 0 then
/// overwrites. The good kernel's racing loads are never used: explored and certified, on both
/// front-ends, the races are `race-dead` findings and the verdict is clean. RD1 (warp 1 stores
/// the value it loaded) keeps the race.
#[test]
fn dead_racy_loads_are_race_dead_and_used_ones_stay_races() {
    let case = Case::load(root().join("fixtures/e_race_dead")).unwrap();
    for front in ["ptx", "sass"] {
        let m = machine(&case, "good", front);
        let ev = explore(m.clone(), 200_000);
        assert!(ev.complete);
        assert_eq!(ev.classes(), ["clean"], "{front}: {:#?}", ev.findings);
        assert!(
            ev.findings.iter().any(|f| f.kind == "race-dead"),
            "{front}: {:#?}",
            ev.findings
        );
        let o = certify(m);
        let cv = o
            .verdict
            .unwrap_or_else(|| panic!("{front}: good does not certify: {:#?}", o.reasons));
        assert_eq!(cv.classes(), ["clean"]);
        assert!(cv.findings.iter().any(|f| f.kind == "race-dead"));
        assert_eq!(cv.outputs, ev.outputs);
        let m = machine(&case, "RD1", front);
        let ev = explore(m.clone(), 200_000);
        assert_eq!(
            ev.classes(),
            case.variant("RD1").unwrap().expect_for(front),
            "{front}: {:#?}",
            ev.findings
        );
        assert!(!ev.findings.iter().any(|f| f.kind == "race-dead"));
        let o = certify(m);
        assert!(o.verdict.is_none(), "{front}: RD1 certified");
    }
}

/// E-RACE-DEAD and mixed-size data races (MISS 226; probes/race_dead_mixed, e_race_dead
/// RDMIX). PTX ISA 9.5 section 8.7.2: the memory model's axioms do not apply to a program with a
/// mixed-size data race, so a race whose operations overlap partially stays `race` on the PTX
/// front although its loaded value is dead: a strong (good) or weak (PLAIN) u8 read of a u32
/// write, masked to 0, and RDMIX's byte loads; the certificate does not hold. Uniform-size races
/// with a dead value stay `race-dead` and clean, explored and certified (UNIFORM, UNIFORM_PLAIN).
/// The NIC's accesses too (MISS 194): e_nic_cta WORDDEAD, a 4-byte read of an RDMA write's
/// 8-byte chunk masked to 0, races; so does CQEDEAD, a masked .cta read of CQE bytes 60-63,
/// whose overlap with the CQE write is not modeled (only a pair known to be one location is
/// dropped). ptxas drops both masked loads on SASS: no race there.
/// The SASS front does not decide overlap: RDMIX's byte loads are `race-dead` there, also under
/// the GB300-profile re-run, and ptxas 13.0 drops the probe's masked loads (it stores 0), so
/// those variants have no race at all. Neither does the level-2 hardware model (`hw_order`).
/// Until 2026-11-12 good, PLAIN and RDMIX were race-dead and clean on the PTX front.
#[test]
fn mixed_size_races_are_not_race_dead() {
    let probe = Case::load(root().join("probes/race_dead_mixed")).unwrap();
    let fx = Case::load(root().join("fixtures/e_race_dead")).unwrap();
    let nic = Case::load(root().join("fixtures/e_nic_cta")).unwrap();
    // (case, variant, front, race-dead finding expected)
    let rows = [
        (&probe, "good", "ptx", false),
        (&probe, "good", "sass", false),
        (&probe, "PLAIN", "ptx", false),
        (&probe, "PLAIN", "sass", false),
        (&probe, "UNIFORM", "ptx", true),
        (&probe, "UNIFORM", "sass", false),
        (&probe, "UNIFORM_PLAIN", "ptx", true),
        (&probe, "UNIFORM_PLAIN", "sass", false),
        (&fx, "RDMIX", "ptx", false),
        (&fx, "RDMIX", "sass", true),
        (&nic, "WORDDEAD", "ptx", false),
        (&nic, "WORDDEAD", "sass", false),
        (&nic, "CQEDEAD", "ptx", false),
        (&nic, "CQEDEAD", "sass", false),
    ];
    for (case, v, front, dead) in rows {
        let m = machine(case, v, front);
        let ev = explore(m.clone(), 200_000);
        assert!(ev.complete, "{v}/{front}");
        let want = case.variant(v).unwrap().expect_for(front);
        assert_eq!(ev.classes(), want, "{v}/{front}: {:#?}", ev.findings);
        assert_eq!(
            ev.findings.iter().any(|f| f.kind == "race-dead"),
            dead,
            "{v}/{front}: {:#?}",
            ev.findings
        );
        let o = certify(m);
        if want == ["race"] {
            assert!(o.verdict.is_none(), "{v}/{front}: certified");
        } else if case.m.nic_qps.is_empty() {
            // The certificate leaves NIC queue pairs to the exploration (docs/nic.md).
            let cv = o
                .verdict
                .unwrap_or_else(|| panic!("{v}/{front}: not certified: {:#?}", o.reasons));
            assert_eq!(cv.classes(), want, "{v}/{front}: certificate");
            assert_eq!(cv.findings.iter().any(|f| f.kind == "race-dead"), dead);
        }
    }
    // Placement: only the strict model on the PTX front decides overlap. The GB300-profile
    // re-run (the CLI's `sass_gb300`) runs the SASS front, which takes every pair as one
    // location: RDMIX is race-dead and certified there, as before.
    let p = ptxv_sim::hwprofile::GB300;
    let m = ptxv_sim::hwprofile::machine(&machine(&fx, "RDMIX", "sass"), p);
    let mut v = ptxv_sim::wellsync::check_certified(m.clone(), 100_000);
    ptxv_sim::hwprofile::discharge(&mut v, &m, p);
    let dead_only = |fs: &[ptxv_sim::machine::Finding]| {
        fs.iter().any(|f| f.kind == "race-dead") && !fs.iter().any(|f| f.kind == "race")
    };
    assert!(
        dead_only(&v.findings),
        "RDMIX GB300 profile: {:#?}",
        v.findings
    );
    assert_eq!(ptxv_sim::hwprofile::result(&v), "certified");
    // The level-2 hardware model that labels the strict level's races (twolevel, E-HW-L1)
    // orders by `hw_order` and does not decide overlap either: RDMIX's byte load stays
    // race-dead there on the PTX front, so the strict PTX race is labelled, not dropped.
    let st = Arc::new(ptxv_sim::l1::L1Stats::default());
    let m2 = ptxv_sim::twolevel::hw_machine(&machine(&fx, "RDMIX", "ptx"), &st);
    assert!(m2.launch.hw_order);
    let v2 = explore(m2, 200_000);
    assert!(v2.complete);
    assert!(
        dead_only(&v2.findings),
        "RDMIX/ptx hardware model: {:#?}",
        v2.findings
    );
}

/// E-RACE-DEAD on a tensormap proxy fence (exec.rs `Op::TmapFence`): in e_race_dead_tmap warp
/// 1 acquire-fences a tensor map that warp 0 publishes concurrently and never uses it (a CuTe
/// grouped GEMM's epilogue warps 1..3). The fence's read races with the publication, but its
/// snapshot reaches no TMA op: explored and certified, on both front-ends, the race is
/// `race-dead` and the verdict is clean (SASS: but for sb-unmeasured: contracts, MISS 170). TF1 (thread 32 TMA-loads with the map after its fence)
/// keeps the race. TF2 (warp 1 fences after a relaxed wait for the publication, so its snapshot
/// is current in every interleaving, and thread 32 then TMA-loads relying on it after an acquire)
/// has the fence race as its only finding: it stays `race` only because the probe leaves the
/// snapshot undefined (the killing variant for a probe that does not).
#[test]
fn unused_tensormap_fence_snapshots_are_race_dead() {
    let case = Case::load(root().join("fixtures/e_race_dead_tmap")).unwrap();
    for front in ["ptx", "sass"] {
        let m = machine(&case, "good", front);
        // At the gb300 level. On the SASS front the LDS address reads rest on another opcode's
        // read scoreboard outside R10's measured window: sb-unmeasured: contracts at every level
        // (MISS 170; before, sb-hwq: and discharged). Nothing else.
        let ev = gb300(explore(m.clone(), 200_000));
        assert!(ev.complete);
        let unmeasured = |f: &ptxv_sim::machine::Finding| {
            f.kind == "contract" && f.sig.starts_with(ptxv_sass::scoreboard::HW_UNMEASURED_SIG)
        };
        let want: &[&str] = if front == "sass" { &["contract"] } else { &["clean"] };
        assert_eq!(ev.classes(), want, "{front}: {:#?}", ev.findings);
        assert!(
            ev.findings
                .iter()
                .all(|f| f.kind != "contract" || unmeasured(f)),
            "{front}: {:#?}",
            ev.findings
        );
        assert!(
            ev.findings.iter().any(|f| f.kind == "race-dead"),
            "{front}: {:#?}",
            ev.findings
        );
        let o = certify(m);
        let cv = gb300(
            o.verdict
                .unwrap_or_else(|| panic!("{front}: good does not certify: {:#?}", o.reasons)),
        );
        assert_eq!(cv.classes(), want);
        assert!(cv.findings.iter().any(|f| f.kind == "race-dead"));
        let m = machine(&case, "TF1", front);
        let ev = explore(m.clone(), 200_000);
        assert!(
            ev.classes().iter().any(|c| c == "race"),
            "{front}: {:#?}",
            ev.findings
        );
        assert!(!ev.findings.iter().any(|f| f.kind == "race-dead"));
        let o = certify(m);
        assert!(o.verdict.is_none(), "{front}: TF1 certified");
        let m = machine(&case, "TF2", front);
        let ev = gb300(explore(m.clone(), 200_000));
        assert!(ev.complete);
        let race: &[&str] = if front == "sass" { &["contract", "race"] } else { &["race"] };
        assert_eq!(ev.classes(), race, "{front}: {:#?}", ev.findings);
        let o = certify(m);
        assert!(
            o.verdict.as_ref().is_none_or(|v| v.classes() != ["clean"]),
            "{front}: TF2 certified clean"
        );
    }
}

/// Position dependence through special registers other than %ctaid (fixtures/e_pos_sreg):
/// E-CTA-REP and the E-CTA-INDEP symmetry check keep the full classes when %clusterid decides
/// one CTA's fault (CLUSTERID) or schedule-dependent output (CTICKET), and E-CTA-REP keeps a
/// representative's several outputs (NONDET). Before 2026-11-08 position mode seeded only
/// %ctaid and the launch verdict kept one output: PTX CLUSTERID / CTICKET and NONDET on both
/// front-ends were clean.
#[test]
fn position_through_other_special_registers() {
    let mut case = Case::load(root().join("fixtures/e_pos_sreg")).unwrap();
    // One ticket lane per warp preserves both frontends' orders without enumerating 64! outputs.
    for variant in &mut case.m.variants {
        if matches!(variant.id.as_str(), "CTICKET" | "NONDET") {
            let replacement = variant.replace.as_mut().unwrap();
            let atomic = "atom.shared.add.u32 %r5, [s], 1;";
            assert!(replacement.contains(atomic));
            *replacement = replacement.replace(
                atomic,
                "and.b32 %r6, %r1, 31; setp.ne.u32 %p1, %r6, 0; @%p1 bra STORE; atom.shared.add.u32 %r5, [s], 1;",
            );
        }
    }
    for front in ["ptx", "sass"] {
        for variant in ["good", "CLUSTERID", "CTICKET", "NONDET"] {
            let make = || machine(&case, variant, front);
            let expect = case.variant(variant).unwrap().expect_for(front);
            let rep = ptxv_sim::rep::check(&make, 200_000, &mut |_, _| {}, &mut || {
                ptxv_sim::wellsync::check_certified(make(), 200_000)
            });
            assert_eq!(
                rep.classes(),
                expect,
                "rep {front} {variant}: {:#?}",
                rep.findings
            );
            let sym = ptxv_sim::explore::explore_symmetric(&make, 200_000, &mut |_, _| {});
            assert!(sym.complete, "symmetry {front} {variant}: incomplete");
            assert_eq!(
                sym.classes(),
                expect,
                "symmetry {front} {variant}: {:#?}",
                sym.findings
            );
        }
    }
}

/// Value faults under the CTA reductions (fixtures/e_rep_divzero, num.rs
/// `value_fault_operands`): an integer division whose divisor is zero only in CTA 3, from
/// input data (DATA) or from %ctaid (POS). E-CTA-REP and the symmetry check keep the fault
/// (PTX; ptxas's division sequence does not trap on the SASS front), and the data-taint run,
/// the precondition of E-CTA-REP and N-EQUIV, reports DATA's divisor. Before 2026-11-08 both
/// were clean with --rep.
#[test]
fn value_faults_keep_their_class_under_cta_reductions() {
    let case = Case::load(root().join("fixtures/e_rep_divzero")).unwrap();
    for front in ["ptx", "sass"] {
        for variant in ["good", "DATA", "POS"] {
            let make = || machine(&case, variant, front);
            let expect = case.variant(variant).unwrap().expect_for(front);
            let rep = ptxv_sim::rep::check(&make, 200_000, &mut |_, _| {}, &mut || {
                ptxv_sim::wellsync::check_certified(make(), 200_000)
            });
            assert_eq!(
                rep.classes(),
                expect,
                "rep {front} {variant}: {:#?}",
                rep.findings
            );
            let sym = ptxv_sim::explore::explore_symmetric(&make, 200_000, &mut |_, _| {});
            assert_eq!(
                sym.classes(),
                expect,
                "symmetry {front} {variant}: {:#?}",
                sym.findings
            );
        }
        // SASS: ptxas's division sequence has no faulting operation.
        for (variant, flagged) in [("good", false), ("DATA", front == "ptx"), ("POS", false)] {
            let mut m = machine(&case, variant, front);
            let mut ns = ptxv_sim::num::NumState::with_mma_eps(m.launch.mma_model.eps());
            ns.taint.all_global = true;
            m.num = Some(Box::new(ns));
            let (m, _, _) = ptxv_sim::explore::run_once(m);
            let v = &m.num.as_ref().unwrap().taint.violations;
            assert_eq!(
                v.iter().any(|x| x.contains("value-fault operand")),
                flagged,
                "data taint {front} {variant}: {v:?}"
            );
        }
    }
}

/// Cluster symmetry fixture (fixtures/e_sym) under E-CTA-REP and the symmetry check: each
/// variant keeps its classes on both front-ends. Before 2026-11-08 E-CTA-REP dropped the
/// SASS SYM2 representative's second output ([violation] for [nondeterministic, violation]).
#[test]
fn symmetry_fixture_keeps_classes_under_cta_reductions() {
    let case = Case::load(root().join("fixtures/e_sym")).unwrap();
    for front in ["ptx", "sass"] {
        for variant in ["good", "SYM1", "SYM2"] {
            let make = || machine(&case, variant, front);
            let expect = case.variant(variant).unwrap().expect_for(front);
            let rep = ptxv_sim::rep::check(&make, 200_000, &mut |_, _| {}, &mut || {
                ptxv_sim::wellsync::check_certified(make(), 200_000)
            });
            assert_eq!(
                rep.classes(),
                expect,
                "rep {front} {variant}: {:#?}",
                rep.findings
            );
            let sym = ptxv_sim::explore::explore_symmetric(&make, 200_000, &mut |_, _| {});
            assert_eq!(
                sym.classes(),
                expect,
                "symmetry {front} {variant}: {:#?}",
                sym.findings
            );
        }
    }
}

/// E-CTA-REP (rep.rs, fixtures/e_cta_rep): with representative CTAs the check keeps the full
/// certificate's classes. good: two representatives (CTAs 0-6 and the tail CTA 7) for 8 CTAs,
/// clean. TAIL1 (the tail CTA stores past the end), AND and STRIDE (CTAs write each other's
/// elements) and N1024 (CTA 7, never simulated, stores past the end: its replayed address)
/// are found. DATACTL (a branch on loaded data) is refused, and the fallback decides it.
#[test]
fn representative_ctas_keep_verdicts() {
    let case = Case::load(root().join("fixtures/e_cta_rep")).unwrap();
    for front in ["ptx", "sass"] {
        for variant in ["good", "TAIL1", "AND", "STRIDE", "N1024", "DATACTL"] {
            let make = || machine(&case, variant, front);
            let full = ptxv_sim::wellsync::check_certified(make(), 200_000);
            let mut fell_back = false;
            let rep = ptxv_sim::rep::check(&make, 200_000, &mut |_, _| {}, &mut || {
                fell_back = true;
                ptxv_sim::wellsync::check_certified(make(), 200_000)
            });
            let expect = case.variant(variant).unwrap().expect_for(front);
            assert_eq!(
                full.classes(),
                expect,
                "{front} {variant}: {:#?}",
                full.findings
            );
            assert_eq!(
                rep.classes(),
                expect,
                "{front} {variant}: {:#?}",
                rep.findings
            );
            let used = rep.assumptions.iter().any(|a| a == "E-CTA-REP");
            assert_eq!(
                used,
                variant != "DATACTL",
                "{front} {variant}: {:#?}",
                rep.notes
            );
            assert_eq!(fell_back, !used, "{front} {variant}");
            if variant == "good" {
                assert!(
                    rep.notes
                        .iter()
                        .any(|n| n.contains("2 representatives for 8 CTAs")),
                    "{front}: {:#?}",
                    rep.notes
                );
            }
            if variant == "DATACTL" {
                assert!(
                    rep.notes
                        .iter()
                        .any(|n| n.starts_with("E-CTA-REP refused: data-dependent control")),
                    "{front}: {:#?}",
                    rep.notes
                );
            }
        }
    }
}

/// A PTX fixture's variant through one front-end, or None when it cannot be built (no PTX,
/// several launches, a front-end refusal).
fn try_machine(case: &Case, variant: &str, front: &str) -> Option<Machine> {
    if case.m.cubin.is_some() || !case.m.launches.is_empty() {
        return None;
    }
    // Per call: `verdicts_do_not_depend_on_earlier_checks_in_the_thread` builds one variant on
    // two threads at once.
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "ptxv-redall-{}-{variant}-{front}-{}-{n}",
        case.m.name,
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let ptx = case.variant_ptx(variant).ok()?;
    let k = match front {
        "ptx" => ptxv_ptx::parse_kernel(&ptx, Some(&case.m.entry)).ok()?,
        _ => {
            let p = dir.join("k.ptx");
            std::fs::write(&p, &ptx).ok()?;
            let c = dir.join("k.cubin");
            ptxv_sass::ptxas(&p, &c, &case.m.arch).ok()?;
            ptxv_sass::load_cubin(&c, &case.m.entry).ok()?
        }
    };
    let _ = std::fs::remove_dir_all(&dir);
    let launch = build_launch(case, &k).ok()?;
    Some(Machine::new(Arc::new(k), Arc::new(launch)))
}

/// One fixture variant of `every_fixture_keeps_classes_under_cta_reductions`: (failures,
/// checked), or None when the variant does not build or its unreduced exploration exceeds
/// `cap` states.
fn sweep_one(case: &Case, v: &str, front: &str, cap: usize) -> Option<(Vec<String>, bool)> {
    let m = try_machine(case, v, front)?;
    let base = explore(m.clone(), cap);
    if !base.complete {
        return None;
    }
    let make = || m.clone();
    let mut refused = false;
    let mut fails = Vec::new();
    let rep = ptxv_sim::rep::check(&make, cap, &mut |_, _| {}, &mut || {
        refused = true;
        base.clone()
    });
    let name = format!("{}/{v}/{front}", case.m.name);
    // E-CTA-REP's cross-CTA check reports conflicting global accesses of different CTAs as a
    // `cross-cluster` race whatever the values (a same-value write is same-value only on this
    // input): stricter than the base, never cleaner. That race must be there.
    let rep_cross = rep
        .findings
        .iter()
        .any(|f| f.sig.starts_with("cross-cluster:"));
    if rep_cross && !rep.classes().contains(&"race".to_string()) {
        fails.push(format!(
            "rep {name}: cross-cluster finding without a race class"
        ));
    }
    if !refused && !rep_cross && rep.classes() != base.classes() {
        fails.push(format!(
            "rep {name}: {:?} vs {:?}",
            rep.classes(),
            base.classes()
        ));
    }
    let sym = ptxv_sim::explore::explore_symmetric(&make, cap, &mut |_, _| {});
    // Clusters that share global memory refute slicing's premise: the slice reports it as a
    // `cross-cluster` race (and a slice can then block on a cluster it does not run), never
    // the base verdict. That race must be there; the classes are not comparable.
    let cross = sym
        .findings
        .iter()
        .any(|f| f.sig.starts_with("cross-cluster:"));
    if cross && !sym.classes().contains(&"race".to_string()) {
        fails.push(format!(
            "symmetry {name}: cross-cluster finding without a race class"
        ));
    }
    if sym.complete && !cross && sym.classes() != base.classes() {
        fails.push(format!(
            "symmetry {name}: {:?} vs {:?}",
            sym.classes(),
            base.classes()
        ));
    }
    Some((fails, true))
}

/// The parts of `every_fixture_keeps_classes_under_cta_reductions`, each a test of its own, so
/// that CI's shards and lanes run them apart: part k takes the jobs whose index in the sorted
/// job list is k modulo REDUCTION_PARTS. `reduction_part_tests!` below has one test per part,
/// checked against REDUCTION_PARTS when the tests compile.
const REDUCTION_PARTS: usize = 8;

/// Every single-launch PTX fixture of 2 to 16 CTAs, every variant, both front-ends: where the
/// unreduced exploration finishes within the cap, E-CTA-REP (`--rep`, when it does not refuse)
/// and the symmetry check (`--slices`, when complete) give its classes. A reduction cannot drop
/// a class silently anywhere in the suite (2026-11-08: E-CTA-REP dropped a representative's
/// second output, fixtures/e_sym SYM2 on SASS, and a division by zero outside the
/// representative, fixtures/e_rep_divzero). This is part `part` of the jobs
/// (REDUCTION_PARTS).
fn every_fixture_keeps_classes_under_cta_reductions(part: usize) {
    // Variants whose unreduced exploration finishes within CAP states (the others are the
    // dedicated tests' business; this sweep must stay cheap).
    const CAP: usize = 5_000;
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root().join("fixtures"))
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("case.toml").exists())
        .collect();
    dirs.sort();
    let mut all: Vec<(Arc<Case>, String, &str)> = Vec::new();
    for d in dirs {
        let Ok(case) = Case::load(&d) else { continue };
        // At most 16 CTAs: the symmetry check falls back to exploring every slice.
        let ctas = case.m.grid.iter().product::<u32>();
        if !(2..=16).contains(&ctas) {
            continue;
        }
        let case = Arc::new(case);
        for v in case.m.variants.iter().map(|v| v.id.clone()) {
            for front in ["ptx", "sass"] {
                all.push((case.clone(), v.clone(), front));
            }
        }
    }
    let jobs: Vec<_> = all
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % REDUCTION_PARTS == part)
        .map(|(_, j)| j)
        .collect();
    // One job per thread (each a fresh thread: no cache carries over), as many at once as
    // the machine has cores.
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: std::sync::Mutex<(usize, Vec<String>)> = Default::default();
    let n = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(32);
    std::thread::scope(|sc| {
        for _ in 0..n {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((case, v, front)) = jobs.get(i) else {
                        break;
                    };
                    let r = std::thread::scope(|s2| {
                        s2.spawn(|| sweep_one(case, v, front, CAP)).join().unwrap()
                    });
                    if let Some((fs, ok)) = r {
                        let mut o = out.lock().unwrap();
                        o.0 += ok as usize;
                        o.1.extend(fs);
                    }
                }
            });
        }
    });
    let (checked, fails) = out.into_inner().unwrap();
    eprintln!(
        "part {part}: {checked} of {} fixture variants checked under E-CTA-REP and the symmetry \
         check",
        jobs.len()
    );
    // About 50 of each part's jobs finish within CAP today (2026-11-12).
    assert!(
        checked >= 20,
        "part {part}: only {checked} fixture variants checked"
    );
    assert!(fails.is_empty(), "part {part}: {fails:#?}");
}

/// The part tests: `name: part, ignore reason;` per part. The list must name every part
/// 0..REDUCTION_PARTS once, in order, or the tests do not compile: a part without its test
/// would leave its jobs unrun.
macro_rules! reduction_part_tests {
    ($($name:ident: $part:literal, $why:literal;)*) => {
        $(
            #[test]
            #[ignore = $why]
            fn $name() {
                every_fixture_keeps_classes_under_cta_reductions($part);
            }
        )*
        const _: () = {
            let parts: &[usize] = &[$($part),*];
            assert!(parts.len() == REDUCTION_PARTS, "one test per part of the sweep");
            let mut k = 0;
            while k < parts.len() {
                assert!(parts[k] == k, "the part tests in order, 0..REDUCTION_PARTS");
                k += 1;
            }
        };
    };
}

reduction_part_tests! {
    every_fixture_keeps_classes_under_cta_reductions_part0: 0,
        "heavy: an eighth of the sweep's jobs; 75 s";
    every_fixture_keeps_classes_under_cta_reductions_part1: 1,
        "heavy: an eighth of the sweep's jobs; 270 s";
    every_fixture_keeps_classes_under_cta_reductions_part2: 2,
        "heavy: an eighth of the sweep's jobs; 1140 s";
    every_fixture_keeps_classes_under_cta_reductions_part3: 3,
        "heavy: an eighth of the sweep's jobs; 450 s";
    every_fixture_keeps_classes_under_cta_reductions_part4: 4,
        "heavy: an eighth of the sweep's jobs; 100 s";
    every_fixture_keeps_classes_under_cta_reductions_part5: 5,
        "heavy: an eighth of the sweep's jobs; 450 s";
    every_fixture_keeps_classes_under_cta_reductions_part6: 6,
        "heavy: an eighth of the sweep's jobs; 590 s";
    every_fixture_keeps_classes_under_cta_reductions_part7: 7,
        "heavy: an eighth of the sweep's jobs; 75 s";
}

/// Cached analyses must not leak between kernels checked one after another on one thread
/// (2026-11-08: E-BAR-OVER / E-BAR-ARRIVE walk caches keyed by an `Arc` address, which a
/// later kernel's control-flow data can reuse once the first machine is dropped). The named-
/// barrier fixtures, every variant on both front-ends, are checked in sequence on one thread
/// and each again on a fresh thread: the verdicts (classes, states, outputs) must agree.
#[test]
#[ignore = "heavy: 294 jobs run 3 times each; 534 s"]
fn verdicts_do_not_depend_on_earlier_checks_in_the_thread() {
    let names = [
        "e_barover",
        "e_aligned_bar",
        "e_bararv",
        "e_lds_atoms",
        "e_dead_read",
        "e_fa4_warp_roles",
        "e_fmax_fold",
    ];
    let mut jobs = Vec::new();
    for n in names {
        let case = Case::load(root().join("fixtures").join(n)).unwrap();
        for v in case.m.variants.iter().map(|v| v.id.clone()) {
            for front in ["ptx", "sass"] {
                jobs.push((n.to_string(), v.clone(), front.to_string()));
            }
        }
    }
    type Sum = Option<(Vec<String>, usize, usize, bool)>;
    let run = |n: &str, v: &str, front: &str| -> Sum {
        let case = Case::load(root().join("fixtures").join(n)).unwrap();
        let m = try_machine(&case, v, front)?;
        let e = explore(m.clone(), 100_000);
        let c = certify(m).verdict.map(|c| c.classes());
        let mut cl = e.classes();
        cl.extend(
            c.unwrap_or_default()
                .into_iter()
                .map(|x| format!("cert:{x}")),
        );
        Some((cl, e.states, e.distinct_outputs, e.complete))
    };
    // In sequence on one thread, twice over (the second pass sees every kernel's caches);
    // meanwhile each job again on a fresh thread of its own, a few at a time (the sequence
    // uses few of the explorer's threads).
    let pool = (ptxv_sim::explore::threads(usize::MAX) / 2)
        .saturating_sub(1)
        .max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let fresh: Vec<std::sync::Mutex<Option<Sum>>> =
        jobs.iter().map(|_| Default::default()).collect();
    let seq: Vec<Sum> = std::thread::scope(|sc| {
        let seq = sc.spawn(|| {
            let mut out = Vec::new();
            for pass in 0..2 {
                for (n, v, f) in &jobs {
                    let r = run(n, v, f);
                    if pass == 1 {
                        out.push(r);
                    }
                }
            }
            out
        });
        for _ in 0..pool {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((n, v, f)) = jobs.get(i) else {
                        break;
                    };
                    let r = std::thread::scope(|s2| s2.spawn(|| run(n, v, f)).join().unwrap());
                    *fresh[i].lock().unwrap() = Some(r);
                }
            });
        }
        seq.join().unwrap()
    });
    for (((n, v, f), s), fresh) in jobs.iter().zip(seq).zip(fresh) {
        let fresh = fresh.into_inner().unwrap().expect("fresh run");
        // A bounded search shares its cap among workers: its state count and outputs vary
        // from run to run. Complete ones are compared.
        if s.as_ref().is_some_and(|x| !x.3) && fresh.as_ref().is_some_and(|x| !x.3) {
            continue;
        }
        assert_eq!(s, fresh, "{n}/{v}/{f}: in sequence vs fresh thread");
    }
}

/// The verdict at the gb300 level: without the strict level's E-HW-SB-QUEUE contracts
/// (sb-hwq:), which the gb300 profile discharges.
fn gb300(mut v: ptxv_sim::explore::Verdict) -> ptxv_sim::explore::Verdict {
    v.findings.retain(|f| {
        !(f.kind == "contract" && f.sig.starts_with(ptxv_sass::scoreboard::HW_QUEUE_SIG))
    });
    v
}

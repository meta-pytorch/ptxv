/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Adversarial tests of proof rules (docs/soundness-harness.md, "adversarial kernels"): for
//! each rule, a kernel that violates the rule's premise while looking like the good case must
//! not come out clean or certified, and each test is the one a source mutant weakening the rule
//! (tools/mutate/srcmut.py) fails.

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::explore::explore;
use ptxv_sim::machine::Machine;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch_dir(name: &str, variant: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "ptxv-adv-{}-{name}-{variant}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ))
}

/// The front-ends a case runs on: binary-only cases (`cubin`) have no PTX.
fn fronts(case: &Case) -> &'static [&'static str] {
    if case.m.cubin.is_some() {
        &["sass"]
    } else {
        &["ptx", "sass"]
    }
}

/// The kernel of one variant through one front-end (multi-kernel launches concatenated).
fn kernel(case: &Case, variant: &str, front: &str) -> ptxv_ir::Kernel {
    if let Some(bin) = &case.m.cubin {
        assert_eq!(front, "sass", "{}: binary-only", case.m.name);
        let dir = scratch_dir(&case.m.name, variant);
        std::fs::create_dir_all(&dir).unwrap();
        let c = dir.join("k.cubin");
        std::fs::copy(case.dir.join(bin), &c).unwrap();
        ptxv_sass::scoreboard::patch_file(&c, &case.m.entry, &case.variant(variant).unwrap().patch)
            .unwrap();
        let k = ptxv_sass::load_cubin(&c, &case.m.entry).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        return k;
    }
    let ptx = case.variant_ptx(variant).unwrap();
    let one = |entry: &str| match front {
        "ptx" => ptxv_ptx::parse_kernel(&ptx, Some(entry)).unwrap(),
        _ => {
            let dir = scratch_dir(&case.m.name, variant);
            std::fs::create_dir_all(&dir).unwrap();
            let p = dir.join("k.ptx");
            std::fs::write(&p, &ptx).unwrap();
            let c = dir.join("k.cubin");
            ptxv_sass::ptxas(&p, &c, &case.m.arch).unwrap();
            ptxv_sass::scoreboard::patch_file(&c, entry, &case.variant(variant).unwrap().patch)
                .unwrap();
            let k = ptxv_sass::load_cubin(&c, entry).unwrap();
            let _ = std::fs::remove_dir_all(&dir);
            k
        }
    };
    if !case.m.launches.is_empty() {
        let ks = case.m.all_launches().iter().map(|l| one(l.entry)).collect();
        return ptxv_ir::Kernel::concat(ks);
    }
    one(&case.m.entry)
}

fn machine(case: &Case, variant: &str, front: &str) -> Machine {
    let k = kernel(case, variant, front);
    let launch = build_launch(case, &k).unwrap();
    Machine::new(Arc::new(k), Arc::new(launch))
}

fn load(name: &str) -> Case {
    Case::load(root().join("fixtures").join(name)).unwrap()
}

/// A case outside fixtures/ (`probes/<name>`).
fn load_at(dir: &str) -> Case {
    Case::load(root().join(dir)).unwrap()
}

/// Exploration classes of a machine (the search must complete).
fn classes(m: Machine, what: &str) -> Vec<String> {
    let v = explore(m, 2_000_000);
    assert!(v.complete, "{what}: exploration incomplete");
    v.classes()
}

/// Every listed variant gets its expected classes on both front-ends, and a certificate, when
/// one holds, reports exactly the exploration's classes (a certified verdict that hides a
/// finding of the exploration is a soundness miss).
fn adv(case: &str, variants: &[&str]) {
    let c = load(case);
    for v in variants {
        for &front in fronts(&c) {
            let what = format!("{case}/{v}/{front}");
            let m = machine(&c, v, front);
            let ex = classes(m.clone(), &what);
            assert_eq!(ex, c.variant(v).unwrap().expect_for(front), "{what}");
            if let Some(cv) = ptxv_sim::wellsync::certify(m).verdict {
                assert_eq!(cv.classes(), ex, "{what}: certified verdict differs");
            }
        }
    }
}

/// E-CONST-BRA (fixtures/e_const_bra, e_const_bra2). Premise: a branch the control-flow graph
/// folds goes the folded way in every execution. Conclusion: dropping the other edge from the
/// reductions' reachability loses no behaviour. Enforced: `exec.rs` reports a folded branch
/// that goes the other way as `unmodeled` (const-bra).
/// - good (SASS: `BRA.U !UP0` on `c[0x0][0x388]` = flag = 1) folds and is clean; with flag = 0
///   it folds the other way and the unbarriered read races. FLIP (the guard through a `xor`,
///   not propagated) is not folded on PTX and races.
/// - A wrong fold (the analysis decided the other way) must surface as `unmodeled` on every
///   schedule search, not as a clean verdict: this is the check the rule rests on.
/// - e_const_bra2: one entry launched with flag = 1 then flag = 0 (concatenated code, one fold
///   per launch): each copy folds its own way, and the second grid's race is found.
/// - PTX: `ld.param` is a load, so the PTX front folds nothing from parameters.
#[test]
fn e_const_bra_fold_is_enforced() {
    adv("e_const_bra", &["good", "FLIP"]);
    adv("e_const_bra2", &["good"]);
    let base = load("e_const_bra");
    let bra = |m: &Machine| {
        m.k.code
            .iter()
            .position(|i| matches!(i.op, ptxv_ir::Op::Bra { .. }) && !i.guard.is_always())
            .unwrap()
    };
    // flag = 0: folded the other way, race.
    let mut z = base.clone();
    z.m.params[1] = ptxv_case::ParamSpec::U32 { value: 0 };
    for front in ["ptx", "sass"] {
        let m = machine(&z, "good", front);
        assert_eq!(classes(m, front), ["race"], "flag 0 {front}");
    }
    // The SASS fold of good, and the PTX front's absence of one.
    let m = machine(&base, "good", "sass");
    let pc = bra(&m);
    let v = m.bar_cfg.fold[pc].expect("SASS: the parameter branch is folded");
    let mp = machine(&base, "good", "ptx");
    assert!(mp.bar_cfg.fold.iter().all(|f| f.is_none()), "PTX folds");
    // A wrong fold is reported, in every search.
    let mut wrong = m.clone();
    let mut cfg = (*wrong.bar_cfg).clone();
    cfg.fold[pc] = Some(!v);
    wrong.bar_cfg = Arc::new(cfg);
    let ex = explore(wrong.clone(), 100_000);
    assert!(
        ex.findings
            .iter()
            .any(|f| f.kind == "unmodeled" && f.sig.starts_with("const-bra")),
        "wrong fold not reported: {:#?}",
        ex.findings
    );
    let d = ptxv_sim::dpor::explore_dpor(wrong.clone(), 100_000);
    assert!(
        d.classes().iter().any(|c| c == "unmodeled"),
        "DPOR: {:?}",
        d.classes()
    );
    let o = ptxv_sim::wellsync::certify(wrong);
    if let Some(cv) = o.verdict {
        assert!(
            cv.classes().iter().any(|c| c == "unmodeled"),
            "certified: {:?}",
            cv.classes()
        );
    }
}

/// E-RET (fixtures/e_ret, binary-only: a device function called from two sites, the second
/// after a `BAR.SYNC`). Premise: every `RET.REL.NODEC` lands right after a `CALL.REL.NOINC` of
/// its function (`explore::ret_table`). Conclusion: the reductions' control-flow graph covers
/// every return. Enforced: `exec.rs` reports a return elsewhere as `unmodeled` (ret-target).
/// good is clean, NOBAR (the barrier NOPed) races. A table missing the second call site (the
/// return after the barrier then lands outside the graph) must be reported by every search.
#[test]
fn e_ret_sites_are_enforced() {
    adv("e_ret", &["good", "NOBAR"]);
    let case = load("e_ret");
    let m = machine(&case, "good", "sass");
    let ret =
        m.k.code
            .iter()
            .position(|i| matches!(i.op, ptxv_ir::Op::Ret { .. }))
            .unwrap();
    assert_eq!(m.bar_cfg.rets[ret].len(), 2, "two call sites");
    let mut wrong = m.clone();
    let mut cfg = (*wrong.bar_cfg).clone();
    cfg.rets[ret].truncate(1);
    wrong.bar_cfg = Arc::new(cfg);
    let unmod = |fs: &[ptxv_sim::machine::Finding]| {
        fs.iter()
            .any(|f| f.kind == "unmodeled" && f.sig.starts_with("ret-target"))
    };
    let ex = explore(wrong.clone(), 100_000);
    assert!(unmod(&ex.findings), "explore: {:#?}", ex.findings);
    let d = ptxv_sim::dpor::explore_dpor(wrong.clone(), 100_000);
    assert!(unmod(&d.findings), "DPOR: {:#?}", d.findings);
    if let Some(cv) = ptxv_sim::wellsync::certify(wrong).verdict {
        assert!(unmod(&cv.findings), "certified: {:?}", cv.classes());
    }
}

/// E-ELECT (fixtures/e_elect_mid). Premise (PTX): a kernel's correctness depends on the
/// leader only through the two end lanes. Here an end-lane leader stores before `bar.sync 0`
/// and a middle-lane leader after it, racing with thread 32's read. The PTX front explores the
/// lowest and the highest leader only, so good comes out clean: a known gap of the rule (under
/// the ISA any fixed lane may lead); `PTXV_ELECT=all` (every lane) finds the race. GB300
/// elects the lowest lane (probes/elect), so the SASS verdict clean is the hardware's.
/// Mutants: elect-lowonly (no highest-lane run; fixtures/e_elect B46 stops racing on PTX),
/// elect-high-sass (SASS elects the highest lane; B46 races on SASS).
#[test]
fn e_elect_middle_leader_gap() {
    adv("e_elect_mid", &["good", "ENDS"]);
    let c = load("e_elect_mid");
    // Every leader: the environment knob is read per search; no other test here elects.
    std::env::set_var("PTXV_ELECT", "all");
    let all = classes(machine(&c, "good", "ptx"), "e_elect_mid/good/ptx all");
    std::env::remove_var("PTXV_ELECT");
    assert_eq!(all, ["race"], "every leader");
}

/// E-CLOCK (fixtures/e_clock_timeout). Premise: no result depends on elapsed time (clocks read
/// 0, so a timeout never fires). good violates it: a poll with a clock64 timeout whose timeout
/// path reads the data without the flag. Both fronts report clean, while GB300 (the producer
/// sleeps 200 us, longer than the bound) takes the timeout path in 20/20 runs (out[0] = 0, the
/// model's only outcome 7; `ptxv hw`, 2026-11-09). A soundness miss inside the rule's
/// documented limit; this test pins the current verdicts so a fix shows up here.
#[test]
fn e_clock_timeout_path_unreached() {
    adv("e_clock_timeout", &["good", "NOTIMEOUT"]);
}

/// E-NANOSLEEP (fixtures/e_nanosleep): a sleep orders nothing. A read separated from another
/// thread's store by a 1 ms sleep only is a race on both fronts; with a barrier, clean.
#[test]
fn e_nanosleep_orders_nothing() {
    adv("e_nanosleep", &["good", "BAR"]);
}

/// E-BSYNC / E-LOCKSTEP (fixtures/e_lockstep_nobsync, binary-only). Premise: lanes run
/// together after a divergent region only where a `BSYNC` reconverges them. The lanes leave a
/// loop after lane-dependent trip counts and exchange `s[lane ^ 1]` with no `__syncwarp`;
/// ptxas reconverges the loop (good, clean). With the `BSYNC` or the `BSSY` NOPed, nothing
/// reconverges the warp: the early lanes may read their neighbour's slot before it is written
/// (nondeterministic, and an unwritten shared read). GB300 (`ptxv hw --runs 50`, 2026-11-09):
/// good 50/50 bitwise; NOBSYNC and NOBSSY 1/50 runs with 16 words differing (the even lanes),
/// so the hardware does not reconverge the warp without the barrier either.
/// Mutant bsync-own: a `BSYNC` waits only for the lanes that reached it, not for the lanes its
/// `BSSY` recorded (good then runs the early lanes ahead).
/// Certificate miss (2026-11-09, MISS 30, fixed): `--certify` certified NOBSYNC clean in one
/// run; the certificate must now report exploration's classes for every variant.
#[test]
fn e_bsync_reconverges_divergent_loop() {
    adv("e_lockstep_nobsync", &["good", "NOBSYNC", "NOBSSY"]);
}

/// E-WARP-ORDER (fixtures/e_warp_order_div). Premise: program order orders the lanes of a
/// warp only while it is converged. good is probes/warp_order's `diverge` control (writer and
/// reader lanes on the two sides of a branch; GB300 wrong in 99.98% of checks): nondeterministic
/// on both fronts. CONV issues the same volatile accesses predicated per lane on consecutive
/// instructions of the converged warp: SASS clean is the hardware's (probes/warp_order modes
/// 0-6, 0 wrong of 4.5e11). The PTX front also reports CONV clean, although PTX gives the lanes
/// no order without `bar.warp.sync` (a PTX-contract miss, reported 2026-11-09; with plain
/// accesses the shape is a race, checked here too).
#[test]
fn e_warp_order_needs_convergence() {
    adv("e_warp_order_div", &["good", "CONV"]);
    let mut c = load("e_warp_order_div");
    // Plain accesses in CONV's shape: the PTX front reports the intra-warp race.
    let v = c.m.variants.iter_mut().find(|v| v.id == "CONV").unwrap();
    v.replace = Some(v.replace.clone().unwrap().replace(".volatile", ""));
    assert_eq!(
        classes(machine(&c, "CONV", "ptx"), "plain CONV ptx"),
        ["race"]
    );
}

/// E-DRF for tensor memory (fixtures/e_drf_tmem). Premise: races among non-scheduling-point
/// accesses are found by the vector clocks. TMEM accesses are not scheduling points; warp 0's
/// `tcgen05.st` and warp 4's `tcgen05.ld` of one column with no barrier (NOBAR) must be a race
/// on both fronts. NOWAIT (no `wait::st`) is clean by S-TCGEN05's reading (GB300 200/200
/// bitwise for all three variants). Mutant tmem-norace: TMEM write-read races unreported.
#[test]
fn e_drf_tmem_race_found() {
    adv("e_drf_tmem", &["good", "NOBAR", "NOWAIT"]);
}

/// E-ELECT on SASS (fixtures/e_elect_lane): the leader writes its lane id. GB300 elects the
/// lowest active lane (probes/elect; this fixture: `ptxv hw` 20/20 out = 0), so the SASS front's
/// one outcome must be 0; the PTX front has two (lowest and highest leader). Mutant
/// elect-high-sass (the SASS front elects the highest lane) changes the output, not the classes
/// (fixtures/e_elect B46 stays clean on SASS: the converged warp orders the leader's store).
#[test]
fn e_elect_lane_is_lowest() {
    adv("e_elect_lane", &["good"]);
    let c = load("e_elect_lane");
    let v = explore(machine(&c, "good", "sass"), 10_000);
    assert_eq!(v.outputs, vec![vec![0u8; 4]], "SASS leader lane");
}

/// S-DSMEM (fixtures/e_dsmem_rank, a 2-CTA cluster). Premise: a shared address is
/// `(cluster rank << 24) | offset`, and a `shared::cta` access names the CTA's own rank.
/// PEER stores through `shared::cta` with the peer's rank byte: a fault on both fronts (GB300:
/// illegal instruction, `ptxv hw` 2026-11-09). CLUSTER stores to the same address through
/// `shared::cluster`: a store into a CTA that may have exited (S-CLUSTER-LIFE) and an unwritten
/// own word (S-INIT): violation. Mutant dsmem-rank: the rank byte of a `shared::cta` address
/// ignored (PEER then stores into the CTA's own word and is clean).
#[test]
fn e_dsmem_foreign_rank_faults() {
    adv("e_dsmem_rank", &["good", "PEER", "CLUSTER"]);
}

/// E-CLUSTER-BAR with exited threads (fixtures/e_cluster_exit, the probe the rule's Retire
/// line asks for). Premise: `barrier.cluster` waits for the cluster's live threads only. A warp,
/// half a warp, or all of CTA 1 exits before `barrier.cluster.arrive.release` / `wait.acquire`;
/// the model completes every variant, and so does GB300 (`ptxv hw --runs 50`, 2026-11-09: no
/// hang; the words that differ from good's output are the exited threads' own stores). Mutant
/// clbar-relaxed-inits (a relaxed PTX arrive releases the thread's inits without
/// `fence.mbarrier_init`) is caught by fixtures/r1c_cluster B14.
#[test]
fn e_cluster_barrier_skips_exited_threads() {
    adv("e_cluster_exit", &["good", "EXITW", "EXITH", "EXITC"]);
}

/// S-MBAR-MEM (fixtures/e_mbar_sts64). Premise: an mbarrier initialized by ptxas's `STS.64` of
/// the encoded state word (a per-lane `mbarrier.init`, followed by `SYNCS.CCTL.IVALL`) has the
/// count and phase the word encodes. Count 2 with two arrivals per barrier is clean; count 3
/// deadlocks on both fronts, and GB300 hangs (`ptxv hw`, 2026-11-09: good 20/20 bitwise, COUNT3
/// killed by the watchdog). Mutant mbarmem-count: the decoded count one short.
#[test]
fn e_mbar_sts64_count_decoded() {
    adv("e_mbar_sts64", &["good", "COUNT3"]);
}

/// E-SETMAXNREG (probes/setmaxnreg_pool, a recorded miss). Premise: every `setmaxnreg.inc`
/// completes (the model has no register pool). inc96 asks for 4,096 registers when warpgroup 0
/// released 3,072: GB300 hangs (results.txt), the model reports clean on both fronts. Pinned:
/// a register-pool model turns inc96 into a deadlock here.
#[test]
fn e_setmaxnreg_pool_overflow_completes_in_the_model() {
    let c = load_at("probes/setmaxnreg_pool");
    for front in fronts(&c) {
        let ex = classes(machine(&c, "inc96", front), front);
        assert_eq!(ex, ["clean"], "{front}: the E-SETMAXNREG miss is fixed");
    }
}

/// E-POLL-ANY (fixtures/e_poll_any_live). Premise: a strong load commutes with the writes of
/// its location only when its step ends alike for every value, registers differing only where
/// dead. Thread 32's relaxed flag load is stored after a later strong load (live at the end of
/// the step): both outcomes must be reached (nondeterministic). DEAD (the value unused) is
/// clean, and the reduction applies there (fewer states). Mutant vip-nolive (liveness check
/// skipped) loses good's second outcome.
#[test]
fn e_poll_any_keeps_live_values() {
    adv("e_poll_any_live", &["good", "DEAD"]);
}

/// S-SCOPE (fixtures/e_scope_cta). Premise: a release / acquire pair synchronizes only when
/// each one's scope includes the other thread. Message passing between two CTAs is clean with
/// `.gpu` release and acquire and races with a `.cta` release or a `.cta` acquire, on both
/// fronts. Mutant scope-cta-all (`.cta` includes every thread).
#[test]
fn e_scope_cta_excludes_other_ctas() {
    adv("e_scope_cta", &["good", "CTAREL", "CTAACQ"]);
}

/// S-CLUSTER-LIFE (fixtures/e_cluster_life). Premise: a remote shared-memory access happens
/// before some exit of the target CTA. CTA 1's `st.shared::cluster` into CTA 0 with no final
/// cluster barrier (NOFINAL, NOBAR) is a violation on both fronts. Certificate miss
/// (2026-11-09, fixed): `certify` reported NOFINAL clean (plain remote stores were not in the
/// end-of-run check); it must now agree with exploration.
#[test]
fn e_cluster_life_remote_store() {
    adv("e_cluster_life", &["good", "NOFINAL", "NOBAR"]);
}

/// E-ALLOC (fixtures/e_tmem_alloc_block). Premise: a TMEM alloc that cannot succeed blocks
/// until columns are freed. Two warps each allocating all 512 columns complete when neither
/// holds its columns across a barrier the other reaches only after allocating; HOLD deadlocks
/// on both fronts and hangs GB300. (good's SASS verdict is a pinned false alarm in ptxas's
/// tcgen05 guardrail code; GB300 completes 50/50.) Mutant alloc-noblock.
#[test]
fn e_tmem_alloc_blocks() {
    adv("e_tmem_alloc_block", &["good", "HOLD"]);
}

/// L-OPCLASS (fixtures/e_sb_mufu, binary-only). Premise: every variable-latency opcode is in
/// the scoreboard check's set, so a read of its result without a wait on its write scoreboard is
/// reported. NOWAIT clears the wait mask of the `STG` that stores `MUFU.EX2`'s result: a
/// SCOREBOARD RAW violation (GB300: 16/50 runs wrong); NOSB removes MUFU's write scoreboard: a
/// variable-latency producer without one. Mutant opclass-nomufu (MUFU dropped from the set; the
/// RAW of NOWAIT is still found through the scoreboard bits, NOSB is not).
#[test]
fn e_sb_mufu_raw_reported() {
    adv("e_sb_mufu", &["good", "NOWAIT", "NOSB"]);
}

/// E-ALIGNED / E-BAR-EXIT with part of a warp exited before an aligned barrier
/// (fixtures/e_aligned_exit, e_aligned_exit2; soundness-lane cross-front case). PTX contract:
/// the warp does not reach the barrier in uniform control: a violation (MISS 37, PTX contract
/// only), as e_aligned_bar AB3. SASS: ptxas predicates the `BAR` (lanes guarded off at it, still
/// live): a BAR arrives per warp, so they count as arrived: clean, as GB300 (completes 400/400;
/// the two-warp counted control 200/200). e_aligned_bar AB1 (halves at two barriers) still
/// deadlocks with a violation on both fronts. Mutants aligned-sass-guarded, aligned-ptx-exit.
#[test]
fn e_aligned_exit_fronts() {
    adv("e_aligned_exit", &["good"]);
    adv("e_aligned_exit2", &["good"]);
    adv("e_aligned_bar", &["good", "AB1", "AB2", "AB3"]);
}

/// N-FADD (fixtures/e_fadd_nan). Premise: FADD / `add.rn.f32` returns the canonical NaN
/// 0x7fffffff for every NaN operand and IEEE sums otherwise. GB300 (`ptxv probe`, 2026-11-09):
/// out = [0x7fffffff x 5, 2.0, 0.0, +inf] for in = [0x7fc00001, 0xffc00002, 0x7f800001,
/// 0xff800003, 0x7fffffff, 1.0, -1.0, +inf] + 1.0; both fronts must give these bits. Mutant
/// nfadd-nan (the operand's payload kept), which no earlier test caught.
#[test]
fn e_fadd_nan_is_canonical() {
    adv("e_fadd_nan", &["good"]);
    let c = load("e_fadd_nan");
    let want: Vec<u8> = [
        0x7fff_ffffu32,
        0x7fff_ffff,
        0x7fff_ffff,
        0x7fff_ffff,
        0x7fff_ffff,
        0x4000_0000,
        0,
        0x7f80_0000,
    ]
    .iter()
    .flat_map(|w| w.to_le_bytes())
    .collect();
    for front in ["ptx", "sass"] {
        let v = explore(machine(&c, "good", front), 10_000);
        assert_eq!(v.outputs, vec![want.clone()], "{front}");
    }
}

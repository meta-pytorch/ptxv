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
                "ptx" => ptxv_ptx::parse_kernel_for(
                    &ptx,
                    Some(l.entry),
                    case.m
                        .ptxas_toolkit
                        .as_deref()
                        .and_then(ptxv_ptx::toolkit_version),
                )
                .unwrap(),
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
    let mut v = explore(m0.clone(), 2_000_000);
    assert!(v.complete, "{variant}/{front}: exploration incomplete");
    // E-TC-FENCE-NOP: as `ptxv check` reports them (races the tcgen05 thread-sync fences
    // would order are contract findings).
    let raw = v.classes();
    ptxv_sim::twolevel::tc_fence_contract(&mut v, &m0, &mut |m| explore(m, 2_000_000));
    // S-MBAR-SCOPE: races ordered once async mbarrier completions count as the mbarrier
    // CTA's are contract findings.
    ptxv_sim::twolevel::mbar_async_contract(&mut v, &m0, &mut |m| explore(m, 2_000_000));
    let tc_contract = v.classes() != raw;
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
    if tc_contract {
        // DPOR reaches the same raw verdict (before the fence-implied reclassification).
        assert_eq!(d.classes(), raw, "{variant}/{front}: DPOR verdict differs");
        return;
    }
    // A race (E-DRF): both explorers' reductions are exact for race-free kernels only, so
    // they must agree on the race, not on the outcomes it leaves (fixtures/e_elect_mid good:
    // DPOR reaches both values of the racy read, the stateful search one).
    let both_race = |v: &ptxv_sim::explore::Verdict| v.classes().iter().any(|c| c == "race");
    if (rechecked(&v) || both_race(&d)) && both_race(&v) {
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
/// Ragged `ptr_shift` tensor map: negative byte_offset, coordinates wrap back into the buffer.
#[test]
fn r0_ragged_tmap() {
    check("r0_tma_ragged", "good", "ptx");
    check("r0_tma_ragged", "good", "sass");
    check("r0_tma_ragged", "row0", "ptx");
    check("r0_tma_ragged", "row0", "sass");
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

/// S-TMA: a TMA store's clipped (out-of-bounds) box rows are not read from shared memory, so
/// never-written staging rows there are no S-INIT use; unwritten in-bounds rows still are
/// (fixtures/e_tma_store_clip, variant hole).
#[test]
fn tma_store_skips_clipped_rows() {
    for v in ["good", "fill", "hole"] {
        both("e_tma_store_clip", v);
    }
}

/// L-OPREAD: shared loads of a warp write back in issue order, so `LDS.128 R4` then `LDS R6`
/// with no wait is no WAW (fixtures/e_lds_waw, both front-ends; probes/lds_waw's ptxas
/// schedules). An ALU write in place of the second load is still a WAW (probes/lds_waw
/// cb_alu: GB300 keeps the late LDS.128's value there).
#[test]
fn lds_write_back_in_issue_order() {
    both("e_lds_waw", "good");
    let dir = root().join("probes/lds_waw");
    for c in ["bb", "cb", "cc", "cb_alu"] {
        let k = ptxv_sass::load_cubin(&dir.join(format!("{c}.cubin")), "lds_waw").unwrap();
        let waw = k
            .static_findings
            .iter()
            .any(|f| f.msg.starts_with("SCOREBOARD WAW"));
        assert_eq!(waw, c == "cb_alu", "{c}: {:#?}", k.static_findings);
    }
}

/// E-HW-L1: STG.E.NA (st.global.L1::no_allocate) keeps the storing SM's L1 copy current like a
/// plain store; with the hardware L1 model on, the reload after it returns the stored value as
/// on GB300 (probes/stg_na), on both front-ends.
#[test]
fn probe_stg_na_matches_hardware() {
    probe_bitwise("probes/stg_na", "good", "ptx");
    probe_bitwise("probes/stg_na", "good", "sass");
    let case = Case::load(root().join("probes/stg_na")).unwrap();
    let k = kernel(&case, "good", "sass");
    assert!(k.code.iter().any(|i| i.text.contains("STG.E.NA")));
}

/// E-CLOCK: 32-bit clock and global-timer reads (S2UR / CS2R.32 of SR_GLOBALTIMERLO / HI,
/// SR_CLOCKLO / HI) read as 0; a timeout-shaped delta test gives GB300's outputs on both
/// front-ends (probes/clock32). The deltas are stored (clock-dependent output): the wellsync
/// certificate refuses the kernel and the launch check is incomplete, so the one-timing output
/// comes from the exhaustive exploration.
#[test]
fn probe_clock32_matches_hardware() {
    for front in ["ptx", "sass"] {
        let case = Case::load(root().join("probes/clock32")).unwrap();
        let k = Arc::new(kernel(&case, "good", front));
        let launch = Arc::new(build_launch(&case, &k).unwrap());
        let o = ptxv_sim::wellsync::certify(Machine::new(k.clone(), launch.clone()));
        assert!(
            o.verdict.is_none() && o.reasons.iter().any(|r| r.contains("clock-derived value")),
            "{front}: {:?}",
            o.reasons
        );
        let v = explore(Machine::new(k, launch), 2_000_000);
        assert!(v.complete, "{front}");
        assert_eq!(v.classes(), ["clean"], "{front}: {:#?}", v.findings);
        let report = case.expect_report(&v.outputs[0]).unwrap();
        assert!(
            !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
            "{front}: {report:?}"
        );
    }
    let case = Case::load(root().join("probes/clock32")).unwrap();
    let k = kernel(&case, "good", "sass");
    let reads = k
        .code
        .iter()
        .filter(|i| i.text.contains("SR_GLOBALTIMER") || i.text.contains("SR_CLOCK"))
        .count();
    assert_eq!(reads, 8, "S2UR / CS2R.32 of the four registers, twice each");
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

/// N-MMA-SPECIAL: NaN bf16 operands give the canonical NaN 0x7fffffff in every D word (they
/// were "unmodeled" before probes/tc_special measured them).
#[test]
fn r1a_mma_nan_operands_give_canonical_nan() {
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
    // N-MMA-SPECIAL (probes/tc_special): a NaN operand gives 0x7fffffff (before: unmodeled).
    assert_eq!(v.classes(), vec!["clean".to_string()], "{:#?}", v.findings);
    assert_eq!(v.outputs.len(), 1);
    assert!(
        v.outputs[0]
            .chunks_exact(4)
            .all(|c| u32::from_le_bytes(c.try_into().unwrap()) == 0x7fff_ffff),
        "every D word is the canonical NaN"
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
/// One warp's PTX body for the E-ELECT taint tests (64 threads; %r1 = tid.x, %r2 = lane,
/// %rd1 = the output, `sm` 64 shared words).
fn elect_taint_machine(body: &str) -> Machine {
    let ptx = format!(
        ".version 9.0\n.target sm_103a\n.address_size 64\n\
         .visible .entry k(.param .u64 out) {{\n\
         .reg .b32 %r<32>; .reg .b64 %rd<8>; .reg .pred %p<16>;\n\
         .shared .align 4 .b32 sm[64];\n\
         ld.param.u64 %rd1, [out]; cvta.to.global.u64 %rd1, %rd1;\n\
         mov.u32 %r1, %tid.x; and.b32 %r2, %r1, 31; mov.u32 %r4, sm;\n\
         {body}\n\
         DONE:\n ret; }}"
    );
    let k = ptxv_ptx::parse_kernel(&ptx, Some("k")).unwrap();
    let case = Case {
        dir: Default::default(),
        m: serde_json::from_value(serde_json::json!({
            "name":"elect_taint", "description":"E-ELECT taint",
            "entry":"k", "grid":[1,1,1], "block":[64,1,1], "outputs":["out"],
            "buffers":[{"name":"out", "dtype":"u32", "shape":[4], "init":"zeros"}],
            "params":[{"kind":"ptr", "buffer":"out"}],
            "variants":[]
        }))
        .unwrap(),
    };
    let launch = build_launch(&case, &k).unwrap();
    Machine::new(Arc::new(k), Arc::new(launch))
}

fn elect_taint(body: &str) -> Option<u32> {
    ptxv_sim::explore::elect_leader_sensitive(&elect_taint_machine(body))
}

/// E-ELECT: the leader-sensitivity taint (`explore::elect_leader_sensitive`) on its review
/// mutants: the leader reading a lane-indexed word, a leader-written word read at lane
/// addresses, a lane-distributing load (ldmatrix) from a warp-uniform address, a lane-varying
/// synchronization, a lane read reached through a far jump out of the leader's branch (the
/// immediate post-dominator, not the branch's range), the leader lane recovered from a
/// ballot; and two kernels whose leader work is the same for every leader (two policies).
#[test]
fn e_elect_taint_mutants() {
    let sensitive = [
        // The leader loads the word its own lane wrote.
        "shl.b32 %r5, %r2, 2; add.u32 %r6, %r4, %r5; st.shared.u32 [%r6], %r2; bar.sync 0; \
         elect.sync %r8|%p1, -1; @%p1 ld.shared.u32 %r7, [%r6]; @%p1 st.global.u32 [%rd1], %r7;",
        // The leader stores its lane number; every lane reads the word back at a lane address.
        "elect.sync %r8|%p1, -1; @%p1 st.shared.u32 [%r4], %r8; bar.sync 0; \
         shl.b32 %r5, %r2, 2; add.u32 %r6, %r4, %r5; ld.shared.u32 %r7, [%r6]; \
         st.global.u32 [%rd1+4], %r7;",
        // ldmatrix from one row address: lane l gets word l % 4 of it, the leader stores its
        // own (the old taint missed it: the address is warp-uniform).
        "shl.b32 %r5, %r1, 2; add.u32 %r6, %r4, %r5; st.shared.u32 [%r6], %r1; bar.sync 0; \
         ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%r7}, [%r4]; elect.sync %r8|%p1, -1; \
         @%p1 st.global.u32 [%rd1], %r7;",
        // A lane-varying fence next to uniform leader work (a word every leader stores alike:
        // the deciding gate clears it, `sensitive[3]` below).
        "mov.u32 %r3, 7; elect.sync %r8|%p1, -1; @%p1 st.global.u32 [%rd1], %r3; \
         setp.lt.u32 %p2, %r2, 16; @%p2 membar.cta;",
        // The leader's branch target jumps on to code laid out past DONE that reads tid.
        "elect.sync %r8|%p1, -1; @%p1 bra L; bra DONE; L: bra FAR; \
         FAR: st.global.u32 [%rd1+4], %r1; bra DONE;",
        // The leader lane from a ballot of elect's p, compared with the lane.
        "elect.sync %r8|%p1, -1; vote.sync.ballot.b32 %r9, %p1, -1; bfind.u32 %r10, %r9; \
         setp.eq.u32 %p3, %r2, %r10; @%p3 st.global.u32 [%rd1], %r2;",
        // MISS 49: the leader stores its lane number. Two policies store 0 and 31 (verdict
        // nondeterministic either way: the class does not change), but a function of the lane
        // number equal at the two ends (fixtures/e_elect_lane_number) hides middle leaders.
        "elect.sync %r8|%p1, -1; @%p1 st.global.u32 [%rd1+8], %r8;",
        // MISS 49: a function of the lane number, 0 at both ends, decides the leader's work.
        "elect.sync %r8|%p1, -1; add.u32 %r3, %r8, 1; and.b32 %r3, %r3, 31; \
         min.u32 %r5, %r8, %r3; setp.eq.u32 %p2, %r5, 0; and.pred %p3, %p1, %p2; \
         @%p3 st.global.u32 [%rd1], %r3;",
        // MISS 50: %r1 (%tid.x) redefined as %tid.x + 3: its shift by 5 is not warp-uniform.
        "add.u32 %r1, %r1, 3; shr.u32 %r3, %r1, 5; elect.sync %r8|%p1, -1; \
         setp.eq.u32 %p2, %r3, 0; and.pred %p3, %p1, %p2; @%p3 st.global.u32 [%rd1], %r3;",
        // MISS 51: the ballot of elect's p in shared memory, copied to tensor memory by
        // tcgen05.cp and read back per lane: each lane tests its own bit.
        "elect.sync %r8|%p1, -1; vote.sync.ballot.b32 %r9, %p1, -1; \
         @%p1 st.shared.u32 [%r4], %r9; mov.u32 %r5, 0; cvt.u64.u32 %rd2, %r4; \
         @%p1 tcgen05.cp.cta_group::1.32x128b.warpx4 [%r5], %rd2; \
         tcgen05.ld.sync.aligned.32x32b.x1.b32 {%r7}, [%r5]; tcgen05.wait::ld.sync.aligned; \
         shr.u32 %r10, %r7, %r2; and.b32 %r10, %r10, 1; st.global.u32 [%rd1], %r10;",
        // MISS 51: the same through tcgen05.mma's shared-memory operand.
        "elect.sync %r8|%p1, -1; vote.sync.ballot.b32 %r9, %p1, -1; \
         @%p1 st.shared.u32 [%r4], %r9; mov.u32 %r5, 0; cvt.u64.u32 %rd2, %r4; \
         setp.ne.u32 %p3, %r5, 0; \
         @%p1 tcgen05.mma.cta_group::1.kind::f16 [%r5], %rd2, %rd2, %r5, %p3; \
         tcgen05.ld.sync.aligned.32x32b.x1.b32 {%r7}, [%r5]; tcgen05.wait::ld.sync.aligned; \
         shr.u32 %r10, %r7, %r2; and.b32 %r10, %r10, 1; st.global.u32 [%rd1], %r10;",
        // The leader's word in local memory, which each lane wrote with its own lane number
        // (per-lane storage).
        "mov.u64 %rd5, 0; st.local.u32 [%rd5], %r2; elect.sync %r8|%p1, -1; \
         @%p1 ld.local.u32 %r7, [%rd5]; @%p1 st.global.u32 [%rd1], %r7;",
    ];
    for (i, body) in sensitive.iter().enumerate() {
        assert!(
            elect_taint(body).is_some(),
            "mutant {i} not sensitive: {body}"
        );
    }
    // Not sensitive: the leader stores a lane-independent word; and lanes write
    // lane-dependent words, but every lane (so the leader) reads one word at a warp-uniform
    // address, the same for every leader.
    for body in [
        "mov.u32 %r3, 7; elect.sync %r8|%p1, -1; @%p1 st.global.u32 [%rd1+8], %r3;",
        "setp.eq.u32 %p2, %r2, 0; selp.u32 %r3, 1, 0, %p2; shl.b32 %r5, %r2, 2; \
         add.u32 %r6, %r4, %r5; st.shared.u32 [%r6], %r3; bar.sync 0; \
         ld.shared.u32 %r7, [%r4]; elect.sync %r8|%p1, -1; setp.ne.u32 %p3, %r7, 0; \
         and.pred %p4, %p1, %p3; @%p4 st.global.u32 [%rd1], %r7;",
    ] {
        assert_eq!(elect_taint(body), None, "{body}");
    }
    // The leader's tcgen05.mma reads whole tensor-memory tiles, also tiles each lane stored
    // its own row of: not lane-varying (the deciding gate clears it).
    let m = elect_taint_machine(
        "tcgen05.st.sync.aligned.32x32b.x1.b32 [%r4], {%r1}; tcgen05.wait::st.sync.aligned; \
         elect.sync %r8|%p1, -1; setp.ne.u32 %p3, %r4, 0; \
         @%p1 tcgen05.mma.cta_group::1.kind::f16 [%r4], %rd2, %rd3, %r5, %p3;",
    );
    assert_eq!(ptxv_sim::explore::elect_sensitivity(&m, false), None);
    // E-ELECT-2POL: the lane-varying fence passes the deciding gate (two policies explored)
    // but not the strict one: the verdict lists the assumption and a note names the rule and
    // the site.
    let m = elect_taint_machine(sensitive[3]);
    assert!(ptxv_sim::explore::elect_sensitivity(&m, false).is_none());
    let v = explore(m, 100_000);
    assert!(v.complete);
    assert!(
        v.assumptions.iter().any(|a| a == "E-ELECT-2POL"),
        "{:?}",
        v.assumptions
    );
    assert!(
        v.findings.iter().any(|f| f.kind == "note"
            && f.sig.starts_with("elect-2pol:")
            && f.msg.contains("lane-varying synchronization rule fires at")
            && f.msg.contains("membar")),
        "{:#?}",
        v.findings
    );
    // A sensitive kernel explores every leader function: with a 64-thread block both warps
    // elect, so each warp stores to its own word (the word its leader's lane % 4 selects).
    let m = elect_taint_machine(
        "shl.b32 %r5, %r1, 2; add.u32 %r6, %r4, %r5; st.shared.u32 [%r6], %r1; bar.sync 0; \
         ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%r7}, [%r4]; elect.sync %r8|%p1, -1; \
         shr.u32 %r9, %r1, 5; mul.wide.u32 %rd2, %r9, 4; add.u64 %rd3, %rd1, %rd2; \
         @%p1 st.global.u32 [%rd3], %r7;",
    );
    let v = explore(m, 200_000);
    assert!(v.complete);
    assert!(
        v.notes
            .iter()
            .any(|n| n.contains("every leader function explored")),
        "{:?}",
        v.notes
    );
    // Lane l gets word l % 4 of the row: the leader's lane % 4 selects the stored value. A
    // leader function maps the active set (the same full mask in both warps) to one lane, so
    // both warps store the same residue's word: 4 outputs (the two end lanes alone: 2).
    assert_eq!(v.distinct_outputs, 4, "{:?}", v.classes());
}

/// E-ELECT: the taint's control flow gives a guarded exit its fallthrough: the leader's store
/// of a uniform value after `@%p3 exit`, inside a lane-varying branch's region, runs under
/// lane-varying control (without the edge the store was outside the region).
#[test]
fn e_elect_taint_guarded_exit() {
    assert!(
        elect_taint(
            "setp.lt.u32 %p2, %r2, 16; elect.sync %r8|%p1, -1; @%p2 bra SKIP; \
             setp.eq.u32 %p3, %r3, 99; @%p3 exit; @%p1 st.global.u32 [%rd1], %r3; SKIP: bra DONE;"
        )
        .is_some()
    );
}

/// E-ELECT: races only a middle-lane leader causes, which the taint used to miss: a
/// lane-distributing load from a warp-uniform address (fixtures/e_elect_ldmatrix) and a store
/// after a guarded exit in a lane-varying branch's region (fixtures/e_elect_guarded_exit).
#[test]
fn e_elect_middle_leader_races() {
    both("e_elect_ldmatrix", "good");
    both("e_elect_guarded_exit", "good");
}

/// E-ELECT: races and outputs only a middle-lane leader causes, which the leader-sensitivity
/// taint cleared: a function of elect's lane number equal at the two end lanes (MISS 49,
/// fixtures/e_elect_lane_number), a shift by 5 of a register once assigned %tid.x and then
/// redefined (MISS 50, fixtures/e_elect_tid_redef), the leader's ballot copied from shared to
/// tensor memory by tcgen05.cp (MISS 51, fixtures/e_elect_tccp).
#[test]
fn e_elect_middle_leader_misses_49_50_51() {
    for v in ["good", "OUT"] {
        both("e_elect_lane_number", v);
    }
    for v in ["good", "FRESH"] {
        both("e_elect_tid_redef", v);
    }
    both("e_elect_tccp", "good");
}

/// E-ELECT (MISS 51): every operation that reads memory has a read entry in the taint's table
/// (`explore::elect_mem_spaces`, an exhaustive match). Every operation kind of the fixtures'
/// kernels (PTX, and SASS from the fixtures' binaries) is classified here, independently:
/// an unknown kind fails, so a new operation is classified twice.
#[test]
#[ignore = "heavy: lifts every fixture binary; 59 s"]
fn e_elect_mem_spaces_cover_every_op() {
    const READERS: &[&str] = &[
        "Ld",
        "Ld256",
        "LdLocal",
        "LdConst",
        "LdMatrix",
        "LdReduce",
        "Atom",
        "TmemLd",
        "Mma",
        "TcCp",
        "TmaLoad",
        "TmaStore",
        "CpAsync",
        "MbarArrive",
        "MbarTryWait",
    ];
    let name = |op: &ptxv_ir::Op| {
        let d = format!("{op:?}");
        d.split([' ', '{', '(']).next().unwrap_or("").to_string()
    };
    let mut seen = std::collections::BTreeSet::new();
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root().join("fixtures"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("case.toml").exists())
        .collect();
    dirs.sort();
    for d in dirs {
        let Ok(case) = Case::load(&d) else { continue };
        let ptx = case
            .m
            .variants
            .first()
            .and_then(|v| case.variant_ptx(&v.id).ok());
        let cubins: Vec<PathBuf> = std::fs::read_dir(&d)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "cubin"))
            .collect();
        for entry in case.m.all_launches().iter().map(|l| l.entry.to_string()) {
            let mut ks: Vec<ptxv_ir::Kernel> = cubins
                .iter()
                .filter_map(|c| ptxv_sass::load_cubin(c, &entry).ok())
                .collect();
            ks.extend(
                ptx.as_deref()
                    .and_then(|p| ptxv_ptx::parse_kernel(p, Some(&entry)).ok()),
            );
            for k in ks {
                for i in &k.code {
                    seen.insert(name(&i.op));
                    let (r, _) = ptxv_sim::explore::elect_mem_spaces(&i.op);
                    assert_eq!(
                        !r.is_empty(),
                        READERS.contains(&name(&i.op).as_str()),
                        "{}: {:?}",
                        d.display(),
                        i.op
                    );
                }
            }
        }
    }
    // The fixtures reach the readers that matter here.
    for r in [
        "Ld",
        "Ld256",
        "LdConst",
        "Atom",
        "TmemLd",
        "Mma",
        "TcCp",
        "TmaLoad",
        "MbarTryWait",
    ] {
        assert!(seen.contains(r), "no fixture kernel has {r}");
    }
}

/// E-ELECT lane classes (`elect_lanes::lane_classes`): the lanes the leader search may treat
/// as one (two 32-lane warps, the same lanes in both).
#[test]
fn e_elect_lane_classes() {
    let cls = |body: &str| ptxv_sim::explore::elect_lane_classes(&elect_taint_machine(body));
    let groups = |body: &str| {
        cls(body).map_or(32, |c| {
            let mut v = c.to_vec();
            v.sort();
            v.dedup();
            v.len()
        })
    };
    // Nothing lane-specific: one class.
    assert_eq!(
        groups("elect.sync %r8|%p1, -1; @%p1 st.global.u32 [%rd1+8], %r3;"),
        1
    );
    // Lanes 0 and 1 store their own words, the others nothing: {0}, {1}, {2..31}.
    let c = cls(
        "setp.lt.u32 %p2, %r2, 2; @!%p2 bra SKIP; shl.b32 %r5, %r2, 2; add.u32 %r6, %r4, %r5; \
         st.shared.u32 [%r6], %r2; SKIP: elect.sync %r8|%p1, -1; @%p1 st.global.u32 [%rd1], %r3;",
    )
    .unwrap();
    assert_eq!(c[..3], [0, 1, 2]);
    assert!(c[2..].iter().all(|&x| x == 2), "{c:?}");
    // A butterfly shuffle carries the pair {0, 1}'s value to both: {0, 1}, {2..31}.
    assert_eq!(
        groups(
            "setp.lt.u32 %p2, %r2, 2; selp.u32 %r5, 1, 0, %p2; \
             shfl.sync.bfly.b32 %r6, %r5, 1, 31, -1; mul.wide.u32 %rd2, %r6, 4; \
             add.u64 %rd3, %rd1, %rd2; elect.sync %r8|%p1, -1; @%p1 st.global.u32 [%rd3], %r3;"
        ),
        2
    );
    // Outside the analysis: the leader's position carried by a ballot, elect's lane number
    // read, every lane exchanging a word the leader's predicate decides (lane order: the
    // last lane's value stays); and a lane-indexed store by every lane: every lane its own
    // class.
    for body in [
        "elect.sync %r8|%p1, -1; vote.sync.ballot.b32 %r9, %p1, -1; st.global.u32 [%rd1], %r9;",
        "elect.sync %r8|%p1, -1; @%p1 st.global.u32 [%rd1], %r8;",
        "elect.sync %r8|%p1, -1; selp.u32 %r9, 1, 0, %p1; atom.global.exch.b32 %r10, [%rd1], %r9;",
        "shl.b32 %r5, %r2, 2; add.u32 %r6, %r4, %r5; st.shared.u32 [%r6], %r2; \
         elect.sync %r8|%p1, -1; @%p1 st.global.u32 [%rd1], %r3;",
        // The leader stores at an address from the ballot of elect's predicate (its own
        // lane, carried to the warp by the collective).
        "elect.sync %r8|%p1, -1; vote.sync.ballot.b32 %r9, %p1, -1; \
         mul.wide.u32 %rd2, %r9, 4; add.u64 %rd3, %rd1, %rd2; @%p1 st.global.u32 [%rd3], %r3;",
        // A shuffle from lanes the analysis does not know (an atomic's old value, lane
        // ordered): the leader's address may differ between any two lanes.
        "atom.shared.add.u32 %r7, [%r4], 1; and.b32 %r7, %r7, 31; \
         shfl.sync.idx.b32 %r9, %r2, %r7, 31, -1; mul.wide.u32 %rd2, %r9, 4; \
         add.u64 %rd3, %rd1, %rd2; elect.sync %r8|%p1, -1; @%p1 st.global.u32 [%rd3], %r3;",
    ] {
        assert_eq!(groups(body), 32, "{body}");
    }
}

/// E-ELECT lane classes against every leader function: fixtures/e_elect_mid (classes {0},
/// {1..30}, {31}) and warps whose leader class decides what is stored get the same verdict
/// and outputs from one leader per class as from all 32.
#[test]
fn e_elect_lane_classes_match_every_lane() {
    let case = Case::load(root().join("fixtures/e_elect_mid")).unwrap();
    let mut ms: Vec<(String, Machine)> = ["good", "ENDS"]
        .iter()
        .map(|v| {
            let k = Arc::new(kernel(&case, v, "ptx"));
            let launch = Arc::new(build_launch(&case, &k).unwrap());
            (v.to_string(), Machine::new(k, launch))
        })
        .collect();
    // Each warp's leader stores (to its warp's word) whether it is lane 0 / 1 or above, and
    // what lanes 0 and 1 wrote before the barrier: {0}, {1}, {2..31}.
    for body in [
        "setp.lt.u32 %p2, %r2, 2; @!%p2 bra SKIP; shl.b32 %r5, %r1, 2; add.u32 %r6, %r4, %r5; \
         st.shared.u32 [%r6], %r1; SKIP: bar.sync 0; elect.sync %r8|%p1, -1; \
         selp.u32 %r9, 7, 9, %p2; ld.shared.u32 %r10, [%r4+4]; add.u32 %r9, %r9, %r10; \
         shr.u32 %r11, %r1, 5; mul.wide.u32 %rd2, %r11, 4; add.u64 %rd3, %rd1, %rd2; \
         @%p1 st.global.u32 [%rd3], %r9;",
        "setp.lt.u32 %p2, %r2, 2; selp.u32 %r5, 1, 0, %p2; \
         shfl.sync.bfly.b32 %r6, %r5, 1, 31, -1; elect.sync %r8|%p1, -1; \
         shr.u32 %r11, %r1, 5; mul.wide.u32 %rd2, %r11, 4; add.u64 %rd3, %rd1, %rd2; \
         @%p1 st.global.u32 [%rd3], %r6;",
    ] {
        ms.push((body.to_string(), elect_taint_machine(body)));
    }
    for (v, mut m) in ms {
        m.elect_any = true;
        let cls = ptxv_sim::explore::elect_lane_classes(&m);
        assert!(cls.is_some(), "{v}: no classes");
        let mut a = m.clone();
        a.elect_lanes = cls;
        let (va, vb) = (explore(a, 2_000_000), explore(m, 2_000_000));
        assert!(va.complete && vb.complete, "{v}");
        assert_eq!(va.classes(), vb.classes(), "{v}");
        let (mut oa, mut ob) = (va.outputs.clone(), vb.outputs.clone());
        oa.sort();
        ob.sort();
        assert_eq!(oa, ob, "{v}");
        assert!(
            va.states < vb.states,
            "{v}: {} vs {} states",
            va.states,
            vb.states
        );
    }
}

/// E-WARP-ORDER: volatile accesses of two lanes on the two sides of a branch are
/// nondeterministic on both fronts; on consecutive instructions of the converged warp (CONV)
/// the PTX front reports them nondeterministic (no bar.warp.sync), SASS orders them (clean).
#[test]
fn e_warp_order_div_strong_pairs() {
    for v in ["good", "CONV"] {
        both("e_warp_order_div", v);
    }
}

/// S-SCOPE, complete overlap (PTX ISA 9.5 section 8.7): strong accesses that overlap partially
/// are not morally strong. A u8 (MIXED) or u64 (WIDE) acquire of a u32 release flag is a
/// mixed-size data race at the PTX contract and does not synchronize, so the data read after
/// it races with the data write too; a v2.u32 load whose first element is the flag (VEC) is
/// one u32 operation per element and synchronizes (fixtures/e_scope_mixed). Within a converged
/// warp, a u8 load of a byte of a u32 store on the next instruction is a race, not the
/// nondeterministic pair a u32 load is (fixtures/e_warp_order_mixed). The SASS front does not
/// decide overlap (clean). Until 2026-11-11 moral strength checked scopes only, and MIXED and
/// WIDE were clean and the warp's MIXED nondeterministic on the PTX front.
#[test]
fn scope_complete_overlap() {
    for v in ["good", "MIXED", "WIDE", "VEC"] {
        both("e_scope_mixed", v);
    }
    for v in ["good", "MIXED"] {
        both("e_warp_order_mixed", v);
    }
    // The data read races with the data write: the partially overlapping acquire does not
    // observe the release (8.9.2), so it does not order them.
    let case = Case::load(root().join("fixtures/e_scope_mixed")).unwrap();
    let k = kernel(&case, "MIXED", "ptx");
    let data_st = k
        .code
        .iter()
        .position(|i| i.text.contains("st.global.u32 [%rd1+4]"))
        .unwrap();
    let launch = build_launch(&case, &k).unwrap();
    let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 2_000_000);
    assert!(v.complete);
    let sig = format!("race:gmem:{data_st}:write-read");
    assert!(
        v.findings.iter().any(|f| f.sig == sig),
        "no {sig}: {:#?}",
        v.findings
    );
}
/// E-ELECT: a kernel correct for the end-lane leaders only is a race on the PTX front (every
/// leader function explored, also by the certificate), clean on SASS (GB300: the lowest lane).
#[test]
fn e_elect_mid_every_leader() {
    for v in ["good", "ENDS"] {
        both("e_elect_mid", v);
    }
}
/// S-NC: ld.global.nc (LDG.E.CONSTANT) of bytes written while the kernel runs, before (ordered
/// by bar.sync) or after the load, is reported on both fronts; read-only bytes and a plain load
/// are clean.
#[test]
fn e_ld_nc_written_during_kernel() {
    for v in ["good", "written", "coherent", "inplace"] {
        both("e_ld_nc", v);
    }
}
/// S-NC across launches: an .nc read of a stream-ordered earlier grid's output is exact, and so
/// is a PDL dependent's write, after griddepcontrol.wait, of bytes its primary read with .nc
/// (good); a PDL dependent's .nc read of its primary's output after the wait is reported
/// (PDLNC: the grids overlap, probes/nc_pdl), as is a dependent's write before its wait of
/// bytes the primary reads with .nc (EARLY, also a race).
#[test]
fn e_ld_nc_pdl_overlapping_launches() {
    for v in ["good", "PDLNC", "EARLY"] {
        both("e_ld_nc_pdl", v);
    }
}
/// `.uni` / `.aligned`: a guarded bra.uni with a divergent guard is a contract finding; an
/// aligned barrier.cluster run by part of a warp (divergent copies or guarded-off lanes) is a
/// violation. On the SASS front the contracts are compiled away (BRA without BSSY, WARPSYNC.ALL;
/// `expect_sass`).
#[test]
fn e_uni_aligned_contracts() {
    for v in ["good", "unidiv", "aligndiv", "alignguard"] {
        both("e_uni_aligned", v);
    }
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
/// E-BAR-CUT: a counted barrier reached by its other users only through an all-thread
/// barrier (clean), and mutants breaking each premise of the cut (each keeps its bug class).
#[test]
fn e_barcut_all_thread_barrier_cut() {
    both("e_barcut", "good");
    // As for e_barover: the stateful explorer reports each race in one order (E-DRF); DPOR
    // also reverses it. Both must find every expected class.
    for v in ["BC1", "BC2", "BC3", "BC4"] {
        for front in ["ptx", "sass"] {
            check_stateful("e_barcut", v, front, 100_000);
            let case = Case::load(root().join("fixtures").join("e_barcut")).unwrap();
            let k = kernel(&case, v, front);
            let launch = build_launch(&case, &k).unwrap();
            let d =
                ptxv_sim::dpor::explore_dpor(Machine::new(Arc::new(k), Arc::new(launch)), 100_000);
            let expect = case.variant(v).unwrap().expect_for(front);
            assert!(
                d.complete && expect.iter().all(|e| d.classes().contains(e)),
                "{v}/{front}: DPOR got {:?}, want at least {expect:?}",
                d.classes()
            );
        }
    }
}
/// E-BAR-OVER walk values: producers that leave through a role test recomputed on the path
/// (clean), and BW1, where warp 5 arrives at the consumers' counted barrier.
#[test]
fn e_barwalk_recomputed_role() {
    both("e_barwalk", "good");
    for front in ["ptx", "sass"] {
        check_stateful("e_barwalk", "BW1", front, 100_000);
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
/// probes/bar_cut_warp (GB300, 2026-11-09): one warp's lanes at two sites of one barrier id
/// complete with the other warp (same_id, same_all; on SASS same_all's slow path is
/// `BAR.SYNC R2, R2` with thread count 0: all threads). One warp waiting at two barrier ids
/// traps on GB300 (illegal instruction); the model reports a deadlock there.
#[test]
fn probe_bar_cut_warp_matches_hardware() {
    for v in [
        "split",
        "control",
        "w1_first",
        "two_counted",
        "same_id",
        "same_all",
    ] {
        check_dir("probes", "bar_cut_warp", v, "ptx");
        check_dir("probes", "bar_cut_warp", v, "sass");
    }
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
    let mut d = ptxv_sim::dpor::explore_dpor(m0.clone(), max_states);
    // S-MBAR-SCOPE, as `ptxv check` reports it.
    ptxv_sim::twolevel::mbar_async_contract(&mut d, &m0, &mut |m| {
        ptxv_sim::dpor::explore_dpor(m, max_states)
    });
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
#[ignore = "heavy: 36 s"]
fn r1b_b09_ptx() {
    check_dpor("r1b_pipeline", "B09", "ptx", 20_000);
}
#[test]
#[ignore = "heavy: 31 s"]
fn r1b_b09_sass() {
    check_dpor("r1b_pipeline", "B09", "sass", 20_000);
}
#[test]
#[ignore = "heavy: 47 s"]
fn r1b_b10_ptx() {
    check_dpor("r1b_pipeline", "B10", "ptx", 20_000);
}
#[test]
#[ignore = "heavy: 49 s"]
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
#[ignore = "heavy: 47 s"]
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
    // PTX e_mma_pair_m128: the leader's multicast commit waited at .cta scope in the peer is an
    // S-MBAR-SCOPE contract finding; every schedule still matches GB300.
    probe_bitwise_mbar_contract("fixtures/e_mma_pair_m128", "good");
    for case in ["e_mma_pair_m128", "e_mma_m128_n8", "e_ptx_redux"] {
        let case = Case::load(root().join("fixtures").join(case)).unwrap();
        for front in ["ptx", "sass"] {
            if front == "ptx" && case.m.name == "e_mma_pair_m128" {
                continue;
            }
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
/// try_cancel with `[reg+imm]` response and mbarrier operands (MISS 16: the PTX front dropped
/// the offsets, so the response landed early and query_cancel read an undefined handle).
#[test]
fn r1d_addroff() {
    check_dpor("r1d_clc", "addroff", "ptx", 2_000_000);
    check_dpor("r1d_clc", "addroff", "sass", 2_000_000);
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
#[ignore = "heavy: 22 s"]
fn r1d_b53_parity() {
    // Unbounded (the CTA keeps re-requesting): bounded search, Verdict::meets.
    check_dpor("r1d_clc", "B53", "ptx", 20_000);
    // SASS: the wrong-parity wait lets an update land late (an over-arrival), so DPOR hands
    // the search to the stateful explorer, which reaches the fault after ~20k states.
    check_dpor("r1d_clc", "B53", "sass", 200_000);
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

// E-PDL: griddepcontrol.wait blocks until the previous grid completed. AFTER sets the flag
// the primary spins on only after the wait: a deadlock (GB300 hangs). A wait that does not
// block lets the dependent set the flag and run (killed: clean instead of deadlock).
#[test]
fn e_pdl_wait_blocks() {
    for v in ["good", "AFTER"] {
        for f in ["ptx", "sass"] {
            check_dir("fixtures", "e_pdl_wait", v, f);
        }
    }
}

// S-SMEM-STRONG: an LDS waited on before the warp's next memory access is ld.acquire.cta.
// Message passing through a shared flag guarding global data: good (acquire spin) is clean on
// both fronts; RELAX (ld.relaxed spin) races on PTX and is clean on SASS, where ptxas emits
// the same LDS + wait; NOREL (no release) races on both. Without the acquire reading the
// SASS front reports good and RELAX as races. NOINIT never zeroes the flag: the spin's first
// read is undefined (S-INIT; GB300 leaves the spin on garbage), a violation on both fronts.
#[test]
fn s_smem_acquire() {
    for v in ["good", "RELAX", "NOREL", "NOINIT"] {
        for f in ["ptx", "sass"] {
            check_dir("fixtures", "s_smem_acquire", v, f);
        }
    }
}

// S-INIT-DEAD: undefined data left in a declared dead workspace is a note; through an
// output or a branch it is still a violation, and a workspace that is an output gets none.
#[test]
fn s_init_dead() {
    for v in ["good", "LEAK", "READ"] {
        for f in ["ptx", "sass"] {
            check_dir("fixtures", "s_init_dead", v, f);
        }
    }
    for f in ["ptx", "sass"] {
        check_dir("fixtures", "s_init_dead_out", "good", f);
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
// S-MULTIMEM, S-VA-ALIAS), both explorers.
#[test]
fn r1h_nvls() {
    for v in ["good", "round2", "B23", "B63", "B64", "B65", "B66"] {
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

/// S-VA-ALIAS across threads: a pure observer's unicast acquire poll of a flag another rank
/// wrote with multimem.red needs its own fence.proxy.alias (B67 has none).
#[test]
fn r1h_nvls_mp() {
    for v in ["good", "B67"] {
        for f in ["ptx", "sass"] {
            check_dir("fixtures", "r1h_nvls_mp", v, f);
        }
    }
}

/// S-VA-ALIAS within one address space: two mappings of one buffer (a second cuMemMap of
/// the same handle); weak and strong same-thread pairs and a cross-warp message pass.
#[test]
fn r1h_va_alias() {
    for v in ["good", "B68", "B69", "B74", "B75"] {
        for f in ["ptx", "sass"] {
            check_dir("fixtures", "r1h_va_alias", v, f);
        }
    }
}

/// S-VA-ALIAS, write-after-read: a write through one mapping after a read through another
/// (same thread, and across threads behind a release / acquire), a fence that must cover
/// every earlier strong access of its thread (17 of them), and a read-read pair (no
/// conflict, clean).
#[test]
fn r1h_va_war() {
    for v in ["good", "rar", "B76", "B77", "B78", "B79", "B84", "B85"] {
        for f in ["ptx", "sass"] {
            check_dir("fixtures", "r1h_va_war", v, f);
        }
    }
}

/// S-VA-ALIAS: two unordered strong additions to one counter through two mappings, then a
/// read through the second mapping ordered after both (r1h_va_red). The shadow keeps the
/// unordered strong write besides the last one; the alias check compares both. B86 (no fence
/// after the addition through the first mapping) was certified clean when only the last write
/// was compared (the certificate's run ends with the other addition).
#[test]
fn r1h_va_red() {
    let case = Case::load(root().join("fixtures/r1h_va_red")).unwrap();
    for v in ["good", "B86"] {
        for f in ["ptx", "sass"] {
            check_dir("fixtures", "r1h_va_red", v, f);
            let k = kernel(&case, v, f);
            let launch = build_launch(&case, &k).unwrap();
            let o = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)));
            if let Some(cv) = o.verdict {
                assert_eq!(
                    cv.classes(),
                    case.variant(v).unwrap().expect_for(f),
                    "{v}/{f} certified"
                );
            }
        }
    }
}

/// S-VA-ALIAS and bulk copies: the copy engine reads and writes through the mapping its
/// global address names, so a thread access through another mapping needs the alias fence,
/// and one through the same mapping does not.
#[test]
fn r1h_va_async() {
    for v in ["good", "same", "B80", "B81"] {
        for f in ["ptx", "sass"] {
            check_dir("fixtures", "r1h_va_async", v, f);
        }
    }
}

/// S-VA-ALIAS across address spaces: a peer process's mapping against the owner's. Default:
/// an `alias-xspace` finding; `race` and `one` modes; one address space makes it an alias.
#[test]
fn r1h_va_xspace() {
    for f in ["ptx", "sass"] {
        check_dir("fixtures", "r1h_va_xspace", "good", f);
        check_dir("fixtures", "r1h_va_xspace", "X1", f);
    }
    use ptxv_sim::machine::XSpace;
    for (v, f, mode, one_space, want) in [
        ("X1", "ptx", XSpace::Race, false, "race"),
        ("X1", "sass", XSpace::Race, false, "race"),
        ("X1", "ptx", XSpace::One, false, "clean"),
        ("X1", "sass", XSpace::One, false, "clean"),
        ("X1", "ptx", XSpace::Undecided, true, "race"),
        ("X1", "sass", XSpace::Undecided, true, "race"),
        ("good", "ptx", XSpace::Race, false, "clean"),
        ("good", "sass", XSpace::Race, false, "clean"),
    ] {
        let case = Case::load(root().join("fixtures/r1h_va_xspace")).unwrap();
        let k = kernel(&case, v, f);
        let mut launch = build_launch(&case, &k).unwrap();
        launch.alias_xspace = mode;
        if one_space {
            launch.spaces.clear();
        }
        let r = explore(Machine::new(Arc::new(k), Arc::new(launch)), 2_000_000);
        assert!(r.complete, "{v}/{f}: incomplete");
        assert_eq!(
            r.classes(),
            vec![want.to_string()],
            "{v}/{f} {mode:?}: {:#?}",
            r.findings
        );
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
        ("round2", "ptx", "clean"),
        ("round2", "sass", "clean"),
        ("B23", "ptx", "clean"),
        ("B23", "sass", "clean"),
        ("B66", "ptx", "clean"),
        ("B66", "sass", "clean"),
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

/// NVLS multicast and the per-SM L1 (S-VA-ALIAS, E-MC-ALIAS-HW, E-HW-L1). Strict model: both
/// fence-less pairs race. GB300 hardware profile (E-MC-ALIAS-HW on): the read-after-write pair
/// is no alias race, but the line the first weak load filled stays in L1 after the multimem
/// store (lines are per virtual address), so the second load is stale (B82); the
/// write-after-read pair is not covered by E-MC-ALIAS-HW and still races (B83).
#[test]
fn r1h_nvls_l1() {
    for v in ["good", "B82", "B83"] {
        for f in ["ptx", "sass"] {
            check_dir("fixtures", "r1h_nvls_l1", v, f);
        }
    }
    for (v, f, want) in [
        ("good", "ptx", "clean"),
        ("good", "sass", "clean"),
        ("B82", "ptx", "stale"),
        ("B82", "sass", "stale"),
        ("B83", "ptx", "race"),
        ("B83", "sass", "race"),
    ] {
        let case = Case::load(root().join("fixtures/r1h_nvls_l1")).unwrap();
        let k = kernel(&case, v, f);
        let launch = build_launch(&case, &k).unwrap();
        let launch = ptxv_sim::hwprofile::launch(&launch, ptxv_sim::hwprofile::GB300);
        let r = explore(Machine::new(Arc::new(k), Arc::new(launch)), 2_000_000);
        assert!(r.complete, "{v}/{f}: incomplete");
        assert_eq!(
            r.classes(),
            vec![want.to_string()],
            "{v}/{f}: {:#?}",
            r.findings
        );
    }
}

/// NVLS write-after-write in one thread (S-VA-ALIAS): multimem.red through the multicast
/// address, then atom.exch (B91: red.add) of the same word through the unicast address, with no
/// fence.proxy.alias. GB300 applies the red after the exchange (probes/va_alias/xpwaw.cu), so
/// the GB300 hardware profile (E-MC-ALIAS-HW, which drops read-after-write pairs only) must
/// still report the write-after-write pair, at every order and scope (B90: release.sys measured
/// in order, the alias fence is still required). `good-*`: the fence; `same-*`: one mapping.
#[test]
fn r1h_nvls_waw() {
    let ids = [
        "good",
        "good-relaxed-sys",
        "good-release-gpu",
        "good-release-sys",
        "same",
        "same-relaxed-sys",
        "same-release-gpu",
        "same-release-sys",
        "B87",
        "B88",
        "B89",
        "B90",
        "B91",
    ];
    for v in ids {
        for f in ["ptx", "sass"] {
            check_dir("fixtures", "r1h_nvls_waw", v, f);
        }
    }
    let case = Case::load(root().join("fixtures/r1h_nvls_waw")).unwrap();
    for v in ids {
        for f in ["ptx", "sass"] {
            let k = kernel(&case, v, f);
            let launch = build_launch(&case, &k).unwrap();
            let launch = ptxv_sim::hwprofile::launch(&launch, ptxv_sim::hwprofile::GB300);
            let r = explore(Machine::new(Arc::new(k), Arc::new(launch)), 2_000_000);
            assert!(r.complete, "{v}/{f}: incomplete");
            assert_eq!(
                r.classes(),
                case.variant(v).unwrap().expect_for(f),
                "{v}/{f} GB300: {:#?}",
                r.findings
            );
            if v.starts_with('B') {
                assert!(
                    r.findings.iter().any(|x| x.sig.starts_with("race:alias:")
                        && x.msg.contains(": write by")
                        && x.msg.contains("its earlier write")
                        && x.msg.contains("multicast mapping")),
                    "{v}/{f} GB300: no write-after-write alias race: {:#?}",
                    r.findings
                );
            }
        }
    }
}

/// E-HW-L1, predictive stale check: the witness fill sits between an invalidation and the
/// synchronization with the writer, with two later loads of the line after it (`cap`: loads
/// at three clocks of the reader on either side). Reduced exploration (byte footprints: the
/// fill of other bytes commutes with the store) and the certificate see the read fresh in their
/// schedule, so only the predictive check can report it; it must keep that fill. (SASS: clean,
/// the cluster barrier's CCTL.IVALL drops the line.)
#[test]
fn e_hw_l1_pred() {
    let case = Case::load(root().join("fixtures/e_hw_l1_pred")).unwrap();
    for (v, f) in [
        ("good", "ptx"),
        ("good", "sass"),
        ("cap", "ptx"),
        ("cap", "sass"),
    ] {
        check_dir("fixtures", "e_hw_l1_pred", v, f);
        let k = kernel(&case, v, f);
        let launch = build_launch(&case, &k).unwrap();
        let r = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)));
        if let Some(cv) = r.verdict {
            assert_eq!(
                cv.classes(),
                case.variant(v).unwrap().expect_for(f),
                "{v}/{f}: certified"
            );
        }
    }
}

/// E-HW-L1, predictive stale check with the fill after the read: a co-reader warp loads another
/// word of the line, unordered with the writer and the reader. DPOR's and the certificate's
/// schedules run the fill after the read, where the read's own check has no witness yet; the
/// check also looks back from each fill at the earlier reads of the line it is not ordered
/// after. (SASS good: clean, the cluster-scope acquire lowers to an L1 invalidation.)
#[test]
fn e_hw_l1_late() {
    let case = Case::load(root().join("fixtures/e_hw_l1_late")).unwrap();
    for v in ["good", "vpoll"] {
        for f in ["ptx", "sass"] {
            check_dir("fixtures", "e_hw_l1_late", v, f);
            let want = case.variant(v).unwrap().expect_for(f);
            let k = kernel(&case, v, f);
            let launch = build_launch(&case, &k).unwrap();
            let m = Machine::new(Arc::new(k), Arc::new(launch));
            let dv = ptxv_sim::dpor::explore_dpor(m.clone(), 100_000);
            assert!(dv.complete, "{v}/{f}");
            let mut dc = dv.classes();
            dc.retain(|c| c != "nondeterministic");
            assert_eq!(dc, want, "{v}/{f} DPOR: {:#?}", dv.findings);
            if let Some(cv) = ptxv_sim::wellsync::certify(m).verdict {
                assert_eq!(cv.classes(), want, "{v}/{f} certified");
            }
        }
    }
}

/// E-HW-L1: the predictive check's logs are part of the state hash with each load's clocks.
/// l1_litmus pr3-swap under the hardware model (PTX front): the reload logged before the poll
/// saw the flag is a witness for the final read, the one logged after it is not, and the
/// explorer reaches the state after the poll first on the path without the early reload.
/// With the logs hashed as (thread, pc) entries the final read's stale finding depended on
/// the worker count (one worker: never). SASS: ptxas hoists the reload out of the poll loop
/// (a loop-invariant weak load, issued once before the poll), so it races and only the final
/// read is stale.
#[test]
fn e_hw_l1_logs_hashed() {
    let case = Case::load(root().join("fixtures/l1_litmus")).unwrap();
    for (f, n) in [("ptx", 2), ("sass", 1)] {
        let k = kernel(&case, "pr3-swap", f);
        let launch = build_launch(&case, &k).unwrap();
        let m1 = Machine::new(Arc::new(k), Arc::new(launch));
        let st = Arc::new(ptxv_sim::l1::L1Stats::default());
        let m2 = ptxv_sim::twolevel::hw_machine(&m1, &st);
        for workers in [1, 2, 8] {
            let v = ptxv_sim::explore::explore_threads(m2.clone(), 100_000, workers);
            assert!(v.complete, "{f}/{workers}");
            let reads: std::collections::BTreeSet<&str> = v
                .findings
                .iter()
                .filter(|x| x.kind == "stale")
                .filter_map(|x| x.sig.split(':').nth(1))
                .collect();
            assert_eq!(reads.len(), n, "{f}/{workers}: {:#?}", v.findings);
        }
    }
}

/// Stateful explorer only, must complete (r1g: DPOR revisits too many traces here).
fn check_stateful(case: &str, variant: &str, front: &str, max_states: usize) {
    let case = Case::load(root().join("fixtures").join(case)).unwrap();
    let k = kernel(&case, variant, front);
    let launch = build_launch(&case, &k).unwrap();
    let m0 = Machine::new(Arc::new(k), Arc::new(launch));
    let mut v = explore(m0.clone(), max_states);
    // E-TC-FENCE-NOP, as `ptxv check` reports it.
    ptxv_sim::twolevel::tc_fence_contract(&mut v, &m0, &mut |m| explore(m, max_states));
    ptxv_sim::twolevel::mbar_async_contract(&mut v, &m0, &mut |m| explore(m, max_states));
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
#[ignore = "heavy: 49 s"]
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
    let mut v = explore(m0.clone(), max_states);
    ptxv_sim::twolevel::tc_fence_contract(&mut v, &m0, &mut |m| explore(m, max_states));
    ptxv_sim::twolevel::mbar_async_contract(&mut v, &m0, &mut |m| explore(m, max_states));
    let expect = case.variant(variant).unwrap().expect_for(front);
    assert!(
        v.meets(&expect),
        "{variant}/{front}: got {:?} (complete: {}), want {expect:?}",
        v.classes(),
        v.complete
    );
}

/// `check_bounded` as two tests that run in parallel, `part` 0 and 1; together they pass
/// exactly when `check_bounded` does. `check_bounded`'s verdict is the exploration's, with
/// races turned into contract findings by the contract reruns (`tc_fence_rerun`,
/// `mbar_async_rerun`) only when a rerun is complete. So when every rerun is incomplete it
/// has the exploration's classes, and when the exploration is incomplete and the expected
/// classes include neither "contract" nor "clean", meeting them after the reruns implies
/// meeting them before (the reruns only turn races into contracts).
/// - Part 0 explores. In that case it checks the exploration's verdict; otherwise it runs
///   `check_bounded`.
/// - Part 1 runs the reruns. If all are incomplete (part 0 then decides), it passes;
///   otherwise it runs `check_bounded`.
fn check_bounded_part(case: &str, variant: &str, front: &str, max_states: usize, part: u8) {
    let c = Case::load(root().join("fixtures").join(case)).unwrap();
    let expect = c.variant(variant).unwrap().expect_for(front);
    let k = kernel(&c, variant, front);
    let launch = build_launch(&c, &k).unwrap();
    let m0 = Machine::new(Arc::new(k), Arc::new(launch));
    if part == 0 {
        let v = explore(m0, max_states);
        if v.complete || expect.iter().any(|e| e == "contract" || e == "clean") {
            return check_bounded(case, variant, front, max_states);
        }
        assert!(
            v.meets(&expect),
            "{variant}/{front}: got {:?} (complete: {}), want {expect:?}",
            v.classes(),
            v.complete
        );
    } else {
        let reruns = [
            ptxv_sim::twolevel::tc_fence_rerun(&m0),
            ptxv_sim::twolevel::mbar_async_rerun(&m0),
        ];
        for m in reruns.into_iter().flatten() {
            if explore(m, max_states).complete {
                return check_bounded(case, variant, front, max_states);
            }
        }
    }
}

// Rung 2: Triton 3.5 persistent tcgen05 GEMM (docs/rung2.md). SASS is exhaustive for the good
// kernel, the fixed kernel and M3/M4; M1/M2 (missing mbarrier waits) explode and are bounded.
// One test per variant.
#[test]
fn r2_triton_sass_good() {
    check_stateful("r2_triton_gemm", "good", "sass", 2_000_000);
}
#[test]
fn r2_triton_sass_fenced() {
    check_stateful("r2_triton_gemm", "fenced", "sass", 2_000_000);
}
#[test]
#[ignore = "heavy: exhaustive SASS explore at 2M; 7 s"]
fn r2_triton_sass_m3() {
    check_stateful("r2_triton_gemm", "M3", "sass", 2_000_000);
}
#[test]
#[ignore = "heavy: exhaustive SASS explore at 2M; 7 s"]
fn r2_triton_sass_m4() {
    check_stateful("r2_triton_gemm", "M4", "sass", 2_000_000);
}
#[test]
#[ignore = "heavy: bounded SASS explore at 5k; 41 s"]
fn r2_triton_sass_m1() {
    check_bounded("r2_triton_gemm", "M1", "sass", 5_000);
}
#[test]
#[ignore = "heavy: bounded SASS explore at 5k; 30 s"]
fn r2_triton_sass_m2() {
    check_bounded("r2_triton_gemm", "M2", "sass", 5_000);
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
// PTX of CM1 / CM4, whose bug lets the pipeline run off into millions of states. One test per
// (variant, front-end), so the pieces run in parallel; CM4 on PTX is bounded (in two parts,
// below), CM1 on PTX (bounded, its deadlock needs the MMA warp to run ahead) is in
// tests/bar_cut.rs.
fn r2_cute_small(variant: &str, front: &str) {
    if vendor_missing(&["r2_cute_small"]) {
        return;
    }
    check_stateful("r2_cute_small", variant, front, 2_000_000);
}
#[test]
fn r2_cute_small_good_sass() {
    r2_cute_small("good", "sass");
}
#[test]
fn r2_cute_small_fenced_sass() {
    r2_cute_small("fenced", "sass");
}
#[test]
#[ignore = "heavy: exhaustive explore at 2M; 6 s"]
fn r2_cute_small_cm1_sass() {
    r2_cute_small("CM1", "sass");
}
#[test]
fn r2_cute_small_cm2_sass() {
    r2_cute_small("CM2", "sass");
}
#[test]
fn r2_cute_small_cm3_sass() {
    r2_cute_small("CM3", "sass");
}
#[test]
#[ignore = "heavy: exhaustive explore at 2M; 67 s"]
fn r2_cute_small_cm4_sass() {
    r2_cute_small("CM4", "sass");
}
#[test]
#[ignore = "heavy: exhaustive explore at 2M; 853 s"]
fn r2_cute_small_good_ptx() {
    r2_cute_small("good", "ptx");
}
#[test]
#[ignore = "heavy: exhaustive explore at 2M; 18 s"]
fn r2_cute_small_fenced_ptx() {
    r2_cute_small("fenced", "ptx");
}
#[test]
#[ignore = "heavy: exhaustive explore at 2M; 9 s"]
fn r2_cute_small_cm2_ptx() {
    r2_cute_small("CM2", "ptx");
}
#[test]
#[ignore = "heavy: exhaustive explore at 2M; 117 s"]
fn r2_cute_small_cm3_ptx() {
    r2_cute_small("CM3", "ptx");
}
// CM4 on PTX: bounded, in two parts (`check_bounded_part`).
#[test]
#[ignore = "heavy: bounded explore at 50k (one worker); 411 s"]
fn r2_cute_small_cm4_ptx_explore() {
    if vendor_missing(&["r2_cute_small"]) {
        return;
    }
    check_bounded_part("r2_cute_small", "CM4", "ptx", 50_000, 0);
}
#[test]
#[ignore = "heavy: bounded contract rerun at 50k (one worker); 417 s"]
fn r2_cute_small_cm4_ptx_contract_reruns() {
    if vendor_missing(&["r2_cute_small"]) {
        return;
    }
    check_bounded_part("r2_cute_small", "CM4", "ptx", 50_000, 1);
}

/// fixtures/r2_cute_gemm CM1 on SASS (exhaustive): its expectation had no test and went stale.
#[test]
#[ignore = "heavy: 46 s"]
fn r2_cute_gemm_cm1_sass() {
    if vendor_missing(&["r2_cute_gemm"]) {
        return;
    }
    check_stateful("r2_cute_gemm", "CM1", "sass", 2_000_000);
}

// Rung 3b (CuTeDSL half): MXF8 block-scaled GEMM (TMA, tcgen05.cp of the scale factors,
// tcgen05.mma.kind::mxf8f6f4.block_scale). Exhaustive verdicts; both front-ends bitwise equal
// to the recorded GB300 outputs (good and B30); B30 (scale factors of the wrong K-block)
// fails the exact MXF8 spec while the good kernel's dataflow equals it.
// One test per (variant, front).
fn r3_cute_mxf8(v: &str, front: &str) {
    if vendor_missing(&["r3_cute_mxf8"]) {
        return;
    }
    check_stateful("r3_cute_mxf8", v, front, 2_000_000);
    let case = Case::load(root().join("fixtures/r3_cute_mxf8")).unwrap();
    let k = kernel(&case, v, front);
    let launch = build_launch(&case, &k).unwrap();
    let (m, _, _) = ptxv_sim::explore::run_once(Machine::new(Arc::new(k), Arc::new(launch)));
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
#[test]
#[ignore = "heavy: explore at 2M plus 2 rebuilds and runs; 89 s"]
fn r3_cute_mxf8_good_ptx() {
    r3_cute_mxf8("good", "ptx");
}
#[test]
#[ignore = "heavy: explore at 2M plus 2 rebuilds and runs; 12 s"]
fn r3_cute_mxf8_good_sass() {
    r3_cute_mxf8("good", "sass");
}
#[test]
#[ignore = "heavy: explore at 2M plus 2 rebuilds and runs; 85 s"]
fn r3_cute_mxf8_b30_ptx() {
    r3_cute_mxf8("B30", "ptx");
}
#[test]
#[ignore = "heavy: explore at 2M plus 2 rebuilds and runs; 12 s"]
fn r3_cute_mxf8_b30_sass() {
    r3_cute_mxf8("B30", "sass");
}

/// L-OPREAD: UTCCP reads its operands in the tcgen05 issue order. fixtures/r3_cute_mxf8 built
/// by ptxas with both UTCCPs' read scoreboards removed (the operands then covered only by the
/// following UTCQMMA's rd=SB2, as in the GB300 driver's image of a DeepGEMM sm_100 GEMM): no
/// static finding, output = GB300 (1000 of 1000 runs of this mutant equal gb300_C.bin). With
/// the UTCQMMAs' read scoreboards removed too, nothing covers the UTCCP reads: a WAR finding.
#[test]
fn utccp_in_mma_issue_order() {
    if vendor_missing(&["r3_cute_mxf8"]) {
        return;
    }
    let case = Case::load(root().join("fixtures/r3_cute_mxf8")).unwrap();
    let dir = scratch_dir(&case.m.name, "utccp_nord");
    std::fs::create_dir_all(&dir).unwrap();
    let base = dir.join("base.cubin");
    ptxv_sass::ptxas(&case.dir.join("kernel.ptx"), &base, "sm_103a").unwrap();
    let mutant = |name: &str, ops: &[&str]| {
        let c = dir.join(name);
        std::fs::copy(&base, &c).unwrap();
        let lines = ptxv_sass::run_tool("nvdisasm", &["-c", c.to_str().unwrap()]).unwrap();
        let patch: Vec<ptxv_ir::CtrlPatch> = lines
            .lines()
            .filter_map(|l| {
                let (a, rest) = l.trim().strip_prefix("/*")?.split_once("*/")?;
                let op = rest.trim();
                ops.iter()
                    .any(|o| op.starts_with(o))
                    .then(|| ptxv_ir::CtrlPatch {
                        at: format!("@0x{a}"),
                        rbar: Some(7),
                        ..Default::default()
                    })
            })
            .collect();
        assert!(patch.len() >= 2, "{name}: {patch:?}");
        ptxv_sass::scoreboard::patch_file(&c, &case.m.entry, &patch).unwrap();
        ptxv_sass::load_cubin(&c, &case.m.entry).unwrap()
    };
    let k = mutant("utccp.cubin", &["UTCCP"]);
    assert!(k.static_findings.is_empty(), "{:#?}", k.static_findings);
    let launch = build_launch(&case, &k).unwrap();
    let (m, _, _) = ptxv_sim::explore::run_once(Machine::new(Arc::new(k), Arc::new(launch)));
    assert!(m.outputs() == std::fs::read(case.dir.join("gb300_C.bin")).unwrap());
    let k = mutant("utccp_mma.cubin", &["UTCCP", "UTCQMMA", "UTCBAR"]);
    assert!(
        k.static_findings
            .iter()
            .any(|f| f.msg.starts_with("SCOREBOARD WAR") && f.msg.contains("UTCCP")),
        "{:#?}",
        k.static_findings
    );
}

// The same CuTeDSL GEMM for MXF4 (e2m1 codes through 4-bit tensor maps, kind::mxf4nvf4
// block32): exhaustive verdicts; both front-ends bitwise equal to the recorded GB300 output.
#[test]
#[ignore = "heavy: 85 s"]
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
#[ignore = "heavy: 83 s"]
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
///
/// Split in `CAPTURED_PARTS` tests that run in parallel: part `i` takes every
/// `CAPTURED_PARTS`-th fixture directory (sorted by name) from the `i`-th.
fn captured_outputs_match_the_gpu(part: usize) {
    let mut n = 0;
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root().join("fixtures"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    dirs.sort();
    for (i, d) in dirs.into_iter().enumerate() {
        if i % CAPTURED_PARTS != part {
            continue;
        }
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
        // Multi-GPU captures (rung 4a) need millions of states: wellsync.rs
        // r4a_certified_first_call_matches_exploration.
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
                match ptxv_sim::wellsync::certify(m.clone()).verdict {
                    Some(v) => v,
                    // An S-MBAR-SCOPE contract finding (PTX) is a race the certificate
                    // refuses: explore and reclassify, as `ptxv check` does.
                    None if front == "ptx" => {
                        let mut v = explore(m.clone(), 200_000);
                        ptxv_sim::twolevel::mbar_async_contract(&mut v, &m, &mut |m| {
                            explore(m, 200_000)
                        });
                        v
                    }
                    None => panic!("{name}: certificate does not hold"),
                }
            } else {
                let mut v = explore(m.clone(), 200_000);
                ptxv_sim::twolevel::mbar_async_contract(&mut v, &m, &mut |m| explore(m, 200_000));
                v
            };
            assert_eq!(v.classes(), expect, "{name}/{front}: {:#?}", v.findings);
            // Runs that stop early (unmodeled), have several outputs, or none (a strict-model
            // deadlock GB300 does not have: e_bsync_parked_collective, compared under the GB300
            // profile by its own test) are not compared; the others are, clean or not
            // (r4_attn_triton: a contract violation, bitwise = GB300).
            if expect
                .iter()
                .any(|e| e == "unmodeled" || e == "nondeterministic" || e == "deadlock")
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
    assert!(n > 0, "no captured fixtures in part {part}");
}
const CAPTURED_PARTS: usize = 4;
#[test]
#[ignore = "heavy: explores over a quarter of the captured fixtures; 31 s"]
fn captured_outputs_match_the_gpu_part0() {
    captured_outputs_match_the_gpu(0);
}
#[test]
#[ignore = "heavy: explores over a quarter of the captured fixtures; 85 s"]
fn captured_outputs_match_the_gpu_part1() {
    captured_outputs_match_the_gpu(1);
}
#[test]
#[ignore = "heavy: explores over a quarter of the captured fixtures; 71 s"]
fn captured_outputs_match_the_gpu_part2() {
    captured_outputs_match_the_gpu(2);
}
#[test]
#[ignore = "heavy: explores over a quarter of the captured fixtures; 41 s"]
fn captured_outputs_match_the_gpu_part3() {
    captured_outputs_match_the_gpu(3);
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

/// S-TC-COLLECTOR (probes/tc_collector, GB300 2026-11-09, 100 runs per variant): `good`
/// (fill, use, lastuse on unchanged A) is clean; m1..m5 break the collector contract (A
/// rewritten after the fill, another descriptor, no fill, use after lastuse, use after a
/// discard) and are `contract` findings. GB300 read A from memory in every variant, which is
/// what the model computes: each variant's output equals GB300's bitwise on both front-ends.
/// m6 (model only): MMAs of two threads unordered while the collector is in use.
#[test]
fn probe_tc_collector_matches_hardware() {
    let case = Case::load(root().join("probes/tc_collector")).unwrap();
    let dir = root().join("probes/tc_collector/expected");
    for (v, want) in [
        ("good", "good"),
        ("m1", "m1"),
        ("m2", "m2"),
        ("m3", "good"),
        ("m4", "good"),
        ("m5", "m5"),
    ] {
        let want = std::fs::read(dir.join(format!("{want}.D.bin"))).unwrap();
        for f in ["ptx", "sass"] {
            check_dir("probes", "tc_collector", v, f);
            let k = kernel(&case, v, f);
            let launch = build_launch(&case, &k).unwrap();
            let r = explore(Machine::new(Arc::new(k), Arc::new(launch)), 200_000);
            assert_eq!(
                r.outputs,
                vec![want.clone()],
                "{v}/{f}: output differs from GB300"
            );
        }
    }
    // m6: thread 32's use is unordered with thread 0's fill / use / lastuse: a contract
    // finding in one run whichever order it takes (DPOR footprint space 24, the order check).
    for f in ["ptx", "sass"] {
        check_dir("probes", "tc_collector", "m6", f);
    }
}

/// %smid / %nsmid (E-SMID): probes/smid on GB300 (2026-11-09, 3 runs of 1024 CTAs): %nsmid =
/// 152, %smid in 0..=151 (all 152 seen, up to 7 CTAs per SM), unchanged across a 20 us spin.
/// fixtures/smid_slot stores to out[%smid % 4]: race-free with the CTAs on different SMs, a
/// race when they share one; the enumerated placements include that (`same`).
#[test]
fn smid_placements_reach_shared_sm() {
    let case = Case::load(root().join("fixtures/smid_slot")).unwrap();
    let words = |b: &[u8]| -> Vec<u32> {
        b.chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    for f in ["ptx", "sass"] {
        let k = kernel(&case, "good", f);
        let launch = build_launch(&case, &k).unwrap();
        let m0 = Machine::new(Arc::new(k), Arc::new(launch));
        assert!(m0.smid_used && m0.nsmid == 152, "{f}");
        // One placement (spread: SMs 0 and 1): clean.
        let v = explore(m0.clone(), 200_000);
        assert!(
            v.complete && v.findings.is_empty(),
            "{f}: {:#?}",
            v.findings
        );
        assert_eq!(v.outputs.len(), 1);
        assert_eq!(words(&v.outputs[0]), vec![0, 1, 0, 0, 152], "{f}");
        // rev: SMs 151 and 150.
        let mut m = m0.clone();
        m.smid = Arc::new(ptxv_sim::smid::placement(&m.launch, 152, "rev").unwrap());
        let v = explore(m, 200_000);
        assert_eq!(words(&v.outputs[0]), vec![0, 0, 1, 0, 152], "{f}");
        // Every placement class: `same` puts both CTAs on SM 0.
        let v = ptxv_sim::smid::explore_smid(m0.clone(), &mut |m| explore(m, 200_000));
        assert_eq!(
            v.classes(),
            case.variant("good").unwrap().expect_for(f),
            "{f}"
        );
        assert!(
            v.findings
                .iter()
                .all(|x| x.msg.contains("[smid placement: same]")),
            "{f}: {:#?}",
            v.findings
        );
        assert!(v.notes.iter().any(|n| n.starts_with("E-SMID")), "{f}");
        // %smid selects the address: the placements are a sample, the verdict incomplete.
        assert!(!v.complete, "{f}");
        assert!(
            v.notes.iter().any(|n| n.contains("smid-dependent control")),
            "{f}: {:?}",
            v.notes
        );
        // The same through the CLI's entry (machines built per placement), nested
        // enumerations running once.
        let n = std::cell::Cell::new(0);
        let v = ptxv_sim::smid::check_placements(&|| m0.clone(), &mut |mk| {
            n.set(n.get() + 1);
            ptxv_sim::l1_cores::explore_placements(mk(), &mut |m| explore(m, 200_000))
        });
        assert_eq!(n.get(), 4, "{f}");
        assert_eq!(
            v.classes(),
            case.variant("good").unwrap().expect_for(f),
            "{f}"
        );
        assert!(
            ptxv_sim::assumptions::for_kernel(&m0.k)
                .iter()
                .any(|a| a == "E-SMID"),
            "{f}"
        );
    }
}

/// E-SMID's split: %smid that reaches only data (fixtures/smid_data: stored to out[ctaid])
/// leaves the verdict complete, the placement-dependent output nondeterministic; %smid that
/// selects an address (fixtures/smid_three: two slots per SM, a third CTA on one SM spills
/// into the next SM's) makes it incomplete. smid_three races only with three CTAs on one SM
/// and one on the next, which no enumerated class has: that placement, set by hand, races.
#[test]
fn smid_data_complete_control_incomplete() {
    let run = |dir: &str, f: &str| {
        let case = Case::load(root().join("fixtures").join(dir)).unwrap();
        let k = kernel(&case, "good", f);
        let launch = build_launch(&case, &k).unwrap();
        let m0 = Machine::new(Arc::new(k), Arc::new(launch));
        let v = ptxv_sim::smid::explore_smid(m0.clone(), &mut |m| explore(m, 200_000));
        (case, m0, v)
    };
    for f in ["ptx", "sass"] {
        let (case, m0, v) = run("smid_data", f);
        assert!(ptxv_sim::smid::smid_reaches_control(&m0.k).is_none(), "{f}");
        assert!(v.complete, "{f}: {:?}", v.notes);
        assert_eq!(
            v.classes(),
            case.variant("good").unwrap().expect_for(f),
            "{f}"
        );
        let (_, m0, v) = run("smid_three", f);
        assert!(ptxv_sim::smid::smid_reaches_control(&m0.k).is_some(), "{f}");
        assert!(!v.complete, "{f}");
        assert!(
            v.findings.iter().all(|x| x.kind != "race"),
            "{f}: no class reaches the race: {:#?}",
            v.findings
        );
        let mut m = m0.clone();
        m.smid = Arc::new(vec![0, 0, 0, 1]);
        let v = explore(m, 200_000);
        assert!(
            v.classes().contains(&"race".to_string()),
            "{f}: {:#?}",
            v.findings
        );
    }
}

/// E-CLOCK clock-paced loops (clock.rs). fixtures/clock_pace's wait-until-deadline loop is a
/// pure delay loop: its deadline passes at the first check, with the clock at 0 (clean, both
/// fronts). With a store in the loop (clock_pace_store) or a loop register live after it
/// (clock_pace_live) it is not pure: its branch is a clock branch, explored both ways, and
/// after `MAX_CLOCK_ITERS` extra iterations the loop is an `unmodeled` "clock-paced loop"
/// finding. Before, clock_pace ran until the local-step cap (10M instructions per step) and
/// no check ended.
#[test]
fn clock_paced_loops() {
    both("clock_pace", "good");
    both("clock_pace_store", "good");
    both("clock_pace_live", "good");
    for (dir, pure) in [
        ("clock_pace", true),
        ("clock_pace_store", false),
        ("clock_pace_live", false),
    ] {
        let case = Case::load(root().join("fixtures").join(dir)).unwrap();
        for f in ["ptx", "sass"] {
            let k = kernel(&case, "good", f);
            let delay = ptxv_sim::clock::delay_loops(&k);
            assert_eq!(!delay.is_empty(), pure, "{dir}/{f}: pure delay loop");
            let loops = ptxv_sim::clock::clock_loops(&k);
            assert_eq!(loops.len(), 1, "{dir}/{f}");
            // Not pure: the back edge is a clock branch (both sides explored).
            assert_eq!(
                ptxv_sim::clock::clock_branches(&k, &delay) == loops,
                !pure,
                "{dir}/{f}"
            );
            assert_eq!(
                !ptxv_sim::clock::clock_stores(&k).is_empty(),
                !pure,
                "{dir}/{f}"
            );
        }
    }
}

/// E-CLOCK: a branch on the clock whose sides differ in effects is explored both ways
/// (clock.rs `clock_branches`). fixtures/e_clock_timeout: a poll whose clock64 timeout path
/// reads the data without the flag; GB300 takes it (out[0] = 0, 20 / 20 runs). good: the race,
/// outputs 0 and 7, not certified; NOTIMEOUT: clean. Before, the clock read 0 and good was
/// clean. A DeepEP-shaped timeout (print and trap: probes/printf_clock) is a hang guard and
/// stays one path.
#[test]
fn e_clock_timeout_branch_both_ways() {
    // The 64-bit clock and a 32-bit global-timer timeout (SASS: a 32-bit SR_GLOBALTIMERLO).
    for dir in ["e_clock_timeout", "e_clock_timeout32"] {
        both(dir, "good");
        both(dir, "NOTIMEOUT");
    }
    for (dir, f) in ["e_clock_timeout", "e_clock_timeout32"]
        .iter()
        .flat_map(|d| ["ptx", "sass"].map(|f| (*d, f)))
    {
        let case = Case::load(root().join("fixtures").join(dir)).unwrap();
        let k = kernel(&case, "good", f);
        assert_eq!(
            ptxv_sim::clock::clock_branches(&k, &ptxv_sim::clock::delay_loops(&k)).len(),
            1,
            "{f}"
        );
        assert!(ptxv_sim::clock::clock_effects(&k).is_empty(), "{f}");
        let launch = build_launch(&case, &k).unwrap();
        let o = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)));
        assert!(o.verdict.is_none(), "{f}: certified");
    }
    let case = Case::load(root().join("probes/printf_clock")).unwrap();
    for f in ["ptx", "sass"] {
        let k = kernel(&case, "good", f);
        assert!(
            ptxv_sim::clock::clock_branches(&k, &ptxv_sim::clock::delay_loops(&k)).is_empty(),
            "{f}: a print-and-trap timeout is a hang guard"
        );
    }
}

/// E-CLOCK through memory: fixtures/e_clock_mem's thread 0 stores %clock64, thread 32 loads
/// it after a flag and decides out[0] on it (PTX: a branch; SASS: a guarded move). The loaded
/// bytes are clock-derived (`Machine::ctaint`), so the decision is a clock choice for thread
/// 32: outputs 1 and 2 (nondeterministic), not the single output of the clock read as 0.
#[test]
fn e_clock_mem_choice() {
    both("e_clock_mem", "good");
    let case = Case::load(root().join("fixtures/e_clock_mem")).unwrap();
    for f in ["ptx", "sass"] {
        let k = kernel(&case, "good", f);
        let delay = ptxv_sim::clock::delay_loops(&k);
        assert!(!ptxv_sim::clock::clock_stores(&k).is_empty(), "{f}");
        assert!(!ptxv_sim::clock::mem_branches(&k, &delay).is_empty(), "{f}");
        let launch = build_launch(&case, &k).unwrap();
        let o = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)));
        assert!(o.verdict.is_none(), "{f}: certified");
    }
}

/// E-CLOCK clock-dependent data: fixtures/e_clock_data stores a select on a clock comparison.
/// A select is data (not explored both ways), so the launch check (`explore_placements`, as
/// `ptxv check`) marks the verdict incomplete with a "clock-dependent output" note
/// (`clock::mark_clock_outputs`): inconclusive, not clean with one output.
#[test]
fn e_clock_data_output_incomplete() {
    let case = Case::load(root().join("fixtures/e_clock_data")).unwrap();
    for f in ["ptx", "sass"] {
        let k = kernel(&case, "good", f);
        let launch = build_launch(&case, &k).unwrap();
        let m = Machine::new(Arc::new(k.clone()), Arc::new(launch));
        let v = ptxv_sim::l1_cores::explore_placements(m.clone(), &mut |m| explore(m, 200_000));
        assert_eq!(
            v.classes(),
            case.variant("good").unwrap().expect_for(f),
            "{f}"
        );
        // Never complete with one output: PTX keeps the select as data (a clock-dependent
        // output, incomplete); SASS's guarded move is a clock choice (both outputs).
        assert!(!v.complete || v.distinct_outputs > 1, "{f}");
        if f == "ptx" {
            assert!(!ptxv_sim::clock::clock_stores(&k).is_empty());
            assert!(
                v.notes.iter().any(|n| n.contains("clock-dependent output")),
                "{:?}",
                v.notes
            );
            // The CLI's entry for machines built per check, likewise.
            let v =
                ptxv_sim::smid::check_placements(&|| m.clone(), &mut |mk| explore(mk(), 200_000));
            assert!(!v.complete);
        }
    }
}

/// E-CLOCK clock taint and guarded writes: a clock-free write under a guard clears the
/// register only for reads under the same guard. Unguarded, the other path's clock value
/// still reaches the branch (a clock choice); under the same guard it does not.
#[test]
fn clock_taint_guarded_kill() {
    let body = |tail: &str| {
        format!(
            ".version 9.0\n.target sm_103a\n.address_size 64\n.visible .entry k(.param .u64 o)\n{{\n\
             .reg .pred %p<4>;\n.reg .b32 %r<4>;\n.reg .b64 %rd<8>;\n\
             ld.param.u64 %rd1, [o];\nmov.u32 %r1, %tid.x;\nsetp.eq.u32 %p1, %r1, 0;\n\
             mov.u64 %rd2, %clock64;\n@%p1 mov.u64 %rd2, 0;\n{tail}\n\
             st.global.u32 [%rd1], %r1;\nX:\nret;\n}}\n"
        )
    };
    let unguarded = body("setp.gt.u64 %p2, %rd2, 1000;\n@%p2 bra X;");
    let k = ptxv_ptx::parse_kernel(&unguarded, None).unwrap();
    assert_eq!(
        ptxv_sim::clock::clock_branches(&k, &Default::default()).len(),
        1,
        "the !p1 path's clock value reaches the branch"
    );
    let same = body("@%p1 setp.gt.u64 %p2, %rd2, 1000;\n@%p2 bra X;");
    let k = ptxv_ptx::parse_kernel(&same, None).unwrap();
    assert!(
        ptxv_sim::clock::clock_branches(&k, &Default::default()).is_empty(),
        "read under the clearing write's guard"
    );
}

/// E-CLOCK: no clock guard is left unaccounted. (1) An instruction guarded by a hang guard's
/// predicate before the trap edge (`@p mov r, 1; @p bra TRAP`) is a clock choice, and the
/// store of what it wrote is a clock-dependent store. (2) A clock value stored to a computed
/// local address taints every local load of the kernel: a branch on one is a clock choice.
#[test]
fn clock_taint_no_third_category() {
    let wrap = |decl: &str, body: &str| {
        format!(
            ".version 9.0\n.target sm_103a\n.address_size 64\n.visible .entry k(.param .u64 o)\n{{\n\
             .reg .pred %p<4>;\n.reg .b32 %r<8>;\n.reg .b64 %rd<8>;\n{decl}\n\
             ld.param.u64 %rd1, [o];\nmov.u64 %rd2, %clock64;\nmov.u64 %rd3, %clock64;\n\
             sub.s64 %rd4, %rd3, %rd2;\n{body}\nret;\n}}\n"
        )
    };
    let hang = wrap(
        "",
        "mov.u32 %r1, 0;\nsetp.gt.s64 %p2, %rd4, 1000;\n@%p2 mov.u32 %r1, 1;\n@%p2 bra TRAP;\n\
         st.global.u32 [%rd1], %r1;\nret;\nTRAP:\ntrap;",
    );
    let k = ptxv_ptx::parse_kernel(&hang, None).unwrap();
    let mov = k
        .code
        .iter()
        .position(|i| {
            i.guard.reg.is_some()
                && matches!(
                    i.op,
                    ptxv_ir::Op::Mov {
                        a: ptxv_ir::Val::Imm(1),
                        ..
                    }
                )
        })
        .unwrap() as u32;
    let delay = ptxv_sim::clock::delay_loops(&k);
    let ch = ptxv_sim::clock::clock_branches(&k, &delay);
    assert!(
        ch.contains(&mov),
        "the guarded move is a clock choice: {ch:?}"
    );
    assert_eq!(ch.len(), 1, "the hang guard itself is not");
    assert_eq!(
        ptxv_sim::clock::clock_stores(&k).len(),
        1,
        "the store of r is clock-dependent"
    );
    let local = wrap(
        ".local .align 8 .b8 buf[16];\n.reg .b64 %SPL;",
        "mov.u64 %SPL, buf;\nst.local.u64 [%SPL], %rd4;\nld.local.u64 %rd5, [%SPL+8];\n\
         setp.gt.s64 %p2, %rd5, 1000;\n@%p2 bra X;\nst.global.u32 [%rd1], 1;\nX:",
    );
    let k = ptxv_ptx::parse_kernel(&local, None).unwrap();
    let delay = ptxv_sim::clock::delay_loops(&k);
    assert_eq!(
        ptxv_sim::clock::clock_branches(&k, &delay).len(),
        1,
        "a computed local address: every slot"
    );
}

/// E-CLOCK opt-in clock step (`PTXV_CLOCK_STEP`, `Machine::clock_step`): clock_pace_live's
/// loop (not a pure delay) ends after 1000 / step clock reads on both fronts, storing the
/// last clock difference.
#[test]
fn clock_step_ends_deadline_wait() {
    let case = Case::load(root().join("fixtures/clock_pace_live")).unwrap();
    for f in ["ptx", "sass"] {
        let k = kernel(&case, "good", f);
        assert!(
            ptxv_sim::assumptions::for_kernel(&k)
                .iter()
                .any(|a| a == "E-CLOCK"),
            "{f}"
        );
        let launch = build_launch(&case, &k).unwrap();
        let mut m = Machine::new(Arc::new(k), Arc::new(launch));
        m.clock_step = 100;
        let v = explore(m, 200_000);
        assert!(
            v.complete && v.findings.iter().all(|x| x.kind == "note"),
            "{f}: {:#?}",
            v.findings
        );
        // out = {1, 2, last difference (1000) as u64}.
        let want: Vec<u8> = [1u32, 2, 1000, 0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        assert_eq!(v.outputs, vec![want], "{f}");
    }
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

/// probes/tc_handback (2-CTA TMEM hand-off, GB300 runs in probes/tc_handback/results.txt):
/// the PTX front's verdicts for the probe's variants. RCTA (the peer's .release.cta remote
/// arrive), EARLY (the peer arrives before its tcgen05.st) and NOWST (no tcgen05.wait::st:
/// fence::before_thread_sync does not release the un-waited st, PTX ISA 9.7.18.6.4.4; MISS 105,
/// before it did and NOWST was race-free) race.
#[test]
fn probe_tc_handback_ptx_verdicts() {
    // Race-free: the stateful explorer and DPOR agree on verdict and outputs.
    check_dir("probes", "tc_handback", "good", "ptx");
    // Racy (reductions are exact for race-free kernels only, E-DRF): the stateful verdict.
    let case = Case::load(root().join("probes/tc_handback")).unwrap();
    for v in ["RCTA", "EARLY", "NOWST"] {
        let k = kernel(&case, v, "ptx");
        let launch = build_launch(&case, &k).unwrap();
        let m0 = Machine::new(Arc::new(k), Arc::new(launch));
        let mut x = explore(m0.clone(), 2_000_000);
        ptxv_sim::twolevel::tc_fence_contract(&mut x, &m0, &mut |m| explore(m, 2_000_000));
        ptxv_sim::twolevel::mbar_async_contract(&mut x, &m0, &mut |m| explore(m, 2_000_000));
        let expect = case.variant(v).unwrap().expect_for("ptx");
        assert!(
            x.complete && x.meets(&expect),
            "{v}: got {:?}, want {expect:?}",
            x.classes()
        );
    }
}

/// The GB300 hardware model's classification of S-MBAR-SCOPE races on the SASS front
/// (twolevel.rs). Its memory is sequentially consistent, so an mbarrier wait acquires every
/// update it observed whatever the scopes (E-HW-L1), and MEMBAR.ALL.GPU delivers the thread's
/// in-flight remote stores (E-HW-REMOTE-MEMBAR). probes/mbar_scope on GB300: v=0 (PFC) read no
/// stale word in about 1.9e8 hand-offs, v=1 (NOF) and v=2 (PCTA) read stale words. So PFC and
/// e_tc_handback RCTA are latent, NOF and PCTA observable (the remote store is still in flight
/// at the arrive).
///
/// `stateful`: also `check_stateful(fx, v, "sass", 2_000_000)`'s check, on the same exploration
/// (the contract reruns it adds are PTX-only).
fn gb300_model_classifies(fx: &str, v: &str, want: &str, stateful: bool) {
    let case = Case::load(root().join("fixtures").join(fx)).unwrap();
    let k = kernel(&case, v, "sass");
    let launch = build_launch(&case, &k).unwrap();
    let m1 = Machine::new(Arc::new(k), Arc::new(launch));
    let mut v1 = explore(m1.clone(), 2_000_000);
    assert!(v1.complete, "{fx}/{v}: incomplete");
    if stateful {
        let expect = case.variant(v).unwrap().expect_for("sass");
        assert!(
            v1.meets(&expect),
            "{v}/sass: got {:?} (complete: {}), want {expect:?}",
            v1.classes(),
            v1.complete
        );
    }
    assert!(
        v1.classes().iter().any(|c| c == "race"),
        "{fx}/{v}: {:?}",
        v1.classes()
    );
    let c = ptxv_sim::twolevel::run(&mut v1, &m1, String::new(), &mut |m| explore(m, 2_000_000));
    assert!(c.complete, "{fx}/{v}: hardware model incomplete");
    let classes: Vec<&str> = v1
        .findings
        .iter()
        .filter(|f| f.kind == "race")
        .map(|f| {
            f.gb300
                .as_ref()
                .map_or("unclassified", |g| g.class.as_str())
        })
        .collect();
    assert!(
        !classes.is_empty() && classes.iter().all(|c| *c == want),
        "{fx}/{v}: {classes:?}, want {want}"
    );
}
#[test]
fn gb300_model_classifies_scope_races_by_probe_e_mbar_scope_pfc() {
    gb300_model_classifies("e_mbar_scope", "PFC", "latent", false);
}
#[test]
fn gb300_model_classifies_scope_races_by_probe_e_mbar_scope_nof() {
    gb300_model_classifies("e_mbar_scope", "NOF", "observable", false);
}
#[test]
fn gb300_model_classifies_scope_races_by_probe_e_mbar_scope_pcta() {
    gb300_model_classifies("e_mbar_scope", "PCTA", "observable", false);
}
#[test]
#[ignore = "heavy: explore at 2M plus the hardware-model explore; 232 s"]
fn gb300_model_classifies_scope_races_by_probe_e_tc_handback_rcta() {
    gb300_model_classifies("e_tc_handback", "RCTA", "latent", true);
}

/// S-MBAR-SCOPE, S-TCGEN05 (fixtures/e_tc_handback): the peer CTA hands the pair's TMEM
/// accumulator back to the leader with a remote arrive between two rounds of 2-CTA MMAs. A
/// SASS PHASECHK (no CCTL.IVALL) acquires the peer's cluster-scope release for SM-local memory
/// only, tensor memory included, and round 2's MMA must see that happens-before at its TMEM
/// accesses (`tmem_block`, issuer's `AsyncOp::sh`): ACTA is clean on SASS (it raced before),
/// a race on PTX (.acquire.cta wait); RCTA (.release.cta arrive) races on both fronts.
// One test per (variant, front). (good on SASS is the same exploration as ACTA's, with a
// CCTL.IVALL: left to `ptxv check`.)
#[test]
fn e_tc_handback_mma_sees_sm_local_acquire_good_ptx() {
    check_stateful("e_tc_handback", "good", "ptx", 2_000_000);
}
#[test]
fn e_tc_handback_mma_sees_sm_local_acquire_rcta_ptx() {
    check_stateful("e_tc_handback", "RCTA", "ptx", 2_000_000);
}
#[test]
fn e_tc_handback_mma_sees_sm_local_acquire_acta_ptx() {
    check_stateful("e_tc_handback", "ACTA", "ptx", 2_000_000);
}
// RCTA on SASS: gb300_model_classifies_scope_races_by_probe_e_tc_handback_rcta, which checks
// the same exploration (`check_stateful`'s check) before classifying its races.
#[test]
#[ignore = "heavy: explore at 2M with contract reruns; 56 s"]
fn e_tc_handback_mma_sees_sm_local_acquire_acta_sass() {
    check_stateful("e_tc_handback", "ACTA", "sass", 2_000_000);
}
// The join must not hide a real hand-back race. ACTANW (no wait::ld before the hand-back): a
// race and an undefined-register violation on PTX; the SASS is still ordered (ptxas makes the
// stores of the loaded registers wait on the LDTM scoreboard), clean. ACTAE (the hand-back
// arrive moved ahead of the peer's reads): a race on both fronts.
#[test]
fn e_tc_handback_acta_nowait_ptx() {
    check_stateful("e_tc_handback", "ACTANW", "ptx", 2_000_000);
}
#[test]
fn e_tc_handback_acta_early_handback_ptx() {
    check_stateful("e_tc_handback", "ACTAE", "ptx", 2_000_000);
}
#[test]
#[ignore = "heavy: explore at 2M with contract reruns; about 60 s"]
fn e_tc_handback_acta_nowait_sass() {
    check_stateful("e_tc_handback", "ACTANW", "sass", 2_000_000);
}
#[test]
#[ignore = "heavy: explore to the 2M cap; about 280 s"]
fn e_tc_handback_acta_early_handback_sass() {
    // The race is found before the cap; the search does not complete, which a race verdict
    // does not need.
    let case = Case::load(root().join("fixtures/e_tc_handback")).unwrap();
    let k = kernel(&case, "ACTAE", "sass");
    let launch = build_launch(&case, &k).unwrap();
    let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 2_000_000);
    assert!(
        v.findings
            .iter()
            .any(|f| f.kind == "race" && f.sig.starts_with("race:tmem:")),
        "ACTAE/sass: no TMEM race: {:?}",
        v.classes()
    );
}

/// MISS 105, PTX front (fixtures/e_tblk_late): `tcgen05.fence::before_thread_sync` released
/// the thread's whole tcgen05 view (`FenceKind::TcBefore`), un-waited `tcgen05.ld` reads
/// included, so the bar.sync and the hand-back arrive carried the round-1 reads and the PTX
/// front reported no race between them and round 2's MMA (`good`: contract only). PTX ISA
/// 9.7.18.6.4.4, "Non-pipelined instructions, different thread": "the producer threads that
/// issue the asynchronous tcgen05 instructions must explicitly wait for the instructions'
/// completion before synchronizing with the consumer threads" (example: ld; wait::ld;
/// fence::before_thread_sync; arrive); 9.7.18.8.5: "tcgen05.wait::ld causes the executing
/// thread to block until all prior tcgen05.ld operations issued by the executing thread have
/// completed". Now the fence does not release a load issued since the thread's last wait::ld
/// (`tc_released`): `good` races on PTX, with and without the E-TC-FENCE-NOP rerun. (ACTA
/// raced already, for its .acquire.cta wait.)
#[test]
fn e_tblk_late_ptx_unwaited_ld_is_not_released() {
    let case = Case::load(root().join("fixtures/e_tblk_late")).unwrap();
    for v in ["good", "ACTA"] {
        let k = kernel(&case, v, "ptx");
        let launch = build_launch(&case, &k).unwrap();
        let m0 = Machine::new(Arc::new(k), Arc::new(launch));
        let mut x = explore(m0.clone(), 2_000_000);
        ptxv_sim::twolevel::tc_fence_contract(&mut x, &m0, &mut |m| explore(m, 2_000_000));
        ptxv_sim::twolevel::mbar_async_contract(&mut x, &m0, &mut |m| explore(m, 2_000_000));
        assert!(x.complete, "{v}/ptx: incomplete");
        assert_eq!(x.classes(), ["contract", "race"], "{v}/ptx");
        assert!(
            x.findings.iter().any(|f| f.kind == "race"
                && f.sig.starts_with("race:tmem:")
                && f.msg.contains("tcgen05.ld")),
            "{v}/ptx: no TMEM race with a tcgen05.ld: {:?}",
            x.findings.iter().map(|f| &f.sig).collect::<Vec<_>>()
        );
    }
}

/// S-TCGEN05 (fixtures/e_tblk_late): the round-1 tcgen05.wait::ld moved after the 2-CTA
/// hand-back. The SASS front completes an LDTM's tensor-memory read only when its write
/// scoreboard is waited (`ldtm_issue` / `ldtm_complete`); ptxas waits for it after the bar.sync
/// that precedes the arrive, so round 2's MMA races with the other warps' reads, with either
/// wait scope (before: clean for ACTA).
#[test]
#[ignore = "heavy: explore to the 700k cap twice"]
fn e_tblk_late_sass_races_on_tmem() {
    let case = Case::load(root().join("fixtures/e_tblk_late")).unwrap();
    for v in ["good", "ACTA"] {
        let k = kernel(&case, v, "sass");
        let launch = build_launch(&case, &k).unwrap();
        let x = explore(Machine::new(Arc::new(k), Arc::new(launch)), 700_000);
        assert!(
            x.findings
                .iter()
                .any(|f| f.kind == "race" && f.sig.starts_with("race:tmem:")),
            "{v}/sass: no TMEM race: {:?}",
            x.classes()
        );
    }
}

/// Actor ids (fixtures/e_actor_base): 4096 threads, the last one's bulk copy into shared
/// memory. With a fourth fixed actor per thread (MISS 104's second tcgen05 pipe), every decoder
/// of an actor id (the CTA of an access, the thread of an actor, the issuer of an async
/// operation, the E-CTA-REP position) must use `async_base`; one still at 3 n indexed past the
/// last thread (a panic) or took another thread's CTA.
#[test]
fn e_actor_base_last_thread_async_actor() {
    both("e_actor_base", "good");
}

/// S-TCGEN05 (fixtures/e_sttm_late): a tcgen05.st of D before the 2-CTA hand-back's sync. With
/// tcgen05.wait::st (STW; SASS `FENCE.VIEW.ASYNC.T`) the store is ordered before round 2's MMA;
/// without it (STNW) it races, on PTX (MISS 105: fence::before_thread_sync does not release an
/// un-waited st) and on SASS (MISS 107: an STTM's TMEM write completes only at the thread's
/// `FENCE.VIEW.ASYNC.T`; before, at issue, and STNW was clean).
#[test]
fn e_sttm_late_ptx() {
    for v in ["good", "STW", "STNW"] {
        check_stateful("e_sttm_late", v, "ptx", 2_000_000);
    }
}
#[test]
#[ignore = "heavy: three SASS explorations of about 520k states; about 180 s"]
fn e_sttm_late_sass() {
    for v in ["good", "STW", "STNW"] {
        check_stateful("e_sttm_late", v, "sass", 2_000_000);
    }
    let case = Case::load(root().join("fixtures/e_sttm_late")).unwrap();
    let k = kernel(&case, "STNW", "sass");
    let launch = build_launch(&case, &k).unwrap();
    let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 2_000_000);
    assert!(
        v.findings
            .iter()
            .any(|f| f.kind == "race" && f.msg.contains("STTM writes")),
        "STNW/sass: no race with the STTM: {:?}",
        v.findings.iter().map(|f| &f.sig).collect::<Vec<_>>()
    );
}

/// MISS 111 (fixtures/e_dealloc_unwaited): a thread's own tcgen05.ld / st is complete only at
/// wait::ld / wait::st (PTX ISA 9.7.18.6.1: asynchronous, "not inherently ordered with respect
/// to other tcgen05 operations in the same thread"), so its later tcgen05.dealloc with no wait
/// between races with it: LDNW (ld; dealloc; wait::ld) and STNW (st; dealloc). The PTX front
/// checked the dealloc on the thread's tcgen05.ld/st actor, whose clock covered them: no race
/// for lane 0's own access. On SASS (MISS 103 / 107: LDTM and STTM complete at their waits)
/// STNW races; LDNW is ordered as compiled (ptxas waits on the LDTM's scoreboard first).
#[test]
fn e_dealloc_unwaited_races() {
    both("e_dealloc_unwaited", "good");
    let case = Case::load(root().join("fixtures/e_dealloc_unwaited")).unwrap();
    // LDNW on SASS is clean: ptxas waits on the LDTM's scoreboard before the dealloc.
    check_stateful("e_dealloc_unwaited", "LDNW", "sass", 100_000);
    for (v, f) in [("LDNW", "ptx"), ("STNW", "ptx"), ("STNW", "sass")] {
        {
            check_stateful("e_dealloc_unwaited", v, f, 100_000);
            // Lane 0's own access against its own dealloc (MISS 111), not only the other
            // lanes' (MISS 105 / 103 / 107).
            let k = kernel(&case, v, f);
            let launch = build_launch(&case, &k).unwrap();
            let x = explore(Machine::new(Arc::new(k), Arc::new(launch)), 100_000);
            assert!(
                x.findings.iter().any(|g| g.kind == "race"
                    && g.msg.contains("deallocation by t0")
                    && (g.msg.contains(" t0's tcgen05.ld/st") || g.msg.contains(" t0's STTM"))),
                "{v}/{f}: no race between t0's own access and its dealloc: {:?}",
                x.findings.iter().map(|g| &g.msg).collect::<Vec<_>>()
            );
        }
    }
}

/// E-TMEM-QUEUE (fixtures/e_tmem_queue, probes/tmem_queue): a thread's SASS STTM after its own
/// LDTM of the same column with no wait on the LDTM's scoreboard (Q), and two STTMs of one
/// thread to the same column with no FENCE.VIEW.ASYNC.T between (QQ), are ordered only by the
/// thread's TMEM queue: `tmem-queue` contract findings at the strict level and under the GB300
/// profile (GB300 measured no reordering, without a discriminating control: not discharged).
/// With the wait (good): clean. PTX: the same contract findings (MISS 116; before, clean).
#[test]
fn e_tmem_queue_is_a_contract() {
    for v in ["good", "Q", "QQ"] {
        both("e_tmem_queue", v);
    }
    assert_eq!(sass_gb300("e_tmem_queue", "Q"), "contract");
    assert_eq!(sass_gb300("e_tmem_queue", "QQ"), "contract");
    // MISS 121: a load after the thread's own un-waited store races on both fronts, under the
    // GB300 profile too (mode 2 had no positive control).
    both("e_tmem_queue", "SL");
    assert!(sass_gb300("e_tmem_queue", "SL").contains("race"));
}

/// E-TMEM-QUEUE (fixtures/e_queue_mask): a store's own thread's pending LDTM does not hide
/// another thread's in-flight LDTM racing with the store. Every unordered reader is examined:
/// the finding is a tmem-queue contract only when all are the thread's own pending LDTMs. S0
/// (the own LDTM sorts first) was a contract on SASS, clean under the GB300 profile; S4 raced.
#[test]
fn e_queue_mask_own_pending_ldtm_does_not_hide_a_racer() {
    for v in ["S0", "S4"] {
        both("e_queue_mask", v);
        let g = sass_gb300("e_queue_mask", v);
        assert!(g.split(',').any(|c| c == "race"), "{v}: {g}");
    }
}

/// MISS 122 (fixtures/e_rerun_drop_ptx): a TMEM store with two racing readers, one ordered once
/// the tcgen05 fences are implied (tc_fence_contract reclassifies it), one not. Every racer is
/// reported, so the second survives the rerun: XLOW (its racer sorted second) and YLOW race.
#[test]
fn e_rerun_drop_reports_every_tmem_racer() {
    for v in ["XLOW", "YLOW"] {
        both("e_rerun_drop_ptx", v);
    }
}

/// MISS 124 (fixtures/e_rerun_drop_proxy_mb, e_twolevel_proxy_first): an async-proxy read
/// reports every uncovered generic writer of its bytes, not only the first (the lower byte):
/// mbar_async_contract reclassifies X's proxy race (ordered once the complete-tx counts at the
/// mbarrier's CTA) and Y's stays a race; the GB300 model classifies Y's proxy race, which is
/// observable (before: 0 observable, 2 latent on PTX).
#[test]
fn proxy_check_reports_every_uncovered_writer() {
    for v in ["XLOW", "YLOW"] {
        // PTX only: on SASS this kernel's DPOR and full explorations disagree (a violation the
        // DPOR run misses, as on int7 before the fix; reported with MISS 124).
        check("e_rerun_drop_proxy_mb", v, "ptx");
        both("e_twolevel_proxy_first", v);
        let case = Case::load(root().join("fixtures/e_twolevel_proxy_first")).unwrap();
        let k = kernel(&case, v, "ptx");
        let launch = build_launch(&case, &k).unwrap();
        let m1 = Machine::new(Arc::new(k), Arc::new(launch));
        let mut v1 = explore(m1.clone(), 2_000_000);
        let proxy = v1
            .findings
            .iter()
            .filter(|f| f.sig.starts_with("race:proxy:"))
            .count();
        assert_eq!(proxy, 2, "{v}: X's and Y's proxy races");
        let c =
            ptxv_sim::twolevel::run(&mut v1, &m1, String::new(), &mut |m| explore(m, 2_000_000));
        assert!(
            c.complete && c.observable >= 1,
            "{v}: Y's proxy race is observable"
        );
    }
}

/// One .cta_group per kernel (fixtures/e_cta_group_mix): a kernel whose tcgen05 instructions
/// name both groups is a violation (PTX ISA, tcgen05; ptxas refuses it, so PTX only), and a
/// commit tracks only its own group's operations (C2: the cta_group::2 commit after
/// cta_group::1 MMAs orders none of them, a race).
#[test]
fn e_cta_group_mix_is_a_violation() {
    both("e_cta_group_mix", "good");
    for v in ["C2", "D2"] {
        check_stateful("e_cta_group_mix", v, "ptx", 2_000_000);
    }
    let case = Case::load(root().join("fixtures/e_cta_group_mix")).unwrap();
    let k = kernel(&case, "D2", "ptx");
    let launch = build_launch(&case, &k).unwrap();
    let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 2_000_000);
    assert!(
        v.findings
            .iter()
            .any(|f| f.sig.starts_with("cta-group-mix:")),
        "D2: {:?}",
        v.classes()
    );
}

/// S-SAME-VALUE through a polled ring (fixtures/e_ring_warp W1): both consumer warps copy every
/// slot, so each output word gets two unordered plain writes, but both read the slot after the
/// same tail and store one value: clean with same-value notes on both front-ends, and one
/// output. (The certificate refuses it: wellsync `ring_mutants_are_refused_with_their_reason`.)
#[test]
fn e_ring_warp_w1_same_value_is_clean() {
    both("e_ring_warp", "W1");
    let case = Case::load(root().join("fixtures/e_ring_warp")).unwrap();
    for f in ["ptx", "sass"] {
        let k = kernel(&case, "W1", f);
        let launch = build_launch(&case, &k).unwrap();
        let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 200_000);
        assert!(v.complete, "W1/{f}: incomplete");
        assert_eq!(v.classes(), ["clean"], "W1/{f}");
        assert!(
            v.findings
                .iter()
                .any(|x| x.sig.starts_with("same-value:gmem")),
            "W1/{f}: no same-value note: {:?}",
            v.findings.iter().map(|x| &x.sig).collect::<Vec<_>>()
        );
        assert_eq!(v.outputs.len(), 1, "W1/{f}");
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
#[ignore = "heavy: 77 s"]
fn probe_proxy_fence_matches_hardware() {
    for v in ["good", "after_all", "issuer", "none", "global"] {
        check_dir("probes", "proxy_fence", v, "ptx");
        check_dir("probes", "proxy_fence", v, "sass");
    }
}

/// E-CPASYNC-PROXY: cp.async writes then a TMA store with no fence.proxy.async is a contract
/// finding, st.shared writers a proxy race (probes/proxy_cpasync; GB300 1000/1000 correct in
/// every variant, st_none included: no positive control, so not evidence, not even a measured
/// null). One test per (variant, front-end).
#[test]
#[ignore = "heavy: stateful + DPOR check at 2M; 7 s"]
fn probe_proxy_cpasync_verdicts_good_ptx() {
    check_dir("probes", "proxy_cpasync", "good", "ptx");
}
#[test]
#[ignore = "heavy: stateful + DPOR check at 2M; 18 s"]
fn probe_proxy_cpasync_verdicts_good_sass() {
    check_dir("probes", "proxy_cpasync", "good", "sass");
}
#[test]
#[ignore = "heavy: stateful + DPOR check at 2M; 379 s"]
fn probe_proxy_cpasync_verdicts_none_ptx() {
    check_dir("probes", "proxy_cpasync", "none", "ptx");
}
#[test]
#[ignore = "heavy: stateful + DPOR check at 2M; 506 s"]
fn probe_proxy_cpasync_verdicts_none_sass() {
    check_dir("probes", "proxy_cpasync", "none", "sass");
}
#[test]
#[ignore = "heavy: stateful + DPOR check at 2M; 100 s"]
fn probe_proxy_cpasync_verdicts_st_none_ptx() {
    check_dir("probes", "proxy_cpasync", "st_none", "ptx");
}
#[test]
#[ignore = "heavy: stateful + DPOR check at 2M; 128 s"]
fn probe_proxy_cpasync_verdicts_st_none_sass() {
    check_dir("probes", "proxy_cpasync", "st_none", "sass");
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
    probe_bitwise_walks(dir, variant, front, None)
}

/// `probe_bitwise` for a probe whose kernel overwrites TMEM words of a thread with no wait
/// between the stores (a sentinel fill, then the stores under test): every finding is a
/// strict-level `tmem-queue` contract (E-TMEM-QUEUE, not assumed), and it stays one under the
/// GB300 profile; the outputs equal GB300's.
fn probe_bitwise_queue(dir: &str, variant: &str, front: &str) {
    let case = Case::load(root().join(dir)).unwrap();
    let k = kernel(&case, variant, front);
    let launch = build_launch(&case, &k).unwrap();
    let m = Machine::new(Arc::new(k), Arc::new(launch));
    let mut v = ptxv_sim::wellsync::certify(m.clone())
        .verdict
        .unwrap_or_else(|| panic!("{dir} {variant} {front} certifies"));
    assert_eq!(v.classes(), ["contract"], "{dir} {front}: {:#?}", v.findings);
    assert!(
        v.findings.iter().all(|f| f.sig.starts_with("tmem-queue:")),
        "{dir} {front}: {:#?}",
        v.findings
    );
    ptxv_sim::hwprofile::discharge(&mut v, &m, ptxv_sim::hwprofile::GB300);
    assert_eq!(v.classes(), ["contract"], "{dir} {front} under GB300: {:#?}", v.findings);
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
        "{dir} {variant} {front}: {report:?}"
    );
}

/// `probe_bitwise` with `walks` cross-check walks (None: PTXV_CERT_CROSSCHECK's).
fn probe_bitwise_walks(dir: &str, variant: &str, front: &str, walks: Option<usize>) {
    let case = Case::load(root().join(dir)).unwrap();
    let k = kernel(&case, variant, front);
    let launch = build_launch(&case, &k).unwrap();
    let m = Machine::new(Arc::new(k), Arc::new(launch));
    let o = match walks {
        Some(n) => ptxv_sim::wellsync::certify_crosschecked(m, n),
        None => ptxv_sim::wellsync::certify(m),
    };
    let v = o
        .verdict
        .unwrap_or_else(|| panic!("{dir} {variant} {front} certifies: {:#?}", o.reasons));
    assert_eq!(v.classes(), ["clean"], "{dir} {front}: {:#?}", v.findings);
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
        "{dir} {variant} {front}: {report:?}"
    );
}

/// A probe's PTX variant whose only findings are S-MBAR-SCOPE contract ones (an async unit's
/// complete-tx from the peer CTA waited at .cta scope): every schedule's output equals the
/// GB300 bytes in its `expected/`.
fn probe_bitwise_mbar_contract(dir: &str, variant: &str) {
    let case = Case::load(root().join(dir)).unwrap();
    let k = kernel(&case, variant, "ptx");
    let launch = build_launch(&case, &k).unwrap();
    let m0 = Machine::new(Arc::new(k), Arc::new(launch));
    let mut v = explore(m0.clone(), 2_000_000);
    assert!(v.complete, "{dir} {variant}: incomplete");
    ptxv_sim::twolevel::mbar_async_contract(&mut v, &m0, &mut |m| explore(m, 2_000_000));
    assert_eq!(v.classes(), ["contract"], "{dir}: {:#?}", v.findings);
    assert!(!v.outputs.is_empty());
    for o in &v.outputs {
        let report = case.expect_report(o).unwrap();
        assert!(
            !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
            "{dir} {variant}: {report:?}"
        );
    }
}

/// N-MMA-SPECIAL: tcgen05.mma.kind::f16 (f32 D) on subnormal / Inf / NaN f16 and bf16 operands
/// and on special accumulators (probes/tc_special, 17 input sets x 8,192 outputs) = GB300
/// (2026-11-09, 50 scrubbed runs per set), both front-ends bitwise. Before, every such MMA was
/// "unmodeled".
#[test]
fn probe_tc_special_matches_hardware() {
    let dir = root().join("probes/tc_special");
    let mut sets: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("case.toml").exists())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    sets.sort();
    assert_eq!(sets.len(), 17, "{sets:?}");
    for s in sets {
        for f in ["ptx", "sass"] {
            probe_bitwise(&format!("probes/tc_special/{s}"), "good", f);
        }
    }
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
/// instructions enter together (E-COLLECTIVE). SASS without the certificate cross-check: a
/// random schedule of the model deadlocks there (MISS 52, open; pinned by tests/wellsync.rs
/// `cert_crosscheck_deadlocking_schedule_refutes`).
#[test]
fn probe_warpsync_pairs_verdicts() {
    probe_bitwise("probes/warpsync_pairs", "good", "ptx");
    probe_bitwise_walks("probes/warpsync_pairs", "good", "sass", Some(0));
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

/// PTX add.sat.s32 / sub.sat.s32 (probes/int_sat: INT_MIN / INT_MAX edges, overflowing pairs,
/// immediates) = GB300, both fronts (ptxas: IADD, PLOP3 sign tests, SEL).
#[test]
fn probe_int_sat_matches_hardware() {
    probe_bitwise("probes/int_sat", "good", "ptx");
    probe_bitwise("probes/int_sat", "good", "sass");
}

/// MISS 23: `R.SIGN` / `UR.SIGN` (PLOP3's sign-bit inputs in ptxas's add.sat.s32) are predicate
/// inputs only; as a predicate destination they are refused (before, pdst took the register).
#[test]
fn sass_sign_predicate_not_a_destination() {
    for (line, ok) in [
        (
            "PLOP3.LUT P4, PT, R2.SIGN, UR6.SIGN, R9.reuse.SIGN, 0x2, 0x0",
            true,
        ),
        ("ISETP.GE.U32.AND P1, PT, R0, 0x10000, PT", true),
        ("ISETP.GE.U32.AND R2.SIGN, PT, R0, 0x10000, PT", false),
        (
            "PLOP3.LUT UR6.SIGN, PT, R2.SIGN, R3.SIGN, PT, 0x2, 0x0",
            false,
        ),
        (
            "PLOP3.LUT P4, R3.SIGN, R2.SIGN, R3.SIGN, PT, 0x2, 0x0",
            false,
        ),
    ] {
        assert_eq!(ptxv_sass::lower_text(line, None).is_ok(), ok, "{line}");
    }
}

/// PTX sub.u16x2 / sub.s16x2 (probes/sub16x2; ptxas 13.3 only: VIADD.16x2 d, a, -b): the SASS
/// front on the committed 13.3 cubin = GB300; the PTX front refuses the forms (not in the PTX
/// ISA; measured-only semantics stay off the contract front, MISS 23).
#[test]
fn probe_sub16x2_matches_hardware() {
    probe_bitwise_cubin("probes/sub16x2", Some("kernel.cubin"));
    let case = Case::load(root().join("probes/sub16x2")).unwrap();
    let err = ptxv_ptx::parse_kernel(&case.variant_ptx("good").unwrap(), Some(&case.m.entry))
        .unwrap_err();
    assert!(
        format!("{err:#}").contains("sub.u16x2 / sub.s16x2"),
        "{err:#}"
    );
}

/// PTX mul.wide.u16 / .s16 on 16-bit registers from ld.s16 / ld.u16 / cvt / add.s16 and
/// immediates (probes/mulwide16) = GB300, both fronts.
#[test]
fn probe_mulwide16_matches_hardware() {
    probe_bitwise("probes/mulwide16", "good", "ptx");
    probe_bitwise("probes/mulwide16", "good", "sass");
}

/// bfe.u64 / .s64 (probes/bfe64): GB300 runs ptxas's code, which takes pos and len as full
/// 32-bit values; the PTX ISA takes them mod 256. SASS front = GB300 bitwise; the PTX front
/// follows the ISA, differs from GB300 in exactly the 42 words whose pos or len reach 256,
/// and notes the divergence (bfe64-isa-vs-gb300).
#[test]
fn probe_bfe64_matches_hardware() {
    probe_bitwise("probes/bfe64", "good", "sass");
    let case = Case::load(root().join("probes/bfe64")).unwrap();
    let k = kernel(&case, "good", "ptx");
    let launch = build_launch(&case, &k).unwrap();
    let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
        .verdict
        .expect("probes/bfe64 ptx certifies");
    assert_eq!(v.classes(), ["clean"], "{:#?}", v.findings);
    assert!(
        v.findings
            .iter()
            .any(|f| f.kind == "note" && f.sig.starts_with("bfe64-isa-vs-gb300:")),
        "{:#?}",
        v.findings
    );
    let gpu = std::fs::read(case.dir.join("expected/out.bin")).unwrap();
    let out = &v.outputs[0];
    assert_eq!(out.len(), gpu.len());
    let differ = out
        .chunks(8)
        .zip(gpu.chunks(8))
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(differ, 42, "words where the ISA rule and GB300 differ");
}

/// PTX-FMAD: mul.sat.f32 read by add.f32 / sub.f32 stays FMUL.SAT + FADD (no contraction), a
/// plain mul read by add.sat.f32 / sub.sat.f32 is FFMA.SAT (probes/fmad_sat: every triple of
/// 16 specials, clamp edges, signed zeros, cancellation, `.ftz`; each form its own triple, so
/// ptxas cannot merge the products) = GB300, both fronts.
#[test]
fn probe_fmad_sat_matches_hardware() {
    probe_bitwise("probes/fmad_sat", "good", "ptx");
    probe_bitwise("probes/fmad_sat", "good", "sass");
}

/// probes/cvt_f16's planes: (plane, form, the front-ends that model it), from the kernel.ptx
/// header lines `//   k: form [tag]`.
fn cvt_f16_planes(ptx: &str) -> Vec<(usize, String, String)> {
    ptx.lines()
        .filter_map(|l| {
            let (k, rest) = l.strip_prefix("//")?.trim().split_once(": ")?;
            let (form, tag) = rest.rsplit_once(" [")?;
            Some((
                k.parse().ok()?,
                form.to_string(),
                tag.trim_end_matches(']').into(),
            ))
        })
        .collect()
}

/// MISS 22, N-CVT-F16: cvt between f32 and f16 / bf16 (probes/cvt_f16: 32,768 f32 words at the
/// f16 and bf16 rounding edges, every 16-bit code for the widening forms). Each front's variant
/// keeps the forms that front models; it certifies clean and each of its planes equals the
/// GB300 bytes of the full kernel.
#[test]
fn probe_cvt_f16_matches_hardware() {
    let case = Case::load(root().join("probes/cvt_f16")).unwrap();
    let gpu = std::fs::read(case.dir.join("expected/out.bin")).unwrap();
    let planes = cvt_f16_planes(&std::fs::read_to_string(case.dir.join("kernel.ptx")).unwrap());
    assert_eq!(planes.len(), 23, "{planes:?}");
    let n = 4 * 32768;
    assert_eq!(gpu.len(), n * planes.len());
    for front in ["ptx", "sass"] {
        let k = kernel(&case, front, front);
        let launch = build_launch(&case, &k).unwrap();
        let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
            .verdict
            .unwrap_or_else(|| panic!("probes/cvt_f16 {front} certifies"));
        assert_eq!(v.classes(), ["clean"], "{front}: {:#?}", v.findings);
        let mut checked = 0;
        for (p, form, tag) in &planes {
            if !(tag == "both" || tag == front) {
                continue;
            }
            let (got, want) = (&v.outputs[0][p * n..(p + 1) * n], &gpu[p * n..(p + 1) * n]);
            let bad = (0..n / 4)
                .filter(|i| got[4 * i..4 * i + 4] != want[4 * i..4 * i + 4])
                .count();
            assert_eq!(
                bad, 0,
                "{front} plane {p} ({form}): words differing from GB300"
            );
            checked += 1;
        }
        assert_eq!(checked, if front == "ptx" { 14 } else { 20 }, "{front}");
    }
}

/// MISS 22: the cvt forms between f32 and f16 / bf16 that a front-end does not model are
/// refused, never lowered with a modifier dropped. Before 22a0be18 the PTX front took
/// `cvt.sat.f32.f16` as `cvt.f32.f16` (no clamp) and `cvt.ftz.f32.bf16` as `cvt.f32.bf16` (no
/// flush), and the narrow-format path ignored unknown modifiers; before the scalar `.rn` arms
/// matched exact lists, `.relu` / `.satfinite` were dropped there.
#[test]
fn cvt_f16_unmodeled_forms_refused() {
    let case = Case::load(root().join("probes/cvt_f16")).unwrap();
    let src = std::fs::read_to_string(case.dir.join("kernel.ptx")).unwrap();
    let planes = cvt_f16_planes(&src);
    // The full kernel with every tagged line deleted except plane k's.
    let only = |k: usize| -> String {
        src.lines()
            .filter(|l| {
                !["@PTXNO", "@SASSNO", "@NONE"]
                    .iter()
                    .any(|m| l.contains(&format!("// {m}")))
                    || l.ends_with(&format!(" k{k}"))
            })
            .map(|l| format!("{l}\n"))
            .collect()
    };
    let mut n = [0, 0];
    for (k, form, tag) in &planes {
        if !matches!(tag.as_str(), "sass" | "none") {
            continue;
        }
        let err = ptxv_ptx::parse_kernel(&only(*k), Some("cvt_f16")).unwrap_err();
        assert!(format!("{err:#}").contains("cvt"), "{form}: {err:#}");
        n[0] += 1;
    }
    let dir = std::env::temp_dir().join(format!("ptxv-test-cvt_f16-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (k, form, tag) in &planes {
        if !matches!(tag.as_str(), "ptx" | "none") {
            continue;
        }
        let p = dir.join(format!("k{k}.ptx"));
        std::fs::write(&p, only(*k)).unwrap();
        let c = dir.join(format!("k{k}.cubin"));
        ptxv_sass::ptxas(&p, &c, "sm_103a").unwrap();
        assert!(
            ptxv_sass::load_cubin(&c, "cvt_f16").is_err(),
            "{form}: the SASS front lifts it"
        );
        n[1] += 1;
    }
    assert_eq!(n, [9, 3], "refused planes per front");
    // Forms neither front models (no plane): each in place of a modelled line of the same
    // result type, on the PTX front.
    let ptx = case.variant_ptx("ptx").unwrap();
    for (from, to) in [
        ("cvt.rn.f16.f32 %h7,", "cvt.rn.relu.f16.f32 %h7,"),
        ("cvt.rn.f16.f32 %h7,", "cvt.rn.satfinite.f16.f32 %h7,"),
        ("cvt.rn.f16.f32 %h7,", "cvt.rn.relu.satfinite.f16.f32 %h7,"),
        ("cvt.rn.f16.f32 %h7,", "cvt.rz.f16.f32 %h7,"),
        ("cvt.rn.f16.f32 %h7,", "cvt.rz.sat.f16.f32 %h7,"),
        ("cvt.rn.f16.f32 %h7,", "cvt.rm.f16.f32 %h7,"),
        ("cvt.rn.bf16.f32 %h8,", "cvt.rn.relu.bf16.f32 %h8,"),
        ("cvt.rn.bf16.f32 %h8,", "cvt.rn.satfinite.bf16.f32 %h8,"),
        ("cvt.rn.bf16.f32 %h8,", "cvt.rn.ftz.bf16.f32 %h8,"),
        ("cvt.rn.bf16.f32 %h8,", "cvt.rz.bf16.f32 %h8,"),
        ("cvt.rn.f16x2.f32 %r20,", "cvt.rn.relu.f16x2.f32 %r20,"),
        ("cvt.rn.f16x2.f32 %r20,", "cvt.rz.relu.f16x2.f32 %r20,"),
        ("cvt.rn.f16x2.f32 %r20,", "cvt.rz.satfinite.f16x2.f32 %r20,"),
        ("cvt.rn.bf16x2.f32 %r21,", "cvt.rn.relu.bf16x2.f32 %r21,"),
        ("cvt.rn.bf16x2.f32 %r21,", "cvt.rz.bf16x2.f32 %r21,"),
        (
            "cvt.rn.bf16x2.f32 %r21,",
            "cvt.rz.relu.satfinite.bf16x2.f32 %r21,",
        ),
    ] {
        let bad = ptx.replacen(from, to, 1);
        assert_ne!(bad, ptx, "{to}");
        assert!(
            ptxv_ptx::parse_kernel(&bad, Some("cvt_f16")).is_err(),
            "{to}"
        );
    }
    // The narrow-format path: one rounding, `.satfinite` / `.relu` once each, nothing else.
    let lp = std::fs::read_to_string(root().join("probes/cvt_lp/kernel.ptx")).unwrap();
    for to in [
        "cvt.rn.ftz.satfinite.e4m3x2.f32",
        "cvt.rn.satfinite.satfinite.e4m3x2.f32",
        "cvt.rn.rz.satfinite.e4m3x2.f32",
        "cvt.rn.satfinite.scaled::n2::ue8m0.e4m3x2.f32",
    ] {
        let bad = lp.replacen("cvt.rn.satfinite.e4m3x2.f32", to, 1);
        assert_ne!(bad, lp, "{to}");
        assert!(
            ptxv_ptx::parse_kernel(&bad, Some("cvt_lp")).is_err(),
            "{to}"
        );
    }
}

/// MISS 59: cvt.bf16.bf16 without a modifier on every 16-bit code (probes/cvt_bf16_same;
/// ptxas 13.0 / 13.3: HFMA2.BF16_V2 d, -RZ, RZ, a, every NaN 0x7fff). The SASS front equals
/// GB300; the PTX front refuses the form (before, a 16-bit copy: 253 of 65,536 words differed).
#[test]
fn probe_cvt_bf16_same_verdicts() {
    probe_bitwise("probes/cvt_bf16_same", "good", "sass");
    let case = Case::load(root().join("probes/cvt_bf16_same")).unwrap();
    let err =
        ptxv_ptx::parse_kernel(&case.variant_ptx("good").unwrap(), Some("bf16_same")).unwrap_err();
    assert!(format!("{err:#}").contains("cvt.bf16.bf16"), "{err:#}");
}

/// MISS 59: cvt takes the integer path for the eight integer types only. Before, any type
/// starting with u / s / b counted: int <-> bf16 without a rounding modifier (ptxas rejects
/// them) lowered as a bit copy, and pairs with bf16x2 or a narrow format panicked.
#[test]
fn cvt_non_integer_types_refused() {
    let src = std::fs::read_to_string(root().join("probes/cvt_bf16_same/kernel.ptx")).unwrap();
    let line = "cvt.bf16.bf16   %h2, %h1;";
    let mut forms: Vec<String> = Vec::new();
    for (t, r) in [
        ("s8", "%h"),
        ("u8", "%h"),
        ("s16", "%h"),
        ("u16", "%h"),
        ("s32", "%r"),
        ("u32", "%r"),
        ("s64", "%rd"),
        ("u64", "%rd"),
    ] {
        forms.push(format!("cvt.bf16.{t} %h2, {r}1;"));
        forms.push(format!("cvt.{t}.bf16 {r}3, %h1;"));
    }
    for t in ["s2f6x2", "ue5m3x2", "ue8m0x2"] {
        forms.push(format!("cvt.bf16.{t} %h2, %h1;"));
        forms.push(format!("cvt.bf16x2.{t} %r3, %h1;"));
        forms.push(format!("cvt.{t}.bf16 %h2, %h1;"));
        forms.push(format!("cvt.{t}.bf16x2 %h2, %r1;"));
    }
    for f in [
        "cvt.bf16.bf16x2 %h2, %r1;",
        "cvt.bf16x2.bf16 %r3, %h1;",
        "cvt.bf16x2.bf16x2 %r3, %r1;",
    ] {
        forms.push(f.into());
    }
    assert_eq!(forms.len(), 31);
    for to in &forms {
        let bad = src.replacen(line, to, 1);
        assert_ne!(bad, src, "{to}");
        match std::panic::catch_unwind(|| ptxv_ptx::parse_kernel(&bad, Some("bf16_same"))) {
            Ok(Err(e)) => assert!(format!("{e:#}").contains("cvt"), "{to}: {e:#}"),
            Ok(Ok(_)) => panic!("{to}: lowered"),
            Err(_) => panic!("{to}: the PTX front panicked"),
        }
    }
    for to in [
        "cvt.u16.u32 %h2, %r1;",
        "cvt.s32.s8 %r3, %h1;",
        "cvt.u64.u16 %rd3, %h1;",
    ] {
        let ok = src.replacen(line, to, 1);
        assert!(
            ptxv_ptx::parse_kernel(&ok, Some("bf16_same")).is_ok(),
            "{to}"
        );
    }
}

/// MISS 60: `.satfinite` on the bf16x2 decode of e5m2 (probes/cvt_lp_satfinite, ptxas 13.3):
/// GB300 maps +-Inf to +-0x7f7f, as the PTX ISA says. The decode without `.satfinite` equals
/// GB300 on the PTX front; both fronts refuse the `.satfinite` forms (before, the PTX front
/// dropped it: 1,020 of 65,536 words differed, 511 with `.relu`).
#[test]
fn probe_cvt_lp_satfinite_verdicts() {
    let case = Case::load(root().join("probes/cvt_lp_satfinite")).unwrap();
    let gpu = std::fs::read(case.dir.join("expected/out.bin")).unwrap();
    let n = 4 * 65536;
    let k = kernel(&case, "ptx", "ptx");
    let launch = build_launch(&case, &k).unwrap();
    let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
        .verdict
        .expect("probes/cvt_lp_satfinite ptx certifies");
    assert_eq!(v.classes(), ["clean"], "{:#?}", v.findings);
    let bad = (0..n / 4)
        .filter(|i| v.outputs[0][4 * i..4 * i + 4] != gpu[4 * i..4 * i + 4])
        .count();
    assert_eq!(bad, 0, "cvt.rn.bf16x2.e5m2x2: words differing from GB300");
    let tk = ptxv_ptx::toolkit_version("13.3");
    let err = ptxv_ptx::parse_kernel_for(&case.variant_ptx("good").unwrap(), Some("sf_e5m2"), tk)
        .unwrap_err();
    assert!(format!("{err:#}").contains("satfinite"), "{err:#}");
    assert!(
        ptxv_sass::load_cubin(&case.dir.join("kernel.cubin"), "sf_e5m2").is_err(),
        "the SASS front lifts F2FP.SATFINITE.BF16.E5M2.UNPACK_B"
    );
}

/// MISS 61: SASS F2FP [.RELU] .BF16.<fmt>.UNPACK_B d, a, s.H0 (ptxas 13.3's bf16x2 decodes) is
/// lifted only with the identity scale 0x7f7f.H0 of the unscaled forms: the decode op has no
/// scale. probes/cvt_lp_scaled (cvt.rn [.relu] .scaled::n2::ue8m0 .bf16x2.<f8 / f6 / f4>x2 on
/// every code pair, mixed scales): the SASS front refuses its register scales (before, it
/// ignored them: 562,392 of 655,360 words differed from GB300), the PTX front the `.scaled`
/// forms; the unscaled decode (probes/cvt_lp_satfinite, 13.3 cubin of variant ptx) = GB300.
#[test]
fn probe_cvt_lp_scaled_verdicts() {
    let case = Case::load(root().join("probes/cvt_lp_scaled")).unwrap();
    let err = ptxv_sass::load_cubin(&case.dir.join("kernel.cubin"), &case.m.entry).unwrap_err();
    assert!(format!("{err:#}").contains("UNPACK_B scale"), "{err:#}");
    let tk = ptxv_ptx::toolkit_version("13.3");
    let ptx = case.variant_ptx("good").unwrap();
    assert!(ptxv_ptx::parse_kernel_for(&ptx, Some(&case.m.entry), tk).is_err());
    for (line, ok) in [
        (
            "F2FP.BF16.E4M3.UNPACK_B R13, R0.reuse, 4.5736980577097704378e-41.H0",
            true,
        ),
        (
            "F2FP.RELU.BF16.E2M1.UNPACK_B R9, R8, 4.5736980577097704378e-41.H0",
            true,
        ),
        ("F2FP.F16.E4M3.UNPACK_B R23, R0", true),
        ("F2FP.BF16.E4M3.UNPACK_B R13, R8.reuse, R9.reuse.H0", false),
        (
            "F2FP.BF16.E5M2.UNPACK_B R13, R8, 1.1754943508222875079e-38.H0",
            false,
        ),
        (
            "F2FP.BF16.E5M2.UNPACK_B R13, R8, 4.5736980577097704378e-41.H1",
            false,
        ),
    ] {
        assert_eq!(ptxv_sass::lower_text(line, None).is_ok(), ok, "{line}");
    }
    let sf = Case::load(root().join("probes/cvt_lp_satfinite")).unwrap();
    let gpu = std::fs::read(sf.dir.join("expected/out.bin")).unwrap();
    let k = ptxv_sass::load_cubin(&sf.dir.join("kernel_ptx.cubin"), &sf.m.entry).unwrap();
    let launch = build_launch(&sf, &k).unwrap();
    let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
        .verdict
        .expect("probes/cvt_lp_satfinite kernel_ptx.cubin certifies");
    assert_eq!(v.classes(), ["clean"], "{:#?}", v.findings);
    let bad = (0..65536)
        .filter(|i| v.outputs[0][4 * i..4 * i + 4] != gpu[4 * i..4 * i + 4])
        .count();
    assert_eq!(
        bad, 0,
        "F2FP.BF16.E5M2.UNPACK_B (scale 0x7f7f.H0): words differing from GB300"
    );
}

/// MISS 23: integer add / sub / logic / shift forms other than one type (or add / sub.sat.s32,
/// probes/int_sat, and the packed add.{u,s}16x2) are refused, never lowered as the plain op:
/// `.cc` lost its carry, `.sat` on other types its clamp, a packed sub its lanes (a 32-bit sub
/// borrows across them; probes/sub16x2). sub.{u,s}16x2 is not in the PTX ISA: refused too.
#[test]
fn int_arith_unmodeled_forms_refused() {
    let src = std::fs::read_to_string(root().join("probes/int_sat/kernel.ptx")).unwrap();
    let line = "add.sat.s32     %r4, %r2, %r3;";
    for to in [
        "add.cc.u32 %r4, %r2, %r3;",
        "add.cc.s32 %r4, %r2, %r3;",
        "sub.cc.u32 %r4, %r2, %r3;",
        "add.sat.u32 %r4, %r2, %r3;",
        "add.sat.s16x2 %r4, %r2, %r3;",
        "add.sat.u16x2 %r4, %r2, %r3;",
        "sub.sat.s16x2 %r4, %r2, %r3;",
        "add.s8x4 %r4, %r2, %r3;",
        "sub.u8x4 %r4, %r2, %r3;",
        "and.cc.b32 %r4, %r2, %r3;",
        "add.b32 %r4, %r2, %r3;",
        "shl.sat.b32 %r4, %r2, %r3;",
        "sub.u16x2 %r4, %r2, %r3;",
        "sub.s16x2 %r4, %r2, %r3;",
    ] {
        let bad = src.replacen(line, to, 1);
        assert_ne!(bad, src, "{to}");
        assert!(
            ptxv_ptx::parse_kernel(&bad, Some("int_sat")).is_err(),
            "{to}"
        );
    }
    for to in [
        "add.u32 %r4, %r2, %r3;",
        "sub.s32 %r4, %r2, %r3;",
        "add.u16x2 %r4, %r2, %r3;",
        "xor.b32 %r4, %r2, %r3;",
    ] {
        let ok = src.replacen(line, to, 1);
        assert!(ptxv_ptx::parse_kernel(&ok, Some("int_sat")).is_ok(), "{to}");
    }
}

/// MISS 25: multimem `.acc::f16` (the fp8 types' f16 accumulation, PTX ISA 9.5 section
/// 9.7.10.15; not modeled) and `.acc::f32` anywhere but a 16-bit float `ld_reduce` are refused
/// (fixtures/r1h_nvls's instructions rewritten); before 7e2c0687 `.acc::f16` was taken as
/// `.acc::f32`.
#[test]
fn multimem_acc_unmodeled_forms_refused() {
    let src = std::fs::read_to_string(root().join("fixtures/r1h_nvls/kernel.ptx")).unwrap();
    let ldr = "multimem.ld_reduce.relaxed.sys.global.add.f32 %f2, [%rd8];";
    let red = "multimem.red.release.sys.global.add.u32 [%rd3+64], 1;";
    for (from, to) in [
        (
            ldr,
            "multimem.ld_reduce.relaxed.sys.global.add.acc::f16.e5m2x4 %r3, [%rd8];",
        ),
        (
            ldr,
            "multimem.ld_reduce.relaxed.sys.global.add.acc::f16.v4.e4m3x4 {%r0, %r1, %r2, %r3}, [%rd8];",
        ),
        (
            ldr,
            "multimem.ld_reduce.relaxed.sys.global.add.acc::f16.f16x2 %r3, [%rd8];",
        ),
        (
            ldr,
            "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.f32 %f2, [%rd8];",
        ),
        (
            red,
            "multimem.red.release.sys.global.add.acc::f32.u32 [%rd3+64], 1;",
        ),
    ] {
        let bad = src.replacen(from, to, 1);
        assert_ne!(bad, src, "{to}");
        assert!(ptxv_ptx::parse_kernel(&bad, Some("nvls")).is_err(), "{to}");
    }
    for to in [
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.bf16x2 %r3, [%rd8];",
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.f16x2 {%r0, %r1, %r2, %r3}, [%rd8];",
    ] {
        let ok = src.replacen(ldr, to, 1);
        assert!(ptxv_ptx::parse_kernel(&ok, Some("nvls")).is_ok(), "{to}");
    }
}

/// S-NC on GB300 (probes/nc_hoist): ptxas 13.0 / 13.3 hoist an ld.global.nc above a store and
/// a bar.sync, so GB300 returns the word's old value (expected/: zeros, 20/20 scrubbed runs)
/// where a coherent load after the barrier reads t ^ 1. Both fronts report the load.
#[test]
fn probe_nc_hoist_reported() {
    check_dir("probes", "nc_hoist", "good", "ptx");
    check_dir("probes", "nc_hoist", "good", "sass");
}

/// S-NC with PDL on GB300 (probes/nc_pdl): ptxas 13.0 / 13.3 move a dependent's ld.global.nc
/// above griddepcontrol.wait (ACQBULK) and above the flag store that lets the primary write,
/// so GB300 returns the primary's old words (hoist: 0, 200/200 runs) where the coherent load
/// of good reads t + 7 (200/200 = model). Both fronts report hoist and late (the .nc load kept
/// after the wait; GB300 0 of 200 runs stale: a bound, not a proof).
#[test]
fn probe_nc_pdl_reported() {
    for v in ["good", "hoist", "late"] {
        check_dir("probes", "nc_pdl", v, "ptx");
        check_dir("probes", "nc_pdl", v, "sass");
    }
}

/// E-HW-L1, E-PDL (probes/pdl_acqbulk): a PDL dependent loads D[0], filling the line of the
/// word D[5] its primary then stores, and loads D[5] after griddepcontrol.wait (ACQBULK, no
/// CCTL.IVALL). WAIT and POLLF (a relaxed poll and fence.acq_rel.gpu) are clean, NOWAIT and
/// POLL (the relaxed poll alone) races, on both fronts. GB300 (two jobs): WAIT read 5 in 3,072,000
/// of 3,072,000 CTAs, POLL the stale 0 in 3,048,401.
#[test]
fn probe_pdl_acqbulk() {
    for v in ["WAIT", "NOWAIT", "POLL", "POLLF"] {
        check_dir("probes", "pdl_acqbulk", v, "ptx");
        check_dir("probes", "pdl_acqbulk", v, "sass");
    }
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
/// before the read = GB300 on both front-ends (SASS CCTL.E.RML2); read after the discard: a
/// violation; the rewrite unordered with the discard: a race.
#[test]
fn probe_discard_l2_verdicts() {
    for front in ["ptx", "sass"] {
        probe_bitwise("probes/discard_l2", "good", front);
        check_dir("probes", "discard_l2", "read_after", front);
        check_dir("probes", "discard_l2", "race", front);
    }
}

/// tcgen05.mma cta_group::2 M = 128 with A in TMEM (probes/tc_pair_atmem) = GB300 on both
/// front-ends (PTX: the peer's commit waited at .cta scope is an S-MBAR-SCOPE contract finding;
/// SASS: the overlapping un-fenced STTMs are E-TMEM-QUEUE, `probe_bitwise_queue`).
#[test]
fn probe_tc_pair_atmem_matches_hardware() {
    probe_bitwise_mbar_contract("probes/tc_pair_atmem", "good");
    probe_bitwise_queue("probes/tc_pair_atmem", "good", "sass");
}

/// tcgen05.mma kind::tf32 MN-major through SWIZZLE_128B_BASE32B (probes/tc_tf32_mn) = GB300 on
/// both front-ends.
#[test]
fn probe_tc_tf32_mn_matches_hardware() {
    probe_bitwise("probes/tc_tf32_mn", "good", "ptx");
    probe_bitwise("probes/tc_tf32_mn", "good", "sass");
}

/// tcgen05.mma finite overflow to Inf (probes/tc_mma_ovf) = GB300 on both front-ends (the
/// overlapping un-waited stores are E-TMEM-QUEUE, `probe_bitwise_queue`; PTX: MISS 116).
#[test]
fn probe_tc_mma_ovf_matches_hardware() {
    probe_bitwise_queue("probes/tc_mma_ovf", "good", "ptx");
    probe_bitwise_queue("probes/tc_mma_ovf", "good", "sass");
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
/// front-ends (the overlapping un-waited stores are E-TMEM-QUEUE, `probe_bitwise_queue`; PTX:
/// MISS 116).
#[test]
fn probe_tc_mma_imm_matches_hardware() {
    probe_bitwise_queue("probes/tc_mma_imm", "good", "ptx");
    probe_bitwise_queue("probes/tc_mma_imm", "good", "sass");
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

/// Counted arrivals in the reductions (fixtures/mbar_count_order): the count-3 arrival that
/// completes the phase alone must be explored before the count-1 arrival (the race; on the
/// SASS front, two outputs) and after it (the fault), by both explorers on both front-ends.
/// DPOR may also report the race's second outcome on the PTX front (E-DRF). Variant `aba`
/// (two count-3 arrivals: an arrival lands in phase 0 or 1 depending on the order) and `good`
/// both over-arrive, so DPOR must hand the search to the stateful explorer (its note says so).
#[test]
fn mbar_count_order_reaches_both_orders() {
    let case = Case::load(root().join("fixtures/mbar_count_order")).unwrap();
    for (variant, front) in [
        ("good", "ptx"),
        ("good", "sass"),
        ("aba", "ptx"),
        ("aba", "sass"),
    ] {
        let k = Arc::new(kernel(&case, variant, front));
        let launch = Arc::new(build_launch(&case, &k).unwrap());
        let m0 = Machine::new(k, launch);
        let want = case.variant(variant).unwrap().expect_for(front);
        let v = ptxv_sim::explore::explore(m0.clone(), 100_000);
        assert!(v.complete, "{variant}/{front}: incomplete");
        assert_eq!(v.classes(), want, "{variant}/{front}: {:#?}", v.findings);
        let d = ptxv_sim::dpor::explore_dpor(m0, 100_000);
        assert!(d.complete, "{variant}/{front}: DPOR incomplete");
        let mut dc = d.classes();
        if front == "ptx" {
            dc.retain(|c| c != "nondeterministic");
        }
        assert_eq!(dc, want, "{variant}/{front}: DPOR: {:#?}", d.findings);
        // Every execution over-arrives: DPOR hands the search to the stateful explorer.
        assert!(
            d.notes.iter().any(|n| n.contains("over-arrival")),
            "{variant}/{front}: DPOR did not fall back: {:?}",
            d.notes
        );
    }
}

/// Plain over-arrivals in the reductions (fixtures/mbar_over_arrival): with 4 plain arrivals
/// at a barrier expecting 3, the order in which the three that do not follow the data write
/// complete the phase alone (the race; on the SASS front, two outputs) must be explored, by both
/// explorers on both front-ends, whether the extra arrivals are separate warps (`good`) or lanes
/// of one warp step (`lanes`). DPOR hands such a kernel to the stateful explorer
/// (`OVER_ARRIVAL` in dpor.rs).
#[test]
fn mbar_over_arrival_reaches_late_order() {
    let case = Case::load(root().join("fixtures/mbar_over_arrival")).unwrap();
    for variant in ["good", "lanes"] {
        for front in ["ptx", "sass"] {
            let k = Arc::new(kernel(&case, variant, front));
            let launch = Arc::new(build_launch(&case, &k).unwrap());
            let m0 = Machine::new(k, launch);
            let want = case.variant(variant).unwrap().expect_for(front);
            let v = ptxv_sim::explore::explore(m0.clone(), 100_000);
            assert!(v.complete, "{variant}/{front}: incomplete");
            assert_eq!(v.classes(), want, "{variant}/{front}: {:#?}", v.findings);
            let d = ptxv_sim::dpor::explore_dpor(m0, 100_000);
            assert!(d.complete, "{variant}/{front}: DPOR incomplete");
            assert_eq!(
                d.classes(),
                want,
                "{variant}/{front}: DPOR: {:#?}",
                d.findings
            );
        }
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

/// N-CVT-LP, L-NVDISASM: cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2 (F2FP.SATFINITE
/// .BF16.S2_6.UNPACK_B, register and immediate scales, either data half; ptxas 13.1 cubin whose
/// F2FP encodings nvdisasm 13.0 does not print, decoded by undis.rs) on every (code, scale) pair
/// of both lanes = GB300 (probes/s2f6_cvt), on both front-ends.
#[test]
fn probe_s2f6_cvt_matches_hardware() {
    probe_bitwise_cubin("probes/s2f6_cvt", None);
    // The PTX front on the cubin's source (the case is binary-only: ptxas 13.0 rejects it).
    let case = Case::load(root().join("probes/s2f6_cvt")).unwrap();
    let ptx = std::fs::read_to_string(case.dir.join("kernel.ptx")).unwrap();
    let k = ptxv_ptx::parse_kernel_for(
        &ptx,
        Some(&case.m.entry),
        case.m
            .ptxas_toolkit
            .as_deref()
            .and_then(ptxv_ptx::toolkit_version),
    )
    .unwrap();
    let launch = build_launch(&case, &k).unwrap();
    let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
        .verdict
        .expect("s2f6_cvt (ptx) certifies");
    assert_eq!(v.classes(), ["clean"], "{:#?}", v.findings);
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
        "{report:?}"
    );
}

/// L-ROSETTA, S-SCOPE: red.global.{or,and,xor}.b32 / .b64 at cta, gpu and sys scope (REDG.E
/// .{OR,AND,XOR}[.64].STRONG.{SM,GPU,SYS}), four threads per word = GB300 (probes/red_bitops).
#[test]
fn probe_red_bitops_matches_hardware() {
    probe_bitwise("probes/red_bitops", "good", "ptx");
    probe_bitwise("probes/red_bitops", "good", "sass");
}

/// N-MMA: tcgen05.mma scale-input-d (UTCHMMA ..., UPd, s; D = A B + D * 2^-s) for kind::f16
/// (bf16) and kind::tf32 over 9 scales = GB300 (probes/mma_scale_d), on both front-ends; the
/// PTX statements include `..., 0, 8` (enable-input-d 0: the scale of an unread accumulator is
/// moot, D is overwritten).
#[test]
fn probe_mma_scale_d_matches_hardware() {
    probe_bitwise("probes/mma_scale_d", "good", "sass");
    probe_bitwise("probes/mma_scale_d", "good", "ptx");
}

/// S-TMA, S-TMAP: im2col TMA loads (UTMALDG.<n>D.IM2COL) and im2col_no_offs stores
/// (UTMASTG.<n>D.IM2COL) through cuTensorMapEncodeIm2col maps = GB300 (probes/tma_im2col).
/// SASS front only: the PTX front-end does not lower im2col.
#[test]
fn probe_tma_im2col_matches_hardware() {
    probe_bitwise("probes/tma_im2col", "good", "sass");
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

/// N-CVT-LP: the s2f6 dequant `cvt.rn[.relu][.satfinite].scaled::n2::ue8m0.bf16x2.s2f6x2` in
/// its four forms on 1024 sampled threads (probes/s2f6_cvt/forms) = GB300 on both front-ends (the
/// SASS one on kernel.cubin, ptxas 13.1's F2FP[.SATFINITE][.RELU].BF16.S2_6.UNPACK_B). GB200's
/// output of the same threads (gb200_out.bin) is identical.
#[test]
fn probe_s2f6_cvt_forms_match_hardware() {
    probe_bitwise("probes/s2f6_cvt/forms", "good", "ptx");
    probe_bitwise_cubin("probes/s2f6_cvt/forms", Some("kernel.cubin"));
    let dir = root().join("probes/s2f6_cvt/forms");
    assert_eq!(
        std::fs::read(dir.join("gb200_out.bin")).unwrap(),
        std::fs::read(dir.join("gb300_out.bin")).unwrap()
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

/// L-ROSETTA: `LEA.HI.X Rd, Ra, ~Rb, RZ, s, P` (the high word of a zero-extended word shifted
/// left by s = 1..30 minus a 64-bit value, after either low-word LEA) = GB300
/// (probes/lea_hi_not): the PTX front, the local ptxas's SASS, and the checked-in ptxas 13.1
/// cubin, which holds the form 60 times.
#[test]
fn probe_lea_hi_not_matches_hardware() {
    probe_bitwise("probes/lea_hi_not", "good", "ptx");
    probe_bitwise("probes/lea_hi_not", "good", "sass");
    probe_bitwise_cubin("probes/lea_hi_not", Some("kernel.cubin"));
    let case = Case::load(root().join("probes/lea_hi_not")).unwrap();
    let k = ptxv_sass::load_cubin(&case.dir.join("kernel.cubin"), &case.m.entry).unwrap();
    let forms = k
        .code
        .iter()
        .filter(|i| {
            matches!(
                i.op,
                ptxv_ir::Op::LeaHi {
                    not_b: true,
                    sx32: false,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        forms, 60,
        "LEA.HI.X Rd, Ra, ~Rb, RZ, s, P for s = 1..30, twice"
    );
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

/// L-ROSETTA: a negated register source on f32 -> 32-bit integer conversions, F2I[.FTZ][.U32]
/// [.TRUNC|.FLOOR|.CEIL].NTZ Rd, -Ra (neg.f32 folded into cvt.r{n,z,m,p}i[.ftz].{s32,u32}.f32)
/// = GB300 (probes/f2i_neg): the PTX front, the local ptxas's SASS, and the checked-in ptxas
/// 13.1 cubin, which holds all 16 forms.
#[test]
fn probe_f2i_neg_matches_hardware() {
    probe_bitwise("probes/f2i_neg", "good", "ptx");
    probe_bitwise("probes/f2i_neg", "good", "sass");
    probe_bitwise_cubin("probes/f2i_neg", Some("kernel.cubin"));
    let case = Case::load(root().join("probes/f2i_neg")).unwrap();
    let k = ptxv_sass::load_cubin(&case.dir.join("kernel.cubin"), &case.m.entry).unwrap();
    let forms = k
        .code
        .iter()
        .filter(|i| matches!(i.op, ptxv_ir::Op::F2I { neg: true, .. }))
        .count();
    assert_eq!(
        forms, 16,
        "F2I ... Rd, -Ra: four roundings x s32 / u32 x [.FTZ]"
    );
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
        let m0 = Machine::new(k, launch);
        let mut v = ptxv_sim::explore::explore(m0.clone(), 500_000);
        assert!(v.complete, "{front}: incomplete");
        ptxv_sim::twolevel::mbar_async_contract(&mut v, &m0, &mut |m| {
            ptxv_sim::explore::explore(m, 500_000)
        });
        // PTX: the bulk copy's complete-tx from the peer CTA, waited at .cta scope, orders the
        // data only if it counts as the mbarrier CTA's (S-MBAR-SCOPE contract).
        let want: &[&str] = if front == "ptx" {
            &["contract"]
        } else {
            &["clean"]
        };
        assert_eq!(v.classes(), want, "{front}: {:#?}", v.findings);
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

/// L-OPCLASS by module SM (probes/f64_arch): the sm_100f build's stall-count-only schedule (and
/// the same with every f64 op at st=1) = GB300, because GB300 runs the image the driver reschedules
/// at load (cubins/sm100f.loaded.cubin: an sm_103 image with f64 scoreboards), which certifies
/// clean and equals GB300 too; the sm_103a build with its f64 scoreboards removed mismatches on
/// GB300 and is a SCOREBOARD violation here. Verdicts on the sm_100 cubins are marked as not
/// GB300 verdicts (`static_image`); the toolkit note's `-arch` is read but selects nothing.
#[test]
fn probe_f64_arch_matches_hardware() {
    let dir = "probes/f64_arch";
    probe_bitwise(dir, "good", "ptx");
    probe_bitwise(dir, "good", "sass");
    for c in [
        "cubins/sm100f.cubin",
        "cubins/sm100f_st1.cubin",
        "cubins/sm100f_st1_tk103a.cubin",
        "cubins/sm100f.loaded.cubin",
        "cubins/sm103a.cubin",
    ] {
        probe_bitwise_cubin(dir, Some(c));
        let k = ptxv_sass::load_cubin(&root().join(dir).join(c), "f64h").unwrap();
        assert_eq!(
            k.static_image.is_some(),
            c.contains("sm100f") && !c.contains("loaded"),
            "{c}"
        );
    }
    for c in [
        "cubins/sm103a_nosb.cubin",
        "cubins/sm103a_nosb_tk100f.cubin",
    ] {
        let k = ptxv_sass::load_cubin(&root().join(dir).join(c), "f64h").unwrap();
        assert!(
            k.static_findings
                .iter()
                .any(|f| f.msg.starts_with("SCOREBOARD: ") && f.msg.contains("`DFMA ")),
            "{c}: {:#?}",
            k.static_findings
        );
        assert!(k.static_image.is_none());
    }
    let meta = |c: &str| ptxv_sass::read_meta(&root().join(dir).join(c), "f64h").unwrap();
    let (a, b) = (
        meta("cubins/sm100f.cubin"),
        meta("cubins/sm103a_nosb_tk100f.cubin"),
    );
    assert_eq!((a.sm, a.target.as_str()), (100, "sm_100f"));
    assert_eq!((b.sm, b.target.as_str()), (103, "sm_100f"));
    assert_eq!(meta("cubins/sm103a.cubin").target, "sm_103a");
    assert_eq!(meta("cubins/sm100f.loaded.cubin").sm, 103);
}

/// L-ROSETTA: stmatrix.m16n8.x{1,2,4}.trans.b8 (SASS STSM.8.MT168[.N]) = GB300 on both
/// front-ends (probes/stsm_b8).
#[test]
fn probe_stsm_b8_matches_hardware() {
    probe_bitwise("probes/stsm_b8", "good", "ptx");
    probe_bitwise("probes/stsm_b8", "good", "sass");
}

/// L-ROSETTA: VIADDMNMX.RELU with a negated second operand (`-Rb`, a fused subtraction) =
/// GB300 (probes/iaddmnmx_relu), both front-ends.
#[test]
fn probe_iaddmnmx_relu_matches_hardware() {
    probe_bitwise("probes/iaddmnmx_relu", "good", "ptx");
    probe_bitwise("probes/iaddmnmx_relu", "good", "sass");
}

/// L-ROSETTA: a register-indexed constant-bank load with a negative offset
/// (`LDC R4, c[0x0][R0+-0x18f4]`, added mod 2^32) = GB300 (probes/cbank_neg), both front-ends.
#[test]
fn probe_cbank_neg_matches_hardware() {
    probe_bitwise("probes/cbank_neg", "good", "ptx");
    probe_bitwise("probes/cbank_neg", "good", "sass");
}

/// N-QMUL4: QMUL4.E4M3x4.E2M1x4.E4M3x4 (.N3210 / .N7654, .B0) on every e2m1 code x e4m3 scale
/// x nibble position = GB300 (probes/qmul4; SASS only, the instruction is patched in).
#[test]
fn probe_qmul4_matches_hardware() {
    probe_bitwise_cubin("probes/qmul4", None);
}

/// N-RED16: two unordered bf16 tensor bulk reductions into one box (probes/tma_red_order,
/// mode 0). The verdict is `nondeterministic` (never deterministic); every explored output is,
/// per element, one of the two values GB300 produced over 300 launches (each element was seen
/// in both orders); and each element's numerics bound (`out_rec`, every order: two roundings
/// per value) holds both observed values, at a meaningful fraction of it. Mutants the checks
/// must catch: a model fusing the two adds into one rounding (its sums are not among the
/// observed values), and a bound for the simulated order only (an observed value exceeds it).
#[test]
fn red16_order_bound_covers_gb300() {
    use ptxv_sim::halfin::Half;
    let case = Case::load(root().join("probes/tma_red_order")).unwrap();
    let rd16 = |f: &str| -> Vec<u16> {
        std::fs::read(case.dir.join(f))
            .unwrap()
            .chunks(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect()
    };
    let (g, a, obs) = (rd16("g.bin"), rd16("a.bin"), rd16("observed.bin"));
    let bf = |h: u16| f32::from_bits((h as u32) << 16) as f64;
    // Round to nearest even to bf16 (normal results only, as gen.py's triples give).
    let rn = |x: f64| -> u16 {
        let b = x.to_bits();
        let e = (b >> 52 & 0x7ff) as i64 - 1023;
        assert!((-126..128).contains(&e), "{x}");
        let m = b & ((1u64 << 52) - 1);
        let (keep, rest) = (m >> 45, m & ((1u64 << 45) - 1));
        let half = 1u64 << 44;
        let mut k = (1u64 << 7) | keep;
        if rest > half || rest == half && k & 1 == 1 {
            k += 1;
        }
        let v = (k as f64) * 2f64.powi(e as i32 - 7) * x.signum();
        ((v as f32).to_bits() >> 16) as u16
    };
    let (ia, ig) = (0usize, 1usize);
    for front in ["ptx", "sass"] {
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let m0 = Machine::new(Arc::new(k), Arc::new(launch));
        let v = explore(m0.clone(), 2_000_000);
        assert!(v.complete, "{front}: incomplete");
        assert_eq!(
            v.classes(),
            ["nondeterministic"],
            "{front}: {:#?}",
            v.findings
        );
        let seen = |i: usize, h: u16| h == obs[i] || h == obs[4096 + i];
        for o in &v.outputs {
            for i in 0..4096 {
                let h = u16::from_le_bytes([o[2 * i], o[2 * i + 1]]);
                assert!(
                    seen(i, h),
                    "{front}: element {i} = {h:#06x} not observed on GB300"
                );
            }
        }
        let mut m = m0.clone();
        let mut ns = ptxv_sim::num::NumState::with_mma_eps(m.launch.mma_model.eps());
        ns.half.track = true;
        ns.half.inputs.insert(ia, Half::Bf16);
        ns.half.inputs.insert(ig, Half::Bf16);
        m.num = Some(Box::new(ns));
        let (m, _, ok) = ptxv_sim::explore::run_once(m);
        assert!(ok);
        let ns = m.num.as_ref().unwrap();
        let (u, ab) = ptxv_sim::equiv::round_unit(false);
        let (mut worst, mut fused_seen, mut order_only_broken) = (0f64, 0, 0);
        for i in 0..4096 {
            let n = m.half_out_node(ig, i as u64, false).expect("output node");
            let r = ns.out_rec(i, ns.nodes[n as usize].hw, Some(n));
            let e = bf(g[i]) + bf(a[i]) + bf(a[4096 + i]);
            assert_eq!(r.exact, e, "{front}: element {i}");
            for h in [obs[i], obs[4096 + i]] {
                let d = (bf(h) - e).abs();
                assert!(
                    d <= r.bound_cor1a,
                    "{front}: element {i}: |{} - {e}| > {}",
                    bf(h),
                    r.bound_cor1a
                );
                worst = worst.max(d / r.bound_cor1a);
            }
            // Mutant: one rounding of g + a0 + a1.
            fused_seen += usize::from(seen(i, rn(e)));
            // Mutant: the bound of the simulated order only (the first value added under two
            // roundings, the second under one).
            let ptxv_sim::num::NodeKind::AtomAdd16(o, x2, _) = ns.nodes[n as usize].kind else {
                panic!("{front}: element {i} is not a 16-bit sum")
            };
            let ptxv_sim::num::NodeKind::AtomAdd16(b0, x1, _) = ns.nodes[o as usize].kind else {
                panic!("{front}: element {i}: one add")
            };
            // The chain keeps the element's history across each add's own write: it starts at
            // the input element's leaf, not at a fresh leaf of an earlier rounded sum.
            assert!(
                matches!(
                    ns.nodes[b0 as usize].kind,
                    ptxv_sim::num::NodeKind::Input(_)
                ),
                "{front}: element {i}: chain start is {:?} ({})",
                ns.nodes[b0 as usize].kind,
                f32::from_bits(ns.nodes[b0 as usize].hw)
            );
            let mag = |k: u32| (f32::from_bits(ns.nodes[k as usize].hw) as f64).abs();
            let g2 = (1.0 + u) * (1.0 + u) - 1.0;
            let order_only =
                (mag(b0) + mag(x1)) * g2 + mag(x2) * u + 2.0 * ab * (1.0 + u) * (1.0 + u);
            order_only_broken += [obs[i], obs[4096 + i]]
                .iter()
                .filter(|&&h| (bf(h) - e).abs() > order_only)
                .count();
        }
        assert!(
            worst > 0.5,
            "{front}: bound too loose (worst ratio {worst})"
        );
        assert_eq!(fused_seen, 0, "{front}: a fused model would pass");
        assert!(
            order_only_broken > 0,
            "{front}: an order-only bound would pass"
        );
    }
}

/// N-F2FP-RZ: F2FP.F16.F32.PACK_AB.RZ (cvt.rz.f16x2.f32) on edge and random pairs = GB300 on
/// both front-ends (probes/f2fp_rz; its sweep.ptx checks every f32 per lane on the GPU).
#[test]
fn probe_f2fp_rz_matches_hardware() {
    probe_bitwise("probes/f2fp_rz", "good", "ptx");
    probe_bitwise("probes/f2fp_rz", "good", "sass");
}

/// LDC.U16 / LDCU.U16 (ld.param.u16 at 2-byte offsets, odd halfwords included; values with bit
/// 15 set zero-extended) = GB300 on both front-ends (probes/ldc_u16).
#[test]
fn probe_ldc_u16_matches_hardware() {
    probe_bitwise("probes/ldc_u16", "good", "ptx");
    probe_bitwise("probes/ldc_u16", "good", "sass");
}

/// L-SB-ORDER: a guarded-off STG's read scoreboard is released in queue order, so a wait on it
/// covers an earlier untracked STG's operand reads (probes/stg_guard_war: GB300 stores every
/// A intact with the wait, P0 false or true; without it A stores the rewritten registers).
/// Before: a false SCOREBOARD WAR on Triton's TBE forward (`@P5 STG.E.64 [R12.64]`, then R12
/// rewritten after a wait on a later `@P4 STG`'s read scoreboard).
#[test]
fn stg_guarded_off_releases_in_order() {
    let dir = root().join("probes/stg_guard_war");
    for c in ["base", "data_w0", "addr_w0", "data_nw", "addr_nw"] {
        let k = ptxv_sass::load_cubin(&dir.join(format!("{c}.cubin")), "stg_war").unwrap();
        let war = k
            .static_findings
            .iter()
            .any(|f| f.msg.starts_with("SCOREBOARD WAR"));
        assert_eq!(war, c.ends_with("_nw"), "{c}: {:#?}", k.static_findings);
    }
}

/// L-OPREAD: UGETNEXTWORKID consumes its address URs at issue (probes/clc_war: the response
/// address or mbarrier register rewritten by the next instruction, with or without a read
/// scoreboard, does not change where GB300 delivers the response). Before: a false SCOREBOARD
/// WAR on ptxas's TLX CLC kernels (`UMOV UR8, URZ` after `UGETNEXTWORKID.BROADCAST [UR8], [UR9]`
/// with rd=SB2 never waited on).
#[test]
fn clc_war_operands_read_at_issue() {
    let dir = root().join("probes/clc_war");
    for c in ["base", "st1", "st1_rb", "st1_bar"] {
        let k = ptxv_sass::load_cubin(&dir.join(format!("{c}.cubin")), "probe_clc_war").unwrap();
        assert!(
            k.code.iter().any(|i| i.text.contains("UGETNEXTWORKID"))
                && k.static_findings.is_empty(),
            "{c}: {:#?}",
            k.static_findings
        );
    }
}

/// N-FHADD: FHADD / FHFMA without .BF16 (f16 halves, .RZ / .RM / .RP / .SAT, .H1, -c) = GB300
/// on both front-ends (probes/fhadd; the exhaustive run is checked against model.py).
#[test]
fn probe_fhadd_matches_hardware() {
    probe_bitwise("probes/fhadd", "good", "ptx");
    probe_bitwise("probes/fhadd", "good", "sass");
}

/// N-HMNMX2: f16 HMNMX2[.NAN][.XORSIGN] (|b|, immediate pairs, scalar halves) = GB300
/// (probes/hmnmx2; the exhaustive lane-0 run is checked against model.py, see case.toml).
#[test]
fn probe_hmnmx2_matches_hardware() {
    probe_bitwise_cubin("probes/hmnmx2", None);
}

/// L-OPCLASS: DSETP in the f64 queue (a DADD with no write scoreboard read after a wait on a
/// later DSETP's), PTX setp.<cmp>.f64 (probes/f64_dsetp) = GB300 on both front-ends.
#[test]
fn probe_f64_dsetp_matches_hardware() {
    probe_bitwise("probes/f64_dsetp", "good", "ptx");
    probe_bitwise("probes/f64_dsetp", "good", "sass");
}

/// The recorded cubin of a probe holds each of `forms` (the SASS its test is about; `.reuse`
/// marks dropped).
fn probe_cubin_has(dir: &str, forms: &[&str]) {
    let case = Case::load(root().join(dir)).unwrap();
    let k = ptxv_sass::load_cubin(&case.dir.join("kernel.cubin"), &case.m.entry).unwrap();
    let text: Vec<String> = k
        .code
        .iter()
        .map(|i| i.text.replace(".reuse", ""))
        .collect();
    for f in forms {
        assert!(text.iter().any(|t| t.contains(f)), "{dir}: no {f}");
    }
}

/// N-FALU: DSETP.{LE,LTU}.{AND,OR,XOR} Pd, PT, a, b, Pc | !Pc | PT (setp.le.f64 / setp.ltu.f64
/// folded with and.pred / or.pred / xor.pred; b a register pair, a uniform pair, an immediate
/// or RZ) on every pair of 30 f64 patterns, NaNs of both kinds and signs among them, with the
/// predicate both ways = GB300 (probes/dsetp_ltu_le). The SASS front-end runs the recorded
/// ptxas 13.1 cubin: which forms ptxas picks is its choice, and these are the captured ones.
#[test]
fn probe_dsetp_ltu_le_matches_hardware() {
    probe_bitwise("probes/dsetp_ltu_le", "good", "ptx");
    probe_bitwise_cubin("probes/dsetp_ltu_le", Some("kernel.cubin"));
    probe_cubin_has(
        "probes/dsetp_ltu_le",
        &[
            "DSETP.LTU.OR P0, PT, R4, UR6, P0",
            "DSETP.LE.AND P1, PT, R4, 1, PT",
            "DSETP.LTU.AND P1, PT, R4, UR6, PT",
            "DSETP.LE.OR P3, PT, R4, R6, !P1",
            "DSETP.LE.XOR P6, PT, -|R4|, UR6, !P1",
            "DSETP.LTU.XOR P0, PT, R4, 1, P1",
            "DSETP.LTU.OR P6, PT, R4, RZ, P1",
        ],
    );
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

/// L-ROSETTA: PTX lop3.b32 (16 LUTs, immediate operands; ptxas LOP3.LUT) = GB300 on both
/// fronts (probes/lop3_ptx).
#[test]
fn probe_lop3_ptx_matches_hardware() {
    probe_bitwise("probes/lop3_ptx", "good", "ptx");
    probe_bitwise("probes/lop3_ptx", "good", "sass");
}

/// Plain mul.f32x2 read as an fma addend, unpacked into scalar ops, or repacked across lanes
/// into an add.f32x2 (probes/f32x2_unpack): ptxas keeps FMUL2 apart; = GB300 on both fronts.
#[test]
fn probe_f32x2_unpack_matches_hardware() {
    probe_bitwise("probes/f32x2_unpack", "good", "ptx");
    probe_bitwise("probes/f32x2_unpack", "good", "sass");
}

/// Plain mul.f32x2 read through an unpack whose halves only scalar ops read and directly by a
/// plain add / sub.f32x2 (probes/f32x2_unpack_add): ptxas keeps FMUL2 + FADD2, and contracts a
/// second product only the sub reads; = GB300 on both fronts.
#[test]
fn probe_f32x2_unpack_add_matches_hardware() {
    probe_bitwise("probes/f32x2_unpack_add", "good", "ptx");
    probe_bitwise("probes/f32x2_unpack_add", "good", "sass");
}

/// Registers named without `%` (declared by `.reg`, at kernel scope and in a nested scope) on
/// plain mul.f32x2 + add.f32x2 (probes/f32x2_bare): their readers are tracked like `%` names
/// (FFMA2 where only the add reads the product); = GB300 on both fronts.
#[test]
fn probe_f32x2_bare_matches_hardware() {
    probe_bitwise("probes/f32x2_bare", "good", "ptx");
    probe_bitwise("probes/f32x2_bare", "good", "sass");
}

/// PTX-FMAD-X2-VERSION: a case built by ptxas 13.3 (probes/fmad2_v133, `ptxas_toolkit`), in
/// which only the plain pair contracts. The PTX front under the 13.3 rule and the SASS front
/// on the 13.3 cubin = GB300. The 13.0 rule (every same-mode pair contracts) applied to the
/// same PTX gives different bits (mutant killed); with no known version the front refuses.
#[test]
fn probe_fmad2_v133_version_rule() {
    probe_bitwise("probes/fmad2_v133", "good", "ptx");
    probe_bitwise_cubin("probes/fmad2_v133", Some("kernel.cubin"));
    let case = Case::load(root().join("probes/fmad2_v133")).unwrap();
    let ptx = case.variant_ptx("good").unwrap();
    let k = ptxv_ptx::parse_kernel_for(&ptx, Some(&case.m.entry), Some((13, 0))).unwrap();
    let launch = build_launch(&case, &k).unwrap();
    let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
        .verdict
        .expect("13.0-rule mutant certifies");
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        report.iter().any(|l| !l.contains("bitwise equal")),
        "the 13.0 rule on a 13.3 case must differ from GB300: {report:?}"
    );
    let e = ptxv_ptx::parse_kernel(&ptx, Some(&case.m.entry)).unwrap_err();
    assert!(format!("{e:#}").contains("PTX-FMAD"), "{e:#}");
}

/// mad.lo.s16 / .u16 (low 16 bits of a * b + c; register and immediate operands, d aliasing
/// c) on random operands (probes/mad_lo16) = GB300 on both fronts.
#[test]
fn probe_mad_lo16_matches_hardware() {
    probe_bitwise("probes/mad_lo16", "good", "ptx");
    probe_bitwise("probes/mad_lo16", "good", "sass");
}

/// setp.<cmp>.<bop>.f32 with a plain or negated combining predicate, ordered and unordered
/// compares on NaN / signed-zero / subnormal operands, one .ftz form (probes/setp_bop) =
/// GB300 on both fronts.
#[test]
fn probe_setp_bop_matches_hardware() {
    probe_bitwise("probes/setp_bop", "good", "ptx");
    probe_bitwise("probes/setp_bop", "good", "sass");
}

/// applypriority[.global].L2::evict_normal (SASS CCTL.E.DML2) and
/// prefetch.global.L2::evict_last (CCTL.E.PML2) are no-ops: stores, the hints on each warp's
/// line, cross-warp reads after a barrier (probes/applypriority) = GB300 on both fronts.
#[test]
fn probe_applypriority_matches_hardware() {
    probe_bitwise("probes/applypriority", "good", "ptx");
    probe_bitwise("probes/applypriority", "good", "sass");
}

/// redux.sync min / max [.abs] [.NaN] .f32 with full mask reached through divergent paths,
/// over NaN / signed-zero / inf / subnormal lanes (probes/redux_f32_paths): the collective
/// waits for every lane, the case certifies (C-WARP) and = GB300 on both fronts.
#[test]
fn probe_redux_f32_paths_matches_hardware() {
    probe_bitwise("probes/redux_f32_paths", "good", "ptx");
    probe_bitwise("probes/redux_f32_paths", "good", "sass");
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

/// tcgen05.mma `[taddr + off]` D and A-in-TMEM operands (probes/tc_mma_off) = GB300 (the
/// overlapping un-waited stores are E-TMEM-QUEUE, `probe_bitwise_queue`; PTX: MISS 116).
#[test]
fn probe_tc_mma_off_matches_hardware() {
    probe_bitwise_queue("probes/tc_mma_off", "good", "ptx");
    probe_bitwise_queue("probes/tc_mma_off", "good", "sass");
}

/// FFMA2 with `-URn.F32` / `-URn.F32x2.HI_LO` operands (probes/ffma2_ur) = GB300.
#[test]
fn probe_ffma2_ur_matches_hardware() {
    probe_bitwise("probes/ffma2_ur", "good", "ptx");
    probe_bitwise("probes/ffma2_ur", "good", "sass");
}

/// L-ROSETTA: `.ftz` on packed f32x2 fma / add / sub / mul in every rounding mode (FFMA2 /
/// FADD2 / FMUL2 .FTZ[.RZ|.RM|.RP]) on boundary, subnormal and edge operands = GB200
/// (probes/f32x2_ftz). Mutant: the same kernel without `.ftz` differs from the hardware.
#[test]
fn probe_f32x2_ftz_matches_hardware() {
    probe_bitwise("probes/f32x2_ftz", "good", "ptx");
    probe_bitwise("probes/f32x2_ftz", "good", "sass");
    let case = Case::load(root().join("probes/f32x2_ftz")).unwrap();
    let ptx = case.variant_ptx("good").unwrap().replace(".ftz", "");
    let k = ptxv_ptx::parse_kernel_for(
        &ptx,
        Some(&case.m.entry),
        case.m
            .ptxas_toolkit
            .as_deref()
            .and_then(ptxv_ptx::toolkit_version),
    )
    .unwrap();
    let launch = build_launch(&case, &k).unwrap();
    let v = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)))
        .verdict
        .unwrap();
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        report.iter().any(|l| !l.contains("bitwise equal")),
        "{report:?}"
    );
}

/// S-TMEM, N-MMA: tcgen05.mma kind::f16 M = 128 with A in TMEM and N = 8 / 24 (N % 16 = 8,
/// the FP8-KV decode's P.V shape) writes D columns col0..col0 + N of every lane = GB200
/// (probes/tc_mma_ts_n8, n24/: D equals GB300's N = 16 output on the shared columns). The
/// PTX ISA requires N % 16 == 0 with A in TMEM: a `contract` finding (and `tmem-queue` ones:
/// the sentinel fill and the A store overlap with no wait between, E-TMEM-QUEUE).
#[test]
fn probe_tc_mma_ts_n8_matches_hardware() {
    for dir in ["probes/tc_mma_ts_n8", "probes/tc_mma_ts_n8/n24"] {
        for front in ["ptx", "sass"] {
            let case = Case::load(root().join(dir)).unwrap();
            let k = kernel(&case, "good", front);
            let launch = build_launch(&case, &k).unwrap();
            let m = Machine::new(Arc::new(k), Arc::new(launch));
            let v = ptxv_sim::wellsync::certify(m)
                .verdict
                .unwrap_or_else(|| panic!("{dir} {front} certifies"));
            assert_eq!(
                v.classes(),
                ["contract"],
                "{dir} {front}: {:#?}",
                v.findings
            );
            assert!(
                v.findings
                    .iter()
                    .all(|f| f.sig.starts_with("mma-ts-shape:") || f.sig.starts_with("tmem-queue:")),
                "{dir} {front}: {:#?}",
                v.findings
            );
            let report = case.expect_report(&v.outputs[0]).unwrap();
            assert!(
                report.iter().all(|l| l.contains("bitwise equal")),
                "{dir} {front}: {report:?}"
            );
        }
    }
}

/// S-TMEM, N-MMA-BS: block-scaled kind::mxf8f6f4 with A in TMEM (byte k of a row's K slice at
/// column k / 4, byte k % 4) = GB200, which equals the A-from-shared-memory run
/// (probes/mma_bs/ts).
#[test]
fn probe_mma_bs_tmem_a_matches_hardware() {
    probe_bitwise("probes/mma_bs/ts", "good", "ptx");
    probe_bitwise("probes/mma_bs/ts", "good", "sass");
}

/// S-TMA, N-MMA-BS, E-PAIR: block-scaled kind::mxf8f6f4 with MN-major 8-bit operands
/// (swizzled 128 / 64 / 32 B and unswizzled; A, B, both; a start 512 B into a swizzle atom),
/// one CTA and cta_group::2 (A in TMEM or shared memory) = GB200, whose D equals the K-major
/// layout's for every case (probes/mma_bs/mn).
/// One test per case (and one for the refused forms).
fn probe_mma_bs_mn_major(c: &str) {
    let dir = format!("probes/mma_bs/mn/{c}");
    // PTX cta_group::2: the leader's commit waited at .cta scope in the peer is an
    // S-MBAR-SCOPE contract finding; every schedule still matches GB300.
    if c.starts_with("pair_") {
        probe_bitwise_mbar_contract(&dir, "good");
    } else {
        probe_bitwise(&dir, "good", "ptx");
    }
    probe_bitwise(&dir, "good", "sass");
}
#[test]
fn probe_mma_bs_mn_major_matches_hardware_kmajor() {
    probe_mma_bs_mn_major("kmajor");
}
#[test]
fn probe_mma_bs_mn_major_matches_hardware_a_sw128() {
    probe_mma_bs_mn_major("a_sw128");
}
#[test]
fn probe_mma_bs_mn_major_matches_hardware_a_sw128_off512() {
    probe_mma_bs_mn_major("a_sw128_off512");
}
#[test]
fn probe_mma_bs_mn_major_matches_hardware_a_sw64() {
    probe_mma_bs_mn_major("a_sw64");
}
#[test]
fn probe_mma_bs_mn_major_matches_hardware_a_sw32() {
    probe_mma_bs_mn_major("a_sw32");
}
#[test]
fn probe_mma_bs_mn_major_matches_hardware_a_none() {
    probe_mma_bs_mn_major("a_none");
}
#[test]
fn probe_mma_bs_mn_major_matches_hardware_b_sw64() {
    probe_mma_bs_mn_major("b_sw64");
}
#[test]
fn probe_mma_bs_mn_major_matches_hardware_b_sw32() {
    probe_mma_bs_mn_major("b_sw32");
}
#[test]
fn probe_mma_bs_mn_major_matches_hardware_ab_sw128_sw64() {
    probe_mma_bs_mn_major("ab_sw128_sw64");
}
#[test]
fn probe_mma_bs_mn_major_matches_hardware_pair_ts_kmajor() {
    probe_mma_bs_mn_major("pair_ts_kmajor");
}
#[test]
fn probe_mma_bs_mn_major_matches_hardware_pair_ts_b_sw64() {
    probe_mma_bs_mn_major("pair_ts_b_sw64");
}
#[test]
fn probe_mma_bs_mn_major_matches_hardware_pair_ts_b_sw32() {
    probe_mma_bs_mn_major("pair_ts_b_sw32");
}
#[test]
fn probe_mma_bs_mn_major_matches_hardware_pair_ts_b_none() {
    probe_mma_bs_mn_major("pair_ts_b_none");
}
#[test]
fn probe_mma_bs_mn_major_matches_hardware_pair_ss_kmajor() {
    probe_mma_bs_mn_major("pair_ss_kmajor");
}
#[test]
fn probe_mma_bs_mn_major_matches_hardware_pair_ss_a_sw128() {
    probe_mma_bs_mn_major("pair_ss_a_sw128");
}
#[test]
fn probe_mma_bs_mn_major_matches_hardware_pair_ss_ab() {
    probe_mma_bs_mn_major("pair_ss_ab");
}
/// Refused (not measured): an MN-major start inside a swizzle row, MN-major FP4.
#[test]
fn probe_mma_bs_mn_major_refused_forms() {
    for v in ["mid_row", "fp4_mn"] {
        for front in ["ptx", "sass"] {
            check_dir("probes/mma_bs/mn", "a_sw128", v, front);
        }
    }
}

/// S-RED-ASYNC: ptxas 13.1 encodes red.async.global `min.s32` / `inc.u32` / `or.b32` as
/// UREDGR.E.ADD; GB200 adds (probes/red_async_ops), as the SASS front does. The PTX front
/// refuses those ops.
#[test]
fn probe_red_async_ops_add() {
    probe_bitwise("probes/red_async_ops", "good", "sass");
    let case = Case::load(root().join("probes/red_async_ops")).unwrap();
    let ptx = case.variant_ptx("good").unwrap();
    assert!(
        ptxv_ptx::parse_kernel_for(
            &ptx,
            Some(&case.m.entry),
            case.m
                .ptxas_toolkit
                .as_deref()
                .and_then(ptxv_ptx::toolkit_version)
        )
        .is_err()
    );
}

/// cp.async.bulk.prefetch.tensor (SASS UTMAPF.L2.2D) as a cache hint (probes/tma_prefetch)
/// = GB300.
#[test]
fn probe_tma_prefetch_matches_hardware() {
    probe_bitwise("probes/tma_prefetch", "good", "ptx");
    probe_bitwise("probes/tma_prefetch", "good", "sass");
}

/// ATOMG / REDG .E.ADD.BF16x4.RN (atom / red .add.noftz.v4.bf16) = two BF16x2 atomics per 8
/// bytes (probes/atom_bf16x4) = GB300 on the SASS front (old values and final words).
#[test]
fn probe_atom_bf16x4_matches_hardware() {
    probe_bitwise("probes/atom_bf16x4", "good", "sass");
}

/// FMUL.D2 .. .M8 (.FTZ, .SAT) = one rounding of the scaled product, and the PTX front's
/// PTX-FMUL-SCALE fold of `mul.f32` + `mul` by 0.125 .. 8 (probes/fmul_scale, MISS 55) = GB300
/// on both fronts (ptxas 13.0 cubin).
#[test]
fn probe_fmul_scale_matches_hardware() {
    probe_bitwise("probes/fmul_scale", "good", "ptx");
    probe_bitwise("probes/fmul_scale", "good", "sass");
}

/// div.approx.ftz.f32 (probes/div_approx_ftz): MUFU.RCP + FMUL.FTZ, the product contracted
/// with a following add.ftz / sub.ftz and folded with a mul.ftz by 0.5 / 2.0 as ptxas does;
/// 16 shapes x 1024 inputs = GB300 bitwise on both fronts.
#[test]
fn probe_div_approx_ftz_matches_hardware() {
    probe_bitwise("probes/div_approx_ftz", "good", "ptx");
    probe_bitwise("probes/div_approx_ftz", "good", "sass");
}

/// atom.add.f64 / red.add.f64 = ATOMG / REDG .E.ADD.F64.RN (probes/atom_f64) = GB300. The
/// red adds to shared words do not commute for the reductions (f64 order changes bits), so the
/// stateful explorer checks every order: one output, GB300's.
#[test]
#[ignore = "heavy: 37 s"]
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
/// mul, adds of two products, a mul.ftz through neg.f32 / mov into add.ftz / sub.ftz [.sat]
/// (MISS 62) = GB300 on both fronts.
#[test]
fn probe_fmad_multi_matches_hardware() {
    probe_bitwise("probes/fmad_multi", "good", "ptx");
    probe_bitwise("probes/fmad_multi", "good", "sass");
}

/// MISS 62: a mul.ftz.f32 read only by neg.f32, whose result add.ftz / sub.ftz [.sat] reads, is
/// one FFMA.FTZ on ptxas 13.0 / 13.3 (probes/fmad_ftz_neg: each form its own operands, half of
/// them cancelling) = GB300 on both fronts. Before, the PTX front followed a neg only for a mul
/// without .ftz and kept the FMUL.
#[test]
fn probe_fmad_ftz_neg_matches_hardware() {
    probe_bitwise("probes/fmad_ftz_neg", "good", "ptx");
    probe_bitwise("probes/fmad_ftz_neg", "good", "sass");
}

/// min / max [.NaN] [.ftz] .abs.f32 with three sources (FMNMX3 |a|, |b|, |c|) on every triple
/// of 16 edge patterns = GB300 on both fronts (probes/fmnmx3_abs).
#[test]
fn probe_fmnmx3_abs_matches_hardware() {
    probe_bitwise("probes/fmnmx3_abs", "good", "ptx");
    probe_bitwise("probes/fmnmx3_abs", "good", "sass");
}

/// kind::mxf8f6f4 with A in TMEM / B MN-major and with A MN-major = the K-major path = GB300 on
/// both fronts (probes/mma_mxf8_tmem).
#[test]
fn probe_mma_mxf8_tmem_matches_hardware() {
    probe_bitwise("probes/mma_mxf8_tmem", "good", "ptx");
    probe_bitwise("probes/mma_mxf8_tmem", "good", "sass");
}

/// red.async.release.gpu.global.add as a release flag (probes/red_async) = GB300 on both
/// fronts; the .relaxed form is not a PTX ISA `.global` form (refused), while ptxas's UREDGR releases.
#[test]
fn probe_red_async_matches_hardware() {
    // A cross-CTA poll: explored (not certifiable as well-synchronized).
    let case = Case::load(root().join("probes/red_async")).unwrap();
    for front in ["ptx", "sass"] {
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 2_000_000);
        assert!(v.complete, "{front}");
        assert_eq!(v.classes(), ["clean"], "{front}: {:#?}", v.findings);
        for out in &v.outputs {
            let report = case.expect_report(out).unwrap();
            assert!(
                report.iter().all(|l| l.contains("bitwise equal")),
                "{front}: {report:?}"
            );
        }
    }
    // `.relaxed` on `.global` is not a PTX ISA form (`.relaxed` needs `.cluster` scope,
    // `.shared::cluster` and an mbarrier): the PTX front refuses it; the SASS ptxas emits for it
    // is the release's, clean.
    let relaxed = case.variant_ptx("relaxed").unwrap();
    assert!(ptxv_ptx::parse_kernel(&relaxed, Some(&case.m.entry)).is_err());
    check_dir("probes", "red_async", "relaxed", "sass");
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

/// N-FALU: FSET.BF.NAN.AND Rd, a, b, Pc | !Pc | PT (set.nan[.and].f32.f32, and setp.nan.f32 +
/// selp.f32 1.0, 0.0), 1.0f when an operand is a NaN and the predicate holds, on every pair
/// of 22 f32 patterns with the predicate both ways = GB300 (probes/fset_nan): the PTX
/// front-end, and the SASS front-end on the recorded ptxas 13.1 cubin.
#[test]
fn probe_fset_nan_matches_hardware() {
    probe_bitwise("probes/fset_nan", "good", "ptx");
    probe_bitwise_cubin("probes/fset_nan", Some("kernel.cubin"));
    probe_cubin_has(
        "probes/fset_nan",
        &[
            "FSET.BF.NAN.AND R9, R0, R0, PT",
            "FSET.BF.NAN.AND R15, R0, UR6, PT",
            "FSET.BF.NAN.AND R17, R0, 20, PT",
            "FSET.BF.NAN.AND R19, |R0|, -R7, PT",
            "FSET.BF.NAN.AND R21, R0, R7, P0",
            "FSET.BF.NAN.AND R7, R7, R0, !P0",
            "FSET.BF.NAN.AND R23, R7, R0, PT",
        ],
    );
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
        let m0 = Machine::new(Arc::new(k), Arc::new(launch));
        let mut v = explore(m0.clone(), 2_000_000);
        assert!(v.complete, "{front}: incomplete");
        // PTX: the multicast response's complete-tx waited at .cta scope (S-MBAR-SCOPE).
        ptxv_sim::twolevel::mbar_async_contract(&mut v, &m0, &mut |m| explore(m, 2_000_000));
        let want: &[&str] = if front == "ptx" {
            &["contract", "nondeterministic"]
        } else {
            &["nondeterministic"]
        };
        assert_eq!(v.classes(), want, "{front}: {:#?}", v.findings);
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
                    // {2, 0, 0, 0} or all zero (probes/clc_one_cluster).
                    assert!(
                        a[2..7] == [0, 0, 0, 0, 0] || a[2..7] == [0, 2, 0, 0, 0],
                        "{front}: failure: {w:?}"
                    );
                }
            }
        }
        assert!(
            cancels > 0 && fails > 0,
            "{front}: {cancels} cancels, {fails} failures"
        );
    }
}

/// A multicast try_cancel in a grid of one cluster (probes/clc_one_cluster): nothing to cancel;
/// the failed response is {2, 0, 0, 0} (GB300's output, one of the model's outcomes) or all
/// zero, on both fronts, with every CTA launched up front (E-CLC eager launch).
#[test]
fn probe_clc_one_cluster_follows_hardware() {
    let case = Case::load(root().join("probes/clc_one_cluster")).unwrap();
    for front in ["ptx", "sass"] {
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let m0 = Machine::new(Arc::new(k), Arc::new(launch));
        let mut v = explore(m0.clone(), 2_000_000);
        assert!(v.complete, "{front}: incomplete");
        // PTX: the multicast response's complete-tx waited at .cta scope (S-MBAR-SCOPE).
        ptxv_sim::twolevel::mbar_async_contract(&mut v, &m0, &mut |m| explore(m, 2_000_000));
        let want: &[&str] = if front == "ptx" {
            &["contract", "nondeterministic"]
        } else {
            &["nondeterministic"]
        };
        assert_eq!(v.classes(), want, "{front}: {:#?}", v.findings);
        assert_eq!(v.outputs.len(), 2, "{front}");
        let gpu = v.outputs.iter().filter(|o| {
            case.expect_report(o)
                .unwrap()
                .iter()
                .all(|l| l.contains("bitwise equal"))
        });
        assert_eq!(
            gpu.count(),
            1,
            "{front}: GB300's output is one of the outcomes"
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

fn r4a_machine(case: &Case, variant: &str) -> Machine {
    let k = binary_launches(case, variant);
    let launch = build_launch(case, &k).unwrap();
    Machine::new(Arc::new(k), Arc::new(launch))
}

// Rung 4a (docs/rung4.md): one call of torch's one-shot and two-shot all-reduce on 3 GPUs,
// explored exhaustively (clean, the one-shot outputs equal every rank's GPU bytes, with and
// without E-EXIT-FENCE): wellsync.rs `r4a_certified_first_call_matches_exploration`, which
// runs the same explorations and certifies them too.

/// Rung 4a: without E-EXIT-FENCE (the strict PTX model) the one-shot's closing barrier does
/// not order a peer's reads of call 1 before this rank's restaging in call 2: the waiting side
/// of torch's closing barrier has no acquire (`sync_remote_blocks<memory_order_release>`).
#[test]
#[ignore = "heavy: bounded 3M-state explore; 342 s"]
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
/// hundreds of millions of states). One test per mutant.
fn r4a_mutant(c: &str, v: &str) {
    let case = Case::load(root().join("fixtures").join(c)).unwrap();
    let r = explore(r4a_machine(&case, v), 3_000_000);
    let expect = case.variant(v).unwrap().expect_for("sass");
    assert!(
        r.meets(&expect),
        "{c}/{v}: got {:?} (complete: {}), want {expect:?}",
        r.classes(),
        r.complete
    );
}
#[test]
#[ignore = "heavy: bounded 3M-state explore of the 2-call case; 259 s"]
fn r4a_mutants_b57() {
    r4a_mutant("r4a_oneshot", "B57");
}
#[test]
#[ignore = "heavy: bounded 3M-state explore of the 2-call case; 260 s"]
fn r4a_mutants_b58() {
    r4a_mutant("r4a_oneshot", "B58");
}
#[test]
#[ignore = "heavy: bounded 3M-state explore of the 2-call case; 256 s"]
fn r4a_mutants_b59() {
    r4a_mutant("r4a_oneshot", "B59");
}
#[test]
#[ignore = "heavy: bounded 3M-state explore of the 2-call case; 320 s"]
fn r4a_mutants_b60() {
    r4a_mutant("r4a_oneshot", "B60");
}
#[test]
#[ignore = "heavy: bounded 3M-state explore of the 2-call case; 22 s"]
fn r4a_mutants_b61() {
    r4a_mutant("r4a_oneshot", "B61");
}
#[test]
fn r4a_mutants_b62() {
    r4a_mutant("r4a_twoshot", "B62");
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
#[ignore = "heavy: 91 s"]
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
    // S-NC (violation): cached_notify_combine reads send_head with __ldg (LDG.E.CONSTANT
    // 0x1290) and the same kernel writes those entries (STG 0x1780); the read-only data
    // cache requires data read-only for the kernel's lifetime (PTX ISA 9.5 section 8.1).
    assert_eq!(
        kinds,
        ["contract", "race", "violation"].into_iter().collect(),
        "{fs:?}"
    );
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
    // S-NC: cached_notify_combine's __ldg of entries it writes, as above.
    assert_eq!(
        kinds,
        ["contract", "violation"].into_iter().collect(),
        "{fs:?}"
    );
}

/// B32-B40: each mutant's expected class shows up in the deterministic run (races are
/// happens-before facts, so one schedule that reaches both accesses reports them). One test
/// per mutant.
fn r4b_moe_mutant(v: &str) {
    if vendor_missing(&["r4b_moe"]) {
        return;
    }
    let case = Case::load(root().join("fixtures/r4b_moe")).unwrap();
    let (_, fs, ok) = r4b_run(&case, v);
    let mut kinds: Vec<String> = fs.iter().map(|f| f.kind.clone()).collect();
    if !ok {
        kinds.push("deadlock".into());
    }
    for e in &case.variant(v).unwrap().expect {
        assert!(kinds.contains(e), "{v}: want {e}, got {kinds:?}");
    }
}
#[test]
#[ignore = "heavy: a run of the 28-launch MoE layer; 35 s"]
fn r4b_moe_mutants_b32() {
    r4b_moe_mutant("B32");
}
#[test]
#[ignore = "heavy: a run of the 28-launch MoE layer; 42 s"]
fn r4b_moe_mutants_b33() {
    r4b_moe_mutant("B33");
}
#[test]
#[ignore = "heavy: a run of the 28-launch MoE layer; 89 s"]
fn r4b_moe_mutants_b34() {
    r4b_moe_mutant("B34");
}
#[test]
#[ignore = "heavy: a run of the 28-launch MoE layer; 7 s"]
fn r4b_moe_mutants_b35() {
    r4b_moe_mutant("B35");
}
#[test]
#[ignore = "heavy: a run of the 28-launch MoE layer; 45 s"]
fn r4b_moe_mutants_b36() {
    r4b_moe_mutant("B36");
}
#[test]
#[ignore = "heavy: a run of the 28-launch MoE layer; 38 s"]
fn r4b_moe_mutants_b37() {
    r4b_moe_mutant("B37");
}
#[test]
#[ignore = "heavy: a run of the 28-launch MoE layer; 18 s"]
fn r4b_moe_mutants_b38() {
    r4b_moe_mutant("B38");
}
#[test]
#[ignore = "heavy: a run of the 28-launch MoE layer; 45 s"]
fn r4b_moe_mutants_b39() {
    r4b_moe_mutant("B39");
}
#[test]
#[ignore = "heavy: a run of the 28-launch MoE layer; 46 s"]
fn r4b_moe_mutants_b40() {
    r4b_moe_mutant("B40");
}

/// The dependency claim: output token t of rank s depends exactly on its top-k experts.
/// Perturbing expert e's weights changes exactly the rows of the tokens routed to e, on
/// every rank (and the outputs stay a deterministic function of the inputs).
#[test]
#[ignore = "heavy: 3 runs of the MoE layer; 225 s"]
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
#[ignore = "heavy: 39 s"]
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
#[ignore = "heavy: two exhaustive group checks at 1M; 374 s"]
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
#[ignore = "heavy: 32 s"]
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
#[ignore = "heavy: group check on the 28-launch r4b layer; 2540 s"]
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
        "pr3-swap",
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
#[ignore = "heavy: 81 s"]
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
    // S-NC (violation, as in r4b_moe_run_matches_the_gpu): cached_notify_combine reads send_head with __ldg (LDG.E.CONSTANT
    // 0x1290) and the same kernel writes those entries (STG 0x1780); the read-only data
    // cache requires data read-only for the kernel's lifetime (PTX ISA 9.5 section 8.1).
    assert_eq!(
        kinds,
        ["contract", "violation"].into_iter().collect(),
        "{fs:?}"
    );
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
#[ignore = "heavy: 73 s"]
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
    r4c_keep(r4a_machine(&vcase, variant), keep)
}

/// `m` with only the launches in `keep` live (the others' threads exited).
fn r4c_keep(mut m: Machine, keep: &[usize]) -> Machine {
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
#[ignore = "heavy: 10-launch machine, one run; 108 s"]
fn r4c_dispatch_splits_into_flows() {
    if vendor_missing(&["r4c_nowrap"]) {
        return;
    }
    let case = Case::load(root().join("fixtures/r4c_nowrap")).unwrap();
    let full = r4c_machine(&case, "good", &(0..10).collect::<Vec<_>>());
    // Built once: the run's machine is the full one with launches 4-9 exited.
    let m = r4c_keep(full.clone(), &[0, 1, 2, 3]);
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
        ("pr3-swap", true),
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
                    // pr3, pr3-swap: the reload in the poll loop and the read after it.
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
        // PTX: a warp-order finding (below) leaves the decision to the explorer.
        let o = ptxv_sim::wellsync::certify(m.clone());
        match o.verdict {
            Some(v) => v,
            None if front == "ptx" && o.reasons.iter().any(|r| r.contains("converged warp")) => {
                explore(m, 2_000_000)
            }
            None => panic!("{variant}/{front}: the certificate does not hold"),
        }
    } else {
        explore(m, 2_000_000)
    };
    // E-WARP-ORDER (PTX): unordered strong pairs of a converged warp (red.min / red.max of
    // one word by two lanes) are nondeterministic under the PTX model; GB300 runs them in
    // program order, which is the explored outcome the bytes are compared with.
    let warp_order_only = front == "ptx"
        && v.classes() == ["nondeterministic"]
        && v.findings
            .iter()
            .filter(|f| f.kind == "nondeterministic")
            .all(|f| f.sig.starts_with("warp-order:"));
    assert!(
        v.complete && (v.classes() == ["clean"] || warp_order_only),
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
#[ignore = "heavy: 40 s"]
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
// One test per (variant, front). The racy pre-fix: the explorers agree only up to the race
// (E-DRF), so stateful only. SASS pre-fix: LDS vs ATOMS is a race kept by the strong re-check
// (S-SMEM-STRONG); its schedules exceed 2M states, so the test asks only for the finding.
#[test]
#[ignore = "heavy: stateful and DPOR explores at 2M; 10 s"]
fn e_fmax_fold_good_ptx() {
    check("e_fmax_fold", "good", "ptx");
}
#[test]
#[ignore = "heavy: stateful and DPOR explores at 2M; 12 s"]
fn e_fmax_fold_good_sass() {
    check("e_fmax_fold", "good", "sass");
}
#[test]
#[ignore = "heavy: explore at 2M; 8 s"]
fn e_fmax_fold_prefix_ptx() {
    check_stateful("e_fmax_fold", "prefix", "ptx", 2_000_000);
}
#[test]
#[ignore = "heavy: explore at 2M; 11 s"]
fn e_fmax_fold_good_3tiles_ptx() {
    check_stateful("e_fmax_fold", "good_3tiles", "ptx", 2_000_000);
}
#[test]
#[ignore = "heavy: explore at 2M; 120 s"]
fn e_fmax_fold_prefix_3tiles_ptx() {
    check_stateful("e_fmax_fold", "prefix_3tiles", "ptx", 2_000_000);
}
#[test]
#[ignore = "heavy: explore at 2M; 10 s"]
fn e_fmax_fold_good_3tiles_sass() {
    check_stateful("e_fmax_fold", "good_3tiles", "sass", 2_000_000);
}
#[test]
#[ignore = "heavy: bounded explore at 200k; 25 s"]
fn e_fmax_fold_prefix_sass() {
    check_finds("e_fmax_fold", "prefix", "sass", 200_000);
}
#[test]
#[ignore = "heavy: bounded explore at 200k; 25 s"]
fn e_fmax_fold_prefix_3tiles_sass() {
    check_finds("e_fmax_fold", "prefix_3tiles", "sass", 200_000);
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
#[ignore = "heavy: 38 s"]
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

/// S-MBAR-SCOPE (fixtures/e_mbar_scope): a 2-CTA hand-off through a remote mbarrier arrive
/// synchronizes only when the arrive and the wait are morally strong (both at cluster scope).
/// A .cta-scoped arrive or wait across CTAs completes the phase but orders nothing; so does a
/// fence.acq_rel.cluster before a .cta arrive (PTX front). These variants were clean before.
#[test]
fn e_mbar_scope_handoff() {
    for v in [
        "good", "PFC", "PFCW", "PFCA", "FEN", "FRC", "STA", "STAC", "NOF", "PCTA",
    ] {
        both("e_mbar_scope", v);
    }
}

/// S-MBAR-SCOPE (fixtures/e_mbar_mailbox): a peer's st.async complete-tx pairs only with a
/// cluster-scope acquire in the owner (wait at .cluster with .acquire, or a relaxed wait and
/// fence.acquire.sync_restrict::shared::cluster.cluster).
#[test]
fn e_mbar_mailbox_scopes() {
    for v in ["good", "MB", "MBNF", "MBCA"] {
        both("e_mbar_mailbox", v);
    }
}

/// S-MBAR-SCOPE (fixtures/e_mbar_restrict): fence.release.sync_restrict::shared::cta.cluster +
/// a relaxed cluster arrive releases the peer's shared-memory accesses at cluster scope, and
/// nothing else.
#[test]
fn e_mbar_restrict_slot_reuse() {
    for v in ["good", "NOF", "RLX", "GM", "GMF"] {
        both("e_mbar_restrict", v);
    }
}

/// S-PROXY for async-proxy writes, MISS 54 (fixtures/e_mbar_lww): a CTA's generic store, a
/// remote arrive, and the peer's bulk copy over the same shared-memory bytes. Happens-before
/// alone (SASS `MEMBAR.ALL.CTA` before the bare arrive, LWWC; a cluster release, LWWK) is a
/// proxy race without a shared-memory proxy fence after the store: GB300 let the store land
/// after the copy (probes/handoff_window v=11). LWWC was clean on the SASS front before.
#[test]
fn e_mbar_lww_proxy_write() {
    for v in ["good", "LWW", "LWWC", "LWWP", "LWWK"] {
        both("e_mbar_lww", v);
    }
}

/// S-PROXY-WAR (fixtures/e_mbar_lww WAR / WARF): a generic read, a cluster release, then the
/// peer's bulk copy over the same shared-memory bytes, with no proxy fence after the read, is a
/// contract finding at the strict level on both fronts (not a race, not silent), and the GB300
/// profile discharges it (E-HW-PROXY-WAR, probes/handoff_window v=16/17). With the fence: clean.
#[test]
fn e_mbar_lww_proxy_war_contract() {
    for v in ["WAR", "WARF"] {
        both("e_mbar_lww", v);
    }
    assert_eq!(sass_gb300("e_mbar_lww", "WAR"), "certified");
    assert_eq!(sass_gb300("e_mbar_lww", "WARF"), "certified");
}

/// S-PROXY-WAR, which read is reported (fixtures/e_mbar_lww WAR2): rank 0 reads bytes 4-7, a
/// proxy fence, then bytes 12-15 and 8-11 at one (actor, clock); the peer's bulk copy covers
/// all 16 bytes. The check decides a read's coverage once per (actor, clock); the finding is the
/// first uncovered read in byte order (bytes 8-11), not the first in program order (12-15) and
/// not the fenced read (4-7). Both fronts give a contract; GB300 discharges it.
#[test]
fn e_mbar_lww_proxy_war_reports_the_first_uncovered_byte() {
    both("e_mbar_lww", "WAR2");
    assert_eq!(sass_gb300("e_mbar_lww", "WAR2"), "certified");
    let case = Case::load(root().join("fixtures/e_mbar_lww")).unwrap();
    let k = kernel(&case, "WAR2", "ptx");
    let pc = |what: &str| {
        let v: Vec<usize> = (0..k.code.len())
            .filter(|&i| k.code[i].text.contains(what))
            .collect();
        assert_eq!(v.len(), 1, "{what}: {v:?}");
        v[0]
    };
    let (r8, r12, copy) = (pc("[%r4+8]"), pc("[%r4+12]"), pc("cp.async.bulk"));
    assert!(r12 < r8, "program order: bytes 12-15 read first");
    let launch = build_launch(&case, &k).unwrap();
    let r = explore(Machine::new(Arc::new(k), Arc::new(launch)), 100_000);
    assert!(r.complete);
    let sigs: std::collections::BTreeSet<&str> = r
        .findings
        .iter()
        .filter(|x| x.sig.starts_with("proxy-war:"))
        .map(|x| x.sig.as_str())
        .collect();
    let want = format!("proxy-war:{r8}:{copy}");
    assert_eq!(sigs, [want.as_str()].into(), "{:#?}", r.findings);
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

/// S-INIT per byte of narrow conversions on the PTX front (fixtures/e_uninit_cvt_lanes): the
/// defined low byte of an e4m3x2 pair and of a narrowed word of e2m1x2 bytes, stored with
/// st.b8, is clean; storing the 16-bit value with its undefined byte is reported.
#[test]
fn cvt_pack_bytes_keep_their_own_undefined_bits() {
    for v in ["good", "store_pair", "store_packed"] {
        both("e_uninit_cvt_lanes", v);
    }
}

/// PTX vector operands (fixtures/e_vec_regs): `Op::dst_regs` / `src_regs` name every element
/// register, so each analysis built on them agrees between the PTX and SASS fronts:
/// verdicts and outputs (explorer, DPOR, S-INIT poison of mma.sync's four accumulator
/// registers), input coverage (element 4's comparison and address), the input-data taint
/// (E-CTA-REP / N-EQUIV precondition) and the constant-guard analysis (E-CONST-BRA).
#[test]
fn vector_operand_registers_ptx_sass_parity() {
    for v in ["good", "mma"] {
        both("e_vec_regs", v);
    }
    let case = Case::load(root().join("fixtures/e_vec_regs")).unwrap();
    for front in ["ptx", "sass"] {
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let m = Machine::new(Arc::new(k), Arc::new(launch));
        // Input coverage: element 4 (in[4t+3], bytes 12..511) is compared and addresses the
        // store.
        let c = ptxv_sim::coverage::analyze(&m);
        assert_eq!(c.comparisons.len(), 1, "{front}: {:#?}", c.comparisons);
        let i = &c.comparisons[0].inputs[0];
        assert_eq!(
            (i.buffer.as_str(), i.bytes),
            ("in", Some([12, 511])),
            "{front}"
        );
        assert!(
            !c.input_independent && !c.input_dependent.is_empty(),
            "{front}: {c:#?}"
        );
        // The input-data taint (rep.rs data run): the address depends on loaded bytes.
        let mut mt = m.clone();
        let mut ns = ptxv_sim::num::NumState::with_mma_eps(mt.launch.mma_model.eps());
        ns.taint.all_global = true;
        mt.num = Some(Box::new(ns));
        let (mt, _, ok) = ptxv_sim::explore::run_once(mt);
        assert!(ok);
        let tv = &mt.num.as_ref().unwrap().taint.violations;
        assert!(
            tv.iter().any(|v| v.contains("depends on input data")),
            "{front}: {tv:?}"
        );
        // E-CONST-BRA: the branch on element 4 is not folded (0 before the load); ptxas
        // predicates the store instead of branching.
        let bra =
            m.k.code
                .iter()
                .position(|i| matches!(i.op, ptxv_ir::Op::Bra { .. }) && !i.guard.is_always());
        assert!(bra.is_some() || front == "sass");
        if let Some(bra) = bra {
            assert_eq!(m.bar_cfg.fold[bra], None, "{front}: {}", m.k.describe(bra));
        }
    }
}

/// S-INIT through 16-bit-source conversions (fixtures/e_uninit_half_cvt): F2F.F64.BF16 reads
/// only the low half that ptxas's PRMT zeroed (or the predicated LDG.E.U16 wrote), not the
/// unwritten high half.
#[test]
fn half_source_conversion_reads_its_half() {
    both("e_uninit_half_cvt", "good");
}

/// E-CPASYNC-EXIT (fixtures/e_cpasync_exit): an exit with a cp.async never waited for whose
/// destination nobody reads is a contract finding; reading the destination without a wait
/// (the issuer itself, or another warp after a bar.sync) is a race; after wait_group 0 both
/// reads are clean. The GB300 profile discharges the contract.
#[test]
fn cpasync_exit_is_contract_unless_read() {
    for v in ["good", "read", "wait", "other_wait"] {
        both("e_cpasync_exit", v);
    }
    // The race leaves outcomes DPOR also enumerates: the stateful verdict only.
    check_stateful("e_cpasync_exit", "other", "ptx", 100_000);
    check_stateful("e_cpasync_exit", "other", "sass", 100_000);
    let p = ptxv_sim::hwprofile::GB300;
    assert!(
        p.iter()
            .any(|i| i.id == "E-CPASYNC-EXIT" && i.discharges == ["cp-async-exit:"])
    );
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

/// Input coverage with the PTX predicate %p7 (fixtures/e_cov_pt): PTX predicates are ordinary
/// registers (`Reg::is_pt`), so an input-derived `%p7` guard is input-dependent control on both
/// front-ends (the PTX front called the kernel input-independent before 2026-11-08).
#[test]
fn ptx_p7_is_not_pt_for_input_coverage() {
    let case = Case::load(root().join("fixtures/e_cov_pt")).unwrap();
    for variant in ["good", "P4"] {
        for front in ["ptx", "sass"] {
            let k = kernel(&case, variant, front);
            let launch = build_launch(&case, &k).unwrap();
            let c = ptxv_sim::coverage::analyze(&Machine::new(Arc::new(k), Arc::new(launch)));
            assert!(
                !c.input_independent && !c.input_dependent.is_empty(),
                "{variant}/{front}: {c:#?}"
            );
        }
        both("e_cov_pt", variant);
    }
}

/// Input coverage through TMEM: every TMEM writer carries the provenance of what it wrote.
/// fixtures/e_cov_tccp (input words to TMEM by tcgen05.cp, good, or tcgen05.st, ST) and
/// r1a_tcgen05_tile COV1 (an MMA accumulator), each read back by tcgen05.ld into a guard:
/// input-dependent on both front-ends. Before 2026-11-08 the tcgen05.cp words had no
/// provenance and e_cov_tccp was called input-independent.
#[test]
fn tmem_writers_carry_input_provenance() {
    for (name, variant) in [
        ("e_cov_tccp", "good"),
        ("e_cov_tccp", "ST"),
        ("r1a_tcgen05_tile", "COV1"),
    ] {
        let case = Case::load(root().join("fixtures").join(name)).unwrap();
        for front in ["ptx", "sass"] {
            let k = kernel(&case, variant, front);
            let launch = build_launch(&case, &k).unwrap();
            let c = ptxv_sim::coverage::analyze(&Machine::new(Arc::new(k), Arc::new(launch)));
            assert!(
                !c.input_independent && !c.input_dependent.is_empty(),
                "{name}/{variant}/{front}: {c:#?}"
            );
        }
        both(name, variant);
    }
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

#[test]
fn undefined_values_through_match() {
    both("e_uninit_match", "good");
    both("e_uninit_match", "store_all");
}

#[test]
fn asynchronous_memory_rewrites_preserve_execution_semantics() {
    for case in ["e_bulk_load_history", "e_bulk_shared_half", "e_clc_history"] {
        both(case, "reference");
        both(case, "candidate");
    }
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
#[ignore = "heavy: 106 s"]
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

/// S-INIT through an atomic (fixtures/smem_spill_sink): an atomic add into never-written shared
/// memory whose result is discarded is not reported at the atomic; the stored word carries the
/// undefined state, so only a later read that feeds it to a sink (readspill) is a violation;
/// a returned old value that reaches a sink stays undefined (useret).
#[test]
fn undefined_values_through_shared_atomics() {
    for v in ["good", "add0", "branch", "readspill", "useret"] {
        both("smem_spill_sink", v);
    }
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
#[ignore = "heavy: 92 s"]
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
// One test per (variant, front, check). PTX: the S-MBAR-SCOPE races (the peer's st.async
// complete-tx waited at .cta scope; with K = 3 also stage 0 reused after a .release.cta
// give-back) blow up the state space: bounded. K = 2 on PTX: a race within a bounded search;
// with async complete-tx counted as the mbarrier CTA's (the contract rerun) the search
// completes and leaves only the SASS verdict.
fn e_rmsnorm_ptx_contract(v: &str, want: &str) {
    let case = Case::load(root().join("fixtures/e_rmsnorm_cluster_drain")).unwrap();
    let k = Arc::new(kernel(&case, v, "ptx"));
    let mut launch = build_launch(&case, &k).unwrap();
    let r = explore(Machine::new(k.clone(), Arc::new(launch.clone())), 200_000);
    assert!(
        r.classes().iter().any(|c| c == "race"),
        "{v}: {:?}",
        r.classes()
    );
    launch.async_mb_dest = true;
    let r = explore(Machine::new(k, Arc::new(launch)), 2_000_000);
    assert!(r.complete, "{v}: contract rerun incomplete");
    assert_eq!(r.classes(), [want], "{v}: {:#?}", r.findings);
}
#[test]
fn e_rmsnorm_cluster_drain_remote_arrive_exit_drain_good_sass() {
    check("e_rmsnorm_cluster_drain", "good", "sass");
}
#[test]
fn e_rmsnorm_cluster_drain_remote_arrive_exit_drain_bug_sass() {
    check("e_rmsnorm_cluster_drain", "bug", "sass");
}
#[test]
#[ignore = "heavy: stateful and DPOR explores at 2M; 11 s"]
fn e_rmsnorm_cluster_drain_remote_arrive_exit_drain_good_k3_sass() {
    check("e_rmsnorm_cluster_drain", "good_k3", "sass");
}
#[test]
#[ignore = "heavy: stateful and DPOR explores at 2M; 8 s"]
fn e_rmsnorm_cluster_drain_remote_arrive_exit_drain_bug_k3_sass() {
    check("e_rmsnorm_cluster_drain", "bug_k3", "sass");
}
#[test]
fn e_rmsnorm_cluster_drain_remote_arrive_exit_drain_nodrain_sass() {
    check("e_rmsnorm_cluster_drain", "nodrain", "sass");
}
#[test]
#[ignore = "heavy: bounded explore at 200k; 54 s"]
fn e_rmsnorm_cluster_drain_remote_arrive_exit_drain_good_k3_ptx() {
    check_bounded("e_rmsnorm_cluster_drain", "good_k3", "ptx", 200_000);
}
#[test]
#[ignore = "heavy: bounded explore at 200k; 46 s"]
fn e_rmsnorm_cluster_drain_remote_arrive_exit_drain_bug_k3_ptx() {
    check_bounded("e_rmsnorm_cluster_drain", "bug_k3", "ptx", 200_000);
}
#[test]
#[ignore = "heavy: bounded explore at 200k, contract rerun at 2M; 57 s"]
fn e_rmsnorm_cluster_drain_remote_arrive_exit_drain_good_ptx() {
    e_rmsnorm_ptx_contract("good", "clean");
}
#[test]
#[ignore = "heavy: bounded explore at 200k, contract rerun at 2M; 45 s"]
fn e_rmsnorm_cluster_drain_remote_arrive_exit_drain_bug_ptx() {
    e_rmsnorm_ptx_contract("bug", "violation");
}
#[test]
#[ignore = "heavy: bounded explore at 200k, contract rerun at 2M; 30 s"]
fn e_rmsnorm_cluster_drain_remote_arrive_exit_drain_nodrain_ptx() {
    e_rmsnorm_ptx_contract("nodrain", "violation");
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

/// C-ORDER dead reads of strong loads: a relaxed load whose value is unused certifies; an
/// acquiring load that orders a later plain read (ACQ: PTX certified it before) and a 256-bit
/// strong load whose value is stored (V8: SASS certified it before) are refused.
#[test]
fn e_dead_read_strong_loads() {
    for v in ["good", "V8"] {
        both("e_dead_read_strong", v);
    }
    // ACQ races (DPOR's reductions are exact for race-free kernels only): the stateful
    // explorer's verdict.
    for front in ["ptx", "sass"] {
        check_stateful("e_dead_read_strong", "ACQ", front, 1_000_000);
    }
}

/// E-WARPID: %warpid is the warp's SM slot (GB300: not tid / 32), so storing it or branching
/// on it is reported on both front-ends; the warp index from %tid is clean.
#[test]
fn e_warpid_is_not_the_warp_index() {
    for v in ["good", "WARPID", "BRANCH"] {
        both("e_warpid", v);
    }
}

/// C-WAIT for successful peeks (fixtures/e_peek_result): a peek used only to skip a blocking
/// wait certifies; one whose result is stored (READ) or that alone orders a read (ACQ) can fail
/// in another schedule (certified clean before 2026-11-08; GB300 shows both results).
#[test]
fn e_peek_result_successful_peeks() {
    for v in ["good", "READ"] {
        both("e_peek_result", v);
    }
    // ACQ races (DPOR's reductions are exact for race-free kernels only): the stateful
    // explorer's verdict.
    for front in ["ptx", "sass"] {
        check_stateful("e_peek_result", "ACQ", front, 1_000_000);
    }
}

/// C-WAIT with several outstanding peeks per thread (fixtures/e_peek_multi): two peeks, then a
/// skip branch on each result, certify; a stored result (READ), crossed skips (SWAP) and an
/// unused peek without its wait (ACQ) are refused by the certificate, and the exploration
/// reports them. Before 2026-11-08 the log kept one peek per thread (TRT-LLM's four peeks
/// in a row were refused once a successful unused peek had to be ordered).
#[test]
fn e_peek_multi_outstanding_peeks() {
    let case = Case::load(root().join("fixtures/e_peek_multi")).unwrap();
    for front in ["ptx", "sass"] {
        for v in ["good", "READ", "SWAP", "ACQ"] {
            let k = kernel(&case, v, front);
            let launch = build_launch(&case, &k).unwrap();
            let c = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)));
            match (v, c.verdict) {
                ("good", Some(cv)) => assert_eq!(cv.classes(), ["clean"], "{front}"),
                ("good", None) => panic!("{front}: good does not certify: {:?}", c.reasons),
                (_, Some(cv)) => panic!("{front}: {v} certified: {:?}", cv.classes()),
                (_, None) => {}
            }
        }
        for v in ["good", "READ", "SWAP", "ACQ"] {
            check_stateful("e_peek_multi", v, front, 1_000_000);
        }
    }
}

/// Fence-acquired relaxed reads and read-modify-writes (fixtures/e_relaxed_fence_acq,
/// `Machine::fence_acq`): the last-block reduction certifies (C-TICKET with fence-acquired
/// claims); without its release fence (TICKET) it races in some claim order; RMW (C-ORDER
/// commutative atomics), DEAD (C-ORDER dead read) and ATOM (C-POLL-ATOM) are refused by the
/// certificate and race (certified clean before 2026-11-08; RMW wrong on GB300 200/200).
#[test]
fn e_relaxed_fence_acq_reads() {
    let case = Case::load(root().join("fixtures/e_relaxed_fence_acq")).unwrap();
    for front in ["ptx", "sass"] {
        for v in ["good", "TICKET", "RMW", "DEAD", "ATOM", "CHAIN"] {
            let k = kernel(&case, v, front);
            let launch = build_launch(&case, &k).unwrap();
            let c = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)));
            match (v, c.verdict) {
                ("good", Some(cv)) => assert_eq!(cv.classes(), ["clean"], "{front}"),
                ("good", None) => panic!("{front}: good does not certify: {:?}", c.reasons),
                (_, Some(cv)) => panic!("{front}: {v} certified: {:?}", cv.classes()),
                (_, None) => {}
            }
            check_stateful("e_relaxed_fence_acq", v, front, 1_000_000);
        }
    }
}

/// C-POLL-ATOM (E-WELLSYNC): an arrival counter of atomic adds polled to its full count is
/// certified; a poll that exits on a partial count is refused and races, also when the
/// partial count includes the poller's own add made after another CTA's add in the
/// certificate's run (own2; certified before the base included that add).
#[test]
fn e_poll_atom_counter_poll() {
    for v in ["good", "reset", "ge1", "own3"] {
        both("e_poll_atom", v);
    }
    // own2 races (DPOR's reductions are exact for race-free kernels only): the stateful
    // explorer's verdict.
    for front in ["ptx", "sass"] {
        check_stateful("e_poll_atom", "own2", front, 1_000_000);
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

/// S-SAME-VALUE per byte across a whole shadow page (fixtures/s_sv_page): a 512-byte bulk
/// reload whose first byte keeps its value and whose other bytes change races with a bulk store
/// ordered after it only in time (PAGE) or not at all (CONC). Until 2026-11-11 the page's first
/// byte stood for the page (`Pages::for_run`), and both were clean on both fronts. A release /
/// acquire flag (good) and a reload of identical bytes (SAME) are clean; SHIFT changes the first
/// byte too.
#[test]
fn s_sv_page_split_by_byte() {
    for v in ["good", "PAGE", "SHIFT", "CONC", "SAME"] {
        both("s_sv_page", v);
    }
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

/// Input coverage: copies of a machine get a placeholder instead of the provenance shadow,
/// and on the certificate path the shadow rides on the certificate's first run
/// (`coverage::tap_request`). On every fixture (every variant and front that loads), A's
/// whole report is the same in all three ways (copies copying the shadow, as before; the run
/// of its own; the certificate's run, `wellsync::certify`, without the exploration a refused
/// certificate falls back to). A's runs over 20 s are skipped; the other two ways run where
/// A's run took under 5 s (counted; `PTXV_COV_DIFF_FULL=1`: 120 s, all).
#[test]
#[ignore = "heavy: about 1300 analyses over all fixtures; 117 s"]
fn coverage_same_without_shadow_in_look_ahead_copies() {
    let dir = root().join("fixtures");
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("case.toml").exists())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    names.sort();
    let full = std::env::var_os("PTXV_COV_DIFF_FULL").is_some();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let counts = std::sync::Mutex::new([0usize; 4]);
    std::thread::scope(|sc| {
        for _ in 0..6 {
            sc.spawn(|| {
                // A run that ends on a copy of its machine panics instead of reporting nothing.
                ptxv_sim::coverage::set_strict(true);
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(name) = names.get(i) else { break };
                    let c = coverage_three_ways(&dir.join(name), name, full);
                    let mut t = counts.lock().unwrap();
                    for k in 0..4 {
                        t[k] += c[k];
                    }
                }
            });
        }
    });
    let [same, skipped, rode, not_rode] = *counts.lock().unwrap();
    eprintln!(
        "coverage with and without the shadow in copies: {same} identical, {skipped} skipped (long); on the certificate's run: {rode} identical, {not_rode} not compared or not carried"
    );
    assert!(same > 100, "{same}");
    assert!(rode > 100, "{rode}");
}

/// `coverage_same_without_shadow_in_look_ahead_copies` for one fixture: [identical, skipped,
/// identical on the certificate's run, not compared there].
fn coverage_three_ways(dir: &std::path::Path, name: &str, full: bool) -> [usize; 4] {
    let mut out = [0usize; 4];
    let Ok(case) = Case::load(dir) else {
        return out;
    };
    let b = ptxv_sim::coverage::Budget {
        seconds: Some(if full { 120.0 } else { 20.0 }),
        rss: None,
    };
    let j = |x: &ptxv_sim::coverage::Coverage| {
        let mut v = serde_json::to_value(x).unwrap();
        v.as_object_mut().unwrap().remove("seconds");
        v
    };
    for v in case.m.variants.iter().map(|x| x.id.clone()) {
        for front in ["ptx", "sass"] {
            let k = match &case.m.cubin {
                Some(_) if front == "ptx" => continue,
                Some(c) => match ptxv_sass::load_cubin(&case.dir.join(c), &case.m.entry) {
                    Ok(k) => k,
                    Err(_) => continue,
                },
                None => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    kernel(&case, &v, front)
                })) {
                    Ok(k) => k,
                    Err(_) => continue,
                },
            };
            let Ok(vc) = case.for_variant(&v) else {
                continue;
            };
            let Ok(launch) = build_launch(&vc, &k) else {
                continue;
            };
            let m = Machine::new(Arc::new(k), Arc::new(launch));
            let a = ptxv_sim::coverage::analyze_budget(&m, b);
            if a.stopped.is_some() {
                out[1] += 1;
                continue;
            }
            if !full && a.seconds > 5.0 {
                out[3] += 1;
                continue;
            }
            ptxv_sim::machine::set_scratch_copies_shadow(true);
            let c = ptxv_sim::coverage::analyze_budget(&m, b);
            ptxv_sim::machine::set_scratch_copies_shadow(false);
            if c.stopped.is_some() {
                out[1] += 1;
                continue;
            }
            assert_eq!(j(&a), j(&c), "{name} {v} {front}");
            out[0] += 1;
            ptxv_sim::coverage::tap_request(b);
            let _ = ptxv_sim::wellsync::certify(m.clone());
            match ptxv_sim::coverage::tap_take() {
                Some(f) if f.stopped.is_none() => {
                    assert_eq!(
                        j(&a),
                        j(&f),
                        "riding on the certificate: {name} {v} {front}"
                    );
                    out[2] += 1;
                }
                _ => out[3] += 1,
            }
        }
    }
    out
}

/// B's pre-pass (`coverage::mutated_pass`, fixtures/e_sentinel_fill): the planned mutation
/// (ok = 0 in two lanes) takes the untaken side; writing the value the capture already holds
/// opens nothing, and the pass says so; a hand mutation (no planned side) is undecided.
#[test]
fn mutated_pass_tells_flipped_from_unchanged() {
    use ptxv_sim::coverage::Budget;
    use ptxv_sim::coverage::Pass;
    let case = Case::load(root().join("fixtures/e_sentinel_fill")).unwrap();
    for front in ["ptx", "sass"] {
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let m = Machine::new(Arc::new(k), Arc::new(launch));
        let c = ptxv_sim::coverage::analyze(&m);
        let (muts, _) = ptxv_sim::coverage::plan(&m, &c, 4, 2, &Default::default());
        let mu = &muts[0];
        let pass = |mu: &ptxv_sim::coverage::Mutation| {
            ptxv_sim::coverage::mutated_pass(
                &ptxv_sim::coverage::apply(&m, mu),
                &c,
                mu,
                Budget::default(),
            )
        };
        assert_eq!(pass(mu), Pass::Flipped, "{front}");
        // The same elements, written with the values they hold (ok = 1): nothing changes.
        let mut same = mu.clone();
        for w in &mut same.writes {
            w.3 = 1;
        }
        assert_eq!(pass(&same), Pass::Unchanged, "{front}");
        let mut hand = mu.clone();
        hand.pc = u32::MAX;
        assert_eq!(pass(&hand), Pass::Undecided, "{front}");
    }
}

/// S-INIT scratch buffers (fixtures/e_scratch_undef, `scratch = true`): a buffer the case's
/// launches write before reading is produced inside the case, so bytes no launch writes are
/// undefined. good (the consumer reads only what the producer wrote) is clean; OFS1 (producer
/// and consumer disagree on the row stride) reports the undefined rows; RDFIRST (warp 1 reads
/// pd, a CTA barrier, then warp 0 writes) keeps pd as an input. RACEW1 / RACEW0: a read
/// unordered with the writes (another warp; lanes of the same warp, which the SASS front joins
/// in whichever order the paths run) makes pd produced whichever access the resolving run
/// makes first, and the copied never-written bytes are reported, never clean (RACEW0's first
/// access is the read). Without `scratch`, OFS1 reads the captured bytes and is clean: the
/// miss this closes. An explicit `init = "undef"` gives the same verdicts as the resolved
/// scratch buffer.
#[test]
fn e_scratch_undef_buffers() {
    for v in ["good", "OFS1", "RDFIRST", "RACEW1", "RACEW0"] {
        both("e_scratch_undef", v);
    }
    let base = Case::load(root().join("fixtures/e_scratch_undef")).unwrap();
    let pd = base.m.buffers.iter().position(|b| b.name == "pd").unwrap();
    let run = |case: &Case, variant: &str, front: &str| {
        let k = kernel(case, variant, front);
        let launch = build_launch(case, &k).unwrap();
        let notes: Vec<String> = launch.notes.iter().map(|n| n.0.clone()).collect();
        let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 100_000);
        assert!(v.complete);
        (v.classes(), notes)
    };
    for front in ["ptx", "sass"] {
        let (_, n) = run(&base, "OFS1", front);
        assert_eq!(n, ["scratch-undef:pd"], "{front}");
        let (_, n) = run(&base, "RDFIRST", front);
        assert_eq!(n, ["scratch-input:pd"], "{front}");
        for v in ["RACEW1", "RACEW0"] {
            let (c, n) = run(&base, v, front);
            assert_eq!(n, ["scratch-undef:pd"], "{front} {v}");
            assert!(!c.iter().any(|x| x == "clean"), "{front} {v}: {c:?}");
            let mut off = base.clone();
            off.m.buffers[pd].scratch = false;
            assert_eq!(run(&off, v, front).0, ["clean"], "{front} {v}");
        }
        let mut off = base.clone();
        off.m.buffers[pd].scratch = false;
        let (c, n) = run(&off, "OFS1", front);
        assert_eq!(c, ["clean"], "{front}: captured bytes hide the stride bug");
        assert!(n.is_empty());
        let mut ud = base.clone();
        ud.m.buffers[pd].scratch = false;
        ud.m.buffers[pd].init = "undef".into();
        assert_eq!(run(&ud, "good", front).0, ["clean"], "{front}");
        assert_eq!(run(&ud, "OFS1", front).0, ["violation"], "{front}");
    }
}

/// C-TICKET-ADDR (fixtures/e_ticket_addr, wellsync/perm.rs): slot assignment by fetch-add.
/// good certifies `clean-up-to-permutation` (never `clean`), and every output the explorer
/// reaches equals the certified one up to the permutation of each counter's rows. Without the
/// `[[permuted]]` declaration the certificate refuses (C-ORDER). Each mutant is refused:
/// BRANCH (a ticket-dependent branch), VALUE (a ticket-dependent stored value) and GUARD (a
/// claim behind a ticket-dependent guard) by the taint probes, SHIFT (a row that depends on
/// the claimant: a race in another order, not in the run) by the replays, READ (a later launch
/// reads a row) by the order condition, OVERLAP and NOHB by the race detector and the counter
/// check.
#[test]
fn e_ticket_addr_permutation_certificate() {
    for v in ["good", "BRANCH", "VALUE", "GUARD", "READ"] {
        both("e_ticket_addr", v);
    }
    let base = Case::load(root().join("fixtures/e_ticket_addr")).unwrap();
    let machine = |case: &Case, v: &str, front: &str| {
        let k = kernel(case, v, front);
        let launch = build_launch(case, &k).unwrap();
        Machine::new(Arc::new(k), Arc::new(launch))
    };
    // Racy variants: the explorer's classes (partial-order reduction is exact for race-free
    // kernels only, E-DRF).
    for v in ["OVERLAP", "NOHB", "SHIFT"] {
        for front in ["ptx", "sass"] {
            let ex = explore(machine(&base, v, front), 100_000);
            assert!(ex.complete);
            assert_eq!(
                ex.classes(),
                base.variant(v).unwrap().expect_for(front),
                "{front} {v}"
            );
        }
    }
    // Output byte offset of each allocation (outputs: val, src, cnt, back; buffers: ids, pre,
    // cnt, val, src, back).
    let off: std::collections::BTreeMap<usize, usize> =
        [(3usize, 0usize), (4, 32), (2, 64), (5, 80)]
            .into_iter()
            .collect();
    let canon = |o: &[u8], rows: &[Vec<ptxv_case::RowRanges>]| -> Vec<Vec<Vec<u8>>> {
        rows.iter()
            .map(|w| {
                let mut t: Vec<Vec<u8>> = w
                    .iter()
                    .map(|r| {
                        r.iter()
                            .flat_map(|&(a, lo, hi)| {
                                o[off[&a] + lo as usize..off[&a] + hi as usize].to_vec()
                            })
                            .collect()
                    })
                    .collect();
                t.sort();
                t
            })
            .collect()
    };
    for front in ["ptx", "sass"] {
        let o = ptxv_sim::wellsync::certify(machine(&base, "good", front));
        let v = o
            .verdict
            .unwrap_or_else(|| panic!("{front}: good not certified: {:?}", o.reasons));
        assert_eq!(v.classes(), ["clean-up-to-permutation"], "{front}");
        let f = v
            .findings
            .iter()
            .find(|f| f.kind == "clean-up-to-permutation")
            .unwrap();
        assert_eq!(f.perm_rows.len(), 2, "{front}: two counter words");
        assert!(f.perm_rows.iter().all(|w| w.len() == 3), "{front}");
        let ex = explore(machine(&base, "good", front), 100_000);
        assert!(ex.complete && ex.outputs.len() >= 2, "{front}");
        let want = canon(&v.outputs[0], &f.perm_rows);
        for x in &ex.outputs {
            assert_eq!(canon(x, &f.perm_rows), want, "{front}");
            // Bytes outside the rows are equal.
            let rowb: std::collections::BTreeSet<usize> = f
                .perm_rows
                .iter()
                .flatten()
                .flatten()
                .flat_map(|&(a, lo, hi)| {
                    let base = off[&a];
                    (lo..hi).map(move |b| base + b as usize)
                })
                .collect();
            for (i, (a, b)) in x.iter().zip(&v.outputs[0]).enumerate() {
                assert!(rowb.contains(&i) || a == b, "{front}: output byte {i}");
            }
        }
        let mut undeclared = base.clone();
        undeclared.m.permuted.clear();
        let o = ptxv_sim::wellsync::certify(machine(&undeclared, "good", front));
        assert!(
            o.verdict.is_none(),
            "{front}: certified without the declaration"
        );
        for (v, why) in [
            ("BRANCH", "reaches more than its own row stores"),
            ("VALUE", "reaches more than its own row stores"),
            ("GUARD", "reaches more than its own row stores"),
            ("SHIFT", "depends on more than the ticket"),
            ("READ", "rows must be outputs only"),
            ("OVERLAP", "race"),
            ("NOHB", "C-TICKET-ADDR"),
        ] {
            let o = ptxv_sim::wellsync::certify(machine(&base, v, front));
            assert!(o.verdict.is_none(), "{front} {v}: certified");
            assert!(
                o.reasons.iter().any(|r| r.contains(why)),
                "{front} {v}: {:?}",
                o.reasons
            );
        }
    }
}

/// C-TICKET-ADDR, fence-acquired claims (fixtures/e_ticket_addr_fence, a pre-land review
/// finding): thread 0 stores x, fences and claims; thread 32 claims, fences and reads x into its
/// row. The reader synchronizes with the writer only when its add reads from the writer's, so
/// happens-before depends on the claim order. The certificate's runs give claims no
/// synchronization, so its run has the race and it refuses; the explorer finds the race too.
#[test]
fn e_ticket_addr_fence_acquired_claims_refused() {
    let case = Case::load(root().join("fixtures/e_ticket_addr_fence")).unwrap();
    for v in ["good", "SWAP"] {
        for front in ["ptx", "sass"] {
            let k = kernel(&case, v, front);
            let launch = build_launch(&case, &k).unwrap();
            let m = Machine::new(Arc::new(k), Arc::new(launch));
            let o = ptxv_sim::wellsync::certify(m.clone());
            assert!(o.verdict.is_none(), "{front} {v}: certified");
            assert!(
                o.reasons.iter().any(|r| r.contains("race")),
                "{front} {v}: {:?}",
                o.reasons
            );
            let ex = explore(m, 100_000);
            assert!(ex.complete);
            assert_eq!(
                ex.classes(),
                case.variant(v).unwrap().expect_for(front),
                "{front} {v}"
            );
        }
    }
}

/// S-GPROXY: a semaphore-ordered chain of bulk add-reductions from two CTAs into one global
/// accumulator needs a global proxy fence after the acquire (`NOF`, `SH`: the order is not
/// guaranteed); so does a generic store followed by another CTA's reduction (`ST`). Contract
/// findings (the PTX ISA requires the fence; GB300 showed no failure).
#[test]
fn e_gproxy_verdicts() {
    for v in ["good", "NOF", "ALL", "SH", "ST", "STF"] {
        both("e_gproxy", v);
    }
}

/// S-CLUSTER-LIFE: a thread's plain st.shared::cluster into the peer CTA must happen before an
/// exit of that CTA. NOFINAL (no final barrier.cluster) and NOBAR (no barrier at all: the
/// target's only action is its exit, which the explorers' local reductions treat as commuting)
/// are violations on both fronts under the stateful explorer, DPOR and the certificate's
/// single run (before, the end-of-run check saw only remote arrivals, multicast TMA and
/// commits, and the exit-time check depended on the explored order).
#[test]
fn e_cluster_life_remote_store() {
    // `both`: stateful explorer with its reductions, then DPOR on the same verdict.
    for v in ["good", "NOFINAL", "NOBAR"] {
        both("e_cluster_life", v);
    }
    let case = Case::load(root().join("fixtures/e_cluster_life")).unwrap();
    for (v, front) in [
        ("NOFINAL", "ptx"),
        ("NOFINAL", "sass"),
        ("NOBAR", "ptx"),
        ("NOBAR", "sass"),
    ] {
        let k = kernel(&case, v, front);
        let launch = build_launch(&case, &k).unwrap();
        let c = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)));
        assert!(
            c.verdict
                .as_ref()
                .is_none_or(|v| v.classes().iter().any(|x| x == "violation")),
            "{v}/{front}: certified without the violation"
        );
    }
}

/// MISS 39 (E-WARPCOLL): a SASS SHFL runs with the split that reaches it. With the BSYNC
/// before it removed, GB300 runs the early half-warp's SHFL alone (inactive source lanes read
/// 0, 1000 of 1000 runs); the model let that half wait for the other and called the joined
/// exchange clean. NOBSYNC / NOBSSY must fault; good matches GB300's output.
#[test]
fn shfl_does_not_wait_for_the_rest_of_the_warp() {
    binary_classes("e_warpcoll_split");
}

/// Every variant of a binary-only fixture gets its expected classes (stateful explorer and
/// DPOR agree).
fn binary_classes(name: &str) {
    let case = Case::load(root().join("fixtures").join(name)).unwrap();
    for v in case.m.variants.iter().map(|v| v.id.clone()) {
        let k = binary_kernel(&case, &v).unwrap();
        let launch = build_launch(&case, &k).unwrap();
        let m0 = Machine::new(Arc::new(k), Arc::new(launch));
        let r = explore(m0.clone(), 1_000_000);
        assert!(r.complete, "{name}/{v}: exploration incomplete");
        assert_eq!(
            r.classes(),
            case.variant(&v).unwrap().expect_for("sass"),
            "{name}/{v}: {:#?}",
            r.findings
        );
        let d = ptxv_sim::dpor::explore_dpor(m0, 1_000_000);
        assert_eq!(d.classes(), r.classes(), "{name}/{v}: DPOR verdict differs");
    }
}

/// E-COLLECTIVE / E-WARPCOLL binary mutants of a divergent full-mask shfl.sync (each half
/// runs its own SHFL; ptxas's BRA.DIV slow paths): dropped BRA.DIVs, a collective member mask
/// that excludes the other half or the executing lanes, a dropped ENDCOLLECTIVE.
#[test]
fn collective_shfl_mutants() {
    binary_classes("e_warpcoll_bin");
}

/// E-WARPCOLL: register member masks of shfl.sync (exited lanes named, a source lane outside
/// the mask, masks that disagree), both front-ends.
#[test]
fn shfl_member_masks() {
    for v in ["good", "EXITED", "XSRC", "OVERLAP"] {
        both("e_warpcoll_mask", v);
    }
}

/// E-ALLOC / E-TC-EXIT mutants: unobserved tcgen05.cp at dealloc / exit, a trailing commit
/// alone (contract), a leaked or wrongly sized allocation, allocations that never succeed.
#[test]
fn tmem_alloc_and_exit_mutants() {
    for v in [
        "good",
        "CPNOWAIT",
        "TRAILCOMMIT",
        "CP2",
        "NODEALLOC",
        "REALLOC",
        "DEALLOCSZ",
        "W1ALLOC",
    ] {
        both("e_tc_alloc_exit", v);
    }
}

/// E-HW-BSYNC-COLLECTIVE (fixtures/e_bsync_parked_collective, the shape of recsys_tbe's
/// split_embedding_backward warp-per-row loop): the strict SASS model deadlocks with lanes
/// 0-15 at the join's BSYNC and lanes 16-31 at a WARPSYNC.COLLECTIVE; GB300 passes the BSYNC
/// (probes/bsync_coll: every warp of 10^4 single-warp runs and 875,520 warps at full occupancy).
/// The GB300 profile releases it, names the assumption, and gives GB300's output bitwise.
#[test]
fn bsync_releases_beside_a_parked_collective_under_the_gb300_profile() {
    binary_classes("e_bsync_parked_collective");
    let case = Case::load(root().join("fixtures/e_bsync_parked_collective")).unwrap();
    let k = binary_kernel(&case, "good").unwrap();
    let launch = build_launch(&case, &k).unwrap();
    let m = Machine::new(Arc::new(k), Arc::new(launch));
    let p = ptxv_sim::hwprofile::GB300;
    let m2 = ptxv_sim::hwprofile::machine(&m, p);
    let mut v = explore(m2.clone(), 100_000);
    let used = ptxv_sim::hwprofile::discharge(&mut v, &m2, p);
    assert!(v.complete, "{:#?}", v.findings);
    assert_eq!(v.classes(), ["clean"], "{:#?}", v.findings);
    assert!(
        used.iter().any(|(id, _)| id == "E-HW-BSYNC-COLLECTIVE"),
        "{used:?}"
    );
    assert_eq!(v.outputs.len(), 1);
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        !report.is_empty() && report.iter().all(|l| l.contains("bitwise equal")),
        "{report:?}"
    );
}

/// E-ASYNC-STEP (fixtures/e_tma_store_readback): `cp.async.bulk.wait_group.read` completes only
/// the group's shared-memory source reads. A thread that loads back an element of its own TMA
/// store's destination after `.read 0` races with the store's global write on both fronts
/// (GB300: 196/200 runs read the old value); after a full `wait_group 0` it is clean. The PTX
/// front used to treat `.read` as the full wait (clean).
#[test]
fn tma_store_readback_after_wait_read_races() {
    for v in ["good", "READ", "NOWAIT"] {
        both("e_tma_store_readback", v);
        certify_meets("e_tma_store_readback", v, "ptx");
        certify_meets("e_tma_store_readback", v, "sass");
    }
}

/// E-ASYNC-STEP for the non-tensor bulk store and the bulk add-reduction
/// (fixtures/e_bulk_read_l1): `cp.async.bulk.wait_group.read` completes only the group's
/// shared-memory reads, so a load of the destination after it races with the global writes, a
/// reduction's too (READ, RREAD); after a full wait it is clean (good, RGOOD). GB300 reads the
/// stale pre-store L1 line in 1000/1000 runs of READ and RREAD, the stored value after a full
/// wait. The SASS front's strict level acquires the writes at the issuer's next `CCTL.IVALL`
/// only (MISS 102; it used to call READ and RREAD clean), so both fronts report the races; its
/// GB300 level (L1 model) acquires them at the `DEPBAR` and reports the stale reads.
#[test]
fn bulk_store_and_reduction_after_wait_read_race() {
    for v in ["good", "READ", "RGOOD", "RREAD"] {
        both("e_bulk_read_l1", v);
        certify_meets("e_bulk_read_l1", v, "ptx");
        certify_meets("e_bulk_read_l1", v, "sass");
    }
    gb300_level_is(
        "e_bulk_read_l1",
        &[
            ("good", &["clean"]),
            ("READ", &["stale"]),
            ("RGOOD", &["clean"]),
            ("RREAD", &["stale"]),
        ],
    );
}

/// E-ASYNC-STEP, SASS front (fixtures/e_bulk_read_l1_other, MISS 102): another thread of the CTA
/// rereads a bulk store's destination, whose L1 line it filled before the store, after a
/// `bar.sync` that follows the issuer's `.read` wait: GB300 returns the stale line in 1000/1000
/// runs. The issuer acquires the writes only at a `CCTL.IVALL` (none after `.read`), so the
/// barrier carries nothing: a race on both fronts (READ); the full wait is clean (good). The
/// GB300 level reports the stale read.
#[test]
fn bulk_store_read_by_another_thread_after_wait_read_races() {
    for v in ["good", "READ"] {
        both("e_bulk_read_l1_other", v);
        certify_meets("e_bulk_read_l1_other", v, "ptx");
        certify_meets("e_bulk_read_l1_other", v, "sass");
    }
    gb300_level_is(
        "e_bulk_read_l1_other",
        &[("good", &["clean"]), ("READ", &["stale"])],
    );
}

/// The SASS front's GB300 level (the GB300 profile: the L1 model) of each variant.
fn gb300_level_is(case: &str, want: &[(&str, &[&str])]) {
    let case = Case::load(root().join("fixtures").join(case)).unwrap();
    for &(v, w) in want {
        let k = kernel(&case, v, "sass");
        let launch = build_launch(&case, &k).unwrap();
        let launch = ptxv_sim::hwprofile::launch(&launch, ptxv_sim::hwprofile::GB300);
        let r = explore(Machine::new(Arc::new(k), Arc::new(launch)), 2_000_000);
        assert!(r.complete, "{v} GB300: incomplete");
        assert_eq!(r.classes(), w, "{v} GB300: {:#?}", r.findings);
    }
}

/// E-ASYNC-STEP (fixtures/e_tma_read_then_full): a `.read` wait completes the group's
/// shared-memory reads and keeps the group, so a full `wait_group 0` after it still orders the
/// store's global writes before the load of the destination (clean on both fronts). A `.read`
/// wait that dropped the group left the writes unordered (a race).
#[test]
fn tma_store_full_wait_after_wait_read_orders_the_writes() {
    both("e_tma_read_then_full", "good");
    certify_meets("e_tma_read_then_full", "good", "ptx");
    certify_meets("e_tma_read_then_full", "good", "sass");
}

/// E-WELLSYNC on a fixture variant: a certificate that holds must give the variant's
/// expected classes (READ / LATEWAIT used to certify clean); a refused one must say why
/// (`ptxv check --certify` then explores).
fn certify_meets(case: &str, variant: &str, front: &str) {
    let case = Case::load(root().join("fixtures").join(case)).unwrap();
    let k = kernel(&case, variant, front);
    let launch = build_launch(&case.for_variant(variant).unwrap(), &k).unwrap();
    let o = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)));
    let expect = case.variant(variant).unwrap().expect_for(front);
    match o.verdict {
        Some(v) => assert!(
            v.complete && v.classes() == expect,
            "{variant}/{front}: certified {:?} (complete: {}), want {expect:?}",
            v.classes(),
            v.complete
        ),
        None => assert!(
            !o.reasons.is_empty(),
            "{variant}/{front}: refused without a reason"
        ),
    }
}

/// E-CPASYNC (fixtures/e_cpasync_early): a cp.async's shared write lands at the wait that
/// retires it, as an async actor of its own. A barrier the issuer reaches before its
/// wait_group does not order the write before another thread's read of the slot (LATEWAIT: a
/// race on both fronts; GB300 reads the old value in 200/200 runs); waiting before the barrier
/// (good) is clean. Both fronts used to call LATEWAIT clean, race-checking the write with the
/// issue clock.
#[test]
fn cpasync_write_is_not_ordered_by_a_barrier_before_the_wait() {
    both("e_cpasync_early", "good");
    // The stateful explorer reports the race in one order; DPOR may also reach its second
    // outcome (E-DRF), but must find the race.
    for front in ["ptx", "sass"] {
        check_stateful("e_cpasync_early", "LATEWAIT", front, 100_000);
        let case = Case::load(root().join("fixtures/e_cpasync_early")).unwrap();
        let k = kernel(&case, "LATEWAIT", front);
        let launch = build_launch(&case, &k).unwrap();
        let d = ptxv_sim::dpor::explore_dpor(Machine::new(Arc::new(k), Arc::new(launch)), 100_000);
        assert!(
            d.complete && d.classes().iter().any(|c| c == "race"),
            "LATEWAIT/{front}: DPOR {:?}",
            d.classes()
        );
        for v in ["good", "LATEWAIT"] {
            certify_meets("e_cpasync_early", v, front);
        }
    }
}

/// E-CPASYNC-MBAR (fixtures/e_cpasync_mbar_early): copies tracked by cp.async.mbarrier.arrive
/// land at their own issue clocks, not the arrive's. EARLY (the copy issued before a barrier,
/// the arrive after it; another thread's read before the barrier) and OWN (the issuer's own
/// store between the copy and the arrive) race on both fronts; GB300 shows the copy landing
/// first in 194/200 and 200/200 runs. Both used to be clean and certified.
#[test]
fn cpasync_mbar_copies_land_at_their_issue_clocks() {
    for v in ["good", "EARLY", "OWN"] {
        both("e_cpasync_mbar_early", v);
        certify_meets("e_cpasync_mbar_early", v, "ptx");
        certify_meets("e_cpasync_mbar_early", v, "sass");
    }
}

/// E-CPASYNC (fixtures/e_cpasync_overlap): one thread's copies to the same bytes, in flight
/// together, race (SAME: one group; GROUPS: two groups, one wait; MBAR: one
/// cp.async.mbarrier.arrive), though they land as one actor; so do copies retired by two
/// waits, the second issued before the first wait (LATER: two epochs of the thread's wait
/// actor); a wait between them orders them (SEQ). GB300 leaves the first copy's bytes in
/// 1000/1000 runs of SAME, where the model used to call it clean with the second copy's.
#[test]
fn cpasync_copies_in_flight_together_race() {
    for v in ["good", "SAME", "GROUPS", "SEQ", "MBAR", "LATER"] {
        both("e_cpasync_overlap", v);
        certify_meets("e_cpasync_overlap", v, "ptx");
        certify_meets("e_cpasync_overlap", v, "sass");
    }
}

/// E-CPASYNC-PROXY (fixtures/e_cpasync_proxy_early): an async-proxy read of a cp.async's
/// destination needs a `fence.proxy.async` after the copy completed (after its wait) that
/// happens-before the read. A fence between the copies' issue and their wait (EARLY, as in
/// Triton's attention kernels) covers nothing, like no fence (NONE): contract findings on both
/// fronts; the fence after the wait (good) is clean.
#[test]
fn cpasync_proxy_fence_must_follow_the_copy_s_completion() {
    for v in ["good", "EARLY", "NONE"] {
        both("e_cpasync_proxy_early", v);
        certify_meets("e_cpasync_proxy_early", v, "ptx");
        certify_meets("e_cpasync_proxy_early", v, "sass");
    }
}

/// MISS 68 (fixtures/e_async_actor_cap): a thread's async actor ids come from one checked
/// allocator (`Machine::async_actor`); past them the step fails closed, an unmodeled fault that
/// stops the thread, and never wraps (an id past the limit would alias the wait actor, then
/// the next thread's ids). With the ids cut to 4096 (`Machine::async_shift` 12; a lone thread
/// has 2^30), MANY's 4096th arrive is that fault and nothing else is reported; 4000 arrives
/// (good) stay clean. With the real ids both are clean. Stateful search only: DPOR takes about
/// 1000 s on one run of this 4000-arrive loop.
#[test]
fn async_actor_ids_fail_closed() {
    let case = Case::load(root().join("fixtures/e_async_actor_cap")).unwrap();
    for front in ["ptx", "sass"] {
        for v in ["good", "MANY"] {
            check_stateful("e_async_actor_cap", v, front, 100_000);
            let k = kernel(&case, v, front);
            let launch = build_launch(&case, &k).unwrap();
            let mut m0 = Machine::new(Arc::new(k), Arc::new(launch));
            m0.async_shift = 12;
            let r = explore(m0, 100_000);
            let want: &[&str] = if v == "good" {
                &["clean"]
            } else {
                &["unmodeled"]
            };
            assert!(
                r.complete && r.classes() == want,
                "{v}/{front} with 4096 ids: {:?} (complete: {}) {:#?}",
                r.classes(),
                r.complete,
                r.findings
            );
            assert!(
                r.findings.iter().all(|f| f.kind != "unmodeled"
                    || (f.sig.starts_with("fault:") && f.msg.contains("async actor ids"))),
                "{v}/{front}: {:#?}",
                r.findings
            );
        }
    }
}

/// MISS 68: with a fixed 4096 async actor ids per thread (`tma_actor_id` = 3 n + 4096 t +
/// serial), a thread's 4097th async operation took the next thread's first id, or one past the
/// last thread. Now a thread has the largest power of two of ids below 2^31 (`async_shift`;
/// 2^25 with 32 threads). fixtures/e_async_ids_nc: the S-NC violation that was missed (the
/// in-place bulk store's id decoded as a later, stream-ordered kernel's thread) is found;
/// e_async_ids_stream: the false violation (an earlier kernel's store decoded as the reading
/// kernel's) is gone; e_async_ids_last: no panic. The control (the store as the thread's
/// 4096th async operation) keeps its verdict.
#[test]
fn async_ids_past_4096_do_not_alias() {
    for front in ["ptx", "sass"] {
        check_stateful("e_async_ids_nc", "good", front, 100_000);
        check_stateful("e_async_ids_nc", "n4095", front, 100_000);
    }
}

/// MISS 68, decoding (see async_ids_past_4096_do_not_alias): an earlier kernel's store is not
/// decoded as the reading kernel's (e_async_ids_stream), nor past the last thread
/// (e_async_ids_last).
#[test]
fn async_ids_past_4096_decode_to_their_issuer() {
    for front in ["ptx", "sass"] {
        check_stateful("e_async_ids_stream", "good", front, 100_000);
        check_stateful("e_async_ids_last", "good", front, 100_000);
    }
}

/// MISS 68 (fixtures/e_async_ids_race): thread 0's 4097th async operation, a bulk store, and
/// thread 1's first write one slot unordered (a relaxed flag only): with the fixed 4096 ids they
/// were one actor, whose writes the race check orders with each other, and the race was missed
/// (clean on both fronts). n4095 (the store is thread 0's 4096th) is the control.
#[test]
#[ignore = "heavy: 106 s, 8 GB"]
fn async_ids_race_past_4096() {
    for v in ["good", "n4095"] {
        for front in ["ptx", "sass"] {
            check_stateful("e_async_ids_race", v, front, 200_000);
        }
    }
}

/// MISS 68 with `--slices` (fixtures/e_async_ids_slices): CTA 1 reads with .nc, unsynchronized,
/// what CTA 0's last thread writes with its 4097th async operation. With the fixed 4096 ids the
/// write's id decoded as CTA 1's thread, slice exploration put it in cluster 1's footprint and
/// found no cross-slice conflict (clean). Now a race there, as for n0 (the store is the thread's
/// first async operation); explored whole, race and violation.
#[test]
fn async_ids_slices_attribute_the_issuer() {
    let case = Case::load(root().join("fixtures/e_async_ids_slices")).unwrap();
    check_stateful("e_async_ids_slices", "good", "ptx", 100_000);
    for front in ["ptx", "sass"] {
        for v in ["good", "n0"] {
            let k = Arc::new(kernel(&case, v, front));
            let launch = Arc::new(build_launch(&case, &k).unwrap());
            let make = || Machine::new(k.clone(), launch.clone());
            let r = ptxv_sim::explore::explore_symmetric(&make, 100_000, &mut |_, _| {});
            assert!(
                r.complete && r.classes().iter().any(|c| c == "race"),
                "{v}/{front} by slices: {:?} (complete: {})",
                r.classes(),
                r.complete
            );
        }
    }
}

/// E-CPASYNC, SASS cp.async scoreboard (fixtures/e_cpasync_div_wait): one path's `DEPBAR`
/// retires the copies of lanes on another path; those lanes are ordered after their copies at
/// their own later `DEPBAR` (good: clean on both fronts, as before the copies landed as an
/// async actor), and not without one (NOWAIT: the even lanes' read races with their copy).
#[test]
fn cpasync_divergent_waits_order_their_lanes() {
    for v in ["good", "NOWAIT"] {
        both("e_cpasync_div_wait", v);
        certify_meets("e_cpasync_div_wait", v, "ptx");
        certify_meets("e_cpasync_div_wait", v, "sass");
    }
}

/// E-CPASYNC, SASS divergent waits (fixtures/e_cpasync_div_cover): a lane is ordered after the
/// copies another path's `DEPBAR` retired only at its own later `DEPBAR` covering them
/// (`Warp::cp_landed`). NEIGH: the even lanes' `DEPBAR.LE SB0, 1` covers their own commit, not
/// the odd lanes' later one, so the read of the odd neighbour's slot races (both fronts; the
/// PTX front's per-thread waits cover no neighbour's copy). good: each lane reads its own slot
/// after wait_group 0 (clean).
#[test]
fn cpasync_divergent_waits_join_only_the_copies_they_cover() {
    for v in ["good", "NEIGH"] {
        both("e_cpasync_div_cover", v);
        certify_meets("e_cpasync_div_cover", v, "ptx");
        certify_meets("e_cpasync_div_cover", v, "sass");
    }
}

/// E-ASYNC-STEP, SASS front (fixtures/e_tma_read_l1, MISS 102): after
/// `cp.async.bulk.wait_group.read` ptxas's `DEPBAR` waits for the TMA store's global writes,
/// but without the full form's `CCTL.IVALL` a line of the destination the issuing SM filled
/// before the store stays in its L1: GB300 reads the old bytes in 1000/1000 runs of READ3 and
/// READ4 (out = NaN), the stored value in READ2 (no earlier fill) and with the full wait. The
/// strict level acquires the writes at the thread's next `CCTL.IVALL` only, so READ2-READ4 are
/// races on both fronts (the SASS front used to call READ4 clean and certified, and then
/// skipped the GB300 level). The GB300 profile (L1 model) acquires them at the `DEPBAR`: the
/// stale reads of READ3 / READ4 (E-HW-L1), READ2 clean; the hardware model calls READ4's race
/// observable.
#[test]
fn tma_store_read_l1_needs_cctl() {
    for v in ["good", "READ2", "READ3", "GOOD3", "READ4", "GOOD4"] {
        both("e_tma_read_l1", v);
        certify_meets("e_tma_read_l1", v, "sass");
    }
    let case = Case::load(root().join("fixtures/e_tma_read_l1")).unwrap();
    for (v, want) in [
        ("good", &["clean"][..]),
        ("READ2", &["clean"]),
        ("READ3", &["contract", "stale"]),
        ("GOOD3", &["contract"]),
        ("READ4", &["stale"]),
        ("GOOD4", &["clean"]),
    ] {
        let k = kernel(&case, v, "sass");
        let launch = build_launch(&case, &k).unwrap();
        let launch = ptxv_sim::hwprofile::launch(&launch, ptxv_sim::hwprofile::GB300);
        let r = explore(Machine::new(Arc::new(k), Arc::new(launch)), 2_000_000);
        assert!(r.complete, "{v} GB300: incomplete");
        assert_eq!(r.classes(), want, "{v} GB300: {:#?}", r.findings);
    }
    gb300_model_classifies("e_tma_read_l1", "READ4", "observable", true);
}

/// E-ASYNC-STEP (fixtures/e_tma_store_next_kernel): a TMA store whose issuer waited with
/// `cp.async.bulk.wait_group.read` only is still ordered before a dependent kernel that waits
/// for its grid (`griddepcontrol.wait`): the grid's completion covers the store's writes, one
/// epoch after its reads (good, clean). Without the wait the load races with the write.
#[test]
fn tma_store_after_wait_read_is_ordered_by_grid_completion() {
    for v in ["good", "NOWAIT"] {
        both("e_tma_store_next_kernel", v);
    }
}

/// E-ASYNC-STEP with launch groups (`ptxv check --groups`): fixtures/e_tma_store_next_kernel good
/// with the load kernel stream-ordered after the store grid (no PDL), checked as two groups. At
/// the group boundary the store's completion joins its CTA's clocks at the store actor's next
/// epoch, the writes' (`groups.rs` `fold_completed`), so the load in the second group is ordered
/// after the writes its issuer waited for with `.read` only: clean on both fronts, as in one
/// group. Joining the reads' epoch only made the load race with the writes.
#[test]
fn launch_group_boundary_orders_a_bulk_store_s_writes() {
    let mut case = Case::load(root().join("fixtures/e_tma_store_next_kernel")).unwrap();
    case.m.launches[0].pdl = false;
    for front in ["ptx", "sass"] {
        let k = kernel(&case, "good", front);
        let launch = build_launch(&case, &k).unwrap();
        let m = Machine::new(Arc::new(k), Arc::new(launch));
        let v = ptxv_sim::groups::check_spec(m, "1", 200_000, &mut |_, _, _| {});
        assert!(
            v.complete && v.classes() == ["clean"],
            "{front}: {:?} (complete: {}) {:#?}",
            v.classes(),
            v.complete,
            v.findings
        );
    }
}

/// Evidence classes (probes/gb300_runs.toml): the probes whose expected/ outputs are GB200's
/// also ran on GB300 (sm_103a builds); each recorded GB300 output (size and FNV-1a hash of the
/// case's outputs in order) must equal the GB200 record bit for bit, so the probe's model
/// claims hold on both GPUs and a regenerated expected/ cannot silently drop the GB300 match.
#[test]
fn gb300_runs_equal_gb200_probe_records() {
    let rec = std::fs::read_to_string(root().join("probes/gb300_runs.toml")).unwrap();
    let field = |b: &str, k: &str| -> String {
        b.lines()
            .find_map(|l| l.strip_prefix(&format!("{k} = ")))
            .unwrap_or_else(|| panic!("{k} missing in {b}"))
            .trim_matches('"')
            .to_string()
    };
    let mut n = 0;
    for b in rec.split("[[run]]").skip(1) {
        let probe = field(b, "probe");
        let case = Case::load(root().join("probes").join(&probe)).unwrap();
        let mut gb200 = Vec::new();
        for o in &case.m.outputs {
            let buf = case.m.buffers.iter().find(|x| &x.name == o).unwrap();
            let e = buf
                .expect
                .as_ref()
                .unwrap_or_else(|| panic!("{probe}: {o} has no expect"));
            let p = if std::path::Path::new(e).is_absolute() {
                PathBuf::from(e)
            } else {
                case.dir.join(e)
            };
            gb200.extend(std::fs::read(&p).unwrap());
        }
        assert_eq!(gb200.len().to_string(), field(b, "bytes"), "{probe}: size");
        assert_eq!(
            format!("{:016x}", fnv(&gb200)),
            field(b, "fnv1a64"),
            "{probe}: GB300 differs from the GB200 record"
        );
        n += 1;
    }
    assert_eq!(n, 21);
}

/// SASS `@P BRA.U` (probes/bra_u, GB300): the branch is taken by every lane of the executing
/// split when the guard holds in all of them, else by none; ptxas's plain `@P BRA` is per lane.
/// Each case is one run of a patched ptxas binary (patch.py sets the encoding's .U bit), with a
/// full warp (full), lanes exited before the branch (exit) and lanes on another path (split);
/// both binaries' outputs (the body's and the join's active masks, the elected lanes, the body
/// count) must equal GB300's.
#[test]
fn probe_bra_u_matches_hardware() {
    let base = root().join("probes/bra_u");
    let mut n = 0;
    for kind in ["full", "exit", "split"] {
        let mut dirs: Vec<_> = std::fs::read_dir(base.join(kind))
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        dirs.sort();
        for d in dirs {
            let case = Case::load(&d).unwrap();
            for (cubin, gpu) in [
                (format!("{kind}_bra.cubin"), "gb300_bra.bin"),
                (format!("{kind}_bra_u.cubin"), "gb300_bra_u.bin"),
            ] {
                let k = ptxv_sass::load_cubin(&base.join(&cubin), &case.m.entry).unwrap();
                let launch = build_launch(&case, &k).unwrap();
                let v = explore(Machine::new(Arc::new(k), Arc::new(launch)), 100_000);
                let at = format!("{} {cubin}", d.display());
                assert!(v.complete, "{at}: incomplete");
                assert_eq!(v.classes(), ["clean"], "{at}: {:#?}", v.findings);
                assert_eq!(v.outputs.len(), 1, "{at}: several outcomes");
                assert!(
                    v.outputs[0] == std::fs::read(d.join(gpu)).unwrap(),
                    "{at}: differs from GB300"
                );
                n += 1;
            }
        }
    }
    assert_eq!(n, 36);
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! E-BAR-CUT and E-BAR-OVER walk values differential: the stateful explorer with and without
//! the all-thread barrier cut (`ptxv_sim::explore::set_bar_cut`) and the walk's values
//! (`set_bar_values`) on every named-barrier fixture, both fronts; and fixtures/r2_cute_small
//! CM1 on PTX, bounded, without them. A separate test binary, because the settings are
//! process-wide: one test per (fixture, variant) so the pieces can run in parallel processes,
//! and within one process the tests hold `LOCK` while they change the settings (they run one
//! at a time, and each leaves the settings unset).

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::explore::Verdict;
use ptxv_sim::explore::explore;
use ptxv_sim::explore::set_bar_cut;
use ptxv_sim::explore::set_bar_values;
use ptxv_sim::machine::Machine;

fn kernel(case: &Case, variant: &str, front: &str) -> ptxv_ir::Kernel {
    let dir = std::env::temp_dir().join(format!(
        "ptxv-test-barcut-{}-{}-{variant}",
        std::process::id(),
        case.m.name
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let c = dir.join("k.cubin");
    if let Some(bin) = &case.m.cubin {
        std::fs::copy(case.dir.join(bin), &c).unwrap();
    } else {
        let ptx = case.variant_ptx(variant).unwrap();
        if front == "ptx" {
            return ptxv_ptx::parse_kernel(&ptx, Some(&case.m.entry)).unwrap();
        }
        let p = dir.join("k.ptx");
        std::fs::write(&p, &ptx).unwrap();
        ptxv_sass::ptxas(&p, &c, &case.m.arch).unwrap();
    }
    let patch = &case.variant(variant).unwrap().patch;
    ptxv_sass::scoreboard::patch_file(&c, &case.m.entry, patch).unwrap();
    ptxv_sass::load_cubin(&c, &case.m.entry).unwrap()
}

fn run(case: &Case, variant: &str, front: &str, on: bool) -> Verdict {
    set_bar_cut(Some(on));
    set_bar_values(Some(on));
    let k = kernel(case, variant, front);
    let launch = build_launch(case, &k).unwrap();
    explore(Machine::new(Arc::new(k), Arc::new(launch)), 200_000)
}

fn outputs(v: &Verdict) -> Vec<Vec<u8>> {
    let mut o = v.outputs.clone();
    o.sort();
    o.dedup();
    o
}

/// Serializes the tests of this binary: the settings are process-wide.
static LOCK: Mutex<()> = Mutex::new(());

/// Holds `LOCK`; unsets both settings when dropped (also when the test fails).
struct Settings(#[allow(dead_code)] MutexGuard<'static, ()>);

impl Settings {
    fn take() -> Self {
        let g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_bar_cut(None);
        set_bar_values(None);
        Settings(g)
    }
}

impl Drop for Settings {
    fn drop(&mut self) {
        set_bar_cut(None);
        set_bar_values(None);
    }
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// The named-barrier fixtures and their variants, one test each below.
const FIXTURES: &[(&str, &[&str])] = &[
    ("e_barcut", &["good", "BC1", "BC2", "BC3", "BC4", "BC5"]),
    ("e_barwalk", &["good", "BW1", "BW2"]),
    ("e_barover", &["good", "BO1", "BO2", "BO3"]),
    ("e_bararv", &["good", "AB1", "AB2", "AB3", "AB4"]),
    ("e_aligned_bar", &["good", "AB1", "AB2", "AB3"]),
    ("e_bardiv_hang", &["good"]),
    ("e_bardiv_short", &["good"]),
    ("s_bardiv", &["good"]),
    ("s_bardiv_mixed", &["good"]),
];

/// The tests below cover every variant of every fixture in `FIXTURES`: a variant added to one
/// of the fixtures needs its test.
#[test]
fn every_variant_has_a_test() {
    for (name, vars) in FIXTURES {
        let case = Case::load(root().join(name)).unwrap();
        let have: Vec<&str> = case.m.variants.iter().map(|v| v.id.as_str()).collect();
        assert_eq!(&have[..], *vars, "{name}: variants without a test");
    }
}

/// One fixture variant, on both fronts (SASS only for a cubin fixture): the explorer with the
/// barrier cut and the walk's values on and off keeps its classes and outputs.
fn on_off(name: &str, var: &str) {
    let _s = Settings::take();
    let case = Case::load(root().join(name)).unwrap();
    let fronts: &[&str] = if case.m.cubin.is_some() {
        &["sass"]
    } else {
        &["ptx", "sass"]
    };
    let mut fewer = 0;
    for &front in fronts {
        let on = run(&case, var, front, true);
        let off = run(&case, var, front, false);
        let tag = format!("{name}/{var}/{front}");
        assert!(on.complete && off.complete, "{tag}: incomplete");
        assert_eq!(on.classes(), off.classes(), "{tag}: classes");
        assert_eq!(outputs(&on), outputs(&off), "{tag}: outputs");
        if name == "e_barcut" || name == "e_barwalk" {
            let expect = case.variant(var).unwrap().expect_for(front);
            assert!(
                on.meets(&expect),
                "{tag}: got {:?}, want {expect:?}",
                on.classes()
            );
            if name == "e_barcut" && var == "good" {
                assert!(
                    on.states < off.states,
                    "{tag}: {} vs {} states",
                    on.states,
                    off.states
                );
            }
        }
        fewer += (on.states < off.states) as usize;
    }
    eprintln!("{name}/{var}: fewer states with the cut and values: {fewer} runs");
}

#[test]
fn e_barcut_good() {
    on_off("e_barcut", "good");
}
#[test]
fn e_barcut_bc1() {
    on_off("e_barcut", "BC1");
}
#[test]
fn e_barcut_bc2() {
    on_off("e_barcut", "BC2");
}
#[test]
fn e_barcut_bc3() {
    on_off("e_barcut", "BC3");
}
#[test]
fn e_barcut_bc4() {
    on_off("e_barcut", "BC4");
}
#[test]
fn e_barcut_bc5() {
    on_off("e_barcut", "BC5");
}
#[test]
fn e_barwalk_good() {
    on_off("e_barwalk", "good");
}
#[test]
fn e_barwalk_bw1() {
    on_off("e_barwalk", "BW1");
}
#[test]
fn e_barwalk_bw2() {
    on_off("e_barwalk", "BW2");
}
#[test]
fn e_barover_good() {
    on_off("e_barover", "good");
}
#[test]
fn e_barover_bo1() {
    on_off("e_barover", "BO1");
}
#[test]
fn e_barover_bo2() {
    on_off("e_barover", "BO2");
}
#[test]
fn e_barover_bo3() {
    on_off("e_barover", "BO3");
}
#[test]
fn e_bararv_good() {
    on_off("e_bararv", "good");
}
#[test]
fn e_bararv_ab1() {
    on_off("e_bararv", "AB1");
}
#[test]
fn e_bararv_ab2() {
    on_off("e_bararv", "AB2");
}
#[test]
fn e_bararv_ab3() {
    on_off("e_bararv", "AB3");
}
#[test]
fn e_bararv_ab4() {
    on_off("e_bararv", "AB4");
}
#[test]
fn e_aligned_bar_good() {
    on_off("e_aligned_bar", "good");
}
#[test]
fn e_aligned_bar_ab1() {
    on_off("e_aligned_bar", "AB1");
}
#[test]
fn e_aligned_bar_ab2() {
    on_off("e_aligned_bar", "AB2");
}
#[test]
fn e_aligned_bar_ab3() {
    on_off("e_aligned_bar", "AB3");
}
#[test]
fn e_bardiv_hang_good() {
    on_off("e_bardiv_hang", "good");
}
#[test]
fn e_bardiv_short_good() {
    on_off("e_bardiv_short", "good");
}
#[test]
fn s_bardiv_good() {
    on_off("s_bardiv", "good");
}
#[test]
fn s_bardiv_mixed_good() {
    on_off("s_bardiv_mixed", "good");
}

/// fixtures/r2_cute_small CM1 on PTX, bounded (moved from fixtures.rs `r2_cute_small`, as the
/// setting is process-wide): its deadlock needs the MMA warp to run ahead. The walk values
/// change which barrier arrivals the reduction commutes, and so the depth-first order: the
/// deadlock lies within 1M states with them off, past 2.5M with them on. (Every later
/// arrival on that deadlock's trace is one the walk with values predicts at each earlier
/// state.) CM1 on SASS is explored exhaustively with them on (fixtures.rs).
///
/// The check is the exploration, then the E-TC-FENCE-NOP rerun (`tc_fence_contract`), which
/// changes the verdict only when it is complete. It runs as two tests in parallel, part 0 and
/// 1, that together pass exactly when the whole check does:
/// - part 0 explores; when the exploration is incomplete (the expectation has neither
///   "contract" nor "clean", and the rerun only turns races into contracts, so meeting it
///   after the rerun implies meeting it before) it checks the exploration's verdict, else
///   the whole check;
/// - part 1 runs the rerun; when it is incomplete (the verdict is then the exploration's,
///   which part 0 checks) it passes, else it runs the whole check.
fn r2_cute_small_cm1_ptx(part: u8) {
    let root = root();
    if ptxv_case::skip_missing_vendor_files(&root, &["r2_cute_small"]) {
        return;
    }
    let _s = Settings::take();
    set_bar_cut(Some(false));
    set_bar_values(Some(false));
    let case = Case::load(root.join("r2_cute_small")).unwrap();
    let k = kernel(&case, "CM1", "ptx");
    let launch = build_launch(&case, &k).unwrap();
    let m0 = Machine::new(Arc::new(k), Arc::new(launch));
    let expect = case.variant("CM1").unwrap().expect_for("ptx");
    let whole = |m0: Machine| {
        let mut v = explore(m0.clone(), 1_000_000);
        ptxv_sim::twolevel::tc_fence_contract(&mut v, &m0, &mut |m| explore(m, 1_000_000));
        assert!(
            v.meets(&expect),
            "r2_cute_small/CM1/ptx: got {:?}, want {expect:?}",
            v.classes()
        );
    };
    if part == 0 {
        let v = explore(m0.clone(), 1_000_000);
        if v.complete || expect.iter().any(|e| e == "contract" || e == "clean") {
            return whole(m0);
        }
        assert!(
            v.meets(&expect),
            "r2_cute_small/CM1/ptx: got {:?} (before the rerun), want {expect:?}",
            v.classes()
        );
    } else if let Some(m) = ptxv_sim::twolevel::tc_fence_rerun(&m0) {
        if explore(m, 1_000_000).complete {
            whole(m0);
        }
    }
}
#[test]
#[ignore = "heavy: r2_cute_small CM1 on PTX explored at 1M; 630 s"]
fn r2_cute_small_cm1_ptx_without_cut_and_values() {
    r2_cute_small_cm1_ptx(0);
}
#[test]
#[ignore = "heavy: r2_cute_small CM1 on PTX, the E-TC-FENCE-NOP rerun at 1M; 657 s"]
fn r2_cute_small_cm1_ptx_without_cut_and_values_tc_fence_rerun() {
    r2_cute_small_cm1_ptx(1);
}

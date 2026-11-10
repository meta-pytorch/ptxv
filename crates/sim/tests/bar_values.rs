/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! E-BAR-OVER walk values differential: the stateful explorer with and without the walk's
//! values (`ptxv_sim::explore::set_bar_values`) on every named-barrier fixture, both fronts;
//! and fixtures/r2_cute_small CM1 on PTX, bounded, without them. A separate test binary with
//! one test, because the setting is process-wide.

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::explore::Verdict;
use ptxv_sim::explore::explore;
use ptxv_sim::explore::set_bar_values;
use ptxv_sim::machine::Machine;

fn kernel(case: &Case, variant: &str, front: &str) -> ptxv_ir::Kernel {
    let dir = std::env::temp_dir().join(format!(
        "ptxv-test-barvalues-{}-{}-{variant}",
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

#[test]
fn bar_values_keep_every_verdict() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    let mut fewer = 0;
    for name in [
        "e_barwalk",
        "e_barover",
        "e_bararv",
        "e_aligned_bar",
        "e_bardiv_hang",
        "e_bardiv_short",
        "s_bardiv",
        "s_bardiv_mixed",
    ] {
        let case = Case::load(root.join(name)).unwrap();
        for var in case.m.variants.iter().map(|v| v.id.clone()) {
            let fronts: &[&str] = if case.m.cubin.is_some() {
                &["sass"]
            } else {
                &["ptx", "sass"]
            };
            for &front in fronts {
                let on = run(&case, &var, front, true);
                let off = run(&case, &var, front, false);
                let tag = format!("{name}/{var}/{front}");
                assert!(on.complete && off.complete, "{tag}: incomplete");
                assert_eq!(on.classes(), off.classes(), "{tag}: classes");
                assert_eq!(outputs(&on), outputs(&off), "{tag}: outputs");
                if name == "e_barwalk" {
                    let expect = case.variant(&var).unwrap().expect_for(front);
                    assert!(
                        on.meets(&expect),
                        "{tag}: got {:?}, want {expect:?}",
                        on.classes()
                    );
                }
                fewer += (on.states < off.states) as usize;
            }
        }
    }
    set_bar_values(None);
    eprintln!("fewer states with the walk values: {fewer} runs");
    // fixtures/r2_cute_small CM1 on PTX, bounded (moved from fixtures.rs `r2_cute_small`, as the
    // setting is process-wide): its deadlock needs the MMA warp to run ahead. The walk values
    // change which barrier arrivals the reduction commutes, and so the depth-first order: the
    // deadlock lies within 1M states with them off, past 2.5M with them on. (Every later
    // arrival on that deadlock's trace is one the walk with values predicts at each earlier
    // state.) CM1 on SASS is explored exhaustively with them on (fixtures.rs).
    if ptxv_case::skip_missing_vendor_files(&root, &["r2_cute_small"]) {
        return;
    }
    set_bar_values(Some(false));
    let case = Case::load(root.join("r2_cute_small")).unwrap();
    let k = kernel(&case, "CM1", "ptx");
    let launch = build_launch(&case, &k).unwrap();
    let m0 = Machine::new(Arc::new(k), Arc::new(launch));
    let mut v = explore(m0.clone(), 1_000_000);
    ptxv_sim::twolevel::tc_fence_contract(&mut v, &m0, &mut |m| explore(m, 1_000_000));
    let expect = case.variant("CM1").unwrap().expect_for("ptx");
    assert!(
        v.meets(&expect),
        "r2_cute_small/CM1/ptx: got {:?}, want {expect:?}",
        v.classes()
    );
    set_bar_values(None);
}

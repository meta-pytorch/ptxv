/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Limited residency (E-RESIDENCY) and the TMEM co-tenant (E-TMEM-COTENANT, E-TMEM-PERMIT):
//! fixtures/e_resid and fixtures/e_tmem_cotenant, both front-ends.

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::explore::explore;
use ptxv_sim::machine::Machine;
use ptxv_sim::residency;
use ptxv_sim::tenant;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn machine(case: &Case, variant: &str, front: &str) -> Machine {
    let ptx = case.variant_ptx(variant).unwrap();
    let k = match front {
        "ptx" => ptxv_ptx::parse_kernel(&ptx, Some(&case.m.entry)).unwrap(),
        _ => {
            // Tests run in parallel and may build the same (case, variant): a counter keeps
            // each build's directory distinct.
            static CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "ptxv-test-resid-{}-{variant}-{}-{}",
                case.m.name,
                std::process::id(),
                CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).unwrap();
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

fn variants(case: &Case) -> Vec<String> {
    case.m.variants.iter().map(|v| v.id.clone()).collect()
}

/// Bounds R that deadlock, by exhaustive exploration with each bound.
fn explored_deadlocks(m: &Machine) -> Vec<u32> {
    let n = m.launch.ctas();
    (1..=n)
        .filter(|&r| {
            let v = explore(residency::with_residency(m.clone(), r), 200_000);
            assert!(v.complete, "R = {r}: incomplete");
            v.findings.iter().any(|f| f.kind == "deadlock")
        })
        .collect()
}

/// Every variant is clean with every CTA resident, and needs the residency its case names:
/// the grid barrier and the ctaid chain all 4 CTAs, the ticket chain and the last-block
/// reduction one.
#[test]
fn e_resid_minimum_residency() {
    let case = Case::load(root().join("fixtures/e_resid")).unwrap();
    for v in variants(&case) {
        for front in ["ptx", "sass"] {
            let m = machine(&case, &v, front);
            let d = explore(m.clone(), 200_000);
            assert_eq!(d.classes(), ["clean"], "{v}/{front}");
            let rep = residency::analyze(&m, 200_000, 1 << 14);
            assert!(rep.proven, "{v}/{front}: {:?}", rep.lines());
            let want = case.variant(&v).unwrap().min_residency.unwrap();
            assert_eq!(rep.min_of(0), Some(want), "{v}/{front}: {:#?}", rep.lines());
        }
    }
}

/// The launch-set search (over deterministic runs, certified or not) finds exactly the bounds
/// that exhaustive exploration finds deadlocking.
#[test]
fn e_resid_launch_search_matches_exploration() {
    let case = Case::load(root().join("fixtures/e_resid")).unwrap();
    for v in variants(&case) {
        for front in ["ptx", "sass"] {
            let m = machine(&case, &v, front);
            let s = residency::launch_search(&m, 1 << 14);
            assert!(s.complete);
            let got: Vec<u32> = s
                .deadlocks
                .get(&0)
                .map(|d| d.keys().copied().collect())
                .unwrap_or_default();
            assert_eq!(got, explored_deadlocks(&m), "{v}/{front}");
        }
    }
}

/// A residency deadlock is reported as such; with every CTA allowed the verdict is the
/// default one; the default verdict names its reliance on co-residency; the certificate
/// refuses under limited residency.
#[test]
fn residency_verdicts_and_notes() {
    let case = Case::load(root().join("fixtures/e_resid")).unwrap();
    for front in ["ptx", "sass"] {
        let m = machine(&case, "good", front);
        let v = explore(residency::with_residency(m.clone(), 2), 200_000);
        assert_eq!(v.classes(), ["deadlock"]);
        let f = &v.findings[0];
        assert!(f.sig.starts_with("deadlock:residency"), "{}", f.sig);
        assert!(f.msg.contains("never becomes resident"), "{}", f.msg);
        assert!(v.assumptions.iter().any(|a| a == "E-RESIDENCY"));
        let full = explore(residency::with_residency(m.clone(), 4), 200_000);
        let d = explore(m.clone(), 200_000);
        assert_eq!(full.classes(), d.classes());
        assert_eq!(full.outputs, d.outputs);
        assert!(
            d.notes
                .iter()
                .any(|n| n.starts_with("relies on E-RESIDENCY")),
            "{:?}",
            d.notes
        );
        assert!(d.assumptions.iter().any(|a| a == "E-RESIDENCY"));
        // The certified chain carries the same note; under limited residency it refuses.
        let c = machine(&case, "chain", front);
        let o = ptxv_sim::wellsync::certify(c.clone());
        let cv = o.verdict.expect("chain certifies");
        assert!(
            cv.notes
                .iter()
                .any(|n| n.starts_with("relies on E-RESIDENCY"))
        );
        let o = ptxv_sim::wellsync::certify(residency::with_residency(c, 4));
        assert!(o.verdict.is_none());
        assert!(o.reasons[0].contains("E-RESIDENCY"));
    }
}

/// A case's `residency` makes `build_launch` start every CTA unlaunched under that bound.
#[test]
fn case_residency_option() {
    let mut case = Case::load(root().join("fixtures/e_resid")).unwrap();
    case.m.residency = Some(3);
    let m = machine(&case, "chain", "ptx");
    assert_eq!(m.launch.env.residency, Some(3));
    assert_eq!(explore(m, 200_000).classes(), ["deadlock"]);
}

/// Co-tenant verdicts: the case's `expect_cotenant` for every variant, and the
/// happens-before analysis agrees with the explored pair placement on whether the two CTAs
/// deadlock on one SM.
#[test]
fn e_tmem_cotenant_verdicts() {
    let case = Case::load(root().join("fixtures/e_tmem_cotenant")).unwrap();
    for v in variants(&case) {
        for front in ["ptx", "sass"] {
            let m = machine(&case, &v, front);
            assert_eq!(
                explore(m.clone(), 200_000).classes(),
                ["clean"],
                "{v}/{front}: default"
            );
            let t = tenant::check(&m, 200_000, 8);
            assert!(t.complete, "{v}/{front}");
            let want = case.variant(&v).unwrap().expect_cotenant.clone().unwrap();
            assert_eq!(t.classes(), want, "{v}/{front}: {:#?}", t.findings);
            let (hb, _) = tenant::hb_findings(&m);
            let pair = explore(tenant::with_tenant(m.clone(), vec![(0, 1)]), 200_000);
            assert_eq!(
                hb.iter().any(|f| f.kind == "deadlock"),
                pair.findings.iter().any(|f| f.kind == "deadlock"),
                "{v}/{front}: {hb:#?} vs {:#?}",
                pair.findings
            );
            assert_eq!(
                hb.iter().any(|f| f.sig.starts_with("tmem-relinquish")),
                want.iter().any(|c| c == "contract"),
                "{v}/{front}"
            );
        }
    }
}

/// Without the permit model only columns count: the missing relinquish alone no longer
/// deadlocks (norelinq), the 512-column pair still does (big).
#[test]
fn e_tmem_cotenant_columns_only() {
    let mut case = Case::load(root().join("fixtures/e_tmem_cotenant")).unwrap();
    case.m.tmem_permit = Some(false);
    for (v, dl) in [("norelinq", false), ("big", true), ("good", false)] {
        let m = machine(&case, v, "sass");
        let pair = explore(tenant::with_tenant(m.clone(), vec![(0, 1)]), 200_000);
        assert_eq!(
            pair.findings.iter().any(|f| f.kind == "deadlock"),
            dl,
            "{v}: {:#?}",
            pair.findings
        );
        let (hb, _) = tenant::hb_findings(&m);
        assert_eq!(hb.iter().any(|f| f.kind == "deadlock"), dl, "{v}: {hb:#?}");
    }
}

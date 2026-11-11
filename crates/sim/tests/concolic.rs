// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Concolic input exploration (solve.rs): solvers flip what B flips and more, every reported
//! flip replays, injected solver faults never reach the report.

mod cc_common;

use cc_common::*;
use ptxv_sim::machine::Machine;

/// The concolic run and search on `m` (no invariants unless given).
fn concolic(
    m: &Machine,
    dom: &ptxv_sim::coverage::Domain,
    opts: &ptxv_sim::solve::Options,
) -> (ptxv_sim::coverage::Coverage, ptxv_sim::solve::Report, usize) {
    let (cov, s) = ptxv_sim::coverage::analyze_with(m, Some(Default::default()));
    let (sym, mfin) = s.unwrap();
    let rep = ptxv_sim::solve::run(m, &cov, &sym, &mfin, dom, opts);
    (cov, rep, sym.n_mismatch)
}

/// The bucket-fill sentinel shape: the untaken `ok != 0` false side is flipped (T0) and the
/// replay confirms it; within the declared invariants it writes ok = 0.
#[test]
fn sentinel_fill_flipped_and_replayed() {
    for front in ["ptx", "sass"] {
        for v in ["good", "fixed", "derived"] {
            let m = fixture("e_sentinel_fill", v, front);
            let (cov, rep, mism) = concolic(&m, &Default::default(), &Default::default());
            assert_eq!(mism, 0, "{v}/{front}");
            assert_eq!(cov.untaken.len(), 1, "{v}/{front}");
            assert_eq!(rep.flipped, 1, "{v}/{front}: {rep:#?}");
            let s = &rep.sides[0];
            assert_eq!(s.outcome, "flipped");
            println!(
                "{v}/{front}: {} by {:?}: {}",
                s.condition,
                s.solver,
                s.witness.as_deref().unwrap_or("")
            );
        }
    }
}

/// Print a report's sides (debugging aid for the tests below).
fn show(tag: &str, rep: &ptxv_sim::solve::Report) {
    for s in &rep.sides {
        println!(
            "{tag}: {} never={} {} solver={:?} witness={:?} b={:?} reason={:?}\n      {}",
            s.at, s.never, s.outcome, s.solver, s.witness, s.b_flip, s.reason, s.condition
        );
    }
    println!("{tag}: solvers {:?}", rep.solvers);
}

/// e_concolic: every condition shape of the plan. CC flips each side except `a * 6 == 9`
/// (solver-unsat, T0's exact inversion), every flip replays, and CC's flips are a superset of
/// B's (replayed).
#[test]
fn concolic_condition_shapes() {
    for front in ["ptx", "sass"] {
        let m = fixture("e_concolic", "good", front);
        let dom = ptxv_sim::coverage::Domain::default();
        let (cov, mut rep, mism) = concolic(&m, &dom, &Default::default());
        let b = ptxv_sim::solve::replay_b(&m, &cov, &dom, 2);
        ptxv_sim::solve::attach_b(&mut rep, &b);
        show(front, &rep);
        assert_eq!(mism, 0, "{front}");
        // Every reported flip replays (the search only reports replayed ones; check again).
        for s in rep.sides.iter().filter(|s| s.outcome == "flipped") {
            assert!(
                ptxv_sim::solve::replay(&m, s.pc, s.never, &s.writes),
                "{front}: {s:?}"
            );
        }
        // Superset of B's replayed flips.
        for s in &rep.sides {
            if s.b_flip == Some(true) {
                assert_eq!(
                    s.outcome, "flipped",
                    "{front}: B flips {} but CC does not: {s:?}",
                    s.at
                );
            }
        }
        let unsat: Vec<_> = rep
            .sides
            .iter()
            .filter(|s| s.outcome == "solver-unsat")
            .collect();
        let flipped = rep.sides.iter().filter(|s| s.outcome == "flipped").count();
        // PTX: nine sites plus the combining one; ptxas may fold the unsatisfiable compare.
        assert!(unsat.len() <= 1, "{front}: {unsat:?}");
        assert_eq!(flipped + unsat.len(), rep.sides.len(), "{front}");
        if front == "ptx" {
            assert_eq!(unsat.len(), 1, "{front}");
            assert!(unsat[0].condition.contains("Mul"), "{unsat:?}");
        }
        // `a * 6` spans the width, so interval analysis does not refute it: not proved, the
        // summary counts it unverified
        for u in &unsat {
            assert_ne!(u.solver.as_deref(), Some("interval"), "{u:?}");
            assert!(u.reason.as_deref().unwrap().contains("unverified"), "{u:?}");
        }
        assert_eq!(
            (rep.solver_unsat, rep.solver_unsat_proved),
            (unsat.len(), 0),
            "{front}"
        );
    }
}

/// Mutation test of the solvers: with a wrong modular inverse, with the evaluation check
/// disabled (the replay the only gate) and with corrupted witness values, no side is reported
/// flipped unless its witness replays; the modular-arithmetic side is not flipped with the
/// broken inverse and no evaluation check.
#[test]
fn solver_faults_never_reach_the_report() {
    use ptxv_sim::solve::Fault;
    use ptxv_sim::solve::Options;
    for front in ["ptx", "sass"] {
        for fixture_name in ["e_concolic", "e_sentinel_fill"] {
            let m = fixture(fixture_name, "good", front);
            for fault in [Fault::ModInv, Fault::ModInvNoEval, Fault::Writes] {
                let opts = Options {
                    fault,
                    smt: false,
                    ..Default::default()
                };
                let (_, rep, _) = concolic(&m, &Default::default(), &opts);
                let mut rejected = 0;
                for s in &rep.sides {
                    if s.outcome == "flipped" || s.outcome == "outside-invariants" {
                        assert!(
                            ptxv_sim::solve::replay(&m, s.pc, s.never, &s.writes),
                            "{fixture_name}/{front} {fault:?}: reported flip does not replay: {s:?}"
                        );
                    }
                    if fault == Fault::ModInvNoEval && s.condition.contains("0x9e3779b1") {
                        assert_ne!(s.outcome, "flipped", "{front}: {s:?}");
                    }
                }
                for c in rep.solvers.values() {
                    rejected += c.rejected;
                }
                println!(
                    "{fixture_name}/{front} {fault:?}: flipped {} unknown {} rejected {rejected}",
                    rep.flipped, rep.unknown
                );
                if fault != Fault::ModInv && fixture_name == "e_concolic" {
                    assert!(
                        rejected > 0,
                        "{fixture_name}/{front} {fault:?}: the replay rejected nothing"
                    );
                }
            }
        }
    }
}

/// The SMT step alone (feature `smt`, z3 on PATH): the product of three input words is
/// flipped by it, `a * 6 == 9` is solver-unsat (unverified), every flip replays.
#[cfg(feature = "smt")]
#[test]
fn smt_step_flips_and_reports_unsat() {
    if std::process::Command::new(std::env::var("PTXV_SMT_SOLVER").unwrap_or("z3".into()))
        .arg("-version")
        .output()
        .is_err()
    {
        println!("no SMT solver: skipped");
        return;
    }
    for front in ["ptx", "sass"] {
        let m = fixture("e_concolic", "smt", front);
        let opts = ptxv_sim::solve::Options {
            only: vec!["smt".into()],
            smt: true,
            ..Default::default()
        };
        let (_, rep, _) = concolic(&m, &Default::default(), &opts);
        show(front, &rep);
        for s in rep.sides.iter().filter(|s| s.outcome == "flipped") {
            assert!(
                ptxv_sim::solve::replay(&m, s.pc, s.never, &s.writes),
                "{front}: {s:?}"
            );
            assert_eq!(s.solver.as_deref(), Some("smt"));
        }
        let prod = rep
            .sides
            .iter()
            .find(|s| s.condition.contains("0x13579bdf"))
            .unwrap();
        assert_eq!(prod.outcome, "flipped", "{front}: {prod:?}");
        if front == "ptx" {
            let six = rep
                .sides
                .iter()
                .find(|s| s.condition.contains("Mul(a[0], 6)"))
                .unwrap();
            assert_eq!(six.outcome, "solver-unsat", "{six:?}");
        }
    }
}

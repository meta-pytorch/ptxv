// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Exhaustive concolic expansion (cert.rs): the good kernel of fixtures/e_exh_concolic is
//! certified; no mutant with a bug behind an untaken side is (each is refuted, or blocked
//! when the expansion cannot close); an input-dependent address blocks. Needs two of the
//! SMT solvers z3, cvc5, bitwuzla (PTXV_EXH_SOLVERS); skipped without them.

mod cc_common;

use cc_common::*;
use ptxv_sim::cert::Options;
use ptxv_sim::cert::RepCheck;
use ptxv_sim::cert::expand;
use ptxv_sim::machine::Machine;

fn check(m: &Machine) -> RepCheck {
    let v = ptxv_sim::explore::explore(m.clone(), 200_000);
    RepCheck {
        classes: v.classes(),
        complete: v.complete,
        certified: false,
        states: v.states,
        findings: v
            .findings
            .iter()
            .take(2)
            .map(|f| f.msg.lines().next().unwrap_or("").to_string())
            .collect(),
        assumes: v.assumptions.clone(),
        seconds: v.seconds,
    }
}

fn run(variant: &str, front: &str, max_paths: usize) -> ptxv_sim::cert::Report {
    run_with(variant, front, max_paths, None)
}

fn run_with(
    variant: &str,
    front: &str,
    max_paths: usize,
    solvers: Option<Vec<ptxv_sim::certsmt::Solver>>,
) -> ptxv_sim::cert::Report {
    let m = fixture("e_exh_concolic", variant, front);
    let opts = Options {
        max_paths,
        budget: 600.0,
        solvers,
        ..Default::default()
    };
    expand(&m, &Default::default(), &opts, &mut check)
}

#[test]
fn certified_refuted_blocked() {
    if ptxv_sim::certsmt::solvers().len() < 2 {
        println!("fewer than two SMT solvers: skipped");
        return;
    }
    for front in ["ptx", "sass"] {
        let r = run("good", front, 8);
        assert_eq!(r.status, "certified", "{front}: {:#?}", r.blockers);
        assert_eq!(r.paths.len(), 4, "{front}");
        for v in [
            "race_eq",
            "race_nan",
            "race_combo",
            "race_fmul",
            "race_hash",
            "fault_oob",
        ] {
            let r = run(v, front, 8);
            assert_eq!(r.status, "refuted", "{v}/{front}: {:#?}", r.blockers);
        }
        // Eight trip counts: a small cap blocks (or the race is on an early path); a larger one
        // refutes.
        let r8 = run("race_loop", front, 8).status;
        assert!(r8 == "blocked" || r8 == "refuted", "{front}: {r8}");
        assert_eq!(run("race_loop", front, 64).status, "refuted", "{front}");
        let r = run("addr", front, 8);
        assert_eq!(r.status, "blocked", "{front}");
        assert!(
            r.blockers.iter().any(|b| b.kind == "operand-varies"),
            "{front}: {:#?}",
            r.blockers
        );
    }
}

/// Value faults (C-EXH-TRANSFER, C-EXH-CONTROL): a division by an input word, zero only off
/// the captured input, unguarded (div0) or under an input-dependent guard (div0g). The divisor
/// is an operand fixed per path, and the guard of a faulting operation is a decision: never
/// certified on the PTX front (until 2026-11-10 both certified; m = 0 faults). ptxas's
/// division sequence does not trap, so the SASS front has no value-fault operand there.
#[test]
fn value_faults_are_never_certified() {
    if ptxv_sim::certsmt::solvers().len() < 2 {
        println!("fewer than two SMT solvers: skipped");
        return;
    }
    for v in ["div0", "div0g"] {
        let r = run(v, "ptx", 8);
        assert_ne!(r.status, "certified", "{v}: {:#?}", r.blockers);
        assert!(
            r.status == "refuted" || r.blockers.iter().any(|b| b.kind == "operand-varies"),
            "{v}: {} {:#?}",
            r.status,
            r.blockers
        );
    }
}

/// Clock reads (a fresh value per read on hardware, 0 under E-CLOCK): two reads must not be
/// one leaf, and the IR has no clock leaf at all, so the record blocks on `clock-read`.
#[test]
fn clock_reads_block() {
    if ptxv_sim::certsmt::solvers().len() < 2 {
        println!("fewer than two SMT solvers: skipped");
        return;
    }
    for front in ["ptx", "sass"] {
        let r = run("clock", front, 8);
        assert_ne!(r.status, "certified", "{front}");
        assert!(
            r.blockers.iter().any(|b| b.kind == "clock-read"),
            "{front}: {:#?}",
            r.blockers
        );
    }
}

/// Repeat closure through negation (cert.rs `norm`): d2's guard again, negated, is the same
/// atom; each of the four paths closes it as a repeat, without a solver.
#[test]
fn negated_repeat_closes_without_a_solver() {
    if ptxv_sim::certsmt::solvers().len() < 2 {
        println!("fewer than two SMT solvers: skipped");
        return;
    }
    for front in ["ptx", "sass"] {
        let r = run("neg", front, 8);
        assert_eq!(r.status, "certified", "{front}: {:#?}", r.blockers);
        assert_eq!(r.paths.len(), 4, "{front}");
        assert!(r.closed_repeat >= 4, "{front}: {} repeats", r.closed_repeat);
        assert_eq!(r.closed_infeasible, 0, "{front}");
    }
}

/// C-EXH-UNSAT: two solvers that answer unsat to everything. The prefix sentinel (the path's
/// decisions alone, which the captured input satisfies) drops their answers, so no region is
/// closed by them and the race is still refuted.
#[test]
fn lying_solvers_close_nothing() {
    let dir = std::env::temp_dir().join(format!("ptxv-exh-liar-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let liar = dir.join("liar.sh");
    std::fs::write(
        &liar,
        concat!(
            "#!/bin/sh\n",
            "if [ \"$1\" = --version ]; then echo liar 0; exit 0; fi\n",
            "awk '/^\\(echo /{gsub(/[()\"]/,\"\"); print $2} /^\\(check-sat\\)/{print \"unsat\"}'\n",
        ),
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&liar, std::fs::Permissions::from_mode(0o755)).unwrap();
    let cmd = liar.to_string_lossy().to_string();
    let liars: Vec<ptxv_sim::certsmt::Solver> = ["liar1", "liar2"]
        .iter()
        .map(|n| ptxv_sim::certsmt::Solver {
            name: n.to_string(),
            cmd: cmd.clone(),
            version: "liar 0".into(),
        })
        .collect();
    for front in ["ptx", "sass"] {
        let r = run_with("race_eq", front, 8, Some(liars.clone()));
        assert_eq!(r.status, "refuted", "{front}: {:#?}", r.blockers);
        assert_eq!(r.closed_infeasible, 0, "{front}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Open misses (cert.rs `OPEN_MISSES`): the good kernel matches no trigger; the clock variant
/// matches MISS-28 on both fronts.
#[test]
fn open_miss_triggers() {
    for front in ["ptx", "sass"] {
        let good = fixture("e_exh_concolic", "good", front);
        let hit: Vec<&str> = ptxv_sim::cert::OPEN_MISSES
            .iter()
            .filter(|x| (x.trigger)(&good).is_some())
            .map(|x| x.id)
            .collect();
        assert!(hit.is_empty(), "{front}: {hit:?}");
        let clock = fixture("e_exh_concolic", "clock", front);
        assert!(
            ptxv_sim::cert::OPEN_MISSES
                .iter()
                .any(|x| x.id == "MISS-28" && (x.trigger)(&clock).is_some()),
            "{front}"
        );
    }
}

/// C-EXH-UNSAT: infeasible needs at least two unsat answers and no sat.
#[test]
fn proven_unsat_needs_two_solvers_and_no_sat() {
    use ptxv_sim::certsmt::Answer;
    use ptxv_sim::certsmt::Verdict;
    let v = |a: &[Answer]| Verdict {
        answers: a
            .iter()
            .enumerate()
            .map(|(i, x)| (format!("s{i}"), *x))
            .collect(),
    };
    assert!(!v(&[Answer::Unsat]).proven_unsat());
    assert!(!v(&[Answer::Unsat, Answer::Unknown]).proven_unsat());
    assert!(v(&[Answer::Unsat, Answer::Unsat]).proven_unsat());
    assert!(!v(&[Answer::Unsat, Answer::Unsat, Answer::Sat]).proven_unsat());
    assert!(v(&[Answer::Unsat, Answer::Unsat, Answer::Unknown]).proven_unsat());
}

/// C-EXH-UNSAT: the exact parts of the certificate encoding (certsmt.rs) give every node the
/// value symb.rs computes, at random leaf values, for every solver found.
#[test]
fn exact_encoding_matches_eval_node() {
    use ptxv_ir::Cmp;
    use ptxv_ir::FCmp;
    use ptxv_sim::symb::Arena;
    use ptxv_sim::symb::Bin;
    use ptxv_sim::symb::Node;
    use ptxv_sim::symb::Un;
    let solvers = ptxv_sim::certsmt::solvers();
    if solvers.is_empty() {
        println!("no SMT solver: skipped");
        return;
    }
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let specials = [
        0u64,
        1,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0x8000_0000,
        0x0000_0001,
        0x3f80_0000,
        0xffff_ffff,
        31,
        32,
        33,
    ];
    let mut checked = 0;
    for round in 0..40 {
        let mut ar = Arena::default();
        let mut pool32 = Vec::new();
        for i in 0..4u64 {
            let v = if next() % 3 == 0 {
                specials[(next() % specials.len() as u64) as usize]
            } else {
                next() & 0xffff_ffff
            };
            pool32.push(ar.input(0, 4 * i, 4, v));
        }
        let d0 = ar.input(
            1,
            0,
            8,
            if round % 2 == 0 {
                0x7ff8_0000_0000_0000
            } else {
                next()
            },
        );
        let d1 = ar.input(1, 8, 8, next());
        let mut nodes = Vec::new();
        for _ in 0..40 {
            let pick = |p: &Vec<u32>, r: u64| p[(r % p.len() as u64) as usize];
            let (a, b) = (pick(&pool32, next()), pick(&pool32, next()));
            let e = match next() % 20 {
                0 => ar.bin(Bin::Add, a, b),
                1 => ar.bin(Bin::Sub, a, b),
                2 => ar.bin(Bin::Mul, a, b),
                3 => ar.bin(Bin::MulHiU, a, b),
                4 => ar.bin(Bin::MulHiS, a, b),
                5 => ar.bin(Bin::Xor, a, b),
                6 => {
                    let k = ar.konst(32, next() % 40);
                    ar.bin(Bin::Shl, a, k)
                }
                7 => {
                    let k = ar.konst(32, next() % 40);
                    ar.bin(Bin::AShr, a, k)
                }
                8 => ar.bin(Bin::LShr, a, b),
                9 => ar.bin(Bin::SMin, a, b),
                10 => ar.bin(Bin::UMax, a, b),
                11 => ar.lop3((next() & 0xff) as u8, a, b, pick(&pool32, next())),
                12 => {
                    let c = ar.cmp(Cmp::LtS, a, b);
                    ar.ite(c, a, b)
                }
                13 => {
                    let c = [
                        FCmp::Lt,
                        FCmp::Equ,
                        FCmp::Ne,
                        FCmp::Num,
                        FCmp::Geu,
                        FCmp::Nan,
                    ][(next() % 6) as usize];
                    let f = ar.fcmp(c, false, a, b);
                    ar.zext(32, f)
                }
                14 => ar.un(Un::FNeg, a),
                15 => ar.un(Un::FAbs, a),
                16 => {
                    let h = ar.extract((next() % 17) as u8, 16, a);
                    ar.un(Un::B2F, h)
                }
                17 => {
                    let x = ar.concat(a, b);
                    ar.extract((next() % 32) as u8, 32, x)
                }
                18 => {
                    let c = [FCmp::Le, FCmp::Neu, FCmp::Gt][(next() % 3) as usize];
                    let f = ar.mk_pub(
                        Node::DCmp {
                            cmp: c,
                            a: d0,
                            b: d1,
                        },
                        1,
                    );
                    ar.sext(32, f)
                }
                _ => {
                    let x = ar.extract(0, 8, a);
                    ar.sext(32, x)
                }
            };
            if ar.w(e) == 32 {
                pool32.push(e);
            }
            nodes.push(e);
        }
        let leaves: std::collections::HashMap<u32, u64> = (0..ar.len() as u32)
            .filter(|&x| matches!(ar.nodes[x as usize], Node::Input { .. }))
            .map(|x| (x, ar.conc[x as usize]))
            .collect();
        for s in &solvers {
            let got = ptxv_sim::certsmt::eval_with(s, &ar, &nodes, &leaves)
                .unwrap_or_else(|| panic!("{}: no answer", s.name));
            for (e, g) in nodes.iter().zip(got) {
                assert_eq!(
                    g,
                    Some(ar.conc[*e as usize]),
                    "{} round {round}: {}",
                    s.name,
                    ar.show(*e, &|a| format!("a{a}"), 4)
                );
                checked += 1;
            }
        }
    }
    println!(
        "{checked} node values checked against {} solver(s)",
        solvers.len()
    );
}

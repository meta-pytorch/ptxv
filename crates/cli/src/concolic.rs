// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! `ptxv check --concolic` (ptxv_sim::symb, ptxv_sim::solve): the coverage run with
//! expressions, the search over the untaken sides, and the confirmed witnesses for the checks
//! on mutated inputs. Off unless asked for: the default check runs `coverage::analyze` only.

use ptxv_sim::coverage::Coverage;
use ptxv_sim::coverage::Domain;
use ptxv_sim::coverage::Mutation;
use ptxv_sim::machine::Machine;

/// `--concolic`.
pub static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

pub fn on() -> bool {
    ON.get().copied().unwrap_or(false)
}

/// The coverage run (A), with expressions when `--concolic`.
pub fn analyze(
    m: &Machine,
    b: ptxv_sim::coverage::Budget,
) -> (Coverage, Option<(Box<ptxv_sim::symb::SymState>, Machine)>) {
    if on() {
        ptxv_sim::coverage::analyze_sym(m, b, Some(Default::default()))
    } else {
        (ptxv_sim::coverage::analyze_budget(m, b), None)
    }
}

fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k)
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(d)
}

/// The search over `cov`'s untaken sides (`cov.concolic`), printed; the confirmed witnesses
/// (at most `PTXV_CONCOLIC_CHECKS`, default 4) as mutations to check.
pub fn search(
    m: &Machine,
    cov: &mut Coverage,
    sym: Option<(Box<ptxv_sim::symb::SymState>, Machine)>,
    dom: &Domain,
    k: usize,
) -> Vec<Mutation> {
    let Some((sym, mfin)) = sym else {
        return vec![];
    };
    let opts = ptxv_sim::solve::Options {
        k,
        budget: env_f64("PTXV_CONCOLIC_BUDGET", 600.0),
        max_replays: env_f64("PTXV_CONCOLIC_REPLAYS", 400.0) as usize,
        smt: std::env::var("PTXV_CONCOLIC_SMT").map_or(cfg!(feature = "smt"), |v| v != "0"),
        only: std::env::var("PTXV_CONCOLIC_SOLVERS")
            .map(|v| {
                v.split(',')
                    .map(|x| x.trim().to_string())
                    .filter(|x| !x.is_empty())
                    .collect()
            })
            .unwrap_or_default(),
        ..Default::default()
    };
    let mut rep = ptxv_sim::solve::run(m, cov, &sym, &mfin, dom, &opts);
    rep.seconds_run = cov.seconds;
    if std::env::var("PTXV_CONCOLIC_DIFF").is_ok() {
        let b = ptxv_sim::solve::replay_b(m, cov, dom, k);
        ptxv_sim::solve::attach_b(&mut rep, &b);
    }
    println!(
        "    concolic: {} side(s): {} flipped, {} outside declared invariants, {} solver-unsat (unverified), {} unknown; {} replays, {} nodes, {} mismatch(es) ({:.1}s)",
        rep.sides.len(),
        rep.flipped,
        rep.outside_invariants,
        rep.solver_unsat,
        rep.unknown,
        rep.replays,
        rep.nodes,
        rep.mismatches,
        rep.seconds_search
    );
    for s in rep.sides.iter().take(20) {
        match s.outcome.as_str() {
            "flipped" | "outside-invariants" => println!(
                "    concolic: {} side of {} {} by {} ({}): {}",
                if s.never { "true" } else { "false" },
                s.at,
                if s.outcome == "flipped" {
                    "flipped"
                } else {
                    "flipped outside declared invariants"
                },
                s.solver.as_deref().unwrap_or("?"),
                if s.within_invariants {
                    "within declared invariants, replayed"
                } else {
                    "replayed"
                },
                s.witness.as_deref().unwrap_or("")
            ),
            _ => println!(
                "    concolic: {} side of {}: {}: {}",
                if s.never { "true" } else { "false" },
                s.at,
                s.outcome,
                s.reason.as_deref().unwrap_or("")
            ),
        }
    }
    let n = env_f64("PTXV_CONCOLIC_CHECKS", 4.0) as usize;
    let muts: Vec<Mutation> = ptxv_sim::solve::mutations(m, &rep, dom)
        .into_iter()
        .take(n)
        .collect();
    cov.concolic = Some(rep);
    muts
}

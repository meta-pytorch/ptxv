// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Target-neutral pieces of `--concolic` front-ends (ptxv_sim::solve): the search options from
//! the environment (`PTXV_CONCOLIC_*`), the number of confirmed witnesses checked as mutated
//! inputs, and the printed summary of a `solve::Report`. Shared by the `ptxv` binary and other
//! front-end binaries built on the ptxv crates.

fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k)
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(d)
}

/// Confirmed witnesses checked as mutated inputs (`PTXV_CONCOLIC_CHECKS`, default 4).
pub fn checks() -> usize {
    env_f64("PTXV_CONCOLIC_CHECKS", 4.0) as usize
}

/// The search options from the environment (`PTXV_CONCOLIC_*`), `k` elements per mutation.
pub fn options(k: usize) -> ptxv_sim::solve::Options {
    ptxv_sim::solve::Options {
        k,
        budget: env_f64("PTXV_CONCOLIC_BUDGET", 600.0),
        max_replays: env_f64("PTXV_CONCOLIC_REPLAYS", 400.0) as usize,
        smt: std::env::var("PTXV_CONCOLIC_SMT")
            .map_or(ptxv_sim::solve::Options::default().smt, |v| v != "0"),
        only: std::env::var("PTXV_CONCOLIC_SOLVERS")
            .map(|v| {
                v.split(',')
                    .map(|x| x.trim().to_string())
                    .filter(|x| !x.is_empty())
                    .collect()
            })
            .unwrap_or_default(),
        ..Default::default()
    }
}

/// The search's summary and its first 20 sides.
pub fn print(rep: &ptxv_sim::solve::Report) {
    println!(
        "    concolic: {} side(s): {} flipped, {} outside declared invariants, {} solver-unsat ({} proved by interval analysis, {} unverified), {} unknown; {} replays, {} nodes, {} mismatch(es) ({:.1}s)",
        rep.sides.len(),
        rep.flipped,
        rep.outside_invariants,
        rep.solver_unsat,
        rep.solver_unsat_proved,
        rep.solver_unsat - rep.solver_unsat_proved,
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
        for n in &s.notes {
            println!("      note: {n}");
        }
    }
    if rep.truncated_paths + rep.partial_paths > 0 {
        println!(
            "    concolic: {} thread path(s) truncated at the atom cap, {} partial (bounded prefix)",
            rep.truncated_paths, rep.partial_paths
        );
    }
}

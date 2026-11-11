// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! `ptxv check --concolic` (ptxv_sim::symb, ptxv_sim::solve): the coverage run with
//! expressions, the search over the untaken sides, and the confirmed witnesses for the checks
//! on mutated inputs. Off unless asked for: the default check runs `coverage::analyze` only.

use ptxv_concolic::checks;
use ptxv_concolic::options;
use ptxv_concolic::print;
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
    let opts = options(k);
    let mut rep = ptxv_sim::solve::run(m, cov, &sym, &mfin, dom, &opts);
    rep.seconds_run = cov.seconds;
    if std::env::var("PTXV_CONCOLIC_DIFF").is_ok() {
        let b = ptxv_sim::solve::replay_b(m, cov, dom, k);
        ptxv_sim::solve::attach_b(&mut rep, &b);
    }
    print(&rep);
    let n = checks();
    let muts: Vec<Mutation> = ptxv_sim::solve::mutations(m, &rep, dom)
        .into_iter()
        .take(n)
        .collect();
    cov.concolic = Some(rep);
    muts
}

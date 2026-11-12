/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The time budgets (`membudget`) bound the synchronization reduction's seed walks: once the
//! deadline has passed no walk takes a step, and the verdict is incomplete with a note naming
//! the walks. The whole check's budget (`set_time_budget`, `--time-budget`) also stops
//! certificates, which then do not hold. A separate test binary, because the budgets are
//! process-wide (the tests take `LOCK`).

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::time::Duration;
use std::time::Instant;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::explore::Verdict;
use ptxv_sim::explore::explore;
use ptxv_sim::explore::explore_threads;
use ptxv_sim::machine::Machine;
use ptxv_sim::membudget;
use ptxv_sim::wellsync::certify;

/// The budgets are process-wide: one test at a time, with no budget on entry and on exit
/// (also when it fails).
static LOCK: Mutex<()> = Mutex::new(());

struct Budgets(#[allow(dead_code)] MutexGuard<'static, ()>);

impl Drop for Budgets {
    fn drop(&mut self) {
        membudget::set_time_budget(None);
        membudget::set_deadline(None);
    }
}

fn budgets() -> Budgets {
    let g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    membudget::set_time_budget(None);
    membudget::set_deadline(None);
    Budgets(g)
}

/// A fixture's good variant through the PTX front-end.
fn machine(name: &str) -> Machine {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let case = Case::load(root.join("fixtures").join(name)).unwrap();
    let ptx = case.variant_ptx("good").unwrap();
    let toolkit = case
        .m
        .ptxas_toolkit
        .as_deref()
        .and_then(ptxv_ptx::toolkit_version);
    let k = ptxv_ptx::parse_kernel_for(&ptx, Some(&case.m.entry), toolkit).unwrap();
    let launch = build_launch(&case.for_variant("good").unwrap(), &k).unwrap();
    Machine::new(Arc::new(k), Arc::new(launch))
}

/// The note of a verdict whose seed walks a budget stopped, if any.
fn seed_note(v: &Verdict) -> Option<&String> {
    v.notes.iter().find(|n| n.contains("seed walks stopped at"))
}

#[test]
fn seed_walks_stop_at_the_deadline() {
    let _b = budgets();
    // e_cert_fasttrack: 128 threads, bar.sync and red.add; the reduction is on, so the walks
    // run before the first round.
    let m = machine("e_cert_fasttrack");
    // Control, no deadline: complete and clean, no note.
    membudget::set_deadline(None);
    let v0 = explore(m.clone(), 2_000_000);
    assert!(
        v0.complete && v0.classes() == ["clean"] && seed_note(&v0).is_none(),
        "control: {:?} complete={} notes={:?}",
        v0.classes(),
        v0.complete,
        v0.notes
    );
    // A deadline that has already passed, with the explorer's threads and with one thread
    // (one random stream): the walks stop before their first step.
    for one in [false, true] {
        membudget::set_deadline(Some(Instant::now()));
        let v = if one {
            explore_threads(m.clone(), 2_000_000, 1)
        } else {
            explore(m.clone(), 2_000_000)
        };
        membudget::set_deadline(None);
        let note = seed_note(&v).unwrap_or_else(|| {
            panic!(
                "one thread: {one}: no seed-walk note: {:?} complete={} notes={:?}",
                v.classes(),
                v.complete,
                v.notes
            )
        });
        assert!(
            note.contains("the time budget") && note.contains("(0 steps)"),
            "one thread: {one}: {note}"
        );
        assert!(!v.complete, "one thread: {one}: {:?}", v.notes);
        assert_eq!(v.classes(), ["inconclusive"], "one thread: {one}");
    }
    // Cleared again: the control's verdict.
    let v2 = explore(m, 2_000_000);
    assert!(
        v2.complete && v2.classes() == v0.classes() && seed_note(&v2).is_none(),
        "after: {:?} complete={} notes={:?}",
        v2.classes(),
        v2.complete,
        v2.notes
    );
}

/// The note of a verdict an exploration deadline stopped, if any.
fn deadline_note(v: &Verdict) -> Option<&String> {
    v.notes
        .iter()
        .find(|n| n.starts_with("exploration stopped at"))
}

#[test]
fn certificates_stop_at_the_time_budget() {
    let _b = budgets();
    let m = machine("e_cert_fasttrack");
    // Control, no budget: the certificate holds.
    let o0 = certify(m.clone());
    assert!(
        o0.verdict.is_some() && o0.steps > 0,
        "control: {:?}",
        o0.reasons
    );
    // The whole check's budget already spent: the certificate's run takes no step and the
    // certificate does not hold; the exploration that would replace it stops too.
    membudget::set_time_budget(Some(Instant::now()));
    assert_eq!(membudget::time_budget_left(), Some(Duration::ZERO));
    let o = certify(m.clone());
    let v = explore(m.clone(), 2_000_000);
    membudget::set_time_budget(None);
    assert!(o.verdict.is_none(), "spent budget: the certificate held");
    assert_eq!(o.steps, 0, "spent budget: {:?}", o.reasons);
    assert!(
        o.reasons.iter().any(|r| r.contains(
            "the certificate's run stopped at the time budget (--time-budget) after 0 steps"
        )),
        "{:?}",
        o.reasons
    );
    assert!(
        !v.complete && v.classes() == ["inconclusive"],
        "{:?}",
        v.notes
    );
    assert_eq!(
        deadline_note(&v).map(String::as_str),
        Some("exploration stopped at the time budget (--time-budget): incomplete")
    );
    // The nested deadline alone (--mutate-budget-seconds) stops explorations, not
    // certificates (as before --time-budget).
    membudget::set_deadline(Some(Instant::now()));
    let o2 = certify(m.clone());
    let v2 = explore(m.clone(), 2_000_000);
    membudget::set_deadline(None);
    assert!(o2.verdict.is_some(), "nested deadline: {:?}", o2.reasons);
    assert!(!v2.complete, "nested deadline: {:?}", v2.notes);
    assert!(
        deadline_note(&v2).is_some_and(|n| n.contains("(--mutate-budget-seconds)")),
        "{:?}",
        v2.notes
    );
    // Both: explorations stop at the earlier, and the note names its flag.
    let (now, hour) = (Instant::now(), Duration::from_secs(3600));
    for (total, nested, flag) in [
        (now + hour, now, "(--mutate-budget-seconds)"),
        (now, now + hour, "(--time-budget)"),
    ] {
        membudget::set_time_budget(Some(total));
        membudget::set_deadline(Some(nested));
        let left = membudget::time_budget_left().unwrap();
        let v3 = explore(m.clone(), 2_000_000);
        membudget::set_deadline(None);
        membudget::set_time_budget(None);
        assert!(
            (total == now) == left.is_zero() && left <= hour,
            "{flag}: {left:?} left"
        );
        assert!(!v3.complete, "{flag}: {:?}", v3.notes);
        assert!(
            deadline_note(&v3).is_some_and(|n| n.contains(flag)),
            "{flag}: {:?}",
            v3.notes
        );
    }
    // Cleared: the control's certificate.
    let o4 = certify(m);
    assert!(o4.verdict.is_some(), "after: {:?}", o4.reasons);
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Cause 2, one mechanism per instruction (`Machine::lane_orders_suffice`): a converged PTX
//! scalar atomic whose memory no other actor can reach before its last lane retires takes
//! its lanes' orders (`Machine::lane_orders`, one step) instead of AtomicComponent actions.
//! Each row compares the unreduced search and both explorers with the unreduced component
//! reference (`Machine::atomic_components_always`), and the named rows pin where the
//! components and the refusal stay.

use std::collections::BTreeSet;
use std::collections::HashSet;
use std::hash::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::explore::Action;
use ptxv_sim::explore::Verdict;
use ptxv_sim::machine::Machine;

const CAP: usize = 200_000;

/// DPOR's note when it hands a kernel to the stateful explorer
/// (`Machine::private_register_projection`).
const STATEFUL_NOTE: &str = "Stateful explorer selected";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

const HEAD: &str = ".version 8.8\n.target sm_103a\n.address_size 64\n";

/// A case from its manifest and PTX text with one entry per launch (the main one first).
fn build(manifest: serde_json::Value, text: &str) -> Machine {
    let case = Case {
        dir: PathBuf::new(),
        m: serde_json::from_value(manifest).unwrap(),
    };
    let launches = case.m.all_launches();
    let kernel = if launches.len() > 1 {
        ptxv_ir::Kernel::concat(
            launches
                .iter()
                .map(|l| ptxv_ptx::parse_kernel(text, Some(l.entry)).unwrap())
                .collect(),
        )
    } else {
        ptxv_ptx::parse_kernel(text, Some(&case.m.entry)).unwrap()
    };
    let launch = Arc::new(ptxv_sim::build_launch(&case, &kernel).unwrap());
    Machine::new(Arc::new(kernel), launch)
}

/// Entry `k` with buffers x (two words), out (64 words) and flag (one word), passed in that
/// order.
fn manifest(block: u32, ctas: u32, cluster: Option<u32>) -> serde_json::Value {
    let mut m = serde_json::json!({
        "name": "lane-order-components", "description": "cause 2 row", "entry": "k",
        "grid": [ctas, 1, 1], "block": [block, 1, 1], "outputs": ["x", "out"],
        "buffers": [
            {"name": "x", "dtype": "u32", "shape": [2], "init": "zeros"},
            {"name": "out", "dtype": "u32", "shape": [64], "init": "zeros"},
            {"name": "flag", "dtype": "u32", "shape": [1], "init": "zeros"}
        ],
        "params": [
            {"kind": "ptr", "buffer": "x"},
            {"kind": "ptr", "buffer": "out"},
            {"kind": "ptr", "buffer": "flag"}
        ],
        "variants": []
    });
    if let Some(c) = cluster {
        m["cluster"] = serde_json::json!([c, 1, 1]);
    }
    m
}

/// `prmt` is outside the instruction list of `FutureReads::analyze`: with it, DPOR runs on an
/// integer-add kernel itself instead of handing it to the stateful explorer.
const UNPROJECTED: &str = "prmt.b32 %r6, %r0, 0, 0;";

/// A one-CTA launch of `body` after `prologue`: `%rd0`, `%rd1`, `%rd4` hold the addresses of
/// x, out and flag, `%r0` = tid.x, `%rd3` = &out[tid.x], `s` is a shared word and `mb` a
/// shared mbarrier.
fn kernel(block: u32, prologue: &str, body: &str) -> Machine {
    let text = format!(
        r#"{HEAD}
.visible .entry k(.param .u64 x, .param .u64 out, .param .u64 flag) {{
 .reg .b32 %r<10>; .reg .b64 %rd<8>; .reg .pred %p<8>; .reg .f32 %f<8>;
 .shared .align 4 .b32 s; .shared .align 8 .b64 mb;
 ld.param.u64 %rd0, [x]; ld.param.u64 %rd1, [out]; ld.param.u64 %rd4, [flag];
 mov.u32 %r0, %tid.x; {prologue}
 mul.wide.u32 %rd2, %r0, 4; add.u64 %rd3, %rd1, %rd2;
{body}
}}
"#
    );
    build(manifest(block, 1, None), &text)
}

/// `kernel` with `UNPROJECTED`, so that DPOR runs itself on every row (`dpor`).
fn machine(block: u32, body: &str) -> Machine {
    kernel(block, UNPROJECTED, body)
}

/// What a search can tell apart: the findings other than notes (kind, signature), and the
/// terminal outputs.
#[derive(Debug, Default, PartialEq, Eq)]
struct Observed {
    findings: BTreeSet<(String, String)>,
    outputs: BTreeSet<Vec<u8>>,
}

fn observe(out: &mut Observed, machine: &Machine) {
    for f in &machine.findings {
        if !matches!(
            f.kind.as_str(),
            "note" | "probe" | "unordered" | "race-dead"
        ) {
            out.findings.insert((f.kind.clone(), f.sig.clone()));
        }
    }
}

/// An unreduced search: what it tells apart, and how many states have a split standing at an
/// atomic that offers AtomicComponent actions, or Split actions (its lanes' orders) instead.
#[derive(Debug, Default)]
struct Walk {
    observed: Observed,
    component_states: usize,
    lane_order_states: usize,
}

/// Every action of every state (states told apart by their hash); `always`: AtomicComponent
/// actions for every scalar atomic (the reference), otherwise lane orders where they suffice.
fn walk(initial: &Machine, always: bool) -> Walk {
    let mut initial = initial.clone();
    initial.atomic_components_always = always;
    let mut seen = HashSet::new();
    let mut stack = vec![initial];
    let mut out = Walk::default();
    while let Some(mut machine) = stack.pop() {
        observe(&mut out.observed, &machine);
        let mut hasher = DefaultHasher::new();
        machine.hash(&mut hasher);
        if !seen.insert(hasher.finish()) {
            continue;
        }
        assert!(seen.len() < CAP, "unreduced search hit the cap");
        let actions = machine.actions_bare();
        for (w, pc, _) in machine.splits() {
            if !matches!(machine.k.code[pc as usize].op, ptxv_ir::Op::Atom { .. }) {
                continue;
            }
            let offers = |component: bool| {
                actions.iter().any(|(a, _)| match a {
                    Action::AtomicComponent { warp, pc: at, .. } => {
                        component && *warp == w && *at == pc
                    }
                    Action::Split { warp, pc: at, .. } => !component && *warp == w && *at == pc,
                    _ => false,
                })
            };
            if offers(true) {
                out.component_states += 1;
            } else if offers(false) {
                out.lane_order_states += 1;
            }
        }
        if actions.is_empty() {
            if machine.terminal() {
                machine.exit_checks();
                observe(&mut out.observed, &machine);
                out.observed.outputs.insert(machine.outputs());
            } else {
                out.observed
                    .findings
                    .insert(("deadlock".into(), "raw reference".into()));
            }
        }
        for (action, _) in actions {
            let mut next = machine.clone();
            next.apply(&action);
            stack.push(next);
        }
    }
    out
}

/// The unreduced component reference.
fn raw(initial: &Machine) -> Observed {
    walk(initial, true).observed
}

fn reduced(v: &Verdict) -> Observed {
    assert!(v.complete, "reduced search hit the cap");
    let mut out = Observed {
        outputs: v.outputs.iter().cloned().collect(),
        ..Observed::default()
    };
    for f in &v.findings {
        if f.kind == "deadlock" {
            out.findings
                .insert(("deadlock".into(), "raw reference".into()));
        } else if !matches!(
            f.kind.as_str(),
            "note" | "probe" | "unordered" | "race-dead"
        ) {
            out.findings.insert((f.kind.clone(), f.sig.clone()));
        }
    }
    out
}

fn stateful(initial: &Machine) -> Verdict {
    ptxv_sim::explore::explore_with_workers(initial.clone(), CAP, 2)
}

/// DPOR's verdict, from DPOR itself: a row DPOR hands to the stateful explorer fails here.
fn dpor(initial: &Machine) -> Verdict {
    dpor_within(initial, CAP)
}

fn dpor_within(initial: &Machine, cap: usize) -> Verdict {
    let v = ptxv_sim::dpor::explore_dpor(initial.clone(), cap);
    assert!(
        !v.notes.iter().any(|n| n.contains(STATEFUL_NOTE)),
        "DPOR handed the row to the stateful explorer: {:?}",
        v.notes
    );
    v
}

/// The unreduced search (lane orders where they suffice) and both explorers against the
/// component reference.
fn matches_raw(name: &str, initial: &Machine) -> (Observed, Walk) {
    let reference = raw(initial);
    let lanes = walk(initial, false);
    let s = stateful(initial);
    let d = dpor(initial);
    eprintln!(
        "{name}: reference {} outputs {:?}; with lane orders: {} outputs, {} component / {} lane-order states; stateful {} states, DPOR {} states",
        reference.outputs.len(),
        reference.findings,
        lanes.observed.outputs.len(),
        lanes.component_states,
        lanes.lane_order_states,
        s.states,
        d.states
    );
    assert_eq!(lanes.observed, reference, "{name}: unreduced");
    assert_eq!(reduced(&s), reference, "{name}: stateful");
    assert_eq!(reduced(&d), reference, "{name}: DPOR");
    (reference, lanes)
}

/// An unobserved row: no state offers components, and every search reaches the reference.
fn unobserved(name: &str, initial: &Machine) -> Observed {
    let (reference, lanes) = matches_raw(name, initial);
    assert!(
        lanes.component_states == 0 && lanes.lane_order_states > 0,
        "{name}: {} component, {} lane-order states",
        lanes.component_states,
        lanes.lane_order_states
    );
    reference
}

/// A row with an outside observer: some state offers components, the unreduced search reaches
/// the reference, and each explorer finds with lane orders where they suffice what it finds
/// with components everywhere. (The stateful explorer's own gaps against the unreduced search
/// on rows that trap or race are there with components everywhere too: not cause 2's.)
fn observed(name: &str, initial: &Machine) -> Observed {
    let reference = raw(initial);
    let lanes = walk(initial, false);
    eprintln!(
        "{name}: reference {} outputs {:?}; with lane orders: {} outputs, {} component / {} lane-order states",
        reference.outputs.len(),
        reference.findings,
        lanes.observed.outputs.len(),
        lanes.component_states,
        lanes.lane_order_states
    );
    assert_eq!(lanes.observed, reference, "{name}: unreduced");
    assert!(
        lanes.component_states > 0,
        "{name}: no state offers components"
    );
    let mut always = initial.clone();
    always.atomic_components_always = true;
    for (explorer, lanes, components) in [
        ("stateful", stateful(initial), stateful(&always)),
        ("DPOR", dpor(initial), dpor(&always)),
    ] {
        assert_eq!(reduced(&lanes), reduced(&components), "{name}: {explorer}");
    }
    reference
}

/// Checks every row, then fails naming the rows that failed (so a mutant's log shows each row
/// that catches it); the rows' references, in order.
fn each_row<T>(rows: Vec<(&str, Machine)>, check: fn(&str, &Machine) -> T) -> Vec<T> {
    let mut failed = Vec::new();
    let mut out = Vec::new();
    for (name, initial) in &rows {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check(name, initial))) {
            Ok(reference) => out.push(reference),
            Err(_) => failed.push(*name),
        }
    }
    assert!(failed.is_empty(), "rows that fail: {failed:?}");
    out
}

/// Does the first state where a split stands at an atomic offer AtomicComponent actions?
/// Applies the first action until then (on `m` itself: a clone drops the certificate log).
fn components_at_issue(mut m: Machine) -> bool {
    for _ in 0..10_000 {
        let at_atomic = m
            .splits()
            .iter()
            .any(|(_, pc, _)| matches!(m.k.code[*pc as usize].op, ptxv_ir::Op::Atom { .. }));
        let actions = m.actions_bare();
        if at_atomic {
            return actions
                .iter()
                .any(|(a, _)| matches!(a, Action::AtomicComponent { .. }));
        }
        let Some((first, _)) = actions.into_iter().next() else {
            break;
        };
        m.apply(&first);
    }
    panic!("no split reached an atomic");
}

const STORE_OLD: &str = "st.global.u32 [%rd3], %r1;";
const TICKET: &str = "atom.global.add.u32 %r1, [%rd0], 1; st.global.u32 [%rd3], %r1;";
/// Thread 0 zeroes the shared word `s` (its address in `%r4`) before a CTA barrier.
const SHARED_INIT: &str = "mov.u32 %r4, s; mov.u32 %r5, 0; setp.eq.u32 %p2, %r0, 0; @%p2 st.shared.u32 [%r4], %r5; bar.sync 0;";
/// The reading thread (its read in `%r2`) traps on an intermediate count (1 or 2 of 3 adds).
const TRAP_MID: &str = "setp.eq.u32 %p5, %r2, 1; setp.eq.or.u32 %p5, %r2, 2, %p5; @%p5 trap;";
/// Thread 32 goes to OBS, threads 0-2 run on, the others to DONE.
const SPLIT_OBS: &str =
    "setp.eq.u32 %p0, %r0, 32; @%p0 bra OBS; setp.ge.u32 %p0, %r0, 3; @%p0 bra DONE;";
/// Waits for flag to be set (acquire).
const WAIT_FLAG: &str =
    "SPIN: ld.acquire.gpu.global.u32 %r5, [%rd4]; setp.eq.u32 %p3, %r5, 0; @%p3 bra SPIN;";
/// Sets flag (release).
const SET_FLAG: &str = "mov.u32 %r5, 1; st.release.gpu.global.u32 [%rd4], %r5;";

/// Unobserved single-warp rows: the lanes' orders replace the components, and the unreduced
/// search and both explorers reach exactly what every component interleaving reaches.
#[test]
fn unobserved_scalar_atomics_take_lane_orders_exactly() {
    let operand = "add.u32 %r3, %r0, 1;";
    // 2^24 from lane 0, 1.0 from the others: the sum's rounding depends on the order.
    let float = "setp.eq.u32 %p1, %r0, 0; selp.u32 %r3, 1266679808, 1065353216, %p1;";
    let rows: Vec<(&str, u32, String)> = vec![
        (
            "add_dead",
            3,
            format!("{operand} atom.global.add.u32 %r1, [%rd0], %r3; ret;"),
        ),
        ("add_tickets", 3, format!("{TICKET} ret;")),
        (
            "exch_dead",
            3,
            format!("{operand} atom.global.exch.b32 %r1, [%rd0], %r3; ret;"),
        ),
        (
            "exch_live",
            3,
            format!("{operand} atom.global.exch.b32 %r1, [%rd0], %r3; {STORE_OLD} ret;"),
        ),
        (
            "fadd_live",
            3,
            format!("{float} atom.global.add.f32 %r1, [%rd0], %r3; {STORE_OLD} ret;"),
        ),
        (
            "inc_live",
            3,
            format!("atom.global.inc.u32 %r1, [%rd0], 1; {STORE_OLD} ret;"),
        ),
        (
            "max_live",
            3,
            format!("{operand} atom.global.max.u32 %r1, [%rd0], %r3; {STORE_OLD} ret;"),
        ),
        (
            "shared_tickets",
            3,
            format!("{SHARED_INIT} atom.shared.add.u32 %r1, [%r4], 1; {STORE_OLD} ret;"),
        ),
        (
            "exch_dead_five",
            5,
            format!("{operand} atom.global.exch.b32 %r1, [%rd0], %r3; ret;"),
        ),
        ("add_four_live", 4, format!("{TICKET} ret;")),
    ];
    each_row(
        rows.into_iter()
            .map(|(name, block, body)| (name, machine(block, &body)))
            .collect(),
        unobserved,
    );
}

/// The integer-add rows as written (no `prmt`): DPOR hands them to the stateful explorer, whose
/// register projection runs with the lane orders and reaches the reference.
#[test]
fn projected_add_rows_take_lane_orders_in_the_stateful_explorer() {
    let rows: [(&str, u32, String); 4] = [
        (
            "add_dead",
            3,
            "add.u32 %r3, %r0, 1; atom.global.add.u32 %r1, [%rd0], %r3; ret;".into(),
        ),
        ("add_tickets", 3, format!("{TICKET} ret;")),
        ("add_four_live", 4, format!("{TICKET} ret;")),
        (
            "shared_tickets",
            3,
            format!("{SHARED_INIT} atom.shared.add.u32 %r1, [%r4], 1; {STORE_OLD} ret;"),
        ),
    ];
    for (name, block, body) in rows {
        let initial = kernel(block, "", &body);
        let reference = raw(&initial);
        let lanes = walk(&initial, false);
        assert_eq!(lanes.component_states, 0, "{name}");
        assert_eq!(lanes.observed, reference, "{name}: unreduced");
        let v = ptxv_sim::dpor::explore_dpor(initial.clone(), CAP);
        assert!(
            v.notes.iter().any(|n| n.contains(STATEFUL_NOTE)),
            "{name}: {:?}",
            v.notes
        );
        assert_eq!(reduced(&v), reference, "{name}: DPOR's stateful explorer");
        assert_eq!(reduced(&stateful(&initial)), reference, "{name}: stateful");
    }
}

/// Five to eight unobserved lanes whose results are read: MISS 74's lane orders would be
/// bounded there (lane order and its reverse), so the components stay, and the launch check is
/// complete with every order (five lanes: the reference's 120 outputs).
#[test]
fn bounded_lane_orders_up_to_eight_lanes_keep_the_components() {
    let five = machine(5, &format!("{TICKET} ret;"));
    let reference = raw(&five);
    assert_eq!(reference.outputs.len(), 120);
    let lanes = walk(&five, false);
    assert!(lanes.component_states > 0 && lanes.lane_order_states == 0);
    assert_eq!(lanes.observed, reference);
    for v in [
        ptxv_sim::smid::check_placements(&|| five.clone(), &mut |mk| {
            ptxv_sim::explore::explore_with_workers(mk(), CAP, 2)
        }),
        ptxv_sim::smid::check_placements(&|| five.clone(), &mut |mk| dpor(&mk())),
    ] {
        assert_eq!(reduced(&v), reference, "{:?}", v.notes);
    }
    for block in 6..=8 {
        assert!(
            components_at_issue(machine(block, &format!("{TICKET} ret;"))),
            "{block} lanes"
        );
    }
}

/// Beyond eight lanes whose results are read the components would pass the state cap: MISS
/// 74's bound stands (lane order and its reverse, with the note), and the launch check is
/// incomplete.
#[test]
fn bounded_lane_orders_beyond_eight_lanes_stand_with_the_note() {
    let initial = machine(9, &format!("{TICKET} ret;"));
    assert!(!components_at_issue(initial.clone()));
    for v in [
        ptxv_sim::smid::check_placements(&|| initial.clone(), &mut |mk| {
            ptxv_sim::explore::explore_with_workers(mk(), CAP, 2)
        }),
        ptxv_sim::smid::check_placements(&|| initial.clone(), &mut |mk| dpor(&mk())),
    ] {
        assert!(!v.complete, "{:?}", v.notes);
        assert!(
            v.notes.iter().any(|n| n.contains("lane orders of")),
            "{:?}",
            v.notes
        );
        assert_eq!(v.classes(), ["nondeterministic"]);
        assert_eq!(v.outputs.len(), 2);
    }
}

/// Where a lane faults or reads an undefined operand: a misaligned lane faults the components'
/// issue before any lane runs; a lane whose address names no memory faults its own component
/// while the others run theirs (the result read, or dead); an operand nothing wrote carries
/// S-INIT's undefinedness into the word and the returned values. Each keeps the components,
/// and every search reaches the reference.
#[test]
fn issue_faults_and_undefined_operands_keep_the_components() {
    each_row(
        vec![
            (
                "misaligned_lane",
                machine(
                    3,
                    &format!(
                        "setp.eq.u32 %p1, %r0, 2; selp.u32 %r6, 2, 0, %p1; cvt.u64.u32 %rd5, %r6; add.u64 %rd6, %rd0, %rd5;
 atom.global.add.u32 %r1, [%rd6], 1; {STORE_OLD} ret;"
                    ),
                ),
            ),
            (
                "unmapped_lane",
                machine(
                    3,
                    &format!(
                        "setp.eq.u32 %p1, %r0, 2; selp.u32 %r6, 4096, 0, %p1; cvt.u64.u32 %rd5, %r6; add.u64 %rd6, %rd0, %rd5;
 atom.global.add.u32 %r1, [%rd6], 1; {STORE_OLD} ret;"
                    ),
                ),
            ),
            (
                "unmapped_lane_dead",
                machine(
                    3,
                    "setp.eq.u32 %p1, %r0, 2; selp.u32 %r6, 4096, 0, %p1; cvt.u64.u32 %rd5, %r6; add.u64 %rd6, %rd0, %rd5;
 atom.global.add.u32 %r1, [%rd6], 1; ret;",
                ),
            ),
            (
                "undefined_operand",
                machine(
                    3,
                    &format!("atom.global.add.u32 %r1, [%rd0], %r7; {STORE_OLD} ret;"),
                ),
            ),
        ],
        observed,
    );
}

/// Controls (the lane-cut shapes): lanes 0-3 of warp 0 add 1 under a guard, results dead;
/// thread 32 does not read (noread), reads only past a bar.sync that warp 0 reaches after the
/// add (bar), or adds 1 with a dead red (red), then traps on a partial count. No thread can
/// tell the lanes apart: every search reaches the reference, clean.
#[test]
fn lane_cut_controls_match_the_reference() {
    let shape = |read: &str, sync: &str| {
        machine(
            64,
            &format!(
                "setp.ge.u32 %p1, %r0, 32; @%p1 bra READER;
 setp.lt.u32 %p2, %r0, 4; @%p2 atom.relaxed.gpu.global.add.u32 %r3, [%rd0], 1; {sync} bra DONE;
READER: {sync} setp.ne.u32 %p3, %r0, 32; @%p3 bra DONE;
 {read} setp.ne.u32 %p4, %r4, 0; setp.ne.and.u32 %p4, %r4, 4, %p4; @%p4 trap;
 mov.u32 %r5, 7; st.global.u32 [%rd3], %r5;
DONE: ret;"
            ),
        )
    };
    let references = each_row(
        vec![
            ("noread", shape("mov.u32 %r4, 0;", "")),
            (
                "bar",
                shape("ld.relaxed.gpu.global.u32 %r4, [%rd0];", "bar.sync 0;"),
            ),
            (
                "red",
                shape(
                    "red.relaxed.gpu.global.add.u32 [%rd0], 1; mov.u32 %r4, 0;",
                    "",
                ),
            ),
        ],
        control,
    );
    for reference in references {
        assert!(reference.findings.is_empty(), "{:?}", reference.findings);
        assert_eq!(reference.outputs.len(), 1);
    }
}

/// A control row: every search reaches the reference, whichever mechanism each instance takes.
fn control(name: &str, initial: &Machine) -> Observed {
    matches_raw(name, initial).0
}

/// A 32-lane exchange with distinct values (fixtures/e_lane_order EXCH's shape): one order per
/// value, within a small budget (the component search exceeded 2M states).
#[test]
fn unobserved_warp_exchange_stays_within_budget() {
    let initial = machine(
        32,
        "add.u32 %r3, %r0, 1; atom.global.exch.b32 %r1, [%rd0], %r3; ret;",
    );
    for v in [
        ptxv_sim::explore::explore_with_workers(initial.clone(), 5_000, 2),
        dpor_within(&initial, 5_000),
    ] {
        assert!(v.complete, "{} states", v.states);
        assert_eq!(v.outputs.len(), 32);
        eprintln!("warp exchange: {} states", v.states);
    }
}

/// MISS 169 (partial count) and MISS 176 (one lane's release): a thread outside the warp may
/// read the word between the lanes, so the components stay and both explorers match the
/// reference (trap, race).
#[test]
fn observed_atomics_keep_their_components() {
    let partial = machine(
        64,
        "setp.eq.u32 %p0, %r0, 32; @%p0 bra READ; setp.ge.u32 %p0, %r0, 3; @%p0 bra DONE;
 atom.relaxed.gpu.global.add.u32 %r1, [%rd0], 1; bra DONE;
READ: ld.relaxed.gpu.global.u32 %r2, [%rd0]; setp.ne.u32 %p1, %r2, 0; setp.ne.and.u32 %p1, %r2, 3, %p1; @%p1 trap;
DONE: ret;",
    );
    let (reference, lanes) = matches_raw("miss169_partial", &partial);
    assert!(lanes.component_states > 0);
    assert!(reference.findings.iter().any(|(k, _)| k == "trap"));
    let one_lane = machine(
        64,
        "setp.lt.u32 %p1, %r0, 2; @%p1 bra LANES; setp.ne.u32 %p2, %r0, 32; @%p2 bra DONE;
SPIN: ld.relaxed.gpu.global.u32 %r4, [%rd4]; setp.eq.u32 %p3, %r4, 0; @%p3 bra SPIN;
 fence.acq_rel.gpu; ld.global.u32 %r5, [%rd1+200]; st.global.u32 [%rd1+252], %r5; bra DONE;
LANES: add.u32 %r2, %r0, 100; st.global.u32 [%rd3+200], %r2;
 fence.acq_rel.gpu; atom.global.add.u32 %r3, [%rd4], 1;
DONE: ret;",
    );
    let (reference, lanes) = matches_raw("miss176_one_lane", &one_lane);
    assert!(lanes.component_states > 0);
    assert!(reference.findings.iter().any(|(k, _)| k == "race"));
}

/// Outside observers of every kind (the review's rows): a weak read, stored or trapping; a
/// strong read; a diverged lane of the members' warp; a read past a named barrier the members
/// never use; a cp.async writer; a trap with no access; a register-only path to a trap; an
/// SC fence over shared memory beside a global atomic; a weak store before the atomic.
#[test]
fn outside_observers_keep_the_components() {
    let at = |obs: &str| format!("{SPLIT_OBS} {TICKET} bra DONE;\nOBS: {obs}\nDONE: ret;");
    let rows: Vec<(&str, u32, String)> = vec![
        (
            "weak_store",
            64,
            at("ld.global.u32 %r2, [%rd0]; st.global.u32 [%rd1+252], %r2;"),
        ),
        (
            "weak_trap",
            64,
            at(&format!("ld.global.u32 %r2, [%rd0]; {TRAP_MID}")),
        ),
        (
            "strong_trap",
            64,
            at(&format!("ld.relaxed.gpu.global.u32 %r2, [%rd0]; {TRAP_MID}")),
        ),
        (
            "lane_trap",
            4,
            format!(
                "setp.eq.u32 %p0, %r0, 3; @%p0 bra OBS; {TICKET} bra DONE;
OBS: ld.relaxed.gpu.global.u32 %r2, [%rd0]; {TRAP_MID}
DONE: ret;"
            ),
        ),
        (
            "bar_trap",
            96,
            format!(
                "setp.ge.u32 %p0, %r0, 32; @%p0 bra OBS; setp.ge.u32 %p0, %r0, 3; @%p0 bra DONE;
 {TICKET} bra DONE;
OBS: bar.sync 1, 64; setp.ne.u32 %p1, %r0, 32; @%p1 bra DONE;
 ld.relaxed.gpu.global.u32 %r2, [%rd0]; {TRAP_MID}
DONE: ret;"
            ),
        ),
        (
            "cpasync",
            64,
            format!(
                "{SHARED_INIT} {SPLIT_OBS}
 atom.shared.add.u32 %r1, [%r4], 1; {STORE_OLD} bra DONE;
OBS: cp.async.ca.shared.global [%r4], [%rd0+4], 4; cp.async.commit_group; cp.async.wait_group 0;
DONE: ret;"
            ),
        ),
        ("outsider_trap", 64, at("trap;")),
        (
            "outsider_trap_late",
            64,
            at("add.u32 %r5, %r0, 1; mul.lo.u32 %r5, %r5, 3; setp.ne.u32 %p6, %r5, 0; @%p6 trap;"),
        ),
        (
            "sc_fence",
            64,
            "mov.u32 %r4, s; mov.u32 %r7, 0; setp.eq.u32 %p0, %r0, 0; @%p0 st.shared.u32 [%r4], %r7; bar.sync 0;
 setp.eq.u32 %p0, %r0, 32; @%p0 bra OBS; setp.ge.u32 %p0, %r0, 2; @%p0 bra DONE;
 atom.global.add.u32 %r1, [%rd0], 1; fence.sc.gpu; ld.relaxed.cta.shared.u32 %r2, [%r4]; st.global.u32 [%rd3], %r2; bra DONE;
OBS: mov.u32 %r5, 1; st.relaxed.cta.shared.u32 [%r4], %r5; fence.sc.gpu; ld.relaxed.cta.shared.u32 %r8, [%r4];
DONE: ret;"
                .into(),
        ),
        (
            "past_weak_store",
            64,
            at("mov.u32 %r5, 5; st.global.u32 [%rd0], %r5;"),
        ),
    ];
    each_row(
        rows.into_iter()
            .map(|(name, block, body)| (name, machine(block, &body)))
            .collect(),
        observed,
    );
}

/// Outside observers in other CTAs and grids: a read past a branch on %ctaid; a DSMEM read
/// from the other CTA of a cluster; a later grid in an unknown launch order; a PDL grid the
/// members' grid lets start before its atomic.
#[test]
fn observers_in_other_ctas_and_grids_keep_the_components() {
    let ctaid = format!(
        r#"{HEAD}
.visible .entry k(.param .u64 x, .param .u64 out, .param .u64 flag) {{
 .reg .b32 %r<10>; .reg .b64 %rd<8>; .reg .pred %p<8>;
 ld.param.u64 %rd0, [x]; ld.param.u64 %rd1, [out]; ld.param.u64 %rd4, [flag];
 mov.u32 %r0, %tid.x; mov.u32 %r6, %ctaid.x;
 mul.wide.u32 %rd2, %r0, 4; add.u64 %rd3, %rd1, %rd2;
 setp.eq.u32 %p0, %r6, 1; @%p0 bra C1;
 setp.ge.u32 %p0, %r0, 3; @%p0 bra DONE;
 {TICKET} bra DONE;
C1: setp.ne.u32 %p1, %r0, 0; @%p1 bra DONE;
 ld.relaxed.gpu.global.u32 %r2, [%rd0]; {TRAP_MID}
DONE: ret;
}}
"#
    );
    let dsmem = format!(
        r#"{HEAD}
.visible .entry k(.param .u64 x, .param .u64 out, .param .u64 flag)
.reqnctapercluster 2, 1, 1
{{
 .reg .b32 %r<10>; .reg .b64 %rd<8>; .reg .pred %p<8>;
 .shared .align 4 .b32 s;
 ld.param.u64 %rd0, [x]; ld.param.u64 %rd1, [out]; ld.param.u64 %rd4, [flag];
 mov.u32 %r0, %tid.x; mov.u32 %r6, %cluster_ctarank; mov.u32 %r4, s;
 mul.wide.u32 %rd2, %r0, 4; add.u64 %rd3, %rd1, %rd2;
 mov.u32 %r7, 0; setp.eq.u32 %p0, %r0, 0; @%p0 st.shared.u32 [%r4], %r7;
 barrier.cluster.arrive.release; barrier.cluster.wait.acquire;
 setp.eq.u32 %p0, %r6, 1; @%p0 bra C1;
 setp.ge.u32 %p0, %r0, 3; @%p0 bra FIN;
 atom.shared.add.u32 %r1, [%r4], 1; st.global.u32 [%rd3], %r1; bra FIN;
C1: setp.ne.u32 %p1, %r0, 0; @%p1 bra FIN;
 mapa.shared::cluster.u32 %r5, %r4, 0; ld.relaxed.cluster.shared::cluster.u32 %r2, [%r5];
 {TRAP_MID}
FIN: barrier.cluster.arrive.release; barrier.cluster.wait.acquire;
 ret;
}}
"#
    );
    each_row(
        vec![
            ("ctaid_trap", build(manifest(32, 2, None), &ctaid)),
            ("dsmem_trap", build(manifest(32, 2, Some(2)), &dsmem)),
            ("grid_unknown_trap", two_grids("unknown", false)),
            ("grid_pdl_trap", two_grids("declared", true)),
        ],
        observed,
    );
}

/// Two launches: k (lanes 0-2 take tickets) and k2 (one thread reads x[0] and traps on an
/// intermediate count). `order`: the case's launch order; `pdl`: k2 is a PDL launch and k lets
/// it start before its atomic.
fn two_grids(order: &str, pdl: bool) -> Machine {
    let trigger = if pdl {
        "griddepcontrol.launch_dependents;"
    } else {
        ""
    };
    let text = format!(
        r#"{HEAD}
.visible .entry k(.param .u64 x, .param .u64 out, .param .u64 flag) {{
 .reg .b32 %r<10>; .reg .b64 %rd<8>; .reg .pred %p<8>;
 ld.param.u64 %rd0, [x]; ld.param.u64 %rd1, [out]; ld.param.u64 %rd4, [flag];
 mov.u32 %r0, %tid.x;
 mul.wide.u32 %rd2, %r0, 4; add.u64 %rd3, %rd1, %rd2;
 {trigger}
 {TICKET}
 ret;
}}
.visible .entry k2(.param .u64 x) {{
 .reg .b32 %r<10>; .reg .b64 %rd<8>; .reg .pred %p<8>;
 ld.param.u64 %rd0, [x];
 ld.relaxed.gpu.global.u32 %r2, [%rd0]; {TRAP_MID}
 ret;
}}
"#
    );
    let mut m = manifest(3, 1, None);
    m["launch_order"] = serde_json::json!(order);
    let mut second = serde_json::json!({
        "entry": "k2", "grid": [1, 1, 1], "block": [1, 1, 1],
        "params": [{"kind": "ptr", "buffer": "x"}]
    });
    if pdl {
        second["pdl"] = serde_json::json!(true);
    }
    m["launches"] = serde_json::json!([second]);
    build(m, &text)
}

/// Outsiders whose remaining code touches no memory, yet whose steps do not commute with the
/// lanes': a trap (it ends the grid with the lanes that have run), a thread whose unwaited
/// cp.async lands at its exit (E-CPASYNC-EXIT), and a cp.async landing through an mbarrier
/// arrive (a pending asynchronous operation after its issuer exits). Each outsider sets flag,
/// which the members wait for, so the members issue only once it stands at its last steps; 33
/// threads, so that no other lane of the outsider's warp still stands before the branches (it
/// would count, and give the components on its own). Without the components the outputs
/// differ: the trap's partial counts, the copies' resets between the lanes.
#[test]
fn outsiders_that_end_the_grid_or_land_copies_keep_the_components() {
    // Lane 0 adds 1, lane 1 adds -1: x is 0 before and after the atomic, so only a trap between
    // the lanes makes the outputs differ (nondeterministic, beside the trap).
    let trap = machine(
        33,
        &format!(
            "setp.eq.u32 %p0, %r0, 32; @%p0 bra OBS; setp.ge.u32 %p0, %r0, 2; @%p0 bra DONE;
 {WAIT_FLAG}
 setp.eq.u32 %p1, %r0, 0; selp.u32 %r3, 1, 4294967295, %p1; atom.global.add.u32 %r1, [%rd0], %r3; bra DONE;
OBS: {SET_FLAG} trap;
DONE: ret;"
        ),
    );
    let copy_at_exit = machine(
        33,
        &format!(
            "{SHARED_INIT} {SPLIT_OBS}
 {WAIT_FLAG} atom.shared.add.u32 %r1, [%r4], 1; {STORE_OLD} bra DONE;
OBS: cp.async.ca.shared.global [%r4], [%rd0+4], 4; {SET_FLAG}
DONE: ret;"
        ),
    );
    let copy_by_arrive = machine(
        33,
        &format!(
            "mov.u32 %r7, mb; setp.eq.u32 %p4, %r0, 0; @%p4 mbarrier.init.shared.b64 [%r7], 1;
 {SHARED_INIT} {SPLIT_OBS}
 {WAIT_FLAG} atom.shared.add.u32 %r1, [%r4], 1; {STORE_OLD} bra DONE;
OBS: cp.async.ca.shared.global [%r4], [%rd0+4], 4; cp.async.mbarrier.arrive.noinc.shared.b64 [%r7]; {SET_FLAG}
DONE: ret;"
        ),
    );
    let references = each_row(
        vec![
            ("trap_after_release", trap),
            ("copy_lands_at_exit", copy_at_exit),
            ("copy_lands_by_arrive", copy_by_arrive),
        ],
        observed,
    );
    assert_eq!(references[0].outputs.len(), 3);
    assert!(references[0].findings.iter().any(|(k, _)| k == "trap"));
}

/// What a check path reports, for comparing the mechanisms path by path.
#[derive(Debug, PartialEq, Eq)]
struct PathVerdict {
    complete: bool,
    outputs: BTreeSet<Vec<u8>>,
    findings: BTreeSet<(String, String)>,
}

fn path_verdict(v: &Verdict) -> PathVerdict {
    PathVerdict {
        complete: v.complete,
        outputs: v.outputs.iter().cloned().collect(),
        findings: v
            .findings
            .iter()
            .filter(|f| {
                !matches!(
                    f.kind.as_str(),
                    "note" | "probe" | "unordered" | "race-dead"
                )
            })
            .map(|f| (f.kind.clone(), f.sig.clone()))
            .collect(),
    }
}

/// The stateful explorer, DPOR and the launch check (`smid::check_placements`), each with lane
/// orders where they suffice and with components everywhere: the same completeness, outputs
/// and findings on every path. Returns the stateful explorer's verdict with lane orders.
fn same_on_every_path(name: &str, initial: &Machine) -> PathVerdict {
    same_on_every_path_within(name, initial, CAP)
}

fn same_on_every_path_within(name: &str, initial: &Machine, cap: usize) -> PathVerdict {
    let mut always = initial.clone();
    always.atomic_components_always = true;
    let run = |m: &Machine, path: &str| match path {
        "stateful" => ptxv_sim::explore::explore_with_workers(m.clone(), cap, 2),
        "DPOR" => dpor_within(m, cap),
        _ => ptxv_sim::smid::check_placements(&|| m.clone(), &mut |mk| {
            ptxv_sim::explore::explore_with_workers(mk(), cap, 2)
        }),
    };
    let mut first = None;
    for path in ["stateful", "DPOR", "launch check"] {
        let (lanes, components) = (run(initial, path), run(&always, path));
        eprintln!(
            "{name} {path}: lane orders {} states, components {} states",
            lanes.states, components.states
        );
        let lanes = path_verdict(&lanes);
        assert_eq!(lanes, path_verdict(&components), "{name}: {path}");
        first.get_or_insert(lanes);
    }
    first.unwrap()
}

/// The review's shapes, each on every check path with lane orders where they suffice and with
/// components everywhere: an outsider that sets the flag the members wait for and traps
/// (tickets; and lane 0 adding 1, lane 1 adding -1, where only a trap between them shows); a
/// misaligned lane; a lane whose address names no memory (results read, or dead); an operand
/// nothing wrote; an unwaited cp.async landing at its issuer's exit while the lanes take
/// tickets on the shared word it writes; a shared::cluster atomic on the other CTA of a
/// cluster, which exits when it likes, or once it has set the flag the members wait for (its
/// exit between the lanes faults the lanes after it, and leaves no output apart); five and six
/// lanes taking tickets (complete with components).
#[test]
fn review_rows_match_the_components_on_every_path() {
    let wait_trap = |atom: &str, last_lane: u32| {
        machine(
            33,
            &format!(
                "setp.eq.u32 %p0, %r0, 32; @%p0 bra OBS; setp.ge.u32 %p0, %r0, {last_lane}; @%p0 bra DONE;
SPIN: ld.relaxed.gpu.global.u32 %r5, [%rd4]; setp.eq.u32 %p1, %r5, 0; @%p1 bra SPIN;
 {atom} bra DONE;
OBS: mov.u32 %r6, 1; st.relaxed.gpu.global.u32 [%rd4], %r6; trap;
DONE: ret;"
            ),
        )
    };
    let lane2_at = |offset: u32, after: &str| {
        machine(
            3,
            &format!(
                "setp.eq.u32 %p1, %r0, 2; selp.u32 %r6, {offset}, 0, %p1; cvt.u64.u32 %rd5, %r6; add.u64 %rd6, %rd0, %rd5;
 atom.global.add.u32 %r1, [%rd6], 1; {after} ret;"
            ),
        )
    };
    let cluster_atomic = |wait: &str, set: &str| {
        build(
            manifest(32, 2, Some(2)),
            &format!(
                r#"{HEAD}
.visible .entry k(.param .u64 x, .param .u64 out, .param .u64 flag)
.reqnctapercluster 2, 1, 1
{{
 .reg .b32 %r<10>; .reg .b64 %rd<8>; .reg .pred %p<8>;
 .shared .align 4 .b32 s;
 ld.param.u64 %rd0, [x]; ld.param.u64 %rd1, [out]; ld.param.u64 %rd4, [flag];
 mov.u32 %r0, %tid.x; mov.u32 %r6, %cluster_ctarank; mov.u32 %r4, s;
 mul.wide.u32 %rd2, %r0, 4; add.u64 %rd3, %rd1, %rd2;
 mov.u32 %r7, 0; setp.eq.u32 %p0, %r0, 0; @%p0 st.shared.u32 [%r4], %r7;
 barrier.cluster.arrive.release; barrier.cluster.wait.acquire;
 setp.eq.u32 %p0, %r6, 1; @%p0 bra C1;
 setp.ge.u32 %p0, %r0, 3; @%p0 bra FIN;
 {wait}
 mapa.shared::cluster.u32 %r8, %r4, 1; atom.shared::cluster.add.u32 %r1, [%r8], 1; {STORE_OLD} bra FIN;
C1: setp.ne.u32 %p1, %r0, 0; @%p1 bra FIN; {set}
FIN: ret;
}}
"#
            ),
        )
    };
    each_row(
        vec![
            ("trap_outside_the_members", wait_trap(TICKET, 3)),
            (
                "trap_between_two_lanes",
                wait_trap(
                    "setp.eq.u32 %p2, %r0, 0; selp.u32 %r3, 1, 4294967295, %p2; atom.global.add.u32 %r1, [%rd0], %r3;",
                    2,
                ),
            ),
            ("misaligned_member_lane", lane2_at(2, STORE_OLD)),
            ("unmapped_member_lane", lane2_at(4096, STORE_OLD)),
            ("unmapped_member_lane_dead", lane2_at(4096, "")),
            (
                "undefined_operand",
                machine(
                    3,
                    &format!("atom.global.add.u32 %r1, [%rd0], %r7; {STORE_OLD} ret;"),
                ),
            ),
            (
                "cp_async_landing_at_exit",
                machine(
                    33,
                    &format!(
                        "{SHARED_INIT} {SPLIT_OBS}
 atom.shared.add.u32 %r1, [%r4], 1; {STORE_OLD} bra DONE;
OBS: cp.async.ca.shared.global [%r4], [%rd0+4], 4; cp.async.commit_group;
DONE: ret;"
                    ),
                ),
            ),
            ("remote_cta_exit", cluster_atomic("", "")),
            (
                "remote_cta_exit_after_flag",
                cluster_atomic(WAIT_FLAG, SET_FLAG),
            ),
            ("tickets_five_lanes", tickets(5)),
            ("tickets_six_lanes", tickets(6)),
        ],
        same_on_every_path,
    );
}

/// Lanes 0 to `lanes - 1` of one warp take tickets, unobserved.
fn tickets(lanes: u32) -> Machine {
    machine(
        32,
        &format!("setp.ge.u32 %p1, %r0, {lanes}; @%p1 bra DONE; {TICKET}\nDONE: ret;"),
    )
}

/// Eight lanes taking tickets keep the components: every path is complete with every order
/// (8! outputs), the same as with components everywhere. Ten lanes: the launch check is
/// incomplete either way, with lane order and its reverse and the note, or with the
/// components at the state cap.
#[test]
#[ignore = "heavy: eight lanes' components on three paths, twice; 41 s"]
fn tickets_of_eight_lanes_are_complete_and_of_ten_bounded() {
    let eight = same_on_every_path_within("tickets_eight_lanes", &tickets(8), 2_000_000);
    assert!(eight.complete);
    assert_eq!(eight.outputs.len(), 40_320);
    let ten = tickets(10);
    let mut always = ten.clone();
    always.atomic_components_always = true;
    let launch_check = |m: &Machine| {
        ptxv_sim::smid::check_placements(&|| m.clone(), &mut |mk| {
            ptxv_sim::explore::explore_with_workers(mk(), CAP, 2)
        })
    };
    let lanes = launch_check(&ten);
    assert!(!lanes.complete);
    assert!(
        lanes.notes.iter().any(|n| n.contains("lane orders of")),
        "{:?}",
        lanes.notes
    );
    assert_eq!(lanes.outputs.len(), 2);
    assert!(!launch_check(&always).complete);
}

/// The lane orders replace the components only in a plain search: with the certificate log on
/// (as for the other modes that give components a meaning of their own), they stay.
#[test]
fn non_plain_searches_keep_the_components() {
    let initial = machine(3, &format!("{TICKET} ret;"));
    assert!(!components_at_issue(initial.clone()));
    let mut logged = initial.clone();
    logged.wlog =
        ptxv_sim::wellsync::LogSlot(Some(Box::new(ptxv_sim::wellsync::SyncLog::new(&logged))));
    assert!(components_at_issue(logged));
}

/// The choice is per instance: an outsider warp that may still reach the word gives
/// components, one that has left gives lane orders; the union matches the reference.
#[test]
fn per_instance_choice_matches_the_reference() {
    let initial = machine(
        64,
        "setp.ge.u32 %p0, %r0, 32; @%p0 bra OUT; setp.ge.u32 %p0, %r0, 3; @%p0 bra DONE;
 atom.global.add.u32 %r1, [%rd0], 1; st.global.u32 [%rd3], %r1; bra DONE;
OUT: setp.ne.u32 %p1, %r0, 32; @%p1 bra DONE; ld.relaxed.gpu.global.u32 %r2, [%rd0]; st.global.u32 [%rd3], %r2;
DONE: ret;",
    );
    let (_, lanes) = matches_raw("per_instance", &initial);
    assert!(lanes.component_states > 0 && lanes.lane_order_states > 0);
}

/// Element-wise atomicity: the lanes of a vector or packed atomic may interleave element by
/// element, so FloatWords and PackedBf16 keep their components even unobserved (lane 0 sees
/// one element before lane 1's add and the other after it).
#[test]
fn vector_and_packed_atomics_keep_their_components() {
    let rows = [
        // Lane 0 adds (1.0, 1.0), lane 1 (2.0, 2.0) to two f32 words.
        (
            "v2f32_two_lanes",
            "cvt.rn.f32.u32 %f1, %r0; add.f32 %f4, %f1, 0f3F800000; mov.f32 %f5, %f4;
 atom.global.add.v2.f32 {%f2, %f3}, [%rd0], {%f4, %f5};
 mov.b32 %r1, %f2; st.global.u32 [%rd3], %r1; mov.b32 %r2, %f3; st.global.u32 [%rd3+8], %r2; ret;",
        ),
        // Lane 0 adds (1.0, 2.0), lane 1 (4.0, 8.0) to the halves of one bf16x2 word.
        (
            "bf16x2_two_lanes",
            "setp.eq.u32 %p1, %r0, 0; selp.u32 %r3, 1073758080, 1090535552, %p1;
 atom.global.add.noftz.bf16x2 %r1, [%rd0], %r3; st.global.u32 [%rd3], %r1; ret;",
        ),
    ];
    each_row(
        rows.into_iter()
            .map(|(name, body)| (name, machine(2, body)))
            .collect(),
        element_wise,
    );
}

/// A vector or packed row: components and no lane order anywhere, and every search reaches the
/// reference's four outputs (each element's order on its own).
fn element_wise(name: &str, initial: &Machine) -> Observed {
    let (reference, lanes) = matches_raw(name, initial);
    assert!(
        lanes.component_states > 0 && lanes.lane_order_states == 0,
        "{name}"
    );
    assert_eq!(reference.outputs.len(), 4, "{name}");
    reference
}

/// Review point 1: a member's own later strong access of the word, with no barrier between, is
/// reported by E-WARP-ORDER (warp-order:), on both explorers, whichever mechanism runs the
/// atomic.
#[test]
fn member_later_access_is_a_warp_order_pair() {
    let rows = [
        (
            "add_layout",
            "atom.global.add.u32 %r1, [%rd0], 1;",
            vec!["nondeterministic"],
        ),
        (
            "redf32_no_layout",
            "red.global.add.f32 [%rd0], 0f3F800000;",
            vec!["nondeterministic", "unmodeled"],
        ),
        (
            "cas_no_layout",
            "add.u32 %r3, %r0, 1; atom.global.cas.b32 %r1, [%rd0], 0, %r3;",
            vec!["nondeterministic", "unmodeled"],
        ),
    ];
    for (name, atom, classes) in rows {
        let initial = machine(
            3,
            &format!("{atom} ld.relaxed.gpu.global.u32 %r1, [%rd0]; {STORE_OLD} ret;"),
        );
        for v in [stateful(&initial), dpor(&initial)] {
            assert!(v.complete, "{name}");
            assert_eq!(v.classes(), classes, "{name}: {:?}", v.findings);
            assert!(
                v.findings.iter().any(|f| f.sig.starts_with("warp-order:")),
                "{name}: {:?}",
                v.findings
            );
        }
    }
}

/// Review point 2: a returnless scalar f32 red whose lanes have no lane orders (same operand,
/// or exact adds) while thread 32 may observe the word: the certificate refuses (C-ORDER per
/// lane, or C-POLL), and both explorers keep the refusal.
#[test]
fn red_with_an_outside_observer_stays_refused() {
    let same = "mov.u32 %r3, 1065353216;";
    let exact = "setp.eq.u32 %p1, %r0, 1; selp.u32 %r3, 1073741824, 1065353216, %p1; setp.eq.u32 %p2, %r0, 2; selp.u32 %r3, 1082130432, %r3, %p2;";
    let lane0_flag =
        "mov.u32 %r5, 1; setp.eq.u32 %p4, %r0, 0; @%p4 st.release.gpu.global.u32 [%rd4], %r5;";
    let read_trap = |bits: u32| {
        format!("ld.relaxed.gpu.global.u32 %r2, [%rd0]; setp.eq.u32 %p5, %r2, {bits}; @%p5 trap;")
    };
    let sync1 = |bits: u32| {
        format!(
            "SPIN: ld.acquire.gpu.global.u32 %r4, [%rd4]; setp.eq.u32 %p3, %r4, 0; @%p3 bra SPIN; {}",
            read_trap(bits)
        )
    };
    let rows: Vec<(&str, &str, &str, String)> = vec![
        ("same_read", same, "", read_trap(1065353216)),
        ("same_sync1", same, lane0_flag, sync1(1065353216)),
        (
            "exact_poll",
            exact,
            "",
            "POLL: ld.relaxed.gpu.global.u32 %r2, [%rd0]; setp.eq.u32 %p3, %r2, 0; @%p3 bra POLL; setp.eq.u32 %p5, %r2, 1073741824; @%p5 trap;".into(),
        ),
        ("exact_read", exact, "", read_trap(1073741824)),
        (
            "exact_fetch",
            exact,
            "",
            "atom.global.add.f32 %r2, [%rd0], 0f00000000; setp.eq.u32 %p5, %r2, 1073741824; @%p5 trap;".into(),
        ),
        ("exact_sync1", exact, lane0_flag, sync1(1065353216)),
    ];
    for (name, operand, after, observer) in rows {
        let initial = machine(
            64,
            &format!(
                "setp.eq.u32 %p0, %r0, 32; @%p0 bra OBS; setp.ge.u32 %p0, %r0, 3; @%p0 bra DONE;
 {operand} red.global.add.f32 [%rd0], %r3; {after} bra DONE;
OBS: {observer}
DONE: ret;"
            ),
        );
        let c = ptxv_sim::wellsync::certify(initial.clone());
        assert!(c.verdict.is_none(), "{name}: certified");
        assert!(
            c.reasons
                .iter()
                .any(|r| r.starts_with("C-ORDER") || r.starts_with("C-POLL")),
            "{name}: {:?}",
            c.reasons
        );
        for v in [
            stateful(&initial),
            dpor(&initial),
            ptxv_sim::wellsync::check_certified(initial.clone(), CAP),
        ] {
            assert!(v.complete, "{name}");
            assert_eq!(v.classes(), ["unmodeled"], "{name}: {:?}", v.findings);
            assert!(
                v.findings
                    .iter()
                    .any(|f| f.sig.starts_with("atomic-lane-order:")),
                "{name}"
            );
        }
    }
}

/// e_mm_lane UNIRED: `explore`, DPOR, the launch check and the certified launch check agree.
#[test]
fn unicast_red_paths_agree() {
    let case = Case::load(root().join("fixtures/e_mm_lane")).unwrap();
    let ptx = case.variant_ptx("UNIRED").unwrap();
    let kernel = ptxv_ptx::parse_kernel_for(
        &ptx,
        Some(&case.m.entry),
        case.m
            .ptxas_toolkit
            .as_deref()
            .and_then(ptxv_ptx::toolkit_version),
    )
    .unwrap();
    let vcase = case.for_variant("UNIRED").unwrap();
    let launch = ptxv_sim::build_launch(&vcase, &kernel).unwrap();
    let initial = Machine::new(Arc::new(kernel), Arc::new(launch));
    let paths = [
        stateful(&initial),
        dpor(&initial),
        ptxv_sim::smid::check_placements(&|| initial.clone(), &mut |mk| {
            ptxv_sim::explore::explore_with_workers(mk(), CAP, 2)
        }),
        ptxv_sim::smid::check_placements(&|| initial.clone(), &mut |mk| {
            ptxv_sim::wellsync::check_certified(mk(), CAP)
        }),
    ];
    let outputs = |v: &Verdict| v.outputs.iter().cloned().collect::<BTreeSet<_>>();
    for v in &paths {
        assert_eq!(
            v.classes(),
            ["nondeterministic", "unmodeled"],
            "{:?}",
            v.findings
        );
        assert_eq!(outputs(v), outputs(&paths[0]));
    }
}

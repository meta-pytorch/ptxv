/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_case::Case;
use ptxv_sim::{build_launch, machine::Machine};
use serde_json::json;
use std::collections::{BTreeSet, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

#[derive(Debug, Default, PartialEq, Eq)]
struct Observed {
    diagnostic: BTreeSet<(String, String)>,
    race: bool,
    deadlock: bool,
    outputs: BTreeSet<Vec<u8>>,
}
fn observe(out: &mut Observed, machine: &Machine) {
    for f in &machine.findings {
        if f.kind == "race" {
            out.race = true;
        } else if !matches!(
            f.kind.as_str(),
            "note" | "probe" | "unordered" | "race-dead"
        ) {
            out.diagnostic.insert((f.kind.clone(), f.sig.clone()));
        }
    }
}
fn raw(initial: Machine, require_distinct_notification: bool) -> (usize, Observed) {
    let mut saw_distinct_notification = false;
    let mut seen = HashSet::new();
    let mut stack = vec![initial];
    let mut out = Observed::default();
    while let Some(mut machine) = stack.pop() {
        observe(&mut out, &machine);
        if require_distinct_notification {
            for op in &machine.pending {
                if let ptxv_sim::machine::AsyncKind::CpArrive {
                    cta, mbar, writes, ..
                } = &op.kind
                {
                    saw_distinct_notification |= writes.iter().any(|write| {
                        write.off != *mbar
                            && write.data == [0; 8]
                            && !machine.mbars.contains_key(&(*cta, write.off))
                            && machine.mbar_decode_mem(*cta, write.off).is_some()
                    });
                }
            }
        }
        let mut hasher = DefaultHasher::new();
        machine.hash(&mut hasher);
        if !seen.insert(hasher.finish()) {
            continue;
        }
        assert!(seen.len() < 200_000, "raw reference hit cap");
        let actions = machine.actions_bare();
        if actions.is_empty() {
            if machine.terminal() {
                machine.exit_checks();
                observe(&mut out, &machine);
                out.outputs.insert(machine.outputs());
            } else {
                out.deadlock = true;
            }
        }
        for (action, _) in actions {
            let mut next = machine.clone();
            next.apply(&action);
            stack.push(next);
        }
    }
    assert!(
        !require_distinct_notification || saw_distinct_notification,
        "test never reached a copy into an unmaterialized encoded barrier with a distinct notification"
    );
    (seen.len(), out)
}
fn reduced(v: &ptxv_sim::explore::Verdict) -> Observed {
    assert!(v.complete, "reduced search hit cap");
    let mut out = Observed {
        outputs: v.outputs.iter().cloned().collect(),
        ..Observed::default()
    };
    for f in &v.findings {
        if f.kind == "race" {
            out.race = true;
        } else if f.kind == "deadlock" {
            out.deadlock = true;
        } else if !matches!(
            f.kind.as_str(),
            "note" | "probe" | "unordered" | "race-dead"
        ) {
            out.diagnostic.insert((f.kind.clone(), f.sig.clone()));
        }
    }
    out
}
#[test]
fn encoded_barrier_queries_match_unreduced_schedules() {
    let scratch = std::env::temp_dir().join(format!(
        "ptxv-barrier-materialization-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&scratch).unwrap();
    let mut comparisons = 0;
    let mut reference_states = 0;
    for (name, init, parity) in [
        ("uninitialized", "", 0),
        (
            "initialized_pass",
            "mbarrier.init.shared::cta.b64 [bar], 1;",
            1,
        ),
        (
            "initialized_pending",
            "mbarrier.init.shared::cta.b64 [bar], 1;",
            0,
        ),
        (
            "stored_pending",
            "mov.b64 %rd0, 0x7ffff800001ffffe; st.shared.u64 [bar], %rd0;",
            0,
        ),
        (
            "stored_low_first",
            "mov.b32 %r0, 0x001ffffe; st.shared.u32 [bar], %r0; mov.b32 %r0, 0x7ffff800; st.shared.u32 [bar+4], %r0;",
            0,
        ),
        (
            "stored_high_first",
            "mov.b32 %r0, 0x7ffff800; st.shared.u32 [bar+4], %r0; mov.b32 %r0, 0x001ffffe; st.shared.u32 [bar], %r0;",
            0,
        ),
        (
            "stored_then_invalid",
            "mov.b64 %rd0, 0x7ffff800001ffffe; st.shared.u64 [bar], %rd0; mov.b32 %r0, 0; st.shared.u8 [bar+7], %r0;",
            0,
        ),
        (
            "async_wait_pending",
            "mov.u64 %rd0, init_word; cp.async.ca.shared.global [bar], [%rd0], 8; cp.async.commit_group; cp.async.wait_group 0;",
            0,
        ),
        (
            "async_exit_pending",
            "mov.u64 %rd0, init_word; cp.async.ca.shared.global [bar], [%rd0], 8; cp.async.commit_group;",
            0,
        ),
        (
            "async_overwrite_other_barrier",
            "mov.u64 %rd0, zero_word; cp.async.ca.shared.global [bar], [%rd0], 8; cp.async.mbarrier.arrive.noinc.shared.b64 [bar+8];",
            0,
        ),
        (
            "heterogeneous_waits",
            "mbarrier.init.shared::cta.b64 [bar], 1;",
            0,
        ),
        ("ordered_stored_pending", "", 0),
        (
            "partial_unordered_high",
            "mov.b32 %r0, 0x7ffff800; st.shared.u32 [bar+4], %r0;",
            0,
        ),
        (
            "ordered_then_overwritten",
            "mov.b64 %rd0, 0; st.shared.u64 [bar], %rd0;",
            0,
        ),
        (
            "stored_pass",
            "mov.b64 %rd0, 0x7ffff800001ffffe; st.shared.u64 [bar], %rd0;",
            1,
        ),
        (
            "invalidated",
            "mbarrier.init.shared::cta.b64 [bar], 1; mbarrier.inval.shared::cta.b64 [bar];",
            0,
        ),
        (
            "initialized_complete",
            "mbarrier.init.shared::cta.b64 [bar], 1; mbarrier.arrive.shared::cta.b64 _, [bar];",
            0,
        ),
    ] {
        for threads in [3, 96] {
            let setup = if name == "async_overwrite_other_barrier" {
                "setp.eq.u32 %p0, %r0, 0; @%p0 mov.b64 %rd0, 0x7ffff800001ffffe; @%p0 st.shared.u64 [bar], %rd0; @%p0 mbarrier.init.shared::cta.b64 [bar+8], 1; bar.sync 0;"
            } else if matches!(name, "ordered_stored_pending" | "ordered_then_overwritten") {
                "setp.eq.u32 %p0, %r0, 0; @%p0 mov.b64 %rd0, 0x7ffff800001ffffe; @%p0 st.shared.u64 [bar], %rd0; bar.sync 0;"
            } else if name == "partial_unordered_high" {
                "setp.eq.u32 %p0, %r0, 0; @%p0 st.shared.u32 [bar], 0x001ffffe; bar.sync 0;"
            } else {
                ""
            };
            let address_setup = if name == "heterogeneous_waits" {
                "sub.u32 %r0, %r1, 1; mul.lo.u32 %r0, %r0, 8; mov.u64 %rd1, bar; cvt.u64.u32 %rd0, %r0; add.u64 %rd1, %rd1, %rd0;"
            } else {
                ""
            };
            let address = if name == "heterogeneous_waits" {
                "%rd1"
            } else {
                "bar"
            };
            let separate_pcs = if name == "heterogeneous_waits" {
                ""
            } else {
                "setp.eq.u32 %p0, %r1, 1; @%p0 bra POLL1;"
            };
            let select = if threads == 3 {
                "mov.u32 %r0, %tid.x; mov.u32 %r1, %r0;"
            } else {
                "mov.u32 %r0, %tid.x; shr.u32 %r1, %r0, 5;"
            };
            let inactive = if threads == 96 {
                "and.b32 %r0, %r0, 31; setp.ne.u32 %p0, %r0, 0; @%p0 ret;"
            } else {
                ""
            };
            let text = format!(
                r#".version 9.0
.target sm_100a
.address_size 64
.global .align 8 .u64 init_word = 0x7ffff800001ffffe;
.global .align 8 .u64 zero_word = 0;
.visible .entry k() {{
 .reg .b32 %r<2>; .reg .b64 %rd<2>; .reg .pred %p<2>; .shared .align 8 .b64 bar[2];
 {select}
 {setup}
 {inactive}
 setp.eq.u32 %p0, %r1, 0; @%p0 bra INIT;
 {address_setup}
 {separate_pcs}
 mbarrier.try_wait.parity.shared::cta.b64 %p1, [{address}], {parity}; ret;
POLL1:
 mbarrier.try_wait.parity.shared::cta.b64 %p1, [{address}], {parity}; ret;
INIT:
 {init}
 ret;
}}
"#
            );
            let ptx = scratch.join(format!("{name}-{threads}.ptx"));
            std::fs::write(&ptx, &text).unwrap();
            let cubin = ptx.with_extension("cubin");
            ptxv_sass::ptxas(&ptx, &cubin, "sm_100a").unwrap();
            for front in ["ptx", "sass"] {
                let kernel = Arc::new(if front == "ptx" {
                    ptxv_ptx::parse_kernel(&text, Some("k")).unwrap()
                } else {
                    ptxv_sass::load_cubin(&cubin, "k").unwrap()
                });
                assert!(
                    kernel.code.iter().any(|instruction| matches!(
                        instruction.op,
                        ptxv_ir::Op::MbarTryWait { .. }
                    )),
                    "{name}/{front}: the compiled query must remain visible to the test"
                );
                let case = Case { dir: scratch.clone(), m: serde_json::from_value(json!({
                    "name":name,"description":"raw versus reduced invalid-poll review","entry":"k",
                    "grid":[1,1,1],"block":[threads,1,1],"outputs":[],"buffers":[],"params":[],"variants":[]
                })).unwrap() };
                let initial = Machine::new(
                    kernel.clone(),
                    Arc::new(build_launch(&case, &kernel).unwrap()),
                );
                let (states, reference) =
                    raw(initial.clone(), name == "async_overwrite_other_barrier");
                reference_states += states;
                if name == "async_overwrite_other_barrier" {
                    assert!(
                        !reference.race,
                        "async data/control test became racy: {front}: {reference:?}"
                    );
                    if front == "ptx" || threads == 96 {
                        assert!(
                            reference.deadlock && !reference.outputs.is_empty(),
                            "both async orders must be reachable: {front}: {reference:?}"
                        );
                        assert!(
                            reference
                                .diagnostic
                                .iter()
                                .any(|(_, sig)| sig.starts_with("mbar-uninit:")),
                            "async landing must invalidate the encoded barrier: {front}: {reference:?}"
                        );
                    }
                }
                if matches!(
                    name,
                    "stored_pending"
                        | "uninitialized"
                        | "async_overwrite_other_barrier"
                        | "partial_unordered_high"
                        | "ordered_then_overwritten"
                ) {
                    let certificate = ptxv_sim::wellsync::check_certified(initial.clone(), 200_000);
                    if certificate.complete {
                        let observed = reduced(&certificate);
                        assert!(
                            observed.race || observed.deadlock || !observed.diagnostic.is_empty(),
                            "certificate accepted an unsafe raw graph: {name}/{threads}/{front}"
                        );
                    }
                }
                if name == "ordered_stored_pending" {
                    assert!(
                        !reference.race && !reference.deadlock && reference.diagnostic.is_empty()
                    );
                    let certificate = ptxv_sim::wellsync::certify(initial.clone())
                        .verdict
                        .expect("ordered encoded initialization should certify");
                    let observed = reduced(&certificate);
                    assert!(!observed.race && !observed.deadlock && observed.diagnostic.is_empty());
                }
                for (engine, verdict) in [
                    (
                        "stateful",
                        ptxv_sim::explore::explore(initial.clone(), 200_000),
                    ),
                    (
                        "dpor",
                        ptxv_sim::dpor::explore_dpor(initial.clone(), 200_000),
                    ),
                ] {
                    let actual = reduced(&verdict);
                    if reference.race {
                        // E-DRF preserves a race finding; outcomes after a race are not promised.
                        assert!(
                            actual.race,
                            "lost raw race: {name}/{threads}/{front}/{engine}: {actual:?}"
                        );
                    } else {
                        assert_eq!(actual, reference, "{name}/{threads}/{front}/{engine}");
                    }
                    comparisons += 1;
                }
            }
        }
    }
    std::fs::remove_dir_all(&scratch).unwrap();
    eprintln!(
        "{comparisons} raw-versus-reduced barrier comparisons passed ({reference_states} reference states)"
    );
}

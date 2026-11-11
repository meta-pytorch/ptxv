/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Debugging / figures: `PTXV_TRACE_DUMP=<dir> ptxv check ...` replays each finding's
//! witness schedule (its trace) on a fresh machine and writes `<dir>/<variant>.<front>.trace.json`: per step, the warps' pcs and
//! states, the mbarrier and named-barrier states, and the shared-memory bytes each step
//! accessed (from the race detector's shadow). Without findings it replays the schedule
//! `explore::run_once` picks. Read-only: the verdict is not affected (tools/figures/sim_gif.py
//! renders the file).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use anyhow::anyhow;
use ptxv_sim::explore::Action;
use ptxv_sim::explore::Verdict;
use ptxv_sim::machine::Epoch;
use ptxv_sim::machine::Machine;
use serde_json::Value;
use serde_json::json;

/// Shared memory larger than this (bytes per CTA) is not diffed.
const SMEM_DIFF_MAX: usize = 1 << 16;

fn warps(m: &Machine) -> Value {
    let mut by: BTreeMap<u32, Vec<(u32, u32, String)>> = BTreeMap::new();
    for (_, t) in m.live_threads() {
        by.entry(t.warp)
            .or_default()
            .push((t.lane, t.pc, format!("{:?}", t.st)));
    }
    let mut out = vec![];
    for (w, ts) in by {
        // Distinct (pc, state) groups of the warp, with their lanes.
        let mut g: BTreeMap<(u32, String), Vec<u32>> = BTreeMap::new();
        for (l, pc, st) in ts {
            g.entry((pc, st)).or_default().push(l);
        }
        let cta = m.warps[w as usize].cta;
        out.push(json!({
            "warp": w,
            "cta": cta,
            "groups": g.into_iter().map(|((pc, st), lanes)| json!({
                "pc": pc, "state": st, "lanes": lanes,
            })).collect::<Vec<_>>(),
        }));
    }
    Value::Array(out)
}

fn barriers(m: &Machine) -> Value {
    json!({
        "mbarriers": m.mbars.iter().map(|((cta, addr), b)| json!({
            "cta": cta, "addr": addr, "expected": b.expected, "pending": b.pending,
            "tx": b.tx, "phase": b.phase,
        })).collect::<Vec<_>>(),
        "bars": m.bars.iter().map(|((cta, id), b)| json!({
            "cta": cta, "id": id, "waiting": b.arrived.len(), "arrived": b.arrivers.len(),
            "count": b.count,
        })).collect::<Vec<_>>(),
    })
}

fn who(m: &Machine, e: &Epoch) -> Value {
    let n = m.nthreads();
    if e.actor < n {
        let t = &m.threads[e.actor as usize];
        json!({"thread": t.tid[0], "warp": t.warp, "cta": t.cta, "pc": e.pc})
    } else {
        json!({"async": e.actor, "pc": e.pc})
    }
}

/// Shared-memory accesses recorded between two machines' shadows, merged into byte ranges.
fn smem_diff(a: &Machine, b: &Machine) -> Vec<Value> {
    let mut out: Vec<Value> = vec![];
    for cta in 0..b.smem.len() {
        let (sa, sb) = (&a.smem[cta].sh, &b.smem[cta].sh);
        if sb.len() > SMEM_DIFF_MAX || sa.len() != sb.len() {
            continue;
        }
        // (rw, actor, pc) -> sorted byte offsets
        let mut acc: BTreeMap<(&str, u32, u32), Vec<usize>> = BTreeMap::new();
        let mut eps: BTreeMap<(u32, u32), Epoch> = BTreeMap::new();
        for i in 0..sb.len() {
            let (x, y) = (sa.get(i), sb.get(i));
            if x == y {
                continue;
            }
            if let (Some(e), true) = (y.w, y.w != x.w) {
                acc.entry(("write", e.actor, e.pc)).or_default().push(i);
                eps.insert((e.actor, e.pc), e);
            }
            for e in y.r.iter() {
                if !x.r.iter().any(|o| o == e) {
                    acc.entry(("read", e.actor, e.pc)).or_default().push(i);
                    eps.insert((e.actor, e.pc), *e);
                }
            }
        }
        for ((rw, actor, pc), offs) in acc {
            let mut runs: Vec<(usize, usize)> = vec![];
            for o in offs {
                match runs.last_mut() {
                    Some((s, n)) if *s + *n == o => *n += 1,
                    _ => runs.push((o, 1)),
                }
            }
            let e = eps[&(actor, pc)];
            for (s, n) in runs {
                out.push(json!({
                    "cta": cta, "off": s, "bytes": n, "rw": rw, "by": who(b, &e),
                    "strong": e.strong,
                }));
            }
        }
    }
    out
}

fn action_json(a: &Action) -> Value {
    match a {
        Action::Split { warp, pc, .. } => json!({"kind": "warp", "warp": warp, "pc": pc}),
        Action::Async(i) => json!({"kind": "async", "index": i}),
        Action::Launch(c) => json!({"kind": "launch", "cta": c}),
        Action::Clc(i, _) => json!({"kind": "clc", "index": i}),
    }
}

/// Replay `trace` (action descriptions) from `m`; an empty trace: run_once's schedule. With
/// `sig`, the trace is the path to the state the finding was found from: the replay then takes
/// the enabled action that reports it, and stops there.
fn replay(mut m: Machine, trace: &[String], sig: Option<&str>) -> Result<Value> {
    let mut steps = vec![];
    let initial = json!({"warps": warps(&m), "barriers": barriers(&m)});
    let mut i = 0;
    let mut found = false;
    while !found {
        let acts = m.actions();
        let pick = if trace.is_empty() && sig.is_none() {
            acts.iter()
                .find(|(a, _)| matches!(a, Action::Split { .. }))
                .or(acts.first())
                .cloned()
        } else if let Some(d) = trace.get(i) {
            Some(
                acts.iter()
                    .find(|(_, x)| x == d)
                    .cloned()
                    .ok_or_else(|| anyhow!("trace step {i} `{d}` is not enabled on replay"))?,
            )
        } else {
            // The action that reports the finding.
            let s = sig.unwrap_or_default();
            acts.iter()
                .find(|(a, _)| {
                    let mut c = m.clone();
                    c.apply(a);
                    c.findings.iter().any(|f| f.sig == s)
                })
                .cloned()
        };
        let Some((a, d)) = pick else { break };
        let before = m.clone();
        m.apply(&a);
        found = sig.is_some_and(|s| m.findings.iter().any(|f| f.sig == s));
        let fs: Vec<Value> = m
            .findings
            .drain(..)
            .map(|f| json!({"kind": f.kind, "msg": f.msg, "pcs": f.pcs, "sig": f.sig}))
            .collect();
        steps.push(json!({
            "desc": d,
            "action": action_json(&a),
            "warps": warps(&m),
            "barriers": barriers(&m),
            "smem": smem_diff(&before, &m),
            "findings": fs,
        }));
        i += 1;
    }
    let terminal = m.terminal();
    Ok(json!({"initial": initial, "steps": steps, "terminal": terminal, "reported": found}))
}

/// Write the replays of `v`'s findings (or of one run_once schedule) to `path`.
pub(crate) fn write(
    m: Machine,
    v: &Verdict,
    variant: &str,
    front: &str,
    path: &Path,
) -> Result<()> {
    let code: Vec<Value> =
        m.k.code
            .iter()
            .enumerate()
            .map(|(pc, i)| json!({"pc": pc, "loc": i.loc, "text": i.text}))
            .collect();
    let mut runs = vec![];
    if v.findings.is_empty() {
        let mut r = replay(m.clone(), &[], None)?;
        r["finding"] = Value::Null;
        runs.push(r);
    }
    for f in &v.findings {
        let mut r = match replay(m.clone(), &f.trace, Some(&f.sig)) {
            Ok(r) => r,
            Err(e) => json!({"error": e.to_string()}),
        };
        r["finding"] = json!({
            "kind": f.kind, "msg": f.msg, "pcs": f.pcs, "sig": f.sig, "gb300": f.gb300,
        });
        runs.push(r);
    }
    let doc = json!({
        "kind": "ptxv-trace-dump",
        "variant": variant,
        "front": front,
        "classes": v.classes(),
        "states": v.states,
        "complete": v.complete,
        "notes": v.notes,
        "code": code,
        "runs": runs,
    });
    std::fs::write(path, serde_json::to_vec_pretty(&doc)?)?;
    Ok(())
}

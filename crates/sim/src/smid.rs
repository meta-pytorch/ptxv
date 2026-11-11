/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `%smid` / `%nsmid` (ASSUME(E-SMID), ASSUMPTIONS.md).
//!
//! `%smid` is the SM a CTA runs on: a launch-time nondeterministic choice, in `[0, %nsmid)`.
//! A run fixes one value per CTA (`Machine::smid`, a placement); `explore_smid` runs the
//! explorer once per placement of `placements` and merges the verdicts, so a finding that
//! needs, say, two CTAs on one SM or a CTA on the upper half of the SM range is reached.
//! Only placements of the enumerated classes are covered, not every assignment.
//! `%nsmid` is `PTXV_NSMID` (default 152, GB300). A kernel that does not read `%smid` runs
//! once.

use std::collections::HashSet;
use std::sync::Arc;

use ptxv_ir::Kernel;
use ptxv_ir::Op;
use ptxv_ir::Reg;
use ptxv_ir::SReg;
use ptxv_ir::Space;
use ptxv_ir::Val;

use crate::explore::Verdict;
use crate::machine::Launch;
use crate::machine::Machine;

/// `%nsmid`: `PTXV_NSMID`, else 152 (GB300, probes/smid).
pub fn nsmid() -> u32 {
    std::env::var("PTXV_NSMID")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &u32| n > 0)
        .unwrap_or(152)
}

/// Does some instruction of `k` read `%smid`?
pub fn reads_smid(k: &Kernel) -> bool {
    k.code.iter().any(|i| {
        i.op.src_vals()
            .iter()
            .any(|v| matches!(v, Val::SReg(SReg::SmId)))
    })
}

/// Does `%smid` reach control, an address, a synchronization operand or any operand of a
/// non-arithmetic instruction other than the value it moves? A flow-insensitive register
/// taint from every `%smid` read through the arithmetic (`num::pure_op`); a tainted value
/// stored to memory taints every register written by a non-arithmetic instruction other
/// than a parameter load (a load may read it back). Returns the first such use (`Some(description)`); None: `%smid` only
/// reaches data, so control, addresses and synchronization are the same in every placement.
pub fn smid_reaches_control(k: &Kernel) -> Option<String> {
    let is_smid = |v: &Val| matches!(v, Val::SReg(SReg::SmId));
    let mut t: HashSet<Reg> = HashSet::new();
    let mut mem = false;
    loop {
        let n0 = (t.len(), mem);
        for (pc, i) in k.code.iter().enumerate() {
            let srcs = i.op.src_vals();
            let reads_smid = srcs.iter().any(is_smid);
            if i.guard.reg.is_some_and(|g| t.contains(&g)) {
                return Some(format!("guard of {}", k.describe(pc)));
            }
            if crate::num::pure_op(&i.op) {
                if reads_smid || i.op.src_regs().iter().any(|r| t.contains(r)) {
                    t.extend(i.op.dst_regs());
                }
                continue;
            }
            let ctl = crate::num::control_regs(&i.op);
            let ctl_smid = {
                let mut vals = srcs.clone();
                for v in crate::num::value_vals_pub(&i.op) {
                    if let Some(x) = vals.iter().position(|o| *o == v) {
                        vals.remove(x);
                    }
                }
                vals.iter().any(is_smid)
            };
            if ctl_smid || ctl.iter().any(|r| t.contains(r)) {
                return Some(format!("operand of {}", k.describe(pc)));
            }
            let val_t = reads_smid || crate::num::value_regs(&i.op).iter().any(|r| t.contains(r));
            // A load from parameter space never reads a stored value.
            let param_ld = matches!(
                i.op,
                Op::Ld {
                    space: Space::Param,
                    ..
                }
            );
            if val_t || (mem && !param_ld) {
                mem |= val_t && !param_ld;
                t.extend(i.op.dst_regs());
            }
        }
        if (t.len(), mem) == n0 {
            return None;
        }
    }
}

/// CTA index within its grid, per global CTA id.
fn local_ids(l: &Launch) -> Vec<u32> {
    (0..l.ctas()).map(|c| c - l.g(c).cta0).collect()
}

/// The `spread` placement: CTA i of each grid on SM i mod nsmid (every CTA alone while the
/// grid fits the GPU).
pub fn spread(l: &Launch, nsmid: u32) -> Vec<u32> {
    local_ids(l).into_iter().map(|i| i % nsmid).collect()
}

/// One placement by class name: `spread`, `same` (every CTA on SM 0), `rev` (CTA i on
/// nsmid - 1 - i mod nsmid), `alt` (alternately from the bottom and the top of the range:
/// both halves with few CTAs), `rand<k>` (pseudo-random, seed k).
pub fn placement(l: &Launch, nsmid: u32, class: &str) -> anyhow::Result<Vec<u32>> {
    let ids = local_ids(l);
    Ok(match class {
        "spread" => spread(l, nsmid),
        "same" => vec![0; ids.len()],
        "rev" => ids.iter().map(|i| nsmid - 1 - i % nsmid).collect(),
        "alt" => ids
            .iter()
            .map(|i| {
                let h = (i / 2) % nsmid;
                if i % 2 == 0 { h } else { nsmid - 1 - h }
            })
            .collect(),
        c if c.starts_with("rand") => {
            let seed: u64 = c[4..]
                .parse()
                .map_err(|_| anyhow::anyhow!("PTXV_SMID: bad class {c}"))?;
            let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x5851_f42d_4c95_7f2d;
            (0..ids.len())
                .map(|_| {
                    // splitmix64
                    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
                    let mut z = x;
                    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                    ((z ^ (z >> 31)) % nsmid as u64) as u32
                })
                .collect()
        }
        c => anyhow::bail!("PTXV_SMID: unknown class {c} (spread, same, rev, alt, rand<k>)"),
    })
}

/// The default classes.
pub const DEFAULT_CLASSES: &str = "spread,same,rev,alt";

/// Placements named by `mode` (comma-separated classes), duplicates dropped.
pub fn placements(l: &Launch, nsmid: u32, mode: &str) -> Placements {
    let mut out: Vec<(String, Vec<u32>)> = Vec::new();
    for c in mode.split(',').map(str::trim).filter(|c| !c.is_empty()) {
        let p = placement(l, nsmid, c)?;
        if !out.iter().any(|(_, q)| *q == p) {
            out.push((c.to_string(), p));
        }
    }
    if out.is_empty() {
        anyhow::bail!("PTXV_SMID: no placement class");
    }
    Ok(out)
}

/// Merge verdicts of runs of one launch under different nondeterministic choices: findings
/// of every run (deduplicated by signature, the message prefixed with `[tag: name]`),
/// complete only if every run was, outputs pooled, notes united.
pub fn merge(tag: &str, runs: Vec<(String, Verdict)>) -> Verdict {
    let mut merged: Option<Verdict> = None;
    let mut outs: std::collections::BTreeSet<Vec<u8>> = Default::default();
    for (name, mut v) in runs {
        for f in &mut v.findings {
            f.msg = format!("[{tag}: {name}] {}", f.msg);
        }
        outs.extend(v.outputs.iter().cloned());
        match &mut merged {
            None => merged = Some(v),
            Some(a) => {
                a.states += v.states;
                a.transitions += v.transitions;
                a.complete &= v.complete;
                a.terminal_states += v.terminal_states;
                a.seconds += v.seconds;
                for f in v.findings {
                    if !a.findings.iter().any(|x| x.sig == f.sig) {
                        a.findings.push(f);
                    }
                }
                for n in v.notes {
                    if !a.notes.contains(&n) {
                        a.notes.push(n);
                    }
                }
                for s in v.assumptions {
                    if !a.assumptions.contains(&s) {
                        a.assumptions.push(s);
                    }
                }
            }
        }
    }
    let mut v = merged.expect("at least one run");
    v.outputs = outs.into_iter().collect();
    v.distinct_outputs = v.outputs.len();
    v
}

/// The placements to run for `m0`: None for one plain run (no `%smid` read, or the
/// placement already fixed by an outer enumeration).
/// Named placements (`placements`).
type Placements = anyhow::Result<Vec<(String, Vec<u32>)>>;

fn to_run(m0: &Machine) -> Option<(String, Placements)> {
    if !m0.smid_used || m0.smid_fixed {
        return None;
    }
    let mode = std::env::var("PTXV_SMID").unwrap_or_else(|_| DEFAULT_CLASSES.into());
    let ps = placements(&m0.launch, m0.nsmid, &mode);
    Some((mode, ps))
}

/// ASSUME(E-SMID): when `%smid` reaches control, an address or synchronization, the
/// placements are a sample of the assignments (and `%smid` may change mid-run): the merged
/// verdict is not complete. Findings stand. When it reaches only data, control and accesses
/// are those of every placement, and the pooled outputs show any placement dependence.
fn sampled(v: &mut Verdict, k: &Kernel, mode: &str) {
    if let Some(why) = smid_reaches_control(k) {
        v.complete = false;
        v.notes.push(format!(
            "E-SMID: smid-dependent control ({why}): placements sampled ({mode}), not every assignment, and %smid fixed per CTA; the verdict is incomplete"
        ));
    }
}

fn note(n: usize, mode: &str, nsmid: u32) -> String {
    format!(
        "E-SMID: %smid read; {n} SM placement(s) explored ({mode}, nsmid {nsmid}); other placements are not covered"
    )
}

/// ASSUME(E-SMID): `explore_smid` for checks that build their machines (`make`): `check`
/// runs once per placement with a `make` whose machines have it fixed.
pub fn check_placements(
    make: &dyn Fn() -> Machine,
    check: &mut dyn FnMut(&dyn Fn() -> Machine) -> Verdict,
) -> Verdict {
    let m0 = make();
    let probe = crate::clock::clock_mark(&m0);
    let Some((mode, ps)) = to_run(&m0) else {
        drop(m0);
        let mut v = check(make);
        crate::clock::mark_clock_outputs(&mut v, &probe);
        return v;
    };
    let nsmid = m0.nsmid;
    let m0k = m0.k.clone();
    drop(m0);
    let ps = match ps {
        Ok(p) => p,
        Err(e) => {
            let mut v = check(make);
            v.complete = false;
            v.notes.push(format!("E-SMID: {e}"));
            return v;
        }
    };
    let n = ps.len();
    let runs = ps
        .into_iter()
        .map(|(name, p)| {
            let p = Arc::new(p);
            let mk = || {
                let mut m = make();
                m.smid = p.clone();
                m.smid_fixed = true;
                m
            };
            (name, check(&mk))
        })
        .collect();
    let mut v = merge("smid placement", runs);
    v.notes.push(note(n, &mode, nsmid));
    sampled(&mut v, &m0k, &mode);
    crate::clock::mark_clock_outputs(&mut v, &probe);
    v
}

/// ASSUME(E-SMID): run `run` once per `%smid` placement (`PTXV_SMID`, default
/// `DEFAULT_CLASSES`) and merge the verdicts. One plain run when no instruction reads
/// `%smid`.
pub fn explore_smid(m0: Machine, run: &mut dyn FnMut(Machine) -> Verdict) -> Verdict {
    let fixed = m0.smid_fixed;
    let probe = crate::clock::clock_mark(&m0);
    let mut v = explore_smid_inner(m0, run);
    // ASSUME(E-CLOCK): once per launch check (not again inside an outer enumeration).
    if !fixed {
        crate::clock::mark_clock_outputs(&mut v, &probe);
    }
    v
}

fn explore_smid_inner(m0: Machine, run: &mut dyn FnMut(Machine) -> Verdict) -> Verdict {
    let Some((mode, ps)) = to_run(&m0) else {
        return run(m0);
    };
    let ps = match ps {
        Ok(p) => p,
        Err(e) => {
            let mut v = run(m0);
            v.complete = false;
            v.notes.push(format!("E-SMID: {e}"));
            return v;
        }
    };
    let n = ps.len();
    let runs = ps
        .into_iter()
        .map(|(name, p)| {
            let mut m = m0.clone();
            m.smid = Arc::new(p);
            m.smid_fixed = true;
            (name, run(m))
        })
        .collect();
    let mut v = merge("smid placement", runs);
    v.notes.push(note(n, &mode, m0.nsmid));
    sampled(&mut v, &m0.k, &mode);
    v
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Multi-fill rerun (S-INIT-FILLS): an evidence class for undefined values that reach only
//! output data, not a certificate.
//!
//! When every S-INIT report of a verdict is undefined data left in global memory at the end
//! of the launch (`uninit-gmem-exit`), no branch, address, predicate, guard or
//! synchronization operand read an undefined bit on the explored paths (those are sinks,
//! reported as `uninit-reg`), so the undefined values steer nothing: they only feed data.
//! The launch is then rerun on one deterministic schedule once per fill, with every
//! unwritten register, uniform register, shared-memory byte outside the reserved prefix,
//! TMEM word, never-stored local word and unwritten byte of a scratch buffer holding the
//! fill (0, NaN, +Inf, -Inf, two pseudo-random patterns), and the output buffers are compared bitwise. Equal outputs turn
//! those reports into kind `undef-insensitive` (empirical, N fills); anything else leaves
//! them as they are, with a note saying why.

use crate::explore::DetSched;
use crate::explore::FaultLoop;
use crate::explore::Verdict;
use crate::machine::Finding;
use crate::machine::Machine;
use crate::machine::SMEM_RESERVED;

/// What unwritten state holds for one rerun.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fill {
    /// The same 32-bit word everywhere.
    Word(u32),
    /// A pseudo-random word per location (splitmix64 of the seed and the location).
    Random(u64),
}

impl Fill {
    /// The 32-bit word at location `key` (any stable per-location number).
    pub fn word(self, key: u64) -> u32 {
        match self {
            Fill::Word(w) => w,
            Fill::Random(s) => {
                let mut z = s
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add(key)
                    .wrapping_add(0x9e37_79b9_7f4a_7c15);
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                (z ^ (z >> 31)) as u32
            }
        }
    }
}

/// The fills, in run order.
pub const FILLS: [(&str, Fill); 6] = [
    ("0", Fill::Word(0)),
    ("NaN", Fill::Word(0x7fff_ffff)),
    ("+Inf", Fill::Word(0x7f80_0000)),
    ("-Inf", Fill::Word(0xff80_0000)),
    ("random 1", Fill::Random(1)),
    ("random 2", Fill::Random(2)),
];

const NOTE: &str = "undef-fills:";

/// Gives every unwritten location of `m` (a machine before its first step) the fill's
/// value. Never-stored local words take it when loaded (`Machine::fill`).
pub fn apply_fill(m: &mut Machine, f: Fill) {
    m.fill = Some(f);
    for t in 0..m.threads.len() {
        let th = m.threads[t].make_mut();
        let written = th.written.clone();
        for (i, r) in th.regs.iter_mut().enumerate() {
            if written.get(i / 64).is_none_or(|w| w >> (i % 64) & 1 == 0) {
                let k = (t as u64) << 20 | i as u64;
                *r = f.word(k << 1) as u64 | (f.word(k << 1 | 1) as u64) << 32;
            }
        }
    }
    for w in 0..m.warps.len() {
        let wp = m.warps[w].make_mut();
        let written = wp.ur_written;
        for (i, r) in wp.ur.iter_mut().enumerate() {
            if written.get(i / 64).is_none_or(|x| x >> (i % 64) & 1 == 0) {
                *r = f.word(1 << 40 | (w as u64) << 20 | i as u64) as u64;
            }
        }
    }
    for c in 0..m.smem.len() {
        let mem = &mut m.smem[c];
        for b in SMEM_RESERVED..mem.bytes.len() {
            if mem.sh.get(b).w.is_none() {
                let wd = f.word(2 << 40 | (c as u64) << 24 | (b / 4) as u64);
                *mem.bytes.get_mut(b) = (wd >> (8 * (b % 4))) as u8;
            }
        }
    }
    // S-INIT scratch buffers: global bytes undefined from the start (`Alloc::undef`) and not
    // written yet.
    for a in 0..m.gmem.len() {
        if !m.launch.allocs.get(a).is_some_and(|x| x.undef) {
            continue;
        }
        let mem = &mut m.gmem[a];
        for b in 0..mem.bytes.len() {
            let sh = mem.sh.get(b);
            if sh.w.is_none() && sh.undef {
                let wd = f.word(4 << 40 | (a as u64) << 32 | (b / 4) as u64);
                *mem.bytes.get_mut(b) = (wd >> (8 * (b % 4))) as u8;
            }
        }
    }
    for c in 0..m.tmem.len() {
        let tm = &mut m.tmem[c];
        for i in 0..tm.words.len() {
            if *tm.init.get(i) != 1 {
                *tm.words.get_mut(i) = f.word(3 << 40 | (c as u64) << 24 | i as u64);
            }
        }
    }
}

/// One deterministic run of `m` to its end: (output bytes, findings), or why not.
fn run(mut m: Machine) -> Result<(Vec<u8>, Vec<Finding>), String> {
    let mut fs = Vec::new();
    let mut sched = DetSched::default();
    let mut faults = FaultLoop::default();
    while let Some(a) = sched.pick(&m) {
        m.apply(&a);
        if let Some(why) = faults.repeats(&m.findings) {
            return Err(why);
        }
        fs.append(&mut m.findings);
    }
    if !m.terminal() {
        return Err("the deterministic run does not reach the end of the launch".into());
    }
    m.exit_checks();
    fs.append(&mut m.findings);
    Ok((m.outputs(), fs))
}

/// An S-INIT report other than undefined data left in global memory at the end: an
/// undefined value reached a branch, an address, a predicate, a synchronization operand, or
/// a reader other than a load.
fn steering(f: &Finding) -> bool {
    f.sig.starts_with("uninit-") && !f.sig.starts_with("uninit-gmem-exit:")
}

/// Runs the fills when `v`'s S-INIT reports are all `uninit-gmem-exit` violations, and
/// reclassifies them as `undef-insensitive` if every fill gives the same outputs. `m0` is
/// the machine `v` was computed from (before its first step). Idempotent; off with
/// `PTXV_NO_FILLS`.
// ASSUME(S-INIT-FILLS): see ASSUMPTIONS.md
pub fn check(m0: &Machine, v: &mut Verdict) {
    let exits: Vec<String> = v
        .findings
        .iter()
        .filter(|f| f.kind == "violation" && f.sig.starts_with("uninit-gmem-exit:"))
        .map(|f| f.sig.clone())
        .collect();
    if exits.is_empty()
        || v.findings.iter().any(steering)
        || v.notes.iter().any(|n| n.starts_with(NOTE))
        || std::env::var("PTXV_NO_FILLS").is_ok()
    {
        return;
    }
    // Undefined data that comes from an ordering bug (a read before the write in some
    // schedule) depends on the schedule: one deterministic schedule says nothing about it.
    // ASSUME(E-WARPID): the fills are arbitrary words, not the slots %warpid can hold; a
    // kernel that reads it keeps its reports.
    if m0.k.code.iter().any(|i| ptxv_ir::reads_warpid(&i.op)) {
        v.notes.push(format!(
            "{NOTE} not run (the kernel reads %warpid, E-WARPID: its undefined values stand for the warp's SM slot, which the fills do not range over)"
        ));
        return;
    }
    let classes = v.classes();
    if classes
        .iter()
        .any(|c| c == "race" || c == "nondeterministic")
    {
        v.notes.push(format!(
            "{NOTE} not run (the verdict has a race or several outcomes: undefined values may depend on the schedule)"
        ));
        return;
    }
    let mut first: Option<Vec<u8>> = None;
    let mut differs = None;
    for (name, f) in FILLS {
        // The whole check's time budget (membudget.rs) is spent: the reruns stop, the
        // reports stay.
        if crate::membudget::time_budget_spent() {
            v.notes.push(format!(
                "{NOTE} not run (stopped at {} before fill {name})",
                crate::membudget::TIME_BUDGET
            ));
            return;
        }
        let mut m = m0.clone();
        apply_fill(&mut m, f);
        match run(m) {
            Err(why) => {
                v.notes.push(format!("{NOTE} not run (fill {name}: {why})"));
                return;
            }
            Ok((out, fs)) => {
                // The schedule must reach every reported undefined output, or the fills test
                // nothing.
                if let Some(sig) = exits.iter().find(|s| !fs.iter().any(|f| &f.sig == *s)) {
                    v.notes.push(format!(
                        "{NOTE} not run (the deterministic schedule with fill {name} does not reach {sig})"
                    ));
                    return;
                }
                if let Some(s) = fs.iter().find(|f| steering(f)) {
                    v.notes.push(format!(
                        "{NOTE} refused: with fill {name} an undefined value is used beyond output data ({})",
                        s.sig
                    ));
                    return;
                }
                match &first {
                    None => first = Some(out),
                    Some(o) if *o != out => {
                        let at = o.iter().zip(&out).position(|(a, b)| a != b);
                        differs = Some((name, at));
                        break;
                    }
                    Some(_) => {}
                }
            }
        }
    }
    if let Some((name, at)) = differs {
        v.notes.push(format!(
            "{NOTE} outputs depend on the undefined values (fill {name} differs from fill 0{})",
            at.map(|b| format!(" at output byte {b:#x}"))
                .unwrap_or_default()
        ));
        return;
    }
    let n = FILLS.len();
    v.notes.push(format!(
        "{NOTE} {n} fills (0, NaN, +Inf, -Inf, 2 random) of every unwritten register, shared / TMEM / local word: outputs bitwise equal (S-INIT-FILLS, empirical)"
    ));
    for f in v.findings.iter_mut() {
        if f.kind == "violation" && f.sig.starts_with("uninit-gmem-exit:") {
            f.kind = "undef-insensitive".into();
            f.msg.push_str(&format!(
                "; the outputs do not depend on the undefined values in {n} fills (empirical, not a certificate: S-INIT-FILLS)"
            ));
        }
    }
}

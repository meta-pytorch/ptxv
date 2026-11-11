/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! ASSUME(E-HW-SPLIT-UR): uniform-register windows of a SASS kernel.
//!
//! A warp has one copy of its uniform registers, which its splits share. ptxas computes
//! split-dependent values there (`VOTEU.ANY UR` is the split's lane mask, `R2UR` its first
//! active lane's value, `REDUX` its reduction) and reads them a few instructions later
//! (`bar.warp.sync`'s fast path: `VOTEU.ANY UR5; STS; LOP3 P0, .., UR5`). GB300 does not
//! switch between a warp's splits inside a stretch without a `YIELD` or a blocking instruction
//! (probes/split_yield, probes/ur_window), so a split inside such a window runs to its read
//! before another split of its warp moves: the model's switch there (another split's VOTEU
//! between the write and the read) is not a hardware schedule.
//!
//! A window starts at a split-dependent uniform write and covers the instructions up to the
//! last one that reads the value (or a uniform value computed from it) inside one stretch:
//! no instruction in it is a branch target, and it ends at a control, synchronization,
//! `YIELD` or `NANOSLEEP` instruction (that instruction is in the window when it reads the
//! value). `windows(k)[pc]`: a split whose next instruction is `pc` is inside a window.

use std::collections::HashSet;

use ptxv_ir::File;
use ptxv_ir::Front;
use ptxv_ir::Kernel;
use ptxv_ir::Op;
use ptxv_ir::Reg;

fn uniform(r: &Reg) -> bool {
    matches!(r.file, File::UR | File::UP)
}

/// A uniform-register write whose value can differ between the splits of a warp.
fn split_dependent(op: &Op) -> bool {
    matches!(
        op,
        Op::VoteAny { .. }
            | Op::VoteAll { .. }
            | Op::R2Ur { .. }
            | Op::Redux { .. }
            | Op::MatchAny { .. }
    ) && op.dst_regs().iter().any(uniform)
}

/// An instruction at which a split may stop or the hardware may switch splits, or that leaves
/// the stretch.
fn stop(k: &Kernel, pc: usize) -> bool {
    let i = &k.code[pc];
    let t = i.text.trim_start();
    let t = t.strip_prefix('@').map_or(t, |r| {
        r.split_once(char::is_whitespace)
            .map_or("", |x| x.1.trim_start())
    });
    t.starts_with("YIELD")
        || t.starts_with("NANOSLEEP")
        || matches!(
            i.op,
            Op::Bra { .. }
                | Op::Brx { .. }
                | Op::Exit
                | Op::Ret { .. }
                | Op::Call { .. }
                | Op::CallAbs { .. }
                | Op::Trap { .. }
                | Op::WarpSync { .. }
                | Op::EndCollective { .. }
                | Op::Bssy { .. }
                | Op::Bsync { .. }
                | Op::Break { .. }
                | Op::BMov { .. }
                | Op::BarSync { .. }
                | Op::BarResult { .. }
                | Op::ClusterArrive { .. }
                | Op::ClusterWait
                | Op::MbarTryWait { .. }
                | Op::Converge
        )
}

/// Instructions control can reach other than from the one before: branch and call targets and
/// the instruction after a call (a return lands there).
fn entries(k: &Kernel) -> HashSet<usize> {
    let mut out = HashSet::new();
    for (pc, i) in k.code.iter().enumerate() {
        match &i.op {
            Op::Bra { target, .. } | Op::Call { target } => {
                out.insert(*target);
            }
            Op::Brx { targets, .. } => out.extend(targets.iter().copied()),
            Op::EndCollective { start } => {
                out.insert(*start);
            }
            _ => {}
        }
        if matches!(i.op, Op::Call { .. } | Op::CallAbs { .. }) {
            out.insert(pc + 1);
        }
    }
    out
}

fn reads(i: &ptxv_ir::Inst) -> Vec<Reg> {
    let mut r = i.op.src_regs();
    r.extend(i.guard.reg);
    if let Op::Bra { cond: Some(c), .. } | Op::Break { cond: Some(c), .. } = &i.op {
        r.extend(c.reg);
    }
    r
}

/// `windows(k)[pc]`: a split about to execute `pc` is inside a uniform-register window (SASS
/// only; empty when the kernel has none).
pub fn windows(k: &Kernel) -> Vec<bool> {
    if k.front != Front::Sass {
        return Vec::new();
    }
    let n = k.code.len();
    let entry = entries(k);
    let mut inside = vec![false; n];
    let mut any = false;
    for (i, inst) in k.code.iter().enumerate() {
        if !split_dependent(&inst.op) {
            continue;
        }
        let mut live: HashSet<Reg> = inst.op.dst_regs().into_iter().filter(uniform).collect();
        let mut last = None;
        let mut j = i + 1;
        while j < n && !entry.contains(&j) && !live.is_empty() {
            let x = &k.code[j];
            let rd = reads(x).iter().any(|r| live.contains(r));
            if rd {
                last = Some(j);
            }
            if stop(k, j) {
                break;
            }
            for d in x.op.dst_regs().into_iter().filter(uniform) {
                if rd {
                    live.insert(d);
                } else {
                    live.remove(&d);
                }
            }
            j += 1;
        }
        if let Some(l) = last {
            for x in &mut inside[i + 1..=l] {
                *x = true;
            }
            any = true;
        }
    }
    if any { inside } else { Vec::new() }
}

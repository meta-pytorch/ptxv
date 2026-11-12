/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Clock-paced loops (ASSUME(E-CLOCK), ASSUMPTIONS.md).
//!
//! With the clock reading 0 (E-CLOCK's default), a loop that waits for the clock to advance
//! never ends. Two cases:
//! - a **pure delay loop** (`delay_loops`): a backward conditional branch whose body, from
//!   its target to the branch, is straight-line register arithmetic that reads the clock,
//!   whose branch condition derives from those clock reads, and whose written registers are
//!   all dead after the loop. Its iterations have no effect but time, and the hardware clock
//!   is monotone and otherwise unconstrained, so the deadline may pass at the first check:
//!   the branch falls through (`Machine::delay_pcs`). Every other thread still interleaves
//!   around it, so no behaviour is lost.
//! - any **other loop that reads the clock** (`clock_loops`) and comes back to its branch
//!   with the same thread state makes no progress: it ends with an `unmodeled` finding
//!   ("clock-paced loop"), not a run that never stops.

/// `Action::Split::leader` at a clock branch: fall through / take it (`clock_branches`).
pub const CLOCK_FALL: usize = usize::MAX - 1;
pub const CLOCK_TAKE: usize = usize::MAX;
/// Back edges of clock branches a thread takes in a row before the loop is a `clock-paced`
/// finding (each iteration is another wait the hardware clock may allow).
pub const MAX_CLOCK_ITERS: u8 = 2;

use std::collections::BTreeSet;
use std::collections::HashSet;

use ptxv_ir::Kernel;
use ptxv_ir::Op;
use ptxv_ir::Reg;
use ptxv_ir::Val;

fn reads_clock(op: &Op) -> bool {
    op.src_vals()
        .iter()
        .any(|v| matches!(v, Val::SReg(s) if s.is_clock()))
}

/// Backward conditional branches `(branch pc, target)` (plain `BRA` / `bra`).
fn back_edges(k: &Kernel) -> Vec<(usize, usize)> {
    k.code
        .iter()
        .enumerate()
        .filter_map(|(pc, i)| match &i.op {
            Op::Bra {
                target,
                any: false,
                tid0_falls: false,
                all: false,
                cond: None,
                div: None,
            } if *target <= pc && !i.guard.is_always() => Some((pc, *target)),
            _ => None,
        })
        .collect()
}

/// Back-edge branches of loops whose body reads the clock.
pub fn clock_loops(k: &Kernel) -> HashSet<u32> {
    back_edges(k)
        .into_iter()
        .filter(|&(b, h)| k.code[h..=b].iter().any(|i| reads_clock(&i.op)))
        .map(|(b, _)| b as u32)
        .collect()
}

/// Back-edge branches of pure delay loops (module doc).
pub fn delay_loops(k: &Kernel) -> HashSet<u32> {
    let mut out = HashSet::new();
    for (b, h) in back_edges(k) {
        let body = &k.code[h..b];
        if !body.iter().any(|i| reads_clock(&i.op)) {
            continue;
        }
        // Straight-line register arithmetic only (a guarded instruction is fine: it has no
        // effect but on registers).
        if !body.iter().all(|i| crate::num::pure_op(&i.op)) {
            continue;
        }
        // Clock taint through the body (one pass: straight-line code, and a value carried
        // around the loop is a written register, which must be dead after it anyway).
        let mut t: HashSet<Reg> = HashSet::new();
        let mut written: BTreeSet<Reg> = BTreeSet::new();
        for i in body {
            if reads_clock(&i.op) || i.op.src_regs().iter().any(|r| t.contains(r)) {
                t.extend(i.op.dst_regs());
            } else {
                for r in i.op.dst_regs() {
                    t.remove(&r);
                }
            }
            written.extend(i.op.dst_regs());
        }
        let Some(g) = k.code[b].guard.reg else {
            continue;
        };
        if !t.contains(&g) {
            continue;
        }
        if crate::spin::live_after(&k.code, b + 1, &written).is_some() {
            continue;
        }
        out.insert(b as u32);
    }
    out
}

fn succs(k: &Kernel, pc: usize) -> Vec<usize> {
    let i = &k.code[pc];
    let next = pc + 1;
    let v = match &i.op {
        Op::Bra {
            target,
            cond,
            tid0_falls,
            any,
            ..
        } => {
            if i.guard.reg.is_none() && cond.is_none() && !tid0_falls && !any {
                vec![*target]
            } else {
                vec![*target, next]
            }
        }
        Op::Brx { targets, .. } => targets.clone(),
        Op::Exit | Op::Ret { .. } | Op::Trap { .. } if i.guard.reg.is_none() => vec![],
        Op::Call { target } => vec![*target, next],
        _ => vec![next],
    };
    v.into_iter().filter(|&x| x < k.code.len()).collect()
}

/// Per instruction, the registers that may hold a clock-derived value when it starts
/// (forward dataflow over the code's control flow, union at joins; an unguarded write of a
/// clock-free value clears its destination). Through local memory: a clock value stored to
/// a stack slot (by the address's immediate offset: vprintf arguments, a spilled start time)
/// taints local loads that overlap it. Shared / global memory: `Machine::ctaint` at run time.
/// Empty when the kernel never reads the clock.
/// Taint at the start of an instruction, per register: clean (absent), tainted on every
/// path (`t`), or tainted except under one guard, where a clean write replaced it (`k`:
/// register -> that guard). A read under the same guard sees the clean value (`@!P1 LDC R4;
/// @!P1 LDG [R4]`): ptxas's predicated code reuses registers that held a clock value on
/// other paths.
#[derive(Clone, Default, PartialEq)]
pub(crate) struct TState {
    t: HashSet<Reg>,
    k: std::collections::HashMap<Reg, ptxv_ir::Pred>,
}

impl TState {
    /// Is `r` tainted for an instruction guarded by `g`?
    fn has(&self, r: &Reg, g: ptxv_ir::Pred) -> bool {
        self.t.contains(r) || self.k.get(r).is_some_and(|h| g.reg.is_none() || *h != g)
    }
    /// Join `o` into `self` (clean < killed-under-g < tainted); true if it changed.
    fn join(&mut self, o: &TState) -> bool {
        let before = (self.t.len(), self.k.len());
        let mut changed = false;
        for r in &o.t {
            if self.t.insert(*r) {
                changed = true;
            }
            self.k.remove(r);
        }
        for (r, g) in &o.k {
            if self.t.contains(r) {
                continue;
            }
            match self.k.get(r) {
                Some(h) if h == g => {}
                Some(_) => {
                    self.k.remove(r);
                    self.t.insert(*r);
                    changed = true;
                }
                None => {
                    self.k.insert(*r, *g);
                    changed = true;
                }
            }
        }
        changed || before != (self.t.len(), self.k.len())
    }
}

/// The stack bytes a local access may touch: its immediate offset range when the address is
/// the stack pointer (SASS `R1`, the PTX front's frame) plus an immediate, else every byte (a
/// computed local address).
fn slot(a: &ptxv_ir::Addr, bytes: u32) -> (i64, i64) {
    let sp = |v: &Option<Val>| match v {
        None => true,
        Some(Val::Reg(r, _)) => r.file == ptxv_ir::File::R && r.idx == 1,
        Some(Val::Imm(_)) => true,
        _ => false,
    };
    if sp(&a.base) && sp(&a.base2) {
        (a.off, a.off + bytes as i64)
    } else {
        (i64::MIN, i64::MAX)
    }
}

fn clock_taint(k: &Kernel) -> Vec<TState> {
    taint(k, false, false)
}

/// `clock_taint` for data: also through guards (an instruction under a clock-derived guard
/// writes a clock-dependent value: `@P0 MOV R5, 1`), for `clock_stores`.
fn clock_data_taint(k: &Kernel) -> Vec<TState> {
    taint(k, false, true)
}

/// `clock_taint`, or with `loads` the registers that may hold a value derived from a
/// shared / global memory read (the result of any non-arithmetic instruction); with `data`,
/// a clock-derived guard taints what its instruction writes.
fn taint(k: &Kernel, loads: bool, data: bool) -> Vec<TState> {
    let n = k.code.len();
    let mut ins: Vec<TState> = vec![TState::default(); n];
    if !loads && !k.code.iter().any(|i| reads_clock(&i.op)) {
        return ins;
    }
    // Local memory a clock value is stored to, per kernel of a concatenated launch: (entry,
    // first byte, end) by the address's immediate offset (stack slots: vprintf arguments, a
    // spilled start time).
    let mut local: HashSet<(usize, i64, i64)> = HashSet::new();
    loop {
        let flags = local.len();
        let mut work: Vec<usize> = (0..n).rev().collect();
        let mut queued = vec![true; n];
        while let Some(pc) = work.pop() {
            queued[pc] = false;
            let i = &k.code[pc];
            let st = &ins[pc];
            let g = i.guard;
            // (A clock-derived guard is a choice, `clock_branches`: each side writes its
            // destination from its own sources.)
            let src_t = reads_clock(&i.op)
                || i.op.src_regs().iter().any(|r| st.has(r, g))
                || data
                    && g.reg.is_some_and(|p| {
                        st.has(
                            &p,
                            ptxv_ir::Pred {
                                reg: None,
                                neg: false,
                            },
                        )
                    });
            let val_t = crate::num::value_regs(&i.op).iter().any(|r| st.has(r, g));
            let dst_t = if crate::num::pure_op(&i.op) {
                src_t
            } else {
                match &i.op {
                    Op::Ld {
                        space: ptxv_ir::Space::Param,
                        ..
                    }
                    | Op::St {
                        space: ptxv_ir::Space::Param,
                        ..
                    }
                    | Op::CallAbs { .. } => false,
                    Op::StLocal { bytes, addr, .. } => {
                        if val_t {
                            let (lo, hi) = slot(addr, *bytes);
                            local.insert((k.entry_of(pc), lo, hi));
                        }
                        false
                    }
                    Op::LdLocal { bytes, addr, .. } => {
                        let (e, (lo, hi)) = (k.entry_of(pc), slot(addr, *bytes));
                        src_t || local.iter().any(|&(x, a, b)| x == e && a < hi && lo < b)
                    }
                    // A clock value stored to shared / global memory: control that loads it
                    // back is a clock branch once the loaded bytes are clock-derived
                    // (`mem_branches`, `Machine::ctaint`).
                    _ => src_t || loads,
                }
            };
            let mut out = st.clone();
            for r in i.op.dst_regs() {
                if dst_t {
                    out.t.insert(r);
                    out.k.remove(&r);
                } else if g.is_always() {
                    out.t.remove(&r);
                    out.k.remove(&r);
                } else if out.t.remove(&r) {
                    out.k.insert(r, g);
                } else if out.k.get(&r).is_some_and(|h| *h != g) {
                    out.k.remove(&r);
                    out.t.insert(r);
                }
                // A rewritten predicate ends the kills under it: tainted again.
                if matches!(r.file, ptxv_ir::File::P | ptxv_ir::File::UP) {
                    let ended: Vec<Reg> = out
                        .k
                        .iter()
                        .filter(|(_, q)| q.reg == Some(r))
                        .map(|(x, _)| *x)
                        .collect();
                    for x in ended {
                        out.k.remove(&x);
                        out.t.insert(x);
                    }
                }
            }
            // A hang guard's (clock.rs `leads_to_trap`) non-trap side knows its guard: the
            // timeout did not fire (E-CLOCK), so the guard carries no clock value there.
            let guard_known = match &i.op {
                Op::Bra { target, .. } if g.reg.is_some() => {
                    if leads_to_trap(k, *target) {
                        Some(pc + 1)
                    } else if leads_to_trap(k, pc + 1) {
                        Some(*target)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            let part = k.entry_of(pc);
            for s in succs(k, pc) {
                // Control flow stays within one kernel of a concatenated launch.
                if k.entry_of(s) != part {
                    continue;
                }
                let changed = if guard_known == Some(s) {
                    let mut o = out.clone();
                    let gr = g.reg.unwrap();
                    o.t.remove(&gr);
                    o.k.remove(&gr);
                    ins[s].join(&o)
                } else {
                    ins[s].join(&out)
                };
                if changed && !queued[s] {
                    queued[s] = true;
                    work.push(s);
                }
            }
        }
        if local.len() == flags {
            return ins;
        }
    }
}

/// Does the code from `pc` reach a trap (BPT.TRAP, `__assertfail`) through straight-line
/// code (unconditional branches followed)? A timeout's print-and-trap path: a hang guard
/// (E-CLOCK). What runs before the trap does not matter: the trap aborts the launch, and
/// the trap itself is what E-CLOCK assumes never happens.
fn leads_to_trap(k: &Kernel, mut pc: usize) -> bool {
    for _ in 0..256 {
        let Some(i) = k.code.get(pc) else {
            return false;
        };
        match &i.op {
            Op::Trap { .. } if i.guard.is_always() => return true,
            Op::CallAbs { syms, .. } if syms.iter().any(|(_, n)| n.contains("__assertfail")) => {
                return i.guard.is_always();
            }
            Op::Bra {
                target,
                any: false,
                tid0_falls: false,
                all: false,
                cond: None,
                div: None,
            } if i.guard.is_always() => pc = *target,
            Op::Bra { .. } | Op::Brx { .. } | Op::Exit | Op::Ret { .. } | Op::Call { .. } => {
                return false;
            }
            _ => pc += 1,
        }
    }
    false
}

/// The predicate of instruction `i` that decides whether it runs: its guard (a branch or a
/// guarded instruction); None for an unguarded instruction and for branches the explorer
/// cannot steer (`BRA.DIV`, `.U`, `NOT_TID0`, a second condition). A select (`selp` / SEL)
/// on a clock-derived predicate is data: its result is a clock-derived value
/// (`clock_stores` when stored).
pub fn choice_pred(i: &ptxv_ir::Inst) -> Option<ptxv_ir::Pred> {
    match &i.op {
        Op::Bra {
            any: false,
            tid0_falls: false,
            cond: None,
            div: None,
            ..
        } => i.guard.reg.map(|_| i.guard),
        Op::Bra { .. } | Op::Brx { .. } => None,
        _ => i.guard.reg.map(|_| i.guard),
    }
}

/// Instructions whose deciding predicate (`choice_pred`) is in `t` at them, other than pure
/// delay loops' back edges and hang guards (a branch with a side that reaches a trap).
fn choices(k: &Kernel, delay: &HashSet<u32>, t: &[TState]) -> HashSet<u32> {
    k.code
        .iter()
        .enumerate()
        .filter_map(|(pc, i)| {
            let p = choice_pred(i)?;
            if !p.reg.is_some_and(|g| {
                t[pc].has(
                    &g,
                    ptxv_ir::Pred {
                        reg: None,
                        neg: false,
                    },
                )
            }) || delay.contains(&(pc as u32))
            {
                return None;
            }
            if let Op::Bra { target, .. } = &i.op {
                if leads_to_trap(k, *target) || leads_to_trap(k, pc + 1) {
                    return None;
                }
            }
            Some(pc as u32)
        })
        .collect()
}

/// Instructions whose guard may derive from the clock (a branch, a guarded instruction),
/// other than pure delay loops' back edges and hang guards. Their
/// outcomes can differ in effects, and the hardware clock decides: the explorer takes both
/// (`Machine::clock_choice_pcs`). ASSUME(E-CLOCK)
pub fn clock_branches(k: &Kernel, delay: &HashSet<u32>) -> HashSet<u32> {
    choices(k, delay, &clock_taint(k))
}

/// Instructions decided by a predicate that may derive from a shared / global memory read.
/// When the kernel stores clock values (`clock_stores`) and a thread has read clock-derived
/// bytes (`Thread::clock_mem`), they are clock choices for it. ASSUME(E-CLOCK)
pub fn mem_branches(k: &Kernel, delay: &HashSet<u32>) -> HashSet<u32> {
    if clock_stores(k).is_empty() {
        return HashSet::new();
    }
    choices(k, delay, &taint(k, true, false))
}

/// Non-arithmetic instructions other than branches with a control operand that may derive
/// from the clock (a clock-dependent address, count or descriptor): not modeled (`unmodeled`
/// when executed). A clock-derived guard is a choice (`clock_branches`).
pub fn clock_effects(k: &Kernel) -> HashSet<u32> {
    let t = clock_taint(k);
    k.code
        .iter()
        .enumerate()
        .filter_map(|(pc, i)| {
            // A branch the explorer cannot steer (`choice_pred` None) under a clock guard.
            if matches!(i.op, Op::Bra { .. } | Op::Brx { .. }) {
                let g = i.guard.reg.is_some_and(|p| {
                    t[pc].has(
                        &p,
                        ptxv_ir::Pred {
                            reg: None,
                            neg: false,
                        },
                    )
                });
                return (g && choice_pred(i).is_none()).then_some(pc as u32);
            }
            if crate::num::pure_op(&i.op) {
                return None;
            }
            let c = crate::num::control_regs(&i.op)
                .iter()
                .any(|r| t[pc].has(r, i.guard));
            c.then_some(pc as u32)
        })
        .collect()
}

/// Instructions that store a clock-derived value to shared or global memory: control in
/// another thread (or later) that loads it back is not followed by `clock_taint` (a verdict
/// note, E-CLOCK).
pub fn clock_stores(k: &Kernel) -> Vec<u32> {
    let t = clock_data_taint(k);
    k.code
        .iter()
        .enumerate()
        .filter(|(pc, i)| {
            !crate::num::pure_op(&i.op)
                && !matches!(
                    i.op,
                    Op::StLocal { .. }
                        | Op::CallAbs { .. }
                        | Op::St {
                            space: ptxv_ir::Space::Param,
                            ..
                        }
                )
                && crate::num::value_regs(&i.op)
                    .iter()
                    .any(|r| t[*pc].has(r, i.guard))
        })
        .map(|(pc, _)| pc as u32)
        .collect()
}

/// What `mark_clock_outputs` needs of a machine: its kernel and clock-store instructions.
pub type ClockMark = (
    std::sync::Arc<Kernel>,
    std::sync::Arc<std::collections::HashSet<u32>>,
);

/// `ClockMark` of `m`.
pub fn clock_mark(m: &crate::machine::Machine) -> ClockMark {
    (m.k.clone(), m.clock_store_pcs.clone())
}

/// ASSUME(E-CLOCK): a kernel that stores clock-derived data (`clock_stores`: a timestamp, a
/// select on a clock comparison) has outputs that depend on timing, but data is computed from
/// the one clock the run has (0, or `PTXV_CLOCK_STEP`'s): the verdict does not claim the
/// outputs for every timing, so it is incomplete, with a named note. Applied where every
/// check of a launch ends (`smid::explore_smid`, `smid::check_placements`).
pub fn mark_clock_outputs(v: &mut crate::explore::Verdict, m: &ClockMark) {
    let Some(&pc) = m.1.iter().min() else {
        return;
    };
    v.complete = false;
    let note = format!(
        "E-CLOCK: clock-dependent output: a clock-derived value is stored at {} (and {} other store(s)); the outputs are checked for one timing only, the verdict is incomplete",
        m.0.describe(pc as usize),
        m.1.len() - 1
    );
    if !v.notes.contains(&note) {
        v.notes.push(note);
    }
}

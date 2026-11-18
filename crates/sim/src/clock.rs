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

/// The instructions after each `CALL.REL` of the code: where a `RET` may return to.
fn return_sites(k: &Kernel) -> Vec<usize> {
    (0..k.code.len())
        .filter(|&pc| matches!(k.code[pc].op, Op::Call { .. }))
        .map(|pc| pc + 1)
        .collect()
}

/// Successors of `pc`. A `RET` returns to every call's return site (`rets`, context
/// insensitive): what a callee computes, a clock value say, flows back to its callers.
fn succs(k: &Kernel, pc: usize, rets: &[usize]) -> Vec<usize> {
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
        Op::Ret { .. } => {
            let mut v = rets.to_vec();
            if i.guard.reg.is_some() {
                v.push(next);
            }
            v
        }
        Op::Exit | Op::Trap { .. } if i.guard.reg.is_none() => vec![],
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
    let rets = return_sites(k);
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
            for s in succs(k, pc, &rets) {
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
fn leads_to_trap(k: &Kernel, pc: usize) -> bool {
    leads_to_trap_past(k, pc, &HashSet::new())
}

/// `leads_to_trap`, falling through the back edges in `delay` (pure delay loops: their
/// deadline passes, `delay_loops`), as DeepEP-style timeouts wait a while after the print and
/// before the trap.
fn leads_to_trap_past(k: &Kernel, mut pc: usize, delay: &HashSet<u32>) -> bool {
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
            Op::Bra { .. } if delay.contains(&(pc as u32)) => pc += 1,
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
            if trap_side(k, pc, delay).is_some() {
                return None;
            }
            Some(pc as u32)
        })
        .collect()
}

/// For a guarded branch: `Some(true)` when its taken side reaches a trap (through straight-line
/// code and pure delay loops), `Some(false)` when its fall-through does, None otherwise.
fn trap_side(k: &Kernel, pc: usize, delay: &HashSet<u32>) -> Option<bool> {
    let Op::Bra { target, .. } = &k.code[pc].op else {
        return None;
    };
    if leads_to_trap_past(k, *target, delay) {
        Some(true)
    } else if leads_to_trap_past(k, pc + 1, delay) {
        Some(false)
    } else {
        None
    }
}

/// Clock units a hang guard's clock must advance by, after its first clock read, before the
/// guard may take its trap side, for the guard to be exempt from the clock choices
/// (`Machine::new`, `hang_bound_holds`): 2^31 cycles is over a second at GB300's clock, 2^31
/// ns of the global timer two. ASSUME(E-CLOCK)
pub const HANG_BOUND: u64 = 1 << 31;

/// What `Machine::new` re-executes to decide whether a hang guard's bound is at least
/// `HANG_BOUND`: the backward slice of a predicate the trap side needs (the guard, or the
/// operand of the guard's AND / OR combine that the trap side forces), through unique
/// reaching definitions (pure register arithmetic, parameter and constant loads, clock
/// reads), in dependency order, and its clock reads in the order they run.
#[derive(Clone, Debug)]
pub struct HangSlice {
    pub pcs: Vec<u32>,
    pub reads: Vec<u32>,
    /// The predicate the trap side needs ...
    pub pred: ptxv_ir::Pred,
    /// ... and its value there.
    pub trap_when: bool,
}

/// Hang guards (a branch on a clock-derived predicate with a side that reaches a trap, other
/// than pure delay loops' back edges), each with the slice that decides its bound, None
/// where no slice can be built (a loaded value, a register with several reaching definitions,
/// clock reads in no dominance order): such a guard is a clock choice. ASSUME(E-CLOCK)
pub fn hang_guards(k: &Kernel, delay: &HashSet<u32>) -> Vec<(u32, Option<HangSlice>)> {
    let t = clock_taint(k);
    let mut out = Vec::new();
    let mut preds: Option<Vec<Vec<usize>>> = None;
    for (pc, i) in k.code.iter().enumerate() {
        let Some(p) = choice_pred(i) else {
            continue;
        };
        let Some(g) = p.reg else {
            continue;
        };
        let clean = ptxv_ir::Pred {
            reg: None,
            neg: false,
        };
        if !t[pc].has(&g, clean) || delay.contains(&(pc as u32)) {
            continue;
        }
        let Some(trap_taken) = trap_side(k, pc, delay) else {
            continue;
        };
        let preds = preds.get_or_insert_with(|| predecessors(k));
        out.push((pc as u32, hang_slice(k, preds, pc, p, trap_taken)));
    }
    out
}

fn predecessors(k: &Kernel) -> Vec<Vec<usize>> {
    let rets = return_sites(k);
    let mut p = vec![Vec::new(); k.code.len()];
    for pc in 0..k.code.len() {
        for s in succs(k, pc, &rets) {
            if k.entry_of(s) == k.entry_of(pc) {
                p[s].push(pc);
            }
        }
    }
    p
}

/// The definitions of `r` that reach `pc` (a guarded definition does not end the search).
/// Empty when a path from the kernel's entry reaches `pc` with no definition (the register's
/// initial value is no definition the slice can re-execute) or the search did not finish.
fn reaching_defs(k: &Kernel, preds: &[Vec<usize>], r: Reg, pc: usize) -> Vec<usize> {
    let entry = k.entry_of(pc);
    if pc == entry {
        return Vec::new();
    }
    let mut seen = HashSet::new();
    let mut work: Vec<usize> = preds[pc].clone();
    let mut defs = Vec::new();
    while let Some(q) = work.pop() {
        if !seen.insert(q) {
            continue;
        }
        if seen.len() > 100_000 {
            // Not searched to the end: no definition is known to be the only one.
            return Vec::new();
        }
        let i = &k.code[q];
        if i.op.dst_regs().contains(&r) {
            defs.push(q);
            if i.guard.is_always() {
                continue;
            }
        }
        if q == entry {
            return Vec::new();
        }
        work.extend(preds[q].iter().copied());
    }
    defs
}

/// The slice of the hang guard at `pc` (predicate `guard`, trap side taken when
/// `trap_taken`): of the guard itself, else of a predicate the trap side forces through the
/// guard's definition (DeepEP: `ISETP.GT.U32.OR P0, PT, R19, 0x1, !P0` over the timeout
/// compare `P0`; the trap side, P0 false, needs the timeout compare true).
fn hang_slice(
    k: &Kernel,
    preds: &[Vec<usize>],
    pc: usize,
    guard: ptxv_ir::Pred,
    trap_taken: bool,
) -> Option<HangSlice> {
    let mut cands = vec![(pc, guard, trap_taken)];
    let (mut at, mut p, mut when) = (pc, guard, trap_taken);
    for _ in 0..4 {
        let Some(r) = p.reg else { break };
        let &[d] = reaching_defs(k, preds, r, at).as_slice() else {
            break;
        };
        let i = &k.code[d];
        let Op::Setp {
            and,
            or,
            xor: false,
            ex: None,
            ..
        } = &i.op
        else {
            break;
        };
        if !i.guard.is_always() || and.reg.is_none() {
            break;
        }
        // The register's value on the trap side (the predicate reads it negated or not).
        let v = when != p.neg;
        // `r = cmp || and` false needs `and` false; `r = cmp && and` true needs `and` true.
        if *or == v {
            break;
        }
        (at, p, when) = (d, *and, v);
        cands.push((at, p, when));
    }
    cands
        .into_iter()
        .find_map(|(at, p, when)| slice_at(k, preds, at, p, when))
}

/// The slice of predicate `p` read at `pc`, whose value `trap_when` the trap side needs.
fn slice_at(
    k: &Kernel,
    preds: &[Vec<usize>],
    pc: usize,
    p: ptxv_ir::Pred,
    trap_when: bool,
) -> Option<HangSlice> {
    let g = p.reg?;
    // (register, use) pairs to resolve; definitions in post-order (operands first).
    let mut order: Vec<usize> = Vec::new();
    let mut reads: Vec<usize> = Vec::new();
    let mut done: HashSet<usize> = HashSet::new();
    #[allow(clippy::too_many_arguments)]
    fn visit(
        k: &Kernel,
        preds: &[Vec<usize>],
        r: Reg,
        at: usize,
        order: &mut Vec<usize>,
        reads: &mut Vec<usize>,
        done: &mut HashSet<usize>,
        depth: usize,
    ) -> Option<()> {
        if depth > 64 || order.len() > 64 {
            return None;
        }
        let d = match reaching_defs(k, preds, r, at).as_slice() {
            [d] => *d,
            _ => return None,
        };
        let i = &k.code[d];
        if !i.guard.is_always() {
            return None;
        }
        if done.contains(&d) {
            return Some(());
        }
        let vals = i.op.src_vals();
        if reads_clock(&i.op) {
            if !matches!(i.op, Op::Mov { .. }) {
                return None;
            }
            done.insert(d);
            order.push(d);
            reads.push(d);
            return Some(());
        }
        let param = matches!(
            i.op,
            Op::Ld {
                space: ptxv_ir::Space::Param,
                ..
            } | Op::LdConst { .. }
        );
        if !(crate::num::pure_op(&i.op) || param) || vals.iter().any(|v| matches!(v, Val::SReg(_)))
        {
            return None;
        }
        for s in i.op.src_regs() {
            visit(k, preds, s, d, order, reads, done, depth + 1)?;
        }
        done.insert(d);
        order.push(d);
        Some(())
    }
    visit(k, preds, g, pc, &mut order, &mut reads, &mut done, 0)?;
    if reads.is_empty() {
        return None;
    }
    // Each slice instruction must stand for one execution: none in a callee (a callee called
    // twice gives two clock values from one read), and every user of one in a loop reads its
    // latest execution (below).
    let rets = return_sites(k);
    let mut callee: HashSet<usize> = HashSet::new();
    let mut work: Vec<usize> = k
        .code
        .iter()
        .filter_map(|i| match i.op {
            Op::Call { target } => Some(target),
            _ => None,
        })
        .collect();
    while let Some(q) = work.pop() {
        if q >= k.code.len() || !callee.insert(q) {
            continue;
        }
        if !matches!(k.code[q].op, Op::Ret { .. }) {
            work.extend(succs(k, q, &rets));
        }
    }
    if order.iter().any(|d| callee.contains(d)) {
        return None;
    }
    let in_cycle = |d: usize| -> bool {
        let mut seen = HashSet::new();
        let mut w = succs(k, d, &rets);
        while let Some(q) = w.pop() {
            if q == d {
                return true;
            }
            if seen.insert(q) {
                w.extend(succs(k, q, &rets));
            }
        }
        false
    };
    // Whether `pc` is reachable from `d` (past it) without passing `u`.
    let avoids = |d: usize, u: usize| -> bool {
        let mut seen = HashSet::new();
        let mut w = succs(k, d, &rets);
        while let Some(q) = w.pop() {
            if q == u || !seen.insert(q) {
                continue;
            }
            if q == pc {
                return true;
            }
            w.extend(succs(k, q, &rets));
        }
        false
    };
    let set: HashSet<usize> = order.iter().copied().collect();
    let mut users: std::collections::HashMap<usize, Vec<usize>> = Default::default();
    for &u in order.iter().chain(std::iter::once(&pc)) {
        for r in k.code[u]
            .op
            .src_regs()
            .into_iter()
            .chain(k.code[u].guard.reg)
        {
            if let [d] = reaching_defs(k, preds, r, u).as_slice() {
                if set.contains(d) {
                    users.entry(*d).or_default().push(u);
                }
            }
        }
    }
    // A definition in a loop: every path from it to `pc` passes each of its users, so after
    // its last execution before `pc` each user runs again and reads that execution (no user
    // keeps an older iteration's value while another reads a newer one).
    for (d, us) in &users {
        if in_cycle(*d) && us.iter().any(|&u| u != pc && avoids(*d, u)) {
            return None;
        }
    }
    // Time order of the clock reads: each must dominate the next.
    let entry = k.entry_of(pc);
    let dominates = |a: usize, b: usize| -> bool {
        // b unreachable from the entry without passing a.
        let rets = return_sites(k);
        let mut seen = HashSet::new();
        let mut work = vec![entry];
        while let Some(q) = work.pop() {
            if q == a || !seen.insert(q) {
                continue;
            }
            if q == b {
                return false;
            }
            for s in succs(k, q, &rets) {
                if k.entry_of(s) == entry {
                    work.push(s);
                }
            }
        }
        true
    };
    let mut sorted: Vec<usize> = Vec::new();
    for r in reads {
        let pos = sorted.iter().position(|&x| dominates(r, x));
        match pos {
            Some(p) => sorted.insert(p, r),
            None => sorted.push(r),
        }
    }
    // In a loop, a read's next execution must not come between the next read and `pc`: every
    // path from it to `pc` passes the next read (their last executions in time order).
    if !sorted
        .windows(2)
        .all(|w| dominates(w[0], w[1]) && !(in_cycle(w[0]) && avoids(w[0], w[1])))
    {
        return None;
    }
    Some(HangSlice {
        pcs: order.into_iter().map(|x| x as u32).collect(),
        reads: sorted.into_iter().map(|x| x as u32).collect(),
        pred: p,
        trap_when,
    })
}

/// Clock choices (`clock_branches`, `mem_branches`) whose deciding predicate may differ
/// between the lanes of one warp split: derived from a lane-varying value (`%tid`, the lane
/// id and masks, a load or other non-arithmetic result, a value written under lane-varying
/// control) or, on the PTX front, from a clock read itself (each thread reads `%clock` on
/// its own; the SASS front reads one value per warp instruction, E-LOCKSTEP). A clock choice
/// fixes one predicate for the whole split, so such a choice with more than one lane is
/// `unmodeled` (`exec`). ASSUME(E-CLOCK)
pub fn lane_varying_choices(k: &Kernel, delay: &HashSet<u32>) -> HashSet<u32> {
    use ptxv_ir::SReg;
    let ptx = k.front == ptxv_ir::Front::Ptx;
    let varying_sreg = |v: &Val| match v {
        Val::SReg(s) => {
            matches!(
                s,
                SReg::Tid(_)
                    | SReg::LaneId
                    | SReg::LaneMaskLt
                    | SReg::LaneMaskLe
                    | SReg::LaneMaskGe
                    | SReg::LaneMaskGt
                    | SReg::LaneMaskEq
            ) || ptx && s.is_clock()
        }
        _ => false,
    };
    let cfg = crate::elect_lanes::Cfg::new(k);
    let mut lv: HashSet<Reg> = HashSet::new();
    let mut ctl = vec![false; k.code.len()];
    loop {
        let before = (lv.len(), ctl.iter().filter(|&&x| x).count());
        for (pc, i) in k.code.iter().enumerate() {
            let uniform_dst = |r: &Reg| matches!(r.file, ptxv_ir::File::UR | ptxv_ir::File::UP);
            let param = matches!(
                i.op,
                Op::Ld {
                    space: ptxv_ir::Space::Param,
                    ..
                } | Op::LdConst { .. }
            );
            let v = ctl[pc]
                || i.guard.reg.is_some_and(|g| lv.contains(&g))
                || i.op.src_regs().iter().any(|r| lv.contains(r))
                || i.op.src_vals().iter().any(varying_sreg)
                || !(crate::num::pure_op(&i.op) || param);
            if v {
                for d in i.op.dst_regs() {
                    if !uniform_dst(&d) {
                        lv.insert(d);
                    }
                }
            }
            // A branch on a lane-varying guard, or an indexed branch on a lane-varying index.
            if matches!(i.op, Op::Bra { .. } | Op::Brx { .. })
                && (i.guard.reg.is_some_and(|g| lv.contains(&g))
                    || matches!(i.op, Op::Brx { .. })
                        && i.op.src_regs().iter().any(|r| lv.contains(r)))
            {
                for x in cfg.region(pc) {
                    ctl[x] = true;
                }
            }
        }
        if (lv.len(), ctl.iter().filter(|&&x| x).count()) == before {
            break;
        }
    }
    let mut all = clock_branches(k, delay);
    all.extend(mem_branches(k, delay));
    all.extend(hang_guards(k, delay).into_iter().map(|x| x.0));
    all.into_iter()
        .filter(|&pc| {
            choice_pred(&k.code[pc as usize])
                .is_some_and(|p| p.reg.is_some_and(|g| lv.contains(&g)))
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
/// from the clock (a clock-dependent address, count or descriptor), branches the explorer
/// cannot steer under a clock guard, and indexed branches with a clock-derived index: not
/// modeled (`unmodeled` when executed). A clock-derived guard is a choice (`clock_branches`).
pub fn clock_effects(k: &Kernel) -> HashSet<u32> {
    let t = clock_taint(k);
    // When the kernel stores clock values, an index loaded from memory may be one (as
    // `mem_branches` for guards).
    let mt = if clock_stores(k).is_empty() {
        None
    } else {
        Some(taint(k, true, false))
    };
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
                // An indexed branch whose index (or target) derives from the clock: the
                // clock picks the target.
                let idx = matches!(i.op, Op::Brx { .. })
                    && i.op.src_regs().iter().any(|r| {
                        t[pc].has(r, i.guard) || mt.as_ref().is_some_and(|m| m[pc].has(r, i.guard))
                    });
                return (g && choice_pred(i).is_none() || idx).then_some(pc as u32);
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

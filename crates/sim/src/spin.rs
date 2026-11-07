/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Static check of ASSUMPTIONS.md E-SPIN: "a failed try_wait blocks until the mbarrier
//! changes". The model never executes the failure path of a `try_wait`; that is exact when the
//! failure path only re-polls the same mbarrier and then rejoins the success path. This module
//! proves that shape for every try_wait in the kernel (both front-ends):
//!
//! ```text
//! w:  d = try_wait [addr], parity
//!     <pure ops: no memory/sync effects, do not read d, do not write addr/parity regs>
//! B:  @!d bra STUB          (or: @d bra C / fall into STUB)
//! C:  ...                   (the continuation the model resumes at)
//! STUB: { nop | yield | d = try_wait [addr], parity | @(!)d bra | bra }*  -- success -> C or w
//! ```
//!
//! The stub may first recompute the parity or the address into dead registers (checked
//! at run time to poll the same phase of the same mbarrier).
//!
//! PTX's `W: try_wait; @!p bra W` is the special case STUB = w. A try_wait that does not have
//! this shape gets an `unmodeled` finding: schedules where the thread works while waiting are
//! not explored, so the verdict is inconclusive.

use std::collections::BTreeSet;

use ptxv_ir::Addr;
use ptxv_ir::FenceKind;
use ptxv_ir::Inst;
use ptxv_ir::Op;
use ptxv_ir::Reg;
use ptxv_ir::Val;

use crate::machine::Finding;
use crate::machine::Machine;

/// Register-only operations: no memory, synchronization, or control effects. Returns the
/// main destination; callers that track writes use `dst_regs` (carry-outs, pairs).
fn pure_dst(op: &Op) -> Option<Option<Reg>> {
    Some(match op {
        Op::Mov { d, .. }
        | Op::Int { d, .. }
        | Op::MulWide { d, .. }
        | Op::Imad { d, .. }
        | Op::Iadd3 { d, .. }
        | Op::CvtInt { d, .. }
        | Op::R2Ur { d, p: None, .. }
        | Op::Flo { d, .. }
        | Op::Lea { d, .. }
        | Op::LeaHi { d, .. }
        | Op::Pack64 { d, .. }
        | Op::I2F { d, .. }
        | Op::F2I { d, .. }
        | Op::Mufu { d, .. }
        | Op::Shf { d, .. }
        | Op::Lop3 { d, .. }
        | Op::Sel { d, .. }
        | Op::Plop3 { d, .. }
        | Op::Setp { d, .. }
        | Op::Fadd { d, .. }
        | Op::Hfma2 { d, .. }
        | Op::Hfma2Bf16 { d, .. }
        | Op::FfmaBf16 { d, .. }
        | Op::Hmnmx2Bf16 { d, .. }
        | Op::Vhmnmx { d, .. }
        | Op::CvtF32Bf16 { d, .. }
        | Op::Prmt { d, .. }
        | Op::Brev { d, .. }
        | Op::AddMinMax { d, .. }
        | Op::IntMnmx3 { d, .. }
        | Op::P2r { d, .. }
        | Op::LdConst { d, .. }
        | Op::Ffma { d, .. }
        | Op::Fmnmx { d, .. }
        | Op::Fsetp { d, .. }
        | Op::Hsetp2Bf16 { d, .. }
        | Op::Fset { d, .. }
        | Op::Fchk { d, .. }
        | Op::Fsel { d, .. }
        | Op::F2fpBf16 { d, .. }
        | Op::Cvta { d, .. } => Some(*d),
        // A register spill reload (`LDL.LU R0, [R1+0xa0]` between CUTLASS's TRYWAIT and its
        // retry branch): thread-private memory, which no op the recognizer accepts writes (a
        // spill store is not pure), so every iteration reloads the same value. ASSUME(E-SPIN)
        Op::LdLocal { d, .. } => Some(*d),
        Op::Nop => None,
        _ => return None,
    })
}

/// Operations the model executes as no-ops for control purposes: `NOP`/`NANOSLEEP`, and the
/// acquire half of a fence (`CCTL.IVALL`), which changes no register or control flow.
fn model_nop(op: &Op) -> bool {
    matches!(
        op,
        Op::Nop
            | Op::Fence {
                kind: FenceKind::Acquire
            }
    )
}

fn reads_pred(op: &Op, d: Reg) -> bool {
    op.src_regs().contains(&d)
}

fn operand_regs(addr: &Addr, parity: &Val) -> Vec<Reg> {
    let mut v = Vec::new();
    for x in [&addr.base, &addr.base2]
        .into_iter()
        .flatten()
        .chain(std::iter::once(parity))
    {
        if let Val::Reg(r, n) = x {
            for i in 0..(*n).max(1) as u16 {
                v.push(Reg {
                    file: r.file,
                    idx: r.idx + i,
                });
            }
        }
    }
    v
}

/// The pc of the branch (or `@d EXIT`) that decides the retry of the try_wait at `w`: the
/// straight-line code before it is pure and keeps the poll operands (`retry_shape` step 1),
/// so a check of the retry stub may run it first (ptxas computes the stub's address register
/// there: FA4's `try_wait [UR10+0xc0]; ...; UR6 = UR10 + 0xc0; @!P0 bra stub` with the stub
/// `R2 = UR6; try_wait [R2+URZ]`).
pub fn retry_branch(code: &[Inst], w: usize) -> Option<usize> {
    let Op::MbarTryWait { d, .. } = &code.get(w)?.op else {
        return None;
    };
    for (pc, i) in code.iter().enumerate().skip(w + 1) {
        match &i.op {
            Op::Bra { .. } if i.guard.reg == Some(*d) => return Some(pc),
            Op::Exit if i.guard.reg == Some(*d) => return Some(pc),
            Op::Bra { .. } | Op::Exit => return None,
            op if model_nop(op) || pure_dst(op).is_some() => {}
            Op::MbarTryWait { .. } if i.guard.reg == Some(*d) && i.guard.neg => {}
            _ => return None,
        }
    }
    None
}

/// Why try_wait at `w` violates the retry shape (Err), or Ok(None) if it has it. Ok(Some((s,
/// w2))): it has it once the parity that the recomputing stub `s..w2` passes to the try_wait
/// at `w2` equals the original; the executor checks that on every successful poll at `w`.
pub fn retry_shape(m: &Machine, w: usize) -> Result<Option<(usize, usize)>, String> {
    let code = &m.k.code;
    let Op::MbarTryWait {
        d,
        addr,
        parity,
        bit31,
    } = &code[w].op
    else {
        return Ok(None);
    };
    let poll_regs = operand_regs(addr, parity);
    let same_poll = |op: &Op| matches!(op, Op::MbarTryWait { d: d2, addr: a2, parity: p2, bit31: b2 } if d2 == d && a2 == addr && p2 == parity && b2 == bit31);
    // 1. Up to the deciding branch: pure, does not read d, keeps the poll operands.
    let mut pc = w + 1;
    let (stub, cont) = loop {
        let Some(i) = code.get(pc) else {
            return Err("falls off the end of the code".into());
        };
        if let Op::Bra {
            target,
            any: false,
            tid0_falls: false,
            cond: None,
            div: None,
        } = i.op
        {
            if i.guard.reg == Some(*d) {
                // `@!d bra T`: T is the failure path; `@d bra T`: T is the continuation.
                break if i.guard.neg {
                    (target, pc + 1)
                } else {
                    (pc + 1, target)
                };
            }
        }
        // `@d EXIT` (CUTLASS's "wait, then leave"): the exit is the continuation, the next
        // instruction the failure path.
        if matches!(i.op, Op::Exit) && i.guard.reg == Some(*d) && !i.guard.neg {
            break (pc + 1, pc);
        }
        // A no-op in the model (e.g. the `@P0 CCTL.IVALL` of a cluster-scope acquire) cannot
        // change what the retry does, whatever its guard.
        if model_nop(&i.op) {
            pc += 1;
            continue;
        }
        // `@!d` re-poll of the same mbarrier (SASS: `@!P0 SYNCS.PHASECHK.TRANS64 P0, ...` after
        // the suspending TRYWAIT): runs only while the wait has failed, and is the same poll.
        if same_poll(&i.op) && i.guard.reg == Some(*d) && i.guard.neg {
            pc += 1;
            continue;
        }
        if i.guard.reg == Some(*d) || reads_pred(&i.op, *d) {
            return Err(format!(
                "{} reads the try_wait result before the retry branch",
                m.k.describe(pc)
            ));
        }
        match pure_dst(&i.op) {
            Some(_) => {
                if i.op
                    .dst_regs()
                    .iter()
                    .any(|r| poll_regs.contains(r) || r == d)
                {
                    return Err(format!(
                        "{} overwrites an operand of the poll",
                        m.k.describe(pc)
                    ));
                }
            }
            None => {
                return Err(format!(
                    "{} runs between the try_wait and its retry branch",
                    m.k.describe(pc)
                ));
            }
        }
        pc += 1;
    };
    // 2. The failure path. State: what d can be (0 = false, 1 = unknown after a re-poll,
    //    2 = true). Only no-ops, branches and re-polls; with d true the walk must reach the
    //    continuation (or the original try_wait) through no-ops and unconditional branches.
    // ptxas may recompute the parity into a free register at the head of the stub
    // (`R4 = UR6 << 31; w2: try_wait [addr], R4`): straight-line pure ops, then a try_wait of
    // the same mbarrier. The recomputed registers must be dead at the continuation, and the
    // parity value equal (checked when the poll runs, `Machine::stubs`).
    //
    // The stub may also start with unconditional branches, and it may loop back to the
    // straight-line prefix of `w` itself (`H: R2 = UR7; w: try_wait [R2+0x40]; ...; @!d bra
    // H`): w2 = w. Then the prefix recomputes the poll operands in place, and the executor
    // checks on every successful poll that rerunning it changes no register (it is what the
    // thread ran before its first poll), so skipping it is exact.
    let mut clob: BTreeSet<Reg> = BTreeSet::new();
    let mut s = stub;
    let mut q = stub;
    let mut w2 = None;
    let mut hops = 0;
    while let Some(i) = code.get(q) {
        if q == w {
            if !clob.is_empty() {
                w2 = Some(w);
            }
            break;
        }
        if !i.guard.is_always() {
            break;
        }
        if let Op::Bra {
            target,
            any: false,
            tid0_falls: false,
            cond: None,
            div: None,
        } = i.op
        {
            if clob.is_empty() && hops < 8 {
                hops += 1;
                (q, s) = (target, target);
                continue;
            }
        }
        match pure_dst(&i.op) {
            Some(dst) if !model_nop(&i.op) => {
                if dst.is_some_and(|r| r == *d) {
                    break;
                }
                clob.extend(i.op.dst_regs());
                q += 1;
            }
            _ => {
                if let Op::MbarTryWait {
                    d: d2, bit31: b2, ..
                } = &i.op
                {
                    // The stub may also poll through a recomputed address register
                    // (`R0 = UR9; try_wait [R0+0x30]` for `try_wait [UR9+0x30]`, or the
                    // same text after rewriting R0), or through a register computed before
                    // the wait (FA4: `try_wait [UR9+0x8]`, stub `try_wait [R2+URZ+0x8]`):
                    // the executor checks that it names the same mbarrier, with the parity.
                    if d2 == d && b2 == bit31 && (!clob.is_empty() || i.op != code[w].op) {
                        w2 = Some(q);
                    }
                }
                break;
            }
        }
    }
    let poll2 = w2.map(|q| code[q].op.clone());
    let same_poll = |op: &Op| same_poll(op) || poll2.as_ref() == Some(op);
    let mut seen = BTreeSet::new();
    let mut work = vec![(w2.unwrap_or(stub), 0u8)];
    while let Some((pc, dv)) = work.pop() {
        if pc == w || (pc == cont && dv == 2) {
            continue;
        }
        if !seen.insert((pc, dv)) {
            continue;
        }
        let Some(i) = code.get(pc) else {
            return Err("retry path falls off the end of the code".into());
        };
        if pc == cont {
            return Err(format!(
                "retry path reaches the continuation {} without a successful poll",
                m.k.describe(pc)
            ));
        }
        match &i.op {
            op if model_nop(op) => work.push((pc + 1, dv)),
            // `@!d` re-poll: skipped when d is true, otherwise a poll (d becomes unknown).
            op if same_poll(op) && i.guard.reg == Some(*d) && i.guard.neg => {
                work.push((pc + 1, if dv == 2 { 2 } else { 1 }))
            }
            _ if !i.guard.is_always() && !matches!(i.op, Op::Bra { .. }) => {
                return Err(format!(
                    "retry path executes a predicated instruction {}",
                    m.k.describe(pc)
                ));
            }
            op if same_poll(op) && dv != 2 => work.push((pc + 1, 1)),
            Op::Bra {
                target,
                any: false,
                tid0_falls: false,
                cond: None,
                div: None,
            } if i.guard.is_always() => work.push((*target, dv)),
            Op::Bra {
                target,
                any: false,
                tid0_falls: false,
                cond: None,
                div: None,
            } if i.guard.reg == Some(*d) && dv != 2 => {
                // Taken when d == !neg.
                let (on_true, on_false) = if i.guard.neg {
                    (pc + 1, *target)
                } else {
                    (*target, pc + 1)
                };
                work.push((on_false, 0));
                if dv == 1 {
                    work.push((on_true, 2));
                }
            }
            _ => return Err(format!("retry path executes {}", m.k.describe(pc))),
        }
    }
    let Some(w2) = w2 else {
        return Ok(None);
    };
    // Looping back to w: the recomputation must be idempotent (checked at run time) instead.
    if w2 == w {
        return Ok(Some((s, w2)));
    }
    if let Some(r) = live_after(code, cont, &clob) {
        return Err(format!(
            "the retry path recomputes {r:?}, which is read after the continuation {}",
            m.k.describe(cont)
        ));
    }
    Ok(Some((s, w2)))
}

/// A register of `regs` that some path from `from` reads before overwriting it, if any.
/// Conservative: paths into calls (other than to a trap) and returns count as reads. A write
/// under a guard kills the register for later reads under the same guard (`@P4 R4 = ..;
/// @P4 use R4`) until the guard's predicate is rewritten.
pub(crate) fn live_after(code: &[Inst], from: usize, regs: &BTreeSet<Reg>) -> Option<Reg> {
    type Cond = BTreeSet<(Reg, Reg, bool)>;
    let mut seen = BTreeSet::new();
    let mut work: Vec<(usize, BTreeSet<Reg>, Cond)> = vec![(from, regs.clone(), Cond::new())];
    while let Some((pc, mut live, mut cond)) = work.pop() {
        if live.is_empty() || !seen.insert((pc, live.clone(), cond.clone())) {
            continue;
        }
        let Some(i) = code.get(pc) else {
            return live.into_iter().next();
        };
        let mut reads = i.op.src_regs();
        reads.extend(i.guard.reg);
        let killed = |r: &Reg| {
            i.guard
                .reg
                .is_some_and(|g| cond.contains(&(*r, g, i.guard.neg)))
        };
        if let Some(r) = reads.into_iter().find(|r| live.contains(r) && !killed(r)) {
            return Some(r);
        }
        let dst = i.op.dst_regs();
        match i.guard.reg {
            None if !i.guard.neg => {
                for r in &dst {
                    live.remove(r);
                }
            }
            Some(g) => {
                for r in dst.iter().filter(|r| live.contains(r)) {
                    cond.insert((*r, g, i.guard.neg));
                }
            }
            None => {}
        }
        cond.retain(|(_, g, _)| !dst.contains(g));
        match &i.op {
            Op::Exit | Op::Trap { .. } if i.guard.is_always() => {}
            Op::Call { target }
                if matches!(code.get(*target).map(|x| &x.op), Some(Op::Trap { .. })) =>
            {
                if !i.guard.is_always() {
                    work.push((pc + 1, live, cond));
                }
            }
            Op::Call { .. } | Op::Ret { .. } | Op::Brx { .. } => return live.into_iter().next(),
            Op::Bra { target, .. } => {
                work.push((*target, live.clone(), cond.clone()));
                if !i.guard.is_always() {
                    work.push((pc + 1, live, cond));
                }
            }
            _ => work.push((pc + 1, live, cond)),
        }
    }
    None
}

/// Is the try_wait at `w` (without the retry shape) a peek: CUTLASS's "test now, work, then
/// wait" idiom? A successful test is progress (the phase completed), so only failures can
/// spin. It is a peek when, assuming it and every other non-blocking try_wait fail, every
/// path from it back to itself passes a try_wait with the retry shape (which blocks until
/// the mbarrier moves). The walk tracks the failed results: predicates known from them (the
/// result itself, a `mov.pred d, -1`, an integer compare of a known register with an
/// immediate) and registers known from them (trtllm-gen keeps the result as a token:
/// `SEL R4, RZ, 0x1, !P1`, later `ISETP.NE.AND P1, PT, R4, 0x1` decides whether to wait). A
/// branch on a known predicate follows that value. Running a peek as a non-blocking test is
/// then exact and cannot spin without progress. Calls (other than to a trap routine) and
/// returns end the walk with "no".
pub fn is_peek(m: &Machine, w: usize) -> bool {
    use std::collections::BTreeMap;
    let code = &m.k.code;
    let Op::MbarTryWait { d, .. } = &code[w].op else {
        return false;
    };
    if retry_shape(m, w).is_ok() {
        return false;
    }
    // Known predicates and registers along a path.
    type Env = (BTreeMap<Reg, bool>, BTreeMap<Reg, u64>);
    let konst = |e: &Env, v: &Val| -> Option<u64> {
        match v {
            Val::Imm(x) => Some(*x),
            Val::Reg(r, _) if r.idx == 255 => Some(0),
            Val::Reg(r, _) => e.1.get(r).copied(),
            _ => None,
        }
    };
    let pval = |e: &Env, p: &ptxv_ir::Pred| -> Option<bool> {
        match p.reg {
            None => Some(!p.neg),
            Some(r) => e.0.get(&r).map(|&b| b != p.neg),
        }
    };
    let mut seen = BTreeSet::new();
    let start: Env = ([(*d, false)].into(), BTreeMap::new());
    let mut work = vec![(w + 1, start)];
    while let Some((pc, env)) = work.pop() {
        if pc == w {
            return false;
        }
        if !seen.insert((pc, env.clone())) {
            continue;
        }
        let Some(i) = code.get(pc) else {
            return false;
        };
        let g = i.guard;
        // Does an instruction guarded by `g` run (None: unknown)?
        let runs = pval(&env, &g);
        match &i.op {
            Op::MbarTryWait { .. } if g.is_always() && retry_shape(m, pc).is_ok() => continue,
            // Another peek: assume it fails too.
            Op::MbarTryWait { d: d2, .. } => {
                let mut nv = env.clone();
                if runs == Some(false) {
                } else if runs == Some(true) {
                    nv.0.insert(*d2, false);
                } else {
                    nv.0.remove(d2);
                }
                work.push((pc + 1, nv));
            }
            Op::Exit | Op::Trap { .. } if g.is_always() => continue,
            // ptxas's guardrail checks call a trap routine: that path ends.
            Op::Call { target }
                if matches!(code.get(*target).map(|x| &x.op), Some(Op::Trap { .. })) =>
            {
                if !g.is_always() {
                    work.push((pc + 1, env));
                }
            }
            Op::Call { .. } | Op::Ret { .. } | Op::Brx { .. } => return false,
            Op::Bra { target, .. } => match runs {
                Some(true) => work.push((*target, env)),
                Some(false) => work.push((pc + 1, env)),
                None => {
                    work.push((*target, env.clone()));
                    work.push((pc + 1, env));
                }
            },
            op => {
                let mut nv = env.clone();
                // The known value written, if the instruction runs.
                let known: Option<(Reg, Option<u64>)> = match op {
                    // A constant predicate (`mov.pred d, -1` / `0`).
                    Op::Plop3 {
                        d: q, a, b, c, lut, ..
                    } => {
                        let bit = [a, b, c]
                            .iter()
                            .map(|p| pval(&env, p))
                            .collect::<Option<Vec<_>>>();
                        Some((
                            *q,
                            bit.map(|v| {
                                let idx = ((v[0] as u8) << 2) | ((v[1] as u8) << 1) | v[2] as u8;
                                (lut >> idx) as u64 & 1
                            }),
                        ))
                    }
                    Op::Sel { d: r, a, b, p } => Some((
                        *r,
                        pval(&env, p).and_then(|c| konst(&env, if c { a } else { b })),
                    )),
                    Op::Setp {
                        d: q,
                        cmp,
                        a,
                        b,
                        and,
                        or: false,
                        xor: false,
                        ex: None,
                        ..
                    } if matches!(cmp, ptxv_ir::Cmp::Eq | ptxv_ir::Cmp::Ne) => Some((
                        *q,
                        konst(&env, a).zip(konst(&env, b)).zip(pval(&env, and)).map(
                            |((x, y), c)| {
                                let eq = x as u32 == y as u32;
                                ((if *cmp == ptxv_ir::Cmp::Eq { eq } else { !eq }) && c) as u64
                            },
                        ),
                    )),
                    _ => None,
                };
                for r in op.dst_regs() {
                    nv.0.remove(&r);
                    nv.1.remove(&r);
                }
                if let (Some((r, Some(v))), Some(true)) = (known, runs) {
                    if r.file == ptxv_ir::File::P || r.file == ptxv_ir::File::UP {
                        nv.0.insert(r, v == 1);
                    } else {
                        nv.1.insert(r, v);
                    }
                }
                // An instruction that may not run leaves its destinations unknown (removed);
                // one that does not run keeps them.
                if runs == Some(false) {
                    nv = env;
                }
                work.push((pc + 1, nv));
            }
        }
    }
    true
}

/// A strong load's pc and its poll loop (`poll_loops`): Ok((H, b)) or why it is not one.
pub type PollLoop = (usize, Result<(usize, usize), String>);

/// Poll loops (E-SPIN for flags): a load at `l` inside
///
/// ```text
/// H:  <pure ops>
/// l:  r = ld.strong [a]
///     <pure ops / acquire fences>
/// b:  @p bra H              (H <= l; p computed in the loop)
/// ```
///
/// with no other memory, synchronization or control operation, and no register carried from
/// one iteration to the next (every register the loop reads it wrote earlier in the same
/// iteration, or never writes). A failed iteration then changes nothing that a later one does
/// not overwrite, so the model runs a thread at `l` only when the iteration exits the loop.
/// Returns, per strong load inside a backward branch's range, Ok((H, b)) for this shape or the
/// reason it fails.
///
/// Plain loads count too, but only with the shape (Ok): ptxas compiles `ld.acquire.cta` /
/// `ld.relaxed.cta` / `ld.volatile` of shared memory to a plain `LDS`, so a flag spin is a
/// plain-load loop in SASS. Without this, its iterations are invisible steps that never end a
/// transition: the thread runs the loop alone until the local-step limit, and a racy poll
/// reports a race per iteration. Blocking keeps every verdict class: the body is
/// register-only, so a failed iteration only re-reads the flag, and the race detector sees
/// that read once (`blocked_poll_read`). A plain loop with another shape (a reduction over
/// loaded values, a carried counter) is not a poll and is not listed.
/// Registers an instruction reads, guard and predicate operands included.
fn inst_reads(inst: &Inst) -> Vec<Reg> {
    let mut reads = inst.op.src_regs();
    reads.extend(inst.guard.reg);
    if let Op::Plop3 { a, b, c, .. } = &inst.op {
        reads.extend([a.reg, b.reg, c.reg].into_iter().flatten());
    }
    if let Op::Setp { and, .. } = &inst.op {
        reads.extend(and.reg);
    }
    reads
}

/// Loop h..=b (back branch at b) around the load at l: does the guard of any of its exits
/// (EXIT, or a branch to a target outside h..=b) depend on the loaded value? Registers only
/// (as `poll_loops`' own test). Forward dataflow over the loop's control flow (fall-through,
/// branches inside h..=b, the back edge b -> h) to a fixpoint: an instruction that reads a
/// tainted register (guard included), and the load itself, taint what they write; an
/// unguarded write that reads no tainted register clears its destinations; a guarded one
/// that reads none changes nothing (it may not run).
fn loop_exit_tainted(code: &[Inst], l: usize, h: usize, b: usize) -> bool {
    use std::collections::HashSet;
    let n = b + 1 - h;
    let mut inn: Vec<HashSet<Reg>> = vec![HashSet::new(); n];
    let mut work: Vec<usize> = (h..=b).collect();
    let mut queued = vec![true; n];
    let succ = |p: usize| -> Vec<usize> {
        let inst = &code[p];
        let mut v = Vec::new();
        let uncond = inst.guard.is_always();
        match &inst.op {
            Op::Bra { target, .. } => {
                if (h..=b).contains(target) {
                    v.push(*target);
                }
                if !uncond && p < b {
                    v.push(p + 1);
                }
            }
            Op::Exit if uncond => {}
            _ if p < b => v.push(p + 1),
            _ => {}
        }
        v
    };
    while let Some(p) = work.pop() {
        queued[p - h] = false;
        let inst = &code[p];
        let mut t = inn[p - h].clone();
        let reads = inst_reads(inst).iter().any(|r| t.contains(r));
        if p == l || reads {
            t.extend(inst.op.dst_regs());
        } else if inst.guard.is_always() {
            for d in inst.op.dst_regs() {
                t.remove(&d);
            }
        }
        for q in succ(p) {
            let before = inn[q - h].len();
            inn[q - h].extend(t.iter().copied());
            if inn[q - h].len() != before && !queued[q - h] {
                queued[q - h] = true;
                work.push(q);
            }
        }
    }
    (h..=b).any(|p| {
        let inst = &code[p];
        let exit = match &inst.op {
            Op::Exit => true,
            Op::Bra { target, .. } => p != b && (*target < h || *target > b),
            _ => false,
        };
        exit && inst_reads(inst).iter().any(|r| inn[p - h].contains(r))
    })
}

pub fn poll_loops(code: &[Inst], front: ptxv_ir::Front) -> Vec<PollLoop> {
    let mut out = Vec::new();
    for (l, i) in code.iter().enumerate() {
        // A compare-and-swap poll (torch symmetric memory's signals: `while (cas(a, 0, 1) !=
        // 0);`): a failed compare only reads (exec.rs `AtomOp::Cas`), so it is a poll like a
        // strong load whose exiting iteration also writes.
        let cas = matches!(
            i.op,
            Op::Atom {
                op: ptxv_ir::AtomOp::Cas,
                d: Some(_),
                ..
            }
        );
        let strong = match &i.op {
            // SASS LDS is strong (S-SMEM-STRONG) but, like a plain load, a poll only with the
            // shape.
            Op::Ld { sync, space, .. } => {
                sync.is_some()
                    && !(front == ptxv_ir::Front::Sass && *space == ptxv_ir::Space::Shared)
            }
            _ if cas => true,
            _ => continue,
        };
        // The first branch after l decides.
        let Some(b) = (l + 1..code.len()).find(|&p| matches!(code[p].op, Op::Bra { .. })) else {
            continue;
        };
        let Op::Bra { target: h, any, .. } = code[b].op else {
            unreachable!()
        };
        if h > l {
            continue; // not a loop around this load
        }
        // A loop whose exit does not depend on the loaded value is not a poll (DeepEP's
        // copy loops read peer data with `ld.volatile`): it terminates on its own.
        if let Some(g) = code[b].guard.reg {
            let mut taint: std::collections::HashSet<Reg> = match &i.op {
                Op::Ld { d, n, .. } => (0..(*n).max(1) as u16)
                    .map(|k| Reg {
                        file: d.file,
                        idx: d.idx + k,
                    })
                    .collect(),
                Op::Atom { d: Some(d), .. } => [*d].into(),
                _ => Default::default(),
            };
            for p in (l + 1..b).chain(h..b) {
                let inst = &code[p];
                let mut reads = inst.op.src_regs();
                reads.extend(inst.guard.reg);
                if let Op::Plop3 { a, b, c, .. } = &inst.op {
                    reads.extend([a.reg, b.reg, c.reg].into_iter().flatten());
                }
                if let Op::Setp { and, .. } = &inst.op {
                    reads.extend(and.reg);
                }
                if p == l || reads.iter().any(|r| taint.contains(r)) {
                    taint.extend(inst.op.dst_regs());
                } else if inst.guard.is_always() {
                    // Overwritten by a value that does not depend on the load.
                    for d in inst.op.dst_regs() {
                        taint.remove(&d);
                    }
                }
            }
            if !taint.contains(&g) {
                continue;
            }
        }
        // An unconditional back branch: the loop leaves through its exits (EXIT, a branch out
        // of h..=b). When no exit's guard depends on the loaded value (a pipelined main loop
        // that reads a `volatile` word each iteration and stores what it read), it is not a
        // poll either. The taint only grows here, lap after lap to a fixpoint, so a value
        // carried around the loop any number of times is followed.
        if code[b].guard.is_always() && !any && !loop_exit_tainted(code, l, h, b) {
            continue;
        }
        let shape = (|| {
            if any || code[b].guard.is_always() {
                return Err("the back branch is unconditional or warp-uniform".to_string());
            }
            if let Some(p) = (h..b).find(|&p| matches!(code[p].op, Op::Bra { .. })) {
                return Err(format!("the loop has another branch at pc {p}"));
            }
            let mut writes = std::collections::HashSet::new();
            for (p, inst) in (h..b).zip(&code[h..b]) {
                if p == l {
                    if let Op::Ld { d, n, .. } = &inst.op {
                        for k in 0..(*n).max(1) as u16 {
                            writes.insert(Reg {
                                file: d.file,
                                idx: d.idx + k,
                            });
                        }
                    }
                    if let Op::Atom { d: Some(d), .. } = &inst.op {
                        writes.insert(*d);
                    }
                    continue;
                }
                match pure_dst(&inst.op) {
                    Some(_) => writes.extend(inst.op.dst_regs()),
                    None if model_nop(&inst.op) => {}
                    // The release fence of a CAS signal (`MEMBAR.SYS` before each attempt):
                    // repeating it after failed attempts, which wrote nothing, publishes
                    // nothing new.
                    None if cas && matches!(inst.op, Op::Fence { .. }) => {}
                    None => return Err(format!("pc {p} is not a pure operation")),
                }
            }
            // Loop-invariant prefix writes (`R6 = 0` for the CAS compare operand at the loop
            // head): recomputed to the same value every iteration, so not carried.
            let mut inv = std::collections::HashSet::new();
            for inst in &code[h..l] {
                if pure_dst(&inst.op).is_some_and(|d| d.is_some()) {
                    let mut reads = inst.op.src_regs();
                    reads.extend(inst.guard.reg);
                    if reads.iter().all(|r| !writes.contains(r) || inv.contains(r)) {
                        inv.extend(inst.op.dst_regs());
                    }
                }
            }
            // Iteration order from l: l..b, then h..l. A read of a loop-written register
            // before this iteration wrote it is carried from the previous iteration.
            let mut done = std::collections::HashSet::new();
            for p in (l..=b).chain(h..l) {
                let inst = &code[p];
                let mut reads = inst.op.src_regs();
                reads.extend(inst.guard.reg);
                if let Op::Plop3 { a, b, c, .. } = &inst.op {
                    reads.extend([a.reg, b.reg, c.reg].into_iter().flatten());
                }
                if let Op::Setp { and, .. } = &inst.op {
                    reads.extend(and.reg);
                }
                if let Some(r) = reads
                    .iter()
                    .find(|r| writes.contains(r) && !done.contains(*r) && !inv.contains(*r))
                {
                    return Err(format!(
                        "register {r:?} is carried from one iteration to the next (pc {p})"
                    ));
                }
                if p == l {
                    if let Op::Ld { d, n, .. } = &inst.op {
                        for k in 0..(*n).max(1) as u16 {
                            done.insert(Reg {
                                file: d.file,
                                idx: d.idx + k,
                            });
                        }
                    }
                    if let Op::Atom { d: Some(d), .. } = &inst.op {
                        done.insert(*d);
                    }
                } else if pure_dst(&inst.op).is_some() {
                    done.extend(inst.op.dst_regs());
                }
            }
            // A prefix write (h..l) that is not loop-invariant and that l..b does not overwrite
            // leaves the loop with a value computed from a failed iteration (`H: mov r7, r5;
            // l: ld r5; ...` with r7 read after the loop: the previous load's value). The
            // model runs the prefix once, before the first iteration, so such a register must
            // be dead after the loop.
            let mut stale: BTreeSet<Reg> = BTreeSet::new();
            for inst in &code[h..l] {
                if pure_dst(&inst.op).is_some() {
                    stale.extend(inst.op.dst_regs().into_iter().filter(|r| !inv.contains(r)));
                }
            }
            for inst in &code[l..=b] {
                if inst.guard.is_always() {
                    for r in inst.op.dst_regs() {
                        stale.remove(&r);
                    }
                    if let Op::Ld { d, n, .. } = &inst.op {
                        for k in 0..(*n).max(1) as u16 {
                            stale.remove(&Reg {
                                file: d.file,
                                idx: d.idx + if d.file == ptxv_ir::File::V { 0 } else { k },
                            });
                        }
                    }
                }
            }
            if let Some(r) = live_after(code, b + 1, &stale) {
                return Err(format!(
                    "register {r:?} is carried from one iteration to the next: the loop head writes it from the previous iteration's values and it is read after the loop"
                ));
            }
            if !code[b].guard.reg.is_some_and(|r| writes.contains(&r)) {
                return Err("the branch condition does not depend on the loop".into());
            }
            Ok((h, b))
        })();
        // A compare-and-swap update loop (ptxas's CAS loop for an atomic without a native form,
        // e.g. `atom.add.noftz.bf16`) is not a poll: the model runs each iteration as a step,
        // and that terminates, since a failed attempt reads a value another thread wrote after
        // this thread's previous read (each failure consumes one of finitely many writes).
        if shape.is_err() && cas && cas_update_loop(code, l, h, b) {
            continue;
        }
        if strong || shape.is_ok() {
            out.push((l, shape));
        }
    }
    out
}

/// `H: <pure ops>; l: d = cas [a], cmp, new; <pure ops>; @p bra H` where the only register
/// carried from one iteration to the next is `cmp`, rewritten from `d` (a move) before the
/// branch: an atomic read-modify-write by compare-and-swap.
fn cas_update_loop(code: &[Inst], l: usize, h: usize, b: usize) -> bool {
    let Op::Atom {
        d: Some(d),
        v: Val::Reg(cmp, _),
        ..
    } = &code[l].op
    else {
        return false;
    };
    let (d, cmp) = (*d, *cmp);
    let pure = (h..b).all(|p| p == l || pure_dst(&code[p].op).is_some() || model_nop(&code[p].op));
    let refresh = (l + 1..b).any(|p| {
        code[p].guard.is_always()
            && code[p].op.dst_regs() == [cmp]
            && code[p].op.src_regs().iter().all(|r| *r == d)
            && !code[p].op.src_regs().is_empty()
    });
    if !pure || !refresh {
        return false;
    }
    let mut writes = std::collections::HashSet::new();
    for inst in &code[h..b] {
        writes.extend(inst.op.dst_regs());
    }
    let mut done = std::collections::HashSet::new();
    for inst in &code[h..=b] {
        let mut reads = inst.op.src_regs();
        reads.extend(inst.guard.reg);
        if reads
            .iter()
            .any(|r| writes.contains(r) && !done.contains(r) && *r != cmp)
        {
            return false;
        }
        done.extend(inst.op.dst_regs());
    }
    true
}

/// A split that arrives at a poll load (`Machine::polls`) that would fail stops there without
/// running it (E-SPIN). It still reads the flag once, for the race detector only: the read is
/// recorded in the shadow memory (and checked against earlier writes) without changing any
/// register. Every later write of the flag by another thread while the split spins is then
/// checked against it, as against the iterations the model does not run: the spinning thread
/// synchronizes with nobody, so such a write races with the poll unless both are morally
/// strong (S-SCOPE). Without this, a write that does not end the spin (and races with it)
/// went unreported. The read does not acquire: dropping that edge can only add races.
pub fn blocked_poll_read(m: &mut Machine, members: &[usize], pc: u32) {
    let k = m.k.clone();
    let inst = &k.code[pc as usize];
    let (space, bytes, addr, sc) = match &inst.op {
        Op::Ld {
            space,
            bytes,
            addr,
            sync,
            ..
        } => (
            space,
            bytes,
            addr,
            sync.map_or(0, |s| crate::machine::strong_code(s.scope)),
        ),
        Op::Atom {
            space,
            bytes,
            addr,
            sync,
            ..
        } => (space, bytes, addr, crate::machine::strong_code(sync.scope)),
        _ => return,
    };
    for &t in members {
        if !m.prd(t, inst.guard) {
            continue;
        }
        let Ok(loc) = m
            .addr(t, addr)
            .and_then(|a| m.resolve(t, *space, a, *bytes as usize))
        else {
            continue; // the real step reports it
        };
        // ASSUME(S-INIT): a stale flag is reported when the loaded value is used, not here.
        let saved = m.stale.replace(None);
        let vc = m.threads[t].vc.clone();
        m.access_x(t as u32, &vc, pc, loc, *bytes as usize, None, sc);
        m.stale = saved;
    }
}

/// A strong global or shared load (a poll the E-POLL-ANY arrival read applies to).
fn strong_poll(op: &Op) -> bool {
    matches!(
        op,
        Op::Ld {
            sync: Some(_),
            space: ptxv_ir::Space::Global | ptxv_ir::Space::Shared,
            ..
        }
    )
}

/// After a step of `warp` by threads `moved`: those now at a poll that would fail read the
/// flag once (`blocked_poll_read`). The split's enabledness is judged with every thread of
/// the warp at that pc, as the explorer will.
pub fn after_step(m: &mut Machine, warp: u32, moved: &[usize]) {
    let mut at: std::collections::BTreeMap<u32, Vec<usize>> = Default::default();
    for &t in moved {
        let th = &m.threads[t];
        if th.st == crate::machine::St::Run && m.polls.contains_key(&th.pc) {
            at.entry(th.pc).or_default().push(t);
        }
    }
    for (pc, ts) in at {
        let split: Vec<usize> = m
            .warp_threads(warp)
            .filter(|&u| m.threads[u].pc == pc && m.threads[u].st == crate::machine::St::Run)
            .collect();
        // E-POLL-ANY (explorer): a strong poll's flag read at arrival goes to `vipr`, without
        // a footprint, whether the poll would fail or not, so that the arriving step commutes
        // with the writes of the flag (the poll's own enabledness carries that dependency,
        // `blocked_sync`). Later writes are checked against it as against the skipped
        // iterations.
        if m.vip.is_some() && strong_poll(&m.k.code[pc as usize].op) {
            let saved = std::mem::replace(&mut m.vip_read, 1);
            blocked_poll_read(m, &ts, pc);
            m.vip_read = saved;
        } else if !m.split_enabled(&split, pc) {
            blocked_poll_read(m, &ts, pc);
        }
    }
    let heads: Vec<(u32, u32)> = moved
        .iter()
        .filter_map(|&t| {
            let th = &m.threads[t];
            (th.st == crate::machine::St::Run)
                .then(|| m.wpolls.get(&th.pc).map(|&b| (th.pc, b)))
                .flatten()
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    for (pc, b) in heads {
        let split: Vec<usize> = m
            .warp_threads(warp)
            .filter(|&u| m.threads[u].pc == pc && m.threads[u].st == crate::machine::St::Run)
            .collect();
        if let Some(reads) = wpoll_dry(m, &split, pc, b) {
            blocked_wpoll_read(m, &reads);
        }
    }
}

pub fn check(m: &Machine) -> Vec<Finding> {
    let mut out = Vec::new();
    for (l, shape) in poll_loops(&m.k.code, m.k.front) {
        let warp_level = m
            .wpolls
            .iter()
            .any(|(&h, &b)| h as usize <= l && l < b as usize);
        if let (Err(why), false) = (shape, warp_level) {
            out.push(Finding {
                kind: "unmodeled".into(),
                sig: format!("poll:{l}"),
                msg: format!(
                    "E-SPIN does not hold for the poll loop around {}: {why}. The model runs every iteration of it as a separate step, which may not terminate",
                    m.k.describe(l)
                ),
                trace: vec![],
                ..Default::default()
            });
        }
    }
    // A try_wait without the retry shape is run as a non-blocking test (its exact semantics)
    // when it is a peek (`is_peek`); any other is blocked on, and inconclusive.
    for (w, i) in m.k.code.iter().enumerate() {
        if matches!(i.op, Op::MbarTryWait { .. }) && !is_peek(m, w) {
            if let Err(why) = retry_shape(m, w) {
                out.push(Finding {
                    kind: "unmodeled".into(),
                    sig: format!("spin:{w}"),
                    msg: format!(
                        "E-SPIN does not hold for the try_wait at {}: {why}. The model blocks instead of running the failure path, so schedules where the thread works while waiting are not explored",
                        m.k.describe(w)
                    ),
                    trace: vec![],
                    ..Default::default()
                });
            }
        }
    }
    out
}

/// Warp-level poll loops (ASSUME(E-SPIN)): DeepEP's `barrier_block` and channel waits,
///
/// ```text
/// H:  ...                                   (pure ops, loads, convergence)
///     @p bra SKIP; ld.strong.sys v, [flag]  (some lanes poll)
///     BSYNC; BRA.DIV; VOTE.ALL q, ...       (the warp agrees)
///     @q bra EXIT                           (forward, out of the loop)
///     <clock timeout check>                 (E-CLOCK: never fires)
/// b:  @r bra H
/// ```
///
/// The body may branch forward, reconverge and vote, but writes no memory: an iteration
/// that brings every member back to H with the registers and predicates it started with is
/// a stutter step, so the split at H blocks until another thread's write changes an
/// iteration (`wpoll_dry`). Returns (H, b) for loops with a strong load that the
/// straight-line shape (`poll_loops`) does not already cover.
pub fn wpoll_loops(code: &[Inst], front: ptxv_ir::Front) -> Vec<(usize, usize)> {
    let straight: BTreeSet<(usize, usize)> = poll_loops(code, front)
        .into_iter()
        .filter_map(|(_, r)| r.ok())
        .collect();
    let mut out = Vec::new();
    for (b, i) in code.iter().enumerate() {
        let Op::Bra { target: h, .. } = i.op else {
            continue;
        };
        if h >= b || straight.contains(&(h, b)) {
            continue;
        }
        // An unconditional back branch: the loop must exit from inside (EXIT or a forward
        // branch past b; DeepEP combine's coordinator: `while (true) { if (done) return; ...}`).
        if i.guard.is_always()
            && !(h..b).any(|p| match &code[p].op {
                Op::Exit => true,
                // Not a `BRA.DIV`: its slow path is the same loop body, out of line.
                Op::Bra {
                    target, div: None, ..
                } => *target > b,
                _ => false,
            })
        {
            continue;
        }
        let mut strong = false;
        let ok = (h..b).all(|p| match &code[p].op {
            // Strong loads, and SASS shared loads (strong, S-SMEM-STRONG: DeepEP combine's
            // warps poll the coordinator's `ld.volatile.shared` channel tails).
            Op::Ld { sync, space, .. } => {
                strong |= sync.is_some()
                    || (front == ptxv_ir::Front::Sass && *space == ptxv_ir::Space::Shared);
                true
            }
            Op::Bra { target, .. } => *target > p,
            // Progress loops (DeepEP combine's receiver coordinator) republish their state
            // every iteration: a stutter iteration rewrites the same bytes (checked at run
            // time, `wpoll_dry`).
            Op::St { .. } | Op::St256 { .. } | Op::Exit | Op::Fence { .. } | Op::Shfl { .. } => {
                true
            }
            Op::VoteAll { .. }
            | Op::VoteAny { .. }
            | Op::MatchAny { .. }
            | Op::Bssy { .. }
            | Op::Bsync { .. }
            | Op::WarpSync { .. }
            | Op::EndCollective { .. } => true,
            op => pure_dst(op).is_some() || model_nop(op),
        });
        if ok && strong {
            out.push((h, b));
        }
    }
    out
}

/// A read of a blocked poll: thread, pc, location, bytes, strength code.
pub type PollRead = (usize, u32, crate::exec::Loc, usize, u8);

/// Runs one iteration of the warp-level poll loop (H = `h`, back branch `b`) for the split
/// `members` on a copy. `Some(reads)` if it is a stutter step (every member back at H with
/// unchanged registers and predicates, and every store rewrote the bytes already there),
/// with the loads it made; `None` if it would exit, change a register or memory, park, or
/// fail (the real step runs and reports). ASSUME(E-SPIN): a same-value rewrite is not
/// recorded as an access while the loop is blocked (its first, real iteration is).
pub fn wpoll_dry(m0: &Machine, members: &[usize], h: u32, b: u32) -> Option<Vec<PollRead>> {
    use crate::machine::St;
    let mut m = m0.clone();
    m.fp = None;
    m.num = None;
    let k = m.k.clone();
    let mut done = vec![false; members.len()];
    let mut reads = Vec::new();
    let mut stores = Vec::new();
    for _ in 0..4096 {
        let run: Vec<usize> = (0..members.len())
            .filter(|&i| !done[i] && m.threads[members[i]].st == St::Run)
            .collect();
        let Some(pc) = run.iter().map(|&i| m.threads[members[i]].pc).min() else {
            break;
        };
        if pc < h || pc > b {
            return None;
        }
        let group: Vec<usize> = run
            .iter()
            .map(|&i| members[i])
            .filter(|&t| m.threads[t].pc == pc)
            .collect();
        let inst = &k.code[pc as usize];
        if let Op::Ld {
            space,
            bytes,
            addr,
            sync,
            ..
        } = &inst.op
        {
            for &t in &group {
                if !m.prd(t, inst.guard) {
                    continue;
                }
                if let Ok(loc) = m
                    .addr(t, addr)
                    .and_then(|a| m.resolve(t, *space, a, *bytes as usize))
                {
                    let sc = sync.map_or(0, |s| crate::machine::strong_code(s.scope));
                    reads.push((t, pc, loc, *bytes as usize, sc));
                }
            }
        }
        if let Op::St {
            space, bytes, addr, ..
        } = &inst.op
        {
            for &t in &group {
                if !m.prd(t, inst.guard) {
                    continue;
                }
                match m
                    .addr(t, addr)
                    .and_then(|a| m.resolve(t, *space, a, *bytes as usize))
                {
                    Ok(loc) => stores.push((loc, *bytes as usize)),
                    Err(_) => return None,
                }
            }
        }
        if m.exec(&group, pc).is_err() {
            return None;
        }
        for &t in &group {
            let th = &m.threads[t];
            if th.st == St::Exited {
                return None;
            }
            if pc == b && th.pc == h {
                done[members.iter().position(|&u| u == t).unwrap()] = true;
            } else if th.st == St::Run && (th.pc <= pc || th.pc > b) {
                return None;
            }
        }
    }
    if done.iter().any(|d| !d) {
        return None;
    }
    let same_mem = stores.iter().all(|&(loc, n)| {
        let byte = |mm: &Machine, i: usize| -> Option<u8> {
            match loc {
                crate::exec::Loc::Shared { cta, off } => {
                    Some(*mm.smem[cta as usize].bytes.get(off as usize + i))
                }
                crate::exec::Loc::Global { alloc, off } => {
                    Some(*mm.gmem[alloc].bytes.get(off as usize + i))
                }
                _ => None,
            }
        };
        (0..n).all(|i| byte(&m, i).is_some() && byte(&m, i) == byte(m0, i))
    });
    (same_mem
        && members.iter().all(|&t| {
            m.threads[t].regs == m0.threads[t].regs
                && m.threads[t].preds == m0.threads[t].preds
                && m.threads[t].pdef == m0.threads[t].pdef
        }))
    .then_some(reads)
}

/// The race-detector reads of a blocked warp-level poll (as `blocked_poll_read`).
pub fn blocked_wpoll_read(m: &mut Machine, reads: &[PollRead]) {
    for &(t, pc, loc, bytes, sc) in reads {
        let saved = m.stale.replace(None);
        let vc = m.threads[t].vc.clone();
        m.access_x(t as u32, &vc, pc, loc, bytes, None, sc);
        m.stale = saved;
    }
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Compare-and-swap retry loops: the lock-free read-modify-write ptxas emits for an atomic
//! without a native form (`atom.add.noftz.f16` / `.bf16` on sm_100 / sm_103: `LD.E` of the
//! containing word, then `HADD2` + `PRMT`, `ATOM.E.CAS`, `Rcmp = Rold`, `@P BRA`), and the
//! same loop written by hand in PTX. E-SPIN (sim `spin.rs`) does not treat one as a poll, and
//! the SASS front-end models the load that seeds its first compare as a relaxed load
//! (`seed_load`). Both recognizers are exact: every condition below is needed for the
//! argument, and a loop missing any of them stays a poll / a plain load.

use crate::Addr;
use crate::AtomOp;
use crate::Cmp;
use crate::Inst;
use crate::Op;
use crate::Reg;
use crate::Space;
use crate::Val;

/// A recognized CAS retry loop: `head..=branch` is the loop, `cas` its compare-and-swap,
/// `cmp` the compare register (carried from one attempt to the next), `old` the register the
/// CAS returns the word's value in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CasRetry {
    pub head: usize,
    pub cas: usize,
    pub branch: usize,
    pub cmp: Reg,
    pub old: Reg,
}

/// Register-only operations with no memory, synchronization or control effect, as they appear
/// between the attempts of a CAS loop (no local or constant loads, no MUFU, no `.uni` checks:
/// a narrower set than E-SPIN's `pure_dst`).
fn pure(op: &Op) -> bool {
    matches!(
        op,
        Op::Mov { .. }
            | Op::Int { .. }
            | Op::MulWide { .. }
            | Op::Imad { .. }
            | Op::Iadd3 { .. }
            | Op::CvtInt { .. }
            | Op::Flo { .. }
            | Op::Lea { .. }
            | Op::LeaHi { .. }
            | Op::Shf { .. }
            | Op::Lop3 { .. }
            | Op::Sel { .. }
            | Op::Plop3 { .. }
            | Op::Setp { .. }
            | Op::Fadd { .. }
            | Op::Ffma { .. }
            | Op::Hfma2 { .. }
            | Op::Hfma2Bf16 { .. }
            | Op::CvtF32Bf16 { .. }
            | Op::F2fpBf16 { .. }
            | Op::Prmt { .. }
            | Op::Nop
    )
}

/// Every register an instruction reads: its operands, its guard, and the predicate inputs
/// `src_regs` leaves out (a `Setp`'s combining predicate, `Plop3`'s inputs).
fn reads(inst: &Inst) -> Vec<Reg> {
    let mut r = inst.op.src_regs();
    r.extend(inst.guard.reg);
    match &inst.op {
        Op::Setp { and, ex, .. } => {
            r.extend(and.reg);
            r.extend(ex.and_then(|p| p.reg));
        }
        Op::Plop3 { a, b, c, .. } => r.extend([a.reg, b.reg, c.reg].into_iter().flatten()),
        _ => {}
    }
    r
}

fn val_regs(v: &Val) -> Vec<Reg> {
    match v {
        Val::Reg(r, n) => (0..(*n).max(1) as u16)
            .map(|i| Reg {
                file: r.file,
                idx: r.idx + i,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn addr_regs(a: &Addr) -> Vec<Reg> {
    [&a.base, &a.base2]
        .into_iter()
        .flatten()
        .flat_map(val_regs)
        .collect()
}

/// `d = s`, a plain 32-bit copy: `MOV d, s` / `mov.b32 d, s`, or `IMAD.MOV.U32 d, RZ, RZ, s`
/// (`d = 0 * 0 + s`). Anything that changes the value (an add, a mask) is not a copy.
fn copy_src(op: &Op) -> Option<(Reg, Reg)> {
    match op {
        Op::Mov {
            d,
            a: Val::Reg(s, n),
        } if *n <= 1 => Some((*d, *s)),
        Op::Imad {
            d,
            a,
            b,
            c: Val::Reg(s, n),
            neg_c: false,
            not_c: false,
            wide: false,
            hi: false,
            cin: None,
            cout: None,
            ..
        } if *n <= 1 && (matches!(a, Val::Imm(0)) || matches!(b, Val::Imm(0))) => Some((*d, *s)),
        _ => None,
    }
}

fn is_reg(v: &Val, r: Reg) -> bool {
    matches!(v, Val::Reg(x, n) if *x == r && *n <= 1)
}

/// The CAS retry loop whose compare-and-swap is at `l`, if it is one exactly:
/// - `l`: a 32-bit `Cas` returning `old`, compare `cmp` (one register; the new value its own
///   register or `cmp + 1`), unconditional, `old` not `cmp` nor an address register;
/// - the first branch after `l` is an unguarded-by-uniformity back branch `@p BRA head` with
///   `head <= l` (not `.U` / `.ANY`, no `NOT_TID0`), and `head..branch` is one block of
///   unconditional register-only operations besides `l` (`pure`);
/// - the next compare is the returned old value: the one write of `cmp` in the loop is a copy
///   `cmp = old` after `l`, and `old` is not written again after `l`;
/// - only the CAS's success decides the exit: the one write of `p` in the loop is
///   `p = (old != cmp)` (or `==` with the branch on `!p`), after `l` and before the copy, so
///   the loop repeats exactly when the compare failed;
/// - the address is loop-invariant: no register of it is written in the loop;
/// - nothing else is carried: every other register the loop reads that the loop writes is
///   written earlier in the same attempt.
///
/// Termination on every schedule: an attempt fails only if the word differs from `cmp`, and
/// from the second attempt on `cmp` is the value the previous attempt's CAS read; so each
/// failure means another thread wrote the word between two of this thread's CASes. Those
/// writes are finitely many (each thread performs finitely many writes in a run that
/// terminates; a thread that writes the word forever is a hang of its own), so the loop exits
/// after finitely many attempts. The first attempt may also fail on a stale seed (one
/// attempt). The loop's effect is one atomic read-modify-write: the successful CAS writes
/// `f(old)` where `old` is the value it read atomically.
pub fn cas_retry_loop(code: &[Inst], l: usize) -> Option<CasRetry> {
    let Op::Atom {
        op: AtomOp::Cas,
        bytes: 4,
        d: Some(old),
        v: Val::Reg(cmp, n),
        new,
        addr,
        ..
    } = &code.get(l)?.op
    else {
        return None;
    };
    let (old, cmp) = (*old, *cmp);
    let pair = new.is_none();
    if (pair && *n != 2) || (!pair && *n != 1) || !code[l].guard.is_always() || old == cmp {
        return None;
    }
    let aregs = addr_regs(addr);
    if aregs.contains(&old) {
        return None;
    }
    let b = (l + 1..code.len()).find(|&p| matches!(code[p].op, Op::Bra { .. }))?;
    // A plain per-thread branch only: `@P BRA.U` (`all`) takes a failed lane out when a
    // warp-mate succeeded, `@P BRA Q, L` (`cond`) when Q is false, and `BRA.DIV` (`div`)
    // sends a lane whose CAS succeeded round again; none keeps "only the CAS's success
    // exits" (MISS 174 / 175).
    let Op::Bra {
        target: h,
        any: false,
        tid0_falls: false,
        all: false,
        cond: None,
        div: None,
    } = &code[b].op
    else {
        return None;
    };
    let h = *h;
    let p = code[b].guard.reg?;
    if h > l || !(h..b).all(|q| q == l || (code[q].guard.is_always() && pure(&code[q].op))) {
        return None;
    }
    // The compare register: written once in the loop, by `cmp = old` after the CAS.
    let cmp_writes: Vec<usize> = (h..=b)
        .filter(|&q| code[q].op.dst_regs().contains(&cmp))
        .collect();
    let [c] = cmp_writes[..] else {
        return None;
    };
    if c <= l || copy_src(&code[c].op) != Some((cmp, old)) || code[c].op.dst_regs() != [cmp] {
        return None;
    }
    if (l + 1..=b).any(|q| code[q].op.dst_regs().contains(&old)) {
        return None;
    }
    // The exit: p = (old != cmp) read before the copy, written nowhere else in the loop.
    let p_writes: Vec<usize> = (h..=b)
        .filter(|&q| code[q].op.dst_regs().contains(&p))
        .collect();
    let [s] = p_writes[..] else {
        return None;
    };
    let Op::Setp {
        d,
        cmp: cc,
        bits: 32,
        a,
        b: bb,
        and,
        or: false,
        xor: false,
        ex: None,
    } = &code[s].op
    else {
        return None;
    };
    let operands = (is_reg(a, old) && is_reg(bb, cmp)) || (is_reg(a, cmp) && is_reg(bb, old));
    let repeats_on_failure = match cc {
        Cmp::Ne => !code[b].guard.neg,
        Cmp::Eq => code[b].guard.neg,
        _ => false,
    };
    if *d != p || s <= l || s >= c || !and.is_always() || !operands || !repeats_on_failure {
        return None;
    }
    // Loop-invariant address.
    if (h..=b).any(|q| q != l && code[q].op.dst_regs().iter().any(|r| aregs.contains(r))) {
        return None;
    }
    // Nothing but `cmp` is carried from one attempt to the next.
    let written: std::collections::HashSet<Reg> =
        (h..=b).flat_map(|q| code[q].op.dst_regs()).collect();
    let mut done = std::collections::HashSet::new();
    for inst in &code[h..=b] {
        if reads(inst)
            .iter()
            .any(|r| *r != cmp && written.contains(r) && !done.contains(r))
        {
            return None;
        }
        done.extend(inst.op.dst_regs());
    }
    Some(CasRetry {
        head: h,
        cas: l,
        branch: b,
        cmp,
        old,
    })
}

/// Every CAS retry loop of `code` (`cas_retry_loop`).
pub fn cas_retry_loops(code: &[Inst]) -> Vec<CasRetry> {
    (0..code.len())
        .filter_map(|l| cas_retry_loop(code, l))
        .collect()
}

/// The pc of the plain load that seeds the first compare of the CAS retry loop `cr`, when its
/// value can only reach that compare (MISS 175): ptxas's `LD.E Rcmp, [Ra]` of the word ahead of
/// the loop of an emulated 16-bit atomic. Exactly:
/// - the nearest write of `cmp` before the loop head is an unconditional plain (weak, coherent)
///   4-byte load of `cmp` in the CAS's state space from the CAS's address (same base
///   registers and offset; the CAS is 4 bytes, so both access the same aligned word);
/// - between the load and the loop head only unconditional register-only operations (and
///   `BSSY`), none of which reads `cmp` (the value goes nowhere else) or writes an address
///   register;
/// - no control transfer enters the code after the load except the loop's own back branch
///   (no branch, `brx` target or call return lands in it).
///
/// Then the loaded value only becomes the first attempt's compare, and inside the loop only
/// the CAS's success exits (`cas_retry_loop`): a stale, torn or garbage value makes the first
/// attempt fail and be retried with the word the CAS returned, so the loop's effect is the
/// same atomic read-modify-write for any value the load could return. Modelling the load as
/// a relaxed load of the CAS's scope therefore drops no outcome of the kernel; its conflicts
/// with plain writes are still checked at the CAS, which follows it with nothing that
/// synchronizes in between.
pub fn seed_load(code: &[Inst], cr: &CasRetry) -> Option<usize> {
    seed_at(code, cr, false)
}

/// The seed load of `cr` after the SASS front-end made it relaxed (`seed_load`'s shape with
/// the CAS's relaxed scope on a load printed without a strength): for the `assumes:` list.
pub fn seeded_load(code: &[Inst], cr: &CasRetry) -> Option<usize> {
    seed_at(code, cr, true)
}

fn seed_at(code: &[Inst], cr: &CasRetry, synced: bool) -> Option<usize> {
    let Op::Atom {
        addr: caddr,
        space: cspace,
        sync: csync,
        ..
    } = &code[cr.cas].op
    else {
        return None;
    };
    let want = synced.then_some(crate::MemSync {
        sem: crate::Sem::Relaxed,
        scope: csync.scope,
    });
    let aregs = addr_regs(caddr);
    let s = (0..cr.head)
        .rev()
        .find(|&q| code[q].op.dst_regs().contains(&cr.cmp))?;
    let Op::Ld {
        d,
        n,
        space,
        bytes: 4,
        addr,
        sync,
        nc: false,
        ..
    } = &code[s].op
    else {
        return None;
    };
    if *sync != want || (synced && code[s].text.contains("STRONG")) {
        return None;
    }
    if *d != cr.cmp
        || *n > 1
        || space != cspace
        || !matches!(space, Space::Global | Space::Generic)
        || !code[s].guard.is_always()
        || addr.base != caddr.base
        || addr.base2 != caddr.base2
        || addr.off != caddr.off
    {
        return None;
    }
    // A `BSSY` (it records the warp's convergence mask: no data, no control transfer) may sit
    // in between too (ptxas 13.0's `ld.global` + `atom.global.cas` loop, fixtures/e_cas_retry).
    if !(s + 1..cr.head).all(|q| {
        code[q].guard.is_always()
            && (pure(&code[q].op) || matches!(code[q].op, Op::Bssy { .. }))
            && !reads(&code[q]).contains(&cr.cmp)
            && !code[q].op.dst_regs().iter().any(|r| aregs.contains(r))
    }) {
        return None;
    }
    let inside = |t: usize| t > s && t <= cr.head;
    for (q, inst) in code.iter().enumerate() {
        let enters = match &inst.op {
            Op::Bra { target, .. } => inside(*target) && !(q == cr.branch && *target == cr.head),
            Op::Brx { targets, .. } => targets.iter().any(|&t| inside(t)),
            // A call returns to the instruction after it; `CALL.ABS` calls only `__assertfail`
            // (modelled as a trap, `Op::CallAbs`), never into the kernel's code.
            Op::Call { .. } | Op::CallAbs { .. } => inside(q + 1),
            _ => false,
        };
        if enters {
            return None;
        }
    }
    Some(s)
}

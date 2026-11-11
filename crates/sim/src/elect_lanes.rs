/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! ASSUME(E-ELECT): control flow shared by the leader-sensitivity taint
//! (`explore::elect_sensitivity`) and lane classes for the leader search (`lane_classes`).
//!
//! Two lanes of a warp are clones when no instruction tells them apart: at every instruction
//! both execute it or neither does, and every operand of an instruction with an effect
//! (memory, synchronization, a collective, control) has equal values in the two lanes, as a
//! function of what each lane received from memory and collectives. Clones of every electing
//! warp can then trade places: swapping their thread identities maps each execution with one
//! as the leader to an execution with the other as the leader, with the same outputs and the
//! same findings up to the two threads' names. So the leader search needs one lane per class
//! among the lanes nothing has told apart yet (`Machine::elect_lanes`).

use std::collections::HashMap;
use std::collections::HashSet;

use ptxv_ir::Cmp;
use ptxv_ir::IntOp;
use ptxv_ir::Kernel;
use ptxv_ir::Op;
use ptxv_ir::Pred;
use ptxv_ir::Reg;
use ptxv_ir::SReg;
use ptxv_ir::ShflMode;
use ptxv_ir::Space;
use ptxv_ir::Val;

/// Control flow of a kernel: successors, predecessors, immediate post-dominators and
/// immediate dominators. Node `n` (the code length) is the exit; `usize::MAX`: a node that
/// cannot reach the exit (post-dominators) or is unreachable from the entry (dominators).
pub(crate) struct Cfg {
    pub succ: Vec<Vec<usize>>,
    pub preds: Vec<Vec<usize>>,
    pub ipdom: Vec<usize>,
    pub idom: Vec<usize>,
}

/// Cooper, Harvey and Kennedy's iterative dominators of `succ` from `root`.
fn dominators(succ: &[Vec<usize>], preds: &[Vec<usize>], root: usize) -> Vec<usize> {
    let n = succ.len();
    let mut order = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    let mut stack = vec![(root, 0usize)];
    seen[root] = true;
    while let Some(&mut (v, ref mut i)) = stack.last_mut() {
        if *i < succ[v].len() {
            let s = succ[v][*i];
            *i += 1;
            if !seen[s] {
                seen[s] = true;
                stack.push((s, 0));
            }
        } else {
            order.push(v);
            stack.pop();
        }
    }
    let mut rank = vec![usize::MAX; n];
    for (r, &v) in order.iter().enumerate() {
        rank[v] = r;
    }
    let mut dom = vec![usize::MAX; n];
    dom[root] = root;
    let mut changed = true;
    while changed {
        changed = false;
        for &v in order.iter().rev() {
            if v == root {
                continue;
            }
            let mut new = usize::MAX;
            for &p in &preds[v] {
                if dom[p] == usize::MAX {
                    continue;
                }
                new = if new == usize::MAX {
                    p
                } else {
                    let (mut a, mut b) = (new, p);
                    while a != b {
                        while rank[a] < rank[b] {
                            a = dom[a];
                        }
                        while rank[b] < rank[a] {
                            b = dom[b];
                        }
                    }
                    a
                };
            }
            if new != usize::MAX && dom[v] != new {
                dom[v] = new;
                changed = true;
            }
        }
    }
    dom
}

impl Cfg {
    pub fn new(k: &Kernel) -> Cfg {
        let n = k.code.len();
        let mut succ: Vec<Vec<usize>> = (0..n)
            .map(|pc| {
                let i = &k.code[pc];
                let guarded = i.guard.reg.is_some();
                match &i.op {
                    Op::Bra { target, .. } if guarded => vec![pc + 1, *target],
                    Op::Bra { target, .. } => vec![*target],
                    // A guarded exit, return or trap falls through for the other lanes.
                    Op::Exit | Op::Ret { .. } | Op::Trap { .. } if guarded => vec![pc + 1, n],
                    Op::Exit | Op::Ret { .. } | Op::Trap { .. } => vec![n],
                    _ => vec![pc + 1],
                }
                .into_iter()
                .map(|s| s.min(n))
                .collect()
            })
            .collect();
        succ.push(Vec::new());
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
        for (v, ss) in succ.iter().enumerate() {
            for &s in ss {
                preds[s].push(v);
            }
        }
        // Post-dominators: dominators of the reverse graph from the exit.
        let ipdom = dominators(&preds, &succ, n);
        let idom = if n > 0 {
            dominators(&succ, &preds, 0)
        } else {
            vec![usize::MAX]
        };
        Cfg {
            succ,
            preds,
            ipdom,
            idom,
        }
    }

    /// The instructions reachable from `from` without passing `stop`.
    pub fn reach(&self, from: usize, stop: usize) -> Vec<usize> {
        let n = self.succ.len() - 1;
        let mut out = Vec::new();
        let mut vis = vec![false; n + 1];
        let mut work = vec![from];
        while let Some(v) = work.pop() {
            if v >= n || v == stop || vis[v] {
                continue;
            }
            vis[v] = true;
            out.push(v);
            work.extend(self.succ[v].iter().copied());
        }
        out
    }

    /// Control region of the branch at `pc`: what runs after it before its immediate
    /// post-dominator.
    pub fn region(&self, pc: usize) -> Vec<usize> {
        let stop = self.ipdom[pc];
        let mut out: Vec<usize> = self.succ[pc]
            .iter()
            .flat_map(|&s| self.reach(s, stop))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Does `a` dominate `b`?
    fn dominates(&self, a: usize, mut b: usize) -> bool {
        loop {
            if a == b {
                return true;
            }
            let d = self.idom[b];
            if d == usize::MAX || d == b {
                return false;
            }
            b = d;
        }
    }

    /// Does every path from the entry to `x` take the edge `from` -> `to`?
    fn edge_dominates(&self, from: usize, to: usize, x: usize) -> bool {
        self.preds[to].len() == 1 && self.preds[to][0] == from && self.dominates(to, x)
    }
}

/// Partition of the 32 lanes: class of each lane, canonical (the lowest lane of the class).
pub type Part = [u8; 32];

const ONE: Part = [0; 32];

fn singletons() -> Part {
    std::array::from_fn(|l| l as u8)
}

fn canon<K: PartialEq + Copy>(keys: &[K; 32]) -> Part {
    std::array::from_fn(|l| (0..=l).find(|&j| keys[j] == keys[l]).unwrap() as u8)
}

fn meet(a: &Part, b: &Part) -> Part {
    let keys: [(u8, u8); 32] = std::array::from_fn(|l| (a[l], b[l]));
    canon(&keys)
}

fn mask_part(m: u32) -> Part {
    let keys: [bool; 32] = std::array::from_fn(|l| m >> l & 1 == 1);
    canon(&keys)
}

/// `p` among the lanes of `m`; the others form one class.
fn restrict(p: &Part, m: u32) -> Part {
    let keys: [(bool, u8); 32] = std::array::from_fn(|l| {
        if m >> l & 1 == 1 {
            (true, p[l])
        } else {
            (false, 0)
        }
    });
    canon(&keys)
}

/// A register's value across the lanes of a warp: known per lane, or only which lanes hold
/// equal values (as functions of what they received).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(clippy::large_enum_variant)] // short-lived, copied per instruction
enum Av {
    Lanes([u64; 32]),
    Part(Part),
}

impl Av {
    fn part(&self) -> Part {
        match self {
            Av::Lanes(v) => canon(v),
            Av::Part(p) => *p,
        }
    }
}

/// Which lanes may execute an instruction (`known`) and which lanes share that status
/// (`part`; lanes in one class all execute it or none does).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Ex {
    known: u32,
    part: Part,
}

impl Ex {
    const ALL: Ex = Ex {
        known: !0,
        part: ONE,
    };
    fn part(&self) -> Part {
        meet(&self.part, &mask_part(self.known))
    }
}

fn trunc(x: u64, bits: u8) -> u64 {
    if bits >= 64 {
        x
    } else {
        x & ((1u64 << bits) - 1)
    }
}

fn sext(x: u64, bits: u8) -> i64 {
    if bits >= 64 {
        x as i64
    } else {
        let s = 64 - bits as u32;
        ((x << s) as i64) >> s
    }
}

fn int_op(op: IntOp, bits: u8, a: u64, b: u64) -> Option<u64> {
    let w = bits.clamp(1, 64) as u64;
    let r = match op {
        IntOp::Add => a.wrapping_add(b),
        IntOp::Sub => a.wrapping_sub(b),
        IntOp::Mul => a.wrapping_mul(b),
        IntOp::And => a & b,
        IntOp::Or => a | b,
        IntOp::Xor => a ^ b,
        IntOp::Shl => {
            let s = trunc(b, 32);
            if s >= w { 0 } else { a << s }
        }
        IntOp::ShrU => {
            let s = trunc(b, 32);
            if s >= w { 0 } else { trunc(a, bits) >> s }
        }
        IntOp::ShrS => {
            let s = trunc(b, 32).min(w - 1);
            (sext(a, bits) >> s) as u64
        }
        IntOp::MinU => trunc(a, bits).min(trunc(b, bits)),
        IntOp::MaxU => trunc(a, bits).max(trunc(b, bits)),
        IntOp::MinS => sext(a, bits).min(sext(b, bits)) as u64,
        IntOp::MaxS => sext(a, bits).max(sext(b, bits)) as u64,
        _ => return None,
    };
    Some(trunc(r, bits))
}

fn cmp(c: Cmp, bits: u8, a: u64, b: u64) -> bool {
    let (ua, ub) = (trunc(a, bits), trunc(b, bits));
    let (sa, sb) = (sext(a, bits), sext(b, bits));
    match c {
        Cmp::Eq => ua == ub,
        Cmp::Ne => ua != ub,
        Cmp::LtU => ua < ub,
        Cmp::LeU => ua <= ub,
        Cmp::GtU => ua > ub,
        Cmp::GeU => ua >= ub,
        Cmp::LtS => sa < sb,
        Cmp::LeS => sa <= sb,
        Cmp::GtS => sa > sb,
        Cmp::GeS => sa >= sb,
    }
}

/// Register computations without an effect and without lane semantics: their result is a
/// function of their operands (the class of the result: the meet of the operands').
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
            | Op::Brev { .. }
            | Op::AddMinMax { .. }
            | Op::IntMnmx3 { .. }
            | Op::P2r { .. }
            | Op::Lea { .. }
            | Op::Shf { .. }
            | Op::Lop3 { .. }
            | Op::Plop3 { .. }
            | Op::Setp { .. }
            | Op::Fadd { .. }
            | Op::Ffma { .. }
            | Op::FmulScale { .. }
            | Op::Fmnmx { .. }
            | Op::Fsetp { .. }
            | Op::Hsetp2Bf16 { .. }
            | Op::Hset2 { .. }
            | Op::Fset { .. }
            | Op::Fsel { .. }
            | Op::Fchk { .. }
            | Op::R2p { .. }
            | Op::F2fpBf16 { .. }
            | Op::Ffma2 { .. }
            | Op::Hfma2 { .. }
            | Op::Hfma2Bf16 { .. }
            | Op::FfmaBf16 { .. }
            | Op::Hmnmx2Bf16 { .. }
            | Op::Qmul4 { .. }
            | Op::Hmnmx2F16 { .. }
            | Op::Vhmnmx { .. }
            | Op::CvtPack { .. }
            | Op::CvtRsX4 { .. }
            | Op::CvtUnpack { .. }
            | Op::CvtS2f6x2 { .. }
            | Op::CvtF32Bf16 { .. }
            | Op::Cvta { .. }
            | Op::Bfe { .. }
            | Op::Bfi { .. }
            | Op::LeaHi { .. }
            | Op::I2F { .. }
            | Op::F2I { .. }
            | Op::Pack64 { .. }
            | Op::Unpack64 { .. }
            | Op::Mufu { .. }
            | Op::Sel { .. }
            | Op::Prmt { .. }
            | Op::Dp2a { .. }
            | Op::Dfma { .. }
            | Op::Rsqrt64Approx { .. }
            | Op::DivSqrt64 { .. }
            | Op::DsetpMinMax { .. }
            | Op::Dsetp { .. }
            | Op::FCvt { .. }
            | Op::F2F { .. }
            | Op::CvtF32F16 { .. }
            | Op::F2fF16 { .. }
            | Op::CvtTf32 { .. }
            | Op::F2fpF16 { .. }
            | Op::DivFull { .. }
            | Op::Mapa { .. }
    )
}

fn writes_memory(op: &Op) -> bool {
    matches!(
        op,
        Op::St { .. }
            | Op::St256 { .. }
            | Op::StAsync { .. }
            | Op::StMatrix { .. }
            | Op::Atom { .. }
            | Op::TmemSt { .. }
            | Op::CpAsync { .. }
            | Op::TmaLoad { .. }
            | Op::TmaStore { .. }
    )
}

fn collective(op: &Op) -> bool {
    matches!(
        op,
        Op::Shfl { .. }
            | Op::VoteAny { .. }
            | Op::VoteAll { .. }
            | Op::Redux { .. }
            | Op::MatchAny { .. }
            | Op::BarResult { .. }
            | Op::R2Ur { .. }
    )
}

/// The lane classes of `k` for blocks of shape `block` (every grid's), or None when every
/// lane is its own class or the kernel is outside the analysis: a call or indexed branch;
/// elect's lane number read; a collective (shuffle, vote, reduction, match, barrier result)
/// that reads a value derived from elect's predicate or runs where it decides, which could
/// carry the leader's position to other lanes; a memory write of such a value by lanes it
/// does not select.
pub fn lane_classes(k: &Kernel, blocks: &[[u32; 3]]) -> Option<Part> {
    if k.code
        .iter()
        .any(|i| matches!(i.op, Op::Call { .. } | Op::CallAbs { .. } | Op::Brx { .. }))
    {
        return None;
    }
    let cfg = Cfg::new(k);
    let n = k.code.len();
    // Elect's predicate and what it reaches, through data and control (flow-insensitive).
    let mut role: HashSet<Reg> = HashSet::new();
    let mut rrole = vec![false; n];
    let mut lane_regs: HashSet<Reg> = HashSet::new();
    for i in &k.code {
        if let Op::Elect { d, lane, .. } = &i.op {
            role.insert(*d);
            lane_regs.extend(*lane);
        }
    }
    loop {
        let before = (role.len(), rrole.iter().filter(|&&x| x).count());
        for (pc, i) in k.code.iter().enumerate() {
            let mut srcs = i.op.src_regs();
            srcs.extend(i.guard.reg);
            if srcs.iter().any(|r| lane_regs.contains(r)) {
                return None;
            }
            let tainted = rrole[pc] || srcs.iter().any(|r| role.contains(r));
            if tainted && collective(&i.op) {
                return None;
            }
            // A write of a value elect's predicate reaches, run by lanes it does not select
            // (not under its control): the lanes of one instruction write in lane order, so
            // which lane leads could decide the value that remains.
            let controlled = rrole[pc] || i.guard.reg.is_some_and(|r| role.contains(&r));
            if !controlled
                && writes_memory(&i.op)
                && i.op.src_regs().iter().any(|r| role.contains(r))
            {
                return None;
            }
            if tainted {
                role.extend(i.op.dst_regs());
                if matches!(i.op, Op::Bra { .. }) {
                    for x in cfg.region(pc) {
                        rrole[x] = true;
                    }
                }
            }
        }
        if (role.len(), rrole.iter().filter(|&&x| x).count()) == before {
            break;
        }
    }
    let local = k
        .code
        .iter()
        .any(|i| matches!(i.op, Op::LdLocal { .. } | Op::StLocal { .. }));
    // Conditional branches with their two sides.
    let branches: Vec<(usize, Vec<usize>, Vec<usize>)> = k
        .code
        .iter()
        .enumerate()
        .filter_map(|(pc, i)| match &i.op {
            Op::Bra { target, .. } if i.guard.reg.is_some() => {
                let stop = cfg.ipdom[pc];
                Some((
                    pc,
                    cfg.reach((*target).min(n), stop),
                    cfg.reach(pc + 1, stop),
                ))
            }
            _ => None,
        })
        .collect();
    let mut all = ONE;
    for block in blocks {
        let threads = (block[0] * block[1] * block[2]) as usize;
        for w in 0..threads.div_ceil(32) {
            if let (p, true) = warp_classes(k, &cfg, &branches, local, block, w)? {
                all = meet(&all, &p);
            }
        }
    }
    (all != singletons()).then_some(all)
}

/// Classes of warp `w` of a block of shape `block`, and whether the warp may elect.
fn warp_classes(
    k: &Kernel,
    cfg: &Cfg,
    branches: &[(usize, Vec<usize>, Vec<usize>)],
    local: bool,
    block: &[u32; 3],
    w: usize,
) -> Option<(Part, bool)> {
    let n = k.code.len();
    let tid = |l: usize, d: u8| -> u64 {
        let t = (w * 32 + l) as u64;
        let (bx, by) = (block[0].max(1) as u64, block[1].max(1) as u64);
        match d {
            0 => t % bx,
            1 => t / bx % by,
            _ => t / (bx * by),
        }
    };
    let sreg = |s: &SReg| -> Av {
        match s {
            SReg::Tid(d) => Av::Lanes(std::array::from_fn(|l| tid(l, *d))),
            SReg::LaneId => Av::Lanes(std::array::from_fn(|l| l as u64)),
            SReg::LaneMaskLt => Av::Lanes(std::array::from_fn(|l| (1u64 << l) - 1)),
            SReg::LaneMaskLe => Av::Lanes(std::array::from_fn(|l| (2u64 << l) - 1)),
            SReg::LaneMaskEq => Av::Lanes(std::array::from_fn(|l| 1u64 << l)),
            SReg::LaneMaskGe => Av::Lanes(std::array::from_fn(|l| trunc(!((1u64 << l) - 1), 32))),
            SReg::LaneMaskGt => Av::Lanes(std::array::from_fn(|l| trunc(!((2u64 << l) - 1), 32))),
            SReg::NTid(d) => Av::Lanes([block.get(*d as usize).copied().unwrap_or(1) as u64; 32]),
            // Everything else is the same in every lane of a warp.
            _ => Av::Part(ONE),
        }
    };
    let mut regs: HashMap<Reg, Av> = HashMap::new();
    let mut ex = vec![Ex::ALL; n];
    for _round in 0..64 {
        let val = |v: &Val, regs: &HashMap<Reg, Av>| -> Av {
            match v {
                Val::Reg(r, _) => regs.get(r).copied().unwrap_or(Av::Part(ONE)),
                Val::Imm(x) => Av::Lanes([*x; 32]),
                Val::Const(..) => Av::Part(ONE),
                Val::SReg(s) => sreg(s),
            }
        };
        let pred = |p: &Pred, regs: &HashMap<Reg, Av>| -> Av {
            match p.reg {
                None => Av::Lanes([!p.neg as u64; 32]),
                Some(r) => match regs.get(&r).copied().unwrap_or(Av::Part(ONE)) {
                    Av::Lanes(v) => Av::Lanes(v.map(|x| ((x != 0) ^ p.neg) as u64)),
                    a => a,
                },
            }
        };
        // Which lanes run each instruction.
        let mut ex_new = vec![Ex::ALL; n];
        for (b, t_side, f_side) in branches {
            let g = pred(&k.code[*b].guard, &regs);
            let target = match &k.code[*b].op {
                Op::Bra { target, .. } => (*target).min(n),
                _ => unreachable!(),
            };
            let (taken, part) = match g {
                Av::Lanes(v) => {
                    let m = (0..32).fold(0u32, |a, l| a | ((v[l] != 0) as u32) << l);
                    (Some(m), mask_part(m))
                }
                Av::Part(p) => (None, p),
            };
            if part == ONE && taken.is_none() {
                continue;
            }
            for (side, to, want) in [(t_side, target, true), (f_side, *b + 1, false)] {
                for &x in side {
                    let e = &mut ex_new[x];
                    e.part = meet(&e.part, &part);
                    if let Some(m) = taken {
                        if cfg.edge_dominates(*b, to, x) {
                            e.known &= if want { m } else { !m };
                        }
                    }
                }
            }
        }
        // Values: each definition with the lanes that run it.
        let mut defs: HashMap<Reg, Vec<(Ex, Av)>> = HashMap::new();
        let mut elects = false;
        for (pc, inst) in k.code.iter().enumerate() {
            let mut e = ex_new[pc];
            match pred(&inst.guard, &regs) {
                Av::Lanes(v) => e.known &= (0..32).fold(0u32, |a, l| a | ((v[l] != 0) as u32) << l),
                Av::Part(p) => e.part = meet(&e.part, &p),
            }
            if e.known == 0 {
                continue;
            }
            if matches!(inst.op, Op::Elect { .. }) {
                elects = true;
            }
            let dsts = inst.op.dst_regs();
            if dsts.is_empty() {
                continue;
            }
            let meet_srcs = |extra: &[Av]| -> Av {
                let mut p = ONE;
                for r in inst.op.src_regs() {
                    p = meet(&p, &val(&Val::Reg(r, 1), &regs).part());
                }
                for v in inst.op.src_vals() {
                    if let Val::SReg(s) = v {
                        p = meet(&p, &sreg(&s).part());
                    }
                }
                for a in extra {
                    p = meet(&p, &a.part());
                }
                Av::Part(p)
            };
            let out: Av = match &inst.op {
                Op::Mov { a, .. } => val(a, &regs),
                Op::Int { op, bits, a, b, .. } => match (val(a, &regs), val(b, &regs)) {
                    (Av::Lanes(x), Av::Lanes(y)) => {
                        let r: Option<Vec<u64>> =
                            (0..32).map(|l| int_op(*op, *bits, x[l], y[l])).collect();
                        match r {
                            Some(r) => Av::Lanes(std::array::from_fn(|l| r[l])),
                            None => meet_srcs(&[]),
                        }
                    }
                    _ => meet_srcs(&[]),
                },
                Op::Setp {
                    cmp: c,
                    bits,
                    a,
                    b,
                    and,
                    or,
                    xor,
                    ex: None,
                    ..
                } => match (val(a, &regs), val(b, &regs), pred(and, &regs)) {
                    (Av::Lanes(x), Av::Lanes(y), Av::Lanes(q)) => {
                        Av::Lanes(std::array::from_fn(|l| {
                            let r = cmp(*c, *bits, x[l], y[l]);
                            let q = q[l] != 0;
                            (if *xor {
                                r ^ q
                            } else if *or {
                                r || q
                            } else {
                                r && q
                            }) as u64
                        }))
                    }
                    _ => meet_srcs(&[]),
                },
                Op::Sel { a, b, p, .. } => match (val(a, &regs), val(b, &regs), pred(p, &regs)) {
                    (Av::Lanes(x), Av::Lanes(y), Av::Lanes(q)) => {
                        Av::Lanes(std::array::from_fn(|l| if q[l] != 0 { x[l] } else { y[l] }))
                    }
                    (x, y, q) => meet_srcs(&[x, y, q]),
                },
                Op::CvtInt {
                    a,
                    from_bits,
                    to_bits,
                    signed,
                    ..
                } => match val(a, &regs) {
                    Av::Lanes(x) => Av::Lanes(x.map(|v| {
                        let v = if *signed {
                            sext(v, *from_bits) as u64
                        } else {
                            trunc(v, *from_bits)
                        };
                        trunc(v, *to_bits)
                    })),
                    _ => meet_srcs(&[]),
                },
                _ if pure(&inst.op) => meet_srcs(&[]),
                // A shuffle: lane l gets the value of lane src(l), or its own.
                Op::Shfl {
                    a,
                    lane,
                    clamp,
                    mode,
                    mask: None,
                    p,
                    ..
                } => {
                    let std_clamp = if *mode == ShflMode::Up { 0 } else { 0x1f };
                    let src: Option<[(usize, bool); 32]> =
                        match (val(lane, &regs), val(clamp, &regs)) {
                            (Av::Lanes(b), Av::Lanes(c))
                                if c.iter().all(|&c| trunc(c, 32) == std_clamp)
                                    && b.iter().all(|&b| trunc(b, 32) < 32) =>
                            {
                                Some(std::array::from_fn(|l| {
                                    let b = trunc(b[l], 32) as i64;
                                    let j = match mode {
                                        ShflMode::Idx => b & 31,
                                        ShflMode::Up => l as i64 - b,
                                        ShflMode::Down => l as i64 + b,
                                        ShflMode::Bfly => l as i64 ^ (b & 31),
                                    };
                                    if (0..32).contains(&j) {
                                        (j as usize, true)
                                    } else {
                                        (l, false)
                                    }
                                }))
                            }
                            _ => None,
                        };
                    let av = val(a, &regs);
                    let d_av = match (src, av) {
                        (Some(s), Av::Lanes(x)) => Av::Lanes(std::array::from_fn(|l| x[s[l].0])),
                        (Some(s), Av::Part(q)) => {
                            let keys: [u8; 32] = std::array::from_fn(|l| q[s[l].0]);
                            Av::Part(canon(&keys))
                        }
                        (None, a) if a.part() == ONE => Av::Part(ONE),
                        (None, _) => Av::Part(singletons()),
                    };
                    let p_av = match src {
                        Some(s) => Av::Lanes(std::array::from_fn(|l| s[l].1 as u64)),
                        None => Av::Part(singletons()),
                    };
                    for d in &dsts {
                        let v = if Some(*d) == *p { p_av } else { d_av };
                        defs.entry(*d).or_default().push((e, v));
                    }
                    continue;
                }
                // One value for the lanes that run the collective together.
                Op::VoteAny { mask, .. } | Op::VoteAll { mask, .. } | Op::Redux { mask, .. } => {
                    let m = mask.as_ref().map(|m| val(m, &regs).part()).unwrap_or(ONE);
                    Av::Part(meet(&e.part(), &m))
                }
                Op::BarResult { .. } => Av::Part(e.part()),
                Op::MatchAny { a, .. } => Av::Part(meet(&e.part(), &val(a, &regs).part())),
                // The leader's predicate: the role the clones trade (elect's lane number
                // is outside the analysis, `lane_classes`).
                Op::Elect { .. } => Av::Part(ONE),
                // A load reads the location its address names: the same for lanes with
                // equal addresses (shared, global, parameters, constants), per lane in
                // local memory. A generic access to the local window is reported
                // (`cvta.local`, S-CBANK); a kernel with local memory counts generic loads
                // as per lane anyway.
                Op::Ld { space, .. } if !(local && *space == Space::Generic) => meet_srcs(&[]),
                Op::Ld256 { .. } | Op::LdConst { .. } | Op::LdReduce { .. } => meet_srcs(&[]),
                Op::MbarTryWait { .. } | Op::ClcQuery { .. } => meet_srcs(&[]),
                // Lane-distributing or lane-ordered results, per-lane storage, and the rest.
                _ => Av::Part(singletons()),
            };
            for d in dsts {
                defs.entry(d).or_default().push((e, out));
            }
        }
        let mut regs_new: HashMap<Reg, Av> = HashMap::new();
        for (r, ds) in defs {
            let v = if ds.len() == 1 && ds[0].0 == Ex::ALL {
                ds[0].1
            } else {
                let mut p = ONE;
                for (e, v) in &ds {
                    p = meet(&p, &e.part());
                    p = meet(&p, &restrict(&v.part(), e.known));
                }
                Av::Part(p)
            };
            regs_new.insert(r, v);
        }
        if regs_new == regs && ex_new == ex {
            // Fixpoint: the classes are what every effect tells apart.
            let mut all = ONE;
            for (pc, inst) in k.code.iter().enumerate() {
                let mut e = ex[pc];
                match pred(&inst.guard, &regs) {
                    Av::Lanes(v) => {
                        e.known &= (0..32).fold(0u32, |a, l| a | ((v[l] != 0) as u32) << l)
                    }
                    Av::Part(p) => e.part = meet(&e.part, &p),
                }
                if e.known == 0 {
                    continue;
                }
                all = meet(&all, &e.part());
                if pure(&inst.op) {
                    continue;
                }
                let mut p = ONE;
                for r in inst.op.src_regs() {
                    p = meet(&p, &val(&Val::Reg(r, 1), &regs).part());
                }
                for v in inst.op.src_vals() {
                    if let Val::SReg(s) = v {
                        p = meet(&p, &sreg(&s).part());
                    }
                }
                all = meet(&all, &restrict(&p, e.known));
            }
            return Some((all, elects));
        }
        regs = regs_new;
        ex = ex_new;
    }
    None
}

/// Lanes of active set `mask` to try as its leader, lowest first: the lowest lane of each
/// group of lanes nothing tells apart given the entries `others` (active set, leader) already
/// fixed: the same class, not a leader there, and in the same of those active sets. Other
/// choices are images of these under a permutation of clones that fixes `others`.
pub fn candidates(classes: Option<&Part>, mask: u32, others: &[(u32, u32)]) -> Vec<u32> {
    let lanes = (0..32).filter(|&l| mask >> l & 1 == 1);
    let Some(cls) = classes else {
        return lanes.collect();
    };
    let used: u32 = others.iter().fold(0, |a, e| a | 1 << (e.1 & 31));
    let mut keys: Vec<(u8, Vec<bool>)> = Vec::new();
    let mut out = Vec::new();
    for l in lanes {
        if used >> l & 1 == 1 {
            out.push(l);
            continue;
        }
        let key = (
            cls[l as usize],
            others.iter().map(|e| e.0 >> l & 1 == 1).collect::<Vec<_>>(),
        );
        if !keys.contains(&key) {
            keys.push(key);
            out.push(l);
        }
    }
    out
}

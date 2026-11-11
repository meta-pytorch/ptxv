/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Representative CTAs (ASSUMPTIONS.md E-CTA-REP, docs/grid-cost.md reduction 1): the
//! certificate for a few CTAs stands for the whole grid.
//!
//! A CTA's execution is a function of its position (`%ctaid`), of values that are the same in
//! every CTA (parameters, `%tid`, constants) and of the bytes it loads. Three runs of a
//! representative CTA r decide which other CTAs behave as r does:
//!
//! - a data run (num.rs taint, every global byte a source): no control operand or address
//!   depends on loaded data (else the reduction refuses);
//! - a position run (num.rs taint, `%ctaid` the source): every position-dependent value that
//!   reaches control (guards, control operands: `Taint::trace`'s points) or a global address is
//!   recorded as a dataflow graph of the per-thread register instructions that computed it,
//!   with the position-independent operands as constants;
//! - the certificate (wellsync.rs) on r's slice.
//!
//! For every other CTA c, the graph is replayed with c's corresponding threads (the
//! instructions themselves, `Machine::exec_pure`, so `%ctaid` reads c's position): if every
//! control value equals r's, c takes r's path step for step (by induction over the path: every
//! other control value is position- and data-independent), with the same synchronization,
//! the same shared-memory addresses and the same global accesses up to their replayed
//! addresses. c's verdict is r's when, in addition, its global accesses overlap each other
//! exactly as r's do (same pairs: the same races and orders) and lie inside their allocations.
//! Across CTAs, no byte one CTA writes may be accessed by another (no global atomics or strong
//! accesses at all). A CTA whose control values differ starts a new regime with its own
//! representative (at most `PTXV_REP_MAX`, default 16).
//!
//! Refusals name their cause (`E-CTA-REP refused: ...`); the caller then checks the launch as
//! before.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::time::Instant;

use ptxv_ir::*;

use crate::exec::Loc;
use crate::explore::Verdict;
use crate::machine::Finding;
use crate::machine::Machine;
use crate::num::pos_free_regs;
use crate::num::pure_op;
use crate::num::value_fault_operands;
use crate::num::value_regs;

/// A source operand of a recorded instruction: a position-independent value, or output `Reg`
/// of node `u32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Src {
    Const(u64),
    Node(u32, Reg),
}

/// One executed register instruction of thread `t` whose result depends on the position.
#[derive(Clone, Debug)]
struct Node {
    t: u32,
    pc: u32,
    srcs: Vec<(Reg, Src)>,
    /// Outputs and their values in the representative.
    outs: Vec<(Reg, u64)>,
    /// Not replayable (a cross-lane operation, a value through memory): why.
    opaque: Option<String>,
}

/// A position-dependent control value (a guard or control operand) and its value in the
/// representative.
#[derive(Clone, Debug)]
struct Ctl {
    pc: u32,
    src: Src,
    val: u64,
}

/// The address operands of one global access instruction of thread `t`.
#[derive(Clone, Debug)]
struct AddrRec {
    t: u32,
    pc: u32,
    srcs: Vec<(Reg, Src)>,
    /// The instruction's address in the representative.
    rep: u64,
}

/// One global access of the representative: `addr` its replayable address operands (None: the
/// address is position-independent, as for every access of an asynchronous unit).
#[derive(Clone, Debug)]
struct Acc {
    pc: u32,
    addr: Option<u32>,
    alloc: usize,
    off: u64,
    n: u32,
    write: bool,
}

#[derive(Default)]
struct Rec {
    nodes: Vec<Node>,
    /// (thread, file, index) -> the node output it holds.
    cur: HashMap<(u32, u8, u16), (u32, Reg)>,
    ctl: Vec<Ctl>,
    addrs: Vec<AddrRec>,
    /// Per thread, the address record of the current instruction.
    cur_addr: HashMap<u32, u32>,
    accs: Vec<Acc>,
    /// Per active thread of the current pure instruction: its source operands.
    pend: Vec<(u32, Vec<(Reg, Src)>)>,
    pend_opaque: Option<String>,
    /// The instruction being executed (between `pre` and `post`).
    cur_pc: Option<u32>,
    refuse: Option<String>,
    /// Nodes of register instructions with a position-dependent value-fault operand (num.rs
    /// `value_fault_operands`: a divisor): replayed for every CTA, where a fault makes the
    /// replay refuse (until 2026-11-08 `div` by `%ctaid - 3` was clean with --rep).
    fault_roots: Vec<u32>,
}

thread_local! {
    static REC: RefCell<Option<Rec>> = const { RefCell::new(None) };
}

/// Cross-lane operations: a lane's result depends on other lanes' operands.
fn cross_lane(op: &Op) -> bool {
    matches!(
        op,
        Op::VoteAny { .. }
            | Op::VoteAll { .. }
            | Op::MatchAny { .. }
            | Op::Redux { .. }
            | Op::R2Ur { .. }
    )
}

fn reg_val(m: &Machine, t: usize, r: Reg) -> u64 {
    match r.file {
        File::P | File::UP => m.prd(
            t,
            Pred {
                reg: Some(r),
                neg: false,
            },
        ) as u64,
        File::V => m.rd(t, &Val::Reg(r, 1)).unwrap_or(0),
        _ => m.rd(t, &Val::Reg(r, 1)).unwrap_or(0) & 0xffff_ffff,
    }
}

fn set_reg(m: &mut Machine, t: usize, r: Reg, v: u64) {
    match r.file {
        File::P | File::UP => m.wp(t, r, v != 0),
        _ => m.wr(t, r, v, 1),
    }
}

impl Rec {
    fn src_of(&mut self, m: &Machine, t: usize, r: Reg) -> Src {
        let tn = &m.num.as_ref().unwrap().taint;
        let v = reg_val(m, t, r);
        if !tn.reg_tainted(t, r) {
            return Src::Const(v);
        }
        if let Some(&(n, rr)) = self.cur.get(&(t as u32, r.file as u8, r.idx)) {
            // The node must still hold the register's value (a write this log did not see
            // makes the value opaque).
            if self.nodes[n as usize]
                .outs
                .iter()
                .any(|&(o, ov)| o == rr && ov == v)
            {
                return Src::Node(n, rr);
            }
        }
        let n = self.nodes.len() as u32;
        self.nodes.push(Node {
            t: t as u32,
            pc: u32::MAX,
            srcs: vec![],
            outs: vec![(r, v)],
            opaque: Some(format!(
                "{r:?} holds a position-dependent value that did not come from a register instruction"
            )),
        });
        Src::Node(n, r)
    }
}

/// Before instruction `pc` of `members` (exec.rs `exec_traced`, numerics on): record the
/// position-dependent control values and address operands it uses, and its sources.
pub(crate) fn pre(m: &Machine, members: &[usize], active: &[usize], pc: u32) {
    REC.with_borrow_mut(|r| {
        let Some(rec) = r.as_mut() else { return };
        if rec.refuse.is_some() {
            return;
        }
        let inst = &m.k.code[pc as usize];
        rec.pend.clear();
        rec.pend_opaque = None;
        rec.cur_addr.clear();
        rec.cur_pc = Some(pc);
        if matches!(inst.op, Op::ClcTryCancel { .. } | Op::ClcQuery { .. }) {
            rec.refuse = Some(format!(
                "cluster launch control at {} (the CTA-to-work mapping is dynamic)",
                m.k.describe(pc as usize)
            ));
            return;
        }
        let tn = &m.num.as_ref().unwrap().taint;
        if let Some(g) = inst.guard.reg {
            let ts: Vec<usize> = members
                .iter()
                .copied()
                .filter(|&t| tn.reg_tainted(t, g))
                .collect();
            for t in ts {
                let src = rec.src_of(m, t, g);
                let val = reg_val(m, t, g);
                rec.ctl.push(Ctl { pc, src, val });
            }
        }
        let srcs = inst.op.src_regs();
        if pure_op(&inst.op) {
            if cross_lane(&inst.op) {
                rec.pend_opaque = Some(format!(
                    "a cross-lane operation at {}",
                    m.k.describe(pc as usize)
                ));
                return;
            }
            // The destinations' old values too: an instruction that writes part of a register
            // (or under its own predicate) keeps the rest.
            let mut regs = srcs.clone();
            for d in inst.op.dst_regs() {
                if !regs.contains(&d) {
                    regs.push(d);
                }
            }
            for &t in active {
                let ss: Vec<(Reg, Src)> = regs.iter().map(|&r| (r, rec.src_of(m, t, r))).collect();
                rec.pend.push((t as u32, ss));
            }
            return;
        }
        let tn = &m.num.as_ref().unwrap().taint;
        let vals = value_regs(&inst.op);
        let free = free_regs(&inst.op);
        for &t in active {
            let ctl: Vec<Reg> = srcs
                .iter()
                .copied()
                .filter(|r| !vals.contains(r) && !free.contains(r) && tn.reg_tainted(t, *r))
                .collect();
            for r in ctl {
                let src = rec.src_of(m, t, r);
                let val = reg_val(m, t, r);
                rec.ctl.push(Ctl { pc, src, val });
            }
            let tn = &m.num.as_ref().unwrap().taint;
            if free.iter().any(|&r| tn.reg_tainted(t, r)) {
                if matches!(inst.op, Op::TmaLoad { .. } | Op::TmaStore { .. }) {
                    rec.refuse = Some(format!(
                        "TMA coordinates depend on the CTA position at {} (boxes are not shifted)",
                        m.k.describe(pc as usize)
                    ));
                    return;
                }
                let Some(rep) = addr_of(&inst.op).and_then(|(a, _)| m.addr(t, a).ok()) else {
                    rec.refuse = Some(format!(
                        "a position-dependent address operand of {}",
                        m.k.describe(pc as usize)
                    ));
                    return;
                };
                let ss: Vec<(Reg, Src)> = free.iter().map(|&r| (r, rec.src_of(m, t, r))).collect();
                let i = rec.addrs.len() as u32;
                rec.addrs.push(AddrRec {
                    t: t as u32,
                    pc,
                    srcs: ss,
                    rep,
                });
                rec.cur_addr.insert(t as u32, i);
            }
        }
    });
}

/// After it: the destinations it left position-dependent hold its node's outputs.
pub(crate) fn post(m: &Machine, _members: &[usize], active: &[usize], pc: u32) {
    REC.with_borrow_mut(|r| {
        let Some(rec) = r.as_mut() else { return };
        if rec.refuse.is_some() {
            return;
        }
        let inst = &m.k.code[pc as usize];
        let tn = &m.num.as_ref().unwrap().taint;
        let dsts = inst.op.dst_regs();
        let pend = std::mem::take(&mut rec.pend);
        for &t in active {
            let tainted: Vec<Reg> = dsts
                .iter()
                .copied()
                .filter(|&d| tn.reg_tainted(t, d))
                .collect();
            for &d in &dsts {
                rec.cur.remove(&(t as u32, d.file as u8, d.idx));
            }
            if tainted.is_empty() {
                continue;
            }
            let outs: Vec<(Reg, u64)> = tainted.iter().map(|&d| (d, reg_val(m, t, d))).collect();
            let srcs = pend.iter().find(|p| p.0 == t as u32).map(|p| p.1.clone());
            let opaque = match (&srcs, &rec.pend_opaque) {
                (_, Some(why)) => Some(why.clone()),
                (Some(_), None) => None,
                (None, None) => Some(format!(
                    "a position-dependent value through {}",
                    m.k.describe(pc as usize)
                )),
            };
            let n = rec.nodes.len() as u32;
            let fault = value_fault_operands(&inst.op);
            if srcs.as_ref().is_some_and(|ss| {
                ss.iter()
                    .any(|(r, s)| fault.contains(r) && matches!(s, Src::Node(..)))
            }) {
                rec.fault_roots.push(n);
            }
            rec.nodes.push(Node {
                t: t as u32,
                pc,
                srcs: srcs.unwrap_or_default(),
                outs,
                opaque,
            });
            for d in tainted {
                rec.cur.insert((t as u32, d.file as u8, d.idx), (n, d));
            }
        }
        rec.cur_addr.clear();
        rec.cur_pc = None;
    });
}

/// A memory access of the recording run (exec.rs `access_x`, numerics on).
pub(crate) fn access(
    m: &Machine,
    actor: u32,
    pc: u32,
    loc: Loc,
    n: usize,
    write: bool,
    strong: u8,
) {
    REC.with_borrow_mut(|r| {
        let Some(rec) = r.as_mut() else { return };
        if rec.refuse.is_some() {
            return;
        }
        let Loc::Global { alloc, off } = loc else {
            return;
        };
        if strong != 0 {
            rec.refuse = Some(format!(
                "a global atomic or strong access at {} (CTAs may synchronize through it)",
                m.k.describe(pc as usize)
            ));
            return;
        }
        let nth = m.threads.len() as u32;
        let t = (actor < 3 * nth).then_some(actor % nth);
        // An access whose address operands may depend on the position must happen during its
        // instruction (its address record is the instruction's).
        if rec.cur_pc != Some(pc) && !free_regs(&m.k.code[pc as usize].op).is_empty() {
            rec.refuse = Some(format!(
                "a global access of {} completes after its instruction",
                m.k.describe(pc as usize)
            ));
            return;
        }
        let addr = t.and_then(|t| rec.cur_addr.get(&t).copied());
        rec.accs.push(Acc {
            pc,
            addr,
            alloc,
            off,
            n: n as u32,
            write,
        });
    });
}

/// What a replay of the representative's graph gives for one CTA.
struct Replayed {
    /// Global accesses, index by index the representative's: (allocation, offset, bytes).
    accs: Vec<(usize, u64, u32)>,
}

enum ReplayErr {
    /// A control value differs (at this pc): another regime.
    Other(u32),
    /// A fault of the CTA (an access outside every allocation): a finding.
    Fault(String),
    Refuse(String),
}

/// Registers of `op` that may depend on the position: its global address operands (num.rs
/// `pos_free_regs`, and a cp.async's source address).
fn free_regs(op: &Op) -> Vec<Reg> {
    match op {
        Op::CpAsync { src, .. } => src
            .base
            .iter()
            .chain(src.base2.iter())
            .flat_map(|v| match v {
                Val::Reg(r, n) if r.file == File::V => vec![*r],
                Val::Reg(r, n) => (0..(*n).max(1) as u16)
                    .map(|i| Reg {
                        file: r.file,
                        idx: r.idx + i,
                    })
                    .collect(),
                _ => vec![],
            })
            .collect(),
        _ => pos_free_regs(op).0,
    }
}

fn addr_of(op: &Op) -> Option<(&Addr, Space)> {
    match op {
        Op::CpAsync {
            src, zfill: false, ..
        } => Some((src, Space::Global)),
        Op::Ld { addr, space, .. } | Op::St { addr, space, .. } | Op::Atom { addr, space, .. } => {
            Some((addr, *space))
        }
        Op::Ld256 { addr, .. } | Op::St256 { addr, .. } | Op::LdReduce { addr, .. } => {
            Some((addr, Space::Global))
        }
        _ => None,
    }
}

impl Rec {
    /// The nodes the control values and addresses need, in order.
    fn needed(&self) -> Vec<u32> {
        let mut need = vec![false; self.nodes.len()];
        let mut todo: Vec<u32> = Vec::new();
        let add = |s: &Src, todo: &mut Vec<u32>| {
            if let Src::Node(n, _) = s {
                todo.push(*n);
            }
        };
        for c in &self.ctl {
            add(&c.src, &mut todo);
        }
        for a in &self.addrs {
            for (_, s) in &a.srcs {
                add(s, &mut todo);
            }
        }
        todo.extend(self.fault_roots.iter().copied());
        while let Some(n) = todo.pop() {
            if need[n as usize] {
                continue;
            }
            need[n as usize] = true;
            for (_, s) in &self.nodes[n as usize].srcs {
                if let Src::Node(k, _) = s {
                    todo.push(*k);
                }
            }
        }
        (0..self.nodes.len() as u32)
            .filter(|&n| need[n as usize])
            .collect()
    }

    fn replay(
        &self,
        need: &[u32],
        m: &mut Machine,
        map: &dyn Fn(u32) -> usize,
    ) -> Result<Replayed, ReplayErr> {
        let mut vals: HashMap<u32, Vec<(Reg, u64)>> = HashMap::new();
        let get = |vals: &HashMap<u32, Vec<(Reg, u64)>>, s: &Src| -> u64 {
            match s {
                Src::Const(v) => *v,
                Src::Node(n, r) => vals[n].iter().find(|x| x.0 == *r).map(|x| x.1).unwrap_or(0),
            }
        };
        for &n in need {
            let node = &self.nodes[n as usize];
            if let Some(why) = &node.opaque {
                return Err(ReplayErr::Refuse(why.clone()));
            }
            let t = map(node.t);
            for (r, s) in &node.srcs {
                let v = get(&vals, s);
                set_reg(m, t, *r, v);
            }
            if !m.exec_pure(t, node.pc) {
                return Err(ReplayErr::Refuse(format!(
                    "the replay of {} did not fall through",
                    m.k.describe(node.pc as usize)
                )));
            }
            let outs = node
                .outs
                .iter()
                .map(|&(r, _)| (r, reg_val(m, t, r)))
                .collect();
            vals.insert(n, outs);
        }
        for c in &self.ctl {
            if get(&vals, &c.src) != c.val {
                return Err(ReplayErr::Other(c.pc));
            }
        }
        // Replayed addresses per address record.
        let mut addr_at: HashMap<u32, u64> = HashMap::new();
        for (i, a) in self.addrs.iter().enumerate() {
            let t = map(a.t);
            for (r, s) in &a.srcs {
                let v = get(&vals, s);
                set_reg(m, t, *r, v);
            }
            let k = m.k.clone();
            let Some((addr, _)) = addr_of(&k.code[a.pc as usize].op) else {
                return Err(ReplayErr::Refuse(format!(
                    "a position-dependent address operand of {}",
                    m.k.describe(a.pc as usize)
                )));
            };
            let v = m.addr(t, addr).map_err(ReplayErr::Fault)?;
            addr_at.insert(i as u32, v);
        }
        m.findings.clear();
        let mut accs = Vec::with_capacity(self.accs.len());
        for acc in &self.accs {
            let Some(ai) = acc.addr else {
                accs.push((acc.alloc, acc.off, acc.n));
                continue;
            };
            let a = &self.addrs[ai as usize];
            let t = map(a.t);
            let k = m.k.clone();
            let (_, space) = addr_of(&k.code[a.pc as usize].op).unwrap();
            // The access's offset from its instruction's address in the representative (a
            // part of a vector access), applied to the replayed address.
            let delta = (m.launch.allocs[acc.alloc].base + acc.off).wrapping_sub(a.rep);
            let ca = addr_at[&ai].wrapping_add(delta);
            match m.resolve(t, space, ca, acc.n as usize) {
                Ok(Loc::Global { alloc, off }) => accs.push((alloc, off, acc.n)),
                Ok(_) => {
                    return Err(ReplayErr::Refuse(format!(
                        "a replayed address of {} is not global",
                        m.k.describe(acc.pc as usize)
                    )));
                }
                Err(e) => return Err(ReplayErr::Fault(e)),
            }
        }
        Ok(Replayed { accs })
    }
}

/// Per-CTA global footprint: (allocation, 64-byte word) -> (read bits, written bits).
type Foot = BTreeMap<(u32, u64), (u64, u64)>;

fn foot_of(accs: &[(usize, u64, u32)], writes: &[bool]) -> Foot {
    let mut f = Foot::new();
    for (&(a, off, n), &w) in accs.iter().zip(writes) {
        for b in off..off + n as u64 {
            let e = f.entry((a as u32, b / 64)).or_insert((0, 0));
            if w {
                e.1 |= 1 << (b % 64);
            } else {
                e.0 |= 1 << (b % 64);
            }
        }
    }
    f
}

/// Pairs of accesses (indices) that share a byte, one of them a write.
fn overlap_pairs(accs: &[(usize, u64, u32)], writes: &[bool]) -> BTreeSet<(u32, u32)> {
    let mut by_word: HashMap<(usize, u64), Vec<u32>> = HashMap::new();
    for (i, &(a, off, n)) in accs.iter().enumerate() {
        // A zero-length access touches no byte (and `off + n - 1` would wrap).
        if n == 0 {
            continue;
        }
        for w in off / 64..=(off + n as u64 - 1) / 64 {
            by_word.entry((a, w)).or_default().push(i as u32);
        }
    }
    let mut out = BTreeSet::new();
    for v in by_word.values() {
        if !v.iter().any(|&i| writes[i as usize]) {
            continue;
        }
        for (x, &i) in v.iter().enumerate() {
            for &j in &v[x + 1..] {
                if !(writes[i as usize] || writes[j as usize]) {
                    continue;
                }
                let (a, b) = (accs[i as usize], accs[j as usize]);
                if a.1 < b.1 + b.2 as u64 && b.1 < a.1 + a.2 as u64 {
                    out.insert((i.min(j), i.max(j)));
                }
            }
        }
    }
    out
}

/// Why the launch is out of scope, before any run.
fn launch_refusal(m: &Machine) -> Option<String> {
    let l = &m.launch;
    if l.grids.len() != 1 {
        return Some("a multi-kernel launch".into());
    }
    if (0..l.ctas()).any(|c| l.cluster_size(c) != 1) || l.grids[0].preferred.is_some() {
        return Some("clusters of more than one CTA".into());
    }
    if !l.mcast.is_empty() || !l.views.is_empty() || l.nic.is_some() || !l.replicas.is_empty() {
        return Some("multicast, further buffer mappings, NIC or multi-GPU memory".into());
    }
    if l.ctas() < 2 {
        return Some("a single CTA".into());
    }
    None
}

/// The recording runs of CTA `c`: the data run's violations, then the position run's log.
fn record(make: &dyn Fn() -> Machine, c: u32) -> Result<Rec, String> {
    // Data run: no control operand or address may depend on loaded bytes.
    let mut m = crate::wellsync::slice_machine(make(), c);
    let mut ns = crate::num::NumState::with_mma_eps(m.launch.mma_model.eps());
    ns.taint.all_global = true;
    m.num = Some(Box::new(ns));
    let (m, _, ok) = crate::explore::run_once(m);
    if !ok {
        return Err("the representative's run did not terminate".into());
    }
    let tn = &m.num.as_ref().unwrap().taint;
    if let Some(v) = tn.violations.iter().next() {
        return Err(format!("data-dependent control or address: {v}"));
    }
    if !tn.fmax_pcs.is_empty() || !tn.fin_pcs.is_empty() {
        return Err("control fixed only by an assumption on the data (N-FMAX / N-FINITE)".into());
    }
    drop(m);
    // Position run.
    let mut m = crate::wellsync::slice_machine(make(), c);
    let mut ns = crate::num::NumState::with_mma_eps(m.launch.mma_model.eps());
    ns.taint.position = true;
    m.num = Some(Box::new(ns));
    REC.with_borrow_mut(|r| *r = Some(Rec::default()));
    let (m, _, ok) = crate::explore::run_once(m);
    let rec = REC.with_borrow_mut(|r| r.take()).unwrap();
    if !ok {
        return Err("the representative's run did not terminate".into());
    }
    if let Some(why) = &rec.refuse {
        return Err(why.clone());
    }
    let tn = &m.num.as_ref().unwrap().taint;
    if let Some(v) = tn.violations.iter().next() {
        return Err(v.clone());
    }
    Ok(rec)
}

/// `ptxv check --certify --rep`: E-CTA-REP, else `fallback` with the refusal as a note.
pub fn check(
    make: &dyn Fn() -> Machine,
    max_states: usize,
    per_slice: &mut dyn FnMut(u32, &Verdict),
    fallback: &mut dyn FnMut() -> Verdict,
) -> Verdict {
    let r =
        crate::smem::two_pass_make(make, &mut |mk| match check_one(mk, max_states, per_slice) {
            Ok(v) => v,
            Err(why) => refused(why),
        });
    if let Some(why) = r.notes.iter().find_map(|n| n.strip_prefix(REFUSED)) {
        let why = why.to_string();
        eprintln!("E-CTA-REP refused: {why}");
        let mut v = fallback();
        v.notes.push(format!("E-CTA-REP refused: {why}"));
        return v;
    }
    r
}

const REFUSED: &str = "\u{0}E-CTA-REP refused: ";

fn refused(why: String) -> Verdict {
    Verdict {
        states: 0,
        transitions: 0,
        complete: false,
        terminal_states: 0,
        distinct_outputs: 0,
        findings: vec![],
        assumptions: vec![],
        notes: vec![format!("{REFUSED}{why}")],
        seconds: 0.0,
        outputs: vec![],
    }
}

fn check_one(
    make: &dyn Fn() -> Machine,
    max_states: usize,
    per_slice: &mut dyn FnMut(u32, &Verdict),
) -> Result<Verdict, String> {
    let t0 = Instant::now();
    let m = make();
    if let Some(why) = launch_refusal(&m) {
        return Err(why);
    }
    let ctas: Vec<u32> = (0..m.launch.ctas()).collect();
    let cid = |c: u32| m.launch.cluster_of(c).0;
    let starts: Vec<usize> = ctas
        .iter()
        .map(|&c| m.cta_threads(c as u16).start)
        .collect();
    let max_reps: usize = std::env::var("PTXV_REP_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let mut scratch = make();
    let mut left: Vec<u32> = ctas.clone();
    let mut all = Verdict {
        states: 0,
        transitions: 0,
        complete: true,
        terminal_states: 0,
        distinct_outputs: 1,
        findings: vec![],
        assumptions: vec![],
        notes: vec![],
        seconds: 0.0,
        outputs: vec![],
    };
    let mut reps: Vec<(u32, usize)> = Vec::new();
    // (cta, footprint) of every CTA.
    let mut foots: Vec<(u32, Foot)> = Vec::new();
    let mut nodes_total = 0usize;
    // The first CTA that left a regime: (CTA, representative, where it decided differently).
    let mut split: Option<(u32, u32, u32)> = None;
    while let Some(&r) = left.first() {
        if reps.len() >= max_reps {
            let (c, r, pc) = split.unwrap_or_default();
            return Err(format!(
                "more than {max_reps} regimes (CTAs whose position-dependent control differs, PTXV_REP_MAX; e.g. CTA {c} decides differently from CTA {r} at {})",
                m.k.describe(pc as usize)
            ));
        }
        let rec = record(make, r)?;
        let need = rec.needed();
        nodes_total += need.len();
        let writes: Vec<bool> = rec.accs.iter().map(|a| a.write).collect();
        let rep_accs: Vec<(usize, u64, u32)> =
            rec.accs.iter().map(|a| (a.alloc, a.off, a.n)).collect();
        let rep_pairs = overlap_pairs(&rep_accs, &writes);
        let r0 = starts[r as usize];
        let mut members = 0usize;
        let mut rest = Vec::new();
        for &c in &left {
            if c == r {
                foots.push((c, foot_of(&rep_accs, &writes)));
                members += 1;
                continue;
            }
            let c0 = starts[c as usize];
            let map = |t: u32| t as usize - r0 + c0;
            // ASSUME(E-CTA-REP): equal control values: c takes r's path.
            match rec.replay(&need, &mut scratch, &map) {
                Ok(rp) => {
                    if overlap_pairs(&rp.accs, &writes) != rep_pairs {
                        return Err(format!(
                            "CTA {c}'s global accesses overlap each other differently from representative CTA {r}'s"
                        ));
                    }
                    foots.push((c, foot_of(&rp.accs, &writes)));
                    members += 1;
                }
                Err(ReplayErr::Other(pc)) => {
                    split.get_or_insert((c, r, pc));
                    rest.push(c)
                }
                Err(ReplayErr::Fault(e)) => {
                    all.findings.push(Finding {
                        kind: "fault".into(),
                        sig: format!("rep-fault:{c}"),
                        msg: format!(
                            "CTA {c} (following representative CTA {r}'s path, E-CTA-REP): {e}"
                        ),
                        trace: vec![],
                        ..Default::default()
                    });
                    members += 1;
                }
                Err(ReplayErr::Refuse(why)) => return Err(why),
            }
        }
        let o = crate::wellsync::certify(crate::wellsync::slice_machine(make(), cid(r)));
        let v = match o.verdict {
            Some(v) => v,
            // A representative that stands for no other CTA needs no single run: its own
            // exploration decides it.
            None if members == 1 => {
                let mut v = crate::explore::explore_slice(make(), max_states, Some(cid(r)));
                v.notes.push(format!(
                    "representative CTA {r}: certificate failed ({}), explored",
                    o.reasons.first().map(String::as_str).unwrap_or("?")
                ));
                v
            }
            None => {
                return Err(format!(
                    "representative CTA {r} (for {members} CTAs) is not certified ({})",
                    o.reasons.first().map(String::as_str).unwrap_or("?")
                ));
            }
        };
        if v.findings.iter().any(|f| f.sig.starts_with("same-value")) {
            return Err(
                "the representative relies on equal values of unordered writes (S-SAME-VALUE)"
                    .into(),
            );
        }
        per_slice(cid(r), &v);
        all.states += v.states;
        all.transitions += v.transitions;
        all.complete &= v.complete;
        all.terminal_states = all.terminal_states.max(v.terminal_states);
        // A representative explored alone may end with several outputs (its CTA's result
        // depends on the schedule): the launch then has at least that many. Until 2026-11-08
        // the launch kept one output and the class `nondeterministic` was dropped.
        if v.distinct_outputs > 1 {
            all.distinct_outputs = all.distinct_outputs.saturating_mul(v.distinct_outputs);
        }
        for f in v.findings {
            if !all.findings.iter().any(|x| x.sig == f.sig) {
                all.findings.push(f);
            }
        }
        for a in v.assumptions {
            if !all.assumptions.contains(&a) {
                all.assumptions.push(a);
            }
        }
        for n in v.notes {
            if !n.starts_with("well-synchronized") && !all.notes.contains(&n) {
                all.notes.push(n);
            }
        }
        reps.push((r, members));
        left = rest;
    }
    // Across CTAs: a byte one CTA writes is accessed by no other.
    // (allocation, word) -> (CTA index, read bits, written bits) of every CTA touching it.
    type Touch = Vec<(usize, u64, u64)>;
    let mut written: HashMap<(u32, u64), Touch> = HashMap::new();
    for (_, f) in &foots {
        for (&k, &(_, w)) in f {
            if w != 0 {
                written.entry(k).or_default();
            }
        }
    }
    for (i, (_, f)) in foots.iter().enumerate() {
        for (k, &(rb, wb)) in f {
            if let Some(v) = written.get_mut(k) {
                v.push((i, rb, wb));
            }
        }
    }
    let mut hits: BTreeMap<u32, (u32, u32, u64)> = BTreeMap::new();
    for (&(a, k), v) in &written {
        for (x, &(i, ri, wi)) in v.iter().enumerate() {
            for &(j, rj, wj) in &v[x + 1..] {
                let y = (wi & (rj | wj)) | (wj & (ri | wi));
                if y != 0 {
                    let b = 64 * k + y.trailing_zeros() as u64;
                    let e = hits.entry(a).or_insert((foots[i].0, foots[j].0, b));
                    if b < e.2 {
                        *e = (foots[i].0, foots[j].0, b);
                    }
                }
            }
        }
    }
    for (alloc, (ci, cj, b)) in hits {
        all.findings.push(Finding {
            kind: "race".into(),
            sig: format!("cross-cluster:{alloc}"),
            msg: format!(
                "clusters {ci} and {cj} both access global byte {b:#x} of allocation {alloc}, at least one writing, with no synchronization between clusters"
            ),
            trace: vec![],
            ..Default::default()
        });
    }
    all.findings
        .sort_by(|a, b| (&a.kind, &a.sig).cmp(&(&b.kind, &b.sig)));
    if !all.assumptions.iter().any(|a| a == "E-CTA-REP") {
        all.assumptions.push("E-CTA-REP".into());
    }
    let reps_s: Vec<String> = reps
        .iter()
        .map(|(r, n)| format!("CTA {r} for {n}"))
        .collect();
    all.notes.push(format!(
        "representative CTAs (E-CTA-REP): {} representatives for {} CTAs ({}); the others replay {} position-dependent values and follow a representative's path; outputs are not assembled",
        reps.len(),
        ctas.len(),
        reps_s.join(", "),
        nodes_total
    ));
    all.seconds = t0.elapsed().as_secs_f64();
    Ok(all)
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Value ranges of symb.rs expressions over every input (interval abstract interpretation),
//! for refuting a side condition: `range(cond)` excluding the wanted value proves that no input
//! makes the condition take it (along the expressions' path).
//!
//! Abstract value of a `w`-bit node: an unsigned interval and a signed interval, the value in
//! both. Leaves the analysis cannot bound (inputs, opaque values, special registers, `Inst`
//! results, floating-point nodes) span their whole width; constants are exact. An `Ite`
//! whose condition compares a node with a bounded one (`x < k`, `!(x > k)`, conjunctions)
//! evaluates each arm with that node narrowed by the condition (path-sensitive), so a clamp
//! `ite(x > hi, hi, ite(x < lo, lo, x))` ranges over `[lo, hi]`.
//!
//! Sound by construction: every transfer function over-approximates the node's values for
//! operand values in their ranges (no rounding inward); an operation it does not model, or an
//! overflow, gives the whole width. A value read through a variable address (`AtAddr`) is
//! bounded only where the address cannot differ from the captured one. The evaluation is bounded (`Budget`); out of budget it
//! proves nothing.

use std::collections::HashMap;

use crate::symb::Arena;
use crate::symb::Bin;
use crate::symb::E;
use crate::symb::Node;
use crate::symb::Un;
use crate::symb::mask;

/// The values of a `w`-bit node: unsigned `[ulo, uhi]` and signed `[slo, shi]` (both hold).
/// Empty when `ulo > uhi` or `slo > shi` (an infeasible path).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iv {
    pub w: u8,
    pub ulo: u64,
    pub uhi: u64,
    pub slo: i64,
    pub shi: i64,
}

fn smin(w: u8) -> i64 {
    if w >= 64 {
        i64::MIN
    } else {
        -(1i64 << (w - 1))
    }
}

fn smax(w: u8) -> i64 {
    if w >= 64 {
        i64::MAX
    } else {
        (1i64 << (w - 1)) - 1
    }
}

/// `v` (w bits) as a signed value.
fn sx(v: u64, w: u8) -> i64 {
    if w >= 64 {
        v as i64
    } else {
        ((v << (64 - w)) as i64) >> (64 - w)
    }
}

/// All ones up to the highest set bit of `v`.
fn smear(v: u64) -> u64 {
    if v == 0 {
        0
    } else {
        u64::MAX >> v.leading_zeros()
    }
}

impl std::fmt::Display for Iv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = |v: u64| {
            if v < 1 << 16 {
                format!("{v}")
            } else {
                format!("{v:#x}")
            }
        };
        let s = |v: i64| {
            if v.unsigned_abs() < 1 << 16 {
                format!("{v}")
            } else if v < 0 {
                format!("-{:#x}", v.unsigned_abs())
            } else {
                format!("{v:#x}")
            }
        };
        if self.is_empty() {
            write!(f, "(empty)")
        } else if self.ulo == self.uhi {
            write!(f, "{}", n(self.ulo))
        } else if self.uhi <= smax(self.w) as u64 || self.slo >= 0 {
            write!(f, "[{}, {}]", n(self.ulo), n(self.uhi))
        } else {
            write!(f, "[{}, {}] (signed)", s(self.slo), s(self.shi))
        }
    }
}

impl Iv {
    pub fn full(w: u8) -> Iv {
        let w = w.clamp(1, 64);
        Iv {
            w,
            ulo: 0,
            uhi: mask(w),
            slo: smin(w),
            shi: smax(w),
        }
    }

    pub fn konst(w: u8, v: u64) -> Iv {
        let w = w.clamp(1, 64);
        let v = v & mask(w);
        Iv {
            w,
            ulo: v,
            uhi: v,
            slo: sx(v, w),
            shi: sx(v, w),
        }
    }

    fn u(w: u8, lo: u64, hi: u64) -> Iv {
        Iv {
            ulo: lo,
            uhi: hi,
            ..Iv::full(w)
        }
        .norm()
    }

    fn s(w: u8, lo: i64, hi: i64) -> Iv {
        Iv {
            slo: lo,
            shi: hi,
            ..Iv::full(w)
        }
        .norm()
    }

    pub fn is_empty(&self) -> bool {
        self.ulo > self.uhi || self.slo > self.shi
    }

    /// The single value, if any.
    pub fn single(&self) -> Option<u64> {
        (!self.is_empty() && self.ulo == self.uhi).then_some(self.ulo)
    }

    /// Each interval tightened by the other.
    fn norm(mut self) -> Iv {
        let w = self.w;
        let sm = smax(w) as u64;
        for _ in 0..2 {
            if self.is_empty() {
                return self;
            }
            // signed -> unsigned: an interval on one side of zero is one unsigned interval
            if self.slo >= 0 {
                self.ulo = self.ulo.max(self.slo as u64);
                self.uhi = self.uhi.min(self.shi as u64);
            } else if self.shi < 0 {
                self.ulo = self.ulo.max(self.slo as u64 & mask(w));
                self.uhi = self.uhi.min(self.shi as u64 & mask(w));
            }
            if self.is_empty() {
                return self;
            }
            // unsigned -> signed: an interval on one side of the sign bit
            if self.uhi <= sm {
                self.slo = self.slo.max(self.ulo as i64);
                self.shi = self.shi.min(self.uhi as i64);
            } else if self.ulo > sm {
                self.slo = self.slo.max(sx(self.ulo, w));
                self.shi = self.shi.min(sx(self.uhi, w));
            }
        }
        self
    }

    pub fn meet(self, o: Iv) -> Iv {
        Iv {
            w: self.w,
            ulo: self.ulo.max(o.ulo),
            uhi: self.uhi.min(o.uhi),
            slo: self.slo.max(o.slo),
            shi: self.shi.min(o.shi),
        }
        .norm()
    }

    pub fn join(self, o: Iv) -> Iv {
        if self.is_empty() {
            return o;
        }
        if o.is_empty() {
            return self;
        }
        Iv {
            w: self.w,
            ulo: self.ulo.min(o.ulo),
            uhi: self.uhi.max(o.uhi),
            slo: self.slo.min(o.slo),
            shi: self.shi.max(o.shi),
        }
        .norm()
    }

    fn bool(t: bool, f: bool) -> Iv {
        match (t, f) {
            (true, false) => Iv::konst(1, 1),
            (false, true) => Iv::konst(1, 0),
            _ => Iv::full(1),
        }
    }
}

/// Integer comparison outcomes possible for operands in `a`, `b`: (can be true, can be false).
fn cmp_iv(c: ptxv_ir::Cmp, a: &Iv, b: &Iv) -> (bool, bool) {
    use ptxv_ir::Cmp::*;
    let ltu = (a.ulo < b.uhi, a.uhi >= b.ulo); // a < b possible, a >= b possible
    let lts = (a.slo < b.shi, a.shi >= b.slo);
    let gtu = (a.uhi > b.ulo, a.ulo <= b.uhi);
    let gts = (a.shi > b.slo, a.slo <= b.shi);
    match c {
        Eq | Ne => {
            let disjoint = a.uhi < b.ulo || b.uhi < a.ulo || a.shi < b.slo || b.shi < a.slo;
            let same = a.single().is_some() && a.single() == b.single();
            let (t, f) = (!disjoint, !same);
            if c == Eq { (t, f) } else { (f, t) }
        }
        LtU => ltu,
        GeU => (ltu.1, ltu.0),
        GtU => gtu,
        LeU => (gtu.1, gtu.0),
        LtS => lts,
        GeS => (lts.1, lts.0),
        GtS => gts,
        LeS => (gts.1, gts.0),
    }
}

/// `x` narrowed by `cmp(x, o) == true` (x on the left; o's range bounds it).
fn narrow(c: ptxv_ir::Cmp, x: Iv, o: Iv) -> Iv {
    use ptxv_ir::Cmp::*;
    let w = x.w;
    let empty = Iv {
        ulo: 1,
        uhi: 0,
        ..x
    };
    let r = match c {
        Eq => o,
        Ne => match o.single() {
            Some(v) if v == x.ulo && v < x.uhi => Iv::u(w, v + 1, x.uhi),
            Some(v) if v == x.uhi && v > x.ulo => Iv::u(w, x.ulo, v - 1),
            Some(v) if x.single() == Some(v) => empty,
            _ => Iv::full(w),
        },
        LtU => match o.uhi.checked_sub(1) {
            Some(h) => Iv::u(w, 0, h),
            None => empty,
        },
        LeU => Iv::u(w, 0, o.uhi),
        GtU => {
            if o.ulo >= mask(w) {
                empty
            } else {
                Iv::u(w, o.ulo + 1, mask(w))
            }
        }
        GeU => Iv::u(w, o.ulo, mask(w)),
        LtS => {
            if o.shi <= smin(w) {
                empty
            } else {
                Iv::s(w, smin(w), o.shi - 1)
            }
        }
        LeS => Iv::s(w, smin(w), o.shi),
        GtS => {
            if o.slo >= smax(w) {
                empty
            } else {
                Iv::s(w, o.slo + 1, smax(w))
            }
        }
        GeS => Iv::s(w, o.slo, smax(w)),
    };
    x.meet(r)
}

fn negate(c: ptxv_ir::Cmp) -> ptxv_ir::Cmp {
    use ptxv_ir::Cmp::*;
    match c {
        Eq => Ne,
        Ne => Eq,
        LtU => GeU,
        GeU => LtU,
        LeU => GtU,
        GtU => LeU,
        LtS => GeS,
        GeS => LtS,
        LeS => GtS,
        GtS => LeS,
    }
}

/// `cmp(a, b)` as `cmp'(b, a)`.
fn mirror(c: ptxv_ir::Cmp) -> ptxv_ir::Cmp {
    use ptxv_ir::Cmp::*;
    match c {
        Eq => Eq,
        Ne => Ne,
        LtU => GtU,
        GtU => LtU,
        LeU => GeU,
        GeU => LeU,
        LtS => GtS,
        GtS => LtS,
        LeS => GeS,
        GeS => LeS,
    }
}

/// Evaluation budget: node evaluations and nested path refinements.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub evals: usize,
    pub depth: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Budget {
            evals: 200_000,
            depth: 6,
        }
    }
}

/// The interval evaluator of one arena.
pub struct Ranges<'a> {
    ar: &'a Arena,
    budget: Budget,
    evals: usize,
    /// Narrowings in force (node, range), innermost last; `ctx` names the set.
    facts: Vec<(E, Iv)>,
    ctx: u32,
    next_ctx: u32,
    memo: HashMap<(E, u32), Iv>,
}

impl<'a> Ranges<'a> {
    pub fn new(ar: &'a Arena, budget: Budget) -> Self {
        Ranges {
            ar,
            budget,
            evals: 0,
            facts: vec![],
            ctx: 0,
            next_ctx: 1,
            memo: HashMap::new(),
        }
    }

    /// The node `e` of the arena.
    pub fn node(&self, e: E) -> &'a Node {
        &self.ar.nodes[e as usize]
    }

    /// The range of `e` over every input; None out of budget (`Budget` per call; the ranges
    /// of nodes outside any refinement stay memoized across calls).
    pub fn range(&mut self, e: E) -> Option<Iv> {
        self.evals = 0;
        let r = self.ev(e, 0);
        if self.next_ctx > 1 {
            self.memo.retain(|k, _| k.1 == 0);
            self.next_ctx = 1;
        }
        r
    }

    /// The range of `e` over the inputs for which every `(cond, value)` of `facts` holds (a
    /// path condition): each fact narrows the nodes it compares (`derive`), then `e` is
    /// evaluated under the narrowings. None out of budget or when a fact cannot hold on its
    /// own; facts that contradict each other narrow a node to an empty range, and a compare
    /// of an empty range spans both values (nothing is claimed).
    pub fn range_under(&mut self, e: E, facts: &[(E, bool)]) -> Option<Iv> {
        self.evals = 0;
        let mut out = Vec::new();
        for &(c, v) in facts {
            if !self.derive(c, v, &mut out, 0)? {
                self.reset_ctx();
                return None;
            }
        }
        let (saved_len, saved_ctx) = (self.facts.len(), self.ctx);
        self.facts.extend(out);
        self.ctx = self.next_ctx;
        self.next_ctx += 1;
        let r = self.ev(e, 1);
        self.facts.truncate(saved_len);
        self.ctx = saved_ctx;
        self.reset_ctx();
        r
    }

    fn reset_ctx(&mut self) {
        if self.next_ctx > 1 {
            self.memo.retain(|k, _| k.1 == 0);
            self.next_ctx = 1;
        }
    }

    fn ev(&mut self, e: E, depth: usize) -> Option<Iv> {
        if let Some(r) = self.memo.get(&(e, self.ctx)) {
            return Some(*r);
        }
        self.evals += 1;
        if self.evals > self.budget.evals {
            return None;
        }
        let w = self.ar.w(e).clamp(1, 64);
        let mut r = self.eval_node(e, w, depth)?;
        for &(x, f) in self.facts.iter() {
            if x == e {
                r = r.meet(f);
            }
        }
        self.memo.insert((e, self.ctx), r);
        Some(r)
    }

    fn eval_node(&mut self, e: E, w: u8, depth: usize) -> Option<Iv> {
        let ar = self.ar;
        Some(match &ar.nodes[e as usize] {
            Node::Const { w, v } => Iv::konst(*w, *v),
            Node::Un { op, a, .. } => {
                let x = self.ev(*a, depth)?;
                match op {
                    Un::Not if !x.is_empty() => Iv {
                        w,
                        ulo: !x.uhi & mask(w),
                        uhi: !x.ulo & mask(w),
                        slo: x.shi.checked_neg().map_or(smin(w), |v| v - 1).max(smin(w)),
                        shi: x.slo.checked_neg().map_or(smax(w), |v| v - 1).min(smax(w)),
                    }
                    .norm(),
                    Un::Not => x,
                    _ => Iv::full(w),
                }
            }
            Node::Bin { op, a, b, .. } => {
                let (x, y) = (self.ev(*a, depth)?, self.ev(*b, depth)?);
                if x.is_empty() || y.is_empty() {
                    return Some(if x.is_empty() { x } else { y });
                }
                bin(*op, w, &x, &y)
            }
            Node::Cmp { cmp, a, b } => {
                let (x, y) = (self.ev(*a, depth)?, self.ev(*b, depth)?);
                if x.is_empty() || y.is_empty() {
                    return Some(Iv::full(1));
                }
                let (t, f) = cmp_iv(*cmp, &x, &y);
                Iv::bool(t, f)
            }
            Node::Ite { c, a, b } => {
                let cv = self.ev(*c, depth)?;
                let (t, f) = (cv.uhi >= 1, cv.ulo == 0);
                let at = if t {
                    self.arm(*c, true, *a, depth)?
                } else {
                    None
                };
                let af = if f {
                    self.arm(*c, false, *b, depth)?
                } else {
                    None
                };
                match (at, af) {
                    (Some(x), Some(y)) => x.join(y),
                    (Some(x), None) | (None, Some(x)) => x,
                    (None, None) => Iv {
                        ulo: 1,
                        uhi: 0,
                        ..Iv::full(w)
                    },
                }
            }
            Node::Zext { a, .. } => {
                let x = self.ev(*a, depth)?;
                Iv::u(w, x.ulo, x.uhi)
            }
            Node::Sext { a, .. } => {
                let x = self.ev(*a, depth)?;
                Iv::s(w, x.slo, x.shi)
            }
            Node::Extract { lo, a, .. } => {
                let x = self.ev(*a, depth)?;
                let (l, h) = (
                    x.ulo.checked_shr(*lo as u32).unwrap_or(0),
                    x.uhi.checked_shr(*lo as u32).unwrap_or(0),
                );
                if h <= mask(w) {
                    Iv::u(w, l, h)
                } else {
                    Iv::full(w)
                }
            }
            Node::Concat { hi, lo } => {
                let (x, y) = (self.ev(*hi, depth)?, self.ev(*lo, depth)?);
                let wl = ar.w(*lo) as u32;
                if wl >= 64 {
                    Iv::full(w)
                } else {
                    Iv::u(w, (x.ulo << wl) | y.ulo, (x.uhi << wl) | y.uhi)
                }
            }
            // `val` is the value only where `addr` is the captured address; elsewhere the
            // bytes of another address, unbounded
            Node::AtAddr { addr, at, val, .. } => {
                let a = self.ev(*addr, depth)?;
                if a.is_empty() {
                    return Some(Iv {
                        ulo: 1,
                        uhi: 0,
                        ..Iv::full(w)
                    });
                }
                if a.single() == Some(*at & mask(a.w)) {
                    self.ev(*val, depth)?
                } else {
                    Iv::full(w)
                }
            }
            // inputs, opaque values, special registers, `Inst` results, floating point
            _ => Iv::full(w),
        })
    }

    /// The range of `arm` where condition `c` is `truth` (None: that path is infeasible).
    fn arm(&mut self, c: E, truth: bool, arm: E, depth: usize) -> Option<Option<Iv>> {
        let mut facts = Vec::new();
        if depth < self.budget.depth && !self.derive(c, truth, &mut facts, depth)? {
            return Some(None);
        }
        if facts.is_empty() {
            return self.ev(arm, depth).map(Some);
        }
        let (saved_len, saved_ctx) = (self.facts.len(), self.ctx);
        self.facts.extend(facts);
        self.ctx = self.next_ctx;
        self.next_ctx += 1;
        let r = self.ev(arm, depth + 1);
        self.facts.truncate(saved_len);
        self.ctx = saved_ctx;
        let r = r?;
        Some((!r.is_empty()).then_some(r))
    }

    /// Facts implied by `c == truth`; false when `c` cannot be `truth`.
    fn derive(&mut self, c: E, truth: bool, out: &mut Vec<(E, Iv)>, depth: usize) -> Option<bool> {
        let ar = self.ar;
        match &ar.nodes[c as usize] {
            Node::Un { op: Un::Not, a, .. } if ar.w(c) == 1 => self.derive(*a, !truth, out, depth),
            Node::Bin {
                op: Bin::And, a, b, ..
            } if ar.w(c) == 1 && truth => {
                Some(self.derive(*a, true, out, depth)? && self.derive(*b, true, out, depth)?)
            }
            Node::Bin {
                op: Bin::Or, a, b, ..
            } if ar.w(c) == 1 && !truth => {
                Some(self.derive(*a, false, out, depth)? && self.derive(*b, false, out, depth)?)
            }
            Node::Cmp { cmp, a, b } => {
                let cmp = if truth { *cmp } else { negate(*cmp) };
                let (x, y) = (self.ev(*a, depth)?, self.ev(*b, depth)?);
                let nx = narrow(cmp, x, y);
                let ny = narrow(mirror(cmp), y, x);
                if nx.is_empty() || ny.is_empty() {
                    return Some(false);
                }
                if nx != x {
                    out.push((*a, nx));
                }
                if ny != y {
                    out.push((*b, ny));
                }
                Some(true)
            }
            _ => {
                let v = self.ev(c, depth)?;
                Some(if truth { v.uhi >= 1 } else { v.ulo == 0 })
            }
        }
    }
}

/// `op` on operands in `x`, `y` (w bits; neither empty).
fn bin(op: Bin, w: u8, x: &Iv, y: &Iv) -> Iv {
    let m = mask(w) as u128;
    let full = Iv::full(w);
    match op {
        Bin::Add => {
            let u = if x.uhi as u128 + y.uhi as u128 <= m {
                Iv::u(w, x.ulo + y.ulo, x.uhi + y.uhi)
            } else {
                full
            };
            let (lo, hi) = (x.slo as i128 + y.slo as i128, x.shi as i128 + y.shi as i128);
            let s = if lo >= smin(w) as i128 && hi <= smax(w) as i128 {
                Iv::s(w, lo as i64, hi as i64)
            } else {
                full
            };
            u.meet(s)
        }
        Bin::Sub => {
            let u = if x.ulo >= y.uhi {
                Iv::u(w, x.ulo - y.uhi, x.uhi - y.ulo)
            } else {
                full
            };
            let (lo, hi) = (x.slo as i128 - y.shi as i128, x.shi as i128 - y.slo as i128);
            let s = if lo >= smin(w) as i128 && hi <= smax(w) as i128 {
                Iv::s(w, lo as i64, hi as i64)
            } else {
                full
            };
            u.meet(s)
        }
        Bin::Mul => {
            if x.uhi as u128 * y.uhi as u128 <= m {
                Iv::u(w, x.ulo * y.ulo, x.uhi * y.uhi)
            } else {
                full
            }
        }
        Bin::MulHiU => Iv::u(
            w,
            ((x.ulo as u128 * y.ulo as u128) >> w) as u64,
            ((x.uhi as u128 * y.uhi as u128) >> w) as u64,
        ),
        _ if matches!(op, Bin::And | Bin::Or | Bin::Xor)
            && x.single().is_some()
            && y.single().is_some() =>
        {
            let (a, b) = (x.ulo, y.ulo);
            Iv::konst(
                w,
                match op {
                    Bin::And => a & b,
                    Bin::Or => a | b,
                    _ => a ^ b,
                },
            )
        }
        // one-bit conjunction: true when both operands are
        Bin::And if w == 1 && x.ulo == 1 && y.ulo == 1 => Iv::konst(1, 1),
        Bin::And => Iv::u(w, 0, x.uhi.min(y.uhi)),
        Bin::Or => Iv::u(w, x.ulo.max(y.ulo), smear(x.uhi | y.uhi) & mask(w)),
        Bin::Xor => Iv::u(w, 0, smear(x.uhi | y.uhi) & mask(w)),
        Bin::Shl => match y.single() {
            Some(k) if k >= w as u64 => Iv::konst(w, 0),
            Some(k) if (x.uhi as u128) << k <= m => Iv::u(w, x.ulo << k, x.uhi << k),
            _ => full,
        },
        Bin::LShr => {
            // an amount of w or more gives 0 (symb.rs bin_eval)
            let sh = |v: u64, k: u64| if k >= w as u64 { 0 } else { v >> k };
            Iv::u(w, sh(x.ulo, y.uhi), sh(x.uhi, y.ulo))
        }
        Bin::AShr => {
            let k = |v: u64| v.min(w as u64 - 1) as u32;
            let c = [
                x.slo >> k(y.ulo),
                x.slo >> k(y.uhi),
                x.shi >> k(y.ulo),
                x.shi >> k(y.uhi),
            ];
            Iv::s(w, *c.iter().min().unwrap(), *c.iter().max().unwrap())
        }
        Bin::UDiv if y.ulo > 0 => Iv::u(w, x.ulo / y.uhi, x.uhi / y.ulo),
        Bin::URem if y.ulo > 0 => {
            if x.uhi < y.ulo {
                *x
            } else {
                Iv::u(w, 0, x.uhi.min(y.uhi - 1))
            }
        }
        Bin::UMin => Iv::u(w, x.ulo.min(y.ulo), x.uhi.min(y.uhi)),
        Bin::UMax => Iv::u(w, x.ulo.max(y.ulo), x.uhi.max(y.uhi)),
        Bin::SMin => Iv::s(w, x.slo.min(y.slo), x.shi.min(y.shi)),
        Bin::SMax => Iv::s(w, x.slo.max(y.slo), x.shi.max(y.shi)),
        _ => full,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use ptxv_ir::Cmp;

    use super::*;
    use crate::symb::Evaluator;
    use crate::symb::InstEval;

    struct NoInst;

    impl InstEval for NoInst {
        fn run_inst(&mut self, _: u32, _: u32, _: u8, _: &[u64]) -> Option<u64> {
            None
        }
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
        /// A value of `w` bits, often at a boundary.
        fn val(&mut self, w: u8) -> u64 {
            let m = mask(w);
            let v = match self.below(8) {
                0 => 0,
                1 => m,
                2 => m >> 1,
                3 => (m >> 1) + 1,
                4 => self.below(18),
                5 => m - self.below(18),
                6 => (m >> 1).wrapping_add(self.below(5)).wrapping_sub(2),
                _ => self.next(),
            };
            v & m
        }
    }

    const CMPS: [Cmp; 10] = [
        Cmp::Eq,
        Cmp::Ne,
        Cmp::LtU,
        Cmp::LeU,
        Cmp::GtU,
        Cmp::GeU,
        Cmp::LtS,
        Cmp::LeS,
        Cmp::GtS,
        Cmp::GeS,
    ];

    const BINS: [Bin; 19] = [
        Bin::Add,
        Bin::Sub,
        Bin::Mul,
        Bin::MulHiU,
        Bin::MulHiS,
        Bin::And,
        Bin::Or,
        Bin::Xor,
        Bin::Shl,
        Bin::LShr,
        Bin::AShr,
        Bin::UDiv,
        Bin::SDiv,
        Bin::URem,
        Bin::SRem,
        Bin::UMin,
        Bin::UMax,
        Bin::SMin,
        Bin::SMax,
    ];

    struct Gen<'r> {
        ar: Arena,
        rng: &'r mut Rng,
        inputs: Vec<E>,
    }

    impl Gen<'_> {
        fn leaf(&mut self, w: u8) -> E {
            if self.rng.below(3) == 0 {
                let v = self.rng.val(w);
                return self.ar.konst(w, v);
            }
            let bytes = (w / 8).max(1);
            let off = self.rng.below(3);
            let v = self.rng.val(8 * bytes);
            let x = self.ar.input(0, off * 8 + bytes as u64, bytes, v);
            self.inputs.push(x);
            if 8 * bytes == w {
                x
            } else {
                self.ar.extract(0, w, x)
            }
        }

        /// A small operand: a constant or a masked / shifted / clamped value.
        fn small(&mut self, w: u8, d: u32) -> E {
            let x = self.expr(w, d);
            let k = self.rng.below(w as u64 + 3);
            let k = self.ar.konst(w, k);
            match self.rng.below(4) {
                0 => k,
                1 => self.ar.bin(Bin::And, x, k),
                2 => self.ar.bin(Bin::UMin, x, k),
                _ => {
                    let s = self.ar.konst(w, w as u64 - 3);
                    self.ar.bin(Bin::LShr, x, s)
                }
            }
        }

        fn cond(&mut self, d: u32) -> E {
            let w = [8u8, 16, 32][self.rng.below(3) as usize];
            match self.rng.below(6) {
                0 if d > 0 => {
                    let c = self.cond(d - 1);
                    self.ar.bnot(c)
                }
                1 if d > 0 => {
                    let (a, b) = (self.cond(d - 1), self.cond(d - 1));
                    let op = if self.rng.below(2) == 0 {
                        Bin::And
                    } else {
                        Bin::Or
                    };
                    self.ar.bin(op, a, b)
                }
                _ => {
                    let a = self.expr(w, d.saturating_sub(1));
                    let b = if self.rng.below(2) == 0 {
                        let v = self.rng.val(w);
                        self.ar.konst(w, v)
                    } else {
                        self.expr(w, d.saturating_sub(1))
                    };
                    let c = CMPS[self.rng.below(10) as usize];
                    if self.rng.below(2) == 0 {
                        self.ar.cmp(c, a, b)
                    } else {
                        self.ar.cmp(c, b, a)
                    }
                }
            }
        }

        /// `x` reused in a narrowed arm: the value, or the value combined with something.
        fn around(&mut self, x: E, d: u32) -> E {
            let w = self.ar.w(x);
            match self.rng.below(4) {
                0 | 1 => x,
                2 => {
                    let y = self.small(w, d);
                    let op = BINS[self.rng.below(19) as usize];
                    self.ar.bin(op, x, y)
                }
                _ => self.expr(w, d),
            }
        }

        fn expr(&mut self, w: u8, d: u32) -> E {
            if d == 0 || self.rng.below(6) == 0 {
                return self.leaf(w);
            }
            let d1 = d - 1;
            match self.rng.below(12) {
                0..=3 => {
                    let op = BINS[self.rng.below(19) as usize];
                    let a = self.expr(w, d1);
                    let b = if matches!(op, Bin::Shl | Bin::LShr | Bin::AShr)
                        || self.rng.below(2) == 0
                    {
                        self.small(w, d1)
                    } else {
                        self.expr(w, d1)
                    };
                    self.ar.bin(op, a, b)
                }
                4 => {
                    let a = self.expr(w, d1);
                    let op = if self.rng.below(2) == 0 {
                        Un::Not
                    } else {
                        Un::Neg
                    };
                    self.ar.un(op, a)
                }
                5 => {
                    let (c, a, b) = (self.cond(d1), self.expr(w, d1), self.expr(w, d1));
                    self.ar.ite(c, a, b)
                }
                6 | 7 => {
                    // a condition on x selecting between expressions of x (clamps, guards)
                    let x = self.expr(w, d1);
                    let k = if self.rng.below(3) == 0 {
                        self.expr(w, d1)
                    } else {
                        let v = self.rng.val(w);
                        self.ar.konst(w, v)
                    };
                    let mut c = {
                        let op = CMPS[self.rng.below(10) as usize];
                        if self.rng.below(2) == 0 {
                            self.ar.cmp(op, x, k)
                        } else {
                            self.ar.cmp(op, k, x)
                        }
                    };
                    if self.rng.below(3) == 0 {
                        let k2 = {
                            let v = self.rng.val(w);
                            self.ar.konst(w, v)
                        };
                        let op = CMPS[self.rng.below(10) as usize];
                        let c2 = self.ar.cmp(op, x, k2);
                        let j = if self.rng.below(2) == 0 {
                            Bin::And
                        } else {
                            Bin::Or
                        };
                        c = self.ar.bin(j, c, c2);
                    }
                    if self.rng.below(3) == 0 {
                        c = self.ar.bnot(c);
                    }
                    let (a, b) = (self.around(x, d1), self.around(x, d1));
                    self.ar.ite(c, a, b)
                }
                8 if w >= 16 => {
                    let a = self.expr(w / 2, d1);
                    if self.rng.below(2) == 0 {
                        self.ar.zext(w, a)
                    } else {
                        self.ar.sext(w, a)
                    }
                }
                9 if w <= 16 => {
                    let a = self.expr(w * 2, d1);
                    let lo = self.rng.below(w as u64 + 1) as u8;
                    self.ar.extract(lo, w, a)
                }
                10 if w >= 16 => {
                    let (h, l) = (self.expr(w / 2, d1), self.expr(w / 2, d1));
                    self.ar.concat(h, l)
                }
                _ => {
                    let a = self.expr(w, d1);
                    let b = self.small(w, d1);
                    let op = [Bin::And, Bin::LShr, Bin::UMin, Bin::URem, Bin::UDiv]
                        [self.rng.below(5) as usize];
                    self.ar.bin(op, a, b)
                }
            }
        }
    }

    fn sx(v: u64, w: u8) -> i64 {
        ((v << (64 - w)) as i64) >> (64 - w)
    }

    /// Every node's range holds its value on random inputs (boundary-heavy): the analysis is
    /// sound, refinement through `ite` conditions included. Also requires the ranges to be
    /// informative (not mostly whole-width), so a vacuous analysis fails.
    #[test]
    fn ranges_hold_every_value() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let (mut checks, mut narrow, mut refuted) = (0usize, 0usize, 0usize);
        for _ in 0..3000 {
            let mut g = Gen {
                ar: Arena::default(),
                rng: &mut rng,
                inputs: vec![],
            };
            g.ar.cap = u32::MAX;
            let w = [8u8, 16, 32][g.rng.below(3) as usize];
            let root = if g.rng.below(3) == 0 {
                g.cond(4)
            } else {
                g.expr(w, 5)
            };
            let Gen { ar, inputs, .. } = g;
            let cone = ar.cone(root);
            let mut rg = Ranges::new(&ar, Budget::default());
            let ranges: Vec<(E, Iv)> = cone
                .iter()
                .filter_map(|&e| Some((e, rg.range(e)?)))
                .collect();
            if ranges
                .iter()
                .any(|(e, r)| *e == root && r.w == 1 && r.single().is_some())
                && ar.is_const(root).is_none()
            {
                refuted += 1;
            }
            narrow += ranges
                .iter()
                .filter(|(e, r)| ar.is_const(*e).is_none() && *r != Iv::full(ar.w(*e)))
                .count();
            let mut ev = Evaluator::with(&ar, Box::new(NoInst));
            for _ in 0..48 {
                let env: HashMap<E, u64> = inputs.iter().map(|&x| (x, rng.val(ar.w(x)))).collect();
                for &(e, r) in &ranges {
                    let Some(v) = ev.eval(e, &env) else { continue };
                    let w = ar.w(e);
                    checks += 1;
                    assert!(
                        !r.is_empty()
                            && r.ulo <= v
                            && v <= r.uhi
                            && r.slo <= sx(v, w)
                            && sx(v, w) <= r.shi,
                        "value {v:#x} outside {r:?} of {}",
                        ar.show(e, &|a| format!("in{a}"), 12)
                    );
                }
            }
        }
        assert!(checks > 1_000_000, "{checks}");
        assert!(narrow > 10_000, "{narrow}");
        assert!(refuted > 100, "{refuted}");
    }

    /// A random 8-bit interval, often narrow and at a boundary, unsigned or signed.
    fn rand_iv(rng: &mut Rng) -> Iv {
        let span = [0, 1, 3, 15, 40][rng.below(5) as usize];
        let a = rng.val(8);
        if rng.below(2) == 0 {
            let lo = a.min(255 - span.min(255));
            Iv::u(8, lo, lo + span)
        } else {
            let lo = sx(a, 8).min(127 - span as i64);
            Iv::s(8, lo, lo + span as i64)
        }
    }

    fn members(r: &Iv) -> Vec<u64> {
        (0..256u64)
            .filter(|&v| r.ulo <= v && v <= r.uhi && r.slo <= sx(v, 8) && sx(v, 8) <= r.shi)
            .collect()
    }

    fn holds(r: &Iv, v: u64) -> bool {
        r.ulo <= v && v <= r.uhi && r.slo <= sx(v, 8) && sx(v, 8) <= r.shi
    }

    /// Each transfer function, comparison outcome and narrowing against every pair of 8-bit
    /// operand values in random operand ranges.
    #[test]
    fn transfers_hold_exhaustively() {
        use crate::symb::bin_eval;
        use crate::symb::cmp_eval;
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let mut pairs = 0usize;
        for _ in 0..6000 {
            let (x, y) = (rand_iv(&mut rng), rand_iv(&mut rng));
            let (xs, ys) = (members(&x), members(&y));
            assert!(!xs.is_empty() && !ys.is_empty(), "{x:?} {y:?}");
            for op in BINS {
                let r = bin(op, 8, &x, &y);
                for &a in &xs {
                    for &b in &ys {
                        if let Some(v) = bin_eval(op, 8, a, b) {
                            pairs += 1;
                            assert!(
                                holds(&r, v),
                                "{op:?} {a:#x} {b:#x} = {v:#x} outside {r:?} ({x:?} {y:?})"
                            );
                        }
                    }
                }
            }
            for c in CMPS {
                let (t, f) = cmp_iv(c, &x, &y);
                let (nx, ny) = (narrow(c, x, y), narrow(mirror(c), y, x));
                let nn = narrow(negate(c), x, y);
                for &a in &xs {
                    for &b in &ys {
                        let v = cmp_eval(c, 8, a, b);
                        assert!(if v { t } else { f }, "{c:?} {a:#x} {b:#x} ({x:?} {y:?})");
                        if v {
                            assert!(
                                holds(&nx, a) && holds(&ny, b),
                                "narrow {c:?} {a:#x} {b:#x} ({x:?} {y:?})"
                            );
                        } else {
                            assert!(holds(&nn, a), "narrow !{c:?} {a:#x} {b:#x} ({x:?} {y:?})");
                        }
                    }
                }
            }
            let mut ar = Arena::default();
            let i = ar.input(0, 0, 1, 0);
            let n = ar.bnot(i);
            let mut rg = Ranges::new(&ar, Budget::default());
            rg.facts.push((i, x));
            let r = rg.range(n).unwrap();
            for &a in &xs {
                assert!(holds(&r, !a & 0xff), "not {a:#x} outside {r:?} ({x:?})");
            }
        }
        assert!(pairs > 5_000_000, "{pairs}");
    }

    /// The clamp shape of a bounds check: `ite(x > 15, 15, ite(x > 0, x, 0))` (signed) ranges
    /// over [0, 15], so `(clamp & 1023) >> 7` is 0 and `0 < that` never holds.
    #[test]
    fn clamp_refutes_bounds_check() {
        let mut ar = Arena::default();
        let x = ar.input(0, 0, 4, 3);
        let (k0, k15) = (ar.konst(32, 0), ar.konst(32, 15));
        let lt = ar.cmp(Cmp::LtS, x, k15);
        let gt = ar.cmp(Cmp::GtS, x, k0);
        let (nlt, ngt) = (ar.bnot(lt), ar.bnot(gt));
        let inner = ar.ite(ngt, k0, x);
        let clamp = ar.ite(nlt, k15, inner);
        let m = ar.konst(32, 1023);
        let s7 = ar.konst(32, 7);
        let a = ar.bin(Bin::And, clamp, m);
        let sh = ar.bin(Bin::LShr, a, s7);
        let c = ar.cmp(Cmp::LtS, k0, sh);
        let mut rg = Ranges::new(&ar, Budget::default());
        assert_eq!(rg.range(clamp).map(|r| (r.ulo, r.uhi)), Some((0, 15)));
        assert_eq!(rg.range(c).and_then(|r| r.single()), Some(0));
        // without the path refinement the clamp is not bounded
        let mut flat = Ranges::new(
            &ar,
            Budget {
                depth: 0,
                ..Budget::default()
            },
        );
        assert_eq!(flat.range(c).and_then(|r| r.single()), None);
    }

    /// A path condition narrows the range: `x <= 0` false on the path, so `x < 1` is never true
    /// there (without the fact both values are possible); contradictory facts claim nothing.
    #[test]
    fn path_condition_refutes_under_its_facts() {
        let mut ar = Arena::default();
        let x = ar.input(0, 0, 4, 3);
        let (k0, k1) = (ar.konst(32, 0), ar.konst(32, 1));
        let le0 = ar.cmp(Cmp::LeS, x, k0);
        let lt1 = ar.cmp(Cmp::LtS, x, k1);
        let mut rg = Ranges::new(&ar, Budget::default());
        assert_eq!(rg.range(lt1), Some(Iv::full(1)));
        assert_eq!(
            rg.range_under(lt1, &[(le0, false)])
                .and_then(|r| r.single()),
            Some(0)
        );
        assert_eq!(
            rg.range_under(lt1, &[(le0, true)]).and_then(|r| r.single()),
            Some(1)
        );
        assert!(
            rg.range_under(lt1, &[(le0, false), (lt1, true)])
                .is_none_or(|r| r.single().is_none())
        );
        // memo outside the facts unchanged
        assert_eq!(rg.range(lt1), Some(Iv::full(1)));
        // randomized: under a fact that holds for the sample, the sample's value is in range
        let mut r = Rng(0x51ed_270b_27c3_9f6b);
        for _ in 0..2000 {
            let v = r.val(32);
            let mut ar = Arena::default();
            let x = ar.input(0, 0, 4, v);
            let k = r.val(32);
            let kk = ar.konst(32, k);
            let c = ar.cmp(CMPS[r.below(10) as usize], x, kk);
            let k2 = r.val(32);
            let kk2 = ar.konst(32, k2);
            let d = ar.cmp(CMPS[r.below(10) as usize], x, kk2);
            let mut ev = Evaluator::with(&ar, Box::new(NoInst));
            let env: HashMap<E, u64> = [(x, v)].into_iter().collect();
            let (cv, dv) = (ev.eval(c, &env).unwrap() == 1, ev.eval(d, &env).unwrap());
            let mut rg = Ranges::new(&ar, Budget::default());
            let Some(iv) = rg.range_under(d, &[(c, cv)]) else {
                panic!("a fact the sample satisfies contradicts: {c} {d}");
            };
            assert!(iv.ulo <= dv && dv <= iv.uhi, "{iv:?} {dv}");
        }
    }

    /// A value read through an input-dependent address is known only at the captured address:
    /// its range is the whole width unless the address cannot be another one, so a condition
    /// on it is never refuted from the captured cell's value.
    #[test]
    fn variable_address_read_is_unbounded() {
        let mut ar = Arena::default();
        let x = ar.input(0, 0, 4, 3);
        let k5 = ar.konst(32, 5);
        let rd = ar.at_addr(x, 3, k5, 32);
        let c = ar.cmp(Cmp::Eq, rd, k5);
        let mut rg = Ranges::new(&ar, Budget::default());
        assert_eq!(rg.range(rd), Some(Iv::full(32)));
        // `rd == 5` can be false (another address holds other bytes)
        assert_eq!(rg.range(c), Some(Iv::full(1)));
        // an address clamped to the captured one: the read is the captured cell's
        let k3 = ar.konst(32, 3);
        let lt = ar.cmp(Cmp::LtS, x, k3);
        let gt = ar.cmp(Cmp::GtS, x, k3);
        let (nlt, ngt) = (ar.bnot(lt), ar.bnot(gt));
        let hi = ar.ite(ngt, x, k3);
        let a3 = ar.ite(nlt, hi, k3);
        let rd3 = ar.at_addr(a3, 3, k5, 32);
        let mut rg = Ranges::new(&ar, Budget::default());
        assert_eq!(rg.range(a3).and_then(|r| r.single()), Some(3));
        assert_eq!(rg.range(rd3).and_then(|r| r.single()), Some(5));
    }
}

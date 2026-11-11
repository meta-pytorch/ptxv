/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Local error tables of instruction slices and MUFU ops: the law of a slice's own error
//! g(a) - f(a) (its bit-exact value against the real function it is identified with) per
//! cell of its index variable, in units of the f32 ulp of f.
//!
//! The index variable t is the slice's inner argument (z = c0 + sum c_i a_i: the exp
//! argument, the rsqrt argument), the argument itself for polynomial and named forms, and
//! the quotient for a0 / a1. A cell is (sign, binade of t, 1/8 of the binade). Each cell is
//! filled by evaluating the slice bit-exactly at `PER_CELL` f32 argument tuples with t in the
//! cell (the other arguments from the identification probes), so the law within a cell is
//! the law for t uniform on the cell's f32 values. The MUFU units are tabulated the same way
//! on their operand. This is the "compile" step: tables depend on the kernel only, and any
//! input law is then a lookup per cell.

use serde::Deserialize;
use serde::Serialize;

use super::ident::Func;
use super::ident::RealFn;
use crate::errdist::det::HashMap;
use crate::seg::SProg;

/// Sub-cells per binade.
pub const SUB: i32 = 8;
/// Histogram bins of the error in ulps: width 1/8 over [-8, 8].
pub const KW: f64 = 0.125;
pub const KHALF: i32 = 64;
pub const PER_CELL: usize = 96;

/// The local error law in one cell.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Cell {
    pub sign: i8,
    pub binade: i16,
    pub sub: u8,
    pub n: u32,
    /// Mean and second moment of k = (g - f) / ulp32(f).
    pub mean: f64,
    pub m2: f64,
    /// P(k = 0) (g equal to the real value: exact results).
    pub p0: f64,
    /// Histogram of k: bins of width KW centred on multiples of KW in [-8, 8], then the
    /// masses below and above.
    pub hist: Vec<f32>,
    pub tails: [f32; 2],
    /// Integer offset from the correctly rounded result, (g - RN(f)) / ulp32(f): -4..=4.
    pub off: [f32; 9],
    /// The slice's error before its last rounding, eta = (v_pre - f) / ulp32(f), where the
    /// slice ends in a rounding FFMA / FMUL / FADD / div.rn (through selects and guards):
    /// fraction of evaluations where it does, mean, second moment, histogram (as `hist`).
    #[serde(default)]
    pub pre_ok: f64,
    #[serde(default)]
    pub pre_mean: f64,
    #[serde(default)]
    pub pre_m2: f64,
    #[serde(default)]
    pub pre_hist: Vec<f32>,
}

impl Cell {
    pub fn var(&self) -> f64 {
        (self.m2 - self.mean * self.mean).max(0.0)
    }
}

/// How the index variable is computed from a slice's arguments.
pub fn index_value(r: &RealFn, a: &[f64]) -> f64 {
    match r.func {
        Func::Div => a[0] / a[1],
        Func::Poly => a[r.parg],
        Func::Named(_) => {
            let k = r.coef.iter().position(|c| *c != 0.0).unwrap_or(0);
            a[k]
        }
        _ => r.z(a),
    }
}

/// The cell coordinates of t (t = 0: binade i16::MIN).
pub fn cell_of(t: f64) -> (i8, i16, u8) {
    if t == 0.0 || !t.is_finite() {
        return (0, i16::MIN, 0);
    }
    let e = super::super::spec::exp_floor(t);
    let m = t.abs() / 2f64.powi(e) - 1.0;
    let s = ((m * SUB as f64).floor() as i32).clamp(0, SUB - 1);
    (if t < 0.0 { -1 } else { 1 }, e as i16, s as u8)
}

/// One slice's (or MUFU op's) table.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Table {
    pub name: String,
    pub cells: Vec<Cell>,
    #[serde(skip)]
    index: HashMap<(i8, i16, u8), usize>,
    /// Per sign: the binade range covered.
    pub range: Vec<(i8, i16, i16)>,
}

impl Table {
    pub fn reindex(&mut self) {
        self.index = self
            .cells
            .iter()
            .enumerate()
            .map(|(i, c)| ((c.sign, c.binade, c.sub), i))
            .collect();
    }
    /// Add cells (replacing equal keys) and widen the covered range of sign `s` to lo..=hi.
    pub fn extend(&mut self, cells: Vec<Cell>, s: i8, lo: i16, hi: i16) {
        for c in cells {
            match self
                .cells
                .iter()
                .position(|x| (x.sign, x.binade, x.sub) == (c.sign, c.binade, c.sub))
            {
                Some(i) => self.cells[i] = c,
                None => self.cells.push(c),
            }
        }
        self.cells.sort_by_key(|c| (c.sign, c.binade, c.sub));
        match self.range.iter_mut().find(|r| r.0 == s) {
            Some(r) => {
                r.1 = r.1.min(lo);
                r.2 = r.2.max(hi);
            }
            None => {
                self.range.push((s, lo, hi));
                self.range.sort();
            }
        }
        self.reindex();
    }
    /// Whether binade e of sign s is inside the covered range.
    pub fn covers(&self, s: i8, e: i16) -> bool {
        self.range.iter().any(|r| r.0 == s && r.1 <= e && e <= r.2)
    }
    /// The cell of t: exact match, else the nearest binade of the same sign within the
    /// covered range (clamped below: tiny arguments behave like the smallest tabulated).
    pub fn lookup(&self, t: f64) -> Option<usize> {
        let k = cell_of(t);
        if let Some(&i) = self.index.get(&k) {
            return Some(i);
        }
        let (s, e, sub) = k;
        let &(_, lo, hi) = self.range.iter().find(|r| r.0 == s)?;
        if e < lo {
            return self
                .index
                .get(&(s, lo, sub))
                .or_else(|| self.index.get(&(s, lo, 0)))
                .copied();
        }
        if e > hi {
            return None;
        }
        (0..SUB as u8)
            .filter_map(|j| self.index.get(&(s, e, j)))
            .next()
            .copied()
    }
}

fn ulp32(v: f64) -> f64 {
    super::super::mc::ulp_out(v, super::super::spec::Format::F32)
}

/// Accumulate one evaluation into a cell's raw sums.
#[derive(Default, Clone)]
struct Acc {
    n: u32,
    s: f64,
    s2: f64,
    z: u32,
    hist: Vec<f64>,
    tails: [f64; 2],
    off: [f64; 9],
    pn: u32,
    ps: f64,
    ps2: f64,
    phist: Vec<f64>,
}

impl Acc {
    fn add(&mut self, g: f64, f: f64) {
        if !g.is_finite() || !f.is_finite() {
            return;
        }
        let q = ulp32(f);
        let k = (g - f) / q;
        if self.hist.is_empty() {
            self.hist = vec![0.0; (2 * KHALF + 1) as usize];
        }
        self.n += 1;
        self.s += k;
        self.s2 += k * k;
        self.z += (g == f) as u32;
        let b = (k / KW).round() as i64;
        if b < -(KHALF as i64) {
            self.tails[0] += 1.0;
        } else if b > KHALF as i64 {
            self.tails[1] += 1.0;
        } else {
            self.hist[(b + KHALF as i64) as usize] += 1.0;
        }
        let rn = f as f32 as f64;
        let o = ((g - rn) / q).round() as i64;
        self.off[(o.clamp(-4, 4) + 4) as usize] += 1.0;
    }
    fn add_pre(&mut self, pre: f64, f: f64) {
        if !pre.is_finite() || !f.is_finite() {
            return;
        }
        let q = ulp32(f);
        let k = (pre - f) / q;
        if self.phist.is_empty() {
            self.phist = vec![0.0; (2 * KHALF + 1) as usize];
        }
        self.pn += 1;
        self.ps += k;
        self.ps2 += k * k;
        let b = ((k / KW).round() as i64).clamp(-(KHALF as i64), KHALF as i64);
        self.phist[(b + KHALF as i64) as usize] += 1.0;
    }
    fn merge(&mut self, a: &Acc) {
        if self.hist.is_empty() {
            *self = a.clone();
            return;
        }
        self.n += a.n;
        self.s += a.s;
        self.s2 += a.s2;
        self.z += a.z;
        for (x, y) in self.hist.iter_mut().zip(&a.hist) {
            *x += y;
        }
        for j in 0..2 {
            self.tails[j] += a.tails[j];
        }
        for j in 0..9 {
            self.off[j] += a.off[j];
        }
        self.pn += a.pn;
        self.ps += a.ps;
        self.ps2 += a.ps2;
        if self.phist.is_empty() {
            self.phist = a.phist.clone();
        } else {
            for (x, y) in self.phist.iter_mut().zip(&a.phist) {
                *x += y;
            }
        }
    }
    fn cell(&self, key: (i8, i16, u8)) -> Cell {
        let n = self.n.max(1) as f64;
        Cell {
            sign: key.0,
            binade: key.1,
            sub: key.2,
            n: self.n,
            mean: self.s / n,
            m2: self.s2 / n,
            p0: self.z as f64 / n,
            hist: self.hist.iter().map(|x| (*x / n) as f32).collect(),
            tails: [(self.tails[0] / n) as f32, (self.tails[1] / n) as f32],
            off: self.off.map(|x| (x / n) as f32),
            pre_ok: self.pn as f64 / n,
            pre_mean: self.ps / self.pn.max(1) as f64,
            pre_m2: self.ps2 / self.pn.max(1) as f64,
            pre_hist: self
                .phist
                .iter()
                .map(|x| (*x / self.pn.max(1) as f64) as f32)
                .collect(),
        }
    }
}

/// The value of a slice's output before its last rounding (an FFMA / FMUL / FADD to nearest
/// or a div.rn, reached through selects and guards), from the op values of an evaluation.
pub fn pre_value(p: &SProg, args: &[u32], vals: &[u64]) -> Option<f64> {
    use ptxv_ir::Op as I;
    use ptxv_ir::Val;

    use crate::seg::SKind;
    use crate::seg::SVal;
    let get = |v: SVal| -> Option<u64> {
        match v {
            SVal::Op(i) => vals.get(i as usize).copied(),
            SVal::Const(c) => Some(c),
            SVal::Arg(i) => args.get(i as usize).map(|x| *x as u64),
        }
    };
    let mut cur = p.out;
    for _ in 0..8 {
        let SVal::Op(i) = cur else { return None };
        match &p.ops[i as usize].kind {
            SKind::Guard {
                pred,
                neg,
                new,
                old,
            } => {
                cur = if (get(*pred)? != 0) != *neg {
                    *new
                } else {
                    *old
                };
            }
            SKind::DivRn { a, b } => {
                let x = f32::from_bits(get(*a)? as u32) as f64;
                let y = f32::from_bits(get(*b)? as u32) as f64;
                return Some(x / y);
            }
            SKind::Inst {
                op, regs, cvals, ..
            } => {
                let sv = |v: &Val| -> Option<SVal> {
                    match v {
                        Val::Reg(r, _) => regs.iter().find(|(x, _)| x == r).map(|x| x.1),
                        Val::Imm(c) => Some(SVal::Const(*c)),
                        v => cvals
                            .iter()
                            .find(|(x, _)| x == v)
                            .map(|(_, c)| SVal::Const(*c)),
                    }
                };
                let f = |v: &Val, neg: bool, abs: bool| -> Option<f64> {
                    let mut x = f32::from_bits(get(sv(v)?)? as u32) as f64;
                    if abs {
                        x = x.abs();
                    }
                    if neg {
                        x = -x;
                    }
                    Some(x)
                };
                match op {
                    I::Sel { a, b, p: pr, .. } => {
                        let t = match pr.reg {
                            None => true,
                            Some(r) => {
                                let s = regs.iter().find(|(x, _)| *x == r)?.1;
                                get(s)? != 0
                            }
                        } != pr.neg;
                        let v = if t { a } else { b };
                        match sv(v)? {
                            s @ SVal::Op(_) => cur = s,
                            _ => return None,
                        }
                    }
                    I::Ffma {
                        a,
                        b,
                        c,
                        neg,
                        abs,
                        rnd: ptxv_ir::Rnd::Rn,
                        sat: false,
                        ..
                    } => {
                        let x = f(a, neg[0], abs[0])? * f(b, neg[1], abs[1])?;
                        let z = match c {
                            Some(c) => f(c, neg[2], abs[2])?,
                            None => 0.0,
                        };
                        return Some(x + z);
                    }
                    I::Fadd {
                        a,
                        b,
                        neg,
                        abs,
                        rnd: ptxv_ir::Rnd::Rn,
                        sat: false,
                        ..
                    } => return Some(f(a, neg[0], abs[0])? + f(b, neg[1], abs[1])?),
                    _ => return None,
                }
            }
        }
    }
    None
}

/// The table of a slice identified as `r`, from its identification probes.
/// The driver argument of a slice: solved for to hit a target t.
fn slice_driver(r: &RealFn, probes: &[(Vec<f64>, f64)]) -> usize {
    let n = r.coef.len();
    match r.func {
        Func::Div => 0,
        Func::Poly => r.parg,
        Func::Named(_) => r.coef.iter().position(|c| *c != 0.0).unwrap_or(0),
        _ => {
            let mut best = (0usize, -1.0f64);
            for i in 0..n {
                if r.coef[i] == 0.0 {
                    continue;
                }
                let m = probes
                    .iter()
                    .map(|q| (r.coef[i] * q.0[i]).abs())
                    .sum::<f64>();
                if m > best.1 {
                    best = (i, m);
                }
            }
            best.0
        }
    }
}

// ASSUME(N-ERRDIST-NL-INDEP): a cell's law is the law for t uniform on its f32 values.
pub fn slice_table(name: &str, p: &SProg, r: &RealFn, probes: &[(Vec<f64>, f64)]) -> Table {
    // Binade ranges of t per sign, from the probes.
    let mut rng: HashMap<i8, (i16, i16)> = HashMap::default();
    let mut zero = false;
    for (a, _) in probes {
        let t = index_value(r, a);
        let (s, e, _) = cell_of(t);
        if s == 0 {
            zero = true;
            continue;
        }
        let x = rng.entry(s).or_insert((e, e));
        x.0 = x.0.min(e);
        x.1 = x.1.max(e);
    }
    let mut ranges = Vec::new();
    for (s, (lo, hi)) in rng {
        let hi = hi.saturating_add(2);
        let lo = lo.saturating_sub(2).max(hi.saturating_sub(60));
        ranges.push((s, lo, hi));
    }
    ranges.sort();
    // Work items: (sign, binade).
    let items: Vec<(i8, i16)> = ranges
        .iter()
        .flat_map(|&(s, lo, hi)| (lo..=hi).map(move |e| (s, e)))
        .collect();
    let mut all = slice_accs(p, r, probes, &items);
    // t = 0 exactly (the maximum's own exp(0)): evaluated at probes with t = 0.
    if zero {
        let mut acc = Acc::default();
        let mut scratch = Vec::new();
        for (a, _) in probes
            .iter()
            .filter(|q| index_value(r, &q.0) == 0.0)
            .take(64)
        {
            let bits: Vec<u32> = a.iter().map(|x| (*x as f32).to_bits()).collect();
            if let Ok(g) = p.eval_with(&bits, &mut scratch) {
                let f = r.eval(a);
                acc.add(f32::from_bits(g) as f64, f);
                if let Some(pre) = pre_value(p, &bits, &scratch) {
                    acc.add_pre(pre, f);
                }
            }
        }
        if acc.n > 0 {
            all.insert((0, i16::MIN, 0), acc);
        }
    }
    let mut cells: Vec<Cell> = all
        .into_iter()
        .filter(|(_, a)| a.n > 0)
        .map(|(k, a)| a.cell(k))
        .collect();
    cells.sort_by_key(|c| (c.sign, c.binade, c.sub));
    let mut t = Table {
        name: name.into(),
        cells,
        index: HashMap::default(),
        range: ranges,
    };
    t.reindex();
    t
}

/// The cells of a slice table in the binades `items`.
pub fn slice_cells(
    p: &SProg,
    r: &RealFn,
    probes: &[(Vec<f64>, f64)],
    items: &[(i8, i16)],
) -> Vec<Cell> {
    slice_accs(p, r, probes, items)
        .into_iter()
        .filter(|(_, a)| a.n > 0)
        .map(|(k, a)| a.cell(k))
        .collect()
}

fn slice_accs(
    p: &SProg,
    r: &RealFn,
    probes: &[(Vec<f64>, f64)],
    items: &[(i8, i16)],
) -> HashMap<(i8, i16, u8), Acc> {
    let driver = slice_driver(r, probes);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism()
        .map_or(8, |n| n.get())
        .min(64)
        .min(items.len().max(1));
    // Per item: its index and its (sign, binade, 1/8 binade) cells -> accumulator, drawn from
    // the item's own stream and merged in item order, so the table does not depend on which
    // thread took which item.
    type Cells = Vec<(usize, Vec<((i8, i16, u8), Acc)>)>;
    let mut results: Cells = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                let items = &items;
                let next = &next;
                sc.spawn(move || {
                    let mut done = Vec::new();
                    let mut scratch = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&(s, e)) = items.get(i) else { break };
                        let mut out: HashMap<(i8, i16, u8), Acc> = HashMap::default();
                        let seed = 0x7ab1e ^ ((s as u8 as u64) << 16) ^ (e as u16 as u64);
                        let mut rnd = super::super::special::Rng::new(seed);
                        for sub in 0..SUB {
                            for _ in 0..PER_CELL {
                                let t = s as f64
                                    * 2f64.powi(e as i32)
                                    * (1.0 + (sub as f64 + rnd.unit()) / SUB as f64);
                                let Some(args) = solve(r, driver, t, probes, &mut rnd) else {
                                    continue;
                                };
                                let bits: Vec<u32> =
                                    args.iter().map(|x| (*x as f32).to_bits()).collect();
                                let a32: Vec<f64> =
                                    bits.iter().map(|b| f32::from_bits(*b) as f64).collect();
                                let Ok(g) = p.eval_with(&bits, &mut scratch) else {
                                    continue;
                                };
                                let g = f32::from_bits(g) as f64;
                                let f = r.eval(&a32);
                                let key = cell_of(index_value(r, &a32));
                                let e = out.entry(key).or_default();
                                e.add(g, f);
                                if let Some(pre) = pre_value(p, &bits, &scratch) {
                                    e.add_pre(pre, f);
                                }
                            }
                        }
                        let mut cells: Vec<_> = out.into_iter().collect();
                        cells.sort_by_key(|c| c.0);
                        done.push((i, cells));
                    }
                    done
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    results.sort_by_key(|x| x.0);
    let mut all: HashMap<(i8, i16, u8), Acc> = HashMap::default();
    for (_, part) in results {
        for (k, a) in part {
            all.entry(k).or_default().merge(&a);
        }
    }
    all
}

/// Whether two slices identified with the same real function compute the same function of
/// its index value: evaluated at index values drawn from the first's probes (each slice's
/// arguments solved for the same t), the outputs are bitwise equal at >= 95% of them (the
/// rest: the slices' internal roundings of t). Their own errors then cancel in a pair
/// (N-ERRDIST-NL-JOINT).
pub fn same_function(
    p1: &SProg,
    r1: &RealFn,
    q1: &[(Vec<f64>, f64)],
    p2: &SProg,
    r2: &RealFn,
    q2: &[(Vec<f64>, f64)],
) -> bool {
    if r1.func != r2.func
        || r1.outer != r2.outer
        || r1.pow != r2.pow
        || q1.is_empty()
        || q2.is_empty()
    {
        return false;
    }
    let drv = |r: &RealFn, q: &[(Vec<f64>, f64)]| -> usize {
        match r.func {
            Func::Div => 0,
            Func::Poly => r.parg,
            Func::Named(_) => r.coef.iter().position(|c| *c != 0.0).unwrap_or(0),
            _ => (0..r.coef.len())
                .filter(|&i| r.coef[i] != 0.0)
                .max_by(|&a, &b| {
                    let m = |i: usize| q.iter().map(|x| (r.coef[i] * x.0[i]).abs()).sum::<f64>();
                    m(a).total_cmp(&m(b))
                })
                .unwrap_or(0),
        }
    };
    let (d1, d2) = (drv(r1, q1), drv(r2, q2));
    let mut rnd = super::super::special::Rng::new(0x5a3e);
    let (mut same, mut n) = (0usize, 0usize);
    let (mut s1, mut s2) = (Vec::new(), Vec::new());
    for _ in 0..400 {
        let t = index_value(r1, &q1[rnd.below(q1.len() as u64) as usize].0);
        if !t.is_finite() {
            continue;
        }
        let (Some(a1), Some(a2)) = (
            solve(r1, d1, t, q1, &mut rnd),
            solve(r2, d2, t, q2, &mut rnd),
        ) else {
            continue;
        };
        let b1: Vec<u32> = a1.iter().map(|x| (*x as f32).to_bits()).collect();
        let b2: Vec<u32> = a2.iter().map(|x| (*x as f32).to_bits()).collect();
        let (Ok(g1), Ok(g2)) = (p1.eval_with(&b1, &mut s1), p2.eval_with(&b2, &mut s2)) else {
            continue;
        };
        n += 1;
        same += (g1 == g2) as usize;
    }
    n >= 50 && same * 100 >= 95 * n
}

/// The joint table of two slices identified with the same real function of the same index
/// variable (two kernels' implementations of exp, rsqrt, ...): per cell of t, the law of the
/// integer difference (g2 - g1) / ulp32(f) of the two slices evaluated at arguments with the
/// same t (`Cell::off`, -4..=4). Their errors are correlated where they share a core (both
/// evaluate MUFU.EX2 at nearly the same fraction), which this captures (N-ERRDIST-NL-JOINT).
pub fn pair_table(
    p1: &SProg,
    r1: &RealFn,
    q1: &[(Vec<f64>, f64)],
    p2: &SProg,
    r2: &RealFn,
    q2: &[(Vec<f64>, f64)],
    range: &[(i8, i16, i16)],
) -> Option<Table> {
    if r1.func != r2.func
        || r1.outer != r2.outer
        || r1.pow != r2.pow
        || r1.mul.is_some() != r2.mul.is_some()
        || q1.is_empty()
        || q2.is_empty()
    {
        return None;
    }
    let drv = |r: &RealFn, q: &[(Vec<f64>, f64)]| -> usize {
        match r.func {
            Func::Div => 0,
            Func::Poly => r.parg,
            Func::Named(_) => r.coef.iter().position(|c| *c != 0.0).unwrap_or(0),
            _ => (0..r.coef.len())
                .filter(|&i| r.coef[i] != 0.0)
                .max_by(|&a, &b| {
                    let m = |i: usize| q.iter().map(|x| (r.coef[i] * x.0[i]).abs()).sum::<f64>();
                    m(a).total_cmp(&m(b))
                })
                .unwrap_or(0),
        }
    };
    let (d1, d2) = (drv(r1, q1), drv(r2, q2));
    // Only slices whose value depends on the driver alone: other arguments would have to
    // correspond between the kernels.
    let alone = |p: &SProg, r: &RealFn, d: usize| {
        r.mul.is_none()
            && r.func != Func::Div
            && (0..p.args.len()).all(|i| i == d || r.coef.get(i).is_none_or(|c| *c == 0.0))
            && (r.func != Func::Poly || r.parg == d)
    };
    if !alone(p1, r1, d1) || !alone(p2, r2, d2) {
        return None;
    }
    let mut rnd = super::super::special::Rng::new(0x9a17);
    let (mut s1, mut s2) = (Vec::new(), Vec::new());
    let mut cells: HashMap<(i8, i16, u8), [f64; 10]> = HashMap::default();
    for &(sg, lo, hi) in range {
        for e in lo..=hi {
            for sub in 0..SUB {
                for _ in 0..PER_CELL {
                    let t = sg as f64
                        * 2f64.powi(e as i32)
                        * (1.0 + (sub as f64 + rnd.unit()) / SUB as f64);
                    let (Some(a1), Some(a2)) = (
                        solve(r1, d1, t, q1, &mut rnd),
                        solve(r2, d2, t, q2, &mut rnd),
                    ) else {
                        continue;
                    };
                    let b1: Vec<u32> = a1.iter().map(|x| (*x as f32).to_bits()).collect();
                    let b2: Vec<u32> = a2.iter().map(|x| (*x as f32).to_bits()).collect();
                    let a1f: Vec<f64> = b1.iter().map(|b| f32::from_bits(*b) as f64).collect();
                    let (Ok(g1), Ok(g2)) = (p1.eval_with(&b1, &mut s1), p2.eval_with(&b2, &mut s2))
                    else {
                        continue;
                    };
                    let (g1, g2) = (f32::from_bits(g1) as f64, f32::from_bits(g2) as f64);
                    let f = r1.eval(&a1f);
                    if !g1.is_finite() || !g2.is_finite() || !f.is_finite() {
                        continue;
                    }
                    let k = ((g2 - g1) / ulp32(f)).round().clamp(-4.0, 4.0) as usize + 4;
                    if std::env::var("PTXV_PT_DEBUG").is_ok()
                        && e == -1
                        && sub == 0
                        && rnd.below(20) == 0
                    {
                        eprintln!(
                            "pt t {t:e} a1 {:?} a2 {:?} g1 {g1:e} g2 {g2:e} f {f:e} k {}",
                            a1f,
                            b2.iter().map(|b| f32::from_bits(*b)).collect::<Vec<_>>(),
                            k as i64 - 4
                        );
                    }
                    let c = cells
                        .entry(cell_of(index_value(r1, &a1f)))
                        .or_insert([0.0; 10]);
                    c[k] += 1.0;
                    c[9] += 1.0;
                }
            }
        }
    }
    let mut out: Vec<Cell> = cells
        .into_iter()
        .filter(|(_, c)| c[9] > 0.0)
        .map(|(k, c)| {
            let n = c[9];
            let mut off = [0f32; 9];
            for j in 0..9 {
                off[j] = (c[j] / n) as f32;
            }
            Cell {
                sign: k.0,
                binade: k.1,
                sub: k.2,
                n: n as u32,
                off,
                ..Default::default()
            }
        })
        .collect();
    out.sort_by_key(|c| (c.sign, c.binade, c.sub));
    let mut t = Table {
        name: "pair".into(),
        cells: out,
        index: HashMap::default(),
        range: range.to_vec(),
    };
    t.reindex();
    Some(t)
}

/// Arguments with index value t: the driver solved for, the others from a random probe.
fn solve(
    r: &RealFn,
    driver: usize,
    t: f64,
    probes: &[(Vec<f64>, f64)],
    rnd: &mut super::super::special::Rng,
) -> Option<Vec<f64>> {
    let q = &probes[rnd.below(probes.len() as u64) as usize].0;
    let mut a = q.clone();
    match r.func {
        Func::Div => {
            // a0 = t a1.
            a[0] = t * a[1];
        }
        Func::Poly | Func::Named(_) => a[driver] = t,
        _ => {
            let c = r.coef[driver];
            if c == 0.0 {
                return None;
            }
            let rest = r.z(&a) - c * a[driver];
            a[driver] = (t - rest) / c;
        }
    }
    a.iter().all(|x| x.is_finite()).then_some(a)
}

/// The table of a MUFU unit (with its operand modifiers) over the operand range of `probes`
/// (operand values).
pub fn mufu_table(
    name: &str,
    func: ptxv_ir::MufuFunc,
    ptx: bool,
    neg: bool,
    abs: bool,
    lo_hi: &[(i8, i16, i16)],
) -> Table {
    let mut t = Table {
        name: name.into(),
        cells: mufu_cells(func, ptx, neg, abs, lo_hi),
        index: HashMap::default(),
        range: lo_hi.to_vec(),
    };
    t.reindex();
    t
}

/// The cells of a MUFU table over the binade ranges `lo_hi`.
pub fn mufu_cells(
    func: ptxv_ir::MufuFunc,
    ptx: bool,
    neg: bool,
    abs: bool,
    lo_hi: &[(i8, i16, i16)],
) -> Vec<Cell> {
    let mut all: HashMap<(i8, i16, u8), Acc> = HashMap::default();
    let mut rnd = super::super::special::Rng::new(0x3f0f);
    for &(s, lo, hi) in lo_hi {
        for e in lo..=hi {
            for sub in 0..SUB {
                for _ in 0..PER_CELL {
                    let x = s as f64
                        * 2f64.powi(e as i32)
                        * (1.0 + (sub as f64 + rnd.unit()) / SUB as f64);
                    let xb = (x as f32).to_bits();
                    let op = super::ops::mufu_operand(xb, neg, abs);
                    let g = f32::from_bits(super::ops::mufu_hw(func, ptx, op)) as f64;
                    let f = crate::num::mufu_true(func, ptx, f32::from_bits(op) as f64);
                    all.entry(cell_of(f32::from_bits(xb) as f64))
                        .or_default()
                        .add(g, f);
                }
            }
        }
    }
    let mut cells: Vec<Cell> = all.into_iter().map(|(k, a)| a.cell(k)).collect();
    cells.sort_by_key(|c| (c.sign, c.binade, c.sub));
    cells
}

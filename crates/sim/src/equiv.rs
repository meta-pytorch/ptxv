/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Kernel equivalence (rung 2e): does a candidate kernel compute "the same thing" as a
//! reference, for every input of the captured shape?
//!
//! Each kernel runs once with the numerics shadow (`num.rs`). With the preconditions
//! (race-free and deadlock-free, so one output per input; control flow and addresses
//! independent of input data, so the one run's dataflow is the dataflow for every input),
//! each output element's rounding DAG is a function of the inputs, and its leaves are
//! identified by origin (input buffer, element index). Per output element:
//! - **L0 bitwise**: the canonical rounding DAGs are identical (operands sorted at commutative
//!   nodes, an MMA step as accumulator + multiset of products, which the calibrated MMA model
//!   makes order-invariant within its K group). Identical DAGs round identically for every
//!   input.
//! - **L1 same function**: the exact polynomials over the input leaves are equal (monomial ->
//!   coefficient maps compared exactly).
//! - **L2 no less accurate**: L1, and per monomial the candidate's first-order error multiplier
//!   (sum of the unit roundoffs of the rounding nodes on its paths) is at most the reference's.
//!   The a-priori bound sum_m |c_m x_m| d_m is monotone in it, so C's bound <= R's for all x.
//! - **L3 tolerance**: L1, and per monomial d_C + d_R <= k u, so |C(x) - R(x)| <= k u sum |terms|
//!   (to first order) for every x.
//!
//! Per output it also gives the **bound achieved** (docs/equiv-reassoc.md): with L1, the
//! fp_noise Cor 1a bound of each kernel, rigorous rather than first order (per monomial the sum
//! over its occurrences of |c| (prod (1 + eps_v) - 1) along the path), combined by Thm 4:
//! |C(x) - R(x)| <= k u sum_m a_m |x_m| (+ an absolute term for flushed subnormals), k the
//! largest (g_R + g_C) / a_m. This covers reassociated sums (linear, pairwise, shuffle trees,
//! split-K) without matching their trees, and fp32 atomics: a chain of S atomic adds on one
//! location is bounded for every order at once (each added value sits under at most S - 1
//! roundings when the location starts at +0, S otherwise: Cor 4b).
//!
//! A MUFU / approx node f(P) is a pseudo-leaf identified by (f, P's exact polynomial), with
//! the unit's error bound (N-MUFU) as its multiplier. L2 also compares, per pseudo-leaf, the
//! multipliers inside P (the error f' amplifies is the same function of x on both sides).
//! L3 is not proven when an f's operand carries rounding error (the bound would need |f'|)
//! or f has only an absolute error bound (lg2, sin, cos).
//!
//! ASSUME(N-EQUIV): see ASSUMPTIONS.md

use std::collections::BTreeMap;
use std::hash::Hash;
use std::hash::Hasher;

use ptxv_ir::MufuFunc;
use serde::Serialize;

use crate::num::NodeKind;
use crate::num::NumState;
use crate::num::U32;

/// An input leaf: (buffer id in the shared name table, element index).
pub type Leaf = (u16, u64);
/// A monomial: its leaves, sorted. The empty monomial is the constant term.
pub type Mono = Vec<Leaf>;

/// Coefficient and error multipliers of one monomial.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Term {
    pub coef: f64,
    pub(crate) coef_rounded: bool,
    /// Sum over occurrences of coef * (unit roundoffs of the rounding nodes on the path).
    pub mult: f64,
    /// Sum over occurrences of |coef|: the scale the rigorous bound is relative to.
    pub abs: f64,
    /// Sum over occurrences of |coef| * (prod over the path's rounding nodes of (1 + eps_v)
    /// - 1): Cor 1a's rigorous multiplier (`mult` is its first-order part).
    pub gam: f64,
}

impl Term {
    pub fn one() -> Term {
        Term {
            coef: 1.0,
            abs: 1.0,
            ..Default::default()
        }
    }
    /// Under `times` more rounding nodes of unit `eps` (on every occurrence).
    pub fn grown(self, eps: f64, times: usize) -> Term {
        let g = (1.0 + eps).powi(times as i32);
        Term {
            coef: self.coef,
            coef_rounded: self.coef_rounded,
            mult: self.mult + self.coef.abs() * eps * times as f64,
            abs: self.abs,
            gam: self.gam * g + self.abs * (g - 1.0),
        }
    }
}

/// Smallest subnormal-free magnitude: an FTZ operation errs by less than this absolutely.
const FTZ_MIN: f64 = 1.1754943508222875e-38; // 2^-126

/// Buffer id of MUFU pseudo-leaves.
pub const MUFU_LEAF: u16 = u16::MAX;
/// Buffer id of max / min pseudo-leaves (segment mode): identified by the kind and the set of
/// their operands' exact polynomials.
pub const MAX_LEAF: u16 = u16::MAX - 1;
/// Buffer id of slice pseudo-leaves (segment mode): identified by the slice's instructions
/// and its arguments' exact polynomials (`softmax.rs` compares different slices).
pub const SEG_LEAF: u16 = u16::MAX - 2;

/// A pseudo-leaf (MUFU, max, slice): no rigorous bound in the polynomial comparison.
pub fn pseudo(l: &Leaf) -> bool {
    l.0 >= SEG_LEAF
}

/// Shared nonlinear error budgets, evaluated with the original grouping of additions and maxima.
#[derive(Clone, Debug, Default)]
pub struct InnerBudget(Option<std::rc::Rc<InnerExpr>>);

#[derive(Debug)]
struct InnerExpr {
    op: InnerOp,
    value: std::cell::OnceCell<std::rc::Rc<BTreeMap<Mono, f64>>>,
    amplified: bool,
}

#[derive(Debug)]
enum InnerOp {
    Terms(std::rc::Rc<BTreeMap<Mono, f64>>),
    Add(InnerBudget, InnerBudget),
    Max(InnerBudget, InnerBudget),
}

impl InnerBudget {
    fn terms(terms: BTreeMap<Mono, f64>) -> Self {
        if terms.is_empty() {
            return Self::default();
        }
        let amplified = terms.values().any(|&v| v > 0.0);
        Self(Some(std::rc::Rc::new(InnerExpr {
            op: InnerOp::Terms(std::rc::Rc::new(terms)),
            value: Default::default(),
            amplified,
        })))
    }

    fn amplified(&self) -> bool {
        self.0.as_ref().is_some_and(|e| e.amplified)
    }

    fn merge(&mut self, other: &Self, maximum: bool) {
        if self.0.is_none() {
            *self = other.clone();
        } else if other.0.is_some() {
            *self = Self(Some(std::rc::Rc::new(InnerExpr {
                amplified: self.amplified() || other.amplified(),
                op: if maximum {
                    InnerOp::Max(self.clone(), other.clone())
                } else {
                    InnerOp::Add(self.clone(), other.clone())
                },
                value: Default::default(),
            })));
        }
    }

    // Keep shared DAG effects compact until a polynomial L2 comparison needs them.
    fn values(&self) -> std::rc::Rc<BTreeMap<Mono, f64>> {
        let Some(root) = &self.0 else {
            return Default::default();
        };
        if let Some(value) = root.value.get() {
            return value.clone();
        }
        let key = |e: &std::rc::Rc<InnerExpr>| std::rc::Rc::as_ptr(e) as usize;
        let mut uses = BTreeMap::<usize, usize>::new();
        let mut seen = std::collections::BTreeSet::new();
        let mut order = Vec::new();
        let mut values = BTreeMap::new();
        let mut pending = vec![(root.clone(), false)];
        while let Some((e, ready)) = pending.pop() {
            let id = key(&e);
            if ready {
                order.push(e);
                continue;
            }
            if !seen.insert(id) {
                continue;
            }
            if let Some(value) = e.value.get() {
                values.insert(id, value.clone());
                continue;
            }
            if let InnerOp::Terms(terms) = &e.op {
                values.insert(id, terms.clone());
                continue;
            }
            pending.push((e.clone(), true));
            if let InnerOp::Add(a, b) | InnerOp::Max(a, b) = &e.op {
                for child in [b, a] {
                    if let Some(child) = &child.0 {
                        *uses.entry(key(child)).or_default() += 1;
                        pending.push((child.clone(), false));
                    }
                }
            }
        }
        for e in order {
            let (a, b) = match &e.op {
                InnerOp::Add(a, b) | InnerOp::Max(a, b) => (a, b),
                InnerOp::Terms(_) => unreachable!(),
            };
            let mut take = |x: &InnerBudget| {
                let Some(x) = &x.0 else {
                    return std::rc::Rc::default();
                };
                let id = key(x);
                let count = uses.get_mut(&id).expect("budget child has a parent");
                *count -= 1;
                if *count == 0 {
                    values.remove(&id).expect("evaluated budget child")
                } else {
                    values.get(&id).expect("evaluated budget child").clone()
                }
            };
            let left = take(a);
            let right = take(b);
            let mut into = std::rc::Rc::unwrap_or_clone(left);
            for (m, v) in right.iter() {
                let old = into.entry(m.clone()).or_default();
                *old = if matches!(e.op, InnerOp::Max(..)) {
                    old.max(*v)
                } else {
                    *old + v
                };
            }
            values.insert(key(&e), std::rc::Rc::new(into));
        }
        root.value
            .set(values.remove(&key(root)).expect("evaluated budget root"))
            .expect("budget evaluated once");
        root.value.get().expect("evaluated budget root").clone()
    }
}

#[derive(Clone, Debug, Default)]
pub struct Canon {
    pub l0: (u64, u64),
    /// The L0 hash with a final `x + (+0)` peeled (cuBLAS's alpha = 1, beta = 0 epilogue):
    /// equal forms then give equal outputs except that -0 becomes +0. `None`: nothing peeled.
    pub l0_z: Option<(u64, u64)>,
    pub poly: BTreeMap<Mono, Term>,
    /// Why the canonical form is not a function of the inputs alone (an untracked leaf).
    pub untracked: Option<String>,
    /// Exact polynomial comparison lost coefficient bits; a separate norm bound may cover them.
    coefficient_loss: Option<String>,
    /// Per nonlinear operand id: shared error multipliers of its monomials.
    pub inner: BTreeMap<u64, InnerBudget>,
    /// Some MUFU node has only an absolute error bound.
    pub abs_bound: bool,
    /// Why the rounding order is schedule-dependent (fp32 atomic chains): not L0.
    pub unordered: Option<String>,
    /// Absolute error term of FTZ operations (atomics), independent of the inputs.
    pub abs_extra: f64,
    /// Some product (segment mode) could underflow: the bound assumes it does not.
    pub products: bool,
    /// A bf16 rounding is in the DAG (its subnormal absolute error is in `abs_extra`).
    pub round_abs: bool,
    /// ... and enters a product, which scales that absolute error: no polynomial bound.
    pub round_in_product: bool,
    /// A TF32 MMA operand (N-TF32): the bound assumes every f32 value read as one, or split
    /// into TF32 parts, is zero or at least 2^-103 in magnitude.
    pub tf32: bool,
}

/// Relative error of hi(x), the TF32 truncation of a normal f32 x: < 2^-10 (N-TF32).
pub const TF32_HI_EPS: f64 = 1.0 / 1024.0;

/// The precondition a TF32 MMA operand adds to a bound (N-TF32).
pub const TF32_COND: &str = "N-TF32: every f32 value read as a TF32 MMA operand or split into TF32 parts is 0 or at least 2^-103 in magnitude";

/// L0 hash of a `Tf32` node from its argument's.
fn tf32_hash(part: crate::num::Tf32Part, ha: (u64, u64)) -> (u64, u64) {
    let mut h = H2::new();
    h.put(("tf32", part, ha));
    h.done()
}

/// Unit roundoff of bf16 (8-bit significand), round to nearest.
pub const BF16_U: f64 = 1.0 / 256.0;
/// Absolute error of a bf16 rounding to a subnormal (half its spacing 2^-133).
pub const BF16_SUB: f64 = 4.591774807899561e-41; // 2^-134

/// The element index of a leaf for input byte `elem` read as narrow format `t` (`halfin`
/// `byte_leaf`): the format in bits 56-59.
pub fn leaf_tagged(elem: u64, t: ptxv_ir::Lp) -> u64 {
    elem | (1 + t as u64) << 56
}

/// The (format, byte index) of a tagged leaf index, None for a plain element index.
pub fn leaf_lp(idx: u64) -> Option<(ptxv_ir::Lp, u64)> {
    use ptxv_ir::Lp;
    let t = match idx >> 56 {
        0 => return None,
        k => [
            Lp::E4m3,
            Lp::E5m2,
            Lp::E2m3,
            Lp::E3m2,
            Lp::E2m1,
            Lp::Ue8m0,
            Lp::Ue4m3,
            Lp::F16,
            Lp::Bf16,
        ]
        .get(k as usize - 1)
        .copied()?,
    };
    Some((t, idx & ((1 << 56) - 1)))
}

/// The code of narrow-format leaf element `k` of format `t` from its buffer's bytes: byte k,
/// or for FP4 (two codes per byte, low nibble first) nibble k % 2 of byte k / 2.
pub fn leaf_code(t: ptxv_ir::Lp, k: u64, byte: impl Fn(u64) -> Option<u32>) -> Option<u32> {
    if t == ptxv_ir::Lp::E2m1 {
        byte(k / 2).map(|b| (b >> (4 * (k % 2))) & 0xf)
    } else {
        byte(k)
    }
}

/// (unit roundoff, absolute error below the normal range) of a rounding to bf16 or f16
/// (`f16`), to nearest even: |rnd(x) - x| <= u |x| and <= a where rnd(x) is subnormal.
pub fn round_unit(f16: bool) -> (f64, f64) {
    if f16 {
        (2f64.powi(-11), 2f64.powi(-25))
    } else {
        (BF16_U, BF16_SUB)
    }
}

/// One side of a comparison: the numerics state of its run and its output nodes.
pub struct Side<'a> {
    pub ns: &'a NumState,
    /// Output element -> (hardware bits, node).
    pub outs: Vec<(u32, Option<u32>)>,
    /// Allocation index -> id in the shared buffer-name table.
    pub buf_id: Vec<u16>,
}

/// Per node: (hash, polynomial, effects on the canonical form).
type Memo = std::collections::HashMap<u32, std::rc::Rc<((u64, u64), BTreeMap<Mono, Term>, Canon)>>;

/// Add a sub-DAG's recorded effects to a canonical form (each visit counts, as before).
fn merge_effects(cx: &mut Canon, sub: &Canon) {
    if cx.untracked.is_none() {
        cx.untracked = sub.untracked.clone();
    }
    if cx.coefficient_loss.is_none() {
        cx.coefficient_loss = sub.coefficient_loss.clone();
    }
    if cx.unordered.is_none() {
        cx.unordered = sub.unordered.clone();
    }
    for (id, m) in &sub.inner {
        match cx.inner.entry(*id) {
            std::collections::btree_map::Entry::Vacant(e) => {
                e.insert(m.clone());
            }
            std::collections::btree_map::Entry::Occupied(mut e) => {
                e.get_mut().merge(m, false);
            }
        }
    }
    cx.abs_bound |= sub.abs_bound;
    cx.abs_extra += sub.abs_extra;
    cx.products |= sub.products;
    cx.round_abs |= sub.round_abs;
    cx.round_in_product |= sub.round_in_product;
    cx.tf32 |= sub.tf32;
}

struct H2(
    std::collections::hash_map::DefaultHasher,
    std::collections::hash_map::DefaultHasher,
);

impl H2 {
    fn new() -> Self {
        let mut b = std::collections::hash_map::DefaultHasher::new();
        0x5eed_u64.hash(&mut b);
        H2(std::collections::hash_map::DefaultHasher::new(), b)
    }
    fn put<T: Hash>(&mut self, v: T) {
        v.hash(&mut self.0);
        v.hash(&mut self.1);
    }
    fn done(&self) -> (u64, u64) {
        (self.0.finish(), self.1.finish())
    }
}

fn add_poly(into: &mut BTreeMap<Mono, Term>, from: &BTreeMap<Mono, Term>, eps: f64) {
    add_poly_n(into, from, eps, 1);
}

/// A set of exactly representable values: N 2^L with |N| < 2^s, L >= lsb, |value| < 2^top
/// (`Side::exact_set`). A format holds every member when it has at least s significand bits,
/// its smallest subnormal spacing is at most 2^lsb and every value below 2^top with s bits is
/// finite in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExactSet {
    pub s: i32,
    pub lsb: i32,
    pub top: i32,
}

impl ExactSet {
    /// f32 (24 bits, spacing 2^-149, finite below 2^128), bf16, f16 (finite below 2^16:
    /// 65504 is the largest 11-bit value there).
    pub const F32: ExactSet = ExactSet {
        s: 24,
        lsb: -149,
        top: 128,
    };
    pub const BF16: ExactSet = ExactSet {
        s: 8,
        lsb: -133,
        top: 128,
    };
    pub const F16: ExactSet = ExactSet {
        s: 11,
        lsb: -24,
        top: 16,
    };

    /// The finite codes of a narrow format: s = mantissa bits + 1, lsb = the subnormal spacing
    /// 1 - bias - mantissa bits, top = 1 + the exponent of the largest finite value (code
    /// `max`). ue8m0 has no subnormals: 2^-127 .. 2^127.
    pub fn of(t: ptxv_ir::Lp) -> ExactSet {
        if t == ptxv_ir::Lp::Ue8m0 {
            return ExactSet {
                s: 1,
                lsb: -127,
                top: 128,
            };
        }
        let f = crate::lowp::fmt(t);
        ExactSet {
            s: f.mbits as i32 + 1,
            lsb: 1 - f.bias - f.mbits as i32,
            top: crate::lowp::value(f.max, t).log2().floor() as i32 + 1,
        }
    }

    pub fn mul(self, o: ExactSet) -> ExactSet {
        ExactSet {
            s: self.s + o.s,
            lsb: self.lsb + o.lsb,
            top: self.top + o.top,
        }
    }

    pub fn fits(self, f: ExactSet) -> bool {
        self.s <= f.s && self.lsb >= f.lsb && self.top <= f.top
    }
}

// NaN marks a coefficient whose exact value would be lost by f64 arithmetic.
fn coefficient_sum(a: f64, b: f64) -> f64 {
    let sum = a + b;
    let bb = sum - a;
    let residual = (a - (sum - bb)) + (b - bb);
    if sum.is_finite() && residual == 0.0 {
        sum
    } else {
        f64::NAN
    }
}

fn coefficient_product(a: f64, b: f64) -> f64 {
    let product = a * b;
    // Below this range the residual of a two-product can itself underflow.
    let residual_safe =
        a == 0.0 || b == 0.0 || product.abs() >= f64::MIN_POSITIVE * 9007199254740992.0;
    if product.is_finite() && residual_safe && a.mul_add(b, -product) == 0.0 {
        product
    } else {
        f64::NAN
    }
}

/// `into += from`, every term of `from` under `times` more rounding nodes of unit `eps`.
pub(crate) fn add_poly_n(
    into: &mut BTreeMap<Mono, Term>,
    from: &BTreeMap<Mono, Term>,
    eps: f64,
    times: usize,
) {
    for (m, t) in from {
        let t = t.grown(eps, times);
        let e = into.entry(m.clone()).or_default();
        e.coef = coefficient_sum(e.coef, t.coef);
        e.mult += t.mult;
        e.abs += t.abs;
        e.gam += t.gam;
    }
}

/// Product of two polynomials (Cor 1a multipliers: (1 + g_a)(1 + g_b) - 1 per pair).
pub(crate) fn poly_mul(a: &BTreeMap<Mono, Term>, b: &BTreeMap<Mono, Term>) -> BTreeMap<Mono, Term> {
    let mut out: BTreeMap<Mono, Term> = BTreeMap::new();
    for (ma, ta) in a {
        for (mb, tb) in b {
            let mut m: Mono = ma.iter().chain(mb).copied().collect();
            m.sort();
            let e = out.entry(m).or_default();
            e.coef = coefficient_sum(e.coef, coefficient_product(ta.coef, tb.coef));
            e.mult += tb.coef.abs() * ta.mult + ta.coef.abs() * tb.mult;
            e.abs += ta.abs * tb.abs;
            e.gam += (ta.abs + ta.gam) * (tb.abs + tb.gam) - ta.abs * tb.abs;
        }
    }
    out
}

fn poly_id(h: &mut H2, p: &BTreeMap<Mono, Term>) {
    h.put(("polynomial", p.values().filter(|t| t.coef != 0.0).count()));
    for (m, t) in p {
        if t.coef != 0.0 {
            h.put((m, t.coef.to_bits()));
        }
    }
}

impl Side<'_> {
    /// Canonical form of output element `i`.
    pub fn canon(&self, i: usize) -> Canon {
        self.canon_with(i, &mut Memo::new())
    }

    /// `canon` sharing the forms of nodes already visited (`compare`: outputs of one row share
    /// their statistics' DAG).
    fn canon_with(&self, i: usize, memo: &mut Memo) -> Canon {
        let (bits, node) = self.outs[i];
        let mut c = Canon::default();
        match node {
            None => {
                // An exact value with no rounding history: a constant unless it moved an input.
                let mut h = H2::new();
                h.put(("const", bits));
                c.l0 = h.done();
                let v = f32::from_bits(bits) as f64;
                if v != 0.0 {
                    c.poly.insert(
                        vec![],
                        Term {
                            coef: v,
                            abs: v.abs(),
                            ..Default::default()
                        },
                    );
                    c.untracked = Some(format!("output {i} has no dataflow history (value {v})"));
                }
            }
            Some(n) => {
                let (l0, poly) = self.node(n, &mut c, memo);
                c.l0 = l0;
                c.poly = poly;
                if let NodeKind::Add(a, b, false) = self.ns.nodes[n as usize].kind {
                    let zero = |x: u32| {
                        let x = &self.ns.nodes[x as usize];
                        matches!(x.kind, NodeKind::Leaf) && x.hw == 0
                    };
                    let x = if zero(b) {
                        Some(a)
                    } else if zero(a) {
                        Some(b)
                    } else {
                        None
                    };
                    if let Some(x) = x {
                        // `node` is memoized: the operand's effects are already merged.
                        c.l0_z = Some(self.node(x, &mut Canon::default(), memo).0);
                    }
                }
            }
        }
        // A cancelled monomial (coefficient 0) keeps its error multipliers.
        c.poly.retain(|_, t| t.coef != 0.0 || t.gam != 0.0);
        c
    }

    /// The values node `n` can take for every input, when they are exactly representable
    /// numbers N 2^L with |N| < 2^s, L >= lsb and |value| < 2^top (`ExactSet`): narrow-format
    /// input leaves (their format's codes), and an f32 product of two such (an FMUL with no
    /// addend) or a 16-bit rounding of one, when that operation is exact on the whole set.
    /// ASSUME(N-EXACT): see ASSUMPTIONS.md
    pub(crate) fn exact_set(&self, n: u32) -> Option<ExactSet> {
        match &self.ns.nodes[n as usize].kind {
            NodeKind::Input((_, e)) => leaf_lp(*e).map(|(t, _)| ExactSet::of(t)),
            NodeKind::Fma { a, b, c: None, .. } => {
                let p = self.exact_set(*a)?.mul(self.exact_set(*b)?);
                p.fits(ExactSet::F32).then_some(p)
            }
            NodeKind::Round { arg, f16 } => {
                let x = self.exact_set(*arg)?;
                x.fits(if *f16 { ExactSet::F16 } else { ExactSet::BF16 })
                    .then_some(x)
            }
            _ => None,
        }
    }

    /// Node `n`'s hash and polynomial; its effects on the canonical form (untracked leaves,
    /// MUFU operand multipliers, FTZ terms, ...) are recorded with it and merged on every visit.
    fn node(&self, n: u32, cx: &mut Canon, memo: &mut Memo) -> ((u64, u64), BTreeMap<Mono, Term>) {
        if let Some(e) = memo.get(&n) {
            merge_effects(cx, &e.2);
            return (e.0, e.1.clone());
        }
        let mut sub = Canon::default();
        let (h, p) = self.node_inner(n, &mut sub, memo);
        if p.values().any(|t| !t.coef.is_finite()) {
            sub.coefficient_loss.get_or_insert_with(|| {
                format!(
                    "polynomial coefficient cannot be represented exactly in f64 at pc {}",
                    self.ns.nodes[n as usize].pc
                )
            });
        }
        merge_effects(cx, &sub);
        memo.insert(n, std::rc::Rc::new((h, p.clone(), sub)));
        (h, p)
    }

    fn node_inner(
        &self,
        n: u32,
        cx: &mut Canon,
        memo: &mut Memo,
    ) -> ((u64, u64), BTreeMap<Mono, Term>) {
        let nd = &self.ns.nodes[n as usize];
        let mut h = H2::new();
        let mut poly = BTreeMap::new();
        match &nd.kind {
            NodeKind::Opaque => {
                h.put(("opaque", n));
                let v = f32::from_bits(nd.hw) as f64;
                poly.insert(
                    vec![],
                    Term {
                        coef: v,
                        abs: v.abs(),
                        ..Default::default()
                    },
                );
                cx.untracked.get_or_insert_with(|| {
                    format!(
                        "an input-derived value without a node or slice ({v}) at pc {}",
                        nd.pc
                    )
                });
            }
            NodeKind::Leaf => {
                let v = f32::from_bits(nd.hw) as f64;
                h.put(("leaf", nd.hw));
                // Segment mode: leaves are input-independent constants.
                if v != 0.0 && self.ns.segments {
                    poly.insert(
                        vec![],
                        Term {
                            coef: v,
                            abs: v.abs(),
                            ..Default::default()
                        },
                    );
                } else if v != 0.0 {
                    poly.insert(
                        vec![],
                        Term {
                            coef: v,
                            abs: v.abs(),
                            ..Default::default()
                        },
                    );
                    cx.untracked.get_or_insert_with(|| {
                        format!("a value of untracked origin ({v}) enters at pc {}", nd.pc)
                    });
                }
            }
            // ASSUME(N-LEAF): see ASSUMPTIONS.md
            NodeKind::Input((a, e)) => {
                let leaf = (self.buf_id[*a], *e);
                h.put(("input", leaf));
                poly.insert(vec![leaf], Term::one());
            }
            NodeKind::AtomAdd(..) => {
                // The whole chain on this location: its added values, in this run's order, and
                // the location's value before the first.
                let mut vals = Vec::new();
                let mut cur = n;
                while let NodeKind::AtomAdd(o, x) = self.ns.nodes[cur as usize].kind {
                    vals.push(x);
                    cur = o;
                }
                let base = &self.ns.nodes[cur as usize];
                let base_zero = matches!(base.kind, NodeKind::Leaf) && base.hw == 0;
                let s = vals.len();
                // ASSUME(N-ATOMIC): in any order, a value added to +0 first is exact, so each
                // value sits under at most S - 1 roundings (S with a nonzero start).
                let d = if base_zero { s - 1 } else { s };
                let mut hs = Vec::with_capacity(s);
                for &x in &vals {
                    let (hx, px) = self.node(x, cx, memo);
                    hs.push(hx);
                    add_poly_n(&mut poly, &px, U32, d);
                }
                let hb = if base_zero {
                    (0, 0)
                } else {
                    let (hb, pb) = self.node(cur, cx, memo);
                    add_poly_n(&mut poly, &pb, U32, d);
                    hb
                };
                hs.sort();
                h.put(("atom", hb, hs));
                if s > if base_zero { 2 } else { 1 } {
                    cx.unordered.get_or_insert_with(|| {
                        format!(
                            "{s} fp32 atomic adds into one location at pc {} (their order depends on the schedule)",
                            nd.pc
                        )
                    });
                }
                // FTZ: per add, two flushed operands and a flushed result, each < 2^-126,
                // carried through the later adds (as the values' own absolute parts).
                cx.abs_extra = cx.abs_extra * (1.0 + U32).powi(d as i32)
                    + s as f64 * 3.0 * FTZ_MIN * (1.0 + U32).powi(s as i32);
            }
            NodeKind::Add(a, b, rz) => {
                let (ha, pa) = self.node(*a, cx, memo);
                let (hb, pb) = self.node(*b, cx, memo);
                let mut ch = [ha, hb];
                ch.sort();
                h.put(("add", *rz, ch));
                let zero = |c: &u32| {
                    let x = &self.ns.nodes[*c as usize];
                    matches!(x.kind, NodeKind::Leaf) && x.hw == 0
                };
                // ASSUME(N-FADD): a round-to-nearest add errs by at most u relative, round toward
                // zero by 2u. Adding the constant +0 is exact (x, or +0 for x = -0).
                let eps = if zero(a) || zero(b) {
                    0.0
                } else if *rz {
                    2.0 * U32
                } else {
                    U32
                };
                add_poly(&mut poly, &pa, eps);
                add_poly(&mut poly, &pb, eps);
                // An absolute error of an operand passes through this rounding: (1 + eps).
                cx.abs_extra *= 1.0 + eps;
            }
            NodeKind::Fma {
                a,
                b,
                c,
                neg_ab,
                neg_c,
            } => {
                let (ha, pa) = self.node(*a, cx, memo);
                let (hb, pb) = self.node(*b, cx, memo);
                // Absolute error terms (bf16 roundings to subnormals) of the factors: one
                // factor an exact constant k scales the other's by |k|; two inexact factors
                // have no polynomial bound.
                let abs_of = |k: &u32| memo.get(k).map_or(0.0, |e| e.2.abs_extra);
                let konst = |p: &BTreeMap<Mono, Term>, k: &u32| -> Option<f64> {
                    (abs_of(k) == 0.0 && p.iter().all(|(m, t)| m.is_empty() && t.gam == 0.0))
                        .then(|| p.get(&vec![]).map_or(0.0, |t| t.coef.abs()))
                };
                let (aa, ab) = (abs_of(a), abs_of(b));
                let abs_ab = match (konst(&pa, a), konst(&pb, b)) {
                    _ if aa == 0.0 && ab == 0.0 => Some(0.0),
                    (_, Some(kb)) => Some(kb * aa),
                    (Some(ka), _) => Some(ka * ab),
                    _ => None,
                };
                if abs_ab.is_none() {
                    cx.round_in_product = true;
                }
                let mut ch = [ha, hb];
                ch.sort();
                let mut prod = poly_mul(&pa, &pb);
                if *neg_ab {
                    prod.values_mut().for_each(|t| t.coef = -t.coef);
                }
                let hc = match c {
                    Some(c) => {
                        let (hc, mut pc) = self.node(*c, cx, memo);
                        if *neg_c {
                            pc.values_mut().for_each(|t| t.coef = -t.coef);
                        }
                        add_poly_n(&mut prod, &pc, 0.0, 0);
                        Some(hc)
                    }
                    None => None,
                };
                h.put(("fma", *neg_ab, *neg_c, ch, hc));
                // ASSUME(N-REASSOC): one rounding, relative u unless the result underflows;
                // none where the product is exact for every input (N-EXACT).
                let eps = if c.is_none() && self.exact_set(n).is_some() {
                    0.0
                } else {
                    U32
                };
                if let Some(x) = abs_ab {
                    let ac = c.map_or(0.0, |c| memo.get(&c).map_or(0.0, |e| e.2.abs_extra));
                    cx.abs_extra = (x + ac) * (1.0 + eps);
                }
                if prod.len() > 100_000 {
                    cx.untracked.get_or_insert_with(|| {
                        format!("a product of more than 100000 monomials at pc {}", nd.pc)
                    });
                }
                add_poly(&mut poly, &prod, eps);
                cx.products = true;
            }
            NodeKind::Max { min, .. } => {
                // The set of operands of nested maxima of one kind, without the identity
                // (-inf for max, +inf for min).
                let ident = if *min { 0x7f80_0000 } else { 0xff80_0000 };
                let mut kids = Vec::new();
                let mut stack = vec![n];
                while let Some(x) = stack.pop() {
                    match &self.ns.nodes[x as usize].kind {
                        NodeKind::Max { a, b, min: m } if m == min => {
                            stack.push(*a);
                            stack.push(*b);
                        }
                        NodeKind::Leaf if self.ns.nodes[x as usize].hw == ident => {}
                        _ => kids.push(x),
                    }
                }
                let mut parts: Vec<((u64, u64), BTreeMap<Mono, Term>)> =
                    kids.iter().map(|&k| self.node(k, cx, memo)).collect();
                parts.sort_by_key(|p| p.0);
                parts.dedup_by_key(|p| p.0);
                h.put(("max", *min, parts.iter().map(|p| p.0).collect::<Vec<_>>()));
                let mut id = H2::new();
                id.put(("max", *min));
                let mut polys: Vec<Vec<(Mono, u64)>> = parts
                    .iter()
                    .map(|p| {
                        p.1.iter()
                            .filter(|(_, t)| t.coef != 0.0)
                            .map(|(m, t)| (m.clone(), t.coef.to_bits()))
                            .collect()
                    })
                    .collect();
                polys.sort();
                polys.dedup();
                id.put(&polys);
                for (_, operand) in &parts {
                    let mut inner_id = H2::new();
                    inner_id.put(("max-operand", *min));
                    poly_id(&mut inner_id, operand);
                    let mut terms = BTreeMap::new();
                    for (m, term) in operand {
                        terms.insert(m.clone(), 0.0f64.max(term.mult));
                    }
                    cx.inner
                        .entry(inner_id.done().0)
                        .or_default()
                        .merge(&InnerBudget::terms(terms), true);
                }
                poly.insert(vec![(MAX_LEAF, id.done().0)], Term::one());
            }
            NodeKind::Sel { pred, neg, a, b } => {
                let mut id = H2::new();
                id.put(("sel", *neg));
                for o in &pred.0.ops {
                    id.put(format!("{:?}", o.kind));
                }
                id.put(format!("{:?}", pred.0.out));
                id.put(("arguments", pred.0.args.len() + 2));
                for &x in pred.0.args.iter().chain([a, b]) {
                    let (hx, px) = self.node(x, cx, memo);
                    h.put(hx);
                    id.put(hx);
                    poly_id(&mut id, &px);
                }
                let id = id.done().0;
                h.put(("sel", id));
                poly.insert(vec![(SEG_LEAF, id)], Term::one());
            }
            NodeKind::Seg(p) => {
                let mut id = H2::new();
                for o in &p.0.ops {
                    id.put(format!("{:?}", o.kind));
                }
                id.put(format!("{:?}", p.0.out));
                id.put(("arguments", p.0.args.len()));
                for &a in &p.0.args {
                    let (ha, pa) = self.node(a, cx, memo);
                    h.put(ha);
                    id.put(ha);
                    poly_id(&mut id, &pa);
                }
                let id = id.done().0;
                h.put(("seg", id));
                poly.insert(vec![(SEG_LEAF, id)], Term::one());
            }
            NodeKind::Mma { ops, sc, .. }
                if sc.is_empty()
                    && ops
                        .iter()
                        .flatten()
                        .any(|&(p, q)| self.is_tf32(p) || self.is_tf32(q)) =>
            {
                return self.tf32_chain(n, cx, memo);
            }
            NodeKind::Tf32 { arg, part } => {
                let (ha, pa) = self.node(*arg, cx, memo);
                let hh = tf32_hash(*part, ha);
                cx.tf32 = true;
                if *part == crate::num::Tf32Part::Hi {
                    // ASSUME(N-TF32): hi(x) = x (1 + d), -2^-10 < d <= 0, for x zero or normal.
                    add_poly(&mut poly, &pa, TF32_HI_EPS);
                    cx.abs_extra *= 1.0 + TF32_HI_EPS;
                } else {
                    cx.untracked.get_or_insert_with(|| {
                        format!(
                            "a TF32 low part outside a matched 3xTF32 product at pc {}",
                            nd.pc
                        )
                    });
                }
                return (hh, poly);
            }
            NodeKind::Mma {
                acc,
                terms,
                ops,
                eps,
                sc,
                units,
            } => {
                // ASSUME(N-MMA): the step's own contract (`MmaModel::eps_terms`).
                let eps = *eps;
                // An accumulator that is the constant +0 (zeroed TMEM, Triton) is the same as
                // none (enable-input-d = 0, CuTeDSL): the model computes 0 + terms either way.
                let zero = |a: &u32| {
                    let x = &self.ns.nodes[*a as usize];
                    matches!(x.kind, NodeKind::Leaf) && x.hw == 0
                };
                let ha = match acc.filter(|a| !zero(a)) {
                    Some(a) => {
                        let (ha, pa) = self.node(a, cx, memo);
                        add_poly(&mut poly, &pa, eps);
                        ha
                    }
                    None => (0, 0),
                };
                let mut th: Vec<(u64, u64)> = Vec::with_capacity(terms.len());
                for (i, (t, o)) in terms.iter().zip(ops).enumerate() {
                    let mut x = H2::new();
                    match o {
                        Some((p, q)) => {
                            let (hp, pp) = self.node(*p, cx, memo);
                            let (hq, pq) = self.node(*q, cx, memo);
                            if [p, q]
                                .iter()
                                .any(|k| memo.get(k).is_some_and(|e| e.2.round_abs))
                            {
                                cx.round_in_product = true;
                            }
                            let mut hh = [hp, hq];
                            hh.sort();
                            x.put(("prod", hh));
                            let mut prod = poly_mul(&pp, &pq);
                            // Block-scaled: times the two scale factors (exact leaves).
                            if let Some(&(u, v)) = sc.get(i) {
                                let (hu, pu) = self.node(u, cx, memo);
                                let (hv, pv) = self.node(v, cx, memo);
                                let mut hs = [hu, hv];
                                hs.sort();
                                x.put(("scale", hs));
                                prod = poly_mul(&prod, &poly_mul(&pu, &pv));
                            }
                            add_poly(&mut poly, &prod, eps);
                        }
                        None => {
                            x.put(("value", t.to_bits()));
                            if *t != 0.0 {
                                let e = poly.entry(vec![]).or_default();
                                e.coef += t;
                                e.mult += t.abs() * eps;
                                e.abs += t.abs();
                                e.gam += t.abs() * eps;
                                cx.untracked.get_or_insert_with(|| {
                                    format!(
                                        "an MMA product of untracked origin ({t}) at pc {}",
                                        nd.pc
                                    )
                                });
                            }
                        }
                    }
                    th.push(x.done());
                }
                if units.is_empty() {
                    th.sort();
                    h.put(("mma", ha, th));
                } else {
                    // FP4 kinds: invariant under permuting a unit's terms and whole units, not
                    // terms across units (N-MMA-BS): a multiset of unit multisets.
                    let mut by: BTreeMap<u8, Vec<(u64, u64)>> = BTreeMap::new();
                    for (u, x) in units.iter().zip(th) {
                        by.entry(*u).or_default().push(x);
                    }
                    let mut us: Vec<Vec<(u64, u64)>> = by
                        .into_values()
                        .map(|mut v| {
                            v.sort();
                            v
                        })
                        .collect();
                    us.sort();
                    h.put(("mma-units", ha, us));
                }
                // An f32 result below the normal range: rounded toward zero, < 2^-149 off.
                cx.abs_extra = cx.abs_extra * (1.0 + eps) + crate::mma_lp::F32_RZ_SUB;
            }
            NodeKind::Round { arg, f16 } => {
                // ASSUME(N-ROUND): round to nearest even to bf16 (f16): relative 2^-8 (2^-11),
                // plus 2^-134 (2^-25) absolute where the result is subnormal.
                let (u16_, a16) = round_unit(*f16);
                let (ha, pa) = self.node(*arg, cx, memo);
                h.put(("round16", *f16, ha));
                // N-EXACT: a rounding of a value exactly representable for every input is
                // the identity.
                if self.exact_set(n).is_some() {
                    add_poly(&mut poly, &pa, 0.0);
                } else {
                    add_poly(&mut poly, &pa, u16_);
                    cx.abs_extra = cx.abs_extra * (1.0 + u16_) + a16;
                    cx.round_abs = true;
                }
            }
            NodeKind::Mufu {
                arg,
                func,
                ptx,
                neg,
                abs,
            } => {
                let (ha, pa) = self.node(*arg, cx, memo);
                h.put(("mufu", *func, *ptx, *neg, *abs, ha));
                // Pseudo-leaf identity: the function and its operand's exact polynomial.
                let mut id = H2::new();
                id.put((*func, *ptx, *neg, *abs));
                for (m, t) in &pa {
                    if t.coef != 0.0 {
                        id.put((m, t.coef.to_bits()));
                    }
                }
                let id = id.done().0;
                // ASSUME(N-MUFU): the unit's sweep-measured error bound.
                // The x-dependent PTX sin/cos bound at its smallest (they are abs-bounded:
                // L3 is refused for them below).
                let b = crate::mufu::error_bound(*func, *ptx, 0.0);
                cx.abs_bound |= matches!(func, MufuFunc::Lg2 | MufuFunc::Sin | MufuFunc::Cos);
                let eps = b.rel_ulps * 2f64.powi(-23) + b.abs;
                let mut terms = BTreeMap::new();
                for (m, t) in &pa {
                    terms.insert(m.clone(), t.mult);
                }
                cx.inner
                    .entry(id)
                    .or_default()
                    .merge(&InnerBudget::terms(terms), false);
                poly.insert(
                    vec![(MUFU_LEAF, id)],
                    Term {
                        coef: 1.0,
                        mult: eps,
                        abs: 1.0,
                        gam: eps,
                        coef_rounded: false,
                    },
                );
            }
        }
        (h.done(), poly)
    }
}

impl Side<'_> {
    fn is_tf32(&self, n: u32) -> bool {
        matches!(self.ns.nodes[n as usize].kind, NodeKind::Tf32 { .. })
    }

    /// A TF32 MMA node and the chain of TF32 MMA steps under it (accumulator links: each step's
    /// value is a term of the next with coefficient 1), as one polynomial (N-TF32). A product
    /// hi(a) hi(b) whose two 3xTF32 corrections lo'(a) hi(b) and hi(a) lo'(b) (a and b with the
    /// same canonical hashes) are also on the chain is one term a b (1 + theta), |theta| <= 2^-20 + 2^-10
    /// (d_a + d_b), entering at the last of the three steps; the steps between the first and
    /// the last round the products apart, which costs (1 + w_a + w_b)(h - 1), w = 2^-10 (1 + d),
    /// h their growth (docs/equiv-mma.md §6). Unmatched hi(a) hi(b) products are hi(a) times hi(b); unmatched corrections are
    /// untracked. Hashes and the per-step growth are those of the step-by-step form.
    fn tf32_chain(
        &self,
        top: u32,
        cx: &mut Canon,
        memo: &mut Memo,
    ) -> ((u64, u64), BTreeMap<Mono, Term>) {
        use crate::num::Tf32Part;
        let nodes = &self.ns.nodes;
        let tf32_step = |k: u32| match &nodes[k as usize].kind {
            NodeKind::Mma { ops, sc, .. } => {
                sc.is_empty()
                    && ops
                        .iter()
                        .flatten()
                        .any(|&(p, q)| self.is_tf32(p) || self.is_tf32(q))
            }
            _ => false,
        };
        let mut chain = vec![top];
        while let NodeKind::Mma { acc: Some(a), .. } = nodes[*chain.last().unwrap() as usize].kind {
            if !tf32_step(a) {
                break;
            }
            chain.push(a);
        }
        let zero =
            |a: u32| matches!(nodes[a as usize].kind, NodeKind::Leaf) && nodes[a as usize].hw == 0;
        let base = match nodes[*chain.last().unwrap() as usize].kind {
            NodeKind::Mma { acc: Some(a), .. } if !zero(a) => Some(a),
            _ => None,
        };
        let part = |k: u32| match nodes[k as usize].kind {
            NodeKind::Tf32 { arg, part } => Some((arg, part)),
            _ => None,
        };
        // Classify every term: (step, term index) -> kind.
        enum Tk {
            Gen,
            Main(u32, u32),
            Corr,
            Value(f64),
        }
        // Operands are identified by their canonical hash: equal hashes, identical rounding
        // DAGs, the same value for every input (one value loaded twice gives two nodes).
        type H = (u64, u64);
        let mut kinds: Vec<Vec<Tk>> = Vec::with_capacity(chain.len());
        let mut corr_a: BTreeMap<(H, H), Vec<(usize, f64)>> = BTreeMap::new();
        let mut corr_b: BTreeMap<(H, H), Vec<(usize, f64)>> = BTreeMap::new();
        for (j, &k) in chain.iter().enumerate() {
            let NodeKind::Mma { terms, ops, .. } = &nodes[k as usize].kind else {
                unreachable!()
            };
            let mut v = Vec::with_capacity(terms.len());
            for (t, o) in terms.iter().zip(ops) {
                let Some((p, q)) = *o else {
                    v.push(Tk::Value(*t));
                    continue;
                };
                let mut hk = |k: u32| self.node(k, cx, memo).0;
                v.push(match (part(p), part(q)) {
                    (Some((a, Tf32Part::Hi)), Some((b, Tf32Part::Hi))) => Tk::Main(a, b),
                    (Some((a, pa)), Some((b, Tf32Part::Hi))) if pa.lo_delta().is_some() => {
                        let key = (hk(a), hk(b));
                        corr_a
                            .entry(key)
                            .or_default()
                            .push((j, pa.lo_delta().unwrap()));
                        Tk::Corr
                    }
                    (Some((a, Tf32Part::Hi)), Some((b, pb))) if pb.lo_delta().is_some() => {
                        let key = (hk(a), hk(b));
                        corr_b
                            .entry(key)
                            .or_default()
                            .push((j, pb.lo_delta().unwrap()));
                        Tk::Corr
                    }
                    _ => Tk::Gen,
                });
            }
            kinds.push(v);
        }
        // Match each hi(a) hi(b) with one unclaimed correction of each kind.
        let mut matched: BTreeMap<(usize, usize), (usize, usize, f64, f64)> = BTreeMap::new();
        for (j, v) in kinds.iter().enumerate() {
            for (i, tk) in v.iter().enumerate() {
                let Tk::Main(a, b) = *tk else { continue };
                let key = (self.node(a, cx, memo).0, self.node(b, cx, memo).0);
                let (Some(ca), Some(cb)) = (corr_a.get(&key), corr_b.get(&key)) else {
                    continue;
                };
                if ca.is_empty() || cb.is_empty() {
                    continue;
                }
                let (ja, da) = corr_a.get_mut(&key).unwrap().pop().unwrap();
                let (jb, db) = corr_b.get_mut(&key).unwrap().pop().unwrap();
                matched.insert((j, i), (j.min(ja).min(jb), j.max(ja).max(jb), da, db));
            }
        }
        if corr_a
            .values()
            .chain(corr_b.values())
            .any(|v| !v.is_empty())
        {
            cx.untracked.get_or_insert_with(|| {
                format!(
                    "a 3xTF32 correction product without its hi * hi product at pc {}",
                    nodes[top as usize].pc
                )
            });
        }
        // Hashes (as step by step) and the terms entering at each step.
        let (mut hcur, mut cur) = match base {
            Some(a) => self.node(a, cx, memo),
            None => ((0, 0), BTreeMap::new()),
        };
        let mut entering: Vec<Vec<BTreeMap<Mono, Term>>> = vec![Vec::new(); chain.len()];
        let mut hashes: Vec<Vec<(u64, u64)>> = vec![Vec::new(); chain.len()];
        let fhash = |s: &Self, k: u32, cx: &mut Canon, memo: &mut Memo| match part(k) {
            Some((arg, pt)) => tf32_hash(pt, s.node(arg, cx, memo).0),
            None => s.node(k, cx, memo).0,
        };
        for (j, v) in kinds.iter().enumerate() {
            let NodeKind::Mma { ops, .. } = &nodes[chain[j] as usize].kind else {
                unreachable!()
            };
            for (i, tk) in v.iter().enumerate() {
                let mut x = H2::new();
                if let Tk::Value(t) = tk {
                    x.put(("value", t.to_bits()));
                    if *t != 0.0 {
                        entering[j].push(BTreeMap::from([(
                            vec![],
                            Term {
                                coef: *t,
                                abs: t.abs(),
                                ..Default::default()
                            },
                        )]));
                        cx.untracked.get_or_insert_with(|| {
                            format!(
                                "an MMA product of untracked origin ({t}) at pc {}",
                                nodes[chain[j] as usize].pc
                            )
                        });
                    }
                    hashes[j].push(x.done());
                    continue;
                }
                let (p, q) = ops[i].unwrap();
                let mut hh = [fhash(self, p, cx, memo), fhash(self, q, cx, memo)];
                hh.sort();
                x.put(("prod", hh));
                hashes[j].push(x.done());
                let in_product = |k: u32, memo: &Memo| memo.get(&k).is_some_and(|e| e.2.round_abs);
                match (tk, matched.get(&(j, i))) {
                    (Tk::Main(a, b), Some(&(top_j, deep_j, da, db))) => {
                        let (_, pa) = self.node(*a, cx, memo);
                        let (_, pb) = self.node(*b, cx, memo);
                        if in_product(*a, memo) || in_product(*b, memo) {
                            cx.round_in_product = true;
                        }
                        cx.tf32 = true;
                        // ASSUME(N-TF32): hi hi + lo' hi + hi lo' = a b (1 + theta). Below the
                        // last of the three steps the products are rounded apart, |t1| + |t2| +
                        // |t3| <= (1 + w) |a b|, under the steps' growth h; from it on, as one.
                        let theta = 2f64.powi(-20) + TF32_HI_EPS * (da + db);
                        let w = TF32_HI_EPS * (1.0 + da) + TF32_HI_EPS * (1.0 + db);
                        let h = (top_j + 1..=deep_j).fold(1.0, |h, k| {
                            let NodeKind::Mma { eps, .. } = nodes[chain[k] as usize].kind else {
                                unreachable!()
                            };
                            h * (1.0 + eps)
                        });
                        let mut prod = poly_mul(&pa, &pb);
                        for t in prod.values_mut() {
                            let (a_, g) = (t.abs, t.gam);
                            let extra = theta + (1.0 + w) * (h - 1.0);
                            *t = Term {
                                coef: t.coef,
                                coef_rounded: t.coef_rounded,
                                mult: t.mult * (1.0 + w) + a_ * extra,
                                abs: a_,
                                gam: g + extra * (a_ + g),
                            };
                        }
                        entering[top_j].push(prod);
                    }
                    (Tk::Corr, _) => {
                        // Folded into its matched hi * hi term (or untracked).
                        cx.tf32 = true;
                    }
                    (Tk::Main(..), None) | (Tk::Gen, _) => {
                        let (_, pp) = self.node(p, cx, memo);
                        let (_, pq) = self.node(q, cx, memo);
                        if in_product(p, memo) || in_product(q, memo) {
                            cx.round_in_product = true;
                        }
                        entering[j].push(poly_mul(&pp, &pq));
                    }
                    (Tk::Value(_), _) => unreachable!(),
                }
            }
        }
        // Bottom-up: each step grows what is under it and what enters at it by its eps.
        for j in (0..chain.len()).rev() {
            let NodeKind::Mma { eps, .. } = nodes[chain[j] as usize].kind else {
                unreachable!()
            };
            let mut nxt = BTreeMap::new();
            add_poly(&mut nxt, &cur, eps);
            for e in &entering[j] {
                add_poly(&mut nxt, e, eps);
            }
            cur = nxt;
            let mut th = std::mem::take(&mut hashes[j]);
            th.sort();
            let mut h = H2::new();
            h.put(("mma", hcur, th));
            hcur = h.done();
            cx.abs_extra = cx.abs_extra * (1.0 + eps) + crate::mma_lp::F32_RZ_SUB;
        }
        (hcur, cur)
    }
}

/// Result of one level over all output elements.
#[derive(Clone, Debug, Serialize)]
pub struct LevelResult {
    pub level: String,
    pub holds: bool,
    /// Output elements where it fails.
    pub failures: usize,
    pub first_failure: Option<usize>,
    pub detail: Option<String>,
}

/// One output element's verdict: the strongest level and the bound achieved.
#[derive(Clone, Debug, Serialize)]
pub struct OutBound {
    pub index: usize,
    /// "L0", "L2", "L1" or "none" (not the same function).
    pub level: &'static str,
    /// With L1: |C(x) - R(x)| <= bound_u * u * scale(x) + abs_extra for every input x, where
    /// scale(x) = sum_m a_m |x_m| over the output's monomials (sum |x_i| for a sum). 0 at L0.
    pub bound_u: Option<f64>,
    /// Each kernel's own rigorous Cor 1a bound against the exact value, same form.
    pub ref_u: Option<f64>,
    pub cand_u: Option<f64>,
    pub abs_extra: f64,
    pub monomials: usize,
    /// Why no bound: a MUFU term (needs |f'| or its operand's range), or not L1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_bound: Option<String>,
    /// "sum" (polynomial: scale(x) = sum_m a_m |x_m|) or "softmax" (normal form exp(L_i) /
    /// sum_j exp(L_j): |C - R| <= rel_u u y + add_u u + abs_extra, y the exact output <= 1, so
    /// `bound_u` = rel_u + add_u with scale 1).
    pub form: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rel_u: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub add_u: Option<f64>,
    /// Softmax: the row (members of the maximum) and this output's leaf, for `bound_at`.
    #[serde(skip)]
    pub sm: Option<(Vec<Leaf>, Leaf)>,
    /// Softmax: each kernel's own (K_rel, K_arg, K_abs) against the exact output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sm_ref: Option<(f64, f64, f64)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sm_cand: Option<(f64, f64, f64)>,
    /// Per monomial (a_m, g_R + g_C, c_m): the bound evaluated on other inputs (`bound_at`).
    #[serde(skip)]
    pub terms: Vec<(Mono, f64, f64, f64)>,
    /// "norm" form (layer norm, RMS norm: `norm.rs`): |C - R| <= (rel_u + kappa_u kappa) u |Y1| +
    /// add_u u T(x) for kappa(x) <= kappa_max; `bound_u` = the coefficient of u T(x) valid for
    /// every covered input (`NormBound::total_u`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub norm: Option<crate::norm::NormBound>,
    /// "attn" form (`softmax::wquot`): |C - R| <= rel_u u V1 + add_u u Vmax + abs_extra
    /// (+ absv Vmax), V1 = sum_j p_j |v_j|, Vmax = max_j |v_j|: `bound_u` = rel_u + add_u
    /// (V1 <= Vmax). `attn_parts`: each kernel's (k_rel, k_arg, k_abs0, k_absv).
    #[serde(skip)]
    pub attn: Option<crate::softmax::AttnEval>,
    /// "attn" form with computed scores: (kq, q_max): the V1 coefficient is rel_u + kq Q(x)
    /// for Q(x) = max_j sum_d |q_d k_jd| <= q_max, and `bound_u` = rel_u + add_u + kq is the
    /// coefficient of u (1 + Q(x)) Vmax; the exact per-input bound has no such limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<(f64, f64)>,
    /// "round" form (both outputs 16-bit roundings of L1 values, N-ROUND): compact, for every
    /// input, |C - R| <= round_rel |y| + sum_m |x_m| g_m + abs_extra, y = sum_m c_m x_m the
    /// exact value, g_m (in `terms`) = (1 + u16) (g_R,m + g_C,m) of the values before the
    /// rounding. `bound_u` = round_rel / u + the largest g_m / (a_m u) (|y| <= sum_m a_m |x_m|).
    /// `bound_at` is the exact form (`round16_gap`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub round_rel: Option<f64>,
    /// "round" form: per monomial of `terms`, each kernel's own g_K,m before the rounding.
    #[serde(skip)]
    pub round_own: Vec<(f64, f64)>,
    /// "round" form: each kernel's absolute error before the rounding.
    #[serde(skip)]
    pub round_abs: (f64, f64),
    /// "round" form: the output format (f16, else bf16).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub round_f16: Option<bool>,
    /// "round" form: (reference, candidate) values before the rounding carry 16-bit roundings
    /// themselves (eager rounds between kernels: different functions, bounded).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub round_inner: Option<(bool, bool)>,
    /// "round" form: C and R are equal or adjacent 16-bit values for every input with
    /// sum_m a_m |x_m| <= kappa1 |y| and |y| >= y1 (None: no kappa >= 1 qualifies).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub round_adjacent: Option<(f64, f64)>,
}

/// v rounded to bf16 (`f16`: f16), to nearest even; beyond the largest finite value: +-inf.
pub fn rnd16(v: f64, f16: bool) -> f64 {
    if v == 0.0 || !v.is_finite() {
        return v;
    }
    let (p, emin, max) = if f16 {
        (11, -14, 65504.0)
    } else {
        (8, -126, f32::from_bits(0x7f7f_0000) as f64)
    };
    let a = v.abs();
    let be = ((a.to_bits() >> 52) & 0x7ff) as i32;
    let e = if be == 0 { -1023 } else { be - 1023 };
    let q = 2f64.powi(e.max(emin) - (p - 1));
    let r = (a / q).round_ties_even() * q;
    let r = if r > max { f64::INFINITY } else { r };
    r.copysign(v)
}

/// The 16-bit pattern of a value `rnd16` returned.
fn bits16(v: f64, f16: bool) -> u32 {
    let b = (v as f32).to_bits();
    if f16 {
        crate::fp::f32_to_f16_rn(b) as u32
    } else {
        b >> 16
    }
}

/// f64 slack for a bound g around an approximate exact value y: a real interval
/// [y - g, y + g] computed in f64 stays inside [y - s, y + s] (y within 2^-52 relative).
fn slack16(y: f64, g: f64) -> f64 {
    g * (1.0 + 2f64.powi(-49)) + y.abs() * 2f64.powi(-49) + 1e-300
}

/// Two kernels' 16-bit roundings of values a in [y - gr, y + gr] and b in [y - gc, y + gc]:
/// rounding is monotone, so |rnd(a) - rnd(b)| <= max(rnd(y + gc) - rnd(y - gr), rnd(y + gr)
/// - rnd(y - gc)). Returns (that bound, in 16-bit steps); None if a rounding can overflow.
pub fn round16_gap(y: f64, gr: f64, gc: f64, f16: bool) -> Option<(f64, u64)> {
    let (sr, sc) = (slack16(y, gr), slack16(y, gc));
    let ty = if f16 {
        crate::elementwise::EType::F16
    } else {
        crate::elementwise::EType::Bf16
    };
    let mut best = (0.0f64, 0u64);
    for (lo, hi) in [(y - sr, y + sc), (y - sc, y + sr)] {
        let (a, b) = (rnd16(lo, f16), rnd16(hi, f16));
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        let steps = ty.ulps_between(bits16(a, f16), bits16(b, f16));
        if b - a > best.0 || steps > best.1 {
            best = (best.0.max(b - a), best.1.max(steps));
        }
    }
    Some(best)
}

/// One kernel's 16-bit rounding of a value in [y - g, y + g], against y.
pub fn round16_own(y: f64, g: f64, f16: bool) -> f64 {
    let s = slack16(y, g);
    let (a, b) = (rnd16(y - s, f16), rnd16(y + s, f16));
    ((b - y).max(y - a) + y.abs() * 2f64.powi(-50)) * (1.0 + 1e-12)
}

impl OutBound {
    /// The bound's value at input `x` (leaf -> value): sum_m |x_m| (g_R + g_C) + abs_extra,
    /// rounded up. None without a bound or with a MUFU pseudo-leaf.
    pub fn bound_at(&self, x: &dyn Fn(Leaf) -> f64) -> Option<f64> {
        self.bound_u?;
        if let Some(nb) = &self.norm {
            return nb.eval.at(x).map(|b| b.0);
        }
        if let Some(a) = &self.attn {
            if !a.covered(x) {
                return None;
            }
            return Some(a.at(x).0);
        }
        if let (Some((row, i)), Some(rel), Some(add)) = (&self.sm, self.rel_u, self.add_u) {
            // The exact softmax in f64 (relative error ~1e-15, covered by the slack).
            let m = row.iter().map(|&l| x(l)).fold(f64::NEG_INFINITY, f64::max);
            let s: f64 = row.iter().map(|&l| (x(l) - m).exp()).sum();
            let y = (x(*i) - m).exp() / s;
            return Some(((rel * y + add) * U32 + self.abs_extra) * (1.0 + 1e-9) + 1e-300);
        }
        if let Some(f16) = self.round_f16 {
            // The exact form: the two roundings of the intervals around y.
            let (y, gr, gc) = self.round_pre_at(x);
            return round16_gap(y, gr, gc, f16).map(|g| g.0);
        }
        let mut b = self.abs_extra;
        for (m, _, g, _) in &self.terms {
            let v: f64 = m.iter().map(|&l| x(l).abs()).product();
            b += v * g;
        }
        Some(b * (1.0 + 1e-12))
    }
    /// "round" form: (exact y, each kernel's bound before its rounding) at x.
    pub fn round_pre_at(&self, x: &dyn Fn(Leaf) -> f64) -> (f64, f64, f64) {
        let y = self.exact_at(x);
        let (mut gr, mut gc) = self.round_abs;
        for ((m, _, _, _), (r, c)) in self.terms.iter().zip(&self.round_own) {
            let v: f64 = m.iter().map(|&l| x(l).abs()).product();
            gr += v * r;
            gc += v * c;
        }
        (y, gr * (1.0 + 1e-12), gc * (1.0 + 1e-12))
    }
    /// "round" form: the bound at x in 16-bit steps (0: bitwise equal, 1: equal or adjacent).
    pub fn round_steps_at(&self, x: &dyn Fn(Leaf) -> f64) -> Option<u64> {
        if let Some(nb) = &self.norm {
            return nb.eval.steps_at(x);
        }
        let f16 = self.round_f16?;
        let (y, gr, gc) = self.round_pre_at(x);
        round16_gap(y, gr, gc, f16).map(|g| g.1)
    }
    /// The exact value sum_m c_m x_m (expansions: exact for products of up to two f32 leaves).
    pub fn exact_at(&self, x: &dyn Fn(Leaf) -> f64) -> f64 {
        let mut e: Vec<f64> = vec![];
        for (m, _, _, c) in &self.terms {
            let v = m.iter().fold(crate::exact::from(1.0), |acc, &l| {
                crate::exact::mul(&acc, &crate::exact::from(x(l)))
            });
            e = crate::exact::add(&e, &crate::exact::mul(&v, &crate::exact::from(*c)));
        }
        crate::exact::approx(&e)
    }
    /// "round" form: (exact y, the reference's own bound, the candidate's) at x.
    pub fn round_own_at(&self, x: &dyn Fn(Leaf) -> f64) -> Option<(f64, f64, f64)> {
        let f16 = self.round_f16?;
        let (y, gr, gc) = self.round_pre_at(x);
        Some((y, round16_own(y, gr, f16), round16_own(y, gc, f16)))
    }
    /// Softmax: the exact output at x and each kernel's own bound there.
    pub fn softmax_at(&self, x: &dyn Fn(Leaf) -> f64) -> Option<(f64, f64, f64)> {
        let ((row, i), r, c) = (self.sm.as_ref()?, self.sm_ref?, self.sm_cand?);
        let m = row.iter().map(|&l| x(l)).fold(f64::NEG_INFINITY, f64::max);
        let s: f64 = row.iter().map(|&l| (x(l) - m).exp()).sum();
        let y = (x(*i) - m).exp() / s;
        let own = |k: (f64, f64, f64)| (k.0 * y + k.1 + k.2) * (1.0 + 1e-9) + 1e-300;
        Some((y, own(r), own(c)))
    }

    /// scale(x) = sum_m a_m |x_m| (softmax: 1).
    pub fn scale_at(&self, x: &dyn Fn(Leaf) -> f64) -> f64 {
        if let Some(nb) = &self.norm {
            return nb.eval.at(x).map_or(0.0, |b| b.2);
        }
        if let Some(a) = &self.attn {
            let vm = a.exact(x).2;
            return if self.score.is_some() {
                (1.0 + a.q_at(x)) * vm
            } else {
                vm
            };
        }
        if self.sm.is_some() {
            return 1.0;
        }
        self.terms
            .iter()
            .map(|(m, a, _, _)| a * m.iter().map(|&l| x(l).abs()).product::<f64>())
            .sum()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub outputs: usize,
    pub levels: Vec<LevelResult>,
    /// A monomial whose coefficient differs at the first L1 failure (for a witness), with the
    /// two coefficients.
    #[serde(skip)]
    pub witness_mono: Option<(usize, Mono, f64, f64)>,
    pub untracked: Option<String>,
    /// Why some output's rounding order depends on the schedule (fp32 atomics).
    pub unordered: Option<String>,
    /// Softmax normal form found (segment mode): the form and the slices' contracts.
    pub softmax: Option<String>,
    /// Input conditions the bounds assume beyond finiteness (N-TF32).
    pub conditions: Vec<String>,
    pub per_output: Vec<OutBound>,
    /// Outputs whose L0 holds only up to the sign of zero (`Canon::l0_z`): bitwise equal
    /// except where one is -0 and the other +0.
    pub l0_zero_sign: usize,
}

impl Report {
    /// The largest per-output bound (units of u), if every output has one.
    pub fn max_bound_u(&self) -> Option<f64> {
        self.per_output
            .iter()
            .map(|o| o.bound_u)
            .try_fold(0.0f64, |a, b| b.map(|b| a.max(b)))
    }
}

fn show_mono(m: &Mono, names: &[String]) -> String {
    if m.is_empty() {
        return "1".into();
    }
    m.iter()
        .map(|(b, i)| {
            if *b == MUFU_LEAF {
                format!("mufu#{i:x}")
            } else if *b == MAX_LEAF {
                format!("max#{i:x}")
            } else if *b == SEG_LEAF {
                format!("slice#{i:x}")
            } else if let Some((t, k)) = leaf_lp(*i) {
                format!(
                    "{}[{k}]:{}",
                    names[*b as usize],
                    format!("{t:?}").to_lowercase()
                )
            } else {
                format!("{}[{i}]", names[*b as usize])
            }
        })
        .collect::<Vec<_>>()
        .join("*")
}

/// The rigorous bound of an L1 pair of canonical forms: (k, k_R, k_C) in units of u with
/// per-monomial (a_m, g_R + g_C, c_m), or why there is none.
// ASSUME(N-REASSOC): see ASSUMPTIONS.md
#[allow(clippy::type_complexity)]
fn rigorous(cr: &Canon, cc: &Canon) -> Result<(f64, f64, f64, Vec<(Mono, f64, f64, f64)>), String> {
    if let Some(reason) = cr
        .untracked
        .as_ref()
        .or(cc.untracked.as_ref())
        .or(cr.coefficient_loss.as_ref())
        .or(cc.coefficient_loss.as_ref())
    {
        return Err(reason.clone());
    }

    if cr
        .poly
        .keys()
        .chain(cc.poly.keys())
        .any(|m| m.iter().any(pseudo))
    {
        return Err(
            "a MUFU, max or slice term: its error is not relative to the inputs' magnitudes".into(),
        );
    }
    if cr.round_in_product || cc.round_in_product {
        return Err(
            "a bf16-rounded value inside a product: its subnormal absolute error is scaled".into(),
        );
    }
    let (mut k, mut kr, mut kc) = (0.0f64, 0.0f64, 0.0f64);
    let mut terms = Vec::with_capacity(cr.poly.len());
    // Every monomial of either side (a cancelled one may be on one side only).
    let keys: std::collections::BTreeSet<&Mono> = cr.poly.keys().chain(cc.poly.keys()).collect();
    for m in keys {
        let tr = cr.poly.get(m).copied().unwrap_or_default();
        let tc = cc.poly.get(m).copied().unwrap_or_default();
        let a = tr.abs.max(tc.abs);
        if a == 0.0 {
            continue;
        }
        // f64 evaluation of the multipliers: rounded up by a relative 2^-40.
        let up = 1.0 + 2f64.powi(-40);
        k = k.max((tr.gam + tc.gam) / a * up);
        kr = kr.max(tr.gam / a * up);
        kc = kc.max(tc.gam / a * up);
        terms.push((m.clone(), a, (tr.gam + tc.gam) * up, tr.coef));
    }
    Ok((k / U32, kr / U32, kc / U32, terms))
}

/// One kernel's own rigorous bound against the exact value at output `i` (Cor 1a, as
/// `compare`'s `ref_u`): |K(x) - y| <= rel |y| + k_u u scale(x) + abs for every input,
/// scale(x) = sum_m |c_m| |x_m|. A 16-bit output rounding (N-ROUND): the value before it has
/// the polynomial bound G, and |rnd(K^) - y| <= u16 |y| + (1 + u16) G + a16. Returns (rel,
/// k_u, abs, form, preconditions) or why there is no bound.
pub fn own_bound(s: &Side, i: usize) -> Result<(f64, f64, f64, &'static str, Vec<String>), String> {
    let (bits, node) = *s.outs.get(i).ok_or("no such output")?;
    let Some(node) = node else {
        return Err(if f32::from_bits(bits) == 0.0 {
            "a constant zero output".into()
        } else {
            "an output the numerics shadow does not express over the inputs".into()
        });
    };
    let mut memo = Memo::new();
    let (round, n) = match s.ns.nodes[node as usize].kind {
        NodeKind::Round { arg, f16 } => (Some(f16), arg),
        _ => (None, node),
    };
    let mut cx = Canon::default();
    let (_, poly) = s.node(n, &mut cx, &mut memo);
    cx.poly = poly;
    cx.poly.retain(|_, t| t.coef != 0.0 || t.gam != 0.0);
    if let Some(u) = cx.untracked.as_ref().or(cx.coefficient_loss.as_ref()) {
        return Err(u.clone());
    }
    if cx.poly.keys().any(|m| m.iter().any(pseudo)) {
        return Err(
            "a MUFU, max or slice term: its error is not relative to the inputs' magnitudes".into(),
        );
    }
    if cx.round_in_product {
        return Err(
            "a bf16-rounded value inside a product: its subnormal absolute error is scaled".into(),
        );
    }
    let up = 1.0 + 2f64.powi(-40);
    let mut k = 0.0f64;
    for t in cx.poly.values() {
        if t.abs > 0.0 {
            k = k.max(t.gam / t.abs * up);
        }
    }
    let mut pre = vec![];
    if cx.tf32 {
        pre.push(TF32_COND.to_string());
    }
    if let Some(u) = &cx.unordered {
        pre.push(format!("rounding order: {u}"));
    }
    Ok(match round {
        Some(f16) => {
            let (u16_, a16) = round_unit(f16);
            (
                u16_,
                k / U32 * (1.0 + u16_),
                cx.abs_extra * (1.0 + u16_) + a16,
                "round",
                pre,
            )
        }
        None => (0.0, k / U32, cx.abs_extra, "sum", pre),
    })
}

/// Compare candidate `c` against reference `r`, element by element (identity correspondence).
/// `tau_k`: the L3 tolerance k (in units of u = 2^-24), if requested.
#[allow(unused_assignments)]
pub fn compare(r: &Side, c: &Side, names: &[String], tau_k: Option<f64>) -> Report {
    let n = r.outs.len().min(c.outs.len());
    let mut fails = [0usize; 4];
    let mut l0_zero_sign = 0usize;
    let mut first: [Option<(usize, String)>; 4] = Default::default();
    let mut witness = None;
    let mut untracked = None;
    let mut unordered = None;
    let mut per_output = Vec::with_capacity(n);
    let mut sm_r = crate::softmax::Ctx::new(r.ns, &r.buf_id);
    let mut sm_c = crate::softmax::Ctx::new(c.ns, &c.buf_id);
    let mut sm_detail: Option<String> = None;
    // Score rows per row of members (shared by the row's outputs).
    let mut score_memo: std::collections::HashMap<
        Vec<Leaf>,
        std::sync::Arc<crate::softmax::ScoreRows>,
    > = Default::default();
    let mut nm_r = crate::norm::Ctx::new(r.ns, &r.buf_id);
    let mut nm_c = crate::norm::Ctx::new(c.ns, &c.buf_id);
    let mut nm_detail: Option<String> = None;
    let (mut memo_r, mut memo_c) = (Memo::new(), Memo::new());
    let mut tf32 = false;
    for i in 0..n {
        // Bound the memo (attention outputs keep many per-output partial sums): drop it when
        // it holds more than 2^22 monomials; it is a cache, the forms do not depend on it.
        if i % 64 == 0 {
            for m in [&mut memo_r, &mut memo_c] {
                if m.values().map(|e| e.1.len()).sum::<usize>() > 1 << 22 {
                    m.clear();
                }
            }
        }
        let (cr, cc) = (r.canon_with(i, &mut memo_r), c.canon_with(i, &mut memo_c));
        let coefficient_loss = cr
            .coefficient_loss
            .as_ref()
            .or(cc.coefficient_loss.as_ref());
        tf32 |= cr.tf32 || cc.tf32;
        if untracked.is_none() {
            untracked = cr.untracked.clone().or(cc.untracked.clone());
        }
        if unordered.is_none() {
            unordered = cr.unordered.clone().or(cc.unordered.clone());
        }
        let mut lv = [true; 4];
        macro_rules! note {
            ($lvl:expr, $why:expr $(,)?) => {{
                let lvl: usize = $lvl;
                if lv[lvl] {
                    fails[lvl] += 1;
                    lv[lvl] = false;
                }
                if first[lvl].is_none() {
                    first[lvl] = Some((i, $why));
                }
            }};
        }
        // L0 up to the sign of zero: one side's output is the other's plus +0.
        let z = |c: &Canon| c.l0_z.unwrap_or(c.l0);
        if cr.l0 != cc.l0 && z(&cr) == z(&cc) {
            l0_zero_sign += 1;
        }
        if z(&cr) != z(&cc)
            || coefficient_loss.is_some()
            || cr.untracked.is_some()
            || cc.untracked.is_some()
            || cr.unordered.is_some()
            || cc.unordered.is_some()
        {
            note!(
                0,
                match cr.unordered.as_ref().or(cc.unordered.as_ref()) {
                    Some(u) if cr.l0 == cc.l0 => u.clone(),
                    _ => "canonical rounding DAGs differ".into(),
                },
            );
        }
        if let Some(reason) = cr.untracked.as_ref().or(cc.untracked.as_ref()) {
            note!(1, reason.clone());
            note!(2, reason.clone());
            if tau_k.is_some() {
                note!(3, reason.clone());
            }
        }
        // L1: equal coefficient maps.
        let mut diff: Vec<(Mono, f64, f64)> = Vec::new();
        for (m, t) in &cr.poly {
            let tc = cc.poly.get(m).map_or(0.0, |x| x.coef);
            if tc != t.coef {
                diff.push((m.clone(), t.coef, tc));
            }
        }
        for (m, t) in &cc.poly {
            if !cr.poly.contains_key(m) && t.coef != 0.0 {
                diff.push((m.clone(), 0.0, t.coef));
            }
        }
        let mut ob = OutBound {
            index: i,
            level: "none",
            bound_u: None,
            ref_u: None,
            cand_u: None,
            abs_extra: cr.abs_extra + cc.abs_extra,
            monomials: cr.poly.len(),
            no_bound: None,
            terms: vec![],
            form: "sum",
            rel_u: None,
            add_u: None,
            sm: None,
            sm_ref: None,
            sm_cand: None,
            norm: None,
            attn: None,
            score: None,
            round_rel: None,
            round_own: vec![],
            round_abs: (0.0, 0.0),
            round_f16: None,
            round_inner: None,
            round_adjacent: None,
        };
        // Segment mode: the softmax normal form when the polynomials differ.
        if (!diff.is_empty() || coefficient_loss.is_some()) && r.ns.segments && c.ns.segments {
            let q = (|| -> Result<_, String> {
                if let Some(reason) = coefficient_loss {
                    return Err(reason.clone());
                }
                let rn = r.outs[i].1.ok_or("reference output without a node")?;
                let cn = c.outs[i].1.ok_or("candidate output without a node")?;
                let qr = sm_r.quot(rn)?;
                let qc = sm_c.quot(cn)?;
                crate::softmax::same(&qr, &qc)?;
                Ok((qr, qc))
            })();
            match q {
                Ok((qr, qc)) => {
                    let (kr, kc) = (crate::softmax::kbound(&qr), crate::softmax::kbound(&qc));
                    let leaf = qr.num.0.terms.iter().find_map(|(a, c)| match a {
                        crate::softmax::Atom::In(l) if *c == 1.0 => Some(*l),
                        _ => None,
                    });
                    ob.level = "L1";
                    ob.form = "softmax";
                    ob.rel_u = Some((kr.k_rel + kc.k_rel) / U32);
                    ob.add_u = Some((kr.k_arg + kc.k_arg) / U32);
                    ob.bound_u = Some((kr.k_rel + kc.k_rel + kr.k_arg + kc.k_arg) / U32);
                    ob.ref_u = Some((kr.k_rel + kr.k_arg) / U32);
                    ob.cand_u = Some((kc.k_rel + kc.k_arg) / U32);
                    ob.abs_extra = kr.k_abs + kc.k_abs;
                    ob.monomials = qr.n;
                    ob.sm = leaf.map(|l| (qr.members.clone(), l));
                    ob.sm_ref = Some((kr.k_rel, kr.k_arg, kr.k_abs));
                    ob.sm_cand = Some((kc.k_rel, kc.k_arg, kc.k_abs));
                    if sm_detail.is_none() {
                        sm_detail = Some(format!(
                            "softmax normal form exp({}) / sum of {} exp(x_j - max); contracts {}",
                            qr.num.0.show(names),
                            qr.n,
                            qr.contracts
                                .iter()
                                .chain(&qc.contracts)
                                .map(|c| format!(
                                    "{} r0={:.2}u r1={:.3}u/|d| a={:.1e} ({} values)",
                                    c.func,
                                    c.r0 / U32,
                                    c.r1 / U32,
                                    c.a,
                                    c.values
                                ))
                                .collect::<Vec<_>>()
                                .join("; ")
                        ));
                    }
                    note!(2, "softmax normal form: L2 is not compared".into());
                    if tau_k.is_some() {
                        note!(3, "softmax normal form: see the per-output bound".into());
                    }
                    per_output.push(ob);
                    continue;
                }
                Err(e) => ob.no_bound = Some(format!("not the same function (softmax form: {e})")),
            }
            // The attention form: sum_j exp(x_j - max) v_j / sum_j exp(x_j - max). bf16
            // outputs (both roundings, N-ROUND): the form of the values before the rounding,
            // then |rnd(o^) - o| <= (1 + 2^-8) |o^ - o| + 2^-8 |o| + 2^-134, |o| <= V1.
            let peel = |side: &Side, n: Option<u32>| -> Option<u32> {
                let n = n?;
                match side.ns.nodes[n as usize].kind {
                    NodeKind::Round { arg, f16: false } => Some(arg),
                    _ => None,
                }
            };
            let out_round = match (peel(r, r.outs[i].1), peel(c, c.outs[i].1)) {
                (Some(a), Some(b)) => Some((a, b)),
                _ => None,
            };
            let ub = if out_round.is_some() { BF16_U } else { 0.0 };
            let wq = (|| -> Result<_, String> {
                if let Some(reason) = coefficient_loss {
                    return Err(reason.clone());
                }
                let (rn, cn) = match out_round {
                    Some(x) => x,
                    None => (
                        r.outs[i].1.ok_or("reference output without a node")?,
                        c.outs[i].1.ok_or("candidate output without a node")?,
                    ),
                };
                let why = std::env::var("PTXV_EQUIV_WHY")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    == Some(i);
                let qr = sm_r.wquot(rn).inspect_err(|e| {
                    if why {
                        eprintln!("reference: {e}")
                    }
                })?;
                let qc = sm_c.wquot(cn).inspect_err(|e| {
                    if why {
                        eprintln!("candidate: {e}")
                    }
                })?;
                crate::softmax::wsame(&qr, &qc)?;
                Ok((qr, qc))
            })();
            match wq {
                Ok((qr, qc)) => {
                    let (kr, kc) = (crate::softmax::wbound(&qr), crate::softmax::wbound(&qc));
                    // The output roundings (compact form): each kernel's bound times (1 + ub)
                    // plus ub V1 (+ 2^-134).
                    let g = 1.0 + ub;
                    let sb = if out_round.is_some() { BF16_SUB } else { 0.0 };
                    ob.level = "L1";
                    ob.form = "attn";
                    ob.rel_u = Some(((kr.k_rel + kc.k_rel) * g + 2.0 * ub) / U32);
                    ob.add_u = Some((kr.k_arg + kc.k_arg + kr.k_absv + kc.k_absv) * g / U32);
                    ob.bound_u = Some(ob.rel_u.unwrap() + ob.add_u.unwrap());
                    ob.ref_u = Some(((kr.k_rel + kr.k_arg + kr.k_absv) * g + ub) / U32);
                    ob.cand_u = Some(((kc.k_rel + kc.k_arg + kc.k_absv) * g + ub) / U32);
                    ob.abs_extra = (kr.k_abs0 + kc.k_abs0) * g + 2.0 * sb;
                    ob.monomials = qr.num.terms.len();
                    ob.no_bound = None;
                    if sm_detail.is_none() {
                        sm_detail = Some(format!(
                            "attention normal form sum_j exp(x_j - max) v_j / sum of {} exp(x_j - max); contracts {}",
                            qr.n,
                            qr.contracts
                                .iter()
                                .chain(&qc.contracts)
                                .map(|c| format!(
                                    "{} r0={:.2}u ({} values)",
                                    c.func,
                                    c.r0 / U32,
                                    c.values
                                ))
                                .collect::<Vec<_>>()
                                .join("; ")
                        ));
                    }
                    let mut ae = crate::softmax::AttnEval::new(&qr, kr, kc);
                    ae.out_round = ub;
                    if qr.base2 != qc.base2 {
                        ob.level = "none";
                        ob.bound_u = None;
                        ob.no_bound =
                            Some("attention form: exponentials of different bases".into());
                        note!(1, "attention form: exponentials of different bases".into());
                        per_output.push(ob);
                        continue;
                    }
                    ae.shifted = qr.shifted || qc.shifted;
                    let memo = match score_memo.get(&qr.members) {
                        Some(a) => Ok(a.clone()),
                        None => crate::softmax::score_rows(&qr.members, &sm_r, &sm_c).map(|rows| {
                            let a = std::sync::Arc::new(crate::softmax::ScoreRows::new(rows));
                            score_memo.insert(qr.members.clone(), a.clone());
                            a
                        }),
                    };
                    match memo {
                        Ok(rows) => ae.scores = rows,
                        Err(e) => {
                            ob.level = "none";
                            ob.bound_u = None;
                            ob.no_bound = Some(format!("attention form: {e}"));
                            note!(1, format!("attention form: {e}"));
                            per_output.push(ob);
                            continue;
                        }
                    }
                    if !ae.scores.is_empty() {
                        // Computed scores: the compact form adds kq u Q(x) to the V1
                        // coefficient, Q = max_j sum_m a_m |x_m| (each score's magnitude), for
                        // Delta <= 2^-10 (e^(2 Delta) - 1 <= 2 c2 Delta).
                        let up = 1.0 + 2f64.powi(-40);
                        let (mut ksr, mut ksc) = (0.0f64, 0.0f64);
                        for rows in ae.scores.values() {
                            for (_, _, a, gr, gc) in rows {
                                ksr = ksr.max(gr / a * up);
                                ksc = ksc.max(gc / a * up);
                            }
                        }
                        let c2 = (2f64.powi(-9).exp() - 1.0) / 2f64.powi(-9) * up;
                        // Base 2: the score error in the natural exponent is ln 2 Delta.
                        let ln = if ae.base2 {
                            std::f64::consts::LN_2 * up
                        } else {
                            1.0
                        };
                        let kq =
                            2.0 * c2 * ln * ((1.0 + kr.k_rel) * ksr + (1.0 + kc.k_rel) * ksc) * g
                                / U32;
                        let mut q_max = 2f64.powi(-10) / (ln * ksr.max(ksc));
                        if ae.shifted {
                            q_max = q_max.min(crate::softmax::SHIFT_X / (1.0 + 2f64.powi(-10)));
                        }
                        ob.score = Some((kq, q_max));
                        // (rel + kq Q) V1 + add Vmax <= (rel + add + kq) (1 + Q) Vmax.
                        ob.bound_u = Some(ob.rel_u.unwrap() + ob.add_u.unwrap() + kq);
                    }
                    ob.attn = Some(ae);
                    note!(2, "attention normal form: L2 is not compared".into());
                    if tau_k.is_some() {
                        note!(3, "attention normal form: see the per-output bound".into());
                    }
                    per_output.push(ob);
                    continue;
                }
                Err(e) => {
                    let s = ob.no_bound.take().unwrap_or_default();
                    ob.no_bound = Some(format!("{s}; attention form: {e}"));
                }
            }
            // The norm form: a polynomial in the inputs and one nonlinear scalar f(A). Both
            // outputs 16-bit roundings (N-ROUND): the form of the values before them.
            let peel16 = |side: &Side, n: u32| match side.ns.nodes[n as usize].kind {
                NodeKind::Round { arg, f16 } => (arg, Some(f16)),
                _ => (n, None),
            };
            let nb = match (r.outs[i].1, c.outs[i].1) {
                (Some(rn), Some(cn)) => match (peel16(r, rn), peel16(c, cn)) {
                    ((ra, Some(fr)), (ca, Some(fc))) if fr == fc => {
                        // abs_extra (both outputs' canonical forms) already holds (1 + u16)
                        // (A_R + A_C) + 2 a16.
                        let (mut sr, mut sc) = (Canon::default(), Canon::default());
                        r.node(ra, &mut sr, &mut memo_r);
                        c.node(ca, &mut sc, &mut memo_c);
                        ob.round_inner = Some((sr.round_abs, sc.round_abs));
                        crate::norm::bound(&mut nm_r, ra, &mut nm_c, ca).map(|b| b.rounded(fr))
                    }
                    _ => crate::norm::bound(&mut nm_r, rn, &mut nm_c, cn),
                },
                _ => Err("an output without a node".into()),
            };
            match nb {
                Ok(nb) => {
                    ob.level = if nb.coef_diff == 0.0 { "L1" } else { "L1~" };
                    ob.form = "norm";
                    ob.bound_u = Some(nb.total_u());
                    let k = if nb.kappa_le_1 { 1.0 } else { nb.kappa_max };
                    ob.ref_u = Some(nb.ref_part.0 + nb.ref_part.1 * k + nb.ref_part.2);
                    ob.cand_u = Some(nb.cand_part.0 + nb.cand_part.1 * k + nb.cand_part.2);
                    ob.rel_u = Some(nb.rel_u);
                    ob.add_u = Some(nb.add_u);
                    ob.no_bound = None;
                    ob.monomials = nb.eval.out.len();
                    if nm_detail.is_none() {
                        nm_detail = Some(format!(
                            "norm form: polynomial in the inputs and rho = A^{} (A: {} monomials); \
                             largest coefficient difference against the reference {:.2e}; contracts {}",
                            nb.p,
                            nb.eval.a.len(),
                            nb.coef_diff,
                            nb.contracts
                                .iter()
                                .map(|c| format!(
                                    "{} r0={:.3}u ({} values)",
                                    c.func,
                                    c.r0 / U32,
                                    c.values
                                ))
                                .collect::<Vec<_>>()
                                .join("; ")
                        ));
                    }
                    ob.norm = Some(nb);
                    if ob.level != "L1" {
                        note!(
                            1,
                            "norm form: coefficients differ by rounded constants (bounded)".into()
                        );
                    }
                    note!(2, "norm form: L2 is not compared".into());
                    if tau_k.is_some() {
                        note!(3, "norm form: see the per-output bound".into());
                    }
                    per_output.push(ob);
                    continue;
                }
                Err(e) => {
                    let s = ob.no_bound.take().unwrap_or_default();
                    ob.no_bound = Some(format!("{s}; norm form: {e}"));
                }
            }
        }
        if let Some(reason) = coefficient_loss {
            untracked.get_or_insert_with(|| reason.clone());
            note!(1, reason.clone());
            note!(2, reason.clone());
            if tau_k.is_some() {
                note!(3, reason.clone());
            }
        }
        if !diff.is_empty() {
            let show: Vec<String> = diff
                .iter()
                .take(4)
                .map(|(m, a, b)| format!("{}: {a} vs {b}", show_mono(m, names)))
                .collect();
            note!(
                1,
                format!(
                    "{} monomials differ (reference vs candidate coefficient), e.g. {}",
                    diff.len(),
                    show.join(", ")
                ),
            );
            if witness.is_none() {
                // Prefer a monomial over input leaves: a constant cannot be isolated, nor a
                // MUFU pseudo-leaf.
                // Not tagged byte leaves either: setting a code to "1" is not a value of 1.
                let plain = |d: &&(Mono, f64, f64)| {
                    !d.0.iter().any(|l| pseudo(l) || leaf_lp(l.1).is_some())
                };
                if let Some(d) = diff
                    .iter()
                    .filter(plain)
                    .find(|d| !d.0.is_empty())
                    .or(diff.iter().find(plain))
                {
                    witness = Some((i, d.0.clone(), d.1, d.2));
                }
            }
            note!(2, "not L1".into());
            if tau_k.is_some() {
                note!(3, "not L1".into());
            }
            ob.no_bound
                .get_or_insert_with(|| "not the same function (L1 fails)".into());
            per_output.push(ob);
            continue;
        }
        // Both outputs bf16 roundings (N-ROUND): bound the values before the rounding, then
        // |rnd(K^) - y| <= 2^-8 |K^| + 2^-134 + |K^ - y| with |K^| <= |y| + |K^ - y|.
        let round_args = match (r.outs[i].1, c.outs[i].1) {
            (Some(rn), Some(cn)) => {
                match (&r.ns.nodes[rn as usize].kind, &c.ns.nodes[cn as usize].kind) {
                    (
                        NodeKind::Round { arg: ra, f16: fr },
                        NodeKind::Round { arg: ca, f16: fc },
                    ) if fr == fc => Some((*ra, *ca, *fr)),
                    _ => None,
                }
            }
            _ => None,
        };
        let round = round_args.map(|(ra, ca, f16)| {
            let mut sr = Canon::default();
            let mut sc = Canon::default();
            let (_, pr) = r.node(ra, &mut sr, &mut memo_r);
            let (_, pc) = c.node(ca, &mut sc, &mut memo_c);
            sr.poly = pr;
            sc.poly = pc;
            sr.poly.retain(|_, t| t.coef != 0.0 || t.gam != 0.0);
            sc.poly.retain(|_, t| t.coef != 0.0 || t.gam != 0.0);
            // f64 evaluation of the multipliers: rounded up by a relative 2^-40 (`rigorous`).
            let up = 1.0 + 2f64.powi(-40);
            // In `rigorous`'s order (both sides' monomials).
            let keys: std::collections::BTreeSet<&Mono> =
                sr.poly.keys().chain(sc.poly.keys()).collect();
            let own: Vec<(f64, f64)> = keys
                .into_iter()
                .filter(|m| {
                    sr.poly
                        .get(*m)
                        .map_or(0.0, |t| t.abs)
                        .max(sc.poly.get(*m).map_or(0.0, |t| t.abs))
                        != 0.0
                })
                .map(|m| {
                    let g = |p: &BTreeMap<Mono, Term>| p.get(m).map_or(0.0, |x| x.gam) * up;
                    (g(&sr.poly), g(&sc.poly))
                })
                .collect();
            (
                rigorous(&sr, &sc),
                (sr.abs_extra, sc.abs_extra),
                own,
                f16,
                (sr.round_abs, sc.round_abs),
            )
        });
        match round {
            // ASSUME(N-ROUND): both outputs are round-to-nearest 16-bit roundings of values
            // within the Cor 1a bounds of y.
            Some((Ok((k, kr, kc, terms)), (abs_r, abs_c), own, f16, inner)) if !lv[0] => {
                let (u16_, a16) = round_unit(f16);
                let g = 1.0 + u16_;
                let rr = 2.0 * u16_;
                ob.form = "round";
                ob.round_rel = Some(rr);
                ob.rel_u = Some(rr / U32);
                ob.add_u = Some(k * g);
                ob.bound_u = Some(rr / U32 + k * g);
                ob.ref_u = Some(u16_ / U32 + kr * g);
                ob.cand_u = Some(u16_ / U32 + kc * g);
                ob.abs_extra = (abs_r + abs_c) * g + 2.0 * a16;
                ob.terms = terms
                    .into_iter()
                    .map(|(m, a, gg, c)| (m, a, gg * g, c))
                    .collect();
                ob.round_own = own;
                ob.round_abs = (abs_r, abs_c);
                ob.round_f16 = Some(f16);
                ob.round_inner = Some(inner);
                // Equal or adjacent outputs: the interval [y - G_R, y + G_C] (length D = G_R +
                // G_C <= k u S + A) holds at most one rounding boundary when D is below the
                // gap between consecutive boundaries, which is > h (|y| - D) with h = u16 / 2
                // (the spacing at v exceeds u16 |v|; docs/equiv-mma.md §2). D (1 + h) < h |y|
                // holds for S <= kappa1 |y| and |y| >= y1, half the budget each.
                let ku = k * U32;
                let h = u16_ / 2.0;
                // (1 - 2^-20): strictly below.
                let kappa1 = h / 2.0 * (1.0 - 2f64.powi(-20)) / ((1.0 + h) * ku.max(1e-300));
                let y1 = 2.0 / h * (1.0 + h) * (abs_r + abs_c);
                ob.round_adjacent = (kappa1 >= 1.0).then_some((kappa1, y1));
            }
            _ => match rigorous(&cr, &cc) {
                Ok((k, kr, kc, terms)) => {
                    ob.bound_u = Some(if lv[0] { 0.0 } else { k });
                    ob.ref_u = Some(kr);
                    ob.cand_u = Some(kc);
                    ob.terms = terms;
                }
                Err(e) => ob.no_bound = Some(e),
            },
        }
        // L2: per monomial, candidate multiplier <= reference multiplier.
        if let Some((m, tr, tc)) = cr
            .poly
            .iter()
            .map(|(m, t)| (m, t.mult, cc.poly.get(m).map_or(0.0, |x| x.mult)))
            .chain(
                cc.poly
                    .iter()
                    .filter(|(m, _)| !cr.poly.contains_key(*m))
                    .map(|(m, t)| (m, 0.0, t.mult)),
            )
            .find(|(_, tr, tc)| tc > tr)
        {
            note!(
                2,
                format!(
                    "{}: candidate error multiplier {:.1}u > reference {:.1}u",
                    show_mono(m, names),
                    tc / U32,
                    tr / U32
                ),
            );
        }
        // L2 inside nonlinear operands: their input error budgets must not grow either.
        if let Some((id, tr, tc)) = cr.inner.iter().find_map(|(id, mr)| {
            let mc = cc.inner.get(id)?.values();
            let mr = mr.values();
            mr.keys()
                .chain(mc.keys())
                .map(|m| {
                    (
                        *id,
                        mr.get(m).copied().unwrap_or(0.0),
                        mc.get(m).copied().unwrap_or(0.0),
                    )
                })
                .find(|(_, tr, tc)| tc > tr)
        }) {
            note!(
                2,
                format!(
                    "inside nonlinear operand {id:#x}: candidate error multiplier {:.1}u > reference {:.1}u",
                    tc / U32,
                    tr / U32
                ),
            );
        }
        ob.level = if lv[0] {
            "L0"
        } else if lv[2] {
            "L2"
        } else {
            "L1"
        };
        per_output.push(ob);
        if tau_k.is_some() {
            let amplified = [&cr, &cc]
                .iter()
                .any(|c| c.inner.values().any(InnerBudget::amplified));
            if amplified || cr.abs_bound || cc.abs_bound {
                note!(
                    3,
                    "a MUFU op of a rounded operand, or with an absolute error bound: a tolerance \
                     bound would need |f'| (not proven)"
                        .into(),
                );
                continue;
            }
        }
        if let Some(k) = tau_k {
            // The largest relative bound of the output, so the report gives the least τ.
            if let Some((m, s)) = cr
                .poly
                .iter()
                .map(|(m, t)| {
                    let d = t.mult + cc.poly.get(m).map_or(0.0, |x| x.mult);
                    (
                        m,
                        if t.coef == 0.0 {
                            f64::INFINITY
                        } else {
                            d / t.coef.abs()
                        },
                    )
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                // Relative slack: the multipliers are sums of many f64 terms.
                .filter(|(_, s)| *s > k * U32 * (1.0 + 1e-6))
            {
                note!(
                    3,
                    format!(
                        "{}: d_R + d_C = {:.6}u > tolerance {k}u",
                        show_mono(m, names),
                        s / U32
                    ),
                );
            }
        }
    }
    let names_l = [
        "L0 bitwise",
        "L1 same function",
        "L2 no less accurate",
        "L3 tolerance",
    ];
    let levels = (0..4)
        .filter(|&l| l < 3 || tau_k.is_some())
        .map(|l| LevelResult {
            level: if l == 3 {
                format!("{} ({}u)", names_l[l], tau_k.unwrap())
            } else {
                names_l[l].into()
            },
            holds: fails[l] == 0 && n > 0,
            failures: fails[l],
            first_failure: first[l].as_ref().map(|x| x.0),
            detail: first[l].as_ref().map(|x| x.1.clone()),
        })
        .collect();
    Report {
        outputs: n,
        levels,
        witness_mono: witness,
        untracked,
        unordered,
        softmax: sm_detail.or(nm_detail),
        conditions: if tf32 { vec![TF32_COND.into()] } else { vec![] },
        per_output,
        l0_zero_sign,
    }
}

#[cfg(test)]
mod round16_tests {
    use super::*;

    fn xs(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                f32::from_bits((s >> 16) as u32 & 0x7fff_ffff | (s as u32 & 1) << 31) as f64
            })
            .filter(|v| v.is_finite())
            .collect()
    }

    /// `rnd16` of an f32 value is the hardware conversion (fp.rs, N-FALU / N-ROUND).
    #[test]
    fn rnd16_matches_the_conversions() {
        for v in xs(1 << 20, 0x1234_5678_9abc_def1) {
            let b = (v as f32).to_bits();
            let bf = f32::from_bits((crate::fp::f32_to_bf16_rn(b) as u32) << 16) as f64;
            if bf.is_finite() {
                assert_eq!(rnd16(v, false).to_bits(), bf.to_bits(), "bf16 {v:e}");
            }
            let h = crate::fp::f32_to_f16_rn(b);
            let hf = f32::from_bits(crate::elementwise::f16_to_f32_bits(h)) as f64;
            assert_eq!(rnd16(v, true).to_bits(), hf.to_bits(), "f16 {v:e}");
        }
    }

    /// Roundings of any a, b within gr, gc of y are within `round16_gap`; where the
    /// adjacency condition D (1 + h) < h |y| holds (h = u16 / 2), the gap is one step.
    #[test]
    fn round16_gap_covers_and_adjacency_holds() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for f16 in [false, true] {
            let (u16_, _) = round_unit(f16);
            let h = u16_ / 2.0;
            for _ in 0..200_000 {
                let y = (rnd() - 0.5) * 2f64.powi((rnd() * 40.0) as i32 - 20);
                let scale = 2f64.powi(-((rnd() * 16.0) as i32));
                let (gr, gc) = (
                    rnd() * u16_ * y.abs() * scale,
                    rnd() * u16_ * y.abs() * scale,
                );
                let Some((gap, steps)) = round16_gap(y, gr, gc, f16) else {
                    continue;
                };
                for _ in 0..4 {
                    let a = y + (2.0 * rnd() - 1.0) * gr;
                    let b = y + (2.0 * rnd() - 1.0) * gc;
                    let d = (rnd16(a, f16) - rnd16(b, f16)).abs();
                    assert!(
                        d <= gap,
                        "{f16} y {y:e} gr {gr:e} gc {gc:e}: {d:e} > {gap:e}"
                    );
                }
                let dd = gr + gc;
                if dd * (1.0 + h) < h * y.abs() {
                    assert!(
                        steps <= 1,
                        "{f16} y {y:e} gr {gr:e} gc {gc:e}: {steps} steps"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tf32_tests {
    use super::*;
    use crate::num::Node;
    use crate::num::Tf32Part;

    fn push(ns: &mut NumState, kind: NodeKind, hw: f32) -> u32 {
        ns.nodes.push(Node {
            kind,
            hw: hw.to_bits(),
            exact: vec![],
            err: vec![],
            pc: 0,
        });
        (ns.nodes.len() - 1) as u32
    }

    fn report(r: &NumState, rn: u32, c: &NumState, cn: u32) -> Report {
        let names = vec!["x".to_string()];
        let rs = Side {
            ns: r,
            outs: vec![(0, Some(rn))],
            buf_id: vec![0],
        };
        let cs = Side {
            ns: c,
            outs: vec![(0, Some(cn))],
            buf_id: vec![0],
        };
        compare(&rs, &cs, &names, None)
    }

    /// (x + y) - x against y: the x monomial cancels (coefficient 0) but carries the first
    /// add's error; dropping it bounded |C - R| by about 2u |y|, while x = 1e8, y = 1 gives 1.
    #[test]
    fn cancelled_monomial_keeps_its_error() {
        let mut r = NumState::default();
        let ry = push(&mut r, NodeKind::Input((0, 1)), 1.0);
        let mut c = NumState::default();
        c.segments = true;
        let x = push(&mut c, NodeKind::Input((0, 0)), 1e8);
        let y = push(&mut c, NodeKind::Input((0, 1)), 1.0);
        let s = push(&mut c, NodeKind::Add(x, y, false), 1e8);
        let one = push(&mut c, NodeKind::Leaf, 1.0);
        let d = push(
            &mut c,
            NodeKind::Fma {
                a: s,
                b: one,
                c: Some(x),
                neg_ab: false,
                neg_c: true,
            },
            0.0,
        );
        let rep = report(&r, ry, &c, d);
        assert!(rep.levels[1].holds, "{:?}", rep.levels);
        let ob = &rep.per_output[0];
        let at = |l: Leaf| if l.1 == 0 { 1e8 } else { 1.0 };
        let b = ob.bound_at(&at).expect("a bound");
        // fl(fl(1e8 + 1) - 1e8) = 0: |C - R| = 1.
        assert!(b >= 1.0, "bound {b} below the error 1");
    }

    /// A 3xTF32 chain hi*hi, lo'*hi, hi*lo' of the same a, b is a*b (1 + theta) with grown
    /// multipliers; a correction of other nodes (a, c) is untracked.
    #[test]
    fn tf32_triple_matches_only_same_nodes() {
        let eps = crate::fp::TMMA_MODEL.eps_terms(2);
        let build = |mismatch: bool| {
            let mut c = NumState::default();
            let a = push(&mut c, NodeKind::Input((0, 0)), 1.5);
            let b = push(&mut c, NodeKind::Input((0, 1)), 1.25);
            let bb = push(&mut c, NodeKind::Input((0, 2)), 1.75);
            let tf = |c: &mut NumState, arg, part| push(c, NodeKind::Tf32 { arg, part }, 1.0);
            let (ha, hb) = (tf(&mut c, a, Tf32Part::Hi), tf(&mut c, b, Tf32Part::Hi));
            let la = tf(&mut c, a, Tf32Part::LoR);
            let lb = tf(&mut c, if mismatch { bb } else { b }, Tf32Part::LoR);
            let hb2 = if mismatch {
                tf(&mut c, bb, Tf32Part::Hi)
            } else {
                hb
            };
            let zero = push(&mut c, NodeKind::Leaf, 0.0);
            let mma = |c: &mut NumState, acc, p, q| {
                push(
                    c,
                    NodeKind::Mma {
                        acc: Some(acc),
                        terms: vec![1.0],
                        ops: vec![Some((p, q))],
                        eps,
                        sc: vec![],
                        units: vec![],
                    },
                    1.0,
                )
            };
            let m1 = mma(&mut c, zero, la, hb);
            let m2 = mma(&mut c, m1, ha, lb);
            let m3 = mma(&mut c, m2, ha, hb2);
            (c, m3)
        };
        let mut r = NumState::default();
        r.segments = true;
        let ra = push(&mut r, NodeKind::Input((0, 0)), 1.5);
        let rb = push(&mut r, NodeKind::Input((0, 1)), 1.25);
        let rp = push(
            &mut r,
            NodeKind::Fma {
                a: ra,
                b: rb,
                c: None,
                neg_ab: false,
                neg_c: false,
            },
            1.875,
        );
        let (c, top) = build(false);
        let rep = report(&r, rp, &c, top);
        assert!(rep.levels[1].holds, "{:?}", rep.levels);
        assert!(rep.untracked.is_none(), "{:?}", rep.untracked);
        let k = rep.per_output[0].cand_u.unwrap() * U32;
        // theta = 2^-20 + 2^-10 (2 2^-11) = 2^-19; the two steps below the last round the
        // products apart (weight 1 + w), the last rounds their sum.
        let w = 2.0 * TF32_HI_EPS * (1.0 + 2f64.powi(-11));
        let g0 = 2f64.powi(-19) + (1.0 + w) * ((1.0 + eps).powi(2) - 1.0);
        let want = g0 * (1.0 + eps) + eps;
        // The true worst case: theta, the products rounded apart at three steps.
        assert!(k >= 2f64.powi(-19) + (1.0 + eps).powi(3) - 1.0, "k {k}");
        assert!((k - want).abs() <= want * 1e-9, "k {k} vs {want}");
        let (c, top) = build(true);
        let rep = report(&r, rp, &c, top);
        assert!(
            rep.untracked.is_some(),
            "mismatched correction must be untracked"
        );
    }

    /// theta: for x, y zero or at least 2^-103 in magnitude, hi(x) hi(y) + rh(lo(x)) hi(y) +
    /// hi(x) rh(lo(y)) is within (2^-20 + 2^-10 (2^-11 + 2^-11)) |x y| of x y, where hi clears
    /// the low 13 mantissa bits and rh is CUTLASS's +0x1000 then truncation.
    #[test]
    fn tf32_split_theta_holds() {
        let hi = |x: f32| f32::from_bits(crate::fp::tf32_in(x.to_bits()));
        let rh = |l: f32| f32::from_bits(crate::fp::tf32_in(l.to_bits().wrapping_add(0x1000)));
        let theta = 2f64.powi(-20) + TF32_HI_EPS * 2.0 * 2f64.powi(-11);
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut worst = 0f64;
        for i in 0..2_000_000u64 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let mut bits = [s as u32, (s >> 32) as u32];
            // Exponents from 2^-103 up, plus mantissas near the rounding boundaries.
            for (j, b) in bits.iter_mut().enumerate() {
                let e = 24 + ((*b >> 23) % 200);
                *b = (*b & 0x807f_ffff) | e << 23;
                if (i >> j) & 3 == 0 {
                    *b = (*b & !0x1fff) | 0x0fff | (*b & 0x1000);
                }
            }
            let (x, y) = (f32::from_bits(bits[0]), f32::from_bits(bits[1]));
            let (hx, hy) = (hi(x), hi(y));
            let (lx, ly) = (x - hx, y - hy);
            assert_eq!(lx as f64, x as f64 - hx as f64, "lo(x) exact");
            let t = hx as f64 * hy as f64 + rh(lx) as f64 * hy as f64 + hx as f64 * rh(ly) as f64;
            let e = (t - x as f64 * y as f64).abs() / (x as f64 * y as f64).abs();
            worst = worst.max(e);
            assert!(e <= theta, "x {x:e} y {y:e}: {e:e} > {theta:e}");
        }
        assert!(worst > theta / 4.0, "worst {worst:e}");
    }
}

#[cfg(test)]
mod exact_tests {
    use ptxv_ir::Lp;

    use super::ExactSet;

    fn rep(v: f64, f: ExactSet) -> bool {
        // v exactly representable in the format f describes (finite, s bits, spacing 2^lsb).
        if v == 0.0 {
            return true;
        }
        let a = v.abs();
        if a >= 2f64.powi(f.top) {
            return false;
        }
        let e = a.log2().floor() as i32;
        let q = 2f64.powi((e - f.s + 1).max(f.lsb));
        (a / q).fract() == 0.0
    }

    /// N-EXACT: the sets' parameters are attained by the formats, and every product a set
    /// claims exact in f32 / bf16 is exact there, by brute force over all code pairs; the
    /// MXF8 dequantization (e4m3 x ue8m0) is not claimed exact in bf16, and is not.
    #[test]
    fn exact_sets_are_exact() {
        let set = |t| ExactSet::of(t);
        assert_eq!(
            set(Lp::E2m1),
            ExactSet {
                s: 2,
                lsb: -1,
                top: 3
            }
        );
        assert_eq!(
            set(Lp::E4m3),
            ExactSet {
                s: 4,
                lsb: -9,
                top: 9
            }
        );
        assert_eq!(
            set(Lp::Ue4m3),
            ExactSet {
                s: 4,
                lsb: -9,
                top: 9
            }
        );
        assert_eq!(
            set(Lp::Ue8m0),
            ExactSet {
                s: 1,
                lsb: -127,
                top: 128
            }
        );
        let codes = |t: Lp| -> Vec<f64> {
            (0..1u32 << crate::lowp::width(t))
                .map(|c| crate::lowp::value(c, t))
                .filter(|v| v.is_finite())
                .collect()
        };
        let pairs = [
            (Lp::E2m1, Lp::Ue4m3),
            (Lp::E2m1, Lp::E2m1),
            (Lp::E4m3, Lp::Ue8m0),
            (Lp::E4m3, Lp::E4m3),
            (Lp::E2m1, Lp::Ue8m0),
            (Lp::E5m2, Lp::Ue4m3),
        ];
        for (ta, tb) in pairs {
            let p = set(ta).mul(set(tb));
            for (f, name) in [
                (ExactSet::F32, "f32"),
                (ExactSet::BF16, "bf16"),
                (ExactSet::F16, "f16"),
            ] {
                let all = codes(ta)
                    .iter()
                    .all(|x| codes(tb).iter().all(|y| rep(x * y, f)));
                if p.fits(f) {
                    assert!(all, "{ta:?} x {tb:?} claimed exact in {name}");
                }
                if (ta, tb, name) == (Lp::E4m3, Lp::Ue8m0, "bf16") {
                    assert!(!p.fits(f) && !all);
                }
            }
            // Each leaf alone, rounded to bf16 / f16.
            for t in [ta, tb] {
                for (f, name) in [(ExactSet::BF16, "bf16"), (ExactSet::F16, "f16")] {
                    if set(t).fits(f) {
                        assert!(codes(t).iter().all(|x| rep(*x, f)), "{t:?} in {name}");
                    }
                }
            }
        }
        assert!(set(Lp::E2m1).mul(set(Lp::Ue4m3)).fits(ExactSet::BF16));
    }
}

#[cfg(test)]
mod soundness_tests {
    use std::sync::Arc;

    use ptxv_ir::File;
    use ptxv_ir::IntOp;
    use ptxv_ir::Op;
    use ptxv_ir::Reg;
    use ptxv_ir::Val;

    use super::*;
    use crate::num::Node;
    use crate::num::SegRef;
    use crate::seg::SKind;
    use crate::seg::SOp;
    use crate::seg::SProg;
    use crate::seg::SVal;

    fn push(ns: &mut NumState, kind: NodeKind, hw: f32) -> u32 {
        let id = ns.nodes.len() as u32;
        ns.nodes.push(Node {
            kind,
            hw: hw.to_bits(),
            exact: vec![],
            err: vec![],
            pc: id,
        });
        id
    }

    fn compare_nodes(ns: &NumState, a: u32, b: u32) -> Report {
        let side = |n| Side {
            ns,
            outs: vec![(ns.nodes[n as usize].hw, Some(n))],
            buf_id: vec![0],
        };
        compare(&side(a), &side(b), &["x".into()], None)
    }

    fn sign_slice(ns: &mut NumState, arg: u32) -> u32 {
        let r = |idx| Reg { file: File::R, idx };
        let prog = SProg {
            args: vec![arg],
            ops: vec![SOp {
                pc: 0,
                kind: SKind::Inst {
                    op: Op::Int {
                        op: IntOp::ShrU,
                        bits: 32,
                        d: r(1),
                        a: Val::Reg(r(0), 1),
                        b: Val::Imm(31),
                    },
                    dst: r(1),
                    regs: vec![(r(0), SVal::Arg(0))],
                    cvals: vec![],
                },
            }],
            out: SVal::Op(0),
            extra_pcs: Default::default(),
        };
        push(ns, NodeKind::Seg(SegRef(Arc::new(prog))), 0.0)
    }

    #[test]
    fn coefficient_precision_loss_cannot_prove_cancellation() {
        let mut ns = NumState::default();
        ns.segments = true;
        let x = push(&mut ns, NodeKind::Input((0, 0)), 0.0);
        let one = push(&mut ns, NodeKind::Leaf, 1.0);
        let tiny = push(&mut ns, NodeKind::Leaf, 2f32.powi(-60));
        let plus = push(
            &mut ns,
            NodeKind::Fma {
                a: x,
                b: tiny,
                c: Some(x),
                neg_ab: false,
                neg_c: false,
            },
            0.0,
        );
        let result = push(
            &mut ns,
            NodeKind::Fma {
                a: plus,
                b: one,
                c: Some(x),
                neg_ab: false,
                neg_c: true,
            },
            0.0,
        );
        let zero = push(&mut ns, NodeKind::Leaf, 0.0);
        let report = compare_nodes(&ns, result, zero);
        assert!(!report.levels[1].holds, "{report:?}");
        assert!(!report.levels[2].holds, "{report:?}");
        assert!(
            report
                .untracked
                .unwrap()
                .contains("coefficient cannot be represented exactly")
        );
        assert!(coefficient_product(1.0 + 2f64.powi(-52), 1.0 + 2f64.powi(-52)).is_nan());
        assert_eq!(coefficient_sum(1.0, -1.0), 0.0);
        assert_eq!(coefficient_product(0.5, 2.0), 1.0);
    }

    #[test]
    fn polynomial_argument_boundaries_are_unambiguous() {
        let mut poly = BTreeMap::new();
        poly.insert(vec![(0, 0)], Term::one());
        let empty = BTreeMap::new();
        let mut a = H2::new();
        let mut b = H2::new();
        poly_id(&mut a, &poly);
        poly_id(&mut a, &empty);
        poly_id(&mut b, &empty);
        poly_id(&mut b, &poly);
        assert_ne!(a.done(), b.done());
    }

    #[test]
    fn bit_observing_slices_preserve_signed_zero_arguments() {
        let mut ns = NumState::default();
        ns.segments = true;
        let x = push(&mut ns, NodeKind::Input((0, 0)), 1.0);
        let one = push(&mut ns, NodeKind::Leaf, 1.0);
        let z = push(
            &mut ns,
            NodeKind::Fma {
                a: x,
                b: one,
                c: Some(x),
                neg_ab: false,
                neg_c: true,
            },
            0.0,
        );
        let pos = push(&mut ns, NodeKind::Leaf, 0.0);
        let neg = push(&mut ns, NodeKind::Leaf, -0.0);
        let a = push(
            &mut ns,
            NodeKind::Fma {
                a: x,
                b: z,
                c: Some(pos),
                neg_ab: false,
                neg_c: false,
            },
            0.0,
        );
        let b = push(
            &mut ns,
            NodeKind::Fma {
                a: x,
                b: z,
                c: Some(neg),
                neg_ab: false,
                neg_c: false,
            },
            0.0,
        );
        let sa = sign_slice(&mut ns, a);
        let sb = sign_slice(&mut ns, b);
        assert!(!compare_nodes(&ns, sa, sb).levels[1].holds);
        assert!(compare_nodes(&ns, sa, sa).levels[0].holds);
    }

    #[test]
    fn deferred_budgets_preserve_grouping_and_release_intermediates() {
        let terms = |v| InnerBudget::terms(BTreeMap::from([(vec![(0, 0)], v)]));
        let big = terms(1e16);
        let one = terms(1.0);
        let mut left = big.clone();
        left.merge(&one, false);
        let intermediate = left.clone();
        left.merge(&one, false);
        let mut two = one.clone();
        two.merge(&one, false);
        let mut right = big.clone();
        right.merge(&two, false);
        let coefficient = |x: &InnerBudget| x.values()[&vec![(0, 0)]];
        assert_eq!(coefficient(&left), 1e16);
        assert!(intermediate.0.as_ref().unwrap().value.get().is_none());
        assert_eq!(coefficient(&right), 1e16 + 2.0);
        let mut maximum = left.clone();
        maximum.merge(&right, true);
        assert_eq!(coefficient(&maximum), 1e16 + 2.0);
        let mut shared = maximum.clone();
        shared.merge(&maximum, false);
        assert_eq!(coefficient(&shared), 2.0 * (1e16 + 2.0));
        assert_eq!(coefficient(&intermediate), 1e16);
    }

    #[test]
    fn maxima_keep_their_operands_error_budgets() {
        let mut ns = NumState::default();
        ns.segments = true;
        let x = push(&mut ns, NodeKind::Input((0, 0)), 1.0);
        let y = push(&mut ns, NodeKind::Input((0, 1)), 16777216.0);
        let zero = push(&mut ns, NodeKind::Leaf, 0.0);
        let one = push(&mut ns, NodeKind::Leaf, 1.0);
        let sum = push(&mut ns, NodeKind::Add(x, y, false), 16777216.0);
        let difference = push(
            &mut ns,
            NodeKind::Fma {
                a: y,
                b: one,
                c: Some(sum),
                neg_ab: true,
                neg_c: false,
            },
            0.0,
        );
        let r = push(
            &mut ns,
            NodeKind::Max {
                a: x,
                b: zero,
                min: false,
            },
            1.0,
        );
        let c = push(
            &mut ns,
            NodeKind::Max {
                a: difference,
                b: zero,
                min: false,
            },
            0.0,
        );
        let report = compare_nodes(&ns, r, c);
        assert!(report.levels[1].holds, "{report:?}");
        assert!(!report.levels[2].holds, "{report:?}");
        assert!(compare_nodes(&ns, r, r).levels[2].holds);
        let rr = push(&mut ns, NodeKind::Add(r, r, false), 2.0);
        let rc = push(&mut ns, NodeKind::Add(r, c, false), 1.0);
        let cc = push(&mut ns, NodeKind::Add(c, c, false), 0.0);
        assert!(compare_nodes(&ns, rr, rc).levels[1].holds);
        assert!(!compare_nodes(&ns, rr, rc).levels[2].holds);
        assert!(!compare_nodes(&ns, rc, cc).levels[2].holds);
        assert!(compare_nodes(&ns, cc, rc).levels[2].holds);
        let nested = push(
            &mut ns,
            NodeKind::Max {
                a: r,
                b: c,
                min: false,
            },
            1.0,
        );
        assert!(compare_nodes(&ns, c, nested).levels[2].holds);
    }
}

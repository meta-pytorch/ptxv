// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Concolic input exploration (`--concolic`): for each untaken side of an input-dependent
//! comparison (coverage.rs A), a goal on the expressions of symb.rs: the predicate of a
//! thread that produced the other value must take the untaken value, the atoms of that
//! thread's path condition that share inputs with it keeping their values where possible.
//! Solvers, cheapest first:
//! - `T0`: inversion of the condition through the chain of operations above one input leaf
//!   (modular inverse of odd multipliers, shifts, masks, extensions, extraction), exact for
//!   equality targets, boundary candidates for orderings;
//! - `search`: bounded search over one or two leaves (exhaustive for one leaf of at most 16
//!   bits and two of at most 8), candidates from the condition's constants, the captured
//!   values, declared invariants and special values;
//! - `T3`: local search on the distance of the deciding comparison's operands (float or
//!   integer), with bisection across a sign change of that distance;
//! - `smt` (cargo feature `smt`): the cone as SMT-LIB2 (bit-vectors, IEEE floats, MUFU and
//!   `Inst` nodes as uninterpreted functions with sign / special-value axioms) for an external
//!   solver (`PTXV_SMT_SOLVER`, default `z3`).
//!
//! No answer is trusted. A solver's assignment is evaluated on the expressions, written into
//! the input (respecting the declared invariants, coverage.rs `Domain`; values outside them are
//! their own class), and the mutated input is run again with the coverage shadow
//! (`coverage::apply`, `coverage::analyze`): the side is reported flipped only when that run
//! takes it. A solver's "no solution" is `solver-unsat (<solver>, unverified)`, never a proof:
//! it covers the expressions of the observed path, with unmodelled values fixed.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;

use serde::Serialize;

use crate::coverage::Coverage;
use crate::coverage::Domain;
use crate::coverage::Mutation;
use crate::machine::Machine;
use crate::symb::Bin;
use crate::symb::E;
use crate::symb::Evaluator;
use crate::symb::Node;
use crate::symb::SymState;
use crate::symb::Un;
use crate::symb::mask;

/// Options of the concolic search.
#[derive(Clone, Debug)]
pub struct Options {
    /// Elements per mutation (threads in different warps first), as `--mutate-elements`.
    pub k: usize,
    /// Observations tried per side.
    pub tries: usize,
    /// Replays (coverage runs of mutated inputs) in total.
    pub max_replays: usize,
    /// Wall-clock budget for the whole search, seconds.
    pub budget: f64,
    /// Expression evaluations per solver per goal.
    pub evals: usize,
    /// Run the external SMT solver (feature `smt`).
    pub smt: bool,
    /// Seconds per SMT query.
    pub smt_timeout: f64,
    /// Testing: faults injected into the solvers (the gates must catch what follows).
    pub fault: Fault,
    /// Solvers to run (empty: all, in order T0, search, T3, smt).
    pub only: Vec<String>,
}

/// Faults for mutation-testing the solvers and their gates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Fault {
    #[default]
    None,
    /// T0's modular inverse is wrong.
    ModInv,
    /// T0's modular inverse is wrong and the evaluation check accepts anything: the replay
    /// is the only gate.
    ModInvNoEval,
    /// Every witness value is corrupted after solving (bit 0 flipped).
    Writes,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            k: 2,
            tries: 6,
            max_replays: 400,
            budget: 600.0,
            evals: 20_000,
            smt: cfg!(feature = "smt"),
            smt_timeout: 5.0,
            fault: Fault::None,
            only: vec![],
        }
    }
}

/// Per-solver counts (`rows[].input_coverage.concolic.solvers`).
#[derive(Clone, Debug, Default, Serialize)]
pub struct SolverCount {
    /// Goals given to the solver.
    pub tried: usize,
    /// Assignments it returned that satisfy the goal on the expressions.
    pub sat: usize,
    /// Sides whose replay took the flipped side with its assignment.
    pub confirmed: usize,
    /// Replays of its assignments that did not take the flipped side.
    pub rejected: usize,
    /// Goals it reported without solution.
    pub unsat: usize,
    /// Assignments that did not satisfy the goal on the expressions (not replayed).
    pub wrong: usize,
}

/// The outcome for one untaken side.
#[derive(Clone, Debug, Serialize)]
pub struct Side {
    pub pc: u32,
    pub at: String,
    /// The predicate value never produced on the captured input.
    pub never: bool,
    /// `flipped`, `outside-invariants`, `unknown`, `solver-unsat`.
    pub outcome: String,
    /// The solver whose witness the replay confirmed, or that reported no solution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solver: Option<String>,
    /// The confirmed witness, `buffer[element]=value` per element.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub witness: Option<String>,
    /// Every element written lies in a buffer whose declared invariant admits the value.
    pub within_invariants: bool,
    /// The condition of the first observation tried (text, depth-limited).
    pub condition: String,
    /// Why not flipped (unknown, solver-unsat).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Differential (`PTXV_CONCOLIC_DIFF`): did B's mutation for this side replay-flip it
    /// (None: B proposed none).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b_flip: Option<bool>,
    /// B's refusal reason (differential).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b_refusal: Option<String>,
    #[serde(skip)]
    pub writes: Vec<(usize, u64, u8, u64)>,
}

/// The concolic search's result (`rows[].input_coverage.concolic`).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Report {
    pub sides: Vec<Side>,
    pub flipped: usize,
    pub outside_invariants: usize,
    pub unknown: usize,
    pub solver_unsat: usize,
    pub solvers: BTreeMap<String, SolverCount>,
    /// Expression nodes, `Inst` destinations, opaque destinations by reason.
    pub nodes: usize,
    pub inst_nodes: usize,
    pub opaque: BTreeMap<String, usize>,
    /// Nodes that disagreed with the simulator on the captured input (each replaced).
    pub mismatches: usize,
    pub path_atoms: usize,
    pub replays: usize,
    /// Seconds of the coverage run with expressions, and of the search with its replays.
    pub seconds_run: f64,
    pub seconds_search: f64,
}

/// Input writes: (allocation, byte offset, bytes, value).
type Writes = Vec<(usize, u64, u8, u64)>;

/// `replay_b`'s result: per (pc, never) (replayed flip, label); refusals by pc.
pub type BDiff = (BTreeMap<(u32, bool), (bool, String)>, BTreeMap<u32, String>);

/// A goal: `root == want`, keeping `prefix` atoms (condition, value) where possible.
struct Goal {
    root: E,
    want: bool,
    prefix: Vec<(E, bool)>,
    /// Input leaves of the root's cone (the variables).
    vars: Vec<E>,
}

enum Ans {
    /// An assignment satisfying the goal on the expressions; whether it keeps the prefix.
    Sat(HashMap<E, u64>, bool),
    Unsat,
    Unknown(String),
}

/// The leaf (alloc, off, bytes) of an input node.
fn leaf_of(sym: &SymState, e: E) -> Option<(u32, u64, u8)> {
    match sym.ar.nodes[e as usize] {
        Node::Input { alloc, off, bytes } => Some((alloc, off, bytes)),
        _ => None,
    }
}

struct Ctx<'a> {
    sym: &'a SymState,
    ev: Evaluator<'a>,
    m0: &'a Machine,
    dom: &'a Domain,
    opts: &'a Options,
    cones: HashMap<E, Vec<E>>,
    leaves: HashMap<E, HashSet<E>>,
    evals: usize,
}

impl<'a> Ctx<'a> {
    fn cone(&mut self, e: E) -> Vec<E> {
        if let Some(c) = self.cones.get(&e) {
            return c.clone();
        }
        let c = self.sym.ar.cone(e);
        self.cones.insert(e, c.clone());
        c
    }

    fn leafset(&mut self, e: E) -> HashSet<E> {
        if let Some(l) = self.leaves.get(&e) {
            return l.clone();
        }
        let l: HashSet<E> = self
            .sym
            .ar
            .leaves(e)
            .into_iter()
            .filter(|&x| matches!(self.sym.ar.nodes[x as usize], Node::Input { .. }))
            .collect();
        self.leaves.insert(e, l.clone());
        l
    }

    fn eval(&mut self, e: E, env: &HashMap<E, u64>) -> Option<u64> {
        self.evals += 1;
        let c = self.cone(e);
        self.ev.eval_cone(&c, env)
    }

    /// Does `env` satisfy the goal (root), and does it keep the prefix?
    fn check(&mut self, g: &Goal, env: &HashMap<E, u64>) -> (bool, bool) {
        if self.opts.fault == Fault::ModInvNoEval {
            return (true, true);
        }
        let ok = self.eval(g.root, env).map(|v| v & 1 == 1) == Some(g.want);
        if !ok {
            return (false, false);
        }
        let keep = g
            .prefix
            .iter()
            .all(|&(c, v)| self.eval(c, env).map(|x| x & 1 == 1) == Some(v));
        (true, keep)
    }

    /// Does the declared invariant admit `v` for leaf `e`?
    fn admits(&self, e: E, v: u64) -> bool {
        match leaf_of(self.sym, e) {
            Some((a, off, b)) => self.dom.admits(self.m0, a as usize, off, b, v) != Some(false),
            None => false,
        }
    }

    fn env_admitted(&self, env: &HashMap<E, u64>) -> bool {
        env.iter().all(|(&e, &v)| self.admits(e, v))
    }
}

// ---------------------------------------------------------------- T0: inversion

fn modinv(a: u64, w: u8) -> u64 {
    // Newton iteration for the inverse of an odd a modulo 2^64, then truncated.
    let mut x = a;
    for _ in 0..6 {
        x = x.wrapping_mul(2u64.wrapping_sub(a.wrapping_mul(x)));
    }
    x & mask(w)
}

/// Candidate values of a `w`-bit operand x with `cmp(x, c) == t` (x on the left) near the
/// boundary.
fn cmp_targets(cmp: ptxv_ir::Cmp, w: u8, c: u64, t: bool, left: bool) -> Vec<u64> {
    let m = mask(w);
    let smax = m >> 1;
    let smin = smax + 1;
    let base = [
        c,
        c.wrapping_add(1),
        c.wrapping_sub(1),
        c.wrapping_add(2),
        c.wrapping_sub(2),
        0,
        1,
        m,
        smax,
        smin,
        c.wrapping_add(16),
        c.wrapping_sub(16),
    ];
    let mut v: Vec<u64> = Vec::new();
    for x in base {
        let x = x & m;
        let r = if left {
            crate::symb::cmp_eval(cmp, w, x, c)
        } else {
            crate::symb::cmp_eval(cmp, w, c, x)
        };
        if r == t && !v.contains(&x) {
            v.push(x);
        }
    }
    v
}

/// Assignments making `e` take a value in `targets` by inverting the operations above one
/// input leaf; `exact` is cleared when a step's candidate set is not the whole preimage.
fn invert(
    cx: &mut Ctx,
    e: E,
    targets: &[u64],
    exact: &mut bool,
    depth: usize,
) -> Vec<HashMap<E, u64>> {
    if targets.is_empty() || depth > 64 {
        return vec![];
    }
    let ar = &cx.sym.ar;
    let w = ar.width[e as usize];
    let m = mask(w);
    let conc = |x: E| ar.conc[x as usize];
    let n = ar.nodes[e as usize].clone();
    let mut out: Vec<HashMap<E, u64>> = Vec::new();
    fn push1(out: &mut Vec<HashMap<E, u64>>, sub: Vec<HashMap<E, u64>>) {
        for s in sub {
            if out.len() < 32 {
                out.push(s);
            }
        }
    }
    let free = |cx: &mut Ctx, x: E| !cx.leafset(x).is_empty();
    match n {
        Node::Input { .. } => {
            for &v in targets {
                out.push(HashMap::from([(e, v & m)]));
            }
        }
        Node::Const { .. } | Node::Opaque { .. } | Node::Sreg { .. } => {
            if targets.contains(&conc(e)) {
                out.push(HashMap::new());
            }
        }
        Node::Un { op, a, .. } => {
            let t: Vec<u64> = match op {
                Un::Not => targets.iter().map(|&v| !v & m).collect(),
                Un::Neg | Un::FNeg => targets
                    .iter()
                    .map(|&v| crate::symb::un_eval(op, w, v))
                    .collect(),
                Un::B2F => targets
                    .iter()
                    .filter(|&&v| v & 0xffff == 0)
                    .map(|&v| v >> 16)
                    .collect(),
                _ => {
                    *exact = false;
                    return vec![];
                }
            };
            push1(&mut out, invert(cx, a, &t, exact, depth + 1));
        }
        Node::Bin { op, a, b, .. } => {
            let (ca, cb) = (conc(a), conc(b));
            let (fa, fb) = (free(cx, a), free(cx, b));
            if fa && fb {
                *exact = false;
                return vec![];
            }
            let (x, k, x_left) = if fa { (a, cb, true) } else { (b, ca, false) };
            let cx_conc = if fa { ca } else { cb };
            let mut t: Vec<u64> = Vec::new();
            for &v in targets {
                let cand: Option<u64> = match (op, x_left) {
                    (Bin::Add, _) => Some(v.wrapping_sub(k)),
                    (Bin::Sub, true) => Some(v.wrapping_add(k)),
                    (Bin::Sub, false) => Some(k.wrapping_sub(v)),
                    (Bin::Xor, _) => Some(v ^ k),
                    (Bin::Mul, _) => {
                        let k = k & m;
                        if k == 0 {
                            None
                        } else {
                            let s = k.trailing_zeros().min(63) as u8;
                            let o = k >> s;
                            if (v & mask(s)) != 0 {
                                None
                            } else {
                                let mut inv = modinv(o, w);
                                if matches!(cx.opts.fault, Fault::ModInv | Fault::ModInvNoEval) {
                                    inv = inv.wrapping_add(2);
                                }
                                let ws = w.saturating_sub(s);
                                let low = ((v >> s).wrapping_mul(inv)) & mask(ws);
                                Some(low | (cx_conc & !mask(ws)))
                            }
                        }
                    }
                    (Bin::Shl, true) => {
                        let k = k.min(64) as u8;
                        if k >= w || v & mask(k) != 0 {
                            None
                        } else {
                            Some((v >> k) | (cx_conc & !(m >> k)))
                        }
                    }
                    (Bin::LShr, true) => {
                        let k = k.min(64) as u8;
                        if k >= w || v > (m >> k) {
                            None
                        } else {
                            Some((v << k) | (cx_conc & mask(k)))
                        }
                    }
                    (Bin::AShr, true) => {
                        let k = k.min(64) as u8;
                        if k >= w {
                            None
                        } else {
                            *exact = false;
                            Some(((v << k) | (cx_conc & mask(k))) & m)
                        }
                    }
                    (Bin::And, _) => {
                        if v & !k & m != 0 {
                            None
                        } else {
                            Some(v | (cx_conc & !k))
                        }
                    }
                    (Bin::Or, _) => {
                        if v & k != k & m {
                            None
                        } else {
                            Some((v & !k) | (cx_conc & k))
                        }
                    }
                    _ => {
                        *exact = false;
                        None
                    }
                };
                if let Some(c) = cand {
                    t.push(c & m);
                }
            }
            push1(&mut out, invert(cx, x, &t, exact, depth + 1));
        }
        Node::Zext { a, .. } => {
            let wa = ar.width[a as usize];
            let t: Vec<u64> = targets.iter().copied().filter(|&v| v <= mask(wa)).collect();
            push1(&mut out, invert(cx, a, &t, exact, depth + 1));
        }
        Node::Sext { a, .. } => {
            let wa = ar.width[a as usize];
            let t: Vec<u64> = targets
                .iter()
                .copied()
                .filter(|&v| {
                    let x = v & mask(wa);
                    let s = ((x << (64 - wa as u32)) as i64 >> (64 - wa as u32)) as u64 & m;
                    s == v
                })
                .map(|v| v & mask(wa))
                .collect();
            push1(&mut out, invert(cx, a, &t, exact, depth + 1));
        }
        Node::Extract { lo, a, .. } => {
            let ca = conc(a);
            let t: Vec<u64> = targets
                .iter()
                .map(|&v| (ca & !(m << lo)) | (v << lo))
                .collect();
            push1(&mut out, invert(cx, a, &t, exact, depth + 1));
        }
        Node::Concat { hi, lo } => {
            let wl = ar.width[lo as usize];
            let (chi, clo) = (conc(hi), conc(lo));
            let (fh, fl) = (free(cx, hi), free(cx, lo));
            for &v in targets {
                let (vh, vl) = (v >> wl, v & mask(wl));
                let sh = if fh {
                    invert(cx, hi, &[vh], exact, depth + 1)
                } else if chi == vh {
                    vec![HashMap::new()]
                } else {
                    vec![]
                };
                let sl = if fl {
                    invert(cx, lo, &[vl], exact, depth + 1)
                } else if clo == vl {
                    vec![HashMap::new()]
                } else {
                    vec![]
                };
                if let (Some(a1), Some(b1)) = (sh.first(), sl.first()) {
                    let mut env = a1.clone();
                    let clash = b1
                        .iter()
                        .any(|(k2, v2)| env.insert(*k2, *v2).is_some_and(|o| o != *v2));
                    if !clash {
                        out.push(env);
                    }
                }
            }
        }
        Node::Cmp { cmp, a, b } => {
            let (ca, cb) = (conc(a), conc(b));
            let wa = ar.width[a as usize];
            let (fa, fb) = (free(cx, a), free(cx, b));
            if fa && fb {
                *exact = false;
                return vec![];
            }
            for &t in targets {
                let t = t & 1 == 1;
                if !matches!(
                    (cmp, t),
                    (ptxv_ir::Cmp::Eq, true) | (ptxv_ir::Cmp::Ne, false)
                ) {
                    *exact = false;
                }
                let (x, c, left) = if fa { (a, cb, true) } else { (b, ca, false) };
                let vals = cmp_targets(cmp, wa, c, t, left);
                push1(&mut out, invert(cx, x, &vals, exact, depth + 1));
            }
        }
        // The address stays as captured (the evaluation check rejects any other).
        Node::AtAddr { val, .. } => push1(&mut out, invert(cx, val, targets, exact, depth + 1)),
        Node::Ite { c, a, b } => {
            // Keep the selector: invert the selected value; or flip the selector when the
            // other value is already a target.
            let cv = conc(c) & 1 == 1;
            let (sel, oth) = if cv { (a, b) } else { (b, a) };
            let co = conc(oth);
            push1(&mut out, invert(cx, sel, targets, exact, depth + 1));
            if targets.contains(&co) {
                push1(&mut out, invert(cx, c, &[!cv as u64], exact, depth + 1));
            }
            *exact = false;
        }
        _ => {
            *exact = false;
        }
    }
    out
}

/// Flatten a boolean goal (Not / And / Or / Xor of atoms) into choices: each choice is a set
/// of (atom, value) targets that together give `want`, the other atoms held.
fn bool_choices(cx: &mut Ctx, e: E, want: bool, out: &mut Vec<Vec<(E, bool)>>, depth: usize) {
    if depth > 16 || out.len() > 16 {
        return;
    }
    let n = cx.sym.ar.nodes[e as usize].clone();
    let conc = |cx: &Ctx, x: E| cx.sym.ar.conc[x as usize] & 1 == 1;
    match n {
        Node::Un { op: Un::Not, a, .. } => bool_choices(cx, a, !want, out, depth + 1),
        Node::Bin { op, w: 1, a, b } if matches!(op, Bin::And | Bin::Or | Bin::Xor) => {
            let (ca, cb) = (conc(cx, a), conc(cx, b));
            let fa = !cx.leafset(a).is_empty();
            let fb = !cx.leafset(b).is_empty();
            let mut subs: Vec<Vec<(E, bool)>> = Vec::new();
            match (op, want) {
                (Bin::And, true) | (Bin::Or, false) => {
                    let mut req = Vec::new();
                    if ca != want {
                        req.push((a, want));
                    }
                    if cb != want {
                        req.push((b, want));
                    }
                    if (ca == want || fa) && (cb == want || fb) {
                        subs.push(req);
                    }
                }
                (Bin::And, false) | (Bin::Or, true) => {
                    if fa {
                        subs.push(vec![(a, want)]);
                    }
                    if fb {
                        subs.push(vec![(b, want)]);
                    }
                }
                _ => {
                    if fa {
                        subs.push(vec![(a, want ^ cb)]);
                    }
                    if fb {
                        subs.push(vec![(b, want ^ ca)]);
                    }
                }
            }
            for s in subs {
                if s.len() == 1 {
                    let (x, v) = s[0];
                    bool_choices(cx, x, v, out, depth + 1);
                } else {
                    out.push(s);
                }
            }
        }
        _ => out.push(vec![(e, want)]),
    }
}

fn t0(cx: &mut Ctx, g: &Goal) -> Ans {
    let mut choices = Vec::new();
    bool_choices(cx, g.root, g.want, &mut choices, 0);
    let mut fallback: Option<HashMap<E, u64>> = None;
    // "No solution" only for one exact inversion of one leaf.
    let mut proof = choices.len() == 1 && choices[0].len() == 1 && cx.leafset(g.root).len() == 1;
    for ch in choices {
        let mut envs: Vec<HashMap<E, u64>> = vec![HashMap::new()];
        let mut exact = true;
        for (atom, v) in ch {
            let sub = invert(cx, atom, &[v as u64], &mut exact, 0);
            let mut next = Vec::new();
            for e1 in &envs {
                for e2 in &sub {
                    let mut env = e1.clone();
                    if e2
                        .iter()
                        .all(|(k, v)| env.insert(*k, *v).is_none_or(|o| o == *v))
                    {
                        next.push(env);
                    }
                }
            }
            envs = next;
            envs.truncate(64);
        }
        if !envs.is_empty() || !exact {
            proof = false;
        }
        envs.sort_by_key(|e| !cx.env_admitted(e));
        for env in envs {
            if env.is_empty() {
                continue;
            }
            let (ok, keep) = cx.check(g, &env);
            if ok && keep && cx.env_admitted(&env) {
                return Ans::Sat(env, true);
            }
            if ok && fallback.is_none() {
                fallback = Some(env);
            }
        }
    }
    if let Some(env) = fallback {
        let keep = cx.check(g, &env).1;
        return Ans::Sat(env, keep);
    }
    if proof {
        return Ans::Unsat;
    }
    Ans::Unknown("no inversion".into())
}

// ---------------------------------------------------------------- search

/// Constants of a cone (values worth trying for its leaves).
fn cone_consts(cx: &mut Ctx, e: E) -> Vec<u64> {
    let c = cx.cone(e);
    let mut v = Vec::new();
    for x in c {
        if let Node::Const { v: k, .. } = cx.sym.ar.nodes[x as usize] {
            v.push(k);
        }
    }
    v.sort_unstable();
    v.dedup();
    v.truncate(64);
    v
}

fn f32_specials() -> Vec<u64> {
    [
        0u32,
        0x8000_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x0080_0000,
        0x0000_0001,
        0x4000_0000,
        0xc000_0000,
        0x42b0_0000, // 88
        0x4300_0000, // 128
        0xc2fc_0000, // -126
        0x3f00_0000,
        0x4b80_0000, // 2^24
    ]
    .iter()
    .map(|&x| x as u64)
    .collect()
}

fn leaf_cands(cx: &mut Ctx, leaf: E, consts: &[u64]) -> Vec<u64> {
    let w = cx.sym.ar.width[leaf as usize];
    let m = mask(w);
    let c = cx.sym.ar.conc[leaf as usize];
    let mut v: Vec<u64> = vec![
        c,
        c ^ 1,
        c.wrapping_add(1),
        c.wrapping_sub(1),
        0,
        1,
        2,
        m,
        m >> 1,
        (m >> 1) + 1,
    ];
    if let Some((a, _, _)) = leaf_of(cx.sym, leaf) {
        v.extend(cx.dom.cands(a as usize));
    }
    for &k in consts {
        v.extend([k, k.wrapping_add(1), k.wrapping_sub(1), k.wrapping_neg()]);
    }
    if w == 32 {
        v.extend(f32_specials());
        for &k in consts {
            let k = k as u32;
            v.extend([
                k.wrapping_add(1) as u64,
                k.wrapping_sub(1) as u64,
                (k ^ 0x8000_0000) as u64,
            ]);
        }
    }
    for s in [8u32, 16, 24, 31, 32, 48, 63] {
        if (s as u8) < w {
            v.push(1u64 << s);
            v.push((1u64 << s) - 1);
        }
    }
    let mut out: Vec<u64> = Vec::new();
    for x in v {
        let x = x & m;
        if !out.contains(&x) {
            out.push(x);
        }
    }
    out.sort_by_key(|&x| !cx.admits(leaf, x));
    out.truncate(160);
    out
}

type Fallback = Option<(HashMap<E, u64>, bool)>;

/// Check `env`; a full success (goal, prefix kept, admitted) is returned, a partial one kept.
fn consider(cx: &mut Ctx, g: &Goal, env: HashMap<E, u64>, fb: &mut Fallback) -> Option<Ans> {
    let (ok, keep) = cx.check(g, &env);
    if !ok {
        return None;
    }
    let adm = cx.env_admitted(&env);
    if keep && adm {
        return Some(Ans::Sat(env, true));
    }
    let better = match fb {
        None => true,
        Some((e0, k0)) => {
            let a0 = cx.env_admitted(e0);
            (adm && !a0) || (keep && !*k0 && adm == a0)
        }
    };
    if better {
        *fb = Some((env, keep));
    }
    None
}

fn search(cx: &mut Ctx, g: &Goal) -> Ans {
    let vars = g.vars.clone();
    if vars.is_empty() {
        return Ans::Unknown("no input leaf".into());
    }
    let consts = cone_consts(cx, g.root);
    let budget = cx.evals + cx.opts.evals;
    let mut fb: Fallback = None;
    let mut complete = vars.len() <= 2;
    // One leaf at a time: exhaustive when it is the only one and narrow.
    for &x in &vars {
        let w = cx.sym.ar.width[x as usize];
        if w <= 16 && vars.len() == 1 {
            for v in 0..=mask(w) {
                if let Some(a) = consider(cx, g, HashMap::from([(x, v)]), &mut fb) {
                    return a;
                }
            }
        } else {
            if vars.len() == 1 {
                complete = false;
            }
            for v in leaf_cands(cx, x, &consts) {
                if cx.evals > budget {
                    complete = false;
                    break;
                }
                if let Some(a) = consider(cx, g, HashMap::from([(x, v)]), &mut fb) {
                    return a;
                }
            }
        }
    }
    // Two leaves.
    if vars.len() >= 2 {
        let pairs: Vec<(E, E)> = vars
            .iter()
            .enumerate()
            .flat_map(|(i, &a)| vars[i + 1..].iter().map(move |&b| (a, b)))
            .take(6)
            .collect();
        for (a, b) in pairs {
            let (wa, wb) = (cx.sym.ar.width[a as usize], cx.sym.ar.width[b as usize]);
            let (ca, cb): (Vec<u64>, Vec<u64>) = if wa <= 8 && wb <= 8 && vars.len() == 2 {
                ((0..=mask(wa)).collect(), (0..=mask(wb)).collect())
            } else {
                complete = false;
                let mut x = leaf_cands(cx, a, &consts);
                let mut y = leaf_cands(cx, b, &consts);
                x.truncate(96);
                y.truncate(96);
                (x, y)
            };
            for &va in &ca {
                for &vb in &cb {
                    if cx.evals > budget {
                        complete = false;
                        break;
                    }
                    if let Some(r) = consider(cx, g, HashMap::from([(a, va), (b, vb)]), &mut fb) {
                        return r;
                    }
                }
            }
        }
    }
    if let Some((env, keep)) = fb {
        return Ans::Sat(env, keep);
    }
    if complete {
        return Ans::Unsat;
    }
    Ans::Unknown("bounded search found no assignment".into())
}

// ---------------------------------------------------------------- T3: local search

/// Order-preserving map of a `w`-bit value read as a float (32 bits) or signed integer.
fn to_ord(v: u64, w: u8, float: bool) -> i128 {
    if float && w == 32 {
        let u = v as u32;
        if u >> 31 == 1 {
            -((u & 0x7fff_ffff) as i128)
        } else {
            u as i128
        }
    } else {
        let s = ((v & mask(w)) << (64 - w as u32)) as i64 >> (64 - w as u32);
        s as i128
    }
}

fn from_ord(o: i128, w: u8, float: bool) -> u64 {
    if float && w == 32 {
        let o = o.clamp(-(0x7fff_ffff_i128), 0x7fff_ffff);
        if o < 0 {
            (((-o) as u32) | 0x8000_0000) as u64
        } else {
            o as u32 as u64
        }
    } else {
        (o as i64 as u64) & mask(w)
    }
}

/// The deciding atom of the goal: the comparison whose operands' distance the local search
/// moves.
fn deciding(cx: &mut Ctx, g: &Goal) -> Option<(E, bool)> {
    let mut choices = Vec::new();
    bool_choices(cx, g.root, g.want, &mut choices, 0);
    choices.into_iter().find(|c| c.len() == 1).map(|c| c[0])
}

/// Signed distance of a comparison's operands (float or integer), None when undefined.
fn distance(cx: &mut Ctx, atom: E, env: &HashMap<E, u64>) -> Option<f64> {
    match cx.sym.ar.nodes[atom as usize].clone() {
        Node::FCmp { a, b, .. } => {
            let x = f32::from_bits(cx.eval(a, env)? as u32) as f64;
            let y = f32::from_bits(cx.eval(b, env)? as u32) as f64;
            let d = x - y;
            (!d.is_nan()).then_some(d)
        }
        Node::DCmp { a, b, .. } => {
            let x = f64::from_bits(cx.eval(a, env)?);
            let y = f64::from_bits(cx.eval(b, env)?);
            let d = x - y;
            (!d.is_nan()).then_some(d)
        }
        Node::Cmp { cmp, a, b } => {
            let w = cx.sym.ar.width[a as usize];
            let signed = matches!(
                cmp,
                ptxv_ir::Cmp::LtS | ptxv_ir::Cmp::LeS | ptxv_ir::Cmp::GtS | ptxv_ir::Cmp::GeS
            );
            let (x, y) = (cx.eval(a, env)?, cx.eval(b, env)?);
            let f = |v: u64| {
                if signed {
                    to_ord(v, w, false) as f64
                } else {
                    (v & mask(w)) as f64
                }
            };
            Some(f(x) - f(y))
        }
        _ => None,
    }
}

fn is_float_leaf(cx: &mut Ctx, leaf: E, root: E) -> bool {
    if cx.sym.ar.width[leaf as usize] != 32 {
        return false;
    }
    let c = cx.cone(root);
    c.iter().any(|&x| match &cx.sym.ar.nodes[x as usize] {
        Node::Fma { a, b, c, .. } => *a == leaf || *b == leaf || *c == Some(leaf),
        Node::FCmp { a, b, .. } | Node::FMinMax { a, b, .. } => *a == leaf || *b == leaf,
        Node::Mufu { a, .. } | Node::Narrow { a, .. } => *a == leaf,
        Node::Pack { a, b, c, .. } => *a == leaf || *b == leaf || *c == leaf,
        Node::Un { a, op, .. } => *a == leaf && matches!(op, Un::FNeg | Un::FAbs | Un::FSat),
        _ => false,
    })
}

fn t3(cx: &mut Ctx, g: &Goal) -> Ans {
    let vars: Vec<E> = g.vars.iter().copied().take(8).collect();
    if vars.is_empty() {
        return Ans::Unknown("no input leaf".into());
    }
    let Some((atom, _)) = deciding(cx, g) else {
        return Ans::Unknown("no single deciding comparison".into());
    };
    if !matches!(
        cx.sym.ar.nodes[atom as usize],
        Node::FCmp { .. } | Node::Cmp { .. } | Node::DCmp { .. }
    ) {
        return Ans::Unknown("the deciding atom is not a comparison".into());
    }
    let budget = cx.evals + cx.opts.evals;
    let floats: Vec<bool> = vars.iter().map(|&v| is_float_leaf(cx, v, g.root)).collect();
    let mut env: HashMap<E, u64> = vars
        .iter()
        .map(|&v| (v, cx.sym.ar.conc[v as usize]))
        .collect();
    let mut fb: Fallback = None;
    let consts = cone_consts(cx, g.root);
    for _round in 0..4 {
        for (i, &x) in vars.iter().enumerate() {
            if cx.evals > budget {
                break;
            }
            let w = cx.sym.ar.width[x as usize];
            let fl = floats[i];
            let cur = env[&x];
            let co = to_ord(cur, w, fl);
            let mut pts: Vec<u64> = Vec::new();
            if fl {
                let f = f32::from_bits(cur as u32);
                for s in [
                    2.0f32,
                    0.5,
                    -1.0,
                    1.5,
                    0.75,
                    4.0,
                    0.25,
                    16.0,
                    1.0 / 16.0,
                    1e3,
                    1e-3,
                    1e6,
                    1e-6,
                ] {
                    pts.push((f * s).to_bits() as u64);
                }
                for d in [
                    1.0f32, -1.0, 10.0, -10.0, 100.0, -100.0, 1e4, -1e4, 1e8, -1e8,
                ] {
                    pts.push((f + d).to_bits() as u64);
                }
                pts.extend(f32_specials());
                for &k in &consts {
                    let kf = f32::from_bits(k as u32);
                    if kf.is_finite() {
                        for d in [0.0f32, 1.0, -1.0, 0.5, -0.5] {
                            pts.push((kf + d).to_bits() as u64);
                            pts.push((-kf + d).to_bits() as u64);
                        }
                    }
                }
            } else {
                for k in 0..(w as u32).min(63) {
                    let d = 1i128 << k;
                    pts.push(from_ord(co + d, w, false));
                    pts.push(from_ord(co - d, w, false));
                }
                pts.extend(consts.iter().copied());
            }
            pts.sort_by_key(|&p| !cx.admits(x, p));
            let mut best: Option<(f64, u64)> = None;
            let d0 = distance(cx, atom, &env);
            let mut samples: Vec<(i128, Option<f64>)> = vec![(co, d0)];
            for p in pts {
                if cx.evals > budget {
                    break;
                }
                let mut e2 = env.clone();
                e2.insert(x, p);
                if let Some(a) = consider(cx, g, e2.clone(), &mut fb) {
                    return a;
                }
                let d = distance(cx, atom, &e2);
                if let Some(dv) = d {
                    if best.is_none_or(|b| dv.abs() < b.0) {
                        best = Some((dv.abs(), p));
                    }
                }
                samples.push((to_ord(p, w, fl), d));
            }
            // Bisect across sign changes of the distance (ordered by the coordinate).
            samples.sort_by_key(|s| s.0);
            let mut crossings = 0;
            for j in 1..samples.len() {
                if crossings > 6 || cx.evals > budget {
                    break;
                }
                let (lo, hi) = (samples[j - 1], samples[j]);
                let (Some(dl), Some(dh)) = (lo.1, hi.1) else {
                    continue;
                };
                if (dl < 0.0) == (dh < 0.0) && dl != 0.0 && dh != 0.0 {
                    continue;
                }
                crossings += 1;
                let (mut a, mut b) = (lo.0, hi.0);
                let sa = dl < 0.0;
                for _ in 0..80 {
                    if b - a <= 1 {
                        break;
                    }
                    let mid = a + (b - a) / 2;
                    let mut e2 = env.clone();
                    e2.insert(x, from_ord(mid, w, fl));
                    if let Some(r) = consider(cx, g, e2.clone(), &mut fb) {
                        return r;
                    }
                    match distance(cx, atom, &e2) {
                        Some(dm) if (dm < 0.0) == sa && dm != 0.0 => a = mid,
                        Some(_) => b = mid,
                        None => break,
                    }
                }
                for o in [a - 1, a, a + 1, b - 1, b, b + 1] {
                    let mut e2 = env.clone();
                    e2.insert(x, from_ord(o, w, fl));
                    if let Some(r) = consider(cx, g, e2, &mut fb) {
                        return r;
                    }
                }
            }
            if let Some((_, p)) = best {
                env.insert(x, p);
            }
        }
    }
    if let Some((env, keep)) = fb {
        return Ans::Sat(env, keep);
    }
    Ans::Unknown("local search found no crossing".into())
}

// ---------------------------------------------------------------- SMT

#[cfg(feature = "smt")]
mod smt {
    use std::collections::HashMap;
    use std::fmt::Write;

    use ptxv_ir::Cmp;
    use ptxv_ir::FCmp;
    use ptxv_ir::Rnd;

    use super::Ans;
    use super::Ctx;
    use super::Goal;
    use crate::symb::Bin;
    use crate::symb::E;
    use crate::symb::Node;
    use crate::symb::Un;

    fn bvc(w: u8, v: u64) -> String {
        format!("(_ bv{v} {w})")
    }

    fn rm(r: Rnd) -> &'static str {
        match r {
            Rnd::Rn => "RNE",
            Rnd::Rz => "RTZ",
            Rnd::Rm => "RTN",
            Rnd::Rp => "RTP",
        }
    }

    fn fp(x: String) -> String {
        format!("((_ to_fp 8 24) {x})")
    }

    /// FTZ: a subnormal reads as the zero of its sign.
    fn ftz(x: String) -> String {
        format!(
            "(ite (fp.isSubnormal {x}) (ite (fp.isNegative {x}) (_ -zero 8 24) (_ +zero 8 24)) {x})"
        )
    }

    /// An f32 result as its encoding (NaN: the canonical 0x7fffffff).
    fn to_bv(x: String) -> String {
        format!("(ite (fp.isNaN {x}) #x7fffffff ((_ fp.to_ieee_bv 8 24) {x}))")
    }

    fn fcmp_text(cmp: FCmp, x: &str, y: &str) -> String {
        let un = format!("(or (fp.isNaN {x}) (fp.isNaN {y}))");
        match cmp {
            FCmp::F => "false".to_string(),
            FCmp::T => "true".to_string(),
            FCmp::Lt => format!("(fp.lt {x} {y})"),
            FCmp::Eq => format!("(fp.eq {x} {y})"),
            FCmp::Le => format!("(fp.leq {x} {y})"),
            FCmp::Gt => format!("(fp.gt {x} {y})"),
            FCmp::Ne => format!("(and (not {un}) (not (fp.eq {x} {y})))"),
            FCmp::Ge => format!("(fp.geq {x} {y})"),
            FCmp::Num => format!("(not {un})"),
            FCmp::Nan => un,
            FCmp::Ltu => format!("(or {un} (fp.lt {x} {y}))"),
            FCmp::Equ => format!("(or {un} (fp.eq {x} {y}))"),
            FCmp::Leu => format!("(or {un} (fp.leq {x} {y}))"),
            FCmp::Gtu => format!("(or {un} (fp.gt {x} {y}))"),
            FCmp::Neu => format!("(or {un} (not (fp.eq {x} {y})))"),
            FCmp::Geu => format!("(or {un} (fp.geq {x} {y}))"),
        }
    }

    /// SMT-LIB2 for the cone of `roots`: one `define-fun` per node.
    fn emit(cx: &mut Ctx, roots: &[E], decls: &mut String, defs: &mut String) -> Option<()> {
        let mut cone: Vec<E> = roots.iter().flat_map(|&r| cx.cone(r)).collect();
        cone.sort_unstable();
        cone.dedup();
        let ar = &cx.sym.ar;
        let n = |x: E| format!("n{x}");
        let mut ufs: HashMap<String, usize> = HashMap::new();
        for &x in &cone {
            let w = ar.width[x as usize];
            let ty = format!("(_ BitVec {w})");
            let body: String = match &ar.nodes[x as usize] {
                Node::Const { v, .. } => bvc(w, *v),
                Node::Input { .. } => {
                    writeln!(decls, "(declare-fun {} () {ty})", n(x)).ok()?;
                    continue;
                }
                Node::Opaque { .. } | Node::Sreg { .. } => bvc(w, ar.conc[x as usize]),
                Node::Un { op, a, .. } => match op {
                    Un::Not => format!("(bvnot {})", n(*a)),
                    Un::Neg => format!("(bvneg {})", n(*a)),
                    Un::FNeg => format!("(bvxor {} #x80000000)", n(*a)),
                    Un::FAbs => format!("(bvand {} #x7fffffff)", n(*a)),
                    Un::FSat => {
                        let f = fp(n(*a));
                        format!(
                            "(ite (fp.isNaN {f}) #x00000000 (ite (fp.leq {f} (_ +zero 8 24)) #x00000000 (ite (fp.geq {f} ((_ to_fp 8 24) RNE 1.0)) #x3f800000 {})))",
                            n(*a)
                        )
                    }
                    Un::H2F => format!(
                        "(let ((h ((_ to_fp 5 11) {}))) (ite (fp.isNaN h) #x7fffffff ((_ fp.to_ieee_bv 8 24) ((_ to_fp 8 24) RNE h))))",
                        n(*a)
                    ),
                    Un::B2F => format!("(concat {} #x0000)", n(*a)),
                    Un::DNegK => format!(
                        "(ite (bvugt (bvand {0} #x7fffffffffffffff) #x7ff0000000000000) {0} (bvxor {0} #x8000000000000000))",
                        n(*a)
                    ),
                    Un::DAbsK => format!(
                        "(ite (bvugt (bvand {0} #x7fffffffffffffff) #x7ff0000000000000) {0} (bvand {0} #x7fffffffffffffff))",
                        n(*a)
                    ),
                    Un::Popc | Un::Brev => return None,
                },
                Node::Bin { op, a, b, .. } => {
                    let (a, b) = (n(*a), n(*b));
                    let wv = bvc(w, w as u64);
                    match op {
                        Bin::Add => format!("(bvadd {a} {b})"),
                        Bin::Sub => format!("(bvsub {a} {b})"),
                        Bin::Mul => format!("(bvmul {a} {b})"),
                        Bin::And => format!("(bvand {a} {b})"),
                        Bin::Or => format!("(bvor {a} {b})"),
                        Bin::Xor => format!("(bvxor {a} {b})"),
                        Bin::Shl => format!("(ite (bvuge {b} {wv}) {} (bvshl {a} {b}))", bvc(w, 0)),
                        Bin::LShr => {
                            format!("(ite (bvuge {b} {wv}) {} (bvlshr {a} {b}))", bvc(w, 0))
                        }
                        Bin::AShr => format!(
                            "(bvashr {a} (ite (bvuge {b} {wv}) {} {b}))",
                            bvc(w, w as u64 - 1)
                        ),
                        Bin::UDiv => format!("(bvudiv {a} {b})"),
                        Bin::URem => format!("(bvurem {a} {b})"),
                        Bin::SDiv => format!("(bvsdiv {a} {b})"),
                        Bin::SRem => format!("(bvsrem {a} {b})"),
                        Bin::UMin => format!("(ite (bvult {a} {b}) {a} {b})"),
                        Bin::UMax => format!("(ite (bvugt {a} {b}) {a} {b})"),
                        Bin::SMin => format!("(ite (bvslt {a} {b}) {a} {b})"),
                        Bin::SMax => format!("(ite (bvsgt {a} {b}) {a} {b})"),
                        Bin::MulHiU => format!(
                            "((_ extract {} {w}) (bvmul ((_ zero_extend {w}) {a}) ((_ zero_extend {w}) {b})))",
                            2 * w as u32 - 1
                        ),
                        Bin::MulHiS => format!(
                            "((_ extract {} {w}) (bvmul ((_ sign_extend {w}) {a}) ((_ sign_extend {w}) {b})))",
                            2 * w as u32 - 1
                        ),
                    }
                }
                Node::Lop3 { lut, a, b, c, .. } => {
                    let mut terms = Vec::new();
                    for i in 0..8u8 {
                        if (lut >> i) & 1 == 1 {
                            let f = |bit: u8, x: &E| {
                                if i & bit != 0 {
                                    n(*x)
                                } else {
                                    format!("(bvnot {})", n(*x))
                                }
                            };
                            terms.push(format!(
                                "(bvand {} (bvand {} {}))",
                                f(4, a),
                                f(2, b),
                                f(1, c)
                            ));
                        }
                    }
                    terms
                        .into_iter()
                        .reduce(|x, y| format!("(bvor {x} {y})"))
                        .unwrap_or_else(|| bvc(w, 0))
                }
                Node::Cmp { cmp, a, b } => {
                    let o = match cmp {
                        Cmp::Eq => "=",
                        Cmp::Ne => "distinct",
                        Cmp::LtU => "bvult",
                        Cmp::LeU => "bvule",
                        Cmp::GtU => "bvugt",
                        Cmp::GeU => "bvuge",
                        Cmp::LtS => "bvslt",
                        Cmp::LeS => "bvsle",
                        Cmp::GtS => "bvsgt",
                        Cmp::GeS => "bvsge",
                    };
                    format!("(ite ({o} {} {}) #b1 #b0)", n(*a), n(*b))
                }
                Node::Ite { c, a, b } => format!("(ite (= {} #b1) {} {})", n(*c), n(*a), n(*b)),
                Node::Zext { a, .. } => {
                    format!("((_ zero_extend {}) {})", w - ar.width[*a as usize], n(*a))
                }
                Node::Sext { a, .. } => {
                    format!("((_ sign_extend {}) {})", w - ar.width[*a as usize], n(*a))
                }
                Node::Extract { lo, a, .. } => {
                    format!("((_ extract {} {lo}) {})", lo + w - 1, n(*a))
                }
                Node::Concat { hi, lo } => format!("(concat {} {})", n(*hi), n(*lo)),
                Node::Fma {
                    a,
                    b,
                    c,
                    rnd,
                    ftz: fz,
                } => {
                    let r = rm(*rnd);
                    let f = |x: E| if *fz { ftz(fp(n(x))) } else { fp(n(x)) };
                    let v = match c {
                        Some(c) => format!("(fp.fma {r} {} {} {})", f(*a), f(*b), f(*c)),
                        None => format!("(fp.mul {r} {} {})", f(*a), f(*b)),
                    };
                    to_bv(if *fz { ftz(v) } else { v })
                }
                Node::FMinMax {
                    min,
                    nan,
                    ftz: fz,
                    a,
                    b,
                } => {
                    let f = |x: E| if *fz { ftz(fp(n(x))) } else { fp(n(x)) };
                    let (x, y) = (f(*a), f(*b));
                    let lt = format!(
                        "(or (fp.lt {x} {y}) (and (fp.isZero {x}) (fp.isZero {y}) (fp.isNegative {x}) (fp.isPositive {y})))"
                    );
                    let (bx, by) = (to_bv(x.clone()), to_bv(y.clone()));
                    let pick = if *min {
                        format!("(ite {lt} {bx} {by})")
                    } else {
                        format!("(ite {lt} {by} {bx})")
                    };
                    if *nan {
                        format!("(ite (or (fp.isNaN {x}) (fp.isNaN {y})) #x7fffffff {pick})")
                    } else {
                        format!(
                            "(ite (and (fp.isNaN {x}) (fp.isNaN {y})) #x7fffffff (ite (fp.isNaN {x}) {by} (ite (fp.isNaN {y}) {bx} {pick})))"
                        )
                    }
                }
                Node::FCmp { cmp, ftz: fz, a, b } => {
                    let f = |x: E| if *fz { ftz(fp(n(x))) } else { fp(n(x)) };
                    let (x, y) = (f(*a), f(*b));
                    format!("(ite {} #b1 #b0)", fcmp_text(*cmp, &x, &y))
                }
                Node::Mufu {
                    func,
                    fmt,
                    hi,
                    ptx,
                    a,
                } => {
                    // Uninterpreted, with the unit's sign and special-value behaviour.
                    let name = format!(
                        "mufu_{func:?}_{fmt:?}_{hi}_{}",
                        ptx.map_or("s".to_string(), |f| f.to_string())
                    );
                    if ufs.insert(name.clone(), 1).is_none() {
                        writeln!(decls, "(declare-fun {name} ((_ BitVec 32)) (_ BitVec 32))")
                            .ok()?;
                    }
                    let v = format!("({name} {})", n(*a));
                    let xf = fp(n(*a));
                    let r = fp(v.clone());
                    let ax = match func {
                        ptxv_ir::MufuFunc::Ex2 => Some(format!(
                            "(and (or (fp.isNaN {xf}) (fp.isNaN {r}) (not (fp.isNegative {r}))) (=> (and (not (fp.isNaN {xf})) (fp.geq {xf} ((_ to_fp 8 24) RNE 128.0))) (fp.isInfinite {r})) (=> (fp.isZero {xf}) (= {v} #x3f800000)))"
                        )),
                        ptxv_ir::MufuFunc::Rcp => Some(format!(
                            "(and (=> (fp.isZero {xf}) (fp.isInfinite {r})) (=> (and (not (fp.isNaN {xf})) (not (fp.isZero {xf}))) (= (fp.isNegative {xf}) (fp.isNegative {r}))))"
                        )),
                        ptxv_ir::MufuFunc::Rsq
                        | ptxv_ir::MufuFunc::Lg2
                        | ptxv_ir::MufuFunc::Sqrt => Some(format!(
                            "(=> (and (fp.isNegative {xf}) (not (fp.isZero {xf})) (not (fp.isNaN {xf}))) (fp.isNaN {r}))"
                        )),
                        _ => None,
                    };
                    if let Some(ax) = ax {
                        writeln!(defs, "(assert {ax})").ok()?;
                    }
                    v
                }
                Node::DFma { a, b, c, rnd } => {
                    let f = |x: E| format!("((_ to_fp 11 53) {})", n(x));
                    let v = match c {
                        Some(c) => format!("(fp.fma {} {} {} {})", rm(*rnd), f(*a), f(*b), f(*c)),
                        None => format!("(fp.mul {} {} {})", rm(*rnd), f(*a), f(*b)),
                    };
                    format!(
                        "(let ((r {v})) (ite (fp.isNaN r) #xfff8000000000000 ((_ fp.to_ieee_bv 11 53) r)))"
                    )
                }
                Node::DCmp { cmp, a, b } => {
                    let (x, y) = (
                        format!("((_ to_fp 11 53) {})", n(*a)),
                        format!("((_ to_fp 11 53) {})", n(*b)),
                    );
                    format!("(ite {} #b1 #b0)", fcmp_text(*cmp, &x, &y))
                }
                Node::HFma { bf16, a, b, c } => {
                    let (e, m) = if *bf16 { (8, 8) } else { (5, 11) };
                    let f = |x: E| format!("((_ to_fp {e} {m}) {})", n(x));
                    format!(
                        "(let ((r (fp.fma RNE {} {} {}))) (ite (fp.isNaN r) #x7fff ((_ fp.to_ieee_bv {e} {m}) r)))",
                        f(*a),
                        f(*b),
                        f(*c)
                    )
                }
                Node::Narrow { bf16, a } => {
                    let (e, m) = if *bf16 { (8, 8) } else { (5, 11) };
                    format!(
                        "(let ((r ((_ to_fp {e} {m}) RNE ((_ to_fp 8 24) {})))) (ite (fp.isNaN r) #x7fff ((_ fp.to_ieee_bv {e} {m}) r)))",
                        n(*a)
                    )
                }
                Node::FCvt {
                    from,
                    to,
                    rnd,
                    ftz: false,
                    int: false,
                    a,
                } => {
                    let fm = |f: ptxv_ir::FFmt| match f {
                        ptxv_ir::FFmt::F16 => (5, 11),
                        ptxv_ir::FFmt::Bf16 => (8, 8),
                        ptxv_ir::FFmt::F32 => (8, 24),
                        ptxv_ir::FFmt::F64 => (11, 53),
                    };
                    let ((e1, m1), (e2, m2)) = (fm(*from), fm(*to));
                    let nan = match to {
                        ptxv_ir::FFmt::F64 => "#xfff8000000000000".to_string(),
                        ptxv_ir::FFmt::F32 => "#x7fffffff".to_string(),
                        _ => "#x7fff".to_string(),
                    };
                    format!(
                        "(let ((r ((_ to_fp {e2} {m2}) {} ((_ to_fp {e1} {m1}) {})))) (ite (fp.isNaN r) {nan} ((_ fp.to_ieee_bv {e2} {m2}) r)))",
                        rm(*rnd),
                        n(*a)
                    )
                }
                Node::I2F {
                    signed,
                    rnd,
                    f64,
                    a,
                } => {
                    let (e, m) = if *f64 { (11, 53) } else { (8, 24) };
                    let conv = if *signed { "to_fp" } else { "to_fp_unsigned" };
                    format!(
                        "((_ fp.to_ieee_bv {e} {m}) ((_ {conv} {e} {m}) {} {}))",
                        rm(*rnd),
                        n(*a)
                    )
                }
                Node::FCvt { a, .. } | Node::F2F { a, .. } | Node::F2I { a, .. } => {
                    // Saturation, NaN and payload rules: uninterpreted.
                    let wa = ar.width[*a as usize];
                    let name = format!("cvt_{x}");
                    writeln!(decls, "(declare-fun {name} ((_ BitVec {wa})) {ty})").ok()?;
                    format!("({name} {})", n(*a))
                }
                Node::Pack { to, a, b, c, .. } => {
                    let name = format!("pack_{to:?}_{x}");
                    writeln!(decls, "(declare-fun {name} ((_ BitVec 32) (_ BitVec 32) (_ BitVec 32)) (_ BitVec 32))").ok()?;
                    format!("({name} {} {} {})", n(*a), n(*b), n(*c))
                }
                // Another address's bytes: unconstrained.
                Node::AtAddr { addr, at, val } => {
                    let wa = ar.width[*addr as usize];
                    writeln!(decls, "(declare-fun u{x} () {ty})").ok()?;
                    format!("(ite (= {} {}) {} u{x})", n(*addr), bvc(wa, *at), n(*val))
                }
                Node::Inst { pc, out, args, .. } => {
                    let name = format!("inst_{pc}_{out}");
                    let tys: Vec<String> = args
                        .iter()
                        .map(|&k| format!("(_ BitVec {})", ar.width[k as usize]))
                        .collect();
                    if ufs.insert(name.clone(), 1).is_none() {
                        writeln!(decls, "(declare-fun {name} ({}) {ty})", tys.join(" ")).ok()?;
                    }
                    if args.is_empty() {
                        name
                    } else {
                        format!(
                            "({name} {})",
                            args.iter().map(|&k| n(k)).collect::<Vec<_>>().join(" ")
                        )
                    }
                }
            };
            writeln!(defs, "(define-fun {} () {ty} {body})", n(x)).ok()?;
        }
        Some(())
    }

    pub(super) fn solve(cx: &mut Ctx, g: &Goal, with_prefix: bool) -> Ans {
        let mut roots = vec![g.root];
        if with_prefix {
            roots.extend(g.prefix.iter().map(|p| p.0));
        }
        let (mut decls, mut defs) = (String::new(), String::new());
        if emit(cx, &roots, &mut decls, &mut defs).is_none() {
            return Ans::Unknown("smt: node not expressible".into());
        }
        let mut q = String::from("(set-logic ALL)\n(set-option :produce-models true)\n");
        q.push_str(&decls);
        q.push_str(&defs);
        let bit = |v: bool| if v { "#b1" } else { "#b0" };
        writeln!(q, "(assert (= n{} {}))", g.root, bit(g.want)).ok();
        if with_prefix {
            for &(c, v) in &g.prefix {
                writeln!(q, "(assert (= n{c} {}))", bit(v)).ok();
            }
        }
        q.push_str("(check-sat)\n");
        if !g.vars.is_empty() {
            let vs: Vec<String> = g.vars.iter().map(|v| format!("n{v}")).collect();
            writeln!(q, "(get-value ({}))", vs.join(" ")).ok();
        }
        if let Ok(d) = std::env::var("PTXV_SMT_DUMP") {
            let _ = std::fs::write(format!("{d}/q{}.smt2", g.root), &q);
        }
        let solver = std::env::var("PTXV_SMT_SOLVER").unwrap_or_else(|_| "z3".into());
        let t = format!("-T:{}", cx.opts.smt_timeout.ceil().max(1.0) as u64);
        let child = std::process::Command::new(&solver)
            .args(["-in", "-smt2", &t])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn();
        let Ok(mut child) = child else {
            return Ans::Unknown(format!("smt: cannot run {solver}"));
        };
        {
            use std::io::Write as _;
            let Some(mut si) = child.stdin.take() else {
                return Ans::Unknown("smt: no stdin".into());
            };
            if si.write_all(q.as_bytes()).is_err() {
                return Ans::Unknown("smt: write failed".into());
            }
        }
        let Ok(out) = child.wait_with_output() else {
            return Ans::Unknown("smt: no output".into());
        };
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        let first = text.lines().next().unwrap_or("").trim().to_string();
        match first.as_str() {
            "unsat" => Ans::Unsat,
            "sat" => {
                let mut env = HashMap::new();
                for v in &g.vars {
                    let key = format!("(n{v} ");
                    if let Some(i) = text.find(&key) {
                        let rest = &text[i + key.len()..];
                        let tok: String = rest
                            .chars()
                            .take_while(|c| !c.is_whitespace() && *c != ')')
                            .collect();
                        let val = if let Some(h) = tok.strip_prefix("#x") {
                            u64::from_str_radix(h, 16).ok()
                        } else if let Some(b) = tok.strip_prefix("#b") {
                            u64::from_str_radix(b, 2).ok()
                        } else {
                            None
                        };
                        if let Some(x) = val {
                            env.insert(*v, x);
                        }
                    }
                }
                let (ok, keep) = cx.check(g, &env);
                if ok {
                    Ans::Sat(env, keep)
                } else {
                    // The model's MUFU / Inst functions are not the hardware's.
                    Ans::Unknown("smt: model does not satisfy the goal on the expressions".into())
                }
            }
            _ => Ans::Unknown(format!(
                "smt: {}",
                if first.is_empty() {
                    "no answer"
                } else {
                    &first
                }
            )),
        }
    }
}

// ---------------------------------------------------------------- driver

/// Writes of an assignment: (allocation, byte offset, bytes, value) per changed input leaf.
fn writes_of(sym: &SymState, env: &HashMap<E, u64>, fault: Fault) -> Vec<(usize, u64, u8, u64)> {
    let flip = if fault == Fault::Writes { 1 } else { 0 };
    let mut w: Vec<(usize, u64, u8, u64)> = env
        .iter()
        .filter_map(|(&e, &v)| {
            let (a, off, b) = leaf_of(sym, e)?;
            (sym.ar.conc[e as usize] != v).then_some((a as usize, off, b, v ^ flip))
        })
        .collect();
    w.sort_unstable();
    w
}

fn label_of(m: &Machine, writes: &[(usize, u64, u8, u64)]) -> String {
    writes
        .iter()
        .map(|&(a, off, b, v)| {
            let name = m
                .launch
                .allocs
                .get(a)
                .map_or("?".to_string(), |x| x.name.clone());
            let vs = if v < 1 << 16 {
                format!("{v}")
            } else {
                format!("{v:#x}")
            };
            if b > 0 && off.is_multiple_of(b as u64) {
                format!("{name}[{}]={vs}", off / b as u64)
            } else {
                format!("{name}[byte {off:#x}:{b}]={vs}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Did the coverage run of `m0` with `writes` take side `want` at `pc`?
pub fn replay(m0: &Machine, pc: u32, want: bool, writes: &[(usize, u64, u8, u64)]) -> bool {
    let mu = Mutation {
        pc,
        never: want,
        writes: writes.to_vec(),
        label: String::new(),
        within_invariants: false,
    };
    let mm = crate::coverage::apply(m0, &mu);
    // A witness can make the kernel run far longer than the captured input (a size-like
    // input): the run stops at its time budget (`coverage::Budget`); a side it observed before is still observed.
    let secs = std::env::var("PTXV_CONCOLIC_REPLAY_SECS")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(30.0f64);
    let b = crate::coverage::Budget {
        seconds: Some(secs),
        rss: None,
    };
    let c = crate::coverage::analyze_budget(&mm, b);
    c.comparisons
        .iter()
        .find(|x| x.pc == pc)
        .is_some_and(|x| if want { x.n_true > 0 } else { x.n_false > 0 })
}

/// Move each witness value toward its captured value (bisection) while the goal (and the
/// prefix, when kept) still holds on the expressions: a smaller change to the input (a
/// solver's arbitrary model can make a size-like input huge, and the replay run long).
fn shrink(cx: &mut Ctx, g: &Goal, mut env: HashMap<E, u64>, keep: bool) -> HashMap<E, u64> {
    if cx.opts.fault != Fault::None {
        return env;
    }
    let holds = |cx: &mut Ctx, env: &HashMap<E, u64>| {
        let (ok, k) = cx.check(g, env);
        ok && (k || !keep)
    };
    let mut keys: Vec<E> = env.keys().copied().collect();
    keys.sort_unstable();
    let mut evals = 0;
    for x in keys {
        let c = cx.sym.ar.conc[x as usize];
        let w = env[&x];
        if w == c {
            env.remove(&x);
            continue;
        }
        // Unchanged altogether?
        env.insert(x, c);
        evals += 1;
        if holds(cx, &env) {
            env.remove(&x);
            continue;
        }
        // Bisect between the captured value (fails) and the witness (holds).
        let (mut bad, mut good) = (c, w);
        for _ in 0..24 {
            if evals >= 256 {
                break;
            }
            let d = good.abs_diff(bad);
            if d <= 1 {
                break;
            }
            let mid = if good > bad { bad + d / 2 } else { bad - d / 2 };
            env.insert(x, mid);
            evals += 1;
            if holds(cx, &env) {
                good = mid;
            } else {
                bad = mid;
            }
        }
        env.insert(x, good);
    }
    env
}

fn dedup_join(v: &[String]) -> String {
    let mut seen: Vec<&String> = Vec::new();
    for x in v {
        if !seen.contains(&x) {
            seen.push(x);
        }
    }
    seen.iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join("; ")
}

/// The concolic search over `cov`'s untaken sides. `sym` and `mfin` from
/// `coverage::analyze_with`; `m0` the launch's initial machine.
pub fn run(
    m0: &Machine,
    cov: &Coverage,
    sym: &SymState,
    mfin: &Machine,
    dom: &Domain,
    opts: &Options,
) -> Report {
    let start = std::time::Instant::now();
    let mut rep = Report {
        nodes: sym.ar.len(),
        inst_nodes: sym.n_inst,
        opaque: sym.n_opaque.clone(),
        mismatches: sym.n_mismatch,
        path_atoms: sym.path.values().map(|p| p.len()).sum(),
        ..Default::default()
    };
    let mut cx = Ctx {
        sym,
        ev: Evaluator::new(&sym.ar, mfin),
        m0,
        dom,
        opts,
        cones: HashMap::new(),
        leaves: HashMap::new(),
        evals: 0,
    };
    let names = |a: u32| {
        m0.launch
            .allocs
            .get(a as usize)
            .map_or("?".to_string(), |x| x.name.clone())
    };
    let mut solvers: Vec<&str> = vec!["T0", "search", "T3"];
    if opts.smt && cfg!(feature = "smt") {
        solvers.push("smt");
    }
    if !opts.only.is_empty() {
        solvers.retain(|s| opts.only.iter().any(|o| o == s));
    }
    for u in &cov.untaken {
        let want = u.never;
        let mut side = Side {
            pc: u.pc,
            at: u.at.clone(),
            never: want,
            outcome: "unknown".into(),
            solver: None,
            witness: None,
            within_invariants: false,
            condition: String::new(),
            reason: None,
            b_flip: None,
            b_refusal: None,
            writes: vec![],
        };
        let obs: Vec<crate::symb::Obs> = sym
            .sites
            .get(&u.pc)
            .map(|s| s[!want as usize].clone())
            .unwrap_or_default();
        if obs.is_empty() {
            side.reason = Some(
                "no thread's predicate has an expression (operands opaque or not followed)".into(),
            );
            rep.unknown += 1;
            rep.sides.push(side);
            continue;
        }
        if start.elapsed().as_secs_f64() > opts.budget || rep.replays >= opts.max_replays {
            side.reason = Some("search budget exhausted".into());
            rep.unknown += 1;
            rep.sides.push(side);
            continue;
        }
        // Observations in different warps first.
        let mut order: Vec<crate::symb::Obs> = Vec::new();
        for pass in 0..2 {
            for o in &obs {
                if order.len() >= opts.tries.max(opts.k) {
                    break;
                }
                if order.iter().any(|x| x.t == o.t)
                    || (pass == 0 && order.iter().any(|x| x.warp == o.warp))
                {
                    continue;
                }
                order.push(*o);
            }
        }
        // Solutions: (assignment, keeps prefix, solver, warp).
        let mut sols: Vec<(HashMap<E, u64>, bool, String, u32)> = Vec::new();
        let mut unsat_by: Option<String> = None;
        let mut n_unsat = 0;
        let mut reasons: Vec<String> = Vec::new();
        for o in &order {
            let path = &sym.path[&o.t];
            let atom = path[o.atom as usize];
            if side.condition.is_empty() {
                side.condition = format!(
                    "{} == {}",
                    sym.ar.show(atom.cond, &names, 8),
                    if want { "true" } else { "false" }
                );
            }
            let mut vars: Vec<E> = cx.leafset(atom.cond).into_iter().collect();
            vars.sort_unstable();
            if vars.is_empty() {
                let opq: Vec<String> = sym
                    .ar
                    .leaves(atom.cond)
                    .iter()
                    .filter_map(|&l| match sym.ar.nodes[l as usize] {
                        Node::Opaque { id, .. } => Some(sym.ar.opaque[id as usize].why.clone()),
                        _ => None,
                    })
                    .collect();
                reasons.push(format!(
                    "the condition depends only on opaque values ({})",
                    dedup_join(&opq)
                ));
                continue;
            }
            // Self-check: the cone evaluates to its captured value.
            if cx.eval(atom.cond, &HashMap::new()) != Some(sym.ar.conc[atom.cond as usize]) {
                reasons.push("the expression does not evaluate to its captured value".into());
                continue;
            }
            let vs: HashSet<E> = vars.iter().copied().collect();
            let prefix: Vec<(E, bool)> = path[..o.atom as usize]
                .iter()
                .filter(|a| cx.leafset(a.cond).iter().any(|l| vs.contains(l)))
                .map(|a| (a.cond, a.val))
                .collect();
            let g = Goal {
                root: atom.cond,
                want,
                prefix,
                vars,
            };
            let mut solved = false;
            for &sv in &solvers {
                let ans = match sv {
                    "T0" => t0(&mut cx, &g),
                    "search" => search(&mut cx, &g),
                    "T3" => t3(&mut cx, &g),
                    #[cfg(feature = "smt")]
                    "smt" => match smt::solve(&mut cx, &g, true) {
                        Ans::Unsat if !g.prefix.is_empty() => smt::solve(&mut cx, &g, false),
                        a => a,
                    },
                    _ => Ans::Unknown("solver not built".into()),
                };
                rep.solvers.entry(sv.to_string()).or_default().tried += 1;
                match ans {
                    Ans::Sat(env, keep) => {
                        // Never trusted: checked again here, replayed below.
                        let ok = cx.check(&g, &env).0;
                        let env = if ok {
                            shrink(&mut cx, &g, env, keep)
                        } else {
                            env
                        };
                        let cnt = rep.solvers.entry(sv.to_string()).or_default();
                        if ok {
                            cnt.sat += 1;
                            sols.push((env, keep, sv.to_string(), o.warp));
                            solved = true;
                            break;
                        }
                        cnt.wrong += 1;
                    }
                    Ans::Unsat => {
                        rep.solvers.entry(sv.to_string()).or_default().unsat += 1;
                        n_unsat += 1;
                        unsat_by.get_or_insert(sv.to_string());
                        break;
                    }
                    Ans::Unknown(r) => reasons.push(format!("{sv}: {r}")),
                }
            }
            if solved && sols.len() >= opts.k {
                break;
            }
        }
        if sols.is_empty() {
            if n_unsat > 0 && n_unsat == order.len() {
                let by = unsat_by.unwrap_or_default();
                side.outcome = "solver-unsat".into();
                side.solver = Some(by.clone());
                side.reason = Some(format!(
                    "unreachable on the observed paths (solver {by}, unverified)"
                ));
                rep.solver_unsat += 1;
            } else {
                side.reason = Some(if reasons.is_empty() {
                    "no solver found an assignment".into()
                } else {
                    dedup_join(&reasons)
                });
                rep.unknown += 1;
            }
            rep.sides.push(side);
            continue;
        }
        // Replay: the solutions of different warps with disjoint inputs together (K
        // elements), then each alone; admitted, prefix-keeping ones first.
        sols.sort_by_key(|s| (!cx.env_admitted(&s.0), !s.1));
        let mut tries: Vec<(Writes, String)> = Vec::new();
        {
            let mut comb: Vec<(usize, u64, u8, u64)> = Vec::new();
            let mut warps = HashSet::new();
            let mut used: HashSet<E> = HashSet::new();
            let mut first: Option<String> = None;
            for (env, _, sv, w) in &sols {
                if warps.len() >= opts.k {
                    break;
                }
                if env.keys().any(|k| used.contains(k)) || warps.contains(w) {
                    continue;
                }
                warps.insert(*w);
                used.extend(env.keys().copied());
                comb.extend(writes_of(sym, env, opts.fault));
                first.get_or_insert(sv.clone());
            }
            if !comb.is_empty() {
                comb.sort_unstable();
                comb.dedup();
                tries.push((comb, first.unwrap_or_default()));
            }
            for (env, _, sv, _) in sols.iter().take(3) {
                let w = writes_of(sym, env, opts.fault);
                if !w.is_empty() && !tries.iter().any(|t| t.0 == w) {
                    tries.push((w, sv.clone()));
                }
            }
        }
        let mut confirmed = false;
        for (writes, sv) in tries {
            if rep.replays >= opts.max_replays {
                break;
            }
            rep.replays += 1;
            if replay(m0, u.pc, want, &writes) {
                let outside = writes
                    .iter()
                    .any(|&(a, off, b, v)| dom.admits(m0, a, off, b, v) == Some(false));
                side.outcome = if outside {
                    "outside-invariants"
                } else {
                    "flipped"
                }
                .into();
                side.within_invariants = dom.within(m0, &writes);
                side.solver = Some(sv.clone());
                side.witness = Some(label_of(m0, &writes));
                side.writes = writes;
                rep.solvers.entry(sv).or_default().confirmed += 1;
                if outside {
                    rep.outside_invariants += 1;
                } else {
                    rep.flipped += 1;
                }
                confirmed = true;
                break;
            }
            rep.solvers.entry(sv).or_default().rejected += 1;
        }
        if !confirmed {
            side.reason = Some("no solver assignment took the side on replay".into());
            rep.unknown += 1;
        }
        rep.sides.push(side);
    }
    rep.seconds_search = start.elapsed().as_secs_f64();
    rep
}

/// Today's B on the same sides, replayed (the differential): per (pc, never), whether B's
/// mutation took the side; B's refusals by pc.
pub fn replay_b(m0: &Machine, cov: &Coverage, dom: &Domain, k: usize) -> BDiff {
    let (muts, refused) = crate::coverage::plan(m0, cov, usize::MAX, k, dom);
    let mut out = BTreeMap::new();
    for mu in muts {
        let ok = replay(m0, mu.pc, mu.never, &mu.writes);
        out.insert((mu.pc, mu.never), (ok, mu.label.clone()));
    }
    let r = refused.into_iter().map(|r| (r.pc, r.reason)).collect();
    (out, r)
}

/// Fill the differential fields of `rep` from `replay_b`.
pub fn attach_b(rep: &mut Report, b: &BDiff) {
    for s in &mut rep.sides {
        s.b_flip = b.0.get(&(s.pc, s.never)).map(|x| x.0);
        if s.b_flip.is_none() {
            s.b_refusal = b.1.get(&s.pc).cloned();
        }
    }
}

/// The confirmed witnesses as mutations for `ptxv check` (evidence class on-mutated-input).
pub fn mutations(m0: &Machine, rep: &Report, dom: &Domain) -> Vec<Mutation> {
    rep.sides
        .iter()
        .filter(|s| !s.writes.is_empty())
        .map(|s| Mutation {
            pc: s.pc,
            never: s.never,
            writes: s.writes.clone(),
            label: label_of(m0, &s.writes),
            within_invariants: dom.within(m0, &s.writes),
        })
        .collect()
}

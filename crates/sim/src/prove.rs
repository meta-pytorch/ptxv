// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Validity of a condition over every input value: `hypotheses ⇒ claim`, both 1-bit nodes of
//! a symb.rs arena.
//!
//! The query goes to an external SMT solver (cargo feature `smt`; `PTXV_SMT_SOLVER`, default
//! `z3`) as bit-vectors. Unlike the concolic search's queries, opaque and special-register
//! leaves are unconstrained here (a proof covers every value they can take); another address's
//! bytes (`AtAddr` away from its captured address, one free value per dynamic read) and
//! instruction results (pure functions of their arguments) are uninterpreted. Only the node
//! kinds the encoder is validated for against the evaluator (`encoder_matches_evaluator`:
//! integer bit-vector operations, compares, selects, extensions, extracts, concatenations,
//! three-input logic) are interpreted; floating-point and conversion nodes are uninterpreted
//! functions of their children, and so is a division's value at a zero divisor. So `Valid` is a proof over the expressions, never weakened by a
//! captured value. A model of the negation is returned with its input leaves and checked by
//! evaluation when the cone has no uninterpreted part (`Invalid { checked: true }`); otherwise
//! it is a candidate only.
//!
//! Without the feature every query is `Unknown`.

use std::collections::HashMap;

use crate::symb::Arena;
use crate::symb::E;

/// The answer to a validity query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The claim holds for every input value satisfying the hypotheses.
    Valid,
    /// A counterexample: input leaves (node, value) satisfying the hypotheses and not the
    /// claim. `checked`: confirmed by evaluating the hypotheses and the claim on the
    /// expressions (no uninterpreted node in the cone); otherwise the solver's model only.
    Invalid { model: Vec<(E, u64)>, checked: bool },
    /// No answer (no solver built, timeout, a node not expressible).
    Unknown(String),
}

impl Verdict {
    pub fn is_valid(&self) -> bool {
        matches!(self, Verdict::Valid)
    }
}

/// Is `claim` true for every input value for which every node of `hyps` is true? Each node
/// is 1 bit wide. `timeout` in seconds.
pub fn valid(ar: &Arena, hyps: &[E], claim: E, timeout: f64) -> Verdict {
    for &h in hyps.iter().chain(std::iter::once(&claim)) {
        if ar.width[h as usize] != 1 {
            return Verdict::Unknown(format!("node {h} is not a 1-bit condition"));
        }
    }
    if ar.is_const(claim) == Some(1) {
        return Verdict::Valid;
    }
    backend(ar, hyps, claim, timeout)
}

#[cfg(not(feature = "smt"))]
fn backend(_ar: &Arena, _hyps: &[E], _claim: E, _timeout: f64) -> Verdict {
    Verdict::Unknown("no SMT solver built (cargo feature smt)".into())
}

#[cfg(feature = "smt")]
fn backend(ar: &Arena, hyps: &[E], claim: E, timeout: f64) -> Verdict {
    use std::fmt::Write as _;

    use crate::symb::Node;

    let mut cone: Vec<E> = hyps
        .iter()
        .chain(std::iter::once(&claim))
        .flat_map(|&r| ar.cone(r))
        .collect();
    cone.sort_unstable();
    cone.dedup();
    let (mut decls, mut defs) = (String::new(), String::new());
    if crate::solve::smt::emit_cone(ar, &cone, true, &mut HashMap::new(), &mut decls, &mut defs)
        .is_none()
    {
        return Verdict::Unknown("a node is not expressible in SMT-LIB".into());
    }
    let inputs: Vec<E> = cone
        .iter()
        .copied()
        .filter(|&x| matches!(ar.nodes[x as usize], Node::Input { .. }))
        .collect();
    // uninterpreted parts: a model is a candidate only
    let exact = cone.iter().all(|&x| {
        let n = &ar.nodes[x as usize];
        crate::solve::smt::validated(n)
            && !matches!(
                n,
                Node::Opaque { .. }
                    | Node::Sreg { .. }
                    | Node::Inst { .. }
                    | Node::AtAddr { .. }
                    | Node::Bin {
                        op: crate::symb::Bin::UDiv
                            | crate::symb::Bin::URem
                            | crate::symb::Bin::SDiv
                            | crate::symb::Bin::SRem,
                        ..
                    }
            )
    });
    let mut q = String::from("(set-logic ALL)\n(set-option :produce-models true)\n");
    q.push_str(&decls);
    q.push_str(&defs);
    for &h in hyps {
        writeln!(q, "(assert (= n{h} #b1))").ok();
    }
    writeln!(q, "(assert (= n{claim} #b0))").ok();
    q.push_str("(check-sat)\n");
    if !inputs.is_empty() {
        let vs: Vec<String> = inputs.iter().map(|v| format!("n{v}")).collect();
        writeln!(q, "(get-value ({}))", vs.join(" ")).ok();
    }
    if let Ok(d) = std::env::var("PTXV_SMT_DUMP") {
        let _ = std::fs::write(format!("{d}/valid{claim}.smt2"), &q);
    }
    let text = match run_solver(&q, timeout) {
        Ok(t) => t,
        Err(e) => return Verdict::Unknown(e),
    };
    let first = text.lines().next().unwrap_or("").trim().to_string();
    match first.as_str() {
        "unsat" => Verdict::Valid,
        "sat" => {
            let mut model = Vec::new();
            for &v in &inputs {
                let key = format!("(n{v} ");
                let Some(i) = text.find(&key) else { continue };
                let tok: String = text[i + key.len()..]
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
                    model.push((v, x));
                }
            }
            let checked = exact && check(ar, hyps, claim, &model);
            Verdict::Invalid { model, checked }
        }
        _ => Verdict::Unknown(format!(
            "solver: {}",
            if first.is_empty() {
                "no answer"
            } else {
                &first
            }
        )),
    }
}

#[cfg(feature = "smt")]
fn run_solver(q: &str, timeout: f64) -> Result<String, String> {
    use std::io::Write as _;
    let solver = std::env::var("PTXV_SMT_SOLVER").unwrap_or_else(|_| "z3".into());
    let t = format!("-T:{}", timeout.ceil().max(1.0) as u64);
    let mut child = std::process::Command::new(&solver)
        .args(["-in", "-smt2", &t])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|_| format!("cannot run {solver}"))?;
    child
        .stdin
        .take()
        .ok_or("no solver stdin")?
        .write_all(q.as_bytes())
        .map_err(|_| "write to the solver failed".to_string())?;
    let out = child
        .wait_with_output()
        .map_err(|_| "no solver output".to_string())?;
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

/// An incremental solver session: one solver process, conditions asserted one at a time (the
/// cones emitted once), `check` asks for a model of everything asserted so far. A query of
/// "C ∧ ¬π_1 ∧ … ∧ ¬π_n" grows by one blocking condition per explored class without
/// resending the others. `Unsat` from `check` means no input satisfies the assertions (so the
/// negation of their conjunction is valid); `Sat` returns the input leaves' values.
pub struct Session {
    #[cfg(feature = "smt")]
    child: Option<std::process::Child>,
    #[cfg(feature = "smt")]
    out: Option<std::io::BufReader<std::process::ChildStdout>>,
    #[cfg(feature = "smt")]
    emitted: std::collections::HashSet<E>,
    #[cfg(feature = "smt")]
    ufs: HashMap<String, usize>,
    /// Input leaves in the asserted cones (the model's variables).
    #[cfg_attr(not(feature = "smt"), allow(dead_code))]
    inputs: Vec<E>,
    /// A node is not expressible, or the solver failed: every check is unknown.
    broken: Option<String>,
}

/// The answer of `Session::check`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Check {
    /// No input satisfies the assertions.
    Unsat,
    /// A model: input leaves (node, value).
    Sat(Vec<(E, u64)>),
    Unknown(String),
}

impl Session {
    /// A session with `timeout` seconds per check. Without the feature (or the solver) every
    /// check is unknown.
    pub fn new(timeout: f64) -> Session {
        #[cfg(feature = "smt")]
        {
            use std::io::Write as _;
            let solver = std::env::var("PTXV_SMT_SOLVER").unwrap_or_else(|_| "z3".into());
            let spawned = std::process::Command::new(&solver)
                .args(["-in", "-smt2"])
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .spawn();
            let mut s = Session {
                child: None,
                out: None,
                emitted: Default::default(),
                ufs: HashMap::new(),
                inputs: Vec::new(),
                broken: None,
            };
            let Ok(mut child) = spawned else {
                s.broken = Some(format!("cannot run {solver}"));
                return s;
            };
            s.out = child.stdout.take().map(std::io::BufReader::new);
            let ms = (timeout.max(1.0) * 1000.0) as u64;
            let init = format!(
                "(set-option :print-success false)\n(set-option :produce-models true)\n(set-logic ALL)\n(set-option :timeout {ms})\n"
            );
            let ok = child
                .stdin
                .as_mut()
                .is_some_and(|si| si.write_all(init.as_bytes()).is_ok());
            if !ok || s.out.is_none() {
                s.broken = Some("solver pipes".into());
            }
            s.child = Some(child);
            s
        }
        #[cfg(not(feature = "smt"))]
        {
            let _ = timeout;
            Session {
                inputs: Vec::new(),
                broken: Some("no SMT solver built (cargo feature smt)".into()),
            }
        }
    }

    /// Assert that 1-bit node `e` of `ar` equals `val`. Every node asserted in one session
    /// must come from the same arena.
    pub fn assert(&mut self, ar: &Arena, e: E, val: bool) {
        let _ = val;
        if self.broken.is_some() {
            return;
        }
        if ar.width[e as usize] != 1 {
            self.broken = Some(format!("node {e} is not a 1-bit condition"));
            return;
        }
        #[cfg(feature = "smt")]
        {
            use std::fmt::Write as _;
            use std::io::Write as _;

            use crate::symb::Node;
            let cone: Vec<E> = ar
                .cone(e)
                .into_iter()
                .filter(|x| !self.emitted.contains(x))
                .collect();
            let mut cone = cone;
            cone.sort_unstable();
            let (mut decls, mut defs) = (String::new(), String::new());
            if crate::solve::smt::emit_cone(ar, &cone, true, &mut self.ufs, &mut decls, &mut defs)
                .is_none()
            {
                self.broken = Some("a node is not expressible in SMT-LIB".into());
                return;
            }
            for &x in &cone {
                self.emitted.insert(x);
                if matches!(ar.nodes[x as usize], Node::Input { .. }) {
                    self.inputs.push(x);
                }
            }
            let mut q = decls;
            q.push_str(&defs);
            writeln!(q, "(assert (= n{e} {}))", if val { "#b1" } else { "#b0" }).ok();
            let ok = self
                .child
                .as_mut()
                .and_then(|c| c.stdin.as_mut())
                .is_some_and(|si| si.write_all(q.as_bytes()).is_ok());
            if !ok {
                self.broken = Some("write to the solver failed".into());
            }
        }
    }

    /// A model of every assertion so far, or unsat, or unknown.
    pub fn check(&mut self) -> Check {
        if let Some(b) = &self.broken {
            return Check::Unknown(b.clone());
        }
        #[cfg(feature = "smt")]
        {
            use std::io::BufRead as _;
            use std::io::Write as _;
            let (Some(child), Some(out)) = (self.child.as_mut(), self.out.as_mut()) else {
                return Check::Unknown("no solver".into());
            };
            let Some(si) = child.stdin.as_mut() else {
                return Check::Unknown("no solver stdin".into());
            };
            if si.write_all(b"(check-sat)\n").is_err() || si.flush().is_err() {
                self.broken = Some("write to the solver failed".into());
                return Check::Unknown("write to the solver failed".into());
            }
            let mut line = String::new();
            loop {
                line.clear();
                match out.read_line(&mut line) {
                    Ok(0) | Err(_) => {
                        self.broken = Some("the solver exited".into());
                        return Check::Unknown("the solver exited".into());
                    }
                    Ok(_) if line.trim().is_empty() => continue,
                    Ok(_) => break,
                }
            }
            match line.trim() {
                "unsat" => Check::Unsat,
                "sat" => {
                    if self.inputs.is_empty() {
                        return Check::Sat(Vec::new());
                    }
                    let vs: Vec<String> = self.inputs.iter().map(|v| format!("n{v}")).collect();
                    let q = format!("(get-value ({}))\n", vs.join(" "));
                    if si.write_all(q.as_bytes()).is_err() || si.flush().is_err() {
                        return Check::Unknown("write to the solver failed".into());
                    }
                    // read until the parentheses balance
                    let (mut text, mut depth, mut started) = (String::new(), 0i64, false);
                    loop {
                        line.clear();
                        match out.read_line(&mut line) {
                            Ok(0) | Err(_) => break,
                            Ok(_) => {}
                        }
                        for ch in line.chars() {
                            if ch == '(' {
                                depth += 1;
                                started = true;
                            } else if ch == ')' {
                                depth -= 1;
                            }
                        }
                        text.push_str(&line);
                        if started && depth <= 0 {
                            break;
                        }
                    }
                    let mut model = Vec::new();
                    for &v in &self.inputs {
                        let key = format!("(n{v} ");
                        let Some(i) = text.find(&key) else { continue };
                        let tok: String = text[i + key.len()..]
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
                            model.push((v, x));
                        }
                    }
                    Check::Sat(model)
                }
                other => Check::Unknown(format!("solver: {other}")),
            }
        }
        #[cfg(not(feature = "smt"))]
        Check::Unknown("no SMT solver built (cargo feature smt)".into())
    }
}

#[cfg(feature = "smt")]
impl Drop for Session {
    fn drop(&mut self) {
        if let Some(c) = self.child.as_mut() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// The hypotheses hold and the claim fails under `model` (every other input leaf at its
/// captured value), on the expressions.
#[cfg_attr(not(feature = "smt"), allow(dead_code))]
fn check(ar: &Arena, hyps: &[E], claim: E, model: &[(E, u64)]) -> bool {
    struct NoInst;
    impl crate::symb::InstEval for NoInst {
        fn run_inst(&mut self, _: u32, _: u32, _: u8, _: &[u64]) -> Option<u64> {
            None
        }
    }
    let env: HashMap<E, u64> = model.iter().copied().collect();
    let mut ev = crate::symb::Evaluator::with(ar, Box::new(NoInst));
    hyps.iter().all(|&h| ev.eval(h, &env) == Some(1)) && ev.eval(claim, &env) == Some(0)
}

#[cfg(all(test, feature = "smt"))]
mod tests {
    use ptxv_ir::Cmp;

    use super::*;
    use crate::symb::Bin;
    use crate::symb::SymState;

    fn have_solver() -> bool {
        let solver = std::env::var("PTXV_SMT_SOLVER").unwrap_or_else(|_| "z3".into());
        std::process::Command::new(solver)
            .arg("-version")
            .output()
            .is_ok()
    }

    /// `(a << 3) + (b << 3) == (((a << 8) + (b << 8)) >> 5)` holds for a, b < 2^20 and fails
    /// without the bound (the shifts overflow): a valid claim, a checked counterexample.
    #[test]
    fn shifted_sum_identity_needs_its_bound() {
        if !have_solver() {
            eprintln!("no SMT solver: skipped");
            return;
        }
        let mut sym = SymState::new(Default::default(), 1);
        let ar = &mut sym.ar;
        let a = ar.input(0, 0, 4, 3);
        let b = ar.input(0, 4, 4, 5);
        let (k3, k5, k8) = (ar.konst(32, 3), ar.konst(32, 5), ar.konst(32, 8));
        let (a3, b3) = (ar.bin(Bin::Shl, a, k3), ar.bin(Bin::Shl, b, k3));
        let lhs = ar.bin(Bin::Add, a3, b3);
        let (a8, b8) = (ar.bin(Bin::Shl, a, k8), ar.bin(Bin::Shl, b, k8));
        let s8 = ar.bin(Bin::Add, a8, b8);
        let rhs = ar.bin(Bin::LShr, s8, k5);
        let eq = ar.cmp(Cmp::Eq, lhs, rhs);
        let lim = ar.konst(32, 1 << 20);
        let ha = ar.cmp(Cmp::LtU, a, lim);
        let hb = ar.cmp(Cmp::LtU, b, lim);
        assert_eq!(valid(ar, &[ha, hb], eq, 10.0), Verdict::Valid);
        match valid(ar, &[], eq, 10.0) {
            Verdict::Invalid { checked, .. } => assert!(checked),
            v => panic!("{v:?}"),
        }
    }

    /// The encoder against the evaluator, every validated kind (prove mode): per kind, inputs
    /// at random and edge values over widths 1..64, the claim `node == evaluated value` under
    /// `inputs == values` must be valid (one query per kind and width: the conjunction).
    #[test]
    fn encoder_matches_evaluator() {
        if !have_solver() {
            eprintln!("no SMT solver: skipped");
            return;
        }
        use crate::symb::Un;
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let bins = [
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
        let cmps = [
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
        struct NoInst;
        impl crate::symb::InstEval for NoInst {
            fn run_inst(&mut self, _: u32, _: u32, _: u8, _: &[u64]) -> Option<u64> {
                None
            }
        }
        let mut checked = 0usize;
        for w in [1u8, 8, 16, 32, 64] {
            let m = crate::symb::mask(w);
            for kind in 0..(2 + bins.len() + cmps.len() + 6) {
                let mut sym = SymState::new(Default::default(), 1);
                let mut hyps = Vec::new();
                let mut claims = Vec::new();
                let mut env = HashMap::new();
                let mut roots = Vec::new();
                for i in 0..12u64 {
                    let edge = [0, 1, m, m >> 1, (m >> 1) + 1, w as u64, w as u64 + 1];
                    let mut val = |k: u64| -> u64 {
                        if (i + k) % 3 == 0 {
                            edge[((i + k) as usize) % edge.len()] & m
                        } else {
                            rnd() & m
                        }
                    };
                    let (va, vb, vc) = (val(0), val(1), val(2));
                    let ar = &mut sym.ar;
                    let a = ar.input(0, 24 * i, w.div_ceil(8).max(1), va);
                    let b = ar.input(0, 24 * i + 8, w.div_ceil(8).max(1), vb);
                    let c = ar.input(0, 24 * i + 16, w.div_ceil(8).max(1), vc);
                    // inputs of width w (input leaves are whole bytes: take the low w bits)
                    let (a, b, c) = if w % 8 != 0 {
                        (
                            ar.extract(0, w, a),
                            ar.extract(0, w, b),
                            ar.extract(0, w, c),
                        )
                    } else {
                        (a, b, c)
                    };
                    let root = match kind {
                        0 => ar.un(Un::Not, a),
                        1 => ar.un(Un::Neg, a),
                        k if k < 2 + bins.len() => ar.bin(bins[k - 2], a, b),
                        k if k < 2 + bins.len() + cmps.len() => {
                            ar.cmp(cmps[k - 2 - bins.len()], a, b)
                        }
                        k => match k - 2 - bins.len() - cmps.len() {
                            0 => {
                                let p = ar.cmp(Cmp::LtU, a, b);
                                ar.ite(p, b, c)
                            }
                            1 => ar.zext(w.max(1).saturating_mul(2).min(64).max(w), a),
                            2 => ar.sext(w.saturating_mul(2).min(64).max(w), a),
                            3 => ar.extract(0, (w / 2).max(1), a),
                            4 if w <= 32 => ar.concat(a, b),
                            _ => ar.lop3((rnd() & 0xff) as u8, a, b, c),
                        },
                    };
                    // constants, and undefined results the arena made opaque (a division by a
                    // captured zero), test nothing
                    if ar.is_const(root).is_some()
                        || matches!(ar.nodes[root as usize], crate::symb::Node::Opaque { .. })
                    {
                        continue;
                    }
                    roots.push(root);
                    for (leaf, v) in [(a, va), (b, vb), (c, vc)] {
                        for l in ar.leaves(leaf) {
                            if matches!(ar.nodes[l as usize], crate::symb::Node::Input { .. }) {
                                let lw = ar.width[l as usize];
                                env.insert(l, v & crate::symb::mask(lw));
                            }
                        }
                    }
                }
                let ar = &mut sym.ar;
                for (&l, &v) in env.clone().iter() {
                    let k = ar.konst(ar.width[l as usize], v);
                    hyps.push(ar.cmp(Cmp::Eq, l, k));
                }
                for &r in &roots {
                    let mut ev = crate::symb::Evaluator::with(ar, Box::new(NoInst));
                    let Some(v) = ev.eval(r, &env) else { continue };
                    drop(ev);
                    let k = ar.konst(ar.width[r as usize], v);
                    claims.push(ar.cmp(Cmp::Eq, r, k));
                }
                if claims.is_empty() {
                    continue;
                }
                let mut all = claims[0];
                for &c in &claims[1..] {
                    all = ar.bin(Bin::And, all, c);
                }
                checked += claims.len();
                assert_eq!(
                    valid(ar, &hyps, all, 60.0),
                    Verdict::Valid,
                    "kind {kind} width {w}"
                );
            }
        }
        assert!(checked > 500, "{checked}");
    }

    /// A session: x < 10 blocked by x < 5 and then by 5 <= x: unsat; one blocking condition
    /// less: a model with 5 <= x < 10.
    #[test]
    fn session_blocks_conditions_incrementally() {
        if !have_solver() {
            eprintln!("no SMT solver: skipped");
            return;
        }
        let mut sym = SymState::new(Default::default(), 1);
        let ar = &mut sym.ar;
        let x = ar.input(0, 0, 4, 0);
        let (k5, k10) = (ar.konst(32, 5), ar.konst(32, 10));
        let c = ar.cmp(Cmp::LtU, x, k10);
        let p1 = ar.cmp(Cmp::LtU, x, k5);
        let p2 = ar.cmp(Cmp::GeU, x, k5);
        let mut s = Session::new(10.0);
        s.assert(ar, c, true);
        s.assert(ar, p1, false);
        match s.check() {
            Check::Sat(m) => {
                assert_eq!(m.len(), 1);
                assert!((5..10).contains(&m[0].1), "{m:?}");
            }
            c => panic!("{c:?}"),
        }
        s.assert(ar, p2, false);
        assert_eq!(s.check(), Check::Unsat);
    }

    /// An opaque leaf is unconstrained: `x == x_captured` is not valid for an opaque x even
    /// though the search's queries fix it at its captured value.
    #[test]
    fn opaque_leaf_is_unconstrained() {
        if !have_solver() {
            eprintln!("no SMT solver: skipped");
            return;
        }
        let mut sym = SymState::new(Default::default(), 1);
        let ar = &mut sym.ar;
        let o = ar.opaque(32, 7, "test");
        let k = ar.konst(32, 7);
        let eq = ar.cmp(Cmp::Eq, o, k);
        match valid(ar, &[], eq, 10.0) {
            Verdict::Invalid { checked, .. } => assert!(!checked),
            v => panic!("{v:?}"),
        }
    }
}

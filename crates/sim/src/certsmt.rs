// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Infeasibility proofs for exhaustive concolic expansion (cert.rs): "no input satisfies these
//! atoms" as an SMT-LIB2 query whose every model over-approximates the simulator, answered by
//! several independent external solvers.
//!
//! The encoding is sound for UNSAT by construction:
//! - exact only where the SMT-LIB semantics equals the simulator's (symb.rs `eval_node`):
//!   bit-vector add, sub, mul, high multiplies, bitwise operations and LUTs, shifts (an amount
//!   of the width or more gives 0, or the sign for an arithmetic shift, in both), min / max,
//!   comparisons, selects, extension, extraction, concatenation, f32 / f64 sign operations and
//!   the bf16 widening (bit moves), and IEEE comparisons without FTZ;
//! - every other operation (division and remainder, every floating-point arithmetic, MUFU,
//!   conversions, packs, re-run instructions) is an uninterpreted function of its operands:
//!   any function, the hardware's included, is a model;
//! - opaque leaves, placement-dependent special registers and the bytes at another address
//!   (`AtAddr` off its captured address) are free variables; input leaves are free.
//!
//! An UNSAT therefore holds for the simulator's semantics whatever the uninterpreted parts
//! compute. It is accepted only when at least two solvers answer `unsat` and none answers
//! `sat` (ASSUMPTIONS.md C-EXH-UNSAT).

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Write;

use ptxv_ir::Cmp;
use ptxv_ir::FCmp;
use serde::Serialize;

use crate::symb::Arena;
use crate::symb::Bin;
use crate::symb::E;
use crate::symb::Node;
use crate::symb::Un;

/// One external solver: a name and the command reading SMT-LIB2 on stdin.
#[derive(Clone, Debug, Serialize)]
pub struct Solver {
    pub name: String,
    pub cmd: String,
    /// What `cmd` reports as its version (`version_of`), or why none.
    pub version: String,
}

/// The solvers of `PTXV_EXH_SOLVERS` (`name=command,...`; default `z3`, `cvc5`, `bitwuzla`
/// from PATH), with their versions; those that do not run are dropped.
pub fn solvers() -> Vec<Solver> {
    let spec = std::env::var("PTXV_EXH_SOLVERS")
        .unwrap_or_else(|_| "z3=z3,cvc5=cvc5,bitwuzla=bitwuzla".into());
    spec.split(',')
        .filter_map(|kv| {
            let (name, cmd) = kv.split_once('=').unwrap_or((kv, kv));
            let version = version_of(name.trim(), cmd.trim())?;
            Some(Solver {
                name: name.trim().into(),
                cmd: cmd.trim().into(),
                version,
            })
        })
        .collect()
}

fn version_of(name: &str, cmd: &str) -> Option<String> {
    let arg = if name == "z3" {
        "-version"
    } else {
        "--version"
    };
    let out = std::process::Command::new(cmd).arg(arg).output().ok()?;
    let text =
        String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim()
        .to_string();
    (!line.is_empty()).then_some(line)
}

/// The answer of one solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Answer {
    Sat,
    Unsat,
    Unknown,
}

/// The verdict on one query: per solver, its answer.
#[derive(Clone, Debug, Serialize)]
pub struct Verdict {
    pub answers: BTreeMap<String, Answer>,
}

impl Verdict {
    /// At least two `unsat` and no `sat` (C-EXH-UNSAT).
    pub fn proven_unsat(&self) -> bool {
        let u = self
            .answers
            .values()
            .filter(|&&a| a == Answer::Unsat)
            .count();
        let s = self.answers.values().filter(|&&a| a == Answer::Sat).count();
        u >= 2 && s == 0
    }

    pub fn any_sat(&self) -> bool {
        self.answers.values().any(|&a| a == Answer::Sat)
    }
}

fn bvc(w: u8, v: u64) -> String {
    format!("(_ bv{v} {w})")
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

/// The parameters of an operation without its operands (`Fma { a: 3, b: 4, c: Some(5), rnd:
/// Rn, ftz: false }` -> `Fma__rnd__Rn__ftz__false`): the name of its uninterpreted function.
fn op_name(node: &Node) -> String {
    let d = format!("{node:?}");
    let mut out = String::new();
    let mut rest = d.as_str();
    while !rest.is_empty() {
        let field = ["a: ", "b: ", "c: "]
            .iter()
            .find(|f| rest.starts_with(**f) && (out.ends_with(' ') || out.ends_with('{')));
        if let Some(f) = field {
            rest = &rest[f.len()..];
            let skip = rest.find([',', '}']).unwrap_or(rest.len());
            rest = &rest[skip..];
            out.push('_');
            continue;
        }
        let c = rest.chars().next().unwrap_or(' ');
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// Declarations and definitions of the cones of `roots` (one `define-fun` per node), and
/// whether they use the floating-point theory.
fn cone_text(ar: &Arena, roots: &[E]) -> (String, bool) {
    let mut cone: Vec<E> = roots.iter().flat_map(|&e| ar.cone(e)).collect();
    cone.sort_unstable();
    cone.dedup();
    let n = |x: E| format!("n{x}");
    let mut decls = String::new();
    let mut defs = String::new();
    let mut ufs: HashSet<String> = HashSet::new();
    let mut fp = false;
    let mut uf = |decls: &mut String, name: String, args: &[E], w: u8| -> String {
        let ws: Vec<String> = args
            .iter()
            .map(|&k| ar.width[k as usize].to_string())
            .collect();
        let full = format!("{name}_{}_{w}", ws.join("_"));
        if ufs.insert(full.clone()) {
            let tys: Vec<String> = ws.iter().map(|x| format!("(_ BitVec {x})")).collect();
            let _ = writeln!(
                decls,
                "(declare-fun {full} ({}) (_ BitVec {w}))",
                tys.join(" ")
            );
        }
        if args.is_empty() {
            full
        } else {
            format!(
                "({full} {})",
                args.iter().map(|&k| n(k)).collect::<Vec<_>>().join(" ")
            )
        }
    };
    for &x in &cone {
        let w = ar.width[x as usize];
        let ty = format!("(_ BitVec {w})");
        let node = &ar.nodes[x as usize];
        let body: String = match node {
            Node::Const { v, .. } => bvc(w, *v),
            Node::Input { .. } | Node::Opaque { .. } | Node::Sreg { .. } => {
                let _ = writeln!(decls, "(declare-fun {} () {ty})", n(x));
                continue;
            }
            Node::Un { op, a, .. } => match op {
                Un::Not => format!("(bvnot {})", n(*a)),
                Un::Neg => format!("(bvneg {})", n(*a)),
                Un::FNeg => format!("(bvxor {} #x80000000)", n(*a)),
                Un::FAbs => format!("(bvand {} #x7fffffff)", n(*a)),
                Un::B2F => format!("(concat {} #x0000)", n(*a)),
                _ => uf(&mut decls, format!("un_{op:?}"), &[*a], w),
            },
            Node::Bin { op, a, b, .. } => {
                let (x1, y1) = (n(*a), n(*b));
                match op {
                    Bin::Add => format!("(bvadd {x1} {y1})"),
                    Bin::Sub => format!("(bvsub {x1} {y1})"),
                    Bin::Mul => format!("(bvmul {x1} {y1})"),
                    Bin::And => format!("(bvand {x1} {y1})"),
                    Bin::Or => format!("(bvor {x1} {y1})"),
                    Bin::Xor => format!("(bvxor {x1} {y1})"),
                    Bin::Shl => format!("(bvshl {x1} {y1})"),
                    Bin::LShr => format!("(bvlshr {x1} {y1})"),
                    Bin::AShr => format!("(bvashr {x1} {y1})"),
                    Bin::MulHiU => format!(
                        "((_ extract {} {w}) (bvmul ((_ zero_extend {w}) {x1}) ((_ zero_extend {w}) {y1})))",
                        2 * w as u32 - 1
                    ),
                    Bin::MulHiS => format!(
                        "((_ extract {} {w}) (bvmul ((_ sign_extend {w}) {x1}) ((_ sign_extend {w}) {y1})))",
                        2 * w as u32 - 1
                    ),
                    Bin::UMin => format!("(ite (bvule {x1} {y1}) {x1} {y1})"),
                    Bin::UMax => format!("(ite (bvuge {x1} {y1}) {x1} {y1})"),
                    Bin::SMin => format!("(ite (bvsle {x1} {y1}) {x1} {y1})"),
                    Bin::SMax => format!("(ite (bvsge {x1} {y1}) {x1} {y1})"),
                    // Division by zero has no value in the simulator: uninterpreted.
                    Bin::UDiv | Bin::SDiv | Bin::URem | Bin::SRem => {
                        uf(&mut decls, format!("bin_{op:?}"), &[*a, *b], w)
                    }
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
                let (x1, y1) = (n(*a), n(*b));
                let p = match cmp {
                    Cmp::Eq => format!("(= {x1} {y1})"),
                    Cmp::Ne => format!("(not (= {x1} {y1}))"),
                    Cmp::LtU => format!("(bvult {x1} {y1})"),
                    Cmp::LeU => format!("(bvule {x1} {y1})"),
                    Cmp::GtU => format!("(bvugt {x1} {y1})"),
                    Cmp::GeU => format!("(bvuge {x1} {y1})"),
                    Cmp::LtS => format!("(bvslt {x1} {y1})"),
                    Cmp::LeS => format!("(bvsle {x1} {y1})"),
                    Cmp::GtS => format!("(bvsgt {x1} {y1})"),
                    Cmp::GeS => format!("(bvsge {x1} {y1})"),
                };
                format!("(ite {p} #b1 #b0)")
            }
            Node::Ite { c, a, b } => format!("(ite (= {} #b1) {} {})", n(*c), n(*a), n(*b)),
            Node::Zext { a, .. } => {
                format!("((_ zero_extend {}) {})", w - ar.width[*a as usize], n(*a))
            }
            Node::Sext { a, .. } => {
                format!("((_ sign_extend {}) {})", w - ar.width[*a as usize], n(*a))
            }
            Node::Extract { lo, a, .. } => format!("((_ extract {} {lo}) {})", lo + w - 1, n(*a)),
            Node::Concat { hi, lo } => format!("(concat {} {})", n(*hi), n(*lo)),
            Node::FCmp {
                cmp,
                ftz: false,
                a,
                b,
            } if ar.width[*a as usize] == 32 => {
                fp = true;
                let (x1, y1) = (
                    format!("((_ to_fp 8 24) {})", n(*a)),
                    format!("((_ to_fp 8 24) {})", n(*b)),
                );
                format!("(ite {} #b1 #b0)", fcmp_text(*cmp, &x1, &y1))
            }
            Node::DCmp { cmp, a, b } => {
                fp = true;
                let (x1, y1) = (
                    format!("((_ to_fp 11 53) {})", n(*a)),
                    format!("((_ to_fp 11 53) {})", n(*b)),
                );
                format!("(ite {} #b1 #b0)", fcmp_text(*cmp, &x1, &y1))
            }
            // one free value per node: each dynamic read is its own node (`inst`)
            Node::AtAddr { addr, at, val, .. } => {
                let _ = writeln!(decls, "(declare-fun o{x} () {ty})");
                format!(
                    "(ite (= {} {}) {} o{x})",
                    n(*addr),
                    bvc(ar.width[*addr as usize], *at),
                    n(*val)
                )
            }
            // A re-run instruction: a function of its arguments, one per instruction and
            // thread (it may read the thread's special registers).
            Node::Inst {
                pc, t, out, args, ..
            } => uf(&mut decls, format!("inst_{pc}_{t}_{out}"), args, w),
            // Every other operation: an uninterpreted function of its operands, named by the
            // operation and its parameters (the same parameters, the same function).
            other => {
                let args = crate::symb::kids(other);
                uf(&mut decls, format!("op_{}", op_name(other)), &args, w)
            }
        };
        let _ = writeln!(defs, "(define-fun {} () {ty} {body})", n(x));
    }
    (decls + &defs, fp)
}

/// The query "these 1-bit nodes take these values" over `ar` (SMT-LIB2).
pub fn query(ar: &Arena, atoms: &[(E, bool)]) -> String {
    query_ne(ar, atoms, None)
}

/// `query`, and (`ne`) "node `e` differs from `v`".
pub fn query_ne(ar: &Arena, atoms: &[(E, bool)], ne: Option<(E, u64)>) -> String {
    let mut roots: Vec<E> = atoms.iter().map(|a| a.0).collect();
    roots.extend(ne.map(|x| x.0));
    let (text, fp) = cone_text(ar, &roots);
    let logic = if fp { "ALL" } else { "QF_UFBV" };
    let mut q = format!("(set-logic {logic})\n{text}");
    for &(e, v) in atoms {
        let _ = writeln!(q, "(assert (= n{e} {}))", if v { "#b1" } else { "#b0" });
    }
    if let Some((e, v)) = ne {
        let _ = writeln!(
            q,
            "(assert (not (= n{e} {})))",
            bvc(ar.width[e as usize], v)
        );
    }
    q.push_str("(check-sat)\n");
    q
}

/// Run `q` through every solver (`timeout` seconds each).
pub fn decide(solvers: &[Solver], q: &str, timeout: f64) -> Verdict {
    let mut answers = BTreeMap::new();
    let mut hs = Vec::new();
    for s in solvers {
        let (cmd, name, q) = (s.cmd.clone(), s.name.clone(), q.to_string());
        let t = timeout.ceil().max(1.0) as u64;
        hs.push(std::thread::spawn(move || {
            (name.clone(), run_one(&name, &cmd, &q, t))
        }));
    }
    for h in hs {
        if let Ok((name, a)) = h.join() {
            answers.insert(name, a);
        }
    }
    Verdict { answers }
}

fn run_one(name: &str, cmd: &str, q: &str, t: u64) -> Answer {
    use std::io::Write as _;
    let args: Vec<String> = if name == "z3" {
        vec!["-in".into(), "-smt2".into(), format!("-T:{t}")]
    } else {
        vec![format!("-T:{t}")]
    };
    let Ok(mut child) = std::process::Command::new("timeout")
        .arg(format!("{}", t + 5))
        .arg(cmd)
        .args(&args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    else {
        return Answer::Unknown;
    };
    if let Some(mut si) = child.stdin.take() {
        if si.write_all(q.as_bytes()).is_err() {
            return Answer::Unknown;
        }
    }
    let Ok(out) = child.wait_with_output() else {
        return Answer::Unknown;
    };
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    if text.lines().any(|l| l.trim_start().starts_with("(error")) {
        return Answer::Unknown;
    }
    match text.lines().next().unwrap_or("").trim() {
        "unsat" => Answer::Unsat,
        "sat" => Answer::Sat,
        _ => Answer::Unknown,
    }
}

/// One query of a batch: atoms with their values, and optionally `(e, v)` for "e != v"
/// (without it: the atoms alone).
pub type Query = (Vec<(E, bool)>, Option<(E, u64)>);

/// Per solver: queries and answers (for the record's `solver_stats`).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Stats {
    pub queries: usize,
    pub per_solver: BTreeMap<String, BTreeMap<String, usize>>,
    pub versions: BTreeMap<String, String>,
}

impl Stats {
    pub fn add(&mut self, v: &Verdict) {
        self.queries += 1;
        for (k, a) in &v.answers {
            *self
                .per_solver
                .entry(k.clone())
                .or_default()
                .entry(format!("{a:?}").to_lowercase())
                .or_default() += 1;
        }
    }
}

/// The values a solver gives the encodings of `es` with the leaves of their cones at the
/// values in `leaves` (tests: the exact encodings must agree with `eval_node`).
pub fn eval_with(
    solver: &Solver,
    ar: &Arena,
    es: &[E],
    leaves: &HashMap<E, u64>,
) -> Option<Vec<Option<u64>>> {
    let (text, fp) = cone_text(ar, es);
    let mut q = format!("(set-logic {})\n{text}", if fp { "ALL" } else { "QF_UFBV" });
    let cone: HashSet<E> = es.iter().flat_map(|&e| ar.cone(e)).collect();
    for (&l, &v) in leaves.iter().filter(|x| cone.contains(x.0)) {
        let _ = writeln!(q, "(assert (= n{l} {}))", bvc(ar.width[l as usize], v));
    }
    q.push_str("(check-sat)\n");
    let names: Vec<String> = es.iter().map(|e| format!("n{e}")).collect();
    let _ = writeln!(q, "(get-value ({}))", names.join(" "));
    use std::io::Write as _;
    let args: Vec<&str> = if solver.name == "z3" {
        vec!["-in", "-smt2"]
    } else {
        vec![]
    };
    let mut child = std::process::Command::new(&solver.cmd)
        .args(&args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    child.stdin.take()?.write_all(q.as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout).to_string();
    if s.contains("(error") {
        return None;
    }
    Some(
        es.iter()
            .map(|e| {
                let key = format!("(n{e} ");
                let i = s.find(&key)?;
                let tok: String = s[i + key.len()..]
                    .chars()
                    .take_while(|c| !c.is_whitespace() && *c != ')')
                    .collect();
                if let Some(h) = tok.strip_prefix("#x") {
                    u64::from_str_radix(h, 16).ok()
                } else if let Some(b) = tok.strip_prefix("#b") {
                    u64::from_str_radix(b, 2).ok()
                } else {
                    None
                }
            })
            .collect(),
    )
}

/// Many queries over one arena in one solver run per solver: the cone of `roots` declared
/// once, then per item `(push 1)`, its assertions (`(E, bool)` atoms and an optional
/// "differs from"), `(check-sat)`, `(pop 1)`. One verdict per item.
pub fn decide_many(
    solvers: &[Solver],
    ar: &Arena,
    roots: &[E],
    items: &[Query],
    timeout: f64,
) -> Vec<Verdict> {
    decide_along(
        solvers,
        ar,
        roots,
        &[],
        &vec![0; items.len()],
        items,
        timeout,
    )
}

/// [`decide_many`] along a path: item `i` holds under `steps[..ks[i]]` as well as its own
/// assertions. `ks` is non-decreasing; the steps are asserted once, outside every push, as
/// the items reach them (one incremental solver context instead of the prefix per item).
pub fn decide_along(
    solvers: &[Solver],
    ar: &Arena,
    roots: &[E],
    steps: &[(E, bool)],
    ks: &[usize],
    items: &[Query],
    timeout: f64,
) -> Vec<Verdict> {
    assert!(ks.len() == items.len() && ks.windows(2).all(|w| w[0] <= w[1]));
    // Two solvers first (bitwuzla, z3, cvc5 in that order of preference; cvc5 is the slowest
    // in incremental mode on long paths); the others only for
    // the items those two leave undecided (not two unsat, no sat).
    let rank = |s: &Solver| {
        ["bitwuzla", "z3", "cvc5"]
            .iter()
            .position(|n| *n == s.name)
            .unwrap_or(3)
    };
    let mut order: Vec<Solver> = solvers.to_vec();
    order.sort_by_key(rank);
    let (first, rest) = order.split_at(order.len().min(2));
    let mut out = decide_batch(first, ar, roots, steps, ks, items, timeout);
    if rest.is_empty() {
        return out;
    }
    let open: Vec<usize> = (0..items.len())
        .filter(|&i| !out[i].proven_unsat() && !out[i].any_sat())
        .collect();
    if open.is_empty() {
        return out;
    }
    let sub: Vec<Query> = open.iter().map(|&i| items[i].clone()).collect();
    let sub_ks: Vec<usize> = open.iter().map(|&i| ks[i]).collect();
    let more = decide_batch(rest, ar, roots, steps, &sub_ks, &sub, timeout);
    for (k, &i) in open.iter().enumerate() {
        out[i].answers.extend(more[k].answers.clone());
    }
    out
}

fn decide_batch(
    solvers: &[Solver],
    ar: &Arena,
    roots: &[E],
    steps: &[(E, bool)],
    ks: &[usize],
    items: &[Query],
    timeout: f64,
) -> Vec<Verdict> {
    let (text, fp) = cone_text(ar, roots);
    let t = timeout.ceil().max(1.0) as u64;
    let mut script = format!(
        "(set-option :print-success false)\n(set-logic {})\n{text}",
        if fp { "ALL" } else { "QF_UFBV" }
    );
    let mut done = 0;
    for (i, (atoms, ne)) in items.iter().enumerate() {
        for &(e, v) in &steps[done..ks[i]] {
            let _ = writeln!(
                script,
                "(assert (= n{e} {}))",
                if v { "#b1" } else { "#b0" }
            );
        }
        done = done.max(ks[i]);
        script.push_str("(push 1)\n");
        for &(e, v) in atoms {
            let _ = writeln!(
                script,
                "(assert (= n{e} {}))",
                if v { "#b1" } else { "#b0" }
            );
        }
        if let Some((e, v)) = ne {
            let _ = writeln!(
                script,
                "(assert (not (= n{e} {})))",
                bvc(ar.width[*e as usize], *v)
            );
        }
        let _ = writeln!(script, "(echo \"q{i}\")\n(check-sat)\n(pop 1)");
    }
    // Sentinel: the asserted steps alone. The captured input satisfies them, so a solver that
    // calls them unsat has a wrong encoding or a wrong answer: all its answers are dropped.
    let sentinel = !steps.is_empty();
    if sentinel {
        let _ = writeln!(script, "(echo \"q{}\")\n(check-sat)", items.len());
    }
    if let Ok(d) = std::env::var("PTXV_EXH_DUMP") {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let k = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _ = std::fs::write(format!("{d}/batch{k}.smt2"), &script);
    }
    let t0 = std::time::Instant::now();
    let mut per: Vec<BTreeMap<String, Answer>> = vec![BTreeMap::new(); items.len()];
    let mut hs = Vec::new();
    for s in solvers {
        let (cmd, name, q) = (s.cmd.clone(), s.name.clone(), script.clone());
        let n = items.len() + usize::from(sentinel);
        hs.push(std::thread::spawn(move || {
            (name.clone(), run_many(&name, &cmd, &q, t, n))
        }));
    }
    for h in hs {
        if let Ok((name, mut ans)) = h.join() {
            if sentinel && ans.pop() == Some(Answer::Unsat) {
                ans = vec![Answer::Unknown; items.len()];
            }
            for (i, a) in ans.into_iter().enumerate() {
                per[i].insert(name.clone(), a);
            }
        }
    }
    if std::env::var_os("PTXV_EXH_PROGRESS").is_some() {
        eprintln!(
            "exhaustive concolic: {} queries over {} nodes, {} solver(s): {:.1}s",
            items.len(),
            text.len(),
            solvers.len(),
            t0.elapsed().as_secs_f64()
        );
    }
    per.into_iter().map(|answers| Verdict { answers }).collect()
}

fn run_many(name: &str, cmd: &str, q: &str, t: u64, n: usize) -> Vec<Answer> {
    use std::io::Write as _;
    let mut out = vec![Answer::Unknown; n];
    let args: Vec<String> = if name == "z3" {
        vec!["-in".into(), "-smt2".into(), format!("-t:{}", t * 1000)]
    } else {
        vec![format!("-T:{t}")]
    };
    // The whole batch: at most every query's limit plus a margin.
    let total = (t * n as u64 + 30).min(4 * 3600);
    let Ok(mut child) = std::process::Command::new("timeout")
        .arg(format!("{total}"))
        .arg(cmd)
        .args(&args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    else {
        return out;
    };
    if let Some(mut si) = child.stdin.take() {
        if si.write_all(q.as_bytes()).is_err() {
            return out;
        }
    }
    let Ok(o) = child.wait_with_output() else {
        return out;
    };
    let text = String::from_utf8_lossy(&o.stdout).to_string();
    // An error anywhere: no answer of this run is trusted (a rejected assertion would leave
    // a weaker query).
    if text.lines().any(|l| l.trim_start().starts_with("(error")) {
        return out;
    }
    let mut cur: Option<usize> = None;
    for line in text.lines() {
        let l = line.trim().trim_matches('"');
        if let Some(i) = l.strip_prefix('q').and_then(|x| x.parse::<usize>().ok()) {
            cur = Some(i);
            continue;
        }
        if let Some(i) = cur {
            if i < n {
                out[i] = match l {
                    "unsat" => Answer::Unsat,
                    "sat" => Answer::Sat,
                    _ => Answer::Unknown,
                };
            }
            cur = None;
        }
    }
    out
}

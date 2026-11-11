// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Exhaustive concolic expansion (`ptxv check --exhaustive-concolic`, docs/design.md 3.7):
//! every feasible path of one launch (kernel binary, shape, parameters, launch configuration)
//! over all values of its input buffers is traversed on a concrete, replayed input or proven
//! infeasible; each traversed path's input gets a full ptxv check. The result is a certificate
//! (`certified`: the verdict is clean for every input), a refutation (`refuted`: a replayed
//! input whose check has a finding) or `blocked` with what blocked it.
//!
//! A path is the sequence of atoms of the deterministic coverage run (symb.rs `Atom`, every
//! thread's input-dependent comparisons, guards, branch conditions and selects) in the order
//! the run first observed them. The input space is closed when, for every traversed path and
//! every position i, the region "atoms before i as on the path, atom i the other way" is either
//! the prefix of another traversed path (same decisions and values before i, the other value
//! at i) or infeasible (certsmt.rs: at least two solvers answer unsat for an encoding that
//! over-approximates the simulator). Within one path, the verdict transfers from the path's
//! input to every input of the path only if everything else the verdict depends on is fixed by
//! the path: every input-dependent control, address or synchronization operand
//! (`SymState::operands`) is proven constant under the path's atoms (ASSUMPTIONS.md
//! C-EXH-TRANSFER).

use std::collections::BTreeMap;
use std::collections::HashMap;

use serde::Serialize;

use crate::certsmt;
use crate::coverage::Budget;
use crate::coverage::Domain;
use crate::coverage::Mutation;
use crate::machine::Machine;
use crate::symb::AtomKind;
use crate::symb::E;
use crate::symb::SymState;

type Writes = Vec<(usize, u64, u8, u64)>;

/// Limits of the expansion.
#[derive(Clone, Debug)]
pub struct Options {
    /// Paths traversed (each gets a full check); more feasible paths: blocked.
    pub max_paths: usize,
    /// Wall-clock seconds for the whole expansion (checks of the representatives included).
    pub budget: f64,
    /// Seconds per solver query.
    pub solver_timeout: f64,
    /// Seconds per coverage run (a path's expressions).
    pub run_seconds: f64,
    /// Witness attempts per open region.
    pub tries: usize,
    /// The witness search (solve.rs).
    pub solve: crate::solve::Options,
    /// The SMT solvers (None: `certsmt::solvers()`, from PTXV_EXH_SOLVERS).
    pub solvers: Option<Vec<certsmt::Solver>>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            max_paths: 8,
            budget: 3600.0,
            solver_timeout: 5.0,
            run_seconds: 600.0,
            tries: 2,
            solve: crate::solve::Options {
                smt: false,
                ..Default::default()
            },
            solvers: None,
        }
    }
}

/// The full ptxv check of one representative input, as the caller ran it.
#[derive(Clone, Debug, Serialize)]
pub struct RepCheck {
    /// Verdict classes (`rows[].got`).
    pub classes: Vec<String>,
    /// Exhaustive exploration or certificate (`verdict.complete`).
    pub complete: bool,
    /// Well-synchronized certificate (E-WELLSYNC).
    pub certified: bool,
    pub states: usize,
    /// First finding lines.
    pub findings: Vec<String>,
    /// ASSUMPTIONS.md IDs the check rests on.
    pub assumes: Vec<String>,
    pub seconds: f64,
}

impl RepCheck {
    fn clean(&self) -> bool {
        self.complete && self.classes.iter().all(|c| c == "clean")
    }

    /// A complete check with a finding (a class other than clean).
    fn refutes(&self) -> bool {
        self.complete
            && self
                .classes
                .iter()
                .any(|c| c != "clean" && c != "inconclusive" && c != "unmodeled")
    }
}

/// What kept the expansion from a certificate.
#[derive(Clone, Debug, Serialize)]
pub struct Blocker {
    /// `coverage-incomplete`, `path-truncated`, `provenance-only`, `open-region`,
    /// `operand-varies`, `operand-unproven`, `path-cap`, `deadline`, `check-incomplete`,
    /// `nic-writer`, `path-nondeterminism`, `no-solver`, `clock-read`, `pending-soundness-fix` (a known miss
    /// of the check applies, named in `detail`).
    pub kind: String,
    pub pc: Option<u32>,
    pub at: String,
    pub detail: String,
}

/// One traversed path.
#[derive(Clone, Debug, Serialize)]
pub struct PathInfo {
    /// The input of the path, relative to the captured one (`buffer[i]=v`).
    pub input: String,
    pub atoms: usize,
    pub operands: usize,
    /// Every element the input writes lies in a buffer whose declared invariant admits it
    /// (case `buffers[].invariant`; true for the captured input).
    pub within_invariants: bool,
    pub check: Option<RepCheck>,
}

/// A known miss of the per-path check whose fix has not landed, with a static trigger: the
/// first instruction (or `Some(None)` for the launch) it can apply to.
pub struct OpenMiss {
    pub id: &'static str,
    pub what: &'static str,
    /// The miss affects explored verdicts only (a representative whose check is a
    /// well-synchronized certificate, E-WELLSYNC, is not affected).
    pub explorer_only: bool,
    pub trigger: fn(&Machine) -> Option<Option<u32>>,
}

/// The first instruction whose operation or source text matches.
fn first_inst(m: &Machine, f: impl Fn(&ptxv_ir::Inst, &str) -> bool) -> Option<Option<u32>> {
    m.k.code
        .iter()
        .position(|i| f(i, &i.text.to_ascii_lowercase()))
        .map(|pc| Some(pc as u32))
}

/// A PTX source line (lower-case opcode) rather than a SASS instruction (upper case).
fn is_ptx(i: &ptxv_ir::Inst) -> bool {
    opcode(&i.text).chars().any(|c| c.is_ascii_lowercase())
}

/// PTX opcode of a source line: the text before the first space, without a guard.
fn opcode(t: &str) -> &str {
    let t = t.trim_start();
    let t = if t.starts_with('@') {
        t.split_once(char::is_whitespace)
            .map_or("", |x| x.1)
            .trim_start()
    } else {
        t
    };
    t.split(|c: char| c.is_whitespace() || c == ';')
        .next()
        .unwrap_or("")
}

/// Open misses (the shared registry of known misses), each with a static trigger. The
/// expansion inherits every unlanded fix of the per-path check: a launch a trigger matches
/// blocks on `pending-soundness-fix`. Triggers over-approximate; remove an entry when its fix
/// lands. Text triggers match the PTX source line or the SASS instruction. No trigger: MISS
/// 35 and 38 (hardware L1 levels, outside the claim's verdict classes) and 36 (numerics).
/// Landed: 18, 30, 33, 39.
pub static OPEN_MISSES: &[OpenMiss] = &[
    OpenMiss {
        id: "MISS-28",
        what: "clock-dependent control unreachable (clocks read 0, nanosleep a no-op)",
        explorer_only: false,
        trigger: |m| {
            first_inst(m, |_, t| {
                t.contains("%clock")
                    || t.contains("%globaltimer")
                    || t.contains("sr_clock")
                    || t.contains("sr_globaltimer")
                    || opcode(t).starts_with("nanosleep")
            })
        },
    },
    OpenMiss {
        id: "MISS-29",
        what: "PTX elect.sync: only the lowest and highest leader tried",
        explorer_only: false,
        trigger: |m| first_inst(m, |i, t| is_ptx(i) && opcode(t).starts_with("elect.sync")),
    },
    OpenMiss {
        id: "MISS-31",
        what: "PTX: cross-lane order of strong shared accesses in a converged warp not explored",
        explorer_only: false,
        trigger: |m| {
            first_inst(m, |i, t| {
                let o = opcode(t);
                let parts: Vec<&str> = o.split('.').collect();
                is_ptx(i)
                    && matches!(parts[0], "ld" | "st" | "atom" | "red")
                    && parts.iter().any(|x| x.starts_with("shared"))
                    && parts
                        .iter()
                        .any(|x| matches!(*x, "volatile" | "relaxed" | "acquire" | "release"))
            })
        },
    },
    OpenMiss {
        id: "MISS-32",
        what: "cp.async.bulk.wait_group.read modelled as a full wait",
        explorer_only: false,
        trigger: |m| {
            first_inst(m, |_, t| {
                opcode(t).starts_with("cp.async.bulk.wait_group.read")
            })
        },
    },
    OpenMiss {
        id: "MISS-34",
        what: "cp.async / LDGSTS shared write checked with the issuer's issue-time clock",
        explorer_only: false,
        trigger: |m| {
            first_inst(m, |_, t| {
                let o = opcode(t);
                o.starts_with("cp.async.ca")
                    || o.starts_with("cp.async.cg")
                    || o.starts_with("ldgsts")
            })
        },
    },
    OpenMiss {
        id: "MISS-37",
        what: "PTX: a partial-exit warp reaching an .aligned barrier reported clean",
        explorer_only: false,
        trigger: |m| {
            let aligned = |i: &ptxv_ir::Inst, t: &str| {
                let o = opcode(t);
                is_ptx(i)
                    && (o.starts_with("bar.sync")
                        || o.starts_with("bar.cta.sync")
                        || o.starts_with("bar.red")
                        || o.starts_with("bar.cta.red")
                        || o.starts_with("bar.arrive")
                        || o.starts_with("bar.cta.arrive")
                        || ((o.starts_with("barrier.") || o.starts_with("bar.warp"))
                            && o.contains("aligned")))
            };
            let exit = |i: &ptxv_ir::Inst, t: &str| {
                is_ptx(i) && matches!(opcode(t), "ret" | "exit" | "ret.uni" | "exit.uni")
            };
            let code = &m.k.code;
            let last_bar = code
                .iter()
                .rposition(|i| aligned(i, &i.text.to_ascii_lowercase()))?;
            code.iter()
                .enumerate()
                .position(|(pc, i)| {
                    let t = i.text.to_ascii_lowercase();
                    exit(i, &t) && (pc < last_bar || t.trim_start().starts_with('@'))
                })
                .map(|pc| Some(pc as u32))
        },
    },
    OpenMiss {
        id: "MISS-21",
        what: "ld.global.nc modelled as coherent",
        explorer_only: false,
        trigger: |m| {
            first_inst(m, |_, t| {
                let o = opcode(t);
                (o.starts_with("ld.") && o.split('.').any(|x| x == "nc"))
                    || o.starts_with("ldg.e.constant")
                    || (o.starts_with("ldg") && o.split('.').any(|x| x == "constant"))
            })
        },
    },
    OpenMiss {
        id: "MISS-22",
        what: "cvt .sat / .relu / .satfinite dropped on f32 to f16 / bf16",
        explorer_only: false,
        trigger: |m| {
            first_inst(m, |_, t| {
                let o = opcode(t);
                o.starts_with("cvt.")
                    && o.split('.')
                        .any(|x| matches!(x, "sat" | "relu" | "satfinite"))
                    && o.split('.')
                        .any(|x| matches!(x, "f16" | "bf16" | "f16x2" | "bf16x2"))
            })
        },
    },
    OpenMiss {
        id: "MISS-23",
        what: "packed and saturating integer operations (16x2 borrow, add.sat, carry chains)",
        explorer_only: false,
        trigger: |m| {
            first_inst(m, |_, t| {
                let o = opcode(t);
                let parts: Vec<&str> = o.split('.').collect();
                let int = parts.iter().any(|x| {
                    matches!(
                        *x,
                        "s8" | "s16"
                            | "s32"
                            | "s64"
                            | "u8"
                            | "u16"
                            | "u32"
                            | "u64"
                            | "s16x2"
                            | "u16x2"
                    )
                });
                (matches!(parts[0], "add" | "sub" | "addc" | "subc" | "mad" | "madc")
                    && (parts.contains(&"cc")
                        || parts[0].ends_with('c')
                        || (int && parts.contains(&"sat"))
                        || parts.iter().any(|x| matches!(*x, "s16x2" | "u16x2"))))
                    || (matches!(parts[0], "min" | "max" | "neg" | "abs")
                        && parts.iter().any(|x| matches!(*x, "s16x2" | "u16x2")))
            })
        },
    },
    OpenMiss {
        id: "MISS-24",
        what: "fma contraction of mul.sat drops the clamp",
        explorer_only: false,
        trigger: |m| {
            first_inst(m, |_, t| {
                let o = opcode(t);
                o.starts_with("mul.") && o.split('.').any(|x| x == "sat")
            })
        },
    },
    OpenMiss {
        id: "MISS-25",
        what: "multimem .acc::f16 treated as f32",
        explorer_only: false,
        trigger: |m| {
            first_inst(m, |_, t| {
                opcode(t).starts_with("multimem.") && t.contains("acc::f16")
            })
        },
    },
    OpenMiss {
        id: "MISS-26",
        what: "proxy alias fence not required for virtual aliases (multicast objects, peer \
               mappings of one buffer on several GPUs)",
        explorer_only: false,
        trigger: |m| {
            let mut devs: Vec<u32> = m.launch.grids.iter().map(|g| g.device).collect();
            devs.sort_unstable();
            devs.dedup();
            (!m.launch.mcast.is_empty() || devs.len() > 1).then_some(None)
        },
    },
    OpenMiss {
        id: "MISS-27",
        what: "f32x2 mul + add contraction modelled for one ptxas version only",
        explorer_only: false,
        trigger: |m| {
            first_inst(m, |_, t| {
                let o = opcode(t);
                o.starts_with("mul.") && o.split('.').any(|x| x == "f32x2")
            })
        },
    },
];

/// The record's `exhaustive_concolic`.
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    /// `certified`, `blocked` or `refuted`.
    pub status: String,
    /// What the claim covers.
    pub scope: String,
    /// Regions closed by another traversed path / proven infeasible; operands proven constant.
    pub closed_by_path: usize,
    pub closed_infeasible: usize,
    /// Regions whose decision repeats an earlier atom of the path (the same expression, or
    /// its negation): infeasible without a solver.
    pub closed_repeat: usize,
    pub operands_constant: usize,
    /// Operands of paths after the first not sent to the solvers because the expansion was
    /// already blocked (path cap, deadline, an unfinished run): no certificate either way.
    pub operands_skipped: usize,
    pub blockers: Vec<Blocker>,
    pub paths: Vec<PathInfo>,
    pub solver_stats: certsmt::Stats,
    /// The checks the certificate rests on: per path, classes and assumptions (union).
    pub check_assumes: Vec<String>,
    pub replays: usize,
    /// `refuted` only: some refuting path's input lies within the declared invariants. False:
    /// every counterexample is outside them (a value the producer may not generate; per-buffer
    /// ranges only, relations between buffers are not declared).
    pub refuted_within_invariants: bool,
    pub seconds: f64,
}

/// A decision of a path: thread, pc, kind and the occurrence of (thread, pc) on the path.
type Key = (u32, u32, u8, u32);

struct Path {
    writes: Writes,
    steps: Vec<(Key, bool, E)>,
    sym: Box<SymState>,
    mfin: Machine,
}

fn kind_id(k: AtomKind) -> u8 {
    match k {
        AtomKind::Site => 0,
        AtomKind::Guard => 1,
        AtomKind::Select => 2,
        AtomKind::DataGuard => 3,
    }
}

/// The steps of a recorded run: every thread's control atoms (guards and branch
/// conditions) in observation order. Comparisons and selects are data: they reach control
/// only through a guard or branch condition (an atom of its own, with the same expression)
/// and addresses only through an operand (`SymState::operands`).
fn steps_of(sym: &SymState) -> Vec<(Key, bool, E)> {
    // ASSUME(C-EXH-CONTROL)
    let mut all: Vec<(u64, u32, u32, u8, bool, E)> = Vec::new();
    for (&t, atoms) in &sym.path {
        for a in atoms.iter().filter(|a| a.kind == AtomKind::Guard) {
            all.push((a.seq, t, a.pc, kind_id(a.kind), a.val, a.cond));
        }
    }
    all.sort_unstable_by_key(|x| x.0);
    let mut occ: HashMap<(u32, u32, u8), u32> = HashMap::new();
    all.into_iter()
        .map(|(_, t, pc, k, v, c)| {
            let o = occ.entry((t, pc, k)).or_default();
            *o += 1;
            ((t, pc, k, *o - 1), v, c)
        })
        .collect()
}

fn label(m: &Machine, w: &Writes) -> String {
    if w.is_empty() {
        return "captured input".into();
    }
    w.iter()
        .map(|&(a, off, b, v)| {
            let name = m
                .launch
                .allocs
                .get(a)
                .map_or("?".to_string(), |x| x.name.clone());
            format!("{name}[byte {off:#x}:{b}]={v:#x}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Merge `extra` over `base` (later writes win on the same bytes).
fn merge(base: &Writes, extra: &Writes) -> Writes {
    let mut out: Vec<(usize, u64, u8, u64)> = base
        .iter()
        .copied()
        .filter(|&(a, off, b, _)| {
            !extra
                .iter()
                .any(|&(a2, o2, b2, _)| a == a2 && off < o2 + b2 as u64 && o2 < off + b as u64)
        })
        .collect();
    out.extend(extra.iter().copied());
    out.sort_unstable();
    out
}

/// An atom with its negations folded: `(not x) == v` is `x == !v`.
fn norm(sym: &SymState, mut c: E, mut v: bool) -> (E, bool) {
    while let crate::symb::Node::Un {
        op: crate::symb::Un::Not,
        a,
        ..
    } = sym.ar.nodes[c as usize]
    {
        c = a;
        v = !v;
    }
    (c, v)
}

/// The expansion of `m0` (the captured launch). `check` runs the full ptxv check of a
/// representative input (the caller's verdict pipeline).
pub fn expand(
    m0: &Machine,
    dom: &Domain,
    opts: &Options,
    check: &mut dyn FnMut(&Machine) -> RepCheck,
) -> Report {
    let start = std::time::Instant::now();
    let solvers = opts.solvers.clone().unwrap_or_else(certsmt::solvers);
    let mut rep = Report {
        status: "blocked".into(),
        scope: format!(
            "kernel {} with this shape, these parameters and this launch configuration; every value of the {} input buffer(s) (floats: NaN, infinities and subnormals included); verdict classes race, deadlock, violation, fault, contract, unmodeled; value-dependent parts (same-value writes, output determinism, the GPU comparison, numerics) excluded",
            m0.k.name,
            m0.launch.allocs.len()
        ),
        closed_by_path: 0,
        closed_infeasible: 0,
        closed_repeat: 0,
        operands_constant: 0,
        operands_skipped: 0,
        blockers: vec![],
        paths: vec![],
        solver_stats: certsmt::Stats {
            versions: solvers
                .iter()
                .map(|s| (s.name.clone(), s.version.clone()))
                .collect(),
            ..Default::default()
        },
        check_assumes: vec![],
        replays: 0,
        refuted_within_invariants: false,
        seconds: 0.0,
    };
    let block = |rep: &mut Report, kind: &str, pc: Option<u32>, detail: String| {
        let at = pc.map_or(String::new(), |p| m0.k.describe(p as usize));
        if rep.blockers.iter().any(|b| {
            b.kind == kind && b.pc == pc && (kind != "pending-soundness-fix" || b.detail == detail)
        }) {
            return;
        }
        rep.blockers.push(Blocker {
            kind: kind.into(),
            pc,
            at,
            detail,
        });
    };
    if solvers.len() < 2 {
        block(
            &mut rep,
            "no-solver",
            None,
            format!(
                "{} SMT solver(s) found; infeasibility needs two (PTXV_EXH_SOLVERS)",
                solvers.len()
            ),
        );
    }
    // Clock and global-timer reads: the fronts read them as 0 (E-CLOCK), so the expressions
    // see one constant where the hardware has a fresh value per read; a decision on two reads
    // would be invisible. Blocked (no fresh leaf per read is possible from the IR).
    if let Some(pc) = first_inst(m0, |_, t| {
        t.contains("%clock")
            || t.contains("%globaltimer")
            || t.contains("sr_clock")
            || t.contains("sr_globaltimer")
    }) {
        block(
            &mut rep,
            "clock-read",
            pc,
            "a clock or global-timer read: modelled as 0 (E-CLOCK), not a fresh value per read"
                .into(),
        );
    }
    // Known misses of the per-path check (`OPEN_MISSES`): a launch one can apply to is not
    // certified until the fix lands.
    for miss in OPEN_MISSES.iter().filter(|x| !x.explorer_only) {
        if let Some(pc) = (miss.trigger)(m0) {
            block(
                &mut rep,
                "pending-soundness-fix",
                pc,
                format!("{}: {}", miss.id, miss.what),
            );
        }
    }
    // ASSUME(C-EXH-LAUNCH): every byte write of the launch goes through the shadow (a NIC
    // actor's do not).
    if m0.nic.is_some() {
        block(
            &mut rep,
            "nic-writer",
            None,
            "a NIC actor writes memory outside the shadow".into(),
        );
    }
    let mut paths: Vec<Path> = Vec::new();
    let mut frontier: Vec<Writes> = vec![vec![]];
    let mut tried: HashMap<(usize, usize), usize> = HashMap::new();
    let mut closed: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    let budget_left = |start: std::time::Instant| opts.budget - start.elapsed().as_secs_f64();
    let cfg = crate::symb::Config::default();
    'outer: loop {
        // Traverse the pending inputs.
        while let Some(w) = frontier.pop() {
            if budget_left(start) <= 0.0 {
                block(
                    &mut rep,
                    "deadline",
                    None,
                    format!("the {:.0} s budget is spent", opts.budget),
                );
                break 'outer;
            }
            let mm = crate::coverage::apply(
                m0,
                &Mutation {
                    pc: u32::MAX,
                    never: false,
                    writes: w.clone(),
                    label: String::new(),
                    within_invariants: false,
                },
            );
            rep.replays += 1;
            let b = Budget {
                seconds: Some(opts.run_seconds.min(budget_left(start).max(1.0))),
                rss: None,
            };
            let (cov, s) = crate::coverage::analyze_sym(&mm, b, Some(cfg.clone()));
            let Some((sym, mfin)) = s else { continue };
            if cov.stopped.is_some() || !cov.terminal {
                block(
                    &mut rep,
                    "coverage-incomplete",
                    None,
                    format!(
                        "the coverage run of {} did not finish ({})",
                        label(m0, &w),
                        cov.stopped.clone().unwrap_or("not terminal".into())
                    ),
                );
                break 'outer;
            }
            if let Some(&t) = sym.truncated.iter().next() {
                block(
                    &mut rep,
                    "path-truncated",
                    None,
                    format!("thread {t}'s path passed {} atoms", cfg.max_atoms),
                );
                break 'outer;
            }
            // Every decision the provenance calls input-dependent must be one the expressions
            // see (else the path would miss it).
            let atom_pcs = &sym.atom_pcs;
            let opnd_pcs: std::collections::HashSet<u32> =
                sym.operands.iter().map(|o| o.pc).collect();
            // ASSUME(C-EXH-DEP)
            for c in &cov.comparisons {
                if !atom_pcs.contains(&c.pc) {
                    block(
                        &mut rep,
                        "provenance-only",
                        Some(c.pc),
                        "an input-dependent comparison by the provenance with no expression atom"
                            .into(),
                    );
                }
            }
            for d in &cov.input_dependent {
                if !atom_pcs.contains(&d.pc) && !opnd_pcs.contains(&d.pc) {
                    block(
                        &mut rep,
                        "provenance-only",
                        Some(d.pc),
                        format!(
                            "an input-dependent {} by the provenance with no expression",
                            d.what
                        ),
                    );
                }
            }
            let steps = steps_of(&sym);
            if paths.iter().any(|p| {
                p.steps.len() == steps.len()
                    && p.steps
                        .iter()
                        .zip(&steps)
                        .all(|(a, b)| a.0 == b.0 && a.1 == b.1)
            }) {
                continue;
            }
            if paths.len() >= opts.max_paths {
                block(
                    &mut rep,
                    "path-cap",
                    None,
                    format!("more than {} feasible paths", opts.max_paths),
                );
                break 'outer;
            }
            if std::env::var_os("PTXV_EXH_PROGRESS").is_some() && paths.is_empty() {
                let mut by: BTreeMap<u32, usize> = BTreeMap::new();
                for s in &steps {
                    *by.entry(s.0.1).or_default() += 1;
                }
                let mut v: Vec<(u32, usize)> = by.into_iter().collect();
                v.sort_by_key(|x| std::cmp::Reverse(x.1));
                for (pc, n) in v.iter().take(12) {
                    eprintln!(
                        "exhaustive concolic:   {n} decisions at {}",
                        m0.k.describe(*pc as usize)
                    );
                }
                let mut ob: BTreeMap<u32, usize> = BTreeMap::new();
                for o in &sym.operands {
                    *ob.entry(o.pc).or_default() += 1;
                }
                let mut v: Vec<(u32, usize)> = ob.into_iter().collect();
                v.sort_by_key(|x| std::cmp::Reverse(x.1));
                for (pc, n) in v.iter().take(8) {
                    eprintln!(
                        "exhaustive concolic:   {n} operands at {}",
                        m0.k.describe(*pc as usize)
                    );
                }
            }
            eprintln!(
                "exhaustive concolic: path {} ({}): {} decisions, {} operands, {:.1}s",
                paths.len(),
                label(m0, &w),
                steps.len(),
                sym.operands.len(),
                start.elapsed().as_secs_f64()
            );
            paths.push(Path {
                writes: w,
                steps,
                sym,
                mfin,
            });
        }
        // Close the regions of every traversed path.
        let mut new_inputs: Vec<Writes> = Vec::new();
        for pi in 0..paths.len() {
            if budget_left(start) <= 0.0 {
                block(
                    &mut rep,
                    "deadline",
                    None,
                    format!("the {:.0} s budget is spent", opts.budget),
                );
                break 'outer;
            }
            let mut pending: Vec<usize> = Vec::new();
            // Atoms seen so far on the path (normalized through negation): a decision that
            // repeats one has no other side under the prefix.
            let mut seen: HashMap<E, bool> = HashMap::new();
            for i in 0..paths[pi].steps.len() {
                let (c, v) = norm(&paths[pi].sym, paths[pi].steps[i].2, paths[pi].steps[i].1);
                let repeat = seen.get(&c) == Some(&v);
                seen.entry(c).or_insert(v);
                if closed.contains(&(pi, i)) {
                    continue;
                }
                if repeat {
                    closed.insert((pi, i));
                    rep.closed_repeat += 1;
                    continue;
                }
                let (key, val, _) = paths[pi].steps[i];
                // Another traversed path with the same prefix and the other value here?
                let mut by_path = false;
                for (qi, q) in paths.iter().enumerate() {
                    if qi == pi || q.steps.len() <= i {
                        continue;
                    }
                    if !q.steps[..i]
                        .iter()
                        .zip(&paths[pi].steps[..i])
                        .all(|(a, b)| a.0 == b.0 && a.1 == b.1)
                    {
                        continue;
                    }
                    if q.steps[i].0 != key {
                        // ASSUME(C-EXH-PATH): the same atoms before i make the same decision i.
                        block(&mut rep, "path-nondeterminism", Some(key.1), "two inputs agree on every decision before one but reach different decisions there".into());
                        continue;
                    }
                    if q.steps[i].1 != val {
                        by_path = true;
                        break;
                    }
                }
                if by_path {
                    closed.insert((pi, i));
                    rep.closed_by_path += 1;
                } else {
                    pending.push(i);
                }
            }
            eprintln!(
                "exhaustive concolic: path {pi}: {} region(s) to solve ({} closed by a path, {} repeats so far), {:.1}s",
                pending.len(),
                rep.closed_by_path,
                rep.closed_repeat,
                start.elapsed().as_secs_f64()
            );
            if pending.is_empty() {
                continue;
            }
            let p = &paths[pi];
            let roots: Vec<E> = p.steps.iter().map(|s| s.2).collect();
            let steps: Vec<(E, bool)> = p.steps.iter().map(|s| (s.2, s.1)).collect();
            // In chunks, so that a path explosion stops the expansion early.
            for chunk in pending.chunks(256) {
                if budget_left(start) <= 0.0 || paths.len() + new_inputs.len() > opts.max_paths {
                    break;
                }
                // Region i: the path's decisions before i (asserted incrementally along the
                // path), decision i the other way.
                let items: Vec<certsmt::Query> = chunk
                    .iter()
                    .map(|&i| (vec![(p.steps[i].2, !p.steps[i].1)], None))
                    .collect();
                // ASSUME(C-EXH-UNSAT)
                let vs = certsmt::decide_along(
                    &solvers,
                    &p.sym.ar,
                    &roots,
                    &steps,
                    chunk,
                    &items,
                    opts.solver_timeout,
                );
                for (k, &i) in chunk.iter().enumerate() {
                    let v = &vs[k];
                    rep.solver_stats.add(v);
                    if v.proven_unsat() {
                        closed.insert((pi, i));
                        rep.closed_infeasible += 1;
                        continue;
                    }
                    let (key, val, cond) = p.steps[i];
                    if std::env::var_os("PTXV_EXH_DEBUG").is_some() {
                        let show = |e: E| p.sym.ar.show(e, &|a| format!("a{a}"), 5);
                        eprintln!(
                            "exh debug: path {pi} step {i} key {key:?} val {val} cond {} answers {:?}",
                            show(cond),
                            v.answers
                        );
                    }
                    let n = tried.entry((pi, i)).or_default();
                    if *n >= opts.tries {
                        let why = if v.any_sat() {
                            "satisfiable on the expressions, no replayed input takes it"
                        } else {
                            "the solvers could not decide"
                        };
                        block(
                            &mut rep,
                            "open-region",
                            Some(key.1),
                            format!("thread {} decision {} the other way: {why}", key.0, key.3),
                        );
                        continue;
                    }
                    *n += 1;
                    let prefix: Vec<(E, bool)> = p.steps[..i].iter().map(|s| (s.2, s.1)).collect();
                    let ws = crate::solve::witnesses(
                        m0,
                        &p.sym,
                        &p.mfin,
                        dom,
                        &opts.solve,
                        cond,
                        !val,
                        &prefix,
                        2,
                    );
                    if ws.is_empty() {
                        *n = opts.tries;
                        let why = if v.any_sat() {
                            "satisfiable on the expressions, no witness found"
                        } else {
                            "the solvers could not decide"
                        };
                        block(
                            &mut rep,
                            "open-region",
                            Some(key.1),
                            format!("thread {} decision {} the other way: {why}", key.0, key.3),
                        );
                    }
                    for w in ws {
                        let w = merge(&p.writes, &w);
                        if !new_inputs.contains(&w) {
                            new_inputs.push(w);
                        }
                    }
                }
            }
            if new_inputs.len() + paths.len() > opts.max_paths * 4 {
                break;
            }
        }
        if new_inputs.is_empty() {
            break;
        }
        frontier.extend(new_inputs);
    }
    // Operands: constant under each path's atoms. Once a blocker that no operand proof can
    // lift is in, only the first path's operands are tried (they name the operand blockers).
    let hard = |rep: &Report| {
        rep.blockers.iter().any(|b| {
            matches!(
                b.kind.as_str(),
                "path-cap" | "deadline" | "coverage-incomplete" | "path-truncated" | "no-solver"
            )
        })
    };
    for (pi, p) in paths.iter().enumerate() {
        if p.sym.operands.is_empty() {
            continue;
        }
        if pi > 0 && hard(&rep) {
            rep.operands_skipped += p.sym.operands.len();
            continue;
        }
        if budget_left(start) <= 0.0 {
            block(
                &mut rep,
                "deadline",
                None,
                format!("the {:.0} s budget is spent", opts.budget),
            );
            break;
        }
        let prefix: Vec<(E, bool)> = p.steps.iter().map(|s| (s.2, s.1)).collect();
        // One query per distinct (expression, value); one per instruction first, so that a
        // blocked expansion still names the operand blockers per instruction.
        let mut seen_ev = std::collections::HashSet::new();
        let mut uniq: Vec<&crate::symb::Operand> = p
            .sym
            .operands
            .iter()
            .filter(|o| seen_ev.insert((o.e, o.val)))
            .collect();
        let mut seen_pc = std::collections::HashSet::new();
        uniq.sort_by_key(|o| !seen_pc.insert(o.pc));
        let blocked = hard(&rep);
        for (ci, chunk) in uniq.chunks(256).enumerate() {
            if budget_left(start) <= 0.0 {
                block(
                    &mut rep,
                    "deadline",
                    None,
                    format!("the {:.0} s budget is spent", opts.budget),
                );
                break;
            }
            if blocked && ci > 0 {
                rep.operands_skipped += chunk.len();
                continue;
            }
            let items: Vec<certsmt::Query> =
                chunk.iter().map(|o| (vec![], Some((o.e, o.val)))).collect();
            let mut roots: Vec<E> = prefix.iter().map(|x| x.0).collect();
            roots.extend(chunk.iter().map(|o| o.e));
            let ks = vec![prefix.len(); items.len()];
            // ASSUME(C-EXH-TRANSFER)
            let vs = certsmt::decide_along(
                &solvers,
                &p.sym.ar,
                &roots,
                &prefix,
                &ks,
                &items,
                opts.solver_timeout,
            );
            for (o, v) in chunk.iter().zip(&vs) {
                rep.solver_stats.add(v);
                if v.proven_unsat() {
                    rep.operands_constant += 1;
                } else if v.any_sat() {
                    block(
                        &mut rep,
                        "operand-varies",
                        Some(o.pc),
                        format!(
                            "thread {}'s operand can take another value on the same path",
                            o.t
                        ),
                    );
                } else {
                    block(
                        &mut rep,
                        "operand-unproven",
                        Some(o.pc),
                        format!("thread {}'s operand: the solvers could not decide", o.t),
                    );
                }
            }
        }
    }
    // The representatives' checks.
    let mut assumes: BTreeMap<String, ()> = BTreeMap::new();
    let mut refuted = false;
    for p in &paths {
        let mut info = PathInfo {
            input: label(m0, &p.writes),
            atoms: p.steps.len(),
            operands: p.sym.operands.len(),
            within_invariants: p.writes.is_empty() || dom.within(m0, &p.writes),
            check: None,
        };
        if budget_left(start) <= 0.0 && !p.writes.is_empty() {
            block(
                &mut rep,
                "deadline",
                None,
                format!(
                    "the {:.0} s budget is spent before every path's check",
                    opts.budget
                ),
            );
            rep.paths.push(info);
            continue;
        }
        let mm = crate::coverage::apply(
            m0,
            &Mutation {
                pc: u32::MAX,
                never: false,
                writes: p.writes.clone(),
                label: String::new(),
                within_invariants: false,
            },
        );
        let c = check(&mm);
        for a in &c.assumes {
            assumes.insert(a.clone(), ());
        }
        if c.refutes() {
            refuted = true;
            rep.refuted_within_invariants |= info.within_invariants;
        } else if !c.clean() {
            block(
                &mut rep,
                "check-incomplete",
                None,
                format!(
                    "the check of {} is {:?} (complete {})",
                    info.input, c.classes, c.complete
                ),
            );
        }
        info.check = Some(c);
        rep.paths.push(info);
    }
    // Misses of explored verdicts only: they apply unless every representative's check is a
    // certificate.
    if rep
        .paths
        .iter()
        .any(|p| !p.check.as_ref().is_some_and(|c| c.certified))
    {
        for miss in OPEN_MISSES.iter().filter(|x| x.explorer_only) {
            if let Some(pc) = (miss.trigger)(m0) {
                block(
                    &mut rep,
                    "pending-soundness-fix",
                    pc,
                    format!("{}: {}", miss.id, miss.what),
                );
            }
        }
    }
    rep.check_assumes = assumes.into_keys().collect();
    rep.status = if refuted {
        "refuted".into()
    } else if rep.blockers.is_empty() {
        "certified".into()
    } else {
        "blocked".into()
    };
    rep.seconds = start.elapsed().as_secs_f64();
    rep
}

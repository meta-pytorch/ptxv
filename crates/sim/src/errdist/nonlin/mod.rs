/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Non-chain computations in errdist (docs/errdist.md §9): softmax (two-pass and online),
//! RMSNorm / layer norm, attention, and elementwise ops built from MUFU slices and type
//! conversions.
//!
//! - `ops`: the program ops beyond accumulation chains (slices, MUFU, maxima), bit-exact;
//! - `ident`: the real function each instruction slice computes;
//! - `real`: the exact value (the algorithm in real arithmetic) of any output;

pub mod agg;
pub mod enumerate;
pub mod ident;
pub mod lin;
pub mod ops;
pub mod real;
pub mod table;

use std::collections::BTreeMap;
use std::collections::HashMap;

use ops::NlOp;
use table::Table;

use super::chain::Op;
use super::chain::Prog;
use super::report::Estimate;
use super::spec::Format;
use super::spec::Layout;
use super::spec::Spec;
use crate::machine::Machine;

/// The compiled local-error model of one kernel or pair: the tables of every slice function
/// (by `ident::func_key`) and MUFU variant the programs use.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Tables {
    pub tables: Vec<Table>,
    pub by_key: BTreeMap<String, u32>,
    /// Pairs of slice tables (lower index first) computing the same function of their index
    /// value (`table::same_function`).
    #[serde(default)]
    pub same: Vec<(u32, u32)>,
    /// Joint tables of slice pairs of the same real function (a < b: the law of g_b - g_a).
    #[serde(default)]
    pub pairs: Vec<(u32, u32, Table)>,
    pub seconds: f64,
}

fn mufu_key(n: &NlOp) -> Option<String> {
    match n {
        NlOp::Mufu {
            func,
            ptx,
            neg,
            abs,
            ..
        } => Some(format!("mufu {func:?} ptx={ptx} neg={neg} abs={abs}")),
        _ => None,
    }
}

/// Build the tables of the programs' slices (from their identification probes) and MUFU ops
/// (over the operand ranges of `nominal` sampled inputs).
pub fn compile(
    progs: &[(&Prog, &ident::Idents)],
    spec: &Spec,
    layouts: &[Layout],
) -> Result<Tables, String> {
    let t0 = std::time::Instant::now();
    let mut out = Tables::default();
    for (_, ids) in progs {
        for (k, info) in ids.iter() {
            if out.by_key.contains_key(k) {
                continue;
            }
            if let Ok(f) = &info.result {
                let t = table::slice_table(
                    &format!("slice @{} {}", info.pc, f.show()),
                    &info.prog,
                    f,
                    &info.probes,
                );
                out.by_key.insert(k.clone(), out.tables.len() as u32);
                out.tables.push(t);
            }
        }
    }
    // Slices of different kernels computing the same function of their index value.
    let mut infos: Vec<(u32, &ident::IdentInfo)> = Vec::new();
    for (_, ids) in progs {
        for (k, info) in ids.iter() {
            if let (Some(&ti), Ok(_)) = (out.by_key.get(k), &info.result) {
                if !infos.iter().any(|x| x.0 == ti) {
                    infos.push((ti, info));
                }
            }
        }
    }
    for a in 0..infos.len() {
        for b in a + 1..infos.len() {
            let (ta, ia) = infos[a];
            let (tb, ib) = infos[b];
            let (Ok(ra), Ok(rb)) = (&ia.result, &ib.result) else {
                continue;
            };
            if table::same_function(&ia.prog, ra, &ia.probes, &ib.prog, rb, &ib.probes) {
                out.same.push((ta.min(tb), ta.max(tb)));
            }
            let range = out.tables[ta as usize].range.clone();
            if let Some(pt) =
                table::pair_table(&ia.prog, ra, &ia.probes, &ib.prog, rb, &ib.probes, &range)
            {
                out.pairs.push((ta, tb, pt));
            }
        }
    }
    // MUFU ops: operand ranges from a few sampled inputs.
    // name -> (op, per sign: binade range)
    type Ranges = HashMap<String, (NlOp, HashMap<i8, (i16, i16)>)>;
    let mut ranges: Ranges = HashMap::new();
    for (p, _) in progs {
        if !p.ops.iter().any(|o| matches!(o, Op::Nl(NlOp::Mufu { .. }))) {
            continue;
        }
        for t in 0..8u64 {
            let smp = super::spec::sample(spec, layouts, (1u64 << 43) + t)?;
            let w = p.leaves(&smp.bytes)?;
            let rv = real::eval(p, &w);
            for op in &p.ops {
                if let Op::Nl(n @ NlOp::Mufu { arg, .. }) = op {
                    let x = rv[*arg as usize].0.v();
                    let (sg, e, _) = table::cell_of(x);
                    if sg == 0 {
                        continue;
                    }
                    let k = mufu_key(n).unwrap();
                    let ent = ranges
                        .entry(k)
                        .or_insert_with(|| (n.clone(), HashMap::new()));
                    let r = ent.1.entry(sg).or_insert((e, e));
                    r.0 = r.0.min(e);
                    r.1 = r.1.max(e);
                }
            }
        }
    }
    for (k, (n, r)) in ranges {
        if let NlOp::Mufu {
            func,
            ptx,
            neg,
            abs,
            ..
        } = n
        {
            let lo_hi: Vec<(i8, i16, i16)> = r
                .into_iter()
                .map(|(s, (lo, hi))| (s, (lo - 4).max(hi - 60), hi + 4))
                .collect();
            let t = table::mufu_table(&k, func, ptx, neg, abs, &lo_hi);
            out.by_key.insert(k, out.tables.len() as u32);
            out.tables.push(t);
        }
    }
    out.seconds = t0.elapsed().as_secs_f64();
    Ok(out)
}

/// Per op of a program: its table index.
pub fn op_tables(p: &Prog, t: &Tables) -> Vec<Option<u32>> {
    p.ops
        .iter()
        .map(|op| match op {
            Op::Nl(NlOp::Seg {
                prog,
                real: Some(_),
                ..
            }) => t.by_key.get(&ident::func_key(prog)).copied(),
            Op::Nl(n @ NlOp::Mufu { .. }) => t.by_key.get(&mufu_key(n).unwrap()).copied(),
            _ => None,
        })
        .collect()
}

/// Tiers 1 and 2 of a non-chain single kernel or pair.
pub fn analytic(
    r: &Prog,
    c: Option<&Prog>,
    tabs: &Tables,
    spec: &Spec,
    layouts: &[Layout],
    fmt: Format,
) -> (Vec<Estimate>, Vec<String>) {
    let mut out = Vec::new();
    let mut notes = Vec::new();
    if let Some(c) = c {
        let (same, n) = super::tier0::identical_outputs(r, c);
        if same == n && n > 0 {
            let mut e = super::tier0::certified(fmt, true, &[]);
            e.outputs = n as u64;
            e.method = format!(
                "identical programs: all {n} outputs have equal canonical DAGs (slices compared by function), so C = R up to the sign of a zero result"
            );
            out.push(e);
            notes.push("tiers 1-2 not run: the kernels are identical programs (tier 0)".into());
            return (out, notes);
        }
    }
    match enumerate::elementwise(r, c, spec) {
        Some(Ok(e)) => out.push(e),
        Some(Err(e)) => notes.push(format!("elementwise enumeration not run: {e}")),
        None => {}
    }
    let mut ts = tabs.tables.clone();
    for t in ts.iter_mut() {
        t.reindex();
    }
    let pairs: Vec<(u32, u32, Table)> = tabs
        .pairs
        .iter()
        .map(|(a, b, t)| {
            let mut t = t.clone();
            t.reindex();
            (*a, *b, t)
        })
        .collect();
    if std::iter::once(r).chain(c).any(|p| {
        p.ops
            .iter()
            .any(|o| matches!(o, super::chain::Op::Tf32 { .. }))
    }) {
        notes.push(
            "linearized tiers not run: TF32 splits (the chain tiers' DAG CLT models them)".into(),
        );
        return (out, notes);
    }
    let kr = lin::KModel::new(r, &ts, op_tables(r, tabs)).with_input_errors(spec);
    let kc = c.map(|c| lin::KModel::new(c, &ts, op_tables(c, tabs)).with_input_errors(spec));
    if std::env::var("PTXV_NL_DIAG").is_ok() {
        eprintln!("reference:\n{}", diagnose(&kr, spec, layouts, 8));
        if let Some(k) = &kc {
            eprintln!("candidate:\n{}", diagnose(k, spec, layouts, 8));
        }
    }
    for tier in [2u8, 1] {
        let cfg = agg::Cfg {
            tier,
            ..Default::default()
        };
        match agg::estimate(
            &cfg,
            &kr,
            kc.as_ref(),
            spec,
            layouts,
            fmt,
            &tabs.same,
            &pairs,
        ) {
            Ok((e, n)) => {
                if tier == 1 {
                    notes.extend(n);
                }
                out.push(e);
            }
            Err(e) => notes.push(format!("non-chain tier {tier} not run: {e}")),
        }
    }
    (out, notes)
}

/// After extraction: identify the program's slices on the captured input (and perturbed
/// ones). Returns one note per distinct slice.
pub fn prepare(p: &mut Prog, m: &Machine) -> Result<(Vec<String>, ident::Idents), String> {
    let bytes: std::collections::BTreeMap<String, Vec<u8>> = m
        .launch
        .allocs
        .iter()
        .map(|a| (a.name.clone(), a.init.clone()))
        .collect();
    let base = p.leaves(&bytes)?;
    let ids = ident::identify_prog(p, &base, 24, 1);
    let mut notes = Vec::new();
    for info in ids.values() {
        let (pc, n) = (info.pc, info.count);
        notes.push(match &info.result {
            Ok(f) => format!(
                "slice @{pc} (x{n}): {}{} [max rel dev {:.2e} on {} probes]",
                f.show(),
                if f.snapped {
                    ""
                } else {
                    " (constants fitted, not snapped)"
                },
                f.fit_err,
                f.probes
            ),
            Err(e) => format!("slice @{pc} (x{n}): not identified: {e}"),
        });
    }
    Ok((notes, ids))
}

/// Debug: the first ops whose evaluation at the captured input differs from the simulator.
pub fn mismatches(p: &Prog, m: &Machine, max: usize) -> Vec<String> {
    let bytes: std::collections::BTreeMap<String, Vec<u8>> = m
        .launch
        .allocs
        .iter()
        .map(|a| (a.name.clone(), a.init.clone()))
        .collect();
    let Ok(lv) = p.leaves(&bytes) else {
        return vec![];
    };
    let v = p.eval(&lv);
    let mut out = vec![];
    for (i, (a, b)) in v.iter().zip(&p.hw).enumerate() {
        if a.to_bits() != *b && !(a.is_nan() && f32::from_bits(*b).is_nan()) {
            let op = &p.ops[i];
            let args: Vec<String> = match op {
                super::chain::Op::Nl(n) => n
                    .args()
                    .iter()
                    .map(|&j| format!("{:e}/{:e}", v[j as usize], f32::from_bits(p.hw[j as usize])))
                    .collect(),
                _ => vec![],
            };
            let mut d = format!("{op:?}");
            if let super::chain::Op::Nl(super::nonlin::ops::NlOp::Seg { prog, args: sa, .. }) = op {
                let ab: Vec<u32> = sa.iter().map(|&j| p.hw[j as usize]).collect();
                d = String::new();
                for (k, o) in prog.ops.iter().enumerate() {
                    let t = match &o.kind {
                        crate::seg::SKind::Inst { op, regs, .. } => {
                            let n = format!("{op:?}");
                            format!(
                                "{} {:?}",
                                &n[..n.find(' ').unwrap_or(n.len())],
                                regs.iter().map(|x| x.1).collect::<Vec<_>>()
                            )
                        }
                        k => format!("{k:?}"),
                    };
                    // Value of the prefix ending at op k.
                    let mut q = (**prog).clone();
                    q.out = crate::seg::SVal::Op(k as u32);
                    q.ops.truncate(k + 1);
                    let v = q.eval(&ab).map(f32::from_bits).unwrap_or(f32::NAN);
                    d += &format!("\n    [{k}] pc {} {t} = {v:e}", o.pc);
                }
                d += &format!("\n    out {:?}", prog.out);
            }
            out.push(format!(
                "op {i}: eval {a:e} hw {:e}: {} args {args:?}",
                f32::from_bits(*b),
                &d[..d.len().min(20000)]
            ));
            if out.len() >= max {
                break;
            }
        }
    }
    out
}

/// Outputs (and ops) whose evaluation at the captured input differs from the simulator's
/// node value: (output mismatches, op mismatches).
pub fn captured_mismatch(p: &Prog, m: &Machine) -> Result<(usize, usize), String> {
    let bytes: std::collections::BTreeMap<String, Vec<u8>> = m
        .launch
        .allocs
        .iter()
        .map(|a| (a.name.clone(), a.init.clone()))
        .collect();
    let v = p.eval(&p.leaves(&bytes)?);
    let bad = |i: usize| {
        let (a, b) = (v[i], p.hw[i]);
        a.to_bits() != b && !(a.is_nan() && f32::from_bits(b).is_nan())
    };
    let ops = (0..v.len()).filter(|&i| bad(i)).count();
    let outs = p
        .outputs
        .iter()
        .flatten()
        .filter(|&&o| bad(o as usize))
        .count();
    Ok((outs, ops))
}

/// The headline estimate of a non-chain report: Tier 0 if present, else the analytic tier
/// (1, then 2) whose P(differ) agrees with Tier 3 within max(Tier 3's interval, 5% relative);
/// None: keep the default (Tier 3).
pub fn pick_best(rep: &super::report::Report) -> Option<usize> {
    if let Some(i) = rep.estimates.iter().position(|e| e.tier == 0) {
        return Some(i);
    }
    let t3 = rep.estimates.iter().find(|e| e.tier == 3)?;
    let (p3, lo, hi) = t3.p_differ?;
    // The elementwise law first, then the linearized tiers 1 and 2.
    let kinds: [(u8, &str); 3] = [(1, "elementwise"), (1, "linearized"), (2, "linearized")];
    for (tier, kind) in kinds {
        let Some((i, e)) = rep
            .estimates
            .iter()
            .enumerate()
            .find(|(_, e)| e.tier == tier && e.method.starts_with(kind))
        else {
            continue;
        };
        let p = e.p_differ?.0;
        let tol = (0.05 * p3).max(0.5 * (hi - lo));
        if (p - p3).abs() <= tol || (p >= lo && p <= hi) {
            return Some(i);
        }
    }
    None
}

/// Diagnostics (`PTXV_NL_DIAG`): per site class, the observed local errors at sampled inputs
/// (bit-exact evaluation; the error of each op against its exact operation on the computed
/// arguments) against the model's laws, in units of the f32 ulp of the op's value.
pub fn diagnose(k: &lin::KModel, spec: &Spec, layouts: &[Layout], n: usize) -> String {
    use std::fmt::Write;
    let mut tab: BTreeMap<String, [f64; 6]> = BTreeMap::new();
    for t in 0..n as u64 {
        let Ok(smp) = super::spec::sample(spec, layouts, (1u64 << 42) + t) else {
            continue;
        };
        let Ok(mut w) = k.p.leaves(&smp.bytes) else {
            continue;
        };
        let st = k.state(&w);
        k.p.eval_into(&mut w);
        let v = |i: u32| w[i as usize].v;
        for (i, op) in k.p.ops.iter().enumerate() {
            let exact = match op {
                Op::Add { a, b, .. } => v(*a) + v(*b),
                Op::Fma {
                    a,
                    b,
                    c,
                    neg_ab,
                    neg_c,
                } => {
                    let p = v(*a) * v(*b) * if *neg_ab { -1.0 } else { 1.0 };
                    p + c.map_or(0.0, |c| if *neg_c { -v(c) } else { v(c) })
                }
                Op::Round { arg, .. } => v(*arg),
                Op::Nl(NlOp::Seg {
                    args,
                    real: Some(f),
                    ..
                }) => {
                    let a: Vec<f64> = args.iter().map(|&j| v(j)).collect();
                    f.eval(&a)
                }
                _ => continue,
            };
            let q = super::mc::ulp_out(st.v[i], Format::F32);
            let d = (w[i].v - exact) / q;
            let (m, var) = st.law[i].moments(k.tabs);
            let cls = match op {
                Op::Add { .. } => "f32 add".to_string(),
                Op::Fma { c: None, .. } => "f32 mul".into(),
                Op::Fma { .. } => "f32 fma".into(),
                Op::Round { .. } => "16-bit rounding".into(),
                Op::Nl(NlOp::Seg { real: Some(f), .. }) => format!("slice {}", f.show()),
                _ => "other".into(),
            };
            let e = tab.entry(cls).or_default();
            e[0] += 1.0;
            e[1] += d;
            e[2] += d * d;
            e[3] += m / q;
            e[4] += var / (q * q) + (m / q).powi(2);
            e[5] += (d != 0.0) as u8 as f64;
        }
    }
    let mut s = String::new();
    let _ = writeln!(
        s,
        "site class: n  observed mean / rms (f32 ulps of the value)  model mean / rms  observed P(!=0)"
    );
    for (k, e) in tab {
        let n = e[0].max(1.0);
        let _ = writeln!(
            s,
            "  {k}: {} obs {:.4} / {:.4} model {:.4} / {:.4} P!=0 {:.3}",
            e[0],
            e[1] / n,
            (e[2] / n).sqrt(),
            e[3] / n,
            (e[4] / n).sqrt(),
            e[5] / n
        );
    }
    s
}

/// The persisted non-chain model ("compile once, query many"): the slice / MUFU tables of a
/// kernel or pair, keyed by the programs' canonical hash. It is the `nonlin` section of an
/// errdist model file (other sections, e.g. the chains', sit beside it). A query with an
/// input distribution whose index values leave the tabulated binades extends the tables
/// first (`ensure`; seconds per binade).
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Model {
    pub version: u32,
    /// `model_key` of the programs it was compiled for.
    pub key: String,
    pub tables: Tables,
}

pub const MODEL_VERSION: u32 = 1;

/// The canonical hash of a kernel (or pair) for the model: every output's canonical DAG.
pub fn model_key(progs: &[&Prog]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for p in progs {
        let mut memo = HashMap::new();
        for o in &p.outputs {
            let v = o.map_or(0, |o| super::tier0::canon(p, o, &mut memo));
            h = (h ^ v).wrapping_mul(0x100000001b3);
        }
        h = (h ^ 0xff).wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

impl Model {
    pub fn compile(
        progs: &[(&Prog, &ident::Idents)],
        spec: &Spec,
        layouts: &[Layout],
    ) -> Result<Model, String> {
        let ps: Vec<&Prog> = progs.iter().map(|x| x.0).collect();
        Ok(Model {
            version: MODEL_VERSION,
            key: model_key(&ps),
            tables: compile(progs, spec, layouts)?,
        })
    }
    /// Read the `nonlin` section of a model file.
    pub fn load(path: &std::path::Path) -> Result<Model, String> {
        let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&s).map_err(|e| e.to_string())?;
        let m: Model = serde_json::from_value(
            v.get("nonlin")
                .cloned()
                .ok_or_else(|| format!("{}: no nonlin section", path.display()))?,
        )
        .map_err(|e| e.to_string())?;
        if m.version != MODEL_VERSION {
            return Err(format!(
                "nonlin model version {} (this ptxv reads {MODEL_VERSION})",
                m.version
            ));
        }
        Ok(m)
    }
    /// Write the `nonlin` section into a model file, keeping its other sections.
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let mut v: serde_json::Value = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .filter(|v: &serde_json::Value| v.is_object())
            .unwrap_or_else(|| serde_json::json!({"kind": "ptxv-errdist-model"}));
        v["nonlin"] = serde_json::to_value(self).map_err(|e| e.to_string())?;
        std::fs::write(path, serde_json::to_string(&v).map_err(|e| e.to_string())?)
            .map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// Extend the tables to every binade the index values reach at `n` sampled inputs of `spec`
/// (with a margin of 2 binades, contiguous with the covered range). Returns the number of
/// binades added.
pub fn ensure(
    t: &mut Tables,
    progs: &[(&Prog, &ident::Idents)],
    spec: &Spec,
    layouts: &[Layout],
    n: u64,
) -> Result<usize, String> {
    for x in t.tables.iter_mut() {
        x.reindex();
    }
    // Per table: the (sign, binade)s needed.
    let mut need: BTreeMap<u32, std::collections::BTreeSet<(i8, i16)>> = BTreeMap::new();
    for (p, _) in progs {
        let ot = op_tables(p, t);
        if ot.iter().all(|x| x.is_none()) {
            continue;
        }
        for i in 0..n {
            let smp = super::spec::sample(spec, layouts, (1u64 << 42) + i)?;
            let w = p.leaves(&smp.bytes)?;
            let rv = real::eval(p, &w);
            for (k, op) in p.ops.iter().enumerate() {
                let Some(ti) = ot[k] else { continue };
                let x = match op {
                    Op::Nl(NlOp::Seg {
                        args,
                        real: Some(r),
                        ..
                    }) => {
                        let a: Vec<f64> = args.iter().map(|&a| rv[a as usize].0.v()).collect();
                        table::index_value(r, &a)
                    }
                    Op::Nl(NlOp::Mufu { arg, .. }) => rv[*arg as usize].0.v(),
                    _ => continue,
                };
                let (s, e, _) = table::cell_of(x);
                if s != 0 && !t.tables[ti as usize].covers(s, e) {
                    need.entry(ti).or_default().insert((s, e));
                }
            }
        }
    }
    let mut added = 0;
    for (ti, set) in need {
        let range0 = t.tables[ti as usize].range.clone();
        let key = t
            .by_key
            .iter()
            .find(|x| *x.1 == ti)
            .map(|x| x.0.clone())
            .unwrap_or_default();
        // Per sign: the new contiguous range, and the binades not yet covered.
        let mut signs: BTreeMap<i8, (i16, i16)> = BTreeMap::new();
        for &(s, e) in &set {
            let r = signs.entry(s).or_insert((e, e));
            r.0 = r.0.min(e - 2);
            r.1 = r.1.max(e + 2);
        }
        for (s, (lo, hi)) in signs {
            let (lo, hi) = match range0.iter().find(|r| r.0 == s) {
                Some(r) => (lo.min(r.1), hi.max(r.2)),
                None => (lo, hi),
            };
            let items: Vec<(i8, i16)> = (lo..=hi)
                .filter(|&e| !t.tables[ti as usize].covers(s, e))
                .map(|e| (s, e))
                .collect();
            added += items.len();
            let cells = if key.starts_with("mufu ") {
                let Some(NlOp::Mufu {
                    func,
                    ptx,
                    neg,
                    abs,
                    ..
                }) = progs.iter().find_map(|(p, _)| {
                    p.ops.iter().find_map(|o| match o {
                        Op::Nl(n @ NlOp::Mufu { .. }) if mufu_key(n).as_deref() == Some(&key) => {
                            Some(n.clone())
                        }
                        _ => None,
                    })
                })
                else {
                    continue;
                };
                let rs: Vec<(i8, i16, i16)> = items.iter().map(|&(s, e)| (s, e, e)).collect();
                table::mufu_cells(func, ptx, neg, abs, &rs)
            } else {
                let Some(info) = progs.iter().find_map(|(_, ids)| ids.get(&key)) else {
                    continue;
                };
                let Ok(r) = &info.result else { continue };
                table::slice_cells(&info.prog, r, &info.probes, &items)
            };
            t.tables[ti as usize].extend(cells, s, lo, hi);
            // Joint tables with this slice: their new binades too.
            for pi in 0..t.pairs.len() {
                let (a, b) = (t.pairs[pi].0, t.pairs[pi].1);
                if a != ti && b != ti {
                    continue;
                }
                let info = |x: u32| {
                    let k = t.by_key.iter().find(|y| *y.1 == x)?.0;
                    progs.iter().find_map(|(_, ids)| ids.get(k))
                };
                let (Some(ia), Some(ib)) = (info(a), info(b)) else {
                    continue;
                };
                let (Ok(ra), Ok(rb)) = (&ia.result, &ib.result) else {
                    continue;
                };
                let rs: Vec<(i8, i16, i16)> = items.iter().map(|&(s, e)| (s, e, e)).collect();
                if let Some(pt) =
                    table::pair_table(&ia.prog, ra, &ia.probes, &ib.prog, rb, &ib.probes, &rs)
                {
                    t.pairs[pi].2.extend(pt.cells, s, lo, hi);
                }
            }
        }
    }
    Ok(added)
}

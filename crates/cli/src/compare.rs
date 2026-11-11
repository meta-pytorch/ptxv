/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv compare REF CAND`: one report on whether a candidate kernel computes the same thing as
//! a reference, for two captured cases (tools/capture.py; tools/compare.py captures and calls
//! this).
//!
//! 1. Each side: lift (or list every instruction the lifter does not lower), then the
//!    execution verdict (`check --certify`: the E-WELLSYNC certificate, else exhaustive
//!    exploration; one CTA at a time when a multi-CTA launch is not finished in --max-states),
//!    and the simulator's output against the captured GPU output.
//! 2. `ptxv equiv` on the pair (a child process; its log is kept), segment mode when either
//!    kernel uses the MUFU unit.
//! 3. The captured GPU outputs against each other, with the proven bound evaluated at the
//!    captured inputs.
//!
//! Writes `report.json` and `report.md` into --out and prints a short summary. Every step that
//! cannot run or prove its part says why (the blockers).

use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
use ptxv_case::Case;
use ptxv_sim::explore::Verdict;
use serde_json::Value;
use serde_json::json;

use super::equiv::gpu_output_f32;
use super::load_with;
use super::machine;

pub struct Opts {
    pub out: PathBuf,
    pub max_states: usize,
    pub trials: usize,
    /// None: automatic (segment mode when a kernel uses MUFU).
    pub segments: Option<bool>,
    pub sweep_cache: Option<PathBuf>,
    pub hw: bool,
    /// The real function an elementwise pair computes (default: identified).
    pub function: Option<String>,
    /// Human-readable descriptions of the two sides (what was captured).
    pub ref_label: Option<String>,
    pub cand_label: Option<String>,
    /// Write the chosen equiv run's bound data here (`ptxv equiv --emit-bound`).
    pub emit_bound: Option<PathBuf>,
}

/// One side: lift, execution verdict, simulator vs GPU.
struct SideOut {
    json: Value,
    blockers: Vec<String>,
    /// A blocker only when the equivalence is not proven (equiv may excuse it: N-ATOMIC).
    nondet: Option<String>,
    lifted: bool,
    front: String,
    slices: bool,
    mufu: bool,
    assumptions: Vec<String>,
}

fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    e.downcast_ref::<String>()
        .cloned()
        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "panic".into())
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").trim().to_string()
}

/// Lift blockers of every launch (binary cases: their cubins; PTX cases: ptxas's output).
fn lift_blockers(case: &Case, dir: &Path, variant: &str) -> Vec<Value> {
    let mut out = Vec::new();
    for l in case.m.all_launches() {
        let cubin = match (l.cubin, &case.m.cubin) {
            (Some(own), _) => case.dir.join(own),
            (None, Some(c)) => case.dir.join(c),
            (None, None) => dir.join(format!("{variant}.cubin")),
        };
        match ptxv_sass::lift_report(&cubin, l.entry) {
            Ok(u) if u.is_empty() => {}
            Ok(u) => out.push(json!({
                "kernel": l.entry,
                "unlifted": u.iter().map(|x| json!({"instruction": x.why.replace('`', ""), "count": x.count, "example": first_line(&x.example)})).collect::<Vec<_>>(),
            })),
            Err(e) => out.push(json!({"kernel": l.entry, "error": first_line(&e.to_string())})),
        }
    }
    out
}

fn short_kernel(k: &str) -> String {
    if k.len() > 60 {
        format!("{}...", &k[..57])
    } else {
        k.to_string()
    }
}

fn verdict_json(v: &Verdict, method: &str) -> Value {
    json!({
        "verdict": v.classes(),
        "complete": v.complete,
        "method": method,
        "certified": v.assumptions.iter().any(|a| a == "E-WELLSYNC"),
        "states": v.states,
        "distinct_outputs": v.distinct_outputs,
        "assumptions": v.assumptions,
        "notes": v.notes,
        "findings": v.findings.iter().map(|f| json!({"kind": f.kind, "msg": first_line(&f.msg)})).collect::<Vec<_>>(),
        "seconds": v.seconds,
    })
}

fn side(who: &str, dir_case: &Path, out: &Path, max_states: usize) -> Result<SideOut> {
    let case = Case::load(dir_case)?;
    eprintln!("compare: {who} {}: lift + check", case.dir.display());
    let dir = out.join(who);
    std::fs::create_dir_all(&dir)?;
    let variant = if case.variant("good").is_ok() {
        "good".to_string()
    } else {
        case.m
            .variants
            .first()
            .map_or("good".into(), |v| v.id.clone())
    };
    let kernels: Vec<String> = case
        .m
        .all_launches()
        .iter()
        .map(|l| l.entry.to_string())
        .collect();
    let mut blockers = Vec::new();
    let mut j = json!({
        "case": case.dir.canonicalize().unwrap_or(case.dir.clone()),
        "name": case.m.name,
        "kernels": kernels,
        "variant": variant,
    });
    // Lift: the SASS front (the binary that ran, or ptxas's output of a PTX case); a PTX case
    // falls back to the PTX front.
    let mut fronts = vec!["sass"];
    if case.m.cubin.is_none() {
        fronts.push("ptx");
    }
    let mut lifted = None;
    let mut errs = Vec::new();
    for f in &fronts {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            load_with(&case, &variant, f, &dir, &[])
        }));
        match r {
            Ok(Ok((k, _))) => {
                lifted = Some((k, f.to_string()));
                break;
            }
            Ok(Err(e)) => errs.push(format!("{f}: {}", first_line(&format!("{e:#}")))),
            Err(p) => errs.push(format!("{f}: panic: {}", panic_msg(p))),
        }
    }
    let Some((k, front)) = lifted else {
        let lb = lift_blockers(&case, &dir, &variant);
        for l in &lb {
            let kname = short_kernel(l["kernel"].as_str().unwrap_or("?"));
            if let Some(u) = l["unlifted"].as_array() {
                let list = u
                    .iter()
                    .map(|x| {
                        format!(
                            "`{}` x{}",
                            x["instruction"].as_str().unwrap_or("?"),
                            x["count"]
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                blockers.push(format!(
                    "{who}: kernel {kname} does not lift, {} kind(s) of unlifted instruction: {list}",
                    u.len()
                ));
            } else if let Some(e) = l["error"].as_str() {
                blockers.push(format!("{who}: kernel {kname}: {e}"));
            }
        }
        if lb.is_empty() {
            blockers.push(format!("{who}: does not load: {}", errs.join("; ")));
        }
        j["lifted"] = json!(false);
        j["lift_errors"] = json!(errs);
        j["unlifted"] = json!(lb);
        return Ok(SideOut {
            json: j,
            blockers,
            nondet: None,
            lifted: false,
            front: "sass".into(),
            slices: false,
            mufu: false,
            assumptions: vec![],
        });
    };
    j["lifted"] = json!(true);
    j["front"] = json!(front);
    let mufu = k
        .code
        .iter()
        .any(|i| matches!(i.op, ptxv_ir::Op::Mufu { .. }));
    // Execution verdict: certificate, else exploration; per CTA when a multi-CTA launch does
    // not finish whole.
    let ctas: u32 = case
        .m
        .all_launches()
        .iter()
        .map(|l| l.grid.iter().product::<u32>())
        .max()
        .unwrap_or(1);
    // ASSUME(E-SMID, E-CLOCK): as the launch check (`ptxv check`), one check per %smid
    // placement and clock-dependent outputs incomplete (smid::check_placements).
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> Result<(Verdict, bool)> {
            drop(machine(&case, k.clone())?);
            let make = || machine(&case, k.clone()).expect("machine");
            if ctas <= 1 {
                return Ok((
                    ptxv_sim::smid::check_placements(&make, &mut |mk| {
                        ptxv_sim::wellsync::check_certified(mk(), max_states)
                    }),
                    false,
                ));
            }
            // Several CTAs: the certificate alone, then one cluster at a time (symmetry, or every
            // cluster) rather than the whole launch's interleavings.
            let only = std::env::var_os("PTXV_CERTIFY_ONLY");
            std::env::set_var("PTXV_CERTIFY_ONLY", "1");
            let v = ptxv_sim::smid::check_placements(&make, &mut |mk| {
                ptxv_sim::wellsync::check_certified(mk(), max_states)
            });
            match only {
                Some(o) => std::env::set_var("PTXV_CERTIFY_ONLY", o),
                None => std::env::remove_var("PTXV_CERTIFY_ONLY"),
            }
            if v.complete {
                return Ok((v, false));
            }
            let mut vs = ptxv_sim::smid::check_placements(&make, &mut |mk| {
                ptxv_sim::explore::explore_symmetric(mk, max_states, &mut |_, _| {})
            });
            if let Some(n) = v.notes.first() {
                vs.notes.insert(0, n.clone());
            }
            Ok((vs, true))
        },
    ));
    let (v, slices) = match r {
        Ok(Ok(x)) => x,
        Ok(Err(e)) => {
            blockers.push(format!(
                "{who}: check failed: {}",
                first_line(&format!("{e:#}"))
            ));
            j["check"] = json!({"error": format!("{e:#}")});
            return Ok(SideOut {
                json: j,
                blockers,
                nondet: None,
                lifted: true,
                front,
                slices: false,
                mufu,
                assumptions: vec![],
            });
        }
        Err(p) => {
            let m = panic_msg(p);
            blockers.push(format!("{who}: check panicked: {m}"));
            j["check"] = json!({"error": format!("panic: {m}")});
            return Ok(SideOut {
                json: j,
                blockers,
                nondet: None,
                lifted: true,
                front,
                slices: false,
                mufu,
                assumptions: vec![],
            });
        }
    };
    let method = match (v.assumptions.iter().any(|a| a == "E-WELLSYNC"), slices) {
        (true, false) => "certified, E-WELLSYNC",
        (_, true) => "one cluster at a time (E-CTA-INDEP), explored",
        (false, false) => "explored",
    };
    j["check"] = verdict_json(&v, method);
    let classes = v.classes();
    // One line per kind of finding: how many, and the first.
    let mut kinds: Vec<(&str, usize, String)> = Vec::new();
    // Informational notes (vprintf sites, S-GLOBAL-INIT zero globals) are not blockers.
    for f in v.findings.iter().filter(|f| {
        f.kind != "contract"
            && !(f.kind == "note" && (f.sig.starts_with("printf:") || f.sig == "global-zero"))
    }) {
        match kinds.iter_mut().find(|k| k.0 == f.kind) {
            Some(k) => k.1 += 1,
            None => kinds.push((&f.kind, 1, first_line(&f.msg))),
        }
    }
    for (kind, n, first) in kinds {
        blockers.push(if n == 1 {
            format!("{who}: {kind}: {first}")
        } else {
            format!("{who}: {n} {kind} findings, first: {first}")
        });
    }
    if !v.complete {
        blockers.push(format!(
            "{who}: exploration not finished in {} states (no proof of race / deadlock freedom)",
            v.states
        ));
    }
    let nondet = classes.iter().any(|c| c == "nondeterministic").then(|| {
        format!(
            "{who}: {} distinct outputs over schedules (nondeterministic)",
            v.distinct_outputs
        )
    });
    // The simulator against the captured GPU output.
    let mut sim_gpu = Vec::new();
    if v.outputs.len() > 1 {
        let reps: Vec<Vec<String>> = v
            .outputs
            .iter()
            .map(|o| case.expect_report(o).unwrap_or_default())
            .collect();
        let hit = reps
            .iter()
            .position(|r| !r.is_empty() && r.iter().all(|l| l.contains("bitwise equal")));
        sim_gpu.push(match hit {
            Some(i) => format!(
                "the captured GPU output is bitwise equal to simulated output {} of {} (schedule-dependent)",
                i + 1,
                v.outputs.len()
            ),
            None => format!(
                "MISMATCH: the captured GPU output equals none of the {} simulated outputs",
                v.outputs.len()
            ),
        });
    } else if v.outputs.is_empty() {
        // Slice-wise exploration keeps no whole-launch output: one deterministic run.
        if let Ok(m) = machine(&case, k.clone()) {
            let (m, _, ok) = ptxv_sim::explore::run_once(m);
            if ok {
                for l in case.expect_report(&m.outputs()).unwrap_or_default() {
                    sim_gpu.push(if nondet.is_some() {
                        // The output depends on the schedule: one simulated schedule need
                        // not be the GPU's.
                        format!(
                            "schedule-dependent output; one simulated schedule: {}",
                            l.replace("MISMATCH vs", "differs from")
                        )
                    } else {
                        format!("{l} (one schedule)")
                    });
                }
            }
        }
    } else {
        for o in &v.outputs {
            sim_gpu.extend(case.expect_report(o).unwrap_or_default());
        }
    }
    j["sim_vs_gpu"] = json!(sim_gpu);
    if sim_gpu.iter().any(|l| l.contains("MISMATCH")) {
        blockers.push(format!(
            "{who}: the simulator's output differs from the captured GPU output ({})",
            sim_gpu.iter().find(|l| l.contains("MISMATCH")).unwrap()
        ));
    }
    Ok(SideOut {
        json: j,
        blockers,
        nondet,
        lifted: true,
        front,
        slices,
        mufu,
        assumptions: v.assumptions.clone(),
    })
}

/// Input buffers of both cases with the same name must hold the same bytes for the GPU
/// outputs to be comparable. Returns the first differing one.
fn inputs_differ(r: &Case, c: &Case) -> Option<String> {
    for b in &r.m.buffers {
        if crate::equiv::out_only(r, &b.name) || b.init != "file" {
            continue;
        }
        if let Some(o) = c.m.buffers.iter().find(|o| o.name == b.name) {
            // Workspaces (`W0`, ...: not captured inputs) are not compared.
            if crate::equiv::out_only(c, &o.name) || o.init != "file" {
                continue;
            }
            if b.dtype != o.dtype || b.bytes() != o.bytes() {
                return Some(format!(
                    "{} ({} {:?} vs {} {:?})",
                    b.name, b.dtype, b.shape, o.dtype, o.shape
                ));
            }
            if b.init_bytes() != o.init_bytes() {
                return Some(b.name.clone());
            }
        }
    }
    None
}

fn ulps(a: f32, b: f32) -> u64 {
    let key = |x: f32| {
        let i = x.to_bits() as i32;
        if i < 0 {
            i32::MIN as i64 - i as i64
        } else {
            i as i64
        }
    };
    (key(a) - key(b)).unsigned_abs()
}

/// The two captured GPU outputs against each other.
fn gpu_compare(r: &Case, c: &Case) -> Option<Value> {
    let (a, b) = (gpu_output_f32(r)?, gpu_output_f32(c)?);
    if a.len() != b.len() {
        return Some(json!({"error": format!("output sizes differ: {} vs {}", a.len(), b.len())}));
    }
    let (mut max_abs, mut max_rel, mut max_ulps, mut worst) = (0.0f64, 0.0f64, 0u64, 0usize);
    let mut eq = 0;
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        if x.to_bits() == y.to_bits() {
            eq += 1;
        }
        let d = (*x as f64 - *y as f64).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
        if *x != 0.0 {
            max_rel = max_rel.max(d / (*x as f64).abs());
        }
        if x.is_finite() && y.is_finite() {
            max_ulps = max_ulps.max(ulps(*x, *y));
        }
    }
    Some(json!({
        "outputs": a.len(),
        "bitwise_equal": eq,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "max_ulps": max_ulps,
        "worst_output": worst,
        "worst_values": [a[worst], b[worst]],
    }))
}

/// max, median, min and a histogram of the per-output bounds (units of u).
fn bound_stats(per: &[Value]) -> Value {
    let mut b: Vec<f64> = per.iter().filter_map(|o| o["bound_u"].as_f64()).collect();
    let n = per.len();
    if b.is_empty() {
        return json!({"outputs": n, "with_bound": 0});
    }
    b.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let (lo, hi) = (b[0], b[b.len() - 1]);
    let med = if b.len() % 2 == 1 {
        b[b.len() / 2]
    } else {
        (b[b.len() / 2 - 1] + b[b.len() / 2]) / 2.0
    };
    let mut distinct: Vec<(f64, usize)> = Vec::new();
    for &x in &b {
        match distinct.last_mut() {
            Some(d) if (d.0 - x).abs() <= 1e-9 * x.abs().max(1.0) => d.1 += 1,
            _ => distinct.push((x, 1)),
        }
    }
    let hist: Vec<Value> = if distinct.len() <= 8 {
        distinct
            .iter()
            .map(|(v, c)| json!({"lo": v, "hi": v, "count": c}))
            .collect()
    } else {
        let nb = 8;
        let w = (hi - lo) / nb as f64;
        let mut cnt = vec![0usize; nb];
        for &x in &b {
            cnt[(((x - lo) / w) as usize).min(nb - 1)] += 1;
        }
        cnt.iter()
            .enumerate()
            .map(|(i, c)| json!({"lo": lo + w * i as f64, "hi": lo + w * (i + 1) as f64, "count": c}))
            .collect()
    };
    let own = |k: &str| {
        per.iter()
            .filter_map(|o| o[k].as_f64())
            .fold(None, |a: Option<f64>, x| Some(a.map_or(x, |a| a.max(x))))
    };
    json!({"outputs": n, "with_bound": b.len(), "max_u": hi, "median_u": med, "min_u": lo, "histogram": hist,
           "reference_own_max_u": own("ref_u"), "candidate_own_max_u": own("cand_u")})
}

/// Blockers from the equiv report (why something is not proven).
/// Did elementwise mode prove the pair (every output one input element's function, swept
/// over every input value)?
fn elementwise_proven(e: &Value) -> bool {
    e["proven"] == true && e["elementwise"]["max_ulps"].is_u64()
}

fn equiv_blockers(e: &Value, inputs_bf16: bool) -> Vec<String> {
    let mut out = Vec::new();
    // Elementwise mode replaces the normal forms: their blockers do not apply.
    if elementwise_proven(e) {
        return out;
    }
    for (who, key) in [
        ("reference", "pre_reference"),
        ("candidate", "pre_candidate"),
    ] {
        let pre: Vec<&str> = e[key]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
            .unwrap_or_default();
        let guards: Vec<&str> = pre
            .iter()
            .filter(|p| p.starts_with(crate::equiv::DATA_DEPENDENT))
            .copied()
            .collect();
        if !guards.is_empty() {
            let first = guards[0]
                .trim_start_matches(crate::equiv::DATA_DEPENDENT)
                .replace("guard of ", "guard at ")
                .replace(" depends on input data", "");
            out.push(format!(
                "{who}: {} data-dependent branch(es)/guard(s) or address(es) (first: {first}); the taint check refuses a proof for every input",
                guards.len()
            ));
        }
        // "not clean" repeats the execution verdict's findings, reported per side already.
        for p in pre
            .iter()
            .filter(|p| !p.starts_with("data-dependent:") && !p.starts_with("not clean"))
        {
            out.push(format!("{who}: {p}"));
        }
    }
    for p in e["pre_pair"].as_array().into_iter().flatten() {
        out.push(format!("pair: {}", p.as_str().unwrap_or("")));
    }
    let rep = &e["report"];
    if let Some(u) = rep["untracked"].as_str() {
        let n = rep["outputs"].as_u64().unwrap_or(0);
        out.push(format!(
            "outputs with no dataflow history over the inputs: reference {}/{n}, candidate {}/{n} (e.g. {u}){}",
            e["untracked_reference"],
            e["untracked_candidate"],
            if inputs_bf16 {
                ": the outputs are bf16 / f16: the numerics express f32 outputs (bf16 / f16 inputs are leaves, N-HALF-LEAF), and elementwise mode handles 16-bit outputs of one input element only"
            } else {
                ": the numerics shadow does not express this computation over the inputs (it covers sums and products of inputs, and the softmax normal form in segment mode)"
            }
        ));
    }
    // With untracked outputs an L1 failure is only their constants differing.
    if let (Some(levels), true) = (rep["levels"].as_array(), rep["untracked"].is_null()) {
        if let Some(l1) = levels
            .iter()
            .find(|l| l["level"].as_str().is_some_and(|s| s.starts_with("L1")))
        {
            if l1["holds"] == false {
                out.push(format!(
                    "not the same function at {} of {} outputs (first: out[{}]): {}",
                    l1["failures"],
                    rep["outputs"],
                    l1["first_failure"],
                    l1["detail"].as_str().unwrap_or("")
                ));
            }
        }
    }
    // Outputs without a bound, grouped by reason (L1 failures are reported above).
    let mut reasons: Vec<(String, usize)> = Vec::new();
    for o in rep["per_output"].as_array().into_iter().flatten() {
        if let Some(w) = o["no_bound"].as_str() {
            if w.contains("L1 fails") {
                continue;
            }
            match reasons.iter_mut().find(|r| r.0 == w) {
                Some(r) => r.1 += 1,
                None => reasons.push((w.to_string(), 1)),
            }
        }
    }
    for (w, n) in reasons {
        out.push(format!("{n} output(s) without a bound: {w}"));
    }
    out
}

/// Run `ptxv equiv` on the pair (a child process: a crash or panic stays a report line).
#[allow(clippy::too_many_arguments)]
fn run_equiv(
    out: &Path,
    o: &Opts,
    ref_dir: &Path,
    cand_dir: &Path,
    rs: &SideOut,
    cs: &SideOut,
    segments: bool,
    cache: &Path,
) -> Result<(Option<Value>, String)> {
    let tag = if segments { "equiv-seg" } else { "equiv" };
    let edir = out.join(tag);
    let q = |p: &Path| {
        p.canonicalize()
            .unwrap_or(p.to_path_buf())
            .display()
            .to_string()
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
    };
    let toml = format!(
        "# written by ptxv compare\nmax_states = {}\ntrials = {}\nsegments = {segments}\nsweep_cache = \"{}\"\n\
         reference = {{ case = \"{}\", variant = \"{}\", front = \"{}\", slices = {}, certify = true }}\n\n\
         [[candidates]]\ncase = \"{}\"\nvariant = \"{}\"\nfront = \"{}\"\nslices = {}\ncertify = true\n",
        o.max_states,
        o.trials,
        q(cache),
        q(ref_dir),
        rs.json["variant"].as_str().unwrap(),
        rs.front,
        rs.slices,
        q(cand_dir),
        cs.json["variant"].as_str().unwrap(),
        cs.front,
        cs.slices,
    );
    let toml = match &o.function {
        Some(f) => toml.replacen(
            "\nreference = ",
            &format!("\nfunction = \"{f}\"\nreference = "),
            1,
        ),
        None => toml,
    };
    let tp = out.join(format!("{tag}.toml"));
    std::fs::write(&tp, toml)?;
    let mut cmd = std::process::Command::new(std::env::current_exe()?);
    cmd.arg("equiv")
        .arg(&tp)
        .arg("--out")
        .arg(&edir)
        .env("PTXV_QUIET", "1");
    if !o.hw {
        cmd.arg("--no-hw");
    }
    if o.emit_bound.is_some() {
        cmd.arg("--emit-bound").arg(edir.join("bound.json"));
    }
    let _ = std::fs::remove_file(edir.join("equiv.json"));
    let res = cmd.output()?;
    let log = format!(
        "$ ptxv equiv {}\n{}{}{}",
        tp.display(),
        String::from_utf8_lossy(&res.stdout),
        String::from_utf8_lossy(&res.stderr),
        if res.status.success() {
            String::new()
        } else {
            format!("\n[exit: {}]\n", res.status)
        }
    );
    let parsed = std::fs::read_to_string(edir.join("equiv.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v.get(0).cloned());
    Ok((parsed, log))
}

fn bounded(e: &Value) -> usize {
    e["report"]["per_output"]
        .as_array()
        .map_or(0, |a| a.iter().filter(|o| o["bound_u"].is_f64()).count())
}

fn fully_proven(e: Option<&Value>) -> bool {
    e.is_some_and(|e| {
        elementwise_proven(e)
            || e["proven"] == true
                && bounded(e) == e["report"]["outputs"].as_u64().unwrap_or(0) as usize
    })
}

/// Which of two equiv results is better: proven, then outputs with a bound, then the level.
fn rank(e: Option<&Value>) -> (bool, usize, bool) {
    match e {
        None => (false, 0, false),
        Some(e) => (
            e["proven"] == true,
            if elementwise_proven(e) {
                e["report"]["outputs"].as_u64().unwrap_or(0) as usize
            } else {
                bounded(e)
            },
            !e["strongest"]
                .as_str()
                .unwrap_or("none")
                .starts_with("none"),
        ),
    }
}

/// A contract note ("... (E-BULK-EXIT)") as its ID and location; others unchanged.
fn short_relies(r: &str) -> String {
    let Some(id) = r
        .strip_suffix(')')
        .and_then(|x| x.rsplit_once('('))
        .map(|x| x.1)
        .filter(|id| id.starts_with("E-") && !id.contains(' '))
    else {
        return r.to_string();
    };
    let at = r
        .find("sass:0x")
        .map(|i| {
            let t = &r[i..];
            format!(" at {}", &t[..t.find(' ').unwrap_or(t.len())])
        })
        .unwrap_or_default();
    format!("{id}{at}")
}

fn fmt_u(x: f64) -> String {
    if x == 0.0 {
        "0".into()
    } else {
        format!("{x:.2}u")
    }
}

pub fn cmd_compare(ref_dir: &Path, cand_dir: &Path, o: &Opts) -> Result<bool> {
    std::fs::create_dir_all(&o.out)?;
    let out = o.out.canonicalize()?;
    let (rc, cc) = (Case::load(ref_dir)?, Case::load(cand_dir)?);
    let rs = side("reference", ref_dir, &out, o.max_states)?;
    let cs = side("candidate", cand_dir, &out, o.max_states)?;
    let mut blockers: Vec<String> = rs.blockers.iter().chain(&cs.blockers).cloned().collect();
    let differ = inputs_differ(&rc, &cc);
    if let Some(d) = &differ {
        blockers.push(format!(
            "input buffer {d} differs between the two cases: the captured GPU outputs are not comparable"
        ));
    }
    let gpu = if differ.is_none() {
        gpu_compare(&rc, &cc)
    } else {
        None
    };
    // 16-bit outputs (the numerics shadow's outputs are f32 words).
    let inputs_bf16 = rc.m.buffers.iter().chain(&cc.m.buffers).any(|b| {
        (rc.m.outputs.contains(&b.name) || cc.m.outputs.contains(&b.name))
            && matches!(b.dtype.as_str(), "bf16" | "f16")
    }) && rc.m.buffers.iter().chain(&cc.m.buffers).any(|b| {
        b.init == "file"
            && !rc.m.outputs.contains(&b.name)
            && !cc.m.outputs.contains(&b.name)
            && b.dtype != "f32"
    });
    // Equivalence.
    let mut equiv: Option<Value> = None;
    let mut equiv_log: Option<PathBuf> = None;
    let mut segments = o.segments.unwrap_or(false);
    if rs.lifted && cs.lifted {
        let cache = match &o.sweep_cache {
            Some(c) => std::path::absolute(c)?,
            None => {
                let c = out.join("sweeps.json");
                // Seed with the committed softmax sweeps (keyed by slice instructions).
                let seed = Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../fixtures/c_equiv_softmax/sweeps.json");
                if !c.exists() && seed.exists() {
                    let _ = std::fs::copy(&seed, &c);
                }
                // And the elementwise sweeps of the committed pairs.
                let ec = c.with_extension("elementwise.json");
                let eseed = Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../fixtures/c_equiv_elementwise/sweeps.elementwise.json");
                if !ec.exists() && eseed.exists() {
                    let _ = std::fs::copy(&eseed, &ec);
                }
                c
            }
        };
        let run = |seg: bool| -> Result<(Option<Value>, String)> {
            run_equiv(&out, o, ref_dir, cand_dir, &rs, &cs, seg, &cache)
        };
        eprintln!("compare: equiv (segments = {segments})");
        let (mut e, mut log) = run(segments)?;
        // Automatic: plain numerics first (sums, moves); segment mode (products, maxima,
        // instruction slices: softmax) when a kernel uses the MUFU unit and the plain run
        // leaves outputs without a proven bound. The better of the two is kept.
        // Also when outputs have no dataflow history: elementwise kernels (segment mode's
        // slices, then elementwise mode).
        let untracked = e
            .as_ref()
            .is_some_and(|v| !v["report"]["untracked"].is_null());
        if o.segments.is_none() && (rs.mufu || cs.mufu || untracked) && !fully_proven(e.as_ref()) {
            eprintln!(
                "compare: equiv again in segment mode ({})",
                if rs.mufu || cs.mufu {
                    "a kernel uses MUFU"
                } else {
                    "outputs without dataflow history"
                }
            );
            let (e2, log2) = run(true)?;
            if rank(e2.as_ref()) > rank(e.as_ref()) {
                (e, log, segments) = (e2, log2, true);
            }
        }
        std::fs::write(out.join("equiv.log"), &log)?;
        equiv_log = Some(out.join("equiv.log"));
        if let Some(p) = &o.emit_bound {
            let src = out
                .join(if segments { "equiv-seg" } else { "equiv" })
                .join("bound.json");
            if src.exists() {
                std::fs::copy(&src, p)?;
            }
        }
        match e {
            Some(v) => equiv = Some(v),
            None => {
                let tail: Vec<&str> = log.lines().rev().take(3).collect();
                blockers.push(format!(
                    "equiv did not finish: {}",
                    tail.into_iter().rev().collect::<Vec<_>>().join(" | ")
                ));
            }
        }
    } else {
        blockers.push("equivalence not attempted: a kernel does not lift".into());
    }
    // Assemble.
    let mut eq_json = json!(null);
    let mut level = "not attempted".to_string();
    let mut proven = false;
    let mut relies: Vec<String> = Vec::new();
    let mut frac = None;
    if let Some(e) = &equiv {
        level = e["strongest"].as_str().unwrap_or("?").to_string();
        proven = e["proven"] == true;
        for (who, key) in [
            ("reference", "notes_reference"),
            ("candidate", "notes_candidate"),
        ] {
            for n in e[key]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|x| x.as_str())
            {
                if let Some(r) = n.strip_prefix("relies on: ") {
                    relies.push(format!("{who}: {r}"));
                }
            }
        }
        if !proven {
            blockers.extend(rs.nondet.iter().chain(&cs.nondet).cloned());
        }
        blockers.extend(equiv_blockers(e, inputs_bf16));
        // Untracked outputs are compared as constants: their "bounds" mean nothing.
        let per = if e["report"]["untracked"].is_null() {
            e["report"]["per_output"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        } else {
            vec![]
        };
        let stats = bound_stats(&per);
        let forms: Vec<&str> = {
            let mut f: Vec<&str> = per.iter().filter_map(|o| o["form"].as_str()).collect();
            f.sort();
            f.dedup();
            f
        };
        let softmax_form = per
            .iter()
            .filter_map(|o| Some((o["rel_u"].as_f64()?, o["add_u"].as_f64()?)))
            .fold(None, |a: Option<(f64, f64)>, (r, d)| {
                Some(a.map_or((r, d), |(x, y)| (x.max(r), y.max(d))))
            });
        let cap = &e["captured"];
        frac = cap["max_fraction_of_bound"].as_f64();
        eq_json = json!({
            "level": level,
            "proven": proven,
            "output_scope": e["output_scope"],
            "levels": e["report"]["levels"],
            "segments": segments,
            "bound_form": forms,
            "bound_meaning": if forms == ["softmax"] {
                "|C - R| <= rel_u u y + add_u u, y the exact output (<= 1)"
            } else {
                "|C - R| <= bound_u u sum_m a_m |x_m| (+ abs_extra), for every input"
            },
            "softmax_form": softmax_form.map(|(r, a)| json!({"rel_u": r, "add_u": a})),
            // 16-bit output forms ("round"; "norm" of rounded outputs): the format, whether a
            // kernel rounds to it before the output (eager between kernels), and at the
            // captured inputs the outputs the bound proves bitwise equal / within 1 ulp.
            "round16": round16_summary(&per, cap),
            "per_output_bound": stats,
            "max_bound_u": e["max_bound_u"],
            "unordered": e["report"]["unordered"],
            "softmax": e["report"]["softmax"],
            "untracked": e["report"]["untracked"],
            "trials": e["trials"],
            "captured": cap,
            "witness": e["witness"],
            "elementwise": e["elementwise"],
            "relies_on": relies,
            "log": equiv_log,
        });
    }
    let mut assumptions: Vec<String> = rs
        .assumptions
        .iter()
        .chain(&cs.assumptions)
        .cloned()
        .collect();
    assumptions.sort();
    assumptions.dedup();
    let mut gpu_j = gpu.clone().unwrap_or(json!(null));
    if let (Some(f), false) = (frac, gpu_j.is_null()) {
        gpu_j["max_fraction_of_bound"] = json!(f);
    }
    let headline = headline(&level, proven, &eq_json, &gpu_j);
    let report = json!({
        "reference": {"label": o.ref_label, "side": rs.json},
        "candidate": {"label": o.cand_label, "side": cs.json},
        "headline": headline,
        "equivalence": eq_json,
        "gpu": gpu_j,
        "inputs_identical": differ.is_none(),
        "assumptions": assumptions,
        "relies_on": relies,
        "blockers": blockers,
    });
    std::fs::write(
        out.join("report.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    let md = markdown(&report);
    std::fs::write(out.join("report.md"), &md)?;
    print!("{}", summary(&report));
    println!(
        "report: {}  {}",
        out.join("report.md").display(),
        out.join("report.json").display()
    );
    Ok(true)
}

/// The 16-bit output form over the outputs (None: no output is in it).
fn round16_summary(per: &[Value], cap: &Value) -> Value {
    let r: Vec<&Value> = per
        .iter()
        .filter(|o| o["round_f16"].is_boolean() || o["norm"]["out_round"].is_boolean())
        .collect();
    if r.is_empty() {
        return Value::Null;
    }
    let any = |k: usize| {
        r.iter()
            .any(|o| o["round_inner"].as_array().is_some_and(|a| a[k] == true))
    };
    let f16 = r
        .iter()
        .any(|o| o["round_f16"] == true || o["norm"]["out_round"] == true);
    json!({
        "format": if f16 { "f16" } else { "bf16" },
        "outputs": r.len(),
        "inner_reference": any(0),
        "inner_candidate": any(1),
        "captured_steps": cap["round16_steps"],
    })
}

fn headline(level: &str, proven: bool, eq: &Value, gpu: &Value) -> String {
    let b = &eq["per_output_bound"];
    let ew = &eq["elementwise"];
    let mut s = if level == "not attempted" {
        "equivalence not attempted".to_string()
    } else if proven && ew["max_ulps"].is_u64() {
        let n = ew["inputs_swept"].as_u64().unwrap_or(0);
        let all = format!(
            "every {} input (all {} values)",
            ew["input_type"].as_str().unwrap_or("?"),
            if n == 1 << 32 {
                "2^32".to_string()
            } else {
                n.to_string()
            }
        );
        let p16 = &ew["product16"];
        let mut s = if ew["product"] == true && p16.is_object() {
            // Outputs bf16(rn(f(x) * y)): both kernels at every (x, y) pair.
            let (pairs, eq) = (
                p16["pairs"].as_f64().unwrap_or(1.0),
                p16["bitwise_equal"].as_f64().unwrap_or(0.0),
            );
            if ew["bitwise_all"] == true {
                format!(
                    "PROVEN bitwise identical (L0): every output is {o}(rn(f(x) * y)), y another input element; equal at every (x, y) pair of {} values",
                    ew["input_type"].as_str().unwrap_or("?"),
                    o = ew["output_type"].as_str().unwrap_or("?")
                )
            } else {
                format!(
                    "PROVEN elementwise: every output is {o}(rn(f(x) * y)), y another input element; at every (x, y) pair ({:.0} pairs, exhaustive): max {} {o} ulp apart where f and the outputs are normal, max {} overall, |C - R| <= {:.3e}, bitwise equal on {:.4}%, {} special-value mismatches",
                    pairs,
                    p16["max_ulps_normal"],
                    p16["max_ulps"],
                    p16["max_abs"].as_f64().unwrap_or(f64::NAN),
                    100.0 * eq / pairs,
                    p16["special_mismatch"],
                    o = ew["output_type"].as_str().unwrap_or("?")
                )
            }
        } else if ew["product"] == true {
            // Outputs rn(f(x) * y): the sweep is of f.
            if ew["bitwise_all"] == true {
                format!(
                    "PROVEN bitwise identical (L0): every output is rn(f(x) * y), y another input element, with f bitwise equal for {all}"
                )
            } else {
                format!(
                    "PROVEN elementwise: every output is rn(f(x) * y), y another input element; over {all}, |f_C - f_R| <= {:.3e} (max {} ulp, bitwise equal on {:.4}%), so |C - R| <= |y| {:.3e} + 2^-24 (|C| + |R|) + 2^-149",
                    ew["max_abs"].as_f64().unwrap_or(f64::NAN),
                    ew["max_ulps"],
                    100.0 * ew["bitwise_equal_inputs"].as_f64().unwrap_or(0.0)
                        / ew["inputs"].as_f64().unwrap_or(1.0),
                    ew["max_abs"].as_f64().unwrap_or(f64::NAN),
                )
            }
        } else if ew["bitwise_all"] == true {
            format!("PROVEN bitwise identical (L0) elementwise for {all}")
        } else {
            format!(
                "PROVEN elementwise for {all}: |C - R| <= {:.3e}, max {} ulp apart, bitwise equal on {:.4}% of inputs",
                ew["max_abs"].as_f64().unwrap_or(f64::NAN),
                ew["max_ulps"],
                100.0 * ew["bitwise_equal_inputs"].as_f64().unwrap_or(0.0)
                    / ew["inputs"].as_f64().unwrap_or(1.0)
            )
        };
        if let (Some(f), Some(r), Some(c)) = (
            ew["true_fn"].as_str(),
            ew["reference_vs_true"].as_object(),
            ew["candidate_vs_true"].as_object(),
        ) {
            s += &format!(
                "; vs {f} (f64): reference |err| <= {:.3e}, candidate |err| <= {:.3e}",
                r["max_abs"].as_f64().unwrap_or(f64::NAN),
                c["max_abs"].as_f64().unwrap_or(f64::NAN)
            );
        }
        s
    } else if level.starts_with("L0") && proven {
        "PROVEN bitwise identical (L0) for every input".to_string()
    } else if proven && b["with_bound"].as_u64().unwrap_or(0) > 0 {
        let form = if let Some(sf) = eq["softmax_form"].as_object() {
            format!(
                "{:.2}u*y + {:.2}u",
                sf["rel_u"].as_f64().unwrap_or(0.0),
                sf["add_u"].as_f64().unwrap_or(0.0)
            )
        } else {
            format!("{} * sum|terms|", fmt_u(b["max_u"].as_f64().unwrap_or(0.0)))
        };
        let r16 = &eq["round16"];
        if r16.is_object() && eq["bound_form"] == json!(["round"]) {
            // Both outputs 16-bit roundings of L1-equal values.
            let fmt = r16["format"].as_str().unwrap_or("bf16");
            let inner = match (
                r16["inner_reference"] == true,
                r16["inner_candidate"] == true,
            ) {
                (false, false) => String::new(),
                (r, c) => format!(
                    " ({} rounds to {fmt} before the output too: different functions, those roundings bounded as errors)",
                    match (r, c) {
                        (true, true) => "each kernel",
                        (true, false) => "the reference",
                        _ => "the candidate",
                    }
                ),
            };
            let steps = r16["captured_steps"]
                .as_array()
                .map(|a| {
                    format!(
                        "; at the captured inputs it proves {} outputs bitwise equal, {} within 1 ulp, {} more",
                        a[0], a[1], a[2]
                    )
                })
                .unwrap_or_default();
            let sf = &eq["softmax_form"];
            format!(
                "PROVEN {level} for every input: both outputs are {fmt} roundings of values with the same exact polynomial y{inner}; |C - R| <= {:.0}u*|y| + {:.2}u*sum|terms|, and per input <= the distance between the {fmt} roundings of [y - G_R, y + G_C]{steps}",
                sf["rel_u"].as_f64().unwrap_or(0.0),
                sf["add_u"].as_f64().unwrap_or(0.0)
            )
        } else if r16.is_object() && eq["bound_form"] == json!(["norm"]) {
            let fmt = r16["format"].as_str().unwrap_or("bf16");
            let inner = match (
                r16["inner_reference"] == true,
                r16["inner_candidate"] == true,
            ) {
                (false, false) => String::new(),
                (r, c) => format!(
                    " ({} rounds to {fmt} before the output too: different functions, those roundings bounded as errors)",
                    match (r, c) {
                        (true, true) => "each kernel",
                        (true, false) => "the reference",
                        _ => "the candidate",
                    }
                ),
            };
            let steps = r16["captured_steps"]
                .as_array()
                .map(|a| {
                    format!(
                        "; at the captured inputs it proves {} outputs bitwise equal, {} within 1 ulp, {} more",
                        a[0], a[1], a[2]
                    )
                })
                .unwrap_or_default();
            format!(
                "PROVEN {level} for every input (norm form): both outputs are {fmt} roundings of the same normalization{inner}; |C - R| <= {} * T(x), and per input <= the distance between the {fmt} roundings of [y - G_R, y + G_C]{steps}",
                fmt_u(b["max_u"].as_f64().unwrap_or(0.0))
            )
        } else {
            format!("PROVEN {level} (same function) for every input, |C - R| <= {form}")
        }
    } else {
        match level.trim_end_matches(" (unproven)") {
            "none" if proven && eq["untracked"].is_null() => {
                let w = &eq["witness"];
                // A witness refutes only if its outputs differ (simulator or GB300).
                let differs = !w.is_null()
                    && (w["sim_reference"] != w["sim_candidate"]
                        || (!w["gpu_reference"].is_null()
                            && w["gpu_reference"] != w["gpu_candidate"]));
                if w.is_null() || w["leaves"].as_array().is_some_and(|a| a.is_empty()) || !differs {
                    // Differing exact forms are not a counterexample: opaque instruction
                    // slices (N-SEG) can compute the same real function differently. Only a
                    // witness input shows the functions differ.
                    "NOT PROVEN: the exact forms differ, and no witness input shows the functions differ".to_string()
                } else {
                    format!(
                        "NOT EQUIVALENT: with every input 0 except {} = 1, out[{}] is {} (reference) vs {} (candidate)",
                        w["leaves"]
                            .as_array()
                            .map(|a| a
                                .iter()
                                .filter_map(|x| x.as_str())
                                .collect::<Vec<_>>()
                                .join(", "))
                            .unwrap_or_default(),
                        w["output"],
                        w["sim_reference"],
                        w["sim_candidate"]
                    )
                }
            }
            "none" => "NOT PROVEN (not shown to be the same function)".to_string(),
            l => format!("NOT PROVEN for every input ({l} holds on the captured inputs only)"),
        }
    };
    if let Some(n) = gpu["outputs"].as_u64() {
        s += &format!(
            "; GPU: {}/{} outputs bitwise equal, max |C - R| {:.3e}",
            gpu["bitwise_equal"],
            n,
            gpu["max_abs_diff"].as_f64().unwrap_or(f64::NAN)
        );
        if let (Some(f), false) = (
            gpu["max_fraction_of_bound"].as_f64(),
            level.starts_with("L0"),
        ) {
            s += &format!(" ({f:.3} of the bound)");
        }
    }
    let scope = &eq["output_scope"];
    let unchecked = |side: &str| {
        scope[side]["unchecked"]
            .as_array()
            .is_some_and(|buffers| !buffers.is_empty())
    };
    if unchecked("reference") || unchecked("candidate") {
        s.push_str(&format!(
            "; equivalence scope: primary outputs {} and {}; other output buffers unchecked",
            scope["reference"]["checked"], scope["candidate"]["checked"]
        ));
    }

    s
}

/// The verdict classes, `contract` findings (PTX-contract deviations GB300 tolerates,
/// ASSUMPTIONS.md) shown as what a clean verdict relies on.
fn verdict_text(c: &Value) -> String {
    let mut cl: Vec<&str> = c["verdict"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
        .unwrap_or_default();
    let contract = cl.contains(&"contract");
    cl.retain(|x| *x != "contract");
    if cl.is_empty() {
        cl.push("clean");
    }
    let mut s = cl.join(",");
    if contract {
        let ids: Vec<String> = c["findings"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|f| f["kind"] == "contract")
            .filter_map(|f| {
                let m = f["msg"].as_str()?;
                let i = m.rfind("(E-")?;
                Some(m[i + 1..].trim_end_matches(')').to_string())
            })
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        s += &format!(
            " (relies on {})",
            if ids.is_empty() {
                "a contract finding".into()
            } else {
                ids.join(", ")
            }
        );
    }
    s
}

fn side_line(s: &Value) -> String {
    let side = &s["side"];
    let name = side["name"].as_str().unwrap_or("?");
    if side["lifted"] == false {
        return format!("{name}: does not lift");
    }
    let c = &side["check"];
    if let Some(e) = c["error"].as_str() {
        return format!("{name}: check failed: {}", first_line(e));
    }
    let v = verdict_text(c);
    let simgpu = side["sim_vs_gpu"]
        .as_array()
        .map(|a| {
            if a.is_empty() {
                "no captured GPU output".to_string()
            } else if a.iter().all(|l| {
                l.as_str().is_some_and(|l| {
                    l.contains("bitwise equal") && !l.contains("schedule-dependent output")
                })
            }) {
                "sim = GPU bitwise".to_string()
            } else if a
                .iter()
                .any(|l| l.as_str().is_some_and(|l| l.contains("schedule-dependent")))
            {
                "schedule-dependent output (sim vs GPU: see report)".to_string()
            } else {
                "sim != GPU".to_string()
            }
        })
        .unwrap_or_default();
    format!(
        "{name} [{}]: {v}{} ({}, {} states), {simgpu}",
        side["front"].as_str().unwrap_or(""),
        if c["complete"] == true {
            ""
        } else {
            " (INCOMPLETE)"
        },
        c["method"].as_str().unwrap_or(""),
        c["states"]
    )
}

fn summary(r: &Value) -> String {
    let mut s = String::new();
    let lab = |x: &Value| {
        x["label"]
            .as_str()
            .map(|l| format!(" ({l})"))
            .unwrap_or_default()
    };
    s += &format!(
        "reference{}: {}\n",
        lab(&r["reference"]),
        side_line(&r["reference"])
    );
    s += &format!(
        "candidate{}: {}\n",
        lab(&r["candidate"]),
        side_line(&r["candidate"])
    );
    s += &format!("=> {}\n", r["headline"].as_str().unwrap_or(""));
    let b = &r["equivalence"]["per_output_bound"];
    if b["with_bound"].as_u64().unwrap_or(0) > 0 {
        s += &format!(
            "   per-output bound: max {}, median {}, min {} ({} of {} outputs bounded)\n",
            fmt_u(b["max_u"].as_f64().unwrap_or(0.0)),
            fmt_u(b["median_u"].as_f64().unwrap_or(0.0)),
            fmt_u(b["min_u"].as_f64().unwrap_or(0.0)),
            b["with_bound"],
            b["outputs"]
        );
    }
    let rel = r["relies_on"].as_array().cloned().unwrap_or_default();
    if !rel.is_empty() {
        s += "   relies on: ";
        s += &rel
            .iter()
            .filter_map(|x| x.as_str())
            .map(|x| match x.split_once(": ") {
                Some((who, r)) => format!("{who}: {}", short_relies(r)),
                None => x.to_string(),
            })
            .collect::<Vec<_>>()
            .join("; ");
        s += "\n";
    }
    for b in r["blockers"].as_array().into_iter().flatten() {
        s += &format!("   blocker: {}\n", b.as_str().unwrap_or(""));
    }
    s
}

fn markdown(r: &Value) -> String {
    let mut s = String::new();
    s += "# ptxv compare\n\n";
    s += &format!("**{}**\n\n", r["headline"].as_str().unwrap_or(""));
    s += "| | reference | candidate |\n|---|---|---|\n";
    let row = |k: &str, f: &dyn Fn(&Value) -> String| {
        format!(
            "| {k} | {} | {} |\n",
            f(&r["reference"]),
            f(&r["candidate"])
        )
    };
    s += &row("what", &|x| {
        x["label"]
            .as_str()
            .map(|l| format!("`{l}`"))
            .unwrap_or_else(|| format!("case {}", x["side"]["name"].as_str().unwrap_or("")))
    });
    s += &row("kernels", &|x| {
        x["side"]["kernels"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|k| k.as_str())
                    .map(|k| format!("`{}`", short_kernel(k)))
                    .collect::<Vec<_>>()
                    .join("<br>")
            })
            .unwrap_or_default()
    });
    s += &row("lifts", &|x| {
        let sd = &x["side"];
        if sd["lifted"] == true {
            format!("yes ({})", sd["front"].as_str().unwrap_or(""))
        } else {
            let mut v = vec![];
            for k in sd["unlifted"].as_array().into_iter().flatten() {
                for u in k["unlifted"].as_array().into_iter().flatten() {
                    v.push(format!("`{}`", u["instruction"].as_str().unwrap_or("")));
                }
            }
            format!("no: {}", v.join(", "))
        }
    });
    s += &row("execution verdict", &|x| {
        let c = &x["side"]["check"];
        if c.is_null() {
            return "-".into();
        }
        if let Some(e) = c["error"].as_str() {
            return format!("error: {}", first_line(e));
        }
        format!(
            "{}{}, {} ({} states)",
            verdict_text(c),
            if c["complete"] == true {
                ""
            } else {
                " INCOMPLETE"
            },
            c["method"].as_str().unwrap_or(""),
            c["states"]
        )
    });
    s += &row("sim vs GPU (captured input)", &|x| {
        x["side"]["sim_vs_gpu"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|l| l.as_str())
                    .collect::<Vec<_>>()
                    .join("; ")
            })
            .filter(|t| !t.is_empty())
            .unwrap_or("-".into())
    });
    let e = &r["equivalence"];
    s += "\n## Equivalence\n\n";
    if e.is_null() {
        s += "Not attempted.\n";
    } else {
        s += &format!(
            "- level: **{}**{}\n",
            e["level"].as_str().unwrap_or(""),
            if e["proven"] == true {
                " (proven for every input)"
            } else {
                " (not proven)"
            }
        );
        for l in e["levels"].as_array().into_iter().flatten() {
            s += &format!(
                "  - {}: {}\n",
                l["level"].as_str().unwrap_or(""),
                if l["holds"] == true {
                    "holds".to_string()
                } else {
                    format!("fails at {} outputs", l["failures"])
                }
            );
        }
        let b = &e["per_output_bound"];
        if b["with_bound"].as_u64().unwrap_or(0) > 0 {
            s += &format!(
                "- per-output bound ({}): max {}, median {}, min {}; {} of {} outputs bounded\n",
                e["bound_meaning"].as_str().unwrap_or(""),
                fmt_u(b["max_u"].as_f64().unwrap_or(0.0)),
                fmt_u(b["median_u"].as_f64().unwrap_or(0.0)),
                fmt_u(b["min_u"].as_f64().unwrap_or(0.0)),
                b["with_bound"],
                b["outputs"]
            );
            if let (Some(a), Some(c)) = (
                b["reference_own_max_u"].as_f64(),
                b["candidate_own_max_u"].as_f64(),
            ) {
                s += &format!(
                    "- each kernel against the exact value (largest over outputs): reference {}, candidate {}\n",
                    fmt_u(a),
                    fmt_u(c)
                );
            }
            if let Some(sf) = e["softmax_form"].as_object() {
                s += &format!(
                    "- softmax form: |C - R| <= {:.2}u * y + {:.2}u\n",
                    sf["rel_u"].as_f64().unwrap_or(0.0),
                    sf["add_u"].as_f64().unwrap_or(0.0)
                );
            }
            s += "- histogram of the bound over outputs:\n\n  | bound (u) | outputs |\n  |---|---|\n";
            for h in b["histogram"].as_array().into_iter().flatten() {
                let (lo, hi) = (
                    h["lo"].as_f64().unwrap_or(0.0),
                    h["hi"].as_f64().unwrap_or(0.0),
                );
                let k = if lo == hi {
                    format!("{lo:.2}")
                } else {
                    format!("{lo:.2} – {hi:.2}")
                };
                s += &format!("  | {k} | {} |\n", h["count"]);
            }
            s += "\n";
        }
        if let Some(t) = e["trials"].as_object() {
            s += &format!(
                "- simulator trials: {} random inputs, {} over the bound, at most {:.4} of the bound used\n",
                t["trials"],
                t["violations"],
                t["max_used"].as_f64().unwrap_or(0.0)
            );
        }
        for k in ["unordered", "softmax"] {
            if let Some(x) = e[k].as_str() {
                s += &format!("- {k}: {x}\n");
            }
        }
        if let Some(w) = e["witness"]
            .as_object()
            .filter(|w| w["leaves"].as_array().is_some_and(|a| !a.is_empty()))
        {
            s += &format!(
                "- witness: out[{}] with every input 0 except {} = 1: reference {}, candidate {} (simulator{})\n",
                w["output"],
                w["leaves"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(|x| x.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                w["sim_reference"],
                w["sim_candidate"],
                if w["gpu_reference"].is_null() {
                    String::new()
                } else {
                    format!("; GB300: {} vs {}", w["gpu_reference"], w["gpu_candidate"])
                }
            );
        }
        if let Some(l) = e["log"].as_str() {
            s += &format!("- equiv log: `{l}`\n");
        }
    }
    s += "\n## Captured GPU outputs\n\n";
    let g = &r["gpu"];
    if g["outputs"].is_null() {
        s += if r["inputs_identical"] == false {
            "Not compared: the two cases' inputs differ.\n"
        } else {
            "Not available (no f32 captured outputs on both sides).\n"
        };
    } else {
        s += &format!(
            "- {} of {} outputs bitwise equal; max |C - R| = {:.3e} (out[{}]: {} vs {}), max relative {:.3e}, max {} ulp\n",
            g["bitwise_equal"],
            g["outputs"],
            g["max_abs_diff"].as_f64().unwrap_or(0.0),
            g["worst_output"],
            g["worst_values"][0],
            g["worst_values"][1],
            g["max_rel_diff"].as_f64().unwrap_or(0.0),
            g["max_ulps"]
        );
        if let Some(f) = g["max_fraction_of_bound"].as_f64() {
            s += &format!(
                "- largest |C - R| / (bound at the captured input): {f:.4} (out[{}])\n",
                e["captured"]["worst_output"]
            );
        }
    }
    s += "\n## Rests on\n\n";
    let rel: Vec<&str> = r["relies_on"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|x| x.as_str())
        .collect();
    if !rel.is_empty() {
        for x in rel {
            s += &format!("- {x}\n");
        }
    }
    s += &format!(
        "- execution assumptions (ASSUMPTIONS.md): {}\n",
        r["assumptions"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|x| x.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    );
    s += "\n## Not proven, and why\n\n";
    let bl: Vec<&str> = r["blockers"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|x| x.as_str())
        .collect();
    if bl.is_empty() {
        s += "Nothing: every step is proven.\n";
    }
    for b in bl {
        s += &format!("- {b}\n");
    }
    s
}

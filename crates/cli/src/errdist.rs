/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv errdist <case-dir | equiv.toml> --inputs spec.json`: the distribution of a kernel's
//! rounding error (C - exact) or of a pair's difference (C - R) under an input distribution
//! (docs/errdist.md).

use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use ptxv_sim::errdist::chain::Prog;
use ptxv_sim::errdist::spec::Format;
use ptxv_sim::errdist::spec::Layout;
use ptxv_sim::errdist::spec::Spec;

use super::equiv::EquivFile;
use super::equiv::KernelRef;
use super::equiv::Run;
use super::equiv::out_only;
use super::equiv::prepare;

pub struct Args {
    pub path: Option<PathBuf>,
    pub inputs: Option<PathBuf>,
    /// Write the compiled model (programs section, and the nonlin section when --inputs is
    /// given) here; "-": the cache, keyed by the case hash. Without --inputs, stop there.
    pub compile: Option<PathBuf>,
    /// Query a compiled model (no simulator) instead of compiling the case.
    pub model: Option<PathBuf>,
    pub json: Option<PathBuf>,
    pub html: Option<PathBuf>,
    pub candidate: usize,
    pub mc: Option<usize>,
    pub sim_trials: Option<usize>,
    pub gpu_trials: Option<usize>,
    pub per_output: bool,
    pub threads: usize,
    pub out: Option<PathBuf>,
    pub explore: bool,
    pub pre_round: bool,
    pub uniform: bool,
    /// Run Tier 1 too (opt-in).
    pub tier1: bool,
}

/// One kernel ready for errdist: its preconditions and its program.
pub(crate) struct Kernel {
    pub run: Run,
    pub prog: Prog,
    /// Agent AG: the program's identified slices.
    pub idents: ptxv_sim::errdist::nonlin::ident::Idents,
}

/// The `programs` section of an errdist model file (docs/errdist.md §1.1): what a query needs
/// without the simulator. The file is a JSON object {kind, programs, nonlin}; the `nonlin`
/// section (agent AG's slice / MUFU tables, `nonlin::Model`) sits beside it, and each writer
/// keeps the other's section.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Model {
    pub version: u32,
    /// Hash of the case files (and candidate index) the model was compiled from: the cache
    /// key (`case_key`).
    pub key: String,
    pub reference: ptxv_sim::errdist::report::KernelId,
    pub candidate: Option<ptxv_sim::errdist::report::KernelId>,
    pub r: Prog,
    pub c: Option<Prog>,
    pub layouts: Vec<Layout>,
    pub labels: Vec<String>,
    /// Agent AG: each program's identified slices (reference, candidate).
    pub idents: Vec<ptxv_sim::errdist::nonlin::ident::Idents>,
    /// The proven worst case (equivalence bound of the pair, or the kernel's own bound).
    #[serde(default)]
    pub worst: Option<ptxv_sim::errdist::report::WorstCase>,
}

impl Model {
    pub(crate) fn load(path: &Path) -> Result<Model> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|e| anyhow!("{}: {e}", path.display()))?;
        let m: Model = serde_json::from_value(v.get("programs").cloned().ok_or_else(|| {
            anyhow!(
                "{}: no programs section (compile it with --compile)",
                path.display()
            )
        })?)
        .map_err(|e| anyhow!("{}: {e}", path.display()))?;
        if m.version != MODEL_VERSION {
            bail!(
                "{}: model version {} (this ptxv reads {MODEL_VERSION}); recompile",
                path.display(),
                m.version
            );
        }
        Ok(m)
    }

    /// Write the programs section, keeping the file's other sections (the nonlin tables) when
    /// they are for the same case.
    pub(crate) fn save(&self, path: &Path) -> Result<()> {
        let mut v: serde_json::Value = std::fs::read(path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .filter(|v: &serde_json::Value| {
                v.is_object() && v["programs"]["key"] == self.key.as_str()
            })
            .unwrap_or_else(|| serde_json::json!({"kind": "ptxv-errdist-model"}));
        v["programs"] = serde_json::to_value(self)?;
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(path, serde_json::to_vec(&v)?)?;
        Ok(())
    }
}

pub(crate) const MODEL_VERSION: u32 = 2;

/// FNV-1a over the case files: every regular file of each case directory (sorted), the
/// equiv.toml, the candidate index and the ptxv version.
pub(crate) fn case_key(path: &Path, candidate: usize) -> Result<String> {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut eat = |b: &[u8]| {
        for &x in b {
            h ^= x as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    eat(env!("CARGO_PKG_VERSION").as_bytes());
    eat(&MODEL_VERSION.to_le_bytes());
    let mut dirs = vec![];
    if path.extension().is_some_and(|e| e == "toml") {
        let txt = std::fs::read_to_string(path)?;
        eat(txt.as_bytes());
        eat(&(candidate as u64).to_le_bytes());
        let base = path.parent().unwrap_or(Path::new("."));
        let f: EquivFile = toml::from_str(&txt)?;
        dirs.push(base.join(&f.reference.case));
        if let Some(c) = f.candidates.get(candidate) {
            dirs.push(base.join(&c.case));
        }
    } else {
        dirs.push(path.to_path_buf());
    }
    for d in dirs {
        let mut fs: Vec<PathBuf> = std::fs::read_dir(&d)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file())
            .collect();
        fs.sort();
        for f in fs {
            eat(f.file_name().unwrap_or_default().as_encoded_bytes());
            eat(&std::fs::read(&f)?);
        }
    }
    Ok(format!("{h:016x}"))
}

/// The model cache: $PTXV_CACHE/errdist, else ~/.cache/ptxv/errdist.
pub(crate) fn cache_path(key: &str) -> PathBuf {
    let base = std::env::var_os("PTXV_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/ptxv")
        });
    base.join("errdist").join(format!("{key}.model.json"))
}

fn load_spec(p: &Path) -> Result<Spec> {
    let s = std::fs::read_to_string(p)?;
    let v: serde_json::Value = if p.extension().is_some_and(|e| e == "toml") {
        serde_json::to_value(toml::from_str::<toml::Value>(&s)?)?
    } else {
        serde_json::from_str(&s)?
    };
    Spec::from_value(v, p.parent().unwrap_or(Path::new("."))).map_err(|e| anyhow!(e))
}

fn kernel(
    base: &Path,
    k: &KernelRef,
    max_states: usize,
    dir: &Path,
    segments: bool,
) -> Result<Kernel> {
    let mut run = prepare(base, k, max_states, dir, segments)?;
    let mut prog = Prog::extract(&run.m).map_err(|e| anyhow!("{}: {e}", run.label()))?;
    // Untracked outputs: why no slice could be built (segment mode).
    if prog.outputs.iter().any(|o| o.is_none()) {
        if let Some(ns) = run.m.num.as_ref() {
            for (pc, e) in ns.seg_fails.iter().take(4) {
                run.notes.push(format!(
                    "no slice at {}: {e}",
                    run.m.k.describe(*pc as usize)
                ));
            }
        }
    }
    // Agent AG: identify the instruction slices (non-chain outputs).
    let (ids, idents) =
        ptxv_sim::errdist::nonlin::prepare(&mut prog, &run.m).map_err(|e| anyhow!(e))?;
    run.notes.extend(ids);
    let bad = prog.check_captured(&run.m).map_err(|e| anyhow!(e))?;
    if bad > 0 && std::env::var("PTXV_ERRDIST_DEBUG").is_ok() {
        for l in ptxv_sim::errdist::nonlin::mismatches(&prog, &run.m, 8) {
            eprintln!("{l}");
        }
    }
    // Agent AG: a slice node's simulator value can be a register read at materialization
    // that differs from the slice of its arguments (intermediate nodes only); the outputs and
    // the sampled-input validation are what must agree.
    let bad = if bad > 0
        && prog
            .ops
            .iter()
            .any(|o| matches!(o, ptxv_sim::errdist::chain::Op::Nl(_)))
    {
        let (outs, ops) =
            ptxv_sim::errdist::nonlin::captured_mismatch(&prog, &run.m).map_err(|e| anyhow!(e))?;
        if outs == 0 {
            run.notes.push(format!("{ops} intermediate slice values differ from the simulator's node values at the captured input; all outputs agree"));
        }
        outs
    } else {
        bad
    };
    if bad > 0 {
        bail!(
            "{}: {bad} of {} DAG ops evaluate differently from the simulator at the captured input",
            run.label(),
            prog.ops.len()
        );
    }
    // Agent AG: the DAG's outputs at the captured input must be the captured GPU outputs; a
    // region whose capture misses work between launches (a device-to-device memcpy feeding
    // an addmm's C operand) simulates a different function.
    if let Some(g) = super::equiv::gpu_output_f32(&run.case) {
        if g.len() == prog.outputs.len() {
            let differ = prog
                .outputs
                .iter()
                .zip(&g)
                .filter(|(o, y)| {
                    o.is_some_and(|o| {
                        let h = f32::from_bits(prog.hw[o as usize]);
                        h.to_bits() != y.to_bits() && !(h.is_nan() && y.is_nan())
                    })
                })
                .count();
            if differ > 0 {
                bail!(
                    "{}: {differ} of {} outputs of the numerics DAG differ from the captured GPU outputs at the captured input (work between the captured launches missing from the case, e.g. a device-to-device memcpy?)",
                    run.label(),
                    g.len()
                );
            }
        }
    }
    Ok(Kernel { run, prog, idents })
}

/// The input buffers' layouts (byte buffers: the format the kernel reads, scale blocks).
pub(crate) fn layouts(ks: &[&Kernel]) -> Result<Vec<Layout>> {
    let mut fmts: BTreeMap<String, Format> = BTreeMap::new();
    let mut blocks: BTreeMap<String, Vec<(u64, Vec<u64>)>> = BTreeMap::new();
    for k in ks {
        fmts.extend(k.prog.leaf_formats());
        for (code, (_, b)) in k.prog.scale_blocks().map_err(|e| anyhow!(e))? {
            blocks.entry(code).or_insert(b);
        }
    }
    let case = &ks[0].run.case;
    let mut out = Vec::new();
    for b in &case.m.buffers {
        if out_only(case, &b.name) {
            continue;
        }
        // Agent AG: a byte workspace no leaf reads (written by an earlier launch of the case,
        // W0 of the multi-kernel captures) keeps its captured bytes.
        if !fmts.contains_key(&b.name) && !matches!(b.dtype.as_str(), "f32" | "bf16" | "f16") {
            continue;
        }
        let fmt = match fmts.get(&b.name) {
            Some(f) => *f,
            None => match b.dtype.as_str() {
                "f32" => Format::F32,
                "bf16" => Format::Bf16,
                "f16" => Format::F16,
                // A byte buffer no leaf reads: zero bytes.
                _ => Format::E4m3,
            },
        };
        out.push(Layout {
            name: b.name.clone(),
            dtype: b.dtype.clone(),
            shape: b.shape.clone(),
            bytes: b.bytes(),
            fmt,
            blocks: blocks.get(&b.name).cloned(),
        });
    }
    Ok(out)
}

/// Simulate the case's kernel(s). Agent AG: segment mode when some output is untracked
/// without it (softmax, norms, multi-launch regions whose launches communicate through global
/// memory).
fn simulate(
    path: &Path,
    candidate: usize,
    dir: &Path,
    force_seg: bool,
) -> Result<(Kernel, Option<Kernel>)> {
    let untracked = |k: &Kernel| k.prog.outputs.iter().any(|o| o.is_none());
    if path.extension().is_some_and(|e| e == "toml") {
        let base = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let f: EquivFile = toml::from_str(&std::fs::read_to_string(path)?)?;
        let ck = f
            .candidates
            .get(candidate)
            .ok_or_else(|| anyhow!("no candidate {candidate}"))?;
        let seg = f.segments || force_seg;
        let r = kernel(&base, &f.reference, f.max_states, dir, seg)?;
        let c = kernel(&base, ck, f.max_states, dir, seg)?;
        if !seg && (untracked(&r) || untracked(&c)) {
            let r = kernel(&base, &f.reference, f.max_states, dir, true)?;
            let c = kernel(&base, ck, f.max_states, dir, true)?;
            Ok((r, Some(c)))
        } else {
            Ok((r, Some(c)))
        }
    } else {
        let k = KernelRef {
            case: path.canonicalize()?,
            variant: "good".into(),
            front: "sass".into(),
            expect: None,
            slices: false,
            certify: true,
            expect_bound_u: None,
        };
        match kernel(Path::new("/"), &k, 200_000, dir, force_seg) {
            Ok(x) if !untracked(&x) => Ok((x, None)),
            _ => Ok((kernel(Path::new("/"), &k, 200_000, dir, true)?, None)),
        }
    }
}

/// A scratch directory removed when dropped.
struct TmpDir(PathBuf);

impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn cmd(a: &Args) -> Result<()> {
    // Scratch: --out, else a directory of this process (concurrent runs must not share one),
    // removed at the end.
    let tmp = a.out.is_none().then(|| {
        TmpDir(std::env::temp_dir().join(format!(
            "ptxv-errdist-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        )))
    });
    let dir = a
        .out
        .clone()
        .unwrap_or_else(|| tmp.as_ref().unwrap().0.clone());
    std::fs::create_dir_all(&dir)?;
    // The model file of this run: queried (--model) or written (--compile).
    let mut model_file: Option<PathBuf> = a.model.clone();
    // Query stage: a compiled model (no simulator, no validation runs).
    let (model, kernels) = if let Some(mp) = &a.model {
        let m = Model::load(mp)?;
        if let Some(path) = &a.path {
            let k = case_key(path, a.candidate)?;
            if k != m.key {
                bail!(
                    "{} was compiled from other case files (key {} != {k}); recompile",
                    mp.display(),
                    m.key
                );
            }
        }
        println!("model {} (key {})", mp.display(), m.key);
        for l in &m.labels {
            println!("{l}");
        }
        (m, None)
    } else {
        let path = a
            .path
            .as_ref()
            .ok_or_else(|| anyhow!("a case directory or equiv.toml (or --model FILE)"))?;
        let t0 = std::time::Instant::now();
        let (mut r, mut c) = simulate(path, a.candidate, &dir, false)?;
        // Never report from a graph that disagrees with the simulator: probe two sampled
        // inputs; on a mismatch, retry in segment mode, and refuse if that disagrees too.
        if let Some(ip) = &a.inputs {
            let spec = load_spec(ip)?;
            let (m, n) = dag_probe(&r, c.as_ref(), &spec, &dir)?;
            if m > 0 {
                println!(
                    "the numerics DAG disagrees with the simulator on {m} of {n} tracked outputs over 2 probe inputs: retrying in segment mode"
                );
                (r, c) = simulate(path, a.candidate, &dir, true)?;
                let (m2, n2) = dag_probe(&r, c.as_ref(), &spec, &dir)?;
                if m2 > 0 {
                    bail!(
                        "the numerics DAG disagrees with the simulator on {m2} of {n2} tracked outputs over 2 probe inputs, in segment mode too (default extraction: {m} of {n}); refusing to report from it"
                    );
                }
            }
        }
        let mut labels = vec![];
        for k in std::iter::once(&r).chain(c.iter()) {
            labels.push(k.run.label());
            println!("{}", k.run.label());
            for p in &k.run.pre {
                println!("    precondition: {p}");
            }
            for n in &k.run.notes {
                println!("    {n}");
            }
        }
        let ks: Vec<&Kernel> = std::iter::once(&r).chain(c.iter()).collect();
        if std::env::var("PTXV_ERRDIST_DEBUG").is_ok() {
            for k in &ks {
                eprintln!("leaves of {}: {:?}", k.run.label(), k.prog.leaf_formats());
            }
        }
        // Agent AG: nothing to estimate when the DAG tracks no output.
        for k in &ks {
            if !k.prog.outputs.is_empty() && k.prog.outputs.iter().all(|o| o.is_none()) {
                bail!(
                    "{}: the numerics DAG tracks none of its {} outputs{}",
                    k.run.label(),
                    k.prog.outputs.len(),
                    k.run
                        .notes
                        .iter()
                        .filter(|n| n.starts_with("no slice at"))
                        .map(|n| format!("; {n}"))
                        .collect::<String>()
                );
            }
        }
        let m = Model {
            version: MODEL_VERSION,
            key: case_key(path, a.candidate)?,
            reference: kid(&r),
            candidate: c.as_ref().map(kid),
            r: r.prog.clone(),
            c: c.as_ref().map(|c| c.prog.clone()),
            layouts: layouts(&ks)?,
            labels,
            idents: ks.iter().map(|k| k.idents.clone()).collect(),
            worst: Some(worst_case(&r, c.as_ref())),
        };
        if let Some(cp) = &a.compile {
            let out = if cp.as_os_str().is_empty() || cp.as_os_str() == "-" {
                cache_path(&m.key)
            } else {
                cp.clone()
            };
            m.save(&out)?;
            println!(
                "compiled model {} (key {}, {:.1}s)",
                out.display(),
                m.key,
                t0.elapsed().as_secs_f64()
            );
            if a.inputs.is_none() {
                return Ok(());
            }
            model_file = Some(out);
        }
        (m, Some((r, c)))
    };
    let spec = load_spec(a.inputs.as_ref().ok_or_else(|| anyhow!("--inputs spec"))?)?;
    for p in std::iter::once(&model.r).chain(model.c.iter()) {
        let tracked = p.outputs.iter().filter(|o| o.is_some()).count();
        let steps = p.chain(0).map_or(0, |c| c.0.len());
        println!(
            "    {} DAG ops, {} outputs ({tracked} tracked, {:?}), output 0: {steps} MMA steps",
            p.ops.len(),
            p.outputs.len(),
            p.out_fmt
        );
    }
    let ls = model.layouts.clone();
    let threads = if a.threads == 0 {
        std::thread::available_parallelism().map_or(8, |n| n.get())
    } else {
        a.threads
    };
    let (rp, cp) = if a.pre_round {
        (model.r.pre_round(), model.c.as_ref().map(|c| c.pre_round()))
    } else {
        (model.r.clone(), model.c.clone())
    };
    let pair = model.c.is_some();
    let mut st = ptxv_sim::errdist::mc::Setup::new(&rp, cp.as_ref()).map_err(|e| anyhow!(e))?;
    // The worst-case bound's distribution and the check of every observed error against it
    // (closed forms only; --pre-round compares other values).
    if !a.pre_round {
        st.wc = model
            .worst
            .as_ref()
            .filter(|w| w.available && matches!(w.form.as_str(), "sum" | "round" | "softmax"))
            .map(|w| (w.rel, w.k_u * 2f64.powi(-24), w.abs, w.scale_one));
    }
    let st = st;
    if a.explore {
        print!(
            "{}",
            ptxv_sim::errdist::mc::explore_steps(
                &st,
                &spec,
                &ls,
                std::env::var("PTXV_EXPLORE_N")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(50)
            )
            .map_err(|e| anyhow!(e))?
        );
        return Ok(());
    }
    // Tier 3: the cross-check of the analytic tiers and the fallback headline (--mc 0 skips
    // it; a model query then rests on the structural checks).
    let n = a.mc.or(spec.engine.mc_inputs).unwrap_or(MC_DEFAULT);
    let t0 = std::time::Instant::now();
    let obs = ptxv_sim::errdist::mc::run(
        &st,
        &spec,
        &ls,
        &ptxv_sim::errdist::mc::McCfg {
            n,
            first: 0,
            threads,
            keep: 256,
            per_output: a.per_output || spec.engine.per_output,
        },
    )
    .map_err(|e| anyhow!(e))?;
    if std::env::var("PTXV_ERRDIST_DEBUG").is_ok() {
        let g = ptxv_sim::errdist::mc::DBG.lock().unwrap();
        let tot: f64 = g.values().map(|r| r[0]).sum();
        for (b, r) in g.iter() {
            eprintln!(
                "mc b {b:>4}: p {:.4e} P0 {:.4} E|e| {:.3} sd {:.3}",
                r[0] / tot,
                r[1] / r[0],
                r[2] / r[0],
                (r[3] / r[0]).sqrt()
            );
        }
    }
    let mut est = ptxv_sim::errdist::mc::estimate(
        &obs,
        3,
        "Monte Carlo through the numerics DAG",
        model.r.out_fmt,
    );
    est.seconds = t0.elapsed().as_secs_f64();
    let mut rep = ptxv_sim::errdist::report::Report {
        version: ptxv_sim::errdist::report::REPORT_VERSION,
        kind: "ptxv-errdist".into(),
        mode: if pair { "pair" } else { "single" }.into(),
        quantity: if pair { "C - R" } else { "C - exact" }.into(),
        reference: model.reference.clone(),
        candidate: model.candidate.clone(),
        spec: serde_json::to_value(&spec)?,
        outputs: model.r.outputs.len() as u64,
        output_format: format!("{:?}", model.r.out_fmt).to_lowercase(),
        templates: 1,
        ..Default::default()
    };
    rep.out_of_domain = ood(&obs);
    if obs.nonfinite > 0 {
        rep.notes.push(format!(
            "{} of {} outputs ({:.3e}) are NaN / Inf in Tier 3 ({} of {} inputs): out of domain, outside the error statistics",
            obs.nonfinite,
            obs.seen,
            obs.nonfinite as f64 / obs.seen.max(1) as f64,
            obs.nonfinite_inputs,
            obs.inputs
        ));
    }
    if n > 0 && obs.compared == 0 {
        rep.notes.push(
            "TIER 3 COMPARED NO OUTPUTS: every output is non-finite, untracked or has no nonzero term; its P(differ) says nothing".into(),
        );
    }
    rep.worst_case = model.worst.clone().map(|mut w| {
        if st.wc.is_some() && obs.wc_checked > 0 {
            let mut v = obs.wc_sample.clone();
            v.sort_by(f64::total_cmp);
            w.dist = Some(ptxv_sim::errdist::report::WorstDist {
                n: obs.wc_n,
                max_ulp: obs.wc_max,
                mean_ulp: obs.wc_sum / obs.wc_n.max(1) as f64,
                quantiles_ulp: [0.5, 0.9, 0.99, 0.999]
                    .iter()
                    .filter(|_| !v.is_empty())
                    .map(|&q| (q, v[((q * v.len() as f64) as usize).min(v.len() - 1)]))
                    .collect(),
                tiny: obs.wc_tiny,
            });
            w.checks.push(wc_check("tier 3 (DAG evaluation)", &obs));
        }
        w
    });
    if n > 0 {
        rep.estimates.push(est);
    } else {
        rep.notes
            .push("tier 3 not run (--mc 0): no cross-check of the analytic tiers".into());
    }
    let ks: Vec<&Kernel> = kernels
        .as_ref()
        .map(|(r, c)| std::iter::once(r).chain(c.iter()).collect())
        .unwrap_or_default();
    // Validation inputs come from another index range than the prediction's.
    let first = 1u64 << 40;
    let n_sim = if ks.is_empty() {
        0
    } else {
        a.sim_trials.or(spec.engine.sim_trials).unwrap_or(4)
    };
    let mut observed: Vec<(String, Observed, Option<u32>)> = vec![];
    if n_sim > 0 {
        let t1 = std::time::Instant::now();
        let v = validate_sim(&st, &spec, &ls, &ks, n_sim, first, threads.min(n_sim), &dir)?;
        if v.mismatch > 0 && !a.pre_round {
            bail!(
                "the numerics DAG disagrees with the simulator on {} tracked outputs over {} validation inputs; refusing to report from it",
                v.mismatch,
                v.inputs
            );
        }
        let mut val =
            ptxv_sim::errdist::validate::against_sample("sim", "tier 3", &obs, &v.obs, v.mismatch);
        flag_nonfinite(&mut val, &obs, &v.obs);
        val.notes
            .push(format!("{:.1}s", t1.elapsed().as_secs_f64()));
        rep.validation.push(val);
        observed.push(("sim".into(), v, None));
    }
    let n_gpu = if ks.is_empty() {
        0
    } else {
        a.gpu_trials.or(spec.engine.gpu_trials).unwrap_or(0)
    };
    if n_gpu > 0 {
        let t1 = std::time::Instant::now();
        match validate_gpu(&st, &spec, &ls, &ks, n_gpu, first + (1 << 20), &dir) {
            Ok((v, gpu)) => {
                let mut val = ptxv_sim::errdist::validate::against_sample(
                    "gpu", "tier 3", &obs, &v.obs, v.mismatch,
                );
                flag_nonfinite(&mut val, &obs, &v.obs);
                val.notes
                    .push(format!("GPU {gpu}, {:.1}s", t1.elapsed().as_secs_f64()));
                rep.validation.push(val);
                observed.push(("gpu".into(), v, Some(gpu)));
            }
            Err(e) => rep.notes.push(format!("GPU validation not run: {e}")),
        }
    }
    if let Some(w) = rep.worst_case.as_mut().filter(|_| st.wc.is_some()) {
        for (src, v, gpu) in &observed {
            let lab = match gpu {
                Some(g) => format!("{src} (GPU {g})"),
                None => src.clone(),
            };
            w.checks.push(wc_check(&lab, &v.obs));
        }
        if w.checks.iter().any(|c| c.exceed > 0) {
            rep.notes.push("WORST CASE EXCEEDED: an observed difference is larger than the proven bound (see worst_case.checks)".into());
        }
    }
    // A preliminary report before the analytic tiers (tier 1 can take minutes).
    if let Some(p) = &a.json {
        let mut pre = serde_json::to_value(&rep)?;
        if let Some(n) = pre["notes"].as_array_mut() {
            n.push("preliminary: analytic tiers still running".into())
        }
        std::fs::write(p, serde_json::to_string_pretty(&pre)?)?;
    }
    // The analytic tiers. Chain programs: tiers 0-2 of the chains (and the DAG CLT). Agent
    // AG: non-chain programs (softmax, norms, attention, elementwise) also take the
    // linearized tiers of `nonlin`, from the model file's nonlin section.
    let nonchain = st.chain.iter().all(|c| c.is_none());
    let mut anas = vec![];
    if nonchain {
        let mut ids = vec![(st.r, &model.idents[0])];
        if let (Some(cp), Some(ci)) = (st.c, model.idents.get(1)) {
            ids.push((cp, ci));
        }
        anas.push(
            nl_query(&ids, st.r, st.c, &spec, &ls, model_file.as_deref()).map_err(|e| anyhow!(e)),
        );
    }
    anas.push(analytic(&st, &spec, &ls, false, !a.tier1));
    for ana in anas {
        match ana {
            Ok(mut e) => {
                rep.notes.extend(std::mem::take(&mut e.1));
                // Tier 1 (the chains' residue chain, the non-chain exact-site convolution) is
                // opt-in.
                let n1 = e.0.iter().filter(|x| x.tier == 1).count();
                if !a.tier1 && n1 > 0 {
                    e.0.retain(|x| x.tier != 1);
                    rep.notes.push("tier 1 not run (opt-in: --tier1)".into());
                }
                rep.estimates.extend(e.0);
                rep.residue.extend(e.2);
            }
            Err(e) => rep.notes.push(format!("analytic tiers not run: {e}")),
        }
    }
    // The Tier 3 cross-check (docs/errdist.md §3.2): every analytic estimate (tiers 1-2,
    // chain and non-chain) must agree with this run's Tier 3, |p - p3| <= max(5% of p3,
    // half Tier 3's interval) or p inside it; one that does not is not applicable and the
    // headline falls back. Without Tier 3 (a model query without --mc), the non-chain
    // estimates (validated only through this check) are not applicable and the chain ones
    // rest on their structural checks.
    let t3 = rep
        .estimates
        .iter()
        .find(|e| e.tier == 3)
        .and_then(|e| e.p_differ);
    let mut disagree = vec![];
    for (k, e) in rep.estimates.iter_mut().enumerate() {
        if !(e.tier == 1 || e.tier == 2) {
            continue;
        }
        let nl = e.method.starts_with("elementwise") || e.method.starts_with("linearized");
        let Some((p, _, _)) = e.p_differ else {
            continue;
        };
        match t3 {
            Some((p3, lo, hi)) => {
                let tol = (0.05 * p3).max(0.5 * (hi - lo));
                let ok = (p - p3).abs() <= tol || (p >= lo && p <= hi);
                e.applicability.push(format!(
                    "Tier 3 cross-check ({n} inputs): {p:.3e} vs {p3:.3e} [{lo:.3e}, {hi:.3e}], tolerance {tol:.2e} {}",
                    if ok { "ok" } else { "FAILS" }
                ));
                if !ok {
                    disagree.push(format!(
                        "tier {} ({:.40}...) {p:.3e} vs tier 3 {p3:.3e} [{lo:.3e}, {hi:.3e}]",
                        e.tier, e.method
                    ));
                }
                e.applicable = Some(e.applicable.unwrap_or(true) && ok);
                rep.crosscheck.push(ptxv_sim::errdist::report::CrossCheck {
                    estimate: k,
                    tier: e.tier,
                    p,
                    p3,
                    p3_lo: lo,
                    p3_hi: hi,
                    tol,
                    inputs: n as u64,
                    agrees: ok,
                });
            }
            None => {
                if nl || e.applicable.is_none() {
                    e.applicable = Some(false);
                    e.applicability
                        .push("no Tier 3 to cross-check against (--mc N) FAILS".into());
                } else {
                    e.applicability
                        .push("no Tier 3 cross-check (--mc N): structural checks only".into());
                }
            }
        }
    }
    for d in disagree {
        rep.notes.push(format!(
            "TIER 3 CROSS-CHECK FAILS: {d}; the headline falls back to tier 3"
        ));
    }
    if a.uniform {
        match analytic(&st, &spec, &ls, true, false) {
            Ok(e) => {
                for mut x in e.0.into_iter().filter(|x| x.tier == 1) {
                    x.method += " [uniform-residue heuristic]";
                    rep.estimates.push(x);
                }
            }
            Err(e) => rep.notes.push(format!("uniform heuristic not run: {e}")),
        }
    }
    // Every tier's PMF on the output-ulp axis: Tier 0's exact zero is {0: 1}; a tier that
    // has none says so.
    for e in rep.estimates.iter_mut().filter(|e| e.pmf_ulp.is_empty()) {
        if e.tier == 0 && e.p_differ.is_some_and(|p| p.0 == 0.0) {
            e.pmf_ulp = vec![(0, 1.0)];
        } else {
            e.error_bar.notes.push(
                "no pmf_ulp: this tier's law is not on the output-ulp lattice (P(differ) and the moments stand)".into(),
            );
        }
    }
    for (src, v, gpu) in &observed {
        for e in rep.estimates.iter().filter(|e| e.tier < 3) {
            let mut x = ptxv_sim::errdist::validate::against_estimate(src, e, &v.obs, v.mismatch);
            flag_nonfinite(&mut x, &obs, &v.obs);
            if let Some(g) = gpu {
                x.notes.push(format!("GPU {g}"));
            }
            rep.validation.push(x);
        }
    }
    // The headline: tier 0 when it applies, else tier 1 (opt-in) or tier 2 where they pass
    // their applicability checks and the Tier 3 cross-check, else tier 3.
    let ok = |e: &ptxv_sim::errdist::report::Estimate| {
        e.applicable == Some(true) && e.p_differ.is_some()
    };
    rep.best = rep
        .estimates
        .iter()
        .position(|e| e.tier == 0)
        .or_else(|| rep.estimates.iter().position(|e| e.tier == 1 && ok(e)))
        .or_else(|| rep.estimates.iter().position(|e| e.tier == 2 && ok(e)))
        .or_else(|| rep.estimates.iter().position(|e| e.tier == 3))
        .unwrap_or(0);
    rep.assumptions = vec![
        "N-ERRDIST-IID".into(),
        "N-ERRDIST-MC".into(),
        "N-EQUIV".into(),
    ];
    // Agent AG: a non-chain program's estimates rest on the nonlin assumptions.
    for e in rep.estimates.iter().filter(|_| nonchain) {
        let mut ids = vec!["N-ERRDIST-IDENT"];
        if !pair {
            ids.push("N-ERRDIST-REAL");
        }
        if e.method.starts_with("elementwise") {
            ids.push("N-ERRDIST-NL-ELEM");
        } else if e.method.starts_with("linearized") {
            ids.extend(["N-ERRDIST-NL-LIN", "N-ERRDIST-NL-INDEP"]);
            if pair {
                ids.push("N-ERRDIST-NL-JOINT");
            }
        }
        for id in ids {
            if !rep.assumptions.iter().any(|a| a == id) {
                rep.assumptions.push(id.to_string());
            }
        }
    }
    for e in rep.estimates.iter().filter(|_| !nonchain) {
        let ids: &[&str] = match e.tier {
            1 => &[
                "N-ERRDIST-TEMPLATE",
                "N-ERRDIST-RESIDUE",
                "N-ERRDIST-MARKOV",
                "N-ERRDIST-CF",
                "N-ERRDIST-FLIP",
            ],
            2 => &["N-ERRDIST-TEMPLATE", "N-ERRDIST-CLT", "N-ERRDIST-FLIP"],
            0 => &["N-ERRDIST-TEMPLATE"],
            _ => &[],
        };
        for id in ids.iter().chain(if pair && (e.tier == 1 || e.tier == 2) {
            &["N-ERRDIST-PAIR"][..]
        } else {
            &[]
        }) {
            if !rep.assumptions.iter().any(|a| a == id) {
                rep.assumptions.push(id.to_string());
            }
        }
    }
    if !rep.validation.iter().any(|v| v.source == "gpu") {
        rep.notes
            .push("no GPU validation in this run (--gpu-trials N)".into());
    }
    let text = report_text(&rep);
    print!("{text}");
    if let Some(p) = &a.json {
        std::fs::write(p, serde_json::to_string_pretty(&rep)?)?;
        println!("wrote {}", p.display());
    }
    if let Some(p) = &a.html {
        std::fs::write(p, crate::errdist_html::render(&rep, &text))?;
        println!("wrote {}", p.display());
    }
    Ok(())
}

/// NaN / Inf outputs in a validation run: noted with Tier 3's rate; the validation fails when
/// the observed run has them and Tier 3 (the model) has none, or none of the observed
/// outputs could be compared.
fn flag_nonfinite(
    v: &mut ptxv_sim::errdist::report::Validation,
    t3: &ptxv_sim::errdist::mc::Obs,
    o: &ptxv_sim::errdist::mc::Obs,
) {
    if o.nonfinite == 0 && o.compared > 0 {
        return;
    }
    let fr = |x: &ptxv_sim::errdist::mc::Obs| x.nonfinite as f64 / x.seen.max(1) as f64;
    v.notes.push(format!(
        "non-finite outputs (NaN / Inf): observed {} of {} ({:.3e}), Tier 3 {:.3e}; out of domain, outside the compared statistics",
        o.nonfinite,
        o.seen,
        fr(o),
        fr(t3)
    ));
    if (o.nonfinite > 0 && t3.nonfinite == 0) || o.compared == 0 {
        v.agree = false;
        v.notes.push(if o.compared == 0 {
            "DISAGREE: no observed output could be compared".to_string()
        } else {
            "DISAGREE: the observed run has NaN / Inf outputs the model does not account for"
                .to_string()
        });
    }
}

fn wc_check(source: &str, o: &ptxv_sim::errdist::mc::Obs) -> ptxv_sim::errdist::report::WorstCheck {
    ptxv_sim::errdist::report::WorstCheck {
        source: source.into(),
        checked: o.wc_checked,
        exceed: o.wc_exceed,
        max_ratio: o.wc_ratio,
    }
}

/// Tier 3 inputs when compiling a case without --mc: the cross-check of the analytic tiers
/// and the fallback headline when they refuse.
const MC_DEFAULT: usize = 300;

/// Above this many monomials summed over the DAG's polynomial forms (`proof_work`), errdist
/// skips the worst-case proof: building them is quadratic in the chain length (about 5e6
/// monomials a second: a K = 960 bf16 GEMM pair, 2.5e8, takes 45 s; K = 8192 takes hours), and
/// the estimate tiers do not need it.
const WORST_MAX_WORK: f64 = 1e9;

/// The proof's work: per op, the monomials of its polynomial form (the products below it,
/// counted once per op as `equiv::Side` memoizes them), summed over the ops.
fn proof_work(p: &Prog) -> f64 {
    use ptxv_sim::errdist::chain::Op;
    let mut m = vec![0f64; p.ops.len()];
    for (i, op) in p.ops.iter().enumerate() {
        let at = |j: &u32| m[*j as usize];
        m[i] = match op {
            Op::Mma { acc, terms, .. } => terms.len() as f64 + acc.as_ref().map_or(0.0, at),
            Op::Round { arg, .. } | Op::Tf32 { arg, .. } => at(arg),
            Op::Add { a, b, .. } => at(a) + at(b),
            Op::Fma { a, b, c, .. } => at(a) * at(b) + c.as_ref().map_or(0.0, at),
            _ => 1.0,
        };
    }
    m.iter().sum()
}

/// The proven worst case: a pair's equivalence bound (`equiv::compare`, as `ptxv equiv`), a
/// single kernel's own bound against the exact value (`equiv::own_bound`); largest
/// coefficients over the outputs, as a formula and in words.
fn worst_case(r: &Kernel, c: Option<&Kernel>) -> ptxv_sim::errdist::report::WorstCase {
    use ptxv_sim::errdist::report::WorstCase;
    let u = 2f64.powi(-24);
    let mut names: Vec<String> = r
        .run
        .m
        .launch
        .allocs
        .iter()
        .chain(c.iter().flat_map(|c| c.run.m.launch.allocs.iter()))
        .map(|a| a.name.clone())
        .collect();
    names.sort();
    names.dedup();
    let sr = super::equiv::side(&r.run, &names);
    let mut w = WorstCase {
        kind: if c.is_some() { "pair" } else { "single" }.into(),
        outputs: sr.outs.len(),
        ..Default::default()
    };
    for (who, k) in std::iter::once(("reference", r)).chain(c.map(|c| ("candidate", c))) {
        w.preconditions
            .extend(k.run.pre.iter().map(|p| format!("{who}: {p}")));
    }
    let mut forms: std::collections::BTreeSet<&str> = Default::default();
    let mut why: Option<String> = None;
    let work = proof_work(&r.prog) + c.map_or(0.0, |c| proof_work(&c.prog));
    if work > WORST_MAX_WORK {
        why = Some(format!(
            "proof skipped: its polynomial forms hold about {work:.1e} monomials over the DAG \
             (limit {WORST_MAX_WORK:.0e}; `ptxv equiv` runs it without the limit)"
        ));
    }
    match c {
        _ if why.is_some() => {}
        Some(c) => {
            let sc = super::equiv::side(&c.run, &names);
            let rep = ptxv_sim::equiv::compare(&sr, &sc, &names, None);
            if let Some(u) = &rep.untracked {
                why = Some(format!("untracked: {u}"));
            }
            w.preconditions.extend(rep.conditions.iter().cloned());
            if let Some(o) = &rep.unordered {
                w.preconditions.push(format!("rounding order: {o}"));
            }
            let rank = |l: &str| match l {
                "L0" => 0,
                "L2" => 1,
                "L1" | "L1~" => 2,
                _ => 3,
            };
            let mut worst_level = "L0";
            for o in &rep.per_output {
                if rank(o.level) > rank(worst_level) {
                    worst_level = o.level;
                }
                let Some(b) = o.bound_u else {
                    if why.is_none() {
                        why = Some(format!(
                            "out[{}]: {}",
                            o.index,
                            o.no_bound
                                .clone()
                                .unwrap_or_else(|| format!("level {}", o.level))
                        ));
                    }
                    continue;
                };
                w.bounded += 1;
                forms.insert(o.form);
                w.abs = w.abs.max(o.abs_extra);
                match o.form {
                    "round" => {
                        w.rel = w.rel.max(o.round_rel.unwrap_or(0.0));
                        w.k_u = w.k_u.max(o.add_u.unwrap_or(b));
                    }
                    // |C - R| <= rel_u u y + add_u u + abs_extra, y the exact output.
                    "softmax" => {
                        w.rel = w.rel.max(o.rel_u.unwrap_or(0.0) * u);
                        w.k_u = w.k_u.max(o.add_u.unwrap_or(b));
                        w.scale_one = true;
                    }
                    _ => w.k_u = w.k_u.max(b),
                }
            }
            if why.is_none() && worst_level == "none" {
                why = rep
                    .levels
                    .iter()
                    .find(|l| l.level.starts_with("L1") && !l.holds)
                    .and_then(|l| l.detail.clone())
                    .or(Some(
                        "the kernels compute different functions (not L1)".into(),
                    ));
            }
            w.level = Some(worst_level.into());
        }
        None => {
            for i in 0..sr.outs.len() {
                match ptxv_sim::equiv::own_bound(&sr, i) {
                    Ok((rel, k, ab, form, pre)) => {
                        w.bounded += 1;
                        forms.insert(form);
                        w.rel = w.rel.max(rel);
                        w.k_u = w.k_u.max(k);
                        w.abs = w.abs.max(ab);
                        for p in pre {
                            if !w.preconditions.contains(&p) {
                                w.preconditions.push(p);
                            }
                        }
                    }
                    // A constant zero output is exact.
                    Err(e) if e.starts_with("a constant zero") => w.bounded += 1,
                    Err(e) => {
                        if why.is_none() {
                            why = Some(format!("out[{i}]: {e}"));
                        }
                    }
                }
            }
        }
    }
    w.form = match forms.len() {
        0 => "none".into(),
        1 => forms.iter().next().unwrap().to_string(),
        _ if forms.iter().all(|f| *f == "sum" || *f == "round") => "round".into(),
        _ => forms.iter().cloned().collect::<Vec<_>>().join("+"),
    };
    w.available = why.is_none() && w.bounded == w.outputs && w.outputs > 0;
    w.why = why;
    w.proven = w.available && w.preconditions.is_empty();
    let lhs = if c.is_some() { "|C - R|" } else { "|C - y|" };
    let fmt = r.prog.out_fmt;
    if !w.available {
        w.formula = format!("{lhs}: no proven bound");
        w.human = format!(
            "no worst-case bound: {}",
            w.why
                .clone()
                .unwrap_or_else(|| "not every output has one".into())
        );
        return w;
    }
    if w.form == "softmax" {
        w.formula = format!(
            "{lhs} <= {:.2}u*y + {:.2}u{} (u = 2^-24, y the exact softmax output)",
            w.rel / u,
            w.k_u,
            if w.abs > 0.0 {
                format!(" + {:.3e}", w.abs)
            } else {
                String::new()
            }
        );
        w.human = format!(
            "{lhs} <= {:.2e} x y + {:.2e} (y the exact softmax output)",
            w.rel,
            w.k_u * u + w.abs
        );
    } else if w.form == "sum" || w.form == "round" {
        let mut f = format!("{lhs} <= ");
        let mut h = format!("{lhs} <= ");
        let mut first = true;
        if w.rel > 0.0 {
            f += &format!("{:.0}u*|y|", w.rel / u);
            let u16_ = 2f64.powi(-(fmt.mbits() + 1));
            let n = w.rel / u16_;
            h += &format!(
                "{} {} ulp{} of the exact result",
                fmt_num(n),
                format!("{fmt:?}").to_lowercase(),
                if (n - 1.0).abs() < 1e-9 { "" } else { "s" }
            );
            first = false;
        }
        if w.k_u > 0.0 || first {
            if !first {
                f += " + ";
                h += " + ";
            }
            f += &format!("{:.2}u*sum|terms|", w.k_u);
            h += &format!("{:.2e} x sum|a_k*b_k|", w.k_u * u);
        }
        if w.abs > 0.0 {
            f += &format!(" + {:.3e}", w.abs);
            h += &format!(" + {:.1e}", w.abs);
        }
        f += " (u = 2^-24, y the exact output, sum|terms| = sum over the output's products of |a_k*b_k|)";
        w.formula = f;
        w.human = h;
    } else {
        w.formula = format!(
            "{lhs} <= {:.2}u * scale(x) ({} form: see ptxv equiv)",
            w.k_u, w.form
        );
        w.human = format!(
            "{lhs} <= {:.2e} x scale(x) ({} form; its distribution is not evaluated here)",
            w.k_u * u,
            w.form
        );
    }
    w
}

/// A coefficient in words: integers as integers, else 3 significant digits.
fn fmt_num(x: f64) -> String {
    if (x - x.round()).abs() < 1e-9 * x.abs().max(1.0) {
        format!("{:.0}", x)
    } else {
        format!("{x:.3}")
    }
}

/// Agent AG: the non-chain analytic tiers from the model file's nonlin section (compiled when
/// absent, extended to the spec's binades, and written back to the model file).
type NlOut = (
    Vec<ptxv_sim::errdist::report::Estimate>,
    Vec<String>,
    Vec<ptxv_sim::errdist::report::ResidueSite>,
);
fn nl_query(
    ids: &[(&Prog, &ptxv_sim::errdist::nonlin::ident::Idents)],
    r: &Prog,
    c: Option<&Prog>,
    spec: &Spec,
    ls: &[ptxv_sim::errdist::spec::Layout],
    mf: Option<&Path>,
) -> Result<NlOut, String> {
    use ptxv_sim::errdist::nonlin;
    let mut notes = vec![];
    let ps: Vec<&Prog> = ids.iter().map(|x| x.0).collect();
    let key = nonlin::model_key(&ps);
    let loaded = match mf.filter(|f| f.exists()) {
        Some(f) => match nonlin::Model::load(f) {
            Ok(m) if m.key == key => {
                notes.push(format!(
                    "non-chain model {} loaded ({} tables)",
                    f.display(),
                    m.tables.tables.len()
                ));
                Some(m)
            }
            Ok(m) => {
                notes.push(format!(
                    "non-chain model {} is for other code (key {} != {key}): recompiled",
                    f.display(),
                    m.key
                ));
                None
            }
            Err(e) => {
                // A model file without the section yet (compiled before any query) is the
                // normal first query, not a failure.
                if !e.contains("no nonlin section") {
                    notes.push(format!("non-chain model not loaded ({e}): recompiled"));
                }
                None
            }
        },
        None => None,
    };
    let fresh = loaded.is_none();
    let mut m = match loaded {
        Some(m) => m,
        None => {
            let m = nonlin::Model::compile(ids, spec, ls)?;
            notes.push(format!(
                "non-chain model compiled: {} slice / MUFU tables in {:.1}s",
                m.tables.tables.len(),
                m.tables.seconds
            ));
            m
        }
    };
    let t0 = std::time::Instant::now();
    let added = nonlin::ensure(&mut m.tables, ids, spec, ls, 48)?;
    if added > 0 {
        notes.push(format!(
            "non-chain model extended by {added} binades for this input distribution ({:.1}s)",
            t0.elapsed().as_secs_f64()
        ));
    }
    // The model file keeps the tables (compiled or extended here) for the next query.
    if let Some(f) = mf.filter(|_| fresh || added > 0) {
        m.save(f)?;
        notes.push(format!("non-chain model written to {}", f.display()));
    }
    let t1 = std::time::Instant::now();
    let (e, n) = nonlin::analytic(r, c, &m.tables, spec, ls, r.out_fmt);
    notes.extend(n);
    notes.push(format!(
        "non-chain query: {:.1}s",
        t1.elapsed().as_secs_f64()
    ));
    Ok((e, notes, vec![]))
}

/// Tier 0 (when it applies) and Tier 1 estimates (and notes) for the setup.
// ASSUME(N-ERRDIST-TEMPLATE): output 0's chain stands for every output.
/// Worker threads for the analytic tiers (PTXV_ERRDIST_THREADS, else up to 48).
fn t1_threads() -> usize {
    std::env::var("PTXV_ERRDIST_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map_or(8, |n| n.get())
                .min(48)
        })
}

fn analytic(
    st: &ptxv_sim::errdist::mc::Setup,
    spec: &Spec,
    _ls: &[Layout],
    uniform: bool,
    no_tier1: bool,
) -> Result<(
    Vec<ptxv_sim::errdist::report::Estimate>,
    Vec<String>,
    Vec<ptxv_sim::errdist::report::ResidueSite>,
)> {
    use ptxv_sim::errdist::*;
    let mut fmts = st.r.leaf_formats();
    if let Some(c) = st.c {
        fmts.extend(c.leaf_formats());
    }
    // Outputs that are not one MMA chain (add / FMA / atomic DAGs, TF32 splits): the DAG CLT.
    let chain_ok = |p: &ptxv_sim::errdist::chain::Prog,
                    l: &Result<schema::Laws, String>|
     -> Result<(), String> {
        let l = l.as_ref().map_err(|e| e.clone())?;
        schema::build(p, 0, l).map(|_| ())
    };
    let laws_r = schema::Laws::build(spec, &fmts);
    let why = chain_ok(st.r, &laws_r)
        .err()
        .or_else(|| st.c.and_then(|c| chain_ok(c, &laws_r).err()));
    // Tier 2 on the output DAG (any signed sum of products without block scales).
    let dag_t2 = if uniform {
        Err(anyhow!("uniform heuristic"))
    } else {
        let t2 = std::time::Instant::now();
        tier2::leaf_laws(spec, &fmts)
            .and_then(|ll| {
                tier2::dag_estimate(
                    st.r,
                    st.c,
                    &ll,
                    std::env::var("PTXV_T2_OUTS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(64),
                )
            })
            .map(|mut e| {
                e.outputs = st.r.outputs.len() as u64;
                e.seconds = t2.elapsed().as_secs_f64();
                e
            })
            .map_err(|e| anyhow!("tier 2: {e}"))
    };
    if let Some(why) = why {
        if uniform {
            return Err(anyhow!("not an MMA chain: {why}"));
        }
        let mut notes = vec![format!(
            "tier 1 not run (not one MMA chain: {why}); tier 2 on the output DAG"
        )];
        let e = dag_t2?;
        notes.extend(e.applicability.iter().cloned());
        return Ok((vec![e], notes, vec![]));
    }
    let laws = laws_r.map_err(|e| anyhow!(e))?;
    let mut notes = laws.notes.clone();
    // Templates: outputs with the same chain shape.
    let shapes: std::collections::BTreeSet<_> = (0..st.r.outputs.len())
        .filter_map(|o| schema::shape_key(st.r, o))
        .collect();
    if shapes.len() > 1 {
        notes.push(format!(
            "{} chain shapes; tier 1 analyzes output 0's",
            shapes.len()
        ));
    }
    let opts = markov::Opts { uniform };
    let sr = schema::build(st.r, 0, &laws).map_err(|e| anyhow!("reference: {e}"))?;
    let sc = match st.c {
        Some(c) => Some(schema::build(c, 0, &laws).map_err(|e| anyhow!("candidate: {e}"))?),
        None => None,
    };
    let mut out = vec![];
    // Tier 0: the certificate, else (single kernels, small supports) exact enumeration.
    let certs: Vec<tier0::Cert> = std::iter::once(&sr)
        .chain(sc.iter())
        .map(tier0::certify)
        .collect();
    if !uniform {
        let t = std::time::Instant::now();
        let ident = st.c.map(|c| tier0::identical_outputs(st.r, c));
        let e0 = if let Some((same, n)) = ident.filter(|x| x.0 == x.1 && x.1 > 0) {
            let mut e = tier0::certified(st.r.out_fmt, true, &[]);
            e.method = format!(
                "identical programs: all {n} outputs have equal canonical DAGs (same products per step, same models; x + 0 folded), so C = R up to the sign of a zero result"
            );
            let _ = same;
            Some(e)
        } else if certs.iter().all(|c| c.exact) {
            Some(tier0::certified(st.r.out_fmt, sc.is_some(), &certs))
        } else if sc.is_none() {
            match tier0::estimate(&sr, st.r.out_fmt, 1 << 18) {
                Ok(e) => Some(e),
                Err(e) => {
                    notes.push(format!("tier 0 enumeration not run: {e}"));
                    None
                }
            }
        } else {
            None
        };
        if let Some(mut e) = e0 {
            e.outputs = st.r.outputs.len() as u64;
            e.seconds = t.elapsed().as_secs_f64();
            out.push(e);
        }
        if let Some((same, n)) = ident.filter(|x| x.0 < x.1) {
            notes.push(format!("{same} of {n} outputs have identical canonical DAGs in both kernels (the pair model treats the kernels' deviations as independent given y's binade, N-ERRDIST-PAIR)"));
        }
        for (k, c) in certs.iter().enumerate() {
            if let Some(s) = c.first_inexact {
                notes.push(format!("tier 0 certificate fails for kernel {k} at step {s} (lattice 2^{}, |partial sums| <= {:.3e})", c.lattice, c.bound));
            }
        }
    }
    if let Some(c) = st.c {
        let (same, n) = tier0::identical_outputs(st.r, c);
        if same == n && n > 0 {
            notes.push("tier 1 not run: the kernels are identical programs (tier 0)".into());
            return Ok((out, notes, vec![]));
        }
    }
    // Size guard: term atoms per step summed over the scale parts (16k ue4m3 scale pairs x
    // 4k-atom unit laws is beyond the analytic tiers' memory).
    let atoms = std::iter::once(&sr)
        .chain(sc.iter())
        .flat_map(|s| s.steps.iter())
        .map(|st| {
            st.pop
                .groups
                .iter()
                .map(|g| g.parts.iter().map(|p| p.1.atoms.len()).sum::<usize>())
                .sum::<usize>()
        })
        .max()
        .unwrap_or(0);
    if atoms > 5_000_000 {
        notes.push(format!(
            "tiers 1-2 not run: {atoms} term atoms in a step (> 5e6); tier 3 only"
        ));
        return Ok((out, notes, vec![]));
    }
    // Tier 2 assumes uniform residues; a tier 0 result (exact) supersedes it.
    if !uniform && out.iter().any(|e| e.tier == 0) {
        notes.push(
            "tier 2 not run: tier 0 is exact here (its uniform-residue model would be wrong)"
                .into(),
        );
    } else if !uniform {
        match dag_t2 {
            Ok(e) => out.push(e),
            Err(why) => {
                // Block-scaled chains: the chain CLT (per-step residue-exact local moments).
                notes.push(format!("tier 2 on the chain schema ({why})"));
                let t2 = std::time::Instant::now();
                let mut e = match &sc {
                    None => tier2::single(&sr, st.r.out_fmt, t1_threads()),
                    Some(sc) => tier2::pair(&sr, sc, st.r.out_fmt, t1_threads()),
                };
                e.outputs = st.r.outputs.len() as u64;
                e.seconds = t2.elapsed().as_secs_f64();
                // Block-scaled chains: the matrix has this tier 2-100x off (docs/errdist.md
                // §6); reported for comparison, never the headline.
                e.applicable = Some(false);
                e.applicability.push(
                    "block-scaled chain CLT: not validated (the matrix has it 2-100x off) FAILS"
                        .into(),
                );
                out.push(e);
            }
        }
    }
    if no_tier1 {
        notes.push("tier 1 not run (opt-in: --tier1)".into());
        return Ok((out, notes, vec![]));
    }
    let t1 = std::time::Instant::now();
    let run1 = |s: &schema::ChainSchema| -> Result<markov::ChainLaw> {
        if std::env::var("PTXV_ERRDIST_T1_OLD").is_ok() {
            markov::run(s, opts).map_err(|e| anyhow!(e))
        } else {
            tier1::run(s, tier1::Opts { uniform }).map_err(|e| anyhow!(e))
        }
    };
    let lr = run1(&sr)?;
    let mut sites = residue_sites("reference", &lr, uniform);
    let mut est = match &sc {
        None => combine::single(&lr, st.r.out_fmt, sr.round16),
        Some(sc) => {
            let lc = run1(sc)?;
            sites.extend(residue_sites("candidate", &lc, uniform));
            combine::pair(&lr, &lc, sr.round16.or(sc.round16))
        }
    };
    est.outputs = st.r.outputs.len() as u64;
    est.seconds = t1.elapsed().as_secs_f64();
    // Tier 1's applicability: the chain's law kept its mass (pruned / unmatched tail small
    // against P(differ)), and the mode is one the matrix validates.
    {
        let p = est.p_differ.map_or(0.0, |x| x.0);
        let lost = est.error_bar.model_mass.unwrap_or(0.0) + sr.pruned;
        let mass_ok = lost <= (0.05 * p).max(1e-7);
        est.applicability.push(format!(
            "law mass outside the model {lost:.2e} (needs <= max(5% of P(differ) {p:.2e}, 1e-7)) {}",
            if mass_ok { "ok" } else { "FAILS" }
        ));
        let scaled = fmts
            .values()
            .any(|f| matches!(f, Format::Ue8m0 | Format::Ue4m3));
        let pair_ok = !(scaled && st.c.is_some());
        est.applicability.push(if pair_ok {
            "mode validated by the matrix ok".to_string()
        } else {
            "pair of block-scaled kernels: the U-conditional pair combination is not validated there (docs/errdist.md: 1.2-1.3x) FAILS".to_string()
        });
        est.applicable = Some(mass_ok && pair_ok);
    }
    if sr.pruned > 0.0 {
        est.error_bar.notes.push(format!(
            "{:.2e} probability pruned from the element laws",
            sr.pruned
        ));
    }
    out.push(est);
    Ok((out, notes, sites))
}

/// Per step: weight-averaged residue diagnostics of the tier 1 chain.
fn residue_sites(
    k: &str,
    l: &ptxv_sim::errdist::markov::ChainLaw,
    uniform: bool,
) -> Vec<ptxv_sim::errdist::report::ResidueSite> {
    let mut m: std::collections::BTreeMap<usize, (f64, f64, f64)> = Default::default();
    for &(s, f, tv, w) in &l.residue {
        let e = m.entry(s).or_default();
        e.0 += w * f;
        e.1 += w * tv;
        e.2 += w;
    }
    m.into_iter()
        .filter(|x| x.1.2 > 0.0)
        .map(|(s, (f, tv, w))| ptxv_sim::errdist::report::ResidueSite {
            site: format!(
                "{k} step {s}: accumulator residue mod ulp(S) (weighted mean over cells)"
            ),
            max_fourier: f / w,
            tv_uniform: tv / w,
            uniform_used: uniform,
            weight: w,
        })
        .collect()
}

fn kid(k: &Kernel) -> ptxv_sim::errdist::report::KernelId {
    ptxv_sim::errdist::report::KernelId {
        case: k.run.case.m.name.clone(),
        variant: k.run.variant.clone(),
        front: k.run.front.clone(),
    }
}

fn ood(o: &ptxv_sim::errdist::mc::Obs) -> ptxv_sim::errdist::report::OutOfDomain {
    let n = o.inputs.max(1) as f64;
    ptxv_sim::errdist::report::OutOfDomain {
        mass: o.ood_inputs as f64 / n,
        by_cause: o
            .ood
            .iter()
            .map(|(k, v)| (k.clone(), *v as f64 / n))
            .collect(),
        method: format!(
            "fraction of {} sampled inputs with the event (Tier 3)",
            o.inputs
        ),
    }
}

/// writeln! into a String.
macro_rules! outln {
    ($o:expr, $($t:tt)*) => {{
        use std::fmt::Write as _;
        let _ = writeln!($o, $($t)*);
    }};
}

/// The text summary of a report.
pub(crate) fn report_text(rep: &ptxv_sim::errdist::report::Report) -> String {
    let mut o = String::new();
    outln!(
        o,
        "\nerrdist {} ({}): {} outputs ({}) per input",
        rep.mode,
        rep.quantity,
        rep.outputs,
        rep.output_format
    );
    for e in &rep.estimates {
        estimate_text(&mut o, e);
    }
    // Applicability of the analytic tiers (structural checks and the Tier 3 cross-check), and
    // the headline.
    for e in rep.estimates.iter().filter(|e| e.tier == 1 || e.tier == 2) {
        outln!(
            o,
            "tier {} {}:",
            e.tier,
            match e.applicable {
                Some(true) => "applicable",
                Some(false) => "REFUSED",
                None => "applicability unchecked",
            }
        );
        for a in &e.applicability {
            outln!(o, "    {a}");
        }
    }
    for r in &rep.residue {
        outln!(
            o,
            "residue {}: max |phi_k| {:.3e}, TV to uniform <= {:.3e}",
            r.site,
            r.max_fourier,
            r.tv_uniform
        );
    }
    outln!(
        o,
        "out of domain: {:.4e} of inputs {:?}",
        rep.out_of_domain.mass,
        rep.out_of_domain.by_cause
    );
    for v in &rep.validation {
        outln!(
            o,
            "validation {} vs {}: {} inputs, {} outputs; DAG mismatches {}; mean {:.4e} / {:.4e}, std {:.4e} / {:.4e} (pred / obs, {} scale); P(differ) {:.4e} / {:.4e}; KS D {:.4} p {:.3}; AD {:.2} p {:.3}: {}",
            v.source,
            v.against,
            v.inputs,
            v.outputs,
            v.dag_mismatch,
            v.mean_pred,
            v.mean_obs,
            v.std_pred,
            v.std_obs,
            if v.against == "tier 3" { "u" } else { "ulp" },
            v.p_differ_pred.unwrap_or(f64::NAN),
            v.p_differ_obs.map_or(f64::NAN, |x| x.0),
            v.ks_d,
            v.ks_p,
            v.ad,
            v.ad_p,
            if v.agree { "agree" } else { "DISAGREE" }
        );
        for n in &v.notes {
            outln!(o, "    {n}");
        }
    }
    if let Some(w) = &rep.worst_case {
        outln!(
            o,
            "worst case ({}{}): {}",
            if w.kind == "pair" {
                "equivalence proof"
            } else {
                "the kernel's own bound"
            },
            if w.available && !w.proven {
                ", with preconditions"
            } else {
                ""
            },
            w.human
        );
        if w.available {
            outln!(o, "    {}", w.formula);
        }
        for p in &w.preconditions {
            outln!(o, "    precondition: {p}");
        }
        if let Some(d) = &w.dist {
            outln!(
                o,
                "    the bound over the Tier 3 inputs, in output ulps of y: mean {:.3}, {}, max {:.3} ({} outputs; {} with y = 0 or subnormal left out)",
                d.mean_ulp,
                d.quantiles_ulp
                    .iter()
                    .map(|(q, x)| format!("q{q} {x:.3}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                d.max_ulp,
                d.n,
                d.tiny
            );
        }
        for c in &w.checks {
            outln!(
                o,
                "    observed vs bound ({}): {} outputs, largest |difference| / bound {:.3e}, {} exceed{}",
                c.source,
                c.checked,
                c.max_ratio,
                c.exceed,
                if c.exceed > 0 {
                    " -- BOUND VIOLATED"
                } else {
                    ""
                }
            );
        }
    }
    for n in &rep.notes {
        outln!(o, "note: {n}");
    }
    summary_text(&mut o, rep);
    o
}

/// |k| (output ulps) at cumulative probability q of a PMF over output ulps.
fn abs_ulp_quantile(pmf: &[(i64, f64)], q: f64) -> Option<i64> {
    let mut a: BTreeMap<i64, f64> = BTreeMap::new();
    for &(k, p) in pmf {
        *a.entry(k.abs()).or_default() += p;
    }
    let t: f64 = a.values().sum();
    if t <= 0.0 {
        return None;
    }
    let mut c = 0.0;
    for (k, p) in a {
        c += p / t;
        if c >= q - 1e-12 {
            return Some(k);
        }
    }
    None
}

/// The closing summary: which tier answered, P(differ), typical and largest errors in output
/// ulps, the proven worst case.
fn summary_text(o: &mut String, rep: &ptxv_sim::errdist::report::Report) {
    let Some(e) = rep.estimates.get(rep.best) else {
        return;
    };
    let t3 = rep.estimates.iter().find(|e| e.tier == 3);
    let why = match e.tier {
        0 => "exact (tier 0)".to_string(),
        3 => {
            let refused = rep
                .estimates
                .iter()
                .any(|x| (x.tier == 1 || x.tier == 2) && x.applicable == Some(false));
            if refused {
                "tier 3 (Monte Carlo; the analytic tiers refused, see above)".into()
            } else {
                "tier 3 (Monte Carlo)".into()
            }
        }
        t => format!(
            "tier {t} ({}){}",
            e.method.split([':', ';']).next().unwrap_or(""),
            if rep
                .crosscheck
                .iter()
                .any(|c| c.estimate == rep.best && c.agrees)
            {
                ", agrees with the tier 3 cross-check"
            } else {
                ""
            }
        ),
    };
    outln!(o, "\nsummary: {why}");
    if let Some(p) = e.p_differ {
        let x = t3
            .filter(|_| e.tier != 3)
            .and_then(|t| t.p_differ)
            .map(|(p3, lo, hi)| format!(" (tier 3: {p3:.3e} [{lo:.3e}, {hi:.3e}])"))
            .unwrap_or_default();
        outln!(o, "    P(bits differ) = {:.3e}{x}", p.0);
    }
    if !e.pmf_ulp.is_empty() {
        let q = |p| abs_ulp_quantile(&e.pmf_ulp, p).map_or("-".to_string(), |k| k.to_string());
        let seen = t3
            .and_then(|t| t.pmf_ulp.iter().map(|x| x.0.abs()).max())
            .map(|k| format!(", largest seen in tier 3: {k}"))
            .unwrap_or_default();
        outln!(
            o,
            "    error in output ulps (|k|): median {}, 99% within {}, 99.9% within {}{seen}",
            q(0.5),
            q(0.99),
            q(0.999)
        );
    }
    if let Some(w) = &rep.worst_case {
        outln!(
            o,
            "    worst case{}: {}",
            if w.proven { " (proven)" } else { "" },
            w.human
        );
    }
}

fn estimate_text(o: &mut String, e: &ptxv_sim::errdist::report::Estimate) {
    outln!(
        o,
        "tier {} ({}), n = {}, {:.1}s",
        e.tier,
        e.method,
        e.n,
        e.seconds
    );
    let q = |m: &ptxv_sim::errdist::report::Moments| {
        m.quantiles
            .iter()
            .filter(|(p, _)| [1e-3, 0.01, 0.5, 0.99, 0.999].contains(p))
            .map(|(p, x)| format!("q{p}={x:.4}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    if let Some(u) = &e.u_scale {
        outln!(
            o,
            "    error / (u S): mean {:.4} std {:.4} mean|.| {:.4} P(0) {:.4} max {:.3} | {}",
            u.mean,
            u.std,
            u.mean_abs,
            u.p_zero,
            u.max_abs.unwrap_or(0.0),
            q(u)
        );
    }
    if let Some(l) = &e.ulp {
        outln!(
            o,
            "    error / ulp_out(y): mean {:.4} std {:.4} mean|.| {:.4} | {}",
            l.mean,
            l.std,
            l.mean_abs,
            q(l)
        );
    }
    if !e.pmf_ulp.is_empty() {
        // The center (|k| <= 4) and the tail masses beyond it.
        let mut top: Vec<String> = vec![];
        let lo: f64 = e.pmf_ulp.iter().filter(|x| x.0 < -4).map(|x| x.1).sum();
        let hi: f64 = e.pmf_ulp.iter().filter(|x| x.0 > 4).map(|x| x.1).sum();
        if lo > 0.0 {
            top.push(format!("<-4:{lo:.3e}"));
        }
        top.extend(
            e.pmf_ulp
                .iter()
                .filter(|x| x.1 > 0.0 && x.0.abs() <= 4)
                .map(|(k, p)| format!("{k}:{p:.3e}")),
        );
        if hi > 0.0 {
            top.push(format!(">4:{hi:.3e}"));
        }
        outln!(o, "    PMF over output ulps: {}", top.join(" "));
    }
    if let Some((p, lo, hi)) = e.p_differ {
        outln!(o, "    P(bits differ) = {p:.4e} [{lo:.4e}, {hi:.4e}]");
    }
    if let Some(pm) = &e.pair {
        outln!(
            o,
            "    each against exact (abs): mean_r {:.3e} sd_r {:.3e} mean_c {:.3e} sd_c {:.3e} corr {:.4}",
            pm.mean_r,
            pm.var_r.sqrt(),
            pm.mean_c,
            pm.var_c.sqrt(),
            pm.corr_rc
        );
    }
}

/// Output 0 of a kernel run in the simulator on the given input bytes (f32 values; 16-bit
/// outputs widened exactly).
fn sim_outputs(
    run: &Run,
    bytes: &BTreeMap<String, Vec<u8>>,
    dir: &Path,
    tag: &str,
) -> Result<Vec<f32>> {
    let mut c = run.case.clone();
    for b in &mut c.m.buffers {
        if let Some(v) = bytes.get(&b.name).filter(|_| !out_only(&run.case, &b.name)) {
            let p = dir.join(format!("{tag}.{}.bin", b.name));
            std::fs::write(&p, v)?;
            b.init = "file".into();
            b.file = Some(p.to_str().unwrap().to_string());
        }
    }
    let (m, _, ok) = ptxv_sim::explore::run_once(super::machine(&c, run.kern.clone())?);
    if !ok {
        bail!("{tag}: {} did not terminate", run.label());
    }
    let oi = m.launch.outputs[0];
    let mem = &m.gmem[oi];
    let raw: Vec<u8> = (0..mem.bytes.len()).map(|i| *mem.bytes.get(i)).collect();
    let dt =
        c.m.buffers
            .iter()
            .find(|b| b.name == m.launch.allocs[oi].name)
            .map_or("f32", |b| b.dtype.as_str());
    Ok(widen(&raw, dt))
}

fn widen(raw: &[u8], dt: &str) -> Vec<f32> {
    match dt {
        "bf16" => raw
            .chunks_exact(2)
            .map(|h| f32::from_bits((u16::from_le_bytes([h[0], h[1]]) as u32) << 16))
            .collect(),
        "f16" => raw
            .chunks_exact(2)
            .map(|h| {
                f32::from_bits(ptxv_sim::elementwise::f16_to_f32_bits(u16::from_le_bytes(
                    [h[0], h[1]],
                )))
            })
            .collect(),
        _ => raw
            .chunks_exact(4)
            .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
            .collect(),
    }
}

/// Observed errors of one validation source, and how many outputs differ from the DAG.
pub(crate) struct Observed {
    pub obs: ptxv_sim::errdist::mc::Obs,
    pub mismatch: u64,
    pub inputs: u64,
}

/// Runs the kernels on one input (index, buffers): the reference's outputs and (if run) the
/// candidate's.
type RunKernels<'a> =
    dyn Fn(u64, &BTreeMap<String, Vec<u8>>) -> Result<(Vec<f32>, Option<Vec<f32>>)> + 'a;

/// Run both kernels on `n` sampled inputs (indices from `first`) through `run_kernel`, compare
/// with the DAG evaluation and collect the errors.
/// Tracked outputs where the DAG's value differs from the run's (NaN matches NaN).
fn dag_mismatch(p: &Prog, dag: &[f32], run: &[f32]) -> u64 {
    p.outputs
        .iter()
        .zip(dag.iter().zip(run))
        .filter(|(o, (a, b))| {
            o.is_some() && a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan())
        })
        .count() as u64
}

/// The extracted DAG against the simulator on two sampled inputs (outside every other index
/// range): (mismatching tracked outputs, compared). A graph that does not track the inputs
/// (a conversion the extraction folded to a constant, a compensation it lost) reproduces the
/// captured input only.
fn dag_probe(r: &Kernel, c: Option<&Kernel>, spec: &Spec, dir: &Path) -> Result<(u64, u64)> {
    let st =
        ptxv_sim::errdist::mc::Setup::new(&r.prog, c.map(|c| &c.prog)).map_err(|e| anyhow!(e))?;
    let ks: Vec<&Kernel> = std::iter::once(r).chain(c).collect();
    let ls = layouts(&ks)?;
    let v = validate_sim(&st, spec, &ls, &ks, 2, 3 << 40, 2, dir)?;
    let tracked: u64 = ks
        .iter()
        .map(|k| k.prog.outputs.iter().filter(|o| o.is_some()).count() as u64)
        .sum();
    Ok((v.mismatch, 2 * tracked))
}

fn validate_with(
    st: &ptxv_sim::errdist::mc::Setup,
    spec: &Spec,
    ls: &[Layout],
    n: usize,
    first: u64,
    outs: &RunKernels<'_>,
) -> Result<Observed> {
    use ptxv_sim::errdist::mc;
    let mut obs = mc::Obs::default();
    let mut mismatch = 0u64;
    let mut acc = ptxv_sim::errdist::chain::ExpAcc::default();
    for i in 0..n as u64 {
        let t = first + i;
        let smp = ptxv_sim::errdist::spec::sample(spec, ls, t).map_err(|e| anyhow!(e))?;
        let (ro, co) = outs(t, &smp.bytes)?;
        let mut wr = st.r.leaves(&smp.bytes).map_err(|e| anyhow!(e))?;
        st.r.eval_into(&mut wr);
        let rd = mc::outputs(st.r, &wr);
        mismatch += dag_mismatch(st.r, &rd, &ro);
        if let (Some(c), Some(co)) = (st.c, co.as_ref()) {
            let mut wc = c.leaves(&smp.bytes).map_err(|e| anyhow!(e))?;
            c.eval_into(&mut wc);
            let cd = mc::outputs(c, &wc);
            mismatch += dag_mismatch(c, &cd, co);
        }
        let keep = mc::keep_set(st.r.outputs.len(), 256, t);
        mc::observe(
            st,
            &mc::InputEval {
                w: &wr,
                r_out: &ro,
                c_out: co.as_deref(),
            },
            &mut obs,
            &keep,
            false,
            &mut acc,
        );
    }
    Ok(Observed {
        obs,
        mismatch,
        inputs: n as u64,
    })
}

/// Simulator validation, in parallel over inputs.
#[allow(clippy::too_many_arguments)]
fn validate_sim(
    st: &ptxv_sim::errdist::mc::Setup,
    spec: &Spec,
    ls: &[Layout],
    ks: &[&Kernel],
    n: usize,
    first: u64,
    threads: usize,
    dir: &Path,
) -> Result<Observed> {
    let per = n.div_ceil(threads.max(1));
    let parts: Vec<Result<Observed>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads.max(1))
            .filter(|i| i * per < n)
            .map(|i| {
                let lo = i * per;
                let cnt = per.min(n - lo);
                s.spawn(move || {
                    validate_with(st, spec, ls, cnt, first + lo as u64, &|t, bytes| {
                        let r = sim_outputs(&ks[0].run, bytes, dir, &format!("sim{t}r"))?;
                        let c = match ks.get(1) {
                            Some(k) => Some(sim_outputs(&k.run, bytes, dir, &format!("sim{t}c"))?),
                            None => None,
                        };
                        Ok((r, c))
                    })
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut all = Observed {
        obs: Default::default(),
        mismatch: 0,
        inputs: 0,
    };
    for p in parts {
        let p = p?;
        all.obs.merge(p.obs);
        all.mismatch += p.mismatch;
        all.inputs += p.inputs;
    }
    Ok(all)
}

/// GPU validation: write the inputs, run each kernel's captured binary over all of them in one
/// child process (`errdist-gpu-worker`), read the outputs back.
fn validate_gpu(
    st: &ptxv_sim::errdist::mc::Setup,
    spec: &Spec,
    ls: &[Layout],
    ks: &[&Kernel],
    n: usize,
    first: u64,
    dir: &Path,
) -> Result<(Observed, u32)> {
    let gdir = dir.join("gpu");
    std::fs::create_dir_all(&gdir)?;
    for i in 0..n as u64 {
        let t = first + i;
        let smp = ptxv_sim::errdist::spec::sample(spec, ls, t).map_err(|e| anyhow!(e))?;
        for (name, b) in &smp.bytes {
            std::fs::write(gdir.join(format!("t{i}.{name}.bin")), b)?;
        }
    }
    let free = ptxv_hw::free_gpus()?;
    let gpu = *free.iter().find(|&&g| g != 2).ok_or_else(|| {
        anyhow!(
            "no free GPU (PTXV_GPUS: {:?})",
            std::env::var("PTXV_GPUS").ok()
        )
    })?;
    let exe = std::env::current_exe()?;
    for (ki, k) in ks.iter().enumerate() {
        let tag = if ki == 0 { "r" } else { "c" };
        let mut child = std::process::Command::new(&exe)
            .args([
                "errdist-gpu-worker",
                k.run.case.dir.to_str().unwrap(),
                k.run.cubin.to_str().unwrap(),
                gdir.to_str().unwrap(),
                &n.to_string(),
                tag,
            ])
            .env("CUDA_VISIBLE_DEVICES", super::gpu_env(&k.run.case, gpu)?)
            .env("PTXV_PATCHED_PREFIX", k.run.cubin.with_extension(""))
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()?;
        let deadline =
            std::time::Instant::now() + std::time::Duration::from_secs_f64(60.0 + 2.0 * n as f64);
        loop {
            if let Some(s) = child.try_wait()? {
                if !s.success() {
                    let mut e = String::new();
                    std::io::Read::read_to_string(&mut child.stderr.take().unwrap(), &mut e)?;
                    bail!("GPU worker for {} failed: {}", k.run.label(), e.trim());
                }
                break;
            }
            if std::time::Instant::now() > deadline {
                child.kill()?;
                child.wait()?;
                bail!("GPU worker for {} timed out", k.run.label());
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
    let dt: Vec<String> = ks
        .iter()
        .map(|k| {
            let o = &k.run.case.m.outputs[0];
            k.run
                .case
                .buffer(o)
                .map_or("f32".into(), |b| b.dtype.clone())
        })
        .collect();
    let first_ = first;
    let obs = validate_with(st, spec, ls, n, first, &|t, _| {
        let i = t - first_;
        let rd = |tag: &str, d: &str| -> Result<Vec<f32>> {
            Ok(widen(
                &std::fs::read(gdir.join(format!("t{i}.out.{tag}.bin")))?,
                d,
            ))
        };
        let r = rd("r", &dt[0])?;
        let c = if ks.len() > 1 {
            Some(rd("c", &dt[1])?)
        } else {
            None
        };
        Ok((r, c))
    })?;
    Ok((obs, gpu))
}

/// `errdist-gpu-worker`: run one binary on every trial input of a directory (one process, one
/// context). Writes t<i>.out.<tag>.bin (output 0's bytes).
pub fn gpu_worker(case: &Path, cubin: &Path, dir: &Path, n: usize, tag: &str) -> Result<()> {
    let c = ptxv_case::Case::load(case)?;
    let img = std::fs::read(cubin)?;
    let o0 = c.buffer(&c.m.outputs[0])?.bytes();
    for i in 0..n {
        let mut inputs = BTreeMap::new();
        for b in &c.m.buffers {
            let p = dir.join(format!("t{i}.{}.bin", b.name));
            if !out_only(&c, &b.name) && p.exists() {
                inputs.insert(b.name.clone(), std::fs::read(&p)?);
            }
        }
        let r = ptxv_hw::run_with(&c, &img, 1, None, false, &inputs)?;
        std::fs::write(
            dir.join(format!("t{i}.out.{tag}.bin")),
            &r.first_output[..o0.min(r.first_output.len())],
        )?;
    }
    Ok(())
}

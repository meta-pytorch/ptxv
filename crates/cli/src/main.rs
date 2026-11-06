/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv`: verify execution and numerics of PTX/SASS kernels.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use clap::Parser;
use clap::Subcommand;
use ptxv_case::Case;
use ptxv_ir::Kernel;
use ptxv_sim::explore::Verdict;
use ptxv_sim::explore::explore;
use ptxv_sim::explore::run_once;
use ptxv_sim::machine::Machine;
use serde::Serialize;

mod calib;
mod calib_lp;
mod compare;
mod cvt_sweep;
mod equiv;
mod errdist;
mod errdist_html;
mod jsonout;
mod mufu_sweep;
mod sweep;
mod trace_dump;

// The explorer's threads clone and drop machine pages at a high rate; glibc malloc's arenas
// contend under that (and mimalloc is faster single-threaded too).
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser)]
#[command(
    version,
    about = "Verify execution (all schedules) and numerics of Blackwell PTX/SASS kernels"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Exhaustively explore all schedules for each variant and front-end.
    Check {
        case: PathBuf,
        #[arg(long, default_value = "all")]
        variant: String,
        #[arg(long, default_value = "both")]
        front: String,
        #[arg(long, default_value_t = 2_000_000)]
        max_states: usize,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Explore one cluster (CTA) at a time plus a cross-cluster global-memory check, for
        /// launches whose clusters are independent (E-CTA-INDEP); --max-states is per slice.
        /// With --groups: each explored group one thread slice at a time when it splits
        /// (E-SLICE).
        #[arg(long)]
        slices: bool,
        /// Try the well-synchronization certificate first (E-WELLSYNC: one deterministic run
        /// decides every schedule when every barrier phase, wait and value is determined);
        /// explore when it does not hold, printing why. With --slices: per cluster.
        #[arg(long)]
        certify: bool,
        /// Check a multi-launch case one group of launches at a time (E-LAUNCH-GROUPS):
        /// `auto`, or the later groups as `a,b;c-d` (group 1 is the rest). Each group by
        /// certificate, the last one by exploration if that fails, with a cross-group race
        /// check on global memory.
        #[arg(long)]
        groups: Option<String>,
        /// Print one JSON document on stdout (docs/json.md); the report goes to stderr.
        #[arg(long)]
        json: bool,
        /// Limited residency (E-RESIDENCY): `N` explores with at most N CTAs per GPU resident
        /// at once (CTAs become resident in any order as slots free up); `auto` computes the
        /// residency bounds that deadlock and the minimum each kernel needs (with --groups:
        /// per launch group, the earlier groups run once each).
        #[arg(long)]
        residency: Option<String>,
        /// TMEM co-tenant (E-TMEM-COTENANT): allocations land at any free columns, CTA pairs
        /// share an SM's tensor memory, and the happens-before analysis reports CTAs that hold
        /// columns or the allocation permit while waiting on a CTA that cannot allocate
        /// (with --groups: per launch group, as --residency auto).
        #[arg(long)]
        tmem_cotenant: bool,
        /// Allow the case to rely on a refuted assumption (ASSUMPTIONS.md), e.g. E-POLL-ACQ
        /// for a case that sets `poll_acquire`; the verdict names it. Repeatable, or a,b.
        #[arg(long, value_delimiter = ',')]
        assume_refuted: Vec<String>,
        /// Skip the second level: re-checking race / stale findings under the GB300 hardware
        /// model (E-HW-L1) to classify each as observable or latent (also PTXV_GB300=0).
        #[arg(long)]
        no_gb300: bool,
    },
    /// Numerics of one deterministic run: exact error per output, bounds, Thm 1 check.
    Num {
        case: PathBuf,
        #[arg(long, default_value = "good")]
        variant: String,
        /// Override the tensor-core arithmetic model (e.g. g16-exact-rn-accin-tzero-eprod for
        /// an idealized exact-dot, round-to-nearest tensor core).
        #[arg(long)]
        mma_model: Option<String>,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Is a candidate kernel equivalent to a reference (L0 bitwise .. L3 tolerance)? Reads an
    /// equiv.toml naming both (see crates/cli/src/equiv.rs).
    Equiv {
        file: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Do not replay witnesses on a GPU.
        #[arg(long)]
        no_hw: bool,
        /// Random-input trials checking each proven bound empirically (overrides the file's
        /// `trials`).
        #[arg(long)]
        trials: Option<usize>,
        /// Print every output's level and bound (default: grouped).
        #[arg(long)]
        per_output: bool,
        /// Recompute the exhaustive slice sweeps instead of reading the file's `sweep_cache`.
        #[arg(long)]
        no_sweep_cache: bool,
        /// Print one JSON document on stdout (docs/json.md); the report goes to stderr.
        #[arg(long)]
        json: bool,
        /// Write each candidate's per-output bound as data (docs/json.md, "ptxv-bound").
        #[arg(long)]
        emit_bound: Option<PathBuf>,
    },
    /// The distribution of a kernel's rounding error (C - exact; a case directory) or of a
    /// pair's difference (C - R; an equiv.toml) under an input distribution (docs/errdist.md).
    Errdist {
        /// A case directory (one kernel) or an equiv.toml (reference vs a candidate); not
        /// needed with --model.
        path: Option<PathBuf>,
        /// Input-distribution spec, JSON or TOML (docs/errdist.md §2); not needed with
        /// --compile.
        #[arg(long)]
        inputs: Option<PathBuf>,
        /// Compile stage: simulate the kernel(s) once and write the model file (programs
        /// section; with --inputs also the non-chain `nonlin` tables, and the run's report) to
        /// FILE, or with no FILE to the cache keyed by the case hash ($PTXV_CACHE/errdist, else
        /// ~/.cache/ptxv/errdist). Without --inputs it stops there (docs/errdist.md §1.1).
        #[arg(long, num_args = 0..=1, default_missing_value = "-")]
        compile: Option<PathBuf>,
        /// Query stage: analyze a compiled model under --inputs (no simulation; non-chain
        /// tables compiled or extended here are written back).
        #[arg(long)]
        model: Option<PathBuf>,
        /// Write the JSON report here (docs/errdist.md §3).
        #[arg(long)]
        json: Option<PathBuf>,
        /// Write a self-contained HTML report here: P(differ) per tier with its interval,
        /// the tiers' and observations' PMFs over output ulps, and the text summary.
        #[arg(long)]
        html: Option<PathBuf>,
        /// Which candidate of the equiv.toml.
        #[arg(long, default_value_t = 0)]
        candidate: usize,
        /// Monte Carlo inputs through the numerics DAG (Tier 3). Default 300: the
        /// cross-check of Tier 2 and the fallback headline when Tier 2 refuses; 0 skips it.
        #[arg(long)]
        mc: Option<usize>,
        /// Validation inputs run through the simulator.
        #[arg(long)]
        sim_trials: Option<usize>,
        /// Validation inputs run on GB300 (PTXV_GPUS restricts which GPU).
        #[arg(long)]
        gpu_trials: Option<usize>,
        #[arg(long)]
        per_output: bool,
        /// Worker threads (0: all cores).
        #[arg(long, default_value_t = 0)]
        threads: usize,
        /// Scratch directory.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Print per-step local-error diagnostics instead of the report.
        #[arg(long, hide = true)]
        explore: bool,
        /// Analyze the outputs before their 16-bit rounding (the f32 accumulators).
        #[arg(long)]
        pre_round: bool,
        /// Also run the uniform-residue heuristic (every residue law uniform) for comparison.
        #[arg(long)]
        uniform: bool,
        /// Write the output's error state (docs/errdist.md §10): value and error moments,
        /// their covariance and joint samples from Tier 3, the next kernel's input error
        /// (`"error": {"kind": "pairs", "file": FILE}` in its spec).
        #[arg(long)]
        errstate: Option<PathBuf>,
        /// Also run Tier 1 (the residue-exact chain of MMA outputs; minutes on wide scale
        /// ranges, hours at K = 4096; memory flat in K). Default: tier 0, tier 2 where its
        /// applicability check passes, else tier 3.
        #[arg(long)]
        tier1: bool,
        /// Accepted for compatibility (Tier 1 is opt-in now: --tier1).
        #[arg(long, hide = true)]
        no_tier1: bool,
    },
    /// Compare a candidate case with a reference case (both captured by tools/capture.py):
    /// execution verdicts, equivalence level, per-output bounds, the captured GPU outputs
    /// against the bound, and why anything is unproven. Writes report.json / report.md.
    Compare {
        reference: PathBuf,
        candidate: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value_t = 200_000)]
        max_states: usize,
        /// Random-input simulator trials checking the proven bound.
        #[arg(long, default_value_t = 20)]
        trials: usize,
        /// Segment mode for the numerics: on / off (default: on when a kernel uses MUFU).
        #[arg(long)]
        segments: Option<bool>,
        /// Cache of the exhaustive slice sweeps (default: <out>/sweeps.json, seeded with
        /// fixtures/c_equiv_softmax/sweeps.json).
        #[arg(long)]
        sweep_cache: Option<PathBuf>,
        /// Replay an L1 witness on a GPU (PTXV_GPUS restricts which).
        #[arg(long)]
        hw: bool,
        /// Descriptions of what each side is (shown in the report).
        #[arg(long)]
        ref_label: Option<String>,
        #[arg(long)]
        cand_label: Option<String>,
        /// The real function an elementwise pair computes (gelu, gelu_tanh, silu, exp, ...;
        /// default: identified from the kernels).
        #[arg(long)]
        function: Option<String>,
        /// Print report.json on stdout (docs/json.md); the summary goes to stderr.
        #[arg(long)]
        json: bool,
        /// Write the proven per-output bound as data (docs/json.md, "ptxv-bound").
        #[arg(long)]
        emit_bound: Option<PathBuf>,
    },
    /// Run variants on a free GPU (child process + watchdog) and compare with the simulator.
    Hw {
        case: PathBuf,
        #[arg(long, default_value = "all")]
        variant: String,
        #[arg(long, default_value_t = 1000)]
        runs: usize,
        #[arg(long, default_value_t = 20.0)]
        timeout: f64,
        /// Poison every SM's shared memory before each launch.
        #[arg(long)]
        scrub: bool,
        /// Witness replay: insert `nanosleep.u32 NS` before PTX line LINE, or with `@W` make
        /// only warp W spin NS ns on %globaltimer there (repeatable).
        #[arg(long, value_name = "LINE:NS[@WARP]")]
        delay: Vec<String>,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// check + num + hw for every variant; writes summary.json / summary.md.
    Suite {
        case: PathBuf,
        #[arg(long, default_value_t = 1000)]
        runs: usize,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Calibrate the tensor-core arithmetic model against GB300 (uses a GEMM case with
    /// bf16 buffers A[M,K], B[N,K] and f32 output D[M,N]; f32 A / B: a TF32 MMA, probes/tmma).
    CalibMma {
        case: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Narrow kinds: only the format configurations whose name contains this.
        #[arg(long)]
        only: Option<String>,
    },
    /// Narrow-format cvt sweep on GB300 (probes/cvt_lp): compare with the model; `--record`
    /// writes the per-op hardware hashes cargo test replays.
    CvtSweep {
        dir: PathBuf,
        #[arg(long)]
        gpu: Option<u32>,
        #[arg(long)]
        record: bool,
        /// Compare the dumps of an earlier run (out/cvt_lp) instead of running the GPU.
        #[arg(long)]
        reuse: bool,
    },
    /// Compare an exhaustive GB300 sweep of one MUFU / approx op (probes/mufu) with the model,
    /// measure its error against the true function, optionally record the replay sample.
    MufuSweep {
        /// ex2.ftz, ex2, rcp.ftz, ..., sin.raw, tanh, ex2.f16, ... (see mufu_sample::OPS).
        op: String,
        /// Dump of all outputs in input order (u32 little-endian), or - for stdin.
        input: PathBuf,
        /// Merge this op's replay sample into a samples.json.
        #[arg(long)]
        sample: Option<PathBuf>,
    },
    /// Scoreboard mutation sweep: remove each scoreboard bit of the good binary in turn, run
    /// the static scoreboard check and the GPU on every mutant, and compare.
    SbSweep {
        case: PathBuf,
        #[arg(long, default_value_t = 200)]
        runs: usize,
        #[arg(long, default_value_t = 20.0)]
        timeout: f64,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Hardware probe: compile each variant, run it once on a free GPU under the watchdog,
    /// and print the raw output words (for probing assumptions, see ASSUMPTIONS.md).
    Probe {
        case: PathBuf,
        #[arg(long, default_value = "all")]
        variant: String,
        #[arg(long, default_value_t = 10.0)]
        timeout: f64,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// errdist's GPU validation: run one binary over a directory of trial inputs.
    #[command(hide = true)]
    ErrdistGpuWorker {
        case: PathBuf,
        cubin: PathBuf,
        dir: PathBuf,
        n: usize,
        tag: String,
    },
    #[command(hide = true)]
    HwWorker {
        case: PathBuf,
        image: PathBuf,
        #[arg(long)]
        runs: usize,
        #[arg(long)]
        expected: Option<PathBuf>,
        #[arg(long)]
        scrub: bool,
        /// NAME=PATH: replace a buffer's initial contents.
        #[arg(long)]
        input: Vec<String>,
        /// Write the first run's output bytes here.
        #[arg(long)]
        dump: Option<PathBuf>,
    },
}

pub(crate) fn out_dir(case: &Case, out: &Option<PathBuf>) -> Result<PathBuf> {
    let d = match out {
        Some(o) => o.clone(),
        None => {
            let root = std::env::var("PTXV_ROOT")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("."));
            root.join("out").join(&case.m.name)
        }
    };
    // Multi-node ranks share the file system: each writes its own copies.
    let d = match ptxv_sim::dist::get() {
        Some(r) => d.join(format!("rank{}", r.rank)),
        None => d,
    };
    std::fs::create_dir_all(&d)?;
    Ok(d)
}

/// Materialize a variant and load it through one front-end. Returns the kernel and the
/// cubin path (always built, since hardware runs use it).
pub(crate) fn load(
    case: &Case,
    variant: &str,
    front: &str,
    dir: &Path,
) -> Result<(Kernel, PathBuf)> {
    load_with(case, variant, front, dir, &[])
}

/// Insert delays before given (1-based) PTX lines of the variant: `LINE:NS` puts a
/// `nanosleep.u32 NS` there (every thread); `LINE:NS@W` makes only warp W of the CTA (tid.x / 32,
/// 1-D blocks) spin until %globaltimer has advanced NS ns, which skews that warp against the
/// others.
fn add_delays(ptx: &str, delays: &[String]) -> Result<String> {
    let mut d = std::collections::BTreeMap::<usize, Vec<String>>::new();
    for (k, x) in delays.iter().enumerate() {
        let (l, rest) = x
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("--delay wants LINE:NS or LINE:NS@WARP"))?;
        let code = match rest.split_once('@') {
            None => format!(
                "    nanosleep.u32 {};  // ptxv witness delay\n",
                rest.parse::<u32>()?
            ),
            Some((ns, w)) => {
                let (ns, w) = (ns.parse::<u64>()?, w.parse::<u32>()?);
                format!(
                    "    {{  // ptxv witness delay: warp {w} spins {ns} ns\n\
                     \t.reg .b32 %ptxv_w;\n\t.reg .b64 %ptxv_t<2>;\n\t.reg .pred %ptxv_p;\n\
                     \tmov.u32 %ptxv_w, %tid.x;\n\tshr.u32 %ptxv_w, %ptxv_w, 5;\n\
                     \tsetp.ne.u32 %ptxv_p, %ptxv_w, {w};\n\t@%ptxv_p bra PTXV_SKIP{k};\n\
                     \tmov.u64 %ptxv_t0, %globaltimer;\nPTXV_SPIN{k}:\n\
                     \tmov.u64 %ptxv_t1, %globaltimer;\n\tsub.u64 %ptxv_t1, %ptxv_t1, %ptxv_t0;\n\
                     \tsetp.lt.u64 %ptxv_p, %ptxv_t1, {ns};\n\t@%ptxv_p bra PTXV_SPIN{k};\n\
                     PTXV_SKIP{k}:\n    }}\n"
                )
            }
        };
        d.entry(l.parse::<usize>()?).or_default().push(code);
    }
    let mut out = String::new();
    for (i, line) in ptx.lines().enumerate() {
        if let Some(code) = d.get(&(i + 1)) {
            let code = code.concat();
            // Labels must stay on their own line, before the delay.
            if let Some((lbl, rest)) = line
                .split_once(':')
                .filter(|(l, _)| !l.contains(' ') && !l.trim().is_empty() && !l.contains('.'))
            {
                out += &format!("{lbl}:\n{code}{rest}\n");
                continue;
            }
            out += &code;
        }
        out += line;
        out.push('\n');
    }
    Ok(out)
}

/// A PTX case's variant (with witness delays) compiled by ptxas, plus any control-bit patch:
/// (the PTX text, the cubin).
fn ptx_cubin(
    case: &Case,
    variant: &str,
    dir: &Path,
    delays: &[String],
) -> Result<(String, PathBuf)> {
    let ptx = add_delays(&case.variant_ptx(variant)?, delays)?;
    let variant = &if delays.is_empty() {
        variant.to_string()
    } else {
        format!("{variant}.delayed")
    };
    let ptx_path = dir.join(format!("{variant}.ptx"));
    std::fs::write(&ptx_path, &ptx)?;
    let cubin = dir.join(format!("{variant}.cubin"));
    ptxv_sass::ptxas(&ptx_path, &cubin, &case.m.arch)
        .with_context(|| format!("ptxas on variant {variant}"))?;
    let patch = &case.variant(variant.trim_end_matches(".delayed"))?.patch;
    if !patch.is_empty() {
        let log = ptxv_sass::scoreboard::patch_file(&cubin, &case.m.entry, patch)?;
        std::fs::write(
            dir.join(format!("{variant}.patch.txt")),
            log.join("\n") + "\n",
        )?;
    }
    Ok((ptx, cubin))
}

fn load_with(
    case: &Case,
    variant: &str,
    front: &str,
    dir: &Path,
    delays: &[String],
) -> Result<(Kernel, PathBuf)> {
    if let Some(cubin) = &case.m.cubin {
        // Binary-only case: the cubin as captured (plus any control-bit patch).
        if front != "sass" {
            bail!("binary-only case: there is no PTX front-end");
        }
        let c = dir.join(format!("{variant}.cubin"));
        std::fs::copy(case.dir.join(cubin), &c)?;
        let patch = &case.variant(variant)?.patch;
        let (own, main): (Vec<_>, Vec<_>) = patch.iter().cloned().partition(|p| p.cubin.is_some());
        if !main.is_empty() {
            ptxv_sass::scoreboard::patch_file(&c, &case.m.entry, &main)?;
        }
        // Patches of a launch's own cubin go to `{variant}.{cubin}` next to the variant's
        // cubin (the hardware worker picks them up the same way, `ptxv_hw`).
        let mut patched = std::collections::BTreeMap::new();
        for p in &own {
            let name = p.cubin.clone().unwrap();
            let dst = dir.join(format!("{variant}.{name}"));
            if !patched.contains_key(&name) {
                std::fs::copy(case.dir.join(&name), &dst)?;
                patched.insert(name.clone(), dst.clone());
            }
            let entry = p.kernel.clone().unwrap_or_default();
            ptxv_sass::scoreboard::patch_file(&dst, &entry, std::slice::from_ref(p))?;
        }
        std::fs::write(
            dir.join(format!("{variant}.sass")),
            ptxv_sass::run_tool("nvdisasm", &["-c", c.to_str().unwrap()])?,
        )?;
        // Several launches of kernels in the same cubin (rung 3a: prepare + GEMM) are one
        // combined kernel, as for PTX (Kernel::concat). A launch may name its own cubin (a
        // kernel from another library module, tools/capture.py).
        let mut ks = Vec::new();
        let mut first_err = None;
        for l in case.m.all_launches() {
            let k = match l.cubin {
                Some(own) if patched.contains_key(own) => {
                    ptxv_sass::load_cubin(&patched[own], l.entry)
                }
                Some(own) => ptxv_sass::load_cubin(&case.dir.join(own), l.entry),
                None => ptxv_sass::load_cubin(&c, l.entry),
            };
            match k {
                Ok(k) => ks.push(k),
                // PTXV_LIFT_REPORT: report every launch's kernel before failing.
                Err(e) if std::env::var("PTXV_LIFT_REPORT").is_ok() => {
                    first_err.get_or_insert(e);
                }
                Err(e) => return Err(e),
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        let k = if ks.len() == 1 {
            ks.pop().unwrap()
        } else {
            Kernel::concat(ks)
        };
        return Ok((k, c));
    }
    let (ptx, cubin) = ptx_cubin(case, variant, dir, delays)?;
    let variant = &if delays.is_empty() {
        variant.to_string()
    } else {
        format!("{variant}.delayed")
    };
    let mut ks = Vec::new();
    for l in case.m.all_launches() {
        if l.cubin.is_some() {
            bail!(
                "launch {} names its own cubin: only binary-only cases may",
                l.entry
            );
        }
        ks.push(match front {
            "ptx" => ptxv_ptx::parse_kernel(&ptx, Some(l.entry))?,
            "sass" => ptxv_sass::load_cubin(&cubin, l.entry)?,
            f => bail!("unknown front {f}"),
        });
    }
    if front == "sass" {
        std::fs::write(
            dir.join(format!("{variant}.sass")),
            ptxv_sass::run_tool("nvdisasm", &["-c", cubin.to_str().unwrap()])?,
        )?;
    }
    // A multi-kernel launch (rung 1e) is one combined kernel (Kernel::concat).
    let k = if ks.len() == 1 {
        ks.pop().unwrap()
    } else {
        Kernel::concat(ks)
    };
    Ok((k, cubin))
}

fn machine(case: &Case, k: Kernel) -> Result<Machine> {
    let launch = ptxv_sim::build_launch(case, &k)?;
    Ok(Machine::new(Arc::new(k), Arc::new(launch)))
}

fn variants(case: &Case, v: &str) -> Vec<String> {
    if v == "all" {
        case.m.variants.iter().map(|x| x.id.clone()).collect()
    } else {
        v.split(',').map(String::from).collect()
    }
}

fn fronts(f: &str) -> Vec<&'static str> {
    match f {
        "ptx" => vec!["ptx"],
        "sass" => vec!["sass"],
        _ => vec!["ptx", "sass"],
    }
}

#[derive(Serialize)]
struct CheckRow {
    variant: String,
    front: String,
    /// Captured cases: the simulator's outputs against the GPU's (`expect`), per output.
    expect_report: Vec<String>,
    expect: Vec<String>,
    got: Vec<String>,
    pass: bool,
    /// Refuted assumptions the verdict relies on, each explicitly opted into (case
    /// `assume_refuted`, `--assume-refuted`).
    refuted_opt_in: Vec<String>,
    verdict: Verdict,
    /// The second level (ptxv_sim::twolevel): the hardware-model check that classified the
    /// verdict's race / stale findings (each finding's `gb300`). None: not needed or disabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    gb300: Option<ptxv_sim::twolevel::Gb300Check>,
}

fn print_verdict(
    variant: &str,
    front: &str,
    v: &Verdict,
    expect: &[String],
    refuted: &[String],
    gb: Option<&ptxv_sim::twolevel::Gb300Check>,
) {
    let got = v.classes();
    let pass = v.meets(expect);
    println!(
        "{} {variant:<6} {front:<4} {:<26} states={:<7} terminal={:<5} outputs={} {}{:.2}s",
        if pass { "PASS" } else { "FAIL" },
        format!("[{}]", got.join(",")),
        v.states,
        v.terminal_states,
        v.distinct_outputs,
        if v.complete { "" } else { "INCOMPLETE " },
        v.seconds
    );
    println!("    assumes: {}", v.assumptions.join(" "));
    for a in &v.assumptions {
        if ptxv_sim::assumptions::get(a).is_some_and(|x| x.status == "refuted") {
            println!(
                "    WARNING: relies on {a}, which GB300 tests refuted (ASSUMPTIONS.md){}; the verdict can miss what it assumes away",
                if refuted.iter().any(|r| r == a) {
                    " (explicitly opted in)"
                } else {
                    ""
                }
            );
        }
    }
    for n in &v.notes {
        println!("    note: {n}");
    }
    if let Some(c) = gb {
        println!("    {}", ptxv_sim::twolevel::summary_line(c));
    }
    for f in &v.findings {
        println!(
            "    {}: {}",
            f.kind.to_uppercase(),
            f.msg.trim_end().replace('\n', "\n      ")
        );
        if let Some(g) = &f.gb300 {
            println!(
                "      GB300: {} ({}: {})",
                g.class.to_uppercase(),
                match g.class.as_str() {
                    "observable" => "reachable under the hardware model",
                    "latent" => "a PTX-model bug not reachable on current GB300 hardware",
                    "latent-closed-world" =>
                        "observable on GB300 with a co-resident kernel that fills the line",
                    _ => "not decided",
                },
                g.reason
            );
        }
        let n = f.trace.len();
        let show = &f.trace[n.saturating_sub(8)..];
        println!(
            "      trace ({} steps{}):",
            n,
            if n > 8 { ", last 8" } else { "" }
        );
        for s in show {
            println!("        - {s}");
        }
    }
    if let Some(c) = gb {
        for f in &c.extra {
            println!(
                "    GB300 MODEL ONLY, {} {}: {}",
                f.gb300
                    .as_ref()
                    .map(|g| g.class.to_uppercase())
                    .unwrap_or_default(),
                f.kind.to_uppercase(),
                f.msg.trim_end().replace('\n', "\n      ")
            );
        }
    }
}

/// The hardware-model check (second level) on by default; `--no-gb300` or `PTXV_GB300=0`
/// turn it off.
fn gb300_enabled(flag_off: bool) -> bool {
    !flag_off && std::env::var("PTXV_GB300").as_deref() != Ok("0")
}

/// `ptxv check --residency auto` / `--tmem-cotenant`: per variant, front-end and (with
/// `--groups`) launch group, the residency analysis and / or the co-tenant check. Fails when a
/// variant's `min_residency` or `expect_cotenant` is not met.
#[allow(clippy::too_many_arguments)]
fn cmd_coresidency(
    case: &Case,
    variant: &str,
    front: &str,
    max_states: usize,
    out: &Option<PathBuf>,
    groups: &Option<String>,
    resid: bool,
    tenant: bool,
) -> Result<bool> {
    let dir = out_dir(case, out)?;
    let mut pass = true;
    for v in variants(case, variant) {
        for f in fronts(front) {
            if f == "ptx" && case.m.cubin.is_some() {
                continue;
            }
            let (k, _) = load(case, &v, f, &dir)?;
            let vcase = case.for_variant(&v)?;
            let var = case.variant(&v)?.clone();
            let m = machine(&vcase, k)?;
            let parts: Vec<(String, Result<Machine, String>)> = match groups {
                None => vec![("launch".into(), Ok(m))],
                Some(spec) => {
                    let gs = ptxv_sim::groups::parse(spec, m.launch.grids.len())
                        .map_err(|e| anyhow::anyhow!("--groups: {e}"))?;
                    ptxv_sim::residency::group_machines(&m, &gs)
                        .into_iter()
                        .zip(gs.iter())
                        .enumerate()
                        .map(|(i, (r, g))| (format!("group {} {:?}", i + 1, g), r))
                        .collect()
                }
            };
            // Debugging / large cases: PTXV_GROUPS_ONLY=k[,k2..] analyzes only those launch
            // groups (1-based).
            let only: Option<Vec<usize>> = std::env::var("PTXV_GROUPS_ONLY")
                .ok()
                .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect());
            for (gi, (name, r)) in parts.into_iter().enumerate() {
                if groups.is_some() && only.as_ref().is_some_and(|o| !o.contains(&(gi + 1))) {
                    continue;
                }
                let m = match r {
                    Ok(m) => m,
                    Err(e) => {
                        println!("{v} {f} {name}: not analyzed: {e}");
                        pass = false;
                        continue;
                    }
                };
                if resid {
                    let t0 = Instant::now();
                    let rep = ptxv_sim::residency::analyze(&m, max_states, 1 << 14);
                    let ok = groups.is_some()
                        || var.min_residency.is_none_or(|n| rep.min_of(0) == Some(n));
                    pass &= ok;
                    println!(
                        "{} {v} {f} {name}: residency {:.1}s",
                        if ok { "PASS" } else { "FAIL" },
                        t0.elapsed().as_secs_f64()
                    );
                    for l in rep.lines() {
                        println!("    {l}");
                    }
                    std::fs::write(
                        dir.join(format!("{v}.{f}.residency.json")),
                        serde_json::to_string_pretty(&rep)?,
                    )?;
                }
                if tenant {
                    let mt = ptxv_sim::tenant::with_tenant(m, vec![]);
                    let verdict = ptxv_sim::tenant::check(&mt, max_states, 8);
                    let expect = if groups.is_some() {
                        verdict.classes()
                    } else {
                        var.expect_cotenant
                            .clone()
                            .unwrap_or_else(|| var.expect_for(f))
                    };
                    pass &= verdict.meets(&expect);
                    print_verdict(
                        &format!("{v} {name} co-tenant"),
                        f,
                        &verdict,
                        &expect,
                        &[],
                        None,
                    );
                }
            }
        }
    }
    Ok(pass)
}

#[allow(clippy::too_many_arguments)]
fn cmd_check(
    case: &Case,
    variant: &str,
    front: &str,
    max_states: usize,
    out: &Option<PathBuf>,
    slices: bool,
    certify: bool,
    groups: &Option<String>,
    gb300: bool,
) -> Result<Vec<CheckRow>> {
    let dir = out_dir(case, out)?;
    let mut rows = Vec::new();
    for v in variants(case, variant) {
        for f in fronts(front) {
            if f == "ptx" && case.m.cubin.is_some() {
                continue; // binary-only case
            }
            let expect = case.variant(&v)?.expect_for(f);
            let (k, _) = load(case, &v, f, &dir)?;
            let vcase = case.for_variant(&v)?;
            let case = &vcase;
            let m = machine(case, k.clone())?;
            let refuted = m.launch.refuted_opt_in.clone();
            // Debugging (rung 4b, large launch groups): PTXV_RUN_ONCE=1 runs one
            // deterministic schedule (explore::run_once) and reports its findings and outputs.
            if std::env::var("PTXV_RUN_ONCE").is_ok() {
                let t0 = Instant::now();
                let m1 = m.clone();
                let (m, fs, ok) = ptxv_sim::explore::run_once(m);
                println!(
                    "RUN_ONCE {v} {f}: terminal={ok} findings={} {:.1}s",
                    fs.len(),
                    t0.elapsed().as_secs_f64()
                );
                let mut seen = std::collections::BTreeSet::new();
                let mut fs1: Vec<_> = fs
                    .into_iter()
                    .filter(|x| seen.insert(x.sig.clone()))
                    .collect();
                // The second level on one run of the hardware model: like the level-1 run's
                // findings, its happens-before facts and predictive L1 check hold for every
                // schedule with the same order; other orders are not explored. The
                // classification is as strong as the one-run finding it classifies.
                let mut gbc = None;
                if gb300 {
                    let mut v1 = Verdict {
                        states: 1,
                        transitions: 0,
                        complete: false,
                        terminal_states: ok as usize,
                        distinct_outputs: 0,
                        findings: std::mem::take(&mut fs1),
                        assumptions: vec![],
                        notes: vec![],
                        seconds: 0.0,
                        outputs: vec![],
                    };
                    if let Some(why) = ptxv_sim::twolevel::wanted(&v1, &m1.launch) {
                        let why = format!("{why}; one deterministic run of each model");
                        let c = ptxv_sim::twolevel::run(&mut v1, &m1, why, &mut |m2| {
                            let (_, fs2, ok2) = ptxv_sim::explore::run_once(m2);
                            let mut seen = std::collections::BTreeSet::new();
                            Verdict {
                                states: 1,
                                transitions: 0,
                                complete: ok2,
                                terminal_states: ok2 as usize,
                                distinct_outputs: 0,
                                findings: fs2
                                    .into_iter()
                                    .filter(|x| seen.insert(x.sig.clone()))
                                    .collect(),
                                assumptions: vec![],
                                notes: vec![],
                                seconds: 0.0,
                                outputs: vec![],
                            }
                        });
                        gbc = Some(c);
                    }
                    fs1 = v1.findings;
                }
                for x in &fs1 {
                    println!(
                        "    {}: {}",
                        x.kind.to_uppercase(),
                        x.msg.lines().next().unwrap_or("")
                    );
                    if let Some(g) = &x.gb300 {
                        println!("      GB300: {} ({})", g.class.to_uppercase(), g.reason);
                    }
                }
                if let Some(c) = &gbc {
                    println!("    {}", ptxv_sim::twolevel::summary_line(c));
                    for x in &c.extra {
                        println!(
                            "    GB300 MODEL ONLY, {} {}: {}",
                            x.gb300
                                .as_ref()
                                .map(|g| g.class.to_uppercase())
                                .unwrap_or_default(),
                            x.kind.to_uppercase(),
                            x.msg.lines().next().unwrap_or("")
                        );
                    }
                }
                if ok {
                    for line in case.expect_report(&m.outputs())? {
                        println!("    {line}");
                    }
                }
                continue;
            }
            let m1 = m.clone();
            // Launch groups: per run, each group's (index, decided) as reported.
            let group_log: std::cell::RefCell<Vec<(usize, bool)>> = Default::default();
            // One check of the case (level 1: the machine as configured; level 2: the
            // hardware model, `make` building its machines).
            let check = |make: &dyn Fn() -> Machine, label: &str| -> Verdict {
                let m = make();
                // PTXV_EXPLORER=dpor selects partial-order reduction (crates/sim/src/dpor.rs);
                // the default is the stateful explorer. Both are cross-checked in the test suite.
                match std::env::var("PTXV_EXPLORER").as_deref() {
                    // E-L1-CORES: one run per SM placement when `l1_coresident` (else one run).
                    Ok("dpor") => ptxv_sim::l1_cores::explore_placements(m, &mut |m| {
                        ptxv_sim::dpor::explore_dpor(m, max_states)
                    }),
                    _ if groups.is_some() => ptxv_sim::l1_cores::explore_placements(m, &mut |m| {
                        if slices {
                            ptxv_sim::groups::SLICES
                                .store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        ptxv_sim::groups::check_spec(
                            m,
                            groups.as_deref().unwrap(),
                            max_states,
                            &mut |i, g, v| {
                                group_log.borrow_mut().push((
                                    i,
                                    v.complete || v.assumptions.iter().any(|a| a == "E-WELLSYNC"),
                                ));
                                println!(
                                    "  {label}launch group {} {:?}: {:?} states={} terminal={} outputs={}{}{} {:.1}s",
                                    i + 1,
                                    g,
                                    v.classes(),
                                    v.states,
                                    v.terminal_states,
                                    v.distinct_outputs,
                                    if v.complete { "" } else { " INCOMPLETE" },
                                    if v.assumptions.iter().any(|a| a == "E-WELLSYNC") {
                                        " CERTIFIED"
                                    } else {
                                        ""
                                    },
                                    v.seconds
                                )
                            },
                        )
                    }),
                    _ if slices && certify => {
                        drop(m);
                        ptxv_sim::wellsync::check_sliced(&make, max_states, &mut |c, v| {
                            println!(
                                "  {label}slice cluster {c}: {:?} states={} terminal={} outputs={}{}{} {:.1}s",
                                v.classes(),
                                v.states,
                                v.terminal_states,
                                v.distinct_outputs,
                                if v.complete { "" } else { " INCOMPLETE" },
                                if v.assumptions.iter().any(|a| a == "E-WELLSYNC") {
                                    " CERTIFIED"
                                } else {
                                    ""
                                },
                                v.seconds
                            )
                        })
                    }
                    _ if certify => ptxv_sim::wellsync::check_certified(m, max_states),
                    _ if slices => {
                        drop(m);
                        ptxv_sim::explore::explore_symmetric(&make, max_states, &mut |c, v| {
                            println!(
                                "  {label}slice cluster {c}: {:?} states={} terminal={} outputs={}{} {:.1}s",
                                v.classes(),
                                v.states,
                                v.terminal_states,
                                v.distinct_outputs,
                                if v.complete { "" } else { " INCOMPLETE" },
                                v.seconds
                            )
                        })
                    }
                    _ => ptxv_sim::l1_cores::explore_placements(m, &mut |m| explore(m, max_states)),
                }
            };
            let make1 = || machine(case, k.clone()).expect("machine");
            let mut verdict = check(&make1, "");
            for a in &refuted {
                verdict.notes.push(format!(
                    "relies on {a}, REFUTED on GB300, by explicit opt-in (case assume_refuted / --assume-refuted / PTXV_ASSUME_REFUTED)"
                ));
            }
            // Level 2: classify PTX-model races / stale reads against the hardware model.
            let mut gbc = None;
            if gb300 {
                if let Some(why) = ptxv_sim::twolevel::wanted(&verdict, &m1.launch) {
                    let t0 = Instant::now();
                    let st = Arc::new(ptxv_sim::l1::L1Stats::default());
                    let make2 = || ptxv_sim::twolevel::hw_machine(&m1, &st);
                    // Launch groups: level 2 checks the groups level 1 got to (it stops at the
                    // first group with findings), and is decided when each of them is.
                    let n1 = group_log.borrow().iter().map(|g| g.0 + 1).max();
                    let upto = std::env::var("PTXV_GROUPS_UPTO").ok();
                    if let (Some(n), true) = (n1, groups.is_some()) {
                        if upto
                            .as_ref()
                            .and_then(|u| u.parse::<usize>().ok())
                            .is_none_or(|u| u > n)
                        {
                            unsafe { std::env::set_var("PTXV_GROUPS_UPTO", n.to_string()) };
                        }
                    }
                    group_log.borrow_mut().clear();
                    let mut v2 = check(&make2, "GB300 model: ");
                    match &upto {
                        Some(u) => unsafe { std::env::set_var("PTXV_GROUPS_UPTO", u) },
                        None => unsafe { std::env::remove_var("PTXV_GROUPS_UPTO") },
                    }
                    if let (Some(n), true) = (n1, groups.is_some()) {
                        let g2 = group_log.borrow();
                        // Decided when every run of each group was (one run per L1 placement).
                        v2.complete = (0..n).all(|i| {
                            g2.iter().any(|&(j, _)| j == i)
                                && g2.iter().filter(|&&(j, _)| j == i).all(|&(_, d)| d)
                        });
                    }
                    let m2 = make2();
                    let mut c = ptxv_sim::twolevel::classify(&mut verdict, &v2, &st, &m2);
                    c.why = why;
                    c.seconds = t0.elapsed().as_secs_f64();
                    gbc = Some(c);
                }
            }
            print_verdict(&v, f, &verdict, &expect, &refuted, gbc.as_ref());
            // Debugging / figures: PTXV_TRACE_DUMP=<dir> writes each finding's replayed witness.
            if let Some(d) = std::env::var_os("PTXV_TRACE_DUMP") {
                let p = Path::new(&d).join(format!("{v}.{f}.trace.json"));
                trace_dump::write(make1(), &verdict, &v, f, &p)?;
            }
            // Captured cases (tools/capture.py) carry the GPU's output bytes.
            let mut expect_report = Vec::new();
            for (i, o) in verdict.outputs.iter().enumerate() {
                for line in case.expect_report(o)? {
                    println!("    expect (output {i}): {line}");
                    expect_report.push(format!("output {i}: {line}"));
                }
            }
            // Debugging nondeterminism: every distinct terminal output, one file each.
            if std::env::var("PTXV_DUMP_OUTPUTS").is_ok() {
                for (i, o) in verdict.outputs.iter().enumerate() {
                    std::fs::write(dir.join(format!("{v}.{f}.output{i}.bin")), o)?;
                }
            }
            let got = verdict.classes();
            std::fs::write(
                dir.join(format!("{v}.{f}.check.json")),
                serde_json::to_string_pretty(&verdict)?,
            )?;
            rows.push(CheckRow {
                variant: v.clone(),
                front: f.into(),
                expect_report,
                pass: verdict.meets(&expect),
                expect,
                got,
                refuted_opt_in: refuted,
                verdict,
                gb300: gbc,
            });
        }
    }
    Ok(rows)
}

#[derive(Serialize)]
struct NumSummary {
    front: String,
    outputs: usize,
    thm1_all_exact: bool,
    spec_all_match: Option<bool>,
    cor1a_never_violated: bool,
    running_never_violated: bool,
    max_abs_err: f64,
    max_err_ulps: f64,
    mean_cor1a_over_err: f64,
    warnings: Vec<String>,
    output_hash: String,
}

pub(crate) fn cmd_num(
    case: &Case,
    variant: &str,
    out: &Option<PathBuf>,
) -> Result<(Vec<NumSummary>, Vec<u8>)> {
    cmd_num_with(case, variant, out, None)
}

fn cmd_num_with(
    case: &Case,
    variant: &str,
    out: &Option<PathBuf>,
    mma_model: Option<&str>,
) -> Result<(Vec<NumSummary>, Vec<u8>)> {
    let dir = out_dir(case, out)?;
    let mut case = case.clone();
    let suffix = match mma_model {
        Some(m) => {
            case.m.mma_model = Some(m.to_string());
            format!(".{m}")
        }
        None => String::new(),
    };
    let case = &case;
    let spec = ptxv_sim::spec_exact(case)?;
    let mut sums = Vec::new();
    let mut outs: Vec<Vec<u8>> = Vec::new();
    for f in ["ptx", "sass"] {
        if f == "ptx" && case.m.cubin.is_some() {
            continue; // binary-only case
        }
        let (k, _) = load(case, variant, f, &dir)?;
        let mut m = machine(case, k)?;
        m.num = Some(Box::new(ptxv_sim::num::NumState::with_mma_eps(
            m.launch.mma_model.eps(),
        )));
        let (m, findings, ok) = run_once(m);
        if !ok {
            bail!("numerics run of {variant} ({f}) did not terminate cleanly: {findings:?}");
        }
        let ns = m.num.as_ref().unwrap();
        let oi = m.launch.outputs[0];
        let mem = &m.gmem[oi];
        let n = mem.bytes.len() / 4;
        std::fs::write(dir.join(format!("{variant}.{f}.num.out.bin")), m.outputs())?;
        let mut recs = Vec::new();
        let mut spec_ok = true;
        for i in 0..n {
            let bits = u32::from_le_bytes([0, 1, 2, 3].map(|j| *mem.bytes.get(i * 4 + j)));
            let node = ns.mem_node(ptxv_sim::exec::Loc::Global {
                alloc: oi,
                off: (i * 4) as u64,
            });
            let r = ns.out_rec(i, bits, node);
            if let Some(s) = &spec {
                let ex = node
                    .map(|nd| ns.nodes[nd as usize].exact.clone())
                    .unwrap_or_else(|| ptxv_sim::exact::from(f32::from_bits(bits) as f64));
                spec_ok &= ptxv_sim::exact::is_zero(&ptxv_sim::exact::sub(&ex, &s[i]));
            }
            recs.push(r);
        }
        let mut csv = String::from(
            "index,hw,exact,abs_err,rel_err,err_ulps,bound_cor1a,bound_running,thm1_exact,n_leaves,n_roundings\n",
        );
        for r in &recs {
            csv += &format!(
                "{},{:e},{:e},{:e},{:e},{},{:e},{:e},{},{},{}\n",
                r.index,
                r.hw,
                r.exact,
                r.abs_err,
                r.rel_err,
                r.err_ulps,
                r.bound_cor1a,
                r.bound_running,
                r.thm1_exact,
                r.n_leaves,
                r.n_roundings
            );
        }
        std::fs::write(dir.join(format!("{variant}.{f}.num{suffix}.csv")), csv)?;
        std::fs::write(
            dir.join(format!("{variant}.{f}.num{suffix}.json")),
            serde_json::to_string(&recs)?,
        )?;
        let nz: Vec<&_> = recs.iter().filter(|r| r.abs_err > 0.0).collect();
        let s = NumSummary {
            front: f.into(),
            outputs: n,
            thm1_all_exact: recs.iter().all(|r| r.thm1_exact),
            spec_all_match: spec.as_ref().map(|_| spec_ok),
            cor1a_never_violated: recs.iter().all(|r| r.abs_err <= r.bound_cor1a),
            running_never_violated: recs.iter().all(|r| r.abs_err <= r.bound_running),
            max_abs_err: recs.iter().map(|r| r.abs_err).fold(0.0, f64::max),
            max_err_ulps: recs.iter().map(|r| r.err_ulps).fold(0.0, f64::max),
            mean_cor1a_over_err: if nz.is_empty() {
                0.0
            } else {
                nz.iter().map(|r| r.bound_cor1a / r.abs_err).sum::<f64>() / nz.len() as f64
            },
            warnings: ns.warnings.iter().cloned().collect(),
            output_hash: format!("{:016x}", fnv(&m.outputs())),
        };
        println!(
            "num {variant} {f:<4}: outputs={} thm1_exact={} spec_match={:?} cor1a_ok={} running_ok={} max_err={:.3e} ({:.2} ulp) bound/err(mean)={:.1} out={}",
            s.outputs,
            s.thm1_all_exact,
            s.spec_all_match,
            s.cor1a_never_violated,
            s.running_never_violated,
            s.max_abs_err,
            s.max_err_ulps,
            s.mean_cor1a_over_err,
            s.output_hash
        );
        for w in &s.warnings {
            println!("    warning: {w}");
        }
        std::fs::write(
            dir.join(format!("{variant}.{f}.out{suffix}.bin")),
            m.outputs(),
        )?;
        outs.push(m.outputs());
        sums.push(s);
    }
    if outs.len() == 2 {
        if outs[0] != outs[1] {
            println!("    MISMATCH: PTX and SASS simulations produce different outputs");
        } else {
            println!("    PTX and SASS simulations agree bitwise on all outputs");
        }
    }
    Ok((sums, outs.pop().unwrap()))
}

fn fnv(b: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in b {
        h = (h ^ *x as u64).wrapping_mul(0x100000001b3);
    }
    h
}

#[derive(Serialize, Clone)]
pub(crate) struct HwRow {
    variant: String,
    scrub: bool,
    mismatch_runs: Vec<usize>,
    gpu: u32,
    outcome: String,
    runs: usize,
    mismatches: usize,
    distinct: Vec<(usize, usize)>,
    seconds: f64,
}

/// CUDA_VISIBLE_DEVICES for a case: `gpu`, plus further free GPUs when the case's kernels run
/// on several GPUs (rung 1f). Device i of the case is the i-th listed GPU.
pub(crate) fn gpu_env(case: &Case, gpu: u32) -> Result<String> {
    let n = case.m.ndevices();
    if n <= 1 {
        return Ok(gpu.to_string());
    }
    let mut list = vec![gpu];
    list.extend(ptxv_hw::free_gpus()?.into_iter().filter(|&g| g != gpu));
    if list.len() < n as usize {
        bail!("the case needs {n} GPUs; only {} are free", list.len());
    }
    Ok(list[..n as usize]
        .iter()
        .map(|g| g.to_string())
        .collect::<Vec<_>>()
        .join(","))
}

/// Run one cubin `runs` times on `gpu` in a child process guarded by a watchdog.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hw_run_cubin(
    case: &Case,
    v: &str,
    cubin: &Path,
    runs: usize,
    timeout: f64,
    exp_path: &Path,
    scrub: bool,
    gpu: u32,
) -> Result<HwRow> {
    let exe = std::env::current_exe()?;
    let t0 = Instant::now();
    let mut child = std::process::Command::new(&exe)
        .args([
            "hw-worker",
            case.dir.to_str().unwrap(),
            cubin.to_str().unwrap(),
            "--runs",
            &runs.to_string(),
            "--expected",
            exp_path.to_str().unwrap(),
        ])
        .args(if scrub { vec!["--scrub"] } else { vec![] })
        .env("CUDA_VISIBLE_DEVICES", gpu_env(case, gpu)?)
        // Patched launch cubins sit next to the variant's (`load_with`).
        .env("PTXV_PATCHED_PREFIX", cubin.with_extension(""))
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    let deadline = Instant::now() + Duration::from_secs_f64(timeout);
    let status = loop {
        if let Some(s) = child.try_wait()? {
            break Some(s);
        }
        if Instant::now() > deadline {
            child.kill()?;
            child.wait()?;
            break None;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let secs = t0.elapsed().as_secs_f64();
    let v = v.to_string();
    Ok(match status {
        None => HwRow {
            variant: v,
            scrub,
            mismatch_runs: vec![],
            gpu,
            outcome: "hang (killed by watchdog)".into(),
            runs: 0,
            mismatches: 0,
            distinct: vec![],
            seconds: secs,
        },
        Some(s) => {
            let o = child.wait_with_output()?;
            if !s.success() {
                HwRow {
                    variant: v,
                    scrub,
                    mismatch_runs: vec![],
                    gpu,
                    outcome: format!("error: {}", String::from_utf8_lossy(&o.stderr).trim()),
                    runs: 0,
                    mismatches: 0,
                    distinct: vec![],
                    seconds: secs,
                }
            } else {
                let r: serde_json::Value = serde_json::from_slice(&o.stdout)?;
                let mism = r["mismatches"].as_u64().unwrap_or(0) as usize;
                let runs = r["runs"].as_u64().unwrap_or(0) as usize;
                let distinct: Vec<(usize, usize)> = serde_json::from_value(r["distinct"].clone())?;
                HwRow {
                    variant: v,
                    scrub,
                    mismatch_runs: serde_json::from_value(r["mismatch_runs"].clone())?,
                    gpu,
                    outcome: if mism == 0 {
                        "matches simulator bitwise".into()
                    } else {
                        format!("{mism}/{runs} runs differ from the correct output")
                    },
                    runs,
                    mismatches: mism,
                    distinct,
                    seconds: secs,
                }
            }
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn cmd_hw(
    case: &Case,
    variant: &str,
    runs: usize,
    timeout: f64,
    out: &Option<PathBuf>,
    expected: &[u8],
    scrub: bool,
    delays: &[String],
) -> Result<Vec<HwRow>> {
    let dir = out_dir(case, out)?;
    let exp_path = dir.join("expected.bin");
    std::fs::write(&exp_path, expected)?;
    let free = ptxv_hw::free_gpus()?;
    let Some(&gpu) = free.first() else {
        bail!("no free GPU (all have foreign compute processes)")
    };
    println!("hw: using GPU(s) {} (free: {free:?})", gpu_env(case, gpu)?);
    let mut rows = Vec::new();
    for v in variants(case, variant) {
        let front = if case.m.cubin.is_some() {
            "sass"
        } else {
            "ptx"
        };
        // The hardware needs only the cubin: a PTX case is not parsed (a delayed variant may
        // use forms the front-end does not model).
        let cubin = if front == "ptx" {
            ptx_cubin(case, &v, &dir, delays)?.1
        } else {
            load_with(case, &v, front, &dir, delays)?.1
        };
        let row = hw_run_cubin(case, &v, &cubin, runs, timeout, &exp_path, scrub, gpu)?;
        println!(
            "hw {:<6}{} {} ({} runs, {:.1}s) distinct outputs (count, words differing): {}{}",
            row.variant,
            if scrub { " [scrubbed smem]" } else { "" },
            row.outcome,
            row.runs,
            row.seconds,
            if row.distinct.len() > 4 {
                format!(
                    "{} distinct, first: {:?}",
                    row.distinct.len(),
                    &row.distinct[..4]
                )
            } else {
                format!("{:?}", row.distinct)
            },
            if row.mismatch_runs.is_empty() {
                String::new()
            } else {
                format!(" first failing runs: {:?}", row.mismatch_runs)
            }
        );
        rows.push(row);
    }
    Ok(rows)
}

fn cmd_probe(case: &Case, variant: &str, timeout: f64, out: &Option<PathBuf>) -> Result<()> {
    let dir = out_dir(case, out)?;
    let gpu = *ptxv_hw::free_gpus()?
        .first()
        .ok_or_else(|| anyhow::anyhow!("no free GPU"))?;
    let exe = std::env::current_exe()?;
    for v in variants(case, variant) {
        // Probes are compiled only: they may use instructions the simulator does not model.
        let ptx_path = dir.join(format!("{v}.ptx"));
        std::fs::write(&ptx_path, case.variant_ptx(&v)?)?;
        let cubin = dir.join(format!("{v}.cubin"));
        ptxv_sass::ptxas(&ptx_path, &cubin, &case.m.arch)?;
        let dump = dir.join(format!("{v}.probe.bin"));
        let _ = std::fs::remove_file(&dump);
        let mut child = std::process::Command::new(&exe)
            .args([
                "hw-worker",
                case.dir.to_str().unwrap(),
                cubin.to_str().unwrap(),
                "--runs",
                "1",
                "--dump",
                dump.to_str().unwrap(),
            ])
            .env("CUDA_VISIBLE_DEVICES", gpu_env(case, gpu)?)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()?;
        let deadline = Instant::now() + Duration::from_secs_f64(timeout);
        let status = loop {
            if let Some(s) = child.try_wait()? {
                break Some(s);
            }
            if Instant::now() > deadline {
                child.kill()?;
                child.wait()?;
                break None;
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let summary = &case.variant(&v)?.summary;
        match status {
            None => println!("probe {v:<12} HANG (killed after {timeout}s)  {summary}"),
            Some(s) if !s.success() => {
                let mut e = String::new();
                std::io::Read::read_to_string(&mut child.stderr.take().unwrap(), &mut e)?;
                println!("probe {v:<12} ERROR {}  {summary}", e.trim());
            }
            Some(_) => {
                let b = std::fs::read(&dump)?;
                let words: Vec<String> = b
                    .chunks(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()).to_string())
                    .collect();
                println!(
                    "probe {v:<12} completed, out = [{}]  {summary}",
                    words.join(", ")
                );
            }
        }
    }
    Ok(())
}

fn cmd_suite(case: &Case, runs: usize, out: &Option<PathBuf>) -> Result<()> {
    let dir = out_dir(case, out)?;
    println!("== {} ==\n{}\n", case.m.name, case.m.description.trim());
    println!("-- model checking (exhaustive over all schedules of this launch) --");
    let checks = cmd_check(
        case,
        "all",
        "both",
        2_000_000,
        out,
        false,
        false,
        &None,
        gb300_enabled(false),
    )?;
    println!("\n-- numerics (good variant) --");
    let (nums, expected) = cmd_num(case, "good", out)?;
    println!("\n-- hardware --");
    let mut hw = cmd_hw(case, "all", runs, 20.0, out, &expected, false, &[])?;
    hw.extend(cmd_hw(case, "all", runs, 20.0, out, &expected, true, &[])?);

    // Markdown summary.
    let mut md = format!(
        "# {} results\n\nGenerated by `ptxv suite`. Hardware: GB300 (sm_103a), {runs} launches per variant.\n\n",
        case.m.name
    );
    md += "| variant | bug | expected | PTX front-end | SASS front-end | states (ptx/sass) | hardware | hardware, smem scrubbed |\n|---|---|---|---|---|---|---|---|\n";
    for v in &case.m.variants {
        let c: Vec<&CheckRow> = checks.iter().filter(|r| r.variant == v.id).collect();
        let h = hw.iter().find(|r| r.variant == v.id && !r.scrub);
        let hs = hw.iter().find(|r| r.variant == v.id && r.scrub);
        let cell =
            |r: &CheckRow| format!("{} {}", if r.pass { "✅" } else { "❌" }, r.got.join(", "));
        md += &format!(
            "| {} | {} | {} | {} | {} | {}/{} | {} | {} |\n",
            v.id,
            v.summary,
            match &v.expect_sass {
                Some(e) if e != &v.expect =>
                    format!("PTX: {}; SASS: {}", v.expect.join(", "), e.join(", ")),
                _ => v.expect.join(", "),
            },
            cell(c[0]),
            cell(c[1]),
            c[0].verdict.states,
            c[1].verdict.states,
            h.map(|h| h.outcome.clone()).unwrap_or_default(),
            hs.map(|h| h.outcome.clone()).unwrap_or_default()
        );
    }
    md += "\n## Numerics (good variant)\n\n| front | outputs | Thm 1 exact | matches spec | Cor 1a never violated | running bound never violated | max error | max ulp | mean Cor1a/err |\n|---|---|---|---|---|---|---|---|---|\n";
    for n in &nums {
        md += &format!(
            "| {} | {} | {} | {:?} | {} | {} | {:.3e} | {:.2} | {:.1} |\n",
            n.front,
            n.outputs,
            n.thm1_all_exact,
            n.spec_all_match,
            n.cor1a_never_violated,
            n.running_never_violated,
            n.max_abs_err,
            n.max_err_ulps,
            n.mean_cor1a_over_err
        );
    }
    // Assumptions every verdict above depends on (ASSUMPTIONS.md).
    let mut ids: Vec<&String> = checks
        .iter()
        .flat_map(|r| r.verdict.assumptions.iter())
        .collect();
    ids.sort_by_key(|id| {
        ptxv_sim::assumptions::REGISTRY
            .iter()
            .position(|a| &a.id == id)
    });
    ids.dedup();
    md += "\n## Assumptions these verdicts rest on\n\nSee [ASSUMPTIONS.md](../../ASSUMPTIONS.md). Fronts: which front-end's verdicts depend on it.\n\n| id | status | assumption | fronts |\n|---|---|---|---|\n";
    for id in ids {
        let a = ptxv_sim::assumptions::get(id).unwrap();
        let mut fr: Vec<&str> = checks
            .iter()
            .filter(|r| r.verdict.assumptions.contains(id))
            .map(|r| r.front.as_str())
            .collect();
        fr.sort();
        fr.dedup();
        md += &format!(
            "| {} | {} | {} | {} |\n",
            a.id,
            a.status,
            a.title,
            fr.join(", ")
        );
    }
    md += "\n## Findings\n\n";
    for r in &checks {
        for f in &r.verdict.findings {
            md += &format!(
                "**{} / {} — {}**\n\n```\n{}\ntrace ({} steps):\n{}\n```\n\n",
                r.variant,
                r.front,
                f.kind,
                f.msg.trim_end(),
                f.trace.len(),
                f.trace
                    .iter()
                    .map(|s| format!("  - {s}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
    }
    std::fs::write(dir.join("summary.md"), &md)?;
    #[derive(Serialize)]
    struct S<'a> {
        checks: &'a [CheckRow],
        numerics: &'a [NumSummary],
        hardware: &'a [HwRow],
    }
    std::fs::write(
        dir.join("summary.json"),
        serde_json::to_string_pretty(&S {
            checks: &checks,
            numerics: &nums,
            hardware: &hw,
        })?,
    )?;
    let all_pass = checks.iter().all(|r| r.pass);
    println!(
        "\nsummary: {} ({}), written to {}",
        if all_pass {
            "all verdicts as expected"
        } else {
            "SOME VERDICTS UNEXPECTED"
        },
        checks.len(),
        dir.join("summary.md").display()
    );
    Ok(())
}

fn main() -> Result<()> {
    let r = run();
    if let Err(e) = &r {
        jsonout::fail(&format!("{e:#}"));
    }
    r
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    // Multi-node (PTXV_DIST_DIR): every rank runs the whole program; only rank 0 reports.
    if let Some(d) = ptxv_sim::dist::get() {
        if d.rank != 0 {
            use std::os::fd::AsRawFd;
            extern "C" {
                fn dup2(a: i32, b: i32) -> i32;
            }
            let null = std::fs::OpenOptions::new().write(true).open("/dev/null")?;
            unsafe { dup2(null.as_raw_fd(), 1) };
        }
    }
    match cli.cmd {
        Cmd::Check {
            case,
            variant,
            front,
            max_states,
            out,
            slices,
            certify,
            groups,
            json,
            residency,
            tmem_cotenant,
            assume_refuted,
            no_gb300,
        } => {
            let j = json.then(|| jsonout::begin("check"));
            if !assume_refuted.is_empty() {
                // build_launch reads it (ptxv_sim::assume_refuted), on every check path.
                let mut ids = assume_refuted.join(",");
                if let Ok(e) = std::env::var("PTXV_ASSUME_REFUTED") {
                    ids = format!("{e},{ids}");
                }
                unsafe { std::env::set_var("PTXV_ASSUME_REFUTED", ids) };
            }
            let mut c = Case::load(&case)?;
            if residency.as_deref() == Some("auto") || tmem_cotenant {
                let pass = cmd_coresidency(
                    &c,
                    &variant,
                    &front,
                    max_states,
                    &out,
                    &groups,
                    residency.as_deref() == Some("auto"),
                    tmem_cotenant,
                )?;
                if !pass {
                    std::process::exit(1);
                }
                return Ok(());
            }
            if let Some(r) = &residency {
                c.m.residency = Some(
                    r.parse()
                        .map_err(|_| anyhow::anyhow!("--residency wants N or auto"))?,
                );
            }
            let rows = cmd_check(
                &c,
                &variant,
                &front,
                max_states,
                &out,
                slices,
                certify,
                &groups,
                gb300_enabled(no_gb300),
            )?;
            let pass = rows.iter().all(|r| r.pass);
            if let Some(j) = j {
                j.emit(
                    "check",
                    serde_json::json!({
                        "case": case.canonicalize().unwrap_or(case.clone()),
                        "name": c.m.name,
                        "pass": pass,
                        "rows": rows,
                    }),
                );
            }
            if !pass {
                std::process::exit(1);
            }
        }
        Cmd::Num {
            case,
            variant,
            mma_model,
            out,
        } => {
            cmd_num_with(&Case::load(&case)?, &variant, &out, mma_model.as_deref())?;
        }
        Cmd::Hw {
            case,
            variant,
            runs,
            timeout,
            scrub,
            delay,
            out,
        } => {
            let c = Case::load(&case)?;
            // PTXV_HW_EXPECT=variant: each variant against its own deterministic simulator
            // run (a numerics mutant's wrong output, rung 3b).
            if std::env::var("PTXV_HW_EXPECT").as_deref() == Ok("variant") {
                let dir = out_dir(&c, &out)?;
                for v in variants(&c, &variant) {
                    let (k, _) = load(&c, &v, "sass", &dir)?;
                    let (m, _, ok) = ptxv_sim::explore::run_once(machine(&c, k)?);
                    if !ok {
                        bail!("{v} does not terminate in the simulator");
                    }
                    cmd_hw(&c, &v, runs, timeout, &out, &m.outputs(), scrub, &delay)?;
                }
                return Ok(());
            }
            // PTXV_HW_EXPECT=run: the expected bytes are one deterministic simulator run of
            // the good variant (explore::run_once) instead of the numerics run, for launch
            // groups too large for `num` (rung 4b: 28 launches on 2 GPUs).
            let expected = if std::env::var("PTXV_HW_EXPECT").as_deref() == Ok("run") {
                let dir = out_dir(&c, &out)?;
                let (k, _) = load(&c, "good", "sass", &dir)?;
                let (m, _, ok) = ptxv_sim::explore::run_once(machine(&c, k)?);
                if !ok {
                    bail!("the good variant does not terminate in the simulator");
                }
                m.outputs()
            } else {
                cmd_num(&c, "good", &out)?.1
            };
            cmd_hw(&c, &variant, runs, timeout, &out, &expected, scrub, &delay)?;
        }
        Cmd::Equiv {
            file,
            out,
            no_hw,
            trials,
            per_output,
            no_sweep_cache,
            json,
            emit_bound,
        } => {
            let j = json.then(|| jsonout::begin("equiv"));
            let (ok, results) = equiv::cmd_equiv(
                &file,
                &out,
                no_hw,
                trials,
                per_output,
                no_sweep_cache,
                emit_bound.as_deref(),
            )?;
            if let Some(j) = j {
                j.emit(
                    "equiv",
                    serde_json::json!({
                        "file": file.canonicalize().unwrap_or(file.clone()),
                        "ok": ok,
                        "bound_file": emit_bound,
                        "candidates": results,
                    }),
                );
            }
            if !ok {
                std::process::exit(1);
            }
        }
        Cmd::Errdist {
            path,
            inputs,
            compile,
            model,
            json,
            html,
            candidate,
            mc,
            sim_trials,
            gpu_trials,
            per_output,
            threads,
            out,
            explore,
            pre_round,
            uniform,
            errstate,
            tier1,
            no_tier1,
        } => errdist::cmd(&errdist::Args {
            path,
            inputs,
            compile,
            model,
            json,
            html,
            candidate,
            mc,
            sim_trials,
            gpu_trials,
            per_output,
            threads,
            out,
            explore,
            pre_round,
            uniform,
            errstate,
            tier1: tier1 && !no_tier1,
        })?,
        Cmd::Compare {
            reference,
            candidate,
            out,
            max_states,
            trials,
            segments,
            sweep_cache,
            hw,
            ref_label,
            cand_label,
            function,
            json,
            emit_bound,
        } => {
            let j = json.then(|| jsonout::begin("compare"));
            let out = out.unwrap_or_else(|| {
                std::env::temp_dir().join(format!(
                    "ptxv-compare-{}-{}",
                    reference
                        .file_name()
                        .map_or("".into(), |d| d.to_string_lossy()),
                    candidate
                        .file_name()
                        .map_or("".into(), |d| d.to_string_lossy())
                ))
            });
            let o = compare::Opts {
                out,
                max_states,
                trials,
                segments,
                sweep_cache,
                hw,
                ref_label,
                cand_label,
                function,
                emit_bound,
            };
            compare::cmd_compare(&reference, &candidate, &o)?;
            if let Some(j) = j {
                let rp = o.out.join("report.json");
                let mut v: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(&rp)?)?;
                v["report_dir"] = serde_json::json!(o.out.canonicalize()?);
                j.emit("compare", v);
            }
        }
        Cmd::Suite { case, runs, out } => cmd_suite(&Case::load(&case)?, runs, &out)?,
        Cmd::ErrdistGpuWorker {
            case,
            cubin,
            dir,
            n,
            tag,
        } => errdist::gpu_worker(&case, &cubin, &dir, n, &tag)?,
        Cmd::HwWorker {
            case,
            image,
            runs,
            expected,
            scrub,
            input,
            dump,
        } => {
            let c = Case::load(&case)?;
            let img = std::fs::read(&image)?;
            let exp = match expected {
                Some(p) => Some(std::fs::read(p)?),
                None => None,
            };
            let mut inputs = std::collections::BTreeMap::new();
            for i in &input {
                let (n, p) = i
                    .split_once('=')
                    .ok_or_else(|| anyhow::anyhow!("--input wants NAME=PATH"))?;
                inputs.insert(n.to_string(), std::fs::read(p)?);
            }
            let r = ptxv_hw::run_with(&c, &img, runs, exp.as_deref(), scrub, &inputs)?;
            if let Some(d) = dump {
                std::fs::write(d, &r.first_output)?;
            }
            println!("{}", serde_json::to_string(&r)?);
        }
        Cmd::CalibMma { case, out, only } => {
            let c = Case::load(&case)?;
            // A byte-operand probe (probes/mma_f8) calibrates the narrow kinds.
            if c.buffer("A")?.dtype == "u8" {
                calib_lp::run(&c, &out, only.as_deref())?
            } else {
                calib::run(&c, &out)?
            }
        }
        Cmd::Probe {
            case,
            variant,
            timeout,
            out,
        } => cmd_probe(&Case::load(&case)?, &variant, timeout, &out)?,
        Cmd::SbSweep {
            case,
            runs,
            timeout,
            out,
        } => sweep::run(&Case::load(&case)?, runs, timeout, &out)?,
        Cmd::CvtSweep {
            dir,
            gpu,
            record,
            reuse,
        } => cvt_sweep::run(&dir, gpu, record, reuse)?,
        Cmd::MufuSweep { op, input, sample } => mufu_sweep::run(&op, &input, sample.as_deref())?,
    }
    Ok(())
}

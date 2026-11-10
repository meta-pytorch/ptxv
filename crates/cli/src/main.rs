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
use ptxv_jsonout as jsonout;
use ptxv_sim::explore::Verdict;
use ptxv_sim::explore::explore;
use ptxv_sim::explore::run_once;
use ptxv_sim::machine::Machine;
use serde::Serialize;

mod calib;
mod calib_lp;
mod compare;
mod concolic;
mod cvt_sweep;
mod equiv;
mod errdist;
mod errdist_html;
mod mufu_sweep;
mod nonfinite;
mod sweep;
mod trace_dump;
mod vcache;

// The explorer's threads clone and drop machine pages at a high rate; glibc malloc's arenas
// contend under that (and mimalloc is faster single-threaded too). mimalloc 2's arenas stop
// growing near 400 GB of live heap (on 1.3 TB hosts: "cannot use OS memory
// since it is too large with respect to the maximum object size", then allocation failure),
// so PTXV_SYSTEM_ALLOC=1 selects the system allocator for the largest explorations. The choice
// is made once, at the first allocation (getenv allocates nothing), so every block is freed
// by the allocator that made it.
#[global_allocator]
static GLOBAL: Alloc = Alloc;

struct Alloc;

static ALLOC_SYS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

unsafe extern "C" {
    fn getenv(name: *const std::ffi::c_char) -> *const std::ffi::c_char;
}

impl Alloc {
    #[inline]
    fn system() -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        match ALLOC_SYS.load(Relaxed) {
            1 => false,
            2 => true,
            _ => {
                // SAFETY: a NUL-terminated name; the result is only compared with null.
                let set = unsafe { !getenv(c"PTXV_SYSTEM_ALLOC".as_ptr()).is_null() };
                ALLOC_SYS.store(if set { 2 } else { 1 }, Relaxed);
                set
            }
        }
    }
}

// SAFETY: forwards to one of two allocators, fixed before the first allocation.
unsafe impl std::alloc::GlobalAlloc for Alloc {
    unsafe fn alloc(&self, l: std::alloc::Layout) -> *mut u8 {
        unsafe {
            if Self::system() {
                std::alloc::System.alloc(l)
            } else {
                mimalloc::MiMalloc.alloc(l)
            }
        }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: std::alloc::Layout) {
        unsafe {
            if Self::system() {
                std::alloc::System.dealloc(p, l)
            } else {
                mimalloc::MiMalloc.dealloc(p, l)
            }
        }
    }
    unsafe fn alloc_zeroed(&self, l: std::alloc::Layout) -> *mut u8 {
        unsafe {
            if Self::system() {
                std::alloc::System.alloc_zeroed(l)
            } else {
                mimalloc::MiMalloc.alloc_zeroed(l)
            }
        }
    }
    unsafe fn realloc(&self, p: *mut u8, l: std::alloc::Layout, n: usize) -> *mut u8 {
        unsafe {
            if Self::system() {
                std::alloc::System.realloc(p, l, n)
            } else {
                mimalloc::MiMalloc.realloc(p, l, n)
            }
        }
    }
}

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
        /// With --certify: representative CTAs (E-CTA-REP): certify a few CTAs and replay the
        /// position-dependent values of the others; the launch is checked as before when the
        /// reduction refuses (also PTXV_REP=1).
        #[arg(long)]
        rep: bool,
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
        /// Hardware profile of the two-level verdict: when the strict check reports anything
        /// undefined (race, violation, contract deviation, undefined values, ...), the SASS
        /// front is checked again with every probe-backed assumption of the profile on, and
        /// both levels are reported (`ptx_contract`, `sass_gb300`). `gb300` (default) or
        /// `none` (also PTXV_HW_PROFILE).
        #[arg(long)]
        hw_profile: Option<String>,
        /// Scan every terminal state's float outputs for NaN / Inf: a `nonfinite-output`
        /// finding when every float input element is finite (else a note), with the
        /// instructions that produced the first non-finite values from operands without one.
        #[arg(long)]
        nonfinite: bool,
        /// --nonfinite, and a finding even when an input already holds NaN / Inf.
        #[arg(long)]
        nonfinite_strict: bool,
        /// Skip the input-coverage notes: one deterministic run that records which sides of
        /// input-dependent comparisons any thread took (also PTXV_NO_INPUT_COVERAGE).
        #[arg(long)]
        no_input_coverage: bool,
        /// Input coverage only: the one deterministic run with the provenance shadow, no
        /// exploration and no verdict (rows[].got = ["coverage-only"]).
        #[arg(long)]
        coverage_only: bool,
        /// Concolic input exploration (ptxv_sim::symb / solve): expressions over the input for
        /// every input-dependent condition, solvers for the untaken sides, every witness
        /// replayed; flips go in rows[].input_coverage.concolic and (up to
        /// PTXV_CONCOLIC_CHECKS) are checked as mutated inputs. Off by default.
        #[arg(long)]
        concolic: bool,
        /// Check again on mutated inputs: for up to N untaken input-dependent comparison
        /// sides that compare one input word against a constant, write a value that flips
        /// the comparison into --mutate-elements elements and run the check on that input.
        /// Results are reported apart (`on-mutated-input`), never in the verdict.
        #[arg(long, default_value_t = 0)]
        mutate_inputs: usize,
        /// Elements per mutation (--mutate-inputs), spread over threads of different warps.
        #[arg(long, default_value_t = 2)]
        mutate_elements: usize,
        /// A mutation given by hand, checked like the --mutate-inputs ones: `name[i]=v`
        /// element writes, comma-separated (e.g. 'ok[0]=0,ok[432]=0'). Repeatable.
        #[arg(long)]
        mutate_input: Vec<String>,
        /// Input coverage's own wall-clock cap in seconds (also PTXV_COVERAGE_SECONDS; default
        /// 1800, none with --coverage-only). It comes before any exploration (on the plain
        /// --certify path it rides on the certificate's run); past the cap (or its memory
        /// cap, PTXV_COVERAGE_MAX_RSS, default 75% of the process's memory budget) it stops
        /// recording with a note and the verdict is unaffected.
        #[arg(long)]
        coverage_seconds: Option<f64>,
        /// Total wall-clock budget of the --mutate-inputs checks in seconds (default 1800):
        /// no mutated check starts after it, and a running one stops there, incomplete.
        #[arg(long, default_value_t = 1800.0)]
        mutate_budget_seconds: f64,
        /// Total state budget of the --mutate-inputs checks (default 2 x --max-states): each
        /// mutated check gets what is left, at most --max-states.
        #[arg(long)]
        mutate_budget_states: Option<usize>,
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
    /// The GPU is picked among those PTXV_GPUS lists (nvidia-smi indices), else among those
    /// CUDA_VISIBLE_DEVICES names (indices or GPU- UUID prefixes), else among all free ones.
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
        /// Skip the simulator pre-pass: compare every GPU run with the GPU output captured
        /// with the case (each output buffer's `expect` bytes) instead of a `num` run. For
        /// cases whose numerics run is too slow; needs `expect` covering every output buffer.
        /// The comparison is bitwise: a kernel that accumulates with atomics in a varying
        /// order differs from the capture without being wrong.
        #[arg(long)]
        no_num: bool,
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
        /// Products per MMA step (default: 16 for bf16, 8 for tf32; 8 for a bf16 m16n8k8
        /// case, probes/bmma8).
        #[arg(long)]
        step: Option<usize>,
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

/// The default output directory of a case under `root`: `<root>/<case name>`, owned by the
/// case directory recorded in its `.case` file. A directory of that name that belongs to
/// another case (two cases whose names agree, e.g. truncated kernel names) is not reused:
/// the case gets `<root>/<name>-<hash of its directory>` instead, so a run never reads
/// another case's cubin.
fn default_out(root: &Path, case: &Case) -> Result<PathBuf> {
    let me = std::fs::canonicalize(&case.dir).unwrap_or_else(|_| case.dir.clone());
    let me = me.to_string_lossy().to_string();
    let claim = |d: &Path| -> Result<bool> {
        std::fs::create_dir_all(d)?;
        let f = d.join(".case");
        match std::fs::read_to_string(&f) {
            Ok(owner) => Ok(owner.trim() == me),
            Err(_) => {
                std::fs::write(&f, format!("{me}\n"))?;
                Ok(true)
            }
        }
    };
    let plain = root.join(&case.m.name);
    if ptxv_sim::dist::get().is_some() || claim(&plain)? {
        return Ok(plain);
    }
    // FNV-1a of the case directory.
    let h = me.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
    });
    let d = root.join(format!("{}-{:08x}", case.m.name, h as u32));
    claim(&d)?;
    Ok(d)
}

pub(crate) fn out_dir(case: &Case, out: &Option<PathBuf>) -> Result<PathBuf> {
    let d = match out {
        Some(o) => o.clone(),
        None => {
            let root = std::env::var("PTXV_ROOT")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("."));
            default_out(&root.join("out"), case)?
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
    let used = ptxv_sass::ptxas_with(
        &ptx_path,
        &cubin,
        &case.m.arch,
        case.m.ptxas_toolkit.as_deref(),
    )
    .with_context(|| format!("ptxas on variant {variant}"))?;
    // The rebuilt code vs the executed cubin, for a variant whose PTX is the captured one.
    let captured = std::fs::read_to_string(case.dir.join(&case.m.ptx)).unwrap_or_default();
    if delays.is_empty() && ptx == captured {
        if let Some(r) = ptx_rebuild(case, &cubin, used.as_deref()) {
            PTX_REBUILD
                .lock()
                .unwrap()
                .insert(rebuild_key(case, variant), r);
        }
    }
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

/// Rebuild status of a PTX case variant's cubin, keyed by `rebuild_key`.
static PTX_REBUILD: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<String, PtxRebuild>>,
> = std::sync::LazyLock::new(Default::default);

fn rebuild_key(case: &Case, variant: &str) -> String {
    format!("{}#{variant}", case.dir.display())
}

/// How the cubin ptxv compiled from a captured case's PTX compares with the cubin the GPU
/// executed (`executed_cubin`): `status` is "match" (every launched kernel's `.text` equal) or
/// "mismatch" (the verdicts then check a binary the GPU did not run).
#[derive(Clone, Debug, serde::Serialize)]
struct PtxRebuild {
    status: &'static str,
    /// Toolkit of the ptxas that built the executed cubin, and of the one ptxv used.
    executed_toolkit: Option<String>,
    rebuilt_toolkit: Option<String>,
    /// Launched kernels whose code differs.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    differing: Vec<String>,
}

fn ptx_rebuild(case: &Case, rebuilt: &Path, used: Option<&str>) -> Option<PtxRebuild> {
    // (entry in the rebuilt cubin, executed cubin, entry in it) per launch: a launch's own
    // `executed_cubin` / `executed_entry` (launches from several modules), else the case's.
    let top = case.m.executed_cubin.as_deref();
    let launches: Vec<(&str, Option<&str>, &str)> =
        std::iter::once((case.m.entry.as_str(), top, case.m.entry.as_str()))
            .chain(case.m.launches.iter().map(|l| {
                (
                    l.entry.as_str(),
                    l.executed_cubin.as_deref().or(top),
                    l.executed_entry.as_deref().unwrap_or(&l.entry),
                )
            }))
            .collect();
    if launches.iter().all(|(_, c, _)| c.is_none()) {
        return None;
    }
    let new = std::fs::read(rebuilt).ok()?;
    let mut exes: std::collections::BTreeMap<&str, Option<Vec<u8>>> = Default::default();
    let mut differing = Vec::new();
    let mut toolkit = None;
    for (entry, cubin, exe_entry) in launches {
        let a = cubin.and_then(|c| {
            exes.entry(c)
                .or_insert_with(|| std::fs::read(case.dir.join(c)).ok())
                .as_deref()
                .and_then(|exe| {
                    toolkit = toolkit.take().or_else(|| ptxv_sass::cubin_toolkit(exe));
                    ptxv_sass::kernel_text(exe, exe_entry)
                })
        });
        if a.is_none() || a != ptxv_sass::kernel_text(&new, entry) {
            differing.push(entry.to_string());
        }
    }
    differing.sort();
    differing.dedup();
    Some(PtxRebuild {
        status: if differing.is_empty() {
            "match"
        } else {
            "mismatch"
        },
        executed_toolkit: toolkit.or_else(|| case.m.ptxas_toolkit.clone()),
        rebuilt_toolkit: used.map(String::from),
        differing,
    })
}

/// The image note of a PTX case variant whose rebuilt cubin differs from the executed one.
fn rebuild_note(case: &Case, variant: &str) -> Option<String> {
    let r = PTX_REBUILD
        .lock()
        .unwrap()
        .get(&rebuild_key(case, variant))
        .cloned()?;
    (r.status == "mismatch").then(|| {
        format!(
            "ptx_rebuild: mismatch: the cubin ptxv built from the PTX (ptxas {}) differs from the \
             executed one (ptxas {}) in {}: verdicts check a binary the GPU did not run",
            r.rebuilt_toolkit.as_deref().unwrap_or("?"),
            r.executed_toolkit.as_deref().unwrap_or("?"),
            r.differing.join(", ")
        )
    })
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
    let mut k = if ks.len() == 1 {
        ks.pop().unwrap()
    } else {
        Kernel::concat(ks)
    };
    // A rebuilt cubin that differs from the executed one: not a verdict on what the GPU ran.
    if let Some(n) = rebuild_note(case, variant) {
        k.static_image.get_or_insert(n);
    }
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

/// `--mutate-input` specs.
static MUTATE_SPECS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();

/// `--coverage-only`.
static COVERAGE_ONLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
/// Input coverage (coverage.rs): (notes on, --mutate-inputs, --mutate-elements).
static INPUT_COV: std::sync::OnceLock<(bool, usize, usize)> = std::sync::OnceLock::new();

/// Budgets of input coverage and mutation (`--coverage-seconds`, `--mutate-budget-*`).
#[derive(Clone, Copy, Default)]
struct CovBudgets {
    coverage_seconds: Option<f64>,
    mutate_seconds: f64,
    mutate_states: Option<usize>,
}

static COV_BUDGETS: std::sync::OnceLock<CovBudgets> = std::sync::OnceLock::new();

/// A's caps for a check (`coverage_only`: the coverage run is the whole job, no time cap
/// unless one is given).
fn coverage_budget(coverage_only: bool) -> ptxv_sim::coverage::Budget {
    let b = COV_BUDGETS.get().copied().unwrap_or_default();
    let env_secs = std::env::var("PTXV_COVERAGE_SECONDS")
        .ok()
        .and_then(|s| s.parse::<f64>().ok());
    let seconds = b
        .coverage_seconds
        .or(env_secs)
        .or((!coverage_only).then_some(1800.0))
        .filter(|&s| s > 0.0);
    let rss = match std::env::var("PTXV_COVERAGE_MAX_RSS") {
        Ok(v) => parse_rss(&v),
        Err(_) => ptxv_sim::membudget::rss_limit().map(|l| l / 4 * 3),
    };
    ptxv_sim::coverage::Budget { seconds, rss }
}

/// `"96G"` / `"512M"` / bytes; 0: none.
fn parse_rss(s: &str) -> Option<u64> {
    let s = s.trim();
    let (n, mul) = match s.chars().last()?.to_ascii_uppercase() {
        'K' => (&s[..s.len() - 1], 1u64 << 10),
        'M' => (&s[..s.len() - 1], 1 << 20),
        'G' => (&s[..s.len() - 1], 1 << 30),
        'T' => (&s[..s.len() - 1], 1 << 40),
        _ => (s, 1),
    };
    let v: f64 = n.trim().parse().ok()?;
    let b = (v * mul as f64) as u64;
    (b > 0).then_some(b)
}

unsafe extern "C" {
    /// mimalloc: return freed memory to the OS (`force`: all of it).
    fn mi_collect(force: bool);
}

/// Hands the memory freed after input coverage back to the OS, so the exploration's memory
/// budget (on the resident set) starts where it would have without it.
fn release_freed_memory() {
    if std::env::var_os("PTXV_SYSTEM_ALLOC").is_none() {
        // SAFETY: mimalloc is the global allocator (PTXV_SYSTEM_ALLOC unset); mi_collect
        // only returns free pages.
        unsafe { mi_collect(true) };
    }
}

/// One check on a mutated input (`--mutate-inputs`): its own evidence, never the verdict.
#[derive(Serialize)]
struct MutatedRow {
    /// Where the mutation comes from: `B` (--mutate-inputs), `concolic` (a replayed
    /// --concolic witness) or `given` (--mutate-input).
    source: &'static str,
    /// `buffer[element]=value, ...`.
    mutation: String,
    /// The comparison it flips and the side it makes taken (null: given with --mutate-input).
    pc: Option<u32>,
    /// Every element written stays within its buffer's declared invariant (checked on the
    /// capture): evidence class `on-mutated-input (within declared invariants)`.
    within_invariants: bool,
    at: String,
    makes: bool,
    classes: Vec<String>,
    complete: bool,
    certified: bool,
    states: usize,
    findings: Vec<ptxv_sim::machine::Finding>,
    notes: Vec<String>,
    /// Not checked, and why (`coverage::Pass::Unchanged`).
    #[serde(skip_serializing_if = "Option::is_none")]
    skipped: Option<String>,
    seconds: f64,
}

/// The case's declared input invariants (`buffers[].invariant`), checked on the captured bytes
/// of `m` (coverage.rs `Domain::new`), with their notes.
fn input_domain(case: &Case, m: &Machine) -> Result<(ptxv_sim::coverage::Domain, Vec<String>)> {
    let invs: Vec<ptxv_sim::coverage::Invariant> = case
        .m
        .buffers
        .iter()
        .filter_map(|b| {
            let i = b.invariant.as_ref()?;
            Some(ptxv_sim::coverage::Invariant {
                buffer: b.name.clone(),
                values: i.values.clone(),
                min: i.min,
                max: i.max,
                elem: b.elem_bytes() as u8,
                signed: b.dtype.starts_with('i'),
            })
        })
        .collect();
    let float_inv: Vec<&str> = case
        .m
        .buffers
        .iter()
        .filter(|b| b.invariant.is_some() && !b.dtype.starts_with('i') && !b.dtype.starts_with('u'))
        .map(|b| b.name.as_str())
        .collect();
    if !float_inv.is_empty() {
        anyhow::bail!(
            "input invariants are for integer buffers: {}",
            float_inv.join(", ")
        );
    }
    Ok(ptxv_sim::coverage::Domain::new(m, invs))
}

/// `--hw-profile` (`PTXV_HW_PROFILE`, default "gb300").
static HW_PROFILE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

fn hw_profile() -> (String, Option<&'static [ptxv_sim::hwprofile::Item]>) {
    let name = HW_PROFILE
        .get()
        .cloned()
        .or_else(|| std::env::var("PTXV_HW_PROFILE").ok())
        .unwrap_or_else(|| "gb300".into());
    let p = ptxv_sim::hwprofile::named(&name);
    (name, p)
}

/// One hardware assumption a second-level verdict used.
#[derive(Serialize, Clone)]
struct HwUsed {
    id: String,
    probe: String,
    /// The contract findings it discharged (signatures); empty for a model switch.
    discharged: Vec<String>,
}

/// The two-level verdict: the strict (contract) check, and the SASS front under the hardware
/// profile. Both always appear; the second never replaces the first.
#[derive(Serialize, Clone)]
struct TwoLevel {
    /// The strict check's classes (of this row's front).
    ptx_contract: String,
    contract_front: String,
    /// "certified" | "clean" | "incomplete" | the remaining classes.
    sass_gb300: String,
    hw_profile: String,
    hw_assumptions: Vec<HwUsed>,
    sass_classes: Vec<String>,
    sass_complete: bool,
    sass_certified: bool,
    sass_states: usize,
    sass_findings: Vec<ptxv_sim::machine::Finding>,
    sass_notes: Vec<String>,
    seconds: f64,
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
    /// False when the analyzed control bits are not known to be the ones GB300 executes (an
    /// sm_100 / sm_100f cubin, which the driver may reschedule at load: `Kernel::static_image`):
    /// not a GB300 verdict.
    executed_image: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_note: Option<String>,
    /// PTX case with an executed cubin: whether the cubin ptxv rebuilt matches it.
    #[serde(skip_serializing_if = "Option::is_none")]
    ptx_rebuild: Option<PtxRebuild>,
    verdict: Verdict,
    /// The second level (ptxv_sim::twolevel): the hardware-model check that classified the
    /// verdict's race / stale findings (each finding's `gb300`). None: not needed or disabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    gb300: Option<ptxv_sim::twolevel::Gb300Check>,
    /// The two-level verdict (`--hw-profile`): set when the strict check reported anything
    /// undefined.
    #[serde(skip_serializing_if = "Option::is_none")]
    two_level: Option<TwoLevel>,
    /// `--nonfinite`: NaN / Inf scan of the outputs and their producing instructions.
    #[serde(skip_serializing_if = "Option::is_none")]
    nonfinite: Option<nonfinite::Report>,
    /// Input coverage (coverage.rs): input-dependent comparison sides no thread took in one
    /// run of the captured input. A note, not part of the verdict.
    #[serde(skip_serializing_if = "Option::is_none")]
    input_coverage: Option<ptxv_sim::coverage::Coverage>,
    /// `--mutate-inputs`: checks on mutated inputs (evidence class `on-mutated-input`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    mutated_inputs: Vec<MutatedRow>,
    /// `--mutate-inputs`: untaken sides not mutated, and why.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    mutation_refusals: Vec<ptxv_sim::coverage::Refusal>,
    /// The --mutate-inputs budget stopped the mutated checks: how many were left.
    #[serde(skip_serializing_if = "Option::is_none")]
    mutation_budget: Option<String>,
}

fn print_verdict(
    variant: &str,
    front: &str,
    v: &Verdict,
    expect: &[String],
    refuted: &[String],
    gb: Option<&ptxv_sim::twolevel::Gb300Check>,
    image: Option<&str>,
) {
    let got = v.classes();
    let pass = v.meets(expect);
    println!(
        "{} {variant:<6} {front:<4} {:<26} states={:<7} terminal={:<5} outputs={} {}{:.2}s{}",
        if pass { "PASS" } else { "FAIL" },
        format!("[{}]", got.join(",")),
        v.states,
        v.terminal_states,
        v.distinct_outputs,
        if v.complete { "" } else { "INCOMPLETE " },
        v.seconds,
        image
            .map(|n| format!("  NOT A GB300 VERDICT: {n}"))
            .unwrap_or_default()
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
                    "latent-needs-divergence" =>
                        "intra-warp, latent: needs divergence (or a compiler reordering)",
                    _ => "not decided",
                },
                g.reason
            );
        }
        let n = f.trace.len();
        if n == 0 && f.kind == "note" {
            continue;
        }
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
    nonfinite: Option<bool>,
) -> Result<Vec<CheckRow>> {
    let dir = out_dir(case, out)?;
    let mut rows = Vec::new();
    let (profile_name, profile) = hw_profile();
    for v in variants(case, variant) {
        // The variant's SASS check under the hardware profile, shared by its fronts' rows.
        let mut level2: Option<(Verdict, Vec<HwUsed>, f64)> = None;
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
            if COVERAGE_ONLY.get().copied().unwrap_or(false) {
                let (mut cov, sym) = concolic::analyze(&m, coverage_budget(true));
                println!(
                    "COVERAGE {v} {f}: {} input-dependent comparison(s), {} side(s) not taken, input-independent={}, {} unexecuted instruction(s) ({} behind input-dependent conditions, {} behind parameter conditions), terminal={} {:.1}s",
                    cov.sites,
                    cov.untaken.len(),
                    cov.input_independent,
                    cov.unexecuted_instructions,
                    cov.behind_input,
                    cov.behind_parameter,
                    cov.terminal,
                    cov.seconds
                );
                for u in cov.untaken.iter().take(20) {
                    println!("    note: {}", u.note);
                }
                let k_mut = INPUT_COV.get().map_or(2, |x| x.2);
                concolic::search(&m, &mut cov, sym, &input_domain(case, &m)?.0, k_mut);
                rows.push(CheckRow {
                    variant: v.clone(),
                    front: f.into(),
                    expect_report: vec![],
                    pass: true,
                    expect,
                    got: vec!["coverage-only".into()],
                    refuted_opt_in: refuted,
                    executed_image: k.static_image.is_none(),
                    image_note: k.static_image.clone(),
                    ptx_rebuild: PTX_REBUILD
                        .lock()
                        .unwrap()
                        .get(&rebuild_key(case, v.as_str()))
                        .cloned(),
                    verdict: Verdict {
                        states: 1,
                        transitions: 0,
                        complete: false,
                        terminal_states: cov.terminal as usize,
                        distinct_outputs: 0,
                        findings: vec![],
                        assumptions: vec![],
                        notes: vec![],
                        seconds: cov.seconds,
                        outputs: vec![],
                    },
                    gb300: None,
                    two_level: None,
                    nonfinite: None,
                    input_coverage: Some(cov),
                    mutated_inputs: vec![],
                    mutation_refusals: vec![],
                    mutation_budget: None,
                });
                continue;
            }
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
                    for line in case.expect_report_tol(&m.outputs(), m.mc_inexact)? {
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
            // The state cap of the next `check` (--max-states; less for a mutated input when
            // the --mutate-inputs budget has less left).
            let state_cap = std::cell::Cell::new(max_states);
            let check = |make: &dyn Fn() -> Machine, label: &str| -> Verdict {
                let m = make();
                // PTXV_EXPLORER=dpor selects partial-order reduction (crates/sim/src/dpor.rs);
                // the default is the stateful explorer. Both are cross-checked in the test suite.
                match std::env::var("PTXV_EXPLORER").as_deref() {
                    // E-L1-CORES: one run per SM placement when `l1_coresident` (else one run).
                    Ok("dpor") => ptxv_sim::l1_cores::explore_placements(m, &mut |m| {
                        ptxv_sim::dpor::explore_dpor(m, state_cap.get())
                    }),
                    _ if groups.is_some() => ptxv_sim::l1_cores::explore_placements(m, &mut |m| {
                        if slices {
                            ptxv_sim::groups::SLICES
                                .store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        ptxv_sim::groups::check_spec(
                            m,
                            groups.as_deref().unwrap(),
                            state_cap.get(),
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
                    _ if slices
                        && certify
                        && !std::env::var("PTXV_REP").is_ok_and(|v| v == "1") =>
                    {
                        drop(m);
                        ptxv_sim::wellsync::check_sliced(&make, state_cap.get(), &mut |c, v| {
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
                    _ if certify && std::env::var("PTXV_REP").is_ok_and(|v| v == "1") => {
                        drop(m);
                        ptxv_sim::rep::check(
                            &make,
                            state_cap.get(),
                            &mut |c, v| {
                                println!(
                                    "  {label}representative CTA {c}: {:?} states={} {:.1}s",
                                    v.classes(),
                                    v.states,
                                    v.seconds
                                )
                            },
                            &mut || {
                                if slices {
                                    ptxv_sim::wellsync::check_sliced(
                                        &make,
                                        state_cap.get(),
                                        &mut |_, _| {},
                                    )
                                } else {
                                    ptxv_sim::wellsync::check_certified(make(), state_cap.get())
                                }
                            },
                        )
                    }
                    _ if certify => ptxv_sim::wellsync::check_certified(m, state_cap.get()),
                    _ if slices => {
                        drop(m);
                        ptxv_sim::explore::explore_symmetric(&make, state_cap.get(), &mut |c, v| {
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
                    _ => ptxv_sim::l1_cores::explore_placements(m, &mut |m| {
                        explore(m, state_cap.get())
                    }),
                }
            };
            let make1 = || machine(case, k.clone()).expect("machine");
            // The plain certificate check is what errdist / equiv run as their precondition
            // (equiv.rs `prepare`): store it for them (vcache.rs).
            // (errdist builds the machine from the case without the variant's `after` edits.)
            let plain = certify
                && case.variant(&v)?.after.is_empty()
                && !slices
                && groups.is_none()
                && std::env::var("PTXV_EXPLORER").as_deref() != Ok("dpor");
            let vkey = plain
                .then(|| vcache::key(&case.dir, &v, f, "certify"))
                .flatten();
            // Input coverage (A) runs first, within its own caps: it reports even when the
            // exploration does not finish, and its memory is freed before the exploration.
            let (cov_on, n_mut, k_mut) = INPUT_COV.get().copied().unwrap_or((true, 0, 2));
            let specs = MUTATE_SPECS.get().cloned().unwrap_or_default();
            // On the plain certificate path, A rides on the certificate's own run instead
            // (coverage.rs `Tap`): that run is the same deterministic run, and comes before
            // any exploration. PTXV_COVERAGE_SEPARATE=1 runs it apart.
            let want_cov = cov_on || n_mut > 0 || !specs.is_empty() || concolic::on();
            let ride = want_cov
                && !concolic::on()
                && certify
                && !slices
                && groups.is_none()
                && std::env::var("PTXV_EXPLORER").as_deref() != Ok("dpor")
                && !std::env::var("PTXV_REP").is_ok_and(|v| v == "1")
                && std::env::var_os("PTXV_COVERAGE_SEPARATE").is_none()
                && m1.launch.env.residency.is_none()
                && !m1.launch.env.tmem_cotenant
                && m1.nic.is_none();
            let mut early_sym = None;
            let mut early_cov = (want_cov && !ride).then(|| {
                let (c, s) = concolic::analyze(&m1, coverage_budget(false));
                early_sym = s;
                release_freed_memory();
                c
            });
            if ride {
                ptxv_sim::coverage::tap_request(coverage_budget(false));
            }
            if nonfinite.is_some() {
                ptxv_sim::nonfinite::enable();
            }
            let mut verdict = check(&make1, "");
            if ride {
                early_cov = ptxv_sim::coverage::tap_take().or_else(|| {
                    // No certificate run carried it (a launch the certificate does not run
                    // the plain way): the run of its own.
                    Some(ptxv_sim::coverage::analyze_budget(
                        &m1,
                        coverage_budget(false),
                    ))
                });
            }
            // ASSUME(E-TC-FENCE-NOP): races the tcgen05 thread-sync fences would order are
            // contract deviations (twolevel::tc_fence_contract).
            if f == "ptx" {
                ptxv_sim::twolevel::tc_fence_contract(&mut verdict, &m1, &mut |m| {
                    check(&|| m.clone(), "tcgen05 fences implied: ")
                });
            }
            let births = if nonfinite.is_some() {
                ptxv_sim::nonfinite::take()
            } else {
                vec![]
            };
            if let Some(key) = &vkey {
                vcache::store(key, max_states, &verdict);
            }
            // ASSUME(S-CGA-SHAPE): only the sliced certificate check covers both cluster shapes
            // of a launch with a preferred cluster dimension (and their mixes).
            if m1.launch.preferred_shape().is_some() && !(slices && certify && groups.is_none()) {
                verdict.findings.push(ptxv_sim::machine::Finding {
                    kind: "unmodeled".into(),
                    sig: "cga-shape".into(),
                    msg: "the launch has a preferred cluster dimension: only `--slices --certify` checks both cluster shapes and their mixes (S-CGA-SHAPE); this check used the regular shape only".into(),
                    trace: vec![],
                    ..Default::default()
                });
            }
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
            let nf =
                nonfinite.map(|strict| nonfinite::analyze(case, &m1, &mut verdict, births, strict));
            print_verdict(
                &v,
                f,
                &verdict,
                &expect,
                &refuted,
                gbc.as_ref(),
                k.static_image.as_deref(),
            );
            // The two-level verdict: anything undefined under the strict model -> the SASS
            // front under the hardware profile. Both levels are reported.
            let mut two_level = None;
            if let Some(p) = profile.filter(|_| ptxv_sim::hwprofile::wanted(&verdict)) {
                if level2.is_none() {
                    let t0 = Instant::now();
                    let k2 = if f == "sass" {
                        k.clone()
                    } else {
                        load(case, &v, "sass", &dir)?.0
                    };
                    let ms = machine(case, k2)?;
                    let make2 = || ptxv_sim::hwprofile::machine(&ms, p);
                    group_log.borrow_mut().clear();
                    let mut v2 = check(&make2, "GB300 profile: ");
                    let used = ptxv_sim::hwprofile::discharge(&mut v2, &make2(), p)
                        .into_iter()
                        .map(|(id, sigs)| HwUsed {
                            probe: p
                                .iter()
                                .find(|i| i.id == id)
                                .map(|i| i.probe.to_string())
                                .unwrap_or_default(),
                            id,
                            discharged: sigs,
                        })
                        .collect();
                    level2 = Some((v2, used, t0.elapsed().as_secs_f64()));
                }
                let (v2, used, secs) = level2.as_ref().unwrap();
                let t = TwoLevel {
                    ptx_contract: verdict.classes().join(","),
                    contract_front: f.into(),
                    sass_gb300: ptxv_sim::hwprofile::result(v2),
                    hw_profile: profile_name.clone(),
                    hw_assumptions: used.clone(),
                    sass_classes: v2.classes(),
                    sass_complete: v2.complete,
                    sass_certified: v2.assumptions.iter().any(|a| a == "E-WELLSYNC"),
                    sass_states: v2.states,
                    sass_findings: v2.findings.clone(),
                    sass_notes: v2.notes.clone(),
                    seconds: *secs,
                };
                println!(
                    "    ptx_contract: {} ({} front, strict model)",
                    t.ptx_contract, t.contract_front
                );
                println!(
                    "    sass_gb300: {} (SASS front, hardware profile {}; {} states{}) using {}",
                    t.sass_gb300,
                    t.hw_profile,
                    t.sass_states,
                    if t.sass_complete || t.sass_certified {
                        ""
                    } else {
                        ", incomplete"
                    },
                    if t.hw_assumptions.is_empty() {
                        "no hardware assumption".to_string()
                    } else {
                        t.hw_assumptions
                            .iter()
                            .map(|u| {
                                if u.discharged.is_empty() {
                                    format!("{} [{}]", u.id, u.probe)
                                } else {
                                    format!(
                                        "{} [{}; discharges {}]",
                                        u.id,
                                        u.probe,
                                        u.discharged.join(" ")
                                    )
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(", ")
                    }
                );
                for x in t.sass_findings.iter().take(6) {
                    println!(
                        "      sass_gb300 {}: {}",
                        x.kind.to_uppercase(),
                        x.msg.lines().next().unwrap_or("")
                    );
                }
                two_level = Some(t);
            }
            if let Some(r) = &nf {
                println!(
                    "    nonfinite: {} terminal output(s), {} with NaN / Inf; inputs with NaN / Inf: {}; {} producing instruction(s)",
                    r.terminal_outputs,
                    r.terminal_outputs_nonfinite,
                    r.inputs.len(),
                    r.births.len()
                );
                for b in r.births.iter().take(8) {
                    println!(
                        "      {}: {} ({} -> {}), {} time(s)",
                        b.at,
                        b.birth.kind,
                        b.birth.operands.join(", "),
                        b.birth.result,
                        b.birth.count
                    );
                }
            }
            // Input coverage (A) and checks on mutated inputs (B), apart from the verdict.
            let mut input_coverage = None;
            let mut mutated_inputs = Vec::new();
            let mut mutation_refusals = Vec::new();
            let mut mutation_budget = None;
            if let Some(mut cov) = early_cov {
                let sym = early_sym.take();
                println!(
                    "    input coverage: {} input-dependent comparison(s), {} side(s) not taken on this input ({:.1}s)",
                    cov.sites,
                    cov.untaken.len(),
                    cov.seconds
                );
                if let Some(why) = &cov.stopped {
                    println!(
                        "    input coverage: the run stopped at its {why} cap (--coverage-seconds / PTXV_COVERAGE_MAX_RSS): partial, no side claimed untaken, not input-independent"
                    );
                }
                for u in cov.untaken.iter().take(20) {
                    println!("    note: {}", u.note);
                }
                // ASSUME(E-INPUT-INDEP): the provenance run saw every flow of input bytes into
                // control, addresses and synchronization.
                if cov.input_independent {
                    let raced = verdict.findings.iter().any(|x| x.kind == "race");
                    println!(
                        "    input-independent: in one run no branch, guard, address or synchronization operand derives from input data through the modeled provenance (registers; shared, global and local memory; TMEM; asynchronous copies), so control path, accesses and synchronization are the same for every input of this shape and these parameters as far as that provenance is complete. It covers control, addresses and synchronization only: value faults (an integer division by zero, a value that leaves a modeled range) and value-dependent parts (same-value writes, output determinism, the comparison with the GPU output) hold for this input only{}",
                        if raced {
                            "; the verdict has a race, under which another schedule can move data differently"
                        } else {
                            ""
                        }
                    );
                } else if let Some(d) = cov.input_dependent.first() {
                    println!(
                        "    input-dependent control: {} instruction(s) with an input-derived guard, address, control or synchronization operand (first: {} {} from {})",
                        cov.input_dependent.len(),
                        d.what,
                        d.at,
                        d.inputs
                            .iter()
                            .map(|x| x.buffer.clone())
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
                if cov.unexecuted_instructions > 0 {
                    println!(
                        "    unexecuted: {} instruction(s) no thread executed: {} behind input-dependent conditions, {} behind conditions free of input data (parameters, special registers), {} other",
                        cov.unexecuted_instructions,
                        cov.behind_input,
                        cov.behind_parameter,
                        cov.unexecuted_instructions - cov.behind_input - cov.behind_parameter
                    );
                }
                // Declared input invariants, checked on the captured bytes.
                let (dom, inv_notes) = input_domain(case, &m1)?;
                for n in &inv_notes {
                    println!("    {n}");
                }
                let cc_muts = concolic::search(&m1, &mut cov, sym, &dom, k_mut);
                if n_mut > 0 || !specs.is_empty() || !cc_muts.is_empty() {
                    let (mut muts, refused) = if n_mut > 0 {
                        ptxv_sim::coverage::plan(&m1, &cov, n_mut, k_mut, &dom)
                    } else {
                        (vec![], vec![])
                    };
                    let n_b = muts.len();
                    muts.extend(cc_muts);
                    let elem = |name: &str| {
                        case.m
                            .buffers
                            .iter()
                            .find(|b| b.name == name)
                            .map(|b| b.elem_bytes())
                    };
                    for sp in &specs {
                        muts.push(
                            ptxv_sim::coverage::parse_mutation(&m1, sp, &elem, &dom)
                                .map_err(|e| anyhow::anyhow!("--mutate-input {sp}: {e}"))?,
                        );
                    }
                    for r in &refused {
                        println!(
                            "    mutate-inputs: not mutated: {}: {}",
                            m1.k.describe(r.pc as usize),
                            r.reason
                        );
                    }
                    // The total budget of the mutated checks (wall and states).
                    let bud = COV_BUDGETS.get().copied().unwrap_or_default();
                    let b_t0 = Instant::now();
                    let b_end = b_t0 + Duration::from_secs_f64(bud.mutate_seconds.max(0.0));
                    let mut states_left = bud.mutate_states.unwrap_or(max_states.saturating_mul(2));
                    let n_muts = muts.len();
                    ptxv_sim::membudget::set_deadline(Some(b_end));
                    for (checked, mu) in muts.into_iter().enumerate() {
                        let now = Instant::now();
                        if now >= b_end || states_left == 0 {
                            let why = if now >= b_end {
                                format!(
                                    "the total wall budget ({:.0} s, --mutate-budget-seconds) is spent",
                                    bud.mutate_seconds
                                )
                            } else {
                                "the total state budget (--mutate-budget-states) is spent"
                                    .to_string()
                            };
                            let note = format!(
                                "budget: {} of {} mutated input(s) not checked: {why}",
                                n_muts - checked,
                                n_muts
                            );
                            println!("    mutate-inputs: {note}");
                            mutation_budget = Some(note);
                            break;
                        }
                        let t0 = Instant::now();
                        let source = if mu.pc == u32::MAX {
                            "given"
                        } else if checked >= n_b {
                            "concolic"
                        } else {
                            "B"
                        };
                        let mm = ptxv_sim::coverage::apply(&m1, &mu);
                        // A mutation that opens no code on its deterministic pass (the side it
                        // was to make taken stays untaken, the same instructions are reached)
                        // is not checked again: B checks paths, and that pass has none new.
                        let left = (b_end - Instant::now()).as_secs_f64();
                        let pass = if std::env::var_os("PTXV_MUTATE_NO_SKIP").is_some() {
                            ptxv_sim::coverage::Pass::Undecided
                        } else {
                            ptxv_sim::coverage::mutated_pass(
                                &mm,
                                &cov,
                                &mu,
                                ptxv_sim::coverage::Budget {
                                    seconds: Some(left),
                                    rss: None,
                                },
                            )
                        };
                        if pass == ptxv_sim::coverage::Pass::Unchanged {
                            let at = m1.k.describe(mu.pc as usize);
                            let row = MutatedRow {
                                source,
                                mutation: mu.label.clone(),
                                pc: Some(mu.pc),
                                within_invariants: mu.within_invariants,
                                at: at.clone(),
                                makes: mu.never,
                                classes: vec![],
                                complete: false,
                                certified: false,
                                states: 0,
                                findings: vec![],
                                notes: vec![],
                                skipped: Some(format!(
                                    "the {} side of {at} is not taken on the mutated input's deterministic run, which reaches the same instructions as the captured input's: no new code, not checked again",
                                    if mu.never { "true" } else { "false" }
                                )),
                                seconds: t0.elapsed().as_secs_f64(),
                            };
                            println!(
                                "    on-mutated-input: {}: not checked: {}",
                                row.mutation,
                                row.skipped.as_deref().unwrap_or("")
                            );
                            mutated_inputs.push(row);
                            continue;
                        }
                        let make_m = || mm.clone();
                        state_cap.set(states_left.min(max_states));
                        let vm = check(&make_m, "mutated input: ");
                        state_cap.set(max_states);
                        states_left = states_left.saturating_sub(vm.states);
                        let at = if mu.pc == u32::MAX {
                            "given with --mutate-input".to_string()
                        } else {
                            m1.k.describe(mu.pc as usize)
                        };
                        let row = MutatedRow {
                            source,
                            mutation: mu.label.clone(),
                            pc: (mu.pc != u32::MAX).then_some(mu.pc),
                            within_invariants: mu.within_invariants,
                            at: at.clone(),
                            makes: mu.never,
                            classes: vm.classes(),
                            complete: vm.complete,
                            certified: vm.assumptions.iter().any(|a| a == "E-WELLSYNC"),
                            states: vm.states,
                            findings: vm.findings.clone(),
                            notes: vm.notes.clone(),
                            skipped: None,
                            seconds: t0.elapsed().as_secs_f64(),
                        };
                        println!(
                            "    on-mutated-input{}: {} ({}): {:?} states={}{} {:.1}s",
                            if row.within_invariants {
                                " (within declared invariants)"
                            } else {
                                ""
                            },
                            row.mutation,
                            if mu.pc == u32::MAX {
                                at.clone()
                            } else {
                                format!(
                                    "makes {} side of {} taken",
                                    if mu.never { "the true" } else { "the false" },
                                    at
                                )
                            },
                            row.classes,
                            row.states,
                            if row.complete || row.certified {
                                ""
                            } else {
                                " INCOMPLETE"
                            },
                            row.seconds
                        );
                        for x in row.findings.iter().filter(|x| x.kind != "note").take(6) {
                            println!(
                                "      on-mutated-input{} {}: {}",
                                if row.within_invariants {
                                    " (within declared invariants)"
                                } else {
                                    ""
                                },
                                x.kind.to_uppercase(),
                                x.msg.lines().next().unwrap_or("")
                            );
                        }
                        mutated_inputs.push(row);
                    }
                    ptxv_sim::membudget::set_deadline(None);
                    mutation_refusals = refused;
                }
                input_coverage = Some(cov);
            }
            // Debugging / figures: PTXV_TRACE_DUMP=<dir> writes each finding's replayed witness.
            if let Some(d) = std::env::var_os("PTXV_TRACE_DUMP") {
                let p = Path::new(&d).join(format!("{v}.{f}.trace.json"));
                trace_dump::write(make1(), &verdict, &v, f, &p)?;
            }
            // Captured cases (tools/capture.py) carry the GPU's output bytes.
            let mut expect_report = Vec::new();
            // N-MULTIMEM-ROUND16: multimem reductions whose rounding the model only calibrates.
            let ulp_ok = !case.m.multicast.is_empty();
            // C-TICKET-ADDR: outputs compared up to the certified permutation of rows.
            let perm_rows = verdict
                .findings
                .iter()
                .find(|f| f.kind == "clean-up-to-permutation")
                .map(|f| f.perm_rows.clone());
            for (i, o) in verdict.outputs.iter().enumerate() {
                let lines = match &perm_rows {
                    Some(r) => case.expect_report_perm(o, r)?,
                    None => case.expect_report_tol(o, ulp_ok)?,
                };
                for line in lines {
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
            // `executed_image: false`: an sm_100 / sm_100f cubin, not what GB300 runs.
            let mut cj = serde_json::to_value(&verdict)?;
            cj["executed_image"] = k.static_image.is_none().into();
            if let Some(n) = &k.static_image {
                cj["image_note"] = n.clone().into();
            }
            if let Some(r) = PTX_REBUILD
                .lock()
                .unwrap()
                .get(&rebuild_key(case, v.as_str()))
            {
                cj["ptx_rebuild"] = serde_json::to_value(r)?;
            }
            std::fs::write(
                dir.join(format!("{v}.{f}.check.json")),
                serde_json::to_string_pretty(&cj)?,
            )?;
            rows.push(CheckRow {
                variant: v.clone(),
                front: f.into(),
                expect_report,
                pass: verdict.meets(&expect),
                expect,
                got,
                refuted_opt_in: refuted,
                executed_image: k.static_image.is_none(),
                image_note: k.static_image.clone(),
                ptx_rebuild: PTX_REBUILD
                    .lock()
                    .unwrap()
                    .get(&rebuild_key(case, v.as_str()))
                    .cloned(),
                verdict,
                gb300: gbc,
                two_level,
                nonfinite: nf,
                input_coverage,
                mutated_inputs,
                mutation_refusals,
                mutation_budget,
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
        let k = match load(case, variant, f, &dir) {
            Ok((k, _)) => k,
            // A PTX the front-end does not parse (forms ptxas compiles but the PTX lift does
            // not model): the SASS front alone gives the run (`hw`'s expected bytes).
            Err(e) if f == "ptx" => {
                println!(
                    "num {variant} ptx : not run, the PTX front-end refuses it ({}); SASS front only",
                    format!("{e:#}").lines().last().unwrap_or("").trim()
                );
                continue;
            }
            Err(e) => return Err(e),
        };
        let mut m = machine(case, k)?;
        m.num = Some(Box::new(ptxv_sim::num::NumState::with_mma_eps(
            m.launch.mma_model.eps(),
        )));
        let (m, findings, ok) = run_once(m);
        if !ok {
            bail!("numerics run of {variant} ({f}) did not terminate cleanly: {findings:?}");
        }
        note_ref_faults(&format!("numerics run of {variant} ({f})"), &findings);
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
                        format!("matches {} bitwise", REF_KIND.lock().unwrap())
                    } else {
                        match *REF_KIND.lock().unwrap() {
                            "simulator" => {
                                format!("{mism}/{runs} runs differ from the correct output")
                            }
                            k => format!("{mism}/{runs} runs differ from {k} bitwise"),
                        }
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

/// Faults (and unmodeled steps) of the simulator runs that produced `ptxv hw`'s expected
/// bytes: printed by `cmd_hw`, so that a GPU / simulator mismatch is not taken for a GPU
/// problem when the reference itself went wrong.
static REF_FAULTS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// What `ptxv hw`'s expected bytes are (printed in each row's outcome).
static REF_KIND: std::sync::Mutex<&str> = std::sync::Mutex::new("simulator");

/// `ptxv hw --no-num`'s expected bytes: the output buffers' captured GPU bytes (`expect`),
/// concatenated in `outputs` order. Refuses a buffer without `expect` or whose `expect`
/// covers only part of it (the rest of the buffer's final bytes are unknown without a run).
fn captured_outputs(case: &Case) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for o in &case.m.outputs {
        let b = case.buffer(o)?;
        let Some(p) = &b.expect else {
            bail!("--no-num: output {o} has no captured GPU bytes (`expect`)")
        };
        let want = std::fs::read(p).with_context(|| p.clone())?;
        if b.expect_offset != 0 || want.len() != b.bytes() {
            bail!(
                "--no-num: output {o}'s captured bytes cover [{}, {}) of its {} bytes; the whole buffer is needed",
                b.expect_offset,
                b.expect_offset as usize + want.len(),
                b.bytes()
            );
        }
        out.extend(want);
    }
    Ok(out)
}

fn note_ref_faults(run: &str, findings: &[ptxv_sim::machine::Finding]) {
    let mut g = REF_FAULTS.lock().unwrap();
    for f in findings
        .iter()
        .filter(|f| f.kind == "fault" || f.kind == "unmodeled")
    {
        let line = format!("{run}: {} {}", f.kind, f.msg.lines().next().unwrap_or(""));
        if !g.contains(&line) {
            g.push(line);
        }
    }
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
    let ref_faults = REF_FAULTS.lock().unwrap().clone();
    if !ref_faults.is_empty() {
        println!(
            "hw: the simulator run that produced the expected output reported {} fault(s) / unmodeled step(s); a mismatch below may be the simulator's, not the GPU's:",
            ref_faults.len()
        );
        for l in ref_faults.iter().take(5) {
            println!("hw:   {l}");
        }
    }
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
            },
        );
        if !ref_faults.is_empty() && !row.mismatch_runs.is_empty() {
            println!(
                "hw {:<6} mismatch against a faulted simulator reference ({} fault(s), above): not evidence of a GPU problem",
                row.variant,
                ref_faults.len()
            );
        }
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
        None,
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
            rep,
            groups,
            json,
            residency,
            tmem_cotenant,
            assume_refuted,
            no_gb300,
            hw_profile,
            nonfinite,
            nonfinite_strict,
            no_input_coverage,
            coverage_only,
            concolic,
            mutate_inputs,
            mutate_elements,
            mutate_input,
            coverage_seconds,
            mutate_budget_seconds,
            mutate_budget_states,
        } => {
            let _ = concolic::ON.set(concolic);
            let _ = MUTATE_SPECS.set(mutate_input);
            let _ = COVERAGE_ONLY.set(coverage_only);
            let _ = COV_BUDGETS.set(CovBudgets {
                coverage_seconds,
                mutate_seconds: mutate_budget_seconds,
                mutate_states: mutate_budget_states,
            });
            let j = json.then(|| jsonout::begin("check"));
            let _ = INPUT_COV.set((
                !no_input_coverage && std::env::var("PTXV_NO_INPUT_COVERAGE").is_err(),
                mutate_inputs,
                mutate_elements.max(1),
            ));
            if let Some(p) = hw_profile {
                let _ = HW_PROFILE.set(p);
            }
            if rep {
                // The check paths read it (`ptxv_sim::rep`).
                // SAFETY: argument handling on the main thread; no other thread runs yet.
                unsafe { std::env::set_var("PTXV_REP", "1") };
            }
            if !assume_refuted.is_empty() {
                // build_launch reads it (ptxv_sim::assume_refuted), on every check path.
                let mut ids = assume_refuted.join(",");
                if let Ok(e) = std::env::var("PTXV_ASSUME_REFUTED") {
                    ids = format!("{e},{ids}");
                }
                // SAFETY: argument handling on the main thread; no other thread runs yet.
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
                (nonfinite || nonfinite_strict).then_some(nonfinite_strict),
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
            no_num,
            out,
        } => {
            let c = Case::load(&case)?;
            if no_num {
                let expected = captured_outputs(&c)?;
                *REF_KIND.lock().unwrap() = "the captured GPU output";
                cmd_hw(&c, &variant, runs, timeout, &out, &expected, scrub, &delay)?;
                return Ok(());
            }
            // PTXV_HW_EXPECT=variant: each variant against its own deterministic simulator
            // run (a numerics mutant's wrong output, rung 3b).
            if std::env::var("PTXV_HW_EXPECT").as_deref() == Ok("variant") {
                let dir = out_dir(&c, &out)?;
                for v in variants(&c, &variant) {
                    let (k, _) = load(&c, &v, "sass", &dir)?;
                    let (m, fs, ok) = ptxv_sim::explore::run_once(machine(&c, k)?);
                    if !ok {
                        bail!("{v} does not terminate in the simulator");
                    }
                    REF_FAULTS.lock().unwrap().clear();
                    note_ref_faults(&format!("simulator run of {v}"), &fs);
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
                let (m, fs, ok) = ptxv_sim::explore::run_once(machine(&c, k)?);
                if !ok {
                    bail!("the good variant does not terminate in the simulator");
                }
                note_ref_faults("simulator run of good", &fs);
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
        Cmd::CalibMma {
            case,
            out,
            only,
            step,
        } => {
            let c = Case::load(&case)?;
            // A byte-operand probe (probes/mma_f8) calibrates the narrow kinds.
            if c.buffer("A")?.dtype == "u8" {
                calib_lp::run(&c, &out, only.as_deref())?
            } else {
                calib::run(&c, &out, step)?
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

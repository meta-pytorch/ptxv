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
mod concolic;
mod cvt_sweep;
mod equiv;
mod errdist;
mod errdist_html;
use ptxv_jsonout as jsonout;
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
    version = jsonout::VERSION,
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
        /// Directory for the materialized variants (PTX, cubin, patched cubin, SASS), kept.
        /// Default: a directory of this run under $PTXV_ROOT/out/<case> (./out/<case> without
        /// PTXV_ROOT), removed when the run ends; a killed run's by the next run of the case on
        /// the same host, boot and pid namespace (a batch job in a pid namespace of its own
        /// never removes an earlier job's); concurrent runs never share one.
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
        /// Skip both GB300 checks (also PTXV_GB300=0): the second level, which re-checks race /
        /// stale findings under the GB300 hardware model (E-HW-L1) to classify each as
        /// observable or latent, and the GB300-profile re-run of the two-level verdict
        /// (`sass_gb300`, --hw-profile). The row then says the re-run was skipped
        /// (`two_level_skipped`) where the strict verdict would have needed it.
        #[arg(long)]
        no_gb300: bool,
        /// Hardware profile of the two-level verdict: when the strict check reports anything
        /// undefined (race, violation, contract deviation, undefined values, ...), the SASS
        /// front is checked again with every probe-backed assumption of the profile on, and
        /// both levels are reported (`ptx_contract`, `sass_gb300`). `gb300` (default) or
        /// `none` (also PTXV_HW_PROFILE); any other name is an error. `none` and --no-gb300 /
        /// PTXV_GB300=0 turn the re-run off, and the row says so (`two_level_skipped`).
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
        /// PTXV_CONCOLIC_CHECKS) are checked as mutated inputs. Off by default. Refused in
        /// multi-node mode, as --mutate-inputs.
        #[arg(long)]
        concolic: bool,
        /// Exhaustive concolic expansion (ptxv_sim::cert): every feasible path over all values
        /// of the input buffers is traversed on a replayed input or proven infeasible (two
        /// SMT solvers, PTXV_EXH_SOLVERS), every input-dependent control, address and
        /// synchronization operand is proven fixed per path, and each path's input gets this
        /// check: rows[].exhaustive_concolic {status certified | blocked | refuted, ...}. Limits:
        /// PTXV_EXH_PATHS (8), PTXV_EXH_BUDGET seconds (3600). Off by default. Refused in
        /// multi-node mode, as --mutate-inputs.
        #[arg(long)]
        exhaustive_concolic: bool,
        /// Check again on mutated inputs: for up to N untaken input-dependent comparison
        /// sides that compare one input word against a constant, write a value that flips
        /// the comparison into --mutate-elements elements and run the check on that input.
        /// Results are reported apart (`on-mutated-input`), never in the verdict. Refused in
        /// multi-node mode (PTXV_DIST_DIR with more than one rank), and so is --mutate-input:
        /// which mutated inputs are checked depends on each rank's clock (the coverage run's
        /// cap, --mutate-budget-seconds, the capped pass that skips a mutation), and ranks that
        /// disagree wait for each other's explorations forever.
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
        /// Wall-clock budget of the whole check in seconds (also PTXV_TIME_BUDGET; default
        /// none). Certificates, seed walks, every exploration (model reruns, LDS/STS passes,
        /// leader policies, both levels) and the GB300-profile re-run stop once it is spent:
        /// each cut phase leaves its verdict incomplete with a note naming it, and a cut
        /// certificate never holds. The re-run states the budget it starts with, and does not
        /// start once the budget is spent (`two_level_skipped`). Input coverage gets at most
        /// what is left; the --mutate-inputs budget is nested inside (its checks stop at the
        /// earlier end). --max-states stays per exploration. Not bounded by it: the work
        /// outside these phases (loading, lifting and setting up each row; on large kernels up
        /// to about half a minute past the budget), --concolic / --exhaustive-concolic (their
        /// own budgets), a fill rerun already running when the budget is spent (the next one
        /// does not start), and --groups prefix runs. Refused in multi-node mode
        /// (PTXV_DIST_DIR with more than one rank): every rank would cut its own phases, and
        /// ranks in different phases wait for each other forever.
        #[arg(long)]
        time_budget: Option<f64>,
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
        /// Directory for the materialized variants, witnesses and the JSON summary
        /// (`equiv.json`), kept. Default: a directory of this run under
        /// $TMPDIR/ptxv-equiv-<file's directory>-<file stem> (printed); the next run of the file
        /// on the same host, boot and pid namespace removes those of finished runs.
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
        /// Also run the chains' Tier 1: the residue-exact chain of MMA outputs (minutes on
        /// wide scale ranges, hours at K = 4096; memory flat in K). Default: tier 0, the
        /// elementwise law where it applies, tier 2 where its applicability check passes,
        /// else tier 3.
        #[arg(long)]
        tier1: bool,
        /// Importance sampling: also run Tier 3 with inputs from this proposal (JSON,
        /// docs/errdist.md §5.4.1), reported apart from the plain estimate and only when its
        /// gates pass.
        #[arg(long, value_name = "FILE")]
        is: Option<PathBuf>,
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
        /// Directory for the report (report.json, report.md) and the files behind it, kept.
        /// Default: a directory of this run under $TMPDIR/ptxv-compare-<ref>-<cand> (printed;
        /// `report_dir` with --json); the next run of the pair on the same host, boot and pid
        /// namespace removes those of finished runs.
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
        /// Watchdog per variant, seconds (default: 20 plus 1 per MB of the case's files).
        #[arg(long)]
        timeout: Option<f64>,
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
    /// First access of each buffer in one complete deterministic run of the launches (JSON on
    /// stdout: name -> {"first": "write" | "read", "read_back": bool, "by": actor};
    /// never-accessed buffers are absent). A discard (`discard.global.L2`, CCTL.E.RML2) counts
    /// as a write; read_back: read again after a first write. The capture side uses it to
    /// suggest scratch buffers (`init = "undef"`, `scratch = true`).
    #[command(hide = true)]
    FirstAccess {
        case: PathBuf,
        #[arg(long, default_value = "good")]
        variant: String,
        #[arg(long, default_value = "sass")]
        front: String,
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
        /// The variant whose launch-order edits (`after`, `waits`, `launch_order`) apply.
        #[arg(long)]
        variant: Option<String>,
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
        /// The variant whose launch-order edits (`after`, `waits`, `launch_order`) apply.
        #[arg(long)]
        variant: Option<String>,
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

/// Per-invocation directories made by `run_dir`: (case default directory, run directory).
static RUN_DIRS: std::sync::Mutex<Vec<(PathBuf, PathBuf)>> = std::sync::Mutex::new(Vec::new());
/// Set by commands whose files in the run directory are their result (`keep_run_dirs`).
static KEEP_RUN_DIRS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// This invocation's directory under a case's default output directory `base` (no `--out`):
/// `<base>/run-<host>-<pid>-<start>-<ns>` (`<host>`: `run_host`; `<start>`: the process's start
/// time in clock ticks since boot; `run-<pid>-<ns>` without /proc), the same for every call in
/// one process.
/// Concurrent runs of one case never share materialized variants: a variant's cubin is copied
/// (or built by ptxas), patched in place and read back, so in a shared directory another run's
/// copy in between was read as this run's variant (a scoreboard-patched variant checked
/// unpatched: clean), and two cases of one name that claimed `base` at once read each other's
/// cubins. Removed when the command ends (`remove_run_dirs`) unless it keeps its files; the
/// directories of killed runs are removed by the next run of the case with the same host token
/// (`reap_run_dirs`); `--out` directories are never removed.
fn run_dir(base: &Path) -> Result<PathBuf> {
    let mut dirs = RUN_DIRS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, d)) = dirs.iter().find(|(b, _)| b == base) {
        return Ok(d.clone());
    }
    let d = new_run_dir(base)?;
    if KEEP_RUN_DIRS.load(std::sync::atomic::Ordering::Relaxed) {
        let _ = std::fs::write(d.join(".kept"), "");
    }
    dirs.push((base.to_path_buf(), d.clone()));
    Ok(d)
}

/// A new directory of this invocation under `base`, `run-<host>-<pid>-<start>-<ns>`, made
/// after removing those of runs gone on this host (`reap_run_dirs`), and locked by this process
/// until it ends (`lock_run_dir`).
fn new_run_dir(base: &Path) -> Result<PathBuf> {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let d = match (run_host(), proc_self()) {
        (Some(host), Some((pid, start))) => base.join(format!("run-{host}-{pid}-{start}-{ns}")),
        _ => base.join(format!("run-{}-{ns}", std::process::id())),
    };
    reap_run_dirs(base);
    std::fs::create_dir_all(&d)?;
    lock_run_dir(&d);
    Ok(d)
}

/// The directory of a command whose files are its result (`compare`'s report, `equiv`'s
/// summary) when `--out` is not given: this invocation's own under `base` (the pair's
/// directory in the temporary directory), kept when the command ends. Concurrent runs of one
/// pair shared `base` and overwrote each other's files (a report read back after the other
/// run wrote its own, variants materialized over each other's). The next run of the pair on
/// the host removes the directories of finished runs, as `base` kept only the latest run's
/// files before.
pub(crate) fn result_dir(base: &Path) -> Result<PathBuf> {
    new_run_dir(base)
}

/// This host in run directory names (the host token), so that a run judges only directories
/// whose owners' locks it sees (`run_dir_dead`): the host name; this boot, the first 16 hex
/// digits of `/proc/sys/kernel/random/boot_id` (machines of one name that share the file
/// system, and this machine before a reboot, are other hosts); and the pid namespace when it is
/// not the initial one (a container that shares the host name). So reaping is per host, boot and
/// pid namespace: a batch system that gives each job a pid namespace of its own never removes
/// an earlier job's directories. None without /proc: nothing is reaped.
fn run_host() -> Option<&'static str> {
    static HOST: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    HOST.get_or_init(|| {
        let name: String = std::fs::read_to_string("/proc/sys/kernel/hostname")
            .ok()?
            .trim()
            .chars()
            .map(|c| match c {
                'a'..='z' | 'A'..='Z' | '0'..='9' | '.' | '-' | '_' => c,
                _ => '_',
            })
            .collect();
        let boot: String = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
            .ok()?
            .chars()
            .filter(char::is_ascii_hexdigit)
            .take(16)
            .collect();
        let ns = std::fs::read_link("/proc/self/ns/pid").ok()?;
        let ns = ns.to_string_lossy();
        let ino = ns.strip_prefix("pid:[")?.strip_suffix(']')?;
        if name.is_empty() || boot.len() < 16 {
            return None;
        }
        let host = format!("{name}.boot{boot}");
        Some(match ino {
            "4026531836" => host,
            _ => format!("{host}.pidns{ino}"),
        })
    })
    .as_deref()
}

/// Start time of process `pid` in clock ticks since boot (`/proc/<pid>/stat` field 22, as this
/// process's time namespace shows it): with the pid it names one process in run directory
/// names. Err(NotFound): there is no such process.
fn proc_start(pid: u32) -> std::io::Result<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"))?;
    // `pid (comm) state ...`, and comm may hold spaces and parentheses.
    stat.rsplit_once(')')
        .and_then(|(_, rest)| rest.split_whitespace().nth(19)?.parse().ok())
        .ok_or_else(|| std::io::ErrorKind::InvalidData.into())
}

/// This process as /proc names it (pid, start time), for its run directory names: the pid from
/// `/proc/self`, even where it differs from `std::process::id()`.
fn proc_self() -> Option<(u32, u64)> {
    let pid = std::fs::read_link("/proc/self")
        .ok()?
        .to_str()?
        .parse()
        .ok()?;
    Some((pid, proc_start(pid).ok()?))
}

/// (host, pid, start) of a run directory name `run-<host>-<pid>-<start>-<ns>`; None for other
/// names, `run-<pid>-<ns>` (older binaries, or no /proc) included.
fn run_dir_owner(name: &str) -> Option<(&str, u32, u64)> {
    let mut it = name.strip_prefix("run-")?.rsplitn(4, '-');
    let ns = it.next()?;
    let (start, pid, host) = (it.next()?, it.next()?, it.next()?);
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    (digits(ns) && digits(start) && digits(pid) && !host.is_empty())
        .then(|| Some((host, pid.parse().ok()?, start.parse().ok()?)))
        .flatten()
}

/// Whether `name` is a run directory's (`run_dir`), of this binary or an older one
/// (`run-<pid>-<ns>`): ptxv output, never case input (`vcache::key`).
pub(crate) fn is_run_dir_name(name: &str) -> bool {
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    run_dir_owner(name).is_some()
        || name
            .strip_prefix("run-")
            .and_then(|r| r.split_once('-'))
            .is_some_and(|(pid, ns)| digits(pid) && digits(ns))
}

/// Remove the run directories under `base` that killed runs on this host left behind (a
/// signal or the OOM killer ends a run before `remove_run_dirs`): `run-<host>-<pid>-<start>-
/// <ns>` of this host token and user whose run is gone (`run_dir_dead`: no process holds its
/// lock), unless kept (`.kept`, `keep_run_dirs`). Directories of other host tokens (a shared
/// file system, another boot or pid namespace), of older binaries and of live runs stay. A
/// live run is judged by its lock, not by /proc: the start time /proc shows differs between
/// time namespaces, so a live run in another one looked like a reused pid and lost its
/// directory. Reapers that meet at a dead directory both remove it (`remove_run_dir`: what one
/// leaves keeps `.lock`, and the next run removes it). `base` is listed in full before anything
/// in it is removed (read while its entries go, an NFS listing can skip some: `remove_run_dir`).
fn reap_run_dirs(base: &Path) {
    use std::os::unix::fs::MetadataExt;
    let Some(host) = run_host() else {
        return;
    };
    let Ok(uid) = std::fs::metadata("/proc/self").map(|m| m.uid()) else {
        return;
    };
    let Ok(entries) = std::fs::read_dir(base) else {
        return;
    };
    let entries: Vec<std::fs::DirEntry> = entries.flatten().collect();
    for e in entries {
        let name = e.file_name();
        let Some((h, _, _)) = name.to_str().and_then(run_dir_owner) else {
            continue;
        };
        let d = e.path();
        let mine = std::fs::symlink_metadata(&d).is_ok_and(|m| m.is_dir() && m.uid() == uid);
        if h == host && mine && !d.join(".kept").exists() && run_dir_dead(&d) {
            remove_run_dir(&d);
        }
    }
}

/// Remove run directory `d` (a dead run's, or this invocation's own at its end): everything in
/// it but `.lock`, then `.lock`, then `d`, so that whatever is left keeps `.lock` and the next
/// run of the case removes it (`reap_run_dirs`). `remove_dir_all` removes entries while it
/// reads the directory, and on NFS that skips some: with two runs reaping at once, dead
/// directories stayed behind partly removed, and those that had lost `.lock` were never
/// reaped. Here each directory is read in full before anything in it is removed, and read
/// again until it is empty (`empty_dir`). Should `d` still not be empty once `.lock` is gone
/// (an entry that every listing missed while another process removed others), `.lock` is made
/// again.
fn remove_run_dir(d: &Path) {
    if !empty_dir(d, Some(".lock"), &mut false) {
        return;
    }
    let _ = std::fs::remove_file(d.join(".lock"));
    match std::fs::remove_dir(d) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            let _ = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .open(d.join(".lock"));
        }
        _ => {}
    }
}

/// Remove everything in directory `d` but its entry `keep`, without following symbolic links:
/// the whole listing first, then its entries (directories emptied the same way), and again
/// while anything is left, until a round removes nothing, at most 8 rounds (another process may
/// be removing the same entries: what it removed first counts as gone, a directory it removed
/// before this run could read it included, and what a listing missed is found by the next
/// round). Whether `d` holds nothing but `keep` (true once `d` is gone); `removed` is set when
/// anything under `d` went. A round that removes nothing ends its level: with rounds
/// retried at every level, what a run could not remove k levels down cost up to 8^k listings
/// per run.
fn empty_dir(d: &Path, keep: Option<&str>, removed: &mut bool) -> bool {
    for _ in 0..8 {
        let entries = match std::fs::read_dir(d) {
            Ok(entries) => entries,
            // Removed by another process since its parent was listed: its parent's removal of
            // it finds it gone too.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return true,
            Err(_) => return false,
        };
        let names: Vec<std::ffi::OsString> = entries
            .flatten()
            .map(|e| e.file_name())
            .filter(|n| keep.is_none_or(|k| n != k))
            .collect();
        if names.is_empty() {
            return true;
        }
        let mut round = false;
        for n in names {
            let p = d.join(n);
            let went = match std::fs::symlink_metadata(&p) {
                Ok(m) if m.is_dir() => {
                    empty_dir(&p, None, &mut round) && is_gone(std::fs::remove_dir(&p))
                }
                Ok(_) => is_gone(std::fs::remove_file(&p)),
                Err(e) => e.kind() == std::io::ErrorKind::NotFound,
            };
            round |= went;
        }
        *removed |= round;
        if !round {
            return false;
        }
    }
    false
}

/// Whether a removal leaves its entry gone: removed here, or by another process first.
fn is_gone(r: std::io::Result<()>) -> bool {
    match r {
        Ok(()) => true,
        Err(e) => e.kind() == std::io::ErrorKind::NotFound,
    }
}

/// The locks of this invocation's run directories (`lock_run_dir`), with their directories.
static RUN_LOCKS: std::sync::Mutex<Vec<(PathBuf, std::fs::File)>> =
    std::sync::Mutex::new(Vec::new());

/// Lock run directory `d` for as long as this process lives: an exclusive flock(2) on
/// `<d>/.lock` (`File::try_lock`), which the kernel releases when the process ends, however it
/// ends (SIGKILL and the OOM killer included), and which every process of the host sees,
/// whatever its pid, time or user namespace. `.lock` appears only once it is locked: the lock
/// is taken on `<d>/.lock.new`, which is then renamed (the lock stays with the file). Created
/// as `.lock` and locked after, it read as a dead run's lock in between, and a reaper removed
/// the new directory of a live run. When the lock cannot be taken (a file system without
/// locks), the file goes again: a directory without `.lock` is never reaped, so a reaper that
/// could take the lock later never removes a live run's directory. Once taken, the lock is held
/// whatever the rename reports: a rename reported as failed may have happened all the same (on
/// NFS, a reply lost after the server renamed), and releasing the lock then left an unlocked
/// `.lock`, read as a dead run's. When the rename did fail, `.lock.new` is unlinked, and the
/// lock on the removed file is harmless; on NFS the file, still open, stays as `.nfs*` in the
/// directory until the run ends (the directory has no `.lock` and counts as live either way).
fn lock_run_dir(d: &Path) {
    let new = d.join(".lock.new");
    let Ok(f) = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&new)
    else {
        return;
    };
    if f.try_lock().is_err() {
        drop(f);
        let _ = std::fs::remove_file(&new);
        return;
    }
    if std::fs::rename(&new, d.join(".lock")).is_err() {
        let _ = std::fs::remove_file(&new);
    }
    let mut locks = RUN_LOCKS.lock().unwrap_or_else(|e| e.into_inner());
    locks.push((d.to_path_buf(), f));
}

/// Release the lock of this invocation's run directory `d` (before removing it: on NFS an open
/// file in a directory is renamed to `.nfs*` and keeps the directory).
fn unlock_run_dir(d: &Path) {
    RUN_LOCKS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|(p, _)| p != d);
}

/// Whether the run of directory `d` is gone: no process holds the lock of `<d>/.lock` (this
/// process could take it; it lets it go at once). A directory without the file (its run is
/// making it or could not lock it, or an older binary's) or whose lock cannot be taken for any
/// other reason counts as live, and so do this process's own directories.
fn run_dir_dead(d: &Path) -> bool {
    let locks = RUN_LOCKS.lock().unwrap_or_else(|e| e.into_inner());
    if locks.iter().any(|(p, _)| p == d) {
        return false;
    }
    drop(locks);
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(d.join(".lock"))
        .is_ok_and(|f| f.try_lock().is_ok())
}

/// Keep this invocation's run directories: the command's result is a file in them (calib,
/// sb-sweep and probe print or dump into them). Marked `.kept`, so no later run reaps them.
pub(crate) fn keep_run_dirs() {
    let dirs = RUN_DIRS.lock().unwrap_or_else(|e| e.into_inner());
    KEEP_RUN_DIRS.store(true, std::sync::atomic::Ordering::Relaxed);
    for (_, d) in dirs.iter() {
        let _ = std::fs::write(d.join(".kept"), "");
    }
}

/// Remove this invocation's run directories (at the end of every command, also on an error,
/// a panic or a failing check's exit status), unless kept.
fn remove_run_dirs() {
    let mut dirs = RUN_DIRS.lock().unwrap_or_else(|e| e.into_inner());
    if KEEP_RUN_DIRS.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    for (_, d) in dirs.drain(..) {
        unlock_run_dir(&d);
        remove_run_dir(&d);
    }
}

/// `std::process::exit` after removing the run directories (exit skips destructors).
fn exit_cleanly(code: i32) -> ! {
    remove_run_dirs();
    std::process::exit(code)
}

/// Decided by the E-WELLSYNC certificate: a complete verdict that lists it. Incomplete verdicts
/// list it too, since the assumptions of their parts accumulate (a certified launch group or
/// %smid placement beside an incomplete one; smid_slot, smid_three), and are not certified.
pub(crate) fn certified(v: &Verdict) -> bool {
    v.complete && v.assumptions.iter().any(|a| a == "E-WELLSYNC")
}

/// Multi-node mode as `ptxv_sim::dist::raw` decides it (PTXV_DIST_DIR, and more than one rank
/// in PTXV_RANKS or SLURM_NTASKS; a missing count counts as more), read from the environment
/// without connecting to the other ranks.
fn multi_node() -> bool {
    std::env::var_os("PTXV_DIST_DIR").is_some()
        && std::env::var("PTXV_RANKS")
            .or_else(|_| std::env::var("SLURM_NTASKS"))
            .ok()
            .and_then(|n| n.parse::<usize>().ok())
            .is_none_or(|n| n > 1)
}

struct RunDirsGuard;

impl Drop for RunDirsGuard {
    fn drop(&mut self) {
        remove_run_dirs();
    }
}

/// The directory a command materializes a case's variants in (PTX, cubin, patched cubin,
/// SASS) and writes its other files to: `--out`, else this invocation's own directory under
/// the case's default one (`run_dir`).
pub(crate) fn out_dir(case: &Case, out: &Option<PathBuf>) -> Result<PathBuf> {
    let d = match out {
        Some(o) => o.clone(),
        None => {
            let root = std::env::var("PTXV_ROOT")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("."));
            run_dir(&default_out(&root.join("out"), case)?)?
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
    hidden_ptxas_options(case)?;
    let used = ptxv_sass::ptxas_with(
        &ptx_path,
        &cubin,
        &case.m.arch,
        case.m.ptxas_toolkit.as_deref(),
        case.m.ptxas_args.as_deref().unwrap_or_default(),
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

/// Fail closed on a PTX case whose executed cubin Triton's bundled `ptxas-blackwell` built
/// when the case does not record `ptxas_args`: Triton passes options the toolkit note omits
/// (`--regAllocOptLevel=2`), so a rebuild without them can be a different binary.
fn hidden_ptxas_options(case: &Case) -> Result<()> {
    if case.m.ptxas_args.is_some() {
        return Ok(());
    }
    let exes = case.m.executed_cubin.iter().chain(
        case.m
            .launches
            .iter()
            .filter_map(|l| l.executed_cubin.as_ref()),
    );
    for c in exes {
        let Ok(b) = std::fs::read(case.dir.join(c)) else {
            continue;
        };
        if let Some(p) = ptxv_sass::tk_producer(&b).filter(|p| p == "ptxas-blackwell") {
            bail!(
                "the executed cubin {c} was built by Triton's {p}, which runs with options its \
                 toolkit note does not record (--regAllocOptLevel=2): the case needs \
                 ptxas_args (the options that reproduce the executed code; ptxv capture \
                 records them) before the PTX can be rebuilt"
            );
        }
    }
    Ok(())
}

/// Rebuild status of a PTX case variant's cubin, keyed by `rebuild_key`.
static PTX_REBUILD: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<String, PtxRebuild>>,
> = std::sync::LazyLock::new(Default::default);

fn rebuild_key(case: &Case, variant: &str) -> String {
    format!("{}#{variant}", case.dir.display())
}

/// How the cubin ptxv compiled from a captured case's PTX compares with the cubin the GPU
/// executed (`executed_cubin`): `status` is "match" (every launched kernel's `.text` equal),
/// "mismatch" (the verdicts then check a binary the GPU did not run), or "what-if" (a
/// `variant_of` case: the PTX was edited by hand, so the difference is expected).
#[derive(Clone, Debug, serde::Serialize)]
struct PtxRebuild {
    status: &'static str,
    /// The case this one is a hand-edited variant of (`variant_of`).
    #[serde(skip_serializing_if = "Option::is_none")]
    variant_of: Option<String>,
    /// The extra ptxas options used (`ptxas_args`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    ptxas_args: Vec<String>,
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
        } else if case.m.variant_of.is_some() {
            "what-if"
        } else {
            "mismatch"
        },
        variant_of: case.m.variant_of.clone(),
        ptxas_args: case.m.ptxas_args.clone().unwrap_or_default(),
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
    if r.status == "what-if" {
        return Some(format!(
            "ptx_rebuild: what-if: the PTX is a hand-edited variant of {}, so the cubin ptxv \
             built from it differs from the executed one in {} (expected): verdicts check a \
             binary the GPU did not run",
            r.variant_of.as_deref().unwrap_or("?"),
            r.differing.join(", ")
        ));
    }
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

/// ptxas 13.0's error for a kernel whose tcgen05 instructions name both .cta_group values.
const PTXAS_CTA_GROUP_MIX: &str = "uses single CTA(.cta_group::1) and CTA pair granularity(.cta_group::2) and that is not allowed";

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
    let (ptx, cubin) = match ptx_cubin(case, variant, dir, delays) {
        Ok(x) => x,
        // ptxas refuses a kernel whose tcgen05 instructions mix .cta_group values; the PTX
        // front still checks it, so the user gets the `cta-group-mix` violation, not an error
        // (no cubin: no SASS front, no GPU run).
        Err(e) if front == "ptx" && format!("{e:#}").contains(PTXAS_CTA_GROUP_MIX) => {
            let ptx = add_delays(&case.variant_ptx(variant)?, delays)?;
            (ptx, dir.join(format!("{variant}.cubin")))
        }
        Err(e) => return Err(e),
    };
    let variant = &if delays.is_empty() {
        variant.to_string()
    } else {
        format!("{variant}.delayed")
    };
    let mut ks = Vec::new();
    // The ptxas whose lowering the PTX front models (PTX-FMAD-X2-VERSION): the case's
    // ptxas_toolkit, else the registry toolkit that rebuilds it; never the ambient default.
    let toolkit = case
        .m
        .ptxas_toolkit
        .as_deref()
        .and_then(ptxv_ptx::toolkit_version)
        .or_else(|| ptxv_sass::registry_toolkit(&ptx, None));
    for l in case.m.all_launches() {
        if l.cubin.is_some() {
            bail!(
                "launch {} names its own cubin: only binary-only cases may",
                l.entry
            );
        }
        ks.push(match front {
            "ptx" => ptxv_ptx::parse_kernel_for(&ptx, Some(l.entry), toolkit)?,
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

/// `ptxv hw`'s default watchdog: 20 s plus 1 s per MB of the case directory's files (a
/// captured case's buffers: a 75 MB case gets 95 s).
fn default_hw_timeout(case: &Case) -> f64 {
    let bytes: u64 = std::fs::read_dir(&case.dir)
        .map(|d| {
            d.filter_map(|e| e.ok()?.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0);
    20.0 + bytes as f64 / 1e6
}

/// The variants `--variant` names: `all` or a comma list of ids. Fails closed: an unknown id
/// or a case without variants is an error, not an empty (passing) run.
fn variants(case: &Case, v: &str) -> Result<Vec<String>> {
    let vs: Vec<String> = if v == "all" {
        case.m.variants.iter().map(|x| x.id.clone()).collect()
    } else {
        v.split(',').map(|x| x.trim().to_string()).collect()
    };
    if vs.is_empty() || vs.iter().any(|x| x.is_empty()) {
        bail!(
            "--variant {v}: case {} names no variant to run",
            case.m.name
        );
    }
    for x in &vs {
        case.variant(x)?;
    }
    Ok(vs)
}

/// The front-ends `--front` names for this case. Fails closed: `--front ptx` on a
/// binary-only case (no `ptx =`, a `cubin =`) and an unknown front are errors, not an empty
/// (passing) run; `both` on a binary-only case is the SASS front.
fn fronts(case: &Case, f: &str) -> Result<Vec<&'static str>> {
    let binary_only = case.m.cubin.is_some() || case.m.ptx.is_empty();
    Ok(match f {
        "ptx" if binary_only => bail!(
            "--front ptx: case {} has no PTX (binary-only{}); only the SASS front applies",
            case.m.name,
            case.m
                .cubin
                .as_ref()
                .map(|c| format!(", cubin = {c}"))
                .unwrap_or_default()
        ),
        "ptx" => vec!["ptx"],
        "sass" => vec!["sass"],
        "both" if binary_only => vec!["sass"],
        "both" => vec!["ptx", "sass"],
        _ => bail!("--front {f}: expected ptx, sass or both"),
    })
}

/// `--mutate-input` specs.
static MUTATE_SPECS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();

/// `--coverage-only`.
static COVERAGE_ONLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
/// `--exhaustive-concolic`.
static EXH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
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
    // --time-budget: at most what is left of it (a spent budget stops the run at once; 0
    // would mean no cap).
    let seconds = match ptxv_sim::membudget::time_budget_left() {
        Some(left) => Some(
            seconds
                .map_or(left.as_secs_f64(), |s| s.min(left.as_secs_f64()))
                .max(1e-3),
        ),
        None => seconds,
    };
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

/// Where the hardware profile name comes from: the flag (`check --hw-profile`), else the
/// environment.
fn hw_profile_source() -> &'static str {
    if HW_PROFILE.get().is_some() {
        "--hw-profile"
    } else {
        "PTXV_HW_PROFILE"
    }
}

/// `hw_profile`, refused for a name ptxv does not know: such a name used to turn the
/// GB300-profile re-run off without a word (a typo such as GB300). Every path that checks
/// (`check`, `suite`) goes through it.
fn checked_hw_profile() -> Result<(String, Option<&'static [ptxv_sim::hwprofile::Item]>)> {
    let (name, profile) = hw_profile();
    anyhow::ensure!(
        profile.is_some() || name == "none",
        "unknown hardware profile {name:?} ({}): gb300 or none",
        hw_profile_source()
    );
    Ok((name, profile))
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
    /// --time-budget: what was left of the whole check's budget, in seconds, when the re-run
    /// started (it shares that budget); absent without a budget.
    #[serde(skip_serializing_if = "Option::is_none")]
    budget_left_seconds: Option<f64>,
}

/// One comparison of a terminal output with the captured GPU bytes (`CheckRow::gpu_compare`).
#[derive(Serialize)]
struct GpuCompareRow {
    /// The terminal output (`output N` of `expect_report`).
    output: usize,
    /// The output buffer, or `counter word N` (C-TICKET-ADDR).
    part: String,
    result: ptxv_case::GpuMatch,
}

#[derive(Serialize)]
struct CheckRow {
    variant: String,
    front: String,
    /// Captured cases: the simulator's outputs against the GPU's (`expect`), per output.
    expect_report: Vec<String>,
    /// The decision each `expect_report` line states, one entry per comparison (and per
    /// output buffer without captured bytes): `bitwise`, `within_ulp` (not a match),
    /// `mismatch`, `permuted_equal`, `not_compared`, `no_capture`.
    gpu_compare: Vec<GpuCompareRow>,
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
    /// Why there is no two-level verdict although the strict check reported something
    /// undefined: --no-gb300, PTXV_GB300=0, --hw-profile none or PTXV_HW_PROFILE=none turned
    /// the GB300-profile re-run off (the reason names which), or the time budget was spent.
    #[serde(skip_serializing_if = "Option::is_none")]
    two_level_skipped: Option<String>,
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
    /// `--exhaustive-concolic`: the certificate over all input values, or what blocked it.
    #[serde(skip_serializing_if = "Option::is_none")]
    exhaustive_concolic: Option<ptxv_sim::cert::Report>,
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

/// The GB300 checks are on by default: the hardware-model check (second level) and the
/// GB300-profile re-run of the two-level verdict. `--no-gb300` or `PTXV_GB300=0` turn both off;
/// what turned them off (the flag first), or None.
fn gb300_off(flag_off: bool) -> Option<&'static str> {
    if flag_off {
        Some("--no-gb300")
    } else if std::env::var("PTXV_GB300").as_deref() == Ok("0") {
        Some("PTXV_GB300=0")
    } else {
        None
    }
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
    for v in variants(case, variant)? {
        for f in fronts(case, front)? {
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
    gb300_off: Option<&'static str>,
    nonfinite: Option<bool>,
) -> Result<Vec<CheckRow>> {
    let (profile_name, profile) = checked_hw_profile()?;
    let gb300 = gb300_off.is_none();
    let dir = out_dir(case, out)?;
    let mut rows = Vec::new();
    // `--slices` on a multi-launch case: one cluster at a time would check clusters of
    // different launches against each other as unordered (false cross-launch races, e.g.
    // fixtures/r1e_pdl), so the launches are checked as launch groups (E-LAUNCH-GROUPS), which
    // keep their order, each group sliced.
    let auto_groups = (slices && groups.is_none() && !case.m.launches.is_empty()).then(|| {
        println!(
            "note: --slices on a case with {} launches: checked as `--groups auto --slices` (launch order)",
            case.m.all_launches().len()
        );
        "auto".to_string()
    });
    let groups = if auto_groups.is_some() {
        &auto_groups
    } else {
        groups
    };
    for v in variants(case, variant)? {
        // The variant's SASS check under the hardware profile, shared by its fronts' rows.
        let mut level2: Option<(Verdict, Vec<HwUsed>, f64, Option<f64>)> = None;
        for f in fronts(case, front)? {
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
                    gpu_compare: vec![],
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
                    two_level_skipped: None,
                    nonfinite: None,
                    input_coverage: Some(cov),
                    mutated_inputs: vec![],
                    mutation_refusals: vec![],
                    mutation_budget: None,
                    exhaustive_concolic: None,
                });
                continue;
            }
            // Debugging (rung 4b, large launch groups): PTXV_RUN_ONCE=1 runs one
            // deterministic schedule (explore::run_once) and reports its findings and outputs.
            if std::env::var("PTXV_RUN_ONCE").is_ok() {
                let t0 = Instant::now();
                let m1 = m.clone();
                let mut m = m;
                // N-MULTIMEM-ROUND16: the output bytes that hold a calibrated f16 / bf16 lane.
                m.track_round16();
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
                    let marks = m.round16_marks();
                    for line in case.expect_report_tol(&m.outputs(), marks.as_deref())? {
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
            let check_one = |make: &dyn Fn() -> Machine, label: &str| -> Verdict {
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
                                // Decided: complete (a certificate is a complete verdict; one
                                // listing E-WELLSYNC while incomplete is not decided).
                                group_log.borrow_mut().push((i, v.complete));
                                println!(
                                    "  {label}launch group {} {:?}: {:?} states={} terminal={} outputs={}{}{} {:.1}s",
                                    i + 1,
                                    g,
                                    v.classes(),
                                    v.states,
                                    v.terminal_states,
                                    v.distinct_outputs,
                                    if v.complete { "" } else { " INCOMPLETE" },
                                    if certified(v) { " CERTIFIED" } else { "" },
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
                                if certified(v) { " CERTIFIED" } else { "" },
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
            // ASSUME(E-SMID): one check per %smid placement when the kernel reads %smid.
            let check = |make: &dyn Fn() -> Machine, label: &str| -> Verdict {
                ptxv_sim::smid::check_placements(make, &mut |mk| check_one(mk, label))
            };
            let make1 = || machine(case, k.clone()).expect("machine");
            // The plain certificate check is what errdist / equiv run as their precondition
            // (equiv.rs `prepare`): store it for them (vcache.rs).
            // Not for a variant that edits the launch order (`after`, `waits`, `launch_order`):
            // errdist and equiv apply the edits too (`Case::for_variant`), and the cache keeps
            // to the variants without them all the same (equiv.rs `prepare_with`). Not for a
            // case changed in memory (`--residency N`): the key covers the case files, and
            // errdist and equiv check the case as they load it (`vcache::as_on_disk`).
            let plain = certify
                && !case.variant(&v)?.edits_order()
                && !slices
                && groups.is_none()
                && std::env::var("PTXV_EXPLORER").as_deref() != Ok("dpor")
                && vcache::as_on_disk(case);
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
                // ASSUME(S-MBAR-SCOPE): races ordered once async mbarrier completions count as
                // the mbarrier CTA's are contract deviations.
                ptxv_sim::twolevel::mbar_async_contract(&mut verdict, &m1, &mut |m| {
                    check(&|| m.clone(), "async completions at the mbarrier's CTA: ")
                });
            }
            let births = if nonfinite.is_some() {
                ptxv_sim::nonfinite::take()
            } else {
                vec![]
            };
            // A check the time budget cut is no verdict for later requests (vcache keys on the
            // state cap only).
            if let Some(key) = vkey
                .as_ref()
                .filter(|_| !ptxv_sim::membudget::time_budget_spent())
            {
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
            // front under the hardware profile. Both levels are reported. --no-gb300 /
            // PTXV_GB300=0 and --hw-profile none turn the re-run off (the row says so).
            let mut two_level = None;
            let mut two_level_skipped = None;
            let wanted2 = ptxv_sim::hwprofile::wanted(&verdict);
            if let Some(cause) = gb300_off.filter(|_| wanted2) {
                let why = format!(
                    "the strict verdict has {}; {cause} turned the GB300-profile re-run off",
                    verdict.classes().join(",")
                );
                println!("    sass_gb300: not checked: {why}");
                two_level_skipped = Some(why);
            } else if wanted2 && profile.is_none() {
                // `none` (an unknown name is refused up front): skipped, and said so, as with
                // --no-gb300 (a missing `two_level` alone reads like an older binary's).
                let cause = match hw_profile_source() {
                    "--hw-profile" => "--hw-profile none",
                    _ => "PTXV_HW_PROFILE=none",
                };
                let why = format!(
                    "the strict verdict has {}; {cause} turned the GB300-profile re-run off",
                    verdict.classes().join(",")
                );
                println!("    sass_gb300: not checked: {why}");
                two_level_skipped = Some(why);
            }
            // --time-budget: the re-run shares the whole check's budget, and does not start once
            // it is spent (a re-run the budget cuts is incomplete, with its notes).
            let left = ptxv_sim::membudget::time_budget_left();
            let spent = ptxv_sim::membudget::time_budget_spent() && level2.is_none();
            if wanted2 && gb300 && profile.is_some() && spent {
                let why = format!(
                    "the strict verdict has {}; the time budget (--time-budget) was spent before the GB300-profile re-run",
                    verdict.classes().join(",")
                );
                println!("    sass_gb300: not checked: {why}");
                two_level_skipped = Some(why);
            }
            // No SASS for a kernel that mixes .cta_group values (ptxas refuses it): no GB300
            // profile level.
            let no_sass = f == "ptx" && !ptxv_sim::explore::cta_group_mix(&k).is_empty();
            if wanted2 && gb300 && profile.is_some() && !spent && no_sass {
                let why = format!(
                    "the strict verdict has {}; ptxas refuses a kernel that mixes .cta_group values, so there is no SASS to re-run",
                    verdict.classes().join(",")
                );
                println!("    sass_gb300: not checked: {why}");
                two_level_skipped = Some(why);
            }
            if let Some(p) = profile.filter(|_| wanted2 && gb300 && !spent && !no_sass) {
                if level2.is_none() {
                    if let Some(l) = left {
                        println!(
                            "    GB300 profile: re-run with {:.1} s of the time budget left (--time-budget)",
                            l.as_secs_f64()
                        );
                    }
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
                    level2 = Some((
                        v2,
                        used,
                        t0.elapsed().as_secs_f64(),
                        left.map(|l| l.as_secs_f64()),
                    ));
                }
                let (v2, used, secs, budget_left) = level2.as_ref().unwrap();
                let t = TwoLevel {
                    ptx_contract: verdict.classes().join(","),
                    contract_front: f.into(),
                    sass_gb300: ptxv_sim::hwprofile::result(v2),
                    hw_profile: profile_name.clone(),
                    hw_assumptions: used.clone(),
                    sass_classes: v2.classes(),
                    sass_complete: v2.complete,
                    sass_certified: certified(v2),
                    sass_states: v2.states,
                    sass_findings: v2.findings.clone(),
                    sass_notes: v2.notes.clone(),
                    seconds: *secs,
                    budget_left_seconds: *budget_left,
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
                        "    input-independent: in one run no branch, guard, address or synchronization operand derives from input data through the modeled provenance (registers; shared, global and local memory; TMEM; asynchronous copies), so control path, accesses and synchronization are the same for every input of this shape and these parameters as far as that provenance is complete. It covers control, addresses and synchronization only: value faults (an integer division by zero, a value that leaves a modeled range) and value-dependent parts (same-value writes, output determinism, the comparison with the GPU output) hold for this input only{}{}",
                        if raced {
                            "; the verdict has a race, under which another schedule can move data differently"
                        } else {
                            ""
                        },
                        // ASSUME(E-SMID): the provenance run has one SM placement.
                        if ptxv_sim::smid::reads_smid(&k) {
                            "; %smid is read: the provenance run used the spread SM placement only (E-SMID)"
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
                    // Nested in the whole check's budget (--time-budget): the earlier end
                    // applies, here and to the explorations (membudget::set_deadline).
                    let end =
                        ptxv_sim::membudget::time_budget_end().map_or(b_end, |t| t.min(b_end));
                    ptxv_sim::membudget::set_deadline(Some(b_end));
                    for (checked, mu) in muts.into_iter().enumerate() {
                        let now = Instant::now();
                        if now >= end || states_left == 0 {
                            let why = if now >= b_end {
                                format!(
                                    "the total wall budget ({:.0} s, --mutate-budget-seconds) is spent",
                                    bud.mutate_seconds
                                )
                            } else if now >= end {
                                "the time budget of the whole check (--time-budget) is spent"
                                    .to_string()
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
                        let left = (end - Instant::now()).as_secs_f64();
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
                            certified: certified(&vm),
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
            let mut gpu_compare = Vec::new();
            // N-MULTIMEM-ROUND16: the output bytes that hold an f16 / bf16 multimem.ld_reduce
            // lane whose rounding the model only calibrates, from one deterministic run of a
            // multicast case (`Machine::round16_marks`); a terminal output takes them only
            // when it is that run's output byte for byte, any other is compared bitwise
            // (MISS 110).
            let round16 = if case.m.multicast.is_empty() {
                None
            } else {
                let mut m = make1();
                m.track_round16();
                let (m, _, ok) = run_once(m);
                ok.then(|| (m.outputs(), m.round16_marks().unwrap_or_default()))
            };
            // C-TICKET-ADDR: outputs compared up to the certified permutation of rows.
            let perm_rows = verdict
                .findings
                .iter()
                .find(|f| f.kind == "clean-up-to-permutation")
                .map(|f| f.perm_rows.clone());
            for (i, o) in verdict.outputs.iter().enumerate() {
                let compared = match &perm_rows {
                    Some(r) => case.expect_compare_perm(o, r)?,
                    None => case.expect_compare_tol(
                        o,
                        round16
                            .as_ref()
                            .filter(|(out, _)| out == o)
                            .map(|(_, marks)| marks.as_slice()),
                    )?,
                };
                for c in compared {
                    if let Some(line) = &c.line {
                        println!("    expect (output {i}): {line}");
                        expect_report.push(format!("output {i}: {line}"));
                    }
                    gpu_compare.push(GpuCompareRow {
                        output: i,
                        part: c.part,
                        result: c.result,
                    });
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
            let exhaustive_concolic = if EXH.get().copied().unwrap_or(false) {
                let (dom, _) = input_domain(case, &m1)?;
                let env = |k: &str, d: f64| {
                    std::env::var(k)
                        .ok()
                        .and_then(|x| x.parse().ok())
                        .unwrap_or(d)
                };
                let opts = ptxv_sim::cert::Options {
                    max_paths: env("PTXV_EXH_PATHS", 8.0) as usize,
                    budget: env("PTXV_EXH_BUDGET", 3600.0),
                    ..Default::default()
                };
                let rep_of = |vm: &Verdict| ptxv_sim::cert::RepCheck {
                    classes: vm.classes(),
                    complete: vm.complete,
                    certified: certified(vm),
                    states: vm.states,
                    findings: vm
                        .findings
                        .iter()
                        .take(4)
                        .map(|x| x.msg.lines().next().unwrap_or("").to_string())
                        .collect(),
                    assumes: vm.assumptions.clone(),
                    seconds: vm.seconds,
                };
                let captured = rep_of(&verdict);
                let r = ptxv_sim::cert::expand(&m1, &dom, &opts, &mut |mm| {
                    let same = mm
                        .launch
                        .allocs
                        .iter()
                        .zip(&m1.launch.allocs)
                        .all(|(a, b)| a.init == b.init);
                    if same {
                        return captured.clone();
                    }
                    // Triage: the expansion without the representatives' checks.
                    if std::env::var_os("PTXV_EXH_NO_CHECK").is_some() {
                        return ptxv_sim::cert::RepCheck {
                            classes: vec!["not-run".into()],
                            complete: false,
                            certified: false,
                            states: 0,
                            findings: vec![],
                            assumes: vec![],
                            seconds: 0.0,
                        };
                    }
                    let vm = check(&|| mm.clone(), "exhaustive concolic path: ");
                    rep_of(&vm)
                });
                println!(
                    "    exhaustive concolic expansion: {} ({} path(s), {} region(s) closed by a path, {} infeasible, {} operand(s) fixed, {} blocker(s), {:.1}s)",
                    r.status,
                    r.paths.len(),
                    r.closed_by_path,
                    r.closed_infeasible,
                    r.operands_constant,
                    r.blockers.len(),
                    r.seconds
                );
                if r.status == "refuted" {
                    println!(
                        "    exhaustive concolic: refuted {} the declared input invariants",
                        if r.refuted_within_invariants {
                            "within"
                        } else {
                            "only outside"
                        }
                    );
                }
                for b in r.blockers.iter().take(8) {
                    println!(
                        "    exhaustive concolic: blocked by {} {}: {}",
                        b.kind, b.at, b.detail
                    );
                }
                Some(r)
            } else {
                None
            };
            rows.push(CheckRow {
                variant: v.clone(),
                front: f.into(),
                expect_report,
                gpu_compare,
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
                two_level_skipped,
                nonfinite: nf,
                input_coverage,
                mutated_inputs,
                mutation_refusals,
                mutation_budget,
                exhaustive_concolic,
            });
        }
    }
    Ok(rows)
}

/// The spacing of the bf16 (`f16`: f16) grid just above |x| (x: the 16-bit value widened to
/// f32 bits); the smallest subnormal at 0.
fn ulp16(bits: u32, f16: bool) -> f64 {
    let x = (f32::from_bits(bits) as f64).abs();
    if f16 {
        // 10 fraction bits; subnormal spacing 2^-24 below 2^-14.
        let e = if x < 2f64.powi(-14) {
            -14
        } else {
            x.log2().floor() as i32
        };
        2f64.powi(e - 10)
    } else {
        // bf16: f32's exponent range, 7 fraction bits.
        ptxv_sim::num::ulp_f32(f32::from_bits(bits)) * 65536.0
    }
}

/// Exact values `a` and `b` are equal: both finite and equal, or the same non-finite value
/// (the same infinity, or both NaN) (N-NUM-NONFINITE: the f64 expansions of infinities do not
/// subtract to zero).
fn exact_eq(a: &ptxv_sim::exact::Expansion, b: &ptxv_sim::exact::Expansion) -> bool {
    let (x, y) = (ptxv_sim::exact::approx(a), ptxv_sim::exact::approx(b));
    if !x.is_finite() || !y.is_finite() {
        return x == y || x.is_nan() && y.is_nan();
    }
    ptxv_sim::exact::is_zero(&ptxv_sim::exact::sub(a, b))
}

#[derive(Serialize)]
struct NumSummary {
    front: String,
    /// The output buffer this summary covers (every output gets one).
    output: String,
    outputs: usize,
    /// Elements whose error is a claim. The others (`untracked`) have an input-derived value
    /// without a tracked derivation in their tree: the fields below cover the analyzed ones.
    analyzed: usize,
    untracked: usize,
    /// Of the `untracked` elements, those with a non-finite simulated or exact value whose
    /// exact value is not determined (N-NUM-NONFINITE).
    nonfinite: usize,
    /// Why no element of this output was analyzed (e.g. not an f32 buffer).
    refused: Option<String>,
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
    // E-LAUNCH-ORDER: the run is of the variant's launch order (`after`, `waits`,
    // `launch_order`), as in `ptxv check`, not the case's own.
    let mut case = case.for_variant(variant)?;
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
    // ASSUME(N-NUM-TRACK): (MISS 108) the PTX front's verdict per output, when it ran: None for
    // an output it refused, else per element the exact value of an analyzed element. ptxas folds
    // roundings of constants (`mul.rn`, `add.rn`, `ex2.approx` of immediates) into immediates the
    // SASS front would take as exact constants; the PTX front has the roundings. So the SASS front
    // claims only elements the PTX front claims, with the same exact value.
    type PtxOut = Option<Vec<Option<ptxv_sim::exact::Expansion>>>;
    let mut ptx_verdict: Option<std::collections::HashMap<String, PtxOut>> = None;
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
        if f == "ptx" {
            ptx_verdict = Some(std::collections::HashMap::new());
        }
        let mut m = machine(case, k)?;
        let mut ns = ptxv_sim::num::NumState::with_mma_eps(m.launch.mma_model.eps());
        // ASSUME(N-NUM-TRACK): input-derived values are tainted, so a value whose rounding
        // history is dropped becomes an `Opaque` leaf (or an untracked output word), never an
        // exact constant. So is a rounding instruction's result without a node whatever its
        // operands (`rounding_taint`: a product of kernel parameters, a MUFU of a constant).
        // Input elements stay exact `Input` leaves (as in equiv, ASSUME(N-HALF-LEAF)).
        ns.rounding_taint = true;
        for &o in &m.launch.outputs {
            if equiv::out_only(case, &m.launch.allocs[o].name) {
                ns.not_inputs.insert(o);
            }
            // bf16 / f16 outputs: one value per 16-bit element, its node from the 16-bit
            // provenance (ASSUME(N-HALF-LEAF, N-ROUND)), as equiv reads them.
            let name = &m.launch.allocs[o].name;
            match case
                .m
                .buffers
                .iter()
                .find(|b| &b.name == name)
                .map(|b| b.dtype.as_str())
            {
                Some("bf16") => {
                    ns.half.bf16_outputs.insert(o);
                    ns.half.track = true;
                }
                Some("f16") => {
                    ns.half.f16_outputs.insert(o);
                    ns.half.track = true;
                }
                _ => {}
            }
        }
        for (i, a) in m.launch.allocs.iter().enumerate() {
            if equiv::out_only(case, &a.name) {
                continue;
            }
            ns.taint.taint_alloc(i, a.init.len());
            match case
                .m
                .buffers
                .iter()
                .find(|b| b.name == a.name)
                .map(|b| b.dtype.as_str())
            {
                Some("f32") => {
                    ns.f32_inputs.insert(i);
                }
                Some("bf16") => {
                    ns.half.inputs.insert(i, ptxv_sim::halfin::Half::Bf16);
                }
                Some("f16") => {
                    ns.half.inputs.insert(i, ptxv_sim::halfin::Half::F16);
                }
                Some("u8" | "u16" | "e4m3") => {
                    ns.half.inputs.insert(i, ptxv_sim::halfin::Half::Byte);
                }
                _ => {}
            }
        }
        m.num = Some(Box::new(ns));
        let (mut m, findings, ok) = run_once(m);
        if !ok {
            bail!("numerics run of {variant} ({f}) did not terminate cleanly: {findings:?}");
        }
        m.half_out_leaves();
        note_ref_faults(&format!("numerics run of {variant} ({f})"), &findings);
        // A run that faulted or hit a model limit did not execute the kernel as the hardware
        // does (a faulting thread stops): its outputs carry no numerics claim (N-NUM-TRACK).
        let ns = m.num.as_ref().unwrap();
        // ASSUME(N-NUM-TRACK): a branch, exit or store (any effect) under a guard that
        // depends on input data, or an input-dependent branch target, memory address or
        // shuffle lane: the exact computation may take another path than this run or access
        // other elements, so no output carries a claim.
        let control = ns.taint.control_pcs.iter().next().map(|&pc| {
            format!(
                "control flow or an address depends on input data or on a rounded value at {} (the exact computation may take another path or access other elements)",
                m.k.describe(pc as usize)
            )
        });
        let broken = findings
            .iter()
            .find(|x| x.kind == "fault" || x.kind == "unmodeled")
            .map(|x| {
                let msg = x.msg.lines().next().unwrap_or("");
                format!("the numerics run reported {}: {msg}", x.kind)
            })
            .or(control);
        std::fs::write(dir.join(format!("{variant}.{f}.num.out.bin")), m.outputs())?;
        let warnings: Vec<String> = ns.warnings.iter().cloned().collect();
        let output_hash = format!("{:016x}", fnv(&m.outputs()));
        for (oslot, &oi) in m.launch.outputs.iter().enumerate() {
            let oname = m.launch.allocs[oi].name.clone();
            // Per-output file names; the first output keeps the historical names.
            let osfx = if oslot == 0 {
                suffix.clone()
            } else {
                format!("{suffix}.{oname}")
            };
            let mem = &m.gmem[oi];
            let dtype = case
                .m
                .buffers
                .iter()
                .find(|b| b.name == oname)
                .map(|b| b.dtype.clone())
                .unwrap_or_default();
            // Element size: f32, or bf16 / f16 (values widened to f32, errors in that format's
            // ulps).
            let es = match dtype.as_str() {
                "f32" => 4,
                "bf16" | "f16" => 2,
                _ => 0,
            };
            let refusal = broken.clone().or_else(|| {
                (es == 0 || mem.bytes.len() % es != 0).then(|| {
                    format!(
                        "output {oname} is a {dtype} buffer of {} bytes; num analyzes f32, bf16 and f16 elements only",
                        mem.bytes.len()
                    )
                })
            });
            // The PTX front's verdict on this output (SASS front, when the PTX front ran).
            let ptx_out = match (f, ptx_verdict.as_ref()) {
                ("sass", Some(v)) => Some(v.get(&oname).cloned().flatten()),
                _ => None,
            };
            let refusal = refusal.or_else(|| {
                matches!(ptx_out, Some(None)).then(|| {
                    "the PTX front makes no claim on this output (ptxas may fold a rounding of constants into an immediate the SASS front takes as exact)".to_string()
                })
            });
            if let Some(why) = refusal {
                if f == "ptx" {
                    if let Some(v) = ptx_verdict.as_mut() {
                        v.insert(oname.clone(), None);
                    }
                }
                println!("num {variant} {f:<4}: output={oname} NOT ANALYZED: {why}");
                sums.push(NumSummary {
                    front: f.into(),
                    output: oname,
                    outputs: 0,
                    analyzed: 0,
                    untracked: 0,
                    nonfinite: 0,
                    refused: Some(why),
                    thm1_all_exact: false,
                    spec_all_match: None,
                    cor1a_never_violated: false,
                    running_never_violated: false,
                    max_abs_err: f64::NAN,
                    max_err_ulps: f64::NAN,
                    mean_cor1a_over_err: f64::NAN,
                    warnings: warnings.clone(),
                    output_hash: output_hash.clone(),
                });
                continue;
            }
            let n = mem.bytes.len() / es;
            let mut recs = Vec::new();
            let mut spec_ok = true;
            let half = es == 2;
            let f16 = dtype == "f16";
            let ptx_out = ptx_out.flatten();
            // PTX front: each analyzed element's exact value; SASS front: the elements withheld
            // for want of a PTX claim with the same exact value.
            let mut exacts = Vec::with_capacity(n);
            let mut withheld = 0;
            for i in 0..n {
                let (bits, node) = if half {
                    let h = u16::from_le_bytes([0, 1].map(|j| *mem.bytes.get(i * 2 + j)));
                    let bits = if f16 {
                        ptxv_sim::elementwise::f16_to_f32_bits(h)
                    } else {
                        (h as u32) << 16
                    };
                    (bits, m.half_out_node(oi, i as u64, f16))
                } else {
                    let bits = u32::from_le_bytes([0, 1, 2, 3].map(|j| *mem.bytes.get(i * 4 + j)));
                    let node = ns.mem_node(ptxv_sim::exec::Loc::Global {
                        alloc: oi,
                        off: (i * 4) as u64,
                    });
                    (bits, node)
                };
                let mut r = ns.out_rec(i, bits, node);
                // An infinite or undetermined (NaN) error keeps its value (N-NUM-NONFINITE).
                if half && r.abs_err.is_finite() {
                    r.err_ulps = r.abs_err / ulp16(bits, f16);
                }
                if node.is_none()
                    && (0..es).any(|b| {
                        ns.taint
                            .memory_unit_input_derived((1, oi as u64, (i * es + b) as u64))
                    })
                {
                    r.untracked = true;
                    r.thm1_exact = false;
                }
                let ex = node
                    .map(|nd| ns.nodes[nd as usize].exact.clone())
                    .unwrap_or_else(|| ptxv_sim::exact::from(f32::from_bits(bits) as f64));
                if let Some(pe) = &ptx_out {
                    let same = pe
                        .get(i)
                        .and_then(|e| e.as_ref())
                        .is_some_and(|e| exact_eq(e, &ex));
                    if !r.untracked && !same {
                        r.untracked = true;
                        r.thm1_exact = false;
                        withheld += 1;
                    }
                }
                exacts.push((!r.untracked).then_some(ex));
                if oslot == 0 && !half {
                    if let Some(s) = &spec {
                        let ex = node
                            .map(|nd| ns.nodes[nd as usize].exact.clone())
                            .unwrap_or_else(|| ptxv_sim::exact::from(f32::from_bits(bits) as f64));
                        spec_ok &= !r.untracked && exact_eq(&ex, &s[i]);
                    }
                }
                recs.push(r);
            }
            if f == "ptx" {
                if let Some(v) = ptx_verdict.as_mut() {
                    v.insert(oname.clone(), Some(exacts));
                }
            }
            let mut csv = String::from(
                "index,hw,exact,abs_err,rel_err,err_ulps,bound_cor1a,bound_running,thm1_exact,n_leaves,n_roundings,untracked,nonfinite\n",
            );
            for r in &recs {
                csv += &format!(
                    "{},{:e},{:e},{:e},{:e},{},{:e},{:e},{},{},{},{},{}\n",
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
                    r.n_roundings,
                    r.untracked,
                    r.nonfinite
                );
            }
            std::fs::write(dir.join(format!("{variant}.{f}.num{osfx}.csv")), csv)?;
            std::fs::write(
                dir.join(format!("{variant}.{f}.num{osfx}.json")),
                serde_json::to_string(&recs)?,
            )?;
            let ok: Vec<&_> = recs.iter().filter(|r| !r.untracked).collect();
            // ASSUME(N-NUM-NONFINITE): no analyzed element is skipped: an infinite error makes
            // the maximum infinite and violates both bounds (an undetermined one, NaN, would
            // count as infinite; analyzed elements have none). The mean bound / error ratio takes
            // every analyzed element with a nonzero error; an infinite error counts 0.
            let worst = |x: f64| if x.is_nan() { f64::INFINITY } else { x };
            let nz: Vec<&_> = ok
                .iter()
                .filter(|r| r.abs_err > 0.0 || r.abs_err.is_nan())
                .copied()
                .collect();
            let ratio = |r: &ptxv_sim::num::OutRec| {
                if r.abs_err.is_finite() {
                    r.bound_cor1a / r.abs_err
                } else {
                    0.0
                }
            };
            let nonfinite = recs.iter().filter(|r| r.nonfinite).count();
            let s = NumSummary {
                front: f.into(),
                output: oname.clone(),
                outputs: n,
                analyzed: ok.len(),
                untracked: n - ok.len(),
                nonfinite,
                refused: None,
                thm1_all_exact: !ok.is_empty() && ok.iter().all(|r| r.thm1_exact),
                spec_all_match: (oslot == 0 && !half)
                    .then_some(())
                    .and(spec.as_ref())
                    .map(|_| spec_ok),
                cor1a_never_violated: ok
                    .iter()
                    .all(|r| r.abs_err.is_finite() && r.abs_err <= r.bound_cor1a),
                running_never_violated: ok
                    .iter()
                    .all(|r| r.abs_err.is_finite() && r.abs_err <= r.bound_running),
                max_abs_err: ok.iter().map(|r| worst(r.abs_err)).fold(0.0, f64::max),
                max_err_ulps: ok.iter().map(|r| worst(r.err_ulps)).fold(0.0, f64::max),
                mean_cor1a_over_err: if nz.is_empty() {
                    0.0
                } else {
                    nz.iter().map(|&r| ratio(r)).sum::<f64>() / nz.len() as f64
                },
                warnings: warnings.clone(),
                output_hash: output_hash.clone(),
            };
            let oname_note = if m.launch.outputs.len() > 1 {
                format!("output={oname} ")
            } else {
                String::new()
            };
            // The earliest instruction that read an untracked input-derived value (none: the
            // output words themselves were written untracked).
            let first = recs
                .iter()
                .filter_map(|r| r.untracked_pc)
                .min()
                .map(|pc| format!("; first read untracked at {}", m.k.describe(pc as usize)))
                .unwrap_or_default();
            // Why elements are not analyzed: a dropped rounding history, or (N-NUM-NONFINITE) a
            // non-finite value whose exact value is not determined.
            let hist = recs.iter().filter(|r| r.untracked && !r.nonfinite).count() - withheld;
            let mut why = vec![];
            if hist > 0 {
                let k = if nonfinite > 0 || withheld > 0 {
                    format!(": {hist}")
                } else {
                    String::new()
                };
                why.push(format!("rounding history dropped{k}{first}"));
            }
            if nonfinite > 0 {
                why.push(format!(
                    "a non-finite value without a determined exact value: {nonfinite}"
                ));
            }
            if withheld > 0 {
                why.push(format!(
                    "no PTX claim with the same exact value (ptxas may fold a rounding of constants into an immediate): {withheld}"
                ));
            }
            let why = why.join("; ");
            if s.untracked == n && nonfinite == 0 && withheld == 0 {
                println!(
                    "num {variant} {f:<4}: {oname_note}outputs={n} NOT ANALYZED: every element depends on an input-derived value whose rounding history was dropped (no error claim{first})"
                );
            } else if s.untracked == n {
                println!(
                    "num {variant} {f:<4}: {oname_note}outputs={n} NOT ANALYZED: no element carries an error claim ({why})"
                );
            } else {
                let partial = if s.untracked > 0 {
                    format!(
                        " NOT ANALYZED for {} of {n} ({why}); over the {} analyzed:",
                        s.untracked, s.analyzed
                    )
                } else {
                    String::new()
                };
                println!(
                    "num {variant} {f:<4}: {oname_note}outputs={}{partial} thm1_exact={} spec_match={:?} cor1a_ok={} running_ok={} max_err={:.3e} ({:.2} ulp) bound/err(mean)={:.1} out={}",
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
            }
            sums.push(s);
        }
        for w in &warnings {
            println!("    warning: {w}");
        }
        std::fs::write(
            dir.join(format!("{variant}.{f}.out{suffix}.bin")),
            m.outputs(),
        )?;
        outs.push(m.outputs());
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

/// The case a hardware worker runs (`hw-worker`, `errdist-gpu-worker`): the one in the case
/// directory with `variant`'s launch-order edits (`after`, `waits`, `launch_order`) applied
/// (`Case::for_variant`), so that the GPU runs the launch order the simulator checked.
pub(crate) fn worker_case(case: &Path, variant: Option<&str>) -> Result<Case> {
    let c = Case::load(case)?;
    match variant {
        Some(v) => c.for_variant(v),
        None => Ok(c),
    }
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
            "--variant",
            v,
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
            outcome: format!(
                "timeout after {timeout:.0} s (killed by the watchdog: a hang or a slow run; raise --timeout)"
            ),
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
    for v in variants(case, variant)? {
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
    for v in variants(case, variant)? {
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
                "--variant",
                &v,
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
            None => println!(
                "probe {v:<12} TIMEOUT (killed after {timeout}s: a hang or a slow run; raise --timeout)  {summary}"
            ),
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
        gb300_off(false),
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
    md += "\n## Numerics (good variant)\n\nRows cover the analyzed elements only; untracked elements have no error claim.\n\n| front | output | elements | analyzed | Thm 1 exact | matches spec | Cor 1a never violated | running bound never violated | max error | max ulp | mean Cor1a/err |\n|---|---|---|---|---|---|---|---|---|---|---|\n";
    for n in &nums {
        md += &format!(
            "| {} | {} | {} | {} | {} | {:?} | {} | {} | {:.3e} | {:.2} | {:.1} |\n",
            n.front,
            n.output,
            n.outputs,
            match &n.refused {
                Some(why) => format!("none: {why}"),
                None => n.analyzed.to_string(),
            },
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
    // Removes the run directories on return and on a panic (exit sites: `exit_cleanly`).
    let _run_dirs = RunDirsGuard;
    let r = run();
    if let Err(e) = &r {
        jsonout::fail(&format!("{e:#}"));
        // Multi-node: the other ranks stop too instead of waiting for this one.
        ptxv_sim::dist::refuse(&format!("{e:#}"));
    }
    r
}

/// The check's wall-clock budget in seconds (`--time-budget`, else PTXV_TIME_BUDGET; an empty
/// PTXV_TIME_BUDGET is none). Refused in multi-node mode: each rank would decide its own cuts
/// (certificates, seed walks, explorations), and ranks in different phases wait for messages
/// the others never send.
fn time_budget_of(flag: Option<f64>) -> Result<Option<f64>> {
    let budget = match (flag, std::env::var("PTXV_TIME_BUDGET")) {
        (Some(s), _) => Some(s),
        (None, Ok(v)) if v.trim().is_empty() => None,
        (None, Ok(v)) => Some(
            v.trim()
                .parse::<f64>()
                .map_err(|_| anyhow::anyhow!("PTXV_TIME_BUDGET={v}: not a number of seconds"))?,
        ),
        (None, Err(_)) => None,
    };
    if let Some(s) = budget {
        anyhow::ensure!(
            s.is_finite() && s >= 0.0,
            "--time-budget / PTXV_TIME_BUDGET wants seconds >= 0, got {s}"
        );
        anyhow::ensure!(
            !multi_node(),
            "--time-budget / PTXV_TIME_BUDGET is not supported in multi-node mode \
             (PTXV_DIST_DIR): every rank would cut its own phases and the ranks would \
             wait for each other; run without a budget or on one node"
        );
    }
    Ok(budget)
}

/// What `check` runs only on one node: input mutation (--mutate-inputs, --mutate-input) and
/// the concolic modes (--concolic, --exhaustive-concolic). Which mutated inputs they check, and
/// how far, is decided by each rank's own clock: the coverage run's cap (a stopped run claims
/// no untaken side, so nothing is planned), --mutate-budget-seconds before each mutated check,
/// the capped deterministic pass that skips a mutation, the solvers' budgets. Ranks that
/// decide differently run different explorations, and each waits for messages of a search the
/// others never start. (Input coverage alone only prints: rank 0's notes are the report.)
fn single_node_only(
    mutate_inputs: usize,
    mutate_input: &[String],
    concolic: bool,
    exhaustive_concolic: bool,
) -> Result<()> {
    let on: Vec<&str> = [
        (mutate_inputs > 0, "--mutate-inputs"),
        (!mutate_input.is_empty(), "--mutate-input"),
        (concolic, "--concolic"),
        (exhaustive_concolic, "--exhaustive-concolic"),
    ]
    .into_iter()
    .filter_map(|(on, flag)| on.then_some(flag))
    .collect();
    anyhow::ensure!(
        on.is_empty() || !multi_node(),
        "{} not supported in multi-node mode (PTXV_DIST_DIR): each rank would decide by its \
         own clock which mutated inputs to check, and ranks that disagree wait for each other \
         forever; run it on one node",
        on.join(", ")
    );
    Ok(())
}

fn run() -> Result<()> {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            // A rank that cannot read its arguments stops the others too (`main`).
            if e.use_stderr() {
                ptxv_sim::dist::refuse(e.to_string().trim());
            }
            e.exit()
        }
    };
    // Checked before the first `dist::get` below, which connects and waits for every rank: a
    // rank that refuses tells the others (`main`, dist::refuse), which stop too.
    let budget = match &cli.cmd {
        Cmd::Check {
            time_budget,
            hw_profile,
            mutate_inputs,
            mutate_input,
            concolic,
            exhaustive_concolic,
            ..
        } => {
            if let Some(p) = hw_profile {
                let _ = HW_PROFILE.set(p.clone());
            }
            checked_hw_profile()?;
            single_node_only(
                *mutate_inputs,
                mutate_input,
                *concolic,
                *exhaustive_concolic,
            )?;
            time_budget_of(*time_budget)?
        }
        _ => None,
    };
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
            hw_profile: _,
            nonfinite,
            nonfinite_strict,
            no_input_coverage,
            coverage_only,
            concolic,
            exhaustive_concolic,
            mutate_inputs,
            mutate_elements,
            mutate_input,
            coverage_seconds,
            mutate_budget_seconds,
            mutate_budget_states,
            time_budget: _,
        } => {
            // One wall-clock budget for the whole check, set before anything runs (one past
            // what an Instant holds is none).
            if let Some(s) = budget {
                let end = Duration::try_from_secs_f64(s)
                    .ok()
                    .and_then(|d| Instant::now().checked_add(d));
                ptxv_sim::membudget::set_time_budget(end);
                eprintln!("time budget: {s} s for the whole check (--time-budget)");
            }
            let _ = concolic::ON.set(concolic);
            let _ = EXH.set(exhaustive_concolic);
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
                    exit_cleanly(1);
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
                gb300_off(no_gb300),
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
                exit_cleanly(1);
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
            let timeout = timeout.unwrap_or_else(|| default_hw_timeout(&c));
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
                for v in variants(&c, &variant)? {
                    let (k, _) = load(&c, &v, "sass", &dir)?;
                    // In the variant's launch order, as the GPU runs it (`hw-worker --variant`).
                    let (m, fs, ok) = ptxv_sim::explore::run_once(machine(&c.for_variant(&v)?, k)?);
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
                let (m, fs, ok) = ptxv_sim::explore::run_once(machine(&c.for_variant("good")?, k)?);
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
                exit_cleanly(1);
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
            is,
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
            is,
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
            // Without --out: a directory of this run under the pair's (`result_dir`), and the
            // sweeps (computed once per slice) carried from the pair's last run to the next: a
            // whole copy in, published back by rename (concurrent runs read whole files).
            const SWEEPS: [&str; 2] = ["sweeps.json", "sweeps.elementwise.json"];
            let (out, pair) = match out {
                Some(o) => (o, None),
                None => {
                    let pair = std::env::temp_dir().join(format!(
                        "ptxv-compare-{}-{}",
                        reference
                            .file_name()
                            .map_or("".into(), |d| d.to_string_lossy()),
                        candidate
                            .file_name()
                            .map_or("".into(), |d| d.to_string_lossy())
                    ));
                    let d = result_dir(&pair)?;
                    for f in SWEEPS {
                        let _ = std::fs::copy(pair.join(f), d.join(f));
                    }
                    (d, Some(pair))
                }
            };
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
            if let (Some(pair), None) = (&pair, &o.sweep_cache) {
                for f in SWEEPS {
                    let tmp = pair.join(format!(".{f}.{}", std::process::id()));
                    if std::fs::copy(o.out.join(f), &tmp).is_ok() {
                        let _ = std::fs::rename(&tmp, pair.join(f));
                    }
                }
            }
            if let Some(j) = j {
                let rp = o.out.join("report.json");
                let mut v: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(&rp)?)?;
                v["report_dir"] = serde_json::json!(o.out.canonicalize()?);
                j.emit("compare", v);
            }
        }
        Cmd::Suite { case, runs, out } => cmd_suite(&Case::load(&case)?, runs, &out)?,
        Cmd::FirstAccess {
            case,
            variant,
            front,
            out,
        } => {
            let c = Case::load(&case)?;
            let dir = out_dir(&c, &out)?;
            let (k, _) = load(&c, &variant, &front, &dir)?;
            // E-LAUNCH-ORDER: the run is of the variant's launch order (`after`, `waits`,
            // `launch_order`), as in `ptxv check`, not the case's own.
            let c = c.for_variant(&variant)?;
            let launch = ptxv_sim::build_launch(&c, &k)?;
            let want: Vec<usize> = (0..c.m.buffers.len()).collect();
            let m = Machine::new(Arc::new(k), Arc::new(launch));
            let first = ptxv_sim::coverage::access_summary(&m, &want);
            let mut doc = serde_json::Map::new();
            for (i, (write, read_back, by)) in first {
                doc.insert(
                    c.m.buffers[i].name.clone(),
                    serde_json::json!({
                        "first": if write { "write" } else { "read" },
                        "read_back": read_back,
                        "by": by,
                    }),
                );
            }
            println!("{}", serde_json::Value::Object(doc));
        }
        Cmd::ErrdistGpuWorker {
            case,
            cubin,
            dir,
            n,
            tag,
            variant,
        } => errdist::gpu_worker(&case, &cubin, &dir, n, &tag, variant.as_deref())?,
        Cmd::HwWorker {
            case,
            image,
            runs,
            expected,
            scrub,
            input,
            dump,
            variant,
        } => {
            let c = worker_case(&case, variant.as_deref())?;
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
            keep_run_dirs(); // calib.json and the measured operands are the result
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
        } => {
            keep_run_dirs(); // the dumped outputs (<variant>.probe.bin)
            cmd_probe(&Case::load(&case)?, &variant, timeout, &out)?
        }
        Cmd::SbSweep {
            case,
            runs,
            timeout,
            out,
        } => {
            keep_run_dirs(); // sweep.md / sweep.json
            sweep::run(&Case::load(&case)?, runs, timeout, &out)?
        }
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

#[cfg(test)]
mod worker_tests {
    use std::path::Path;

    /// E-LAUNCH-ORDER: a hardware worker runs the variant's launch order (`worker_case`,
    /// `Case::for_variant`). fixtures/e_equiv_order declares its order; UNK makes it unknown,
    /// UNORD gives the reader no predecessor (`after = -1`).
    #[test]
    fn worker_case_applies_the_variants_launch_order() {
        let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/e_equiv_order");
        let c = super::worker_case(&fx, None).unwrap();
        assert_eq!(c.m.launch_order.as_deref(), Some("declared"));
        assert_eq!(c.m.launches[0].after, None);
        let c = super::worker_case(&fx, Some("good")).unwrap();
        assert_eq!(c.m.launch_order.as_deref(), Some("declared"));
        let c = super::worker_case(&fx, Some("UNK")).unwrap();
        assert_eq!(c.m.launch_order.as_deref(), Some("unknown"));
        let c = super::worker_case(&fx, Some("UNORD")).unwrap();
        assert_eq!(c.m.launches[0].after, Some(-1));
        assert!(super::worker_case(&fx, Some("NO_SUCH_VARIANT")).is_err());
    }
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tier 3: Monte Carlo over inputs drawn from the spec, through the extracted numerics DAG
//! (`chain::Prog`), and the shared bookkeeping of observed errors (also used for simulator and
//! GPU outputs in validation).

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use super::chain::ExpAcc;
use super::chain::Op;
use super::chain::Prog;
use super::spec::Format;
use super::spec::Layout;
use super::spec::Spec;
use super::spec::exp_floor;
use super::special::Rng;
use super::stats::Acc;

pub static DEBUG: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("PTXV_ERRDIST_DEBUG").is_ok());
pub static DBG: Mutex<BTreeMap<i32, [f64; 4]>> = Mutex::new(BTreeMap::new());

pub const U: f64 = 1.0 / 16_777_216.0; // 2^-24

/// One kernel (or a pair) prepared for evaluation.
pub struct Setup<'a> {
    pub r: &'a Prog,
    pub c: Option<&'a Prog>,
    /// The outputs the reference tracks (an input-dependent value), ascending: the outputs
    /// Monte Carlo and validation observe.
    pub tracked: Vec<u32>,
    /// Per tracked output: the reference's MMA chain (steps), for the exact value and S.
    pub chain: Vec<Option<Vec<u32>>>,
    /// The worst-case bound |e| <= rel |y| + k S + abs (k in absolute units: k_u u; S = 1
    /// with the fourth field), when known: its distribution and the check of every observed
    /// error against it.
    pub wc: Option<(f64, f64, f64, bool)>,
}

impl<'a> Setup<'a> {
    pub fn new(r: &'a Prog, c: Option<&'a Prog>) -> Result<Setup<'a>, String> {
        if let Some(c) = c {
            if c.outputs.len() != r.outputs.len() {
                return Err(format!(
                    "outputs differ in number: {} and {}",
                    r.outputs.len(),
                    c.outputs.len()
                ));
            }
        }
        // TF32 splits: the exact value is that of the f32 inputs (hi(x) stands for x, lo parts
        // are errors), so such outputs take the real-arithmetic path, not the chain's products
        // of parts (N-ERRDIST-DAG).
        let tracked: Vec<u32> = (0..r.outputs.len() as u32)
            .filter(|&o| r.outputs[o as usize].is_some())
            .collect();
        let chain: Vec<Option<Vec<u32>>> = tracked
            .iter()
            .map(|&o| match r.outputs[o as usize] {
                Some(root) if r.has_tf32(root) => None,
                _ => r.chain(o as usize).map(|x| x.0),
            })
            .collect();
        Ok(Setup {
            r,
            c,
            tracked,
            chain,
            wc: None,
        })
    }
    pub fn pair(&self) -> bool {
        self.c.is_some()
    }
}

/// ulp of the output format at |y|.
pub fn ulp_out(y: f64, f: Format) -> f64 {
    let (mb, emin) = (f.mbits(), f.emin());
    let e = if y == 0.0 {
        emin
    } else {
        exp_floor(y).max(emin)
    };
    2f64.powi(e - mb)
}

/// Round an exact value (y + y_lo) to the output format, to nearest even.
pub fn round_out(y: f64, y_lo: f64, f: Format) -> f64 {
    let (r, _) = f.round(y, false);
    if y_lo == 0.0 {
        return r;
    }
    // Ties of y alone are broken by y_lo; elsewhere y decides.
    let q = ulp_out(y, f);
    let half = (r - y).abs();
    if half * 2.0 == q && (r - y).signum() * y_lo.signum() < 0.0 {
        // r is on the wrong side of a tie that y_lo resolves: step toward y_lo.
        return f.round(y + y_lo.signum() * q * 0.25, false).0;
    }
    r
}

/// Observed errors of one source (DAG Monte Carlo, simulator, GPU).
#[derive(Clone, Debug, Default)]
pub struct Obs {
    pub inputs: u64,
    pub outputs_per_input: u64,
    pub abs: Acc,
    pub u: Acc,
    pub ulp: Acc,
    /// Pair mode: each kernel against the exact value (abs and u scale) and the cross terms.
    pub er: Acc,
    pub ec: Acc,
    pub erc: f64,
    pub er_u: Acc,
    pub ec_u: Acc,
    pub erc_u: f64,
    /// Outputs whose bits differ (pair: C vs R; single: C vs the correctly rounded exact).
    pub differ: u64,
    pub compared: u64,
    /// Exact multiples of the output ulp: k -> count; and how many were not multiples.
    pub pmf_ulp: BTreeMap<i64, u64>,
    pub off_lattice: u64,
    /// Outputs whose exact value is 0 or subnormal in the output format: left out of the
    /// ulp-scale moments (e / ulp(y) is unbounded there; the PMF and P(differ) keep them).
    pub ulp_tiny: u64,
    /// The worst-case bound (`Setup::wc`): in output ulps (sum, max, kept subsample, outputs
    /// left out), and observed |e| against it (checked, exceeding, largest ratio).
    pub wc_sum: f64,
    pub wc_max: f64,
    pub wc_n: u64,
    pub wc_tiny: u64,
    pub wc_sample: Vec<f64>,
    pub wc_checked: u64,
    pub wc_exceed: u64,
    pub wc_ratio: f64,
    /// Subsample of (u-scale, ulp-scale, abs) errors for quantiles and tests.
    pub sample: Vec<(f64, f64, f64)>,
    /// For the subsample: (exact value y, the computed output's error against it: R's in
    /// single mode, C's in a pair), the errstate's joint pairs.
    pub pairs: Vec<(f64, f64)>,
    /// Per pair: (output index, input sample index) (cross-output correlations).
    pub pair_at: Vec<(u32, u64)>,
    /// The input sample being observed (set by the caller).
    pub cur: u64,
    /// Per input: mean u-scale error and mean |u-scale error| (batch means for CIs).
    pub input_means: Vec<(f64, f64)>,
    /// Per input: the largest |u-scale error| (for "largest fraction of the bound" claims).
    pub input_max_u: Vec<f64>,
    /// Per input: the fraction of its outputs that differ (batch means for P(differ)).
    pub input_differ: Vec<f64>,
    /// Inputs with an out-of-domain event, by cause.
    pub ood: BTreeMap<String, u64>,
    pub ood_inputs: u64,
    /// Outputs with S = 0 (no nonzero term) or an untracked value: skipped.
    pub skipped: u64,
    /// Outputs (of R or C, or the exact value) that are NaN / Inf: out of domain, outside
    /// the error statistics; and the inputs with any.
    pub nonfinite: u64,
    pub nonfinite_inputs: u64,
    /// Tracked outputs seen (finite or not).
    pub seen: u64,
    /// Per output: u-scale moments (when requested).
    pub per_output: Vec<Acc>,
}

impl Obs {
    pub fn merge(&mut self, o: Obs) {
        self.inputs += o.inputs;
        self.outputs_per_input = self.outputs_per_input.max(o.outputs_per_input);
        self.abs.merge(&o.abs);
        self.u.merge(&o.u);
        self.ulp.merge(&o.ulp);
        self.er.merge(&o.er);
        self.ec.merge(&o.ec);
        self.erc += o.erc;
        self.er_u.merge(&o.er_u);
        self.ec_u.merge(&o.ec_u);
        self.erc_u += o.erc_u;
        self.differ += o.differ;
        self.compared += o.compared;
        for (k, v) in o.pmf_ulp {
            *self.pmf_ulp.entry(k).or_default() += v;
        }
        self.off_lattice += o.off_lattice;
        self.ulp_tiny += o.ulp_tiny;
        self.wc_sum += o.wc_sum;
        self.wc_max = self.wc_max.max(o.wc_max);
        self.wc_n += o.wc_n;
        self.wc_tiny += o.wc_tiny;
        self.wc_sample.extend(o.wc_sample);
        self.wc_checked += o.wc_checked;
        self.wc_exceed += o.wc_exceed;
        self.wc_ratio = self.wc_ratio.max(o.wc_ratio);
        self.sample.extend(o.sample);
        self.pairs.extend(o.pairs);
        self.pair_at.extend(o.pair_at);
        self.input_means.extend(o.input_means);
        self.input_max_u.extend(o.input_max_u);
        self.input_differ.extend(o.input_differ);
        for (k, v) in o.ood {
            *self.ood.entry(k).or_default() += v;
        }
        self.ood_inputs += o.ood_inputs;
        self.skipped += o.skipped;
        self.nonfinite += o.nonfinite;
        self.nonfinite_inputs += o.nonfinite_inputs;
        self.seen += o.seen;
        if self.per_output.is_empty() {
            self.per_output = o.per_output;
        } else {
            for (a, b) in self.per_output.iter_mut().zip(&o.per_output) {
                a.merge(b);
            }
        }
    }
}

/// What one input contributes: the outputs of R (and C) as f32 values at the tracked outputs
/// (`Setup::tracked`, in that order), given the evaluated reference program state (for the
/// exact values).
pub struct InputEval<'a> {
    pub w: &'a [crate::mma_lp::Operand],
    pub r_out: &'a [f32],
    pub c_out: Option<&'a [f32]>,
}

/// Accumulate one input's outputs into `obs`. `keep`: positions in `Setup::tracked` to keep
/// in the subsample.
pub fn observe(
    st: &Setup,
    ev: &InputEval,
    obs: &mut Obs,
    keep: &[usize],
    per_output: bool,
    acc: &mut ExpAcc,
) {
    let n = st.r.outputs.len();
    let f = st.r.out_fmt;
    obs.inputs += 1;
    obs.outputs_per_input = n as u64;
    if per_output && obs.per_output.len() != n {
        obs.per_output = vec![Acc::default(); n];
    }
    let (mut ms, mut ma, mut cnt, mut mx) = (0.0, 0.0, 0u64, 0f64);
    let (d0, c0) = (obs.differ, obs.compared);
    let nf0 = obs.nonfinite;
    let mut ki = 0;
    // Agent AG: non-chain outputs take their exact value from the real-arithmetic evaluation.
    let mut real: Option<Vec<(super::nonlin::real::Dd, f64)>> = None;
    obs.skipped += (n - st.tracked.len()) as u64;
    for (k, &o32) in st.tracked.iter().enumerate() {
        let o = o32 as usize;
        let ex = match (&st.chain[k], st.r.outputs[o]) {
            (Some(ch), _) => st.r.exact(ch, ev.w, acc),
            (None, Some(op)) => {
                let t = real.get_or_insert_with(|| super::nonlin::real::eval(st.r, ev.w));
                super::nonlin::real::exact_of(t, op)
            }
            (None, None) => {
                obs.skipped += 1;
                continue;
            }
        };
        obs.seen += 1;
        let rv = ev.r_out[k] as f64;
        let cnf = ev.c_out.is_some_and(|c| !c[k].is_finite());
        if !ex.y.is_finite() || !ex.s.is_finite() || !rv.is_finite() || cnf {
            obs.nonfinite += 1;
            continue;
        }
        if ex.s == 0.0 {
            obs.skipped += 1;
            continue;
        }
        // Error of the reference against the exact value: (rv - y_hi) - y_lo, exactly enough
        // (rv and y agree to ~24 bits).
        let er = (rv - ex.y) - ex.y_lo;
        let us = U * ex.s;
        let ul = ulp_out(ex.y, f);
        let mut kref = None;
        let e = match ev.c_out {
            Some(c) => {
                let cv = c[k] as f64;
                let ec = (cv - ex.y) - ex.y_lo;
                obs.er.add(er);
                obs.ec.add(ec);
                obs.erc += er * ec;
                obs.er_u.add(er / us);
                obs.ec_u.add(ec / us);
                obs.erc_u += (er / us) * (ec / us);
                obs.differ += (c[k].to_bits() != ev.r_out[k].to_bits()) as u64;
                cv - rv
            }
            None => {
                let rn = round_out(ex.y, ex.y_lo, f);
                obs.differ += (rn != rv) as u64;
                // pmf_ulp axis in single mode: output - RN(y), in output ulps.
                kref = Some(((rv - rn) / ul).round());
                er
            }
        };
        obs.compared += 1;
        let (eu, el) = (e / us, e / ul);
        if *DEBUG {
            let b = exp_floor(ex.y);
            let mut g = DBG.lock().unwrap();
            let r = g.entry(b).or_insert([0.0; 4]);
            r[0] += 1.0;
            r[1] += (e == 0.0) as u8 as f64;
            r[2] += el.abs();
            r[3] += el * el;
        }
        obs.abs.add(e);
        obs.u.add(eu);
        if ex.y.abs() >= 2f64.powi(f.emin()) {
            obs.ulp.add(el);
        } else {
            obs.ulp_tiny += 1;
        }
        let kl = kref.unwrap_or(el);
        if kl.fract() == 0.0 && kl.abs() < 1e15 {
            *obs.pmf_ulp.entry(kl as i64).or_default() += 1;
        } else {
            obs.off_lattice += 1;
        }
        if per_output {
            obs.per_output[o].add(eu);
        }
        ms += eu;
        ma += eu.abs();
        mx = mx.max(eu.abs());
        cnt += 1;
        while ki < keep.len() && keep[ki] < k {
            ki += 1;
        }
        let kept = ki < keep.len() && keep[ki] == k;
        if kept {
            obs.sample.push((eu, el, e));
            let ey = match ev.c_out {
                Some(c) => (c[k] as f64 - ex.y) - ex.y_lo,
                None => er,
            };
            obs.pairs.push((ex.y + ex.y_lo, ey));
            obs.pair_at.push((o as u32, obs.cur));
        }
        if let Some((rel, k, ab, one)) = st.wc {
            let b = (rel * ex.y.abs() + k * if one { 1.0 } else { ex.s } + ab) * (1.0 + 1e-9);
            obs.wc_checked += 1;
            if e.abs() > b {
                obs.wc_exceed += 1;
            }
            if b > 0.0 {
                obs.wc_ratio = obs.wc_ratio.max(e.abs() / b);
            }
            if ex.y.abs() >= 2f64.powi(f.emin()) {
                let bu = b / ul;
                obs.wc_sum += bu;
                obs.wc_max = obs.wc_max.max(bu);
                obs.wc_n += 1;
                if kept {
                    obs.wc_sample.push(bu);
                }
            } else {
                obs.wc_tiny += 1;
            }
        }
    }
    if cnt > 0 {
        obs.input_means.push((ms / cnt as f64, ma / cnt as f64));
        obs.input_max_u.push(mx);
    }
    if obs.nonfinite > nf0 {
        obs.nonfinite_inputs += 1;
    }
    if obs.compared > c0 {
        obs.input_differ
            .push((obs.differ - d0) as f64 / (obs.compared - c0) as f64);
    }
}

/// Out-of-domain events of one evaluated input: subnormal accumulators or bf16 operands of
/// the calibrated MMA models, NaN / Inf values.
pub fn ood_events(p: &Prog, w: &[crate::mma_lp::Operand]) -> Vec<&'static str> {
    let mut ev = Vec::new();
    let sub32 = |x: f64| x != 0.0 && x.abs() < f32::MIN_POSITIVE as f64;
    for (i, op) in p.ops.iter().enumerate() {
        match op {
            Op::Mma { acc, terms, .. } => {
                if let Some(a) = acc {
                    if sub32(w[*a as usize].v) && !ev.contains(&"subnormal f32 accumulator") {
                        ev.push("subnormal f32 accumulator");
                    }
                }
                if !w[i].v.is_finite() && !ev.contains(&"non-finite MMA result") {
                    ev.push("non-finite MMA result");
                }
                for &(a, b) in terms {
                    for x in [a, b] {
                        if let Op::Leaf {
                            fmt: Format::Bf16 | Format::F16,
                            ..
                        } = p.ops[x as usize]
                        {
                            let v = w[x as usize].v;
                            if v != 0.0
                                && v.abs() < 2f64.powi(-126)
                                && !ev.contains(&"subnormal bf16 operand")
                            {
                                ev.push("subnormal bf16 operand");
                            }
                        }
                    }
                }
            }
            Op::Leaf { .. } if !w[i].v.is_finite() && !ev.contains(&"non-finite input") => {
                ev.push("non-finite input");
            }
            _ => {}
        }
    }
    ev
}

/// Monte Carlo configuration.
pub struct McCfg {
    pub n: usize,
    /// Sample indices start here (validation uses another range than the prediction).
    pub first: u64,
    pub threads: usize,
    /// Outputs per input kept for quantiles.
    pub keep: usize,
    pub per_output: bool,
}

/// A sampling plan for the programs' leaves and each program's leaf positions in it.
pub struct Sparse {
    pub plan: super::spec::Plan,
    pub pos_r: Vec<u32>,
    pub pos_c: Option<Vec<u32>>,
}

impl Sparse {
    /// Sample only what the leaves of R (and C) read (`spec::Plan`).
    pub fn new(st: &Setup, spec: &Spec, layouts: &[Layout]) -> Result<Sparse, String> {
        let mut reads = BTreeMap::new();
        st.r.leaf_reads(&mut reads);
        if let Some(c) = st.c {
            c.leaf_reads(&mut reads);
        }
        // PTXV_MC_DENSE: draw whole buffers (the planned and the whole sample agree on every
        // element the leaves read; this is the check).
        let plan = if std::env::var("PTXV_MC_DENSE").is_ok() {
            super::spec::Plan::default()
        } else {
            super::spec::Plan::new(spec, layouts, &reads)?
        };
        Ok(Sparse {
            pos_r: st.r.leaf_positions(&plan)?,
            pos_c: st.c.map(|c| c.leaf_positions(&plan)).transpose()?,
            plan,
        })
    }
}

/// Output values at the tracked outputs (`Setup::tracked`; untracked in `p`: 0).
pub fn tracked_outputs(p: &Prog, w: &[crate::mma_lp::Operand], tracked: &[u32]) -> Vec<f32> {
    tracked
        .iter()
        .map(|&o| p.outputs[o as usize].map_or(0.0, |i| w[i as usize].v as f32))
        .collect()
}

/// Bytes one Tier 3 worker holds while evaluating an input (the sample, the programs' op
/// states, the real-arithmetic evaluation of non-chain outputs), for the default thread
/// count.
pub fn bytes_per_worker(st: &Setup, sp: &Sparse, layouts: &[Layout]) -> u64 {
    let op = std::mem::size_of::<crate::mma_lp::Operand>() as u64;
    let ops = st.r.ops.len() as u64 + st.c.map_or(0, |c| c.ops.len() as u64);
    let sample: u64 = layouts
        .iter()
        .map(|l| match sp.plan.elems.get(&l.name) {
            Some(v) => 8 * v.len() as u64 + 4 * v.len() as u64,
            None => 8 * l.elems() as u64 + l.bytes as u64,
        })
        .sum();
    // Leaves + clean leaves + C, the real evaluation (Dd + f64 per op), outputs.
    2 * sample + ops * (2 * op + 24) + 8 * st.tracked.len() as u64 + (64 << 20)
}

/// Threads for Tier 3 within a memory budget: 70% of what is left (`memory_budget`).
pub fn threads_for_memory(per_worker: u64, want: usize) -> usize {
    let n = (memory_budget() / 10 * 7 / per_worker.max(1)).max(1) as usize;
    n.min(want.max(1))
}

/// Bytes this process may still allocate: the tightest cgroup on its path (memory.max less
/// memory.current; systemd-run MemoryMax), else MemAvailable.
pub fn memory_budget() -> u64 {
    let mut lim = u64::MAX;
    let num = |p: &std::path::Path| -> Option<u64> {
        std::fs::read_to_string(p).ok()?.trim().parse::<u64>().ok()
    };
    if let Ok(cg) = std::fs::read_to_string("/proc/self/cgroup") {
        for line in cg.lines() {
            if let Some(path) = line.strip_prefix("0::") {
                let mut p = std::path::PathBuf::from("/sys/fs/cgroup");
                p.push(path.trim_start_matches('/'));
                loop {
                    if let (Some(max), Some(cur)) =
                        (num(&p.join("memory.max")), num(&p.join("memory.current")))
                    {
                        lim = lim.min(max.saturating_sub(cur));
                    }
                    if !p.pop() || p == std::path::Path::new("/sys/fs/cgroup") {
                        break;
                    }
                }
            }
        }
    }
    let avail = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| {
            m.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .map_or(u64::MAX, |kb| kb * 1024);
    lim.min(avail)
}

/// Tier 3: evaluate both programs on `cfg.n` inputs.
pub fn run(st: &Setup, spec: &Spec, layouts: &[Layout], cfg: &McCfg) -> Result<Obs, String> {
    let sp = Sparse::new(st, spec, layouts)?;
    let next = AtomicU64::new(0);
    let out = Mutex::new(Obs::default());
    let err: Mutex<Option<String>> = Mutex::new(None);
    let threads = cfg.threads.max(1).min(cfg.n.max(1));
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut obs = Obs::default();
                let mut acc = ExpAcc::default();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= cfg.n as u64 || err.lock().unwrap().is_some() {
                        break;
                    }
                    let t = cfg.first + i;
                    match one(st, &sp, spec, layouts, t, cfg, &mut obs, &mut acc) {
                        Ok(()) => {}
                        Err(e) => {
                            *err.lock().unwrap() = Some(e);
                            break;
                        }
                    }
                }
                out.lock().unwrap().merge(obs);
            });
        }
    });
    if let Some(e) = err.into_inner().unwrap() {
        return Err(e);
    }
    Ok(out.into_inner().unwrap())
}

/// The outputs kept in input t's subsample.
pub fn keep_set(n: usize, keep: usize, t: u64) -> Vec<usize> {
    if keep >= n {
        return (0..n).collect();
    }
    let mut r = Rng::new(t ^ 0x5eed_0f00);
    let mut v: Vec<usize> = (0..keep).map(|_| r.below(n as u64) as usize).collect();
    v.sort();
    v.dedup();
    v
}

#[allow(clippy::too_many_arguments)]
fn one(
    st: &Setup,
    sp: &Sparse,
    spec: &Spec,
    layouts: &[Layout],
    t: u64,
    cfg: &McCfg,
    obs: &mut Obs,
    acc: &mut ExpAcc,
) -> Result<(), String> {
    let smp = super::spec::sample_with(spec, layouts, t, Some(&sp.plan))?;
    let mut causes: Vec<String> = Vec::new();
    if smp.ood.overflow > 0 {
        causes.push("input overflow (saturated or infinite)".into());
    }
    if smp.ood.nonfinite > 0 {
        causes.push("non-finite input".into());
    }
    if smp.ood.bad_scale > 0 {
        causes.push("zero ue4m3 scale".into());
    }
    let mut wr = st.r.leaves_at(&smp.bytes, &sp.pos_r)?;
    st.r.eval_into(&mut wr);
    let r_out = tracked_outputs(st.r, &wr, &st.tracked);
    for e in ood_events(st.r, &wr) {
        causes.push(e.to_string());
    }
    // Input errors: the exact reference is the program on the error-free values.
    let wx = if smp.clean.is_empty() {
        None
    } else {
        let mut w = st.r.leaves_at(&smp.clean_bytes(), &sp.pos_r)?;
        st.r.eval_into(&mut w);
        Some(w)
    };
    let c_vals = match st.c {
        Some(c) => {
            let mut wc = c.leaves_at(&smp.bytes, sp.pos_c.as_deref().unwrap_or(&[]))?;
            c.eval_into(&mut wc);
            for e in ood_events(c, &wc) {
                causes.push(e.to_string());
            }
            Some(tracked_outputs(c, &wc, &st.tracked))
        }
        None => None,
    };
    causes.sort();
    causes.dedup();
    let causes_empty = causes.is_empty();
    if !causes.is_empty() {
        obs.ood_inputs += 1;
        for c in causes {
            *obs.ood.entry(c).or_default() += 1;
        }
    }
    let keep = keep_set(st.tracked.len(), cfg.keep, t);
    let nf0 = obs.nonfinite;
    obs.cur = t;
    observe(
        st,
        &InputEval {
            w: wx.as_deref().unwrap_or(&wr),
            r_out: &r_out,
            c_out: c_vals.as_deref(),
        },
        obs,
        &keep,
        cfg.per_output,
        acc,
    );
    // NaN / Inf outputs: out of domain (counted, never dropped silently).
    if obs.nonfinite > nf0 {
        obs.ood_inputs += causes_empty as u64;
        *obs.ood
            .entry("non-finite output (NaN / Inf)".to_string())
            .or_default() += 1;
    }
    Ok(())
}

/// Output values of an evaluated program (untracked outputs: 0).
pub fn outputs(p: &Prog, w: &[crate::mma_lp::Operand]) -> Vec<f32> {
    p.outputs
        .iter()
        .map(|o| o.map_or(0.0, |i| w[i as usize].v as f32))
        .collect()
}

/// The interval of P(differ) at level 1 - alpha. Inputs are the independent unit: outputs of
/// one input share its elements (a GEMM row, an MX block's scale), so the interval is the
/// wider of the binomial (Clopper-Pearson) interval over outputs and the batch-means interval
/// over inputs (Student t over the per-input fractions). Outputs alone were ~3x too narrow
/// on MXF8 (seed-to-seed spread +-0.0002 against +-0.00007).
// ASSUME(N-ERRDIST-MC): inputs are independent.
pub fn p_interval(obs: &Obs, alpha: f64) -> (f64, f64) {
    let (cl, ch) = super::special::clopper_pearson(obs.differ, obs.compared, alpha);
    let f = &obs.input_differ;
    let n = f.len();
    if n < 2 {
        return (cl, ch);
    }
    let p = obs.differ as f64 / obs.compared.max(1) as f64;
    let m = f.iter().sum::<f64>() / n as f64;
    let v = f.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64;
    let hw = t_quantile(1.0 - alpha / 2.0, (n - 1) as f64) * (v / n as f64).sqrt();
    ((p - hw).max(0.0).min(cl), (p + hw).min(1.0).max(ch))
}

/// The q quantile of Student's t with `df` degrees of freedom (bisection).
pub fn t_quantile(q: f64, df: f64) -> f64 {
    let (mut lo, mut hi) = (0.0f64, 1e3f64);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if super::special::student_t_cdf(mid, df) < q {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// A Tier 3 (or observed) estimate from collected errors.
// ASSUME(N-ERRDIST-MC): output-level intervals treat outputs as independent.
pub fn estimate(obs: &Obs, tier: u8, method: &str, out_fmt: Format) -> super::report::Estimate {
    use super::report::*;
    use super::stats::*;
    let mut su: Vec<f64> = obs.sample.iter().map(|x| x.0).collect();
    let mut sl: Vec<f64> = obs.sample.iter().map(|x| x.1).collect();
    let mut sa: Vec<f64> = obs.sample.iter().map(|x| x.2).collect();
    sort(&mut su);
    sort(&mut sl);
    sort(&mut sa);
    let n = obs.inputs.max(1) as f64;
    // Batch means over inputs (independent): CI of the mean.
    let bm: Vec<f64> = obs.input_means.iter().map(|x| x.0).collect();
    let bmean = bm.iter().sum::<f64>() / bm.len().max(1) as f64;
    let bvar = bm.iter().map(|x| (x - bmean).powi(2)).sum::<f64>() / (bm.len().max(2) - 1) as f64;
    let mean_hw_u = 1.96 * (bvar / bm.len().max(1) as f64).sqrt();
    let mut eb = ErrorBar {
        kind: "dkw".into(),
        cdf_sup: Some(dkw(n, 0.05)),
        mean_hw: Some(mean_hw_u),
        std_rel_hw: Some(1.96 / (2.0 * n).sqrt()),
        model_mass: None,
        notes: vec![
            format!(
                "DKW band at 95% with n = {} independent inputs (outputs within an input are dependent)",
                obs.inputs
            ),
            "mean_hw is in the u-scale unit (batch means over inputs)".into(),
        ],
    };
    if obs.skipped > 0 {
        eb.notes.push(format!(
            "{} output-inputs skipped (no nonzero term, or untracked)",
            obs.skipped
        ));
    }
    if obs.ulp_tiny > 0 {
        eb.notes.push(format!(
            "ulp-scale mean / std / mean|.| leave out {} outputs whose exact value is 0 or subnormal in the output format (P(differ) and the PMF keep them)",
            obs.ulp_tiny
        ));
    }
    let lattice = obs.off_lattice == 0 && !obs.pmf_ulp.is_empty();
    let tot = obs.compared.max(1) as f64;
    let pmf_ulp: Vec<(i64, f64)> = if lattice || out_fmt != Format::F32 {
        obs.pmf_ulp
            .iter()
            .map(|(k, v)| (*k, *v as f64 / tot))
            .collect()
    } else {
        vec![]
    };
    let (lo, hi) = p_interval(obs, 0.05);
    // Histogram of the u-scale error over the central 99.98%.
    let hist_u = if su.len() > 10 {
        let (a, b) = (quantile(&su, 1e-4), quantile(&su, 1.0 - 1e-4));
        if b > a {
            let k = 60;
            let edges: Vec<f64> = (0..=k).map(|i| a + (b - a) * i as f64 / k as f64).collect();
            let mut c = vec![0f64; k];
            for x in &su {
                let i = (((x - a) / (b - a)) * k as f64).floor();
                if i >= 0.0 && (i as usize) < k {
                    c[i as usize] += 1.0;
                }
            }
            let t = su.len() as f64;
            Some((edges, c.into_iter().map(|x| x / t).collect()))
        } else {
            None
        }
    } else {
        None
    };
    let pair = (obs.er.n > 0).then(|| {
        let m = obs.er.n as f64;
        let cov = obs.erc / m - obs.er.mean() * obs.ec.mean();
        PairMoments {
            mean_r: obs.er.mean(),
            var_r: obs.er.var(),
            mean_c: obs.ec.mean(),
            var_c: obs.ec.var(),
            cov_rc: cov,
            corr_rc: cov / (obs.er.var() * obs.ec.var()).sqrt().max(1e-300),
        }
    });
    Estimate {
        tier,
        method: method.into(),
        n: obs.inputs,
        outputs: obs.outputs_per_input,
        abs: obs.abs.moments(&sa),
        u_scale: Some(obs.u.moments(&su)),
        ulp: Some(obs.ulp.moments(&sl)),
        pmf_ulp,
        hist_u,
        error_bar: eb,
        pair,
        p_differ: Some((obs.differ as f64 / tot, lo, hi)),
        seconds: 0.0,
        ..Default::default()
    }
}

/// Diagnostics: per kernel and step position, the local errors' statistics in units of the
/// step result's f32 ulp; for pairs, how often the two kernels' errors on the same terms are
/// equal. Printed by `ptxv errdist --explore`.
pub fn explore_steps(
    st: &Setup,
    spec: &Spec,
    layouts: &[Layout],
    n: usize,
) -> Result<String, String> {
    use std::fmt::Write;
    let mut out = String::new();
    let kernels: Vec<&Prog> = std::iter::once(st.r).chain(st.c).collect();
    // Per kernel, per step position: (count, nonzero, sum e/ulp, sum (e/ulp)^2, rho nonzero,
    // tau nonzero, acc truncated).
    let mut tab: Vec<Vec<[f64; 8]>> = vec![vec![]; kernels.len()];
    let mut same = [0f64; 2];
    // Kernel 0, per (step, bS - bX, bS - E): (count, nonzero local errors).
    let mut ctab: BTreeMap<(usize, i32, i32), (f64, f64)> = BTreeMap::new();
    let mut acc = ExpAcc::default();
    for t in 0..n as u64 {
        let smp = super::spec::sample(spec, layouts, t)?;
        let mut per_k: Vec<Vec<BTreeMap<Vec<u32>, f64>>> = vec![];
        for (ki, p) in kernels.iter().enumerate() {
            let mut w = p.leaves(&smp.bytes)?;
            p.eval_into(&mut w);
            let mut errs: Vec<BTreeMap<Vec<u32>, f64>> = vec![];
            for o in 0..p.outputs.len().min(512) {
                let Some((steps, _)) = p.chain(o) else {
                    continue;
                };
                let mut m = BTreeMap::new();
                let mut exs = ExpAcc::default();
                for (si, &s) in steps.iter().enumerate() {
                    let Op::Mma {
                        acc: a,
                        terms,
                        scales,
                        ..
                    } = &p.ops[s as usize]
                    else {
                        continue;
                    };
                    acc.clear();
                    let av = a.map_or(0.0, |a| w[a as usize].v);
                    acc.add(av);
                    let mut key = Vec::new();
                    let mut dropped = false;
                    let mut emax = if av != 0.0 { exp_floor(av) } else { i32::MIN };
                    for (j, &(x, y)) in terms.iter().enumerate() {
                        let mut v = w[x as usize].v * w[y as usize].v;
                        let mut o = w[x as usize].mul(&w[y as usize]);
                        if let Some(&(sa, sb)) = scales.get(j) {
                            v *= w[sa as usize].v * w[sb as usize].v;
                            o = o.mul(&w[sa as usize]).mul(&w[sb as usize]);
                        }
                        if v != 0.0 {
                            emax = emax.max(o.ef);
                        }
                        acc.add(v);
                        exs.add(v);
                        key.push(x.min(y));
                        let _ = &mut dropped;
                    }
                    let hw = w[s as usize].v;
                    if ki == 0 && hw != 0.0 {
                        let mut d = exs.clone();
                        d.add(-hw);
                        let lb = ((hw.abs() / ulp_out(hw, Format::F32)) as u64 & 3) as i32;
                        let c = ctab.entry((si, 800 + lb, 0)).or_default();
                        c.0 += 1.0;
                        c.1 += (!d.is_zero()) as u8 as f64;
                    }
                    acc.add(-hw);
                    let e = -acc.approx();
                    let ul = ulp_out(hw, Format::F32);
                    if ki == 0
                        && hw != 0.0
                        && e != 0.0
                        && si == 0
                        && std::env::var("PTXV_EXPLORE_DUMP").is_ok()
                    {
                        let mut fine = vec![];
                        for (j, &(x, y)) in terms.iter().enumerate() {
                            let mut v = w[x as usize].v * w[y as usize].v;
                            if let Some(&(sa, sb)) = scales.get(j) {
                                v *= w[sa as usize].v * w[sb as usize].v;
                            }
                            if v != 0.0 && (v / ul).fract() != 0.0 {
                                fine.push(format!("{:.4}", v / ul));
                            }
                        }
                        eprintln!(
                            "dump e/ulp {:.4} hw/ulp {:.1} emax {emax} fine [{}]",
                            e / ul,
                            hw / ul,
                            fine.join(" ")
                        );
                    }
                    if ki == 0 && hw != 0.0 && e != 0.0 {
                        let c = ctab.entry((si, 999, 0)).or_default();
                        c.0 += 1.0;
                        c.1 += if (e / ul).abs() > 0.5 {
                            1.0
                        } else if (e / ul).abs() == 0.5 {
                            0.5
                        } else {
                            0.0
                        };
                    }
                    if ki == 0 && si < 2 && emax > i32::MIN {
                        let c = ctab.entry((si, 998, emax)).or_default();
                        c.0 += 1.0;
                        let c = ctab
                            .entry((si, 997, if hw != 0.0 { exp_floor(hw) } else { -999 }))
                            .or_default();
                        c.0 += 1.0;
                    }
                    if ki == 0 && hw != 0.0 && emax > i32::MIN {
                        let bx = if av != 0.0 {
                            exp_floor(hw) - exp_floor(av)
                        } else {
                            -999
                        };
                        let c = ctab.entry((si, bx, exp_floor(hw) - emax)).or_default();
                        c.0 += 1.0;
                        c.1 += (e != 0.0) as u8 as f64;
                        if bx == 1 && exp_floor(hw) - emax == 1 && si >= 2 {
                            // acc lsb, fine terms (bits below ulp_S), e != 0
                            let lsb = ((av.abs() / ulp_out(av, Format::F32)) as u64 & 1) as usize;
                            let mut fine = 0usize;
                            for (j, &(x, y)) in terms.iter().enumerate() {
                                let mut v = w[x as usize].v * w[y as usize].v;
                                if let Some(&(sa, sb)) = scales.get(j) {
                                    v *= w[sa as usize].v * w[sb as usize].v;
                                }
                                if v != 0.0 && (v / ul).fract() != 0.0 {
                                    fine = 1;
                                }
                            }
                            let c = ctab
                                .entry((si, 900 + 4 * lsb as i32 + 2 * fine as i32, 0))
                                .or_default();
                            c.0 += 1.0;
                            c.1 += (e != 0.0) as u8 as f64;
                        }
                    }
                    if tab[ki].len() <= si {
                        tab[ki].resize(si + 1, [0.0; 8]);
                    }
                    let r = &mut tab[ki][si];
                    r[0] += 1.0;
                    r[1] += (e != 0.0) as u8 as f64;
                    r[2] += e / ul;
                    r[3] += (e / ul).powi(2);
                    r[4] += (e / ul).abs();
                    key.sort();
                    // Leaf op ids differ between programs: key by the leaves' buffer/index.
                    let k2: Vec<u32> = terms
                        .iter()
                        .map(|&(x, _)| match &p.ops[x as usize] {
                            Op::Leaf { idx, .. } => *idx as u32,
                            _ => u32::MAX,
                        })
                        .collect::<std::collections::BTreeSet<u32>>()
                        .into_iter()
                        .collect();
                    m.insert(k2, e);
                }
                errs.push(m);
            }
            per_k.push(errs);
        }
        if per_k.len() == 2 {
            for (a, b) in per_k[0].iter().zip(&per_k[1]) {
                for (k, e) in a {
                    if let Some(f) = b.get(k) {
                        same[0] += 1.0;
                        same[1] += (e == f) as u8 as f64;
                    }
                }
            }
        }
    }
    for (ki, rows) in tab.iter().enumerate() {
        let _ = writeln!(
            out,
            "kernel {ki}: step  P(e != 0)  mean e/ulp  sd e/ulp  mean |e|/ulp"
        );
        for (si, r) in rows.iter().enumerate() {
            let c = r[0].max(1.0);
            let m = r[2] / c;
            let _ = writeln!(
                out,
                "    {si:>4}  {:>9.4}  {:>10.4}  {:>8.4}  {:>8.4}",
                r[1] / c,
                m,
                (r[3] / c - m * m).max(0.0).sqrt(),
                r[4] / c
            );
        }
    }
    let mut per_step: BTreeMap<usize, f64> = BTreeMap::new();
    for ((si, a, _), v) in &ctab {
        if *a < 797 {
            *per_step.entry(*si).or_default() += v.0;
        }
    }
    for ((si, a, b), v) in &ctab {
        if *a == 999 {
            let _ = writeln!(
                out,
                "  mc step {si}: P(|e| > ulp/2 | e != 0) {:.3} (n {})",
                v.1 / v.0,
                v.0
            );
        }
        if (800..804).contains(a) {
            let n: f64 = (800..804)
                .map(|k| ctab.get(&(*si, k, 0)).map_or(0.0, |x| x.0))
                .sum();
            let _ = writeln!(
                out,
                "  mc step {si} result low bits {}: P {:.4} P(delta != 0 | bits) {:.4}",
                a - 800,
                v.0 / n,
                v.1 / v.0
            );
        }
        if (900..908).contains(a) {
            let _ = writeln!(
                out,
                "  mc step {si} cell(1,1) lsb {} fine {}: n {} P(e!=0) {:.3}",
                (a - 900) / 4,
                ((a - 900) / 2) % 2,
                v.0,
                v.1 / v.0
            );
        }
        if *a == 998 || *a == 997 {
            let _ = writeln!(
                out,
                "  mc step {si} {} {:>4}: {:.4}",
                if *a == 998 { "E" } else { "bS" },
                b,
                v.0 / per_step[si]
            );
        }
    }
    for ((si, a, b), v) in &ctab {
        if *a < 797 && v.0 > 0.01 * per_step[si] {
            let _ = writeln!(
                out,
                "  mc step {si} bS-bX {a:>4} bS-E {b:>3}: w {:.3} P(e!=0) {:.3}",
                v.0 / per_step[si],
                v.1 / v.0
            );
        }
    }
    if same[0] > 0.0 {
        let _ = writeln!(
            out,
            "same-term steps with equal local errors in both kernels: {:.4}",
            same[1] / same[0]
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p_interval_counts_inputs_not_outputs() {
        // 100 inputs of 1000 outputs: half the inputs differ everywhere, half nowhere. The
        // outputs' binomial interval is +-0.03; the inputs carry the information: +-0.1.
        let mut o = Obs {
            differ: 50_000,
            compared: 100_000,
            ..Default::default()
        };
        o.input_differ = (0..100).map(|i| (i % 2) as f64).collect();
        let (lo, hi) = p_interval(&o, 0.05);
        assert!(lo < 0.41 && hi > 0.59, "[{lo}, {hi}]");
        // Independent outputs: the binomial interval stands.
        let mut o = Obs {
            differ: 50_000,
            compared: 100_000,
            ..Default::default()
        };
        o.input_differ = vec![0.5; 100];
        let (lo, hi) = p_interval(&o, 0.05);
        let (cl, ch) = super::super::special::clopper_pearson(50_000, 100_000, 0.05);
        assert_eq!((lo, hi), (cl, ch));
        assert!((t_quantile(0.975, 1e6) - 1.96).abs() < 1e-3);
    }
}

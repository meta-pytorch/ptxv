/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv equiv <file.toml>`: is a candidate kernel equivalent to a reference (rung 2e)?
//!
//! ```toml
//! tau_k = 64                 # optional: L3 tolerance in units of u = 2^-24
//! max_states = 2000000       # race / deadlock-freedom check of each kernel
//! reference = { case = "../r1a_tcgen05_tile", variant = "good", front = "sass" }
//! [[candidates]]
//! case = "../r1a_tcgen05_tile"
//! variant = "E1"
//! front = "sass"
//! expect = "L1"              # optional: the strongest level that should hold ("none": not L1;
//!                            # " (unproven)" appended when a precondition fails)
//! expect_bound_u = 254       # optional: the largest per-output bound must be at most this
//! ```
//! `trials = N` (or `--trials N`): N random inputs (alternately normal and uniform in [1, 2),
//! the same-sign case where rounding errors accumulate), each kernel run on a random schedule;
//! every output's |C - R| must be within its proven bound at that input, and the report gives
//! how much of the bound was used. The JSON summary goes to `<out>/equiv.json`.
//!
//! Each kernel is checked for the preconditions (exhaustive `check`: no race, deadlock or
//! violation; the taint check: control flow and addresses independent of input data), run
//! once with the numerics shadow, and compared element by element (`ptxv_sim::equiv`). When
//! L1 fails, a witness input isolates one differing monomial (every input zero except its
//! factors, set to 1), and both kernels are run on it in the simulator and on GB300.

use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use anyhow::bail;
use ptxv_case::Case;
use ptxv_sim::equiv::OutBound;
use ptxv_sim::equiv::Report;
use ptxv_sim::equiv::Side;
use ptxv_sim::exec::Loc;
use ptxv_sim::explore::explore;
use ptxv_sim::explore::run_once;
use ptxv_sim::machine::Machine;
use ptxv_sim::num::NumState;
use ptxv_sim::num::U32;
use serde::Deserialize;
use serde::Serialize;

use super::gpu_env;
use super::load_with;
use super::machine;

#[derive(Deserialize, Clone)]
pub(crate) struct KernelRef {
    pub(crate) case: PathBuf,
    pub(crate) variant: String,
    #[serde(default = "sass")]
    pub(crate) front: String,
    #[serde(default)]
    pub(crate) expect: Option<String>,
    /// Check the preconditions one cluster at a time (`ptxv check --slices`): for large
    /// launches of independent clusters; `max_states` is then per cluster.
    #[serde(default)]
    pub(crate) slices: bool,
    /// Try the E-WELLSYNC certificate before exploring (`ptxv check --certify`).
    #[serde(default)]
    pub(crate) certify: bool,
    /// The largest per-output bound (units of u) must not exceed this.
    #[serde(default)]
    pub(crate) expect_bound_u: Option<f64>,
}

/// An output that is not also an input: every output except one updated in place (its
/// buffer starts from captured bytes, `init = "file"`: an Inductor `in_out_ptr`, `index_add_`),
/// whose initial elements are inputs like any other (leaves, tainted, randomized in trials).
pub(crate) fn out_only(case: &Case, name: &str) -> bool {
    case.m.outputs.iter().any(|o| o == name)
        && !case
            .m
            .buffers
            .iter()
            .any(|b| b.name == name && b.init == "file")
}

fn sass() -> String {
    "sass".into()
}

#[derive(Deserialize)]
pub(crate) struct EquivFile {
    pub(crate) reference: KernelRef,
    pub(crate) candidates: Vec<KernelRef>,
    #[serde(default)]
    pub(crate) tau_k: Option<f64>,
    #[serde(default = "default_states")]
    pub(crate) max_states: usize,
    /// Skip the GB300 replay of witnesses.
    #[serde(default)]
    pub(crate) no_hw: bool,
    /// Random-input trials checking each proven bound empirically.
    #[serde(default)]
    pub(crate) trials: usize,
    /// Segment mode (`NumState::segments`): products, maxima and instruction slices get
    /// nodes, for kernels beyond sums (softmax).
    #[serde(default)]
    pub(crate) segments: bool,
    /// File (relative to the equiv.toml) caching the exhaustive slice sweeps
    /// (`softmax::load_sweeps`).
    #[serde(default)]
    pub(crate) sweep_cache: Option<PathBuf>,
    /// Elementwise mode (docs/equiv-elementwise.md): every output a function of one input
    /// element, swept over every input value. None: tried whenever segment mode is on;
    /// true: also turns segment mode on.
    #[serde(default)]
    pub(crate) elementwise: Option<bool>,
    /// The real function an elementwise pair computes (`elementwise::TRUE_FUNCS`); default:
    /// identified from the kernels.
    #[serde(default)]
    pub(crate) function: Option<String>,
    /// Elementwise mode: also run each kernel on GB300 for this many batches of input values
    /// (all bf16 bit patterns first, widened for f32 inputs; then random patterns) and check
    /// every output against its element function, bitwise.
    #[serde(default)]
    pub(crate) elementwise_gpu: usize,
}

fn default_states() -> usize {
    2_000_000
}

/// One kernel, loaded, with its precondition verdicts and numerics run.
pub(crate) struct Run {
    pub(crate) name: String,
    pub(crate) case: Case,
    pub(crate) variant: String,
    pub(crate) front: String,
    pub(crate) cubin: PathBuf,
    pub(crate) kern: ptxv_ir::Kernel,
    pub(crate) m: Machine,
    pub(crate) outs: Vec<(u32, Option<u32>)>,
    pub(crate) pre: Vec<String>,
    /// Assumptions beyond the usual ones that the preconditions rely on.
    pub(crate) notes: Vec<String>,
    /// errdist with non-finite captured inputs replaced (N-ERRDIST-FINCAP): the simulator's
    /// first output at the unmodified captured input, for the comparison with the GPU's.
    pub(crate) captured_out: Option<Vec<f32>>,
}

impl Run {
    pub(crate) fn label(&self) -> String {
        format!("{} {} {}", self.name, self.variant, self.front)
    }
}

pub(crate) fn prepare(
    base: &Path,
    k: &KernelRef,
    max_states: usize,
    dir: &Path,
    segments: bool,
) -> Result<Run> {
    let case = Case::load(base.join(&k.case))?;
    if case.m.outputs.is_empty() {
        bail!(
            "equivalence requires an output buffer; {} declares no outputs",
            case.m.name
        );
    }
    prepare_with(base, k, max_states, dir, segments, false)
}

/// Integer buffers (i8 / i16 / i32 / i64 / u32 / u64: lengths, indices, packed integers) are
/// never errdist leaves: Tier 3 holds them at their captured bytes.
pub(crate) fn int_buffer(case: &Case, name: &str) -> bool {
    case.m.buffers.iter().any(|b| {
        b.name == name
            && matches!(
                b.dtype.as_str(),
                "i8" | "i16" | "i32" | "i64" | "u32" | "u64"
            )
    })
}

/// errdist's extraction run: every non-finite element of an f32 / bf16 / f16 input buffer
/// becomes +0 (errdist's input distributions are finite, so a captured infinity or NaN would
/// put the run on a path no sample takes). Returns the number of elements changed.
// ASSUME(N-ERRDIST-FINCAP): see ASSUMPTIONS.md
fn finite_inputs(case: &Case, launch: &mut ptxv_sim::machine::Launch) -> usize {
    let mut n = 0;
    let outputs = launch.outputs.clone();
    for (i, a) in launch.allocs.iter_mut().enumerate() {
        if outputs.contains(&i) && out_only(case, &a.name) {
            continue;
        }
        let Some(b) = case.m.buffers.iter().find(|b| b.name == a.name) else {
            continue;
        };
        match b.dtype.as_str() {
            "f32" => {
                for w in a.init.chunks_exact_mut(4) {
                    let x = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
                    if x & 0x7f80_0000 == 0x7f80_0000 {
                        w.fill(0);
                        n += 1;
                    }
                }
            }
            "bf16" | "f16" => {
                let e = if b.dtype == "bf16" { 0x7f80u16 } else { 0x7c00 };
                for w in a.init.chunks_exact_mut(2) {
                    if u16::from_le_bytes([w[0], w[1]]) & e == e {
                        w.fill(0);
                        n += 1;
                    }
                }
            }
            _ => {}
        }
    }
    n
}

/// `prepare`; `fixed_ints` (errdist): integer input buffers are not taint sources, as errdist
/// holds them fixed at their captured values (control on a sequence length is then fixed).
// ASSUME(N-ERRDIST-INTFIX): see ASSUMPTIONS.md
pub(crate) fn prepare_with(
    base: &Path,
    k: &KernelRef,
    max_states: usize,
    dir: &Path,
    segments: bool,
    fixed_ints: bool,
) -> Result<Run> {
    let case = Case::load(base.join(&k.case))?;
    let (kern, cubin) = load_with(&case, &k.variant, &k.front, dir, &[])?;
    // Both kernels' binaries land in `dir` under the variant's name: keep this one apart (the
    // GB300 replays run it later).
    let own = dir.join(format!("{}.{}.cubin", case.m.name, k.variant));
    let cubin = if std::fs::copy(&cubin, &own).is_ok() {
        own
    } else {
        cubin
    };
    let mut pre = Vec::new();
    let mut notes = Vec::new();
    // Precondition 1: race-free and deadlock-free (one output per input).
    let v = if k.slices {
        let make = || machine(&case, kern.clone()).expect("machine");
        ptxv_sim::explore::explore_symmetric(&make, max_states, &mut |_, _| {})
    } else if k.certify {
        // `ptxv check --certify` may have run this check already (vcache.rs).
        let key = case
            .variant(&k.variant)?
            .after
            .is_empty()
            .then(|| crate::vcache::key(&case.dir, &k.variant, &k.front, "certify"))
            .flatten();
        match key
            .as_deref()
            .and_then(|key| crate::vcache::load(key, max_states))
        {
            Some(v) => {
                notes.push(format!(
                    "execution check: the verdict of an earlier `ptxv check --certify` of the same case, variant and binary ({} states{})",
                    v.states,
                    if v.complete { ", complete" } else { "" }
                ));
                v
            }
            None => {
                let v =
                    ptxv_sim::wellsync::check_certified(machine(&case, kern.clone())?, max_states);
                if let Some(key) = &key {
                    crate::vcache::store(key, max_states, &v);
                }
                v
            }
        }
    } else {
        explore(machine(&case, kern.clone())?, max_states)
    };
    // `contract` findings (PTX-contract deviations GB300 tolerates, ASSUMPTIONS.md) do not
    // break the preconditions; they are listed as what the verdict relies on.
    notes.extend(v.notes.iter().cloned());
    let mut classes = v.classes();
    let nondet = classes.iter().any(|c| c == "nondeterministic");
    // ASSUME(E-DESC-VERSION): a tcgen05.mma descriptor with bits 46-48 = 0 is a `contract`
    // finding (GB300 ignores the bits, probes/desc_version), listed here as relied on.
    if classes.iter().any(|c| c == "contract") {
        for f in v.findings.iter().filter(|f| f.kind == "contract") {
            notes.push(format!("relies on: {}", f.msg));
        }
        classes.retain(|c| c != "contract");
        if classes.is_empty() {
            classes.push("clean".into());
        }
    }
    // The numerics run, with the taint shadow (precondition 2).
    let mut captured_out = None;
    let mut m = if fixed_ints {
        let mut launch = ptxv_sim::build_launch(&case, &kern)?;
        let n = finite_inputs(&case, &mut launch);
        if n > 0 {
            let (m0, _, ok) = run_once(machine(&case, kern.clone())?);
            if ok {
                captured_out = Some(out_f32(&m0, &case));
            }
            notes.push(format!(
                "relies on: N-ERRDIST-FINCAP ({n} non-finite captured float input elements set to 0 for the extraction run)"
            ));
        }
        Machine::new(
            std::sync::Arc::new(kern.clone()),
            std::sync::Arc::new(launch),
        )
    } else {
        machine(&case, kern.clone())?
    };
    let mut ns = NumState::with_mma_eps(m.launch.mma_model.eps());
    ns.segments = segments;
    for &o in &m.launch.outputs {
        if out_only(&case, &m.launch.allocs[o].name) {
            ns.not_inputs.insert(o);
        }
    }
    for (i, a) in m.launch.allocs.iter().enumerate() {
        let f32_buf = case
            .m
            .buffers
            .iter()
            .any(|b| b.name == a.name && b.dtype == "f32");
        if f32_buf && !out_only(&case, &a.name) {
            ns.f32_inputs.insert(i);
        }
        // bf16 / f16 inputs: elements widened to f32 are leaves too (halfin.rs).
        // ASSUME(N-HALF-LEAF)
        if !out_only(&case, &a.name) {
            let half = case
                .m
                .buffers
                .iter()
                .find(|b| b.name == a.name)
                .and_then(|b| match b.dtype.as_str() {
                    "bf16" => Some(ptxv_sim::halfin::Half::Bf16),
                    "f16" => Some(ptxv_sim::halfin::Half::F16),
                    _ => None,
                });
            if let Some(h) = half {
                ns.half.inputs.insert(i, h);
            }
            // Byte buffers: FP8 operands and scale factors of the narrow tensor-core kinds,
            // leaves of the format a step reads them as (halfin `byte_leaf`).
            if case
                .m
                .buffers
                .iter()
                .any(|b| b.name == a.name && matches!(b.dtype.as_str(), "u8" | "u16" | "e4m3"))
            {
                ns.half.inputs.insert(i, ptxv_sim::halfin::Half::Byte);
            }
        }
    }
    // 16-bit provenance (bf16 rounding nodes, tensor-core operands, bf16 outputs).
    // ASSUME(N-HALF-LEAF, N-ROUND)
    ns.half.track = true;
    for &o in &m.launch.outputs {
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
            }
            Some("f16") => {
                ns.half.f16_outputs.insert(o);
            }
            _ => {}
        }
    }
    let output = m.launch.outputs[0];
    let stride = if ns.half.bf16_outputs.contains(&output) || ns.half.f16_outputs.contains(&output)
    {
        2
    } else {
        4
    };
    let output_bytes = m.gmem[output].bytes.len();
    if output_bytes == 0 || output_bytes % stride != 0 {
        bail!(
            "equivalence cannot cover all {output_bytes} bytes of output {}: a nonempty buffer of whole {stride}-byte numeric elements is required",
            m.launch.allocs[output].name
        );
    }
    let mut fixed = vec![];
    for (i, a) in m.launch.allocs.iter().enumerate() {
        if fixed_ints && !out_only(&case, &a.name) && int_buffer(&case, &a.name) {
            fixed.push(a.name.clone());
            continue;
        }
        if !out_only(&case, &a.name) {
            ns.taint.taint_alloc(i, a.init.len());
        }
    }
    if !fixed.is_empty() {
        notes.push(format!(
            "integer input buffers held at their captured values (N-ERRDIST-INTFIX): {}",
            fixed.join(", ")
        ));
    }
    m.num = Some(Box::new(ns));
    let (mut m, findings, ok) = run_once(m);
    m.half_out_leaves();
    let output_nodes = output_nodes(&mut m);
    if !ok {
        bail!(
            "{} {}: the numerics run did not terminate: {findings:?}",
            k.variant,
            k.front
        );
    }
    let ns = m.num.as_ref().unwrap();
    // ASSUME(N-ATOMIC): several outputs from fp32 atomic adds alone (no other atomic, no
    // returned value) differ only in the order of each location's adds, which the bound
    // covers for every order.
    if nondet && ns.atomic_order_only(&m.k) {
        classes.retain(|c| c != "nondeterministic");
        if classes.is_empty() {
            classes.push("clean".into());
        }
        notes.push(format!(
            "relies on: N-ATOMIC ({} distinct outputs from the order of {} fp32 atomic adds; bounded over every order)",
            v.distinct_outputs, ns.atomic_adds
        ));
    }
    if classes != ["clean"] {
        pre.push(format!("not clean: {classes:?}"));
    } else if !v.complete {
        pre.push(format!(
            "clean only up to {} states (not exhaustive)",
            v.states
        ));
    }
    let mut excused = 0;
    let mut sel_excused = 0;
    for v in &ns.taint.violations {
        // ASSUME(N-ERRDIST-SEL): a store under a data-dependent guard is a selection node
        // (`NodeKind::Sel`, both outcomes), not a different dataflow.
        if ns.segments
            && ns
                .taint
                .violation_pcs
                .get(v)
                .is_some_and(|pc| ns.sel_pcs.contains(pc) && !ns.seg_fails.contains_key(pc))
        {
            sel_excused += 1;
            continue;
        }
        // ASSUME(N-SEG): a data-dependent guard of an instruction inside a materialized slice
        // is part of the slice's function (both outcomes), not a different dataflow.
        if ns.segments
            && ns
                .taint
                .violation_pcs
                .get(v)
                .is_some_and(|pc| ns.seg_pcs.contains(pc))
        {
            excused += 1;
            continue;
        }
        pre.push(format!("data-dependent: {v}"));
    }
    for v in ptxv_sim::seg::divrn_dead(&m) {
        pre.push(format!("data-dependent: {v}"));
    }
    if sel_excused > 0 {
        notes.push(format!(
            "relies on: N-ERRDIST-SEL ({sel_excused} stores under a data-dependent guard, modeled as selections)"
        ));
    }
    if excused > 0 {
        notes.push(format!(
            "relies on: N-SEG ({excused} data-dependent guards inside instruction slices)"
        ));
    }
    // ASSUME(N-FMAX): float atomic max diamonds, both sides the same update.
    if !ns.taint.fmax_pcs.is_empty() {
        notes.push(format!(
            "relies on: N-FMAX ({} float atomic max branches: signed max / unsigned min by the sign bit, the same float maximum of non-NaN values; inputs finite, no overflow)",
            ns.taint.fmax_pcs.len()
        ));
    }
    // ASSUME(N-FINITE): comparisons fixed by finite inputs and no overflow; the proof is for
    // such inputs.
    if !ns.taint.fin_pcs.is_empty() {
        notes.push(format!(
            "relies on: N-FINITE ({} comparisons decided by finiteness; proven for inputs with finite elements on which no operation overflows)",
            ns.taint.fin_pcs.len()
        ));
    }
    Ok(Run {
        name: case.m.name.clone(),
        case,
        variant: k.variant.clone(),
        front: k.front.clone(),
        cubin,
        kern,
        m,
        outs: output_nodes,
        pre,
        notes,
        captured_out,
    })
}

/// The output buffer's elements: (bits, node).
fn outs(m: &Machine) -> Vec<(u32, Option<u32>)> {
    let oi = m.launch.outputs[0];
    let mem = &m.gmem[oi];
    let ns = m.num.as_ref().unwrap();
    // bf16 / f16 outputs: one value per element, widened to f32 (exact), with the node of its
    // 16-bit provenance (a rounding node, or an input element moved).
    let f16 = ns.half.f16_outputs.contains(&oi);
    if f16 || ns.half.bf16_outputs.contains(&oi) {
        return (0..mem.bytes.len() / 2)
            .map(|i| {
                let h = u16::from_le_bytes([*mem.bytes.get(i * 2), *mem.bytes.get(i * 2 + 1)]);
                let bits = if f16 {
                    ptxv_sim::elementwise::f16_to_f32_bits(h)
                } else {
                    (h as u32) << 16
                };
                (bits, m.half_out_node(oi, i as u64, f16))
            })
            .collect();
    }
    (0..mem.bytes.len() / 4)
        .map(|i| {
            let bits = u32::from_le_bytes([0, 1, 2, 3].map(|j| *mem.bytes.get(i * 4 + j)));
            let node = ns.mem_node(Loc::Global {
                alloc: oi,
                off: (i * 4) as u64,
            });
            (bits, node)
        })
        .collect()
}

fn output_nodes(m: &mut Machine) -> Vec<(u32, Option<u32>)> {
    let allocation = m.launch.outputs[0];
    let state = m.num.as_ref().unwrap();
    let bytes = if state.half.f16_outputs.contains(&allocation)
        || state.half.bf16_outputs.contains(&allocation)
    {
        2
    } else {
        4
    };
    let mut outputs = outs(m);
    for (index, (bits, node)) in outputs.iter_mut().enumerate() {
        let input_derived = (0..bytes).any(|byte| {
            m.num.as_ref().unwrap().taint.memory_unit_input_derived((
                1,
                allocation as u64,
                (index * bytes + byte) as u64,
            ))
        });
        if node.is_none() && input_derived {
            let state = m.num.as_mut().unwrap();
            let id = state.nodes.len() as u32;
            state.nodes.push(ptxv_sim::num::Node {
                kind: ptxv_sim::num::NodeKind::Opaque,
                hw: *bits,
                exact: ptxv_sim::exact::from(f32::from_bits(*bits) as f64),
                err: vec![],
                pc: 0,
            });
            *node = Some(id);
        }
    }
    outputs
}

/// Nonzero outputs without a numerics node: values the shadow does not express over the
/// inputs.
fn untracked_count(m: &Machine) -> usize {
    outs(m)
        .iter()
        .filter(|(b, n)| n.is_none() && f32::from_bits(*b) != 0.0)
        .count()
}

pub(crate) fn side<'a>(r: &'a Run, names: &[String]) -> Side<'a> {
    Side {
        ns: r.m.num.as_ref().unwrap(),
        outs: r.outs.clone(),
        buf_id: r
            .m
            .launch
            .allocs
            .iter()
            .map(|a| names.iter().position(|n| *n == a.name).unwrap() as u16)
            .collect(),
    }
}

/// The case with every input buffer zero except the witness leaves, set to 1.
fn witness_case(case: &Case, leaves: &[(String, u64)]) -> Case {
    let mut c = case.clone();
    let skip: Vec<String> =
        c.m.buffers
            .iter()
            .filter(|b| out_only(case, &b.name))
            .map(|b| b.name.clone())
            .collect();
    for b in &mut c.m.buffers {
        if skip.contains(&b.name) {
            continue;
        }
        b.init = "zeros".into();
        b.values = leaves
            .iter()
            .filter(|(n, _)| *n == b.name)
            .map(|(_, i)| (*i, 1.0))
            .collect();
    }
    c
}

/// Output element `i` of `run` on the witness input, in the simulator and (unless `no_hw`)
/// on GB300.
fn replay(
    run: &Run,
    leaves: &[(String, u64)],
    i: usize,
    dir: &Path,
    no_hw: bool,
) -> Result<(f32, Option<f32>)> {
    let wc = witness_case(&run.case, leaves);
    let (k, _) = load_with(&wc, &run.variant, &run.front, dir, &[])?;
    let (m, f, ok) = run_once(machine(&wc, k)?);
    if !ok {
        bail!("witness run of {} did not terminate: {f:?}", run.label());
    }
    let out = m.outputs();
    let dt = run
        .case
        .m
        .outputs
        .first()
        .and_then(|o| run.case.m.buffers.iter().find(|b| &b.name == o))
        .map_or("f32", |b| b.dtype.as_str());
    // Element i of the output (16-bit widened to f32).
    let elem = |b: &[u8]| -> f32 {
        let h = || u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
        match dt {
            "bf16" => f32::from_bits((h() as u32) << 16),
            "f16" => f32::from_bits(ptxv_sim::elementwise::f16_to_f32_bits(h())),
            _ => f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()),
        }
    };
    let sim = elem(&out);
    if no_hw {
        return Ok((sim, None));
    }
    let Some(&gpu) = ptxv_hw::free_gpus()?.first() else {
        bail!("no free GPU for the witness replay")
    };
    let dump = dir.join(format!("{}.witness.bin", run.variant));
    let _ = std::fs::remove_file(&dump);
    let mut args = vec![
        "hw-worker".to_string(),
        run.case.dir.to_str().unwrap().to_string(),
        run.cubin.to_str().unwrap().to_string(),
        "--runs".into(),
        "1".into(),
        "--dump".into(),
        dump.to_str().unwrap().to_string(),
    ];
    for b in &wc.m.buffers {
        if out_only(&wc, &b.name) {
            continue;
        }
        let p = dir.join(format!("{}.witness.{}.bin", run.variant, b.name));
        std::fs::write(&p, b.init_bytes())?;
        args.push("--input".into());
        args.push(format!("{}={}", b.name, p.display()));
    }
    let mut child = std::process::Command::new(std::env::current_exe()?)
        .args(&args)
        .env("CUDA_VISIBLE_DEVICES", gpu_env(&run.case, gpu)?)
        .stdout(std::process::Stdio::null())
        .spawn()?;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(s) = child.try_wait()? {
            if !s.success() {
                bail!("hw witness run of {} failed", run.label());
            }
            break;
        }
        if Instant::now() > deadline {
            child.kill()?;
            bail!("hw witness run of {} hung", run.label());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let b = std::fs::read(&dump)?;
    let hw = elem(&b);
    Ok((sim, Some(hw)))
}

/// Element type of buffer `name` of a case.
fn etype_of(case: &Case, name: &str) -> Result<ptxv_sim::elementwise::EType, String> {
    let b = case.buffer(name).map_err(|e| e.to_string())?;
    ptxv_sim::elementwise::EType::parse(&b.dtype)
        .ok_or_else(|| format!("buffer {name} is {}, not f32 / bf16 / f16", b.dtype))
}

/// Elementwise mode: both kernels' element functions (`ptxv_sim::elementwise`), output i
/// reading the same input element in both, each distinct pair of functions swept over every
/// input value. Returns the JSON summary and the level ("L0": bitwise equal for every input,
/// "EW": a proven largest difference). ASSUME(N-ELEMENTWISE)
/// Elementwise mode on GB300: run `run` on `batches` batches of input values for its input
/// buffer (batch b: bf16 bit patterns 1024 b .. 1024 b + 1023, widened for f32 inputs; past
/// 2^16 random patterns) and compare every output with its element function, bitwise.
/// Returns (outputs checked, mismatches, first mismatch).
fn elementwise_gpu_check(
    run: &Run,
    ex: &ptxv_sim::elementwise::Extract,
    in_ty: ptxv_sim::elementwise::EType,
    out_ty: ptxv_sim::elementwise::EType,
    batches: usize,
    dir: &Path,
) -> std::result::Result<(u64, u64, Option<String>), String> {
    use ptxv_sim::elementwise::EType;
    let e = |x: anyhow::Error| x.to_string();
    let Some(&gpu) = ptxv_hw::free_gpus().map_err(e)?.first() else {
        return Err("no free GPU".into());
    };
    if ex.outs.iter().any(|o| o.factor.is_some()) {
        return Err("product outputs f(x) * y are not checked on the GPU".into());
    }
    let alloc = ex.outs.first().ok_or("no outputs")?.leaf.0;
    let name = run.m.launch.allocs[alloc].name.clone();
    let buf = run.case.buffer(&name).map_err(e)?;
    let es = if in_ty == EType::F32 { 4 } else { 2 };
    let n = buf.bytes() / es;
    let (mut checked, mut bad, mut first) = (0u64, 0u64, None);
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    let mut scratch = Vec::new();
    for b in 0..batches {
        // Element pattern j of this batch (in the input format).
        let pats: Vec<u32> = (0..n)
            .map(|j| {
                let k = (b * n + j) as u64;
                if k < 1 << 16 {
                    if in_ty == EType::F32 {
                        (k as u32) << 16
                    } else {
                        k as u32
                    }
                } else {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    if in_ty == EType::F32 {
                        rng as u32
                    } else {
                        (rng & 0xffff) as u32
                    }
                }
            })
            .collect();
        let bytes: Vec<u8> = pats
            .iter()
            .flat_map(|&p| p.to_le_bytes()[..es].to_vec())
            .collect();
        let ip = dir.join(format!("ew.{}.{name}.bin", run.variant));
        std::fs::write(&ip, &bytes).map_err(|x| x.to_string())?;
        let dump = dir.join(format!("ew.{}.out.bin", run.variant));
        let _ = std::fs::remove_file(&dump);
        let st = std::process::Command::new(std::env::current_exe().map_err(|x| x.to_string())?)
            .args([
                "hw-worker",
                run.case.dir.to_str().unwrap(),
                run.cubin.to_str().unwrap(),
                "--runs",
                "1",
                "--dump",
                dump.to_str().unwrap(),
                "--input",
                &format!("{name}={}", ip.display()),
            ])
            .env("CUDA_VISIBLE_DEVICES", gpu.to_string())
            .stdout(std::process::Stdio::null())
            .status()
            .map_err(|x| x.to_string())?;
        if !st.success() {
            return Err(format!("hw-worker failed on batch {b}"));
        }
        let out = std::fs::read(&dump).map_err(|x| x.to_string())?;
        let os = if out_ty == EType::F32 { 4 } else { 2 };
        for o in &ex.outs {
            let p = pats[o.leaf.1 as usize];
            let want =
                ptxv_sim::elementwise::out_bits(&ex.fns[o.f], in_ty, out_ty, p, &mut scratch)?;
            let want = if os == 4 { want } else { want & 0xffff };
            let got = out
                .get(o.index * os..o.index * os + os)
                .map(|w| {
                    let mut a = [0u8; 4];
                    a[..os].copy_from_slice(w);
                    u32::from_le_bytes(a)
                })
                .ok_or("short GPU output")?;
            checked += 1;
            if got != want {
                bad += 1;
                if first.is_none() {
                    first = Some(format!(
                        "out[{}] at input bits {p:#x}: GPU {got:#x}, element function {want:#x}",
                        o.index
                    ));
                }
            }
        }
    }
    Ok((checked, bad, first))
}

fn elementwise_pair(
    r: &Run,
    c: &Run,
    func: Option<&str>,
    gpu_batches: usize,
    dir: &Path,
) -> std::result::Result<(serde_json::Value, String), String> {
    use ptxv_sim::elementwise::*;
    let (or, oc) = (r.outs.clone(), c.outs.clone());
    let out_ty = |run: &Run| -> std::result::Result<EType, String> {
        let o = &run.m.launch.allocs[run.m.launch.outputs[0]].name;
        etype_of(&run.case, o)
    };
    let (otr, otc) = (out_ty(r)?, out_ty(c)?);
    if otr != otc {
        return Err(format!(
            "outputs are {} in the reference, {} in the candidate",
            otr.name(),
            otc.name()
        ));
    }
    // bf16 outputs are read element by element (`outs`, bf16 rounding nodes); other 16-bit
    // outputs as packed words.
    let per_elem = |run: &Run| {
        let ns = run.m.num.as_ref().unwrap();
        let o = run.m.launch.outputs[0];
        ns.half.bf16_outputs.contains(&o) || ns.half.f16_outputs.contains(&o)
    };
    if per_elem(r) != per_elem(c) {
        return Err("outputs read differently in the two cases".into());
    }
    let half = otr != EType::F32 && !per_elem(r);
    let er =
        extract(r.m.num.as_ref().unwrap(), &or, half).map_err(|e| format!("reference: {e}"))?;
    let ec =
        extract(c.m.num.as_ref().unwrap(), &oc, half).map_err(|e| format!("candidate: {e}"))?;
    if er.outs.len() != ec.outs.len() {
        return Err(format!(
            "{} outputs in the reference, {} in the candidate",
            er.outs.len(),
            ec.outs.len()
        ));
    }
    let name = |run: &Run, a: usize| run.m.launch.allocs[a].name.clone();
    let mut in_ty = None;
    let mut pairs: std::collections::BTreeMap<(usize, usize), usize> = Default::default();
    for (a, b) in er.outs.iter().zip(&ec.outs) {
        let (na, nb) = (name(r, a.leaf.0), name(c, b.leaf.0));
        if (na.as_str(), a.leaf.1) != (nb.as_str(), b.leaf.1) {
            return Err(format!(
                "output {} reads {na}[{}] in the reference but {nb}[{}] in the candidate",
                a.index, a.leaf.1, b.leaf.1
            ));
        }
        if a.round16 != b.round16 {
            return Err(format!(
                "output {} is bf16(f(x) * y) on one side only",
                a.index
            ));
        }
        // Product outputs: the same factor element on both sides.
        match (a.factor, b.factor) {
            (None, None) => {}
            (Some(fa), Some(fb)) if (name(r, fa.0), fa.1) == (name(c, fb.0), fb.1) => {}
            _ => {
                return Err(format!(
                    "output {} is f(x) * y on one side only, or with different factors y",
                    a.index
                ));
            }
        }
        let t = etype_of(&r.case, &na)?;
        if *in_ty.get_or_insert(t) != t {
            return Err("inputs of different element types".into());
        }
        *pairs.entry((a.f, b.f)).or_default() += 1;
    }
    let in_ty = in_ty.ok_or("no outputs")?;
    // bf16(rn(f(x) * y)) outputs: f is an f32 function; the outputs at every (x, y) pair.
    let round16 = er.outs.iter().any(|o| o.round16);
    let y_ty = match er.outs.iter().find_map(|o| o.factor) {
        Some(f) => Some(etype_of(&r.case, &name(r, f.0))?),
        None => None,
    };
    let f_out = if er.outs.iter().any(|o| o.factor.is_some()) {
        EType::F32
    } else {
        otr
    };
    let (identified, func) = match func {
        Some(f) => {
            if true_fn(f, 1.0).is_none() {
                return Err(format!(
                    "unknown function {f} (known: {})",
                    TRUE_FUNCS.join(", ")
                ));
            }
            (false, Some(f.to_string()))
        }
        None => {
            let (a, b) = *pairs.keys().next().unwrap();
            (
                true,
                identify(&[&er.fns[a], &ec.fns[b]], in_ty, f_out).map(|(n, _)| n.to_string()),
            )
        }
    };
    let mut pj = Vec::new();
    let mut all = PairSweep::default();
    let (mut tr, mut tc): (Option<VsTrue>, Option<VsTrue>) = (None, None);
    let mut bitwise_all = true;
    let mut secs = 0.0;
    let mut prod16: Option<ProductSweep> = None;
    for (&(a, b), &n) in &pairs {
        let s = sweep(&er.fns[a], &ec.fns[b], in_ty, f_out, func.as_deref())?;
        if round16 {
            let ps = product_sweep(&er.fns[a], &ec.fns[b], in_ty, y_ty.unwrap(), otr)?;
            secs += ps.seconds;
            bitwise_all &= ps.bitwise_equal == ps.pairs;
            let acc = prod16.get_or_insert_with(ProductSweep::default);
            acc.pairs += ps.pairs;
            acc.bitwise_equal += ps.bitwise_equal;
            acc.both_nan += ps.both_nan;
            acc.special_mismatch += ps.special_mismatch;
            acc.one_ulp += ps.one_ulp;
            acc.max_abs = acc.max_abs.max(ps.max_abs);
            if ps.max_ulps_normal > acc.max_ulps_normal {
                acc.max_ulps_normal = ps.max_ulps_normal;
                acc.at_ulps_normal = ps.at_ulps_normal;
            }
            if ps.max_ulps > acc.max_ulps {
                acc.max_ulps = ps.max_ulps;
                acc.at_ulps = ps.at_ulps;
            }
        } else {
            bitwise_all &= s.bitwise_equal == s.inputs;
        }
        secs += s.seconds;
        if s.max_ulps > all.max_ulps {
            all.max_ulps = s.max_ulps;
            all.at_ulps = s.at_ulps;
            all.values_at_ulps = s.values_at_ulps;
        }
        if s.max_abs > all.max_abs {
            all.max_abs = s.max_abs;
            all.at_abs = s.at_abs;
            all.values_at_abs = s.values_at_abs;
        }
        if s.max_rel > all.max_rel {
            all.max_rel = s.max_rel;
            all.at_rel = s.at_rel;
        }
        all.special_mismatch += s.special_mismatch;
        all.bitwise_equal += s.bitwise_equal;
        all.inputs += s.inputs;
        for (acc, v) in [
            (&mut tr, &s.reference_vs_true),
            (&mut tc, &s.candidate_vs_true),
        ] {
            if let Some(v) = v {
                let a = acc.get_or_insert_with(VsTrue::default);
                if v.max_ulps >= a.max_ulps {
                    a.max_ulps = v.max_ulps;
                    a.at_ulps = v.at_ulps;
                }
                if v.max_ulps_big >= a.max_ulps_big {
                    a.max_ulps_big = v.max_ulps_big;
                    a.at_ulps_big = v.at_ulps_big;
                }
                if v.max_abs >= a.max_abs {
                    a.max_abs = v.max_abs;
                    a.at_abs = v.at_abs;
                }
                if v.max_rel >= a.max_rel {
                    a.max_rel = v.max_rel;
                    a.at_rel = v.at_rel;
                }
                a.non_finite += v.non_finite;
                a.overflow += v.overflow;
            }
        }
        pj.push(serde_json::json!({
            "outputs": n,
            "reference_ops": er.fns[a].size(),
            "candidate_ops": ec.fns[b].size(),
            "sweep": s,
        }));
    }
    let n_in = in_ty.count();
    println!(
        "    elementwise: every output i is f(x[i]){}: {} reference / {} candidate function(s) ({} / {} slice ops), {} pair(s) swept over all {} {} inputs ({:.0} s)",
        if er.outs.iter().any(|o| o.factor.is_some()) {
            " * y[i]"
        } else {
            ""
        },
        er.fns.len(),
        ec.fns.len(),
        er.fns.iter().map(|f| f.size()).max().unwrap_or(0),
        ec.fns.iter().map(|f| f.size()).max().unwrap_or(0),
        pairs.len(),
        if n_in == 1 << 32 {
            "2^32".to_string()
        } else {
            "2^16".to_string()
        },
        in_ty.name(),
        secs
    );
    println!(
        "      reference vs candidate: bitwise equal on {} of {} inputs; where both finite: max {} ulp (x = {:e}: {:e} vs {:e}), |C - R| <= {:.3e} (x = {:e}), relative <= {:.3e}; {} special-value mismatches",
        all.bitwise_equal,
        all.inputs,
        all.max_ulps,
        all.at_ulps,
        all.values_at_ulps.0,
        all.values_at_ulps.1,
        all.max_abs,
        all.at_abs,
        all.max_rel,
        all.special_mismatch
    );
    if let (Some(f), Some(a), Some(b)) = (&func, &tr, &tc) {
        println!(
            "      vs {f}{} in f64, finite inputs: reference |err| <= {:.3e} (x = {:e}), max {:.3} ulp (x = {:e}), {:.3} ulp where |{f}| >= 2^-10; candidate |err| <= {:.3e} (x = {:e}), max {:.3} ulp (x = {:e}), {:.3} ulp where |{f}| >= 2^-10",
            if identified { " (identified)" } else { "" },
            a.max_abs,
            a.at_abs,
            a.max_ulps,
            a.at_ulps,
            a.max_ulps_big,
            b.max_abs,
            b.at_abs,
            b.max_ulps,
            b.at_ulps,
            b.max_ulps_big
        );
    } else if func.is_none() {
        println!(
            "      no library function within 2^-10 of both kernels: not compared with a true function"
        );
    }
    if let Some(p) = &prod16 {
        println!(
            "      outputs {}(rn(f(x) * y)), every (x, y) pair ({} pairs): bitwise equal on {}, 1 ulp apart on {}, max {} ulp (x = {:e}, y = {:e}: {:e} vs {:e}), |C - R| <= {:.3e}; {} special-value mismatches; where f and both outputs are normal: max {} ulp (x = {:e}, y = {:e}: {:e} vs {:e})",
            otr.name(),
            p.pairs,
            p.bitwise_equal,
            p.one_ulp,
            p.max_ulps,
            p.at_ulps.0,
            p.at_ulps.1,
            p.at_ulps.2,
            p.at_ulps.3,
            p.max_abs,
            p.special_mismatch,
            p.max_ulps_normal,
            p.at_ulps_normal.0,
            p.at_ulps_normal.1,
            p.at_ulps_normal.2,
            p.at_ulps_normal.3,
        );
    }
    let level = if bitwise_all { "L0" } else { "EW" }.to_string();
    let mut gpu = serde_json::Value::Null;
    if gpu_batches > 0 {
        let mut g = Vec::new();
        for (who, run, ex, ot) in [("reference", r, &er, otr), ("candidate", c, &ec, otc)] {
            match elementwise_gpu_check(run, ex, in_ty, ot, gpu_batches, dir) {
                Ok((n, bad, first)) => {
                    println!(
                        "      GB300 ({who}): {n} outputs over {gpu_batches} batches of input values, {bad} differ from the element functions{}",
                        first
                            .as_ref()
                            .map(|f| format!(" (first: {f})"))
                            .unwrap_or_default()
                    );
                    g.push(serde_json::json!({"side": who, "checked": n, "mismatches": bad, "first": first}));
                }
                Err(e) => {
                    println!("      GB300 ({who}): not run: {e}");
                    g.push(serde_json::json!({"side": who, "error": e}));
                }
            }
        }
        gpu = serde_json::json!(g);
    }
    Ok((
        serde_json::json!({
            "level": level,
            "outputs": er.outs.len(),
            "input_type": in_ty.name(),
            "output_type": otr.name(),
            // Every output rn(f(x) * y), y another input element (the same on both sides):
            // the sweep is of f; bitwise-equal f gives bitwise-equal outputs (one identical
            // rounding), else |C - R| <= |y| max|f_C - f_R| + 2^-24 (|C| + |R|) + 2^-149.
            "product": er.outs.iter().any(|o| o.factor.is_some()),
            // 16-bit product outputs: both kernels at every (x, y) pair (exhaustive).
            "product16": prod16,
            "inputs_swept": n_in,
            "functions_reference": er.fns.len(),
            "functions_candidate": ec.fns.len(),
            "true_fn": func,
            "identified": identified,
            "bitwise_all": bitwise_all,
            "bitwise_equal_inputs": all.bitwise_equal,
            "inputs": all.inputs,
            "max_ulps": all.max_ulps,
            "at_ulps": all.at_ulps,
            "values_at_ulps": all.values_at_ulps,
            "max_abs": all.max_abs,
            "at_abs": all.at_abs,
            "max_rel": all.max_rel,
            "special_mismatch": all.special_mismatch,
            "reference_vs_true": tr,
            "candidate_vs_true": tc,
            "pairs": pj,
            "seconds": secs,
            "gpu_check": gpu,
        }),
        level,
    ))
}

/// The strongest level that holds: "L0".."L3", or "none".
fn strongest(rep: &Report) -> String {
    let holds = |p: &str| rep.levels.iter().any(|l| l.level.starts_with(p) && l.holds);
    if holds("L0") {
        "L0".into()
    } else if holds("L2") {
        if holds("L3") {
            "L2+L3".into()
        } else {
            "L2".into()
        }
    } else if holds("L1") {
        if holds("L3") {
            "L1+L3".into()
        } else {
            "L1".into()
        }
    } else if !rep.per_output.is_empty()
        && rep
            .per_output
            .iter()
            .all(|o| matches!(o.level, "L1" | "L1~") && o.bound_u.is_some())
    {
        // The same function up to rounded constants (norm form), with a bound per output.
        "L1~".into()
    } else {
        "none".into()
    }
}

/// Print the per-output levels and bounds: every output, or runs of consecutive outputs at
/// the same level with the range of their bounds.
fn print_outputs(rep: &Report, all: bool) {
    let show = |b: Option<f64>| b.map_or("-".to_string(), |b| format!("{b:.2}u"));
    // The bound's form: sum|terms| for polynomials, y (the exact output) for softmax.
    let form = |o: &OutBound| match (o.form, o.rel_u, o.add_u) {
        ("softmax", Some(r), Some(a)) => format!("{r:.2}u * y + {a:.2}u"),
        ("norm", _, _) => norm_form(o),
        ("round", Some(r), Some(a)) => format!("{r:.0}u * |y| + {a:.2}u * sum|terms|"),
        ("attn", Some(r), Some(a)) => match o.score {
            Some((kq, _)) => format!("({r:.2} + {kq:.2} Q)u * V1 + {a:.2}u * Vmax"),
            None => format!("{r:.2}u * V1 + {a:.2}u * Vmax"),
        },
        _ => format!("{} * sum|terms|", show(o.bound_u)),
    };
    if all || rep.per_output.len() <= 16 {
        for o in &rep.per_output {
            println!(
                "      out[{}]: {:<4} |C-R| <= {}{}  (reference {}, candidate {}){}",
                o.index,
                o.level,
                form(o),
                if o.abs_extra > 0.0 {
                    format!(" + {:.1e}", o.abs_extra)
                } else {
                    String::new()
                },
                show(o.ref_u),
                show(o.cand_u),
                o.no_bound
                    .as_ref()
                    .map(|w| format!("  no bound: {w}"))
                    .unwrap_or_default()
            );
        }
        return;
    }
    // Runs of consecutive outputs with a bound (or without), L0 kept apart.
    let key = |o: &OutBound| (o.bound_u.is_some(), o.level == "L0", o.form);
    let mut i = 0;
    while i < rep.per_output.len() {
        let o = &rep.per_output[i];
        let mut j = i;
        while j + 1 < rep.per_output.len() && key(&rep.per_output[j + 1]) == key(o) {
            j += 1;
        }
        let run = &rep.per_output[i..=j];
        let mut counts: Vec<(&str, usize)> = Vec::new();
        for o in run {
            match counts.iter_mut().find(|c| c.0 == o.level) {
                Some(c) => c.1 += 1,
                None => counts.push((o.level, 1)),
            }
        }
        let levels = counts
            .iter()
            .map(|(l, n)| format!("{l} x{n}"))
            .collect::<Vec<_>>()
            .join(", ");
        let lo = run
            .iter()
            .filter_map(|o| o.bound_u)
            .fold(f64::INFINITY, f64::min);
        let (hi_i, hi) = run
            .iter()
            .filter_map(|o| o.bound_u.map(|b| (o.index, b)))
            .fold((i, 0.0), |a, b| if b.1 > a.1 { b } else { a });
        let extra = run.iter().map(|o| o.abs_extra).fold(0.0, f64::max);
        match o.bound_u {
            Some(_) if o.form == "attn" => {
                let rel = run.iter().filter_map(|o| o.rel_u).fold(0.0, f64::max);
                let add = run.iter().filter_map(|o| o.add_u).fold(0.0, f64::max);
                let kq = run
                    .iter()
                    .filter_map(|o| o.score)
                    .fold(None, |m: Option<(f64, f64)>, s| {
                        Some(m.map_or(s, |m| (m.0.max(s.0), m.1.min(s.1))))
                    });
                let rel = match kq {
                    Some((k, qm)) => format!(
                        "({rel:.2} + {k:.2} Q)u * V1 + {add:.2}u * Vmax, Q = max_j sum_d |q_d k_jd| <= {qm:.0}"
                    ),
                    None => format!("{rel:.2}u * V1 + {add:.2}u * Vmax"),
                };
                println!(
                    "      out[{i}..={j}]: {levels}; attention form, |C-R| <= {rel}{} (V1 = sum_j p_j |v_j| <= Vmax = max |v_j|; largest total {hi:.2}u at out[{hi_i}]: reference {}, candidate {})",
                    if extra > 0.0 {
                        format!(" + {extra:.1e}")
                    } else {
                        String::new()
                    },
                    show(rep.per_output[hi_i].ref_u),
                    show(rep.per_output[hi_i].cand_u),
                )
            }
            Some(_) if o.form == "round" => {
                let rel = run.iter().filter_map(|o| o.rel_u).fold(0.0, f64::max);
                let add = run.iter().filter_map(|o| o.add_u).fold(0.0, f64::max);
                let (ri, rm) = run
                    .iter()
                    .filter_map(|o| o.add_u.map(|a| (o.index, a)))
                    .fold((i, 0.0), |a, b| if b.1 > a.1 { b } else { a });
                let _ = rm;
                let fmt = if o.round_f16 == Some(true) {
                    "f16"
                } else {
                    "bf16"
                };
                let inner = run
                    .iter()
                    .filter_map(|o| o.round_inner)
                    .fold((false, false), |a, b| (a.0 | b.0, a.1 | b.1));
                let inner = match inner {
                    (false, false) => String::new(),
                    (r, c) => format!(
                        "; {} round(s) to {fmt} before the output: not the same function, the roundings are bounded as errors",
                        match (r, c) {
                            (true, true) => "both kernels",
                            (true, false) => "the reference",
                            _ => "the candidate",
                        }
                    ),
                };
                // Equal or adjacent outputs (<= 1 ulp) where the inputs are well conditioned.
                let adj = run
                    .iter()
                    .map(|o| o.round_adjacent)
                    .try_fold((f64::INFINITY, 0.0f64), |a, b| {
                        b.map(|b| (a.0.min(b.0), a.1.max(b.1)))
                    });
                let adj = match adj {
                    Some((k1, y1)) => format!(
                        "; C and R are equal or adjacent {fmt} values (<= 1 ulp) for every input with sum|terms| <= {k1:.0} |y| and |y| >= {y1:.1e}"
                    ),
                    None => format!("; <= 1 {fmt} ulp is not implied for any conditioning"),
                };
                println!(
                    "      out[{i}..={j}]: {levels}; {fmt} output form, |C-R| <= {rel:.0}u * |y| + {add:.2}u * sum|terms|{} (y = exact value; 2 u16 |y| for the two output roundings; per input: the distance between the {fmt} roundings of [y - G_R, y + G_C]{adj}{inner}; largest at out[{ri}]: reference {}, candidate {})",
                    if extra > 0.0 {
                        format!(" + {extra:.1e}")
                    } else {
                        String::new()
                    },
                    show(rep.per_output[ri].ref_u),
                    show(rep.per_output[ri].cand_u),
                )
            }
            Some(_) if o.form == "norm" => {
                let w = run
                    .iter()
                    .max_by(|a, b| {
                        a.bound_u
                            .unwrap_or(0.0)
                            .total_cmp(&b.bound_u.unwrap_or(0.0))
                    })
                    .unwrap();
                println!(
                    "      out[{i}..={j}]: {levels}; norm form, largest at out[{hi_i}]: {} (reference {}, candidate {})",
                    norm_form(w),
                    show(w.ref_u),
                    show(w.cand_u),
                )
            }
            Some(_) if o.form == "softmax" => {
                let rel = run.iter().filter_map(|o| o.rel_u).fold(0.0, f64::max);
                let add = run.iter().filter_map(|o| o.add_u).fold(0.0, f64::max);
                println!(
                    "      out[{i}..={j}]: {levels}; softmax form, |C-R| <= {rel:.2}u * y + {add:.2}u{} (y = exact output <= 1; largest total {hi:.2}u at out[{hi_i}]: reference {}, candidate {})",
                    if extra > 0.0 {
                        format!(" + {extra:.1e}")
                    } else {
                        String::new()
                    },
                    show(rep.per_output[hi_i].ref_u),
                    show(rep.per_output[hi_i].cand_u),
                )
            }
            Some(_) => println!(
                "      out[{i}..={j}]: {levels}; |C-R| <= k u sum|terms|{}, k from {lo:.2} to {hi:.2} (largest at out[{hi_i}]: reference {}, candidate {})",
                if extra > 0.0 {
                    format!(" + {extra:.1e}")
                } else {
                    String::new()
                },
                show(rep.per_output[hi_i].ref_u),
                show(rep.per_output[hi_i].cand_u),
            ),
            None => println!(
                "      out[{i}..={j}]: {levels}; no bound ({})",
                o.no_bound.as_deref().unwrap_or("?")
            ),
        }
        i = j + 1;
    }
}

/// The norm form's bound in words (`ptxv_sim::norm`).
fn norm_form(o: &OutBound) -> String {
    let Some(n) = &o.norm else {
        return "-".into();
    };
    let kappa = if n.kappa_le_1 {
        format!("kappa <= 1 (proven), so <= {:.2}u * T", n.total_u())
    } else {
        format!(
            "for kappa <= {:.0} (so <= {:.2}u * T)",
            n.kappa_max,
            n.total_u()
        )
    };
    format!(
        "|C-R| <= ({:.2} + {:.2} kappa)u * |Y1| + {:.2}u * T, {kappa}",
        n.rel_u, n.kappa_u, n.add_u
    )
}

/// A random execution of `m` (every enabled action equally likely): another schedule than
/// `run_once`'s, which reorders atomics.
fn run_random(mut m: Machine, seed: u64) -> (Machine, bool) {
    let mut rng = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    loop {
        let acts = m.actions();
        if acts.is_empty() {
            break;
        }
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let pick = acts
            .iter()
            .find(|(a, _)| matches!(a, ptxv_sim::explore::Action::Split { .. }))
            .unwrap_or(&acts[(rng % acts.len() as u64) as usize])
            .0
            .clone();
        m.apply(&pick);
        m.findings.clear();
    }
    let ok = m.terminal();
    (m, ok)
}

/// Random inputs of trial `t` for every f32 input buffer (by name, so both kernels see the
/// same values): normal for even t, uniform in [1, 2) for odd t.
fn trial_inputs(
    cases: &[&Case],
    t: usize,
    bytes: &std::collections::BTreeMap<String, Vec<ptxv_ir::Lp>>,
) -> std::collections::BTreeMap<String, Vec<f32>> {
    use ptxv_ir::Lp;
    let mut out = std::collections::BTreeMap::new();
    // NVFP4 (ue4m3 scales): every buffer its own stream.
    let nv = bytes.values().any(|f| f.contains(&Lp::Ue4m3));
    for c in cases {
        for b in &c.m.buffers {
            // Byte buffers read as narrow-format codes (tagged leaves): random codes, one value
            // per byte. FP8 operands: any code but the e5m2 / e4m3 Inf and NaN ones; u16
            // buffers hold ue8m0 scales: 2^-8 .. 2^7. ue4m3 scales: any finite nonzero code
            // (subnormals included; a zero scale is outside the FP4 contract, N-MMA-BS); e2m1
            // codes only (NVFP4): any byte.
            if let Some(fmts) = bytes
                .get(&b.name)
                .filter(|_| !out.contains_key(&b.name) && !out_only(c, &b.name))
            {
                let n = b.init_bytes().len();
                let mut st = (t as u64 + 7).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
                if nv {
                    st ^= b
                        .name
                        .bytes()
                        .fold(0u64, |h, x| h.wrapping_mul(31).wrapping_add(x as u64))
                        .wrapping_mul(0x2545_f491_4f6c_dd1d);
                    st |= 1;
                }
                let ue4 = fmts.contains(&Lp::Ue4m3);
                let fp4 = fmts.iter().all(|f| *f == Lp::E2m1);
                let v: Vec<f32> = (0..n)
                    .map(|_| {
                        st ^= st << 13;
                        st ^= st >> 7;
                        st ^= st << 17;
                        let c = if ue4 {
                            1 + st % 126
                        } else if b.dtype == "u16" {
                            119 + st % 16
                        } else if fp4 && nv {
                            st % 256
                        } else {
                            let c = st % 248;
                            if c >= 124 { c + 4 } else { c }
                        };
                        c as f32
                    })
                    .collect();
                out.insert(b.name.clone(), v);
                continue;
            }
            if out_only(c, &b.name)
                || !matches!(b.dtype.as_str(), "f32" | "bf16" | "f16")
                || out.contains_key(&b.name)
            {
                continue;
            }
            let bf16 = b.dtype == "bf16";
            let f16 = b.dtype == "f16";
            let mut st = (t as u64 + 1).wrapping_mul(0x2545_f491_4f6c_dd1d)
                ^ b.name
                    .bytes()
                    .fold(0u64, |h, x| h.wrapping_mul(31).wrapping_add(x as u64));
            let mut unit = || {
                st = st.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = st;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
            };
            let n = b.shape.iter().product::<u64>() as usize;
            let v: Vec<f32> = (0..n)
                .map(|_| {
                    let x = if t.is_multiple_of(2) {
                        let (a, r) = (unit().max(1e-300), unit());
                        ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * r).cos()) as f32
                    } else {
                        (1.0 + unit()) as f32
                    };
                    // bf16 inputs: a bf16 value (truncated; exact in f32).
                    if bf16 {
                        f32::from_bits(x.to_bits() & 0xffff_0000)
                    } else if f16 {
                        // An f16 value (exact in f32).
                        f32::from_bits(ptxv_sim::elementwise::f16_to_f32_bits(
                            ptxv_sim::fp::f32_to_f16_rn(x.to_bits()),
                        ))
                    } else {
                        x
                    }
                })
                .collect();
            out.insert(b.name.clone(), v);
        }
    }
    out
}

/// Output 0 of `run` on the given inputs, on a random schedule.
fn run_on(
    run: &Run,
    inputs: &std::collections::BTreeMap<String, Vec<f32>>,
    dir: &Path,
    t: usize,
) -> Result<Vec<f32>> {
    let mut c = run.case.clone();
    for b in &mut c.m.buffers {
        if let Some(v) = inputs
            .get(&b.name)
            .filter(|_| !out_only(&run.case, &b.name))
        {
            let p = dir.join(format!("trial{t}.{}.bin", b.name));
            let bytes: Vec<u8> = if matches!(b.dtype.as_str(), "u8" | "u16" | "e4m3") {
                // Byte codes (`trial_inputs`).
                v.iter().map(|x| *x as u8).collect()
            } else if b.dtype == "bf16" {
                v.iter()
                    .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                    .collect()
            } else if b.dtype == "f16" {
                v.iter()
                    .flat_map(|x| ptxv_sim::fp::f32_to_f16_rn(x.to_bits()).to_le_bytes())
                    .collect()
            } else {
                v.iter().flat_map(|x| x.to_le_bytes()).collect()
            };
            std::fs::write(&p, bytes)?;
            b.init = "file".into();
            b.file = Some(p.to_str().unwrap().to_string());
        }
    }
    // A random schedule only matters for fp32 atomic adds (their order); other kernels are
    // race-free (precondition), so the first-action schedule gives the same outputs.
    let atomics = run.m.num.as_ref().is_some_and(|n| n.atomic_adds > 0);
    let (m, ok) = if atomics {
        run_random(machine(&c, run.kern.clone())?, t as u64)
    } else {
        let (m, _, ok) = run_once(machine(&c, run.kern.clone())?);
        (m, ok)
    };
    if !ok {
        bail!("trial {t} of {} did not terminate", run.label());
    }
    Ok(out_f32(&m, &c))
}

/// The first output buffer of a finished run as f32 values (bf16 / f16 widened).
pub(crate) fn out_f32(m: &Machine, c: &Case) -> Vec<f32> {
    let oi = m.launch.outputs[0];
    let mem = &m.gmem[oi];
    let dt =
        c.m.buffers
            .iter()
            .find(|b| b.name == m.launch.allocs[oi].name)
            .map_or("f32", |b| b.dtype.as_str());
    if dt == "bf16" || dt == "f16" {
        return (0..mem.bytes.len() / 2)
            .map(|i| {
                let h = u16::from_le_bytes([*mem.bytes.get(i * 2), *mem.bytes.get(i * 2 + 1)]);
                f32::from_bits(if dt == "f16" {
                    ptxv_sim::elementwise::f16_to_f32_bits(h)
                } else {
                    (h as u32) << 16
                })
            })
            .collect();
    }
    (0..mem.bytes.len() / 4)
        .map(|i| f32::from_le_bytes([0, 1, 2, 3].map(|j| *mem.bytes.get(i * 4 + j))))
        .collect()
}

/// Empirical check of the proven bounds over `n` random trials.
#[derive(Serialize, Default)]
struct Trials {
    trials: usize,
    /// Output-trials with a bound, and how many exceeded it (must be 0).
    checked: usize,
    violations: usize,
    /// max |C - R| / bound(x) over all output-trials: how much of the bound was used.
    max_used: f64,
    /// max |C - R| / (u scale(x)): the observed k, against the proven max k.
    max_observed_u: f64,
    /// Outputs whose bits differed in some trial.
    differing: usize,
    /// The same for each kernel against the exact value (degree <= 2 monomials only).
    max_used_ref: f64,
    max_used_cand: f64,
    /// Norm form: output-trials whose input the bound does not cover (`NormEval::at`).
    uncovered: usize,
    /// Norm form: the largest kappa (A's conditioning) over the trials.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_kappa: Option<f64>,
}

fn trials(
    r: &Run,
    c: &Run,
    rep: &Report,
    names: &[String],
    n: usize,
    dir: &Path,
) -> Result<Trials> {
    let mut out = Trials {
        trials: n,
        ..Default::default()
    };
    let mut differ = vec![false; rep.per_output.len()];
    // Byte buffers whose codes are leaves (narrow tensor-core operands, scale factors).
    let mut bytes: std::collections::BTreeMap<String, Vec<ptxv_ir::Lp>> = Default::default();
    for l in rep
        .per_output
        .iter()
        .flat_map(|o| o.terms.iter().flat_map(|t| t.0.iter()))
    {
        if let Some((f, _)) = ptxv_sim::equiv::leaf_lp(l.1) {
            let e = bytes.entry(names[l.0 as usize].clone()).or_default();
            if !e.contains(&f) {
                e.push(f);
            }
        }
    }
    // GPU trials (tools/equiv_gpu_check.py --dump): inputs and both kernels' outputs as run on
    // GB300, in PTXV_EQUIV_HW_TRIALS/<candidate case>/t<T>.<buffer>.bin, t<T>.ref.bin,
    // t<T>.cand.bin (f32), checked against the same per-input bounds.
    let hw = std::env::var("PTXV_EQUIV_HW_TRIALS")
        .ok()
        .map(|d| PathBuf::from(d).join(&c.case.m.name))
        .filter(|d| d.is_dir());
    let n = match &hw {
        Some(d) => (0..)
            .take_while(|t| d.join(format!("t{t}.ref.bin")).exists())
            .count(),
        None => n,
    };
    out.trials = n;
    let rd = |p: PathBuf| -> Result<Vec<f32>> {
        Ok(std::fs::read(&p)?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    };
    for t in 0..n {
        let (inp, ro, co) = match &hw {
            Some(d) => {
                let mut inp = trial_inputs(&[&r.case, &c.case], t, &bytes);
                for (name, v) in inp.iter_mut() {
                    *v = rd(d.join(format!("t{t}.{name}.bin")))?;
                }
                let ro = rd(d.join(format!("t{t}.ref.bin")))?;
                let co = rd(d.join(format!("t{t}.cand.bin")))?;
                (inp, ro, co)
            }
            None => {
                let inp = trial_inputs(&[&r.case, &c.case], t, &bytes);
                let (ro, co) = (run_on(r, &inp, dir, t)?, run_on(c, &inp, dir, t)?);
                (inp, ro, co)
            }
        };
        // Values by buffer id (the bounds evaluate many monomials per output).
        let by_id: Vec<Option<&Vec<f32>>> = names.iter().map(|n| inp.get(n)).collect();
        // A new input: the attention bounds' cached score values are stale.
        ptxv_sim::softmax::ATTN_INPUT_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let x = |(b, i): ptxv_sim::equiv::Leaf| -> f64 {
            // A byte read as a narrow format: its code decoded.
            if let Some((t, k)) = ptxv_sim::equiv::leaf_lp(i) {
                let v = by_id.get(b as usize).copied().flatten();
                return ptxv_sim::equiv::leaf_code(t, k, |j| v?.get(j as usize).map(|&c| c as u32))
                    .map_or(f64::NAN, |c| ptxv_sim::lowp::value(c, t));
            }
            by_id
                .get(b as usize)
                .copied()
                .flatten()
                .and_then(|v| v.get(i as usize))
                .map_or(f64::NAN, |&v| v as f64)
        };
        for o in &rep.per_output {
            // Norm form: one evaluation gives the bound, T and kappa.
            let norm_at = o.norm.as_ref().map(|nb| nb.eval.at(&x));
            let bound = match norm_at {
                Some(a) => a.map(|a| a.0),
                None => o.bound_at(&x),
            };
            let Some(bound) = bound else {
                // Norm form: an input outside the covered ones (A's error above 1/2, or A* < 2^-125).
                if o.norm.is_some() {
                    out.uncovered += 1;
                }
                continue;
            };
            let (rv, cv) = (ro[o.index] as f64, co[o.index] as f64);
            let d = (rv - cv).abs();
            differ[o.index] |= ro[o.index].to_bits() != co[o.index].to_bits();
            out.checked += 1;
            if d > bound {
                out.violations += 1;
            }
            if bound > 0.0 {
                out.max_used = out.max_used.max(d / bound);
            }
            let scale = match norm_at {
                Some(Some(a)) => a.2,
                _ => o.scale_at(&x),
            };
            if scale > 0.0 {
                out.max_observed_u = out.max_observed_u.max(d / (U32 * scale));
            }
            // Norm form: each kernel against the reference's exact polynomial at rho*.
            if let Some(nb) = &o.norm {
                if let Some(Some(a)) = norm_at {
                    out.max_kappa = Some(out.max_kappa.unwrap_or(0.0).max(a.3));
                }
                if let Some((e, br, bc)) = nb.eval.own(&x) {
                    out.max_used_ref = out.max_used_ref.max((rv - e).abs() / br);
                    out.max_used_cand = out.max_used_cand.max((cv - e).abs() / bc);
                }
                continue;
            }
            // Attention form: each kernel against the exact sum_j p_j v_j.
            if let Some(a) = &o.attn {
                let (_, e, br, bc) = a.at(&x);
                out.max_used_ref = out.max_used_ref.max((rv - e).abs() / br);
                out.max_used_cand = out.max_used_cand.max((cv - e).abs() / bc);
                continue;
            }
            // Softmax: each kernel against the exact softmax (f64).
            if let Some((y, br, bc)) = o.softmax_at(&x) {
                out.max_used_ref = out.max_used_ref.max((rv - y).abs() / br);
                out.max_used_cand = out.max_used_cand.max((cv - y).abs() / bc);
                continue;
            }
            // bf16 output form: each kernel against the exact value, its own bound there.
            if let Some((e, br, bc)) = o.round_own_at(&x) {
                out.max_used_ref = out.max_used_ref.max((rv - e).abs() / br);
                out.max_used_cand = out.max_used_cand.max((cv - e).abs() / bc);
                continue;
            }
            // Each kernel against the exact value sum_m c_m x_m (expansions: exact products).
            {
                let ex = o.exact_at(&x);
                let bscale = |k: Option<f64>| k.map(|k| k * U32 * scale + o.abs_extra / 2.0);
                if let (Some(br), Some(bc)) = (bscale(o.ref_u), bscale(o.cand_u)) {
                    if br > 0.0 {
                        out.max_used_ref = out.max_used_ref.max((rv - ex).abs() / br);
                    }
                    if bc > 0.0 {
                        out.max_used_cand = out.max_used_cand.max((cv - ex).abs() / bc);
                    }
                }
            }
        }
    }
    out.differing = differ.iter().filter(|&&d| d).count();
    Ok(out)
}

/// The captured GPU outputs (`expect` of output 0) of a case, as f32, when it has them.
/// (bf16 and f16 outputs are widened exactly.)
pub(crate) fn gpu_output_f32(case: &Case) -> Option<Vec<f32>> {
    let b = case.buffer(case.m.outputs.first()?).ok()?;
    let bytes = std::fs::read(b.expect.as_ref()?).ok()?;
    let half = |c: &[u8]| u16::from_le_bytes([c[0], c[1]]);
    Some(match b.dtype.as_str() {
        "f32" => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        "bf16" => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((half(c) as u32) << 16))
            .collect(),
        "f16" => bytes.chunks_exact(2).map(|c| f16_to_f32(half(c))).collect(),
        _ => return None,
    })
}

fn f16_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    match e {
        0 => s * m * 2f32.powi(-24),
        31 if m == 0.0 => s * f32::INFINITY,
        31 => f32::NAN,
        _ => s * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// The proven bound evaluated at the captured inputs, against the two captured GPU outputs:
/// how much of the bound the GPU's actual difference uses. None when either case lacks GPU
/// outputs or the two cases' shared input buffers hold different bytes.
fn captured(r: &Run, c: &Run, rep: &Report, names: &[String]) -> Option<serde_json::Value> {
    let rep_bounds = rep.untracked.is_none();
    let (gr, gc) = (gpu_output_f32(&r.case)?, gpu_output_f32(&c.case)?);
    if gr.len() != gc.len() {
        return None;
    }
    let mut inputs: std::collections::BTreeMap<String, Vec<u8>> = Default::default();
    for case in [&r.case, &c.case] {
        for b in &case.m.buffers {
            if out_only(case, &b.name) {
                continue;
            }
            let bytes = b.init_bytes();
            match inputs.get(&b.name) {
                Some(o) if *o != bytes => {
                    return Some(
                        serde_json::json!({"inputs_identical": false, "differing_buffer": b.name}),
                    );
                }
                _ => {
                    inputs.insert(b.name.clone(), bytes);
                }
            }
        }
    }
    let f32_of = |n: &str| -> Option<Vec<f64>> {
        let b = r.case.buffer(n).or_else(|_| c.case.buffer(n)).ok()?;
        match b.dtype.as_str() {
            "f32" => Some(
                inputs[n]
                    .chunks_exact(4)
                    .map(|w| f32::from_le_bytes(w.try_into().unwrap()) as f64)
                    .collect(),
            ),
            "bf16" => Some(
                inputs[n]
                    .chunks_exact(2)
                    .map(|w| f32::from_bits((u16::from_le_bytes([w[0], w[1]]) as u32) << 16) as f64)
                    .collect(),
            ),
            "f16" => Some(
                inputs[n]
                    .chunks_exact(2)
                    .map(|w| f16_to_f32(u16::from_le_bytes([w[0], w[1]])) as f64)
                    .collect(),
            ),
            _ => None,
        }
    };
    let vals: Vec<Option<Vec<f64>>> = names
        .iter()
        .map(|n| inputs.contains_key(n).then(|| f32_of(n)).flatten())
        .collect();
    ptxv_sim::softmax::ATTN_INPUT_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let x = |(b, i): ptxv_sim::equiv::Leaf| -> f64 {
        vals.get(b as usize)
            .and_then(|v| v.as_ref())
            .and_then(|v| v.get(i as usize))
            .copied()
            .unwrap_or(f64::NAN)
    };
    let (mut max_frac, mut worst, mut max_diff, mut with_bound) = (0.0f64, None, 0.0f64, 0);
    let mut per = Vec::with_capacity(gr.len());
    // 16-bit output form: outputs whose bound is 0 / 1 / more steps of the format.
    let mut steps = [0usize; 3];
    for (i, (a, b)) in gr.iter().zip(&gc).enumerate() {
        let d = (*a as f64 - *b as f64).abs();
        max_diff = max_diff.max(d);
        let bound = rep
            .per_output
            .get(i)
            .filter(|_| rep_bounds)
            .and_then(|o| o.bound_at(&x))
            .filter(|b| b.is_finite());
        if let Some(bd) = bound {
            with_bound += 1;
            let f = if bd > 0.0 {
                d / bd
            } else if d == 0.0 {
                0.0
            } else {
                f64::INFINITY
            };
            if f > max_frac || worst.is_none() {
                max_frac = max_frac.max(f);
                worst = Some(i);
            }
        }
        per.push(bound);
        if let Some(k) = rep
            .per_output
            .get(i)
            .filter(|_| rep_bounds)
            .and_then(|o| o.round_steps_at(&x))
        {
            steps[(k as usize).min(2)] += 1;
        }
    }
    Some(serde_json::json!({
        "round16_steps": (steps.iter().sum::<usize>() > 0).then_some(steps),
        "inputs_identical": true,
        "outputs": gr.len(),
        "bitwise_equal": gr.iter().zip(&gc).filter(|(a, b)| a.to_bits() == b.to_bits()).count(),
        "max_abs_diff": max_diff,
        "outputs_with_bound": with_bound,
        "max_fraction_of_bound": (with_bound > 0).then_some(max_frac),
        "worst_output": worst,
        "bound_abs": per,
    }))
}

fn output_scope(reference: &Run, candidate: &Run) -> serde_json::Value {
    serde_json::json!({
        "reference": {"checked": reference.case.m.outputs[0], "unchecked": reference.case.m.outputs[1..]},
        "candidate": {"checked": candidate.case.m.outputs[0], "unchecked": candidate.case.m.outputs[1..]},
    })
}

pub fn cmd_equiv(
    path: &Path,
    out: &Option<PathBuf>,
    no_hw: bool,
    n_trials: Option<usize>,
    all_outputs: bool,
    no_sweep_cache: bool,
    emit_bound: Option<&Path>,
) -> Result<(bool, Vec<serde_json::Value>)> {
    let base = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let mut f: EquivFile = toml::from_str(&std::fs::read_to_string(path)?)?;
    f.no_hw |= no_hw;
    if f.elementwise == Some(true) {
        f.segments = true;
    }
    if let Some(n) = n_trials {
        f.trials = n;
    }
    let cache = f.sweep_cache.as_ref().map(|c| base.join(c));
    if let (Some(c), false) = (&cache, no_sweep_cache) {
        ptxv_sim::softmax::load_sweeps(c);
    }
    let ew_cache = cache.as_ref().map(|c| c.with_extension("elementwise.json"));
    if let Some(c) = &ew_cache {
        ptxv_sim::elementwise::load_cache(c);
    }
    let dir = match out {
        Some(d) => d.clone(),
        // Named after the fixture directory too: every fixture's file is `equiv.toml`.
        None => std::env::temp_dir().join(format!(
            "ptxv-equiv-{}-{}",
            base.file_name().map_or("".into(), |d| d.to_string_lossy()),
            path.file_stem().unwrap().to_string_lossy()
        )),
    };
    let dir = match ptxv_sim::dist::get() {
        Some(r) => dir.join(format!("rank{}", r.rank)),
        None => dir,
    };
    std::fs::create_dir_all(&dir)?;
    let r = prepare(&base, &f.reference, f.max_states, &dir, f.segments)?;
    println!("reference {}", r.label());
    for p in &r.pre {
        println!("    precondition: {p}");
    }
    for n in &r.notes {
        println!("    {n}");
    }
    let mut all_ok = true;
    let mut json = Vec::new();
    let mut bounds = Vec::new();
    for c in &f.candidates {
        let cand = prepare(&base, c, f.max_states, &dir, f.segments)?;
        println!("\ncandidate {}", cand.label());
        if r.case.m.outputs.len() > 1 || cand.case.m.outputs.len() > 1 {
            println!(
                "    equivalence scope: primary output buffers {} and {}; unchecked reference buffers {:?}, candidate buffers {:?}",
                r.case.m.outputs[0],
                cand.case.m.outputs[0],
                &r.case.m.outputs[1..],
                &cand.case.m.outputs[1..]
            );
        }

        for n in &cand.notes {
            println!("    {n}");
        }
        for p in &cand.pre {
            println!("    precondition: {p}");
        }
        let mut names: Vec<String> =
            r.m.launch
                .allocs
                .iter()
                .chain(cand.m.launch.allocs.iter())
                .map(|a| a.name.clone())
                .collect();
        names.sort();
        names.dedup();
        // Leaves are (buffer name, element index): an input both kernels read must be the
        // same tensor in both cases.
        let mut pre_pair = Vec::new();
        for b in &r.case.m.buffers {
            if let Some(o) = cand.case.m.buffers.iter().find(|o| o.name == b.name) {
                // Same-named workspaces (`W0`: neither side's captured input) are unrelated.
                if b.init != "file"
                    && o.init != "file"
                    && !r.case.m.outputs.contains(&b.name)
                    && !cand.case.m.outputs.contains(&o.name)
                {
                    continue;
                }
                // Leading unit dimensions do not change the row-major layout ([1, 1, 128, 64]
                // and [1, 128, 64] are the same tensor).
                let squeeze =
                    |s: &[u64]| -> Vec<u64> { s.iter().copied().skip_while(|&d| d == 1).collect() };
                // Byte buffers (FP8 codes, scale factors): leaves are byte offsets (halfin
                // `half_mem`), so u8 [1024] and u16 [2, 256] are the same bytes.
                let byte = |d: &str| matches!(d, "u8" | "u16");
                let nbytes = |x: &ptxv_case::Buffer| x.bytes();
                if byte(&b.dtype) && byte(&o.dtype) && nbytes(b) == nbytes(o) {
                    continue;
                }
                if (b.dtype.as_str(), squeeze(&b.shape)) != (o.dtype.as_str(), squeeze(&o.shape)) {
                    pre_pair.push(format!(
                        "buffer {} is {} {:?} in the reference but {} {:?} in the candidate",
                        b.name, b.dtype, b.shape, o.dtype, o.shape
                    ));
                }
            }
        }
        for p in &pre_pair {
            println!("    precondition: {p}");
        }
        // PTXV_EQUIV_DUMP_NODES="r:ID,c:ID,...": dump numerics nodes (debugging).
        if let Ok(v) = std::env::var("PTXV_EQUIV_DUMP_NODES") {
            for it in v.split(',') {
                let (who, id) = it.split_once(':').unwrap_or(("r", it));
                let run = if who == "c" { &cand } else { &r };
                if let Ok(id) = id.parse::<u32>() {
                    let ns = run.m.num.as_ref().unwrap();
                    if (id as usize) < ns.nodes.len() {
                        println!(
                            "    node {who}:{id} (pc {}): {}",
                            run.m.k.describe(ns.nodes[id as usize].pc as usize),
                            ns.dump(id, 4)
                        );
                        if let ptxv_sim::num::NodeKind::Seg(p) = &ns.nodes[id as usize].kind {
                            for o in &p.0.ops {
                                println!("        {:?}", o.kind);
                            }
                        }
                    }
                }
            }
        }
        if let Some(i) = std::env::var("PTXV_EQUIV_DUMP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            for (who, run) in [("reference", &r), ("candidate", &cand)] {
                let ns = run.m.num.as_ref().unwrap();
                let o = &run.outs;
                if let Some((_, Some(n))) = o.get(i) {
                    println!("    dump {who} out[{i}]: {}", ns.dump(*n, 6));
                }
                for (pc, e) in &ns.seg_fails {
                    println!(
                        "    {who}: no slice at {}: {e}",
                        run.m.k.describe(*pc as usize)
                    );
                }
            }
        }
        let rep =
            ptxv_sim::equiv::compare(&side(&r, &names), &side(&cand, &names), &names, f.tau_k);
        let proven = r.pre.is_empty()
            && cand.pre.is_empty()
            && pre_pair.is_empty()
            && rep.untracked.is_none();
        for l in &rep.levels {
            println!(
                "    {:<24} {}{}",
                l.level,
                if l.holds {
                    if proven {
                        "PROVEN".to_string()
                    } else {
                        "holds (preconditions not all met: for the captured inputs only)".into()
                    }
                } else {
                    format!(
                        "FAILS at {} of {} outputs (first: {})",
                        l.failures,
                        rep.outputs,
                        l.first_failure.unwrap()
                    )
                },
                l.detail
                    .as_ref()
                    .filter(|_| !l.holds)
                    .map(|d| format!("\n        {d}"))
                    .unwrap_or_default()
            );
        }
        if rep.l0_zero_sign > 0 {
            println!(
                "    L0 at {} outputs is up to the sign of zero: one kernel's output is the other's \
                 plus +0 (equal bits, except -0 there is +0)",
                rep.l0_zero_sign
            );
        }
        if let Some(u) = &rep.untracked {
            println!("    untracked: {u}");
        }
        if let Some(u) = &rep.unordered {
            println!("    schedule-dependent order: {u}");
        }
        if let Some(u) = &rep.softmax {
            println!("    {u}");
        }
        for c in &rep.conditions {
            println!("    condition: {c}");
        }
        println!("    per output (level, bound achieved |C - R| <= k u sum_m a_m |x_m|):");
        print_outputs(&rep, all_outputs);
        let max_k = rep.max_bound_u();
        if let Some(k) = max_k {
            let extra = rep
                .per_output
                .iter()
                .map(|o| o.abs_extra)
                .fold(0.0, f64::max);
            let scale = if rep.per_output.iter().all(|o| o.form == "softmax") {
                " (absolute: softmax outputs are <= 1)"
            } else if rep
                .per_output
                .iter()
                .all(|o| o.form == "attn" && o.score.is_some())
            {
                " * (1 + Q) max|v_j| (attention form, computed scores: Q = max_j sum_d |q_d k_jd|)"
            } else if rep.per_output.iter().all(|o| o.form == "attn") {
                " * max|v_j| (attention form)"
            } else if rep.per_output.iter().all(|o| o.form == "norm") {
                " * T(x) (norm form: T = sum_m |c_m x_m| rho^e_m)"
            } else {
                " * sum|terms|"
            };
            // Norm form without kappa <= 1: the compact bound covers kappa <= kappa_max.
            let kmax = rep
                .per_output
                .iter()
                .filter_map(|o| o.norm.as_ref())
                .filter(|n| !n.kappa_le_1)
                .map(|n| n.kappa_max)
                .fold(f64::INFINITY, f64::min);
            let qmax = rep
                .per_output
                .iter()
                .filter_map(|o| o.score.map(|s| s.1))
                .fold(f64::INFINITY, f64::min);
            let inputs = if kmax.is_finite() {
                format!(
                    "every input with kappa <= {kmax:.0} (beyond: the exact per-input bound, for A's relative error <= 1/2)"
                )
            } else if qmax.is_finite() {
                format!("every input with Q <= {qmax:.0} (beyond: the exact per-input bound)")
            } else {
                "every input".into()
            };
            let round_all = rep.per_output.iter().all(|o| o.form == "round");
            let k_txt = if round_all {
                let add = rep
                    .per_output
                    .iter()
                    .filter_map(|o| o.add_u)
                    .fold(0.0, f64::max);
                let rel = rep
                    .per_output
                    .iter()
                    .filter_map(|o| o.rel_u)
                    .fold(0.0, f64::max);
                format!("{rel:.0}u * |y| + {add:.2}u")
            } else {
                format!("{k:.2}u")
            };
            let scale = if round_all {
                " * sum|terms| (16-bit outputs: y the exact value, 2 u16 |y| for the two output roundings; per input: the distance between the roundings of [y - G_R, y + G_C])"
            } else {
                scale
            };
            println!(
                "    bound: |C - R| <= {k_txt}{scale}{} for every output and {inputs}{}",
                if extra > 0.0 {
                    format!(" + {extra:.1e}")
                } else {
                    String::new()
                },
                if proven {
                    " (PROVEN)"
                } else {
                    " (preconditions not all met)"
                }
            );
        }
        // Untracked outputs are constants to the comparison: their "bounds" mean nothing.
        let hw_trials = std::env::var("PTXV_EQUIV_HW_TRIALS").is_ok();
        let tr = if (f.trials > 0 || hw_trials)
            && rep.untracked.is_none()
            && rep.per_output.iter().any(|o| o.bound_u.is_some())
        {
            let t = trials(&r, &cand, &rep, &names, f.trials, &dir)?;
            if std::env::var("PTXV_EQUIV_HW_TRIALS").is_ok() {
                println!("    (trials: GB300 inputs and outputs from PTXV_EQUIV_HW_TRIALS)");
            }
            println!(
                "    trials: {} random inputs and schedules, {} output-trials, {} over the bound\n      \
                 |C - R| used at most {:.4} of the bound (observed k {:.2}u, proven {:.2}u)\n      \
                 each kernel vs exact: at most {:.4} (reference) / {:.4} (candidate) of its own \
                 bound; {} outputs differed bitwise in some trial",
                t.trials,
                t.checked,
                t.violations,
                t.max_used,
                t.max_observed_u,
                max_k.unwrap_or(f64::NAN),
                t.max_used_ref,
                t.max_used_cand,
                t.differing
            );
            if let Some(k) = t.max_kappa {
                println!(
                    "      norm form: kappa (A's conditioning) at most {k:.2} over the trials; {} output-trials not covered",
                    t.uncovered
                );
            }
            all_ok &= t.violations == 0;
            Some(t)
        } else {
            None
        };
        let cap = captured(&r, &cand, &rep, &names);
        if let Some(cv) = &cap {
            if cv["inputs_identical"] == false {
                println!(
                    "    captured GPU outputs: not compared (input buffer {} differs between the cases)",
                    cv["differing_buffer"]
                );
            } else {
                println!(
                    "    captured GPU outputs: {} of {} bitwise equal, max |C - R| {:.3e}{}",
                    cv["bitwise_equal"],
                    cv["outputs"],
                    cv["max_abs_diff"].as_f64().unwrap_or(f64::NAN),
                    cv["max_fraction_of_bound"]
                        .as_f64()
                        .map(|f| format!(
                            "; at most {f:.4} of the bound at the captured inputs (out[{}])",
                            cv["worst_output"]
                        ))
                        .unwrap_or_default()
                );
                if let Some(st) = cv["round16_steps"].as_array() {
                    println!(
                        "      16-bit outputs at the captured inputs: the bound proves bitwise equality at {}, at most 1 ulp at {}, more at {} outputs",
                        st[0], st[1], st[2]
                    );
                }
            }
        }
        let mut witness = None;
        if let Some((i, mono, cr, cc)) = &rep.witness_mono {
            let leaves: Vec<(String, u64)> = mono
                .iter()
                .map(|(b, e)| (names[*b as usize].clone(), *e))
                .collect();
            println!(
                "    witness: output {i}, inputs zero except {} = 1 (coefficient {cr} vs {cc})",
                leaves
                    .iter()
                    .map(|(n, e)| format!("{n}[{e}]"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            match (
                replay(&r, &leaves, *i, &dir, f.no_hw),
                replay(&cand, &leaves, *i, &dir, f.no_hw),
            ) {
                (Ok((rs, rh)), Ok((cs, ch))) => {
                    witness = Some(serde_json::json!({
                        "output": i,
                        "leaves": leaves.iter().map(|(n, e)| format!("{n}[{e}]")).collect::<Vec<_>>(),
                        "coefficient_reference": cr,
                        "coefficient_candidate": cc,
                        "sim_reference": rs,
                        "sim_candidate": cs,
                        "gpu_reference": rh,
                        "gpu_candidate": ch,
                    }));
                    println!("        simulator: reference {rs}, candidate {cs}");
                    if let (Some(rh), Some(ch)) = (rh, ch) {
                        println!(
                            "        GB300:     reference {rh}, candidate {ch}{}",
                            if rh != ch {
                                "  (differ: not equivalent)"
                            } else {
                                ""
                            }
                        );
                    }
                }
                (a, b) => println!("        replay failed: {:?} {:?}", a.err(), b.err()),
            }
        }
        // Elementwise mode: every output one input element's function, swept exhaustively.
        let mut ew_json = serde_json::Value::Null;
        let mut ew_level = None;
        if f.segments && f.elementwise != Some(false) {
            match elementwise_pair(&r, &cand, f.function.as_deref(), f.elementwise_gpu, &dir) {
                Ok((j, lvl)) => {
                    ew_level = Some(lvl);
                    ew_json = j;
                }
                Err(e) => {
                    println!("    elementwise: not applicable: {e}");
                    if f.elementwise == Some(true) {
                        pre_pair.push(format!("elementwise: {e}"));
                    }
                    ew_json = serde_json::json!({"error": e});
                }
            }
        }
        let proven = proven && pre_pair.is_empty();
        let got = match ew_level.clone() {
            Some(l) if r.pre.is_empty() && cand.pre.is_empty() && pre_pair.is_empty() => l,
            Some(l) => format!("{l} (unproven)"),
            None => strongest(&rep) + if proven { "" } else { " (unproven)" },
        };
        let proven = proven || (ew_level.is_some() && !got.ends_with("(unproven)"));
        println!("    => strongest level: {got}");
        if let Some(e) = &c.expect {
            let ok = *e == got;
            println!("    expected {e}: {}", if ok { "ok" } else { "MISMATCH" });
            all_ok &= ok;
        }
        if let Some(e) = c.expect_bound_u {
            let ok = max_k.is_some_and(|k| k <= e);
            println!(
                "    expected bound <= {e}u: {}",
                if ok { "ok" } else { "MISMATCH" }
            );
            all_ok &= ok;
        }
        if emit_bound.is_some() {
            bounds.push(bound_json(&rep, &names, &r, &cand, proven, &got));
        }
        json.push(serde_json::json!({
            "reference": r.label(),
            "candidate": cand.label(),
            "output_scope": output_scope(&r, &cand),
            "strongest": got,
            "proven": proven,
            "preconditions": r.pre.iter().chain(&cand.pre).collect::<Vec<_>>(),
            "relies_on": r.notes.iter().chain(&cand.notes).collect::<Vec<_>>(),
            "untracked_reference": untracked_count(&r.m),
            "untracked_candidate": untracked_count(&cand.m),
            "notes_reference": r.notes,
            "notes_candidate": cand.notes,
            "pre_reference": r.pre,
            "pre_candidate": cand.pre,
            "pre_pair": pre_pair,
            "max_bound_u": max_k,
            "elementwise": ew_json,
            "report": rep,
            "trials": tr,
            "captured": cap,
            "witness": witness,
        }));
    }
    if let Some(c) = &cache {
        ptxv_sim::softmax::save_sweeps(c)?;
    }
    if let Some(c) = &ew_cache {
        ptxv_sim::elementwise::save_cache(c)?;
    }
    let jp = dir.join("equiv.json");
    std::fs::write(&jp, serde_json::to_string_pretty(&json)?)?;
    println!("\nJSON summary: {}", jp.display());
    if let Some(p) = emit_bound {
        // One document per candidate; a single candidate is written bare.
        let v = if bounds.len() == 1 {
            bounds.pop().unwrap()
        } else {
            serde_json::Value::Array(bounds)
        };
        std::fs::write(p, serde_json::to_string(&v)?)?;
        println!("bound: {}", p.display());
    }
    Ok((all_ok, json))
}

/// The per-output bound as data (`--emit-bound`, docs/json.md "ptxv-bound"): enough to
/// evaluate `OutBound::bound_at` on other inputs without ptxv. Forms:
/// - "sum": sum_m |prod x_m| g_m + abs_extra, times (1 + 1e-12);
/// - "round": round_rel |y| + sum_m |prod x_m| g_m + abs_extra, y = sum_m c_m prod x_m (the
///   compact form, valid for every input; `bound_at` uses the tighter rounding gap);
/// - "softmax": ((rel_u y + add_u) u + abs_extra)(1 + 1e-9), y = exp(x_i - max) / sum_j
///   exp(x_j - max) over the row;
/// - "none": no bound (`no_bound` says why); "opaque": a bound exists (norm, attention) but
///   its evaluation data is not exported; `bound_u` is its input-independent coefficient.
///
/// Terms are listed output by output; a term's leaves are `deg` consecutive entries.
fn bound_json(
    rep: &Report,
    names: &[String],
    r: &Run,
    cand: &Run,
    proven: bool,
    level: &str,
) -> serde_json::Value {
    use serde_json::json;
    let n = rep.per_output.len();
    let (mut form, mut abs_extra, mut bound_u, mut round_rel, mut lvl, mut why) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    let (mut t_out, mut t_g, mut t_c, mut t_deg, mut l_buf, mut l_idx) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    let mut rows: Vec<Vec<ptxv_sim::equiv::Leaf>> = vec![];
    let mut row_of: std::collections::HashMap<Vec<ptxv_sim::equiv::Leaf>, usize> =
        Default::default();
    let (mut s_out, mut s_row, mut s_buf, mut s_idx, mut s_rel, mut s_add) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    let untracked = rep.untracked.is_some();
    for (i, o) in rep.per_output.iter().enumerate() {
        lvl.push(o.level);
        abs_extra.push(o.abs_extra);
        bound_u.push(o.bound_u);
        why.push(o.no_bound.clone());
        let f = if untracked || o.bound_u.is_none() {
            "none"
        } else if o.norm.is_some() || o.attn.is_some() {
            "opaque"
        } else if let (Some((row, leaf)), Some(rel), Some(add)) = (&o.sm, o.rel_u, o.add_u) {
            let k = *row_of.entry(row.clone()).or_insert_with(|| {
                rows.push(row.clone());
                rows.len() - 1
            });
            s_out.push(i);
            s_row.push(k);
            s_buf.push(leaf.0);
            s_idx.push(leaf.1);
            s_rel.push(rel);
            s_add.push(add);
            "softmax"
        } else if o.form == "softmax" || o.form == "attn" || o.form == "norm" {
            "opaque"
        } else {
            for (m, _, g, c) in &o.terms {
                t_out.push(i);
                t_g.push(*g);
                t_c.push(*c);
                t_deg.push(m.len());
                for &(b, e) in m {
                    l_buf.push(b);
                    l_idx.push(e);
                }
            }
            if o.round_f16.is_some() {
                "round"
            } else {
                "sum"
            }
        };
        form.push(f);
        round_rel.push(if f == "round" { o.round_rel } else { None });
    }
    let buf = |name: &str| {
        r.case
            .buffer(name)
            .or_else(|_| cand.case.buffer(name))
            .ok()
            .map(|b| json!({"dtype": b.dtype, "shape": b.shape}))
    };
    let out_name = r.case.m.outputs.first().cloned();
    json!({
        "format": "ptxv-bound",
        "version": 1,
        "u": ptxv_sim::num::U32,
        "output_scope": output_scope(r, cand),
        "reference": r.label(),
        "candidate": cand.label(),
        "level": level,
        "proven": proven,
        "output": {
            "name": out_name,
            "count": n,
            "buffer": out_name.as_deref().and_then(buf),
        },
        "inputs": names.iter().map(|nm| json!({"name": nm, "buffer": buf(nm)})).collect::<Vec<_>>(),
        "untracked": rep.untracked,
        "per_output": {
            "level": lvl,
            "form": form,
            "bound_u": bound_u,
            "abs_extra": abs_extra,
            "round_rel": round_rel,
            "no_bound": why,
        },
        "terms": {
            "out": t_out,
            "g": t_g,
            "coef": t_c,
            "deg": t_deg,
            "leaf_buf": l_buf,
            "leaf_idx": l_idx,
        },
        "softmax": {
            "out": s_out,
            "row": s_row,
            "leaf_buf": s_buf,
            "leaf_idx": s_idx,
            "rel_u": s_rel,
            "add_u": s_add,
            "rows": rows.iter().map(|r| json!({
                "leaf_buf": r.iter().map(|l| l.0).collect::<Vec<_>>(),
                "leaf_idx": r.iter().map(|l| l.1).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        },
    })
}

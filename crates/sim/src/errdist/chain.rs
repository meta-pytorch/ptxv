/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The numerics DAG of one kernel as a program over input leaves: what errdist evaluates on
//! every sampled input (Tier 3), and what the analytic tiers read their step structure from.
//!
//! Extraction walks each output's node back to the input leaves (`num.rs`). Under N-EQUIV's
//! preconditions (an exhaustive check and the taint check) that DAG is the kernel's dataflow
//! for every input, so evaluating it with the calibrated models is the kernel. Each program is
//! checked against the simulator's node values at the captured input (`check_captured`) and,
//! in validation, against simulator runs on sampled inputs.

use std::collections::BTreeMap;

use super::spec::Format;
use crate::errdist::det::HashMap;
use crate::errdist::det::HashSet;
use crate::machine::Machine;
use crate::mma_lp::Operand;
use crate::num::NodeKind;

/// The arithmetic of one tensor-core step.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum StepModel {
    /// kind::f16 / HMMA (`fp::MmaModel`, N-MMA).
    Mma(crate::fp::MmaModel),
    /// kind::f8f6f4 / mxf8f6f4 (`mma_lp::LpMmaModel`, N-MMA-F8 / N-MMA-BS).
    Lp(crate::mma_lp::LpMmaModel),
    /// kind::mxf4 / mxf4nvf4 (`mma_lp::Fp4Model`, N-MMA-BS).
    Fp4(crate::mma_lp::Fp4Model),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Op {
    /// Input element: buffer id, element index (narrow codes: code index), format.
    Leaf {
        buf: u16,
        idx: u64,
        fmt: Format,
    },
    Const(f64),
    Mma {
        model: StepModel,
        acc: Option<u32>,
        /// Factor leaves of each product.
        terms: Vec<(u32, u32)>,
        /// Scale leaves of each product (block-scaled kinds), else empty.
        scales: Vec<(u32, u32)>,
        /// FP4: unit of each product.
        units: Vec<u8>,
    },
    /// Round to nearest even to bf16 (f16 with `f16`).
    Round {
        arg: u32,
        f16: bool,
    },
    /// f32 add rounded to nearest (toward zero with `rz`); `ftz`: subnormal operands and
    /// result flushed to zero (an fp32 atomic add's `.FTZ`).
    Add {
        a: u32,
        b: u32,
        rz: bool,
        #[serde(default)]
        ftz: bool,
    },
    Fma {
        a: u32,
        b: u32,
        c: Option<u32>,
        neg_ab: bool,
        neg_c: bool,
    },
    /// Agent AG: a non-chain op (instruction slice, MUFU, maximum), `nonlin::ops`.
    Nl(super::nonlin::ops::NlOp),
    /// A TF32 part of an f32 value (`num::Tf32Part`): hi(x) (low 13 bits cleared), lo(x) =
    /// x - hi(x), lo rounded half away (LoR) or truncated (LoT) to 11 significant bits.
    Tf32 {
        arg: u32,
        part: crate::num::Tf32Part,
    },
}

/// An op's operands (program op indices).
pub fn op_args(op: &Op) -> Vec<u32> {
    match op {
        Op::Leaf { .. } | Op::Const(_) => vec![],
        Op::Nl(nl) => nl.args(),
        Op::Round { arg, .. } | Op::Tf32 { arg, .. } => vec![*arg],
        Op::Add { a, b, .. } => vec![*a, *b],
        Op::Fma { a, b, c, .. } => [Some(*a), Some(*b), *c].into_iter().flatten().collect(),
        Op::Mma {
            acc, terms, scales, ..
        } => acc
            .iter()
            .copied()
            .chain(terms.iter().chain(scales).flat_map(|&(x, y)| [x, y]))
            .collect(),
    }
}

/// The f32 value of a TF32 part of x.
pub fn tf32_part(x: f32, part: crate::num::Tf32Part) -> f32 {
    use crate::num::Tf32Part;
    let hi = f32::from_bits(x.to_bits() & 0xffff_e000);
    match part {
        Tf32Part::Hi => hi,
        Tf32Part::Lo => x - hi,
        Tf32Part::LoR => f32::from_bits((x - hi).to_bits().wrapping_add(0x1000) & 0xffff_e000),
        Tf32Part::LoT => f32::from_bits((x - hi).to_bits() & 0xffff_e000),
    }
}

/// One kernel's numerics as a program.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Prog {
    /// Children before parents.
    pub ops: Vec<Op>,
    /// The simulator's value of each op at the captured input (f32 bits).
    pub hw: Vec<u32>,
    /// Per output element: its op (None: not an input-dependent value).
    pub outputs: Vec<Option<u32>>,
    /// Output element format (f32, bf16, f16).
    pub out_fmt: Format,
    /// Buffer names of leaves.
    pub bufs: Vec<String>,
    /// Output elements without a DAG whose bytes are input-derived (the simulator's taint):
    /// errdist refuses rather than report on the other outputs alone.
    #[serde(default)]
    pub dep_untracked: Vec<u32>,
    /// The adds of fp32 atomics (`NodeKind::AtomAdd` / `AtomAdd16`): the DAG takes the
    /// simulator's order of each location's adds, GB300 its own (N-ERRDIST-DAG).
    #[serde(default)]
    pub atomic: Vec<u32>,
    /// Per entry of `atomic`, the source that issued the add (`NumState::atom_src`: a thread,
    /// or a bulk reduction with the top bit set; `u32::MAX` unknown).
    #[serde(default)]
    pub atomic_src: Vec<u32>,
    /// Per entry of `atomic`, whether the run's happens-before leaves the add unordered with
    /// the previous add on its location (`NumState::atom_free`, MISS 84 follow-up): only then
    /// can GB300 add the two in the other order.
    #[serde(default)]
    pub atomic_free: Vec<bool>,
    /// Per output, whether `pre_round` took it before its rounding (empty: none): the
    /// kernel's value differs from the DAG's there by design, and the validations do not
    /// compare it bit for bit.
    #[serde(default)]
    pub pre_rounded: Vec<bool>,
}

/// An output whose value is the last add of an atomic chain on its location
/// (`Prog::atomic_chains`).
#[derive(Clone, Debug, PartialEq)]
pub struct AtomChain {
    /// The chain's add ops, last first.
    pub adds: Vec<u32>,
    /// The op of the location's value before the first add.
    pub init: u32,
    /// The sources that issued the adds (threads, bulk operations).
    pub sources: usize,
    /// Some two adds of the chain, adjacent in the run's order, that no synchronization of the
    /// run orders (`Prog::atomic_free`): without them the chain has one order, the DAG's.
    pub free: bool,
    /// Unit roundoff of one add: 2^-24 (f32), 2^-8 (bf16), 2^-11 (f16).
    pub u: f64,
    /// Absolute error of one add below the normal range: a flush to zero (FTZ, 2^-126), else
    /// half the smallest subnormal.
    pub eta: f64,
}

impl AtomChain {
    /// Whether another order of the adds explains GB300's value `gpu` where the DAG, in the
    /// simulator's order, gives `dag`, at an input whose op values `v` gives (MISS 58,
    /// narrowed): the adds come from two or more sources, not one thread in program order;
    /// some two of them are unordered by the run's happens-before (`free`: a barrier, an
    /// mbarrier, a release / acquire pair or a launch boundary fixes GB300's order too); they
    /// add three or more operands counting a nonzero initial value (two operands from zero
    /// round once in either order); and the two values are within twice the bound on any
    /// order's error, gamma_k * sum |operand| + k * eta for k adds (recursive summation,
    /// Higham 4.4; each add's flush or subnormal rounding adds at most eta). A value outside
    /// it is no order's.
    pub fn order_explains(&self, p: &Prog, v: &dyn Fn(u32) -> f64, dag: f32, gpu: f32) -> bool {
        if self.sources < 2 || !self.free || !dag.is_finite() || !gpu.is_finite() {
            return false;
        }
        let init = v(self.init);
        let k = self.adds.len();
        if k + usize::from(init != 0.0) < 3 {
            return false;
        }
        let mut s = init.abs();
        for &a in &self.adds {
            if let Op::Add { b, .. } = p.ops[a as usize] {
                s += v(b).abs();
            }
        }
        let ku = k as f64 * self.u;
        if ku >= 0.5 {
            // gamma_k has no bound (k u >= 1/2: bf16 chains of 128 adds and more, f16 of 1024,
            // f32 of 2^23): nothing tells another order from an error, so nothing is excused
            // (until the MISS 84 follow-up, any value was).
            return false;
        }
        let bound = 2.0 * (ku / (1.0 - ku) * s + k as f64 * self.eta);
        (gpu as f64 - dag as f64).abs() <= bound
    }
}

/// Decoded leaf values of one sample: per op (leaves only), value and alignment exponents.
pub type LeafVals = Vec<Operand>;

/// Per code buffer: the scale buffer and, per scale byte, the code indices it multiplies
/// (`Prog::scale_blocks`).
pub type ScaleBlocks = BTreeMap<String, (String, Vec<(u64, Vec<u64>)>)>;

impl Prog {
    /// Extract from a machine after a numerics run (`Machine::num` set, `half_out_leaves`
    /// called).
    pub fn extract(m: &Machine) -> Result<Prog, String> {
        let ns = m.num.as_ref().ok_or("no numerics run")?;
        let oi = *m.launch.outputs.first().ok_or("no output buffer")?;
        let mem = &m.gmem[oi];
        let f16 = ns.half.f16_outputs.contains(&oi);
        let bf16 = ns.half.bf16_outputs.contains(&oi);
        let roots: Vec<Option<u32>> = if f16 || bf16 {
            (0..mem.bytes.len() / 2)
                .map(|i| m.half_out_node(oi, i as u64, f16))
                .collect()
        } else {
            (0..mem.bytes.len() / 4)
                .map(|i| {
                    ns.mem_node(crate::exec::Loc::Global {
                        alloc: oi,
                        off: (i * 4) as u64,
                    })
                })
                .collect()
        };
        let mut p = Prog {
            ops: vec![],
            hw: vec![],
            outputs: vec![],
            out_fmt: if f16 {
                Format::F16
            } else if bf16 {
                Format::Bf16
            } else {
                Format::F32
            },
            bufs: m.launch.allocs.iter().map(|a| a.name.clone()).collect(),
            dep_untracked: vec![],
            atomic: vec![],
            atomic_src: vec![],
            atomic_free: vec![],
            pre_rounded: vec![],
        };
        let mut map: HashMap<u32, u32> = HashMap::default();
        let model = m.launch.mma_model;
        let w = if f16 || bf16 { 2 } else { 4 };
        for (i, r) in roots.into_iter().enumerate() {
            let o = match r {
                Some(n) => Some(p.convert(m, n, &mut map, model)?),
                None => None,
            };
            if o.is_none()
                && (0..w).any(|b| ns.taint.byte_tainted((1, oi as u64, (w * i + b) as u64)))
            {
                p.dep_untracked.push(i as u32);
            }
            p.outputs.push(o);
        }
        Ok(p)
    }

    fn convert(
        &mut self,
        m: &Machine,
        root: u32,
        map: &mut HashMap<u32, u32>,
        model: crate::fp::MmaModel,
    ) -> Result<u32, String> {
        let ns = m.num.as_ref().unwrap();
        // Iterative post-order.
        let mut stack = vec![(root, false)];
        while let Some((n, done)) = stack.pop() {
            if map.contains_key(&n) {
                continue;
            }
            let node = &ns.nodes[n as usize];
            let kids: Vec<u32> = match &node.kind {
                NodeKind::Mma { acc, ops, sc, .. } => {
                    let mut k: Vec<u32> = acc.iter().copied().collect();
                    for o in ops {
                        let (a, b) = o.ok_or("an MMA product with an untracked factor")?;
                        k.extend([a, b]);
                    }
                    for (a, b) in sc {
                        k.extend([*a, *b]);
                    }
                    k
                }
                NodeKind::Round { arg, .. } => vec![*arg],
                NodeKind::Add(a, b, _)
                | NodeKind::AtomAdd(a, b, _)
                | NodeKind::AtomAdd16(a, b, _) => {
                    vec![*a, *b]
                }
                NodeKind::Tf32 { arg, .. } => vec![*arg],
                NodeKind::Fma { a, b, c, .. } => {
                    [Some(*a), Some(*b), *c].into_iter().flatten().collect()
                }
                NodeKind::Leaf | NodeKind::Input(_) => vec![],
                NodeKind::Opaque => {
                    return Err(format!(
                        "errdist does not model node kind Opaque: an input-derived value without \
                         a node, read at {}{}",
                        m.k.describe(node.pc as usize),
                        seg_fail_note(ns)
                    ));
                }
                k => match super::nonlin::ops::kids(k) {
                    Some(v) => v,
                    None => {
                        return Err(format!("errdist does not model node kind {}", kind_name(k)));
                    }
                },
            };
            if !done {
                stack.push((n, true));
                for k in kids {
                    if !map.contains_key(&k) {
                        stack.push((k, false));
                    }
                }
                continue;
            }
            let g = |x: &u32| map[x];
            if matches!(node.kind, NodeKind::AtomAdd(..) | NodeKind::AtomAdd16(..)) {
                // The add (an `AtomAdd16`'s rounding comes after it).
                self.atomic.push(self.ops.len() as u32);
                self.atomic_src
                    .push(ns.atom_src.get(&n).copied().unwrap_or(u32::MAX));
                self.atomic_free
                    .push(ns.atom_free.get(&n).copied().unwrap_or(false));
            }
            let op = match &node.kind {
                NodeKind::Leaf => Op::Const(f32::from_bits(node.hw) as f64),
                NodeKind::Input((alloc, idx)) => {
                    let (fmt, k) = match crate::equiv::leaf_lp(*idx) {
                        Some((t, k)) => (Format::from_lp(t).ok_or("leaf format")?, k),
                        None => {
                            let f = if ns.f32_inputs.contains(alloc) {
                                Format::F32
                            } else {
                                match ns.half.inputs.get(alloc) {
                                    Some(crate::halfin::Half::Bf16) => Format::Bf16,
                                    Some(crate::halfin::Half::F16) => Format::F16,
                                    _ => {
                                        return Err(format!(
                                            "input leaf of alloc {alloc}: unknown format"
                                        ));
                                    }
                                }
                            };
                            (f, *idx)
                        }
                    };
                    Op::Leaf {
                        buf: *alloc as u16,
                        idx: k,
                        fmt,
                    }
                }
                NodeKind::Mma {
                    acc,
                    ops,
                    sc,
                    units,
                    ..
                } => {
                    let terms: Vec<(u32, u32)> = ops
                        .iter()
                        .map(|o| o.unwrap())
                        .map(|(a, b)| (g(&a), g(&b)))
                        .collect();
                    let scales: Vec<(u32, u32)> = sc.iter().map(|(a, b)| (g(a), g(b))).collect();
                    let lp = terms.iter().any(|(a, _)| {
                        matches!(&self.ops[*a as usize], Op::Leaf { fmt, .. } if !matches!(fmt, Format::Bf16 | Format::F16 | Format::F32))
                    });
                    let st = if !units.is_empty() {
                        StepModel::Fp4(crate::mma_lp::DEFAULT_FP4_MMA)
                    } else if lp {
                        StepModel::Lp(crate::mma_lp::DEFAULT_LP_MMA)
                    } else {
                        StepModel::Mma(model)
                    };
                    Op::Mma {
                        model: st,
                        acc: acc.map(|a| g(&a)),
                        terms,
                        scales,
                        units: units.clone(),
                    }
                }
                NodeKind::Round { arg, f16 } => Op::Round {
                    arg: g(arg),
                    f16: *f16,
                },
                NodeKind::Add(a, b, rz) => Op::Add {
                    a: g(a),
                    b: g(b),
                    rz: *rz,
                    ftz: false,
                },
                // One order of the atomics: the simulator's run (N-ERRDIST-DAG), with the
                // flush of `red/atom.add.f32` (exec.rs `AtomOp::FAddFtz`). Until 2026-11-11
                // the DAG kept subnormal atomic operands and results, which GB300 flushes.
                NodeKind::AtomAdd(a, b, ftz) => Op::Add {
                    a: g(a),
                    b: g(b),
                    rz: false,
                    ftz: *ftz,
                },
                // A 16-bit add as an f32 add then a rounding to 16 bits: the f32 sum of two
                // bf16 (f16) values rounds once more to the correctly rounded 16-bit sum (24
                // >= 2p + 2 significand bits; subnormal sums are exact in f32), so the pair is
                // the hardware's single rounding. In the simulator's order (N-ERRDIST-DAG,
                // N-RED16).
                NodeKind::AtomAdd16(a, b, f16) => {
                    let sum = f32::from_bits(ns.nodes[*a as usize].hw)
                        + f32::from_bits(ns.nodes[*b as usize].hw);
                    self.ops.push(Op::Add {
                        a: g(a),
                        b: g(b),
                        rz: false,
                        ftz: false,
                    });
                    self.hw.push(sum.to_bits());
                    Op::Round {
                        arg: (self.ops.len() - 1) as u32,
                        f16: *f16,
                    }
                }
                NodeKind::Tf32 { arg, part } => Op::Tf32 {
                    arg: g(arg),
                    part: *part,
                },
                NodeKind::Fma {
                    a,
                    b,
                    c,
                    neg_ab,
                    neg_c,
                } => Op::Fma {
                    a: g(a),
                    b: g(b),
                    c: c.map(|c| g(&c)),
                    neg_ab: *neg_ab,
                    neg_c: *neg_c,
                },
                k => Op::Nl(super::nonlin::ops::convert(k, &g).unwrap()),
            };
            self.ops.push(op);
            self.hw.push(node.hw);
            map.insert(n, (self.ops.len() - 1) as u32);
        }
        Ok(map[&root])
    }

    /// Per output: whether its DAG holds an atomic add (`atomic`), whose order GB300 chooses
    /// at run time, anywhere upstream.
    pub fn atomic_fed(&self) -> Vec<bool> {
        let mut fed = vec![false; self.ops.len()];
        for &a in &self.atomic {
            fed[a as usize] = true;
        }
        if !self.atomic.is_empty() {
            for (i, op) in self.ops.iter().enumerate() {
                fed[i] = fed[i] || op_args(op).iter().any(|&k| fed[k as usize]);
            }
        }
        self.outputs
            .iter()
            .map(|o| o.is_some_and(|i| fed[i as usize]))
            .collect()
    }

    /// Per output: the chain of atomic adds whose last add is the output's value (a 16-bit
    /// chain: the rounding after it), where GB300's order of the adds can differ from the
    /// DAG's (MISS 58, narrowed; `AtomChain::order_explains`). None for an output that is not
    /// an atomic location's value: a GB300 value that differs from the DAG's there is not
    /// another order of the adds, whatever its DAG holds upstream.
    pub fn atomic_chains(&self) -> Vec<Option<AtomChain>> {
        if self.atomic.is_empty() {
            return vec![None; self.outputs.len()];
        }
        let at: HashMap<u32, usize> = self
            .atomic
            .iter()
            .enumerate()
            .map(|(i, &a)| (a, i))
            .collect();
        // The last add of the chain whose value op `o` is: itself, or a 16-bit rounding's add.
        let end = |o: u32| -> Option<(u32, Option<bool>)> {
            if at.contains_key(&o) {
                return Some((o, None));
            }
            match &self.ops[o as usize] {
                Op::Round { arg, f16 } if at.contains_key(arg) => Some((*arg, Some(*f16))),
                _ => None,
            }
        };
        self.outputs
            .iter()
            .map(|&o| {
                let (mut cur, bits16) = end(o?)?;
                let mut adds = vec![];
                let mut srcs = std::collections::BTreeSet::new();
                let mut ftz = false;
                let mut free = false;
                let init = loop {
                    adds.push(cur);
                    srcs.insert(self.atomic_src.get(at[&cur]).copied().unwrap_or(u32::MAX));
                    free |= self.atomic_free.get(at[&cur]).copied().unwrap_or(false);
                    let Op::Add { a, ftz: f, .. } = self.ops[cur as usize] else {
                        return None;
                    };
                    ftz |= f;
                    // A location's previous value that is itself an atomic sum (a copy of
                    // another location's): its adds join the chain, whose order changes this
                    // value as well.
                    match end(a) {
                        // Children come before parents: the walk ends.
                        Some((prev, _)) if prev < cur => cur = prev,
                        _ => break a,
                    }
                };
                let (u, eta) = match bits16 {
                    Some(true) => (2f64.powi(-11), 2f64.powi(-25)),
                    Some(false) => (2f64.powi(-8), 2f64.powi(-134)),
                    None if ftz => (2f64.powi(-24), 2f64.powi(-126)),
                    None => (2f64.powi(-24), 2f64.powi(-150)),
                };
                Some(AtomChain {
                    adds,
                    init,
                    sources: srcs.len() - (srcs.len() > 1 && srcs.contains(&u32::MAX)) as usize,
                    free,
                    u,
                    eta,
                })
            })
            .collect()
    }

    /// Number of data-dependent selections (`NlOp::Sel`) in the program.
    pub fn selections(&self) -> usize {
        self.ops
            .iter()
            .filter(|o| matches!(o, Op::Nl(super::nonlin::ops::NlOp::Sel { .. })))
            .count()
    }

    /// Number of instruction slice ops whose identified function stands for the slice on part
    /// of its data-dependent decision patterns only (`ident::Outcomes`, MISS 57).
    pub fn partial_slices(&self) -> usize {
        self.ops
            .iter()
            .filter(|o| {
                matches!(o, Op::Nl(super::nonlin::ops::NlOp::Seg { real: Some(f), .. }) if f.outcomes.is_some())
            })
            .count()
    }

    /// The selections and maxima whose branch can differ between the kernel and the exact
    /// computation (N-ERRDIST-SEL): a decision input (a `Sel`'s predicate arguments, a `Max`'s
    /// operands) is not exact, i.e. depends on a rounding or, with `input_errors`, on a leaf.
    /// The others decide alike on every input. (selections, maxima), op indices.
    pub fn flippable(&self, input_errors: bool) -> (Vec<u32>, Vec<u32>) {
        use super::nonlin::ops::NlOp;
        // Whether an op's value is the same in the kernel and in exact arithmetic.
        let mut exact = vec![false; self.ops.len()];
        let (mut sels, mut maxs) = (vec![], vec![]);
        for (i, op) in self.ops.iter().enumerate() {
            exact[i] = match op {
                Op::Leaf { .. } => !input_errors,
                Op::Const(_) => true,
                Op::Nl(NlOp::Max { a, b, .. }) => {
                    let e = exact[*a as usize] && exact[*b as usize];
                    if !e {
                        maxs.push(i as u32);
                    }
                    e
                }
                Op::Nl(NlOp::Sel { args, a, b, .. }) => {
                    let decided = args.iter().all(|&j| exact[j as usize]);
                    if !decided {
                        sels.push(i as u32);
                    }
                    decided && exact[*a as usize] && exact[*b as usize]
                }
                // Roundings (an exact result on some inputs is not exact by construction), and
                // TF32 parts (hi(x) stands for x).
                _ => false,
            };
        }
        (sels, maxs)
    }

    /// Leaf values decoded from buffer bytes (by buffer name).
    pub fn leaves(&self, bytes: &BTreeMap<String, Vec<u8>>) -> Result<LeafVals, String> {
        let mut out = vec![Operand::from_value(0.0); self.ops.len()];
        for (i, op) in self.ops.iter().enumerate() {
            if let Op::Leaf { buf, idx, fmt } = op {
                let name = &self.bufs[*buf as usize];
                let b = bytes
                    .get(name)
                    .ok_or_else(|| format!("no bytes for buffer {name}"))?;
                out[i] =
                    decode(b, *idx, *fmt).ok_or_else(|| format!("{name}[{idx}] out of range"))?;
            }
        }
        Ok(out)
    }

    /// The element indices the leaves read, per buffer name (a sampling `Plan`'s input).
    pub fn leaf_reads(&self, out: &mut BTreeMap<String, Vec<u64>>) {
        for op in &self.ops {
            if let Op::Leaf { buf, idx, .. } = op {
                out.entry(self.bufs[*buf as usize].clone())
                    .or_default()
                    .push(*idx);
            }
        }
    }

    /// Per op: the leaf's position in a planned sample's compact buffer (`u32::MAX` for
    /// other ops).
    pub fn leaf_positions(&self, plan: &super::spec::Plan) -> Result<Vec<u32>, String> {
        self.ops
            .iter()
            .map(|op| match op {
                Op::Leaf { buf, idx, .. } => {
                    let name = &self.bufs[*buf as usize];
                    plan.position(name, *idx)
                        .and_then(|j| u32::try_from(j).ok())
                        .ok_or_else(|| format!("{name}[{idx}] is not in the sampling plan"))
                }
                _ => Ok(u32::MAX),
            })
            .collect()
    }

    /// Leaf values from a planned sample's compact buffers (`leaf_positions`).
    pub fn leaves_at(
        &self,
        bytes: &BTreeMap<String, Vec<u8>>,
        pos: &[u32],
    ) -> Result<LeafVals, String> {
        let mut out = vec![Operand::from_value(0.0); self.ops.len()];
        for (i, op) in self.ops.iter().enumerate() {
            if let Op::Leaf { buf, idx, fmt } = op {
                let name = &self.bufs[*buf as usize];
                let b = bytes
                    .get(name)
                    .ok_or_else(|| format!("no bytes for buffer {name}"))?;
                out[i] = decode(b, pos[i] as u64, *fmt)
                    .ok_or_else(|| format!("{name}[{idx}] out of range"))?;
            }
        }
        Ok(out)
    }

    /// Evaluate every op (f32 values) with the calibrated models.
    pub fn eval(&self, lv: &LeafVals) -> Vec<f32> {
        let mut w = lv.clone();
        self.eval_into(&mut w);
        w.iter().map(|o| o.v as f32).collect()
    }

    /// Evaluate in place: `w` holds the leaves on entry and every op's value on return
    /// (computed ops as `Operand::from_value`).
    pub fn eval_into(&self, w: &mut LeafVals) {
        let mut scratch = Vec::with_capacity(64);
        let mut nl_scratch: Vec<u64> = Vec::new();
        for (i, op) in self.ops.iter().enumerate() {
            let r: f32 = match op {
                Op::Leaf { .. } => continue,
                Op::Const(c) => *c as f32,
                Op::Mma {
                    model,
                    acc,
                    terms,
                    scales,
                    units,
                } => step(
                    *model,
                    acc.map(|a| w[a as usize].v as f32),
                    terms,
                    scales,
                    units,
                    w,
                    &mut scratch,
                ),
                Op::Round { arg, f16 } => {
                    let x = (w[*arg as usize].v as f32).to_bits();
                    if *f16 {
                        f32::from_bits(crate::elementwise::f16_to_f32_bits(
                            crate::fp::f32_to_f16_rn(x),
                        ))
                    } else {
                        f32::from_bits((crate::fp::f32_to_bf16_rn(x) as u32) << 16)
                    }
                }
                Op::Add { a, b, rz, ftz } => {
                    let (x, y) = (
                        (w[*a as usize].v as f32).to_bits(),
                        (w[*b as usize].v as f32).to_bits(),
                    );
                    f32::from_bits(if *ftz {
                        // As the simulator's FADD.FTZ (`AtomOp::FAddFtz`).
                        let rnd = if *rz {
                            ptxv_ir::Rnd::Rz
                        } else {
                            ptxv_ir::Rnd::Rn
                        };
                        crate::fp::f32_fma(x, 0x3f80_0000, Some(y), rnd, true)
                    } else if *rz {
                        crate::fp::f32_add_rz(x, y)
                    } else {
                        crate::fp::f32_add_rn(x, y)
                    })
                }
                Op::Fma {
                    a,
                    b,
                    c,
                    neg_ab,
                    neg_c,
                } => {
                    let x =
                        (w[*a as usize].v as f32).to_bits() ^ if *neg_ab { 0x8000_0000 } else { 0 };
                    let cc = c.map(|c| {
                        (w[c as usize].v as f32).to_bits() ^ if *neg_c { 0x8000_0000 } else { 0 }
                    });
                    f32::from_bits(crate::fp::f32_fma(
                        x,
                        (w[*b as usize].v as f32).to_bits(),
                        cc,
                        ptxv_ir::Rnd::Rn,
                        false,
                    ))
                }
                Op::Nl(nl) => {
                    let mut sc = std::mem::take(&mut nl_scratch);
                    let v = nl.eval(&|j| w[j as usize].v as f32, &mut sc);
                    nl_scratch = sc;
                    v
                }
                Op::Tf32 { arg, part } => tf32_part(w[*arg as usize].v as f32, *part),
            };
            w[i] = Operand::from_value(r as f64);
        }
    }

    /// Ops whose evaluation at the captured input differs from the simulator's value.
    pub fn check_captured(&self, m: &Machine) -> Result<usize, String> {
        let bytes: BTreeMap<String, Vec<u8>> = m
            .launch
            .allocs
            .iter()
            .map(|a| (a.name.clone(), a.init.clone()))
            .collect();
        let v = self.eval(&self.leaves(&bytes)?);
        Ok(v.iter()
            .zip(&self.hw)
            .filter(|(a, b)| a.to_bits() != **b && !(a.is_nan() && f32::from_bits(**b).is_nan()))
            .count())
    }

    /// The input buffers the leaves read, with the formats they are read as.
    pub fn leaf_formats(&self) -> BTreeMap<String, Format> {
        let mut out = BTreeMap::new();
        for op in &self.ops {
            if let Op::Leaf { buf, fmt, .. } = op {
                out.insert(self.bufs[*buf as usize].clone(), *fmt);
            }
        }
        out
    }

    /// Scale blocks per code buffer: (scale byte, code indices) from which scale leaf each
    /// code leaf is multiplied with. Err if a code meets two different scale bytes.
    pub fn scale_blocks(&self) -> Result<ScaleBlocks, String> {
        let mut pairs: BTreeMap<(String, u64), (String, u64)> = BTreeMap::new();
        let leaf = |i: u32| match &self.ops[i as usize] {
            Op::Leaf { buf, idx, .. } => Some((self.bufs[*buf as usize].clone(), *idx)),
            _ => None,
        };
        for op in &self.ops {
            if let Op::Mma { terms, scales, .. } = op {
                for (t, s) in terms.iter().zip(scales) {
                    for (c, sc) in [(t.0, s.0), (t.1, s.1)] {
                        let (Some(c), Some(sc)) = (leaf(c), leaf(sc)) else {
                            continue;
                        };
                        if let Some(prev) = pairs.insert(c.clone(), sc.clone()) {
                            if prev != sc {
                                return Err(format!(
                                    "{}[{}] is scaled by both {}[{}] and {}[{}]",
                                    c.0, c.1, prev.0, prev.1, sc.0, sc.1
                                ));
                            }
                        }
                    }
                }
            }
        }
        let mut out: BTreeMap<String, (String, BTreeMap<u64, Vec<u64>>)> = BTreeMap::new();
        for ((cb, ci), (sb, si)) in pairs {
            let e = out
                .entry(cb)
                .or_insert_with(|| (sb.clone(), BTreeMap::new()));
            if e.0 != sb {
                return Err(format!("codes scaled from two buffers {} and {sb}", e.0));
            }
            e.1.entry(si).or_default().push(ci);
        }
        Ok(out
            .into_iter()
            .map(|(k, (s, m))| (k, (s, m.into_iter().collect())))
            .collect())
    }

    /// The MMA chain of an output: its steps from first to last (following accumulators),
    /// and the ops above the last step (rounding, epilogue). None when the output is not one
    /// accumulator chain.
    pub fn chain(&self, out: usize) -> Option<(Vec<u32>, Vec<u32>)> {
        let mut top = Vec::new();
        let mut n = (*self.outputs.get(out)?)?;
        loop {
            match &self.ops[n as usize] {
                Op::Round { arg, .. } => {
                    top.push(n);
                    n = *arg;
                }
                Op::Add { a, b, .. } if matches!(self.ops[*b as usize], Op::Const(c) if c == 0.0) =>
                {
                    top.push(n);
                    n = *a;
                }
                Op::Mma { .. } => break,
                _ => return None,
            }
        }
        let mut steps = vec![n];
        while let Op::Mma { acc: Some(a), .. } = &self.ops[*steps.last().unwrap() as usize] {
            match &self.ops[*a as usize] {
                Op::Mma { .. } => steps.push(*a),
                Op::Const(c) if *c == 0.0 => break,
                _ => return None,
            }
        }
        steps.reverse();
        Some((steps, top))
    }
}

/// The first slice failures of a segment-mode run, for an error message.
fn seg_fail_note(ns: &crate::num::NumState) -> String {
    let v: Vec<String> = ns.seg_fails.values().take(4).cloned().collect();
    if v.is_empty() {
        String::new()
    } else {
        format!(" (slice failures: {})", v.join("; "))
    }
}

fn kind_name(k: &NodeKind) -> &'static str {
    match k {
        NodeKind::Leaf => "Leaf",
        NodeKind::Opaque => "Opaque",
        NodeKind::Input(_) => "Input",
        NodeKind::Add(..) => "Add",
        NodeKind::AtomAdd(..) => "AtomAdd",
        NodeKind::AtomAdd16(..) => "AtomAdd16",
        NodeKind::Mma { .. } => "Mma",
        NodeKind::Round { .. } => "Round",
        NodeKind::Mufu { .. } => "Mufu",
        NodeKind::Fma { .. } => "Fma",
        NodeKind::Max { .. } => "Max",
        NodeKind::Seg(_) => "Seg",
        NodeKind::Sel { .. } => "Sel",
        NodeKind::Tf32 { .. } => "Tf32",
    }
}

/// Decode element `idx` of a buffer read as `fmt`.
pub fn decode(b: &[u8], idx: u64, fmt: Format) -> Option<Operand> {
    let i = idx as usize;
    Some(match fmt {
        Format::F32 => Operand::from_value(f32::from_le_bytes(
            b.get(4 * i..4 * i + 4)?.try_into().ok()?,
        ) as f64),
        Format::Bf16 | Format::F16 => {
            let h = u16::from_le_bytes([*b.get(2 * i)?, *b.get(2 * i + 1)?]) as u32;
            Operand::from_value(fmt.value(h))
        }
        Format::E2m1 => Operand::new(
            ((*b.get(i / 2)? >> (4 * (i % 2))) & 0xf) as u32,
            ptxv_ir::Lp::E2m1,
        ),
        Format::Ue8m0 | Format::Ue4m3 => Operand::scale(*b.get(i)? as u32, fmt.lp().unwrap()),
        _ => Operand::new(*b.get(i)? as u32, fmt.lp().unwrap()),
    })
}

fn exp_of(x: f64) -> i32 {
    super::spec::exp_floor(x)
}

/// One step, as the calibrated models compute it (`fp::MmaModel::apply_terms`,
/// `mma_lp::LpMmaModel::step_terms`, `mma_lp::Fp4Model::step`), without allocation.
#[allow(clippy::too_many_arguments)]
pub fn step(
    model: StepModel,
    acc: Option<f32>,
    terms: &[(u32, u32)],
    scales: &[(u32, u32)],
    units: &[u8],
    lv: &LeafVals,
    scratch: &mut Vec<(f64, i32)>,
) -> f32 {
    match model {
        StepModel::Mma(md)
            if md.acc_in_group
                && !md.trunc_floor
                && md.frac_bits.is_some()
                && terms.len() <= md.group as usize =>
        {
            scratch.clear();
            for &(a, b) in terms {
                let (x, y) = (&lv[a as usize], &lv[b as usize]);
                let v = x.v * y.v;
                if v != 0.0 {
                    let e = if md.exp_from_inputs {
                        exp_of(x.v) + exp_of(y.v)
                    } else {
                        exp_of(v)
                    };
                    scratch.push((v, e));
                }
            }
            let a = acc.unwrap_or(0.0) as f64;
            if a != 0.0 {
                scratch.push((a, exp_of(a)));
            }
            trunc_sum(scratch, md.frac_bits.unwrap() as i32, md.final_rz)
        }
        StepModel::Mma(md) => {
            let t: Vec<crate::fp::Term> = terms
                .iter()
                .map(|&(a, b)| {
                    crate::fp::Term::prod(lv[a as usize].v, lv[b as usize].v, md.exp_from_inputs)
                })
                .collect();
            md.apply_terms(acc.unwrap_or(0.0), &t)
        }
        StepModel::Lp(md)
            if md.acc_in
                && !md.floor
                && md.frac.is_some()
                && md.rz
                && md.align == crate::mma_lp::Align::Fields
                && terms.len() <= md.group as usize =>
        {
            scratch.clear();
            for (j, &(a, b)) in terms.iter().enumerate() {
                let mut o = lv[a as usize].mul(&lv[b as usize]);
                if let Some(&(sa, sb)) = scales.get(j) {
                    o = o.mul(&lv[sa as usize]).mul(&lv[sb as usize]);
                }
                if o.v != 0.0 {
                    scratch.push((o.v, o.ef));
                }
            }
            if let Some(a) = acc {
                let a = a as f64;
                if a != 0.0 {
                    scratch.push((a, exp_of(a)));
                }
            }
            trunc_sum(scratch, md.frac.unwrap() as i32, true)
        }
        StepModel::Lp(md) => {
            let t: Vec<Operand> = terms
                .iter()
                .enumerate()
                .map(|(j, &(a, b))| {
                    let o = lv[a as usize].mul(&lv[b as usize]);
                    match scales.get(j) {
                        Some(&(sa, sb)) => o.mul(&lv[sa as usize]).mul(&lv[sb as usize]),
                        None => o,
                    }
                })
                .collect();
            f32::from_bits(md.step_terms(acc.map(|a| a.to_bits()), &t, false))
        }
        StepModel::Fp4(md) => {
            // Units: exact sums times their scale pair; emax over live units' scale fields.
            let nu = units.iter().map(|u| *u as usize + 1).max().unwrap_or(0);
            let mut sums = [0f64; 8];
            let mut se: [Option<i32>; 8] = [None; 8];
            let mut live = [false; 8];
            let mut sv = [0f64; 8];
            for (j, &(a, b)) in terms.iter().enumerate() {
                let u = units[j] as usize;
                let p = lv[a as usize].v * lv[b as usize].v;
                sums[u] += p; // exact: multiples of 1/4, below 2^10
                live[u] |= p != 0.0;
                let (sa, sb) = scales[j];
                let (x, y) = (&lv[sa as usize], &lv[sb as usize]);
                sv[u] = x.v * y.v;
                se[u] = Some(if md.fields {
                    x.ef + y.ef
                } else {
                    exp_of(x.v * y.v)
                });
            }
            let mut emax: Option<i32> = None;
            for u in 0..nu {
                if let Some(e) = se[u] {
                    if !md.live || live[u] {
                        emax = Some(emax.map_or(e, |m| m.max(e)));
                    }
                }
            }
            let av = acc.map(|a| a as f64);
            if let Some(a) = av.filter(|a| *a != 0.0) {
                emax = Some(emax.map_or(exp_of(a), |e| e.max(exp_of(a))));
            }
            let Some(emax) = emax else {
                return 0.0; // a zero result is +0 (N-MMA)
            };
            let q = emax - md.frac as i32;
            let mut s: i128 = 0;
            for u in 0..nu {
                s += (sums[u] * sv[u] * 2f64.powi(-q)).trunc() as i128;
            }
            if let Some(a) = av {
                s += (a * 2f64.powi(-q)).trunc() as i128;
            }
            crate::fp::round_int_to_f32(s, q, true, 0)
        }
    }
}

/// Truncate each (value, alignment exponent) toward zero at 2^(emax - frac), add exactly,
/// round once to f32 (toward zero with `rz`). A zero result (empty, cancelled or underflowed)
/// is +0, as on GB300 (N-MMA, probes/mma_zero).
fn trunc_sum(t: &[(f64, i32)], frac: i32, rz: bool) -> f32 {
    if t.is_empty() {
        return 0.0;
    }
    let emax = t.iter().map(|x| x.1).max().unwrap();
    let q = emax - frac;
    let sc = 2f64.powi(-q);
    let mut s: i128 = 0;
    for x in t {
        s += (x.0 * sc).trunc() as i128;
    }
    crate::fp::round_int_to_f32(s, q, rz, 0)
}

// ------------------------------------------------------------------ exact values

/// A nonoverlapping expansion (Shewchuk), grown in place.
#[derive(Clone, Debug, Default)]
pub struct ExpAcc {
    pub c: Vec<f64>,
}

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

impl ExpAcc {
    pub fn clear(&mut self) {
        self.c.clear();
    }
    pub fn add(&mut self, b: f64) {
        if b == 0.0 {
            return;
        }
        let mut q = b;
        let mut j = 0;
        for i in 0..self.c.len() {
            let (s, e) = two_sum(q, self.c[i]);
            if e != 0.0 {
                self.c[j] = e;
                j += 1;
            }
            q = s;
        }
        self.c.truncate(j);
        if q != 0.0 {
            self.c.push(q);
        }
    }
    /// The value rounded to f64 (components summed from the smallest).
    pub fn approx(&self) -> f64 {
        self.c.iter().sum()
    }
    pub fn is_zero(&self) -> bool {
        self.c.is_empty()
    }
}

/// Per output: the exact value of the kernel's algorithm, S = sum |terms|, and the binade
/// data the report needs.
#[derive(Clone, Copy, Debug, Default)]
pub struct Exact {
    pub y: f64,
    /// y as an expansion's leading two components (y_hi + y_lo == y to 106 bits).
    pub y_lo: f64,
    pub s: f64,
}

impl Prog {
    /// Exact value and scale of output `o`'s MMA chain (products of leaves; the rounding and
    /// epilogue above the chain are rounding errors, not part of the exact value). None when
    /// the output is not a chain.
    pub fn exact(&self, chain: &[u32], lv: &[Operand], acc: &mut ExpAcc) -> Exact {
        acc.clear();
        let mut s = 0.0;
        for &st in chain {
            if let Op::Mma { terms, scales, .. } = &self.ops[st as usize] {
                for (j, &(a, b)) in terms.iter().enumerate() {
                    let mut v = lv[a as usize].v * lv[b as usize].v;
                    if let Some(&(sa, sb)) = scales.get(j) {
                        v *= lv[sa as usize].v * lv[sb as usize].v;
                    }
                    acc.add(v);
                    s += v.abs();
                }
            }
        }
        // A constant first accumulator (nonzero) is part of the exact value.
        if let Some(&first) = chain.first() {
            if let Op::Mma { acc: Some(a), .. } = &self.ops[first as usize] {
                if let Op::Const(c) = self.ops[*a as usize] {
                    acc.add(c);
                    s += c.abs();
                }
            }
        }
        let y = acc.approx();
        let mut r = acc.clone();
        r.add(-y);
        Exact {
            y,
            y_lo: r.approx(),
            s,
        }
    }
}

impl Prog {
    /// The same program with each output taken before its 16-bit rounding and epilogue (the
    /// f32 value of its last MMA step): for pre-rounding analyses. `pre_rounded` marks the
    /// outputs whose op that changes; an output that is its chain's last MMA step already (an
    /// f32 D) keeps its value and every check (until the MISS 87 follow-up every chain output
    /// was marked, fixtures/e_tc_handback).
    pub fn pre_round(&self) -> Prog {
        let mut p = self.clone();
        p.pre_rounded = vec![false; p.outputs.len()];
        for o in 0..p.outputs.len() {
            if let Some((steps, _)) = self.chain(o) {
                let last = steps.last().copied();
                p.pre_rounded[o] = last != self.outputs[o];
                p.outputs[o] = last;
            }
        }
        p.out_fmt = Format::F32;
        p
    }

    /// Whether `pre_round` took some output before its rounding (its value changed): only
    /// there do the kernel's outputs and the program's differ by design (`--pre-round`).
    pub fn pre_round_changes(&self) -> bool {
        self.pre_rounded.iter().any(|&b| b)
    }

    /// Whether op `root`'s DAG contains a TF32 split.
    pub fn has_tf32(&self, root: u32) -> bool {
        let mut seen = HashSet::default();
        let mut st = vec![root];
        while let Some(i) = st.pop() {
            if !seen.insert(i) {
                continue;
            }
            match &self.ops[i as usize] {
                Op::Tf32 { .. } => return true,
                Op::Leaf { .. } | Op::Const(_) => {}
                Op::Nl(nl) => st.extend(nl.args()),
                Op::Round { arg, .. } => st.push(*arg),
                Op::Add { a, b, .. } => st.extend([*a, *b]),
                Op::Fma { a, b, c, .. } => {
                    st.extend([*a, *b]);
                    st.extend(c.iter().copied());
                }
                Op::Mma {
                    acc, terms, scales, ..
                } => {
                    st.extend(acc.iter().copied());
                    for &(x, y) in terms.iter().chain(scales) {
                        st.extend([x, y]);
                    }
                }
            }
        }
        false
    }

    /// Whether op `root` depends on an input leaf (outputs of kernels whose inputs the
    /// numerics did not track are constants).
    pub fn reads_leaves(&self, root: u32) -> bool {
        let mut seen = HashSet::default();
        let mut st = vec![root];
        while let Some(i) = st.pop() {
            if !seen.insert(i) {
                continue;
            }
            match &self.ops[i as usize] {
                Op::Leaf { .. } => return true,
                Op::Const(_) => {}
                Op::Nl(nl) => st.extend(nl.args()),
                Op::Round { arg, .. } | Op::Tf32 { arg, .. } => st.push(*arg),
                Op::Add { a, b, .. } => st.extend([*a, *b]),
                Op::Fma { a, b, c, .. } => {
                    st.extend([*a, *b]);
                    st.extend(c.iter().copied());
                }
                Op::Mma {
                    acc, terms, scales, ..
                } => {
                    st.extend(acc.iter().copied());
                    for &(x, y) in terms.iter().chain(scales) {
                        st.extend([x, y]);
                    }
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errdist::special::Rng;

    fn rand_bf16(r: &mut Rng) -> f64 {
        let x = (r.normal() * 2f64.powi(r.below(20) as i32 - 10)) as f32;
        f32::from_bits(x.to_bits() & 0xffff_0000) as f64
    }

    /// MISS 87 follow-up: `pre_round` marks an output taken before its rounding only where its
    /// value changes. An f32 D that is its chain's last MMA step keeps its op and every check
    /// (until the follow-up it was marked, and --pre-round skipped the worst-case check and Tier
    /// 3's validation refusal for the whole run: fixtures/e_tc_handback); a bf16 D's rounding
    /// is taken off.
    #[test]
    fn pre_round_marks_only_outputs_it_changes() {
        let leaf = |idx| Op::Leaf {
            buf: 0,
            idx,
            fmt: Format::Bf16,
        };
        let ops = vec![
            leaf(0),
            leaf(1),
            Op::Mma {
                model: StepModel::Mma(crate::fp::DEFAULT_MMA),
                acc: None,
                terms: vec![(0, 1)],
                scales: vec![],
                units: vec![],
            },
            Op::Round { arg: 2, f16: false },
        ];
        let p = Prog {
            hw: vec![0; ops.len()],
            ops,
            outputs: vec![Some(2), Some(3), Some(0)],
            out_fmt: Format::F32,
            bufs: vec!["x".into()],
            dep_untracked: vec![],
            atomic: vec![],
            atomic_src: vec![],
            atomic_free: vec![],
            pre_rounded: vec![],
        };
        let q = p.pre_round();
        assert_eq!(q.pre_rounded, [false, true, false]);
        assert_eq!(q.outputs, [Some(2), Some(2), Some(0)]);
        assert!(q.pre_round_changes());
        let f32_only = Prog {
            outputs: vec![Some(2), Some(0)],
            ..p.clone()
        };
        assert!(!f32_only.pre_round().pre_round_changes());
    }

    #[test]
    fn fast_steps_match_the_models() {
        let mut r = Rng::new(9);
        let mut scratch = vec![];
        // kind::f16 (N-MMA).
        for _ in 0..20_000 {
            let n = 1 + r.below(16) as usize;
            let mut lv: LeafVals = (0..2 * n)
                .map(|_| Operand::from_value(rand_bf16(&mut r)))
                .collect();
            if r.below(4) == 0 {
                lv[0] = Operand::from_value(0.0);
            }
            let terms: Vec<(u32, u32)> = (0..n as u32).map(|i| (2 * i, 2 * i + 1)).collect();
            let acc = if r.below(3) == 0 {
                None
            } else {
                Some(rand_bf16(&mut r) as f32 * 64.0)
            };
            let md = crate::fp::DEFAULT_MMA;
            let ours = step(StepModel::Mma(md), acc, &terms, &[], &[], &lv, &mut scratch);
            let t: Vec<crate::fp::Term> = terms
                .iter()
                .map(|&(a, b)| crate::fp::Term::prod(lv[a as usize].v, lv[b as usize].v, true))
                .collect();
            let lib = md.apply_terms(acc.unwrap_or(0.0), &t);
            assert_eq!(ours.to_bits(), lib.to_bits());
        }
        // mxf8f6f4 with ue8m0 scales (N-MMA-F8 / N-MMA-BS).
        for _ in 0..20_000 {
            let n = 32;
            let mut lv: LeafVals = Vec::new();
            for _ in 0..n {
                lv.push(Operand::new(r.below(256) as u32 & 0xf7, ptxv_ir::Lp::E4m3));
                lv.push(Operand::new(r.below(256) as u32 & 0xf7, ptxv_ir::Lp::E4m3));
            }
            let s0 = lv.len() as u32;
            lv.push(Operand::scale(100 + r.below(50) as u32, ptxv_ir::Lp::Ue8m0));
            lv.push(Operand::scale(100 + r.below(50) as u32, ptxv_ir::Lp::Ue8m0));
            let terms: Vec<(u32, u32)> = (0..n as u32).map(|i| (2 * i, 2 * i + 1)).collect();
            let scales = vec![(s0, s0 + 1); n];
            let acc = if r.below(3) == 0 {
                None
            } else {
                Some((r.normal() * 2f64.powi(r.below(40) as i32 - 20)) as f32)
            };
            let md = crate::mma_lp::DEFAULT_LP_MMA;
            let ours = step(
                StepModel::Lp(md),
                acc,
                &terms,
                &scales,
                &[],
                &lv,
                &mut scratch,
            );
            let t: Vec<Operand> = terms
                .iter()
                .map(|&(a, b)| {
                    lv[a as usize]
                        .mul(&lv[b as usize])
                        .mul(&lv[s0 as usize])
                        .mul(&lv[s0 as usize + 1])
                })
                .collect();
            let lib = f32::from_bits(md.step_terms(acc.map(|a| a.to_bits()), &t, false));
            assert_eq!(ours.to_bits(), lib.to_bits());
        }
        // mxf4nvf4 block16 ue4m3 (N-MMA-BS).
        for _ in 0..20_000 {
            let mut lv: LeafVals = Vec::new();
            for _ in 0..64 {
                lv.push(Operand::new(r.below(16) as u32, ptxv_ir::Lp::E2m1));
                lv.push(Operand::new(r.below(16) as u32, ptxv_ir::Lp::E2m1));
            }
            let s0 = lv.len() as u32;
            for _ in 0..8 {
                lv.push(Operand::scale(1 + r.below(126) as u32, ptxv_ir::Lp::Ue4m3));
            }
            let terms: Vec<(u32, u32)> = (0..64u32).map(|i| (2 * i, 2 * i + 1)).collect();
            let scales: Vec<(u32, u32)> =
                (0..64u32).map(|i| (s0 + i / 16, s0 + 4 + i / 16)).collect();
            let units: Vec<u8> = (0..64u8).map(|i| i / 16).collect();
            let acc = if r.below(3) == 0 {
                None
            } else {
                Some((r.normal() * 2f64.powi(r.below(30) as i32 - 10)) as f32)
            };
            let md = crate::mma_lp::DEFAULT_FP4_MMA;
            let ours = step(
                StepModel::Fp4(md),
                acc,
                &terms,
                &scales,
                &units,
                &lv,
                &mut scratch,
            );
            let prods: Vec<f64> = terms
                .iter()
                .map(|&(a, b)| lv[a as usize].v * lv[b as usize].v)
                .collect();
            let sa: Vec<Operand> = (0..4).map(|u| lv[s0 as usize + u]).collect();
            let sb: Vec<Operand> = (0..4).map(|u| lv[s0 as usize + 4 + u]).collect();
            let lib = f32::from_bits(md.step(acc.map(|a| a.to_bits()), &prods, &sa, &sb, 16));
            assert_eq!(ours.to_bits(), lib.to_bits());
        }
    }

    #[test]
    fn expansion_sums_exactly() {
        let mut a = ExpAcc::default();
        for x in [1e30, 1.0, -1e30, 2f64.powi(-60), -1.0] {
            a.add(x);
        }
        assert_eq!(a.approx(), 2f64.powi(-60));
        a.add(-2f64.powi(-60));
        assert!(a.is_zero());
    }
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! First-order error propagation through a program at one input: every op's real value,
//! its local error law (`Law`) and, per output, the adjoint of every op in the output's cone.
//!
//! The computed output is y^ = R(v^), the final op's rounding R of its real result v^ on
//! the computed arguments, and
//!
//!   v^ - y = sum_k G_k delta_k + O(delta^2),   G_k = d y / d v_k at the real values,
//!
//! over the local errors delta_k of the ops below the final one. The local error law of an
//! op given its real value v:
//! - an f32 add / FMA / 16-bit rounding to nearest: zero when the operands' lattice (the
//!   coarsest power of two every operand is a multiple of: an input's lowest set bit, a sum's
//!   minimum, a product's sum; a computed inexact value's own ulp) is at least the result's
//!   ulp q; else the rounding residual of a residue uniform on the r = q / 2^lattice residues
//!   (ties to even), continuous uniform for large r (N-ERRDIST-NL-RESIDUE);
//! - an instruction slice or MUFU op: its table cell at the op's index value (`table.rs`),
//!   in units of ulp32 of its real value;
//! - a tensor-core step: each term truncated at Q = 2^(emax - F) (a uniform residue on its
//!   lattice), the sum rounded to f32 (N-MMA models), as a normal law;
//! - maxima, leaves and constants: exact.

use std::collections::HashMap;

/// (site, derivative) pairs.
type Grads = Vec<(u32, f64)>;

use super::super::chain::Op;
use super::super::chain::Prog;
use super::super::chain::StepModel;
use super::super::mc::ulp_out;
use super::super::spec::Format;
use super::super::spec::exp_floor;
use super::ident::Func;
use super::ops::NlOp;
use super::table::Cell;
use super::table::Table;
use super::table::index_value;

/// A local error law (absolute units).
#[derive(Clone, Debug)]
pub enum Law {
    Zero,
    /// Round to nearest on grid q, residues on r values (r = 0: continuous), the lowest
    /// residue bit 1 with probability p1 and the higher bits uniform.
    Rn {
        q: f64,
        r: u64,
        p1: f64,
    },
    /// Toward zero on grid q for a value of sign `sgn` (r residues; 0: continuous).
    Rz {
        q: f64,
        r: u64,
        sgn: f64,
    },
    /// A table cell scaled by `ulp` (the slice's or unit's own error).
    Cell {
        tab: u32,
        cell: u32,
        ulp: f64,
    },
    /// A table cell's error before the slice's last rounding (a final slice).
    CellPre {
        tab: u32,
        cell: u32,
        ulp: f64,
    },
    Normal {
        mean: f64,
        var: f64,
    },
    /// q times an integer law (the difference of two kernels' roundings of the same value).
    Joint {
        q: f64,
        k: Vec<(i64, f64)>,
    },
    /// Explicit atoms (value, probability), absolute.
    Atoms(Vec<(f64, f64)>),
    /// No law (unidentified slice, value outside a table): the tier does not apply.
    Missing,
}

impl Law {
    /// The law of the error before the op's last rounding, in units of the rounding grid q,
    /// when the op rounds to nearest (an f32 / 16-bit rounding of an exact operation: 0; a
    /// slice ending in a rounding: its table's pre-rounding law). (q, atoms).
    pub fn pre_rounding(&self, tabs: &[Table]) -> Option<(f64, Vec<(f64, f64)>)> {
        match *self {
            Law::Rn { q, .. } => Some((q, vec![(0.0, 1.0)])),
            Law::Rz { q, sgn, .. } => Some((q, vec![(-0.5 * sgn, 1.0)])),
            Law::Normal { mean, var } => {
                // A tensor-core step: its rounding grid is not kept; one ulp of the spread.
                let q = (12.0 * var).sqrt().max(1e-300);
                let v = (var / (q * q) - 1.0 / 12.0).max(0.0);
                Some((
                    q,
                    vec![(mean / q - v.sqrt(), 0.5), (mean / q + v.sqrt(), 0.5)],
                ))
            }
            Law::Cell { tab, cell, ulp } => {
                let c = &tabs[tab as usize].cells[cell as usize];
                if c.pre_ok < 0.99 {
                    // A unit without a separate last rounding (MUFU): its error taken as a
                    // rounding of f (1 + eta), eta normal with the error's mean and its
                    // variance less the rounding's 1/12.
                    let (m, v) = (c.mean, (c.var() - 1.0 / 12.0).max(0.0));
                    let sd = v.sqrt();
                    if sd < 1e-3 {
                        return Some((ulp, vec![(m, 1.0)]));
                    }
                    let k = 16;
                    let mut at = Vec::with_capacity(2 * k + 1);
                    let mut tot = 0.0;
                    for i in -(k as i32)..=k as i32 {
                        let z = i as f64 * 4.0 / k as f64;
                        let w = (-0.5 * z * z).exp();
                        at.push((m + z * sd, w));
                        tot += w;
                    }
                    for a in at.iter_mut() {
                        a.1 /= tot;
                    }
                    return Some((ulp, at));
                }
                let v: Vec<(f64, f64)> = c
                    .pre_hist
                    .iter()
                    .enumerate()
                    .filter(|x| *x.1 > 0.0)
                    .map(|(i, p)| {
                        (
                            (i as i32 - super::table::KHALF) as f64 * super::table::KW,
                            *p as f64,
                        )
                    })
                    .collect();
                Some((ulp, v))
            }
            _ => None,
        }
    }

    /// Mean and variance.
    pub fn moments(&self, tabs: &[Table]) -> (f64, f64) {
        match *self {
            Law::Zero => (0.0, 0.0),
            Law::Rn { q, r, p1 } => {
                if r == 0 || r > 4096 {
                    (0.0, q * q / 12.0)
                } else {
                    let a = rn_atoms(q, r, p1);
                    let m: f64 = a.iter().map(|(x, p)| x * p).sum();
                    let m2: f64 = a.iter().map(|(x, p)| x * x * p).sum();
                    (m, (m2 - m * m).max(0.0))
                }
            }
            Law::Rz { q, r, sgn } => {
                let (m, v) = if r == 0 {
                    (0.5, 1.0 / 12.0)
                } else {
                    let rf = r as f64;
                    ((rf - 1.0) / (2.0 * rf), (rf * rf - 1.0) / (12.0 * rf * rf))
                };
                (-sgn * q * m, q * q * v)
            }
            Law::Cell { tab, cell, ulp } => {
                let c = &tabs[tab as usize].cells[cell as usize];
                (c.mean * ulp, c.var() * ulp * ulp)
            }
            Law::CellPre { tab, cell, ulp } => {
                let c = &tabs[tab as usize].cells[cell as usize];
                (
                    c.pre_mean * ulp,
                    (c.pre_m2 - c.pre_mean * c.pre_mean).max(0.0) * ulp * ulp,
                )
            }
            Law::Normal { mean, var } => (mean, var),
            Law::Atoms(ref a) => {
                let m: f64 = a.iter().map(|(x, p)| x * p).sum();
                let m2: f64 = a.iter().map(|(x, p)| x * x * p).sum();
                (m, (m2 - m * m).max(0.0))
            }
            Law::Joint { q, ref k } => {
                let m: f64 = k.iter().map(|(x, p)| *x as f64 * p).sum();
                let m2: f64 = k.iter().map(|(x, p)| (*x as f64).powi(2) * p).sum();
                (q * m, q * q * (m2 - m * m).max(0.0))
            }
            Law::Missing => (f64::NAN, f64::NAN),
        }
    }
    /// Atoms (value, probability) of a discrete law with at most `max` atoms, else None
    /// (continuous or too many).
    pub fn atoms(&self, tabs: &[Table], max: usize) -> Option<Vec<(f64, f64)>> {
        match *self {
            Law::Zero => Some(vec![(0.0, 1.0)]),
            Law::Joint { q, ref k } => Some(k.iter().map(|(x, p)| (*x as f64 * q, *p)).collect()),
            Law::Atoms(ref a) if a.len() <= max => Some(a.clone()),
            Law::Rn { q, r, p1 } if r > 0 && (r as usize) <= max => Some(rn_atoms(q, r, p1)),
            Law::Rz { q, r, sgn } if r > 0 && (r as usize) <= max => {
                let h = q / r as f64;
                Some(
                    (0..r)
                        .map(|j| (-sgn * j as f64 * h, 1.0 / r as f64))
                        .collect(),
                )
            }
            Law::Cell { tab, cell, ulp } => {
                let c: &Cell = &tabs[tab as usize].cells[cell as usize];
                let mut v = Vec::new();
                for (i, p) in c.hist.iter().enumerate() {
                    if *p > 0.0 {
                        let k = (i as i32 - super::table::KHALF) as f64 * super::table::KW;
                        v.push((k * ulp, *p as f64));
                    }
                }
                // Tails: at their side's mean beyond the histogram (approximate: +-12 ulps).
                if c.tails[0] > 0.0 {
                    v.push((-12.0 * ulp, c.tails[0] as f64));
                }
                if c.tails[1] > 0.0 {
                    v.push((12.0 * ulp, c.tails[1] as f64));
                }
                (v.len() <= max).then_some(v)
            }
            Law::CellPre { tab, cell, ulp } => {
                let c: &Cell = &tabs[tab as usize].cells[cell as usize];
                let v: Vec<(f64, f64)> = c
                    .pre_hist
                    .iter()
                    .enumerate()
                    .filter(|x| *x.1 > 0.0)
                    .map(|(i, p)| {
                        let k = (i as i32 - super::table::KHALF) as f64 * super::table::KW;
                        (k * ulp, *p as f64)
                    })
                    .collect();
                (v.len() <= max).then_some(v)
            }
            _ => None,
        }
    }
}

/// The rounding residual law RN(x) - x on grid q for a residue j (x = n q + j q / r) with the
/// lowest residue bit 1 with probability p1, the others uniform; ties (j = r/2) to even with
/// probability 1/2 each way.
pub fn rn_atoms(q: f64, r: u64, p1: f64) -> Vec<(f64, f64)> {
    let h = q / r as f64;
    let half = (r / 2).max(1) as f64;
    let mut v = Vec::with_capacity(r as usize + 1);
    for j in 0..r {
        let p = if r == 1 {
            1.0
        } else if j % 2 == 1 {
            p1 / half
        } else {
            (1.0 - p1) / half
        };
        if p == 0.0 {
            continue;
        }
        if 2 * j < r {
            v.push((-(j as f64) * h, p));
        } else if 2 * j > r {
            v.push(((r - j) as f64 * h, p));
        } else {
            v.push((0.5 * q, p / 2.0));
            v.push((-0.5 * q, p / 2.0));
        }
    }
    v
}

/// P(the rounded result's last bit is odd) after RN with residue law (r, p1): ties go to even,
/// other results' last bits are uniform.
fn podd_after(r: u64, p1: f64) -> f64 {
    let tie = if r == 0 {
        0.0
    } else if r == 2 {
        p1
    } else {
        (1.0 - p1) / (r / 2) as f64
    };
    0.5 * (1.0 - tie)
}

/// How an output's final value is formed from its real result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Final {
    /// Rounded to nearest (an add, FMA, 16-bit rounding, tensor-core step to nearest).
    Rn,
    /// Rounded toward zero.
    Rz,
    /// A slice or MUFU op (its table cell's integer offsets from the correctly rounded value).
    Slice { tab: u32, cell: u32 },
    /// A slice ending in a rounding to nearest: its pre-rounding error is a site, then RN.
    SlicePre { tab: u32, cell: u32, ulp: f64 },
    /// A copy of an input (exact).
    Exact,
}

/// Per op of one program at one input.
pub struct State {
    pub v: Vec<f64>,
    /// Lattice exponent of the computed value (i32::MAX for 0).
    pub lat: Vec<i32>,
    /// Lattice of the real value (None: not on a lattice, e.g. a transcendental).
    pub rlat: Vec<Option<i32>>,
    pub law: Vec<Law>,
    /// P(the computed value's bit at its lattice exponent is 1).
    pub podd: Vec<f64>,
}

/// The operating point of the linearization (N-ERRDIST-NL-FTRAJ).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Trajectory {
    /// Every op at its real value on real arguments (the original model).
    #[default]
    Real,
    /// Every op that is not an output at its computed (bit-exact, `Prog::eval`) value: an
    /// op's local error law is that of rounding the exact result of its *computed*
    /// arguments, and the adjoints are taken there. Values that are rounding artefacts (a
    /// Kahan compensation, zero in real arithmetic) then carry their computed magnitude, so
    /// the roundings they feed become sites.
    Float,
}

impl Trajectory {
    /// From `PTXV_ERRDIST_LIN_TRAJ`: `real` (default), `float`, or `both` (both modes, the
    /// real one first). Unknown values are an error.
    pub fn from_env() -> Result<Vec<Trajectory>, String> {
        match std::env::var("PTXV_ERRDIST_LIN_TRAJ").as_deref() {
            Err(_) | Ok("") | Ok("real") => Ok(vec![Trajectory::Real]),
            Ok("float") => Ok(vec![Trajectory::Float]),
            Ok("both") => Ok(vec![Trajectory::Real, Trajectory::Float]),
            Ok(o) => Err(format!(
                "PTXV_ERRDIST_LIN_TRAJ={o}: expected real, float or both"
            )),
        }
    }
}

/// A kernel prepared for linearization: its program, the tables of its slices and MUFU ops,
/// and per op the table it uses and its canonical hash (shared sub-computations of pairs).
// ASSUME(N-ERRDIST-NL-LIN): first-order propagation of local errors at nominal inputs.
pub struct KModel<'a> {
    pub p: &'a Prog,
    pub tabs: &'a [Table],
    pub op_tab: Vec<Option<u32>>,
    pub canon: Vec<u64>,
    /// Per leaf buffer (`Prog::bufs` index): the input's error model (docs/errdist.md §2.1),
    /// a local error of the leaf.
    pub in_err: Vec<Option<super::super::spec::InputError>>,
    /// The linearization's operating point.
    pub traj: Trajectory,
}

fn low_bit(v: f64) -> i32 {
    if v == 0.0 || !v.is_finite() {
        return i32::MAX;
    }
    let b = v.abs().to_bits();
    let e = ((b >> 52) & 0x7ff) as i32;
    let m = b & ((1u64 << 52) - 1);
    let (m, e) = if e == 0 {
        (m, -1074)
    } else {
        (m | (1u64 << 52), e - 1075)
    };
    e + m.trailing_zeros() as i32
}

fn lmin(a: i32, b: i32) -> i32 {
    a.min(b)
}

/// The residue count of rounding a value on lattice 2^lat to grid q (None: exact).
fn residues(q: f64, lat: i32) -> Option<u64> {
    let eq = exp_floor(q);
    if lat >= eq {
        return None;
    }
    let d = (eq - lat) as u32;
    Some(if d > 12 { 0 } else { 1u64 << d })
}

impl<'a> KModel<'a> {
    pub fn new(p: &'a Prog, tabs: &'a [Table], op_tab: Vec<Option<u32>>) -> KModel<'a> {
        let mut memo = HashMap::new();
        let canon = (0..p.ops.len() as u32)
            .map(|i| super::super::tier0::canon(p, i, &mut memo))
            .collect();
        KModel {
            p,
            tabs,
            op_tab,
            canon,
            in_err: vec![],
            traj: Trajectory::Real,
        }
    }

    /// The linearization's operating point (N-ERRDIST-NL-FTRAJ).
    pub fn with_trajectory(mut self, t: Trajectory) -> KModel<'a> {
        self.traj = t;
        self
    }

    /// The inputs' error models from the spec (N-ERRDIST-INERR).
    pub fn with_input_errors(mut self, spec: &super::super::spec::Spec) -> KModel<'a> {
        self.in_err = self
            .p
            .bufs
            .iter()
            .map(|b| spec.tensor(b).and_then(|t| t.error.clone()))
            .collect();
        self
    }

    /// Real values, lattices and local laws at the given leaf values.
    pub fn state(&self, w: &[crate::mma_lp::Operand]) -> State {
        let p = self.p;
        let n = p.ops.len();
        let mut s = State {
            v: vec![0.0; n],
            lat: vec![i32::MAX; n],
            rlat: vec![None; n],
            law: vec![Law::Zero; n],
            podd: vec![0.5; n],
        };
        let ul32 = |v: f64| ulp_out(v, Format::F32);
        let xor = |a: f64, b: f64| a * (1.0 - b) + b * (1.0 - a);
        // ASSUME(N-ERRDIST-NL-FTRAJ): float-trajectory operating point (opt-in).
        // Float trajectory: the computed value of every op, and which ops are outputs (an
        // output keeps the exact result of its computed arguments: its own rounding is the
        // last site, positioned by that value).
        let fp: Option<(Vec<f32>, Vec<bool>)> = (self.traj == Trajectory::Float).then(|| {
            let mut is_out = vec![false; n];
            for &o in p.outputs.iter().flatten() {
                is_out[o as usize] = true;
            }
            (p.eval(&w.to_vec()), is_out)
        });
        for (i, op) in p.ops.iter().enumerate() {
            // The bit of operand x at lattice exponent l.
            let bit = |s: &State, x: usize, l: i32| if s.lat[x] == l { s.podd[x] } else { 0.0 };
            let (v, lat, rlat, law, podd) = match op {
                Op::Leaf { buf, fmt, .. } => {
                    let v = w[i].v;
                    match self.in_err.get(*buf as usize).and_then(|e| e.as_ref()) {
                        None => (
                            v,
                            low_bit(v),
                            Some(low_bit(v)),
                            Law::Zero,
                            (v != 0.0) as u8 as f64,
                        ),
                        // ASSUME(N-ERRDIST-INERR): the error is a local law of the leaf,
                        // independent across elements, pushed through first order.
                        // An input error: the kernel reads round_fmt(v + e) (a value on the
                        // format's grid, its low bit uniform); the reference v.
                        Some(e) => {
                            let qf = ulp_out(v, *fmt);
                            let law = match e.round_format() {
                                Some(rf) => {
                                    let q = ulp_out(v, rf);
                                    match residues(q, low_bit(v)) {
                                        None => Law::Zero,
                                        Some(r) => Law::Rn { q, r, p1: 0.5 },
                                    }
                                }
                                None => {
                                    let (mean, var) = e.moments(v).unwrap_or_default();
                                    Law::Normal {
                                        mean,
                                        var: var + qf * qf / 12.0,
                                    }
                                }
                            };
                            (v, exp_floor(qf), None, law, 0.5)
                        }
                    }
                }
                Op::Const(c) => (
                    *c,
                    low_bit(*c),
                    Some(low_bit(*c)),
                    Law::Zero,
                    (*c != 0.0) as u8 as f64,
                ),
                // A TF32 part as a value of its own (nonlin::analytic refuses TF32 programs;
                // the chain tiers' DAG CLT models the split errors).
                Op::Tf32 { arg, part } => {
                    let v = super::super::chain::tf32_part(s.v[*arg as usize] as f32, *part) as f64;
                    (
                        v,
                        low_bit(v),
                        Some(low_bit(v)),
                        Law::Zero,
                        (v != 0.0) as u8 as f64,
                    )
                }
                Op::Round { arg, f16 } => {
                    let a = *arg as usize;
                    let v = s.v[a];
                    let fmt = if *f16 { Format::F16 } else { Format::Bf16 };
                    let q = ulp_out(v, fmt);
                    match residues(q, s.lat[a]) {
                        None => (v, s.lat[a], s.rlat[a], Law::Zero, s.podd[a]),
                        Some(r) => {
                            let p1 = s.podd[a];
                            (
                                v,
                                exp_floor(q),
                                s.rlat[a],
                                Law::Rn { q, r, p1 },
                                podd_after(r, p1),
                            )
                        }
                    }
                }
                Op::Add { a, b, rz } => {
                    let (x, y) = (*a as usize, *b as usize);
                    let v = s.v[x] + s.v[y];
                    let lin = lmin(s.lat[x], s.lat[y]);
                    let rl = match (s.rlat[x], s.rlat[y]) {
                        (Some(p), Some(q)) => Some(p.min(q)),
                        _ => None,
                    };
                    let p1 = xor(bit(&s, x, lin), bit(&s, y, lin));
                    let q = ul32(v);
                    match residues(q, lin) {
                        None => (v, lin, rl, Law::Zero, p1),
                        Some(r) if *rz => (
                            v,
                            exp_floor(q),
                            rl,
                            Law::Rz {
                                q,
                                r,
                                sgn: v.signum(),
                            },
                            0.5,
                        ),
                        Some(r) => (v, exp_floor(q), rl, Law::Rn { q, r, p1 }, podd_after(r, p1)),
                    }
                }
                Op::Fma {
                    a,
                    b,
                    c,
                    neg_ab,
                    neg_c,
                } => {
                    let (x, y) = (*a as usize, *b as usize);
                    let mut v = s.v[x] * s.v[y];
                    if *neg_ab {
                        v = -v;
                    }
                    let plat = s.lat[x].saturating_add(s.lat[y]);
                    let mut lin = if v == 0.0 { i32::MAX } else { plat };
                    let mut rl = match (s.rlat[x], s.rlat[y]) {
                        (Some(p), Some(q)) => Some(p.saturating_add(q)),
                        _ => None,
                    };
                    let pp = s.podd[x] * s.podd[y];
                    let mut zc = None;
                    if let Some(c) = c {
                        let z = *c as usize;
                        v += if *neg_c { -s.v[z] } else { s.v[z] };
                        lin = lmin(lin, s.lat[z]);
                        rl = match (rl, s.rlat[z]) {
                            (Some(p), Some(q)) => Some(p.min(q)),
                            _ => None,
                        };
                        zc = Some(z);
                    }
                    let bp = if plat == lin { pp } else { 0.0 };
                    let p1 = match zc {
                        Some(z) => xor(bp, bit(&s, z, lin)),
                        None => bp,
                    };
                    let q = ul32(v);
                    match residues(q, lin) {
                        None => (v, lin, rl, Law::Zero, p1),
                        Some(r) => (v, exp_floor(q), rl, Law::Rn { q, r, p1 }, podd_after(r, p1)),
                    }
                }
                Op::Mma {
                    model,
                    acc,
                    terms,
                    scales,
                    ..
                } => {
                    let (v, l, rl, law) = self.mma_state(&s, *model, *acc, terms, scales);
                    (v, l, rl, law, 0.5)
                }
                Op::Nl(nl) => match nl {
                    // The branch the predicate takes at the nominal values; the tier is refused
                    // on programs with a selection (N-ERRDIST-SEL), this only keeps values.
                    NlOp::Sel {
                        pred,
                        args,
                        neg,
                        a,
                        b,
                    } => {
                        let v: Vec<f32> = args.iter().map(|&j| s.v[j as usize] as f32).collect();
                        let mut scratch = Vec::new();
                        let pick = if NlOp::sel_holds(pred, *neg, &v, &mut scratch) {
                            *a as usize
                        } else {
                            *b as usize
                        };
                        (
                            s.v[pick],
                            s.lat[pick],
                            s.rlat[pick],
                            Law::Zero,
                            s.podd[pick],
                        )
                    }
                    NlOp::Max { a, b, min } => {
                        let (x, y) = (*a as usize, *b as usize);
                        let pick = if (s.v[x] >= s.v[y]) != *min || s.v[y].is_nan() {
                            x
                        } else {
                            y
                        };
                        (
                            s.v[pick],
                            s.lat[pick],
                            s.rlat[pick],
                            Law::Zero,
                            s.podd[pick],
                        )
                    }
                    NlOp::Mufu {
                        arg,
                        func,
                        ptx,
                        neg,
                        abs,
                    } => {
                        let a = s.v[*arg as usize];
                        let mut z = if *abs { a.abs() } else { a };
                        if *neg {
                            z = -z;
                        }
                        let v = crate::num::mufu_true(*func, *ptx, z);
                        let law = self.cell_law(i, a, v);
                        (v, exp_floor(ul32(v)), None, law, 0.5)
                    }
                    NlOp::Seg { args, real, .. } => match real {
                        None => (f64::NAN, i32::MIN, None, Law::Missing, 0.5),
                        Some(f) => {
                            let a: Vec<f64> = args.iter().map(|&j| s.v[j as usize]).collect();
                            let v = f.eval(&a);
                            // An exact power-of-two scaling keeps the lattice and is exact.
                            if let Some((j, k)) = pow2_scale(f) {
                                let src = args[j] as usize;
                                (
                                    v,
                                    s.lat[src].saturating_add(k),
                                    s.rlat[src].map(|l| l + k),
                                    Law::Zero,
                                    s.podd[src],
                                )
                            } else {
                                let law = self.cell_law(i, index_value(f, &a), v);
                                (v, exp_floor(ul32(v)), None, law, 0.5)
                            }
                        }
                    },
                },
            };
            let v = match &fp {
                Some((fv, is_out))
                    if !is_out[i]
                        && !matches!(op, Op::Leaf { .. } | Op::Const(_))
                        && fv[i].is_finite() =>
                {
                    fv[i] as f64
                }
                _ => v,
            };
            s.v[i] = v;
            s.lat[i] = lat;
            s.rlat[i] = rlat;
            s.law[i] = law;
            s.podd[i] = podd;
        }
        s
    }

    fn cell_law(&self, i: usize, t: f64, v: f64) -> Law {
        let Some(tab) = self.op_tab[i] else {
            return Law::Missing;
        };
        let tb = &self.tabs[tab as usize];
        match tb.lookup(t) {
            Some(ci) => Law::Cell {
                tab,
                cell: ci as u32,
                ulp: ulp_out(v, Format::F32),
            },
            None => Law::Missing,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mma_state(
        &self,
        s: &State,
        model: StepModel,
        acc: Option<u32>,
        terms: &[(u32, u32)],
        scales: &[(u32, u32)],
    ) -> (f64, i32, Option<i32>, Law) {
        let (frac, rz, inputs) = match model {
            StepModel::Mma(md) => (
                md.frac_bits.map_or(23, |f| f as i32),
                md.final_rz,
                md.exp_from_inputs,
            ),
            StepModel::Lp(md) => (md.frac.map_or(23, |f| f as i32), md.rz, true),
            StepModel::Fp4(md) => (md.frac as i32, true, true),
        };
        let mut v = acc.map_or(0.0, |a| s.v[a as usize]);
        let mut items: Vec<(f64, i32, i32)> = Vec::new(); // (value, align exp, lattice)
        let mut rl = acc.and_then(|a| s.rlat[a as usize]);
        if let Some(a) = acc {
            let x = s.v[a as usize];
            if x != 0.0 {
                items.push((x, exp_floor(x), s.lat[a as usize]));
            }
        }
        for (j, &(a, b)) in terms.iter().enumerate() {
            let (x, y) = (a as usize, b as usize);
            let mut t = s.v[x] * s.v[y];
            let mut lat = s.lat[x].saturating_add(s.lat[y]);
            let mut e = if inputs {
                exp_floor(s.v[x]).saturating_add(exp_floor(s.v[y]))
            } else {
                exp_floor(t)
            };
            if let Some(&(sa, sb)) = scales.get(j) {
                let k = s.v[sa as usize] * s.v[sb as usize];
                t *= k;
                lat = lat.saturating_add(exp_floor(k));
                e = e.saturating_add(exp_floor(k));
            }
            rl = match (rl, s.rlat[x], s.rlat[y]) {
                (Some(p), Some(q1), Some(q2)) if acc.is_some() || j > 0 => {
                    Some(p.min(q1.saturating_add(q2)))
                }
                (_, Some(q1), Some(q2)) => Some(q1.saturating_add(q2)),
                _ => None,
            };
            v += t;
            if t != 0.0 {
                items.push((t, e, lat));
            }
        }
        if items.is_empty() {
            return (v, i32::MAX, rl, Law::Zero);
        }
        let emax = items.iter().map(|x| x.1).max().unwrap();
        let qe = emax - frac;
        let q = 2f64.powi(qe);
        let (mut mean, mut var) = (0.0, 0.0);
        let mut lin = i32::MAX;
        for &(t, _, lat) in &items {
            lin = lin.min(lat.max(qe));
            if let Some(r) = residues(q, lat) {
                let (m, vv) = Law::Rz {
                    q,
                    r,
                    sgn: t.signum(),
                }
                .moments(&[]);
                mean += m;
                var += vv;
            }
        }
        let q32 = ulp_out(v, Format::F32);
        let law_final = match residues(q32, lin) {
            None => Law::Zero,
            Some(r) if rz => Law::Rz {
                q: q32,
                r,
                sgn: v.signum(),
            },
            Some(r) => Law::Rn { q: q32, r, p1: 0.5 },
        };
        let (m2, v2) = law_final.moments(&[]);
        let law = if mean == 0.0 && var == 0.0 && matches!(law_final, Law::Zero) {
            Law::Zero
        } else {
            Law::Normal {
                mean: mean + m2,
                var: var + v2,
            }
        };
        let lat = if matches!(law, Law::Zero) {
            lin
        } else {
            exp_floor(q32)
        };
        (v, lat, rl, law)
    }

    /// The cone of output op `o` (ops it depends on), in decreasing order.
    pub fn cone(&self, o: u32) -> Vec<u32> {
        let p = self.p;
        let mut seen = vec![false; o as usize + 1];
        let mut st = vec![o];
        seen[o as usize] = true;
        let mut out = Vec::new();
        while let Some(i) = st.pop() {
            out.push(i);
            for k in op_args(&p.ops[i as usize]) {
                if !seen[k as usize] {
                    seen[k as usize] = true;
                    st.push(k);
                }
            }
        }
        out.sort_unstable_by(|a, b| b.cmp(a));
        out
    }

    /// The final op of an output at this input (through maxima to the chosen argument) and
    /// how it forms the value.
    pub fn final_of(&self, s: &State, o: u32) -> (u32, Final) {
        let mut i = o;
        loop {
            match &self.p.ops[i as usize] {
                Op::Nl(NlOp::Max { a, b, min }) => {
                    let (x, y) = (*a as usize, *b as usize);
                    i = if (s.v[x] >= s.v[y]) != *min { *a } else { *b };
                }
                Op::Leaf { .. } | Op::Const(_) => return (i, Final::Exact),
                Op::Nl(NlOp::Seg { .. }) | Op::Nl(NlOp::Mufu { .. }) => {
                    return match s.law[i as usize] {
                        Law::Cell { tab, cell, ulp }
                            if self.tabs[tab as usize].cells[cell as usize].pre_ok >= 0.99 =>
                        {
                            (i, Final::SlicePre { tab, cell, ulp })
                        }
                        Law::Cell { tab, cell, .. } => (i, Final::Slice { tab, cell }),
                        Law::Zero => (i, Final::Exact),
                        _ => (i, Final::Rn),
                    };
                }
                Op::Add { rz: true, .. } => return (i, Final::Rz),
                Op::Mma { model, .. } => {
                    let rz = match model {
                        StepModel::Mma(md) => md.final_rz,
                        StepModel::Lp(md) => md.rz,
                        StepModel::Fp4(_) => true,
                    };
                    return (i, if rz { Final::Rz } else { Final::Rn });
                }
                _ => return (i, Final::Rn),
            }
        }
    }

    /// Adjoints d y_o / d v_k over the cone (decreasing order), sparse: (op, adjoint).
    pub fn adjoints(&self, s: &State, cone: &[u32], adj: &mut Vec<f64>) -> Vec<(u32, f64)> {
        self.adjoints_cut(s, cone, adj, &|_| false).0
    }

    /// `adjoints`, not propagating through the ops `cut` selects (other than the root): those
    /// are returned apart with their adjoints.
    pub fn adjoints_cut(
        &self,
        s: &State,
        cone: &[u32],
        adj: &mut Vec<f64>,
        cut: &dyn Fn(u32) -> bool,
    ) -> (Grads, Grads) {
        let mut cuts = Vec::new();
        let out = self.adjoints_inner(s, cone, adj, cut, &mut cuts);
        (out, cuts)
    }

    fn adjoints_inner(
        &self,
        s: &State,
        cone: &[u32],
        adj: &mut Vec<f64>,
        cut: &dyn Fn(u32) -> bool,
        cuts: &mut Vec<(u32, f64)>,
    ) -> Vec<(u32, f64)> {
        let p = self.p;
        if adj.len() < p.ops.len() {
            adj.resize(p.ops.len(), 0.0);
        }
        let o = cone[0];
        adj[o as usize] = 1.0;
        let mut out = Vec::with_capacity(cone.len());
        for &i in cone {
            let g = adj[i as usize];
            adj[i as usize] = 0.0;
            if g == 0.0 {
                continue;
            }
            if i != o && cut(i) {
                cuts.push((i, g));
                continue;
            }
            out.push((i, g));
            let mut push = |k: u32, d: f64| adj[k as usize] += g * d;
            match &p.ops[i as usize] {
                Op::Leaf { .. } | Op::Const(_) => {}
                Op::Round { arg, .. } => push(*arg, 1.0),
                Op::Tf32 { arg, part } => {
                    if *part == crate::num::Tf32Part::Hi {
                        push(*arg, 1.0);
                    }
                }
                Op::Add { a, b, .. } => {
                    push(*a, 1.0);
                    push(*b, 1.0);
                }
                Op::Fma {
                    a,
                    b,
                    c,
                    neg_ab,
                    neg_c,
                } => {
                    let sg = if *neg_ab { -1.0 } else { 1.0 };
                    let (x, y) = (s.v[*a as usize], s.v[*b as usize]);
                    push(*a, sg * y);
                    push(*b, sg * x);
                    if let Some(c) = c {
                        push(*c, if *neg_c { -1.0 } else { 1.0 });
                    }
                }
                Op::Mma {
                    acc, terms, scales, ..
                } => {
                    if let Some(a) = acc {
                        push(*a, 1.0);
                    }
                    for (j, &(a, b)) in terms.iter().enumerate() {
                        let k = scales
                            .get(j)
                            .map_or(1.0, |&(x, y)| s.v[x as usize] * s.v[y as usize]);
                        push(a, s.v[b as usize] * k);
                        push(b, s.v[a as usize] * k);
                    }
                }
                Op::Nl(NlOp::Max { a, b, min }) => {
                    let (x, y) = (*a as usize, *b as usize);
                    push(if (s.v[x] >= s.v[y]) != *min { *a } else { *b }, 1.0);
                }
                Op::Nl(NlOp::Mufu {
                    arg,
                    func,
                    ptx,
                    neg,
                    abs,
                }) => {
                    let a = s.v[*arg as usize];
                    let mut z = if *abs { a.abs() } else { a };
                    let mut dm = if *abs { a.signum() } else { 1.0 };
                    if *neg {
                        z = -z;
                        dm = -dm;
                    }
                    push(*arg, crate::num::mufu_true_deriv(*func, *ptx, z) * dm);
                }
                Op::Nl(NlOp::Sel {
                    pred,
                    args,
                    neg,
                    a,
                    b,
                }) => {
                    let v: Vec<f32> = args.iter().map(|&j| s.v[j as usize] as f32).collect();
                    let mut scratch = Vec::new();
                    let k = if NlOp::sel_holds(pred, *neg, &v, &mut scratch) {
                        *a
                    } else {
                        *b
                    };
                    push(k, 1.0);
                }
                Op::Nl(NlOp::Seg { args, real, .. }) => {
                    if let Some(f) = real {
                        let a: Vec<f64> = args.iter().map(|&j| s.v[j as usize]).collect();
                        for (j, d) in f.grad(&a).into_iter().enumerate() {
                            if d != 0.0 && d.is_finite() {
                                push(args[j], d);
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

/// An identified slice that multiplies one argument by a power of two exactly: (arg, k).
fn pow2_scale(f: &super::ident::RealFn) -> Option<(usize, i32)> {
    if f.func != Func::Ident || f.c0 != 0.0 || f.mul.is_some() || f.outer != (0.0, 1.0) {
        return None;
    }
    let nz: Vec<usize> = (0..f.coef.len()).filter(|&i| f.coef[i] != 0.0).collect();
    let [j] = nz.as_slice() else { return None };
    let c = f.coef[*j].abs();
    let k = exp_floor(c);
    (c == 2f64.powi(k)).then_some((*j, k))
}

pub fn op_args(op: &Op) -> Vec<u32> {
    match op {
        Op::Leaf { .. } | Op::Const(_) => vec![],
        Op::Round { arg, .. } | Op::Tf32 { arg, .. } => vec![*arg],
        Op::Add { a, b, .. } => vec![*a, *b],
        Op::Fma { a, b, c, .. } => [Some(*a), Some(*b), *c].into_iter().flatten().collect(),
        Op::Mma {
            acc, terms, scales, ..
        } => {
            let mut v: Vec<u32> = acc.iter().copied().collect();
            for &(a, b) in terms.iter().chain(scales) {
                v.extend([a, b]);
            }
            v
        }
        Op::Nl(n) => n.args(),
    }
}

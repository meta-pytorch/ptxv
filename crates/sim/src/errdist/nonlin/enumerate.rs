/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Elementwise outputs of one input element: the exact error law by enumeration.
//!
//! When every output is the same function of a single input element (gelu, silu, tanh, exp,
//! a type conversion) and the elements are i.i.d. with a plain law (no axis scales or
//! outliers, round-to-nearest quantization), the output error's law is the pushforward of the
//! element's code law through the bit-exact computation. For 16-bit inputs every code of
//! positive probability is evaluated (Tier 0: exact up to N-ERRDIST-IID); for f32 inputs the
//! law is integrated by deterministic quadrature over 256 cells per binade, 4 codes each
//! (Tier 1). The function is evaluated on output 0's cone extracted as a small program; the
//! other outputs are checked to have the same structure.

use std::collections::BTreeMap;

use super::super::chain::Op;
use super::super::chain::Prog;
use super::super::mc::U;
use super::super::mc::round_out;
use super::super::mc::ulp_out;
use super::super::report::ErrorBar;
use super::super::report::Estimate;
use super::super::report::Moments;
use super::super::spec::Component;
use super::super::spec::Dist;
use super::super::spec::Format;
use super::super::spec::Quant;
use super::super::spec::Spec;
use super::lin::op_args;
use super::ops::NlOp;
use crate::errdist::det::HashMap;

/// The cone of op `root` as a program of its own (ops renumbered), and the new index of
/// every leaf.
pub fn sub_prog(p: &Prog, root: u32) -> (Prog, Vec<u32>) {
    let (s, l, _) = sub_prog_cut(p, root, &[]);
    (s, l)
}

/// `sub_prog` with the ops `stops` as inputs (f32 leaves, buffer 0, element = position in
/// `stops`; their cones dropped). Returns also each stop's new index (None: not in the cone).
pub fn sub_prog_cut(p: &Prog, root: u32, stops: &[u32]) -> (Prog, Vec<u32>, Vec<Option<u32>>) {
    let mut seen = vec![false; root as usize + 1];
    let mut st = vec![root];
    seen[root as usize] = true;
    let mut cone = Vec::new();
    while let Some(i) = st.pop() {
        cone.push(i);
        if stops.contains(&i) {
            continue;
        }
        for k in op_args(&p.ops[i as usize]) {
            if !seen[k as usize] {
                seen[k as usize] = true;
                st.push(k);
            }
        }
    }
    cone.sort_unstable();
    let map: HashMap<u32, u32> = cone
        .iter()
        .enumerate()
        .map(|(j, &i)| (i, j as u32))
        .collect();
    let m = |x: &u32| map[x];
    let mut ops = Vec::with_capacity(cone.len());
    let mut leaves = Vec::new();
    for &i in &cone {
        if let Some(k) = stops.iter().position(|&x| x == i) {
            leaves.push(map[&i]);
            ops.push(Op::Leaf {
                buf: 0,
                idx: k as u64,
                fmt: Format::F32,
            });
            continue;
        }
        let op = match &p.ops[i as usize] {
            Op::Leaf { buf, idx, fmt } => {
                leaves.push(map[&i]);
                Op::Leaf {
                    buf: *buf,
                    idx: *idx,
                    fmt: *fmt,
                }
            }
            Op::Const(c) => Op::Const(*c),
            Op::Mma {
                model,
                acc,
                terms,
                scales,
                units,
            } => Op::Mma {
                model: *model,
                acc: acc.as_ref().map(m),
                terms: terms.iter().map(|(a, b)| (m(a), m(b))).collect(),
                scales: scales.iter().map(|(a, b)| (m(a), m(b))).collect(),
                units: units.clone(),
            },
            Op::Round { arg, f16 } => Op::Round {
                arg: m(arg),
                f16: *f16,
            },
            Op::Tf32 { arg, part } => Op::Tf32 {
                arg: m(arg),
                part: *part,
            },
            Op::Add { a, b, rz } => Op::Add {
                a: m(a),
                b: m(b),
                rz: *rz,
            },
            Op::Fma {
                a,
                b,
                c,
                neg_ab,
                neg_c,
            } => Op::Fma {
                a: m(a),
                b: m(b),
                c: c.as_ref().map(m),
                neg_ab: *neg_ab,
                neg_c: *neg_c,
            },
            Op::Nl(n) => Op::Nl(match n {
                NlOp::Seg { prog, args, real } => NlOp::Seg {
                    prog: prog.clone(),
                    args: args.iter().map(m).collect(),
                    real: real.clone(),
                },
                NlOp::Mufu {
                    arg,
                    func,
                    ptx,
                    neg,
                    abs,
                } => NlOp::Mufu {
                    arg: m(arg),
                    func: *func,
                    ptx: *ptx,
                    neg: *neg,
                    abs: *abs,
                },
                NlOp::Max { a, b, min } => NlOp::Max {
                    a: m(a),
                    b: m(b),
                    min: *min,
                },
                NlOp::Sel {
                    pred,
                    args,
                    neg,
                    a,
                    b,
                } => NlOp::Sel {
                    pred: pred.clone(),
                    args: args.iter().map(m).collect(),
                    neg: *neg,
                    a: m(a),
                    b: m(b),
                },
            }),
        };
        ops.push(op);
    }
    let hw = cone.iter().map(|&i| p.hw[i as usize]).collect();
    (
        Prog {
            ops,
            hw,
            outputs: vec![Some(map[&root])],
            out_fmt: p.out_fmt,
            bufs: p.bufs.clone(),
            dep_untracked: vec![],
        },
        leaves,
        stops.iter().map(|x| map.get(x).copied()).collect(),
    )
}

/// A structural hash of an output's cone, leaves by format only.
fn shape(p: &Prog, root: u32) -> u64 {
    use std::hash::Hash;
    use std::hash::Hasher;
    let (s, _) = sub_prog(p, root);
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for op in &s.ops {
        match op {
            Op::Leaf { fmt, .. } => (0u8, format!("{fmt:?}")).hash(&mut h),
            Op::Const(c) => (1u8, c.to_bits()).hash(&mut h),
            Op::Nl(n) => (2u8, n.shape_key(), n.args()).hash(&mut h),
            o => format!("{:?}", std::mem::discriminant(o)).hash(&mut h),
        }
        op_args(op).hash(&mut h);
    }
    h.finish()
}

/// The element law's support: (value, probability) atoms; 16-bit formats exactly, f32 by
/// quadrature. None when the tensor's law is not plain.
fn element_atoms(spec: &Spec, buf: &str, fmt: Format) -> Option<(Vec<(f64, f64)>, bool)> {
    let t = spec.tensor(buf)?;
    if !t.scales.is_empty() || t.outliers.is_some() || !matches!(t.quant, Quant::Round) {
        return None;
    }
    match fmt {
        Format::Bf16 | Format::F16 => {
            let pmf = super::super::laws::code_pmf(&t.dist, fmt, 1.0);
            Some((pmf.v.into_iter().zip(pmf.p).collect(), true))
        }
        Format::F32 => Some(f32_codes(&t.dist, F32_POINTS)),
        _ => None,
    }
}

/// f32 codes of a law, (value, probability): its atoms when it is discrete; a mixture's
/// components each on their own, `points` shared in proportion to their weights, so that a
/// point mass stays one code (spread over its 1/256-binade cell like a density, the 95% of
/// v = 1 in `lin_gap/vmix` became values near 1 that round like N(0, 1) draws: 0.26 against
/// Tier 3's 0.013); else `points` codes over cells of 1/256 binade. Exact for discrete laws.
fn f32_codes(d: &Dist, points: usize) -> (Vec<(f64, f64)>, bool) {
    if let Some(a) = d.atoms() {
        return (
            a.into_iter().map(|(x, p)| (x as f32 as f64, p)).collect(),
            true,
        );
    }
    let mixture = match d {
        Dist::Mixture { components } => Some(components.clone()),
        // a X + b of a mixture: the mixture of the components' a X + b.
        Dist::Affine { a, b, dist } => match &**dist {
            Dist::Mixture { components } => Some(
                components
                    .iter()
                    .map(|c| Component {
                        weight: c.weight,
                        dist: Dist::Affine {
                            a: *a,
                            b: *b,
                            dist: Box::new(c.dist.clone()),
                        },
                    })
                    .collect(),
            ),
            _ => None,
        },
        _ => None,
    };
    if let Some(cs) = mixture {
        let tot: f64 = cs.iter().map(|c| c.weight).sum();
        let (mut out, mut exact) = (Vec::new(), true);
        for c in &cs {
            let w = c.weight / tot;
            let (a, ex) = f32_codes(&c.dist, ((points as f64 * w).ceil() as usize).max(1));
            exact &= ex;
            out.extend(a.into_iter().map(|(x, p)| (x, p * w)));
        }
        return (out, exact);
    }
    // Cells of 1/256 binade; `points` codes allotted in proportion to the cells' masses
    // (systematically over the cumulative mass, at least one per cell of positive mass), each
    // at a pseudo-random point of its cell: full-mantissa codes (midpoints would have ~10
    // significant bits, and rounding-sensitive results would come out biased) and an
    // effective sample size near `points` (equal weights per binade would leave a few
    // thousand effective codes where the mass is).
    let mut cells: Vec<(f64, f64, f64)> = Vec::new();
    for sg in [1.0f64, -1.0] {
        for e in -149..128 {
            let (lo, hi) = (2f64.powi(e), 2f64.powi(e + 1));
            let m = if sg > 0.0 {
                d.mass(lo, true, hi, false)
            } else {
                d.mass(-hi, false, -lo, true)
            };
            if m < 1e-14 {
                continue;
            }
            for c in 0..256 {
                let a = lo + (hi - lo) * c as f64 / 256.0;
                let b = lo + (hi - lo) * (c + 1) as f64 / 256.0;
                let w = if sg > 0.0 {
                    d.mass(a, true, b, false)
                } else {
                    d.mass(-b, false, -a, true)
                };
                if w > 0.0 {
                    cells.push((sg * a, sg * b, w));
                }
            }
        }
    }
    let tot: f64 = cells.iter().map(|c| c.2).sum();
    let mut rng = super::super::special::Rng::new(0x9ad);
    let mut out = Vec::new();
    let mut acc = 0.0;
    let mut next = rng.unit();
    for &(a, b, w) in &cells {
        acc += w / tot * points as f64;
        let mut k = 0usize;
        while next < acc {
            k += 1;
            next += 1.0;
        }
        let k = k.max(1);
        for j in 0..k {
            let x = a + (b - a) * (j as f64 + rng.unit()) / k as f64;
            out.push((x as f32 as f64, w / k as f64));
        }
    }
    let z = d.mass(-f64::MIN_POSITIVE, false, f64::MIN_POSITIVE, false);
    if z > 0.0 {
        out.push((0.0, z));
    }
    (out, false)
}

/// Whether the program's output is unchanged when its leaves outside `main` change (at 64
/// random values of the `main` leaves).
fn independent(p: &Prog, main: &[u32], leaves: &[u32], fmt: Format) -> bool {
    let mut rng = super::super::special::Rng::new(0x1ead);
    let mut w = vec![crate::mma_lp::Operand::from_value(0.0); p.ops.len()];
    let root = p.ops.len() - 1;
    for _ in 0..64 {
        let xs: Vec<f64> = main
            .iter()
            .map(|_| {
                fmt.round(rng.normal() * 2f64.powi(rng.below(8) as i32 - 4), false)
                    .0
            })
            .collect();
        let mut first: Option<u32> = None;
        for _ in 0..3 {
            for &l in leaves {
                let v = match main.iter().position(|&m| m == l) {
                    Some(k) => xs[k],
                    None => fmt.round(rng.normal() * 4.0, false).0,
                };
                w[l as usize] = crate::mma_lp::Operand::from_value(v);
            }
            p.eval_into(&mut w);
            let b = (w[root].v as f32).to_bits();
            match first {
                None => first = Some(b),
                Some(f) if f != b => return false,
                _ => {}
            }
        }
    }
    true
}

/// `n` stratified draws from a discrete law: the quantile at (i + u_i) / n for each i.
fn stratified(atoms: &[(f64, f64)], n: usize, rng: &mut super::super::special::Rng) -> Vec<f64> {
    let mut a: Vec<(f64, f64)> = atoms.iter().copied().filter(|x| x.1 > 0.0).collect();
    a.sort_by(|x, y| x.0.total_cmp(&y.0));
    let tot: f64 = a.iter().map(|x| x.1).sum();
    let mut out = Vec::with_capacity(n);
    let (mut acc, mut k) = (0.0, 0);
    for i in 0..n {
        let q = (i as f64 + rng.unit()) / n as f64 * tot;
        while k + 1 < a.len() && acc + a[k].1 < q {
            acc += a[k].1;
            k += 1;
        }
        out.push(a[k].0);
    }
    out
}

/// f32 element codes per input law.
const F32_POINTS: usize = 1 << 20;

/// Evaluations of the element function budgeted for the product quadrature.
const BUDGET: f64 = 4e6;

/// Weighted samples to moments.
fn moments(xs: &mut [(f64, f64)]) -> Moments {
    xs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let w: f64 = xs.iter().map(|x| x.1).sum::<f64>().max(1e-300);
    let mean = xs.iter().map(|x| x.0 * x.1).sum::<f64>() / w;
    let var = (xs.iter().map(|x| x.0 * x.0 * x.1).sum::<f64>() / w - mean * mean).max(0.0);
    let q = |p: f64| {
        let mut acc = 0.0;
        for &(x, pw) in xs.iter() {
            acc += pw / w;
            if acc >= p {
                return x;
            }
        }
        xs.last().map_or(0.0, |x| x.0)
    };
    Moments {
        mean,
        var,
        std: var.sqrt(),
        mean_abs: xs.iter().map(|x| x.0.abs() * x.1).sum::<f64>() / w,
        p_zero: xs.iter().filter(|x| x.0 == 0.0).map(|x| x.1).sum::<f64>() / w,
        quantiles: super::super::stats::QUANTILES
            .iter()
            .map(|&p| (p, q(p)))
            .collect(),
        max_abs: Some(xs.iter().map(|x| x.0.abs()).fold(0.0, f64::max)),
    }
}

/// The exact (16-bit inputs) or quadrature (f32) law of elementwise outputs; None when the
/// programs are not elementwise in one plain input element.
// ASSUME(N-ERRDIST-NL-ELEM): exact on 16-bit codes, stratified quadrature for f32.
pub fn elementwise(r: &Prog, c: Option<&Prog>, spec: &Spec) -> Option<Result<Estimate, String>> {
    let t0 = std::time::Instant::now();
    let o_r = r.outputs.first().copied().flatten()?;
    let (sr, lr) = sub_prog(r, o_r);
    // One to three leaves (input elements); every output the same structure.
    if lr.is_empty() || lr.len() > 3 {
        return None;
    }
    let mut keys: Vec<(String, u64, Format)> = Vec::new();
    for &l in &lr {
        let Op::Leaf { buf, idx, fmt } = sr.ops[l as usize] else {
            return None;
        };
        keys.push((r.bufs[buf as usize].clone(), idx, fmt));
    }
    // Two leaves of one buffer would need their joint law (always i.i.d. here, but the
    // elements of a row are not what elementwise means): one leaf per buffer.
    for a in 0..keys.len() {
        for b in a + 1..keys.len() {
            if keys[a].0 == keys[b].0 {
                return None;
            }
        }
    }
    let sh = shape(r, o_r);
    let n = r.outputs.len();
    for o in (0..n).step_by((n / 64).max(1)) {
        let x = r.outputs[o]?;
        if shape(r, x) != sh {
            return None;
        }
    }
    // The candidate: the same leaves (by buffer and element); other leaves (the other half
    // of a packed 16-bit word) must not change the output.
    let sc = match c {
        Some(c) => {
            let o_c = c.outputs.first().copied().flatten()?;
            let (sp, lc) = sub_prog(c, o_c);
            let mut map = Vec::new();
            for (name, idx, _) in &keys {
                map.push(lc.iter().copied().find(|&l| {
                    matches!(sp.ops[l as usize], Op::Leaf { buf: b2, idx: i2, .. } if &c.bufs[b2 as usize] == name && i2 == *idx)
                })?);
            }
            if lc.len() > map.len() && !independent(&sp, &map, &lc, keys[0].2) {
                return None;
            }
            let sh_c = shape(c, o_c);
            for o in (0..n).step_by((n / 64).max(1)) {
                if shape(c, c.outputs.get(o).copied().flatten()?) != sh_c {
                    return None;
                }
            }
            let rest: Vec<u32> = lc.iter().copied().filter(|l| !map.contains(l)).collect();
            Some((sp, map, rest))
        }
        None => None,
    };
    // Per leaf: its atoms.
    let mut per: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut exact = true;
    for (name, _, fmt) in &keys {
        let (a, ex) = element_atoms(spec, name, *fmt)?;
        exact &= ex;
        per.push(a);
    }
    // The product of the laws when it fits the budget; else a Latin hypercube: every leaf
    // stratified over its whole law at `budget` points, the pairing a fixed shuffle (each
    // leaf keeps full resolution; a resampled product grid would see a hash-like indicator,
    // "the two exp slices differ at a", at a few thousand values only).
    let budget: f64 = std::env::var("PTXV_NL_ENUM_BUDGET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(BUDGET);
    let full: f64 = per.iter().map(|a| a.len() as f64).product();
    let lhs: Option<Vec<Vec<f64>>> = if full <= budget {
        None
    } else {
        exact = false;
        let n = budget as usize;
        let mut rng = super::super::special::Rng::new(0x1a7e);
        let mut cols = Vec::new();
        for (k, a) in per.iter().enumerate() {
            let mut col = stratified(a, n, &mut rng);
            if k > 0 {
                for i in (1..n).rev() {
                    col.swap(i, rng.below(i as u64 + 1) as usize);
                }
            }
            cols.push(col);
        }
        Some(cols)
    };
    let total: usize = match &lhs {
        Some(c) => c[0].len(),
        None => full as usize,
    };
    let bname = keys
        .iter()
        .map(|k| format!("{} ({:?})", k.0, k.2))
        .collect::<Vec<_>>()
        .join(" x ");
    let ofmt = r.out_fmt;
    let mut ulp_s: Vec<(f64, f64)> = Vec::new();
    let mut abs_s: Vec<(f64, f64)> = Vec::new();
    let mut u_s: Vec<(f64, f64)> = Vec::new();
    let mut pmf: BTreeMap<i64, f64> = BTreeMap::new();
    let (mut pd, mut tot, mut skipped, mut sum_p2) = (0.0, 0.0, 0.0, 0.0);
    // Mass whose PMF key is off the output-ulp lattice (pairs only: C - R across binades).
    let mut off_lattice = 0.0;
    let mut wr = vec![crate::mma_lp::Operand::from_value(0.0); sr.ops.len()];
    let mut wc = sc
        .as_ref()
        .map(|(s, _, _)| vec![crate::mma_lp::Operand::from_value(0.0); s.ops.len()]);
    // The product of the leaves' atoms, by mixed-radix index.
    let mut xs = vec![0.0f64; per.len()];
    for t in 0..total {
        let (mut q, mut p) = (t, 1.0);
        match &lhs {
            Some(cols) => {
                for (k, c) in cols.iter().enumerate() {
                    xs[k] = c[t];
                }
                p = 1.0 / total as f64;
            }
            None => {
                for (k, a) in per.iter().enumerate() {
                    let (x, w) = a[q % a.len()];
                    q /= a.len();
                    xs[k] = x;
                    p *= w;
                }
            }
        }
        if p <= 0.0 {
            continue;
        }
        for (k, &l) in lr.iter().enumerate() {
            wr[l as usize] = crate::mma_lp::Operand::from_value(xs[k]);
        }
        sr.eval_into(&mut wr);
        let rv = wr[sr.ops.len() - 1].v as f32 as f64;
        let real = super::real::eval(&sr, &wr);
        let ex = super::real::exact_of(&real, (sr.ops.len() - 1) as u32);
        if !ex.y.is_finite() || !rv.is_finite() || ex.s == 0.0 {
            skipped += p;
            continue;
        }
        let ul = ulp_out(ex.y, ofmt);
        let us = U * ex.s;
        let (e, differ, kref) = match (&sc, wc.as_mut()) {
            (Some((spc, map, rest)), Some(w)) => {
                for &l in rest {
                    w[l as usize] = crate::mma_lp::Operand::from_value(xs[0]);
                }
                for (k, &l) in map.iter().enumerate() {
                    w[l as usize] = crate::mma_lp::Operand::from_value(xs[k]);
                }
                spc.eval_into(w);
                let cv = w[spc.ops.len() - 1].v as f32 as f64;
                if !cv.is_finite() {
                    skipped += p;
                    continue;
                }
                (
                    cv - rv,
                    (cv as f32).to_bits() != (rv as f32).to_bits(),
                    None,
                )
            }
            _ => {
                let rn = round_out(ex.y, ex.y_lo, ofmt);
                // The pmf_ulp axis of a single kernel, as Tier 3's: output - RN(y), in output
                // ulps. Keyed by the error against y itself, only tuples whose output error was
                // a whole number of ulps entered the PMF (c_gelu_bf16: total mass 1e-16).
                (
                    (rv - ex.y) - ex.y_lo,
                    rn != rv,
                    Some(((rv - rn) / ul).round()),
                )
            }
        };
        tot += p;
        sum_p2 += p * p;
        pd += p * differ as u8 as f64;
        let el = e / ul;
        let kl = kref.unwrap_or(el);
        if kl.fract() == 0.0 && kl.abs() < 1e15 {
            *pmf.entry(kl as i64).or_default() += p;
        } else {
            off_lattice += p;
        }
        ulp_s.push((el, p));
        abs_s.push((e, p));
        u_s.push((e / us, p));
    }
    if tot == 0.0 {
        return Some(Err("no finite output".into()));
    }
    let n_eff = tot * tot / sum_p2.max(1e-300);
    let mut eb = ErrorBar {
        kind: if exact { "exact" } else { "quadrature" }.into(),
        notes: vec![format!(
            "{} element tuples of {bname} evaluated bit-exactly{}",
            total,
            if exact {
                String::new()
            } else if lhs.is_some() {
                " (a Latin hypercube over the leaves' laws, each stratified at every point)".into()
            } else {
                format!(
                    " (f32: {F32_POINTS} codes stratified over the law, effective sample size {n_eff:.0})"
                )
            }
        )],
        ..Default::default()
    };
    if skipped > 0.0 {
        eb.model_mass = Some(skipped);
        eb.notes.push(format!(
            "{skipped:.3e} probability with a non-finite value (out of domain)"
        ));
    }
    let p = pd / tot;
    // Latin hypercube: at most the binomial spread of as many independent draws.
    let half = if lhs.is_some() {
        1.96 * (p * (1.0 - p) / total as f64).sqrt()
    } else if !exact {
        // Stratified f32 codes: at most the binomial spread at the effective sample size.
        1.96 * (p * (1.0 - p) / n_eff.max(1.0)).sqrt()
    } else {
        0.0
    };
    Some(Ok(Estimate {
        tier: if exact { 0 } else { 1 },
        method: format!(
            "elementwise: every output is one function of {} input element(s); its law {} through the bit-exact element function",
            keys.len(),
            if exact {
                "enumerated over the element's codes"
            } else {
                "integrated by quadrature"
            }
        ),
        n: total as u64,
        outputs: n as u64,
        abs: moments(&mut abs_s),
        u_scale: Some(moments(&mut u_s)),
        ulp: Some(moments(&mut ulp_s)),
        // As Tier 3: an f32 law with mass off the lattice has no PMF (the CLI notes it).
        pmf_ulp: if off_lattice == 0.0 || ofmt != Format::F32 {
            pmf.into_iter().map(|(k, q)| (k, q / tot)).collect()
        } else {
            vec![]
        },
        hist_u: None,
        error_bar: eb,
        pair: None,
        p_differ: Some((p, (p - half).max(0.0), (p + half).min(1.0))),
        seconds: t0.elapsed().as_secs_f64(),
        // Agent AG: checked against Tier 3 by the CLI (docs/errdist.md §9).
        applicable: None,
        applicability: vec![],
    }))
}

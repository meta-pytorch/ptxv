/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tier 0: exact results (docs/errdist.md §5.1).
//!
//! `certify`: a support-bound certificate that every step of a chain is exact (all terms and
//! the accumulator are multiples of 2^q, and every partial sum fits f32's 24 bits), so the
//! error is 0 on the support of the element laws (up to their pruned mass).
//!
//! `enumerate`: the exact signed law of (output, error) of a chain by dynamic programming
//! over the accumulator's value; per step and emax E, the terms' (truncated integer, dropped
//! part) law is the product of the groups' laws restricted to alignment <= E, minus the same
//! restricted to <= E - 1 (inclusion-exclusion on the order statistic). Only for small supports.

use std::collections::HashMap;

use super::chain::Op;
use super::chain::Prog;
use super::chain::StepModel;
use super::combine::Law;
use super::combine::single_laws;
use super::fine::NO_ALIGN;
use super::fine::lowest_bit;
use super::mc::ulp_out;
use super::report::ErrorBar;
use super::report::Estimate;
use super::schema::ChainSchema;
use super::schema::StepSchema;
use super::spec::Format;
use super::spec::exp_floor;

#[derive(Clone, Debug, Default)]
pub struct Cert {
    pub exact: bool,
    /// The first step the certificate fails at.
    pub first_inexact: Option<usize>,
    /// Exponent of the coarsest power of two all values are multiples of.
    pub lattice: i32,
    /// Bound on |partial sums|.
    pub bound: f64,
}

fn rz(st: &StepSchema) -> bool {
    match st.model {
        StepModel::Mma(md) => md.final_rz,
        _ => true,
    }
}

/// Support-bound exactness certificate for a chain.
pub fn certify(sch: &ChainSchema) -> Cert {
    let mut lat = i32::MAX;
    let mut bound = 0f64;
    let mut first = None;
    for (si, st) in sch.steps.iter().enumerate() {
        let mut amax = i32::MIN;
        let mut add = 0.0;
        for g in &st.pop.groups {
            let mut gmax = 0f64;
            for (w, t) in &g.parts {
                if *w == 0.0 {
                    continue;
                }
                for a in &t.atoms {
                    gmax = gmax.max(a.0.abs());
                    if a.1 > NO_ALIGN {
                        amax = amax.max(a.1);
                    }
                    if a.0 != 0.0 {
                        lat = lat.min(lowest_bit(a.0));
                    }
                }
            }
            add += g.n as f64 * gmax;
        }
        let eacc = if bound > 0.0 {
            exp_floor(bound)
        } else {
            i32::MIN
        };
        let e = amax.max(eacc);
        bound += add;
        let ok =
            lat == i32::MAX || (e.saturating_sub(st.frac) <= lat && bound < 2f64.powi(lat + 24));
        if !ok && first.is_none() {
            first = Some(si);
        }
    }
    Cert {
        exact: first.is_none(),
        first_inexact: first,
        lattice: if lat == i32::MAX { 0 } else { lat },
        bound,
    }
}

/// (truncated integer sum, dropped part as f64 bits) -> signed weight.
type TermMap = HashMap<(i128, u64), f64>;

fn conv(a: &TermMap, b: &TermMap, cap: usize) -> Result<TermMap, String> {
    if a.len().saturating_mul(b.len()) > 16 * cap {
        return Err(format!("tier 0: more than {cap} states"));
    }
    let mut o = TermMap::with_capacity(a.len().max(b.len()));
    for (&(i, t), &p) in a {
        for (&(j, u), &q) in b {
            let k = (i + j, (f64::from_bits(t) + f64::from_bits(u)).to_bits());
            *o.entry(k).or_default() += p * q;
        }
        if o.len() > cap {
            return Err(format!("tier 0: more than {cap} states"));
        }
    }
    o.retain(|_, w| *w != 0.0);
    Ok(o)
}

fn power(a: &TermMap, n: usize, cap: usize) -> Result<TermMap, String> {
    let mut r: TermMap = [((0i128, 0f64.to_bits()), 1.0)].into_iter().collect();
    let mut b = a.clone();
    let mut k = n;
    while k > 0 {
        if k & 1 == 1 {
            r = conv(&r, &b, cap)?;
        }
        k >>= 1;
        if k > 0 {
            b = conv(&b, &b, cap)?;
        }
    }
    Ok(r)
}

/// The step's terms (all alignments <= e) truncated at 2^q: signed law of (int sum, dropped).
fn terms_law(st: &StepSchema, e: i32, q: i32, cap: usize) -> Result<TermMap, String> {
    let sc = 2f64.powi(-q);
    let mut out: TermMap = [((0i128, 0f64.to_bits()), 1.0)].into_iter().collect();
    let atoms: usize = st
        .pop
        .groups
        .iter()
        .map(|g| g.parts.iter().map(|p| p.1.atoms.len()).sum::<usize>())
        .sum();
    if atoms > cap {
        return Err(format!("tier 0: {atoms} term atoms (> {cap})"));
    }
    for g in &st.pop.groups {
        let mut gl = TermMap::new();
        for (w, t) in &g.parts {
            let mut one = TermMap::new();
            for a in t.atoms.iter().filter(|a| a.1 <= e) {
                let i = (a.0 * sc).trunc();
                let tau = a.0 - i / sc;
                *one.entry((i as i128, (tau + 0.0).to_bits())).or_default() += a.2;
            }
            if one.is_empty() {
                continue;
            }
            for (k, p) in power(&one, g.n, cap)? {
                *gl.entry(k).or_default() += w * p;
            }
            if gl.len() > cap {
                return Err(format!("tier 0: more than {cap} states"));
            }
        }
        out = conv(&out, &gl, cap)?;
        if out.is_empty() {
            break;
        }
    }
    Ok(out)
}

/// Exact signed law of (output value before any 16-bit rounding, error = output - exact sum)
/// of a chain; Err when a state space exceeds `cap`.
pub fn enumerate(sch: &ChainSchema, cap: usize) -> Result<Vec<(f64, f64, f64)>, String> {
    // State: (accumulator f32 bits, error f64 bits) -> weight.
    let mut st: HashMap<(u32, u64), f64> = [((0f32.to_bits(), 0f64.to_bits()), 1.0)]
        .into_iter()
        .collect();
    for s in &sch.steps {
        let mut aligns: Vec<i32> = s
            .pop
            .groups
            .iter()
            .flat_map(|g| g.parts.iter().flat_map(|p| p.1.atoms.iter().map(|a| a.1)))
            .filter(|&a| a > NO_ALIGN)
            .collect();
        aligns.sort();
        aligns.dedup();
        let mut cache: HashMap<(i32, i32), TermMap> = HashMap::new();
        let mut law = |e: i32, q: i32| -> Result<TermMap, String> {
            if let Some(m) = cache.get(&(e, q)) {
                return Ok(m.clone());
            }
            let m = terms_law(s, e, q, cap)?;
            cache.insert((e, q), m.clone());
            Ok(m)
        };
        let mut next: HashMap<(u32, u64), f64> = HashMap::new();
        let put = |next: &mut HashMap<(u32, u64), f64>, a: f32, err: f64, w: f64| {
            *next
                .entry(((a + 0.0).to_bits(), (err + 0.0).to_bits()))
                .or_default() += w;
        };
        for (&(ab, eb), &w0) in &st {
            let a = f32::from_bits(ab) as f64;
            let err0 = f64::from_bits(eb);
            let ea = if a != 0.0 { Some(exp_floor(a)) } else { None };
            // Cases: (emax, the terms' signed law given that emax).
            let mut cases: Vec<(i32, TermMap)> = vec![];
            if let Some(ea) = ea {
                cases.push((ea, law(ea, ea - s.frac)?));
            } else {
                // All terms without alignment (zero / dead): the step returns the accumulator.
                let none = law(NO_ALIGN, 0)?;
                let m: f64 = none.values().sum();
                put(&mut next, a as f32, err0, w0 * m);
            }
            for &e in aligns.iter().filter(|&&e| ea.is_none_or(|ea| e > ea)) {
                let q = e - s.frac;
                let mut m = law(e, q)?;
                let below = law(e - 1, q)?;
                for (k, p) in below {
                    *m.entry(k).or_default() -= p;
                }
                cases.push((e, m));
            }
            for (e, m) in cases {
                let q = e - s.frac;
                let sc = 2f64.powi(-q);
                let ai = (a * sc).trunc();
                let atau = a - ai / sc;
                for ((i, t), p) in m {
                    if p == 0.0 {
                        continue;
                    }
                    let si = i + ai as i128;
                    let r = crate::fp::round_int_to_f32(si, q, rz(s), 0);
                    let loc = (r as f64 - si as f64 / sc) - (f64::from_bits(t) + atau);
                    put(&mut next, r, err0 + loc, w0 * p);
                }
            }
            if next.len() > cap {
                return Err(format!("tier 0: more than {cap} accumulator states"));
            }
        }
        next.retain(|_, w| w.abs() > 1e-300);
        st = next;
    }
    Ok(st
        .into_iter()
        .map(|((a, e), w)| (f32::from_bits(a) as f64, f64::from_bits(e), w))
        .collect())
}

/// Tier 0 estimate of a single kernel from its exact law; with a 16-bit output, P(differ) is
/// the exact misrounding probability P(rnd16(out) != rnd16(exact)).
pub fn estimate(sch: &ChainSchema, out_fmt: Format, cap: usize) -> Result<Estimate, String> {
    let law = enumerate(sch, cap)?;
    let mut lu: HashMap<u64, f64> = HashMap::new();
    let mut la: HashMap<u64, f64> = HashMap::new();
    let mut mis = 0.0;
    let mut tot = 0.0;
    let mut pmf16: std::collections::BTreeMap<i64, f64> = Default::default();
    for &(out, err, w) in &law {
        let y = out - err;
        let u = ulp_out(y, Format::F32);
        *lu.entry((err / u + 0.0).to_bits()).or_default() += w;
        *la.entry((err + 0.0).to_bits()).or_default() += w;
        tot += w;
        if let Some(f) = sch.round16 {
            let (a, b) = (f.round(out, false).0, f.round(y, false).0);
            mis += w * (a != b) as u8 as f64;
            let k = ((a - b) / ulp_out(y, f)).round();
            if k.is_finite() {
                *pmf16.entry(k as i64).or_default() += w;
            }
        }
    }
    let to_law = |m: HashMap<u64, f64>| {
        Law::from(
            m.into_iter()
                .map(|(k, w)| (f64::from_bits(k), w.max(0.0)))
                .collect(),
        )
    };
    let (lu, la) = (to_law(lu), to_law(la));
    let est = Estimate {
        tier: 0,
        method: "exact enumeration of the chain's signed law (accumulator value x error)".into(),
        outputs: 1,
        error_bar: ErrorBar {
            kind: "exact".into(),
            notes: vec![format!(
                "{} (output, error) atoms; total mass {:.12}",
                law.len(),
                tot
            )],
            ..Default::default()
        },
        ..Default::default()
    };
    let mut est = single_laws(est, &lu, &la, out_fmt, sch.round16);
    if let Some(f) = sch.round16.filter(|_| out_fmt != Format::F32) {
        let p = (mis / tot.max(1e-300)).max(0.0) + 0.0;
        est.p_differ = Some((p, p, p));
        est.pmf_ulp = pmf16
            .into_iter()
            .map(|(k, w)| (k, w / tot.max(1e-300)))
            .filter(|x| x.1 > 0.0)
            .collect();
        est.method = format!(
            "exact enumeration of the chain's signed law; P(rnd{f:?}(out) != rnd{f:?}(exact)) exact"
        );
    }
    Ok(est)
}

/// Tier 0 estimate when the certificate proves the error identically 0 (single: C = exact
/// sum; pair: C = R).
pub fn certified(out_fmt: Format, pair: bool, certs: &[Cert]) -> Estimate {
    let z = Law::from(vec![(0.0, 1.0)]);
    let mut est = single_laws(Estimate::default(), &z, &z, out_fmt, None);
    est.tier = 0;
    est.outputs = 1;
    est.method = format!(
        "exactness certificate: every step's terms are multiples of 2^q and all partial sums fit 24 bits ({})",
        certs
            .iter()
            .map(|c| format!("lattice 2^{}, |sum| <= {:.4e}", c.lattice, c.bound))
            .collect::<Vec<_>>()
            .join("; ")
    );
    est.error_bar = ErrorBar {
        kind: "exact".into(),
        notes: vec![if pair {
            "C = R = the exact sum on the support of the element laws".into()
        } else {
            "C = the exact sum on the support of the element laws".into()
        }],
        ..Default::default()
    };
    est.p_differ = Some((0.0, 0.0, 0.0));
    est
}

/// Canonical hash of an op's subtree: leaves by (buffer name, index), MMA steps with the
/// products as a multiset where the model sums the truncated terms in one exact addition
/// (order-free), else in order. Equal hashes of two outputs: the same function of the inputs.
pub fn canon(p: &Prog, i: u32, memo: &mut HashMap<u32, u64>) -> u64 {
    use std::hash::Hash;
    use std::hash::Hasher;
    if let Some(&h) = memo.get(&i) {
        return h;
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    match &p.ops[i as usize] {
        Op::Leaf { buf, idx, fmt } => {
            (0u8, &p.bufs[*buf as usize], idx, format!("{fmt:?}")).hash(&mut h);
        }
        Op::Const(v) => (1u8, v.to_bits()).hash(&mut h),
        Op::Mma {
            model,
            acc,
            terms,
            scales,
            units,
        } => {
            let order_free = match model {
                StepModel::Mma(md) => {
                    md.acc_in_group
                        && !md.trunc_floor
                        && md.frac_bits.is_some()
                        && terms.len() <= md.group as usize
                }
                StepModel::Lp(md) => {
                    md.acc_in
                        && !md.floor
                        && md.frac.is_some()
                        && md.rz
                        && terms.len() <= md.group as usize
                }
                StepModel::Fp4(_) => true,
            };
            let mut ts: Vec<u64> = terms
                .iter()
                .enumerate()
                .map(|(j, &(a, b))| {
                    let mut f = [canon(p, a, memo), canon(p, b, memo)];
                    f.sort();
                    let sc = scales.get(j).map(|&(x, y)| {
                        let mut g = [canon(p, x, memo), canon(p, y, memo)];
                        g.sort();
                        g
                    });
                    let mut th = std::collections::hash_map::DefaultHasher::new();
                    (f, sc, units.get(j)).hash(&mut th);
                    th.finish()
                })
                .collect();
            if order_free {
                ts.sort();
            }
            (
                2u8,
                format!("{model:?}"),
                acc.map(|a| canon(p, a, memo)),
                ts,
            )
                .hash(&mut h);
        }
        Op::Round { arg, f16 } => (3u8, canon(p, *arg, memo), f16).hash(&mut h),
        Op::Tf32 { arg, part } => (6u8, canon(p, *arg, memo), format!("{part:?}")).hash(&mut h),
        // x + 0 is x up to the sign of a zero result (-0 + +0 = +0): an epilogue's beta * C with
        // beta = 0 folds away.
        Op::Add { a, b, .. } if matches!(p.ops[*b as usize], Op::Const(z) if z == 0.0) => {
            let v = canon(p, *a, memo);
            memo.insert(i, v);
            return v;
        }
        Op::Add { a, b, .. } if matches!(p.ops[*a as usize], Op::Const(z) if z == 0.0) => {
            let v = canon(p, *b, memo);
            memo.insert(i, v);
            return v;
        }
        Op::Add { a, b, rz } => {
            let mut f = [canon(p, *a, memo), canon(p, *b, memo)];
            f.sort();
            (4u8, f, rz).hash(&mut h)
        }
        Op::Fma {
            a,
            b,
            c,
            neg_ab,
            neg_c,
        } => {
            let mut f = [canon(p, *a, memo), canon(p, *b, memo)];
            f.sort();
            (5u8, f, c.map(|c| canon(p, c, memo)), neg_ab, neg_c).hash(&mut h)
        }
        // Agent AG: non-chain ops (ordered children).
        Op::Nl(nl) => {
            // A max (min) tree is the max over the set of its non-max operands, in any order.
            let ks: Vec<u64> = match nl {
                super::nonlin::ops::NlOp::Max { min, .. } => {
                    let mut st = vec![i];
                    let mut ks = Vec::new();
                    while let Some(j) = st.pop() {
                        match &p.ops[j as usize] {
                            Op::Nl(super::nonlin::ops::NlOp::Max { a, b, min: m2 })
                                if m2 == min =>
                            {
                                st.extend([*a, *b]);
                            }
                            _ => ks.push(canon(p, j, memo)),
                        }
                    }
                    ks.sort();
                    ks.dedup();
                    ks
                }
                _ => nl.args().iter().map(|&a| canon(p, a, memo)).collect(),
            };
            (6u8, nl.shape_key(), ks).hash(&mut h)
        }
    }
    let v = h.finish();
    memo.insert(i, v);
    v
}

/// The number of outputs where the two programs compute the same function (equal canonical
/// hashes; up to the sign of a zero result), and the number compared.
pub fn identical_outputs(r: &Prog, c: &Prog) -> (usize, usize) {
    let (mut mr, mut mc) = (HashMap::new(), HashMap::new());
    if std::env::var("PTXV_ERRDIST_DEBUG").is_ok() {
        for p in [r, c] {
            let Some(Some(o)) = p.outputs.first() else {
                continue;
            };
            let mut i = *o;
            loop {
                match &p.ops[i as usize] {
                    Op::Mma {
                        model, acc, terms, ..
                    } => {
                        let lf = |x: u32| match &p.ops[x as usize] {
                            Op::Leaf { buf, idx, .. } => {
                                format!("{}[{idx}]", p.bufs[*buf as usize])
                            }
                            o => format!("{o:?}"),
                        };
                        eprintln!(
                            "  mma {} terms: {:?} {:?} .. | {model:?}",
                            terms.len(),
                            terms.first().map(|t| (lf(t.0), lf(t.1))),
                            terms.last().map(|t| (lf(t.0), lf(t.1)))
                        );
                        match acc {
                            Some(a) => i = *a,
                            None => break,
                        }
                    }
                    Op::Round { arg, .. } => {
                        eprintln!("  round");
                        i = *arg
                    }
                    o => {
                        eprintln!("  {o:?}");
                        break;
                    }
                }
            }
            eprintln!("--");
        }
    }
    let mut same = 0;
    let mut n = 0;
    for (a, b) in r.outputs.iter().zip(&c.outputs) {
        if let (Some(a), Some(b)) = (a, b) {
            n += 1;
            same += (canon(r, *a, &mut mr) == canon(c, *b, &mut mc)) as usize;
        }
    }
    (same, n)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc as Rc;

    use super::*;
    use crate::errdist::fine::Group;
    use crate::errdist::fine::Pop;
    use crate::errdist::fine::TermLaw;

    /// (term count, [(weight, atoms (v, p))]) per group.
    type GroupSpec = (usize, Vec<(f64, Vec<(f64, f64)>)>);

    fn step(frac: i32, groups: Vec<GroupSpec>) -> StepSchema {
        let al = |v: f64| if v == 0.0 { i32::MIN / 4 } else { exp_floor(v) };
        StepSchema {
            model: StepModel::Mma(crate::fp::DEFAULT_MMA),
            frac,
            fp4: false,
            pop: Pop {
                groups: groups
                    .into_iter()
                    .map(|(n, parts)| Group {
                        n,
                        parts: parts
                            .into_iter()
                            .map(|(w, a)| {
                                (
                                    w,
                                    Rc::new(TermLaw::new(
                                        a.into_iter().map(|(v, p)| (v, al(v), p)).collect(),
                                    )),
                                )
                            })
                            .collect(),
                    })
                    .collect(),
            },
        }
    }

    /// Brute force over every term tuple: truncate at 2^(emax - frac), add, round to f32
    /// toward zero; signed weights multiply.
    fn brute(sch: &ChainSchema) -> HashMap<(u64, u64), f64> {
        let mut st: Vec<(f64, f64, f64)> = vec![(0.0, 0.0, 1.0)]; // (acc, exact, weight)
        for s in &sch.steps {
            // All term tuples of the step: (values, weight).
            let mut tuples: Vec<(Vec<(f64, i32)>, f64)> = vec![(vec![], 1.0)];
            for g in &s.pop.groups {
                let mut gt: Vec<(Vec<(f64, i32)>, f64)> = vec![];
                for (w, t) in &g.parts {
                    let mut pt: Vec<(Vec<(f64, i32)>, f64)> = vec![(vec![], *w)];
                    for _ in 0..g.n {
                        pt = pt
                            .into_iter()
                            .flat_map(|(v, p)| {
                                t.atoms
                                    .iter()
                                    .map(move |a| ([v.clone(), vec![(a.0, a.1)]].concat(), p * a.2))
                            })
                            .collect();
                    }
                    gt.extend(pt);
                }
                tuples = tuples
                    .into_iter()
                    .flat_map(|(v, p)| {
                        gt.iter()
                            .map(move |(u, q)| ([v.clone(), u.clone()].concat(), p * q))
                    })
                    .collect();
            }
            let mut next = vec![];
            for &(a, x, w) in &st {
                for (t, p) in &tuples {
                    let mut all: Vec<(f64, i32)> =
                        t.iter().copied().filter(|v| v.1 > NO_ALIGN).collect();
                    if a != 0.0 {
                        all.push((a, exp_floor(a)));
                    }
                    let ex = x + t.iter().map(|v| v.0).sum::<f64>();
                    let r = if all.is_empty() {
                        a
                    } else {
                        let q = all.iter().map(|v| v.1).max().unwrap() - s.frac;
                        let sc = 2f64.powi(-q);
                        let si: i128 = all.iter().map(|v| (v.0 * sc).trunc() as i128).sum::<i128>()
                            + t.iter()
                                .filter(|v| v.1 <= NO_ALIGN)
                                .map(|v| (v.0 * sc).trunc() as i128)
                                .sum::<i128>();
                        crate::fp::round_int_to_f32(si, q, true, 0) as f64
                    };
                    next.push((r, ex, w * p));
                }
            }
            st = next;
        }
        let mut m: HashMap<(u64, u64), f64> = HashMap::new();
        for (a, x, w) in st {
            *m.entry(((a + 0.0).to_bits(), (a - x + 0.0).to_bits()))
                .or_default() += w;
        }
        m
    }

    fn check(sch: &ChainSchema) {
        let b = brute(sch);
        let mut e: HashMap<(u64, u64), f64> = HashMap::new();
        for (a, err, w) in enumerate(sch, 1 << 20).unwrap() {
            *e.entry(((a + 0.0).to_bits(), (err + 0.0).to_bits()))
                .or_default() += w;
        }
        let mut keys: Vec<_> = b.keys().chain(e.keys()).copied().collect();
        keys.sort();
        keys.dedup();
        let mut nz = 0.0;
        for k in keys {
            let (x, y) = (
                b.get(&k).copied().unwrap_or(0.0),
                e.get(&k).copied().unwrap_or(0.0),
            );
            assert!(
                (x - y).abs() < 1e-12,
                "atom (out {}, err {}): brute {x} vs tier 0 {y}",
                f64::from_bits(k.0),
                f64::from_bits(k.1)
            );
            if f64::from_bits(k.1) != 0.0 {
                nz += x;
            }
        }
        assert!(nz > 0.01, "the test chain should have nonzero errors");
    }

    #[test]
    fn enumeration_matches_brute_force() {
        let law = vec![
            (1.0, 0.3),
            (-0.375, 0.2),
            (3.25, 0.25),
            (0.0, 0.1),
            (0.0390625, 0.15),
        ];
        let sch = ChainSchema {
            steps: vec![
                step(4, vec![(3, vec![(1.0, law.clone())])]),
                step(
                    4,
                    vec![
                        (2, vec![(1.0, law.clone())]),
                        (1, vec![(1.0, vec![(-5.5, 0.5), (0.625, 0.5)])]),
                    ],
                ),
            ],
            round16: None,
            pruned: 0.0,
            notes: vec![],
        };
        check(&sch);
        // A signed mixture (as the MX block-max inclusion-exclusion produces).
        let lo = vec![(1.0, 0.5), (-0.375, 0.5)];
        let sch = ChainSchema {
            steps: vec![
                step(3, vec![(2, vec![(1.5, law.clone()), (-0.5, lo.clone())])]),
                step(3, vec![(2, vec![(1.0, law)])]),
            ],
            round16: None,
            pruned: 0.0,
            notes: vec![],
        };
        check(&sch);
    }

    #[test]
    fn certificate() {
        let ints = vec![(1.0, 0.25), (-2.0, 0.25), (3.0, 0.25), (0.0, 0.25)];
        let sch = ChainSchema {
            steps: vec![step(25, vec![(16, vec![(1.0, ints.clone())])]); 4],
            round16: None,
            pruned: 0.0,
            notes: vec![],
        };
        let c = certify(&sch);
        assert!(c.exact, "{c:?}");
        // And the enumeration agrees: all mass at error 0.
        let fine = vec![(1.0, 0.5), (2f64.powi(-30), 0.5)];
        let sch2 = ChainSchema {
            steps: vec![step(25, vec![(2, vec![(1.0, fine)])])],
            round16: None,
            pruned: 0.0,
            notes: vec![],
        };
        assert!(!certify(&sch2).exact);
        let l = enumerate(&sch2, 1 << 16).unwrap();
        let p0: f64 = l.iter().filter(|x| x.1 == 0.0).map(|x| x.2).sum();
        assert!((p0 - 0.5).abs() < 1e-12, "{p0}");
    }
}

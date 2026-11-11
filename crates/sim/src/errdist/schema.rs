/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A kernel's output chain as the analytic tiers see it: per step, the arithmetic model and
//! the law of its terms (groups of i.i.d. terms with signed-mixture laws), built from the
//! numerics DAG (`chain::Prog`) and the spec's element laws (`laws`).

use std::collections::BTreeMap;
use std::sync::Arc as Rc;

use super::chain::Op;
use super::chain::Prog;
use super::chain::StepModel;
use super::fine::C64;
use super::fine::Group;
use super::fine::Pop;
use super::fine::TermLaw;
use super::fine::fft;
use super::laws::Pmf;
use super::laws::TensorLaw;
use super::spec::Format;
use super::spec::Spec;
use super::spec::exp_floor;
use crate::errdist::det::HashMap;

/// One step: its truncation (frac bits), its terms.
#[derive(Clone, Debug)]
pub struct StepSchema {
    pub model: StepModel,
    /// Terms are truncated at 2^(emax - frac).
    pub frac: i32,
    pub pop: Pop,
    /// FP4: units' alignment comes from the scales only (a dead unit, all products zero,
    /// does not set emax); otherwise from each term.
    pub fp4: bool,
}

#[derive(Clone, Debug)]
pub struct ChainSchema {
    pub steps: Vec<StepSchema>,
    /// The output is rounded to this 16-bit format after the chain.
    pub round16: Option<Format>,
    /// Probability mass dropped from the element laws (pruning).
    pub pruned: f64,
    pub notes: Vec<String>,
}

/// A factor of a product: a code leaf times an optional scale leaf (dequantized operands).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Leaf {
    buf: u16,
    idx: u64,
}

fn leaf(p: &Prog, i: u32) -> Option<(Leaf, Format)> {
    match &p.ops[i as usize] {
        Op::Leaf { buf, idx, fmt } => Some((
            Leaf {
                buf: *buf,
                idx: *idx,
            },
            *fmt,
        )),
        _ => None,
    }
}

/// (code leaf, its format, scale leaf and its format) of a factor (`factor`).
type Factor = (Leaf, Format, Option<(Leaf, Format)>);

/// (code leaf, its format, scale leaf) of a factor: a leaf, or a 16-bit rounding of an exact
/// product of a code and a scale (N-EXACT: the dequantization is exact).
fn factor(p: &Prog, i: u32) -> Option<Factor> {
    match &p.ops[i as usize] {
        Op::Leaf { .. } => {
            let (l, f) = leaf(p, i)?;
            Some((l, f, None))
        }
        Op::Round { arg, .. } => factor(p, *arg),
        Op::Fma {
            a,
            b,
            c: None,
            neg_ab: false,
            ..
        } => {
            let (la, fa) = leaf(p, *a)?;
            let (lb, fb) = leaf(p, *b)?;
            let scale = |f: Format| matches!(f, Format::Ue8m0 | Format::Ue4m3);
            if scale(fb) && !scale(fa) {
                Some((la, fa, Some((lb, fb))))
            } else if scale(fa) && !scale(fb) {
                Some((lb, fb, Some((la, fa))))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Per tensor (buffer name): its law and the format of its scale buffer if a recipe derives
/// one; for scale buffers with their own spec entry: an i.i.d. law.
pub struct Laws {
    pub tensors: BTreeMap<String, TensorLaw>,
    pub notes: Vec<String>,
}

impl Laws {
    pub fn build(spec: &Spec, fmts: &BTreeMap<String, Format>) -> Result<Laws, String> {
        let mut notes = vec![];
        let mut tensors = BTreeMap::new();
        let scale_bufs: BTreeMap<String, String> = spec.scale_buffers().into_iter().collect();
        for (name, fmt) in fmts {
            if scale_bufs.contains_key(name) {
                continue;
            }
            let t = spec
                .tensor(name)
                .ok_or_else(|| format!("no distribution for {name}"))?;
            let sf = match &t.quant {
                super::spec::Quant::Mx { scales, .. }
                | super::spec::Quant::Nvfp4 { scales, .. } => fmts.get(scales).copied(),
                _ => None,
            };
            tensors.insert(
                name.clone(),
                super::laws::tensor_law(t, *fmt, sf, &mut notes)?,
            );
        }
        notes.sort();
        notes.dedup();
        Ok(Laws { tensors, notes })
    }
}

/// A factor's law in one context: the code law (or the signed parts of a block law at a scale)
/// times the scale value.
#[derive(Clone)]
struct FactorCtx {
    /// (weight, code PMF) parts.
    parts: Vec<(f64, Rc<Pmf>)>,
    /// The scale's value and exponent field (1, 0 when unscaled).
    scale: f64,
    scale_ef: i32,
    prob: f64,
}

fn factor_contexts(
    laws: &Laws,
    bufs: &[String],
    code: Leaf,
    scale: Option<(Leaf, Format)>,
) -> Result<Vec<FactorCtx>, String> {
    let name = &bufs[code.buf as usize];
    let law = laws
        .tensors
        .get(name)
        .ok_or_else(|| format!("no law for {name}"))?;
    match law {
        TensorLaw::Iid(p) => {
            let p = Rc::new(p.clone());
            match scale {
                None => Ok(vec![FactorCtx {
                    parts: vec![(1.0, p)],
                    scale: 1.0,
                    scale_ef: 0,
                    prob: 1.0,
                }]),
                Some((sl, _)) => {
                    // An independent scale buffer with its own law.
                    let sname = &bufs[sl.buf as usize];
                    let sl = laws
                        .tensors
                        .get(sname)
                        .ok_or_else(|| format!("no law for scale buffer {sname}"))?;
                    let TensorLaw::Iid(sp) = sl else {
                        return Err(format!("scale buffer {sname} has a block law"));
                    };
                    Ok(sp
                        .v
                        .iter()
                        .zip(&sp.p)
                        .zip(&sp.ef)
                        .map(|((v, pr), ef)| FactorCtx {
                            parts: vec![(1.0, p.clone())],
                            scale: *v,
                            scale_ef: *ef,
                            prob: *pr,
                        })
                        .collect())
                }
            }
        }
        TensorLaw::Blocked { scales, .. } => Ok(scales
            .iter()
            .filter(|s| s.prob > 1e-13)
            .map(|s| FactorCtx {
                parts: s
                    .parts
                    .iter()
                    .map(|(w, m)| (*w, Rc::new(m.clone())))
                    .collect(),
                scale: s.value,
                scale_ef: s.ef,
                prob: s.prob,
            })
            .collect()),
    }
}

/// Product term atoms of two code PMFs times scale values, aligned per the model.
pub(crate) fn product_law(
    a: &Pmf,
    b: &Pmf,
    (sa, sea): (f64, i32),
    (sb, seb): (f64, i32),
    model: StepModel,
) -> TermLaw {
    if matches!(model, StepModel::Mma(_)) && sa == 1.0 && sb == 1.0 && a.len() * b.len() > 1 << 20 {
        if let Some(t) = product_law_8bit(a, b) {
            return t;
        }
    }
    let mut acc: HashMap<(u64, i32), f64> =
        HashMap::with_capacity_and_hasher(a.len() * b.len() / 4 + 16, Default::default());
    let mut zero = 0.0;
    for i in 0..a.len() {
        let (av, ap) = (a.v[i], a.p[i]);
        if av == 0.0 {
            zero += ap;
            continue;
        }
        for j in 0..b.len() {
            let (bv, bp) = (b.v[j], b.p[j]);
            let p = ap * bp;
            if bv == 0.0 {
                zero += p;
                continue;
            }
            if p < 1e-22 {
                continue;
            }
            let v = av * sa * bv * sb;
            let e = match model {
                StepModel::Mma(_) => exp_floor(av * sa) + exp_floor(bv * sb),
                _ => a.ef[i] + b.ef[j] + sea + seb,
            };
            *acc.entry((v.to_bits(), e)).or_default() += p;
        }
    }
    let mut atoms: Vec<(f64, i32, f64)> = acc
        .into_iter()
        .map(|((v, e), p)| (f64::from_bits(v), e, p))
        .collect();
    if zero > 0.0 {
        atoms.push((0.0, i32::MIN / 4, zero));
    }
    TermLaw::new(atoms)
}

/// Products of two laws of 8-significant-bit values (bf16) by exponent pairs into a dense
/// table over (e_a + e_b, M_a M_b): exact, no hashing. None if a value has more bits.
fn product_law_8bit(a: &Pmf, b: &Pmf) -> Option<TermLaw> {
    // Per element: (exponent, significand M in [128, 256), prob); zeros apart.
    type Split = (Vec<(i32, u32, f64)>, f64);
    let split = |p: &Pmf| -> Option<Split> {
        let mut v = vec![];
        let mut z = 0.0;
        for i in 0..p.len() {
            if p.v[i] == 0.0 {
                z += p.p[i];
                continue;
            }
            let e = exp_floor(p.v[i]);
            let m = p.v[i].abs() / 2f64.powi(e - 7);
            if m.fract() != 0.0 || !(128.0..256.0).contains(&m) {
                return None;
            }
            v.push((
                e,
                if p.v[i] < 0.0 {
                    256 + m as u32
                } else {
                    m as u32
                },
                p.p[i],
            ));
        }
        Some((v, z))
    };
    let (va, za) = split(a)?;
    let (vb, zb) = split(b)?;
    let emin = va.iter().map(|x| x.0).min()? + vb.iter().map(|x| x.0).min()?;
    let emax = va.iter().map(|x| x.0).max()? + vb.iter().map(|x| x.0).max()?;
    let ne = (emax - emin + 1) as usize;
    // Index: (s - emin) * 2^17 + sign * 2^16 + M_a M_b.
    let mut tab = vec![0f64; ne << 17];
    for &(ea, ma, pa) in &va {
        let (sa_, ma) = (ma >= 256, ma & 255);
        for &(eb, mb, pb) in &vb {
            let (sb_, mb) = (mb >= 256, mb & 255);
            let idx = (((ea + eb - emin) as usize) << 17)
                | (((sa_ != sb_) as usize) << 16)
                | (ma * mb) as usize;
            tab[idx] += pa * pb;
        }
    }
    let mut atoms = Vec::new();
    for (i, &p) in tab.iter().enumerate() {
        if p < 1e-24 {
            continue;
        }
        let s = (i >> 17) as i32 + emin;
        let neg = (i >> 16) & 1 == 1;
        let mm = (i & 0xffff) as f64;
        let v = mm * 2f64.powi(s - 14);
        atoms.push((if neg { -v } else { v }, s, p));
    }
    let z = za + zb - za * zb;
    if z > 0.0 {
        atoms.push((0.0, i32::MIN / 4, z));
    }
    Some(TermLaw::new(atoms))
}

/// FP4 unit law: the exact sum of 16 code products (lattice 1/4), times the scales; aligned by
/// the scale fields when live (some product nonzero).
fn unit_sums(a: &Pmf, b: &Pmf, unit: usize) -> (f64, BTreeMap<i64, f64>) {
    // Product PMF on the lattice of 1/4 (e2m1 x e2m1), split: zero products vs nonzero.
    let mut prod: BTreeMap<i64, f64> = BTreeMap::new();
    let mut p_zero = 0.0;
    for i in 0..a.len() {
        for j in 0..b.len() {
            let v = a.v[i] * b.v[j];
            let p = a.p[i] * b.p[j];
            if v == 0.0 {
                p_zero += p;
            }
            *prod.entry((v * 4.0).round() as i64).or_default() += p;
        }
    }
    // Sum of `unit` products on the integer lattice: the product PMF's DFT to the power
    // `unit` (exact up to FFT round-off, ~1e-16 absolute; atoms below 1e-15 dropped).
    let base: Vec<(i64, f64)> = prod.into_iter().collect();
    let lo = base.first().map_or(0, |x| x.0);
    let hi = base.last().map_or(0, |x| x.0);
    let width = ((hi - lo) as usize) * unit + 1;
    let n = width.next_power_of_two().max(2);
    let mut f = vec![C64::new(0.0, 0.0); n];
    for &(k, p) in &base {
        f[(k - lo) as usize].re += p;
    }
    fft(&mut f, false);
    for c in f.iter_mut() {
        let (m, a) = ((c.re * c.re + c.im * c.im).sqrt(), c.im.atan2(c.re));
        let (m, a) = (m.powi(unit as i32), a * unit as f64);
        *c = C64::new(m * a.cos(), m * a.sin());
    }
    fft(&mut f, true);
    let mut s: BTreeMap<i64, f64> = BTreeMap::new();
    for (i, c) in f.iter().enumerate().take(width) {
        let p = c.re / n as f64;
        if p > 1e-15 {
            s.insert(i as i64 + lo * unit as i64, p);
        }
    }
    (p_zero, s)
}

/// A unit's sum law (`unit_law`): (P(an element product is 0), law by lattice point).
type UnitSum = Rc<(f64, BTreeMap<i64, f64>)>;

thread_local! {
    /// Integer-lattice law of a unit's sum per (element PMF contents, unit size); cleared per
    /// `build`. For i.i.d. codes it does not depend on the scale.
    static UNIT_SUMS: std::cell::RefCell<HashMap<(usize, usize, usize), UnitSum>> = Default::default();
}

fn unit_law(
    a: &Pmf,
    b: &Pmf,
    (sa, sea): (f64, i32),
    (sb, seb): (f64, i32),
    unit: usize,
) -> TermLaw {
    // Keyed by content (block laws carry a copy of the element PMF per scale).
    let h = |m: &Pmf| -> usize {
        use std::hash::Hash;
        use std::hash::Hasher;
        let mut x = std::collections::hash_map::DefaultHasher::new();
        for i in 0..m.len() {
            (m.v[i].to_bits(), m.p[i].to_bits()).hash(&mut x);
        }
        x.finish() as usize
    };
    let key = (h(a), h(b), unit);
    let sums = UNIT_SUMS
        .with(|c| c.borrow().get(&key).cloned())
        .unwrap_or_else(|| {
            let v = Rc::new(unit_sums(a, b, unit));
            UNIT_SUMS.with(|c| c.borrow_mut().insert(key, v.clone()));
            v
        });
    let (p_zero, s) = (sums.0, sums.1.clone());
    let dead = p_zero.powi(unit as i32);
    let mut atoms = vec![];
    let sc = sa * sb;
    for (k, p) in s {
        let mut p = p;
        if k == 0 {
            p -= dead; // live units whose sum cancels to 0
        }
        if p > 1e-30 {
            atoms.push((k as f64 * 0.25 * sc, sea + seb, p));
        }
    }
    if dead > 0.0 {
        atoms.push((0.0, i32::MIN / 4, dead));
    }
    // A live unit summing to 0 still sets emax: keep it as a "zero with alignment" atom.
    TermLaw::new_fp4(atoms)
}

impl TermLaw {
    /// FP4 units: zero-valued atoms may carry an alignment (live units that cancel).
    pub fn new_fp4(mut atoms: Vec<(f64, i32, f64)>) -> TermLaw {
        atoms.retain(|a| a.2 > 0.0);
        atoms.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.partial_cmp(&b.0).unwrap()));
        let lattice = atoms
            .iter()
            .filter(|a| a.0 != 0.0)
            .map(|a| super::fine::lowest_bit(a.0))
            .min()
            .unwrap_or(0);
        TermLaw { atoms, lattice }
    }
}

/// Build the schema of output `o`'s chain.
pub fn build(p: &Prog, o: usize, laws: &Laws) -> Result<ChainSchema, String> {
    UNIT_SUMS.with(|c| c.borrow_mut().clear());
    let (steps, top) = p
        .chain(o)
        .ok_or("the output is not one MMA accumulator chain")?;
    let round16 = top.iter().find_map(|&t| match &p.ops[t as usize] {
        Op::Round { f16, .. } => Some(if *f16 { Format::F16 } else { Format::Bf16 }),
        _ => None,
    });
    let mut out = ChainSchema {
        steps: vec![],
        round16,
        pruned: laws
            .tensors
            .values()
            .map(|l| match l {
                TensorLaw::Iid(p) => p.pruned,
                TensorLaw::Blocked { scales, .. } => scales
                    .iter()
                    .flat_map(|s| s.parts.iter().map(|x| x.1.pruned * x.0.abs()))
                    .sum(),
            })
            .sum(),
        notes: laws.notes.clone(),
    };
    // The chain's first accumulator must be zero.
    if let Op::Mma { acc: Some(a), .. } = &p.ops[steps[0] as usize] {
        if !matches!(p.ops[*a as usize], Op::Const(c) if c == 0.0) {
            return Err("the chain starts from a nonzero accumulator".into());
        }
    }
    let mut cache: HashMap<String, Rc<TermLaw>> = HashMap::default();
    let mut ctx_cache: HashMap<(u16, Option<u16>), Rc<Vec<FactorCtx>>> = HashMap::default();
    for &s in &steps {
        let Op::Mma {
            model,
            terms,
            scales,
            units,
            ..
        } = &p.ops[s as usize]
        else {
            unreachable!()
        };
        let (frac, fp4) = match model {
            StepModel::Mma(m) => (m.frac_bits.ok_or("an exact MMA model")? as i32, false),
            StepModel::Lp(m) => (m.frac.ok_or("an exact MMA model")? as i32, false),
            StepModel::Fp4(m) => (m.frac as i32, true),
        };
        if let StepModel::Mma(m) = model {
            if !m.final_rz || !m.acc_in_group || m.trunc_floor || !m.exp_from_inputs {
                return Err(format!(
                    "analytic tiers: MMA model {} not supported",
                    m.name()
                ));
            }
        }
        // Group terms by context (their scale leaves); FP4: by unit.
        let mut groups: BTreeMap<(Option<Leaf>, Option<Leaf>, u8), Vec<usize>> = BTreeMap::new();
        let mut fac: Vec<(Factor, Factor)> = vec![];
        for (j, &(a, b)) in terms.iter().enumerate() {
            let mut fa =
                factor(p, a).ok_or("a factor that is not a leaf or an exact dequantization")?;
            let mut fb =
                factor(p, b).ok_or("a factor that is not a leaf or an exact dequantization")?;
            if let Some(&(sa, sb)) = scales.get(j) {
                fa.2 = Some(leaf(p, sa).ok_or("a scale that is not a leaf")?);
                fb.2 = Some(leaf(p, sb).ok_or("a scale that is not a leaf")?);
            }
            let u = if fp4 { units[j] } else { 0 };
            groups
                .entry((fa.2.map(|x| x.0), fb.2.map(|x| x.0), u))
                .or_default()
                .push(j);
            fac.push((fa, fb));
        }
        // Merge FP4 units that share a context into one group of n units.
        let mut merged: BTreeMap<(Option<Leaf>, Option<Leaf>), (usize, usize)> = BTreeMap::new(); // ctx -> (units, terms per unit)
        for ((la, lb, _), js) in &groups {
            let e = merged.entry((*la, *lb)).or_insert((0, js.len()));
            e.0 += 1;
            if e.1 != js.len() {
                return Err("FP4 units of one context with different sizes".into());
            }
        }
        let mut pop = Pop::default();
        for ((la, lb), (nunits, per)) in merged {
            // A representative term of this context.
            let j = groups
                .iter()
                .find(|((a, b, _), _)| *a == la && *b == lb)
                .unwrap()
                .1[0];
            let (fa, fb) = fac[j];
            // Contexts by buffer (stable Rc's: equal steps share their term laws).
            let mut ctx =
                |f: (Leaf, Format, Option<(Leaf, Format)>)| -> Result<Rc<Vec<FactorCtx>>, String> {
                    let k = (f.0.buf, f.2.map(|x| x.0.buf));
                    if let Some(c) = ctx_cache.get(&k) {
                        return Ok(c.clone());
                    }
                    let c = Rc::new(factor_contexts(laws, &p.bufs, f.0, f.2)?);
                    ctx_cache.insert(k, c.clone());
                    Ok(c)
                };
            let ca = ctx(fa)?;
            let cb = ctx(fb)?;
            let n = if fp4 { nunits } else { per };
            let mut parts: Vec<(f64, Rc<TermLaw>)> = vec![];
            for x in ca.iter() {
                for y in cb.iter() {
                    for (wa, pa) in &x.parts {
                        for (wb, pb) in &y.parts {
                            let w = x.prob * y.prob * wa * wb;
                            if w.abs() < 1e-14 {
                                continue;
                            }
                            let key = format!(
                                "{:p}{:p}{}{}{}{}{}{}",
                                Rc::as_ptr(pa),
                                Rc::as_ptr(pb),
                                x.scale,
                                y.scale,
                                x.scale_ef,
                                y.scale_ef,
                                fp4,
                                per
                            );
                            let law = cache
                                .entry(key)
                                .or_insert_with(|| {
                                    Rc::new(if fp4 {
                                        unit_law(
                                            pa,
                                            pb,
                                            (x.scale, x.scale_ef),
                                            (y.scale, y.scale_ef),
                                            per,
                                        )
                                    } else {
                                        product_law(
                                            pa,
                                            pb,
                                            (x.scale, x.scale_ef),
                                            (y.scale, y.scale_ef),
                                            *model,
                                        )
                                    })
                                })
                                .clone();
                            parts.push((w, law));
                        }
                    }
                }
            }
            pop.groups.push(Group { n, parts });
        }
        if std::env::var("PTXV_ERRDIST_DEBUG").is_ok() {
            for g in &pop.groups {
                eprintln!(
                    "group n={} parts={} atoms={:?}",
                    g.n,
                    g.parts.len(),
                    g.parts
                        .iter()
                        .map(|p| p.1.atoms.len())
                        .take(4)
                        .collect::<Vec<_>>()
                );
            }
        }
        out.steps.push(StepSchema {
            model: *model,
            frac,
            pop,
            fp4,
        });
    }
    Ok(out)
}

/// Whether two outputs have the same chain shape (steps, models, term counts per group).
pub fn shape_key(p: &Prog, o: usize) -> Option<Vec<(usize, usize)>> {
    let (steps, top) = p.chain(o)?;
    let mut k: Vec<(usize, usize)> = steps
        .iter()
        .map(|&s| match &p.ops[s as usize] {
            Op::Mma { terms, units, .. } => (terms.len(), units.len()),
            _ => (0, 0),
        })
        .collect();
    k.push((top.len(), 0));
    Some(k)
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tensor-core arithmetic for the narrow-format kinds (`tcgen05.mma.kind::f8f6f4` and the
//! block-scaled kinds): a parameterized family like `fp::MmaModel`, extended with the choices
//! that only narrow operands expose (32-wide steps, subnormal operands, f16 accumulators).
//! The member that matches GB300 is chosen by `ptxv calib-mma probes/mma_f8` (N-MMA-F8).

use ptxv_ir::Lp;

use crate::lowp;

/// One operand element: exact value and the exponents the alignment may use.
#[derive(Clone, Copy, Debug)]
pub struct Operand {
    pub v: f64,
    /// Exponent from the code's exponent field (subnormals: the format's minimum, 1 - bias).
    pub ef: i32,
    /// floor(log2 |v|) (meaningless for 0).
    pub et: i32,
}

impl Operand {
    pub fn new(c: u32, t: Lp) -> Operand {
        let v = lowp::value(c, t);
        let f = lowp::fmt(t);
        let mag = c & ((1 << (f.ebits + f.mbits)) - 1);
        let field = (mag >> f.mbits) as i32;
        Operand {
            v,
            ef: field.max(1) - f.bias,
            et: if v == 0.0 || !v.is_finite() {
                i32::MIN / 4
            } else {
                exp_of(v)
            },
        }
    }
    /// The exact product, with field and true exponents added (narrow significands: exact).
    pub fn mul(&self, o: &Operand) -> Operand {
        Operand {
            v: self.v * o.v,
            ef: self.ef + o.ef,
            et: self.et + o.et,
        }
    }

    /// A scale-factor code (ue8m0: 2^(c - 127), no subnormals; ue4m3: unsigned e4m3).
    pub fn scale(c: u32, t: Lp) -> Operand {
        if t == Lp::Ue8m0 {
            let e = c as i32 - 127;
            return Operand {
                v: if c == 0xff { f64::NAN } else { 2f64.powi(e) },
                ef: e,
                et: e,
            };
        }
        Operand::new(c, t)
    }

    pub fn from_value(v: f64) -> Operand {
        let e = if v == 0.0 { i32::MIN / 4 } else { exp_of(v) };
        Operand { v, ef: e, et: e }
    }
}

fn exp_of(x: f64) -> i32 {
    let b = x.abs().to_bits();
    let e = ((b >> 52) & 0x7ff) as i32;
    if e == 0 {
        -1074 + (63 - (b & ((1 << 52) - 1)).leading_zeros() as i32)
    } else {
        e - 1023
    }
}

/// How products are aligned before truncation.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize
)]
pub enum Align {
    /// By the exact product's exponent.
    Prod,
    /// By e_a + e_b from the exponent fields (subnormals at the format minimum).
    Fields,
    /// By floor(log2 a) + floor(log2 b).
    Inputs,
}

/// One narrow-format MMA step (K = 32 products into one accumulator): split the products
/// into groups of `group`; for each group, align the terms (and the running accumulator, if
/// `acc_in`) to the largest alignment exponent, keep `frac` bits below it (None = exact,
/// truncating toward zero, or toward -inf with `floor`), add exactly and round once to the D
/// format (toward zero with `rz`). Without `acc_in` the group sum is rounded first and then
/// added to the accumulator with a second rounding. f16 D: `via32` rounds to f32 first and
/// then to f16 (round to nearest), else directly.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize
)]
pub struct LpMmaModel {
    pub group: u8,
    pub frac: Option<u8>,
    pub rz: bool,
    pub acc_in: bool,
    pub floor: bool,
    pub align: Align,
    pub via32: bool,
}

/// The kind::f8f6f4 arithmetic on GB300 (sm_103a), the unique member of the family matching
/// every output of `ptxv calib-mma probes/mma_f8` (FP8 / FP6 / FP4 pairs, f32 and f16 D),
/// 2026-11-03: all 32 products of an instruction and the accumulator are aligned to the
/// largest e_a + e_b (exponent fields; subnormals at the format minimum) or accumulator
/// exponent, truncated toward zero 25 bits below it, summed exactly, and rounded once: toward
/// zero to f32, to nearest even to f16.
// ASSUME(N-MMA-F8): see ASSUMPTIONS.md
pub const DEFAULT_LP_MMA: LpMmaModel = LpMmaModel {
    group: 32,
    frac: Some(25),
    rz: true,
    acc_in: true,
    floor: false,
    align: Align::Fields,
    via32: false,
};

/// The same with an f16 D, which GB300 rounds to nearest even (`DEFAULT_LP_MMA`).
// ASSUME(N-MMA-F8): see ASSUMPTIONS.md
pub const DEFAULT_LP_MMA_F16: LpMmaModel = LpMmaModel {
    rz: false,
    ..DEFAULT_LP_MMA
};

impl LpMmaModel {
    pub fn name(&self) -> String {
        format!(
            "g{}-{}-{}-{}-{}-{}{}",
            self.group,
            self.frac
                .map(|f| format!("f{f}"))
                .unwrap_or_else(|| "exact".into()),
            if self.rz { "rz" } else { "rn" },
            if self.acc_in { "accin" } else { "accsep" },
            if self.floor { "tfloor" } else { "tzero" },
            match self.align {
                Align::Prod => "eprod",
                Align::Fields => "efields",
                Align::Inputs => "einputs",
            },
            if self.via32 { "-via32" } else { "" }
        )
    }

    pub fn parse(s: &str) -> Option<LpMmaModel> {
        Self::family().into_iter().find(|m| m.name() == s)
    }

    /// The calibration candidates.
    pub fn family() -> Vec<LpMmaModel> {
        let mut v = Vec::new();
        for group in [8u8, 16, 32, 64] {
            for frac in std::iter::once(None).chain((12u8..=36).map(Some)) {
                for rz in [true, false] {
                    for acc_in in [true, false] {
                        for floor in [false, true] {
                            for align in [Align::Prod, Align::Fields, Align::Inputs] {
                                if frac.is_none() && (floor || align != Align::Prod) {
                                    continue;
                                }
                                for via32 in [false, true] {
                                    v.push(LpMmaModel {
                                        group,
                                        frac,
                                        rz,
                                        acc_in,
                                        floor,
                                        align,
                                        via32,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        v
    }

    /// One MMA step of one output: acc (f32 bits, or f16 bits with `f16_d`; None = overwrite)
    /// plus the products a[j] * b[j]. Returns D's bits.
    pub fn step(&self, acc: Option<u32>, a: &[Operand], b: &[Operand], f16_d: bool) -> u32 {
        let terms: Vec<Operand> = a.iter().zip(b).map(|(x, y)| x.mul(y)).collect();
        self.step_terms(acc, &terms, f16_d)
    }

    /// One step over precomputed terms (products, scaled products of the block-scaled kinds:
    /// `Operand::mul` keeps the value exact and adds the exponents).
    pub fn step_terms(&self, acc: Option<u32>, terms: &[Operand], f16_d: bool) -> u32 {
        let terms: Vec<(f64, i32)> = terms
            .iter()
            .map(|x| {
                let e = match self.align {
                    _ if x.v == 0.0 => i32::MIN / 2,
                    Align::Prod => exp_of(x.v),
                    Align::Fields => x.ef,
                    Align::Inputs => x.et,
                };
                (x.v, e)
            })
            .collect();
        let tgt = if f16_d { F16 } else { F32 };
        let mut c: Option<f64> = acc.map(|u| {
            if f16_d {
                crate::fp::f16_to_f64(u as u16)
            } else {
                f32::from_bits(u) as f64
            }
        });
        for chunk in terms.chunks(self.group as usize) {
            let mut t = chunk.to_vec();
            let r = if self.acc_in {
                if let Some(cv) = c {
                    t.push((cv, if cv == 0.0 { i32::MIN / 2 } else { exp_of(cv) }));
                }
                self.sum_round(&t, tgt)
            } else {
                let s = self.sum_round(&t, tgt);
                match c {
                    Some(cv) => {
                        let tt = [
                            (cv, if cv == 0.0 { i32::MIN / 2 } else { exp_of(cv) }),
                            (s, if s == 0.0 { i32::MIN / 2 } else { exp_of(s) }),
                        ];
                        round_exact(&tt, self.rz, tgt, self.via32)
                    }
                    None => s,
                }
            };
            c = Some(r);
        }
        let v = c.unwrap_or(0.0);
        if f16_d {
            crate::fp::f32_to_f16_rn((v as f32).to_bits()) as u32 // exact: v is an f16 value
        } else {
            (v as f32).to_bits()
        }
    }

    fn sum_round(&self, t: &[(f64, i32)], tgt: Tgt) -> f64 {
        let Some(f) = self.frac else {
            return round_exact(t, self.rz, tgt, self.via32);
        };
        let nz: Vec<&(f64, i32)> = t.iter().filter(|x| x.0 != 0.0).collect();
        if nz.is_empty() {
            return if !t.is_empty() && t.iter().all(|x| x.0.is_sign_negative()) {
                -0.0
            } else {
                0.0
            };
        }
        let emax = nz.iter().map(|x| x.1).max().unwrap();
        let q = emax - f as i32;
        let mut acc: i128 = 0;
        for x in &nz {
            let s = x.0 * 2f64.powi(-q);
            acc += if self.floor { s.floor() } else { s.trunc() } as i128;
        }
        round_int(acc, q, self.rz, tgt, self.via32)
    }
}

#[derive(Clone, Copy)]
struct Tgt {
    mbits: i32,
    emin: i32,
    emax: i32,
}
const F32: Tgt = Tgt {
    mbits: 23,
    emin: -126,
    emax: 127,
};
const F16: Tgt = Tgt {
    mbits: 10,
    emin: -14,
    emax: 15,
};

/// Exact sum of the terms, rounded once.
fn round_exact(t: &[(f64, i32)], rz: bool, tgt: Tgt, via32: bool) -> f64 {
    let nz: Vec<f64> = t.iter().map(|x| x.0).filter(|v| *v != 0.0).collect();
    if nz.is_empty() {
        return if !t.is_empty() && t.iter().all(|x| x.0.is_sign_negative()) {
            -0.0
        } else {
            0.0
        };
    }
    let lo = nz.iter().map(|&v| exp_of(v) - 52).min().unwrap();
    let mut acc: i128 = 0;
    for &v in &nz {
        acc += (v * 2f64.powi(-lo)) as i128; // exact while the span stays below ~120 bits
    }
    round_int(acc, lo, rz, tgt, via32)
}

/// Round v * 2^e to the target (RZ or RN-even; subnormals; overflow to max (RZ) or Inf).
fn round_int(v: i128, e: i32, rz: bool, tgt: Tgt, via32: bool) -> f64 {
    if tgt.mbits == F16.mbits && via32 {
        let x = round_int(v, e, rz, F32, false);
        let h = crate::fp::f32_to_f16_rn((x as f32).to_bits());
        return crate::fp::f16_to_f64(h);
    }
    if v == 0 {
        return 0.0;
    }
    let neg = v < 0;
    let mag = v.unsigned_abs();
    let lead = 127 - mag.leading_zeros() as i32 + e;
    let q = lead.max(tgt.emin) - tgt.mbits;
    let sh = q - e;
    let mut m = if sh <= 0 {
        mag << (-sh) as u32
    } else if sh >= 128 {
        0
    } else {
        let s = sh as u32;
        let kept = mag >> s;
        let rem = mag & ((1u128 << s) - 1);
        let half = 1u128 << (s - 1);
        kept + (!rz && (rem > half || rem == half && kept & 1 == 1)) as u128
    };
    let mut qe = q;
    if m >= 1u128 << (tgt.mbits + 1) {
        m >>= 1;
        qe += 1;
    }
    let val = if qe + tgt.mbits > tgt.emax {
        if rz {
            (2.0 - 2f64.powi(-tgt.mbits)) * 2f64.powi(tgt.emax)
        } else {
            f64::INFINITY
        }
    } else {
        m as f64 * 2f64.powi(qe)
    };
    if neg { -val } else { val }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_member_rounds_correctly() {
        let md = LpMmaModel {
            group: 32,
            frac: None,
            rz: false,
            acc_in: true,
            floor: false,
            align: Align::Prod,
            via32: false,
        };
        let one = Operand::new(0x38, Lp::E4m3); // 1.0
        let tiny = Operand::new(0x01, Lp::E4m3); // 2^-9
        assert_eq!(tiny.ef, -6);
        assert_eq!(tiny.et, -9);
        let a = [one, tiny];
        let b = [one, one];
        let d = md.step(Some(1.0f32.to_bits()), &a, &b, false);
        assert_eq!(f32::from_bits(d), 2.0 + 2f32.powi(-9));
        let h = md.step(None, &a, &b, true);
        assert_eq!(
            h,
            crate::fp::f32_to_f16_rn((1.0f32 + 2f32.powi(-9)).to_bits()) as u32
        );
    }
}

/// The K = 64 packed-FP4 kinds (kind::mxf4, kind::mxf4nvf4): products are summed exactly in
/// units of `unit` K elements, each unit sum is multiplied by its block's scale factors
/// (sa * sb), and the unit sums and the accumulator are aligned to the largest of the
/// accumulator's exponent and the scale exponents of the units with a nonzero product (`live`;
/// else of every unit), truncated toward zero `frac` bits below it, summed exactly and rounded
/// toward zero to f32. The scale exponent is that of sa * sb (`fields`: e(sa) + e(sb)).
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize
)]
pub struct Fp4Model {
    pub unit: u8,
    pub frac: u8,
    pub live: bool,
    pub fields: bool,
}

/// The kind::mxf4 / mxf4nvf4 arithmetic on GB300 (sm_103a): the unique member of the family
/// matching every output of `ptxv calib-mma probes/mma_bs` (2026-11-03). See N-MMA-BS.
// ASSUME(N-MMA-BS): see ASSUMPTIONS.md
pub const DEFAULT_FP4_MMA: Fp4Model = Fp4Model {
    unit: 16,
    frac: 35,
    live: true,
    fields: true,
};

impl Fp4Model {
    pub fn name(&self) -> String {
        format!(
            "u{}-f{}-{}-{}",
            self.unit,
            self.frac,
            if self.live { "live" } else { "allscales" },
            if self.fields { "sfields" } else { "sprod" }
        )
    }

    pub fn family() -> Vec<Fp4Model> {
        let mut v = Vec::new();
        for unit in [1u8, 2, 4, 8, 16, 32] {
            for frac in 28u8..=42 {
                for live in [true, false] {
                    for fields in [false, true] {
                        v.push(Fp4Model {
                            unit,
                            frac,
                            live,
                            fields,
                        });
                    }
                }
            }
        }
        v
    }

    /// One instruction for one output: acc (f32 bits; None = overwrite), the 64 exact FP4
    /// products, and per block of `blk` elements the A and B scale factors.
    pub fn step(
        &self,
        acc: Option<u32>,
        prods: &[f64],
        sa: &[Operand],
        sb: &[Operand],
        blk: usize,
    ) -> u32 {
        let u = self.unit as usize;
        let mut units = Vec::with_capacity(prods.len() / u);
        let mut emax: Option<i32> = None;
        for (i, ch) in prods.chunks(u).enumerate() {
            let b = i * u / blk;
            let s = sa[b].v * sb[b].v;
            let se = if self.fields {
                sa[b].ef + sb[b].ef
            } else {
                exp_of(s)
            };
            if !self.live || ch.iter().any(|p| *p != 0.0) {
                emax = Some(emax.map_or(se, |e: i32| e.max(se)));
            }
            units.push(ch.iter().sum::<f64>() * s); // exact: small integers x 2^-2, 8-bit scales
        }
        let accv = acc.map(|w| f32::from_bits(w) as f64);
        if let Some(a) = accv.filter(|a| *a != 0.0) {
            emax = Some(emax.map_or(exp_of(a), |e| e.max(exp_of(a))));
        }
        let Some(emax) = emax else {
            return acc.unwrap_or(0);
        };
        let q = emax - self.frac as i32;
        let mut sum: i128 = 0;
        for t in units.iter().chain(accv.iter()) {
            sum += (t * 2f64.powi(-q)).trunc() as i128;
        }
        (round_int(sum, q, true, F32, false) as f32).to_bits()
    }
}

// ---------------------------------------------------------------- error contracts

/// |RZ(s) - s| for an f32 result below the normal range: less than its spacing 2^-149.
pub const F32_RZ_SUB: f64 = 1.401298464324817e-45; // 2^-149

/// (alignment exponent ef, subnormal significand bits s) of every finite nonzero code of `t`,
/// as `Operand::new` reads them: |v| >= 2^(ef - s), s = mbits for a subnormal code, else 0.
fn exps(t: Lp) -> Vec<(i32, i32)> {
    let f = lowp::fmt(t);
    let mut v: Vec<(i32, i32)> = (0..1u32 << lowp::width(t))
        .filter_map(|c| {
            let o = Operand::new(c, t);
            if o.v == 0.0 || !o.v.is_finite() {
                return None;
            }
            let mag = c & ((1 << (f.ebits + f.mbits)) - 1);
            let sub = (mag >> f.mbits) == 0;
            Some((o.ef, if sub { f.mbits as i32 } else { 0 }))
        })
        .collect();
    v.sort();
    v.dedup();
    v
}

/// The truncation part tau of the contract of one `kind::f8f6f4` / `mxf8f6f4` step of
/// `DEFAULT_LP_MMA` (efields, 25 bits, toward zero): with up to `np` possibly nonzero products
/// of A format `fa` and B format `fb` and (`acc`) a nonzero accumulator, the truncation loses
/// less than tau * sum |terms|. Derivation (docs/equiv-mma.md §5):
/// - the terms are truncated to multiples of Q = 2^(emax - 25); a term that is a multiple of Q
///   loses nothing, any other less than Q;
/// - the term k that sets emax is such a multiple: a product is M_a M_b 2^(e_k - mb_a - mb_b)
///   (integers M), and mb_a + mb_b <= 25; the accumulator, a normal f32, is a multiple of
///   2^(e - 23);
/// - normal setter (both operands normal, or the accumulator): 2^emax <= |t_k|, so the other
///   n - 1 terms lose less than (n - 1) 2^-25 sum |terms|;
/// - setter with a subnormal operand: |a| >= 2^(ef_a - mb_a), so 2^emax <= 2^s_k |t_k|, s_k
///   the subnormal operands' mantissa bits. A product j is not a multiple of Q only if
///   e_j <= emax - 26 + mb_a + mb_b; when no product of the formats can be that far below
///   this setter, only the accumulator can lose, else the np - 1 other products too.
///   The block scales of mxf8f6f4 (one ue8m0 pair per K = 32 step) scale every product
///   alike and change neither ratio.
pub fn lp_trunc(fa: Lp, fb: Lp, np: usize, acc: bool) -> f64 {
    let n = np + acc as usize;
    let mut tau = n.saturating_sub(1) as f64 * 2f64.powi(-25);
    let (xa, xb) = (exps(fa), exps(fb));
    let mbsum = (lowp::fmt(fa).mbits + lowp::fmt(fb).mbits) as i32;
    let emin_prod =
        xa.iter().map(|x| x.0).min().unwrap_or(0) + xb.iter().map(|x| x.0).min().unwrap_or(0);
    for &(ea, sa) in &xa {
        for &(eb, sb) in &xb {
            let s = sa + sb;
            if s == 0 || np == 0 {
                continue;
            }
            let lossy_prod = emin_prod <= ea + eb - 26 + mbsum;
            let k = if lossy_prod {
                np - 1 + acc as usize
            } else {
                acc as usize
            };
            tau = tau.max(k as f64 * 2f64.powi(s - 25));
        }
    }
    tau
}

/// The relative contract eps of such a step: |computed - sum terms| <= eps sum |terms| +
/// F32_RZ_SUB, eps = tau + 2^-23 (1 + tau) (the final rounding toward zero to f32 is
/// relative 2^-23 for a normal result, below 2^-149 for a subnormal one).
// ASSUME(N-MMA-F8): see ASSUMPTIONS.md
pub fn lp_eps(fa: Lp, fb: Lp, np: usize, acc: bool) -> f64 {
    let tau = lp_trunc(fa, fb, np, acc);
    tau + 2f64.powi(-23) * (1.0 + tau)
}

/// The contract of one `Fp4Model` step (`DEFAULT_FP4_MMA`: units of 16, 35 bits, live
/// scale-exponent sums) with `units` units and scales of format `st`: eps as `lp_eps`.
/// Derivation: a unit's exact sum P_u (a multiple of 1/4) times its scale product s_u is a
/// multiple of 2^(se_u - 2) for ue8m0 (s_u = 2^se_u), of 2^(se_u - 8) for ue4m3 (s_u = M_a M_b
/// 2^(se_u - 6)); so the setter (a live unit, or the accumulator) loses nothing. A live unit
/// has a product |a b| >= 1/4, so 2^se_u <= C sum_{its products} |a b s_u| with C = 4 (ue8m0)
/// or 2^8 (ue4m3, both scales nonzero: M >= 1). The other units and the accumulator lose less
/// than Q = 2^(emax - 35) each.
/// Requires every ue4m3 scale of a unit with a nonzero product to be nonzero: a zero scale
/// still sets emax while its unit adds nothing to sum |terms| (no relative bound); `None` for
/// formats this does not cover.
// ASSUME(N-MMA-BS): see ASSUMPTIONS.md
pub fn fp4_eps(st: Lp, units: usize, acc: bool) -> Option<f64> {
    let c = match st {
        Lp::Ue8m0 => 2,
        Lp::Ue4m3 => 8,
        _ => return None,
    };
    let lossy = (units.saturating_sub(1) + acc as usize).max(units * acc as usize);
    let tau = lossy as f64 * 2f64.powi(c - 35);
    Some(tau + 2f64.powi(-23) * (1.0 + tau))
}

/// The data-dependent bound of one `DEFAULT_LP_MMA` step (f32 D) at the given terms (products
/// with their field exponents, `Operand::mul`) and accumulator: sum over the terms that are not
/// multiples of Q of min(Q, |t|), plus the final rounding 2^-23 (|sum| + that) + 2^-149. The
/// model's error never exceeds it; it is at most the contract (tests).
pub fn lp_step_bound(acc: Option<f64>, terms: &[Operand]) -> f64 {
    let mut ts: Vec<(f64, i32)> = terms
        .iter()
        .filter(|t| t.v != 0.0)
        .map(|t| (t.v, t.ef))
        .collect();
    if let Some(a) = acc.filter(|a| *a != 0.0) {
        ts.push((a, exp_of(a)));
    }
    let Some(emax) = ts.iter().map(|t| t.1).max() else {
        return 0.0;
    };
    let q = 2f64.powi(emax - DEFAULT_LP_MMA.frac.unwrap() as i32);
    let mut loss = 0.0;
    let mut sum = 0.0;
    for &(v, _) in &ts {
        sum += v.abs();
        let r = v / q;
        if r != r.trunc() {
            loss += q.min(v.abs());
        }
    }
    (loss + 2f64.powi(-23) * (sum + loss) + F32_RZ_SUB) * (1.0 + 1e-12)
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    use crate::exact;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// A finite code of `t`: any (`mode` 0), subnormal or zero (1), the largest binade (2), the
    /// smallest normal binade (3).
    fn code(r: &mut Rng, t: Lp, mode: u64) -> u32 {
        let f = lowp::fmt(t);
        let w = lowp::width(t);
        loop {
            let c = r.below(1 << w) as u32;
            let mag = c & ((1 << (f.ebits + f.mbits)) - 1);
            let field = mag >> f.mbits;
            let v = lowp::value(c, t);
            if !v.is_finite() {
                continue;
            }
            let fmax = (f.max >> f.mbits) & ((1 << f.ebits) - 1);
            let ok = match mode {
                1 => field == 0,
                2 => field == fmax,
                3 => field == 1,
                _ => true,
            };
            if ok {
                return c;
            }
        }
    }

    /// A normal f32 accumulator of magnitude about 2^e, random sign and significand.
    fn acc_of(r: &mut Rng, e: i32) -> f32 {
        let e = e.clamp(-125, 126);
        let m = 1.0 + (r.below(1 << 23) as f64) / (1u64 << 23) as f64;
        let s = if r.below(2) == 0 { 1.0 } else { -1.0 };
        (s * m * 2f64.powi(e)) as f32
    }

    /// Model error and sum |terms| of one step (exact arithmetic); None outside the model's
    /// range (f32 overflow).
    fn check(acc: Option<f32>, terms: &[Operand]) -> Option<(f64, f64)> {
        let d =
            f32::from_bits(DEFAULT_LP_MMA.step_terms(acc.map(f32::to_bits), terms, false)) as f64;
        let mut s = exact::from(acc.unwrap_or(0.0) as f64);
        let mut abs = (acc.unwrap_or(0.0) as f64).abs();
        for t in terms {
            s = exact::add(&s, &exact::from(t.v));
            abs += t.v.abs();
        }
        let ex = exact::approx(&s);
        if !d.is_finite() || ex.abs() > f32::MAX as f64 * 0.5 {
            return None;
        }
        Some((exact::approx(&exact::sub(&exact::from(d), &s)).abs(), abs))
    }

    const PAIRS: [(Lp, Lp); 10] = [
        (Lp::E4m3, Lp::E4m3),
        (Lp::E5m2, Lp::E5m2),
        (Lp::E4m3, Lp::E5m2),
        (Lp::E5m2, Lp::E4m3),
        (Lp::E2m3, Lp::E2m3),
        (Lp::E3m2, Lp::E3m2),
        (Lp::E2m1, Lp::E2m1),
        (Lp::E4m3, Lp::E2m1),
        (Lp::E5m2, Lp::E2m3),
        (Lp::E3m2, Lp::E5m2),
    ];

    /// The model's error never exceeds the step's data-dependent bound, which never exceeds
    /// the contract, on adversarial steps: random codes, subnormal-heavy operands, a setter
    /// with subnormal operands next to a lossy accumulator, wide exponent spreads, cancelling
    /// pairs, ue8m0 block scales.
    #[test]
    fn lp_contract_covers_the_model() {
        let mut r = Rng(0x9e37_79b9_7f4a_7c15);
        for (fa, fb) in PAIRS {
            let mut worst: f64 = 0.0;
            for it in 0..6000 {
                let set = it % 6;
                let mut terms = Vec::with_capacity(32);
                for j in 0..32 {
                    let (ma, mb) = match set {
                        1 => (1, if j % 2 == 0 { 1 } else { 0 }),
                        2 if j == 0 => (if r.below(2) == 0 { 1 } else { 2 }, 1),
                        2 => (3, 3),
                        3 if j == 0 => (2, 2),
                        3 => (r.below(4), r.below(4)),
                        _ => (0, 0),
                    };
                    let (ca, cb) = (code(&mut r, fa, ma), code(&mut r, fb, mb));
                    let (mut oa, ob) = (Operand::new(ca, fa), Operand::new(cb, fb));
                    // Cancelling pairs: the odd term negates the even one.
                    if set == 4 && j % 2 == 1 {
                        let p = &terms[j - 1];
                        let p: &Operand = p;
                        terms.push(Operand { v: -p.v, ..*p });
                        continue;
                    }
                    if set == 5 && r.below(3) == 0 {
                        oa.v = 0.0;
                    }
                    terms.push(oa.mul(&ob));
                }
                // A common ue8m0 block-scale pair (mxf8f6f4) on some steps.
                if r.below(3) == 0 {
                    let (sa, sb) = (
                        Operand::scale(97 + r.below(60) as u32, Lp::Ue8m0),
                        Operand::scale(97 + r.below(60) as u32, Lp::Ue8m0),
                    );
                    terms = terms.iter().map(|t| t.mul(&sa).mul(&sb)).collect();
                }
                let emax = terms
                    .iter()
                    .filter(|t| t.v != 0.0)
                    .map(|t| t.ef)
                    .max()
                    .unwrap_or(0);
                let acc = match r.below(4) {
                    0 => None,
                    // Far below the products: truncated away (lossy).
                    1 => {
                        let k = r.below(40) as i32;
                        Some(acc_of(&mut r, emax - 3 - k))
                    }
                    2 => {
                        let k = r.below(8) as i32;
                        Some(acc_of(&mut r, emax + k - 4))
                    }
                    _ => {
                        let k = r.below(10) as i32;
                        Some(acc_of(&mut r, emax + 20 + k))
                    }
                };
                let Some((err, abs)) = check(acc, &terms) else {
                    continue;
                };
                let ac = acc.is_some_and(|a| a != 0.0);
                let data = lp_step_bound(acc.map(|a| a as f64), &terms);
                let contract = lp_eps(fa, fb, 32, ac) * abs + F32_RZ_SUB;
                assert!(
                    err <= data,
                    "{fa:?} x {fb:?} set {set}: error {err:e} > data bound {data:e}"
                );
                assert!(
                    data <= contract * (1.0 + 1e-9),
                    "{fa:?} x {fb:?} set {set}: data bound {data:e} > contract {contract:e}"
                );
                worst = worst.max(err / contract);
            }
            eprintln!(
                "{fa:?} x {fb:?}: eps {:.2}u (acc), {:.2}u (no acc); largest error / contract {worst:.3}",
                lp_eps(fa, fb, 32, true) / 2f64.powi(-24),
                lp_eps(fa, fb, 32, false) / 2f64.powi(-24)
            );
            assert!(worst > 0.0, "{fa:?} x {fb:?}: no step had an error");
        }
    }

    /// N-MMA's derivation (2^e <= |term|) fails for a subnormal setter: e4m3 a = 2^-9 (field
    /// 0, aligned at 2^-6), b = 1, accumulator 1.75 * 2^-31 truncated at Q = 2^-31. The error
    /// exceeds the bf16-style contract `MmaModel::eps_terms(2)`, but not `lp_eps`.
    #[test]
    fn subnormal_setter_breaks_the_bf16_derivation() {
        let a = Operand::new(0x01, Lp::E4m3);
        let b = Operand::new(0x38, Lp::E4m3);
        assert_eq!((a.v, a.ef), (2f64.powi(-9), -6));
        let acc = (1.75 * 2f64.powi(-31)) as f32;
        let (err, abs) = check(Some(acc), &[a.mul(&b)]).unwrap();
        let bf16_style = crate::fp::DEFAULT_MMA.eps_terms(2) * abs;
        assert!(err > bf16_style, "{err:e} <= {bf16_style:e}");
        assert!(err <= lp_eps(Lp::E4m3, Lp::E4m3, 1, true) * abs + F32_RZ_SUB);
        // e4m3 x e4m3: a subnormal setter leaves only the accumulator lossy (2^(6 - 25)).
        assert_eq!(lp_trunc(Lp::E4m3, Lp::E4m3, 32, true), 2f64.powi(-19));
        assert_eq!(
            lp_trunc(Lp::E4m3, Lp::E4m3, 32, false),
            31.0 * 2f64.powi(-25)
        );
        // e5m2 x e5m2: products 22 binades below a one-subnormal setter (2^(2 - 25) each).
        assert_eq!(
            lp_trunc(Lp::E5m2, Lp::E5m2, 32, true),
            32.0 * 2f64.powi(-23)
        );
    }

    /// The contract is reached: e4m3 x e5m2, the setter 2^-9 (e4m3 subnormal, aligned at 2^-6)
    /// times 2^15 aligns at 2^9 while its value is 2^6, so Q = 2^-16; the 31 other products are
    /// the largest product below Q, each truncated to 0. The model's error is over 0.9 of
    /// `lp_eps` (and within it).
    #[test]
    fn lp_contract_is_reached() {
        let (fa, fb) = (Lp::E4m3, Lp::E5m2);
        let set = Operand::new(0x01, fa).mul(&Operand::new(0x78, fb));
        assert_eq!((set.v, set.ef), (64.0, 9));
        let q = 2f64.powi(9 - 25);
        let mut best: Option<Operand> = None;
        for ca in 0..0x7fu32 {
            for cb in 0..0x7cu32 {
                let t = Operand::new(ca, fa).mul(&Operand::new(cb, fb));
                if t.v > 0.0 && t.v < q && best.is_none_or(|b| t.v > b.v) {
                    best = Some(t);
                }
            }
        }
        let t = best.unwrap();
        let mut terms = vec![set];
        terms.extend(std::iter::repeat_n(t, 31));
        let (err, abs) = check(None, &terms).unwrap();
        let contract = lp_eps(fa, fb, 32, false) * abs + F32_RZ_SUB;
        assert!(
            err <= contract && err > 0.9 * contract,
            "{err:e} vs {contract:e}"
        );
    }

    /// The FP4 kinds' contract covers `DEFAULT_FP4_MMA` with nonzero ue8m0 / ue4m3 scales
    /// (spread scales, live units with cancelling products, lossy accumulators); a zero ue4m3
    /// scale on a live unit has no relative bound.
    #[test]
    fn fp4_contract_covers_the_model() {
        let mut r = Rng(0x2545_f491_4f6c_dd1d);
        for st in [Lp::Ue8m0, Lp::Ue4m3] {
            let mut worst: f64 = 0.0;
            for it in 0..20000 {
                let prods: Vec<f64> = (0..64)
                    .map(|j| {
                        let a = lowp::value(code(&mut r, Lp::E2m1, 0), Lp::E2m1);
                        let b = lowp::value(code(&mut r, Lp::E2m1, 0), Lp::E2m1);
                        if it % 3 == 0 && j % 16 >= 2 {
                            0.0
                        } else {
                            a * b
                        }
                    })
                    .collect();
                let sc = |r: &mut Rng| loop {
                    let c = match st {
                        Lp::Ue8m0 => 70 + r.below(110) as u32,
                        _ => 1 + r.below(0x7e) as u32,
                    };
                    let o = Operand::scale(c, st);
                    if o.v > 0.0 && o.v.is_finite() {
                        break o;
                    }
                };
                let sa: Vec<Operand> = (0..4).map(|_| sc(&mut r)).collect();
                let sb: Vec<Operand> = (0..4).map(|_| sc(&mut r)).collect();
                let mut abs = 0.0;
                let mut s = vec![];
                for (j, p) in prods.iter().enumerate() {
                    let t = p * sa[j / 16].v * sb[j / 16].v;
                    abs += t.abs();
                    s = exact::add(&s, &exact::from(t));
                }
                let big = abs.max(1e-300).log2() as i32;
                let acc = match r.below(3) {
                    0 => None,
                    1 => {
                        let k = r.below(20) as i32;
                        Some(acc_of(&mut r, big - 30 - k))
                    }
                    _ => {
                        let k = r.below(6) as i32;
                        Some(acc_of(&mut r, big + k - 3))
                    }
                };
                if let Some(a) = acc {
                    abs += (a as f64).abs();
                    s = exact::add(&s, &exact::from(a as f64));
                }
                if exact::approx(&s).abs() > f32::MAX as f64 * 0.5 || abs > f32::MAX as f64 {
                    continue;
                }
                let d = DEFAULT_FP4_MMA.step(acc.map(f32::to_bits), &prods, &sa, &sb, 16);
                let err =
                    exact::approx(&exact::sub(&exact::from(f32::from_bits(d) as f64), &s)).abs();
                let contract = fp4_eps(st, 4, acc.is_some()).unwrap() * abs + F32_RZ_SUB;
                assert!(
                    err <= contract,
                    "{st:?}: error {err:e} > contract {contract:e}"
                );
                worst = worst.max(err / contract);
            }
            eprintln!(
                "{st:?}: eps {:.4}u (acc), {:.4}u (no acc); largest error / contract {worst:.3}",
                fp4_eps(st, 4, true).unwrap() / 2f64.powi(-24),
                fp4_eps(st, 4, false).unwrap() / 2f64.powi(-24)
            );
            assert!(worst > 0.0);
        }
        // A zero ue4m3 scale on a live unit: it sets emax (fields 2^-6 * 2^8) while adding 0
        // to sum |terms|; an accumulator 1.75 * 2^-36 is truncated at Q = 2^-33 to 0.
        let mut prods = vec![0.0; 64];
        prods[..16].iter_mut().for_each(|p| *p = 1.0);
        let zero = Operand::scale(0, Lp::Ue4m3);
        let big = Operand::scale(0x7e, Lp::Ue4m3);
        let sa = [zero; 4];
        let sb = [big; 4];
        let acc = (1.75 * 2f64.powi(-36)) as f32;
        let d = DEFAULT_FP4_MMA.step(Some(acc.to_bits()), &prods, &sa, &sb, 16);
        assert_eq!(f32::from_bits(d), 0.0);
    }
}

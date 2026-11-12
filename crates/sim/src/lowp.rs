/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Narrow float formats: `cvt` / SASS `F2FP` to and from FP8 (e4m3, e5m2), FP6 (e2m3,
//! e3m2), FP4 (e2m1), ue8m0, and the stochastic-rounding f16 / bf16 packs; plus the decoders
//! the tensor-core model uses for its operands and scale factors.
//!
//! Every rule here was measured on GB300 (probes/cvt_lp, `ptxv cvt-sweep`): all 2^16 inputs of
//! the 16-bit sources, all 2^8 / 2^4 codes of the decoders, edge cases and 2^20 shaped random
//! f32 inputs (and random bits) for the f32 sources.
// ASSUME(N-CVT-LP): see ASSUMPTIONS.md

use ptxv_ir::Lp;
use ptxv_ir::LpRnd;

/// Field layout of a sign-exponent-mantissa format.
#[derive(Clone, Copy, Debug)]
pub struct Fmt {
    pub ebits: u32,
    pub mbits: u32,
    pub bias: i32,
    /// Magnitude code of the largest finite value.
    pub max: u32,
    /// Magnitude code of infinity, if the format has one.
    pub inf: Option<u32>,
    /// Code produced for NaN (positive, as GB300 writes it).
    pub nan: u32,
    /// Has a sign bit (ue8m0 / ue4m3 do not).
    pub signed: bool,
}

pub fn fmt(t: Lp) -> Fmt {
    let f = |ebits, mbits, bias, max, inf, nan| Fmt {
        ebits,
        mbits,
        bias,
        max,
        inf,
        nan,
        signed: true,
    };
    match t {
        // OCP FP8 e4m3 ("fn"): no infinity, S.1111.111 is NaN.
        Lp::E4m3 => f(4, 3, 7, 0x7e, None, 0x7f),
        Lp::E5m2 => f(5, 2, 15, 0x7b, Some(0x7c), 0x7f),
        // FP6 / FP4: no infinity and no NaN; GB300 writes NaN as the positive maximum.
        Lp::E2m3 => f(2, 3, 1, 0x1f, None, 0x1f),
        Lp::E3m2 => f(3, 2, 3, 0x1f, None, 0x1f),
        Lp::E2m1 => f(2, 1, 1, 0x7, None, 0x7),
        Lp::F16 => f(5, 10, 15, 0x7bff, Some(0x7c00), 0x7fff),
        Lp::Bf16 => f(8, 7, 127, 0x7f7f, Some(0x7f80), 0x7fff),
        // Unsigned e4m3 (the nvf4 scale-factor format): 0x7f is NaN.
        Lp::Ue4m3 => Fmt {
            signed: false,
            ..f(4, 3, 7, 0x7e, None, 0x7f)
        },
        Lp::Ue8m0 => Fmt {
            ebits: 8,
            mbits: 0,
            bias: 127,
            max: 0xfe,
            inf: None,
            nan: 0xff,
            signed: false,
        },
    }
}

/// Code width in bits.
pub fn width(t: Lp) -> u32 {
    let f = fmt(t);
    f.ebits + f.mbits + f.signed as u32
}

/// Container width of one lane in a packed register: 8 for the FP8 / FP6 / ue8m0 codes (FP6
/// sits in the low 6 bits of a byte), 4 for e2m1, 16 for f16 / bf16.
pub fn lane_bits(t: Lp) -> u32 {
    match t {
        Lp::E2m1 => 4,
        Lp::F16 | Lp::Bf16 => 16,
        _ => 8,
    }
}

/// Random bits per lane for `.rs` (the width of the stochastic addend): 13 for f16, 16
/// otherwise.
pub fn rs_width(t: Lp) -> u32 {
    if t == Lp::F16 { 13 } else { 16 }
}

/// The random addend of each lane of one SASS `F2FP...RS Rd, Ra, Rb, Rc` (or PTX x2 `.rs`
/// form): (for a, for b). GB300: f16 / bf16 lanes take c[16..] (a) and c[0..] (b), low
/// `rs_width` bits each; 8-bit lanes take b' = c[16..32) and a' = bit-reverse16(b'); e2m1
/// lanes take b' = c[24..32) << 8 | c[8..16) and a' = bit-reverse16(b').
pub fn rs_bits(t: Lp, c: u32) -> (u32, u32) {
    match lane_bits(t) {
        16 => {
            let m = (1 << rs_width(t)) - 1;
            ((c >> 16) & m, c & m)
        }
        8 => {
            let b = c >> 16;
            ((b as u16).reverse_bits() as u32, b)
        }
        _ => {
            let b = (c >> 24) << 8 | (c >> 8) & 0xff;
            ((b as u16).reverse_bits() as u32, b)
        }
    }
}

/// Encode (-1)^neg * mag * 2^e into `t` (mag may be 0). `rs`: the random addend for
/// `LpRnd::Rs` (its `rs_width(t)` bits sit just below the result's last bit: the result
/// rounds up iff dropped + addend carries into it).
fn encode(neg: bool, mag: u64, e: i32, t: Lp, rnd: LpRnd, rs: u32, sat: bool) -> u32 {
    let f = fmt(t);
    let sign = if f.signed && neg {
        1 << (f.ebits + f.mbits)
    } else {
        0
    };
    if mag == 0 {
        return sign;
    }
    let top = e + 63 - mag.leading_zeros() as i32;
    let emin = 1 - f.bias;
    let mut qe = top.max(emin) - f.mbits as i32;
    let mut k: u64 = if qe <= e {
        mag << (e - qe) as u32
    } else {
        let n = (qe - e) as u32;
        let (keep, d) = if n >= 64 {
            (0, mag as u128)
        } else {
            (mag >> n, (mag & ((1u64 << n) - 1)) as u128)
        };
        let up = match rnd {
            // mag < 2^64 lies far below half a unit (and below the random addend's reach).
            _ if n > 100 => rnd == LpRnd::Rp && d != 0 && (!f.signed || !neg),
            LpRnd::Rn => {
                let half = 1u128 << (n - 1);
                d > half || d == half && keep & 1 == 1
            }
            LpRnd::Rz => false,
            // ue8m0's .rp rounds the magnitude up (GB300: negative inputs as their absolute
            // value); a signed format rounds toward +inf.
            LpRnd::Rp => d != 0 && (!f.signed || !neg),
            LpRnd::Rs => {
                let w = rs_width(t);
                if n >= w {
                    (d + ((rs as u128) << (n - w))) >> n != 0
                } else {
                    ((d << (w - n)) + rs as u128) >> w != 0
                }
            }
        };
        keep + up as u64
    };
    if k == 0 {
        return sign;
    }
    let one = 1u64 << f.mbits;
    while k >= one << 1 {
        k >>= 1; // exact: only a carry out of an all-ones significand gets here
        qe += 1;
    }
    let code = if k < one {
        k as u32 // subnormal
    } else {
        let be = qe + f.mbits as i32 + f.bias;
        if be >= 1 << f.ebits {
            u32::MAX
        } else {
            (be as u32) << f.mbits | (k - one) as u32
        }
    };
    if code > f.max {
        return sign
            | match (sat, f.inf) {
                (false, Some(i)) => i,
                (false, None) => f.nan,
                (true, _) => f.max,
            };
    }
    sign | code
}

/// A finite f32 as (sign, significand, exponent); None for Inf / NaN.
fn split32(u: u32) -> Option<(bool, u64, i32)> {
    let ex = (u >> 23) & 0xff;
    let fr = (u & 0x7f_ffff) as u64;
    match ex {
        0xff => None,
        0 => Some((u >> 31 == 1, fr, -149)),
        _ => Some((u >> 31 == 1, fr | 1 << 23, ex as i32 - 150)),
    }
}

/// f32 -> one lane of `t`. `rs`: random addend (`LpRnd::Rs`). GB300: NaN -> the format's NaN
/// code (positive); +-Inf -> +-max with `sat`, else +-Inf (ue8m0: 0xff); `.relu` maps
/// negative inputs (and -0) to +0; ue8m0 converts |x| (`.rz`: floor of log2, `.rp`: ceiling;
/// below 2^-127 -> 0, overflow -> 0xfe with `sat`, else 0xff).
pub fn from_f32(u: u32, t: Lp, rnd: LpRnd, rs: u32, sat: bool, relu: bool) -> u32 {
    if t == Lp::Ue8m0 {
        return ue8m0(u, rnd, sat);
    }
    let f = fmt(t);
    let Some((neg, mag, e)) = split32(u) else {
        if u & 0x7f_ffff != 0 {
            return f.nan;
        }
        let neg = u >> 31 == 1;
        if relu && neg {
            return 0;
        }
        let sign = if f.signed && neg {
            1 << (f.ebits + f.mbits)
        } else {
            0
        };
        return sign
            | match (sat, f.inf) {
                (false, Some(i)) => i,
                (false, None) => f.nan,
                (true, _) => f.max,
            };
    };
    if relu && neg {
        return 0;
    }
    encode(neg, mag, e, t, rnd, rs, sat)
}

fn ue8m0(u: u32, rnd: LpRnd, sat: bool) -> u32 {
    let over = if sat { 0xfe } else { 0xff };
    let Some((_, mag, e)) = split32(u) else {
        return if u & 0x7f_ffff != 0 { 0xff } else { over };
    };
    if mag == 0 {
        return 0;
    }
    let mut x = e + 63 - mag.leading_zeros() as i32;
    if rnd == LpRnd::Rp && !mag.is_power_of_two() {
        x += 1;
    }
    match x + 127 {
        c if c < 0 => 0,
        c if c >= 255 => over,
        c => c as u32,
    }
}

/// SASS `QMUL4.E4M3x4.E2M1x4.E4M3x4 d, a.N3210 / .N7654, b.B<bsel>` (`ptxv_ir::Op::Qmul4`):
/// byte k of the result is e2m1 nibble k (`hi`: k + 4) of `a` times e4m3 byte `bsel` of `b`
/// (exact), rounded to e4m3 to nearest even; a result past 448 (rounding) and a NaN scale give
/// 0x7f, with no sign. GB300: probes/qmul4, every nibble x scale x position. ASSUME(N-QMUL4)
pub fn qmul4(a: u32, b: u32, hi: bool, bsel: u8) -> u32 {
    let s = value((b >> (8 * (bsel as u32 & 3))) & 0xff, Lp::E4m3);
    let mut r = 0;
    for k in 0..4u32 {
        let nib = (a >> (4 * (k + if hi { 4 } else { 0 }))) & 15;
        let p = value(nib, Lp::E2m1) * s;
        let c = if p.is_nan() {
            0x7f
        } else {
            from_f32((p as f32).to_bits(), Lp::E4m3, LpRnd::Rn, 0, false, false)
        };
        let c = if c & 0x7f == 0x7f { 0x7f } else { c };
        r |= c << (8 * k);
    }
    r
}

/// f16 bits -> f32 bits (exact; NaN stays NaN).
pub fn f16_to_f32(h: u16) -> u32 {
    let s = (h as u32 & 0x8000) << 16;
    let e = (h >> 10 & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    match e {
        0 if m == 0 => s,
        0 => {
            let v = m as f32 * 2f32.powi(-24);
            s | v.to_bits()
        }
        31 => s | 0x7f80_0000 | m << 13,
        _ => s | (e + 112) << 23 | m << 13,
    }
}

/// One 16-bit lane (f16 or bf16, per `src`) -> `t`.
pub fn from_16(h: u16, src: Lp, t: Lp, rnd: LpRnd, sat: bool, relu: bool) -> u32 {
    let u = if src == Lp::Bf16 {
        (h as u32) << 16
    } else {
        f16_to_f32(h)
    };
    from_f32(u, t, rnd, 0, sat, relu)
}

/// Exact value of a code of `t` (low `width(t)` bits of `c`; higher bits ignored, as GB300
/// does for FP6 in a byte). NaN as f64 NaN.
pub fn value(c: u32, t: Lp) -> f64 {
    let f = fmt(t);
    let nb = width(t);
    let c = c & ((1 << nb) - 1);
    let neg = f.signed && c >> (nb - 1) == 1;
    let mag = if f.signed {
        c & ((1 << (nb - 1)) - 1)
    } else {
        c
    };
    if t == Lp::Ue8m0 {
        return if mag == 0xff {
            f64::NAN
        } else {
            2f64.powi(mag as i32 - 127)
        };
    }
    if mag == f.nan && f.inf.is_some() || mag > f.max && f.inf.is_none() && f.nan != f.max {
        return f64::NAN;
    }
    if let Some(i) = f.inf {
        if mag == i {
            return if neg {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
        if mag > i {
            return f64::NAN;
        }
    }
    let e = (mag >> f.mbits) as i32;
    let m = (mag & ((1 << f.mbits) - 1)) as f64;
    let v = if e == 0 {
        m * 2f64.powi(1 - f.bias - f.mbits as i32)
    } else {
        (1.0 + m / (1u32 << f.mbits) as f64) * 2f64.powi(e - f.bias)
    };
    if neg { -v } else { v }
}

/// One code -> f16 / bf16 bits (exact: every value of the narrow formats is representable).
/// NaN -> 0x7fff; `.relu`: negative (and -0) -> +0. ue8m0 0 -> bf16 0x0040 (2^-127).
pub fn to_16(c: u32, from: Lp, to: Lp, relu: bool) -> u16 {
    let v = value(c, from);
    if v.is_nan() {
        return 0x7fff;
    }
    if relu && (v < 0.0 || v == 0.0 && v.is_sign_negative()) {
        return 0;
    }
    let u = (v as f32).to_bits(); // exact
    if to == Lp::Bf16 {
        (u >> 16) as u16
    } else {
        crate::fp::f32_to_f16_rn(u)
    }
}

/// `CvtPack`: (a, b) -> d, see `ptxv_ir::Op::CvtPack`.
#[allow(clippy::too_many_arguments)]
pub fn pack(
    a: u32,
    b: u32,
    c: u32,
    half: Option<Lp>,
    to: Lp,
    rnd: LpRnd,
    sat: bool,
    relu: bool,
) -> u32 {
    let lb = lane_bits(to);
    let (ra, rb) = if rnd == LpRnd::Rs {
        rs_bits(to, c)
    } else {
        (0, 0)
    };
    let (ca, cb) = match half {
        Some(src) => (
            from_16((a >> 16) as u16, src, to, rnd, sat, relu),
            from_16(a as u16, src, to, rnd, sat, relu),
        ),
        None => (
            from_f32(a, to, rnd, ra, sat, relu),
            from_f32(b, to, rnd, rb, sat, relu),
        ),
    };
    let lanes = ca << lb | cb;
    if lb == 16 {
        lanes // F2FP.{F16,BF16}.F32.PACK_AB.RS: no merge (c is only the random bits)
    } else {
        c.checked_shl(2 * lb).unwrap_or(0) | lanes
    }
}

/// `CvtRsX4`: {a, b, e, f} -> d, a in the most significant lane: what ptxas's two
/// `F2FP...PACK_AB_MERGE_C.RS` compute (the second takes the first's result as c).
pub fn pack_rs_x4(src: [u32; 4], rbits: u32, to: Lp, relu: bool) -> u32 {
    let lo = pack(src[0], src[1], rbits, None, to, LpRnd::Rs, true, relu);
    let d = pack(src[2], src[3], lo, None, to, LpRnd::Rs, true, relu);
    if lane_bits(to) == 4 { d & 0xffff } else { d }
}

/// `CvtUnpack`: two codes in the low 16 (e2m1: 8) bits of `a` -> f16x2 / bf16x2.
pub fn unpack(a: u32, from: Lp, to: Lp, relu: bool) -> u32 {
    let lb = lane_bits(from);
    let m = (1 << lb) - 1;
    let hi = to_16((a >> lb) & m, from, to, relu) as u32;
    let lo = to_16(a & m, from, to, relu) as u32;
    hi << 16 | lo
}

/// One lane of `cvt.rn[.relu][.satfinite].scaled::n2::ue8m0.bf16x2.s2f6x2` (SASS
/// `F2FP[.SATFINITE][.RELU].BF16.S2_6.UNPACK_B`): the s2f6 code `c` is a two's-complement
/// 8-bit fixed point with 6 fraction bits (int8(c) / 64), times the ue8m0 scale 2^(s - 127).
/// The product is exact in bf16 (8 significant bits at most; the smallest, 2^-133, is bf16's
/// smallest subnormal, kept). Scale 0xff (ue8m0 NaN) gives 0x7fff for every code and form;
/// code 0 gives +0; `.relu` maps negative lanes to +0; the one overflow, -128 at scale 254
/// (-2^128), gives -Inf, or -max (0xff7f) with `.satfinite`. GB300: probes/s2f6_cvt (the
/// `.satfinite` form, every (data, scale) pair of both lanes, 2^32 in all) and
/// probes/s2f6_cvt/forms (all four forms, every (code, scale) pair in every lane position); GB200:
/// probes/s2f6_cvt/forms (the same sweep). 0 mismatches each.
pub fn s2f6_to_bf16(c: u8, s: u8, relu: bool, sat: bool) -> u16 {
    if s == 0xff {
        return 0x7fff;
    }
    let i = c as i8 as i32;
    if relu && i < 0 {
        return 0;
    }
    encode(
        i < 0,
        i.unsigned_abs() as u64,
        s as i32 - 127 - 6,
        Lp::Bf16,
        LpRnd::Rn,
        0,
        sat,
    ) as u16
}

/// `cvt.rn[.relu][.satfinite].scaled::n2::ue8m0.bf16x2.s2f6x2 d, a, scale` / `F2FP
/// [.SATFINITE][.RELU].BF16.S2_6.UNPACK_B Rd, Ra, Rb.H0`: lane i (bits 16i..16i+16 of the
/// result) from byte i of the 16-bit code pair `a` and byte i of the 16-bit scale pair.
/// N-CVT-LP.
pub fn cvt_s2f6x2_scaled(a: u16, scale: u16, relu: bool, sat: bool) -> u32 {
    let lane =
        |i: u32| s2f6_to_bf16((a >> (8 * i)) as u8, (scale >> (8 * i)) as u8, relu, sat) as u32;
    lane(1) << 16 | lane(0)
}

/// The ops of the probes/cvt_lp kernel, in output order: the `// j: name` comments that
/// probes/cvt_lp/gen.py writes into kernel.ptx (`name` = the cvt modifiers).
pub fn probe_ops(ptx: &str) -> Vec<String> {
    ptx.lines()
        .filter_map(|l| {
            let (j, n) = l.trim().strip_prefix("// ")?.split_once(": ")?;
            j.parse::<usize>().ok()?;
            Some(n.to_string())
        })
        .collect()
}

/// The model's output of one probe op (`name` as in `probe_ops`) for thread inputs x, y, z,
/// w (f32 operands; x also as f16x2 / bf16x2 and as packed codes) and random bits r.
pub fn probe_eval(name: &str, x: u32, y: u32, z: u32, w: u32, r: u32) -> u32 {
    let m: Vec<&str> = name.split('.').collect();
    let has = |s: &str| m.contains(&s);
    let n = m.len();
    let (to, from) = (m[n - 2], m[n - 1]);
    let lp = |t: &str| match t.trim_end_matches(char::is_numeric).trim_end_matches('x') {
        "e4m3" => Lp::E4m3,
        "e5m2" => Lp::E5m2,
        "e2m3" => Lp::E2m3,
        "e3m2" => Lp::E3m2,
        "e2m1" => Lp::E2m1,
        "ue8m0" => Lp::Ue8m0,
        "f16" => Lp::F16,
        "bf16" => Lp::Bf16,
        _ => panic!("probe op {name}"),
    };
    let rnd = match m[0] {
        "rn" => LpRnd::Rn,
        "rz" => LpRnd::Rz,
        "rp" => LpRnd::Rp,
        "rs" => LpRnd::Rs,
        _ => panic!("probe op {name}"),
    };
    let (sat, relu) = (has("satfinite"), has("relu"));
    match from {
        "f32" if to.ends_with("x4") => pack_rs_x4([x, y, z, w], r, lp(to), relu),
        "f32" => {
            let c = if rnd == LpRnd::Rs { r } else { 0 };
            pack(x, y, c, None, lp(to), rnd, sat, relu)
        }
        "f16x2" | "bf16x2" => pack(x, 0, 0, Some(lp(from)), lp(to), rnd, sat, relu),
        // Decodes read the low 16 (e2m1: 8) bits of x.
        _ => unpack(x & 0xffff, lp(from), lp(to), relu),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn s2f6_spot_values() {
        // int8(c) / 64 * 2^(s - 127), GB300 (probes/s2f6_cvt, probes/s2f6_cvt/forms).
        let f = |c, s| s2f6_to_bf16(c, s, false, true);
        assert_eq!(f(0x40, 0x7f), 0x3f80); // 1.0
        assert_eq!(f(0x80, 0x7f), 0xc000); // -2.0
        assert_eq!(f(0xff, 0x7f), 0xbc80); // -1/64
        assert_eq!(f(0x01, 0x00), 0x0001); // 2^-133: smallest bf16 subnormal
        assert_eq!(f(0x80, 0x00), 0x8080); // -2^-126
        assert_eq!(f(0x7f, 0xfe), 0x7f7e);
        assert_eq!(f(0x80, 0xfe), 0xff7f); // -2^128: satfinite
        assert_eq!(f(0x00, 0xff), 0x7fff); // NaN scale
        assert_eq!(f(0x00, 0x00), 0x0000);
        assert_eq!(cvt_s2f6x2_scaled(0x8040, 0x807f, false, true), 0xc080_3f80);
        // Without .satfinite the overflow is -Inf; .relu: negatives +0, a NaN scale stays.
        assert_eq!(s2f6_to_bf16(0x80, 0xfe, false, false), 0xff80);
        assert_eq!(s2f6_to_bf16(0x80, 0xfe, true, false), 0);
        assert_eq!(s2f6_to_bf16(0xff, 0x7f, true, true), 0);
        assert_eq!(s2f6_to_bf16(0x80, 0xff, true, true), 0x7fff);
        assert_eq!(s2f6_to_bf16(0x40, 0x7f, true, false), 0x3f80);
    }

    #[test]
    fn spot_values() {
        let f = |x: f32, t| from_f32(x.to_bits(), t, LpRnd::Rn, 0, true, false);
        assert_eq!(f(448.0, Lp::E4m3), 0x7e);
        assert_eq!(f(1e9, Lp::E4m3), 0x7e);
        assert_eq!(f(-1e9, Lp::E4m3), 0xfe);
        assert_eq!(f(f32::NAN, Lp::E4m3), 0x7f);
        assert_eq!(f(1.0, Lp::E4m3), 0x38);
        assert_eq!(f(2f32.powi(-9), Lp::E4m3), 0x01);
        assert_eq!(f(57344.0, Lp::E5m2), 0x7b);
        assert_eq!(f(6.0, Lp::E2m1), 0x7);
        assert_eq!(f(0.25, Lp::E2m1), 0x0); // tie to even (0)
        assert_eq!(f(0.75, Lp::E2m1), 0x2); // tie to even (1.0)
        assert_eq!(value(0x38, Lp::E4m3), 1.0);
        assert!(value(0x7f, Lp::E4m3).is_nan());
        assert_eq!(value(0x7e, Lp::E4m3), 448.0);
        assert_eq!(value(0x1f, Lp::E2m3), 7.5);
        assert_eq!(value(0x1f, Lp::E3m2), 28.0);
        assert_eq!(value(0x7c, Lp::E5m2), f64::INFINITY);
        assert_eq!(value(0, Lp::Ue8m0), 2f64.powi(-127));
        assert_eq!(to_16(0, Lp::Ue8m0, Lp::Bf16, false), 0x0040);
        let e8 = |x: f32, r| from_f32(x.to_bits(), Lp::Ue8m0, r, 0, true, false);
        assert_eq!(e8(1.0, LpRnd::Rz), 127);
        assert_eq!(e8(1.5, LpRnd::Rz), 127);
        assert_eq!(e8(1.5, LpRnd::Rp), 128);
        assert_eq!(e8(-1.5, LpRnd::Rp), 128);
    }
}

/// Inputs of the probes/cvt_lp sweep (`ptxv cvt-sweep`, replayed by cargo test).
pub mod sweep {
    use super::*;

    pub struct Set {
        pub name: &'static str,
        /// x, y, z, w, r per thread.
        pub v: [Vec<u32>; 5],
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn u32(&mut self) -> u32 {
            (self.next() >> 32) as u32
        }
        /// f32 bits: 7/8 sign, exponent in [-30, 20], random mantissa (the narrow formats'
        /// range, their subnormals and overflow); 1/8 raw bits.
        fn shaped(&mut self) -> u32 {
            let r = self.next();
            if r & 7 == 0 {
                return self.u32();
            }
            let e = (97 + (r >> 3) % 51) as u32;
            ((r >> 32) as u32 & 0x8000_0000) | e << 23 | (r >> 8) as u32 & 0x7f_ffff
        }
    }

    /// f32 edge inputs: for every FP8 / FP6 / FP4 format, each representable magnitude (up to
    /// twice the largest) and each midpoint, +-2 f32 ulps; powers of two +-1 ulp (ue8m0); zeros,
    /// subnormals, the largest finite, Inf, NaNs. Both signs.
    pub fn edges() -> Vec<u32> {
        let mut e = std::collections::BTreeSet::new();
        for t in [Lp::E4m3, Lp::E5m2, Lp::E2m3, Lp::E3m2, Lp::E2m1] {
            let f = fmt(t);
            let lim = 2.2 * value(f.max, t);
            let mut q: Vec<f64> = (0..1u32 << f.mbits)
                .map(|k| k as f64 * 2f64.powi(1 - f.bias - f.mbits as i32))
                .collect();
            let mut x = 1 - f.bias;
            while q.last().copied().unwrap_or(0.0) < lim {
                for k in 0..1u32 << f.mbits {
                    q.push((1.0 + k as f64 / (1u32 << f.mbits) as f64) * 2f64.powi(x));
                }
                x += 1;
            }
            for w in q.windows(2) {
                for v in [w[0], (w[0] + w[1]) / 2.0] {
                    let u = (v as f32).to_bits() as i64;
                    for d in -2..=2 {
                        if (0..0x7f80_0000).contains(&(u + d)) {
                            e.insert((u + d) as u32);
                        }
                    }
                }
            }
        }
        for b in 0..256u32 {
            for d in -1i64..=1 {
                let u = ((b << 23) as i64) + d;
                if (0..0x8000_0000).contains(&u) {
                    e.insert(u as u32);
                }
            }
        }
        for u in [
            0,
            1,
            2,
            0x7f_ffff,
            0x80_0000,
            0x7f7f_ffff,
            0x7f80_0000,
            0x7f80_0001,
            0x7fc0_0000,
            0x7fff_ffff,
        ] {
            e.insert(u);
        }
        let pos: Vec<u32> = e.into_iter().collect();
        pos.iter()
            .copied()
            .chain(pos.iter().map(|u| u | 0x8000_0000))
            .collect()
    }

    /// "x16": 2^16 threads, x's low half = thread index and high half a permutation of it (every
    /// f16 / bf16 lane value, every 8-bit / 4-bit code pair); "f32": 2^20 threads, the edges
    /// (in four different orders for x, y, z, w) then shaped random f32s; r random throughout.
    pub fn sets() -> Vec<Set> {
        let mut rng = Rng(0x5eed_c0de);
        let n = 1usize << 16;
        let x: Vec<u32> = (0..n as u32)
            .map(|i| i | ((i.wrapping_mul(40503).wrapping_add(12345)) & 0xffff) << 16)
            .collect();
        let gen = |rng: &mut Rng, n| (0..n).map(|_| rng.shaped()).collect::<Vec<u32>>();
        let (y, z, w) = (gen(&mut rng, n), gen(&mut rng, n), gen(&mut rng, n));
        let r: Vec<u32> = (0..n).map(|_| rng.u32()).collect();
        let x16 = Set {
            name: "x16",
            v: [x, y, z, w, r],
        };
        let n = 1usize << 20;
        let e = edges();
        let mut cols: Vec<Vec<u32>> = Vec::new();
        for c in 0..4 {
            let mut v = gen(&mut rng, n);
            // Fisher-Yates with the probe RNG (identity order for x).
            let mut p: Vec<u32> = e.clone();
            if c > 0 {
                for i in (1..p.len()).rev() {
                    let j = (rng.next() % (i as u64 + 1)) as usize;
                    p.swap(i, j);
                }
            }
            v[..p.len()].copy_from_slice(&p);
            cols.push(v);
        }
        let r: Vec<u32> = (0..n).map(|_| rng.u32()).collect();
        let [x, y, z, w]: [Vec<u32>; 4] = cols.try_into().unwrap();
        vec![
            x16,
            Set {
                name: "f32",
                v: [x, y, z, w, r],
            },
        ]
    }

    /// FNV-1a over u32 words.
    pub fn hash(words: impl Iterator<Item = u32>) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for w in words {
            for b in w.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        h
    }

    /// The model's outputs of op `name` on every thread of `s`.
    pub fn model(s: &Set, name: &str) -> Vec<u32> {
        let [x, y, z, w, r] = &s.v;
        (0..x.len())
            .map(|i| probe_eval(name, x[i], y[i], z[i], w[i], r[i]))
            .collect()
    }
}

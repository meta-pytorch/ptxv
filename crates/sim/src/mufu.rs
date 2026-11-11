/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Bit-exact model of the SASS multi-function unit (`MUFU.EX2/RCP/RSQ/LG2/SIN/COS/SQRT`, their
//! 16-bit forms) and of the PTX `.approx` ops ptxas builds from it.
//!
//! Reconstructed from exhaustive sweeps on GB300 (all 2^32 f32 inputs per op, all 2^16 /
//! 2^32 inputs of the 16-bit forms; probes/mufu/README.md). Every reconstructed function
//! matches the hardware on every input. Each unit is a quadratic interpolator in the style of
//! Oberman & Siu (ARITH 2005): the reduced argument's top `m` bits select a segment, the low
//! `n = 23 - m` (24 - m for RSQ/SQRT) bits `k` enter
//!
//!   S = C0 * 2^a + R + C1 * k * 2^b + C2 * sq(k) * 2^c,   result = floor(S / 2^sh)
//!
//! where sq(k) is k^2 computed by a truncated squarer that drops every partial-product bit of
//! weight below 2^col (col = 2n - 15), and (C0, C1, C2) come from a 64- or 128-entry table
//! (`mufu_tables.rs`, generated from the fits in probes/mufu/fits). R is a constant rounding
//! offset. The tables were fitted to the sweep, then the whole op (argument reduction,
//! special values, exponent assembly) was checked against all 2^32 recorded outputs.
//!
//! `MUFU.TANH` follows the same form per input binade, on a truncated input (`tanh`).
//! `MUFU.RCP64H/RSQ64H` (f64 high words) reuse the f32 RCP/RSQ units (`rcp64h`, `rsq64h`).
//!
//! ASSUME(N-MUFU): see ASSUMPTIONS.md

use ptxv_ir::MufuFmt;
use ptxv_ir::MufuFunc;

use crate::mufu_tables as tab;

/// One interpolator: see the module comment. `tab[s] = (C0, C1, C2)`.
pub(crate) struct Interp {
    pub n: u32,
    pub col: u32,
    pub a: u32,
    pub r: i64,
    pub b: u32,
    pub c: i32,
    pub sh: u32,
    pub tab: &'static [(i64, i64, i64)],
}

/// k^2 with the partial-product bits of weight < 2^col dropped (bit i * bit j for i != j has
/// weight 2^(i+j+1), bit i squared 2^(2i)).
fn sq_trunc(k: i64, col: u32) -> i64 {
    let mut dropped = 0i64;
    for i in 0..col.div_ceil(2) {
        if k >> i & 1 == 0 {
            continue;
        }
        if 2 * i < col {
            dropped += 1 << (2 * i);
        }
        // j in (i, col - 2 - i]: weight i + j + 1 < col.
        if col >= 2 * i + 2 {
            let hi = col - 2 - i; // highest j kept in the dropped set
            if hi > i {
                let mask = ((1i64 << (hi + 1)) - 1) & !((1i64 << (i + 1)) - 1);
                dropped += (k & mask) << (i + 1);
            }
        }
    }
    k * k - dropped
}

impl Interp {
    fn eval(&self, idx: u32) -> i64 {
        let s = (idx >> self.n) as usize;
        let k = (idx & ((1 << self.n) - 1)) as i64;
        let (c0, c1, c2) = self.tab[s];
        let t2 = c2 * sq_trunc(k, self.col);
        let t2 = if self.c >= 0 {
            t2 << self.c
        } else {
            t2 >> -self.c
        };
        ((c0 << self.a) + self.r + ((c1 * k) << self.b) + t2) >> self.sh
    }
}

pub const CANON_NAN: u32 = crate::fp::F32_CANON_NAN;

fn is_nan(x: u32) -> bool {
    x & 0x7fff_ffff > 0x7f80_0000
}

/// Positive `mag * 2^-frac` (mag > 0, a normal f32 range) to f32 bits, truncated to 24
/// significant bits.
fn fixed_to_f32_rz(mag: u64, frac: i32) -> u32 {
    let lead = 63 - mag.leading_zeros() as i32;
    let m = if lead >= 23 {
        mag >> (lead - 23)
    } else {
        mag << (23 - lead)
    } as u32;
    let e = lead - frac + 127;
    debug_assert!((1..255).contains(&e));
    ((e as u32) << 23) | (m & 0x7f_ffff)
}

/// `MUFU.EX2` (f32): 2^x. Denormal inputs count as 0, results below 2^-126 flush to +0.
pub fn ex2(x: u32) -> u32 {
    if is_nan(x) {
        return CANON_NAN;
    }
    let neg = x >> 31 == 1;
    let e = (x >> 23 & 0xff) as i32;
    if e >= 127 + 8 {
        // |x| >= 256 (and infinities)
        return if neg { 0 } else { 0x7f80_0000 };
    }
    // q = trunc(|x| * 2^23), a fixed-point argument with 23 fraction bits.
    let m = (x & 0x7f_ffff | 0x80_0000) as i64;
    let mut q = if e == 0 {
        0
    } else if e >= 127 {
        m << (e - 127)
    } else if 127 - e < 24 {
        m >> (127 - e)
    } else {
        0
    };
    if neg {
        // A negative argument is the one's complement of q, except when q's fraction is 0.
        q = -q - (q & 0x7f_ffff != 0) as i64;
    }
    let mant = tab::EX2.eval((q & 0x7f_ffff) as u32);
    debug_assert!((1 << 23..1 << 24).contains(&mant));
    let be = (q >> 23) + 127;
    if be <= 0 {
        0
    } else if be >= 255 {
        0x7f80_0000
    } else {
        ((be as u32) << 23) | (mant as u32 & 0x7f_ffff)
    }
}

/// `MUFU.RCP` (f32): 1/x. Denormal inputs count as 0 (giving +-inf), results below 2^-126
/// flush to +-0.
pub fn rcp(x: u32) -> u32 {
    if is_nan(x) {
        return CANON_NAN;
    }
    let s = x & 0x8000_0000;
    let e = (x >> 23 & 0xff) as i32;
    match e {
        0 => return s | 0x7f80_0000,
        255 => return s,
        _ => {}
    }
    let mant = tab::RCP.eval(x & 0x7f_ffff); // 1/m in units 2^-24, (2^23, 2^24]
    let (be, frac) = if mant == 1 << 24 {
        (254 - e, 0)
    } else {
        (253 - e, mant as u32 & 0x7f_ffff)
    };
    if be <= 0 {
        s
    } else {
        s | (be as u32) << 23 | frac
    }
}

/// Index of the RSQ / SQRT tables: the exponent's parity, then the mantissa; and the exponent
/// halved (x = m * 2^p * 4^t).
fn root_index(e: i32, x: u32) -> (u32, i32) {
    let p = (e - 127) & 1;
    ((p as u32) << 23 | (x & 0x7f_ffff), (e - 127 - p) / 2)
}

/// `MUFU.RSQ` (f32): 1/sqrt(x). Denormals count as 0 (+-inf); negative inputs give NaN.
pub fn rsq(x: u32) -> u32 {
    if is_nan(x) {
        return CANON_NAN;
    }
    let s = x & 0x8000_0000;
    let e = (x >> 23 & 0xff) as i32;
    if e == 0 {
        return s | 0x7f80_0000;
    }
    if s != 0 {
        return CANON_NAN;
    }
    if e == 255 {
        return 0;
    }
    let (idx, t) = root_index(e, x);
    // The interpolator yields 1 - 2^-24 at x = 4^t; the unit returns the exact power of two.
    if idx == 0 {
        return ((127 - t) as u32) << 23;
    }
    let mant = tab::RSQ.eval(idx);
    if mant == 1 << 24 {
        ((127 - t) as u32) << 23
    } else {
        ((126 - t) as u32) << 23 | (mant as u32 & 0x7f_ffff)
    }
}

/// `MUFU.SQRT` (f32). Denormals count as 0 (keeping the sign); negative inputs give NaN.
pub fn sqrt(x: u32) -> u32 {
    if is_nan(x) {
        return CANON_NAN;
    }
    let s = x & 0x8000_0000;
    let e = (x >> 23 & 0xff) as i32;
    if e == 0 {
        return s;
    }
    if s != 0 {
        return CANON_NAN;
    }
    if e == 255 {
        return 0x7f80_0000;
    }
    let (idx, t) = root_index(e, x);
    let mant = tab::SQRT.eval(idx); // [2^23, 2^24), units 2^-23
    debug_assert!(mant < 1 << 24);
    ((127 + t) as u32) << 23 | (mant as u32 & 0x7f_ffff)
}

/// `MUFU.LG2` (f32): log2(x). The result is e + L(m) in fixed point (36 fraction bits),
/// truncated to 24 significant bits; a negative sum is negated in one's complement.
/// Denormals and zeros give -inf, negative inputs NaN, log2(1) = +0.
pub fn lg2(x: u32) -> u32 {
    if is_nan(x) {
        return CANON_NAN;
    }
    let e = (x >> 23 & 0xff) as i32;
    if e == 0 {
        return 0xff80_0000;
    }
    if x >> 31 == 1 {
        return CANON_NAN;
    }
    if e == 255 {
        return 0x7f80_0000;
    }
    if x == 0x3f80_0000 {
        return 0;
    }
    let l = tab::LG2.eval(x & 0x7f_ffff); // [0, 2^36)
    let ex = (e - 127) as i64;
    let v = (ex << 36) + l;
    if v > 0 {
        fixed_to_f32_rz(v as u64, 36)
    } else {
        let mag = ((-ex - 1) << 36) + ((1 << 36) - 1 - l);
        if mag == 0 {
            0x8000_0000
        } else {
            0x8000_0000 | fixed_to_f32_rz(mag as u64, 36)
        }
    }
}

/// `MUFU.SIN` / `MUFU.COS` (f32) of an angle in revolutions: sin(2 pi x) / cos(2 pi x). The
/// argument is truncated to 25 fraction bits (a quarter turn is 2^23 steps); quadrants 1 and 3
/// mirror the table in one's complement. Infinities give NaN.
pub fn sincos(x: u32, cos: bool) -> u32 {
    let e = (x >> 23 & 0xff) as i32;
    if e == 255 {
        return CANON_NAN;
    }
    let m = (x & 0x7f_ffff | 0x80_0000) as u64;
    let sh = e - 125; // q = |x| * 2^25 = m * 2^(e - 125)
    let mut q = if e == 0 || sh >= 25 {
        0
    } else if sh >= 0 {
        (m << sh) as u32
    } else if -sh < 32 {
        (m >> -sh) as u32
    } else {
        0
    } & ((1 << 25) - 1);
    let neg = if cos {
        q = (q + (1 << 23)) & ((1 << 25) - 1);
        q >> 24 == 1
    } else {
        (x >> 31 == 1) ^ (q >> 24 == 1)
    };
    let u = q & 0x7f_ffff;
    let u = if q >> 23 & 1 == 1 { 0x7f_ffff - u } else { u };
    let l = tab::SIN.eval(u); // units 2^-37
    let sign = (neg as u32) << 31;
    if l == 0 {
        sign
    } else {
        sign | fixed_to_f32_rz(l as u64, 37)
    }
}

/// `MUFU.TANH` (f32): |x| < 2^-8 returns x (denormals and zeros included), |x| >= 8 (and
/// infinities) returns +-1, NaN the canonical NaN. For |x| in binade [2^e, 2^(e+1)),
/// e = -8..=2, the unit reads only the top K[e] (17..21) mantissa bits; their top K[e] - 17
/// bits pick one of 51 segments, the low 17 bits k enter
///
///   S = C0 + C1 * k * 2^17 + C2 * sq19(k)   (units 2^(e-66), C0 up to 2^67),
///
/// and the result is S truncated to 24 significant bits, sign-mirrored.
pub fn tanh(x: u32) -> u32 {
    let e = x >> 23 & 0xff;
    let sign = x & 0x8000_0000;
    if is_nan(x) {
        return CANON_NAN;
    } else if e < 127 - 8 {
        return x;
    } else if e >= 127 + 3 {
        return sign | 0x3f80_0000;
    }
    let i = (e - (127 - 8)) as usize;
    let kb = tab::TANH_K[i];
    let q = (x & 0x7f_ffff) >> (23 - kb);
    let (c0, c1, c2) = tab::TANH[i][(q >> 17) as usize];
    let k = (q & 0x1_ffff) as i64;
    let s = c0 + ((c1 * k) as i128) * (1 << 17) + (c2 * sq_trunc(k, 19)) as i128;
    debug_assert!(s > 0);
    let s = s as u128;
    let lead = 127 - s.leading_zeros() as i32;
    let m = (s >> (lead - 23)) as u32;
    let be = lead + e as i32 - 127 - 66 + 127;
    sign | (be as u32) << 23 | m & 0x7f_ffff
}

/// `MUFU.RCP64H` on the high word `x` of an f64 (the low word is not an input): the high
/// word of 1/x, low word 0. The mantissa is the f32 unit's: `rcp` of 1.m (the 20 mantissa
/// bits, three zero bits below), truncated to 20 bits; the exponent is rebuilt for the f64
/// range. Zero / denormal -> +-inf, results below 2^-1022 -> +-0, inf -> +-0, NaN ->
/// 0x7fffffff.
pub fn rcp64h(x: u32) -> u32 {
    let (s, e, m) = (x & 0x8000_0000, (x >> 20 & 0x7ff) as i32, x & 0xf_ffff);
    match e {
        0 => s | 0x7ff0_0000,
        0x7ff if m == 0 => s,
        0x7ff => 0x7fff_ffff,
        _ => {
            let r = rcp(0x3f80_0000 | m << 3);
            s | f64_hi(1023 + (r >> 23) as i32 - 127 - (e - 1023), r)
        }
    }
}

/// `MUFU.RSQ64H`, likewise from the f32 `rsq` of 1.m * 2^p (p = exponent parity):
/// +-0 / denormal -> +-inf, negative -> NaN (0x7fffffff), +inf -> +0.
pub fn rsq64h(x: u32) -> u32 {
    let (s, e, m) = (x & 0x8000_0000, (x >> 20 & 0x7ff) as i32, x & 0xf_ffff);
    match e {
        0 => s | 0x7ff0_0000,
        _ if s != 0 => 0x7fff_ffff,
        0x7ff if m == 0 => 0,
        0x7ff => 0x7fff_ffff,
        _ => {
            let p = (e - 1023) & 1;
            let r = rsq(((127 + p) as u32) << 23 | m << 3);
            f64_hi(1023 + (r >> 23) as i32 - 127 - ((e - 1023 - p) >> 1), r)
        }
    }
}

/// f64 high word with biased exponent `be` and the top 20 of f32 `r`'s mantissa bits; a
/// result below the normal range flushes to 0.
fn f64_hi(be: i32, r: u32) -> u32 {
    if be <= 0 {
        0
    } else {
        (be as u32) << 20 | (r & 0x7f_ffff) >> 3
    }
}

/// The f32 unit, `None` if not modeled.
pub fn unit_f32(func: MufuFunc, x: u32) -> Option<u32> {
    Some(match func {
        MufuFunc::Ex2 => ex2(x),
        MufuFunc::Rcp => rcp(x),
        MufuFunc::Rsq => rsq(x),
        MufuFunc::Lg2 => lg2(x),
        MufuFunc::Sin => sincos(x, false),
        MufuFunc::Cos => sincos(x, true),
        MufuFunc::Sqrt => sqrt(x),
        MufuFunc::Tanh => tanh(x),
        MufuFunc::Rcp64h => rcp64h(x),
        MufuFunc::Rsq64h => rsq64h(x),
    })
}

/// f32 -> f16, truncating (toward zero); |v| >= 2^16 overflows to inf; NaN -> 0x7fff.
pub fn f32_to_f16_rz(x: u32) -> u16 {
    if is_nan(x) {
        return 0x7fff;
    }
    let s = (x >> 16 & 0x8000) as u16;
    let e = (x >> 23 & 0xff) as i32;
    let m = x & 0x7f_ffff | if e == 0 { 0 } else { 0x80_0000 };
    let ue = e - 127; // value = m * 2^(ue - 23)
    if e == 255 || ue >= 16 {
        return s | 0x7c00;
    }
    if ue >= -14 {
        return s | (((ue + 15) as u16) << 10) | ((m >> 13) as u16 & 0x3ff);
    }
    // f16 subnormal: units 2^-24 = m * 2^(ue - 23 + 24)
    let sh = 23 - 24 - ue; // right shift
    s | if sh >= 32 { 0 } else { (m >> sh) as u16 }
}

fn f16_to_f32(h: u16) -> u32 {
    let v = crate::fp::f16_to_f64(h);
    if v.is_nan() {
        CANON_NAN
    } else {
        (v as f32).to_bits()
    }
}

/// One 16-bit lane: f32 unit, then truncation to the format.
fn lane16(func: MufuFunc, bf16: bool, h: u16) -> Option<u16> {
    let x = if bf16 {
        (h as u32) << 16
    } else {
        f16_to_f32(h)
    };
    let y = unit_f32(func, x)?;
    Some(if bf16 {
        if is_nan(y) { 0x7fff } else { (y >> 16) as u16 }
    } else {
        f32_to_f16_rz(y)
    })
}

/// SASS `MUFU.<func>` in format `fmt` on register value `a` (`hi`: scalar 16-bit operand is
/// the high half). `None` if not modeled.
pub fn mufu(func: MufuFunc, fmt: MufuFmt, hi: bool, a: u32) -> Option<u32> {
    let h = if hi { (a >> 16) as u16 } else { a as u16 };
    Some(match fmt {
        MufuFmt::F32 | MufuFmt::F64 => unit_f32(func, a)?,
        MufuFmt::F16 => lane16(func, false, h)? as u32,
        MufuFmt::Bf16 => lane16(func, true, h)? as u32,
        MufuFmt::F16x2 | MufuFmt::Bf16x2 => {
            let bf = fmt == MufuFmt::Bf16x2;
            lane16(func, bf, a as u16)? as u32 | (lane16(func, bf, (a >> 16) as u16)? as u32) << 16
        }
    })
}

/// f32 multiply, IEEE, denormals kept, canonical NaN (SASS `FMUL` / `FMUL.RZ`).
fn fmul(a: u32, b: u32, rz: bool) -> u32 {
    let p = f32::from_bits(a) as f64 * f32::from_bits(b) as f64; // exact
    if p.is_nan() {
        return CANON_NAN;
    }
    let rnd = if rz {
        ptxv_ir::Rnd::Rz
    } else {
        ptxv_ir::Rnd::Rn
    };
    crate::fp::f64_to_f32(p, rnd).to_bits()
}

const TWO24: u32 = 0x4b80_0000;
const MIN_NORMAL: f32 = f32::MIN_POSITIVE;

/// PTX `<func>.approx[.ftz]` in format `fmt`, as ptxas (CUDA 13.0, sm_103a) expands it:
/// - `.ftz` f32 forms, `tanh` and the 16-bit forms are the bare unit;
/// - `ex2.approx.f32`: x < -126 -> MUFU.EX2(x * 0.5)^2 (the square keeps denormal results);
/// - `rcp.approx.f32`: |x| < 2^-126 -> scale x and the result by 2^24, |x| > 2^126 -> by 0.25;
/// - `rsqrt` / `sqrt` / `lg2.approx.f32`: |x| < 2^-126 -> MUFU(x * 2^24), then * 2^12 /
///   * 2^-12 / - 24;
/// - `sin` / `cos.approx[.ftz].f32`: MUFU.SIN/COS(x * 1/(2 pi) rounded toward zero).
pub fn ptx_approx(func: MufuFunc, fmt: MufuFmt, ftz: bool, x: u32) -> Option<u32> {
    if fmt != MufuFmt::F32 {
        return mufu(func, fmt, false, x);
    }
    let xf = f32::from_bits(x);
    let tiny = xf.abs() < MIN_NORMAL; // FSETP.GEU |x|, 2^-126 false (NaN: not tiny)
    Some(match func {
        MufuFunc::Sin | MufuFunc::Cos => sincos(fmul(x, 0x3e22_f983, true), func == MufuFunc::Cos),
        MufuFunc::Tanh | MufuFunc::Rcp64h | MufuFunc::Rsq64h => unit_f32(func, x)?,
        _ if ftz => unit_f32(func, x)?,
        MufuFunc::Ex2 => {
            if xf < -126.0 {
                let y = ex2(fmul(x, 0x3f00_0000, false));
                fmul(y, y, false)
            } else {
                ex2(x)
            }
        }
        MufuFunc::Rcp => {
            let s = if xf.abs() > 2f32.powi(126) {
                0x3e80_0000 // 0.25
            } else if tiny {
                TWO24
            } else {
                0x3f80_0000
            };
            fmul(s, rcp(fmul(x, s, false)), false)
        }
        MufuFunc::Rsq if tiny => fmul(rsq(fmul(x, TWO24, false)), 0x4580_0000, false),
        MufuFunc::Sqrt if tiny => fmul(sqrt(fmul(x, TWO24, false)), 0x3980_0000, false),
        MufuFunc::Lg2 if tiny => {
            crate::fp::f32_add_rn(lg2(fmul(x, TWO24, false)), (-24f32).to_bits())
        }
        MufuFunc::Rsq => rsq(x),
        MufuFunc::Sqrt => sqrt(x),
        MufuFunc::Lg2 => lg2(x),
    })
}

/// Error contract of a MUFU / approx op against the true function f, from the whole GB300
/// sweep (probes/mufu/README.md): |hw - f(x)| <= rel_ulps * ulp(f(x)) + abs on `domain`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrBound {
    pub rel_ulps: f64,
    pub abs: f64,
    pub domain: &'static str,
}

const U24: f64 = 1.0 / (1u64 << 24) as f64;

/// The bound for input `x` (PTX `sin`/`cos` error grows with |x|: their range reduction is one
/// f32 multiply by 1/(2 pi)). Each constant is the sweep maximum rounded up.
// ASSUME(N-MUFU): see ASSUMPTIONS.md
pub fn error_bound(func: MufuFunc, ptx: bool, x: f32) -> ErrBound {
    let b = |rel_ulps, abs, domain| ErrBound {
        rel_ulps,
        abs,
        domain,
    };
    let flush = f32::MIN_POSITIVE as f64; // ftz results below 2^-126 become 0
    match func {
        MufuFunc::Ex2 => b(2.056, flush, "all inputs"),
        MufuFunc::Rcp => b(0.975, flush, "all inputs"),
        MufuFunc::Rsq => b(1.515, 0.0, "positive inputs (.ftz: normal)"),
        // .ftz: a denormal input gives 0 (|sqrt(x)| < 2^-63).
        MufuFunc::Sqrt => b(
            0.944,
            if ptx { 0.0 } else { 2f64.powi(-63) },
            "positive inputs",
        ),
        // The fixed-point sum e + log2(m) truncated to 24 bits.
        MufuFunc::Lg2 => b(1.0, 2.62 * U24, "positive inputs (.ftz: normal)"),
        MufuFunc::Sin | MufuFunc::Cos if ptx => {
            if x.abs() <= 1024.0 {
                b(0.0, (8.0 + 2.7 * x.abs() as f64) * U24, "|x| <= 1024")
            } else {
                b(0.0, 2.0, "|x| > 1024 (no useful bound)")
            }
        }
        MufuFunc::Sin | MufuFunc::Cos => b(1.0, 5.46 * U24, "finite inputs (revolutions)"),
        MufuFunc::Tanh => b(134.0, 0.0, "all inputs"),
        // In f32 ulps of f(x), x the whole f64 (the unit sees only 20 mantissa bits, and
        // truncates the result to 20).
        MufuFunc::Rcp64h => b(16.6, 0.0, "normal f64 inputs and results, any low word"),
        MufuFunc::Rsq64h => b(9.1, 0.0, "positive normal f64 inputs, any low word"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sq_trunc_matches_definition() {
        for (n, col) in [(16u32, 17u32), (17, 19)] {
            for k in (0..1i64 << n).step_by(97).chain([(1 << n) - 1]) {
                let mut want = 0i64;
                for i in 0..n {
                    for j in i..n {
                        let w = i + j + (i != j) as u32;
                        if w >= col && k >> i & 1 == 1 && k >> j & 1 == 1 {
                            want += 1 << w;
                        }
                    }
                }
                assert_eq!(sq_trunc(k, col), want, "k={k} col={col}");
            }
        }
    }

    #[test]
    fn spot_values() {
        // Recorded GB300 outputs.
        assert_eq!(ex2(0x3f80_0000), 0x4000_0000);
        assert_eq!(ex2(0xbf80_0000), 0x3f00_0000);
        assert_eq!(ex2(0xb400_0000), 0x3f7f_fffe);
        assert_eq!(rcp(0x3f80_0001), 0x3f7f_fffe);
        assert_eq!(rcp(0xbfc0_0000), 0xbf2a_aaab);
        assert_eq!(rcp(0x7e80_0000), 0x0080_0000);
        assert_eq!(rsq(0x4000_0000), 0x3f35_04f2);
        assert_eq!(rsq(0x0080_0000), 0x5f00_0000);
        assert_eq!(lg2(0x3f80_0001), 0x3495_f600);
        assert_eq!(lg2(0x3f00_0000), 0xbf7f_fffe);
        assert_eq!(lg2(0x0080_0000), 0xc2fb_ffff);
        assert_eq!(sincos(0x3e00_0000, false), 0x3f35_04f2);
        assert_eq!(sincos(0x3f00_0000, false), 0x8000_0000);
        assert_eq!(sincos(0x3300_0000, false), 0x3449_1200);
        assert_eq!(sqrt(0x4000_0000), 0x3fb5_04f3);
        assert_eq!(sqrt(0x7f7f_ffff), 0x5f7f_ffff);
    }
}

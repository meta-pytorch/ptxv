/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Bit-exact floating-point operations used by the interpreter.

/// Canonical NaN produced by NVIDIA fp32 arithmetic.
pub const F32_CANON_NAN: u32 = 0x7fff_ffff;

// ASSUME(N-FADD): see ASSUMPTIONS.md
pub fn f32_add_rn(a: u32, b: u32) -> u32 {
    let r = f32::from_bits(a) + f32::from_bits(b);
    if r.is_nan() {
        F32_CANON_NAN
    } else {
        r.to_bits()
    }
}

/// IEEE f32 addition rounded toward zero.
pub fn f32_add_rz(a: u32, b: u32) -> u32 {
    let (x, y) = (f32::from_bits(a) as f64, f32::from_bits(b) as f64);
    let s = x + y;
    if s.is_nan() {
        return F32_CANON_NAN;
    }
    if s.is_infinite() {
        return f64_to_f32(s, ptxv_ir::Rnd::Rz).to_bits();
    }
    // TwoSum: x + y = s + e exactly.
    let bb = s - x;
    let e = (x - (s - bb)) + (y - bb);
    let mut r = f64_to_f32(s, ptxv_ir::Rnd::Rz);
    // s itself was rounded (to nearest, in f64): if it landed on an f32 value above the true
    // magnitude, step one f32 toward zero.
    if r as f64 == s && e != 0.0 && (e < 0.0) == (s > 0.0) {
        r = f32::from_bits(r.to_bits() - 1);
    }
    if r == 0.0 && s == 0.0 {
        // Exact zero sum: +0 except (-0) + (-0).
        return if a == 0x8000_0000 && b == 0x8000_0000 {
            0x8000_0000
        } else {
            0
        };
    }
    r.to_bits()
}

// ------------------------------------------------- f32 ALU (FADD / FMUL / FFMA / FMNMX ...)
// ASSUME(N-FALU): see ASSUMPTIONS.md

pub fn f32_is_nan(u: u32) -> bool {
    u & 0x7fff_ffff > 0x7f80_0000
}

fn f32_is_inf(u: u32) -> bool {
    u & 0x7fff_ffff == 0x7f80_0000
}

fn f32_is_sub(u: u32) -> bool {
    u & 0x7f80_0000 == 0 && u & 0x7f_ffff != 0
}

/// `.FTZ`: a subnormal becomes zero of the same sign.
pub fn f32_ftz(u: u32) -> u32 {
    if f32_is_sub(u) { u & 0x8000_0000 } else { u }
}

/// A finite f32 as (sign, significand, exponent): value = (-1)^sign * m * 2^e.
fn f32_split(u: u32) -> (bool, u64, i32) {
    let ex = ((u >> 23) & 0xff) as i32;
    let fr = (u & 0x7f_ffff) as u64;
    if ex == 0 {
        (u >> 31 == 1, fr, -149)
    } else {
        (u >> 31 == 1, fr | 1 << 23, ex - 150)
    }
}

/// Round (-1)^neg * mag * 2^e (mag > 0) to f32 in mode `rnd`: 24-bit significand, gradual
/// underflow, overflow to infinity or the largest finite value as IEEE 754 prescribes.
pub fn round_f32(neg: bool, mag: u128, e: i32, rnd: ptxv_ir::Rnd) -> u32 {
    round_f32_ftz(neg, mag, e, rnd, false)
}

/// `round_f32`; with `ftz` the value is rounded to 24 significant bits as if the exponent were
/// unbounded below, and a result below 2^-126 in magnitude becomes zero of its sign (GB300,
/// s_alu2: under FMUL.FTZ, (1 - 2^-24) * 2^-126 is 0, while (1 - 2^-46) * 2^-126 rounds up
/// to 2^-126).
pub fn round_f32_ftz(neg: bool, mag: u128, e: i32, rnd: ptxv_ir::Rnd, ftz: bool) -> u32 {
    use ptxv_ir::Rnd;
    let sign = (neg as u32) << 31;
    debug_assert!(mag > 0);
    let top = e + 127 - mag.leading_zeros() as i32; // exponent of the leading bit
    // Exponent of the result's last bit.
    let mut q = if ftz { top - 23 } else { (top - 23).max(-149) };
    let sh = q - e;
    let (mut keep, guard, rest) = if sh <= 0 {
        (mag << (-sh) as u32, false, false)
    } else if sh > 128 {
        (0, false, true)
    } else {
        let sh = sh as u32;
        let keep = if sh == 128 { 0 } else { mag >> sh };
        let guard = mag >> (sh - 1) & 1 == 1;
        let rest = mag & ((1u128 << (sh - 1)) - 1) != 0;
        (keep, guard, rest)
    };
    let inc = match rnd {
        Rnd::Rn => guard && (rest || keep & 1 == 1),
        Rnd::Rz => false,
        Rnd::Rp => !neg && (guard || rest),
        Rnd::Rm => neg && (guard || rest),
    };
    keep += inc as u128;
    if keep >> 24 != 0 {
        keep >>= 1;
        q += 1;
    }
    if ftz && q + 23 < -126 {
        return sign;
    }
    if keep < 1 << 23 {
        return sign | keep as u32; // subnormal (q = -149) or zero
    }
    let be = q + 150;
    if be >= 255 {
        let inf = match rnd {
            Rnd::Rn => true,
            Rnd::Rz => false,
            Rnd::Rp => !neg,
            Rnd::Rm => neg,
        };
        return sign | if inf { 0x7f80_0000 } else { 0x7f7f_ffff };
    }
    sign | (be as u32) << 23 | (keep as u32 & 0x7f_ffff)
}

/// f32 fused multiply-add `a*b + c` with one rounding in mode `rnd` (FFMA; FMUL when `c` is
/// None: the rounded product; FADD is a*1 + c). Computed exactly in integers. NaN results are
/// the canonical NaN. `ftz`: subnormal inputs and results are flushed to signed zero.
pub fn f32_fma(a: u32, b: u32, c: Option<u32>, rnd: ptxv_ir::Rnd, ftz: bool) -> u32 {
    let fl = |u: u32| if ftz { f32_ftz(u) } else { u };
    let (a, b, c) = (fl(a), fl(b), c.map(fl));
    if f32_is_nan(a) || f32_is_nan(b) || c.is_some_and(f32_is_nan) {
        return F32_CANON_NAN;
    }
    let ps = (a ^ b) >> 31 == 1;
    let pzero = a & 0x7fff_ffff == 0 || b & 0x7fff_ffff == 0;
    if f32_is_inf(a) || f32_is_inf(b) {
        if pzero {
            return F32_CANON_NAN; // inf * 0
        }
        if let Some(c) = c.filter(|&c| f32_is_inf(c) && (c >> 31 == 1) != ps) {
            let _ = c;
            return F32_CANON_NAN; // inf - inf
        }
        return (ps as u32) << 31 | 0x7f80_0000;
    }
    if let Some(c) = c.filter(|&c| f32_is_inf(c)) {
        return c;
    }
    let (_, ma, ea) = f32_split(a);
    let (_, mb, eb) = f32_split(b);
    let (pm, pe) = ((ma * mb) as u128, ea + eb);
    let r = match c {
        None if pm == 0 => (ps as u32) << 31,
        None => round_f32_ftz(ps, pm, pe, rnd, ftz),
        Some(c) => {
            let (cs, cm, ce) = f32_split(c);
            if pm == 0 && cm == 0 {
                // An exact zero sum: -0 only for (-0) + (-0), or opposite signs rounding down.
                if ps == cs {
                    (ps as u32) << 31
                } else {
                    ((rnd == ptxv_ir::Rnd::Rm) as u32) << 31
                }
            } else if pm == 0 {
                c
            } else if cm == 0 {
                round_f32_ftz(ps, pm, pe, rnd, ftz)
            } else {
                add_terms((ps, pm, pe), (cs, cm as u128, ce), rnd, ftz)
            }
        }
    };
    fl(r)
}

/// Round (-1)^s1 m1 2^e1 + (-1)^s2 m2 2^e2 (m1, m2 nonzero, at most 48 bits) once. The term
/// with the larger exponent is shifted up by at most 72 bits; the other is aligned to it in
/// units of 2^ue, with an odd last bit standing for any bits shifted out (a sticky bit that
/// lies strictly inside the interval of the true value, below every rounding boundary).
fn add_terms(t1: (bool, u128, i32), t2: (bool, u128, i32), rnd: ptxv_ir::Rnd, ftz: bool) -> u32 {
    let (hi, lo) = if t1.2 >= t2.2 { (t1, t2) } else { (t2, t1) };
    let d = (hi.2 - lo.2) as u32;
    let s = d.min(72);
    let k = d - s;
    let ue = hi.2 - s as i32 - 3;
    let h = hi.1 << (s + 3);
    let l = if k <= 2 {
        lo.1 << (3 - k)
    } else if k - 2 >= 64 {
        1
    } else {
        let sh = k - 2;
        ((lo.1 >> sh) << 1) | (lo.1 & ((1u128 << sh) - 1) != 0) as u128
    };
    let (neg, mag) = if hi.0 == lo.0 {
        (hi.0, h + l)
    } else if h >= l {
        (hi.0, h - l)
    } else {
        (lo.0, l - h)
    };
    if mag == 0 {
        return ((rnd == ptxv_ir::Rnd::Rm) as u32) << 31; // exact cancellation
    }
    round_f32_ftz(neg, mag, ue, rnd, ftz)
}

/// The f64 NaN that an invalid DFMA / DMUL / DADD (inf * 0, inf - inf) produces (GB300,
/// fixtures/s_f64).
pub const F64_CANON_NAN: u64 = 0xfff8_0000_0000_0000;

/// A finite f64 as (sign, significand, exponent): value = (-1)^sign * m * 2^e.
fn f64_split(u: u64) -> (bool, u128, i32) {
    let ex = ((u >> 52) & 0x7ff) as i32;
    let fr = (u & ((1u64 << 52) - 1)) as u128;
    if ex == 0 {
        (u >> 63 == 1, fr, -1074)
    } else {
        (u >> 63 == 1, fr | 1 << 52, ex - 1075)
    }
}

/// Round (-1)^neg * mag * 2^e (mag > 0) to f64 in mode `rnd`: 53-bit significand, gradual
/// underflow, IEEE overflow.
pub fn round_f64(neg: bool, mag: u128, e: i32, rnd: ptxv_ir::Rnd) -> u64 {
    use ptxv_ir::Rnd;
    let sign = (neg as u64) << 63;
    let top = e + 127 - mag.leading_zeros() as i32;
    let mut q = (top - 52).max(-1074);
    let sh = q - e;
    let (mut keep, guard, rest) = if sh <= 0 {
        (mag << (-sh) as u32, false, false)
    } else if sh > 128 {
        (0, false, true)
    } else {
        let sh = sh as u32;
        let keep = if sh == 128 { 0 } else { mag >> sh };
        let guard = mag >> (sh - 1) & 1 == 1;
        let rest = mag & ((1u128 << (sh - 1)) - 1) != 0;
        (keep, guard, rest)
    };
    let inc = match rnd {
        Rnd::Rn => guard && (rest || keep & 1 == 1),
        Rnd::Rz => false,
        Rnd::Rp => !neg && (guard || rest),
        Rnd::Rm => neg && (guard || rest),
    };
    keep += inc as u128;
    if keep >> 53 != 0 {
        keep >>= 1;
        q += 1;
    }
    if keep < 1 << 52 {
        return sign | keep as u64;
    }
    let be = q + 1075;
    if be >= 2047 {
        let inf = match rnd {
            Rnd::Rn => true,
            Rnd::Rz => false,
            Rnd::Rp => !neg,
            Rnd::Rm => neg,
        };
        return sign
            | if inf {
                0x7ff0_0000_0000_0000
            } else {
                0x7fef_ffff_ffff_ffff
            };
    }
    sign | (be as u64) << 52 | (keep as u64 & ((1u64 << 52) - 1))
}

/// f64 fused multiply-add `a*b + c`, one rounding in mode `rnd` (DFMA; DMUL when `c` is None;
/// DADD is a*1 + c). Exact in integers. A NaN operand propagates, quieted, sign and payload
/// kept, in the priority b, c, a (GB300, fixtures/s_f64); an invalid operation gives
/// `F64_CANON_NAN`. ASSUME(N-FALU)
pub fn f64_fma(a: u64, b: u64, c: Option<u64>, rnd: ptxv_ir::Rnd) -> u64 {
    let nan = |u: u64| u & 0x7fff_ffff_ffff_ffff > 0x7ff0_0000_0000_0000;
    let inf = |u: u64| u & 0x7fff_ffff_ffff_ffff == 0x7ff0_0000_0000_0000;
    if let Some(n) = [Some(b), c, Some(a)]
        .into_iter()
        .flatten()
        .find(|&u| nan(u))
    {
        return n | 1 << 51;
    }
    let ps = (a ^ b) >> 63 == 1;
    let pzero = a << 1 == 0 || b << 1 == 0;
    if inf(a) || inf(b) {
        if pzero || c.is_some_and(|c| inf(c) && (c >> 63 == 1) != ps) {
            return F64_CANON_NAN;
        }
        return (ps as u64) << 63 | 0x7ff0_0000_0000_0000;
    }
    if let Some(c) = c.filter(|&c| inf(c)) {
        return c;
    }
    let (_, ma, ea) = f64_split(a);
    let (_, mb, eb) = f64_split(b);
    let (pm, pe) = (ma * mb, ea + eb);
    match c {
        None if pm == 0 => (ps as u64) << 63,
        None => round_f64(ps, pm, pe, rnd),
        Some(c) => {
            let (cs, cm, ce) = f64_split(c);
            if pm == 0 && cm == 0 {
                if ps == cs {
                    (ps as u64) << 63
                } else {
                    ((rnd == ptxv_ir::Rnd::Rm) as u64) << 63
                }
            } else if pm == 0 {
                c
            } else if cm == 0 {
                round_f64(ps, pm, pe, rnd)
            } else {
                add_terms64((ps, pm, pe), (cs, cm, ce), rnd)
            }
        }
    }
}

/// Round the sum of two f64 terms (significands of at most 106 bits) once. Exact when both
/// fit in 126 bits aligned to the smaller exponent; otherwise the term with the lower leading
/// bit lies more than 19 bits below the other (it cannot cancel more than one bit) and is
/// aligned with a sticky bit far below the rounding position.
fn add_terms64(t1: (bool, u128, i32), t2: (bool, u128, i32), rnd: ptxv_ir::Rnd) -> u64 {
    let top = |t: &(bool, u128, i32)| t.2 + 128 - t.1.leading_zeros() as i32;
    let (big, small) = if top(&t1) >= top(&t2) {
        (t1, t2)
    } else {
        (t2, t1)
    };
    let tt = top(&big);
    let emin = t1.2.min(t2.2);
    let (ue, h, l) = if tt - emin <= 125 {
        (emin, big.1 << (big.2 - emin), small.1 << (small.2 - emin))
    } else {
        let ue = tt - 125;
        let sh = (ue - small.2) as u32;
        let l = if sh >= 127 {
            1
        } else {
            (small.1 >> sh) << 1 | (small.1 & ((1u128 << sh) - 1) != 0) as u128
        };
        // `l` carries one extra (sticky) bit: align `h` to it too.
        (ue - 1, big.1 << (big.2 - ue + 1), l)
    };
    let (neg, mag) = if big.0 == small.0 {
        (big.0, h + l)
    } else if h >= l {
        (big.0, h - l)
    } else {
        (small.0, l - h)
    };
    if mag == 0 {
        return ((rnd == ptxv_ir::Rnd::Rm) as u64) << 63;
    }
    round_f64(neg, mag, ue, rnd)
}

fn isqrt_u128(m: u128) -> u128 {
    if m == 0 {
        return 0;
    }
    let mut x = ((m as f64).sqrt() as u128).max(1);
    for _ in 0..4 {
        x = (x + m / x) / 2; // Newton
    }
    // Fix the estimate: x = floor(sqrt(m)).
    while x * x > m {
        x -= 1;
    }
    while (x + 1) * (x + 1) <= m {
        x += 1;
    }
    x
}

/// IEEE f64 division a / b (`b` Some) or square root of a, correctly rounded in `rnd`. NaN
/// results: `nan_of` gives the propagated NaN of an operand, `F64_CANON_NAN` an invalid
/// operation (see the PTX op's documentation).
pub fn f64_div_sqrt(a: u64, b: Option<u64>, rnd: ptxv_ir::Rnd) -> u64 {
    let nan = |u: u64| u & 0x7fff_ffff_ffff_ffff > 0x7ff0_0000_0000_0000;
    let inf = |u: u64| u & 0x7fff_ffff_ffff_ffff == 0x7ff0_0000_0000_0000;
    let zero = |u: u64| u << 1 == 0;
    let norm = |u: u64| {
        let (s, mut m, mut e) = f64_split(u);
        while m >> 52 == 0 {
            m <<= 1;
            e -= 1;
        }
        (s, m, e)
    };
    match b {
        Some(b) => {
            if let Some(n) = [a, b].into_iter().find(|&u| nan(u)) {
                return n | 1 << 51;
            }
            let s = (a ^ b) >> 63 == 1;
            let sg = (s as u64) << 63;
            if inf(a) && inf(b) || zero(a) && zero(b) {
                return F64_CANON_NAN;
            }
            if inf(a) || zero(b) {
                return sg | 0x7ff0_0000_0000_0000;
            }
            if zero(a) || inf(b) {
                return sg;
            }
            let (_, ma, ea) = norm(a);
            let (_, mb, eb) = norm(b);
            let num = ma << 64;
            let (q, r) = (num / mb, num % mb);
            round_f64(s, q << 1 | (r != 0) as u128, ea - eb - 65, rnd)
        }
        None => {
            if nan(a) {
                return a | 1 << 51;
            }
            if zero(a) {
                return a;
            }
            if a >> 63 == 1 {
                return F64_CANON_NAN;
            }
            if inf(a) {
                return a;
            }
            let (_, mut m, mut e) = norm(a);
            if e & 1 != 0 {
                m <<= 1;
                e -= 1;
            }
            let k = (126 - (128 - m.leading_zeros() as i32)) / 2;
            let mm = m << (2 * k);
            let s = isqrt_u128(mm);
            let r = mm - s * s;
            round_f64(false, s << 1 | (r != 0) as u128, (e - 2 * k) / 2 - 1, rnd)
        }
    }
}

/// f64 comparison (DSETP), as `f32_cmp`.
pub fn f64_cmp(a: u64, b: u64, cmp: ptxv_ir::FCmp) -> bool {
    use ptxv_ir::FCmp::*;
    let (x, y) = (f64::from_bits(a), f64::from_bits(b));
    let un = x.is_nan() || y.is_nan();
    match cmp {
        F => false,
        T => true,
        Lt => x < y,
        Eq => x == y,
        Le => x <= y,
        Gt => x > y,
        Ne => !un && x != y,
        Ge => x >= y,
        Num => !un,
        Nan => un,
        Ltu => un || x < y,
        Equ => un || x == y,
        Leu => un || x <= y,
        Gtu => un || x > y,
        Neu => un || x != y,
        Geu => un || x >= y,
    }
}

/// `.SAT`: clamp to [+0, 1]; NaN becomes +0.
pub fn f32_sat(u: u32) -> u32 {
    if f32_is_nan(u) || u >> 31 == 1 {
        0
    } else if u > 0x3f80_0000 {
        0x3f80_0000
    } else {
        u
    }
}

/// `min.f32` / `max.f32` (SASS FMNMX): a NaN operand is ignored unless both are NaN (`nan`,
/// `.NAN`: any NaN operand gives NaN); -0 is below +0.
pub fn f32_minmax(a: u32, b: u32, min: bool, nan: bool, ftz: bool) -> u32 {
    let fl = |u: u32| if ftz { f32_ftz(u) } else { u };
    let (a, b) = (fl(a), fl(b));
    match (f32_is_nan(a), f32_is_nan(b)) {
        (true, true) => return F32_CANON_NAN,
        (true, false) | (false, true) if nan => return F32_CANON_NAN,
        (true, false) => return b,
        (false, true) => return a,
        _ => {}
    }
    // Order as sign-magnitude integers: -0 < +0.
    let key = |u: u32| {
        if u >> 31 == 1 {
            -((u & 0x7fff_ffff) as i64) - 1
        } else {
            u as i64
        }
    };
    let a_lt = key(a) < key(b);
    if a_lt == min { a } else { b }
}

/// `setp.<cmp>.f32` (SASS FSETP).
pub fn f32_cmp(a: u32, b: u32, cmp: ptxv_ir::FCmp, ftz: bool) -> bool {
    use ptxv_ir::FCmp::*;
    let fl = |u: u32| if ftz { f32_ftz(u) } else { u };
    let (x, y) = (f32::from_bits(fl(a)), f32::from_bits(fl(b)));
    let un = x.is_nan() || y.is_nan();
    match cmp {
        F => false,
        T => true,
        Lt => x < y,
        Eq => x == y,
        Le => x <= y,
        Gt => x > y,
        Ne => !un && x != y,
        Ge => x >= y,
        Num => !un,
        Nan => un,
        Ltu => un || x < y,
        Equ => un || x == y,
        Leu => un || x <= y,
        Gtu => un || x > y,
        Neu => un || x != y,
        Geu => un || x >= y,
    }
}

/// SASS `FCHK P, a, b`: does a / b leave the range where ptxas's fast IEEE division sequence
/// (reciprocal estimate, Newton step, two residual corrections) is exact? Measured on GB300
/// (probes/fchk, every pair of 2 x 128 operands): with biased exponent fields ea, eb, the
/// predicate is clear exactly when 25 <= ea <= 254, 2 <= eb <= 251 and -124 <= ea - eb <= 126;
/// signs and significands do not matter (zero, subnormal, infinite and NaN operands set it).
pub fn fchk(a: u32, b: u32) -> bool {
    let ea = ((a >> 23) & 0xff) as i32;
    let eb = ((b >> 23) & 0xff) as i32;
    !((25..=254).contains(&ea) && (2..=251).contains(&eb) && (-124..=126).contains(&(ea - eb)))
}

/// Round an f64 to f32 in the given mode (exact for integer inputs below 2^53).
pub fn f64_to_f32(v: f64, rnd: ptxv_ir::Rnd) -> f32 {
    use ptxv_ir::Rnd;
    let n = v as f32;
    if n.is_nan() || (n as f64) == v {
        return n;
    }
    let up = |f: f32| {
        f32::from_bits(if f >= 0.0 {
            f.to_bits() + 1
        } else {
            f.to_bits() - 1
        })
    };
    let down = |f: f32| -up(-f);
    match rnd {
        Rnd::Rn => n,
        Rnd::Rp if (n as f64) < v => up(n),
        Rnd::Rm if (n as f64) > v => down(n),
        Rnd::Rz if (n as f64).abs() > v.abs() => {
            if n > 0.0 {
                down(n)
            } else {
                up(n)
            }
        }
        _ => n,
    }
}

/// Integer (u64, or i64 when `signed`) to f32 in the given mode, exactly rounded.
pub fn int_to_f32(v: u64, signed: bool, rnd: ptxv_ir::Rnd) -> f32 {
    let (neg, m) = if signed && (v as i64) < 0 {
        (true, (v as i64).unsigned_abs())
    } else {
        (false, v)
    };
    // Beyond 53 bits, round to odd at 53 bits first (exact in f64, and rounding that to 24
    // bits gives the correctly rounded result).
    let s = (64 - m.leading_zeros()).saturating_sub(53);
    let t = if s > 0 {
        let t = (m >> s) << s;
        t | (((t != m) as u64) << s)
    } else {
        m
    };
    let f = if neg { -(t as f64) } else { t as f64 };
    f64_to_f32(f, rnd)
}

/// f32 -> bf16 as the GPU's converters (`F2FP.BF16.F32.PACK_AB`, `F2F.BF16.F32`): round to
/// nearest even, every NaN -> the canonical 0x7fff (GB300: fixtures/s_ptx_cute_ops).
pub fn f32_to_bf16_rn_gpu(u: u32) -> u16 {
    if f32::from_bits(u).is_nan() {
        0x7fff
    } else {
        f32_to_bf16_rn(u)
    }
}

/// f32 -> bf16, round to nearest even (NaN: quieted, high half kept; torch's host conversion).
pub fn f32_to_bf16_rn(u: u32) -> u16 {
    if f32::from_bits(u).is_nan() {
        return ((u >> 16) | 0x40) as u16;
    }
    ((u as u64 + 0x7fff + ((u as u64 >> 16) & 1)) >> 16) as u16
}

/// f32 -> f16, round to nearest even (subnormals kept, overflow to infinity, NaN -> 0x7fff).
pub fn f32_to_f16_rn(x: u32) -> u16 {
    let s = (x >> 16 & 0x8000) as u16;
    let f = f32::from_bits(x);
    if f.is_nan() {
        return 0x7fff;
    }
    let v = f.abs() as f64;
    if v.is_infinite() || v >= 65520.0 {
        return s | 0x7c00;
    }
    let rne = |q: f64| {
        let fl = q.floor();
        let d = q - fl;
        (if d > 0.5 || d == 0.5 && fl % 2.0 != 0.0 {
            fl + 1.0
        } else {
            fl
        }) as u32
    };
    if v < 2f64.powi(-14) {
        // Subnormal (or zero): units of 2^-24; 1024 is the smallest normal.
        return s | rne(v * 2f64.powi(24)) as u16;
    }
    let mut e = v.log2().floor() as i32;
    // log2 can be off by one at binade edges.
    if 2f64.powi(e) > v {
        e -= 1;
    } else if 2f64.powi(e + 1) <= v {
        e += 1;
    }
    let mut m = rne(v * 2f64.powi(10 - e));
    if m == 2048 {
        m = 1024;
        e += 1;
    }
    if e + 15 >= 31 {
        return s | 0x7c00;
    }
    s | (((e + 15) as u16) << 10) | (m - 1024) as u16
}

pub fn f16_to_f64(h: u16) -> f64 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    s * match e {
        0 => m * 2f64.powi(-24),
        31 => {
            if m == 0.0 {
                f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

/// f16 fused multiply-add with a single round-to-nearest-even, computed exactly in i128.
/// Every finite f16 is an integer multiple of 2^-24, so a*b is a multiple of 2^-48 and
/// a*b + c is an exact integer after scaling by 2^48.
pub fn f16_fma_rn(a: u16, b: u16, c: u16) -> u16 {
    let fa = f16_to_f64(a);
    let fb = f16_to_f64(b);
    let fc = f16_to_f64(c);
    if fa.is_nan()
        || fb.is_nan()
        || fc.is_nan()
        || fa.is_infinite()
        || fb.is_infinite()
        || fc.is_infinite()
    {
        let r = fa * fb + fc; // special values: f64 semantics match IEEE here
        return if r.is_nan() {
            0x7fff
        } else if r > 0.0 {
            0x7c00
        } else {
            0xfc00
        };
    }
    let ia = (fa * 2f64.powi(24)) as i128;
    let ib = (fb * 2f64.powi(24)) as i128;
    let ic = (fc * 2f64.powi(24)) as i128;
    let exact = ia * ib + (ic << 24); // scaled by 2^48
    if exact == 0 {
        // Sign of an exact zero sum under RN: +0 unless both addends are -0.
        let prod_neg = (a ^ b) & 0x8000 != 0 && (fa == 0.0 || fb == 0.0);
        let both_neg = prod_neg && c & 0x8000 != 0;
        return if both_neg { 0x8000 } else { 0 };
    }
    round_scaled_to_f16(exact, 48)
}

/// Round `v * 2^-scale` (v != 0) to f16, round-to-nearest-even.
fn round_scaled_to_f16(v: i128, scale: i32) -> u16 {
    let sign: u16 = if v < 0 { 0x8000 } else { 0 };
    let mag = v.unsigned_abs();
    // exponent of the leading bit: value = mag * 2^-scale
    let lead = 127 - mag.leading_zeros() as i32; // bit index
    let e_unb = lead - scale; // floor(log2(value))
    // f16: normal exponent range [-14, 15]; quantum = 2^(max(e,-14) - 10)
    let q_exp = e_unb.max(-14) - 10;
    let shift = q_exp + scale; // number of low bits to drop from mag
    let (mut m, round_up) = if shift <= 0 {
        (mag << (-shift) as u32, false)
    } else {
        let s = shift as u32;
        let kept = mag >> s;
        let rem = mag & ((1u128 << s) - 1);
        let half = 1u128 << (s - 1);
        let up = rem > half || (rem == half && kept & 1 == 1);
        (kept, up)
    };
    if round_up {
        m += 1;
    }
    // m is the significand in units of 2^q_exp
    let mut e = q_exp + 10; // exponent if m in [1024, 2048)
    if m >= 2048 {
        m >>= 1;
        e += 1;
    }
    if e > 15 {
        return sign | 0x7c00;
    }
    if m < 1024 {
        return sign | m as u16; // subnormal (e == -14 here)
    }
    sign | (((e + 15) as u16) << 10) | (m as u16 & 0x3ff)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hfma2_constant_idiom() {
        // HFMA2 R4, -RZ, RZ, 0, 5.96e-08 materializes 0x0001 in the low half.
        assert_eq!(f16_fma_rn(0x8000, 0, 1), 1);
        assert_eq!(f16_fma_rn(0x8000, 0, 0), 0);
        assert_eq!(f16_fma_rn(0x8000, 0, 0x3f80), 0x3f80);
        // 1.5 * 2 + 0.25 = 3.25
        assert_eq!(f16_to_f64(f16_fma_rn(0x3e00, 0x4000, 0x3400)), 3.25);
    }
}

// ------------------------------------------------------------------ tensor-core model

/// Parameterized model of one tcgen05.mma dot-product step (fp32 accumulate).
///
/// The hardware algorithm is undocumented. Following Fasi et al. (2021) the candidate family
/// is: split the K products into groups of `group`; for each group, align the products (and
/// the running accumulator, if `acc_in_group`) to the largest exponent, keep `frac_bits` bits
/// below it (`None` = exact), add exactly, and round once to fp32 (toward zero if
/// `final_rz`, else to nearest-even). Dropped bits are truncated toward zero, or toward
/// minus infinity if `trunc_floor` (a two's-complement right shift). Products are aligned by
/// their true exponent, or by e_a + e_b from the input exponent fields if `exp_from_inputs`.
/// Which member matches the silicon is decided by calibration on GB300 (`ptxv calib-mma`).
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
pub struct MmaModel {
    pub group: u8,
    pub frac_bits: Option<u8>,
    pub final_rz: bool,
    pub acc_in_group: bool,
    pub trunc_floor: bool,
    pub exp_from_inputs: bool,
}

/// One term of a dot-product step: exact value and the exponent used to align it.
#[derive(Clone, Copy, Debug)]
pub struct Term {
    pub v: f64,
    pub e: i32,
}

impl Term {
    pub fn plain(v: f64) -> Term {
        Term {
            v,
            e: if v == 0.0 { i32::MIN } else { exp_of(v) },
        }
    }
    /// Product of two bf16/f16/tf32 values; `inputs` selects alignment by e_a + e_b.
    pub fn prod(a: f64, b: f64, inputs: bool) -> Term {
        let v = a * b; // exact for <= 26-bit significands
        if v == 0.0 {
            return Term { v, e: i32::MIN };
        }
        Term {
            v,
            e: if inputs {
                exp_of(a) + exp_of(b)
            } else {
                exp_of(v)
            },
        }
    }
}

/// Model used when a case does not override it: the unique family member that matched all
/// 49,152 outputs of `ptxv calib-mma` on GB300 (sm_103a, kind::f16, bf16 x bf16 -> f32),
/// 2026-10-30, see docs/rung1a.md. All 16 products and the accumulator are aligned to the
/// largest e_a + e_b (or accumulator exponent), truncated toward zero to 25 bits below it,
/// summed exactly, and rounded toward zero to fp32.
// ASSUME(N-MMA): see ASSUMPTIONS.md
pub const DEFAULT_MMA: MmaModel = MmaModel {
    group: 16,
    frac_bits: Some(25),
    final_rz: true,
    acc_in_group: true,
    trunc_floor: false,
    exp_from_inputs: true,
};

/// SASS `HMMA.16816.F32.BF16` (`mma.sync.m16n8k16` bf16 x bf16 + f32) on GB300: see
/// N-HMMA in ASSUMPTIONS.md (probes/hmma).
// ASSUME(N-HMMA): see ASSUMPTIONS.md
pub const HMMA_MODEL: MmaModel = DEFAULT_MMA;

/// Numerics error contract of the narrow and block-scaled kinds (N-MMA-F8, N-MMA-BS). They
/// align on exponent / scale fields, not on term magnitudes, so N-MMA's derivation does not
/// apply; truncating toward zero still loses at most each term itself, and `frac_bits: 0`
/// gives `eps_terms` >= 2 (sound, too loose to prove a tolerance).
pub const NARROW_MMA_CONTRACT: MmaModel = MmaModel {
    group: 32,
    frac_bits: Some(0),
    final_rz: true,
    acc_in_group: true,
    trunc_floor: false,
    exp_from_inputs: true,
};

/// SASS `HMMA.1688.F32.TF32` (`mma.sync.m16n8k8` tf32 x tf32 + f32) on GB300: see N-HMMA in
/// ASSUMPTIONS.md (probes/tmma, `ptxv calib-mma`).
// ASSUME(N-HMMA): see ASSUMPTIONS.md
pub const TMMA_MODEL: MmaModel = MmaModel {
    group: 8,
    frac_bits: Some(25),
    final_rz: true,
    acc_in_group: true,
    trunc_floor: false,
    exp_from_inputs: true,
};

/// How the TF32 MMA reads an f32 register word: the low 13 mantissa bits are dropped
/// (probes/tmma). ASSUME(N-HMMA)
pub fn tf32_in(x: u32) -> u32 {
    x & !0x1fff
}

impl MmaModel {
    pub const EXACT_RN: MmaModel = MmaModel {
        group: 16,
        frac_bits: None,
        final_rz: false,
        acc_in_group: true,
        trunc_floor: false,
        exp_from_inputs: false,
    };

    pub fn name(&self) -> String {
        format!(
            "g{}-{}-{}-{}-{}-{}",
            self.group,
            self.frac_bits
                .map(|f| format!("f{f}"))
                .unwrap_or_else(|| "exact".into()),
            if self.final_rz { "rz" } else { "rn" },
            if self.acc_in_group { "accin" } else { "accsep" },
            if self.trunc_floor { "tfloor" } else { "tzero" },
            if self.exp_from_inputs {
                "einputs"
            } else {
                "eprod"
            }
        )
    }

    /// Parse a name produced by `name()`, e.g. `g16-f25-rz-accin-tfloor-einputs`.
    pub fn parse(s: &str) -> Option<MmaModel> {
        let p: Vec<&str> = s.split('-').collect();
        if p.len() != 6 {
            return None;
        }
        Some(MmaModel {
            group: p[0].strip_prefix('g')?.parse().ok()?,
            frac_bits: if p[1] == "exact" {
                None
            } else {
                Some(p[1].strip_prefix('f')?.parse().ok()?)
            },
            final_rz: p[2] == "rz",
            acc_in_group: p[3] == "accin",
            trunc_floor: p[4] == "tfloor",
            exp_from_inputs: p[5] == "einputs",
        })
    }

    /// Error contract of one full group step (`group` products and the accumulator):
    /// |error| <= eps * (sum of |terms| in the step). See `eps_terms`.
    pub fn eps(&self) -> f64 {
        self.eps_terms(self.group as usize + 1)
    }

    /// Error contract of one step that adds at most `n` possibly nonzero terms (products and
    /// the accumulator): |computed - exact sum| <= eps * sum |terms|, for every input.
    ///
    /// Derivation (bf16 / f16 products, f32 accumulator; ASSUME(N-MMA), docs/equiv-mma.md):
    /// - each term's alignment exponent e satisfies 2^e <= |term| (e_a + e_b <= the product's
    ///   exponent; the accumulator's own exponent), so 2^emax <= max |term| <= sum |terms|;
    /// - the terms are truncated to multiples of Q = 2^(emax - f): each loses less than Q
    ///   (toward zero, or toward -inf with `trunc_floor`), and the term that sets emax loses
    ///   nothing (a product of two 8- or 11-bit significands has at most 22 bits, the f32
    ///   accumulator 24, both <= f + 1 for f >= 23): at most (n - 1) Q < (n - 1) 2^-f sum|t|;
    /// - the exact sum of the truncated terms is rounded once to f32: relative 2^-23 toward
    ///   zero, 2^-24 to nearest, of a value <= (1 + trunc) sum |terms|.
    ///
    /// With `acc_in_group` false the group sum is rounded on its own and then added to the
    /// accumulator: one more rounding on the products' path. Exact models (no truncation)
    /// have only the final rounding.
    pub fn eps_terms(&self, n: usize) -> f64 {
        let fin = if self.final_rz {
            2f64.powi(-23)
        } else {
            2f64.powi(-24)
        };
        let n = if self.acc_in_group {
            n
        } else {
            n.saturating_sub(1).max(1)
        };
        let trunc = match self.frac_bits {
            Some(f) if f >= 23 => n.saturating_sub(1) as f64 * 2f64.powi(-(f as i32)),
            // Fewer kept bits than the accumulator's significand: the largest term may lose bits.
            Some(f) => n as f64 * 2f64.powi(1 - f as i32),
            None => 0.0,
        };
        let step = trunc + fin * (1.0 + trunc);
        if self.acc_in_group {
            step
        } else {
            (1.0 + step) * (1.0 + fin) - 1.0
        }
    }

    /// One MMA instruction's update of one output: acc + sum(products), per the model.
    pub fn apply_terms(&self, acc: f32, prods: &[Term]) -> f32 {
        let mut c = acc;
        for chunk in prods.chunks(self.group as usize) {
            if self.acc_in_group {
                let mut terms: Vec<Term> = chunk.to_vec();
                terms.push(Term::plain(c as f64));
                c = sum_round_terms(&terms, self.frac_bits, self.final_rz, self.trunc_floor);
            } else {
                let s = sum_round_terms(chunk, self.frac_bits, self.final_rz, self.trunc_floor);
                c = sum_round_f32(&[c as f64, s as f64], None, self.final_rz);
            }
        }
        c
    }

    /// Convenience: products given as exact values (aligned by their true exponent).
    pub fn apply(&self, acc: f32, prods: &[f64]) -> f32 {
        let t: Vec<Term> = prods.iter().map(|&v| Term::plain(v)).collect();
        self.apply_terms(acc, &t)
    }
}

/// Sum `terms`, optionally truncating each to `frac` bits below the largest alignment
/// exponent, and round once to f32.
pub fn sum_round_terms(terms: &[Term], frac: Option<u8>, rz: bool, floor: bool) -> f32 {
    let Some(f) = frac else {
        let v: Vec<f64> = terms.iter().map(|t| t.v).collect();
        return sum_round_f32(&v, None, rz);
    };
    let nz: Vec<&Term> = terms.iter().filter(|t| t.v != 0.0).collect();
    if nz.is_empty() {
        return if !terms.is_empty() && terms.iter().all(|t| t.v.is_sign_negative()) {
            -0.0
        } else {
            0.0
        };
    }
    if nz.iter().any(|t| !t.v.is_finite()) {
        return nz.iter().map(|t| t.v).sum::<f64>() as f32;
    }
    let emax = nz.iter().map(|t| t.e).max().unwrap();
    let q_exp = emax - f as i32;
    let mut acc: i128 = 0;
    for t in &nz {
        let scaled = t.v * 2f64.powi(-q_exp);
        acc += if floor {
            scaled.floor()
        } else {
            scaled.trunc()
        } as i128;
    }
    round_int_to_f32(acc, q_exp, rz, 0)
}

fn exp_of(x: f64) -> i32 {
    // floor(log2 |x|) for nonzero finite x (handles subnormal doubles too)
    let b = x.abs().to_bits();
    let e = ((b >> 52) & 0x7ff) as i32;
    if e == 0 {
        -1074 + (63 - (b & ((1 << 52) - 1)).leading_zeros() as i32)
    } else {
        e - 1023
    }
}

/// Sum `terms` (exact doubles), optionally truncating each to `frac` bits below the largest
/// exponent, and round once to f32.
pub fn sum_round_f32(terms: &[f64], frac: Option<u8>, rz: bool) -> f32 {
    let nz: Vec<f64> = terms.iter().copied().filter(|t| *t != 0.0).collect();
    if nz.is_empty() {
        // exact zero: -0 only if every term is -0
        return if terms.iter().all(|t| t.is_sign_negative()) && !terms.is_empty() {
            -0.0
        } else {
            0.0
        };
    }
    if nz.iter().any(|t| !t.is_finite()) {
        let s: f64 = nz.iter().sum();
        return s as f32;
    }
    let emax = nz.iter().map(|&t| exp_of(t)).max().unwrap();
    let q_exp = match frac {
        Some(f) => emax - f as i32,
        None => {
            let emin = nz.iter().map(|&t| exp_of(t) - 52).min().unwrap();
            emin.max(emax - 100)
        }
    };
    let mut acc: i128 = 0;
    let mut sticky = 0i32; // sign of any dropped remainder (exact path only, when span > 100 bits)
    for &t in &nz {
        let scaled = t * 2f64.powi(-q_exp); // exact: power-of-two scaling
        let tr = scaled.trunc();
        acc += tr as i128;
        if frac.is_none() && tr != scaled && sticky == 0 {
            sticky = if scaled > tr { 1 } else { -1 };
        }
    }
    round_int_to_f32(acc, q_exp, rz, sticky)
}

/// Round `v * 2^e` to f32 (24-bit significand, subnormals, overflow); `sticky` breaks ties.
pub fn round_int_to_f32(v: i128, e: i32, rz: bool, sticky: i32) -> f32 {
    if v == 0 {
        return 0.0;
    }
    let neg = v < 0;
    let mag = v.unsigned_abs();
    let lead = 127 - mag.leading_zeros() as i32;
    let e_unb = lead + e;
    let q = (e_unb.max(-126)) - 23; // quantum exponent
    let shift = q - e;
    let (mut m, inexact_up) = if shift <= 0 {
        (mag << (-shift) as u32, false)
    } else if shift >= 128 {
        (0u128, false)
    } else {
        let s = shift as u32;
        let kept = mag >> s;
        let rem = mag & ((1u128 << s) - 1);
        let half = 1u128 << (s - 1);
        let up = if rz {
            false
        } else {
            rem > half || (rem == half && (sticky > 0 || (sticky == 0 && kept & 1 == 1)))
        };
        (kept, up)
    };
    if inexact_up {
        m += 1;
    }
    let mut qe = q;
    if m >= (1u128 << 24) {
        m >>= 1;
        qe += 1;
    }
    let val = if qe + 23 > 127 {
        if rz { f32::MAX as f64 } else { f64::INFINITY }
    } else {
        m as f64 * 2f64.powi(qe)
    };
    let r = val as f32; // exact: m fits in 24 bits
    if neg { -r } else { r }
}

#[cfg(test)]
mod mma_tests {
    use super::*;
    #[test]
    fn exact_model_is_correct_rounding() {
        let p = [1.0, 2f64.powi(-30), -1.0, 3.0];
        assert_eq!(
            MmaModel::EXACT_RN.apply(0.0, &p),
            (3.0f64 + 2f64.powi(-30)) as f32
        );
        assert_eq!(sum_round_f32(&[1.0, 2f64.powi(-25)], None, false), 1.0);
        assert_eq!(
            sum_round_f32(&[1.0, 2f64.powi(-24) + 2f64.powi(-40)], None, false),
            1.0 + 2f32.powi(-23)
        );
        assert_eq!(
            sum_round_f32(&[1.0, -2f64.powi(-40)], None, true),
            1.0 - 2f32.powi(-24)
        );
    }
}

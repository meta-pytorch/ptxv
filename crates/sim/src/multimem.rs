/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Arithmetic of `multimem.ld_reduce` (SASS `LDGMC`): the NVLink switch reduces the values
//! loaded from every location a multicast address names. Calibrated on GB300 (4 GPUs, one
//! NVLink domain; probes/multimem, ASSUMPTIONS.md N-MULTIMEM):
//! - integers (add / min / max / and / or / xor, 32 and 64 bits): exact;
//! - f32 add: each operand truncated toward zero to a window of `F32_WINDOW` bits below the
//!   largest operand's last mantissa bit, the truncated values summed exactly, the sum rounded
//!   to nearest even. Bitwise on every lane where no operand bit falls below the window;
//!   lanes where one does are reported `inexact` (99.98% still match); results are
//!   independent of member order, of the issuing GPU and of the repetition;
//! - f16 / bf16 add (with or without `.acc::f32`): the result is one of the two neighbours
//!   of the sum; which one depends on the other lanes of its 8-byte group (`add16`; a
//!   calibrated rule that reproduces most but not all measured lanes): reported `inexact`;
//! - f16 / bf16 min / max: IEEE minNum / maxNum (a NaN operand is ignored, -0 < +0);
//! - a NaN operand in an add: the result is one of the NaN operands, quieted; several
//!   distinct NaNs are not modeled; inf + -inf is the canonical NaN.
//!
//! Everything else (f64, fp8, f16 overflow, all-NaN min / max) is reported as not modeled.
// ASSUME(N-MULTIMEM): see ASSUMPTIONS.md

use ptxv_ir::McOp;
use ptxv_ir::McTy;

/// Bits kept below the largest f32 operand's last mantissa bit (probes/multimem: 63
/// reproduces 262094 of 262144 scalar lanes; every miss truncated some operand bits, and
/// such lanes are reported `inexact`).
pub const F32_WINDOW: i32 = 63;

/// The reduced bytes, and whether some lane's rounding direction rests on the calibrated
/// (not exact) f16 / bf16 rule: GB300 returns that neighbour or the other one.
pub struct Reduced {
    pub bytes: Vec<u8>,
    pub inexact: bool,
}

/// Format of a float lane: (exponent bits, mantissa bits).
fn fmt(ty: McTy) -> Option<(u32, u32)> {
    match ty {
        McTy::F32 => Some((8, 23)),
        McTy::F16 => Some((5, 10)),
        McTy::Bf16 => Some((8, 7)),
        _ => None,
    }
}

fn lane(b: &[u8], w: usize, i: usize) -> u64 {
    let mut v = 0u64;
    for k in 0..w {
        v |= (b[i * w + k] as u64) << (8 * k);
    }
    v
}

pub fn reduce(op: McOp, ty: McTy, members: &[Vec<u8>]) -> Result<Reduced, String> {
    let w = ty.lane_bytes();
    let n = members.first().map_or(0, |m| m.len());
    if members.is_empty() || !n.is_multiple_of(w) || members.iter().any(|m| m.len() != n) {
        return Err("multimem.ld_reduce: bad operand sizes".into());
    }
    if matches!(ty, McTy::F16 | McTy::Bf16) && op == McOp::Add {
        return add16(ty, members, w);
    }
    let mut out = Vec::with_capacity(n);
    let mut inexact = false;
    for i in 0..n / w {
        let xs: Vec<u64> = members.iter().map(|m| lane(m, w, i)).collect();
        let r = match (ty, op) {
            (McTy::U32 | McTy::S32 | McTy::U64 | McTy::S64, _) => int_lane(op, ty, &xs),
            (McTy::F32, McOp::Add) => match add_lane(ty, &xs)? {
                Lane::Done(b) => b,
                Lane::Sum {
                    neg,
                    mag,
                    lsb,
                    dropped,
                } => {
                    inexact |= dropped;
                    match round_ne(neg, mag, lsb, (8, 23)) {
                        Some(b) => b,
                        // Overflow rounds to infinity (round to nearest).
                        None => (neg as u64) << 31 | 0x7f80_0000,
                    }
                }
            },
            (McTy::F16 | McTy::Bf16, McOp::Min | McOp::Max) => minmax_lane(op, ty, &xs)?,
            _ => {
                return Err(format!(
                    "multimem.ld_reduce .{op:?} on {ty:?} lanes: not modeled (not probed on GB300)"
                ));
            }
        };
        for k in 0..w {
            out.push((r >> (8 * k)) as u8);
        }
    }
    Ok(Reduced {
        bytes: out,
        inexact,
    })
}

/// f16 / bf16 add. A sum that is not representable rounds to one of its two neighbours: up
/// when frac + r >= 1, with frac its position between them and r a dither built from the
/// other three lanes of its 8-byte group (lanes l+1, l+2, l+3 mod 4): their truncated
/// mantissas interleaved, most significant bits first (probes/multimem sr3: exact on the
/// controlled sweeps; 99.7% of random bf16 lanes, 96.7% of random f16 lanes). Every such lane
/// is reported `inexact`.
// ASSUME(N-MULTIMEM-ROUND16): see ASSUMPTIONS.md
fn add16(ty: McTy, members: &[Vec<u8>], w: usize) -> Result<Reduced, String> {
    let f = fmt(ty).unwrap();
    let (_eb, mb) = f;
    let nl = members[0].len() / w;
    // Per lane: final bits, or (lo bits, frac numerator, ulp) for a sum between neighbours.
    enum R16 {
        Done(u64),
        Between { lo: u64, num: u128, ulp: u128 },
    }
    let mut lanes = Vec::with_capacity(nl);
    for i in 0..nl {
        let xs: Vec<u64> = members.iter().map(|m| lane(m, w, i)).collect();
        lanes.push(match add_lane(ty, &xs)? {
            Lane::Done(b) => R16::Done(b),
            Lane::Sum { neg, mag, lsb, .. } => match neighbours(neg, mag, lsb, f) {
                None => {
                    return Err("multimem.ld_reduce f16 / bf16 add overflow: not modeled".into());
                }
                Some((lo, 0, _)) => R16::Done(lo),
                Some((lo, num, ulp)) => R16::Between { lo, num, ulp },
            },
        });
    }
    let trunc = |l: &R16| match l {
        R16::Done(b) => *b,
        R16::Between { lo, .. } => *lo,
    };
    let mut out = Vec::with_capacity(members[0].len());
    let mut inexact = false;
    for i in 0..nl {
        let bits = match &lanes[i] {
            R16::Done(b) => *b,
            R16::Between { lo, num, ulp } => {
                inexact = true;
                let g0 = i / 4 * 4;
                let others: Vec<u64> = (1..4)
                    .filter_map(|j| lanes.get(g0 + (i - g0 + j) % 4).map(trunc))
                    .collect();
                let nb = mb * others.len() as u32;
                let mut r: u128 = 0;
                for b in (0..mb).rev() {
                    for o in &others {
                        r = r << 1 | ((o >> b) & 1) as u128;
                    }
                }
                // up iff num / ulp + r / 2^nb >= 1
                if num << nb >= ((1u128 << nb) - r) * ulp {
                    lo + 1
                } else {
                    *lo
                }
            }
        };
        for k in 0..w {
            out.push((bits >> (8 * k)) as u8);
        }
    }
    Ok(Reduced {
        bytes: out,
        inexact,
    })
}

/// The neighbours of `±mag * 2^e` (mag > 0) in the format: (the one toward zero with its
/// sign, the distance from it to the value and the spacing, both in units of 2^e). None:
/// beyond the largest finite value.
fn neighbours(neg: bool, mag: u128, e: i32, (eb, mb): (u32, u32)) -> Option<(u64, u128, u128)> {
    let bias = (1i32 << (eb - 1)) - 1;
    let emin = 1 - bias - mb as i32;
    let top = 127 - mag.leading_zeros() as i32;
    let lsb = (top + e - mb as i32).max(emin);
    let sh = lsb - e;
    let (q, num, ulp) = if sh <= 0 {
        (mag << (-sh) as u32, 0, 1)
    } else {
        (
            mag >> sh as u32,
            mag & ((1u128 << sh as u32) - 1),
            1u128 << sh as u32,
        )
    };
    let s = (neg as u64) << (eb + mb);
    let bexp = if q >> mb == 0 { 0 } else { lsb - emin + 1 };
    if bexp >= (1 << eb) - 1 {
        return None;
    }
    Some((
        s | (bexp as u64) << mb | (q as u64 & ((1 << mb) - 1)),
        num,
        ulp,
    ))
}

fn int_lane(op: McOp, ty: McTy, xs: &[u64]) -> u64 {
    let wide = matches!(ty, McTy::U64 | McTy::S64);
    let mask = if wide { u64::MAX } else { 0xffff_ffff };
    let sx = |x: u64| -> i64 {
        if wide {
            x as i64
        } else {
            x as u32 as i32 as i64
        }
    };
    let signed = matches!(ty, McTy::S32 | McTy::S64);
    let mut acc = xs[0];
    for &x in &xs[1..] {
        acc = match op {
            McOp::Add => acc.wrapping_add(x),
            McOp::And => acc & x,
            McOp::Or => acc | x,
            McOp::Xor => acc ^ x,
            McOp::Min if signed => {
                if sx(x) < sx(acc) {
                    x
                } else {
                    acc
                }
            }
            McOp::Max if signed => {
                if sx(x) > sx(acc) {
                    x
                } else {
                    acc
                }
            }
            McOp::Min => acc.min(x),
            McOp::Max => acc.max(x),
        } & mask;
    }
    acc & mask
}

struct Dec {
    neg: bool,
    /// Value = man * 2^exp (man includes the hidden bit; 0 for zeros).
    man: u64,
    exp: i32,
}

enum Cls {
    Nan(u64),
    Inf(bool),
    Fin(Dec),
}

fn decode(x: u64, (eb, mb): (u32, u32)) -> Cls {
    let neg = x >> (eb + mb) & 1 == 1;
    let e = (x >> mb) & ((1 << eb) - 1);
    let m = x & ((1 << mb) - 1);
    let bias = (1i32 << (eb - 1)) - 1;
    if e == (1 << eb) - 1 {
        return if m != 0 {
            Cls::Nan(x | 1 << (mb - 1))
        } else {
            Cls::Inf(neg)
        };
    }
    let (man, exp) = if e == 0 {
        (m, 1 - bias - mb as i32)
    } else {
        (m | 1 << mb, e as i32 - bias - mb as i32)
    };
    Cls::Fin(Dec { neg, man, exp })
}

/// Round `mag * 2^e` (mag > 0) to nearest even in the format; overflow gives `None`.
fn round_ne(neg: bool, mag: u128, e: i32, (eb, mb): (u32, u32)) -> Option<u64> {
    let bias = (1i32 << (eb - 1)) - 1;
    let emin = 1 - bias - mb as i32; // exponent of the subnormal lsb
    let top = 127 - mag.leading_zeros() as i32; // mag in [2^top, 2^(top+1))
    // Exponent of the result's lsb: mb bits below the leading bit, at least emin.
    let lsb = (top + e - mb as i32).max(emin);
    let sh = lsb - e;
    let mut q = if sh <= 0 {
        mag << (-sh) as u32
    } else if sh >= 128 {
        0
    } else {
        let q = mag >> sh as u32;
        let rem = mag & ((1u128 << sh as u32) - 1);
        let half = 1u128 << (sh as u32 - 1);
        if rem > half || (rem == half && q & 1 == 1) {
            q + 1
        } else {
            q
        }
    };
    let mut lsb = lsb;
    if q >> (mb + 1) != 0 {
        q >>= 1;
        lsb += 1;
    }
    let s = (neg as u64) << (eb + mb);
    if q == 0 {
        return Some(s);
    }
    let bexp = if q >> mb == 0 {
        0 // subnormal
    } else {
        lsb - emin + 1
    };
    if bexp >= (1 << eb) - 1 {
        return None;
    }
    Some(s | (bexp as u64) << mb | (q as u64 & ((1 << mb) - 1)))
}

fn canonical_nan(ty: McTy) -> u64 {
    match ty {
        McTy::F32 => 0x7fc0_0000,
        McTy::Bf16 => 0x7fc0,
        _ => 0x7e00,
    }
}

enum Lane {
    Done(u64),
    /// The (window-truncated) sum `±mag * 2^lsb`, nonzero.
    Sum {
        neg: bool,
        mag: u128,
        lsb: i32,
        /// Some operand had bits below the window.
        dropped: bool,
    },
}

fn add_lane(ty: McTy, xs: &[u64]) -> Result<Lane, String> {
    let f = fmt(ty).unwrap();
    let (eb, mb) = f;
    let ds: Vec<Cls> = xs.iter().map(|&x| decode(x, f)).collect();
    let mut nans: Vec<u64> = ds
        .iter()
        .filter_map(|d| match d {
            Cls::Nan(q) => Some(*q),
            _ => None,
        })
        .collect();
    nans.sort_unstable();
    nans.dedup();
    match nans.len() {
        0 => {}
        1 => return Ok(Lane::Done(nans[0])),
        _ => {
            return Err(
                "multimem.ld_reduce add with several distinct NaN operands: which one GB300 returns is not modeled"
                    .into(),
            );
        }
    }
    let infs: Vec<bool> = ds
        .iter()
        .filter_map(|d| match d {
            Cls::Inf(n) => Some(*n),
            _ => None,
        })
        .collect();
    if let Some(&s) = infs.first() {
        return Ok(Lane::Done(if infs.iter().all(|&x| x == s) {
            (s as u64) << (eb + mb) | ((1 << eb) - 1) << mb
        } else {
            canonical_nan(ty)
        }));
    }
    let fin: Vec<&Dec> = ds
        .iter()
        .filter_map(|d| match d {
            Cls::Fin(d) => Some(d),
            _ => None,
        })
        .collect();
    if fin.iter().all(|d| d.man == 0) {
        // Zeros only: -0 when every operand is -0.
        let neg = fin.iter().all(|d| d.neg);
        return Ok(Lane::Done((neg as u64) << (eb + mb)));
    }
    // ASSUME(N-MULTIMEM): each operand truncated toward zero to `F32_WINDOW` bits below the
    // largest operand's last mantissa bit, then summed exactly.
    let emax = fin
        .iter()
        .filter(|d| d.man != 0)
        .map(|d| d.exp)
        .max()
        .unwrap();
    let lsb = emax - F32_WINDOW;
    let mut tot: i128 = 0;
    let mut dropped = false;
    for d in &fin {
        if d.man == 0 {
            continue;
        }
        let sh = d.exp - lsb;
        let q: i128 = if sh >= 0 {
            (d.man as i128) << sh
        } else if sh <= -64 {
            dropped = true;
            0
        } else {
            dropped |= d.man & ((1u64 << (-sh)) - 1) != 0;
            (d.man >> (-sh)) as i128
        };
        tot += if d.neg { -q } else { q };
    }
    if tot == 0 {
        return Ok(Lane::Done(0));
    }
    Ok(Lane::Sum {
        neg: tot < 0,
        mag: tot.unsigned_abs(),
        lsb,
        dropped,
    })
}

fn minmax_lane(op: McOp, ty: McTy, xs: &[u64]) -> Result<u64, String> {
    let f = fmt(ty).unwrap();
    // (value, zero sign): -0 below +0 (minNum / maxNum on GB300). f16 / bf16 are exact in f64.
    let key = |x: u64| -> Option<(f64, i8)> {
        match decode(x, f) {
            Cls::Nan(_) => None,
            Cls::Inf(n) => Some((if n { f64::NEG_INFINITY } else { f64::INFINITY }, 0)),
            Cls::Fin(d) => {
                let v = d.man as f64 * 2f64.powi(d.exp);
                Some((if d.neg { -v } else { v }, if d.neg { -1 } else { 0 }))
            }
        }
    };
    let mut best: Option<(u64, (f64, i8))> = None;
    for &x in xs {
        let Some(k) = key(x) else { continue };
        best = match best {
            None => Some((x, k)),
            Some((bx, bk)) => {
                let better = if op == McOp::Max { k > bk } else { k < bk };
                Some(if better { (x, k) } else { (bx, bk) })
            }
        };
    }
    best.map(|b| b.0).ok_or_else(|| {
        "multimem.ld_reduce min / max with every operand NaN: not modeled".to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32s(v: &[f32]) -> Vec<Vec<u8>> {
        v.iter()
            .map(|x| x.to_bits().to_le_bytes().to_vec())
            .collect()
    }

    #[test]
    fn f32_sum_rounds_to_nearest() {
        let r = reduce(McOp::Add, McTy::F32, &f32s(&[1.0, 2.0, 0.5, -0.25])).unwrap();
        assert_eq!(r.bytes, 3.25f32.to_bits().to_le_bytes());
        // A tie broken by a tiny operand below the window: truncated, so to even.
        let one = 1.0f32;
        let half_ulp = f32::from_bits(0x3380_0000); // 2^-24
        let tiny = f32::from_bits(0x0080_0000); // 2^-126: 102 bits below
        let r = reduce(McOp::Add, McTy::F32, &f32s(&[one, half_ulp, tiny, 0.0])).unwrap();
        assert_eq!(r.bytes, one.to_bits().to_le_bytes());
    }

    #[test]
    fn bf16_minmax_ignores_nan_and_orders_zeros() {
        let m = |v: &[u16]| {
            v.iter()
                .map(|x| x.to_le_bytes().to_vec())
                .collect::<Vec<_>>()
        };
        let r = reduce(McOp::Max, McTy::Bf16, &m(&[0x7fc1, 0x8000, 0x0000, 0xbf80])).unwrap();
        assert_eq!(r.bytes, [0, 0]);
        let r = reduce(McOp::Min, McTy::Bf16, &m(&[0x7fc1, 0x8000, 0x0000, 0x3f80])).unwrap();
        assert_eq!(r.bytes, [0, 0x80]);
    }
}

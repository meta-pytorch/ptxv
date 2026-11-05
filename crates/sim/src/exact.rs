/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Exact real arithmetic on sums of doubles (Shewchuk expansions).
//!
//! An expansion is a list of non-overlapping f64 components whose exact sum is the value.
//! Sums and differences of floating-point numbers are exact, which is all the numerics of
//! reduction trees need: every f32/bf16 value embeds exactly in f64, so exact partial sums,
//! exact local rounding errors (fp_noise Thm 1) and exact references come for free.

pub type Expansion = Vec<f64>;

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

/// e + b, zero components eliminated.
pub fn grow(e: &[f64], b: f64) -> Expansion {
    let mut out = Vec::with_capacity(e.len() + 1);
    let mut q = b;
    for &x in e {
        let (s, err) = two_sum(q, x);
        if err != 0.0 {
            out.push(err);
        }
        q = s;
    }
    if q != 0.0 || out.is_empty() {
        out.push(q);
    }
    out
}

pub fn add(e: &[f64], f: &[f64]) -> Expansion {
    let mut out: Expansion = e.to_vec();
    for &x in f {
        out = grow(&out, x);
    }
    out.retain(|&c| c != 0.0);
    out
}

/// Exact product of two expansions (each pair of components: a two-product).
pub fn mul(e: &[f64], f: &[f64]) -> Expansion {
    let mut out: Expansion = vec![];
    for &a in e {
        for &b in f {
            let p = a * b;
            let lo = a.mul_add(b, -p);
            out = grow(&out, p);
            if lo != 0.0 {
                out = grow(&out, lo);
            }
        }
    }
    out.retain(|&c| c != 0.0);
    out
}

pub fn neg(e: &[f64]) -> Expansion {
    e.iter().map(|x| -x).collect()
}

pub fn sub(e: &[f64], f: &[f64]) -> Expansion {
    add(e, &neg(f))
}

/// Nearest-double approximation of the value.
pub fn approx(e: &[f64]) -> f64 {
    // Components are ordered by increasing magnitude; summing from the small end is accurate
    // to within one rounding.
    let mut s = 0.0;
    for &c in e {
        s += c;
    }
    s
}

pub fn is_zero(e: &[f64]) -> bool {
    e.iter().all(|&c| c == 0.0)
}

pub fn from(x: f64) -> Expansion {
    if x == 0.0 { vec![] } else { vec![x] }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_cancellation() {
        let e = add(&from(1e20), &from(1.0));
        let e = add(&e, &from(-1e20));
        assert_eq!(approx(&e), 1.0);
        assert!(is_zero(&sub(&e, &from(1.0))));
    }
}

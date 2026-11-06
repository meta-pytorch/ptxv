/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The exact value of a non-chain output: the kernel's algorithm in real arithmetic on the
//! same (quantized) inputs. Every rounding is the identity, sums and products are evaluated in
//! double-double (106 bits), MUFU ops are their true functions (`num::mufu_true`), and
//! instruction slices the real functions they are identified with (`ident.rs`). The scale
//! S(x) of the u-scale unit propagates magnitudes: S(a + b) = S(a) + S(b),
//! S(a b + c) = S(a) S(b) + S(c), an MMA step adds its terms' magnitudes, and a nonlinear
//! op's S is |its value| (a product form a f(z): S(a) |f(z)|; a quotient: S(a) / |b|).
//! ASSUME(N-ERRDIST-REAL): see ASSUMPTIONS.md

use super::super::chain::Exact;
use super::super::chain::Op;
use super::super::chain::Prog;
use super::ident::Func;
use super::ops::NlOp;

/// A double-double value hi + lo.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Dd {
    pub hi: f64,
    pub lo: f64,
}

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

// Called by name throughout (r.add(x)); std::ops impls would need the traits at every call site.
#[allow(clippy::should_implement_trait)]
impl Dd {
    pub fn new(x: f64) -> Dd {
        Dd { hi: x, lo: 0.0 }
    }
    pub fn add(self, o: Dd) -> Dd {
        let (s, e) = two_sum(self.hi, o.hi);
        let e = e + self.lo + o.lo;
        let (h, l) = two_sum(s, e);
        if !h.is_finite() {
            return Dd::new(h);
        }
        Dd { hi: h, lo: l }
    }
    pub fn neg(self) -> Dd {
        Dd {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
    pub fn mul(self, o: Dd) -> Dd {
        let (p, e) = two_prod(self.hi, o.hi);
        let e = e + self.hi * o.lo + self.lo * o.hi;
        let (h, l) = two_sum(p, e);
        if !h.is_finite() {
            return Dd::new(h);
        }
        Dd { hi: h, lo: l }
    }
    pub fn v(self) -> f64 {
        self.hi + self.lo
    }
    fn gt(self, o: Dd) -> bool {
        self.hi > o.hi || (self.hi == o.hi && self.lo > o.lo)
    }
}

/// Real values (and scales) of every op of a program at the leaves of `w` (an evaluated or a
/// leaves-only state: only the leaves are read). NaN where an op has no real counterpart (an
/// unidentified slice) and above it.
/// S of a sum: the sum of |terms| of its expanded form, except that an exactly cancelling
/// sum (Kahan's compensation (t - s) - v, zero in real arithmetic) has none: counting its
/// terms would compound S through every later step (3^512 over a 512-step Kahan sum).
fn s_of(v: Dd, s: f64) -> f64 {
    if v.hi == 0.0 && v.lo == 0.0 { 0.0 } else { s }
}

pub fn eval(p: &Prog, w: &[crate::mma_lp::Operand]) -> Vec<(Dd, f64)> {
    let mut r: Vec<(Dd, f64)> = Vec::with_capacity(p.ops.len());
    let nan = (Dd::new(f64::NAN), f64::NAN);
    for (i, op) in p.ops.iter().enumerate() {
        let x = match op {
            Op::Leaf { .. } => (Dd::new(w[i].v), w[i].v.abs()),
            Op::Const(c) => (Dd::new(*c), c.abs()),
            // A TF32 split: hi(x) stands for x, the lo parts are errors of exact value 0
            // (N-ERRDIST-DAG).
            // ASSUME(N-ERRDIST-DAG): TF32 hi(x) stands for x, lo parts are errors; atomics in the simulator's order.
            Op::Tf32 { arg, part } => match part {
                crate::num::Tf32Part::Hi => r[*arg as usize],
                _ => (Dd::default(), 0.0),
            },
            Op::Mma {
                acc, terms, scales, ..
            } => {
                let (mut v, mut s) = match acc {
                    Some(a) => r[*a as usize],
                    None => (Dd::default(), 0.0),
                };
                for (j, &(a, b)) in terms.iter().enumerate() {
                    let mut t = r[a as usize].0.mul(r[b as usize].0);
                    if let Some(&(sa, sb)) = scales.get(j) {
                        t = t.mul(r[sa as usize].0).mul(r[sb as usize].0);
                    }
                    s += t.v().abs();
                    v = v.add(t);
                }
                (v, s)
            }
            Op::Round { arg, .. } => r[*arg as usize],
            Op::Add { a, b, .. } => {
                let (x, y) = (r[*a as usize], r[*b as usize]);
                let v = x.0.add(y.0);
                (v, s_of(v, x.1 + y.1))
            }
            Op::Fma {
                a,
                b,
                c,
                neg_ab,
                neg_c,
            } => {
                let (x, y) = (r[*a as usize], r[*b as usize]);
                let mut v = x.0.mul(y.0);
                if *neg_ab {
                    v = v.neg();
                }
                let mut s = x.1 * y.1;
                if let Some(c) = c {
                    let z = r[*c as usize];
                    v = v.add(if *neg_c { z.0.neg() } else { z.0 });
                    s += z.1;
                }
                (v, s_of(v, s))
            }
            Op::Nl(nl) => match nl {
                NlOp::Max { a, b, min } => {
                    let (x, y) = (r[*a as usize], r[*b as usize]);
                    // NaN-propagation order kept explicit (max.f32 returns the non-NaN operand).
                    #[allow(clippy::if_same_then_else)]
                    if x.0.hi.is_nan() {
                        y
                    } else if y.0.hi.is_nan() {
                        x
                    } else if x.0.gt(y.0) != *min {
                        x
                    } else {
                        y
                    }
                }
                NlOp::Mufu {
                    arg,
                    func,
                    ptx,
                    neg,
                    abs,
                } => {
                    let a = r[*arg as usize].0.v();
                    // Operand modifiers: |a|, then the sign.
                    let mut z = if *abs { a.abs() } else { a };
                    if *neg {
                        z = -z;
                    }
                    let v = crate::num::mufu_true(*func, *ptx, z);
                    (Dd::new(v), v.abs())
                }
                NlOp::Seg { args, real, .. } => match real {
                    None => nan,
                    Some(f) => {
                        let a: Vec<f64> = args.iter().map(|&j| r[j as usize].0.v()).collect();
                        if f.func == Func::Div {
                            // The quotient in double-double: q + (a - q b) / b.
                            let (n, d) = (r[args[0] as usize], r[args[1] as usize]);
                            let q = n.0.v() / d.0.v();
                            let rem = n.0.add(d.0.mul(Dd::new(q)).neg());
                            (
                                Dd::new(q).add(Dd::new(rem.v() / d.0.v())),
                                n.1 / d.0.v().abs(),
                            )
                        } else {
                            let v = f.eval(&a);
                            let s = match f.mul {
                                Some(j) if f.pow > 0 => {
                                    r[args[j] as usize].1 * f.func.f(f.z(&a)).abs()
                                }
                                Some(_) => v.abs(),
                                None => v.abs(),
                            };
                            (Dd::new(v), s)
                        }
                    }
                },
            },
        };
        r.push(x);
    }
    r
}

/// The `Exact` record of an op from `eval`'s table.
pub fn exact_of(t: &[(Dd, f64)], op: u32) -> Exact {
    let (v, s) = t[op as usize];
    Exact {
        y: v.hi,
        y_lo: v.lo,
        s,
    }
}

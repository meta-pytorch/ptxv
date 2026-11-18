/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Non-chain ops of the errdist program (`chain::Op::Nl`): instruction slices, MUFU ops and
//! exact maxima, extracted from segment-mode numerics DAGs (softmax, norms, attention,
//! elementwise kernels). Each evaluates bit-exactly (`seg::SProg::eval`, `mufu.rs`).

use std::sync::Arc;

use ptxv_ir::MufuFmt;
use ptxv_ir::MufuFunc;

use super::ident::RealFn;
use crate::num::MaxKind;
use crate::num::NodeKind;
use crate::seg::SProg;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum NlOp {
    /// An instruction slice of its arguments (`seg.rs`), and the real function it is identified
    /// with (`ident.rs`; None until identified, or when no candidate fits).
    Seg {
        prog: Arc<SProg>,
        args: Vec<u32>,
        real: Option<RealFn>,
    },
    /// A MUFU / PTX approx op of one child with its operand modifiers.
    Mufu {
        arg: u32,
        func: MufuFunc,
        ptx: bool,
        neg: bool,
        abs: bool,
    },
    /// The exact maximum (minimum) of two values, choosing as `kind` (FMNMX with or without
    /// .NAN, or a compare and select: the bits at zeros of either sign and at NaN operands;
    /// MISS 197, 202).
    Max {
        a: u32,
        b: u32,
        min: bool,
        kind: MaxKind,
    },
    /// A choice made by input data (`num::NodeKind::Sel`): `a` where the predicate slice of
    /// `args` (0 / 1) xor `neg` holds, else `b`. Re-evaluated per input (Tier 3); the analytic
    /// tiers follow one branch, with Tier 3's count of flipped branches as a bound
    /// (N-ERRDIST-SEL).
    Sel {
        pred: Arc<SProg>,
        args: Vec<u32>,
        neg: bool,
        a: u32,
        b: u32,
    },
}

/// Children of a numerics node errdist models as an `NlOp` (None: not such a node).
pub fn kids(k: &NodeKind) -> Option<Vec<u32>> {
    Some(match k {
        NodeKind::Seg(p) => p.0.args.clone(),
        NodeKind::Mufu { arg, .. } => vec![*arg],
        NodeKind::Max { a, b, .. } => vec![*a, *b],
        NodeKind::Sel { pred, a, b, .. } => pred.0.args.iter().copied().chain([*a, *b]).collect(),
        _ => return None,
    })
}

/// The `NlOp` of a node whose children are already mapped by `g`.
pub fn convert(k: &NodeKind, g: &dyn Fn(&u32) -> u32) -> Option<NlOp> {
    Some(match k {
        NodeKind::Seg(p) => NlOp::Seg {
            prog: p.0.clone(),
            args: p.0.args.iter().map(g).collect(),
            real: None,
        },
        NodeKind::Mufu {
            arg,
            func,
            ptx,
            neg,
            abs,
        } => NlOp::Mufu {
            arg: g(arg),
            func: *func,
            ptx: *ptx,
            neg: *neg,
            abs: *abs,
        },
        NodeKind::Max {
            a, b, min, kind, ..
        } => NlOp::Max {
            a: g(a),
            b: g(b),
            min: *min,
            kind: *kind,
        },
        NodeKind::Sel { pred, neg, a, b } => NlOp::Sel {
            pred: pred.0.clone(),
            args: pred.0.args.iter().map(g).collect(),
            neg: *neg,
            a: g(a),
            b: g(b),
        },
        _ => return None,
    })
}

/// Operand modifiers of a MUFU op applied to x (f32 bits).
pub fn mufu_operand(x: u32, neg: bool, abs: bool) -> u32 {
    let mut v = x;
    if abs {
        v &= 0x7fff_ffff;
    }
    if neg {
        v ^= 0x8000_0000;
    }
    v
}

/// The hardware value of a MUFU op at operand bits x (modifiers applied).
pub fn mufu_hw(func: MufuFunc, ptx: bool, x: u32) -> u32 {
    let r = if ptx {
        crate::mufu::ptx_approx(func, MufuFmt::F32, false, x)
    } else {
        crate::mufu::mufu(func, MufuFmt::F32, false, x)
    };
    r.unwrap_or(0x7fc0_0000)
}

impl NlOp {
    /// Children (program op indices).
    pub fn args(&self) -> Vec<u32> {
        match self {
            NlOp::Seg { args, .. } => args.clone(),
            NlOp::Mufu { arg, .. } => vec![*arg],
            NlOp::Max { a, b, .. } => vec![*a, *b],
            NlOp::Sel { args, a, b, .. } => args.iter().copied().chain([*a, *b]).collect(),
        }
    }

    /// The child a `Sel` / `Max` passes on at the children's f32 values (`val(i)`): the
    /// kernel's branch (`eval`). None for other ops.
    pub fn pick(&self, val: &dyn Fn(u32) -> f32, scratch: &mut Vec<u64>) -> Option<u32> {
        match self {
            NlOp::Max { a, b, min, kind } => {
                // The operand it returns (`b` for FMNMX's canonical NaN).
                let (x, y) = (val(*a).to_bits(), val(*b).to_bits());
                Some(if kind.picks_bits(*min, x, y) { *a } else { *b })
            }
            NlOp::Sel {
                pred,
                args,
                neg,
                a,
                b,
            } => {
                let v: Vec<f32> = args.iter().map(|&i| val(i)).collect();
                Some(if NlOp::sel_holds(pred, *neg, &v, scratch) {
                    *a
                } else {
                    *b
                })
            }
            NlOp::Seg { .. } | NlOp::Mufu { .. } => None,
        }
    }

    /// A `Sel`'s predicate at the given argument values (f32).
    pub fn sel_holds(pred: &SProg, neg: bool, args: &[f32], scratch: &mut Vec<u64>) -> bool {
        let a: Vec<u32> = args.iter().map(|x| x.to_bits()).collect();
        (pred.eval_with(&a, scratch).unwrap_or(0) != 0) != neg
    }

    /// The f32 value given each child's f32 value (`val(i)`), bit-exact.
    pub fn eval(&self, val: &dyn Fn(u32) -> f32, scratch: &mut Vec<u64>) -> f32 {
        match self {
            NlOp::Seg { prog, args, .. } => {
                let a: Vec<u32> = args.iter().map(|&i| val(i).to_bits()).collect();
                f32::from_bits(prog.eval_with(&a, scratch).unwrap_or(0x7fc0_0000))
            }
            NlOp::Mufu {
                arg,
                func,
                ptx,
                neg,
                abs,
            } => f32::from_bits(mufu_hw(
                *func,
                *ptx,
                mufu_operand(val(*arg).to_bits(), *neg, *abs),
            )),
            NlOp::Max { a, b, min, kind } => {
                f32::from_bits(kind.eval(*min, val(*a).to_bits(), val(*b).to_bits()))
            }
            NlOp::Sel { .. } => val(self.pick(val, scratch).unwrap()),
        }
    }

    /// A hash of the op's function (children excluded) for canonical forms.
    pub fn shape_key(&self) -> u64 {
        use std::hash::Hash;
        use std::hash::Hasher;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        match self {
            NlOp::Seg { prog, .. } => (0u8, super::ident::func_key(prog)).hash(&mut h),
            NlOp::Mufu {
                func,
                ptx,
                neg,
                abs,
                ..
            } => (1u8, format!("{func:?}"), ptx, neg, abs).hash(&mut h),
            NlOp::Max { min, kind, .. } => (2u8, min, kind).hash(&mut h),
            NlOp::Sel { pred, neg, .. } => (3u8, super::ident::func_key(pred), neg).hash(&mut h),
        }
        h.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MISS 197, 202: a maximum evaluates, picks and is keyed as its kind. a > b ? a : b passes
    /// b at a tie of +0 and -0 where FMNMX gives +0; FMNMX.NAN gives the canonical NaN where
    /// FMNMX passes the other operand.
    #[test]
    fn maxima_evaluate_as_their_kind() {
        let mut s = vec![];
        let max = |kind| NlOp::Max {
            a: 0,
            b: 1,
            min: false,
            kind,
        };
        let (gt, f, nan) = (
            max(MaxKind::Select(0b1110)),
            max(MaxKind::FMNMX),
            max(MaxKind::Fmnmx { nan: true }),
        );
        let zeros = |i: u32| [0.0f32, -0.0][i as usize];
        assert_eq!(gt.eval(&zeros, &mut s).to_bits(), 0x8000_0000);
        assert_eq!(gt.pick(&zeros, &mut s), Some(1));
        assert_eq!(f.eval(&zeros, &mut s).to_bits(), 0);
        assert_eq!(f.pick(&zeros, &mut s), Some(0));
        let one_nan = |i: u32| [1.0f32, f32::NAN][i as usize];
        assert_eq!(f.eval(&one_nan, &mut s), 1.0);
        assert_eq!(nan.eval(&one_nan, &mut s).to_bits(), 0x7fff_ffff);
        assert!(gt.eval(&one_nan, &mut s).is_nan());
        assert_ne!(gt.shape_key(), f.shape_key());
        assert_ne!(nan.shape_key(), f.shape_key());
    }
}

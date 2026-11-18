/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_ir::Op;
use ptxv_ir::Pred;
use ptxv_ir::Reg;
use ptxv_ir::Val;

use super::ComparisonEvaluator;
use super::ComparisonRequest;
use super::SKind;
use super::SOp;
use super::SVal;
use super::binding;
use super::comparison_instruction;
use super::program::ProgramEvaluator;

pub trait ComparisonPrimitive {
    fn compare(&self, request: ComparisonRequest) -> bool;
}

#[derive(Clone, Copy)]
pub struct IeeeComparison;

impl ComparisonPrimitive for IeeeComparison {
    fn compare(&self, request: ComparisonRequest) -> bool {
        crate::fp::f32_cmp(request.left, request.right, request.comparison, request.ftz)
    }
}

pub trait FallbackEvaluator {
    fn evaluate(&mut self, instruction: &SOp, inputs: binding::Inputs<'_>) -> Result<u64, String>;
}

pub struct Evaluator<C, F> {
    pub comparison: C,
    pub fallback: F,
}

pub fn direct_value(value: &Val) -> bool {
    matches!(value, Val::Reg(_, _) | Val::Imm(_))
}

pub fn eligible(operation: &Op) -> bool {
    match operation {
        Op::Fsetp { a, b, .. } | Op::Sel { a, b, .. } => direct_value(a) && direct_value(b),
        _ => false,
    }
}

struct Sources<'a, C> {
    bindings: &'a [(Reg, SVal)],
    inputs: binding::Inputs<'a>,
    comparison: C,
}

impl<C: ComparisonPrimitive> ComparisonEvaluator for Sources<'_, C> {
    fn read(&mut self, value: &Val) -> u32 {
        match value {
            Val::Reg(register, _) => {
                binding::register_bits(*register, self.bindings, self.inputs) as u32
            }
            Val::Imm(bits) => *bits as u32,
            _ => 0,
        }
    }

    fn predicate(&mut self, predicate: &Pred) -> bool {
        binding::read_predicate(predicate, self.bindings, self.inputs)
    }

    fn compare(&mut self, request: ComparisonRequest) -> bool {
        self.comparison.compare(request)
    }
}

pub fn try_evaluate<C: ComparisonPrimitive + Copy>(
    instruction: &SOp,
    inputs: binding::Inputs<'_>,
    comparison: C,
) -> Option<u64> {
    let SKind::Inst { op, regs, .. } = &instruction.kind else {
        return None;
    };
    if !eligible(op) {
        return None;
    }
    let mut sources = Sources {
        bindings: regs,
        inputs,
        comparison,
    };
    match comparison_instruction(op, &mut sources) {
        None => None,
        Some(bits) => Some(bits as u64),
    }
}

impl<C: ComparisonPrimitive + Copy, F: FallbackEvaluator> ProgramEvaluator for Evaluator<C, F> {
    fn evaluate(&mut self, instruction: &SOp, args: &[u32], prior: &[u64]) -> Result<u64, String> {
        let inputs = binding::Inputs { args, prior };
        match try_evaluate(instruction, inputs, self.comparison) {
            Some(bits) => Ok(bits),
            None => self.fallback.evaluate(instruction, inputs),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::panic::AssertUnwindSafe;
    use std::panic::catch_unwind;

    use ptxv_ir::BoolOp;
    use ptxv_ir::FCmp;
    use ptxv_ir::File;

    use super::*;
    use crate::seg::SProg;

    type Observation = (FCmp, u32, u32, bool);

    #[derive(Clone, Copy)]
    struct Primitive<'a> {
        calls: &'a RefCell<Vec<Observation>>,
        result: bool,
        panic: bool,
    }

    impl ComparisonPrimitive for Primitive<'_> {
        fn compare(&self, request: ComparisonRequest) -> bool {
            self.calls.borrow_mut().push((
                request.comparison,
                request.left,
                request.right,
                request.ftz,
            ));
            assert!(!self.panic, "primitive panic");
            self.result
        }
    }

    #[derive(Default)]
    struct Fallback {
        calls: Vec<(String, Vec<u32>, Vec<u64>)>,
        error: bool,
        panic: bool,
    }

    impl FallbackEvaluator for Fallback {
        fn evaluate(
            &mut self,
            instruction: &SOp,
            inputs: binding::Inputs<'_>,
        ) -> Result<u64, String> {
            self.calls.push((
                format!("{instruction:?}"),
                inputs.args.to_vec(),
                inputs.prior.to_vec(),
            ));
            assert!(!self.panic, "fallback panic");
            if self.error {
                Err("fallback error".into())
            } else {
                Ok(0x1234_5678_9abc_def0)
            }
        }
    }

    fn reg(file: File, idx: u16) -> Reg {
        Reg { file, idx }
    }

    fn instruction(op: Op, regs: Vec<(Reg, SVal)>) -> SOp {
        let dst = match &op {
            Op::Fsetp { d, .. } | Op::Sel { d, .. } => *d,
            _ => reg(File::R, 9),
        };
        SOp {
            pc: 0x7654_3210,
            kind: SKind::Inst {
                op,
                dst,
                regs,
                cvals: Vec::new(),
            },
        }
    }

    fn compare(a: Val, b: Val, c: Pred) -> Op {
        Op::Fsetp {
            cmp: FCmp::Neu,
            d: reg(File::P, 0),
            a,
            b,
            neg: [true, false],
            abs: [true, true],
            c,
            bop: BoolOp::Xor,
            ftz: true,
        }
    }

    #[test]
    fn actual_bindings_modifiers_predicate_and_request_reach_direct_path() {
        let log = RefCell::new(Vec::new());
        let mut evaluator = Evaluator {
            comparison: Primitive {
                calls: &log,
                result: true,
                panic: false,
            },
            fallback: Fallback::default(),
        };
        let left = reg(File::R, 17);
        let right = reg(File::UR, 9);
        let predicate = reg(File::UP, 3);
        let op = instruction(
            compare(
                Val::Reg(left, 2),
                Val::Reg(right, 1),
                Pred {
                    reg: Some(predicate),
                    neg: true,
                },
            ),
            vec![
                (left, SVal::Arg(1)),
                (right, SVal::Arg(0)),
                (predicate, SVal::Op(0)),
            ],
        );
        assert_eq!(
            evaluator.evaluate(&op, &[0x8123_4567, 0xff80_0000], &[0]),
            Ok(0)
        );
        assert_eq!(*log.borrow(), [(FCmp::Neu, 0xff80_0000, 0x0123_4567, true)]);
        assert!(evaluator.fallback.calls.is_empty());
    }

    #[test]
    fn direct_selection_preserves_negation_and_lazy_invalid_sources() {
        let log = RefCell::new(Vec::new());
        let mut evaluator = Evaluator {
            comparison: Primitive {
                calls: &log,
                result: true,
                panic: false,
            },
            fallback: Fallback::default(),
        };
        let a = reg(File::R, 4);
        let b = reg(File::UR, 4);
        let p = reg(File::P, 0);
        let mut op = instruction(
            Op::Sel {
                d: reg(File::R, 7),
                a: Val::Reg(a, 1),
                b: Val::Reg(b, 1),
                p: Pred {
                    reg: Some(p),
                    neg: false,
                },
            },
            vec![
                (a, SVal::Arg(u16::MAX)),
                (b, SVal::Const(u64::MAX)),
                (p, SVal::Op(0)),
            ],
        );
        assert_eq!(evaluator.evaluate(&op, &[], &[0]), Ok(u32::MAX as u64));
        if let SKind::Inst {
            op: Op::Sel { p, .. },
            ..
        } = &mut op.kind
        {
            p.neg = true;
        }
        assert!(catch_unwind(AssertUnwindSafe(|| evaluator.evaluate(&op, &[], &[0]))).is_err());
        assert!(log.borrow().is_empty());
        assert!(evaluator.fallback.calls.is_empty());
    }

    #[test]
    fn unsupported_kinds_and_values_preserve_complete_fallback_inputs() {
        let log = RefCell::new(Vec::new());
        let mut evaluator = Evaluator {
            comparison: Primitive {
                calls: &log,
                result: true,
                panic: false,
            },
            fallback: Fallback::default(),
        };
        let ops = [
            SOp {
                pc: 1,
                kind: SKind::Guard {
                    pred: SVal::Op(0),
                    neg: true,
                    new: SVal::Arg(1),
                    old: SVal::Const(8),
                },
            },
            SOp {
                pc: 2,
                kind: SKind::DivRn {
                    a: SVal::Arg(0),
                    b: SVal::Op(0),
                },
            },
            instruction(Op::Nop, Vec::new()),
            instruction(
                compare(Val::Const(7, 1), Val::Imm(8), Pred::TRUE),
                Vec::new(),
            ),
            instruction(
                Op::Sel {
                    d: reg(File::R, 8),
                    a: Val::Imm(9),
                    b: Val::Const(7, 1),
                    p: Pred::TRUE,
                },
                Vec::new(),
            ),
        ];
        for op in ops {
            assert_eq!(
                evaluator.evaluate(&op, &[0x8765_4321], &[u64::MAX]),
                Ok(0x1234_5678_9abc_def0)
            );
            assert_eq!(
                evaluator.fallback.calls.last().unwrap(),
                &(format!("{op:?}"), vec![0x8765_4321], vec![u64::MAX])
            );
        }
        assert_eq!(evaluator.fallback.calls.len(), 5);
        assert!(log.borrow().is_empty());
    }

    #[test]
    fn fallback_error_and_panic_observations_are_not_added_to_selected_path() {
        let log = RefCell::new(Vec::new());
        let mut evaluator = Evaluator {
            comparison: Primitive {
                calls: &log,
                result: true,
                panic: false,
            },
            fallback: Fallback {
                error: true,
                ..Default::default()
            },
        };
        let unsupported = instruction(Op::Nop, Vec::new());
        assert_eq!(
            evaluator.evaluate(&unsupported, &[5], &[9]),
            Err("fallback error".into())
        );
        assert_eq!(evaluator.fallback.calls.len(), 1);
        evaluator.fallback.panic = true;
        assert!(
            catch_unwind(AssertUnwindSafe(|| evaluator.evaluate(
                &unsupported,
                &[5],
                &[9]
            )))
            .is_err()
        );
        assert_eq!(evaluator.fallback.calls.len(), 2);
        let direct = instruction(
            Op::Sel {
                d: reg(File::R, 0),
                a: Val::Imm(17),
                b: Val::Imm(23),
                p: Pred::TRUE,
            },
            Vec::new(),
        );
        assert_eq!(evaluator.evaluate(&direct, &[], &[]), Ok(17));
        assert_eq!(evaluator.fallback.calls.len(), 2);
        assert!(log.borrow().is_empty());
    }

    #[test]
    fn primitive_panics_preserve_driver_prefix_and_skip_fallback() {
        let log = RefCell::new(Vec::new());
        let mut evaluator = Evaluator {
            comparison: Primitive {
                calls: &log,
                result: true,
                panic: true,
            },
            fallback: Fallback::default(),
        };
        let ops = [
            instruction(
                Op::Sel {
                    d: reg(File::R, 0),
                    a: Val::Imm(17),
                    b: Val::Imm(23),
                    p: Pred::TRUE,
                },
                Vec::new(),
            ),
            instruction(compare(Val::Imm(1), Val::Imm(2), Pred::TRUE), Vec::new()),
        ];
        let mut scratch = vec![999];
        let failure = catch_unwind(AssertUnwindSafe(|| {
            super::super::program::execute(
                super::super::program::Program {
                    args: &[],
                    ops: &ops,
                    out: SVal::Op(1),
                },
                &mut scratch,
                &mut evaluator,
            )
        }));
        assert!(failure.is_err());
        assert_eq!(scratch, [17]);
        assert_eq!(log.borrow().len(), 1);
        assert!(evaluator.fallback.calls.is_empty());
    }

    #[test]
    fn actual_sprog_mixes_constant_bank_fallback_and_direct_prior_binding() {
        let bank = Val::Const(7, 1);
        let mut comparison = compare(bank.clone(), Val::Imm(0x4000_0000), Pred::TRUE);
        if let Op::Fsetp {
            cmp,
            neg,
            abs,
            bop,
            ftz,
            ..
        } = &mut comparison
        {
            *cmp = FCmp::Lt;
            *neg = [false, false];
            *abs = [false, false];
            *bop = BoolOp::And;
            *ftz = false;
        }
        let mut first = instruction(comparison, Vec::new());
        if let SKind::Inst { cvals, .. } = &mut first.kind {
            cvals.push((bank, 0x4040_0000));
        }
        let left = reg(File::R, 0);
        let right = reg(File::R, 1);
        let predicate = reg(File::P, 0);
        let second = instruction(
            Op::Sel {
                d: reg(File::R, 9),
                a: Val::Reg(left, 1),
                b: Val::Reg(right, 1),
                p: Pred {
                    reg: Some(predicate),
                    neg: false,
                },
            },
            vec![
                (left, SVal::Arg(0)),
                (right, SVal::Arg(1)),
                (predicate, SVal::Op(0)),
            ],
        );
        let program = SProg {
            args: vec![0, 1],
            ops: vec![first, second],
            out: SVal::Op(1),
            extra_pcs: Default::default(),
        };
        let mut scratch = vec![999];
        assert_eq!(program.eval_with(&[5, 9], &mut scratch), Ok(9));
        assert_eq!(scratch, [0, 9]);
    }
}

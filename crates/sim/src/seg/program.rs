/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use super::SOp;
use super::SVal;
use super::binding;

pub struct Program<'a> {
    pub args: &'a [u32],
    pub ops: &'a [SOp],
    pub out: SVal,
}

pub trait ProgramEvaluator {
    fn evaluate(&mut self, instruction: &SOp, args: &[u32], prior: &[u64]) -> Result<u64, String>;
}

pub fn execute<E: ProgramEvaluator>(
    program: Program<'_>,
    values: &mut Vec<u64>,
    evaluator: &mut E,
) -> Result<u32, String> {
    values.clear();
    let mut index = 0;
    while index < program.ops.len() {
        let result = match evaluator.evaluate(&program.ops[index], program.args, values) {
            Ok(bits) => bits,
            Err(error) => return Err(error),
        };
        values.push(result);
        index += 1;
    }
    Ok(binding::read_sval(program.out, program.args, values) as u32)
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;
    use std::panic::catch_unwind;

    use ptxv_ir::File;
    use ptxv_ir::Op;
    use ptxv_ir::Reg;

    use super::*;
    use crate::seg::SKind;
    use crate::seg::SProg;

    #[derive(Default)]
    struct Recorder {
        calls: Vec<(u32, Vec<u32>, Vec<u64>)>,
        fail_at: Option<usize>,
        panic_at: Option<usize>,
    }

    impl ProgramEvaluator for Recorder {
        fn evaluate(&mut self, op: &SOp, args: &[u32], prior: &[u64]) -> Result<u64, String> {
            let index = self.calls.len();
            self.calls.push((op.pc, args.to_vec(), prior.to_vec()));
            assert_ne!(self.panic_at, Some(index), "callback panic");
            if self.fail_at == Some(index) {
                return Err("callback error".into());
            }
            Ok(((op.pc as u64) << 32) | (index as u64))
        }
    }

    fn operations(count: usize) -> Vec<SOp> {
        (0..count)
            .map(|index| SOp {
                pc: 17 + index as u32,
                kind: SKind::Guard {
                    pred: SVal::Const(1),
                    neg: false,
                    new: SVal::Const(100 + index as u64),
                    old: SVal::Const(0),
                },
            })
            .collect()
    }

    fn expected(count: usize) -> Vec<u64> {
        (0..count)
            .map(|index| (((17 + index) as u64) << 32) | (index as u64))
            .collect()
    }

    #[test]
    fn ordered_callbacks_observe_exact_args_and_only_completed_prefix() {
        for count in 0..=5 {
            let ops = operations(count);
            let args = [0x8000_0000, 73];
            let outputs = if count == 0 {
                vec![SVal::Arg(1), SVal::Const(u64::MAX)]
            } else {
                vec![
                    SVal::Arg(1),
                    SVal::Const(u64::MAX),
                    SVal::Op(count as u32 - 1),
                ]
            };
            for out in outputs {
                let mut recorder = Recorder::default();
                let mut values = vec![999, 888];
                let result = execute(
                    Program {
                        args: &args,
                        ops: &ops,
                        out,
                    },
                    &mut values,
                    &mut recorder,
                );
                let wanted = match out {
                    SVal::Arg(index) => args[index as usize],
                    SVal::Const(bits) => bits as u32,
                    SVal::Op(index) => expected(count)[index as usize] as u32,
                };
                assert_eq!(result, Ok(wanted));
                assert_eq!(values, expected(count));
                assert_eq!(recorder.calls.len(), count);
                for (index, (pc, observed_args, observed_prior)) in
                    recorder.calls.iter().enumerate()
                {
                    assert_eq!(*pc, 17 + index as u32);
                    assert_eq!(observed_args, &args);
                    assert_eq!(observed_prior, &expected(index));
                }
            }
        }
    }

    #[test]
    fn returned_callback_errors_preserve_prefix_and_stop_later_observations() {
        let ops = operations(4);
        for failed in 0..ops.len() {
            let mut recorder = Recorder {
                fail_at: Some(failed),
                ..Default::default()
            };
            let mut values = vec![999];
            assert_eq!(
                execute(
                    Program {
                        args: &[41],
                        ops: &ops,
                        out: SVal::Op(99)
                    },
                    &mut values,
                    &mut recorder
                ),
                Err("callback error".into())
            );
            assert_eq!(values, expected(failed));
            assert_eq!(recorder.calls.len(), failed + 1);
            assert_eq!(recorder.calls[failed].2, expected(failed));
        }
    }

    #[test]
    fn callback_panics_leave_the_same_completed_prefix() {
        let ops = operations(4);
        for failed in 0..ops.len() {
            let mut recorder = Recorder {
                panic_at: Some(failed),
                ..Default::default()
            };
            let mut values = vec![999];
            let panic = catch_unwind(AssertUnwindSafe(|| {
                execute(
                    Program {
                        args: &[],
                        ops: &ops,
                        out: SVal::Const(3),
                    },
                    &mut values,
                    &mut recorder,
                )
            }));
            assert!(panic.is_err());
            assert_eq!(values, expected(failed));
            assert_eq!(recorder.calls.len(), failed + 1);
        }
    }

    #[test]
    fn invalid_output_is_read_after_all_callbacks_and_empty_program_still_clears() {
        for count in 0..=4 {
            let ops = operations(count);
            for out in [SVal::Arg(1), SVal::Op(count as u32)] {
                let mut recorder = Recorder::default();
                let mut values = vec![999];
                let panic = catch_unwind(AssertUnwindSafe(|| {
                    execute(
                        Program {
                            args: &[1],
                            ops: &ops,
                            out,
                        },
                        &mut values,
                        &mut recorder,
                    )
                }));
                assert!(panic.is_err());
                assert_eq!(values, expected(count));
                assert_eq!(recorder.calls.len(), count);
            }
        }
    }

    #[test]
    fn actual_sprog_delegation_preserves_scratch_success_and_error_prefixes() {
        let mut program = SProg {
            args: Vec::new(),
            ops: operations(2),
            out: SVal::Op(1),
            extra_pcs: Default::default(),
        };
        let mut scratch = vec![u64::MAX, u64::MAX, u64::MAX];
        assert_eq!(program.eval_with(&[], &mut scratch), Ok(101));
        assert_eq!(scratch, [100, 101]);
        program.ops.push(SOp {
            pc: 99,
            kind: SKind::Inst {
                op: Op::Nop,
                dst: Reg {
                    file: File::R,
                    idx: 0,
                },
                regs: Vec::new(),
                cvals: Vec::new(),
            },
        });
        program.ops.extend(operations(2));
        let error = program.eval_with(&[], &mut scratch).unwrap_err();
        assert!(error.contains("not modeled in a segment"));
        assert_eq!(scratch, [100, 101]);
    }
}

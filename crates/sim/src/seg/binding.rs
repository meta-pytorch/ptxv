/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_ir::Pred;
use ptxv_ir::Reg;

use super::SVal;

pub fn read_sval(value: SVal, args: &[u32], prior: &[u64]) -> u64 {
    match value {
        SVal::Arg(index) => args[index as usize] as u64,
        SVal::Const(bits) => bits,
        SVal::Op(index) => prior[index as usize],
    }
}

pub fn register_binding(bindings: &[(Reg, SVal)], register: Reg) -> Option<SVal> {
    let mut index = 0;
    while index < bindings.len() {
        let (candidate, source) = bindings[index];
        if candidate.file == register.file && candidate.idx == register.idx {
            return Some(source);
        }
        index += 1;
    }
    None
}

pub fn predicate_binding(predicate: &Pred, bindings: &[(Reg, SVal)]) -> (Option<SVal>, bool) {
    let source = match predicate.reg {
        None => None,
        Some(register) => register_binding(bindings, register),
    };
    (source, predicate.neg)
}

#[derive(Clone, Copy)]
pub struct Inputs<'a> {
    pub args: &'a [u32],
    pub prior: &'a [u64],
}

pub fn register_bits(register: Reg, bindings: &[(Reg, SVal)], inputs: Inputs<'_>) -> u64 {
    match register_binding(bindings, register) {
        None => 0,
        Some(source) => read_sval(source, inputs.args, inputs.prior),
    }
}

pub fn read_predicate(predicate: &Pred, bindings: &[(Reg, SVal)], inputs: Inputs<'_>) -> bool {
    let (source, neg) = predicate_binding(predicate, bindings);
    let raw = match (predicate.reg, source) {
        (Some(register), Some(source)) => {
            register.pred_bit(read_sval(source, inputs.args, inputs.prior))
        }
        _ => true,
    };
    raw != neg
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;
    use std::panic::catch_unwind;

    use ptxv_ir::File;
    use ptxv_ir::Op;
    use ptxv_ir::Val;

    use super::*;
    use crate::seg::SKind;
    use crate::seg::SOp;
    use crate::seg::SProg;

    fn register(file: File, idx: u16) -> Reg {
        Reg { file, idx }
    }

    #[test]
    fn sval_reads_exact_arrays_constants_and_invalid_bounds() {
        let args = [0x8000_0000, 0x7fc0_1234];
        let prior = [u64::MAX, 0x1234_5678_9abc_def0];
        assert_eq!(read_sval(SVal::Arg(0), &args, &prior), args[0] as u64);
        assert_eq!(read_sval(SVal::Arg(1), &args, &prior), args[1] as u64);
        assert_eq!(read_sval(SVal::Op(0), &args, &prior), prior[0]);
        assert_eq!(read_sval(SVal::Op(1), &args, &prior), prior[1]);
        assert_eq!(read_sval(SVal::Const(u64::MAX), &args, &prior), u64::MAX);
        for source in [
            SVal::Arg(2),
            SVal::Arg(u16::MAX),
            SVal::Op(2),
            SVal::Op(u32::MAX),
        ] {
            assert!(catch_unwind(|| read_sval(source, &args, &prior)).is_err());
        }
    }

    #[test]
    fn first_matching_binding_preserves_file_index_and_duplicate_order() {
        let files = [File::V, File::R, File::UR, File::P, File::UP];
        for first_file in files {
            for queried_file in files {
                for first_index in [0, 1, u16::MAX] {
                    for queried_index in [0, 1, u16::MAX] {
                        let first = register(first_file, first_index);
                        let queried = register(queried_file, queried_index);
                        let bindings = [
                            (first, SVal::Arg(0)),
                            (queried, SVal::Arg(1)),
                            (queried, SVal::Const(u64::MAX)),
                        ];
                        let expected = if first == queried {
                            SVal::Arg(0)
                        } else {
                            SVal::Arg(1)
                        };
                        assert_eq!(register_binding(&bindings, queried), Some(expected));
                    }
                }
            }
        }
        assert_eq!(register_binding(&[], register(File::R, 0)), None);
    }

    #[test]
    fn register_and_predicate_defaults_negation_and_read_failures() {
        let present = register(File::P, 0);
        let missing = register(File::UP, 0);
        let bindings = [(register(File::R, 0), SVal::Arg(0)), (present, SVal::Op(0))];
        let inputs = Inputs {
            args: &[42],
            prior: &[0],
        };
        assert_eq!(register_bits(missing, &bindings, inputs), 0);
        assert_eq!(register_bits(present, &bindings, inputs), 0);
        assert_eq!(register_bits(register(File::R, 0), &bindings, inputs), 42);
        for negative in [false, true] {
            assert_eq!(
                read_predicate(
                    &Pred {
                        reg: None,
                        neg: negative
                    },
                    &bindings,
                    inputs
                ),
                !negative
            );
            assert_eq!(
                read_predicate(
                    &Pred {
                        reg: Some(missing),
                        neg: negative
                    },
                    &bindings,
                    inputs
                ),
                !negative
            );
            assert_eq!(
                read_predicate(
                    &Pred {
                        reg: Some(present),
                        neg: negative
                    },
                    &bindings,
                    inputs
                ),
                negative
            );
        }
        for file in [File::V, File::R, File::UR, File::P, File::UP] {
            for bits in [0, 1, 0x7fff_ffff, 0x8000_0000, u32::MAX] {
                let register = register(file, 0);
                let inputs = Inputs {
                    args: &[bits],
                    prior: &[bits as u64],
                };
                let expected = match file {
                    File::R | File::UR => bits & 0x8000_0000 != 0,
                    _ => bits != 0,
                };
                for source in [SVal::Arg(0), SVal::Op(0), SVal::Const(bits as u64)] {
                    for negative in [false, true] {
                        assert_eq!(
                            read_predicate(
                                &Pred {
                                    reg: Some(register),
                                    neg: negative,
                                },
                                &[(register, source)],
                                inputs,
                            ),
                            expected != negative,
                            "file={file:?}, bits={bits:#x}, source={source:?}, neg={negative}",
                        );
                    }
                }
            }
        }
        let bad = [(present, SVal::Arg(1))];
        assert!(catch_unwind(|| register_bits(present, &bad, inputs)).is_err());
        assert!(catch_unwind(|| read_predicate(
            &Pred {
                reg: Some(present),
                neg: false
            },
            &bad,
            inputs
        ))
        .is_err());
        assert!(read_predicate(&Pred::TRUE, &bad, inputs));
    }

    fn selected(regs: Vec<(Reg, SVal)>, predicate: Pred) -> SProg {
        SProg {
            args: vec![77, 99],
            ops: vec![SOp {
                pc: 0,
                kind: SKind::Inst {
                    op: Op::Sel {
                        d: register(File::R, 2),
                        a: Val::Reg(register(File::R, 0), 1),
                        b: Val::Reg(register(File::R, 1), 1),
                        p: predicate,
                    },
                    dst: register(File::R, 2),
                    regs,
                    cvals: Vec::new(),
                },
            }],
            out: SVal::Op(0),
            extra_pcs: Default::default(),
        }
    }

    #[test]
    fn actual_program_uses_first_bindings_and_keeps_predicate_namespace() {
        let predicate = Pred {
            reg: Some(register(File::P, 0)),
            neg: false,
        };
        let program = selected(
            vec![
                (register(File::R, 0), SVal::Arg(0)),
                (register(File::R, 0), SVal::Arg(1)),
                (register(File::R, 1), SVal::Arg(1)),
                (register(File::P, 0), SVal::Const(0)),
            ],
            predicate,
        );
        assert_eq!(program.eval(&[5, 0x8000_0000]).unwrap(), 0x8000_0000);
        let mut program = program;
        if let SKind::Inst { regs, .. } = &mut program.ops[0].kind {
            regs[3].1 = SVal::Const(1);
        }
        assert_eq!(program.eval(&[5, 0x8000_0000]).unwrap(), 5);
    }

    #[test]
    fn actual_program_preserves_missing_defaults_and_lazy_invalid_sources() {
        let absent = Pred {
            reg: Some(register(File::P, 0)),
            neg: false,
        };
        let program = selected(vec![(register(File::R, 1), SVal::Arg(u16::MAX))], absent);
        assert_eq!(program.eval(&[]).unwrap(), 0);
        let program = selected(
            vec![
                (register(File::R, 0), SVal::Arg(u16::MAX)),
                (register(File::R, 1), SVal::Const(u64::MAX)),
            ],
            Pred {
                neg: true,
                ..absent
            },
        );
        assert_eq!(program.eval(&[]).unwrap(), u32::MAX);
        let program = selected(vec![(register(File::R, 0), SVal::Arg(u16::MAX))], absent);
        assert!(catch_unwind(AssertUnwindSafe(|| program.eval(&[]))).is_err());
    }
}

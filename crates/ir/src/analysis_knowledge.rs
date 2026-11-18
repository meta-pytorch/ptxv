/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::Front;
use crate::Pred;
use crate::Val;
use crate::predicate_read::KnownPredicate;

/// Mutable uniform values can change when another lane runs.
pub fn value_is_thread_private(value: &Val) -> bool {
    match value {
        Val::Reg(register, _) => crate::register_owner::is_thread_private(*register),
        _ => true,
    }
}

#[inline]
pub fn reads_are_thread_private(registers: &[crate::Reg]) -> bool {
    let mut index = 0;
    while index < registers.len() {
        if !crate::register_owner::is_thread_private(registers[index]) {
            return false;
        }
        index += 1;
    }
    true
}

#[inline]
pub fn direct_reads_are_thread_private(operation: &crate::Op) -> Option<bool> {
    match operation {
        crate::Op::Mov { a, .. } | crate::Op::LdConst { base: a, .. } => {
            Some(value_is_thread_private(a))
        }
        _ => None,
    }
}

pub fn predicate<R: KnownPredicate>(reader: &mut R, front: Front, value: Pred) -> Option<bool> {
    let normalized = crate::register_owner::normalize_predicate(front, value);
    if let Some(register) = normalized.reg {
        if !crate::register_owner::is_thread_private(register) {
            return None;
        }
    }
    crate::predicate_read::evaluate(reader, front, normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::File;
    use crate::Reg;

    #[derive(Default)]
    struct Reader {
        reads: Vec<Reg>,
        value: Option<bool>,
    }

    impl KnownPredicate for Reader {
        fn read(&mut self, register: Reg) -> Option<bool> {
            self.reads.push(register);
            self.value
        }
    }

    #[test]
    fn temporal_reads_normalize_constants_and_preserve_private_knowledge() {
        for front in [Front::Ptx, Front::Sass] {
            for file in [File::V, File::R, File::UR, File::P, File::UP] {
                for idx in [0, 7, u16::MAX] {
                    let register = Reg { file, idx };
                    assert_eq!(
                        value_is_thread_private(&Val::Reg(register, 1)),
                        matches!(file, File::V | File::R | File::P)
                    );
                    for neg in [false, true] {
                        for known in [None, Some(false), Some(true)] {
                            let mut reader = Reader {
                                value: known,
                                ..Default::default()
                            };
                            let result = predicate(
                                &mut reader,
                                front,
                                Pred {
                                    reg: Some(register),
                                    neg,
                                },
                            );
                            let constant = idx == 7
                                && (file == File::UP || (file == File::P && front == Front::Sass));
                            if constant {
                                assert_eq!(result, Some(!neg));
                                assert!(reader.reads.is_empty());
                            } else if matches!(file, File::UR | File::UP) {
                                assert_eq!(result, None);
                                assert!(reader.reads.is_empty());
                            } else {
                                assert_eq!(result, known.map(|value| value != neg));
                                assert_eq!(reader.reads, [register]);
                            }
                        }
                    }
                }
            }
        }
        assert!(value_is_thread_private(&Val::Imm(3)));
    }
}

#[cfg(test)]
mod temporal_data_read_tests {
    use super::*;
    use crate::File;
    use crate::Reg;
    #[test]
    fn temporal_data_reads_require_every_register_to_be_private() {
        assert!(reads_are_thread_private(&[]));
        for file in [File::V, File::R, File::UR, File::P, File::UP] {
            let register = Reg { file, idx: 0 };
            let private = matches!(file, File::V | File::R | File::P);
            assert_eq!(reads_are_thread_private(&[register]), private);
            let first = Reg {
                file: File::R,
                idx: 1,
            };
            assert_eq!(reads_are_thread_private(&[first, register]), private);
            assert_eq!(reads_are_thread_private(&[register, first]), private);
        }
    }
    #[test]
    fn selected_exit_data_producer_rejects_shared_actual_operands_without_a_fallback() {
        for file in [File::V, File::R, File::UR, File::P, File::UP] {
            let source = Val::Reg(Reg { file, idx: 0 }, 1);
            let destination = Reg {
                file: File::R,
                idx: 2,
            };
            let expected = matches!(file, File::V | File::R | File::P);
            assert_eq!(
                direct_reads_are_thread_private(&crate::Op::Mov {
                    d: destination,
                    a: source.clone(),
                }),
                Some(expected)
            );
            assert_eq!(
                direct_reads_are_thread_private(&crate::Op::LdConst {
                    d: destination,
                    bank: 0,
                    base: source,
                    off: 0,
                    bytes: 4,
                    data: Vec::new(),
                }),
                Some(expected)
            );
        }
    }
}

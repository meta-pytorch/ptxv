/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::Front;
use crate::Pred;
use crate::Reg;

pub trait KnownPredicate {
    fn read(&mut self, register: Reg) -> Option<bool>;
}

pub fn evaluate<R: KnownPredicate>(reader: &mut R, front: Front, predicate: Pred) -> Option<bool> {
    let normalized = crate::register_owner::normalize_predicate(front, predicate);
    let known = match normalized.reg {
        None => Some(true),
        Some(register) => reader.read(register),
    };
    known.map(|value| value != normalized.neg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::File;

    struct Reader {
        value: Option<bool>,
        reads: Vec<Reg>,
    }
    impl KnownPredicate for Reader {
        fn read(&mut self, register: Reg) -> Option<bool> {
            self.reads.push(register);
            self.value
        }
    }

    #[test]
    fn constants_skip_reads_and_mutable_predicates_apply_negation_once() {
        for front in [Front::Ptx, Front::Sass] {
            for register in [
                None,
                Some(Reg {
                    file: File::P,
                    idx: 0,
                }),
                Some(Reg {
                    file: File::P,
                    idx: 7,
                }),
                Some(Reg {
                    file: File::UP,
                    idx: 7,
                }),
            ] {
                let constant = register.is_none_or(|r| {
                    r.idx == 7 && (r.file == File::UP || front == Front::Sass && r.file == File::P)
                });
                for neg in [false, true] {
                    for value in [None, Some(false), Some(true)] {
                        let mut reader = Reader {
                            value,
                            reads: Vec::new(),
                        };
                        let actual = evaluate(&mut reader, front, Pred { reg: register, neg });
                        let expected = if constant {
                            Some(!neg)
                        } else {
                            value.map(|v| v != neg)
                        };
                        assert_eq!(actual, expected);
                        assert_eq!(
                            reader.reads,
                            if constant {
                                vec![]
                            } else {
                                vec![register.unwrap()]
                            }
                        );
                    }
                }
            }
        }
    }
}

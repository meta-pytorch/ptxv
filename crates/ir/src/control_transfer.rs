/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::Front;
use crate::Op;
use crate::Pred;
use crate::Val;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbstractValue {
    Known(u64),
    UniformUnknown,
    Divergent,
}

pub trait ControlReader {
    fn predicate(&mut self, predicate: Pred) -> AbstractValue;
    fn value(&mut self, value: &Val) -> AbstractValue;
}

pub struct Context<'a> {
    pub front: Front,
    pub pc: u32,
    pub entry: usize,
    /// Absent when the kernel has no return table; an empty list has no valid targets.
    pub returns: Option<&'a [usize]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    DivergentControl,
    InvalidIndex,
    InvalidTarget,
    UnknownReturn,
    PcOverflow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Next,
    Jump(u32),
    Fork(Vec<u32>),
    Wild(Fault),
}

fn target_words(targets: &[usize]) -> Vec<u32> {
    let mut result = Vec::new();
    let mut index = 0;
    while index < targets.len() {
        result.push(targets[index] as u32);
        index += 1;
    }
    result
}

fn with_fallthrough(taken: Outcome, pc: u32) -> Outcome {
    let Some(next) = pc.checked_add(1) else {
        return Outcome::Wild(Fault::PcOverflow);
    };
    match taken {
        Outcome::Jump(target) => {
            let mut targets = Vec::new();
            targets.push(target);
            targets.push(next);
            Outcome::Fork(targets)
        }
        Outcome::Fork(mut targets) => {
            targets.push(next);
            Outcome::Fork(targets)
        }
        other => other,
    }
}

fn return_target_allowed(returns: Option<&[usize]>, target: usize) -> bool {
    match returns {
        Some(targets) => targets.contains(&target),
        None => true,
    }
}

fn unknown_return(returns: Option<&[usize]>) -> Outcome {
    match returns {
        Some(targets) => {
            if targets.is_empty() {
                Outcome::Wild(Fault::UnknownReturn)
            } else {
                Outcome::Fork(target_words(targets))
            }
        }
        None => Outcome::Wild(Fault::UnknownReturn),
    }
}

fn taken_operation<R: ControlReader>(
    context: &Context<'_>,
    operation: &Op,
    reader: &mut R,
) -> Option<Outcome> {
    let taken = match operation {
        Op::Brx { base, targets, .. } if context.front == Front::Ptx => match reader.value(base) {
            AbstractValue::Known(base) => match targets.get(base as u32 as usize) {
                Some(target) => Outcome::Jump(*target as u32),
                None => Outcome::Wild(Fault::InvalidIndex),
            },
            AbstractValue::UniformUnknown => Outcome::Fork(target_words(targets)),
            AbstractValue::Divergent => Outcome::Wild(Fault::DivergentControl),
        },
        Op::Brx { base, off, targets } => match reader.value(base) {
            AbstractValue::Known(base) => {
                let address = (base as i64).wrapping_add(*off);
                let target = (address >> 4) as usize + context.entry;
                if address & 15 != 0 || !targets.contains(&target) {
                    Outcome::Wild(Fault::InvalidTarget)
                } else {
                    Outcome::Jump(target as u32)
                }
            }
            AbstractValue::UniformUnknown => Outcome::Fork(target_words(targets)),
            AbstractValue::Divergent => Outcome::Wild(Fault::DivergentControl),
        },
        Op::Call { target } => Outcome::Jump(*target as u32),
        Op::Ret { addr } => match reader.value(addr) {
            AbstractValue::Known(address) => {
                let target = (address >> 4) as usize + context.entry;
                if return_target_allowed(context.returns, target) {
                    Outcome::Jump(target as u32)
                } else {
                    Outcome::Wild(Fault::InvalidTarget)
                }
            }
            AbstractValue::UniformUnknown => unknown_return(context.returns),
            _ => Outcome::Wild(Fault::UnknownReturn),
        },
        _ => return None,
    };
    Some(taken)
}

/// Abstract control transfer for the production lookahead's indirect branches,
/// calls and returns. Other operations retain their existing interpreter.
pub fn from_operation<R: ControlReader>(
    context: &Context<'_>,
    predicate: Pred,
    operation: &Op,
    reader: &mut R,
) -> Option<Outcome> {
    match operation {
        Op::Brx { .. } | Op::Call { .. } | Op::Ret { .. } => {
            let guard = reader.predicate(predicate);
            match guard {
                AbstractValue::Known(0) => Some(Outcome::Next),
                AbstractValue::Divergent => Some(Outcome::Wild(Fault::DivergentControl)),
                _ => match taken_operation(context, operation, reader) {
                    Some(taken) => Some(match guard {
                        AbstractValue::UniformUnknown => with_fallthrough(taken, context.pc),
                        _ => taken,
                    }),
                    None => None,
                },
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Reader {
        guard: AbstractValue,
        value: AbstractValue,
        value_reads: usize,
    }

    impl ControlReader for Reader {
        fn predicate(&mut self, _predicate: Pred) -> AbstractValue {
            self.guard
        }

        fn value(&mut self, _value: &Val) -> AbstractValue {
            self.value_reads += 1;
            self.value
        }
    }

    #[test]
    fn return_table_absence_differs_from_an_empty_target_set() {
        let operation = Op::Ret { addr: Val::Imm(0) };
        for front in [Front::Ptx, Front::Sass] {
            for (returns, known, unknown) in [
                (None, Outcome::Jump(1), Outcome::Wild(Fault::UnknownReturn)),
                (
                    Some([].as_slice()),
                    Outcome::Wild(Fault::InvalidTarget),
                    Outcome::Wild(Fault::UnknownReturn),
                ),
                (
                    Some([2].as_slice()),
                    Outcome::Wild(Fault::InvalidTarget),
                    Outcome::Fork(vec![2]),
                ),
                (
                    Some([1].as_slice()),
                    Outcome::Jump(1),
                    Outcome::Fork(vec![1]),
                ),
            ] {
                let context = Context {
                    front,
                    pc: 0,
                    entry: 1,
                    returns,
                };
                let mut reader = Reader {
                    guard: AbstractValue::Known(1),
                    value: AbstractValue::Known(0),
                    value_reads: 0,
                };
                assert_eq!(
                    from_operation(&context, Pred::TRUE, &operation, &mut reader),
                    Some(known),
                );
                reader.value = AbstractValue::UniformUnknown;
                assert_eq!(
                    from_operation(&context, Pred::TRUE, &operation, &mut reader),
                    Some(unknown),
                );
            }
        }
    }

    #[test]
    fn false_controls_skip_target_reads_and_validation() {
        let mut reader = Reader {
            guard: AbstractValue::Known(0),
            value: AbstractValue::Known(u64::MAX),
            value_reads: 0,
        };
        for front in [Front::Ptx, Front::Sass] {
            let context = Context {
                front,
                pc: u32::MAX,
                entry: usize::MAX,
                returns: None,
            };
            for operation in [
                Op::Brx {
                    base: Val::Imm(0),
                    off: i64::MAX,
                    targets: vec![],
                },
                Op::Call { target: usize::MAX },
                Op::Ret { addr: Val::Imm(0) },
            ] {
                assert_eq!(
                    from_operation(&context, Pred::TRUE, &operation, &mut reader),
                    Some(Outcome::Next),
                );
            }
        }
        assert_eq!(reader.value_reads, 0);
    }

    #[test]
    fn uniform_unknown_controls_retain_fallthrough() {
        let mut reader = Reader {
            guard: AbstractValue::UniformUnknown,
            value: AbstractValue::Known(0),
            value_reads: 0,
        };
        let context = Context {
            front: Front::Ptx,
            pc: 3,
            entry: 0,
            returns: None,
        };
        let operation = Op::Brx {
            base: Val::Imm(0),
            off: 0,
            targets: vec![9],
        };
        assert_eq!(
            from_operation(&context, Pred::TRUE, &operation, &mut reader),
            Some(Outcome::Fork(vec![9, 4])),
        );
        reader.value = AbstractValue::UniformUnknown;
        assert_eq!(
            from_operation(&context, Pred::TRUE, &operation, &mut reader),
            Some(Outcome::Fork(vec![9, 4])),
        );
    }
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::Inst;
use crate::Op;
use crate::Pred;

#[derive(Clone, Copy, Debug)]
pub struct Context<'a> {
    pub pc: usize,
    pub count: usize,
    pub cut: u8,
    pub fold: Option<bool>,
    pub returns: Option<&'a [usize]>,
}

fn append(result: &mut Vec<usize>, values: &[usize]) {
    let mut position = 0;
    while position < values.len() {
        result.push(values[position]);
        position += 1;
    }
}

fn append_next(result: &mut Vec<usize>, context: &Context<'_>) {
    if context.pc < context.count {
        let next = context.pc + 1;
        if next < context.count {
            result.push(next);
        }
    }
}

pub fn successors(op: &Op, guard: Pred, context: Context<'_>) -> Vec<usize> {
    let mut result = Vec::new();
    let always = guard.is_always();
    match op {
        Op::ClusterWait if context.cut & 1 != 0 => {}
        Op::BarSync {
            count: None,
            arrive: false,
            ..
        } if context.cut & 2 != 0 && always => {}
        Op::Bra {
            target,
            any,
            all,
            tid0_falls,
            cond,
            div,
        } => {
            if always && !*tid0_falls && cond.is_none() && div.is_none() {
                result.push(*target);
            } else if !*any && !*all && !*tid0_falls && cond.is_none() && div.is_none() {
                match context.fold {
                    Some(true) => result.push(*target),
                    Some(false) => append_next(&mut result, &context),
                    None => {
                        result.push(*target);
                        append_next(&mut result, &context);
                    }
                }
            } else {
                result.push(*target);
                append_next(&mut result, &context);
            }
        }
        Op::Call { target } => {
            result.push(*target);
            append_next(&mut result, &context);
        }
        Op::Ret { .. } => {
            match context.returns {
                Some(returns) => append(&mut result, returns),
                None => {
                    let mut destination = 0;
                    while destination < context.count {
                        result.push(destination);
                        destination += 1;
                    }
                }
            }
            if !always {
                append_next(&mut result, &context);
            }
        }
        Op::Brx { targets, .. } => {
            append(&mut result, targets);
            if !always {
                append_next(&mut result, &context);
            }
        }
        Op::Exit | Op::Trap { .. } if always => {}
        _ => append_next(&mut result, &context),
    }
    result
}

pub fn graph_successors(
    code: &[Inst],
    cut: u8,
    fold: &[Option<bool>],
    returns: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut pc = 0;
    while pc < code.len() {
        let decision = if pc < fold.len() { fold[pc] } else { None };
        let destinations = if pc < returns.len() {
            let row: &[usize] = &returns[pc];
            Some(row)
        } else {
            None
        };
        result.push(successors(
            &code[pc].op,
            code[pc].guard,
            Context {
                pc,
                count: code.len(),
                cut,
                fold: decision,
                returns: destinations,
            },
        ));
        pc += 1;
    }
    result
}

pub fn barrier_seed(operation: &Op) -> u32 {
    match operation {
        Op::BarSync {
            id: crate::Val::Imm(value),
            ..
        } if *value < 32 => 1u32 << *value,
        Op::BarSync { .. } | Op::CallAbs { .. } => u32::MAX,
        _ => 0,
    }
}

pub fn barrier_seeds(code: &[Inst]) -> Vec<u32> {
    let mut result = Vec::new();
    let mut pc = 0;
    while pc < code.len() {
        result.push(barrier_seed(&code[pc].op));
        pc += 1;
    }
    result
}

#[derive(Debug)]
pub struct GraphSource {
    pub successors: Vec<Vec<usize>>,
    pub seeds: Vec<u32>,
}

pub fn graph_and_seeds(
    code: &[Inst],
    cut: u8,
    fold: &[Option<bool>],
    returns: &[Vec<usize>],
) -> GraphSource {
    let successors = graph_successors(code, cut, fold, returns);
    let seeds = barrier_seeds(code);
    GraphSource { successors, seeds }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::File;
    use crate::Reg;
    use crate::Val;

    fn conditional() -> Pred {
        Pred {
            reg: Some(Reg {
                file: File::P,
                idx: 0,
            }),
            neg: false,
        }
    }

    fn context() -> Context<'static> {
        Context {
            pc: 0,
            count: 4,
            cut: 0,
            fold: None,
            returns: None,
        }
    }

    fn indirect() -> Op {
        Op::Brx {
            base: Val::Imm(0),
            off: 0,
            targets: vec![3],
        }
    }

    #[test]
    fn guarded_indirect_branch_preserves_fallthrough() {
        assert_eq!(
            successors(&indirect(), conditional(), context()),
            vec![3, 1]
        );
        assert_eq!(successors(&indirect(), Pred::TRUE, context()), vec![3]);
        let invalid_target = Op::Brx {
            base: Val::Imm(0),
            off: 0,
            targets: vec![usize::MAX],
        };
        assert!(successors(&invalid_target, conditional(), context()).contains(&1));
    }

    #[test]
    fn secondary_predicate_is_not_replaced_by_guard_or_fold() {
        let operation = Op::Bra {
            target: 3,
            any: false,
            tid0_falls: false,
            all: false,
            cond: Some(conditional()),
            div: None,
        };
        assert_eq!(successors(&operation, Pred::TRUE, context()), vec![3, 1]);
        assert_eq!(
            successors(
                &operation,
                Pred::TRUE,
                Context {
                    fold: Some(true),
                    ..context()
                }
            ),
            vec![3, 1]
        );
    }

    #[test]
    fn folded_plain_branch_preserves_both_decisions() {
        let operation = Op::Bra {
            target: 3,
            any: false,
            tid0_falls: false,
            all: false,
            cond: None,
            div: None,
        };
        assert_eq!(successors(&operation, conditional(), context()), vec![3, 1]);
        assert_eq!(
            successors(
                &operation,
                conditional(),
                Context {
                    fold: Some(false),
                    ..context()
                }
            ),
            vec![1]
        );
        assert_eq!(
            successors(
                &operation,
                conditional(),
                Context {
                    fold: Some(true),
                    ..context()
                }
            ),
            vec![3]
        );
    }

    #[test]
    fn collective_branch_ignores_per_thread_fold() {
        let operation = Op::Bra {
            target: 3,
            any: false,
            tid0_falls: false,
            all: true,
            cond: None,
            div: None,
        };
        for fold in [Some(false), Some(true)] {
            assert_eq!(
                successors(&operation, conditional(), Context { fold, ..context() }),
                vec![3, 1]
            );
        }
    }

    #[test]
    fn guarded_return_preserves_fallthrough() {
        let operation = Op::Ret { addr: Val::Imm(0) };
        let returns = [2];
        let context = Context {
            returns: Some(&returns),
            ..context()
        };
        assert_eq!(successors(&operation, conditional(), context), vec![2, 1]);
        assert_eq!(successors(&operation, Pred::TRUE, context), vec![2]);
    }

    #[test]
    fn graph_source_publishes_matching_edges_and_barrier_seeds() {
        let instruction = |op, guard| Inst {
            op,
            guard,
            uniform: false,
            loc: 0,
            text: String::new(),
        };
        let code = vec![
            instruction(
                Op::Brx {
                    base: Val::Imm(0),
                    off: 0,
                    targets: vec![2],
                },
                conditional(),
            ),
            instruction(
                Op::BarSync {
                    id: Val::Imm(5),
                    count: None,
                    arrive: false,
                    packed: false,
                    aligned: false,
                    red: None,
                },
                Pred::TRUE,
            ),
            instruction(Op::Ret { addr: Val::Imm(0) }, Pred::TRUE),
        ];
        let source = graph_and_seeds(&code, 0, &[], &[vec![], vec![], vec![]]);
        assert_eq!(source.successors, vec![vec![2, 1], vec![2], vec![]]);
        assert_eq!(source.seeds, vec![0, 32, 0]);
    }

    #[test]
    fn graph_rows_use_each_instruction_guard() {
        let code = [
            Inst {
                guard: conditional(),
                op: indirect(),
                uniform: false,
                loc: 0,
                text: String::new(),
            },
            Inst {
                guard: Pred::TRUE,
                op: Op::Nop,
                uniform: false,
                loc: 1,
                text: String::new(),
            },
        ];
        assert_eq!(
            graph_successors(&code, 0, &[], &[]),
            vec![vec![3, 1], vec![]]
        );
    }

    #[test]
    fn barrier_labels_cover_highest_id_and_unknowns() {
        let barrier = |id| Op::BarSync {
            id,
            count: None,
            arrive: false,
            packed: false,
            aligned: false,
            red: None,
        };
        for id in 0..32 {
            assert_eq!(barrier_seed(&barrier(Val::Imm(id))), 1u32 << id);
        }
        assert_eq!(barrier_seed(&barrier(Val::Imm(32))), u32::MAX);
        assert_eq!(
            barrier_seed(&barrier(Val::Reg(
                Reg {
                    file: File::V,
                    idx: 0
                },
                1
            ))),
            u32::MAX
        );
        assert_eq!(
            barrier_seed(&Op::CallAbs {
                target: Val::Imm(0),
                syms: vec![]
            }),
            u32::MAX
        );
        assert_eq!(barrier_seed(&Op::Nop), 0);
    }
}

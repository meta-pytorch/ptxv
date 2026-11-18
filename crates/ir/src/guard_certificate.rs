/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::Cmp;
use crate::File;
use crate::Front;
use crate::Inst;
use crate::Op;
use crate::Pred;
use crate::Reg;
use crate::Val;

#[derive(Clone, Copy, Debug)]
pub struct Fact {
    pub register: Reg,
    pub value: u64,
}

pub trait Constants {
    fn read(&self, offset: u32, bytes: u8) -> Option<u64>;
}

pub struct Evaluation<'a, C> {
    pub front: Front,
    pub instruction: &'a Inst,
    pub facts: &'a [Fact],
    pub constants: &'a C,
}

impl<C> Copy for Evaluation<'_, C> {}

impl<C> Clone for Evaluation<'_, C> {
    fn clone(&self) -> Self {
        *self
    }
}

pub struct Certificate<'a, C> {
    pub front: Front,
    pub code: &'a [Inst],
    pub states: &'a [Option<Vec<Fact>>],
    pub entry: usize,
    pub returns: &'a [Vec<usize>],
    pub constants: &'a C,
}

pub fn same_register(left: Reg, right: Reg) -> bool {
    left.file as u8 == right.file as u8 && left.idx == right.idx
}

pub fn lookup(facts: &[Fact], register: Reg) -> Option<u64> {
    let mut position = 0;
    while position < facts.len() {
        if same_register(facts[position].register, register) {
            return Some(facts[position].value);
        }
        position += 1;
    }
    None
}

struct Predicates<'a> {
    facts: &'a [Fact],
}

impl crate::predicate_read::KnownPredicate for Predicates<'_> {
    fn read(&mut self, register: Reg) -> Option<bool> {
        lookup(self.facts, register).map(|value| value != 0)
    }
}

pub fn decision(front: Front, facts: &[Fact], predicate: Pred) -> Option<bool> {
    let normalized = crate::register_owner::normalize_predicate(front, predicate);
    if let Some(register) = normalized.reg {
        if !matches!(register.file, File::P | File::UP) {
            return None;
        }
    }
    let mut reader = Predicates { facts };
    crate::analysis_knowledge::predicate(&mut reader, front, normalized)
}

pub fn value<C: Constants>(facts: &[Fact], constants: &C, operand: &Val) -> Option<u64> {
    match operand {
        Val::Imm(value) => Some(*value),
        Val::Const(offset, bytes) => constants.read(*offset, *bytes),
        Val::Reg(register, slots)
            if register.file == File::V || register.file == File::R && *slots == 1 =>
        {
            if register.file == File::R && register.idx >= 255 {
                Some(0)
            } else {
                lookup(facts, *register)
            }
        }
        _ => None,
    }
}

fn scalar_move(destination: Reg, operand: &Val) -> bool {
    if matches!(operand, Val::Const(_, 16)) {
        return false;
    }
    if destination.file == File::V {
        return true;
    }
    match operand {
        Val::Const(_, bytes) => *bytes <= 4,
        Val::Reg(_, slots) => *slots == 1,
        _ => true,
    }
}

pub fn compare(comparison: Cmp, bits: u8, x: u64, y: u64) -> Option<bool> {
    if bits == 0 || bits > 64 {
        return None;
    }
    let width = bits;
    let mask = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let (x, y) = (x & mask, y & mask);
    let shift = 64 - width;
    let (sx, sy) = (
        ((x << shift) as i64) >> shift,
        ((y << shift) as i64) >> shift,
    );
    let comparison = match comparison {
        Cmp::Eq => x == y,
        Cmp::Ne => x != y,
        Cmp::LtU => x < y,
        Cmp::LeU => x <= y,
        Cmp::GtU => x > y,
        Cmp::GeU => x >= y,
        Cmp::LtS => sx < sy,
        Cmp::LeS => sx <= sy,
        Cmp::GtS => sx > sy,
        Cmp::GeS => sx >= sy,
    };
    Some(comparison)
}

pub struct Comparison {
    pub front: Front,
    pub destination: Reg,
    pub comparison: Cmp,
    pub bits: u8,
    pub and: Pred,
    pub or: bool,
}

pub fn move_assignment<C: Constants>(
    destination: Reg,
    operand: &Val,
    facts: &[Fact],
    constants: &C,
) -> Option<Fact> {
    if !matches!(destination.file, File::V | File::R)
        || destination.file == File::R && destination.idx >= 255
    {
        return None;
    }
    if !scalar_move(destination, operand) {
        return None;
    }
    value(facts, constants, operand).map(|value| Fact {
        register: destination,
        value: if destination.file == File::V {
            value
        } else {
            value & 0xffff_ffff
        },
    })
}

pub fn comparison_assignment<C: Constants>(
    plan: Comparison,
    operands: (&Val, &Val),
    facts: &[Fact],
    constants: &C,
) -> Option<Fact> {
    if plan.destination.file != File::P || plan.front == Front::Sass && plan.destination.idx == 7 {
        return None;
    }
    match (
        value(facts, constants, operands.0),
        value(facts, constants, operands.1),
        decision(plan.front, facts, plan.and),
    ) {
        (Some(x), Some(y), Some(c)) => {
            compare(plan.comparison, plan.bits, x, y).map(|comparison| Fact {
                register: plan.destination,
                value: (if plan.or {
                    comparison || c
                } else {
                    comparison && c
                }) as u64,
            })
        }
        _ => None,
    }
}

pub fn assignment<C: Constants>(
    front: Front,
    instruction: &Inst,
    facts: &[Fact],
    constants: &C,
) -> Option<Fact> {
    match &instruction.op {
        Op::Mov { d, a } => move_assignment(*d, a, facts, constants),
        Op::Setp {
            d,
            cmp,
            bits,
            a,
            b,
            and,
            or,
            ex: None,
            xor: false,
        } => comparison_assignment(
            Comparison {
                front,
                destination: *d,
                comparison: *cmp,
                bits: *bits,
                and: *and,
                or: *or,
            },
            (a, b),
            facts,
            constants,
        ),
        _ => None,
    }
}

fn move_count(destination: Reg, operand: &Val) -> u16 {
    if matches!(operand, Val::Const(_, 16)) {
        return 4;
    }
    if destination.file == File::V {
        return 1;
    }
    match operand {
        Val::Reg(_, count) => (*count as u16).max(1),
        Val::Const(_, bytes) => (*bytes as u16 / 4).max(1),
        _ => 1,
    }
}

fn span(destination: Reg, count: u16) -> Vec<Reg> {
    let mut result = Vec::new();
    let mut offset = 0;
    while offset < count {
        result.push(Reg {
            file: destination.file,
            idx: destination.idx + offset,
        });
        offset += 1;
    }
    result
}

fn empty(facts: &[Fact]) -> bool {
    facts.is_empty()
}

/// None forgets every fact, including facts unrelated to a legacy destination list.
pub fn frame(operation: &Op) -> Option<Vec<Reg>> {
    if let Some(registers) = crate::register_effect::memory_writes(operation) {
        return Some(registers);
    }
    match operation {
        Op::Mov { d, a } => {
            let count = move_count(*d, a);
            Some(span(*d, count))
        }
        Op::Setp { d, ex: None, .. } => Some(span(*d, 1)),
        Op::Nop
        | Op::Bra { .. }
        | Op::Brx { .. }
        | Op::Exit
        | Op::Trap { .. }
        | Op::WarpSync { .. }
        | Op::Bssy { .. }
        | Op::Bsync { .. }
        | Op::Break { .. }
        | Op::BarSync { .. }
        | Op::ClusterArrive { .. }
        | Op::ClusterWait
        | Op::Fence { .. }
        | Op::MbarInit { .. }
        | Op::MbarArrive { .. }
        | Op::MbarInval { .. }
        | Op::TmaLoad { .. }
        | Op::TmaStore { .. }
        | Op::BulkCommit
        | Op::BulkWait { .. }
        | Op::CpAsync { .. }
        | Op::CpAsyncCommit
        | Op::CpAsyncWait { .. }
        | Op::CpAsyncMbarArrive { .. }
        | Op::TmemWaitLd
        | Op::TmemWaitSt => Some(Vec::new()),
        _ => None,
    }
}

pub fn same_value(left: Option<u64>, right: Option<u64>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left == right,
        (None, None) => true,
        _ => false,
    }
}

pub fn post_value<C: Constants>(context: Evaluation<'_, C>, register: Reg) -> Option<u64> {
    let old = lookup(context.facts, register);
    let guard = decision(context.front, context.facts, context.instruction.guard);
    if matches!(guard, Some(false)) {
        return old;
    }
    let writes = frame(&context.instruction.op)?;
    let mut position = 0;
    let mut written = false;
    while position < writes.len() {
        if same_register(writes[position], register) {
            written = true;
        }
        position += 1;
    }
    if !written {
        return old;
    }
    let assigned = match assignment(
        context.front,
        context.instruction,
        context.facts,
        context.constants,
    ) {
        Some(fact) if same_register(fact.register, register) => Some(fact.value),
        _ => None,
    };
    if matches!(guard, Some(true)) {
        assigned
    } else {
        None
    }
}

pub fn edge<C: Constants>(context: Evaluation<'_, C>, incoming: &[Fact]) -> bool {
    let mut position = 0;
    while position < incoming.len() {
        let fact = incoming[position];
        if !crate::register_owner::is_thread_private(fact.register)
            || !same_value(post_value(context, fact.register), Some(fact.value))
        {
            return false;
        }
        position += 1;
    }
    true
}

pub fn check_successors<C: Constants>(
    context: Evaluation<'_, C>,
    states: &[Option<Vec<Fact>>],
    successors: &[usize],
) -> bool {
    let mut position = 0;
    while position < successors.len() {
        let target = successors[position];
        if target >= states.len() {
            return false;
        }
        match &states[target] {
            Some(incoming) if edge(context, incoming) => {}
            _ => return false,
        }
        position += 1;
    }
    true
}

pub fn validate<C: Constants>(
    certificate: Certificate<'_, C>,
) -> Option<Vec<Option<Option<bool>>>> {
    if certificate.code.len() != certificate.states.len()
        || certificate.entry >= certificate.states.len()
    {
        return None;
    }
    match &certificate.states[certificate.entry] {
        Some(facts) if empty(facts) => {}
        _ => return None,
    }
    let mut decisions = Vec::new();
    let mut pc = 0;
    while pc < certificate.code.len() {
        if let Some(facts) = &certificate.states[pc] {
            let instruction = &certificate.code[pc];
            let context = Evaluation {
                front: certificate.front,
                instruction,
                facts,
                constants: certificate.constants,
            };
            let known = decision(certificate.front, facts, instruction.guard);
            let guard = if matches!(known, Some(true)) {
                Pred::TRUE
            } else {
                instruction.guard
            };
            let returns = if pc < certificate.returns.len() {
                let row: &[usize] = &certificate.returns[pc];
                Some(row)
            } else {
                None
            };
            let successors = crate::control_flow::successors(
                &instruction.op,
                guard,
                crate::control_flow::Context {
                    pc,
                    count: certificate.code.len(),
                    cut: 0,
                    fold: known,
                    returns,
                },
            );
            if !check_successors(context, certificate.states, &successors) {
                return None;
            }
            decisions.push(Some(known));
        } else {
            decisions.push(None);
        }
        pc += 1;
    }
    Some(decisions)
}

pub fn merge_grid(
    previous: Option<Option<bool>>,
    incoming: Option<Option<bool>>,
) -> Option<Option<bool>> {
    match (previous, incoming) {
        (_, None) => previous,
        (None, Some(value)) => Some(value),
        (Some(Some(old)), Some(Some(value))) if old == value => Some(Some(value)),
        _ => Some(None),
    }
}

pub fn branch_fold(instruction: &Inst, observation: Option<Option<bool>>) -> Option<bool> {
    match (&instruction.op, observation) {
        (
            Op::Bra {
                any: false,
                all: false,
                tid0_falls: false,
                cond: None,
                div: None,
                ..
            },
            Some(Some(value)),
        ) if !instruction.guard.is_always() => Some(value),
        _ => None,
    }
}

pub fn merge_rows(
    previous: &mut [Option<Option<bool>>],
    incoming: &[Option<Option<bool>>],
) -> bool {
    if previous.len() != incoming.len() {
        return false;
    }
    let mut position = 0;
    while position < previous.len() {
        previous[position] = merge_grid(previous[position], incoming[position]);
        position += 1;
    }
    true
}

pub fn publish(code: &[Inst], observations: &[Option<Option<bool>>]) -> Option<Vec<Option<bool>>> {
    if code.len() != observations.len() {
        return None;
    }
    let mut output = Vec::with_capacity(code.len());
    let mut position = 0;
    while position < code.len() {
        output.push(branch_fold(&code[position], observations[position]));
        position += 1;
    }
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoConstants;
    impl Constants for NoConstants {
        fn read(&self, _: u32, _: u8) -> Option<u64> {
            None
        }
    }

    fn instruction(op: Op, guard: Pred) -> Inst {
        Inst {
            op,
            guard,
            uniform: false,
            loc: 0,
            text: String::new(),
        }
    }

    fn r(idx: u16) -> Reg {
        Reg { file: File::R, idx }
    }
    fn p(idx: u16) -> Reg {
        Reg { file: File::P, idx }
    }

    #[test]
    fn unknown_guards_forget_equal_written_facts() {
        let facts = [Fact {
            register: r(0),
            value: 7,
        }];
        let unknown = Pred {
            reg: Some(p(2)),
            neg: false,
        };
        let inst = instruction(
            Op::Mov {
                d: r(0),
                a: Val::Imm(7),
            },
            unknown,
        );
        let context = Evaluation {
            front: Front::Sass,
            instruction: &inst,
            facts: &facts,
            constants: &NoConstants,
        };
        assert_eq!(post_value(context, r(0)), None);
        let always = instruction(inst.op.clone(), Pred::TRUE);
        assert_eq!(
            post_value(
                Evaluation {
                    instruction: &always,
                    ..context
                },
                r(0)
            ),
            Some(7)
        );
        let false_facts = [
            facts[0],
            Fact {
                register: p(2),
                value: 0,
            },
        ];
        assert_eq!(
            post_value(
                Evaluation {
                    facts: &false_facts,
                    ..context
                },
                r(0)
            ),
            Some(7)
        );
        let unrelated = instruction(
            Op::Mov {
                d: r(1),
                a: Val::Imm(7),
            },
            unknown,
        );
        assert_eq!(
            post_value(
                Evaluation {
                    instruction: &unrelated,
                    ..context
                },
                r(0)
            ),
            Some(7)
        );
        assert!(move_assignment(r(0), &Val::Reg(r(1), 1), &[], &NoConstants).is_none());
        let undefined_comparison = instruction(
            Op::Setp {
                d: p(0),
                cmp: Cmp::Eq,
                bits: 32,
                a: Val::Reg(r(1), 1),
                b: Val::Imm(0),
                and: Pred::TRUE,
                or: false,
                xor: false,
                ex: None,
            },
            Pred::TRUE,
        );
        assert!(assignment(Front::Sass, &undefined_comparison, &[], &NoConstants).is_none());
        assert_eq!(
            frame(&Op::Mov {
                d: r(0),
                a: Val::Const(0, 8)
            }),
            Some(vec![r(0), r(1)])
        );
    }

    #[test]
    fn unknown_branch_requires_the_fallthrough_state() {
        let code = [
            instruction(
                Op::Bra {
                    target: 2,
                    any: false,
                    tid0_falls: false,
                    all: false,
                    cond: None,
                    div: None,
                },
                Pred {
                    reg: Some(p(0)),
                    neg: false,
                },
            ),
            instruction(Op::Nop, Pred::TRUE),
            instruction(Op::Exit, Pred::TRUE),
        ];
        let states = [Some(vec![]), None, Some(vec![])];
        let returns = [vec![], vec![], vec![]];
        assert!(validate(Certificate {
            front: Front::Sass,
            code: &code,
            states: &states,
            entry: 0,
            returns: &returns,
            constants: &NoConstants
        })
        .is_none());
    }

    #[test]
    fn row_merge_preserves_positions_and_rejects_mismatches() {
        let mut previous = [Some(Some(true)), None, Some(None), Some(Some(false))];
        let incoming = [Some(Some(false)), Some(Some(true)), None, Some(Some(false))];
        assert!(merge_rows(&mut previous, &incoming));
        assert_eq!(
            previous,
            [Some(None), Some(Some(true)), Some(None), Some(Some(false))]
        );
        let saved = previous;
        assert!(!merge_rows(&mut previous, &incoming[..3]));
        assert_eq!(previous, saved);
        assert!(!merge_rows(&mut previous[..3], &incoming));
        assert_eq!(previous, saved);
        assert!(merge_rows(&mut [], &[]));
    }

    #[test]
    fn publication_maps_actual_code_and_observation_positions() {
        let guard = Pred {
            reg: Some(p(0)),
            neg: false,
        };
        let code = [
            instruction(
                Op::Bra {
                    target: 4,
                    any: false,
                    tid0_falls: false,
                    all: false,
                    cond: None,
                    div: None,
                },
                guard,
            ),
            instruction(
                Op::Bra {
                    target: 4,
                    any: true,
                    tid0_falls: false,
                    all: false,
                    cond: None,
                    div: None,
                },
                guard,
            ),
            instruction(Op::Nop, guard),
            instruction(
                Op::Bra {
                    target: 4,
                    any: false,
                    tid0_falls: false,
                    all: false,
                    cond: None,
                    div: None,
                },
                guard,
            ),
        ];
        let observations = [
            Some(Some(true)),
            Some(Some(true)),
            Some(Some(false)),
            Some(Some(false)),
        ];
        assert_eq!(
            publish(&code, &observations),
            Some(vec![Some(true), None, None, Some(false)])
        );
        assert!(publish(&code, &observations[..3]).is_none());
        assert!(publish(&code[..3], &observations).is_none());
        assert_eq!(publish(&[], &[]), Some(vec![]));
    }

    #[test]
    fn branch_publication_requires_plain_conditional_branch() {
        let guard = Pred {
            reg: Some(p(0)),
            neg: false,
        };
        let plain = Op::Bra {
            target: 4,
            any: false,
            tid0_falls: false,
            all: false,
            cond: None,
            div: None,
        };
        let inst = instruction(plain.clone(), guard);
        assert_eq!(branch_fold(&inst, Some(Some(true))), Some(true));
        assert_eq!(branch_fold(&inst, Some(Some(false))), Some(false));
        assert_eq!(branch_fold(&inst, Some(None)), None);
        assert_eq!(branch_fold(&inst, None), None);
        assert_eq!(
            branch_fold(&instruction(plain, Pred::TRUE), Some(Some(true))),
            None
        );
        for operation in [
            Op::Nop,
            Op::Bra {
                target: 4,
                any: false,
                tid0_falls: false,
                all: true,
                cond: None,
                div: None,
            },
            Op::Bra {
                target: 4,
                any: true,
                tid0_falls: false,
                all: false,
                cond: None,
                div: None,
            },
            Op::Bra {
                target: 4,
                any: false,
                tid0_falls: true,
                all: false,
                cond: None,
                div: None,
            },
            Op::Bra {
                target: 4,
                any: false,
                tid0_falls: false,
                all: false,
                cond: Some(guard),
                div: None,
            },
            Op::Bra {
                target: 4,
                any: false,
                tid0_falls: false,
                all: false,
                cond: None,
                div: Some(Val::Reg(r(0), 1)),
            },
        ] {
            assert_eq!(
                branch_fold(&instruction(operation, guard), Some(Some(true))),
                None
            );
        }
    }

    #[test]
    fn grid_meet_requires_every_reaching_observation_to_agree() {
        let observations = [None, Some(None), Some(Some(false)), Some(Some(true))];
        for previous in observations {
            for incoming in observations {
                let combined = merge_grid(previous, incoming);
                if let Some(Some(value)) = combined {
                    for observation in [previous, incoming].into_iter().flatten() {
                        assert_eq!(observation, Some(value));
                    }
                }
                if incoming.is_none() {
                    assert_eq!(combined, previous);
                }
            }
        }
    }

    #[test]
    fn accepted_private_certificate_and_stale_successor_rejection() {
        let code = vec![
            instruction(
                Op::Mov {
                    d: r(0),
                    a: Val::Imm(7),
                },
                Pred::TRUE,
            ),
            instruction(
                Op::Setp {
                    d: p(0),
                    cmp: Cmp::Eq,
                    bits: 32,
                    a: Val::Reg(r(0), 1),
                    b: Val::Imm(7),
                    and: Pred::TRUE,
                    or: false,
                    ex: None,
                    xor: false,
                },
                Pred::TRUE,
            ),
            instruction(
                Op::Bra {
                    target: 4,
                    any: false,
                    tid0_falls: false,
                    all: false,
                    cond: None,
                    div: None,
                },
                Pred {
                    reg: Some(p(0)),
                    neg: false,
                },
            ),
            instruction(Op::Trap { why: String::new() }, Pred::TRUE),
            instruction(Op::Exit, Pred::TRUE),
        ];
        let r0 = Fact {
            register: r(0),
            value: 7,
        };
        let p0 = Fact {
            register: p(0),
            value: 1,
        };
        let mut states = vec![
            Some(vec![]),
            Some(vec![r0]),
            Some(vec![r0, p0]),
            None,
            Some(vec![r0, p0]),
        ];
        let returns = vec![Vec::new(); code.len()];
        let accepted = validate(Certificate {
            front: Front::Sass,
            code: &code,
            states: &states,
            entry: 0,
            returns: &returns,
            constants: &NoConstants,
        })
        .unwrap();
        assert_eq!(accepted[2], Some(Some(true)));
        states[0] = Some(vec![r0]);
        assert!(validate(Certificate {
            front: Front::Sass,
            code: &code,
            states: &states,
            entry: 0,
            returns: &returns,
            constants: &NoConstants,
        })
        .is_none());
        states[0] = Some(vec![]);
        states[1] = Some(vec![Fact {
            register: r(0),
            value: 8,
        }]);
        assert!(validate(Certificate {
            front: Front::Sass,
            code: &code,
            states: &states,
            entry: 0,
            returns: &returns,
            constants: &NoConstants
        })
        .is_none());
    }

    #[test]
    fn transfer_respects_width_constants_and_unsupported_forms() {
        let facts = [];
        let upper = (1u64 << 48) | 7;
        assert_eq!(
            move_assignment(r(0), &Val::Imm(upper), &facts, &NoConstants)
                .unwrap()
                .value,
            7
        );
        let virtual_register = Reg {
            file: File::V,
            idx: 0,
        };
        assert_eq!(
            move_assignment(virtual_register, &Val::Imm(upper), &facts, &NoConstants)
                .unwrap()
                .value,
            upper
        );
        for destination in [
            r(255),
            r(u16::MAX),
            p(0),
            Reg {
                file: File::UR,
                idx: 0,
            },
        ] {
            assert!(move_assignment(destination, &Val::Imm(7), &facts, &NoConstants).is_none());
        }
        assert!(move_assignment(r(0), &Val::Reg(r(2), 2), &facts, &NoConstants).is_none());
        assert!(
            move_assignment(virtual_register, &Val::Const(0, 16), &facts, &NoConstants).is_none()
        );
        assert_eq!(value(&facts, &NoConstants, &Val::Reg(r(255), 1)), Some(0));
        assert_eq!(
            decision(
                Front::Sass,
                &facts,
                Pred {
                    reg: Some(p(7)),
                    neg: true
                }
            ),
            Some(false)
        );
        assert_eq!(
            decision(
                Front::Ptx,
                &facts,
                Pred {
                    reg: Some(p(7)),
                    neg: true
                }
            ),
            None
        );
        assert_eq!(
            decision(
                Front::Sass,
                &[Fact {
                    register: r(0),
                    value: 1
                }],
                Pred {
                    reg: Some(r(0)),
                    neg: false
                }
            ),
            None
        );
        assert_eq!(compare(Cmp::LtS, 8, 0xff, 0), Some(true));
        assert_eq!(compare(Cmp::LtU, 8, 0xff, 0), Some(false));
        assert_eq!(compare(Cmp::LtS, 64, 1u64 << 63, 0), Some(true));
        let second_word = [Fact {
            register: r(1),
            value: 7,
        }];
        let multiword = instruction(
            Op::Mov {
                d: r(0),
                a: Val::Reg(r(2), 2),
            },
            Pred::TRUE,
        );
        assert_eq!(
            post_value(
                Evaluation {
                    front: Front::Sass,
                    instruction: &multiword,
                    facts: &second_word,
                    constants: &NoConstants
                },
                r(1)
            ),
            None
        );
        let uniform = Fact {
            register: Reg {
                file: File::UR,
                idx: 0,
            },
            value: 7,
        };
        let nop = instruction(Op::Nop, Pred::TRUE);
        assert!(!edge(
            Evaluation {
                front: Front::Sass,
                instruction: &nop,
                facts: &[uniform],
                constants: &NoConstants
            },
            &[uniform]
        ));
        let unsupported = instruction(
            Op::Brev {
                d: r(1),
                a: Val::Imm(0),
            },
            Pred::TRUE,
        );
        assert_eq!(
            post_value(
                Evaluation {
                    front: Front::Sass,
                    instruction: &unsupported,
                    facts: &second_word,
                    constants: &NoConstants
                },
                r(1)
            ),
            None
        );
        for bits in [0, 65, u8::MAX] {
            assert!(compare(Cmp::Eq, bits, 0, 0).is_none());
        }
        let operation = instruction(
            Op::Setp {
                d: p(7),
                cmp: Cmp::Eq,
                bits: 32,
                a: Val::Imm(0),
                b: Val::Imm(0),
                and: Pred::TRUE,
                or: false,
                ex: None,
                xor: false,
            },
            Pred::TRUE,
        );
        assert!(assignment(Front::Sass, &operation, &facts, &NoConstants).is_none());
        let mut secondary = operation.clone();
        if let Op::Setp { ex, .. } = &mut secondary.op {
            *ex = Some(Pred {
                reg: Some(p(1)),
                neg: false,
            });
        }
        assert!(frame(&secondary.op).is_none());
    }
}

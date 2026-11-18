/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::checked_execution;
use crate::execution_core;
use crate::symbolic::Metadata;
use crate::value_flow::Action;
use crate::value_flow_dependencies::Transfer;
use crate::value_flow_dependencies::{self as dependencies};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    pub concrete: checked_execution::State,
    pub dependencies: dependencies::State,
}

fn initial_registers(registers: &crate::value_flow::Registers) -> Option<dependencies::Registers> {
    let predicate_count = match registers.raw.preds.len().checked_mul(64) {
        Some(count) => count,
        None => return None,
    };
    let mut virtuals = Vec::new();
    let mut index = 0;
    while index < registers.raw.regs.len() {
        virtuals.push(dependencies::metadata(true, false));
        index += 1;
    }
    let mut predicates = Vec::new();
    index = 0;
    while index < predicate_count {
        predicates.push(dependencies::metadata(true, false));
        index += 1;
    }
    Some(dependencies::Registers {
        virtuals,
        predicates,
    })
}

fn initial_memory(memory: &execution_core::AllocationBytes) -> Vec<Metadata> {
    let mut result = Vec::new();
    let mut index = 0;
    while index < memory.defined.len() {
        result.push(dependencies::metadata(true, memory.defined[index]));
        index += 1;
    }
    result
}

pub fn initial_dependencies(state: &execution_core::State) -> Option<dependencies::State> {
    let mut threads = Vec::new();
    let mut index = 0;
    while index < state.threads.len() {
        let registers = match initial_registers(&state.threads[index].registers) {
            Some(registers) => registers,
            None => return None,
        };
        threads.push(registers);
        index += 1;
    }
    let mut memory = Vec::new();
    index = 0;
    while index < state.allocations.len() {
        memory.push(initial_memory(&state.allocations[index]));
        index += 1;
    }
    Some(dependencies::State { threads, memory })
}

pub fn initial(program: &execution_core::Program) -> Option<State> {
    let concrete = match checked_execution::initial(program) {
        Some(concrete) => concrete,
        None => return None,
    };
    let dependencies = match initial_dependencies(&concrete.execution) {
        Some(dependencies) => dependencies,
        None => return None,
    };
    Some(State {
        concrete,
        dependencies,
    })
}

fn loaded_metadata(
    memory: &[Vec<Metadata>],
    access: execution_core::ResolvedAccess,
) -> Option<Vec<Metadata>> {
    if access.allocation >= memory.len() {
        return None;
    }
    let bytes = &memory[access.allocation];
    if access.offset > bytes.len() || access.width > bytes.len() - access.offset {
        return None;
    }
    let mut result = Vec::new();
    let mut index = 0;
    while index < access.width {
        result.push(bytes[access.offset + index]);
        index += 1;
    }
    Some(result)
}

fn apply_store(
    memory: &mut [Vec<Metadata>],
    access: execution_core::ResolvedAccess,
    metadata: Metadata,
) -> bool {
    if access.allocation >= memory.len() {
        return false;
    }
    let bytes = &mut memory[access.allocation];
    if access.offset > bytes.len() || access.width > bytes.len() - access.offset {
        return false;
    }
    let mut index = 0;
    while index < access.width {
        bytes[access.offset + index] = metadata;
        index += 1;
    }
    true
}

fn apply_transfer(
    program: &execution_core::Program,
    state: &execution_core::State,
    shadow: &dependencies::State,
    actor: usize,
    action: Action,
    transfer: Transfer,
) -> Option<dependencies::State> {
    let mut after = dependencies::snapshot_state(shadow);
    match transfer {
        Transfer::None => (),
        Transfer::Virtual {
            destination,
            metadata,
        } => {
            if destination as usize >= after.threads[actor].virtuals.len() {
                return None;
            }
            after.threads[actor].virtuals[destination as usize] = metadata;
        }
        Transfer::Predicate {
            destination,
            metadata,
        } => {
            if destination as usize >= after.threads[actor].predicates.len() {
                return None;
            }
            after.threads[actor].predicates[destination as usize] = metadata;
        }
        Transfer::Store { metadata } => {
            let (pointer, bytes) = match action {
                Action::Store { pointer, bytes, .. } => (pointer, bytes),
                _ => return None,
            };
            let access = match execution_core::resolve_access(program, state, pointer, bytes, true)
            {
                Ok(access) => access,
                Err(_) => return None,
            };
            if !apply_store(&mut after.memory, access, metadata) {
                return None;
            }
        }
    }
    Some(after)
}

fn action_metadata(
    program: &execution_core::Program,
    state: &execution_core::State,
    shadow: &dependencies::State,
    action: Action,
) -> Option<Vec<Metadata>> {
    match action {
        Action::Load { pointer, bytes, .. } => {
            let access = match execution_core::resolve_access(program, state, pointer, bytes, false)
            {
                Ok(access) => access,
                Err(_) => return None,
            };
            loaded_metadata(&shadow.memory, access)
        }
        _ => Some(Vec::new()),
    }
}

pub fn advance_dependencies(
    program: &execution_core::Program,
    state: &execution_core::State,
    shadow: &dependencies::State,
    actor: usize,
) -> Option<dependencies::State> {
    let commit = match execution_core::prepare(program, state, actor) {
        Ok(commit) => commit,
        Err(_) => return None,
    };
    if actor >= shadow.threads.len() {
        return None;
    }
    let instruction = &program.code[state.threads[actor].pc as usize];
    let loaded = match action_metadata(program, state, shadow, commit.action) {
        Some(loaded) => loaded,
        None => return None,
    };
    let transfer = match dependencies::transfer(
        instruction,
        &commit.action,
        &shadow.threads[actor],
        &loaded,
    ) {
        Ok(transfer) => transfer,
        Err(_) => return None,
    };
    apply_transfer(program, state, shadow, actor, commit.action, transfer)
}

pub fn step(program: &execution_core::Program, state: &State, actor: usize) -> Option<State> {
    let dependencies = match advance_dependencies(
        program,
        &state.concrete.execution,
        &state.dependencies,
        actor,
    ) {
        Some(dependencies) => dependencies,
        None => return None,
    };
    let concrete = match checked_execution::step(program, &state.concrete, actor) {
        Some(concrete) => concrete,
        None => return None,
    };
    Some(State {
        concrete,
        dependencies,
    })
}

pub fn state_equal(left: &State, right: &State) -> bool {
    execution_core::state_equal(&left.concrete.execution, &right.concrete.execution)
        && left.concrete.monitor == right.concrete.monitor
        && dependencies::state_equal(&left.dependencies, &right.dependencies)
}

pub fn state_present(states: &[State], state: &State) -> bool {
    let mut index = 0;
    while index < states.len() {
        if state_equal(&states[index], state) {
            return true;
        }
        index += 1;
    }
    false
}

fn state_closed(program: &execution_core::Program, certificate: &[State], state: &State) -> bool {
    if !checked_execution::good(program, &state.concrete) {
        return false;
    }
    let mut actor = 0;
    while actor < state.concrete.execution.threads.len() {
        if execution_core::running(&state.concrete.execution, actor) {
            let next = match step(program, state, actor) {
                Some(next) => next,
                None => return false,
            };
            if !state_present(certificate, &next) {
                return false;
            }
        }
        actor += 1;
    }
    true
}

pub fn concrete_certificate(certificate: &[State]) -> Vec<checked_execution::State> {
    let mut result = Vec::new();
    let mut index = 0;
    while index < certificate.len() {
        result.push(checked_execution::snapshot_state(
            &certificate[index].concrete,
        ));
        index += 1;
    }
    result
}

pub fn verify(program: &execution_core::Program, certificate: &[State]) -> bool {
    let initial = match initial(program) {
        Some(initial) => initial,
        None => return false,
    };
    if !state_present(certificate, &initial) {
        return false;
    }
    let concrete = concrete_certificate(certificate);
    if !checked_execution::verify(program, &concrete) {
        return false;
    }
    let mut index = 0;
    while index < certificate.len() {
        if !state_closed(program, certificate, &certificate[index]) {
            return false;
        }
        index += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_core::AllocationBytes;
    use crate::execution_core::Program;
    use crate::execution_core::ThreadState;
    use crate::machine_registers::LaneRegisters;
    use crate::value_flow::Address;
    use crate::value_flow::AllocationLayout;
    use crate::value_flow::Comparison;
    use crate::value_flow::Guard;
    use crate::value_flow::GuardedInstruction;
    use crate::value_flow::Instruction;
    use crate::value_flow::Operand;
    use crate::value_flow::Pointer;
    use crate::value_flow::Registers;
    use crate::value_flow::ThreadIds;

    fn instruction(instruction: Instruction) -> GuardedInstruction {
        GuardedInstruction {
            guard: Guard {
                predicate: None,
                negated: false,
            },
            instruction,
        }
    }

    fn address(offset: u64) -> Address {
        Address {
            base: Operand::Pointer(Pointer {
                allocation: 0,
                offset,
            }),
            index: Operand::Immediate(0),
            offset: 0,
        }
    }

    fn program(code: Vec<GuardedInstruction>) -> Program {
        Program {
            code,
            layouts: vec![AllocationLayout {
                base: 4096,
                bytes: 16,
                alignment: 8,
            }],
            initial: execution_core::State {
                threads: vec![ThreadState {
                    pc: 0,
                    running: true,
                    registers: Registers {
                        raw: LaneRegisters {
                            regs: vec![0; 4],
                            preds: vec![0],
                            pdef: vec![0],
                            written: vec![0],
                            vund: vec![],
                        },
                        pointers: vec![None; 4],
                    },
                    ids: ThreadIds {
                        tid: [0, 0, 0],
                        ctaid: [0, 0, 0],
                        ntid: [1, 1, 1],
                        nctaid: [1, 1, 1],
                        lane: 0,
                    },
                }],
                allocations: vec![AllocationBytes {
                    bytes: vec![0; 16],
                    defined: vec![true; 16],
                    live: true,
                }],
            },
            outputs: vec![0],
        }
    }

    #[test]
    fn a_captured_false_guard_does_not_establish_input_universal_safety() {
        let program = program(vec![
            instruction(Instruction::Load {
                destination: 0,
                destination_bits: 32,
                bytes: 4,
                signed: false,
                address: address(0),
            }),
            instruction(Instruction::Compare {
                bits: 32,
                destination: 0,
                comparison: Comparison::NotEqual,
                left: Operand::Virtual(0),
                right: Operand::Immediate(0),
            }),
            GuardedInstruction {
                guard: Guard {
                    predicate: Some(0),
                    negated: false,
                },
                instruction: Instruction::Store {
                    bytes: 4,
                    address: address(16),
                    source: Operand::Immediate(1),
                },
            },
            instruction(Instruction::Exit),
        ]);
        assert!(checked_execution::check(&program, 100).status.is_safe());
        let start = initial(&program).unwrap();
        let loaded = step(&program, &start, 0).unwrap();
        let compared = step(&program, &loaded, 0).unwrap();
        assert!(compared.dependencies.threads[0].predicates[0].tainted);
        assert!(step(&program, &compared, 0).is_none());
    }

    #[test]
    fn a_constant_overwrite_clears_every_written_byte_dependency() {
        let program = program(vec![
            instruction(Instruction::Store {
                bytes: 8,
                address: address(0),
                source: Operand::Immediate(0),
            }),
            instruction(Instruction::Load {
                destination: 0,
                destination_bits: 64,
                bytes: 8,
                signed: false,
                address: address(0),
            }),
            instruction(Instruction::Exit),
        ]);
        let start = initial(&program).unwrap();
        let stored = step(&program, &start, 0).unwrap();
        assert!(
            stored.dependencies.memory[0][..8]
                .iter()
                .all(|m| !m.tainted && m.defined)
        );
        assert!(
            stored.dependencies.memory[0][8..]
                .iter()
                .all(|m| m.tainted && m.defined)
        );
        let loaded = step(&program, &stored, 0).unwrap();
        assert!(!loaded.dependencies.threads[0].virtuals[0].tainted);
        let ended = step(&program, &loaded, 0).unwrap();
        let certificate = vec![start, stored, loaded, ended];
        assert!(verify(&program, &certificate));
        assert!(!verify(&program, &certificate[..3]));
        let mut forged = certificate.clone();
        forged[2].dependencies.threads[0].virtuals[0].tainted = true;
        assert!(!verify(&program, &forged));
    }

    #[test]
    fn every_predicate_bit_has_independent_metadata() {
        for predicate in [7, 65] {
            let mut program = program(vec![
                instruction(Instruction::Compare {
                    bits: 32,
                    destination: predicate,
                    comparison: Comparison::Equal,
                    left: Operand::Immediate(0),
                    right: Operand::Immediate(1),
                }),
                GuardedInstruction {
                    guard: Guard {
                        predicate: Some(predicate),
                        negated: false,
                    },
                    instruction: Instruction::Store {
                        bytes: 4,
                        address: address(16),
                        source: Operand::Immediate(1),
                    },
                },
                instruction(Instruction::Exit),
            ]);
            let words = predicate as usize / 64 + 1;
            program.initial.threads[0].registers.raw.preds = vec![0; words];
            program.initial.threads[0].registers.raw.pdef = vec![0; words];
            let start = initial(&program).unwrap();
            let compared = step(&program, &start, 0).unwrap();
            assert!(!compared.dependencies.threads[0].predicates[predicate as usize].tainted);
            let skipped = step(&program, &compared, 0).unwrap();
            let ended = step(&program, &skipped, 0).unwrap();
            assert!(verify(&program, &[start, compared, skipped, ended]));
        }
    }

    #[test]
    fn taint_propagates_through_shared_memory_between_actors() {
        let mut program = program(vec![
            instruction(Instruction::Load {
                destination: 0,
                destination_bits: 32,
                bytes: 4,
                signed: false,
                address: address(0),
            }),
            instruction(Instruction::Store {
                bytes: 4,
                address: address(8),
                source: Operand::Virtual(0),
            }),
            instruction(Instruction::Load {
                destination: 1,
                destination_bits: 32,
                bytes: 4,
                signed: false,
                address: address(8),
            }),
            instruction(Instruction::Exit),
        ]);
        program
            .initial
            .threads
            .push(program.initial.threads[0].clone());
        let start = initial(&program).unwrap();
        let loaded = step(&program, &start, 0).unwrap();
        let stored = step(&program, &loaded, 0).unwrap();
        assert!(
            stored.dependencies.memory[0][8..12]
                .iter()
                .all(|m| m.tainted)
        );
        let one_loaded = step(&program, &stored, 1).unwrap();
        let one_stored = step(&program, &one_loaded, 1).unwrap();
        let one_read = step(&program, &one_stored, 1).unwrap();
        assert!(one_read.dependencies.threads[1].virtuals[1].tainted);
    }
}

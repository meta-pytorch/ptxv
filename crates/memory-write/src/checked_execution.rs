/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::execution_core;
use crate::execution_core::Program;
use crate::memory_safety;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    pub execution: execution_core::State,
    pub monitor: memory_safety::State,
}

pub fn snapshot_state(state: &State) -> State {
    State {
        execution: execution_core::snapshot_state(&state.execution),
        monitor: state.monitor.clone(),
    }
}

fn uniform_defined(flags: &[bool]) -> Option<bool> {
    if flags.is_empty() {
        return None;
    }
    let defined = flags[0];
    let mut index = 1;
    while index < flags.len() {
        if flags[index] != defined {
            return None;
        }
        index += 1;
    }
    Some(defined)
}

fn monitor_allocation(
    layout: &crate::value_flow::AllocationLayout,
    memory: &execution_core::AllocationBytes,
) -> Option<memory_safety::Allocation> {
    let defined = match uniform_defined(&memory.defined) {
        Some(defined) => defined,
        None => return None,
    };
    Some(memory_safety::Allocation {
        base: layout.base,
        bytes: layout.bytes,
        initially_defined: defined,
    })
}

fn monitor_allocations(program: &Program) -> Option<Vec<memory_safety::Allocation>> {
    let mut allocations = Vec::new();
    let mut index = 0;
    while index < program.layouts.len() {
        let allocation = match monitor_allocation(
            &program.layouts[index],
            &program.initial.allocations[index],
        ) {
            Some(allocation) => allocation,
            None => return None,
        };
        allocations.push(allocation);
        index += 1;
    }
    Some(allocations)
}

fn monitor_threads(count: usize) -> Vec<Vec<memory_safety::Instruction>> {
    let mut threads = Vec::new();
    let mut index = 0;
    while index < count {
        threads.push(Vec::new());
        index += 1;
    }
    threads
}

pub fn monitor_program(program: &Program) -> Option<memory_safety::Program> {
    if !execution_core::layout_valid(program, &program.initial) {
        return None;
    }
    let allocations = match monitor_allocations(program) {
        Some(allocations) => allocations,
        None => return None,
    };
    Some(memory_safety::Program {
        allocations,
        threads: monitor_threads(program.initial.threads.len()),
    })
}

pub fn initial(program: &Program) -> Option<State> {
    let execution = match execution_core::initial_state(program) {
        Ok(state) => state,
        Err(_) => return None,
    };
    let frame = match monitor_program(program) {
        Some(frame) => frame,
        None => return None,
    };
    let monitor = match memory_safety::initial(&frame) {
        Some(monitor) => monitor,
        None => return None,
    };
    Some(State { execution, monitor })
}

pub fn good(program: &Program, state: &State) -> bool {
    !state.monitor.bad
        && (!execution_core::terminal(&state.execution)
            || match execution_core::outputs(&state.execution, &program.outputs) {
                Ok(_) => true,
                Err(_) => false,
            })
}

pub fn step(program: &Program, state: &State, actor: usize) -> Option<State> {
    let mut execution = execution_core::snapshot_state(&state.execution);
    let event = match execution_core::execute(program, &mut execution, actor) {
        Ok(event) => event,
        Err(_) => return None,
    };
    let frame = match monitor_program(program) {
        Some(frame) => frame,
        None => return None,
    };
    let monitor = memory_safety::update_memory(&frame, &state.monitor, event);
    Some(State { execution, monitor })
}

pub fn successors(program: &Program, state: &State) -> Option<Vec<State>> {
    let mut result = Vec::new();
    let mut actor = 0;
    while actor < state.execution.threads.len() {
        if execution_core::running(&state.execution, actor) {
            let next = match step(program, state, actor) {
                Some(next) => next,
                None => return None,
            };
            result.push(next);
        }
        actor += 1;
    }
    Some(result)
}

pub fn state_present(states: &[State], state: &State) -> bool {
    let mut index = 0;
    while index < states.len() {
        if execution_core::state_equal(&states[index].execution, &state.execution)
            && states[index].monitor == state.monitor
        {
            return true;
        }
        index += 1;
    }
    false
}

fn state_closed(program: &Program, certificate: &[State], state: &State) -> bool {
    if !good(program, state) {
        return false;
    }
    let Some(next) = successors(program, state) else {
        return false;
    };
    let mut index = 0;
    while index < next.len() {
        if !state_present(certificate, &next[index]) {
            return false;
        }
        index += 1;
    }
    true
}

pub fn verify(program: &Program, certificate: &[State]) -> bool {
    let Some(start) = initial(program) else {
        return false;
    };
    if !state_present(certificate, &start) {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Safe,
    Unsafe,
    Inconclusive,
}

impl Status {
    pub fn is_safe(self) -> bool {
        matches!(self, Self::Safe)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Report {
    pub status: Status,
    pub states: usize,
}

enum Candidate {
    Complete(Vec<State>),
    Unsafe(usize),
    Inconclusive(usize),
}

fn append_next(next: &[State], states: &mut Vec<State>, state_cap: usize) -> Option<Candidate> {
    let mut index = 0;
    while index < next.len() {
        if !state_present(states, &next[index]) {
            if states.len() >= state_cap {
                return Some(Candidate::Inconclusive(states.len()));
            }
            states.push(snapshot_state(&next[index]));
        }
        index += 1;
    }
    None
}

fn candidate(program: &Program, state_cap: usize) -> Candidate {
    if state_cap == 0 {
        return Candidate::Inconclusive(0);
    }
    let Some(start) = initial(program) else {
        return Candidate::Inconclusive(0);
    };
    let mut states = Vec::new();
    states.push(start);
    let mut cursor = 0;
    while cursor < states.len() {
        let state = &states[cursor];
        if !good(program, state) {
            return Candidate::Unsafe(states.len());
        }
        let Some(next) = successors(program, state) else {
            return Candidate::Inconclusive(states.len());
        };
        if let Some(stop) = append_next(&next, &mut states, state_cap) {
            return stop;
        }
        cursor += 1;
    }
    Candidate::Complete(states)
}

pub fn check(program: &Program, state_cap: usize) -> Report {
    match candidate(program, state_cap) {
        Candidate::Complete(certificate) => Report {
            status: if verify(program, &certificate) {
                Status::Safe
            } else {
                Status::Inconclusive
            },
            states: certificate.len(),
        },
        Candidate::Unsafe(states) => Report {
            status: Status::Unsafe,
            states,
        },
        Candidate::Inconclusive(states) => Report {
            status: Status::Inconclusive,
            states,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_core::AllocationBytes;
    use crate::execution_core::ThreadState;
    use crate::machine_registers::LaneRegisters;
    use crate::value_flow::Address;
    use crate::value_flow::AllocationLayout;
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

    fn program(code: Vec<GuardedInstruction>, threads: usize) -> Program {
        let thread = ThreadState {
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
                ntid: [threads as u32, 1, 1],
                nctaid: [1, 1, 1],
                lane: 0,
            },
        };
        Program {
            code,
            layouts: vec![AllocationLayout {
                base: 4096,
                bytes: 16,
                alignment: 8,
            }],
            initial: execution_core::State {
                threads: vec![thread; threads],
                allocations: vec![AllocationBytes {
                    bytes: vec![0; 16],
                    defined: vec![true; 16],
                    live: true,
                }],
            },
            outputs: vec![0],
        }
    }

    fn load(offset: u64) -> GuardedInstruction {
        instruction(Instruction::Load {
            destination: 0,
            destination_bits: 32,
            bytes: 4,
            signed: false,
            address: address(offset),
        })
    }

    #[test]
    fn safe_requires_full_closure_and_cap_is_inconclusive() {
        let program = program(vec![load(0), instruction(Instruction::Exit)], 1);
        assert_eq!(check(&program, 0).status, Status::Inconclusive);
        assert_eq!(check(&program, 2).status, Status::Inconclusive);
        assert_eq!(check(&program, 3).status, Status::Safe);
        let initial = initial(&program).unwrap();
        assert!(!verify(&program, &[initial]));
        assert!(!verify(&program, &[]));
    }

    #[test]
    fn malformed_running_pc_and_undefined_load_cannot_disappear() {
        let mut program = program(vec![load(0)], 1);
        assert_eq!(check(&program, 100).status, Status::Inconclusive);
        program.initial.allocations[0].defined.fill(false);
        assert_eq!(check(&program, 100).status, Status::Inconclusive);
        program.initial.threads[0].pc = 1;
        assert!(initial(&program).is_none());
    }

    #[test]
    fn every_interleaving_checks_same_value_store_conflicts() {
        let store = instruction(Instruction::Store {
            bytes: 4,
            address: address(0),
            source: Operand::Immediate(0),
        });
        let program = program(vec![store, instruction(Instruction::Exit)], 2);
        assert_eq!(check(&program, 100).status, Status::Unsafe);
        let start = initial(&program).unwrap();
        let first = step(&program, &start, 0).unwrap();
        let racy = step(&program, &first, 1).unwrap();
        assert!(racy.monitor.bad);
        assert!(!verify(&program, &[start, first, racy]));
    }

    #[test]
    fn terminal_outputs_and_mixed_initial_definedness_are_checked() {
        let mut program = program(vec![instruction(Instruction::Exit)], 1);
        program.initial.allocations[0].defined.fill(false);
        assert_eq!(check(&program, 100).status, Status::Unsafe);
        program.initial.allocations[0].defined[0] = true;
        assert!(initial(&program).is_none());
        program.initial.allocations[0].defined.fill(true);
        program.outputs.push(1);
        assert_eq!(check(&program, 100).status, Status::Unsafe);
    }

    #[test]
    fn forged_initial_certificate_does_not_supply_entry() {
        let program = program(vec![load(0), instruction(Instruction::Exit)], 1);
        let mut forged = initial(&program).unwrap();
        forged.execution.threads[0].running = false;
        assert!(good(&program, &forged));
        assert_eq!(successors(&program, &forged), Some(vec![]));
        assert!(!verify(&program, &[forged]));
    }

    #[test]
    fn finite_register_loop_initializes_every_output_byte() {
        use crate::value_flow::BinaryOperation;
        use crate::value_flow::Comparison;

        let code = vec![
            instruction(Instruction::Move {
                bits: 32,
                destination: 0,
                source: Operand::Immediate(0),
            }),
            instruction(Instruction::ConvertUnsigned {
                destination: 1,
                source: Operand::Virtual(0),
                from_bits: 32,
                to_bits: 64,
            }),
            instruction(Instruction::Store {
                bytes: 4,
                address: Address {
                    base: Operand::Pointer(Pointer {
                        allocation: 0,
                        offset: 0,
                    }),
                    index: Operand::Virtual(1),
                    offset: 0,
                },
                source: Operand::Virtual(0),
            }),
            instruction(Instruction::Binary {
                bits: 32,
                destination: 0,
                operation: BinaryOperation::Add,
                left: Operand::Virtual(0),
                right: Operand::Immediate(4),
            }),
            instruction(Instruction::Compare {
                bits: 32,
                destination: 0,
                comparison: Comparison::LessUnsigned,
                left: Operand::Virtual(0),
                right: Operand::Immediate(16),
            }),
            GuardedInstruction {
                guard: Guard {
                    predicate: Some(0),
                    negated: false,
                },
                instruction: Instruction::Branch { target: 1 },
            },
            instruction(Instruction::Exit),
        ];
        let mut program = program(code, 1);
        program.initial.allocations[0].defined.fill(false);
        assert_eq!(check(&program, 100).status, Status::Safe);
        program.code[4] = instruction(Instruction::Compare {
            bits: 32,
            destination: 0,
            comparison: Comparison::LessUnsigned,
            left: Operand::Virtual(0),
            right: Operand::Immediate(20),
        });
        assert_ne!(check(&program, 100).status, Status::Safe);
    }

    #[test]
    fn thread_id_indexed_stores_cover_all_schedules() {
        use crate::value_flow::SpecialRegister;

        let code = vec![
            instruction(Instruction::MultiplyWide {
                destination: 0,
                left: Operand::Special(SpecialRegister::Thread(0)),
                right: Operand::Immediate(4),
            }),
            instruction(Instruction::Store {
                bytes: 4,
                address: Address {
                    base: Operand::Pointer(Pointer {
                        allocation: 0,
                        offset: 0,
                    }),
                    index: Operand::Virtual(0),
                    offset: 0,
                },
                source: Operand::Special(SpecialRegister::Thread(0)),
            }),
            instruction(Instruction::Exit),
        ];
        let mut program = program(code, 4);
        program.initial.allocations[0].defined.fill(false);
        for (index, thread) in program.initial.threads.iter_mut().enumerate() {
            thread.ids.tid[0] = index as u32;
            thread.ids.lane = index as u32;
        }
        assert_eq!(check(&program, 100_000).status, Status::Safe);
        program.initial.threads[3].ids.tid[0] = 0;
        assert_eq!(check(&program, 100_000).status, Status::Unsafe);
    }
}

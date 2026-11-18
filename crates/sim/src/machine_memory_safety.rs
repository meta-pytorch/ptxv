/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_memory_write::execution_core;
use ptxv_memory_write::execution_core::Fault;
use ptxv_memory_write::execution_core::Program;
use ptxv_memory_write::execution_core::State;
use ptxv_memory_write::memory_safety::Event;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrictMachine {
    program: Program,
    state: State,
    fault: Option<Fault>,
}

/// The strict execution variant has no legacy state or mutable legacy view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionMachine<L> {
    Strict(StrictMachine),
    Legacy(L),
    Fault(Fault),
}

impl<L> ExecutionMachine<L> {
    pub fn strict(program: Program) -> Result<Self, Fault> {
        let state = match execution_core::initial_state(&program) {
            Ok(state) => state,
            Err(fault) => return Err(fault),
        };
        Ok(Self::Strict(StrictMachine {
            program,
            state,
            fault: None,
        }))
    }

    pub fn state(&self) -> Option<&State> {
        match self {
            Self::Strict(machine) => Some(&machine.state),
            Self::Legacy(_) | Self::Fault(_) => None,
        }
    }

    pub fn program(&self) -> Option<&Program> {
        match self {
            Self::Strict(machine) => Some(&machine.program),
            Self::Legacy(_) | Self::Fault(_) => None,
        }
    }

    pub fn fault(&self) -> Option<Fault> {
        match self {
            Self::Strict(machine) => machine.fault,
            Self::Legacy(_) => Some(Fault::WrongMode),
            Self::Fault(fault) => Some(*fault),
        }
    }

    pub fn step_owned<F>(self, actor: usize, legacy: F) -> (Result<Event, Fault>, Self)
    where
        F: FnOnce(L, usize) -> (Result<Event, Fault>, L),
    {
        match self {
            Self::Strict(mut machine) => {
                if let Some(fault) = machine.fault {
                    return (Err(fault), Self::Strict(machine));
                }
                let result = execution_core::execute(&machine.program, &mut machine.state, actor);
                if let Err(fault) = result {
                    machine.fault = Some(fault);
                }
                (result, Self::Strict(machine))
            }
            Self::Legacy(machine) => {
                let (result, state) = legacy(machine, actor);
                (result, Self::Legacy(state))
            }
            Self::Fault(fault) => (Err(fault), Self::Fault(fault)),
        }
    }

    pub fn step<F>(&mut self, actor: usize, legacy: F) -> Result<Event, Fault>
    where
        F: FnOnce(L, usize) -> (Result<Event, Fault>, L),
    {
        let current = std::mem::replace(self, Self::Fault(Fault::WrongMode));
        let (result, next) = current.step_owned(actor, legacy);
        *self = next;
        result
    }

    pub fn outputs<F>(&self, legacy: F) -> Result<Vec<u8>, Fault>
    where
        F: FnOnce(&L) -> Result<Vec<u8>, Fault>,
    {
        match self {
            Self::Strict(machine) => {
                if let Some(fault) = machine.fault {
                    return Err(fault);
                }
                if !execution_core::terminal(&machine.state) {
                    return Err(Fault::ThreadsRunning);
                }
                execution_core::outputs(&machine.state, &machine.program.outputs)
            }
            Self::Legacy(machine) => legacy(machine),
            Self::Fault(fault) => Err(*fault),
        }
    }

    pub fn enabled<F>(&self, actor: usize, legacy: F) -> bool
    where
        F: FnOnce(&L, usize) -> bool,
    {
        match self {
            Self::Strict(machine) => {
                machine.fault.is_none() && execution_core::running(&machine.state, actor)
            }
            Self::Legacy(machine) => legacy(machine, actor),
            Self::Fault(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use ptxv_memory_write::execution_core::AllocationBytes;
    use ptxv_memory_write::execution_core::ThreadState;
    use ptxv_memory_write::machine_registers::LaneRegisters;
    use ptxv_memory_write::value_flow::Address;
    use ptxv_memory_write::value_flow::AllocationLayout;
    use ptxv_memory_write::value_flow::Guard;
    use ptxv_memory_write::value_flow::GuardedInstruction;
    use ptxv_memory_write::value_flow::Instruction;
    use ptxv_memory_write::value_flow::Operand;
    use ptxv_memory_write::value_flow::Pointer;
    use ptxv_memory_write::value_flow::Registers;
    use ptxv_memory_write::value_flow::ThreadIds;

    use super::*;

    fn instruction(instruction: Instruction) -> GuardedInstruction {
        GuardedInstruction {
            guard: Guard {
                predicate: None,
                negated: false,
            },
            instruction,
        }
    }

    fn pointer(offset: u64) -> Address {
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
                base: 0x1000,
                bytes: 8,
                alignment: 8,
            }],
            initial: State {
                threads: vec![ThreadState {
                    pc: 0,
                    running: true,
                    registers: Registers {
                        raw: LaneRegisters {
                            regs: vec![0; 2],
                            preds: vec![0],
                            pdef: vec![0],
                            written: vec![0],
                            vund: vec![],
                        },
                        pointers: vec![None; 2],
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
                    bytes: vec![0; 8],
                    defined: vec![true; 8],
                    live: true,
                }],
            },
            outputs: vec![0],
        }
    }

    #[test]
    fn strict_machine_owns_registers_bytes_pc_and_output_without_legacy_execution() {
        let program = program(vec![
            instruction(Instruction::Store {
                bytes: 4,
                address: pointer(0),
                source: Operand::Immediate(0x80ff_7f01),
            }),
            instruction(Instruction::Load {
                destination: 0,
                destination_bits: 64,
                bytes: 1,
                signed: true,
                address: pointer(3),
            }),
            instruction(Instruction::Exit),
        ]);
        let mut machine = ExecutionMachine::<()>::strict(program).unwrap();
        assert!(machine.enabled(0, |_, _| panic!("legacy enabledness")));
        assert_eq!(
            machine.outputs(|_| panic!("legacy early observer")),
            Err(Fault::ThreadsRunning)
        );
        let store = machine.step(0, |_, _| panic!("legacy store")).unwrap();
        assert!(matches!(store, Event::Access { thread: 0, .. }));
        assert_eq!(machine.state().unwrap().threads[0].pc, 1);
        machine.step(0, |_, _| panic!("legacy load")).unwrap();
        assert_eq!(
            machine.state().unwrap().threads[0].registers.raw.regs[0],
            0xffff_ffff_ffff_ff80
        );
        machine.step(0, |_, _| panic!("legacy exit")).unwrap();
        assert!(!machine.enabled(0, |_, _| panic!("legacy enabledness")));
        assert_eq!(
            machine.outputs(|_| panic!("legacy observer")).unwrap(),
            vec![1, 0x7f, 0xff, 0x80, 0, 0, 0, 0]
        );
    }

    #[test]
    fn strict_origin_escape_fault_is_sticky_and_preserves_owned_state() {
        let mut program = program(vec![instruction(Instruction::Store {
            bytes: 4,
            address: pointer(1u64 << 32),
            source: Operand::Immediate(7),
        })]);
        program.layouts.push(AllocationLayout {
            base: 0x1000 + (1u64 << 32),
            bytes: 8,
            alignment: 8,
        });
        program.initial.allocations.push(AllocationBytes {
            bytes: vec![0; 8],
            defined: vec![true; 8],
            live: true,
        });
        let expected = program.initial.clone();
        let mut machine = ExecutionMachine::<()>::strict(program).unwrap();
        assert_eq!(
            machine.step(0, |_, _| panic!("legacy escape")),
            Err(Fault::OutOfBounds)
        );
        assert_eq!(machine.state(), Some(&expected));
        assert_eq!(
            machine.step(0, |_, _| panic!("legacy retry")),
            Err(Fault::OutOfBounds)
        );
        assert_eq!(
            machine.outputs(|_| panic!("legacy observer")),
            Err(Fault::OutOfBounds)
        );
    }

    #[test]
    fn strict_fallthrough_is_an_enabled_fault_not_silently_terminal() {
        let mut machine =
            ExecutionMachine::<()>::strict(program(vec![instruction(Instruction::Nop)])).unwrap();
        machine.step(0, |_, _| panic!("legacy nop")).unwrap();
        assert!(machine.enabled(0, |_, _| panic!("legacy enabledness")));
        assert_eq!(
            machine.step(0, |_, _| panic!("legacy fallthrough")),
            Err(Fault::InvalidPc)
        );
    }

    #[test]
    fn strict_entry_rejects_duplicate_undefined_register_metadata() {
        let mut program = program(vec![instruction(Instruction::Exit)]);
        program.initial.threads[0].registers.raw.vund = vec![(0, 1), (0, 2)];
        assert!(matches!(
            ExecutionMachine::<()>::strict(program),
            Err(Fault::InvalidLayout)
        ));
    }
}

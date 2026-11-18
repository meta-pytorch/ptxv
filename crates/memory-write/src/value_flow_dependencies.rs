/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::symbolic::Metadata;
use crate::symbolic::{self};
use crate::value_flow::Action;
use crate::value_flow::Address;
use crate::value_flow::GuardedInstruction;
use crate::value_flow::Instruction;
use crate::value_flow::Operand;
use crate::value_flow::Value;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Registers {
    pub virtuals: Vec<Metadata>,
    pub predicates: Vec<Metadata>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    pub threads: Vec<Registers>,
    pub memory: Vec<Vec<Metadata>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DependencyError {
    Shape,
    Undefined,
    Guard,
    Address,
    Action,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transfer {
    None,
    Virtual {
        destination: u16,
        metadata: Metadata,
    },
    Predicate {
        destination: u16,
        metadata: Metadata,
    },
    Store {
        metadata: Metadata,
    },
}

pub fn metadata_equal(left: Metadata, right: Metadata) -> bool {
    let history = match (left.history, right.history) {
        (None, None) => true,
        (Some(left), Some(right)) => left == right,
        _ => false,
    };
    left.captured == right.captured
        && left.tainted == right.tainted
        && left.defined == right.defined
        && history
}

pub fn metadata_slice_equal(left: &[Metadata], right: &[Metadata]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut equal = true;
    let mut index = 0;
    while index < left.len() {
        equal = equal && metadata_equal(left[index], right[index]);
        index += 1;
    }
    equal
}

pub fn registers_equal(left: &Registers, right: &Registers) -> bool {
    metadata_slice_equal(&left.virtuals, &right.virtuals)
        && metadata_slice_equal(&left.predicates, &right.predicates)
}

pub fn state_equal(left: &State, right: &State) -> bool {
    if left.threads.len() != right.threads.len() || left.memory.len() != right.memory.len() {
        return false;
    }
    let mut equal = true;
    let mut thread = 0;
    while thread < left.threads.len() {
        equal = equal && registers_equal(&left.threads[thread], &right.threads[thread]);
        thread += 1;
    }
    let mut allocation = 0;
    while allocation < left.memory.len() {
        equal = equal && metadata_slice_equal(&left.memory[allocation], &right.memory[allocation]);
        allocation += 1;
    }
    equal
}

pub fn snapshot_metadata(values: &[Metadata]) -> Vec<Metadata> {
    let mut result = Vec::new();
    let mut index = 0;
    while index < values.len() {
        result.push(values[index]);
        index += 1;
    }
    result
}

pub fn snapshot_registers(registers: &Registers) -> Registers {
    Registers {
        virtuals: snapshot_metadata(&registers.virtuals),
        predicates: snapshot_metadata(&registers.predicates),
    }
}

pub fn snapshot_state(state: &State) -> State {
    let mut threads = Vec::new();
    let mut thread = 0;
    while thread < state.threads.len() {
        threads.push(snapshot_registers(&state.threads[thread]));
        thread += 1;
    }
    let mut memory = Vec::new();
    let mut allocation = 0;
    while allocation < state.memory.len() {
        memory.push(snapshot_metadata(&state.memory[allocation]));
        allocation += 1;
    }
    State { threads, memory }
}

pub fn metadata(tainted: bool, defined: bool) -> Metadata {
    symbolic::prepare(0, None, tainted, defined)
}

pub fn join(left: Metadata, right: Metadata) -> Metadata {
    symbolic::binary(symbolic::BinaryOp::Or, left, right)
}

pub fn read(registers: &Registers, operand: Operand) -> Result<Metadata, DependencyError> {
    match operand {
        Operand::Virtual(index) => {
            if index as usize >= registers.virtuals.len() {
                return Err(DependencyError::Shape);
            }
            let value = registers.virtuals[index as usize];
            if !value.defined {
                return Err(DependencyError::Undefined);
            }
            Ok(value)
        }
        Operand::Immediate(_) | Operand::Pointer(_) | Operand::Special(_) => {
            Ok(metadata(false, true))
        }
    }
}

pub fn guard(
    registers: &Registers,
    instruction: &GuardedInstruction,
) -> Result<(), DependencyError> {
    let value = match instruction.guard.predicate {
        Some(index) => {
            if index as usize >= registers.predicates.len() {
                return Err(DependencyError::Shape);
            }
            registers.predicates[index as usize]
        }
        None => metadata(false, true),
    };
    if !value.defined {
        return Err(DependencyError::Undefined);
    }
    // captured is not a control value: it is deliberately absent from this test.
    if symbolic::control(value, metadata(false, true), metadata(false, true)).guard_dependent {
        return Err(DependencyError::Guard);
    }
    Ok(())
}

fn address(registers: &Registers, location: Address) -> Result<(), DependencyError> {
    let base = match read(registers, location.base) {
        Ok(value) => value,
        Err(error) => return Err(error),
    };
    let index = match read(registers, location.index) {
        Ok(value) => value,
        Err(error) => return Err(error),
    };
    if symbolic::control(metadata(false, true), base, index).address_dependent {
        return Err(DependencyError::Address);
    }
    Ok(())
}

fn pair(registers: &Registers, left: Operand, right: Operand) -> Result<Metadata, DependencyError> {
    let left = match read(registers, left) {
        Ok(value) => value,
        Err(error) => return Err(error),
    };
    let right = match read(registers, right) {
        Ok(value) => value,
        Err(error) => return Err(error),
    };
    Ok(join(left, right))
}

pub fn loaded_metadata(loaded: &[Metadata], bytes: u8) -> Result<Metadata, DependencyError> {
    if loaded.len() != bytes as usize || loaded.is_empty() {
        return Err(DependencyError::Shape);
    }
    let mut result = metadata(false, true);
    let mut index = 0;
    while index < loaded.len() {
        result = join(result, loaded[index]);
        index += 1;
    }
    if !result.defined {
        return Err(DependencyError::Undefined);
    }
    Ok(result)
}

pub fn transfer(
    instruction: &GuardedInstruction,
    action: &Action,
    before: &Registers,
    loaded: &[Metadata],
) -> Result<Transfer, DependencyError> {
    match guard(before, instruction) {
        Ok(()) => (),
        Err(error) => return Err(error),
    }
    match *action {
        Action::Next => Ok(Transfer::None),
        Action::Jump { .. } => match instruction.instruction {
            Instruction::Branch { .. } => Ok(Transfer::None),
            _ => Err(DependencyError::Action),
        },
        Action::Halt => match instruction.instruction {
            Instruction::Exit => Ok(Transfer::None),
            _ => Err(DependencyError::Action),
        },
        Action::WriteVirtual { destination, value } => {
            let source = match instruction.instruction {
                Instruction::Move { source, .. } | Instruction::ConvertUnsigned { source, .. } => {
                    read(before, source)
                }
                Instruction::Binary { left, right, .. }
                | Instruction::MultiplyWide { left, right, .. } => pair(before, left, right),
                _ => return Err(DependencyError::Action),
            };
            match source {
                Ok(metadata) => {
                    if metadata.tainted && matches!(value, Value::Pointer(_)) {
                        return Err(DependencyError::Address);
                    }
                    Ok(Transfer::Virtual {
                        destination,
                        metadata,
                    })
                }
                Err(error) => Err(error),
            }
        }
        Action::WritePredicate { destination, .. } => match instruction.instruction {
            Instruction::Compare { left, right, .. } => match pair(before, left, right) {
                Ok(metadata) => Ok(Transfer::Predicate {
                    destination,
                    metadata,
                }),
                Err(error) => Err(error),
            },
            _ => Err(DependencyError::Action),
        },
        Action::Load {
            destination, bytes, ..
        } => match instruction.instruction {
            Instruction::Load {
                address: location, ..
            } => {
                match address(before, location) {
                    Ok(()) => (),
                    Err(error) => return Err(error),
                }
                match loaded_metadata(loaded, bytes) {
                    Ok(metadata) => Ok(Transfer::Virtual {
                        destination,
                        metadata,
                    }),
                    Err(error) => Err(error),
                }
            }
            _ => Err(DependencyError::Action),
        },
        Action::Store { .. } => match instruction.instruction {
            Instruction::Store {
                address: location,
                source,
                ..
            } => {
                match address(before, location) {
                    Ok(()) => (),
                    Err(error) => return Err(error),
                }
                match read(before, source) {
                    Ok(metadata) => Ok(Transfer::Store { metadata }),
                    Err(error) => Err(error),
                }
            }
            _ => Err(DependencyError::Action),
        },
    }
}

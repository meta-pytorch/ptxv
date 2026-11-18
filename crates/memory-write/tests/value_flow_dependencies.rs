/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_memory_write::value_flow::*;
use ptxv_memory_write::value_flow_dependencies::DependencyError;
use ptxv_memory_write::value_flow_dependencies::Transfer;
use ptxv_memory_write::value_flow_dependencies::{self as deps};

fn bank() -> deps::Registers {
    deps::Registers {
        virtuals: vec![deps::metadata(false, true); 8],
        predicates: vec![deps::metadata(false, true); 8],
    }
}

fn instruction(instruction: Instruction) -> GuardedInstruction {
    GuardedInstruction {
        guard: Guard {
            predicate: None,
            negated: false,
        },
        instruction,
    }
}

fn address() -> Address {
    Address {
        base: Operand::Pointer(Pointer {
            allocation: 0,
            offset: 0,
        }),
        index: Operand::Immediate(0),
        offset: 0,
    }
}

#[test]
fn false_guard_cannot_hide_input_dependence() {
    let mut registers = bank();
    registers.predicates[7] = deps::metadata(true, true);
    let mut operation = instruction(Instruction::Exit);
    operation.guard = Guard {
        predicate: Some(7),
        negated: true,
    };
    assert_eq!(
        deps::transfer(&operation, &Action::Next, &registers, &[]),
        Err(DependencyError::Guard)
    );
    registers.predicates[7] = deps::metadata(false, true);
    assert_eq!(
        deps::transfer(&operation, &Action::Next, &registers, &[]),
        Ok(Transfer::None)
    );
}

#[test]
fn whole_address_register_dependency_is_checked() {
    let mut registers = bank();
    registers.virtuals[1] = deps::metadata(true, true);
    let mut location = address();
    location.index = Operand::Virtual(1);
    let operation = instruction(Instruction::Store {
        bytes: 4,
        address: location,
        source: Operand::Immediate(0),
    });
    let action = Action::Store {
        bytes: 4,
        pointer: Pointer {
            allocation: 0,
            offset: 0,
        },
        value: 0,
    };
    assert_eq!(
        deps::transfer(&operation, &action, &registers, &[]),
        Err(DependencyError::Address)
    );
}

#[test]
fn every_loaded_byte_contributes_to_the_result() {
    let registers = bank();
    let operation = instruction(Instruction::Load {
        destination: 2,
        destination_bits: 64,
        bytes: 8,
        signed: false,
        address: address(),
    });
    let action = Action::Load {
        destination: 2,
        destination_bits: 64,
        bytes: 8,
        signed: false,
        pointer: Pointer {
            allocation: 0,
            offset: 0,
        },
    };
    for changed in 0..8 {
        let mut loaded = vec![deps::metadata(false, true); 8];
        loaded[changed] = deps::metadata(true, true);
        let Ok(Transfer::Virtual {
            destination: 2,
            metadata,
        }) = deps::transfer(&operation, &action, &registers, &loaded)
        else {
            panic!("load transfer refused");
        };
        assert!(metadata.tainted, "byte {changed} dependency disappeared");
    }
    assert_eq!(
        deps::transfer(&operation, &action, &registers, &[]),
        Err(DependencyError::Shape)
    );
}

#[test]
fn stores_and_comparisons_preserve_dependency_even_for_zero_values() {
    let mut registers = bank();
    registers.virtuals[0] = deps::metadata(true, true);
    let store = instruction(Instruction::Store {
        bytes: 8,
        address: address(),
        source: Operand::Virtual(0),
    });
    let action = Action::Store {
        bytes: 8,
        pointer: Pointer {
            allocation: 0,
            offset: 0,
        },
        value: 0,
    };
    let Ok(Transfer::Store { metadata }) = deps::transfer(&store, &action, &registers, &[]) else {
        panic!("store transfer refused");
    };
    assert!(metadata.tainted);
    let compare = instruction(Instruction::Compare {
        bits: 64,
        destination: 1,
        comparison: Comparison::Equal,
        left: Operand::Virtual(0),
        right: Operand::Immediate(0),
    });
    let action = Action::WritePredicate {
        destination: 1,
        value: true,
    };
    let Ok(Transfer::Predicate { metadata, .. }) =
        deps::transfer(&compare, &action, &registers, &[])
    else {
        panic!("comparison transfer refused");
    };
    assert!(metadata.tainted);
}

#[test]
fn scalar_overwrite_clears_dependency_but_does_not_mutate_frozen_operands() {
    let mut registers = bank();
    registers.virtuals[0] = deps::metadata(true, true);
    let operation = instruction(Instruction::Move {
        bits: 64,
        destination: 0,
        source: Operand::Immediate(1),
    });
    let action = Action::WriteVirtual {
        destination: 0,
        value: Value::Scalar(1),
    };
    let Ok(Transfer::Virtual { metadata, .. }) =
        deps::transfer(&operation, &action, &registers, &[])
    else {
        panic!("move transfer refused");
    };
    assert!(!metadata.tainted);
    assert!(registers.virtuals[0].tainted);
}

#[test]
fn undefined_metadata_and_mismatched_action_are_refused() {
    let mut registers = bank();
    registers.virtuals[0] = deps::metadata(false, false);
    let operation = instruction(Instruction::Move {
        bits: 32,
        destination: 1,
        source: Operand::Virtual(0),
    });
    let action = Action::WriteVirtual {
        destination: 1,
        value: Value::Scalar(0),
    };
    assert_eq!(
        deps::transfer(&operation, &action, &registers, &[]),
        Err(DependencyError::Undefined)
    );
    let operation = instruction(Instruction::Exit);
    assert_eq!(
        deps::transfer(&operation, &action, &registers, &[]),
        Err(DependencyError::Action)
    );
}

#[test]
fn input_dependent_pointer_arithmetic_cannot_hide_alternate_overflow() {
    let mut registers = bank();
    registers.virtuals[0] = deps::metadata(true, true);
    let operation = instruction(Instruction::Binary {
        bits: 64,
        destination: 1,
        operation: BinaryOperation::Add,
        left: Operand::Pointer(Pointer {
            allocation: 0,
            offset: 0,
        }),
        right: Operand::Virtual(0),
    });
    let action = Action::WriteVirtual {
        destination: 1,
        value: Value::Pointer(Pointer {
            allocation: 0,
            offset: 0,
        }),
    };
    assert_eq!(
        deps::transfer(&operation, &action, &registers, &[]),
        Err(DependencyError::Address)
    );
}

#[test]
fn metadata_equality_and_snapshot_preserve_every_field_and_shape() {
    let original = deps::State {
        threads: vec![bank()],
        memory: vec![vec![deps::metadata(true, true); 8]],
    };
    let saved = deps::snapshot_state(&original);
    assert!(deps::state_equal(&saved, &original));
    for field in 0..4 {
        let mut changed = deps::snapshot_state(&original);
        match field {
            0 => changed.memory[0][7].captured = 1,
            1 => changed.memory[0][7].history = Some(0),
            2 => changed.memory[0][7].tainted = false,
            _ => changed.memory[0][7].defined = false,
        }
        assert!(
            !deps::state_equal(&changed, &original),
            "field {field} ignored"
        );
        assert!(deps::state_equal(&saved, &original));
    }
    let mut changed = deps::snapshot_state(&original);
    changed.threads[0].predicates.pop();
    assert!(!deps::state_equal(&changed, &original));
    changed = deps::snapshot_state(&original);
    changed.memory.push(Vec::new());
    assert!(!deps::state_equal(&changed, &original));
    let mut left = deps::metadata(false, true);
    let mut right = left;
    left.history = Some(1);
    right.history = Some(2);
    assert!(!deps::metadata_equal(left, right));
}

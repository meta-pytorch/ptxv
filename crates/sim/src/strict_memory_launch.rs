/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_memory_write::memory_safety::Allocation;
use ptxv_memory_write::memory_safety::Instruction;

use crate::machine::Alloc;
use crate::machine::Launch;
use crate::strict_memory_admission::Refusal;

pub(crate) fn dimensions_product(dimensions: [u32; 3]) -> Option<usize> {
    let mut product = Some(1usize);
    let mut index = 0;
    while index < 3 {
        product = match product {
            Some(value) if dimensions[index] != 0 => value.checked_mul(dimensions[index] as usize),
            _ => None,
        };
        index += 1;
    }
    product
}

pub(crate) fn aliases_present(aliases: &[bool]) -> bool {
    let mut found = false;
    let mut index = 0;
    while index < aliases.len() {
        found = found || aliases[index];
        index += 1;
    }
    found
}

pub(crate) fn validate_outputs(launch: &Launch) -> Result<(), Refusal> {
    let mut failure = None;
    let mut index = 0;
    while index < launch.outputs.len() && failure.is_none() {
        let output = launch.outputs[index];
        if output >= launch.allocs.len() {
            failure = Some(Refusal::Launch);
        } else if launch.allocs[output].undef {
            failure = Some(Refusal::Allocation(output));
        }
        index += 1;
    }
    match failure {
        Some(reason) => Err(reason),
        None => Ok(()),
    }
}

pub(crate) fn project_allocation(allocation: &Alloc, index: usize) -> Result<Allocation, Refusal> {
    let base = allocation.base;
    let bytes = allocation.init.len() as u64;
    let initially_defined = !allocation.undef;
    if base.checked_add(bytes).is_none() {
        Err(Refusal::Allocation(index))
    } else {
        Ok(Allocation {
            base,
            bytes,
            initially_defined,
        })
    }
}

pub(crate) fn project_allocations(launch: &Launch) -> Result<Vec<Allocation>, Refusal> {
    let mut allocations = Vec::new();
    let mut failure = None;
    let mut index = 0;
    while index < launch.allocs.len() && failure.is_none() {
        match project_allocation(&launch.allocs[index], index) {
            Ok(allocation) => allocations.push(allocation),
            Err(reason) => failure = Some(reason),
        }
        index += 1;
    }
    match failure {
        Some(reason) => Err(reason),
        None => Ok(allocations),
    }
}

pub(crate) fn replicate_code(instructions: &[Instruction], count: usize) -> Vec<Vec<Instruction>> {
    let mut threads = Vec::new();
    let mut thread = 0;
    while thread < count {
        let mut code = Vec::new();
        let mut pc = 0;
        while pc < instructions.len() {
            code.push(instructions[pc].clone());
            pc += 1;
        }
        threads.push(code);
        thread += 1;
    }
    threads
}

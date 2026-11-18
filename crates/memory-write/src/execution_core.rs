/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::memory_safety::AccessKind;
use crate::memory_safety::Event;
use crate::memory_safety::Request;
use crate::value_flow::Action;
use crate::value_flow::AllocationLayout;
use crate::value_flow::GuardedInstruction;
use crate::value_flow::Pointer;
use crate::value_flow::Registers;
use crate::value_flow::ThreadIds;
use crate::value_flow::Value;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub code: Vec<GuardedInstruction>,
    pub layouts: Vec<AllocationLayout>,
    pub initial: State,
    pub outputs: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThreadState {
    pub pc: u32,
    pub running: bool,
    pub registers: Registers,
    pub ids: ThreadIds,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllocationBytes {
    pub bytes: Vec<u8>,
    pub defined: Vec<bool>,
    pub live: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    pub threads: Vec<ThreadState>,
    pub allocations: Vec<AllocationBytes>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    InvalidLayout,
    InvalidActor,
    Exited,
    InvalidPc,
    InvalidOperand,
    InvalidDestination,
    InvalidWidth,
    InvalidAllocation,
    OutOfBounds,
    Misaligned,
    Uninitialized,
    DeadAllocation,
    InvalidOutput,
    UnsupportedAction,
    WrongMode,
    ThreadsRunning,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedAccess {
    pub allocation: usize,
    pub offset: usize,
    pub width: usize,
    pub request: Request,
}

fn allocation_compatible(program: &Program, index: usize) -> bool {
    let layout = program.layouts[index];
    let mut valid = true;
    let mut j = 0;
    while j < index && valid {
        let other = program.layouts[j];
        valid = !((layout.base as u128) < other.base as u128 + other.bytes as u128
            && (other.base as u128) < layout.base as u128 + layout.bytes as u128);
        j += 1;
    }
    valid
}

fn allocation_shape(layout: AllocationLayout, bytes: usize, defined: usize) -> bool {
    layout.bytes != 0
        && layout.bytes as u128 == bytes as u128
        && bytes == defined
        && layout.alignment != 0
        && layout.base % layout.alignment == 0
        && layout.base as u128 + layout.bytes as u128 <= u64::MAX as u128 + 1
}

pub fn layout_valid(program: &Program, state: &State) -> bool {
    let mut valid = program.layouts.len() == state.allocations.len();
    let mut i = 0;
    while i < program.layouts.len() && valid {
        let layout = program.layouts[i];
        let bytes = state.allocations[i].bytes.len();
        let defined = state.allocations[i].defined.len();
        valid = allocation_shape(layout, bytes, defined) && allocation_compatible(program, i);
        i += 1;
    }
    valid
}

fn virtual_undefined_valid(registers: &Registers) -> bool {
    let mut valid = true;
    let mut first = true;
    let mut previous = 0;
    let mut at = 0;
    while at < registers.raw.vund.len() && valid {
        let (index, mask) = registers.raw.vund[at];
        valid = usize::from(index) < registers.raw.regs.len()
            && mask != 0
            && mask != u64::MAX
            && (first || previous < index);
        first = false;
        previous = index;
        at += 1;
    }
    valid
}

fn registers_valid(registers: &Registers, layouts: &[AllocationLayout]) -> bool {
    let count = registers.raw.regs.len();
    let mut valid = registers.pointers.len() == count
        && registers.raw.written.len() >= count / 64 + if count % 64 == 0 { 0 } else { 1 }
        && registers.raw.preds.len() == registers.raw.pdef.len()
        && virtual_undefined_valid(registers);
    let mut index = 0;
    while index < count && valid {
        if let Some(pointer) = registers.pointers[index] {
            match crate::value_flow::render_value(Value::Pointer(pointer), layouts) {
                Ok(bits) => valid = registers.raw.regs[index] == bits,
                Err(_) => valid = false,
            }
        }
        index += 1;
    }
    valid
}

pub fn snapshot_state(state: &State) -> State {
    let mut threads = Vec::new();
    let mut index = 0;
    while index < state.threads.len() {
        let source = &state.threads[index];
        threads.push(ThreadState {
            pc: source.pc,
            running: source.running,
            registers: crate::value_flow::snapshot_registers(&source.registers),
            ids: source.ids,
        });
        index += 1;
    }
    let mut allocations = Vec::new();
    index = 0;
    while index < state.allocations.len() {
        let source = &state.allocations[index];
        allocations.push(AllocationBytes {
            bytes: source.bytes.clone(),
            defined: source.defined.clone(),
            live: source.live,
        });
        index += 1;
    }
    State {
        threads,
        allocations,
    }
}

pub fn state_equal(left: &State, right: &State) -> bool {
    left == right
}

pub fn initial_state(program: &Program) -> Result<State, Fault> {
    if !layout_valid(program, &program.initial) {
        return Err(Fault::InvalidLayout);
    }
    let mut valid = true;
    let mut index = 0;
    while index < program.initial.threads.len() && valid {
        let thread = &program.initial.threads[index];
        valid = thread.pc == 0
            && thread.running
            && registers_valid(&thread.registers, &program.layouts);
        index += 1;
    }
    index = 0;
    while index < program.initial.allocations.len() && valid {
        valid = program.initial.allocations[index].live;
        index += 1;
    }
    if valid {
        Ok(snapshot_state(&program.initial))
    } else {
        Err(Fault::InvalidLayout)
    }
}

pub fn supported_width(width: u8) -> bool {
    width == 1 || width == 2 || width == 4 || width == 8
}

/// An origin-relative range is checked before its physical address is formed.
pub fn resolve_access(
    program: &Program,
    state: &State,
    pointer: Pointer,
    width: u8,
    write: bool,
) -> Result<ResolvedAccess, Fault> {
    if !supported_width(width) {
        return Err(Fault::InvalidWidth);
    }
    if pointer.allocation >= program.layouts.len() || pointer.allocation >= state.allocations.len()
    {
        return Err(Fault::InvalidAllocation);
    }
    let layout = &program.layouts[pointer.allocation];
    let memory = &state.allocations[pointer.allocation];
    if !memory.live {
        return Err(Fault::DeadAllocation);
    }
    let bytes = u64::from(width);
    if pointer.offset > layout.bytes || bytes > layout.bytes - pointer.offset {
        return Err(Fault::OutOfBounds);
    }
    if layout.alignment == 0
        || layout.alignment % bytes != 0
        || pointer.offset % bytes != 0
        || layout.base % bytes != 0
    {
        return Err(Fault::Misaligned);
    }
    let address = layout.base as u128 + pointer.offset as u128;
    if address + bytes as u128 > u64::MAX as u128 + 1 {
        return Err(Fault::OutOfBounds);
    }
    if pointer.offset as u128 > usize::MAX as u128 {
        return Err(Fault::OutOfBounds);
    }
    let offset = pointer.offset as usize;
    let count = usize::from(width);
    if offset > memory.bytes.len()
        || count > memory.bytes.len() - offset
        || memory.defined.len() != memory.bytes.len()
    {
        return Err(Fault::InvalidLayout);
    }
    let mut at = 0;
    let mut defined = true;
    while at < count && !write && defined {
        defined = memory.defined[offset + at];
        at += 1;
    }
    if !defined {
        return Err(Fault::Uninitialized);
    }
    Ok(ResolvedAccess {
        allocation: pointer.allocation,
        offset,
        width: count,
        request: Request {
            address: address as u64,
            width: bytes,
            alignment: bytes,
            kind: if write {
                AccessKind::Store { defined: true }
            } else {
                AccessKind::Load
            },
        },
    })
}

pub fn load_value(state: &State, access: ResolvedAccess, signed: bool) -> u64 {
    let memory = &state.allocations[access.allocation];
    let mut value = 0;
    let mut at = 0;
    while at < access.width {
        value |= u64::from(memory.bytes[access.offset + at]) << (at * 8);
        at += 1;
    }
    if signed && access.width < 8 {
        let bits = access.width * 8;
        let shift = 64 - bits;
        value = ((value << shift) as i64 >> shift) as u64;
    }
    value
}

pub fn store_value(state: &mut State, access: ResolvedAccess, value: u64) {
    let memory = &mut state.allocations[access.allocation];
    let mut at = 0;
    while at < access.width {
        memory.bytes[access.offset + at] = (value >> (at * 8)) as u8;
        memory.defined[access.offset + at] = true;
        at += 1;
    }
}

fn append_output(state: &State, index: usize, output: &mut Vec<u8>) -> Result<(), Fault> {
    if index >= state.allocations.len() {
        return Err(Fault::InvalidOutput);
    }
    let memory = &state.allocations[index];
    if !memory.live || memory.defined.len() != memory.bytes.len() {
        return Err(Fault::InvalidOutput);
    }
    let mut at = 0;
    let mut defined = true;
    while at < memory.defined.len() && defined {
        defined = memory.defined[at];
        at += 1;
    }
    if !defined {
        return Err(Fault::Uninitialized);
    }
    at = 0;
    while at < memory.bytes.len() {
        output.push(memory.bytes[at]);
        at += 1;
    }
    Ok(())
}

pub fn outputs(state: &State, allocations: &[usize]) -> Result<Vec<u8>, Fault> {
    let mut bytes = Vec::new();
    let mut fault = None;
    let mut i = 0;
    while i < allocations.len() && fault.is_none() {
        match append_output(state, allocations[i], &mut bytes) {
            Ok(()) => {}
            Err(error) => fault = Some(error),
        }
        i += 1;
    }
    match fault {
        None => Ok(bytes),
        Some(error) => Err(error),
    }
}

pub fn running(state: &State, actor: usize) -> bool {
    actor < state.threads.len() && state.threads[actor].running
}

pub fn terminal(state: &State) -> bool {
    let mut terminal = true;
    let mut actor = 0;
    while actor < state.threads.len() && terminal {
        terminal = !state.threads[actor].running;
        actor += 1;
    }
    terminal
}

/// Invalid PCs remain enabled so certificate closure must reject their fault.
pub fn prepare(program: &Program, state: &State, actor: usize) -> Result<Commit, Fault> {
    if !layout_valid(program, state) {
        return Err(Fault::InvalidLayout);
    }
    if actor >= state.threads.len() {
        return Err(Fault::InvalidActor);
    }
    let thread = &state.threads[actor];
    if !thread.running {
        return Err(Fault::Exited);
    }
    if thread.pc as usize >= program.code.len() || thread.pc == u32::MAX {
        return Err(Fault::InvalidPc);
    }
    let next_pc = thread.pc + 1;
    let action = match crate::value_flow::evaluate(
        &thread.registers,
        &thread.ids,
        &program.code[thread.pc as usize],
    ) {
        Ok(action) => action,
        Err(_) => return Err(Fault::InvalidOperand),
    };
    Ok(Commit {
        actor,
        next_pc,
        action,
    })
}

pub fn execute(program: &Program, state: &mut State, actor: usize) -> Result<Event, Fault> {
    let commit = match prepare(program, state, actor) {
        Ok(commit) => commit,
        Err(fault) => return Err(fault),
    };
    commit_action(program, state, commit)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Commit {
    pub actor: usize,
    pub next_pc: u32,
    pub action: Action,
}

pub fn commit_action(program: &Program, state: &mut State, commit: Commit) -> Result<Event, Fault> {
    let Commit {
        actor,
        next_pc,
        action,
    } = commit;
    if actor >= state.threads.len() {
        return Err(Fault::InvalidActor);
    }
    let mut event = Event::Control;
    match action {
        Action::Next => state.threads[actor].pc = next_pc,
        Action::WriteVirtual { destination, value } => {
            match crate::value_flow::write_value(
                &mut state.threads[actor].registers,
                destination,
                value,
                &program.layouts,
            ) {
                Ok(()) => state.threads[actor].pc = next_pc,
                Err(_) => return Err(Fault::InvalidDestination),
            }
        }
        Action::WritePredicate { destination, value } => {
            match crate::value_flow::write_predicate(
                &mut state.threads[actor].registers,
                destination,
                value,
            ) {
                Ok(()) => state.threads[actor].pc = next_pc,
                Err(_) => return Err(Fault::InvalidDestination),
            }
        }
        Action::Jump { target } => {
            if target as usize >= program.code.len() {
                return Err(Fault::InvalidPc);
            }
            state.threads[actor].pc = target;
        }
        Action::Halt => state.threads[actor].running = false,
        Action::Load {
            destination,
            destination_bits,
            bytes,
            signed,
            pointer,
        } => {
            let access = match resolve_access(program, state, pointer, bytes, false) {
                Ok(access) => access,
                Err(fault) => return Err(fault),
            };
            let value = load_value(state, access, signed);
            let mask = match crate::value_flow::width_mask(destination_bits) {
                Ok(mask) => mask,
                Err(_) => return Err(Fault::InvalidWidth),
            };
            match crate::value_flow::write_value(
                &mut state.threads[actor].registers,
                destination,
                Value::Scalar(value & mask),
                &program.layouts,
            ) {
                Ok(()) => state.threads[actor].pc = next_pc,
                Err(_) => return Err(Fault::InvalidDestination),
            }
            event = Event::Access {
                thread: actor,
                request: access.request,
            };
        }
        Action::Store {
            bytes,
            pointer,
            value,
        } => {
            let access = match resolve_access(program, state, pointer, bytes, true) {
                Ok(access) => access,
                Err(fault) => return Err(fault),
            };
            store_value(state, access, value);
            state.threads[actor].pc = next_pc;
            event = Event::Access {
                thread: actor,
                request: access.request,
            };
        }
    }
    Ok(event)
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::atomic_integer;
use crate::machine_registers;
use crate::machine_registers::LaneRegisters;
use crate::symbolic;

macro_rules! value_or_error {
    ($computation:expr) => {
        match $computation {
            Ok(value) => value,
            Err(error) => return Err(error),
        }
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AllocationLayout {
    pub base: u64,
    pub bytes: u64,
    pub alignment: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Pointer {
    pub allocation: usize,
    pub offset: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Value {
    Scalar(u64),
    Pointer(Pointer),
}

#[derive(Clone, Debug, Eq)]
pub struct Registers {
    pub raw: LaneRegisters,
    pub pointers: Vec<Option<Pointer>>,
}

impl PartialEq for Registers {
    fn eq(&self, other: &Self) -> bool {
        registers_equal(self, other)
    }
}

impl std::hash::Hash for Registers {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.raw, state);
        std::hash::Hash::hash(&self.pointers, state);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ThreadIds {
    pub tid: [u32; 3],
    pub ctaid: [u32; 3],
    pub ntid: [u32; 3],
    pub nctaid: [u32; 3],
    pub lane: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpecialRegister {
    Thread(u8),
    Block(u8),
    BlockSize(u8),
    GridSize(u8),
    Lane,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operand {
    Immediate(u64),
    Virtual(u16),
    Pointer(Pointer),
    Special(SpecialRegister),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Guard {
    pub predicate: Option<u16>,
    pub negated: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Address {
    pub base: Operand,
    pub index: Operand,
    pub offset: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOperation {
    Add,
    Subtract,
    And,
    Or,
    Xor,
    MultiplyLow,
    ShiftLeft,
    ShiftRightUnsigned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Comparison {
    Equal,
    NotEqual,
    LessUnsigned,
    LessEqualUnsigned,
    GreaterUnsigned,
    GreaterEqualUnsigned,
    LessSigned,
    LessEqualSigned,
    GreaterSigned,
    GreaterEqualSigned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Instruction {
    Move {
        bits: u8,
        destination: u16,
        source: Operand,
    },
    Binary {
        bits: u8,
        destination: u16,
        operation: BinaryOperation,
        left: Operand,
        right: Operand,
    },
    MultiplyWide {
        destination: u16,
        left: Operand,
        right: Operand,
    },
    ConvertUnsigned {
        destination: u16,
        source: Operand,
        from_bits: u8,
        to_bits: u8,
    },
    Compare {
        bits: u8,
        destination: u16,
        comparison: Comparison,
        left: Operand,
        right: Operand,
    },
    Branch {
        target: u32,
    },
    Load {
        destination: u16,
        destination_bits: u8,
        bytes: u8,
        signed: bool,
        address: Address,
    },
    Store {
        bytes: u8,
        address: Address,
        source: Operand,
    },
    Nop,
    Exit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GuardedInstruction {
    pub guard: Guard,
    pub instruction: Instruction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Next,
    WriteVirtual {
        destination: u16,
        value: Value,
    },
    WritePredicate {
        destination: u16,
        value: bool,
    },
    Jump {
        target: u32,
    },
    Halt,
    Load {
        destination: u16,
        destination_bits: u8,
        bytes: u8,
        signed: bool,
        pointer: Pointer,
    },
    Store {
        bytes: u8,
        pointer: Pointer,
        value: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalError {
    UnsupportedWidth,
    RegisterShape,
    UndefinedRegister,
    UndefinedPredicate,
    UnsupportedSpecialRegister,
    PointerArithmetic,
    PointerOffsetOverflow,
    ExpectedPointer,
    Allocation,
}

pub fn pointer_tags_equal(left: &[Option<Pointer>], right: &[Option<Pointer>]) -> bool {
    let mut equal = left.len() == right.len();
    let mut index = 0;
    while equal && index < left.len() {
        equal = match (left[index], right[index]) {
            (None, None) => true,
            (Some(left), Some(right)) => left == right,
            _ => false,
        };
        index += 1;
    }
    equal
}

pub fn registers_equal(left: &Registers, right: &Registers) -> bool {
    left.raw == right.raw && pointer_tags_equal(&left.pointers, &right.pointers)
}

pub fn snapshot_registers(registers: &Registers) -> Registers {
    let raw = LaneRegisters {
        regs: registers.raw.regs.clone(),
        preds: registers.raw.preds.clone(),
        pdef: registers.raw.pdef.clone(),
        written: registers.raw.written.clone(),
        vund: registers.raw.vund.clone(),
    };
    let mut pointers = Vec::new();
    let mut index = 0;
    while index < registers.pointers.len() {
        let value = match registers.pointers[index] {
            None => None,
            Some(pointer) => Some(pointer),
        };
        pointers.push(value);
        index += 1;
    }
    Registers { raw, pointers }
}

pub fn width_mask(bits: u8) -> Result<u64, EvalError> {
    match bits {
        8 => Ok(0xff),
        16 => Ok(0xffff),
        32 => Ok(0xffff_ffff),
        64 => Ok(u64::MAX),
        _ => Err(EvalError::UnsupportedWidth),
    }
}

pub fn writable_virtual(registers: &Registers, index: u16) -> bool {
    usize::from(index) < registers.raw.regs.len()
        && usize::from(index) < registers.pointers.len()
        && usize::from(index / 64) < registers.raw.written.len()
}

pub fn writable_predicate(registers: &Registers, index: u16) -> bool {
    usize::from(index / 64) < registers.raw.preds.len()
        && usize::from(index / 64) < registers.raw.pdef.len()
}

pub fn read_virtual(registers: &Registers, index: u16, bits: u8) -> Result<Value, EvalError> {
    let mask = value_or_error!(width_mask(bits));
    if !writable_virtual(registers, index) {
        return Err(EvalError::RegisterShape);
    }
    if registers.raw.written[usize::from(index / 64)] >> (index % 64) & 1 == 0 {
        return Err(EvalError::UndefinedRegister);
    }
    let mut undefined = 0;
    let mut position = 0;
    while position < registers.raw.vund.len() {
        let (key, value) = registers.raw.vund[position];
        if key == index {
            undefined |= value;
        }
        position += 1;
    }
    if undefined & mask != 0 {
        return Err(EvalError::UndefinedRegister);
    }
    match registers.pointers[usize::from(index)] {
        Some(pointer) if bits == 64 => Ok(Value::Pointer(pointer)),
        Some(_) => Err(EvalError::PointerArithmetic),
        None => Ok(Value::Scalar(registers.raw.regs[usize::from(index)] & mask)),
    }
}

pub fn read_predicate(registers: &Registers, index: u16) -> Result<bool, EvalError> {
    if !writable_predicate(registers, index) {
        return Err(EvalError::RegisterShape);
    }
    if registers.raw.pdef[usize::from(index / 64)] >> (index % 64) & 1 == 0 {
        return Err(EvalError::UndefinedPredicate);
    }
    Ok(registers.raw.preds[usize::from(index / 64)] >> (index % 64) & 1 != 0)
}

pub fn read_guard(registers: &Registers, guard: Guard) -> Result<bool, EvalError> {
    let value = match guard.predicate {
        None => true,
        Some(index) => value_or_error!(read_predicate(registers, index)),
    };
    Ok(value != guard.negated)
}

pub fn read_special(ids: &ThreadIds, special: SpecialRegister) -> Result<u64, EvalError> {
    let value = match special {
        SpecialRegister::Thread(axis) if axis < 3 => ids.tid[usize::from(axis)],
        SpecialRegister::Block(axis) if axis < 3 => ids.ctaid[usize::from(axis)],
        SpecialRegister::BlockSize(axis) if axis < 3 => ids.ntid[usize::from(axis)],
        SpecialRegister::GridSize(axis) if axis < 3 => ids.nctaid[usize::from(axis)],
        SpecialRegister::Lane => ids.lane,
        _ => return Err(EvalError::UnsupportedSpecialRegister),
    };
    Ok(u64::from(value))
}

pub fn read_operand(
    registers: &Registers,
    ids: &ThreadIds,
    operand: Operand,
    bits: u8,
) -> Result<Value, EvalError> {
    let mask = value_or_error!(width_mask(bits));
    match operand {
        Operand::Immediate(value) => Ok(Value::Scalar(value & mask)),
        Operand::Virtual(index) => read_virtual(registers, index, bits),
        Operand::Pointer(pointer) if bits == 64 => Ok(Value::Pointer(pointer)),
        Operand::Pointer(_) => Err(EvalError::PointerArithmetic),
        Operand::Special(special) => Ok(Value::Scalar(
            value_or_error!(read_special(ids, special)) & mask,
        )),
    }
}

pub fn scalar(value: Value) -> Result<u64, EvalError> {
    match value {
        Value::Scalar(value) => Ok(value),
        Value::Pointer(_) => Err(EvalError::PointerArithmetic),
    }
}

pub fn offset_pointer(pointer: Pointer, value: u64, subtract: bool) -> Result<Pointer, EvalError> {
    let offset = if subtract {
        pointer.offset.checked_sub(value)
    } else {
        pointer.offset.checked_add(value)
    };
    match offset {
        Some(offset) => Ok(Pointer {
            allocation: pointer.allocation,
            offset,
        }),
        None => Err(EvalError::PointerOffsetOverflow),
    }
}

pub fn scalar_binary(
    operation: BinaryOperation,
    bits: u8,
    left: u64,
    right: u64,
) -> Result<u64, EvalError> {
    if bits != 32 && bits != 64 {
        return Err(EvalError::UnsupportedWidth);
    }
    let mask = value_or_error!(width_mask(bits));
    let left = left & mask;
    let right = right & mask;
    let result = match operation {
        BinaryOperation::Add
        | BinaryOperation::And
        | BinaryOperation::Or
        | BinaryOperation::Xor => {
            let op = match operation {
                BinaryOperation::Add => atomic_integer::IntegerOp::Add,
                BinaryOperation::And => atomic_integer::IntegerOp::And,
                BinaryOperation::Or => atomic_integer::IntegerOp::Or,
                _ => atomic_integer::IntegerOp::Xor,
            };
            if bits == 32 {
                let symbolic_op = match operation {
                    BinaryOperation::Add => symbolic::BinaryOp::Add,
                    BinaryOperation::And => symbolic::BinaryOp::And,
                    BinaryOperation::Or => symbolic::BinaryOp::Or,
                    _ => symbolic::BinaryOp::Xor,
                };
                u64::from(symbolic::binary_bits(
                    symbolic_op,
                    left as u32,
                    right as u32,
                ))
            } else {
                match atomic_integer::replacement64(op, left, right, 0) {
                    Ok(value) => value,
                    Err(_) => return Err(EvalError::UnsupportedWidth),
                }
            }
        }
        BinaryOperation::Subtract => left.wrapping_sub(right),
        BinaryOperation::MultiplyLow => left.wrapping_mul(right),
        BinaryOperation::ShiftLeft => {
            if right >= u64::from(bits) {
                0
            } else {
                left << right
            }
        }
        BinaryOperation::ShiftRightUnsigned => {
            if right >= u64::from(bits) {
                0
            } else {
                left >> right
            }
        }
    };
    Ok(result & mask)
}

pub fn binary(
    operation: BinaryOperation,
    bits: u8,
    left: Value,
    right: Value,
) -> Result<Value, EvalError> {
    match (left, right) {
        (Value::Scalar(left), Value::Scalar(right)) => Ok(Value::Scalar(value_or_error!(
            scalar_binary(operation, bits, left, right)
        ))),
        (Value::Pointer(pointer), Value::Scalar(value)) if bits == 64 => match operation {
            BinaryOperation::Add => Ok(Value::Pointer(value_or_error!(offset_pointer(
                pointer, value, false
            )))),
            BinaryOperation::Subtract => Ok(Value::Pointer(value_or_error!(offset_pointer(
                pointer, value, true
            )))),
            _ => Err(EvalError::PointerArithmetic),
        },
        (Value::Scalar(value), Value::Pointer(pointer))
            if bits == 64 && operation == BinaryOperation::Add =>
        {
            Ok(Value::Pointer(value_or_error!(offset_pointer(
                pointer, value, false
            ))))
        }
        _ => Err(EvalError::PointerArithmetic),
    }
}

pub fn compare(comparison: Comparison, bits: u8, left: u64, right: u64) -> Result<bool, EvalError> {
    if bits != 32 && bits != 64 {
        return Err(EvalError::UnsupportedWidth);
    }
    let mask = value_or_error!(width_mask(bits));
    let left = left & mask;
    let right = right & mask;
    let signed_left = if bits == 32 {
        left as u32 as i32 as i64
    } else {
        left as i64
    };
    let signed_right = if bits == 32 {
        right as u32 as i32 as i64
    } else {
        right as i64
    };
    Ok(match comparison {
        Comparison::Equal => left == right,
        Comparison::NotEqual => left != right,
        Comparison::LessUnsigned => left < right,
        Comparison::LessEqualUnsigned => left <= right,
        Comparison::GreaterUnsigned => left > right,
        Comparison::GreaterEqualUnsigned => left >= right,
        Comparison::LessSigned => signed_left < signed_right,
        Comparison::LessEqualSigned => signed_left <= signed_right,
        Comparison::GreaterSigned => signed_left > signed_right,
        Comparison::GreaterEqualSigned => signed_left >= signed_right,
    })
}

pub fn address(
    registers: &Registers,
    ids: &ThreadIds,
    address: Address,
) -> Result<Pointer, EvalError> {
    let base = value_or_error!(read_operand(registers, ids, address.base, 64));
    let index = value_or_error!(read_operand(registers, ids, address.index, 64));
    let value = value_or_error!(binary(BinaryOperation::Add, 64, base, index));
    let pointer = match value {
        Value::Pointer(pointer) => pointer,
        Value::Scalar(_) => return Err(EvalError::ExpectedPointer),
    };
    if address.offset >= 0 {
        offset_pointer(pointer, address.offset as u64, false)
    } else {
        offset_pointer(pointer, 0u64.wrapping_sub(address.offset as u64), true)
    }
}

pub fn render_value(value: Value, layouts: &[AllocationLayout]) -> Result<u64, EvalError> {
    match value {
        Value::Scalar(value) => Ok(value),
        Value::Pointer(pointer) => {
            if pointer.allocation >= layouts.len() {
                return Err(EvalError::Allocation);
            }
            Ok(layouts[pointer.allocation]
                .base
                .wrapping_add(pointer.offset))
        }
    }
}

pub fn write_value(
    registers: &mut Registers,
    index: u16,
    value: Value,
    layouts: &[AllocationLayout],
) -> Result<(), EvalError> {
    if !writable_virtual(registers, index) {
        return Err(EvalError::RegisterShape);
    }
    let bits = value_or_error!(render_value(value, layouts));
    machine_registers::write_virtual(&mut registers.raw, index, bits, 0);
    registers.pointers[usize::from(index)] = match value {
        Value::Scalar(_) => None,
        Value::Pointer(pointer) => Some(pointer),
    };
    Ok(())
}

pub fn write_predicate(
    registers: &mut Registers,
    index: u16,
    value: bool,
) -> Result<(), EvalError> {
    if !writable_predicate(registers, index) {
        return Err(EvalError::RegisterShape);
    }
    machine_registers::write_predicate(&mut registers.raw, index, value, true);
    Ok(())
}

pub fn pointer_in_bounds(pointer: Pointer, bytes: u8, layouts: &[AllocationLayout]) -> bool {
    if pointer.allocation >= layouts.len() || !matches!(bytes, 1 | 2 | 4 | 8) {
        return false;
    }
    let layout = layouts[pointer.allocation];
    layout.alignment != 0
        && layout.alignment % u64::from(bytes) == 0
        && pointer.offset % u64::from(bytes) == 0
        && pointer.offset <= layout.bytes
        && u64::from(bytes) <= layout.bytes - pointer.offset
}

pub fn evaluate(
    registers: &Registers,
    ids: &ThreadIds,
    instruction: &GuardedInstruction,
) -> Result<Action, EvalError> {
    if !value_or_error!(read_guard(registers, instruction.guard)) {
        return Ok(Action::Next);
    }
    match instruction.instruction {
        Instruction::Nop => Ok(Action::Next),
        Instruction::Exit => Ok(Action::Halt),
        Instruction::Branch { target } => Ok(Action::Jump { target }),
        Instruction::Move {
            bits,
            destination,
            source,
        } => {
            let value = value_or_error!(read_operand(registers, ids, source, bits));
            Ok(Action::WriteVirtual { destination, value })
        }
        Instruction::Binary {
            bits,
            destination,
            operation,
            left,
            right,
        } => {
            let left = value_or_error!(read_operand(registers, ids, left, bits));
            let right = value_or_error!(read_operand(registers, ids, right, bits));
            let value = value_or_error!(binary(operation, bits, left, right));
            Ok(Action::WriteVirtual { destination, value })
        }
        Instruction::MultiplyWide {
            destination,
            left,
            right,
        } => {
            let left = value_or_error!(scalar(value_or_error!(read_operand(
                registers, ids, left, 32
            ))));
            let right = value_or_error!(scalar(value_or_error!(read_operand(
                registers, ids, right, 32
            ))));
            let value = Value::Scalar(left * right);
            Ok(Action::WriteVirtual { destination, value })
        }
        Instruction::ConvertUnsigned {
            destination,
            source,
            from_bits,
            to_bits,
        } => {
            let value = value_or_error!(read_operand(registers, ids, source, from_bits));
            let value = match value {
                Value::Scalar(value) => Value::Scalar(value & value_or_error!(width_mask(to_bits))),
                Value::Pointer(pointer) if to_bits == 64 => Value::Pointer(pointer),
                Value::Pointer(_) => return Err(EvalError::PointerArithmetic),
            };
            Ok(Action::WriteVirtual { destination, value })
        }
        Instruction::Compare {
            bits,
            destination,
            comparison,
            left,
            right,
        } => {
            let left = value_or_error!(scalar(value_or_error!(read_operand(
                registers, ids, left, bits
            ))));
            let right = value_or_error!(scalar(value_or_error!(read_operand(
                registers, ids, right, bits
            ))));
            Ok(Action::WritePredicate {
                destination,
                value: value_or_error!(compare(comparison, bits, left, right)),
            })
        }
        Instruction::Load {
            destination,
            destination_bits,
            bytes,
            signed,
            address: addr,
        } => {
            if !matches!(bytes, 1 | 2 | 4 | 8)
                || !matches!(destination_bits, 32 | 64)
                || u16::from(bytes) * 8 > u16::from(destination_bits)
            {
                return Err(EvalError::UnsupportedWidth);
            }
            Ok(Action::Load {
                destination,
                destination_bits,
                bytes,
                signed,
                pointer: value_or_error!(address(registers, ids, addr)),
            })
        }
        Instruction::Store {
            bytes,
            address: addr,
            source,
        } => {
            if !matches!(bytes, 1 | 2 | 4 | 8) {
                return Err(EvalError::UnsupportedWidth);
            }
            let pointer = value_or_error!(address(registers, ids, addr));
            let value = value_or_error!(scalar(value_or_error!(read_operand(
                registers,
                ids,
                source,
                bytes * 8
            ))));
            Ok(Action::Store {
                bytes,
                pointer,
                value,
            })
        }
    }
}

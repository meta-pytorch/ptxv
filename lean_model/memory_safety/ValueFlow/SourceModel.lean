/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.Semantics
import ValueFlow.SourceTypes

open Aeneas Aeneas.Std

set_option autoImplicit false

namespace ValueFlow.Source



def pointerModel (pointer : V.Pointer) : Pointer :=
  ⟨pointer.allocation.val, pointer.offset.val⟩

def valueModel : V.Value → Value
  | .Scalar bits => .scalar bits.val
  | .Pointer pointer => .pointer (pointerModel pointer)

def specialModel : V.SpecialRegister → Special × Nat
  | .Thread axis => (.thread, axis.val)
  | .Block axis => (.block, axis.val)
  | .BlockSize axis => (.blockSize, axis.val)
  | .GridSize axis => (.gridSize, axis.val)
  | .Lane => (.lane, 0)

def operandModel : V.Operand → Operand
  | .Immediate bits => .immediate bits.val
  | .Virtual index => .virtual index.val
  | .Pointer pointer => .pointer (pointerModel pointer)
  | .Special special => .special (specialModel special).1 (specialModel special).2

def guardModel (guard : V.Guard) : Guard :=
  ⟨guard.predicate.map UScalar.val, guard.negated⟩

def addressModel (address : V.Address) : Address :=
  ⟨operandModel address.base, operandModel address.index, address.offset.val⟩

def binaryModel : V.BinaryOperation → Binary
  | .Add => .add
  | .Subtract => .subtract
  | .And => .bitAnd
  | .Or => .bitOr
  | .Xor => .bitXor
  | .MultiplyLow => .multiply
  | .ShiftLeft => .shiftLeft
  | .ShiftRightUnsigned => .shiftRight

def comparisonModel : V.Comparison → Comparison
  | .Equal => .equal
  | .NotEqual => .notEqual
  | .LessUnsigned => .lessUnsigned
  | .LessEqualUnsigned => .lessEqualUnsigned
  | .GreaterUnsigned => .greaterUnsigned
  | .GreaterEqualUnsigned => .greaterEqualUnsigned
  | .LessSigned => .lessSigned
  | .LessEqualSigned => .lessEqualSigned
  | .GreaterSigned => .greaterSigned
  | .GreaterEqualSigned => .greaterEqualSigned

def instructionModel : V.Instruction → Instruction
  | .Move bits destination source =>
      .move bits.val destination.val (operandModel source)
  | .Binary bits destination operation left right =>
      .binary bits.val destination.val (binaryModel operation) (operandModel left) (operandModel right)
  | .MultiplyWide destination left right =>
      .multiplyWide destination.val (operandModel left) (operandModel right)
  | .ConvertUnsigned destination source fromBits toBits =>
      .convertUnsigned destination.val (operandModel source) fromBits.val toBits.val
  | .Compare bits destination comparison left right =>
      .compare bits.val destination.val (comparisonModel comparison) (operandModel left) (operandModel right)
  | .Branch target => .branch target.val
  | .Load destination destinationBits bytes signed address =>
      .load destination.val destinationBits.val bytes.val signed (addressModel address)
  | .Store bytes address source => .store bytes.val (addressModel address) (operandModel source)
  | .Nop => .nop
  | .Exit => .exit

def guardedModel (instruction : V.GuardedInstruction) : GuardedInstruction :=
  ⟨guardModel instruction.guard, instructionModel instruction.instruction⟩

def actionModel : V.Action → Action
  | .Next => .next
  | .WriteVirtual destination value => .writeVirtual destination.val (valueModel value)
  | .WritePredicate destination value => .writePredicate destination.val value
  | .Jump target => .jump target.val
  | .Halt => .halt
  | .Load destination destinationBits bytes signed pointer =>
      .load destination.val destinationBits.val bytes.val signed (pointerModel pointer)
  | .Store bytes pointer value => .store bytes.val (pointerModel pointer) value.val

def idsModel (ids : V.ThreadIds) : ThreadIds where
  tid := fun axis => (ids.tid.val[axis]?.getD 0#u32).val
  ctaid := fun axis => (ids.ctaid.val[axis]?.getD 0#u32).val
  ntid := fun axis => (ids.ntid.val[axis]?.getD 0#u32).val
  nctaid := fun axis => (ids.nctaid.val[axis]?.getD 0#u32).val
  lane := ids.lane.val

def undefinedBits (entries : List (U16 × U64)) (index : Nat) : Nat :=
  entries.foldl (fun mask entry => if entry.1.val = index then mask ||| entry.2.val else mask) 0

def wordModel (registers : V.Registers) (index : Nat) : Option Word :=
  if index < registers.raw.regs.val.length ∧ index < registers.pointers.val.length ∧
      index / 64 < registers.raw.written.val.length then
    if ((registers.raw.written.val[index / 64]?.getD 0#u64).val / 2 ^ (index % 64)) % 2 = 1 then
      some {
        bits := (registers.raw.regs.val[index]?.getD 0#u64).val
        undefined := undefinedBits registers.raw.vund.val index
        pointer := (registers.pointers.val[index]?.getD none).map pointerModel
      }
    else none
  else none

def predicateModel (registers : V.Registers) (index : Nat) : Option Bool :=
  if index / 64 < registers.raw.preds.val.length ∧ index / 64 < registers.raw.pdef.val.length then
    if ((registers.raw.pdef.val[index / 64]?.getD 0#u64).val / 2 ^ (index % 64)) % 2 = 1 then
      some <| ((registers.raw.preds.val[index / 64]?.getD 0#u64).val / 2 ^ (index % 64)) % 2 = 1
    else none
  else none

def bankModel (registers : V.Registers) : RegisterBank :=
  ⟨wordModel registers, predicateModel registers⟩

def outcomeModel {α β ε : Type} (model : α → β) : core.result.Result α ε → Option β
  | .Ok value => some (model value)
  | .Err _ => none

end ValueFlow.Source

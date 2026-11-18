/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.Provenance
import ValueFlow.Arithmetic

set_option autoImplicit false

namespace ValueFlow

inductive Value where
  | scalar (bits : Nat)
  | pointer (value : Pointer)
  deriving DecidableEq

structure Word where
  bits : Nat
  undefined : Nat
  pointer : Option Pointer

structure RegisterBank where
  virtual : Nat → Option Word
  predicate : Nat → Option Bool

inductive Special where
  | thread | block | blockSize | gridSize | lane
  deriving DecidableEq

structure ThreadIds where
  tid : Nat → Nat
  ctaid : Nat → Nat
  ntid : Nat → Nat
  nctaid : Nat → Nat
  lane : Nat

inductive Operand where
  | immediate (value : Nat)
  | virtual (index : Nat)
  | pointer (value : Pointer)
  | special (kind : Special) (axis : Nat)

structure Guard where
  predicate : Option Nat
  negated : Bool

structure Address where
  base : Operand
  index : Operand
  offset : Int

inductive Binary where
  | add | subtract | bitAnd | bitOr | bitXor | multiply | shiftLeft | shiftRight
  deriving DecidableEq

inductive Comparison where
  | equal | notEqual | lessUnsigned | lessEqualUnsigned | greaterUnsigned | greaterEqualUnsigned
  | lessSigned | lessEqualSigned | greaterSigned | greaterEqualSigned
  deriving DecidableEq

inductive Instruction where
  | move (width destination : Nat) (source : Operand)
  | binary (width destination : Nat) (operation : Binary) (left right : Operand)
  | multiplyWide (destination : Nat) (left right : Operand)
  | convertUnsigned (destination : Nat) (source : Operand) (fromBits toBits : Nat)
  | compare (width destination : Nat) (comparison : Comparison) (left right : Operand)
  | branch (target : Nat)
  | load (destination destinationBits bytes : Nat) (signed : Bool) (address : Address)
  | store (bytes : Nat) (address : Address) (source : Operand)
  | nop | exit

structure GuardedInstruction where
  guard : Guard
  instruction : Instruction

inductive Action where
  | next
  | writeVirtual (destination : Nat) (value : Value)
  | writePredicate (destination : Nat) (value : Bool)
  | jump (target : Nat)
  | halt
  | load (destination destinationBits bytes : Nat) (signed : Bool) (pointer : Pointer)
  | store (bytes : Nat) (pointer : Pointer) (value : Nat)
  deriving DecidableEq

def supportedBits (bits : Nat) : Bool :=
  bits = 8 || bits = 16 || bits = 32 || bits = 64

def scalarBits (bits : Nat) : Bool := bits = 32 || bits = 64

def truncate (bits value : Nat) : Nat := value % 2 ^ bits

def scalarValue : Value → Option Nat
  | .scalar value => some value
  | .pointer _ => none

def readWord (bits : Nat) (word : Word) : Option Value :=
  if supportedBits bits && word.undefined &&& (2 ^ bits - 1) = 0 then
    match word.pointer with
    | some pointer => if bits = 64 then some (.pointer pointer) else none
    | none => some (.scalar (truncate bits word.bits))
  else none

def readGuard (bank : RegisterBank) (guard : Guard) : Option Bool := do
  let value ← match guard.predicate with
    | none => some true
    | some index => bank.predicate index
  pure (value != guard.negated)

def readSpecial (ids : ThreadIds) (special : Special) (axis : Nat) : Option Nat :=
  match special with
  | .lane => some ids.lane
  | .thread => if axis < 3 then some (ids.tid axis) else none
  | .block => if axis < 3 then some (ids.ctaid axis) else none
  | .blockSize => if axis < 3 then some (ids.ntid axis) else none
  | .gridSize => if axis < 3 then some (ids.nctaid axis) else none

def readOperand (bank : RegisterBank) (ids : ThreadIds) (bits : Nat)
    (operand : Operand) : Option Value :=
  if supportedBits bits then
    match operand with
    | .immediate value => some (.scalar (truncate bits value))
    | .virtual index => bank.virtual index >>= readWord bits
    | .pointer pointer => if bits = 64 then some (.pointer pointer) else none
    | .special kind axis => (readSpecial ids kind axis).map fun value =>
        .scalar (truncate bits value)
  else none

def scalarBinary (operation : Binary) (bits left right : Nat) : Option Nat :=
  if scalarBits bits then
    let left := truncate bits left
    let right := truncate bits right
    some <| truncate bits <| match operation with
      | .add => left + right
      | .subtract => left + 2 ^ bits - right
      | .bitAnd => left &&& right
      | .bitOr => left ||| right
      | .bitXor => left ^^^ right
      | .multiply => left * right
      | .shiftLeft => if right < bits then left * 2 ^ right else 0
      | .shiftRight => if right < bits then left / 2 ^ right else 0
  else none

def offsetPointer (pointer : Pointer) (amount : Int) : Option Pointer :=
  let offset := (pointer.offset : Int) + amount
  if 0 ≤ offset ∧ offset < 2 ^ 64 then
    some { pointer with offset := offset.toNat }
  else none

def binary (operation : Binary) (bits : Nat) (left right : Value) : Option Value :=
  match left, right with
  | .scalar left, .scalar right => (scalarBinary operation bits left right).map .scalar
  | .pointer pointer, .scalar amount =>
      if bits = 64 then
        match operation with
        | .add => (offsetPointer pointer amount).map .pointer
        | .subtract => (offsetPointer pointer (-(amount : Int))).map .pointer
        | _ => none
      else none
  | .scalar amount, .pointer pointer =>
      if bits = 64 ∧ operation = .add then (offsetPointer pointer amount).map .pointer else none
  | .pointer _, .pointer _ => none

def comparisonValue (comparison : Comparison) (left right : Nat)
    (signedLeft signedRight : Int) : Bool :=
  match comparison with
  | .equal => left = right
  | .notEqual => left != right
  | .lessUnsigned => left < right
  | .lessEqualUnsigned => left ≤ right
  | .greaterUnsigned => left > right
  | .greaterEqualUnsigned => left ≥ right
  | .lessSigned => signedLeft < signedRight
  | .lessEqualSigned => signedLeft ≤ signedRight
  | .greaterSigned => signedLeft > signedRight
  | .greaterEqualSigned => signedLeft ≥ signedRight

def compare (comparison : Comparison) (bits left right : Nat) : Option Bool :=
  if scalarBits bits then
    let left := truncate bits left
    let right := truncate bits right
    some (comparisonValue comparison left right (integerSigned bits left) (integerSigned bits right))
  else none

def address (bank : RegisterBank) (ids : ThreadIds) (address : Address) : Option Pointer := do
  let base ← readOperand bank ids 64 address.base
  let index ← readOperand bank ids 64 address.index
  let pointer ← match ← binary .add 64 base index with
    | .pointer pointer => some pointer
    | .scalar _ => none
  offsetPointer pointer address.offset

def evaluateInstruction (bank : RegisterBank) (ids : ThreadIds) : Instruction → Option Action
  | .nop => some .next
  | .exit => some .halt
  | .branch target => some (.jump target)
  | .move bits destination source => do
      pure (.writeVirtual destination (← readOperand bank ids bits source))
  | .binary bits destination operation left right => do
      let left ← readOperand bank ids bits left
      let right ← readOperand bank ids bits right
      pure (.writeVirtual destination (← binary operation bits left right))
  | .multiplyWide destination left right => do
      let left ← scalarValue (← readOperand bank ids 32 left)
      let right ← scalarValue (← readOperand bank ids 32 right)
      pure (.writeVirtual destination (.scalar (left * right)))
  | .convertUnsigned destination source fromBits toBits => do
      let value ← readOperand bank ids fromBits source
      let converted ← match value with
        | .scalar value => if supportedBits toBits then
            some (.scalar (truncate toBits value)) else none
        | .pointer pointer => if toBits = 64 then some (.pointer pointer) else none
      pure (.writeVirtual destination converted)
  | .compare bits destination comparison left right => do
      let left ← scalarValue (← readOperand bank ids bits left)
      let right ← scalarValue (← readOperand bank ids bits right)
      pure (.writePredicate destination (← compare comparison bits left right))
  | .load destination destinationBits bytes signed location => do
      if supportedWidth bytes ∧ scalarBits destinationBits ∧ bytes * 8 ≤ destinationBits then
        pure (.load destination destinationBits bytes signed (← address bank ids location))
      else none
  | .store bytes location source => do
      if supportedWidth bytes then
        let pointer ← address bank ids location
        let value ← scalarValue (← readOperand bank ids (bytes * 8) source)
        pure (.store bytes pointer value)
      else none

def evaluate (bank : RegisterBank) (ids : ThreadIds)
    (instruction : GuardedInstruction) : Option Action := do
  if ← readGuard bank instruction.guard then
    evaluateInstruction bank ids instruction.instruction
  else pure .next

theorem offset_pointer_preserves_origin (pointer : Pointer) (amount : Int) (after : Pointer)
    (computed : offsetPointer pointer amount = some after) :
    after.allocation = pointer.allocation := by
  dsimp only [offsetPointer] at computed
  split at computed
  next =>
    simp only [Option.some.injEq] at computed
    subst after
    rfl
  next => simp_all

theorem offset_pointer_has_representable_offset (pointer : Pointer) (amount : Int) (after : Pointer)
    (computed : offsetPointer pointer amount = some after) : after.offset < 2 ^ 64 := by
  dsimp only [offsetPointer] at computed
  split at computed
  next condition =>
    simp only [Option.some.injEq] at computed
    subst after
    simp only
    omega
  next => simp_all

end ValueFlow

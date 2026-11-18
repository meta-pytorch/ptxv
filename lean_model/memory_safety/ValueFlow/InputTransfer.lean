/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.InputOperations

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option debug.skipKernelTC false

namespace ValueFlow

def AddressTracked (dependencies : Dependencies) (location : Address) : Prop :=
  operandDependency dependencies location.base = some false ∧
  operandDependency dependencies location.index = some false

def PairTracked (dependencies : Dependencies) (left right : Operand) (tainted : Bool) : Prop :=
  ∃ a b, operandDependency dependencies left = some a ∧
    operandDependency dependencies right = some b ∧ tainted = (a || b)

inductive InstructionTransfer (dependencies : Dependencies) : Instruction → Action → Bool → Prop where
  | next (instruction) : InstructionTransfer dependencies instruction .next false
  | jump (target otherTarget) :
      InstructionTransfer dependencies (.branch target) (.jump otherTarget) false
  | halt : InstructionTransfer dependencies .exit .halt false
  | move (bits destination source otherDestination value tainted)
      (tracked : operandDependency dependencies source = some tainted)
      (pointer : PointerIndependent tainted value) :
      InstructionTransfer dependencies (.move bits destination source)
        (.writeVirtual otherDestination value) tainted
  | binary (bits destination operation left right otherDestination value tainted)
      (tracked : PairTracked dependencies left right tainted)
      (pointer : PointerIndependent tainted value) :
      InstructionTransfer dependencies (.binary bits destination operation left right)
        (.writeVirtual otherDestination value) tainted
  | multiplyWide (destination left right otherDestination value tainted)
      (tracked : PairTracked dependencies left right tainted)
      (pointer : PointerIndependent tainted value) :
      InstructionTransfer dependencies (.multiplyWide destination left right)
        (.writeVirtual otherDestination value) tainted
  | convert (destination source fromBits toBits otherDestination value tainted)
      (tracked : operandDependency dependencies source = some tainted)
      (pointer : PointerIndependent tainted value) :
      InstructionTransfer dependencies (.convertUnsigned destination source fromBits toBits)
        (.writeVirtual otherDestination value) tainted
  | compare (bits destination comparison left right otherDestination value tainted)
      (tracked : PairTracked dependencies left right tainted) :
      InstructionTransfer dependencies (.compare bits destination comparison left right)
        (.writePredicate otherDestination value) tainted
  | load (destination destinationBits bytes signed location otherDestination otherBits otherBytes otherSigned pointer tainted)
      (tracked : AddressTracked dependencies location) :
      InstructionTransfer dependencies (.load destination destinationBits bytes signed location)
        (.load otherDestination otherBits otherBytes otherSigned pointer) tainted
  | store (bytes location source otherBytes pointer value tainted)
      (address : AddressTracked dependencies location)
      (tracked : operandDependency dependencies source = some tainted) :
      InstructionTransfer dependencies (.store bytes location source)
        (.store otherBytes pointer value) tainted

structure TransferCondition (dependencies : Dependencies) (instruction : GuardedInstruction)
    (action : Action) (tainted : Bool) : Prop where
  guard : ∀ index, instruction.guard.predicate = some index → dependencies.predicate index = some false
  body : InstructionTransfer dependencies instruction.instruction action tainted

def ActionsRelated (tainted : Bool) (left right : Action) : Prop :=
  match left, right with
  | .writeVirtual destination value, .writeVirtual otherDestination otherValue =>
      destination = otherDestination ∧ ValuesRelated tainted value otherValue
  | .writePredicate destination value, .writePredicate otherDestination otherValue =>
      destination = otherDestination ∧ (tainted = false → value = otherValue)
  | .store bytes pointer value, .store otherBytes otherPointer otherValue =>
      bytes = otherBytes ∧ pointer = otherPointer ∧ (tainted = false → value = otherValue)
  | _, _ => left = right

theorem read_operand_input_success (dependencies : Dependencies) (left right : RegisterBank)
    (related : InputBanksRelated dependencies left right) (ids : ThreadIds) (bits : Nat)
    (operand : Operand) (tainted : Bool) (tracked : operandDependency dependencies operand = some tainted)
    (value : Value) (computed : readOperand left ids bits operand = some value) :
    ∃ other, readOperand right ids bits operand = some other ∧ ValuesRelated tainted value other := by
  have relation := read_operand_inputs dependencies left right related ids bits operand tainted tracked
  rw [computed] at relation
  cases result : readOperand right ids bits operand with
  | none => simp [result, OutcomesRelated] at relation
  | some other => exact ⟨other, rfl, by simpa only [result, OutcomesRelated] using relation⟩

theorem read_operand_input_exact (dependencies : Dependencies) (left right : RegisterBank)
    (related : InputBanksRelated dependencies left right) (ids : ThreadIds) (bits : Nat)
    (operand : Operand) (tracked : operandDependency dependencies operand = some false) :
    readOperand left ids bits operand = readOperand right ids bits operand :=
  (outcomes_related_false _ _).mp
    (read_operand_inputs dependencies left right related ids bits operand false tracked)

theorem input_move (dependencies : Dependencies) (left right : RegisterBank)
    (related : InputBanksRelated dependencies left right) (ids : ThreadIds)
    (bits destination otherDestination : Nat) (source : Operand) (value : Value) (tainted : Bool)
    (tracked : operandDependency dependencies source = some tainted)
    (computed : evaluateInstruction left ids (.move bits destination source) =
      some (.writeVirtual otherDestination value)) :
    ∃ other, evaluateInstruction right ids (.move bits destination source) = some other ∧
      ActionsRelated tainted (.writeVirtual otherDestination value) other := by
  simp only [evaluateInstruction, Option.bind_eq_bind, Option.pure_def, Option.bind_eq_some_iff, Option.some.injEq, Action.writeVirtual.injEq] at computed
  obtain ⟨sourceValue, read, destinationEq, valueEq⟩ := computed
  subst otherDestination sourceValue
  obtain ⟨otherValue, otherRead, valueRelated⟩ :=
    read_operand_input_success dependencies left right related ids bits source tainted tracked value read
  refine ⟨.writeVirtual destination otherValue, ?_, ?_⟩
  · simp only [evaluateInstruction, otherRead, Option.bind_eq_bind, Option.bind_some, Option.pure_def]
  · exact ⟨rfl, valueRelated⟩

theorem actions_related_exact (action : Action) : ActionsRelated false action action := by
  cases action <;> simp [ActionsRelated, ValuesRelated]

theorem pair_tracked_false (dependencies : Dependencies) (left right : Operand)
    (tracked : PairTracked dependencies left right false) :
    operandDependency dependencies left = some false ∧ operandDependency dependencies right = some false := by
  obtain ⟨a, b, leftTracked, rightTracked, neither⟩ := tracked
  cases a <;> cases b
  · exact ⟨leftTracked, rightTracked⟩
  all_goals simp at neither

theorem scalar_value_some (value : Value) (bits : Nat) :
    scalarValue value = some bits ↔ value = .scalar bits := by
  cases value <;> simp [scalarValue]

theorem read_scalar_input_success (dependencies : Dependencies) (left right : RegisterBank)
    (related : InputBanksRelated dependencies left right) (ids : ThreadIds) (bits : Nat)
    (operand : Operand) (tainted : Bool) (tracked : operandDependency dependencies operand = some tainted)
    (value : Value) (scalar : Nat) (computed : readOperand left ids bits operand = some value)
    (extracted : scalarValue value = some scalar) :
    ∃ other, readOperand right ids bits operand = some (.scalar other) ∧
      (tainted = false → scalar = other) := by
  have valueEq := (scalar_value_some value scalar).mp extracted
  obtain ⟨otherValue, otherRead, valueRelated⟩ :=
    read_operand_input_success dependencies left right related ids bits operand tainted tracked value computed
  obtain ⟨other, otherEq, scalarEqual⟩ :=
    related_values_scalar tainted value otherValue valueRelated scalar valueEq
  exact ⟨other, by simpa only [otherEq] using otherRead, scalarEqual⟩

theorem input_binary (dependencies : Dependencies) (left right : RegisterBank)
    (related : InputBanksRelated dependencies left right) (ids : ThreadIds)
    (bits destination otherDestination : Nat) (operation : Binary) (sourceLeft sourceRight : Operand)
    (value : Value) (tainted : Bool) (tracked : PairTracked dependencies sourceLeft sourceRight tainted)
    (pointer : PointerIndependent tainted value)
    (computed : evaluateInstruction left ids (.binary bits destination operation sourceLeft sourceRight) =
      some (.writeVirtual otherDestination value)) :
    ∃ other, evaluateInstruction right ids (.binary bits destination operation sourceLeft sourceRight) = some other ∧
      ActionsRelated tainted (.writeVirtual otherDestination value) other := by
  simp only [evaluateInstruction, Option.bind_eq_bind, Option.pure_def, Option.bind_eq_some_iff,
    Option.some.injEq, Action.writeVirtual.injEq] at computed
  obtain ⟨a, aRead, b, bRead, result, produced, destinationEq, resultEq⟩ := computed
  subst otherDestination result
  obtain ⟨aTaint, bTaint, aTracked, bTracked, joined⟩ := tracked
  subst tainted
  obtain ⟨otherA, otherARead, aRelated⟩ :=
    read_operand_input_success dependencies left right related ids bits sourceLeft aTaint aTracked a aRead
  obtain ⟨otherB, otherBRead, bRelated⟩ :=
    read_operand_input_success dependencies left right related ids bits sourceRight bTaint bTracked b bRead
  obtain ⟨otherResult, otherProduced, resultRelated⟩ :=
    binary_inputs operation bits a b otherA otherB aTaint bTaint aRelated bRelated value produced pointer
  refine ⟨.writeVirtual destination otherResult, ?_, ⟨rfl, resultRelated⟩⟩
  simp only [evaluateInstruction, otherARead, otherBRead, otherProduced,
    Option.bind_eq_bind, Option.bind_some, Option.pure_def]

theorem input_load (dependencies : Dependencies) (left right : RegisterBank)
    (related : InputBanksRelated dependencies left right) (ids : ThreadIds)
    (destination destinationBits bytes : Nat) (signed : Bool) (location : Address)
    (action : Action) (tainted : Bool) (tracked : AddressTracked dependencies location)
    (computed : evaluateInstruction left ids (.load destination destinationBits bytes signed location) = some action) :
    ∃ other, evaluateInstruction right ids (.load destination destinationBits bytes signed location) = some other ∧
      ActionsRelated tainted action other := by
  have sameAddress := address_inputs dependencies left right related ids location tracked.1 tracked.2
  have same : evaluateInstruction left ids (.load destination destinationBits bytes signed location) =
      evaluateInstruction right ids (.load destination destinationBits bytes signed location) := by
    simp only [evaluateInstruction, sameAddress]
  refine ⟨action, same ▸ computed, ?_⟩
  simp only [evaluateInstruction, Option.bind_eq_bind, Option.pure_def, Option.bind_eq_some_iff] at computed
  split at computed <;> simp_all only [Option.some.injEq, reduceCtorEq, Option.bind_eq_some_iff]
  obtain ⟨pointer, _, equal⟩ := computed
  subst action
  rfl

theorem input_multiply_wide (dependencies : Dependencies) (left right : RegisterBank)
    (related : InputBanksRelated dependencies left right) (ids : ThreadIds)
    (destination otherDestination : Nat) (sourceLeft sourceRight : Operand)
    (value : Value) (tainted : Bool) (tracked : PairTracked dependencies sourceLeft sourceRight tainted)
    (computed : evaluateInstruction left ids (.multiplyWide destination sourceLeft sourceRight) =
      some (.writeVirtual otherDestination value)) :
    ∃ other, evaluateInstruction right ids (.multiplyWide destination sourceLeft sourceRight) = some other ∧
      ActionsRelated tainted (.writeVirtual otherDestination value) other := by
  simp only [evaluateInstruction, Option.bind_eq_bind, Option.pure_def, Option.bind_eq_some_iff,
    Option.some.injEq, Action.writeVirtual.injEq] at computed
  obtain ⟨aValue, aRead, a, aScalar, bValue, bRead, b, bScalar, destinationEq, valueEq⟩ := computed
  subst otherDestination value
  obtain ⟨aTaint, bTaint, aTracked, bTracked, joined⟩ := tracked
  subst tainted
  obtain ⟨otherA, otherARead, aEqual⟩ :=
    read_scalar_input_success dependencies left right related ids 32 sourceLeft aTaint aTracked aValue a aRead aScalar
  obtain ⟨otherB, otherBRead, bEqual⟩ :=
    read_scalar_input_success dependencies left right related ids 32 sourceRight bTaint bTracked bValue b bRead bScalar
  refine ⟨.writeVirtual destination (.scalar (otherA * otherB)), ?_, ?_⟩
  · simp [evaluateInstruction, otherARead, otherBRead, scalarValue]
  · refine ⟨rfl, ?_⟩
    cases joined : aTaint || bTaint with
    | true => exact ⟨a * b, otherA * otherB, rfl, rfl⟩
    | false =>
      have aFalse : aTaint = false := by cases aTaint <;> cases bTaint <;> simp_all
      have bFalse : bTaint = false := by cases aTaint <;> cases bTaint <;> simp_all
      rw [aEqual aFalse, bEqual bFalse]
      rfl

theorem input_compare (dependencies : Dependencies) (left right : RegisterBank)
    (related : InputBanksRelated dependencies left right) (ids : ThreadIds)
    (bits destination otherDestination : Nat) (comparison : Comparison) (sourceLeft sourceRight : Operand)
    (value : Bool) (tainted : Bool) (tracked : PairTracked dependencies sourceLeft sourceRight tainted)
    (computed : evaluateInstruction left ids (.compare bits destination comparison sourceLeft sourceRight) =
      some (.writePredicate otherDestination value)) :
    ∃ other, evaluateInstruction right ids (.compare bits destination comparison sourceLeft sourceRight) = some other ∧
      ActionsRelated tainted (.writePredicate otherDestination value) other := by
  simp only [evaluateInstruction, Option.bind_eq_bind, Option.pure_def, Option.bind_eq_some_iff,
    Option.some.injEq, Action.writePredicate.injEq] at computed
  obtain ⟨aValue, aRead, a, aScalar, bValue, bRead, b, bScalar, result, produced, destinationEq, valueEq⟩ := computed
  subst otherDestination result
  obtain ⟨aTaint, bTaint, aTracked, bTracked, joined⟩ := tracked
  subst tainted
  obtain ⟨otherA, otherARead, aEqual⟩ :=
    read_scalar_input_success dependencies left right related ids bits sourceLeft aTaint aTracked aValue a aRead aScalar
  obtain ⟨otherB, otherBRead, bEqual⟩ :=
    read_scalar_input_success dependencies left right related ids bits sourceRight bTaint bTracked bValue b bRead bScalar
  have supported : scalarBits bits = true := by
    cases support : scalarBits bits <;> simp_all [compare]
  obtain ⟨otherValue, otherProduced⟩ := compare_total comparison bits otherA otherB supported
  refine ⟨.writePredicate destination otherValue, ?_, rfl, ?_⟩
  · simp [evaluateInstruction, otherARead, otherBRead, scalarValue, otherProduced]
  · intro neither
    have aFalse : aTaint = false := by cases aTaint <;> cases bTaint <;> simp_all
    have bFalse : bTaint = false := by cases aTaint <;> cases bTaint <;> simp_all
    rw [← aEqual aFalse, ← bEqual bFalse, produced] at otherProduced
    exact Option.some.inj otherProduced

theorem input_store (dependencies : Dependencies) (left right : RegisterBank)
    (related : InputBanksRelated dependencies left right) (ids : ThreadIds)
    (bytes otherBytes : Nat) (location : Address) (source : Operand)
    (pointer : Pointer) (value : Nat) (tainted : Bool)
    (addressTracked : AddressTracked dependencies location)
    (tracked : operandDependency dependencies source = some tainted)
    (computed : evaluateInstruction left ids (.store bytes location source) =
      some (.store otherBytes pointer value)) :
    ∃ other, evaluateInstruction right ids (.store bytes location source) = some other ∧
      ActionsRelated tainted (.store otherBytes pointer value) other := by
  simp only [evaluateInstruction] at computed
  split at computed
  next supported =>
    simp only [Option.bind_eq_bind, Option.pure_def, Option.bind_eq_some_iff,
      Option.some.injEq, Action.store.injEq] at computed
    obtain ⟨capturedPointer, addressRead, sourceValue, sourceRead, scalar, sourceScalar,
      bytesEq, pointerEq, valueEq⟩ := computed
    subst otherBytes capturedPointer scalar
    obtain ⟨otherValue, otherRead, valueEqual⟩ :=
      read_scalar_input_success dependencies left right related ids (bytes * 8) source tainted tracked
        sourceValue value sourceRead sourceScalar
    have sameAddress := address_inputs dependencies left right related ids location addressTracked.1 addressTracked.2
    refine ⟨.store bytes pointer otherValue, ?_, rfl, rfl, valueEqual⟩
    simp [evaluateInstruction, supported, ← sameAddress, addressRead, otherRead, scalarValue]
  next unsupported => simp at computed

theorem input_convert (dependencies : Dependencies) (left right : RegisterBank)
    (related : InputBanksRelated dependencies left right) (ids : ThreadIds)
    (destination otherDestination fromBits toBits : Nat) (source : Operand)
    (value : Value) (tainted : Bool)
    (tracked : operandDependency dependencies source = some tainted)
    (pointerIndependent : PointerIndependent tainted value)
    (computed : evaluateInstruction left ids (.convertUnsigned destination source fromBits toBits) =
      some (.writeVirtual otherDestination value)) :
    ∃ other, evaluateInstruction right ids (.convertUnsigned destination source fromBits toBits) = some other ∧
      ActionsRelated tainted (.writeVirtual otherDestination value) other := by
  by_cases untainted : tainted = false
  · subst tainted
    have sameRead := read_operand_input_exact dependencies left right related ids fromBits source tracked
    have same : evaluateInstruction left ids (.convertUnsigned destination source fromBits toBits) =
        evaluateInstruction right ids (.convertUnsigned destination source fromBits toBits) := by
      simp only [evaluateInstruction, sameRead]
    exact ⟨_, same ▸ computed, actions_related_exact _⟩
  · have taintTrue : tainted = true := by cases tainted <;> simp_all
    subst tainted
    obtain ⟨result, resultEq⟩ :=
      (show ∃ result, value = .scalar result by simpa only [PointerIndependent, Bool.true_eq_false, false_or]
        using pointerIndependent)
    subst value
    simp only [evaluateInstruction, Option.bind_eq_bind, Option.bind_eq_some_iff] at computed
    obtain ⟨sourceValue, sourceRead, remainder⟩ := computed
    cases sourceValue with
    | pointer pointer =>
      simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def] at remainder
      by_cases width : toBits = 64 <;> simp [width] at remainder
    | scalar scalar =>
      simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def] at remainder
      by_cases supported : supportedBits toBits = true
      · simp only [supported, ite_true, Option.some.injEq,
          Action.writeVirtual.injEq, Value.scalar.injEq] at remainder
        obtain ⟨destinationEq, resultEq⟩ := remainder
        subst otherDestination result
        obtain ⟨otherScalar, otherRead, _⟩ :=
          read_scalar_input_success dependencies left right related ids fromBits source true tracked
            (.scalar scalar) scalar sourceRead rfl
        refine ⟨.writeVirtual destination (.scalar (truncate toBits otherScalar)), ?_, rfl, ?_⟩
        · simp [evaluateInstruction, otherRead, supported]
        · exact ⟨truncate toBits scalar, truncate toBits otherScalar, rfl, rfl⟩
      · simp only [supported, ite_false, Option.bind_none, reduceCtorEq] at remainder

theorem instruction_transfer_inputs (dependencies : Dependencies) (left right : RegisterBank)
    (related : InputBanksRelated dependencies left right) (ids : ThreadIds)
    (instruction : Instruction) (action : Action) (tainted : Bool)
    (transferred : InstructionTransfer dependencies instruction action tainted)
    (computed : evaluateInstruction left ids instruction = some action) :
    ∃ other, evaluateInstruction right ids instruction = some other ∧ ActionsRelated tainted action other := by
  cases transferred with
  | next instruction =>
      have same := evaluate_instruction_next left ids instruction computed
      subst instruction
      exact ⟨.next, rfl, rfl⟩
  | jump target otherTarget =>
      simp only [evaluateInstruction, Option.some.injEq, Action.jump.injEq] at computed
      subst otherTarget
      exact ⟨_, rfl, rfl⟩
  | halt => exact ⟨_, rfl, rfl⟩
  | move bits destination source otherDestination value tainted tracked pointer =>
      exact input_move dependencies left right related ids bits destination otherDestination source value tainted tracked computed
  | binary bits destination operation a b otherDestination value tainted tracked pointer =>
      exact input_binary dependencies left right related ids bits destination otherDestination operation a b value tainted tracked pointer computed
  | multiplyWide destination a b otherDestination value tainted tracked pointer =>
      exact input_multiply_wide dependencies left right related ids destination otherDestination a b value tainted tracked computed
  | convert destination source fromBits toBits otherDestination value tainted tracked pointer =>
      exact input_convert dependencies left right related ids destination otherDestination fromBits toBits source value tainted tracked pointer computed
  | compare bits destination comparison a b otherDestination value tainted tracked =>
      exact input_compare dependencies left right related ids bits destination otherDestination comparison a b value tainted tracked computed
  | load destination destinationBits bytes signed location otherDestination otherBits otherBytes otherSigned pointer tainted tracked =>
      exact input_load dependencies left right related ids destination destinationBits bytes signed location _ tainted tracked computed
  | store bytes location source otherBytes pointer value tainted address tracked =>
      exact input_store dependencies left right related ids bytes otherBytes location source pointer value tainted address tracked computed

theorem evaluate_inputs (dependencies : Dependencies) (left right : RegisterBank)
    (related : InputBanksRelated dependencies left right) (ids : ThreadIds)
    (instruction : GuardedInstruction) (action : Action) (tainted : Bool)
    (transferred : TransferCondition dependencies instruction action tainted)
    (computed : evaluate left ids instruction = some action) :
    ∃ other, evaluate right ids instruction = some other ∧ ActionsRelated tainted action other := by
  have sameGuard := read_guard_inputs dependencies left right related instruction.guard transferred.guard
  simp only [evaluate, Option.bind_eq_bind, Option.bind_eq_some_iff] at computed
  obtain ⟨enabled, guardRead, rest⟩ := computed
  cases enabled with
  | false =>
      simp only [Bool.false_eq_true, ite_false, Option.pure_def, Option.some.injEq] at rest
      subst action
      refine ⟨.next, ?_, rfl⟩
      simp [evaluate, ← sameGuard, guardRead]
  | true =>
      simp only [ite_true] at rest
      obtain ⟨other, otherComputed, actions⟩ :=
        instruction_transfer_inputs dependencies left right related ids instruction.instruction action tainted transferred.body rest
      refine ⟨other, ?_, actions⟩
      simp [evaluate, ← sameGuard, guardRead, otherComputed]

end ValueFlow

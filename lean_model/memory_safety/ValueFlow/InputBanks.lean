/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.LayoutIndependence

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option debug.skipKernelTC false

namespace ValueFlow

structure Dependencies where
  virtual : Nat → Option Bool
  predicate : Nat → Option Bool

def operandDependency (dependencies : Dependencies) : Operand → Option Bool
  | .virtual index => dependencies.virtual index
  | _ => some false

def ValuesRelated (tainted : Bool) (left right : Value) : Prop :=
  if tainted then ∃ a b, left = .scalar a ∧ right = .scalar b else left = right

def OutcomesRelated (tainted : Bool) (left right : Option Value) : Prop :=
  match left, right with
  | none, none => True
  | some left, some right => ValuesRelated tainted left right
  | _, _ => False

structure InputBanksRelated (dependencies : Dependencies) (left right : RegisterBank) : Prop where
  virtuals : ∀ index tainted, dependencies.virtual index = some tainted →
    ∃ l r, left.virtual index = some l ∧ right.virtual index = some r ∧
      l.undefined = 0 ∧ r.undefined = 0 ∧
      (if tainted then l.pointer = none ∧ r.pointer = none else WordRelated l r)
  predicates : ∀ index tainted, dependencies.predicate index = some tainted →
    ∃ l r, left.predicate index = some l ∧ right.predicate index = some r ∧
      (tainted = false → l = r)

theorem outcomes_related_exact (value : Option Value) : OutcomesRelated false value value := by
  cases value <;> simp [OutcomesRelated, ValuesRelated]

theorem read_operand_inputs (dependencies : Dependencies) (left right : RegisterBank)
    (related : InputBanksRelated dependencies left right) (ids : ThreadIds)
    (bits : Nat) (operand : Operand) (tainted : Bool)
    (tracked : operandDependency dependencies operand = some tainted) :
    OutcomesRelated tainted (readOperand left ids bits operand) (readOperand right ids bits operand) := by
  cases operand with
  | immediate value | pointer value | special kind axis =>
    simp only [operandDependency, Option.some.injEq] at tracked
    subst tainted
    exact outcomes_related_exact _
  | virtual index =>
    obtain ⟨l, r, leftWord, rightWord, leftDefined, rightDefined, contents⟩ :=
      related.virtuals index tainted tracked
    cases tainted with
    | false =>
      have same := read_word_rebase l r contents bits
      simp only [readOperand]
      split
      · simp only [leftWord, rightWord, Option.bind_eq_bind, Option.bind_some, same]
        exact outcomes_related_exact _
      · exact outcomes_related_exact _
    | true =>
      simp only [Bool.true_eq, ite_true] at contents
      simp only [readOperand]
      split
      · rename_i supported
        simp only [leftWord, rightWord, Option.bind_eq_bind, Option.bind_some,
          readWord, leftDefined, rightDefined, contents.1, contents.2,
          Nat.zero_and, Nat.zero_eq, and_self, ite_true]
        simp [OutcomesRelated, ValuesRelated, supported]
      · trivial

theorem outcomes_related_false (left right : Option Value) :
    OutcomesRelated false left right ↔ left = right := by
  cases left <;> cases right <;> simp [OutcomesRelated, ValuesRelated]

theorem address_inputs (dependencies : Dependencies) (left right : RegisterBank)
    (related : InputBanksRelated dependencies left right) (ids : ThreadIds) (location : Address)
    (base : operandDependency dependencies location.base = some false)
    (index : operandDependency dependencies location.index = some false) :
    address left ids location = address right ids location := by
  have baseEq := (outcomes_related_false _ _).mp
    (read_operand_inputs dependencies left right related ids 64 location.base false base)
  have indexEq := (outcomes_related_false _ _).mp
    (read_operand_inputs dependencies left right related ids 64 location.index false index)
  simp only [address, baseEq, indexEq]

theorem read_guard_inputs (dependencies : Dependencies) (left right : RegisterBank)
    (related : InputBanksRelated dependencies left right) (guard : Guard)
    (independent : ∀ index, guard.predicate = some index → dependencies.predicate index = some false) :
    readGuard left guard = readGuard right guard := by
  cases predicate : guard.predicate with
  | none => simp only [readGuard, predicate]
  | some index =>
    obtain ⟨l, r, leftValue, rightValue, same⟩ :=
      related.predicates index false (independent index predicate)
    have equal := same rfl
    subst r
    simp only [readGuard, predicate, leftValue, rightValue]

theorem related_values_scalar (tainted : Bool) (left right : Value)
    (related : ValuesRelated tainted left right) (scalar : Nat) (leftScalar : left = .scalar scalar) :
    ∃ other, right = .scalar other ∧ (tainted = false → scalar = other) := by
  subst left
  cases tainted with
  | false => simp only [ValuesRelated, Bool.false_eq_true, ite_false] at related
             exact ⟨scalar, related.symm, fun _ => rfl⟩
  | true =>
    obtain ⟨a, b, equal, rightEq⟩ := related
    exact ⟨b, rightEq, by simp⟩

theorem related_values_pointer (tainted : Bool) (left right : Value)
    (related : ValuesRelated tainted left right) (pointer : Pointer) (leftPointer : left = .pointer pointer) :
    tainted = false ∧ right = .pointer pointer := by
  subst left
  cases tainted with
  | false => exact ⟨rfl, related.symm⟩
  | true => obtain ⟨a, b, impossible, _⟩ := related
            cases impossible

end ValueFlow

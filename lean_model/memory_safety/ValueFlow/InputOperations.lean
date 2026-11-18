/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.InputBanks
import ValueFlow.Properties

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option debug.skipKernelTC false

namespace ValueFlow

theorem scalar_binary_total (operation : Binary) (bits left right : Nat)
    (supported : scalarBits bits = true) : ∃ result, scalarBinary operation bits left right = some result := by
  simp only [scalarBinary, supported, ite_true]
  exact ⟨_, rfl⟩

theorem binary_scalar_arguments (operation : Binary) (bits : Nat) (left right : Value)
    (result : Nat) (computed : binary operation bits left right = some (.scalar result)) :
    ∃ a b, left = .scalar a ∧ right = .scalar b ∧ scalarBinary operation bits a b = some result := by
  cases left with
  | scalar left =>
    cases right with
    | scalar right =>
      simp only [binary, Option.map_eq_some_iff] at computed
      obtain ⟨value, found, same⟩ := computed
      cases same
      exact ⟨left, right, rfl, rfl, found⟩
    | pointer pointer =>
      by_cases accepted : bits = 64 ∧ operation = .add <;>
        simp [binary, accepted, Option.map_eq_some_iff] at computed
  | pointer pointer =>
    cases right with
    | pointer right => simp [binary] at computed
    | scalar right =>
      by_cases width : bits = 64
      · cases operation <;> simp [binary, width, Option.map_eq_some_iff] at computed
      · simp [binary, width] at computed

def PointerIndependent (tainted : Bool) (value : Value) : Prop :=
  tainted = false ∨ ∃ scalar, value = .scalar scalar

theorem binary_inputs (operation : Binary) (bits : Nat) (left right otherLeft otherRight : Value)
    (leftTaint rightTaint : Bool) (leftRelated : ValuesRelated leftTaint left otherLeft)
    (rightRelated : ValuesRelated rightTaint right otherRight) (value : Value)
    (computed : binary operation bits left right = some value)
    (pointerIndependent : PointerIndependent (leftTaint || rightTaint) value) :
    ∃ otherValue, binary operation bits otherLeft otherRight = some otherValue ∧
      ValuesRelated (leftTaint || rightTaint) value otherValue := by
  cases joined : leftTaint || rightTaint with
  | false =>
    have leftFalse : leftTaint = false := by cases leftTaint <;> cases rightTaint <;> simp_all
    have rightFalse : rightTaint = false := by cases leftTaint <;> cases rightTaint <;> simp_all
    subst leftTaint rightTaint
    simp only [ValuesRelated, Bool.false_eq_true, ite_false] at leftRelated rightRelated
    subst otherLeft otherRight
    exact ⟨value, computed, rfl⟩
  | true =>
    have scalarResult : ∃ result, value = .scalar result := by
      simpa only [PointerIndependent, joined, Bool.true_eq_false, false_or] using pointerIndependent
    obtain ⟨result, equal⟩ := scalarResult
    subst value
    obtain ⟨a, b, leftScalar, rightScalar, produced⟩ :=
      binary_scalar_arguments operation bits left right result computed
    obtain ⟨otherA, otherLeftEq, _⟩ :=
      related_values_scalar leftTaint left otherLeft leftRelated a leftScalar
    obtain ⟨otherB, otherRightEq, _⟩ :=
      related_values_scalar rightTaint right otherRight rightRelated b rightScalar
    have supported : scalarBits bits = true := by
      cases support : scalarBits bits <;> simp_all [scalarBinary]
    obtain ⟨otherResult, otherComputed⟩ := scalar_binary_total operation bits otherA otherB supported
    refine ⟨.scalar otherResult, ?_, ?_⟩
    · simp only [otherLeftEq, otherRightEq, binary, otherComputed, Option.map_some]
    · exact ⟨result, otherResult, rfl, rfl⟩

theorem compare_total (comparison : Comparison) (bits left right : Nat)
    (supported : scalarBits bits = true) : ∃ result, compare comparison bits left right = some result := by
  simp only [compare, supported, ite_true]
  exact ⟨_, rfl⟩

theorem evaluate_instruction_next (bank : RegisterBank) (ids : ThreadIds) (instruction : Instruction)
    (computed : evaluateInstruction bank ids instruction = some .next) : instruction = .nop := by
  cases instruction <;>
    simp [evaluateInstruction, Option.bind_eq_some_iff] at computed ⊢
  obtain ⟨value, _, impossible⟩ := computed
  cases value <;> split at impossible <;> simp_all

end ValueFlow

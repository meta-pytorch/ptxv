/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicComponentPlan
import MemoryWrite.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite.AtomicComponentPlan

def componentValue : atomic_components.Component → Component
  | .First => .first
  | .Second => .second
  | .Third => .third
  | .Fourth => .fourth

def planValue (plan : atomic_components.Plan) : Plan :=
  ⟨plan.first, plan.second, plan.third, plan.fourth⟩

@[step] theorem rust_vector_plan (bytes : U8) :
    atomic_components.vector_plan bytes
      ⦃ result => result.map planValue = vectorPlan bytes.val ⦄ := by
  by_cases eight : bytes = 8#u8
  · subst bytes; simp [atomic_components.vector_plan, planValue, vectorPlan]
  · by_cases sixteen : bytes = 16#u8
    · subst bytes; simp [atomic_components.vector_plan, planValue, vectorPlan]
    · have notEight := UScalar.neq_to_neq_val.mp eight
      have notSixteen := UScalar.neq_to_neq_val.mp sixteen
      change bytes.val ≠ 8 at notEight
      change bytes.val ≠ 16 at notSixteen
      simp [atomic_components.vector_plan, vectorPlan, eight, sixteen, notEight, notSixteen]

@[step] theorem rust_pending (plan : atomic_components.Plan) (component : atomic_components.Component) :
    atomic_components.pending plan component
      ⦃ result => result = pending (planValue plan) (componentValue component) ⦄ := by
  cases component <;> simp [atomic_components.pending, pending, planValue, componentValue]

@[step] theorem rust_finish (plan : atomic_components.Plan) (component : atomic_components.Component) :
    atomic_components.finish plan component
      ⦃ result => result.map planValue = finish (planValue plan) (componentValue component) ⦄ := by
  cases plan with
  | mk a b c d =>
    cases component <;> cases a <;> cases b <;> cases c <;> cases d <;>
      simp [atomic_components.finish, atomic_components.pending,
        finish, pending, clear, planValue, componentValue]

@[step] theorem rust_retired (plan : atomic_components.Plan) :
    atomic_components.retired plan
      ⦃ result => result = retired (planValue plan) ⦄ := by
  cases plan with
  | mk a b c d =>
    cases a <;> cases b <;> cases c <;> cases d <;>
      simp [atomic_components.retired, retired, planValue]

@[step] theorem rust_byte_offset (component : atomic_components.Component) :
    atomic_components.byte_offset component
      ⦃ result => result.val = byteOffset (componentValue component) ⦄ := by
  cases component <;> simp [atomic_components.byte_offset, componentValue, byteOffset]

@[step] theorem rust_ticket_access_eligible (floatAdd : Bool) (bytes : U8) :
    atomic_components.ticket_access_eligible floatAdd bytes
      ⦃ result => result = ticketEligible floatAdd bytes.val ⦄ := by
  cases floatAdd with
  | false => simp [atomic_components.ticket_access_eligible, ticketEligible]
  | true =>
    by_cases eight : bytes = 8#u8
    · subst bytes; simp [atomic_components.ticket_access_eligible, ticketEligible]
    · by_cases sixteen : bytes = 16#u8
      · subst bytes; simp [atomic_components.ticket_access_eligible, ticketEligible]
      · have notEight := UScalar.neq_to_neq_val.mp eight
        have notSixteen := UScalar.neq_to_neq_val.mp sixteen
        change bytes.val ≠ 8 at notEight
        change bytes.val ≠ 16 at notSixteen
        simp [atomic_components.ticket_access_eligible, ticketEligible,
          eight, sixteen, notEight, notSixteen]

@[step] theorem rust_choices (plan : atomic_components.Plan) :
    atomic_components.choices plan
      ⦃ result => result.val.map componentValue = choices (planValue plan) ⦄ := by
  cases plan with
  | mk a b c d =>
    cases a <;> cases b <;> cases c <;> cases d <;>
      simp [atomic_components.choices, alloc.vec.Vec.with_capacity, alloc.vec.Vec.new,
        alloc.vec.Vec.push, U32.max_eq, choices, allComponents, pending, planValue, componentValue] <;> decide

@[step] theorem source_choices_preserve_future_enabledness (plan : atomic_components.Plan) :
    atomic_components.choices plan
      ⦃ result => (result.val.map componentValue).Nodup ∧
        ∀ component, component ∈ result.val.map componentValue ↔
          pending (planValue plan) component = true ⦄ := by
  step with rust_choices plan as ⟨result, projected⟩
  rw [projected]
  exact ⟨choices_no_duplicates _, choices_exactly_enabled _⟩

def frozenOperand (input : atomic_components.FrozenOperands) : Component → symbolic.Metadata
  | .first => input.first
  | .second => input.second
  | .third => input.third
  | .fourth => input.fourth

@[step] theorem rust_frozen_operand (input : atomic_components.FrozenOperands)
    (component : atomic_components.Component) :
    atomic_components.frozen_operand input component
      ⦃ source => source = frozenOperand input (componentValue component) ⦄ := by
  cases component <;> simp [atomic_components.frozen_operand, frozenOperand, componentValue]

@[step] theorem rust_frozen_address (input : atomic_components.FrozenOperands)
    (component : atomic_components.Component) :
    atomic_components.frozen_address input component
      ⦃ address => address.val =
        (input.address.val + byteOffset (componentValue component)) % (2^64) ⦄ := by
  unfold atomic_components.frozen_address
  step with rust_byte_offset component as ⟨offset, offsetValue⟩
  step as ⟨wide, wideValue⟩
  simp only [core.num.U64.wrapping_add_val_eq]
  rw [wideValue, U8.cast_U64_val_eq, offsetValue]
  simp [U64.size, U64.numBits]

@[step] theorem rust_begin_component (input : atomic_components.FrozenOperands)
    (plan : atomic_components.Plan) (component : atomic_components.Component) :
    atomic_components.begin_component input plan component
      ⦃ result =>
        result.map (fun selected => planValue selected.remaining) =
          finish (planValue plan) (componentValue component) ∧
        ∀ selected, result = some selected →
          selected.source = frozenOperand input (componentValue component) ∧
          selected.address.val =
            (input.address.val + byteOffset (componentValue component)) % (2^64) ⦄ := by
  unfold atomic_components.begin_component
  step with rust_finish plan component as ⟨next, nextValue⟩
  cases next with
  | none => simpa using nextValue
  | some remaining =>
    step with rust_frozen_address input component as ⟨address, addressValue⟩
    step with rust_frozen_operand input component as ⟨source, sourceValue⟩
    simp_all

end MemoryWrite.AtomicComponentPlan

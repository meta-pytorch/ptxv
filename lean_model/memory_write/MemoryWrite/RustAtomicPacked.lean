/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RustAtomicComponentTrace

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.AtomicPacked

@[step] theorem rust_plan : atomic_packed.plan ⦃ result =>
    AtomicComponentPlan.planValue result = ⟨true, true, false, false⟩ ⦄ := by
  simp [atomic_packed.plan, AtomicComponentPlan.planValue]

@[step] theorem rust_offset (component : atomic_components.Component) :
    atomic_packed.offset component ⦃ result => result.map UScalar.val =
      match component with
      | .First => some 0
      | .Second => some 2
      | _ => none ⦄ := by
  cases component <;> simp [atomic_packed.offset]

def selectedValue (step : atomic_components.Step) : Nat × symbolic.Metadata × AtomicComponentPlan.Plan :=
  (step.address.val, step.source, AtomicComponentPlan.planValue step.remaining)

@[step] theorem rust_begin_component (input : atomic_components.FrozenOperands)
    (plan : atomic_components.Plan) (component : atomic_components.Component) :
    atomic_packed.begin_component input plan component ⦃ result =>
      result.map selectedValue =
        match component with
        | .First => (AtomicComponentPlan.finish (AtomicComponentPlan.planValue plan) .first).map
            (fun remaining => (input.address.val % (2^64), input.first, remaining))
        | .Second => (AtomicComponentPlan.finish (AtomicComponentPlan.planValue plan) .second).map
            (fun remaining => ((input.address.val + 2) % (2^64), input.second, remaining))
        | _ => none ⦄ := by
  cases component <;> unfold atomic_packed.begin_component <;>
    simp only [atomic_packed.offset, bind_ok]
  · step with AtomicComponentPlan.rust_finish plan .First as ⟨next,nextValue⟩
    cases next with
    | none => simp_all [AtomicComponentPlan.componentValue]
    | some remaining =>
      step as ⟨wide,wideValue⟩
      step as ⟨address,addressValue⟩
      step with AtomicComponentPlan.rust_frozen_operand input .First as ⟨source,sourceValue⟩
      simp_all [selectedValue, AtomicComponentPlan.frozenOperand,
        AtomicComponentPlan.componentValue, core.num.U64.wrapping_add_val_eq, U64.size, U64.numBits]
  · step with AtomicComponentPlan.rust_finish plan .Second as ⟨next,nextValue⟩
    cases next with
    | none => simp_all [AtomicComponentPlan.componentValue]
    | some remaining =>
      step as ⟨wide,wideValue⟩
      step as ⟨address,addressValue⟩
      step with AtomicComponentPlan.rust_frozen_operand input .Second as ⟨source,sourceValue⟩
      simp_all [selectedValue, AtomicComponentPlan.frozenOperand,
        AtomicComponentPlan.componentValue, core.num.U64.wrapping_add_val_eq, U64.size, U64.numBits]
  · simp
  · simp

@[step] theorem rust_operand (word : U32) (component : atomic_components.Component) :
    atomic_packed.operand word component ⦃ result => result.map UScalar.val =
      match component with
      | .First => some (word.val % 65536)
      | .Second => some (word.val / 65536)
      | _ => none ⦄ := by
  cases component with
  | First =>
    have castValue : (UScalar.cast .U16 word).val = word.val % 65536 := by
      simp only [UScalar.cast, UScalar.val, BitVec.truncate_eq_setWidth, BitVec.toNat_setWidth]
      rfl
    simpa only [atomic_packed.operand, lift, bind_ok, WP.spec_ok,
      Option.map_some, Option.some.injEq] using castValue
  | Second =>
    unfold atomic_packed.operand
    step with U32.div_spec (x := word) (y := 65536#u32) (by decide) as ⟨high,highValue⟩
    step with UScalar.cast_inBounds_spec .U16 high (by scalar_tac)
      as ⟨half,halfValue⟩
    simp only [Option.map_some]
    congr 1
    scalar_tac
  | Third => simp [atomic_packed.operand]
  | Fourth => simp [atomic_packed.operand]

@[step] theorem rust_operand_defined (mask : U32) (component : atomic_components.Component) :
    atomic_packed.operand_defined mask component ⦃ result => result =
      match component with
      | .First => some (decide (mask.val % 65536 = 0))
      | .Second => some (decide (mask.val / 65536 = 0))
      | _ => none ⦄ := by
  unfold atomic_packed.operand_defined
  step with rust_operand mask component as ⟨half,halfValue⟩
  cases component <;> cases half <;> simp_all only [Option.map_none, Option.map_some,
    Option.some.injEq, reduceCtorEq] <;> simp_all <;> scalar_tac

@[step] theorem rust_ticket_access_eligible (packedBf16 : Bool) (bytes : U8) :
    atomic_packed.ticket_access_eligible packedBf16 bytes
      ⦃ result => result = (!packedBf16 || decide (bytes.val ≠ 4)) ⦄ := by
  cases packedBf16 with
  | false => simp [atomic_packed.ticket_access_eligible]
  | true =>
    by_cases four : bytes = 4#u8
    · subst bytes; simp [atomic_packed.ticket_access_eligible]
    · have notFour := UScalar.neq_to_neq_val.mp four
      change bytes.val ≠ 4 at notFour
      simp [atomic_packed.ticket_access_eligible, four, notFour]

@[step] theorem source_packed_ticket_forcing_is_excluded :
    atomic_packed.ticket_access_eligible true 4#u8 ⦃ result => result = false ⦄ := by
  simp [atomic_packed.ticket_access_eligible]

@[step] theorem rust_assemble (low high : U16) :
    atomic_packed.assemble low high ⦃ result => result.val = low.val + high.val * 65536 ⦄ := by
  unfold atomic_packed.assemble
  step with UScalar.cast_inBounds_spec .U32 low (by scalar_tac) as ⟨lo,loValue⟩
  step with UScalar.cast_inBounds_spec .U32 high (by scalar_tac) as ⟨hi,hiValue⟩
  step with U32.mul_spec (x := hi) (y := 65536#u32) (by scalar_tac) as ⟨upper,upperValue⟩
  step with U32.add_spec (x := lo) (y := upper) (by scalar_tac) as ⟨result,resultValue⟩
  scalar_tac

@[step] theorem source_selected_halves_reassemble_original (word : U32) :
    (do
      let low ← atomic_packed.operand word .First
      let high ← atomic_packed.operand word .Second
      match low, high with
      | some low, some high => atomic_packed.assemble low high
      | _, _ => fail .panic)
    ⦃ result => result = word ⦄ := by
  step with rust_operand word .First as ⟨low,lowValue⟩
  step with rust_operand word .Second as ⟨high,highValue⟩
  cases low with
  | none => simp at lowValue
  | some low =>
    cases high with
    | none => simp at highValue
    | some high =>
      simp only [Option.map_some, Option.some.injEq] at lowValue highValue
      step with rust_assemble low high as ⟨result,resultValue⟩
      scalar_tac

def recorded (before : atomic_packed.Returns) (component : atomic_components.Component)
    (old : U16) (undefined : Bool) : Option atomic_packed.Returns :=
  match component with
  | .First => some { before with low := old, low_undefined := undefined }
  | .Second => some { before with high := old, high_undefined := undefined }
  | _ => none

@[step] theorem rust_record (before : atomic_packed.Returns) (component : atomic_components.Component)
    (old : U16) (undefined : Bool) :
    atomic_packed.record before component old undefined ⦃ result =>
      result = recorded before component old undefined ⦄ := by
  cases component <;> simp [atomic_packed.record, recorded]

@[step] theorem rust_returned (returns : atomic_packed.Returns) :
    atomic_packed.returned returns ⦃ result =>
      result.val = returns.low.val + returns.high.val * 65536 ⦄ := by
  unfold atomic_packed.returned
  exact rust_assemble _ _

@[step] theorem rust_undefined_mask (returns : atomic_packed.Returns) :
    atomic_packed.undefined_mask returns ⦃ result => result.val =
      (if returns.low_undefined then 65535 else 0) +
      (if returns.high_undefined then 65535 else 0) * 65536 ⦄ := by
  unfold atomic_packed.undefined_mask
  cases low : returns.low_undefined <;> cases high : returns.high_undefined <;>
    (try simp only [Bool.false_eq_true,ite_true,ite_false]) <;>
    step with rust_assemble _ _ as ⟨result,resultValue⟩ <;>
    exact resultValue

@[step] theorem source_record_preserves_the_other_half (before : atomic_packed.Returns)
    (component : atomic_components.Component) (old : U16) (undefined : Bool) :
    atomic_packed.record before component old undefined ⦃ result =>
      match component with
      | .First => ∃ after, result = some after ∧ after.low = old ∧
          after.low_undefined = undefined ∧ after.high = before.high ∧
          after.high_undefined = before.high_undefined
      | .Second => ∃ after, result = some after ∧ after.high = old ∧
          after.high_undefined = undefined ∧ after.low = before.low ∧
          after.low_undefined = before.low_undefined
      | _ => result = none ⦄ := by
  step with rust_record before component old undefined as ⟨result,resultValue⟩
  cases component <;> simp [resultValue,recorded]

@[step] theorem source_both_return_orders_assemble_the_observed_halves
    (before : atomic_packed.Returns) (low high : U16) (lowUndefined highUndefined : Bool)
    (highFirst : Bool) :
    (do
      let first ← atomic_packed.record before (if highFirst then .Second else .First)
        (if highFirst then high else low) (if highFirst then highUndefined else lowUndefined)
      match first with
      | none => fail .panic
      | some first =>
        let second ← atomic_packed.record first (if highFirst then .First else .Second)
          (if highFirst then low else high) (if highFirst then lowUndefined else highUndefined)
        match second with
        | none => fail .panic
        | some second => do
          let value ← atomic_packed.returned second
          let mask ← atomic_packed.undefined_mask second
          ok (value,mask))
    ⦃ result => result.1.val = low.val + high.val * 65536 ∧
      result.2.val = (if lowUndefined then 65535 else 0) +
        (if highUndefined then 65535 else 0) * 65536 ⦄ := by
  cases highFirst <;>
    simp only [Bool.false_eq_true,ite_true,ite_false] <;>
    step with rust_record _ _ _ _ as ⟨first,firstValue⟩ <;>
    simp only [recorded] at firstValue <;>
    simp only [firstValue] <;>
    step with rust_record _ _ _ _ as ⟨second,secondValue⟩ <;>
    simp only [recorded] at secondValue <;>
    simp only [secondValue] <;>
    step with rust_returned _ as ⟨value,valueSpec⟩ <;>
    step with rust_undefined_mask _ as ⟨mask,maskSpec⟩ <;>
    simp_all

def sourcePlan : atomic_components.Plan := ⟨true,true,false,false⟩

theorem source_plan_is_the_packed_constructor : atomic_packed.plan = ok sourcePlan := by
  simp [atomic_packed.plan, sourcePlan]

open AtomicComponentPlan in
@[step] theorem source_every_enabled_packed_prefix_preserves_observations
    {Data : Type} {Finding : Type}
    (execute : Data → Nat → Component → Nat → Effect Data Finding)
    (environment : Data → Unit → Effect Data Finding)
    (world : World Data Finding)
    (schedule : List (Nat × atomic_components.Component)) (modelNext : Nat → Plan)
    (initial : world.plans = fun _ => planValue sourcePlan)
    (enabled : runPlans (fun _ => planValue sourcePlan) (requests schedule) = some modelNext) :
    sourceRunPlans (fun _ => sourcePlan) schedule ⦃ result =>
      ∃ sourceNext, result = some sourceNext ∧
      ∃ next, runWorld execute environment world (requests schedule) = some next ∧
        next.observed = (requests schedule).foldl
          (observe execute environment world.latched) world.observed ∧
        next.latched = world.latched ∧
        ∀ actor component, pending (next.plans actor) component =
          pending (planValue (sourceNext actor)) component ⦄ := by
  step with source_any_interleaved_prefix (fun _ => sourcePlan) schedule as ⟨result,resultValue⟩
  rw [enabled] at resultValue
  cases result with
  | none => simp at resultValue
  | some sourceNext =>
    simp only [Option.map_some, Option.some.injEq] at resultValue
    obtain ⟨next,runValue,observed,latched,future⟩ :=
      every_admitted_prefix_has_its_exact_returns_and_future_enabledness execute environment
        world (requests schedule) modelNext (by simpa only [initial] using enabled)
    refine ⟨sourceNext,rfl,next,runValue,observed,latched,?_⟩
    intro actor component
    rw [future, ← resultValue]

open AtomicComponentPlan in
@[step] theorem source_mixed_actor_half_schedule_is_admitted :
    sourceRunPlans (fun _ => sourcePlan)
      [(0,.First),(1,.First),(1,.Second),(0,.Second)] ⦃ result =>
      ∃ next, result = some next ∧ retired (planValue (next 0)) = true ∧
        retired (planValue (next 1)) = true ⦄ := by
  simp [sourceRunPlans, atomic_components.finish, atomic_components.pending, update,
    sourcePlan, planValue, retired]

end MemoryWrite.AtomicPacked

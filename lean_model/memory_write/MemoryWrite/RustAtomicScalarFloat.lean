/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RustAtomicScalar
import MemoryWrite.AtomicScalarFloat

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.ScalarFloat

@[step] theorem source_float_instruction_selects_the_lane_plan :
    atomic_scalar.plan true .AddF32Ftz 4#u8 true false false
      ⦃ result => result = some AtomicScalar.sourcePlan ⦄ := by
  simp [atomic_scalar.plan, atomic_scalar.supported_operation, AtomicScalar.sourcePlan]

def floatInput (source : symbolic.Metadata) (old : U32) : atomic_exec.AtomicInput :=
  ⟨.AddF32Ftz, 4#u8, UScalar.cast .U64 old, UScalar.cast .U64 source.captured, 0#u64⟩

@[step] theorem source_frozen_float_request_executes_model {F : Type}
    (implementation : atomic_exec.FloatingArithmetic F) (semantics : AtomicFloatSemantics)
    (contract : AtomicFloatContract implementation semantics) (source : symbolic.Metadata)
    (old : U32) :
    (do
      let request ← atomic_scalar.request .AddF32Ftz source
      let input ← atomic_exec.resolve_old request (UScalar.cast .U64 old)
      atomic_exec.execute implementation input)
      ⦃ result => result = atomicExecutionModel semantics (floatInput source old) ⦄ := by
  step with AtomicScalar.rust_frozen_request .AddF32Ftz source as
    ⟨request,operationValue,widthValue,sourceValue,compareValue⟩
  have inputValue : atomic_exec.resolve_old request (UScalar.cast .U64 old) =
      ok (floatInput source old) := by
    cases request
    simp_all [atomic_exec.resolve_old, floatInput]
    scalar_tac
  rw [inputValue]
  simp only [bind_ok]
  exact rust_atomic_execute_spec implementation semantics contract (floatInput source old)

@[step] theorem source_float_returns_the_old_word_and_writes_the_captured_sum {F : Type}
    (implementation : atomic_exec.FloatingArithmetic F) (semantics : AtomicFloatSemantics)
    (contract : AtomicFloatContract implementation semantics) (source : symbolic.Metadata)
    (old : U32) :
    (do
      let request ← atomic_scalar.request .AddF32Ftz source
      let input ← atomic_exec.resolve_old request (UScalar.cast .U64 old)
      atomic_exec.execute implementation input)
      ⦃ result => ∃ outcome, result = .Ok outcome ∧
        outcome.bytes = 4#u8 ∧ outcome.returned.val = old.val ∧
        outcome.replacement.val = (semantics.addF32Ftz old source.captured).val ∧
        outcome.did_write = true ∧ outcome.advance_epoch = true ∧
        outcome.return_words = 1#u8 ⦄ := by
  apply WP.spec_mono
    (source_frozen_float_request_executes_model implementation semantics contract source old)
  intro result resultValue
  have back (value : U32) : UScalar.cast .U32 (UScalar.cast .U64 value) = value := by
    scalar_tac
  simp only [atomicExecutionModel, atomicComputedModel, floatInput,
    AtomicSupported, ↓reduceIte, atomicFloatResult, back, atomicFinishModel,
    atomicIsCas, atomicIsSpin] at resultValue
  refine ⟨_, resultValue, rfl, ?_, ?_, rfl, rfl, rfl⟩ <;> scalar_tac

@[step] theorem source_every_distinct_float_lane_prefix_is_admitted
    (actors : List Nat) (plans : Nat → atomic_components.Plan) (distinct : actors.Nodup)
    (available : ∀ actor ∈ actors, plans actor = AtomicScalar.sourcePlan) :
    AtomicComponentPlan.sourceRunPlans plans (actors.map (fun actor => (actor,.First)))
      ⦃ result => ∃ next, result = some next ∧
        (fun actor => AtomicComponentPlan.planValue (next actor)) =
          claimed (fun actor => AtomicComponentPlan.planValue (plans actor)) actors ⦄ := by
  have ready : ∀ actor ∈ actors, AtomicComponentPlan.planValue (plans actor) = one := by
    intro actor member
    rw [available actor member]
    rfl
  have requestValue :
      AtomicComponentPlan.requests (actors.map (fun actor => (actor,atomic_components.Component.First))) =
        events actors := by
    simp [AtomicComponentPlan.requests, events, List.map_map,
      AtomicComponentPlan.componentValue]
  step with AtomicComponentPlan.source_any_interleaved_prefix plans
    (actors.map (fun actor => (actor,atomic_components.Component.First))) as ⟨result,projected⟩
  rw [requestValue, every_distinct_lane_order_is_admitted actors _ distinct ready] at projected
  cases result with
  | none => simp at projected
  | some next => exact ⟨next,rfl,Option.some.inj projected⟩

@[step] theorem source_contextual_float_plans_preserve_the_word_effect_trace
    {Context Finding : Type} (semantics : AtomicFloatSemantics)
    (address destination : Nat → Nat)
    (environment : State U32 → Context → AtomicComponentPlan.Effect (State U32) Finding)
    (world : AtomicComponentPlan.World (State U32) Finding)
    (plans : Nat → atomic_components.Plan)
    (actions : List (AtomicComponentPlan.Action Context))
    (distinct : (componentActors actions).Nodup) (first : firstActions actions)
    (initial : world.plans = fun actor => AtomicComponentPlan.planValue (plans actor))
    (available : ∀ actor ∈ componentActors actions, plans actor = AtomicScalar.sourcePlan) :
    AtomicComponentPlan.sourceRunPlans plans
      ((componentActors actions).map (fun actor => (actor,.First)))
      ⦃ result => ∃ sourceNext, result = some sourceNext ∧
        ∃ next, AtomicComponentPlan.runWorld
          (execute (fun word : U32 => word.val) (fun bits => ⟨BitVec.ofNat 32 bits⟩)
            semantics.addF32Ftz address destination) environment world actions = some next ∧
          next.observed = actions.foldl
            (AtomicComponentPlan.observe
              (execute (fun word : U32 => word.val) (fun bits => ⟨BitVec.ofNat 32 bits⟩)
                semantics.addF32Ftz address destination) environment world.latched) world.observed ∧
          next.latched = world.latched ∧
          ∀ actor component, AtomicComponentPlan.pending (next.plans actor) component =
            AtomicComponentPlan.pending (AtomicComponentPlan.planValue (sourceNext actor)) component ⦄ := by
  step with source_every_distinct_float_lane_prefix_is_admitted
    (componentActors actions) plans distinct available as ⟨result,sourceNext,sourceValue,projection⟩
  have ready : ∀ actor ∈ componentActors actions, world.plans actor = one := by
    intro actor member
    rw [initial]
    change AtomicComponentPlan.planValue (plans actor) = one
    rw [available actor member]
    rfl
  obtain ⟨next,runValue,observed,latched,future⟩ :=
    every_contextual_capture_prefix_preserves_observations_and_enabledness
      (fun word : U32 => word.val) (fun bits => ⟨BitVec.ofNat 32 bits⟩)
      semantics.addF32Ftz address destination environment world actions distinct first ready
  refine ⟨sourceNext,sourceValue,next,runValue,observed,latched,?_⟩
  intro actor component
  rw [congrFun projection actor]
  simpa only [initial] using future actor component

end MemoryWrite.ScalarFloat

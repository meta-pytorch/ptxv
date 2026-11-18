/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicLaneCoverage
import MemoryWrite.RustAtomicComponentTrace
import MemoryWrite.AtomicRefinement

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.AtomicScalar

def sourcePlan : atomic_components.Plan := ⟨true,false,false,false⟩

def wordOperation : atomic_exec.Operation → Bool
  | .AddF32Ftz => true
  | .Integer .Or | .Integer .And | .Integer .Xor | .Integer .Add | .Integer .Exch
  | .Integer .MaxU | .Integer .Inc | .Integer .Dec => true
  | _ => false

def eligible (ptx : Bool) (operation : atomic_exec.Operation) (bytes : Nat)
    (resultUsed predicate explicitNew : Bool) : Bool :=
  ptx && resultUsed && !predicate && !explicitNew && decide (bytes = 4) && wordOperation operation

@[step] theorem rust_plan (ptx : Bool) (operation : atomic_exec.Operation) (bytes : U8)
    (resultUsed predicate explicitNew : Bool) :
    atomic_scalar.plan ptx operation bytes resultUsed predicate explicitNew ⦃ result =>
      result = if eligible ptx operation bytes.val resultUsed predicate explicitNew
        then some sourcePlan else none ⦄ := by
  by_cases width : bytes = 4#u8
  · subst bytes
    cases operation with
    | Integer op =>
      cases op <;> cases ptx <;> cases resultUsed <;> cases predicate <;> cases explicitNew <;>
        simp [atomic_scalar.plan, atomic_scalar.supported_operation, eligible, wordOperation, sourcePlan]
    | AddF32Ftz | AddF64 | AddBf16 | AddBf16x2 =>
      cases ptx <;> cases resultUsed <;> cases predicate <;> cases explicitNew <;>
        simp [atomic_scalar.plan, atomic_scalar.supported_operation, eligible, wordOperation, sourcePlan]
  · have notWidth := UScalar.neq_to_neq_val.mp width
    change bytes.val ≠ 4 at notWidth
    cases ptx <;> cases resultUsed <;> cases predicate <;> cases explicitNew <;>
      simp [atomic_scalar.plan, eligible, width, notWidth]

@[step] theorem source_supported_instruction_constructs_the_lane_plan :
    atomic_scalar.plan true (.Integer .Add) 4#u8 true false false
      ⦃ result => result = some sourcePlan ⦄ := by
  simp [atomic_scalar.plan, atomic_scalar.supported_operation, sourcePlan]

@[step] theorem rust_supported_operation (operation : atomic_exec.Operation) :
    atomic_scalar.supported_operation operation ⦃ result => result = wordOperation operation ⦄ := by
  cases operation with
  | Integer op => cases op <;> simp [atomic_scalar.supported_operation, wordOperation]
  | AddF32Ftz | AddF64 | AddBf16 | AddBf16x2 =>
    simp [atomic_scalar.supported_operation, wordOperation]

@[step] theorem rust_frozen_request (operation : atomic_exec.Operation)
    (source : symbolic.Metadata) :
    atomic_scalar.request operation source ⦃ result => result.op = operation ∧
      result.bytes = 4#u8 ∧ result.value.val = source.captured.val ∧
      result.compare = 0#u64 ⦄ := by
  unfold atomic_scalar.request
  step with UScalar.cast_inBounds_spec .U64 source.captured (by scalar_tac) as ⟨value,valueSpec⟩
  simp_all

def frozenInput (operation : atomic_integer.IntegerOp) (source : symbolic.Metadata)
    (old : U64) : atomic_exec.AtomicInput :=
  ⟨.Integer operation, 4#u8, old, UScalar.cast .U64 source.captured, 0#u64⟩

@[step] theorem source_frozen_request_executes_integer_model {F : Type}
    (implementation : atomic_exec.FloatingArithmetic F) (semantics : AtomicFloatSemantics)
    (operation : atomic_integer.IntegerOp) (source : symbolic.Metadata) (old : U64) :
    (do
      let request ← atomic_scalar.request (.Integer operation) source
      let input ← atomic_exec.resolve_old request old
      atomic_exec.execute implementation input)
      ⦃ result => result = atomicExecutionModel semantics (frozenInput operation source old) ⦄ := by
  step with rust_frozen_request (.Integer operation) source as
    ⟨request,operationValue,widthValue,sourceValue,compareValue⟩
  have inputValue : atomic_exec.resolve_old request old = ok (frozenInput operation source old) := by
    cases request
    simp_all [atomic_exec.resolve_old, frozenInput]
    scalar_tac
  rw [inputValue]
  simp only [bind_ok]
  exact rust_atomic_integer_execute_spec implementation semantics
    (frozenInput operation source old) operation rfl

def exchange : atomic_exec.Operation → Bool
  | .Integer .Exch => true
  | _ => false

@[step] theorem rust_replacement_defined (operation : atomic_exec.Operation)
    (oldDefined sourceDefined : Bool) :
    atomic_scalar.replacement_defined operation oldDefined sourceDefined
      ⦃ result => result = (sourceDefined &&
        (exchange operation || oldDefined)) ⦄ := by
  cases operation with
  | Integer op => cases op <;> cases oldDefined <;> cases sourceDefined <;>
      simp [atomic_scalar.replacement_defined, exchange]
  | AddF32Ftz | AddF64 | AddBf16 | AddBf16x2 =>
    cases oldDefined <;> cases sourceDefined <;> simp [atomic_scalar.replacement_defined, exchange]

@[step] theorem source_exchange_publishes_a_defined_source_over_undefined_old :
    atomic_scalar.replacement_defined (.Integer .Exch) false true
      ⦃ result => result = true ⦄ := by
  simp [atomic_scalar.replacement_defined]

@[step] theorem rust_bundled_order_unmodeled (ptx resultUsed continued sameWarp sameWord sharedResult : Bool) :
    atomic_scalar.bundled_order_unmodeled ptx resultUsed continued sameWarp sameWord sharedResult
      ⦃ result => result = AtomicLaneCoverage.needsDiagnostic
        ⟨ptx,resultUsed,continued,sameWarp,sameWord,sharedResult⟩ ⦄ := by
  cases ptx <;> cases resultUsed <;> cases continued <;> cases sameWarp <;>
    cases sameWord <;> cases sharedResult <;>
    simp [atomic_scalar.bundled_order_unmodeled, AtomicLaneCoverage.needsDiagnostic,
      AtomicLaneCoverage.interferes]

@[step] theorem source_uncontinued_returning_interfering_pair_is_diagnosed
    (sameWord sharedResult : Bool) (interfering : (sameWord || sharedResult) = true) :
    atomic_scalar.bundled_order_unmodeled true true false true sameWord sharedResult
      ⦃ result => result = true ⦄ := by
  step with rust_bundled_order_unmodeled true true false true sameWord sharedResult
    as ⟨result,resultValue⟩
  simpa [AtomicLaneCoverage.needsDiagnostic,AtomicLaneCoverage.interferes,interfering]
    using resultValue

@[step] theorem source_an_undiagnosed_returning_interfering_pair_has_a_continuation
    (continued sameWord sharedResult : Bool) (interfering : (sameWord || sharedResult) = true) :
    atomic_scalar.bundled_order_unmodeled true true continued true sameWord sharedResult
      ⦃ result => result = false → continued = true ⦄ := by
  apply WP.spec_mono
    (rust_bundled_order_unmodeled true true continued true sameWord sharedResult)
  intro result resultValue absent
  apply AtomicLaneCoverage.an_undiagnosed_returning_interfering_pair_has_a_continuation
    ⟨true,true,continued,true,sameWord,sharedResult⟩ rfl rfl
  · simpa [AtomicLaneCoverage.interferes] using interfering
  · exact resultValue.symm.trans absent

@[step] theorem source_disjoint_private_word_pairs_do_not_add_this_diagnostic
    (ptx used continued sameWarp : Bool) :
    atomic_scalar.bundled_order_unmodeled ptx used continued sameWarp false false
      ⦃ result => result = false ⦄ := by
  step with rust_bundled_order_unmodeled ptx used continued sameWarp false false
    as ⟨result,resultValue⟩
  simpa [AtomicLaneCoverage.needsDiagnostic,AtomicLaneCoverage.interferes] using resultValue

@[step] theorem rust_claim_allowed (actor : U32) (forced : Option U32) :
    atomic_scalar.claim_allowed actor forced ⦃ result => result =
      match forced with
      | none => true
      | some other => decide (actor.val = other.val) ⦄ := by
  cases forced with
  | none => simp [atomic_scalar.claim_allowed]
  | some other =>
    by_cases same : actor = other
    · subst other; simp [atomic_scalar.claim_allowed]
    · have distinct := UScalar.neq_to_neq_val.mp same
      simp [atomic_scalar.claim_allowed, same, distinct]

@[step] theorem rust_ticket_progress (active ready : Bool) :
    atomic_scalar.ticket_progress active ready ⦃ result => result = (!active || ready) ⦄ := by
  cases active <;> cases ready <;> simp [atomic_scalar.ticket_progress]

@[step] theorem source_guarded_off_ticket_advances :
    atomic_scalar.ticket_progress false false ⦃ result => result = true ⦄ := by
  simp [atomic_scalar.ticket_progress]

@[step] theorem source_unresolved_word_can_report_its_fault (actor : U32) :
    atomic_scalar.claim_allowed actor none ⦃ result => result = true ⦄ := by
  simp [atomic_scalar.claim_allowed]

@[step] theorem source_matching_forced_actor_can_advance (actor : U32) :
    atomic_scalar.claim_allowed actor (some actor) ⦃ result => result = true ⦄ := by
  simp [atomic_scalar.claim_allowed]

@[step] theorem rust_ticket_pair_eligible (actor warp pc otherActor otherWarp otherPc : U32)
    (continued : Bool) :
    atomic_scalar.ticket_pair_eligible actor warp pc otherActor otherWarp otherPc continued
      ⦃ result => result = (decide (actor = otherActor) || decide (warp ≠ otherWarp) ||
        decide (pc ≠ otherPc) || continued) ⦄ := by
  by_cases sameActor : actor = otherActor <;> by_cases sameWarp : warp = otherWarp <;>
    by_cases samePc : pc = otherPc <;> cases continued <;>
    simp_all [atomic_scalar.ticket_pair_eligible]

@[step] theorem source_unsupported_bundled_claims_are_excluded (actor other : U32)
    (warp pc : U32) (distinct : actor ≠ other) :
    atomic_scalar.ticket_pair_eligible actor warp pc other warp pc false
      ⦃ result => result = false ⦄ := by
  simp [atomic_scalar.ticket_pair_eligible, distinct]

@[step] theorem source_selected_lane_keeps_its_frozen_word_and_address
    (input : atomic_components.FrozenOperands) :
    atomic_components.begin_component input sourcePlan .First ⦃ result =>
      ∃ selected, result = some selected ∧ selected.address = input.address ∧
        selected.source = input.first ∧ AtomicComponentPlan.retired
          (AtomicComponentPlan.planValue selected.remaining) = true ⦄ := by
  step with AtomicComponentPlan.rust_begin_component input sourcePlan .First
    as ⟨result,plans,selectedSpec⟩
  cases result with
  | none => simp [sourcePlan, AtomicComponentPlan.planValue, AtomicComponentPlan.componentValue,
      AtomicComponentPlan.finish, AtomicComponentPlan.pending] at plans
  | some selected =>
    obtain ⟨source,address⟩ := selectedSpec selected rfl
    refine ⟨selected,rfl,?_,?_,?_⟩
    · simp only [AtomicComponentPlan.componentValue, AtomicComponentPlan.byteOffset,
        Nat.add_zero] at address
      scalar_tac
    · simpa [AtomicComponentPlan.frozenOperand, AtomicComponentPlan.componentValue] using source
    · simpa [sourcePlan, AtomicComponentPlan.planValue, AtomicComponentPlan.componentValue,
        AtomicComponentPlan.finish, AtomicComponentPlan.clear, AtomicComponentPlan.pending, AtomicComponentPlan.retired]
        using congrArg (Option.map AtomicComponentPlan.retired) plans

open AtomicComponentPlan in
@[step] theorem source_every_enabled_scalar_prefix_preserves_observations
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
@[step] theorem source_middle_lane_claim_order_is_admitted :
    sourceRunPlans (fun _ => sourcePlan) [(1,.First),(0,.First),(2,.First)] ⦃ result =>
      ∃ next, result = some next ∧ retired (planValue (next 0)) = true ∧
        retired (planValue (next 1)) = true ∧ retired (planValue (next 2)) = true ⦄ := by
  simp [sourceRunPlans, atomic_components.finish, atomic_components.pending, update,
    sourcePlan, planValue, retired]

end MemoryWrite.AtomicScalar

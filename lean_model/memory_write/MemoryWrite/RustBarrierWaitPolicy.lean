/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierWaitPolicy

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

universe u v

namespace MemoryWrite.BarrierTransition

def waitModeValue : barrier.WaitMode → PollMode
  | .Blocking => .blocking
  | .NonBlocking => .nonblocking

def queryValue {T : Type u} {E : Type v} (release : T → BarrierReduction.Clock) :
    core.result.Result (Option T) E → PollObservation
  | .Ok none => .blocked
  | .Ok (some clock) => .ready (release clock)
  | .Err _ => .error

@[step]
theorem rust_wait_mode (isPeek blockingOverride : Bool) :
    barrier.wait_mode isPeek blockingOverride
      ⦃ mode => waitModeValue mode =
        if isPeek && !blockingOverride then .nonblocking else .blocking ⦄ := by
  cases isPeek <;> cases blockingOverride <;> simp [barrier.wait_mode, waitModeValue]

@[step]
theorem rust_wait_policy {T : Type u} {E : Type v}
    (release : T → BarrierReduction.Clock) (query : core.result.Result (Option T) E)
    (mode : barrier.WaitMode) (reported : Option U32) (pc : U32) (forceRequested : Bool) :
    barrier.wait_policy query mode reported pc forceRequested
      ⦃ policy =>
        policy.progress = (queryPolicy (queryValue release query) (waitModeValue mode)
          (reported.map UScalar.val) pc.val forceRequested).progress ∧
        policy.force_failed = (queryPolicy (queryValue release query) (waitModeValue mode)
          (reported.map UScalar.val) pc.val forceRequested).forceFailed ⦄ := by
  cases query with
  | Ok value =>
    cases value <;> cases mode <;> cases forceRequested <;>
      simp [barrier.wait_policy, barrier.WaitMode.Insts.CoreCmpPartialEqWaitMode.eq, barrier.WaitMode.read_discriminant,
        barrier.wait_can_progress, queryPolicy, queryValue, waitModeValue]
  | Err error =>
    cases reported with
    | none =>
      cases mode <;> cases forceRequested <;>
        simp [barrier.wait_policy, barrier.WaitMode.Insts.CoreCmpPartialEqWaitMode.eq, barrier.WaitMode.read_discriminant,
          barrier.wait_can_progress, queryPolicy, queryValue]
    | some previous =>
      cases mode <;> cases forceRequested <;>
        simp [barrier.wait_policy, barrier.WaitMode.Insts.CoreCmpPartialEqWaitMode.eq, barrier.WaitMode.read_discriminant,
          barrier.wait_can_progress, queryPolicy, queryValue, bne]
      all_goals
        by_cases same : previous = pc
        · simp [same]
        · have different := UScalar.neq_to_neq_val.mp same
          simp [same, different]

end MemoryWrite.BarrierTransition

namespace MemoryWrite.BarrierTransition

theorem source_dispatch_policy {T : Type u} {E : Type v}
    (release : T → BarrierReduction.Clock) (query : core.result.Result (Option T) E)
    (isPeek blockingOverride : Bool) (reported : Option U32) (pc : U32) (forceRequested : Bool) :
    (do
      let mode ← barrier.wait_mode isPeek blockingOverride
      barrier.wait_policy query mode reported pc forceRequested)
      ⦃ policy =>
        let expected := queryPolicy (queryValue release query)
          (if isPeek && !blockingOverride then .nonblocking else .blocking)
          (reported.map UScalar.val) pc.val forceRequested
        policy.progress = expected.progress ∧ policy.force_failed = expected.forceFailed ⦄ := by
  step with rust_wait_mode isPeek blockingOverride as ⟨mode, modeValue⟩
  step with rust_wait_policy release query mode reported pc forceRequested as ⟨policy, policyProgress, policyForced⟩
  simpa only [modeValue] using And.intro policyProgress policyForced

theorem source_error_cannot_be_forced_failed {T : Type u} {E : Type v}
    (error : E) (mode : barrier.WaitMode) (reported : Option U32) (pc : U32)
    (forceRequested : Bool) :
    @barrier.wait_policy T E (.Err error) mode reported pc forceRequested
      ⦃ policy => policy.progress = decide (reported.map UScalar.val ≠ some pc.val) ∧
        policy.force_failed = false ⦄ := by
  apply WP.spec_mono (rust_wait_policy (fun _ : T => fun _ => 0)
    (.Err error : core.result.Result (Option T) E) mode reported pc forceRequested)
  intro policy projected
  simpa [queryValue, queryPolicy] using projected

theorem source_error_reports_once_in_both_modes {T : Type u} {E : Type v}
    (error : E) (mode : barrier.WaitMode) (reported : Option U32) (pc : U32)
    (forceRequested : Bool) :
    (do
      let first ← @barrier.wait_policy T E (.Err error) mode reported pc forceRequested
      let next ← barrier.mark_wait_error reported pc
      let repeated ← @barrier.wait_policy T E (.Err error) mode next pc forceRequested
      pure (first, next, repeated))
      ⦃ result => result.1.progress = decide (reported.map UScalar.val ≠ some pc.val) ∧
        result.1.force_failed = false ∧ result.2.1 = some pc ∧
        result.2.2.progress = false ∧ result.2.2.force_failed = false ⦄ := by
  step with source_error_cannot_be_forced_failed (T := T) error mode reported pc forceRequested
    as ⟨first, firstProgress, firstForced⟩
  step with rust_mark_wait_error reported pc as ⟨next, nextValue⟩
  step with source_error_cannot_be_forced_failed (T := T) error mode next pc forceRequested
    as ⟨repeated, repeatedProgress, repeatedForced⟩
  simp_all

end MemoryWrite.BarrierTransition

namespace MemoryWrite.BarrierTransition

theorem source_policy_preserves_enabledness {T : Type u} {E : Type v}
    (release : T → BarrierReduction.Clock) (query : core.result.Result (Option T) E)
    (state : PollState) (actor : Nat) (mode : barrier.WaitMode) (reported : Option U32)
    (pc : U32) (forceRequested : Bool)
    (projection : state.reported actor = reported.map UScalar.val) :
    barrier.wait_policy query mode reported pc forceRequested
      ⦃ policy => (policy.progress = true ↔ observationEnabledMode state actor pc.val
        (waitModeValue mode) (queryValue release query)) ∧
        (policy.force_failed = true → waitModeValue mode = .nonblocking ∧
          forceRequested = true ∧ observationEnabledMode state actor pc.val
            (waitModeValue mode) (queryValue release query) ∧
          queryValid (queryValue release query)) ⦄ := by
  apply WP.spec_mono (rust_wait_policy release query mode reported pc forceRequested)
  intro policy projected
  rcases projected with ⟨progress, forced⟩
  rw [← projection] at progress forced
  constructor
  · rw [progress]
    exact policy_progress_exactly state actor pc.val (queryValue release query)
      (waitModeValue mode) forceRequested
  · intro forcing
    rw [forced] at forcing
    exact forced_failure_is_valid_nonblocking_progress state actor pc.val
      (queryValue release query) (waitModeValue mode) forceRequested forcing

end MemoryWrite.BarrierTransition

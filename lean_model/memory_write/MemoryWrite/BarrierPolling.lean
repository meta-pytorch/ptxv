/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierPhaseKey

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

universe u v

namespace MemoryWrite.BarrierTransition

def pollValue {T : Type u} (phase parity : Nat) (completed : Array T 2#usize) : Option T :=
  if phase % 2 = parity % 2 then none else
    some (completed.val[parity % 2]'(by have := Nat.mod_lt parity (show 0 < 2 by decide); simpa))

@[step]
theorem rust_poll_release {T : Type u} (phase parity : U32) (completed : Array T 2#usize) :
    barrier.poll_release phase parity completed
      ⦃ result => result = pollValue phase.val parity.val completed ⦄ := by
  unfold barrier.poll_release
  step as ⟨requested, requestedValue⟩; try scalar_tac
  step as ⟨current, currentValue⟩; try scalar_tac
  split
  · rename_i same
    have sameValue : phase.val % 2 = parity.val % 2 := by scalar_tac
    simp [pollValue, sameValue]
  · rename_i different
    have differentValue : phase.val % 2 ≠ parity.val % 2 := by scalar_tac
    split
    · rename_i zero
      have requestedZero : parity.val % 2 = 0 := by scalar_tac
      step
      simp_all [pollValue]
    · rename_i nonzero
      have requestedOne : parity.val % 2 = 1 := by scalar_tac
      step
      simp_all [pollValue]

def canProgress {T : Type u} {E : Type v}
    (result : core.result.Result (Option T) E) (reported : Option Nat) (pc : Nat) : Bool :=
  match result with
  | .Ok none => false
  | .Ok (some _) => true
  | .Err _ => match reported with
    | none => true
    | some previous => decide (previous ≠ pc)

@[step]
theorem rust_wait_can_progress {T : Type u} {E : Type v}
    (result : core.result.Result (Option T) E) (reported : Option U32) (pc : U32) :
    barrier.wait_can_progress result reported pc
      ⦃ ready => ready = canProgress result (reported.map UScalar.val) pc.val ⦄ := by
  cases result with
  | Ok value => cases value <;> simp [barrier.wait_can_progress, canProgress]
  | Err error =>
    cases reported with
    | none => simp [barrier.wait_can_progress, canProgress]
    | some previous =>
      simp only [barrier.wait_can_progress, canProgress, Option.map_some, WP.spec_ok]
      by_cases same : previous = pc
      · simp [same]
      · have different := UScalar.neq_to_neq_val.mp same
        simp [bne, same, different]

@[step]
theorem rust_mark_wait_error (reported : Option U32) (pc : U32) :
    barrier.mark_wait_error reported pc ⦃ next => next = some pc ⦄ := by
  simp [barrier.mark_wait_error]

@[step]
theorem rust_clear_wait_error (reported : Option U32) :
    barrier.clear_wait_error reported ⦃ next => next = none ⦄ := by
  simp [barrier.clear_wait_error]

 theorem source_error_progress_exactly {T : Type u} {E : Type v} (error : E)
    (reported : Option U32) (pc : U32) :
    @barrier.wait_can_progress T E (.Err error) reported pc
      ⦃ ready => ready = decide (reported.map UScalar.val ≠ some pc.val) ⦄ := by
  apply WP.spec_mono (rust_wait_can_progress (.Err error : core.result.Result (Option T) E) reported pc)
  intro ready projected
  rw [projected]
  cases reported <;> simp [canProgress]

 theorem source_first_error_enabled_then_duplicate_blocks {T : Type u} {E : Type v}
    (error : E) (reported : Option U32) (pc : U32) :
    (do
      let first ← @barrier.wait_can_progress T E (.Err error) reported pc
      let next ← barrier.mark_wait_error reported pc
      let repeated ← @barrier.wait_can_progress T E (.Err error) next pc
      pure (first, next, repeated))
      ⦃ result => result.1 = decide (reported.map UScalar.val ≠ some pc.val) ∧
        result.2.1 = some pc ∧ result.2.2 = false ⦄ := by
  step with source_error_progress_exactly (T := T) error reported pc as ⟨first, firstValue⟩
  step with rust_mark_wait_error reported pc as ⟨next, nextValue⟩
  step with source_error_progress_exactly (T := T) error next pc as ⟨repeated, repeatedValue⟩
  simp_all

 theorem source_valid_wait_progress_exactly {T : Type u} {E : Type v}
    (phase parity : U32) (completed : Array T 2#usize) (reported : Option U32) (pc : U32) :
    (do
      let result ← barrier.poll_release phase parity completed
      @barrier.wait_can_progress T E (.Ok result) reported pc)
      ⦃ ready => ready = decide (phase.val % 2 ≠ parity.val % 2) ⦄ := by
  step with rust_poll_release phase parity completed as ⟨result, polled⟩
  step with rust_wait_can_progress (.Ok result : core.result.Result (Option T) E) reported pc
    as ⟨ready, readyValue⟩
  simp_all [pollValue, canProgress]
  split <;> simp_all
inductive PollObservation where
  | blocked
  | ready (release : BarrierReduction.Clock)
  | error

structure PollState where
  execution : FullState
  reported : Nat → Option Nat

def observationEnabled (state : PollState) (actor pc : Nat) : PollObservation → Prop
  | .blocked => False
  | .ready _ => True
  | .error => state.reported actor ≠ some pc

def observe (state : PollState) (actor pc : Nat) : PollObservation → PollState
  | .blocked => state
  | .ready release =>
    { state with
      execution := { state.execution with
        positions := BarrierReduction.update state.execution.positions actor
          (state.execution.positions actor + 1)
        acquired := BarrierReduction.update state.execution.acquired actor
          (BarrierReduction.join (state.execution.acquired actor) release) }
      reported := BarrierReduction.update state.reported actor none }
  | .error =>
    if state.reported actor = some pc then state else
      { state with
        execution := { state.execution with
          failed := true
          findings := state.execution.findings ++ [pc] }
        reported := BarrierReduction.update state.reported actor (some pc) }

 theorem first_error_reports_and_preserves_actor_state (state : PollState) (actor pc : Nat)
    (first : state.reported actor ≠ some pc) :
    observationEnabled state actor pc .error ∧
    (observe state actor pc .error).execution.findings = state.execution.findings ++ [pc] ∧
    (observe state actor pc .error).execution.failed = true ∧
    (observe state actor pc .error).execution.positions = state.execution.positions ∧
    (observe state actor pc .error).execution.acquired = state.execution.acquired ∧
    (observe state actor pc .error).execution.counters = state.execution.counters ∧
    (observe state actor pc .error).reported actor = some pc := by
  simp [observe, observationEnabled, first, BarrierReduction.update]

 theorem diagnosed_error_is_observational_stutter (state : PollState) (actor pc : Nat)
    (diagnosed : state.reported actor = some pc) :
    ¬observationEnabled state actor pc .error ∧ observe state actor pc .error = state := by
  simp [observationEnabled, observe, diagnosed]

 theorem error_reports_once_then_blocks (state : PollState) (actor pc : Nat) :
    let next := observe state actor pc .error
    ¬observationEnabled next actor pc .error ∧ observe next actor pc .error = next := by
  by_cases diagnosed : state.reported actor = some pc
  · simpa only [observe, ite_eq_left diagnosed] using
      diagnosed_error_is_observational_stutter state actor pc diagnosed
  · apply diagnosed_error_is_observational_stutter
    simp [observe, diagnosed, BarrierReduction.update]

 theorem readiness_reenables_and_clears_diagnostic (state : PollState) (actor pc : Nat)
    (release : BarrierReduction.Clock) :
    observationEnabled state actor pc (.ready release) ∧
    (observe state actor pc (.ready release)).execution.positions actor =
      state.execution.positions actor + 1 ∧
    (observe state actor pc (.ready release)).execution.acquired actor =
      BarrierReduction.join (state.execution.acquired actor) release ∧
    (observe state actor pc (.ready release)).reported actor = none ∧
    (observe state actor pc (.ready release)).execution.findings = state.execution.findings := by
  simp [observationEnabled, observe, BarrierReduction.update]

 theorem poll_failure_sticky (state : PollState) (actor pc : Nat) (query : PollObservation)
    (failed : state.execution.failed = true) :
    (observe state actor pc query).execution.failed = true := by
  cases query with
  | blocked => exact failed
  | ready release => exact failed
  | error =>
    by_cases diagnosed : state.reported actor = some pc <;> simp [observe, diagnosed, failed]

 theorem poll_findings_prefix (state : PollState) (actor pc : Nat) (query : PollObservation) :
    ∃ extra, (observe state actor pc query).execution.findings = state.execution.findings ++ extra := by
  cases query with
  | blocked => exact ⟨[], by simp [observe]⟩
  | ready release => exact ⟨[], by simp [observe]⟩
  | error =>
    by_cases diagnosed : state.reported actor = some pc
    · exact ⟨[], by simp [observe, diagnosed]⟩
    · exact ⟨[pc], by simp [observe, diagnosed]⟩

def resultObservation {T : Type u} {E : Type v} (project : T → BarrierReduction.Clock) :
    core.result.Result (Option T) E → PollObservation
  | .Ok none => .blocked
  | .Ok (some release) => .ready (project release)
  | .Err _ => .error

 theorem source_progress_refines_complete_blocked_failed {T : Type u} {E : Type v}
    (result : core.result.Result (Option T) E) (project : T → BarrierReduction.Clock)
    (state : PollState) (actor : Nat) (reported : Option U32) (pc : U32)
    (projection : reported.map UScalar.val = state.reported actor) :
    barrier.wait_can_progress result reported pc
      ⦃ ready => ready = true ↔ observationEnabled state actor pc.val (resultObservation project result) ⦄ := by
  apply WP.spec_mono (rust_wait_can_progress result reported pc)
  intro ready readyValue
  rw [readyValue]
  cases result with
  | Ok value => cases value <;> simp [canProgress, resultObservation, observationEnabled]
  | Err error =>
    simp only [resultObservation, observationEnabled]
    rw [← projection]
    cases reported <;> simp [canProgress]

 theorem source_poll_acquires_observed_release {T : Type u}
    (phase parity : U32) (completed : Array T 2#usize) (project : T → BarrierReduction.Clock)
    (state : FullState) (phaseProjection : phase.val = state.counters.phase)
    (slotsProjection : projectSlots project state.completedRelease completed = state.completedRelease) :
    barrier.poll_release phase parity completed
      ⦃ result => result.map project =
        if waitEnabled state parity.val then some (state.completedRelease (parity.val % 2)) else none ⦄ := by
  apply WP.spec_mono (rust_poll_release phase parity completed)
  intro result polled
  rw [polled]
  have slot := congrFun slotsProjection (parity.val % 2)
  have bound : parity.val % 2 < 2 := Nat.mod_lt _ (by decide)
  simp only [projectSlots, bound, ↓reduceDIte] at slot
  by_cases enabled : waitEnabled state parity.val
  · have different : phase.val % 2 ≠ parity.val % 2 := by
      simp only [waitEnabled] at enabled
      rw [phaseProjection]
      exact Ne.symm enabled
    simp [pollValue, different, enabled, slot]
  · have same : phase.val % 2 = parity.val % 2 := by
      simp only [waitEnabled, ne_eq, not_not] at enabled
      rw [phaseProjection]
      exact enabled.symm
    simp [pollValue, same, enabled]
 theorem two_completions_restore_phase_parity (phase : Nat) :
    ((((phase + 1) % phaseModulus) + 1) % phaseModulus) % 2 = phase % 2 := by
  have first := phase_wrap_toggles_parity phase
  have second := phase_wrap_toggles_parity ((phase + 1) % phaseModulus)
  omega

 theorem source_two_completions_close_poll_window {T : Type u}
    (source : barrier.Counters) (deltaFirst deltaSecond : barrier.TransactionDelta)
    (arrivalsFirst arrivalsSecond : U32) (completed : Array T 2#usize)
    (firstCompletes : (transition (projectCounters source) (deltaValue deltaFirst) arrivalsFirst.val).completed = true)
    (secondCompletes : (transition
      (transition (projectCounters source) (deltaValue deltaFirst) arrivalsFirst.val).after
      (deltaValue deltaSecond) arrivalsSecond.val).completed = true) :
    (do
      let first ← barrier.transition source deltaFirst arrivalsFirst
      let firstPoll ← barrier.poll_release first.after.phase source.phase completed
      let second ← barrier.transition first.after deltaSecond arrivalsSecond
      let secondPoll ← barrier.poll_release second.after.phase source.phase completed
      pure (firstPoll, secondPoll))
      ⦃ result => result.1.isSome = true ∧ result.2 = none ⦄ := by
  step with rust_transition source deltaFirst arrivalsFirst as ⟨first, firstModel⟩
  step with rust_poll_release first.after.phase source.phase completed as ⟨firstPoll, firstValue⟩
  step with rust_transition first.after deltaSecond arrivalsSecond as ⟨second, secondModel⟩
  step with rust_poll_release second.after.phase source.phase completed as ⟨secondPoll, secondValue⟩
  have firstPhase := (completion_resets_expected (projectCounters source)
    (deltaValue deltaFirst) arrivalsFirst.val firstCompletes).2.1
  rw [← firstModel] at firstPhase
  change first.after.phase.val = (source.phase.val + 1) % phaseModulus at firstPhase
  have middle : projectCounters first.after =
      (transition (projectCounters source) (deltaValue deltaFirst) arrivalsFirst.val).after :=
    congrArg Transition.after firstModel
  have secondActualCompletes :
      (transition (projectCounters first.after) (deltaValue deltaSecond) arrivalsSecond.val).completed = true := by
    rw [middle]
    exact secondCompletes
  have secondPhase := (completion_resets_expected (projectCounters first.after)
    (deltaValue deltaSecond) arrivalsSecond.val secondActualCompletes).2.1
  rw [← secondModel] at secondPhase
  change second.after.phase.val = (first.after.phase.val + 1) % phaseModulus at secondPhase
  have firstDifferent : first.after.phase.val % 2 ≠ source.phase.val % 2 := by
    rw [firstPhase, phase_wrap_toggles_parity]
    omega
  have secondSame : second.after.phase.val % 2 = source.phase.val % 2 := by
    rw [secondPhase, firstPhase]
    exact two_completions_restore_phase_parity source.phase.val
  simp [firstValue, secondValue, pollValue, firstDifferent, secondSame]
def runPollTrace (state : PollState) : List (Nat × Nat × PollObservation) → PollState
  | [] => state
  | (actor, pc, query) :: rest => runPollTrace (observe state actor pc query) rest

 theorem poll_trace_failure_sticky (state : PollState) (events : List (Nat × Nat × PollObservation))
    (failed : state.execution.failed = true) :
    (runPollTrace state events).execution.failed = true := by
  induction events generalizing state with
  | nil => exact failed
  | cons event rest ih =>
    rcases event with ⟨actor, pc, query⟩
    exact ih _ (poll_failure_sticky state actor pc query failed)

 theorem poll_trace_findings_prefix (state : PollState) (events : List (Nat × Nat × PollObservation)) :
    ∃ extra, (runPollTrace state events).execution.findings = state.execution.findings ++ extra := by
  induction events generalizing state with
  | nil => exact ⟨[], by simp [runPollTrace]⟩
  | cons event rest ih =>
    rcases event with ⟨actor, pc, query⟩
    obtain ⟨first, firstValue⟩ := poll_findings_prefix state actor pc query
    obtain ⟨later, laterValue⟩ := ih (observe state actor pc query)
    exact ⟨first ++ later, by simpa only [runPollTrace, firstValue, List.append_assoc] using laterValue⟩

 theorem duplicate_error_erasure_preserves_future_trace (state : PollState) (actor pc : Nat)
    (rest : List (Nat × Nat × PollObservation)) (diagnosed : state.reported actor = some pc) :
    runPollTrace state ((actor, pc, .error) :: rest) = runPollTrace state rest := by
  rw [runPollTrace, (diagnosed_error_is_observational_stutter state actor pc diagnosed).2]

 theorem error_preserves_other_actor_enabledness (state : PollState) (actor pc other otherPc : Nat)
    (query : PollObservation) (different : other ≠ actor) :
    observationEnabled (observe state actor pc .error) other otherPc query ↔
      observationEnabled state other otherPc query := by
  cases query with
  | blocked => rfl
  | ready release => rfl
  | error =>
    by_cases diagnosed : state.reported actor = some pc <;>
      simp [observe, observationEnabled, diagnosed, BarrierReduction.update, different]

end MemoryWrite.BarrierTransition

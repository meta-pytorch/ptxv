/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierPolling

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.BarrierTransition

inductive PollMode where
  | blocking
  | nonblocking
  deriving DecidableEq

structure PollPolicy where
  progress : Bool
  forceFailed : Bool
  deriving DecidableEq

def queryValid : PollObservation → Prop
  | .error => False
  | _ => True

def queryPolicy (query : PollObservation) (mode : PollMode) (reported : Option Nat)
    (pc : Nat) (forceRequested : Bool) : PollPolicy :=
  match query with
  | .blocked => ⟨decide (mode = .nonblocking), forceRequested && decide (mode = .nonblocking)⟩
  | .ready _ => ⟨true, forceRequested && decide (mode = .nonblocking)⟩
  | .error => ⟨decide (reported ≠ some pc), false⟩

def observationEnabledMode (state : PollState) (actor pc : Nat) (mode : PollMode) :
    PollObservation → Prop
  | .blocked => mode = .nonblocking
  | .ready _ => True
  | .error => state.reported actor ≠ some pc

def observeMode (state : PollState) (actor pc : Nat) (mode : PollMode) :
    PollObservation → PollState
  | .blocked =>
    if mode = .nonblocking then
      { state with
        execution := { state.execution with positions :=
          (BarrierReduction.update state.execution.positions actor (state.execution.positions actor + 1)) }
        reported := BarrierReduction.update state.reported actor none }
    else state
  | query => observe state actor pc query

theorem blocking_mode_matches_observation (state : PollState) (actor pc : Nat)
    (query : PollObservation) :
    observationEnabledMode state actor pc .blocking query ↔
      observationEnabled state actor pc query := by
  cases query <;> simp [observationEnabledMode, observationEnabled]

theorem policy_progress_exactly (state : PollState) (actor pc : Nat)
    (query : PollObservation) (mode : PollMode) (forceRequested : Bool) :
    (queryPolicy query mode (state.reported actor) pc forceRequested).progress = true ↔
      observationEnabledMode state actor pc mode query := by
  cases query <;> simp [queryPolicy, observationEnabledMode]

theorem forced_failure_excludes_error (mode : PollMode) (reported : Option Nat)
    (pc : Nat) (forceRequested : Bool) :
    (queryPolicy .error mode reported pc forceRequested).forceFailed = false := rfl

theorem forced_failure_is_valid_nonblocking_progress (state : PollState) (actor pc : Nat)
    (query : PollObservation) (mode : PollMode) (forceRequested : Bool)
    (forced : (queryPolicy query mode (state.reported actor) pc forceRequested).forceFailed = true) :
    mode = .nonblocking ∧ forceRequested = true ∧
      observationEnabledMode state actor pc mode query ∧
      queryValid query := by
  cases query with
  | blocked =>
    change forceRequested && decide (mode = .nonblocking) = true at forced
    cases forceRequested with
    | false => change false = true at forced; cases forced
    | true =>
      have hm : mode = .nonblocking := of_decide_eq_true forced
      exact ⟨hm, rfl, hm, trivial⟩
  | ready release =>
    change forceRequested && decide (mode = .nonblocking) = true at forced
    cases forceRequested with
    | false => change false = true at forced; cases forced
    | true =>
      have hm : mode = .nonblocking := of_decide_eq_true forced
      exact ⟨hm, rfl, trivial, trivial⟩
  | error =>
    change false = true at forced
    cases forced

theorem fault_parks_in_both_modes (state : PollState) (actor pc : Nat)
    (mode : PollMode) (forceRequested : Bool) :
    let next := observeMode state actor pc mode .error
    ¬observationEnabledMode next actor pc mode .error ∧
      (queryPolicy .error mode (next.reported actor) pc forceRequested).forceFailed = false ∧
      observeMode next actor pc mode .error = next := by
  obtain ⟨blocked, duplicate⟩ := error_reports_once_then_blocks state actor pc
  exact ⟨blocked, rfl, duplicate⟩

theorem pending_nonblocking_peek_advances_without_acquire (state : PollState) (actor pc : Nat) :
    observationEnabledMode state actor pc .nonblocking .blocked ∧
    (observeMode state actor pc .nonblocking .blocked).execution.positions actor =
      state.execution.positions actor + 1 ∧
    (observeMode state actor pc .nonblocking .blocked).execution.counters = state.execution.counters ∧
    (observeMode state actor pc .nonblocking .blocked).execution.currentRelease = state.execution.currentRelease ∧
    (observeMode state actor pc .nonblocking .blocked).execution.completedRelease = state.execution.completedRelease ∧
    (observeMode state actor pc .nonblocking .blocked).execution.acquired = state.execution.acquired ∧
    (observeMode state actor pc .nonblocking .blocked).execution.findings = state.execution.findings ∧
    (observeMode state actor pc .nonblocking .blocked).reported actor = none := by
  simp [observationEnabledMode, observeMode, BarrierReduction.update]

theorem mode_failure_sticky (state : PollState) (actor pc : Nat) (mode : PollMode)
    (query : PollObservation) (failed : state.execution.failed = true) :
    (observeMode state actor pc mode query).execution.failed = true := by
  cases query with
  | blocked => cases mode <;> exact failed
  | ready release => exact poll_failure_sticky state actor pc (.ready release) failed
  | error => exact poll_failure_sticky state actor pc .error failed

theorem mode_findings_prefix (state : PollState) (actor pc : Nat) (mode : PollMode)
    (query : PollObservation) :
    ∃ extra, (observeMode state actor pc mode query).execution.findings = state.execution.findings ++ extra := by
  cases query with
  | blocked => cases mode <;> exact ⟨[], (List.append_nil _).symm⟩
  | ready release => exact poll_findings_prefix state actor pc (.ready release)
  | error => exact poll_findings_prefix state actor pc .error

end MemoryWrite.BarrierTransition

namespace MemoryWrite.BarrierTransition

def runModeTrace (state : PollState) : List (Nat × Nat × PollMode × PollObservation) → PollState
  | [] => state
  | (actor, pc, mode, query) :: rest => runModeTrace (observeMode state actor pc mode query) rest

theorem mode_trace_failure_sticky (state : PollState)
    (events : List (Nat × Nat × PollMode × PollObservation)) (failed : state.execution.failed = true) :
    (runModeTrace state events).execution.failed = true := by
  induction events generalizing state with
  | nil => exact failed
  | cons event rest ih =>
    rcases event with ⟨actor, pc, mode, query⟩
    exact ih _ (mode_failure_sticky state actor pc mode query failed)

theorem mode_trace_findings_prefix (state : PollState)
    (events : List (Nat × Nat × PollMode × PollObservation)) :
    ∃ extra, (runModeTrace state events).execution.findings = state.execution.findings ++ extra := by
  induction events generalizing state with
  | nil => exact ⟨[], by simp [runModeTrace]⟩
  | cons event rest ih =>
    rcases event with ⟨actor, pc, mode, query⟩
    obtain ⟨first, firstValue⟩ := mode_findings_prefix state actor pc mode query
    obtain ⟨later, laterValue⟩ := ih (observeMode state actor pc mode query)
    exact ⟨first ++ later, by simpa only [runModeTrace, firstValue, List.append_assoc] using laterValue⟩

theorem duplicate_fault_erasure_preserves_mode_trace (state : PollState) (actor pc : Nat)
    (mode : PollMode) (rest : List (Nat × Nat × PollMode × PollObservation))
    (diagnosed : state.reported actor = some pc) :
    runModeTrace state ((actor, pc, mode, .error) :: rest) = runModeTrace state rest := by
  change runModeTrace (observe state actor pc .error) rest = runModeTrace state rest
  rw [(diagnosed_error_is_observational_stutter state actor pc diagnosed).2]

theorem fault_preserves_other_actor_mode_enabledness (state : PollState)
    (actor pc other otherPc : Nat) (mode : PollMode) (query : PollObservation)
    (different : other ≠ actor) :
    observationEnabledMode (observeMode state actor pc mode .error) other otherPc mode query ↔
      observationEnabledMode state other otherPc mode query := by
  cases query with
  | blocked => rfl
  | ready release => rfl
  | error => exact error_preserves_other_actor_enabledness state actor pc other otherPc .error different

end MemoryWrite.BarrierTransition

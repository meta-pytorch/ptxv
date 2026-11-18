/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierStaging

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

universe u

namespace MemoryWrite.BarrierTransition

def releaseFields {T : Type u} (current : T) (completed : Array T 2#usize)
    (empty : T) (outcome : barrier.Transition) : T × Array T 2#usize :=
  if outcome.completed then
    (empty, completed.set (if outcome.intermediate.phase.val % 2 = 0 then 0#usize else 1#usize) current)
  else (current, completed)

@[step]
theorem rust_publish_release {T : Type u} (current : T) (completed : Array T 2#usize)
    (empty : T) (outcome : barrier.Transition) :
    barrier.publish_release current completed empty outcome
      ⦃ result => result = releaseFields current completed empty outcome ⦄ := by
  unfold barrier.publish_release
  step with rust_publication_parity outcome as ⟨parity, hparity⟩
  cases parity with
  | none =>
    have done : outcome.completed = false := by
      cases h : outcome.completed <;> simp_all
    simp_all [releaseFields]
  | some parity =>
    have done : outcome.completed = true := by
      cases h : outcome.completed <;> simp_all
    have hvalue : parity.val = outcome.intermediate.phase.val % 2 := by
      simpa only [done, ↓reduceIte, Option.map_some, Option.some.injEq] using hparity
    simp only
    split
    · change 0 = outcome.intermediate.phase.val % 2 at hvalue
      step
      step
      step
      simp_all [releaseFields, Array.eq_iff, Array.set_val_eq]

    · rename_i ignored different
      have notZero : outcome.intermediate.phase.val % 2 ≠ 0 := by
        intro zero
        apply different
        apply UScalar.eq_of_val_eq
        change parity.val = 0
        omega
      step
      step
      step
      simp_all [releaseFields, Array.eq_iff, Array.set_val_eq]



def projectSlots {T : Type u} (project : T → BarrierReduction.Clock)
    (outside : Nat → BarrierReduction.Clock) (completed : Array T 2#usize) :
    Nat → BarrierReduction.Clock :=
  fun parity => if h : parity < 2 then
    project (completed.val[parity]'(by simpa using h)) else outside parity

 theorem projectSlots_set {T : Type u} (project : T → BarrierReduction.Clock)
    (outside : Nat → BarrierReduction.Clock) (completed : Array T 2#usize)
    (index : Usize) (value : T) (bound : index.val < 2) :
    projectSlots project outside (completed.set index value) =
      BarrierReduction.update (projectSlots project outside completed) index.val (project value) := by
  funext parity
  by_cases inside : parity < 2
  · simp only [projectSlots, inside, ↓reduceDIte, Array.set_val_eq, List.getElem_set]
    by_cases same : index.val = parity
    · simp [same, BarrierReduction.update]
    · simp [same, Ne.symm same, BarrierReduction.update, projectSlots, inside]
  · have different : parity ≠ index.val := by omega
    simp [projectSlots, inside, BarrierReduction.update, different]

 theorem source_release_move_refines_publication {T : Type u}
    (state : FullState) (current empty : T) (completed : Array T 2#usize)
    (project : T → BarrierReduction.Clock) (source : barrier.Transition)
    (actor : Nat) (delta : Int) (arrivals : Nat) (release : BarrierReduction.Clock)
    (outcome : projectTransition source = transition state.counters delta arrivals)
    (success : source.fault = none)
    (currentProjection : project current = BarrierReduction.join state.currentRelease release)
    (emptyProjection : project empty = fun _ => 0)
    (completedProjection : projectSlots project state.completedRelease completed = state.completedRelease) :
    barrier.publish_release current completed empty source
      ⦃ result =>
        project result.1 = (advance state actor delta arrivals release).currentRelease ∧
        projectSlots project state.completedRelease result.2 =
          (advance state actor delta arrivals release).completedRelease ⦄ := by
  apply WP.spec_mono (rust_publish_release current completed empty source)
  intro result resultValue
  rw [resultValue]
  have frame := intermediate_keeps_expected_and_phase state.counters delta arrivals
  rw [← outcome] at frame
  have phase : source.intermediate.phase.val = state.counters.phase := frame.2.1
  have modelSuccess : (transition state.counters delta arrivals).fault = none := by
    rw [← outcome]
    simp [projectTransition, success]
  have modelCompleted : (transition state.counters delta arrivals).completed = source.completed := by
    rw [← outcome]
    rfl
  by_cases done : source.completed = true
  · have bound : (if source.intermediate.phase.val % 2 = 0 then 0#usize else 1#usize).val < 2 := by
      split <;> decide
    have index : (if source.intermediate.phase.val % 2 = 0 then 0#usize else 1#usize).val =
        source.intermediate.phase.val % 2 := by
      split <;> simp_all
    simp only [releaseFields, done, ↓reduceIte]
    rw [projectSlots_set project state.completedRelease completed _ current bound,
      index, phase, completedProjection, currentProjection, emptyProjection]
    simp [advance, publish, modelSuccess, modelCompleted, done]
  · have notDone : source.completed = false := by cases h : source.completed <;> simp_all
    simp [releaseFields, notDone, currentProjection, completedProjection,
      advance, publish, modelSuccess, modelCompleted]
 theorem source_counter_release_sequence_refines_publication {T : Type u}
    (state : FullState) (source : barrier.Counters)
    (delta : barrier.TransactionDelta) (arrivals : U32)
    (current empty : T) (completed : Array T 2#usize)
    (project : T → BarrierReduction.Clock) (actor : Nat) (release : BarrierReduction.Clock)
    (projection : projectCounters source = state.counters)
    (success : (transition state.counters (deltaValue delta) arrivals.val).fault = none)
    (currentProjection : project current = BarrierReduction.join state.currentRelease release)
    (emptyProjection : project empty = fun _ => 0)
    (completedProjection : projectSlots project state.completedRelease completed = state.completedRelease) :
    (do
      let (outcome, duringPending, duringTransactions) ← barrier.begin_update
        source.expected source.pending source.transactions source.phase delta arrivals
      let (newCurrent, newCompleted) ← barrier.publish_release current completed empty outcome
      let (afterPending, afterTransactions, afterPhase) ←
        barrier.finish_update duringPending duringTransactions source.phase outcome
      pure (barrier.Counters.mk source.expected afterPending afterTransactions afterPhase,
        newCurrent, newCompleted))
      ⦃ result =>
        projectCounters result.1 = (advance state actor (deltaValue delta) arrivals.val release).counters ∧
        project result.2.1 = (advance state actor (deltaValue delta) arrivals.val release).currentRelease ∧
        projectSlots project state.completedRelease result.2.2 =
          (advance state actor (deltaValue delta) arrivals.val release).completedRelease ⦄ := by
  step with rust_begin_update source.expected source.pending source.phase source.transactions delta arrivals
    as ⟨outcome, duringPending, duringTransactions, model, pendingValue, transactionValue⟩
  change projectTransition outcome = transition (projectCounters source) (deltaValue delta) arrivals.val at model
  rw [projection] at model
  have outcomeSuccess : outcome.fault = none := by
    have mappedNone : outcome.fault.map projectFault = none := by
      change (projectTransition outcome).fault = none
      rw [model]
      exact success
    cases h : outcome.fault with
    | none => rfl
    | some fault => simp [h] at mappedNone
  step with source_release_move_refines_publication state current empty completed project outcome actor
      (deltaValue delta) arrivals.val release model outcomeSuccess currentProjection
      emptyProjection completedProjection
    as ⟨newCurrent, newCompleted, currentValue, completedValue⟩
  step with rust_finish_update duringPending source.phase duringTransactions outcome
    as ⟨afterPending, afterTransactions, afterPhase, pendingAfter, transactionsAfter, phaseAfter⟩
  refine ⟨?_, currentValue, completedValue⟩
  have expectedFrame := (intermediate_keeps_expected_and_phase state.counters
    (deltaValue delta) arrivals.val).2.2
  rw [← model] at expectedFrame
  have inputExpected := congrArg Counters.expected projection
  simp only [projectTransition, projectCounters] at expectedFrame inputExpected
  simp only [pendingAfter, transactionsAfter, phaseAfter, projectCounters]
  unfold advance publish
  rw [success]
  simp only
  rw [← model]
  simp only [projectTransition, projectCounters, expectedFrame, inputExpected]

end MemoryWrite.BarrierTransition

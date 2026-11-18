/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierTransitionRefinement

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.BarrierTransition

 theorem intermediate_keeps_expected_and_phase (state : Counters) (delta : Int) (arrivals : Nat) :
    (transition state delta arrivals).intermediate.expected = state.expected ∧
    (transition state delta arrivals).intermediate.phase = state.phase ∧
    (transition state delta arrivals).after.expected = state.expected := by
  unfold transition
  dsimp only
  split
  · split
    · split <;> exact ⟨rfl, rfl, rfl⟩
    · exact ⟨rfl, rfl, rfl⟩
  · exact ⟨rfl, rfl, rfl⟩

def beginState (state : FullState) (outcome : Transition) : FullState :=
  { state with counters := outcome.intermediate }

def clockStage (state : FullState) (actor : Nat) (release : BarrierReduction.Clock)
    (outcome : Transition) : FullState :=
  { publish state actor release outcome with counters := state.counters }

def finishState (state : FullState) (outcome : Transition) : FullState :=
  { state with counters := outcome.after }

 theorem clock_stage_frames_counters (state : FullState) (actor : Nat)
    (release : BarrierReduction.Clock) (outcome : Transition) :
    (clockStage state actor release outcome).counters = state.counters := rfl

 theorem staged_publication_matches_atomic_model (state : FullState) (actor : Nat)
    (delta : Int) (arrivals : Nat) (release : BarrierReduction.Clock) :
    let outcome := transition state.counters delta arrivals
    finishState (clockStage (beginState state outcome) actor release outcome) outcome =
      advance state actor delta arrivals release := by
  have phase := (intermediate_keeps_expected_and_phase state.counters delta arrivals).2.1
  dsimp only
  unfold beginState clockStage finishState advance publish
  split <;> simp_all

 theorem completion_clock_stage_precedes_reset (state : FullState) (actor : Nat)
    (delta : Int) (arrivals : Nat) (release : BarrierReduction.Clock)
    (completed : (transition state.counters delta arrivals).completed = true) :
    let outcome := transition state.counters delta arrivals
    let during := clockStage (beginState state outcome) actor release outcome
    during.counters.pending = 0 ∧ during.counters.transactions = 0 ∧
    during.counters.phase = state.counters.phase ∧
    during.completedRelease (state.counters.phase % 2) =
      BarrierReduction.join state.currentRelease release ∧
    (finishState during outcome).counters.pending = state.counters.expected := by
  have phase := (intermediate_keeps_expected_and_phase state.counters delta arrivals).2.1
  obtain ⟨fits, count, debt⟩ := (completion_exactly state.counters delta arrivals).mp completed
  simp [clockStage, beginState, finishState, publish, transition, transactionFits,
    count, debt, BarrierReduction.update]

@[step]
theorem rust_begin_update (expected pending phase : U32) (transactions : I64)
    (delta : barrier.TransactionDelta) (arrivals : U32) :
    barrier.begin_update expected pending transactions phase delta arrivals
      ⦃ result => projectTransition result.1 =
          transition ⟨expected.val, pending.val, transactions.val, phase.val⟩
            (deltaValue delta) arrivals.val ∧
        result.2.1 = result.1.intermediate.pending ∧
        result.2.2 = result.1.intermediate.transactions ⦄ := by
  unfold barrier.begin_update
  step with rust_transition ⟨expected, pending, transactions, phase⟩ delta arrivals
    as ⟨outcome, houtcome⟩
  simpa only [projectCounters] using houtcome

@[step]
theorem rust_finish_update (pending phase : U32) (transactions : I64)
    (outcome : barrier.Transition) :
    barrier.finish_update pending transactions phase outcome
      ⦃ result => result = (outcome.after.pending, outcome.after.transactions, outcome.after.phase) ⦄ := by
  simp [barrier.finish_update]

 theorem source_live_counter_stages (expected pending phase : U32) (transactions : I64)
    (delta : barrier.TransactionDelta) (arrivals : U32) :
    (do
      let (outcome, duringPending, duringTransactions) ←
        barrier.begin_update expected pending transactions phase delta arrivals
      let (afterPending, afterTransactions, afterPhase) ←
        barrier.finish_update duringPending duringTransactions phase outcome
      pure (outcome,
        (barrier.Counters.mk expected duringPending duringTransactions phase),
        (barrier.Counters.mk expected afterPending afterTransactions afterPhase)))
      ⦃ result =>
        let model := transition ⟨expected.val, pending.val, transactions.val, phase.val⟩
          (deltaValue delta) arrivals.val
        projectTransition result.1 = model ∧
        projectCounters result.2.1 = model.intermediate ∧
        projectCounters result.2.2 = model.after ⦄ := by
  step with rust_begin_update expected pending phase transactions delta arrivals
    as ⟨outcome, duringPending, duringTransactions, model, pendingValue, transactionValue⟩
  step with rust_finish_update duringPending phase duringTransactions outcome
    as ⟨afterPending, afterTransactions, afterPhase, afterPendingValue, afterTransactionValue, afterPhaseValue⟩
  have frame := intermediate_keeps_expected_and_phase
    ⟨expected.val, pending.val, transactions.val, phase.val⟩ (deltaValue delta) arrivals.val
  rw [← model] at frame
  simp only [projectTransition, projectCounters] at frame
  simp only [afterPendingValue, afterTransactionValue, afterPhaseValue, pendingValue, transactionValue]
  refine ⟨model, ?_, ?_⟩
  · rw [← model]
    simp only [projectTransition, projectCounters, frame.1, frame.2.1]
  · rw [← model]
    simp only [projectTransition, projectCounters, frame.2.2]

 theorem source_live_stages_refine_publication (state : FullState) (source : barrier.Counters)
    (delta : barrier.TransactionDelta) (arrivals : U32) (actor : Nat)
    (release : BarrierReduction.Clock) (projection : projectCounters source = state.counters) :
    (do
      let (outcome, duringPending, duringTransactions) ← barrier.begin_update
        source.expected source.pending source.transactions source.phase delta arrivals
      let (afterPending, afterTransactions, afterPhase) ←
        barrier.finish_update duringPending duringTransactions source.phase outcome
      pure (outcome,
        (barrier.Counters.mk source.expected duringPending duringTransactions source.phase),
        (barrier.Counters.mk source.expected afterPending afterTransactions afterPhase)))
      ⦃ result =>
        finishState
          (clockStage { state with counters := projectCounters result.2.1 }
            actor release (projectTransition result.1)) (projectTransition result.1) =
          advance state actor (deltaValue delta) arrivals.val release ∧
        projectCounters result.2.2 =
          (advance state actor (deltaValue delta) arrivals.val release).counters ⦄ := by
  apply WP.spec_mono (source_live_counter_stages source.expected source.pending source.phase
    source.transactions delta arrivals)
  rintro ⟨outcome, during, after⟩ ⟨model, hduring, hafter⟩
  change projectTransition outcome = transition (projectCounters source) (deltaValue delta) arrivals.val at model
  change projectCounters during =
    (transition (projectCounters source) (deltaValue delta) arrivals.val).intermediate at hduring
  change projectCounters after =
    (transition (projectCounters source) (deltaValue delta) arrivals.val).after at hafter
  rw [projection] at model hduring hafter
  constructor
  · rw [hduring, model]
    exact staged_publication_matches_atomic_model state actor (deltaValue delta) arrivals.val release
  · rw [hafter]
    unfold advance publish
    split <;> rfl

end MemoryWrite.BarrierTransition

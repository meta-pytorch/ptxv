/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RustClockJoin
import MemoryWrite.BarrierClockMove

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.SparseClock

structure Data where
  base : alloc.vec.Vec (U32 × U32)
  own : U32 × U32

def projectData (data : Data) : BarrierReduction.Clock :=
  value (baseModel data.base.val) (entryModel data.own)

def applySourcePlan (left right : Data) : clock.JoinPlan → Data
  | .Keep => left
  | .ShareRight => { left with base := right.base }
  | .Replace entries => { left with base := entries }

theorem apply_source_plan_base (left right : Data) (plan : clock.JoinPlan) :
    baseModel (applySourcePlan left right plan).base.val =
      applyPlan (baseModel left.base.val) (baseModel right.base.val) (planModel plan) := by
  cases plan <;> rfl

theorem apply_source_plan_own (left right : Data) (plan : clock.JoinPlan) :
    (applySourcePlan left right plan).own = left.own := by
  cases plan <;> rfl

@[step] theorem source_join_refines_pointwise_max (left right : Data) (sharedBase : Bool)
    (hl : Sorted (baseModel left.base.val)) (hr : Sorted (baseModel right.base.val))
    (hshared : sharedBase = true → left.base.val = right.base.val)
    (hsize : left.base.val.length + right.base.val.length + 1 < Usize.max) :
    clock.join_plan left.base.slice left.own right.base.slice right.own sharedBase ⦃ plan =>
      projectData (applySourcePlan left right plan) =
          BarrierReduction.join (projectData left) (projectData right) ∧
      Sorted (baseModel (applySourcePlan left right plan).base.val) ∧
      (applySourcePlan left right plan).own = left.own ⦄ := by
  step with rust_join_plan_bounded left.base.slice right.base.slice left.own right.own
    sharedBase hl hr hsize as ⟨plan, hplan⟩
  refine ⟨?_, ?_, apply_source_plan_own left right plan⟩
  · simp only [projectData, apply_source_plan_base, apply_source_plan_own, hplan]
    exact join_plan_value _ _ _ _ sharedBase (by intro h; rw [hshared h])
  · rw [apply_source_plan_base, hplan]
    exact join_plan_sorted _ _ _ _ sharedBase hl hr

open BarrierTransition

theorem source_join_counter_release_sequence_refines_publication
    (state : FullState) (source : barrier.Counters)
    (delta : barrier.TransactionDelta) (arrivals : U32)
    (current incoming : Data) (completed : Array Data 2#usize) (sharedBase : Bool) (actor : Nat)
    (projection : projectCounters source = state.counters)
    (success : (transition state.counters (deltaValue delta) arrivals.val).fault = none)
    (currentProjection : projectData current = state.currentRelease)
    (completedProjection : projectSlots projectData state.completedRelease completed = state.completedRelease)
    (hc : Sorted (baseModel current.base.val)) (hi : Sorted (baseModel incoming.base.val))
    (hshared : sharedBase = true → current.base.val = incoming.base.val)
    (hsize : current.base.val.length + incoming.base.val.length + 1 < Usize.max) :
    (do
      let (outcome, duringPending, duringTransactions) ← barrier.begin_update
        source.expected source.pending source.transactions source.phase delta arrivals
      let plan ← clock.join_plan current.base.slice current.own incoming.base.slice incoming.own sharedBase
      let (emptyBase, emptyOwn) ← clock.empty
      let (newCurrent, newCompleted) ← barrier.publish_release
        (applySourcePlan current incoming plan) completed (Data.mk emptyBase emptyOwn) outcome
      let (afterPending, afterTransactions, afterPhase) ←
        barrier.finish_update duringPending duringTransactions source.phase outcome
      pure (barrier.Counters.mk source.expected afterPending afterTransactions afterPhase,
        newCurrent, newCompleted))
      ⦃ result =>
        projectCounters result.1 =
          (advance state actor (deltaValue delta) arrivals.val (projectData incoming)).counters ∧
        projectData result.2.1 =
          (advance state actor (deltaValue delta) arrivals.val (projectData incoming)).currentRelease ∧
        projectSlots projectData state.completedRelease result.2.2 =
          (advance state actor (deltaValue delta) arrivals.val (projectData incoming)).completedRelease ⦄ := by
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
  step with source_join_refines_pointwise_max current incoming sharedBase hc hi hshared hsize
    as ⟨plan, joined, sorted, ownFrame⟩
  step with rust_empty as ⟨emptyBase, emptyOwn, emptyBaseValue, emptyOwnValue⟩
  have joinedProjection : projectData (applySourcePlan current incoming plan) =
      BarrierReduction.join state.currentRelease (projectData incoming) := by
    simpa only [currentProjection] using joined
  have emptyProjection : projectData (Data.mk emptyBase emptyOwn) = fun _ => 0 := by
    change emptyOwn = (0#u32, 0#u32) at emptyOwnValue
    funext component
    simp [projectData, emptyBaseValue, emptyOwnValue, baseModel, entryModel, value, entriesValue]
  step with source_release_move_refines_publication state (applySourcePlan current incoming plan)
      (Data.mk emptyBase emptyOwn) completed projectData outcome actor
      (deltaValue delta) arrivals.val (projectData incoming) model outcomeSuccess joinedProjection
      emptyProjection completedProjection
    as ⟨newCurrent, newCompleted, currentValue, completedValue⟩
  step with rust_finish_update duringPending source.phase duringTransactions outcome
    as ⟨afterPending, afterTransactions, afterPhase, pendingAfter, transactionsAfter, phaseAfter⟩
  refine ⟨?_, currentValue, completedValue⟩
  have expectedFrame := (intermediate_keeps_expected_and_phase state.counters
    (deltaValue delta) arrivals.val).2.2
  rw [←model] at expectedFrame
  have inputExpected := congrArg Counters.expected projection
  simp only [projectTransition, projectCounters] at expectedFrame inputExpected
  simp only [pendingAfter, transactionsAfter, phaseAfter, projectCounters]
  unfold advance publish
  rw [success]
  simp only
  rw [←model]
  simp only [projectTransition, projectCounters, expectedFrame, inputExpected]

end MemoryWrite.SparseClock

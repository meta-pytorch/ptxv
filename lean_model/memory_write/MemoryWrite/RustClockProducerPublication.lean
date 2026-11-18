/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RustClockProducerHistory

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite.SparseClock
open BarrierTransition

theorem source_release_from_producer_histories
    (state : FullState) (source : barrier.Counters)
    (delta : barrier.TransactionDelta) (arrivals : U32)
    (current incoming : Data) (completed : Array Data 2#usize) (sharedBase : Bool) (actor : Nat)
    (projection : projectCounters source = state.counters)
    (success : (transition state.counters (deltaValue delta) arrivals.val).fault = none)
    (currentProjection : projectData current = state.currentRelease)
    (completedProjection : projectSlots projectData state.completedRelease completed = state.completedRelease)
    (currentHistory : ProducerHistory current) (incomingHistory : ProducerHistory incoming)
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
  exact source_join_counter_release_sequence_refines_publication state source delta arrivals
    current incoming completed sharedBase actor projection success currentProjection completedProjection
    (producer_histories_have_sorted_bases current currentHistory)
    (producer_histories_have_sorted_bases incoming incomingHistory) hshared hsize

end MemoryWrite.SparseClock

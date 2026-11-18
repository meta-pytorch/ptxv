/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierBatchModel
import MemoryWrite.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.BarrierBatchBudget

def batchView (batch : barrier.ArrivalBatch) : Batch :=
  ⟨batch.pending.val, batch.arrivals.val, batch.valid⟩

def deltaValue : barrier.TransactionDelta → Int
  | .Add value => value.val
  | .Subtract value => -value.val

def contributionView (step : barrier.ArrivalStep) : Contribution :=
  ⟨step.pending_before.val, step.arrivals.val, deltaValue step.delta, step.completed, step.faulted⟩

@[step]
theorem source_initial (pending : U32) :
    barrier.arrival_batch pending ⦃ batch => batchView batch = initial pending.val ⦄ := by
  simp [barrier.arrival_batch, batchView, initial]

@[step]
theorem source_invalidate (batch : barrier.ArrivalBatch) :
    barrier.invalidate_arrival_batch batch ⦃ next => batchView next = invalidate (batchView batch) ⦄ := by
  simp [barrier.invalidate_arrival_batch, batchView, invalidate]

@[step]
theorem source_policy (batch : barrier.ArrivalBatch) :
    barrier.arrival_batch_writes batch ⦃ write => write = writes (batchView batch) ⦄ := by
  unfold barrier.arrival_batch_writes
  split
  · split
    · simp_all [writes, batchView]
    · step
      simp_all [writes, batchView]
  · simp_all [writes, batchView]

@[step]
theorem source_record (batch : barrier.ArrivalBatch) (step : barrier.ArrivalStep) :
    barrier.record_arrival_batch batch step
      ⦃ next => batchView next = record (batchView batch) (contributionView step) ⦄ := by
  cases hd : step.delta <;> cases hv : batch.valid <;>
    cases hc : step.completed <;> cases hf : step.faulted
  all_goals simp [barrier.record_arrival_batch, hd, hv, hc, hf,
    barrier.invalidate_arrival_batch, batchView, contributionView, deltaValue, record, invalidate]
  all_goals scalar_tac_preprocess
  all_goals split <;> simp_all
  all_goals split <;> simp_all
  all_goals step as ⟨o, ho⟩
  all_goals cases o <;> simp_all [WP.spec_ok]
  all_goals split <;> simp_all [WP.spec_ok]
  all_goals step as ⟨o2, ho2⟩
  all_goals cases o2 <;> simp_all [WP.spec_ok, U32.max]
  all_goals norm_num [U32.numBits] at *
  all_goals omega

theorem source_record_valid (batch : barrier.ArrivalBatch) (step : barrier.ArrivalStep) :
    barrier.record_arrival_batch batch step ⦃ next => next.valid = true →
      batch.valid = true ∧ step.completed = false ∧ step.faulted = false ∧
        0 < step.arrivals.val ∧ deltaValue step.delta = 0 ∧
        batch.arrivals.val + step.pending_before.val = batch.pending.val ∧
        next.arrivals.val = batch.arrivals.val + step.arrivals.val ∧
        next.pending.val = batch.pending.val ⦄ := by
  step with source_record batch step as ⟨next, matched⟩
  rename_i accepted
  have valid : (record (batchView batch) (contributionView step)).valid = true := by
    rw [← matched]
    exact accepted
  have result := record_valid (batchView batch) (contributionView step) valid
  rw [← matched] at result
  exact result

open BarrierReduction

theorem source_pair_preserves_continuation (state : State)
    (left right : barrier.ArrivalBatch) (leftActor rightActor : Nat)
    (leftRelease rightRelease : Clock) (suffix : List Action)
    (leftProjection : left.pending.val = state.barrier.pending)
    (rightProjection : right.pending.val = state.barrier.pending)
    (different : leftActor ≠ rightActor) :
    barrier.arrival_batch_writes left ⦃ leftWrite =>
      barrier.arrival_batch_writes right ⦃ rightWrite =>
        leftWrite = false → rightWrite = false →
          execute state (.arrival leftActor left.arrivals.val leftRelease ::
            .arrival rightActor right.arrivals.val rightRelease :: suffix) =
          execute state (.arrival rightActor right.arrivals.val rightRelease ::
            .arrival leftActor left.arrivals.val leftRelease :: suffix) ⦄ ⦄ := by
  step with source_policy left as ⟨leftWrite, leftMatched⟩
  step with source_policy right as ⟨rightWrite, rightMatched⟩
  rename_i leftRead rightRead
  apply read_pair_preserves_continuation state (batchView left) (batchView right)
    leftActor rightActor leftRelease rightRelease suffix leftProjection rightProjection different
  · simpa [leftMatched] using leftRead
  · simpa [rightMatched] using rightRead

end MemoryWrite.BarrierBatchBudget

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierReduction

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.BarrierBatchBudget

structure Batch where
  pending : Nat
  arrivals : Nat
  valid : Bool
  deriving DecidableEq, Repr

structure Contribution where
  pendingBefore : Nat
  arrivals : Nat
  delta : Int
  completed : Bool
  faulted : Bool
  deriving DecidableEq, Repr

def initial (pending : Nat) : Batch := ⟨pending, 0, true⟩

def invalidate (batch : Batch) : Batch := {batch with valid := false}

def record (batch : Batch) (step : Contribution) : Batch :=
  if batch.valid = false then batch else
  if step.completed = true ∨ step.faulted = true ∨ step.arrivals = 0 ∨ step.delta ≠ 0 ∨
      batch.pending < batch.arrivals ∨ batch.pending - batch.arrivals ≠ step.pendingBefore ∨
      4294967295 < batch.arrivals + step.arrivals then invalidate batch else
    {batch with arrivals := batch.arrivals + step.arrivals}

def writes (batch : Batch) : Bool :=
  !batch.valid || decide (batch.arrivals = 0) || decide (batch.pending / 2 < batch.arrivals)

theorem initial_total (pending : Nat) : (initial pending).arrivals = 0 := rfl

theorem invalid_is_sticky (batch : Batch) (step : Contribution) (bad : batch.valid = false) :
    record batch step = batch := by simp [record, bad]

theorem record_valid (batch : Batch) (step : Contribution)
    (good : (record batch step).valid = true) :
    batch.valid = true ∧ step.completed = false ∧ step.faulted = false ∧
      0 < step.arrivals ∧ step.delta = 0 ∧
      batch.arrivals + step.pendingBefore = batch.pending ∧
      (record batch step).arrivals = batch.arrivals + step.arrivals ∧
      (record batch step).pending = batch.pending := by
  unfold record at good ⊢
  split_ifs at good ⊢ <;> simp_all [invalidate]
  all_goals omega

theorem record_pending (batch : Batch) (step : Contribution) :
    (record batch step).pending = batch.pending := by
  unfold record
  split_ifs <;> rfl

theorem collected_valid_total (steps : List Contribution) (batch : Batch)
    (good : (steps.foldl record batch).valid = true) :
    batch.valid = true ∧
      (steps.foldl record batch).arrivals = batch.arrivals + (steps.map Contribution.arrivals).sum ∧
      (steps.foldl record batch).pending = batch.pending := by
  induction steps generalizing batch with
  | nil => simpa using good
  | cons first rest ih =>
    obtain ⟨nextGood, total, pending⟩ := ih (record batch first) good
    obtain ⟨oldGood, _, _, _, _, _, count, _⟩ := record_valid batch first nextGood
    refine ⟨oldGood, ?_, ?_⟩
    · simp only [List.foldl_cons, List.map_cons, List.sum_cons]
      omega
    · simpa only [List.foldl_cons, record_pending] using pending

theorem initialized_collection_counts_every_contribution (steps : List Contribution) (pending : Nat)
    (good : (steps.foldl record (initial pending)).valid = true) :
    (steps.foldl record (initial pending)).arrivals = (steps.map Contribution.arrivals).sum ∧
      (steps.foldl record (initial pending)).pending = pending := by
  obtain ⟨_, count, origin⟩ := collected_valid_total steps (initial pending) good
  simpa [initial] using And.intro count origin

theorem writes_false_iff (batch : Batch) :
    writes batch = false ↔ batch.valid = true ∧ 0 < batch.arrivals ∧
      2 * batch.arrivals ≤ batch.pending := by
  cases hv : batch.valid <;> simp [writes, hv]
  omega

theorem read_pair_budget (left right : Batch)
    (samePending : left.pending = right.pending)
    (leftRead : writes left = false) (rightRead : writes right = false) :
    left.arrivals + right.arrivals ≤ left.pending ∧
      left.arrivals < left.pending ∧ right.arrivals < left.pending := by
  have hleft := (writes_false_iff left).mp leftRead
  have hright := (writes_false_iff right).mp rightRead
  omega

open BarrierReduction

theorem read_pair_full_state (state : State) (left right : Batch)
    (leftActor rightActor : Nat) (leftRelease rightRelease : Clock)
    (leftProjection : left.pending = state.barrier.pending)
    (rightProjection : right.pending = state.barrier.pending)
    (different : leftActor ≠ rightActor)
    (leftRead : writes left = false) (rightRead : writes right = false) :
    arrive (arrive state leftActor left.arrivals leftRelease)
        rightActor right.arrivals rightRelease =
      arrive (arrive state rightActor right.arrivals rightRelease)
        leftActor left.arrivals leftRelease := by
  obtain ⟨total, hn, hm⟩ := read_pair_budget left right
    (leftProjection.trans rightProjection.symm) leftRead rightRead
  apply arrive_commute state leftActor rightActor left.arrivals right.arrivals
    leftRelease rightRelease different
  all_goals omega

theorem read_pair_preserves_continuation (state : State) (left right : Batch)
    (leftActor rightActor : Nat) (leftRelease rightRelease : Clock) (suffix : List Action)
    (leftProjection : left.pending = state.barrier.pending)
    (rightProjection : right.pending = state.barrier.pending)
    (different : leftActor ≠ rightActor)
    (leftRead : writes left = false) (rightRead : writes right = false) :
    execute state (.arrival leftActor left.arrivals leftRelease ::
        .arrival rightActor right.arrivals rightRelease :: suffix) =
      execute state (.arrival rightActor right.arrivals rightRelease ::
        .arrival leftActor left.arrivals leftRelease :: suffix) := by
  simp only [execute_arrival]
  rw [read_pair_full_state state left right leftActor rightActor leftRelease rightRelease
    leftProjection rightProjection different leftRead rightRead]

def oversubscribed : State :=
  ⟨⟨4, 4, 0, false, fun _ => 0, fun _ _ => 0⟩,
    fun _ => none, fun _ => 0, false, [], fun _ _ => 0⟩

theorem merely_individually_fitting_batches_lose_fault_actor :
    (arrive (arrive oversubscribed 0 2 (fun _ => 0)) 1 3 (fun _ => 0)).findings = [1] ∧
    (arrive (arrive oversubscribed 1 3 (fun _ => 0)) 0 2 (fun _ => 0)).findings = [0] ∧
    writes ⟨4, 2, true⟩ = false ∧ writes ⟨4, 3, true⟩ = true := by decide

end MemoryWrite.BarrierBatchBudget

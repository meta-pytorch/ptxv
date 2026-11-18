/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierReduction
import MemoryWrite.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.BarrierUnitArrival

open BarrierReduction

def writesBatch (delta : Int) (pending arrivals : Nat)
    (completed faulted priorSamePhase : Bool) : Bool :=
  faulted || priorSamePhase || decide (1 < arrivals) ||
    decide (0 < arrivals ∧ pending ≤ 1) || completed || decide (delta ≠ 0)

def deltaValue : barrier.TransactionDelta → Int
  | .Add value => value.val
  | .Subtract value => -value.val

theorem read_characterization (delta : Int) (pending arrivals : Nat)
    (completed faulted priorSamePhase : Bool) :
    writesBatch delta pending arrivals completed faulted priorSamePhase = false ↔
      faulted = false ∧ priorSamePhase = false ∧ completed = false ∧ delta = 0 ∧
        (arrivals = 0 ∨ (arrivals = 1 ∧ 1 < pending)) := by
  cases completed <;> cases faulted <;> cases priorSamePhase <;> simp [writesBatch]
  omega

@[step]
theorem rust_batch_policy (delta : barrier.TransactionDelta) (pending arrivals : U32)
    (completed faulted priorSamePhase : Bool) :
    barrier.batch_writes_phase_membership delta pending arrivals completed faulted priorSamePhase
      ⦃ writes => writes = writesBatch (deltaValue delta) pending.val arrivals.val
          completed faulted priorSamePhase ⦄ := by
  cases delta <;> cases faulted <;> cases priorSamePhase <;> cases completed <;>
    simp [barrier.batch_writes_phase_membership, writesBatch, deltaValue]
  all_goals scalar_tac_preprocess
  all_goals split_ifs <;> simp_all [WP.spec_ok]
  all_goals first
    | omega
    | (apply Bool.eq_iff_iff.mpr; simp_all [bne_iff_ne])

theorem same_actor_unit_pair_commutes (state : State) (actor : Nat) (a b : Clock)
    (room : 2 ≤ state.barrier.pending) :
    arrive (arrive state actor 1 a) actor 1 b =
      arrive (arrive state actor 1 b) actor 1 a := by
  have first : ¬state.barrier.pending < 1 := by omega
  have second : ¬state.barrier.pending - 1 < 1 := by omega
  have bound : 1 < state.barrier.pending := by omega
  have ba := decrement_noncompleting state.barrier 1 a bound
  have bb := decrement_noncompleting state.barrier 1 b bound
  have barrierEq := decrement_commute state.barrier 1 1 a b bound bound (by omega)
  simp only [arrive, ite_false, first]
  simp only [ba, bb, second, ite_false]
  apply State.ext
  · simpa only [ba, bb] using barrierEq
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl

theorem unit_pair_commutes (state : State) (left right : Nat) (a b : Clock)
    (room : 2 ≤ state.barrier.pending) :
    arrive (arrive state left 1 a) right 1 b =
      arrive (arrive state right 1 b) left 1 a := by
  by_cases same : left = right
  · subst right
    exact same_actor_unit_pair_commutes state left a b room
  · exact arrive_commute state left right 1 1 a b same (by omega) (by omega) (by omega)

theorem unit_pair_preserves_execution (state : State) (left right : Nat) (a b : Clock)
    (suffix : List Action) (room : 2 ≤ state.barrier.pending) :
    execute state (.arrival left 1 a :: .arrival right 1 b :: suffix) =
      execute state (.arrival right 1 b :: .arrival left 1 a :: suffix) := by
  simp only [execute_arrival]
  rw [unit_pair_commutes state left right a b room]

theorem source_unit_read_preserves_execution (delta : barrier.TransactionDelta)
    (pending : U32) (completed faulted priorSamePhase : Bool) (state : State)
    (left right : Nat) (a b : Clock) (suffix : List Action)
    (projection : pending.val = state.barrier.pending) :
    barrier.batch_writes_phase_membership delta pending 1#u32 completed faulted priorSamePhase
      ⦃ writes => writes = false → deltaValue delta = 0 ∧ faulted = false ∧
        priorSamePhase = false ∧ completed = false ∧
        execute state (.arrival left 1 a :: .arrival right 1 b :: suffix) =
          execute state (.arrival right 1 b :: .arrival left 1 a :: suffix) ⦄ := by
  step with rust_batch_policy delta pending 1#u32 completed faulted priorSamePhase as ⟨writes, hwrites⟩
  rename_i read
  have hread := (read_characterization (deltaValue delta) pending.val 1 completed faulted priorSamePhase).mp
    (by simpa [hwrites] using read)
  refine ⟨hread.2.2.2.1, hread.1, hread.2.1, hread.2.2.1, ?_⟩
  apply unit_pair_preserves_execution state left right a b suffix
  rcases hread.2.2.2.2 with impossible | room
  · omega
  · rw [← projection]
    omega

theorem grouped_units_remain_writes (pending : Nat) :
    writesBatch 0 pending 1 false false true = true := by
  simp [writesBatch]

theorem counted_arrivals_remain_writes (pending arrivals : Nat) (counted : 1 < arrivals) :
    writesBatch 0 pending arrivals false false false = true := by
  simp [writesBatch, counted]

theorem transaction_changes_remain_writes (delta : Int) (pending arrivals : Nat)
    (changes : delta ≠ 0) : writesBatch delta pending arrivals false false false = true := by
  simp [writesBatch, changes]

theorem rejected_operations_remain_writes (delta : Int) (pending arrivals : Nat)
    (completed priorSamePhase : Bool) :
    writesBatch delta pending arrivals completed true priorSamePhase = true := by
  simp [writesBatch]

def debtState : State :=
  ⟨⟨2, 1, 0, false, fun _ => 0, fun _ _ => 0⟩,
    fun _ => none, fun _ => 0, false, [], fun _ _ => 0⟩

theorem outstanding_transactions_do_not_make_one_pending_arrival_commute :
    (arrive (arrive debtState 0 1 (fun _ => 0)) 1 1 (fun _ => 0)).findings = [1] ∧
    (arrive (arrive debtState 1 1 (fun _ => 0)) 0 1 (fun _ => 0)).findings = [0] ∧
    writesBatch 0 1 1 false false false = true := by decide

end MemoryWrite.BarrierUnitArrival

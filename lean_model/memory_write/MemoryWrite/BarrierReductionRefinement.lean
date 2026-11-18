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

namespace MemoryWrite.BarrierReduction

@[step]
theorem rust_arrival_pending (pending arrivals : U32) :
    reduction.arrival_pending pending arrivals
      ⦃ result => result.map UScalar.val =
        if pending.val < arrivals.val then none else some (pending.val - arrivals.val) ⦄ := by
  unfold reduction.arrival_pending
  split
  · rename_i h
    have hval : pending.val < arrivals.val := by scalar_tac
    simp [hval]
  · rename_i h
    have hval : ¬pending.val < arrivals.val := by scalar_tac
    step as ⟨remaining, hremaining⟩; try scalar_tac
    simp [hval, hremaining]

@[step]
theorem rust_phase_completes (pending : U32) (txZero : Bool) :
    reduction.phase_completes pending txZero
      ⦃ result => result = decide (pending.val = 0 ∧ txZero = true) ⦄ := by
  unfold reduction.phase_completes
  split
  · rename_i h
    have hzero : pending.val = 0 := by scalar_tac
    simp [hzero]
  · rename_i h
    have hzero : pending.val ≠ 0 := by scalar_tac
    simp [hzero]

@[step]
theorem rust_arrival_writes_pending (arrivals : U32) :
    reduction.arrival_writes_pending arrivals
      ⦃ result => result = decide (0 < arrivals.val) ⦄ := by
  simp only [reduction.arrival_writes_pending, WP.spec_ok]
  scalar_tac

@[step]
theorem rust_allocation_ready (request : reduction.AllocationRequest) :
    reduction.allocation_ready request
      ⦃ result => result = match request with
        | .Disabled | .Invalid => true
        | .Valid available => available ⦄ := by
  cases request <;> simp [reduction.allocation_ready]

 theorem source_historical_count_footprint :
    reduction.arrival_writes_pending 3#u32 ⦃ result => result = true ⦄ := by
  exact rust_arrival_writes_pending 3#u32

 theorem source_disabled_allocation_progress_and_frame (s : AllocationState)
    (commit : Allocator → Allocator) :
    reduction.allocation_ready .Disabled
      ⦃ ready => ready = true ∧ ∃ next,
        allocationStep s .disabled commit = some next ∧ next.pc = s.pc + 1 ∧
        next.allocator = s.allocator ∧ next.findings = s.findings ⦄ := by
  simp only [reduction.allocation_ready, WP.spec_ok]
  exact ⟨True.intro, disabled_allocation_progress_and_frame s commit⟩

theorem source_noncompleting_arrival_matches_model (s : State) (pending count : U32)
    (actor : Nat) (release : Clock) (projection : pending.val = s.barrier.pending)
    (noncompleting : count.val < s.barrier.pending) :
    reduction.arrival_pending pending count
      ⦃ result => result.map UScalar.val = some (arrive s actor count.val release).barrier.pending ⦄ := by
  have noOver : ¬s.barrier.pending < count.val := by omega
  have modelPending : (arrive s actor count.val release).barrier.pending =
      s.barrier.pending - count.val := by
    simp only [arrive, ite_false, noOver,
      decrement_noncompleting s.barrier count.val release noncompleting]
  step with rust_arrival_pending pending count as ⟨result, hresult⟩
  simpa [projection, noOver, modelPending] using hresult

 theorem source_completion_selects_model_phase (barrier : Barrier) (pending : U32)
    (projection : pending.val = barrier.pending) :
    reduction.phase_completes pending barrier.txZero
      ⦃ completes => completes = true →
        (complete barrier).phase = barrier.phase + 1 ∧
        (complete barrier).pending = barrier.expected ∧
        (complete barrier).completedRelease (barrier.phase % 2) = barrier.currentRelease ∧
        (complete barrier).currentRelease = fun _ => 0 ⦄ := by
  step with rust_phase_completes pending barrier.txZero as ⟨result, hresult⟩
  rename_i completes
  have condition : barrier.pending = 0 ∧ barrier.txZero = true := by
    simpa [hresult, projection] using completes
  simp [complete, condition, update]

theorem source_unwritten_pending_count_is_frame (pending arrivals : U32) :
    (do
      let writes ← reduction.arrival_writes_pending arrivals
      let remaining ← reduction.arrival_pending pending arrivals
      pure (writes, remaining))
      ⦃ result => result.1 = false → result.2 = some pending ⦄ := by
  step with rust_arrival_writes_pending arrivals as ⟨writes, hwrites⟩
  step with rust_arrival_pending pending arrivals as ⟨remaining, hremaining⟩
  intro unwritten
  have zero : arrivals.val = 0 := by
    have : ¬0 < arrivals.val := by simpa [hwrites] using unwritten
    omega
  have hsome : remaining.map UScalar.val = some pending.val := by
    simpa [zero] using hremaining
  cases remaining with
  | none => simp at hsome
  | some value =>
    simp only [Option.map_some, Option.some.injEq] at hsome
    have same : value = pending := by scalar_tac
    simp [same]

end MemoryWrite.BarrierReduction

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierClockMove

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.BarrierTransition

def keyModulus : Nat := 2 ^ 20

def phaseKey (offset phase : Nat) : Nat :=
  offset * keyModulus + phase % keyModulus

 theorem phase_key_stays_in_object (offset phase : Nat) :
    offset * keyModulus ≤ phaseKey offset phase ∧
    phaseKey offset phase < (offset + 1) * keyModulus := by
  have remainder := Nat.mod_lt phase (show 0 < keyModulus by decide)
  unfold phaseKey keyModulus at *
  omega

 theorem phase_key_identifies_object_and_phase (firstOffset firstPhase secondOffset secondPhase : Nat) :
    phaseKey firstOffset firstPhase = phaseKey secondOffset secondPhase ↔
      firstOffset = secondOffset ∧ firstPhase % keyModulus = secondPhase % keyModulus := by
  have first := phase_key_stays_in_object firstOffset firstPhase
  have second := phase_key_stays_in_object secondOffset secondPhase
  unfold phaseKey keyModulus at *
  omega

@[step]
theorem rust_phase_key (offset phase : U32) :
    barrier.phase_key offset phase ⦃ key => key.val = phaseKey offset.val phase.val ⦄ := by
  unfold barrier.phase_key
  step with UScalar.cast_inBounds_spec .U64 offset (by scalar_tac) as ⟨wide, hwide⟩
  step as ⟨base, hbase⟩; try scalar_tac
  step as ⟨part, hpart⟩; try scalar_tac
  step with UScalar.cast_inBounds_spec .U64 part (by scalar_tac) as ⟨widePart, hwidePart⟩
  step as ⟨key, hkey⟩; try scalar_tac
  simp only [phaseKey, keyModulus]
  scalar_tac

@[step]
theorem rust_next_arrival_key (offset : U32) (outcome : barrier.Transition) :
    barrier.next_arrival_key offset outcome
      ⦃ key => key.val = phaseKey offset.val outcome.after.phase.val ⦄ := by
  unfold barrier.next_arrival_key
  exact rust_phase_key offset outcome.after.phase

 theorem source_completion_dependency_matches_next_arrival (offset : U32)
    (source : barrier.Counters) (delta : barrier.TransactionDelta) (arrivals : U32)
    (completed : (transition (projectCounters source) (deltaValue delta) arrivals.val).completed = true) :
    (do
      let outcome ← barrier.transition source delta arrivals
      let written ← barrier.next_arrival_key offset outcome
      let read ← barrier.phase_key offset outcome.after.phase
      pure (written, read))
      ⦃ result => result.1 = result.2 ∧
        result.1.val = phaseKey offset.val ((source.phase.val + 1) % phaseModulus) ⦄ := by
  step with rust_transition source delta arrivals as ⟨outcome, model⟩
  step with rust_next_arrival_key offset outcome as ⟨written, writeValue⟩
  step with rust_phase_key offset outcome.after.phase as ⟨read, readValue⟩
  have same : written = read := UScalar.eq_of_val_eq (writeValue.trans readValue.symm)
  refine ⟨same, ?_⟩
  have advanced := (completion_resets_expected (projectCounters source)
    (deltaValue delta) arrivals.val completed).2.1
  rw [← model] at advanced
  change outcome.after.phase.val = (source.phase.val + 1) % phaseModulus at advanced
  simpa only [advanced] using writeValue


 theorem source_phase_key_stays_in_object (offset phase : U32) :
    barrier.phase_key offset phase
      ⦃ key => offset.val * keyModulus ≤ key.val ∧
        key.val < (offset.val + 1) * keyModulus ⦄ := by
  apply WP.spec_mono (rust_phase_key offset phase)
  intro key projected
  rw [projected]
  exact phase_key_stays_in_object offset.val phase.val

 theorem old_successor_packing_crosses_into_the_next_object :
    phaseKey 1024 (keyModulus - 1) + 1 = phaseKey 1025 0 ∧
    phaseKey 1024 keyModulus ≠ phaseKey 1025 0 := by decide

end MemoryWrite.BarrierTransition

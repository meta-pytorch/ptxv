/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificatePublicationSource

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction.GuardCertificate

def RowMerged (previous incoming after : Slice Observation) : Prop :=
  after.length = previous.length ∧ ∀ pc < previous.length,
    after[pc]! = mergeObservation previous[pc]! incoming[pc]!

def MergeInvariant (original incoming current : Slice Observation) (position : Usize) : Prop :=
  current.length = original.length ∧ position.val ≤ original.length ∧
  ∀ pc < original.length, current[pc]! =
    if pc < position.val then mergeObservation original[pc]! incoming[pc]! else original[pc]!

@[step]
theorem rust_merge_rows_loop (original incoming current : Slice Observation) (position : Usize)
    (lengths : original.length = incoming.length)
    (initial : MergeInvariant original incoming current position) :
    guard_certificate.merge_rows_loop current incoming position
      ⦃ after => RowMerged original incoming after ⦄ := by
  unfold guard_certificate.merge_rows_loop
  apply loop.spec_decr_nat
    (fun state : Slice Observation × Usize => original.length - state.2.val)
    (fun state : Slice Observation × Usize => MergeInvariant original incoming state.1 state.2)
  · rintro ⟨current, position⟩ invariant
    obtain ⟨sameLength, bound, meaning⟩ := invariant
    unfold guard_certificate.merge_rows_loop.body
    dsimp only at *
    split
    · rename_i test
      have inside : position.val < current.length := by
        simpa only [UScalar.lt_equiv, Slice.len_val] using test
      have originalInside : position.val < original.length := by omega
      have incomingInside : position.val < incoming.length := by omega
      step with Slice.index_usize_spec current position inside as ⟨old, oldAt⟩
      step with Slice.index_usize_spec incoming position incomingInside as ⟨value, valueAt⟩
      have oldValue : old = original[position.val]! := by
        have priorMeaning := meaning position.val originalInside
        simp only [Nat.lt_irrefl, ite_false] at priorMeaning
        rw [oldAt, ←priorMeaning]
        simpa only [Slice.getElem!_Nat_eq] using List.Inhabited_getElem_eq_getElem! current.val position.val inside
      have incomingValue : value = incoming[position.val]! := by
        rw [valueAt]
        simpa only [Slice.getElem!_Nat_eq] using List.Inhabited_getElem_eq_getElem! incoming.val position.val incomingInside
      step with rust_merge_grid old value as ⟨merged, mergedAt⟩
      step with Slice.update_spec current position merged inside as ⟨updated, updatedAt⟩
      step with Usize.add_spec (x := position) (y := 1#usize) (by scalar_tac) as ⟨next, nextAt⟩
      subst updated
      have nextValue : next.val = position.val + 1 := by scalar_tac
      refine ⟨?_, by omega⟩
      refine ⟨?_, by omega, ?_⟩
      · simpa using sameLength
      · intro pc pcInside
        by_cases same : position.val = pc
        · subst pc
          rw [Slice.getElem!_Nat_set_eq current position position.val merged ⟨rfl, inside⟩]
          simp only [nextValue, Nat.lt_add_one, le_refl, ite_true]
          simpa [oldValue, incomingValue] using mergedAt
        · rw [Slice.getElem!_Nat_set_ne current position pc merged same, meaning pc pcInside]
          have equivalent : pc < next.val ↔ pc < position.val := by omega
          simp only [equivalent]
    · rename_i outside
      have outside' : ¬position.val < current.length := by
        simpa only [UScalar.lt_equiv, Slice.len_val] using outside
      have complete : position.val = original.length := by omega
      simp only [WP.spec_ok, RowMerged]
      refine ⟨sameLength, ?_⟩
      intro pc pcInside
      simpa [complete, pcInside] using meaning pc pcInside
  · exact initial

@[step]
theorem rust_merge_rows (previous incoming : Slice Observation)
    (lengths : previous.length = incoming.length) :
    guard_certificate.merge_rows previous incoming
      ⦃ result => result.1 = true ∧ RowMerged previous incoming result.2 ⦄ := by
  unfold guard_certificate.merge_rows
  have same : Slice.len previous = Slice.len incoming := by
    apply (UScalar.eq_equiv _ _).mpr
    simpa only [Slice.len_val] using lengths
  rw [same]
  simp only [bne_self_eq_false, Bool.false_eq_true, ite_false]
  step with rust_merge_rows_loop previous incoming previous 0#usize lengths
    (by simp [MergeInvariant]) as ⟨after, afterMeaning⟩
  exact afterMeaning

end Instruction.GuardCertificate

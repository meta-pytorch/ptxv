/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierReachModel

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.BarrierReach

def MergePost (successors : List Usize) (reach : List U32) (initial merged : U32) : Prop :=
  Includes initial merged ∧ ∀ destination ∈ successors, Includes reach[destination.val]! merged

@[step]
theorem rust_merge_successors (successors : Slice Usize) (reach : Slice U32) (initial : U32)
    (bounded : ∀ destination ∈ successors.val, destination.val < reach.val.length) :
    barrier_reach.merge_successors successors reach initial
      ⦃ merged => MergePost successors.val reach.val initial merged ⦄ := by
  unfold barrier_reach.merge_successors barrier_reach.merge_successors_loop
  apply loop.spec_decr_nat
    (fun state : U32 × Usize => successors.val.length - state.2.val)
    (fun state => state.2.val ≤ successors.val.length ∧ Includes initial state.1 ∧
      ∀ position < state.2.val, Includes reach.val[successors.val[position]!.val]! state.1)
  · rintro ⟨merged, position⟩ ⟨positionBound, initialIncluded, prefixIncluded⟩
    dsimp only at positionBound initialIncluded prefixIncluded
    have positionLe : position.val ≤ successors.val.length := positionBound
    unfold barrier_reach.merge_successors_loop.body
    by_cases inside : position.val < successors.val.length
    · have test : position < Slice.len successors := by scalar_tac
      simp only [test, ite_true]
      step with Slice.index_usize_spec successors position inside as ⟨destination, destinationEq⟩
      have destinationAt : destination = successors.val[position.val]! := by
        rw [destinationEq, List.Inhabited_getElem_eq_getElem! _ _ inside]
      have destinationMember : destination ∈ successors.val := by
        rw [destinationEq]
        exact List.getElem_mem inside
      have destinationBound := bounded destination destinationMember
      step with Slice.index_usize_spec reach destination destinationBound as ⟨word, wordEq⟩
      have wordAt : word = reach.val[destination.val]! := by
        rw [wordEq, List.Inhabited_getElem_eq_getElem! _ _ destinationBound]
      step with rust_or merged word as ⟨updated, updatedEq⟩
      step with Usize.add_spec (x := position) (y := 1#usize) (by scalar_tac) as ⟨next, nextEq⟩
      refine ⟨by scalar_tac, ?_, ?_, by scalar_tac⟩
      · rw [updatedEq]
        exact initialIncluded.trans (includes_or_left merged word)
      · intro previous previousBound
        by_cases same : previous = position.val
        · subst previous
          rw [←destinationAt, ←wordAt, updatedEq]
          exact includes_or_right merged word
        · have previousInside : previous < position.val := by scalar_tac
          rw [updatedEq]
          exact (prefixIncluded previous previousInside).trans (includes_or_left merged word)
    · have test : ¬position < Slice.len successors := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      refine ⟨initialIncluded, ?_⟩
      intro destination member
      obtain ⟨entryPosition, atBound, atEq⟩ := List.getElem_of_mem member
      have includedPrefix := prefixIncluded entryPosition (by omega)
      simpa only [←List.Inhabited_getElem_eq_getElem! _ _ atBound, atEq] using includedPrefix
  · exact ⟨by scalar_tac, Includes.refl initial, by intro position bound; scalar_tac⟩

end MemoryWrite.BarrierReach

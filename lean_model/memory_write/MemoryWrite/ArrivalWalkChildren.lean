/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.ArrivalWalkFrontier

open Aeneas Aeneas.Std Result PtxvMemoryWrite
set_option autoImplicit false
set_option maxHeartbeats 5000000
namespace ArrivalWalk

@[step]
theorem rust_push_children {State : Type} (pending children : arrival_walk.Frontier State)
    (pendingValues childrenValues : List State) (pendingRep : Represents pending pendingValues)
    (childrenRep : Represents children childrenValues) :
    arrival_walk.push_children pending children ⦃ success result =>
      ∃ values, Represents result values ∧
        (success = true → values = pendingValues ++ childrenValues.reverse) ⦄ := by
  unfold arrival_walk.push_children arrival_walk.push_children_loop
  apply loop.spec_decr_nat
    (fun state : arrival_walk.Frontier State × arrival_walk.Frontier State => state.1.len.val)
    (fun state => ∃ remaining values, Represents state.1 remaining ∧ Represents state.2 values ∧
      values ++ remaining.reverse = pendingValues ++ childrenValues.reverse)
  · rintro ⟨currentChildren, currentPending⟩ ⟨remaining, values, childRep, valueRep, combined⟩
    dsimp only at childRep valueRep combined
    unfold arrival_walk.push_children_loop.body
    rcases empty_or_append remaining with empty | ⟨prior, state, last⟩
    · subst remaining
      step with rust_pop_empty currentChildren childRep as ⟨popped, after, poppedEq, afterEq⟩
      simp only [poppedEq, WP.spec_ok]
      exact ⟨values, valueRep, fun _ => by simpa using combined⟩
    · subst remaining
      step with rust_pop_append currentChildren prior state childRep
        as ⟨popped, after, poppedEq, afterRep, decreased⟩
      simp only [poppedEq]
      step with rust_push currentPending values valueRep state as ⟨success, result, pushed⟩
      rcases pushed with ⟨failed, same⟩ | ⟨succeeded, resultRep⟩
      · simp only [failed, Bool.false_eq_true, ite_false, WP.spec_ok]
        exact ⟨values, by rw [same]; exact valueRep, by simp⟩
      · simp only [succeeded, ite_true, WP.spec_ok]
        refine ⟨⟨prior, values ++ [state], afterRep, resultRep, ?_⟩, ?_⟩
        · simpa [List.reverse_append, List.append_assoc] using combined
        · omega
  · exact ⟨childrenValues, pendingValues, childrenRep, pendingRep, rfl⟩

end ArrivalWalk

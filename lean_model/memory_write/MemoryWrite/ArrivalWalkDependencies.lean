/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.ArrivalWalkRefinement

open Aeneas Aeneas.Std Result PtxvMemoryWrite
set_option autoImplicit false
set_option maxHeartbeats 5000000
namespace ArrivalWalk

@[step]
theorem rust_conflict_loop {D : Type} (reader : arrival_walk.ReadDependencies D) (dependencies : D)
    (count index : Usize) (conflicts : Nat → Prop)
    (reads : ∀ position : Usize, position.val < count.val →
      reader.conflict dependencies position ⦃ result => result = true ↔ conflicts position.val ⦄)
    (inside : index.val ≤ count.val) (prior : ∀ position < index.val, ¬conflicts position) :
    arrival_walk.has_conflict_loop reader dependencies count index ⦃ result =>
      result = true ↔ ∃ position < count.val, conflicts position ⦄ := by
  unfold arrival_walk.has_conflict_loop
  apply loop.spec_decr_nat (fun current : Usize => count.val - current.val)
    (fun current => current.val ≤ count.val ∧ ∀ position < current.val, ¬conflicts position)
  · intro current invariant
    rcases invariant with ⟨bound, checked⟩
    unfold arrival_walk.has_conflict_loop.body
    by_cases remaining : current < count
    · simp only [remaining, ite_true]
      have currentBound : current.val < count.val := by scalar_tac
      step with reads current currentBound as ⟨result, observed⟩
      cases result with
      | true =>
        simp only [ite_true, WP.spec_ok]
        exact ⟨fun _ => ⟨current.val, currentBound, observed.mp rfl⟩, fun _ => trivial⟩
      | false =>
        simp only [Bool.false_eq_true, ite_false]
        step with Usize.add_spec (x := current) (y := 1#usize) (by scalar_tac) as ⟨next, nextEq⟩
        refine ⟨by omega, ?_, by omega⟩
        intro position beforeNext
        by_cases previous : position < current.val
        · exact checked position previous
        · have same : position = current.val := by omega
          rw [same]
          intro conflict
          have impossible := observed.mpr conflict
          cases impossible
    · simp only [remaining, ite_false, WP.spec_ok]
      constructor
      · intro impossible; cases impossible
      · rintro ⟨position, within, conflict⟩
        have all : count.val ≤ current.val := by scalar_tac
        exact False.elim (checked position (by omega) conflict)
  · exact ⟨inside, prior⟩

@[step]
theorem rust_has_conflict {D : Type} (reader : arrival_walk.ReadDependencies D) (dependencies : D)
    (count : Nat) (conflicts : Nat → Prop)
    (countSource : reader.count dependencies ⦃ result => result.val = count ⦄)
    (reads : ∀ position : Usize, position.val < count →
      reader.conflict dependencies position ⦃ result => result = true ↔ conflicts position.val ⦄) :
    arrival_walk.has_conflict reader dependencies ⦃ result => result = true ↔ ∃ position < count, conflicts position ⦄ := by
  unfold arrival_walk.has_conflict
  step with countSource as ⟨actual, countEq⟩
  simpa only [countEq] using rust_conflict_loop reader dependencies actual 0#usize conflicts
    (by simpa only [countEq] using reads) (by scalar_tac) (by simp)

def SettledDependencies {Register : Type} (used : List Register) (written : Register → Prop) : Prop :=
  ∀ register ∈ used, ¬written register

theorem actual_dependency_scan_certifies_settlement {D Register : Type}
    (reader : arrival_walk.ReadDependencies D) (dependencies : D) (used : List Register) (written : Register → Prop)
    (countSource : reader.count dependencies ⦃ result => result.val = used.length ⦄)
    (reads : ∀ position : Usize, position.val < used.length →
      reader.conflict dependencies position ⦃ result => result = true ↔
        ∃ register, used[position.val]? = some register ∧ written register ⦄)
    (scanned : arrival_walk.has_conflict reader dependencies = .ok false) : SettledDependencies used written := by
  have source := rust_has_conflict reader dependencies used.length
    (fun position => ∃ register, used[position]? = some register ∧ written register) countSource reads
  simp only [scanned, WP.spec_ok, Bool.false_eq_true, false_iff] at source
  intro register member overlap
  obtain ⟨position, bound, atPosition⟩ := List.mem_iff_getElem.mp member
  apply source
  refine ⟨position, bound, register, ?_, overlap⟩
  simpa only [List.getElem?_eq_getElem bound] using congrArg some atPosition

end ArrivalWalk

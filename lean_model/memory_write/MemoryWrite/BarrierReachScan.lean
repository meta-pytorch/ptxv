/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierReachMerge

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite.BarrierReach

def ProcessedClosed (successors : List (alloc.vec.Vec Usize)) (reach : List U32) (lower : Nat) : Prop :=
  ∀ pc, lower ≤ pc → pc < reach.length → ∀ destination ∈ successors[pc]!.val,
    Includes reach[destination.val]! reach[pc]!

def ScanInvariant (successors : List (alloc.vec.Vec Usize)) (initial current : List U32)
    (changed : Bool) (pc : Nat) : Prop :=
  current.length = initial.length ∧ pc ≤ initial.length ∧ Grows initial current ∧
  (changed = true → (facts initial.length initial).card < (facts initial.length current).card) ∧
  (changed = false → current = initial ∧ ProcessedClosed successors current pc)

def ScanPost (successors : List (alloc.vec.Vec Usize)) (initial current : List U32) (changed : Bool) : Prop :=
  current.length = initial.length ∧ Grows initial current ∧
  (changed = true → (facts initial.length initial).card < (facts initial.length current).card) ∧
  (changed = false → Closed successors current)

@[step]
theorem rust_scan (successors : Slice (alloc.vec.Vec Usize)) (initial : alloc.vec.Vec U32)
    (bounded : EdgesBounded successors.val initial.val.length) :
    barrier_reach.scan successors initial
      ⦃ result => ScanPost successors.val initial.val result.2.val result.1 ⦄ := by
  unfold barrier_reach.scan barrier_reach.scan_loop
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec U32 × Bool × Usize => state.2.2.val)
    (fun state => ScanInvariant successors.val initial.val state.1.val state.2.1 state.2.2.val)
  · rintro ⟨reach, changed, pc⟩ invariant
    obtain ⟨sameLength, pcBound, growth, strict, stable⟩ := invariant
    dsimp only at sameLength pcBound growth strict stable
    unfold barrier_reach.scan_loop.body
    by_cases positive : 0 < pc.val
    · have test : pc > 0#usize := by scalar_tac
      simp only [test, ite_true]
      step with Usize.sub_spec (x := pc) (y := 1#usize) (by scalar_tac) as ⟨previous, previousValue⟩
      have previousBound : previous.val < reach.val.length := by scalar_tac
      have successorBound : previous.val < successors.val.length := by have := bounded.1; scalar_tac
      step with Slice.index_usize_spec successors previous successorBound as ⟨row, rowValue⟩
      have rowAt : row = successors.val[previous.val]! := by
        rw [rowValue, List.Inhabited_getElem_eq_getElem! _ _ successorBound]
      step with alloc.vec.Vec.index_usize_spec reach previous previousBound as ⟨current, currentValue⟩
      have currentAt : current = reach.val[previous.val]! := by
        rw [currentValue, List.Inhabited_getElem_eq_getElem! _ _ previousBound]
      have rowBound : ∀ destination ∈ row.val, destination.val < reach.val.length := by
        intro destination member
        rw [rowAt] at member
        rw [sameLength]
        exact bounded.2 previous.val (by scalar_tac) destination member
      step with rust_merge_successors (alloc.vec.Vec.deref row) (alloc.vec.Vec.deref reach) current
        (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using rowBound)
        as ⟨merged, mergedPost⟩
      obtain ⟨includesCurrent, includesSuccessors⟩ := mergedPost
      simp only [alloc.vec.Vec.deref, Slice.from_val] at includesSuccessors
      have includesAt : Includes reach.val[previous.val]! merged := by
        simpa only [currentAt] using includesCurrent
      split
      · rename_i changedValue
        have different : merged ≠ current := by simpa using changedValue
        step with alloc.vec.Vec.index_mut_usize_spec reach previous previousBound
          as ⟨old, restore, oldValue, restoreValue⟩
        simp only [restoreValue, alloc.vec.Vec.set_val_eq]
        constructor
        · refine ⟨by simpa only [List.length_set] using sameLength, by scalar_tac,
            growth.trans (update_grows reach.val previous.val merged previousBound includesAt), ?_, ?_⟩
          · intro _
            have beforeCard := facts_card_grow initial.val.length initial.val reach.val growth
            have changedCard := changed_update_strict initial.val.length reach.val previous.val merged
              previousBound (by scalar_tac) includesAt (by simpa only [currentAt] using different)
            omega
          · intro impossible
            cases impossible
        · scalar_tac
      · rename_i unchangedValue
        have equal : merged = current := UScalar.eq_of_val_eq (by simpa using unchangedValue)
        simp only [WP.spec_ok]
        constructor
        · refine ⟨sameLength, by scalar_tac, growth, strict, ?_⟩
          intro unchanged
          obtain ⟨unchangedRows, processed⟩ := stable unchanged
          refine ⟨unchangedRows, ?_⟩
          intro address addressLower addressBound destination member
          by_cases same : address = previous.val
          · subst address
            rw [←rowAt] at member
            have included := includesSuccessors destination member
            simpa only [equal, currentAt] using included
          · exact processed address (by scalar_tac) addressBound destination member
        · scalar_tac
    · have test : ¬pc > 0#usize := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      refine ⟨sameLength, growth, strict, ?_⟩
      intro unchanged address addressBound destination member
      exact (stable unchanged).2 address (by scalar_tac) addressBound destination member
  · change ScanInvariant successors.val initial.val initial.val false (alloc.vec.Vec.len initial).val
    unfold ScanInvariant
    refine ⟨rfl, by scalar_tac, Grows.refl initial.val, ?_, ?_⟩
    · intro impossible
      cases impossible
    · intro _
      refine ⟨rfl, ?_⟩
      intro pc lower inside destination member
      have initialLength : (alloc.vec.Vec.len initial).val = initial.val.length := by scalar_tac
      omega

end MemoryWrite.BarrierReach

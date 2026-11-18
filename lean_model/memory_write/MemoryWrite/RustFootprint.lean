/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.ExecutionFootprint

abbrev Range := U8 × U32 × U64 × U64

def space (range : Range) : Nat := range.1.val
def owner (range : Range) : Nat := range.2.1.val
def low (range : Range) : Nat := range.2.2.1.val
def high (range : Range) : Nat := range.2.2.2.val

def KeyBefore (left right : Range) : Prop :=
  space left < space right ∨ space left = space right ∧ owner left < owner right

def SameKey (left right : Range) : Prop :=
  space left = space right ∧ owner left = owner right

def Overlap (left right : Range) : Prop :=
  SameKey left right ∧ low right < high left ∧ low left < high right

def Ordered (left right : Range) : Prop :=
  KeyBefore left right ∨ SameKey left right ∧ low left ≤ low right

def SortedRanges (ranges : List Range) : Prop :=
  ∀ first second, first ≤ second → second < ranges.length → Ordered ranges[first]! ranges[second]!

def AnyOverlap (left right : List Range) : Prop :=
  ∃ first, first < left.length ∧ ∃ second, second < right.length ∧ Overlap left[first]! right[second]!

theorem overlap_symm (left right : Range) : Overlap left right ↔ Overlap right left := by
  simp only [Overlap, SameKey]
  omega

theorem same_key_of_not_before (left right : Range)
    (forward : ¬KeyBefore left right) (backward : ¬KeyBefore right left) : SameKey left right := by
  simp only [KeyBefore, SameKey] at *
  omega

theorem skip_left_is_disjoint (left right later : Range) (ordered : Ordered right later)
    (skip : KeyBefore left right ∨ SameKey left right ∧ high left ≤ low right) :
    ¬Overlap left later := by
  simp only [Ordered, KeyBefore, SameKey, Overlap] at *
  omega

structure ScanInvariant (left right : List Range) (first second : Nat) : Prop where
  first_bound : first ≤ left.length
  second_bound : second ≤ right.length
  remaining : ∀ a b, a < left.length → b < right.length → Overlap left[a]! right[b]! →
    first ≤ a ∧ second ≤ b

theorem advance_left (left right : List Range) (first second : Nat)
    (ordered : SortedRanges right) (invariant : ScanInvariant left right first second)
    (inside : first < left.length)
    (skip : KeyBefore left[first]! right[second]! ∨
      SameKey left[first]! right[second]! ∧ high left[first]! ≤ low right[second]!) :
    ScanInvariant left right (first + 1) second := by
  refine ⟨by omega, invariant.second_bound, ?_⟩
  intro a b aInside bInside overlap
  obtain ⟨firstBound, secondBound⟩ := invariant.remaining a b aInside bInside overlap
  by_cases same : a = first
  · have impossible := skip_left_is_disjoint left[first]! right[second]! right[b]!
      (ordered second b secondBound bInside) skip
    exact False.elim (impossible (by simpa only [same] using overlap))
  · exact ⟨by omega, secondBound⟩

theorem advance_right (left right : List Range) (first second : Nat)
    (ordered : SortedRanges left) (invariant : ScanInvariant left right first second)
    (otherInside : second < right.length)
    (skip : KeyBefore right[second]! left[first]! ∨
      SameKey right[second]! left[first]! ∧ high right[second]! ≤ low left[first]!) :
    ScanInvariant left right first (second + 1) := by
  refine ⟨invariant.first_bound, by omega, ?_⟩
  intro a b aInside bInside overlap
  obtain ⟨firstBound, secondBound⟩ := invariant.remaining a b aInside bInside overlap
  by_cases same : b = second
  · have impossible := skip_left_is_disjoint right[second]! left[first]! left[a]!
      (ordered first a firstBound aInside) skip
    exact False.elim (impossible ((overlap_symm _ _).mp (by simpa only [same] using overlap)))
  · exact ⟨firstBound, by omega⟩

@[step] theorem rust_shared_register_range (warp : U32) :
    footprint.shared_register_range warp ⦃ result => result = (17#u8, warp, 0#u64, 1#u64) ⦄ := by
  simp [footprint.shared_register_range]

@[step] theorem rust_key_before (left right : Range) :
    footprint.key_before left right ⦃ result => result = true ↔ KeyBefore left right ⦄ := by
  rcases left with ⟨s, a, lo, hi⟩
  rcases right with ⟨t, b, lowRight, highRight⟩
  by_cases first : s.val < t.val
  · simp [footprint.key_before, first, KeyBefore, space, owner]
  · by_cases same : s = t
    · simp [footprint.key_before, same, KeyBefore, space, owner]
    · have unequal : s.val ≠ t.val := by scalar_tac
      simp [footprint.key_before, first, same, unequal, KeyBefore, space, owner]

@[step] theorem rust_overlaps (left right : Slice Range)
    (leftSorted : SortedRanges left.val) (rightSorted : SortedRanges right.val) :
    footprint.overlaps left right ⦃ result => result = true ↔ AnyOverlap left.val right.val ⦄ := by
  unfold footprint.overlaps footprint.overlaps_loop
  apply loop.spec_decr_nat
    (fun cursor : Usize × Usize => left.val.length + right.val.length - cursor.1.val - cursor.2.val)
    (fun cursor : Usize × Usize => ScanInvariant left.val right.val cursor.1.val cursor.2.val)
  · rintro ⟨i,j⟩ invariant
    change ScanInvariant left.val right.val i.val j.val at invariant
    have firstBound := invariant.first_bound
    have secondBound := invariant.second_bound
    unfold footprint.overlaps_loop.body
    by_cases leftInside : i.val < left.val.length
    · have test : i < Slice.len left := by scalar_tac
      simp only [test, ite_true]
      by_cases rightInside : j.val < right.val.length
      · have test : j < Slice.len right := by scalar_tac
        simp only [test, ite_true]
        step with Slice.index_usize_spec left i leftInside as ⟨x,xEq⟩
        step with Slice.index_usize_spec right j rightInside as ⟨y,yEq⟩
        have xAt : x = left.val[i.val]! := by
          rw [xEq,List.Inhabited_getElem_eq_getElem! _ _ leftInside]
        have yAt : y = right.val[j.val]! := by
          rw [yEq,List.Inhabited_getElem_eq_getElem! _ _ rightInside]
        step with rust_key_before x y as ⟨forward,forwardSpec⟩
        split
        · rename_i forwardTrue
          have skip : KeyBefore left.val[i.val]! right.val[j.val]! := by
            simpa only [xAt,yAt] using forwardSpec.mp forwardTrue
          step as ⟨next,nextEq⟩
          refine ⟨?_,by scalar_tac⟩
          simpa only [nextEq] using advance_left left.val right.val i.val j.val rightSorted invariant
            leftInside (Or.inl skip)
        · rename_i forwardFalse
          have notForward : ¬KeyBefore x y := by
            intro key; exact forwardFalse (forwardSpec.mpr key)
          step with rust_key_before y x as ⟨backward,backwardSpec⟩
          split
          · rename_i backwardTrue
            have skip : KeyBefore right.val[j.val]! left.val[i.val]! := by
              simpa only [xAt,yAt] using backwardSpec.mp backwardTrue
            step as ⟨next,nextEq⟩
            refine ⟨?_,by scalar_tac⟩
            simpa only [nextEq] using advance_right left.val right.val i.val j.val leftSorted invariant
              rightInside (Or.inl skip)
          · rename_i backwardFalse
            have notBackward : ¬KeyBefore y x := by
              intro key; exact backwardFalse (backwardSpec.mpr key)
            have same : SameKey x y := same_key_of_not_before x y notForward notBackward
            rcases x with ⟨xs,xo,xl,xh⟩
            rcases y with ⟨ys,yo,yl,yh⟩
            let x : Range := (xs,xo,xl,xh)
            let y : Range := (ys,yo,yl,yh)
            change (if xh ≤ yl then _ else _) ⦃ _ ⦄
            split
            · rename_i disjoint
              have skip : SameKey left.val[i.val]! right.val[j.val]! ∧
                  high left.val[i.val]! ≤ low right.val[j.val]! := by
                simpa only [←xAt,←yAt] using And.intro same
                  (show high x ≤ low y by dsimp only [high,low]; scalar_tac)
              step as ⟨next,nextEq⟩
              refine ⟨?_,by scalar_tac⟩
              simpa only [nextEq] using advance_left left.val right.val i.val j.val rightSorted invariant
                leftInside (Or.inr skip)
            · rename_i notDisjoint
              split
              · rename_i disjoint
                have sameReverse : SameKey y x := ⟨same.1.symm, same.2.symm⟩
                have skip : SameKey right.val[j.val]! left.val[i.val]! ∧
                    high right.val[j.val]! ≤ low left.val[i.val]! := by
                  simpa only [←xAt,←yAt] using And.intro sameReverse
                    (show high y ≤ low x by dsimp only [high,low]; scalar_tac)
                step as ⟨next,nextEq⟩
                refine ⟨?_,by scalar_tac⟩
                simpa only [nextEq] using advance_right left.val right.val i.val j.val leftSorted invariant
                  rightInside (Or.inr skip)
              · rename_i notDisjointReverse
                simp only [WP.spec_ok,eq_self,true_iff]
                refine ⟨i.val,leftInside,j.val,rightInside,?_⟩
                rw [←xAt,←yAt]
                refine ⟨same,?_,?_⟩ <;> dsimp only [low,high] <;> scalar_tac
      · have test : ¬ j < Slice.len right := by scalar_tac
        simp only [test,ite_false,WP.spec_ok,Bool.false_eq_true,false_iff]
        rintro ⟨a,aInside,b,bInside,overlap⟩
        have remaining := invariant.remaining a b aInside bInside overlap
        omega
    · have test : ¬ i < Slice.len left := by scalar_tac
      simp only [test,ite_false,WP.spec_ok,Bool.false_eq_true,false_iff]
      rintro ⟨a,aInside,b,bInside,overlap⟩
      have remaining := invariant.remaining a b aInside bInside overlap
      omega
  · refine ⟨by scalar_tac,by scalar_tac,?_⟩
    intro a b aInside bInside overlap
    dsimp only
    scalar_tac

def SharedRange (warp : U32) : Range := (17#u8, warp, 0#u64, 1#u64)

theorem shared_range_overlap (first second : U32) :
    Overlap (SharedRange first) (SharedRange second) ↔ first = second := by
  simp only [Overlap, SameKey, SharedRange, space, owner, low, high]
  scalar_tac

def Conflicts (leftAll : Bool) (leftReads leftWrites : List Range)
    (rightAll : Bool) (rightReads rightWrites : List Range) : Prop :=
  leftAll = true ∨ rightAll = true ∨ AnyOverlap leftWrites rightWrites ∨
    AnyOverlap leftWrites rightReads ∨ AnyOverlap rightWrites leftReads

@[step] theorem rust_conflicts (leftAll rightAll : Bool)
    (leftReads leftWrites rightReads rightWrites : Slice Range)
    (leftReadSorted : SortedRanges leftReads.val)
    (leftWriteSorted : SortedRanges leftWrites.val)
    (rightReadSorted : SortedRanges rightReads.val)
    (rightWriteSorted : SortedRanges rightWrites.val) :
    footprint.conflicts leftAll leftReads leftWrites rightAll rightReads rightWrites
      ⦃ result => result = true ↔
        Conflicts leftAll leftReads.val leftWrites.val rightAll rightReads.val rightWrites.val ⦄ := by
  unfold footprint.conflicts
  cases leftAll <;> cases rightAll <;> simp only [Bool.false_eq_true] <;>
    simp only [ite_true, ite_false, WP.spec_ok, Conflicts, Bool.false_eq_true,
      false_or, true_or, iff_true]
  step with rust_overlaps leftWrites rightWrites leftWriteSorted rightWriteSorted
    as ⟨bothWrites,bothWritesSpec⟩
  split
  · rename_i found
    simp only [WP.spec_ok]
    exact Iff.intro (by intro _; exact Or.inl (bothWritesSpec.mp found)) (by intro _; trivial)
  · rename_i absent
    have noWriteWrite : ¬AnyOverlap leftWrites.val rightWrites.val := by
      intro overlap; exact absent (bothWritesSpec.mpr overlap)
    step with rust_overlaps leftWrites rightReads leftWriteSorted rightReadSorted
      as ⟨forward,forwardSpec⟩
    split
    · rename_i found
      simp only [WP.spec_ok]
      exact Iff.intro (by intro _; exact Or.inr (Or.inl (forwardSpec.mp found))) (by intro _; trivial)
    · rename_i absent
      have noWriteRead : ¬AnyOverlap leftWrites.val rightReads.val := by
        intro overlap; exact absent (forwardSpec.mpr overlap)
      step with rust_overlaps rightWrites leftReads rightWriteSorted leftReadSorted
        as ⟨backward,backwardSpec⟩
      simpa only [noWriteWrite, noWriteRead, false_or] using backwardSpec

theorem shared_mutation_is_a_dependency (leftReads leftWrites rightReads rightWrites : List Range)
    (warp : U32) (writeIndex readIndex : Nat)
    (writeInside : writeIndex < leftWrites.length) (readInside : readIndex < rightReads.length)
    (write : leftWrites[writeIndex]! = SharedRange warp)
    (read : rightReads[readIndex]! = SharedRange warp) :
    Conflicts false leftReads leftWrites false rightReads rightWrites := by
  refine Or.inr (Or.inr (Or.inr (Or.inl ?_)))
  refine ⟨writeIndex,writeInside,readIndex,readInside,?_⟩
  rw [write,read]
  exact (shared_range_overlap warp warp).mpr rfl

@[step] theorem source_shared_mutation_is_a_dependency
    (leftReads leftWrites rightReads rightWrites : Slice Range)
    (leftReadSorted : SortedRanges leftReads.val)
    (leftWriteSorted : SortedRanges leftWrites.val)
    (rightReadSorted : SortedRanges rightReads.val)
    (rightWriteSorted : SortedRanges rightWrites.val)
    (warp : U32) (writeIndex readIndex : Nat)
    (writeInside : writeIndex < leftWrites.val.length)
    (readInside : readIndex < rightReads.val.length)
    (write : leftWrites.val[writeIndex]! = SharedRange warp)
    (read : rightReads.val[readIndex]! = SharedRange warp) :
    footprint.conflicts false leftReads leftWrites false rightReads rightWrites
      ⦃ result => result = true ⦄ := by
  step with rust_conflicts false false leftReads leftWrites rightReads rightWrites
    leftReadSorted leftWriteSorted rightReadSorted rightWriteSorted as ⟨result,resultSpec⟩
  exact resultSpec.mpr (shared_mutation_is_a_dependency leftReads.val leftWrites.val
    rightReads.val rightWrites.val warp writeIndex readIndex writeInside readInside write read)

end MemoryWrite.ExecutionFootprint

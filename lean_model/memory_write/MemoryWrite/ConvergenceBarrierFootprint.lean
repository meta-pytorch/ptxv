/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite.ConvergenceBarrierFootprint

abbrev Range := U8 × U32 × U64 × U64

def Spec (warp : U32) (barrier : U8) (range : Range) : Prop :=
  range.1 = 19#u8 ∧ range.2.1 = warp ∧
  range.2.2.1.val = barrier.val ∧ range.2.2.2.val = barrier.val + 1

def Overlap (left right : Range) : Prop :=
  left.1 = right.1 ∧ left.2.1 = right.2.1 ∧
  left.2.2.1.val < right.2.2.2.val ∧ right.2.2.1.val < left.2.2.2.val

@[step] theorem rust_convergence_barrier_range (warp : U32) (barrier : U8) :
    footprint.convergence_barrier_range warp barrier ⦃ result => Spec warp barrier result ⦄ := by
  unfold footprint.convergence_barrier_range
  simp only [lift, bind_ok]
  step
  simp only [Spec]
  scalar_tac

theorem covers_exactly (warp : U32) (barrier : U8) (range : Range)
    (source : Spec warp barrier range) (position : Nat) :
    (range.2.2.1.val ≤ position ∧ position < range.2.2.2.val) ↔ position = barrier.val := by
  rcases source with ⟨_, _, low, high⟩
  omega

theorem overlap_exactly (warp otherWarp : U32) (barrier otherBarrier : U8)
    (left right : Range) (leftSource : Spec warp barrier left)
    (rightSource : Spec otherWarp otherBarrier right) :
    Overlap left right ↔ warp = otherWarp ∧ barrier = otherBarrier := by
  rcases leftSource with ⟨leftSpace, leftOwner, leftLow, leftHigh⟩
  rcases rightSource with ⟨rightSpace, rightOwner, rightLow, rightHigh⟩
  simp only [Overlap, leftSpace, rightSpace, leftOwner, rightOwner, true_and,
    leftLow, leftHigh, rightLow, rightHigh]
  constructor
  · rintro ⟨owner, low, high⟩
    exact ⟨owner, by scalar_tac⟩
  · rintro ⟨rfl, rfl⟩
    exact ⟨rfl, by omega, by omega⟩

theorem other_namespace_disjoint (warp : U32) (barrier : U8) (left right : Range)
    (source : Spec warp barrier left) (otherSpace : right.1 ≠ 19#u8) :
    ¬Overlap left right := by
  intro overlap
  exact otherSpace (overlap.1.symm.trans source.1)

end MemoryWrite.ConvergenceBarrierFootprint

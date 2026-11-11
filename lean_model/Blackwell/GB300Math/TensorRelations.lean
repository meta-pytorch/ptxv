/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorGroupError

namespace GB300Math.TensorRelations
open GB300Math.Tensor GB300Math.TensorGroupError

def groupedResult (groups : List (List Int)) (s t : Nat) : Int :=
  shiftTruncate (((groups.map List.sum).map (fun x => shiftTruncate x (-(s:Int)))).sum)
    (-(t:Int)) * (2^(s+t):Int)

def errorBudget (groups : List (List Int)) (s t : Nat) : Nat :=
  groups.length*(2^s-1) + (2^t-1)*2^s

theorem grouped_result_bound (groups : List (List Int)) (s t : Nat) :
    (groups.flatten.sum - groupedResult groups s t).natAbs ≤ errorBudget groups s t := by
  exact grouped_two_stage_residual groups s t

theorem different_groupings_bound (left right : List (List Int))
    (s₁ t₁ s₂ t₂ : Nat) (same_sum : left.flatten.sum = right.flatten.sum) :
    (groupedResult left s₁ t₁ - groupedResult right s₂ t₂).natAbs ≤
      errorBudget left s₁ t₁ + errorBudget right s₂ t₂ := by
  have hl := grouped_result_bound left s₁ t₁
  have hr := grouped_result_bound right s₂ t₂
  have triangle := Int.natAbs_add_le
    (-(left.flatten.sum - groupedResult left s₁ t₁))
    (right.flatten.sum - groupedResult right s₂ t₂)
  rw [Int.natAbs_neg] at triangle
  have equality : groupedResult left s₁ t₁ - groupedResult right s₂ t₂ =
      -(left.flatten.sum - groupedResult left s₁ t₁) +
        (right.flatten.sum - groupedResult right s₂ t₂) := by omega
  rw [equality]
  omega

theorem k96_alignment_budget (groups : List (List Int)) (s : Nat)
    (at_most_seven : groups.length ≤ 7) :
    (groups.flatten.sum - groupedResult groups s 0).natAbs ≤ 7*(2^s-1) := by
  have bound := grouped_result_bound groups s 0
  simp only [errorBudget, Nat.pow_zero, Nat.sub_self, Nat.zero_mul, Nat.add_zero] at bound
  exact Nat.le_trans bound (Nat.mul_le_mul_right (2^s-1) at_most_seven)

theorem positive_result_above_budget (groups : List (List Int)) (s t : Nat)
    (positive_margin : (errorBudget groups s t : Int) < groups.flatten.sum) :
    0 < groupedResult groups s t := by
  have bound := grouped_result_bound groups s t
  have signed_bound : groups.flatten.sum - groupedResult groups s t ≤
      ((groups.flatten.sum - groupedResult groups s t).natAbs : Int) := Int.le_natAbs
  omega

theorem negative_result_below_budget (groups : List (List Int)) (s t : Nat)
    (negative_margin : groups.flatten.sum < -(errorBudget groups s t : Int)) :
    groupedResult groups s t < 0 := by
  have bound := grouped_result_bound groups s t
  have signed_bound : -(groups.flatten.sum - groupedResult groups s t) ≤
      ((-(groups.flatten.sum - groupedResult groups s t)).natAbs : Int) := Int.le_natAbs
  rw [Int.natAbs_neg] at signed_bound
  omega

#print axioms grouped_result_bound
#print axioms different_groupings_bound
#print axioms k96_alignment_budget
#print axioms positive_result_above_budget
#print axioms negative_result_below_budget
end GB300Math.TensorRelations

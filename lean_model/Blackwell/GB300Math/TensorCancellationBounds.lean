/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorAlignmentLoss

namespace GB300Math.TensorCancellationBounds
open GB300Math.TensorAlignmentLoss GB300Math.TensorAlignmentProperties

def positiveBudget (xs : List Int) (step : Nat) : Int :=
  xs.foldr (fun x total => (if 0 < x then ((step-1 : Nat) : Int) else 0)+total) 0

def negativeBudget (xs : List Int) (step : Nat) : Int :=
  xs.foldr (fun x total => (if x < 0 then ((step-1 : Nat) : Int) else 0)+total) 0

theorem signed_loss (n : Int) (step : Nat) (positive : 0 < step) :
    -(if n < 0 then ((step-1 : Nat) : Int) else 0) ≤ n-truncateGrid n step ∧
    n-truncateGrid n step ≤ (if 0 < n then ((step-1 : Nat) : Int) else 0) := by
  cases n with
  | ofNat n =>
    have h := unsigned_alignment_loss n step positive
    have hl := Int.ofNat_le.mpr h.1
    have hu := Int.ofNat_lt.mpr h.2
    simp at hl hu
    have hn : ¬ (n : Int) < 0 := by omega
    by_cases hp : 0 < n
    · simp [truncateGrid, hn, hp]
      omega
    · have hz : n = 0 := by omega
      subst n
      simp [truncateGrid]
  | negSucc n =>
    have h := unsigned_alignment_loss (n+1) step positive
    have hl := Int.ofNat_le.mpr h.1
    have hu := Int.ofNat_lt.mpr h.2
    simp at hl hu
    simp [truncateGrid]
    omega

theorem cancellation_aware_loss (xs : List Int) (step : Nat) (positive : 0 < step) :
    -negativeBudget xs step ≤ exactSum xs-truncatedSum xs step ∧
    exactSum xs-truncatedSum xs step ≤ positiveBudget xs step := by
  induction xs with
  | nil => simp [negativeBudget, positiveBudget, exactSum, truncatedSum]
  | cons x xs ih =>
    have hx := signed_loss x step positive
    simp only [negativeBudget, positiveBudget, exactSum, truncatedSum, List.foldr_cons] at *
    omega

theorem truncate_neg (n : Int) (step : Nat) :
    truncateGrid (-n) step = -truncateGrid n step := by
  by_cases hz : n = 0
  · subst n
    simp [truncateGrid]
  by_cases h : n < 0
  · have hn : ¬ -n < 0 := by omega
    simp only [truncateGrid, Int.natAbs_neg, h, hn, ite_true, ite_false, Int.neg_neg]
  · have hn : -n < 0 := by omega
    simp only [truncateGrid, Int.natAbs_neg, h, hn, ite_true, ite_false]

theorem fixed_lattice_pair_cancels (n : Int) (xs : List Int) (step : Nat) :
    truncatedSum (n :: -n :: xs) step = truncatedSum xs step := by
  simp only [truncatedSum, List.foldr_cons, truncate_neg]
  omega

theorem positive_ideal_retains_sign (xs : List Int) (step : Nat) (positive : 0 < step)
    (large : positiveBudget xs step < exactSum xs) : 0 < truncatedSum xs step := by
  have h := cancellation_aware_loss xs step positive
  omega

theorem negative_ideal_retains_sign (xs : List Int) (step : Nat) (positive : 0 < step)
    (large : exactSum xs < -negativeBudget xs step) : truncatedSum xs step < 0 := by
  have h := cancellation_aware_loss xs step positive
  omega

theorem positive_budget_attained :
    exactSum [7,7,-8]-truncatedSum [7,7,-8] 8 = positiveBudget [7,7,-8] 8 := by
  decide +kernel

theorem negative_budget_attained :
    exactSum [8,-7,-7]-truncatedSum [8,-7,-7] 8 = -negativeBudget [8,-7,-7] 8 := by
  decide +kernel

theorem distinct_lattices_can_break_cancellation :
    truncateGrid 7 8 + truncateGrid (-7) 4 = -4 := by decide +kernel

#print axioms signed_loss
#print axioms cancellation_aware_loss
#print axioms truncate_neg
#print axioms fixed_lattice_pair_cancels
#print axioms positive_ideal_retains_sign
#print axioms negative_ideal_retains_sign
#print axioms positive_budget_attained
#print axioms negative_budget_attained
#print axioms distinct_lattices_can_break_cancellation
end GB300Math.TensorCancellationBounds

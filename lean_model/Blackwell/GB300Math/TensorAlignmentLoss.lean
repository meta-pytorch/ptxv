/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorAlignmentProperties
namespace GB300Math.TensorAlignmentLoss
open GB300Math.TensorAlignmentProperties

def truncateGrid (n : Int) (step : Nat) : Int :=
  let magnitude := n.natAbs / step * step
  if n < 0 then -(magnitude : Int) else magnitude

theorem truncation_loss (n : Int) (step : Nat) (positive : 0 < step) :
    -((step-1 : Nat) : Int) ≤ n-truncateGrid n step ∧
    n-truncateGrid n step ≤ ((step-1 : Nat) : Int) := by
  cases n with
  | ofNat n =>
    have h := unsigned_alignment_loss n step positive
    have hl := Int.ofNat_le.mpr h.1
    have hu := Int.ofNat_lt.mpr h.2
    simp at hl hu
    simp [truncateGrid]
    omega
  | negSucc n =>
    have h := unsigned_alignment_loss (n+1) step positive
    have hl := Int.ofNat_le.mpr h.1
    have hu := Int.ofNat_lt.mpr h.2
    simp at hl hu
    simp [truncateGrid]
    omega

def exactSum (xs : List Int) : Int := xs.foldr (· + ·) 0

def truncatedSum (xs : List Int) (step : Nat) : Int :=
  xs.foldr (fun x total => truncateGrid x step + total) 0

theorem sum_truncation_loss (xs : List Int) (step : Nat) (positive : 0 < step) :
    -((xs.length*(step-1) : Nat) : Int) ≤ exactSum xs-truncatedSum xs step ∧
    exactSum xs-truncatedSum xs step ≤ ((xs.length*(step-1) : Nat) : Int) := by
  induction xs with
  | nil => simp [exactSum, truncatedSum]
  | cons x xs ih =>
    have hx := truncation_loss x step positive
    simp only [exactSum, truncatedSum, List.foldr_cons, List.length_cons, Nat.add_mul,
      Nat.one_mul, Int.natCast_add] at *
    omega

#print axioms truncation_loss
#print axioms sum_truncation_loss

theorem shiftTruncate_grid (n : Int) (k : Nat) :
    GB300Math.Tensor.shiftTruncate n (-(k : Int)) * (2^k : Nat) = truncateGrid n (2^k) := by
  cases k with
  | zero => simp [GB300Math.Tensor.shiftTruncate, truncateGrid]
  | succ k =>
    simp [GB300Math.Tensor.shiftTruncate, truncateGrid]
    have h : ¬ ((k : Int) + 1 ≤ 0) := by omega
    split <;> simp_all [Int.neg_mul]

#print axioms shiftTruncate_grid

theorem shiftTruncate_loss (n : Int) (k : Nat) :
    -((2^k-1 : Nat) : Int) ≤
      n-GB300Math.Tensor.shiftTruncate n (-(k : Int)) * (2^k : Nat) ∧
    n-GB300Math.Tensor.shiftTruncate n (-(k : Int)) * (2^k : Nat) ≤
      ((2^k-1 : Nat) : Int) := by
  rw [shiftTruncate_grid]
  exact truncation_loss n (2^k) (Nat.two_pow_pos k)

def shiftedSum (xs : List Int) (k : Nat) : Int :=
  xs.foldr (fun n total =>
    GB300Math.Tensor.shiftTruncate n (-(k : Int)) * (2^k : Nat) + total) 0

theorem shiftedSum_eq (xs : List Int) (k : Nat) :
    shiftedSum xs k = truncatedSum xs (2^k) := by
  simp only [shiftedSum, truncatedSum, shiftTruncate_grid]

theorem shiftedSum_loss (xs : List Int) (k : Nat) :
    -((xs.length*(2^k-1) : Nat) : Int) ≤ exactSum xs-shiftedSum xs k ∧
    exactSum xs-shiftedSum xs k ≤ ((xs.length*(2^k-1) : Nat) : Int) := by
  rw [shiftedSum_eq]
  exact sum_truncation_loss xs (2^k) (Nat.two_pow_pos k)

#print axioms shiftTruncate_loss
#print axioms shiftedSum_eq
#print axioms shiftedSum_loss

end GB300Math.TensorAlignmentLoss

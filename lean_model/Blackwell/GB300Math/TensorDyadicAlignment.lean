/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorAlignmentLoss

namespace GB300Math.TensorDyadicAlignment
open GB300Math.Tensor GB300Math.TensorAlignmentLoss

theorem quotient_scale (n a b : Nat) :
    n*2^a/2^b = if a ≥ b then n*2^(a-b) else n/2^(b-a) := by
  split
  next h =>
    have e : a = (a-b)+b := by omega
    conv => lhs; rw [e, Nat.pow_add, ← Nat.mul_assoc]
    exact Nat.mul_div_cancel _ (Nat.two_pow_pos _)
  next h =>
    have e : b = (b-a)+a := by omega
    conv => lhs; rw [e, Nat.pow_add]
    exact Nat.mul_div_mul_right _ _ (Nat.two_pow_pos _)

theorem rescale_truncate (n : Int) (a b : Nat) :
    shiftTruncate n ((a : Int)-(b : Int)) * (2^b : Nat) =
      truncateGrid (n*(2^a : Nat)) (2^b) := by
  have hp : (0 : Int) < (2^a : Nat) := by exact_mod_cast Nat.two_pow_pos a
  have hn : n*(2^a : Nat) < 0 ↔ n < 0 := by
    constructor
    · intro h
      by_cases h' : n < 0
      · exact h'
      · have hh := Int.mul_nonneg (by omega : 0 ≤ n) (by omega : (0 : Int) ≤ (2^a : Nat))
        omega
    · intro h
      exact Int.mul_neg_of_neg_of_pos h hp
  have hm : (if (a : Int)-(b : Int) ≥ 0 then n.natAbs * 2^((a : Int)-(b : Int)).toNat
      else n.natAbs / 2^(-((a : Int)-(b : Int))).toNat) = n.natAbs*2^a/2^b := by
    rw [quotient_scale]
    by_cases h : a ≥ b
    · have he : (a : Int)-(b : Int) ≥ 0 := by omega
      have e : ((a : Int)-(b : Int)).toNat = a-b := by omega
      simp only [h, he, e, ite_true]
    · have he : ¬ ((a : Int)-(b : Int) ≥ 0) := by omega
      have e : (-((a : Int)-(b : Int))).toNat = b-a := by omega
      simp only [h, he, e, ite_false]
  simp only [shiftTruncate, truncateGrid, hm, Int.natAbs_mul, Int.natAbs_natCast, hn]
  split <;> simp_all [Int.natCast_mul, Int.neg_mul]

abbrev ScaledTerm := Int × Nat

def exactValue (xs : List ScaledTerm) : Int :=
  xs.foldr (fun x s => x.1*(2^x.2 : Nat)+s) 0

def alignedValue (xs : List ScaledTerm) (scale : Nat) : Int :=
  xs.foldr (fun x s => shiftTruncate x.1 ((x.2 : Int)-(scale : Int))*(2^scale : Nat)+s) 0

def errorBudget (xs : List ScaledTerm) (scale : Nat) : Int :=
  (xs.length*(2^scale-1) : Nat)

theorem alignment_error (xs : List ScaledTerm) (scale : Nat) :
    -errorBudget xs scale ≤ exactValue xs-alignedValue xs scale ∧
    exactValue xs-alignedValue xs scale ≤ errorBudget xs scale := by
  induction xs with
  | nil => simp [exactValue, alignedValue, errorBudget]
  | cons x xs ih =>
    have hx := truncation_loss (x.1*(2^x.2 : Nat)) (2^scale) (Nat.two_pow_pos scale)
    rw [← rescale_truncate] at hx
    simp only [exactValue, alignedValue, errorBudget, List.foldr_cons, List.length_cons,
      Nat.add_mul, Nat.one_mul, Int.natCast_add] at *
    omega

theorem positive_ideal_preserves_sign (xs : List ScaledTerm) (scale : Nat)
    (large : errorBudget xs scale < exactValue xs) : 0 < alignedValue xs scale := by
  have h := alignment_error xs scale
  omega

theorem negative_ideal_preserves_sign (xs : List ScaledTerm) (scale : Nat)
    (large : exactValue xs < -errorBudget xs scale) : alignedValue xs scale < 0 := by
  have h := alignment_error xs scale
  omega

theorem lattice_disagreement_bound (xs : List ScaledTerm) (a b : Nat) :
    -(errorBudget xs a+errorBudget xs b) ≤ alignedValue xs a-alignedValue xs b ∧
    alignedValue xs a-alignedValue xs b ≤ errorBudget xs a+errorBudget xs b := by
  have ha := alignment_error xs a
  have hb := alignment_error xs b
  omega

def commonBaseTerms (xs : List GB300Math.Tensor.Dyadic) (base : Int) : List ScaledTerm :=
  xs.map fun x => (x.significand, (x.exponent-base).toNat)

def actualAlignedGrid (xs : List GB300Math.Tensor.Dyadic) (base : Int) (scale : Nat) : Int :=
  xs.foldr (fun x total =>
    shiftTruncate x.significand (x.exponent-(base+(scale : Int)))*(2^scale : Nat)+total) 0

theorem common_base_bridge (xs : List GB300Math.Tensor.Dyadic) (base : Int) (scale : Nat)
    (lower : ∀ x ∈ xs, base ≤ x.exponent) :
    alignedValue (commonBaseTerms xs base) scale = actualAlignedGrid xs base scale := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    have hx := lower x (by simp)
    have tail : ∀ y ∈ xs, base ≤ y.exponent := by
      intro y hy
      exact lower y (by simp [hy])
    have he : (((x.exponent-base).toNat : Nat) : Int)-(scale : Int) =
        x.exponent-(base+(scale : Int)) := by omega
    simp only [commonBaseTerms, List.map_cons, alignedValue, actualAlignedGrid,
      List.foldr_cons, he]
    have hh := ih tail
    exact congrArg (fun total =>
      shiftTruncate x.significand (x.exponent-(base+(scale : Int)))*(2^scale : Nat)+total) hh

theorem dyadic_alignment_error (xs : List GB300Math.Tensor.Dyadic) (base : Int) (scale : Nat)
    (lower : ∀ x ∈ xs, base ≤ x.exponent) :
    -((xs.length*(2^scale-1) : Nat) : Int) ≤
      exactValue (commonBaseTerms xs base)-actualAlignedGrid xs base scale ∧
    exactValue (commonBaseTerms xs base)-actualAlignedGrid xs base scale ≤
      ((xs.length*(2^scale-1) : Nat) : Int) := by
  have h := alignment_error (commonBaseTerms xs base) scale
  rw [common_base_bridge xs base scale lower] at h
  simpa only [errorBudget, commonBaseTerms, List.length_map] using h

theorem budget_is_attained :
    exactValue [(7,0),(7,0)]-alignedValue [(7,0),(7,0)] 3 =
      errorBudget [(7,0),(7,0)] 3 := by decide +kernel

theorem small_positive_ideal_can_vanish :
    exactValue [(1,0)] = 1 ∧ alignedValue [(1,0)] 1 = 0 := by decide +kernel

#print axioms quotient_scale
#print axioms rescale_truncate
#print axioms alignment_error
#print axioms positive_ideal_preserves_sign
#print axioms negative_ideal_preserves_sign
#print axioms lattice_disagreement_bound

#print axioms common_base_bridge
#print axioms dyadic_alignment_error

#print axioms budget_is_attained
#print axioms small_positive_ideal_can_vanish

end GB300Math.TensorDyadicAlignment

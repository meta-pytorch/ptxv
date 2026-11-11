/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorSubnormalBounds
namespace GB300Math.TensorArrayEncodingBounds
open GB300Math.Tensor GB300Math.TensorAlignmentProperties
open GB300Math.TensorEncodingBounds GB300Math.TensorDyadicAlignment

def totalAt (xs : List Term) (scale : Int) : Int :=
  xs.foldr (fun x total => shiftTruncate x.1.significand (x.1.exponent-scale)+total) 0

theorem fold_sum (xs : List Term) (f : Term → Int) (acc : Int) :
    xs.foldl (fun s x => s+f x) acc = acc+xs.foldr (fun x s => f x+s) 0 := by
  induction xs generalizing acc with
  | nil => simp
  | cons x xs ih => simp only [List.foldl_cons,List.foldr_cons,ih]; omega

theorem aligned_total_eq (xs : List Term) :
    alignedTotal xs = totalAt xs (selectedScale xs) := by
  unfold alignedTotal totalAt
  rw [fold_sum]
  omega

theorem grid_total_eq (xs : List Term) (base : Int) (scale : Nat) :
    actualAlignedGrid (xs.map Prod.fst) base scale =
      totalAt xs (base+(scale : Int))*(2^scale : Nat) := by
  induction xs with
  | nil => simp [actualAlignedGrid,totalAt]
  | cons x xs ih =>
    simp only [actualAlignedGrid,totalAt,List.map_cons,List.foldr_cons] at *
    rw [ih,Int.add_mul]

def idealUnits (xs : List Term) (base : Int) : Int :=
  exactValue (commonBaseTerms (xs.map Prod.fst) base)

theorem actual_alignment_error (xs : List Term) (base : Int)
    (lower : ∀ x ∈ xs, base ≤ x.1.exponent)
    (scaleLower : base ≤ selectedScale xs) :
    let k := (selectedScale xs-base).toNat;
    let loss : Int := (xs.length*(2^k-1) : Nat);
    -loss ≤ idealUnits xs base-alignedTotal xs*(2^k : Nat) ∧
      idealUnits xs base-alignedTotal xs*(2^k : Nat) ≤ loss := by
  have lower' : ∀ x ∈ xs.map Prod.fst, base ≤ x.exponent := by
    intro x hx
    obtain ⟨y,hy,rfl⟩ := List.mem_map.mp hx
    exact lower y hy
  have hs : base+(((selectedScale xs-base).toNat : Nat) : Int)=selectedScale xs := by omega
  have h := dyadic_alignment_error (xs.map Prod.fst) base (selectedScale xs-base).toNat lower'
  rw [grid_total_eq,hs,← aligned_total_eq] at h
  simpa only [idealUnits,List.length_map] using h

def normalUnits (xs : List Term) (base : Int) : Int :=
  encodedUnits (alignedTotal xs) (selectedScale xs)*(2^(selectedScale xs-base).toNat : Nat)

def normalBudget (xs : List Term) (base : Int) : Int :=
  ((xs.length*(2^(selectedScale xs-base).toNat-1) : Nat) : Int)*expansion (alignedTotal xs) +
    roundingBudget (alignedTotal xs)*(2^(selectedScale xs-base).toNat : Nat)

theorem actual_normal_error (xs : List Term) (base : Int)
    (lower : ∀ x ∈ xs, base ≤ x.1.exponent)
    (scaleLower : base ≤ selectedScale xs)
    (nonzero : alignedTotal xs ≠ 0)
    (range : -126 ≤ selectedScale xs+((alignedTotal xs).natAbs.log2 : Int) ∧
      selectedScale xs+((alignedTotal xs).natAbs.log2 : Int) ≤ 127) :
    -normalBudget xs base ≤ idealUnits xs base*expansion (alignedTotal xs)-normalUnits xs base ∧
      idealUnits xs base*expansion (alignedTotal xs)-normalUnits xs base ≤ normalBudget xs base := by
  have ha := actual_alignment_error xs base lower scaleLower
  have hr := normal_encoder_error (alignedTotal xs) (selectedScale xs) nonzero range
  have hp : 0 ≤ expansion (alignedTotal xs) := Int.ofNat_le.mpr (Nat.zero_le _)
  have hq : 0 ≤ ((2^(selectedScale xs-base).toNat : Nat) : Int) := Int.ofNat_le.mpr (Nat.zero_le _)
  have al := Int.mul_le_mul_of_nonneg_right ha.1 hp
  have ah := Int.mul_le_mul_of_nonneg_right ha.2 hp
  have rl := Int.mul_le_mul_of_nonneg_right hr.1 hq
  have rh := Int.mul_le_mul_of_nonneg_right hr.2 hq
  change -roundingBudget (alignedTotal xs)*(2^(selectedScale xs-base).toNat : Nat) ≤
    (alignedTotal xs*expansion (alignedTotal xs)-encodedUnits (alignedTotal xs) (selectedScale xs))*(2^(selectedScale xs-base).toNat : Nat) at rl
  change (alignedTotal xs*expansion (alignedTotal xs)-encodedUnits (alignedTotal xs) (selectedScale xs))*(2^(selectedScale xs-base).toNat : Nat) ≤
    roundingBudget (alignedTotal xs)*(2^(selectedScale xs-base).toNat : Nat) at rh
  have he : alignedTotal xs*(2^(selectedScale xs-base).toNat : Nat)*expansion (alignedTotal xs) =
    alignedTotal xs*expansion (alignedTotal xs)*(2^(selectedScale xs-base).toNat : Nat) := by
    rw [Int.mul_assoc,Int.mul_comm (2^(selectedScale xs-base).toNat : Nat),← Int.mul_assoc]
  simp only [Int.sub_mul,Int.neg_mul,he] at al ah rl rh
  unfold normalBudget normalUnits
  omega

def decodedUnits (word : Nat) (base : Int) : Int :=
  let value := decode word 23 8
  value.significand*(2^(value.exponent-base).toNat : Nat)

theorem normal_word_units (xs : List Term) (base : Int)
    (scaleLower : base ≤ selectedScale xs)
    (nonzero : alignedTotal xs ≠ 0)
    (range : -126 ≤ selectedScale xs+((alignedTotal xs).natAbs.log2 : Int) ∧
      selectedScale xs+((alignedTotal xs).natAbs.log2 : Int) ≤ 127) :
    decodedUnits (roundedTotal xs) (base-((23-((alignedTotal xs).natAbs.log2 : Int)).toNat : Nat)) =
      normalUnits xs base := by
  let t := alignedTotal xs
  let e := selectedScale xs
  have he : (e+(t.natAbs.log2 : Int)-23-(base-((23-(t.natAbs.log2 : Int)).toNat : Nat))).toNat =
      ((t.natAbs.log2 : Int)-23).toNat+(e-base).toNat := by
    dsimp [e,t] at *
    omega
  unfold decodedUnits roundedTotal normalUnits encodedUnits
  rw [decode_encode_normal _ _ nonzero range]
  dsimp only
  rw [he,Nat.pow_add,Int.natCast_mul,← Int.mul_assoc]

theorem array_normal_error (xs : Array Term) (base : Int)
    (lower : ∀ x ∈ xs.toList, base ≤ x.1.exponent)
    (scaleLower : base ≤ selectedScale xs.toList)
    (nonzero : alignedTotal xs.toList ≠ 0)
    (range : -126 ≤ selectedScale xs.toList+((alignedTotal xs.toList).natAbs.log2 : Int) ∧
      selectedScale xs.toList+((alignedTotal xs.toList).natAbs.log2 : Int) ≤ 127) :
    let t := alignedTotal xs.toList;
    let output := decodedUnits (arrayCore xs) (base-((23-(t.natAbs.log2 : Int)).toNat : Nat));
    -normalBudget xs.toList base ≤ idealUnits xs.toList base*expansion t-output ∧
      idealUnits xs.toList base*expansion t-output ≤ normalBudget xs.toList base := by
  dsimp only
  rw [arrayCore_eq_list,normal_word_units _ _ scaleLower nonzero range]
  exact actual_normal_error xs.toList base lower scaleLower nonzero range

open GB300Math.TensorSubnormalBounds

def subnormalUnits (xs : List Term) (base : Int) : Int :=
  resultUnits (alignedTotal xs) (selectedScale xs)*(2^(selectedScale xs-base).toNat : Nat)

def subnormalBudget (xs : List Term) (base : Int) : Int :=
  ((xs.length*(2^(selectedScale xs-base).toNat-1) : Nat) : Int)*(2^(selectedScale xs+149).toNat : Nat) +
    budget (selectedScale xs)*(2^(selectedScale xs-base).toNat : Nat)

theorem actual_subnormal_error (xs : List Term) (base : Int)
    (lower : ∀ x ∈ xs, base ≤ x.1.exponent)
    (scaleLower : base ≤ selectedScale xs)
    (nonzero : alignedTotal xs ≠ 0)
    (range : selectedScale xs+((alignedTotal xs).natAbs.log2 : Int) < -126) :
    -subnormalBudget xs base ≤ idealUnits xs base*(2^(selectedScale xs+149).toNat : Nat)-subnormalUnits xs base ∧
      idealUnits xs base*(2^(selectedScale xs+149).toNat : Nat)-subnormalUnits xs base ≤ subnormalBudget xs base := by
  have ha := actual_alignment_error xs base lower scaleLower
  have hr := subnormal_encoder_error (alignedTotal xs) (selectedScale xs) nonzero range
  have hp : 0 ≤ ((2^(selectedScale xs+149).toNat : Nat) : Int) := Int.ofNat_le.mpr (Nat.zero_le _)
  have hq : 0 ≤ ((2^(selectedScale xs-base).toNat : Nat) : Int) := Int.ofNat_le.mpr (Nat.zero_le _)
  have al := Int.mul_le_mul_of_nonneg_right ha.1 hp
  have ah := Int.mul_le_mul_of_nonneg_right ha.2 hp
  have rl := Int.mul_le_mul_of_nonneg_right hr.1 hq
  have rh := Int.mul_le_mul_of_nonneg_right hr.2 hq
  unfold sourceUnits at rl rh
  have he : alignedTotal xs*(2^(selectedScale xs-base).toNat : Nat)*(2^(selectedScale xs+149).toNat : Nat) =
    alignedTotal xs*(2^(selectedScale xs+149).toNat : Nat)*(2^(selectedScale xs-base).toNat : Nat) := by
    rw [Int.mul_assoc,Int.mul_comm (2^(selectedScale xs-base).toNat : Nat),← Int.mul_assoc]
  simp only [Int.sub_mul,Int.neg_mul,he] at al ah rl rh
  unfold subnormalBudget subnormalUnits
  omega

theorem subnormal_word_units (xs : List Term) (base : Int)
    (scaleLower : base ≤ selectedScale xs)
    (nonzero : alignedTotal xs ≠ 0)
    (range : selectedScale xs+((alignedTotal xs).natAbs.log2 : Int) < -126) :
    decodedUnits (roundedTotal xs) (base-((selectedScale xs+149).toNat : Nat)) =
      subnormalUnits xs base := by
  have he : (-149-(base-((selectedScale xs+149).toNat : Nat))).toNat =
      (-149-selectedScale xs).toNat+(selectedScale xs-base).toNat := by omega
  unfold decodedUnits roundedTotal subnormalUnits resultUnits
  rw [decode_encode_subnormal _ _ nonzero range]
  dsimp only
  rw [he,Nat.pow_add,Int.natCast_mul,← Int.mul_assoc]

theorem array_subnormal_error (xs : Array Term) (base : Int)
    (lower : ∀ x ∈ xs.toList, base ≤ x.1.exponent)
    (scaleLower : base ≤ selectedScale xs.toList)
    (nonzero : alignedTotal xs.toList ≠ 0)
    (range : selectedScale xs.toList+((alignedTotal xs.toList).natAbs.log2 : Int) < -126) :
    let output := decodedUnits (arrayCore xs) (base-((selectedScale xs.toList+149).toNat : Nat));
    -subnormalBudget xs.toList base ≤ idealUnits xs.toList base*(2^(selectedScale xs.toList+149).toNat : Nat)-output ∧
      idealUnits xs.toList base*(2^(selectedScale xs.toList+149).toNat : Nat)-output ≤ subnormalBudget xs.toList base := by
  dsimp only
  rw [arrayCore_eq_list,subnormal_word_units _ _ scaleLower nonzero range]
  exact actual_subnormal_error xs.toList base lower scaleLower nonzero range

theorem array_zero_error (xs : Array Term) (base : Int)
    (lower : ∀ x ∈ xs.toList, base ≤ x.1.exponent)
    (scaleLower : base ≤ selectedScale xs.toList)
    (zero : alignedTotal xs.toList=0) :
    let loss : Int := (xs.size*(2^(selectedScale xs.toList-base).toNat-1) : Nat);
    arrayCore xs=0 ∧ -loss ≤ idealUnits xs.toList base ∧ idealUnits xs.toList base ≤ loss := by
  have h := actual_alignment_error xs.toList base lower scaleLower
  simp only [zero,Int.zero_mul,Int.sub_zero,Array.length_toList] at h
  exact ⟨by rw [arrayCore_eq_list]; simp [roundedTotal,zero,encode_zero],h⟩

#print axioms normal_word_units
#print axioms array_normal_error
#print axioms actual_subnormal_error
#print axioms subnormal_word_units
#print axioms array_subnormal_error
#print axioms array_zero_error

#print axioms fold_sum
#print axioms aligned_total_eq
#print axioms grid_total_eq
#print axioms actual_alignment_error
#print axioms actual_normal_error
end GB300Math.TensorArrayEncodingBounds

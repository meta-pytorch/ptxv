/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MufuRcp64Output
import MufuRcp64FullDecode
set_option maxRecDepth 8000
namespace GB300Math.Rcp64Bounds
open FloatLib.Floats.Formats.BinaryInterchange
open MufuRcp32Bounds

theorem low_word_product (m q low : Int) (qlo : 0 ≤ q) (qhi : q ≤ 2^21)
    (llo : 0 ≤ low) (lhi : low ≤ 2^32)
    (lo : -(2^21 : Int) ≤ m*q-2^41) (hi : m*q-2^41 ≤ 2^18) :
    -(2^53 : Int) ≤ (m*2^32+low)*q-2^73 ∧
    (m*2^32+low)*q-2^73 ≤ 9*2^50 := by
  have hp : low*q ≤ (2^32 : Int)*2^21 := mul_le_mul lhi qhi qlo (by positivity)
  have hp0 : 0 ≤ low*q := mul_nonneg llo qlo
  constructor
  · calc
      -(2^53 : Int) = -(2^21 : Int)*2^32+0 := by decide
      _ ≤ (m*q-2^41)*2^32+low*q := add_le_add (mul_le_mul_of_nonneg_right lo (by positivity)) hp0
      _ = _ := by ring
  · calc
      _ = (m*q-2^41)*2^32+low*q := by ring
      _ ≤ (2^18 : Int)*2^32+(2^32 : Int)*2^21 := add_le_add (mul_le_mul_of_nonneg_right hi (by positivity)) hp
      _ = 9*2^50 := by decide

#print axioms low_word_product

theorem quotient_bound (n : Nat) (hn : n ≤ 2^24) : ((n/8 : Nat) : Int) ≤ 2^21 := by omega

noncomputable def fullInput (f low : Nat) : ℝ := ((2^20+(f : ℝ))*2^32+(low : ℝ))/2^52

theorem real_product_error (m q low : Nat) (hm : 0 < m)
    (rlo : -(2^53 : Int) ≤ ((m : Int)*2^32+low)*q-2^73)
    (rhi : ((m : Int)*2^32+low)*q-2^73 ≤ 9*2^50) :
    |(q : ℝ)/2^21-1/(((m : ℝ)*2^32+low)/2^52)| /
      |1/(((m : ℝ)*2^32+low)/2^52)| ≤ 9/(2^23 : ℝ) := by
  have hlo : -(2^53 : ℝ) ≤ ((m : ℝ)*2^32+low)*q-2^73 := by
    have h := (Int.cast_le (R := ℝ)).mpr rlo
    simpa only [Int.cast_neg,Int.cast_pow,Int.cast_ofNat,Int.cast_sub,Int.cast_mul,Int.cast_add,Int.cast_natCast] using h
  have hhi : ((m : ℝ)*2^32+low)*q-2^73 ≤ 9*2^50 := by
    have h := (Int.cast_le (R := ℝ)).mpr rhi
    simpa only [Int.cast_neg,Int.cast_pow,Int.cast_ofNat,Int.cast_sub,Int.cast_mul,Int.cast_add,Int.cast_natCast] using h
  rw [relative_eq_product _ _ (by positivity)]
  have identity : ((m : ℝ)*2^32+low)/2^52*((q : ℝ)/2^21)-1 =
      (((m : ℝ)*2^32+low)*q-2^73)/2^73 := by ring
  rw [identity,abs_div]
  have habs : |((m : ℝ)*2^32+low)*q-2^73| ≤ 9*2^50 :=
    abs_le.mpr ⟨by linarith,hhi⟩
  rw [abs_of_pos (by positivity : (0 : ℝ) < 2^73)]
  apply (div_le_iff₀ (by positivity : (0 : ℝ) < 2^73)).2
  calc
    _ ≤ 9*2^50 := habs
    _ = (9/(2^23 : ℝ))*2^73 := by norm_num

#print axioms real_product_error

theorem full_normalized_error (f : Fin (2^20)) (low : Fin (2^32)) :
    |output f.val-1/fullInput f.val low.val|/|1/fullInput f.val low.val| ≤ 9/(2^23 : ℝ) := by
  let n := Fp64Seed.interpolate false (f.val*8)
  let i : Fin (2^23) := ⟨f.val*8,by have h := f.isLt; omega⟩
  have nbound : n ≤ 2^24 := by
    dsimp [n]
    rw [same_interpolation]
    exact Int.toNat_le_toNat (mantissa_range i).2
  have qhi : ((n/8 : Nat) : Int) ≤ 2^21 := quotient_bound n nbound
  have lhi : (low.val : Int) ≤ 2^32 := by have h := low.isLt; omega
  have h := seed_product_bound f
  have result := low_word_product (2^20+(f.val : Int)) (n/8 : Nat) low.val
    (Int.natCast_nonneg _) qhi (Int.natCast_nonneg _) lhi h.1 h.2
  have hlo : -((2: Int)^53) ≤ (((2^20+f.val : Nat): Int)*2^32+low.val)*(n/8 : Nat)-2^73 := by
    simpa only [Nat.cast_add, Nat.cast_pow, Nat.cast_ofNat] using result.1
  have hhi : (((2^20+f.val : Nat): Int)*2^32+low.val)*(n/8 : Nat)-2^73 ≤ 9*2^50 := by
    simpa only [Nat.cast_add, Nat.cast_pow, Nat.cast_ofNat] using result.2
  have answer := real_product_error (2^20+f.val) (n/8) low.val (by omega) hlo hhi
  simpa only [output,fullInput,Nat.cast_add,Nat.cast_pow,Nat.cast_ofNat] using answer

#print axioms full_normalized_error

theorem encoded_full_normalized_error (f : Fin (2^20)) (low : Fin (2^32)) :
    let word := (1023*2^20+f.val)*2^32+low.val
    ∃ x y : Rat,
      (Model.ofNatBits word : Model FloatFormat.binary64).toRat? = some x ∧
      (Model.ofNatBits (Fp64Seed.evaluate false word) : Model FloatFormat.binary64).toRat? = some y ∧
      |(y : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| ≤ 9/(2^23 : ℝ) := by
  dsimp only
  let n := Fp64Seed.interpolate false (f.val*8)
  let i : Fin (2^23) := ⟨f.val*8,by have h := f.isLt; omega⟩
  have hr := mantissa_range i
  have lo : 2^23 ≤ n := by
    dsimp [n];rw [same_interpolation]
    exact Int.toNat_le_toNat hr.1
  have hi : n ≤ 2^24 := by
    dsimp [n];rw [same_interpolation]
    exact Int.toNat_le_toNat hr.2
  refine ⟨((((2^20+f.val)*2^32+low.val : Nat) : Rat)/2^52), ((n/8 : Nat) : Rat)/2^21,
    decode_full_normalized f low, ?_, ?_⟩
  · rw [Fp64Seed.ignores_low_word false _ low.val low.isLt, high_word_normalized]
    exact decode_seed_mantissa n lo hi
  · have h := full_normalized_error f low
    dsimp [fullInput,output,n] at h ⊢
    norm_num at h ⊢
    exact h

#print axioms low_word_product
#print axioms full_normalized_error
#print axioms encoded_full_normalized_error
end GB300Math.Rcp64Bounds

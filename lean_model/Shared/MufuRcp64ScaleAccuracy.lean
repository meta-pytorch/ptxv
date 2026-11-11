/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MufuRcp64Scaling
set_option maxRecDepth 8000
namespace GB300Math.Rcp64Bounds
open FloatLib.Floats.Formats.BinaryInterchange
open MufuRcp32Bounds

theorem decode_full_scaled (e : Fin 2046) (f : Fin (2^20)) (low : Fin (2^32)) :
    (Model.ofNatBits ((((1+e.val)*2^20+f.val)*2^32+low.val)) : Model FloatFormat.binary64).toRat? =
      some ((((2^20+f.val)*2^32+low.val : Nat) : Rat)*(2 : Rat)^((e.val : Int)-1074)) := by
  let full : Fin (2^52) := ⟨f.val*2^32+low.val,by have hf:=f.isLt;have hl:=low.isLt;omega⟩
  have d := full_decode_normal e full
  have hw : (1+e.val)*2^52+full.val = ((1+e.val)*2^20+f.val)*2^32+low.val := by dsimp [full];ring
  have hm : 2^52+full.val = (2^20+f.val)*2^32+low.val := by dsimp [full];ring
  rwa [hw,hm] at d

theorem decode_full_rcp_scaled (e : Fin 2044) (f : Fin (2^20)) (low : Fin (2^32)) :
    (Model.ofNatBits (Fp64Seed.evaluate false (((1+e.val)*2^20+f.val)*2^32+low.val)) : Model FloatFormat.binary64).toRat? =
      some (((Fp64Seed.interpolate false (f.val*8)/8 : Nat) : Rat)*(2 : Rat)^(1001-(e.val : Int))) := by
  let n := Fp64Seed.interpolate false (f.val*8)
  let i : Fin (2^23) := ⟨f.val*8,by have hf:=f.isLt;omega⟩
  have hr := mantissa_range i
  have lo : 2^23 ≤ n := by dsimp [n];rw [same_interpolation];exact Int.toNat_le_toNat hr.1
  have hi : n ≤ 2^24 := by dsimp [n];rw [same_interpolation];exact Int.toNat_le_toNat hr.2
  have hw : (((1+e.val)*2^20+f.val)*2^32+low.val)/2^32 = (1+e.val)*2^20+f.val := by have hl:=low.isLt;omega
  unfold Fp64Seed.evaluate
  rw [hw,high_word_scaled]
  exact decode_scaled_seed e n lo hi

theorem full_scaled_relative_error (e : Int) (f : Fin (2^20)) (low : Fin (2^32)) :
    let x : ℝ := (((2^20+f.val)*2^32+low.val : Nat) : ℝ)*2^(e-1074)
    let y : ℝ := ((Fp64Seed.interpolate false (f.val*8)/8 : Nat) : ℝ)*2^(1001-e)
    |y-1/x|/|1/x| ≤ 9/(2^23 : ℝ) := by
  dsimp only
  have hx : ((((2^20+f.val)*2^32+low.val : Nat) : ℝ))*(2 : ℝ)^(e-1074) ≠ 0 := by positivity
  rw [relative_eq_product _ _ hx]
  have product (m q : ℝ) : (m*2^(e-1074))*(q*2^(1001-e)) = m*q/2^73 := by
    calc
      _ = m*q*((2 : ℝ)^(e-1074)*2^(1001-e)) := by ring
      _ = _ := by
        rw [←zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
        have hp : e-1074+(1001-e) = -73 := by ring
        rw [hp]
        norm_num
        ring
  rw [product]
  have h := full_normalized_error f low
  rw [relative_eq_product _ _ (by dsimp [fullInput];positivity)] at h
  dsimp [fullInput,output] at h
  norm_num only [Nat.cast_add,Nat.cast_mul,Nat.cast_pow,Nat.cast_ofNat] at h ⊢
  convert h using 1
  ring_nf

theorem positive_full_scaled_error (e : Fin 2044) (f : Fin (2^20)) (low : Fin (2^32)) :
    ∃ x y : Rat,
      (Model.ofNatBits (((1+e.val)*2^20+f.val)*2^32+low.val) : Model FloatFormat.binary64).toRat? = some x ∧
      (Model.ofNatBits (Fp64Seed.evaluate false (((1+e.val)*2^20+f.val)*2^32+low.val)) : Model FloatFormat.binary64).toRat? = some y ∧
      |(y : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| ≤ 9/(2^23 : ℝ) := by
  refine ⟨(((2^20+f.val)*2^32+low.val : Nat) : Rat)*(2 : Rat)^((e.val : Int)-1074),
    ((Fp64Seed.interpolate false (f.val*8)/8 : Nat) : Rat)*(2 : Rat)^(1001-(e.val : Int)),?_,decode_full_rcp_scaled e f low,?_⟩
  · exact decode_full_scaled ⟨e.val,by have he:=e.isLt;omega⟩ f low
  · have h := full_scaled_relative_error e.val f low
    norm_num at h ⊢
    exact h

#print axioms decode_full_scaled
#print axioms decode_full_rcp_scaled
#print axioms full_scaled_relative_error
#print axioms positive_full_scaled_error
end GB300Math.Rcp64Bounds

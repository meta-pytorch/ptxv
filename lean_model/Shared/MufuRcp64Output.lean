/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MufuRcp64Bounds
import MufuRcp64Decode
set_option maxRecDepth 8000
namespace GB300Math.Rcp64Bounds
open FloatLib.Floats.Formats.BinaryInterchange
open MufuRcp32Bounds

theorem high_word_fields (w f n : Nat)
    (hw : w < 2^32) (hs : w &&& 0x80000000 = 0)
    (he : w/2^20%2048 = 1023) (hf : w%2^20 = f)
    (hn : Fp64Seed.interpolate false (f*8) = n) :
    Fp64Seed.highWord false w =
      (if n = 2^24 then 1023 else 1022)*2^20 ||| (n%2^23)/8 := by
  unfold Fp64Seed.highWord
  rw [Nat.mod_eq_of_lt hw]
  dsimp only
  rw [hs,he,hf]
  simp only [Bool.false_eq_true,false_and,ite_false,Nat.zero_mul,Nat.zero_add]
  rw [hn]
  have hne : ¬(1023 : Nat) = 2047 := by decide
  have hnz : ¬(1023 : Nat) = 0 := by decide
  simp only [hne,hnz,false_and,ite_false,Nat.zero_or]
  by_cases h : n = 2^24
  · simp only [ite_eq_left h]
    have calcExp : (2046-(1023 : Nat)-0 : Int) = 1023 := by decide
    rw [calcExp]
    have hp : ¬(1023 : Int) ≤ 0 := by decide
    rw [ite_eq_right hp, show Int.toNat 1023 = 1023 by decide]
  · simp only [ite_eq_right h]
    have calcExp : (2046-(1023 : Nat)-1 : Int) = 1022 := by decide
    rw [calcExp]
    have hp : ¬(1022 : Int) ≤ 0 := by decide
    rw [ite_eq_right hp, show Int.toNat 1022 = 1022 by decide]

theorem high_word_normalized (f : Fin (2^20)) :
    Fp64Seed.highWord false (1023*2^20+f.val) =
      let n := Fp64Seed.interpolate false (f.val*8)
      (if n = 2^24 then 1023 else 1022)*2^20 + (n%2^23)/8 := by
  have hf := f.isLt
  have hw : 1023*2^20+f.val < 2^32 := by omega
  have hs : (1023*2^20+f.val) &&& 0x80000000 = 0 := sign_zero_of_lt _ (by omega)
  have he : (1023*2^20+f.val)/2^20%2048 = 1023 := by omega
  have hm : (1023*2^20+f.val)%2^20 = f.val := by omega
  have frac : (Fp64Seed.interpolate false (f.val*8)%2^23)/8 < 2^20 := by omega
  have hor (a : Nat) : a*2^20 ||| ((Fp64Seed.interpolate false (f.val*8)%2^23)/8) =
      a*2^20 + ((Fp64Seed.interpolate false (f.val*8)%2^23)/8) := by
    simpa only [Nat.mul_comm] using (Nat.two_pow_add_eq_or_of_lt frac a).symm
  rw [high_word_fields _ _ _ hw hs he hm rfl]
  exact hor _

theorem decode_seed_mantissa (n : Nat) (lo : 2^23 ≤ n) (hi : n ≤ 2^24) :
    (Model.ofNatBits (((if n = 2^24 then 1023 else 1022)*2^20+(n%2^23)/8)*2^32) : Model FloatFormat.binary64).toRat? =
      some (((n/8 : Nat) : Rat)/2^21) := by
  by_cases h : n = 2^24
  · subst n
    decide +kernel
  · have hf : (n%2^23)/8 < 2^20 := by omega
    let f : Fin (2^20) := ⟨(n%2^23)/8,hf⟩
    have d := decode_normal_high (1021 : Fin 2046) f
    have hh : 2^20+f.val = n/8 := by dsimp [f]; omega
    rw [ite_eq_right h]
    change (Model.ofNatBits (((1+(1021 : Fin 2046).val)*2^20+f.val)*2^32) : Model FloatFormat.binary64).toRat? = _
    rw [d,hh]
    norm_num
    ring

theorem encoded_normalized_error (f : Fin (2^20)) :
    ∃ x y : Rat,
      (Model.ofNatBits ((1023*2^20+f.val)*2^32) : Model FloatFormat.binary64).toRat? = some x ∧
      (Model.ofNatBits (Fp64Seed.highWord false (1023*2^20+f.val)*2^32) : Model FloatFormat.binary64).toRat? = some y ∧
      |(y : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| ≤ 1/(2^20 : ℝ) := by
  let n := Fp64Seed.interpolate false (f.val*8)
  let i : Fin (2^23) := ⟨f.val*8,by have h := f.isLt; omega⟩
  have hr := mantissa_range i
  have lo : 2^23 ≤ n := by
    dsimp [n]; rw [same_interpolation]
    exact Int.toNat_le_toNat hr.1
  have hi : n ≤ 2^24 := by
    dsimp [n]; rw [same_interpolation]
    exact Int.toNat_le_toNat hr.2
  refine ⟨((2^20+f.val : Nat) : Rat)/2^20, ((n/8 : Nat) : Rat)/2^21, ?_, ?_, ?_⟩
  · have d := decode_normal_high (1022 : Fin 2046) f
    norm_num at d ⊢
    convert d using 1 <;> ring
  · rw [high_word_normalized]
    exact decode_seed_mantissa n lo hi
  · have h := normalized_error f
    dsimp [input,output] at h
    norm_num at h ⊢
    exact h

#print axioms high_word_fields
#print axioms high_word_normalized
#print axioms decode_seed_mantissa
#print axioms encoded_normalized_error
end GB300Math.Rcp64Bounds

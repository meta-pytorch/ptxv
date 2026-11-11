/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Fp64RsqBounds
import MufuRcp64FullDecode

namespace GB300Math.Fp64RsqFloatLib
open FloatLib.Floats.Formats.BinaryInterchange

theorem normal_decode (word : Nat)
    (he : 1 ≤ word / 2^52 ∧ word / 2^52 ≤ 2046) :
    (Model.ofNatBits word : Model FloatFormat.binary64).toRat? =
      some (Fp64SeedBounds.positiveNormalValue word) := by
  let e : Fin 2046 := ⟨word / 2^52 - 1, by omega⟩
  let f : Fin (2^52) := ⟨word % 2^52, by omega⟩
  have hw : (1+e.val)*2^52+f.val = word := by dsimp [e,f]; omega
  have de := Rcp64Bounds.full_decode_normal e f
  rw [hw] at de
  rw [de]
  congr 1
  unfold Fp64SeedBounds.positiveNormalValue
  have hm : word / 2^52 % 2048 = word / 2^52 := by omega
  rw [hm]
  dsimp [e,f]
  push_cast
  congr 1
  congr 1
  omega

theorem normal_output (word : Nat)
    (he : 1 ≤ word / 2^52 ∧ word / 2^52 ≤ 2046) :
    1 ≤ Fp64Seed.evaluate true word / 2^52 ∧
      Fp64Seed.evaluate true word / 2^52 ≤ 2046 := by
  have hw : word/2^52*2^52+(word/2^32%2^20)*2^32+word%2^32 = word := by omega
  have h := Fp64RsqBounds.output_is_positive_normal (word/2^52)
    (word/2^32%2^20) (word%2^32) he (by omega) (by omega)
  simpa only [hw] using h

theorem normal_input_positive (word : Nat) :
    0 < Fp64SeedBounds.positiveNormalValue word := by
  unfold Fp64SeedBounds.positiveNormalValue
  positivity

theorem decoded_ideal_error (word : Nat)
    (he : 1 ≤ word / 2^52 ∧ word / 2^52 ≤ 2046) :
    ∃ x y : Rat,
      (Model.ofNatBits word : Model FloatFormat.binary64).toRat? = some x ∧
      (Model.ofNatBits (Fp64Seed.evaluate true word) : Model FloatFormat.binary64).toRat? = some y ∧
      0 < x ∧
      |(y : ℝ)-1/Real.sqrt (x : ℝ)| ≤ (9/2^23)/Real.sqrt (x : ℝ) := by
  refine ⟨Fp64SeedBounds.positiveNormalValue word,
    Fp64SeedBounds.positiveNormalValue (Fp64Seed.evaluate true word),
    normal_decode word he, normal_decode _ (normal_output word he),
    normal_input_positive word, ?_⟩
  exact Fp64RsqBounds.all_positive_normal_words word he

theorem decoded_relative_error (word : Nat)
    (he : 1 ≤ word / 2^52 ∧ word / 2^52 ≤ 2046) :
    ∃ x y : Rat,
      (Model.ofNatBits word : Model FloatFormat.binary64).toRat? = some x ∧
      (Model.ofNatBits (Fp64Seed.evaluate true word) : Model FloatFormat.binary64).toRat? = some y ∧
      0 < x ∧
      |(y : ℝ)-1/Real.sqrt (x : ℝ)| / |1/Real.sqrt (x : ℝ)| ≤ 9/2^23 := by
  obtain ⟨x,y,hx,hy,hp,hbound⟩ := decoded_ideal_error word he
  refine ⟨x,y,hx,hy,hp,?_⟩
  have hpx : (0 : ℝ) < x := by exact_mod_cast hp
  have hs := Real.sqrt_pos.mpr hpx
  rw [abs_of_pos (by positivity : (0 : ℝ) < 1/Real.sqrt (x : ℝ))]
  apply (div_le_iff₀ (by positivity : (0 : ℝ) < 1/Real.sqrt (x : ℝ))).mpr
  simpa only [div_eq_mul_inv, one_mul] using hbound

theorem positive_normal_count : 2046*2^52 = 9214364837600034816 := by decide +kernel

#print axioms normal_decode
#print axioms normal_output
#print axioms normal_input_positive
#print axioms decoded_ideal_error
#print axioms decoded_relative_error
#print axioms positive_normal_count
end GB300Math.Fp64RsqFloatLib

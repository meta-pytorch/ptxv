/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import TanhDiscontinuity

namespace GB300Math.TanhAccuracyFloor
open FloatLib.Floats.Formats.BinaryInterchange TanhDiscontinuity Exp2Analytic
set_option maxRecDepth 1000000
set_option maxHeartbeats 2000000

def input : Rat := 10223615/8388608
def center : Rat := 100741182370871859/2^60

theorem rational_certificate :
    0 ≤ seriesQ (-input/8) ∧ seriesQ (-input/8) ≤ 1 ∧
    |(seriesQ (-input/8))^16-center| ≤ 1/10^15 := by decide +kernel

theorem native_bits : Mufu.tanh 0x3f9bffff = some 0x3f56dbe4 := by decide +kernel

theorem decoded_values :
    value 0x3f9bffff = (input:ℝ) ∧ value 0x3f56dbe4 = 3520249/4194304 := by
  have h :
      (Model.ofNatBits 0x3f9bffff : Model FloatFormat.binary32).toRat? = some input ∧
      (Model.ofNatBits 0x3f56dbe4 : Model FloatFormat.binary32).toRat? = some (3520249/4194304) := by
    decide +kernel
  have h0 := Model.Algebraic.toReal_of_toRat?_eq_some h.1
  have h1 := Model.Algebraic.toReal_of_toRat?_eq_some h.2
  norm_num [value] at h0 h1 ⊢
  exact ⟨h0,h1⟩

theorem exponential_bound :
    |Real.exp (-2*(input:ℝ))-(center:ℝ)| ≤ 17/10^12 := by
  have hr : 0 ≤ series (-(input:ℝ)/8) ∧ series (-(input:ℝ)/8) ≤ 1 := by
    have h := (show (0:ℝ) ≤ (seriesQ (-input/8):ℝ) ∧ (seriesQ (-input/8):ℝ) ≤ 1 by
      exact_mod_cast And.intro rational_certificate.1 rational_certificate.2.1)
    simpa only [series_cast,Rat.cast_div,Rat.cast_neg,Rat.cast_ofNat] using h
  apply TanhCentralAnalytic.center_error (input:ℝ) _ (center:ℝ) (by norm_num [input]) rfl hr
  have h := (Rat.cast_le (K:=ℝ)).mpr rational_certificate.2.2
  simpa only [Rat.cast_abs,Rat.cast_sub,Rat.cast_pow,series_cast,Rat.cast_div,
    Rat.cast_neg,Rat.cast_ofNat,Rat.cast_one] using h

theorem strict_lower_bound : 1/2^17 < error 0x3f9bffff := by
  have he := exponential_bound
  have hepos := Real.exp_pos (-2*(input:ℝ))
  have hlow := (abs_le.mp he).1
  have ht : Real.tanh (input:ℝ) < (3520249:ℝ)/4194304-1/2^17 := by
    rw [TanhCentralAnalytic.tanh_negative_exp]
    apply (div_lt_iff₀ (by positivity)).mpr
    norm_num [center] at hlow ⊢
    nlinarith
  unfold error
  rw [native_bits,Option.getD_some,decoded_values.1,decoded_values.2]
  have ha := le_abs_self ((3520249:ℝ)/4194304-Real.tanh (input:ℝ))
  linarith

theorem not_uniform_2neg17 :
    ¬ (∀ word : Nat, word < 0x7f800000 → error word ≤ 1/2^17) := by
  intro h
  exact (not_le.mpr strict_lower_bound) (h _ (by decide))

theorem uniform_upper_bound (word : Nat) (hw : word < 2^32)
    (hf : Mufu.magnitude word < 0x7f800000) : error word ≤ 1/2^16 := by
  obtain ⟨out,ho,_,hb⟩ := TanhCentralUniform.all_finite word hw hf
  unfold error value
  rw [ho,Option.getD_some]
  exact hb

theorem within_factor_two (bound : ℝ)
    (hb : ∀ word : Nat, word < 0x7f800000 → error word ≤ bound) : 1/2^16 < 2*bound := by
  have hw := hb 0x3f9bffff (by decide)
  have hl := strict_lower_bound
  norm_num only [Nat.reducePow] at hl ⊢
  linarith

#print axioms rational_certificate
#print axioms native_bits
#print axioms decoded_values
#print axioms exponential_bound
#print axioms strict_lower_bound
#print axioms not_uniform_2neg17
#print axioms uniform_upper_bound
#print axioms within_factor_two
end GB300Math.TanhAccuracyFloor

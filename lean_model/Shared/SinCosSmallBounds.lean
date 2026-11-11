/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Exp2Bounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.Real.Pi.Bounds

set_option maxHeartbeats 0
set_option maxRecDepth 100000
namespace GB300Math.SinCosSmallBounds
open Exp2Bounds

theorem phase_zero (e m : Nat) (he : e ≤ 101) (hm : m < 2^24) :
    (if e = 0 ∨ (e : Int)-125 ≥ 25 then 0
     else if (e : Int)-125 ≥ 0 then m * 2^((e : Int)-125).toNat
     else if -((e : Int)-125) < 32 then m / 2^(-((e : Int)-125)).toNat
     else 0) % 2^25 = 0 := by
  have hneg : ¬(e : Int)-125 ≥ 0 := by omega
  have hbig : ¬(e : Int)-125 ≥ 25 := by omega
  have hpow : 2^24 ≤ 2^(-((e : Int)-125)).toNat := by
    exact Nat.pow_le_pow_right (by omega) (by omega)
  have hdiv : m / 2^(-((e : Int)-125)).toNat = 0 := Nat.div_eq_of_lt (lt_of_lt_of_le hm hpow)
  simp only [hneg, hbig, or_false, ite_false, hdiv]
  split_ifs <;> rfl

theorem sin_small_bits (word : Nat) (he : Mufu.exponent word ≤ 101) :
    Mufu.sincos word false = if Mufu.sign word ≠ 0 then 0x80000000 else 0 := by
  have hn : Mufu.exponent word ≠ 255 := by omega
  have hm : Mufu.fraction word + 2^23 < 2^24 := by unfold Mufu.fraction; omega
  have hp := phase_zero (Mufu.exponent word) (Mufu.fraction word+2^23) he hm
  have hz : Mufu.interpolate MufuTables.sin 0 = 0 := by decide +kernel
  simp only [Mufu.sincos, hn, hp, ↓reduceIte]
  simp [hz]

theorem cos_small_bits (word : Nat) (he : Mufu.exponent word ≤ 101) :
    Mufu.sincos word true = 0x3f800000 := by
  have hn : Mufu.exponent word ≠ 255 := by omega
  have hm : Mufu.fraction word + 2^23 < 2^24 := by unfold Mufu.fraction; omega
  have hp := phase_zero (Mufu.exponent word) (Mufu.fraction word+2^23) he hm
  simp only [Mufu.sincos, hn, hp, ↓reduceIte]
  decide +kernel

theorem small_input_magnitude (word : Nat) (he : Mufu.exponent word ≤ 101) :
    |finiteValue word| ≤ 1/2^25 := by
  have hfq : (Mufu.fraction word : ℝ) ≤ 2^23 := by
    exact_mod_cast (show Mufu.fraction word ≤ 2^23 by unfold Mufu.fraction; omega)
  unfold finiteValue
  rw [abs_mul]
  have hsign : |(if word < 2^31 then 1 else -1 : ℝ)| = 1 := by split <;> norm_num
  rw [hsign, one_mul]
  by_cases hz : Mufu.exponent word = 0
  · rw [ite_eq_left hz, abs_of_nonneg (by positivity)]
    norm_num
    linarith
  · rw [ite_eq_right hz, abs_of_nonneg (by positivity)]
    have hp : (2 : ℝ)^((Mufu.exponent word : Int)-150) ≤ (2 : ℝ)^(-49 : Int) := by
      apply zpow_le_zpow_right₀ (by norm_num : (1 : ℝ) ≤ 2)
      omega
    have h := mul_le_mul (show (2^23 : ℝ)+Mufu.fraction word ≤ 2^24 by norm_num; linarith)
      hp (by positivity) (by positivity)
    norm_num at h ⊢
    exact h

theorem sin_small_value (word : Nat) (he : Mufu.exponent word ≤ 101) :
    finiteValue (Mufu.sincos word false) = 0 := by
  rw [sin_small_bits word he]
  split <;> norm_num [finiteValue, Mufu.exponent, Mufu.fraction]

theorem sin_small_error (word : Nat) (he : Mufu.exponent word ≤ 101) :
    |finiteValue (Mufu.sincos word false)-Real.sin (2*Real.pi*finiteValue word)| ≤
      Real.pi/2^24 := by
  rw [sin_small_value word he, zero_sub, abs_neg]
  calc
    |Real.sin (2*Real.pi*finiteValue word)| ≤ |2*Real.pi*finiteValue word| := Real.abs_sin_le_abs
    _ = (2*Real.pi)*|finiteValue word| := by rw [abs_mul, abs_of_pos (by positivity)]
    _ ≤ (2*Real.pi)*(1/2^25) := mul_le_mul_of_nonneg_left (small_input_magnitude word he) (by positivity)
    _ = Real.pi/2^24 := by ring

theorem sin_small_rational_bound (word : Nat) (he : Mufu.exponent word ≤ 101) :
    |finiteValue (Mufu.sincos word false)-Real.sin (2*Real.pi*finiteValue word)| <
      22/(7*2^24) := by
  have h := sin_small_error word he
  have hp : Real.pi < 22/7 := by linarith [Real.pi_lt_d4]
  linarith

theorem cos_small_error (word : Nat) (he : Mufu.exponent word ≤ 101) :
    |finiteValue (Mufu.sincos word true)-Real.cos (2*Real.pi*finiteValue word)| ≤
      Real.pi^2/2^49 := by
  rw [cos_small_bits word he]
  have hv : finiteValue 0x3f800000 = 1 := by norm_num [finiteValue, Mufu.exponent, Mufu.fraction]
  rw [hv, abs_of_nonneg (by linarith [Real.cos_le_one (2*Real.pi*finiteValue word)])]
  have ha : (finiteValue word)^2 ≤ (1/2^25 : ℝ)^2 := by
    rw [←sq_abs]
    exact pow_le_pow_left₀ (abs_nonneg _) (small_input_magnitude word he) 2
  have hb := mul_le_mul_of_nonneg_left ha (sq_nonneg Real.pi)
  have hc := Real.cos_two_mul (Real.pi*finiteValue word)
  have hd := Real.sin_sq_add_cos_sq (Real.pi*finiteValue word)
  have hs := Real.sin_sq_le_sq (x := Real.pi*finiteValue word)
  rw [show 2*Real.pi*finiteValue word = 2*(Real.pi*finiteValue word) by ring]
  nlinarith

theorem sin_small_relative_error (word : Nat) (he : Mufu.exponent word ≤ 101)
    (hx : 0 < finiteValue word) :
    |finiteValue (Mufu.sincos word false)-Real.sin (2*Real.pi*finiteValue word)| /
      |Real.sin (2*Real.pi*finiteValue word)| = 1 := by
  have hxmax := (le_abs_self (finiteValue word)).trans (small_input_magnitude word he)
  have hp := Real.pi_pos
  have hs : 0 < Real.sin (2*Real.pi*finiteValue word) :=
    Real.sin_pos_of_pos_of_lt_pi (by positivity) (by nlinarith)
  rw [sin_small_value word he, zero_sub, abs_neg]
  exact div_self (ne_of_gt (abs_pos.mpr (ne_of_gt hs)))

theorem boundary_inside : Mufu.sincos 0x32ffffff false = 0 := by decide +kernel
theorem boundary_outside : Mufu.sincos 0x33000000 false = 877203968 := by decide +kernel

theorem adjacent_input_step :
    finiteValue 0x33000000-finiteValue 0x32ffffff = 1/2^49 := by
  norm_num [finiteValue, Mufu.exponent, Mufu.fraction]

theorem boundary_value : finiteValue 0x32ffffff = 1/2^25-1/2^49 := by
  norm_num [finiteValue, Mufu.exponent, Mufu.fraction]

theorem boundary_error_lower :
    Real.pi/2^24-Real.pi/2^48-(Real.pi/2^24)^3/6 <
      |finiteValue (Mufu.sincos 0x32ffffff false)-Real.sin (2*Real.pi*finiteValue 0x32ffffff)| := by
  rw [boundary_inside]
  have hv : finiteValue 0 = 0 := by norm_num [finiteValue, Mufu.exponent, Mufu.fraction]
  rw [hv, zero_sub, abs_neg, boundary_value]
  have harg : 0 < 2*Real.pi*(1/2^25-1/2^49) := by positivity
  have ht := Real.sin_gt_sub_cube harg
  have hle : 2*Real.pi*(1/2^25-1/2^49) ≤ Real.pi/2^24 := by nlinarith [Real.pi_pos]
  have hcube := pow_le_pow_left₀ harg.le hle 3
  have habs := le_abs_self (Real.sin (2*Real.pi*(1/2^25-1/2^49)))
  nlinarith

theorem boundary_amplification :
    (finiteValue (Mufu.sincos 0x33000000 false)-finiteValue (Mufu.sincos 0x32ffffff false)) /
      (finiteValue 0x33000000-finiteValue 0x32ffffff) > 100000000 := by
  rw [boundary_inside, boundary_outside, adjacent_input_step]
  norm_num [finiteValue, Mufu.exponent, Mufu.fraction]

#print axioms phase_zero
#print axioms sin_small_bits
#print axioms cos_small_bits
#print axioms small_input_magnitude
#print axioms sin_small_value
#print axioms sin_small_error
#print axioms sin_small_rational_bound
#print axioms cos_small_error
#print axioms sin_small_relative_error
#print axioms boundary_inside
#print axioms boundary_outside
#print axioms adjacent_input_step
#print axioms boundary_value
#print axioms boundary_error_lower
#print axioms boundary_amplification
end GB300Math.SinCosSmallBounds

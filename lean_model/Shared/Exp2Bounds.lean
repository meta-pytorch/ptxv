/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.Mufu
import Mathlib.Tactic

namespace GB300Math.Exp2Bounds

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

noncomputable def finiteValue (word : Nat) : ℝ :=
  (if word < 2^31 then 1 else -1) *
  (if Mufu.exponent word = 0 then (Mufu.fraction word : ℝ)*(2 : ℝ)^(-149 : Int)
   else (2^23+(Mufu.fraction word : ℝ))*(2 : ℝ)^((Mufu.exponent word : Int)-150))

theorem ex2_small_bits (word : Nat) (_hw : word < 2^32)
    (he : Mufu.exponent word ≤ 103) : Mufu.ex2 word = 0x3f800000 := by
  have hm : Mufu.magnitude word ≤ 0x7f800000 := by dsimp [Mufu.exponent,Mufu.magnitude] at *;omega
  have h1 : ¬135 ≤ Mufu.exponent word := by omega
  have h2 : ¬127 ≤ Mufu.exponent word := by omega
  have h3 : ¬127-Mufu.exponent word < 24 := by omega
  have ht : Mufu.interpolate MufuTables.ex2 0 = 2^23 := by decide +kernel
  simp [Mufu.ex2,not_lt.mpr hm,h1,h2,h3,ht]

theorem log_two_bound : 0 ≤ Real.log 2 ∧ Real.log 2 ≤ 7/10 := by
  refine ⟨Real.log_nonneg (by norm_num),?_⟩
  apply (Real.log_le_iff_le_exp (by norm_num : (0 : ℝ)<2)).mpr
  have h := Real.sum_le_exp_of_nonneg (by norm_num : (0 : ℝ) ≤ 7/10) 4
  norm_num [Finset.sum_range_succ] at h
  linarith

theorem real_bound (x : ℝ) (hx : |x| ≤ 1/2^23) :
    |(2 : ℝ)^x-1| ≤ 3/(4*2^23) := by
  have hl := log_two_bound
  have hy : |Real.log 2*x| ≤ 7/(10*2^23) := by
    rw [abs_mul,abs_of_nonneg hl.1]
    have h := mul_le_mul hl.2 hx (abs_nonneg x) (by norm_num : (0 : ℝ) ≤ 7/10)
    norm_num at h ⊢
    exact h
  have hs : (Real.log 2*x)^2 ≤ (7/(10*2^23) : ℝ)^2 := by
    rw [←sq_abs]
    exact pow_le_pow_left₀ (abs_nonneg _) hy 2
  rw [Real.rpow_def_of_pos (by norm_num)]
  have h := Real.abs_exp_sub_one_sub_id_le (show |Real.log 2*x| ≤ 1 by linarith)
  have ha := abs_add_le (Real.exp (Real.log 2*x)-1-(Real.log 2*x)) (Real.log 2*x)
  rw [sub_add_cancel] at ha
  norm_num only [Nat.reducePow] at hy hs ⊢
  nlinarith

theorem input_magnitude_bound (word : Nat) (he : Mufu.exponent word ≤ 103) :
    |finiteValue word| ≤ 1/2^23 := by
  have hf : Mufu.fraction word < 2^23 := by dsimp [Mufu.fraction];omega
  have hfq : (Mufu.fraction word : ℝ) ≤ 2^23 := by exact_mod_cast (show Mufu.fraction word ≤ 2^23 by omega)
  unfold finiteValue
  rw [abs_mul]
  have hsign : |(if word < 2^31 then 1 else -1 : ℝ)| = 1 := by split <;> norm_num
  rw [hsign,one_mul]
  by_cases hz : Mufu.exponent word = 0
  · rw [ite_eq_left hz,abs_of_nonneg (by positivity)]
    norm_num
    linarith
  · rw [ite_eq_right hz,abs_of_nonneg (by positivity)]
    have hp : (2 : ℝ)^((Mufu.exponent word : Int)-150) ≤ (2 : ℝ)^(-47 : Int) := by
      apply zpow_le_zpow_right₀ (by norm_num : (1 : ℝ) ≤ 2)
      omega
    have h := mul_le_mul (show (2^23 : ℝ)+Mufu.fraction word ≤ 2^24 by norm_num;linarith) hp (by positivity) (by positivity)
    norm_num at h ⊢
    exact h

theorem ex2_small_error (word : Nat) (hw : word < 2^32)
    (he : Mufu.exponent word ≤ 103) :
    |finiteValue (Mufu.ex2 word)-(2 : ℝ)^(finiteValue word)| ≤ 3/(4*2^23) := by
  rw [ex2_small_bits word hw he]
  have hv : finiteValue 0x3f800000 = 1 := by norm_num [finiteValue,Mufu.exponent,Mufu.fraction]
  rw [hv,abs_sub_comm]
  exact real_bound _ (input_magnitude_bound word he)

theorem log_two_lower : (3/5 : ℝ) ≤ Real.log 2 := by
  apply (Real.le_log_iff_exp_le (by norm_num : (0 : ℝ)<2)).mpr
  have h := Real.exp_bound' (by norm_num : (0 : ℝ) ≤ 3/5) (by norm_num : (3/5 : ℝ) ≤ 1) (n := 3) (by decide)
  norm_num [Finset.sum_range_succ] at h
  linarith

theorem negative_epsilon_value : finiteValue 0xb3800000 = -(1/2^24 : ℝ) := by
  norm_num [finiteValue,Mufu.exponent,Mufu.fraction]

theorem negative_epsilon_bits : Mufu.ex2 0xb3800000 = 0x3f800000 := by decide +kernel

theorem negative_epsilon_below_midpoint :
    (2 : ℝ)^(finiteValue 0xb3800000) < 1-1/2^25 := by
  rw [negative_epsilon_value,Real.rpow_def_of_pos (by norm_num)]
  have hl := log_two_bound
  have hlow := log_two_lower
  let t : ℝ := Real.log 2 * -(1/2^24)
  have ht : |t| ≤ 7/(10*2^24) := by
    dsimp [t]
    rw [abs_mul,abs_of_nonneg hl.1]
    norm_num
    linarith [hl.2]
  have hs : t^2 ≤ (7/(10*2^24) : ℝ)^2 := by
    rw [←sq_abs]
    exact pow_le_pow_left₀ (abs_nonneg _) ht 2
  have h := (abs_le.mp (Real.abs_exp_sub_one_sub_id_le (show |t| ≤ 1 by linarith))).2
  change Real.exp t < _
  dsimp [t] at h hs ⊢
  norm_num only [Nat.reducePow] at hs ⊢
  nlinarith

theorem negative_epsilon_error_exceeds_half_ulp :
    (1/2^25 : ℝ) < |finiteValue (Mufu.ex2 0xb3800000)-(2 : ℝ)^(finiteValue 0xb3800000)| := by
  rw [negative_epsilon_bits]
  have hv : finiteValue 0x3f800000 = 1 := by norm_num [finiteValue,Mufu.exponent,Mufu.fraction]
  rw [hv]
  have h := negative_epsilon_below_midpoint
  rw [abs_of_pos (by linarith)]
  linarith

theorem lower_neighbor_value : finiteValue 0x3f7fffff = 1-1/2^24 := by
  norm_num [finiteValue,Mufu.exponent,Mufu.fraction]

theorem outside_plateau_witness : Mufu.ex2 0x34000000 ≠ 0x3f800000 := by decide +kernel

#print axioms ex2_small_bits
#print axioms log_two_bound
#print axioms real_bound
#print axioms input_magnitude_bound
#print axioms ex2_small_error
#print axioms log_two_lower
#print axioms negative_epsilon_value
#print axioms negative_epsilon_bits
#print axioms negative_epsilon_below_midpoint
#print axioms negative_epsilon_error_exceeds_half_ulp
#print axioms lower_neighbor_value
#print axioms outside_plateau_witness
end GB300Math.Exp2Bounds

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Exp2Fraction
import Exp2Bounds
namespace GB300Math.Exp2Relative
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

theorem normalized (y a d : ℝ) (ha : 1≤a) (hy : |y-a|≤1/5000000)
    (hd : |d|≤7/(10*2^23)) :
    |y-a*Real.exp d|≤(a*Real.exp d)/2^21 := by
  have he:=Real.abs_exp_sub_one_le (show |d|≤1 by linarith)
  have he' : |a*Real.exp d-a|≤a*(7/(5*2^23)) := by
    rw [show a*Real.exp d-a=a*(Real.exp d-1) by ring,abs_mul,abs_of_nonneg (by linarith)]
    have hm:=mul_le_mul_of_nonneg_left he (show 0≤a by linarith)
    have hm':=mul_le_mul_of_nonneg_left hd (show 0≤a by linarith)
    nlinarith
  have helo: a-a*(7/(5*2^23))≤a*Real.exp d := by have:=abs_le.mp he';linarith
  have ht:=abs_sub_le y a (a*Real.exp d)
  have hrev : |a-a*Real.exp d|≤a*(7/(5*2^23)) := by rwa [abs_sub_comm]
  nlinarith

theorem normalized_exp (index : Nat) (hi : index<2^23) (d : ℝ)
    (hd : |d|≤1/2^23) :
    |(Mufu.interpolate MufuTables.ex2 index:ℝ)/2^23-
      Real.exp (Real.log 2*((index:ℝ)/2^23))*Real.exp (Real.log 2*d)| ≤
      (Real.exp (Real.log 2*((index:ℝ)/2^23))*Real.exp (Real.log 2*d))/2^21 := by
  apply normalized
  · exact Real.one_le_exp_iff.mpr (mul_nonneg (Real.log_nonneg (by norm_num)) (by positivity))
  · exact Exp2Fraction.fractional_error index hi
  · rw [abs_mul,abs_of_nonneg (Real.log_nonneg (by norm_num : (1:ℝ)≤2))]
    have h:=mul_le_mul (Exp2Bounds.log_two_bound).2 hd (abs_nonneg d) (by norm_num : (0:ℝ)≤7/10)
    nlinarith

theorem normalized_upward (y a d : ℝ) (ha : 1≤a)
    (hy : -(123/10^9:ℝ)≤y-a ∧ y-a≤196/10^9)
    (hd : 0≤d ∧ d≤1/2^23) :
    |y-a*Real.exp (Real.log 2*d)|≤(a*Real.exp (Real.log 2*d))/2^22 := by
  have he:=Exp2Bounds.real_bound d (by rw [abs_of_nonneg hd.1];exact hd.2)
  rw [Real.rpow_def_of_pos (by norm_num)] at he
  have hn : 1≤Real.exp (Real.log 2*d) := Real.one_le_exp_iff.mpr (mul_nonneg (Real.log_nonneg (by norm_num)) hd.1)
  have hmul:=mul_le_mul_of_nonneg_left (abs_le.mp he).2 (show 0≤a by linarith)
  have hprod : a≤a*Real.exp (Real.log 2*d) := by nlinarith
  exact abs_le.mpr ⟨by nlinarith,by nlinarith⟩

theorem normalized_zero (d : ℝ) (hd : |d|≤1/2^23) :
    |1-Real.exp (Real.log 2*d)|≤Real.exp (Real.log 2*d)/2^22 := by
  have he:=Exp2Bounds.real_bound d hd
  rw [Real.rpow_def_of_pos (by norm_num)] at he
  have hlo:=(abs_le.mp he).1
  rw [abs_sub_comm]
  linarith

theorem normalized_exp_tight (index : Nat) (hi : index<2^23) (d : ℝ)
    (hd : |d|≤1/2^23) (hdir : 0≤d ∨ index=0) :
    |(Mufu.interpolate MufuTables.ex2 index:ℝ)/2^23-
      Real.exp (Real.log 2*((index:ℝ)/2^23))*Real.exp (Real.log 2*d)| ≤
      (Real.exp (Real.log 2*((index:ℝ)/2^23))*Real.exp (Real.log 2*d))/2^22 := by
  rcases hdir with hp|hz
  · exact normalized_upward _ _ d
      (Real.one_le_exp_iff.mpr (mul_nonneg (Real.log_nonneg (by norm_num)) (by positivity)))
      (Exp2Fraction.fractional_signed_error index hi) ⟨hp,(abs_le.mp hd).2⟩
  · subst index
    have hz : Mufu.interpolate MufuTables.ex2 0=2^23 := by decide +kernel
    rw [hz]
    norm_num
    simpa only [show (2^22:ℝ)=4194304 by norm_num] using normalized_zero d hd

#print axioms normalized_upward
#print axioms normalized_zero
#print axioms normalized_exp_tight
#print axioms normalized
#print axioms normalized_exp
end GB300Math.Exp2Relative

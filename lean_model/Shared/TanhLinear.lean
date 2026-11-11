/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import TanhTail
import Exp2Bounds

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

namespace GB300Math.TanhLinear

theorem linear_error (x : ℝ) (hx : 0 ≤ x) (hu : x ≤ 1/256) :
    0 ≤ x-Real.tanh x ∧ x-Real.tanh x ≤ x^3/3 := by
  let e := Real.exp (2*x)
  let p := 1+2*x+2*x^2+4/3*x^3+2/3*x^4
  have he0 : 0 < e+1 := by dsimp [e];positivity
  have hlo := Real.sum_le_exp_of_nonneg (show 0 ≤ 2*x by linarith) 5
  have hhi := Real.exp_bound' (show 0 ≤ 2*x by linarith) (show 2*x ≤ 1 by linarith) (n:=5) (by decide)
  have hlo' : p ≤ e := by dsimp [p,e];norm_num [Finset.sum_range_succ] at hlo; nlinarith only [hlo]
  have hhi' : e ≤ p+8/25*x^5 := by dsimp [p,e];norm_num [Finset.sum_range_succ] at hhi; nlinarith only [hhi]
  have hpolylo : 0 ≤ (1-x+x^3/3)*p-(1+x-x^3/3) := by dsimp [p];ring_nf;positivity
  have hpolyhi : 0 ≤ (1+x)-(1-x)*(p+8/25*x^5) := by dsimp [p];ring_nf;positivity
  have hmlo := mul_le_mul_of_nonneg_left hlo' (show 0 ≤ 1-x+x^3/3 by nlinarith [pow_nonneg hx 3])
  have hmhi := mul_le_mul_of_nonneg_left hhi' (show 0 ≤ 1-x by linarith)
  have hid : (x-Real.tanh x)*(e+1) = (x-1)*e+x+1 := by
    have h := TanhTail.remainder x
    change 1-Real.tanh x = 2/(e+1) at h
    have hh := (eq_div_iff (ne_of_gt he0)).mp h
    nlinarith only [hh]
  constructor
  · have hnum : 0 ≤ (x-Real.tanh x)*(e+1) := by nlinarith only [hmhi,hpolyhi,hid]
    exact nonneg_of_mul_nonneg_left hnum he0
  · have hnum : (x-Real.tanh x)*(e+1) ≤ (x^3/3)*(e+1) := by nlinarith only [hmlo,hpolylo,hid]
    exact (mul_le_mul_iff_left₀ he0).mp hnum


theorem signed_error (x : ℝ) (hx : |x| ≤ 1/256) :
    |x-Real.tanh x| ≤ |x|^3/3 := by
  by_cases h : 0 ≤ x
  · have he := linear_error x h (by simpa [abs_of_nonneg h] using hx)
    simpa [abs_of_nonneg h,abs_of_nonneg he.1] using he.2
  · have hn : 0 ≤ -x := by linarith
    have he := linear_error (-x) hn (by simpa [abs_of_nonpos (le_of_not_ge h)] using hx)
    rw [Real.tanh_neg] at he
    have hi : x-Real.tanh x = -(-x- -Real.tanh x) := by ring
    rw [hi,abs_neg,abs_of_nonneg he.1,abs_of_nonpos (le_of_not_ge h)]
    exact he.2

theorem small_bits (word : Nat) (_hw : word < 2^32)
    (he : Mufu.exponent word ≤ 118) : Mufu.tanh word = some word := by
  have hm : Mufu.magnitude word ≤ 0x7f800000 := by dsimp [Mufu.exponent,Mufu.magnitude] at *;omega
  unfold Mufu.tanh
  split
  · omega
  · split
    · rfl
    · omega

theorem input_magnitude_bound (word : Nat) (he : Mufu.exponent word ≤ 118) :
    |Exp2Bounds.finiteValue word| ≤ 1/256 := by
  have hf : Mufu.fraction word < 2^23 := by dsimp [Mufu.fraction];omega
  have hfq : (Mufu.fraction word : ℝ) ≤ 2^23 := by exact_mod_cast (show Mufu.fraction word ≤ 2^23 by omega)
  unfold Exp2Bounds.finiteValue
  rw [abs_mul]
  have hsign : |(if word < 2^31 then 1 else -1 : ℝ)| = 1 := by split <;> norm_num
  rw [hsign,one_mul]
  by_cases hz : Mufu.exponent word = 0
  · rw [ite_eq_left hz,abs_of_nonneg (by positivity)]
    norm_num
    linarith
  · rw [ite_eq_right hz,abs_of_nonneg (by positivity)]
    have hp : (2 : ℝ)^((Mufu.exponent word : Int)-150) ≤ (2 : ℝ)^(-32 : Int) := by
      apply zpow_le_zpow_right₀ (by norm_num : (1 : ℝ) ≤ 2)
      omega
    have h := mul_le_mul (show (2^23 : ℝ)+Mufu.fraction word ≤ 2^24 by norm_num;linarith) hp (by positivity) (by positivity)
    norm_num at h ⊢
    exact h

theorem small_bits_ideal_error (word : Nat) (hw : word < 2^32)
    (he : Mufu.exponent word ≤ 118) :
    |Exp2Bounds.finiteValue ((Mufu.tanh word).getD 0)-Real.tanh (Exp2Bounds.finiteValue word)| ≤
      |Exp2Bounds.finiteValue word|^3/3 := by
  rw [small_bits word hw he]
  exact signed_error _ (input_magnitude_bound word he)

theorem small_uniform_error (word : Nat) (hw : word < 2^32)
    (he : Mufu.exponent word ≤ 118) :
    |Exp2Bounds.finiteValue ((Mufu.tanh word).getD 0)-Real.tanh (Exp2Bounds.finiteValue word)| ≤
      1/(3*2^24) := by
  have h := small_bits_ideal_error word hw he
  have hp := pow_le_pow_left₀ (abs_nonneg _) (input_magnitude_bound word he) 3
  norm_num at hp ⊢
  linarith

theorem witness_bits : Mufu.tanh 0x3b7fffff = some 0x3b7fffff := by decide +kernel

theorem witness_value : Exp2Bounds.finiteValue 0x3b7fffff = (16777215 : ℝ)/2^32 := by
  norm_num [Exp2Bounds.finiteValue,Mufu.exponent,Mufu.fraction]

theorem witness_error_exceeds_85_ulp :
    (85 : ℝ)/2^32 < |Exp2Bounds.finiteValue ((Mufu.tanh 0x3b7fffff).getD 0)-
      Real.tanh (Exp2Bounds.finiteValue 0x3b7fffff)| := by
  rw [witness_bits,Option.getD_some,witness_value]
  let x : ℝ := 16777215/2^32
  let e := Real.exp (2*x)
  have he0 : 0 < e+1 := by dsimp [e];positivity
  have h := TanhTail.remainder x
  change 1-Real.tanh x = 2/(e+1) at h
  have hh := (eq_div_iff (ne_of_gt he0)).mp h
  have hid : (x-Real.tanh x)*(e+1) = (x-1)*e+x+1 := by nlinarith only [hh]
  have hhi := Real.exp_bound' (show 0 ≤ 2*x by norm_num [x]) (show 2*x ≤ 1 by norm_num [x]) (n:=5) (by decide)
  have hnonneg := (linear_error x (by norm_num [x]) (by norm_num [x])).1
  change 85/2^32 < |x-Real.tanh x|
  rw [abs_of_nonneg hnonneg]
  apply (mul_lt_mul_iff_left₀ he0).mp
  rw [hid]
  change 85/2^32*(Real.exp (2*x)+1) < (x-1)*Real.exp (2*x)+x+1
  norm_num [x,Finset.sum_range_succ] at hhi ⊢
  linarith

#print axioms linear_error
#print axioms signed_error
#print axioms small_bits
#print axioms small_bits_ideal_error
#print axioms small_uniform_error
#print axioms witness_bits
#print axioms witness_value
#print axioms witness_error_exceeds_85_ulp
end GB300Math.TanhLinear

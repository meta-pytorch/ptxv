/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Log2Local
import Log2NearOne
namespace GB300Math.Log2Perturb
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000

theorem quotient (a a' b b' eps : ℝ)
    (he : 0≤eps) (ha : |a-a'|≤eps) (hb : |b-b'|≤eps)
    (ha' : |a'|≤2) (hb0 : 1/2≤b) (hb' : 1/2≤b') :
    |a/b-a'/b'|≤10*eps := by
  have bp : 0<b := by linarith
  have bp' : 0<b' := by linarith
  have hid : a/b-a'/b' = ((a-a')*b'+a'*(b'-b))/(b*b') := by field_simp;ring
  rw [hid,abs_div,abs_mul,abs_of_pos bp,abs_of_pos bp']
  apply (div_le_iff₀ (mul_pos bp bp')).mpr
  have hsum := abs_add_le ((a-a')*b') (a'*(b'-b))
  rw [abs_mul,abs_mul,abs_of_pos bp',abs_sub_comm b' b] at hsum
  have h1 := mul_le_mul_of_nonneg_right ha bp'.le
  have h2 := mul_le_mul ha' hb (abs_nonneg _) (by norm_num : (0:ℝ)≤2)
  have h3 : 0≤eps*(2*b-1)*b' := mul_nonneg (mul_nonneg he (by linarith)) bp'.le
  have h4 : 0≤eps*(2*b'-1) := mul_nonneg he (by linarith)
  nlinarith only [hsum,h1,h2,h3,h4]

theorem log_quotient (x : Rat) (hx : 1≤x ∧ x≤2) :
    |Real.log (x:ℝ)/Real.log 2-(Log2Series.polynomialQ x/Log2Series.polynomialQ 2:Rat)|≤1/1000000000000 := by
  have hx' : 1≤(x:ℝ) ∧ (x:ℝ)≤2 := by
    constructor
    · exact_mod_cast hx.1
    · exact_mod_cast hx.2
  have ha := Log2Series.approximation (x:ℝ) hx'
  have hb := Log2Series.approximation 2 (by norm_num)
  have hlog := Log2NearOne.log_two_interval
  have hlow : 0≤Real.log (x:ℝ) := Real.log_nonneg hx'.1
  have hhigh : Real.log (x:ℝ)≤Real.log 2 := Real.log_le_log (by linarith) hx'.2
  have ha' : |Log2Series.polynomial (x:ℝ)|≤2 := by
    have h := abs_le.mp ha
    apply abs_le.mpr;constructor <;> linarith
  have hbp : 1/2≤Log2Series.polynomial 2 := by have h:=abs_le.mp hb;linarith
  have h := quotient (Real.log (x:ℝ)) (Log2Series.polynomial (x:ℝ)) (Real.log 2)
    (Log2Series.polynomial 2) (1/10^14) (by norm_num) ha hb ha' (by linarith) hbp
  rw [Rat.cast_div,Log2Series.polynomial_cast,Log2Series.polynomial_cast]
  norm_num at h ⊢
  linarith

theorem inverse_log : |(1:ℝ)/Real.log 2-(1/Log2Series.polynomialQ 2:Rat)|≤1/1000000000000 := by
  have hb := Log2Series.approximation 2 (by norm_num)
  have hlog := Log2NearOne.log_two_interval
  have hbp : 1/2≤Log2Series.polynomial 2 := by have h:=abs_le.mp hb;linarith
  have h := quotient 1 1 (Real.log 2) (Log2Series.polynomial 2) (1/10^14)
    (by norm_num) (by norm_num) hb (by norm_num) (by linarith) hbp
  rw [Rat.cast_div, Rat.cast_one, Log2Series.polynomial_cast]
  norm_num at h ⊢
  linarith
#print axioms quotient
#print axioms log_quotient
#print axioms inverse_log
end GB300Math.Log2Perturb

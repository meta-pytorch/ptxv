/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Log2CellData
import Log2Perturb
namespace GB300Math.Log2Centers
open Log2Cells
set_option maxHeartbeats 5000000
set_option maxRecDepth 10000

theorem x_range (j : Fin 512) : 1≤x j.val ∧ x j.val≤2 := by
  have hj := j.isLt
  have h : (j.val:Rat)≤512 := by exact_mod_cast (show j.val≤512 by omega)
  dsimp [x]
  constructor <;> linarith [show (0:Rat)≤j.val by positivity]

theorem center_error (j : Fin 512) :
    |Real.log (x j.val:ℝ)/Real.log 2-(center j.val:ℝ)|≤2/1000000000000 := by
  have h1 := Log2Perturb.log_quotient (x j.val) (x_range j)
  have hc := (cells_valid j).1
  have h2 : |(center j.val:ℝ)-(Log2Series.polynomialQ (x j.val)/l2:Rat)|≤1/1000000000000 := by
    simpa using (Rat.cast_le (K:=ℝ)).mpr hc
  have ht := abs_sub_le (Real.log (x j.val:ℝ)/Real.log 2)
    (Log2Series.polynomialQ (x j.val)/l2:Rat) (center j.val:ℝ)
  rw [abs_sub_comm] at h2
  dsimp [l2] at h2 ht
  linarith

theorem inverse_scaled (n : Rat) (hn : |n|≤1) :
    |(n:ℝ)/Real.log 2-(n/l2:Rat)|≤1/1000000000000 := by
  have h := Log2Perturb.inverse_log
  have hn' : |(n:ℝ)|≤1 := by simpa using (Rat.cast_le (K:=ℝ)).mpr hn
  have he : (n:ℝ)/Real.log 2-(n/l2:Rat) = (n:ℝ)*((1:ℝ)/Real.log 2-(1/l2:Rat)) := by
    push_cast
    ring
  rw [he,abs_mul]
  have h' := mul_le_mul hn' h (abs_nonneg _) (by norm_num : (0:ℝ)≤1)
  dsimp [l2]
  simpa using h'

theorem linear_error (j : Fin 512) :
    |1/(512*(x j.val:ℝ)*Real.log 2)-(linear j.val:ℝ)|≤2/1000000000000 := by
  have hx := x_range j
  have hxp : (0:Rat)<x j.val := by linarith [hx.1]
  have hxn : x j.val≠0 := ne_of_gt hxp
  have hn : |(1:Rat)/(512*x j.val)|≤1 := by
    rw [abs_of_nonneg (by positivity)]
    apply (div_le_iff₀ (by positivity : (0:Rat)<512*x j.val)).mpr
    linarith
  have h1 := inverse_scaled (1/(512*x j.val)) hn
  have he : ((1/(512*x j.val):Rat):ℝ)/Real.log 2=1/(512*(x j.val:ℝ)*Real.log 2) := by push_cast;ring
  have he' : (1/(512*x j.val):Rat)/l2=1/(512*x j.val*l2) := by ring
  rw [he,he'] at h1
  have h2 : |(linear j.val:ℝ)-(1/(512*x j.val*l2):Rat)|≤1/1000000000000 := by
    simpa using (Rat.cast_le (K:=ℝ)).mpr (cells_valid j).2.1
  have ht := abs_sub_le (1/(512*(x j.val:ℝ)*Real.log 2)) (1/(512*x j.val*l2):Rat) (linear j.val:ℝ)
  rw [abs_sub_comm] at h2
  linarith

theorem quadratic_error (j : Fin 512) :
    |(-1)/(2*512^2*(x j.val:ℝ)^2*Real.log 2)-(quadratic j.val:ℝ)|≤2/1000000000000 := by
  have hx := x_range j
  have hxp : (0:Rat)<x j.val := by linarith [hx.1]
  have hxn : x j.val≠0 := ne_of_gt hxp
  have hx2 : (1:Rat)≤(x j.val)^2 := one_le_pow₀ hx.1
  have hn : |(-1:Rat)/(2*512^2*(x j.val)^2)|≤1 := by
    rw [abs_div,abs_of_pos (by positivity : (0:Rat)<2*512^2*(x j.val)^2)]
    rw [show |(-1:Rat)|=1 by norm_num]
    apply (div_le_iff₀ (by positivity : (0:Rat)<2*512^2*(x j.val)^2)).mpr
    linarith
  have h1 := inverse_scaled (-1/(2*512^2*(x j.val)^2)) hn
  have he : ((-1/(2*512^2*(x j.val)^2):Rat):ℝ)/Real.log 2=(-1)/(2*512^2*(x j.val:ℝ)^2*Real.log 2) := by push_cast;ring
  have he' : (-1/(2*512^2*(x j.val)^2):Rat)/l2=(-1)/(2*512^2*(x j.val)^2*l2) := by ring
  rw [he,he'] at h1
  have h2 : |(quadratic j.val:ℝ)-((-1)/(2*512^2*(x j.val)^2*l2):Rat)|≤1/1000000000000 := by
    simpa using (Rat.cast_le (K:=ℝ)).mpr (cells_valid j).2.2.1
  have ht := abs_sub_le ((-1)/(2*512^2*(x j.val:ℝ)^2*Real.log 2)) (((-1)/(2*512^2*(x j.val)^2*l2):Rat):ℝ) (quadratic j.val:ℝ)
  rw [abs_sub_comm] at h2
  linarith
#print axioms center_error
#print axioms linear_error
#print axioms quadratic_error
end GB300Math.Log2Centers

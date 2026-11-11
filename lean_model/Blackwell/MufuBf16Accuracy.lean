/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Exp2Bounds
import GB300Math.MufuSm103

namespace GB300Math.MufuSm103Accuracy

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

noncomputable def value (half : Nat) : ℝ :=
  Exp2Bounds.finiteValue (half % 65536 * 65536)

def output (input : Nat) : Nat :=
  (MufuSm103.evaluateBf16 "ex2" input).getD 0

theorem adjacent_inputs :
    value 0xb3ff = -(255/2^31 : ℝ) ∧ value 0xb400 = -(1/2^23 : ℝ) := by
  norm_num [value, Exp2Bounds.finiteValue, Mufu.exponent, Mufu.fraction]

theorem adjacent_outputs :
    output 0xb3ff = 0x3f80 ∧ output 0xb400 = 0x3f7f := by
  decide +kernel

theorem output_values : value 0x3f80 = 1 ∧ value 0x3f7f = 1-1/2^8 := by
  norm_num [value, Exp2Bounds.finiteValue, Mufu.exponent, Mufu.fraction]

theorem output_step :
    value (output 0xb3ff) - value (output 0xb400) = 1/2^8 := by
  rw [adjacent_outputs.1, adjacent_outputs.2, output_values.1, output_values.2]
  ring

theorem near_one_enclosure :
    1-3/(4*2^23) ≤ (2 : ℝ)^(value 0xb400) ∧ (2 : ℝ)^(value 0xb400) ≤ 1 := by
  rw [adjacent_inputs.2]
  have h := Exp2Bounds.real_bound (-(1/2^23 : ℝ)) (by norm_num)
  constructor
  · have hl := (abs_le.mp h).1
    linarith
  · rw [Real.rpow_def_of_pos (by norm_num)]
    apply Real.exp_le_one_iff.mpr
    exact mul_nonpos_of_nonneg_of_nonpos Exp2Bounds.log_two_bound.1 (by norm_num)

theorem nearly_one_ulp_error :
    1/2^8-3/(4*2^23) ≤
      |value (output 0xb400) - (2 : ℝ)^(value 0xb400)| ∧
    |value (output 0xb400) - (2 : ℝ)^(value 0xb400)| ≤ 1/2^8 := by
  rw [adjacent_outputs.2, output_values.2]
  have h := near_one_enclosure
  have hd : (1 : ℝ)-1/2^8 ≤ 2^(value 0xb400) := by linarith [h.1]
  rw [abs_of_nonpos (sub_nonpos.mpr hd)]
  constructor <;> linarith [h.1, h.2]

theorem nearer_to_one_than_result :
    |value 0x3f80 - (2 : ℝ)^(value 0xb400)| <
      |value (output 0xb400) - (2 : ℝ)^(value 0xb400)| := by
  rw [output_values.1]
  have h := Exp2Bounds.real_bound (-(1/2^23 : ℝ)) (by norm_num)
  rw [← adjacent_inputs.2, abs_sub_comm] at h
  have hl := nearly_one_ulp_error.1
  linarith

theorem perturbation_gain :
    (value (output 0xb3ff) - value (output 0xb400)) /
      (value 0xb3ff - value 0xb400) = 8388608 := by
  rw [output_step, adjacent_inputs.1, adjacent_inputs.2]
  norm_num

#print axioms adjacent_inputs
#print axioms adjacent_outputs
#print axioms output_values
#print axioms output_step
#print axioms near_one_enclosure
#print axioms nearly_one_ulp_error
#print axioms nearer_to_one_than_result
#print axioms perturbation_gain

end GB300Math.MufuSm103Accuracy

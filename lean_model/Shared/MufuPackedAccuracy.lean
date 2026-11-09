import Exp2Bounds
import GB300Math.MufuPacked

namespace GB300Math.MufuPackedAccuracy

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

noncomputable def value (half : Nat) : ℝ :=
  Exp2Bounds.finiteValue (Mufu.widen16 half)

theorem adjacent_inputs :
    value 0x8001 = -(1/2^24 : ℝ) ∧ value 0x8002 = -(1/2^23 : ℝ) := by
  have h1 : Mufu.widen16 0x8001 = 0xb3800000 := by decide +kernel
  have h2 : Mufu.widen16 0x8002 = 0xb4000000 := by decide +kernel
  simp only [value, h1, h2]
  norm_num [Exp2Bounds.finiteValue, Mufu.exponent, Mufu.fraction]

theorem adjacent_outputs :
    MufuPacked.half 0 0x8001 = 0x3c00 ∧ MufuPacked.half 0 0x8002 = 0x3bff := by
  decide +kernel

theorem output_values : value 0x3c00 = 1 ∧ value 0x3bff = 1-1/2^11 := by
  norm_num [value, Mufu.widen16, Exp2Bounds.finiteValue, Mufu.exponent, Mufu.fraction]

theorem output_step :
    value (MufuPacked.half 0 0x8001) - value (MufuPacked.half 0 0x8002) = 1/2^11 := by
  rw [adjacent_outputs.1, adjacent_outputs.2, output_values.1, output_values.2]
  ring

theorem near_one_enclosure :
    1-3/(4*2^23) ≤ (2 : ℝ)^(value 0x8002) ∧ (2 : ℝ)^(value 0x8002) ≤ 1 := by
  rw [adjacent_inputs.2]
  have h := Exp2Bounds.real_bound (-(1/2^23 : ℝ)) (by norm_num)
  constructor
  · have hl := (abs_le.mp h).1
    linarith
  · rw [Real.rpow_def_of_pos (by norm_num)]
    apply Real.exp_le_one_iff.mpr
    exact mul_nonpos_of_nonneg_of_nonpos Exp2Bounds.log_two_bound.1 (by norm_num)

theorem nearly_one_ulp_error :
    1/2^11-3/(4*2^23) ≤
      |value (MufuPacked.half 0 0x8002) - (2 : ℝ)^(value 0x8002)| ∧
    |value (MufuPacked.half 0 0x8002) - (2 : ℝ)^(value 0x8002)| ≤ 1/2^11 := by
  rw [adjacent_outputs.2, output_values.2]
  have h := near_one_enclosure
  have hd : (1 : ℝ)-1/2^11 ≤ 2^(value 0x8002) := by linarith [h.1]
  rw [abs_of_nonpos (sub_nonpos.mpr hd)]
  constructor <;> linarith [h.1, h.2]

theorem nearer_to_one_than_result :
    |value 0x3c00 - (2 : ℝ)^(value 0x8002)| <
      |value (MufuPacked.half 0 0x8002) - (2 : ℝ)^(value 0x8002)| := by
  rw [output_values.1]
  have h := Exp2Bounds.real_bound (-(1/2^23 : ℝ)) (by norm_num)
  rw [← adjacent_inputs.2, abs_sub_comm] at h
  have hl := nearly_one_ulp_error.1
  linarith

theorem perturbation_gain :
    (value (MufuPacked.half 0 0x8001) - value (MufuPacked.half 0 0x8002)) /
      (value 0x8001 - value 0x8002) = 8192 := by
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

end GB300Math.MufuPackedAccuracy

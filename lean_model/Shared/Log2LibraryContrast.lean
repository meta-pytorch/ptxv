import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Exp2Bounds
import CudaLog2

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

namespace GB300Math.Log2LibraryContrast

theorem log_two_bound : |Real.log 2-(69314718056:ℝ)/100000000000| ≤ 1/10000000000 := by
  have h := Real.sum_range_sub_log_div_le (x := (1:ℝ)/3) (by norm_num) 11
  norm_num [Finset.sum_range_succ] at h ⊢
  rcases abs_le.mp h with ⟨hl,hu⟩
  exact abs_le.mpr ⟨by linarith,by linarith⟩

theorem input_value : Exp2Bounds.finiteValue 0x3f800001 = 1+(1:ℝ)/2^23 := by
  norm_num [Exp2Bounds.finiteValue,Mufu.exponent,Mufu.fraction]

theorem output_value : Exp2Bounds.finiteValue (CudaLog2.model 0x3f800001).toNat = (12102202:ℝ)/2^46 := by
  rw [CudaLog2.at_next_one]
  norm_num [Exp2Bounds.finiteValue,Mufu.exponent,Mufu.fraction]

theorem relative_error :
    |Exp2Bounds.finiteValue (CudaLog2.model 0x3f800001).toNat -
      Real.log (Exp2Bounds.finiteValue 0x3f800001)/Real.log 2| /
      (Real.log (Exp2Bounds.finiteValue 0x3f800001)/Real.log 2) < 1/(25000000:ℝ) := by
  rw [input_value,output_value]
  let t : ℝ := 1/2^23
  let p : ℝ := t-t^2/2+t^3/3-t^4/4
  have h := Real.abs_log_sub_add_sum_range_le (x := -t)
    (by norm_num [t]) 4
  have he : (∑ i ∈ Finset.range 4, (-t)^(i+1)/(i+1)) + Real.log (1-(-t)) = Real.log (1+t)-p := by
    norm_num [Finset.sum_range_succ,p];ring
  rw [he] at h
  norm_num [t] at h
  have hl2 := log_two_bound
  norm_num at hl2
  rcases abs_le.mp h with ⟨hlo,hhi⟩
  rcases abs_le.mp hl2 with ⟨hl2lo,hl2hi⟩
  have hlog : 0 < Real.log (1+(1:ℝ)/2^23) := Real.log_pos (by norm_num)
  have htwo : 0 < Real.log (2:ℝ) := Real.log_pos (by norm_num)
  have hq : 0 < Real.log (1+(1:ℝ)/2^23)/Real.log 2 := div_pos hlog htwo
  have hb : |(12102202:ℝ)/2^46 * Real.log 2 - Real.log (1+(1:ℝ)/2^23)| <
      1/(25000000:ℝ)*Real.log (1+(1:ℝ)/2^23) := by
    apply abs_lt.mpr
    norm_num [p,t] at hlo hhi ⊢
    constructor <;> nlinarith only [hlo,hhi,hl2lo,hl2hi]
  apply (div_lt_iff₀ hq).mpr
  apply (mul_lt_mul_iff_left₀ htwo).mp
  have habs : |(12102202:ℝ)/2^46-Real.log (1+(1:ℝ)/2^23)/Real.log 2| * Real.log 2 =
      |(12102202:ℝ)/2^46*Real.log 2-Real.log (1+(1:ℝ)/2^23)| := by
    have h := abs_mul ((12102202:ℝ)/2^46-Real.log (1+(1:ℝ)/2^23)/Real.log 2) (Real.log 2)
    rw [abs_of_pos htwo] at h
    rw [← h]
    congr 1
    field_simp
  rw [habs]
  convert hb using 1
  field_simp

theorem half_ulp_error :
    |Exp2Bounds.finiteValue (CudaLog2.model 0x3f800001).toNat -
      Real.log (Exp2Bounds.finiteValue 0x3f800001)/Real.log 2| < 1/(2^47:ℝ) := by
  have hr := relative_error
  rw [input_value] at hr ⊢
  have hlog := Real.log_le_sub_one_of_pos (show 0 < (1+(1:ℝ)/2^23) by norm_num)
  have htwo := log_two_bound
  rcases abs_le.mp htwo with ⟨hlo,hhi⟩
  have hp : 0 < Real.log (1+(1:ℝ)/2^23)/Real.log 2 :=
    div_pos (Real.log_pos (by norm_num)) (Real.log_pos (by norm_num))
  have hupper : Real.log (1+(1:ℝ)/2^23)/Real.log 2 < 7/40000000 := by
    apply (div_lt_iff₀ (Real.log_pos (by norm_num : (1:ℝ)<2))).mpr
    norm_num at hlog hlo hhi ⊢
    nlinarith only [hlog,hlo,hhi]
  have hb := (div_lt_iff₀ hp).mp hr
  have hh := mul_lt_mul_of_pos_left hupper (show (0:ℝ)<1/25000000 by norm_num)
  norm_num at hb hh ⊢
  linarith

#print axioms log_two_bound
#print axioms input_value
#print axioms output_value
#print axioms relative_error
#print axioms half_ulp_error
end GB300Math.Log2LibraryContrast

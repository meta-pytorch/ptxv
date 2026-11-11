/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import RsqPtxValues
import RcpPtxScale
import RsqPtxWrapper

set_option maxRecDepth 1000000
set_option maxHeartbeats 10000000
namespace GB300Math.RsqPtxBounds
open SqrtBounds (positiveNormalValue)

theorem subnormal_adapter_relative_error (f : Nat) (hf : 0 < f ∧ f < 2^23) :
    ∃ output, Approx.evaluate "rsq" false f = some output ∧ 2^23 ≤ output ∧
      |positiveNormalValue output-1/Real.sqrt ((f : ℝ)*(2 : ℝ)^(-149 : Int))| ≤
        ((17 : ℝ)/2^27)/Real.sqrt ((f : ℝ)*(2 : ℝ)^(-149 : Int)) := by
  let e := f.log2+2
  let m := f*2^(23-f.log2)-2^23
  have hlog : f.log2 ≤ 22 := by
    have h := (Nat.log2_lt (by omega : f ≠ 0)).mpr hf.2
    omega
  have hm := RcpPtxBounds.normalized_mantissa_range f (by omega) hf.2
  have hem : 1 ≤ e ∧ e ≤ 254 := by dsimp [e];omega
  have hmf : m < 2^23 := by dsimp [m];omega
  have hpromote : Approx.mul f 0x4b800000 = e*2^23+m := by
    simpa [e,m] using RcpPtxBounds.mul_subnormal_normalize_word false f (by omega) hf.2
  let z := Mufu.rsq (e*2^23+m)
  let ze := z/2^23
  let zf := z%2^23
  have hz := output_normal_range e m hem hmf
  change 63*2^23 ≤ z ∧ z < 191*2^23 at hz
  have hze : 63 ≤ ze ∧ ze ≤ 190 := by dsimp [ze];omega
  have hzf : zf < 2^23 := by dsimp [zf];omega
  have hzm : ze*2^23+zf = z := by dsimp [ze,zf];omega
  have hmul : Approx.mul z 0x45800000 = (ze+12)*2^23+zf := by
    have h := RcpPtxBounds.mul_power_two_normal_word false ze zf 139 (by omega) hzf (by norm_num) (by omega)
    have hsum : ze+139-127 = ze+12 := by omega
    simpa only [Bool.false_eq_true,↓reduceIte,Nat.zero_add,hsum,hzm,show 139*2^23 = 0x45800000 by norm_num] using h
  have ht : Mufu.magnitude f < 0x00800000 := by dsimp [Mufu.magnitude];omega
  have hrun : Approx.evaluate "rsq" false f = some ((ze+12)*2^23+zf) := by
    simp [Approx.evaluate,ht]
    rw [hpromote]
    exact hmul
  have hinput : positiveNormalValue (e*2^23+m) = ((f : ℝ)*(2 : ℝ)^(-149 : Int))*2^24 := by
    rw [subnormal_scaled_value f hf]
    norm_num
    ring
  have houtput : positiveNormalValue ((ze+12)*2^23+zf) = positiveNormalValue z*2^12 := by
    rw [shift_output_value ze zf (by omega) hzf,hzm]
  have hb := rsq_bits_relative_error e m hem hmf
  rw [hinput] at hb
  have hx : (0 : ℝ) < (f : ℝ)*(2 : ℝ)^(-149 : Int) := mul_pos (Nat.cast_pos.mpr hf.1) (zpow_pos (by norm_num) _)
  have hb' := relative_rescale ((f : ℝ)*(2 : ℝ)^(-149 : Int)) (positiveNormalValue z)
    ((17 : ℝ)/2^27) hx hb
  refine ⟨_,hrun,by omega,?_⟩
  rw [houtput]
  exact hb'

noncomputable def positiveFiniteValue (word : Nat) : ℝ :=
  if word < 2^23 then (word : ℝ)*(2 : ℝ)^(-149 : Int)
  else positiveNormalValue word

theorem all_positive_adapter_relative_error (word : Nat)
    (hw : 0 < word ∧ word < 0x7f800000) :
    ∃ output, Approx.evaluate "rsq" false word = some output ∧
      |positiveFiniteValue output-1/Real.sqrt (positiveFiniteValue word)| ≤
        ((17 : ℝ)/2^27)/Real.sqrt (positiveFiniteValue word) := by
  by_cases ht : word < 2^23
  · obtain ⟨output,hrun,ho,hb⟩ := subnormal_adapter_relative_error word ⟨hw.1,ht⟩
    refine ⟨output,hrun,?_⟩
    simpa only [positiveFiniteValue,ite_eq_left ht,ite_eq_right (not_lt.mpr ho)] using hb
  · let e := word/2^23
    let f := word%2^23
    have he : 1 ≤ e ∧ e ≤ 254 := by dsimp [e];omega
    have hf : f < 2^23 := by dsimp [f];omega
    have hword : e*2^23+f = word := by dsimp [e,f];omega
    have hrun := normal_adapter false e f he hf
    have hb := rsq_bits_relative_error e f he hf
    have ho := output_normal_range e f he hf
    rw [hword] at hrun hb ho
    refine ⟨_,hrun,?_⟩
    have hot : ¬Mufu.rsq word < 2^23 := by omega
    simpa only [positiveFiniteValue,ite_eq_right ht,ite_eq_right hot] using hb

theorem subnormal_adapter_ptx_relative_error (f : Nat) (hf : 0 < f ∧ f < 2^23) :
    ∃ output, Approx.evaluate "rsq" false f = some output ∧ 2^23 ≤ output ∧
      |positiveNormalValue output-1/Real.sqrt ((f : ℝ)*(2 : ℝ)^(-149 : Int))| ≤
        ((2 : ℝ)^(-(229 : ℝ)/10))/Real.sqrt ((f : ℝ)*(2 : ℝ)^(-149 : Int)) := by
  obtain ⟨output,hrun,ho,hb⟩ := subnormal_adapter_relative_error f hf
  refine ⟨output,hrun,ho,hb.trans ?_⟩
  exact div_le_div_of_nonneg_right rational_bound_stronger_than_ptx.le (Real.sqrt_nonneg _)

theorem all_positive_adapter_ptx_relative_error (word : Nat)
    (hw : 0 < word ∧ word < 0x7f800000) :
    ∃ output, Approx.evaluate "rsq" false word = some output ∧
      |positiveFiniteValue output-1/Real.sqrt (positiveFiniteValue word)| ≤
        ((2 : ℝ)^(-(229 : ℝ)/10))/Real.sqrt (positiveFiniteValue word) := by
  obtain ⟨output,hrun,hb⟩ := all_positive_adapter_relative_error word hw
  refine ⟨output,hrun,hb.trans ?_⟩
  exact div_le_div_of_nonneg_right rational_bound_stronger_than_ptx.le (Real.sqrt_nonneg _)

#print axioms subnormal_adapter_relative_error
#print axioms all_positive_adapter_relative_error
#print axioms subnormal_adapter_ptx_relative_error
#print axioms all_positive_adapter_ptx_relative_error
end GB300Math.RsqPtxBounds

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import RcpPtxAdapter
import RcpPtxScale

set_option maxRecDepth 1000000
set_option maxHeartbeats 10000000
namespace GB300Math.RcpPtxBounds
open RcpBounds

theorem nonzero_fraction_below_one (g : Nat) (hg : 0 < g ∧ g < 2^23) :
    (Mufu.interpolate MufuTables.rcp g).toNat < 2^24 := by
  have h := interpolate_error g hg.2
  have hgp : (0 : Int) < g := by exact_mod_cast hg.1
  have hglt : (g : Int) < 2^23 := by exact_mod_cast hg.2
  have hz : Mufu.interpolate MufuTables.rcp g < 2^24 := by
    by_contra hz
    have hp := mul_le_mul_of_nonneg_left (le_of_not_gt hz) (show (0 : Int) ≤ 2^23+g by omega)
    dsimp only at h
    norm_num at h hp
    nlinarith only [hp,h.2,hgp,hglt]
  omega

theorem subnormal_normalized_value (f : Nat) (hf : 0 < f ∧ f < 2^23) :
    positiveNormalValue ((f.log2+2)*2^23+(f*2^(23-f.log2)-2^23)) =
      (f : ℚ)*(2 : ℚ)^(-125 : Int) := by
  have hl : f.log2 ≤ 22 := by
    have h := (Nat.log2_lt (by omega : f ≠ 0)).mpr hf.2
    omega
  have hm := normalized_mantissa_range f (by omega) hf.2
  rw [normal_decode _ _ (by omega) (by omega)]
  have hc : ((f*2^(23-f.log2)-2^23 : Nat) : ℚ) = (f : ℚ)*2^(23-f.log2)-2^23 := by
    rw [Nat.cast_sub hm.1]
    push_cast
    rfl
  rw [hc]
  push_cast
  rw [show (2^23+((f : ℚ)*2^(23-f.log2)-2^23)) = (f : ℚ)*2^(23-f.log2) by ring]
  rw [mul_assoc,← zpow_natCast,← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
  congr 2
  omega

theorem output_shift_24 (e f : Nat) (he : 1 ≤ e ∧ e ≤ 230) (hf : f < 2^23) :
    positiveNormalValue ((e+24)*2^23+f) = positiveNormalValue (e*2^23+f)*2^24 := by
  rw [normal_decode _ _ (by omega) hf,normal_decode _ _ (by omega) hf]
  push_cast
  have hp : (2 : ℚ)^((e : Int)+24-150) = (2 : ℚ)^((e : Int)-150)*2^24 := by
    rw [show (2 : ℚ)^24 = (2 : ℚ)^(24 : Int) by norm_num,← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
    congr 1
    omega
  rw [hp]
  ring

theorem finite_subnormal_fields (f : Nat) (hf : 2^21 < f ∧ f < 2^23) :
    let e := f.log2+2
    let g := f*2^(23-f.log2)-2^23
    23 ≤ e ∧ e ≤ 24 ∧ g < 2^23 ∧
      229*2^23 ≤ Mufu.rcp (e*2^23+g) ∧ Mufu.rcp (e*2^23+g) < 231*2^23 := by
  have hf0 : f ≠ 0 := by omega
  have hlt := (Nat.log2_lt hf0).mpr hf.2
  have hge := (Nat.le_log2 hf0).mpr (show 2^21 ≤ f by omega)
  have hm := normalized_mantissa_range f hf0 hf.2
  dsimp only
  refine ⟨by omega,by omega,by omega,?_⟩
  rw [output_word_scaled (f.log2+2) (f*2^(23-f.log2)-2^23) (by omega) (by omega)]
  by_cases ht : f.log2 = 21
  · have hg : 0 < f*2^(23-f.log2)-2^23 := by rw [ht];norm_num;omega
    have hz := nonzero_fraction_below_one _ ⟨hg,by omega⟩
    rw [ite_eq_right (show (Mufu.interpolate MufuTables.rcp (f*2^(23-f.log2)-2^23)).toNat ≠ 2^24 by omega)]
    omega
  · have ht' : f.log2 = 22 := by omega
    split <;> omega

theorem finite_subnormal_adapter_absolute_error (f : Nat)
    (hf : 2^21 < f ∧ f < 2^23) :
    ∃ output, Approx.evaluate "rcp" false f = some output ∧
      2^23 ≤ output ∧ output < 0x7f800000 ∧
      |positiveNormalValue output-1/((f : ℚ)*(2 : ℚ)^(-149 : Int))| ≤
        (2 : ℚ)^(125-(f.log2 : Int)) := by
  let e := f.log2+2
  let g := f*2^(23-f.log2)-2^23
  have hfields := finite_subnormal_fields f hf
  change 23 ≤ e ∧ e ≤ 24 ∧ g < 2^23 ∧ _ at hfields
  have he : 1 ≤ e ∧ e ≤ 252 := by omega
  have hg : g < 2^23 := hfields.2.2.1
  have hpromote : Approx.mul f 0x4b800000 = e*2^23+g := by
    simpa [e,g] using mul_subnormal_normalize_word false f (by omega) hf.2
  let z := Mufu.rcp (e*2^23+g)
  let ze := z/2^23
  let zf := z%2^23
  have hz : 229*2^23 ≤ z ∧ z < 231*2^23 := hfields.2.2.2
  have hze : 229 ≤ ze ∧ ze ≤ 230 := by dsimp [ze];omega
  have hzf : zf < 2^23 := by dsimp [zf];omega
  have hzm : ze*2^23+zf = z := by dsimp [ze,zf];omega
  have hmul : Approx.mul 0x4b800000 z = (ze+24)*2^23+zf := by
    have h := mul_power_two_normal_word_left false ze zf 151 (by omega) hzf (by norm_num) (by omega)
    have hsum : ze+151-127 = ze+24 := by omega
    simpa only [Bool.false_eq_true,↓reduceIte,Nat.zero_add,hsum,hzm,show 151*2^23 = 0x4b800000 by norm_num] using h
  have ht : Mufu.magnitude f < 0x00800000 := by dsimp [Mufu.magnitude];omega
  have hl : ¬Mufu.magnitude f > 0x7e800000 := by omega
  have hrun : Approx.evaluate "rcp" false f = some ((ze+24)*2^23+zf) := by
    simp [Approx.evaluate,ht,hl]
    rw [hpromote]
    exact hmul
  have hinput : positiveNormalValue (e*2^23+g) = ((f : ℚ)*(2 : ℚ)^(-149 : Int))*2^24 := by
    rw [subnormal_normalized_value f ⟨by omega,hf.2⟩]
    norm_num
    ring
  have houtput : positiveNormalValue ((ze+24)*2^23+zf) = positiveNormalValue z*2^24 := by
    rw [output_shift_24 ze zf (by omega) hzf,hzm]
  have hb := rcp_bits_scaled_absolute_error e g he hg
  rw [hinput] at hb
  have hm := mul_le_mul_of_nonneg_right hb (show (0 : ℚ) ≤ 2^24 by norm_num)
  have ha : |(positiveNormalValue z-1/(((f : ℚ)*2^(-149 : Int))*2^24))*2^24| =
      |positiveNormalValue z-1/(((f : ℚ)*2^(-149 : Int))*2^24)| * 2^24 := by
    rw [abs_mul,abs_of_pos (by norm_num : (0 : ℚ) < 2^24)]
  rw [← ha] at hm
  have hid : (positiveNormalValue z-1/(((f : ℚ)*2^(-149 : Int))*2^24))*2^24 =
      positiveNormalValue z*2^24-1/((f : ℚ)*2^(-149 : Int)) := by ring
  have hbound : ((1 : ℚ)/2^24)/exponentScale e*2^24 = (2 : ℚ)^(125-(f.log2 : Int)) := by
    rw [show ((1 : ℚ)/2^24)/exponentScale e*2^24 = 1/exponentScale e by ring]
    dsimp [exponentScale,e]
    rw [one_div,← zpow_neg]
    exact congrArg (fun k : Int => (2 : ℚ)^k) (by omega)
  rw [hid,hbound] at hm
  refine ⟨_,hrun,by omega,by omega,?_⟩
  rw [houtput]
  exact hm


theorem finite_subnormal_adapter_sign (f : Nat) (hf : 2^21 < f ∧ f < 2^23) :
    Approx.evaluate "rcp" false (2^31+f) =
      (Approx.evaluate "rcp" false f).map (fun output => 2^31+output) := by
  let e := f.log2+2
  let g := f*2^(23-f.log2)-2^23
  let x := e*2^23+g
  have hfields := finite_subnormal_fields f hf
  change 23 ≤ e ∧ e ≤ 24 ∧ g < 2^23 ∧ _ at hfields
  have hpromote : Approx.mul f 0x4b800000 = x := by
    simpa [e,g,x] using mul_subnormal_normalize_word false f (by omega) hf.2
  have hpromoten : Approx.mul (2^31+f) 0x4b800000 = 2^31+x := by
    simpa [e,g,x,Nat.add_assoc] using mul_subnormal_normalize_word true f (by omega) hf.2
  let z := Mufu.rcp x
  let ze := z/2^23
  let zf := z%2^23
  have hz : 229*2^23 ≤ z ∧ z < 231*2^23 := hfields.2.2.2
  have hze : 229 ≤ ze ∧ ze ≤ 230 := by dsimp [ze];omega
  have hzf : zf < 2^23 := by dsimp [zf];omega
  have hzm : ze*2^23+zf = z := by dsimp [ze,zf];omega
  have hsum : ze+151-127 = ze+24 := by omega
  have hmul (negative : Bool) : Approx.mul 0x4b800000 ((if negative then 2^31 else 0)+z) =
      (if negative then 2^31 else 0)+(ze+24)*2^23+zf := by
    have h := mul_power_two_normal_word_left negative ze zf 151 (by omega) hzf (by norm_num) (by omega)
    simpa only [hsum,Nat.add_assoc,hzm,show 151*2^23 = 0x4b800000 by norm_num] using h
  have ht : Mufu.magnitude f < 0x00800000 := by dsimp [Mufu.magnitude];omega
  have htn : Mufu.magnitude (2^31+f) < 0x00800000 := by dsimp [Mufu.magnitude];omega
  have hl : ¬Mufu.magnitude f > 0x7e800000 := by omega
  have hln : ¬Mufu.magnitude (2^31+f) > 0x7e800000 := by omega
  simp only [Approx.evaluate,Bool.false_eq_true,↓reduceIte,ht,htn,hl,hln]
  rw [hpromote,hpromoten,reciprocal_sign x (by dsimp [x];omega)]
  change some (Approx.mul 0x4b800000 (2^31+z)) =
    (some (Approx.mul 0x4b800000 z)).map (fun output => 2^31+output)
  rw [show Approx.mul 0x4b800000 (2^31+z) = 2^31+(ze+24)*2^23+zf by simpa using hmul true,
      show Approx.mul 0x4b800000 z = (ze+24)*2^23+zf by simpa using hmul false]
  simp [Nat.add_assoc]

theorem finite_subnormal_signed_absolute_error (negative : Bool) (f : Nat)
    (hf : 2^21 < f ∧ f < 2^23) :
    ∃ magnitude, Approx.evaluate "rcp" false ((if negative then 2^31 else 0)+f) =
      some ((if negative then 2^31 else 0)+magnitude) ∧
      2^23 ≤ magnitude ∧ magnitude < 0x7f800000 ∧
      |normalValue ((if negative then 2^31 else 0)+magnitude)-
        1/(((if negative then -1 else 1) : ℚ)*(f : ℚ)*(2 : ℚ)^(-149 : Int))| ≤
        (2 : ℚ)^(125-(f.log2 : Int)) := by
  obtain ⟨output,ho,hlo,hhi,herr⟩ := finite_subnormal_adapter_absolute_error f hf
  have hv := value_sign output (show output < 2^31 by omega)
  refine ⟨output,?_,hlo,hhi,?_⟩
  · cases negative
    · simpa using ho
    · simpa [ho] using finite_subnormal_adapter_sign f hf
  · cases negative
    · simpa only [Bool.false_eq_true,↓reduceIte,Nat.zero_add,one_mul,hv.1] using herr
    · simp only [↓reduceIte,hv.2,neg_mul,one_mul,one_div,inv_neg,sub_neg_eq_add]
      rw [show -positiveNormalValue output+((f : ℚ)*2^(-149 : Int))⁻¹ =
        -(positiveNormalValue output-1/((f : ℚ)*2^(-149 : Int))) by ring,abs_neg]
      exact herr

#print axioms nonzero_fraction_below_one
#print axioms subnormal_normalized_value
#print axioms output_shift_24
#print axioms finite_subnormal_fields
#print axioms finite_subnormal_adapter_absolute_error
#print axioms finite_subnormal_adapter_sign
#print axioms finite_subnormal_signed_absolute_error
end GB300Math.RcpPtxBounds

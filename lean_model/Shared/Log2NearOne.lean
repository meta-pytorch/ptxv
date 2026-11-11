/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Root32FloatLib
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
namespace GB300Math.Log2NearOne
open FloatLib.Floats.Formats.BinaryInterchange
set_option maxRecDepth 100000
set_option maxHeartbeats 3000000

theorem above_bits : Mufu.lg2 0x3f800001=0x3495f600 := by decide +kernel
theorem below_bits : Mufu.lg2 0x3f7fffff=0xb24b4000 := by decide +kernel

theorem above_decode :
    Root32FloatLib.value 0x3f800001=some (8388609/8388608) ∧
    Root32FloatLib.value (Mufu.lg2 0x3f800001)=some (19195/2^36) := by
  rw [above_bits]
  constructor
  · rw [Root32FloatLib.decode_normal _ (by norm_num)]
    norm_num [RcpBounds.positiveNormalValue,Mufu.exponent,Mufu.fraction]
  · rw [Root32FloatLib.decode_normal _ (by norm_num)]
    norm_num [RcpBounds.positiveNormalValue,Mufu.exponent,Mufu.fraction]

theorem below_decode :
    Root32FloatLib.value 0x3f7fffff=some (16777215/16777216) ∧
    Root32FloatLib.value (Mufu.lg2 0x3f7fffff)=some (-(813/2^36)) := by
  rw [below_bits]
  constructor
  · rw [Root32FloatLib.decode_normal _ (by norm_num)]
    norm_num [RcpBounds.positiveNormalValue,Mufu.exponent,Mufu.fraction]
  · have h : (Model.ofNatBits 0x324b4000 : Model FloatFormat.binary32).toRat?=some (813/2^36) := by
      change Root32FloatLib.value _=_
      rw [Root32FloatLib.decode_normal _ (by norm_num)]
      norm_num [RcpBounds.positiveNormalValue,Mufu.exponent,Mufu.fraction]
    exact RcpSigned.signed_decode _ _ h

theorem log_two_interval : (69:ℝ)/100≤Real.log 2 ∧ Real.log 2≤7/10 := by
  constructor
  · apply (Real.le_log_iff_exp_le (by norm_num : (0:ℝ)<2)).mpr
    have h := Real.exp_bound' (by norm_num : (0:ℝ)≤69/100) (by norm_num : (69:ℝ)/100≤1) (n:=8) (by decide)
    norm_num [Finset.sum_range_succ] at h
    linarith
  · apply (Real.log_le_iff_le_exp (by norm_num : (0:ℝ)<2)).mpr
    have h := Real.sum_le_exp_of_nonneg (by norm_num : (0:ℝ)≤7/10) 4
    norm_num [Finset.sum_range_succ] at h
    linarith

theorem above_ideal_interval :
    0<Real.log ((8388609:ℝ)/8388608)/Real.log 2 ∧
    Real.log ((8388609:ℝ)/8388608)/Real.log 2≤100/(69*2^23) := by
  have hlog := Real.log_le_sub_one_of_pos (by norm_num : (0:ℝ)<8388609/8388608)
  have hp : 0<Real.log ((8388609:ℝ)/8388608) := Real.log_pos (by norm_num)
  have hl := log_two_interval
  constructor
  · exact div_pos hp (by linarith)
  · apply (div_le_iff₀ (by linarith : 0<Real.log 2)).mpr
    nlinarith

theorem below_ideal_interval :
    Real.log ((16777215:ℝ)/16777216)/Real.log 2≤-(10/(7*2^24)) := by
  have hlog := Real.log_le_sub_one_of_pos (by norm_num : (0:ℝ)<16777215/16777216)
  have hl := log_two_interval
  apply (div_le_iff₀ (by linarith : 0<Real.log 2)).mpr
  nlinarith

theorem above_relative_error :
    |(19195:ℝ)/2^36-Real.log ((8388609:ℝ)/8388608)/Real.log 2| /
      |Real.log ((8388609:ℝ)/8388608)/Real.log 2| > 3/5 := by
  have h := above_ideal_interval
  rw [abs_of_pos h.1,abs_of_nonneg (by nlinarith : 0≤(19195:ℝ)/2^36-Real.log ((8388609:ℝ)/8388608)/Real.log 2)]
  apply (lt_div_iff₀ h.1).mpr
  nlinarith

theorem below_relative_error :
    |-(813:ℝ)/2^36-Real.log ((16777215:ℝ)/16777216)/Real.log 2| /
      |Real.log ((16777215:ℝ)/16777216)/Real.log 2| > 17/20 := by
  have h := below_ideal_interval
  have hn : Real.log ((16777215:ℝ)/16777216)/Real.log 2<0 := by nlinarith
  rw [abs_of_neg hn,abs_of_nonneg (by nlinarith : 0≤-(813:ℝ)/2^36-Real.log ((16777215:ℝ)/16777216)/Real.log 2)]
  apply (lt_div_iff₀ (by linarith : 0< -(Real.log ((16777215:ℝ)/16777216)/Real.log 2))).mpr
  nlinarith

theorem absolute_error_within_ptx :
    |(19195:ℝ)/2^36-Real.log ((8388609:ℝ)/8388608)/Real.log 2|≤1/2^22 ∧
    |-(813:ℝ)/2^36-Real.log ((16777215:ℝ)/16777216)/Real.log 2|≤1/2^22 := by
  have hl := log_two_interval
  have hlogpos : 0<Real.log 2 := by linarith
  have ha := above_ideal_interval
  have hb := below_ideal_interval
  have hla := Real.one_sub_inv_le_log_of_pos (by norm_num : (0:ℝ)<8388609/8388608)
  have hlb := Real.one_sub_inv_le_log_of_pos (by norm_num : (0:ℝ)<16777215/16777216)
  have hamin : 10/(7*8388609)≤Real.log ((8388609:ℝ)/8388608)/Real.log 2 := by
    apply (le_div_iff₀ hlogpos).mpr
    norm_num at hla
    nlinarith
  have hbmin : -(100/(69*16777215))≤Real.log ((16777215:ℝ)/16777216)/Real.log 2 := by
    apply (le_div_iff₀ hlogpos).mpr
    norm_num at hlb
    nlinarith
  constructor <;> apply abs_le.mpr <;> constructor <;> nlinarith

theorem above_decoded_error :
    ∃ x y : Rat, Root32FloatLib.value 0x3f800001=some x ∧
      Root32FloatLib.value (Mufu.lg2 0x3f800001)=some y ∧
      |(y:ℝ)-Real.log (x:ℝ)/Real.log 2|/|Real.log (x:ℝ)/Real.log 2| > 3/5 ∧
      |(y:ℝ)-Real.log (x:ℝ)/Real.log 2|≤1/2^22 := by
  refine ⟨8388609/8388608,19195/2^36,above_decode.1,above_decode.2,?_,?_⟩
  · norm_num only [Rat.cast_div,Rat.cast_ofNat,Rat.cast_pow]
    convert above_relative_error using 1; norm_num
  · norm_num only [Rat.cast_div,Rat.cast_ofNat,Rat.cast_pow]
    convert absolute_error_within_ptx.1 using 1 <;> norm_num

theorem below_decoded_error :
    ∃ x y : Rat, Root32FloatLib.value 0x3f7fffff=some x ∧
      Root32FloatLib.value (Mufu.lg2 0x3f7fffff)=some y ∧
      |(y:ℝ)-Real.log (x:ℝ)/Real.log 2|/|Real.log (x:ℝ)/Real.log 2| > 17/20 ∧
      |(y:ℝ)-Real.log (x:ℝ)/Real.log 2|≤1/2^22 := by
  refine ⟨16777215/16777216,-(813/2^36),below_decode.1,below_decode.2,?_,?_⟩
  · norm_num only [Rat.cast_neg,Rat.cast_div,Rat.cast_ofNat,Rat.cast_pow]
    convert below_relative_error using 1; norm_num
  · norm_num only [Rat.cast_neg,Rat.cast_div,Rat.cast_ofNat,Rat.cast_pow]
    convert absolute_error_within_ptx.2 using 1 <;> norm_num

#print axioms absolute_error_within_ptx
#print axioms above_decoded_error
#print axioms below_decoded_error

#print axioms above_bits
#print axioms below_bits
#print axioms above_decode
#print axioms below_decode
#print axioms log_two_interval
#print axioms above_relative_error
#print axioms below_relative_error
end GB300Math.Log2NearOne

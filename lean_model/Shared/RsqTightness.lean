/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import RsqBounds
set_option maxRecDepth 100000
set_option maxHeartbeats 1000000
namespace GB300Math.RsqTightness
open GB300Math

theorem output_bits : Mufu.rsq 0x3f820399 = 0x3f7e0267 := by decide +kernel

theorem input_value : SqrtBounds.positiveNormalValue 0x3f820399 = (8520601 : ℝ)/8388608 := by
  norm_num [SqrtBounds.positiveNormalValue,RcpBounds.positiveNormalValue,Mufu.exponent,Mufu.fraction]
theorem output_value : SqrtBounds.positiveNormalValue 0x3f7e0267 = (16646759 : ℝ)/16777216 := by
  norm_num [SqrtBounds.positiveNormalValue,RcpBounds.positiveNormalValue,Mufu.exponent,Mufu.fraction]

theorem lower_square : (8520601 : Nat)*(100000000*16646759+151448512)^2 < 2^71*100000000^2 := by decide +kernel
theorem upper_square : (2 : Nat)^71*100000000^2 < 8520601*(100000000*16646759+151448513)^2 := by decide +kernel

theorem ulp_error_interval :
    let e := |SqrtBounds.positiveNormalValue (Mufu.rsq 0x3f820399)-
      1/Real.sqrt (SqrtBounds.positiveNormalValue 0x3f820399)| *2^24
    (151448512 : ℝ)/100000000 < e ∧ e < (151448513 : ℝ)/100000000 := by
  dsimp only
  rw [output_bits,input_value,output_value]
  let x : ℝ := 8520601/8388608
  let r : ℝ := 16777216/Real.sqrt x
  have hs : 0 < Real.sqrt x := Real.sqrt_pos.mpr (by dsimp [x];positivity)
  have hsq := Real.sq_sqrt (show 0 ≤ x by dsimp [x];positivity)
  have hr : 0 < r := by dsimp [r];positivity
  have he : r^2*8520601 = 2^71 := by
    dsimp [r]
    rw [div_pow,hsq]
    norm_num [x]
  have hls : (8520601 : ℝ)*(16646759+151448512/100000000)^2 < 2^71 := by norm_num
  have hus : (2 : ℝ)^71 < 8520601*(16646759+151448513/100000000)^2 := by norm_num
  have hlo : (16646759 : ℝ)+151448512/100000000 < r := by nlinarith only [he,hls,hr]
  have hhi : r < (16646759 : ℝ)+151448513/100000000 := by nlinarith only [he,hus,hr]
  have hneg : (16646759 : ℝ)/16777216-1/Real.sqrt x ≤ 0 := by
    have h : (16646759 : ℝ) < 16777216/Real.sqrt x := by change (16646759 : ℝ) < r;linarith
    rw [show (16777216 : ℝ)/Real.sqrt x = 16777216*(1/Real.sqrt x) by ring] at h
    nlinarith only [h]
  change _ < |(16646759 : ℝ)/16777216-1/Real.sqrt x| *2^24 ∧ _
  rw [abs_of_nonpos hneg]
  have hid : -((16646759 : ℝ)/16777216-1/Real.sqrt x)*2^24 = r-16646759 := by
    dsimp only [r]
    rw [show (2 : ℝ)^24 = 16777216 by norm_num]
    ring
  rw [hid]
  constructor <;> linarith

#print axioms output_bits
#print axioms input_value
#print axioms output_value
#print axioms lower_square
#print axioms upper_square
#print axioms ulp_error_interval
end GB300Math.RsqTightness

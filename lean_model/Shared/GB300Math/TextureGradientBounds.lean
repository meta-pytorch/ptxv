/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import TextureGradient
import GB300Math.TextureGradientAxis
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic

set_option maxRecDepth 10000
set_option maxHeartbeats 800000

namespace GB300Math.TextureGradientBounds

open TextureGradient

theorem table_brackets : ∀ i ∈ List.range 256,
    2^(2*logTable[i]!+4096) ≤ 8*(256+i)^512 ∧
      (257+i)^512 ≤ 8*2^(2*logTable[i]!+4096) := by
  decide +kernel

theorem cell_error (i : Nat) (hi : i < 256) (x : ℝ)
    (hl : ((256+i:Nat):ℝ)/256 ≤ x) (hu : x ≤ ((257+i:Nat):ℝ)/256) :
    |Real.log x / Real.log 2 - (logTable[i]!:ℝ)/256| ≤ 3/512 := by
  obtain ⟨ha,hb⟩ := table_brackets i (List.mem_range.mpr hi)
  generalize hq : logTable[i]! = q at *
  have ha' : (2:ℝ)^(2*q+4096) ≤ 8*((256+i:Nat):ℝ)^512 := by exact_mod_cast ha
  have hb' : ((257+i:Nat):ℝ)^512 ≤ 8*(2:ℝ)^(2*q+4096) := by exact_mod_cast hb
  have h2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have h8 : Real.log (8:ℝ) = 3*Real.log 2 := by rw [show (8:ℝ)=2^3 by norm_num,Real.log_pow]; norm_num
  have h256 : Real.log (256:ℝ) = 8*Real.log 2 := by rw [show (256:ℝ)=2^8 by norm_num,Real.log_pow]; norm_num
  have hp : (0:ℝ) < ((256+i:Nat):ℝ)  := by exact_mod_cast (show 0 < 256+i by omega)
  have hp' : (0:ℝ) < ((257+i:Nat):ℝ) := by exact_mod_cast (show 0 < 257+i by omega)
  have hx : 0 < x := lt_of_lt_of_le (div_pos hp (by norm_num)) hl
  have la := Real.log_le_log (pow_pos (by norm_num : (0:ℝ) < 2) _) ha'
  have lb := Real.log_le_log (pow_pos hp' _) hb'
  rw [Real.log_mul (by norm_num) (pow_ne_zero _ (ne_of_gt hp)),Real.log_pow,Real.log_pow,h8] at la
  rw [Real.log_mul (by norm_num) (pow_ne_zero _ (by norm_num : (2:ℝ) ≠ 0)),Real.log_pow,Real.log_pow,h8] at lb
  have lx := Real.log_le_log (div_pos hp (by norm_num)) hl
  have ux := Real.log_le_log hx hu
  rw [Real.log_div (ne_of_gt hp) (by norm_num),h256] at lx
  rw [Real.log_div (ne_of_gt hp') (by norm_num),h256] at ux
  push_cast at la lb lx ux
  rw [abs_le]
  constructor
  · have h : (q:ℝ)/256-3/512 ≤ Real.log x/Real.log 2 := by
      apply (le_div_iff₀ h2).2
      nlinarith only [la,lb,lx,ux]
    linarith only [h]
  · have h : Real.log x/Real.log 2 ≤ (q:ℝ)/256+3/512 := by
      apply (div_le_iff₀ h2).2
      nlinarith only [la,lb,lx,ux]
    linarith only [h]

theorem fraction_error (m : Nat) (hm : m < 8388608) :
    |Real.log (((8388608+m:Nat):ℝ)/8388608) / Real.log 2 -
      (logTable[m/32768]!:ℝ)/256| ≤ 3/512 := by
  have hi : m/32768 < 256 := by omega
  have hl : (256+m/32768)*32768 ≤ 8388608+m := by omega
  have hu : 8388608+m ≤ (257+m/32768)*32768 := by omega
  have hl' : (((256+m/32768)*32768:Nat):ℝ) ≤ (8388608+m:Nat) := by exact_mod_cast hl
  have hu' : ((8388608+m:Nat):ℝ) ≤ ((257+m/32768)*32768:Nat) := by exact_mod_cast hu
  apply cell_error (m/32768) hi
  · push_cast at hl' ⊢
    linarith only [hl']
  · push_cast at hu' ⊢
    linarith only [hu']

theorem normal_gradient_error (v dimension : Nat) (he : exponent v ≠ 0) (_hf : exponent v < 255) :
    |((gradientLog v dimension : Int):ℝ)/256 -
      ((exponent v:ℝ)-127+(dimension.log2:ℝ) +
        Real.log (((8388608+v%8388608:Nat):ℝ)/8388608)/Real.log 2)| ≤ 3/512 := by
  have h := fraction_error (v%8388608) (Nat.mod_lt _ (by decide))
  simp only [gradientLog, he, ↓reduceIte]
  rw [abs_sub_comm]
  convert h using 1
  congr 1
  push_cast
  ring

theorem axis_table_agrees : TextureGradientAxis.logTable = TextureGradient.logTable := rfl

theorem normal_axis_position (v : Nat) (he : exponent v ≠ 0) (hf : exponent v < 255) :
    TextureGradientAxis.position v = min 2048 (gradientLog v 256).toNat := by
  have he' : v/2^23%256 ≠ 0 := he
  have hf' : v/2^23%256 ≠ 255 := Nat.ne_of_lt hf
  simp only [TextureGradientAxis.position, gradientLog, exponent, he', hf',
    false_and, false_or, ↓reduceIte, axis_table_agrees]
  have h256 : Nat.log2 256 = 8 := by decide +kernel
  rw [h256]
  apply congrArg (fun z : Int => min 2048 z.toNat)
  ring

def clip (x : ℝ) : ℝ := min 8 (max 0 x)

theorem clipping_nonexpansive (a b : ℝ) : |clip a-clip b| ≤ |a-b| := by
  have h1 := le_abs_self (a-b)
  have h2 := neg_abs_le (a-b)
  simp only [clip, min_def, max_def]
  split_ifs <;> rw [abs_le] <;> constructor <;> linarith

theorem clipped_normal_gradient_error (v dimension : Nat)
    (he : exponent v ≠ 0) (hf : exponent v < 255) :
    |clip (((gradientLog v dimension : Int):ℝ)/256) -
      clip ((exponent v:ℝ)-127+(dimension.log2:ℝ) +
        Real.log (((8388608+v%8388608:Nat):ℝ)/8388608)/Real.log 2)| ≤ 3/512 := by
  exact le_trans (clipping_nonexpansive _ _) (normal_gradient_error v dimension he hf)

#print axioms table_brackets
#print axioms cell_error
#print axioms fraction_error
#print axioms normal_gradient_error
#print axioms axis_table_agrees
#print axioms normal_axis_position
#print axioms clipping_nonexpansive
#print axioms clipped_normal_gradient_error
end GB300Math.TextureGradientBounds

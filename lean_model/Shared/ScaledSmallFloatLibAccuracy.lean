/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured
import Mathlib.Tactic.IntervalCases
import SmallConversions

set_option maxRecDepth 1000000
set_option maxHeartbeats 2000000000

namespace GB300Math.ScaledSmallAccuracy
open FloatLib.Floats GB300Math.SmallConversions

def ideal (name : String) (fmt : Format) (word scale : Nat) : Option ℚ :=
  let e := word / 2^fmt.fractionBits % 2^fmt.exponentBits
  let f := word % 2^fmt.fractionBits
  if (name = "e4m3" ∧ e = 15 ∧ f = 7) ∨ (name = "e5m2" ∧ e = 31) then none else
    let negative := word / 2^(fmt.fractionBits+fmt.exponentBits) % 2 = 1
    let significand := f + if e = 0 then 0 else 2^fmt.fractionBits
    let power : Int := (max e 1 : Int) - (fmt.bias : Int) - (fmt.fractionBits : Int) + (scale : Int)-127
    some ((if negative then -1 else 1) * (significand : ℚ) * (2 : ℚ)^power)

def decoded (name : String) (fmt : Format) (word scale : Nat) : Option ℚ :=
  let bits := upSmall name fmt true false false true word scale
  ExecFloat.Binary.toRat? (ExecFloat.Binary.ofNatBits bits : ExecFloat.Binary 8 7)

def accurate (name : String) (fmt : Format) (word scale : Nat) : Prop :=
  match ideal name fmt word scale with
  | none => decoded name fmt word scale = none
  | some exact =>
    if |exact| > 255 * (2 : ℚ)^120 then True else
      match decoded name fmt word scale with
      | none => False
      | some actual =>
        |actual-exact| ≤ (2 : ℚ)^(-134 : Int) ∧
        (|exact| ≥ (2 : ℚ)^(-126 : Int) → actual = exact)

instance (name : String) (fmt : Format) (word scale : Nat) : Decidable (accurate name fmt word scale) := by
  unfold accurate
  split
  · infer_instance
  · split
    · infer_instance
    · split <;> infer_instance

private theorem e4m3_block_0 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e4m3" ⟨4, 3, 7, 126⟩ (0+code.val) scale.val := by decide +kernel

private theorem e4m3_block_1 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e4m3" ⟨4, 3, 7, 126⟩ (16+code.val) scale.val := by decide +kernel

private theorem e4m3_block_2 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e4m3" ⟨4, 3, 7, 126⟩ (32+code.val) scale.val := by decide +kernel

private theorem e4m3_block_3 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e4m3" ⟨4, 3, 7, 126⟩ (48+code.val) scale.val := by decide +kernel

private theorem e4m3_block_4 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e4m3" ⟨4, 3, 7, 126⟩ (64+code.val) scale.val := by decide +kernel

private theorem e4m3_block_5 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e4m3" ⟨4, 3, 7, 126⟩ (80+code.val) scale.val := by decide +kernel

private theorem e4m3_block_6 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e4m3" ⟨4, 3, 7, 126⟩ (96+code.val) scale.val := by decide +kernel

private theorem e4m3_block_7 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e4m3" ⟨4, 3, 7, 126⟩ (112+code.val) scale.val := by decide +kernel

private theorem e4m3_block_8 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e4m3" ⟨4, 3, 7, 126⟩ (128+code.val) scale.val := by decide +kernel

private theorem e4m3_block_9 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e4m3" ⟨4, 3, 7, 126⟩ (144+code.val) scale.val := by decide +kernel

private theorem e4m3_block_10 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e4m3" ⟨4, 3, 7, 126⟩ (160+code.val) scale.val := by decide +kernel

private theorem e4m3_block_11 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e4m3" ⟨4, 3, 7, 126⟩ (176+code.val) scale.val := by decide +kernel

private theorem e4m3_block_12 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e4m3" ⟨4, 3, 7, 126⟩ (192+code.val) scale.val := by decide +kernel

private theorem e4m3_block_13 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e4m3" ⟨4, 3, 7, 126⟩ (208+code.val) scale.val := by decide +kernel

private theorem e4m3_block_14 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e4m3" ⟨4, 3, 7, 126⟩ (224+code.val) scale.val := by decide +kernel

private theorem e4m3_block_15 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e4m3" ⟨4, 3, 7, 126⟩ (240+code.val) scale.val := by decide +kernel

theorem e4m3_scaled_accuracy : ∀ code : Fin 256, ∀ scale : Fin 255,
    accurate "e4m3" ⟨4, 3, 7, 126⟩ code.val scale.val := by
  intro code scale
  have hcode := code.isLt
  have hlow : code.val % 16 < 16 := Nat.mod_lt _ (by decide)
  have hdiv : code.val / 16 ≤ 15 := by omega
  interval_cases h : code.val / 16
  · have heq : 0 + code.val % 16 = code.val := by omega
    have proof := e4m3_block_0 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 16 + code.val % 16 = code.val := by omega
    have proof := e4m3_block_1 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 32 + code.val % 16 = code.val := by omega
    have proof := e4m3_block_2 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 48 + code.val % 16 = code.val := by omega
    have proof := e4m3_block_3 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 64 + code.val % 16 = code.val := by omega
    have proof := e4m3_block_4 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 80 + code.val % 16 = code.val := by omega
    have proof := e4m3_block_5 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 96 + code.val % 16 = code.val := by omega
    have proof := e4m3_block_6 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 112 + code.val % 16 = code.val := by omega
    have proof := e4m3_block_7 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 128 + code.val % 16 = code.val := by omega
    have proof := e4m3_block_8 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 144 + code.val % 16 = code.val := by omega
    have proof := e4m3_block_9 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 160 + code.val % 16 = code.val := by omega
    have proof := e4m3_block_10 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 176 + code.val % 16 = code.val := by omega
    have proof := e4m3_block_11 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 192 + code.val % 16 = code.val := by omega
    have proof := e4m3_block_12 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 208 + code.val % 16 = code.val := by omega
    have proof := e4m3_block_13 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 224 + code.val % 16 = code.val := by omega
    have proof := e4m3_block_14 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 240 + code.val % 16 = code.val := by omega
    have proof := e4m3_block_15 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof

#print axioms e4m3_scaled_accuracy

private theorem e5m2_block_0 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e5m2" ⟨5, 2, 15, 123⟩ (0+code.val) scale.val := by decide +kernel

private theorem e5m2_block_1 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e5m2" ⟨5, 2, 15, 123⟩ (16+code.val) scale.val := by decide +kernel

private theorem e5m2_block_2 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e5m2" ⟨5, 2, 15, 123⟩ (32+code.val) scale.val := by decide +kernel

private theorem e5m2_block_3 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e5m2" ⟨5, 2, 15, 123⟩ (48+code.val) scale.val := by decide +kernel

private theorem e5m2_block_4 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e5m2" ⟨5, 2, 15, 123⟩ (64+code.val) scale.val := by decide +kernel

private theorem e5m2_block_5 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e5m2" ⟨5, 2, 15, 123⟩ (80+code.val) scale.val := by decide +kernel

private theorem e5m2_block_6 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e5m2" ⟨5, 2, 15, 123⟩ (96+code.val) scale.val := by decide +kernel

private theorem e5m2_block_7 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e5m2" ⟨5, 2, 15, 123⟩ (112+code.val) scale.val := by decide +kernel

private theorem e5m2_block_8 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e5m2" ⟨5, 2, 15, 123⟩ (128+code.val) scale.val := by decide +kernel

private theorem e5m2_block_9 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e5m2" ⟨5, 2, 15, 123⟩ (144+code.val) scale.val := by decide +kernel

private theorem e5m2_block_10 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e5m2" ⟨5, 2, 15, 123⟩ (160+code.val) scale.val := by decide +kernel

private theorem e5m2_block_11 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e5m2" ⟨5, 2, 15, 123⟩ (176+code.val) scale.val := by decide +kernel

private theorem e5m2_block_12 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e5m2" ⟨5, 2, 15, 123⟩ (192+code.val) scale.val := by decide +kernel

private theorem e5m2_block_13 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e5m2" ⟨5, 2, 15, 123⟩ (208+code.val) scale.val := by decide +kernel

private theorem e5m2_block_14 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e5m2" ⟨5, 2, 15, 123⟩ (224+code.val) scale.val := by decide +kernel

private theorem e5m2_block_15 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e5m2" ⟨5, 2, 15, 123⟩ (240+code.val) scale.val := by decide +kernel

theorem e5m2_scaled_accuracy : ∀ code : Fin 256, ∀ scale : Fin 255,
    accurate "e5m2" ⟨5, 2, 15, 123⟩ code.val scale.val := by
  intro code scale
  have hcode := code.isLt
  have hlow : code.val % 16 < 16 := Nat.mod_lt _ (by decide)
  have hdiv : code.val / 16 ≤ 15 := by omega
  interval_cases h : code.val / 16
  · have heq : 0 + code.val % 16 = code.val := by omega
    have proof := e5m2_block_0 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 16 + code.val % 16 = code.val := by omega
    have proof := e5m2_block_1 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 32 + code.val % 16 = code.val := by omega
    have proof := e5m2_block_2 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 48 + code.val % 16 = code.val := by omega
    have proof := e5m2_block_3 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 64 + code.val % 16 = code.val := by omega
    have proof := e5m2_block_4 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 80 + code.val % 16 = code.val := by omega
    have proof := e5m2_block_5 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 96 + code.val % 16 = code.val := by omega
    have proof := e5m2_block_6 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 112 + code.val % 16 = code.val := by omega
    have proof := e5m2_block_7 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 128 + code.val % 16 = code.val := by omega
    have proof := e5m2_block_8 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 144 + code.val % 16 = code.val := by omega
    have proof := e5m2_block_9 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 160 + code.val % 16 = code.val := by omega
    have proof := e5m2_block_10 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 176 + code.val % 16 = code.val := by omega
    have proof := e5m2_block_11 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 192 + code.val % 16 = code.val := by omega
    have proof := e5m2_block_12 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 208 + code.val % 16 = code.val := by omega
    have proof := e5m2_block_13 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 224 + code.val % 16 = code.val := by omega
    have proof := e5m2_block_14 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 240 + code.val % 16 = code.val := by omega
    have proof := e5m2_block_15 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof

#print axioms e5m2_scaled_accuracy

private theorem e2m3_block_0 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e2m3" ⟨2, 3, 1, 31⟩ (0+code.val) scale.val := by decide +kernel

private theorem e2m3_block_1 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e2m3" ⟨2, 3, 1, 31⟩ (16+code.val) scale.val := by decide +kernel

private theorem e2m3_block_2 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e2m3" ⟨2, 3, 1, 31⟩ (32+code.val) scale.val := by decide +kernel

private theorem e2m3_block_3 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e2m3" ⟨2, 3, 1, 31⟩ (48+code.val) scale.val := by decide +kernel

theorem e2m3_scaled_accuracy : ∀ code : Fin 64, ∀ scale : Fin 255,
    accurate "e2m3" ⟨2, 3, 1, 31⟩ code.val scale.val := by
  intro code scale
  have hcode := code.isLt
  have hlow : code.val % 16 < 16 := Nat.mod_lt _ (by decide)
  have hdiv : code.val / 16 ≤ 3 := by omega
  interval_cases h : code.val / 16
  · have heq : 0 + code.val % 16 = code.val := by omega
    have proof := e2m3_block_0 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 16 + code.val % 16 = code.val := by omega
    have proof := e2m3_block_1 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 32 + code.val % 16 = code.val := by omega
    have proof := e2m3_block_2 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 48 + code.val % 16 = code.val := by omega
    have proof := e2m3_block_3 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof

#print axioms e2m3_scaled_accuracy

private theorem e3m2_block_0 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e3m2" ⟨3, 2, 3, 31⟩ (0+code.val) scale.val := by decide +kernel

private theorem e3m2_block_1 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e3m2" ⟨3, 2, 3, 31⟩ (16+code.val) scale.val := by decide +kernel

private theorem e3m2_block_2 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e3m2" ⟨3, 2, 3, 31⟩ (32+code.val) scale.val := by decide +kernel

private theorem e3m2_block_3 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e3m2" ⟨3, 2, 3, 31⟩ (48+code.val) scale.val := by decide +kernel

theorem e3m2_scaled_accuracy : ∀ code : Fin 64, ∀ scale : Fin 255,
    accurate "e3m2" ⟨3, 2, 3, 31⟩ code.val scale.val := by
  intro code scale
  have hcode := code.isLt
  have hlow : code.val % 16 < 16 := Nat.mod_lt _ (by decide)
  have hdiv : code.val / 16 ≤ 3 := by omega
  interval_cases h : code.val / 16
  · have heq : 0 + code.val % 16 = code.val := by omega
    have proof := e3m2_block_0 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 16 + code.val % 16 = code.val := by omega
    have proof := e3m2_block_1 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 32 + code.val % 16 = code.val := by omega
    have proof := e3m2_block_2 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 48 + code.val % 16 = code.val := by omega
    have proof := e3m2_block_3 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof

#print axioms e3m2_scaled_accuracy

private theorem e2m1_block_0 : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e2m1" ⟨2, 1, 1, 7⟩ (0+code.val) scale.val := by decide +kernel

theorem e2m1_scaled_accuracy : ∀ code : Fin 16, ∀ scale : Fin 255,
    accurate "e2m1" ⟨2, 1, 1, 7⟩ code.val scale.val := by
  intro code scale
  have hcode := code.isLt
  have hlow : code.val % 16 < 16 := Nat.mod_lt _ (by decide)
  have hdiv : code.val / 16 ≤ 0 := by omega
  interval_cases h : code.val / 16
  · have heq : 0 + code.val % 16 = code.val := by omega
    have proof := e2m1_block_0 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof

#print axioms e2m1_scaled_accuracy

end GB300Math.ScaledSmallAccuracy

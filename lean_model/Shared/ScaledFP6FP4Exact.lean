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

namespace GB300Math.ScaledFP6FP4Exact
open FloatLib.Floats GB300Math.SmallConversions

def ideal (fmt : Format) (word scale : Nat) : ℚ :=
  let e := word / 2^fmt.fractionBits % 2^fmt.exponentBits
  let f := word % 2^fmt.fractionBits
  let negative := word / 2^(fmt.fractionBits+fmt.exponentBits) % 2 = 1
  let m := f + if e = 0 then 0 else 2^fmt.fractionBits
  let power : Int := (max e 1 : Int) - (fmt.bias : Int) - (fmt.fractionBits : Int) + (scale : Int)-127
  (if negative then -1 else 1) * (m : ℚ) * (2 : ℚ)^power

def exactConversion (name : String) (fmt : Format) (word scale : Nat) : Prop :=
  let value := ideal fmt word scale
  |value| ≤ 255 * (2 : ℚ)^120 →
    ExecFloat.Binary.toRat? (ExecFloat.Binary.ofNatBits
      (upSmall name fmt true false false true word scale) : ExecFloat.Binary 8 7) = some value

instance (name : String) (fmt : Format) (word scale : Nat) :
    Decidable (exactConversion name fmt word scale) := by unfold exactConversion; infer_instance

private theorem e2m3_block0 : ∀ code : Fin 16, ∀ scale : Fin 255,
    exactConversion "e2m3" ⟨2,3,1,31⟩ (0+code.val) scale.val := by decide +kernel

private theorem e2m3_block1 : ∀ code : Fin 16, ∀ scale : Fin 255,
    exactConversion "e2m3" ⟨2,3,1,31⟩ (16+code.val) scale.val := by decide +kernel

private theorem e2m3_block2 : ∀ code : Fin 16, ∀ scale : Fin 255,
    exactConversion "e2m3" ⟨2,3,1,31⟩ (32+code.val) scale.val := by decide +kernel

private theorem e2m3_block3 : ∀ code : Fin 16, ∀ scale : Fin 255,
    exactConversion "e2m3" ⟨2,3,1,31⟩ (48+code.val) scale.val := by decide +kernel

theorem e2m3_scaled_exact : ∀ code : Fin 64, ∀ scale : Fin 255,
    exactConversion "e2m3" ⟨2,3,1,31⟩ code.val scale.val := by
  intro code scale
  have hc := code.isLt
  have hm : code.val % 16 < 16 := Nat.mod_lt _ (by decide)
  have hd : code.val / 16 ≤ 3 := by omega
  interval_cases h : code.val / 16
  · have heq : 0 + code.val % 16 = code.val := by omega
    simpa only [heq] using e2m3_block0 ⟨code.val % 16, hm⟩ scale
  · have heq : 16 + code.val % 16 = code.val := by omega
    simpa only [heq] using e2m3_block1 ⟨code.val % 16, hm⟩ scale
  · have heq : 32 + code.val % 16 = code.val := by omega
    simpa only [heq] using e2m3_block2 ⟨code.val % 16, hm⟩ scale
  · have heq : 48 + code.val % 16 = code.val := by omega
    simpa only [heq] using e2m3_block3 ⟨code.val % 16, hm⟩ scale

#print axioms e2m3_scaled_exact

private theorem e3m2_block0 : ∀ code : Fin 16, ∀ scale : Fin 255,
    exactConversion "e3m2" ⟨3,2,3,31⟩ (0+code.val) scale.val := by decide +kernel

private theorem e3m2_block1 : ∀ code : Fin 16, ∀ scale : Fin 255,
    exactConversion "e3m2" ⟨3,2,3,31⟩ (16+code.val) scale.val := by decide +kernel

private theorem e3m2_block2 : ∀ code : Fin 16, ∀ scale : Fin 255,
    exactConversion "e3m2" ⟨3,2,3,31⟩ (32+code.val) scale.val := by decide +kernel

private theorem e3m2_block3 : ∀ code : Fin 16, ∀ scale : Fin 255,
    exactConversion "e3m2" ⟨3,2,3,31⟩ (48+code.val) scale.val := by decide +kernel

theorem e3m2_scaled_exact : ∀ code : Fin 64, ∀ scale : Fin 255,
    exactConversion "e3m2" ⟨3,2,3,31⟩ code.val scale.val := by
  intro code scale
  have hc := code.isLt
  have hm : code.val % 16 < 16 := Nat.mod_lt _ (by decide)
  have hd : code.val / 16 ≤ 3 := by omega
  interval_cases h : code.val / 16
  · have heq : 0 + code.val % 16 = code.val := by omega
    simpa only [heq] using e3m2_block0 ⟨code.val % 16, hm⟩ scale
  · have heq : 16 + code.val % 16 = code.val := by omega
    simpa only [heq] using e3m2_block1 ⟨code.val % 16, hm⟩ scale
  · have heq : 32 + code.val % 16 = code.val := by omega
    simpa only [heq] using e3m2_block2 ⟨code.val % 16, hm⟩ scale
  · have heq : 48 + code.val % 16 = code.val := by omega
    simpa only [heq] using e3m2_block3 ⟨code.val % 16, hm⟩ scale

#print axioms e3m2_scaled_exact

private theorem e2m1_block0 : ∀ code : Fin 16, ∀ scale : Fin 255,
    exactConversion "e2m1" ⟨2,1,1,7⟩ (0+code.val) scale.val := by decide +kernel

theorem e2m1_scaled_exact : ∀ code : Fin 16, ∀ scale : Fin 255,
    exactConversion "e2m1" ⟨2,1,1,7⟩ code.val scale.val := by
  intro code scale
  have hc := code.isLt
  have hm : code.val % 16 < 16 := Nat.mod_lt _ (by decide)
  have hd : code.val / 16 ≤ 0 := by omega
  interval_cases h : code.val / 16
  · have heq : 0 + code.val % 16 = code.val := by omega
    simpa only [heq] using e2m1_block0 ⟨code.val % 16, hm⟩ scale

#print axioms e2m1_scaled_exact

end GB300Math.ScaledFP6FP4Exact

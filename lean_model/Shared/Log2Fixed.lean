/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MufuRcp32Signed
namespace GB300Math.Log2Fixed
open FloatLib.Floats.Formats.BinaryInterchange
set_option maxRecDepth 12000
set_option maxHeartbeats 5000000

theorem decode (lead mant : Nat) (hl : lead ≤ 42)
    (hm : 2^23 ≤ mant ∧ mant < 2^24) :
    (Model.ofNatBits ((lead+91)*2^23+mant%2^23) : Model FloatFormat.binary32).toRat? =
      some ((mant:Rat)*(2:Rat)^((lead:Int)-59)) := by
  let e : Fin 254 := ⟨lead+90,by omega⟩
  let f : Fin (2^23) := ⟨mant%2^23,by omega⟩
  have h := MufuRcp32Bounds.decode_normal e f
  have he : (e.val:Int)-149=(lead:Int)-59 := by dsimp [e];omega
  have hf : 2^23+f.val=mant := by dsimp [f];omega
  have hw : (1+e.val)*2^23+f.val=(lead+91)*2^23+mant%2^23 := by dsimp [e,f];omega
  rwa [he,hf,hw] at h

theorem fixed_error (mag : Nat) (hm : 0 < mag ∧ mag < 2^43) :
    ∃ y : Rat,
      (Model.ofNatBits (Mufu.fixedToWord mag 36) : Model FloatFormat.binary32).toRat? = some y ∧
      0 ≤ (mag:Rat)/2^36-y ∧ (mag:Rat)/2^36-y ≤ 1/2^17 := by
  have hn : mag ≠ 0 := by omega
  have ha := Nat.log2_self_le hn
  have hb := Nat.lt_log2_self (n:=mag)
  have hhigh : mag.log2 ≤ 42 := by have h := (Nat.log2_lt hn).mpr hm.2;omega
  generalize hlead : mag.log2 = lead at *
  have hexp : ((lead:Int)-36+127).toNat = lead+91 := by omega
  let mant := if lead ≥ 23 then mag/2^(lead-23) else mag*2^(23-lead)
  have hmnt : 2^23 ≤ mant ∧ mant < 2^24 := by
    dsimp [mant]
    interval_cases lead <;> norm_num at * <;> omega
  refine ⟨(mant:Rat)*(2:Rat)^((lead:Int)-59),?_,?_⟩
  · unfold Mufu.fixedToWord
    simp only [hlead]
    rw [hexp]
    exact decode lead mant hhigh hmnt
  have hdiv0 := Nat.div_mul_le_self mag (2^(lead-23))
  have hdiv1 : mag < mag/2^(lead-23)*2^(lead-23)+2^(lead-23) := by
    have hpos : 0 < 2^(lead-23) := by positivity
    have hmod := Nat.mod_lt mag hpos
    have hid := Nat.mod_add_div mag (2^(lead-23))
    nlinarith only [hmod,hid]
  have hd0 : (((mag/2^(lead-23):Nat):Rat)*(2^(lead-23):Nat)) ≤ mag := by exact_mod_cast hdiv0
  have hd1 : (mag:Rat) < ((mag/2^(lead-23):Nat):Rat)*(2^(lead-23):Nat)+(2^(lead-23):Nat) := by exact_mod_cast hdiv1
  dsimp [mant]
  interval_cases lead <;> norm_num at * <;> constructor <;> linarith

#print axioms decode
#print axioms fixed_error
end GB300Math.Log2Fixed

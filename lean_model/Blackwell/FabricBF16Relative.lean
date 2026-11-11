/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricBF16V2Bounds
import FloatLib.Floats.Formats.Flocq.Theory.Error.Relative

namespace FabricBF16Relative
open FabricBF16Block FabricBF16V2 FabricBF16V2Bounds GB300Math.TensorAlignmentLoss
open FloatLib.Floats.Formats.BinaryInterchange

def alignmentLoss (words : List Nat) (values : List Int) : Nat :=
  values.length*(2^latticeShift words-1)

theorem coarse_dominates (words : List Nat) (values : List Int) (hn : values.length≤4) :
    128*alignmentLoss words values < 2^(coarseShift words) := by
  by_cases hz : latticeShift words=0
  · simp [alignmentLoss,hz]
  · have hs : coarseShift words=latticeShift words+9 := by
      dsimp [coarseShift,latticeShift] at *
      omega
    rw [hs,Nat.pow_add]
    have hp : 0<2^latticeShift words := Nat.two_pow_pos _
    dsimp [alignmentLoss]
    have hsub : 2^latticeShift words-1+1=2^latticeShift words := by omega
    nlinarith

theorem unflushed_relative (words : List Nat) (values : List Int)
    (hn : values.length≤4) (hz : acc words values≠0) :
    128*(acc words values-exactSum values).natAbs ≤ (acc words values).natAbs := by
  have hc := coarse_dominates words values hn
  have hl := accumulation_bound words values
  have hnf : ¬(aligned words values).natAbs<2^(coarseShift words) := by
    intro h
    exact hz (by simp [acc,flush,h])
  have he : acc words values=aligned words values := by simp [acc,flush,hnf]
  rw [he]
  have he2 : (aligned words values-exactSum values).natAbs≤alignmentLoss words values := by
    dsimp [alignmentLoss]
    omega
  omega

theorem unflushed_real (words : List Nat) (values : List Int)
    (hn : values.length≤4) (hz : acc words values≠0) :
    128*|(acc words values:ℝ)/2^133-(exactSum values:ℝ)/2^133| ≤
      |(acc words values:ℝ)/2^133| := by
  have h := unflushed_relative words values hn hz
  have hr : (128:ℝ)*|((acc words values-exactSum values:Int):ℝ)|≤|(acc words values:ℝ)| := by
    have hh : (128:ℝ)*(acc words values-exactSum values).natAbs≤((acc words values).natAbs:ℝ) := by exact_mod_cast h
    simpa using hh
  rw [Int.cast_sub] at hr
  rw [←sub_div,abs_div,abs_div,abs_of_pos (by positivity : (0:ℝ)<2^133)]
  rw [←mul_div_assoc,div_le_div_iff_of_pos_right (by positivity)]
  exact hr

theorem round_small_bits : ∀k:Fin 128,
    units (FabricBF16Block.round (k:Nat))=some ((k:Nat):Int) ∧
    units (FabricBF16Block.round (-(k:Nat):Int))=some (-(k:Nat):Int) := by
  decide +kernel

theorem round_relative (value : Int)
    (hf : Model.isFinite (Model.ofNatBits (FabricBF16Block.round value) :
      Model FloatFormat.bfloat16)=true) :
    |Model.toReal (Model.ofNatBits (FabricBF16Block.round value) : Model FloatFormat.bfloat16)-
      (value:ℝ)/2^133| ≤ |(value:ℝ)/2^133|/256 := by
  by_cases hsmall : value.natAbs<128
  · have h:=round_small_bits ⟨value.natAbs,hsmall⟩
    have hu : units (FabricBF16Block.round value)=some value := by
      rcases le_or_gt 0 value with hv|hv
      · have he : (value.natAbs:Int)=value := by omega
        simpa only [Fin.val_mk,he] using h.1
      · have he : -(value.natAbs:Int)=value := by omega
        simpa only [Fin.val_mk,he] using h.2
    rw [FabricBF16Decode.units_value _ value hu]
    simp only [sub_self,abs_zero]
    positivity
  · have hnormal : (2:ℝ)^(-126:Int)≤|(value:ℝ)/2^133| := by
      have hi : (128:ℝ)≤|(value:ℝ)| := by
        have hh : (128:ℝ)≤(value.natAbs:ℝ) := by exact_mod_cast (show 128≤value.natAbs by omega)
        simpa using hh
      rw [abs_div,abs_of_pos (by positivity : (0:ℝ)<2^133)]
      norm_num [zpow_neg] at *
      linarith
    have hx : (value:ℝ)/2^133≠0 := by
      have hv : value≠0 := by intro h; simp [h] at hsmall
      exact div_ne_zero (by exact_mod_cast hv) (by positivity)
    have hr := FloatLib.Floats.Formats.Flocq.relative_error_round_FLT_normal
      (β:=FloatLib.Numerics.binaryRadix) (-133) 8 (by norm_num)
      FloatLib.Floats.Formats.Flocq.nearestEven ((value:ℝ)/2^133) hx hnormal
    rw [FabricBF16RoundBounds.round_word] at hf ⊢
    rw [Model.toReal_roundDyadic_eq_roundAt _ (by decide) _ hf,FabricBF16RoundBounds.dyadic_value]
    have hb : FloatLib.Floats.Formats.Flocq.bpow FloatLib.Numerics.binaryRadix (1-8)= (1:ℝ)/128 := by
      change (2:ℝ)^(-7:Int)=1/128
      norm_num
    rw [hb] at hr
    change |_-_|/|(value:ℝ)/2^133|≤1/128/2 at hr
    have he := (div_le_iff₀ (abs_pos.mpr hx)).mp hr
    convert he using 1 <;> first | rfl | ring


theorem sr_relative (words : List Nat) (states : Array Accum) (lane : Nat)
    (valid : ∀w∈words,w<65536 ∧ w%32768<32640) (hn : words.length≤4)
    (hs : states[lane]! =accumulate words)
    (hf : Model.isFinite (Model.ofNatBits (FabricBF16V2.output states lane (coarseShift words) true) :
      Model FloatFormat.bfloat16)=true) :
    let y := Model.toReal (Model.ofNatBits (FabricBF16V2.output states lane (coarseShift words) true) : Model FloatFormat.bfloat16)
    let s := (words.map fun w => (Model.ofNatBits w : Model FloatFormat.bfloat16).toReal).sum
    |y-s|≤|s| ∧ (y≠0 → 127*|y-s|≤2*|s|) := by
  let values:=words.map fun w => (units w).getD 0
  let a:ℝ:=(acc words values:ℝ)/2^133
  let s:ℝ:=(exactSum values:ℝ)/2^133
  let y:=Model.toReal (Model.ofNatBits (FabricBF16V2.output states lane (coarseShift words) true) : Model FloatFormat.bfloat16)
  have hu : ∀w∈words,∃v,units w=some v := by
    intro w hw
    exact FabricBF16Decode.units_defined w (valid w hw).1 (FabricBF16Decode.finite_raw w (valid w hw).2)
  rw [FabricBF16Decode.sum_units_value words hu]
  change |y-s|≤|s| ∧ (y≠0 → 127*|y-s|≤2*|s|)
  have hv : states[lane]! =.finite (aligned words values) := by
    rw [hs]
    exact FabricBF16RoundBounds.accumulate_finite words (fun w hw => (valid w hw).2)
  have hr : |y-a|≤|a|/128 := output_sr_error states lane (coarseShift words) (aligned words values) hv hf
  by_cases hz : acc words values=0
  · have ha : a=0 := by simp [a,hz]
    have hy : y=0 := by rw [ha] at hr;simpa using hr
    simp [hy]
  · have hvn : values.length≤4 := by simpa [values] using hn
    have ha : 128*|a-s|≤|a| := unflushed_real words values hvn hz
    have habs := abs_sub_le a s 0
    have hsum := abs_sub_le y a s
    have hnonneg := abs_nonneg s
    simp only [sub_zero] at habs
    constructor
    · nlinarith
    · intro _
      nlinarith

theorem rne_relative (words : List Nat) (states : Array Accum) (lane : Nat)
    (valid : ∀w∈words,w<65536 ∧ w%32768<32640) (hn : words.length≤4)
    (hs : states[lane]! =accumulate words)
    (hf : Model.isFinite (Model.ofNatBits (FabricBF16V2.output states lane (coarseShift words) false) :
      Model FloatFormat.bfloat16)=true) :
    let y := Model.toReal (Model.ofNatBits (FabricBF16V2.output states lane (coarseShift words) false) : Model FloatFormat.bfloat16)
    let s := (words.map fun w => (Model.ofNatBits w : Model FloatFormat.bfloat16).toReal).sum
    |y-s|≤|s| ∧ (y≠0 → 254*|y-s|≤3*|s|) := by
  let values:=words.map fun w => (units w).getD 0
  let a:ℝ:=(acc words values:ℝ)/2^133
  let s:ℝ:=(exactSum values:ℝ)/2^133
  let y:=Model.toReal (Model.ofNatBits (FabricBF16V2.output states lane (coarseShift words) false) : Model FloatFormat.bfloat16)
  have hu : ∀w∈words,∃v,units w=some v := by
    intro w hw
    exact FabricBF16Decode.units_defined w (valid w hw).1 (FabricBF16Decode.finite_raw w (valid w hw).2)
  rw [FabricBF16Decode.sum_units_value words hu]
  change |y-s|≤|s| ∧ (y≠0 → 254*|y-s|≤3*|s|)
  have hv : states[lane]! =.finite (aligned words values) := by
    rw [hs]
    exact FabricBF16RoundBounds.accumulate_finite words (fun w hw => (valid w hw).2)
  have ho:=output_rne states lane (coarseShift words) (aligned words values) hv
  have hfinite : Model.isFinite (Model.ofNatBits (FabricBF16Block.round (acc words values)) : Model FloatFormat.bfloat16)=true := by
    simpa only [ho,acc] using hf
  have hr : |y-a|≤|a|/256 := by
    dsimp [y,a]
    rw [ho]
    exact round_relative (acc words values) hfinite
  by_cases hz : acc words values=0
  · have ha : a=0 := by simp [a,hz]
    have hy : y=0 := by rw [ha] at hr;simpa using hr
    simp [hy]
  · have hvn : values.length≤4 := by simpa [values] using hn
    have ha : 128*|a-s|≤|a| := unflushed_real words values hvn hz
    have habs := abs_sub_le a s 0
    have hsum := abs_sub_le y a s
    have hnonneg := abs_nonneg s
    simp only [sub_zero] at habs
    constructor
    · nlinarith
    · intro _
      nlinarith

#print axioms coarse_dominates
#print axioms unflushed_relative
#print axioms unflushed_real
#print axioms round_small_bits
#print axioms round_relative
#print axioms sr_relative
#print axioms rne_relative
end FabricBF16Relative

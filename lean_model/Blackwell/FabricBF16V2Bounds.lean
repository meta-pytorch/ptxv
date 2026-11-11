/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricBF16V2
import FabricBF16StochasticBounds
import FabricBF16Permutation

namespace FabricBF16V2Bounds
open FabricBF16Block FabricBF16V2 GB300Math.TensorAlignmentLoss
open FloatLib.Floats.Formats.BinaryInterchange FloatLib.Floats.Formats.Flocq

def acc (words : List Nat) (values : List Int) : Int :=
  flush (coarseShift words) (aligned words values)

def loss (words : List Nat) (values : List Int) : Nat :=
  values.length*(2^latticeShift words-1)+(2^coarseShift words-1)

theorem flush_loss (shift : Nat) (value : Int) :
    (flush shift value-value).natAbs ≤ 2^shift-1 := by
  unfold flush
  split
  next h => simp only [Int.zero_sub,Int.natAbs_neg];omega
  next h => simp

theorem accumulator_loss (words : List Nat) (values : List Int) :
    (acc words values-exactSum values).natAbs ≤ loss words values := by
  have ht := accumulation_bound words values
  have hf := flush_loss (coarseShift words) (aligned words values)
  have hb : (aligned words values-exactSum values).natAbs ≤
      values.length*(2^latticeShift words-1) := by omega
  have he : acc words values-exactSum values =
      (flush (coarseShift words) (aligned words values)-aligned words values)+
      (aligned words values-exactSum values) := by dsimp [acc];omega
  rw [he]
  exact (Int.natAbs_add_le _ _).trans (Nat.add_le_add hf hb) |>.trans_eq (Nat.add_comm _ _)

theorem accumulator_real_error (words : List Nat) (values : List Int) :
    |(acc words values:ℝ)/2^133-(exactSum values:ℝ)/2^133| ≤
      (loss words values:ℝ)/2^133 := by
  have h := accumulator_loss words values
  have hr : |((acc words values-exactSum values:Int):ℝ)| ≤ (loss words values:ℝ) := by
    have hn : (((acc words values-exactSum values).natAbs:Nat):ℝ)≤(loss words values:ℝ) := by
      exact_mod_cast h
    simpa using hn
  rw [Int.cast_sub] at hr
  rw [←sub_div,abs_div,abs_of_pos (by positivity : (0:ℝ)<2^133)]
  exact div_le_div_of_nonneg_right hr (by positivity)

theorem round_zero : FabricBF16Block.round 0 = 0 := by decide +kernel

theorem decoded_zero : Model.toReal (Model.ofNatBits 0 : Model FloatFormat.bfloat16)=0 := by
  have h := FabricBF16Decode.units_value 0 0 (by decide +kernel)
  simpa using h

theorem output_rne (states : Array Accum) (lane shift : Nat) (value : Int)
    (hv : states[lane]! = .finite value) :
    FabricBF16V2.output states lane shift false = FabricBF16Block.round (flush shift value) := by
  simp only [FabricBF16V2.output,hv,flush]
  split
  · exact round_zero.symm
  · exact FabricBF16RoundBounds.output_rne states lane value hv

theorem output_sr_error (states : Array Accum) (lane shift : Nat) (value : Int)
    (hv : states[lane]! = .finite value)
    (hf : Model.isFinite (Model.ofNatBits (FabricBF16V2.output states lane shift true) :
      Model FloatFormat.bfloat16)=true) :
    |Model.toReal (Model.ofNatBits (FabricBF16V2.output states lane shift true) :
      Model FloatFormat.bfloat16)-(flush shift value:ℝ)/2^133| ≤
      |(flush shift value:ℝ)/2^133|/128 := by
  by_cases h : value.natAbs < 2^shift
  · simp [FabricBF16V2.output,hv,h,flush,decoded_zero]
  · have ho : FabricBF16V2.output states lane shift true = FabricBF16Block.output states lane true := by
      simp [FabricBF16V2.output,hv,h]
    rw [ho] at hf ⊢
    simpa [flush,h] using FabricBF16StochasticBounds.decoded_round_error states lane value hv hf

noncomputable def roundingBudget (value : Int) (sr : Bool) : ℝ :=
  if sr then |(value:ℝ)/2^133|/128
  else ulp FloatLib.Numerics.binaryRadix (Model.fexpOf FloatFormat.bfloat16) ((value:ℝ)/2^133)/2

theorem output_error (states : Array Accum) (lane shift : Nat) (value : Int) (sr : Bool)
    (hv : states[lane]! = .finite value)
    (hf : Model.isFinite (Model.ofNatBits (FabricBF16V2.output states lane shift sr) :
      Model FloatFormat.bfloat16)=true) :
    |Model.toReal (Model.ofNatBits (FabricBF16V2.output states lane shift sr) :
      Model FloatFormat.bfloat16)-(flush shift value:ℝ)/2^133| ≤ roundingBudget (flush shift value) sr := by
  cases sr
  · rw [output_rne states lane shift value hv] at hf ⊢
    exact FabricBF16RoundBounds.round_error _ hf
  · exact output_sr_error states lane shift value hv hf

theorem decoded_model_error (words : List Nat) (states : Array Accum) (lane : Nat) (sr : Bool)
    (valid : ∀ w∈words,w<65536 ∧ w%32768<32640)
    (hs : states[lane]! = accumulate words)
    (hf : Model.isFinite (Model.ofNatBits (FabricBF16V2.output states lane (coarseShift words) sr) :
      Model FloatFormat.bfloat16)=true) :
    let values := words.map fun w => (units w).getD 0
    |Model.toReal (Model.ofNatBits (FabricBF16V2.output states lane (coarseShift words) sr) :
      Model FloatFormat.bfloat16)-
      (words.map fun w => (Model.ofNatBits w : Model FloatFormat.bfloat16).toReal).sum| ≤
      (loss words values:ℝ)/2^133+roundingBudget (acc words values) sr := by
  let values := words.map fun w => (units w).getD 0
  have hv : states[lane]! = .finite (aligned words values) := by
    rw [hs]
    exact FabricBF16RoundBounds.accumulate_finite words (fun w hw => (valid w hw).2)
  have hr := output_error states lane (coarseShift words) (aligned words values) sr hv hf
  have ha := accumulator_real_error words values
  have hu : ∀w∈words,∃v,units w=some v := by
    intro w hw
    exact FabricBF16Decode.units_defined w (valid w hw).1 (FabricBF16Decode.finite_raw w (valid w hw).2)
  rw [FabricBF16Decode.sum_units_value words hu]
  have ht := abs_sub_le
    (Model.toReal (Model.ofNatBits (FabricBF16V2.output states lane (coarseShift words) sr) : Model FloatFormat.bfloat16))
    ((acc words values:ℝ)/2^133) ((exactSum values:ℝ)/2^133)
  change _ ≤ (loss words values:ℝ)/2^133+roundingBudget (acc words values) sr
  dsimp [acc,values] at *
  linarith

theorem coarseShift_perm {xs ys : List Nat} (hp : xs.Perm ys) :
    coarseShift xs = coarseShift ys := by simp [coarseShift,FabricBF16Permutation.maxExponent_perm hp]

#print axioms flush_loss
#print axioms accumulator_loss
#print axioms accumulator_real_error
#print axioms round_zero
#print axioms decoded_zero
#print axioms output_rne
#print axioms output_sr_error
#print axioms output_error
#print axioms decoded_model_error
#print axioms coarseShift_perm
end FabricBF16V2Bounds

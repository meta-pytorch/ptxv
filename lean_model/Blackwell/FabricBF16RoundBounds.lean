/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricBF16CancellationBounds
import FloatLib.Floats.Formats.BinaryInterchange.Rounding.Proof
import FloatLib.Floats.Formats.Flocq.Theory.Error.Bounds
namespace FabricBF16RoundBounds
open FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats.Formats.Flocq
open GB300Math.TensorAlignmentLoss

def asDyadic (value : Int) : FloatLib.Numerics.Dyadic :=
  {negative := value < 0, significand := value.natAbs, exponent := -133}

theorem dyadic_value (value : Int) : (asDyadic value).toReal = (value : ℝ)/(2^133 : ℝ) := by
  cases value <;>
    simp [asDyadic, FloatLib.Numerics.Dyadic.toReal, FloatLib.Numerics.Dyadic.signedSignificand,
      zpow_neg, div_eq_mul_inv]

theorem round_word (value : Int) :
    (Model.ofNatBits (FabricBF16Block.round value) : Model FloatFormat.bfloat16) =
      Model.roundDyadic FloatFormat.bfloat16 (asDyadic value) := by
  unfold FabricBF16Block.round
  exact Model.ofNatBits_toNatBits _

theorem round_error (value : Int)
    (finite : Model.isFinite (Model.ofNatBits (FabricBF16Block.round value) : Model FloatFormat.bfloat16) = true) :
    |Model.toReal (Model.ofNatBits (FabricBF16Block.round value) : Model FloatFormat.bfloat16) -
        (value : ℝ)/(2^133 : ℝ)| ≤
      ulp FloatLib.Numerics.binaryRadix (Model.fexpOf FloatFormat.bfloat16)
        ((value : ℝ)/(2^133 : ℝ))/2 := by
  rw [round_word] at finite ⊢
  rw [Model.toReal_roundDyadic_eq_roundAt _ (by decide) _ finite, dyadic_value]
  exact error_bound_ulp (rnd := nearestEven) _

def acc (words : List Nat) (values : List Int) : Int :=
  FabricBF16Block.aligned words values

def loss (words : List Nat) (values : List Int) : Nat :=
  values.length*(2^FabricBF16Block.latticeShift words-1)

theorem accumulator_real_error (words : List Nat) (values : List Int) :
    |(acc words values : ℝ)/(2^133 : ℝ) - (exactSum values : ℝ)/(2^133 : ℝ)| ≤
      (loss words values : ℝ)/(2^133 : ℝ) := by
  have hl := FabricBF16Block.accumulation_bound words values
  have h : (acc words values - exactSum values).natAbs ≤ loss words values := by
    dsimp [acc, loss]
    omega
  have hr : |((acc words values - exactSum values : Int) : ℝ)| ≤ (loss words values : ℝ) := by
    have hn : (((acc words values-exactSum values).natAbs : Nat) : ℝ) ≤ (loss words values : ℝ) := by
      exact_mod_cast h
    simpa using hn
  rw [Int.cast_sub] at hr
  rw [← sub_div, abs_div, abs_of_pos (by positivity : (0:ℝ)<2^133)]
  exact div_le_div_of_nonneg_right hr (by positivity)

theorem final_error (words : List Nat) (values : List Int)
    (finite : Model.isFinite (Model.ofNatBits (FabricBF16Block.round (acc words values)) : Model FloatFormat.bfloat16) = true) :
    |Model.toReal (Model.ofNatBits (FabricBF16Block.round (acc words values)) : Model FloatFormat.bfloat16) -
        (exactSum values : ℝ)/(2^133 : ℝ)| ≤
      (loss words values : ℝ)/(2^133 : ℝ) +
      ulp FloatLib.Numerics.binaryRadix (Model.fexpOf FloatFormat.bfloat16)
        ((acc words values : ℝ)/(2^133 : ℝ))/2 := by
  have h1 := round_error (acc words values) finite
  have h2 := accumulator_real_error words values
  calc
    _ ≤ |Model.toReal (Model.ofNatBits (FabricBF16Block.round (acc words values)) : Model FloatFormat.bfloat16) -
      (acc words values : ℝ)/(2^133 : ℝ)| +
      |(acc words values : ℝ)/(2^133 : ℝ) - (exactSum values : ℝ)/(2^133 : ℝ)| := abs_sub_le _ _ _
    _ ≤ _ := by linarith

theorem accumulate_finite (words : List Nat)
    (finite_inputs : ∀ w ∈ words, w % 32768 < 32640) :
    FabricBF16Block.accumulate words = .finite
      (acc words (words.map fun w => (FabricBF16Block.units w).getD 0)) := by
  have hn : ∀ w ∈ words, ¬ 32640 < w % 32768 := by
    intro w hw
    have := finite_inputs w hw
    omega
  have hi : ∀ w ∈ words, ¬ w % 32768 = 32640 := by
    intro w hw
    have := finite_inputs w hw
    omega
  have hnf : words.filter (fun w => decide (w % 32768 > 32640)) = [] := by
    simp only [List.filter_eq_nil_iff, Bool.not_eq_true, decide_eq_false_iff_not]
    exact hn
  have hif : words.filter (fun w => decide (w % 32768 = 32640)) = [] := by
    simp only [List.filter_eq_nil_iff, Bool.not_eq_true, decide_eq_false_iff_not]
    exact hi
  simp [FabricBF16Block.accumulate, hnf, hif, acc]

theorem output_rne (states : Array FabricBF16Block.Accum) (lane : Nat) (value : Int)
    (finite_state : states[lane]! = .finite value) :
    FabricBF16Block.output states lane false = FabricBF16Block.round value := by
  simp [FabricBF16Block.output, finite_state]

theorem finite_model_error (words : List Nat)
    (finite_inputs : ∀ w ∈ words, w % 32768 < 32640)
    (finite_output : Model.isFinite (Model.ofNatBits
      (FabricBF16Block.output #[FabricBF16Block.accumulate words] 0 false) :
      Model FloatFormat.bfloat16) = true) :
    let values := words.map fun w => (FabricBF16Block.units w).getD 0
    |Model.toReal (Model.ofNatBits
      (FabricBF16Block.output #[FabricBF16Block.accumulate words] 0 false) :
      Model FloatFormat.bfloat16) - (exactSum values : ℝ)/(2^133 : ℝ)| ≤
      (loss words values : ℝ)/(2^133 : ℝ) +
      ulp FloatLib.Numerics.binaryRadix (Model.fexpOf FloatFormat.bfloat16)
        ((acc words values : ℝ)/(2^133 : ℝ))/2 := by
  have ho : FabricBF16Block.output #[FabricBF16Block.accumulate words] 0 false =
      FabricBF16Block.round (acc words (words.map fun w => (FabricBF16Block.units w).getD 0)) := by
    apply output_rne
    simpa using accumulate_finite words finite_inputs
  rw [ho] at finite_output ⊢
  exact final_error words _ finite_output

theorem midpoint_rounding_bits : FabricBF16Block.round 257 = 256 := by decide +kernel

theorem midpoint_rounding_error :
    (Model.ofNatBits (FabricBF16Block.round 257) : Model FloatFormat.bfloat16).toRat? =
      some ((257:Rat)/2^133 - 1/2^133) := by decide +kernel

#print axioms accumulate_finite
#print axioms output_rne
#print axioms finite_model_error
#print axioms midpoint_rounding_bits
#print axioms midpoint_rounding_error
#print axioms accumulator_real_error
#print axioms final_error
#print axioms dyadic_value
#print axioms round_word
#print axioms round_error
end FabricBF16RoundBounds

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.IEEEComparison.Keys

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.IEEEComparison

@[simp] theorem magnitude_mask (bits : U32) :
    (bits &&& 0x7fffffff#u32).val = bits.val % signBase := by
  simp only [UScalar.val_and]
  change bits.val &&& (2 ^ 31 - 1) = bits.val % 2 ^ 31
  exact Nat.and_two_pow_sub_one_eq_mod bits.val 31

def normalizedWord (bits : U32) (flush : Bool) : U32 :=
  if flush then
    if bits &&& 0x7fffffff#u32 < 0x00800000#u32 then 0#u32 else bits &&& 0x7fffffff#u32
  else bits &&& 0x7fffffff#u32

@[simp] theorem normalized_word_val (bits : U32) (flush : Bool) :
    (normalizedWord bits flush).val = normalizedMagnitude (bits.val % signBase) flush := by
  have test : (bits &&& 0x7fffffff#u32 < 0x00800000#u32) ↔
      bits.val % signBase < fractionBase := by
    change (bits &&& 0x7fffffff#u32).val < 8388608 ↔ _
    rw [magnitude_mask]
  cases flush with
  | false => exact magnitude_mask bits
  | true =>
    by_cases small : bits &&& 0x7fffffff#u32 < 0x00800000#u32
    · have naturalSmall := test.mp small
      simp only [normalizedWord, normalizedMagnitude, small, naturalSmall,
        decide_true, Bool.and_true, ite_true]
      rfl
    · have naturalSmall : ¬ bits.val % signBase < fractionBase := fun h => small (test.mpr h)
      simp only [normalizedWord, normalizedMagnitude, small, naturalSmall,
        decide_false, Bool.and_false, ite_false]
      exact magnitude_mask bits

@[step] theorem source_classify (bits : U32) (flush : Bool) :
    PtxvSim.fp.comparison.classify bits flush ⦃ classified =>
      classified.1.val = signedKey (decide (bits.val ≥ signBase))
        (normalizedMagnitude (bits.val % signBase) flush) ∧
      classified.2 = decide (infinityMagnitude < normalizedMagnitude (bits.val % signBase) flush) ⦄ := by
  have normalizedBound : (normalizedWord bits flush).val ≤ 2147483647 := by
    rw [normalized_word_val]
    unfold normalizedMagnitude
    split <;> simp_all
    have bound := Nat.mod_lt bits.val (show 0 < signBase by decide)
    dsimp only [signBase] at *
    omega
  unfold PtxvSim.fp.comparison.classify
  simp only [lift, bind_ok]
  have normalization :
      (if flush then
        if bits &&& 2147483647#u32 < 8388608#u32 then .ok 0#u32 else .ok (bits &&& 2147483647#u32)
      else .ok (bits &&& 2147483647#u32) : Result U32) = .ok (normalizedWord bits flush) := by
    cases flush <;> simp [normalizedWord]
    split <;> rfl
  rw [normalization]
  simp only [bind_ok]
  have cast : (UScalar.hcast .I32 (normalizedWord bits flush)).val = (normalizedWord bits flush).val := by
    have specification := UScalar.hcast_inBounds_spec .I32 (normalizedWord bits flush) (by scalar_tac)
    simpa only [lift, WP.spec_ok] using specification
  by_cases negative : bits ≥ 2147483648#u32
  · simp only [negative, ite_true]
    step
    simp only [signedKey]
    simp_scalar at negative
    simp_all [signBase, infinityMagnitude, normalized_word_val]
  · simp only [negative, ite_false]
    simp only [signedKey]
    simp_scalar at negative
    simp_all [signBase, infinityMagnitude, normalized_word_val]

@[step] theorem source_compare_spec (first second : U32) (mode : PtxvSim.ptxv_ir.FCmp) (flush : Bool) :
    PtxvSim.fp.comparison.compare first second mode flush ⦃ answer => answer = compare mode first second flush ⦄ := by
  unfold PtxvSim.fp.comparison.compare
  step with source_classify first flush as ⟨firstKey, firstNan, firstValue, firstClass⟩
  step with source_classify second flush as ⟨secondKey, secondNan, secondValue, secondClass⟩
  have expected : keyTruth mode firstKey.val secondKey.val (firstNan || secondNan) =
      compare mode first second flush := by
    rw [firstValue, secondValue, firstClass, secondClass]
    simp only [← Bool.decide_or]
    exact comparison_keys_exact mode first second flush
  rw [← expected]
  cases mode <;> cases firstNan <;> cases secondNan <;>
    simp [keyTruth] <;> simp_scalar
  all_goals first
    | exact IScalar.eq_equiv _ _
    | apply Bool.eq_iff_iff.mpr; simp [IScalar.eq_equiv]

theorem total_result_eq {α : Type} (computation : Result α) (value : α)
    (specification : computation ⦃ returned => returned = value ⦄) : computation = .ok value := by
  cases computation using Result.cases with
  | ret actual =>
    have same := (WP.spec_ok actual).mp specification
    rw [same]
  | vis effect continuation => exact False.elim ((WP.spec_vis effect continuation).mp specification)
  | div => exact False.elim (WP.spec_div.mp specification)

@[simp] theorem source_compare (first second : U32) (mode : PtxvSim.ptxv_ir.FCmp) (flush : Bool) :
    PtxvSim.fp.comparison.compare first second mode flush = .ok (compare mode first second flush) :=
  total_result_eq _ _ (source_compare_spec first second mode flush)

@[simp] theorem source_f32_cmp (first second : U32) (mode : PtxvSim.ptxv_ir.FCmp) (flush : Bool) :
    PtxvSim.fp.f32_cmp first second mode flush = .ok (compare mode first second flush) := by
  exact source_compare first second mode flush

end Instruction.IEEEComparison

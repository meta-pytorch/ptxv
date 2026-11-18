/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourcePrimitives

open Aeneas Aeneas.Std Result
open PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

@[simp]
theorem nat_mask32 (value : Nat) : value &&& 4294967295 = value % 4294967296 :=
  Nat.and_two_pow_sub_one_eq_mod value 32

@[simp]
theorem scalar_mask64 (value : U64) : value &&& core.num.U64.MAX = value := by
  apply UScalar.eq_of_val_eq
  change value.val &&& (2 ^ 64 - 1) = value.val
  rw [Nat.and_two_pow_sub_one_eq_mod]
  exact Nat.mod_eq_of_lt (show value.val < 2 ^ 64 from value.bv.isLt)

@[simp]
theorem nat_mask64 (value : U64) : value.val &&& U64.rMax = value.val := by
  change value.val &&& (2 ^ 64 - 1) = value.val
  rw [Nat.and_two_pow_sub_one_eq_mod]
  exact Nat.mod_eq_of_lt (show value.val < 2 ^ 64 from value.bv.isLt)

@[simp]
theorem scalar_eq_value {ty : UScalarTy} (left right : UScalar ty) :
    left = right ↔ left.val = right.val :=
  ⟨congrArg UScalar.val, UScalar.eq_of_val_eq⟩

@[simp]
theorem scalar_bne_value {ty : UScalarTy} (left right : UScalar ty) :
    (left != right) = (left.val != right.val) := by
  apply Bool.eq_iff_iff.mpr
  simp only [bne_iff_ne]
  exact not_congr (scalar_eq_value left right)

def comparisonResult (comparison : V.Comparison) (left right : U64) (signedLeft signedRight : I64) : Bool :=
  match comparison with
  | .Equal => left = right
  | .NotEqual => left != right
  | .LessUnsigned => left < right
  | .LessEqualUnsigned => left ≤ right
  | .GreaterUnsigned => left > right
  | .GreaterEqualUnsigned => left ≥ right
  | .LessSigned => signedLeft < signedRight
  | .LessEqualSigned => signedLeft ≤ signedRight
  | .GreaterSigned => signedLeft > signedRight
  | .GreaterEqualSigned => signedLeft ≥ signedRight

def comparisonMeaning (comparison : V.Comparison) (left right : Nat)
    (signedLeft signedRight : Int) : Bool :=
  match comparison with
  | .Equal => left = right
  | .NotEqual => left != right
  | .LessUnsigned => left < right
  | .LessEqualUnsigned => left ≤ right
  | .GreaterUnsigned => left > right
  | .GreaterEqualUnsigned => left ≥ right
  | .LessSigned => signedLeft < signedRight
  | .LessEqualSigned => signedLeft ≤ signedRight
  | .GreaterSigned => signedLeft > signedRight
  | .GreaterEqualSigned => signedLeft ≥ signedRight

theorem comparison_meaning_model (comparison : V.Comparison) (left right : Nat)
    (signedLeft signedRight : Int) :
    comparisonMeaning comparison left right signedLeft signedRight =
      comparisonValue (comparisonModel comparison) left right signedLeft signedRight := by
  cases comparison <;> rfl

theorem comparison_result_value (comparison : V.Comparison) (left right : U64)
    (signedLeft signedRight : I64) :
    comparisonResult comparison left right signedLeft signedRight =
      comparisonMeaning comparison left.val right.val signedLeft.val signedRight.val := by
  cases comparison <;> apply Bool.eq_iff_iff.mpr <;>
    simp only [comparisonResult, comparisonMeaning, decide_eq_true_eq, bne_iff_ne,
      ne_eq, scalar_eq_value, UScalar.lt_equiv, UScalar.le_equiv,
      IScalar.lt_equiv, IScalar.le_equiv]

theorem source_compare32_eq (comparison : V.Comparison) (left right : U64) :
    value_flow.compare comparison 32#u8 left right = .ok (.Ok (comparisonResult comparison
      (left &&& 4294967295#u64) (right &&& 4294967295#u64)
      (IScalar.cast .I64 (UScalar.hcast .I32 (UScalar.cast .U32 (left &&& 4294967295#u64))))
      (IScalar.cast .I64 (UScalar.hcast .I32 (UScalar.cast .U32 (right &&& 4294967295#u64)))))) := by
  cases comparison <;> simp [value_flow.compare, comparisonResult, lift, bind_ok]

theorem source_compare64_eq (comparison : V.Comparison) (left right : U64) :
    value_flow.compare comparison 64#u8 left right = .ok (.Ok (comparisonResult comparison
      (left &&& core.num.U64.MAX) (right &&& core.num.U64.MAX)
      (UScalar.hcast .I64 (left &&& core.num.U64.MAX))
      (UScalar.hcast .I64 (right &&& core.num.U64.MAX)))) := by
  cases comparison <;> simp [value_flow.compare, comparisonResult, lift, bind_ok]

theorem signed32_value (value : U64) :
    (IScalar.cast .I64 (UScalar.hcast .I32 (UScalar.cast .U32 (value &&& 4294967295#u64)))).val =
      integerSigned 32 (value.val % 4294967296) := by
  rw [IScalar.val_mod_pow_greater_numBits .I64 _ (by decide)]
  simp only [UScalar.hcast_val_eq, UScalar.cast_val_eq, UScalarTy.numBits,
    IScalarTy.numBits, UScalar.val_and, show (4294967295#u64).val = 4294967295 from rfl,
    nat_mask32, show (2 ^ 32 : Nat) = 4294967296 from rfl, Nat.mod_mod, integerSigned]

theorem signed64_value (value : U64) :
    (UScalar.hcast .I64 value).val = integerSigned 64 value.val := by
  simp only [UScalar.hcast_val_eq, IScalarTy.numBits, integerSigned]

theorem compare32_meaning (comparison : V.Comparison) (left right : U64) :
    some (comparisonResult comparison (left &&& 4294967295#u64) (right &&& 4294967295#u64)
      (IScalar.cast .I64 (UScalar.hcast .I32 (UScalar.cast .U32 (left &&& 4294967295#u64))))
      (IScalar.cast .I64 (UScalar.hcast .I32 (UScalar.cast .U32 (right &&& 4294967295#u64))))) =
      compare (comparisonModel comparison) 32 left.val right.val := by
  rw [comparison_result_value, signed32_value, signed32_value, comparison_meaning_model]
  simp only [UScalar.val_and, show (4294967295#u64).val = 4294967295 from rfl,
    nat_mask32]
  have supported : scalarBits 32 = true := rfl
  simp only [compare, supported, ite_true, truncate,
    show (2 ^ 32 : Nat) = 4294967296 from rfl]

theorem compare64_meaning (comparison : V.Comparison) (left right : U64) :
    some (comparisonResult comparison left right (UScalar.hcast .I64 left) (UScalar.hcast .I64 right)) =
      compare (comparisonModel comparison) 64 left.val right.val := by
  rw [comparison_result_value, signed64_value, signed64_value, comparison_meaning_model]
  have leftMod : truncate 64 left.val = left.val :=
    Nat.mod_eq_of_lt (show left.val < 2 ^ 64 from left.bv.isLt)
  have rightMod : truncate 64 right.val = right.val :=
    Nat.mod_eq_of_lt (show right.val < 2 ^ 64 from right.bv.isLt)
  have supported : scalarBits 64 = true := rfl
  simp only [compare, supported, ite_true, leftMod, rightMod]

@[step]
theorem source_compare (comparison : V.Comparison) (bits : U8) (left right : U64) :
    value_flow.compare comparison bits left right ⦃ result =>
      outcomeModel id result = compare (comparisonModel comparison) bits.val left.val right.val ⦄ := by
  by_cases width32 : bits = 32#u8
  · subst bits
    rw [source_compare32_eq]
    simp only [WP.spec_ok, outcomeModel, id_eq]
    exact compare32_meaning comparison left right
  · by_cases width64 : bits = 64#u8
    · subst bits
      rw [source_compare64_eq]
      simp only [WP.spec_ok, outcomeModel, id_eq, scalar_mask64]
      exact compare64_meaning comparison left right
    · have no32 : bits.val ≠ 32 := by scalar_tac
      have no64 : bits.val ≠ 64 := by scalar_tac
      simp [value_flow.compare, outcomeModel, compare, scalarBits, no32, no64]

end ValueFlow.Source

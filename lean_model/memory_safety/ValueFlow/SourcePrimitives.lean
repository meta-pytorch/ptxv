/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceModel
import MemorySafety.Code.Funs

open Aeneas Aeneas.Std Result
open PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

@[step]
theorem source_width_mask (bits : U8) :
    value_flow.width_mask bits ⦃ result => outcomeModel UScalar.val result =
      if supportedBits bits.val then some (2 ^ bits.val - 1) else none ⦄ := by
  have v8 : ((8#8#uscalar : U8)).val = 8 := rfl
  have v16 : ((16#8#uscalar : U8)).val = 16 := rfl
  have v32 : ((32#8#uscalar : U8)).val = 32 := rfl
  have v64 : ((64#8#uscalar : U8)).val = 64 := rfl
  unfold value_flow.width_mask
  split
  · simp only [WP.spec_ok, outcomeModel, supportedBits, v8]
    decide
  · simp only [WP.spec_ok, outcomeModel, supportedBits, v16]
    decide
  · simp only [WP.spec_ok, outcomeModel, supportedBits, v32]
    decide
  · simp only [WP.spec_ok, outcomeModel, supportedBits, v64]
    decide
  · simp_all [WP.spec_ok, outcomeModel, supportedBits]

@[simp]
theorem source_width_mask32 : value_flow.width_mask 32#u8 = .ok (.Ok 4294967295#u64) := by rfl

@[simp]
theorem source_width_mask64 : value_flow.width_mask 64#u8 = .ok (.Ok core.num.U64.MAX) := by rfl

@[step]
theorem source_scalar (value : V.Value) :
    value_flow.scalar value ⦃ result => outcomeModel UScalar.val result = scalarValue (valueModel value) ⦄ := by
  cases value <;> simp [value_flow.scalar, outcomeModel, scalarValue, valueModel]

@[step]
theorem source_offset_pointer (pointer : V.Pointer) (amount : U64) (subtract : Bool) :
    value_flow.offset_pointer pointer amount subtract ⦃ result =>
      outcomeModel pointerModel result =
        offsetPointer (pointerModel pointer) (if subtract then -(amount.val : Int) else amount.val) ⦄ := by
  cases subtract with
  | false =>
    have checked := U64.checked_add_bv_spec pointer.offset amount
    cases found : U64.checked_add pointer.offset amount with
    | none =>
      simp only [found] at checked
      have outside : ¬ (0 ≤ (pointer.offset.val : Int) + amount.val ∧
          (pointer.offset.val : Int) + amount.val < 2 ^ 64) := by
        have high : 18446744073709551615 < pointer.offset.val + amount.val := by simpa only [U64.max_eq] using checked
        omega
      simp [value_flow.offset_pointer, found, outcomeModel, pointerModel,
        offsetPointer]; omega
    | some offset =>
      simp only [found] at checked
      have inside : 0 ≤ (pointer.offset.val : Int) + amount.val ∧
          (pointer.offset.val : Int) + amount.val < 2 ^ 64 := by
        have high : pointer.offset.val + amount.val ≤ 18446744073709551615 := by simpa only [U64.max_eq] using checked.1
        omega
      have result : ((pointer.offset.val : Int) + amount.val).toNat = offset.val := by
        omega
      simp [value_flow.offset_pointer, found, outcomeModel, pointerModel,
        offsetPointer, result]; omega
  | true =>
    have checked := U64.checked_sub_bv_spec pointer.offset amount
    have bound : pointer.offset.val < 18446744073709551616 := pointer.offset.bv.isLt
    cases found : U64.checked_sub pointer.offset amount with
    | none =>
      simp only [found] at checked
      have outside : ¬ (0 ≤ (pointer.offset.val : Int) - amount.val ∧
          (pointer.offset.val : Int) - amount.val < 2 ^ 64) := by omega
      simp [value_flow.offset_pointer, found, outcomeModel, pointerModel,
        offsetPointer, ←sub_eq_add_neg]; omega
    | some offset =>
      simp only [found] at checked
      have inside : 0 ≤ (pointer.offset.val : Int) - amount.val ∧
          (pointer.offset.val : Int) - amount.val < 2 ^ 64 := by omega
      have result : ((pointer.offset.val : Int) - amount.val).toNat = offset.val := by omega
      simp [value_flow.offset_pointer, found, outcomeModel, pointerModel,
        offsetPointer, ←sub_eq_add_neg, result]; omega

end ValueFlow.Source

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceOperand
import ValueFlow.SourceBinary

open Aeneas Aeneas.Std Result
open PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

theorem signed_offset_nonnegative (offset : I64) (nonnegative : 0 ≤ offset.val) :
    ((IScalar.hcast .U64 offset).val : Int) = offset.val := by
  have bound : offset.val < 18446744073709551616 := by scalar_tac
  simp only [IScalar.hcast_val_eq, UScalarTy.numBits]
  norm_num only at *
  rw [Int.emod_eq_of_lt nonnegative bound, Int.toNat_of_nonneg nonnegative]

theorem signed_offset_negative (offset : I64) (negative : offset.val < 0) :
    -((core.num.U64.wrapping_sub 0#u64 (IScalar.hcast .U64 offset)).val : Int) = offset.val := by
  have lower : -9223372036854775808 ≤ offset.val := by scalar_tac
  simp only [core.num.U64.wrapping_sub_val_eq, IScalar.hcast_val_eq,
    UScalar.size_def, UScalarTy.numBits, show (0#u64).val = 0 from rfl, zero_add]
  norm_num only at *
  omega

@[step]
theorem source_signed_offset (pointer : V.Pointer) (offset : I64) :
    (do
      if offset >= 0#i64 then
        let amount ← lift (IScalar.hcast .U64 offset)
        value_flow.offset_pointer pointer amount false
      else
        let amount ← lift (IScalar.hcast .U64 offset)
        let magnitude ← lift (core.num.U64.wrapping_sub 0#u64 amount)
        value_flow.offset_pointer pointer magnitude true) ⦃ result =>
      outcomeModel pointerModel result = offsetPointer (pointerModel pointer) offset.val ⦄ := by
  split
  · rename_i nonnegative
    have positive : 0 ≤ offset.val := by scalar_tac
    simp only [lift, bind_ok]
    apply WP.spec_mono (source_offset_pointer pointer (IScalar.hcast .U64 offset) false)
    intro result meaning
    simpa only [Bool.false_eq_true, ite_false, signed_offset_nonnegative offset positive] using meaning
  · rename_i negative
    have below : offset.val < 0 := by scalar_tac
    simp only [lift, bind_ok]
    apply WP.spec_mono
      (source_offset_pointer pointer (core.num.U64.wrapping_sub 0#u64 (IScalar.hcast .U64 offset)) true)
    intro result meaning
    simpa only [ite_true, signed_offset_negative offset below] using meaning

@[step]
theorem source_address (registers : V.Registers) (ids : V.ThreadIds) (location : V.Address) :
    value_flow.address registers ids location ⦃ result =>
      outcomeModel pointerModel result = address (bankModel registers) (idsModel ids) (addressModel location) ⦄ := by
  unfold value_flow.address
  step as ⟨base, baseMeaning⟩
  cases base with
  | Err error =>
    simp only [outcomeModel] at baseMeaning
    simp [outcomeModel, address, addressModel, ←baseMeaning]
  | Ok base =>
    simp only [outcomeModel] at baseMeaning
    step as ⟨index, indexMeaning⟩
    cases index with
    | Err error =>
      simp only [outcomeModel] at indexMeaning
      simp [outcomeModel, address, addressModel, ←baseMeaning, ←indexMeaning]
    | Ok index =>
      simp only [outcomeModel] at indexMeaning
      step as ⟨value, valueMeaning⟩
      cases value with
      | Err error =>
        simp only [outcomeModel, binaryModel] at valueMeaning
        simp only [WP.spec_ok, outcomeModel, address, addressModel, ←baseMeaning, ←indexMeaning,
          Option.bind_eq_bind, Option.bind_some, ←valueMeaning, Option.bind_none]
      | Ok value =>
        simp only [outcomeModel, binaryModel] at valueMeaning
        cases value with
        | Scalar scalar =>
          change some (Value.scalar scalar.val) = binary .add 64 (valueModel base) (valueModel index) at valueMeaning
          simp only [WP.spec_ok, outcomeModel, address, addressModel, ←baseMeaning, ←indexMeaning,
            Option.bind_eq_bind, Option.bind_some, ←valueMeaning, Option.bind_none]
        | Pointer pointer =>
          change some (Value.pointer (pointerModel pointer)) =
            binary .add 64 (valueModel base) (valueModel index) at valueMeaning
          apply WP.spec_mono (source_signed_offset pointer location.offset)
          intro result meaning
          simpa only [address, addressModel, ←baseMeaning, ←indexMeaning,
            Option.bind_eq_bind, Option.bind_some, ←valueMeaning, Option.bind_none] using meaning

end ValueFlow.Source

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceVirtual

open Aeneas Aeneas.Std Result
open PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

@[step]
theorem source_read_operand (registers : V.Registers) (ids : V.ThreadIds)
    (operand : V.Operand) (bits : U8) :
    value_flow.read_operand registers ids operand bits ⦃ result =>
      outcomeModel valueModel result =
        readOperand (bankModel registers) (idsModel ids) bits.val (operandModel operand) ⦄ := by
  unfold value_flow.read_operand
  step as ⟨maskResult, maskMeaning⟩
  cases maskResult with
  | Err error =>
    have unsupported : supportedBits bits.val = false := by
      cases supported : supportedBits bits.val <;> simp_all [outcomeModel]
    simp [outcomeModel, readOperand, unsupported]
  | Ok mask =>
    have supported : supportedBits bits.val = true := by
      cases supported : supportedBits bits.val <;> simp_all [outcomeModel]
    have maskVal : mask.val = 2 ^ bits.val - 1 := by
      simpa only [outcomeModel, supported, ite_true, Option.some.injEq] using maskMeaning
    cases operand with
    | Immediate value =>
      simp only [lift, bind_ok, WP.spec_ok, outcomeModel, valueModel, readOperand, operandModel,
        supported, ite_true, UScalar.val_and, maskVal, Nat.and_two_pow_sub_one_eq_mod, truncate]
    | Virtual index =>
      apply WP.spec_mono (source_read_virtual registers index bits)
      intro result meaning
      simpa only [readOperand, supported, ite_true, operandModel, bankModel] using meaning
    | Pointer pointer =>
      dsimp only
      have widthEq : (bits = 64#u8) ↔ bits.val = 64 := by
        rw [UScalar.eq_equiv]
        rfl
      split
      all_goals
        rename_i width
        simp only [WP.spec_ok, outcomeModel, valueModel, readOperand, operandModel,
          supported, ite_true]
        simp only [←widthEq, width, show (64#u8).val = 64 from rfl, ite_true, ite_false]
    | Special special =>
      step as ⟨specialResult, specialMeaning⟩
      cases specialResult with
      | Err error =>
        simp only [outcomeModel] at specialMeaning
        simp [outcomeModel, readOperand, supported, operandModel, ←specialMeaning]
      | Ok value =>
        simp only [outcomeModel] at specialMeaning
        simp [lift, bind_ok, outcomeModel, valueModel, readOperand, supported, operandModel, ←specialMeaning,
          truncate, UScalar.val_and, maskVal, Nat.and_two_pow_sub_one_eq_mod]

end ValueFlow.Source

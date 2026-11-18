/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceArithmetic

open Aeneas Aeneas.Std Result
open PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

@[step]
theorem source_binary (operation : V.BinaryOperation) (bits : U8) (left right : V.Value) :
    value_flow.binary operation bits left right ⦃ result =>
      outcomeModel valueModel result =
        binary (binaryModel operation) bits.val (valueModel left) (valueModel right) ⦄ := by
  cases left with
  | Scalar left =>
    cases right with
    | Scalar right =>
      simp only [value_flow.binary]
      step as ⟨result, meaning⟩
      cases result <;> simp_all [outcomeModel, valueModel, binary]
    | Pointer pointer =>
      by_cases width : bits = 64#u8
      · subst bits
        cases operation <;>
          simp [value_flow.binary, value_flow.BinaryOperation.Insts.CoreCmpPartialEqBinaryOperation.eq,
            value_flow.BinaryOperation.read_discriminant,
            outcomeModel, valueModel, binary, binaryModel]
        step as ⟨result, meaning⟩
        cases result <;> simp_all [outcomeModel, valueModel]
      · have no64 : bits.val ≠ 64 := by scalar_tac
        simp [value_flow.binary, width, outcomeModel, valueModel, binary, no64]
  | Pointer pointer =>
    cases right with
    | Scalar right =>
      by_cases width : bits = 64#u8
      · subst bits
        cases operation <;>
          simp [value_flow.binary, outcomeModel, valueModel, binary, binaryModel]
        all_goals
          step as ⟨result, meaning⟩
          cases result <;> simp_all [outcomeModel, valueModel]
      · have no64 : bits.val ≠ 64 := by scalar_tac
        simp [value_flow.binary, width, outcomeModel, valueModel, binary, no64]
    | Pointer right => simp [value_flow.binary, outcomeModel, valueModel, binary]

end ValueFlow.Source

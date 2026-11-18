/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceAddress
import ValueFlow.SourcePredicate
import ValueFlow.Properties

open Aeneas Aeneas.Std Result
open PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 8000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

theorem source_scalar_operand_bound (registers : V.Registers) (ids : V.ThreadIds)
    (operand : V.Operand) (bits : Nat) (value : V.Value) (scalar : U64)
    (read : some (valueModel value) = readOperand (bankModel registers) (idsModel ids) bits (operandModel operand))
    (extracted : some scalar.val = scalarValue (valueModel value)) : scalar.val < 2 ^ bits := by
  cases value with
  | Scalar raw =>
    have same : scalar.val = raw.val := by simpa [valueModel, scalarValue] using extracted
    rw [same]
    exact read_operand_scalar_bound _ _ bits raw.val _ read.symm
  | Pointer pointer => simp [valueModel, scalarValue] at extracted

@[step]
theorem source_evaluate (registers : V.Registers) (ids : V.ThreadIds)
    (instruction : V.GuardedInstruction) :
    value_flow.evaluate registers ids instruction ⦃ result =>
      outcomeModel actionModel result =
        evaluate (bankModel registers) (idsModel ids) (guardedModel instruction) ⦄ := by
  have lit1 : ((1#8#uscalar : U8)).val = 1 := rfl
  have lit2 : ((2#8#uscalar : U8)).val = 2 := rfl
  have lit4 : ((4#8#uscalar : U8)).val = 4 := rfl
  have lit8 : ((8#8#uscalar : U8)).val = 8 := rfl
  have lit32 : ((32#8#uscalar : U8)).val = 32 := rfl
  have lit64 : ((64#8#uscalar : U8)).val = 64 := rfl
  rcases instruction with ⟨guard, instruction⟩
  unfold value_flow.evaluate
  step as ⟨enabled, enabledMeaning⟩
  cases enabled with
  | Err error =>
    simp only [outcomeModel] at enabledMeaning
    simp [outcomeModel, evaluate, guardedModel, ←enabledMeaning]
  | Ok enabled =>
    simp only [outcomeModel, id_eq] at enabledMeaning
    cases enabled with
    | false => simp [outcomeModel, actionModel, evaluate, guardedModel, ←enabledMeaning]
    | true =>
      simp only [Bool.not_true, Bool.false_eq_true, ite_false]
      simp only [evaluate, guardedModel, ←enabledMeaning, Option.bind_eq_bind, Option.bind_some, ite_true]
      cases instruction <;> simp only [instructionModel, evaluateInstruction]
      case MultiplyWide destination sourceLeft sourceRight =>
        step as ⟨outcome, leftRead⟩
        cases outcome with
        | Err error =>
          simp only [outcomeModel] at leftRead
          simp [outcomeModel, ←leftRead, Option.bind_eq_bind]
        | Ok leftValue =>
          simp only [outcomeModel] at leftRead
          simp only [←leftRead, Option.bind_eq_bind, Option.bind_some, Option.bind_none]
          step as ⟨outcome, leftScalar⟩
          cases outcome with
          | Err error =>
            simp only [outcomeModel] at leftScalar
            simp [outcomeModel, ←leftScalar, Option.bind_eq_bind]
          | Ok left =>
            simp only [outcomeModel] at leftScalar
            simp only [←leftScalar, Option.bind_eq_bind, Option.bind_some, Option.bind_none]
            step as ⟨outcome, rightRead⟩
            cases outcome with
            | Err error =>
              simp only [outcomeModel] at rightRead
              simp [outcomeModel, ←rightRead, Option.bind_eq_bind]
            | Ok rightValue =>
              simp only [outcomeModel] at rightRead
              simp only [←rightRead, Option.bind_eq_bind, Option.bind_some, Option.bind_none]
              step as ⟨outcome, rightScalar⟩
              cases outcome with
              | Err error =>
                simp only [outcomeModel] at rightScalar
                simp [outcomeModel, ←rightScalar, Option.bind_eq_bind]
              | Ok right =>
                simp only [outcomeModel] at rightScalar
                simp only [←rightScalar, Option.bind_eq_bind, Option.bind_some, Option.bind_none]
                have leftBound := source_scalar_operand_bound registers ids sourceLeft 32 leftValue left leftRead leftScalar
                have rightBound := source_scalar_operand_bound registers ids sourceRight 32 rightValue right rightRead rightScalar
                have productBound : left.val * right.val ≤ U64.max := by
                  have bound := Nat.mul_le_mul
                    (show left.val ≤ 4294967295 by omega) (show right.val ≤ 4294967295 by omega)
                  rw [U64.max_eq]
                  norm_num at bound
                  omega
                step as ⟨product, productMeaning⟩
                simp [outcomeModel, actionModel, valueModel, productMeaning]
      case ConvertUnsigned destination source fromBits toBits =>
        step as ⟨outcome, valueRead⟩
        cases outcome with
        | Err error =>
          simp only [outcomeModel] at valueRead
          simp [outcomeModel, ←valueRead, Option.bind_eq_bind]
        | Ok value =>
          simp only [outcomeModel] at valueRead
          simp only [←valueRead, Option.bind_eq_bind, Option.bind_some, Option.bind_none]
          cases value with
          | Pointer pointer =>
              dsimp only
              split <;> simp_all [outcomeModel, actionModel, valueModel]
          | Scalar value =>
            step as ⟨maskResult, maskMeaning⟩
            cases maskResult with
            | Err error =>
              have unsupported : supportedBits toBits.val = false := by
                cases supported : supportedBits toBits.val <;> simp_all [outcomeModel]
              simp [outcomeModel, valueModel, unsupported]
            | Ok mask =>
              have supported : supportedBits toBits.val = true := by
                cases supported : supportedBits toBits.val <;> simp_all [outcomeModel]
              have maskVal : mask.val = 2 ^ toBits.val - 1 := by
                simpa only [outcomeModel, supported, ite_true, Option.some.injEq] using maskMeaning
              simp [outcomeModel, actionModel, valueModel, supported, lift, bind_ok,
                UScalar.val_and, maskVal, Nat.and_two_pow_sub_one_eq_mod, truncate]
      case Load destination destinationBits bytes signed location =>
        split <;> simp only [bind_ok, Bool.true_eq, ite_true, ite_false]
        all_goals first
          | (split <;> simp only [bind_ok, Bool.true_eq, ite_true, ite_false])
          | skip
        all_goals simp_all [outcomeModel, supportedWidth, scalarBits, UScalar.eq_equiv,
          lit1, lit2, lit4, lit8, lit32, lit64]
        all_goals
          try simp only [lift, bind_ok]
          step as ⟨width, widthVal⟩
          all_goals try (solve | norm_num [lit1, lit2, lit4, lit8, U16.max_eq] | scalar_tac)
          try simp only [core.convert.num.FromU16U8.from_val_eq] at widthVal
          split <;> simp_all [outcomeModel]
        all_goals
          repeat' (
            step as ⟨result, meaning⟩
            cases result <;> simp only [outcomeModel] at meaning <;>
              simp only [←meaning, Option.bind_eq_bind, Option.bind_some, Option.bind_none])
        all_goals simp [outcomeModel, actionModel, lit1, lit2, lit4, lit8, lit32, lit64]
      case Store bytes location source =>
        split <;> simp only [bind_ok, Bool.true_eq, ite_true, ite_false]
        all_goals simp_all [outcomeModel, supportedWidth, scalarBits, UScalar.eq_equiv,
          lit1, lit2, lit4, lit8, lit32, lit64]
        all_goals try (solve | scalar_tac)
        all_goals
          step as ⟨outcome, addressMeaning⟩
          cases outcome with
          | Err error =>
            simp only [outcomeModel] at addressMeaning
            simp [outcomeModel, ←addressMeaning, Option.bind_eq_bind]
          | Ok pointer =>
            simp only [outcomeModel] at addressMeaning
            simp only [←addressMeaning, Option.bind_eq_bind, Option.bind_some, Option.bind_none]
            step as ⟨width, widthVal⟩
            all_goals try (solve | norm_num [lit1, lit2, lit4, lit8, U8.max_eq] | scalar_tac)
            try simp [show (8#u8).val = 8 from rfl, lit1, lit2, lit4, lit8] at widthVal
            step as ⟨outcome, valueRead⟩
            cases outcome with
            | Err error =>
              simp only [outcomeModel] at valueRead
              simp only [widthVal] at valueRead
              simp [outcomeModel, ←valueRead, Option.bind_eq_bind]
            | Ok value =>
              simp only [outcomeModel] at valueRead
              simp only [widthVal] at valueRead
              simp only [←valueRead, Option.bind_eq_bind, Option.bind_some, Option.bind_none]
              step as ⟨outcome, scalarRead⟩
              cases outcome with
              | Err error =>
                simp only [outcomeModel] at scalarRead
                simp [outcomeModel, ←scalarRead, Option.bind_eq_bind]
              | Ok scalar =>
                simp only [outcomeModel] at scalarRead
                simp only [←scalarRead, Option.bind_eq_bind, Option.bind_some, Option.bind_none]
                simp [outcomeModel, actionModel, lit1, lit2, lit4, lit8]
      all_goals
        repeat' (
          step as ⟨result, meaning⟩
          cases result <;> simp only [outcomeModel] at meaning <;>
            simp only [←meaning, Option.bind_eq_bind, Option.bind_some, Option.bind_none])
      all_goals simp [outcomeModel, actionModel, lift, bind_ok]

end ValueFlow.Source

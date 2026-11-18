/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceReads

open Aeneas Aeneas.Std Result
open PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

@[step]
theorem source_read_predicate (registers : V.Registers) (index : U16) :
    value_flow.read_predicate registers index ⦃ result =>
      outcomeModel id result = predicateModel registers index.val ⦄ := by
  have oneVal : (1#u64).val = 1 := rfl
  unfold value_flow.read_predicate
  step as ⟨writable, shape⟩
  cases writable with
  | false =>
    have outside : ¬(index.val / 64 < registers.raw.preds.val.length ∧
        index.val / 64 < registers.raw.pdef.val.length) := by simpa using shape.symm
    simp [outcomeModel, predicateModel, outside]
  | true =>
    have bounds : index.val / 64 < registers.raw.preds.val.length ∧
        index.val / 64 < registers.raw.pdef.val.length := by simpa using shape.symm
    simp only [ite_true]
    step as ⟨word, wordVal⟩
    simp only [lift, bind_ok]
    step as ⟨defined, definedEq⟩
    step as ⟨bit, bitVal⟩
    step as ⟨shifted, shiftVal, shiftBits⟩
    have bitBound : bit.val < 64 := by scalar_tac
    have flags : ((shifted &&& 1#u64).val) =
        (registers.raw.pdef.val[index.val / 64]?.getD 0#u64).val / 2 ^ (index.val % 64) % 2 := by
      simp only [UScalar.val_and, oneVal, Nat.and_one_is_mod]
      rw [shiftVal, Nat.shiftRight_eq_div_pow, bitVal, definedEq]
      simp only [core.convert.num.FromUsizeU16.from_val_eq, wordVal,
        List.getD_getElem?, bounds.2, ↓reduceDIte]
    have remainder : ((registers.raw.pdef.val[index.val / 64]?.getD 0#u64).val /
        2 ^ (index.val % 64)) % 2 < 2 := Nat.mod_lt _ (by decide)
    split
    · rename_i missing
      have absent : ¬((registers.raw.pdef.val[index.val / 64]?.getD 0#u64).val /
          2 ^ (index.val % 64)) % 2 = 1 := by
        have valueZero : (shifted &&& 1#u64).val = 0 := by scalar_tac
        omega
      simp only [WP.spec_ok, outcomeModel, predicateModel, bounds, and_self, ite_true, absent, ite_false]
    · rename_i present
      have available : ((registers.raw.pdef.val[index.val / 64]?.getD 0#u64).val /
          2 ^ (index.val % 64)) % 2 = 1 := by
        have valueNonzero : (shifted &&& 1#u64).val ≠ 0 := by scalar_tac
        omega
      step as ⟨value, valueEq⟩
      step as ⟨shiftedValue, shiftedVal, shiftedBits⟩
      simp only [outcomeModel, id_eq,
        predicateModel, bounds, ite_true, available]
      apply congrArg some
      apply Bool.eq_iff_iff.mpr
      simp only [bne_iff_ne, decide_eq_true_eq]
      have valueFlags : (shiftedValue &&& 1#u64).val =
          (registers.raw.preds.val[index.val / 64]?.getD 0#u64).val /
            2 ^ (index.val % 64) % 2 := by
        simp only [UScalar.val_and, oneVal, Nat.and_one_is_mod]
        rw [shiftedVal, Nat.shiftRight_eq_div_pow, bitVal, valueEq]
        simp only [core.convert.num.FromUsizeU16.from_val_eq, wordVal,
          List.getD_getElem?, bounds.1, ↓reduceDIte]
      have small := Nat.mod_lt
        ((registers.raw.preds.val[index.val / 64]?.getD 0#u64).val / 2 ^ (index.val % 64)) (by decide : 0 < 2)
      scalar_tac

@[step]
theorem source_read_guard (registers : V.Registers) (guard : V.Guard) :
    value_flow.read_guard registers guard ⦃ result =>
      outcomeModel id result = readGuard (bankModel registers) (guardModel guard) ⦄ := by
  rcases guard with ⟨predicate, negated⟩
  cases predicate with
  | none => simp [value_flow.read_guard, outcomeModel, readGuard, guardModel]
  | some index =>
    simp only [value_flow.read_guard]
    step as ⟨result, agreement⟩
    cases result with
    | Ok value =>
      simp only [outcomeModel, id_eq] at agreement
      simp [outcomeModel, readGuard, guardModel, bankModel, ←agreement]
    | Err error =>
      simp only [outcomeModel] at agreement
      simp [outcomeModel, readGuard, guardModel, bankModel, ←agreement]

end ValueFlow.Source

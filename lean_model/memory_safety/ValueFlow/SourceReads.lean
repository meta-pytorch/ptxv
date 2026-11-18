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
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

@[step]
theorem source_read_special (ids : V.ThreadIds) (special : V.SpecialRegister) :
    value_flow.read_special ids special ⦃ result =>
      outcomeModel UScalar.val result =
        readSpecial (idsModel ids) (specialModel special).1 (specialModel special).2 ⦄ := by
  have v3 : (3#usize).val = 3 := rfl
  cases special with
  | Lane =>
    simp only [value_flow.read_special, lift, bind_ok, WP.spec_ok, outcomeModel,
      specialModel, readSpecial, idsModel, core.convert.num.FromU64U32.from_val_eq]
  | Thread axis | Block axis | BlockSize axis | GridSize axis =>
    simp only [value_flow.read_special]
    split
    · rename_i valid
      simp only [lift, bind_ok]
      step as ⟨value, same⟩
      have bounded : axis.val < 3 := by scalar_tac
      have indexVal : (core.convert.num.FromUsizeU8.from axis).val = axis.val := by scalar_tac
      simp only [outcomeModel, specialModel, readSpecial, idsModel,
        bounded, ite_true, core.convert.num.FromU64U32.from_val_eq]
      simp only [indexVal] at same
      simp only [List.getD_getElem?, Array.length_eq, v3, bounded, ↓reduceDIte]
      exact congrArg some (congrArg UScalar.val same)
    · rename_i invalid
      have bounded : ¬axis.val < 3 := by scalar_tac
      simp [WP.spec_ok, outcomeModel, specialModel, readSpecial, bounded]

@[step]
theorem source_writable_predicate (registers : V.Registers) (index : U16) :
    value_flow.writable_predicate registers index ⦃ result =>
      result = (index.val / 64 < registers.raw.preds.val.length &&
        index.val / 64 < registers.raw.pdef.val.length) ⦄ := by
  unfold value_flow.writable_predicate
  step as ⟨word, wordVal⟩
  simp only [lift, bind_ok]
  split
  all_goals
    simp only [WP.spec_ok]
    apply Bool.eq_iff_iff.mpr
    simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.false_eq_true]
    scalar_tac

@[step]
theorem source_writable_virtual (registers : V.Registers) (index : U16) :
    value_flow.writable_virtual registers index ⦃ result =>
      result = (index.val < registers.raw.regs.val.length &&
        index.val < registers.pointers.val.length &&
        index.val / 64 < registers.raw.written.val.length) ⦄ := by
  unfold value_flow.writable_virtual
  simp only [lift, bind_ok]
  split
  · split
    · step as ⟨word, wordVal⟩
      apply Bool.eq_iff_iff.mpr
      simp only [Bool.and_eq_true, decide_eq_true_eq]
      scalar_tac
    · simp only [WP.spec_ok]
      apply Bool.eq_iff_iff.mpr
      simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.false_eq_true]
      scalar_tac
  · simp only [WP.spec_ok]
    apply Bool.eq_iff_iff.mpr
    simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.false_eq_true]
    scalar_tac

end ValueFlow.Source

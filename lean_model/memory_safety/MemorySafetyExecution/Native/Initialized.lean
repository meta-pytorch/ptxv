/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Core

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

private theorem source_initialized_loop (values : alloc.vec.Vec Bool)
    (offset count cursor : Usize) (defined : Bool)
    (bounded : cursor.val ≤ count.val)
    (fits : offset.val + count.val ≤ values.val.length)
    (initPrefix : defined = true → ∀ index, index < cursor.val →
      values.val[offset.val + index]? = some true) :
    ptxv_memory_write.execution_core.resolve_access_loop false values offset count cursor defined
      ⦃ output => output = true → ∀ index, index < count.val →
        values.val[offset.val + index]? = some true ⦄ := by
  unfold ptxv_memory_write.execution_core.resolve_access_loop
  apply loop.spec_decr_nat
    (fun current : Usize × Bool => count.val - current.1.val)
    (fun current : Usize × Bool => current.1.val ≤ count.val ∧
      (current.2 = true → ∀ index, index < current.1.val →
        values.val[offset.val + index]? = some true))
  · rintro ⟨cursor, defined⟩ ⟨bounded, initPrefix⟩
    change cursor.val ≤ count.val at bounded
    change (defined = true → ∀ index, index < cursor.val →
      values.val[offset.val + index]? = some true) at initPrefix
    unfold ptxv_memory_write.execution_core.resolve_access_loop.body
    by_cases inside : cursor.val < count.val
    · have test : cursor < count := by scalar_tac
      simp only [test, ite_true, Bool.false_eq_true, ite_false]
      cases defined with
      | false => simp
      | true =>
        simp only [ite_true]
        step as ⟨index, indexEq⟩
        have indexBound : index.val < values.val.length := by scalar_tac
        step as ⟨flag, flagEq⟩
        step as ⟨next, nextEq⟩
        refine ⟨by scalar_tac, ?_, by scalar_tac⟩
        intro trueFlag wanted wantedBound
        by_cases earlier : wanted < cursor.val
        · exact initPrefix rfl wanted earlier
        · have wantedEq : wanted = cursor.val := by scalar_tac
          rw [wantedEq, ← indexEq]
          simpa only [List.getElem?_eq_getElem indexBound] using
            congrArg some (flagEq.symm.trans trueFlag)
    · have test : ¬cursor < count := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      intro trueDefined index indexBound
      exact initPrefix trueDefined index (by omega)
  · exact ⟨bounded, initPrefix⟩

theorem source_initialized_range (values : alloc.vec.Vec Bool) (offset count : Usize)
    (fits : offset.val + count.val ≤ values.val.length)
    (accepted : ptxv_memory_write.execution_core.resolve_access_loop false values offset count
      0#usize true = .ok true) :
    ∀ index, index < count.val → values.val[offset.val + index]? = some true := by
  have spec := source_initialized_loop values offset count 0#usize true (by scalar_tac) fits
    (by simp)
  simpa only [accepted, WP.spec_ok, forall_true_left] using spec

end MemorySafetyExecution.Native

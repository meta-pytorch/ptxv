/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.PlacementEvents
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source
set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native

@[step]
theorem source_compatible_loop_total (layouts : Layouts) (index : Usize) (layout : value_flow.AllocationLayout)
    (flag : Bool) (cursor : Usize) (inside : index.val < layouts.val.length) (bound : cursor.val ≤ index.val) :
    execution_core.allocation_compatible_loop layouts index layout flag cursor ⦃ _ => True ⦄ := by
  unfold execution_core.allocation_compatible_loop
  apply loop.spec_decr_nat (fun current : Bool × Usize => index.val-current.2.val)
    (fun current : Bool × Usize => current.2.val ≤ index.val)
  · rintro ⟨valid,j⟩ before
    unfold execution_core.allocation_compatible_loop.body
    by_cases more : j.val < index.val
    · have test : j < index := by scalar_tac
      simp only [test,ite_true]
      cases valid with
      | false => simp
      | true =>
        simp only [ite_true]
        step as ⟨other,otherAt⟩
        step with UScalar.cast_inBounds_spec .U128 layout.base (by scalar_tac) as ⟨base,baseVal⟩
        step with UScalar.cast_inBounds_spec .U128 other.base (by scalar_tac) as ⟨otherBase,otherBaseVal⟩
        step with UScalar.cast_inBounds_spec .U128 other.bytes (by scalar_tac) as ⟨otherSize,otherSizeVal⟩
        step as ⟨otherEnd,otherEndVal⟩
        split
        · step with UScalar.cast_inBounds_spec .U128 other.base (by scalar_tac) as ⟨otherBaseAgain,otherBaseAgainVal⟩
          step with UScalar.cast_inBounds_spec .U128 layout.base (by scalar_tac) as ⟨baseAgain,baseAgainVal⟩
          step with UScalar.cast_inBounds_spec .U128 layout.bytes (by scalar_tac) as ⟨size,sizeVal⟩
          step as ⟨finish,finishVal⟩
          step as ⟨next,nextVal⟩
          exact ⟨by scalar_tac,by scalar_tac⟩
        · step as ⟨next,nextVal⟩
          exact ⟨by scalar_tac,by scalar_tac⟩
    · have test : ¬j < index := by scalar_tac
      simp [test]
  · exact bound

@[step]
theorem source_allocation_compatible_total (program : CoreProgram) (index : Usize)
    (inside : index.val < program.layouts.val.length) :
    execution_core.allocation_compatible program index ⦃ _ => True ⦄ := by
  unfold execution_core.allocation_compatible
  step as ⟨layout,layoutAt⟩
  exact source_compatible_loop_total program.layouts index layout true 0#usize inside (by scalar_tac)

end MemorySafetyExecution.Native

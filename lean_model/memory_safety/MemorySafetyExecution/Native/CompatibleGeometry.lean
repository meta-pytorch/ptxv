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
theorem source_compatible_loop_geometry (layouts : Layouts) (index : Usize) (layout : value_flow.AllocationLayout)
    (cursor : Usize) (inside : index.val < layouts.val.length) (bound : cursor.val ≤ index.val)
    (selected : layouts.val[index.val]?=some layout) (separated : SeparatedLayouts layouts) :
    execution_core.allocation_compatible_loop layouts index layout true cursor ⦃ result => result=true ⦄ := by
  unfold execution_core.allocation_compatible_loop
  apply loop.spec_decr_nat (fun current : Bool × Usize => index.val-current.2.val)
    (fun current : Bool × Usize => current.2.val ≤ index.val ∧ current.1=true)
  · rintro ⟨valid,j⟩ ⟨before,flagTrue⟩
    dsimp only at before flagTrue
    subst valid
    unfold execution_core.allocation_compatible_loop.body
    by_cases more : j.val < index.val
    · have test : j < index := by scalar_tac
      simp only [test,ite_true]
      step as ⟨other,otherAt⟩
      have otherInside : j.val < layouts.val.length := by omega
      have otherSelected : layouts.val[j.val]?=some other := by
        rw [List.getElem?_eq_getElem otherInside]
        simp only [otherAt]
      have disjoint:=separated index.val j.val layout other selected otherSelected (by omega)
      step with UScalar.cast_inBounds_spec .U128 layout.base (by scalar_tac) as ⟨base,baseVal⟩
      step with UScalar.cast_inBounds_spec .U128 other.base (by scalar_tac) as ⟨otherBase,otherBaseVal⟩
      step with UScalar.cast_inBounds_spec .U128 other.bytes (by scalar_tac) as ⟨otherSize,otherSizeVal⟩
      step as ⟨otherEnd,otherEndVal⟩
      split
      · rename_i overlapsStart
        step with UScalar.cast_inBounds_spec .U128 other.base (by scalar_tac) as ⟨otherBaseAgain,otherBaseAgainVal⟩
        step with UScalar.cast_inBounds_spec .U128 layout.base (by scalar_tac) as ⟨baseAgain,baseAgainVal⟩
        step with UScalar.cast_inBounds_spec .U128 layout.bytes (by scalar_tac) as ⟨size,sizeVal⟩
        step as ⟨finish,finishVal⟩
        have noOverlap : ¬otherBaseAgain < finish := by
          rcases disjoint with beforeOther|afterOther <;> scalar_tac
        simp only [noOverlap,decide_false]
        step as ⟨next,nextVal⟩
        exact ⟨by scalar_tac,by scalar_tac⟩
      · step as ⟨next,nextVal⟩
        exact ⟨by scalar_tac,by scalar_tac⟩
    · have test : ¬j < index := by scalar_tac
      simp [test]
  · exact ⟨bound,rfl⟩

@[step]
theorem source_allocation_compatible_geometry (program : CoreProgram) (index : Usize)
    (inside : index.val < program.layouts.val.length) (separated : SeparatedLayouts program.layouts) :
    execution_core.allocation_compatible program index ⦃ result => result=true ⦄ := by
  unfold execution_core.allocation_compatible
  step as ⟨layout,layoutAt⟩
  exact source_compatible_loop_geometry program.layouts index layout 0#usize inside (by scalar_tac) (by
    rw [List.getElem?_eq_getElem inside]
    simp only [layoutAt]) separated

end MemorySafetyExecution.Native

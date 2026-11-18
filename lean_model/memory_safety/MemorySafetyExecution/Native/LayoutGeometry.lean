/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.PlacementInputResolution
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source
set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

def StorageExtent (layout : value_flow.AllocationLayout) (bytes defined : Usize) : Prop :=
  layout.bytes.val ≠ 0 ∧ layout.bytes.val=bytes.val ∧ bytes=defined ∧ layout.alignment.val≠0

@[step]
theorem source_allocation_shape_extent (layout : value_flow.AllocationLayout) (bytes defined : Usize) :
    execution_core.allocation_shape layout bytes defined ⦃ result => result=true → StorageExtent layout bytes defined ⦄ := by
  unfold execution_core.allocation_shape
  split
  · rename_i nonzero
    step with UScalar.cast_inBounds_spec .U128 layout.bytes (by scalar_tac) as ⟨size,sizeVal⟩
    step with UScalar.cast_inBounds_spec .U128 bytes (by rcases Usize.bounds_eq with narrow|wide <;> scalar_tac) as ⟨length,lengthVal⟩
    split
    · rename_i sameSize
      split
      · rename_i sameDefined
        split
        · rename_i alignmentNonzero
          step as ⟨remainder,remainderVal⟩
          split
          · step with UScalar.cast_inBounds_spec .U128 layout.base (by scalar_tac) as ⟨base,baseVal⟩
            step with UScalar.cast_inBounds_spec .U128 layout.bytes (by scalar_tac) as ⟨sizeAgain,sizeAgainVal⟩
            step as ⟨finish,finishVal⟩
            step with UScalar.cast_inBounds_spec .U128 core.num.U64.MAX (by simp [core.num.U64.MAX,U64.rMax];scalar_tac) as ⟨maximum,maximumVal⟩
            change maximum.val=18446744073709551615 at maximumVal
            step with UScalar.add_spec (x:=maximum) (y:=1#u128) (by scalar_tac) as ⟨limit,limitVal⟩
            intro accepted
            unfold StorageExtent
            refine ⟨?_,?_,sameDefined,?_⟩ <;> scalar_tac
          · simp only [WP.spec_ok,Bool.false_eq_true,false_implies]
        · simp
      · simp
    · simp
  · simp

end MemorySafetyExecution.Native

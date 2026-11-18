/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.LayoutGeometry

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native

@[step]
theorem source_allocation_shape_geometry (layout : value_flow.AllocationLayout) (bytes defined : Usize)
    (extent : StorageExtent layout bytes defined)
    (legal : ValueFlow.LegalPlacement {bytes:=layout.bytes.val,alignment:=layout.alignment.val} layout.base.val) :
    execution_core.allocation_shape layout bytes defined ⦃ result => result=true ⦄ := by
  obtain ⟨nonzero,sizeEq,definedEq,alignment⟩ := extent
  have bytesNonzero : (layout.bytes != 0#u64)=true := by scalar_tac
  have alignmentNonzero : (layout.alignment != 0#u64)=true := by scalar_tac
  unfold execution_core.allocation_shape
  simp only [bytesNonzero,ite_true]
  step with UScalar.cast_inBounds_spec .U128 layout.bytes (by scalar_tac) as ⟨size,sizeVal⟩
  step with UScalar.cast_inBounds_spec .U128 bytes (by rcases Usize.bounds_eq with narrow|wide <;> scalar_tac) as ⟨length,lengthVal⟩
  have castEq : size=length := by scalar_tac
  simp only [castEq,ite_true,definedEq,alignmentNonzero]
  step as ⟨remainder,remainderVal⟩
  have remainderZero : remainder=0#u64 := by
    have divides:=Nat.mod_eq_zero_of_dvd legal.1
    scalar_tac
  simp only [remainderZero,ite_true]
  step with UScalar.cast_inBounds_spec .U128 layout.base (by scalar_tac) as ⟨base,baseVal⟩
  step with UScalar.cast_inBounds_spec .U128 layout.bytes (by scalar_tac) as ⟨sizeAgain,sizeAgainVal⟩
  step as ⟨finish,finishVal⟩
  step with UScalar.cast_inBounds_spec .U128 core.num.U64.MAX (by simp [core.num.U64.MAX,U64.rMax];scalar_tac) as ⟨maximum,maximumVal⟩
  change maximum.val=18446744073709551615 at maximumVal
  step with UScalar.add_spec (x:=maximum) (y:=1#u128) (by scalar_tac) as ⟨limit,limitVal⟩
  have physical:=legal.2
  scalar_tac

end MemorySafetyExecution.Native

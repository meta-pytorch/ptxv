/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.PlacementEvents

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

theorem source_resolved_request_placement (leftProgram rightProgram : CoreProgram) (left right : CoreState)
    (pointer : Pointer) (width : U8) (write : Bool) (leftAccess rightAccess : ResolvedAccess)
    (objects : ∀ leftLayout rightLayout,
      leftProgram.layouts.val[pointer.allocation.val]?=some leftLayout →
      rightProgram.layouts.val[pointer.allocation.val]?=some rightLayout →
      leftLayout.bytes=rightLayout.bytes ∧ leftLayout.alignment=rightLayout.alignment)
    (leftResolved : execution_core.resolve_access leftProgram left pointer width write=.ok (.Ok leftAccess))
    (rightResolved : execution_core.resolve_access rightProgram right pointer width write=.ok (.Ok rightAccess)) :
    Nonempty (RequestPlacement leftProgram.layouts rightProgram.layouts leftAccess.request rightAccess.request) := by
  obtain ⟨leftDescriptor,leftRange⟩ := source_resolve_range leftProgram left pointer width write leftAccess leftResolved
  obtain ⟨rightDescriptor,rightRange⟩ := source_resolve_range rightProgram right pointer width write rightAccess rightResolved
  obtain ⟨leftLayout,leftMemory,leftAt,leftMemoryAt,widths,leftOffset,leftAddress,leftLive,leftOrigin,
    leftAlignment,leftAlignWidth,leftOffsetWidth,leftBaseWidth,leftPhysical,leftStorage,leftShape,leftInitialized⟩ := leftRange
  obtain ⟨rightLayout,rightMemory,rightAt,rightMemoryAt,_,rightOffset,rightAddress,rightLive,rightOrigin,
    rightAlignment,rightAlignWidth,rightOffsetWidth,rightBaseWidth,rightPhysical,rightStorage,rightShape,rightInitialized⟩ := rightRange
  obtain ⟨sameBytes,sameAlignment⟩ := objects leftLayout rightLayout leftAt rightAt
  have sameWidth : leftAccess.request.width=rightAccess.request.width :=
    UScalar.eq_of_val_eq (leftDescriptor.2.2.1.trans rightDescriptor.2.2.1.symm)
  have sameRequestAlignment : leftAccess.request.alignment=rightAccess.request.alignment :=
    UScalar.eq_of_val_eq (leftDescriptor.2.2.2.1.trans rightDescriptor.2.2.2.1.symm)
  have sameKind : leftAccess.request.kind=rightAccess.request.kind := leftDescriptor.2.2.2.2.trans rightDescriptor.2.2.2.2.symm
  refine ⟨⟨pointer.allocation.val,pointer.offset.val,leftLayout,rightLayout,leftAt,rightAt,sameBytes,sameAlignment,
    leftAddress,rightAddress,sameWidth,sameRequestAlignment,sameKind,?_,?_,?_,?_,?_⟩⟩
  · have requested:=leftDescriptor.2.2.1
    rcases widths with one|two|four|eight <;> omega
  · rw [leftDescriptor.2.2.1]
    exact leftOrigin
  · rw [leftDescriptor.2.2.2.1]
    rcases widths with one|two|four|eight <;> omega
  · rw [leftDescriptor.2.2.2.1]
    exact Nat.dvd_of_mod_eq_zero leftAlignWidth
  · rw [leftDescriptor.2.2.2.1]
    exact Nat.dvd_of_mod_eq_zero leftOffsetWidth

end MemorySafetyExecution.Native

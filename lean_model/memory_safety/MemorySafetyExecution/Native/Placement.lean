/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Resolution
import ValueFlow.Provenance

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

theorem source_resolved_all_placements
    (program : CoreProgram) (state : CoreState) (pointer : Pointer)
    (width : U8) (write : Bool) (access : ResolvedAccess)
    (accepted : ptxv_memory_write.execution_core.resolve_access program state pointer width write =
      .ok (.Ok access)) :
    ∃ layout, program.layouts.val[pointer.allocation.val]? = some layout ∧
      ValueFlow.ObjectAccess { bytes := layout.bytes.val, alignment := layout.alignment.val }
        pointer.offset.val width.val ∧
      ∀ base, ValueFlow.LegalPlacement
        { bytes := layout.bytes.val, alignment := layout.alignment.val } base →
        ValueFlow.PhysicalAccess { bytes := layout.bytes.val, alignment := layout.alignment.val }
          base pointer.offset.val width.val := by
  obtain ⟨_, range⟩ := source_resolve_range program state pointer width write access accepted
  obtain ⟨layout, memory, layoutSelected, memorySelected, widthKnown, offsetVal, addressVal,
    live, originBound, alignment, aligned, offsetAligned, baseAligned, physicalBound,
    storageBound, definedShape, initialized⟩ := range
  have objectAccess : ValueFlow.ObjectAccess
      { bytes := layout.bytes.val, alignment := layout.alignment.val }
      pointer.offset.val width.val :=
    ⟨widthKnown, alignment, Nat.dvd_of_mod_eq_zero aligned,
      Nat.dvd_of_mod_eq_zero offsetAligned, originBound⟩
  exact ⟨layout, layoutSelected, objectAccess,
    ValueFlow.object_access_all_placements _ _ _ objectAccess⟩

end MemorySafetyExecution.Native

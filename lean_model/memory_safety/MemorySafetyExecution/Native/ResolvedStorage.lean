/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Resolution

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

theorem source_resolved_storage (program : CoreProgram) (state : CoreState) (pointer : Pointer)
    (bytes : U8) (write : Bool) (access : ResolvedAccess)
    (resolved : execution_core.resolve_access program state pointer bytes write=.ok (.Ok access)) :
    ∃ memory, state.allocations.val[access.allocation.val]?=some memory ∧
      0<access.width.val ∧ access.width.val≤8 ∧
      access.offset.val+access.width.val≤memory.bytes.val.length ∧
      memory.defined.val.length=memory.bytes.val.length := by
  obtain ⟨descriptor,range⟩ := source_resolve_range program state pointer bytes write access resolved
  obtain ⟨layout,memory,layoutAt,memoryAt,widths,offset,address,live,origin,alignment,
    alignmentWidth,offsetWidth,baseWidth,addressBound,storage,shape,initialized⟩ := range
  refine ⟨memory,?_,?_,?_,?_,shape⟩
  · simpa only [descriptor.1] using memoryAt
  · have width := descriptor.2.1
    rcases widths with one|two|four|eight <;> omega
  · have width := descriptor.2.1
    rcases widths with one|two|four|eight <;> omega
  · rw [offset,descriptor.2.1]
    exact storage

end MemorySafetyExecution.Native

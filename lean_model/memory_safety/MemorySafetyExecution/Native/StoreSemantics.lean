/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.StoreValues
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

theorem source_store_semantics (state : CoreState) (access : execution_core.ResolvedAccess) (value : U64)
    (memory : AllocationMemory) (selected : state.allocations.val[access.allocation.val]?=some memory)
    (small : access.width.val ≤ 8)
    (fitsBytes : access.offset.val + access.width.val ≤ memory.bytes.val.length)
    (fitsDefined : access.offset.val + access.width.val ≤ memory.defined.val.length) :
    ∃ written after, execution_core.store_value state access value=.ok written ∧
      written={state with allocations:=state.allocations.set access.allocation after} ∧
      ByteFrame memory after access.offset.val access.width.val ∧
      StoredPayload after access.offset.val access.width.val value := by
  obtain ⟨inside,memoryEq⟩ := List.getElem?_eq_some_iff.mp selected
  have bytesBound : access.offset.val+access.width.val≤state.allocations.val[access.allocation.val].bytes.val.length := by
    simpa only [memoryEq] using fitsBytes
  have definedBound : access.offset.val+access.width.val≤state.allocations.val[access.allocation.val].defined.val.length := by
    simpa only [memoryEq] using fitsDefined
  obtain ⟨written,computed,after,afterEq,frame⟩ := MemoryWrite.spec_returns _ _
    (source_store_value_frame state access value inside small bytesBound definedBound)
  have payloadSpec := source_store_value_payload state access value inside small bytesBound definedBound
  rw [computed] at payloadSpec
  have payloadMeaning : ∃ after, written={state with allocations:=state.allocations.set access.allocation after} ∧
      StoredPayload after access.offset.val access.width.val value := by
    simpa only [WP.spec_ok] using payloadSpec
  obtain ⟨payloadMemory,payloadEq,payload⟩ := payloadMeaning
  have same : after=payloadMemory := by
    have fields := congrArg (fun s:CoreState => s.allocations.val[access.allocation.val]?) (afterEq.symm.trans payloadEq)
    simpa only [alloc.vec.Vec.set_val_eq,List.getElem?_set_self inside,Option.some.injEq] using fields
  refine ⟨written,after,computed,afterEq,?_,?_⟩
  · simpa only [memoryEq] using frame
  · simpa only [same] using payload

end MemorySafetyExecution.Native

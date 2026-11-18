/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Resolution
import MemorySafetyExecution.Native.ByteEffects
import MemorySafetyExecution.Native.CommitEffects

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

/-- A concrete native memory effect covered by its exact emitted request. -/
inductive NativeMemoryEffect (program : CoreProgram) (before : CoreState) :
    ptxv_memory_write.memory_safety.Event → CoreState → Prop where
  | control (after : CoreState) (frame : after.allocations = before.allocations) :
      NativeMemoryEffect program before .Control after
  | load (actor : Usize) (after : CoreState) (pointer : Pointer)
      (width : U8) (access : ResolvedAccess)
      (descriptor : ExactDescriptor pointer width false access)
      (range : OriginRange program before pointer width access)
      (frame : after.allocations = before.allocations) :
      NativeMemoryEffect program before (.Access actor access.request) after
  | store (actor : Usize) (after : CoreState) (pointer : Pointer)
      (width : U8) (access : ResolvedAccess) (oldMemory newMemory : AllocationMemory)
      (descriptor : ExactDescriptor pointer width true access)
      (range : OriginRange program before pointer width access)
      (selected : before.allocations.val[access.allocation.val]? = some oldMemory)
      (frame : after.allocations = before.allocations.set access.allocation newMemory)
      (bytes : ByteFrame oldMemory newMemory access.offset.val access.width.val) :
      NativeMemoryEffect program before (.Access actor access.request) after

theorem commit_effect_covered (program : CoreProgram) (before after : CoreState)
    (event : ptxv_memory_write.memory_safety.Event)
    (effect : CommitEffect program before event after) :
    NativeMemoryEffect program before event after := by
  cases effect with
  | control after frame => exact .control after frame
  | load actor after pointer width access resolved frame =>
    obtain ⟨descriptor, range⟩ := source_resolve_range program before pointer width false access resolved
    exact .load actor after pointer width access descriptor range frame
  | store actor after pointer width access value written resolved stored frame =>
    obtain ⟨descriptor, range⟩ := source_resolve_range program before pointer width true access resolved
    obtain ⟨layout, memory, layoutSelected, memorySelected, widthKnown, offsetVal, addressVal,
      live, originBound, alignment, aligned, offsetAligned, baseAligned, physicalBound,
      storageBound, definedShape, initialized⟩ := range
    have selected : before.allocations.val[access.allocation.val]? = some memory := by
      simpa only [descriptor.1] using memorySelected
    obtain ⟨inside, memoryValue⟩ := List.getElem?_eq_some_iff.mp selected
    have small : access.width.val ≤ 8 := by
      have := descriptor.2.1
      omega
    have fitsBytes : access.offset.val + access.width.val ≤
        before.allocations.val[access.allocation.val].bytes.val.length := by
      rw [memoryValue, offsetVal, descriptor.2.1]
      exact storageBound
    have fitsDefined : access.offset.val + access.width.val ≤
        before.allocations.val[access.allocation.val].defined.val.length := by
      rw [memoryValue, definedShape]
      simpa only [memoryValue] using fitsBytes
    have spec := source_store_value_frame before access value inside small fitsBytes fitsDefined
    have result : ∃ newMemory,
        written = { before with allocations := before.allocations.set access.allocation newMemory } ∧
        ByteFrame before.allocations.val[access.allocation.val] newMemory
          access.offset.val access.width.val := by
      simpa only [stored, WP.spec_ok] using spec
    obtain ⟨newMemory, writtenEq, byteFrame⟩ := result
    apply NativeMemoryEffect.store actor after pointer width access memory newMemory descriptor
      (by exact ⟨layout, memory, layoutSelected, memorySelected, widthKnown, offsetVal, addressVal,
        live, originBound, alignment, aligned, offsetAligned, baseAligned, physicalBound,
        storageBound, definedShape, initialized⟩) selected
    · simpa only [writtenEq] using frame
    · simpa only [memoryValue] using byteFrame

theorem source_commit_memory_effect (program : CoreProgram) (before after : CoreState)
    (commit : ptxv_memory_write.execution_core.Commit) (event : ptxv_memory_write.memory_safety.Event)
    (accepted : ptxv_memory_write.execution_core.commit_action program before commit =
      .ok (.Ok event, after)) : NativeMemoryEffect program before event after :=
  commit_effect_covered program before after event
    (source_commit_effect program before after commit event accepted)

end MemorySafetyExecution.Native

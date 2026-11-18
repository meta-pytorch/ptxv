/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Core
import ValueFlow.SourceEquality
import MemorySafety.SourceCertificate

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

abbrev ThreadState := ptxv_memory_write.execution_core.ThreadState
abbrev AllocationBytes := ptxv_memory_write.execution_core.AllocationBytes

deriving instance DecidableEq for ptxv_memory_write.execution_core.ThreadState,
  ptxv_memory_write.execution_core.AllocationBytes, ptxv_memory_write.execution_core.State

theorem source_vector_equality {α : Type} [DecidableEq α]
    (implementation : core.cmp.PartialEq α α)
    (element : ∀ left right, implementation.eq left right = .ok (decide (left = right)))
    (left right : alloc.vec.Vec α) :
    alloc.vec.partial_eq.PartialEqVec.eq implementation left right = .ok (decide (left = right)) := by
  have specification := MemorySafety.Source.vector_model_equality implementation id element left right
  simpa only [List.map_id, alloc.vec.Vec.eq_iff] using specification

theorem source_thread_equality (left right : ThreadState) :
    ptxv_memory_write.execution_core.ThreadState.Insts.CoreCmpPartialEqThreadState.eq left right =
      .ok (decide (left = right)) := by
  rcases left with ⟨pc, running, registers, ids⟩
  rcases right with ⟨otherPc, otherRunning, otherRegisters, otherIds⟩
  by_cases p : pc = otherPc <;> by_cases r : running = otherRunning <;>
    by_cases b : registers = otherRegisters <;> by_cases i : ids = otherIds <;>
    simp_all [ptxv_memory_write.execution_core.ThreadState.Insts.CoreCmpPartialEqThreadState.eq,
      ValueFlow.Source.source_registers_trait_equality, ValueFlow.Source.source_ids_equality]

theorem source_allocation_equality (left right : AllocationBytes) :
    ptxv_memory_write.execution_core.AllocationBytes.Insts.CoreCmpPartialEqAllocationBytes.eq left right =
      .ok (decide (left = right)) := by
  have bytes := source_vector_equality core.cmp.PartialEqU8 (by intros; rfl) left.bytes right.bytes
  have defined := source_vector_equality core.cmp.PartialEqBool (by intros; rfl) left.defined right.defined
  rcases left with ⟨b, d, live⟩
  rcases right with ⟨otherB, otherD, otherLive⟩
  by_cases bEq : b = otherB <;> by_cases dEq : d = otherD <;> by_cases lEq : live = otherLive <;>
    simp_all [ptxv_memory_write.execution_core.AllocationBytes.Insts.CoreCmpPartialEqAllocationBytes.eq]

theorem source_state_equal (left right : CoreState) :
    ptxv_memory_write.execution_core.state_equal left right = .ok (decide (left = right)) := by
  have threads := source_vector_equality
    ptxv_memory_write.execution_core.ThreadState.Insts.CoreCmpPartialEqThreadState
    source_thread_equality left.threads right.threads
  have allocations := source_vector_equality
    ptxv_memory_write.execution_core.AllocationBytes.Insts.CoreCmpPartialEqAllocationBytes
    source_allocation_equality left.allocations right.allocations
  rcases left with ⟨t, a⟩
  rcases right with ⟨otherT, otherA⟩
  by_cases tEq : t = otherT <;> by_cases aEq : a = otherA <;>
    simp_all [ptxv_memory_write.execution_core.state_equal,
      ptxv_memory_write.execution_core.State.Insts.CoreCmpPartialEqState.eq]

end MemorySafetyExecution.Native

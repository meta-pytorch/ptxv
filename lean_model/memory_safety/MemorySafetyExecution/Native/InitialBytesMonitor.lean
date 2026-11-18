/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InitialBytesConstructor
import MemorySafetyExecution.Native.InputLayout

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

private theorem monitor_allocation_bytes (layout : value_flow.AllocationLayout)
    (left right : execution_core.AllocationBytes) (definitions : left.defined=right.defined) :
    checked_execution.monitor_allocation layout left=checked_execution.monitor_allocation layout right := by
  simp only [checked_execution.monitor_allocation,definitions]

private theorem monitor_allocations_bytes (program : CoreProgram) (right : CoreState)
    (variant : InitialBytesVariant program.initial right) :
    checked_execution.monitor_allocations (withInitial program right)=checked_execution.monitor_allocations program := by
  unfold checked_execution.monitor_allocations checked_execution.monitor_allocations_loop
  congr 1
  funext current
  rcases current with ⟨allocations,index⟩
  unfold checked_execution.monitor_allocations_loop.body
  dsimp only [withInitial]
  by_cases inside : index.val<program.initial.allocations.val.length
  · have rightInside : index.val<right.allocations.val.length := by have:=variant.allocationsLength;omega
    have leftAt:=List.getElem?_eq_getElem inside
    have rightAt:=List.getElem?_eq_getElem rightInside
    have definitions:=(variant.allocations index.val program.initial.allocations.val[index.val]
      right.allocations.val[index.val] leftAt rightAt).2.1
    have sameMonitor : ∀ layout,checked_execution.monitor_allocation layout right.allocations.val[index.val]=
        checked_execution.monitor_allocation layout program.initial.allocations.val[index.val] := by
      intro layout
      exact (monitor_allocation_bytes layout _ _ definitions).symm
    simp only [alloc.vec.Vec.index_slice_index,alloc.vec.Vec.index_usize,alloc.vec.Vec.getElem?_Nat_eq,
      leftAt,rightAt,bind_ok,sameMonitor]
  · have rightOutside : right.allocations.val.length ≤ index.val := by have:=variant.allocationsLength;omega
    have noLeft : program.initial.allocations.val[index.val]?=none:=List.getElem?_eq_none (by omega)
    have noRight : right.allocations.val[index.val]?=none:=List.getElem?_eq_none rightOutside
    simp only [alloc.vec.Vec.index_slice_index,alloc.vec.Vec.index_usize,alloc.vec.Vec.getElem?_Nat_eq,noLeft,noRight]

theorem source_monitor_program_bytes (program : CoreProgram) (right : CoreState)
    (shadow : value_flow_dependencies.State) (variant : InitialBytesVariant program.initial right)
    (related : StateInputsRelated shadow program.initial right) :
    checked_execution.monitor_program (withInitial program right)=checked_execution.monitor_program program := by
  have layoutEq:=source_layout_valid_same_shape program shadow program.initial right related
  have threadCount : alloc.vec.Vec.len right.threads=alloc.vec.Vec.len program.initial.threads := by rw [variant.threads]
  unfold checked_execution.monitor_program
  rw [source_layout_initial_irrelevant,monitor_allocations_bytes program right variant]
  simp only [withInitial]
  rw [←layoutEq,threadCount]

end MemorySafetyExecution.Native

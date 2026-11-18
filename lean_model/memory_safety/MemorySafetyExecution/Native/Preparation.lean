/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.MemoryEffects

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

theorem source_execute_prepared (program : CoreProgram) (before after : CoreState)
    (actor : Usize) (event : ptxv_memory_write.memory_safety.Event)
    (accepted : ptxv_memory_write.execution_core.execute program before actor =
      .ok (.Ok event, after)) :
    ∃ commit, ptxv_memory_write.execution_core.prepare program before actor = .ok (.Ok commit) ∧
      ptxv_memory_write.execution_core.commit_action program before commit = .ok (.Ok event, after) := by
  unfold ptxv_memory_write.execution_core.execute at accepted
  obtain ⟨result, prepared, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  cases result with
  | Err fault => simp at accepted
  | Ok commit => exact ⟨commit, prepared, accepted⟩

theorem source_prepare_fault_frame (program : CoreProgram) (before : CoreState)
    (actor : Usize) (fault : Fault)
    (prepared : ptxv_memory_write.execution_core.prepare program before actor = .ok (.Err fault)) :
    ptxv_memory_write.execution_core.execute program before actor = .ok (.Err fault, before) := by
  simp only [ptxv_memory_write.execution_core.execute, prepared, bind_ok]

theorem source_execute_memory_effect (program : CoreProgram) (before after : CoreState)
    (actor : Usize) (event : ptxv_memory_write.memory_safety.Event)
    (accepted : ptxv_memory_write.execution_core.execute program before actor =
      .ok (.Ok event, after)) : NativeMemoryEffect program before event after := by
  obtain ⟨commit, _, committed⟩ := source_execute_prepared program before after actor event accepted
  exact source_commit_memory_effect program before after commit event committed

theorem core_step_memory_effect (program : CoreProgram) (before after : CoreState)
    (event : ptxv_memory_write.memory_safety.Event)
    (step : CoreStep program before event after) : NativeMemoryEffect program before event after := by
  obtain ⟨actor, _, executed⟩ := step
  exact source_execute_memory_effect program before after actor event executed

end MemorySafetyExecution.Native

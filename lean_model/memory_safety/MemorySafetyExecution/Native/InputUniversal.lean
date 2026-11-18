/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InitialBytesMonitor
import MemorySafetyExecution.Native.Soundness

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source MemorySafetyExecution.Source

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

theorem source_input_verify_all_initial_bytes (program : CoreProgram)
    (certificate : Slice input_independent_execution.State)
    (accepted : input_independent_execution.verify program certificate=.ok true)
    (storage : InitialStorageShape program.initial) (canonical : InitialRegisterCanonical program.initial) :
    ∀ alternate,InitialBytesVariant program.initial alternate → MachineMemorySafe (withInitial program alternate) := by
  obtain ⟨⟨captured,initialized,_⟩,_⟩:=source_input_verify_exact_closure program certificate accepted
  obtain ⟨concreteInitialized,shadowInitialized⟩:=source_input_initial_parts program captured initialized
  obtain ⟨_,coreInitialized,_,_⟩:=source_initial_parts program captured.concrete concreteInitialized
  have exactInitial:=(source_initial_state_exact program captured.concrete.execution coreInitialized).1
  have capturedCore : execution_core.initial_state program=.ok (.Ok program.initial) := by
    simpa only [exactInitial] using coreInitialized
  intro alternate variant
  have related : StateInputsRelated captured.dependencies program.initial alternate :=
    source_initial_inputs_related program.initial alternate captured.dependencies
      (by simpa only [exactInitial] using shadowInitialized) variant storage canonical
  have capturedRelated : StateInputsRelated captured.dependencies captured.concrete.execution alternate := by
    simpa only [exactInitial] using related
  obtain ⟨frame,projected,layout,traces,noFaults⟩:=source_input_verify_safe_from program certificate accepted
    captured alternate initialized capturedRelated
  have alternateLayout : execution_core.layout_valid program alternate=.ok true := by
    rw [←source_layout_valid_same_shape program captured.dependencies program.initial alternate related]
    exact (source_initial_state_exact program program.initial capturedCore).2
  have constructed:=source_initial_state_bytes program alternate variant capturedCore alternateLayout
  apply core_memory_safe_native_memory_safe
  refine ⟨alternate,frame,constructed,?_,layout,?_,?_⟩
  · rw [source_monitor_program_bytes program alternate captured.dependencies variant related]
    exact projected
  · intro events after trace
    exact traces events after (source_trace_initial_irrelevant program alternate alternate after events trace)
  · intro after reachable
    obtain ⟨steps,outputs⟩:=noFaults after
      (source_reachable_initial_irrelevant program alternate alternate after reachable)
    refine ⟨?_,?_⟩
    · intro actor running
      obtain ⟨event,next,executed⟩:=steps actor running
      exact ⟨event,next,by rw [source_execute_initial_irrelevant];exact executed⟩
    · simpa only [withInitial] using outputs

end MemorySafetyExecution.Native

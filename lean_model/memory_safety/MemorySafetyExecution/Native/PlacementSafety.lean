/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.PlacementTrace
import MemorySafetyExecution.Native.PlacementHistory
import MemorySafetyExecution.Native.FullSafety
import MemorySafetyExecution.Native.InputSafety

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source MemorySafetyExecution.Source

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

theorem source_input_verify_full_safety_from (program alternateProgram : CoreProgram)
    (certificate : Slice input_independent_execution.State)
    (accepted : input_independent_execution.verify program certificate=.ok true)
    (sameCode : program.code=alternateProgram.code) (sameOutputs : program.outputs=alternateProgram.outputs)
    (placements : ProgramPlacements program alternateProgram)
    (leftSeparated : SeparatedLayouts program.layouts) (rightSeparated : SeparatedLayouts alternateProgram.layouts)
    (capturedInitial : input_independent_execution.State) (alternateInitial : CoreState)
    (initialized : input_independent_execution.initial program=.ok (some capturedInitial))
    (related : StateInputsRelated capturedInitial.dependencies capturedInitial.concrete.execution alternateInitial) :
    FullCoreSafetyFrom alternateProgram alternateInitial := by
  obtain ⟨concrete,verified⟩:=source_input_verify_concrete program certificate accepted
  obtain ⟨initial,frame,coreInitialized,_,_,capturedTraces⟩:=source_verify_all_traces program concrete verified
  obtain ⟨checkedInitialized,_⟩:=source_input_initial_parts program capturedInitial initialized
  obtain ⟨_,capturedCore,_,_⟩:=source_initial_parts program capturedInitial.concrete checkedInitialized
  have same : initial=capturedInitial.concrete.execution := by
    simpa only [coreInitialized,ok.injEq,core.result.Result.Ok.injEq] using capturedCore
  subst initial
  constructor
  · intro after reachable
    obtain ⟨events,trace⟩:=core_reachable_trace alternateProgram alternateInitial after reachable
    obtain ⟨captured,_,capturedReachable,afterRelated,_,_⟩:=source_input_verify_trace_placements program alternateProgram
      certificate accepted sameCode placements capturedInitial alternateInitial initialized related trace
    have closed:=(source_input_verify_reachable program certificate accepted capturedReachable).2
    constructor
    · intro actor running
      have capturedRunning : execution_core.running captured.concrete.execution actor=.ok true := by
        rw [input_running_equal captured.dependencies captured.concrete.execution after afterRelated actor]
        exact running
      obtain ⟨next,stepped,_⟩:=closed.2 actor capturedRunning
      obtain ⟨advanced,concreteStep⟩:=source_input_step_parts program captured next actor stepped
      obtain ⟨capturedEvent,_,executed,_,_⟩:=source_step_exact program captured.concrete next.concrete actor concreteStep
      obtain ⟨event,nextAfter,alternateExecuted,_,_⟩:=source_advance_native_placements program alternateProgram
        captured.concrete.execution after next.concrete.execution captured.dependencies next.dependencies actor capturedEvent
        sameCode placements afterRelated advanced executed
      exact ⟨event,nextAfter,alternateExecuted,source_execute_concrete_safety alternateProgram after nextAfter actor event alternateExecuted,
        source_execute_instruction_access_complete alternateProgram after nextAfter actor event alternateExecuted⟩
    · intro terminal
      have capturedTerminal : execution_core.terminal captured.concrete.execution=.ok true := by
        rw [input_terminal_equal captured.dependencies captured.concrete.execution after afterRelated]
        exact terminal
      obtain ⟨bytes,observed⟩:=(source_good program captured.concrete closed.1).2 capturedTerminal
      obtain ⟨other,otherObserved,_⟩:=source_outputs_inputs captured.dependencies captured.concrete.execution after
        program.outputs.deref bytes afterRelated observed
      exact ⟨other,by simpa only [←sameOutputs] using otherObserved⟩
  · intro events after trace
    obtain ⟨captured,capturedEvents,_,_,capturedTrace,eventRelations⟩:=source_input_verify_trace_placements program alternateProgram
      certificate accepted sameCode placements capturedInitial alternateInitial initialized related trace
    obtain ⟨checked,_,_,safe⟩:=capturedTraces capturedEvents captured.concrete.execution capturedTrace
    exact placement_trace_no_conflicts program.layouts alternateProgram.layouts leftSeparated rightSeparated eventRelations
      (MemorySafetyExecution.safe_trace_records _ _ (stateModel checked.monitor) safe).2

end MemorySafetyExecution.Native

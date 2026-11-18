/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputTrace
import MemorySafetyExecution.Native.InputInitialState
import MemorySafetyExecution.Native.InputTerminal
import MemorySafetyExecution.Native.InputOutputs
import MemorySafetyExecution.SourceReport

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source MemorySafetyExecution.Source

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

/-- Safety of every execution from an explicitly supplied initial state. -/
def CoreSafeFrom (program : CoreProgram) (initial : CoreState) : Prop :=
  ∃ frame,
    checked_execution.monitor_program program = .ok (some frame) ∧
    MemorySafety.LayoutSafe (programModel frame) ∧
    (∀ events state, CoreTrace program initial events state →
      ∃ memory, SafeTrace (programModel frame) (events.map eventModel) memory) ∧
    (∀ state, CoreReachable program initial state →
      (∀ actor, execution_core.running state actor = .ok true →
        ∃ event after, execution_core.execute program state actor = .ok (.Ok event, after)) ∧
      (execution_core.terminal state = .ok true →
        ∃ bytes, execution_core.outputs state program.outputs.deref = .ok (.Ok bytes)))

theorem source_input_verify_safe_from (program : CoreProgram)
    (certificate : Slice input_independent_execution.State)
    (accepted : input_independent_execution.verify program certificate = .ok true)
    (capturedInitial : input_independent_execution.State) (alternateInitial : CoreState)
    (initialized : input_independent_execution.initial program = .ok (some capturedInitial))
    (related : StateInputsRelated capturedInitial.dependencies capturedInitial.concrete.execution alternateInitial) :
    CoreSafeFrom program alternateInitial := by
  obtain ⟨concrete,verified⟩ := source_input_verify_concrete program certificate accepted
  obtain ⟨initial,frame,coreInitialized,projected,layout,traces⟩ := source_verify_all_traces program concrete verified
  obtain ⟨checkedInitial,dependenciesInitial⟩ := source_input_initial_parts program capturedInitial initialized
  obtain ⟨_,capturedCore,_,_⟩ := source_initial_parts program capturedInitial.concrete checkedInitial
  have same : initial=capturedInitial.concrete.execution := by
    simpa only [coreInitialized,ok.injEq,core.result.Result.Ok.injEq] using capturedCore
  subst initial
  refine ⟨frame,projected,layout,?_,?_⟩
  · intro events after trace
    obtain ⟨captured,_,_,capturedTrace⟩ := source_input_verify_trace_inputs program certificate accepted
      capturedInitial alternateInitial initialized related trace
    obtain ⟨checked,_,_,safe⟩ := traces events captured.concrete.execution capturedTrace
    exact ⟨stateModel checked.monitor,safe⟩
  · intro after reachable
    obtain ⟨events,trace⟩ := core_reachable_trace program alternateInitial after reachable
    obtain ⟨captured,capturedReachable,afterRelated,_⟩ := source_input_verify_trace_inputs program certificate accepted
      capturedInitial alternateInitial initialized related trace
    have closed := (source_input_verify_reachable program certificate accepted capturedReachable).2
    constructor
    · intro actor running
      have capturedRunning : execution_core.running captured.concrete.execution actor=.ok true := by
        rw [input_running_equal captured.dependencies captured.concrete.execution after afterRelated actor]
        exact running
      obtain ⟨next,stepped,_⟩ := closed.2 actor capturedRunning
      obtain ⟨advanced,concreteStep⟩ := source_input_step_parts program captured next actor stepped
      obtain ⟨event,_,executed,_,_⟩ := source_step_exact program captured.concrete next.concrete actor concreteStep
      obtain ⟨alternateNext,alternateExecuted,_⟩ := source_advance_native_inputs program
        captured.concrete.execution after next.concrete.execution captured.dependencies next.dependencies actor event
        afterRelated advanced executed
      exact ⟨event,alternateNext,alternateExecuted⟩
    · intro terminal
      have capturedTerminal : execution_core.terminal captured.concrete.execution=.ok true := by
        rw [input_terminal_equal captured.dependencies captured.concrete.execution after afterRelated]
        exact terminal
      obtain ⟨bytes,outputs⟩ := (source_good program captured.concrete closed.1).2 capturedTerminal
      obtain ⟨alternateBytes,alternateOutputs,_⟩ := source_outputs_inputs captured.dependencies
        captured.concrete.execution after program.outputs.deref bytes afterRelated outputs
      exact ⟨alternateBytes,alternateOutputs⟩

end MemorySafetyExecution.Native

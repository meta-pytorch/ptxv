/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.CoreSoundness
import MemorySafetyExecution.Native.OwnerRefinement

open Aeneas Aeneas.Std Result PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

open MemorySafety.Source MemorySafetyExecution.Source

inductive NativeTrace {L : Type} (initial : ExecutionMachine L) : List Event → ExecutionMachine L → Prop
  | initial : NativeTrace initial [] initial
  | step {before after : ExecutionMachine L} {events : List Event} {event : Event} {actor : Usize}
      {F E : Type} (dictionary : StepDictionary L F) (callback : F)
      (enabledDictionary : EnabledDictionary L E) (enabledCallback : E) :
      NativeTrace initial events before →
      machine_memory_safety.ExecutionMachine.enabled enabledDictionary before actor enabledCallback = .ok true →
      machine_memory_safety.ExecutionMachine.step dictionary before actor callback = .ok (.Ok event, after) →
      NativeTrace initial (events ++ [event]) after

theorem native_trace_projects {L : Type} (program : CoreProgram) (initial : CoreState)
    {events : List Event} {after : ExecutionMachine L}
    (trace : NativeTrace (.Strict { program, state := initial, fault := none }) events after) :
    ∃ state, after = .Strict { program, state, fault := none } ∧ CoreTrace program initial events state := by
  induction trace with
  | initial => exact ⟨initial, rfl, .initial⟩
  | @step before after events event actor F E dictionary callback enabledDictionary enabledCallback
      prior enabled computed ih =>
    obtain ⟨state, rfl, trace⟩ := ih
    obtain ⟨next, executed, rfl⟩ := source_strict_step_projection dictionary program state actor callback
      event after computed
    exact ⟨next, rfl, .step trace ⟨actor,
      by simpa only [source_strict_enabled] using enabled, executed⟩⟩

def MachineMemorySafe (program : CoreProgram) : Prop :=
  ∀ L : Type, ∃ (initial : CoreState) (frame : memory_safety.Program) (machine : ExecutionMachine L),
    machine_memory_safety.ExecutionMachine.strict L program = .ok (.Ok machine) ∧
    machine = .Strict { program, state := initial, fault := none } ∧
    checked_execution.monitor_program program = .ok (some frame) ∧
    MemorySafety.LayoutSafe (programModel frame) ∧
    (∀ events after, NativeTrace machine events after →
      ∃ memory, SafeTrace (programModel frame) (events.map eventModel) memory) ∧
    (∀ after, NativeReachable machine after → ∃ state,
      after = .Strict { program, state, fault := none } ∧
      (∀ (F E : Type) (dictionary : StepDictionary L F) (callback : F)
          (enabledDictionary : EnabledDictionary L E) (enabledCallback : E) actor,
        machine_memory_safety.ExecutionMachine.enabled enabledDictionary after actor enabledCallback = .ok true →
        ∃ event next, machine_memory_safety.ExecutionMachine.step dictionary after actor callback =
          .ok (.Ok event, next)) ∧
      (execution_core.terminal state = .ok true →
        ∀ (F : Type) (dictionary : OutputDictionary L F) (callback : F),
          ∃ bytes, machine_memory_safety.ExecutionMachine.outputs dictionary after callback = .ok (.Ok bytes)))

theorem core_memory_safe_native_memory_safe (program : CoreProgram)
    (safe : CoreMemorySafe program) : MachineMemorySafe program := by
  obtain ⟨initial, frame, initialized, projectedFrame, layout, traces, noFaults⟩ := safe
  unfold MachineMemorySafe
  intro L
  refine ⟨initial, frame, .Strict { program, state := initial, fault := none },
    source_strict_constructor program initial initialized, rfl, projectedFrame, layout, ?_, ?_⟩
  · intro events after trace
    obtain ⟨state, _, projected⟩ := native_trace_projects program initial trace
    exact traces events state projected
  · intro after reachable
    obtain ⟨state, rfl, projected⟩ := native_reachable_projects program initial after reachable
    obtain ⟨enabledSafe, outputsSafe⟩ := noFaults state projected
    refine ⟨state, rfl, ?_, ?_⟩
    · intro F E dictionary callback enabledDictionary enabledCallback actor enabled
      have running : execution_core.running state actor = .ok true := by
        simpa only [source_strict_enabled] using enabled
      obtain ⟨event, next, computed⟩ := enabledSafe actor running
      exact ⟨event, .Strict { program, state := next, fault := none },
        source_strict_step dictionary program state next actor callback event computed⟩
    · intro terminal F dictionary callback
      obtain ⟨bytes, observed⟩ := outputsSafe terminal
      exact ⟨bytes, by rw [source_strict_outputs dictionary program state callback terminal]; exact observed⟩

theorem source_verify_machine_memory_safe (program : CoreProgram)
    (certificate : Slice checked_execution.State)
    (accepted : checked_execution.verify program certificate = .ok true) : MachineMemorySafe program :=
  core_memory_safe_native_memory_safe program (source_verify_core_memory_safe program certificate accepted)

theorem source_check_safe_machine_memory_safe (program : CoreProgram) (cap : Usize)
    (report : checked_execution.Report) (computed : checked_execution.check program cap = .ok report)
    (accepted : report.status = .Safe) : MachineMemorySafe program :=
  core_memory_safe_native_memory_safe program
    (source_check_safe_core_memory_safe program cap report computed accepted)

end MemorySafetyExecution.Native

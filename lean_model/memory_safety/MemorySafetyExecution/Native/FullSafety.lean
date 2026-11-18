/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.RunSafety
import MemorySafetyExecution.TraceSafety
import MemorySafetyExecution.Native.InstructionAccess

open Aeneas Aeneas.Std Result PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source MemorySafetyExecution.Source

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

/-- Concrete byte safety, race freedom, progress, and terminal output validity. -/
def FullCoreSafetyFrom (program : CoreProgram) (initial : CoreState) : Prop :=
  (∀ state,CoreReachable program initial state →
    (∀ actor,execution_core.running state actor=.ok true →
      ∃ event after,execution_core.execute program state actor=.ok (.Ok event,after) ∧
        ConcreteMemorySafety program state event after ∧
        ExecutionInstructionAccessComplete program state actor event) ∧
    (execution_core.terminal state=.ok true →
      ∃ bytes,execution_core.outputs state program.outputs.deref=.ok (.Ok bytes))) ∧
  ∀ events state,CoreTrace program initial events state →
    ((events.map eventModel).flatMap MemorySafetyExecution.readWriteRecords).Pairwise
      (fun before after=>¬MemorySafety.Conflict before after)

/-- The actual owned ExecutionMachine constructor and every native callback and schedule. -/
def FullMachineMemorySafe (program : CoreProgram) : Prop :=
  ∀ L : Type,∃ (initial : CoreState) (machine : ExecutionMachine L),
    machine_memory_safety.ExecutionMachine.strict L program=.ok (.Ok machine) ∧
    machine=.Strict {program,state:=initial,fault:=none} ∧
    (∀ after,NativeReachable machine after →∃ state,
      after=.Strict {program,state,fault:=none} ∧
      (∀ (F E : Type) (dictionary : StepDictionary L F) (callback : F)
          (enabledDictionary : EnabledDictionary L E) (enabledCallback : E) actor,
        machine_memory_safety.ExecutionMachine.enabled enabledDictionary after actor enabledCallback=.ok true →
        ∃ event next nextState,
          machine_memory_safety.ExecutionMachine.step dictionary after actor callback=.ok (.Ok event,next) ∧
          next=.Strict {program,state:=nextState,fault:=none} ∧
          ConcreteMemorySafety program state event nextState ∧
          ExecutionInstructionAccessComplete program state actor event) ∧
      (execution_core.terminal state=.ok true →
        ∀ (F : Type) (dictionary : OutputDictionary L F) (callback : F),
          ∃ bytes,machine_memory_safety.ExecutionMachine.outputs dictionary after callback=.ok (.Ok bytes))) ∧
    (∀ events after,NativeTrace machine events after →
      ((events.map eventModel).flatMap MemorySafetyExecution.readWriteRecords).Pairwise
        (fun before after=>¬MemorySafety.Conflict before after))

theorem full_core_native_memory_safe (program : CoreProgram) (initial : CoreState)
    (constructed : execution_core.initial_state program=.ok (.Ok initial))
    (safe : FullCoreSafetyFrom program initial) : FullMachineMemorySafe program := by
  intro L
  refine ⟨initial,.Strict {program,state:=initial,fault:=none},source_strict_constructor program initial constructed,rfl,?_,?_⟩
  · intro after reachable
    obtain ⟨state,rfl,projected⟩:=native_reachable_projects program initial after reachable
    obtain ⟨steps,outputs⟩:=safe.1 state projected
    refine ⟨state,rfl,?_,?_⟩
    · intro F E dictionary callback enabledDictionary enabledCallback actor enabled
      have running : execution_core.running state actor=.ok true := by
        simpa only [source_strict_enabled] using enabled
      obtain ⟨event,next,executed,concrete,complete⟩:=steps actor running
      exact ⟨event,.Strict {program,state:=next,fault:=none},next,
        source_strict_step dictionary program state next actor callback event executed,rfl,concrete,complete⟩
    · intro terminal F dictionary callback
      obtain ⟨bytes,observed⟩:=outputs terminal
      exact ⟨bytes,by rw [source_strict_outputs dictionary program state callback terminal];exact observed⟩
  · intro events after trace
    obtain ⟨state,_,projected⟩:=native_trace_projects program initial trace
    exact safe.2 events state projected

end MemorySafetyExecution.Native

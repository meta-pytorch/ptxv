/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Owner

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

private theorem bind_success {α β : Type} (computation : Result α) (next : α → Result β)
    (value : β) (accepted : Aeneas.Std.bind computation next = .ok value) :
    ∃ input, computation = .ok input ∧ next input = .ok value := by
  cases computation with
  | ret input => exact ⟨input, rfl, by simpa only [bind_ok] using accepted⟩
  | vis effect continuation => simp only [bind_vis, vis_not_ok] at accepted
  | div => simp only [bind_div, div_not_ok] at accepted

theorem source_step_is_owned {L F : Type} (dictionary : StepDictionary L F)
    (machine : ExecutionMachine L) (actor : Usize) (callback : F) :
    machine_memory_safety.ExecutionMachine.step dictionary machine actor callback =
      machine_memory_safety.ExecutionMachine.step_owned dictionary machine actor callback := by
  rfl

theorem source_strict_constructor {L : Type} (program : CoreProgram) (initial : CoreState)
    (computed : ptxv_memory_write.execution_core.initial_state program = .ok (.Ok initial)) :
    machine_memory_safety.ExecutionMachine.strict L program =
      .ok (.Ok (.Strict { program, state := initial, fault := none })) := by
  simp [machine_memory_safety.ExecutionMachine.strict, computed]

theorem source_strict_enabled {L F : Type} (dictionary : EnabledDictionary L F)
    (program : CoreProgram) (state : CoreState) (actor : Usize) (callback : F) :
    machine_memory_safety.ExecutionMachine.enabled dictionary
      (.Strict { program, state, fault := none } : ExecutionMachine L) actor callback =
      ptxv_memory_write.execution_core.running state actor := by
  rfl

theorem source_strict_step {L F : Type} (dictionary : StepDictionary L F)
    (program : CoreProgram) (state next : CoreState) (actor : Usize) (callback : F)
    (event : Event)
    (computed : ptxv_memory_write.execution_core.execute program state actor = .ok (.Ok event, next)) :
    machine_memory_safety.ExecutionMachine.step dictionary
      (.Strict { program, state, fault := none } : ExecutionMachine L) actor callback =
      .ok (.Ok event, .Strict { program, state := next, fault := none }) := by
  simp [source_step_is_owned, machine_memory_safety.ExecutionMachine.step_owned, computed]

theorem source_strict_step_projection {L F : Type} (dictionary : StepDictionary L F)
    (program : CoreProgram) (state : CoreState) (actor : Usize) (callback : F)
    (event : Event) (after : ExecutionMachine L)
    (computed : machine_memory_safety.ExecutionMachine.step dictionary
      (.Strict { program, state, fault := none } : ExecutionMachine L) actor callback =
      .ok (.Ok event, after)) :
    ∃ next, ptxv_memory_write.execution_core.execute program state actor = .ok (.Ok event, next) ∧
      after = .Strict { program, state := next, fault := none } := by
  simp only [source_step_is_owned, machine_memory_safety.ExecutionMachine.step_owned] at computed
  obtain ⟨⟨result, next⟩, run, same⟩ := bind_success _ _ _ computed
  cases result with
  | Ok value =>
    change Result.ok (core.result.Result.Ok value,
      machine_memory_safety.ExecutionMachine.Strict { program, state := next, fault := none }) =
      Result.ok (core.result.Result.Ok event, after) at same
    simp only [ok.injEq, Prod.mk.injEq, core.result.Result.Ok.injEq] at same
    rcases same with ⟨rfl, same⟩
    exact ⟨next, run, same.symm⟩
  | Err fault =>
    change Result.ok (core.result.Result.Err fault,
      machine_memory_safety.ExecutionMachine.Strict { program, state := next, fault := some fault }) =
      Result.ok (core.result.Result.Ok event, after) at same
    simp at same

theorem source_strict_outputs {L F : Type} (dictionary : OutputDictionary L F)
    (program : CoreProgram) (state : CoreState) (callback : F)
    (terminal : ptxv_memory_write.execution_core.terminal state = .ok true) :
    machine_memory_safety.ExecutionMachine.outputs dictionary
      (.Strict { program, state, fault := none } : ExecutionMachine L) callback =
      ptxv_memory_write.execution_core.outputs state (alloc.vec.Vec.deref program.outputs) := by
  simp only [machine_memory_safety.ExecutionMachine.outputs, terminal, bind_ok, ite_true]

theorem source_strict_outputs_postjoin {L F : Type} (dictionary : OutputDictionary L F)
    (program : CoreProgram) (state : CoreState) (callback : F) (bytes : alloc.vec.Vec U8)
    (computed : machine_memory_safety.ExecutionMachine.outputs dictionary
      (.Strict { program, state, fault := none } : ExecutionMachine L) callback = .ok (.Ok bytes)) :
    ptxv_memory_write.execution_core.terminal state = .ok true ∧
      ptxv_memory_write.execution_core.outputs state (alloc.vec.Vec.deref program.outputs) =
        .ok (.Ok bytes) := by
  simp only [machine_memory_safety.ExecutionMachine.outputs] at computed
  obtain ⟨terminal, checked, rest⟩ := bind_success _ _ _ computed
  cases terminal with
  | false => simp at rest
  | true => exact ⟨checked, by simpa only [ite_true] using rest⟩

theorem native_reachable_projects {L : Type} (program : CoreProgram) (initial : CoreState)
    (after : ExecutionMachine L)
    (reachable : NativeReachable (.Strict { program, state := initial, fault := none }) after) :
    ∃ state, after = .Strict { program, state, fault := none } ∧
      CoreReachable program initial state := by
  induction reachable with
  | initial => exact ⟨initial, rfl, .initial⟩
  | @step before after event actor F E dictionary callback enabledDictionary enabledCallback
      prior enabled computed ih =>
    obtain ⟨state, rfl, reachable⟩ := ih
    obtain ⟨next, run, rfl⟩ := source_strict_step_projection dictionary program state actor callback
      event after computed
    exact ⟨next, rfl, .step reachable ⟨actor,
      by simpa only [source_strict_enabled] using enabled, run⟩⟩

end MemorySafetyExecution.Native

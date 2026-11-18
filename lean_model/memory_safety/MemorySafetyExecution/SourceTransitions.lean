/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourceApplication

open Aeneas Aeneas.Std Result PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Source

open MemorySafety.Source

theorem source_initial_parts (program : execution_core.Program) (state : checked_execution.State)
    (computed : checked_execution.initial program = .ok (some state)) :
    ∃ frame,
      execution_core.initial_state program = .ok (.Ok state.execution) ∧
      checked_execution.monitor_program program = .ok (some frame) ∧
      memory_safety.initial frame = .ok (some state.monitor) := by
  unfold checked_execution.initial at computed
  obtain ⟨result, initialized, remaining⟩ := source_bind_success _ _ _ computed
  cases result with
  | Err fault => simp at remaining
  | Ok execution =>
    obtain ⟨result, projected, remaining⟩ := source_bind_success _ _ _ remaining
    cases result with
    | none => simp at remaining
    | some frame =>
      obtain ⟨result, monitored, remaining⟩ := source_bind_success _ _ _ remaining
      cases result with
      | none => simp at remaining
      | some monitor =>
        simp only [ok.injEq, Option.some.injEq] at remaining
        subst state
        exact ⟨frame, initialized, projected, monitored⟩

theorem source_good (program : execution_core.Program) (state : checked_execution.State)
    (computed : checked_execution.good program state = .ok true) :
    state.monitor.bad = false ∧
      (execution_core.terminal state.execution = .ok true →
        ∃ bytes, execution_core.outputs state.execution program.outputs.deref = .ok (.Ok bytes)) := by
  unfold checked_execution.good at computed
  cases bad : state.monitor.bad with
  | true => simp [bad] at computed
  | false =>
    simp only [bad, Bool.false_eq_true, ite_false] at computed
    obtain ⟨terminal, tested, remaining⟩ := source_bind_success _ _ _ computed
    refine ⟨rfl, ?_⟩
    intro finished
    have same : terminal = true := by simpa only [tested, ok.injEq] using finished
    subst terminal
    simp only [ite_true] at remaining
    obtain ⟨result, observed, remaining⟩ := source_bind_success _ _ _ remaining
    cases result with
    | Err fault => simp at remaining
    | Ok bytes => exact ⟨bytes, observed⟩

theorem source_running_bounds (state : execution_core.State) (actor : Usize)
    (computed : execution_core.running state actor = .ok true) :
    actor.val < state.threads.val.length := by
  unfold execution_core.running at computed
  by_cases inside : actor < alloc.vec.Vec.len state.threads
  · scalar_tac
  · simp [inside] at computed

theorem source_step_parts (program : execution_core.Program) (before after : checked_execution.State)
    (actor : Usize) (computed : checked_execution.step program before actor = .ok (some after)) :
    ∃ copied event frame,
      execution_core.snapshot_state before.execution = .ok copied ∧
      execution_core.execute program copied actor = .ok (.Ok event, after.execution) ∧
      checked_execution.monitor_program program = .ok (some frame) ∧
      memory_safety.update_memory frame before.monitor event = .ok after.monitor := by
  unfold checked_execution.step at computed
  obtain ⟨copied, snapshot, remaining⟩ := source_bind_success _ _ _ computed
  obtain ⟨pair, executed, remaining⟩ := source_bind_success _ _ _ remaining
  rcases pair with ⟨result, execution⟩
  cases result with
  | Err fault => simp at remaining
  | Ok event =>
    obtain ⟨result, projected, remaining⟩ := source_bind_success _ _ _ remaining
    cases result with
    | none => simp at remaining
    | some frame =>
      obtain ⟨monitor, updated, remaining⟩ := source_bind_success _ _ _ remaining
      simp only [ok.injEq, Option.some.injEq] at remaining
      subst after
      exact ⟨copied, event, frame, snapshot, executed, projected, updated⟩

theorem source_step_memory_safe (program : execution_core.Program)
    (before after : checked_execution.State) (actor : Usize)
    (computed : checked_execution.step program before actor = .ok (some after))
    (clean : after.monitor.bad = false) :
    ∃ copied event frame,
      execution_core.snapshot_state before.execution = .ok copied ∧
      execution_core.execute program copied actor = .ok (.Ok event, after.execution) ∧
      checked_execution.monitor_program program = .ok (some frame) ∧
      MemorySafety.EventSafe (programModel frame) (stateModel before.monitor) (eventModel event) ∧
      stateModel after.monitor = MemorySafety.updateMemory (programModel frame)
        (stateModel before.monitor) (eventModel event) := by
  obtain ⟨copied, event, frame, snapshot, executed, projected, updated⟩ :=
    source_step_parts program before after actor computed
  exact ⟨copied, event, frame, snapshot, executed, projected,
    source_update_memory_agrees frame before.monitor event after.monitor updated clean⟩

end MemorySafetyExecution.Source

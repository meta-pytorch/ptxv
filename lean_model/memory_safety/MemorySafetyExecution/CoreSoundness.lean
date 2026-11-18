/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.SourceCertificate
import MemorySafetyExecution.SourceTransitions
import MemorySafetyExecution.Native.StateSnapshot
import MemorySafety.SourceInitial

open Aeneas Aeneas.Std Result PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Source

open MemorySafety.Source

theorem source_step_exact (program : execution_core.Program)
    (before after : checked_execution.State) (actor : Usize)
    (computed : checked_execution.step program before actor = .ok (some after)) :
    ∃ event frame,
      execution_core.execute program before.execution actor = .ok (.Ok event, after.execution) ∧
      checked_execution.monitor_program program = .ok (some frame) ∧
      memory_safety.update_memory frame before.monitor event = .ok after.monitor := by
  obtain ⟨copied, event, frame, snapshot, executed, projected, updated⟩ :=
    source_step_parts program before after actor computed
  have same : before.execution = copied := by
    simpa only [Native.source_snapshot_state, ok.injEq] using snapshot
  rw [← same] at executed
  exact ⟨event, frame, executed, projected, updated⟩

theorem source_verify_core_reachable (program : execution_core.Program)
    (certificate : Slice checked_execution.State)
    (accepted : checked_execution.verify program certificate = .ok true) :
    ∃ initial,
      execution_core.initial_state program = .ok (.Ok initial) ∧
      ∀ state, Native.CoreReachable program initial state →
        ∃ checked, SourceReachable program checked ∧ checked.execution = state := by
  obtain ⟨⟨initial, initialized, _⟩, _⟩ := source_verify_exact_closure program certificate accepted
  obtain ⟨_, initializedCore, _, _⟩ := source_initial_parts program initial initialized
  refine ⟨initial.execution, initializedCore, ?_⟩
  intro state reachable
  induction reachable with
  | initial => exact ⟨initial, .initial initialized, rfl⟩
  | @step before after event reachable transition ih =>
    obtain ⟨checked, checkedReachable, same⟩ := ih
    obtain ⟨actor, enabled, executed⟩ := transition
    rw [← same] at enabled executed
    obtain ⟨next, stepped, _, _⟩ :=
      source_verify_actor program certificate accepted checkedReachable actor enabled
    obtain ⟨actual, _, execution, _, _⟩ := source_step_exact program checked next actor stepped
    have result : actual = event ∧ next.execution = after := by
      simpa only [execution, ok.injEq, Prod.mk.injEq, core.result.Result.Ok.injEq] using executed
    exact ⟨next, .step checkedReachable enabled stepped, result.2⟩

theorem source_verify_core_no_fault (program : execution_core.Program)
    (certificate : Slice checked_execution.State)
    (accepted : checked_execution.verify program certificate = .ok true) :
    ∃ initial,
      execution_core.initial_state program = .ok (.Ok initial) ∧
      ∀ state, Native.CoreReachable program initial state →
        (∀ actor, execution_core.running state actor = .ok true →
          ∃ event after, execution_core.execute program state actor = .ok (.Ok event, after)) ∧
        (execution_core.terminal state = .ok true →
          ∃ bytes, execution_core.outputs state program.outputs.deref = .ok (.Ok bytes)) := by
  obtain ⟨initial, initialized, projects⟩ := source_verify_core_reachable program certificate accepted
  refine ⟨initial, initialized, ?_⟩
  intro state reachable
  obtain ⟨checked, checkedReachable, same⟩ := projects state reachable
  rw [← same]
  constructor
  · intro actor enabled
    obtain ⟨next, stepped, _, _⟩ :=
      source_verify_actor program certificate accepted checkedReachable actor enabled
    obtain ⟨event, _, executed, _, _⟩ := source_step_exact program checked next actor stepped
    exact ⟨event, next.execution, executed⟩
  · exact (source_good program checked
      (source_verify_reachable program certificate accepted checkedReachable).2).2

inductive CoreTrace (program : execution_core.Program) (initial : execution_core.State) :
    List memory_safety.Event → execution_core.State → Prop
  | initial : CoreTrace program initial [] initial
  | step {events before event after} : CoreTrace program initial events before →
      Native.CoreStep program before event after → CoreTrace program initial (events ++ [event]) after

inductive SafeTrace (frame : MemorySafety.Program) : List MemorySafety.Event → MemorySafety.State → Prop
  | initial : SafeTrace frame [] (MemorySafety.initialState frame)
  | step {events before event} : SafeTrace frame events before →
      MemorySafety.EventSafe frame before event →
      SafeTrace frame (events ++ [event]) (MemorySafety.updateMemory frame before event)

theorem source_verify_all_traces (program : execution_core.Program)
    (certificate : Slice checked_execution.State)
    (accepted : checked_execution.verify program certificate = .ok true) :
    ∃ initial frame,
      execution_core.initial_state program = .ok (.Ok initial) ∧
      checked_execution.monitor_program program = .ok (some frame) ∧
      MemorySafety.LayoutSafe (programModel frame) ∧
      ∀ events state, CoreTrace program initial events state →
        ∃ checked, SourceReachable program checked ∧ checked.execution = state ∧
          SafeTrace (programModel frame) (events.map eventModel) (stateModel checked.monitor) := by
  obtain ⟨⟨initial, initialized, _⟩, _⟩ := source_verify_exact_closure program certificate accepted
  obtain ⟨frame, initializedCore, projected, initializedMonitor⟩ :=
    source_initial_parts program initial initialized
  have initialFacts : MemorySafety.LayoutSafe (programModel frame) ∧
      stateModel initial.monitor = MemorySafety.initialState (programModel frame) := by
    have specification := source_initial frame
    have result : ∀ state, initial.monitor = state →
        MemorySafety.LayoutSafe (programModel frame) ∧
        stateModel state = MemorySafety.initialState (programModel frame) := by
      simpa only [initializedMonitor, WP.spec_ok, Option.some.injEq] using specification
    exact result initial.monitor rfl
  refine ⟨initial.execution, frame, initializedCore, projected, initialFacts.1, ?_⟩
  intro events state trace
  induction trace with
  | initial =>
    refine ⟨initial, .initial initialized, rfl, ?_⟩
    simpa only [List.map_nil, initialFacts.2] using SafeTrace.initial (frame := programModel frame)
  | @step events before event after trace transition ih =>
    obtain ⟨checked, checkedReachable, same, memoryTrace⟩ := ih
    obtain ⟨actor, enabled, executed⟩ := transition
    rw [← same] at enabled executed
    obtain ⟨next, stepped, _, good⟩ :=
      source_verify_actor program certificate accepted checkedReachable actor enabled
    obtain ⟨actual, otherFrame, actualExecuted, otherProjected, updated⟩ :=
      source_step_exact program checked next actor stepped
    have frameSame : frame = otherFrame := by
      simpa only [projected, ok.injEq, Option.some.injEq] using otherProjected
    subst otherFrame
    have result : actual = event ∧ next.execution = after := by
      simpa only [actualExecuted, ok.injEq, Prod.mk.injEq, core.result.Result.Ok.injEq] using executed
    rw [result.1] at updated
    obtain ⟨safe, modelUpdated⟩ := source_update_memory_agrees frame checked.monitor event next.monitor
      updated (source_good program next good).1
    refine ⟨next, .step checkedReachable enabled stepped, result.2, ?_⟩
    rw [List.map_append, List.map_singleton, modelUpdated]
    exact .step memoryTrace safe

def CoreMemorySafe (program : execution_core.Program) : Prop :=
  ∃ initial frame,
    execution_core.initial_state program = .ok (.Ok initial) ∧
    checked_execution.monitor_program program = .ok (some frame) ∧
    MemorySafety.LayoutSafe (programModel frame) ∧
    (∀ events state, CoreTrace program initial events state →
      ∃ memory, SafeTrace (programModel frame) (events.map eventModel) memory) ∧
    (∀ state, Native.CoreReachable program initial state →
      (∀ actor, execution_core.running state actor = .ok true →
        ∃ event after, execution_core.execute program state actor = .ok (.Ok event, after)) ∧
      (execution_core.terminal state = .ok true →
        ∃ bytes, execution_core.outputs state program.outputs.deref = .ok (.Ok bytes)))

theorem source_verify_core_memory_safe (program : execution_core.Program)
    (certificate : Slice checked_execution.State)
    (accepted : checked_execution.verify program certificate = .ok true) : CoreMemorySafe program := by
  obtain ⟨initial, frame, initialized, projected, layout, traces⟩ :=
    source_verify_all_traces program certificate accepted
  obtain ⟨other, otherInitialized, noFaults⟩ := source_verify_core_no_fault program certificate accepted
  have same : initial = other := by
    simpa only [initialized, ok.injEq, core.result.Result.Ok.injEq] using otherInitialized
  subst other
  refine ⟨initial, frame, initialized, projected, layout, ?_, noFaults⟩
  intro events state trace
  obtain ⟨checked, _, _, safe⟩ := traces events state trace
  exact ⟨stateModel checked.monitor, safe⟩

theorem source_check_safe_core_memory_safe (program : execution_core.Program) (cap : Usize)
    (report : checked_execution.Report) (computed : checked_execution.check program cap = .ok report)
    (accepted : report.status = .Safe) : CoreMemorySafe program := by
  obtain ⟨certificate, verified⟩ := source_check_requires_certificate program cap report computed accepted
  exact source_verify_core_memory_safe program certificate verified

end MemorySafetyExecution.Source

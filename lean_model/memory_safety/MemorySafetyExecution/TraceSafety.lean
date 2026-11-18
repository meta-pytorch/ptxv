/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Soundness
import MemorySafetyExecution.Native.ConcreteSafety
import MemorySafetyExecution.MonitorFrame

open Aeneas Aeneas.Std Result PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution

open MemorySafety Source Native

def readWriteRecords : MemorySafety.Event → List MemorySafety.AccessRecord
  | .access actor request => [MemorySafety.accessRecord actor request]
  | .control | .free _ _ => []

private theorem update_history_preserves (frame : MemorySafety.Program)
    (memory : MemorySafety.State) (event : MemorySafety.Event)
    (safe : MemorySafety.EventSafe frame memory event) :
    ∀ record ∈ memory.history ++ readWriteRecords event,
      record ∈ (MemorySafety.updateMemory frame memory event).history := by
  cases event with
  | control => simp [MemorySafety.updateMemory, readWriteRecords]
  | access actor request =>
    change MemorySafety.AccessSafe frame memory actor request at safe
    simpa only [MemorySafety.updateMemory, safe, ite_true, readWriteRecords] using
      (fun record member => member : ∀ record ∈ memory.history ++
        [MemorySafety.accessRecord actor request], record ∈ memory.history ++
        [MemorySafety.accessRecord actor request])
  | free actor allocation =>
    cases selected : frame.allocations[allocation]? with
    | none => simp [MemorySafety.EventSafe, MemorySafety.FreeSafe, selected] at safe
    | some block =>
      change MemorySafety.FreeSafe frame memory actor allocation at safe
      intro record member
      simp only [readWriteRecords, List.append_nil] at member
      simp only [MemorySafety.updateMemory, safe, ite_true, selected]
      exact List.mem_append_left _ member

theorem safe_trace_records (frame : MemorySafety.Program)
    (events : List MemorySafety.Event) (memory : MemorySafety.State)
    (trace : SafeTrace frame events memory) :
    (∀ record ∈ events.flatMap readWriteRecords, record ∈ memory.history) ∧
      (events.flatMap readWriteRecords).Pairwise (fun before after => ¬Conflict before after) := by
  induction trace with
  | initial => simp [MemorySafety.initialState]
  | @step events before event prior safe ih =>
    rw [List.flatMap_append, List.flatMap_singleton]
    constructor
    · intro record member
      apply update_history_preserves frame before event safe record
      rcases List.mem_append.mp member with member | member
      · exact List.mem_append_left _ (ih.1 record member)
      · exact List.mem_append_right _ member
    · apply List.pairwise_append.mpr
      refine ⟨ih.2, ?_, ?_⟩
      · cases event <;> simp [readWriteRecords]
      · intro previous previousMember current currentMember
        cases event with
        | control => simp [readWriteRecords] at currentMember
        | free actor allocation => simp [readWriteRecords] at currentMember
        | access actor request =>
          have same : current = MemorySafety.accessRecord actor request := by
            simpa only [readWriteRecords, List.mem_singleton] using currentMember
          subst current
          exact MemorySafety.safe_access_conflicts_with_no_previous_access frame before actor
            request safe previous (ih.1 previous previousMember)

theorem accepted_native_traces_have_no_conflicts
    (program : execution_core.Program) (safe : MachineMemorySafe program) :
    ∀ L : Type, ∃ machine : ExecutionMachine L,
      machine_memory_safety.ExecutionMachine.strict L program = .ok (.Ok machine) ∧
      ∀ events after, NativeTrace machine events after →
        ((events.map MemorySafety.Source.eventModel).flatMap readWriteRecords).Pairwise
          (fun before after => ¬Conflict before after) := by
  intro L
  obtain ⟨initial, frame, machine, constructed, _, _, _, traces, _⟩ := safe L
  refine ⟨machine, constructed, ?_⟩
  intro events after trace
  obtain ⟨memory, safeTrace⟩ := traces events after trace
  exact (safe_trace_records _ _ memory safeTrace).2

end MemorySafetyExecution

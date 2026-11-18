/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.AccessSafety

set_option autoImplicit false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.MachineModel

def Execution (program : Program) (initial : State)
    (states : Nat → State) (events : Nat → MemorySafety.Event) : Prop :=
  states 0 = initial ∧
    ∀ index, Step program (states index) (events index) (states (index + 1))

def eventPrefix (events : Nat → MemorySafety.Event) : Nat → List MemorySafety.Event
  | 0 => []
  | length + 1 => eventPrefix events length ++ [events length]

theorem execution_prefix (program : Program) (initial : State)
    (states : Nat → State) (events : Nat → MemorySafety.Event)
    (execution : Execution program initial states events) (length : Nat) :
    Trace program initial (eventPrefix events length) (states length) := by
  induction length with
  | zero => simpa only [eventPrefix, execution.1] using (Trace.initial (program := program) (initial := initial))
  | succ length ih => exact .next ih (execution.2 length)

theorem execution_reachable (program : Program) (initial : State)
    (states : Nat → State) (events : Nat → MemorySafety.Event)
    (execution : Execution program initial states events) (index : Nat) :
    Reachable program initial (states index) :=
  trace_reachable program initial (eventPrefix events index) (states index)
    (execution_prefix program initial states events execution index)

theorem event_mem_prefix (events : Nat → MemorySafety.Event) (index length : Nat)
    (before : index < length) : events index ∈ eventPrefix events length := by
  induction length with
  | zero => omega
  | succ length ih =>
    by_cases same : index = length
    · subst index
      exact List.mem_append_right _ (List.mem_singleton_self _)
    · exact List.mem_append_left _ (ih (by omega))

/-- Any two accesses in an unbounded execution are covered by a finite checked prefix. -/
theorem memory_safe_execution_no_conflict (program : Program) (initial : State)
    (safe : MemorySafeFrom program initial)
    (states : Nat → State) (events : Nat → MemorySafety.Event)
    (execution : Execution program initial states events)
    (first second : Nat) (ordered : first < second)
    (before after : MemorySafety.AccessRecord)
    (earlier : before ∈ readWriteRecords (events first))
    (later : after ∈ readWriteRecords (events second)) :
    ¬MemorySafety.Conflict before after := by
  have checked := safe.2 (eventPrefix events (second + 1)) (states (second + 1))
    (execution_prefix program initial states events execution (second + 1))
  rw [eventPrefix, List.flatMap_append, List.flatMap_singleton] at checked
  apply (List.pairwise_append.mp checked).2.2 before _ after later
  exact List.mem_flatMap.mpr ⟨events first, event_mem_prefix events first second ordered, earlier⟩

theorem memory_safe_execution_running_instruction (program : Program) (initial : State)
    (safe : MemorySafeFrom program initial)
    (states : Nat → State) (events : Nat → MemorySafety.Event)
    (execution : Execution program initial states events)
    (index actor : Nat) (thread : Thread)
    (selected : (states index).threads[actor]? = some thread) (running : thread.running = true) :
    ∃ instruction action, program.code[thread.pc]? = some instruction ∧
      ValueFlow.evaluate thread.registers thread.ids instruction = some action ∧
      ActionSafe program (states index) action :=
  memory_safe_running_instruction program initial (states index) safe
    (execution_reachable program initial states events execution index) actor thread selected running

end MemorySafetyExecution.MachineModel

namespace MemorySafetyExecution.Source

open Aeneas Aeneas.Std Result PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

theorem source_check_reports_safe_unbounded_no_conflict {F : Type _}
    (producerInterface : core.ops.function.FnOnce F
      (execution_core.Program × Usize) strict_memory_source.Candidate)
    (source entry : Slice U8) (manifest : ptxv_case.Manifest) (cap : Usize) (producer : F)
    (report : strict_memory_source.Report)
    (computed : strict_memory_source.check producerInterface source entry manifest cap producer = .ok report)
    (accepted : strict_memory_source.Report.status_name report = .ok (toStr "safe")) :
    ∃ program, PtxFrontendProgram.SourceProgram source.val entry.val manifest program ∧
      ∀ alternate, Native.MemoryInstance program alternate →
      ∀ (states : Nat → MachineModel.State) (events : Nat → MemorySafety.Event),
        MachineModel.Execution (Native.Independent.programModel alternate)
          (Native.Independent.stateModel alternate.initial) states events →
        ∀ (first second : Nat), first < second →
        ∀ (before after : MemorySafety.AccessRecord),
          before ∈ readWriteRecords (events first) →
          after ∈ readWriteRecords (events second) →
          ¬MemorySafety.Conflict before after := by
  obtain ⟨program, sourceMeaning, _, _, allInstances⟩ :=
    source_check_reports_safe_full_memory_safety producerInterface source entry manifest cap producer
      report computed accepted
  refine ⟨program, sourceMeaning, ?_⟩
  intro alternate instanceOf states events execution first second ordered before after earlier later
  exact MachineModel.memory_safe_execution_no_conflict _ _ (allInstances alternate instanceOf).2
    states events execution first second ordered before after earlier later

end MemorySafetyExecution.Source

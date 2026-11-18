/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.IndependentSemantics
import MemorySafetyExecution.TraceSafety

set_option autoImplicit false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.MachineModel

inductive Trace (program : Program) (initial : State) : List MemorySafety.Event → State → Prop
  | initial : Trace program initial [] initial
  | next {events before event after} : Trace program initial events before →
      Step program before event after → Trace program initial (events++[event]) after

def Terminal (state : State) : Prop :=
  ∀ thread∈state.threads,thread.running=false

def OutputsInitialized (program : Program) (state : State) : Prop :=
  ∀ object∈program.outputs,∃ memory,state.allocations[object]?=some memory ∧
    memory.live=true ∧ memory.defined.length=memory.bytes.length ∧
    ∀ flag∈memory.defined,flag=true

/-- Every running actor can step, every terminal output is initialized, and no trace races. -/
def MemorySafeFrom (program : Program) (initial : State) : Prop :=
  (∀ state,Reachable program initial state →
    (∀ actor thread,state.threads[actor]?=some thread → thread.running=true →
      ∃ event after,step program state actor=some (event,after)) ∧
    (Terminal state → OutputsInitialized program state)) ∧
  ∀ events after,Trace program initial events after →
    (events.flatMap MemorySafetyExecution.readWriteRecords).Pairwise
      (fun before after=>¬MemorySafety.Conflict before after)


theorem step_selected (program : Program) (state : State) (actor : Nat)
    (event : MemorySafety.Event) (after : State)
    (computed : step program state actor=some (event,after)) :
    ∃ thread,state.threads[actor]?=some thread ∧ thread.running=true := by
  unfold step at computed
  cases selected : state.threads[actor]? with
  | none => simp [selected] at computed
  | some thread =>
    cases running : thread.running with
    | false => simp [selected,running] at computed
    | true => exact ⟨thread,rfl,running⟩

theorem trace_reachable (program : Program) (initial : State)
    (events : List MemorySafety.Event) (after : State)
    (trace : Trace program initial events after) : Reachable program initial after := by
  induction trace with
  | initial => exact .initial
  | next trace stepped ih => exact .next ih stepped

theorem reachable_trace (program : Program) (initial after : State)
    (reachable : Reachable program initial after) :
    ∃ events,Trace program initial events after := by
  induction reachable with
  | initial => exact ⟨[],.initial⟩
  | next reachable stepped ih =>
    obtain ⟨events,trace⟩:=ih
    exact ⟨_,.next trace stepped⟩

end MemorySafetyExecution.MachineModel

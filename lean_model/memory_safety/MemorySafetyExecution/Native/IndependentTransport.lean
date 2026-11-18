/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.IndependentObservation
import MemorySafetyExecution.Native.FullSafety

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source MemorySafetyExecution.Source

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native.Independent

def StepRefinement (program : CoreProgram) : Prop :=
  ∀ before after actor event,Canonical before →
    execution_core.execute program before actor=.ok (.Ok event,after) →
    MachineModel.step (programModel program) (stateModel before) actor.val=
      some (eventModel event,stateModel after) ∧ Canonical after

theorem core_trace_reachable (program : CoreProgram) (initial after : CoreState)
    (events : List memory_safety.Event) (trace : CoreTrace program initial events after) :
    CoreReachable program initial after := by
  induction trace with
  | initial => exact .initial
  | step trace stepped ih => exact .step ih stepped

theorem reverse_model_trace (program : CoreProgram) (initial : CoreState)
    (safe : FullCoreSafetyFrom program initial) (canonical : Canonical initial)
    (refinement : StepRefinement program)
    (events : List MemorySafety.Event) (after : MachineModel.State)
    (trace : MachineModel.Trace (programModel program) (stateModel initial) events after) :
    ∃ nativeEvents nativeAfter,CoreTrace program initial nativeEvents nativeAfter ∧
      Canonical nativeAfter ∧ events=nativeEvents.map eventModel ∧ after=stateModel nativeAfter := by
  induction trace with
  | initial => exact ⟨[],initial,.initial,canonical,rfl,rfl⟩
  | @next events before event after prior stepped ih =>
    obtain ⟨nativeEvents,nativeBefore,nativeTrace,beforeCanonical,eventsEq,beforeEq⟩:=ih
    subst before
    obtain ⟨actor,computed⟩:=stepped
    obtain ⟨thread,selected,running⟩:=MachineModel.step_selected (programModel program)
      (stateModel nativeBefore) actor event after computed
    obtain ⟨nativeActor,actorEq,nativeRunning⟩:=source_running_of_model nativeBefore actor thread selected running
    have reachable:=core_trace_reachable program initial nativeBefore nativeEvents nativeTrace
    obtain ⟨nativeEvent,nativeAfter,executed,_,_⟩:=(safe.1 nativeBefore reachable).1 nativeActor nativeRunning
    obtain ⟨modeled,afterCanonical⟩:=refinement nativeBefore nativeAfter nativeActor nativeEvent beforeCanonical executed
    rw [actorEq] at modeled
    have same : eventModel nativeEvent=event ∧ stateModel nativeAfter=after := by
      simpa only [Option.some.injEq,Prod.mk.injEq] using modeled.symm.trans computed
    refine ⟨nativeEvents++[nativeEvent],nativeAfter,
      .step nativeTrace ⟨nativeActor,nativeRunning,executed⟩,afterCanonical,?_,same.2.symm⟩
    simp only [eventsEq,List.map_append,List.map_singleton,same.1]

theorem core_safety_transports (program : CoreProgram) (initial : CoreState)
    (safe : FullCoreSafetyFrom program initial) (canonical : Canonical initial)
    (refinement : StepRefinement program) :
    MachineModel.MemorySafeFrom (programModel program) (stateModel initial) := by
  constructor
  · intro after reachable
    obtain ⟨events,trace⟩:=MachineModel.reachable_trace _ _ _ reachable
    obtain ⟨nativeEvents,nativeAfter,nativeTrace,afterCanonical,_,afterEq⟩:=
      reverse_model_trace program initial safe canonical refinement events after trace
    subst after
    have nativeReachable:=core_trace_reachable program initial nativeAfter nativeEvents nativeTrace
    obtain ⟨steps,outputs⟩:=safe.1 nativeAfter nativeReachable
    constructor
    · intro actor thread selected running
      obtain ⟨nativeActor,actorEq,nativeRunning⟩:=source_running_of_model nativeAfter actor thread selected running
      obtain ⟨event,next,executed,_,_⟩:=steps nativeActor nativeRunning
      obtain ⟨modeled,_⟩:=refinement nativeAfter next nativeActor event afterCanonical executed
      exact ⟨eventModel event,stateModel next,by simpa only [actorEq] using modeled⟩
    · intro terminal
      have actualTerminal:=(source_terminal_iff nativeAfter).mpr terminal
      obtain ⟨bytes,observed⟩:=outputs actualTerminal
      exact source_outputs_initialized program nativeAfter bytes observed
  · intro events after trace
    obtain ⟨nativeEvents,nativeAfter,nativeTrace,_,eventsEq,_⟩:=
      reverse_model_trace program initial safe canonical refinement events after trace
    simpa only [eventsEq] using safe.2 nativeEvents nativeAfter nativeTrace

end MemorySafetyExecution.Native.Independent

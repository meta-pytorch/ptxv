/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.PlacementAdvance
import MemorySafetyExecution.Native.InputTrace

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source MemorySafetyExecution.Source

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

private theorem forall₂_append_event {α β : Type} {relation : α→β→Prop}
    {left : List α} {right : List β} {first : α} {second : β}
    (before : List.Forall₂ relation left right) (event : relation first second) :
    List.Forall₂ relation (left++[first]) (right++[second]) := by
  induction before with
  | nil => exact .cons event .nil
  | cons pair tail ih => exact .cons pair ih

theorem source_input_verify_trace_placements (program alternateProgram : CoreProgram)
    (certificate : Slice input_independent_execution.State)
    (accepted : input_independent_execution.verify program certificate=.ok true)
    (sameCode : program.code=alternateProgram.code)
    (placements : ProgramPlacements program alternateProgram)
    (capturedInitial : input_independent_execution.State) (alternateInitial : CoreState)
    (initialized : input_independent_execution.initial program=.ok (some capturedInitial))
    (related : StateInputsRelated capturedInitial.dependencies capturedInitial.concrete.execution alternateInitial)
    {events : List memory_safety.Event} {alternateAfter : CoreState}
    (trace : CoreTrace alternateProgram alternateInitial events alternateAfter) :
    ∃ capturedAfter capturedEvents,InputSourceReachable program capturedAfter ∧
      StateInputsRelated capturedAfter.dependencies capturedAfter.concrete.execution alternateAfter ∧
      CoreTrace program capturedInitial.concrete.execution capturedEvents capturedAfter.concrete.execution ∧
      List.Forall₂ (EventPlacement program.layouts alternateProgram.layouts) capturedEvents events := by
  induction trace with
  | initial => exact ⟨capturedInitial,[],.initial initialized,related,.initial,.nil⟩
  | @step events before event after trace transition ih =>
    obtain ⟨captured,capturedEvents,reachable,related,capturedTrace,eventRelations⟩:=ih
    obtain ⟨actor,running,executed⟩:=transition
    have capturedRunning : execution_core.running captured.concrete.execution actor=.ok true := by
      rw [input_running_equal captured.dependencies captured.concrete.execution before related actor]
      exact running
    obtain ⟨next,stepped,_⟩:=
      (source_input_verify_reachable program certificate accepted reachable).2.2 actor capturedRunning
    obtain ⟨advanced,concreteStep⟩:=source_input_step_parts program captured next actor stepped
    obtain ⟨capturedEvent,_,capturedExecuted,_,_⟩:=source_step_exact program captured.concrete next.concrete actor concreteStep
    obtain ⟨alternateEvent,alternateNext,alternateExecuted,placement,nextRelated⟩:=
      source_advance_native_placements program alternateProgram captured.concrete.execution before next.concrete.execution
        captured.dependencies next.dependencies actor capturedEvent sameCode placements related advanced capturedExecuted
    have equality : alternateEvent=event ∧ alternateNext=after := by
      simpa only [alternateExecuted,ok.injEq,Prod.mk.injEq,core.result.Result.Ok.injEq] using executed
    rw [equality.1] at placement
    rw [equality.2] at nextRelated
    exact ⟨next,capturedEvents++[capturedEvent],.step reachable capturedRunning stepped,nextRelated,
      .step capturedTrace ⟨actor,capturedRunning,capturedExecuted⟩,forall₂_append_event eventRelations placement⟩

end MemorySafetyExecution.Native

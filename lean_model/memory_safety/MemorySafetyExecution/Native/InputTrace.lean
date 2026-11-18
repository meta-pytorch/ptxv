/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputStep
import MemorySafetyExecution.InputCertificate
import MemorySafetyExecution.CoreSoundness

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source MemorySafetyExecution.Source

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

theorem source_input_step_parts (program : CoreProgram)
    (before after : input_independent_execution.State) (actor : Usize)
    (computed : input_independent_execution.step program before actor = .ok (some after)) :
    input_independent_execution.advance_dependencies program before.concrete.execution before.dependencies actor =
      .ok (some after.dependencies) ∧
    checked_execution.step program before.concrete actor = .ok (some after.concrete) := by
  unfold input_independent_execution.step at computed
  obtain ⟨dependencies,advanced,computed⟩ := source_bind_success _ _ _ computed
  cases dependencies with
  | none => simp at computed
  | some dependencies =>
    obtain ⟨concrete,stepped,computed⟩ := source_bind_success _ _ _ computed
    cases concrete with
    | none => simp at computed
    | some concrete =>
      have equal : ({concrete,dependencies} : input_independent_execution.State)=after := by simpa using computed
      subst after
      exact ⟨advanced,stepped⟩

theorem source_input_verify_trace_inputs (program : CoreProgram)
    (certificate : Slice input_independent_execution.State)
    (accepted : input_independent_execution.verify program certificate = .ok true)
    (capturedInitial : input_independent_execution.State) (alternateInitial : CoreState)
    (initialized : input_independent_execution.initial program = .ok (some capturedInitial))
    (related : StateInputsRelated capturedInitial.dependencies capturedInitial.concrete.execution alternateInitial)
    {events : List memory_safety.Event} {alternateAfter : CoreState}
    (trace : CoreTrace program alternateInitial events alternateAfter) :
    ∃ capturedAfter, InputSourceReachable program capturedAfter ∧
      StateInputsRelated capturedAfter.dependencies capturedAfter.concrete.execution alternateAfter ∧
      CoreTrace program capturedInitial.concrete.execution events capturedAfter.concrete.execution := by
  induction trace with
  | initial => exact ⟨capturedInitial,.initial initialized,related,.initial⟩
  | @step events before event after trace transition ih =>
    obtain ⟨captured,reachable,related,capturedTrace⟩ := ih
    obtain ⟨actor,running,executed⟩ := transition
    have capturedRunning : execution_core.running captured.concrete.execution actor=.ok true := by
      rw [input_running_equal captured.dependencies captured.concrete.execution before related actor]
      exact running
    obtain ⟨next,stepped,_⟩ :=
      (source_input_verify_reachable program certificate accepted reachable).2.2 actor capturedRunning
    obtain ⟨advanced,concreteStep⟩ := source_input_step_parts program captured next actor stepped
    obtain ⟨capturedEvent,_,capturedExecuted,_,_⟩ := source_step_exact program captured.concrete next.concrete actor concreteStep
    obtain ⟨alternateNext,alternateExecuted,nextRelated⟩ := source_advance_native_inputs program
      captured.concrete.execution before next.concrete.execution captured.dependencies next.dependencies actor
      capturedEvent related advanced capturedExecuted
    have equality : capturedEvent=event ∧ alternateNext=after := by
      simpa only [alternateExecuted,ok.injEq,Prod.mk.injEq,core.result.Result.Ok.injEq] using executed
    rw [equality.1] at capturedExecuted
    rw [equality.2] at nextRelated
    exact ⟨next,.step reachable capturedRunning stepped,nextRelated,
      .step capturedTrace ⟨actor,capturedRunning,capturedExecuted⟩⟩

theorem core_reachable_trace (program : CoreProgram) (initial state : CoreState)
    (reachable : CoreReachable program initial state) :
    ∃ events, CoreTrace program initial events state := by
  induction reachable with
  | initial => exact ⟨[],.initial⟩
  | @step before after event prior transition ih =>
    obtain ⟨events,trace⟩ := ih
    exact ⟨events++[event],.step trace transition⟩

end MemorySafetyExecution.Native

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.CommitRuns
import MemorySafetyExecution.Native.PreparedEvaluation

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native

/-- The event covers the evaluated operation, even when its bytes do not change. -/
def ActionAccess (program : CoreProgram) (before : CoreState) (actor : Usize)
    (action : value_flow.Action) (event : memory_safety.Event) : Prop :=
  match action with
  | .Load _ _ bytes _ pointer => ∃ access,
      execution_core.resolve_access program before pointer bytes false=.ok (.Ok access) ∧
      event=.Access actor access.request
  | .Store bytes pointer _ => ∃ access,
      execution_core.resolve_access program before pointer bytes true=.ok (.Ok access) ∧
      event=.Access actor access.request
  | _ => event=.Control

theorem source_prepared_action_access (program : CoreProgram) (before after : CoreState)
    (actor : Usize) (commit : execution_core.Commit) (event : memory_safety.Event)
    (prepared : execution_core.prepare program before actor=.ok (.Ok commit))
    (executed : execution_core.commit_action program before commit=.ok (.Ok event,after)) :
    ActionAccess program before actor commit.action event := by
  obtain ⟨thread,instruction,threadAt,running,instructionAt,commitActor,nextPc,evaluated⟩ :=
    source_prepare_evaluation program before actor commit prepared
  have inside:=(List.getElem?_eq_some_iff.mp threadAt).1
  have commitEq : commit=({actor,next_pc:=commit.next_pc,action:=commit.action}:execution_core.Commit) := by
    cases commit
    simp_all only
  rw [commitEq] at executed
  cases actionEq:commit.action with
  | Next =>
    rw [actionEq,source_commit_next program before actor commit.next_pc thread inside threadAt] at executed
    have meaning : .Control=event ∧ update_thread before actor {thread with pc:=commit.next_pc}=after := by
      simpa only [ok.injEq,Prod.mk.injEq,core.result.Result.Ok.injEq] using executed
    exact meaning.1.symm
  | Halt =>
    rw [actionEq,source_commit_halt program before actor commit.next_pc thread inside threadAt] at executed
    have meaning : .Control=event ∧ update_thread before actor {thread with running:=false}=after := by
      simpa only [ok.injEq,Prod.mk.injEq,core.result.Result.Ok.injEq] using executed
    exact meaning.1.symm
  | Jump target =>
    rw [actionEq] at executed
    obtain ⟨index,cast,available,eventEq,afterEq⟩ := source_jump_commit_components program before after actor
      commit.next_pc target thread event inside threadAt executed
    exact eventEq
  | WriteVirtual destination value =>
    rw [actionEq] at executed
    obtain ⟨registers,written,eventEq,afterEq⟩ := source_virtual_commit_components program before after actor
      commit.next_pc thread destination value event inside threadAt executed
    exact eventEq
  | WritePredicate destination value =>
    rw [actionEq] at executed
    obtain ⟨registers,written,eventEq,afterEq⟩ := source_predicate_commit_components program before after actor
      commit.next_pc thread destination value event inside threadAt executed
    exact eventEq
  | Load destination bits bytes signed pointer =>
    rw [actionEq] at executed
    obtain ⟨access,value,mask,registers,resolved,loaded,maskRun,written,eventEq,afterEq⟩ :=
      source_load_commit_components program before after actor commit.next_pc thread destination bits bytes signed pointer event
        inside threadAt executed
    exact ⟨access,resolved,eventEq⟩
  | Store bytes pointer value =>
    rw [actionEq] at executed
    obtain ⟨access,written,resolved,stored,eventEq,afterEq⟩ := source_store_commit_components program before after actor
      commit.next_pc thread bytes pointer value event inside threadAt executed
    exact ⟨access,resolved,eventEq⟩

def ExecutionAccessComplete (program : CoreProgram) (before : CoreState) (actor : Usize)
    (event : memory_safety.Event) : Prop :=
  ∃ commit, execution_core.prepare program before actor=.ok (.Ok commit) ∧
    ActionAccess program before actor commit.action event

theorem source_execute_access_complete (program : CoreProgram) (before after : CoreState)
    (actor : Usize) (event : memory_safety.Event)
    (executed : execution_core.execute program before actor=.ok (.Ok event,after)) :
    ExecutionAccessComplete program before actor event := by
  unfold execution_core.execute at executed
  obtain ⟨result,prepared,executed⟩:=MemorySafety.Source.source_bind_success _ _ _ executed
  cases result with
  | Err fault => simp at executed
  | Ok commit =>
    exact ⟨commit,prepared,source_prepared_action_access program before after actor commit event prepared executed⟩

end MemorySafetyExecution.Native

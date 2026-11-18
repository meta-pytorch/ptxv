/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.IndependentMemoryLoad
import MemorySafetyExecution.Native.IndependentMemoryStore
import MemorySafetyExecution.Native.IndependentWrites

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native.Independent

theorem source_execute_independent (program : CoreProgram) (before after : CoreState)
    (actor : Usize) (event : memory_safety.Event) (canonical : Canonical before)
    (executed : execution_core.execute program before actor=.ok (.Ok event,after)) :
    MachineModel.step (programModel program) (stateModel before) actor.val=
      some (MemorySafety.Source.eventModel event,stateModel after) ∧ Canonical after := by
  obtain ⟨commit,prepared,committed⟩ := source_execute_prepared program before after actor event executed
  obtain ⟨thread,instruction,selected,running,fetched,commitActor,nextPc,evaluated⟩ :=
    source_prepare_evaluation program before actor commit prepared
  have independent := ValueFlow.Source.source_evaluate thread.registers thread.ids instruction
  simp only [evaluated,WP.spec_ok,ValueFlow.Source.outcomeModel] at independent
  have pureEvaluation : ValueFlow.evaluate (ValueFlow.Source.bankModel thread.registers)
      (ValueFlow.Source.idsModel thread.ids) (ValueFlow.Source.guardedModel instruction)=
      some (ValueFlow.Source.actionModel commit.action) := independent.symm
  have stepMeaning : MachineModel.step (programModel program) (stateModel before) actor.val=
      MachineModel.applyAction (programModel program) (stateModel before) actor.val (threadModel thread)
        (ValueFlow.Source.actionModel commit.action) := by
    simp only [MachineModel.step,programModel,stateModel,threadModel,List.getElem?_map,
      selected,Option.map_some,Option.bind_eq_bind,Option.bind_some,running,ite_true,fetched,pureEvaluation]
  rw [stepMeaning]
  have commitEq : commit=({actor,next_pc:=commit.next_pc,action:=commit.action}:execution_core.Commit) := by
    cases commit
    simp_all only
  rw [commitEq] at committed
  cases actionEq:commit.action with
  | Next =>
    rw [actionEq] at committed
    simpa only [actionEq,ValueFlow.Source.actionModel] using
      source_next_model program before after actor commit.next_pc thread event canonical selected nextPc committed
  | Halt =>
    rw [actionEq] at committed
    simpa only [actionEq,ValueFlow.Source.actionModel] using
      source_halt_model program before after actor commit.next_pc thread event canonical selected committed
  | Jump target =>
    rw [actionEq] at committed
    simpa only [actionEq,ValueFlow.Source.actionModel] using
      source_jump_model program before after actor commit.next_pc target thread event canonical selected committed
  | WriteVirtual destination value =>
    rw [actionEq] at committed
    simpa only [actionEq,ValueFlow.Source.actionModel] using
      source_virtual_model program before after actor commit.next_pc thread event destination value canonical selected nextPc committed
  | WritePredicate destination value =>
    rw [actionEq] at committed
    simpa only [actionEq,ValueFlow.Source.actionModel] using
      source_predicate_model program before after actor commit.next_pc thread event destination value canonical selected nextPc committed
  | Load destination bits bytes signed pointer =>
    rw [actionEq] at committed
    simpa only [actionEq,ValueFlow.Source.actionModel] using
      source_load_model program before after actor commit.next_pc thread event destination bits bytes signed pointer canonical selected nextPc committed
  | Store bytes pointer value =>
    rw [actionEq] at committed
    simpa only [actionEq,ValueFlow.Source.actionModel] using
      source_store_model program before after actor commit.next_pc thread event bytes pointer value canonical selected nextPc committed

theorem source_reachable_canonical (program : CoreProgram) (initial state : CoreState)
    (canonical : Canonical initial) (reachable : CoreReachable program initial state) : Canonical state := by
  induction reachable with
  | initial => exact canonical
  | step reached step induction =>
    obtain ⟨actor,running,executed⟩ := step
    exact (source_execute_independent program _ _ actor _ induction executed).2

end MemorySafetyExecution.Native.Independent

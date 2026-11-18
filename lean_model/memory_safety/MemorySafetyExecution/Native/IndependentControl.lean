/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.IndependentBanks

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native.Independent

private theorem pair_success (event other : memory_safety.Event) (after state : CoreState)
    (same : (Result.ok (.Ok event,state) : Result ((core.result.Result memory_safety.Event Fault)×CoreState))=
      .ok (.Ok other,after)) : other=event ∧ after=state := by
  simpa only [ok.injEq,Prod.mk.injEq,core.result.Result.Ok.injEq] using same.symm

theorem source_next_model (program : CoreProgram) (before after : CoreState) (actor : Usize)
    (pc : U32) (thread : execution_core.ThreadState) (event : memory_safety.Event)
    (canonical : Canonical before) (selected : before.threads.val[actor.val]?=some thread)
    (nextPc : pc.val=thread.pc.val+1)
    (computed : execution_core.commit_action program before {actor,next_pc:=pc,action:=.Next}=.ok (.Ok event,after)) :
    MachineModel.applyAction (programModel program) (stateModel before) actor.val (threadModel thread) .next=
      some (MemorySafety.Source.eventModel event,stateModel after) ∧ Canonical after := by
  have inside := (List.getElem?_eq_some_iff.mp selected).1
  rw [source_commit_next program before actor pc thread inside selected] at computed
  obtain ⟨rfl,rfl⟩ := pair_success _ _ _ _ computed
  constructor
  · simp only [MachineModel.applyAction,MemorySafety.Source.eventModel,model_thread_set,threadModel,nextPc]
  · apply canonical_thread_set before actor _ canonical
    exact canonical thread (List.mem_of_getElem? selected)

theorem source_halt_model (program : CoreProgram) (before after : CoreState) (actor : Usize)
    (pc : U32) (thread : execution_core.ThreadState) (event : memory_safety.Event)
    (canonical : Canonical before) (selected : before.threads.val[actor.val]?=some thread)
    (computed : execution_core.commit_action program before {actor,next_pc:=pc,action:=.Halt}=.ok (.Ok event,after)) :
    MachineModel.applyAction (programModel program) (stateModel before) actor.val (threadModel thread) .halt=
      some (MemorySafety.Source.eventModel event,stateModel after) ∧ Canonical after := by
  have inside := (List.getElem?_eq_some_iff.mp selected).1
  rw [source_commit_halt program before actor pc thread inside selected] at computed
  obtain ⟨rfl,rfl⟩ := pair_success _ _ _ _ computed
  constructor
  · simp only [MachineModel.applyAction,MemorySafety.Source.eventModel,model_thread_set,threadModel]
  · apply canonical_thread_set before actor _ canonical
    exact canonical thread (List.mem_of_getElem? selected)

theorem source_jump_model (program : CoreProgram) (before after : CoreState) (actor : Usize)
    (pc target : U32) (thread : execution_core.ThreadState) (event : memory_safety.Event)
    (canonical : Canonical before) (selected : before.threads.val[actor.val]?=some thread)
    (computed : execution_core.commit_action program before {actor,next_pc:=pc,action:=.Jump target}=.ok (.Ok event,after)) :
    MachineModel.applyAction (programModel program) (stateModel before) actor.val (threadModel thread) (.jump target.val)=
      some (MemorySafety.Source.eventModel event,stateModel after) ∧ Canonical after := by
  have inside := (List.getElem?_eq_some_iff.mp selected).1
  obtain ⟨index,cast,available,rfl,rfl⟩ :=
    source_jump_commit_components program before after actor pc target thread event inside selected computed
  have bounded : target.val  ≤  UScalar.max .Usize := by
    rcases Usize.bounds_eq with narrow|wide <;> scalar_tac
  have indexVal : index.val=target.val := by
    have spec := UScalar.cast_inBounds_spec .Usize target bounded
    simpa only [cast,WP.spec_ok] using spec
  have targetInside : target.val<program.code.val.length := by scalar_tac
  constructor
  · simp only [MachineModel.applyAction,programModel,List.length_map,targetInside,ite_true,
      MemorySafety.Source.eventModel,model_thread_set,threadModel]
  · apply canonical_thread_set before actor _ canonical
    exact canonical thread (List.mem_of_getElem? selected)

end MemorySafetyExecution.Native.Independent

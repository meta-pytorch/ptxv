/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.CommitRules

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

private theorem successful_pair (event other : memory_safety.Event) (state after : CoreState)
    (same : (Result.ok (.Ok event,state) : Result ((core.result.Result memory_safety.Event Fault) × CoreState)) =
      .ok (.Ok other,after)) : other = event ∧ after = state := by
  simpa only [ok.injEq,Prod.mk.injEq,core.result.Result.Ok.injEq] using same.symm

theorem source_virtual_commit_components (program : CoreProgram) (state after : CoreState) (actor : Usize)
    (pc : U32) (thread : execution_core.ThreadState) (destination : U16) (value : value_flow.Value)
    (event : memory_safety.Event) (inside : actor.val < state.threads.val.length)
    (selected : state.threads.val[actor.val]? = some thread)
    (computed : execution_core.commit_action program state {actor,next_pc:=pc,action:=.WriteVirtual destination value} =
      .ok (.Ok event,after)) :
    ∃ registers, value_flow.write_value thread.registers destination value (alloc.vec.Vec.deref program.layouts) =
      .ok (.Ok (),registers) ∧ event = .Control ∧ after=update_thread state actor {thread with registers,pc} := by
  have bound : ¬actor ≥ alloc.vec.Vec.len state.threads := by scalar_tac
  have run := computed
  simp only [execution_core.commit_action,bound,ite_false,alloc.vec.Vec.index_mut_slice_index,
    alloc.vec.Vec.index_mut_usize,alloc.vec.Vec.index_usize,alloc.vec.Vec.getElem?_Nat_eq,
    selected,bind_tc_ok,bind_ok] at run
  obtain ⟨pair,written,remainder⟩ := MemorySafety.Source.source_bind_success _ _ _ run
  rcases pair with ⟨result,registers⟩
  cases result with
  | Err fault => simp at remainder
  | Ok unitValue =>
    cases unitValue
    have replay := source_commit_virtual program state actor pc thread destination value registers inside selected written
    rw [replay] at computed
    obtain ⟨eventEq,afterEq⟩ := successful_pair _ _ _ _ computed
    exact ⟨registers,written,eventEq,afterEq⟩

theorem source_predicate_commit_components (program : CoreProgram) (state after : CoreState) (actor : Usize)
    (pc : U32) (thread : execution_core.ThreadState) (destination : U16) (value : Bool)
    (event : memory_safety.Event) (inside : actor.val < state.threads.val.length)
    (selected : state.threads.val[actor.val]? = some thread)
    (computed : execution_core.commit_action program state {actor,next_pc:=pc,action:=.WritePredicate destination value} =
      .ok (.Ok event,after)) :
    ∃ registers, value_flow.write_predicate thread.registers destination value = .ok (.Ok (),registers) ∧
      event = .Control ∧ after=update_thread state actor {thread with registers,pc} := by
  have bound : ¬actor ≥ alloc.vec.Vec.len state.threads := by scalar_tac
  have run := computed
  simp only [execution_core.commit_action,bound,ite_false,alloc.vec.Vec.index_mut_slice_index,
    alloc.vec.Vec.index_mut_usize,alloc.vec.Vec.index_usize,alloc.vec.Vec.getElem?_Nat_eq,
    selected,bind_tc_ok,bind_ok] at run
  obtain ⟨pair,written,remainder⟩ := MemorySafety.Source.source_bind_success _ _ _ run
  rcases pair with ⟨result,registers⟩
  cases result with
  | Err fault => simp at remainder
  | Ok unitValue =>
    cases unitValue
    have replay := source_commit_predicate program state actor pc thread destination value registers inside selected written
    rw [replay] at computed
    obtain ⟨eventEq,afterEq⟩ := successful_pair _ _ _ _ computed
    exact ⟨registers,written,eventEq,afterEq⟩


theorem source_load_commit_components (program : CoreProgram) (state after : CoreState) (actor : Usize)
    (pc : U32) (thread : execution_core.ThreadState) (destination : U16) (destinationBits bytes : U8)
    (signed : Bool) (pointer : value_flow.Pointer) (event : memory_safety.Event)
    (inside : actor.val < state.threads.val.length)
    (selected : state.threads.val[actor.val]? = some thread)
    (computed : execution_core.commit_action program state {actor,next_pc:=pc, action:=.Load destination destinationBits bytes signed pointer} = .ok (.Ok event,after)) :
    ∃ access value mask registers,
      execution_core.resolve_access program state pointer bytes false = .ok (.Ok access) ∧
      execution_core.load_value state access signed = .ok value ∧
      value_flow.width_mask destinationBits = .ok (.Ok mask) ∧
      value_flow.write_value thread.registers destination (.Scalar (value &&& mask))
        (alloc.vec.Vec.deref program.layouts) = .ok (.Ok (),registers) ∧
      event = .Access actor access.request ∧ after=update_thread state actor {thread with registers,pc} := by
  have bound : ¬actor ≥ alloc.vec.Vec.len state.threads := by scalar_tac
  have run := computed
  simp only [execution_core.commit_action,bound,ite_false] at run
  obtain ⟨resolved,resolvedRun,run⟩ := MemorySafety.Source.source_bind_success _ _ _ run
  cases resolved with
  | Err error => simp at run
  | Ok access =>
    obtain ⟨value,loaded,run⟩ := MemorySafety.Source.source_bind_success _ _ _ run
    obtain ⟨masked,maskRun,run⟩ := MemorySafety.Source.source_bind_success _ _ _ run
    cases masked with
    | Err error => simp at run
    | Ok mask =>
      simp only [alloc.vec.Vec.index_mut_slice_index,alloc.vec.Vec.index_mut_usize,
        alloc.vec.Vec.index_usize,alloc.vec.Vec.getElem?_Nat_eq,selected,bind_tc_ok,bind_ok,lift] at run
      obtain ⟨pair,written,remainder⟩ := MemorySafety.Source.source_bind_success _ _ _ run
      rcases pair with ⟨result,registers⟩
      cases result with
      | Err fault => simp at remainder
      | Ok unitValue =>
        cases unitValue
        have replay := source_commit_load program state actor pc thread destination destinationBits bytes signed pointer
          access value mask registers inside selected resolvedRun loaded maskRun written
        rw [replay] at computed
        obtain ⟨eventEq,afterEq⟩ := successful_pair _ _ _ _ computed
        exact ⟨access,value,mask,registers,resolvedRun,loaded,maskRun,written,eventEq,afterEq⟩

theorem source_store_threads (state written : CoreState) (access : execution_core.ResolvedAccess) (value : U64)
    (computed : execution_core.store_value state access value=.ok written) : written.threads=state.threads := by
  unfold execution_core.store_value at computed
  obtain ⟨pair,_,computed⟩ := MemorySafety.Source.source_bind_success _ _ _ computed
  rcases pair with ⟨memory,back⟩
  obtain ⟨updated,_,computed⟩ := MemorySafety.Source.source_bind_success _ _ _ computed
  have same : written={state with allocations:=back updated} := by simpa only [ok.injEq] using computed.symm
  rw [same]

theorem source_store_commit_components (program : CoreProgram) (state after : CoreState) (actor : Usize)
    (pc : U32) (thread : execution_core.ThreadState) (bytes : U8)
    (pointer : value_flow.Pointer) (value : U64) (event : memory_safety.Event)
    (inside : actor.val < state.threads.val.length)
    (selected : state.threads.val[actor.val]? = some thread)
    (computed : execution_core.commit_action program state {actor,next_pc:=pc, action:=.Store bytes pointer value} = .ok (.Ok event,after)) :
    ∃ access written, execution_core.resolve_access program state pointer bytes true = .ok (.Ok access) ∧
      execution_core.store_value state access value = .ok written ∧
      event = .Access actor access.request ∧ after=update_thread written actor {thread with pc} := by
  have bound : ¬actor ≥ alloc.vec.Vec.len state.threads := by scalar_tac
  have run := computed
  simp only [execution_core.commit_action,bound,ite_false] at run
  obtain ⟨resolved,resolvedRun,run⟩ := MemorySafety.Source.source_bind_success _ _ _ run
  cases resolved with
  | Err error => simp at run
  | Ok access =>
    obtain ⟨written,stored,run⟩ := MemorySafety.Source.source_bind_success _ _ _ run
    have threads := source_store_threads state written access value stored
    have replay := source_commit_store program state written actor pc thread bytes pointer access value inside selected
      resolvedRun stored threads
    rw [replay] at computed
    obtain ⟨eventEq,afterEq⟩ := successful_pair _ _ _ _ computed
    exact ⟨access,written,resolvedRun,stored,eventEq,afterEq⟩


theorem source_jump_commit_components (program : CoreProgram) (state after : CoreState) (actor : Usize)
    (pc target : U32) (thread : execution_core.ThreadState) (event : memory_safety.Event)
    (inside : actor.val < state.threads.val.length)
    (selected : state.threads.val[actor.val]?=some thread)
    (computed : execution_core.commit_action program state {actor,next_pc:=pc,action:=.Jump target}=
      .ok (.Ok event,after)) :
    ∃ index, lift (UScalar.cast .Usize target)=.ok index ∧
      ¬index ≥ alloc.vec.Vec.len program.code ∧ event=.Control ∧
      after=update_thread state actor {thread with pc:=target} := by
  have bound : ¬actor ≥ alloc.vec.Vec.len state.threads := by scalar_tac
  have run:=computed
  simp only [execution_core.commit_action,bound,ite_false] at run
  obtain ⟨index,cast,run⟩ := MemorySafety.Source.source_bind_success _ _ _ run
  by_cases outside : index ≥ alloc.vec.Vec.len program.code
  · simp only [outside,ite_true,ok.injEq,Prod.mk.injEq,reduceCtorEq,false_and] at run
  · have replay := source_commit_jump program state actor pc target index thread inside selected cast outside
    rw [replay] at computed
    obtain ⟨eventEq,afterEq⟩ := successful_pair _ _ _ _ computed
    exact ⟨index,cast,outside,eventEq,afterEq⟩

end MemorySafetyExecution.Native

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.CommitEffects

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

def update_thread (state : CoreState) (actor : Usize) (thread : execution_core.ThreadState) : CoreState :=
  { state with threads := state.threads.set actor thread }

private theorem vector_set_set {α : Type} (values : alloc.vec.Vec α) (index : Usize) (first last : α) :
    (values.set index first).set index last = values.set index last := by
  apply alloc.vec.Vec.ext
  simp only [alloc.vec.Vec.set_val_eq, List.set_set]

theorem source_commit_next (program : CoreProgram) (state : CoreState) (actor : Usize)
    (pc : U32) (thread : execution_core.ThreadState)
    (inside : actor.val < state.threads.val.length)
    (selected : state.threads.val[actor.val]? = some thread) :
    execution_core.commit_action program state {actor,next_pc:=pc,action:=.Next} =
      .ok (.Ok .Control, update_thread state actor {thread with pc}) := by
  have bound : ¬actor ≥ alloc.vec.Vec.len state.threads := by scalar_tac
  simp [execution_core.commit_action, bound, alloc.vec.Vec.index_mut_slice_index,
    alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq,
    selected, bind_ok, bind_tc_ok, update_thread]

theorem source_commit_halt (program : CoreProgram) (state : CoreState) (actor : Usize)
    (pc : U32) (thread : execution_core.ThreadState)
    (inside : actor.val < state.threads.val.length)
    (selected : state.threads.val[actor.val]? = some thread) :
    execution_core.commit_action program state {actor,next_pc:=pc,action:=.Halt} =
      .ok (.Ok .Control, update_thread state actor {thread with running:=false}) := by
  have bound : ¬actor ≥ alloc.vec.Vec.len state.threads := by scalar_tac
  simp [execution_core.commit_action, bound, alloc.vec.Vec.index_mut_slice_index,
    alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq,
    selected, bind_ok, bind_tc_ok, update_thread]

theorem source_commit_virtual (program : CoreProgram) (state : CoreState) (actor : Usize)
    (pc : U32) (thread : execution_core.ThreadState) (destination : U16)
    (value : value_flow.Value) (registers : value_flow.Registers)
    (inside : actor.val < state.threads.val.length)
    (selected : state.threads.val[actor.val]? = some thread)
    (written : value_flow.write_value thread.registers destination value (alloc.vec.Vec.deref program.layouts) =
      .ok (.Ok (),registers)) :
    execution_core.commit_action program state {actor,next_pc:=pc,action:=.WriteVirtual destination value} =
      .ok (.Ok .Control, update_thread state actor {thread with registers,pc}) := by
  have bound : ¬actor ≥ alloc.vec.Vec.len state.threads := by scalar_tac
  simp [execution_core.commit_action, bound, alloc.vec.Vec.index_mut_slice_index,
    alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq,
    selected, bind_ok, bind_tc_ok, written, alloc.vec.Vec.set_val_eq,
    List.getElem?_set_self inside, vector_set_set, update_thread]

theorem source_commit_predicate (program : CoreProgram) (state : CoreState) (actor : Usize)
    (pc : U32) (thread : execution_core.ThreadState) (destination : U16)
    (value : Bool) (registers : value_flow.Registers)
    (inside : actor.val < state.threads.val.length)
    (selected : state.threads.val[actor.val]? = some thread)
    (written : value_flow.write_predicate thread.registers destination value = .ok (.Ok (),registers)) :
    execution_core.commit_action program state {actor,next_pc:=pc,action:=.WritePredicate destination value} =
      .ok (.Ok .Control, update_thread state actor {thread with registers,pc}) := by
  have bound : ¬actor ≥ alloc.vec.Vec.len state.threads := by scalar_tac
  simp [execution_core.commit_action, bound, alloc.vec.Vec.index_mut_slice_index,
    alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq,
    selected, bind_ok, bind_tc_ok, written, alloc.vec.Vec.set_val_eq,
    List.getElem?_set_self inside, vector_set_set, update_thread]


theorem source_commit_load (program : CoreProgram) (state : CoreState) (actor : Usize)
    (pc : U32) (thread : execution_core.ThreadState) (destination : U16) (destinationBits bytes : U8)
    (signed : Bool) (pointer : value_flow.Pointer) (access : execution_core.ResolvedAccess)
    (value mask : U64) (registers : value_flow.Registers)
    (inside : actor.val < state.threads.val.length)
    (selected : state.threads.val[actor.val]? = some thread)
    (resolved : execution_core.resolve_access program state pointer bytes false = .ok (.Ok access))
    (loaded : execution_core.load_value state access signed = .ok value)
    (widthMask : value_flow.width_mask destinationBits = .ok (.Ok mask))
    (written : value_flow.write_value thread.registers destination (.Scalar (value &&& mask))
      (alloc.vec.Vec.deref program.layouts) = .ok (.Ok (),registers)) :
    execution_core.commit_action program state {actor,next_pc:=pc,action:=.Load destination destinationBits bytes signed pointer} =
      .ok (.Ok (.Access actor access.request), update_thread state actor {thread with registers,pc}) := by
  have bound : ¬actor ≥ alloc.vec.Vec.len state.threads := by scalar_tac
  simp [execution_core.commit_action, bound, resolved, loaded, widthMask,
    alloc.vec.Vec.index_mut_slice_index, alloc.vec.Vec.index_mut_usize,
    alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq, selected, bind_ok, bind_tc_ok,
    lift, written, alloc.vec.Vec.set_val_eq, List.getElem?_set_self inside, vector_set_set, update_thread]

theorem source_commit_store (program : CoreProgram) (state written : CoreState) (actor : Usize)
    (pc : U32) (thread : execution_core.ThreadState) (bytes : U8)
    (pointer : value_flow.Pointer) (access : execution_core.ResolvedAccess) (value : U64)
    (inside : actor.val < state.threads.val.length)
    (selected : state.threads.val[actor.val]? = some thread)
    (resolved : execution_core.resolve_access program state pointer bytes true = .ok (.Ok access))
    (stored : execution_core.store_value state access value = .ok written)
    (frame : written.threads = state.threads) :
    execution_core.commit_action program state {actor,next_pc:=pc,action:=.Store bytes pointer value} =
      .ok (.Ok (.Access actor access.request), update_thread written actor {thread with pc}) := by
  have bound : ¬actor ≥ alloc.vec.Vec.len state.threads := by scalar_tac
  simp [execution_core.commit_action, bound, resolved, stored, frame,
    alloc.vec.Vec.index_mut_slice_index, alloc.vec.Vec.index_mut_usize,
    alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq, selected, bind_ok, bind_tc_ok, update_thread]


theorem source_commit_jump (program : CoreProgram) (state : CoreState) (actor : Usize)
    (pc target : U32) (index : Usize) (thread : execution_core.ThreadState)
    (inside : actor.val < state.threads.val.length)
    (selected : state.threads.val[actor.val]?=some thread)
    (cast : lift (UScalar.cast .Usize target)=.ok index)
    (available : ¬index ≥ alloc.vec.Vec.len program.code) :
    execution_core.commit_action program state {actor,next_pc:=pc,action:=.Jump target}=
      .ok (.Ok .Control,update_thread state actor {thread with pc:=target}) := by
  have bound : ¬actor ≥ alloc.vec.Vec.len state.threads := by scalar_tac
  have targetInside : index.val < program.code.val.length := by scalar_tac
  simp [execution_core.commit_action,bound,cast,targetInside,
    alloc.vec.Vec.index_mut_slice_index,alloc.vec.Vec.index_mut_usize,
    alloc.vec.Vec.index_usize,alloc.vec.Vec.getElem?_Nat_eq,selected,bind_ok,bind_tc_ok,update_thread]

end MemorySafetyExecution.Native

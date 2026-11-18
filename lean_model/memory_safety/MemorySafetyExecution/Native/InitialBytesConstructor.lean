/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.ProgramInitial
import FrontendProgram.SourceLoop

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source MemorySafetyExecution.Source

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

private theorem thread_check_inputs (layouts : alloc.vec.Vec value_flow.AllocationLayout)
    (left right : CoreState) (threads : left.threads=right.threads) (valid : Bool) (index : Usize)
    (output : alloc.vec.Vec execution_core.ThreadState × alloc.vec.Vec execution_core.AllocationBytes × Bool)
    (computed : execution_core.initial_state_loop0 layouts left valid index=.ok output) :
    execution_core.initial_state_loop0 layouts right valid index=
      .ok (right.threads,right.allocations,output.2.2) := by
  unfold execution_core.initial_state_loop0 at computed ⊢
  apply PtxFrontendProgram.source_loop_map _ _ (fun current=>left.threads.val.length-current.2.val)
    (fun result=>(right.threads,right.allocations,result.2.2)) ?_ (valid,index) output computed
  rintro ⟨valid,index⟩ flow computed
  unfold execution_core.initial_state_loop0.body at computed ⊢
  rw [←threads]
  by_cases inside : index<alloc.vec.Vec.len left.threads
  · simp only [inside,ite_true] at computed ⊢
    cases valid with
    | false =>
      simp only [Bool.false_eq_true,ite_false,ok.injEq] at computed ⊢
      subst flow
      exact ⟨rfl,by intro after impossible;cases impossible⟩
    | true =>
      simp only [ite_true] at computed ⊢
      obtain ⟨thread,read,computed⟩:=source_bind_success _ _ _ computed
      obtain ⟨nextValid,validated,computed⟩:=source_bind_success _ _ _ computed
      obtain ⟨nextIndex,increment,computed⟩:=source_bind_success _ _ _ computed
      have nextVal : nextIndex.val=index.val+1 := by
        have room:=alloc.vec.Vec.len_ineq left.threads
        have law:=Usize.add_spec (x:=index) (y:=1#usize) (by scalar_tac)
        rw [increment] at law
        simpa only [WP.spec_ok,UScalar.ofNatCore_val_eq] using law
      simp only [ok.injEq] at computed
      subst flow
      refine ⟨?_,?_⟩
      · rw [read]
        simp only [bind_ok]
        rw [validated]
        simp only [bind_ok]
        rw [increment]
        simp only [bind_ok]
      · intro after same
        cases same
        dsimp only
        scalar_tac
  · simp only [inside,ite_false,ok.injEq] at computed ⊢
    subst flow
    exact ⟨rfl,by intro after impossible;cases impossible⟩

private theorem live_check_inputs (left right : alloc.vec.Vec execution_core.AllocationBytes)
    (lengths : left.val.length=right.val.length)
    (lives : ∀ (index : Nat) (old fresh : execution_core.AllocationBytes),left.val[index]?=some old →
      right.val[index]?=some fresh → old.live=fresh.live) (valid : Bool) (index : Usize) :
    execution_core.initial_state_loop1 left valid index=execution_core.initial_state_loop1 right valid index := by
  unfold execution_core.initial_state_loop1
  congr 1
  funext current
  rcases current with ⟨valid,index⟩
  have lengthEq : alloc.vec.Vec.len left=alloc.vec.Vec.len right := by scalar_tac
  unfold execution_core.initial_state_loop1.body
  rw [lengthEq]
  by_cases inside : index.val<left.val.length
  · have rightInside : index.val<right.val.length := by omega
    have sameLive:=lives index.val left.val[index.val] right.val[index.val]
      (List.getElem?_eq_getElem inside) (List.getElem?_eq_getElem rightInside)
    simp only [alloc.vec.Vec.index_slice_index,alloc.vec.Vec.index_usize,alloc.vec.Vec.getElem?_Nat_eq,
      List.getElem?_eq_getElem inside,List.getElem?_eq_getElem rightInside,bind_ok,sameLive]
  · have rightOutside : right.val.length ≤ index.val := by omega
    have noLeft : left.val[index.val]?=none:=List.getElem?_eq_none (by omega)
    have noRight : right.val[index.val]?=none:=List.getElem?_eq_none rightOutside
    simp only [alloc.vec.Vec.index_slice_index,alloc.vec.Vec.index_usize,alloc.vec.Vec.getElem?_Nat_eq,noLeft,noRight]

theorem source_initial_state_bytes (program : CoreProgram) (right : CoreState)
    (variant : InitialBytesVariant program.initial right)
    (computed : execution_core.initial_state program=.ok (.Ok program.initial))
    (layout : execution_core.layout_valid program right=.ok true) :
    execution_core.initial_state (withInitial program right)=.ok (.Ok right) := by
  unfold execution_core.initial_state at computed ⊢
  rw [source_layout_initial_irrelevant]
  dsimp only [withInitial]
  obtain ⟨checkedLayout,layoutComputed,computed⟩:=source_bind_success _ _ _ computed
  cases checkedLayout with
  | false => simp at computed
  | true =>
    simp only [ite_true] at computed
    obtain ⟨result,checkedThreads,computed⟩:=source_bind_success _ _ _ computed
    obtain ⟨live,checkedLive,computed⟩:=source_bind_success _ _ _ computed
    cases live with
    | false => simp at computed
    | true =>
      have rightThreads:=thread_check_inputs program.layouts program.initial right variant.threads true 0#usize result checkedThreads
      have frame:=initial_thread_check_frame program.layouts program.initial true 0#usize result checkedThreads
      have allLives : ∀ (index : Nat) (old fresh : execution_core.AllocationBytes),
          program.initial.allocations.val[index]?=some old → right.allocations.val[index]?=some fresh → old.live=fresh.live := by
        intro index old fresh oldAt freshAt
        exact (variant.allocations index old fresh oldAt freshAt).2.2
      have rightLive:=live_check_inputs program.initial.allocations right.allocations variant.allocationsLength allLives result.2.2 0#usize
      rw [frame.2] at checkedLive
      have liveComputed : execution_core.initial_state_loop1 right.allocations result.2.2 0#usize=.ok true :=
        rightLive.symm.trans checkedLive
      rw [layout]
      simp [bind_ok,rightThreads,liveComputed,source_snapshot_state]

end MemorySafetyExecution.Native

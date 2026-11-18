/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InitialBytesConstructor
import MemorySafetyExecution.Native.LayoutPlacement

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source MemorySafetyExecution.Source
set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native

theorem source_registers_valid_no_pointers (registers : value_flow.Registers)
    (leftLayouts rightLayouts : Slice value_flow.AllocationLayout)
    (noPointers : ∀ tag∈registers.pointers.val,tag=none) :
    execution_core.registers_valid registers leftLayouts=execution_core.registers_valid registers rightLayouts := by
  have loopEq : ∀ count flag index,
      execution_core.registers_valid_loop registers.raw.regs registers.pointers leftLayouts count flag index=
      execution_core.registers_valid_loop registers.raw.regs registers.pointers rightLayouts count flag index := by
    intro count flag index
    unfold execution_core.registers_valid_loop
    congr 1
    funext current
    rcases current with ⟨valid,index⟩
    unfold execution_core.registers_valid_loop.body
    simp only [alloc.vec.Vec.index_slice_index,alloc.vec.Vec.index_usize,alloc.vec.Vec.getElem?_Nat_eq]
    cases entry : registers.pointers.val[index.val]? with
    | none => simp
    | some tag =>
      have noTag := noPointers tag (List.mem_of_getElem? entry)
      subst tag
      simp only [bind_ok]
  unfold execution_core.registers_valid
  simp only [loopEq]

private theorem placement_thread_check (leftLayouts rightLayouts : alloc.vec.Vec value_flow.AllocationLayout)
    (left right : CoreState) (threads : left.threads=right.threads)
    (noPointers : ∀ thread∈left.threads.val,∀ tag∈thread.registers.pointers.val,tag=none)
    (valid : Bool) (index : Usize)
    (output : alloc.vec.Vec execution_core.ThreadState × alloc.vec.Vec execution_core.AllocationBytes × Bool)
    (computed : execution_core.initial_state_loop0 leftLayouts left valid index=.ok output) :
    execution_core.initial_state_loop0 rightLayouts right valid index=
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
      have threadAt:=vec_index_success left.threads index thread (by
        simpa only [alloc.vec.Vec.index_slice_index] using read)
      have registersEq:=source_registers_valid_no_pointers thread.registers
        (alloc.vec.Vec.deref leftLayouts) (alloc.vec.Vec.deref rightLayouts)
        (noPointers thread (List.mem_of_getElem? threadAt))
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
        rw [←registersEq,validated]
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

private theorem placement_live_check (left right : alloc.vec.Vec execution_core.AllocationBytes)
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

theorem source_initial_state_placements (program rightProgram : CoreProgram)
    (placements : ProgramPlacements program rightProgram)
    (variant : InitialBytesVariant program.initial rightProgram.initial)
    (noPointers : ∀ thread∈program.initial.threads.val,∀ tag∈thread.registers.pointers.val,tag=none)
    (computed : execution_core.initial_state program=.ok (.Ok program.initial)) :
    execution_core.initial_state rightProgram=.ok (.Ok rightProgram.initial) := by
  have capturedLayout:=(source_initial_state_exact program program.initial computed).2
  have storageShapes : StorageShapesRelated program.initial rightProgram.initial :=
    ⟨variant.allocationsLength,variant.allocations⟩
  have layout:=source_layout_valid_placements program rightProgram program.initial rightProgram.initial placements storageShapes capturedLayout
  unfold execution_core.initial_state at computed ⊢
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
      have rightThreads:=placement_thread_check program.layouts rightProgram.layouts program.initial rightProgram.initial
        variant.threads noPointers true 0#usize result checkedThreads
      have frame:=initial_thread_check_frame program.layouts program.initial true 0#usize result checkedThreads
      have allLives : ∀ (index : Nat) (old fresh : execution_core.AllocationBytes),
          program.initial.allocations.val[index]?=some old → rightProgram.initial.allocations.val[index]?=some fresh → old.live=fresh.live := by
        intro index old fresh oldAt freshAt
        exact (variant.allocations index old fresh oldAt freshAt).2.2
      have rightLive:=placement_live_check program.initial.allocations rightProgram.initial.allocations
        variant.allocationsLength allLives result.2.2 0#usize
      rw [frame.2] at checkedLive
      have liveComputed : execution_core.initial_state_loop1 rightProgram.initial.allocations result.2.2 0#usize=.ok true :=
        rightLive.symm.trans checkedLive
      rw [layout]
      simp [bind_ok,rightThreads,liveComputed,source_snapshot_state]

end MemorySafetyExecution.Native

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.SourceLoop
import MemorySafetyExecution.Native.StateSnapshot
import MemorySafetyExecution.SourceTransitions

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

theorem initial_thread_check_frame (layouts : alloc.vec.Vec value_flow.AllocationLayout)
    (state : CoreState) (valid : Bool) (index : Usize)
    (output : alloc.vec.Vec execution_core.ThreadState × alloc.vec.Vec execution_core.AllocationBytes × Bool)
    (computed : execution_core.initial_state_loop0 layouts state valid index=.ok output) :
    output.1=state.threads ∧ output.2.1=state.allocations := by
  unfold execution_core.initial_state_loop0 at computed
  apply PtxFrontendProgram.source_loop_success _ (fun current=>state.threads.val.length-current.2.val)
    (fun _=>True) (fun output=>output.1=state.threads ∧ output.2.1=state.allocations)
    ?_ (valid,index) output True.intro computed
  rintro ⟨valid,index⟩ flow _ computed
  unfold execution_core.initial_state_loop0.body at computed
  by_cases inside : index < alloc.vec.Vec.len state.threads
  · simp only [inside,ite_true] at computed
    cases valid with
    | false =>
      simp only [Bool.false_eq_true,ite_false,ok.injEq] at computed
      subst flow
      exact ⟨rfl,rfl⟩
    | true =>
      simp only [ite_true] at computed
      obtain ⟨thread,_,computed⟩:=source_bind_success _ _ _ computed
      obtain ⟨validAfter,_,computed⟩:=source_bind_success _ _ _ computed
      obtain ⟨next,increment,computed⟩:=source_bind_success _ _ _ computed
      have nextVal : next.val=index.val+1 := by
        have room:=alloc.vec.Vec.len_ineq state.threads
        have law:=Usize.add_spec (x:=index) (y:=1#usize) (by scalar_tac)
        rw [increment] at law
        simpa only [WP.spec_ok,UScalar.ofNatCore_val_eq] using law
      simp only [ok.injEq] at computed
      subst flow
      exact ⟨True.intro,by scalar_tac⟩
  · simp only [inside,ite_false,ok.injEq] at computed
    subst flow
    exact ⟨rfl,rfl⟩

theorem source_initial_state_exact (program : CoreProgram) (state : CoreState)
    (computed : execution_core.initial_state program=.ok (.Ok state)) :
    state=program.initial ∧ execution_core.layout_valid program program.initial=.ok true := by
  unfold execution_core.initial_state at computed
  obtain ⟨valid,layoutComputed,computed⟩:=source_bind_success _ _ _ computed
  cases valid with
  | false => simp at computed
  | true =>
    simp only [ite_true] at computed
    obtain ⟨checked,checkedComputed,computed⟩:=source_bind_success _ _ _ computed
    have frame:=initial_thread_check_frame program.layouts program.initial true 0#usize checked checkedComputed
    obtain ⟨valid,_,computed⟩:=source_bind_success _ _ _ computed
    cases valid with
    | false => simp at computed
    | true =>
      simp only [ite_true] at computed
      obtain ⟨copied,copiedComputed,computed⟩:=source_bind_success _ _ _ computed
      have copyEq : copied=({threads:=checked.1,allocations:=checked.2.1}:CoreState) := by
        simpa only [source_snapshot_state,ok.injEq] using copiedComputed.symm
      have stateEq : copied=state := by simpa only [ok.injEq,core.result.Result.Ok.injEq] using computed
      refine ⟨?_,layoutComputed⟩
      rw [←stateEq,copyEq,frame.1,frame.2]

 theorem source_input_initial_parts (program : CoreProgram) (state : input_independent_execution.State)
    (computed : input_independent_execution.initial program=.ok (some state)) :
    checked_execution.initial program=.ok (some state.concrete) ∧
      input_independent_execution.initial_dependencies state.concrete.execution=.ok (some state.dependencies) := by
  unfold input_independent_execution.initial at computed
  obtain ⟨concrete,initialized,computed⟩:=source_bind_success _ _ _ computed
  cases concrete with
  | none => simp at computed
  | some concrete =>
    obtain ⟨dependencies,shadowComputed,computed⟩:=source_bind_success _ _ _ computed
    cases dependencies with
    | none => simp at computed
    | some dependencies =>
      have same : ({concrete,dependencies}:input_independent_execution.State)=state := by simpa using computed
      subst state
      exact ⟨initialized,shadowComputed⟩

end MemorySafetyExecution.Native

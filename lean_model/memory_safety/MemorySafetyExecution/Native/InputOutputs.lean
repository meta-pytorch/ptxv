/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputAppendOutput

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

private theorem outputs_unroll (state : CoreState) (allocations : Slice Usize)
    (bytes : alloc.vec.Vec U8) (fault : Option Fault) (index : Usize) :
    execution_core.outputs_loop state allocations bytes fault index=(do
      let control ← execution_core.outputs_loop.body state allocations bytes fault index
      match control with
      | .cont triple => execution_core.outputs_loop state allocations triple.1 triple.2.1 triple.2.2
      | .done result => .ok result) := by
  unfold execution_core.outputs_loop
  rw [loop.eq_def]
  congr 1
  funext control
  cases control with
  | cont triple => rcases triple with ⟨bytes,fault,index⟩; rfl
  | done result => rfl

private theorem outputs_fault (state : CoreState) (allocations : Slice Usize) (bytes : alloc.vec.Vec U8)
    (fault : Fault) (index : Usize) :
    execution_core.outputs_loop state allocations bytes (some fault) index=.ok (.Err fault) := by
  rw [outputs_unroll]
  by_cases inside : index<Slice.len allocations
  · simp [execution_core.outputs_loop.body,inside,core.option.Option.is_none,Option.isNone,bind_ok]
  · simp [execution_core.outputs_loop.body,inside,bind_ok]

theorem source_outputs_loop_inputs (shadow : value_flow_dependencies.State) (left right : CoreState)
    (allocations : Slice Usize) (leftOutput rightOutput leftAfter : alloc.vec.Vec U8) (index : Usize)
    (related : StateInputsRelated shadow left right)
    (outputLength : leftOutput.val.length=rightOutput.val.length)
    (computed : execution_core.outputs_loop left allocations leftOutput none index=.ok (.Ok leftAfter)) :
    ∃ rightAfter, execution_core.outputs_loop right allocations rightOutput none index=.ok (.Ok rightAfter) ∧
      leftAfter.val.length=rightAfter.val.length := by
  generalize fuelEq : allocations.val.length-index.val=fuel
  induction fuel using Nat.strong_induction_on generalizing leftOutput rightOutput leftAfter index with
  | h fuel ih =>
    rw [outputs_unroll] at computed ⊢
    by_cases inside : index.val<allocations.val.length
    · have test : index<Slice.len allocations := by scalar_tac
      have incrementBound : index.val+1<2^UScalarTy.Usize.numBits := by have := allocations.property; scalar_tac
      let following : Usize:=Usize.ofNatCore (index.val+1) incrementBound
      have followingVal : following.val=index.val+1 := by simp [following]
      have increment:=MemorySafety.Source.source_usize_increment index incrementBound
      simp [execution_core.outputs_loop.body,test,core.option.Option.is_none,Option.isNone,Std.bind_assoc] at computed
      obtain ⟨allocation,read,computed⟩ := MemorySafety.Source.source_bind_success _ _ _ computed
      obtain ⟨pair,appended,computed⟩ := MemorySafety.Source.source_bind_success _ _ _ computed
      rcases pair with ⟨result,leftNext⟩
      cases result with
      | Err fault =>
        simp [bind_ok,increment,outputs_fault,ok.injEq,reduceCtorEq] at computed
      | Ok unitValue =>
        cases unitValue
        obtain ⟨rightNext,rightAppended,nextLength⟩ := source_append_output_inputs shadow left right allocation
          leftOutput rightOutput leftNext related outputLength appended
        simp [bind_ok,increment] at computed
        obtain ⟨rightAfter,rightComputed,afterLength⟩ := ih (allocations.val.length-following.val) (by omega)
          leftNext rightNext leftAfter following nextLength computed rfl
        refine ⟨rightAfter,?_,afterLength⟩
        simp [execution_core.outputs_loop.body,test,core.option.Option.is_none,Option.isNone,read,rightAppended,
          bind_ok,increment ]
        exact rightComputed
    · have test : ¬index<Slice.len allocations := by scalar_tac
      simp [execution_core.outputs_loop.body,test,bind_ok,ok.injEq,core.result.Result.Ok.injEq] at computed
      subst leftAfter
      exact ⟨rightOutput,by simp [execution_core.outputs_loop.body,test,bind_ok],outputLength⟩

theorem source_outputs_inputs (shadow : value_flow_dependencies.State) (left right : CoreState)
    (allocations : Slice Usize) (leftOutput : alloc.vec.Vec U8)
    (related : StateInputsRelated shadow left right)
    (computed : execution_core.outputs left allocations=.ok (.Ok leftOutput)) :
    ∃ rightOutput, execution_core.outputs right allocations=.ok (.Ok rightOutput) ∧
      leftOutput.val.length=rightOutput.val.length := by
  exact source_outputs_loop_inputs shadow left right allocations _ _ leftOutput 0#usize related rfl computed

end MemorySafetyExecution.Native

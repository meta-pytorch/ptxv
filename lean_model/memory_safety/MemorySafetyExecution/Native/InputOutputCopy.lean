/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputState
import MemorySafety.SourceVectors

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

private theorem output_copy_unroll (output values : alloc.vec.Vec U8) (index : Usize) :
    execution_core.append_output_loop1 output values index = (do
      let control ← execution_core.append_output_loop1.body values output index
      match control with
      | .cont pair => execution_core.append_output_loop1 pair.1 values pair.2
      | .done result => .ok result) := by
  unfold execution_core.append_output_loop1
  rw [loop.eq_def]
  congr 1
  funext control
  cases control with
  | cont pair => cases pair; rfl
  | done result => rfl

theorem source_output_copy_inputs (leftValues rightValues leftOutput rightOutput leftAfter : alloc.vec.Vec U8)
    (index : Usize) (valuesLength : leftValues.val.length=rightValues.val.length)
    (outputLength : leftOutput.val.length=rightOutput.val.length)
    (computed : execution_core.append_output_loop1 leftOutput leftValues index=.ok leftAfter) :
    ∃ rightAfter, execution_core.append_output_loop1 rightOutput rightValues index=.ok rightAfter ∧
      leftAfter.val.length=rightAfter.val.length := by
  generalize fuelEq : leftValues.val.length-index.val=fuel
  induction fuel using Nat.strong_induction_on generalizing leftOutput rightOutput index leftAfter with
  | h fuel ih =>
    rw [output_copy_unroll] at computed ⊢
    by_cases inside : index.val<leftValues.val.length
    · have leftTest : index<alloc.vec.Vec.len leftValues := by scalar_tac
      have rightTest : index<alloc.vec.Vec.len rightValues := by scalar_tac
      have incrementBound : index.val+1<2^UScalarTy.Usize.numBits := by
        have lengthBound:=alloc.vec.Vec.len_ineq leftValues
        scalar_tac
      let following : Usize:=Usize.ofNatCore (index.val+1) incrementBound
      have followingVal : following.val=index.val+1 := by simp [following]
      have increment := MemorySafety.Source.source_usize_increment index incrementBound
      simp only [execution_core.append_output_loop1.body,leftTest,ite_true,Std.bind_assoc] at computed
      obtain ⟨leftByte,leftRead,computed⟩ := MemorySafety.Source.source_bind_success _ _ _ computed
      obtain ⟨leftNext,leftPush,computed⟩ := MemorySafety.Source.source_bind_success _ _ _ computed
      have pushed:=MemorySafety.Source.vec_push_success leftOutput leftNext leftByte leftPush
      have leftNextLength : leftNext.val.length=leftOutput.val.length+1 := by simp only [pushed,List.length_append,List.length_singleton]
      have room : rightOutput.val.length<Usize.max := by
        have nextBound:=alloc.vec.Vec.len_ineq leftNext
        omega
      have rightInside : index.val<rightValues.val.length := by omega
      let rightByte:=rightValues.val[index.val]
      have rightRead : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice U8) rightValues index=.ok rightByte := by
        simp only [alloc.vec.Vec.index_slice_index,alloc.vec.Vec.index_usize,
          alloc.vec.Vec.getElem?_Nat_eq,List.getElem?_eq_getElem rightInside]
        rfl
      obtain ⟨rightNext,rightPush,rightPushed⟩ := MemoryWrite.spec_returns _ _ (alloc.vec.Vec.push_spec rightOutput rightByte room)
      have nextLength : leftNext.val.length=rightNext.val.length := by
        simp only [leftNextLength,rightPushed,List.length_append,List.length_singleton,outputLength]
      simp only [increment,bind_ok] at computed
      obtain ⟨rightAfter,rightComputed,afterLength⟩ := ih (leftValues.val.length-following.val) (by omega)
        leftNext rightNext leftAfter following nextLength computed rfl
      refine ⟨rightAfter,?_,afterLength⟩
      simp only [execution_core.append_output_loop1.body,rightTest,ite_true,rightRead,rightPush,
        increment,bind_ok]
      exact rightComputed
    · have leftTest : ¬index<alloc.vec.Vec.len leftValues := by scalar_tac
      have rightTest : ¬index<alloc.vec.Vec.len rightValues := by scalar_tac
      simp only [execution_core.append_output_loop1.body,leftTest,ite_false,bind_ok,ok.injEq] at computed
      subst leftAfter
      exact ⟨rightOutput,by simp only [execution_core.append_output_loop1.body,rightTest,ite_false,bind_ok],outputLength⟩

end MemorySafetyExecution.Native

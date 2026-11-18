/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Names

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.split false
set_option backward.isDefEq.respectTransparency false

namespace PtxFrontendProgram

theorem name_bytes_length (source : List U8) (token : SourceToken)
    (ordered : token.start.val≤token.end.val) (bound : token.end.val≤source.length) :
    (nameBytes source token).length=token.end.val-token.start.val := by
  simp only [nameBytes,List.length_take,List.length_drop]
  omega

theorem name_equal_points (source : List U8) (left right : SourceToken) (meaning : NameEqual source left right) :
    left.end.val-left.start.val=right.end.val-right.start.val ∧
    ∀ slot,slot<left.end.val-left.start.val → source[left.start.val+slot]?=source[right.start.val+slot]? := by
  have leftP := meaning.1
  have rightP := meaning.2.2.1
  have length := congrArg List.length meaning.2.2.2.2
  rw [name_bytes_length source left (by omega) meaning.2.1,
      name_bytes_length source right (by omega) meaning.2.2.2.1] at length
  refine ⟨length,?_⟩
  intro slot inside
  have point := congrArg (fun values : List U8 => values[slot]?) meaning.2.2.2.2
  simpa [nameBytes,List.getElem?_take,List.getElem?_drop,inside,←length,Nat.add_comm] using point

theorem source_names_loop_complete (source : Slice U8) (start finish right offset : Usize)
    (ordered : start.val≤finish.val) (upper : finish.val≤source.val.length)
    (rightBound : right.val+(finish.val-start.val)≤source.val.length)
    (lower : offset.val≤finish.val-start.val)
    (points : ∀ slot,offset.val≤slot → slot<finish.val-start.val → source.val[start.val+slot]?=source.val[right.val+slot]?) :
    strict_memory_frontend_parse.names_equal_loop source start finish right true offset=.ok true := by
  generalize fuelEq : finish.val-start.val-offset.val=fuel
  induction fuel using Nat.strong_induction_on generalizing offset with
  | h fuel ih =>
    have unroll : strict_memory_frontend_parse.names_equal_loop source start finish right true offset = (do
        let outcome ← strict_memory_frontend_parse.names_equal_loop.body source start finish right true offset
        match outcome with
        | .cont pair => strict_memory_frontend_parse.names_equal_loop source start finish right pair.1 pair.2
        | .done value => .ok value) := by
      unfold strict_memory_frontend_parse.names_equal_loop
      rw [loop.eq_def]
      congr 1
      funext outcome
      cases outcome with
      | cont pair => cases pair;rfl
      | done value => rfl
    rw [unroll]
    have width := source_usize_sub finish start ordered
    let length := Usize.ofNatCore (finish.val-start.val) (by scalar_tac)
    have lengthVal : length.val=finish.val-start.val := by simp [length]
    change (finish-start : Result Usize)=.ok length at width
    simp only [strict_memory_frontend_parse.names_equal_loop.body,ite_true,width,bind_ok]
    by_cases inside : offset.val<finish.val-start.val
    · have test : offset<length := by scalar_tac
      have leftBound : start.val+offset.val<2^UScalarTy.Usize.numBits := by
        have := Slice.length_ineq source
        scalar_tac
      have secondBound : right.val+offset.val<2^UScalarTy.Usize.numBits := by
        have := Slice.length_ineq source
        scalar_tac
      let leftIndex := Usize.ofNatCore (start.val+offset.val) leftBound
      let rightIndex := Usize.ofNatCore (right.val+offset.val) secondBound
      have leftVal : leftIndex.val=start.val+offset.val := by simp [leftIndex]
      have rightVal : rightIndex.val=right.val+offset.val := by simp [rightIndex]
      have leftInside : leftIndex.val<source.val.length := by omega
      have rightInside : rightIndex.val<source.val.length := by omega
      let leftByte := source.val[leftIndex.val]
      let rightByte := source.val[rightIndex.val]
      have leftRead : Slice.index_usize source leftIndex=.ok leftByte := by
        simp [Slice.index_usize,List.getElem?_eq_getElem leftInside,leftByte]
      have rightRead : Slice.index_usize source rightIndex=.ok rightByte := by
        simp [Slice.index_usize,List.getElem?_eq_getElem rightInside,rightByte]
      have leftAdd := source_usize_add start offset leftBound
      have rightAdd := source_usize_add right offset secondBound
      have incrementBound : offset.val+1<2^UScalarTy.Usize.numBits := by
        have := Slice.length_ineq source
        scalar_tac
      let after := Usize.ofNatCore (offset.val+1) incrementBound
      have afterVal : after.val=offset.val+1 := by simp [after]
      have increment := source_usize_increment offset incrementBound
      simp only [leftIndex] at leftRead
      simp only [rightIndex] at rightRead
      simp only [test,ite_true,leftAdd,rightAdd,leftRead,rightRead,increment,bind_ok]
      have same := points offset.val (by omega) inside
      rw [←leftVal,←rightVal] at same
      simp only [List.getElem?_eq_getElem leftInside,List.getElem?_eq_getElem rightInside,Option.some.injEq] at same
      have byteEq : leftByte=rightByte := same
      have recursive := ih (finish.val-start.val-after.val) (by omega) after (by omega)
        (fun slot low high => points slot (by omega) high) rfl
      simpa [byteEq,bind_ok] using recursive
    · have test : ¬offset<length := by scalar_tac
      simp [test]

theorem source_names_equal_complete (source : Slice U8) (left right : SourceToken)
    (meaning : NameEqual source.val left right) :
    strict_memory_frontend_parse.names_equal source left right=.ok true := by
  have points := name_equal_points source.val left right meaning
  have leftP := meaning.1
  have leftU := meaning.2.1
  have rightP := meaning.2.2.1
  have rightU := meaning.2.2.2.1
  have leftPositive : ¬left.start≥left.end := by scalar_tac
  have leftBound : ¬left.end>Slice.len source := by scalar_tac
  have rightPositive : ¬right.start≥right.end := by scalar_tac
  have rightBound : ¬right.end>Slice.len source := by scalar_tac
  unfold strict_memory_frontend_parse.names_equal
  simp only [leftPositive,leftBound,rightPositive,rightBound,ite_false,
    source_usize_sub left.end left.start (by omega),source_usize_sub right.end right.start (by omega),bind_ok]
  have widthEq : Usize.ofNatCore (left.end.val-left.start.val) (by scalar_tac)=
      Usize.ofNatCore (right.end.val-right.start.val) (by scalar_tac) := by
    apply UScalar.eq_of_val_eq
    simpa using points.1
  simp only [widthEq,bne_self_eq_false,Bool.false_eq_true,ite_false]
  apply source_names_loop_complete source left.start left.end right.start 0#usize (by omega) meaning.2.1 (by
    have := meaning.2.2.2.1
    have := meaning.2.2.1
    omega) (by simp)
  intro slot _ inside
  exact points.2 slot inside

end PtxFrontendProgram

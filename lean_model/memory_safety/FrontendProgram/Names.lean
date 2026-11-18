/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Grammar

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.split false

namespace PtxFrontendProgram

theorem source_usize_sub (finish start : Usize) (ordered : start.val ≤ finish.val) :
    (finish-start : Result Usize) = .ok (Usize.ofNatCore (finish.val-start.val) (by scalar_tac)) := by
  change UScalar.sub finish start = _
  unfold UScalar.sub
  rw [ite_eq_right (by omega)]
  apply congrArg (fun value : Usize => (.ok value : Result Usize))
  apply UScalar.eq_of_val_eq
  have bound : finish.val-start.val<2^UScalarTy.Usize.numBits := by scalar_tac
  change (finish.val-start.val)%2^UScalarTy.Usize.numBits=finish.val-start.val
  exact Nat.mod_eq_of_lt bound

theorem source_names_false (source : Slice U8) (start finish right offset : Usize) :
    strict_memory_frontend_parse.names_equal_loop source start finish right false offset = .ok false := by
  unfold strict_memory_frontend_parse.names_equal_loop
  rw [loop.eq_def]
  simp [strict_memory_frontend_parse.names_equal_loop.body, bind_ok]

theorem source_names_loop (source : Slice U8) (start finish right offset : Usize)
    (ordered : start.val≤finish.val) (upper : finish.val≤source.val.length)
    (rightBound : right.val+(finish.val-start.val)≤source.val.length)
    (lower : offset.val≤finish.val-start.val)
    (computed : strict_memory_frontend_parse.names_equal_loop source start finish right true offset=.ok true) :
    ∀ slot, offset.val≤slot → slot<finish.val-start.val → source.val[start.val+slot]?=source.val[right.val+slot]? := by
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
      | cont pair => cases pair; rfl
      | done value => rfl
    rw [unroll] at computed
    have width := source_usize_sub finish start ordered
    let length := Usize.ofNatCore (finish.val-start.val) (by scalar_tac)
    have lengthVal : length.val=finish.val-start.val := by simp [length]
    change (finish-start : Result Usize)=.ok length at width
    simp only [strict_memory_frontend_parse.names_equal_loop.body, ite_true, width, bind_ok] at computed
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
        simp [Slice.index_usize, List.getElem?_eq_getElem leftInside, leftByte]
      have rightRead : Slice.index_usize source rightIndex=.ok rightByte := by
        simp [Slice.index_usize, List.getElem?_eq_getElem rightInside, rightByte]
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
      simp only [test, ite_true, leftAdd, rightAdd, leftRead, rightRead, increment, bind_ok] at computed
      by_cases same : leftByte=rightByte
      · simp only [same] at computed
        have tail := ih (finish.val-start.val-after.val) (by omega) after (by omega) computed rfl
        intro slot low high
        by_cases first : slot=offset.val
        · subst slot
          rw [←leftVal, ←rightVal]
          simp only [List.getElem?_eq_getElem leftInside, List.getElem?_eq_getElem rightInside]
          exact congrArg some same
        · exact tail slot (by omega) high
      · simp only [same, decide_false, source_names_false] at computed
        simp at computed
    · intro slot low high
      omega

theorem name_equal_from_points (source : List U8) (left right : SourceToken)
    (leftPositive : left.start.val<left.end.val) (leftBound : left.end.val≤source.length)
    (rightPositive : right.start.val<right.end.val) (rightBound : right.end.val≤source.length)
    (length : left.end.val-left.start.val=right.end.val-right.start.val)
    (points : ∀ slot, slot<left.end.val-left.start.val → source[left.start.val+slot]?=source[right.start.val+slot]?) :
    NameEqual source left right := by
  refine ⟨leftPositive,leftBound,rightPositive,rightBound,?_⟩
  apply List.ext_getElem?
  intro slot
  by_cases inside : slot<left.end.val-left.start.val
  · simpa [nameBytes,List.getElem?_take,List.getElem?_drop,inside,←length,Nat.add_comm] using points slot inside
  · simp [nameBytes,inside,←length]

theorem source_names_equal (source : Slice U8) (left right : SourceToken)
    (computed : strict_memory_frontend_parse.names_equal source left right=.ok true) :
    NameEqual source.val left right := by
  unfold strict_memory_frontend_parse.names_equal at computed
  by_cases leftEmpty : left.start≥left.end
  · simp only [leftEmpty,ite_true] at computed; simp at computed
  · simp only [leftEmpty,ite_false] at computed
    by_cases leftOutside : left.end>Slice.len source
    · simp only [leftOutside,ite_true] at computed; simp at computed
    · simp only [leftOutside,ite_false] at computed
      by_cases rightEmpty : right.start≥right.end
      · simp only [rightEmpty,ite_true] at computed; simp at computed
      · simp only [rightEmpty,ite_false] at computed
        by_cases rightOutside : right.end>Slice.len source
        · simp only [rightOutside,ite_true] at computed; simp at computed
        · simp only [rightOutside,ite_false] at computed
          have leftPositive : left.start.val<left.end.val := by scalar_tac
          have rightPositive : right.start.val<right.end.val := by scalar_tac
          have leftBound : left.end.val≤source.val.length := by scalar_tac
          have rightBound : right.end.val≤source.val.length := by scalar_tac
          rw [source_usize_sub left.end left.start (by omega),source_usize_sub right.end right.start (by omega)] at computed
          simp only [bind_ok] at computed
          split at computed
          · simp at computed
          · have length : left.end.val-left.start.val=right.end.val-right.start.val := by scalar_tac
            apply name_equal_from_points source.val left right leftPositive leftBound rightPositive rightBound length
            exact fun slot inside => source_names_loop source left.start left.end right.start 0#usize
              (by omega) leftBound (by omega) (by simp) computed slot (by simp) inside

end PtxFrontendProgram

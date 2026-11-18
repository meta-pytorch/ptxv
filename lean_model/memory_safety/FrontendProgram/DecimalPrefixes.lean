/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.NamesDecision
import FrontendProgram.Identifiers

open Aeneas Aeneas.Std Result PtxvMemorySafety
open MemorySafety.Source
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.split false
set_option backward.isDefEq.respectTransparency false

namespace PtxFrontendProgram

theorem source_decimal_prefix_complete (source : Slice U8) (shorter longer : SourceToken)
    (leftValid : Identifier source.val shorter true) (rightValid : Identifier source.val longer true)
    (extension : DecimalExtension source.val shorter longer) :
    strict_memory_frontend_parse.decimal_prefix source shorter longer=.ok true := by
  have leftPositive := leftValid.2.1
  have leftUpper := leftValid.2.2.1
  have rightPositive := rightValid.2.1
  have rightUpper := rightValid.2.2.1
  obtain ⟨byte,rest,extension,lower,upper⟩ := extension
  have lengths := congrArg List.length extension
  simp only [List.length_append,List.length_cons] at lengths
  rw [name_bytes_length source.val longer (by omega) rightUpper,
      name_bytes_length source.val shorter (by omega) leftUpper] at lengths
  have widthLess : shorter.end.val-shorter.start.val<longer.end.val-longer.start.val := by omega
  have middleBound : longer.start.val+(shorter.end.val-shorter.start.val)<2^UScalarTy.Usize.numBits := by
    have := Slice.length_ineq source
    scalar_tac
  let middle := Usize.ofNatCore (longer.start.val+(shorter.end.val-shorter.start.val)) middleBound
  have middleVal : middle.val=longer.start.val+(shorter.end.val-shorter.start.val) := by simp [middle]
  let head : SourceToken := {longer with «end»:=middle,word:=true}
  have headBytes : nameBytes source.val head=nameBytes source.val shorter := by
    have takeEq := congrArg (List.take (shorter.end.val-shorter.start.val)) extension
    have leftLength := name_bytes_length source.val shorter (by omega) leftUpper
    simp only [List.take_append,List.take_of_length_le (by omega : (nameBytes source.val shorter).length≤shorter.end.val-shorter.start.val),leftLength,Nat.sub_self,List.take_zero,List.append_nil] at takeEq
    have minimum : min (shorter.end.val-shorter.start.val) (longer.end.val-longer.start.val)=shorter.end.val-shorter.start.val := by omega
    simpa [nameBytes,head,middleVal,List.take_take,minimum] using takeEq
  have headMeaning : NameEqual source.val shorter head := by
    refine ⟨leftPositive,leftUpper,?_,?_,headBytes.symm⟩
    · change longer.start.val<middle.val
      omega
    · change middle.val≤source.val.length
      omega
  have matched := source_names_equal_complete source shorter head headMeaning
  have byteFound : source.val[middle.val]?=some byte := by
    have point := congrArg (fun values : List U8 => values[shorter.end.val-shorter.start.val]?) extension
    have leftLength := name_bytes_length source.val shorter (by omega) leftUpper
    simp only [List.getElem?_append,List.getElem?_cons_zero,leftLength,Nat.lt_irrefl,ite_false,Nat.sub_self] at point
    simpa [nameBytes,List.getElem?_take,List.getElem?_drop,widthLess,middleVal,Nat.add_comm] using point
  have read : Slice.index_usize source middle=.ok byte := by simp [Slice.index_usize,byteFound]
  have leftNot : ¬shorter.start≥shorter.end := by scalar_tac
  have leftBound : ¬shorter.end>Slice.len source := by scalar_tac
  have rightNot : ¬longer.start≥longer.end := by scalar_tac
  have rightBound : ¬longer.end>Slice.len source := by scalar_tac
  unfold strict_memory_frontend_parse.decimal_prefix
  simp only [leftNot,leftBound,rightNot,rightBound,ite_false,
    source_usize_sub shorter.end shorter.start (by omega),
    source_usize_sub longer.end longer.start (by omega),bind_ok]
  have widthTest : ¬Usize.ofNatCore (shorter.end.val-shorter.start.val) (by scalar_tac)≥
      Usize.ofNatCore (longer.end.val-longer.start.val) (by scalar_tac) := by scalar_tac
  let width := Usize.ofNatCore (shorter.end.val-shorter.start.val) (by scalar_tac)
  have widthVal : width.val=shorter.end.val-shorter.start.val := by simp [width]
  have addition := source_usize_add longer.start width (by simpa [widthVal] using middleBound)
  change (longer.start+width : Result Usize)=.ok middle at addition
  simp only [widthTest,ite_false]
  change Aeneas.Std.bind (longer.start+width) _=.ok true
  rw [addition]
  simp only [bind_ok]
  change Aeneas.Std.bind (strict_memory_frontend_parse.names_equal source shorter head) _=.ok true
  rw [matched]
  simp only [bind_ok,ite_true,read]
  have lowerTest : byte≥48#u8 := by scalar_tac
  have upperTest : (byte≤57#u8)=true := by scalar_tac
  simp [lowerTest,upperTest]

end PtxFrontendProgram

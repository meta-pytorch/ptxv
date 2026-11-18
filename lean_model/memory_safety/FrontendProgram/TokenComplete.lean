/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.NamesComplete
import FrontendProgram.RegisterTypes

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.split false

namespace PtxFrontendProgram

theorem source_equals_loop_complete (source expected : Slice U8) (base index : Usize)
    (lower : index.val ≤ expected.val.length)
    (upper : base.val + expected.val.length ≤ source.val.length)
    (points : ∀ offset, index.val ≤ offset → offset < expected.val.length →
      source.val[base.val + offset]? = expected.val[offset]?) :
    strict_memory_frontend.token_equals_loop source base expected true index = .ok true := by
  generalize fuelEq : expected.val.length-index.val=fuel
  induction fuel using Nat.strong_induction_on generalizing index with
  | h fuel ih =>
    have unroll : strict_memory_frontend.token_equals_loop source base expected true index = (do
        let outcome ← strict_memory_frontend.token_equals_loop.body source base expected true index
        match outcome with
        | .cont pair => strict_memory_frontend.token_equals_loop source base expected pair.1 pair.2
        | .done value => .ok value) := by
      unfold strict_memory_frontend.token_equals_loop
      rw [loop.eq_def]
      congr 1
      funext outcome
      cases outcome with
      | cont pair => cases pair;rfl
      | done value => rfl
    rw [unroll]
    by_cases inside : index.val < expected.val.length
    · have test : index < Slice.len expected := by scalar_tac
      have addBound : base.val + index.val < 2^UScalarTy.Usize.numBits := by
        have := Slice.length_ineq source
        scalar_tac
      let slot := Usize.ofNatCore (base.val+index.val) addBound
      have slotVal : slot.val=base.val+index.val := by simp [slot]
      have sourceInside : slot.val<source.val.length := by omega
      let left := source.val[slot.val]
      let right := expected.val[index.val]
      have leftRead : Slice.index_usize source slot=.ok left := by
        simp [Slice.index_usize,List.getElem?_eq_getElem sourceInside,left]
      have rightRead : Slice.index_usize expected index=.ok right := by
        simp [Slice.index_usize,List.getElem?_eq_getElem inside,right]
        rfl
      have incrementBound : index.val+1<2^UScalarTy.Usize.numBits := by
        have := Slice.length_ineq expected
        scalar_tac
      let after := Usize.ofNatCore (index.val+1) incrementBound
      have afterVal : after.val=index.val+1 := by simp [after]
      have added := source_usize_add base index addBound
      have increment := source_usize_increment index incrementBound
      simp only [slot] at leftRead
      simp only [strict_memory_frontend.token_equals_loop.body,test,ite_true,added,
        leftRead,rightRead,increment,bind_ok]
      have same := points index.val (by omega) inside
      rw [←slotVal] at same
      simp only [List.getElem?_eq_getElem sourceInside,List.getElem?_eq_getElem inside,
        Option.some.injEq] at same
      have byteEq : left=right := same
      have recursive := ih (expected.val.length-after.val) (by omega) after (by omega)
        (fun offset low high => points offset (by omega) high) rfl
      simpa only [byteEq,decide_true] using recursive
    · have test : ¬index<Slice.len expected := by scalar_tac
      simp [strict_memory_frontend.token_equals_loop.body,test]

theorem source_token_equals_complete (source expected : Slice U8) (token : SourceToken)
    (meaning : TokenText source.val token.start.val token.end.val expected.val) :
    strict_memory_frontend.token_equals source token expected=.ok true := by
  have ordered := meaning.1
  have upper := meaning.2.1
  have lengthEq := meaning.2.2.1
  have backwards : ¬token.start>token.end := by scalar_tac
  have outside : ¬token.end>Slice.len source := by scalar_tac
  unfold strict_memory_frontend.token_equals
  simp only [backwards,outside,ite_false,source_usize_sub token.end token.start ordered,bind_ok]
  have widthEq : Usize.ofNatCore (token.end.val-token.start.val) (by scalar_tac)=Slice.len expected := by
    apply UScalar.eq_of_val_eq
    simpa using lengthEq
  simp only [widthEq,bne_self_eq_false,Bool.false_eq_true,ite_false]
  apply source_equals_loop_complete source expected token.start 0#usize (by simp) (by omega)
  intro offset _ inside
  exact meaning.2.2.2 offset inside

theorem source_token_equals_false (source expected : Slice U8) (token : SourceToken)
    (computed : strict_memory_frontend.token_equals source token expected=.ok false) :
    ¬TokenText source.val token.start.val token.end.val expected.val := by
  intro meaning
  have positive := source_token_equals_complete source expected token meaning
  rw [computed] at positive
  simp at positive

end PtxFrontendProgram

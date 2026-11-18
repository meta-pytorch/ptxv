/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Digits

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option maxHeartbeats 12000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false
set_option backward.isDefEq.respectTransparency false
set_option backward.split false

open PtxFrontend

namespace PtxFrontendProgram

def TokenText (source : List U8) (start finish : Nat) (expected : List U8) : Prop :=
  start ≤ finish ∧ finish ≤ source.length ∧ finish - start = expected.length ∧
    ∀ offset, offset < expected.length → source[start + offset]? = expected[offset]?

theorem source_usize_add (left right : Usize)
    (bound : left.val + right.val < 2 ^ UScalarTy.Usize.numBits) :
    (left + right : Result Usize) = .ok (Usize.ofNatCore (left.val + right.val) bound) := by
  have checked : UScalar.check_bounds .Usize (left.val + right.val) = true := by
    simpa using bound
  change UScalar.tryMk .Usize (left.val + right.val) = _
  simp only [UScalar.tryMk, UScalar.tryMkOpt, checked, Result.ofOption]
  rfl

theorem source_equals_false (source expected : Slice U8) (base index : Usize) :
    strict_memory_frontend.token_equals_loop source base expected false index = .ok false := by
  unfold strict_memory_frontend.token_equals_loop
  rw [loop.eq_def]
  by_cases inside : index < Slice.len expected <;>
    simp only [strict_memory_frontend.token_equals_loop.body, inside, ite_true,
      ite_false, Bool.false_eq_true, bind_ok]

theorem source_equals_loop (source expected : Slice U8) (base index : Usize)
    (indexBound : index.val ≤ expected.val.length)
    (sourceBound : base.val + expected.val.length ≤ source.val.length)
    (computed : strict_memory_frontend.token_equals_loop source base expected true index = .ok true) :
    ∀ offset, index.val ≤ offset → offset < expected.val.length →
      source.val[base.val + offset]? = expected.val[offset]? := by
  generalize fuelEq : expected.val.length - index.val = fuel
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
      | cont pair => cases pair; rfl
      | done value => rfl
    rw [unroll] at computed
    by_cases inside : index.val < expected.val.length
    · have test : index < Slice.len expected := by scalar_tac
      have addBound : base.val + index.val < 2 ^ UScalarTy.Usize.numBits := by
        have := Slice.length_ineq source
        scalar_tac
      let slot :=  Usize.ofNatCore (base.val + index.val) addBound
      have atVal : slot.val = base.val + index.val := by simp [slot]
      have sourceInside : slot.val < source.val.length := by omega
      let left := source.val[slot.val]
      let right := expected.val[index.val]
      have leftRead : Slice.index_usize source slot = .ok left := by
        simp [Slice.index_usize, List.getElem?_eq_getElem sourceInside, left]
      have rightRead : Slice.index_usize expected index = .ok right := by
        simp [Slice.index_usize, List.getElem?_eq_getElem inside, right]
      have add := source_usize_add base index addBound
      have incrementBound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := Slice.length_ineq expected
        scalar_tac
      let after := Usize.ofNatCore (index.val + 1) incrementBound
      have afterVal : after.val = index.val + 1 := by simp [after]
      have increment := source_usize_increment index incrementBound
      simp only [slot] at leftRead
      simp only [strict_memory_frontend.token_equals_loop.body, test, ite_true,
        add, leftRead, rightRead, increment, bind_ok] at computed
      by_cases same : left = right
      · simp only [same] at computed
        have tail := ih (expected.val.length - after.val) (by omega) after (by omega) computed rfl
        intro offset lower upper
        by_cases first : offset = index.val
        · subst offset
          rw [←atVal]
          simp only [List.getElem?_eq_getElem sourceInside, List.getElem?_eq_getElem inside]
          exact congrArg some same
        · exact tail offset (by omega) upper
      · simp only [same, decide_false, source_equals_false] at computed
        simp at computed
    · intro offset lower upper
      omega

theorem source_token_equals (source expected : Slice U8) (token : strict_memory_frontend.Token)
    (computed : strict_memory_frontend.token_equals source token expected = .ok true) :
    TokenText source.val token.start.val token.end.val expected.val := by
  unfold strict_memory_frontend.token_equals at computed
  by_cases backwards : token.start > token.end
  · simp only [backwards, ite_true] at computed
    simp at computed
  · simp only [backwards, ite_false] at computed
    have ordered : token.start.val ≤ token.end.val := by scalar_tac
    by_cases outside : token.end > Slice.len source
    · simp only [outside, ite_true] at computed
      simp at computed
    · simp only [outside, ite_false] at computed
      have upper : token.end.val ≤ source.val.length := by scalar_tac
      obtain ⟨length, subtracted, computed⟩ := source_bind_success _ _ _ computed
      have subLaw := UScalar.sub_spec (x := token.end) (y := token.start) ordered
      rw [subtracted] at subLaw
      simp only [Aeneas.Std.WP.spec_ok] at subLaw
      split at computed
      · simp at computed
      · have lengthEq : length.val = expected.val.length := by scalar_tac
        refine ⟨ordered, upper, by omega, ?_⟩
        have exactBytes := source_equals_loop source expected token.start 0#usize (by simp) (by omega) computed
        intro offset bound
        exact exactBytes offset (by simp) bound

end PtxFrontendProgram

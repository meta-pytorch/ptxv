/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Tokens

open Aeneas Aeneas.Std Aeneas.Std.WP Result PtxvMemorySafety
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

def Digit (byte : U8) (radix digit : Nat) : Prop :=
  digit < radix ∧
    ((48 ≤ byte.val ∧ byte.val ≤ 57 ∧ digit = byte.val - 48) ∨
     (97 ≤ byte.val ∧ byte.val ≤ 102 ∧ digit = byte.val - 97 + 10) ∨
     (65 ≤ byte.val ∧ byte.val ≤ 70 ∧ digit = byte.val - 65 + 10))

inductive DigitSequence (source : List U8) (radix : Nat) : Nat → Nat → Nat → Nat → Prop
  | empty (position accumulator) : DigitSequence source radix position position accumulator accumulator
  | next {start finish accumulator digit result byte} :
      source[start]? = some byte → Digit byte radix digit →
      DigitSequence source radix (start + 1) finish (accumulator * radix + digit) result →
      DigitSequence source radix start finish accumulator result

def UnsignedNumeral (source : List U8) (start finish result : Nat) : Prop :=
  start < finish ∧ finish ≤ source.length ∧
    ((DigitSequence source 10 start finish 0 result ∧
      (finish = start + 1 ∨ ∃ byte, source[start]? = some byte ∧ byte.val ≠ 48)) ∨
     (∃ second, source[start]? = some 48#u8 ∧ source[start + 1]? = some second ∧
       (second.val = 120 ∨ second.val = 88) ∧ start + 2 < finish ∧
       DigitSequence source 16 (start + 2) finish 0 result))

@[step]
theorem source_digit_value (byte : U8) (radix : U64) :
    strict_memory_frontend.digit_value byte radix ⦃ result =>
      ∀ digit, result = some digit → Digit byte radix.val digit.val ⦄ := by
  unfold strict_memory_frontend.digit_value
  simp only [lift, bind_ok]
  repeat' first | step | split
  all_goals try simp_all [Digit, core.convert.num.FromU64U8.from_val_eq]
  all_goals scalar_tac


theorem source_digit_value_success (byte : U8) (radix digit : U64)
    (computed : strict_memory_frontend.digit_value byte radix = .ok (some digit)) :
    Digit byte radix.val digit.val := by
  have law := source_digit_value byte radix
  rw [computed] at law
  simp only [spec_ok] at law
  exact law digit rfl

theorem source_digits_loops_equal (source : Slice U8) (finish : Usize) (radix : U64)
    (value : Option U64) (position : Usize) :
    strict_memory_frontend.parse_digits_loop0 source finish radix value position =
      strict_memory_frontend.parse_digits_loop1 source finish radix value position := by
  unfold strict_memory_frontend.parse_digits_loop0 strict_memory_frontend.parse_digits_loop1
  congr 1

theorem source_digits_none (source : Slice U8) (finish : Usize) (radix : U64) (position : Usize) :
    strict_memory_frontend.parse_digits_loop1 source finish radix none position = .ok none := by
  unfold strict_memory_frontend.parse_digits_loop1
  rw [loop.eq_def]
  by_cases inside : position < finish <;>
    simp only [strict_memory_frontend.parse_digits_loop1.body, inside, ite_true,
      ite_false, core.option.Option.is_some, Option.isSome_none, Bool.false_eq_true, bind_ok]

theorem source_digits_loop (source : Slice U8) (finish : Usize) (radix accumulator result : U64)
    (position : Usize) (lower : position.val ≤ finish.val) (upper : finish.val ≤ source.val.length)
    (computed : strict_memory_frontend.parse_digits_loop1 source finish radix (some accumulator) position =
      .ok (some result)) :
    DigitSequence source.val radix.val position.val finish.val accumulator.val result.val := by
  generalize fuelEq : finish.val - position.val = fuel
  induction fuel using Nat.strong_induction_on generalizing position accumulator with
  | h fuel ih =>
    have unroll : strict_memory_frontend.parse_digits_loop1 source finish radix (some accumulator) position = (do
        let outcome ← strict_memory_frontend.parse_digits_loop1.body source finish radix (some accumulator) position
        match outcome with
        | .cont pair => strict_memory_frontend.parse_digits_loop1 source finish radix pair.1 pair.2
        | .done value => .ok value) := by
      unfold strict_memory_frontend.parse_digits_loop1
      rw [loop.eq_def]
      congr 1
      funext outcome
      cases outcome with
      | cont pair => cases pair; rfl
      | done value => rfl
    rw [unroll] at computed
    by_cases inside : position < finish
    · have indexBound : position.val < source.val.length := by scalar_tac
      let byte := source.val[position.val]
      have selected : source.val[position.val]? = some byte := by
        simp only [List.getElem?_eq_getElem indexBound, byte]
      have read : Slice.index_usize source position = .ok byte := by
        simp [Slice.index_usize, List.getElem?_eq_getElem indexBound, byte]
      have incrementBound : position.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := Slice.length_ineq source
        scalar_tac
      let after := Usize.ofNatCore (position.val + 1) incrementBound
      have afterVal : after.val = position.val + 1 := by simp [after]
      have increment := source_usize_increment position incrementBound
      simp only [strict_memory_frontend.parse_digits_loop1.body, inside, ite_true,
        core.option.Option.is_some, Option.isSome_some, read, bind_ok, Std.bind_assoc] at computed
      obtain ⟨digitOption, decoded, computed⟩ := source_bind_success _ _ _ computed
      cases digitOption with
      | none =>
        simp only [bind_ok, increment, source_digits_none] at computed
        have impossible := Eq.mp Result.ok.injEq computed
        cases impossible
      | some digit =>
        have digitMeaning := source_digit_value_success byte radix digit decoded
        cases multipliedEq : U64.checked_mul accumulator radix with
        | none =>
          simp only [lift, bind_ok, multipliedEq, increment, source_digits_none] at computed
          have impossible := Eq.mp Result.ok.injEq computed
          cases impossible
        | some multiplied =>
          have multiplyLaw := U64.checked_mul_bv_spec accumulator radix
          rw [multipliedEq] at multiplyLaw
          cases addedEq : U64.checked_add multiplied digit with
          | none =>
            simp only [lift, bind_ok, multipliedEq, addedEq, increment, source_digits_none] at computed
            have impossible := Eq.mp Result.ok.injEq computed
            cases impossible
          | some next =>
            have addLaw := U64.checked_add_bv_spec multiplied digit
            rw [addedEq] at addLaw
            simp only [lift, bind_ok, multipliedEq, addedEq, increment] at computed
            have increasing : position.val < finish.val := by scalar_tac
            have tail := ih (finish.val - after.val) (by omega) next after (by omega) computed rfl
            refine .next selected digitMeaning ?_
            rw [addLaw.2.1, multiplyLaw.2.1] at tail
            simpa only [afterVal] using tail
    · have same : position = finish := by scalar_tac
      subst position
      simp only [strict_memory_frontend.parse_digits_loop1.body, lt_self_iff_false,
        ite_false, bind_ok, Result.ok.injEq, Option.some.injEq] at computed
      subst result
      exact .empty finish.val accumulator.val


theorem source_parse_digits (source : Slice U8) (start finish : Usize) (radix result : U64)
    (computed : strict_memory_frontend.parse_digits source start finish radix = .ok (some result)) :
    start.val < finish.val ∧ finish.val ≤ source.val.length ∧ (radix.val = 10 ∨ radix.val = 16) ∧
      DigitSequence source.val radix.val start.val finish.val 0 result.val := by
  unfold strict_memory_frontend.parse_digits at computed
  by_cases ordered : start ≥ finish
  · simp only [ordered, ite_true] at computed
    have impossible := Eq.mp Result.ok.injEq computed
    cases impossible
  · simp only [ordered, ite_false] at computed
    have lower : start.val < finish.val := by scalar_tac
    by_cases outside : finish > Slice.len source
    · simp only [outside, ite_true] at computed
      have impossible := Eq.mp Result.ok.injEq computed
      cases impossible
    · simp only [outside, ite_false] at computed
      have upper : finish.val ≤ source.val.length := by scalar_tac
      split at computed
      · split at computed
        · have impossible := Eq.mp Result.ok.injEq computed
          cases impossible
        · have radixEq : radix.val = 16 := by scalar_tac
          rw [source_digits_loops_equal] at computed
          exact ⟨lower, upper, Or.inr radixEq,
            source_digits_loop source finish radix 0#u64 result start (by omega) upper computed⟩
      · have radixEq : radix.val = 10 := by scalar_tac
        exact ⟨lower, upper, Or.inl radixEq,
          source_digits_loop source finish radix 0#u64 result start (by omega) upper computed⟩

theorem source_parse_unsigned (source : Slice U8) (token : strict_memory_frontend.Token) (result : U64)
    (computed : strict_memory_frontend.parse_unsigned source token = .ok (some result)) :
    UnsignedNumeral source.val token.start.val token.end.val result.val := by
  unfold strict_memory_frontend.parse_unsigned at computed
  cases wordEq : token.word with
  | false =>
    simp only [wordEq, Bool.false_eq_true, ite_false] at computed
    have impossible := Eq.mp Result.ok.injEq computed
    cases impossible
  | true =>
    simp only [wordEq, ite_true] at computed
    by_cases ordered : token.start ≥ token.end
    · simp only [ordered, ite_true] at computed
      have impossible := Eq.mp Result.ok.injEq computed
      cases impossible
    · simp only [ordered, ite_false] at computed
      have lower : token.start.val < token.end.val := by scalar_tac
      by_cases outside : token.end > Slice.len source
      · simp only [outside, ite_true] at computed
        have impossible := Eq.mp Result.ok.injEq computed
        cases impossible
      · simp only [outside, ite_false] at computed
        have upper : token.end.val ≤ source.val.length := by scalar_tac
        have inside : token.start.val < source.val.length := by omega
        let byte := source.val[token.start.val]
        have selected : source.val[token.start.val]? = some byte := by
          simp only [List.getElem?_eq_getElem inside, byte]
        have read : Slice.index_usize source token.start = .ok byte := by
          simp [Slice.index_usize, List.getElem?_eq_getElem inside, byte]
        simp only [read, bind_ok] at computed
        by_cases zero : byte = 48#u8
        · have incrementBound : token.start.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
            have := Slice.length_ineq source
            scalar_tac
          let after := Usize.ofNatCore (token.start.val + 1) incrementBound
          have afterVal : after.val = token.start.val + 1 := by simp [after]
          have increment := source_usize_increment token.start incrementBound
          simp only [zero, ite_true, increment, bind_ok] at computed
          by_cases followingInside : after < token.end
          · have followingBound : after.val < source.val.length := by scalar_tac
            let following := source.val[after.val]
            have followingSelected : source.val[token.start.val + 1]? = some following := by
              simpa only [←afterVal, List.getElem?_eq_getElem followingBound] using
                (show some source.val[after.val] = some following from rfl)
            have followingRead : Slice.index_usize source after = .ok following := by
              simp [Slice.index_usize, List.getElem?_eq_getElem followingBound, following]
            simp only [after] at followingInside followingRead
            simp only [followingInside, ite_true, followingRead, bind_ok] at computed
            have advanceBound : token.start.val + 2 < 2 ^ UScalarTy.Usize.numBits := by
              have := Slice.length_ineq source
              scalar_tac
            let advance := Usize.ofNatCore (token.start.val + 2) advanceBound
            have advanceVal : advance.val = token.start.val + 2 := by simp [advance]
            have addTwo := source_usize_add_two token.start advanceBound
            by_cases smallX : following = 120#u8
            · simp only [smallX, ite_true, addTwo, bind_ok] at computed
              have digits := source_parse_digits source advance token.end 16#u64 result computed
              refine ⟨lower, upper, Or.inr ⟨following, ?_, followingSelected, Or.inl ?_, ?_, ?_⟩⟩
              · simpa only [zero] using selected
              · scalar_tac
              · have := digits.1; omega
              · simpa [advanceVal] using digits.2.2.2
            · simp only [smallX, ite_false] at computed
              by_cases bigX : following = 88#u8
              · simp only [bigX, ite_true, addTwo, bind_ok] at computed
                have digits := source_parse_digits source advance token.end 16#u64 result computed
                refine ⟨lower, upper, Or.inr ⟨following, ?_, followingSelected, Or.inr ?_, ?_, ?_⟩⟩
                · simpa only [zero] using selected
                · scalar_tac
                · have := digits.1; omega
                · simpa [advanceVal] using digits.2.2.2
              · simp only [bigX, ite_false] at computed
                have impossible := Eq.mp Result.ok.injEq computed
                cases impossible
          · simp only [after] at followingInside
            simp only [followingInside, ite_false] at computed
            have digits := source_parse_digits source token.start token.end 10#u64 result computed
            have same : token.end.val = token.start.val + 1 := by scalar_tac
            exact ⟨lower, upper, Or.inl ⟨digits.2.2.2, Or.inl same⟩⟩
        · simp only [zero, ite_false] at computed
          have digits := source_parse_digits source token.start token.end 10#u64 result computed
          exact ⟨lower, upper, Or.inl ⟨digits.2.2.2, Or.inr ⟨byte, selected, by scalar_tac⟩⟩⟩

end PtxFrontendProgram

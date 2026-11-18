/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Blocks

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

theorem trivia_trans (source : List U8) (start middle finish : Nat)
    (headTrivia : Trivia source start middle) (suffix : Trivia source middle finish) :
    Trivia source start finish := by
  induction headTrivia with
  | empty => exact suffix
  | space selected space next tail ih => exact .space selected space next (ih suffix)
  | line first second body tail ih => exact .line first second body (ih suffix)
  | block first second body tail ih => exact .block first second body (ih suffix)

theorem source_trivia_advance (source : Slice U8) (start finish : Usize)
    (computed : strict_memory_frontend.trivia_step source start =
      .ok (.Advance finish)) :
    start.val < finish.val ∧ finish.val ≤ source.val.length ∧
      Trivia source.val start.val finish.val := by
  unfold strict_memory_frontend.trivia_step at computed
  by_cases outside : start ≥ Slice.len source
  · simp only [outside, ite_true, ok.injEq] at computed
    cases computed
  · have inside : start.val < source.val.length := by scalar_tac
    let byte := source.val[start.val]
    have selected : source.val[start.val]? = some byte := by
      simp only [List.getElem?_eq_getElem inside, byte]
    have read : Slice.index_usize source start = .ok byte := by
      simp [Slice.index_usize, List.getElem?_eq_getElem inside, byte]
    have incrementBound : start.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
      have := Slice.length_ineq source
      scalar_tac
    let after := Usize.ofNatCore (start.val + 1) incrementBound
    have afterVal : after.val = start.val + 1 := by simp [after]
    have increment := source_usize_increment start incrementBound
    simp only [outside, ite_false, read, bind_ok, source_is_space] at computed
    by_cases space : Space byte
    · simp only [space, decide_true, ite_true, increment, bind_ok, ok.injEq,
        strict_memory_frontend.TriviaStep.Advance.injEq] at computed
      subst finish
      change start.val < after.val ∧ after.val ≤ source.val.length ∧ _
      refine ⟨by omega, by omega, ?_⟩
      exact .space selected space (by omega) (.empty after.val)
    · simp only [space, decide_false] at computed
      by_cases slash : byte = 47#u8
      · simp only [slash, ite_true, increment, bind_ok] at computed
        by_cases secondInside : after < Slice.len source
        · let second := source.val[after.val]
          have secondBound : after.val < source.val.length := by scalar_tac
          have secondRead : Slice.index_usize source after = .ok second := by
            simp [Slice.index_usize, List.getElem?_eq_getElem secondBound, second]
          have secondSelected : source.val[start.val + 1]? = some second := by
            simpa only [←afterVal, List.getElem?_eq_getElem secondBound] using
              (show some source.val[after.val] = some second from rfl)
          simp only [after] at secondInside secondRead
          simp only [secondInside, ite_true, secondRead, bind_ok] at computed
          have advanceBound : start.val + 2 < 2 ^ UScalarTy.Usize.numBits := by
            have := Slice.length_ineq source
            scalar_tac
          let advance := Usize.ofNatCore (start.val + 2) advanceBound
          have advanceVal : advance.val = start.val + 2 := by simp [advance]
          have addTwo := source_usize_add_two start advanceBound
          by_cases line : second = 47#u8
          · simp only [line, ite_true, addTwo, bind_ok] at computed
            obtain ⟨result, scanned, computed⟩ := source_bind_success _ _ _ computed
            simp only [ok.injEq, strict_memory_frontend.TriviaStep.Advance.injEq] at computed
            subst result
            have body := source_scan_line source advance finish (by omega) scanned
            have lower := body.1.1
            have upper := body.1.2.1
            refine ⟨by omega, upper, ?_⟩
            exact .line (by simpa only [slash] using selected)
              (by simpa only [line] using secondSelected)
              (by simpa only [advanceVal] using body) (.empty finish.val)
          · simp only [line, ite_false] at computed
            by_cases block : second = 42#u8
            · simp only [block, ite_true, addTwo, bind_ok] at computed
              obtain ⟨result, scanned, computed⟩ := source_bind_success _ _ _ computed
              cases result with
              | none =>
                change (ok .Invalid : Result strict_memory_frontend.TriviaStep) = ok (.Advance finish) at computed
                have impossible := Eq.mp Result.ok.injEq computed
                cases impossible
              | some position =>
                simp only [ok.injEq, strict_memory_frontend.TriviaStep.Advance.injEq] at computed
                subst position
                have body := source_scan_block source advance finish (by omega) scanned
                have lower := body.1
                have upper := body.2.1
                refine ⟨by omega, upper, ?_⟩
                exact .block (by simpa only [slash] using selected)
                  (by simpa only [block] using secondSelected)
                  (by simpa only [advanceVal] using body) (.empty finish.val)
            · simp only [block, ite_false] at computed
              have impossible := Eq.mp Result.ok.injEq computed
              cases impossible
        · simp only [after] at secondInside
          simp only [secondInside, ite_false] at computed
          have impossible := Eq.mp Result.ok.injEq computed
          cases impossible
      · simp only [slash, ite_false] at computed
        have impossible := Eq.mp Result.ok.injEq computed
        cases impossible

theorem source_skip_trivia_inactive (source : Slice U8) (position : Usize) (valid : Bool) :
    strict_memory_frontend.skip_trivia_loop source position false valid = .ok (position, valid) := by
  unfold strict_memory_frontend.skip_trivia_loop
  rw [loop.eq_def]
  simp only [strict_memory_frontend.skip_trivia_loop.body, Bool.false_eq_true,
    ite_false, bind_ok]

theorem source_skip_trivia_invalid (source : Slice U8) (position : Usize) (active : Bool) :
    strict_memory_frontend.skip_trivia_loop source position active false = .ok (position, false) := by
  unfold strict_memory_frontend.skip_trivia_loop
  rw [loop.eq_def]
  cases active <;> simp only [strict_memory_frontend.skip_trivia_loop.body,
    Bool.false_eq_true, ite_false, ite_true, bind_ok]

theorem source_skip_trivia_loop (source : Slice U8) (start finish : Usize)
    (bound : start.val ≤ source.val.length)
    (computed : strict_memory_frontend.skip_trivia_loop source start true true = .ok (finish, true)) :
    finish.val ≤ source.val.length ∧ Trivia source.val start.val finish.val := by
  generalize fuelEq : source.val.length - start.val = fuel
  induction fuel using Nat.strong_induction_on generalizing start with
  | h fuel ih =>
    have unroll : strict_memory_frontend.skip_trivia_loop source start true true = (do
        let result ← strict_memory_frontend.skip_trivia_loop.body source start true true
        match result with
        | .cont triple => strict_memory_frontend.skip_trivia_loop source triple.1 triple.2.1 triple.2.2
        | .done result => .ok result) := by
      unfold strict_memory_frontend.skip_trivia_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont triple => cases triple; rfl
      | done result => rfl
    rw [unroll] at computed
    simp only [strict_memory_frontend.skip_trivia_loop.body, ite_true, Std.bind_assoc] at computed
    obtain ⟨result, scanned, computed⟩ := source_bind_success _ _ _ computed
    cases result with
    | Advance after =>
      simp only [bind_ok] at computed
      have headTrivia := source_trivia_advance source start after scanned
      have tail := ih (source.val.length - after.val) (by omega) after headTrivia.2.1 computed rfl
      exact ⟨tail.1, trivia_trans source.val start.val after.val finish.val headTrivia.2.2 tail.2⟩
    | Done =>
      simp only [bind_ok, source_skip_trivia_inactive, ok.injEq, Prod.mk.injEq, and_true] at computed
      subst finish
      exact ⟨bound, .empty start.val⟩
    | Invalid =>
      simp only [bind_ok, source_skip_trivia_invalid, ok.injEq, Prod.mk.injEq,
        Bool.false_eq_true, and_false] at computed

theorem source_skip_trivia (source : Slice U8) (start finish : Usize)
    (bound : start.val ≤ source.val.length)
    (computed : strict_memory_frontend.skip_trivia source start = .ok (some finish)) :
    finish.val ≤ source.val.length ∧ Trivia source.val start.val finish.val := by
  unfold strict_memory_frontend.skip_trivia at computed
  obtain ⟨pair, scanned, computed⟩ := source_bind_success _ _ _ computed
  rcases pair with ⟨position, valid⟩
  cases valid with
  | false =>
    change (ok none : Result (Option Usize)) = ok (some finish) at computed
    have impossible := Eq.mp Result.ok.injEq computed
    cases impossible
  | true =>
    change (ok (some position) : Result (Option Usize)) = ok (some finish) at computed
    simp only [ok.injEq, Option.some.injEq] at computed
    subst position
    exact source_skip_trivia_loop source start finish bound scanned

end PtxFrontendProgram

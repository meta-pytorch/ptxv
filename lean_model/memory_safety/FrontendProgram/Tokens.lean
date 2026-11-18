/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Trivia

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

def tokenModel (token : strict_memory_frontend.Token) : Token :=
  ⟨token.start.val, token.end.val, token.word⟩

theorem trivia_monotone (source : List U8) (start finish : Nat)
    (trivia : Trivia source start finish) : start ≤ finish := by
  induction trivia with
  | empty => omega
  | space selected space next tail ih => omega
  | line first second body tail ih => have lower := body.1.1; omega
  | block first second body tail ih => have lower := body.1; omega

theorem token_positive (source : List U8) (token : Token)
    (meaning : TokenMeaning source token) : token.start < token.finish ∧ token.finish ≤ source.length := by
  unfold TokenMeaning at meaning
  cases wordEq : token.word with
  | true =>
    simp only [wordEq, ite_true] at meaning
    exact ⟨meaning.1, meaning.2.1.2.1⟩
  | false =>
    simp only [wordEq, Bool.false_eq_true, ite_false] at meaning
    exact ⟨by have := meaning.1; omega, meaning.2.1⟩

theorem source_next_token (source : Slice U8) (start : Usize)
    (token : strict_memory_frontend.Token) (bound : start.val ≤ source.val.length)
    (computed : strict_memory_frontend.next_token source start = .ok (.Token token)) :
    Trivia source.val start.val token.start.val ∧ TokenMeaning source.val (tokenModel token) := by
  unfold strict_memory_frontend.next_token at computed
  obtain ⟨position, skipped, computed⟩ := source_bind_success _ _ _ computed
  cases position with
  | none =>
    change (ok .Invalid : Result strict_memory_frontend.Scan) = ok (.Token token) at computed
    have impossible := Eq.mp Result.ok.injEq computed
    cases impossible
  | some position =>
    have trivia := source_skip_trivia source start position bound skipped
    by_cases atEnd : position = Slice.len source
    · simp only [atEnd, ite_true, ok.injEq] at computed
      cases computed
    · have inside : position.val < source.val.length := by
        have different : position.val ≠ source.val.length := by scalar_tac
        omega
      have notAfter : ¬position > Slice.len source := by scalar_tac
      let byte := source.val[position.val]
      have selected : source.val[position.val]? = some byte := by
        simp only [List.getElem?_eq_getElem inside, byte]
      have read : Slice.index_usize source position = .ok byte := by
        simp [Slice.index_usize, List.getElem?_eq_getElem inside, byte]
      simp only [atEnd, ite_false, notAfter, read, bind_ok, source_is_word] at computed
      by_cases word : WordByte byte
      · simp only [word, decide_true, ite_true] at computed
        obtain ⟨finish, scanned, computed⟩ := source_bind_success _ _ _ computed
        simp only [ok.injEq, strict_memory_frontend.Scan.Token.injEq] at computed
        subst token
        refine ⟨trivia.2, ?_⟩
        change WordSpan source.val position.val finish.val
        exact source_scan_word_nonempty source position finish byte selected word scanned
      · simp only [word, decide_false, Bool.false_eq_true, ite_false, source_is_punctuation] at computed
        by_cases punctuation : Punctuation byte
        · have incrementBound : position.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
            have := Slice.length_ineq source
            scalar_tac
          let after := Usize.ofNatCore (position.val + 1) incrementBound
          have afterVal : after.val = position.val + 1 := by simp [after]
          have increment := source_usize_increment position incrementBound
          simp only [punctuation, decide_true, ite_true, increment, bind_ok,
            ok.injEq, strict_memory_frontend.Scan.Token.injEq] at computed
          subst token
          refine ⟨trivia.2, ?_⟩
          change after.val = position.val + 1 ∧ after.val ≤ source.val.length ∧ _
          exact ⟨afterVal, by omega, byte, selected, punctuation⟩
        · simp only [punctuation, decide_false, bind_ok, Bool.false_eq_true, ite_false] at computed
          have impossible := Eq.mp Result.ok.injEq computed
          cases impossible

theorem source_next_end (source : Slice U8) (start : Usize)
    (bound : start.val ≤ source.val.length)
    (computed : strict_memory_frontend.next_token source start = .ok .End) :
    Trivia source.val start.val source.val.length := by
  unfold strict_memory_frontend.next_token at computed
  obtain ⟨position, skipped, computed⟩ := source_bind_success _ _ _ computed
  cases position with
  | none =>
    change (ok .Invalid : Result strict_memory_frontend.Scan) = ok .End at computed
    have impossible := Eq.mp Result.ok.injEq computed
    cases impossible
  | some position =>
    have trivia := source_skip_trivia source start position bound skipped
    by_cases atEnd : position = Slice.len source
    · have same : position.val = source.val.length := by scalar_tac
      simpa only [same] using trivia.2
    · have notAfter : ¬position > Slice.len source := by scalar_tac
      simp only [atEnd, ite_false, notAfter] at computed
      obtain ⟨byte, read, computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨word, classified, computed⟩ := source_bind_success _ _ _ computed
      cases word with
      | true =>
        obtain ⟨finish, scanned, computed⟩ := source_bind_success _ _ _ computed
        simp only [ok.injEq] at computed
        cases computed
      | false =>
        obtain ⟨punctuation, classified, computed⟩ := source_bind_success _ _ _ computed
        cases punctuation with
        | true =>
          obtain ⟨finish, incremented, computed⟩ := source_bind_success _ _ _ computed
          simp only [ok.injEq] at computed
          cases computed
        | false =>
          have impossible := Eq.mp Result.ok.injEq computed
          cases impossible

theorem source_lex_inactive (source : Slice U8) (tokens : alloc.vec.Vec strict_memory_frontend.Token)
    (position : Usize) (valid : Bool) :
    strict_memory_frontend.lex_loop source tokens position false valid = .ok (tokens, valid) := by
  unfold strict_memory_frontend.lex_loop
  rw [loop.eq_def]
  simp only [strict_memory_frontend.lex_loop.body, Bool.false_eq_true, ite_false, bind_ok]

theorem source_lex_invalid (source : Slice U8) (tokens : alloc.vec.Vec strict_memory_frontend.Token)
    (position : Usize) (active : Bool) :
    strict_memory_frontend.lex_loop source tokens position active false = .ok (tokens, false) := by
  unfold strict_memory_frontend.lex_loop
  rw [loop.eq_def]
  cases active <;> simp only [strict_memory_frontend.lex_loop.body,
    Bool.false_eq_true, ite_false, ite_true, bind_ok]

theorem source_lex_loop (source : Slice U8) (tokens output : alloc.vec.Vec strict_memory_frontend.Token)
    (position : Usize) (bound : position.val ≤ source.val.length)
    (computed : strict_memory_frontend.lex_loop source tokens position true true = .ok (output, true)) :
    ∃ tail finish, output.val.map tokenModel = tokens.val.map tokenModel ++ tail ∧
      TokenSequence source.val position.val tail finish ∧ Trivia source.val finish source.val.length := by
  generalize fuelEq : source.val.length - position.val = fuel
  induction fuel using Nat.strong_induction_on generalizing position tokens with
  | h fuel ih =>
    have unroll : strict_memory_frontend.lex_loop source tokens position true true = (do
        let result ← strict_memory_frontend.lex_loop.body source tokens position true true
        match result with
        | .cont state => strict_memory_frontend.lex_loop source state.1 state.2.1 state.2.2.1 state.2.2.2
        | .done result => .ok result) := by
      unfold strict_memory_frontend.lex_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont state => cases state; rfl
      | done result => rfl
    rw [unroll] at computed
    simp only [strict_memory_frontend.lex_loop.body, ite_true, Std.bind_assoc] at computed
    obtain ⟨result, scanned, computed⟩ := source_bind_success _ _ _ computed
    cases result with
    | Token token =>
      simp only [Std.bind_assoc] at computed
      obtain ⟨next, pushed, computed⟩ := source_bind_success _ _ _ computed
      simp only [bind_ok] at computed
      have headToken := source_next_token source position token bound scanned
      have progress := token_positive source.val (tokenModel token) headToken.2
      have earlier := trivia_monotone source.val position.val token.start.val headToken.1
      have nextBound : token.end.val ≤ source.val.length := progress.2
      obtain ⟨tail, finish, outputEq, sequence, ending⟩ :=
        ih (source.val.length - token.end.val) (by
          have increasing : token.start.val < token.end.val := progress.1
          omega) next token.end nextBound computed rfl
      refine ⟨tokenModel token :: tail, finish, ?_, .next headToken.1 headToken.2 sequence, ending⟩
      rw [vec_push_success tokens next token pushed, List.map_append, List.map_cons, List.map_nil] at outputEq
      simpa only [List.append_assoc, List.singleton_append] using outputEq
    | End =>
      simp only [bind_ok, source_lex_inactive, ok.injEq, Prod.mk.injEq, and_true] at computed
      subst output
      exact ⟨[], position.val, by simp, .empty position.val, source_next_end source position bound scanned⟩
    | Invalid =>
      simp only [bind_ok, source_lex_invalid, ok.injEq, Prod.mk.injEq,
        Bool.false_eq_true, and_false] at computed

theorem source_lex (source : Slice U8) (tokens : alloc.vec.Vec strict_memory_frontend.Token)
    (computed : strict_memory_frontend.lex source = .ok (some tokens)) :
    Lexed source.val (tokens.val.map tokenModel) := by
  unfold strict_memory_frontend.lex at computed
  obtain ⟨pair, scanned, computed⟩ := source_bind_success _ _ _ computed
  rcases pair with ⟨output, valid⟩
  cases valid with
  | false =>
    change (ok none : Result (Option (alloc.vec.Vec strict_memory_frontend.Token))) = ok (some tokens) at computed
    have impossible := Eq.mp Result.ok.injEq computed
    cases impossible
  | true =>
    change (ok (some output) : Result (Option (alloc.vec.Vec strict_memory_frontend.Token))) = ok (some tokens) at computed
    simp only [ok.injEq, Option.some.injEq] at computed
    subst output
    obtain ⟨tail, finish, outputEq, sequence, ending⟩ :=
      source_lex_loop source (alloc.vec.Vec.new _) tokens 0#usize (by simp) scanned
    have equal : tokens.val.map tokenModel = tail := by simpa [alloc.vec.Vec.new] using outputEq
    exact ⟨finish, equal.symm ▸ sequence, ending⟩

end PtxFrontendProgram

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Characters

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option maxHeartbeats 8000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false
set_option backward.isDefEq.respectTransparency false

open PtxFrontend

namespace PtxFrontendProgram

theorem span_empty (source : List U8) (index : Nat) (bound : index ≤ source.length)
    (predicate : U8 → Prop) : Span source index index predicate :=
  ⟨Nat.le_refl _, bound, by intros; omega⟩

theorem span_step (source : List U8) (index finish : Nat) (byte : U8) (predicate : U8 → Prop)
    (selected : source[index]? = some byte) (first : predicate byte)
    (tail : Span source (index + 1) finish predicate) : Span source index finish predicate := by
  rcases tail with ⟨lower, upper, remaining⟩
  refine ⟨by omega, upper, ?_⟩
  intro position after before
  by_cases same : position = index
  · exact ⟨byte, by simpa only [same] using selected, first⟩
  · exact remaining position (by omega) before

theorem source_scan_word (source : Slice U8) (start finish : Usize)
    (bound : start.val ≤ source.val.length)
    (computed : strict_memory_frontend.scan_word source start = .ok finish) :
    Span source.val start.val finish.val WordByte ∧ Stopped source.val finish.val WordByte := by
  generalize fuelEq : source.val.length - start.val = fuel
  induction fuel using Nat.strong_induction_on generalizing start with
  | h fuel ih =>
    have unroll : strict_memory_frontend.scan_word source start = (do
        let result ← strict_memory_frontend.scan_word_loop.body source start
        match result with
        | .cont after => strict_memory_frontend.scan_word source after
        | .done result => .ok result) := by
      unfold strict_memory_frontend.scan_word strict_memory_frontend.scan_word_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result <;> rfl
    rw [unroll] at computed
    by_cases inside : start.val < source.val.length
    · have test : start < Slice.len source := by scalar_tac
      let byte := source.val[start.val]
      have selected : source.val[start.val]? = some byte := by
        simp only [List.getElem?_eq_getElem inside, byte]
      have read : Slice.index_usize source start = .ok byte := by
        simp [Slice.index_usize, List.getElem?_eq_getElem inside, byte]
      simp only [strict_memory_frontend.scan_word_loop.body, test, ite_true, read,
        bind_ok, source_is_word] at computed
      by_cases word : WordByte byte
      · have incrementBound : start.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
          have := Slice.length_ineq source
          scalar_tac
        let after : Usize := Usize.ofNatCore (start.val + 1) incrementBound
        have afterVal : after.val = start.val + 1 := by simp [after]
        have increment := source_usize_increment start incrementBound
        simp only [word, decide_true, ite_true, bind_ok, increment] at computed
        obtain ⟨span, stopped⟩ := ih (source.val.length - after.val) (by omega) after
          (by omega) computed rfl
        refine ⟨?_, stopped⟩
        apply span_step source.val start.val finish.val byte WordByte selected word
        simpa only [afterVal] using span
      · simp only [word, decide_false, Bool.false_eq_true, ite_false, bind_ok, ok.injEq] at computed
        subst finish
        exact ⟨span_empty source.val start.val bound WordByte, .inr ⟨byte, selected, word⟩⟩
    · have test : ¬start < Slice.len source := by scalar_tac
      simp only [strict_memory_frontend.scan_word_loop.body, test, ite_false, bind_ok, ok.injEq] at computed
      subst finish
      exact ⟨span_empty source.val start.val bound WordByte, .inl (by omega)⟩

theorem source_scan_word_nonempty (source : Slice U8) (start finish : Usize) (byte : U8)
    (selected : source.val[start.val]? = some byte) (word : WordByte byte)
    (computed : strict_memory_frontend.scan_word source start = .ok finish) :
    WordSpan source.val start.val finish.val := by
  have inside := (List.getElem?_eq_some_iff.mp selected).1
  obtain ⟨span, stopped⟩ := source_scan_word source start finish (by omega) computed
  refine ⟨?_, span, stopped⟩
  by_contra notBefore
  have lower : start.val ≤ finish.val := span.1
  have equal : finish.val = start.val := by omega
  rcases stopped with exhausted | ⟨last, read, nonword⟩
  · omega
  · rw [equal, selected] at read
    have same : byte = last := Option.some.inj read
    subst last
    exact nonword word

theorem source_scan_line (source : Slice U8) (start finish : Usize)
    (bound : start.val ≤ source.val.length)
    (computed : strict_memory_frontend.scan_line source start = .ok finish) :
    LineBody source.val start.val finish.val := by
  generalize fuelEq : source.val.length - start.val = fuel
  induction fuel using Nat.strong_induction_on generalizing start with
  | h fuel ih =>
    have unroll : strict_memory_frontend.scan_line source start = (do
        let result ← strict_memory_frontend.scan_line_loop.body source start
        match result with
        | .cont after => strict_memory_frontend.scan_line source after
        | .done result => .ok result) := by
      unfold strict_memory_frontend.scan_line strict_memory_frontend.scan_line_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result <;> rfl
    rw [unroll] at computed
    by_cases inside : start.val < source.val.length
    · have test : start < Slice.len source := by scalar_tac
      let byte := source.val[start.val]
      have selected : source.val[start.val]? = some byte := by
        simp only [List.getElem?_eq_getElem inside, byte]
      have read : Slice.index_usize source start = .ok byte := by
        simp [Slice.index_usize, List.getElem?_eq_getElem inside, byte]
      simp only [strict_memory_frontend.scan_line_loop.body, test, ite_true, read,
        bind_ok] at computed
      by_cases line : byte ≠ 10#u8
      · have incrementBound : start.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
          have := Slice.length_ineq source
          scalar_tac
        let after : Usize := Usize.ofNatCore (start.val + 1) incrementBound
        have afterVal : after.val = start.val + 1 := by simp [after]
        have increment := source_usize_increment start incrementBound
        simp only [bne_iff_ne] at computed
        rw [ite_eq_left line] at computed
        simp only [bind_ok, increment] at computed
        obtain ⟨span, stopped⟩ := ih (source.val.length - after.val) (by omega) after
          (by omega) computed rfl
        refine ⟨?_, stopped⟩
        apply span_step source.val start.val finish.val byte (fun value => value.val ≠ 10) selected (by scalar_tac)
        simpa only [afterVal] using span
      · simp only [ne_eq, Decidable.not_not] at line
        simp only [line, bne_self_eq_false, Bool.false_eq_true, ite_false, bind_ok, ok.injEq] at computed
        subst finish
        exact ⟨span_empty source.val start.val bound (fun value => value.val ≠ 10), .inr ⟨byte, selected, by scalar_tac⟩⟩
    · have test : ¬start < Slice.len source := by scalar_tac
      simp only [strict_memory_frontend.scan_line_loop.body, test, ite_false, bind_ok, ok.injEq] at computed
      subst finish
      exact ⟨span_empty source.val start.val bound (fun value => value.val ≠ 10), .inl (by omega)⟩

end PtxFrontendProgram

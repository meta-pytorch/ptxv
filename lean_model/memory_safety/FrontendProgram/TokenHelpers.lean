/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

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

theorem source_matches (source : Slice U8) (tokens : Slice SourceToken) (position : Usize) (expected : Slice U8)
    (computed : strict_memory_frontend_parse.matches source tokens position expected=.ok true) :
    ∃ token,tokens.val[position.val]?=some token ∧ TokenText source.val token.start.val token.end.val expected.val := by
  unfold strict_memory_frontend_parse.matches at computed
  dsimp only at computed
  split at computed
  · obtain ⟨token,read,matched⟩ := source_bind_success _ _ _ computed
    exact ⟨token,source_slice_index tokens position token read,source_token_equals source expected token matched⟩
  · simp at computed

theorem source_take (source : Slice U8) (tokens : Slice SourceToken) (position after : Usize) (expected : Slice U8)
    (computed : strict_memory_frontend_parse.take source tokens position expected=.ok (some after)) :
    after.val=position.val+1 ∧ ∃ token,tokens.val[position.val]?=some token ∧
      TokenText source.val token.start.val token.end.val expected.val := by
  unfold strict_memory_frontend_parse.take at computed
  obtain ⟨matched,matchComputed,computed⟩ := source_bind_success _ _ _ computed
  cases matched with
  | false => simp at computed
  | true =>
    simp only [ite_true,ok.injEq] at computed
    have law := Usize.checked_add_bv_spec position 1#usize
    rw [computed] at law
    exact ⟨by simpa using law.2.1,source_matches source tokens position expected matchComputed⟩

theorem source_unsigned_word (source : Slice U8) (token : SourceToken) (value : U64)
    (computed : strict_memory_frontend.parse_unsigned source token=.ok (some value)) : token.word=true := by
  unfold strict_memory_frontend.parse_unsigned at computed
  cases word : token.word with
  | false => simp only [word,Bool.false_eq_true,ite_false] at computed; simp at computed
  | true => rfl

theorem source_unsigned (source : Slice U8) (tokens : Slice SourceToken) (position after : Usize) (value : U64)
    (computed : strict_memory_frontend_parse.unsigned source tokens position=.ok (some (value,after))) :
    after.val=position.val+1 ∧ NumberAt source.val tokens.val position.val value.val := by
  unfold strict_memory_frontend_parse.unsigned at computed
  dsimp only at computed
  split at computed
  · simp at computed
  · obtain ⟨token,read,computed⟩ := source_bind_success _ _ _ computed
    obtain ⟨parsed,parsedComputed,computed⟩ := source_bind_success _ _ _ computed
    cases parsed with
    | none => simp at computed
    | some raw =>
      obtain ⟨next,increment,computed⟩ := source_bind_success _ _ _ computed
      simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
      rcases computed with ⟨rfl,rfl⟩
      have law := Usize.add_spec (x:=position) (y:=1#usize) (by
        have inside := source_slice_index tokens position token read
        have := List.getElem?_eq_some_iff.mp inside
        have := Slice.length_ineq tokens
        scalar_tac)
      rw [increment] at law
      simp only [WP.spec_ok] at law
      refine ⟨by simpa using law,token,source_slice_index tokens position token read,
        source_unsigned_word source token raw parsedComputed,source_parse_unsigned source token raw parsedComputed⟩

end PtxFrontendProgram

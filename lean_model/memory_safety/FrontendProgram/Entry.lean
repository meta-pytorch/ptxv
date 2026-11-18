/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Header

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

private theorem entry_take_spelling (source : Slice U8) (tokens : Slice SourceToken) (position after : Usize) (text : String)
    (expected : Slice U8) (literal : expected.val=utf8 text)
    (computed : strict_memory_frontend_parse.take source tokens position expected=.ok (some after)) :
    after.val=position.val+1 ∧ TokenAt source.val tokens.val position.val text := by
  obtain ⟨increment,token,read,meaning⟩ := source_take source tokens position after expected computed
  refine ⟨increment,token,read,?_⟩
  simpa only [TokenSpelling,←literal] using meaning

theorem source_entry (source entry : Slice U8) (tokens : Slice SourceToken) (position finish : Usize) (selected : SourceToken)
    (computed : strict_memory_frontend_parse.parse_entry source tokens position entry=.ok (some (selected,finish))) :
    EntrySyntax source.val entry.val tokens.val position.val finish.val selected := by
  unfold strict_memory_frontend_parse.parse_entry at computed
  simp only [lift,bind_ok] at computed
  obtain ⟨visible,visibleComputed,computed⟩ := source_bind_success _ _ _ computed
  obtain ⟨beginning,beginComputed,computed⟩ := source_bind_success _ _ _ computed
  obtain ⟨namePosition,entryComputed,computed⟩ := source_bind_success _ _ _ computed
  cases namePosition with
  | none => simp at computed
  | some nameAt =>
    dsimp only at computed
    split at computed
    · simp at computed
    · obtain ⟨token,read,computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨valid,identifierComputed,computed⟩ := source_bind_success _ _ _ computed
      cases valid with
      | false => simp at computed
      | true =>
        simp only [ite_true] at computed
        obtain ⟨same,nameComputed,computed⟩ := source_bind_success _ _ _ computed
        cases same with
        | false => simp at computed
        | true =>
          simp only [ite_true] at computed
          obtain ⟨parenAt,parenAdded,computed⟩ := source_bind_success _ _ _ computed
          obtain ⟨closePosition,parenComputed,computed⟩ := source_bind_success _ _ _ computed
          cases closePosition with
          | none => simp at computed
          | some closeAt =>
            obtain ⟨bracePosition,closeComputed,computed⟩ := source_bind_success _ _ _ computed
            cases bracePosition with
            | none => simp at computed
            | some braceAt =>
              obtain ⟨afterPosition,braceComputed,computed⟩ := source_bind_success _ _ _ computed
              cases afterPosition with
              | none => simp at computed
              | some after =>
                simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
                rcases computed with ⟨rfl,rfl⟩
                have entryMeaning := entry_take_spelling source tokens beginning nameAt ".entry" _ (by decide) entryComputed
                have parenMeaning := entry_take_spelling source tokens parenAt closeAt "(" _ (by decide) parenComputed
                have closeMeaning := entry_take_spelling source tokens closeAt braceAt ")" _ (by decide) closeComputed
                have braceMeaning := entry_take_spelling source tokens braceAt after "{" _ (by decide) braceComputed
                have parenPosition := source_usize_add_value nameAt 1#usize parenAt parenAdded
                have closePosition : closeAt.val=beginning.val+3 := by scalar_tac
                have bracePosition : braceAt.val=beginning.val+4 := by scalar_tac
                have endPosition : after.val=beginning.val+5 := by scalar_tac
                have body : EntrySyntax source.val entry.val tokens.val beginning.val after.val token := by
                  rw [endPosition]
                  apply EntrySyntax.entry
                  · exact entryMeaning.2
                  · simpa [entryMeaning.1] using source_slice_index tokens nameAt token read
                  · exact source_identifier source token false identifierComputed
                  · exact source_token_equals source entry token nameComputed
                  · have indexEq : parenAt.val=beginning.val+2 := by scalar_tac
                    simpa [indexEq] using parenMeaning.2
                  · simpa [closePosition] using closeMeaning.2
                  · simpa [bracePosition] using braceMeaning.2
                cases visible with
                | false =>
                  simp only [Bool.false_eq_true,ite_false,ok.injEq] at beginComputed
                  cases beginComputed
                  exact body
                | true =>
                  simp only [ite_true] at beginComputed
                  have beginPosition := source_usize_add_value position 1#usize beginning beginComputed
                  obtain ⟨visibleToken,visibleRead,visibleMeaning⟩ := source_matches source tokens position _ visibleComputed
                  apply EntrySyntax.visible
                  · refine ⟨visibleToken,visibleRead,?_⟩
                    have literal : utf8 ".visible"=[46#u8,118#u8,105#u8,115#u8,105#u8,98#u8,108#u8,101#u8] := by decide
                    simpa [TokenSpelling,literal,Array.to_slice,Array.make] using visibleMeaning
                  · simpa [beginPosition] using body

end PtxFrontendProgram

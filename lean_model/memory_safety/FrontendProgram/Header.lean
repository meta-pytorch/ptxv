/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.TokenHelpers

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

theorem source_version_supported (source : Slice U8) (token : SourceToken)
    (computed : strict_memory_frontend_parse.version_supported source token=.ok true) :
    ∃ text∈(["8.0","8.7","9.0"] : List String),TokenSpelling source.val token text := by
  unfold strict_memory_frontend_parse.version_supported at computed
  repeat' first
    | fail_if_no_progress simp only [lift,bind_ok,ite_true,Bool.false_eq_true,ite_false] at computed
    | peel_frontend_bool computed
  all_goals
    have meaning := source_token_equals source _ token (by assumption)
    first
    | solve
      | refine ⟨"8.0",?_,?_⟩
        · simp
        · have literal : utf8 "8.0"=[56#u8, 46#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"8.7",?_,?_⟩
        · simp
        · have literal : utf8 "8.7"=[56#u8, 46#u8, 55#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"9.0",?_,?_⟩
        · simp
        · have literal : utf8 "9.0"=[57#u8, 46#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning

theorem source_target_supported (source : Slice U8) (token : SourceToken)
    (computed : strict_memory_frontend_parse.target_supported source token=.ok true) :
    ∃ text∈(["sm_80","sm_90","sm_90a","sm_100","sm_100a","sm_103a"] : List String),TokenSpelling source.val token text := by
  unfold strict_memory_frontend_parse.target_supported at computed
  repeat' first
    | fail_if_no_progress simp only [lift,bind_ok,ite_true,Bool.false_eq_true,ite_false] at computed
    | peel_frontend_bool computed
  all_goals
    have meaning := source_token_equals source _ token (by assumption)
    first
    | solve
      | refine ⟨"sm_80",?_,?_⟩
        · simp
        · have literal : utf8 "sm_80"=[115#u8, 109#u8, 95#u8, 56#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"sm_90",?_,?_⟩
        · simp
        · have literal : utf8 "sm_90"=[115#u8, 109#u8, 95#u8, 57#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"sm_90a",?_,?_⟩
        · simp
        · have literal : utf8 "sm_90a"=[115#u8, 109#u8, 95#u8, 57#u8, 48#u8, 97#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"sm_100",?_,?_⟩
        · simp
        · have literal : utf8 "sm_100"=[115#u8, 109#u8, 95#u8, 49#u8, 48#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"sm_100a",?_,?_⟩
        · simp
        · have literal : utf8 "sm_100a"=[115#u8, 109#u8, 95#u8, 49#u8, 48#u8, 48#u8, 97#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"sm_103a",?_,?_⟩
        · simp
        · have literal : utf8 "sm_103a"=[115#u8, 109#u8, 95#u8, 49#u8, 48#u8, 51#u8, 97#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning

theorem source_header_pair_supported (source : Slice U8) (version target : SourceToken)
    (computed : strict_memory_frontend_parse.header_pair_supported source version target=.ok true) :
    ∃ versionText targetText,(versionText,targetText)∈headerPairs ∧
      TokenSpelling source.val version versionText ∧ TokenSpelling source.val target targetText := by
  unfold strict_memory_frontend_parse.header_pair_supported
    strict_memory_frontend_parse.target_supported at computed
  repeat' first
    | fail_if_no_progress simp only [lift,bind_ok,ite_true,Bool.false_eq_true,ite_false] at computed
    | peel_frontend_bool computed
  all_goals try { simp only [ok.injEq, Bool.false_eq_true] at computed }
  all_goals
    have versionMeaning := source_token_equals source _ version (by assumption)
    have targetMeaning := source_token_equals source _ target (by assumption)
    first
    | solve
      | refine ⟨"8.0","sm_80",by decide,?_,?_⟩
        · have literal : utf8 "8.0"=[56#u8, 46#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using versionMeaning
        · have literal : utf8 "sm_80"=[115#u8, 109#u8, 95#u8, 56#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using targetMeaning
    | solve
      | refine ⟨"8.0","sm_90",by decide,?_,?_⟩
        · have literal : utf8 "8.0"=[56#u8, 46#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using versionMeaning
        · have literal : utf8 "sm_90"=[115#u8, 109#u8, 95#u8, 57#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using targetMeaning
    | solve
      | refine ⟨"8.0","sm_90a",by decide,?_,?_⟩
        · have literal : utf8 "8.0"=[56#u8, 46#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using versionMeaning
        · have literal : utf8 "sm_90a"=[115#u8, 109#u8, 95#u8, 57#u8, 48#u8, 97#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using targetMeaning
    | solve
      | refine ⟨"8.7","sm_80",by decide,?_,?_⟩
        · have literal : utf8 "8.7"=[56#u8, 46#u8, 55#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using versionMeaning
        · have literal : utf8 "sm_80"=[115#u8, 109#u8, 95#u8, 56#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using targetMeaning
    | solve
      | refine ⟨"8.7","sm_90",by decide,?_,?_⟩
        · have literal : utf8 "8.7"=[56#u8, 46#u8, 55#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using versionMeaning
        · have literal : utf8 "sm_90"=[115#u8, 109#u8, 95#u8, 57#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using targetMeaning
    | solve
      | refine ⟨"8.7","sm_90a",by decide,?_,?_⟩
        · have literal : utf8 "8.7"=[56#u8, 46#u8, 55#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using versionMeaning
        · have literal : utf8 "sm_90a"=[115#u8, 109#u8, 95#u8, 57#u8, 48#u8, 97#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using targetMeaning
    | solve
      | refine ⟨"8.7","sm_100",by decide,?_,?_⟩
        · have literal : utf8 "8.7"=[56#u8, 46#u8, 55#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using versionMeaning
        · have literal : utf8 "sm_100"=[115#u8, 109#u8, 95#u8, 49#u8, 48#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using targetMeaning
    | solve
      | refine ⟨"8.7","sm_100a",by decide,?_,?_⟩
        · have literal : utf8 "8.7"=[56#u8, 46#u8, 55#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using versionMeaning
        · have literal : utf8 "sm_100a"=[115#u8, 109#u8, 95#u8, 49#u8, 48#u8, 48#u8, 97#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using targetMeaning
    | solve
      | refine ⟨"9.0","sm_80",by decide,?_,?_⟩
        · have literal : utf8 "9.0"=[57#u8, 46#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using versionMeaning
        · have literal : utf8 "sm_80"=[115#u8, 109#u8, 95#u8, 56#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using targetMeaning
    | solve
      | refine ⟨"9.0","sm_90",by decide,?_,?_⟩
        · have literal : utf8 "9.0"=[57#u8, 46#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using versionMeaning
        · have literal : utf8 "sm_90"=[115#u8, 109#u8, 95#u8, 57#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using targetMeaning
    | solve
      | refine ⟨"9.0","sm_90a",by decide,?_,?_⟩
        · have literal : utf8 "9.0"=[57#u8, 46#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using versionMeaning
        · have literal : utf8 "sm_90a"=[115#u8, 109#u8, 95#u8, 57#u8, 48#u8, 97#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using targetMeaning
    | solve
      | refine ⟨"9.0","sm_100",by decide,?_,?_⟩
        · have literal : utf8 "9.0"=[57#u8, 46#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using versionMeaning
        · have literal : utf8 "sm_100"=[115#u8, 109#u8, 95#u8, 49#u8, 48#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using targetMeaning
    | solve
      | refine ⟨"9.0","sm_100a",by decide,?_,?_⟩
        · have literal : utf8 "9.0"=[57#u8, 46#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using versionMeaning
        · have literal : utf8 "sm_100a"=[115#u8, 109#u8, 95#u8, 49#u8, 48#u8, 48#u8, 97#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using targetMeaning
    | solve
      | refine ⟨"9.0","sm_103a",by decide,?_,?_⟩
        · have literal : utf8 "9.0"=[57#u8, 46#u8, 48#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using versionMeaning
        · have literal : utf8 "sm_103a"=[115#u8, 109#u8, 95#u8, 49#u8, 48#u8, 51#u8, 97#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using targetMeaning

theorem source_usize_add_value (left right next : Usize)
    (computed : (left+right : Result Usize)=.ok next) : next.val=left.val+right.val := by
  have meaning := UScalar.add_equiv left right
  rw [computed] at meaning
  exact meaning.2.1

private theorem source_take_spelling (source : Slice U8) (tokens : Slice SourceToken) (position after : Usize) (text : String)
    (expected : Slice U8) (literal : expected.val=utf8 text)
    (computed : strict_memory_frontend_parse.take source tokens position expected=.ok (some after)) :
    after.val=position.val+1 ∧ TokenAt source.val tokens.val position.val text := by
  obtain ⟨increment,token,read,meaning⟩ := source_take source tokens position after expected computed
  refine ⟨increment,token,read,?_⟩
  simpa only [TokenSpelling,←literal] using meaning

theorem source_header (source : Slice U8) (tokens : Slice SourceToken) (finish : Usize)
    (computed : strict_memory_frontend_parse.parse_header source tokens=.ok (some finish)) :
    HeaderSyntax source.val tokens.val ∧ finish.val=6 := by
  unfold strict_memory_frontend_parse.parse_header at computed
  simp only [lift,bind_ok] at computed
  obtain ⟨first,firstComputed,computed⟩ := source_bind_success _ _ _ computed
  cases first with
  | none => simp at computed
  | some versionAt =>
    dsimp only at computed
    by_cases missingVersion : versionAt≥Slice.len tokens
    · simp only [missingVersion,ite_true] at computed; simp at computed
    · simp only [missingVersion,ite_false] at computed
      obtain ⟨version,versionRead,computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨supported,versionComputed,computed⟩ := source_bind_success _ _ _ computed
      cases supported with
      | false => simp at computed
      | true =>
        simp only [ite_true] at computed
        obtain ⟨targetMarkAt,targetMarkAdded,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨targetPosition,targetMarkComputed,computed⟩ := source_bind_success _ _ _ computed
        cases targetPosition with
        | none => simp at computed
        | some targetAt =>
          dsimp only at computed
          by_cases missingTarget : targetAt≥Slice.len tokens
          · simp only [missingTarget,ite_true] at computed; simp at computed
          · simp only [missingTarget,ite_false] at computed
            obtain ⟨target,targetRead,computed⟩ := source_bind_success _ _ _ computed
            obtain ⟨supported,targetComputed,computed⟩ := source_bind_success _ _ _ computed
            cases supported with
            | false => simp at computed
            | true =>
              simp only [ite_true] at computed
              obtain ⟨addressMarkAt,addressMarkAdded,computed⟩ := source_bind_success _ _ _ computed
              obtain ⟨addressPosition,addressMarkComputed,computed⟩ := source_bind_success _ _ _ computed
              cases addressPosition with
              | none => simp at computed
              | some addressAt =>
                obtain ⟨addressPair,addressComputed,computed⟩ := source_bind_success _ _ _ computed
                cases addressPair with
                | none => simp at computed
                | some pair =>
                  rcases pair with ⟨addressSize,after⟩
                  dsimp only at computed
                  change (if (addressSize != 64#u64)=true then (ok none : Result (Option Usize)) else ok (some after))=ok (some finish) at computed
                  by_cases wrong : addressSize≠64#u64
                  · have test : (addressSize != 64#u64)=true := by scalar_tac
                    simp only [test,ite_true] at computed; simp at computed
                  · have equal : addressSize=64#u64 := by scalar_tac
                    have test : (addressSize != 64#u64)=false := by simp [equal]
                    simp only [test,Bool.false_eq_true,ite_false,ok.injEq,Option.some.injEq] at computed
                    cases computed
                    have firstMeaning := source_take_spelling source tokens 0#usize versionAt ".version" _ (by decide) firstComputed
                    have versionSelected := source_slice_index tokens versionAt version versionRead
                    have targetMarkMeaning := source_take_spelling source tokens targetMarkAt targetAt ".target" _ (by decide) targetMarkComputed
                    have pairMeaning := source_header_pair_supported source version target targetComputed
                    have targetSelected := source_slice_index tokens targetAt target targetRead
                    have addressMarkMeaning := source_take_spelling source tokens addressMarkAt addressAt ".address_size" _ (by decide) addressMarkComputed
                    have addressMeaning := source_unsigned source tokens addressAt finish addressSize addressComputed
                    have firstPosition : versionAt.val=1 := by simpa using firstMeaning.1
                    have targetMarkPosition : targetMarkAt.val=2 := by
                      have law := source_usize_add_value versionAt 1#usize targetMarkAt targetMarkAdded
                      scalar_tac
                    have targetPosition : targetAt.val=3 := by omega
                    have addressMarkPosition : addressMarkAt.val=4 := by
                      have law := source_usize_add_value targetAt 1#usize addressMarkAt addressMarkAdded
                      scalar_tac
                    have addressPosition : addressAt.val=5 := by omega
                    refine ⟨?_,by omega⟩
                    apply HeaderSyntax.header
                    · simpa using firstMeaning.2
                    · obtain ⟨versionText,targetText,listed,versionSpelling,targetSpelling⟩ := pairMeaning
                      exact ⟨versionText,targetText,listed,
                        ⟨version,by simpa [firstPosition] using versionSelected,versionSpelling⟩,
                        ⟨target,by simpa [targetPosition] using targetSelected,targetSpelling⟩⟩
                    · simpa [targetMarkPosition] using targetMarkMeaning.2
                    · simpa [addressMarkPosition] using addressMarkMeaning.2
                    · simpa [addressPosition,equal] using addressMeaning.2

end PtxFrontendProgram

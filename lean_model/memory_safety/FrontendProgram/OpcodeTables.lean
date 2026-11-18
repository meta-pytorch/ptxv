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

theorem source_binary (source : Slice U8) (token : SourceToken) (bits : U8) (operation : ptxv_memory_write.value_flow.BinaryOperation)
    (computed : strict_memory_frontend_parse.parse_binary source token=.ok (some (bits,operation))) :
    ∃ text,(text,bits,operation)∈binaryNames ∧ TokenSpelling source.val token text := by
  unfold strict_memory_frontend_parse.parse_binary at computed
  repeat' first
    | fail_if_no_progress simp only [lift,bind_ok,ite_true,Bool.false_eq_true,ite_false] at computed
    | peel_frontend_bool computed
  all_goals simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq,reduceCtorEq] at computed
  all_goals rcases computed with ⟨rfl,rfl⟩
  all_goals rename_i matched
  all_goals
    have meaning := source_token_equals source _ token matched
    first
    | solve
      | refine ⟨"add.u32",?_,?_⟩
        · simp [binaryNames]
        · have literal : utf8 "add.u32"=[97#u8, 100#u8, 100#u8, 46#u8, 117#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"add.u64",?_,?_⟩
        · simp [binaryNames]
        · have literal : utf8 "add.u64"=[97#u8, 100#u8, 100#u8, 46#u8, 117#u8, 54#u8, 52#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"sub.u32",?_,?_⟩
        · simp [binaryNames]
        · have literal : utf8 "sub.u32"=[115#u8, 117#u8, 98#u8, 46#u8, 117#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"sub.u64",?_,?_⟩
        · simp [binaryNames]
        · have literal : utf8 "sub.u64"=[115#u8, 117#u8, 98#u8, 46#u8, 117#u8, 54#u8, 52#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"and.b32",?_,?_⟩
        · simp [binaryNames]
        · have literal : utf8 "and.b32"=[97#u8, 110#u8, 100#u8, 46#u8, 98#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"and.b64",?_,?_⟩
        · simp [binaryNames]
        · have literal : utf8 "and.b64"=[97#u8, 110#u8, 100#u8, 46#u8, 98#u8, 54#u8, 52#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"or.b32",?_,?_⟩
        · simp [binaryNames]
        · have literal : utf8 "or.b32"=[111#u8, 114#u8, 46#u8, 98#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"or.b64",?_,?_⟩
        · simp [binaryNames]
        · have literal : utf8 "or.b64"=[111#u8, 114#u8, 46#u8, 98#u8, 54#u8, 52#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"xor.b32",?_,?_⟩
        · simp [binaryNames]
        · have literal : utf8 "xor.b32"=[120#u8, 111#u8, 114#u8, 46#u8, 98#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"xor.b64",?_,?_⟩
        · simp [binaryNames]
        · have literal : utf8 "xor.b64"=[120#u8, 111#u8, 114#u8, 46#u8, 98#u8, 54#u8, 52#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"mul.lo.u32",?_,?_⟩
        · simp [binaryNames]
        · have literal : utf8 "mul.lo.u32"=[109#u8, 117#u8, 108#u8, 46#u8, 108#u8, 111#u8, 46#u8, 117#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"mul.lo.u64",?_,?_⟩
        · simp [binaryNames]
        · have literal : utf8 "mul.lo.u64"=[109#u8, 117#u8, 108#u8, 46#u8, 108#u8, 111#u8, 46#u8, 117#u8, 54#u8, 52#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning

theorem source_compare (source : Slice U8) (token : SourceToken) (bits : U8) (comparison : ptxv_memory_write.value_flow.Comparison)
    (computed : strict_memory_frontend_parse.parse_compare source token=.ok (some (bits,comparison))) :
    ∃ text,(text,bits,comparison)∈comparisonNames ∧ TokenSpelling source.val token text := by
  unfold strict_memory_frontend_parse.parse_compare at computed
  repeat' first
    | fail_if_no_progress simp only [lift,bind_ok,ite_true,Bool.false_eq_true,ite_false] at computed
    | peel_frontend_bool computed
  all_goals simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq,reduceCtorEq] at computed
  all_goals rcases computed with ⟨rfl,rfl⟩
  all_goals rename_i matched
  all_goals
    have meaning := source_token_equals source _ token matched
    first
    | solve
      | refine ⟨"setp.eq.u32",?_,?_⟩
        · simp [comparisonNames]
        · have literal : utf8 "setp.eq.u32"=[115#u8, 101#u8, 116#u8, 112#u8, 46#u8, 101#u8, 113#u8, 46#u8, 117#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"setp.eq.u64",?_,?_⟩
        · simp [comparisonNames]
        · have literal : utf8 "setp.eq.u64"=[115#u8, 101#u8, 116#u8, 112#u8, 46#u8, 101#u8, 113#u8, 46#u8, 117#u8, 54#u8, 52#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"setp.ne.u32",?_,?_⟩
        · simp [comparisonNames]
        · have literal : utf8 "setp.ne.u32"=[115#u8, 101#u8, 116#u8, 112#u8, 46#u8, 110#u8, 101#u8, 46#u8, 117#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"setp.ne.u64",?_,?_⟩
        · simp [comparisonNames]
        · have literal : utf8 "setp.ne.u64"=[115#u8, 101#u8, 116#u8, 112#u8, 46#u8, 110#u8, 101#u8, 46#u8, 117#u8, 54#u8, 52#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"setp.lt.u32",?_,?_⟩
        · simp [comparisonNames]
        · have literal : utf8 "setp.lt.u32"=[115#u8, 101#u8, 116#u8, 112#u8, 46#u8, 108#u8, 116#u8, 46#u8, 117#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"setp.lt.u64",?_,?_⟩
        · simp [comparisonNames]
        · have literal : utf8 "setp.lt.u64"=[115#u8, 101#u8, 116#u8, 112#u8, 46#u8, 108#u8, 116#u8, 46#u8, 117#u8, 54#u8, 52#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"setp.le.u32",?_,?_⟩
        · simp [comparisonNames]
        · have literal : utf8 "setp.le.u32"=[115#u8, 101#u8, 116#u8, 112#u8, 46#u8, 108#u8, 101#u8, 46#u8, 117#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"setp.le.u64",?_,?_⟩
        · simp [comparisonNames]
        · have literal : utf8 "setp.le.u64"=[115#u8, 101#u8, 116#u8, 112#u8, 46#u8, 108#u8, 101#u8, 46#u8, 117#u8, 54#u8, 52#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"setp.gt.u32",?_,?_⟩
        · simp [comparisonNames]
        · have literal : utf8 "setp.gt.u32"=[115#u8, 101#u8, 116#u8, 112#u8, 46#u8, 103#u8, 116#u8, 46#u8, 117#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"setp.gt.u64",?_,?_⟩
        · simp [comparisonNames]
        · have literal : utf8 "setp.gt.u64"=[115#u8, 101#u8, 116#u8, 112#u8, 46#u8, 103#u8, 116#u8, 46#u8, 117#u8, 54#u8, 52#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"setp.ge.u32",?_,?_⟩
        · simp [comparisonNames]
        · have literal : utf8 "setp.ge.u32"=[115#u8, 101#u8, 116#u8, 112#u8, 46#u8, 103#u8, 101#u8, 46#u8, 117#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"setp.ge.u64",?_,?_⟩
        · simp [comparisonNames]
        · have literal : utf8 "setp.ge.u64"=[115#u8, 101#u8, 116#u8, 112#u8, 46#u8, 103#u8, 101#u8, 46#u8, 117#u8, 54#u8, 52#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning

theorem source_special (source : Slice U8) (token : SourceToken) (register : ptxv_memory_write.value_flow.SpecialRegister)
    (computed : strict_memory_frontend_parse.special source token=.ok (some register)) :
    ∃ text,(text,register)∈specialNames ∧ TokenSpelling source.val token text := by
  unfold strict_memory_frontend_parse.special at computed
  repeat' first
    | fail_if_no_progress simp only [lift,bind_ok,ite_true,Bool.false_eq_true,ite_false] at computed
    | peel_frontend_bool computed
  all_goals simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq,reduceCtorEq] at computed
  all_goals cases computed
  all_goals rename_i matched
  all_goals
    have meaning := source_token_equals source _ token matched
    first
    | solve
      | refine ⟨"%tid.x",?_,?_⟩
        · simp [specialNames]
        · have literal : utf8 "%tid.x"=[37#u8, 116#u8, 105#u8, 100#u8, 46#u8, 120#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"%tid.y",?_,?_⟩
        · simp [specialNames]
        · have literal : utf8 "%tid.y"=[37#u8, 116#u8, 105#u8, 100#u8, 46#u8, 121#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"%tid.z",?_,?_⟩
        · simp [specialNames]
        · have literal : utf8 "%tid.z"=[37#u8, 116#u8, 105#u8, 100#u8, 46#u8, 122#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"%ctaid.x",?_,?_⟩
        · simp [specialNames]
        · have literal : utf8 "%ctaid.x"=[37#u8, 99#u8, 116#u8, 97#u8, 105#u8, 100#u8, 46#u8, 120#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"%ctaid.y",?_,?_⟩
        · simp [specialNames]
        · have literal : utf8 "%ctaid.y"=[37#u8, 99#u8, 116#u8, 97#u8, 105#u8, 100#u8, 46#u8, 121#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"%ctaid.z",?_,?_⟩
        · simp [specialNames]
        · have literal : utf8 "%ctaid.z"=[37#u8, 99#u8, 116#u8, 97#u8, 105#u8, 100#u8, 46#u8, 122#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"%ntid.x",?_,?_⟩
        · simp [specialNames]
        · have literal : utf8 "%ntid.x"=[37#u8, 110#u8, 116#u8, 105#u8, 100#u8, 46#u8, 120#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"%ntid.y",?_,?_⟩
        · simp [specialNames]
        · have literal : utf8 "%ntid.y"=[37#u8, 110#u8, 116#u8, 105#u8, 100#u8, 46#u8, 121#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"%ntid.z",?_,?_⟩
        · simp [specialNames]
        · have literal : utf8 "%ntid.z"=[37#u8, 110#u8, 116#u8, 105#u8, 100#u8, 46#u8, 122#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"%nctaid.x",?_,?_⟩
        · simp [specialNames]
        · have literal : utf8 "%nctaid.x"=[37#u8, 110#u8, 99#u8, 116#u8, 97#u8, 105#u8, 100#u8, 46#u8, 120#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"%nctaid.y",?_,?_⟩
        · simp [specialNames]
        · have literal : utf8 "%nctaid.y"=[37#u8, 110#u8, 99#u8, 116#u8, 97#u8, 105#u8, 100#u8, 46#u8, 121#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"%nctaid.z",?_,?_⟩
        · simp [specialNames]
        · have literal : utf8 "%nctaid.z"=[37#u8, 110#u8, 99#u8, 116#u8, 97#u8, 105#u8, 100#u8, 46#u8, 122#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"%laneid",?_,?_⟩
        · simp [specialNames]
        · have literal : utf8 "%laneid"=[37#u8, 108#u8, 97#u8, 110#u8, 101#u8, 105#u8, 100#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning

theorem source_global_type (source : Slice U8) (token : SourceToken) (width : U8)
    (computed : strict_memory_frontend_parse.global_type source token=.ok (some width)) :
    GlobalElementType source.val token width.val := by
  unfold strict_memory_frontend_parse.global_type at computed
  repeat' first
    | fail_if_no_progress simp only [lift,bind_ok,ite_true,Bool.false_eq_true,ite_false] at computed
    | peel_frontend_bool computed
  all_goals simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq,reduceCtorEq] at computed
  all_goals cases computed
  all_goals rename_i matched
  all_goals
    have meaning := source_token_equals source _ token matched
    first
    | solve
      | refine ⟨".u8",?_,?_⟩
        · simp [GlobalElementType]
        · have literal : utf8 ".u8"=[46#u8, 117#u8, 56#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨".b8",?_,?_⟩
        · simp [GlobalElementType]
        · have literal : utf8 ".b8"=[46#u8, 98#u8, 56#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨".u16",?_,?_⟩
        · simp [GlobalElementType]
        · have literal : utf8 ".u16"=[46#u8, 117#u8, 49#u8, 54#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨".b16",?_,?_⟩
        · simp [GlobalElementType]
        · have literal : utf8 ".b16"=[46#u8, 98#u8, 49#u8, 54#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨".u32",?_,?_⟩
        · simp [GlobalElementType]
        · have literal : utf8 ".u32"=[46#u8, 117#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨".b32",?_,?_⟩
        · simp [GlobalElementType]
        · have literal : utf8 ".b32"=[46#u8, 98#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨".u64",?_,?_⟩
        · simp [GlobalElementType]
        · have literal : utf8 ".u64"=[46#u8, 117#u8, 54#u8, 52#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨".b64",?_,?_⟩
        · simp [GlobalElementType]
        · have literal : utf8 ".b64"=[46#u8, 98#u8, 54#u8, 52#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning

end PtxFrontendProgram

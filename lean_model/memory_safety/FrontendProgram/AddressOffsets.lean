/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.OffsetPrimitives

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

private theorem offset_maximum : IScalar.hcast .U64 core.num.I64.MAX=9223372036854775807#u64 := by decide

private theorem offset_sign (source : Slice U8) (tokens : Slice SourceToken) (position : Usize)
    (sign : Slice U8) (text : String) (literal : sign.val=utf8 text)
    (computed : strict_memory_frontend_parse.matches source tokens position sign=.ok true) :
    TokenAt source.val tokens.val position.val text := by
  obtain ⟨token,found,meaning⟩ := source_matches source tokens position sign computed
  exact ⟨token,found,by simpa only [TokenSpelling,←literal] using meaning⟩

theorem source_parse_address_offset (source : Slice U8) (tokens : Slice SourceToken) (position finish : Usize)
    (offset : I64)
    (computed : strict_memory_frontend_parse.parse_address_offset source tokens position=.ok (some (offset,finish))) :
    OffsetSyntax source.val tokens.val position.val finish.val offset.val := by
  unfold strict_memory_frontend_parse.parse_address_offset at computed
  simp only [lift,bind_ok] at computed
  obtain ⟨plus,plusComputed,computed⟩ := source_bind_success _ _ _ computed
  cases plus with
  | true =>
    simp only [ite_true] at computed
    have plusMeaning := offset_sign source tokens position _ "+" (by decide) plusComputed
    obtain ⟨numberPosition,numberAdded,computed⟩ := source_bind_success _ _ _ computed
    obtain ⟨numberResult,numberComputed,computed⟩ := source_bind_success _ _ _ computed
    cases numberResult with
    | none => simp at computed
    | some pair =>
      rcases pair with ⟨number,next⟩
      change (if number>9223372036854775807#u64 then (ok none : Result (Option (I64 × Usize)))
        else ok (some (UScalar.hcast .I64 number,next)))=ok (some (offset,finish)) at computed
      split at computed
      · simp at computed
      · simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
        rcases computed with ⟨rfl,rfl⟩
        have bound : number.val<2^63 := by scalar_tac
        have numberPositionEq := source_usize_add_value position 1#usize numberPosition numberAdded
        obtain ⟨finishEq,numberMeaning⟩ := source_unsigned source tokens numberPosition next number numberComputed
        have nextEq : next.val=position.val+2 := by scalar_tac
        rw [offset_cast_value number bound,nextEq]
        exact OffsetSyntax.positive plusMeaning (by simpa [numberPositionEq] using numberMeaning) bound
  | false =>
    simp only [Bool.false_eq_true,ite_false,ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
    rcases computed with ⟨rfl,rfl⟩
    exact OffsetSyntax.absent position.val

end PtxFrontendProgram

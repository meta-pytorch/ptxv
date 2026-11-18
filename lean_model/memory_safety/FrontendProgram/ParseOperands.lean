/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Operands
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

theorem source_parse_operand (source : Slice U8) (tokens : Slice SourceToken) (position : Usize) (scope : Scope)
    (bits : U8) (operand : Operand) (finish : Usize) (counts : PositiveCounts scope)
    (computed : strict_memory_frontend_parse.parse_operand source tokens position scope bits=.ok (some (operand,finish))) :
    OperandSyntax source.val scope tokens.val bits.val position.val finish.val operand := by
  by_cases bits32 : bits=32#u8
  · subst bits
    unfold strict_memory_frontend_parse.parse_operand at computed
    dsimp only at computed
    split at computed
    · simp at computed
    · have different : (64#u8:U8)≠32#u8 := by decide
      simp only [bne_self_eq_false,Bool.false_eq_true,ite_false,different,bne_iff_ne,ite_true,lift,bind_ok] at computed
      obtain ⟨negative,negativeComputed,computed⟩ := source_bind_success _ _ _ computed
      cases negative with
      | false =>
        simp only [Bool.false_eq_true,ite_false] at computed
        obtain ⟨token,tokenComputed,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨value,valueComputed,computed⟩ := source_bind_success _ _ _ computed
        cases value with
        | none => simp at computed
        | some raw =>
          obtain ⟨next,nextComputed,computed⟩ := source_bind_success _ _ _ computed
          simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
          rcases computed with ⟨rfl,rfl⟩
          have nextEq := source_usize_add_value position 1#usize next nextComputed
          rw [nextEq]
          exact OperandSyntax.single (source_slice_index tokens position token tokenComputed)
            (source_operand source scope token 32#u8 raw counts valueComputed)
      | true =>
        simp only [ite_true] at computed
        obtain ⟨after,afterComputed,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨value,valueComputed,computed⟩ := source_bind_success _ _ _ computed
        cases value with
        | none => simp at computed
        | some pair =>
          rcases pair with ⟨magnitude,next⟩
          dsimp only at computed
          change (if magnitude>2147483648#u64 then (ok none : Result (Option (Operand × Usize))) else ok (some (.Immediate (core.num.U64.wrapping_sub 0#u64 magnitude),next)))=ok (some (operand,finish)) at computed
          split at computed
          · simp at computed
          · simp only [lift,bind_ok,ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
            rcases computed with ⟨rfl,rfl⟩
            have afterEq := source_usize_add_value position 1#usize after afterComputed
            obtain ⟨finishEq,number⟩ := source_unsigned source tokens after next magnitude valueComputed
            have minus : TokenAt source.val tokens.val position.val "-" := by
              obtain ⟨token,found,meaning⟩ := source_matches source tokens position _ negativeComputed
              have literal : utf8 "-"=[45#u8] := by decide
              exact ⟨token,found,by simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning⟩
            have finishPosition : next.val=position.val+2 := by scalar_tac
            rw [finishPosition]
            refine OperandSyntax.negative _ minus (by simpa [afterEq] using number) (Or.inl rfl) (by scalar_tac) ?_
            simp [core.num.U64.wrapping_sub_val_eq,UScalar.size_def]
  · by_cases bits64 : bits=64#u8
    · subst bits
      unfold strict_memory_frontend_parse.parse_operand at computed
      dsimp only at computed
      split at computed
      · simp at computed
      · have different : (64#u8:U8)≠32#u8 := by decide
        simp only [bne_self_eq_false,Bool.false_eq_true,ite_false,different,bne_iff_ne,ite_true,lift,bind_ok] at computed
        obtain ⟨negative,negativeComputed,computed⟩ := source_bind_success _ _ _ computed
        cases negative with
        | false =>
          simp only [Bool.false_eq_true,ite_false] at computed
          obtain ⟨token,tokenComputed,computed⟩ := source_bind_success _ _ _ computed
          obtain ⟨value,valueComputed,computed⟩ := source_bind_success _ _ _ computed
          cases value with
          | none => simp at computed
          | some raw =>
            obtain ⟨next,nextComputed,computed⟩ := source_bind_success _ _ _ computed
            simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
            rcases computed with ⟨rfl,rfl⟩
            have nextEq := source_usize_add_value position 1#usize next nextComputed
            rw [nextEq]
            exact OperandSyntax.single (source_slice_index tokens position token tokenComputed)
              (source_operand source scope token 64#u8 raw counts valueComputed)
        | true =>
          simp only [ite_true] at computed
          obtain ⟨after,afterComputed,computed⟩ := source_bind_success _ _ _ computed
          obtain ⟨value,valueComputed,computed⟩ := source_bind_success _ _ _ computed
          cases value with
          | none => simp at computed
          | some pair =>
            rcases pair with ⟨magnitude,next⟩
            dsimp only at computed
            change (if magnitude>9223372036854775808#u64 then (ok none : Result (Option (Operand × Usize))) else ok (some (.Immediate (core.num.U64.wrapping_sub 0#u64 magnitude),next)))=ok (some (operand,finish)) at computed
            split at computed
            · simp at computed
            · simp only [lift,bind_ok,ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
              rcases computed with ⟨rfl,rfl⟩
              have afterEq := source_usize_add_value position 1#usize after afterComputed
              obtain ⟨finishEq,number⟩ := source_unsigned source tokens after next magnitude valueComputed
              have minus : TokenAt source.val tokens.val position.val "-" := by
                obtain ⟨token,found,meaning⟩ := source_matches source tokens position _ negativeComputed
                have literal : utf8 "-"=[45#u8] := by decide
                exact ⟨token,found,by simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning⟩
              have finishPosition : next.val=position.val+2 := by scalar_tac
              rw [finishPosition]
              refine OperandSyntax.negative _ minus (by simpa [afterEq] using number) (Or.inr rfl) (by scalar_tac) ?_
              simp [core.num.U64.wrapping_sub_val_eq,UScalar.size_def]
    · have not32 : (bits != 32#u8)=true := by scalar_tac
      have not64 : (bits != 64#u8)=true := by scalar_tac
      simp [strict_memory_frontend_parse.parse_operand,not32,not64] at computed

end PtxFrontendProgram

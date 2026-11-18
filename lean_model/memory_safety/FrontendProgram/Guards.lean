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

theorem source_parse_guard (source : Slice U8) (tokens : Slice SourceToken) (position : Usize) (scope : Scope)
    (guard : ptxv_memory_write.value_flow.Guard) (finish : Usize) (counts : PositiveCounts scope)
    (computed : strict_memory_frontend_parse.parse_guard source tokens position scope=.ok (some (guard,finish))) :
    GuardSyntax source.val scope tokens.val position.val finish.val guard := by
  unfold strict_memory_frontend_parse.parse_guard at computed
  simp only [lift,bind_ok] at computed
  obtain ⟨present,presentComputed,computed⟩ := source_bind_success _ _ _ computed
  cases present with
  | false =>
    simp only [Bool.false_eq_true,ite_false,ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
    rcases computed with ⟨rfl,rfl⟩
    exact GuardSyntax.unguarded position.val
  | true =>
    simp only [ite_true] at computed
    obtain ⟨after,afterComputed,computed⟩ := source_bind_success _ _ _ computed
    obtain ⟨negative,negativeComputed,computed⟩ := source_bind_success _ _ _ computed
    have afterEq := source_usize_add_value position 1#usize after afterComputed
    have atMeaning := source_matches source tokens position _ presentComputed
    have atSign : TokenAt source.val tokens.val position.val "@" := by
      obtain ⟨token,found,meaning⟩ := atMeaning
      have literal : utf8 "@"=[64#u8] := by decide
      exact ⟨token,found,by simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning⟩
    cases negative with
    | false =>
      simp only [Bool.false_eq_true,ite_false,bind_ok] at computed
      split at computed
      · simp at computed
      · obtain ⟨token,tokenComputed,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨register,registerComputed,computed⟩ := source_bind_success _ _ _ computed
        cases register with
        | none => simp at computed
        | some index =>
          obtain ⟨next,nextComputed,computed⟩ := source_bind_success _ _ _ computed
          simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
          rcases computed with ⟨rfl,rfl⟩
          have nextEq := source_usize_add_value after 1#usize next nextComputed
          have ref := source_lookup_register source scope token 1#u8 true index counts registerComputed
          have found := source_slice_index tokens after token tokenComputed
          have nextPosition : next.val=position.val+2 := by scalar_tac
          rw [nextPosition]
          exact GuardSyntax.positive index atSign (by simpa [afterEq] using found) ref
    | true =>
      simp only [ite_true] at computed
      obtain ⟨second,secondComputed,computed⟩ := source_bind_success _ _ _ computed
      split at computed
      · simp at computed
      · obtain ⟨token,tokenComputed,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨register,registerComputed,computed⟩ := source_bind_success _ _ _ computed
        cases register with
        | none => simp at computed
        | some index =>
          obtain ⟨next,nextComputed,computed⟩ := source_bind_success _ _ _ computed
          simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
          rcases computed with ⟨rfl,rfl⟩
          have secondEq := source_usize_add_value after 1#usize second secondComputed
          have nextEq := source_usize_add_value second 1#usize next nextComputed
          have ref := source_lookup_register source scope token 1#u8 true index counts registerComputed
          have found := source_slice_index tokens second token tokenComputed
          have notSign : TokenAt source.val tokens.val (position.val+1) "!" := by
            obtain ⟨token,found,meaning⟩ := source_matches source tokens after _ negativeComputed
            refine ⟨token,?_,?_⟩
            · simpa [afterEq] using found
            · have literal : utf8 "!"=[33#u8] := by decide
              simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
          have secondPosition : second.val=position.val+2 := by scalar_tac
          have nextPosition : next.val=position.val+3 := by scalar_tac
          rw [nextPosition]
          exact GuardSyntax.negative index atSign notSign (by simpa [secondPosition] using found) ref

end PtxFrontendProgram

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Identifiers
import FrontendProgram.RegisterNames

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.split false
namespace PtxFrontendProgram

private def nondigitResult (byte : U8) : Result Bool :=
  if byte<48#u8 then .ok true else .ok (byte>57#u8)

private theorem nondigit_result (byte : U8) :
    nondigitResult byte=.ok (decide (byte.val<48 ∨ 57<byte.val)) := by
  unfold nondigitResult
  split <;> simp_all

theorem source_indexed_stem_supported (source : Slice U8) (token : SourceToken)
    (computed : strict_memory_frontend_parse.indexed_stem_supported source token=.ok true) :
    IndexedStem source.val token := by
  unfold strict_memory_frontend_parse.indexed_stem_supported at computed
  cases wordEq : token.word with
  | false => simp only [wordEq,Bool.false_eq_true,ite_false] at computed; simp at computed
  | true =>
    simp only [wordEq,ite_true] at computed
    by_cases empty : token.start≥token.end
    · simp only [empty,ite_true] at computed; simp at computed
    · simp only [empty,ite_false] at computed
      by_cases outside : token.end>Slice.len source
      · simp only [outside,ite_true] at computed; simp at computed
      · simp only [outside,ite_false] at computed
        have positive : token.start.val<token.end.val := by scalar_tac
        have bound : token.end.val≤source.val.length := by scalar_tac
        have ordered : (1#usize).val≤token.end.val := by scalar_tac
        rw [source_usize_sub token.end 1#usize ordered] at computed
        simp only [bind_ok] at computed
        obtain ⟨byte,read,computed⟩:=source_bind_success _ _ _ computed
        change nondigitResult byte=.ok true at computed
        rw [nondigit_result] at computed
        have understood : byte.val<48 ∨ 57<byte.val := by
          apply of_decide_eq_true
          simpa only [ok.injEq] using computed
        refine ⟨wordEq,positive,bound,byte,?_,understood⟩
        simpa using source_slice_index source _ byte read

private theorem indexed_names_false (source : Slice U8) (scope : Scope)
    (index : Usize) :
    strict_memory_frontend_parse.scope_indexed_names_supported_loop source scope false index=.ok false := by
  unfold strict_memory_frontend_parse.scope_indexed_names_supported_loop
  rw [loop.eq_def]
  simp [strict_memory_frontend_parse.scope_indexed_names_supported_loop.body,bind_ok]

private theorem indexed_names_loop (source : Slice U8) (scope : Scope)
    (index : Usize)
    (computed : strict_memory_frontend_parse.scope_indexed_names_supported_loop source scope true index=.ok true) :
    ∀ position declaration,index.val≤position → scope.declarations.val[position]?=some declaration →
      declaration.indexed=true → IndexedStem source.val declaration.name := by
  generalize fuelEq : scope.declarations.val.length-index.val=fuel
  induction fuel using Nat.strong_induction_on generalizing index with
  | h fuel ih =>
    have unroll : strict_memory_frontend_parse.scope_indexed_names_supported_loop source scope true index=(do
        let outcome ← strict_memory_frontend_parse.scope_indexed_names_supported_loop.body source scope true index
        match outcome with
        | .cont pair => strict_memory_frontend_parse.scope_indexed_names_supported_loop source scope pair.1 pair.2
        | .done value => .ok value) := by
      unfold strict_memory_frontend_parse.scope_indexed_names_supported_loop
      rw [loop.eq_def]
      congr 1
      funext outcome
      cases outcome with
      | cont pair => cases pair;rfl
      | done value => rfl
    rw [unroll] at computed
    by_cases inside : index.val<scope.declarations.val.length
    · have test : index<alloc.vec.Vec.len scope.declarations := by scalar_tac
      let declaration := scope.declarations.val[index.val]
      have read : alloc.vec.Vec.index_usize scope.declarations index=.ok declaration := by
        simp [alloc.vec.Vec.index_usize,List.getElem?_eq_getElem inside,declaration]; rfl
      have incrementBound : index.val+1<2^UScalarTy.Usize.numBits := by
        have := alloc.vec.Vec.len_ineq scope.declarations
        scalar_tac
      let after := Usize.ofNatCore (index.val+1) incrementBound
      have afterVal : after.val=index.val+1 := by simp [after]
      have increment := source_usize_increment index incrementBound
      simp only [strict_memory_frontend_parse.scope_indexed_names_supported_loop.body,
        ite_true,test,alloc.vec.Vec.index_slice_index,read,bind_ok,Std.bind_assoc] at computed
      obtain ⟨supported,supportedComputed,computed⟩:=source_bind_success _ _ _ computed
      cases supported with
      | false =>
        simp only [increment,bind_ok] at computed
        change strict_memory_frontend_parse.scope_indexed_names_supported_loop source scope false after=.ok true at computed
        rw [indexed_names_false] at computed
        simp at computed
      | true =>
        simp only [increment,bind_ok] at computed
        change strict_memory_frontend_parse.scope_indexed_names_supported_loop source scope true after=.ok true at computed
        have tail:=ih (scope.declarations.val.length-after.val) (by omega) after computed rfl
        intro position selected lower found
        by_cases samePosition : position=index.val
        · subst position
          have selectedEq : selected=declaration := by
            have actual : scope.declarations.val[index.val]?=some declaration:=List.getElem?_eq_getElem inside
            rw [found] at actual
            exact Option.some.inj actual
          subst selected
          intro pred
          simp only [pred,ite_true] at supportedComputed
          exact source_indexed_stem_supported source declaration.name supportedComputed
        · exact tail position selected (by omega) found
    · intro position declaration lower found
      have := (List.getElem?_eq_some_iff.mp found).1
      omega

theorem source_scope_indexed_names_supported (source : Slice U8) (scope : Scope)
    (computed : strict_memory_frontend_parse.scope_indexed_names_supported source scope=.ok true) :
    IndexedNamesSupported source.val scope := by
  unfold strict_memory_frontend_parse.scope_indexed_names_supported at computed
  intro declaration member
  obtain ⟨position,found⟩:=List.mem_iff_getElem?.mp member
  exact indexed_names_loop source scope 0#usize computed position declaration (by simp) found

end PtxFrontendProgram

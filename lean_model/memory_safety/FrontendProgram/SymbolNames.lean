/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Grammar
import FrontendProgram.TokenComplete

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.split false

namespace PtxFrontendProgram

theorem source_symbol_name_supported (source : Slice U8) (token : SourceToken)
    (computed : strict_memory_frontend_parse.symbol_name_supported source token=.ok true) :
    SymbolNameSupported source.val token := by
  unfold strict_memory_frontend_parse.symbol_name_supported at computed
  repeat' first
    | fail_if_no_progress simp only [lift,bind_ok,ite_true,Bool.false_eq_true,ite_false] at computed
    | peel_frontend_bool computed
  all_goals try { simp at computed }
  intro text member spelling
  simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl
  · apply source_token_equals_false source ((Array.make 1#usize [95#u8]).to_slice) token (by assumption)
    have literal : utf8 "_"=[95#u8] := by decide
    simpa [TokenSpelling,literal,Array.to_slice,Array.make] using spelling
  · apply source_token_equals_false source ((Array.make 1#usize [36#u8]).to_slice) token (by assumption)
    have literal : utf8 "$"=[36#u8] := by decide
    simpa [TokenSpelling,literal,Array.to_slice,Array.make] using spelling
  · apply source_token_equals_false source ((Array.make 7#usize [87#u8, 65#u8, 82#u8, 80#u8, 95#u8, 83#u8, 90#u8]).to_slice) token (by assumption)
    have literal : utf8 "WARP_SZ"=[87#u8, 65#u8, 82#u8, 80#u8, 95#u8, 83#u8, 90#u8] := by decide
    simpa [TokenSpelling,literal,Array.to_slice,Array.make] using spelling

private theorem symbol_globals_loop (source : Slice U8) (scope : Scope)
    (valid : Bool) (index : Usize)
    (resultScope : Scope)
    (computed : strict_memory_frontend_parse.scope_symbol_names_supported_loop0 source scope valid index=.ok (resultScope,true)) :
    resultScope=scope ∧ valid=true ∧
      ∀ position selected,index.val≤position → scope.globals.val[position]?=some selected →
        SymbolNameSupported source.val selected.name := by
  generalize fuelEq : scope.globals.val.length-index.val=fuel
  induction fuel using Nat.strong_induction_on generalizing index valid with
  | h fuel ih =>
    cases valid with
    | false =>
      unfold strict_memory_frontend_parse.scope_symbol_names_supported_loop0 at computed
      rw [loop.eq_def] at computed
      simp [strict_memory_frontend_parse.scope_symbol_names_supported_loop0.body,bind_ok] at computed
    | true =>
      have unroll : strict_memory_frontend_parse.scope_symbol_names_supported_loop0 source scope true index=(do
          let outcome ← strict_memory_frontend_parse.scope_symbol_names_supported_loop0.body source scope true index
          match outcome with
          | .cont pair => strict_memory_frontend_parse.scope_symbol_names_supported_loop0 source scope pair.1 pair.2
          | .done value => .ok value) := by
        unfold strict_memory_frontend_parse.scope_symbol_names_supported_loop0
        rw [loop.eq_def]
        congr 1
        funext outcome
        cases outcome with
        | cont pair => cases pair;rfl
        | done value => rfl
      rw [unroll] at computed
      by_cases inside : index.val<scope.globals.val.length
      · have test : index<alloc.vec.Vec.len scope.globals := by scalar_tac
        let selected := scope.globals.val[index.val]
        have read : alloc.vec.Vec.index_usize scope.globals index=.ok selected := by
          simp [alloc.vec.Vec.index_usize,List.getElem?_eq_getElem inside,selected]; rfl
        have incrementBound : index.val+1<2^UScalarTy.Usize.numBits := by
          have := alloc.vec.Vec.len_ineq scope.globals
          scalar_tac
        let after := Usize.ofNatCore (index.val+1) incrementBound
        have afterVal : after.val=index.val+1 := by simp [after]
        have increment := source_usize_increment index incrementBound
        simp only [strict_memory_frontend_parse.scope_symbol_names_supported_loop0.body,
          ite_true,test,alloc.vec.Vec.index_slice_index,read,bind_ok,Std.bind_assoc] at computed
        obtain ⟨supported,supportedComputed,computed⟩:=source_bind_success _ _ _ computed
        simp only [increment,bind_ok] at computed
        change strict_memory_frontend_parse.scope_symbol_names_supported_loop0 source scope supported after=.ok (resultScope,true) at computed
        have tail:=ih (scope.globals.val.length-after.val) (by omega) supported after computed rfl
        refine ⟨tail.1,rfl,?_⟩
        intro position found lower selectedAt
        by_cases samePosition : position=index.val
        · subst position
          have foundEq : found=selected := by
            have actual : scope.globals.val[index.val]?=some selected:=List.getElem?_eq_getElem inside
            rw [selectedAt] at actual
            exact Option.some.inj actual
          subst found
          apply source_symbol_name_supported source selected.name
          simpa only [tail.2.1] using supportedComputed
        · exact tail.2.2 position found (by omega) selectedAt
      · have test : ¬index<alloc.vec.Vec.len scope.globals := by scalar_tac
        simp only [strict_memory_frontend_parse.scope_symbol_names_supported_loop0.body,ite_true,test,ite_false] at computed
        have pairEq : (scope,true)=(resultScope,true) := by
          simpa only [bind_ok,ok.injEq] using computed
        have frame : resultScope=scope := congrArg Prod.fst pairEq.symm
        refine ⟨frame,rfl,?_⟩
        intro position selected lower found
        have := (List.getElem?_eq_some_iff.mp found).1
        omega

private theorem symbol_labels_loop (source : Slice U8) (scope : Scope)
    (valid : Bool) (index : Usize)
    (computed : strict_memory_frontend_parse.scope_symbol_names_supported_loop1 source scope valid index=.ok true) :
    valid=true ∧
      ∀ position selected,index.val≤position → scope.labels.val[position]?=some selected →
        SymbolNameSupported source.val selected.name := by
  generalize fuelEq : scope.labels.val.length-index.val=fuel
  induction fuel using Nat.strong_induction_on generalizing index valid with
  | h fuel ih =>
    cases valid with
    | false =>
      unfold strict_memory_frontend_parse.scope_symbol_names_supported_loop1 at computed
      rw [loop.eq_def] at computed
      simp [strict_memory_frontend_parse.scope_symbol_names_supported_loop1.body,bind_ok] at computed
    | true =>
      have unroll : strict_memory_frontend_parse.scope_symbol_names_supported_loop1 source scope true index=(do
          let outcome ← strict_memory_frontend_parse.scope_symbol_names_supported_loop1.body source scope true index
          match outcome with
          | .cont pair => strict_memory_frontend_parse.scope_symbol_names_supported_loop1 source scope pair.1 pair.2
          | .done value => .ok value) := by
        unfold strict_memory_frontend_parse.scope_symbol_names_supported_loop1
        rw [loop.eq_def]
        congr 1
        funext outcome
        cases outcome with
        | cont pair => cases pair;rfl
        | done value => rfl
      rw [unroll] at computed
      by_cases inside : index.val<scope.labels.val.length
      · have test : index<alloc.vec.Vec.len scope.labels := by scalar_tac
        let selected := scope.labels.val[index.val]
        have read : alloc.vec.Vec.index_usize scope.labels index=.ok selected := by
          simp [alloc.vec.Vec.index_usize,List.getElem?_eq_getElem inside,selected]; rfl
        have incrementBound : index.val+1<2^UScalarTy.Usize.numBits := by
          have := alloc.vec.Vec.len_ineq scope.labels
          scalar_tac
        let after := Usize.ofNatCore (index.val+1) incrementBound
        have afterVal : after.val=index.val+1 := by simp [after]
        have increment := source_usize_increment index incrementBound
        simp only [strict_memory_frontend_parse.scope_symbol_names_supported_loop1.body,
          ite_true,test,alloc.vec.Vec.index_slice_index,read,bind_ok,Std.bind_assoc] at computed
        obtain ⟨supported,supportedComputed,computed⟩:=source_bind_success _ _ _ computed
        simp only [increment,bind_ok] at computed
        change strict_memory_frontend_parse.scope_symbol_names_supported_loop1 source scope supported after=.ok true at computed
        have tail:=ih (scope.labels.val.length-after.val) (by omega) supported after computed rfl
        refine ⟨rfl,?_⟩
        intro position found lower selectedAt
        by_cases samePosition : position=index.val
        · subst position
          have foundEq : found=selected := by
            have actual : scope.labels.val[index.val]?=some selected:=List.getElem?_eq_getElem inside
            rw [selectedAt] at actual
            exact Option.some.inj actual
          subst found
          apply source_symbol_name_supported source selected.name
          simpa only [tail.1] using supportedComputed
        · exact tail.2 position found (by omega) selectedAt
      · have test : ¬index<alloc.vec.Vec.len scope.labels := by scalar_tac
        simp only [strict_memory_frontend_parse.scope_symbol_names_supported_loop1.body,ite_true,test,ite_false] at computed
        refine ⟨rfl,?_⟩
        intro position selected lower found
        have := (List.getElem?_eq_some_iff.mp found).1
        omega

theorem source_scope_symbol_names_supported (source : Slice U8) (scope : Scope) (entry : SourceToken)
    (computed : strict_memory_frontend_parse.scope_symbol_names_supported source scope entry=.ok true) :
    SymbolNamesSupported source.val scope entry := by
  unfold strict_memory_frontend_parse.scope_symbol_names_supported at computed
  obtain ⟨entrySupported,entryComputed,computed⟩:=source_bind_success _ _ _ computed
  obtain ⟨middle,globalsComputed,computed⟩:=source_bind_success _ _ _ computed
  obtain ⟨scopeAfter,valid⟩:=middle
  have labels:=symbol_labels_loop source scopeAfter valid 0#usize computed
  have globals:=symbol_globals_loop source scope entrySupported 0#usize scopeAfter
    (by simpa only [labels.1] using globalsComputed)
  obtain ⟨rfl,entryTrue,globalFacts⟩:=globals
  refine ⟨source_symbol_name_supported source entry (by simpa only [entryTrue] using entryComputed),?_,?_⟩
  · intro global member
    obtain ⟨position,found⟩:=List.mem_iff_getElem?.mp member
    exact globalFacts position global (by simp) found
  · intro label member
    obtain ⟨position,found⟩:=List.mem_iff_getElem?.mp member
    exact labels.2 position label (by simp) found

end PtxFrontendProgram

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.IndexedNames
import FrontendProgram.PredicateNames
import FrontendProgram.NamesDecision
import FrontendProgram.RegisterNames
import FrontendProgram.SymbolNames
import FrontendProgram.SourceText

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.split false

namespace PtxFrontendProgram

private theorem global_symbols_false (source : Slice U8) (token : SourceToken)
    (labels : Slice Label) (index : Usize) :
    strict_memory_frontend_parse.global_symbol_supported_loop source token labels false index=.ok false := by
  unfold strict_memory_frontend_parse.global_symbol_supported_loop
  rw [loop.eq_def]
  simp [strict_memory_frontend_parse.global_symbol_supported_loop.body,bind_ok]

private theorem global_symbols_loop (source : Slice U8) (token : SourceToken)
    (labels : Slice Label) (index : Usize)
    (computed : strict_memory_frontend_parse.global_symbol_supported_loop source token labels true index=.ok true) :
    ∀ position label,index.val≤position → labels.val[position]?=some label →
      ¬NameEqual source.val token label.name := by
  generalize fuelEq : labels.val.length-index.val=fuel
  induction fuel using Nat.strong_induction_on generalizing index with
  | h fuel ih =>
    have unroll : strict_memory_frontend_parse.global_symbol_supported_loop source token labels true index=(do
        let outcome ← strict_memory_frontend_parse.global_symbol_supported_loop.body source token labels true index
        match outcome with
        | .cont pair => strict_memory_frontend_parse.global_symbol_supported_loop source token labels pair.1 pair.2
        | .done value => .ok value) := by
      unfold strict_memory_frontend_parse.global_symbol_supported_loop
      rw [loop.eq_def]
      congr 1
      funext outcome
      cases outcome with
      | cont pair => cases pair;rfl
      | done value => rfl
    rw [unroll] at computed
    by_cases inside : index.val<labels.val.length
    · have test : index<Slice.len labels := by scalar_tac
      let label := labels.val[index.val]
      have read : Slice.index_usize labels index=.ok label := by
        simp [Slice.index_usize,List.getElem?_eq_getElem inside,label]; rfl
      have incrementBound : index.val+1<2^UScalarTy.Usize.numBits := by
        have := Slice.length_ineq labels
        scalar_tac
      let after := Usize.ofNatCore (index.val+1) incrementBound
      have afterVal : after.val=index.val+1 := by simp [after]
      have increment := source_usize_increment index incrementBound
      simp only [strict_memory_frontend_parse.global_symbol_supported_loop.body,
        ite_true,test,read,bind_ok,Std.bind_assoc] at computed
      obtain ⟨same,sameComputed,computed⟩:=source_bind_success _ _ _ computed
      cases same with
      | true =>
        simp only [increment,bind_ok] at computed
        change strict_memory_frontend_parse.global_symbol_supported_loop source token labels false after=.ok true at computed
        rw [global_symbols_false] at computed
        simp at computed
      | false =>
        simp only [increment,bind_ok] at computed
        change strict_memory_frontend_parse.global_symbol_supported_loop source token labels true after=.ok true at computed
        have tail:=ih (labels.val.length-after.val) (by omega) after computed rfl
        intro position selected lower found
        by_cases samePosition : position=index.val
        · subst position
          have selectedEq : selected=label := by
            have actual : labels.val[index.val]?=some label:=List.getElem?_eq_getElem inside
            rw [found] at actual
            exact Option.some.inj actual
          subst selected
          exact source_names_equal_false source token label.name sameComputed
        · exact tail position selected (by omega) found
    · intro position label lower found
      have := (List.getElem?_eq_some_iff.mp found).1
      omega

theorem source_global_symbol_supported (source : Slice U8) (token entry : SourceToken)
    (labels : Slice Label)
    (computed : strict_memory_frontend_parse.global_symbol_supported source token entry labels=.ok true) :
    ¬NameEqual source.val token entry ∧
      ∀ label∈labels.val,¬NameEqual source.val token label.name := by
  unfold strict_memory_frontend_parse.global_symbol_supported at computed
  obtain ⟨same,sameComputed,computed⟩:=source_bind_success _ _ _ computed
  cases same with
  | true =>
    change strict_memory_frontend_parse.global_symbol_supported_loop source token labels false 0#usize=.ok true at computed
    rw [global_symbols_false] at computed
    simp at computed
  | false =>
    change strict_memory_frontend_parse.global_symbol_supported_loop source token labels true 0#usize=.ok true at computed
    refine ⟨source_names_equal_false source token entry sameComputed,?_⟩
    intro label member
    obtain ⟨position,found⟩:=List.mem_iff_getElem?.mp member
    exact global_symbols_loop source token labels 0#usize computed position label (by simp) found

private theorem scope_global_symbols_false (source : Slice U8) (scope : Scope)
    (entry : SourceToken) (index : Usize) :
    strict_memory_frontend_parse.scope_global_symbols_supported_loop source scope entry false index=.ok false := by
  unfold strict_memory_frontend_parse.scope_global_symbols_supported_loop
  rw [loop.eq_def]
  simp [strict_memory_frontend_parse.scope_global_symbols_supported_loop.body,bind_ok]

private theorem scope_global_symbols_loop (source : Slice U8) (scope : Scope)
    (entry : SourceToken) (index : Usize)
    (computed : strict_memory_frontend_parse.scope_global_symbols_supported_loop source scope entry true index=.ok true) :
    ∀ position global,index.val≤position → scope.globals.val[position]?=some global →
      ¬NameEqual source.val global.name entry ∧
      ∀ label∈scope.labels.val,¬NameEqual source.val global.name label.name := by
  generalize fuelEq : scope.globals.val.length-index.val=fuel
  induction fuel using Nat.strong_induction_on generalizing index with
  | h fuel ih =>
    have unroll : strict_memory_frontend_parse.scope_global_symbols_supported_loop source scope entry true index=(do
        let outcome ← strict_memory_frontend_parse.scope_global_symbols_supported_loop.body source scope entry true index
        match outcome with
        | .cont pair => strict_memory_frontend_parse.scope_global_symbols_supported_loop source scope entry pair.1 pair.2
        | .done value => .ok value) := by
      unfold strict_memory_frontend_parse.scope_global_symbols_supported_loop
      rw [loop.eq_def]
      congr 1
      funext outcome
      cases outcome with
      | cont pair => cases pair;rfl
      | done value => rfl
    rw [unroll] at computed
    by_cases inside : index.val<scope.globals.val.length
    · have test : index<alloc.vec.Vec.len scope.globals := by scalar_tac
      let global := scope.globals.val[index.val]
      have read : alloc.vec.Vec.index_usize scope.globals index=.ok global := by
        simp [alloc.vec.Vec.index_usize,List.getElem?_eq_getElem inside,global]; rfl
      have incrementBound : index.val+1<2^UScalarTy.Usize.numBits := by
        have := alloc.vec.Vec.len_ineq scope.globals
        scalar_tac
      let after := Usize.ofNatCore (index.val+1) incrementBound
      have afterVal : after.val=index.val+1 := by simp [after]
      have increment := source_usize_increment index incrementBound
      simp only [strict_memory_frontend_parse.scope_global_symbols_supported_loop.body,
        ite_true,test,alloc.vec.Vec.index_slice_index,read,bind_ok,Std.bind_assoc] at computed
      obtain ⟨supported,supportedComputed,computed⟩:=source_bind_success _ _ _ computed
      cases supported with
      | false =>
        simp only [increment,bind_ok] at computed
        change strict_memory_frontend_parse.scope_global_symbols_supported_loop source scope entry false after=.ok true at computed
        rw [scope_global_symbols_false] at computed
        simp at computed
      | true =>
        simp only [increment,bind_ok] at computed
        change strict_memory_frontend_parse.scope_global_symbols_supported_loop source scope entry true after=.ok true at computed
        have tail:=ih (scope.globals.val.length-after.val) (by omega) after computed rfl
        intro position selected lower found
        by_cases samePosition : position=index.val
        · subst position
          have selectedEq : selected=global := by
            have actual : scope.globals.val[index.val]?=some global:=List.getElem?_eq_getElem inside
            rw [found] at actual
            exact Option.some.inj actual
          subst selected
          simpa only [alloc.vec.Vec.deref,Slice.from_val] using
            source_global_symbol_supported source global.name entry scope.labels.deref supportedComputed
        · exact tail position selected (by omega) found
    · intro position global lower found
      have := (List.getElem?_eq_some_iff.mp found).1
      omega

theorem source_scope_global_symbols_supported (source : Slice U8) (scope : Scope) (entry : SourceToken)
    (computed : strict_memory_frontend_parse.scope_global_symbols_supported source scope entry=.ok true) :
    GlobalSymbolsSeparate source.val scope entry := by
  unfold strict_memory_frontend_parse.scope_global_symbols_supported at computed
  intro global member
  obtain ⟨position,found⟩:=List.mem_iff_getElem?.mp member
  exact scope_global_symbols_loop source scope entry 0#usize computed position global (by simp) found

theorem source_scope_symbols_supported (source : Slice U8) (scope : Scope) (entry : SourceToken)
    (computed : strict_memory_frontend_parse.scope_symbols_supported source scope entry=.ok true) :
    ScopeSymbolsSeparate source.val scope entry := by
  unfold strict_memory_frontend_parse.scope_symbols_supported at computed
  obtain ⟨globals,globalsComputed,computed⟩:=source_bind_success _ _ _ computed
  cases globals with
  | false => simp at computed
  | true =>
    simp only [ite_true] at computed
    obtain ⟨registers,registersComputed,computed⟩:=source_bind_success _ _ _ computed
    cases registers with
    | false => simp at computed
    | true =>
      simp only [ite_true] at computed
      obtain ⟨names,namesComputed,computed⟩:=source_bind_success _ _ _ computed
      cases names with
      | false => simp at computed
      | true =>
        simp only [ite_true] at computed
        obtain ⟨text,textComputed,computed⟩:=source_bind_success _ _ _ computed
        cases text with
        | false => simp at computed
        | true =>
          simp only [ite_true] at computed
          obtain ⟨predicates,predicatesComputed,computed⟩:=source_bind_success _ _ _ computed
          cases predicates with
          | false => simp at computed
          | true =>
            simp only [ite_true] at computed
            exact ⟨source_scope_global_symbols_supported source scope entry globalsComputed,
              source_scope_register_names_supported source scope registersComputed,
              source_scope_symbol_names_supported source scope entry namesComputed,
              source_text_supported source textComputed,
              source_scope_predicate_names_supported source scope predicatesComputed,
              source_scope_indexed_names_supported source scope computed⟩

theorem module_scope_symbols_separate (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram)
    (meaning : ModuleMeaning source entry program) :
    ScopeSymbolsSeparate source program.scope program.entry := by
  obtain ⟨tokens,globalsEnd,declarationsStart,instructionsStart,finish,meaning⟩:=meaning
  exact meaning.2.2.2.2.2.2.2.2.2.2.2.2


theorem module_source_text (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram)
    (meaning : ModuleMeaning source entry program) : SourceText source :=
  (module_scope_symbols_separate source entry program meaning).2.2.2.1

theorem module_predicate_names_supported (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram)
    (meaning : ModuleMeaning source entry program) : PredicateNamesSupported source program.scope :=
  (module_scope_symbols_separate source entry program meaning).2.2.2.2.1

theorem module_indexed_names_supported (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram)
    (meaning : ModuleMeaning source entry program) : IndexedNamesSupported source program.scope :=
  (module_scope_symbols_separate source entry program meaning).2.2.2.2.2

end PtxFrontendProgram

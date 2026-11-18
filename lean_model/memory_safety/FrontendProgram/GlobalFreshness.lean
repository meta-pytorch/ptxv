/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.GlobalReferences
import FrontendProgram.NamesDecision

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.split false
set_option backward.isDefEq.respectTransparency false
namespace PtxFrontendProgram

def GlobalFoundBefore (source : List U8) (scope : Scope) (token : SourceToken)
    (found : Option Usize) (index : Usize) : Prop :=
  ∀ selected,found=some selected → selected.val < index.val ∧
    ∃ global,scope.globals.val[selected.val]?=some global ∧ NameEqual source global.«name» token

private theorem matched_names_equal (source : List U8) (left right token : SourceToken)
    (first : NameEqual source left token) (second : NameEqual source right token) : NameEqual source left right :=
  ⟨first.1,first.2.1,second.1,second.2.1,first.2.2.2.2.trans second.2.2.2.2.symm⟩

theorem source_global_lookup_loop_unique (source : Slice U8) (scope : Scope) (token : SourceToken)
    (found output : Option Usize) (index : Usize) (unique : Bool)
    (separate : scope.globals.val.Pairwise fun left right => ¬NameEqual source.val left.«name» right.«name»)
    (prior : GlobalFoundBefore source.val scope token found index)
    (computed : strict_memory_frontend_parse.lookup_global_loop source scope token found true index=.ok (output,unique)) :
    unique=true ∧ (output=none → found=none ∧
      ∀ position global,index.val≤position → scope.globals.val[position]?=some global → ¬NameEqual source.val global.«name» token) := by
  generalize fuelEq : scope.globals.val.length-index.val=fuel
  induction fuel using Nat.strong_induction_on generalizing index found with
  | h fuel ih =>
    have unroll : strict_memory_frontend_parse.lookup_global_loop source scope token found true index=(do
        let outcome ← strict_memory_frontend_parse.lookup_global_loop.body source scope token found true index
        match outcome with
        | .cont triple => strict_memory_frontend_parse.lookup_global_loop source scope token triple.1 triple.2.1 triple.2.2
        | .done value => .ok value) := by
      unfold strict_memory_frontend_parse.lookup_global_loop
      rw [loop.eq_def]
      congr 1
      funext outcome
      cases outcome with
      | cont triple => rcases triple with ⟨_,_,_⟩;rfl
      | done value => rfl
    rw [unroll] at computed
    by_cases inside : index.val<scope.globals.val.length
    · have test : index<alloc.vec.Vec.len scope.globals := by scalar_tac
      let global := scope.globals.val[index.val]
      have read : alloc.vec.Vec.index_usize scope.globals index=.ok global := by
        simp [alloc.vec.Vec.index_usize,List.getElem?_eq_getElem inside,global]
      have incrementBound : index.val+1<2^UScalarTy.Usize.numBits := by
        have := alloc.vec.Vec.len_ineq scope.globals
        scalar_tac
      let after := Usize.ofNatCore (index.val+1) incrementBound
      have afterVal : after.val=index.val+1 := by simp [after]
      have increment := source_usize_increment index incrementBound
      simp only [strict_memory_frontend_parse.lookup_global_loop.body,ite_true,test,
        alloc.vec.Vec.index_slice_index,read,bind_ok,Std.bind_assoc] at computed
      obtain ⟨same,sameComputed,computed⟩ := source_bind_success _ _ _ computed
      cases same with
      | false =>
        simp only [Bool.false_eq_true,ite_false,bind_ok,increment] at computed
        change Aeneas.Std.bind (ok (ControlFlow.cont (found,true,after) : ControlFlow ((Option Usize) × Bool × Usize) ((Option Usize) × Bool))) _ = ok (output,unique) at computed
        simp only [bind_ok] at computed
        have nextPrior : GlobalFoundBefore source.val scope token found after := by
          intro selected same
          obtain ⟨earlier,global,foundGlobal,matched⟩ := prior selected same
          exact ⟨by omega,global,foundGlobal,matched⟩
        obtain ⟨uniqueEq,missing⟩ := ih (scope.globals.val.length-after.val) (by omega) found after nextPrior computed rfl
        refine ⟨uniqueEq,?_⟩
        intro absent
        obtain ⟨priorMissing,tailMissing⟩ := missing absent
        refine ⟨priorMissing,?_⟩
        intro position selected lower foundSelected
        by_cases samePosition : position=index.val
        · subst position
          have selectedEq : selected=global := by
            have actual : scope.globals.val[index.val]?=some global := List.getElem?_eq_getElem inside
            rw [foundSelected] at actual
            exact Option.some.inj actual
          subst selected
          exact source_names_equal_false source global.«name» token sameComputed
        · exact tailMissing position selected (by omega) foundSelected
      | true =>
        have matched := source_names_equal source global.«name» token sameComputed
        cases found with
        | none =>
          simp only [ite_true,core.option.Option.is_none,Option.isNone,bind_ok,increment] at computed
          change Aeneas.Std.bind (ok (ControlFlow.cont (some index,true,after) : ControlFlow ((Option Usize) × Bool × Usize) ((Option Usize) × Bool))) _ = ok (output,unique) at computed
          simp only [bind_ok] at computed
          have nextPrior : GlobalFoundBefore source.val scope token (some index) after := by
            intro selected selectedEq
            cases selectedEq
            exact ⟨by omega,global,List.getElem?_eq_getElem inside,matched⟩
          obtain ⟨uniqueEq,missing⟩ := ih (scope.globals.val.length-after.val) (by omega) (some index) after nextPrior computed rfl
          refine ⟨uniqueEq,?_⟩
          intro absent
          have impossible := (missing absent).1
          cases impossible
        | some selected =>
          obtain ⟨earlier,previous,previousFound,previousMatched⟩ := prior selected rfl
          obtain ⟨previousBound,previousValue⟩ := List.getElem?_eq_some_iff.mp previousFound
          have excluded := separate.rel_getElem_of_lt previousBound inside earlier
          rw [previousValue] at excluded
          exact False.elim (excluded (matched_names_equal source.val previous.«name» global.«name» token previousMatched matched))
    · have test : ¬index<alloc.vec.Vec.len scope.globals := by scalar_tac
      simp only [strict_memory_frontend_parse.lookup_global_loop.body,ite_true,test,ite_false,bind_ok,ok.injEq,Prod.mk.injEq] at computed
      refine ⟨computed.2.symm,?_⟩
      intro absent
      refine ⟨computed.1.trans absent,?_⟩
      intro position global lower foundGlobal
      have bound := (List.getElem?_eq_some_iff.mp foundGlobal).1
      omega

theorem source_lookup_global_none (source : Slice U8) (scope : Scope) (token : SourceToken)
    (separate : scope.globals.val.Pairwise fun left right => ¬NameEqual source.val left.«name» right.«name»)
    (computed : strict_memory_frontend_parse.lookup_global source scope token=.ok none) :
    ∀ global∈scope.globals.val,¬NameEqual source.val global.«name» token := by
  unfold strict_memory_frontend_parse.lookup_global at computed
  obtain ⟨pair,loopComputed,computed⟩ := source_bind_success _ _ _ computed
  rcases pair with ⟨found,unique⟩
  have prior : GlobalFoundBefore source.val scope token none 0#usize := by
    intro selected impossible
    cases impossible
  obtain ⟨uniqueEq,missing⟩ := source_global_lookup_loop_unique source scope token none found 0#usize unique separate prior loopComputed
  cases uniqueEq
  have absent : found=none := by simpa using computed
  intro global member
  obtain ⟨position,foundGlobal⟩ := List.mem_iff_getElem?.mp member
  exact (missing absent).2 position global (by simp) foundGlobal

end PtxFrontendProgram

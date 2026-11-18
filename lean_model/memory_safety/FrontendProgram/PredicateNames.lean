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

private def digitResult (byte : U8) : Result Bool :=
  if byte>=48#u8 then .ok (byte<=57#u8) else .ok false

private theorem digit_result (byte : U8) :
    digitResult byte=.ok (decide (48≤byte.val ∧ byte.val≤57)) := by
  unfold digitResult
  split <;> simp_all

private theorem predicate_tail_false (source : Slice U8) (finish : Usize) (index : Usize) :
    strict_memory_frontend_parse.predicate_stem_supported_loop source finish false index=.ok false := by
  unfold strict_memory_frontend_parse.predicate_stem_supported_loop
  rw [loop.eq_def]
  simp [strict_memory_frontend_parse.predicate_stem_supported_loop.body,bind_ok]

private theorem predicate_tail_loop (source : Slice U8) (finish : Usize) (index : Usize)
    (bound : finish.val≤source.val.length)
    (computed : strict_memory_frontend_parse.predicate_stem_supported_loop source finish true index=.ok true) :
    ∀ position byte,index.val≤position → position<finish.val → source.val[position]?=some byte → 48≤byte.val ∧ byte.val≤57 := by
  generalize fuelEq : finish.val-index.val=fuel
  induction fuel using Nat.strong_induction_on generalizing index with
  | h fuel ih =>
    have unroll : strict_memory_frontend_parse.predicate_stem_supported_loop source finish true index=(do
        let outcome ← strict_memory_frontend_parse.predicate_stem_supported_loop.body source finish true index
        match outcome with
        | .cont pair => strict_memory_frontend_parse.predicate_stem_supported_loop source finish pair.1 pair.2
        | .done value => .ok value) := by
      unfold strict_memory_frontend_parse.predicate_stem_supported_loop
      rw [loop.eq_def]
      congr 1
      funext outcome
      cases outcome with
      | cont pair => cases pair;rfl
      | done value => rfl
    rw [unroll] at computed
    by_cases inside : index.val<finish.val
    · have test : index<finish := by scalar_tac
      have sourceInside : index.val<source.val.length := by omega
      let byte := source.val[index.val]
      have read : Slice.index_usize source index=.ok byte := by
        simp [Slice.index_usize,List.getElem?_eq_getElem sourceInside,byte]
      have incrementBound : index.val+1<2^UScalarTy.Usize.numBits := by
        have := Slice.length_ineq source
        scalar_tac
      let after := Usize.ofNatCore (index.val+1) incrementBound
      have afterVal : after.val=index.val+1 := by simp [after]
      have increment := source_usize_increment index incrementBound
      have digitEq : (if byte>=48#u8 then (ok (byte<=57#u8) : Result Bool) else ok false)=ok (decide (48≤byte.val ∧ byte.val≤57)) := digit_result byte
      simp only [strict_memory_frontend_parse.predicate_stem_supported_loop.body,
        ite_true,test,read,bind_ok,digitEq,Std.bind_assoc] at computed
      by_cases good : 48≤byte.val ∧ byte.val≤57
      · simp only [good,increment,bind_ok] at computed
        change strict_memory_frontend_parse.predicate_stem_supported_loop source finish true after=.ok true at computed
        have tail:=ih (finish.val-after.val) (by omega) after computed rfl
        intro position selected lower upper found
        by_cases samePosition : position=index.val
        · subst position
          have actual : source.val[index.val]?=some byte:=List.getElem?_eq_getElem sourceInside
          have selectedEq : selected=byte := Option.some.inj (found.symm.trans actual)
          subst selected
          exact good
        · exact tail position selected (by omega) upper found
      · simp only [good,decide_false,increment,bind_ok] at computed
        change strict_memory_frontend_parse.predicate_stem_supported_loop source finish false after=.ok true at computed
        rw [predicate_tail_false] at computed
        simp at computed
    · intro position byte lower upper found
      have := (List.getElem?_eq_some_iff.mp found).1
      omega

theorem source_predicate_stem_supported (source : Slice U8) (token : SourceToken)
    (computed : strict_memory_frontend_parse.predicate_stem_supported source token=.ok true) :
    PredicateStem source.val token := by
  unfold strict_memory_frontend_parse.predicate_stem_supported at computed
  cases wordEq : token.word with
  | false => simp only [wordEq,Bool.false_eq_true,ite_false] at computed; simp at computed
  | true =>
    simp only [wordEq,ite_true] at computed
    by_cases reversed : token.start>token.end
    · simp only [reversed,ite_true] at computed; simp at computed
    · simp only [reversed,ite_false] at computed
      by_cases outside : token.end>Slice.len source
      · simp only [outside,ite_true] at computed; simp at computed
      · simp only [outside,ite_false] at computed
        have ordered : token.start.val≤token.end.val := by scalar_tac
        have bound : token.end.val≤source.val.length := by scalar_tac
        rw [source_usize_sub token.end token.start ordered] at computed
        simp only [bind_ok] at computed
        split at computed
        · simp at computed
        · rename_i longEnough
          have width : token.start.val+2≤token.end.val := by scalar_tac
          obtain ⟨first,firstRead,computed⟩:=source_bind_success _ _ _ computed
          by_cases wrong : first≠37#u8
          · have test : (first !=37#u8)=true := by scalar_tac
            simp [test] at computed
          · have firstEq : first=37#u8 := by scalar_tac
            subst first
            simp only [bne_self_eq_false,Bool.false_eq_true,ite_false] at computed
            have nextBound : token.start.val+1<2^UScalarTy.Usize.numBits := by
              have := Slice.length_ineq source
              scalar_tac
            let next := Usize.ofNatCore (token.start.val+1) nextBound
            have nextVal : next.val=token.start.val+1 := by simp [next]
            rw [source_usize_increment token.start nextBound] at computed
            simp only [bind_ok] at computed
            obtain ⟨second,secondRead,computed⟩:=source_bind_success _ _ _ computed
            by_cases wrong : second≠112#u8
            · have test : (second !=112#u8)=true := by scalar_tac
              simp [test] at computed
            · have secondEq : second=112#u8 := by scalar_tac
              subst second
              simp only [bne_self_eq_false,Bool.false_eq_true,ite_false] at computed
              have afterBound : token.start.val+2<2^UScalarTy.Usize.numBits := by
                have := Slice.length_ineq source
                scalar_tac
              let after := Usize.ofNatCore (token.start.val+2) afterBound
              have afterVal : after.val=token.start.val+2 := by simp [after]
              rw [source_usize_add token.start 2#usize afterBound] at computed
              simp only [bind_ok] at computed
              have tail := predicate_tail_loop source token.end after bound computed
              refine ⟨wordEq,width,bound,source_slice_index source token.start _ firstRead,?_,?_⟩
              · simpa [nextVal] using source_slice_index source next _ secondRead
              · intro position byte lower upper found
                exact tail position byte (by simpa [afterVal] using lower) upper found

private theorem predicate_names_false (source : Slice U8) (scope : Scope)
    (index : Usize) :
    strict_memory_frontend_parse.scope_predicate_names_supported_loop source scope false index=.ok false := by
  unfold strict_memory_frontend_parse.scope_predicate_names_supported_loop
  rw [loop.eq_def]
  simp [strict_memory_frontend_parse.scope_predicate_names_supported_loop.body,bind_ok]

private theorem predicate_names_loop (source : Slice U8) (scope : Scope)
    (index : Usize)
    (computed : strict_memory_frontend_parse.scope_predicate_names_supported_loop source scope true index=.ok true) :
    ∀ position declaration,index.val≤position → scope.declarations.val[position]?=some declaration →
      declaration.predicate=true → PredicateStem source.val declaration.name := by
  generalize fuelEq : scope.declarations.val.length-index.val=fuel
  induction fuel using Nat.strong_induction_on generalizing index with
  | h fuel ih =>
    have unroll : strict_memory_frontend_parse.scope_predicate_names_supported_loop source scope true index=(do
        let outcome ← strict_memory_frontend_parse.scope_predicate_names_supported_loop.body source scope true index
        match outcome with
        | .cont pair => strict_memory_frontend_parse.scope_predicate_names_supported_loop source scope pair.1 pair.2
        | .done value => .ok value) := by
      unfold strict_memory_frontend_parse.scope_predicate_names_supported_loop
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
      simp only [strict_memory_frontend_parse.scope_predicate_names_supported_loop.body,
        ite_true,test,alloc.vec.Vec.index_slice_index,read,bind_ok,Std.bind_assoc] at computed
      obtain ⟨supported,supportedComputed,computed⟩:=source_bind_success _ _ _ computed
      cases supported with
      | false =>
        simp only [increment,bind_ok] at computed
        change strict_memory_frontend_parse.scope_predicate_names_supported_loop source scope false after=.ok true at computed
        rw [predicate_names_false] at computed
        simp at computed
      | true =>
        simp only [increment,bind_ok] at computed
        change strict_memory_frontend_parse.scope_predicate_names_supported_loop source scope true after=.ok true at computed
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
          exact source_predicate_stem_supported source declaration.name supportedComputed
        · exact tail position selected (by omega) found
    · intro position declaration lower found
      have := (List.getElem?_eq_some_iff.mp found).1
      omega

theorem source_scope_predicate_names_supported (source : Slice U8) (scope : Scope)
    (computed : strict_memory_frontend_parse.scope_predicate_names_supported source scope=.ok true) :
    PredicateNamesSupported source.val scope := by
  unfold strict_memory_frontend_parse.scope_predicate_names_supported at computed
  intro declaration member
  obtain ⟨position,found⟩:=List.mem_iff_getElem?.mp member
  exact predicate_names_loop source scope 0#usize computed position declaration (by simp) found

end PtxFrontendProgram

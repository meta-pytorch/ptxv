/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.TokenHelpers

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

private theorem global_loop_not_unique (source : Slice U8) (scope : Scope) (token : SourceToken)
    (found : Option Usize) (index : Usize) :
    strict_memory_frontend_parse.lookup_global_loop source scope token found false index=.ok (found,false) := by
  unfold strict_memory_frontend_parse.lookup_global_loop
  rw [loop.eq_def]
  simp [strict_memory_frontend_parse.lookup_global_loop.body,bind_ok]

theorem source_global_loop (source : Slice U8) (scope : Scope) (token : SourceToken)
    (found : Option Usize) (index selected : Usize)
    (prior : ∀ value,found=some value → GlobalReference source.val scope token value)
    (computed : strict_memory_frontend_parse.lookup_global_loop source scope token found true index=.ok (some selected,true)) :
    GlobalReference source.val scope token selected := by
  generalize fuelEq : scope.globals.val.length-index.val=fuel
  induction fuel using Nat.strong_induction_on generalizing index found with
  | h fuel ih =>
    have unroll : strict_memory_frontend_parse.lookup_global_loop source scope token found true index = (do
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
        change Aeneas.Std.bind (ok (ControlFlow.cont (found,true,after) : ControlFlow ((Option Usize) × Bool × Usize) ((Option Usize) × Bool))) _ = ok (some selected,true) at computed
        simp only [bind_ok] at computed
        exact ih (scope.globals.val.length-after.val) (by omega) found after prior computed rfl
      | true =>
        have selectedMeaning : GlobalReference source.val scope token index :=
          ⟨global,List.getElem?_eq_getElem inside,source_names_equal source global.name token sameComputed⟩
        cases found with
        | none =>
          simp only [ite_true,core.option.Option.is_none,Option.isNone,bind_ok,increment] at computed
          change Aeneas.Std.bind (ok (ControlFlow.cont (some index,true,after) : ControlFlow ((Option Usize) × Bool × Usize) ((Option Usize) × Bool))) _ = ok (some selected,true) at computed
          simp only [bind_ok] at computed
          apply ih (scope.globals.val.length-after.val) (by omega) (some index) after _ computed rfl
          intro value same
          cases same
          exact selectedMeaning
        | some value =>
          simp only [ite_true,core.option.Option.is_none,Option.isNone,bind_ok,increment] at computed
          change Aeneas.Std.bind (ok (ControlFlow.cont (some index,false,after) : ControlFlow ((Option Usize) × Bool × Usize) ((Option Usize) × Bool))) _ = ok (some selected,true) at computed
          simp only [bind_ok] at computed
          rw [global_loop_not_unique] at computed
          simp at computed
    · have test : ¬index<alloc.vec.Vec.len scope.globals := by scalar_tac
      simp only [strict_memory_frontend_parse.lookup_global_loop.body,ite_true,test,ite_false,bind_ok,ok.injEq,Prod.mk.injEq] at computed
      exact prior selected computed.1

theorem source_lookup_global (source : Slice U8) (scope : Scope) (token : SourceToken) (selected : Usize)
    (computed : strict_memory_frontend_parse.lookup_global source scope token=.ok (some selected)) :
    GlobalReference source.val scope token selected := by
  unfold strict_memory_frontend_parse.lookup_global at computed
  obtain ⟨pair,loopComputed,computed⟩ := source_bind_success _ _ _ computed
  rcases pair with ⟨found,unique⟩
  cases unique with
  | false => simp at computed
  | true =>
    have equal : found=some selected := by simpa using computed
    rw [equal] at loopComputed
    exact source_global_loop source scope token none 0#usize selected (by intro value impossible;cases impossible) loopComputed

end PtxFrontendProgram

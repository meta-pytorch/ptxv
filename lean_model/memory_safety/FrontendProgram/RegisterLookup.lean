/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.RegisterReferences

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

def PositiveCounts (scope : Scope) : Prop := ∀ declaration∈scope.declarations.val,0<declaration.count.val

private theorem register_loop_not_unique (source : Slice U8) (scope : Scope) (token : SourceToken)
    (bits : U8) (predicate : Bool) (found : Option U16) (index : Usize) :
    strict_memory_frontend_parse.lookup_register_loop source scope token bits predicate found false index=.ok (found,false) := by
  unfold strict_memory_frontend_parse.lookup_register_loop
  rw [loop.eq_def]
  simp [strict_memory_frontend_parse.lookup_register_loop.body,bind_ok]

theorem source_register_loop (source : Slice U8) (scope : Scope) (token : SourceToken) (bits : U8) (predicate : Bool)
    (found : Option U16) (index : Usize) (selected : U16) (counts : PositiveCounts scope)
    (prior : ∀ value,found=some value → RegisterReference source.val scope token bits.val predicate value)
    (computed : strict_memory_frontend_parse.lookup_register_loop source scope token bits predicate found true index=.ok (some selected,true)) :
    RegisterReference source.val scope token bits.val predicate selected := by
  generalize fuelEq : scope.declarations.val.length-index.val=fuel
  induction fuel using Nat.strong_induction_on generalizing index found with
  | h fuel ih =>
    have unroll : strict_memory_frontend_parse.lookup_register_loop source scope token bits predicate found true index = (do
        let outcome ← strict_memory_frontend_parse.lookup_register_loop.body source scope token bits predicate found true index
        match outcome with
        | .cont triple => strict_memory_frontend_parse.lookup_register_loop source scope token bits predicate triple.1 triple.2.1 triple.2.2
        | .done value => .ok value) := by
      unfold strict_memory_frontend_parse.lookup_register_loop
      rw [loop.eq_def]
      congr 1
      funext outcome
      cases outcome with
      | cont triple => rcases triple with ⟨_,_,_⟩;rfl
      | done value => rfl
    rw [unroll] at computed
    by_cases inside : index.val<scope.declarations.val.length
    · have test : index<alloc.vec.Vec.len scope.declarations := by scalar_tac
      let declaration := scope.declarations.val[index.val]
      have member : declaration∈scope.declarations.val := List.getElem_mem inside
      have read : alloc.vec.Vec.index_usize scope.declarations index=.ok declaration := by
        simp [alloc.vec.Vec.index_usize,List.getElem?_eq_getElem inside,declaration]
      have incrementBound : index.val+1<2^UScalarTy.Usize.numBits := by
        have := alloc.vec.Vec.len_ineq scope.declarations
        scalar_tac
      let after := Usize.ofNatCore (index.val+1) incrementBound
      have afterVal : after.val=index.val+1 := by simp [after]
      have increment := source_usize_increment index incrementBound
      simp only [strict_memory_frontend_parse.lookup_register_loop.body,ite_true,test,
        alloc.vec.Vec.index_slice_index,read,bind_ok,Std.bind_assoc] at computed
      obtain ⟨decoded,registerComputed,computed⟩ := source_bind_success _ _ _ computed
      cases decoded with
      | none =>
        simp only [bind_ok,increment] at computed
        change Aeneas.Std.bind (ok (ControlFlow.cont (found,true,after) : ControlFlow ((Option U16) × Bool × Usize) ((Option U16) × Bool))) _ = ok (some selected,true) at computed
        simp only [bind_ok] at computed
        exact ih (scope.declarations.val.length-after.val) (by omega) found after prior computed rfl
      | some register =>
        have decodedMeaning := source_declaration_register source declaration token register (counts declaration member) registerComputed
        have failed (failedComputed : strict_memory_frontend_parse.lookup_register_loop source scope token bits predicate (some register) false after=.ok (some selected,true)) : False := by
          rw [register_loop_not_unique] at failedComputed
          simp at failedComputed
        cases found with
        | some previous =>
          simp only [core.option.Option.is_none,Option.isNone,Bool.false_eq_true,ite_false,bind_ok,increment] at computed
          change Aeneas.Std.bind (ok (some register,false)) _ = ok (some selected,true) at computed
          simp only [bind_ok] at computed
          change Aeneas.Std.bind (ok (ControlFlow.cont (some register,false,after) : ControlFlow ((Option U16) × Bool × Usize) ((Option U16) × Bool))) _ = ok (some selected,true) at computed
          simp only [bind_ok] at computed
          exact (failed computed).elim
        | none =>
          simp only [core.option.Option.is_none,Option.isNone,ite_true] at computed
          by_cases matchingBits : declaration.bits=bits
          · simp only [matchingBits,ite_true] at computed
            by_cases matchingPredicate : declaration.predicate=predicate
            · simp only [matchingPredicate,decide_true,bind_ok,increment] at computed
              change Aeneas.Std.bind (ok (some register,true)) _ = ok (some selected,true) at computed
              simp only [bind_ok] at computed
              change Aeneas.Std.bind (ok (ControlFlow.cont (some register,true,after) : ControlFlow ((Option U16) × Bool × Usize) ((Option U16) × Bool))) _ = ok (some selected,true) at computed
              simp only [bind_ok] at computed
              apply ih (scope.declarations.val.length-after.val) (by omega) (some register) after _ computed rfl
              intro value same
              cases same
              exact ⟨declaration,member,by rw [matchingBits],matchingPredicate,decodedMeaning⟩
            · simp only [matchingPredicate,decide_false,bind_ok,increment] at computed
              change Aeneas.Std.bind (ok (some register,false)) _ = ok (some selected,true) at computed
              simp only [bind_ok] at computed
              change Aeneas.Std.bind (ok (ControlFlow.cont (some register,false,after) : ControlFlow ((Option U16) × Bool × Usize) ((Option U16) × Bool))) _ = ok (some selected,true) at computed
              simp only [bind_ok] at computed
              exact (failed computed).elim
          · simp only [matchingBits,ite_false,bind_ok,increment] at computed
            change Aeneas.Std.bind (ok (some register,false)) _ = ok (some selected,true) at computed
            simp only [bind_ok] at computed
            change Aeneas.Std.bind (ok (ControlFlow.cont (some register,false,after) : ControlFlow ((Option U16) × Bool × Usize) ((Option U16) × Bool))) _ = ok (some selected,true) at computed
            simp only [bind_ok] at computed
            exact (failed computed).elim
    · have test : ¬index<alloc.vec.Vec.len scope.declarations := by scalar_tac
      simp only [strict_memory_frontend_parse.lookup_register_loop.body,ite_true,test,ite_false,bind_ok,ok.injEq,Prod.mk.injEq] at computed
      exact prior selected computed.1

theorem source_lookup_register (source : Slice U8) (scope : Scope) (token : SourceToken) (bits : U8) (predicate : Bool) (selected : U16)
    (counts : PositiveCounts scope)
    (computed : strict_memory_frontend_parse.lookup_register source scope token bits predicate=.ok (some selected)) :
    RegisterReference source.val scope token bits.val predicate selected := by
  unfold strict_memory_frontend_parse.lookup_register at computed
  obtain ⟨pair,loopComputed,computed⟩ := source_bind_success _ _ _ computed
  rcases pair with ⟨found,unique⟩
  cases unique with
  | false => simp at computed
  | true =>
    have equal : found=some selected := by simpa using computed
    rw [equal] at loopComputed
    exact source_register_loop source scope token bits predicate none 0#usize selected counts
      (by intro value impossible;cases impossible) loopComputed

end PtxFrontendProgram

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.CodeBodies

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

private theorem code_loop_false (source : Slice U8) (tokens : Slice SourceToken) (scope : Scope)
    (position labelIndex : Usize) (code : alloc.vec.Vec GuardedInstruction) :
    strict_memory_frontend_parse.parse_code_loop source tokens position scope false code labelIndex=
      .ok (position,scope,false,code,labelIndex) := by
  unfold strict_memory_frontend_parse.parse_code_loop
  rw [loop.eq_def]
  simp [strict_memory_frontend_parse.parse_code_loop.body,bind_ok]

theorem source_code_loop (source : Slice U8) (tokens : Slice SourceToken) (scope outScope : Scope)
    (position finish labelIndex outIndex : Usize) (code outCode : alloc.vec.Vec GuardedInstruction)
    (counts : PositiveCounts scope) (labelsBound : labelIndex.val≤scope.labels.val.length)
    (complete : outIndex.val=outScope.labels.val.length) (closing : TokenAt source.val tokens.val finish.val "}")
    (computed : strict_memory_frontend_parse.parse_code_loop source tokens position scope true code labelIndex=
      .ok (finish,outScope,true,outCode,outIndex)) :
    outScope=scope ∧ ∃ instructions,
      CodeSyntax source.val scope tokens.val position.val code.val.length instructions (scope.labels.val.drop labelIndex.val) finish.val ∧
      outCode.val=code.val++instructions := by
  generalize fuelEq : tokens.val.length-position.val=fuel
  induction fuel using Nat.strong_induction_on generalizing position code labelIndex with
  | h fuel ih =>
    have unroll : strict_memory_frontend_parse.parse_code_loop source tokens position scope true code labelIndex=(do
        let outcome ← strict_memory_frontend_parse.parse_code_loop.body source tokens scope position true code labelIndex
        match outcome with
        | .cont next => strict_memory_frontend_parse.parse_code_loop source tokens next.1 scope next.2.1 next.2.2.1 next.2.2.2
        | .done value => .ok value) := by
      unfold strict_memory_frontend_parse.parse_code_loop
      rw [loop.eq_def]
      congr 1
      funext outcome
      cases outcome with
      | cont next => rcases next with ⟨_,_,_,_⟩;rfl
      | done value => rfl
    rw [unroll] at computed
    obtain ⟨outcome,bodyComputed,computed⟩ := source_bind_success _ _ _ computed
    cases outcome with
    | done final =>
      rcases final with ⟨after,state,valid,values,index⟩
      simp only [ok.injEq,Prod.mk.injEq] at computed
      rcases computed with ⟨rfl,rfl,rfl,rfl,rfl⟩
      obtain ⟨positionEq,scopeEq,_,codeEq,labelEq⟩ :=
        source_code_body_done source tokens scope state position after labelIndex index code values true bodyComputed
      rw [scopeEq] at complete
      refine ⟨scopeEq,[],?_,?_⟩
      · rw [←positionEq,←labelEq,complete,List.drop_length]
        exact CodeSyntax.empty closing
      · rw [codeEq]
        simp
    | cont next =>
      rcases next with ⟨after,valid,values,index⟩
      change strict_memory_frontend_parse.parse_code_loop source tokens after scope valid values index=.ok (finish,outScope,true,outCode,outIndex) at computed
      cases valid with
      | false =>
        rw [code_loop_false] at computed
        simp at computed
      | true =>
        have meaning := source_code_body_continue source tokens scope position after labelIndex index code values counts bodyComputed
        rcases meaning with instructionCase|labelCase
        · obtain ⟨instruction,instructionMeaning,appended,labelEq⟩ := instructionCase
          have bounds := instruction_syntax_bounds source.val scope tokens.val position.val after.val instruction instructionMeaning
          have indexBound : index.val≤scope.labels.val.length := by omega
          obtain ⟨scopeEq,instructions,tailMeaning,tailEq⟩ :=
            ih (tokens.val.length-after.val) (by omega) after index values indexBound computed rfl
          refine ⟨scopeEq,instruction::instructions,?_,?_⟩
          · have size : values.val.length=code.val.length+1 := by simp [appended]
            rw [size,labelEq] at tailMeaning
            exact CodeSyntax.instruction instructionMeaning tailMeaning
          · rw [tailEq,appended]
            simp [List.append_assoc]
        · obtain ⟨token,label,found,valid,colon,labelFound,nameEq,targetEq,advance,codeEq,labelEq⟩ := labelCase
          have afterBound : after.val≤tokens.val.length := by
            obtain ⟨colonToken,colonFound,_⟩ := colon
            have := (List.getElem?_eq_some_iff.mp colonFound).1
            omega
          have indexBound : index.val≤scope.labels.val.length := by
            have := (List.getElem?_eq_some_iff.mp labelFound).1
            omega
          obtain ⟨scopeEq,instructions,tailMeaning,tailEq⟩ :=
            ih (tokens.val.length-after.val) (by omega) after index values indexBound computed rfl
          refine ⟨scopeEq,instructions,?_,?_⟩
          · obtain ⟨labelBound,labelValue⟩ := List.getElem?_eq_some_iff.mp labelFound
            have dropEq : scope.labels.val.drop labelIndex.val=label::scope.labels.val.drop index.val := by
              rw [List.drop_eq_getElem_cons labelBound,labelValue,labelEq]
            rw [dropEq]
            have same : (⟨token,label.target⟩ : Label)=label := by
              rcases label with ⟨labelName,labelTarget⟩
              simp only at nameEq
              cases nameEq
              rfl
            rw [codeEq,advance] at tailMeaning
            have result := CodeSyntax.label label.target found valid colon targetEq tailMeaning
            simpa only [same] using result
          · simpa only [codeEq] using tailEq

theorem source_parse_code (source : Slice U8) (tokens : Slice SourceToken) (scope : Scope)
    (position : Usize) (code : alloc.vec.Vec GuardedInstruction) (counts : PositiveCounts scope)
    (computed : strict_memory_frontend_parse.parse_code source tokens position scope=.ok (some code)) :
    ∃ finish,CodeSyntax source.val scope tokens.val position.val 0 code.val scope.labels.val finish ∧
      finish+1=tokens.val.length ∧ code.val≠[] := by
  unfold strict_memory_frontend_parse.parse_code at computed
  obtain ⟨final,loopComputed,computed⟩ := source_bind_success _ _ _ computed
  rcases final with ⟨finish,outScope,valid,outCode,labelIndex⟩
  cases valid with
  | false => simp at computed
  | true =>
    change (if alloc.vec.Vec.len outCode=0#usize then (ok none : Result (Option (alloc.vec.Vec GuardedInstruction))) else _)=_ at computed
    split at computed
    · simp at computed
    · rename_i nonempty
      dsimp only at computed
      split at computed
      · simp at computed
      · rename_i allLabels
        simp only [lift,bind_ok] at computed
        obtain ⟨taken,takenComputed,computed⟩ := source_bind_success _ _ _ computed
        cases taken with
        | none => simp at computed
        | some after =>
          dsimp only at computed
          split at computed
          · simp at computed
          · rename_i allTokens
            have same : outCode=code := by simpa using computed
            subst outCode
            obtain ⟨afterEq,token,found,closeMeaning⟩ := source_take source tokens finish after _ takenComputed
            have closing : TokenAt source.val tokens.val finish.val "}" := by
              have literal : utf8 "}"=[125#u8] := by decide
              exact ⟨token,found,by simpa [TokenSpelling,literal,Array.to_slice,Array.make] using closeMeaning⟩
            have complete : labelIndex.val=outScope.labels.val.length := by scalar_tac
            obtain ⟨scopeFrame,instructions,meaning,codeEq⟩ := source_code_loop source tokens scope outScope position finish 0#usize labelIndex
              (alloc.vec.Vec.new GuardedInstruction) code counts (by simp) complete closing loopComputed
            change code.val=[]++instructions at codeEq
            simp only [List.nil_append] at codeEq
            rw [←codeEq] at meaning
            refine ⟨finish.val,?_,?_,?_⟩
            · simpa [alloc.vec.Vec.new] using meaning
            · scalar_tac
            · intro empty
              apply nonempty
              apply UScalar.eq_of_val_eq
              simp [empty,alloc.vec.Vec.len]

end PtxFrontendProgram

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.CodeSteps

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

def CodeStepMeaning (source : List U8) (tokens : List SourceToken) (scope : Scope)
    (position next labelIndex nextLabel : Nat) (code nextCode : alloc.vec.Vec GuardedInstruction) : Prop :=
  (∃ instruction,InstructionSyntax source scope tokens position next instruction ∧
    nextCode.val=code.val++[instruction] ∧ nextLabel=labelIndex) ∨
  (∃ token label,tokens[position]?=some token ∧ Identifier source token false ∧ TokenAt source tokens (position+1) ":" ∧
    scope.labels.val[labelIndex]?=some label ∧ label.«name»=token ∧ label.target.val=code.val.length ∧
    next=position+2 ∧ nextCode=code ∧ nextLabel=labelIndex+1)

theorem source_code_body_continue (source : Slice U8) (tokens : Slice SourceToken) (scope : Scope)
    (position next labelIndex nextLabel : Usize) (code nextCode : alloc.vec.Vec GuardedInstruction)
    (counts : PositiveCounts scope)
    (computed : strict_memory_frontend_parse.parse_code_loop.body source tokens scope position true code labelIndex=
      .ok (.cont (next,true,nextCode,nextLabel))) :
    CodeStepMeaning source.val tokens.val scope position.val next.val labelIndex.val nextLabel.val code nextCode := by
  unfold strict_memory_frontend_parse.parse_code_loop.body at computed
  simp only [ite_true] at computed
  split at computed
  · simp only [lift,bind_ok] at computed
    obtain ⟨closing,closingComputed,computed⟩ := source_bind_success _ _ _ computed
    cases closing with
    | true => simp at computed
    | false =>
      simp only [Bool.false_eq_true,ite_false] at computed
      obtain ⟨afterOne,oneAdded,computed⟩ := source_bind_success _ _ _ computed
      split at computed
      · obtain ⟨colon,colonComputed,computed⟩ := source_bind_success _ _ _ computed
        cases colon with
        | false =>
          change instructionCodeStep source tokens scope position code labelIndex=.ok (.cont (next,true,nextCode,nextLabel)) at computed
          obtain ⟨instruction,meaning,appended,labelEq⟩ := source_instruction_code_step source tokens scope position next labelIndex nextLabel code nextCode counts computed
          exact Or.inl ⟨instruction,meaning,appended,congrArg UScalar.val labelEq⟩
        | true =>
          change labelCodeStep source tokens scope position code labelIndex=.ok (.cont (next,true,nextCode,nextLabel)) at computed
          obtain ⟨token,label,found,valid,labelFound,nameEq,targetEq,advance,codeEq,labelEq⟩ :=
            source_label_code_step source tokens scope position next labelIndex nextLabel code nextCode computed
          obtain ⟨colonToken,colonFound,colonMeaning⟩ := source_matches source tokens afterOne _ colonComputed
          have oneEq := source_usize_add_value position 1#usize afterOne oneAdded
          have colonAt : TokenAt source.val tokens.val (position.val+1) ":" := by
            have literal : utf8 ":"=[58#u8] := by decide
            refine ⟨colonToken,?_,?_⟩
            · simpa [oneEq] using colonFound
            · simpa [TokenSpelling,literal,Array.to_slice,Array.make] using colonMeaning
          exact Or.inr ⟨token,label,found,valid,colonAt,labelFound,nameEq,targetEq,advance,codeEq,labelEq⟩
      · change instructionCodeStep source tokens scope position code labelIndex=.ok (.cont (next,true,nextCode,nextLabel)) at computed
        obtain ⟨instruction,meaning,appended,labelEq⟩ := source_instruction_code_step source tokens scope position next labelIndex nextLabel code nextCode counts computed
        exact Or.inl ⟨instruction,meaning,appended,congrArg UScalar.val labelEq⟩
  · simp at computed

theorem source_code_body_done (source : Slice U8) (tokens : Slice SourceToken) (scope outScope : Scope)
    (position finish labelIndex outIndex : Usize) (code outCode : alloc.vec.Vec GuardedInstruction) (outValid : Bool)
    (computed : strict_memory_frontend_parse.parse_code_loop.body source tokens scope position true code labelIndex=
      .ok (.done (finish,outScope,outValid,outCode,outIndex))) :
    finish=position ∧ outScope=scope ∧ outValid=true ∧ outCode=code ∧ outIndex=labelIndex := by
  unfold strict_memory_frontend_parse.parse_code_loop.body at computed
  simp only [ite_true] at computed
  split at computed
  · simp only [lift,bind_ok] at computed
    obtain ⟨closing,closingComputed,computed⟩ := source_bind_success _ _ _ computed
    cases closing with
    | true =>
      simp only [ite_true,ok.injEq,ControlFlow.done.injEq,Prod.mk.injEq] at computed
      exact ⟨computed.1.symm,computed.2.1.symm,computed.2.2.1.symm,computed.2.2.2.1.symm,computed.2.2.2.2.symm⟩
    | false =>
      simp only [Bool.false_eq_true,ite_false] at computed
      obtain ⟨afterOne,oneAdded,computed⟩ := source_bind_success _ _ _ computed
      split at computed
      · obtain ⟨colon,colonComputed,computed⟩ := source_bind_success _ _ _ computed
        cases colon with
        | false =>
          change instructionCodeStep source tokens scope position code labelIndex=.ok (.done (finish,outScope,outValid,outCode,outIndex)) at computed
          exact False.elim (instruction_code_step_not_done source tokens scope position labelIndex code _ computed)
        | true =>
          change labelCodeStep source tokens scope position code labelIndex=.ok (.done (finish,outScope,outValid,outCode,outIndex)) at computed
          exact False.elim (label_code_step_not_done source tokens scope position labelIndex code _ computed)
      · change instructionCodeStep source tokens scope position code labelIndex=.ok (.done (finish,outScope,outValid,outCode,outIndex)) at computed
        exact False.elim (instruction_code_step_not_done source tokens scope position labelIndex code _ computed)
  · simp only [ite_false,ok.injEq,ControlFlow.done.injEq,Prod.mk.injEq] at computed
    exact ⟨computed.1.symm,computed.2.1.symm,computed.2.2.1.symm,computed.2.2.2.1.symm,computed.2.2.2.2.symm⟩

end PtxFrontendProgram

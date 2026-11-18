/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.LabelChecks

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

abbrev CodeContinue := Usize × Bool × alloc.vec.Vec GuardedInstruction × Usize
abbrev CodeDone := Usize × Scope × Bool × alloc.vec.Vec GuardedInstruction × Usize

def instructionCodeStep (source : Slice U8) (tokens : Slice SourceToken) (scope : Scope)
    (position : Usize) (code : alloc.vec.Vec GuardedInstruction) (labelIndex : Usize) :
    Result (ControlFlow CodeContinue CodeDone) := do
  let parsed ← strict_memory_frontend_parse.parse_instruction source tokens position scope
  match parsed with
  | none => .ok (.cont (position,false,code,labelIndex))
  | some pair =>
    let (instruction,next) := pair
    do
    let nextCode ← alloc.vec.Vec.push code instruction
    .ok (.cont (next,true,nextCode,labelIndex))

theorem source_instruction_code_step (source : Slice U8) (tokens : Slice SourceToken) (scope : Scope)
    (position next labelIndex nextLabel : Usize) (code nextCode : alloc.vec.Vec GuardedInstruction)
    (counts : PositiveCounts scope)
    (computed : instructionCodeStep source tokens scope position code labelIndex=.ok (.cont (next,true,nextCode,nextLabel))) :
    ∃ instruction,InstructionSyntax source.val scope tokens.val position.val next.val instruction ∧
      nextCode.val=code.val++[instruction] ∧ nextLabel=labelIndex := by
  unfold instructionCodeStep at computed
  obtain ⟨parsed,parsedComputed,computed⟩ := source_bind_success _ _ _ computed
  cases parsed with
  | none => simp at computed
  | some pair =>
    rcases pair with ⟨instruction,after⟩
    change Aeneas.Std.bind (alloc.vec.Vec.push code instruction) _ = _ at computed
    obtain ⟨pushed,pushedComputed,computed⟩ := source_bind_success _ _ _ computed
    simp only [ok.injEq,ControlFlow.cont.injEq,Prod.mk.injEq] at computed
    obtain ⟨positionEq,_,codeEq,labelEq⟩ := computed
    rw [positionEq] at parsedComputed
    refine ⟨instruction,source_parse_instruction source tokens position next scope instruction counts parsedComputed,?_,labelEq.symm⟩
    rw [←codeEq]
    exact vec_push_success code pushed instruction pushedComputed

theorem instruction_code_step_not_done (source : Slice U8) (tokens : Slice SourceToken) (scope : Scope)
    (position labelIndex : Usize) (code : alloc.vec.Vec GuardedInstruction) (done : CodeDone)
    (computed : instructionCodeStep source tokens scope position code labelIndex=.ok (.done done)) : False := by
  unfold instructionCodeStep at computed
  obtain ⟨parsed,parsedComputed,computed⟩ := source_bind_success _ _ _ computed
  cases parsed with
  | none => simp at computed
  | some pair =>
    rcases pair with ⟨instruction,after⟩
    change Aeneas.Std.bind (alloc.vec.Vec.push code instruction) _ = _ at computed
    obtain ⟨pushed,pushedComputed,computed⟩ := source_bind_success _ _ _ computed
    simp at computed

def labelCodeStep (source : Slice U8) (tokens : Slice SourceToken) (scope : Scope)
    (position : Usize) (code : alloc.vec.Vec GuardedInstruction) (labelIndex : Usize) :
    Result (ControlFlow CodeContinue CodeDone) := do
  let token ← Slice.index_usize tokens position
  let valid ← strict_memory_frontend_parse.identifier source token false
  if valid then
    if labelIndex≥alloc.vec.Vec.len scope.labels then .ok (.cont (position,false,code,labelIndex))
    else do
      let label ← alloc.vec.Vec.index_usize scope.labels labelIndex
      let matched ← strict_memory_frontend_parse.label_matches label token (alloc.vec.Vec.len code)
      if matched then do
        let nextLabel ← labelIndex+1#usize
        let next ← position+2#usize
        .ok (.cont (next,true,code,nextLabel))
      else .ok (.cont (position,false,code,labelIndex))
  else .ok (.cont (position,false,code,labelIndex))

theorem source_label_code_step (source : Slice U8) (tokens : Slice SourceToken) (scope : Scope)
    (position next labelIndex nextLabel : Usize) (code nextCode : alloc.vec.Vec GuardedInstruction)
    (computed : labelCodeStep source tokens scope position code labelIndex=.ok (.cont (next,true,nextCode,nextLabel))) :
    ∃ token label,tokens.val[position.val]?=some token ∧ Identifier source.val token false ∧
      scope.labels.val[labelIndex.val]?=some label ∧ label.«name»=token ∧ label.target.val=code.val.length ∧
      next.val=position.val+2 ∧ nextCode=code ∧ nextLabel.val=labelIndex.val+1 := by
  unfold labelCodeStep at computed
  obtain ⟨token,tokenComputed,computed⟩ := source_bind_success _ _ _ computed
  obtain ⟨valid,validComputed,computed⟩ := source_bind_success _ _ _ computed
  cases valid with
  | false => simp at computed
  | true =>
    simp only [ite_true] at computed
    split at computed
    · simp at computed
    · obtain ⟨label,labelComputed,computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨matched,matchedComputed,computed⟩ := source_bind_success _ _ _ computed
      cases matched with
      | false => simp at computed
      | true =>
        simp only [ite_true] at computed
        obtain ⟨afterLabel,labelAdded,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨after,positionAdded,computed⟩ := source_bind_success _ _ _ computed
        simp only [ok.injEq,ControlFlow.cont.injEq,Prod.mk.injEq] at computed
        obtain ⟨positionEq,_,codeEq,labelEq⟩ := computed
        have matchedMeaning := source_label_matches label token (alloc.vec.Vec.len code) matchedComputed
        have added := source_usize_add_value position 2#usize after positionAdded
        have labelAfter := source_usize_add_value labelIndex 1#usize afterLabel labelAdded
        refine ⟨token,label,source_slice_index tokens position token tokenComputed,
          source_identifier source token false validComputed,?_,matchedMeaning.1,?_,?_,codeEq.symm,?_⟩
        · exact vec_index_success scope.labels labelIndex label labelComputed
        · simpa only [alloc.vec.Vec.len_val] using matchedMeaning.2
        · rw [←positionEq]
          simpa using added
        · rw [←labelEq]
          simpa using labelAfter

theorem label_code_step_not_done (source : Slice U8) (tokens : Slice SourceToken) (scope : Scope)
    (position labelIndex : Usize) (code : alloc.vec.Vec GuardedInstruction) (finished : CodeDone)
    (computed : labelCodeStep source tokens scope position code labelIndex=.ok (.done finished)) : False := by
  unfold labelCodeStep at computed
  obtain ⟨token,tokenComputed,computed⟩ := source_bind_success _ _ _ computed
  obtain ⟨valid,validComputed,computed⟩ := source_bind_success _ _ _ computed
  cases valid with
  | false => simp at computed
  | true =>
    simp only [ite_true] at computed
    split at computed
    · simp at computed
    · obtain ⟨label,labelComputed,computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨matched,matchedComputed,computed⟩ := source_bind_success _ _ _ computed
      cases matched with
      | false => simp at computed
      | true =>
        simp only [ite_true] at computed
        obtain ⟨afterLabel,labelAdded,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨after,positionAdded,computed⟩ := source_bind_success _ _ _ computed
        simp at computed

end PtxFrontendProgram

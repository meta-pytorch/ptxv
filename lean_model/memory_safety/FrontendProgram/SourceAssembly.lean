/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.ScopeSymbols
import FrontendProgram.LabelUnique
import FrontendProgram.Declarations
import FrontendProgram.Code
import FrontendProgram.Entry
import FrontendProgram.GrammarGlobalNames

open Aeneas Aeneas.Std Result PtxvMemorySafety
open MemorySafety.Source
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.split false
set_option backward.isDefEq.respectTransparency false
namespace PtxFrontendProgram

private def emptyScope : Scope :=
  ⟨alloc.vec.Vec.new Declaration,alloc.vec.Vec.new Global,alloc.vec.Vec.new Label,0#u16,0#u16⟩

private theorem empty_declaration_invariant (source : List U8) : DeclarationInvariant source emptyScope := by
  simp [DeclarationInvariant,PositiveCounts,DeclarationLayout,bankSize,SeparateDeclarations,emptyScope,alloc.vec.Vec.new]

def GlobalsStageSound : Prop :=
  ∀ (source : Slice U8) (tokens : Slice SourceToken) (position finish : Usize) (scope selected : Scope),
    scope.globals.val=[] → strict_memory_frontend_parse.parse_globals source tokens position scope=.ok (some (selected,finish)) →
    GlobalsSyntax source.val tokens.val position.val selected.globals.val finish.val ∧
    selected.globals.val.Pairwise (fun left right => ¬NameEqual source.val left.«name» right.«name») ∧
    selected.declarations=scope.declarations ∧ selected.labels=scope.labels ∧
    selected.virtual_count=scope.virtual_count ∧ selected.predicate_count=scope.predicate_count

theorem source_parse_source_of_globals (globalsSound : GlobalsStageSound) (source entry : Slice U8) (program : strict_memory_frontend_parse.SourceProgram)
    (computed : strict_memory_frontend_parse.parse_source source entry=.ok (some program)) :
    ModuleMeaning source.val entry.val program := by
  unfold strict_memory_frontend_parse.parse_source at computed
  obtain ⟨lexed,lexComputed,computed⟩ := source_bind_success _ _ _ computed
  cases lexed with
  | none => simp at computed
  | some tokens =>
    obtain ⟨headerResult,headerComputed,computed⟩ := source_bind_success _ _ _ computed
    cases headerResult with
    | none => simp at computed
    | some globalsStart =>
      obtain ⟨globalsResult,globalsComputed,computed⟩ := source_bind_success _ _ _ computed
      cases globalsResult with
      | none => simp at computed
      | some pair =>
        rcases pair with ⟨globalsScope,globalsEnd⟩
        obtain ⟨entryResult,entryComputed,computed⟩ := source_bind_success _ _ _ computed
        cases entryResult with
        | none => simp at computed
        | some pair =>
          rcases pair with ⟨selected,declarationsStart⟩
          obtain ⟨declarationsResult,declarationsComputed,computed⟩ := source_bind_success _ _ _ computed
          cases declarationsResult with
          | none => simp at computed
          | some pair =>
            rcases pair with ⟨declarationsScope,instructionsStart⟩
            obtain ⟨labelsResult,labelsComputed,computed⟩ := source_bind_success _ _ _ computed
            cases labelsResult with
            | none => simp at computed
            | some labels =>
              let finalScope : Scope := {declarationsScope with labels:=labels}
              obtain ⟨supported,supportedComputed,computed⟩ := source_bind_success _ _ _ computed
              have symbolsSeparate : ScopeSymbolsSeparate source.val finalScope selected := by
                cases supported with
                | false => simp at computed
                | true => exact source_scope_symbols_supported source finalScope selected supportedComputed
              have supportedTrue : supported=true := by cases supported <;> simp_all
              simp only [supportedTrue,Bool.not_true,Bool.false_eq_true,ite_false] at computed
              obtain ⟨codeResult,codeComputed,computed⟩ := source_bind_success _ _ _ computed
              cases codeResult with
              | none => simp at computed
              | some code =>
                have programEq : program=⟨selected,finalScope,code⟩ := by simpa [finalScope] using computed.symm
                have lexMeaning := source_lex source tokens lexComputed
                obtain ⟨headerMeaning,headerEnd⟩ := source_header source (alloc.vec.Vec.deref tokens) globalsStart headerComputed
                obtain ⟨globalsMeaning,globalSeparate,globalDeclarations,globalLabels,globalVirtual,globalPredicate⟩ :=
                  globalsSound source (alloc.vec.Vec.deref tokens) globalsStart globalsEnd emptyScope globalsScope (by rfl) globalsComputed
                have globalsInvariant : DeclarationInvariant source.val globalsScope := by
                  have initial := empty_declaration_invariant source.val
                  simpa only [DeclarationInvariant,PositiveCounts,DeclarationLayout,globalDeclarations,globalVirtual,globalPredicate] using initial
                have entryMeaning := source_entry source entry (alloc.vec.Vec.deref tokens) globalsEnd declarationsStart selected entryComputed
                obtain ⟨declarations,declarationsMeaning,declarationsEq,declarationInvariant,declarationGlobals,declarationLabels⟩ :=
                  source_parse_declarations source (alloc.vec.Vec.deref tokens) declarationsStart instructionsStart globalsScope declarationsScope globalsInvariant declarationsComputed
                have finalInvariant : DeclarationInvariant source.val finalScope := declarationInvariant
                obtain ⟨finish,codeMeaning,consumed,nonempty⟩ := source_parse_code source (alloc.vec.Vec.deref tokens) finalScope instructionsStart code finalInvariant.2.1 codeComputed
                have derefEq : (alloc.vec.Vec.deref tokens).val=tokens.val := by simp [alloc.vec.Vec.deref,Slice.from_val]
                rw [derefEq] at headerMeaning globalsMeaning entryMeaning declarationsMeaning codeMeaning consumed
                rw [programEq]
                refine ⟨tokens.val,globalsEnd.val,declarationsStart.val,instructionsStart.val,finish,lexMeaning,headerMeaning,?_,entryMeaning,?_,codeMeaning,consumed,nonempty,finalInvariant.2.2.1,finalInvariant.2.2.2,?_,?_,symbolsSeparate⟩
                · simpa only [finalScope,declarationGlobals,headerEnd] using globalsMeaning
                · have emptyDeclarations : globalsScope.declarations.val=[] := by rw [globalDeclarations];rfl
                  rw [emptyDeclarations,List.nil_append] at declarationsEq
                  simpa only [finalScope,declarationsEq] using declarationsMeaning
                · simpa only [finalScope,declarationGlobals] using globalSeparate
                · exact source_collect_labels_unique source (alloc.vec.Vec.deref tokens) instructionsStart labels labelsComputed


end PtxFrontendProgram

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Grammar

open Aeneas Aeneas.Std PtxvMemorySafety
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false

namespace PtxFrontendProgram

theorem global_syntax_identifier (source : List U8) (tokens : List SourceToken) (start finish : Nat) (global : Global)
    (meaning : GlobalSyntax source tokens start finish global) : Identifier source global.«name» false := by
  cases meaning
  aesop

theorem globals_syntax_identifiers (source : List U8) (tokens : List SourceToken) (start finish : Nat) (globals : List Global)
    (meaning : GlobalsSyntax source tokens start globals finish) :
    ∀ global∈globals,Identifier source global.«name» false := by
  induction meaning with
  | empty position => simp
  | @cons start middle finish global globals head tail ih =>
    intro candidate member
    rcases List.mem_cons.mp member with same|other
    · subst candidate
      exact global_syntax_identifier source tokens start middle global head
    · exact ih candidate other

theorem module_global_names_injective (source entry : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (meaning : ModuleMeaning source entry parsed) :
    ∀ (i j : Nat) (left right : Global),parsed.scope.globals.val[i]?=some left →
      parsed.scope.globals.val[j]?=some right → nameBytes source left.«name»=nameBytes source right.«name» → i=j := by
  obtain ⟨tokens,globalsEnd,declarationsStart,instructionsStart,finish,lexed,header,globalsMeaning,entryMeaning,
    declarationsMeaning,codeMeaning,consumed,nonempty,layout,separate,globalSeparate,labelSeparate,symbolSeparate⟩ := meaning
  have identifiers := globals_syntax_identifiers source tokens 6 globalsEnd parsed.scope.globals.val globalsMeaning
  intro i j left right leftFound rightFound names
  obtain ⟨leftBound,leftEq⟩ := List.getElem?_eq_some_iff.mp leftFound
  obtain ⟨rightBound,rightEq⟩ := List.getElem?_eq_some_iff.mp rightFound
  have leftMember : left∈parsed.scope.globals.val := by simpa only [leftEq] using List.getElem_mem leftBound
  have rightMember : right∈parsed.scope.globals.val := by simpa only [rightEq] using List.getElem_mem rightBound
  have leftValid := identifiers left leftMember
  have rightValid := identifiers right rightMember
  have same : NameEqual source left.«name» right.«name» :=
    ⟨leftValid.2.1,leftValid.2.2.1,rightValid.2.1,rightValid.2.2.1,names⟩
  have reverse : NameEqual source right.«name» left.«name» :=
    ⟨rightValid.2.1,rightValid.2.2.1,leftValid.2.1,leftValid.2.2.1,names.symm⟩
  by_contra different
  rcases Nat.lt_or_gt_of_ne different with before|after
  · have excluded := globalSeparate.rel_getElem_of_lt leftBound rightBound before
    rw [leftEq,rightEq] at excluded
    exact excluded same
  · have excluded := globalSeparate.rel_getElem_of_lt rightBound leftBound after
    rw [rightEq,leftEq] at excluded
    exact excluded reverse

end PtxFrontendProgram

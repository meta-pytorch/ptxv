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

theorem declaration_syntax_properties (source : List U8) (tokens : List SourceToken)
    (start finish : Nat) (declaration : Declaration)
    (meaning : DeclarationSyntax source tokens start finish declaration) :
    Identifier source declaration.«name» true ∧ 0<declaration.count.val ∧ start<finish := by
  cases meaning with
  | single _ _ _ _ _ valid same count _ _ =>
    exact ⟨same ▸ valid,by omega,by omega⟩
  | indexed _ _ _ _ _ valid same _ _ positive count _ _ _ =>
    exact ⟨same ▸ valid,by omega,by omega⟩

theorem declarations_syntax_properties (source : List U8) (tokens : List SourceToken)
    (start finish : Nat) (declarations : List Declaration)
    (meaning : DeclarationsSyntax source tokens start declarations finish) :
    start≤finish ∧ ∀ declaration∈declarations,Identifier source declaration.«name» true ∧ 0<declaration.count.val := by
  induction meaning with
  | empty position => exact ⟨by omega,by simp⟩
  | @cons start middle finish declaration declarations head tail ih =>
    have property := declaration_syntax_properties source tokens start middle declaration head
    refine ⟨by omega,?_⟩
    intro candidate member
    rcases List.mem_cons.mp member with same|other
    · subst candidate
      exact ⟨property.1,property.2.1⟩
    · exact ih.2 candidate other

end PtxFrontendProgram

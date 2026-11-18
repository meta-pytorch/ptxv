/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.SyntaxFacts

open Aeneas Aeneas.Std PtxvMemorySafety
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
namespace PtxFrontendProgram

theorem register_reference_full_unique (source : List U8) (scope : Scope) (token : SourceToken)
    (leftBits rightBits : Nat) (leftPredicate rightPredicate : Bool) (left right : U16)
    (identifiers : ∀ declaration∈scope.declarations.val,Identifier source declaration.name true)
    (separate : SeparateDeclarations source scope.declarations.val)
    (first : RegisterReference source scope token leftBits leftPredicate left)
    (second : RegisterReference source scope token rightBits rightPredicate right) : leftBits=rightBits ∧ leftPredicate=rightPredicate ∧ left=right := by
  obtain ⟨leftDeclaration,leftMember,leftBitsEq,leftPredicateEq,leftNumber,_,leftNamed,leftIndex⟩:=first
  obtain ⟨rightDeclaration,rightMember,rightBitsEq,rightPredicateEq,rightNumber,_,rightNamed,rightIndex⟩:=second
  have sameDeclaration : leftDeclaration=rightDeclaration := by
    obtain ⟨i,leftFound⟩:=List.mem_iff_getElem?.mp leftMember
    obtain ⟨j,rightFound⟩:=List.mem_iff_getElem?.mp rightMember
    obtain ⟨leftBound,leftEq⟩:=List.getElem?_eq_some_iff.mp leftFound
    obtain ⟨rightBound,rightEq⟩:=List.getElem?_eq_some_iff.mp rightFound
    rcases lt_trichotomy i j with before|equal|after
    · have excluded:=separate.rel_getElem_of_lt leftBound rightBound before
      rw [leftEq,rightEq] at excluded
      exact False.elim (separate_reference_names source leftDeclaration rightDeclaration token leftNumber rightNumber
        (identifiers leftDeclaration leftMember) (identifiers rightDeclaration rightMember) leftNamed rightNamed excluded)
    · subst j
      exact Option.some.inj (leftFound.symm.trans rightFound)
    · have excluded:=separate.rel_getElem_of_lt rightBound leftBound after
      rw [rightEq,leftEq] at excluded
      exact False.elim (separate_reference_names source rightDeclaration leftDeclaration token rightNumber leftNumber
        (identifiers rightDeclaration rightMember) (identifiers leftDeclaration leftMember) rightNamed leftNamed excluded)
  subst rightDeclaration
  refine ⟨leftBitsEq.symm.trans rightBitsEq,leftPredicateEq.symm.trans rightPredicateEq,?_⟩
  have numberEq : leftNumber=rightNumber := by
    cases indexed : leftDeclaration.indexed with
    | false =>
      simp only [RegisterName,indexed] at leftNamed rightNamed
      exact leftNamed.2.trans rightNamed.2.symm
    | true =>
      simp only [RegisterName,indexed,ite_true] at leftNamed rightNamed
      obtain ⟨_,_,_,_,_,leftDigits,_⟩:=leftNamed
      obtain ⟨_,_,_,_,_,rightDigits,_⟩:=rightNamed
      exact digits_unique _ _ _ _ _ _ _ leftDigits rightDigits
  apply UScalar.eq_of_val_eq
  omega


theorem module_register_reference_full_unique (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (token : SourceToken) (leftBits rightBits : Nat) (leftPredicate rightPredicate : Bool) (left right : U16)
    (first : RegisterReference source program.scope token leftBits leftPredicate left)
    (second : RegisterReference source program.scope token rightBits rightPredicate right) :
    leftBits=rightBits ∧ leftPredicate=rightPredicate ∧ left=right := by
  obtain ⟨tokens,globalsEnd,declarationsStart,instructionsStart,finish,
    _,_,_,_,declarations,_,_,_,_,separate,_,_,_⟩:=meaning
  have identifiers:=declarations_syntax_properties source tokens declarationsStart instructionsStart
    program.scope.declarations.val declarations
  exact register_reference_full_unique source program.scope token leftBits rightBits leftPredicate rightPredicate left right
    (fun declaration member=>(identifiers.2 declaration member).1) separate first second

theorem module_register_at_full_unique (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (tokens : List SourceToken) (position leftBits rightBits : Nat)
    (leftPredicate rightPredicate : Bool) (left right : U16)
    (first : RegisterAt source program.scope tokens position leftBits leftPredicate left)
    (second : RegisterAt source program.scope tokens position rightBits rightPredicate right) :
    leftBits=rightBits ∧ leftPredicate=rightPredicate ∧ left=right := by
  obtain ⟨leftToken,leftSelected,leftMeaning⟩:=first
  obtain ⟨rightToken,rightSelected,rightMeaning⟩:=second
  have sameToken:=Option.some.inj (leftSelected.symm.trans rightSelected)
  subst rightToken
  exact module_register_reference_full_unique source entry program meaning leftToken
    leftBits rightBits leftPredicate rightPredicate left right leftMeaning rightMeaning

end PtxFrontendProgram

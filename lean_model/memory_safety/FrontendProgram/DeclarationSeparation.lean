/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.DecimalPrefixes

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

def DeclarationSeparate (source : List U8) (left right : Declaration) : Prop :=
  ¬NameEqual source left.«name» right.«name» ∧
  (left.indexed=true → ¬DecimalExtension source left.«name» right.«name») ∧
  (right.indexed=true → ¬DecimalExtension source right.«name» left.«name»)

private theorem source_decimal_prefix_false (source : Slice U8) (shorter longer : SourceToken)
    (shortValid : Identifier source.val shorter true) (longValid : Identifier source.val longer true)
    (computed : strict_memory_frontend_parse.decimal_prefix source shorter longer=.ok false) :
    ¬DecimalExtension source.val shorter longer := by
  intro extension
  have positive := source_decimal_prefix_complete source shorter longer shortValid longValid extension
  rw [computed] at positive
  simp at positive

theorem source_declarations_separate (source : Slice U8) (left right : Declaration)
    (leftValid : Identifier source.val left.«name» true) (rightValid : Identifier source.val right.«name» true)
    (computed : strict_memory_frontend_parse.declarations_separate source left right=.ok true) :
    DeclarationSeparate source.val left right := by
  unfold strict_memory_frontend_parse.declarations_separate at computed
  obtain ⟨same,sameComputed,computed⟩ := source_bind_success _ _ _ computed
  cases same with
  | true => simp at computed
  | false =>
    simp only [Bool.false_eq_true,ite_false] at computed
    have different := source_names_equal_false source left.«name» right.«name» sameComputed
    cases leftIndexed : left.indexed with
    | false =>
      simp only [leftIndexed,Bool.false_eq_true,ite_false] at computed
      cases rightIndexed : right.indexed with
      | false =>
        exact ⟨different,by simp [leftIndexed],by simp [rightIndexed]⟩
      | true =>
        simp only [rightIndexed,ite_true] at computed
        obtain ⟨extended,extendedComputed,computed⟩ := source_bind_success _ _ _ computed
        cases extended with
        | true => simp at computed
        | false =>
          exact ⟨different,by simp [leftIndexed],fun _ => source_decimal_prefix_false source right.«name» left.«name» rightValid leftValid extendedComputed⟩
    | true =>
      simp only [leftIndexed,ite_true] at computed
      obtain ⟨extended,extendedComputed,computed⟩ := source_bind_success _ _ _ computed
      cases extended with
      | true => simp at computed
      | false =>
        simp only [Bool.false_eq_true,ite_false] at computed
        have leftSeparated := source_decimal_prefix_false source left.«name» right.«name» leftValid rightValid extendedComputed
        cases rightIndexed : right.indexed with
        | false => exact ⟨different,fun _ => leftSeparated,by simp [rightIndexed]⟩
        | true =>
          simp only [rightIndexed,ite_true] at computed
          obtain ⟨extended,extendedComputed,computed⟩ := source_bind_success _ _ _ computed
          cases extended with
          | true => simp at computed
          | false => exact ⟨different,fun _ => leftSeparated,fun _ => source_decimal_prefix_false source right.«name» left.«name» rightValid leftValid extendedComputed⟩

end PtxFrontendProgram

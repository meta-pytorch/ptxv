/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.DeclarationCommit
import FrontendProgram.RegisterLookup

open Aeneas Aeneas.Std Result PtxvMemorySafety
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false

namespace PtxFrontendProgram

def DeclarationInvariant (source : List U8) (scope : Scope) : Prop :=
  (∀ declaration∈scope.declarations.val,Identifier source declaration.«name» true) ∧
  PositiveCounts scope ∧ DeclarationLayout scope ∧ SeparateDeclarations source scope.declarations.val

theorem declaration_syntax_bound (source : List U8) (tokens : List SourceToken)
    (start finish : Nat) (declaration : Declaration)
    (meaning : DeclarationSyntax source tokens start finish declaration) : finish≤tokens.length := by
  cases meaning with
  | single _ _ _ _ _ _ _ _ _ semicolon =>
    obtain ⟨token,found,_⟩ := semicolon
    have := (List.getElem?_eq_some_iff.mp found).1
    omega
  | indexed _ _ _ _ _ _ _ _ _ _ _ _ _ semicolon =>
    obtain ⟨token,found,_⟩ := semicolon
    have := (List.getElem?_eq_some_iff.mp found).1
    omega

theorem bank_size_append (left right : List Declaration) (predicate : Bool) :
    bankSize (left++right) predicate=bankSize left predicate+bankSize right predicate := by
  simp [bankSize,List.filter_append,List.map_append,List.sum_append]

theorem bank_size_single (declaration : Declaration) (predicate : Bool) :
    bankSize [declaration] predicate=(if declaration.predicate=predicate then declaration.count.val else 0) := by
  by_cases same : declaration.predicate=predicate <;> simp [bankSize,same]

theorem declaration_invariant_append (source : List U8) (scope next : Scope) (declaration : Declaration)
    (prior : DeclarationInvariant source scope)
    (valid : Identifier source declaration.«name» true) (positive : 0<declaration.count.val)
    (first : declaration.first=(if declaration.predicate then scope.predicate_count else scope.virtual_count))
    (fresh : ∀ old∈scope.declarations.val,DeclarationSeparate source old declaration)
    (declarations : next.declarations.val=scope.declarations.val++[declaration])
    (virtualCount : next.virtual_count.val=scope.virtual_count.val+(if declaration.predicate then 0 else declaration.count.val))
    (predicateCount : next.predicate_count.val=scope.predicate_count.val+(if declaration.predicate then declaration.count.val else 0)) :
    DeclarationInvariant source next := by
  rcases prior with ⟨names,counts,⟨virtualLayout,predicateLayout,positions⟩,separate⟩
  refine ⟨?_,?_,⟨?_,?_,?_⟩,?_⟩
  · intro selected member
    rw [declarations] at member
    rcases List.mem_append.mp member with old|new
    · exact names selected old
    · have same : selected=declaration := by simpa using new
      exact same ▸ valid
  · intro selected member
    rw [declarations] at member
    rcases List.mem_append.mp member with old|new
    · exact counts selected old
    · have same : selected=declaration := by simpa using new
      exact same ▸ positive
  · rw [declarations,bank_size_append,bank_size_single,virtualCount,virtualLayout]
    cases declaration.predicate <;> simp
  · rw [declarations,bank_size_append,bank_size_single,predicateCount,predicateLayout]
  · intro position selected found
    rw [declarations] at found ⊢
    by_cases inside : position<scope.declarations.val.length
    · rw [List.getElem?_append_left inside] at found
      rw [positions position selected found]
      congr 1
      rw [List.take_append_of_le_length (by omega)]
    · rw [List.getElem?_append_right (by omega)] at found
      have offset : position-scope.declarations.val.length=0 := by
        have := (List.getElem?_eq_some_iff.mp found).1
        simp only [List.length_singleton] at this
        omega
      have equal : selected=declaration := by simpa [offset] using found.symm
      have indexEq : position=scope.declarations.val.length := by omega
      subst selected
      rw [indexEq,List.take_append_of_le_length (by omega),List.take_length]
      rw [first]
      cases declaration.predicate <;> simp_all
  · unfold SeparateDeclarations at *
    rw [declarations,List.pairwise_append]
    refine ⟨separate,by simp,?_⟩
    intro left leftMember right rightMember
    have same : right=declaration := by simpa using rightMember
    subst right
    exact fresh left leftMember

end PtxFrontendProgram

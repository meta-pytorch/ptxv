/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.DeclarationSeparation

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

private theorem declaration_loop_false (source : Slice U8) (declarations : alloc.vec.Vec Declaration)
    (candidate : Declaration) (index : Usize) :
    strict_memory_frontend_parse.parse_declaration_loop0 source declarations candidate.bits candidate.predicate
      candidate.«name» candidate.count candidate.indexed candidate.first index false=.ok false := by
  unfold strict_memory_frontend_parse.parse_declaration_loop0
  rw [loop.eq_def]
  simp [strict_memory_frontend_parse.parse_declaration_loop0.body,bind_ok]

theorem source_declaration_fresh_loop (source : Slice U8) (declarations : alloc.vec.Vec Declaration)
    (candidate : Declaration) (index : Usize)
    (valid : ∀ declaration∈declarations.val,Identifier source.val declaration.«name» true)
    (candidateValid : Identifier source.val candidate.«name» true)
    (computed : strict_memory_frontend_parse.parse_declaration_loop0 source declarations candidate.bits candidate.predicate
      candidate.«name» candidate.count candidate.indexed candidate.first index true=.ok true) :
    ∀ position declaration,index.val≤position → declarations.val[position]?=some declaration →
      DeclarationSeparate source.val declaration candidate := by
  generalize fuelEq : declarations.val.length-index.val=fuel
  induction fuel using Nat.strong_induction_on generalizing index with
  | h fuel ih =>
    have unroll : strict_memory_frontend_parse.parse_declaration_loop0 source declarations candidate.bits candidate.predicate
        candidate.«name» candidate.count candidate.indexed candidate.first index true = (do
        let outcome ← strict_memory_frontend_parse.parse_declaration_loop0.body source declarations candidate.bits candidate.predicate
          candidate.«name» candidate.count candidate.indexed candidate.first index true
        match outcome with
        | .cont pair => strict_memory_frontend_parse.parse_declaration_loop0 source declarations candidate.bits candidate.predicate candidate.«name» candidate.count candidate.indexed candidate.first pair.1 pair.2
        | .done value => .ok value) := by
      unfold strict_memory_frontend_parse.parse_declaration_loop0
      rw [loop.eq_def]
      congr 1
      funext outcome
      cases outcome with
      | cont pair => cases pair;rfl
      | done value => rfl
    rw [unroll] at computed
    by_cases inside : index.val<declarations.val.length
    · have test : index<alloc.vec.Vec.len declarations := by scalar_tac
      let declaration := declarations.val[index.val]
      have read : alloc.vec.Vec.index_usize declarations index=.ok declaration := by
        simp [alloc.vec.Vec.index_usize,List.getElem?_eq_getElem inside,declaration]
      have incrementBound : index.val+1<2^UScalarTy.Usize.numBits := by
        have := alloc.vec.Vec.len_ineq declarations
        scalar_tac
      let after := Usize.ofNatCore (index.val+1) incrementBound
      have afterVal : after.val=index.val+1 := by simp [after]
      have increment := source_usize_increment index incrementBound
      simp only [strict_memory_frontend_parse.parse_declaration_loop0.body,ite_true,test,
        alloc.vec.Vec.index_slice_index,read,bind_ok,Std.bind_assoc] at computed
      obtain ⟨fresh,separateComputed,computed⟩ := source_bind_success _ _ _ computed
      cases fresh with
      | false =>
        simp only [increment,bind_ok] at computed
        rw [declaration_loop_false] at computed
        simp at computed
      | true =>
        have separated := source_declarations_separate source declaration candidate
          (valid declaration (List.getElem_mem inside)) candidateValid separateComputed
        simp only [increment,bind_ok] at computed
        have tail := ih (declarations.val.length-after.val) (by omega) after computed rfl
        intro position selected lower found
        by_cases same : position=index.val
        · subst position
          have equal : selected=declaration := by
            have actual : declarations.val[index.val]?=some declaration := List.getElem?_eq_getElem inside
            rw [found] at actual
            exact Option.some.inj actual
          exact equal ▸ separated
        · exact tail position selected (by omega) found
    · intro position declaration lower found
      have bound := (List.getElem?_eq_some_iff.mp found).1
      omega

theorem source_declaration_fresh_single (source : Slice U8) (declarations : alloc.vec.Vec Declaration)
    (candidate : Declaration) (index : Usize) (count : candidate.count=1#u16)
    (valid : ∀ declaration∈declarations.val,Identifier source.val declaration.«name» true)
    (candidateValid : Identifier source.val candidate.«name» true)
    (computed : strict_memory_frontend_parse.parse_declaration_loop1 source declarations candidate.bits candidate.predicate
      candidate.«name» candidate.indexed candidate.first index true=.ok true) :
    ∀ position declaration,index.val≤position → declarations.val[position]?=some declaration →
      DeclarationSeparate source.val declaration candidate := by
  apply source_declaration_fresh_loop source declarations candidate index valid candidateValid
  rw [count]
  exact computed

end PtxFrontendProgram

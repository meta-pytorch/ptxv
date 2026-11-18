/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.TokenComplete
import FrontendProgram.RegisterTypes

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.split false

namespace PtxFrontendProgram

theorem source_register_name_supported (source : Slice U8) (token : SourceToken)
    (computed : strict_memory_frontend_parse.register_name_supported source token=.ok true) :
    ∀ text ∈ ["%tid", "%ctaid", "%ntid", "%nctaid", "%laneid"],
      ¬TokenSpelling source.val token text := by
  unfold strict_memory_frontend_parse.register_name_supported at computed
  repeat' first
    | fail_if_no_progress simp only [lift,bind_ok,ite_true,Bool.false_eq_true,ite_false] at computed
    | peel_frontend_bool computed
  all_goals simp only [ok.injEq] at computed
  all_goals try contradiction
  all_goals
    intro text listed
    simp only [List.mem_cons,List.not_mem_nil,or_false] at listed
    rcases listed with rfl | rfl | rfl | rfl | rfl
    · have literal : utf8 "%tid"=[37#u8, 116#u8, 105#u8, 100#u8] := by decide
      have decoded : strict_memory_frontend.token_equals source token (Array.to_slice (Array.make 4#usize [37#u8, 116#u8, 105#u8, 100#u8] (by decide)))=.ok false := by assumption
      have meaning := source_token_equals_false source _ token decoded
      simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    · have literal : utf8 "%ctaid"=[37#u8, 99#u8, 116#u8, 97#u8, 105#u8, 100#u8] := by decide
      have decoded : strict_memory_frontend.token_equals source token (Array.to_slice (Array.make 6#usize [37#u8, 99#u8, 116#u8, 97#u8, 105#u8, 100#u8] (by decide)))=.ok false := by assumption
      have meaning := source_token_equals_false source _ token decoded
      simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    · have literal : utf8 "%ntid"=[37#u8, 110#u8, 116#u8, 105#u8, 100#u8] := by decide
      have decoded : strict_memory_frontend.token_equals source token (Array.to_slice (Array.make 5#usize [37#u8, 110#u8, 116#u8, 105#u8, 100#u8] (by decide)))=.ok false := by assumption
      have meaning := source_token_equals_false source _ token decoded
      simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    · have literal : utf8 "%nctaid"=[37#u8, 110#u8, 99#u8, 116#u8, 97#u8, 105#u8, 100#u8] := by decide
      have decoded : strict_memory_frontend.token_equals source token (Array.to_slice (Array.make 7#usize [37#u8, 110#u8, 99#u8, 116#u8, 97#u8, 105#u8, 100#u8] (by decide)))=.ok false := by assumption
      have meaning := source_token_equals_false source _ token decoded
      simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    · have literal : utf8 "%laneid"=[37#u8, 108#u8, 97#u8, 110#u8, 101#u8, 105#u8, 100#u8] := by decide
      have decoded : strict_memory_frontend.token_equals source token (Array.to_slice (Array.make 7#usize [37#u8, 108#u8, 97#u8, 110#u8, 101#u8, 105#u8, 100#u8] (by decide)))=.ok false := by assumption
      have meaning := source_token_equals_false source _ token decoded
      simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning

private theorem register_names_false (source : Slice U8) (scope : Scope)
    (index : Usize) :
    strict_memory_frontend_parse.scope_register_names_supported_loop source scope false index=.ok false := by
  unfold strict_memory_frontend_parse.scope_register_names_supported_loop
  rw [loop.eq_def]
  simp [strict_memory_frontend_parse.scope_register_names_supported_loop.body,bind_ok]

private theorem register_names_loop (source : Slice U8) (scope : Scope)
    (index : Usize)
    (computed : strict_memory_frontend_parse.scope_register_names_supported_loop source scope true index=.ok true) :
    ∀ position declaration,index.val≤position → scope.declarations.val[position]?=some declaration →
      ∀ text ∈ ["%tid", "%ctaid", "%ntid", "%nctaid", "%laneid"],
      ¬TokenSpelling source.val declaration.name text := by
  generalize fuelEq : scope.declarations.val.length-index.val=fuel
  induction fuel using Nat.strong_induction_on generalizing index with
  | h fuel ih =>
    have unroll : strict_memory_frontend_parse.scope_register_names_supported_loop source scope true index=(do
        let outcome ← strict_memory_frontend_parse.scope_register_names_supported_loop.body source scope true index
        match outcome with
        | .cont pair => strict_memory_frontend_parse.scope_register_names_supported_loop source scope pair.1 pair.2
        | .done value => .ok value) := by
      unfold strict_memory_frontend_parse.scope_register_names_supported_loop
      rw [loop.eq_def]
      congr 1
      funext outcome
      cases outcome with
      | cont pair => cases pair;rfl
      | done value => rfl
    rw [unroll] at computed
    by_cases inside : index.val<scope.declarations.val.length
    · have test : index<alloc.vec.Vec.len scope.declarations := by scalar_tac
      let declaration := scope.declarations.val[index.val]
      have read : alloc.vec.Vec.index_usize scope.declarations index=.ok declaration := by
        simp [alloc.vec.Vec.index_usize,List.getElem?_eq_getElem inside,declaration]; rfl
      have incrementBound : index.val+1<2^UScalarTy.Usize.numBits := by
        have := alloc.vec.Vec.len_ineq scope.declarations
        scalar_tac
      let after := Usize.ofNatCore (index.val+1) incrementBound
      have afterVal : after.val=index.val+1 := by simp [after]
      have increment := source_usize_increment index incrementBound
      simp only [strict_memory_frontend_parse.scope_register_names_supported_loop.body,
        ite_true,test,alloc.vec.Vec.index_slice_index,read,bind_ok,Std.bind_assoc] at computed
      obtain ⟨supported,supportedComputed,computed⟩:=source_bind_success _ _ _ computed
      cases supported with
      | false =>
        simp only [increment,bind_ok] at computed
        change strict_memory_frontend_parse.scope_register_names_supported_loop source scope false after=.ok true at computed
        rw [register_names_false] at computed
        simp at computed
      | true =>
        simp only [increment,bind_ok] at computed
        change strict_memory_frontend_parse.scope_register_names_supported_loop source scope true after=.ok true at computed
        have tail:=ih (scope.declarations.val.length-after.val) (by omega) after computed rfl
        intro position selected lower found
        by_cases samePosition : position=index.val
        · subst position
          have selectedEq : selected=declaration := by
            have actual : scope.declarations.val[index.val]?=some declaration:=List.getElem?_eq_getElem inside
            rw [found] at actual
            exact Option.some.inj actual
          subst selected
          exact source_register_name_supported source declaration.name supportedComputed
        · exact tail position selected (by omega) found
    · intro position declaration lower found
      have := (List.getElem?_eq_some_iff.mp found).1
      omega

theorem source_scope_register_names_supported (source : Slice U8) (scope : Scope)
    (computed : strict_memory_frontend_parse.scope_register_names_supported source scope=.ok true) :
    RegisterNamesSeparate source.val scope := by
  unfold strict_memory_frontend_parse.scope_register_names_supported at computed
  intro declaration member
  obtain ⟨position,found⟩:=List.mem_iff_getElem?.mp member
  exact register_names_loop source scope 0#usize computed position declaration (by simp) found


end PtxFrontendProgram

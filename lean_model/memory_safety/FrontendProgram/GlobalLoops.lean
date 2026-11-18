/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Globals
import FrontendProgram.GlobalFreshness

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

theorem initializer_syntax_progress (source : List U8) (tokens : List SourceToken)
    (width start finish : Nat) (values : List Nat)
    (meaning : InitializerSyntax source tokens width start values finish) : start < finish := by
  induction meaning with
  | single => omega
  | cons _ _ _ _ ih => omega

theorem global_syntax_bounds (source : List U8) (tokens : List SourceToken)
    (start finish : Nat) (global : Global)
    (meaning : GlobalSyntax source tokens start finish global) : start < finish ∧ finish ≤ tokens.length := by
  cases meaning with
  | global _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ init _ _ semicolon final _ _ _ =>
    have progress := initializer_syntax_progress _ _ _ _ _ _ init
    obtain ⟨token,found,_⟩ := semicolon
    have bound := (List.getElem?_eq_some_iff.mp found).1
    omega

private theorem globals_loop_false (source : Slice U8) (tokens : Slice SourceToken)
    (position : Usize) (scope : Scope) :
    strict_memory_frontend_parse.parse_globals_loop source tokens position scope false=.ok (position,scope,false) := by
  unfold strict_memory_frontend_parse.parse_globals_loop
  rw [loop.eq_def]
  simp [strict_memory_frontend_parse.parse_globals_loop.body,bind_ok]

theorem source_globals_loop (source : Slice U8) (tokens : Slice SourceToken)
    (position finish : Usize) (scope selected : Scope)
    (prior : scope.globals.val.Pairwise (fun l r => ¬NameEqual source.val l.name r.name))
    (computed : strict_memory_frontend_parse.parse_globals_loop source tokens position scope true=.ok (finish,selected,true)) :
    ∃ globals, GlobalsSyntax source.val tokens.val position.val globals finish.val ∧
      selected.globals.val=scope.globals.val++globals ∧
      selected.globals.val.Pairwise (fun l r => ¬NameEqual source.val l.name r.name) ∧
      selected.declarations=scope.declarations ∧ selected.labels=scope.labels ∧
      selected.virtual_count=scope.virtual_count ∧ selected.predicate_count=scope.predicate_count := by
  generalize fuelEq : tokens.val.length-position.val=fuel
  induction fuel using Nat.strong_induction_on generalizing position scope with
  | h fuel ih =>
    have unroll : strict_memory_frontend_parse.parse_globals_loop source tokens position scope true=(do
        let outcome ← strict_memory_frontend_parse.parse_globals_loop.body source tokens position scope true
        match outcome with
        | .cont triple => strict_memory_frontend_parse.parse_globals_loop source tokens triple.1 triple.2.1 triple.2.2
        | .done value => .ok value) := by
      unfold strict_memory_frontend_parse.parse_globals_loop
      rw [loop.eq_def]
      congr 1
      funext outcome
      cases outcome with
      | cont triple => rcases triple with ⟨_,_,_⟩;rfl
      | done value => rfl
    rw [unroll] at computed
    simp only [strict_memory_frontend_parse.parse_globals_loop.body,ite_true,lift,bind_ok,Std.bind_assoc] at computed
    obtain ⟨matched,matchesComputed,computed⟩ := source_bind_success _ _ _ computed
    cases matched with
    | false =>
      simp only [Bool.false_eq_true,ite_false,bind_ok,ok.injEq,Prod.mk.injEq] at computed
      rcases computed with ⟨rfl,rfl,_⟩
      exact ⟨[],GlobalsSyntax.empty _,by simp,prior,rfl,rfl,rfl,rfl⟩
    | true =>
      simp only [ite_true,Std.bind_assoc] at computed
      obtain ⟨parsed,parsedComputed,computed⟩ := source_bind_success _ _ _ computed
      cases parsed with
      | none =>
        simp only [bind_ok] at computed
        rw [globals_loop_false] at computed
        simp at computed
      | some pair =>
        rcases pair with ⟨global,next⟩
        obtain ⟨globalMeaning,lookupNone⟩ := source_parse_global source tokens position next scope global parsedComputed
        have fresh := source_lookup_global_none source scope global.name prior lookupNone
        change Aeneas.Std.bind (Aeneas.Std.bind (alloc.vec.Vec.push scope.globals global) _) _ = _ at computed
        rw [Std.bind_assoc] at computed
        obtain ⟨globals,pushed,computed⟩ := source_bind_success _ _ _ computed
        let after : Scope := {scope with globals:=globals}
        have appended := vec_push_success scope.globals globals global pushed
        have separated : after.globals.val.Pairwise (fun l r => ¬NameEqual source.val l.name r.name) := by
          change globals.val.Pairwise _
          rw [appended,List.pairwise_append]
          refine ⟨prior,by simp,?_⟩
          intro old oldIn new newIn
          have equal : new=global := by simpa only [List.mem_singleton] using newIn
          subst new
          exact fresh old oldIn
        simp only [bind_ok] at computed
        change strict_memory_frontend_parse.parse_globals_loop source tokens next after true=.ok (finish,selected,true) at computed
        have bounds := global_syntax_bounds source.val tokens.val position.val next.val global globalMeaning
        obtain ⟨tail,tailMeaning,tailEq,tailSeparated,declarationsEq,labelsEq,virtualEq,predicateEq⟩ :=
          ih (tokens.val.length-next.val) (by omega) next after separated computed rfl
        refine ⟨global::tail,GlobalsSyntax.cons globalMeaning tailMeaning,?_,tailSeparated,
          declarationsEq,labelsEq,virtualEq,predicateEq⟩
        rw [tailEq]
        change globals.val++tail=scope.globals.val++global::tail
        rw [appended]
        simp [List.append_assoc]

theorem source_parse_globals (source : Slice U8) (tokens : Slice SourceToken)
    (position finish : Usize) (scope selected : Scope) (empty : scope.globals.val=[])
    (computed : strict_memory_frontend_parse.parse_globals source tokens position scope=.ok (some (selected,finish))) :
    GlobalsSyntax source.val tokens.val position.val selected.globals.val finish.val ∧
      selected.globals.val.Pairwise (fun l r => ¬NameEqual source.val l.name r.name) ∧
      selected.declarations=scope.declarations ∧ selected.labels=scope.labels ∧
      selected.virtual_count=scope.virtual_count ∧ selected.predicate_count=scope.predicate_count := by
  unfold strict_memory_frontend_parse.parse_globals at computed
  obtain ⟨triple,loopComputed,computed⟩ := source_bind_success _ _ _ computed
  rcases triple with ⟨after,state,valid⟩
  cases valid with
  | false => simp at computed
  | true =>
    change ok (some (state,after))=ok (some (selected,finish)) at computed
    simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
    rcases computed with ⟨rfl,rfl⟩
    obtain ⟨globals,grammar,equal,separate,frames⟩ := source_globals_loop source tokens position after scope state
      (by rw [empty]; exact List.Pairwise.nil) loopComputed
    have selectedEq : state.globals.val=globals := by simpa only [empty,List.nil_append] using equal
    exact ⟨by simpa only [selectedEq] using grammar,separate,frames⟩

end PtxFrontendProgram

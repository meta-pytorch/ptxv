/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.DeclarationInvariant

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

private theorem declarations_loop_false (source : Slice U8) (tokens : Slice SourceToken) (position : Usize) (scope : Scope) :
    strict_memory_frontend_parse.parse_declarations_loop source tokens position scope false=.ok (position,scope,false) := by
  unfold strict_memory_frontend_parse.parse_declarations_loop
  rw [loop.eq_def]
  simp [strict_memory_frontend_parse.parse_declarations_loop.body,bind_ok]

theorem source_declarations_loop (source : Slice U8) (tokens : Slice SourceToken)
    (position finish : Usize) (scope selected : Scope)
    (prior : DeclarationInvariant source.val scope)
    (computed : strict_memory_frontend_parse.parse_declarations_loop source tokens position scope true=.ok (finish,selected,true)) :
    ∃ declarations,DeclarationsSyntax source.val tokens.val position.val declarations finish.val ∧
      selected.declarations.val=scope.declarations.val++declarations ∧ DeclarationInvariant source.val selected ∧
      selected.globals=scope.globals ∧ selected.labels=scope.labels := by
  generalize fuelEq : tokens.val.length-position.val=fuel
  induction fuel using Nat.strong_induction_on generalizing position scope with
  | h fuel ih =>
    have unroll : strict_memory_frontend_parse.parse_declarations_loop source tokens position scope true=(do
        let outcome ← strict_memory_frontend_parse.parse_declarations_loop.body source tokens position scope true
        match outcome with
        | .cont triple => strict_memory_frontend_parse.parse_declarations_loop source tokens triple.1 triple.2.1 triple.2.2
        | .done value => .ok value) := by
      unfold strict_memory_frontend_parse.parse_declarations_loop
      rw [loop.eq_def]
      congr 1
      funext outcome
      cases outcome with
      | cont triple => rcases triple with ⟨_,_,_⟩;rfl
      | done value => rfl
    rw [unroll] at computed
    simp only [strict_memory_frontend_parse.parse_declarations_loop.body,ite_true,lift,bind_ok,Std.bind_assoc] at computed
    obtain ⟨matched,matchesComputed,computed⟩ := source_bind_success _ _ _ computed
    cases matched with
    | false =>
      simp only [Bool.false_eq_true,ite_false,bind_ok,ok.injEq,Prod.mk.injEq] at computed
      rcases computed with ⟨rfl,rfl,_⟩
      exact ⟨[],DeclarationsSyntax.empty _,by simp,prior,rfl,rfl⟩
    | true =>
      simp only [ite_true,Std.bind_assoc] at computed
      obtain ⟨parsed,parsedComputed,computed⟩ := source_bind_success _ _ _ computed
      cases parsed with
      | none =>
        simp only [bind_ok] at computed
        rw [declarations_loop_false] at computed
        simp at computed
      | some pair =>
        rcases pair with ⟨declaration,next⟩
        obtain ⟨declarationSyntax,positive,first,total,fresh⟩ := source_parse_declaration source tokens position next scope declaration prior.1 parsedComputed
        have valid := (declaration_syntax_properties _ _ _ _ _ declarationSyntax).1
        have advance := (declaration_syntax_properties _ _ _ _ _ declarationSyntax).2.2
        change Aeneas.Std.bind (Aeneas.Std.bind
          (if declaration.predicate then do
            let total ← scope.predicate_count+declaration.count
            ok (scope.virtual_count,total)
           else do
            let total ← scope.virtual_count+declaration.count
            ok (total,scope.predicate_count)) _) _ = _ at computed
        rw [Std.bind_assoc] at computed
        obtain ⟨counts,countsComputed,computed⟩ := source_bind_success _ _ _ computed
        rcases counts with ⟨virtualCount,predicateCount⟩
        have countsValue : virtualCount.val=scope.virtual_count.val+(if declaration.predicate then 0 else declaration.count.val) ∧
            predicateCount.val=scope.predicate_count.val+(if declaration.predicate then declaration.count.val else 0) := by
          cases predEq : declaration.predicate with
          | false =>
            simp only [predEq,Bool.false_eq_true,ite_false] at countsComputed
            obtain ⟨sum,sumComputed,countsComputed⟩ := source_bind_success _ _ _ countsComputed
            simp only [ok.injEq,Prod.mk.injEq] at countsComputed
            have sumEq : sum.val=scope.virtual_count.val+declaration.count.val := by
              have law := UScalar.add_equiv scope.virtual_count declaration.count
              rw [sumComputed] at law
              exact law.2.1
            rw [←countsComputed.1,←countsComputed.2]
            simp only [predEq,Bool.false_eq_true,ite_false,Nat.add_zero]
            exact ⟨sumEq,True.intro⟩
          | true =>
            simp only [predEq,ite_true] at countsComputed
            obtain ⟨sum,sumComputed,countsComputed⟩ := source_bind_success _ _ _ countsComputed
            simp only [ok.injEq,Prod.mk.injEq] at countsComputed
            have sumEq : sum.val=scope.predicate_count.val+declaration.count.val := by
              have law := UScalar.add_equiv scope.predicate_count declaration.count
              rw [sumComputed] at law
              exact law.2.1
            rw [←countsComputed.1,←countsComputed.2]
            simp only [predEq,ite_true,Nat.add_zero]
            exact ⟨True.intro,sumEq⟩
        change Aeneas.Std.bind (Aeneas.Std.bind (alloc.vec.Vec.push scope.declarations declaration) _) _ = _ at computed
        rw [Std.bind_assoc] at computed
        obtain ⟨declarations,pushed,computed⟩ := source_bind_success _ _ _ computed
        let after : Scope := {scope with declarations:=declarations,virtual_count:=virtualCount,predicate_count:=predicateCount}
        have appended := vec_push_success scope.declarations declarations declaration pushed
        have invariant : DeclarationInvariant source.val after :=
          declaration_invariant_append source.val scope after declaration prior valid positive first fresh appended countsValue.1 countsValue.2
        simp only [bind_ok] at computed
        change strict_memory_frontend_parse.parse_declarations_loop source tokens next after true=.ok (finish,selected,true) at computed
        have nextBound := declaration_syntax_bound source.val tokens.val position.val next.val declaration declarationSyntax
        obtain ⟨tail,tailSyntax,tailEq,tailInvariant,globalsEq,labelsEq⟩ :=
          ih (tokens.val.length-next.val) (by omega) next after invariant computed rfl
        refine ⟨declaration::tail,DeclarationsSyntax.cons declarationSyntax tailSyntax,?_,tailInvariant,globalsEq,labelsEq⟩
        rw [tailEq]
        change declarations.val++tail=scope.declarations.val++declaration::tail
        rw [appended]
        simp [List.append_assoc]

theorem source_parse_declarations (source : Slice U8) (tokens : Slice SourceToken)
    (position finish : Usize) (scope selected : Scope) (prior : DeclarationInvariant source.val scope)
    (computed : strict_memory_frontend_parse.parse_declarations source tokens position scope=.ok (some (selected,finish))) :
    ∃ declarations,DeclarationsSyntax source.val tokens.val position.val declarations finish.val ∧
      selected.declarations.val=scope.declarations.val++declarations ∧ DeclarationInvariant source.val selected ∧
      selected.globals=scope.globals ∧ selected.labels=scope.labels := by
  unfold strict_memory_frontend_parse.parse_declarations at computed
  obtain ⟨triple,loopComputed,computed⟩ := source_bind_success _ _ _ computed
  rcases triple with ⟨after,state,valid⟩
  cases valid with
  | false => simp at computed
  | true =>
    change ok (some (state,after))=ok (some (selected,finish)) at computed
    simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
    rcases computed with ⟨rfl,rfl⟩
    exact source_declarations_loop source tokens position after scope state prior loopComputed

end PtxFrontendProgram

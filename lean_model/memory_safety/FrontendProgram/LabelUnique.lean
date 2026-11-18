/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.CollectBodies

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

private theorem collect_loop_false (source : Slice U8) (tokens : Slice SourceToken) (labels : alloc.vec.Vec Label)
    (position : Usize) (pc : U32) (statement : Bool) :
    strict_memory_frontend_parse.collect_labels_loop0 source tokens labels position pc statement false=.ok (labels,statement,false) := by
  unfold strict_memory_frontend_parse.collect_labels_loop0
  rw [loop.eq_def]
  simp [strict_memory_frontend_parse.collect_labels_loop0.body,bind_ok]

theorem source_collect_loop_unique (source : Slice U8) (tokens : Slice SourceToken) (labels outLabels : alloc.vec.Vec Label)
    (position : Usize) (pc : U32) (statement outStatement : Bool)
    (prior : LabelNamesUnique source.val labels)
    (computed : strict_memory_frontend_parse.collect_labels_loop0 source tokens labels position pc statement true=
      .ok (outLabels,outStatement,true)) : LabelNamesUnique source.val outLabels := by
  generalize fuelEq : tokens.val.length-position.val=fuel
  induction fuel using Nat.strong_induction_on generalizing labels position pc statement with
  | h fuel ih =>
    have unroll : strict_memory_frontend_parse.collect_labels_loop0 source tokens labels position pc statement true=(do
        let outcome ← strict_memory_frontend_parse.collect_labels_loop0.body source tokens labels position pc statement true
        match outcome with
        | .cont next => strict_memory_frontend_parse.collect_labels_loop0 source tokens next.1 next.2.1 next.2.2.1 next.2.2.2.1 next.2.2.2.2
        | .done value => .ok value) := by
      unfold strict_memory_frontend_parse.collect_labels_loop0
      rw [loop.eq_def]
      congr 1
      funext outcome
      cases outcome with
      | cont next => rcases next with ⟨_,_,_,_,_⟩;rfl
      | done value => rfl
    rw [unroll] at computed
    obtain ⟨outcome,bodyComputed,computed⟩ := source_bind_success _ _ _ computed
    cases outcome with
    | done final =>
      rcases final with ⟨values,nextStatement,valid⟩
      simp only [ok.injEq,Prod.mk.injEq] at computed
      rcases computed with ⟨rfl,rfl,rfl⟩
      have frame := source_collect_body_done source tokens labels values position pc statement nextStatement true bodyComputed
      rw [frame.1]
      exact prior
    | cont next =>
      rcases next with ⟨values,after,nextPc,nextStatement,valid⟩
      change strict_memory_frontend_parse.collect_labels_loop0 source tokens values after nextPc nextStatement valid=.ok (outLabels,outStatement,true) at computed
      cases valid with
      | false => rw [collect_loop_false] at computed;simp at computed
      | true =>
        obtain ⟨invariant,advance,bound⟩ := source_collect_body_continue source tokens labels values position after pc nextPc statement nextStatement prior bodyComputed
        exact ih (tokens.val.length-after.val) (by omega) values after nextPc nextStatement invariant computed rfl

theorem source_collect_labels_unique (source : Slice U8) (tokens : Slice SourceToken) (position : Usize) (labels : alloc.vec.Vec Label)
    (computed : strict_memory_frontend_parse.collect_labels source tokens position=.ok (some labels)) :
    labels.val.Pairwise (fun left right => ¬NameEqual source.val left.«name» right.«name») := by
  unfold strict_memory_frontend_parse.collect_labels at computed
  obtain ⟨final,loopComputed,computed⟩ := source_bind_success _ _ _ computed
  rcases final with ⟨values,statement,valid⟩
  cases valid with
  | false => simp at computed
  | true =>
    cases statement with
    | false => simp at computed
    | true =>
      have same : values=labels := by simpa using computed
      subst values
      exact source_collect_loop_unique source tokens (alloc.vec.Vec.new Label) labels position 0#u32 true true (by simp [LabelNamesUnique,alloc.vec.Vec.new]) loopComputed

end PtxFrontendProgram

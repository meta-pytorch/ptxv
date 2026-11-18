/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.NamesDecision

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

private theorem label_fresh_false (source : Slice U8) (labels : alloc.vec.Vec Label) (token : SourceToken) (index : Usize) :
    strict_memory_frontend_parse.collect_labels_loop0_loop0 source labels false token index=.ok false := by
  unfold strict_memory_frontend_parse.collect_labels_loop0_loop0
  rw [loop.eq_def]
  simp [strict_memory_frontend_parse.collect_labels_loop0_loop0.body,bind_ok]

theorem source_label_fresh_loop (source : Slice U8) (labels : alloc.vec.Vec Label) (token : SourceToken) (index : Usize)
    (computed : strict_memory_frontend_parse.collect_labels_loop0_loop0 source labels true token index=.ok true) :
    ∀ position label,index.val≤position → labels.val[position]?=some label → ¬NameEqual source.val label.«name» token := by
  generalize fuelEq : labels.val.length-index.val=fuel
  induction fuel using Nat.strong_induction_on generalizing index with
  | h fuel ih =>
    have unroll : strict_memory_frontend_parse.collect_labels_loop0_loop0 source labels true token index=(do
        let outcome ← strict_memory_frontend_parse.collect_labels_loop0_loop0.body source labels token true index
        match outcome with
        | .cont pair => strict_memory_frontend_parse.collect_labels_loop0_loop0 source labels pair.1 token pair.2
        | .done value => .ok value) := by
      unfold strict_memory_frontend_parse.collect_labels_loop0_loop0
      rw [loop.eq_def]
      congr 1
      funext outcome
      cases outcome with
      | cont pair => rcases pair with ⟨_,_⟩;rfl
      | done value => rfl
    rw [unroll] at computed
    by_cases inside : index.val<labels.val.length
    · have test : index<alloc.vec.Vec.len labels := by scalar_tac
      let label := labels.val[index.val]
      have read : alloc.vec.Vec.index_usize labels index=.ok label := by
        simp [alloc.vec.Vec.index_usize,List.getElem?_eq_getElem inside,label]
      have incrementBound : index.val+1<2^UScalarTy.Usize.numBits := by
        have := alloc.vec.Vec.len_ineq labels
        scalar_tac
      let after := Usize.ofNatCore (index.val+1) incrementBound
      have afterVal : after.val=index.val+1 := by simp [after]
      have increment := source_usize_increment index incrementBound
      simp only [strict_memory_frontend_parse.collect_labels_loop0_loop0.body,ite_true,test,
        alloc.vec.Vec.index_slice_index,read,bind_ok,Std.bind_assoc] at computed
      obtain ⟨same,sameComputed,computed⟩ := source_bind_success _ _ _ computed
      cases same with
      | true =>
        simp only [increment,bind_ok] at computed
        change strict_memory_frontend_parse.collect_labels_loop0_loop0 source labels false token after=.ok true at computed
        rw [label_fresh_false] at computed
        simp at computed
      | false =>
        simp only [increment,bind_ok] at computed
        change strict_memory_frontend_parse.collect_labels_loop0_loop0 source labels true token after=.ok true at computed
        have tail := ih (labels.val.length-after.val) (by omega) after computed rfl
        intro position selected lower found
        by_cases samePosition : position=index.val
        · subst position
          have selectedEq : selected=label := by
            have actual : labels.val[index.val]?=some label := List.getElem?_eq_getElem inside
            rw [found] at actual
            exact Option.some.inj actual
          subst selected
          exact source_names_equal_false source label.«name» token sameComputed
        · exact tail position selected (by omega) found
    · intro position label lower found
      have := (List.getElem?_eq_some_iff.mp found).1
      omega

theorem source_label_fresh (source : Slice U8) (labels : alloc.vec.Vec Label) (token : SourceToken) (valid : Bool)
    (computed : strict_memory_frontend_parse.collect_labels_loop0_loop0 source labels valid token 0#usize=.ok true) :
    ∀ label∈labels.val,¬NameEqual source.val label.«name» token := by
  cases valid with
  | false => rw [label_fresh_false] at computed;simp at computed
  | true =>
    intro label member
    obtain ⟨position,found⟩ := List.mem_iff_getElem?.mp member
    exact source_label_fresh_loop source labels token 0#usize computed position label (by simp) found

end PtxFrontendProgram

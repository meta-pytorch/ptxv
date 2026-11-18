/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.CollectSteps

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

def LabelNamesUnique (source : List U8) (labels : alloc.vec.Vec Label) : Prop :=
  labels.val.Pairwise (fun left right => ¬NameEqual source left.«name» right.«name»)

theorem source_collect_body_continue (source : Slice U8) (tokens : Slice SourceToken)
    (labels outLabels : alloc.vec.Vec Label) (position next : Usize) (pc nextPc : U32) (statement nextStatement : Bool)
    (prior : LabelNamesUnique source.val labels)
    (computed : strict_memory_frontend_parse.collect_labels_loop0.body source tokens labels position pc statement true=
      .ok (.cont (outLabels,next,nextPc,nextStatement,true))) :
    LabelNamesUnique source.val outLabels ∧ position.val<next.val ∧ next.val≤tokens.val.length := by
  unfold strict_memory_frontend_parse.collect_labels_loop0.body at computed
  simp only [ite_true] at computed
  split at computed
  · have inside : position.val<tokens.val.length := by scalar_tac
    simp only [lift,bind_ok] at computed
    obtain ⟨closing,closingComputed,computed⟩ := source_bind_success _ _ _ computed
    cases closing with
    | true => simp at computed
    | false =>
      simp only [Bool.false_eq_true,ite_false] at computed
      cases statement with
      | false =>
        simp only [Bool.false_eq_true,ite_false] at computed
        change countCodeTokenAdded source tokens labels position pc=.ok (.cont (outLabels,next,nextPc,nextStatement,true)) at computed
        obtain ⟨after,pcValue,matched,valid,shape,advance⟩ := count_code_token_added_shape source tokens labels position pc _ computed
        simp only [ControlFlow.cont.injEq,Prod.mk.injEq] at shape
        rw [shape.1,shape.2.1]
        exact ⟨prior,by omega,by omega⟩
      | true =>
        simp only [ite_true] at computed
        obtain ⟨afterOne,oneAdded,computed⟩ := source_bind_success _ _ _ computed
        have oneEq := source_usize_add_value position 1#usize afterOne oneAdded
        have oneValue : afterOne.val=position.val+1 := by simpa using oneEq
        split at computed
        · have secondInside : afterOne.val<tokens.val.length := by scalar_tac
          obtain ⟨colon,colonComputed,computed⟩ := source_bind_success _ _ _ computed
          cases colon with
          | false =>
            simp only [Bool.false_eq_true,ite_false] at computed
            change countCodeToken source tokens labels position afterOne pc=.ok (.cont (outLabels,next,nextPc,nextStatement,true)) at computed
            obtain ⟨pcValue,matched,valid,shape⟩ := count_code_token_shape source tokens labels position afterOne pc _ computed
            simp only [ControlFlow.cont.injEq,Prod.mk.injEq] at shape
            rw [shape.1,shape.2.1]
            exact ⟨prior,by omega,by omega⟩
          | true =>
            simp only [ite_true] at computed
            obtain ⟨token,tokenComputed,computed⟩ := source_bind_success _ _ _ computed
            obtain ⟨nameValid,nameComputed,computed⟩ := source_bind_success _ _ _ computed
            obtain ⟨fresh,freshComputed,computed⟩ := source_bind_success _ _ _ computed
            obtain ⟨pushed,pushedComputed,computed⟩ := source_bind_success _ _ _ computed
            obtain ⟨after,afterAdded,computed⟩ := source_bind_success _ _ _ computed
            simp only [ok.injEq,ControlFlow.cont.injEq,Prod.mk.injEq] at computed
            have freshTrue : fresh=true := computed.2.2.2.2
            rw [freshTrue] at freshComputed
            have newFresh := source_label_fresh source labels token nameValid freshComputed
            have appended := vec_push_success labels pushed (⟨token,pc⟩ : Label) pushedComputed
            have nextValue := source_usize_add_value position 2#usize after afterAdded
            refine ⟨?_,?_,?_⟩
            · unfold LabelNamesUnique
              rw [←computed.1,appended,List.pairwise_append]
              refine ⟨prior,by simp,?_⟩
              intro old oldMember selected selectedMember
              have same : selected=(⟨token,pc⟩ : Label) := by simpa using selectedMember
              subst selected
              exact newFresh old oldMember
            · rw [←computed.2.1]
              scalar_tac
            · rw [←computed.2.1]
              scalar_tac
        · change countCodeToken source tokens labels position afterOne pc=.ok (.cont (outLabels,next,nextPc,nextStatement,true)) at computed
          obtain ⟨pcValue,matched,valid,shape⟩ := count_code_token_shape source tokens labels position afterOne pc _ computed
          simp only [ControlFlow.cont.injEq,Prod.mk.injEq] at shape
          rw [shape.1,shape.2.1]
          exact ⟨prior,by omega,by omega⟩
  · simp at computed

theorem source_collect_body_done (source : Slice U8) (tokens : Slice SourceToken)
    (labels outLabels : alloc.vec.Vec Label) (position : Usize) (pc : U32) (statement outStatement valid : Bool)
    (computed : strict_memory_frontend_parse.collect_labels_loop0.body source tokens labels position pc statement true=
      .ok (.done (outLabels,outStatement,valid))) : outLabels=labels ∧ valid=true := by
  unfold strict_memory_frontend_parse.collect_labels_loop0.body at computed
  simp only [ite_true] at computed
  split at computed
  · simp only [lift,bind_ok] at computed
    obtain ⟨closing,closingComputed,computed⟩ := source_bind_success _ _ _ computed
    cases closing with
    | true =>
      simp only [ite_true,ok.injEq,ControlFlow.done.injEq,Prod.mk.injEq] at computed
      exact ⟨computed.1.symm,computed.2.2.symm⟩
    | false =>
      simp only [Bool.false_eq_true,ite_false] at computed
      cases statement with
      | false =>
        simp only [Bool.false_eq_true,ite_false] at computed
        change countCodeTokenAdded source tokens labels position pc=.ok (.done (outLabels,outStatement,valid)) at computed
        obtain ⟨after,pcValue,matched,nextValid,shape,_⟩ := count_code_token_added_shape source tokens labels position pc _ computed
        cases shape
      | true =>
        simp only [ite_true] at computed
        obtain ⟨afterOne,oneAdded,computed⟩ := source_bind_success _ _ _ computed
        split at computed
        · obtain ⟨colon,colonComputed,computed⟩ := source_bind_success _ _ _ computed
          cases colon with
          | false =>
            simp only [Bool.false_eq_true,ite_false] at computed
            change countCodeToken source tokens labels position afterOne pc=.ok (.done (outLabels,outStatement,valid)) at computed
            obtain ⟨pcValue,matched,nextValid,shape⟩ := count_code_token_shape source tokens labels position afterOne pc _ computed
            cases shape
          | true =>
            simp only [ite_true] at computed
            obtain ⟨token,tokenComputed,computed⟩ := source_bind_success _ _ _ computed
            obtain ⟨nameValid,nameComputed,computed⟩ := source_bind_success _ _ _ computed
            obtain ⟨fresh,freshComputed,computed⟩ := source_bind_success _ _ _ computed
            obtain ⟨pushed,pushedComputed,computed⟩ := source_bind_success _ _ _ computed
            obtain ⟨after,afterAdded,computed⟩ := source_bind_success _ _ _ computed
            simp at computed
        · change countCodeToken source tokens labels position afterOne pc=.ok (.done (outLabels,outStatement,valid)) at computed
          obtain ⟨pcValue,matched,nextValid,shape⟩ := count_code_token_shape source tokens labels position afterOne pc _ computed
          cases shape
  · simp only [ok.injEq,ControlFlow.done.injEq,Prod.mk.injEq] at computed
    exact ⟨computed.1.symm,computed.2.2.symm⟩

end PtxFrontendProgram

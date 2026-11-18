/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputLayout
import MemorySafetyExecution.Native.PreparedEvaluation
import ValueFlow.SourceDependencies

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

open ValueFlow ValueFlow.Source

private theorem successful_false {α : Type} (condition : Prop) [Decidable condition]
    (reason : Fault) (next : Result (core.result.Result α Fault)) (out : α)
    (accepted : (if condition then .ok (.Err reason) else next) = .ok (.Ok out)) :
    ¬condition ∧ next = .ok (.Ok out) := by
  by_cases h : condition <;> simp_all

private theorem successful_true {α : Type} (condition : Prop) [Decidable condition]
    (reason : Fault) (next : Result (core.result.Result α Fault)) (out : α)
    (accepted : (if condition then next else .ok (.Err reason)) = .ok (.Ok out)) :
    condition ∧ next = .ok (.Ok out) := by
  by_cases h : condition <;> simp_all

theorem source_prepare_placements (program rightProgram : CoreProgram) (shadow : value_flow_dependencies.State)
    (left right : CoreState) (related : StateInputsRelated shadow left right)
    (sameCode : program.code=rightProgram.code)
    (rightLayout : execution_core.layout_valid rightProgram right=.ok true)
    (actor : Usize) (commit : execution_core.Commit)
    (thread : execution_core.ThreadState) (instruction : V.GuardedInstruction)
    (metadata : DependencyRegisters) (loaded : Slice Metadata) (transferred : DependencyTransfer)
    (threadAt : left.threads.val[actor.val]? = some thread)
    (instructionAt : program.code.val[thread.pc.val]? = some instruction)
    (metadataAt : shadow.threads.val[actor.val]? = some metadata)
    (transferComputed : value_flow_dependencies.transfer instruction commit.action metadata loaded = .ok (.Ok transferred))
    (prepared : execution_core.prepare program left actor = .ok (.Ok commit)) :
    ∃ other, execution_core.prepare rightProgram right actor = .ok (.Ok other) ∧
      other.actor = commit.actor ∧ other.next_pc = commit.next_pc ∧
      ActionsRelated (transferTaint transferred) (actionModel commit.action) (actionModel other.action) := by
  have preparedMeaning := source_prepare_evaluation program left actor commit prepared
  obtain ⟨originalThread, originalInstruction, originalAt, running, originalInstructionAt,
    commitActor, nextPcValue, evaluated⟩ := preparedMeaning
  have sameThread : originalThread = thread := by simpa only [threadAt, Option.some.injEq] using originalAt.symm
  subst originalThread
  have sameInstruction : originalInstruction = instruction := by
    simpa only [instructionAt, Option.some.injEq] using originalInstructionAt.symm
  subst originalInstruction
  have actorInside : actor.val < left.threads.val.length := by
    have found := List.getElem?_eq_some_iff.mp threadAt
    exact found.1
  have rightInside : actor.val < right.threads.val.length := by
    have lengths := related.threadsLength
    omega
  let otherThread := right.threads.val[actor.val]
  have otherAt : right.threads.val[actor.val]? = some otherThread := List.getElem?_eq_getElem rightInside
  have threadRelation := related.threads actor.val thread otherThread metadata threadAt otherAt metadataAt
  obtain ⟨otherAction, otherEvaluated, actions⟩ := source_evaluate_inputs thread.registers
    otherThread.registers thread.ids instruction commit.action metadata loaded transferred
    threadRelation.registers.banks transferComputed evaluated
  rw [threadRelation.ids] at otherEvaluated
  unfold execution_core.prepare at prepared ⊢
  obtain ⟨layout, layoutRun, prepared⟩ := MemorySafety.Source.source_bind_success _ _ _ prepared
  obtain ⟨valid, prepared⟩ := successful_true _ _ _ _ prepared
  obtain ⟨actorBound, prepared⟩ := successful_false _ _ _ _ prepared
  obtain ⟨fetchedThread, fetched, prepared⟩ := MemorySafety.Source.source_bind_success _ _ _ prepared
  have sameFetched : fetchedThread = thread := by
    have read : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice execution_core.ThreadState)
        left.threads actor = .ok thread := by
      simp only [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
        alloc.vec.Vec.getElem?_Nat_eq, threadAt]
    simpa only [read, ok.injEq] using fetched.symm
  subst fetchedThread
  obtain ⟨isRunning, prepared⟩ := successful_true _ _ _ _ prepared
  obtain ⟨pc, pcCast, prepared⟩ := MemorySafety.Source.source_bind_success _ _ _ prepared
  obtain ⟨pcBound, prepared⟩ := successful_false _ _ _ _ prepared
  obtain ⟨notMaximum, prepared⟩ := successful_false _ _ _ _ prepared
  obtain ⟨nextPc, nextEq, prepared⟩ := MemorySafety.Source.source_bind_success _ _ _ prepared
  obtain ⟨index, indexCast, prepared⟩ := MemorySafety.Source.source_bind_success _ _ _ prepared
  obtain ⟨fetchedInstruction, fetchedCode, prepared⟩ := MemorySafety.Source.source_bind_success _ _ _ prepared
  obtain ⟨evaluation, evaluationRun, prepared⟩ := MemorySafety.Source.source_bind_success _ _ _ prepared
  have castBound : thread.pc.val ≤ UScalar.max .Usize := by
    rcases Usize.bounds_eq with narrow | wide <;> scalar_tac
  have indexVal : index.val = thread.pc.val := by
    have spec := UScalar.cast_inBounds_spec .Usize thread.pc castBound
    simpa only [indexCast, WP.spec_ok] using spec
  have sameIndex : pc = index := by simpa only [pcCast, ok.injEq] using indexCast
  have instructionRead : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice value_flow.GuardedInstruction) program.code index = .ok instruction := by
    simp only [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
      alloc.vec.Vec.getElem?_Nat_eq, indexVal, instructionAt]
  have sameFetchedInstruction : fetchedInstruction = instruction := by
    simpa only [instructionRead, ok.injEq] using fetchedCode.symm
  subst fetchedInstruction
  have sameEvaluation : evaluation = .Ok commit.action := by
    simpa only [evaluated, ok.injEq] using evaluationRun.symm
  subst evaluation
  have commitEq : ({ actor, next_pc := nextPc, action := commit.action } : execution_core.Commit) = commit := by
    simpa only [ok.injEq, core.result.Result.Ok.injEq] using prepared
  have otherRead : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice execution_core.ThreadState)
      right.threads actor = .ok otherThread := by
    simp only [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
      alloc.vec.Vec.getElem?_Nat_eq, otherAt]
  have otherActorBound : ¬actor ≥ alloc.vec.Vec.len right.threads := by scalar_tac
  refine ⟨{ actor, next_pc := nextPc, action := otherAction }, ?_, ?_, ?_, actions⟩
  · rw [rightLayout]
    simp only [bind_ok,ite_true,otherActorBound,ite_false,otherRead]
    rw [←sameCode]
    simp only [←threadRelation.pc, ←threadRelation.running, isRunning, ite_true]
    rw [pcCast]
    simp only [bind_ok, pcBound, notMaximum, ite_false]
    rw [nextEq]
    simp only [bind_ok]
    rw [sameIndex]
    simp only [bind_ok, instructionRead]
    simp only [bind_ok, otherEvaluated]
  · exact commitActor.symm
  · have nextCommit := congrArg (fun value : execution_core.Commit => value.next_pc) commitEq
    simpa only using nextCommit

end MemorySafetyExecution.Native

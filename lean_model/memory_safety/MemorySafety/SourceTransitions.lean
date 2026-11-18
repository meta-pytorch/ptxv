/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourceModel
import MemorySafety.SourceVectors

open Aeneas Aeneas.Std Result PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option backward.isDefEq.respectTransparency false
set_option maxHeartbeats 5000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafety.Source

private theorem choices_unroll (pcs : alloc.vec.Vec Usize) (thread : Usize)
    (result : alloc.vec.Vec memory_safety.Move) (code : alloc.vec.Vec memory_safety.Instruction)
    (targets : alloc.vec.Vec Usize) (index : Usize) (known : Bool) :
    memory_safety.thread_moves_loop pcs thread result code targets (alloc.vec.Vec.len targets) index known = (do
      let outcome ← memory_safety.thread_moves_loop.body pcs thread code targets
        (alloc.vec.Vec.len targets) result index known
      match outcome with
      | .cont (next, i, k) =>
        memory_safety.thread_moves_loop pcs thread next code targets (alloc.vec.Vec.len targets) i k
      | .done output => .ok output) := by
  unfold memory_safety.thread_moves_loop
  rw [loop.eq_def]
  congr 1
  funext outcome
  cases outcome with
  | cont triple => rcases triple with ⟨_, _, _⟩; rfl
  | done output => rfl

private theorem choices_false (pcs : alloc.vec.Vec Usize) (thread : Usize)
    (result : alloc.vec.Vec memory_safety.Move) (code : alloc.vec.Vec memory_safety.Instruction)
    (targets : alloc.vec.Vec Usize) (index : Usize) :
    memory_safety.thread_moves_loop pcs thread result code targets (alloc.vec.Vec.len targets)
      index false = .ok (result, false) := by
  rw [choices_unroll]
  by_cases inside : index < alloc.vec.Vec.len targets <;>
    simp [memory_safety.thread_moves_loop.body, inside]

theorem source_choice_moves (pcs : alloc.vec.Vec Usize) (thread : Usize)
    (code : alloc.vec.Vec memory_safety.Instruction) (targets : alloc.vec.Vec Usize)
    (index : Usize) (accumulator result : alloc.vec.Vec memory_safety.Move)
    (threadInside : thread.val < pcs.val.length)
    (computed : memory_safety.thread_moves_loop pcs thread accumulator code targets
      (alloc.vec.Vec.len targets) index true = .ok (result, true)) :
    (targets.val.drop index.val).all (fun target => decide (target.val ≤ code.val.length)) = true ∧
    result.val = accumulator.val ++ (targets.val.drop index.val).map (fun target =>
      ({ event := memory_safety.Event.Control, pcs := alloc.vec.Vec.set pcs thread target } : memory_safety.Move)) := by
  generalize fuelEq : targets.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing index accumulator with
  | h fuel ih =>
    rw [choices_unroll] at computed
    by_cases inside : index.val < targets.val.length
    · have test : index < alloc.vec.Vec.len targets := by scalar_tac
      have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := targets.property
        scalar_tac
      let nextIndex : Usize := Usize.ofNatCore (index.val + 1) bound
      have nextVal : nextIndex.val = index.val + 1 := by simp [nextIndex]
      have increment : (index + 1#usize : Result Usize) = .ok nextIndex := source_usize_increment index bound
      let target := targets.val[index.val]
      have read : alloc.vec.Vec.index_usize targets index = .ok target := by
        simp [alloc.vec.Vec.index_usize, List.getElem?_eq_getElem inside, target]
      have update : alloc.vec.Vec.index_mut_usize pcs thread = .ok
          (pcs.val[thread.val], alloc.vec.Vec.set pcs thread) := by
        simp [alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize,
          List.getElem?_eq_getElem threadInside]
      by_cases outside : target > alloc.vec.Vec.len code
      · have bodyEq : memory_safety.thread_moves_loop.body pcs thread code targets
            (alloc.vec.Vec.len targets) accumulator index true =
            .ok (.cont (accumulator, nextIndex, false)) := by
          simp only [memory_safety.thread_moves_loop.body, test, ite_true,
            alloc.vec.Vec.index_slice_index, read, bind_ok, outside, ite_true]
          simp [increment]
        rw [bodyEq] at computed
        simp only [bind_ok, choices_false, ok.injEq, Prod.mk.injEq, Bool.false_eq_true,
          and_false] at computed
      · have bodyEq : memory_safety.thread_moves_loop.body pcs thread code targets
            (alloc.vec.Vec.len targets) accumulator index true = (do
              let pushed ← alloc.vec.Vec.push accumulator
                ({ event := memory_safety.Event.Control, pcs := alloc.vec.Vec.set pcs thread target } : memory_safety.Move)
              .ok (.cont (pushed, nextIndex, true))) := by
          simp only [memory_safety.thread_moves_loop.body, test, ite_true,
            alloc.vec.Vec.index_slice_index, read, bind_ok, outside, ite_false,
            vec_clone_exact core.clone.CloneUsize (by intro value; rfl),
            alloc.vec.Vec.index_mut_slice_index, update]
          change (do
            let pair ← (do
              let pushed ← alloc.vec.Vec.push accumulator
                ({ event := memory_safety.Event.Control, pcs := alloc.vec.Vec.set pcs thread target } : memory_safety.Move)
              .ok (pushed, true))
            let i2 ← index + 1#usize
            .ok (ControlFlow.cont (pair.1, i2, pair.2) : ControlFlow
              (alloc.vec.Vec memory_safety.Move × Usize × Bool) (alloc.vec.Vec memory_safety.Move × Bool))) = _
          simp [increment]
          rfl
        rw [bodyEq] at computed
        simp only [Std.bind_assoc] at computed
        obtain ⟨pushed, pushedEq, rest⟩ := source_bind_success _ _ _ computed
        simp only [bind_ok] at rest
        have pushedVal := vec_push_success accumulator pushed
          ({ event := memory_safety.Event.Control, pcs := alloc.vec.Vec.set pcs thread target } : memory_safety.Move) pushedEq
        obtain ⟨all, values⟩ := ih (targets.val.length - nextIndex.val) (by omega)
          nextIndex pushed rest rfl
        have targetValid : target.val ≤ code.val.length := by scalar_tac
        rw [List.drop_eq_getElem_cons inside]
        constructor
        · simp only [List.all_cons, Bool.and_eq_true, decide_eq_true_eq]
          exact ⟨targetValid, by simpa [nextVal] using all⟩
        · simpa only [List.map_cons, nextVal, pushedVal, List.append_assoc,
            List.singleton_append, target] using values
    · have test : ¬ index < alloc.vec.Vec.len targets := by scalar_tac
      simp only [memory_safety.thread_moves_loop.body, test, ite_false, bind_ok,
        ok.injEq, Prod.mk.injEq] at computed
      rcases computed with ⟨rfl, _⟩
      simp [List.drop_eq_nil_of_le (show targets.val.length ≤ index.val by omega)]

private theorem pcs_index_mut_exact (pcs : alloc.vec.Vec Usize) (thread : Usize)
    (inside : thread.val < pcs.val.length) :
    alloc.vec.Vec.index_mut_usize pcs thread = .ok
      (pcs.val[thread.val], alloc.vec.Vec.set pcs thread) := by
  simp [alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize,
    List.getElem?_eq_getElem inside]

private theorem singleton_movement (pcs : alloc.vec.Vec Usize) (thread nextPc : Usize)
    (event : memory_safety.Event) (output : alloc.vec.Vec memory_safety.Move)
    (pushed : alloc.vec.Vec.push (alloc.vec.Vec.new memory_safety.Move)
      ({event := event, pcs := alloc.vec.Vec.set pcs thread nextPc} : memory_safety.Move) = .ok output) :
    output.val.map moveModel = [⟨eventModel event, (pcs.val.map UScalar.val).set thread.val nextPc.val⟩] := by
  have values := vec_push_success _ _ _ pushed
  simp only [values, alloc.vec.Vec.new]
  simp [moveModel, List.map_set]

theorem source_thread_moves_agrees (program : memory_safety.Program) (state : memory_safety.State)
    (thread : Usize) (choices : alloc.vec.Vec memory_safety.Move)
    (computed : memory_safety.thread_moves program state thread = .ok (some choices)) :
    threadMoves (programModel program) (stateModel state) thread.val = some (choices.val.map moveModel) := by
  unfold memory_safety.thread_moves at computed
  obtain ⟨code, codeEq, computed⟩ := source_bind_success _ _ _ computed
  obtain ⟨pc, pcEq, computed⟩ := source_bind_success _ _ _ computed
  simp only [alloc.vec.Vec.index_slice_index] at codeEq pcEq
  have codeFound := vec_index_success program.threads thread code codeEq
  have pcFound := vec_index_success state.pcs thread pc pcEq
  have threadInside : thread.val < state.pcs.val.length := List.getElem?_eq_some_iff.mp pcFound |>.1
  have update := pcs_index_mut_exact state.pcs thread threadInside
  have clone := vec_clone_exact core.clone.CloneUsize (by intro value; rfl) state.pcs
  simp only [threadMoves, programModel, stateModel, List.getElem?_map, codeFound, pcFound,
    Option.map_some, List.length_map]
  by_cases beyond : pc > alloc.vec.Vec.len code
  · simp [beyond] at computed
  · have notBeyond : ¬pc.val > code.val.length := by scalar_tac
    simp only [beyond, ite_false] at computed
    simp only [notBeyond, ite_false]
    by_cases inside : pc < alloc.vec.Vec.len code
    · have pcInside : pc.val < code.val.length := by scalar_tac
      have notEnd : pc.val ≠ code.val.length := by omega
      simp only [inside, ite_true] at computed
      simp only [notEnd, ite_false]
      obtain ⟨instruction, instructionEq, computed⟩ := source_bind_success _ _ _ computed
      simp only [alloc.vec.Vec.index_slice_index] at instructionEq
      have instructionFound := vec_index_success code pc instruction instructionEq
      simp only [instructionFound, Option.map_some]
      have bound : pc.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := code.property
        scalar_tac
      let nextPc : Usize := Usize.ofNatCore (pc.val + 1) bound
      have nextPcVal : nextPc.val = pc.val + 1 := by simp [nextPc]
      have increment : (pc + 1#usize : Result Usize) = .ok nextPc := source_usize_increment pc bound
      cases instruction with
      | Unsupported => simp at computed
      | Access request =>
        simp only [increment, bind_ok, clone, alloc.vec.Vec.index_mut_slice_index, update] at computed
        obtain ⟨output, pushed, rest⟩ := source_bind_success _ _ _ computed
        simp only [ok.injEq, Option.some.injEq] at rest
        subst output
        rw [singleton_movement _ _ _ _ _ pushed]
        simp [instructionModel, eventModel, nextPcVal]
      | Free allocation =>
        simp only [increment, bind_ok, clone, alloc.vec.Vec.index_mut_slice_index, update] at computed
        obtain ⟨output, pushed, rest⟩ := source_bind_success _ _ _ computed
        simp only [ok.injEq, Option.some.injEq] at rest
        subst output
        rw [singleton_movement _ _ _ _ _ pushed]
        simp [instructionModel, eventModel, nextPcVal]
      | Skip =>
        simp only [increment, bind_ok, clone, alloc.vec.Vec.index_mut_slice_index, update] at computed
        obtain ⟨output, pushed, rest⟩ := source_bind_success _ _ _ computed
        simp only [ok.injEq, Option.some.injEq] at rest
        subst output
        rw [singleton_movement _ _ _ _ _ pushed]
        simp [instructionModel, eventModel, nextPcVal]
      | Halt =>
        simp only [clone, bind_ok, alloc.vec.Vec.index_mut_slice_index, update] at computed
        obtain ⟨output, pushed, rest⟩ := source_bind_success _ _ _ computed
        simp only [ok.injEq, Option.some.injEq] at rest
        subst output
        rw [singleton_movement _ _ _ _ _ pushed]
        simp [instructionModel, eventModel]
      | Choice targets =>
        by_cases empty : alloc.vec.Vec.len targets = 0#usize
        · simp only [empty, ite_true, ok.injEq] at computed
          contradiction
        · simp only [empty, ite_false] at computed
          obtain ⟨pair, loopEq, computed⟩ := source_bind_success _ _ _ computed
          rcases pair with ⟨output, known⟩
          cases known with
          | false => simp at computed
          | true =>
            change (.ok (some output) : Result _) = .ok (some choices) at computed
            simp only [ok.injEq, Option.some.injEq] at computed
            subst output
            obtain ⟨all, values⟩ := source_choice_moves state.pcs thread code targets
              0#usize (alloc.vec.Vec.new memory_safety.Move) choices threadInside loopEq
            have nonempty : targets.val ≠ [] := by
              intro nil
              apply empty
              apply UScalar.eq_of_val_eq
              simp [nil]
            simp at all values
            simp only [instructionModel, List.map_eq_nil_iff, nonempty, ite_false,
              List.all_map, Function.comp_def]
            rw [values]
            simp [moveModel, eventModel, List.map_map, List.map_set]
            exact all
    · have endPc : pc.val = code.val.length := by scalar_tac
      simp only [inside, ite_false, ok.injEq, Option.some.injEq] at computed
      subst choices
      simp [endPc]

end MemorySafety.Source

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

private theorem successor_unroll (program : memory_safety.Program) (state : memory_safety.State)
    (choices : alloc.vec.Vec memory_safety.Move) (result : alloc.vec.Vec memory_safety.State)
    (index : Usize) :
    memory_safety.successors_loop program state choices result index = (do
      let outcome ← memory_safety.successors_loop.body program state choices result index
      match outcome with
      | .cont (next, i) => memory_safety.successors_loop program state choices next i
      | .done output => .ok output) := by
  unfold memory_safety.successors_loop
  rw [loop.eq_def]
  congr 1
  funext outcome
  cases outcome with
  | cont pair => cases pair; rfl
  | done output => rfl

theorem source_successor_loop (program : memory_safety.Program) (state : memory_safety.State)
    (choices : alloc.vec.Vec memory_safety.Move) (index : Usize)
    (accumulator result : alloc.vec.Vec memory_safety.State)
    (computed : memory_safety.successors_loop program state choices accumulator index = .ok result) :
    ∃ tail, result.val = accumulator.val ++ tail ∧
      List.Forall₂ (fun movement after => memory_safety.apply_move program state movement = .ok after)
        (choices.val.drop index.val) tail := by
  generalize fuelEq : choices.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing index accumulator with
  | h fuel ih =>
    rw [successor_unroll] at computed
    by_cases inside : index.val < choices.val.length
    · have test : index < alloc.vec.Vec.len choices := by scalar_tac
      have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := choices.property
        scalar_tac
      let nextIndex : Usize := Usize.ofNatCore (index.val + 1) bound
      have nextVal : nextIndex.val = index.val + 1 := by simp [nextIndex]
      have increment : (index + 1#usize : Result Usize) = .ok nextIndex := source_usize_increment index bound
      have read : alloc.vec.Vec.index_usize choices index = .ok choices.val[index.val] := by
        simp [alloc.vec.Vec.index_usize, List.getElem?_eq_getElem inside]
      simp only [memory_safety.successors_loop.body, test, ite_true,
        alloc.vec.Vec.index_slice_index, read, bind_ok] at computed
      simp only [Std.bind_assoc] at computed
      obtain ⟨after, applied, computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨pushed, pushedEq, computed⟩ := source_bind_success _ _ _ computed
      simp only [increment, bind_ok] at computed
      obtain ⟨tail, values, related⟩ := ih (choices.val.length - nextIndex.val) (by omega)
        nextIndex pushed computed rfl
      have pushedVal := vec_push_success accumulator pushed after pushedEq
      refine ⟨after :: tail, ?_, ?_⟩
      · simpa only [pushedVal, List.append_assoc, List.singleton_append] using values
      · rw [List.drop_eq_getElem_cons inside]
        exact .cons applied (by simpa only [nextVal] using related)
    · have test : ¬index < alloc.vec.Vec.len choices := by scalar_tac
      simp only [memory_safety.successors_loop.body, test, ite_false, bind_ok, ok.injEq] at computed
      subst result
      refine ⟨[], by simp, ?_⟩
      rw [List.drop_eq_nil_of_le (show choices.val.length ≤ index.val by omega)]
      exact .nil

theorem source_successors_execution (program : memory_safety.Program) (state : memory_safety.State)
    (next : alloc.vec.Vec memory_safety.State)
    (computed : memory_safety.successors program state = .ok (some next)) :
    ∃ choices, memory_safety.moves program state = .ok (some choices) ∧
      List.Forall₂ (fun movement after => memory_safety.apply_move program state movement = .ok after)
        choices.val next.val := by
  unfold memory_safety.successors at computed
  obtain ⟨possibility, choicesEq, computed⟩ := source_bind_success _ _ _ computed
  cases possibility with
  | none => simp at computed
  | some choices =>
    obtain ⟨output, loopEq, computed⟩ := source_bind_success _ _ _ computed
    simp only [ok.injEq, Option.some.injEq] at computed
    subst output
    obtain ⟨tail, values, related⟩ := source_successor_loop program state choices 0#usize
      (alloc.vec.Vec.new memory_safety.State) next loopEq
    have same : next.val = tail := by simpa using values
    exact ⟨choices, choicesEq, same.symm ▸ related⟩

end MemorySafety.Source

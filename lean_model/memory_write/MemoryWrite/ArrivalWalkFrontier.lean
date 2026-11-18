/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Code.Funs
import MemoryWrite.ArrivalWalkModel

open Aeneas Aeneas.Std Result PtxvMemoryWrite
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace ArrivalWalk

def Represents {State : Type} (frontier : arrival_walk.Frontier State) (states : List State) : Prop :=
  frontier.len.val ≤ frontier.slots.val.length ∧
  frontier.slots.val.take frontier.len.val = states.map some

theorem Represents.length {State : Type} {frontier : arrival_walk.Frontier State} {states : List State}
    (rep : Represents frontier states) : frontier.len.val = states.length := by
  have sizes := congrArg List.length rep.2
  simpa [List.length_take, Nat.min_eq_left rep.1] using sizes

@[step]
theorem rust_new {State : Type} : arrival_walk.Frontier.new State ⦃ result => Represents result [] ⦄ := by
  simp [arrival_walk.Frontier.new, Represents, alloc.vec.Vec.new, alloc.vec.Vec.from_val]

@[step]
theorem rust_push {State : Type} (frontier : arrival_walk.Frontier State) (states : List State)
    (rep : Represents frontier states) (state : State) :
    arrival_walk.Frontier.push frontier state ⦃ success result =>
      (success = false ∧ result = frontier) ∨ (success = true ∧ Represents result (states ++ [state])) ⦄ := by
  unfold arrival_walk.Frontier.push
  by_cases maximum : frontier.len = core.num.Usize.MAX
  · simp only [maximum, ite_true, WP.spec_ok]
    exact .inl ⟨rfl, rfl⟩
  · simp only [maximum, ite_false]
    have space : frontier.len.val < Usize.max := by scalar_tac
    by_cases existing : frontier.len.val < frontier.slots.val.length
    · have test : frontier.len < alloc.vec.Vec.len frontier.slots := by scalar_tac
      simp only [test, ite_true]
      step as ⟨previous, previousEq, backEq⟩
      rcases previous with ⟨old, back⟩
      dsimp only at backEq
      step with Usize.add_spec (x := frontier.len) (y := 1#usize) (by scalar_tac) as ⟨next, nextEq⟩
      refine .inr ⟨?_, ?_⟩
      · rw [backEq]
        simp only [alloc.vec.Vec.set_val_eq, List.length_set]
        omega
      · rw [backEq]
        simp only [alloc.vec.Vec.set_val_eq, nextEq, List.map_append, List.map_cons, List.map_nil]
        rw [List.take_succ_eq_append_getElem (by simp; omega)]
        simp only [List.take_set, List.getElem_set_self, List.append_cancel_right_eq]
        rw [List.set_eq_of_length_le (by simp)]
        exact rep.2
    · have full : frontier.len.val = frontier.slots.val.length := by have := rep.1; omega
      have test : ¬frontier.len < alloc.vec.Vec.len frontier.slots := by scalar_tac
      simp only [test, ite_false]
      step with alloc.vec.Vec.push_spec frontier.slots (some state) (by omega) as ⟨result, resultEq⟩
      step with Usize.add_spec (x := frontier.len) (y := 1#usize) (by scalar_tac) as ⟨next, nextEq⟩
      refine .inr ⟨?_, ?_⟩
      · simp only [resultEq, List.length_append, List.length_singleton]
        omega
      · simp only [resultEq, nextEq, full, List.take_length_add_append, List.map_append,
          List.map_cons, List.map_nil]
        have all : frontier.slots.val = states.map some := by simpa only [full, List.take_length] using rep.2
        simp [all]

@[step]
theorem rust_pop_empty {State : Type} (frontier : arrival_walk.Frontier State)
    (rep : Represents frontier []) :
    arrival_walk.Frontier.pop frontier ⦃ value result => value = none ∧ result = frontier ⦄ := by
  have zero : frontier.len = 0#usize := by have := rep.length; scalar_tac
  simp [arrival_walk.Frontier.pop, zero]

@[step]
theorem rust_pop_append {State : Type} (frontier : arrival_walk.Frontier State)
    (states : List State) (state : State) (rep : Represents frontier (states ++ [state])) :
    arrival_walk.Frontier.pop frontier ⦃ value result =>
      value = some state ∧ Represents result states ∧ result.len.val + 1 = frontier.len.val ⦄ := by
  have size := rep.length
  have nonzero : frontier.len ≠ 0#usize := by simp only [List.length_append, List.length_singleton] at size; scalar_tac
  unfold arrival_walk.Frontier.pop
  simp only [nonzero, ite_false]
  step with Usize.sub_spec (x := frontier.len) (y := 1#usize) (by scalar_tac) as ⟨index, indexEq⟩
  have indexInside : index.val < frontier.slots.val.length := by have := rep.1; simp only [List.length_append, List.length_singleton] at size; omega
  step with alloc.vec.Vec.index_mut_usize_spec frontier.slots index indexInside as ⟨old, back, oldEq, backEq⟩
  have last : index.val = states.length := by simp only [List.length_append, List.length_singleton] at size; omega
  have oldValue : old = some state := by
    have atLast := congrArg (fun values : List (Option State) => values[index.val]?) rep.2
    have inPrefix : index.val < frontier.len.val := by omega
    simp only [List.getElem?_take, inPrefix, ite_true, List.getElem?_eq_getElem indexInside] at atLast
    have observed : frontier.slots.val[index.val] = some state := by simpa [last, List.map_append] using atLast
    exact oldEq.trans observed
  simp only [oldValue]
  refine ⟨trivial, ⟨?_, ?_⟩, ?_⟩
  · rw [backEq]
    simp only [alloc.vec.Vec.set_val_eq, List.length_set]
    omega
  · rw [backEq]
    simp only [alloc.vec.Vec.set_val_eq, List.take_set]
    rw [List.set_eq_of_length_le (by simp)]
    have truncated := congrArg (List.take index.val) rep.2
    have bound : states.length ≤ frontier.len.val := by simp only [List.length_append, List.length_singleton] at size; omega
    simpa [last, List.take_take, List.map_append, Nat.min_eq_left bound] using truncated
  · simp only [List.length_append, List.length_singleton] at size
    omega

end ArrivalWalk

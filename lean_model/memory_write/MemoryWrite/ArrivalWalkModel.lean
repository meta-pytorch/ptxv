/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Aeneas

set_option autoImplicit false
namespace ArrivalWalk

structure Graph (State : Type) where
  hit : State → Prop
  edge : State → State → Prop

inductive Reaches {State : Type} (graph : Graph State) (start : State) : State → Prop
  | start : Reaches graph start start
  | next {before after : State} : Reaches graph start before → graph.edge before after → Reaches graph start after

structure Covered {State : Type} (graph : Graph State) (start : State)
    (done : State → Prop) (pending : List State) : Prop where
  startCovered : done start ∨ start ∈ pending
  closed : ∀ state, done state → ¬graph.hit state ∧ ∀ destination, graph.edge state destination →
    done destination ∨ destination ∈ pending

theorem Covered.completed {State : Type} (graph : Graph State) (start : State) (done : State → Prop)
    (covered : Covered graph start done []) (state : State) (path : Reaches graph start state) :
    ¬graph.hit state := by
  have inside : done state := by
    induction path with
    | start => simpa using covered.startCovered
    | next _ edge ih => simpa using (covered.closed _ ih).2 _ edge
  exact (covered.closed state inside).1

theorem Covered.duplicate {State : Type} (graph : Graph State) (start state : State)
    (done : State → Prop) (pending : List State) (covered : Covered graph start done (pending ++ [state]))
    (already : done state) : Covered graph start done pending := by
  constructor
  · rcases covered.startCovered with existing | remaining
    · exact .inl existing
    · rcases List.mem_append.mp remaining with kept | popped
      · exact .inr kept
      · have same : start = state := by simpa using popped
        exact .inl (by rw [same]; exact already)
  · intro visited member
    refine ⟨(covered.closed visited member).1, ?_⟩
    intro destination edge
    rcases (covered.closed visited member).2 destination edge with existing | remaining
    · exact .inl existing
    · rcases List.mem_append.mp remaining with kept | popped
      · exact .inr kept
      · have same : destination = state := by simpa using popped
        exact .inl (by rw [same]; exact already)

theorem Covered.expand {State : Type} (graph : Graph State) (start state : State)
    (done : State → Prop) (pending children : List State) (covered : Covered graph start done (pending ++ [state]))
    (miss : ¬graph.hit state) (edges : ∀ destination, graph.edge state destination → destination ∈ children) :
    Covered graph start (fun visited => visited = state ∨ done visited) (pending ++ children) := by
  constructor
  · rcases covered.startCovered with existing | remaining
    · exact .inl (.inr existing)
    · rcases List.mem_append.mp remaining with kept | popped
      · exact .inr (List.mem_append_left _ kept)
      · exact .inl (.inl (by simpa using popped))
  · intro visited member
    rcases member with same | previous
    · subst visited
      exact ⟨miss, fun destination edge => .inr (List.mem_append_right _ (edges destination edge))⟩
    · refine ⟨(covered.closed visited previous).1, ?_⟩
      intro destination edge
      rcases (covered.closed visited previous).2 destination edge with existing | remaining
      · exact .inl (.inr existing)
      · rcases List.mem_append.mp remaining with kept | popped
        · exact .inr (List.mem_append_left _ kept)
        · exact .inl (.inl (by simpa using popped))

theorem empty_or_append {State : Type} (states : List State) :
    states = [] ∨ ∃ prior state, states = prior ++ [state] := by
  cases reversed : states.reverse with
  | nil => exact .inl (by simpa using congrArg List.reverse reversed)
  | cons state prior =>
      exact .inr ⟨prior.reverse, state, by simpa using congrArg List.reverse reversed⟩

end ArrivalWalk

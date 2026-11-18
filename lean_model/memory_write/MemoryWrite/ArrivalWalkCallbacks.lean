/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.ArrivalWalkChildren

open Aeneas Aeneas.Std Result PtxvMemoryWrite
set_option autoImplicit false
namespace ArrivalWalk

structure View (State : Type) where
  graph : Graph State
  start : State
  seen : State → Prop

def SameGraph {State : Type} (before after : View State) : Prop :=
  after.graph = before.graph ∧ after.start = before.start

def ExpansionSound {State : Type} (graph : Graph State) (state : State)
    (expansion : arrival_walk.Expansion State) : Prop :=
  match expansion with
  | .Hit => True
  | .Miss => ¬graph.hit state ∧ ∀ destination, ¬graph.edge state destination
  | .Next destination => ¬graph.hit state ∧ ∀ next, graph.edge state next → next = destination
  | .Children frontier => ∃ states, Represents frontier states ∧ ¬graph.hit state ∧
      ∀ next, graph.edge state next → next ∈ states

structure LocalContracts {State W : Type} (walker : arrival_walk.Walker W State) (view : W → View State) (settled : W → Prop) where
  begin : ∀ world, walker.begin world ⦃ initial result =>
    (view result).seen = (fun _ => False) ∧ initial = (view result).start ⦄
  fresh : ∀ world state, walker.fresh world state ⦃ fresh result =>
    SameGraph (view world) (view result) ∧
    if fresh then (view result).seen = (fun visited => visited = state ∨ (view world).seen visited)
    else (view result).seen = (view world).seen ∧ (view world).seen state ⦄
  expand : ∀ world state, walker.expand world state ⦃ expansion result =>
    SameGraph (view world) (view result) ∧ (view result).seen = (view world).seen ∧
      ExpansionSound (view world).graph state expansion ⦄
  retry : ∀ world, walker.retry world ⦃ again result =>
    again = true ∨ (view result = view world ∧ settled result) ⦄

def Invariant {State W : Type} (view : W → View State) (world : W)
    (pending : arrival_walk.Frontier State) (started : Bool) : Prop :=
  ∃ states, Represents pending states ∧
    if started then Covered (view world).graph (view world).start (view world).seen states else states = []

def NoHit {State : Type} (view : View State) : Prop :=
  ∀ state, Reaches view.graph view.start state → ¬view.graph.hit state

theorem Invariant.begin {State W : Type} (view : W → View State) (world : W)
    (pending : arrival_walk.Frontier State) (initial : State)
    (rep : Represents pending [initial]) (fresh : (view world).seen = fun _ => False)
    (start : initial = (view world).start) : Invariant view world pending true := by
  refine ⟨[initial], rep, ?_⟩
  simp only [ite_true, fresh]
  constructor
  · exact .inr (by simp [start])
  · simp

theorem Invariant.completed {State W : Type} (view : W → View State) (world : W)
    (states : List State)
    (covered : Covered (view world).graph (view world).start (view world).seen states)
    (empty : states = []) : NoHit (view world) := by
  subst states
  exact covered.completed _ _ _

end ArrivalWalk

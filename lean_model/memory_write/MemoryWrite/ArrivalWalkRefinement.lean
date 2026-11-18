/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.ArrivalWalkLoop

open Aeneas Aeneas.Std Result PtxvMemoryWrite
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace ArrivalWalk

theorem Reaches.trans {State : Type} {graph : Graph State} {start middle last : State}
    (first : Reaches graph start middle) (second : Reaches graph middle last) : Reaches graph start last := by
  induction second with
  | start => exact first
  | next _ edge ih => exact .next ih edge

structure LocalSimulation {Concrete State : Type} (concrete : Graph Concrete) (initial : Concrete)
    (abstract : View State) where
  relates : Concrete → State → Prop
  entry : relates initial abstract.start
  step : ∀ before after state, concrete.edge before after → relates before state →
    ∃ next, Reaches abstract.graph state next ∧ relates after next
  observe : ∀ value state, concrete.hit value → relates value state →
    ∃ observation, Reaches abstract.graph state observation ∧ abstract.graph.hit observation

theorem LocalSimulation.path {Concrete State : Type} (concrete : Graph Concrete) (initial final : Concrete)
    (abstract : View State) (simulation : LocalSimulation concrete initial abstract)
    (path : Reaches concrete initial final) :
    ∃ state, Reaches abstract.graph abstract.start state ∧ simulation.relates final state := by
  induction path with
  | start => exact ⟨abstract.start, .start, simulation.entry⟩
  | next _ edge ih =>
    obtain ⟨before, path, relates⟩ := ih
    obtain ⟨after, steps, related⟩ := simulation.step _ _ before edge relates
    exact ⟨after, path.trans steps, related⟩

theorem LocalSimulation.no_hit {Concrete State : Type} (concrete : Graph Concrete) (initial : Concrete)
    (abstract : View State) (simulation : LocalSimulation concrete initial abstract) (miss : NoHit abstract) :
    ∀ final, Reaches concrete initial final → ¬concrete.hit final := by
  intro final path hit
  obtain ⟨state, path, relates⟩ := simulation.path concrete initial final abstract path
  obtain ⟨observation, future, found⟩ := simulation.observe final state hit relates
  exact miss observation (path.trans future) found

@[step]
theorem may_arrive_catches_every_locally_simulated_path {Concrete State W : Type}
    (walker : arrival_walk.Walker W State) (view : W → View State) (settled : W → Prop) (contracts : LocalContracts walker view settled)
    (world : W) (budget : Usize) (concrete : Graph Concrete) (start : Concrete) :
    arrival_walk.may_arrive walker world budget ⦃ possible result =>
      (settled result → LocalSimulation concrete start (view result)) →
      (∃ final, Reaches concrete start final ∧ concrete.hit final) → possible = true ⦄ := by
  apply WP.spec_mono (rust_may_arrive walker view settled contracts world budget)
  intro result sound simulation reached
  rcases result with ⟨possible, finalWorld⟩
  cases possible with
  | true => rfl
  | false =>
    obtain ⟨final, path, hit⟩ := reached
    have completion := sound rfl
    exact False.elim ((simulation completion.1).no_hit concrete start (view finalWorld) completion.2 final path hit)

end ArrivalWalk

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.ArrivalWalkCallbacks
import Lean.Util.CollectAxioms

open Aeneas Aeneas.Std Result PtxvMemoryWrite
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace RetryWitness

def worldView (world : Bool × Bool) : ArrivalWalk.View Unit :=
  ⟨⟨fun _ => False, fun _ _ => False⟩, (), fun _ => world.1 = true⟩

def settled (world : Bool × Bool) : Prop := world.2 = true

def walker : arrival_walk.Walker (Bool × Bool) Unit where
  begin := fun _ => .ok ((), (false, false))
  fresh := fun world _ => .ok (!world.1, (true, world.2))
  expand := fun world _ => .ok (.Miss, world)
  retry := fun world => .ok (false, (world.1, true))

theorem local_contracts : ArrivalWalk.LocalContracts walker worldView settled := by
  constructor
  · intro world
    simp only [walker, WP.spec_ok, worldView]
    refine ⟨?_, trivial⟩
    funext state
    simp
  · rintro ⟨seen, stable⟩ ⟨⟩
    cases seen <;> cases stable <;>
      simp [walker, WP.spec_ok, worldView, ArrivalWalk.SameGraph]
  · intro world state
    simp [walker, WP.spec_ok, worldView, ArrivalWalk.SameGraph, ArrivalWalk.ExpansionSound]
  · intro world
    simp [walker, WP.spec_ok, worldView, settled]

def empty : arrival_walk.Frontier Unit := ⟨alloc.vec.Vec.new _, 0#usize⟩

theorem valid_exhausted_frontier : ArrivalWalk.Invariant worldView (true, false) empty true := by
  refine ⟨[], ?_, ?_⟩
  · simp [ArrivalWalk.Represents, empty, alloc.vec.Vec.new, alloc.vec.Vec.from_val]
  · simp only [ite_true, worldView]
    constructor
    · exact .inl trivial
    · simp

theorem spec_returns {α : Type} (computation : Result α) (post : α → Prop)
    (specification : WP.spec computation post) : ∃ value, computation = .ok value ∧ post value := by
  cases computation with
  | ret value => exact ⟨value, rfl, (WP.spec_ok value).mp specification⟩
  | div => simp only [WP.spec_div] at specification
  | vis effect next => simp only [WP.spec_vis] at specification

theorem actual_body_returns_unsettled_false :
    arrival_walk.may_arrive_loop.body walker (true, false) 1#usize empty true =
      .ok (.done (false, (true, false))) := by
  have specified : arrival_walk.may_arrive_loop.body walker (true, false) 1#usize empty true
      ⦃ result => result = .done (false, (true, false)) ⦄ := by
    unfold arrival_walk.may_arrive_loop.body
    have positive : 1#usize > 0#usize := by scalar_tac
    simp only [positive, ite_true]
    step with Usize.sub_spec (x := 1#usize) (y := 1#usize) (by scalar_tac) as ⟨remaining, _⟩
    have represented : ArrivalWalk.Represents empty [] := by
      simp [ArrivalWalk.Represents, empty, alloc.vec.Vec.new, alloc.vec.Vec.from_val]
    step with ArrivalWalk.rust_pop_empty empty represented as ⟨value, frontier, valueEq, _⟩
    simp [valueEq]
  obtain ⟨result, executed, equal⟩ := spec_returns _ _ specified
  simpa only [equal] using executed

theorem actual_false_result_has_no_settlement : ¬settled (true, false) := by simp [settled]

end RetryWitness

open Lean Elab Command
run_cmd do
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  for declaration in [``RetryWitness.local_contracts, ``RetryWitness.valid_exhausted_frontier,
      ``RetryWitness.actual_body_returns_unsettled_false, ``RetryWitness.actual_false_result_has_no_settlement] do
    let assumptions ← collectAxioms declaration
    for assumption in assumptions do
      unless assumption ∈ allowed do throwError "Unexpected witness assumption {assumption}"
    logInfo m!"retry witness {declaration}: {assumptions}"

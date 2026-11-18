/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.SparseClock

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.SparseClock

structure State where
  base : List Entry
  own : Entry

def State.project (state : State) : Nat → Nat := value state.base state.own

def State.WellFormed (state : State) : Prop := Sorted state.base

def relocate (state : State) (actor : Nat) : State :=
  if state.own.2 ≠ 0 ∧ state.own.1 ≠ actor then
    ⟨overlay state.base state.own, (0, 0)⟩
  else state

def tick (state : State) (actor : Nat) : State :=
  let prepared := relocate state actor
  ⟨prepared.base, (actor, prepared.project actor + 1)⟩

theorem relocation_preserves_projection (state : State) (actor : Nat) :
    (relocate state actor).project = state.project := by
  unfold relocate
  split
  · change value (overlay state.base state.own) (0, 0) = state.project
    funext query
    change max (entriesValue (overlay state.base state.own) query)
      (if 0 = query then 0 else 0) = value state.base state.own query
    simpa only [ite_self, Nat.max_zero] using congrFun (overlay_value state.base state.own) query
  · rfl

theorem relocation_preserves_sortedness (state : State) (actor : Nat)
    (formed : state.WellFormed) : (relocate state actor).WellFormed := by
  unfold relocate
  split
  · exact overlay_sorted state.base state.own formed
  · exact formed

theorem relocated_own_belongs_to_actor (state : State) (actor : Nat) :
    (relocate state actor).own.2 = 0 ∨ (relocate state actor).own.1 = actor := by
  unfold relocate
  split
  · exact Or.inl rfl
  · omega

theorem tick_preserves_sortedness (state : State) (actor : Nat) (formed : state.WellFormed) :
    (tick state actor).WellFormed := relocation_preserves_sortedness state actor formed

theorem tick_increments_exactly_one_actor (state : State) (actor : Nat) :
    (tick state actor).project =
      BarrierReduction.update state.project actor (state.project actor + 1) := by
  have relocated := relocation_preserves_projection state actor
  have owner := relocated_own_belongs_to_actor state actor
  funext query
  by_cases same : query = actor
  · subst query
    change max (entriesValue (relocate state actor).base actor)
      (if actor = actor then (relocate state actor).project actor + 1 else 0) =
      if actor = actor then state.project actor + 1 else state.project actor
    simp only [ite_true]
    have dominates : entriesValue (relocate state actor).base actor ≤
        (relocate state actor).project actor := Nat.le_max_left _ _
    rw [relocated] at dominates ⊢
    exact Nat.max_eq_right (by omega)
  · have reverse : actor ≠ query := Ne.symm same
    have oldOwn : (if (relocate state actor).own.1 = query then
        (relocate state actor).own.2 else 0) = 0 := by
      rcases owner with zero | actorOwn
      · simp [zero]
      · simp [actorOwn, reverse]
    have before : state.project query = entriesValue (relocate state actor).base query := by
      rw [← relocated]
      simp [State.project, value, oldOwn]
    change max (entriesValue (relocate state actor).base query)
      (if actor = query then (relocate state actor).project actor + 1 else 0) =
      if query = actor then state.project actor + 1 else state.project query
    simpa only [reverse, same, ite_false, Nat.max_zero] using before.symm

end MemoryWrite.SparseClock

namespace MemoryWrite.SparseClock

def emptyState : State := ⟨[], (0, 0)⟩
def singleState (actor clock : Nat) : State := ⟨[(actor, clock)], (0, 0)⟩

def joinState (left right : State) (sharedBase : Bool) : State :=
  ⟨applyPlan left.base right.base (joinPlan left.base left.own right.base right.own sharedBase), left.own⟩

inductive Constructed : State → Prop
  | empty : Constructed emptyState
  | single (actor clock : Nat) : Constructed (singleState actor clock)
  | joined {left right : State} (hl : Constructed left) (hr : Constructed right) (sharedBase : Bool) :
      Constructed (joinState left right sharedBase)
  | ticked {state : State} (formed : Constructed state) (actor : Nat) : Constructed (tick state actor)

theorem constructed_clocks_have_sorted_bases (state : State) (constructed : Constructed state) :
    state.WellFormed := by
  induction constructed with
  | empty => simp [emptyState, State.WellFormed, Sorted]
  | single actor clock => simp [singleState, State.WellFormed, Sorted]
  | joined hl hr sharedBase ihl ihr => exact join_plan_sorted _ _ _ _ sharedBase ihl ihr
  | ticked formed actor ih => exact tick_preserves_sortedness _ actor ih

end MemoryWrite.SparseClock

namespace MemoryWrite.SparseClock

theorem replacing_own_preserves_sorted_base_for_any_timestamp
    (state : State) (actor nextTimestamp : Nat) (formed : state.WellFormed) :
    (State.mk (relocate state actor).base (actor, nextTimestamp)).WellFormed :=
  relocation_preserves_sortedness state actor formed

end MemoryWrite.SparseClock

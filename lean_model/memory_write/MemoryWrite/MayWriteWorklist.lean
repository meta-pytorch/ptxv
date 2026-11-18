/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWritePaths

set_option autoImplicit false

namespace MemoryWrite.MayWrite

structure WorkState where
  solution : Solution
  pending : List Nat

def Grows (before after : Solution) : Prop :=
  (∀ pc, before.present pc → after.present pc) ∧
  ∀ pc register, before.incoming pc register → after.incoming pc register

def StableWhenUnqueued (before after : WorkState) : Prop :=
  ∀ pc, pc ∉ after.pending → pc ∉ before.pending ∧
    (after.solution.present pc ↔ before.solution.present pc) ∧
    ∀ register, after.solution.incoming pc register ↔ before.solution.incoming pc register

def NeedsQueue (state : WorkState) (target : Nat) (outgoing : Nat → Prop) : Prop :=
  ¬state.solution.present target ∨
  ∃ register, outgoing register ∧ ¬state.solution.incoming target register

noncomputable def transfer (state : WorkState) (target : Nat) (outgoing : Nat → Prop) :
    WorkState := by
  classical
  exact {
    solution := {
      present pc := state.solution.present pc ∨ pc = target
      incoming pc register := state.solution.incoming pc register ∨
        pc = target ∧ outgoing register }
    pending := if NeedsQueue state target outgoing then target :: state.pending else state.pending }

theorem transfer_grows (state : WorkState) (target : Nat) (outgoing : Nat → Prop) :
    Grows state.solution (transfer state target outgoing).solution := by
  constructor <;> intros <;> apply Or.inl <;> assumption

theorem transfer_stable_when_unqueued (state : WorkState) (target : Nat)
    (outgoing : Nat → Prop) : StableWhenUnqueued state (transfer state target outgoing) := by
  classical
  intro pc absent
  by_cases needs : NeedsQueue state target outgoing
  · have pcNe : pc ≠ target := by simpa [transfer, needs] using (fun h => absent (by simp [transfer, needs, h]))
    have oldAbsent : pc ∉ state.pending := by simpa [transfer, needs, pcNe] using absent
    exact ⟨oldAbsent, by simp [transfer, pcNe], by intro register; simp [transfer, pcNe]⟩
  · have targetPresent : state.solution.present target := by
      by_contra missing
      exact needs (Or.inl missing)
    have outgoingIncluded : ∀ register, outgoing register → state.solution.incoming target register := by
      intro register written
      by_contra missing
      exact needs (Or.inr ⟨register, written, missing⟩)
    refine ⟨by simpa [transfer, needs] using absent, ?_, ?_⟩
    · simp only [transfer]
      exact ⟨fun h => h.elim id (fun equal => equal ▸ targetPresent), Or.inl⟩
    · intro register
      simp only [transfer]
      constructor
      · rintro (known | ⟨equal, written⟩)
        · exact known
        · subst pc; exact outgoingIncluded register written
      · exact Or.inl

theorem transfer_contains (state : WorkState) (target : Nat) (outgoing : Nat → Prop) :
    (transfer state target outgoing).solution.present target ∧
    ∀ register, outgoing register →
      (transfer state target outgoing).solution.incoming target register := by
  exact ⟨Or.inr rfl, fun _ written => Or.inr ⟨rfl, written⟩⟩

theorem Grows.trans {first second third : Solution}
    (left : Grows first second) (right : Grows second third) : Grows first third :=
  ⟨fun pc known => right.1 pc (left.1 pc known),
    fun pc register known => right.2 pc register (left.2 pc register known)⟩

theorem StableWhenUnqueued.trans {first second third : WorkState}
    (left : StableWhenUnqueued first second) (right : StableWhenUnqueued second third) :
    StableWhenUnqueued first third := by
  intro pc absent
  obtain ⟨absentSecond, presentSecond, incomingSecond⟩ := right pc absent
  obtain ⟨absentFirst, presentFirst, incomingFirst⟩ := left pc absentSecond
  exact ⟨absentFirst, presentSecond.trans presentFirst,
    fun register => (incomingSecond register).trans (incomingFirst register)⟩

noncomputable def scan (state : WorkState) (successors : List Nat) (outgoing : Nat → Prop) :
    WorkState := successors.foldl (fun state target => transfer state target outgoing) state

theorem scan_grows (state : WorkState) (successors : List Nat) (outgoing : Nat → Prop) :
    Grows state.solution (scan state successors outgoing).solution := by
  induction successors generalizing state with
  | nil => exact ⟨fun _ known => known, fun _ _ known => known⟩
  | cons target remaining ih =>
      exact (transfer_grows state target outgoing).trans
        (ih (transfer state target outgoing))

theorem scan_stable_when_unqueued (state : WorkState) (successors : List Nat)
    (outgoing : Nat → Prop) : StableWhenUnqueued state (scan state successors outgoing) := by
  induction successors generalizing state with
  | nil => exact fun _ absent => ⟨absent, Iff.rfl, fun _ => Iff.rfl⟩
  | cons target remaining ih =>
      exact (transfer_stable_when_unqueued state target outgoing).trans
        (ih (transfer state target outgoing))

theorem scan_contains (state : WorkState) (successors : List Nat) (outgoing : Nat → Prop)
    {target : Nat} (member : target ∈ successors) :
    (scan state successors outgoing).solution.present target ∧
    ∀ register, outgoing register →
      (scan state successors outgoing).solution.incoming target register := by
  induction successors generalizing state with
  | nil => simp at member
  | cons successor remaining ih =>
      rcases List.mem_cons.mp member with equal | member
      · subst target
        have included := transfer_contains state successor outgoing
        have growth := scan_grows (transfer state successor outgoing) remaining outgoing
        exact ⟨growth.1 successor included.1,
          fun register written => growth.2 successor register (included.2 register written)⟩
      · exact ih (transfer state successor outgoing) member

def Settled (graph : Graph) (solution : Solution) (pc : Nat) : Prop :=
  solution.present pc → graph.enabled pc → ∀ successor,
    graph.edge pc successor → solution.present successor ∧
    ∀ register, solution.incoming pc register ∨ graph.writes pc register →
      solution.incoming successor register

def WorkInvariant (graph : Graph) (state : WorkState) : Prop :=
  ∀ pc, pc ∉ state.pending → Settled graph state.solution pc

theorem settled_preserved_when_unqueued (graph : Graph) {before after : WorkState}
    (growth : Grows before.solution after.solution)
    (stable : StableWhenUnqueued before after) {pc : Nat} (absent : pc ∉ after.pending)
    (settled : Settled graph before.solution pc) : Settled graph after.solution pc := by
  obtain ⟨_, presentEqual, incomingEqual⟩ := stable pc absent
  intro present enabled successor edge
  obtain ⟨successorPresent, included⟩ := settled (presentEqual.mp present) enabled successor edge
  exact ⟨growth.1 successor successorPresent, fun register written =>
    growth.2 successor register (included register (written.imp (incomingEqual register).mp id))⟩


theorem propagation_preserves_invariant (graph : Graph) (state : WorkState)
    (current : Nat) (remaining : List Nat) (result : WorkState) (outgoing : Nat → Prop)
    (pending : state.pending = current :: remaining)
    (invariant : WorkInvariant graph state)
    (outgoingCovers : ∀ register,
      state.solution.incoming current register ∨ graph.writes current register → outgoing register)
    (growth : Grows state.solution result.solution)
    (stable : StableWhenUnqueued {state with pending := remaining} result)
    (propagated : ∀ successor, graph.edge current successor → result.solution.present successor ∧
      ∀ register, outgoing register → result.solution.incoming successor register) :
    WorkInvariant graph result := by
  intro pc absent
  obtain ⟨absentOld, presentEqual, incomingEqual⟩ := stable pc absent
  by_cases same : pc = current
  · subst pc
    intro _ _ successor edge
    have includes := propagated successor edge
    exact ⟨includes.1, fun register written =>
      includes.2 register (outgoingCovers register (written.imp (incomingEqual register).mp id))⟩
  · apply settled_preserved_when_unqueued graph
      (before := {state with pending := remaining}) growth stable absent
    apply invariant pc
    simpa only [pending, List.mem_cons, not_or] using And.intro same absentOld

theorem enabled_step_preserves_invariant (graph : Graph) (state : WorkState)
    (current : Nat) (remaining successors : List Nat)
    (pending : state.pending = current :: remaining)
    (invariant : WorkInvariant graph state)
    (successorComplete : ∀ successor, graph.edge current successor ↔ successor ∈ successors) :
    WorkInvariant graph (scan {state with pending := remaining} successors
      (fun register => state.solution.incoming current register ∨ graph.writes current register)) := by
  let popped := {state with pending := remaining}
  let outgoing := fun register => state.solution.incoming current register ∨ graph.writes current register
  let result := scan popped successors outgoing
  have growth := scan_grows popped successors outgoing
  have stable := scan_stable_when_unqueued popped successors outgoing
  intro pc absent
  obtain ⟨absentOld, presentEqual, incomingEqual⟩ := stable pc absent
  by_cases same : pc = current
  · subst pc
    intro _ _ successor edge
    have includes := scan_contains popped successors outgoing ((successorComplete successor).mp edge)
    exact ⟨includes.1, fun register written =>
      includes.2 register (written.imp (incomingEqual register).mp id)⟩
  · apply settled_preserved_when_unqueued graph growth stable absent
    apply invariant pc
    simpa only [pending, List.mem_cons, not_or] using And.intro same absentOld

theorem disabled_step_preserves_invariant (graph : Graph) (state : WorkState)
    (current : Nat) (remaining : List Nat) (pending : state.pending = current :: remaining)
    (disabled : ¬graph.enabled current) (invariant : WorkInvariant graph state) :
    WorkInvariant graph {state with pending := remaining} := by
  intro pc absent present enabled
  by_cases same : pc = current
  · subst pc; exact False.elim (disabled enabled)
  · exact invariant pc (by simpa only [pending, List.mem_cons, not_or] using And.intro same absent)
      present enabled

theorem exhausted_worklist_is_closed (graph : Graph) (state : WorkState)
    (invariant : WorkInvariant graph state) (empty : state.pending = []) :
    Closed graph state.solution := by
  intro pc present enabled
  exact invariant pc (by simp [empty]) present enabled


def initialState (start : Nat) : WorkState where
  solution := {present pc := pc = start, incoming _ _ := False}
  pending := [start]

theorem initial_invariant (graph : Graph) (start : Nat) :
    WorkInvariant graph (initialState start) := by
  intro pc absent present
  subst pc
  exact False.elim (absent (by simp [initialState]))

inductive WorkStep (graph : Graph) : WorkState → WorkState → Prop
  | disabled {state : WorkState} {current : Nat} {remaining : List Nat} :
      state.pending = current :: remaining → ¬graph.enabled current →
      WorkStep graph state {state with pending := remaining}
  | enabled {state : WorkState} {current : Nat} {remaining successors : List Nat} :
      state.pending = current :: remaining → graph.enabled current →
      (∀ successor, graph.edge current successor ↔ successor ∈ successors) →
      WorkStep graph state (scan {state with pending := remaining} successors
        (fun register => state.solution.incoming current register ∨ graph.writes current register))

inductive WorkExecution (graph : Graph) : WorkState → WorkState → Prop
  | refl (state : WorkState) : WorkExecution graph state state
  | next {first middle last : WorkState} : WorkStep graph first middle →
      WorkExecution graph middle last → WorkExecution graph first last

theorem WorkStep.preserves_invariant {graph : Graph} {before after : WorkState}
    (step : WorkStep graph before after) (invariant : WorkInvariant graph before) :
    WorkInvariant graph after := by
  cases step with
  | disabled pending disabled =>
      exact disabled_step_preserves_invariant graph _ _ _ pending disabled invariant
  | enabled pending _ complete =>
      exact enabled_step_preserves_invariant graph _ _ _ _ pending invariant complete

theorem WorkStep.grows {graph : Graph} {before after : WorkState}
    (step : WorkStep graph before after) : Grows before.solution after.solution := by
  cases step with
  | disabled => exact ⟨fun _ known => known, fun _ _ known => known⟩
  | @enabled current remaining successors _ _ _ =>
      exact scan_grows {before with pending := remaining} successors
        (fun register => before.solution.incoming current register ∨ graph.writes current register)

theorem WorkExecution.preserves_invariant {graph : Graph} {before after : WorkState}
    (execution : WorkExecution graph before after) (invariant : WorkInvariant graph before) :
    WorkInvariant graph after := by
  induction execution with
  | refl => exact invariant
  | next step _ ih => exact ih (step.preserves_invariant invariant)

theorem WorkExecution.grows {graph : Graph} {before after : WorkState}
    (execution : WorkExecution graph before after) : Grows before.solution after.solution := by
  induction execution with
  | refl => exact ⟨fun _ known => known, fun _ _ known => known⟩
  | next step _ ih => exact step.grows.trans ih

theorem completed_worklist_covers_every_path_write (graph : Graph) (start : Nat)
    (result : WorkState) (execution : WorkExecution graph (initialState start) result)
    (empty : result.pending = []) {pc register : Nat}
    (written : WrittenBefore graph start pc register) : result.solution.incoming pc register := by
  have invariant := execution.preserves_invariant (initial_invariant graph start)
  have startPresent := execution.grows.1 start (show (initialState start).solution.present start from rfl)
  exact closed_covers_path_writes graph start result.solution startPresent
    (exhausted_worklist_is_closed graph result invariant empty) written

end MemoryWrite.MayWrite

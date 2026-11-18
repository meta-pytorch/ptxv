/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.Semantics

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution

open MemorySafety

structure Execution (Native : Type) where
  initial : Option (List Native)
  actors : Native → List Nat
  moves : Native → Nat → Option (List (Event × Native))
  observe : Native → Bool

structure State (Native : Type) where
  execution : Native
  memory : MemorySafety.State
  deriving DecidableEq

def initial {Native : Type} (program : MemorySafety.Program)
    (execution : Execution Native) : Option (List (State Native)) := do
  let memory ← MemorySafety.initial program
  let states ← execution.initial
  some (states.map (fun state => ⟨state, memory⟩))

def applyMove {Native : Type} (program : MemorySafety.Program)
    (before : State Native) (move : Event × Native) : State Native :=
  ⟨move.2, updateMemory program before.memory move.1⟩

def gather {Native : Type} (program : MemorySafety.Program)
    (execution : Execution Native) (state : State Native) : List Nat → Option (List (State Native))
  | [] => some []
  | actor :: actors => do
    let moves ← execution.moves state.execution actor
    let rest ← gather program execution state actors
    some (moves.map (applyMove program state) ++ rest)

def system {Native : Type} (program : MemorySafety.Program)
    (execution : Execution Native) : Certificate.System (State Native) :=
  ⟨initial program execution,
    fun state => gather program execution state (execution.actors state.execution)⟩

def good {Native : Type} (execution : Execution Native) (state : State Native) : Bool :=
  !state.memory.bad && execution.observe state.execution

def verify {Native : Type} [DecidableEq Native] (program : MemorySafety.Program)
    (execution : Execution Native) (certificate : List (State Native)) : Bool :=
  Certificate.verify (system program execution) (good execution) certificate

def Step {Native : Type} (program : MemorySafety.Program) (execution : Execution Native)
    (before : State Native) (actor : Nat) (event : Event) (after : State Native) : Prop :=
  actor ∈ execution.actors before.execution ∧
    match execution.moves before.execution actor with
    | none => True
    | some moves => ∃ move ∈ moves, move.1 = event ∧ applyMove program before move = after

inductive Reachable {Native : Type} (program : MemorySafety.Program)
    (execution : Execution Native) : State Native → Prop
  | initial {state} : Certificate.Initial (system program execution) state →
      Reachable program execution state
  | step {before after actor event} : Reachable program execution before →
      Step program execution before actor event after → Reachable program execution after

theorem gather_actor {Native : Type} (program : MemorySafety.Program)
    (execution : Execution Native) (state : State Native) (actors : List Nat)
    (next : List (State Native)) (computed : gather program execution state actors = some next)
    (actor : Nat) (enabled : actor ∈ actors) :
    ∃ moves, execution.moves state.execution actor = some moves ∧
      ∀ move ∈ moves, applyMove program state move ∈ next := by
  induction actors generalizing next with
  | nil => simp at enabled
  | cons first rest ih =>
    cases head : execution.moves state.execution first with
    | none => simp [gather, head] at computed
    | some moves =>
      cases tail : gather program execution state rest with
      | none => simp [gather, head, tail] at computed
      | some others =>
        have equal : moves.map (applyMove program state) ++ others = next := by
          simpa [gather, head, tail] using computed
        subst next
        rcases List.mem_cons.mp enabled with same | enabled
        · subst actor
          exact ⟨moves, head, fun move member => List.mem_append_left _ (List.mem_map.mpr ⟨move, member, rfl⟩)⟩
        · obtain ⟨choices, selected, included⟩ := ih others tail enabled
          exact ⟨choices, selected, fun move member => List.mem_append_right _ (included move member)⟩

theorem gather_fault {Native : Type} (program : MemorySafety.Program)
    (execution : Execution Native) (state : State Native) (actors : List Nat)
    (actor : Nat) (enabled : actor ∈ actors)
    (unknown : execution.moves state.execution actor = none) :
    gather program execution state actors = none := by
  cases result : gather program execution state actors with
  | none => rfl
  | some next =>
    obtain ⟨moves, computed, _⟩ := gather_actor program execution state actors next result actor enabled
    rw [unknown] at computed
    cases computed

theorem step_graph {Native : Type} (program : MemorySafety.Program)
    (execution : Execution Native) (before after : State Native) (actor : Nat) (event : Event)
    (transition : Step program execution before actor event after) :
    Certificate.Step (system program execution) before after := by
  rcases transition with ⟨enabled, transition⟩
  cases result : gather program execution before (execution.actors before.execution) with
  | none => simp [Certificate.Step, system, result]
  | some next =>
    obtain ⟨moves, computed, included⟩ := gather_actor program execution before _ next result actor enabled
    rw [computed] at transition
    obtain ⟨move, member, _, equal⟩ := transition
    simp only [Certificate.Step, system, result]
    rw [← equal]
    exact included move member

theorem reachable_graph {Native : Type} (program : MemorySafety.Program)
    (execution : Execution Native) {state : State Native}
    (reachable : Reachable program execution state) :
    Certificate.Reachable (system program execution) state := by
  induction reachable with
  | initial initial => exact .initial initial
  | @step before after actor event reachable transition ih =>
    exact .step ih (step_graph program execution before after actor event transition)

theorem verified_reachable {Native : Type} [DecidableEq Native] (program : MemorySafety.Program)
    (execution : Execution Native) (certificate : List (State Native))
    (accepted : verify program execution certificate = true)
    {state : State Native} (reachable : Reachable program execution state) :
    state ∈ certificate ∧ state.memory.bad = false ∧ execution.observe state.execution = true := by
  have checked := Certificate.verified_reachable (system program execution) (good execution)
    certificate accepted (reachable_graph program execution reachable)
  refine ⟨checked.1, ?_⟩
  simpa only [good, Bool.and_eq_true, Bool.not_eq_true'] using checked.2

theorem verified_actor {Native : Type} [DecidableEq Native] (program : MemorySafety.Program)
    (execution : Execution Native) (certificate : List (State Native))
    (accepted : verify program execution certificate = true)
    {state : State Native} (reachable : Reachable program execution state)
    (actor : Nat) (enabled : actor ∈ execution.actors state.execution) :
    ∃ moves, execution.moves state.execution actor = some moves ∧
      ∀ move ∈ moves, applyMove program state move ∈ certificate := by
  have present := (verified_reachable program execution certificate accepted reachable).1
  obtain ⟨_, next, computed, closed⟩ := Certificate.verified_vertex
    (system program execution) (good execution) certificate accepted state present
  obtain ⟨moves, selected, included⟩ := gather_actor program execution state _ next computed actor enabled
  exact ⟨moves, selected, fun move member => closed _ (included move member)⟩

theorem verified_step_safe {Native : Type} [DecidableEq Native] (program : MemorySafety.Program)
    (execution : Execution Native) (certificate : List (State Native))
    (accepted : verify program execution certificate = true)
    {before after : State Native} {actor : Nat} {event : Event}
    (reachable : Reachable program execution before)
    (transition : Step program execution before actor event after) :
    EventSafe program before.memory event ∧ after.memory.bad = false ∧
      execution.observe after.execution = true := by
  obtain ⟨moves, computed, _⟩ := verified_actor program execution certificate accepted reachable actor transition.1
  have clean := (verified_reachable program execution certificate accepted (.step reachable transition)).2
  have selected := transition.2
  rw [computed] at selected
  obtain ⟨move, _, eventEq, equal⟩ := selected
  have memoryClean : (updateMemory program before.memory move.1).bad = false := by
    change (applyMove program before move).memory.bad = false
    rw [equal]
    exact clean.1
  exact ⟨eventEq ▸ memory_update_safe program before.memory move.1 memoryClean, clean⟩


theorem verified_layout {Native : Type} [DecidableEq Native] (program : MemorySafety.Program)
    (execution : Execution Native) (certificate : List (State Native))
    (accepted : verify program execution certificate = true) : LayoutSafe program := by
  by_cases valid : LayoutSafe program
  · exact valid
  · simp [verify, Certificate.verify, system, initial, MemorySafety.initial, valid] at accepted

def MemorySafe {Native : Type} (program : MemorySafety.Program) (execution : Execution Native) : Prop :=
  LayoutSafe program ∧
    (∀ state, Reachable program execution state →
      state.memory.bad = false ∧ execution.observe state.execution = true ∧
      ∀ actor ∈ execution.actors state.execution, execution.moves state.execution actor ≠ none) ∧
    ∀ before after actor event, Reachable program execution before →
      Step program execution before actor event after → EventSafe program before.memory event

theorem accepted_execution_memory_safe {Native : Type} [DecidableEq Native]
    (program : MemorySafety.Program) (execution : Execution Native)
    (certificate : List (State Native))
    (accepted : verify program execution certificate = true) : MemorySafe program execution := by
  refine ⟨verified_layout program execution certificate accepted, ?_, ?_⟩
  · intro state reachable
    have checked := (verified_reachable program execution certificate accepted reachable).2
    refine ⟨checked.1, checked.2, ?_⟩
    intro actor enabled unknown
    obtain ⟨moves, computed, _⟩ := verified_actor program execution certificate accepted reachable actor enabled
    rw [unknown] at computed
    cases computed
  · intro before after actor event reachable transition
    exact (verified_step_safe program execution certificate accepted reachable transition).1

end MemorySafetyExecution

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Aeneas

set_option autoImplicit false
set_option maxHeartbeats 2000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafety.Certificate

structure System (State : Type) where
  initial : Option (List State)
  successors : State → Option (List State)

def Initial {State : Type} (system : System State) (state : State) : Prop :=
  match system.initial with
  | none => True
  | some states => state ∈ states

def Step {State : Type} (system : System State) (before after : State) : Prop :=
  match system.successors before with
  | none => True
  | some states => after ∈ states

inductive Reachable {State : Type} (system : System State) : State → Prop
  | initial {state} : Initial system state → Reachable system state
  | step {before after} : Reachable system before → Step system before after → Reachable system after

def member {State : Type} [DecidableEq State] (certificate : List State) (state : State) : Bool :=
  decide (state ∈ certificate)

def validVertex {State : Type} [DecidableEq State] (system : System State)
    (good : State → Bool) (certificate : List State) (state : State) : Bool :=
  good state && match system.successors state with
  | none => false
  | some next => next.all (member certificate)

def verify {State : Type} [DecidableEq State] (system : System State)
    (good : State → Bool) (certificate : List State) : Bool :=
  match system.initial with
  | none => false
  | some states => states.all (member certificate) &&
      certificate.all (validVertex system good certificate)

inductive SearchResult (State : Type) where
  | candidate (states : List State)
  | unknown
  | exhausted

def accept {State : Type} [DecidableEq State] (system : System State)
    (good : State → Bool) : SearchResult State → Bool
  | .candidate states => verify system good states
  | .unknown | .exhausted => false

@[simp] theorem unknown_refuses {State : Type} [DecidableEq State]
    (system : System State) (good : State → Bool) : accept system good .unknown = false := rfl

@[simp] theorem exhausted_refuses {State : Type} [DecidableEq State]
    (system : System State) (good : State → Bool) : accept system good .exhausted = false := rfl

theorem verified_initial {State : Type} [DecidableEq State] (system : System State)
    (good : State → Bool) (certificate : List State) (accepted : verify system good certificate = true)
    (state : State) (initial : Initial system state) : state ∈ certificate := by
  cases found : system.initial with
  | none => simp [verify, found] at accepted
  | some states =>
    simp only [verify, found, Bool.and_eq_true, List.all_eq_true] at accepted
    have present : state ∈ states := by simpa [Initial, found] using initial
    simpa [member] using accepted.1 state present

theorem verified_vertex {State : Type} [DecidableEq State] (system : System State)
    (good : State → Bool) (certificate : List State) (accepted : verify system good certificate = true)
    (state : State) (present : state ∈ certificate) :
    good state = true ∧ ∃ next, system.successors state = some next ∧
      ∀ other ∈ next, other ∈ certificate := by
  cases found : system.initial with
  | none => simp [verify, found] at accepted
  | some states =>
    simp only [verify, found, Bool.and_eq_true, List.all_eq_true] at accepted
    have checked := accepted.2 state present
    cases successors : system.successors state with
    | none => simp [validVertex, successors] at checked
    | some next =>
      simp only [validVertex, successors, Bool.and_eq_true, List.all_eq_true] at checked
      exact ⟨checked.1, next, rfl, fun other membership => by
        simpa [member] using checked.2 other membership⟩

theorem verified_reachable {State : Type} [DecidableEq State] (system : System State)
    (good : State → Bool) (certificate : List State) (accepted : verify system good certificate = true)
    {state : State} (reachable : Reachable system state) :
    state ∈ certificate ∧ good state = true := by
  induction reachable with
  | @initial state initial =>
    have present := verified_initial system good certificate accepted state initial
    exact ⟨present, (verified_vertex system good certificate accepted state present).1⟩
  | @step before after reachable transition ih =>
    obtain ⟨_, next, found, closed⟩ := verified_vertex system good certificate accepted before ih.1
    have inside : after ∈ next := by simpa [Step, found] using transition
    have present := closed after inside
    exact ⟨present, (verified_vertex system good certificate accepted after present).1⟩

end MemorySafety.Certificate

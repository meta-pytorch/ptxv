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

namespace MemorySafety

structure Move where
  event : Event
  pcs : List Nat
  deriving DecidableEq, Repr

def applyMove (program : Program) (state : State) (move : Move) : State :=
  let updated := updateMemory program state move.event
  if updated.bad then updated else { updated with pcs := move.pcs }

def threadMoves (program : Program) (state : State) (thread : Nat) : Option (List Move) :=
  match program.threads[thread]?, state.pcs[thread]? with
  | some code, some pc =>
    if pc > code.length then none
    else if pc = code.length then some []
    else
      match code[pc]? with
      | none => none
      | some instruction =>
        match instruction with
        | .unsupported => none
        | .skip => some [⟨.control, state.pcs.set thread (pc + 1)⟩]
        | .halt => some [⟨.control, state.pcs.set thread code.length⟩]
        | .choice targets =>
          if targets = [] then none
          else if targets.all (fun target => decide (target ≤ code.length)) then
            some (targets.map (fun target => ⟨.control, state.pcs.set thread target⟩))
          else none
        | .access request => some [⟨.access thread request, state.pcs.set thread (pc + 1)⟩]
        | .free allocation => some [⟨.free thread allocation, state.pcs.set thread (pc + 1)⟩]
  | _, _ => none

def gatherMoves (program : Program) (state : State) : List Nat → Option (List Move)
  | [] => some []
  | thread :: threads => do
    let first ← threadMoves program state thread
    let rest ← gatherMoves program state threads
    some (first ++ rest)

def moves (program : Program) (state : State) : Option (List Move) :=
  if state.pcs.length ≠ program.threads.length ∨ state.live.length ≠ program.allocations.length then none
  else if state.bad then some []
  else gatherMoves program state (List.range program.threads.length)

def successors (program : Program) (state : State) : Option (List State) :=
  (moves program state).map (fun choices => choices.map (applyMove program state))

def system (program : Program) : Certificate.System State :=
  ⟨(initial program).map (fun start => [start]), successors program⟩

def good (state : State) : Bool := !state.bad

def verify (program : Program) (certificate : List State) : Bool :=
  Certificate.verify (system program) good certificate

def accept (program : Program) : Certificate.SearchResult State → Bool :=
  Certificate.accept (system program) good

-- Unknown transitions permit arbitrary behavior; certificate checking refuses them.
def Step (program : Program) (before : State) (event : Event) (after : State) : Prop :=
  match moves program before with
  | none => True
  | some choices => ∃ move ∈ choices, move.event = event ∧ applyMove program before move = after

inductive Reachable (program : Program) : State → Prop
  | initial : Reachable program (initialState program)
  | step {before event after} : Reachable program before → Step program before event after → Reachable program after

theorem step_graph (program : Program) (before after : State) (event : Event)
    (transition : Step program before event after) : Certificate.Step (system program) before after := by
  cases found : moves program before with
  | none => simp [Certificate.Step, system, successors, found]
  | some choices =>
    obtain ⟨move, member, _, result⟩ : ∃ move ∈ choices,
        move.event = event ∧ applyMove program before move = after := by
      simpa [Step, found] using transition
    simp only [Certificate.Step, system, successors, found, Option.map_some]
    exact List.mem_map.mpr ⟨move, member, result⟩

theorem verified_layout (program : Program) (certificate : List State)
    (accepted : verify program certificate = true) : LayoutSafe program := by
  by_cases valid : LayoutSafe program
  · exact valid
  · simp [verify, Certificate.verify, system, initial, valid] at accepted

theorem reachable_graph (program : Program) (certificate : List State)
    (accepted : verify program certificate = true) {state : State} (reachable : Reachable program state) :
    Certificate.Reachable (system program) state := by
  induction reachable with
  | initial =>
    apply Certificate.Reachable.initial
    simp [Certificate.Initial, system, initial, verified_layout program certificate accepted]
  | @step before event after reachable transition ih =>
    exact Certificate.Reachable.step ih (step_graph program before after event transition)

theorem verified_reachable (program : Program) (certificate : List State)
    (accepted : verify program certificate = true) {state : State} (reachable : Reachable program state) :
    state ∈ certificate ∧ state.bad = false := by
  have result := Certificate.verified_reachable (system program) good certificate accepted
    (reachable_graph program certificate accepted reachable)
  exact ⟨result.1, by simpa [good] using result.2⟩

theorem apply_move_memory_bad (program : Program) (state : State) (move : Move) :
    (applyMove program state move).bad = (updateMemory program state move.event).bad := by
  unfold applyMove
  dsimp only
  split <;> rfl

theorem verified_step_safe (program : Program) (certificate : List State)
    (accepted : verify program certificate = true) {before after : State} {event : Event}
    (reachable : Reachable program before) (transition : Step program before event after) :
    EventSafe program before event ∧ after.bad = false := by
  have beforeMember := (verified_reachable program certificate accepted reachable).1
  obtain ⟨_, next, known, _⟩ :=
    Certificate.verified_vertex (system program) good certificate accepted before beforeMember
  have afterClean := (verified_reachable program certificate accepted (.step reachable transition)).2
  constructor
  · cases found : moves program before with
    | none => simp [system, successors, found] at known
    | some choices =>
      obtain ⟨move, _, eventEq, result⟩ : ∃ move ∈ choices,
          move.event = event ∧ applyMove program before move = after := by
        simpa [Step, found] using transition
      have clean : (updateMemory program before move.event).bad = false := by
        rw [← apply_move_memory_bad, result]
        exact afterClean
      rw [← eventEq]
      exact memory_update_safe program before move.event clean
  · exact afterClean

theorem every_permitted_memory_access_safe (program : Program) (certificate : List State)
    (accepted : verify program certificate = true) {before after : State}
    (thread : Nat) (request : Request) (reachable : Reachable program before)
    (transition : Step program before (.access thread request) after) :
    AccessSafe program before thread request :=
  (verified_step_safe program certificate accepted reachable transition).1

theorem every_permitted_lifetime_end_safe (program : Program) (certificate : List State)
    (accepted : verify program certificate = true) {before after : State}
    (thread allocation : Nat) (reachable : Reachable program before)
    (transition : Step program before (.free thread allocation) after) :
    FreeSafe program before thread allocation :=
  (verified_step_safe program certificate accepted reachable transition).1

@[simp] theorem unknown_search_refuses (program : Program) : accept program .unknown = false := rfl
@[simp] theorem exhausted_search_refuses (program : Program) : accept program .exhausted = false := rfl

end MemorySafety

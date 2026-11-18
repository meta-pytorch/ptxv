/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.Program

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafety

def search (program : Program) : Nat → List State → List State → Certificate.SearchResult State
  | _, seen, [] => .candidate seen
  | 0, _, _ :: _ => .exhausted
  | fuel + 1, seen, state :: frontier =>
    match successors program state with
    | none => .unknown
    | some next =>
      let fresh := next.filter (fun state => decide (state ∉ seen))
      search program fuel (seen ++ fresh) (frontier ++ fresh)

def analyze (program : Program) (budget : Nat) : Bool :=
  match initial program with
  | none => false
  | some start => accept program (search program budget [start] [start])

theorem search_acceptance_is_checked (program : Program) (budget : Nat)
    (accepted : analyze program budget = true) :
    ∃ certificate, verify program certificate = true := by
  unfold analyze at accepted
  cases found : initial program with
  | none => simp [found] at accepted
  | some start =>
    simp only [found] at accepted
    cases result : search program budget [start] [start] with
    | candidate certificate => exact ⟨certificate, by simpa [result, accept, Certificate.accept, verify] using accepted⟩
    | unknown => simp [result] at accepted
    | exhausted => simp [result] at accepted

theorem analyze_sound (program : Program) (budget : Nat) (accepted : analyze program budget = true)
    {before after : State} {event : Event} (reachable : Reachable program before)
    (transition : Step program before event after) : EventSafe program before event ∧ after.bad = false := by
  obtain ⟨certificate, checked⟩ := search_acceptance_is_checked program budget accepted
  exact verified_step_safe program certificate checked reachable transition

end MemorySafety

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.Program

set_option autoImplicit false
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafety

def MemorySafe (program : Program) : Prop :=
  LayoutSafe program ∧
  (∀ state, Reachable program state → state.bad = false) ∧
  ∀ before event after, Reachable program before → Step program before event after →
    EventSafe program before event

theorem verified_memory_safe (program : Program) (certificate : List State)
    (accepted : verify program certificate = true) : MemorySafe program := by
  refine ⟨verified_layout program certificate accepted, ?_, ?_⟩
  · intro state reachable
    exact (verified_reachable program certificate accepted reachable).2
  · intro before event after reachable transition
    exact (verified_step_safe program certificate accepted reachable transition).1

theorem memory_safe_read (program : Program) (safe : MemorySafe program)
    (before after : State) (thread : Nat) (request : Request)
    (reachable : Reachable program before)
    (transition : Step program before (.access thread request) after)
    (load : request.kind = .load) (offset : Nat) (inside : offset < request.width) :
    byteDefined program before (request.address + offset) = true :=
  safe_read_every_byte program before thread request
    (safe.2.2 before (.access thread request) after reachable transition) load offset inside

theorem memory_safe_range (program : Program) (safe : MemorySafe program)
    (before after : State) (thread : Nat) (request : Request)
    (reachable : Reachable program before)
    (transition : Step program before (.access thread request) after) :
    0 < request.width ∧ 0 < request.alignment ∧ request.address % request.alignment = 0 ∧
      request.address + request.width ≤ addressLimit ∧ LiveSpan program before request := by
  have components := access_safe_components program before thread request
    (safe.2.2 before (.access thread request) after reachable transition)
  exact ⟨components.1, components.2.1, components.2.2.1, components.2.2.2.1, components.2.2.2.2.1⟩

theorem memory_safe_races (program : Program) (safe : MemorySafe program)
    (before after : State) (thread : Nat) (request : Request)
    (reachable : Reachable program before)
    (transition : Step program before (.access thread request) after)
    (earlier : AccessRecord) (recorded : earlier ∈ before.history) :
    ¬Conflict earlier (accessRecord thread request) :=
  safe_access_conflicts_with_no_previous_access program before thread request
    (safe.2.2 before (.access thread request) after reachable transition) earlier recorded

end MemorySafety

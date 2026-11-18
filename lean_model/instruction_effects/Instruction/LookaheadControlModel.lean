/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Aeneas

set_option autoImplicit false

namespace LookaheadControl

inductive Observation where
  | at (pc : Nat)
  | fault
  deriving DecidableEq

inductive Guard where
  | known (enabled : Bool)
  | uniformUnknown
  | divergent
  deriving DecidableEq

inductive Plan where
  | next
  | jump (target : Nat)
  | alternatives (targets : List Nat)
  | giveUp
  deriving DecidableEq

def permits (pc : Nat) (plan : Plan) (observation : Observation) : Prop :=
  match plan, observation with
  | .giveUp, _ => True
  | .next, .at destination => destination = pc + 1
  | .jump target, .at destination => destination = target
  | .alternatives targets, .at destination => destination ∈ targets
  | _, .fault => False

def guardRepresents (guard : Guard) (enabled : Bool) : Prop :=
  match guard with
  | .known value => enabled = value
  | _ => True

def withFallthrough (pc : Nat) (taken : Plan) : Plan :=
  match taken with
  | .jump target => .alternatives [target, pc + 1]
  | .alternatives targets => .alternatives (targets ++ [pc + 1])
  | other => other

def guarded (pc : Nat) (guard : Guard) (taken : Plan) : Plan :=
  match guard with
  | .known false => .next
  | .known true => taken
  | .uniformUnknown => withFallthrough pc taken
  | .divergent => .giveUp

theorem with_fallthrough_preserves_taken (pc : Nat) (taken : Plan)
    (observation : Observation) (covered : permits pc taken observation) :
    permits pc (withFallthrough pc taken) observation := by
  cases taken <;> cases observation <;> simp_all [permits, withFallthrough]

theorem with_fallthrough_preserves_disabled (pc : Nat) (taken : Plan) :
    permits pc (withFallthrough pc taken) (.at (pc + 1)) := by
  cases taken <;> simp [permits, withFallthrough]

theorem guarded_preserves_concrete_step (pc : Nat) (guard : Guard) (taken : Plan)
    (enabled : Bool) (observation : Observation)
    (guardSound : guardRepresents guard enabled)
    (takenSound : permits pc taken observation) :
    permits pc (guarded pc guard taken)
      (if enabled then observation else .at (pc + 1)) := by
  cases guard with
  | known value =>
    simp only [guardRepresents] at guardSound
    subst enabled
    cases value <;> simp_all [guarded, permits]
  | uniformUnknown =>
    cases enabled
    · exact with_fallthrough_preserves_disabled pc taken
    · exact with_fallthrough_preserves_taken pc taken observation takenSound
  | divergent => simp [guarded, permits]

theorem false_guard_needs_no_target_premise (pc : Nat) (taken : Plan) :
    guarded pc (.known false) taken = .next ∧
      permits pc (guarded pc (.known false) taken) (.at (pc + 1)) := by
  simp [guarded, permits]

theorem ignoring_false_guard_can_drop_a_step (pc target : Nat)
    (different : target ≠ pc + 1) :
    ¬permits pc (.jump target) (.at (pc + 1)) := by
  simpa [permits, eq_comm] using different

end LookaheadControl

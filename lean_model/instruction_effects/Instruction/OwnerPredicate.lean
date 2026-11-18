/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.OwnerComparison

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false

namespace RegisterOwner

def immutablePredicate (front : Front) (register : Reg) : Bool :=
  match register.file with
  | .UP => decide (register.idx = 7#u16)
  | .P => match front with
    | .Sass => decide (register.idx = 7#u16)
    | .Ptx => false
  | _ => false

def predicateRead (front : Front) (values : Reg → Bool) (predicate : Pred) : Bool :=
  let value := match predicate.reg with
    | none => true
    | some register => if immutablePredicate front register then true else values register
  value != predicate.neg

def normalizedPredicate (front : Front) (predicate : Pred) : Pred :=
  match predicate.reg with
  | none => predicate
  | some register =>
    if immutablePredicate front register then { predicate with reg := none } else predicate

theorem source_normalize_predicate (front : Front) (predicate : Pred) :
    register_owner.normalize_predicate front predicate = .ok (normalizedPredicate front predicate) := by
  rcases predicate with ⟨register, neg⟩
  cases register with
  | none => simp [register_owner.normalize_predicate, normalizedPredicate]
  | some register =>
    rcases register with ⟨file, index⟩
    cases front <;> cases file <;>
      simp [register_owner.normalize_predicate, normalizedPredicate, immutablePredicate,
        Front.Insts.CoreCmpPartialEqFront.eq, Front.read_discriminant]
    all_goals split <;> simp_all

theorem normalized_predicate_preserves_every_read (front : Front) (predicate : Pred)
    (values : Reg → Bool) :
    predicateRead front values (normalizedPredicate front predicate) =
      predicateRead front values predicate := by
  rcases predicate with ⟨register, neg⟩
  cases register with
  | none => rfl
  | some register =>
    by_cases constant : immutablePredicate front register <;>
      simp [normalizedPredicate, predicateRead, constant]

theorem normalized_register_is_mutable (front : Front) (predicate : Pred) (register : Reg)
    (present : (normalizedPredicate front predicate).reg = some register) :
    ¬immutablePredicate front register := by
  rcases predicate with ⟨source, neg⟩
  cases source with
  | none => simp [normalizedPredicate] at present
  | some source =>
    by_cases constant : immutablePredicate front source
    · simp [normalizedPredicate, constant] at present
    · simp [normalizedPredicate, constant] at present
      cases present
      exact constant

theorem source_normalization_preserves_every_read (front : Front) (predicate : Pred)
    (values : Reg → Bool) :
    register_owner.normalize_predicate front predicate ⦃ result =>
      predicateRead front values result = predicateRead front values predicate ∧
      (∀ register, result.reg = some register → ¬immutablePredicate front register) ⦄ := by
  rw [source_normalize_predicate]
  apply (WP.spec_ok _).mpr
  exact ⟨normalized_predicate_preserves_every_read front predicate values,
    normalized_register_is_mutable front predicate⟩

theorem normalized_predicate_idempotent (front : Front) (predicate : Pred) :
    normalizedPredicate front (normalizedPredicate front predicate) =
      normalizedPredicate front predicate := by
  rcases predicate with ⟨register, neg⟩
  cases register with
  | none => rfl
  | some register =>
    by_cases constant : immutablePredicate front register <;>
      simp [normalizedPredicate, constant]

theorem source_repeated_normalization_agrees (front : Front) (predicate : Pred) :
    (do
      let once ← register_owner.normalize_predicate front predicate
      register_owner.normalize_predicate front once) =
        register_owner.normalize_predicate front predicate := by
  simp [source_normalize_predicate, normalized_predicate_idempotent]

end RegisterOwner

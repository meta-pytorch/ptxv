/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.OwnerPredicate
import Instruction.PartialSuccess

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.PredicateRead

open RegisterOwner

def meaning (front : Front) (predicate : Pred) (values : Reg → Option Bool) : Option Bool :=
  let value := match predicate.reg with
    | none => some true
    | some register => if immutablePredicate front register then some true else values register
  value.map (fun value => value != predicate.neg)

theorem normalized_neg (front : Front) (predicate : Pred) :
    (normalizedPredicate front predicate).neg = predicate.neg := by
  rcases predicate with ⟨register, neg⟩
  cases register with
  | none => rfl
  | some register => by_cases constant : immutablePredicate front register <;>
      simp [normalizedPredicate, constant]

theorem meaning_normalized (front : Front) (predicate : Pred) (values : Reg → Option Bool) :
    meaning front (normalizedPredicate front predicate) values = meaning front predicate values := by
  rcases predicate with ⟨register, neg⟩
  cases register with
  | none => rfl
  | some register => by_cases constant : immutablePredicate front register <;>
      simp [normalizedPredicate, meaning, constant]

theorem source_constant_no_read {R : Type} (reader : predicate_read.KnownPredicate R)
    (state : R) (front : Front) (predicate : Pred)
    (constant : (normalizedPredicate front predicate).reg = none) :
    predicate_read.evaluate reader state front predicate = .ok (some (!predicate.neg), state) := by
  unfold predicate_read.evaluate
  rw [source_normalize_predicate]
  simp only [bind_ok, constant, normalized_neg]
  cases predicate.neg <;> simp [core.option.Option.map,
    predicate_read.evaluate.closure.Insts.CoreOpsFunctionFnOnceTupleBoolBool,
    predicate_read.evaluate.closure.Insts.CoreOpsFunctionFnOnceTupleBoolBool.call_once]

theorem source_known_read {R : Type} (reader : predicate_read.KnownPredicate R)
    (state : R) (front : Front) (predicate : Pred) (register : Reg)
    (present : (normalizedPredicate front predicate).reg = some register)
    (value : Option Bool) (next : R) (observed : reader.read state register = .ok (value, next)) :
    predicate_read.evaluate reader state front predicate =
      .ok (value.map (fun truth => truth != predicate.neg), next) := by
  unfold predicate_read.evaluate
  rw [source_normalize_predicate]
  simp only [bind_ok, present, observed, normalized_neg]
  change Std.bind (.ok (next, value)) (fun pair : R × Option Bool =>
    Std.bind (core.option.Option.map
      (predicate_read.evaluate.closure.Insts.CoreOpsFunctionFnOnceTupleBoolBool reader)
      pair.2 predicate.neg) (fun result => .ok (result, pair.1))) = _
  rw [bind_ok]
  cases value <;> simp [core.option.Option.map,
    predicate_read.evaluate.closure.Insts.CoreOpsFunctionFnOnceTupleBoolBool,
    predicate_read.evaluate.closure.Insts.CoreOpsFunctionFnOnceTupleBoolBool.call_once]

@[step]
theorem rust_evaluate {R : Type} (reader : predicate_read.KnownPredicate R)
    (state : R) (front : Front) (predicate : Pred) (values : Reg → Option Bool)
    (frame : R → Prop) (unchanged : frame state)
    (reads : ∀ register, (normalizedPredicate front predicate).reg = some register →
      reader.read state register ⦃ result => result.1 = values register ∧ frame result.2 ⦄) :
    predicate_read.evaluate reader state front predicate ⦃ result =>
      result.1 = meaning front predicate values ∧ frame result.2 ⦄ := by
  cases present : (normalizedPredicate front predicate).reg with
  | none =>
      rw [source_constant_no_read reader state front predicate present, WP.spec_ok]
      refine ⟨?_, unchanged⟩
      rw [←meaning_normalized front predicate values]
      unfold meaning
      rw [present, normalized_neg]
      cases predicate.neg <;> simp [core.option.Option.map,
    predicate_read.evaluate.closure.Insts.CoreOpsFunctionFnOnceTupleBoolBool,
    predicate_read.evaluate.closure.Insts.CoreOpsFunctionFnOnceTupleBoolBool.call_once]
  | some register =>
      obtain ⟨⟨value, next⟩, readOk, valueEq, nextFrame⟩ := Instruction.Partial.spec_returns _ _
        (reads register present)
      rw [source_known_read reader state front predicate register present value next readOk, WP.spec_ok]
      refine ⟨?_, nextFrame⟩
      change value = values register at valueEq
      change value.map (fun truth => truth != predicate.neg) = meaning front predicate values
      rw [valueEq, ←meaning_normalized front predicate values]
      unfold meaning
      rw [present]
      have mutable := normalized_register_is_mutable front predicate register present
      simp only [mutable, Bool.false_eq_true, ite_false, normalized_neg]

theorem source_false_constant_never_reads {R : Type} (reader : predicate_read.KnownPredicate R)
    (state : R) (front : Front) :
    predicate_read.evaluate reader state front {reg := none, neg := true} =
      .ok (some false, state) := by
  exact source_constant_no_read reader state front _ rfl

theorem meaning_matches_architectural_read (front : Front) (predicate : Pred) (values : Reg → Bool) :
    meaning front predicate (fun register => some (values register)) =
      some (predicateRead front values predicate) := by
  unfold meaning predicateRead
  cases predicate.reg with
  | none => rfl
  | some register => by_cases constant : immutablePredicate front register <;>
      simp [constant]


@[step]
theorem rust_evaluate_architectural {R : Type} (reader : predicate_read.KnownPredicate R)
    (state : R) (front : Front) (predicate : Pred) (values : Reg → Bool)
    (frame : R → Prop) (unchanged : frame state)
    (reads : ∀ register, (normalizedPredicate front predicate).reg = some register →
      reader.read state register ⦃ result => result.1 = some (values register) ∧ frame result.2 ⦄) :
    predicate_read.evaluate reader state front predicate ⦃ result =>
      result.1 = some (predicateRead front values predicate) ∧ frame result.2 ⦄ := by
  apply WP.spec_mono (rust_evaluate reader state front predicate
    (fun register => some (values register)) frame unchanged reads)
  intro result post
  simpa only [meaning_matches_architectural_read] using post

theorem source_sass_p7_never_reads {R : Type} (reader : predicate_read.KnownPredicate R)
    (state : R) (neg : Bool) :
    predicate_read.evaluate reader state .Sass {reg := some {file := .P, idx := 7#u16}, neg} =
      .ok (some (!neg), state) := by
  apply source_constant_no_read
  simp [normalizedPredicate, immutablePredicate]

theorem source_up7_never_reads {R : Type} (reader : predicate_read.KnownPredicate R)
    (state : R) (front : Front) (neg : Bool) :
    predicate_read.evaluate reader state front {reg := some {file := .UP, idx := 7#u16}, neg} =
      .ok (some (!neg), state) := by
  apply source_constant_no_read
  simp [normalizedPredicate, immutablePredicate]

end Instruction.PredicateRead

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicComponentTrace
import MemoryWrite.RustAtomicComponentPlan

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite.AtomicComponentPlan

def sourceRun (plan : atomic_components.Plan) : List atomic_components.Component →
    Result (Option atomic_components.Plan)
  | [] => ok (some plan)
  | component :: rest => do
      let next ← atomic_components.finish plan component
      match next with
      | none => ok none
      | some value => sourceRun value rest

@[step] theorem source_any_prefix (plan : atomic_components.Plan)
    (schedule : List atomic_components.Component) :
    sourceRun plan schedule
      ⦃ result => result.map planValue = run (planValue plan) (schedule.map componentValue) ⦄ := by
  induction schedule generalizing plan with
  | nil => simp [sourceRun, run]
  | cons component rest ih =>
    unfold sourceRun
    step with rust_finish plan component as ⟨next, projected⟩
    cases next with
    | none =>
      simp only [Option.map_none] at projected
      simp [run, ← projected]
    | some value =>
      simp only [Option.map_some] at projected
      step with ih value as ⟨result, resultValue⟩
      simpa [run, ← projected] using resultValue

def sourceRunPlans (plans : Nat → atomic_components.Plan) :
    List (Nat × atomic_components.Component) → Result (Option (Nat → atomic_components.Plan))
  | [] => ok (some plans)
  | (actor, component) :: rest => do
      let next ← atomic_components.finish (plans actor) component
      match next with
      | none => ok none
      | some value => sourceRunPlans (update plans actor value) rest

def requests (schedule : List (Nat × atomic_components.Component)) : List (Action Unit) :=
  schedule.map (fun (actor, component) => .component actor (componentValue component))

@[step] theorem source_any_interleaved_prefix (plans : Nat → atomic_components.Plan)
    (schedule : List (Nat × atomic_components.Component)) :
    sourceRunPlans plans schedule
      ⦃ result => result.map (fun value actor => planValue (value actor)) =
        runPlans (fun actor => planValue (plans actor)) (requests schedule) ⦄ := by
  induction schedule generalizing plans with
  | nil => simp [sourceRunPlans, runPlans, requests]
  | cons head rest ih =>
    rcases head with ⟨actor, component⟩
    unfold sourceRunPlans
    step with rust_finish (plans actor) component as ⟨next, projected⟩
    cases next with
    | none =>
      simp only [Option.map_none] at projected
      simp [requests, runPlans, planStep, ← projected]
    | some value =>
      simp only [Option.map_some] at projected
      step with ih (update plans actor value) as ⟨result, resultValue⟩
      have updated : (fun other => planValue (update plans actor value other)) =
          update (fun other => planValue (plans other)) actor (planValue value) := by
        funext other
        by_cases same : other = actor <;> simp [update, same]
      simpa [requests, runPlans, planStep, ← projected, updated] using resultValue

theorem source_admitted_prefix_preserves_returns_and_enabledness
    {Data : Type} {Finding : Type}
    (execute : Data → Nat → Component → Nat → Effect Data Finding)
    (environment : Data → Unit → Effect Data Finding)
    (world : World Data Finding) (plans nextPlans : Nat → atomic_components.Plan)
    (schedule : List (Nat × atomic_components.Component))
    (initial : world.plans = fun actor => planValue (plans actor))
    (accepted : sourceRunPlans plans schedule = ok (some nextPlans)) :
    ∃ next, runWorld execute environment world (requests schedule) = some next ∧
      next.observed = (requests schedule).foldl
        (observe execute environment world.latched) world.observed ∧
      next.latched = world.latched ∧
      ∀ actor component, pending (next.plans actor) component =
        pending (planValue (nextPlans actor)) component := by
  have projected := source_any_interleaved_prefix plans schedule
  simp only [accepted, WP.spec_ok, Option.map_some] at projected
  apply every_admitted_prefix_has_its_exact_returns_and_future_enabledness
    execute environment world (requests schedule) (fun actor => planValue (nextPlans actor))
  simpa only [initial] using projected.symm

end MemoryWrite.AtomicComponentPlan

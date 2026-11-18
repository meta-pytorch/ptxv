/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

set_option autoImplicit false

namespace MemoryWrite.AtomicComponentPlan

inductive Component where
  | first | second | third | fourth
  deriving DecidableEq, Repr

structure Plan where
  first : Bool
  second : Bool
  third : Bool
  fourth : Bool
  deriving DecidableEq, Repr

def vectorPlan (bytes : Nat) : Option Plan :=
  if bytes = 8 then some ⟨true, true, false, false⟩
  else if bytes = 16 then some ⟨true, true, true, true⟩
  else none

def byteOffset : Component → Nat
  | .first => 0
  | .second => 4
  | .third => 8
  | .fourth => 12

def pending (plan : Plan) : Component → Bool
  | .first => plan.first
  | .second => plan.second
  | .third => plan.third
  | .fourth => plan.fourth

def ticketEligible (floatAdd : Bool) (bytes : Nat) : Bool :=
  !(floatAdd && decide (bytes = 8 ∨ bytes = 16))

theorem vector_component_ticket_ineligible (bytes : Nat) (width : bytes = 8 ∨ bytes = 16) :
    ticketEligible true bytes = false := by
  simp [ticketEligible, width]

def allComponents : List Component := [.first, .second, .third, .fourth]

def choices (plan : Plan) : List Component := allComponents.filter (pending plan)

@[simp] theorem choices_exactly_enabled (plan : Plan) (component : Component) :
    component ∈ choices plan ↔ pending plan component = true := by
  cases component <;> simp [choices, allComponents]

theorem choices_no_duplicates (plan : Plan) : (choices plan).Nodup := by
  cases plan with
  | mk a b c d =>
    cases a <;> cases b <;> cases c <;> cases d <;> decide

def clear (plan : Plan) : Component → Plan
  | .first => { plan with first := false }
  | .second => { plan with second := false }
  | .third => { plan with third := false }
  | .fourth => { plan with fourth := false }

def finish (plan : Plan) (component : Component) : Option Plan :=
  if pending plan component then some (clear plan component) else none

def retired (plan : Plan) : Bool :=
  !(plan.first || plan.second || plan.third || plan.fourth)

def count (plan : Plan) : Nat :=
  plan.first.toNat + plan.second.toNat + plan.third.toNat + plan.fourth.toNat

def run (plan : Plan) : List Component → Option Plan
  | [] => some plan
  | component :: rest => (finish plan component).bind (fun next => run next rest)

@[simp] theorem pending_clear (plan : Plan) (component other : Component) :
    pending (clear plan component) other = (pending plan other && decide (other ≠ component)) := by
  cases component <;> cases other <;> simp [clear, pending]

theorem finish_exactly_enabled (plan : Plan) (component : Component) :
    (∃ next, finish plan component = some next) ↔ pending plan component = true := by
  simp [finish]

theorem finish_strict_progress (plan next : Plan) (component : Component)
    (step : finish plan component = some next) :
    count next + 1 = count plan := by
  cases plan with
  | mk a b c d =>
    cases component <;> cases a <;> cases b <;> cases c <;> cases d <;>
      simp only [finish, pending, Bool.false_eq_true, ↓reduceIte, Option.some.injEq] at step <;>
      try contradiction
    all_goals subst next; decide

theorem retirement_exactly_no_enabled (plan : Plan) :
    retired plan = true ↔ ∀ component, pending plan component = false := by
  constructor
  · intro done component
    have fields : plan.first = false ∧ plan.second = false ∧
        plan.third = false ∧ plan.fourth = false := by
      simpa [retired, Bool.or_eq_false_iff, and_assoc] using done
    cases component <;> simp_all [pending]
  · intro none
    have a := none .first
    have b := none .second
    have c := none .third
    have d := none .fourth
    simp_all [retired, pending]

theorem run_pending (plan next : Plan) (schedule : List Component)
    (steps : run plan schedule = some next) (component : Component) :
    pending next component = (pending plan component && decide (component ∉ schedule)) := by
  induction schedule generalizing plan with
  | nil => simp [run] at steps; subst next; simp
  | cons head rest ih =>
    simp only [run, finish] at steps
    split at steps
    · simp only [Option.bind_some] at steps
      rw [ih _ steps, pending_clear]
      simp only [List.mem_cons, not_or, ne_eq]
      cases flag : pending plan component <;> by_cases same : component = head <;>
        by_cases member : component ∈ rest <;> simp_all
    · simp at steps

theorem run_exactly_once (plan : Plan) (schedule : List Component) :
    (∃ next, run plan schedule = some next) ↔
      schedule.Nodup ∧ ∀ component ∈ schedule, pending plan component = true := by
  induction schedule generalizing plan with
  | nil => simp [run]
  | cons component rest ih =>
    simp only [run, finish]
    by_cases enabled : pending plan component = true
    · simp only [enabled, ↓reduceIte, Option.bind_some, ih, List.nodup_cons]
      simp only [pending_clear, Bool.and_eq_true, decide_eq_true_eq]
      constructor
      · rintro ⟨nodup, all⟩
        refine ⟨⟨?_, nodup⟩, ?_⟩
        · intro member; exact (all component member).2 rfl
        · intro other member
          rcases List.mem_cons.mp member with same | later
          · simpa [same] using enabled
          · exact (all other later).1
      · rintro ⟨⟨absent, nodup⟩, all⟩
        refine ⟨nodup, ?_⟩
        intro other member
        refine ⟨all other (List.mem_cons_of_mem component member), ?_⟩
        intro same
        subst other
        exact absent member
    · have disabled : pending plan component = false := Bool.eq_false_iff.mpr enabled
      constructor
      · rintro ⟨next, bad⟩
        simp [disabled] at bad
      · rintro ⟨_, all⟩
        exact False.elim (enabled (all component (List.mem_cons_self ..)))

theorem any_complete_order_retires (plan : Plan) (schedule : List Component)
    (once : schedule.Nodup)
    (exact : ∀ component, component ∈ schedule ↔ pending plan component = true) :
    ∃ next, run plan schedule = some next ∧ retired next = true := by
  obtain ⟨next, steps⟩ := (run_exactly_once plan schedule).mpr
    ⟨once, fun component member => (exact component).mp member⟩
  refine ⟨next, steps, (retirement_exactly_no_enabled next).mpr ?_⟩
  intro component
  rw [run_pending plan next schedule steps component]
  cases enabled : pending plan component
  · simp
  · simp [(exact component).mpr enabled]

end MemoryWrite.AtomicComponentPlan

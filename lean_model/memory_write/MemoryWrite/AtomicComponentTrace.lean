/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicComponentPlan

set_option autoImplicit false

namespace MemoryWrite.AtomicComponentPlan

universe u v w

inductive Action (Context : Type u) where
  | component (actor : Nat) (component : Component)
  | context (event : Context)

structure Effect (Data : Type v) (Finding : Type w) where
  data : Data
  returned : Nat
  findings : List Finding
  failed : Bool

structure Observations (Data : Type v) (Finding : Type w) where
  data : Data
  returns : List (Nat × Component × Nat)
  findings : List Finding
  failed : Bool

structure World (Data : Type v) (Finding : Type w) where
  plans : Nat → Plan
  latched : Nat → Component → Nat
  observed : Observations Data Finding

def update {Value : Type v} (state : Nat → Value) (actor : Nat) (value : Value) : Nat → Value :=
  fun other => if other = actor then value else state other

def planStep {Context : Type u} (plans : Nat → Plan) : Action Context → Option (Nat → Plan)
  | .context _ => some plans
  | .component actor component =>
      (finish (plans actor) component).map (update plans actor)

def runPlans {Context : Type u} (plans : Nat → Plan) : List (Action Context) → Option (Nat → Plan)
  | [] => some plans
  | event :: rest => (planStep plans event).bind (fun next => runPlans next rest)

def observe {Context : Type u} {Data : Type v} {Finding : Type w}
    (execute : Data → Nat → Component → Nat → Effect Data Finding)
    (environment : Data → Context → Effect Data Finding)
    (latched : Nat → Component → Nat) (observed : Observations Data Finding) :
    Action Context → Observations Data Finding
  | .component actor component =>
      let effect := execute observed.data actor component (latched actor component)
      ⟨effect.data, observed.returns ++ [(actor, component, effect.returned)],
        observed.findings ++ effect.findings, observed.failed || effect.failed⟩
  | .context event =>
      let effect := environment observed.data event
      ⟨effect.data, observed.returns, observed.findings ++ effect.findings,
        observed.failed || effect.failed⟩

def stepWorld {Context : Type u} {Data : Type v} {Finding : Type w}
    (execute : Data → Nat → Component → Nat → Effect Data Finding)
    (environment : Data → Context → Effect Data Finding)
    (world : World Data Finding) (event : Action Context) : Option (World Data Finding) :=
  (planStep world.plans event).map (fun plans =>
    { world with plans := plans, observed := observe execute environment world.latched world.observed event })

def runWorld {Context : Type u} {Data : Type v} {Finding : Type w}
    (execute : Data → Nat → Component → Nat → Effect Data Finding)
    (environment : Data → Context → Effect Data Finding)
    (world : World Data Finding) : List (Action Context) → Option (World Data Finding)
  | [] => some world
  | event :: rest => (stepWorld execute environment world event).bind
      (fun next => runWorld execute environment next rest)

theorem component_enabled_exactly {Context : Type u} (plans : Nat → Plan)
    (actor : Nat) (component : Component) :
    (∃ next, planStep plans (Action.component (Context := Context) actor component) = some next) ↔
      pending (plans actor) component = true := by
  simp [planStep, finish]

theorem arbitrary_prefix_preserves_observations {Context : Type u} {Data : Type v} {Finding : Type w}
    (execute : Data → Nat → Component → Nat → Effect Data Finding)
    (environment : Data → Context → Effect Data Finding)
    (world : World Data Finding) (events : List (Action Context)) :
    runWorld execute environment world events =
      (runPlans world.plans events).map (fun plans =>
        { world with
          plans := plans
          observed := events.foldl (observe execute environment world.latched) world.observed }) := by
  induction events generalizing world with
  | nil => rfl
  | cons event rest ih =>
    simp only [runWorld, runPlans, stepWorld]
    cases next : planStep world.plans event with
    | none => simp
    | some plans =>
      simp only [Option.map_some, Option.bind_some]
      rw [ih]
      rfl

theorem every_admitted_prefix_has_its_exact_returns_and_future_enabledness
    {Context : Type u} {Data : Type v} {Finding : Type w}
    (execute : Data → Nat → Component → Nat → Effect Data Finding)
    (environment : Data → Context → Effect Data Finding)
    (world : World Data Finding) (events : List (Action Context)) (plans : Nat → Plan)
    (admitted : runPlans world.plans events = some plans) :
    ∃ next, runWorld execute environment world events = some next ∧
      next.observed = events.foldl (observe execute environment world.latched) world.observed ∧
      next.latched = world.latched ∧
      ∀ actor component, pending (next.plans actor) component = pending (plans actor) component := by
  rw [arbitrary_prefix_preserves_observations, admitted]
  exact ⟨_, rfl, rfl, rfl, fun _ _ => rfl⟩

theorem observe_failure_sticky {Context : Type u} {Data : Type v} {Finding : Type w}
    (execute : Data → Nat → Component → Nat → Effect Data Finding)
    (environment : Data → Context → Effect Data Finding)
    (latched : Nat → Component → Nat) (observed : Observations Data Finding)
    (event : Action Context) (failed : observed.failed = true) :
    (observe execute environment latched observed event).failed = true := by
  cases event <;> simp [observe, failed]

theorem observe_findings_prefix {Context : Type u} {Data : Type v} {Finding : Type w}
    (execute : Data → Nat → Component → Nat → Effect Data Finding)
    (environment : Data → Context → Effect Data Finding)
    (latched : Nat → Component → Nat) (observed : Observations Data Finding)
    (event : Action Context) :
    ∃ extra, (observe execute environment latched observed event).findings = observed.findings ++ extra := by
  cases event <;> exact ⟨_, rfl⟩

theorem observe_trace_failure_sticky {Context : Type u} {Data : Type v} {Finding : Type w}
    (execute : Data → Nat → Component → Nat → Effect Data Finding)
    (environment : Data → Context → Effect Data Finding)
    (latched : Nat → Component → Nat) (observed : Observations Data Finding)
    (events : List (Action Context)) (failed : observed.failed = true) :
    (events.foldl (observe execute environment latched) observed).failed = true := by
  induction events generalizing observed with
  | nil => exact failed
  | cons event rest ih => exact ih _ (observe_failure_sticky execute environment latched observed event failed)

theorem observe_trace_findings_prefix {Context : Type u} {Data : Type v} {Finding : Type w}
    (execute : Data → Nat → Component → Nat → Effect Data Finding)
    (environment : Data → Context → Effect Data Finding)
    (latched : Nat → Component → Nat) (observed : Observations Data Finding)
    (events : List (Action Context)) :
    ∃ extra, (events.foldl (observe execute environment latched) observed).findings = observed.findings ++ extra := by
  induction events generalizing observed with
  | nil => exact ⟨[], by simp⟩
  | cons event rest ih =>
    obtain ⟨first, firstValue⟩ := observe_findings_prefix execute environment latched observed event
    obtain ⟨later, laterValue⟩ := ih (observe execute environment latched observed event)
    exact ⟨first ++ later, by simpa only [List.foldl_cons, firstValue, List.append_assoc] using laterValue⟩

def witnessWorld : World (Component → Nat) Unit :=
  ⟨fun _ => ⟨true, true, false, false⟩,
    fun actor component => if actor = 0 then
      (if component = .first then 1 else 10)
    else (if component = .first then 2 else 20),
    ⟨fun _ => 0, [], [], false⟩⟩

def witnessExecute (memory : Component → Nat) (_actor : Nat) (component : Component)
    (added : Nat) : Effect (Component → Nat) Unit :=
  ⟨fun other => if other = component then memory component + added else memory other,
    memory component, [], false⟩

def witnessPrefix : List (Action Unit) :=
  [.component 0 .first, .component 1 .first, .component 1 .second, .component 0 .second]

theorem mixed_return_observation_prefix_is_admitted :
    (runWorld witnessExecute (fun memory _ => ⟨memory, 0, [], false⟩)
      witnessWorld witnessPrefix).map (fun next =>
        (next.observed.returns, next.observed.data .first, next.observed.data .second,
          retired (next.plans 0), retired (next.plans 1))) =
      some ([(0, .first, 0), (1, .first, 1), (1, .second, 0), (0, .second, 20)], 3, 30, true, true) := by
  decide

end MemoryWrite.AtomicComponentPlan

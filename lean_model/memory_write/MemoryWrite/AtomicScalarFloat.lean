/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicComponentTrace

set_option autoImplicit false

namespace MemoryWrite.ScalarFloat

open AtomicComponentPlan

def one : Plan := ⟨true, false, false, false⟩
def empty : Plan := ⟨false, false, false, false⟩
def events (actors : List Nat) : List (Action Unit) :=
  actors.map (fun actor => .component actor .first)
def claimed (plans : Nat → Plan) (actors : List Nat) : Nat → Plan :=
  fun actor => if actor ∈ actors then empty else plans actor

theorem every_distinct_lane_order_is_admitted (actors : List Nat)
    (plans : Nat → Plan) (distinct : actors.Nodup)
    (available : ∀ actor ∈ actors, plans actor = one) :
    runPlans plans (events actors) = some (claimed plans actors) := by
  induction actors generalizing plans with
  | nil => rfl
  | cons actor rest ih =>
    obtain ⟨fresh, distinct⟩ := List.nodup_cons.mp distinct
    have ready : plans actor = one := available actor (by simp)
    have restReady : ∀ other ∈ rest, update plans actor empty other = one := by
      intro other member
      have different : other ≠ actor := by intro same; subst other; exact fresh member
      simpa [update, different] using available other (by simp [member])
    have next := ih (update plans actor empty) distinct restReady
    simp only [events, List.map_cons, runPlans, planStep, ready]
    change runPlans (update plans actor empty) (events rest) = some (claimed plans (actor :: rest))
    rw [next]
    congr 1
    funext other
    by_cases member : other ∈ rest <;> by_cases same : other = actor <;>
      simp [claimed, update, List.mem_cons, member, same]

def componentActors {Context : Type} : List (Action Context) → List Nat
  | [] => []
  | .context _ :: rest => componentActors rest
  | .component actor _ :: rest => actor :: componentActors rest

def firstActions {Context : Type} : List (Action Context) → Prop
  | [] => True
  | .context _ :: rest => firstActions rest
  | .component _ component :: rest => component = .first ∧ firstActions rest

theorem every_distinct_lane_order_with_context_is_admitted {Context : Type}
    (actions : List (Action Context)) (plans : Nat → Plan)
    (distinct : (componentActors actions).Nodup) (first : firstActions actions)
    (available : ∀ actor ∈ componentActors actions, plans actor = one) :
    runPlans plans actions = some (claimed plans (componentActors actions)) := by
  induction actions generalizing plans with
  | nil => rfl
  | cons action rest ih =>
    cases action with
    | context event =>
      simpa only [runPlans, planStep, Option.bind_some, componentActors] using
        ih plans distinct first available
    | component actor component =>
      obtain ⟨componentValue, laterFirst⟩ := first
      subst component
      obtain ⟨fresh, laterDistinct⟩ := List.nodup_cons.mp distinct
      have ready : plans actor = one := available actor (by simp [componentActors])
      have restReady : ∀ other ∈ componentActors rest,
          update plans actor empty other = one := by
        intro other member
        have different : other ≠ actor := by intro same; subst other; exact fresh member
        simpa [update, different] using available other (by simp [componentActors, member])
      have next := ih (update plans actor empty) laterDistinct laterFirst restReady
      simp only [runPlans, planStep, ready]
      change runPlans (update plans actor empty) rest =
        some (claimed plans (actor :: componentActors rest))
      rw [next]
      congr 1
      funext other
      by_cases member : other ∈ componentActors rest <;> by_cases same : other = actor <;>
        simp [claimed, update, List.mem_cons, member, same]

structure State (Word : Type) where
  memory : Nat → Word
  returned : Nat → Word

def execute {Word Finding : Type} (bits : Word → Nat) (decode : Nat → Word)
    (add : Word → Word → Word) (address destination : Nat → Nat)
    (state : State Word) (actor : Nat) (_component : Component) (source : Nat) :
    Effect (State Word) Finding :=
  let old := state.memory (address actor)
  ⟨⟨update state.memory (address actor) (add old (decode source)),
      update state.returned (destination actor) old⟩, bits old, [], false⟩

theorem scalar_effect_returns_the_preceding_word {Word Finding : Type}
    (bits : Word → Nat) (decode : Nat → Word) (add : Word → Word → Word)
    (address destination : Nat → Nat) (state : State Word) (actor source : Nat) :
    (execute (Finding := Finding) bits decode add address destination state actor .first source).returned =
      bits (state.memory (address actor)) := by rfl

theorem scalar_effect_writes_only_the_captured_address {Word Finding : Type}
    (bits : Word → Nat) (decode : Nat → Word) (add : Word → Word → Word)
    (address destination : Nat → Nat) (state : State Word) (actor source location : Nat) :
    (execute (Finding := Finding) bits decode add address destination state actor .first source).data.memory location =
      if location = address actor then add (state.memory location) (decode source)
      else state.memory location := by
  by_cases same : location = address actor <;> simp [execute, update, same]

theorem scalar_effect_returns_to_only_the_selected_lane {Word Finding : Type}
    (bits : Word → Nat) (decode : Nat → Word) (add : Word → Word → Word)
    (address destination : Nat → Nat) (state : State Word) (actor source other : Nat) :
    (execute (Finding := Finding) bits decode add address destination state actor .first source).data.returned other =
      if other = destination actor then state.memory (address actor) else state.returned other := by rfl

theorem every_distinct_capture_prefix_preserves_effects_and_enabledness
    {Word Finding : Type} (bits : Word → Nat) (decode : Nat → Word)
    (add : Word → Word → Word) (address destination : Nat → Nat)
    (environment : State Word → Unit → Effect (State Word) Finding)
    (world : World (State Word) Finding) (actors : List Nat)
    (distinct : actors.Nodup) (available : ∀ actor ∈ actors, world.plans actor = one) :
    ∃ next, runWorld (execute bits decode add address destination) environment world (events actors) = some next ∧
      next.observed = (events actors).foldl
        (observe (execute bits decode add address destination) environment world.latched) world.observed ∧
      next.latched = world.latched ∧
      ∀ actor component, pending (next.plans actor) component =
        pending (claimed world.plans actors actor) component := by
  exact every_admitted_prefix_has_its_exact_returns_and_future_enabledness _ _ world _ _
    (every_distinct_lane_order_is_admitted actors world.plans distinct available)

theorem every_contextual_capture_prefix_preserves_observations_and_enabledness
    {Context Word Finding : Type} (bits : Word → Nat) (decode : Nat → Word)
    (add : Word → Word → Word) (address destination : Nat → Nat)
    (environment : State Word → Context → Effect (State Word) Finding)
    (world : World (State Word) Finding) (actions : List (Action Context))
    (distinct : (componentActors actions).Nodup) (first : firstActions actions)
    (available : ∀ actor ∈ componentActors actions, world.plans actor = one) :
    ∃ next, runWorld (execute bits decode add address destination) environment world actions = some next ∧
      next.observed = actions.foldl
        (observe (execute bits decode add address destination) environment world.latched) world.observed ∧
      next.latched = world.latched ∧
      ∀ actor component, pending (next.plans actor) component =
        pending (claimed world.plans (componentActors actions) actor) component := by
  exact every_admitted_prefix_has_its_exact_returns_and_future_enabledness _ _ world _ _
    (every_distinct_lane_order_with_context_is_admitted actions world.plans distinct first available)

def oneBits : Nat := 1065353216
def twoBits : Nat := 1073741824
def threeBits : Nat := 1077936128

def witnessWorld : World (State Nat) Unit :=
  ⟨fun _ => one, fun _ _ => oneBits, ⟨⟨fun _ => 0, fun _ => 0⟩, [], [], false⟩⟩

def witnessRun (add : Nat → Nat → Nat) (actors : List Nat) :
    Option (World (State Nat) Unit) :=
  runWorld (execute id id add (fun _ => 0) id)
    (fun state _ => ⟨state, 0, [], false⟩) witnessWorld (events actors)

def witnessProjection (add : Nat → Nat → Nat) (actors : List Nat) : Option (Nat × Nat) :=
  (witnessRun add actors).map (fun next => (next.observed.data.memory 0, next.observed.data.returned 0))

theorem equal_final_words_do_not_preserve_return_observations
    (add : Nat → Nat → Nat) (zeroOne : add 0 oneBits = oneBits)
    (oneOne : add oneBits oneBits = twoBits) (twoOne : add twoBits oneBits = threeBits) :
    witnessProjection add [0,1,2] = some (threeBits, 0) ∧
    witnessProjection add [2,1,0] = some (threeBits, twoBits) ∧
    witnessProjection add [1,0,2] = some (threeBits, oneBits) := by
  simp [witnessProjection, witnessRun, witnessWorld, events, runWorld, stepWorld, planStep,
    finish, pending, clear, observe, execute, update, one, zeroOne, oneOne, twoOne]

theorem middle_prefix_keeps_its_visible_returns
    (add : Nat → Nat → Nat) (zeroOne : add 0 oneBits = oneBits)
    (oneOne : add oneBits oneBits = twoBits) (twoOne : add twoBits oneBits = threeBits) :
    (witnessRun add [1,0,2]).map (fun next => next.observed.returns) =
      some [(1,.first,0),(0,.first,oneBits),(2,.first,twoBits)] := by
  simp [witnessRun, witnessWorld, events, runWorld, stepWorld, planStep,
    finish, pending, clear, observe, execute, update, one, zeroOne, oneOne, twoOne]

end MemoryWrite.ScalarFloat

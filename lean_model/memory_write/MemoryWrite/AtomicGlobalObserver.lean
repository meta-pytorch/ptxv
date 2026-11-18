/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicInteger

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.AtomicGlobalObserver

def add64 (old value : U64) : Result (U64 × U64) := do
  let result ← atomic_integer.eval
    { op := .Add, bytes := 8#u8, old, value, compare := 0#u64 }
  match result with
  | .Ok effect => .ok (effect.old, effect.replacement)
  | .Err _ => fail .panic

theorem source_add64_returns_old_and_wraps (old value : U64) :
    add64 old value = .ok (old, core.num.U64.wrapping_add old value) := by
  simp [add64, atomic_integer.eval, atomic_integer.replacement64,
    atomic_integer.is_cas, lift, bind_ok]

inductive Action where
  | write0 | write1 | read0 | read1
  deriving DecidableEq

structure State where
  word0 : U64
  word1 : U64
  pending0 : Bool
  pending1 : Bool
  returned0 : Option U64
  returned1 : Option U64
  deriving DecidableEq

def initial : State := ⟨0#u64, 0#u64, true, true, none, none⟩

def enabled (action : Action) (state : State) : Bool :=
  match action with
  | .write0 => state.pending0
  | .write1 => state.pending1
  | .read0 => state.returned0.isNone
  | .read1 => state.returned0.isSome && state.returned1.isNone

def operands (action : Action) (state : State) : U64 × U64 :=
  match action with
  | .write0 => (state.word0, 1#u64)
  | .write1 => (state.word1, 1#u64)
  | .read0 => (state.word0, 0#u64)
  | .read1 => (state.word1, 0#u64)

def finish (action : Action) (state : State) (returned replacement : U64) : State :=
  match action with
  | .write0 => { state with word0 := replacement, pending0 := false }
  | .write1 => { state with word1 := replacement, pending1 := false }
  | .read0 => { state with word0 := replacement, returned0 := some returned }
  | .read1 => { state with word1 := replacement, returned1 := some returned }

def step (action : Action) (state : State) : Option State :=
  if enabled action state then
    let input := operands action state
    some (finish action state input.1 (core.num.U64.wrapping_add input.1 input.2))
  else none

def sourceStep (action : Action) (state : State) : Result State := do
  if enabled action state then
    let input := operands action state
    let effect ← add64 input.1 input.2
    .ok (finish action state effect.1 effect.2)
  else fail .panic

def liftState : Option State → Result State
  | some state => .ok state
  | none => fail .panic

theorem source_step_refines_word_effect (action : Action) (state : State) :
    sourceStep action state = liftState (step action state) := by
  simp only [sourceStep, step]
  split <;> simp [source_add64_returns_old_and_wraps, bind_ok, liftState]

def run (state : State) : List Action → Option State
  | [] => some state
  | action :: remaining => (step action state).bind (fun next => run next remaining)

def sourceRun (state : State) : List Action → Result State
  | [] => .ok state
  | action :: remaining => do
      let next ← sourceStep action state
      sourceRun next remaining

theorem source_run_refines_word_effects (state : State) (actions : List Action) :
    sourceRun state actions = liftState (run state actions) := by
  induction actions generalizing state with
  | nil => rfl
  | cons action remaining ih =>
      simp only [sourceRun, source_step_refines_word_effect, run]
      cases h : step action state <;> simp [liftState, bind_ok, ih]

def target (state : State) : Bool :=
  state.returned0 == some 1#u64 && state.returned1 == some 0#u64

def observesTarget (outcome : Option State) : Bool := outcome.any target

def referenceOrder : List Action := [.write0, .read0, .read1, .write1]

def referenceFinal : State :=
  ⟨1#u64, 1#u64, false, false, some 1#u64, some 0#u64⟩

theorem reference_order_is_enabled_and_observes_one_zero :
    run initial referenceOrder = some referenceFinal := by
  decide

theorem actual_integer_effects_observe_one_zero :
    sourceRun initial referenceOrder = .ok referenceFinal := by
  rw [source_run_refines_word_effects, reference_order_is_enabled_and_observes_one_zero]
  rfl

inductive BundledAction where
  | writer | read0 | read1
  deriving DecidableEq

def expand (reverse : Bool) : BundledAction → List Action
  | .writer => if reverse then [.write1, .write0] else [.write0, .write1]
  | .read0 => [.read0]
  | .read1 => [.read1]

def bundledOrders : List (List BundledAction) :=
  [.writer, .read0, .read1].permutations

def bundledRun (reverse : Bool) (order : List BundledAction) : Option State :=
  run initial (order.flatMap (expand reverse))

theorem bundled_before_between_after_observations :
    ∀ reverse : Bool,
      [bundledRun reverse [.writer, .read0, .read1],
       bundledRun reverse [.read0, .writer, .read1],
       bundledRun reverse [.read0, .read1, .writer]].map
        (fun result => result.map (fun state => (state.returned0, state.returned1))) =
      [some (some 1#u64, some 1#u64), some (some 0#u64, some 1#u64),
       some (some 0#u64, some 0#u64)] := by
  intro reverse
  cases reverse <;> decide

theorem complete_bundled_enumeration_excludes_one_zero :
    ∀ reverse : Bool, ∀ order ∈ bundledOrders,
      observesTarget (bundledRun reverse order) = false := by
  intro reverse
  cases reverse <;>
    simp only [bundledOrders, List.permutations, List.permutationsAux_cons,
      List.permutationsAux_nil] <;> decide

theorem any_bundled_permutation_excludes_one_zero
    (reverse : Bool) (order : List BundledAction)
    (permutation : order.Perm [.writer, .read0, .read1]) :
    observesTarget (bundledRun reverse order) = false := by
  apply complete_bundled_enumeration_excludes_one_zero reverse order
  exact List.mem_permutations.mpr permutation

theorem disjoint_writes_commute (state : State)
    (pending0 : state.pending0 = true) (pending1 : state.pending1 = true) :
    run state [.write0, .write1] = run state [.write1, .write0] := by
  cases state
  simp_all [run, step, enabled, operands, finish]

theorem actual_integer_bundles_cannot_observe_one_zero
    (reverse : Bool) (order : List BundledAction) (final : State)
    (permutation : order.Perm [.writer, .read0, .read1])
    (executed : sourceRun initial (order.flatMap (expand reverse)) = .ok final) :
    target final = false := by
  rw [source_run_refines_word_effects] at executed
  have excluded := any_bundled_permutation_excludes_one_zero reverse order permutation
  unfold bundledRun at excluded
  cases result : run initial (order.flatMap (expand reverse)) with
  | none => simp [result, liftState] at executed
  | some state =>
      have same : state = final := by simpa [result, liftState] using executed
      subst final
      simpa [result, observesTarget] using excluded

theorem bundled_valid_runs_have_same_final_memory :
    ∀ reverse : Bool, ∀ order ∈ bundledOrders,
      (bundledRun reverse order).map (fun state => (state.word0, state.word1)) = none ∨
      (bundledRun reverse order).map (fun state => (state.word0, state.word1)) =
        some (1#u64, 1#u64) := by
  intro reverse
  cases reverse <;>
    simp only [bundledOrders, List.permutations, List.permutationsAux_cons,
      List.permutationsAux_nil] <;> decide

theorem final_memory_commutation_does_not_preserve_external_observations :
    (∀ reverse order, order ∈ bundledOrders →
      observesTarget (bundledRun reverse order) = false) ∧
    observesTarget (run initial referenceOrder) = true ∧
    (run initial referenceOrder).map (fun state => (state.word0, state.word1)) =
      some (1#u64, 1#u64) := by
  refine ⟨complete_bundled_enumeration_excludes_one_zero, ?_, ?_⟩ <;>
    rw [reference_order_is_enabled_and_observes_one_zero] <;>
    decide

end MemoryWrite.AtomicGlobalObserver

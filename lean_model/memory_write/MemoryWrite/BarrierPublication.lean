/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierTransition

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.BarrierTransition

open BarrierReduction (Clock join update)

structure FullState where
  counters : Counters
  currentRelease : Clock
  completedRelease : Nat → Clock
  tokens : Nat → Option Nat
  positions : Nat → Nat
  failed : Bool
  findings : List Nat
  acquired : Nat → Clock

def publish (state : FullState) (actor : Nat) (release : Clock)
    (outcome : Transition) : FullState :=
  match outcome.fault with
  | some _ =>
    { state with
      counters := outcome.after
      failed := true
      findings := state.findings ++ [actor] }
  | none =>
    let joined := join state.currentRelease release
    { state with
      counters := outcome.after
      currentRelease := if outcome.completed then fun _ => 0 else joined,
      completedRelease := if outcome.completed then
          update state.completedRelease (state.counters.phase % 2) joined
        else state.completedRelease,
      tokens := update state.tokens actor (some state.counters.phase),
      positions := update state.positions actor (state.positions actor + 1) }

def advance (state : FullState) (actor : Nat) (delta : Int)
    (arrivals : Nat) (release : Clock) : FullState :=
  publish state actor release (transition state.counters delta arrivals)

def waitEnabled (state : FullState) (parity : Nat) : Prop :=
  parity % 2 ≠ state.counters.phase % 2

instance (state : FullState) (parity : Nat) : Decidable (waitEnabled state parity) :=
  inferInstanceAs (Decidable (parity % 2 ≠ state.counters.phase % 2))

def wait (state : FullState) (actor parity : Nat) : Option FullState :=
  if waitEnabled state parity then
    some { state with
      positions := update state.positions actor (state.positions actor + 1)
      acquired := update state.acquired actor
        (join (state.acquired actor) (state.completedRelease (parity % 2))) }
  else none

 theorem advance_exists (state : FullState) (actor : Nat) (delta : Int)
    (arrivals : Nat) (release : Clock) :
    ∃ next, advance state actor delta arrivals release = next :=
  ⟨_, rfl⟩

 theorem advance_failure_sticky (state : FullState) (actor : Nat) (delta : Int)
    (arrivals : Nat) (release : Clock) (failed : state.failed = true) :
    (advance state actor delta arrivals release).failed = true := by
  unfold advance publish
  split <;> simp [failed]

 theorem advance_findings_prefix (state : FullState) (actor : Nat) (delta : Int)
    (arrivals : Nat) (release : Clock) :
    ∃ extra, (advance state actor delta arrivals release).findings = state.findings ++ extra := by
  unfold advance publish
  split
  · exact ⟨[actor], rfl⟩
  · exact ⟨[], by simp⟩

 theorem fault_does_not_publish (state : FullState) (actor : Nat) (delta : Int)
    (arrivals : Nat) (release : Clock) (fault : Fault)
    (rejected : (transition state.counters delta arrivals).fault = some fault) :
    (advance state actor delta arrivals release).currentRelease = state.currentRelease ∧
    (advance state actor delta arrivals release).completedRelease = state.completedRelease ∧
    (advance state actor delta arrivals release).acquired = state.acquired ∧
    (advance state actor delta arrivals release).positions = state.positions ∧
    (advance state actor delta arrivals release).tokens = state.tokens ∧
    (advance state actor delta arrivals release).failed = true ∧
    (advance state actor delta arrivals release).findings = state.findings ++ [actor] := by
  simp [advance, publish, rejected]

 theorem successful_arrival_advances (state : FullState) (actor : Nat) (delta : Int)
    (arrivals : Nat) (release : Clock)
    (success : (transition state.counters delta arrivals).fault = none) :
    (advance state actor delta arrivals release).positions actor = state.positions actor + 1 ∧
    (advance state actor delta arrivals release).tokens actor = some state.counters.phase ∧
    (advance state actor delta arrivals release).findings = state.findings ∧
    (advance state actor delta arrivals release).failed = state.failed := by
  simp [advance, publish, success, update]

 theorem completion_publishes_pre_reset_release (state : FullState) (actor : Nat) (delta : Int)
    (arrivals : Nat) (release : Clock)
    (completed : (transition state.counters delta arrivals).completed = true) :
    (advance state actor delta arrivals release).completedRelease (state.counters.phase % 2) =
        join state.currentRelease release ∧
    (advance state actor delta arrivals release).currentRelease = (fun _ => 0) ∧
    (advance state actor delta arrivals release).counters.pending = state.counters.expected ∧
    (advance state actor delta arrivals release).counters.phase =
        (state.counters.phase + 1) % phaseModulus := by
  obtain ⟨fits, count, debt⟩ := (completion_exactly state.counters delta arrivals).mp completed
  simp [advance, publish, transition, transactionFits, count, debt, update]

 theorem completion_preserves_other_parity (state : FullState) (actor : Nat) (delta : Int)
    (arrivals parity : Nat) (release : Clock)
    (completed : (transition state.counters delta arrivals).completed = true)
    (different : parity ≠ state.counters.phase % 2) :
    (advance state actor delta arrivals release).completedRelease parity =
      state.completedRelease parity := by
  obtain ⟨fits, count, debt⟩ := (completion_exactly state.counters delta arrivals).mp completed
  simp [advance, publish, transition, transactionFits, count, debt, update, different]

 theorem completion_enables_old_parity (state : FullState) (actor : Nat) (delta : Int)
    (arrivals : Nat) (release : Clock)
    (completed : (transition state.counters delta arrivals).completed = true) :
    waitEnabled (advance state actor delta arrivals release) state.counters.phase := by
  have phase := (completion_publishes_pre_reset_release state actor delta arrivals release completed).2.2.2
  simp only [waitEnabled, phase, phase_wrap_toggles_parity]
  omega

 theorem completion_wait_acquires_release (state : FullState) (actor waiter : Nat) (delta : Int)
    (arrivals : Nat) (release : Clock)
    (completed : (transition state.counters delta arrivals).completed = true) :
    ∃ next, wait (advance state actor delta arrivals release) waiter state.counters.phase = some next ∧
      next.acquired waiter = join (state.acquired waiter) (join state.currentRelease release) ∧
      next.findings = (advance state actor delta arrivals release).findings := by
  have enabled := completion_enables_old_parity state actor delta arrivals release completed
  have publication := (completion_publishes_pre_reset_release state actor delta arrivals release completed).1
  have acquiredFrame : (advance state actor delta arrivals release).acquired = state.acquired := by
    unfold advance publish
    split <;> rfl
  simp only [wait, ite_eq_left enabled]
  refine ⟨_, rfl, ?_, rfl⟩
  simp [update, publication, acquiredFrame]

 theorem outstanding_debt_preserves_wait_enabledness (state : FullState) (actor : Nat) (delta : Int)
    (arrivals parity : Nat) (release : Clock)
    (fits : transactionFits (state.counters.transactions + delta))
    (count : arrivals ≤ state.counters.pending)
    (outstanding : state.counters.transactions + delta ≠ 0) :
    waitEnabled (advance state actor delta arrivals release) parity ↔ waitEnabled state parity := by
  simp [advance, publish, transition, fits, count, outstanding, waitEnabled]

def toReduction (state : FullState) : BarrierReduction.State :=
  { barrier :=
      { expected := state.counters.expected
        pending := state.counters.pending
        phase := state.counters.phase
        txZero := decide (state.counters.transactions = 0)
        currentRelease := state.currentRelease
        completedRelease := state.completedRelease }
    tokens := state.tokens
    positions := state.positions
    failed := state.failed
    findings := state.findings
    acquired := state.acquired }

 theorem plain_arrival_refines_existing_model (state : FullState) (actor arrivals : Nat)
    (release : Clock) (fits : transactionFits state.counters.transactions)
    (noWrap : state.counters.phase + 1 < phaseModulus) :
    toReduction (advance state actor 0 arrivals release) =
      BarrierReduction.arrive (toReduction state) actor arrivals release := by
  by_cases count : arrivals ≤ state.counters.pending
  · have notOver : ¬state.counters.pending < arrivals := by omega
    by_cases completes : state.counters.pending - arrivals = 0 ∧ state.counters.transactions = 0
    · simp [advance, publish, transition, count, completes, toReduction,
        BarrierReduction.arrive, BarrierReduction.decrement, BarrierReduction.complete,
        notOver, transactionFits, Nat.mod_eq_of_lt noWrap]
    · simp [advance, publish, transition, fits, count, completes, toReduction,
        BarrierReduction.arrive, BarrierReduction.decrement, BarrierReduction.complete, notOver]
  · have over : state.counters.pending < arrivals := by omega
    simp [advance, publish, transition, fits, count, over, rejected, toReduction,
      BarrierReduction.arrive]

 theorem transaction_order_changes_wait_enabledness (state : FullState) (payer expecter : Nat)
    (payRelease expectRelease : Clock) (initial : state.counters = ⟨1, 0, 1, 0⟩) :
    waitEnabled (advance (advance state payer (-1) 0 payRelease) expecter 1 0 expectRelease) 0 ∧
    ¬waitEnabled (advance (advance state expecter 1 0 expectRelease) payer (-1) 0 payRelease) 0 := by
  simp [advance, publish, transition, transactionFits, initial, waitEnabled, phaseModulus]

end MemoryWrite.BarrierTransition

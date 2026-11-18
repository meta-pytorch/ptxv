/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierReduction

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.BarrierTransition

def phaseModulus : Nat := 2 ^ 32

def transactionFits (value : Int) : Prop := -(2 ^ 63) ≤ value ∧ value < 2 ^ 63

instance (value : Int) : Decidable (transactionFits value) := inferInstanceAs
  (Decidable (-(2 ^ 63) ≤ value ∧ value < 2 ^ 63))

structure Counters where
  expected : Nat
  pending : Nat
  transactions : Int
  phase : Nat
  deriving DecidableEq, Repr

inductive Fault where
  | transactionOverflow
  | overArrival
  deriving DecidableEq, Repr

structure Transition where
  intermediate : Counters
  after : Counters
  completed : Bool
  fault : Option Fault
  deriving DecidableEq, Repr

def rejected (state : Counters) (fault : Fault) : Transition :=
  ⟨state, state, false, some fault⟩

def transition (state : Counters) (delta : Int) (arrivals : Nat) : Transition :=
  let transactions := state.transactions + delta
  if transactionFits transactions then
    let updated := { state with transactions := transactions }
    if arrivals ≤ state.pending then
      let intermediate := { updated with pending := state.pending - arrivals }
      if intermediate.pending = 0 ∧ transactions = 0 then
        ⟨intermediate,
          { intermediate with pending := state.expected, phase := (state.phase + 1) % phaseModulus },
          true, none⟩
      else ⟨intermediate, intermediate, false, none⟩
    else rejected updated .overArrival
  else rejected state .transactionOverflow

 theorem success_exactly (state : Counters) (delta : Int) (arrivals : Nat) :
    (transition state delta arrivals).fault = none ↔
      transactionFits (state.transactions + delta) ∧ arrivals ≤ state.pending := by
  unfold transition
  dsimp only
  split
  · split
    · split <;> simp_all
    · simp_all [rejected]
  · simp_all [rejected]

 theorem transaction_overflow_frame (state : Counters) (delta : Int) (arrivals : Nat)
    (overflow : ¬transactionFits (state.transactions + delta)) :
    transition state delta arrivals = rejected state .transactionOverflow := by
  simp [transition, overflow]

 theorem overarrival_keeps_transaction_update (state : Counters) (delta : Int) (arrivals : Nat)
    (fits : transactionFits (state.transactions + delta)) (over : state.pending < arrivals) :
    transition state delta arrivals =
      rejected { state with transactions := state.transactions + delta } .overArrival := by
  simp [transition, fits, Nat.not_le.mpr over]

 theorem completion_exactly (state : Counters) (delta : Int) (arrivals : Nat) :
    (transition state delta arrivals).completed = true ↔
      transactionFits (state.transactions + delta) ∧
      arrivals = state.pending ∧ state.transactions + delta = 0 := by
  unfold transition
  dsimp only
  split
  · split
    · split
      · simp_all
        omega
      · simp_all
    · simp_all [rejected]
      omega
  · simp_all [rejected]

 theorem completion_resets_expected (state : Counters) (delta : Int) (arrivals : Nat)
    (completed : (transition state delta arrivals).completed = true) :
    (transition state delta arrivals).after.pending = state.expected ∧
    (transition state delta arrivals).after.phase = (state.phase + 1) % phaseModulus ∧
    (transition state delta arrivals).intermediate.pending = 0 ∧
    (transition state delta arrivals).after.transactions = 0 := by
  obtain ⟨fits, count, transactions⟩ := (completion_exactly state delta arrivals).mp completed
  simp [transition, transactionFits, count, transactions]

 theorem outstanding_transactions_do_not_advance (state : Counters) (delta : Int) (arrivals : Nat)
    (fits : transactionFits (state.transactions + delta)) (count : arrivals ≤ state.pending)
    (outstanding : state.transactions + delta ≠ 0) :
    (transition state delta arrivals).after.phase = state.phase ∧
    (transition state delta arrivals).after.pending = state.pending - arrivals ∧
    (transition state delta arrivals).completed = false := by
  simp [transition, fits, count, outstanding]

 theorem phase_wrap_toggles_parity (phase : Nat) :
    ((phase + 1) % phaseModulus) % 2 = (phase + 1) % 2 := by
  have divides : 2 ∣ phaseModulus := by decide
  exact Nat.mod_mod_of_dvd (phase + 1) divides

def writesMembership (delta : Int) (arrivals : Nat) (completed : Bool) : Bool :=
  decide (0 < arrivals ∨ delta ≠ 0 ∨ completed = true)

 theorem unwritten_input_is_zero (delta : Int) (arrivals : Nat) (completed : Bool)
    (unwritten : writesMembership delta arrivals completed = false) :
    arrivals = 0 ∧ delta = 0 ∧ completed = false := by
  have inactive : ¬(0 < arrivals ∨ delta ≠ 0 ∨ completed = true) := by
    simpa [writesMembership] using unwritten
  have count : arrivals = 0 := by omega
  have deltaZero : delta = 0 := by
    by_contra nonzero
    exact inactive (Or.inr (Or.inl nonzero))
  cases completed <;> simp_all

 theorem unwritten_transition_frame (state : Counters) (delta : Int) (arrivals : Nat)
    (fits : transactionFits state.transactions)
    (unwritten : writesMembership delta arrivals (transition state delta arrivals).completed = false) :
    (transition state delta arrivals).after = state ∧
    (transition state delta arrivals).completed = false ∧
    (transition state delta arrivals).fault = none := by
  obtain ⟨count, deltaZero, notCompleted⟩ := unwritten_input_is_zero _ _ _ unwritten
  subst arrivals
  subst delta
  have nonzero : ¬(state.pending = 0 ∧ state.transactions = 0) := by
    intro completed
    simp [transition, transactionFits, completed] at notCompleted
  simp [transition, fits, nonzero]

 theorem transaction_order_changes_generation :
    let initial : Counters := ⟨1, 0, 1, 0⟩
    let paidFirst := (transition (transition initial (-1) 0).after 1 0).after
    let expectedFirst := (transition (transition initial 1 0).after (-1) 0).after
    paidFirst.phase = 1 ∧ paidFirst.pending = 1 ∧ paidFirst.transactions = 1 ∧
    expectedFirst.phase = 0 ∧ expectedFirst.pending = 0 ∧ expectedFirst.transactions = 1 := by
  decide

end MemoryWrite.BarrierTransition

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierPublication
import MemoryWrite.BarrierReductionRefinement

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite.BarrierTransition

def projectCounters (state : barrier.Counters) : Counters :=
  ⟨state.expected.val, state.pending.val, state.transactions.val, state.phase.val⟩

def projectFault : barrier.Fault → Fault
  | .TransactionOverflow => .transactionOverflow
  | .OverArrival => .overArrival

def projectTransition (result : barrier.Transition) : Transition :=
  ⟨projectCounters result.intermediate, projectCounters result.after,
    result.completed, result.fault.map projectFault⟩

def deltaValue : barrier.TransactionDelta → Int
  | .Add value => value.val
  | .Subtract value => -value.val

 theorem delta_bounds (delta : barrier.TransactionDelta) :
    -(2 ^ 63) ≤ deltaValue delta ∧ deltaValue delta ≤ 2 ^ 63 := by
  cases delta <;> simp only [deltaValue] <;> scalar_tac

@[step]
theorem rust_rejected (state : barrier.Counters) (fault : barrier.Fault) :
    barrier.rejected state fault
      ⦃ result => projectTransition result = rejected (projectCounters state) (projectFault fault) ⦄ := by
  simp [barrier.rejected, projectTransition, projectCounters, rejected]

@[step]
theorem rust_difference (delta : barrier.TransactionDelta) :
    (match delta with
      | .Add value => ok (IScalar.cast .I128 value)
      | .Subtract value => do
        let widened ← lift (IScalar.cast .I128 value)
        -.widened)
      ⦃ result => result.val = deltaValue delta ⦄ := by
  cases delta with
  | Add value =>
    simp [WP.spec_ok, deltaValue]
  | Subtract value =>
    step with IScalar.cast_inBounds_spec .I128 value (by scalar_tac) as ⟨wide, hwide⟩
    step as ⟨negated, hnegated⟩; try scalar_tac
    simp [deltaValue, hnegated, hwide]

@[step]
theorem rust_transition (state : barrier.Counters) (delta : barrier.TransactionDelta)
    (arrivals : U32) :
    barrier.transition state delta arrivals
      ⦃ result => projectTransition result =
        transition (projectCounters state) (deltaValue delta) arrivals.val ⦄ := by
  unfold barrier.transition
  step with rust_difference delta as ⟨difference, hdifference⟩
  step with IScalar.cast_inBounds_spec .I128 state.transactions (by scalar_tac)
    as ⟨current, hcurrent⟩
  have bounds := delta_bounds delta
  have minValue : core.num.I64.MIN.val = -(2 ^ 63) := by decide
  have maxValue : core.num.I64.MAX.val = 2 ^ 63 - 1 := by decide
  step as ⟨transactions, htransactions⟩; try scalar_tac
  have sum : transactions.val = state.transactions.val + deltaValue delta := by scalar_tac
  step with IScalar.cast_inBounds_spec .I128 core.num.I64.MIN (by simp only [minValue]; scalar_tac)
    as ⟨minimum, hminimum⟩
  split
  · rename_i below
    have overflow : ¬transactionFits (state.transactions.val + deltaValue delta) := by
      unfold transactionFits
      scalar_tac
    step with rust_rejected state .TransactionOverflow as ⟨result, hresult⟩
    simpa [transition, projectCounters, projectFault, overflow] using hresult
  · rename_i lower
    step with IScalar.cast_inBounds_spec .I128 core.num.I64.MAX (by simp only [maxValue]; scalar_tac)
      as ⟨maximum, hmaximum⟩
    split
    · rename_i above
      have overflow : ¬transactionFits (state.transactions.val + deltaValue delta) := by
        unfold transactionFits
        scalar_tac
      step with rust_rejected state .TransactionOverflow as ⟨result, hresult⟩
      simpa [transition, projectCounters, projectFault, overflow] using hresult
    · rename_i upper
      have fits : transactionFits (state.transactions.val + deltaValue delta) := by
        unfold transactionFits
        scalar_tac
      step with IScalar.cast_inBounds_spec .I64 transactions (by scalar_tac)
        as ⟨updatedTransactions, hupdated⟩
      step with BarrierReduction.rust_arrival_pending state.pending arrivals as ⟨remaining, hremaining⟩
      cases remaining with
      | none =>
        have over : state.pending.val < arrivals.val := by
          by_contra notOver
          simp [notOver] at hremaining
        step with rust_rejected { state with transactions := updatedTransactions } .OverArrival
          as ⟨result, hresult⟩
        simpa [transition, projectCounters, projectFault, fits, Nat.not_le.mpr over, hupdated, sum] using hresult
      | some remaining =>
        have count : arrivals.val ≤ state.pending.val := by
          by_contra notCount
          have over : state.pending.val < arrivals.val := by omega
          simp [over] at hremaining
        have remainingValue : remaining.val = state.pending.val - arrivals.val := by
          simpa [Nat.not_lt.mpr count] using hremaining
        step with BarrierReduction.rust_phase_completes remaining (updatedTransactions = 0#i64)
          as ⟨completed, hcompleted⟩
        have zero : (updatedTransactions = 0#i64) ↔ state.transactions.val + deltaValue delta = 0 := by
          scalar_tac
        have completedValue : completed = decide
            (state.pending.val - arrivals.val = 0 ∧ state.transactions.val + deltaValue delta = 0) := by
          simpa [remainingValue, zero] using hcompleted
        split
        · rename_i completes
          simp only [lift, bind_ok, WP.spec_ok]
          have condition : state.pending.val - arrivals.val = 0 ∧
              state.transactions.val + deltaValue delta = 0 := by
            simpa [completedValue] using completes
          simp [projectTransition, projectCounters, transition, count, condition,
            remainingValue, hupdated, sum, completes, phaseModulus, transactionFits, U32.size, U32.numBits]
        · rename_i pending
          have condition : ¬(state.pending.val - arrivals.val = 0 ∧
              state.transactions.val + deltaValue delta = 0) := by
            simpa [completedValue] using pending
          simp [projectTransition, projectCounters, transition, fits, count, condition,
            remainingValue, hupdated, sum, pending]

@[step]
theorem rust_writes_phase_membership (delta : barrier.TransactionDelta) (arrivals : U32)
    (completed : Bool) :
    barrier.writes_phase_membership delta arrivals completed
      ⦃ writes => writes = writesMembership (deltaValue delta) arrivals.val completed ⦄ := by
  unfold barrier.writes_phase_membership
  step with BarrierReduction.rust_arrival_writes_pending arrivals as ⟨positive, hpositive⟩
  split
  · rename_i hasArrivals
    have count : 0 < arrivals.val := by simpa [hpositive] using hasArrivals
    simp [writesMembership, count]
  · rename_i noArrivals
    have count : arrivals.val = 0 := by
      have notPositive : ¬0 < arrivals.val := by simpa [hpositive] using noArrivals
      omega
    cases completed <;> cases delta <;> simp [writesMembership, deltaValue, count]
    all_goals
      apply Bool.eq_iff_iff.mpr
      simp only [bne_iff_ne]
      scalar_tac

@[step]
theorem rust_publication_parity (outcome : barrier.Transition) :
    barrier.publication_parity outcome
      ⦃ parity => parity.map UScalar.val =
        if outcome.completed then some (outcome.intermediate.phase.val % 2) else none ⦄ := by
  unfold barrier.publication_parity
  split
  · step as ⟨parity, hparity⟩; try scalar_tac
    simp [hparity]
  · simp

 theorem source_nonwriting_transition_frames_counters (source : barrier.Counters)
    (delta : barrier.TransactionDelta) (arrivals : U32) :
    (do
      let outcome ← barrier.transition source delta arrivals
      let writes ← barrier.writes_phase_membership delta arrivals outcome.completed
      pure (outcome, writes))
      ⦃ result => result.2 = false → projectCounters result.1.after = projectCounters source ∧
        result.1.completed = false ∧ result.1.fault = none ⦄ := by
  step with rust_transition source delta arrivals as ⟨outcome, houtcome⟩
  step with rust_writes_phase_membership delta arrivals outcome.completed as ⟨writes, hwrites⟩
  intro unwritten
  have fits : transactionFits (projectCounters source).transactions := by
    unfold transactionFits projectCounters
    scalar_tac
  have modelUnwritten : writesMembership (deltaValue delta) arrivals.val
      (transition (projectCounters source) (deltaValue delta) arrivals.val).completed = false := by
    rw [← houtcome]
    simpa [hwrites, projectTransition] using unwritten
  have framed := unwritten_transition_frame (projectCounters source) (deltaValue delta)
    arrivals.val fits modelUnwritten
  rw [← houtcome] at framed
  refine ⟨framed.1, framed.2.1, ?_⟩
  have noFault : outcome.fault.map projectFault = none := framed.2.2
  simpa using noFault

 theorem source_advance_refines_publication (state : FullState) (source : barrier.Counters)
    (delta : barrier.TransactionDelta) (arrivals : U32) (actor : Nat) (release : BarrierReduction.Clock)
    (projection : projectCounters source = state.counters) :
    barrier.transition source delta arrivals
      ⦃ outcome => publish state actor release (projectTransition outcome) =
        advance state actor (deltaValue delta) arrivals.val release ⦄ := by
  step with rust_transition source delta arrivals as ⟨outcome, houtcome⟩
  simp only [advance, houtcome, projection]

 theorem source_completion_wait_acquires_release (state : FullState) (source : barrier.Counters)
    (delta : barrier.TransactionDelta) (arrivals : U32) (actor waiter : Nat)
    (release : BarrierReduction.Clock) (projection : projectCounters source = state.counters) :
    barrier.transition source delta arrivals
      ⦃ outcome => outcome.completed = true →
        ∃ next, wait (publish state actor release (projectTransition outcome)) waiter
            state.counters.phase = some next ∧
          next.acquired waiter = BarrierReduction.join (state.acquired waiter)
            (BarrierReduction.join state.currentRelease release) ∧
          next.findings = (publish state actor release (projectTransition outcome)).findings ⦄ := by
  step with rust_transition source delta arrivals as ⟨outcome, houtcome⟩
  rename_i completed
  have completion : (transition state.counters (deltaValue delta) arrivals.val).completed = true := by
    rw [← projection, ← houtcome]
    exact completed
  simpa only [advance, ← projection, ← houtcome] using
    completion_wait_acquires_release state actor waiter (deltaValue delta) arrivals.val release completion

end MemoryWrite.BarrierTransition

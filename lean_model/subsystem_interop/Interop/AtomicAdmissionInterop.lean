/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.AtomicResult
import MemoryWrite.RustAtomicLocalCoverage

open Aeneas Aeneas.Std Result

set_option autoImplicit false

namespace Interop.AtomicAdmission

abbrev Claim := MemoryWrite.AtomicLocalCoverage.Claim

def fromIR (front : PtxvIr.Front) (operation : PtxvIr.AtomOp)
    (word predicate : Option PtxvIr.Reg) (scalarRed collecting continued shared : Bool)
    (claims : Slice Claim) : Result Bool := do
  let monitored ← PtxvIr.atomic_result.local_order_monitored front operation word predicate
  PtxvMemoryWrite.atomic_scalar.local_claims_unmodeled
    monitored scalarRed collecting continued shared claims

theorem typed_source_admission_covers_every_selected_pair
    (front : PtxvIr.Front) (operation : PtxvIr.AtomOp)
    (word predicate : Option PtxvIr.Reg) (scalarRed collecting continued shared : Bool)
    (claims : Slice Claim) :
    fromIR front operation word predicate scalarRed collecting continued shared claims
      ⦃ result => result = true ↔ MemoryWrite.AtomicLaneScan.AnyPair
        (MemoryWrite.AtomicLocalCoverage.Reports
          (AtomicResult.localPolicy front operation word predicate)
          scalarRed collecting continued shared) claims.val 0 ⦄ := by
  unfold fromIR
  rw [AtomicResult.source_local_order_monitored]
  simp only [bind_ok]
  exact MemoryWrite.AtomicLocalCoverage.rust_local_claims_unmodeled
    (AtomicResult.localPolicy front operation word predicate)
    scalarRed collecting continued shared claims

theorem actual_compare_exchange_requires_a_continuation_or_diagnostic
    (front : PtxvIr.Front) (operation : PtxvIr.AtomOp)
    (word predicate : Option PtxvIr.Reg) (scalarRed collecting continued shared : Bool)
    (claims : Slice Claim) (comparison : operation = .Cas ∨ operation = .CasSpin) :
    fromIR front operation word predicate scalarRed collecting continued shared claims
      ⦃ result => result = false → ∀ first second, first < second → second < claims.val.length →
        MemoryWrite.AtomicReturningCoverage.interferes
          (MemoryWrite.AtomicLocalCoverage.sourcePair continued shared
            claims.val[first]! claims.val[second]!) = true → continued = true ⦄ := by
  have selected : AtomicResult.localPolicy front operation word predicate = true := by
    rcases comparison with cas | spin <;> subst operation <;>
      simp [AtomicResult.localPolicy,AtomicResult.compareExchangeMemory]
  unfold fromIR
  rw [AtomicResult.source_local_order_monitored,selected]
  simp only [bind_ok]
  exact MemoryWrite.AtomicLocalCoverage.source_monitored_interfering_pairs_have_a_continuation_or_diagnostic
    scalarRed collecting continued shared claims

theorem actual_observable_output_requires_a_continuation_or_diagnostic
    (front : PtxvIr.Front) (operation : PtxvIr.AtomOp)
    (word predicate : Option PtxvIr.Reg) (scalarRed collecting continued shared : Bool)
    (claims : Slice Claim)
    (output : word.isSome = true ∨ ∃ register, predicate = some register ∧
      RegisterOwner.immutablePredicate front register = false) :
    fromIR front operation word predicate scalarRed collecting continued shared claims
      ⦃ result => result = false → ∀ first second, first < second → second < claims.val.length →
        MemoryWrite.AtomicReturningCoverage.interferes
          (MemoryWrite.AtomicLocalCoverage.sourcePair continued shared
            claims.val[first]! claims.val[second]!) = true → continued = true ⦄ := by
  have observed : AtomicResult.outputObserved front word predicate = true :=
    (AtomicResult.outputObserved_iff front word predicate).mpr output
  have selected : AtomicResult.localPolicy front operation word predicate = true := by
    simp [AtomicResult.localPolicy,observed]
  unfold fromIR
  rw [AtomicResult.source_local_order_monitored,selected]
  simp only [bind_ok]
  exact MemoryWrite.AtomicLocalCoverage.source_monitored_interfering_pairs_have_a_continuation_or_diagnostic
    scalarRed collecting continued shared claims

theorem actual_ordinary_returnless_float_has_no_undiagnosed_local_collision
    (front : PtxvIr.Front) (claims : Slice Claim) :
    fromIR front .FAddFtz none none true false false false claims
      ⦃ result => result = false → ∀ first second, first < second → second < claims.val.length →
        MemoryWrite.AtomicLocalCoverage.interferes false
          (MemoryWrite.AtomicLocalCoverage.sourcePair false false
            claims.val[first]! claims.val[second]!) = false ⦄ := by
  have selected : AtomicResult.localPolicy front .FAddFtz none none = false := by
    simp [AtomicResult.localPolicy,AtomicResult.compareExchangeMemory,AtomicResult.exchangeMemory,
      AtomicResult.outputObserved,AtomicResult.predicateObserved]
  unfold fromIR
  rw [AtomicResult.source_local_order_monitored,selected]
  simp only [bind_ok]
  exact MemoryWrite.AtomicLocalCoverage.source_ordinary_red_claims_have_no_unmodeled_local_collision claims

theorem actual_exchange_requires_a_continuation_or_diagnostic
    (front : PtxvIr.Front) (word predicate : Option PtxvIr.Reg)
    (scalarRed collecting continued shared : Bool) (claims : Slice Claim) :
    fromIR front .Exch word predicate scalarRed collecting continued shared claims
      ⦃ result => result = false → ∀ first second, first < second → second < claims.val.length →
        MemoryWrite.AtomicReturningCoverage.interferes
          (MemoryWrite.AtomicLocalCoverage.sourcePair continued shared
            claims.val[first]! claims.val[second]!) = true → continued = true ⦄ := by
  have selected : AtomicResult.localPolicy front .Exch word predicate = true := by
    simp [AtomicResult.localPolicy,AtomicResult.exchangeMemory]
  unfold fromIR
  rw [AtomicResult.source_local_order_monitored,selected]
  simp only [bind_ok]
  exact MemoryWrite.AtomicLocalCoverage.source_monitored_interfering_pairs_have_a_continuation_or_diagnostic
    scalarRed collecting continued shared claims

end Interop.AtomicAdmission

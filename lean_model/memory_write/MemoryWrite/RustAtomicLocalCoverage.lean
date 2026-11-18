/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicLocalCoverage
import MemoryWrite.RustAtomicReturningCoverage

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.AtomicLocalCoverage

abbrev Claim := AtomicLaneScan.Claim

def sourcePair (continued shared : Bool) (left right : Claim) : Pair :=
  AtomicReturningCoverage.sourcePair continued shared left right

def Reports (monitored scalarRed collecting continued shared : Bool) (left right : Claim) : Prop :=
  needsDiagnostic monitored scalarRed collecting (sourcePair continued shared left right) = true

@[step] theorem rust_local_claims_unmodeled
    (monitored scalarRed collecting continued shared : Bool) (claims : Slice Claim) :
    atomic_scalar.local_claims_unmodeled monitored scalarRed collecting continued shared claims
      ⦃ result => result = true ↔ AtomicLaneScan.AnyPair
        (Reports monitored scalarRed collecting continued shared) claims.val 0 ⦄ := by
  cases monitored
  · unfold atomic_scalar.local_claims_unmodeled
    simp only [Bool.false_eq_true, ↓reduceIte, bind_ok]
    step with AtomicReturningCoverage.rust_returned_word_claims_unmodeled
      continued false claims as ⟨conflict,specified⟩
    cases scalarRed <;> cases collecting <;> cases continued <;> cases shared <;>
      simp_all [WP.spec_ok,AtomicLaneScan.AnyPair,AtomicReturningCoverage.Reports,
        AtomicReturningCoverage.sourcePair,AtomicLaneScan.pairModel,Reports,sourcePair,
        needsDiagnostic,selected,interferes,AtomicReturningCoverage.needsDiagnostic,
        AtomicReturningCoverage.interferes]
  · unfold atomic_scalar.local_claims_unmodeled
    simp only [↓reduceIte, bind_ok]
    step with AtomicReturningCoverage.rust_returned_word_claims_unmodeled
      continued shared claims as ⟨conflict,specified⟩
    cases scalarRed <;> cases collecting <;> cases continued <;> cases shared <;>
      simp_all [AtomicLaneScan.AnyPair,AtomicReturningCoverage.Reports,
        AtomicReturningCoverage.sourcePair,AtomicLaneScan.pairModel,Reports,sourcePair,
        needsDiagnostic,selected,interferes,AtomicReturningCoverage.needsDiagnostic,
        AtomicReturningCoverage.interferes]

@[step] theorem source_monitored_interfering_pairs_have_a_continuation_or_diagnostic
    (scalarRed collecting continued shared : Bool) (claims : Slice Claim) :
    atomic_scalar.local_claims_unmodeled true scalarRed collecting continued shared claims
      ⦃ result => result = false → ∀ first second, first < second → second < claims.val.length →
        AtomicReturningCoverage.interferes
          (sourcePair continued shared claims.val[first]! claims.val[second]!) = true →
        continued = true ⦄ := by
  apply WP.spec_mono (rust_local_claims_unmodeled true scalarRed collecting continued shared claims)
  intro result specified absent first second ordered inside conflicting
  cases continuation : continued
  · have reported : Reports true scalarRed collecting continued shared claims.val[first]! claims.val[second]! := by
      rw [Reports,certificate_collection_preserves_monitored_atomic_obligations]
      exact AtomicReturningCoverage.an_uncontinued_interfering_returned_word_pair_is_diagnosed _
        continuation conflicting
    have found := specified.mpr ⟨first,second,by omega,ordered,inside,reported⟩
    simp [absent] at found
  · rfl

@[step] theorem source_ordinary_red_claims_have_no_unmodeled_local_collision
    (claims : Slice Claim) :
    atomic_scalar.local_claims_unmodeled false true false false false claims
      ⦃ result => result = false → ∀ first second, first < second → second < claims.val.length →
        interferes false (sourcePair false false claims.val[first]! claims.val[second]!) = false ⦄ := by
  apply WP.spec_mono (rust_local_claims_unmodeled false true false false false claims)
  intro result specified absent first second ordered inside
  by_cases collision : interferes false (sourcePair false false claims.val[first]! claims.val[second]!) = true
  · have reported : Reports false true false false false claims.val[first]! claims.val[second]! := by
      simpa [Reports,needsDiagnostic,selected,sourcePair,
        AtomicReturningCoverage.sourcePair,AtomicLaneScan.pairModel] using collision
    have found := specified.mpr ⟨first,second,by omega,ordered,inside,reported⟩
    simp [absent] at found
  · exact Bool.eq_false_iff.mpr collision

end MemoryWrite.AtomicLocalCoverage

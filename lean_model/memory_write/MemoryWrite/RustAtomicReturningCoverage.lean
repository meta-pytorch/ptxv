/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicReturningCoverage
import MemoryWrite.RustAtomicLaneScan

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.AtomicReturningCoverage

abbrev Claim := AtomicLaneScan.Claim

def sourcePair (continued shared : Bool) (left right : Claim) : Pair :=
  let pair := AtomicLaneScan.pairModel true true continued shared left right
  ⟨pair.continued,pair.sameWarp,pair.sameWord,pair.sharedResult⟩

def Reports (continued shared : Bool) (left right : Claim) : Prop :=
  needsDiagnostic (sourcePair continued shared left right) = true

@[step] theorem rust_returned_word_claims_unmodeled
    (continued shared : Bool) (claims : Slice Claim) :
    atomic_scalar.returned_word_claims_unmodeled continued shared claims
      ⦃ result => result = true ↔ AtomicLaneScan.AnyPair (Reports continued shared) claims.val 0 ⦄ := by
  unfold atomic_scalar.returned_word_claims_unmodeled
  apply WP.spec_mono (AtomicLaneScan.rust_unmodeled_claims true true continued shared claims)
  intro result specified
  simpa [AtomicLaneScan.AnyPair,AtomicLaneScan.Reports,Reports,sourcePair,
    AtomicLaneScan.pairModel,AtomicLaneCoverage.needsDiagnostic,AtomicLaneCoverage.interferes,
    needsDiagnostic,interferes] using specified

@[step] theorem source_every_undiagnosed_interfering_returned_word_pair_has_a_continuation
    (continued shared : Bool) (claims : Slice Claim) :
    atomic_scalar.returned_word_claims_unmodeled continued shared claims
      ⦃ result => result = false → ∀ first second, first < second → second < claims.val.length →
        interferes (sourcePair continued shared claims.val[first]! claims.val[second]!) = true →
        continued = true ⦄ := by
  apply WP.spec_mono (rust_returned_word_claims_unmodeled continued shared claims)
  intro result specified absent first second ordered inside conflicting
  cases continuation : continued
  · have reported : Reports continued shared claims.val[first]! claims.val[second]! := by
      apply an_uncontinued_interfering_returned_word_pair_is_diagnosed
      · exact continuation
      · exact conflicting
    have found := specified.mpr ⟨first,second,by omega,ordered,inside,reported⟩
    simp [absent] at found
  · rfl

end MemoryWrite.AtomicReturningCoverage

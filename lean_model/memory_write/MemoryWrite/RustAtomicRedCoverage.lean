/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicRedCoverage
import MemoryWrite.RustAtomicLocalCoverage

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.AtomicRedCoverage

abbrev Claim := AtomicLaneScan.Claim

def sourcePair (collecting : Bool) (left right : Claim) : Pair :=
  let pair := AtomicLaneScan.pairModel true true false false left right
  ⟨collecting,pair.sameWarp,pair.sameWord⟩

def Reports (collecting : Bool) (left right : Claim) : Prop :=
  needsDiagnostic (sourcePair collecting left right) = true

@[step] theorem rust_red_word_claims_unmodeled
    (collecting : Bool) (claims : Slice Claim) :
    atomic_scalar.local_claims_unmodeled false true collecting false false claims
      ⦃ result => result = true ↔ AtomicLaneScan.AnyPair (Reports collecting) claims.val 0 ⦄ := by
  apply WP.spec_mono (AtomicLocalCoverage.rust_local_claims_unmodeled false true collecting false false claims)
  intro result specified
  simpa [AtomicLocalCoverage.Reports,AtomicLocalCoverage.needsDiagnostic,
    AtomicLocalCoverage.selected,AtomicLocalCoverage.interferes,AtomicLocalCoverage.sourcePair,
    AtomicReturningCoverage.sourcePair,AtomicLaneScan.AnyPair,Reports,sourcePair,
    AtomicLaneScan.pairModel,needsDiagnostic,interferes] using specified

@[step] theorem source_ordinary_undiagnosed_red_claims_have_no_local_collision
    (claims : Slice Claim) :
    atomic_scalar.local_claims_unmodeled false true false false false claims
      ⦃ result => result = false → ∀ first second, first < second → second < claims.val.length →
        interferes (sourcePair false claims.val[first]! claims.val[second]!) = false ⦄ := by
  apply WP.spec_mono (rust_red_word_claims_unmodeled false claims)
  intro result specified absent first second ordered inside
  by_cases collision : interferes (sourcePair false claims.val[first]! claims.val[second]!) = true
  · have reported : Reports false claims.val[first]! claims.val[second]! := by
      apply ordinary_colliding_reduction_requires_order_coverage
      · rfl
      · exact collision
    have found := specified.mpr ⟨first,second,by omega,ordered,inside,reported⟩
    simp [absent] at found
  · exact Bool.eq_false_iff.mpr collision

end MemoryWrite.AtomicRedCoverage

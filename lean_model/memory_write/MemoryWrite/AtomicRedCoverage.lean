/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicLaneCoverage

set_option autoImplicit false

namespace MemoryWrite.AtomicRedCoverage

structure Pair where
  certificateCollection : Bool
  sameWarp : Bool
  sameWord : Bool

def interferes (pair : Pair) : Bool :=
  pair.sameWarp && pair.sameWord

def needsDiagnostic (pair : Pair) : Bool :=
  !pair.certificateCollection && interferes pair

def legacy (pair : Pair) : AtomicLaneCoverage.Pair :=
  ⟨true,!pair.certificateCollection,false,pair.sameWarp,pair.sameWord,false⟩

theorem legacy_agrees (pair : Pair) :
    AtomicLaneCoverage.needsDiagnostic (legacy pair) = needsDiagnostic pair := by
  simp [AtomicLaneCoverage.needsDiagnostic,AtomicLaneCoverage.interferes,
    legacy,needsDiagnostic,interferes]

theorem ordinary_colliding_reduction_requires_order_coverage (pair : Pair)
    (ordinary : pair.certificateCollection = false) (overlap : interferes pair = true) :
    needsDiagnostic pair = true := by
  simp [needsDiagnostic,ordinary,overlap]

theorem ordinary_undiagnosed_pair_has_no_local_collision (pair : Pair)
    (ordinary : pair.certificateCollection = false) (absent : needsDiagnostic pair = false) :
    interferes pair = false := by
  simpa [needsDiagnostic,ordinary] using absent

theorem certificate_collection_defers_this_check (pair : Pair)
    (collecting : pair.certificateCollection = true) :
    needsDiagnostic pair = false := by
  simp [needsDiagnostic,collecting]

theorem disjoint_words_or_other_warps_do_not_trigger (pair : Pair)
    (separate : pair.sameWarp = false ∨ pair.sameWord = false) :
    needsDiagnostic pair = false := by
  rcases separate with warp | word
  · simp [needsDiagnostic,interferes,warp]
  · simp [needsDiagnostic,interferes,word]

theorem the_three_colliding_cancellation_claims_are_not_admitted_as_complete :
    needsDiagnostic ⟨false,true,true⟩ = true := by rfl

end MemoryWrite.AtomicRedCoverage

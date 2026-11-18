/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicLaneCoverage

set_option autoImplicit false

namespace MemoryWrite.AtomicReturningCoverage

structure Pair where
  continued : Bool
  sameWarp : Bool
  sameWord : Bool
  sharedResult : Bool

def interferes (pair : Pair) : Bool :=
  pair.sameWarp && (pair.sameWord || pair.sharedResult)

def needsDiagnostic (pair : Pair) : Bool :=
  !pair.continued && interferes pair

def legacy (pair : Pair) : AtomicLaneCoverage.Pair :=
  ⟨true,true,pair.continued,pair.sameWarp,pair.sameWord,pair.sharedResult⟩

theorem legacy_agrees (pair : Pair) :
    AtomicLaneCoverage.needsDiagnostic (legacy pair) = needsDiagnostic pair := by
  simp [AtomicLaneCoverage.needsDiagnostic,AtomicLaneCoverage.interferes,
    legacy,needsDiagnostic,interferes]

theorem an_uncontinued_interfering_returned_word_pair_is_diagnosed (pair : Pair)
    (unsupported : pair.continued = false) (conflicting : interferes pair = true) :
    needsDiagnostic pair = true := by
  simp [needsDiagnostic,unsupported,conflicting]

theorem an_undiagnosed_interfering_returned_word_pair_has_a_continuation (pair : Pair)
    (conflicting : interferes pair = true) (undiagnosed : needsDiagnostic pair = false) :
    pair.continued = true := by
  cases continuation : pair.continued <;> simp_all [needsDiagnostic]

theorem a_single_claim_cannot_form_an_interfering_pair :
    ¬∃ first second : Nat, first < second ∧ second < 1 := by omega

theorem a_continuation_or_private_words_avoid_this_diagnostic (pair : Pair)
    (covered : pair.continued = true ∨ pair.sameWord = false ∧ pair.sharedResult = false) :
    needsDiagnostic pair = false := by
  rcases covered with continued | ⟨privateWord,privateReturn⟩
  · simp [needsDiagnostic,continued]
  · simp [needsDiagnostic,interferes,privateWord,privateReturn]

end MemoryWrite.AtomicReturningCoverage

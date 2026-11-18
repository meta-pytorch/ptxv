/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicReturningCoverage
import MemoryWrite.AtomicRedCoverage

set_option autoImplicit false

namespace MemoryWrite.AtomicLocalCoverage

abbrev Pair := AtomicReturningCoverage.Pair

def selected (monitored scalarRed collecting : Bool) : Bool :=
  monitored || (scalarRed && !collecting)

def interferes (monitored : Bool) (pair : Pair) : Bool :=
  pair.sameWarp && (pair.sameWord || (monitored && pair.sharedResult))

def needsDiagnostic (monitored scalarRed collecting : Bool) (pair : Pair) : Bool :=
  selected monitored scalarRed collecting && !pair.continued && interferes monitored pair

theorem certificate_collection_preserves_monitored_atomic_obligations
    (scalarRed collecting : Bool) (pair : Pair) :
    needsDiagnostic true scalarRed collecting pair = AtomicReturningCoverage.needsDiagnostic pair := by
  simp [needsDiagnostic,selected,interferes,AtomicReturningCoverage.needsDiagnostic,
    AtomicReturningCoverage.interferes]

theorem a_monitored_interfering_pair_needs_a_continuation_or_diagnostic
    (scalarRed collecting : Bool) (pair : Pair)
    (conflicting : AtomicReturningCoverage.interferes pair = true)
    (absent : needsDiagnostic true scalarRed collecting pair = false) :
    pair.continued = true := by
  apply AtomicReturningCoverage.an_undiagnosed_interfering_returned_word_pair_has_a_continuation pair conflicting
  simpa [certificate_collection_preserves_monitored_atomic_obligations] using absent

theorem ordinary_red_collision_without_a_continuation_is_diagnosed
    (pair : Pair) (notContinued : pair.continued = false)
    (sameWarp : pair.sameWarp = true) (sameWord : pair.sameWord = true) :
    needsDiagnostic false true false pair = true := by
  simp [needsDiagnostic,selected,interferes,notContinued,sameWarp,sameWord]

theorem certificate_collection_defers_only_unmonitored_red (pair : Pair) :
    needsDiagnostic false true true pair = false := by
  simp [needsDiagnostic,selected]

end MemoryWrite.AtomicLocalCoverage

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicComponentTrace

set_option autoImplicit false

namespace MemoryWrite.AtomicLaneCoverage

structure Pair where
  ptx : Bool
  returning : Bool
  continued : Bool
  sameWarp : Bool
  sameWord : Bool
  sharedResult : Bool

def interferes (pair : Pair) : Bool :=
  pair.sameWarp && (pair.sameWord || pair.sharedResult)

def needsDiagnostic (pair : Pair) : Bool :=
  pair.ptx && pair.returning && !pair.continued && interferes pair

theorem an_uncontinued_returning_interfering_pair_is_diagnosed (pair : Pair)
    (front : pair.ptx = true) (returning : pair.returning = true)
    (noContinuation : pair.continued = false) (conflict : interferes pair = true) :
    needsDiagnostic pair = true := by
  simp [needsDiagnostic, front, returning, noContinuation, conflict]

theorem an_undiagnosed_returning_interfering_pair_has_a_continuation (pair : Pair)
    (front : pair.ptx = true) (returning : pair.returning = true)
    (conflict : interferes pair = true) (undiagnosed : needsDiagnostic pair = false) :
    pair.continued = true := by
  cases continuation : pair.continued <;> simp_all [needsDiagnostic]

theorem continued_pairs_do_not_add_a_diagnostic (pair : Pair)
    (continued : pair.continued = true) : needsDiagnostic pair = false := by
  simp [needsDiagnostic, continued]

theorem different_warps_do_not_add_this_diagnostic (pair : Pair)
    (different : pair.sameWarp = false) : needsDiagnostic pair = false := by
  simp [needsDiagnostic, interferes, different]

theorem disjoint_private_pairs_do_not_add_this_diagnostic (pair : Pair)
    (differentWord : pair.sameWord = false) (privateReturn : pair.sharedResult = false) :
    needsDiagnostic pair = false := by
  simp [needsDiagnostic, interferes, differentWord, privateReturn]

theorem shared_return_requires_coverage_even_when_words_differ (ptx returning continued : Bool) :
    needsDiagnostic ⟨ptx,returning,continued,true,false,true⟩ =
      (ptx && returning && !continued) := by
  simp [needsDiagnostic, interferes]

theorem returning_overlap_without_a_continuation_is_a_gap :
    needsDiagnostic ⟨true,true,false,true,true,false⟩ = true := by rfl

end MemoryWrite.AtomicLaneCoverage

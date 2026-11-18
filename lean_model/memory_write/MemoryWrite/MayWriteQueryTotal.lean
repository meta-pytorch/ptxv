/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteQueryProgress
import MemoryWrite.MayWriteQuery
import MemoryWrite.MayWriteAnalysisTotal

open Aeneas Aeneas.Std Result PtxvMemoryWrite
set_option autoImplicit false
namespace MemoryWrite.MayWrite

theorem may_write_sets_returns_and_queries_catch_every_path_write
    (graph : cache_graph.MayWriteGraph) (start : Usize) (bit : U32)
    (bounded : GraphBounded graph) (start_bound : start.val < graph.reach.val.length)
    (fits : AnalysisFits graph) :
    ∃ sets, cache_graph.may_write_sets graph start bit = .ok sets ∧
      ∀ (pc register : Usize), WrittenBefore (graphView graph bit) start.val pc.val register.val →
        cache_graph.may_at sets pc (some register) = .ok true := by
  obtain ⟨sets, analyzed, _⟩ := MemoryWrite.spec_returns _ _
    (rust_may_write_sets_total graph start bit bounded start_bound fits)
  refine ⟨sets, analyzed, ?_⟩
  intro pc register written
  obtain ⟨answer, queried, _⟩ := MemoryWrite.spec_returns _ _
    (rust_may_at_total sets pc (some register))
  have answer_true := may_write_sets_query_catches_every_path_write graph start pc register bit
    bounded start_bound sets analyzed written answer queried
  simpa [answer_true] using queried

end MemoryWrite.MayWrite

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AnalysisCacheMayWrite
import MemoryWrite.MayWriteQuery

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite.AnalysisCache

theorem rust_accepted_may_write_cache_catches_every_path_write {Program : Type}
    (graph : Program → cache_graph.MayWriteGraph)
    {initialGeneration finalGeneration : U64} {versions : List (RustProgramVersion Program)}
    (issued : RustVersionIssuance Program initialGeneration versions finalGeneration)
    {old current : RustProgramVersion Program}
    (hold : old ∈ versions) (hcurrent : current ∈ versions)
    (oldPc currentPc oldBarrier currentBarrier : U32) (recorded query : cache.CacheKey)
    (producerKey : cache.make_key old.generation oldPc oldBarrier = .ok recorded)
    (queryKey : cache.make_key current.generation currentPc currentBarrier = .ok query)
    (replay : cache.Replay) (begin : cache.begin_replay recorded query = .ok replay)
    (finish : cache.finish_replay replay true = .ok (some true))
    (cached : cache_graph.MayWriteSets)
    (produced : cache_graph.may_write_sets (graph old.program)
      (UScalar.cast .Usize oldPc) oldBarrier = .ok cached)
    (bounded : MayWrite.GraphBounded (graph current.program))
    (startBound : (UScalar.cast .Usize currentPc).val < (graph current.program).reach.val.length)
    (pc register : Usize)
    (written : MayWrite.WrittenBefore (MayWrite.graphView (graph current.program) currentBarrier)
      (UScalar.cast .Usize currentPc).val pc.val register.val)
    (answer : Bool) (queried : cache_graph.may_at cached pc (some register) = .ok answer) :
    answer = true := by
  have fresh := rust_may_write_cache_equals_fresh graph issued hold hcurrent oldPc currentPc
    oldBarrier currentBarrier recorded query producerKey queryKey replay begin finish cached produced
  exact MayWrite.may_write_sets_query_catches_every_path_write (graph current.program)
    (UScalar.cast .Usize currentPc) pc register currentBarrier bounded startBound cached fresh
    written answer queried

end MemoryWrite.AnalysisCache

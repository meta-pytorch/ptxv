/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AnalysisCacheReplay

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite.AnalysisCache

theorem rust_may_write_cache_equals_fresh {Program : Type}
    (graph : Program → cache_graph.MayWriteGraph)
    {initialGeneration finalGeneration : U64} {versions : List (RustProgramVersion Program)}
    (issued : RustVersionIssuance Program initialGeneration versions finalGeneration)
    {old current : RustProgramVersion Program}
    (hold : old ∈ versions) (hcurrent : current ∈ versions)
    (oldPc currentPc oldBarrier currentBarrier : U32)
    (recorded query : cache.CacheKey)
    (producerKey : cache.make_key old.generation oldPc oldBarrier = .ok recorded)
    (queryKey : cache.make_key current.generation currentPc currentBarrier = .ok query)
    (replay : cache.Replay)
    (begin : cache.begin_replay recorded query = .ok replay)
    (finish : cache.finish_replay replay true = .ok (some true))
    (cached : cache_graph.MayWriteSets)
    (produced : cache_graph.may_write_sets (graph old.program)
      (UScalar.cast .Usize oldPc) oldBarrier = .ok cached) :
    cache_graph.may_write_sets (graph current.program)
      (UScalar.cast .Usize currentPc) currentBarrier = .ok cached := by
  have accepted := rust_replay_acceptance_requires_semantic_inputs recorded query replay replay
    [] true true begin (.nil replay) finish
  have oldKey := rust_make_key_spec old.generation oldPc oldBarrier
  have currentKey := rust_make_key_spec current.generation currentPc currentBarrier
  simp only [producerKey, WP.spec_ok] at oldKey
  simp only [queryKey, WP.spec_ok] at currentKey
  have hkey : Key.mk old.generation.val oldPc.val oldBarrier.val =
      Key.mk current.generation.val currentPc.val currentBarrier.val :=
    oldKey.symm.trans (accepted.1.trans currentKey)
  have hg : (versionView old).generation = (versionView current).generation :=
    congrArg Key.generation hkey
  have hp : old.program = current.program :=
    equal_generation_same_program issued.model.versioned
      (List.mem_map.mpr ⟨old, hold, rfl⟩) (List.mem_map.mpr ⟨current, hcurrent, rfl⟩) hg
  have hpc : oldPc = currentPc := UScalar.eq_of_val_eq (congrArg Key.pc hkey)
  have hbarrier : oldBarrier = currentBarrier :=
    UScalar.eq_of_val_eq (congrArg Key.barrier hkey)
  simpa only [hp, hpc, hbarrier] using produced

theorem rust_may_write_cached_query_equals_fresh {Program : Type}
    (graph : Program → cache_graph.MayWriteGraph)
    {initialGeneration finalGeneration : U64} {versions : List (RustProgramVersion Program)}
    (issued : RustVersionIssuance Program initialGeneration versions finalGeneration)
    {old current : RustProgramVersion Program}
    (hold : old ∈ versions) (hcurrent : current ∈ versions)
    (oldPc currentPc oldBarrier currentBarrier : U32)
    (recorded query : cache.CacheKey)
    (producerKey : cache.make_key old.generation oldPc oldBarrier = .ok recorded)
    (queryKey : cache.make_key current.generation currentPc currentBarrier = .ok query)
    (replay : cache.Replay)
    (begin : cache.begin_replay recorded query = .ok replay)
    (finish : cache.finish_replay replay true = .ok (some true))
    (cached : cache_graph.MayWriteSets)
    (produced : cache_graph.may_write_sets (graph old.program)
      (UScalar.cast .Usize oldPc) oldBarrier = .ok cached)
    (pc : Usize) (register : Option Usize) :
    (do
      let fresh ← cache_graph.may_write_sets (graph current.program)
        (UScalar.cast .Usize currentPc) currentBarrier
      cache_graph.may_at fresh pc register) = cache_graph.may_at cached pc register := by
  rw [rust_may_write_cache_equals_fresh graph issued hold hcurrent oldPc currentPc
    oldBarrier currentBarrier recorded query producerKey queryKey replay begin finish cached produced]
  simp only [bind_ok]

end MemoryWrite.AnalysisCache

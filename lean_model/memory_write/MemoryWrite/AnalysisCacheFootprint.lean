/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AnalysisCacheRefinement

set_option autoImplicit false

namespace MemoryWrite.AnalysisCache

theorem run_records_exact_reads {ReadKey Word Value : Type}
    (analysis : Analysis ReadKey Word Value) (environment : ReadKey → Option Word) :
    Agrees (analysis.run environment).2 environment := by
  induction analysis with
  | done value => simp [Analysis.run, Agrees]
  | read key next ih =>
    intro read hm
    simp only [Analysis.run, List.mem_cons] at hm
    rcases hm with heq | htail
    · cases heq
      rfl
    · exact ih (environment key) read htail

def footprint {ReadKey Word : Type} [DecidableEq ReadKey]
    (reads : List (ReadKey × Option Word)) (environment : ReadKey → Option Word) :
    List (ReadKey × Option Word) :=
  (reads.map Prod.fst).dedup.map (fun key => (key, environment key))

def Covers {ReadKey Word : Type}
    (recorded trace : List (ReadKey × Option Word)) : Prop :=
  ∀ read ∈ trace, read ∈ recorded

theorem footprint_covers {ReadKey Word : Type} [DecidableEq ReadKey]
    (reads : List (ReadKey × Option Word)) (environment : ReadKey → Option Word)
    (consistent : Agrees reads environment) : Covers (footprint reads environment) reads := by
  intro read hm
  apply List.mem_map.mpr
  refine ⟨read.1, ?_, ?_⟩
  · simp only [List.mem_dedup]
    exact List.mem_map.mpr ⟨read, hm, rfl⟩
  · exact Prod.ext rfl (consistent read hm)

theorem agrees_of_covers {ReadKey Word : Type}
    {recorded trace : List (ReadKey × Option Word)}
    (covers : Covers recorded trace) (environment : ReadKey → Option Word)
    (agrees : Agrees recorded environment) : Agrees trace environment := by
  intro read hm
  exact agrees read (covers read hm)

def FootprintedProduced {Program ReadKey Word Value : Type}
    (analysis : Program → Nat → Nat → Analysis ReadKey Word Value)
    (version : ProgramVersion Program) (environment : ReadKey → Option Word)
    (entry : Entry ReadKey Word Value) : Prop :=
  entry.key.generation = version.generation ∧
  entry.value = ((analysis version.program entry.key.pc entry.key.barrier).run environment).1 ∧
  Covers entry.reads ((analysis version.program entry.key.pc entry.key.barrier).run environment).2

def produceNormalized {Program ReadKey Word Value : Type} [DecidableEq ReadKey]
    (analysis : Program → Nat → Nat → Analysis ReadKey Word Value)
    (version : ProgramVersion Program) (pc barrier : Nat)
    (environment : ReadKey → Option Word) : Entry ReadKey Word Value :=
  let result := (analysis version.program pc barrier).run environment
  ⟨⟨version.generation, pc, barrier⟩, footprint result.2 environment, result.1⟩

theorem normalized_producer_covers_reads {Program ReadKey Word Value : Type}
    [DecidableEq ReadKey]
    (analysis : Program → Nat → Nat → Analysis ReadKey Word Value)
    (version : ProgramVersion Program) (pc barrier : Nat)
    (environment : ReadKey → Option Word) :
    FootprintedProduced analysis version environment
      (produceNormalized analysis version pc barrier environment) := by
  refine ⟨rfl, rfl, ?_⟩
  exact footprint_covers _ environment (run_records_exact_reads _ environment)

theorem footprinted_cache_transparent {Program ReadKey Word Value : Type}
    (analysis : Program → Nat → Nat → Analysis ReadKey Word Value)
    {maximum initial final : Nat} {versions : List (ProgramVersion Program)}
    (issued : VersionIssuance Program maximum initial versions final)
    {old current : ProgramVersion Program} (hold : old ∈ versions) (hcurrent : current ∈ versions)
    (before after : ReadKey → Option Word) (entry : Entry ReadKey Word Value)
    (produced : FootprintedProduced analysis old before entry)
    (key : entry.key.generation = current.generation)
    (reads : Agrees entry.reads after) :
    entry.value =
      ((analysis current.program entry.key.pc entry.key.barrier).run after).1 := by
  have hp := equal_generation_same_program issued.versioned hold hcurrent
    (produced.1.symm.trans key)
  have hreads := agrees_of_covers produced.2.2 after reads
  have hreplay := recorded_reads_replay
    (analysis old.program entry.key.pc entry.key.barrier) before after hreads
  simpa only [← hp, hreplay] using produced.2.1

end MemoryWrite.AnalysisCache

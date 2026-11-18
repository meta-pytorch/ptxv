/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AnalysisCacheFootprint

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite.AnalysisCache

@[step]
theorem rust_begin_replay_spec (recorded current : cache.CacheKey) :
    cache.begin_replay recorded current
      ⦃ result => result.matches = decide (keyView recorded = keyView current) ⦄ := by
  unfold cache.begin_replay
  step as ⟨matched, hmatches⟩
  simp [hmatches]

@[step]
theorem rust_observe_spec (replay : cache.Replay) (recorded current : Option U64) :
    cache.observe replay recorded current
      ⦃ result => result.matches = (replay.matches && decide (recorded = current)) ⦄ := by
  unfold cache.observe
  cases hm : replay.matches <;> simp only [Bool.false_eq_true, ↓reduceIte,
    Bool.false_and, Bool.true_and, WP.spec_ok]
  step as ⟨matched, hmatches⟩
  simp [hmatches]

@[step]
theorem rust_finish_replay_spec (replay : cache.Replay) (cached : Bool) :
    cache.finish_replay replay cached
      ⦃ result => result = if replay.matches then some cached else none ⦄ := by
  cases hm : replay.matches <;> simp [cache.finish_replay, hm]

abbrev Observation := Option U64 × Option U64

inductive RustReplayTrace : cache.Replay → List Observation → cache.Replay → Prop
  | nil (replay : cache.Replay) : RustReplayTrace replay [] replay
  | cons {initial next final : cache.Replay} {recorded current : Option U64}
      {rest : List Observation}
      (step : cache.observe initial recorded current = .ok next)
      (tail : RustReplayTrace next rest final) :
      RustReplayTrace initial ((recorded, current) :: rest) final

theorem accepted_replay_preserves_every_observation
    {initial final : cache.Replay} {observations : List Observation}
    (trace : RustReplayTrace initial observations final) (accepted : final.matches = true) :
    initial.matches = true ∧ ∀ observation ∈ observations, observation.1 = observation.2 := by
  induction trace with
  | nil replay => exact ⟨accepted, by simp⟩
  | @cons initial next final recorded current rest step tail ih =>
    have ht := ih accepted
    have hs := rust_observe_spec initial recorded current
    simp only [step, WP.spec_ok, ht.1] at hs
    have hparts : initial.matches = true ∧ recorded = current := by
      have htrue : initial.matches && decide (recorded = current) = true := hs.symm
      simpa only [Bool.and_eq_true, decide_eq_true_eq] using htrue
    refine ⟨hparts.1, ?_⟩
    intro observation hm
    rcases List.mem_cons.mp hm with heq | htail
    · cases heq
      exact hparts.2
    · exact ht.2 observation htail

theorem rust_replay_acceptance_requires_semantic_inputs
    (recorded current : cache.CacheKey) (initial final : cache.Replay)
    (observations : List Observation) (cached answer : Bool)
    (begin : cache.begin_replay recorded current = .ok initial)
    (trace : RustReplayTrace initial observations final)
    (finish : cache.finish_replay final cached = .ok (some answer)) :
    keyView recorded = keyView current ∧
    (∀ observation ∈ observations, observation.1 = observation.2) ∧ answer = cached := by
  have hf := rust_finish_replay_spec final cached
  simp only [finish, WP.spec_ok] at hf
  have accepted : final.matches = true := by
    cases hm : final.matches <;> simp_all
  have ha : answer = cached := by simpa only [accepted, ↓reduceIte, Option.some.injEq] using hf
  have ht := accepted_replay_preserves_every_observation trace accepted
  have hb := rust_begin_replay_spec recorded current
  simp only [begin, WP.spec_ok, ht.1] at hb
  exact ⟨of_decide_eq_true hb.symm, ht.2, ha⟩

theorem rust_accepted_cache_equals_fresh {Program ReadKey : Type}
    (analysis : Program → Nat → Nat → Analysis ReadKey U64 Bool)
    {initialGeneration finalGeneration : U64} {versions : List (RustProgramVersion Program)}
    (issued : RustVersionIssuance Program initialGeneration versions finalGeneration)
    {old current : RustProgramVersion Program}
    (hold : old ∈ versions) (hcurrent : current ∈ versions)
    (pc barrier : U32) (before after : ReadKey → Option U64)
    (entry : Entry ReadKey U64 Bool)
    (produced : FootprintedProduced analysis (versionView old) before entry)
    (recorded query : cache.CacheKey)
    (producerKey : entry.key = keyView recorded)
    (queryKey : cache.make_key current.generation pc barrier = .ok query)
    (initial final : cache.Replay) (answer : Bool)
    (begin : cache.begin_replay recorded query = .ok initial)
    (trace : RustReplayTrace initial
      (entry.reads.map (fun read => (read.2, after read.1))) final)
    (finish : cache.finish_replay final entry.value = .ok (some answer)) :
    answer = ((analysis current.program pc.val barrier.val).run after).1 := by
  have accepted := rust_replay_acceptance_requires_semantic_inputs recorded query initial final
    _ entry.value answer begin trace finish
  have hquery := rust_make_key_spec current.generation pc barrier
  simp only [queryKey, WP.spec_ok] at hquery
  have hkey : entry.key = ⟨current.generation.val, pc.val, barrier.val⟩ :=
    producerKey.trans (accepted.1.trans hquery)
  have hreads : Agrees entry.reads after := by
    intro read hm
    exact (accepted.2.1 (read.2, after read.1) (List.mem_map.mpr ⟨read, hm, rfl⟩)).symm
  have transparent := footprinted_cache_transparent analysis issued.model
    (List.mem_map.mpr ⟨old, hold, rfl⟩) (List.mem_map.mpr ⟨current, hcurrent, rfl⟩)
    before after entry produced (congrArg Key.generation hkey) hreads
  have hpc : entry.key.pc = pc.val := congrArg Key.pc hkey
  have hbarrier : entry.key.barrier = barrier.val := congrArg Key.barrier hkey
  simpa only [hpc, hbarrier, versionView] using accepted.2.2.trans transparent

end MemoryWrite.AnalysisCache

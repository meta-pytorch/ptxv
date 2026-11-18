/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AnalysisCache
import MemoryWrite.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite.AnalysisCache

def keyView (key : cache.CacheKey) : Key :=
  ⟨key.generation.val, key.pc.val, key.barrier.val⟩

@[step]
theorem rust_make_key_spec (generation : U64) (pc barrier : U32) :
    cache.make_key generation pc barrier
      ⦃ result => keyView result = ⟨generation.val, pc.val, barrier.val⟩ ⦄ := by
  simp [cache.make_key, keyView]

@[step]
theorem rust_same_key_spec (left right : cache.CacheKey) :
    cache.same_key left right
      ⦃ result => result = decide (keyView left = keyView right) ⦄ := by
  have hscalar {ty : UScalarTy} (a b : UScalar ty) :
      a = b ↔ a.val = b.val := ⟨congrArg UScalar.val, UScalar.eq_of_val_eq⟩
  simp only [cache.same_key, keyView, Key.mk.injEq, ← hscalar]
  split <;> simp_all
  split <;> simp_all

@[step]
theorem rust_same_observation_spec (recorded current : Option U64) :
    cache.same_observation recorded current
      ⦃ result => result = decide (recorded = current) ⦄ := by
  cases recorded <;> cases current <;> simp [cache.same_observation]

@[step]
theorem rust_next_generation_spec (current : U64) :
    cache.next_generation current
      ⦃ result => result.map UScalar.val = successor U64.max current.val ⦄ := by
  unfold cache.next_generation
  split
  · rename_i h
    simp [h, successor, core.num.U64.MAX, U64.max, U64.rMax, U64.numBits,
      UScalarTy.numBits]
  · rename_i h
    have hb : current.val < U64.max := by
      have hmax : current.val ≤ U64.max := by
        have hm := current.hmax
        simp only [U64.max, U64.numBits, UScalarTy.numBits] at *
        omega
      have hne : current.val ≠ U64.max := by
        intro heq
        apply h
        apply UScalar.eq_of_val_eq
        simpa [core.num.U64.MAX, U64.max, U64.rMax, U64.numBits,
          UScalarTy.numBits] using heq
      omega
    step as ⟨next, hnext⟩
    simp [successor, hb, hnext]

theorem rust_generation_strict (current next : U64)
    (h : cache.next_generation current = .ok (some next)) : current.val < next.val := by
  have hs := rust_next_generation_spec current
  simp only [h, WP.spec_ok, Option.map_some] at hs
  exact successor_strict hs.symm

inductive RustIssuance : U64 → List U64 → U64 → Prop
  | nil (next : U64) : RustIssuance next [] next
  | cons {initial next final : U64} {ids : List U64}
      (step : cache.next_generation initial = .ok (some next))
      (rest : RustIssuance next ids final) : RustIssuance initial (initial :: ids) final

theorem RustIssuance.model {initial final : U64} {ids : List U64}
    (issued : RustIssuance initial ids final) :
    Issuance U64.max initial.val (ids.map UScalar.val) final.val := by
  induction issued with
  | nil next => exact .nil next.val
  | @cons initial next final ids step rest ih =>
    have hs := rust_next_generation_spec initial
    simp only [step, WP.spec_ok, Option.map_some] at hs
    exact .cons hs.symm ih

theorem rust_issued_generations_never_repeat {initial final : U64} {ids : List U64}
    (issued : RustIssuance initial ids final) : ids.Nodup := by
  exact List.Nodup.of_map _ (issuance_no_reuse issued.model)

structure RustProgramVersion (Program : Type) where
  generation : U64
  program : Program

def versionView {Program : Type} (version : RustProgramVersion Program) : ProgramVersion Program :=
  ⟨version.generation.val, version.program⟩

inductive RustVersionIssuance (Program : Type) :
    U64 → List (RustProgramVersion Program) → U64 → Prop
  | nil (next : U64) : RustVersionIssuance Program next [] next
  | cons {initial next final : U64} {versions : List (RustProgramVersion Program)}
      (program : Program)
      (step : cache.next_generation initial = .ok (some next))
      (rest : RustVersionIssuance Program next versions final) :
      RustVersionIssuance Program initial (⟨initial, program⟩ :: versions) final

theorem RustVersionIssuance.model {Program : Type} {initial final : U64}
    {versions : List (RustProgramVersion Program)}
    (issued : RustVersionIssuance Program initial versions final) :
    VersionIssuance Program U64.max initial.val (versions.map versionView) final.val := by
  induction issued with
  | nil next => exact .nil next.val
  | @cons initial next final versions program step rest ih =>
    have hs := rust_next_generation_spec initial
    simp only [step, WP.spec_ok, Option.map_some] at hs
    exact .cons program hs.symm ih

theorem rust_observation_match_requires_exact_read (recorded current : Option U64)
    (h : cache.same_observation recorded current = .ok true) : current = recorded := by
  have hs := rust_same_observation_spec recorded current
  simp only [h, WP.spec_ok] at hs
  exact (of_decide_eq_true hs.symm).symm

theorem rust_key_match_requires_same_generation (left right : cache.CacheKey)
    (h : cache.same_key left right = .ok true) : left.generation = right.generation := by
  have hs := rust_same_key_spec left right
  simp only [h, WP.spec_ok] at hs
  have heq : keyView left = keyView right := of_decide_eq_true hs.symm
  apply UScalar.eq_of_val_eq
  exact congrArg Key.generation heq

theorem rust_cache_producer_transparent {Program ReadKey Value : Type}
    (analysis : Program → Nat → Nat → Analysis ReadKey U64 Value)
    {initial final : U64} {versions : List (RustProgramVersion Program)}
    (issued : RustVersionIssuance Program initial versions final)
    {old current : RustProgramVersion Program}
    (hold : old ∈ versions) (hcurrent : current ∈ versions)
    (pc barrier : U32) (before after : ReadKey → Option U64)
    (oldKey currentKey : cache.CacheKey)
    (holdKey : cache.make_key old.generation pc barrier = .ok oldKey)
    (hcurrentKey : cache.make_key current.generation pc barrier = .ok currentKey)
    (matched : cache.same_key oldKey currentKey = .ok true)
    (observations : ∀ read ∈
      (produce analysis (versionView old) pc.val barrier.val before).reads,
      cache.same_observation read.2 (after read.1) = .ok true) :
    (produce analysis (versionView old) pc.val barrier.val before).value =
      ((analysis current.program pc.val barrier.val).run after).1 := by
  have oldKeySpec := rust_make_key_spec old.generation pc barrier
  have currentKeySpec := rust_make_key_spec current.generation pc barrier
  simp only [holdKey, WP.spec_ok] at oldKeySpec
  simp only [hcurrentKey, WP.spec_ok] at currentKeySpec
  have sameGeneration := rust_key_match_requires_same_generation oldKey currentKey matched
  have oldGeneration := congrArg Key.generation oldKeySpec
  have currentGeneration := congrArg Key.generation currentKeySpec
  have hg : (versionView old).generation = (versionView current).generation := by
    change old.generation.val = current.generation.val
    change oldKey.generation.val = old.generation.val at oldGeneration
    change currentKey.generation.val = current.generation.val at currentGeneration
    rw [← oldGeneration, ← currentGeneration, sameGeneration]
  apply cache_producer_transparent analysis issued.model
    (List.mem_map.mpr ⟨old, hold, rfl⟩) (List.mem_map.mpr ⟨current, hcurrent, rfl⟩)
    pc.val barrier.val before after hg
  intro read hm
  exact rust_observation_match_requires_exact_read read.2 (after read.1) (observations read hm)

theorem rust_failed_read_does_not_match_zero :
    cache.same_observation none (some 0#u64) = .ok false := by rfl

end MemoryWrite.AnalysisCache

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Aeneas

set_option autoImplicit false

namespace MemoryWrite.AnalysisCache

structure Key where
  generation : Nat
  pc : Nat
  barrier : Nat
  deriving DecidableEq

def successor (maximum current : Nat) : Option Nat :=
  if current < maximum then some (current + 1) else none

theorem successor_strict {maximum current next : Nat}
    (h : successor maximum current = some next) : current < next := by
  simp only [successor] at h
  split at h
  · cases h
    omega
  · contradiction

theorem successor_exact {maximum current next : Nat}
    (h : successor maximum current = some next) : next = current + 1 := by
  simp only [successor] at h
  split at h
  · simpa using h.symm
  · contradiction

inductive Issuance (maximum : Nat) : Nat → List Nat → Nat → Prop
  | nil (next : Nat) : Issuance maximum next [] next
  | cons {initial next final : Nat} {ids : List Nat}
      (step : successor maximum initial = some next)
      (rest : Issuance maximum next ids final) :
      Issuance maximum initial (initial :: ids) final

theorem issuance_bounds {maximum initial final : Nat} {ids : List Nat}
    (h : Issuance maximum initial ids final) :
    initial ≤ final ∧ ∀ id ∈ ids, initial ≤ id ∧ id < final := by
  induction h with
  | nil next => simp
  | @cons initial next final ids step rest ih =>
    have hstep := successor_strict step
    constructor
    · omega
    · intro id hm
      rcases List.mem_cons.mp hm with heq | htail
      · subst id
        exact ⟨Nat.le_refl _, by omega⟩
      · have hi := ih.2 id htail
        omega

theorem issuance_no_reuse {maximum initial final : Nat} {ids : List Nat}
    (h : Issuance maximum initial ids final) : ids.Nodup := by
  induction h with
  | nil => simp
  | @cons initial next final ids step rest ih =>
    apply List.nodup_cons.mpr
    constructor
    · intro hm
      have hbound := (issuance_bounds rest).2 initial hm
      have hstep := successor_strict step
      omega
    · exact ih

structure ProgramVersion (Program : Type) where
  generation : Nat
  program : Program

def Versioned {Program : Type} (versions : List (ProgramVersion Program)) : Prop :=
  (versions.map ProgramVersion.generation).Nodup

inductive VersionIssuance (Program : Type) (maximum : Nat) :
    Nat → List (ProgramVersion Program) → Nat → Prop
  | nil (next : Nat) : VersionIssuance Program maximum next [] next
  | cons {initial next final : Nat} {versions : List (ProgramVersion Program)}
      (program : Program)
      (step : successor maximum initial = some next)
      (rest : VersionIssuance Program maximum next versions final) :
      VersionIssuance Program maximum initial (⟨initial, program⟩ :: versions) final

theorem VersionIssuance.issued {Program : Type} {maximum initial final : Nat}
    {versions : List (ProgramVersion Program)}
    (h : VersionIssuance Program maximum initial versions final) :
    Issuance maximum initial (versions.map ProgramVersion.generation) final := by
  induction h with
  | nil next => exact .nil next
  | cons program step rest ih => exact .cons step ih

theorem VersionIssuance.versioned {Program : Type} {maximum initial final : Nat}
    {versions : List (ProgramVersion Program)}
    (h : VersionIssuance Program maximum initial versions final) : Versioned versions :=
  issuance_no_reuse h.issued

theorem equal_generation_same_program {Program : Type}
    {versions : List (ProgramVersion Program)} (h : Versioned versions)
    {left right : ProgramVersion Program}
    (hl : left ∈ versions) (hr : right ∈ versions)
    (heq : left.generation = right.generation) : left.program = right.program := by
  have hab : left = right := List.inj_on_of_nodup_map h hl hr heq
  exact congrArg ProgramVersion.program hab

inductive Analysis (ReadKey Word Value : Type) where
  | done (value : Value)
  | read (key : ReadKey) (next : Option Word → Analysis ReadKey Word Value)

def Analysis.run {ReadKey Word Value : Type}
    (analysis : Analysis ReadKey Word Value) (environment : ReadKey → Option Word) :
    Value × List (ReadKey × Option Word) :=
  match analysis with
  | .done value => (value, [])
  | .read key next =>
    let result := (next (environment key)).run environment
    (result.1, (key, environment key) :: result.2)

def Agrees {ReadKey Word : Type} (reads : List (ReadKey × Option Word))
    (environment : ReadKey → Option Word) : Prop :=
  ∀ read ∈ reads, environment read.1 = read.2

theorem recorded_reads_replay {ReadKey Word Value : Type}
    (analysis : Analysis ReadKey Word Value)
    (before after : ReadKey → Option Word)
    (h : Agrees (analysis.run before).2 after) :
    analysis.run after = analysis.run before := by
  induction analysis with
  | done value => rfl
  | read key next ih =>
    have hhead : after key = before key := h (key, before key) (by simp [Analysis.run])
    have htail : Agrees ((next (before key)).run before).2 after := by
      intro read hm
      apply h read
      simpa only [Analysis.run, List.mem_cons] using Or.inr hm
    simpa only [Analysis.run, hhead] using congrArg
      (fun result : Value × List (ReadKey × Option Word) =>
        (result.1, (key, before key) :: result.2))
      (ih (before key) htail)

structure Entry (ReadKey Word Value : Type) where
  key : Key
  reads : List (ReadKey × Option Word)
  value : Value

def Produced {Program ReadKey Word Value : Type}
    (analysis : Program → Nat → Nat → Analysis ReadKey Word Value)
    (version : ProgramVersion Program) (environment : ReadKey → Option Word)
    (entry : Entry ReadKey Word Value) : Prop :=
  entry.key.generation = version.generation ∧
  (entry.value, entry.reads) =
    (analysis version.program entry.key.pc entry.key.barrier).run environment

def produce {Program ReadKey Word Value : Type}
    (analysis : Program → Nat → Nat → Analysis ReadKey Word Value)
    (version : ProgramVersion Program) (pc barrier : Nat)
    (environment : ReadKey → Option Word) : Entry ReadKey Word Value :=
  let result := (analysis version.program pc barrier).run environment
  ⟨⟨version.generation, pc, barrier⟩, result.2, result.1⟩

theorem produce_has_provenance {Program ReadKey Word Value : Type}
    (analysis : Program → Nat → Nat → Analysis ReadKey Word Value)
    (version : ProgramVersion Program) (pc barrier : Nat)
    (environment : ReadKey → Option Word) :
    Produced analysis version environment (produce analysis version pc barrier environment) := by
  exact ⟨rfl, by simp only [produce, Prod.mk.eta]⟩

theorem cached_analysis_equals_fresh {Program ReadKey Word Value : Type}
    (analysis : Program → Nat → Nat → Analysis ReadKey Word Value)
    {versions : List (ProgramVersion Program)} (hversions : Versioned versions)
    {old current : ProgramVersion Program} (hold : old ∈ versions) (hcurrent : current ∈ versions)
    (before after : ReadKey → Option Word) (entry : Entry ReadKey Word Value)
    (produced : Produced analysis old before entry)
    (key : entry.key = ⟨current.generation, entry.key.pc, entry.key.barrier⟩)
    (reads : Agrees entry.reads after) :
    entry.value =
      ((analysis current.program entry.key.pc entry.key.barrier).run after).1 := by
  have hg : old.generation = current.generation := by
    have hkey := congrArg Key.generation key
    exact produced.1.symm.trans hkey
  have hp := equal_generation_same_program hversions hold hcurrent hg
  have htrace := congrArg Prod.snd produced.2
  have hreads : Agrees
      ((analysis old.program entry.key.pc entry.key.barrier).run before).2 after := by
    simpa only [← htrace] using reads
  have hreplay := recorded_reads_replay
    (analysis old.program entry.key.pc entry.key.barrier) before after hreads
  have hvalue := congrArg Prod.fst produced.2
  simpa only [← hp, hreplay] using hvalue

theorem known_zero_does_not_match_failed_read {Word : Type} (zero : Word) :
    (none : Option Word) ≠ some zero := by simp

theorem issued_cache_equals_fresh {Program ReadKey Word Value : Type}
    (analysis : Program → Nat → Nat → Analysis ReadKey Word Value)
    {maximum initial final : Nat} {versions : List (ProgramVersion Program)}
    (issued : VersionIssuance Program maximum initial versions final)
    {old current : ProgramVersion Program} (hold : old ∈ versions) (hcurrent : current ∈ versions)
    (before after : ReadKey → Option Word) (entry : Entry ReadKey Word Value)
    (produced : Produced analysis old before entry)
    (key : entry.key = ⟨current.generation, entry.key.pc, entry.key.barrier⟩)
    (reads : Agrees entry.reads after) :
    entry.value =
      ((analysis current.program entry.key.pc entry.key.barrier).run after).1 :=
  cached_analysis_equals_fresh analysis issued.versioned hold hcurrent before after entry
    produced key reads

theorem cache_producer_transparent {Program ReadKey Word Value : Type}
    (analysis : Program → Nat → Nat → Analysis ReadKey Word Value)
    {maximum initial final : Nat} {versions : List (ProgramVersion Program)}
    (issued : VersionIssuance Program maximum initial versions final)
    {old current : ProgramVersion Program} (hold : old ∈ versions) (hcurrent : current ∈ versions)
    (pc barrier : Nat) (before after : ReadKey → Option Word)
    (key : old.generation = current.generation)
    (reads : Agrees (produce analysis old pc barrier before).reads after) :
    (produce analysis old pc barrier before).value =
      ((analysis current.program pc barrier).run after).1 := by
  apply issued_cache_equals_fresh analysis issued hold hcurrent before after
    (produce analysis old pc barrier before) (produce_has_provenance analysis old pc barrier before)
  · simp only [produce, key]
  · exact reads

end MemoryWrite.AnalysisCache

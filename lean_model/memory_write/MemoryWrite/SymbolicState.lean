/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Producers

open Aeneas Aeneas.Std PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite.Symbolic

inductive RegisterFile where
  | virtual | lane | uniform | predicate | uniformPredicate
  deriving DecidableEq

structure RegisterKey where
  thread : Nat
  file : RegisterFile
  index : Nat
  component : Nat
  deriving DecidableEq

structure Value where
  bits : U32
  defined : Bool
  deriving DecidableEq

structure Metadata where
  captured : U32
  history : Option U32
  tainted : Bool
  defined : Bool
  deriving DecidableEq

abbrev Meaning (Input : Type) := U32 → Input → U32

structure Sound {Input : Type} (sem : Semantics Input) (meaning : Meaning Input)
    (metadata : Metadata) (values : Input → Value) : Prop where
  captured : metadata.captured = (values sem.capture).bits
  defined : metadata.defined = true → ∀ input, sem.domain input → (values input).defined = true
  constant : metadata.tainted = false → ∀ input, sem.domain input →
    (values input).bits = (values sem.capture).bits
  history : ∀ id, metadata.history = some id → ∀ input, sem.domain input →
    (values input).defined = true ∧ (values input).bits = meaning id input

structure State (Input : Type) where
  metadata : RegisterKey → Metadata
  values : RegisterKey → Input → Value
  meaning : Meaning Input

def State.Sound {Input : Type} (sem : Semantics Input) (state : State Input) : Prop :=
  ∀ key, MemoryWrite.Symbolic.Sound sem state.meaning (state.metadata key) (state.values key)

structure Prepared (Input : Type) where
  metadata : Metadata
  values : Input → Value

def prepare {Input : Type} (state : State Input) (source : RegisterKey) : Prepared Input :=
  ⟨state.metadata source, state.values source⟩

def guardedValues {Input : Type} (guard : Input → Bool) (source old : Input → Value) : Input → Value :=
  fun input => if guard input then source input else old input

def assignMetadata (enabled guardDependent : Bool) (source old : Metadata) : Metadata :=
  let selected := if enabled then source else old
  if guardDependent then
    { selected with history := none, tainted := true, defined := false }
  else selected

def commit {Input : Type} (sem : Semantics Input) (state : State Input) (destination : RegisterKey)
    (source : Prepared Input) (guard : Input → Bool) (guardDependent : Bool) : State Input :=
  { state with
    metadata := fun key => if key = destination then
      assignMetadata (guard sem.capture) guardDependent source.metadata (state.metadata key)
      else state.metadata key
    values := fun key => if key = destination then
      guardedValues guard source.values (state.values key) else state.values key }

structure GuardSound {Input : Type} (sem : Semantics Input) (guard : Input → Bool)
    (dependent : Bool) : Prop where
  stable : dependent = false → ∀ input, sem.domain input → guard input = guard sem.capture

def predicateGuard {Input : Type} (values : Input → Value) : Input → Bool :=
  fun input => decide ((values input).bits ≠ 0#u32)

theorem derives_guard_sound {Input : Type} (sem : Semantics Input) (meaning : Meaning Input)
    (metadata : Metadata) (values : Input → Value) (hs : Sound sem meaning metadata values) :
    GuardSound sem (predicateGuard values) metadata.tainted := by
  constructor
  intro ht input hi
  simp [predicateGuard, hs.constant ht input hi]

def addressFromWords {Input : Type} (space : U8) (allocation : U64)
    (low high : Input → Value) : Input → Key :=
  fun input => ⟨space, allocation, ⟨(high input).bits.bv ++ (low input).bits.bv⟩⟩

theorem derives_stable_store_execution {Input : Type} (sem : Semantics Input)
    (meaning : Meaning Input) (guard low high : Metadata)
    (guardValues lowValues highValues : Input → Value)
    (hg : Sound sem meaning guard guardValues)
    (hl : Sound sem meaning low lowValues) (hh : Sound sem meaning high highValues)
    (hguard : guard.tainted = false) (hlow : low.tainted = false) (hhigh : high.tainted = false)
    (space : U8) (allocation : U64) :
    Producers.StableStoreExecution sem (predicateGuard guardValues)
      (addressFromWords space allocation lowValues highValues) := by
  constructor
  · exact (derives_guard_sound sem meaning guard guardValues hg).stable hguard
  · intro _ input hi
    simp [addressFromWords, hl.constant hlow input hi, hh.constant hhigh input hi]

theorem prepare_sound {Input : Type} (sem : Semantics Input) (state : State Input)
    (hstate : state.Sound sem) (source : RegisterKey) :
    Sound sem state.meaning (prepare state source).metadata (prepare state source).values :=
  hstate source

theorem assign_sound {Input : Type} (sem : Semantics Input) (meaning : Meaning Input)
    (source old : Metadata) (sourceValues oldValues : Input → Value)
    (hsource : Sound sem meaning source sourceValues) (hold : Sound sem meaning old oldValues)
    (guard : Input → Bool) (dependent : Bool) (hguard : GuardSound sem guard dependent) :
    Sound sem meaning (assignMetadata (guard sem.capture) dependent source old)
      (guardedValues guard sourceValues oldValues) := by
  cases dependent
  · have hstable := hguard.stable rfl
    cases hcaptured : guard sem.capture
    · constructor
      · simpa [assignMetadata, guardedValues, hcaptured] using hold.captured
      · intro hd input hi
        simpa [guardedValues, hstable input hi, hcaptured] using
          hold.defined (by simpa [assignMetadata, hcaptured] using hd) input hi
      · intro ht input hi
        simpa [guardedValues, hstable input hi, hcaptured] using
          hold.constant (by simpa [assignMetadata, hcaptured] using ht) input hi
      · intro id hid input hi
        simpa [guardedValues, hstable input hi, hcaptured] using
          hold.history id (by simpa [assignMetadata, hcaptured] using hid) input hi
    · constructor
      · simpa [assignMetadata, guardedValues, hcaptured] using hsource.captured
      · intro hd input hi
        simpa [guardedValues, hstable input hi, hcaptured] using
          hsource.defined (by simpa [assignMetadata, hcaptured] using hd) input hi
      · intro ht input hi
        simpa [guardedValues, hstable input hi, hcaptured] using
          hsource.constant (by simpa [assignMetadata, hcaptured] using ht) input hi
      · intro id hid input hi
        simpa [guardedValues, hstable input hi, hcaptured] using
          hsource.history id (by simpa [assignMetadata, hcaptured] using hid) input hi
  · constructor
    · cases hc : guard sem.capture <;>
        simp [assignMetadata, guardedValues, hc, hsource.captured, hold.captured]
    · simp [assignMetadata]
    · simp [assignMetadata]
    · simp [assignMetadata]

theorem prepared_commit_preserves_soundness {Input : Type} (sem : Semantics Input)
    (state : State Input) (hstate : state.Sound sem) (destination : RegisterKey)
    (source : Prepared Input) (hsource : Sound sem state.meaning source.metadata source.values)
    (guard : Input → Bool) (dependent : Bool) (hguard : GuardSound sem guard dependent) :
    (commit sem state destination source guard dependent).Sound sem := by
  intro key
  by_cases hk : key = destination
  · simp only [commit, hk, ite_true]
    exact assign_sound sem state.meaning source.metadata (state.metadata destination)
      source.values (state.values destination) hsource (hstate destination) guard dependent hguard
  · simpa [commit, hk] using hstate key

theorem copy_preserves_soundness {Input : Type} (sem : Semantics Input) (state : State Input)
    (hstate : state.Sound sem) (source destination : RegisterKey)
    (guard : Input → Bool) (dependent : Bool) (hguard : GuardSound sem guard dependent) :
    (commit sem state destination (prepare state source) guard dependent).Sound sem :=
  prepared_commit_preserves_soundness sem state hstate destination _
    (prepare_sound sem state hstate source) guard dependent hguard

theorem prepared_operand_survives_clobber {Input : Type} (state : State Input)
    (source : RegisterKey) :
    (prepare state source).metadata = state.metadata source ∧
      (prepare state source).values = state.values source := by
  exact ⟨rfl, rfl⟩

def extendMeaning {Input : Type} (meaning : Meaning Input) (fresh : U32)
    (values : Input → U32) : Meaning Input :=
  fun id => if id = fresh then values else meaning id

theorem extendMeaning_preserves_old {Input : Type} (meaning : Meaning Input)
    (fresh id : U32) (values : Input → U32) (hne : id ≠ fresh) :
    extendMeaning meaning fresh values id = meaning id := by
  simp [extendMeaning, hne]

theorem extension_preserves_soundness {Input : Type} (sem : Semantics Input)
    (state : State Input) (hstate : state.Sound sem) (fresh : U32)
    (hnew : ∀ key, (state.metadata key).history ≠ some fresh) (values : Input → U32) :
    ({state with meaning := extendMeaning state.meaning fresh values}).Sound sem := by
  intro key
  have hs := hstate key
  refine ⟨hs.captured, hs.defined, hs.constant, ?_⟩
  intro id hid input hi
  have hne : id ≠ fresh := by intro h; subst id; exact hnew key hid
  simpa [extendMeaning, hne] using hs.history id hid input hi

inductive MissingHistory where
  | constant | opaque
  deriving DecidableEq

def classifyMissing (inputDerived : Bool) : MissingHistory :=
  if inputDerived then .opaque else .constant

theorem captured_zero_does_not_justify_constant (inputDerived : Bool)
    (hderived : inputDerived = true) : classifyMissing inputDerived = .opaque := by
  simp [classifyMissing, hderived]

def fallback (captured : U32) (inputDerived defined : Bool) : Metadata :=
  ⟨captured, none, inputDerived, defined⟩

theorem fallback_sound {Input : Type} (sem : Semantics Input) (meaning : Meaning Input)
    (values : Input → Value) (inputDerived defined : Bool)
    (hconstant : inputDerived = false → ∀ input, sem.domain input →
      (values input).bits = (values sem.capture).bits)
    (hdefined : defined = true → ∀ input, sem.domain input → (values input).defined = true) :
    Sound sem meaning (fallback (values sem.capture).bits inputDerived defined) values := by
  exact ⟨rfl, hdefined, hconstant, by simp [fallback]⟩

def byteAt (word : U32) (lane : Fin 4) : U8 :=
  ⟨word.bv.extractLsb' (8 * lane.val) 8⟩

def registerWord (metadata : Metadata) : Producers.RegisterWord :=
  { bytes := fun lane =>
      ⟨byteAt metadata.captured lane, none, metadata.tainted, metadata.defined⟩
    history := metadata.history.map WordHistory.Known }

def byteValues {Input : Type} (values : Input → Value) : Input → Fin 4 → Byte :=
  fun input lane => ⟨byteAt (values input).bits lane, (values input).defined⟩

theorem derives_register_word_sound {Input : Type} (sem : Semantics Input)
    (meaning : Meaning Input) (metadata : Metadata) (values : Input → Value)
    (hs : Sound sem meaning metadata values)
    (hmeaning : ∀ id input lane, sem.word id input lane = byteAt (meaning id input) lane) :
    Producers.RegisterWordSound sem (registerWord metadata) (byteValues values) := by
  constructor
  · intro lane
    constructor
    · exact congrArg (fun bits => byteAt bits lane) hs.captured
    · exact hs.defined
    · simp [registerWord]
    · intro ht input hi
      exact congrArg (fun bits => byteAt bits lane) (hs.constant ht input hi)
  · intro id hid input hi lane
    have hn : metadata.history = some id := by
      simpa [registerWord] using hid
    obtain ⟨hd, hv⟩ := hs.history id hn input hi
    exact ⟨hd, by simp only [byteValues, hv, hmeaning]⟩

theorem derives_payload_sound {Input : Type} (sem : Semantics Input)
    (state : State Input) (hstate : state.Sound sem) (keys : Nat → RegisterKey)
    (registers : List Nat) (hsize : 4 * registers.length ≤ Usize.max)
    (hmeaning : ∀ id input lane, sem.word id input lane = byteAt (state.meaning id input) lane) :
    PayloadSound sem
      (Producers.registerPayload (fun index => registerWord (state.metadata (keys index))) registers hsize)
      (Producers.registerValues (fun index => byteValues (state.values (keys index))) registers) := by
  apply Producers.registerPayload_sound
  intro index _
  exact derives_register_word_sound sem state.meaning _ _ (hstate (keys index)) hmeaning

inductive BinaryOp where
  | add | bitAnd | bitOr | bitXor
  deriving DecidableEq

def binaryBits (op : BinaryOp) (left right : U32) : U32 :=
  match op with
  | .add => ⟨left.bv + right.bv⟩
  | .bitAnd => ⟨left.bv &&& right.bv⟩
  | .bitOr => ⟨left.bv ||| right.bv⟩
  | .bitXor => ⟨left.bv ^^^ right.bv⟩

def binaryMetadata (op : BinaryOp) (left right : Metadata) : Metadata :=
  ⟨binaryBits op left.captured right.captured, none,
    left.tainted || right.tainted, left.defined && right.defined⟩

def binaryValues {Input : Type} (op : BinaryOp) (left right : Input → Value) : Input → Value :=
  fun input => ⟨binaryBits op (left input).bits (right input).bits,
    (left input).defined && (right input).defined⟩

theorem binary_preserves_soundness {Input : Type} (sem : Semantics Input)
    (meaning : Meaning Input) (op : BinaryOp) (left right : Metadata)
    (leftValues rightValues : Input → Value)
    (hl : Sound sem meaning left leftValues) (hr : Sound sem meaning right rightValues) :
    Sound sem meaning (binaryMetadata op left right) (binaryValues op leftValues rightValues) := by
  constructor
  · simp [binaryMetadata, binaryValues, hl.captured, hr.captured]
  · intro hd input hi
    have hboth : left.defined = true ∧ right.defined = true := by
      simpa [binaryMetadata] using hd
    simp [binaryValues, hl.defined hboth.1 input hi, hr.defined hboth.2 input hi]
  · intro ht input hi
    have hboth : left.tainted = false ∧ right.tainted = false := by
      simpa [binaryMetadata] using ht
    simp [binaryValues, hl.constant hboth.1 input hi, hr.constant hboth.2 input hi]
  · simp [binaryMetadata]

def witnessSem : Semantics Bool :=
  { domain := fun _ => True
    capture := false
    capture_valid := trivial
    origin := fun _ _ => 0#u8
    zero_origin := by intros; rfl
    word := fun _ _ _ => 0#u8
    initial := fun _ _ => ⟨0#u8, true⟩ }

def varyingZero : Bool → Value := fun input => ⟨if input then 1#u32 else 0#u32, true⟩

theorem captured_zero_constant_is_unsound :
    ¬Sound witnessSem (fun _ _ => 0#u32) (fallback 0#u32 false true) varyingZero := by
  intro hs
  have bad := hs.constant rfl true trivial
  change (1#u32 : U32) = 0#u32 at bad
  exact (by decide : (1#u32 : U32) ≠ 0#u32) bad

theorem captured_zero_opaque_is_sound :
    Sound witnessSem (fun _ _ => 0#u32) (fallback 0#u32 true true) varyingZero := by
  exact ⟨rfl, by simp [varyingZero], by simp [fallback], by simp [fallback]⟩

theorem captured_false_guard_forgets_history (source old : Metadata) :
    (assignMetadata false true source old).history = none ∧
    (assignMetadata false true source old).tainted = true ∧
    (assignMetadata false true source old).defined = false := by
  simp [assignMetadata]

theorem predicate_namespaces_distinct (thread index component : Nat) :
    (⟨thread, .predicate, index, component⟩ : RegisterKey) ≠
      ⟨thread, .uniformPredicate, index, component⟩ := by
  simp [RegisterKey.mk.injEq]

theorem leaf_256_distinct_from_zero : (256#u32 : U32) ≠ 0#u32 := by decide

end MemoryWrite.Symbolic

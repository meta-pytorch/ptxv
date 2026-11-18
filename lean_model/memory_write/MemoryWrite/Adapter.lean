/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Production

open Aeneas Aeneas.Std PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.Adapter

def assocLookup {KeyType Value : Type} [DecidableEq KeyType]
    (entries : List (KeyType × Value)) (key : KeyType) : Option Value :=
  match entries with
  | [] => none
  | (address, value) :: rest =>
      if address = key then some value else assocLookup rest key

def rawWordKey (key : Key) : Key :=
  if key.space = 3#u8 then
    { key with offset := U64.ofNat (key.offset.val / 4) (by scalar_tac) }
  else key

@[simp] theorem rawWordKey_tmem (allocation offset : U64) :
    (rawWordKey ⟨3#u8, allocation, offset⟩).offset.val = offset.val / 4 := by
  simp [rawWordKey]

@[simp] theorem rawWordKey_other (key : Key) (hspace : key.space ≠ 3#u8) :
    rawWordKey key = key := by simp [rawWordKey, hspace]

def wordByte (word lane : Nat) : U8 :=
  U8.ofNat ((word / 2 ^ (8 * lane)) % 256) (by
    have := Nat.mod_lt (word / 2 ^ (8 * lane)) (by decide : 0 < 256)
    scalar_tac)

structure RawPlanes where
  inventory : Slice Key
  bytes : List (Key × U8) := []
  undefined : List (Key × Bool) := []
  written : List Key := []
  halfOrigins : List (Key × Origin) := []
  halfWritten : List Key := []
  numeric : List (Key × WordHistory) := []
  numericWritten : List Key := []
  taint : List (Key × Bool) := []
  allGlobal : Bool := false
  cleanGlobal : List Key := []
  halfInputKinds : List (U64 × HalfKind) := []
  f32Inputs : List U64 := []
  initialBytes : List (Key × U8) := []
  /-- IDs supplied by the numeric node producer; this adapter does not allocate graph nodes. -/
  initialWordNodes : List (Key × U32) := []
  localPackedWords : List (Key × U64) := []
  tmemWords : List (Key × U32) := []
  tmemInit : List (Key × U8) := []

def RawPlanes.tmemState (raw : RawPlanes) (key : Key) : U8 :=
  (assocLookup raw.tmemInit (rawWordKey key)).getD 2#u8

def RawPlanes.effectiveByte (raw : RawPlanes) (key : Key) : U8 :=
  if key.space = 3#u8 then
    wordByte ((assocLookup raw.tmemWords (rawWordKey key)).getD 0#u32).val (key.offset.val % 4)
  else if key.space = 4#u8 then
    match assocLookup raw.localPackedWords (alignedWordKey key) with
    | some packed => wordByte packed.val (key.offset.val % 4)
    | none => (assocLookup raw.bytes key).getD 0#u8
  else (assocLookup raw.bytes key).getD 0#u8

def RawPlanes.effectiveDefined (raw : RawPlanes) (key : Key) : Bool :=
  if key.space = 3#u8 then raw.tmemState key == 1#u8
  else if key.space = 4#u8 then
    match assocLookup raw.localPackedWords (alignedWordKey key) with
    | some packed => wordByte packed.val (4 + key.offset.val % 4) == 0#u8
    | none => false
  else if key.space = 0#u8 then
    decide (key.offset.val < 1024) ||
      (decide (key ∈ raw.written) && !(assocLookup raw.undefined key).getD false)
  else !(assocLookup raw.undefined key).getD false

def RawPlanes.effectiveTaint (raw : RawPlanes) (key : Key) : Bool :=
  (assocLookup raw.taint (rawWordKey key)).getD false ||
    (raw.allGlobal && key.space == 1#u8 && !decide (key ∈ raw.cleanGlobal))

def RawPlanes.metadataVisible (raw : RawPlanes) (key : Key) : Bool :=
  key.space != 3#u8 || raw.tmemState key == 1#u8

def inputByteOrigin (kind : HalfKind) (key : Key) : Origin :=
  match kind with
  | .byte => .E key.allocation key.offset 0#u8
  | .f16 | .bf16 =>
      .E key.allocation (U64.ofNat (key.offset.val / 2) (by scalar_tac))
        (U8.ofNat (key.offset.val % 2) (by
          have := Nat.mod_lt key.offset.val (by decide : 0 < 2)
          scalar_tac))

def RawPlanes.HalfFallbackEligible (raw : RawPlanes) (key : Key) : Prop :=
  key.space = 1#u8 ∧ key ∉ raw.halfWritten ∧
    assocLookup raw.initialBytes key = some (raw.effectiveByte key)

instance (raw : RawPlanes) (key : Key) : Decidable (raw.HalfFallbackEligible key) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def RawPlanes.fallbackOrigin (raw : RawPlanes) (key : Key) : Option Origin :=
  match assocLookup raw.halfInputKinds key.allocation with
  | none => none
  | some kind =>
      if raw.HalfFallbackEligible key then some (inputByteOrigin kind key) else none

def RawPlanes.effectiveOrigin (raw : RawPlanes) (key : Key) : Option Origin :=
  if raw.metadataVisible key then
    match assocLookup raw.halfOrigins key with
    | some origin => some origin
    | none => raw.fallbackOrigin key
  else none

def RawPlanes.WordMatchesInitial (raw : RawPlanes) (key : Key) : Prop :=
  ∀ lane : Fin 4,
    assocLookup raw.initialBytes (Storage.offsetKey key lane.val) =
      some (raw.effectiveByte (Storage.offsetKey key lane.val))

instance (raw : RawPlanes) (key : Key) : Decidable (raw.WordMatchesInitial key) :=
  inferInstanceAs (Decidable (∀ _lane : Fin 4, _))

def RawPlanes.WordFallbackEligible (raw : RawPlanes) (key : Key) : Prop :=
  key.space = 1#u8 ∧ key.offset.val % 4 = 0 ∧ key.allocation ∈ raw.f32Inputs ∧
    key ∉ raw.numericWritten ∧ raw.WordMatchesInitial key

instance (raw : RawPlanes) (key : Key) : Decidable (raw.WordFallbackEligible key) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _))

def RawPlanes.fallbackWord (raw : RawPlanes) (key : Key) : Option WordHistory :=
  if raw.WordFallbackEligible key then
    (assocLookup raw.initialWordNodes key).map WordHistory.Known
  else none

def RawPlanes.wordVisible (raw : RawPlanes) (key : Key) : Bool :=
  raw.metadataVisible key && (key.space != 3#u8 || decide (key.offset.val % 4 = 0))

def RawPlanes.effectiveWord (raw : RawPlanes) (key : Key) : Option WordHistory :=
  if raw.wordVisible key then
    match assocLookup raw.numeric (rawWordKey key) with
    | some history => some history
    | none => raw.fallbackWord key
  else none

def RawPlanes.effectiveWritten (raw : RawPlanes) (key : Key) : Bool :=
  if key.space = 0#u8 ∨ key.space = 1#u8 then decide (key ∈ raw.written) else true

def RawPlanes.effectiveCell (raw : RawPlanes) (key : Key) : Cell :=
  ⟨key, raw.effectiveByte key, raw.effectiveOrigin key, raw.effectiveTaint key,
    raw.effectiveDefined key, raw.effectiveWritten key, raw.effectiveWord key⟩

def RawPlanes.liveKeys (raw : RawPlanes) : List Key :=
  raw.inventory.val.filter (fun key => key.space != 3#u8 || raw.tmemState key != 2#u8)

def RawPlanes.toStorage (raw : RawPlanes) : Storage :=
  { keys := Slice.from raw.liveKeys (by
      have := raw.inventory.property
      have := List.length_filter_le
        (fun key => key.space != 3#u8 || raw.tmemState key != 2#u8) raw.inventory.val
      dsimp only [RawPlanes.liveKeys]
      omega)
    bytes := raw.inventory.val.map fun key => (key, raw.effectiveByte key)
    origins := raw.liveKeys.map fun key => (key, raw.effectiveOrigin key)
    taint := raw.liveKeys.map fun key => (key, raw.effectiveTaint key)
    defined := raw.liveKeys.map fun key => (key, raw.effectiveDefined key)
    written := raw.liveKeys.map fun key => (key, raw.effectiveWritten key)
    words := raw.liveKeys.map fun key => (key, raw.effectiveWord key) }

@[simp] theorem RawPlanes.toStorage_keys (raw : RawPlanes) :
    raw.toStorage.keys.val = raw.liveKeys := by simp only [RawPlanes.toStorage, Slice.from_val]

theorem RawPlanes.liveKeys_subset (raw : RawPlanes) (key : Key) (hkey : key ∈ raw.liveKeys) :
    key ∈ raw.inventory.val := (List.mem_filter.mp hkey).1

theorem RawPlanes.getCell_toStorage (raw : RawPlanes) (key : Key) (hkey : key ∈ raw.liveKeys) :
    raw.toStorage.getCell key = raw.effectiveCell key := by
  have hraw := raw.liveKeys_subset key hkey
  simp only [Storage.getCell, RawPlanes.toStorage,
    Storage.lookup_tabulate _ _ _ _ hkey, Storage.lookup_tabulate _ _ _ _ hraw,
    RawPlanes.effectiveCell]

theorem RawPlanes.toCells_toStorage (raw : RawPlanes) :
    raw.toStorage.toCells.val = raw.liveKeys.map raw.effectiveCell := by
  simp only [Storage.toCells, Slice.from_val, RawPlanes.toStorage_keys]
  apply List.map_congr_left
  intro key hkey
  exact raw.getCell_toStorage key hkey

@[simp] theorem RawPlanes.backing_byte (raw : RawPlanes) (key : Key)
    (hkey : key ∈ raw.inventory.val) :
    Storage.lookup raw.toStorage.bytes key 0#u8 = raw.effectiveByte key := by
  exact Storage.lookup_tabulate raw.inventory.val raw.effectiveByte key 0#u8 hkey


theorem fallbackOrigin_input (raw : RawPlanes) (key : Key) (kind : HalfKind)
    (hkind : assocLookup raw.halfInputKinds key.allocation = some kind)
    (heligible : raw.HalfFallbackEligible key) :
    raw.fallbackOrigin key = some (inputByteOrigin kind key) := by
  simp [RawPlanes.fallbackOrigin, hkind, heligible]

theorem effectiveOrigin_input (raw : RawPlanes) (key : Key) (kind : HalfKind)
    (hkind : assocLookup raw.halfInputKinds key.allocation = some kind)
    (heligible : raw.HalfFallbackEligible key)
    (hstored : assocLookup raw.halfOrigins key = none) :
    raw.effectiveOrigin key = some (inputByteOrigin kind key) := by
  have hvisible : raw.metadataVisible key = true := by
    simp [RawPlanes.metadataVisible, heligible.1]
  simp [RawPlanes.effectiveOrigin, hvisible, hstored,
    fallbackOrigin_input raw key kind hkind heligible]

theorem fallbackOrigin_written (raw : RawPlanes) (key : Key) (hwritten : key ∈ raw.halfWritten) :
    raw.fallbackOrigin key = none := by
  cases hkind : assocLookup raw.halfInputKinds key.allocation <;>
    simp [RawPlanes.fallbackOrigin, hkind, RawPlanes.HalfFallbackEligible, hwritten]

theorem fallbackOrigin_mismatch (raw : RawPlanes) (key : Key)
    (hmismatch : assocLookup raw.initialBytes key ≠ some (raw.effectiveByte key)) :
    raw.fallbackOrigin key = none := by
  cases hkind : assocLookup raw.halfInputKinds key.allocation <;>
    simp [RawPlanes.fallbackOrigin, hkind, RawPlanes.HalfFallbackEligible, hmismatch]

theorem fallbackWord_input (raw : RawPlanes) (key : Key) (node : U32)
    (heligible : raw.WordFallbackEligible key)
    (hnode : assocLookup raw.initialWordNodes key = some node) :
    raw.fallbackWord key = some (.Known node) := by
  simp [RawPlanes.fallbackWord, heligible, hnode]

theorem effectiveWord_input (raw : RawPlanes) (key : Key) (node : U32)
    (heligible : raw.WordFallbackEligible key)
    (hnode : assocLookup raw.initialWordNodes key = some node)
    (hstored : assocLookup raw.numeric (rawWordKey key) = none) :
    raw.effectiveWord key = some (.Known node) := by
  have hvisible : raw.wordVisible key = true := by
    simp [RawPlanes.wordVisible, RawPlanes.metadataVisible, heligible.1]
  simp [RawPlanes.effectiveWord, hvisible, hstored, fallbackWord_input raw key node heligible hnode]

theorem fallbackWord_written (raw : RawPlanes) (key : Key) (hwritten : key ∈ raw.numericWritten) :
    raw.fallbackWord key = none := by
  simp [RawPlanes.fallbackWord, RawPlanes.WordFallbackEligible, hwritten]

theorem tmem_nonstart_no_word (raw : RawPlanes) (key : Key) (hspace : key.space = 3#u8)
    (hbyte : key.offset.val % 4 ≠ 0) : raw.effectiveWord key = none := by
  simp [RawPlanes.effectiveWord, RawPlanes.wordVisible, hspace, hbyte]

theorem tmem_undefined_no_origin (raw : RawPlanes) (key : Key) (hspace : key.space = 3#u8)
    (hstate : raw.tmemState key = 0#u8) : raw.effectiveOrigin key = none := by
  simp [RawPlanes.effectiveOrigin, RawPlanes.metadataVisible, hspace, hstate]

theorem tmem_undefined_no_word (raw : RawPlanes) (key : Key) (hspace : key.space = 3#u8)
    (hstate : raw.tmemState key = 0#u8) : raw.effectiveWord key = none := by
  simp [RawPlanes.effectiveWord, RawPlanes.wordVisible, RawPlanes.metadataVisible, hspace, hstate]

theorem tmem_unallocated_not_live (raw : RawPlanes) (key : Key) (hspace : key.space = 3#u8)
    (hstate : raw.tmemState key = 2#u8) : key ∉ raw.liveKeys := by
  simp [RawPlanes.liveKeys, hspace, hstate]

def OriginSound {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (key : Key) (origin : Origin) : Prop :=
  ∀ input, sem.domain input →
    (memory input (location key)).defined = true ∧
    (memory input (location key)).bits = sem.origin origin input

structure RawRepresents {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (raw : RawPlanes) : Prop where
  captured : ∀ key ∈ raw.liveKeys,
    raw.effectiveByte key = (memory sem.capture (location key)).bits
  defined : ∀ key ∈ raw.liveKeys, raw.effectiveDefined key = true →
    ∀ input, sem.domain input → (memory input (location key)).defined = true
  independent : ∀ key ∈ raw.liveKeys, raw.effectiveTaint key = false →
    ∀ input, sem.domain input →
      (memory input (location key)).bits = (memory sem.capture (location key)).bits
  storedOrigin : ∀ key ∈ raw.liveKeys, raw.metadataVisible key = true → ∀ origin,
    assocLookup raw.halfOrigins key = some origin → OriginSound sem memory key origin
  initialOrigin : ∀ key ∈ raw.liveKeys, ∀ kind,
    assocLookup raw.halfInputKinds key.allocation = some kind →
      raw.HalfFallbackEligible key → OriginSound sem memory key (inputByteOrigin kind key)
  storedWord : ∀ key ∈ raw.liveKeys, raw.wordVisible key = true → ∀ id,
    assocLookup raw.numeric (rawWordKey key) = some (.Known id) →
      WordSound sem memory key (some (.Known id))
  initialWord : ∀ key ∈ raw.liveKeys, ∀ id, raw.WordFallbackEligible key →
    assocLookup raw.initialWordNodes key = some id →
      WordSound sem memory key (some (.Known id))
  initial : ∀ key ∈ raw.liveKeys, raw.effectiveWritten key = false →
    ∀ input, sem.domain input → memory input (location key) = sem.initial input (location key)

theorem effectiveOrigin_sound {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (raw : RawPlanes) (hraw : RawRepresents sem memory raw) (key : Key)
    (hkey : key ∈ raw.liveKeys) (origin : Origin)
    (horigin : raw.effectiveOrigin key = some origin) : OriginSound sem memory key origin := by
  unfold RawPlanes.effectiveOrigin at horigin
  split at horigin
  · rename_i hvisible
    cases hstored : assocLookup raw.halfOrigins key with
    | some stored =>
      simp only [hstored, Option.some.injEq] at horigin
      subst origin
      exact hraw.storedOrigin key hkey hvisible stored hstored
    | none =>
      simp only [hstored] at horigin
      unfold RawPlanes.fallbackOrigin at horigin
      cases hkind : assocLookup raw.halfInputKinds key.allocation with
      | none => simp [hkind] at horigin
      | some kind =>
        simp only [hkind] at horigin
        split at horigin
        · rename_i heligible
          simp only [Option.some.injEq] at horigin
          subst origin
          exact hraw.initialOrigin key hkey kind hkind heligible
        · simp at horigin
  · simp at horigin

theorem effectiveWord_sound {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (raw : RawPlanes) (hraw : RawRepresents sem memory raw) (key : Key)
    (hkey : key ∈ raw.liveKeys) : WordSound sem memory key (raw.effectiveWord key) := by
  intro id hid
  unfold RawPlanes.effectiveWord at hid
  split at hid
  · rename_i hvisible
    cases hstored : assocLookup raw.numeric (rawWordKey key) with
    | some history =>
      simp only [hstored, Option.some.injEq] at hid
      subst history
      exact hraw.storedWord key hkey hvisible id hstored id rfl
    | none =>
      simp only [hstored] at hid
      unfold RawPlanes.fallbackWord at hid
      split at hid
      · rename_i heligible
        cases hnode : assocLookup raw.initialWordNodes key with
        | none => simp [hnode] at hid
        | some node =>
          simp only [hnode, Option.map_some, Option.some.injEq, WordHistory.Known.injEq] at hid
          subst node
          exact hraw.initialWord key hkey id heligible hnode id rfl
      · simp at hid
  · simp at hid

theorem effectiveCell_represents {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (raw : RawPlanes) (hraw : RawRepresents sem memory raw) (key : Key)
    (hkey : key ∈ raw.liveKeys) : CellRepresents sem memory (raw.effectiveCell key) := by
  constructor
  · constructor
    · exact hraw.captured key hkey
    · exact hraw.defined key hkey
    · intro origin horigin
      exact effectiveOrigin_sound sem memory raw hraw key hkey origin horigin
    · exact hraw.independent key hkey
  · exact effectiveWord_sound sem memory raw hraw key hkey
  · exact hraw.initial key hkey

theorem toStorage_represents {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (raw : RawPlanes) (hraw : RawRepresents sem memory raw) :
    Storage.Represents sem memory raw.toStorage := by
  intro cell hcell
  rw [raw.toCells_toStorage] at hcell
  obtain ⟨key, hkey, rfl⟩ := List.mem_map.mp hcell
  exact effectiveCell_represents sem memory raw hraw key hkey

end MemoryWrite.Adapter

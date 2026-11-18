/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Snapshot

open Aeneas Aeneas.Std PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite

namespace Storage

def lookup {α : Type} (entries : List (Key × α)) (key : Key) (fallback : α) : α :=
  match entries with
  | [] => fallback
  | (address, value) :: rest => if address = key then value else lookup rest key fallback

theorem lookup_tabulate {α : Type} (keys : List Key) (values : Key → α)
    (key : Key) (fallback : α) (hkey : key ∈ keys) :
    lookup (keys.map (fun address => (address, values address))) key fallback = values key := by
  induction keys with
  | nil => simp at hkey
  | cons address rest ih =>
    simp only [List.map_cons, lookup]
    by_cases heq : address = key
    · simp [heq]
    · rw [ite_eq_right heq]
      exact ih (by simpa [Ne.symm heq] using hkey)

theorem lookup_tabulate_missing {α : Type} (keys : List Key) (values : Key → α)
    (key : Key) (fallback : α) (hkey : key ∉ keys) :
    lookup (keys.map (fun address => (address, values address))) key fallback = fallback := by
  induction keys with
  | nil => rfl
  | cons address rest ih =>
    simp only [List.mem_cons, not_or] at hkey
    simp only [List.map_cons, lookup, ite_eq_right (Ne.symm hkey.1)]
    exact ih hkey.2

theorem lookup_tabulate_append {α : Type} (keys : List Key) (values : Key → α)
    (rest : List (Key × α)) (key : Key) (fallback : α) (hkey : key ∈ keys) :
    lookup ((keys.map (fun address => (address, values address))) ++ rest) key fallback = values key := by
  induction keys with
  | nil => simp at hkey
  | cons address keys ih =>
    simp only [List.map_cons, List.cons_append, lookup]
    by_cases heq : address = key
    · simp [heq]
    · rw [ite_eq_right heq]
      exact ih (by simpa [Ne.symm heq] using hkey)

theorem lookup_tabulate_append_missing {α : Type} (keys : List Key) (values : Key → α)
    (rest : List (Key × α)) (key : Key) (fallback : α) (hkey : key ∉ keys) :
    lookup ((keys.map (fun address => (address, values address))) ++ rest) key fallback =
      lookup rest key fallback := by
  induction keys with
  | nil => rfl
  | cons address keys ih =>
    simp only [List.mem_cons, not_or] at hkey
    simp only [List.map_cons, List.cons_append, lookup, ite_eq_right (Ne.symm hkey.1)]
    exact ih hkey.2

end Storage

structure Storage where
  keys : Slice Key
  bytes : List (Key × U8)
  origins : List (Key × Option Origin)
  taint : List (Key × Bool)
  defined : List (Key × Bool)
  written : List (Key × Bool)
  words : List (Key × Option WordHistory)

namespace Storage

def getCell (state : Storage) (key : Key) : Cell :=
  ⟨key,
    lookup state.bytes key 0#u8,
    lookup state.origins key none,
    lookup state.taint key false,
    lookup state.defined key false,
    lookup state.written key false,
    lookup state.words key none⟩

@[simp]
theorem getCell_key (state : Storage) (key : Key) : (getCell state key).key = key := rfl

def toCells (state : Storage) : Slice Cell :=
  Slice.from (state.keys.val.map (getCell state)) (by simp)

@[simp]
theorem toCells_keys (state : Storage) : (toCells state).val.map Cell.key = state.keys.val := by
  simp [toCells, List.map_map, Function.comp_def]

theorem getCell_mem (state : Storage) (key : Key) (hkey : key ∈ state.keys.val) :
    getCell state key ∈ (toCells state).val := by
  simpa only [toCells, Slice.from_val] using
    (List.mem_map.mpr ⟨key, hkey, rfl⟩ :
      getCell state key ∈ state.keys.val.map (getCell state))

def Represents {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (state : Storage) : Prop :=
  MemoryWrite.Represents sem memory (toCells state)

def write (state : Storage) (address : Key) (payload : Slice PayloadByte) : Storage :=
  let changed := fun key => Touches address payload.val.length key
  let byte := fun key => payloadAt payload (key.offset.val - address.offset.val)
  { keys := state.keys
    bytes := (state.keys.val.map fun key =>
      (key, if changed key then (byte key).value else (getCell state key).value)) ++
      state.bytes.filter (fun entry => decide (entry.1 ∉ state.keys.val))
    origins := state.keys.val.map fun key =>
      (key, if changed key then (byte key).origin else (getCell state key).origin)
    taint := state.keys.val.map fun key =>
      (key, if changed key then (byte key).tainted else (getCell state key).tainted)
    defined := state.keys.val.map fun key =>
      (key, if changed key then (byte key).defined else (getCell state key).defined)
    written := state.keys.val.map fun key =>
      (key, if changed key then true else (getCell state key).written)
    words := state.keys.val.map fun key => (key, wordAfter (getCell state key) address payload) }

@[simp]
theorem write_keys (state : Storage) (address : Key) (payload : Slice PayloadByte) :
    (write state address payload).keys = state.keys := rfl

theorem getCell_write (state : Storage) (address key : Key) (payload : Slice PayloadByte)
    (hkey : key ∈ state.keys.val) :
    getCell (write state address payload) key = modelCell (getCell state key) address payload := by
  simp only [getCell, write, lookup_tabulate _ _ _ _ hkey,
    lookup_tabulate_append _ _ _ _ _ hkey, modelCell]
  split <;> simp_all

theorem toCells_write (state : Storage) (address : Key) (payload : Slice PayloadByte) :
    (toCells (write state address payload)).val =
      (toCells state).val.map (fun cell => modelCell cell address payload) := by
  simp only [toCells, Slice.from_val, write_keys, List.map_map]
  apply List.map_congr_left
  intro key hkey
  exact getCell_write state address key payload hkey

def offsetKey (start : Key) (index : Nat) : Key :=
  ⟨start.space, start.allocation,
    U64.ofNatCore ((start.offset.val + index) % (U64.max + 1))
      (by have := Nat.mod_lt (start.offset.val + index) (by scalar_tac : 0 < U64.max + 1); scalar_tac)⟩

theorem offsetKey_location (start : Key) (index : Nat)
    (hindex : start.offset.val + index ≤ U64.max) :
    location (offsetKey start index) = location start index := by
  simp [location, offsetKey, Nat.mod_eq_of_lt (by omega : start.offset.val + index < U64.max + 1)]

def spanKeys (start : Key) (width : Usize) : List Key :=
  List.ofFn (fun index : Fin width.val => offsetKey start index.val)

@[simp]
theorem spanKeys_length (start : Key) (width : Usize) :
    (spanKeys start width).length = width.val := by simp [spanKeys]

def SpanFits (state : Storage) (start : Key) (width : Usize) : Prop :=
  start.offset.val + width.val ≤ U64.max + 1 ∧
    ∀ key ∈ spanKeys start width, key ∈ state.keys.val

instance (state : Storage) (start : Key) (width : Usize) : Decidable (SpanFits state start width) :=
  inferInstanceAs (Decidable (_ ∧ _))

def readSpanUnchecked (state : Storage) (start : Key) (width : Usize) : Slice Cell :=
  Slice.from ((spanKeys start width).map (getCell state)) (by simp; scalar_tac)

def readSpan (state : Storage) (start : Key) (width : Usize) : Option (Slice Cell) :=
  if SpanFits state start width then some (readSpanUnchecked state start width) else none

@[simp]
theorem readSpanUnchecked_length (state : Storage) (start : Key) (width : Usize) :
    (readSpanUnchecked state start width).val.length = width.val := by
  simp [readSpanUnchecked]

theorem spanFits_coverage (state : Storage) (start : Key) (width : Usize)
    (hspan : SpanFits state start width) :
    ValidWriteSpan (toCells state) start width.val := by
  refine ⟨hspan.1, ?_⟩
  intro index hindex
  let key := offsetKey start index
  have hmem : key ∈ spanKeys start width := by
    exact List.mem_ofFn.mpr ⟨⟨index, hindex⟩, rfl⟩
  refine ⟨getCell state key, getCell_mem state key (hspan.2 key hmem), ?_⟩
  have hwrap := hspan.1
  exact offsetKey_location start index (by omega)

theorem readSpanUnchecked_contiguous (state : Storage) (start : Key) (width : Usize)
    (hspan : SpanFits state start width) :
    Contiguous (readSpanUnchecked state start width) start := by
  intro index hindex
  have hi : index < width.val := by simpa using hindex
  simp only [readSpanUnchecked, Slice.from_val, List.getElem_map, getCell_key,
    spanKeys, List.getElem_ofFn]
  have hwrap := hspan.1
  exact offsetKey_location start index (by omega)

theorem readSpanUnchecked_represents {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (state : Storage) (start : Key) (width : Usize)
    (hstate : Represents sem memory state) (hspan : SpanFits state start width) :
    MemoryWrite.Represents sem memory (readSpanUnchecked state start width) := by
  intro cell hcell
  simp only [readSpanUnchecked, Slice.from_val] at hcell
  obtain ⟨key, hkey, rfl⟩ := List.mem_map.mp hcell
  exact hstate (getCell state key) (getCell_mem state key (hspan.2 key hkey))

theorem readSpan_sound {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (state : Storage) (start : Key) (width : Usize) (source : Slice Cell)
    (hstate : Represents sem memory state) (hread : readSpan state start width = some source) :
    source.val.length = width.val ∧ Contiguous source start ∧
      MemoryWrite.Represents sem memory source := by
  unfold readSpan at hread
  split at hread
  · rename_i hspan
    have heq : source = readSpanUnchecked state start width := by simpa using hread.symm
    subst source
    exact ⟨readSpanUnchecked_length state start width,
      readSpanUnchecked_contiguous state start width hspan,
      readSpanUnchecked_represents sem memory state start width hstate hspan⟩
  · simp at hread

theorem readSpan_none_of_missing (state : Storage) (start key : Key) (width : Usize)
    (hspan : key ∈ spanKeys start width) (hmissing : key ∉ state.keys.val) :
    readSpan state start width = none := by
  have hfit : ¬SpanFits state start width := fun h => hmissing (h.2 key hspan)
  simp [readSpan, hfit]

theorem write_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (state : Storage) (address : Key) (payload : Slice PayloadByte)
    (values : Input → Nat → Byte) (hstate : Represents sem memory state)
    (hspan : SpanFits state address (Slice.len payload))
    (hpayload : PayloadSound sem payload values) :
    Represents sem (storeBytes memory address payload.val.length values)
      (write state address payload) ∧
    WriteSpec (toCells state) (toCells (write state address payload)) address payload ∧
    SpanFits (write state address payload) address (Slice.len payload) := by
  obtain ⟨next, hrun, hspec, hrep, _, _⟩ :=
    apply_write_effect_preserves_representation sem memory (toCells state) address payload values
      hstate (by simpa using spanFits_coverage state address (Slice.len payload) hspan) hpayload
  have hmodel := (spec_returns _ _ (apply_write_effect_spec (toCells state) address payload))
  obtain ⟨model, hmodelRun, hmodel⟩ := hmodel
  have heq : next = model := by simpa [hrun] using hmodelRun
  subst model
  have hstorage : toCells (write state address payload) = next := by
    apply Slice.ext
    rw [toCells_write, hmodel]
  exact ⟨by simpa [Represents, hstorage] using hrep,
    by simpa [hstorage] using hspec, hspan⟩

def empty : Storage :=
  ⟨Slice.from [] (by simp), [], [], [], [], [], []⟩

def fromCellsUnchecked (cells : Slice Cell) : Storage :=
  { keys := Slice.from (cells.val.map Cell.key) (by simp)
    bytes := cells.val.map fun cell => (cell.key, cell.value)
    origins := cells.val.map fun cell => (cell.key, cell.origin)
    taint := cells.val.map fun cell => (cell.key, cell.tainted)
    defined := cells.val.map fun cell => (cell.key, cell.defined)
    written := cells.val.map fun cell => (cell.key, cell.written)
    words := cells.val.map fun cell => (cell.key, cell.word) }

def fromCells (cells : Slice Cell) : Option Storage :=
  if (cells.val.map Cell.key).Nodup then some (fromCellsUnchecked cells) else none

theorem lookup_cells {α : Type} (cells : List Cell) (field : Cell → α) (cell : Cell)
    (fallback : α) (hcell : cell ∈ cells) (hunique : (cells.map Cell.key).Nodup) :
    lookup (cells.map (fun value => (value.key, field value))) cell.key fallback = field cell := by
  induction cells with
  | nil => simp at hcell
  | cons head rest ih =>
    have huniq := List.nodup_cons.mp hunique
    rcases List.mem_cons.mp hcell with heq | hrest
    · subst cell
      simp [lookup]
    · have hne : head.key ≠ cell.key := by
        intro heq
        apply huniq.1
        exact List.mem_map.mpr ⟨cell, hrest, heq.symm⟩
      simp only [List.map_cons, lookup, ite_eq_right hne]
      exact ih hrest huniq.2

theorem getCell_fromCellsUnchecked (cells : Slice Cell) (cell : Cell)
    (hcell : cell ∈ cells.val) (hunique : (cells.val.map Cell.key).Nodup) :
    getCell (fromCellsUnchecked cells) cell.key = cell := by
  simp only [getCell, fromCellsUnchecked, lookup_cells _ _ _ _ hcell hunique]

theorem toCells_fromCellsUnchecked (cells : Slice Cell)
    (hunique : (cells.val.map Cell.key).Nodup) :
    toCells (fromCellsUnchecked cells) = cells := by
  apply Slice.ext
  simp only [toCells, fromCellsUnchecked, Slice.from_val, List.map_map]
  have hmap : cells.val.map (fun cell => getCell (fromCellsUnchecked cells) cell.key) =
      cells.val.map id := by
    apply List.map_congr_left
    intro cell hcell
    exact getCell_fromCellsUnchecked cells cell hcell hunique
  simpa only [List.map_id, fromCellsUnchecked, Function.comp_def] using hmap

theorem fromCells_sound (cells : Slice Cell) (state : Storage)
    (hfrom : fromCells cells = some state) : toCells state = cells := by
  unfold fromCells at hfrom
  split at hfrom
  · rename_i hunique
    have heq : state = fromCellsUnchecked cells := by simpa using hfrom.symm
    rw [heq]
    exact toCells_fromCellsUnchecked cells hunique
  · simp at hfrom

def allocation (start : Key) (data : Slice U8) (tainted defined : Bool) : Storage :=
  let keys := spanKeys start (Slice.len data)
  { keys := Slice.from keys (by simp [keys])
    bytes := keys.map fun key => (key, data.val[key.offset.val - start.offset.val]?.getD 0#u8)
    origins := []
    taint := keys.map fun key => (key, tainted)
    defined := keys.map fun key => (key, defined)
    written := []
    words := [] }

def ofBytes (start : Key) (data : Slice U8) (tainted defined : Bool) : Option Storage :=
  if start.offset.val + data.val.length ≤ U64.max + 1
    then some (allocation start data tainted defined) else none

theorem allocation_spanFits (start : Key) (data : Slice U8) (tainted defined : Bool)
    (hwrap : start.offset.val + data.val.length ≤ U64.max + 1) :
    SpanFits (allocation start data tainted defined) start (Slice.len data) := by
  refine ⟨by simpa using hwrap, ?_⟩
  intro key hkey
  simpa only [allocation, Slice.from_val] using hkey

theorem ofBytes_spanFits (start : Key) (data : Slice U8) (tainted defined : Bool)
    (state : Storage) (hfrom : ofBytes start data tainted defined = some state) :
    SpanFits state start (Slice.len data) := by
  unfold ofBytes at hfrom
  split at hfrom
  · rename_i hwrap
    have heq : state = allocation start data tainted defined := by simpa using hfrom.symm
    rw [heq]
    exact allocation_spanFits start data tainted defined hwrap
  · simp at hfrom

theorem allocation_getCell (start : Key) (data : Slice U8) (tainted defined : Bool)
    (index : Nat) (hindex : index < data.val.length)
    (hwrap : start.offset.val + data.val.length ≤ U64.max + 1) :
    getCell (allocation start data tainted defined) (offsetKey start index) =
      ⟨offsetKey start index, data.val[index], none, tainted, defined, false, none⟩ := by
  have hkey : offsetKey start index ∈ spanKeys start (Slice.len data) := by
    exact List.mem_ofFn.mpr ⟨⟨index, by simpa using hindex⟩, rfl⟩
  have hoffset : (offsetKey start index).offset.val - start.offset.val = index := by
    have hloc := offsetKey_location start index (by omega)
    have hpos := congrArg Address.offset hloc
    simp only [location, Nat.add_zero] at hpos
    omega
  simp only [getCell, allocation, lookup_tabulate _ _ _ _ hkey, lookup]
  simp [hoffset, List.getElem?_eq_getElem hindex]

theorem ofBytes_getCell (start : Key) (data : Slice U8) (tainted defined : Bool)
    (state : Storage) (index : Nat) (hindex : index < data.val.length)
    (hfrom : ofBytes start data tainted defined = some state) :
    getCell state (offsetKey start index) =
      ⟨offsetKey start index, data.val[index], none, tainted, defined, false, none⟩ := by
  unfold ofBytes at hfrom
  split at hfrom
  · rename_i hwrap
    have heq : state = allocation start data tainted defined := by simpa using hfrom.symm
    rw [heq]
    exact allocation_getCell start data tainted defined index hindex hwrap
  · simp at hfrom

theorem lookup_filter {α : Type} (entries : List (Key × α)) (keep : Key → Bool)
    (key : Key) (fallback : α) (hkeep : keep key = true) :
    lookup (entries.filter (fun entry => keep entry.1)) key fallback =
      lookup entries key fallback := by
  induction entries with
  | nil => rfl
  | cons entry entries ih =>
    obtain ⟨address, value⟩ := entry
    by_cases heq : address = key
    · subst address
      simp [lookup, hkeep]
    · by_cases haddress : keep address = true <;> simp [lookup, heq, haddress, ih]

theorem lookup_filter_rejected {α : Type} (entries : List (Key × α)) (keep : Key → Bool)
    (key : Key) (fallback : α) (hkeep : keep key = false) :
    lookup (entries.filter (fun entry => keep entry.1)) key fallback = fallback := by
  induction entries with
  | nil => rfl
  | cons entry entries ih =>
    obtain ⟨address, value⟩ := entry
    by_cases heq : address = key
    · subst address
      simp [hkeep, ih]
    · by_cases haddress : keep address = true <;> simp [lookup, heq, haddress, ih]

theorem write_backing_value (state : Storage) (start key : Key) (payload : Slice PayloadByte)
    (hmissing : key ∉ state.keys.val) :
    (getCell (write state start payload) key).value = (getCell state key).value := by
  simp only [getCell, write, lookup_tabulate_append_missing _ _ _ _ _ hmissing]
  exact lookup_filter state.bytes (fun address => decide (address ∉ state.keys.val)) key 0#u8
    (by simp [hmissing])

def restrict (state : Storage) (keep : Key → Bool) : Storage :=
  { keys := Slice.from (state.keys.val.filter keep) (by
      have := state.keys.property
      have := List.length_filter_le keep state.keys.val
      omega)
    bytes := state.bytes.filter fun entry => keep entry.1
    origins := state.origins.filter fun entry => keep entry.1
    taint := state.taint.filter fun entry => keep entry.1
    defined := state.defined.filter fun entry => keep entry.1
    written := state.written.filter fun entry => keep entry.1
    words := state.words.filter fun entry => keep entry.1 }

theorem getCell_restrict (state : Storage) (keep : Key → Bool) (key : Key)
    (hkeep : keep key = true) : getCell (restrict state keep) key = getCell state key := by
  simp [getCell, restrict, lookup_filter _ _ _ _ hkeep]

theorem restrict_represents {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (state : Storage) (keep : Key → Bool) (hstate : Represents sem memory state) :
    Represents sem memory (restrict state keep) := by
  intro cell hcell
  simp only [toCells, Slice.from_val] at hcell
  obtain ⟨key, hkey, rfl⟩ := List.mem_map.mp hcell
  simp only [restrict, Slice.from_val, List.mem_filter] at hkey
  rw [getCell_restrict state keep key hkey.2]
  exact hstate (getCell state key) (getCell_mem state key hkey.1)

def forgetWord (cell : Cell) (start : Key) (width : Usize) : Cell :=
  { cell with word := if Overlaps start width.val cell.key then none else cell.word }

theorem forgetWord_represents {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (cell : Cell) (start : Key) (width : Usize) (hcell : CellRepresents sem memory cell) :
    CellRepresents sem memory (forgetWord cell start width) := by
  constructor
  · exact hcell.byte
  · intro id hid input hinput lane
    by_cases hoverlap : Overlaps start width.val cell.key
    · simp [forgetWord, hoverlap] at hid
    · exact hcell.word id (by simpa [forgetWord, hoverlap] using hid) input hinput lane
  · exact hcell.initial_fallback

def removeSpan (state : Storage) (start : Key) (width : Usize) : Storage :=
  let retained := restrict state (fun key => decide (¬Touches start width.val key))
  { retained with words := retained.keys.val.map fun key =>
      (key, (forgetWord (getCell state key) start width).word) }

theorem removeSpan_keys (state : Storage) (start : Key) (width : Usize) :
    (removeSpan state start width).keys.val =
      state.keys.val.filter (fun key => decide (¬Touches start width.val key)) := by
  simp [removeSpan, restrict]

theorem getCell_removeSpan (state : Storage) (start key : Key) (width : Usize)
    (hkey : key ∈ (removeSpan state start width).keys.val) :
    getCell (removeSpan state start width) key = forgetWord (getCell state key) start width := by
  have hkeep : (decide (¬Touches start width.val key)) = true := by
    rw [removeSpan_keys] at hkey
    exact (List.mem_filter.mp hkey).2
  have hretained : key ∈ (restrict state (fun key => decide (¬Touches start width.val key))).keys.val := by
    simpa [removeSpan] using hkey
  simp only [getCell, removeSpan, lookup_tabulate _ _ _ _ hretained]
  simp only [restrict, lookup_filter _ (fun key => decide (¬Touches start width.val key)) key _ hkeep]
  rfl

theorem removeSpan_represents {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (state : Storage) (start : Key) (width : Usize) (hstate : Represents sem memory state) :
    Represents sem memory (removeSpan state start width) := by
  intro cell hcell
  simp only [toCells, Slice.from_val] at hcell
  obtain ⟨key, hkey, rfl⟩ := List.mem_map.mp hcell
  rw [getCell_removeSpan state start key width hkey]
  have hbefore : key ∈ state.keys.val := by
    rw [removeSpan_keys] at hkey
    exact (List.mem_filter.mp hkey).1
  exact forgetWord_represents sem memory (getCell state key) start width
    (hstate (getCell state key) (getCell_mem state key hbefore))

theorem removeSpan_no_touched (state : Storage) (start key : Key) (width : Usize)
    (hkey : key ∈ (removeSpan state start width).keys.val) : ¬Touches start width.val key := by
  rw [removeSpan_keys] at hkey
  simpa using (List.mem_filter.mp hkey).2

theorem getCell_removeSpan_touched (state : Storage) (start key : Key) (width : Usize)
    (htouched : Touches start width.val key) :
    getCell (removeSpan state start width) key =
      ⟨key, 0#u8, none, false, false, false, none⟩ := by
  have hkeep : decide (¬Touches start width.val key) = false := by simp [htouched]
  have hmissing : key ∉ (restrict state (fun key => decide (¬Touches start width.val key))).keys.val := by
    simp [restrict, htouched]
  simp only [getCell, removeSpan, lookup_tabulate_missing _ _ _ _ hmissing]
  simp only [restrict,
    lookup_filter_rejected _ (fun key => decide (¬Touches start width.val key)) key _ hkeep]

def releaseSpan (state : Storage) (start : Key) (width : Usize) : Storage :=
  { removeSpan state start width with bytes := state.bytes }

@[simp]
theorem releaseSpan_keys (state : Storage) (start : Key) (width : Usize) :
    (releaseSpan state start width).keys = (removeSpan state start width).keys := rfl

@[simp]
theorem releaseSpan_backing (state : Storage) (start : Key) (width : Usize) :
    (releaseSpan state start width).bytes = state.bytes := rfl

theorem getCell_releaseSpan (state : Storage) (start key : Key) (width : Usize)
    (hkey : key ∈ (releaseSpan state start width).keys.val) :
    getCell (releaseSpan state start width) key = getCell (removeSpan state start width) key := by
  change { getCell (removeSpan state start width) key with value := lookup state.bytes key 0#u8 } = _
  rw [getCell_removeSpan state start key width hkey]
  rfl

theorem releaseSpan_represents {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (state : Storage) (start : Key) (width : Usize) (hstate : Represents sem memory state) :
    Represents sem memory (releaseSpan state start width) := by
  intro cell hcell
  simp only [toCells, Slice.from_val] at hcell
  obtain ⟨key, hkey, rfl⟩ := List.mem_map.mp hcell
  rw [getCell_releaseSpan state start key width hkey]
  exact removeSpan_represents sem memory state start width hstate _
    (getCell_mem (removeSpan state start width) key hkey)

def removeAllocation (state : Storage) (allocation : Key) : Storage :=
  restrict state (fun key => decide (¬SameRegion allocation key))

theorem removeAllocation_represents {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (state : Storage) (allocation : Key)
    (hstate : Represents sem memory state) :
    Represents sem memory (removeAllocation state allocation) :=
  restrict_represents sem memory state _ hstate

theorem touches_overlaps (start key : Key) (width : Nat) (h : Touches start width key) :
    Overlaps start width key := by
  obtain ⟨hregion, hlo, hhi⟩ := h
  exact ⟨hregion, by omega, hhi, by omega⟩

theorem modelCell_represents_touched {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (cell : Cell) (start : Key) (payload : Slice PayloadByte)
    (values : Input → Nat → Byte) (hpayload : PayloadSound sem payload values)
    (htouched : Touches start payload.val.length cell.key) :
    CellRepresents sem (storeBytes memory start payload.val.length values)
      (modelCell cell start payload) := by
  have hkey := modelCell_key cell start payload
  constructor
  · have hi : cell.key.offset.val - start.offset.val < payload.val.length := by
      obtain ⟨_, hlo, hhi⟩ := htouched
      omega
    have hin := (inWrite_location start cell.key payload.val.length).mpr htouched
    simp only [modelCell, ite_eq_left htouched, storeBytes, ite_eq_left hin]
    simpa only [location, Nat.add_zero] using hpayload.bytes _ hi
  · intro id hid input hinput lane
    rw [hkey]
    rw [(modelCell_spec cell start payload).word_effect] at hid
    unfold wordAfter at hid
    split at hid
    · rename_i hfull
      have hin := covered_lane start cell.key payload.val.length hfull lane
      have hindex : cell.key.offset.val - start.offset.val + 4 ≤ payload.val.length := by
        obtain ⟨_, hlo, hhi⟩ := hfull
        omega
      have hoff : cell.key.offset.val + lane.val - start.offset.val =
          (cell.key.offset.val - start.offset.val) + lane.val := by
        obtain ⟨_, hlo, _⟩ := hfull
        omega
      simp only [storeBytes, ite_eq_left hin]
      change (values input (cell.key.offset.val + lane.val - start.offset.val)).defined = true ∧
        (values input (cell.key.offset.val + lane.val - start.offset.val)).bits = sem.word id input lane
      rw [hoff]
      exact hpayload.words _ hindex id hid input hinput lane
    · split at hid
      · contradiction
      · rename_i hdisjoint
        exact False.elim (hdisjoint (touches_overlaps start cell.key payload.val.length htouched))
  · intro hnot
    simp [modelCell, htouched] at hnot

def extensionKeys (state : Storage) (start : Key) (width : Usize) : List Key :=
  state.keys.val ++ (spanKeys start width).filter (fun key => decide (key ∉ state.keys.val))

theorem mem_extensionKeys (state : Storage) (start key : Key) (width : Usize) :
    key ∈ extensionKeys state start width ↔ key ∈ state.keys.val ∨ key ∈ spanKeys start width := by
  by_cases hkey : key ∈ state.keys.val <;> simp [extensionKeys, hkey]

def extend (state : Storage) (start : Key) (width : Usize)
    (hsize : (extensionKeys state start width).length ≤ Usize.max) : Storage :=
  { state with keys := Slice.from (extensionKeys state start width) hsize }

@[simp]
theorem getCell_extend (state : Storage) (start key : Key) (width : Usize)
    (hsize : (extensionKeys state start width).length ≤ Usize.max) :
    getCell (extend state start width hsize) key = getCell state key := rfl

theorem extend_spanFits (state : Storage) (start : Key) (width : Usize)
    (hsize : (extensionKeys state start width).length ≤ Usize.max)
    (hwrap : start.offset.val + width.val ≤ U64.max + 1) :
    SpanFits (extend state start width hsize) start width := by
  refine ⟨hwrap, ?_⟩
  intro key hkey
  simp only [extend, Slice.from_val]
  exact (mem_extensionKeys state start key width).mpr (Or.inr hkey)

def registerWrite (state : Storage) (start : Key) (payload : Slice PayloadByte) : Option Storage :=
  if start.offset.val + payload.val.length ≤ U64.max + 1 then
    if hsize : (extensionKeys state start (Slice.len payload)).length ≤ Usize.max then
      some (write (extend state start (Slice.len payload) hsize) start payload)
    else none
  else none

theorem registerWrite_keys (state next : Storage) (start key : Key) (payload : Slice PayloadByte)
    (hrun : registerWrite state start payload = some next) :
    key ∈ next.keys.val ↔ key ∈ state.keys.val ∨ key ∈ spanKeys start (Slice.len payload) := by
  unfold registerWrite at hrun
  split at hrun
  · split at hrun
    · have heq : next = write (extend state start (Slice.len payload) ‹_›) start payload := by
        simpa using hrun.symm
      rw [heq]
      simpa only [write_keys, extend, Slice.from_val] using
        mem_extensionKeys state start key (Slice.len payload)
    · simp at hrun
  · simp at hrun

theorem getCell_registerWrite (state next : Storage) (start key : Key) (payload : Slice PayloadByte)
    (hrun : registerWrite state start payload = some next) (hkey : key ∈ next.keys.val) :
    getCell next key = modelCell (getCell state key) start payload := by
  unfold registerWrite at hrun
  split at hrun
  · split at hrun
    · rename_i hsize
      have heq : next = write (extend state start (Slice.len payload) hsize) start payload := by
        simpa using hrun.symm
      subst next
      exact getCell_write (extend state start (Slice.len payload) hsize) start key payload hkey
    · simp at hrun
  · simp at hrun

theorem spanKeys_touches (start key : Key) (width : Usize)
    (hwrap : start.offset.val + width.val ≤ U64.max + 1)
    (hkey : key ∈ spanKeys start width) : Touches start width.val key := by
  obtain ⟨index, rfl⟩ := List.mem_ofFn.mp hkey
  have hloc := offsetKey_location start index.val (by omega)
  have hoff := congrArg Address.offset hloc
  simp only [location, Nat.add_zero] at hoff
  exact ⟨⟨rfl, rfl⟩, by omega, by omega⟩

theorem registerWrite_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (state next : Storage) (start : Key) (payload : Slice PayloadByte)
    (values : Input → Nat → Byte) (hstate : Represents sem memory state)
    (hpayload : PayloadSound sem payload values)
    (hrun : registerWrite state start payload = some next) :
    Represents sem (storeBytes memory start payload.val.length values) next ∧
      SpanFits next start (Slice.len payload) := by
  unfold registerWrite at hrun
  split at hrun
  · rename_i hwrap
    split at hrun
    · rename_i hsize
      have heq : next = write (extend state start (Slice.len payload) hsize) start payload := by
        simpa using hrun.symm
      subst next
      have hfit : SpanFits (extend state start (Slice.len payload) hsize) start (Slice.len payload) :=
        extend_spanFits state start (Slice.len payload) hsize (by simpa using hwrap)
      refine ⟨?_, hfit⟩
      intro cell hcell
      simp only [toCells, Slice.from_val] at hcell
      obtain ⟨key, hkey, rfl⟩ := List.mem_map.mp hcell
      have hextended : key ∈ (extend state start (Slice.len payload) hsize).keys.val := hkey
      rw [getCell_write (extend state start (Slice.len payload) hsize) start key payload hextended,
        getCell_extend]
      have hmem : key ∈ state.keys.val ∨ key ∈ spanKeys start (Slice.len payload) := by
        apply (mem_extensionKeys state start key (Slice.len payload)).mp
        simpa [write, extend] using hkey
      rcases hmem with hbefore | hnew
      · exact modelCell_represents sem memory (getCell state key) start payload values
          (hstate (getCell state key) (getCell_mem state key hbefore)) hpayload
      · apply modelCell_represents_touched sem memory (getCell state key) start payload values hpayload
        simpa using spanKeys_touches start key (Slice.len payload) (by simpa using hwrap) hnew
    · simp at hrun
  · simp at hrun

structure PagedBytes where
  length : Usize
  pages : List (List U8)

namespace PagedBytes

def pageSize : Nat := 512

def get (memory : PagedBytes) (index : Nat) : Option U8 := do
  if index < memory.length.val then
    let page ← memory.pages[index / pageSize]?
    if page.length = 1 then page[0]? else page[index % pageSize]?
  else none

def toBytes (memory : PagedBytes) : Slice U8 :=
  Slice.from (List.ofFn fun index : Fin memory.length.val =>
    (memory.get index.val).getD 0#u8) (by simp; scalar_tac)

def Valid (memory : PagedBytes) : Prop :=
  ∀ index : Fin memory.length.val, (memory.get index.val).isSome = true

instance (memory : PagedBytes) : Decidable (Valid memory) :=
  inferInstanceAs (Decidable (∀ _index : Fin memory.length.val, _))

def checkedBytes (memory : PagedBytes) : Option (Slice U8) :=
  if Valid memory then some memory.toBytes else none

theorem checkedBytes_sound (memory : PagedBytes) (data : Slice U8)
    (hread : memory.checkedBytes = some data) :
    data.val.length = memory.length.val ∧
      ∀ index (hindex : index < data.val.length), memory.get index = some data.val[index] := by
  unfold checkedBytes at hread
  split at hread
  · rename_i hvalid
    have heq : data = memory.toBytes := by simpa using hread.symm
    subst data
    refine ⟨by simp [toBytes], ?_⟩
    intro index hindex
    have hi : index < memory.length.val := by simpa [toBytes] using hindex
    have hsome := hvalid ⟨index, hi⟩
    cases hvalue : memory.get index with
    | none => simp [hvalue] at hsome
    | some value => simp [toBytes, hvalue]
  · simp at hread

end PagedBytes

def ofPages (start : Key) (memory : PagedBytes) (tainted defined : Bool) : Option Storage := do
  let data ← memory.checkedBytes
  ofBytes start data tainted defined

theorem ofPages_spanFits (start : Key) (memory : PagedBytes) (tainted defined : Bool)
    (state : Storage) (hfrom : ofPages start memory tainted defined = some state) :
    SpanFits state start memory.length := by
  unfold ofPages at hfrom
  cases hbytes : memory.checkedBytes with
  | none => simp [hbytes] at hfrom
  | some data =>
    have hrun : ofBytes start data tainted defined = some state := by simpa [hbytes] using hfrom
    have hlen := (PagedBytes.checkedBytes_sound memory data hbytes).1
    have hwidth : Slice.len data = memory.length := by scalar_tac
    simpa only [hwidth] using ofBytes_spanFits start data tainted defined state hrun

theorem ofPages_getCell (start : Key) (memory : PagedBytes) (tainted defined : Bool)
    (state : Storage) (index : Nat) (hindex : index < memory.length.val)
    (hfrom : ofPages start memory tainted defined = some state) :
    getCell state (offsetKey start index) =
      ⟨offsetKey start index, (memory.get index).getD 0#u8,
        none, tainted, defined, false, none⟩ := by
  unfold ofPages at hfrom
  cases hbytes : memory.checkedBytes with
  | none => simp [hbytes] at hfrom
  | some data =>
    have hrun : ofBytes start data tainted defined = some state := by simpa [hbytes] using hfrom
    obtain ⟨hlen, hget⟩ := PagedBytes.checkedBytes_sound memory data hbytes
    have hi : index < data.val.length := by omega
    rw [ofBytes_getCell start data tainted defined state index hi hrun]
    simp [hget index hi]

end Storage

end MemoryWrite

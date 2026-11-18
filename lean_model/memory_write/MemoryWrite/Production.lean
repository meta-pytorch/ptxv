/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Effects
import MemoryWrite.Address

open Aeneas Aeneas.Std PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite

instance (layout : MemoryLayout) (loc : MemoryLocation) (width : Nat) :
    Decidable (ReadValid layout loc width) := by
  unfold ReadValid
  cases loc <;> infer_instance

instance (layout : MemoryLayout) (loc : MemoryLocation) (width : Nat) :
    Decidable (WriteValid layout loc width) :=
  inferInstanceAs (Decidable (_ ∧ ReadValid layout loc width))

structure CheckedWrite where
  location : MemoryLocation
  effect : WriteEffect

def CheckedWrite.normalizedEffect (write : CheckedWrite) : WriteEffect :=
  { write.effect with destination := write.location.key }

def applyChecked (layout : MemoryLayout) (state : Storage) (write : CheckedWrite) :
    Except WriteError Storage :=
  let effect := write.normalizedEffect
  if effect.addressDependent then .error .inputDependentAddress
  else if effect.guardDependent then .error .inputDependentGuard
  else if !effect.enabled then .ok state
  else if !effect.family.allows effect.destination.space then .error .wrongSpace
  else if WriteValid layout write.location effect.payload.val.length then
    match write.location with
    | .local .. =>
      match state.registerWrite effect.destination effect.payload with
      | none => .error .outOfBounds
      | some next => .ok (next.installWords effect.words)
    | _ => applyEffect state effect
  else .error .outOfBounds

theorem applyChecked_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (layout : MemoryLayout) (before after : Storage)
    (write : CheckedWrite) (values : Input → Nat → Byte)
    (hstate : Storage.Represents sem memory before)
    (hsound : EffectSound sem memory write.normalizedEffect values)
    (hrun : applyChecked layout before write = .ok after) :
    Storage.Represents sem (effectMemory memory write.normalizedEffect values) after := by
  unfold applyChecked at hrun
  dsimp only at hrun
  split at hrun
  · simp at hrun
  · split at hrun
    · simp at hrun
    · split at hrun
      · rename_i hdisabled
        have henabled : write.normalizedEffect.enabled = false := by simpa using hdisabled
        have heq : after = before := by simpa using hrun.symm
        subst after
        simpa [effectMemory, henabled] using hstate
      · rename_i henabled
        have henabled : write.normalizedEffect.enabled = true := by simpa using henabled
        split at hrun
        · simp at hrun
        · split at hrun
          · cases hloc : write.location with
            | shared cta offset =>
              exact (applyEffect_preserves_representation sem memory before after
                write.normalizedEffect values hstate hsound (by simpa [hloc] using hrun)).1
            | global allocation offset =>
              exact (applyEffect_preserves_representation sem memory before after
                write.normalizedEffect values hstate hsound (by simpa [hloc] using hrun)).1
            | param grid offset =>
              exact (applyEffect_preserves_representation sem memory before after
                write.normalizedEffect values hstate hsound (by simpa [hloc] using hrun)).1
            | tmem cta lane column byte =>
              exact (applyEffect_preserves_representation sem memory before after
                write.normalizedEffect values hstate hsound (by simpa [hloc] using hrun)).1
            | «local» thread offset =>
              cases hregister : before.registerWrite write.normalizedEffect.destination
                  write.normalizedEffect.payload with
              | none => simp [hloc, hregister] at hrun
              | some next =>
                have heq : after = next.installWords write.normalizedEffect.words := by
                  simpa [hloc, hregister] using hrun.symm
                subst after
                apply Storage.installWords_preserves_representation
                · have h := (Storage.registerWrite_preserves_representation sem memory before next
                    write.normalizedEffect.destination write.normalizedEffect.payload values
                    hstate hsound.1 hregister).1
                  simpa [effectMemory, henabled] using h
                · exact hsound.2
          · simp at hrun

theorem applyChecked_permission (layout : MemoryLayout) (before after : Storage)
    (write : CheckedWrite) (henabled : write.effect.enabled = true)
    (hrun : applyChecked layout before write = .ok after) :
    WriteValid layout write.location write.effect.payload.val.length := by
  unfold applyChecked at hrun
  dsimp only at hrun
  split at hrun
  · simp at hrun
  · split at hrun
    · simp at hrun
    · split at hrun
      · simp [CheckedWrite.normalizedEffect, henabled] at *
      · split at hrun
        · simp at hrun
        · split at hrun
          · assumption
          · simp at hrun

def readChecked (layout : MemoryLayout) (state : Storage) (source : MemoryLocation)
    (width : Usize) : Except WriteError (Slice Cell) :=
  if ReadValid layout source width.val then
    match state.readSpan source.key width with
    | none => .error .unreadableSource
    | some cells => .ok cells
  else .error .unreadableSource

theorem readChecked_sound {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (layout : MemoryLayout) (state : Storage) (source : MemoryLocation) (width : Usize)
    (cells : Slice Cell) (hstate : Storage.Represents sem memory state)
    (hrun : readChecked layout state source width = .ok cells) :
    ReadValid layout source width.val ∧ cells.val.length = width.val ∧
      Contiguous cells source.key ∧ Represents sem memory cells := by
  unfold readChecked at hrun
  split at hrun
  · rename_i hvalid
    cases hread : state.readSpan source.key width with
    | none => simp [hread] at hrun
    | some actual =>
      have heq : cells = actual := by simpa [hread] using hrun.symm
      subst cells
      exact ⟨hvalid, Storage.readSpan_sound sem memory state source.key width actual hstate hread⟩
  · simp at hrun

structure WrittenMarkers where
  numericWritten : List Key := []
  halfWritten : List Key := []

structure WrittenTracking where
  numericEnabled : Bool := true
  halfEnabled : Bool := true

def alignedWordKey (key : Key) : Key :=
  { key with offset := U64.ofNat (key.offset.val / 4 * 4) (by scalar_tac) }

def numericTouchedKeys (start : Key) (width : Usize) : List Key :=
  (Storage.spanKeys start width).map alignedWordKey

def WrittenMarkers.record (markers : WrittenMarkers) (tracking : WrittenTracking)
    (effect : WriteEffect) : WrittenMarkers :=
  if effect.enabled && effect.destination.space == 1#u8 then
    { numericWritten := if tracking.numericEnabled then
        numericTouchedKeys effect.destination (Slice.len effect.payload) ++ markers.numericWritten
        else markers.numericWritten
      halfWritten := if tracking.halfEnabled then
        Storage.spanKeys effect.destination (Slice.len effect.payload) ++ markers.halfWritten
        else markers.halfWritten }
  else markers

def WrittenMarkers.recordNumeric (markers : WrittenMarkers) (tracking : WrittenTracking)
    (key : Key) : WrittenMarkers :=
  if tracking.numericEnabled && key.space == 1#u8 then
    { markers with numericWritten := key :: markers.numericWritten }
  else markers

@[simp]
theorem WrittenMarkers.recordNumeric_half_frame (markers : WrittenMarkers)
    (tracking : WrittenTracking) (key : Key) :
    (markers.recordNumeric tracking key).halfWritten = markers.halfWritten := by
  simp only [WrittenMarkers.recordNumeric]
  split <;> rfl

theorem WrittenMarkers.recordNumeric_mem (markers : WrittenMarkers)
    (tracking : WrittenTracking) (key other : Key) (henabled : tracking.numericEnabled = true)
    (hglobal : key.space = 1#u8) :
    other ∈ (markers.recordNumeric tracking key).numericWritten ↔
      other = key ∨ other ∈ markers.numericWritten := by
  simp [WrittenMarkers.recordNumeric, henabled, hglobal]

theorem WrittenMarkers.record_disabled (markers : WrittenMarkers) (tracking : WrittenTracking)
    (effect : WriteEffect) (hdisabled : effect.enabled = false) :
    markers.record tracking effect = markers := by simp [WrittenMarkers.record, hdisabled]

theorem WrittenMarkers.record_nonglobal (markers : WrittenMarkers) (tracking : WrittenTracking)
    (effect : WriteEffect) (hspace : effect.destination.space ≠ 1#u8) :
    markers.record tracking effect = markers := by simp [WrittenMarkers.record, hspace]

theorem WrittenMarkers.record_half_mem (markers : WrittenMarkers) (tracking : WrittenTracking)
    (effect : WriteEffect) (key : Key) (henabled : effect.enabled = true)
    (hspace : effect.destination.space = 1#u8) (htracking : tracking.halfEnabled = true) :
    key ∈ (markers.record tracking effect).halfWritten ↔
      key ∈ Storage.spanKeys effect.destination (Slice.len effect.payload) ∨
        key ∈ markers.halfWritten := by
  simp [WrittenMarkers.record, henabled, hspace, htracking]

theorem WrittenMarkers.record_numeric_mem (markers : WrittenMarkers) (tracking : WrittenTracking)
    (effect : WriteEffect) (key : Key) (henabled : effect.enabled = true)
    (hspace : effect.destination.space = 1#u8) (htracking : tracking.numericEnabled = true) :
    key ∈ (markers.record tracking effect).numericWritten ↔
      (∃ byte ∈ Storage.spanKeys effect.destination (Slice.len effect.payload),
        alignedWordKey byte = key) ∨ key ∈ markers.numericWritten := by
  simp [WrittenMarkers.record, henabled, hspace, htracking, numericTouchedKeys]

def poisonPayload (width : Usize) : Slice PayloadByte :=
  Slice.from (List.replicate width.val ⟨0#u8, none, true, false, none⟩) (by simp; scalar_tac)

@[simp]
theorem poisonPayload_length (width : Usize) : (poisonPayload width).val.length = width.val := by
  simp [poisonPayload]

theorem poisonPayload_at (width : Usize) (index : Nat) (hindex : index < width.val) :
    payloadAt (poisonPayload width) index = ⟨0#u8, none, true, false, none⟩ := by
  simp [payloadAt, poisonPayload, hindex]

def poisonValues {Input : Type} : Input → Nat → Byte := fun _ _ => ⟨0#u8, false⟩

theorem poisonPayload_sound {Input : Type} (sem : Semantics Input) (width : Usize) :
    PayloadSound sem (poisonPayload width) poisonValues := by
  constructor
  · intro index hindex
    rw [poisonPayload_at width index (by simpa using hindex)]
    constructor
    · rfl
    · simp
    · simp
    · simp
  · intro index hindex id hid
    rw [poisonPayload_at width index (by simp at hindex; omega)] at hid
    contradiction

def backingPayload (state : Storage) (start : Key) (width : Usize) : Slice PayloadByte :=
  Slice.from (List.ofFn fun index : Fin width.val =>
    (⟨(state.getCell (Storage.offsetKey start index.val)).value, none, true, false, none⟩ : PayloadByte))
    (by simp; scalar_tac)

@[simp]
theorem backingPayload_length (state : Storage) (start : Key) (width : Usize) :
    (backingPayload state start width).val.length = width.val := by simp [backingPayload]

def backingValues {Input : Type} (state : Storage) (start : Key) : Input → Nat → Byte :=
  fun _ index => ⟨(state.getCell (Storage.offsetKey start index)).value, false⟩

theorem backingPayload_sound {Input : Type} (sem : Semantics Input) (state : Storage)
    (start : Key) (width : Usize) :
    PayloadSound sem (backingPayload state start width) (backingValues state start) := by
  have hat (index : Nat) (hi : index < width.val) :
      payloadAt (backingPayload state start width) index =
        ⟨(state.getCell (Storage.offsetKey start index)).value, none, true, false, none⟩ := by
    simp [payloadAt, backingPayload, hi]
  constructor
  · intro index hi
    rw [hat index (by simpa using hi)]
    constructor
    · rfl
    · simp
    · simp
    · simp
  · intro index hi id hid
    rw [hat index (by simp at hi; omega)] at hid
    contradiction

def poisonChecked (layout : MemoryLayout) (state : Storage) (destination : MemoryLocation)
    (width : Usize) : Except WriteError Storage := do
  let source ← readChecked layout state destination width
  let payload := Slice.from (source.val.map fun cell =>
    (⟨cell.value, none, cell.tainted, false, none⟩ : PayloadByte)) (by simp)
  applyChecked layout state ⟨destination, {
    family := .poison, destination := destination.key, payload := payload }⟩

def undefinedValues {Input : Type} (memory : Memory Input) (source : Key) : Input → Nat → Byte :=
  fun input index => { memory input (location source index) with defined := false }

def undefinedPayload (source : Slice Cell) : Slice PayloadByte :=
  Slice.from (source.val.map fun cell =>
    (⟨cell.value, none, cell.tainted, false, none⟩ : PayloadByte)) (by simp)

@[simp]
theorem undefinedPayload_length (source : Slice Cell) :
    (undefinedPayload source).val.length = source.val.length := by simp [undefinedPayload]

theorem undefinedPayload_sound {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (source : Slice Cell) (start : Key) (hsource : Represents sem memory source)
    (hcontiguous : Contiguous source start) :
    PayloadSound sem (undefinedPayload source) (undefinedValues memory start) := by
  have hat (index : Nat) (hi : index < source.val.length) :
      payloadAt (undefinedPayload source) index =
        ⟨source.val[index].value, none, source.val[index].tainted, false, none⟩ := by
    simp [payloadAt, undefinedPayload, List.getElem?_eq_getElem hi]
  constructor
  · intro index hi
    have hi : index < source.val.length := by simpa using hi
    rw [hat index hi]
    have hcell := hsource source.val[index] (List.getElem_mem hi)
    constructor
    · simpa only [undefinedValues, hcontiguous index hi] using hcell.byte.captured
    · simp
    · simp
    · simpa only [undefinedValues, hcontiguous index hi] using hcell.byte.constant
  · intro index hi id hid
    rw [hat index (by simp at hi; omega)] at hid
    contradiction

theorem poisonChecked_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (layout : MemoryLayout) (before after : Storage)
    (destination : MemoryLocation) (width : Usize)
    (hstate : Storage.Represents sem memory before)
    (hrun : poisonChecked layout before destination width = .ok after) :
    Storage.Represents sem
      (storeBytes memory destination.key width.val (undefinedValues memory destination.key)) after := by
  cases hread : readChecked layout before destination width with
  | error error => simp [poisonChecked, Bind.bind, Except.bind, hread] at hrun
  | ok source =>
    have hs := readChecked_sound sem memory layout before destination width source hstate hread
    let effect : WriteEffect := {
      family := .poison, destination := destination.key, payload := undefinedPayload source }
    have hsound : EffectSound sem memory
        (CheckedWrite.normalizedEffect ⟨destination, effect⟩) (undefinedValues memory destination.key) :=
      ⟨undefinedPayload_sound sem memory source destination.key hs.2.2.2 hs.2.2.1,
        by simp [CheckedWrite.normalizedEffect, effect]⟩
    have hstep : applyChecked layout before ⟨destination, effect⟩ = .ok after := by
      simpa [poisonChecked, Bind.bind, Except.bind, hread, effect, undefinedPayload] using hrun
    have h := applyChecked_preserves_representation sem memory layout before after
      ⟨destination, effect⟩ (undefinedValues memory destination.key) hstate hsound hstep
    simpa [effectMemory, CheckedWrite.normalizedEffect, effect, hs.2.1] using h

def initializeSpan (state : Storage) (start : Key) (payload : Slice PayloadByte) : Option Storage :=
  (state.removeSpan start (Slice.len payload)).registerWrite start payload

theorem initializeSpan_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (before after : Storage) (start : Key) (payload : Slice PayloadByte)
    (values : Input → Nat → Byte) (hstate : Storage.Represents sem memory before)
    (hpayload : PayloadSound sem payload values)
    (hrun : initializeSpan before start payload = some after) :
    Storage.Represents sem (storeBytes memory start payload.val.length values) after ∧
      Storage.SpanFits after start (Slice.len payload) :=
  Storage.registerWrite_preserves_representation sem memory _ after start payload values
    (Storage.removeSpan_represents sem memory before start (Slice.len payload) hstate) hpayload hrun

theorem initializeSpan_new_cell (before after : Storage) (start : Key)
    (payload : Slice PayloadByte) (key : Key)
    (htouched : Touches start payload.val.length key)
    (hkey : key ∈ after.keys.val) (hrun : initializeSpan before start payload = some after) :
    after.getCell key =
      modelCell ⟨key, 0#u8, none, false, false, false, none⟩ start payload := by
  rw [Storage.getCell_registerWrite _ _ _ _ _ hrun hkey,
    Storage.getCell_removeSpan_touched before start key (Slice.len payload) (by simpa using htouched)]

def MemoryLayout.setTmemSpan (layout : MemoryLayout) (cta : U64)
    (start width : Nat) (allocated : Bool) : MemoryLayout :=
  { layout with tmemAllocated := fun region word =>
      if region = cta ∧ width ≠ 0 ∧ 4 * word < start + width ∧ start < 4 * word + 4
      then allocated else layout.tmemAllocated region word }

theorem MemoryLayout.setTmemSpan_inside (layout : MemoryLayout) (cta : U64)
    (start width : Nat) (allocated : Bool) (index : Nat) (hindex : index < width) :
    (layout.setTmemSpan cta start width allocated).tmemAllocated cta ((start + index) / 4) =
      allocated := by
  have hwidth : width ≠ 0 := by omega
  have hlo : 4 * ((start + index) / 4) < start + width := by omega
  have hhi : start < 4 * ((start + index) / 4) + 4 := by omega
  simp [setTmemSpan, hwidth, hlo, hhi]

def MemoryLayout.releaseAllocation (layout : MemoryLayout) (loc : MemoryLocation) : MemoryLayout :=
  match loc with
  | .shared cta _ => { layout with sharedBytes := fun region => if region = cta then 0 else layout.sharedBytes region }
  | .global allocation _ => { layout with globalBytes := fun region => if region = allocation then 0 else layout.globalBytes region }
  | .param grid _ => { layout with paramBytes := fun region => if region = grid then 0 else layout.paramBytes region }
  | .local thread _ => { layout with localBytes := fun region => if region = thread then 0 else layout.localBytes region }
  | .tmem cta .. => { layout with tmemAllocated := fun region word => if region = cta then false else layout.tmemAllocated region word }

def freeAllocation (layout : MemoryLayout) (state : Storage) (loc : MemoryLocation) :
    MemoryLayout × Storage :=
  (layout.releaseAllocation loc, state.removeAllocation loc.key)

theorem freeAllocation_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (layout : MemoryLayout) (state : Storage) (loc : MemoryLocation)
    (hstate : Storage.Represents sem memory state) :
    Storage.Represents sem memory (freeAllocation layout state loc).2 :=
  Storage.removeAllocation_represents sem memory state loc.key hstate

theorem freeAllocation_no_stale_keys (layout : MemoryLayout) (state : Storage)
    (loc : MemoryLocation) (key : Key)
    (hkey : key ∈ (freeAllocation layout state loc).2.keys.val) : ¬SameRegion loc.key key := by
  simp only [freeAllocation, Storage.removeAllocation, Storage.restrict, Slice.from_val] at hkey
  simpa using (List.mem_filter.mp hkey).2

def allocateTmem (layout : MemoryLayout) (state : Storage) (cta : U64)
    (lane : Fin 128) (column : Fin 512) (width : Usize) :
    Except WriteError (MemoryLayout × Storage) :=
  let loc := MemoryLocation.tmem cta lane column 0
  if loc.key.offset.val + width.val ≤ 262144 ∧ width.val % 4 = 0 then
    match initializeSpan state loc.key (backingPayload state loc.key width) with
    | none => .error .outOfBounds
    | some next => .ok (layout.setTmemSpan cta loc.key.offset.val width.val true, next)
  else .error .outOfBounds

theorem allocateTmem_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (layout nextLayout : MemoryLayout) (before after : Storage)
    (cta : U64) (lane : Fin 128) (column : Fin 512) (width : Usize)
    (hstate : Storage.Represents sem memory before)
    (hrun : allocateTmem layout before cta lane column width = .ok (nextLayout, after)) :
    Storage.Represents sem
      (storeBytes memory (MemoryLocation.tmem cta lane column 0).key width.val
        (backingValues before (MemoryLocation.tmem cta lane column 0).key)) after ∧
    ReadValid nextLayout (.tmem cta lane column 0) width.val ∧
      Storage.SpanFits after (MemoryLocation.tmem cta lane column 0).key width := by
  unfold allocateTmem at hrun
  dsimp only at hrun
  split at hrun
  · rename_i hbounds
    cases hinit : initializeSpan before (MemoryLocation.tmem cta lane column 0).key
        (backingPayload before (MemoryLocation.tmem cta lane column 0).key width) with
    | none => simp [hinit] at hrun
    | some next =>
      have heq : (nextLayout, after) =
          (layout.setTmemSpan cta (MemoryLocation.tmem cta lane column 0).key.offset.val width.val true, next) := by
        simpa [hinit] using hrun.symm
      cases heq
      have hrep := initializeSpan_preserves_representation sem memory before after
        (MemoryLocation.tmem cta lane column 0).key
        (backingPayload before (MemoryLocation.tmem cta lane column 0).key width)
        (backingValues before (MemoryLocation.tmem cta lane column 0).key)
        hstate (backingPayload_sound sem before _ width) hinit
      refine ⟨by simpa using hrep.1, ?_, ?_⟩
      · refine ⟨hbounds.1, ?_, ?_⟩
        · have := hbounds.1
          have : 262144 ≤ U64.max + 1 := by scalar_tac
          omega
        · intro index hindex
          exact MemoryLayout.setTmemSpan_inside layout cta _ width.val true index hindex
      · have hlen := backingPayload_length before (MemoryLocation.tmem cta lane column 0).key width
        have hwidth : Slice.len (backingPayload before (MemoryLocation.tmem cta lane column 0).key width) =
            width := by scalar_tac
        simpa only [hwidth] using hrep.2
  · simp at hrun

def freeTmemSpan (layout : MemoryLayout) (state : Storage) (cta : U64)
    (lane : Fin 128) (column : Fin 512) (width : Usize) : MemoryLayout × Storage :=
  let key := (MemoryLocation.tmem cta lane column 0).key
  (layout.setTmemSpan cta key.offset.val width.val false, state.releaseSpan key width)

theorem freeTmemSpan_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (layout : MemoryLayout) (state : Storage)
    (cta : U64) (lane : Fin 128) (column : Fin 512) (width : Usize)
    (hstate : Storage.Represents sem memory state) :
    Storage.Represents sem memory (freeTmemSpan layout state cta lane column width).2 :=
  Storage.releaseSpan_represents sem memory state _ width hstate

theorem freeTmemSpan_read_rejected (layout : MemoryLayout) (state : Storage)
    (cta : U64) (lane : Fin 128) (column : Fin 512) (width : Usize)
    (hwidth : 0 < width.val) :
    readChecked (freeTmemSpan layout state cta lane column width).1
      (freeTmemSpan layout state cta lane column width).2
      (.tmem cta lane column 0) width = .error .unreadableSource := by
  have hcleared := MemoryLayout.setTmemSpan_inside layout cta
    (MemoryLocation.tmem cta lane column 0).key.offset.val width.val false 0 hwidth
  have hword : ((MemoryLocation.tmem cta lane column 0).key.offset.val + 0) / 4 =
      tmemWordIndex lane column := by
    simp only [tmem_key_offset, tmemWordIndex]
    omega
  rw [hword] at hcleared
  have hinvalid := tmem_unallocated_read_invalid
    (layout.setTmemSpan cta (MemoryLocation.tmem cta lane column 0).key.offset.val width.val false)
    cta lane column 0 width.val hwidth hcleared
  have hinvalid' : ¬ReadValid (freeTmemSpan layout state cta lane column width).1
      (.tmem cta lane column 0) width.val := hinvalid
  unfold readChecked
  split
  · contradiction
  · rfl

inductive HalfKind where
  | f16 | bf16 | byte
  deriving DecidableEq, Repr

def HalfKind.width : HalfKind → Nat
  | .f16 | .bf16 => 2
  | .byte => 1

def clearHighBit (byte : U8) : U8 :=
  U8.ofNat (byte.val % 128) (by have := Nat.mod_lt byte.val (by decide : 0 < 128); scalar_tac)

@[simp]
theorem clearHighBit_val (byte : U8) : (clearHighBit byte).val = byte.val % 128 := rfl

structure ProductionDenotation (Input : Type) where
  domain : Input → Prop
  capture : Input
  capture_valid : domain capture
  initial : Memory Input
  allocationKind : U64 → HalfKind
  initialGlobalByte : U64 → Input → Nat → U8
  roundedNodeHalfByte : U32 → Input → Nat → U8
  roundedNodeWordByte : U32 → Input → Fin 4 → U8

def ProductionDenotation.origin {Input : Type} (model : ProductionDenotation Input)
    (origin : Origin) (input : Input) : U8 :=
  match origin with
  | .Zero => 0#u8
  | .E allocation element byte =>
      model.initialGlobalByte allocation input
        (element.val * (model.allocationKind allocation).width + byte.val)
  | .N node byte => model.roundedNodeHalfByte node input byte.val
  | .E7 allocation element => clearHighBit (model.initialGlobalByte allocation input element.val)

def ProductionDenotation.toSemantics {Input : Type} (model : ProductionDenotation Input) :
    Semantics Input :=
  { domain := model.domain
    capture := model.capture
    capture_valid := model.capture_valid
    initial := model.initial
    origin := model.origin
    zero_origin := by intro input _; rfl
    word := model.roundedNodeWordByte }

@[simp]
theorem ProductionDenotation.origin_zero {Input : Type} (model : ProductionDenotation Input)
    (input : Input) : model.toSemantics.origin .Zero input = 0#u8 := rfl

@[simp]
theorem ProductionDenotation.origin_element {Input : Type} (model : ProductionDenotation Input)
    (allocation element : U64) (byte : U8) (input : Input) :
    model.toSemantics.origin (.E allocation element byte) input =
      model.initialGlobalByte allocation input
        (element.val * (model.allocationKind allocation).width + byte.val) := rfl

@[simp]
theorem ProductionDenotation.origin_node {Input : Type} (model : ProductionDenotation Input)
    (node : U32) (byte : U8) (input : Input) :
    model.toSemantics.origin (.N node byte) input =
      model.roundedNodeHalfByte node input byte.val := rfl

@[simp]
theorem ProductionDenotation.origin_element7 {Input : Type} (model : ProductionDenotation Input)
    (allocation element : U64) (input : Input) :
    (model.toSemantics.origin (.E7 allocation element) input).val =
      (model.initialGlobalByte allocation input element.val).val % 128 := rfl

end MemoryWrite

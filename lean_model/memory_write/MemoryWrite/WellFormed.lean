/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Module

open Aeneas Aeneas.Std PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite

namespace Storage

structure WellFormed (state : Storage) : Prop where
  unique : state.keys.val.Nodup
  bytes : ∀ key ∈ state.keys.val, ∃ value, (key, value) ∈ state.bytes

theorem empty_wellFormed : WellFormed empty := by
  constructor <;> simp [empty]

theorem offsetKey_injective (start : Key) (width : Usize)
    (hwrap : start.offset.val + width.val ≤ U64.max + 1) :
    Function.Injective (fun index : Fin width.val => offsetKey start index.val) := by
  intro left right heq
  have hl := offsetKey_location start left.val (by have := left.isLt; omega)
  have hr := offsetKey_location start right.val (by have := right.isLt; omega)
  have hloc := congrArg (fun key => location key) heq
  rw [hl, hr] at hloc
  have hoff := congrArg Address.offset hloc
  simp only [location] at hoff
  apply Fin.ext
  omega

theorem spanKeys_nodup (start : Key) (width : Usize)
    (hwrap : start.offset.val + width.val ≤ U64.max + 1) : (spanKeys start width).Nodup :=
  List.nodup_ofFn.mpr (offsetKey_injective start width hwrap)

theorem allocation_wellFormed (start : Key) (data : Slice U8) (tainted defined : Bool)
    (hwrap : start.offset.val + data.val.length ≤ U64.max + 1) :
    WellFormed (allocation start data tainted defined) := by
  constructor
  · simpa only [allocation, Slice.from_val] using
      spanKeys_nodup start (Slice.len data) (by simpa using hwrap)
  · intro key hkey
    simp only [allocation, Slice.from_val] at hkey ⊢
    refine ⟨_, List.mem_map.mpr ⟨key, hkey, rfl⟩⟩

theorem ofBytes_wellFormed (start : Key) (data : Slice U8) (tainted defined : Bool)
    (state : Storage) (hrun : ofBytes start data tainted defined = some state) :
    WellFormed state := by
  unfold ofBytes at hrun
  split at hrun
  · rename_i hwrap
    have heq : state = allocation start data tainted defined := by simpa using hrun.symm
    subst state
    exact allocation_wellFormed start data tainted defined hwrap
  · simp at hrun

theorem fromCells_wellFormed (cells : Slice Cell) (state : Storage)
    (hrun : fromCells cells = some state) : WellFormed state := by
  unfold fromCells at hrun
  split at hrun
  · rename_i hunique
    have heq : state = fromCellsUnchecked cells := by simpa using hrun.symm
    subst state
    constructor
    · simpa only [fromCellsUnchecked, Slice.from_val] using hunique
    · intro key hkey
      simp only [fromCellsUnchecked, Slice.from_val] at hkey ⊢
      obtain ⟨cell, hcell, rfl⟩ := List.mem_map.mp hkey
      exact ⟨cell.value, List.mem_map.mpr ⟨cell, hcell, rfl⟩⟩
  · simp at hrun

theorem write_wellFormed (state : Storage) (start : Key) (payload : Slice PayloadByte)
    (hunique : state.keys.val.Nodup) : WellFormed (state.write start payload) := by
  constructor
  · exact hunique
  · intro key hkey
    apply Exists.intro _
    apply List.mem_append_left
    exact List.mem_map.mpr ⟨key, hkey, rfl⟩

theorem restrict_wellFormed (state : Storage) (keep : Key → Bool) (hstate : WellFormed state) :
    WellFormed (state.restrict keep) := by
  constructor
  · simpa only [restrict, Slice.from_val] using hstate.unique.filter keep
  · intro key hkey
    simp only [restrict, Slice.from_val, List.mem_filter] at hkey
    obtain ⟨value, hvalue⟩ := hstate.bytes key hkey.1
    exact ⟨value, List.mem_filter.mpr ⟨hvalue, hkey.2⟩⟩

theorem removeSpan_wellFormed (state : Storage) (start : Key) (width : Usize)
    (hstate : WellFormed state) : WellFormed (state.removeSpan start width) := by
  have h := restrict_wellFormed state (fun key => decide (¬Touches start width.val key)) hstate
  constructor
  · simpa only [removeSpan] using h.unique
  · simpa only [removeSpan] using h.bytes

theorem releaseSpan_wellFormed (state : Storage) (start : Key) (width : Usize)
    (hstate : WellFormed state) : WellFormed (state.releaseSpan start width) := by
  constructor
  · exact (removeSpan_wellFormed state start width hstate).unique
  · intro key hkey
    have hbefore : key ∈ state.keys.val := by
      rw [releaseSpan_keys, removeSpan_keys] at hkey
      exact (List.mem_filter.mp hkey).1
    exact hstate.bytes key hbefore

theorem removeAllocation_wellFormed (state : Storage) (allocation : Key)
    (hstate : WellFormed state) : WellFormed (state.removeAllocation allocation) :=
  restrict_wellFormed state _ hstate

theorem extensionKeys_nodup (state : Storage) (start : Key) (width : Usize)
    (hunique : state.keys.val.Nodup) (hwrap : start.offset.val + width.val ≤ U64.max + 1) :
    (extensionKeys state start width).Nodup := by
  apply List.nodup_append.mpr
  refine ⟨hunique, (spanKeys_nodup start width hwrap).filter _, ?_⟩
  intro left hleft right hright heq
  have hmissing : right ∉ state.keys.val := by simpa using (List.mem_filter.mp hright).2
  exact hmissing (heq ▸ hleft)

theorem registerWrite_wellFormed (before after : Storage) (start : Key) (payload : Slice PayloadByte)
    (hstate : WellFormed before) (hrun : registerWrite before start payload = some after) :
    WellFormed after := by
  unfold registerWrite at hrun
  split at hrun
  · rename_i hwrap
    split at hrun
    · rename_i hsize
      have heq : after = write (extend before start (Slice.len payload) hsize) start payload := by
        simpa using hrun.symm
      subst after
      apply write_wellFormed
      simpa only [extend, Slice.from_val] using
        extensionKeys_nodup before start (Slice.len payload) hstate.unique (by simpa using hwrap)
    · simp at hrun
  · simp at hrun

theorem setWord_wellFormed (state : Storage) (update : WordUpdate) (hstate : WellFormed state) :
    WellFormed (state.setWord update) := by
  constructor
  · simpa only [setWord] using hstate.unique
  · simpa only [setWord] using hstate.bytes

theorem installWords_wellFormed (state : Storage) (updates : List WordUpdate)
    (hstate : WellFormed state) : WellFormed (state.installWords updates) := by
  induction updates generalizing state with
  | nil => exact hstate
  | cons update updates ih => exact ih (state.setWord update) (setWord_wellFormed state update hstate)

theorem extensionKeys_eq_of_spanFits (state : Storage) (start : Key) (width : Usize)
    (hspan : SpanFits state start width) : extensionKeys state start width = state.keys.val := by
  unfold extensionKeys
  have hfilter : (spanKeys start width).filter (fun key => decide (key ∉ state.keys.val)) = [] := by
    apply List.filter_eq_nil_iff.mpr
    intro key hkey
    simp [hspan.2 key hkey]
  simp only [hfilter, List.append_nil]

theorem registerWrite_preallocated_keys (before after : Storage) (start : Key) (payload : Slice PayloadByte)
    (hspan : SpanFits before start (Slice.len payload))
    (hrun : registerWrite before start payload = some after) : after.keys = before.keys := by
  unfold registerWrite at hrun
  split at hrun
  · split at hrun
    · rename_i hsize
      have heq : after = write (extend before start (Slice.len payload) hsize) start payload := by
        simpa using hrun.symm
      subst after
      apply Slice.ext
      simpa only [write_keys, extend, Slice.from_val] using
        extensionKeys_eq_of_spanFits before start (Slice.len payload) hspan
    · simp at hrun
  · simp at hrun

end Storage

theorem applyEffect_wellFormed (before after : Storage) (effect : WriteEffect)
    (hstate : Storage.WellFormed before) (hrun : applyEffect before effect = .ok after) :
    Storage.WellFormed after := by
  unfold applyEffect at hrun
  split at hrun
  · simp at hrun
  · split at hrun
    · simp at hrun
    · split at hrun
      · have heq : after = before := by simpa using hrun.symm
        subst after
        exact hstate
      · split at hrun
        · simp at hrun
        · split at hrun
          · have heq : after = (before.write effect.destination effect.payload).installWords effect.words := by
              simpa using hrun.symm
            subst after
            exact Storage.installWords_wellFormed _ _
              (Storage.write_wellFormed before effect.destination effect.payload hstate.unique)
          · simp at hrun

theorem applyChecked_wellFormed (layout : MemoryLayout) (before after : Storage) (request : CheckedWrite)
    (hstate : Storage.WellFormed before) (hrun : applyChecked layout before request = .ok after) :
    Storage.WellFormed after := by
  unfold applyChecked at hrun
  dsimp only at hrun
  split at hrun
  · simp at hrun
  · split at hrun
    · simp at hrun
    · split at hrun
      · have heq : after = before := by simpa using hrun.symm
        subst after
        exact hstate
      · split at hrun
        · simp at hrun
        · split at hrun
          · cases hloc : request.location with
            | «local» thread offset =>
              cases hregister : before.registerWrite request.normalizedEffect.destination
                  request.normalizedEffect.payload with
              | none => simp [hloc, hregister] at hrun
              | some next =>
                have heq : after = next.installWords request.normalizedEffect.words := by
                  simpa [hloc, hregister] using hrun.symm
                subst after
                exact Storage.installWords_wellFormed _ _
                  (Storage.registerWrite_wellFormed before next _ _ hstate hregister)
            | shared cta offset | global cta offset | param cta offset =>
              exact applyEffect_wellFormed before after request.normalizedEffect hstate
                (by simpa [hloc] using hrun)
            | tmem cta lane column byte =>
              exact applyEffect_wellFormed before after request.normalizedEffect hstate
                (by simpa [hloc] using hrun)
          · simp at hrun

theorem initializeSpan_wellFormed (before after : Storage) (start : Key) (payload : Slice PayloadByte)
    (hstate : Storage.WellFormed before) (hrun : initializeSpan before start payload = some after) :
    Storage.WellFormed after :=
  Storage.registerWrite_wellFormed _ after start payload
    (Storage.removeSpan_wellFormed before start (Slice.len payload) hstate) hrun

theorem allocateTmem_wellFormed (layout nextLayout : MemoryLayout) (before after : Storage)
    (cta : U64) (lane : Fin 128) (column : Fin 512) (width : Usize)
    (hstate : Storage.WellFormed before)
    (hrun : allocateTmem layout before cta lane column width = .ok (nextLayout, after)) :
    Storage.WellFormed after := by
  unfold allocateTmem at hrun
  dsimp only at hrun
  split at hrun
  · cases hinit : initializeSpan before (MemoryLocation.tmem cta lane column 0).key
        (backingPayload before (MemoryLocation.tmem cta lane column 0).key width) with
    | none => simp [hinit] at hrun
    | some next =>
      have heq : (nextLayout, after) =
          (layout.setTmemSpan cta (MemoryLocation.tmem cta lane column 0).key.offset.val width.val true, next) := by
        simpa [hinit] using hrun.symm
      have hafter : after = next := congrArg Prod.snd heq
      subst after
      exact initializeSpan_wellFormed before next _ _ hstate hinit
  · simp at hrun

theorem freeTmemSpan_wellFormed (layout : MemoryLayout) (state : Storage) (cta : U64)
    (lane : Fin 128) (column : Fin 512) (width : Usize) (hstate : Storage.WellFormed state) :
    Storage.WellFormed (freeTmemSpan layout state cta lane column width).2 :=
  Storage.releaseSpan_wellFormed state _ width hstate

theorem centralWrite_wellFormed (layout : MemoryLayout) (before after : CentralState) (request : CheckedWrite)
    (hstate : Storage.WellFormed before.storage) (hrun : write layout before request = .ok after) :
    Storage.WellFormed after.storage := by
  unfold write at hrun
  cases hstep : applyChecked layout before.storage request with
  | error error => simp [hstep] at hrun
  | ok storage =>
    have heq : after = { before with
        storage := storage
        markers := before.markers.record before.tracking request.normalizedEffect } := by
      simpa [hstep] using hrun.symm
    subst after
    exact applyChecked_wellFormed layout before.storage storage request hstate hstep

theorem runChecked_wellFormed (layout : MemoryLayout) (before after : CentralState)
    (requests : List CheckedWrite) (hstate : Storage.WellFormed before.storage)
    (hrun : runChecked layout before requests = .ok after) : Storage.WellFormed after.storage := by
  induction requests generalizing before with
  | nil =>
    have heq : after = before := by simpa [runChecked] using hrun.symm
    subst after
    exact hstate
  | cons request requests ih =>
    cases hstep : write layout before request with
    | error error => simp [runChecked, hstep, Bind.bind, Except.bind] at hrun
    | ok middle =>
      have hrest : runChecked layout middle requests = .ok after := by
        simpa [runChecked, hstep, Bind.bind, Except.bind] using hrun
      exact ih middle (centralWrite_wellFormed layout before middle request hstate hstep) hrest

theorem readChecked_progress (layout : MemoryLayout) (state : Storage) (source : MemoryLocation)
    (width : Usize) (hvalid : ReadValid layout source width.val)
    (hspan : Storage.SpanFits state source.key width) :
    ∃ cells, readChecked layout state source width = .ok cells ∧ cells.val.length = width.val := by
  refine ⟨Storage.readSpanUnchecked state source.key width, ?_, by simp⟩
  simp [readChecked, Storage.readSpan, hvalid, hspan]

theorem applyChecked_local_progress (layout : MemoryLayout) (state : Storage) (thread offset : U64)
    (payload : Slice PayloadByte)
    (hvalid : WriteValid layout (.local thread offset) payload.val.length)
    (hsize : (Storage.extensionKeys state (MemoryLocation.local thread offset).key (Slice.len payload)).length ≤
      Usize.max) :
    ∃ next, applyChecked layout state
        ⟨.local thread offset, {
          family := .localStore
          destination := (MemoryLocation.local thread offset).key
          payload := payload }⟩ = .ok next ∧
      Storage.SpanFits next (MemoryLocation.local thread offset).key (Slice.len payload) := by
  let key := (MemoryLocation.local thread offset).key
  let next := (Storage.extend state key (Slice.len payload) hsize).write key payload
  have hwrap := read_valid_no_wrap layout (.local thread offset) payload.val.length hvalid.2
  have hregister : state.registerWrite key payload = some next := by
    simp only [Storage.registerWrite, key, dite_eq_left hsize, ite_eq_left hwrap]
    rfl
  refine ⟨next, ?_, ?_⟩
  · have hregister' : state.registerWrite ⟨4#u8, thread, offset⟩ payload = some next := hregister
    simp [applyChecked, CheckedWrite.normalizedEffect, WriteFamily.allows, MemoryLocation.key,
      hvalid, hregister', Storage.installWords]
  · exact Storage.extend_spanFits state key (Slice.len payload) hsize (by simpa using hwrap)

def SupportedWrite (layout : MemoryLayout) (state : Storage) (request : CheckedWrite) : Prop :=
  request.effect.addressDependent = false ∧ request.effect.guardDependent = false ∧
    (request.effect.enabled = false ∨
      request.effect.family.allows request.location.key.space = true ∧
      WriteValid layout request.location request.effect.payload.val.length ∧
      Storage.SpanFits state request.location.key (Slice.len request.effect.payload))

theorem applyChecked_progress (layout : MemoryLayout) (state : Storage) (request : CheckedWrite)
    (hsupported : SupportedWrite layout state request) :
    ∃ next, applyChecked layout state request = .ok next ∧ next.keys = state.keys := by
  obtain ⟨haddress, hguard, hsupport⟩ := hsupported
  rcases hsupport with hdisabled | ⟨hfamily, hvalid, hspan⟩
  · exact ⟨state, by simp [applyChecked, CheckedWrite.normalizedEffect, haddress, hguard, hdisabled], rfl⟩
  · have hsize : (Storage.extensionKeys state request.location.key (Slice.len request.effect.payload)).length ≤
        Usize.max := by
      rw [Storage.extensionKeys_eq_of_spanFits state request.location.key (Slice.len request.effect.payload) hspan]
      exact state.keys.property
    have hregister : ∃ next, state.registerWrite request.location.key request.effect.payload = some next := by
      unfold Storage.registerWrite
      have hwrap : request.location.key.offset.val + request.effect.payload.val.length ≤ U64.max + 1 := by
        simpa using hspan.1
      exact ⟨(state.extend request.location.key (Slice.len request.effect.payload) hsize).write
        request.location.key request.effect.payload, by simp only [dite_eq_left hsize, ite_eq_left hwrap]⟩
    obtain ⟨localNext, hlocal⟩ := hregister
    have hkeys := Storage.registerWrite_preallocated_keys state localNext request.location.key
      request.effect.payload hspan hlocal
    by_cases henabled : request.effect.enabled = true
    · cases hloc : request.location with
      | «local» thread offset =>
        simp only [hloc] at hfamily hvalid hlocal
        refine ⟨localNext.installWords request.effect.words, ?_, by simpa using hkeys⟩
        simp [applyChecked, CheckedWrite.normalizedEffect, haddress, hguard, henabled, hfamily,
          hvalid, hloc, hlocal]
      | shared cta offset | global cta offset =>
        simp only [hloc] at hfamily hvalid hspan
        refine ⟨(state.write request.location.key request.effect.payload).installWords request.effect.words, ?_, by simp⟩
        simp [applyChecked, applyEffect, CheckedWrite.normalizedEffect, haddress, hguard, henabled,
          hfamily, hvalid, hloc, hspan]
      | tmem cta lane column byte =>
        simp only [hloc] at hfamily hvalid hspan
        refine ⟨(state.write request.location.key request.effect.payload).installWords request.effect.words, ?_, by simp⟩
        simp [applyChecked, applyEffect, CheckedWrite.normalizedEffect, haddress, hguard, henabled,
          hfamily, hvalid, hloc, hspan]
      | param grid offset =>
        exact False.elim (param_write_invalid layout grid offset _ (by simpa only [hloc] using hvalid))
    · have hdisabled : request.effect.enabled = false := by simpa using henabled
      exact ⟨state, by simp [applyChecked, CheckedWrite.normalizedEffect, haddress, hguard, hdisabled], rfl⟩

theorem SupportedWrite.of_keys_eq (layout : MemoryLayout) (before after : Storage) (request : CheckedWrite)
    (hkeys : after.keys = before.keys) (hsupported : SupportedWrite layout before request) :
    SupportedWrite layout after request := by
  unfold SupportedWrite Storage.SpanFits at *
  simpa only [hkeys] using hsupported

theorem write_progress (layout : MemoryLayout) (state : CentralState) (request : CheckedWrite)
    (hsupported : SupportedWrite layout state.storage request) :
    ∃ next, write layout state request = .ok next ∧ next.storage.keys = state.storage.keys := by
  obtain ⟨storage, hstep, hkeys⟩ := applyChecked_progress layout state.storage request hsupported
  exact ⟨{ state with
      storage := storage
      markers := state.markers.record state.tracking request.normalizedEffect },
    by simp [write, hstep], hkeys⟩

theorem runChecked_progress (layout : MemoryLayout) (state : CentralState) (requests : List CheckedWrite)
    (hsupported : ∀ request ∈ requests, SupportedWrite layout state.storage request) :
    ∃ next, runChecked layout state requests = .ok next ∧ next.storage.keys = state.storage.keys := by
  induction requests generalizing state with
  | nil => exact ⟨state, rfl, rfl⟩
  | cons request requests ih =>
    obtain ⟨middle, hfirst, hkeys⟩ := write_progress layout state request (hsupported request (by simp))
    have hrest : ∀ other ∈ requests, SupportedWrite layout middle.storage other := by
      intro other hother
      exact SupportedWrite.of_keys_eq layout state.storage middle.storage other hkeys
        (hsupported other (by simp [hother]))
    obtain ⟨next, hfinish, hnext⟩ := ih middle hrest
    exact ⟨next, by simpa [runChecked, hfirst, Bind.bind, Except.bind] using hfinish,
      hnext.trans hkeys⟩

end MemoryWrite

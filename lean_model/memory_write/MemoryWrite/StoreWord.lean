/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.SymbolicState
import MemoryWrite.RustEngine

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.StoreWord

open Symbolic

def payload (source : Metadata) : Slice PayloadByte :=
  Producers.registerPayload (fun _ => registerWord source) [0] (by scalar_tac)

def values {Input : Type} (source : Input → Value) : Input → Nat → Byte :=
  fun input index => byteValues source input ⟨index % 4, by omega⟩

@[simp] theorem payload_length (source : Metadata) : (payload source).val.length = 4 := by
  simp [payload]

theorem payload_at (source : Metadata) (index : Nat) (hi : index < 4) :
    payloadAt (payload source) index =
      ⟨byteAt source.captured ⟨index, hi⟩, none, source.tainted, source.defined,
        if index = 0 then source.history.map WordHistory.Known else none⟩ := by
  rw [payload, Producers.registerPayload_at _ _ _ index (by simpa using hi)]
  have hm : index % 4 = index := Nat.mod_eq_of_lt hi
  simp [registerWord, Producers.RegisterByte.toPayload, hm]

theorem payload_sound {Input : Type} (sem : Semantics Input) (meaning : Meaning Input)
    (source : Metadata) (sourceValues : Input → Value)
    (hs : Sound sem meaning source sourceValues)
    (hm : ∀ id input lane, sem.word id input lane = byteAt (meaning id input) lane) :
    PayloadSound sem (payload source) (values sourceValues) := by
  have hword := derives_register_word_sound sem meaning source sourceValues hs hm
  exact Producers.registerPayload_sound sem (fun _ => registerWord source)
    (fun _ => byteValues sourceValues) [0] (by scalar_tac) (by intros; exact hword)

def destination (space : U8) (allocation : U64) (low high : Metadata) : Key :=
  ⟨space, allocation, ⟨high.captured.bv ++ low.captured.bv⟩⟩

def effect (space : U8) (allocation : U64) (source guard low high : Metadata) : WriteEffect :=
  { family := .store
    destination := destination space allocation low high
    payload := payload source
    enabled := decide (guard.captured ≠ 0#u32)
    addressDependent := low.tainted || high.tainted
    guardDependent := guard.tainted }

theorem effect_sound {Input : Type} (sem : Semantics Input) (meaning : Meaning Input)
    (memory : Memory Input) (source : Metadata) (sourceValues : Input → Value)
    (hs : Sound sem meaning source sourceValues)
    (hm : ∀ id input lane, sem.word id input lane = byteAt (meaning id input) lane)
    (space : U8) (allocation : U64) (guard low high : Metadata) :
    EffectSound sem memory (effect space allocation source guard low high) (values sourceValues) := by
  exact ⟨payload_sound sem meaning source sourceValues hs hm, by simp [effect]⟩

theorem destination_capture {Input : Type} (sem : Semantics Input) (meaning : Meaning Input)
    (low high : Metadata) (lowValues highValues : Input → Value)
    (hl : Sound sem meaning low lowValues) (hh : Sound sem meaning high highValues)
    (space : U8) (allocation : U64) :
    destination space allocation low high =
      addressFromWords space allocation lowValues highValues sem.capture := by
  simp [destination, addressFromWords, hl.captured, hh.captured]

theorem represents_congr {Input : Type} (sem : Semantics Input) (left right : Memory Input)
    (cells : Slice Cell) (heq : ∀ input, sem.domain input → ∀ address,
      left input address = right input address) (hrep : Represents sem left cells) :
    Represents sem right cells := by
  intro cell hc
  have hr := hrep cell hc
  constructor
  · constructor
    · simpa only [←heq sem.capture sem.capture_valid] using hr.byte.captured
    · intro hd input hi
      simpa only [←heq input hi] using hr.byte.defined_sound hd input hi
    · intro id hid input hi
      simpa only [←heq input hi] using hr.byte.origin_sound id hid input hi
    · intro ht input hi
      simpa only [←heq input hi, ←heq sem.capture sem.capture_valid] using
        hr.byte.constant ht input hi
  · intro id hid input hi lane
    simpa only [←heq input hi] using hr.word id hid input hi lane
  · intro hw input hi
    simpa only [←heq input hi] using hr.initial_fallback hw input hi

theorem effect_memory_eq_dynamic {Input : Type} (sem : Semantics Input) (meaning : Meaning Input)
    (memory : Memory Input) (source guard low high : Metadata)
    (sourceValues guardValues lowValues highValues : Input → Value)
    (hg : Sound sem meaning guard guardValues) (hl : Sound sem meaning low lowValues)
    (hh : Sound sem meaning high highValues)
    (hguard : guard.tainted = false) (hlow : low.tainted = false) (hhigh : high.tainted = false)
    (space : U8) (allocation : U64) (input : Input) (hi : sem.domain input) (address : Address) :
    effectMemory memory (effect space allocation source guard low high) (values sourceValues) input address =
      Producers.dynamicStore memory (predicateGuard guardValues)
        (addressFromWords space allocation lowValues highValues) 4 (values sourceValues) input address := by
  have hstable := derives_stable_store_execution sem meaning guard low high
    guardValues lowValues highValues hg hl hh hguard hlow hhigh space allocation
  rw [Producers.dynamicStore_captured_execution sem memory _ _ _ _ hstable input hi address]
  have hd := destination_capture sem meaning low high lowValues highValues hl hh space allocation
  simp only [effectMemory, effect, payload_length, hd, predicateGuard, hg.captured]
  split <;> simp_all

theorem model_frame (before after : Storage) (space : U8) (allocation : U64)
    (source guard low high : Metadata)
    (hrun : applyEffect before (effect space allocation source guard low high) = .ok after)
    (key : Key) (hk : key ∈ before.keys.val)
    (hframe : ¬Overlaps (destination space allocation low high) 4 key) :
    after.getCell key = before.getCell key := by
  have hf := Footprint.applyEffect_cells before after _ hrun
  rw [hf.2.1 key (by simpa [hf.1] using hk)]
  apply Footprint.effectCell_frame
  intro haffected
  rcases haffected with ⟨_, hbyte⟩ | ⟨_, hword⟩
  · exact hframe (Storage.touches_overlaps _ _ _ (by simpa [effect] using hbyte))
  · exact hframe (by simpa [effect] using hword)

theorem accepted_control (before after : Storage) (space : U8) (allocation : U64)
    (source guard low high : Metadata)
    (hrun : applyEffect before (effect space allocation source guard low high) = .ok after) :
    guard.tainted = false ∧ low.tainted = false ∧ high.tainted = false := by
  unfold applyEffect effect at hrun
  split at hrun
  · simp at hrun
  · rename_i ha
    have hadd : low.tainted = false ∧ high.tainted = false := by simpa using ha
    split at hrun
    · simp at hrun
    · rename_i hg
      exact ⟨by simpa using hg, hadd⟩

theorem accepted_preserves_dynamic {Input : Type} (sem : Semantics Input)
    (meaning : Meaning Input) (memory : Memory Input) (before after : Storage)
    (space : U8) (allocation : U64) (source guard low high : Metadata)
    (sourceValues guardValues lowValues highValues : Input → Value)
    (hs : Sound sem meaning source sourceValues) (hg : Sound sem meaning guard guardValues)
    (hl : Sound sem meaning low lowValues) (hh : Sound sem meaning high highValues)
    (hm : ∀ id input lane, sem.word id input lane = byteAt (meaning id input) lane)
    (hstate : Storage.Represents sem memory before)
    (hrun : applyEffect before (effect space allocation source guard low high) = .ok after) :
    Storage.Represents sem
      (Producers.dynamicStore memory (predicateGuard guardValues)
        (addressFromWords space allocation lowValues highValues) 4 (values sourceValues)) after ∧
      Producers.StableStoreExecution sem (predicateGuard guardValues)
        (addressFromWords space allocation lowValues highValues) ∧ after.keys = before.keys := by
  obtain ⟨hgfalse, hlfalse, hhfalse⟩ := accepted_control before after space allocation
    source guard low high hrun
  have heffect := effect_sound sem meaning memory source sourceValues hs hm space allocation guard low high
  have hp := applyEffect_preserves_representation sem memory before after _ _ hstate heffect hrun
  refine ⟨?_, derives_stable_store_execution sem meaning guard low high guardValues lowValues
    highValues hg hl hh hgfalse hlfalse hhfalse space allocation, hp.2⟩
  apply represents_congr sem _ _ after.toCells _ hp.1
  intro input hi address
  exact effect_memory_eq_dynamic sem meaning memory source guard low high
    sourceValues guardValues lowValues highValues hg hl hh hgfalse hlfalse hhfalse
      space allocation input hi address

end MemoryWrite.StoreWord

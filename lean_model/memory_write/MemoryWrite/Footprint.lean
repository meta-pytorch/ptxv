/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Production

open Aeneas Aeneas.Std PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite.Footprint

def postWords (updates : List WordUpdate) (key : Key) (history : Option WordHistory) :
    Option WordHistory :=
  match updates with
  | [] => history
  | update :: rest => postWords rest key (if update.key = key then update.history else history)

def effectCell (effect : WriteEffect) (cell : Cell) : Cell :=
  if effect.enabled then
    let changed := modelCell cell effect.destination effect.payload
    { changed with word := postWords effect.words changed.key changed.word }
  else cell

def ByteAffected (effect : WriteEffect) (key : Key) : Prop :=
  effect.enabled = true ∧ Touches effect.destination effect.payload.val.length key

def WordAffected (effect : WriteEffect) (key : Key) : Prop :=
  effect.enabled = true ∧
    (Overlaps effect.destination effect.payload.val.length key ∨
      ∃ update ∈ effect.words, update.key = key)

def Complete (effect : WriteEffect) (key : Key) : Prop :=
  ByteAffected effect key ∨ WordAffected effect key

instance (effect : WriteEffect) (key : Key) : Decidable (ByteAffected effect key) :=
  inferInstanceAs (Decidable (_ ∧ _))

instance (effect : WriteEffect) (key : Key) : Decidable (WordAffected effect key) :=
  inferInstanceAs (Decidable (_ ∧ (_ ∨ _)))

instance (effect : WriteEffect) (key : Key) : Decidable (Complete effect key) :=
  inferInstanceAs (Decidable (_ ∨ _))

def byteView (cell : Cell) : U8 × Option Origin × Bool × Bool × Bool :=
  (cell.value, cell.origin, cell.tainted, cell.defined, cell.written)

@[simp] theorem effectCell_key (effect : WriteEffect) (cell : Cell) :
    (effectCell effect cell).key = cell.key := by
  simp only [effectCell]
  split
  · exact modelCell_key cell effect.destination effect.payload
  · rfl

theorem getCell_installWords (state : Storage) (updates : List WordUpdate) (key : Key) :
    (state.installWords updates).getCell key =
      { state.getCell key with word := postWords updates key (state.getCell key).word } := by
  induction updates generalizing state with
  | nil => rfl
  | cons update updates ih =>
    change ((state.setWord update).installWords updates).getCell key = _
    rw [ih, Storage.getCell_setWord]
    rfl

theorem postWords_frame (updates : List WordUpdate) (key : Key) (history : Option WordHistory)
    (hframe : ∀ update ∈ updates, update.key ≠ key) :
    postWords updates key history = history := by
  induction updates generalizing history with
  | nil => rfl
  | cons update updates ih =>
    have hupdate := hframe update (by simp)
    simp only [postWords, ite_eq_right hupdate]
    exact ih history (fun other hother => hframe other (by simp [hother]))

theorem wordAfter_frame (cell : Cell) (destination : Key) (payload : Slice PayloadByte)
    (hframe : ¬Overlaps destination payload.val.length cell.key) :
    wordAfter cell destination payload = cell.word := by
  have hcover : ¬CoversWord destination payload.val.length cell.key := by
    intro h
    exact hframe (Storage.touches_overlaps destination cell.key payload.val.length
      (covers_touches destination cell.key payload.val.length h))
  simp [wordAfter, hcover, hframe]

theorem effectCell_byte_effect (effect : WriteEffect) (cell : Cell) :
    byteView (effectCell effect cell) =
      if ByteAffected effect cell.key then
        let byte := payloadAt effect.payload (cell.key.offset.val - effect.destination.offset.val)
        (byte.value, byte.origin, byte.tainted, byte.defined, true)
      else byteView cell := by
  cases henabled : effect.enabled <;>
    by_cases ht : Touches effect.destination effect.payload.val.length cell.key <;>
    simp [effectCell, ByteAffected, byteView, modelCell, henabled, ht]

theorem effectCell_word_effect (effect : WriteEffect) (cell : Cell) :
    (effectCell effect cell).word = if effect.enabled then
      postWords effect.words cell.key (wordAfter cell effect.destination effect.payload)
      else cell.word := by
  cases henabled : effect.enabled <;>
    by_cases ht : Touches effect.destination effect.payload.val.length cell.key <;>
    simp [effectCell, henabled, modelCell, ht]

theorem effectCell_byte_frame (effect : WriteEffect) (cell : Cell)
    (hframe : ¬ByteAffected effect cell.key) :
    byteView (effectCell effect cell) = byteView cell := by
  rw [effectCell_byte_effect]
  simp [hframe]

theorem effectCell_word_frame (effect : WriteEffect) (cell : Cell)
    (hframe : ¬WordAffected effect cell.key) :
    (effectCell effect cell).word = cell.word := by
  cases henabled : effect.enabled with
  | false => simp [effectCell_word_effect, henabled]
  | true =>
    have hoverlap : ¬Overlaps effect.destination effect.payload.val.length cell.key := by
      intro h
      exact hframe ⟨henabled, Or.inl h⟩
    have hupdates : ∀ update ∈ effect.words, update.key ≠ cell.key := by
      intro update hupdate heq
      exact hframe ⟨henabled, Or.inr ⟨update, hupdate, heq⟩⟩
    rw [effectCell_word_effect]
    simp [henabled, wordAfter_frame cell effect.destination effect.payload hoverlap,
      postWords_frame effect.words cell.key cell.word hupdates]

theorem effectCell_frame (effect : WriteEffect) (cell : Cell)
    (hframe : ¬Complete effect cell.key) : effectCell effect cell = cell := by
  have hbyte := effectCell_byte_frame effect cell (fun h => hframe (Or.inl h))
  have hword := effectCell_word_frame effect cell (fun h => hframe (Or.inr h))
  have hkey := effectCell_key effect cell
  cases hcell : effectCell effect cell
  cases cell
  simp only [byteView, Prod.mk.injEq] at hbyte
  simp_all

def IsLocal : MemoryLocation → Prop
  | .local .. => True
  | _ => False

instance (location : MemoryLocation) : Decidable (IsLocal location) := by
  cases location <;> simp [IsLocal] <;> infer_instance

def Introduced (write : CheckedWrite) (key : Key) : Prop :=
  write.effect.enabled = true ∧ IsLocal write.location ∧
    key ∈ Storage.spanKeys write.normalizedEffect.destination (Slice.len write.effect.payload)

structure CheckedSpec (before after : Storage) (write : CheckedWrite) : Prop where
  keys : ∀ key, key ∈ after.keys.val ↔ key ∈ before.keys.val ∨ Introduced write key
  cells : ∀ key ∈ after.keys.val,
    after.getCell key = effectCell write.normalizedEffect (before.getCell key)
  inactive : ∀ key, key ∉ after.keys.val → (after.getCell key).value = (before.getCell key).value

theorem installWords_value (state : Storage) (updates : List WordUpdate) (key : Key) :
    ((state.installWords updates).getCell key).value = (state.getCell key).value := by
  rw [getCell_installWords]

theorem applyEffect_cells (before after : Storage) (effect : WriteEffect)
    (hrun : applyEffect before effect = .ok after) :
    after.keys = before.keys ∧
      (∀ key ∈ after.keys.val, after.getCell key = effectCell effect (before.getCell key)) ∧
      (∀ key, key ∉ after.keys.val → (after.getCell key).value = (before.getCell key).value) := by
  unfold applyEffect at hrun
  split at hrun
  · simp at hrun
  · split at hrun
    · simp at hrun
    · split at hrun
      · rename_i hdisabled
        have henabled : effect.enabled = false := by simpa using hdisabled
        have heq : after = before := by simpa using hrun.symm
        subst after
        exact ⟨rfl, by simp [effectCell, henabled], by simp⟩
      · rename_i henabled
        have henabled : effect.enabled = true := by simpa using henabled
        split at hrun
        · simp at hrun
        · split at hrun
          · have heq : after = (before.write effect.destination effect.payload).installWords effect.words := by
              simpa using hrun.symm
            subst after
            refine ⟨by simp, ?_, ?_⟩
            · intro key hkey
              have hbefore : key ∈ before.keys.val := by simpa using hkey
              rw [getCell_installWords, Storage.getCell_write before effect.destination key effect.payload hbefore]
              simp [effectCell, henabled, modelCell_key]
            · intro key hkey
              rw [installWords_value]
              apply Storage.write_backing_value
              simpa using hkey
          · simp at hrun

theorem registerWrite_inactive (before after : Storage) (destination : Key) (payload : Slice PayloadByte)
    (hrun : before.registerWrite destination payload = some after) (key : Key)
    (hkey : key ∉ after.keys.val) : (after.getCell key).value = (before.getCell key).value := by
  unfold Storage.registerWrite at hrun
  split at hrun
  · split at hrun
    · rename_i hsize
      have heq : after =
          (before.extend destination (Slice.len payload) hsize).write destination payload := by
        simpa using hrun.symm
      subst after
      exact Storage.write_backing_value _ destination key payload hkey
    · simp at hrun
  · simp at hrun

theorem applyChecked_spec (layout : MemoryLayout) (before after : Storage) (write : CheckedWrite)
    (hrun : applyChecked layout before write = .ok after) : CheckedSpec before after write := by
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
        constructor
        · simp [Introduced, CheckedWrite.normalizedEffect] at henabled ⊢
          simp [henabled]
        · simp [effectCell, henabled]
        · simp
      · rename_i henabled
        have henabled : write.normalizedEffect.enabled = true := by simpa using henabled
        have hrawenabled : write.effect.enabled = true := henabled
        split at hrun
        · simp at hrun
        · split at hrun
          · cases hlocation : write.location with
            | «local» thread offset =>
              cases hregister : before.registerWrite write.normalizedEffect.destination write.normalizedEffect.payload with
              | none => simp [hlocation, hregister] at hrun
              | some middle =>
                have heq : after = middle.installWords write.normalizedEffect.words := by
                  simpa [hlocation, hregister] using hrun.symm
                subst after
                constructor
                · intro key
                  simpa [Introduced, IsLocal, hlocation, CheckedWrite.normalizedEffect, hrawenabled] using
                    Storage.registerWrite_keys before middle write.normalizedEffect.destination key
                      write.normalizedEffect.payload hregister
                · intro key hkey
                  have hmiddle : key ∈ middle.keys.val := by simpa using hkey
                  rw [getCell_installWords,
                    Storage.getCell_registerWrite before middle write.normalizedEffect.destination key
                      write.normalizedEffect.payload hregister hmiddle]
                  simp [effectCell, henabled, modelCell_key]
                · intro key hkey
                  rw [installWords_value]
                  exact registerWrite_inactive before middle write.normalizedEffect.destination
                    write.normalizedEffect.payload hregister key (by simpa using hkey)
            | shared cta offset | global cta offset | param cta offset =>
              have heffect := applyEffect_cells before after write.normalizedEffect (by simpa [hlocation] using hrun)
              exact ⟨by intro key; simp [Introduced, IsLocal, hlocation, heffect.1],
                heffect.2.1, heffect.2.2⟩
            | tmem cta lane column byte =>
              have heffect := applyEffect_cells before after write.normalizedEffect (by simpa [hlocation] using hrun)
              exact ⟨by intro key; simp [Introduced, IsLocal, hlocation, heffect.1],
                heffect.2.1, heffect.2.2⟩
          · simp at hrun

theorem checked_byte_effect (layout : MemoryLayout) (before after : Storage) (write : CheckedWrite)
    (hrun : applyChecked layout before write = .ok after) (key : Key)
    (hkey : key ∈ after.keys.val) :
    byteView (after.getCell key) =
      if ByteAffected write.normalizedEffect key then
        let byte := payloadAt write.effect.payload
          (key.offset.val - write.normalizedEffect.destination.offset.val)
        (byte.value, byte.origin, byte.tainted, byte.defined, true)
      else byteView (before.getCell key) := by
  rw [(applyChecked_spec layout before after write hrun).cells key hkey]
  cases henabled : write.effect.enabled <;>
    by_cases ht : Touches write.location.key write.effect.payload.val.length key <;>
    simp [effectCell, CheckedWrite.normalizedEffect, ByteAffected, byteView, modelCell, henabled, ht]

theorem checked_word_effect (layout : MemoryLayout) (before after : Storage) (write : CheckedWrite)
    (hrun : applyChecked layout before write = .ok after) (key : Key)
    (hkey : key ∈ after.keys.val) :
    (after.getCell key).word = if write.effect.enabled then
      postWords write.effect.words key
        (wordAfter (before.getCell key) write.normalizedEffect.destination write.effect.payload)
      else (before.getCell key).word := by
  rw [(applyChecked_spec layout before after write hrun).cells key hkey]
  cases henabled : write.effect.enabled <;>
    by_cases ht : Touches write.location.key write.effect.payload.val.length key <;>
    simp [effectCell, CheckedWrite.normalizedEffect, modelCell, henabled, ht]

theorem checked_word_frame (layout : MemoryLayout) (before after : Storage) (write : CheckedWrite)
    (hrun : applyChecked layout before write = .ok after) (key : Key)
    (hkey : key ∈ after.keys.val) (hframe : ¬WordAffected write.normalizedEffect key) :
    (after.getCell key).word = (before.getCell key).word := by
  rw [(applyChecked_spec layout before after write hrun).cells key hkey]
  exact effectCell_word_frame _ _ (by simpa using hframe)

theorem checked_frame (layout : MemoryLayout) (before after : Storage) (write : CheckedWrite)
    (hrun : applyChecked layout before write = .ok after) (key : Key)
    (hkey : key ∈ before.keys.val) (hframe : ¬Complete write.normalizedEffect key) :
    after.getCell key = before.getCell key := by
  have hs := applyChecked_spec layout before after write hrun
  rw [hs.cells key ((hs.keys key).mpr (Or.inl hkey))]
  exact effectCell_frame _ _ (by simpa using hframe)

theorem checked_byte_frame (layout : MemoryLayout) (before after : Storage) (write : CheckedWrite)
    (hrun : applyChecked layout before write = .ok after) (key : Key)
    (hkey : key ∈ before.keys.val) (hframe : ¬ByteAffected write.normalizedEffect key) :
    byteView (after.getCell key) = byteView (before.getCell key) := by
  have hs := applyChecked_spec layout before after write hrun
  rw [hs.cells key ((hs.keys key).mpr (Or.inl hkey))]
  exact effectCell_byte_frame _ _ (by simpa using hframe)

theorem checked_backing_byte_frame (layout : MemoryLayout) (before after : Storage)
    (write : CheckedWrite) (hrun : applyChecked layout before write = .ok after) (key : Key)
    (hframe : ¬ByteAffected write.normalizedEffect key) :
    (after.getCell key).value = (before.getCell key).value := by
  have hs := applyChecked_spec layout before after write hrun
  by_cases hkey : key ∈ after.keys.val
  · have hview := checked_byte_effect layout before after write hrun key hkey
    rw [ite_eq_right hframe] at hview
    exact congrArg Prod.fst hview
  · exact hs.inactive key hkey

theorem checked_backing_frame (layout : MemoryLayout) (before after : Storage) (write : CheckedWrite)
    (hrun : applyChecked layout before write = .ok after) (key : Key)
    (hframe : ¬Complete write.normalizedEffect key) :
    (after.getCell key).value = (before.getCell key).value := by
  exact checked_backing_byte_frame layout before after write hrun key
    (fun h => hframe (Or.inl h))

end MemoryWrite.Footprint

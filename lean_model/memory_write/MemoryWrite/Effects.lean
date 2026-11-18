/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Storage
import MemoryWrite.Address

open Aeneas Aeneas.Std PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite

inductive WriteFamily where
  | store | vectorStore | packedHalfStore | localStore | tmemStore
  | atomicExchange | atomicCompareExchange | atomicInteger | atomicFloat | atomicPackedHalf
  | bulkCopy | bulkReduction | tmaLoad | tmaStore | tmemCopy | mma
  | clcResponse | allocationResult | remoteStore | nicWrite | nicAtomic | nicCompletion
  | poison
  deriving DecidableEq, Repr

def WriteFamily.allows (family : WriteFamily) (space : U8) : Bool :=
  match family with
  | .localStore => space == 4#u8
  | .tmemStore | .tmemCopy | .mma | .poison => space == 3#u8
  | .clcResponse | .allocationResult | .tmaLoad | .remoteStore => space == 0#u8
  | .nicWrite | .nicAtomic | .nicCompletion | .bulkReduction | .tmaStore => space == 1#u8
  | _ => space == 0#u8 || space == 1#u8

structure WordUpdate where
  key : Key
  history : Option WordHistory

structure WriteEffect where
  family : WriteFamily
  destination : Key
  payload : Slice PayloadByte
  words : List WordUpdate := []
  enabled : Bool := true
  addressDependent : Bool := false
  guardDependent : Bool := false

inductive WriteError where
  | inputDependentAddress | inputDependentGuard | wrongSpace | outOfBounds
  | unreadableSource | missingPending | duplicatePending
  deriving DecidableEq, Repr

def WordSound {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (key : Key) (history : Option WordHistory) : Prop :=
  ∀ id, history = some (.Known id) → ∀ input, sem.domain input → ∀ lane : Fin 4,
    (memory input (location key lane.val)).defined = true ∧
    (memory input (location key lane.val)).bits = sem.word id input lane

theorem opaque_word_sound {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (key : Key) (id : U32) : WordSound sem memory key (some (.Opaque id)) := by
  intro other h
  cases h

def Storage.setWord (state : Storage) (update : WordUpdate) : Storage :=
  { state with words := (update.key, update.history) :: state.words }

@[simp]
theorem Storage.setWord_keys (state : Storage) (update : WordUpdate) :
    (state.setWord update).keys = state.keys := rfl

theorem Storage.getCell_setWord (state : Storage) (update : WordUpdate) (key : Key) :
    (state.setWord update).getCell key =
      { state.getCell key with word :=
        if update.key = key then update.history else (state.getCell key).word } := by
  simp [Storage.setWord, Storage.getCell, Storage.lookup]

theorem Storage.setWord_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (state : Storage) (update : WordUpdate)
    (hstate : Storage.Represents sem memory state)
    (hword : WordSound sem memory update.key update.history) :
    Storage.Represents sem memory (state.setWord update) := by
  intro cell hcell
  simp only [Storage.toCells, Slice.from_val, Storage.setWord_keys, List.mem_map] at hcell
  obtain ⟨key, hkey, rfl⟩ := hcell
  have hbefore := hstate (state.getCell key) (Storage.getCell_mem state key hkey)
  rw [Storage.getCell_setWord]
  constructor
  · exact hbefore.byte
  · intro id hid input hinput lane
    by_cases heq : update.key = key
    · simp only [heq, ite_true] at hid
      simpa only [heq, Storage.getCell_key] using hword id hid input hinput lane
    · simp only [heq, ite_false] at hid
      exact hbefore.word id hid input hinput lane
  · exact hbefore.initial_fallback

def Storage.installWords (state : Storage) (updates : List WordUpdate) : Storage :=
  updates.foldl Storage.setWord state

@[simp]
theorem Storage.installWords_keys (state : Storage) (updates : List WordUpdate) :
    (state.installWords updates).keys = state.keys := by
  induction updates generalizing state with
  | nil => rfl
  | cons update updates ih => simpa [Storage.installWords] using ih (state.setWord update)

theorem Storage.installWords_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (state : Storage) (updates : List WordUpdate)
    (hstate : Storage.Represents sem memory state)
    (hwords : ∀ update ∈ updates, WordSound sem memory update.key update.history) :
    Storage.Represents sem memory (state.installWords updates) := by
  induction updates generalizing state with
  | nil => exact hstate
  | cons update updates ih =>
    apply ih (state.setWord update)
      (Storage.setWord_preserves_representation sem memory state update hstate
        (hwords update (by simp)))
    intro other hother
    exact hwords other (by simp [hother])

def effectMemory {Input : Type} (memory : Memory Input) (effect : WriteEffect)
    (values : Input → Nat → Byte) : Memory Input :=
  if effect.enabled then storeBytes memory effect.destination effect.payload.val.length values
  else memory

def EffectSound {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (effect : WriteEffect) (values : Input → Nat → Byte) : Prop :=
  PayloadSound sem effect.payload values ∧
    ∀ update ∈ effect.words,
      WordSound sem (effectMemory memory effect values) update.key update.history

def applyEffect (state : Storage) (effect : WriteEffect) : Except WriteError Storage :=
  if effect.addressDependent then .error .inputDependentAddress
  else if effect.guardDependent then .error .inputDependentGuard
  else if !effect.enabled then .ok state
  else if !effect.family.allows effect.destination.space then .error .wrongSpace
  else if Storage.SpanFits state effect.destination (Slice.len effect.payload) then
    .ok ((state.write effect.destination effect.payload).installWords effect.words)
  else .error .outOfBounds

theorem applyEffect_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (before after : Storage) (effect : WriteEffect)
    (values : Input → Nat → Byte) (hstate : Storage.Represents sem memory before)
    (hsound : EffectSound sem memory effect values) (hrun : applyEffect before effect = .ok after) :
    Storage.Represents sem (effectMemory memory effect values) after ∧ after.keys = before.keys := by
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
        exact ⟨by simpa [effectMemory, henabled] using hstate, rfl⟩
      · rename_i henabled
        have henabled : effect.enabled = true := by simpa using henabled
        split at hrun
        · simp at hrun
        · split at hrun
          · rename_i hspan
            have heq : after =
                (before.write effect.destination effect.payload).installWords effect.words := by
              simpa using hrun.symm
            subst after
            have hw := Storage.write_preserves_representation sem memory before effect.destination
              effect.payload values hstate hspan hsound.1
            refine ⟨?_, by simp⟩
            apply Storage.installWords_preserves_representation
            · simpa [effectMemory, henabled] using hw.1
            · exact hsound.2
          · simp at hrun

def prepareCopy (state : Storage) (family : WriteFamily) (destination source : Key)
    (width : Usize) : Except WriteError WriteEffect :=
  match state.readSpan source width with
  | none => .error .unreadableSource
  | some cells => .ok { family := family, destination := destination, payload := snapshotPayload cells }

theorem prepareCopy_sound {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (state : Storage) (family : WriteFamily) (destination source : Key) (width : Usize)
    (effect : WriteEffect) (hstate : Storage.Represents sem memory state)
    (hrun : prepareCopy state family destination source width = .ok effect) :
    EffectSound sem memory effect (snapshotValues memory source) ∧
      effect.destination = destination ∧ effect.payload.val.length = width.val := by
  unfold prepareCopy at hrun
  cases hread : state.readSpan source width with
  | none => simp [hread] at hrun
  | some cells =>
    simp only [hread, Except.ok.injEq] at hrun
    subst effect
    obtain ⟨hlength, hcontiguous, hsource⟩ :=
      Storage.readSpan_sound sem memory state source width cells hstate hread
    exact ⟨⟨snapshotPayload_sound sem memory cells source hsource hcontiguous,
      by simp⟩, rfl, by simpa using hlength⟩

structure DescribedEffect (Input : Type) where
  effect : WriteEffect
  values : Input → Nat → Byte

def runEffects (state : Storage) : List WriteEffect → Except WriteError Storage
  | [] => .ok state
  | effect :: effects =>
      match applyEffect state effect with
      | .error error => .error error
      | .ok next => runEffects next effects

def effectsMemory {Input : Type} (memory : Memory Input) : List (DescribedEffect Input) → Memory Input
  | [] => memory
  | description :: descriptions =>
    effectsMemory (effectMemory memory description.effect description.values) descriptions

def EffectsSound {Input : Type} (sem : Semantics Input) (memory : Memory Input) :
    List (DescribedEffect Input) → Prop
  | [] => True
  | description :: descriptions => EffectSound sem memory description.effect description.values ∧
    EffectsSound sem (effectMemory memory description.effect description.values) descriptions

theorem runEffects_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (before after : Storage) (descriptions : List (DescribedEffect Input))
    (hstate : Storage.Represents sem memory before) (hsound : EffectsSound sem memory descriptions)
    (hrun : runEffects before (descriptions.map DescribedEffect.effect) = .ok after) :
    Storage.Represents sem (effectsMemory memory descriptions) after ∧ after.keys = before.keys := by
  induction descriptions generalizing memory before with
  | nil =>
    have heq : after = before := by simpa [runEffects] using hrun.symm
    subst after
    exact ⟨hstate, rfl⟩
  | cons description descriptions ih =>
    cases hstep : applyEffect before description.effect with
    | error error => simp [runEffects, hstep] at hrun
    | ok middle =>
      have hm := applyEffect_preserves_representation sem memory before middle
        description.effect description.values hstate hsound.1 hstep
      have hrest : runEffects middle (descriptions.map DescribedEffect.effect) = .ok after := by
        simpa [runEffects, hstep] using hrun
      have hfinish := ih _ middle hm.1 hsound.2 hrest
      exact ⟨hfinish.1, hfinish.2.trans hm.2⟩

structure PendingWrite where
  ticket : Nat
  effects : List WriteEffect

structure MemoryMachine where
  storage : Storage
  pending : List PendingWrite := []
  nextTicket : Nat := 0

def issueEffects (machine : MemoryMachine) (effects : List WriteEffect) : MemoryMachine × Nat :=
  ({ machine with
      pending := machine.pending ++ [⟨machine.nextTicket, effects⟩]
      nextTicket := machine.nextTicket + 1 }, machine.nextTicket)

def issueCopy (machine : MemoryMachine) (family : WriteFamily) (destination source : Key)
    (width : Usize) : Except WriteError (MemoryMachine × Nat) := do
  let effect ← prepareCopy machine.storage family destination source width
  .ok (issueEffects machine [effect])

def complete (machine : MemoryMachine) (ticket : Nat) : Except WriteError MemoryMachine :=
  match machine.pending.find? (fun pending => pending.ticket == ticket) with
  | none => .error .missingPending
  | some pending =>
    match runEffects machine.storage pending.effects with
    | .error error => .error error
    | .ok storage => .ok { machine with
        storage := storage
        pending := machine.pending.filter (fun pending => pending.ticket != ticket) }

@[simp]
theorem issueEffects_storage (machine : MemoryMachine) (effects : List WriteEffect) :
    (issueEffects machine effects).1.storage = machine.storage := rfl

theorem complete_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (before after : MemoryMachine) (ticket : Nat)
    (pending : PendingWrite) (descriptions : List (DescribedEffect Input))
    (hfind : before.pending.find? (fun pending => pending.ticket == ticket) = some pending)
    (heffects : pending.effects = descriptions.map DescribedEffect.effect)
    (hstate : Storage.Represents sem memory before.storage)
    (hsound : EffectsSound sem memory descriptions) (hrun : complete before ticket = .ok after) :
    Storage.Represents sem (effectsMemory memory descriptions) after.storage ∧
      after.storage.keys = before.storage.keys := by
  unfold complete at hrun
  simp only [hfind] at hrun
  rw [heffects] at hrun
  cases hstep : runEffects before.storage (descriptions.map DescribedEffect.effect) with
  | error error => simp [hstep] at hrun
  | ok storage =>
    have heq : after = { before with
        storage := storage
        pending := before.pending.filter (fun pending => pending.ticket != ticket) } := by
      simpa [hstep] using hrun.symm
    subst after
    exact runEffects_preserves_representation sem memory before.storage storage descriptions
      hstate hsound hstep

end MemoryWrite

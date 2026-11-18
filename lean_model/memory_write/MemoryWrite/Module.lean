/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Production
import MemoryWrite.Transfers

open Aeneas Aeneas.Std PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite

structure CheckedTransfer where
  family : WriteFamily
  destination : MemoryLocation
  source : Option MemoryLocation
  width : Usize

inductive PendingCheckedOperation where
  | captured (requests : List CheckedWrite)
  | deferred (transfers : List CheckedTransfer)

structure PendingChecked where
  ticket : Nat
  operation : PendingCheckedOperation

structure CentralState where
  storage : Storage
  markers : WrittenMarkers := {}
  tracking : WrittenTracking := {}
  pending : List PendingChecked := []
  nextTicket : Nat := 0

def write (layout : MemoryLayout) (state : CentralState) (request : CheckedWrite) :
    Except WriteError CentralState :=
  match applyChecked layout state.storage request with
  | .error error => .error error
  | .ok storage => .ok { state with
      storage := storage
      markers := state.markers.record state.tracking request.normalizedEffect }

theorem write_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (layout : MemoryLayout) (before after : CentralState)
    (request : CheckedWrite) (values : Input → Nat → Byte)
    (hstate : Storage.Represents sem memory before.storage)
    (hpayload : EffectSound sem memory request.normalizedEffect values)
    (hrun : write layout before request = .ok after) :
    Storage.Represents sem (effectMemory memory request.normalizedEffect values) after.storage ∧
      after.markers = before.markers.record before.tracking request.normalizedEffect ∧
      after.tracking = before.tracking := by
  unfold write at hrun
  cases hstep : applyChecked layout before.storage request with
  | error error => simp [hstep] at hrun
  | ok storage =>
    have heq : after = { before with
        storage := storage
        markers := before.markers.record before.tracking request.normalizedEffect } := by
      simpa [hstep] using hrun.symm
    subst after
    exact ⟨applyChecked_preserves_representation sem memory layout before.storage storage
      request values hstate hpayload hstep, rfl, rfl⟩

structure CentralAtomicResult where
  state : CentralState
  old : Slice PayloadByte
  exchanged : Bool

def compareExchange (layout : MemoryLayout) (state : CentralState)
    (destination : MemoryLocation) (expected : Slice U8) (replacement : Slice PayloadByte)
    (compareDependent : Bool := false) : Except WriteError CentralAtomicResult :=
  if WriteValid layout destination replacement.val.length then
    match Transfers.compareExchange state.storage destination.key expected replacement compareDependent with
    | .error error => .error error
    | .ok result =>
      let effect : WriteEffect := {
        family := .atomicCompareExchange, destination := destination.key, payload := replacement }
      .ok ⟨{ state with
        storage := result.storage
        markers := if result.exchanged then state.markers.record state.tracking effect
          else state.markers.recordNumeric state.tracking destination.key }, result.old, result.exchanged⟩
  else .error .outOfBounds

theorem compareExchange_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (layout : MemoryLayout) (state : CentralState)
    (destination : MemoryLocation) (expected : Slice U8) (replacement : Slice PayloadByte)
    (values : Input → Nat → Byte) (result : CentralAtomicResult) (compareDependent : Bool)
    (hstate : Storage.Represents sem memory state.storage)
    (hpayload : PayloadSound sem replacement values)
    (hrun : compareExchange layout state destination expected replacement compareDependent = .ok result)
    (hstable : ∀ input, sem.domain input → Transfers.memoryMatches memory input destination.key
      replacement.val.length expected = result.exchanged) :
    Storage.Represents sem
      (Transfers.compareExchangeMemory memory destination.key replacement expected values)
      result.state.storage ∧ PayloadSound sem result.old (snapshotValues memory destination.key) ∧
      result.old.val.length = replacement.val.length := by
  unfold compareExchange at hrun
  split at hrun
  · cases hstep : Transfers.compareExchange state.storage destination.key expected replacement compareDependent with
    | error error => simp [hstep] at hrun
    | ok atomic =>
      have heq : result = ⟨{ state with
          storage := atomic.storage
          markers := if atomic.exchanged then
              state.markers.record state.tracking {
                family := .atomicCompareExchange, destination := destination.key, payload := replacement }
            else state.markers.recordNumeric state.tracking destination.key }, atomic.old, atomic.exchanged⟩ := by
        simpa [hstep] using hrun.symm
      subst result
      have h := Transfers.compareExchange_semantic_preservation sem memory state.storage
        destination.key expected replacement values atomic compareDependent hstate hpayload hstep hstable
      exact ⟨h.1, h.2.1, h.2.2.1⟩
  · simp at hrun

structure DescribedChecked (Input : Type) where
  request : CheckedWrite
  values : Input → Nat → Byte

def normalizedDescriptions {Input : Type} (descriptions : List (DescribedChecked Input)) :
    List (DescribedEffect Input) :=
  descriptions.map fun description => ⟨description.request.normalizedEffect, description.values⟩

def runChecked (layout : MemoryLayout) : CentralState → List CheckedWrite → Except WriteError CentralState
  | state, [] => .ok state
  | state, request :: requests => do
      let next ← write layout state request
      runChecked layout next requests

def recordChecked (markers : WrittenMarkers) (tracking : WrittenTracking) (requests : List CheckedWrite) :
    WrittenMarkers :=
  requests.foldl (fun markers request => markers.record tracking request.normalizedEffect) markers

theorem write_markers (layout : MemoryLayout) (before after : CentralState) (request : CheckedWrite)
    (hrun : write layout before request = .ok after) :
    after.markers = before.markers.record before.tracking request.normalizedEffect ∧
      after.tracking = before.tracking := by
  unfold write at hrun
  cases hstep : applyChecked layout before.storage request with
  | error error => simp [hstep] at hrun
  | ok storage =>
    have heq : after = { before with
        storage := storage
        markers := before.markers.record before.tracking request.normalizedEffect } := by
      simpa [hstep] using hrun.symm
    subst after
    exact ⟨rfl, rfl⟩

theorem runChecked_markers (layout : MemoryLayout) (before after : CentralState)
    (requests : List CheckedWrite) (hrun : runChecked layout before requests = .ok after) :
    after.markers = recordChecked before.markers before.tracking requests ∧
      after.tracking = before.tracking := by
  induction requests generalizing before with
  | nil =>
    have heq : after = before := by simpa [runChecked] using hrun.symm
    subst after
    exact ⟨rfl, rfl⟩
  | cons request requests ih =>
    cases hstep : write layout before request with
    | error error => simp [runChecked, hstep, Bind.bind, Except.bind] at hrun
    | ok next =>
      have hrest : runChecked layout next requests = .ok after := by
        simpa [runChecked, hstep, Bind.bind, Except.bind] using hrun
      have hfirst := write_markers layout before next request hstep
      have hfinish := ih next hrest
      exact ⟨by simpa [recordChecked, hfirst.1, hfirst.2] using hfinish.1,
        hfinish.2.trans hfirst.2⟩

theorem runChecked_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (layout : MemoryLayout) (before after : CentralState)
    (descriptions : List (DescribedChecked Input))
    (hstate : Storage.Represents sem memory before.storage)
    (hsound : EffectsSound sem memory (normalizedDescriptions descriptions))
    (hrun : runChecked layout before (descriptions.map DescribedChecked.request) = .ok after) :
    Storage.Represents sem (effectsMemory memory (normalizedDescriptions descriptions)) after.storage := by
  induction descriptions generalizing memory before with
  | nil =>
    have heq : after = before := by simpa [runChecked] using hrun.symm
    subst after
    exact hstate
  | cons description descriptions ih =>
    cases hstep : write layout before description.request with
    | error error => simp [runChecked, hstep, Bind.bind, Except.bind] at hrun
    | ok next =>
      have hfirst := write_preserves_representation sem memory layout before next
        description.request description.values hstate hsound.1 hstep
      have hrest : runChecked layout next (descriptions.map DescribedChecked.request) = .ok after := by
        simpa [runChecked, hstep, Bind.bind, Except.bind] using hrun
      exact ih _ next hfirst.1 hsound.2 hrest

structure CapturedCheckedTransfer where
  request : CheckedWrite
  source : Option MemoryLocation

def CapturedCheckedTransfer.values {Input : Type} (memory : Memory Input)
    (captured : CapturedCheckedTransfer) : Input → Nat → Byte :=
  match captured.source with
  | none => Transfers.zeroValues
  | some source => snapshotValues memory source.key

def CapturedCheckedTransfer.Sound {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (captured : CapturedCheckedTransfer) : Prop :=
  PayloadSound sem captured.request.normalizedEffect.payload (captured.values memory) ∧
    captured.request.normalizedEffect.words = []

def captureTransfer (layout : MemoryLayout) (state : Storage) (transfer : CheckedTransfer) :
    Except WriteError CapturedCheckedTransfer :=
  match transfer.source with
  | none => .ok ⟨⟨transfer.destination, {
      family := transfer.family
      destination := transfer.destination.key
      payload := Transfers.zeroPayload transfer.width }⟩, none⟩
  | some source => do
      let cells ← readChecked layout state source transfer.width
      .ok ⟨⟨transfer.destination, {
        family := transfer.family
        destination := transfer.destination.key
        payload := snapshotPayload cells }⟩, some source⟩

theorem captureTransfer_sound {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (layout : MemoryLayout) (state : Storage) (transfer : CheckedTransfer)
    (captured : CapturedCheckedTransfer) (hstate : Storage.Represents sem memory state)
    (hrun : captureTransfer layout state transfer = .ok captured) :
    captured.Sound sem memory ∧ captured.request.location = transfer.destination ∧
      captured.request.effect.payload.val.length = transfer.width.val ∧
      captured.source = transfer.source := by
  cases hsource : transfer.source with
  | none =>
    simp only [captureTransfer, hsource, Except.ok.injEq] at hrun
    subst captured
    exact ⟨⟨Transfers.zeroPayload_sound sem transfer.width, rfl⟩,
      rfl, Transfers.zeroPayload_length _, rfl⟩
  | some source =>
    cases hread : readChecked layout state source transfer.width with
    | error error => simp [captureTransfer, hsource, hread, Bind.bind, Except.bind] at hrun
    | ok cells =>
      have heq : captured = ⟨⟨transfer.destination, {
          family := transfer.family
          destination := transfer.destination.key
          payload := snapshotPayload cells }⟩, some source⟩ := by
        simpa [captureTransfer, hsource, hread, Bind.bind, Except.bind] using hrun.symm
      subst captured
      have hs := readChecked_sound sem memory layout state source transfer.width cells hstate hread
      exact ⟨⟨snapshotPayload_sound sem memory cells source.key hs.2.2.2 hs.2.2.1, rfl⟩,
        rfl, by simpa using hs.2.1, rfl⟩

def captureTransfers (layout : MemoryLayout) (state : Storage) :
    List CheckedTransfer → Except WriteError (List CapturedCheckedTransfer)
  | [] => .ok []
  | transfer :: transfers => do
      let captured ← captureTransfer layout state transfer
      let rest ← captureTransfers layout state transfers
      .ok (captured :: rest)

theorem captureTransfers_sound {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (layout : MemoryLayout) (state : Storage) (transfers : List CheckedTransfer)
    (captured : List CapturedCheckedTransfer) (hstate : Storage.Represents sem memory state)
    (hrun : captureTransfers layout state transfers = .ok captured) :
    ∀ item ∈ captured, item.Sound sem memory := by
  induction transfers generalizing captured with
  | nil => simp [captureTransfers] at hrun; subst captured; simp
  | cons transfer transfers ih =>
    cases hfirst : captureTransfer layout state transfer with
    | error error => simp [captureTransfers, hfirst, Bind.bind, Except.bind] at hrun
    | ok first =>
      cases hrest : captureTransfers layout state transfers with
      | error error => simp [captureTransfers, hfirst, hrest, Bind.bind, Except.bind] at hrun
      | ok rest =>
        have heq : captured = first :: rest := by
          simpa [captureTransfers, hfirst, hrest, Bind.bind, Except.bind] using hrun.symm
        subst captured
        intro item hitem
        rcases List.mem_cons.mp hitem with heq | hrestmem
        · subst item
          exact (captureTransfer_sound sem memory layout state transfer first hstate hfirst).1
        · exact ih rest hrest item hrestmem

def capturedDescriptions {Input : Type} (memory : Memory Input)
    (captured : List CapturedCheckedTransfer) : List (DescribedChecked Input) :=
  captured.map fun item => ⟨item.request, item.values memory⟩

theorem capturedDescriptions_sound {Input : Type} (sem : Semantics Input)
    (sourceMemory destinationMemory : Memory Input) (captured : List CapturedCheckedTransfer)
    (hsound : ∀ item ∈ captured, item.Sound sem sourceMemory) :
    EffectsSound sem destinationMemory (normalizedDescriptions (capturedDescriptions sourceMemory captured)) := by
  induction captured generalizing destinationMemory with
  | nil => trivial
  | cons item captured ih =>
    have hfirst := hsound item (by simp)
    refine ⟨⟨hfirst.1, ?_⟩, ?_⟩
    · simp [hfirst.2]
    · exact ih _ (fun other hother => hsound other (by simp [hother]))

def preparePendingChecked (layout : MemoryLayout) (state : Storage) :
    PendingCheckedOperation → Except WriteError (List CheckedWrite)
  | .captured requests => .ok requests
  | .deferred transfers => do
      let captured ← captureTransfers layout state transfers
      .ok (captured.map CapturedCheckedTransfer.request)

def issueChecked (state : CentralState) (requests : List CheckedWrite) : CentralState × Nat :=
  ({ state with
      pending := state.pending ++ [⟨state.nextTicket, .captured requests⟩]
      nextTicket := state.nextTicket + 1 }, state.nextTicket)

def issueTransfer (layout : MemoryLayout) (state : CentralState) (transfers : List CheckedTransfer) :
    Except WriteError (CentralState × Nat) := do
  let captured ← captureTransfers layout state.storage transfers
  .ok (issueChecked state (captured.map CapturedCheckedTransfer.request))

def issueDeferredTransfer (state : CentralState) (transfers : List CheckedTransfer) : CentralState × Nat :=
  ({ state with
      pending := state.pending ++ [⟨state.nextTicket, .deferred transfers⟩]
      nextTicket := state.nextTicket + 1 }, state.nextTicket)

def finishChecked (layout : MemoryLayout) (state : CentralState) (ticket : Nat) :
    Except WriteError CentralState := do
  let pending ← match state.pending.find? (fun pending => pending.ticket == ticket) with
    | none => .error .missingPending
    | some pending => .ok pending
  let requests ← preparePendingChecked layout state.storage pending.operation
  let next ← runChecked layout state requests
  .ok { next with pending := state.pending.filter (fun pending => pending.ticket != ticket) }

@[simp]
theorem issueChecked_storage (state : CentralState) (requests : List CheckedWrite) :
    (issueChecked state requests).1.storage = state.storage := rfl

@[simp]
theorem issueChecked_markers (state : CentralState) (requests : List CheckedWrite) :
    (issueChecked state requests).1.markers = state.markers := rfl

theorem finishChecked_markers (layout : MemoryLayout) (before after : CentralState)
    (ticket : Nat) (pending : PendingChecked) (requests : List CheckedWrite)
    (hfind : before.pending.find? (fun pending => pending.ticket == ticket) = some pending)
    (hprepare : preparePendingChecked layout before.storage pending.operation = .ok requests)
    (hrun : finishChecked layout before ticket = .ok after) :
    after.markers = recordChecked before.markers before.tracking requests ∧
      after.tracking = before.tracking ∧
      after.pending = before.pending.filter (fun pending => pending.ticket != ticket) := by
  unfold finishChecked at hrun
  simp only [hfind, hprepare, Bind.bind, Except.bind] at hrun
  cases hstep : runChecked layout before requests with
  | error error => simp [hstep] at hrun
  | ok next =>
    have heq : after = { next with pending := before.pending.filter (fun pending => pending.ticket != ticket) } := by
      simpa [hstep] using hrun.symm
    subst after
    have hmarkers := runChecked_markers layout before next requests hstep
    exact ⟨hmarkers.1, hmarkers.2, rfl⟩

theorem finishChecked_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (layout : MemoryLayout) (before after : CentralState)
    (ticket : Nat) (pending : PendingChecked) (descriptions : List (DescribedChecked Input))
    (hfind : before.pending.find? (fun pending => pending.ticket == ticket) = some pending)
    (hprepare : preparePendingChecked layout before.storage pending.operation =
      .ok (descriptions.map DescribedChecked.request))
    (hstate : Storage.Represents sem memory before.storage)
    (hsound : EffectsSound sem memory (normalizedDescriptions descriptions))
    (hrun : finishChecked layout before ticket = .ok after) :
    Storage.Represents sem (effectsMemory memory (normalizedDescriptions descriptions)) after.storage ∧
      after.markers = recordChecked before.markers before.tracking
        (descriptions.map DescribedChecked.request) ∧
      after.tracking = before.tracking ∧
      after.pending = before.pending.filter (fun pending => pending.ticket != ticket) := by
  unfold finishChecked at hrun
  simp only [hfind, hprepare, Bind.bind, Except.bind] at hrun
  cases hstep : runChecked layout before (descriptions.map DescribedChecked.request) with
  | error error => simp [hstep] at hrun
  | ok next =>
    have heq : after = { next with pending := before.pending.filter (fun pending => pending.ticket != ticket) } := by
      simpa [hstep] using hrun.symm
    subst after
    have hmarkers := runChecked_markers layout before next (descriptions.map DescribedChecked.request) hstep
    exact ⟨runChecked_preserves_representation sem memory layout before next descriptions hstate hsound hstep,
      hmarkers.1, hmarkers.2, rfl⟩

theorem finishCapturedTransfer_preserves_representation {Input : Type} (sem : Semantics Input)
    (sourceMemory destinationMemory : Memory Input) (sourceLayout layout : MemoryLayout)
    (sourceState : Storage) (before after : CentralState) (ticket : Nat) (pending : PendingChecked)
    (transfers : List CheckedTransfer) (captured : List CapturedCheckedTransfer)
    (hsource : Storage.Represents sem sourceMemory sourceState)
    (hcapture : captureTransfers sourceLayout sourceState transfers = .ok captured)
    (hfind : before.pending.find? (fun pending => pending.ticket == ticket) = some pending)
    (hoperation : pending.operation = .captured (captured.map CapturedCheckedTransfer.request))
    (hstate : Storage.Represents sem destinationMemory before.storage)
    (hrun : finishChecked layout before ticket = .ok after) :
    Storage.Represents sem
      (effectsMemory destinationMemory (normalizedDescriptions (capturedDescriptions sourceMemory captured)))
      after.storage ∧
      after.markers = recordChecked before.markers before.tracking
        (captured.map CapturedCheckedTransfer.request) := by
  have hrequests : (capturedDescriptions sourceMemory captured).map DescribedChecked.request =
      captured.map CapturedCheckedTransfer.request := by
    simp [capturedDescriptions, List.map_map, Function.comp_def]
  have hprepare : preparePendingChecked layout before.storage pending.operation =
      .ok ((capturedDescriptions sourceMemory captured).map DescribedChecked.request) := by
    rw [hrequests]
    simp [preparePendingChecked, hoperation]
  have h := finishChecked_preserves_representation sem destinationMemory layout before after
    ticket pending (capturedDescriptions sourceMemory captured) hfind hprepare hstate
    (capturedDescriptions_sound sem sourceMemory destinationMemory captured
      (captureTransfers_sound sem sourceMemory sourceLayout sourceState transfers captured hsource hcapture)) hrun
  exact ⟨h.1, by simpa only [hrequests] using h.2.1⟩

theorem finishDeferredTransfer_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (layout : MemoryLayout) (before after : CentralState)
    (ticket : Nat) (pending : PendingChecked) (transfers : List CheckedTransfer)
    (hfind : before.pending.find? (fun pending => pending.ticket == ticket) = some pending)
    (hoperation : pending.operation = .deferred transfers)
    (hstate : Storage.Represents sem memory before.storage)
    (hrun : finishChecked layout before ticket = .ok after) :
    ∃ captured, captureTransfers layout before.storage transfers = .ok captured ∧
      Storage.Represents sem
        (effectsMemory memory (normalizedDescriptions (capturedDescriptions memory captured))) after.storage ∧
      after.markers = recordChecked before.markers before.tracking
        (captured.map CapturedCheckedTransfer.request) := by
  cases hcapture : captureTransfers layout before.storage transfers with
  | error error =>
    simp [finishChecked, hfind, preparePendingChecked, hoperation, hcapture, Bind.bind, Except.bind] at hrun
  | ok captured =>
    have hrequests : (capturedDescriptions memory captured).map DescribedChecked.request =
        captured.map CapturedCheckedTransfer.request := by
      simp [capturedDescriptions, List.map_map, Function.comp_def]
    have hprepare : preparePendingChecked layout before.storage pending.operation =
        .ok ((capturedDescriptions memory captured).map DescribedChecked.request) := by
      rw [hrequests]
      simp [preparePendingChecked, hoperation, hcapture, Bind.bind, Except.bind]
    have h := finishChecked_preserves_representation sem memory layout before after ticket pending
      (capturedDescriptions memory captured) hfind hprepare hstate
      (capturedDescriptions_sound sem memory memory captured
        (captureTransfers_sound sem memory layout before.storage transfers captured hstate hcapture)) hrun
    exact ⟨captured, rfl, h.1, by simpa only [hrequests] using h.2.1⟩

end MemoryWrite

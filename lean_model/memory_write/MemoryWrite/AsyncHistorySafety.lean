/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RustCapture
import MemoryWrite.RustCopies
import MemoryWrite.Consumers

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite

structure CopiedCellFrom (captured before after : Cell) (destination : Key) (width : Nat) : Prop where
  key : after.key = before.key
  value : after.value = captured.value
  origin : after.origin = captured.origin
  taint : after.tainted = captured.tainted
  defined : after.defined = captured.defined
  written : after.written = true
  history : after.word =
    if before.key.offset.val + 4 ≤ destination.offset.val + width then captured.word else none

theorem modelCell_copies_frozen_history (before captured : Cell) (destination : Key)
    (payload : Slice PayloadByte) (width : Nat) (hwidth : payload.val.length = width)
    (htouch : Touches destination width before.key)
    (hbyte : payloadAt payload (before.key.offset.val - destination.offset.val) =
      snapshotByte captured) :
    CopiedCellFrom captured before (modelCell before destination payload) destination width := by
  have ht : Touches destination payload.val.length before.key := by simpa [hwidth] using htouch
  have hoverlap : Overlaps destination payload.val.length before.key := by
    obtain ⟨hregion, hstart, hend⟩ := htouch
    exact ⟨hregion, by omega, by omega, by omega⟩
  have hcover : CoversWord destination payload.val.length before.key ↔
      before.key.offset.val + 4 ≤ destination.offset.val + width := by
    simp only [CoversWord, hwidth]
    exact ⟨fun h => h.2.2, fun h => ⟨htouch.1, htouch.2.1, h⟩⟩
  constructor <;> simp [modelCell, ht, hbyte, snapshotByte, wordAfter, hcover, hoverlap]

theorem rust_capture_freezes_every_history (issue : Slice Cell) (source : Key) (width : Usize)
    (hspan : ValidWriteSpan issue source width.val) :
    ∃ payload : alloc.vec.Vec PayloadByte,
      checked.capture_span issue source width = .ok (some payload) ∧
      payload.val.length = width.val ∧
      ∀ index, index < width.val → ∃ captured,
        findCellModel issue (Storage.offsetKey source index) = some captured ∧
        payloadAt payload.slice index = snapshotByte captured := by
  obtain ⟨result, hrun, hspec⟩ := spec_returns _ _ (rust_capture_span_spec issue source width)
  obtain ⟨cells, payload, hresult, hsource, hpayload⟩ := hspec.1 hspan
  subst result
  have hslice : payload.slice = snapshotPayload cells := by
    apply Slice.ext
    simpa only [snapshotPayload, Slice.from_val, alloc.vec.Vec.val] using hpayload
  refine ⟨payload, hrun, by simp [hpayload, hsource.1], ?_⟩
  intro index hindex
  have hi : index < cells.val.length := by rw [hsource.1]; exact hindex
  exact ⟨cells.val[index], hsource.2 index hi, by
    rw [hslice, snapshotPayload_at cells index hi]⟩

theorem rust_async_copy_preserves_issue_histories (issue completion : Slice Cell)
    (source destination : Key) (width : Usize)
    (hspan : ValidWriteSpan issue source width.val) :
    ∃ payload : alloc.vec.Vec PayloadByte, ∃ next : Slice Cell,
      checked.capture_span issue source width = .ok (some payload) ∧
      apply_write_effect completion destination payload.slice = .ok next ∧
      payload.val.length = width.val ∧
      (∀ before ∈ completion.val, Touches destination width.val before.key →
        ∃ captured after,
          findCellModel issue (Storage.offsetKey source
            (before.key.offset.val - destination.offset.val)) = some captured ∧
          after ∈ next.val ∧ CopiedCellFrom captured before after destination width.val) := by
  obtain ⟨payload, hcapture, hlength, hfrozen⟩ :=
    rust_capture_freezes_every_history issue source width hspan
  obtain ⟨next, hrun, hnext⟩ := spec_returns _ _
    (apply_write_effect_spec completion destination payload.slice)
  refine ⟨payload, next, hcapture, hrun, hlength, ?_⟩
  intro before hbefore htouch
  have hindex : before.key.offset.val - destination.offset.val < width.val := by
    obtain ⟨_, hstart, hend⟩ := htouch
    omega
  obtain ⟨captured, hread, hbyte⟩ := hfrozen _ hindex
  refine ⟨captured, modelCell before destination payload.slice, hread, ?_, ?_⟩
  · rw [hnext]
    exact List.mem_map.mpr ⟨before, hbefore, rfl⟩
  · exact modelCell_copies_frozen_history before captured destination payload.slice width.val
      hlength htouch hbyte

theorem copiedMemory_at {Input : Type} (issue completion : Memory Input)
    (source destination : Key) (width index : Nat) (hindex : index < width) (input : Input) :
    storeBytes completion destination width (snapshotValues issue source)
      input (location destination index) = issue input (location source index) := by
  have ht : InWrite destination width (location destination index) := by
    exact ⟨rfl, rfl, by simp [location], by simp [location]; omega⟩
  simp only [storeBytes, ite_eq_left ht]
  simp [location, snapshotValues]

structure CopyConsumersUseIssue {Input : Type} (sem : Semantics Input)
    (issue : Memory Input) (source destination : Key) (width : Nat) (cells : Slice Cell) : Prop where
  origin : ∀ cell ∈ cells.val, ∀ index, index < width →
    location cell.key = location destination index → ∀ initialOrigin,
    Consumers.InitialOriginsSound sem initialOrigin → ∀ origin,
    Consumers.originWithFallback cell initialOrigin = some origin →
    ∀ input, sem.domain input →
      (issue input (location source index)).defined = true ∧
      (issue input (location source index)).bits = sem.origin origin input
  word : ∀ cell ∈ cells.val, ∀ index, index + 4 ≤ width →
    location cell.key = location destination index → ∀ id,
    cell.word = some (.Known id) → ∀ input, sem.domain input → ∀ lane : Fin 4,
      (issue input (location source (index + lane.val))).defined = true ∧
      (issue input (location source (index + lane.val))).bits = sem.word id input lane
  constant : ∀ cell ∈ cells.val, ∀ index, index < width →
    location cell.key = location destination index → cell.tainted = false →
    ∀ input, sem.domain input → (issue input (location source index)).bits = cell.value

theorem represented_copy_consumers_use_issue {Input : Type} (sem : Semantics Input)
    (issue completion : Memory Input) (source destination : Key) (width : Nat) (cells : Slice Cell)
    (hrep : Represents sem
      (storeBytes completion destination width (snapshotValues issue source)) cells) :
    CopyConsumersUseIssue sem issue source destination width cells := by
  constructor
  · intro cell hcell index hindex hlocation initialOrigin hinitial origin hlookup input hinput
    have hsound := Consumers.originWithFallback_sound sem _ cell initialOrigin
      (hrep cell hcell) hinitial origin hlookup input hinput
    rw [hlocation, copiedMemory_at issue completion source destination width index hindex input]
      at hsound
    exact hsound
  · intro cell hcell index hindex hlocation id hid input hinput lane
    have hsound := (hrep cell hcell).word id hid input hinput lane
    have hloc := location_lane_of_eq cell.key destination index lane.val hlocation
    rw [hloc, copiedMemory_at issue completion source destination width (index + lane.val)
      (by omega) input] at hsound
    exact hsound
  · intro cell hcell index hindex hlocation htaint input hinput
    have hdatum := (hrep cell hcell).byte
    have hsound := (hdatum.constant htaint input hinput).trans hdatum.captured.symm
    rw [hlocation, copiedMemory_at issue completion source destination width index hindex input]
      at hsound
    exact hsound

theorem rust_async_copy_consumers_use_issue {Input : Type} (sem : Semantics Input)
    (issueMemory completionMemory : Memory Input) (issue : Slice Cell) (source : Key)
    (width : Usize) (before : Storage) (state : engine.Engine) (template : checked.PreparedWrite)
    (hsource : Represents sem issueMemory issue)
    (hsourceSpan : ValidWriteSpan issue source width.val)
    (hstate : Storage.Represents sem completionMemory before)
    (hcells : state.cells.slice = before.toCells)
    (hnumeric : state.numeric_written.val.length + width.val ≤ Usize.max)
    (hhalf : state.half_written.val.length + width.val ≤ Usize.max)
    (haddress : template.address_dependent = false) (hguard : template.guard_dependent = false)
    (henabled : template.enabled = true)
    (hfamily : (rustFamily template.class).allows template.destination.space = true)
    (hdestination : Storage.SpanFits before template.destination width) :
    ∃ payload next,
      checked.capture_span issue source width = .ok (some payload) ∧
      engine.apply state { template with payload := payload, words := alloc.vec.Vec.new _ } =
        .ok (.Ok (), next) ∧
      CopyConsumersUseIssue sem issueMemory source template.destination width.val next.cells.slice := by
  obtain ⟨payload, next, hcapture, hrun, _, hrep⟩ :=
    rust_copy_progress_and_preserves_representation sem issueMemory completionMemory issue source
      width before state template hsource hsourceSpan hstate hcells hnumeric hhalf haddress hguard
      henabled hfamily hdestination
  exact ⟨payload, next, hcapture, hrun,
    represented_copy_consumers_use_issue sem issueMemory completionMemory source
      template.destination width.val next.cells.slice hrep⟩

end MemoryWrite

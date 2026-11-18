/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Theorem

open Aeneas Aeneas.Std PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite

def Contiguous (source : Slice Cell) (start : Key) : Prop :=
  ∀ index (hindex : index < source.val.length),
    location source.val[index].key = location start index

def snapshotValues {Input : Type} (memory : Memory Input) (source : Key) : Input → Nat → Byte :=
  fun input index => memory input (location source index)

theorem location_lane_of_eq (left right : Key) (index lane : Nat)
    (h : location left = location right index) :
    location left lane = location right (index + lane) := by
  have hshift := congrArg
    (fun address : Address => { address with offset := address.offset + lane }) h
  change Address.mk left.space left.allocation (left.offset.val + lane) =
    Address.mk right.space right.allocation (right.offset.val + index + lane) at hshift
  simpa only [location, Nat.add_assoc] using hshift

theorem snapshotPayload_sound {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (source : Slice Cell) (start : Key)
    (hsource : Represents sem memory source) (hcontiguous : Contiguous source start) :
    PayloadSound sem (snapshotPayload source) (snapshotValues memory start) := by
  constructor
  · intro index hindex
    have hi : index < source.val.length := by simpa using hindex
    rw [snapshotPayload_at source index hi]
    have hcell := hsource source.val[index] (List.getElem_mem hi)
    simpa only [snapshotByte, snapshotValues, hcontiguous index hi] using hcell.byte
  · intro index hindex id hid input hinput lane
    have hi : index < source.val.length := by simp only [snapshotPayload_length] at hindex; omega
    rw [snapshotPayload_at source index hi] at hid
    have hcell := hsource source.val[index] (List.getElem_mem hi)
    have hword := hcell.word id hid input hinput lane
    have hloc := location_lane_of_eq source.val[index].key start index lane.val
      (hcontiguous index hi)
    simpa only [snapshotValues, hloc] using hword

theorem snapshot_cells_sound {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (source : Slice Cell) (start : Key)
    (hsource : Represents sem memory source) (hcontiguous : Contiguous source start) :
    ∃ payload,
      snapshot_cells source = .ok payload ∧
      payload.slice = snapshotPayload source ∧
      PayloadSound sem payload.slice (snapshotValues memory start) := by
  obtain ⟨payload, hrun, hmodel⟩ := spec_returns _ _ (snapshot_cells_spec source)
  have hslice : payload.slice = snapshotPayload source := by
    apply Slice.ext
    simpa [snapshotPayload, alloc.vec.Vec.val] using hmodel
  refine ⟨payload, hrun, hslice, ?_⟩
  rw [hslice]
  exact snapshotPayload_sound sem memory source start hsource hcontiguous

theorem apply_copy_effect_preserves_representation {Input : Type}
    (sem : Semantics Input) (memory : Memory Input) (state source : Slice Cell)
    (write start : Key) (hstate : Represents sem memory state)
    (hsource : Represents sem memory source) (hcontiguous : Contiguous source start)
    (hspan : ValidWriteSpan state write source.val.length) :
    ∃ next,
      apply_copy_effect state write source = .ok next ∧
      WriteSpec state next write (snapshotPayload source) ∧
      Represents sem (storeBytes memory write source.val.length (snapshotValues memory start)) next ∧
      ValidWriteSpan next write source.val.length ∧
      next.val.map Cell.key = state.val.map Cell.key := by
  obtain ⟨payload, hread, hslice, hpayload⟩ :=
    snapshot_cells_sound sem memory source start hsource hcontiguous
  rw [hslice] at hpayload
  obtain ⟨next, hwrite, hspec, hrep, hspan, hkeys⟩ :=
    apply_write_effect_preserves_representation sem memory state write (snapshotPayload source)
      (snapshotValues memory start) hstate (by simpa using hspan) hpayload
  refine ⟨next, ?_, hspec, ?_, ?_, hkeys⟩
  · have hderef : alloc.vec.Vec.deref payload = payload.slice := by
      apply Slice.ext
      simp [alloc.vec.Vec.deref, alloc.vec.Vec.val]
    simpa [apply_copy_effect, hread, hderef, hslice] using hwrite
  · simpa using hrep
  · simpa using hspan

/-- Completion uses the issue memory even when intervening writes changed the source. -/
theorem complete_snapshot_preserves_representation {Input : Type}
    (sem : Semantics Input) (issueMemory completionMemory : Memory Input)
    (source state : Slice Cell) (start write : Key) (payload : alloc.vec.Vec PayloadByte)
    (hsource : Represents sem issueMemory source) (hcontiguous : Contiguous source start)
    (hread : snapshot_cells source = .ok payload)
    (hstate : Represents sem completionMemory state)
    (hspan : ValidWriteSpan state write source.val.length) :
    ∃ next,
      apply_write_effect state write payload.slice = .ok next ∧
      WriteSpec state next write payload.slice ∧
      Represents sem (storeBytes completionMemory write source.val.length
        (snapshotValues issueMemory start)) next ∧
      ValidWriteSpan next write source.val.length ∧
      next.val.map Cell.key = state.val.map Cell.key := by
  obtain ⟨snapshot, hrun, hslice, hpayload⟩ :=
    snapshot_cells_sound sem issueMemory source start hsource hcontiguous
  have heq : snapshot = payload := by simpa [hread] using hrun.symm
  subst snapshot
  have hlength : payload.slice.val.length = source.val.length := by simp [hslice]
  simpa only [hlength] using
    apply_write_effect_preserves_representation sem completionMemory state write payload.slice
      (snapshotValues issueMemory start) hstate (by simpa only [hlength] using hspan) hpayload

end MemoryWrite

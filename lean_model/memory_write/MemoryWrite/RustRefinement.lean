/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RustWords
import MemoryWrite.Effects

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite

def rustFamily : checked.WriteClass → WriteFamily
  | .Store => .store
  | .LocalStore => .localStore
  | .TensorStore => .tmemStore
  | .SharedTransfer => .tmaLoad
  | .GlobalTransfer => .tmaStore

def rustEffect (write : checked.PreparedWrite) : WriteEffect :=
  { family := rustFamily write.class
    destination := write.destination
    payload := alloc.vec.Vec.deref write.payload
    words := write.words.val.map wordPatchView
    enabled := write.enabled
    addressDependent := write.address_dependent
    guardDependent := write.guard_dependent }

private theorem lawful_beq_decide {α : Type} [BEq α] [LawfulBEq α]
    [DecidableEq α] (left right : α) : (left == right) = decide (left = right) := by
  apply Bool.eq_iff_iff.mpr
  simp

@[step]
theorem rust_class_allows_spec (family : checked.WriteClass) (space : U8) :
    checked.class_allows family space
      ⦃ result => result = (rustFamily family).allows space ⦄ := by
  cases family <;> simp [checked.class_allows, rustFamily, WriteFamily.allows,
    lawful_beq_decide]
  split <;> simp_all

theorem key_location_injective : Function.Injective location := by
  intro left right h
  cases left
  cases right
  simp only [location, Address.mk.injEq, Nat.add_zero] at h
  obtain ⟨hs, ha, ho⟩ := h
  simp only [Key.mk.injEq]
  exact ⟨hs, ha, by scalar_tac⟩

theorem storage_span_iff (state : Storage) (start : Key) (width : Usize) :
    ValidWriteSpan state.toCells start width.val ↔ Storage.SpanFits state start width := by
  constructor
  · intro h
    refine ⟨h.no_wrap, ?_⟩
    intro key hkey
    obtain ⟨index, rfl⟩ := List.mem_ofFn.mp hkey
    obtain ⟨cell, hcell, hlocation⟩ := h.coverage index.val index.isLt
    simp only [Storage.toCells, Slice.from_val, List.mem_map] at hcell
    obtain ⟨other, hother, rfl⟩ := hcell
    have hoffset := Storage.offsetKey_location start index.val (by
      have := h.no_wrap
      omega)
    have heq : other = Storage.offsetKey start index.val := by
      apply key_location_injective
      rw [Storage.getCell_key, ←hoffset] at hlocation
      exact hlocation
    simpa [heq] using hother
  · exact Storage.spanFits_coverage state start width

def rustError : checked.EffectError → WriteError
  | .InputDependentAddress => .inputDependentAddress
  | .InputDependentGuard => .inputDependentGuard
  | .WrongSpace => .wrongSpace
  | .OutOfBounds => .outOfBounds

/-- Live cells refine the separate storage planes; errors retain every cell exactly. -/
def RustWriteRefines (before : Storage) (effect : WriteEffect)
    (result : core.result.Result Unit checked.EffectError × Slice Cell) : Prop :=
  match result.1 with
  | .Err error => applyEffect before effect = .error (rustError error) ∧
      result.2 = before.toCells
  | .Ok _ => ∃ after, applyEffect before effect = .ok after ∧
      result.2 = after.toCells

@[simp]
theorem rust_payload_len (payload : alloc.vec.Vec PayloadByte) :
    Slice.len (alloc.vec.Vec.deref payload) = alloc.vec.Vec.len payload := by
  apply UScalar.eq_of_val_eq
  simp [alloc.vec.Vec.deref, alloc.vec.Vec.val]

@[step]
theorem rust_checked_write_refines (state : Storage) (write : checked.PreparedWrite) :
    checked.checked_write state.toCells write
      ⦃ result => RustWriteRefines state (rustEffect write) result ⦄ := by
  unfold checked.checked_write
  split
  · rename_i haddress
    simp [RustWriteRefines, applyEffect, rustEffect, rustError, haddress]
  · rename_i haddress
    split
    · rename_i hguard
      simp [RustWriteRefines, applyEffect, rustEffect, rustError, haddress, hguard]
    · rename_i hguard
      split
      · rename_i henabled
        step with rust_class_allows_spec write.class write.destination.space as ⟨allowed, hallowed⟩
        split
        · rename_i hall
          have hfamily : (rustFamily write.class).allows write.destination.space = true := by
            simpa [hallowed] using hall
          step with rust_span_fits_spec state.toCells write.destination
            (alloc.vec.Vec.len write.payload) as ⟨fits, hfits⟩
          split
          · rename_i hfit
            have hspan : Storage.SpanFits state write.destination
                (Slice.len (alloc.vec.Vec.deref write.payload)) := by
              apply (storage_span_iff state _ _).mp
              simpa only [rust_payload_len] using hfits.mp hfit
            step with apply_write_effect_spec state.toCells write.destination
              (alloc.vec.Vec.deref write.payload) as ⟨after, hafter⟩
            have hcells : after =
                (state.write write.destination (alloc.vec.Vec.deref write.payload)).toCells := by
              apply Slice.ext
              rw [hafter, Storage.toCells_write]
            rw [hcells]
            step with rust_install_words_refines
              (state.write write.destination (alloc.vec.Vec.deref write.payload))
              (alloc.vec.Vec.deref write.words) as ⟨finalState, hfinalState⟩
            simp only [RustWriteRefines]
            refine ⟨(state.write write.destination (alloc.vec.Vec.deref write.payload)).installWords
              (write.words.val.map wordPatchView), ?_, ?_⟩
            · simp [applyEffect, rustEffect, hfamily, haddress, hguard, henabled]
              simpa [alloc.vec.Vec.deref, alloc.vec.Vec.val, Slice.len] using hspan
            · simpa [alloc.vec.Vec.deref] using hfinalState
          · rename_i hfit
            have hspan : ¬Storage.SpanFits state write.destination
                (Slice.len (alloc.vec.Vec.deref write.payload)) := by
              intro h
              have hvalid := (storage_span_iff state _ _).mpr h
              have : fits = true := hfits.mpr (by simpa only [rust_payload_len] using hvalid)
              exact hfit this
            simp [RustWriteRefines, applyEffect, rustEffect, rustError, hfamily,
              haddress, hguard, henabled]
            simpa [alloc.vec.Vec.deref, alloc.vec.Vec.val, Slice.len] using hspan
        · rename_i hall
          have hfamily : ¬(rustFamily write.class).allows write.destination.space = true := by
            simpa [hallowed] using hall
          simp [RustWriteRefines, applyEffect, rustEffect, rustError, hfamily,
            haddress, hguard, henabled]
      · rename_i henabled
        simp [RustWriteRefines, applyEffect, rustEffect, haddress, hguard, henabled]

/-- The Rust transition terminates, refines the existing model, and preserves its semantics. -/
theorem rust_checked_write_preserves_representation {Input : Type}
    (sem : Semantics Input) (memory : Memory Input) (state : Storage)
    (write : checked.PreparedWrite) (values : Input → Nat → Byte)
    (hstate : Storage.Represents sem memory state)
    (hsound : EffectSound sem memory (rustEffect write) values) :
    ∃ result, checked.checked_write state.toCells write = .ok result ∧
      RustWriteRefines state (rustEffect write) result ∧
      match result.1 with
      | .Err _ => result.2 = state.toCells
      | .Ok _ => Represents sem (effectMemory memory (rustEffect write) values) result.2 := by
  obtain ⟨result, hrun, hrefines⟩ := spec_returns _ _ (rust_checked_write_refines state write)
  refine ⟨result, hrun, hrefines, ?_⟩
  cases hstatus : result.1 with
  | Err error => exact (show applyEffect state (rustEffect write) = .error (rustError error) ∧
        result.2 = state.toCells from by simpa [RustWriteRefines, hstatus] using hrefines).2
  | Ok value =>
    obtain ⟨after, hafter, hcells⟩ := (show ∃ after,
        applyEffect state (rustEffect write) = .ok after ∧ result.2 = after.toCells from by
      simpa [RustWriteRefines, hstatus] using hrefines)
    rw [hcells]
    exact (applyEffect_preserves_representation sem memory state after (rustEffect write)
      values hstate hsound hafter).1

/-- Valid producer input yields an actual successful Rust transition, without a run premise. -/
theorem rust_checked_write_progress {Input : Type}
    (sem : Semantics Input) (memory : Memory Input) (state : Storage)
    (write : checked.PreparedWrite) (values : Input → Nat → Byte)
    (hstate : Storage.Represents sem memory state)
    (hsound : EffectSound sem memory (rustEffect write) values)
    (haddress : write.address_dependent = false)
    (hguard : write.guard_dependent = false) (henabled : write.enabled = true)
    (hfamily : (rustFamily write.class).allows write.destination.space = true)
    (hspan : Storage.SpanFits state write.destination (alloc.vec.Vec.len write.payload)) :
    ∃ next, checked.checked_write state.toCells write = .ok (.Ok (), next) ∧
      RustWriteRefines state (rustEffect write) (.Ok (), next) ∧
      Represents sem (storeBytes memory write.destination write.payload.val.length values) next ∧
      next.val.map Cell.key = state.toCells.val.map Cell.key := by
  let after := (state.write write.destination (alloc.vec.Vec.deref write.payload)).installWords
    (write.words.val.map wordPatchView)
  have hmodel : applyEffect state (rustEffect write) = .ok after := by
    simp [applyEffect, rustEffect, after,
      haddress, hguard, henabled, hfamily, hspan]
  obtain ⟨⟨status, cells⟩, hrun, hrefines⟩ :=
    spec_returns _ _ (rust_checked_write_refines state write)
  cases status with
  | Err error =>
    have := hrefines.1
    rw [hmodel] at this
    contradiction
  | Ok value =>
    cases value
    obtain ⟨model, hresult, hcells⟩ := hrefines
    have heq : model = after := by simpa [hmodel] using hresult.symm
    subst model
    dsimp only [Prod.snd] at hcells
    subst cells
    have hpreserved := applyEffect_preserves_representation sem memory state after
      (rustEffect write) values hstate hsound hmodel
    refine ⟨after.toCells, hrun, ⟨after, hmodel, rfl⟩, ?_, ?_⟩
    · simpa [effectMemory, rustEffect, henabled, alloc.vec.Vec.deref,
        Storage.Represents] using hpreserved.1
    · simp only [Storage.toCells, Slice.from_val, List.map_map, Function.comp_def,
        Storage.getCell_key, hpreserved.2]

end MemoryWrite

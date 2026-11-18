/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.StoreWord
import MemoryWrite.SymbolicStateRefinement

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 6000000

namespace MemoryWrite.StoreWord

open Symbolic

theorem little_endian_byte (word : U32) (index : Nat) (hi : index < 4) :
    (core.num.U32.to_le_bytes word).val[index]'(by simpa [core.num.U32.to_le_bytes] using hi) =
      byteAt word ⟨index, hi⟩ := by
  simp only [core.num.U32.to_le_bytes, Array.from_val, List.getElem_map]
  congr 1
  apply BitVec.eq_of_getElem_eq
  intro bit hb
  have hb8 : bit < 8 := by simpa using hb
  rw [BitVec.getElem_toLEBytes_eq_getElem _ _ _
    (by simp [BitVec.toLEBytes_length]; omega) hb8 (by omega)]
  simp [←BitVec.getLsbD_eq_getElem, hb8]

theorem payload_list (source : Metadata) : (payload source).val =
    [⟨byteAt source.captured ⟨0, by omega⟩, none, source.tainted, source.defined,
        source.history.map WordHistory.Known⟩,
      ⟨byteAt source.captured ⟨1, by omega⟩, none, source.tainted, source.defined, none⟩,
      ⟨byteAt source.captured ⟨2, by omega⟩, none, source.tainted, source.defined, none⟩,
      ⟨byteAt source.captured ⟨3, by omega⟩, none, source.tainted, source.defined, none⟩] := by
  rfl

theorem address_bits (low high : U32) :
    (UScalar.cast .U64 low).bv ||| ((UScalar.cast .U64 high).bv <<< 32) =
      high.bv ++ low.bv := by
  simp only [UScalar.cast_bv_eq]
  rw [BitVec.or_comm, ←BitVec.setWidth_append_eq_shiftLeft_setWidth_or]
  rfl

@[step]
theorem rust_payload_byte_spec (value : U8) (source : SourceMetadata)
    (word : Option WordHistory) :
    store.payload_byte value source word
      ⦃ result => result = ⟨value, none, source.tainted, source.defined, word⟩ ⦄ := by
  simp [store.payload_byte]

def inputEffect (input : store.WordStoreInput) : WriteEffect :=
  effect input.space input.allocation (metadataView input.source) (metadataView input.guard)
    (metadataView input.address_low) (metadataView input.address_high)

@[step]
theorem rust_prepare_word_spec (input : store.WordStoreInput) :
    store.prepare_word input ⦃ result => rustEffect result = inputEffect input ⦄ := by
  unfold store.prepare_word
  step as ⟨decision, henabled, haddress, hguard⟩
  simp only [lift, bind_ok]
  cases hhistory : input.source.history <;> simp only [bind_ok]
  all_goals
    step as ⟨b0, hb0⟩
    step as ⟨p0, hp0⟩
    step with alloc.vec.Vec.push_spec (alloc.vec.Vec.new PayloadByte) p0 (by simp [alloc.vec.Vec.new, alloc.vec.Vec.val, alloc.vec.Vec.from]; scalar_tac)
      as ⟨v0, hv0⟩
    step as ⟨b1, hb1⟩
    step as ⟨p1, hp1⟩
    step with alloc.vec.Vec.push_spec v0 p1 (by simp [hv0, alloc.vec.Vec.new]; scalar_tac)
      as ⟨v1, hv1⟩
    step as ⟨b2, hb2⟩
    step as ⟨p2, hp2⟩
    step with alloc.vec.Vec.push_spec v1 p2 (by simp [hv1, hv0, alloc.vec.Vec.new]; scalar_tac)
      as ⟨v2, hv2⟩
    step as ⟨b3, hb3⟩
    step as ⟨p3, hp3⟩
    step with alloc.vec.Vec.push_spec v2 p3
      (by simp [hv2, hv1, hv0, alloc.vec.Vec.new]; scalar_tac) as ⟨v3, hv3⟩
    let shifted : U64 := ⟨(UScalar.cast .U64 input.address_high.captured).bv <<< 32⟩
    have hshift : ((UScalar.cast .U64 input.address_high.captured) <<< 32#i32) = .ok shifted := by rfl
    rw [hshift]
    simp only [bind_ok]
    have hoffset : UScalar.cast .U64 input.address_low.captured ||| shifted =
        (destination input.space input.allocation (metadataView input.address_low)
          (metadataView input.address_high)).offset := by
      apply congrArg UScalar.mk
      simpa only [UScalar.bv_or, shifted, destination, metadataView] using
        address_bits input.address_low.captured input.address_high.captured
    have hpayload : alloc.vec.Vec.deref v3 = payload (metadataView input.source) := by
      apply Slice.ext
      have hview : (alloc.vec.Vec.deref v3).val = v3.val := by
        simp [alloc.vec.Vec.deref, alloc.vec.Vec.val]
      rw [hview]
      rw [hv3, hv2, hv1, hv0]
      simp only [alloc.vec.Vec.new, alloc.vec.Vec.from, alloc.vec.Vec.val, Slice.from_val,
        List.nil_append, List.cons_append]
      change [p0, p1, p2, p3] = _
      rw [payload_list]
      simp [hp0, hp1, hp2, hp3, hb0, hb1, hb2, hb3, little_endian_byte, metadataView, hhistory]

    simp [rustEffect, inputEffect, effect, henabled, haddress, hguard, hpayload,
      hoffset, alloc.vec.Vec.new, rustFamily, destination, metadataView]

def dynamicMemory {Input : Type} (memory : Memory Input) (input : store.WordStoreInput)
    (sourceValues guardValues lowValues highValues : Input → Value) : Memory Input :=
  Producers.dynamicStore memory (predicateGuard guardValues)
    (addressFromWords input.space input.allocation lowValues highValues) 4 (values sourceValues)

theorem rust_prepare_word_derives_payload {Input : Type} (sem : Semantics Input)
    (meaning : Meaning Input) (input : store.WordStoreInput) (sourceValues : Input → Value)
    (hs : Sound sem meaning (metadataView input.source) sourceValues)
    (hm : ∀ id data lane, sem.word id data lane = byteAt (meaning id data) lane) :
    store.prepare_word input ⦃ write =>
      PayloadSound sem (rustEffect write).payload (values sourceValues) ∧
      write.payload.val.length = 4 ⦄ := by
  step as ⟨write, hw⟩
  constructor
  · rw [hw]
    exact payload_sound sem meaning _ sourceValues hs hm
  · have h := congrArg (fun effect => effect.payload.val.length) hw
    simpa [rustEffect, inputEffect, effect, alloc.vec.Vec.deref] using h

theorem rust_disabled_word_unchanged (state : engine.Engine) (input : store.WordStoreInput)
    (hguard : input.guard.tainted = false)
    (hlow : input.address_low.tainted = false) (hhigh : input.address_high.tainted = false)
    (hdisabled : input.guard.captured = 0#u32) :
    store.commit_word state input = .ok (.Ok (), state) := by
  obtain ⟨write, hprepare, hw⟩ := spec_returns _ _ (rust_prepare_word_spec input)
  have ha : write.address_dependent = false := by
    have h := congrArg WriteEffect.addressDependent hw
    simpa [rustEffect, inputEffect, effect, metadataView, hlow, hhigh] using h
  have hg : write.guard_dependent = false := by
    have h := congrArg WriteEffect.guardDependent hw
    simpa [rustEffect, inputEffect, effect, metadataView, hguard] using h
  have he : write.enabled = false := by
    have h := congrArg WriteEffect.enabled hw
    simpa [rustEffect, inputEffect, effect, metadataView, hdisabled] using h
  simp [store.commit_word, hprepare, engine.apply, checked.checked_write,
    alloc.vec.Vec.deref_mut, ha, hg, he, lift, bind_ok]

def CommitOutcome {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (before : Storage) (state : engine.Engine) (input : store.WordStoreInput)
    (sourceValues guardValues lowValues highValues : Input → Value)
    (result : core.result.Result Unit checked.EffectError × engine.Engine) : Prop :=
  match result.1 with
  | .Err _ => result.2 = state
  | .Ok _ => ∃ after : Storage,
      result.2.cells.slice = after.toCells ∧
      Storage.Represents sem (dynamicMemory memory input sourceValues guardValues lowValues highValues) after ∧
      Producers.StableStoreExecution sem (predicateGuard guardValues)
        (addressFromWords input.space input.allocation lowValues highValues) ∧
      after.keys = before.keys ∧
      ∀ key ∈ before.keys.val,
        ¬Overlaps (destination input.space input.allocation (metadataView input.address_low)
          (metadataView input.address_high)) 4 key → after.getCell key = before.getCell key

theorem rust_commit_word_preserves_dynamic {Input : Type} (sem : Semantics Input)
    (meaning : Meaning Input) (memory : Memory Input) (before : Storage) (state : engine.Engine)
    (input : store.WordStoreInput) (sourceValues guardValues lowValues highValues : Input → Value)
    (hs : Sound sem meaning (metadataView input.source) sourceValues)
    (hg : Sound sem meaning (metadataView input.guard) guardValues)
    (hl : Sound sem meaning (metadataView input.address_low) lowValues)
    (hh : Sound sem meaning (metadataView input.address_high) highValues)
    (hm : ∀ id data lane, sem.word id data lane = byteAt (meaning id data) lane)
    (hstate : Storage.Represents sem memory before) (hcells : state.cells.slice = before.toCells)
    (hnumeric : state.numeric_written.val.length + 4 ≤ Usize.max)
    (hhalf : state.half_written.val.length + 4 ≤ Usize.max) :
    ∃ result, store.commit_word state input = .ok result ∧
      (∃ write, store.prepare_word input = .ok write ∧
        RustEngineRefines before state write result) ∧
      CommitOutcome sem memory before state input sourceValues guardValues lowValues highValues result := by
  obtain ⟨write, hprepare, heffect⟩ := spec_returns _ _ (rust_prepare_word_spec input)
  have hlen : write.payload.val.length = 4 := by
    have h := congrArg (fun value => value.payload.val.length) heffect
    simpa [rustEffect, inputEffect, effect, alloc.vec.Vec.deref] using h
  obtain ⟨result, hrun, hrefines⟩ := spec_returns _ _
    (rust_engine_apply_refines before state write hcells (by simpa [hlen] using hnumeric)
      (by simpa [hlen] using hhalf))
  refine ⟨result, ?_, ⟨write, hprepare, hrefines⟩, ?_⟩
  · simp [store.commit_word, hprepare, hrun]
  · cases hstatus : result.1 with
    | Err error =>
      simpa [CommitOutcome, hstatus] using
        rust_engine_error_unchanged before state write result hcells hrefines error hstatus
    | Ok value =>
      obtain ⟨after, hmodel, hafter⟩ := (show ∃ after,
        applyEffect before (rustEffect write) = .ok after ∧ result.2.cells.slice = after.toCells from by
          simpa [RustWriteRefines, hstatus] using hrefines.1)
      rw [heffect] at hmodel
      have hpreserved := accepted_preserves_dynamic sem meaning memory before after input.space
        input.allocation (metadataView input.source) (metadataView input.guard)
        (metadataView input.address_low) (metadataView input.address_high)
        sourceValues guardValues lowValues highValues hs hg hl hh hm hstate hmodel
      simp only [CommitOutcome, hstatus]
      refine ⟨after, hafter, hpreserved.1, hpreserved.2.1, hpreserved.2.2, ?_⟩
      intro key hk hframe
      exact model_frame before after _ _ _ _ _ _ hmodel key hk hframe

def opaqueCell (cell : Cell) (id : U32) : Cell :=
  if cell.word = none then { cell with word := some (.Opaque id) } else cell

@[step]
theorem rust_materialize_opaque_spec (cell : Cell) (id : U32) :
    store.materialize_opaque cell id ⦃ result => result = opaqueCell cell id ⦄ := by
  cases hw : cell.word <;> simp [store.materialize_opaque, core.option.Option.is_none, hw, opaqueCell]

theorem opaque_cell_represents {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (cell : Cell) (id : U32)
    (h : CellRepresents sem memory cell) :
    CellRepresents sem memory (opaqueCell cell id) := by
  unfold opaqueCell
  split
  · constructor
    · exact h.byte
    · intro known hk
      simp at hk
    · exact h.initial_fallback
  · exact h

theorem rust_opaque_materialization_preserves_representation {Input : Type}
    (sem : Semantics Input) (memory : Memory Input) (cell : Cell) (id : U32)
    (h : CellRepresents sem memory cell) :
    store.materialize_opaque cell id ⦃ result => CellRepresents sem memory result ⦄ := by
  step as ⟨result, hr⟩
  rw [hr]
  exact opaque_cell_represents sem memory cell id h

theorem opaque_cell_does_not_invent_known_history (cell : Cell) (id known : U32)
    (h : (opaqueCell cell id).word = some (.Known known)) :
    cell.word = some (.Known known) := by
  unfold opaqueCell at h
  split at h
  · simp at h
  · exact h

end MemoryWrite.StoreWord

/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Consumers
import MemoryWrite.Theorem

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite

structure WriteConsumerSafety (before after : Cell) (write : Key)
    (payload : Slice PayloadByte) : Prop where
  key : after.key = before.key
  unknown_origin : Touches write payload.val.length before.key →
    (payloadAt payload (before.key.offset.val - write.offset.val)).origin = none →
    after.written = true ∧
      (∀ initial, Consumers.originWithFallback after initial = none) ∧
      (∀ other, Consumers.sameKnownOrigin after other = false)
  partial_word : Overlaps write payload.val.length before.key →
    ¬CoversWord write payload.val.length before.key →
    after.word = none ∧ ∀ other, Consumers.sameKnownWord after other = false
  tainted_capture : Touches write payload.val.length before.key →
    (payloadAt payload (before.key.offset.val - write.offset.val)).tainted = true →
    after.tainted = true ∧ ∀ other, Consumers.sameUntaintedCapture after other = false

theorem modelCell_consumer_safety (before : Cell) (write : Key)
    (payload : Slice PayloadByte) :
    WriteConsumerSafety before (modelCell before write payload) write payload := by
  refine ⟨modelCell_key before write payload, ?_, ?_, ?_⟩
  · intro ht ho
    have hw : (modelCell before write payload).written = true := by simp [modelCell, ht]
    have horigin : (modelCell before write payload).origin = none := by
      simpa only [modelCell, ite_eq_left ht] using ho
    refine ⟨hw, ?_, ?_⟩
    · intro initial
      exact Consumers.originWithFallback_does_not_resurrect _ initial horigin hw
    · intro other
      simp [Consumers.sameKnownOrigin, horigin]
  · intro hover hpartial
    have hword : (modelCell before write payload).word = none := by
      have heffect := (modelCell_spec before write payload).word_effect
      simpa only [wordAfter, ite_eq_right hpartial, ite_eq_left hover] using heffect
    refine ⟨hword, ?_⟩
    intro other
    simp [Consumers.sameKnownWord, hword]
  · intro ht htaint
    have htainted : (modelCell before write payload).tainted = true := by
      simpa only [modelCell, ite_eq_left ht] using htaint
    refine ⟨htainted, ?_⟩
    intro other
    simp [Consumers.sameUntaintedCapture, htainted]

@[step]
theorem rust_cell_after_write_consumer_safety (before : Cell) (write : Key)
    (payload : Slice PayloadByte) :
    cell_after_write before write payload
      ⦃ after => WriteConsumerSafety before after write payload ⦄ := by
  apply WP.spec_mono (cell_after_write_spec before write payload)
  intro after hafter
  rw [hafter]
  exact modelCell_consumer_safety before write payload

theorem rust_apply_write_consumer_safety (state : Slice Cell) (write : Key)
    (payload : Slice PayloadByte) :
    ∃ next, apply_write_effect state write payload = .ok next ∧
      List.Forall₂ (fun before after => WriteConsumerSafety before after write payload)
        state.val next.val := by
  obtain ⟨next, hrun, hnext⟩ := spec_returns _ _ (apply_write_effect_spec state write payload)
  refine ⟨next, hrun, ?_⟩
  rw [hnext]
  induction state.val with
  | nil => exact .nil
  | cons before rest ih =>
    exact .cons (modelCell_consumer_safety before write payload) ih

theorem rust_unknown_write_blocks_initial_origin (before : Cell) (write : Key)
    (payload : Slice PayloadByte)
    (ht : Touches write payload.val.length before.key)
    (ho : (payloadAt payload (before.key.offset.val - write.offset.val)).origin = none)
    (initial : Key → Option Origin) :
    ∃ after, cell_after_write before write payload = .ok after ∧
      after.written = true ∧ Consumers.originWithFallback after initial = none := by
  obtain ⟨after, hrun, hsafe⟩ := spec_returns _ _
    (rust_cell_after_write_consumer_safety before write payload)
  exact ⟨after, hrun, (hsafe.unknown_origin ht ho).1,
    (hsafe.unknown_origin ht ho).2.1 initial⟩

theorem rust_written_lane_blocks_initial_word (cells : Fin 4 → Cell) (lane : Fin 4)
    (write : Key) (payload : Slice PayloadByte) (initial : Option WordHistory)
    (ht : Touches write payload.val.length (cells lane).key) :
    ∃ after, cell_after_write (cells lane) write payload = .ok after ∧
      Consumers.initialWordFallback (Function.update cells lane after) initial = none := by
  obtain ⟨after, hrun, hafter⟩ := spec_returns _ _
    (cell_after_write_spec (cells lane) write payload)
  have hwritten : after.written = true := by simp [hafter, modelCell, ht]
  refine ⟨after, hrun, ?_⟩
  exact Consumers.initialWordFallback_rejects_partial_write
    (Function.update cells lane after) initial lane (by simpa using hwritten)

theorem rust_partial_write_rejects_known_word (before : Cell) (write : Key)
    (payload : Slice PayloadByte)
    (hover : Overlaps write payload.val.length before.key)
    (hpartial : ¬CoversWord write payload.val.length before.key) :
    ∃ after, cell_after_write before write payload = .ok after ∧
      after.word = none ∧ ∀ other, Consumers.sameKnownWord after other = false := by
  obtain ⟨after, hrun, hsafe⟩ := spec_returns _ _
    (rust_cell_after_write_consumer_safety before write payload)
  exact ⟨after, hrun, hsafe.partial_word hover hpartial⟩

theorem short_write_cannot_cover_word (write key : Key) (width : Nat)
    (hshort : width < 4) :
    ¬CoversWord write width key := by
  intro ⟨_, hlo, hhi⟩
  omega

theorem rust_short_write_rejects_intersecting_word (before : Cell) (write : Key)
    (payload : Slice PayloadByte)
    (hover : Overlaps write payload.val.length before.key) (hshort : payload.val.length < 4) :
    ∃ after, cell_after_write before write payload = .ok after ∧
      after.word = none ∧ ∀ other, Consumers.sameKnownWord after other = false := by
  exact rust_partial_write_rejects_known_word before write payload hover
    (short_write_cannot_cover_word write before.key payload.val.length hshort)

theorem rust_unaligned_word_write_rejects_both_words (left right : Cell) (write : Key)
    (payload : Slice PayloadByte) (hwidth : payload.val.length = 4)
    (hleftRegion : SameRegion write left.key) (hrightRegion : SameRegion write right.key)
    (hstart : left.key.offset.val < write.offset.val)
    (hinside : write.offset.val < left.key.offset.val + 4)
    (hadjacent : right.key.offset.val = left.key.offset.val + 4) :
    ∃ leftAfter rightAfter,
      cell_after_write left write payload = .ok leftAfter ∧
      cell_after_write right write payload = .ok rightAfter ∧
      leftAfter.word = none ∧ rightAfter.word = none ∧
      (∀ other, Consumers.sameKnownWord leftAfter other = false) ∧
      (∀ other, Consumers.sameKnownWord rightAfter other = false) := by
  have hleftOverlap : Overlaps write payload.val.length left.key := by
    refine ⟨hleftRegion, ?_, ?_, hinside⟩ <;> omega
  have hrightOverlap : Overlaps write payload.val.length right.key := by
    refine ⟨hrightRegion, ?_, ?_, ?_⟩ <;> omega
  have hleftPartial : ¬CoversWord write payload.val.length left.key := by
    intro ⟨_, hlo, _⟩
    omega
  have hrightPartial : ¬CoversWord write payload.val.length right.key := by
    intro ⟨_, _, hhi⟩
    omega
  obtain ⟨leftAfter, hleftRun, hleftWord, hleftCheck⟩ :=
    rust_partial_write_rejects_known_word left write payload hleftOverlap hleftPartial
  obtain ⟨rightAfter, hrightRun, hrightWord, hrightCheck⟩ :=
    rust_partial_write_rejects_known_word right write payload hrightOverlap hrightPartial
  exact ⟨leftAfter, rightAfter, hleftRun, hrightRun, hleftWord, hrightWord,
    hleftCheck, hrightCheck⟩

theorem varying_datum_is_tainted {Input : Type} (sem : Semantics Input)
    (values : Input → Byte) (value : U8) (origin : Option Origin) (tainted defined : Bool)
    (hsound : DatumSound sem values value origin tainted defined)
    (hvaries : ∃ input, sem.domain input ∧ (values input).bits ≠ (values sem.capture).bits) :
    tainted = true := by
  cases htaint : tainted with
  | true => rfl
  | false =>
    obtain ⟨input, hinput, hne⟩ := hvaries
    exact False.elim (hne (hsound.constant htaint input hinput))

theorem rust_symbolic_zero_cannot_equal_constant (before : Cell) (write : Key)
    (payload : Slice PayloadByte) {Input : Type} (sem : Semantics Input)
    (values : Input → Nat → Byte) (hsound : PayloadSound sem payload values)
    (ht : Touches write payload.val.length before.key)
    (hcapture : (values sem.capture (before.key.offset.val - write.offset.val)).bits = 0#u8)
    (hvaries : ∃ input, sem.domain input ∧
      (values input (before.key.offset.val - write.offset.val)).bits ≠ 0#u8) :
    ∃ after, cell_after_write before write payload = .ok after ∧
      after.value = 0#u8 ∧ after.tainted = true ∧
      ∀ constant, Consumers.sameUntaintedCapture after constant = false := by
  have hindex : before.key.offset.val - write.offset.val < payload.val.length := by
    obtain ⟨_, hlo, hhi⟩ := ht
    omega
  have hbyte := hsound.bytes (before.key.offset.val - write.offset.val) hindex
  have hzero : (payloadAt payload (before.key.offset.val - write.offset.val)).value = 0#u8 :=
    hbyte.captured.trans hcapture
  have htaint : (payloadAt payload (before.key.offset.val - write.offset.val)).tainted = true := by
    apply varying_datum_is_tainted sem _ _ _ _ _ hbyte
    simpa only [hcapture] using hvaries
  obtain ⟨after, hrun, hafter⟩ := spec_returns _ _ (cell_after_write_spec before write payload)
  have hsafe := modelCell_consumer_safety before write payload
  refine ⟨after, hrun, ?_, ?_⟩
  · simpa only [hafter, modelCell, ite_eq_left ht] using hzero
  · simpa only [hafter] using hsafe.tainted_capture ht htaint

private def zeroCaptureSemantics : Semantics Bool where
  domain := fun _ => True
  capture := false
  capture_valid := True.intro
  origin := fun _ _ => 0#u8
  zero_origin := by intros; rfl
  word := fun _ _ _ => 0#u8
  initial := fun _ _ => ⟨0#u8, true⟩

private def varyingZero (input : Bool) (_ : Nat) : Byte :=
  ⟨if input then 1#u8 else 0#u8, true⟩

private def capturedZeroPayload : Slice PayloadByte :=
  Slice.from [⟨0#u8, none, true, true, none⟩] (by have := Usize.bounds_eq; scalar_tac)

private theorem capturedZeroPayload_sound :
    PayloadSound zeroCaptureSemantics capturedZeroPayload varyingZero := by
  constructor
  · intro index hindex
    have hzero : index = 0 := by
      simp only [capturedZeroPayload, Slice.from_val, List.length_singleton] at hindex
      omega
    subst index
    simp only [payloadAt, capturedZeroPayload, Slice.from_val, List.getElem?_cons_zero,
      Option.getD_some]
    constructor
    · rfl
    · intros; rfl
    · intro id hid
      cases hid
    · intro hfalse
      cases hfalse
  · intro index hindex
    simp only [capturedZeroPayload, Slice.from_val, List.length_singleton] at hindex
    omega

theorem rust_captured_zero_is_not_constant (before : Cell) :
    ∃ after, cell_after_write before before.key capturedZeroPayload = .ok after ∧
      after.value = 0#u8 ∧ after.tainted = true ∧
      (∀ constant, Consumers.sameUntaintedCapture after constant = false) ∧
      (varyingZero false 0).bits = 0#u8 ∧ (varyingZero true 0).bits ≠ 0#u8 := by
  have ht : Touches before.key capturedZeroPayload.val.length before.key := by
    refine ⟨⟨rfl, rfl⟩, Nat.le_refl _, ?_⟩
    simp [capturedZeroPayload]
  obtain ⟨after, hrun, hvalue, htaint, hcheck⟩ :=
    rust_symbolic_zero_cannot_equal_constant before before.key capturedZeroPayload
      zeroCaptureSemantics varyingZero capturedZeroPayload_sound ht rfl
      ⟨true, True.intro, by change (1#u8) ≠ 0#u8; decide⟩
  exact ⟨after, hrun, hvalue, htaint, hcheck, rfl, by decide⟩

theorem rust_disabled_engine_write_full_state (state : engine.Engine)
    (write : checked.PreparedWrite) (hdisabled : write.enabled = false) :
    ∃ status, engine.apply state write = .ok (status, state) := by
  unfold engine.apply checked.checked_write
  cases write.address_dependent <;> cases write.guard_dependent <;>
    simp [alloc.vec.Vec.deref_mut, lift, hdisabled]

end MemoryWrite
